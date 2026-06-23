#include "sparse-mla-attn.cuh"

// DSA sparse MLA attention, flash-style. The latent KV is shared across all query heads (MLA
// has one KV head), so a block processes one token and a TILE of SMLA_HTILE heads (one warp
// each): each gathered latent row is streamed into shared memory ONCE and reused by every head
// in the tile, instead of being re-read per head. Online softmax keeps the per-head running
// max/sum and a value accumulator in registers - nothing per-token-sized hits global memory.
//   out[d, h, t] = sum_i softmax_i(scale * q[:,h,t] . k[:,key_i] + mask[key_i,t]) * k[d, key_i]
// with key_i = top_k[i, t] and value = the first n_val dims of the latent row.
#define SMLA_VPL 16  // value regs per lane (n_val <= 32*SMLA_VPL = 512)

static __global__ void sparse_mla_attn_f32(
        const float * __restrict__ k,
        const float * __restrict__ q,
        const int   * __restrict__ top_k,
        const char  * __restrict__ mask,
        float       * __restrict__ dst,
        const int d_lat, const int n_val, const int n_head, const int n_tok,
        const int n_tk, const float scale,
        const int k_row, const int64_t mask_nb1, const int topk_row, const int mask_f16) {
    const int t    = blockIdx.x;
    const int h0   = blockIdx.y * SMLA_HTILE;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int h    = h0 + warp;

    extern __shared__ float smem[];
    float * q_sh   = smem;                       // SMLA_HTILE * d_lat
    float * lat_sh = q_sh + SMLA_HTILE * d_lat;  // d_lat

    const int  * idx_t  = top_k + (size_t) t*topk_row;
    const char * mask_t = mask  + (size_t) t*mask_nb1;

    for (int idx = threadIdx.x; idx < SMLA_HTILE*d_lat; idx += blockDim.x) {
        const int hh = h0 + idx / d_lat;
        q_sh[idx] = hh < n_head ? q[((size_t) t*n_head + hh)*d_lat + idx % d_lat] : 0.0f;
    }

    float acc[SMLA_VPL];
    #pragma unroll
    for (int j = 0; j < SMLA_VPL; j++) {
        acc[j] = 0.0f;
    }
    float m = -INFINITY;
    float l = 0.0f;

    __syncthreads();

    for (int i = 0; i < n_tk; i++) {
        const int key = idx_t[i];
        for (int d = threadIdx.x; d < d_lat; d += blockDim.x) {
            lat_sh[d] = k[(size_t) key*k_row + d];
        }
        __syncthreads();

        float partial = 0.0f;
        for (int d = lane; d < d_lat; d += 32) {
            partial += q_sh[warp*d_lat + d] * lat_sh[d];
        }
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            partial += __shfl_xor_sync(0xffffffff, partial, o);
        }

        const float mval = mask_f16 ? __half2float(((const __half *) mask_t)[key])
                                    : ((const float  *) mask_t)[key];
        const float score = scale*partial + mval;

        const float m_new = fmaxf(m, score);
        const float corr  = m == -INFINITY ? 0.0f : expf(m - m_new);
        const float p     = score == -INFINITY ? 0.0f : expf(score - m_new);
        l = l*corr + p;
        #pragma unroll
        for (int j = 0; j < SMLA_VPL; j++) {
            const int d = lane + j*32;
            acc[j] = acc[j]*corr + (d < n_val ? p*lat_sh[d] : 0.0f);
        }
        m = m_new;
        __syncthreads();
    }

    if (h < n_head) {
        const float inv = l > 0.0f ? 1.0f/l : 0.0f;
        float * d_ht = dst + ((size_t) t*n_head + h)*n_val;
        #pragma unroll
        for (int j = 0; j < SMLA_VPL; j++) {
            const int d = lane + j*32;
            if (d < n_val) {
                d_ht[d] = acc[j]*inv;
            }
        }
    }
}

void ggml_cuda_op_sparse_mla_attn(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * k     = dst->src[0];
    const ggml_tensor * q     = dst->src[1];
    const ggml_tensor * top_k = dst->src[2];
    const ggml_tensor * mask  = dst->src[3];

    GGML_ASSERT(k->type == GGML_TYPE_F32);
    GGML_ASSERT(q->type == GGML_TYPE_F32);
    GGML_ASSERT(top_k->type == GGML_TYPE_I32);
    GGML_ASSERT(mask->type == GGML_TYPE_F32 || mask->type == GGML_TYPE_F16);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(q));
    GGML_ASSERT(ggml_is_contiguous(dst));

    float scale;
    int32_t n_val;
    memcpy(&scale, (const int32_t *) dst->op_params + 0, sizeof(float));
    memcpy(&n_val, (const int32_t *) dst->op_params + 1, sizeof(int32_t));
    GGML_ASSERT(n_val <= 32*SMLA_VPL);

    const int d_lat  = q->ne[0];
    const int n_head = q->ne[1];
    const int n_tok  = q->ne[2];
    const int n_tk   = top_k->ne[0];

    const size_t smem = (size_t)(SMLA_HTILE + 1) * d_lat * sizeof(float);

    dim3 grid(n_tok, (n_head + SMLA_HTILE - 1) / SMLA_HTILE, 1);
    sparse_mla_attn_f32<<<grid, SMLA_HTILE*32, smem, ctx.stream()>>>(
        (const float *) k->data, (const float *) q->data, (const int *) top_k->data,
        (const char *) mask->data, (float *) dst->data,
        d_lat, n_val, n_head, n_tok, n_tk, scale,
        (int)(k->nb[1]/sizeof(float)), (int64_t) mask->nb[1], (int)(top_k->nb[1]/sizeof(int)),
        mask->type == GGML_TYPE_F16);
}
