#include "sparse-mla-attn.cuh"

// DSA sparse MLA attention: each (token, head) attends only to its n_tk top-k latent KV rows,
// flash-style. One block per (token, head). The gathered K/V and the score vector are never
// written to global memory - the n_tk scores live in shared memory and the latent rows are
// read from the cache through the top_k indices.
//   out[d, h, t] = sum_i softmax_i(scale * q[:,h,t] . k[:,key_i] + mask[key_i,t]) * k[d, key_i]
// with key_i = top_k[i, t] and the value being the first n_val dims of the latent row.
static __global__ void sparse_mla_attn_f32(
        const float * __restrict__ k,
        const float * __restrict__ q,
        const int   * __restrict__ top_k,
        const float * __restrict__ mask,
        float       * __restrict__ dst,
        const int d_lat, const int n_val, const int n_head, const int n_tok,
        const int n_tk, const float scale,
        const int k_row, const int mask_row, const int topk_row) {
    const int t   = blockIdx.x;
    const int h   = blockIdx.y;
    const int tid = threadIdx.x;
    const int nth = blockDim.x;

    extern __shared__ float smem[];
    float * q_sh   = smem;             // d_lat
    float * sc_sh  = q_sh  + d_lat;    // n_tk
    int   * idx_sh = (int*)(sc_sh + n_tk);  // n_tk
    float * red    = (float*)(idx_sh + n_tk); // nth

    const float * q_ht  = q + ((size_t) t*n_head + h) * d_lat;
    const int   * idx_t = top_k + (size_t) t*topk_row;
    const float * mask_t = mask + (size_t) t*mask_row;
          float * d_ht  = dst + ((size_t) t*n_head + h) * n_val;

    for (int d = tid; d < d_lat; d += nth) {
        q_sh[d] = q_ht[d];
    }
    __syncthreads();

    // pass 1: scores into shared memory (one thread per key)
    for (int i = tid; i < n_tk; i += nth) {
        const int key = idx_t[i];
        const float * k_key = k + (size_t) key*k_row;
        float dot = 0.0f;
        for (int d = 0; d < d_lat; d++) {
            dot += q_sh[d] * k_key[d];
        }
        idx_sh[i] = key;
        sc_sh[i]  = scale*dot + mask_t[key];
    }
    __syncthreads();

    // softmax over the n_tk scores (block reductions)
    float lm = -INFINITY;
    for (int i = tid; i < n_tk; i += nth) {
        lm = fmaxf(lm, sc_sh[i]);
    }
    red[tid] = lm;
    __syncthreads();
    for (int s = nth/2; s > 0; s >>= 1) {
        if (tid < s) { red[tid] = fmaxf(red[tid], red[tid+s]); }
        __syncthreads();
    }
    const float mx = red[0];
    __syncthreads();

    float ls = 0.0f;
    for (int i = tid; i < n_tk; i += nth) {
        const float e = mx == -INFINITY ? 0.0f : expf(sc_sh[i] - mx);
        sc_sh[i] = e;
        ls += e;
    }
    red[tid] = ls;
    __syncthreads();
    for (int s = nth/2; s > 0; s >>= 1) {
        if (tid < s) { red[tid] += red[tid+s]; }
        __syncthreads();
    }
    const float inv = red[0] > 0.0f ? 1.0f/red[0] : 0.0f;
    __syncthreads();

    // pass 2: weighted sum of the value part (first n_val dims), coalesced over d
    for (int d0 = 0; d0 < n_val; d0 += nth) {
        const int d = d0 + tid;
        float acc = 0.0f;
        for (int i = 0; i < n_tk; i++) {
            if (d < n_val) {
                acc += sc_sh[i] * k[(size_t) idx_sh[i]*k_row + d];
            }
        }
        if (d < n_val) {
            d_ht[d] = acc * inv;
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
    GGML_ASSERT(mask->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(q));
    GGML_ASSERT(ggml_is_contiguous(dst));

    float scale;
    int32_t n_val;
    memcpy(&scale, (const int32_t *) dst->op_params + 0, sizeof(float));
    memcpy(&n_val, (const int32_t *) dst->op_params + 1, sizeof(int32_t));

    const int d_lat  = q->ne[0];
    const int n_head = q->ne[1];
    const int n_tok  = q->ne[2];
    const int n_tk   = top_k->ne[0];

    const int nth = 256;
    const size_t smem = ((size_t) d_lat + n_tk + nth) * sizeof(float) + (size_t) n_tk * sizeof(int);

    dim3 grid(n_tok, n_head, 1);
    sparse_mla_attn_f32<<<grid, nth, smem, ctx.stream()>>>(
        (const float *) k->data, (const float *) q->data, (const int *) top_k->data,
        (const float *) mask->data, (float *) dst->data,
        d_lat, n_val, n_head, n_tok, n_tk, scale,
        (int)(k->nb[1]/sizeof(float)), (int)(mask->nb[1]/sizeof(float)), (int)(top_k->nb[1]/sizeof(int)));
}
