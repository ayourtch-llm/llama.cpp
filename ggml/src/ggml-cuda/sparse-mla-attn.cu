#include "sparse-mla-attn.cuh"

// DSA sparse MLA attention, flash-style with optional split-K (flash-decode).
//
// The latent KV is shared across all query heads (MLA has one KV head), so a block
// processes one token and a TILE of SMLA_HTILE heads (one warp each): each gathered
// latent row is streamed into shared memory ONCE and reused by every head in the tile.
// Online softmax keeps the per-head running max/sum and a value accumulator in registers
// - nothing per-token-sized hits global memory.
//   out[d, h, t] = sum_i softmax_i(scale * q[:,h,t] . k[:,key_i] + mask[key_i,t]) * k[d, key_i]
// with key_i = top_k[i, t] and value = the first n_val dims of the latent row.
//
// SPLIT-K (flash-decode): when n_tok is small (e.g. decode n_tok=1), the per-(tok,head-tile)
// parallelism is too low to fill the GPU. We add a third grid dim that splits the n_tk gathered
// keys into n_splits chunks; each block does online softmax over its chunk and writes the
// unnormalized value accumulator plus the running (m, l) to a tmp buffer. A reduction kernel
// then merges the n_splits partials per (tok, head) into the final output. Same math as the
// n_splits=1 path; just more parallelism.
#define SMLA_VPL 16  // value regs per lane (n_val <= 32*SMLA_VPL = 512)
// Reduce n_val parallelism: gm/gl are d-independent, so each block writes a disjoint val range
// with no second reduction. One block per (tok, head, val-tile); blockDim threads each own one d.
#define SMLA_VAL_TILE 128
// Split-K cap: caps the tmp_acc/tmp_meta traffic (both stages move ~n_splits*n_val*n_head
// floats) and the reduce work, while still filling the GPU (splitk grid = n_tok*n_head_tiles*
// n_splits). At decode (n_tok=1, n_head_tiles=4) this is ~n_splits/4 waves on ~150 SMs.
#define SMLA_MAX_SPLITS 48

// Latent K row loader. f32 is a direct read; q8_0 is dequantized on the fly:
// each row is block_q8_0 = half d + 32 int8 quants (34 bytes/block, d_lat=576 -> 18 blocks).
// Templated on K_Q8_0 so the hot loop has no per-element type branch. k_row is the BYTE
// stride between latent rows (the view's nb[1]); for f32 it equals d_lat*sizeof(float),
// for q8_0 it equals (d_lat/QK8_0)*sizeof(block_q8_0).
template <bool K_Q8_0>
static __device__ __forceinline__ float load_lat(const void * __restrict__ k_base, size_t row_off, int d) {
    const char * row = (const char *) k_base + row_off;
    if constexpr (K_Q8_0) {
        const block_q8_0 * b = (const block_q8_0 *) row;
        return __half2float(b[d >> 5].d) * (float) b[d >> 5].qs[d & 31];
    }
    return ((const float *) row)[d];
}

template <bool K_Q8_0>
static __global__ void sparse_mla_attn_f32(
        const void  * __restrict__ k,
        const float * __restrict__ q,
        const int   * __restrict__ top_k,
        const char  * __restrict__ mask,
        float       * __restrict__ dst,
        const int d_lat, const int n_val, const int n_head, const int n_tok,
        const int n_tk, const float scale,
        const size_t k_row, const int64_t mask_nb1, const int topk_row, const int mask_f16) {
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
        const size_t row_off = (size_t) key * k_row;
        for (int d = threadIdx.x; d < d_lat; d += blockDim.x) {
            lat_sh[d] = load_lat<K_Q8_0>(k, row_off, d);
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

// Split-K stage 1: same math as sparse_mla_attn_f32 but each block handles a CHUNK of the
// n_tk keys (blockIdx.z out of n_splits) and writes the UNNORMALIZED value accumulator
// (sum_i p_i * v_i) plus the running (m, l) to tmp_acc / tmp_meta. The reduction kernel
// below merges the partials.
//
// tmp_acc layout: [n_tok, n_head, n_splits, n_val]  (per head, per split, unnormalized v sum)
// tmp_meta layout: [n_tok, n_head, n_splits, 2]     (m at [.,.,.,0], l at [.,.,.,1])
template <bool K_Q8_0>
static __global__ void sparse_mla_attn_f32_splitk(
        const void  * __restrict__ k,
        const float * __restrict__ q,
        const int   * __restrict__ top_k,
        const char  * __restrict__ mask,
        float       * __restrict__ tmp_acc,
        float       * __restrict__ tmp_meta,
        const int d_lat, const int n_val, const int n_head, const int n_tok,
        const int n_tk, const int n_splits, const float scale,
        const size_t k_row, const int64_t mask_nb1, const int topk_row, const int mask_f16) {
    const int t    = blockIdx.x;
    const int h0   = blockIdx.y * SMLA_HTILE;
    const int s    = blockIdx.z;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int h    = h0 + warp;

    // Even key-chunk boundaries so all splits see the same range (the trailing split may
    // be short; the reduction handles empty partials via l==0 -> weight 0).
    const int per  = (n_tk + n_splits - 1) / n_splits;
    const int k_lo = s*per;
    const int k_hi = k_lo + per > n_tk ? n_tk : k_lo + per;

    extern __shared__ float smem[];
    float * q_sh   = smem;
    float * lat_sh = q_sh + SMLA_HTILE * d_lat;

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

    for (int i = k_lo; i < k_hi; i++) {
        const int key = idx_t[i];
        const size_t row_off = (size_t) key * k_row;
        for (int d = threadIdx.x; d < d_lat; d += blockDim.x) {
            lat_sh[d] = load_lat<K_Q8_0>(k, row_off, d);
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
        // Write UNNORMALIZED acc (will be normalized in the reduction after merging splits).
        float * a_dst = tmp_acc + (((size_t) t*n_head + h)*n_splits)*n_val;
        #pragma unroll
        for (int j = 0; j < SMLA_VPL; j++) {
            const int d = lane + j*32;
            if (d < n_val) {
                a_dst[(size_t) s*n_val + d] = acc[j];
            }
        }
        if (lane == 0) {
            float * meta_dst = tmp_meta + (((size_t) t*n_head + h)*n_splits)*2;
            meta_dst[(size_t) s*2 + 0] = m;
            meta_dst[(size_t) s*2 + 1] = l;
        }
    }
}

// Split-K stage 2: merge n_splits partials per (tok, head), parallelized across n_val.
// One block per (tok, head, val-tile); blockDim threads, each owning exactly one output d in
// the tile's disjoint range [vlo, vhi). Online softmax merge: for split s with (m_s, l_s, acc_s),
//   global_m = max_s m_s
//   global_l = sum_s l_s * exp(m_s - global_m)
//   out[d]   = (sum_s acc_s[d] * exp(m_s - global_m)) / global_l
// global_m and global_l depend only on the per-split (m,l), NOT on d - so every thread computes
// the same values (deterministic, no sync/broadcast) and each writes its own d with no second
// reduction. Splits with l_s == 0 contribute nothing (exp(-inf) == 0 handles m_s == -inf too).
static __global__ void sparse_mla_attn_f32_reduce(
        const float * __restrict__ tmp_acc,
        const float * __restrict__ tmp_meta,
        float       * __restrict__ dst,
        const int n_val, const int n_head, const int n_splits) {
    const int t   = blockIdx.x;
    const int h   = blockIdx.y;
    const int vlo = blockIdx.z * SMLA_VAL_TILE;
    const int vhi = vlo + SMLA_VAL_TILE > n_val ? n_val : vlo + SMLA_VAL_TILE;

    const float * acc_t  = tmp_acc  + ((size_t) t*n_head + h)*n_splits*n_val;
    const float * meta_t = tmp_meta + ((size_t) t*n_head + h)*n_splits*2;

    // gm/gl and the per-split weights exp(m_s - gm) are d-independent; compute them once in
    // thread 0 and share, instead of redoing the same expf in every value lane and every split.
    __shared__ float w_sh[SMLA_MAX_SPLITS];
    __shared__ float inv_sh;
    if (threadIdx.x == 0) {
        float gm = -INFINITY;
        for (int s = 0; s < n_splits; s++) {
            gm = fmaxf(gm, meta_t[(size_t) s*2 + 0]);
        }
        float gl = 0.0f;
        for (int s = 0; s < n_splits; s++) {
            const float ms = meta_t[(size_t) s*2 + 0];
            const float ls = meta_t[(size_t) s*2 + 1];
            float ws = 0.0f;
            if (ls > 0.0f && ms > -INFINITY) {
                ws = expf(ms - gm);
                gl += ls * ws;
            }
            w_sh[s] = ws;
        }
        inv_sh = gl > 0.0f ? 1.0f/gl : 0.0f;
    }
    __syncthreads();

    // Each thread owns one d in [vlo, vhi); accumulate its weighted value across splits.
    const int d = vlo + threadIdx.x;
    if (d < vhi) {
        float a = 0.0f;
        for (int s = 0; s < n_splits; s++) {
            a += acc_t[(size_t) s*n_val + d] * w_sh[s];
        }
        dst[((size_t) t*n_head + h)*n_val + d] = a * inv_sh;
    }
}

void ggml_cuda_op_sparse_mla_attn(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * k     = dst->src[0];
    const ggml_tensor * q     = dst->src[1];
    const ggml_tensor * top_k = dst->src[2];
    const ggml_tensor * mask  = dst->src[3];

    GGML_ASSERT(k->type == GGML_TYPE_F32 || k->type == GGML_TYPE_Q8_0);
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

    // Latent K row stride in BYTES (valid for both f32 and q8_0: nb[1] is the per-row byte
    // stride of the 2D K view; f32 -> d_lat*4, q8_0 -> (d_lat/QK8_0)*sizeof(block_q8_0)).
    const size_t k_row = k->nb[1];

    const size_t smem = (size_t)(SMLA_HTILE + 1) * d_lat * sizeof(float);

    cudaStream_t stream = ctx.stream();

    // Split-K heuristics: target ~4 blocks per SM so the GPU stays filled. Skip the split
    // (and the reduction kernel launch) when there's already enough per-(tok, head-tile)
    // parallelism - prefill and large batches don't need it.
    const int n_head_tiles = (n_head + SMLA_HTILE - 1) / SMLA_HTILE;
    const int nsm          = ggml_cuda_info().devices[ggml_cuda_get_device()].nsm;
    const int target_blocks = 4 * nsm;
    int n_splits = 1;
    if (n_tok > 0 && n_tok*n_head_tiles < target_blocks) {
        n_splits = (target_blocks + n_tok*n_head_tiles - 1) / (n_tok*n_head_tiles);
        // cap so each split sees at least 8 keys (smaller chunks aren't worth the overhead)
        const int max_splits = n_tk > 0 ? (n_tk + 7) / 8 : 1;
        if (n_splits > max_splits) n_splits = max_splits;
        // cap total splits: the reduce is memory-bound (reads n_splits*n_val*n_head partials)
        // and the splitk stage writes that much tmp, so n_splits beyond what fills ~1 wave of
        // SMs only adds traffic. SMLA_MAX_SPLITS ~= 1 wave at decode (n_tok*n_head_tiles = 4).
        if (n_splits > SMLA_MAX_SPLITS) n_splits = SMLA_MAX_SPLITS;
        if (n_splits < 1) n_splits = 1;
    }

    // Branch on K type once at launch time so each kernel is monomorphic (no per-element
    // branch in the load loop).
    // Reduce grid: one block per (tok, head, val-tile). gm/gl are d-independent so each
    // val-tile block writes a disjoint d-range with no second reduction (see the kernel).
    const int n_val_tiles = (n_val + SMLA_VAL_TILE - 1) / SMLA_VAL_TILE;
    if (k->type == GGML_TYPE_Q8_0) {
        if (n_splits == 1) {
            dim3 grid(n_tok, n_head_tiles, 1);
            sparse_mla_attn_f32<true><<<grid, SMLA_HTILE*32, smem, stream>>>(
                (const void *) k->data, (const float *) q->data, (const int *) top_k->data,
                (const char *) mask->data, (float *) dst->data,
                d_lat, n_val, n_head, n_tok, n_tk, scale,
                k_row, (int64_t) mask->nb[1], (int)(top_k->nb[1]/sizeof(int)),
                mask->type == GGML_TYPE_F16);
            return;
        }
        ggml_cuda_pool_alloc<float> tmp_acc_alloc (ctx.pool(), (size_t) n_tok*n_head*n_splits*n_val);
        ggml_cuda_pool_alloc<float> tmp_meta_alloc(ctx.pool(), (size_t) n_tok*n_head*n_splits*2);
        float * tmp_acc  = tmp_acc_alloc.get();
        float * tmp_meta = tmp_meta_alloc.get();

        dim3 grid1(n_tok, n_head_tiles, n_splits);
        sparse_mla_attn_f32_splitk<true><<<grid1, SMLA_HTILE*32, smem, stream>>>(
            (const void *) k->data, (const float *) q->data, (const int *) top_k->data,
            (const char *) mask->data, tmp_acc, tmp_meta,
            d_lat, n_val, n_head, n_tok, n_tk, n_splits, scale,
            k_row, (int64_t) mask->nb[1], (int)(top_k->nb[1]/sizeof(int)),
            mask->type == GGML_TYPE_F16);

        dim3 grid2(n_tok, n_head, n_val_tiles);
        sparse_mla_attn_f32_reduce<<<grid2, SMLA_VAL_TILE, 0, stream>>>(
            tmp_acc, tmp_meta, (float *) dst->data, n_val, n_head, n_splits);
        return;
    }

    if (n_splits == 1) {
        dim3 grid(n_tok, n_head_tiles, 1);
        sparse_mla_attn_f32<false><<<grid, SMLA_HTILE*32, smem, stream>>>(
            (const void *) k->data, (const float *) q->data, (const int *) top_k->data,
            (const char *) mask->data, (float *) dst->data,
            d_lat, n_val, n_head, n_tok, n_tk, scale,
            k_row, (int64_t) mask->nb[1], (int)(top_k->nb[1]/sizeof(int)),
            mask->type == GGML_TYPE_F16);
        return;
    }

    // Tmp buffers: per (tok, head, split) unnormalized acc + (m, l).
    ggml_cuda_pool_alloc<float> tmp_acc_alloc (ctx.pool(), (size_t) n_tok*n_head*n_splits*n_val);
    ggml_cuda_pool_alloc<float> tmp_meta_alloc(ctx.pool(), (size_t) n_tok*n_head*n_splits*2);
    float * tmp_acc  = tmp_acc_alloc.get();
    float * tmp_meta = tmp_meta_alloc.get();

    dim3 grid1(n_tok, n_head_tiles, n_splits);
    sparse_mla_attn_f32_splitk<false><<<grid1, SMLA_HTILE*32, smem, stream>>>(
        (const void *) k->data, (const float *) q->data, (const int *) top_k->data,
        (const char *) mask->data, tmp_acc, tmp_meta,
        d_lat, n_val, n_head, n_tok, n_tk, n_splits, scale,
        k_row, (int64_t) mask->nb[1], (int)(top_k->nb[1]/sizeof(int)),
        mask->type == GGML_TYPE_F16);

    dim3 grid2(n_tok, n_head, n_val_tiles);
    sparse_mla_attn_f32_reduce<<<grid2, SMLA_VAL_TILE, 0, stream>>>(
        tmp_acc, tmp_meta, (float *) dst->data, n_val, n_head, n_splits);
}
