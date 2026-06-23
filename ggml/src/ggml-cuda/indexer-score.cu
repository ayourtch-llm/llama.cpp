#include "indexer-score.cuh"

// dst[key,t] = sum_h relu(sum_d q[d,h,t] * k[d,key]) * w[h,t], reduced over heads so the
// [n_kv, n_tokens, n_head] product is never materialized. One block per (token, stream);
// q[:, t, :] and w[:, t] are staged in shared memory and reused across all keys.
//
// q is staged transposed (q_sh[d*N_HEAD + h]) so the inner head loop is bank-conflict free,
// and each key element k_key[d] is loaded from global ONCE and reused across all heads (the
// head loop accumulates per-head partial dot products in registers) - otherwise k traffic is
// inflated n_head-fold.
//
// K is read with dequant-on-read via its tensor byte strides (k_row = nb[1] = byte stride
// between key rows, k_stream = nb[3] = byte stride between streams), so the op consumes the
// q8_0 indexer KV-cache view directly (no per-token ggml_cast -> f32). f32 K is a direct read;
// q8_0 K is dequantized per element (block_q8_0, QK8_0 = 32). Templated on K_Q8_0 so the hot
// load loop has no per-element type branch.

template <bool K_Q8_0>
static __device__ __forceinline__ float load_k(const char * __restrict__ row, int d) {
    if constexpr (K_Q8_0) {
        const block_q8_0 * b = (const block_q8_0 *) row;
        return __half2float(b[d >> 5].d) * (float) b[d >> 5].qs[d & 31];
    }
    return ((const float *) row)[d];
}

template <int N_HEAD, bool K_Q8_0>
static __global__ void indexer_score(
        const void  * __restrict__ k,
        const float * __restrict__ q,
        const float * __restrict__ w,
        float       * __restrict__ dst,
        const int D, const int n_kv, const int n_tokens,
        const size_t k_row, const size_t k_stream) {
    const int t = blockIdx.x;
    const int s = blockIdx.y;

    const char  * k_s = (const char *) k   + (size_t) s * k_stream;
    const float * q_s = q   + (size_t) s * n_tokens * N_HEAD * D;
    const float * w_s = w   + (size_t) s * n_tokens * N_HEAD;
    float       * d_s = dst + (size_t) s * n_tokens * n_kv;

    extern __shared__ float smem[];
    float * q_sh = smem;               // [d*N_HEAD + h]
    float * w_sh = smem + N_HEAD * D;

    for (int idx = threadIdx.x; idx < N_HEAD * D; idx += blockDim.x) {
        const int h = idx % N_HEAD;
        const int d = idx / N_HEAD;
        q_sh[idx] = q_s[((size_t) h * n_tokens + t) * D + d];
    }
    for (int h = threadIdx.x; h < N_HEAD; h += blockDim.x) {
        w_sh[h] = w_s[(size_t) t * N_HEAD + h];
    }
    __syncthreads();

    for (int key = threadIdx.x; key < n_kv; key += blockDim.x) {
        const char * k_key = k_s + (size_t) key * k_row;
        float dot[N_HEAD];
        #pragma unroll
        for (int h = 0; h < N_HEAD; h++) {
            dot[h] = 0.0f;
        }
        for (int d = 0; d < D; d++) {
            const float kd = load_k<K_Q8_0>(k_key, d);
            const float * q_d = q_sh + d * N_HEAD;
            #pragma unroll
            for (int h = 0; h < N_HEAD; h++) {
                dot[h] += q_d[h] * kd;
            }
        }
        float acc = 0.0f;
        #pragma unroll
        for (int h = 0; h < N_HEAD; h++) {
            if (dot[h] > 0.0f) {
                acc += dot[h] * w_sh[h];
            }
        }
        d_s[(size_t) t * n_kv + key] = acc;
    }
}

// correctness fallback for head counts without a template instance (not perf tuned)
template <bool K_Q8_0>
static __global__ void indexer_score_generic(
        const void  * __restrict__ k,
        const float * __restrict__ q,
        const float * __restrict__ w,
        float       * __restrict__ dst,
        const int D, const int n_kv, const int n_tokens, const int n_head,
        const size_t k_row, const size_t k_stream) {
    const int t = blockIdx.x;
    const int s = blockIdx.y;

    const char  * k_s = (const char *) k   + (size_t) s * k_stream;
    const float * q_s = q   + (size_t) s * n_tokens * n_head * D;
    const float * w_s = w   + (size_t) s * n_tokens * n_head;
    float       * d_s = dst + (size_t) s * n_tokens * n_kv;

    extern __shared__ float smem[];
    float * q_sh = smem;
    float * w_sh = smem + (size_t) n_head * D;

    for (int idx = threadIdx.x; idx < n_head * D; idx += blockDim.x) {
        const int h = idx % n_head;
        const int d = idx / n_head;
        q_sh[idx] = q_s[((size_t) h * n_tokens + t) * D + d];
    }
    for (int h = threadIdx.x; h < n_head; h += blockDim.x) {
        w_sh[h] = w_s[(size_t) t * n_head + h];
    }
    __syncthreads();

    for (int key = threadIdx.x; key < n_kv; key += blockDim.x) {
        const char * k_key = k_s + (size_t) key * k_row;
        float acc = 0.0f;
        for (int h = 0; h < n_head; h++) {
            float dot = 0.0f;
            for (int d = 0; d < D; d++) {
                dot += q_sh[d * n_head + h] * load_k<K_Q8_0>(k_key, d);
            }
            if (dot > 0.0f) {
                acc += dot * w_sh[h];
            }
        }
        d_s[(size_t) t * n_kv + key] = acc;
    }
}

void ggml_cuda_op_indexer_score(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * k = dst->src[0];
    const ggml_tensor * q = dst->src[1];
    const ggml_tensor * w = dst->src[2];

    GGML_ASSERT(k->type == GGML_TYPE_F32 || k->type == GGML_TYPE_Q8_0);
    GGML_ASSERT(q->type == GGML_TYPE_F32);
    GGML_ASSERT(w->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(q));
    GGML_ASSERT(ggml_is_contiguous(w));
    // K is read via its tensor strides (nb[1]/nb[3]) with dequant-on-read, so it no longer
    // needs to be contiguous - the q8_0 indexer KV-cache view is consumed directly.
    GGML_ASSERT(k->nb[0] == ggml_type_size(k->type));

    const int D        = q->ne[0];
    const int n_tokens = q->ne[1];
    const int n_head   = q->ne[2];
    const int n_stream = q->ne[3];
    const int n_kv     = k->ne[1];

    const size_t smem = ((size_t) n_head * D + n_head) * sizeof(float);

    // K byte strides: key-row stride (nb[1]) and stream stride (nb[3]). Valid for both f32
    // (nb[1] = D*4) and q8_0 (nb[1] = (D/QK8_0)*sizeof(block_q8_0)).
    const size_t k_row    = k->nb[1];
    const size_t k_stream = k->nb[3];

    dim3 grid(n_tokens, n_stream, 1);
    cudaStream_t stream = ctx.stream();
    const void * kp = (const void *) k->data;
    const float * qp = (const float *) q->data;
    const float * wp = (const float *) w->data;
    float       * dp = (float *) dst->data;

    if (k->type == GGML_TYPE_Q8_0) {
        switch (n_head) {
            case 16: indexer_score<16, true><<<grid, 256, smem, stream>>>(kp, qp, wp, dp, D, n_kv, n_tokens, k_row, k_stream); break;
            case 32: indexer_score<32, true><<<grid, 256, smem, stream>>>(kp, qp, wp, dp, D, n_kv, n_tokens, k_row, k_stream); break;
            case 64: indexer_score<64, true><<<grid, 256, smem, stream>>>(kp, qp, wp, dp, D, n_kv, n_tokens, k_row, k_stream); break;
            default: indexer_score_generic<true><<<grid, 256, smem, stream>>>(kp, qp, wp, dp, D, n_kv, n_tokens, n_head, k_row, k_stream); break;
        }
    } else {
        switch (n_head) {
            case 16: indexer_score<16, false><<<grid, 256, smem, stream>>>(kp, qp, wp, dp, D, n_kv, n_tokens, k_row, k_stream); break;
            case 32: indexer_score<32, false><<<grid, 256, smem, stream>>>(kp, qp, wp, dp, D, n_kv, n_tokens, k_row, k_stream); break;
            case 64: indexer_score<64, false><<<grid, 256, smem, stream>>>(kp, qp, wp, dp, D, n_kv, n_tokens, k_row, k_stream); break;
            default: indexer_score_generic<false><<<grid, 256, smem, stream>>>(kp, qp, wp, dp, D, n_kv, n_tokens, n_head, k_row, k_stream); break;
        }
    }
}
