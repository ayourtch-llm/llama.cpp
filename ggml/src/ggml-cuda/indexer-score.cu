#include "indexer-score.cuh"

// dst[key,t] = sum_h relu(sum_d q[d,h,t] * k[d,key]) * w[h,t], reduced over heads so the
// [n_kv, n_tokens, n_head] product is never materialized. One block per (token, stream);
// q[:, t, :] and w[:, t] are staged in shared memory and reused across all keys.
//
// q is staged transposed (q_sh[d*N_HEAD + h]) so the inner head loop is bank-conflict free,
// and each key element k_key[d] is loaded from global ONCE and reused across all heads (the
// head loop accumulates per-head partial dot products in registers) - otherwise k traffic is
// inflated n_head-fold.
template <int N_HEAD>
static __global__ void indexer_score_f32(
        const float * __restrict__ k,
        const float * __restrict__ q,
        const float * __restrict__ w,
        float       * __restrict__ dst,
        const int D, const int n_kv, const int n_tokens) {
    const int t = blockIdx.x;
    const int s = blockIdx.y;

    const float * k_s = k   + (size_t) s * n_kv * D;
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
        const float * k_key = k_s + (size_t) key * D;
        float dot[N_HEAD];
        #pragma unroll
        for (int h = 0; h < N_HEAD; h++) {
            dot[h] = 0.0f;
        }
        for (int d = 0; d < D; d++) {
            const float kd = k_key[d];
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
static __global__ void indexer_score_f32_generic(
        const float * __restrict__ k,
        const float * __restrict__ q,
        const float * __restrict__ w,
        float       * __restrict__ dst,
        const int D, const int n_kv, const int n_tokens, const int n_head) {
    const int t = blockIdx.x;
    const int s = blockIdx.y;

    const float * k_s = k   + (size_t) s * n_kv * D;
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
        const float * k_key = k_s + (size_t) key * D;
        float acc = 0.0f;
        for (int h = 0; h < n_head; h++) {
            float dot = 0.0f;
            for (int d = 0; d < D; d++) {
                dot += q_sh[d * n_head + h] * k_key[d];
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

    GGML_ASSERT(k->type == GGML_TYPE_F32);
    GGML_ASSERT(q->type == GGML_TYPE_F32);
    GGML_ASSERT(w->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(k));
    GGML_ASSERT(ggml_is_contiguous(q));
    GGML_ASSERT(ggml_is_contiguous(w));

    const int D        = q->ne[0];
    const int n_tokens = q->ne[1];
    const int n_head   = q->ne[2];
    const int n_stream = q->ne[3];
    const int n_kv     = k->ne[1];

    const size_t smem = ((size_t) n_head * D + n_head) * sizeof(float);

    dim3 grid(n_tokens, n_stream, 1);
    cudaStream_t stream = ctx.stream();
    const float * kp = (const float *) k->data;
    const float * qp = (const float *) q->data;
    const float * wp = (const float *) w->data;
    float       * dp = (float *) dst->data;

    switch (n_head) {
        case 16: indexer_score_f32<16><<<grid, 256, smem, stream>>>(kp, qp, wp, dp, D, n_kv, n_tokens); break;
        case 32: indexer_score_f32<32><<<grid, 256, smem, stream>>>(kp, qp, wp, dp, D, n_kv, n_tokens); break;
        case 64: indexer_score_f32<64><<<grid, 256, smem, stream>>>(kp, qp, wp, dp, D, n_kv, n_tokens); break;
        default: indexer_score_f32_generic<<<grid, 256, smem, stream>>>(kp, qp, wp, dp, D, n_kv, n_tokens, n_head); break;
    }
}
