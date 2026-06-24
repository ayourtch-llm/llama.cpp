#include "indexer-score.cuh"
#include "mma.cuh"

// dst[key,t] = sum_h relu(sum_d q[d,h,t] * k[d,key]) * w[h,t], reduced over heads so the
// [n_kv, n_tokens, n_head] product is never materialized.
//
// Grid is (n_tokens, n_stream, n_key_tiles): each block handles one (token, stream) and a TILE
// of keys. q[:, :, t] and w[:, t] are staged in shared memory once per block and reused across
// the tile's keys. This fills the GPU at decode (n_tokens=1, n_stream=1), where the original
// (n_tokens, n_stream, 1) grid launched ONE block and looped every key inside it - leaving the
// other ~150 SMs idle. Since each dst[key, t] is an INDEPENDENT reduction over (d, h) there is
// no cross-key dependency, so tiling keys across blocks needs no split-K reduction: block z just
// writes dst[tile_lo..tile_hi, t] at the absolute key index.
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

// Keys scored per block (grid.z tile). Picked so a full decode KV (~21840) spreads across
// ~n_kv/INDEXER_KEY_TILE blocks (>= 150 SMs at n_kv >= ~19k), while each block still amortizes
// the q/w shared-mem staging over a worthwhile run of keys. Equal to blockDim so each thread
// owns exactly one key in a full tile (no idle threads, no intra-block striding).
#define INDEXER_KEY_TILE 128

template <bool K_Q8_0>
static __device__ __forceinline__ float load_k(const char * __restrict__ row, int d) {
    if constexpr (K_Q8_0) {
        const block_q8_0 * b = (const block_q8_0 *) row;
        return __half2float(b[d >> 5].d) * (float) b[d >> 5].qs[d & 31];
    }
    return ((const float *) row)[d];
}

template <int N_HEAD, bool K_Q8_0, bool DST_F16>
static __global__ void indexer_score(
        const void  * __restrict__ k,
        const float * __restrict__ q,
        const float * __restrict__ w,
        void        * __restrict__ dst,
        const int D, const int n_kv, const int n_tokens,
        const size_t k_row, const size_t k_stream) {
    const int t = blockIdx.x;
    const int s = blockIdx.y;

    const char  * k_s = (const char *) k   + (size_t) s * k_stream;
    const float * q_s = q   + (size_t) s * n_tokens * N_HEAD * D;
    const float * w_s = w   + (size_t) s * n_tokens * N_HEAD;
    char        * d_s = (char *) dst + (size_t) s * n_tokens * n_kv * (DST_F16 ? sizeof(half) : sizeof(float));

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

    // This block's disjoint key range: [tile_lo, tile_hi). Each key is owned by exactly one
    // thread (threadIdx.x within the tile), so writes never race across blocks.
    const int tile_lo = blockIdx.z * INDEXER_KEY_TILE;
    const int tile_hi = tile_lo + INDEXER_KEY_TILE > n_kv ? n_kv : tile_lo + INDEXER_KEY_TILE;

    for (int key = tile_lo + threadIdx.x; key < tile_hi; key += blockDim.x) {
        const char * k_key = k_s + (size_t) key * k_row;
        float dot[N_HEAD];
        #pragma unroll
        for (int h = 0; h < N_HEAD; h++) {
            dot[h] = 0.0f;
        }
        if constexpr (K_Q8_0) {
            // Iterate q8_0 blocks explicitly so the per-block scale (half->float) is loaded
            // ONCE per 32-element block instead of being recomputed every d (d>>5 is a runtime
            // induction the compiler can't CSE). D % QK8_0 == 0 is guaranteed by the op.
            const block_q8_0 * kb = (const block_q8_0 *) k_key;
            for (int blk = 0; blk < D / QK8_0; blk++) {
                const float   scale = __half2float(kb[blk].d);
                const float * q_blk = q_sh + (blk * QK8_0) * N_HEAD;
                #pragma unroll
                for (int q = 0; q < QK8_0; q++) {
                    const float kd = scale * (float) kb[blk].qs[q];
                    const float * q_d = q_blk + q * N_HEAD;
                    #pragma unroll
                    for (int h = 0; h < N_HEAD; h++) {
                        dot[h] += q_d[h] * kd;
                    }
                }
            }
        } else {
            for (int d = 0; d < D; d++) {
                const float kd = ((const float *) k_key)[d];
                const float * q_d = q_sh + d * N_HEAD;
                #pragma unroll
                for (int h = 0; h < N_HEAD; h++) {
                    dot[h] += q_d[h] * kd;
                }
            }
        }
        float acc = 0.0f;
        #pragma unroll
        for (int h = 0; h < N_HEAD; h++) {
            // relu(x) = max(x,0): branchless so every lane issues the fma (keys diverge per-lane).
            acc += fmaxf(dot[h], 0.0f) * w_sh[h];
        }
        if constexpr (DST_F16) {
            // f32 accumulation, half store: halves the score-tensor write + downstream top-k read.
            ((half *) d_s)[(size_t) t * n_kv + key] = __float2half(acc);
        } else {
            ((float *) d_s)[(size_t) t * n_kv + key] = acc;
        }
    }
}

// correctness fallback for head counts without a template instance (not perf tuned)
template <bool K_Q8_0, bool DST_F16>
static __global__ void indexer_score_generic(
        const void  * __restrict__ k,
        const float * __restrict__ q,
        const float * __restrict__ w,
        void        * __restrict__ dst,
        const int D, const int n_kv, const int n_tokens, const int n_head,
        const size_t k_row, const size_t k_stream) {
    const int t = blockIdx.x;
    const int s = blockIdx.y;

    const char  * k_s = (const char *) k   + (size_t) s * k_stream;
    const float * q_s = q   + (size_t) s * n_tokens * n_head * D;
    const float * w_s = w   + (size_t) s * n_tokens * n_head;
    char        * d_s = (char *) dst + (size_t) s * n_tokens * n_kv * (DST_F16 ? sizeof(half) : sizeof(float));

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

    const int tile_lo = blockIdx.z * INDEXER_KEY_TILE;
    const int tile_hi = tile_lo + INDEXER_KEY_TILE > n_kv ? n_kv : tile_lo + INDEXER_KEY_TILE;

    for (int key = tile_lo + threadIdx.x; key < tile_hi; key += blockDim.x) {
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
        if constexpr (DST_F16) {
            ((half *) d_s)[(size_t) t * n_kv + key] = __float2half(acc);
        } else {
            ((float *) d_s)[(size_t) t * n_kv + key] = acc;
        }
    }
}

// dst[key,t] = sum_h relu(sum_d q[d,h,t] * k[d,key]) * w[h,t], reduced over heads so the
// [n_kv, n_tokens, n_head] product is never materialized.
//
// Prefill tensor-core path. Reached only for q8_0 K, D=128, n_head=64, n_tokens>=16 on
// fp16-mma hardware (see dispatch in ggml_cuda_op_indexer_score). Decodes and all other shapes
// keep the scalar kernel above.
//
// Tile: BM keys/CTA, BT tokens/CTA, BH heads/head-group; loops 8 head-groups over n_head=64.
// One CTA computes a [BM, BT] output tile for one stream. K (q8_0 -> half) is dequantized into
// shared ONCE per CTA and reused across all BT*BH Q columns of every head-group - that K reuse
// is the algorithmic win over the scalar per-token K scan. Q (f32 -> half) and W are re-staged
// per head-group. The MMA accumulates dot[key,t,head] in f32, and the epilogue folds
// relu(dot)*w straight into a per-thread [2 m-rows x BT tokens] register accumulator that
// persists across the 8 head-groups - the 64-head reduction happens inline (quad shuffle per
// head-group), so dot[key,t,head] is never written to memory.
//
// C-fragment (m16n8k16 f32) -> (key, token, head) mapping: for tile<16,8,float> each lane holds
//   x[0]=dot[m_a, hh_a]  x[1]=dot[m_a, hh_b]  x[2]=dot[m_b, hh_a]  x[3]=dot[m_b, hh_b]
// where m_a=tx/4, m_b=8+tx/4, hh_a=(tx%4)*2, hh_b=hh_a+1 (m = row within the warp's 16-key
// stripe, hh = head within the BH=8 head-group). The 8 heads for a given (m, token) are spread
// over the 4 lanes that share tx/4, so a xor{1,2} quad shuffle completes the per-head-group sum.
template <int BM, int BT, int BH, int D, int N_HEAD, bool DST_F16>
static __global__ void indexer_score_hmma_prefill_q8_64(
        const void  * __restrict__ k,
        const float * __restrict__ q,
        const float * __restrict__ w,
        void        * __restrict__ dst,
        const int n_kv, const int n_tokens,
        const size_t k_row, const size_t k_stream) {
    using namespace ggml_cuda_mma;

    static_assert(BM % 16 == 0 && D % 16 == 0 && BH == 8 && N_HEAD % BH == 0, "bad tile");
    constexpr int NWARPS      = BM / 16;            // 1 warp per 16-key stripe
    constexpr int M_PER_WARP  = 16;
    constexpr int D2          = D / 2;              // half2 per K/Q row
    constexpr int STRIDE_H2   = D2 + 4;             // +4 half2 pad/skew vs bank conflicts (cf. fattn-mma)
    constexpr int N_COLS      = BT * BH;            // Q columns per head-group (= 64)
    constexpr int K_TILES     = D / 16;             // mma k16 -> 8 k-tiles
    constexpr int N_TILES     = N_COLS / 8;         // mma n8  -> BT n-tiles (1 token each)

    const int m0 = blockIdx.x * BM;
    const int t0 = blockIdx.y * BT;
    const int s  = blockIdx.z;

    const char  * k_s = (const char *) k + (size_t) s * k_stream;
    const float * q_s = q + (size_t) s * n_tokens * N_HEAD * D;
    const float * w_s = w + (size_t) s * n_tokens * N_HEAD;
    char        * d_s = (char *) dst + ((size_t) s * n_tokens * n_kv) * (DST_F16 ? sizeof(half) : sizeof(float));

    const int tx  = threadIdx.x;       // lane in [0,32)
    const int ty  = threadIdx.y;       // warp in [0,NWARPS)
    const int tid = ty * 32 + tx;      // flat [0, 32*NWARPS)

    extern __shared__ char smem_raw[];
    half2 * K_sh = (half2 *) smem_raw;                                   // [BM][STRIDE_H2]
    half2 * Q_sh = K_sh + BM * STRIDE_H2;                                // [N_COLS][STRIDE_H2]
    float  * w_sh = (float *)(Q_sh + N_COLS * STRIDE_H2);               // [BT][BH]

    // Cooperative dequant of the K tile (q8_0 -> half) once per CTA. Each q8_0 block (32 elems)
    // is owned by one thread: scale is applied in f32 (matches the scalar kernel's math) and the
    // final K value is rounded to half (the only unavoidable f16-HMMA rounding on the K side).
    constexpr int NBLOCKS = BM * (D / QK8_0);
    for (int b = tid; b < NBLOCKS; b += 32 * NWARPS) {
        const int mi  = b / (D / QK8_0);
        const int blk = b % (D / QK8_0);
        const int m   = m0 + mi;
        half2 * dst_h2 = K_sh + mi * STRIDE_H2 + blk * (QK8_0 / 2);
        if (m < n_kv) {
            const block_q8_0 * kb =
                (const block_q8_0 *)(k_s + (size_t) m * k_row + (size_t) blk * sizeof(block_q8_0));
            const float scale = __half2float(kb->d);
            const int8_t * qs = kb->qs;
            #pragma unroll
            for (int r = 0; r < QK8_0 / 2; ++r) {
                dst_h2[r] = make_half2(
                    __float2half(scale * (float) qs[2 * r + 0]),
                    __float2half(scale * (float) qs[2 * r + 1]));
            }
        } else {
            #pragma unroll
            for (int r = 0; r < QK8_0 / 2; ++r) {
                dst_h2[r] = make_half2(__float2half(0.0f), __float2half(0.0f));
            }
        }
    }

    // Per-thread output accumulator: 2 key rows (m_a, m_b) x BT tokens. Persists across all
    // head-groups, so after the head-group loop it holds the fully head-reduced dst[m, t].
    float acc[2][BT];
    #pragma unroll
    for (int e = 0; e < 2; ++e)
        #pragma unroll
        for (int ti = 0; ti < BT; ++ti)
            acc[e][ti] = 0.0f;
    __syncthreads();

    for (int h0 = 0; h0 < N_HEAD; h0 += BH) {
        // Stage Q (f32 -> half) and W for this head-group.
        for (int i = tid; i < N_COLS * D; i += 32 * NWARPS) {
            const int col = i / D;
            const int d   = i % D;
            const int ti  = col / BH;
            const int hh  = col % BH;
            const int t   = t0 + ti;
            const int h   = h0 + hh;
            float v = 0.0f;
            if (t < n_tokens) {
                v = q_s[((size_t) h * n_tokens + t) * D + d];
            }
            ((half *) Q_sh)[col * (STRIDE_H2 * 2) + d] = __float2half(v);
        }
        for (int i = tid; i < BT * BH; i += 32 * NWARPS) {
            const int ti = i / BH;
            const int hh = i % BH;
            const int t  = t0 + ti;
            const int h  = h0 + hh;
            w_sh[i] = t < n_tokens ? w_s[(size_t) t * N_HEAD + h] : 0.0f;
        }
        __syncthreads();

        // dot[BM, N_COLS] = K_sh[BM, D] x Q_sh[N_COLS, D]^T, one m16n8k16 per (warp, n-tile, k-tile).
        #pragma unroll
        for (int ti = 0; ti < N_TILES; ++ti) {
            tile<16, 8, float> c_frag;   // x[ne=4] default-zeroed (mma accumulates D += A*B)
            #pragma unroll
            for (int kk = 0; kk < K_TILES; ++kk) {
                tile<16, 8, half2> a_frag;
                tile<8,  8, half2> b_frag;
                load_ldmatrix(a_frag, K_sh + ty * M_PER_WARP * STRIDE_H2 + kk * (16 / 2), STRIDE_H2);
                load_ldmatrix(b_frag, Q_sh + ti * BH * STRIDE_H2 + kk * (16 / 2), STRIDE_H2);
                mma(c_frag, a_frag, b_frag);
            }

            // Epilogue: fold relu(dot)*w into acc, reducing the BH=8 heads for each (m, token).
            const int hh_a = (tx % 4) * 2;
            const int hh_b = hh_a + 1;
            const float wa = w_sh[ti * BH + hh_a];
            const float wb = w_sh[ti * BH + hh_b];
            float pa = fmaxf(c_frag.x[0], 0.0f) * wa + fmaxf(c_frag.x[1], 0.0f) * wb; // m_a
            float pb = fmaxf(c_frag.x[2], 0.0f) * wa + fmaxf(c_frag.x[3], 0.0f) * wb; // m_b
            // Sum the 4 lanes that share tx/4 (= same key row): xor{1,2} gathers all 8 heads.
            pa += __shfl_xor_sync(0xFFFFFFFF, pa, 1);
            pa += __shfl_xor_sync(0xFFFFFFFF, pa, 2);
            pb += __shfl_xor_sync(0xFFFFFFFF, pb, 1);
            pb += __shfl_xor_sync(0xFFFFFFFF, pb, 2);
            acc[0][ti] += pa;  // m_a = tx/4
            acc[1][ti] += pb;  // m_b = 8 + tx/4
        }
        __syncthreads();  // done reading Q_sh before the next head-group overwrites it
    }

    // Store dst[m, t]. tx%4==0 lanes own the reduced result for their 2 key rows.
    if (tx % 4 == 0) {
        const int ma_local = tx / 4;             // [0,8) -> m_local for acc[0]
        #pragma unroll
        for (int ti = 0; ti < BT; ++ti) {
            const int t = t0 + ti;
            if (t < n_tokens) {
                const int m_a = m0 + ty * M_PER_WARP + ma_local;
                if (m_a < n_kv) {
                    if constexpr (DST_F16) {
                        ((half *) d_s)[(size_t) t * n_kv + m_a] = __float2half(acc[0][ti]);
                    } else {
                        ((float *) d_s)[(size_t) t * n_kv + m_a] = acc[0][ti];
                    }
                }
                const int m_b = m0 + ty * M_PER_WARP + 8 + ma_local;
                if (m_b < n_kv) {
                    if constexpr (DST_F16) {
                        ((half *) d_s)[(size_t) t * n_kv + m_b] = __float2half(acc[1][ti]);
                    } else {
                        ((float *) d_s)[(size_t) t * n_kv + m_b] = acc[1][ti];
                    }
                }
            }
        }
    }
}

void ggml_cuda_op_indexer_score(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * k = dst->src[0];
    const ggml_tensor * q = dst->src[1];
    const ggml_tensor * w = dst->src[2];

    GGML_ASSERT(k->type == GGML_TYPE_F32 || k->type == GGML_TYPE_Q8_0);
    GGML_ASSERT(q->type == GGML_TYPE_F32);
    GGML_ASSERT(w->type == GGML_TYPE_F32);
    // F16 dst halves the score-tensor write + downstream top-k read; accumulation stays f32.
    GGML_ASSERT(dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_F16);
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

    // Key-tiling: spread keys across a third grid dim so total blocks ~= a few x n_SM at decode
    // (where n_tokens*n_stream is tiny and the original 2D grid left the GPU idle). Each block
    // always scores exactly INDEXER_KEY_TILE keys of real work; tile count is just n_kv / tile.
    // This is also fine at prefill - more blocks, each doing real work, identical results.
    const int n_key_tiles = (n_kv + INDEXER_KEY_TILE - 1) / INDEXER_KEY_TILE;

    // blockDim == INDEXER_KEY_TILE so a full tile gives each thread exactly one key (the loop
    // still strides by blockDim.x for the trailing partial tile).
    dim3 grid(n_tokens, n_stream, n_key_tiles);
    cudaStream_t stream = ctx.stream();
    const void * kp = (const void *) k->data;
    const float * qp = (const float *) q->data;
    const float * wp = (const float *) w->data;
    void        * dp = dst->data;
    const bool   dst_f16 = dst->type == GGML_TYPE_F16;

    // Prefill tensor-core path: q8_0 K dequantized to half, Q f32->half, f32-accumulating
    // m16n8k16 HMMA, fused relu*w + 64-head reduction. Specialized to the GLM indexer shape
    // (D=128, n_head=64); only worth it at prefill token counts. Decodes and all other shapes
    // fall through to the scalar kernel below.
    constexpr int INDEXER_TC_MIN_TOKENS = 16;
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    if (k->type == GGML_TYPE_Q8_0 && D == 128 && n_head == 64 &&
        n_tokens >= INDEXER_TC_MIN_TOKENS && turing_mma_available(cc)) {
        constexpr int BM = 64;  // keys/CTA
        constexpr int BT = 8;   // tokens/CTA
        constexpr int BH = 8;   // heads/head-group
        constexpr int NWARPS = BM / 16;
        constexpr int STRIDE_H2 = 128 / 2 + 4;  // mirror kernel's STRIDE_H2 for D=128
        // smem: K tile + Q tile (one head-group) + W tile.
        const size_t smem_tc = ((size_t) BM * STRIDE_H2 * sizeof(half2))
                             + ((size_t) (BT * BH) * STRIDE_H2 * sizeof(half2))
                             + ((size_t) BT * BH * sizeof(float));
        const dim3 grid((n_kv + BM - 1) / BM, (n_tokens + BT - 1) / BT, n_stream);
        const dim3 block(32, NWARPS);
        if (dst_f16) {
            indexer_score_hmma_prefill_q8_64<BM, BT, BH, 128, 64, true>
                <<<grid, block, smem_tc, stream>>>(kp, qp, wp, dp, n_kv, n_tokens, k_row, k_stream);
        } else {
            indexer_score_hmma_prefill_q8_64<BM, BT, BH, 128, 64, false>
                <<<grid, block, smem_tc, stream>>>(kp, qp, wp, dp, n_kv, n_tokens, k_row, k_stream);
        }
        return;
    }

    if (k->type == GGML_TYPE_Q8_0) {
        switch (n_head) {
            case 16: if (dst_f16) indexer_score<16, true, true ><<<grid, INDEXER_KEY_TILE, smem, stream>>>(kp, qp, wp, dp, D, n_kv, n_tokens, k_row, k_stream);
                     else         indexer_score<16, true, false><<<grid, INDEXER_KEY_TILE, smem, stream>>>(kp, qp, wp, dp, D, n_kv, n_tokens, k_row, k_stream); break;
            case 32: if (dst_f16) indexer_score<32, true, true ><<<grid, INDEXER_KEY_TILE, smem, stream>>>(kp, qp, wp, dp, D, n_kv, n_tokens, k_row, k_stream);
                     else         indexer_score<32, true, false><<<grid, INDEXER_KEY_TILE, smem, stream>>>(kp, qp, wp, dp, D, n_kv, n_tokens, k_row, k_stream); break;
            case 64: if (dst_f16) indexer_score<64, true, true ><<<grid, INDEXER_KEY_TILE, smem, stream>>>(kp, qp, wp, dp, D, n_kv, n_tokens, k_row, k_stream);
                     else         indexer_score<64, true, false><<<grid, INDEXER_KEY_TILE, smem, stream>>>(kp, qp, wp, dp, D, n_kv, n_tokens, k_row, k_stream); break;
            default: if (dst_f16) indexer_score_generic<true, true ><<<grid, INDEXER_KEY_TILE, smem, stream>>>(kp, qp, wp, dp, D, n_kv, n_tokens, n_head, k_row, k_stream);
                     else         indexer_score_generic<true, false><<<grid, INDEXER_KEY_TILE, smem, stream>>>(kp, qp, wp, dp, D, n_kv, n_tokens, n_head, k_row, k_stream); break;
        }
    } else {
        switch (n_head) {
            case 16: if (dst_f16) indexer_score<16, false, true ><<<grid, INDEXER_KEY_TILE, smem, stream>>>(kp, qp, wp, dp, D, n_kv, n_tokens, k_row, k_stream);
                     else         indexer_score<16, false, false><<<grid, INDEXER_KEY_TILE, smem, stream>>>(kp, qp, wp, dp, D, n_kv, n_tokens, k_row, k_stream); break;
            case 32: if (dst_f16) indexer_score<32, false, true ><<<grid, INDEXER_KEY_TILE, smem, stream>>>(kp, qp, wp, dp, D, n_kv, n_tokens, k_row, k_stream);
                     else         indexer_score<32, false, false><<<grid, INDEXER_KEY_TILE, smem, stream>>>(kp, qp, wp, dp, D, n_kv, n_tokens, k_row, k_stream); break;
            case 64: if (dst_f16) indexer_score<64, false, true ><<<grid, INDEXER_KEY_TILE, smem, stream>>>(kp, qp, wp, dp, D, n_kv, n_tokens, k_row, k_stream);
                     else         indexer_score<64, false, false><<<grid, INDEXER_KEY_TILE, smem, stream>>>(kp, qp, wp, dp, D, n_kv, n_tokens, k_row, k_stream); break;
            default: if (dst_f16) indexer_score_generic<false, true ><<<grid, INDEXER_KEY_TILE, smem, stream>>>(kp, qp, wp, dp, D, n_kv, n_tokens, n_head, k_row, k_stream);
                     else         indexer_score_generic<false, false><<<grid, INDEXER_KEY_TILE, smem, stream>>>(kp, qp, wp, dp, D, n_kv, n_tokens, n_head, k_row, k_stream); break;
        }
    }
}
