#include "common.cuh"

// heads processed per block (one warp each). 16 warps = 512 threads.
#define SMLA_HTILE 16

// q8_0 non-split prefill uses cp.async to double-buffer the scattered latent-row gather
// (the latency-bound regime on the GB10). Two raw q8_0 staging buffers live in smem after
// q_sh/lat_sh. Row bytes are (d_lat/QK8_0)*sizeof(block_q8_0) (612 for d_lat=576); the
// stride between the two buffers is padded to 16 so the cp.async dst stays aligned.
static inline size_t ggml_cuda_sparse_mla_attn_smem(bool k_q8_0, int64_t d_lat) {
    size_t smem = (size_t)(SMLA_HTILE + 1) * d_lat * sizeof(float);
    if (k_q8_0) {
        const size_t row_bytes = (size_t)(d_lat / QK8_0) * sizeof(block_q8_0);
        const size_t row_pad   = (row_bytes + 15) & ~(size_t)15;
        smem += 2 * row_pad + 16;
    }
    return smem;
}

void ggml_cuda_op_sparse_mla_attn(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
