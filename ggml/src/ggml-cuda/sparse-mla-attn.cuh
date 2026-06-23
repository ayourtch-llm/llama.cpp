#include "common.cuh"

// heads processed per block (one warp each). 16 warps = 512 threads.
#define SMLA_HTILE 16

void ggml_cuda_op_sparse_mla_attn(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
