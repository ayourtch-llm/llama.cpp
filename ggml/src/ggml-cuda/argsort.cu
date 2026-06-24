#include "argsort.cuh"

#include <cuda_fp16.h>

#ifdef GGML_CUDA_USE_CUB
#    include <cub/cub.cuh>
#    if (CCCL_MAJOR_VERSION >= 3 && CCCL_MINOR_VERSION >= 1)
#        define STRIDED_ITERATOR_AVAILABLE
#        include <cuda/iterator>
#    endif
using namespace cub;
#endif  // GGML_CUDA_USE_CUB

static __global__ void init_indices(int * indices, const int ncols, const int nrows) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y;

    if (col < ncols && row < nrows) {
        indices[row * ncols + col] = col;
    }
}

#ifndef STRIDED_ITERATOR_AVAILABLE
static __global__ void init_offsets(int * offsets, const int ncols, const int nrows) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx <= nrows) {
        offsets[idx] = idx * ncols;
    }
}
#endif  // STRIDED_ITERATOR_AVAILABLE

#ifdef GGML_CUDA_USE_CUB
void argsort_f32_i32_cuda_cub(ggml_cuda_pool & pool,
                              const float *    x,
                              int *            dst,
                              const int        ncols,
                              const int        nrows,
                              ggml_sort_order  order,
                              cudaStream_t     stream) {
    ggml_cuda_pool_alloc<int>   temp_indices_alloc(pool, ncols * nrows);
    ggml_cuda_pool_alloc<float> temp_keys_alloc(pool, ncols * nrows);

    int *   temp_indices = temp_indices_alloc.get();
    float * temp_keys    = temp_keys_alloc.get();

    static const int block_size = 256;
    const dim3 grid_size((ncols + block_size - 1) / block_size, nrows);
    init_indices<<<grid_size, block_size, 0, stream>>>(temp_indices, ncols, nrows);

#ifdef STRIDED_ITERATOR_AVAILABLE
    auto offset_iterator = cuda::make_strided_iterator(cuda::make_counting_iterator(0), ncols);
#else
    // offset_iterator needs to populate nrows + 1 elements, so we also have to ceildiv nrows + 1 by block_size
    const int                 nrows_offset = nrows + 1;
    ggml_cuda_pool_alloc<int> offsets_alloc(pool, nrows_offset);
    int *                     offset_iterator = offsets_alloc.get();
    const dim3                offset_grid((nrows_offset + block_size - 1) / block_size);
    init_offsets<<<offset_grid, block_size, 0, stream>>>(offset_iterator, ncols, nrows);
#endif
    CUDA_CHECK(cudaMemcpyAsync(temp_keys, x, ncols * nrows * sizeof(float), cudaMemcpyDeviceToDevice, stream));

    size_t temp_storage_bytes = 0;

    bool is_capturing = false;
#ifdef USE_CUDA_GRAPH
    // Currently (confirmed for CCCL <= 3.2) DeviceSegmentedSort does not support stream capture, while DeviceSegmentedRadixSort does.
    // See https://github.com/NVIDIA/cccl/issues/5661#issuecomment-3229037149
    // TODO: constrain this to the CCCL versions that have this issue once it's resolved in a future CCCL release.
    cudaStreamCaptureStatus capture_status;
    CUDA_CHECK(cudaStreamIsCapturing(stream, &capture_status));
    is_capturing = (capture_status != cudaStreamCaptureStatusNone);
#endif  // USE_CUDA_GRAPH

    if (order == GGML_SORT_ORDER_ASC) {
        if (nrows == 1) {
            CUDA_CHECK(DeviceRadixSort::SortPairs(nullptr, temp_storage_bytes, temp_keys, temp_keys,  // keys (in-place)
                                                  temp_indices, dst,  // values (indices)
                                                  ncols, 0, sizeof(float) * 8, stream));
        } else if (is_capturing) {
            CUDA_CHECK(DeviceSegmentedRadixSort::SortPairs(
                nullptr, temp_storage_bytes, temp_keys, temp_keys,  // keys (in-place)
                temp_indices, dst,                                  // values (indices)
                ncols * nrows, nrows,                               // num items, num segments
                offset_iterator, offset_iterator + 1, 0, sizeof(float) * 8, stream));
        } else {
            CUDA_CHECK(DeviceSegmentedSort::SortPairs(nullptr, temp_storage_bytes, temp_keys,
                                                      temp_keys,             // keys (in-place)
                                                      temp_indices, dst,     // values (indices)
                                                      ncols * nrows, nrows,  // num items, num segments
                                                      offset_iterator, offset_iterator + 1, stream));
        }
    } else {
        if (nrows == 1) {
            CUDA_CHECK(DeviceRadixSort::SortPairsDescending(nullptr, temp_storage_bytes, temp_keys,
                                                            temp_keys,          // keys (in-place)
                                                            temp_indices, dst,  // values (indices)
                                                            ncols, 0, sizeof(float) * 8, stream));
        } else if (is_capturing) {
            CUDA_CHECK(DeviceSegmentedRadixSort::SortPairsDescending(
                nullptr, temp_storage_bytes, temp_keys, temp_keys, temp_indices, dst, ncols * nrows, nrows,
                offset_iterator, offset_iterator + 1, 0, sizeof(float) * 8, stream));
        } else {
            CUDA_CHECK(DeviceSegmentedSort::SortPairsDescending(nullptr, temp_storage_bytes, temp_keys, temp_keys,
                                                                temp_indices, dst, ncols * nrows, nrows,
                                                                offset_iterator, offset_iterator + 1, stream));
        }
    }

    ggml_cuda_pool_alloc<uint8_t> temp_storage_alloc(pool, temp_storage_bytes);
    void *                        d_temp_storage = temp_storage_alloc.get();

    if (order == GGML_SORT_ORDER_ASC) {
        if (nrows == 1) {
            CUDA_CHECK(DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, temp_keys,
                                                  temp_keys,          // keys (in-place)
                                                  temp_indices, dst,  // values (indices)
                                                  ncols, 0, sizeof(float) * 8, stream));
        } else if (is_capturing) {
            CUDA_CHECK(DeviceSegmentedRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, temp_keys, temp_keys,
                                                           temp_indices, dst, ncols * nrows, nrows, offset_iterator,
                                                           offset_iterator + 1, 0, sizeof(float) * 8, stream));
        } else {
            CUDA_CHECK(DeviceSegmentedSort::SortPairs(d_temp_storage, temp_storage_bytes, temp_keys, temp_keys,
                                                      temp_indices, dst, ncols * nrows, nrows, offset_iterator,
                                                      offset_iterator + 1, stream));
        }
    } else {
        if (nrows == 1) {
            CUDA_CHECK(DeviceRadixSort::SortPairsDescending(d_temp_storage, temp_storage_bytes, temp_keys,
                                                            temp_keys,          // keys (in-place)
                                                            temp_indices, dst,  // values (indices)
                                                            ncols, 0, sizeof(float) * 8, stream));
        } else if (is_capturing) {
            CUDA_CHECK(DeviceSegmentedRadixSort::SortPairsDescending(
                d_temp_storage, temp_storage_bytes, temp_keys, temp_keys, temp_indices, dst, ncols * nrows, nrows,
                offset_iterator, offset_iterator + 1, 0, sizeof(float) * 8, stream));
        } else {
            CUDA_CHECK(DeviceSegmentedSort::SortPairsDescending(d_temp_storage, temp_storage_bytes, temp_keys,
                                                                temp_keys, temp_indices, dst, ncols * nrows, nrows,
                                                                offset_iterator, offset_iterator + 1, stream));
        }
    }
}
#endif  // GGML_CUDA_USE_CUB

// F16-keyed CUB radix sort. CUB has no radix specialization for __half, and the comparison-based
// DeviceSegmentedSort with __half keys is broken at scale (wrong results + illegal memory access
// at large nrows). The fix is the standard float-radix bit-trick: reinterpret the half bits as an
// ORDER-PRESERVING uint16 and radix-sort that. This makes the problem identical to the f32 path
// (which radix-sorts float via the same trick CUB applies internally) but with 16-bit keys.
//
// Transform (monotonic over the whole half range, incl. -inf, -0/+0, +inf):
//   ord = h_bits ^ ((h_bits >> 15) ? 0xFFFF : 0x8000)
// flips all bits for negatives (so -inf -> 0x03ff, the smallest) and flips only the sign bit for
// positives (so +0 -> 0x8000). Ascending uint16 order then == ascending half order; a descending
// sort of the u16 keys yields the half-descending order top-k needs.
#ifdef GGML_CUDA_USE_CUB
// half bits -> order-preserving uint16 (one element per thread; half is read, u16 is written).
static __global__ void half_to_order_u16(uint16_t * __restrict__ out, const half * __restrict__ in, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        uint16_t h_bits;
        memcpy(&h_bits, in + i, sizeof(h_bits));
        out[i] = h_bits ^ ((h_bits >> 15) ? 0xFFFF : 0x8000);
    }
}

void argsort_f16_i32_cuda_cub(ggml_cuda_pool & pool,
                              const half *     x,
                              int *            dst,
                              const int        ncols,
                              const int        nrows,
                              ggml_sort_order  order,
                              cudaStream_t     stream) {
    // Per-row temp: keys_in (uint16), indices. keys_out aliases keys_in (CUB DeviceRadixSort
    // supports in-place key output). For multi-row, loop over rows calling the single-row
    // DeviceRadixSort (same proven pattern top_k_cub uses in top-k.cu). This avoids
    // DeviceSegmentedSort/DeviceSegmentedRadixSort on __half (broken at scale) and the
    // segmented-radix path's graph-capture instability on CUB 2.8.2.
    ggml_cuda_pool_alloc<int>      temp_indices_alloc(pool, ncols);
    ggml_cuda_pool_alloc<uint16_t> temp_keys_alloc(pool, ncols);
    ggml_cuda_pool_alloc<int>      temp_dst_alloc(pool, ncols);

    int *      temp_indices = temp_indices_alloc.get();
    uint16_t * temp_keys    = temp_keys_alloc.get();
    int *      temp_dst     = temp_dst_alloc.get();

    static const int block_size = 256;

    for (int r = 0; r < nrows; r++) {
        const half * x_row = x + (size_t) r * ncols;
        int * dst_row = dst + (size_t) r * ncols;

        init_indices<<<(ncols + block_size - 1) / block_size, block_size, 0, stream>>>(temp_indices, ncols, 1);

        // Stage this row's half keys as order-preserving uint16 (no f32 materialization).
        // Ascending u16 order == ascending half order, so a radix sort on these u16 keys
        // ranks the f16 scores exactly.
        half_to_order_u16<<<(ncols + block_size - 1) / block_size, block_size, 0, stream>>>(
            temp_keys, x_row, ncols);

        size_t temp_storage_bytes = 0;

        if (order == GGML_SORT_ORDER_ASC) {
            CUDA_CHECK(DeviceRadixSort::SortPairs(nullptr, temp_storage_bytes, temp_keys, temp_keys,
                                                  temp_indices, temp_dst, ncols, 0, sizeof(uint16_t) * 8, stream));
        } else {
            CUDA_CHECK(DeviceRadixSort::SortPairsDescending(nullptr, temp_storage_bytes, temp_keys, temp_keys,
                                                            temp_indices, temp_dst, ncols, 0, sizeof(uint16_t) * 8, stream));
        }

        ggml_cuda_pool_alloc<uint8_t> temp_storage_alloc(pool, temp_storage_bytes);
        void *                        d_temp_storage = temp_storage_alloc.get();

        if (order == GGML_SORT_ORDER_ASC) {
            CUDA_CHECK(DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, temp_keys, temp_keys,
                                                  temp_indices, temp_dst, ncols, 0, sizeof(uint16_t) * 8, stream));
        } else {
            CUDA_CHECK(DeviceRadixSort::SortPairsDescending(d_temp_storage, temp_storage_bytes, temp_keys, temp_keys,
                                                            temp_indices, temp_dst, ncols, 0, sizeof(uint16_t) * 8, stream));
        }

        CUDA_CHECK(cudaMemcpyAsync(dst_row, temp_dst, ncols * sizeof(int), cudaMemcpyDeviceToDevice, stream));
    }
}
#endif  // GGML_CUDA_USE_CUB

// Bitonic sort implementation
template<typename T>
static inline __device__ void ggml_cuda_swap(T & a, T & b) {
    T tmp = a;
    a = b;
    b = tmp;
}

template<ggml_sort_order order, typename key_t>
static __global__ void k_argsort_i32(const key_t * x, int * dst, const int ncols, int ncols_pad) {
    // bitonic sort
    int col = threadIdx.x;
    int row = blockIdx.x;

    if (col >= ncols_pad) {
        return;
    }

    const key_t * x_row = x + row * ncols;
    extern __shared__ int dst_row[];

    // initialize indices
    dst_row[col] = col;

    __syncthreads();

    for (int k = 2; k <= ncols_pad; k *= 2) {
        for (int j = k / 2; j > 0; j /= 2) {
            int ixj = col ^ j;
            if (ixj > col) {
                if ((col & k) == 0) {
                    if (dst_row[col] >= ncols ||
                        (dst_row[ixj] < ncols && (order == GGML_SORT_ORDER_ASC ?
                            x_row[dst_row[col]] > x_row[dst_row[ixj]] :
                            x_row[dst_row[col]] < x_row[dst_row[ixj]]))
                    ) {
                        ggml_cuda_swap(dst_row[col], dst_row[ixj]);
                    }
                } else {
                    if (dst_row[ixj] >= ncols ||
                        (dst_row[col] < ncols && (order == GGML_SORT_ORDER_ASC ?
                            x_row[dst_row[col]] < x_row[dst_row[ixj]] :
                            x_row[dst_row[col]] > x_row[dst_row[ixj]]))
                    ) {
                        ggml_cuda_swap(dst_row[col], dst_row[ixj]);
                    }
                }
            }
            __syncthreads();
        }
    }

    // copy the result to dst without the padding
    if (col < ncols) {
        dst[row * ncols + col] = dst_row[col];
    }
}

static int next_power_of_2(int x) {
    int n = 1;
    while (n < x) {
        n *= 2;
    }
    return n;
}

void argsort_f32_i32_cuda_bitonic(const float *   x,
                                  int *           dst,
                                  const int       ncols,
                                  const int       nrows,
                                  ggml_sort_order order,
                                  cudaStream_t    stream) {
    // bitonic sort requires ncols to be power of 2
    const int ncols_pad = next_power_of_2(ncols);

    const dim3 block_dims(ncols_pad, 1, 1);
    const dim3 block_nums(nrows, 1, 1);
    const size_t shared_mem = ncols_pad * sizeof(int);

    // FIXME: this limit could be raised by ~2-4x on Ampere or newer
    GGML_ASSERT(shared_mem <= ggml_cuda_info().devices[ggml_cuda_get_device()].smpb);

    if (order == GGML_SORT_ORDER_ASC) {
        k_argsort_i32<GGML_SORT_ORDER_ASC, float>
            <<<block_nums, block_dims, shared_mem, stream>>>(x, dst, ncols, ncols_pad);
    } else if (order == GGML_SORT_ORDER_DESC) {
        k_argsort_i32<GGML_SORT_ORDER_DESC, float>
            <<<block_nums, block_dims, shared_mem, stream>>>(x, dst, ncols, ncols_pad);
    } else {
        GGML_ABORT("fatal error");
    }
}

// F16-keyed bitonic. Keys are gathered from global as half per comparison (no f32 staging), and
// compared with half's native operator< (the indexer's f16-rounded scores are what we want to
// rank anyway). Same ncols<=1024 shared-mem/thread limit as the f32 path; larger ncols is served
// by the CUB segmented-sort path in top-k.cu.
void argsort_f16_i32_cuda_bitonic(const half *     x,
                                  int *            dst,
                                  const int        ncols,
                                  const int        nrows,
                                  ggml_sort_order  order,
                                  cudaStream_t     stream) {
    const int ncols_pad = next_power_of_2(ncols);

    const dim3 block_dims(ncols_pad, 1, 1);
    const dim3 block_nums(nrows, 1, 1);
    const size_t shared_mem = ncols_pad * sizeof(int);

    GGML_ASSERT(shared_mem <= ggml_cuda_info().devices[ggml_cuda_get_device()].smpb);

    if (order == GGML_SORT_ORDER_ASC) {
        k_argsort_i32<GGML_SORT_ORDER_ASC, half>
            <<<block_nums, block_dims, shared_mem, stream>>>(x, dst, ncols, ncols_pad);
    } else if (order == GGML_SORT_ORDER_DESC) {
        k_argsort_i32<GGML_SORT_ORDER_DESC, half>
            <<<block_nums, block_dims, shared_mem, stream>>>(x, dst, ncols, ncols_pad);
    } else {
        GGML_ABORT("fatal error");
    }
}

void ggml_cuda_op_argsort(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_contiguous(src0));

    const int64_t ncols = src0->ne[0];
    const int64_t nrows = ggml_nrows(src0);

    enum ggml_sort_order order = (enum ggml_sort_order) dst->op_params[0];

#ifdef GGML_CUDA_USE_CUB
    const int    ncols_pad      = next_power_of_2(ncols);
    const size_t shared_mem     = ncols_pad * sizeof(int);
    const size_t max_shared_mem = ggml_cuda_info().devices[ggml_cuda_get_device()].smpb;

    if (shared_mem > max_shared_mem || ncols > 1024) {
        ggml_cuda_pool & pool = ctx.pool();
        argsort_f32_i32_cuda_cub(pool, src0_d, (int *) dst_d, ncols, nrows, order, stream);
    } else {
        argsort_f32_i32_cuda_bitonic(src0_d, (int *) dst_d, ncols, nrows, order, stream);
    }
#else
    argsort_f32_i32_cuda_bitonic(src0_d, (int *) dst_d, ncols, nrows, order, stream);
#endif
}
