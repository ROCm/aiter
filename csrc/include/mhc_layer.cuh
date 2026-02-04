#pragma once
// SPDX-License-Identifier: MIT
// Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.

#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hipblaslt/hipblaslt.h>
#include <hip/hip_cooperative_groups.h>
#include "aiter_hip_common.h"

namespace cooperative_groups {
template <typename T>
struct plus {
    __device__ __forceinline__ T operator()(T a, T b) const { return a + b; }
};

template <typename T, typename Op>
__device__ __forceinline__ T reduce(::cooperative_groups::thread_block_tile<32> tile, T val, Op op)
{
    for (int offset = tile.size() / 2; offset > 0; offset /= 2) {
        T other = __shfl_down(val, offset);
        val = op(val, other);
    }
    return val;
}
}  // namespace cooperative_groups

namespace mhc {

struct MHCConfig {
    int sinkhorn_iters;
    int nC;
    float eps;
    bool use_pdl;
};

struct RMSNormParams {
    int n;
    float eps;
};

template<int BLOCK_SIZE>
__global__ void float_to_bf16_kernel(__hip_bfloat16* __restrict__ out,
                                     const float* __restrict__ inp,
                                     int size) {
    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (idx < size) {
        out[idx] = (__hip_bfloat16)inp[idx];
    }
}

template<int BLOCK_SIZE>
__global__ void bf16_to_float_kernel(float* __restrict__ out,
                                     const __hip_bfloat16* __restrict__ inp,
                                     int size) {
    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (idx < size) {
        out[idx] = (float)inp[idx];
    }
}

inline void float_to_bf16(__hip_bfloat16* out, const float* inp, int size,
                          hipStream_t stream = nullptr) {
    constexpr int BLOCK_SIZE = 256;
    int num_blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    float_to_bf16_kernel<BLOCK_SIZE><<<num_blocks, BLOCK_SIZE, 0, stream>>>(out, inp, size);
}

inline void bf16_to_float(float* out, const __hip_bfloat16* inp, int size,
                          hipStream_t stream = nullptr) {
    constexpr int BLOCK_SIZE = 256;
    int num_blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    bf16_to_float_kernel<BLOCK_SIZE><<<num_blocks, BLOCK_SIZE, 0, stream>>>(out, inp, size);
}

__device__ __forceinline__ float fast_exp(float x) {
    return __expf(x);
}

__device__ __forceinline__ float fast_sigmoid(float x) {
    return 1.0f / (1.0f + fast_exp(-x));
}

__device__ __forceinline__ __hip_bfloat162 mhc_floats2bfloat162(float x, float y) {
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__) || defined(__HIP_DEVICE_COMPILE__)
    __hip_bfloat162 out;
    out.x = __float2bfloat16(x);
    out.y = __float2bfloat16(y);
    return out;
#else
    return mhc_floats2bfloat162(x, y);
#endif
}

} // namespace mhc


namespace aiter {
void mhc_layer_fwd(torch::Tensor &out,
                   torch::Tensor &x_expanded,
                   torch::Tensor &rmsnorm_weight,
                   torch::Tensor &phi_pre,
                   torch::Tensor &phi_post,
                   torch::Tensor &phi_res,
                   torch::Tensor &b_pre,
                   torch::Tensor &b_post,
                   torch::Tensor &b_res,
                   double alpha_pre,
                   double alpha_post,
                   double alpha_res,
                   int64_t sinkhorn_iters,
                   double eps,
                   bool use_pdl);

void mhc_layer_fwd_debug(torch::Tensor &out,
                         torch::Tensor &x_expanded,
                         torch::Tensor &rmsnorm_weight,
                         torch::Tensor &phi_pre,
                         torch::Tensor &phi_post,
                         torch::Tensor &phi_res,
                         torch::Tensor &b_pre,
                         torch::Tensor &b_post,
                         torch::Tensor &b_res,
                         double alpha_pre,
                         double alpha_post,
                         double alpha_res,
                         int64_t sinkhorn_iters,
                         double eps,
                         torch::Tensor &H_proj_raw,
                         torch::Tensor &H_pre,
                         torch::Tensor &H_post,
                         torch::Tensor &M,
                         torch::Tensor &x_agg_bf16,
                         torch::Tensor &layer_out_bf16,
                         torch::Tensor &rms_values,
                         bool use_pdl);

} // namespace aiter

namespace cg = cooperative_groups;

namespace mhc {


template<int N_COMPILE, int BLOCK_SIZE>
__global__ void sinkhorn_knopp_warp_optimized_kernel(float* __restrict__ out,
                                                     const float* __restrict__ inp, int M, int N,
                                                     int num_iters, float eps) {
    constexpr int WARPS_PER_BLOCK = BLOCK_SIZE / 32;

    extern __shared__ float smem[];
    float* tile = smem;
    float* col_sums = smem + M * N_COMPILE;

    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);

    const int warp_id = threadIdx.x / 32;
    const int lane_id = warp.thread_rank();

    int total_elems = M * N;
    for (int i = threadIdx.x; i < total_elems; i += BLOCK_SIZE) {
        tile[i] = inp[i];
    }
    block.sync();

    for (int iter = 0; iter < num_iters; iter++) {
        for (int r = warp_id; r < M; r += WARPS_PER_BLOCK) {
            float val = (lane_id < N) ? tile[r * N + lane_id] : 0.0f;
            float row_sum = cg::reduce(warp, val, cg::plus<float>());

            if (lane_id < N && row_sum > eps) {
                tile[r * N + lane_id] = val * __frcp_rn(row_sum);
            }
        }
        block.sync();

        for (int c = threadIdx.x; c < N; c += BLOCK_SIZE) {
            float sum = 0.0f;
#pragma unroll 4
            for (int r = 0; r < M; r++) {
                sum += tile[r * N + c];
            }
            col_sums[c] = (sum > eps) ? __frcp_rn(sum) : 0.0f;
        }
        block.sync();

        for (int i = threadIdx.x; i < total_elems; i += BLOCK_SIZE) {
            int c = i % N;
            tile[i] *= col_sums[c];
        }
        block.sync();
    }

    for (int i = threadIdx.x; i < total_elems; i += BLOCK_SIZE) {
        out[i] = tile[i];
    }
}

template<int BLOCK_SIZE>
__global__ void sinkhorn_knopp_warp_per_row_32x32_kernel(float* __restrict__ out,
                                                         const float* __restrict__ inp,
                                                         int num_iters, float eps) {
    constexpr int N = 32;
    constexpr int WARPS = BLOCK_SIZE / 32;
    constexpr int ROWS_PER_WARP = (N + WARPS - 1) / WARPS;

    __shared__ float tile[N * (N + 1)];
    __shared__ float col_sums[N];

    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);

    const int warp_id = threadIdx.x / 32;
    const int lane_id = warp.thread_rank();
    const int stride = N + 1;

    for (int i = threadIdx.x; i < N * N; i += BLOCK_SIZE) {
        int r = i / N;
        int c = i % N;
        tile[r * stride + c] = inp[i];
    }
    block.sync();

    for (int iter = 0; iter < num_iters; iter++) {
#pragma unroll 4
        for (int rr = 0; rr < ROWS_PER_WARP; rr++) {
            int r = warp_id * ROWS_PER_WARP + rr;
            if (r < N) {
                float val = tile[r * stride + lane_id];
                float sum = cg::reduce(warp, val, cg::plus<float>());

                if (sum > eps) {
                    tile[r * stride + lane_id] = val * __frcp_rn(sum);
                }
            }
        }
        block.sync();

        if (threadIdx.x < N) {
            int c = threadIdx.x;
            float sum = 0.0f;
#pragma unroll 8
            for (int r = 0; r < N; r++) {
                sum += tile[r * stride + c];
            }
            col_sums[c] = (sum > eps) ? __frcp_rn(sum) : 0.0f;
        }
        block.sync();

        for (int i = threadIdx.x; i < N * N; i += BLOCK_SIZE) {
            int r = i / N;
            int c = i % N;
            tile[r * stride + c] *= col_sums[c];
        }
        block.sync();
    }

    for (int i = threadIdx.x; i < N * N; i += BLOCK_SIZE) {
        int r = i / N;
        int c = i % N;
        out[i] = tile[r * stride + c];
    }
}

template<int TILE_M, int TILE_N, int BLOCK_SIZE>
__global__ void sinkhorn_knopp_kernel(float* __restrict__ out, const float* __restrict__ inp, int M,
                                      int N, int num_iters, float eps) {
    extern __shared__ float smem[];
    float* tile = smem;
    float* row_sums = smem + TILE_M * TILE_N;
    float* col_sums = row_sums + TILE_M;

    int tile_row = blockIdx.y * TILE_M;
    int tile_col = blockIdx.x * TILE_N;

    int rows_in_tile = min(TILE_M, M - tile_row);
    int cols_in_tile = min(TILE_N, N - tile_col);

    for (int i = threadIdx.x; i < TILE_M * TILE_N; i += BLOCK_SIZE) {
        int local_r = i / TILE_N;
        int local_c = i % TILE_N;
        int global_r = tile_row + local_r;
        int global_c = tile_col + local_c;

        if (global_r < M && global_c < N) {
            tile[i] = inp[global_r * N + global_c];
        } else {
            tile[i] = 0.0f;
        }
    }
    __syncthreads();

    for (int iter = 0; iter < num_iters; iter++) {
        for (int r = threadIdx.x; r < TILE_M; r += BLOCK_SIZE) {
            float sum = 0.0f;
#pragma unroll 4
            for (int c = 0; c < TILE_N; c++) {
                sum += tile[r * TILE_N + c];
            }
            row_sums[r] = sum;
        }
        __syncthreads();

        for (int i = threadIdx.x; i < TILE_M * TILE_N; i += BLOCK_SIZE) {
            int r = i / TILE_N;
            float row_sum = row_sums[r];
            if (row_sum > eps) {
                tile[i] /= row_sum;
            }
        }
        __syncthreads();

        for (int c = threadIdx.x; c < TILE_N; c += BLOCK_SIZE) {
            float sum = 0.0f;
#pragma unroll 4
            for (int r = 0; r < TILE_M; r++) {
                sum += tile[r * TILE_N + c];
            }
            col_sums[c] = sum;
        }
        __syncthreads();

        for (int i = threadIdx.x; i < TILE_M * TILE_N; i += BLOCK_SIZE) {
            int c = i % TILE_N;
            float col_sum = col_sums[c];
            if (col_sum > eps) {
                tile[i] /= col_sum;
            }
        }
        __syncthreads();
    }

    for (int i = threadIdx.x; i < TILE_M * TILE_N; i += BLOCK_SIZE) {
        int local_r = i / TILE_N;
        int local_c = i % TILE_N;
        int global_r = tile_row + local_r;
        int global_c = tile_col + local_c;

        if (global_r < M && global_c < N) {
            out[global_r * N + global_c] = tile[i];
        }
    }
}

template<int MAX_DIM, int BLOCK_SIZE>
__global__ void sinkhorn_knopp_single_block_kernel(float* __restrict__ out,
                                                   const float* __restrict__ inp, int M, int N,
                                                   int num_iters, float eps) {
    extern __shared__ float smem[];
    float* tile = smem;
    float* row_sums = smem + MAX_DIM * MAX_DIM;
    float* col_sums = row_sums + MAX_DIM;

    int total_elems = M * N;

    for (int i = threadIdx.x; i < total_elems; i += BLOCK_SIZE) {
        tile[i] = inp[i];
    }
    __syncthreads();

    for (int iter = 0; iter < num_iters; iter++) {
        for (int r = threadIdx.x; r < M; r += BLOCK_SIZE) {
            float sum = 0.0f;
            for (int c = 0; c < N; c++) {
                sum += tile[r * N + c];
            }
            row_sums[r] = sum;
        }
        __syncthreads();

        for (int i = threadIdx.x; i < total_elems; i += BLOCK_SIZE) {
            int r = i / N;
            float row_sum = row_sums[r];
            if (row_sum > eps) {
                tile[i] /= row_sum;
            }
        }
        __syncthreads();

        for (int c = threadIdx.x; c < N; c += BLOCK_SIZE) {
            float sum = 0.0f;
            for (int r = 0; r < M; r++) {
                sum += tile[r * N + c];
            }
            col_sums[c] = sum;
        }
        __syncthreads();

        for (int i = threadIdx.x; i < total_elems; i += BLOCK_SIZE) {
            int c = i % N;
            float col_sum = col_sums[c];
            if (col_sum > eps) {
                tile[i] /= col_sum;
            }
        }
        __syncthreads();
    }

    for (int i = threadIdx.x; i < total_elems; i += BLOCK_SIZE) {
        out[i] = tile[i];
    }
}

template<int MAX_DIM, int BLOCK_SIZE>
__global__ void sinkhorn_knopp_single_block_fused_exp_kernel(float* __restrict__ out,
                                                             float* __restrict__ H_res_exp,
                                                             const float* __restrict__ inp, int M,
                                                             int N, int num_iters, float eps) {
    extern __shared__ float smem[];
    float* tile = smem;
    float* row_sums = smem + MAX_DIM * MAX_DIM;
    float* col_sums = row_sums + MAX_DIM;

    int total_elems = M * N;

    for (int i = threadIdx.x; i < total_elems; i += BLOCK_SIZE) {
        float val = fast_exp(inp[i]);
        tile[i] = val;
        if (H_res_exp)
            H_res_exp[i] = val;
    }
    __syncthreads();

    for (int iter = 0; iter < num_iters; iter++) {
        for (int r = threadIdx.x; r < M; r += BLOCK_SIZE) {
            float sum = 0.0f;
            for (int c = 0; c < N; c++) {
                sum += tile[r * N + c];
            }
            row_sums[r] = sum;
        }
        __syncthreads();

        for (int i = threadIdx.x; i < total_elems; i += BLOCK_SIZE) {
            int r = i / N;
            float row_sum = row_sums[r];
            if (row_sum > eps) {
                tile[i] *= __frcp_rn(row_sum);
            }
        }
        __syncthreads();

        for (int c = threadIdx.x; c < N; c += BLOCK_SIZE) {
            float sum = 0.0f;
            for (int r = 0; r < M; r++) {
                sum += tile[r * N + c];
            }
            col_sums[c] = sum;
        }
        __syncthreads();

        for (int i = threadIdx.x; i < total_elems; i += BLOCK_SIZE) {
            int c = i % N;
            float col_sum = col_sums[c];
            if (col_sum > eps) {
                tile[i] *= __frcp_rn(col_sum);
            }
        }
        __syncthreads();
    }

    for (int i = threadIdx.x; i < total_elems; i += BLOCK_SIZE) {
        out[i] = tile[i];
    }
}

inline void sinkhorn_knopp_forward(float* out, const float* inp, int M, int N, int num_iters,
                                   float eps, hipStream_t stream = nullptr) {
    constexpr int BLOCK_SIZE = 256;

    if (M == 32 && N == 32) {
        sinkhorn_knopp_warp_per_row_32x32_kernel<BLOCK_SIZE>
            <<<1, BLOCK_SIZE, 0, stream>>>(out, inp, num_iters, eps);
    } else if (N <= 32 && M <= 64) {
        size_t smem_size = M * 32 * sizeof(float) + 32 * sizeof(float);
        sinkhorn_knopp_warp_optimized_kernel<32, BLOCK_SIZE>
            <<<1, BLOCK_SIZE, smem_size, stream>>>(out, inp, M, N, num_iters, eps);
    } else if (M <= 64 && N <= 64) {
        constexpr int MAX_DIM = 64;
        size_t smem_size =
            MAX_DIM * MAX_DIM * sizeof(float) + MAX_DIM * sizeof(float) + MAX_DIM * sizeof(float);

        sinkhorn_knopp_single_block_kernel<MAX_DIM, BLOCK_SIZE>
            <<<1, BLOCK_SIZE, smem_size, stream>>>(out, inp, M, N, num_iters, eps);
    } else if (M <= 128 && N <= 128) {
        constexpr int MAX_DIM = 128;
        size_t smem_size =
            MAX_DIM * MAX_DIM * sizeof(float) + MAX_DIM * sizeof(float) + MAX_DIM * sizeof(float);

        auto kernel = sinkhorn_knopp_single_block_kernel<MAX_DIM, BLOCK_SIZE>;
        hipFuncSetAttribute(reinterpret_cast<const void*>(kernel),
                            hipFuncAttributeMaxDynamicSharedMemorySize,
                            smem_size);

        kernel<<<1, BLOCK_SIZE, smem_size, stream>>>(out, inp, M, N, num_iters, eps);
    } else {
        constexpr int TILE_SIZE = 32;
        dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
        size_t smem_size = TILE_SIZE * TILE_SIZE * sizeof(float) + TILE_SIZE * sizeof(float) +
                           TILE_SIZE * sizeof(float);

        sinkhorn_knopp_kernel<TILE_SIZE, TILE_SIZE, BLOCK_SIZE>
            <<<grid, BLOCK_SIZE, smem_size, stream>>>(out, inp, M, N, num_iters, eps);
    }
}

inline void sinkhorn_knopp_forward_fused_exp(float* out, float* H_res_exp, const float* inp, int M,
                                             int N, int num_iters, float eps,
                                             hipStream_t stream = nullptr) {
    constexpr int BLOCK_SIZE = 256;

    if (M <= 64 && N <= 64) {
        constexpr int MAX_DIM = 64;
        size_t smem_size =
            MAX_DIM * MAX_DIM * sizeof(float) + MAX_DIM * sizeof(float) + MAX_DIM * sizeof(float);

        sinkhorn_knopp_single_block_fused_exp_kernel<MAX_DIM, BLOCK_SIZE>
            <<<1, BLOCK_SIZE, smem_size, stream>>>(out, H_res_exp, inp, M, N, num_iters, eps);
    } else if (M <= 128 && N <= 128) {
        constexpr int MAX_DIM = 128;
        size_t smem_size =
            MAX_DIM * MAX_DIM * sizeof(float) + MAX_DIM * sizeof(float) + MAX_DIM * sizeof(float);

        auto kernel = sinkhorn_knopp_single_block_fused_exp_kernel<MAX_DIM, BLOCK_SIZE>;
        hipFuncSetAttribute(reinterpret_cast<const void*>(kernel),
                            hipFuncAttributeMaxDynamicSharedMemorySize,
                            smem_size);

        kernel<<<1, BLOCK_SIZE, smem_size, stream>>>(out, H_res_exp, inp, M, N, num_iters, eps);
    } else {
        fprintf(stderr, "sinkhorn_knopp_forward_fused_exp: M > 128 or N > 128 not supported\n");
    }
}

template<int N_COMPILE, int MAX_ITERS, int BLOCK_SIZE>
__global__ void sinkhorn_knopp_backward_checkpointed_kernel(float* __restrict__ d_inp,
                                                            const float* __restrict__ grad,
                                                            const float* __restrict__ M_inp, int N,
                                                            int num_iters, float eps) {
    extern __shared__ float smem[];

    float* checkpoints = smem;
    float* d_tile = checkpoints + MAX_ITERS * N_COMPILE * N_COMPILE;
    float* row_buffer = d_tile + N_COMPILE * N_COMPILE;
    float* col_buffer = row_buffer + N_COMPILE;
    float* tile_work = col_buffer + N_COMPILE;

    int total = N * N;

    for (int i = threadIdx.x; i < total; i += BLOCK_SIZE) {
        tile_work[i] = M_inp[i];
    }
    __syncthreads();

    for (int iter = 0; iter < num_iters; iter++) {
        for (int r = threadIdx.x; r < N; r += BLOCK_SIZE) {
            float sum = 0.0f;
#pragma unroll 4
            for (int c = 0; c < N; c++) {
                sum += tile_work[r * N + c];
            }
            row_buffer[r] = (sum > eps) ? __frcp_rn(sum) : 0.0f;
        }
        __syncthreads();

        for (int i = threadIdx.x; i < total; i += BLOCK_SIZE) {
            int r = i / N;
            tile_work[i] *= row_buffer[r];
        }
        __syncthreads();

        for (int i = threadIdx.x; i < total; i += BLOCK_SIZE) {
            checkpoints[iter * N * N + i] = tile_work[i];
        }
        __syncthreads();

        for (int c = threadIdx.x; c < N; c += BLOCK_SIZE) {
            float sum = 0.0f;
#pragma unroll 4
            for (int r = 0; r < N; r++) {
                sum += tile_work[r * N + c];
            }
            col_buffer[c] = (sum > eps) ? __frcp_rn(sum) : 0.0f;
        }
        __syncthreads();

        for (int i = threadIdx.x; i < total; i += BLOCK_SIZE) {
            int c = i % N;
            tile_work[i] *= col_buffer[c];
        }
        __syncthreads();
    }

    for (int i = threadIdx.x; i < total; i += BLOCK_SIZE) {
        d_tile[i] = grad[i];
    }
    __syncthreads();

    for (int iter = num_iters - 1; iter >= 0; iter--) {
        for (int i = threadIdx.x; i < total; i += BLOCK_SIZE) {
            tile_work[i] = checkpoints[iter * N * N + i];
        }
        __syncthreads();

        for (int c = threadIdx.x; c < N; c += BLOCK_SIZE) {
            float dot = 0.0f;
            for (int r = 0; r < N; r++) {
                dot += d_tile[r * N + c] * tile_work[r * N + c];
            }
            col_buffer[c] = dot;
        }
        __syncthreads();

        for (int i = threadIdx.x; i < total; i += BLOCK_SIZE) {
            int c = i % N;
            d_tile[i] = d_tile[i] - tile_work[i] * col_buffer[c];
        }
        __syncthreads();

        for (int r = threadIdx.x; r < N; r += BLOCK_SIZE) {
            float dot = 0.0f;
            for (int c = 0; c < N; c++) {
                dot += d_tile[r * N + c] * tile_work[r * N + c];
            }
            row_buffer[r] = dot;
        }
        __syncthreads();

        for (int i = threadIdx.x; i < total; i += BLOCK_SIZE) {
            int r = i / N;
            d_tile[i] = d_tile[i] - tile_work[i] * row_buffer[r];
        }
        __syncthreads();
    }

    for (int i = threadIdx.x; i < total; i += BLOCK_SIZE) {
        d_inp[i] = d_tile[i];
    }
}

template<int MAX_DIM, int BLOCK_SIZE>
__global__ void
sinkhorn_knopp_backward_kernel(float* __restrict__ d_inp, const float* __restrict__ grad,
                               const float* __restrict__ M_out, const float* __restrict__ M_inp,
                               int N, int num_iters, float eps) {
    extern __shared__ float smem[];
    float* d_tile = smem;
    float* row_buffer = smem + MAX_DIM * MAX_DIM;
    float* col_buffer = row_buffer + MAX_DIM;
    float* tile_fwd = col_buffer + MAX_DIM;
    float* row_sums = tile_fwd + MAX_DIM * MAX_DIM;
    float* col_sums = row_sums + MAX_DIM;

    int total = N * N;

    for (int i = threadIdx.x; i < total; i += BLOCK_SIZE) {
        d_tile[i] = grad[i];
    }
    __syncthreads();

    for (int iter = num_iters - 1; iter >= 0; iter--) {
        for (int i = threadIdx.x; i < total; i += BLOCK_SIZE) {
            tile_fwd[i] = M_inp[i];
        }
        __syncthreads();

        for (int fwd_iter = 0; fwd_iter < iter; fwd_iter++) {
            for (int r = threadIdx.x; r < N; r += BLOCK_SIZE) {
                float sum = 0.0f;
                for (int c = 0; c < N; c++) {
                    sum += tile_fwd[r * N + c];
                }
                row_sums[r] = (sum > eps) ? __frcp_rn(sum) : 0.0f;
            }
            __syncthreads();

            for (int i = threadIdx.x; i < total; i += BLOCK_SIZE) {
                int r = i / N;
                tile_fwd[i] *= row_sums[r];
            }
            __syncthreads();

            for (int c = threadIdx.x; c < N; c += BLOCK_SIZE) {
                float sum = 0.0f;
                for (int r = 0; r < N; r++) {
                    sum += tile_fwd[r * N + c];
                }
                col_sums[c] = (sum > eps) ? __frcp_rn(sum) : 0.0f;
            }
            __syncthreads();

            for (int i = threadIdx.x; i < total; i += BLOCK_SIZE) {
                int c = i % N;
                tile_fwd[i] *= col_sums[c];
            }
            __syncthreads();
        }

        for (int r = threadIdx.x; r < N; r += BLOCK_SIZE) {
            float sum = 0.0f;
            for (int c = 0; c < N; c++) {
                sum += tile_fwd[r * N + c];
            }
            row_sums[r] = (sum > eps) ? __frcp_rn(sum) : 0.0f;
        }
        __syncthreads();

        for (int i = threadIdx.x; i < total; i += BLOCK_SIZE) {
            int r = i / N;
            tile_fwd[i] *= row_sums[r];
        }
        __syncthreads();

        for (int c = threadIdx.x; c < N; c += BLOCK_SIZE) {
            float dot = 0.0f;
            for (int r = 0; r < N; r++) {
                dot += d_tile[r * N + c] * tile_fwd[r * N + c];
            }
            col_buffer[c] = dot;
        }
        __syncthreads();

        for (int i = threadIdx.x; i < total; i += BLOCK_SIZE) {
            int c = i % N;
            d_tile[i] = d_tile[i] - tile_fwd[i] * col_buffer[c];
        }
        __syncthreads();

        for (int r = threadIdx.x; r < N; r += BLOCK_SIZE) {
            float dot = 0.0f;
            for (int c = 0; c < N; c++) {
                dot += d_tile[r * N + c] * tile_fwd[r * N + c];
            }
            row_buffer[r] = dot;
        }
        __syncthreads();

        for (int i = threadIdx.x; i < total; i += BLOCK_SIZE) {
            int r = i / N;
            d_tile[i] = d_tile[i] - tile_fwd[i] * row_buffer[r];
        }
        __syncthreads();
    }

    for (int i = threadIdx.x; i < total; i += BLOCK_SIZE) {
        d_inp[i] = d_tile[i];
    }
}

inline void sinkhorn_knopp_backward(float* d_inp, const float* grad, const float* M_out,
                                    const float* M_inp, int N, int num_iters, float eps,
                                    hipStream_t stream = nullptr) {
    constexpr int BLOCK_SIZE = 256;

    if (N <= 32 && num_iters <= 20) {
        constexpr int N_COMPILE = 32;
        constexpr int MAX_ITERS = 20;
        size_t smem_size =
            (MAX_ITERS + 3) * N_COMPILE * N_COMPILE * sizeof(float) + 2 * N_COMPILE * sizeof(float);

        auto kernel = sinkhorn_knopp_backward_checkpointed_kernel<N_COMPILE, MAX_ITERS, BLOCK_SIZE>;
        hipFuncSetAttribute(reinterpret_cast<const void*>(kernel),
                            hipFuncAttributeMaxDynamicSharedMemorySize,
                            smem_size);

        kernel<<<1, BLOCK_SIZE, smem_size, stream>>>(d_inp, grad, M_inp, N, num_iters, eps);
    } else if (N <= 64) {
        constexpr int MAX_DIM = 64;
        size_t smem_size = 2 * MAX_DIM * MAX_DIM * sizeof(float) + 4 * MAX_DIM * sizeof(float);

        sinkhorn_knopp_backward_kernel<MAX_DIM, BLOCK_SIZE>
            <<<1, BLOCK_SIZE, smem_size, stream>>>(d_inp, grad, M_out, M_inp, N, num_iters, eps);
    } else {
        fprintf(stderr, "sinkhorn_knopp_backward: N > 64 not supported\n");
    }
}

template<int N_COMPILE>
__global__ void sinkhorn_knopp_batched_n4_kernel(float* __restrict__ out,
                                                 const float* __restrict__ inp, int B,
                                                 int num_iters, float eps) {
    static_assert(N_COMPILE == 4,
                  "This kernel is optimized for the case where n=4, which is the special case "
                  "presented in the paper in section 4.3 introduction.");

    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch_idx >= B)
        return;

    const float* inp_batch = inp + batch_idx * 16;
    float* out_batch = out + batch_idx * 16;

    float4 row0 = *reinterpret_cast<const float4*>(inp_batch);
    float4 row1 = *reinterpret_cast<const float4*>(inp_batch + 4);
    float4 row2 = *reinterpret_cast<const float4*>(inp_batch + 8);
    float4 row3 = *reinterpret_cast<const float4*>(inp_batch + 12);

#pragma unroll
    for (int iter = 0; iter < num_iters; iter++) {
        float s0 = row0.x + row0.y + row0.z + row0.w;
        float s1 = row1.x + row1.y + row1.z + row1.w;
        float s2 = row2.x + row2.y + row2.z + row2.w;
        float s3 = row3.x + row3.y + row3.z + row3.w;

        float inv0 = (s0 > eps) ? __frcp_rn(s0) : 0.0f;
        float inv1 = (s1 > eps) ? __frcp_rn(s1) : 0.0f;
        float inv2 = (s2 > eps) ? __frcp_rn(s2) : 0.0f;
        float inv3 = (s3 > eps) ? __frcp_rn(s3) : 0.0f;

        row0.x *= inv0;
        row0.y *= inv0;
        row0.z *= inv0;
        row0.w *= inv0;
        row1.x *= inv1;
        row1.y *= inv1;
        row1.z *= inv1;
        row1.w *= inv1;
        row2.x *= inv2;
        row2.y *= inv2;
        row2.z *= inv2;
        row2.w *= inv2;
        row3.x *= inv3;
        row3.y *= inv3;
        row3.z *= inv3;
        row3.w *= inv3;

        float c0 = row0.x + row1.x + row2.x + row3.x;
        float c1 = row0.y + row1.y + row2.y + row3.y;
        float c2 = row0.z + row1.z + row2.z + row3.z;
        float c3 = row0.w + row1.w + row2.w + row3.w;

        float cinv0 = (c0 > eps) ? __frcp_rn(c0) : 0.0f;
        float cinv1 = (c1 > eps) ? __frcp_rn(c1) : 0.0f;
        float cinv2 = (c2 > eps) ? __frcp_rn(c2) : 0.0f;
        float cinv3 = (c3 > eps) ? __frcp_rn(c3) : 0.0f;

        row0.x *= cinv0;
        row0.y *= cinv1;
        row0.z *= cinv2;
        row0.w *= cinv3;
        row1.x *= cinv0;
        row1.y *= cinv1;
        row1.z *= cinv2;
        row1.w *= cinv3;
        row2.x *= cinv0;
        row2.y *= cinv1;
        row2.z *= cinv2;
        row2.w *= cinv3;
        row3.x *= cinv0;
        row3.y *= cinv1;
        row3.z *= cinv2;
        row3.w *= cinv3;
    }

    *reinterpret_cast<float4*>(out_batch) = row0;
    *reinterpret_cast<float4*>(out_batch + 4) = row1;
    *reinterpret_cast<float4*>(out_batch + 8) = row2;
    *reinterpret_cast<float4*>(out_batch + 12) = row3;
}

template<int N_MAX, int BLOCK_SIZE>
__global__ void sinkhorn_knopp_batched_kernel(float* __restrict__ out,
                                              const float* __restrict__ inp, int B, int n,
                                              int num_iters, float eps) {
    int batch_idx = blockIdx.x;
    if (batch_idx >= B)
        return;

    extern __shared__ float smem[];
    float* tile = smem;
    float* row_sums = tile + N_MAX * N_MAX;
    float* col_sums = row_sums + N_MAX;

    const float* inp_batch = inp + batch_idx * n * n;
    float* out_batch = out + batch_idx * n * n;

    int total = n * n;

    if (n == 4 && (total % 4) == 0) {
        int total_vec = total / 4;
        for (int i = threadIdx.x; i < total_vec; i += BLOCK_SIZE) {
            reinterpret_cast<float4*>(tile)[i] = reinterpret_cast<const float4*>(inp_batch)[i];
        }
    } else {
        for (int i = threadIdx.x; i < total; i += BLOCK_SIZE) {
            tile[i] = inp_batch[i];
        }
    }
    __syncthreads();

    for (int iter = 0; iter < num_iters; iter++) {
        for (int r = threadIdx.x; r < n; r += BLOCK_SIZE) {
            float sum = 0.0f;
#pragma unroll 4
            for (int c = 0; c < n; c++) {
                sum += tile[r * n + c];
            }
            row_sums[r] = (sum > eps) ? __frcp_rn(sum) : 0.0f;
        }
        __syncthreads();

        for (int i = threadIdx.x; i < total; i += BLOCK_SIZE) {
            int r = i / n;
            tile[i] *= row_sums[r];
        }
        __syncthreads();

        for (int c = threadIdx.x; c < n; c += BLOCK_SIZE) {
            float sum = 0.0f;
#pragma unroll 4
            for (int r = 0; r < n; r++) {
                sum += tile[r * n + c];
            }
            col_sums[c] = (sum > eps) ? __frcp_rn(sum) : 0.0f;
        }
        __syncthreads();

        for (int i = threadIdx.x; i < total; i += BLOCK_SIZE) {
            int c = i % n;
            tile[i] *= col_sums[c];
        }
        __syncthreads();
    }

    if (n == 4 && (total % 4) == 0) {
        int total_vec = total / 4;
        for (int i = threadIdx.x; i < total_vec; i += BLOCK_SIZE) {
            reinterpret_cast<float4*>(out_batch)[i] = reinterpret_cast<float4*>(tile)[i];
        }
    } else {
        for (int i = threadIdx.x; i < total; i += BLOCK_SIZE) {
            out_batch[i] = tile[i];
        }
    }
}

inline void sinkhorn_knopp_forward_batched(float* out, const float* inp, int B, int n,
                                           int num_iters, float eps,
                                           hipStream_t stream = nullptr) {
    if (n == 4) {
        constexpr int THREADS_PER_BLOCK = 256;
        int num_blocks = (B + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

        sinkhorn_knopp_batched_n4_kernel<4>
            <<<num_blocks, THREADS_PER_BLOCK, 0, stream>>>(out, inp, B, num_iters, eps);
        return;
    }

    constexpr int BLOCK_SIZE = 128;
    constexpr int N_MAX = 32;

    if (n > N_MAX) {
        for (int b = 0; b < B; b++) {
            sinkhorn_knopp_forward(out + b * n * n, inp + b * n * n, n, n, num_iters, eps, stream);
        }
        return;
    }

    size_t smem_size = N_MAX * N_MAX * sizeof(float) + 2 * N_MAX * sizeof(float);

    sinkhorn_knopp_batched_kernel<N_MAX, BLOCK_SIZE>
        <<<B, BLOCK_SIZE, smem_size, stream>>>(out, inp, B, n, num_iters, eps);
}



template<int BLOCK_SIZE, bool OUTPUT_RMS = false>
__global__ void rmsnorm_kernel(__hip_bfloat16* __restrict__ out, float* __restrict__ rms_out,
                               const __hip_bfloat16* __restrict__ inp,
                               const __hip_bfloat16* __restrict__ weight, int N, int C,
                               float eps) {
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);

    int idx = blockIdx.x;
    if (idx >= N)
        return;

    const __hip_bfloat16* x = inp + idx * C;
    __hip_bfloat16* o = out + idx * C;

    extern __shared__ float shared[];
    float* s_sum_sq = shared;

    float thread_sum_sq = 0.0f;
    for (int i = threadIdx.x; i < C; i += BLOCK_SIZE) {
        float val = (float)x[i];
        thread_sum_sq += val * val;
    }

    float warp_sum = cg::reduce(warp, thread_sum_sq, cg::plus<float>());

    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    int num_warps = BLOCK_SIZE / 32;

    if (lane_id == 0) {
        s_sum_sq[warp_id] = warp_sum;
    }
    __syncthreads();

    if (warp_id == 0) {
        float val = (lane_id < num_warps) ? s_sum_sq[lane_id] : 0.0f;
        float block_sum = cg::reduce(warp, val, cg::plus<float>());

        if (lane_id == 0) {
            float rms = sqrtf(block_sum / (float)C + eps);
            float rms_inv = 1.0f / rms;
            s_sum_sq[0] = rms_inv;
            if constexpr (OUTPUT_RMS) {
                rms_out[idx] = rms;
            }
        }
    }
    __syncthreads();

    float rms_inv = s_sum_sq[0];

    for (int i = threadIdx.x; i < C; i += BLOCK_SIZE) {
        float val = (float)x[i];
        float w = (float)weight[i];
        o[i] = (__hip_bfloat16)(val * rms_inv * w);
    }
}

template<int BLOCK_SIZE, bool OUTPUT_RMS = false>
__global__ void rmsnorm_kernel_vectorized(__hip_bfloat16* __restrict__ out,
                                          float* __restrict__ rms_out,
                                          const __hip_bfloat16* __restrict__ inp,
                                          const __hip_bfloat16* __restrict__ weight, int N, int C,
                                          float eps) {
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);

    int idx = blockIdx.x;
    if (idx >= N)
        return;

    const __hip_bfloat16* x = inp + idx * C;
    __hip_bfloat16* o = out + idx * C;

    extern __shared__ float shared[];
    float* s_sum_sq = shared;

    constexpr int VEC_SIZE = 8;
    int C_vec = C / VEC_SIZE;

    float thread_sum_sq = 0.0f;

    using vec_t = float4;
    const vec_t* x_vec = reinterpret_cast<const vec_t*>(x);

    for (int i = threadIdx.x; i < C_vec; i += BLOCK_SIZE) {
        vec_t v = x_vec[i];
        __hip_bfloat162* bf_v = reinterpret_cast<__hip_bfloat162*>(&v);

#pragma unroll
        for (int j = 0; j < 4; j++) {
            float2 f = __bfloat1622float2(bf_v[j]);
            thread_sum_sq += f.x * f.x + f.y * f.y;
        }
    }

    int remainder_start = C_vec * VEC_SIZE;
    for (int i = remainder_start + threadIdx.x; i < C; i += BLOCK_SIZE) {
        float val = (float)x[i];
        thread_sum_sq += val * val;
    }

    float warp_sum = cg::reduce(warp, thread_sum_sq, cg::plus<float>());

    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    int num_warps = BLOCK_SIZE / 32;

    if (lane_id == 0) {
        s_sum_sq[warp_id] = warp_sum;
    }
    __syncthreads();

    if (warp_id == 0) {
        float val = (lane_id < num_warps) ? s_sum_sq[lane_id] : 0.0f;
        float block_sum = cg::reduce(warp, val, cg::plus<float>());

        if (lane_id == 0) {
            float rms = sqrtf(block_sum / (float)C + eps);
            float rms_inv = 1.0f / rms;
            s_sum_sq[0] = rms_inv;
            if constexpr (OUTPUT_RMS) {
                rms_out[idx] = rms;
            }
        }
    }
    __syncthreads();

    float rms_inv = s_sum_sq[0];

    vec_t* o_vec = reinterpret_cast<vec_t*>(o);
    const vec_t* w_vec = reinterpret_cast<const vec_t*>(weight);

    for (int i = threadIdx.x; i < C_vec; i += BLOCK_SIZE) {
        vec_t xv = x_vec[i];
        vec_t wv = w_vec[i];

        __hip_bfloat162* bf_x = reinterpret_cast<__hip_bfloat162*>(&xv);
        __hip_bfloat162* bf_w = reinterpret_cast<__hip_bfloat162*>(&wv);

        vec_t ov;
        __hip_bfloat162* bf_o = reinterpret_cast<__hip_bfloat162*>(&ov);

#pragma unroll
        for (int j = 0; j < 4; j++) {
            float2 xf = __bfloat1622float2(bf_x[j]);
            float2 wf = __bfloat1622float2(bf_w[j]);
            float2 of;
            of.x = xf.x * rms_inv * wf.x;
            of.y = xf.y * rms_inv * wf.y;
            bf_o[j] = __float22bfloat162_rn(of);
        }

        o_vec[i] = ov;
    }

    for (int i = remainder_start + threadIdx.x; i < C; i += BLOCK_SIZE) {
        float val = (float)x[i];
        float w = (float)weight[i];
        o[i] = (__hip_bfloat16)(val * rms_inv * w);
    }
}

inline void rmsnorm_forward(__hip_bfloat16* out, const __hip_bfloat16* inp,
                            const __hip_bfloat16* weight, int N, int C, float eps,
                            hipStream_t stream = nullptr) {
    constexpr int BLOCK_SIZE = 512;
    int num_warps = BLOCK_SIZE / 32;
    size_t shared_mem = num_warps * sizeof(float);

    dim3 grid(N);
    dim3 block(BLOCK_SIZE);

    if (C % 8 == 0 && C >= 64) {
        rmsnorm_kernel_vectorized<BLOCK_SIZE, false>
            <<<grid, block, shared_mem, stream>>>(out, nullptr, inp, weight, N, C, eps);
    } else {
        rmsnorm_kernel<BLOCK_SIZE, false>
            <<<grid, block, shared_mem, stream>>>(out, nullptr, inp, weight, N, C, eps);
    }
}

inline void rmsnorm_forward_with_rms(__hip_bfloat16* out, float* rms_out,
                                     const __hip_bfloat16* inp,
                                     const __hip_bfloat16* weight, int N, int C, float eps,
                                     hipStream_t stream = nullptr) {
    constexpr int BLOCK_SIZE = 512;
    int num_warps = BLOCK_SIZE / 32;
    size_t shared_mem = num_warps * sizeof(float);

    dim3 grid(N);
    dim3 block(BLOCK_SIZE);

    if (C % 8 == 0 && C >= 64) {
        rmsnorm_kernel_vectorized<BLOCK_SIZE, true>
            <<<grid, block, shared_mem, stream>>>(out, rms_out, inp, weight, N, C, eps);
    } else {
        rmsnorm_kernel<BLOCK_SIZE, true>
            <<<grid, block, shared_mem, stream>>>(out, rms_out, inp, weight, N, C, eps);
    }
}

template<int BLOCK_SIZE>
__global__ void rmsnorm_backward_kernel(float* __restrict__ d_inp, float* __restrict__ d_weight,
                                        const float* __restrict__ grad,
                                        const __hip_bfloat16* __restrict__ inp,
                                        const __hip_bfloat16* __restrict__ weight,
                                        const float* __restrict__ rms, int N, int C) {
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);

    int idx = blockIdx.x;
    if (idx >= N)
        return;

    const __hip_bfloat16* x = inp + idx * C;
    const float* g = grad + idx * C;
    float* dx = d_inp + idx * C;
    float r = rms[idx];
    float r_inv = 1.0f / r;

    extern __shared__ float shared[];
    float* s_reduce = shared;

    float thread_dot = 0.0f;
    for (int i = threadIdx.x; i < C; i += BLOCK_SIZE) {
        float g_val = g[i];
        float w_val = (float)weight[i];
        float x_val = (float)x[i];
        thread_dot += g_val * w_val * x_val;
    }

    float warp_dot = cg::reduce(warp, thread_dot, cg::plus<float>());

    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    int num_warps = BLOCK_SIZE / 32;

    if (lane_id == 0) {
        s_reduce[warp_id] = warp_dot;
    }
    __syncthreads();

    if (warp_id == 0) {
        float val = (lane_id < num_warps) ? s_reduce[lane_id] : 0.0f;
        float block_dot = cg::reduce(warp, val, cg::plus<float>());
        if (lane_id == 0) {
            s_reduce[0] = block_dot;
        }
    }
    __syncthreads();

    float dot_sum = s_reduce[0];
    float correction = dot_sum / ((float)C * r * r);

    for (int i = threadIdx.x; i < C; i += BLOCK_SIZE) {
        float g_val = g[i];
        float w_val = (float)weight[i];
        float x_val = (float)x[i];

        dx[i] = (g_val * w_val * r_inv) - (x_val * correction * r_inv);

        atomicAdd(&d_weight[i], g_val * x_val * r_inv);
    }
}

inline void rmsnorm_backward(float* d_inp, float* d_weight, const float* grad,
                             const __hip_bfloat16* inp, const __hip_bfloat16* weight,
                             const float* rms, int N, int C, hipStream_t stream = nullptr) {
    constexpr int BLOCK_SIZE = 512;
    int num_warps = BLOCK_SIZE / 32;
    size_t shared_mem = num_warps * sizeof(float);

    rmsnorm_backward_kernel<BLOCK_SIZE>
        <<<N, BLOCK_SIZE, shared_mem, stream>>>(d_inp, d_weight, grad, inp, weight, rms, N, C);
}


template<int BLOCK_SIZE>
__global__ void compute_rms_kernel(float* __restrict__ rms_out, const __hip_bfloat16* __restrict__ inp,
                                   int N, int C, float eps) {
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);

    int idx = blockIdx.x;
    if (idx >= N)
        return;

    const __hip_bfloat16* x = inp + idx * C;

    extern __shared__ float shared[];

    float thread_sum_sq = 0.0f;
    for (int i = threadIdx.x; i < C; i += BLOCK_SIZE) {
        float val = (float)x[i];
        thread_sum_sq += val * val;
    }

    float warp_sum = cg::reduce(warp, thread_sum_sq, cg::plus<float>());

    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    int num_warps = BLOCK_SIZE / 32;

    if (lane_id == 0) {
        shared[warp_id] = warp_sum;
    }
    __syncthreads();

    if (warp_id == 0) {
        float val = (lane_id < num_warps) ? shared[lane_id] : 0.0f;
        float block_sum = cg::reduce(warp, val, cg::plus<float>());

        if (lane_id == 0) {
            float rms = sqrtf(block_sum / (float)C + eps);
            rms_out[idx] = rms;
        }
    }
}

template<int BLOCK_SIZE>
__global__ void compute_rms_kernel_vectorized(float* __restrict__ rms_out,
                                              const __hip_bfloat16* __restrict__ inp, int N, int C,
                                              float eps) {
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);

    int idx = blockIdx.x;
    if (idx >= N)
        return;

    const __hip_bfloat16* x = inp + idx * C;

    extern __shared__ float shared[];

    constexpr int VEC_SIZE = 8;
    int C_vec = C / VEC_SIZE;

    float thread_sum_sq = 0.0f;

    using vec_t = float4;
    const vec_t* x_vec = reinterpret_cast<const vec_t*>(x);

    for (int i = threadIdx.x; i < C_vec; i += BLOCK_SIZE) {
        vec_t v = x_vec[i];
        __hip_bfloat162* bf_v = reinterpret_cast<__hip_bfloat162*>(&v);

#pragma unroll
        for (int j = 0; j < 4; j++) {
            float2 f = __bfloat1622float2(bf_v[j]);
            thread_sum_sq += f.x * f.x + f.y * f.y;
        }
    }

    int remainder_start = C_vec * VEC_SIZE;
    for (int i = remainder_start + threadIdx.x; i < C; i += BLOCK_SIZE) {
        float val = (float)x[i];
        thread_sum_sq += val * val;
    }

    float warp_sum = cg::reduce(warp, thread_sum_sq, cg::plus<float>());

    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    int num_warps = BLOCK_SIZE / 32;

    if (lane_id == 0) {
        shared[warp_id] = warp_sum;
    }
    __syncthreads();

    if (warp_id == 0) {
        float val = (lane_id < num_warps) ? shared[lane_id] : 0.0f;
        float block_sum = cg::reduce(warp, val, cg::plus<float>());

        if (lane_id == 0) {
            float rms = sqrtf(block_sum / (float)C + eps);
            rms_out[idx] = rms;
        }
    }
}

inline void compute_rms(float* rms_out, const __hip_bfloat16* inp, int N, int C, float eps,
                        hipStream_t stream = nullptr) {
    constexpr int BLOCK_SIZE = 512;
    int num_warps = BLOCK_SIZE / 32;
    size_t shared_mem = num_warps * sizeof(float);

    if (C % 8 == 0 && C >= 64) {
        compute_rms_kernel_vectorized<BLOCK_SIZE>
            <<<N, BLOCK_SIZE, shared_mem, stream>>>(rms_out, inp, N, C, eps);
    } else {
        compute_rms_kernel<BLOCK_SIZE>
            <<<N, BLOCK_SIZE, shared_mem, stream>>>(rms_out, inp, N, C, eps);
    }
}

template<int BLOCK_SIZE>
__global__ void divide_by_rms_kernel(float* __restrict__ out, const float* __restrict__ rms, int M,
                                     int N) {
    int row = blockIdx.x;
    if (row >= M)
        return;

    float r_inv = 1.0f / rms[row];
    float* out_row = out + row * N;

    for (int i = threadIdx.x; i < N; i += BLOCK_SIZE) {
        out_row[i] *= r_inv;
    }
}

template<int BLOCK_SIZE>
__global__ void divide_by_rms_kernel_vectorized(float* __restrict__ out,
                                                const float* __restrict__ rms, int M, int N) {
    int row = blockIdx.x;
    if (row >= M)
        return;

    float r_inv = 1.0f / rms[row];
    float* out_row = out + row * N;

    constexpr int VEC_SIZE = 4;
    int N_vec = N / VEC_SIZE;

    float4* out_vec = reinterpret_cast<float4*>(out_row);

    for (int i = threadIdx.x; i < N_vec; i += BLOCK_SIZE) {
        float4 v = out_vec[i];
        v.x *= r_inv;
        v.y *= r_inv;
        v.z *= r_inv;
        v.w *= r_inv;
        out_vec[i] = v;
    }

    int remainder_start = N_vec * VEC_SIZE;
    for (int i = remainder_start + threadIdx.x; i < N; i += BLOCK_SIZE) {
        out_row[i] *= r_inv;
    }
}

inline void divide_by_rms(float* out, const float* rms, int M, int N,
                          hipStream_t stream = nullptr) {
    constexpr int BLOCK_SIZE = 256;

    if (N % 4 == 0 && N >= 16) {
        divide_by_rms_kernel_vectorized<BLOCK_SIZE><<<M, BLOCK_SIZE, 0, stream>>>(out, rms, M, N);
    } else {
        divide_by_rms_kernel<BLOCK_SIZE><<<M, BLOCK_SIZE, 0, stream>>>(out, rms, M, N);
    }
}

struct MatmulDescriptors {
    hipblasLtHandle_t handle;
    hipblasLtMatmulDesc_t matmul_desc;
    hipblasLtMatrixLayout_t A_desc;
    hipblasLtMatrixLayout_t B_desc;
    hipblasLtMatrixLayout_t C_desc;
    hipblasLtMatmulPreference_t preference;
    hipblasLtMatmulHeuristicResult_t heuristic;
    void* workspace;
    size_t workspace_size;
};

inline void init_matmul_descriptors(MatmulDescriptors& desc, int M, int N, int K,
                                    size_t workspace_size = 32 * 1024 * 1024) {
    CHECK_COND(hipblasLtCreate(&desc.handle) == HIPBLAS_STATUS_SUCCESS);

    hipblasComputeType_t compute_type = HIPBLAS_COMPUTE_32F;
    hipDataType ab_type = HIP_R_16BF;
    hipDataType c_type = HIP_R_32F;
    hipDataType scale_type = HIP_R_32F;

    CHECK_COND(hipblasLtMatmulDescCreate(&desc.matmul_desc, compute_type, scale_type) == HIPBLAS_STATUS_SUCCESS);

    hipblasOperation_t trans_a = HIPBLAS_OP_T;
    hipblasOperation_t trans_b = HIPBLAS_OP_N;
    CHECK_COND(hipblasLtMatmulDescSetAttribute(desc.matmul_desc, HIPBLASLT_MATMUL_DESC_TRANSA,
                                                &trans_a, sizeof(trans_a)) == HIPBLAS_STATUS_SUCCESS);
    CHECK_COND(hipblasLtMatmulDescSetAttribute(desc.matmul_desc, HIPBLASLT_MATMUL_DESC_TRANSB,
                                                &trans_b, sizeof(trans_b)) == HIPBLAS_STATUS_SUCCESS);

    // Use column-major layouts and swap operands:
    // op(A) = B_row (N x K) via A_desc (K x N) with transA = T
    // op(B) = A_row^T (K x M) via B_desc (K x M) with transB = N
    // C_desc is (N x M) column-major, which matches row-major (M x N) layout.
    CHECK_COND(hipblasLtMatrixLayoutCreate(&desc.A_desc, ab_type, K, N, K) == HIPBLAS_STATUS_SUCCESS);
    CHECK_COND(hipblasLtMatrixLayoutCreate(&desc.B_desc, ab_type, K, M, K) == HIPBLAS_STATUS_SUCCESS);
    CHECK_COND(hipblasLtMatrixLayoutCreate(&desc.C_desc, c_type, N, M, N) == HIPBLAS_STATUS_SUCCESS);

    CHECK_COND(hipblasLtMatmulPreferenceCreate(&desc.preference) == HIPBLAS_STATUS_SUCCESS);
    CHECK_COND(hipblasLtMatmulPreferenceSetAttribute(desc.preference,
                                                      HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                      &workspace_size, sizeof(workspace_size)) == HIPBLAS_STATUS_SUCCESS);

    int returned_results = 0;
    CHECK_COND(hipblasLtMatmulAlgoGetHeuristic(
        desc.handle, desc.matmul_desc, desc.A_desc, desc.B_desc, desc.C_desc, desc.C_desc,
        desc.preference, 1, &desc.heuristic, &returned_results) == HIPBLAS_STATUS_SUCCESS);

    if (returned_results == 0) {
        fprintf(stderr, "No cuBLASLt algorithm found for row-major matmul\n");
        exit(EXIT_FAILURE);
    }

    desc.workspace_size = workspace_size;
    HIP_CALL(hipMalloc(&desc.workspace, workspace_size));
}

inline void destroy_matmul_descriptors(MatmulDescriptors& desc) {
    hipblasLtMatmulPreferenceDestroy(desc.preference);
    hipblasLtMatrixLayoutDestroy(desc.A_desc);
    hipblasLtMatrixLayoutDestroy(desc.B_desc);
    hipblasLtMatrixLayoutDestroy(desc.C_desc);
    hipblasLtMatmulDescDestroy(desc.matmul_desc);
    hipblasLtDestroy(desc.handle);
    hipFree(desc.workspace);
}

inline void matmul_forward(MatmulDescriptors& desc, float* out, const __hip_bfloat16* A, const __hip_bfloat16* B,
                           float alpha, float beta, hipStream_t stream = nullptr) {
    CHECK_COND(hipblasLtMatmul(desc.handle, desc.matmul_desc, &alpha, A, desc.A_desc, B,
                                desc.B_desc, &beta, out, desc.C_desc, out, desc.C_desc,
                                &desc.heuristic.algo, desc.workspace, desc.workspace_size, stream) == HIPBLAS_STATUS_SUCCESS);
}

struct FusedRMSNormMatmul {
    MatmulDescriptors matmul_desc;
    float* rms_buffer;
    int M, N, K;
    float eps;
    bool initialized;

    FusedRMSNormMatmul() : rms_buffer(nullptr), initialized(false) {}

    void init(int m, int n, int k, float epsilon = 1e-5f) {
        M = m;
        N = n;
        K = k;
        eps = epsilon;

        init_matmul_descriptors(matmul_desc, M, N, K);
        HIP_CALL(hipMalloc(&rms_buffer, M * sizeof(float)));
        initialized = true;
    }

    void destroy() {
        if (initialized) {
            destroy_matmul_descriptors(matmul_desc);
            hipFree(rms_buffer);
            initialized = false;
        }
    }

    void forward(float* out, const __hip_bfloat16* inp, const __hip_bfloat16* proj_weight,
                 hipStream_t stream = nullptr) {
        compute_rms(rms_buffer, inp, M, K, eps, stream);
        matmul_forward(matmul_desc, out, proj_weight, inp, 1.0f, 0.0f, stream);
        divide_by_rms(out, rms_buffer, M, N, stream);
    }

    float* get_rms_values() { return rms_buffer; }
};

template<int BLOCK_SIZE>
__global__ void compute_rms_pdl_kernel(float* __restrict__ rms_out, const __hip_bfloat16* __restrict__ inp,
                                       int N, int C, float eps) {
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);

    int idx = blockIdx.x;
    if (idx >= N)
        return;

    const __hip_bfloat16* x = inp + idx * C;

    extern __shared__ float shared[];

    float thread_sum_sq = 0.0f;
    for (int i = threadIdx.x; i < C; i += BLOCK_SIZE) {
        float val = (float)x[i];
        thread_sum_sq += val * val;
    }

    float warp_sum = cg::reduce(warp, thread_sum_sq, cg::plus<float>());

    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    int num_warps = BLOCK_SIZE / 32;

    if (lane_id == 0) {
        shared[warp_id] = warp_sum;
    }
    __syncthreads();

    if (warp_id == 0) {
        float val = (lane_id < num_warps) ? shared[lane_id] : 0.0f;
        float block_sum = cg::reduce(warp, val, cg::plus<float>());

        if (lane_id == 0) {
            float rms = sqrtf(block_sum / (float)C + eps);
            rms_out[idx] = rms;
        }
    }
}

inline void compute_rms_pdl(float* rms_out, const __hip_bfloat16* inp, int N, int C, float eps,
                            hipStream_t stream = nullptr) {
    constexpr int BLOCK_SIZE = 512;
    int num_warps = BLOCK_SIZE / 32;
    size_t shared_mem = num_warps * sizeof(float);

    compute_rms_pdl_kernel<BLOCK_SIZE>
        <<<N, BLOCK_SIZE, shared_mem, stream>>>(rms_out, inp, N, C, eps);
}

struct FusedRMSNormMatmulPDL {
    MatmulDescriptors matmul_desc;
    float* rms_buffer;
    int M, N, K;
    float eps;
    bool initialized;

    FusedRMSNormMatmulPDL() : rms_buffer(nullptr), initialized(false) {}

    void init(int m, int n, int k, float epsilon = 1e-5f) {
        M = m;
        N = n;
        K = k;
        eps = epsilon;

        init_matmul_descriptors(matmul_desc, M, N, K);
        HIP_CALL(hipMalloc(&rms_buffer, M * sizeof(float)));
        initialized = true;
    }

    void destroy() {
        if (initialized) {
            destroy_matmul_descriptors(matmul_desc);
            hipFree(rms_buffer);
            initialized = false;
        }
    }

    void forward(float* out, const __hip_bfloat16* inp, const __hip_bfloat16* proj_weight,
                 hipStream_t stream = nullptr) {
        compute_rms_pdl(rms_buffer, inp, M, K, eps, stream);
        matmul_forward(matmul_desc, out, proj_weight, inp, 1.0f, 0.0f, stream);
        divide_by_rms(out, rms_buffer, M, N, stream);
    }

    float* get_rms_values() { return rms_buffer; }
};

template<int BLOCK_SIZE>
__global__ void bf16_to_fp32_kernel(float* __restrict__ out, const __hip_bfloat16* __restrict__ inp,
                                    int total) {
    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (idx >= total)
        return;
    out[idx] = (float)inp[idx];
}

template<int BLOCK_SIZE>
__global__ void scale_grad_by_rms_kernel(float* __restrict__ grad_scaled,
                                         const float* __restrict__ grad,
                                         const float* __restrict__ rms, int M, int N) {
    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int total = M * N;
    if (idx >= total)
        return;

    int row = idx / N;
    float r_inv = 1.0f / rms[row];
    grad_scaled[idx] = grad[idx] * r_inv;
}

template<int BLOCK_SIZE>
__global__ void rms_correction_kernel(float* __restrict__ dx, const float* __restrict__ K_buf,
                                      const __hip_bfloat16* __restrict__ x, const float* __restrict__ rms,
                                      int M, int K_dim) {
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);

    int row = blockIdx.x;
    if (row >= M)
        return;

    extern __shared__ float shared[];

    const __hip_bfloat16* x_row = x + row * K_dim;
    const float* K_row = K_buf + row * K_dim;
    float* dx_row = dx + row * K_dim;
    float r = rms[row];

    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    int num_warps = BLOCK_SIZE / 32;

    float thread_dot = 0.0f;
    for (int i = threadIdx.x; i < K_dim; i += BLOCK_SIZE) {
        float xi = (float)x_row[i];
        thread_dot += K_row[i] * xi;
    }

    float warp_dot = cg::reduce(warp, thread_dot, cg::plus<float>());
    if (lane_id == 0) {
        shared[warp_id] = warp_dot;
    }
    __syncthreads();

    float K_dot_x = 0.0f;
    if (warp_id == 0) {
        float val = (lane_id < num_warps) ? shared[lane_id] : 0.0f;
        K_dot_x = cg::reduce(warp, val, cg::plus<float>());
        if (lane_id == 0) {
            shared[0] = K_dot_x;
        }
    }
    __syncthreads();
    K_dot_x = shared[0];

    float correction_scale = K_dot_x / ((float)K_dim * r * r);

    for (int i = threadIdx.x; i < K_dim; i += BLOCK_SIZE) {
        float xi = (float)x_row[i];
        dx_row[i] = K_row[i] - correction_scale * xi;
    }
}

struct FusedRMSNormMatmulBackward {
    hipblasLtHandle_t handle;
    hipblasLtMatmulDesc_t dW_matmul_desc;
    hipblasLtMatrixLayout_t dW_grad_desc;
    hipblasLtMatrixLayout_t dW_x_desc;
    hipblasLtMatrixLayout_t dW_out_desc;
    hipblasLtMatmulPreference_t dW_pref;
    hipblasLtMatmulHeuristicResult_t dW_heuristic;

    hipblasLtMatmulDesc_t dx_matmul_desc;
    hipblasLtMatrixLayout_t dx_grad_desc;
    hipblasLtMatrixLayout_t dx_W_desc;
    hipblasLtMatrixLayout_t dx_out_desc;
    hipblasLtMatmulPreference_t dx_pref;
    hipblasLtMatmulHeuristicResult_t dx_heuristic;

    void* workspace;
    size_t workspace_size;
    float* grad_scaled_buffer;
    float* K_buffer;
    float* x_fp32_buffer;
    float* W_fp32_buffer;

    int M, N, K;
    bool initialized;

    FusedRMSNormMatmulBackward()
        : workspace(nullptr), grad_scaled_buffer(nullptr), K_buffer(nullptr),
          x_fp32_buffer(nullptr), W_fp32_buffer(nullptr), initialized(false) {}

    void init(int m, int n, int k, float epsilon = 1e-5f) {
        M = m;
        N = n;
        K = k;

        workspace_size = 32 * 1024 * 1024;
        HIP_CALL(hipMalloc(&workspace, workspace_size));
        HIP_CALL(hipMalloc(&grad_scaled_buffer, M * N * sizeof(float)));
        HIP_CALL(hipMalloc(&K_buffer, M * K * sizeof(float)));
        HIP_CALL(hipMalloc(&x_fp32_buffer, M * K * sizeof(float)));
        HIP_CALL(hipMalloc(&W_fp32_buffer, N * K * sizeof(float)));

        CHECK_COND(hipblasLtCreate(&handle) == HIPBLAS_STATUS_SUCCESS);

        CHECK_COND(hipblasLtMatmulDescCreate(&dW_matmul_desc, HIPBLAS_COMPUTE_32F, HIP_R_32F) == HIPBLAS_STATUS_SUCCESS);
        hipblasOperation_t trans_a = HIPBLAS_OP_T;
        hipblasOperation_t trans_b = HIPBLAS_OP_N;
        CHECK_COND(hipblasLtMatmulDescSetAttribute(dW_matmul_desc, HIPBLASLT_MATMUL_DESC_TRANSA,
                                                    &trans_a, sizeof(trans_a)) == HIPBLAS_STATUS_SUCCESS);
        CHECK_COND(hipblasLtMatmulDescSetAttribute(dW_matmul_desc, HIPBLASLT_MATMUL_DESC_TRANSB,
                                                    &trans_b, sizeof(trans_b)) == HIPBLAS_STATUS_SUCCESS);

        CHECK_COND(hipblasLtMatrixLayoutCreate(&dW_grad_desc, HIP_R_32F, M, N, M) == HIPBLAS_STATUS_SUCCESS);
        CHECK_COND(hipblasLtMatrixLayoutCreate(&dW_x_desc, HIP_R_32F, M, K, M) == HIPBLAS_STATUS_SUCCESS);
        CHECK_COND(hipblasLtMatrixLayoutCreate(&dW_out_desc, HIP_R_32F, N, K, N) == HIPBLAS_STATUS_SUCCESS);

        CHECK_COND(hipblasLtMatmulPreferenceCreate(&dW_pref) == HIPBLAS_STATUS_SUCCESS);
        CHECK_COND(hipblasLtMatmulPreferenceSetAttribute(dW_pref,
                                                          HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                          &workspace_size, sizeof(workspace_size)) == HIPBLAS_STATUS_SUCCESS);

        int returned = 0;
        CHECK_COND(hipblasLtMatmulAlgoGetHeuristic(handle, dW_matmul_desc, dW_grad_desc, dW_x_desc,
                                                    dW_out_desc, dW_out_desc, dW_pref, 1,
                                                    &dW_heuristic, &returned) == HIPBLAS_STATUS_SUCCESS);
        if (returned == 0) {
            fprintf(stderr, "No cuBLASLt algorithm found for dW backward matmul\n");
            exit(EXIT_FAILURE);
        }

        CHECK_COND(hipblasLtMatmulDescCreate(&dx_matmul_desc, HIPBLAS_COMPUTE_32F, HIP_R_32F) == HIPBLAS_STATUS_SUCCESS);
        trans_a = HIPBLAS_OP_N;
        trans_b = HIPBLAS_OP_N;
        CHECK_COND(hipblasLtMatmulDescSetAttribute(dx_matmul_desc, HIPBLASLT_MATMUL_DESC_TRANSA,
                                                    &trans_a, sizeof(trans_a)) == HIPBLAS_STATUS_SUCCESS);
        CHECK_COND(hipblasLtMatmulDescSetAttribute(dx_matmul_desc, HIPBLASLT_MATMUL_DESC_TRANSB,
                                                    &trans_b, sizeof(trans_b)) == HIPBLAS_STATUS_SUCCESS);

        CHECK_COND(hipblasLtMatrixLayoutCreate(&dx_grad_desc, HIP_R_32F, M, N, M) == HIPBLAS_STATUS_SUCCESS);
        CHECK_COND(hipblasLtMatrixLayoutCreate(&dx_W_desc, HIP_R_32F, N, K, N) == HIPBLAS_STATUS_SUCCESS);
        CHECK_COND(hipblasLtMatrixLayoutCreate(&dx_out_desc, HIP_R_32F, M, K, M) == HIPBLAS_STATUS_SUCCESS);

        CHECK_COND(hipblasLtMatmulPreferenceCreate(&dx_pref) == HIPBLAS_STATUS_SUCCESS);
        CHECK_COND(hipblasLtMatmulPreferenceSetAttribute(dx_pref,
                                                          HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                          &workspace_size, sizeof(workspace_size)) == HIPBLAS_STATUS_SUCCESS);

        returned = 0;
        CHECK_COND(hipblasLtMatmulAlgoGetHeuristic(handle, dx_matmul_desc, dx_grad_desc, dx_W_desc,
                                                    dx_out_desc, dx_out_desc, dx_pref, 1,
                                                    &dx_heuristic, &returned) == HIPBLAS_STATUS_SUCCESS);
        if (returned == 0) {
            fprintf(stderr, "No cuBLASLt algorithm found for dx backward matmul\n");
            exit(EXIT_FAILURE);
        }

        initialized = true;
    }

    void destroy() {
        if (initialized) {
            hipblasLtMatmulPreferenceDestroy(dW_pref);
            hipblasLtMatrixLayoutDestroy(dW_grad_desc);
            hipblasLtMatrixLayoutDestroy(dW_x_desc);
            hipblasLtMatrixLayoutDestroy(dW_out_desc);
            hipblasLtMatmulDescDestroy(dW_matmul_desc);

            hipblasLtMatmulPreferenceDestroy(dx_pref);
            hipblasLtMatrixLayoutDestroy(dx_grad_desc);
            hipblasLtMatrixLayoutDestroy(dx_W_desc);
            hipblasLtMatrixLayoutDestroy(dx_out_desc);
            hipblasLtMatmulDescDestroy(dx_matmul_desc);

            hipblasLtDestroy(handle);
            hipFree(workspace);
            hipFree(grad_scaled_buffer);
            hipFree(K_buffer);
            hipFree(x_fp32_buffer);
            hipFree(W_fp32_buffer);
            initialized = false;
        }
    }

    void backward(float* dW, float* dx_out, const float* grad_output, const __hip_bfloat16* x,
                  const __hip_bfloat16* weight, const float* rms, hipStream_t stream = nullptr) {
        constexpr int BLOCK_SIZE = 256;

        int total_x = M * K;
        int num_blocks_x = (total_x + BLOCK_SIZE - 1) / BLOCK_SIZE;
        bf16_to_fp32_kernel<BLOCK_SIZE>
            <<<num_blocks_x, BLOCK_SIZE, 0, stream>>>(x_fp32_buffer, x, total_x);

        int total_w = N * K;
        int num_blocks_w = (total_w + BLOCK_SIZE - 1) / BLOCK_SIZE;
        bf16_to_fp32_kernel<BLOCK_SIZE>
            <<<num_blocks_w, BLOCK_SIZE, 0, stream>>>(W_fp32_buffer, weight, total_w);

        int total = M * N;
        int num_blocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
        scale_grad_by_rms_kernel<BLOCK_SIZE>
            <<<num_blocks, BLOCK_SIZE, 0, stream>>>(grad_scaled_buffer, grad_output, rms, M, N);

        float alpha = 1.0f, beta = 0.0f;
        CHECK_COND(hipblasLtMatmul(handle, dW_matmul_desc, &alpha, grad_scaled_buffer,
                                    dW_grad_desc, x_fp32_buffer, dW_x_desc, &beta, dW, dW_out_desc,
                                    dW, dW_out_desc, &dW_heuristic.algo, workspace, workspace_size,
                                    stream) == HIPBLAS_STATUS_SUCCESS);

        CHECK_COND(hipblasLtMatmul(handle, dx_matmul_desc, &alpha, grad_scaled_buffer,
                                    dx_grad_desc, W_fp32_buffer, dx_W_desc, &beta, K_buffer,
                                    dx_out_desc, K_buffer, dx_out_desc, &dx_heuristic.algo,
                                    workspace, workspace_size, stream) == HIPBLAS_STATUS_SUCCESS);

        int num_warps = BLOCK_SIZE / 32;
        size_t shared_mem = num_warps * sizeof(float);
        rms_correction_kernel<BLOCK_SIZE>
            <<<M, BLOCK_SIZE, shared_mem, stream>>>(dx_out, K_buffer, x, rms, M, K);
    }
};


constexpr int STREAM_MIX_TC_THRESHOLD = 32;

template<int BLOCK_SIZE>
__global__ void stream_add_kernel(float* __restrict__ out, const float* __restrict__ a,
                                  const float* __restrict__ b, int size) {
    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (idx < size) {
        out[idx] = a[idx] + b[idx];
    }
}

template<int BLOCK_SIZE, int MAX_N>
__global__ void stream_aggregate_bf16_fused_sigmoid_kernel(__hip_bfloat16* __restrict__ out,
                                                           float* __restrict__ H_pre_activated,
                                                           const float* __restrict__ inp,
                                                           const float* __restrict__ H_pre_raw,
                                                           int B, int n, int C) {
    __shared__ float s_H_pre[MAX_N];
    if (threadIdx.x < n) {
        float activated = fast_sigmoid(H_pre_raw[threadIdx.x]);
        s_H_pre[threadIdx.x] = activated;
        H_pre_activated[threadIdx.x] = activated;
    }
    __syncthreads();

    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (idx >= B * C)
        return;

    int b = idx / C, c = idx % C;
    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < MAX_N; i++) {
        if (i < n)
            sum += s_H_pre[i] * inp[b * n * C + i * C + c];
    }
    out[idx] = (__hip_bfloat16)sum;
}

template<int BLOCK_SIZE, int MAX_N>
__global__ void stream_aggregate_bf16_fused_sigmoid_vec4_kernel(__hip_bfloat16* __restrict__ out,
                                                                float* __restrict__ H_pre_activated,
                                                                const float* __restrict__ inp,
                                                                const float* __restrict__ H_pre_raw,
                                                                int B, int n, int C) {
    __shared__ float s_H_pre[MAX_N];
    if (threadIdx.x < n) {
        float activated = fast_sigmoid(H_pre_raw[threadIdx.x]);
        s_H_pre[threadIdx.x] = activated;
        H_pre_activated[threadIdx.x] = activated;
    }
    __syncthreads();

    int idx4 = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int C4 = C / 4;
    if (idx4 >= B * C4)
        return;

    int b = idx4 / C4;
    int c4 = idx4 % C4;
    int c = c4 * 4;

    float4 sum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
#pragma unroll
    for (int i = 0; i < MAX_N; i++) {
        if (i < n) {
            float h = s_H_pre[i];
            const float4* inp4 = reinterpret_cast<const float4*>(&inp[b * n * C + i * C + c]);
            float4 v = *inp4;
            sum.x += h * v.x;
            sum.y += h * v.y;
            sum.z += h * v.z;
            sum.w += h * v.w;
        }
    }
    __hip_bfloat162* out2 = reinterpret_cast<__hip_bfloat162*>(&out[b * C + c]);
    out2[0] = mhc_floats2bfloat162(sum.x, sum.y);
    out2[1] = mhc_floats2bfloat162(sum.z, sum.w);
}

template<int BLOCK_SIZE, int MAX_N>
__global__ void stream_distribute_from_bf16_fused_sigmoid_kernel(
    float* __restrict__ out, float* __restrict__ H_post_activated, const __hip_bfloat16* __restrict__ inp,
    const float* __restrict__ H_post_raw, int B, int n, int C) {
    __shared__ float s_H_post[MAX_N];
    if (threadIdx.x < n) {
        float activated = 2.0f * fast_sigmoid(H_post_raw[threadIdx.x]);
        s_H_post[threadIdx.x] = activated;
        H_post_activated[threadIdx.x] = activated;
    }
    __syncthreads();

    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (idx >= B * n * C)
        return;

    int b = idx / (n * C);
    int remainder = idx % (n * C);
    int stream = remainder / C;
    int c = remainder % C;
    out[idx] = s_H_post[stream] * (float)inp[b * C + c];
}

template<int BLOCK_SIZE, int MAX_N>
__global__ void stream_distribute_mix_add_fused_kernel(
    float* __restrict__ out, float* __restrict__ H_post_activated, const float* __restrict__ x_inp,
    const __hip_bfloat16* __restrict__ y_norm, const float* __restrict__ H_post_raw,
    const float* __restrict__ M, int B, int n, int C) {
    __shared__ float s_M[MAX_N * MAX_N];
    __shared__ float s_H_post[MAX_N];

    if (threadIdx.x < n * n)
        s_M[threadIdx.x] = M[threadIdx.x];
    if (threadIdx.x < n) {
        float activated = 2.0f * fast_sigmoid(H_post_raw[threadIdx.x]);
        s_H_post[threadIdx.x] = activated;
        H_post_activated[threadIdx.x] = activated;
    }
    __syncthreads();

    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (idx >= B * n * C)
        return;

    int b = idx / (n * C);
    int remainder = idx % (n * C);
    int i = remainder / C;
    int c = remainder % C;

    float mix_sum = 0.0f;
#pragma unroll
    for (int j = 0; j < MAX_N; j++) {
        if (j < n)
            mix_sum += s_M[i * n + j] * x_inp[b * n * C + j * C + c];
    }
    out[idx] = mix_sum + s_H_post[i] * (float)y_norm[b * C + c];
}

template<int BLOCK_SIZE, int MAX_N>
__global__ void stream_distribute_mix_add_fused_vec4_kernel(
    float* __restrict__ out, float* __restrict__ H_post_activated, const float* __restrict__ x_inp,
    const __hip_bfloat16* __restrict__ y_norm, const float* __restrict__ H_post_raw,
    const float* __restrict__ M, int B, int n, int C) {
    __shared__ float s_M[MAX_N * MAX_N];
    __shared__ float s_H_post[MAX_N];
    __shared__ float s_x_buf[2][MAX_N * 256];

    if (threadIdx.x < n * n)
        s_M[threadIdx.x] = M[threadIdx.x];
    if (threadIdx.x < n) {
        float activated = 2.0f * fast_sigmoid(H_post_raw[threadIdx.x]);
        s_H_post[threadIdx.x] = activated;
        H_post_activated[threadIdx.x] = activated;
    }
    __syncthreads();

    int vec_idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int C4 = C / 4;
    if (vec_idx >= B * n * C4)
        return;

    int b = vec_idx / (n * C4);
    int remainder = vec_idx % (n * C4);
    int i = remainder / C4;
    int c4 = remainder % C4;
    int c_base = c4 * 4;

    int buf_idx = 0;
    for (int j = 0; j < n; j++) {
        const float4* x_vec = reinterpret_cast<const float4*>(x_inp + b * n * C + j * C + c_base);
        float4 x = *x_vec;
        s_x_buf[buf_idx][j * BLOCK_SIZE * 4 + threadIdx.x * 4 + 0] = x.x;
        s_x_buf[buf_idx][j * BLOCK_SIZE * 4 + threadIdx.x * 4 + 1] = x.y;
        s_x_buf[buf_idx][j * BLOCK_SIZE * 4 + threadIdx.x * 4 + 2] = x.z;
        s_x_buf[buf_idx][j * BLOCK_SIZE * 4 + threadIdx.x * 4 + 3] = x.w;
    }
    __syncthreads();

    float4 result = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
#pragma unroll
    for (int j = 0; j < MAX_N; j++) {
        if (j < n) {
            float m_ij = s_M[i * n + j];
            result.x += m_ij * s_x_buf[buf_idx][j * BLOCK_SIZE * 4 + threadIdx.x * 4 + 0];
            result.y += m_ij * s_x_buf[buf_idx][j * BLOCK_SIZE * 4 + threadIdx.x * 4 + 1];
            result.z += m_ij * s_x_buf[buf_idx][j * BLOCK_SIZE * 4 + threadIdx.x * 4 + 2];
            result.w += m_ij * s_x_buf[buf_idx][j * BLOCK_SIZE * 4 + threadIdx.x * 4 + 3];
        }
    }

    const float4* y_vec = reinterpret_cast<const float4*>(
        reinterpret_cast<const __hip_bfloat16*>(y_norm) + b * C + c_base);
    __hip_bfloat16 y_bf16[4];
    *reinterpret_cast<float2*>(y_bf16) = *reinterpret_cast<const float2*>(y_vec);
    float h_i = s_H_post[i];
    result.x += h_i * __bfloat162float(y_bf16[0]);
    result.y += h_i * __bfloat162float(y_bf16[1]);
    result.z += h_i * __bfloat162float(y_bf16[2]);
    result.w += h_i * __bfloat162float(y_bf16[3]);

    float4* out_vec = reinterpret_cast<float4*>(out + b * n * C + i * C + c_base);
    *out_vec = result;
}

template<int BLOCK_SIZE, int MAX_N>
__global__ void
distribute_add_fused_kernel(float* __restrict__ out, float* __restrict__ H_post_activated,
                            const float* __restrict__ mix_out, const __hip_bfloat16* __restrict__ y_norm,
                            const float* __restrict__ H_post_raw, int B, int n, int C) {
    __shared__ float s_H_post[MAX_N];
    if (threadIdx.x < n) {
        float activated = 2.0f * fast_sigmoid(H_post_raw[threadIdx.x]);
        s_H_post[threadIdx.x] = activated;
        H_post_activated[threadIdx.x] = activated;
    }
    __syncthreads();

    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (idx >= B * n * C)
        return;

    int b = idx / (n * C);
    int remainder = idx % (n * C);
    int i = remainder / C;
    int c = remainder % C;

    float mix_val = mix_out[idx];
    float dist_val = s_H_post[i] * (float)y_norm[b * C + c];
    out[idx] = mix_val + dist_val;
}

class StreamMixTC {
  public:
    hipblasLtHandle_t handle;
    hipblasLtMatmulDesc_t matmulDesc;
    hipblasLtMatrixLayout_t Mdesc, Xdesc, Ydesc;
    hipblasLtMatmulPreference_t preference;
    hipblasLtMatmulHeuristicResult_t heuristic;
    void* workspace;
    size_t workspace_size;
    int B, n, C;
    bool initialized = false;

    void init(int B_, int n_, int C_) {
        B = B_;
        n = n_;
        C = C_;
        workspace_size = 4 * 1024 * 1024;

        hipblasLtCreate(&handle);
        hipblasLtMatmulDescCreate(&matmulDesc, HIPBLAS_COMPUTE_32F, HIP_R_32F);

        hipblasOperation_t trans_a = HIPBLAS_OP_N;
        hipblasOperation_t trans_b = HIPBLAS_OP_T;
        hipblasLtMatmulDescSetAttribute(matmulDesc, HIPBLASLT_MATMUL_DESC_TRANSA, &trans_a,
                                       sizeof(trans_a));
        hipblasLtMatmulDescSetAttribute(matmulDesc, HIPBLASLT_MATMUL_DESC_TRANSB, &trans_b,
                                       sizeof(trans_b));

        hipblasLtOrder_t row_order = HIPBLASLT_ORDER_ROW;
        hipblasLtMatrixLayoutCreate(&Xdesc, HIP_R_32F, B * C, n, n);
        hipblasLtMatrixLayoutSetAttribute(Xdesc, HIPBLASLT_MATRIX_LAYOUT_ORDER, &row_order,
                                         sizeof(row_order));
        hipblasLtMatrixLayoutCreate(&Mdesc, HIP_R_32F, n, n, n);
        hipblasLtMatrixLayoutSetAttribute(Mdesc, HIPBLASLT_MATRIX_LAYOUT_ORDER, &row_order,
                                         sizeof(row_order));
        hipblasLtMatrixLayoutCreate(&Ydesc, HIP_R_32F, B * C, n, n);
        hipblasLtMatrixLayoutSetAttribute(Ydesc, HIPBLASLT_MATRIX_LAYOUT_ORDER, &row_order,
                                         sizeof(row_order));

        hipblasLtMatmulPreferenceCreate(&preference);
        hipblasLtMatmulPreferenceSetAttribute(preference, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                             &workspace_size, sizeof(workspace_size));

        int returned_results = 0;
        hipblasLtMatmulAlgoGetHeuristic(handle, matmulDesc, Xdesc, Mdesc, Ydesc, Ydesc, preference,
                                       1, &heuristic, &returned_results);

        hipMalloc(&workspace, workspace_size);
        initialized = true;
    }

    void destroy() {
        if (!initialized)
            return;
        hipblasLtMatmulPreferenceDestroy(preference);
        hipblasLtMatrixLayoutDestroy(Mdesc);
        hipblasLtMatrixLayoutDestroy(Xdesc);
        hipblasLtMatrixLayoutDestroy(Ydesc);
        hipblasLtMatmulDescDestroy(matmulDesc);
        hipblasLtDestroy(handle);
        hipFree(workspace);
        initialized = false;
    }

    void forward(float* out, const float* inp, const float* M, hipStream_t stream = nullptr) {
        float alpha = 1.0f, beta = 0.0f;
        hipblasLtMatmul(handle, matmulDesc, &alpha, inp, Xdesc, M, Mdesc, &beta, out, Ydesc, out,
                       Ydesc, &heuristic.algo, workspace, workspace_size, stream);
    }

    void forward_fused_distribute_add(float* out, float* H_post_activated, const float* inp,
                                      const __hip_bfloat16* y_norm, const float* M, const float* H_post_raw,
                                      float* mix_out, hipStream_t stream = nullptr) {
        float alpha = 1.0f, beta = 0.0f;
        hipblasLtMatmul(handle, matmulDesc, &alpha, inp, Xdesc, M, Mdesc, &beta, mix_out, Ydesc,
                       mix_out, Ydesc, &heuristic.algo, workspace, workspace_size, stream);

        constexpr int BLOCK = 256, MAX_N = 64;
        int total = B * n * C;
        int blocks = (total + BLOCK - 1) / BLOCK;

        distribute_add_fused_kernel<BLOCK, MAX_N><<<blocks, BLOCK, 0, stream>>>(
            out, H_post_activated, mix_out, y_norm, H_post_raw, B, n, C);
    }
};

inline void stream_aggregate_bf16_fused_sigmoid(__hip_bfloat16* out, float* H_pre_activated,
                                                const float* inp, const float* H_pre_raw, int B,
                                                int n, int C, hipStream_t stream = nullptr) {
    constexpr int BLOCK = 256;

    // Use vectorized kernel when C is aligned to 4
    bool use_vec4 = (C % 4 == 0) && (C >= 64);

    if (use_vec4) {
        int blocks = (B * (C / 4) + BLOCK - 1) / BLOCK;
#define DISPATCH_AGGREGATE_VEC4(MAX_N_VAL)                                                         \
    stream_aggregate_bf16_fused_sigmoid_vec4_kernel<BLOCK, MAX_N_VAL>                              \
        <<<blocks, BLOCK, 0, stream>>>(out, H_pre_activated, inp, H_pre_raw, B, n, C)
        if (n <= 4) {
            DISPATCH_AGGREGATE_VEC4(4);
        } else if (n <= 8) {
            DISPATCH_AGGREGATE_VEC4(8);
        } else if (n <= 16) {
            DISPATCH_AGGREGATE_VEC4(16);
        } else if (n <= 32) {
            DISPATCH_AGGREGATE_VEC4(32);
        }
#undef DISPATCH_AGGREGATE_VEC4
    } else {
        int blocks = (B * C + BLOCK - 1) / BLOCK;
#define DISPATCH_AGGREGATE_FUSED(MAX_N_VAL)                                                        \
    stream_aggregate_bf16_fused_sigmoid_kernel<BLOCK, MAX_N_VAL>                                   \
        <<<blocks, BLOCK, 0, stream>>>(out, H_pre_activated, inp, H_pre_raw, B, n, C)
        if (n <= 4) {
            DISPATCH_AGGREGATE_FUSED(4);
        } else if (n <= 8) {
            DISPATCH_AGGREGATE_FUSED(8);
        } else if (n <= 16) {
            DISPATCH_AGGREGATE_FUSED(16);
        } else if (n <= 32) {
            DISPATCH_AGGREGATE_FUSED(32);
        } else {
            fprintf(stderr, "stream_aggregate_bf16_fused_sigmoid: n > 32 not implemented\n");
        }
#undef DISPATCH_AGGREGATE_FUSED
    }
}

inline void stream_distribute_mix_add_fused(float* out, float* H_post_activated, const float* x_inp,
                                            const __hip_bfloat16* y_norm, const float* H_post_raw,
                                            const float* M, int B, int n, int C,
                                            hipStream_t stream = nullptr) {
    constexpr int BLOCK = 256;
    int blocks = (B * n * C + BLOCK - 1) / BLOCK;

#define DISPATCH_MIX_ADD_FUSED(MAX_N_VAL)                                                          \
    stream_distribute_mix_add_fused_kernel<BLOCK, MAX_N_VAL><<<blocks, BLOCK, 0, stream>>>(        \
        out, H_post_activated, x_inp, y_norm, H_post_raw, M, B, n, C)

    if (n <= 4) {
        DISPATCH_MIX_ADD_FUSED(4);
    } else if (n <= 8) {
        DISPATCH_MIX_ADD_FUSED(8);
    } else if (n <= 16) {
        DISPATCH_MIX_ADD_FUSED(16);
    } else if (n <= 32) {
        DISPATCH_MIX_ADD_FUSED(32);
    } else {
        fprintf(stderr, "stream_distribute_mix_add_fused: n > 32 not implemented\n");
    }
#undef DISPATCH_MIX_ADD_FUSED
}

template<int BLOCK_SIZE, int MAX_N>
__global__ void
stream_aggregate_bf16_dynamic_kernel(__hip_bfloat16* __restrict__ out, const float* __restrict__ inp,
                                     const float* __restrict__ H_pre, int B, int n, int C) {
    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (idx >= B * C)
        return;

    int b = idx / C, c = idx % C;
    const float* h = H_pre + b * n;

    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < MAX_N; i++) {
        if (i < n)
            sum += h[i] * inp[b * n * C + i * C + c];
    }
    out[idx] = (__hip_bfloat16)sum;
}

template<int BLOCK_SIZE, int MAX_N>
__global__ void
stream_aggregate_bf16_dynamic_vec4_kernel(__hip_bfloat16* __restrict__ out, const float* __restrict__ inp,
                                          const float* __restrict__ H_pre, int B, int n, int C) {
    int idx4 = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int C4 = C / 4;
    if (idx4 >= B * C4)
        return;

    int b = idx4 / C4;
    int c4 = idx4 % C4;
    int c = c4 * 4;
    const float* h = H_pre + b * n;

    float4 sum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
#pragma unroll
    for (int i = 0; i < MAX_N; i++) {
        if (i < n) {
            float hi = h[i];
            const float4* inp4 = reinterpret_cast<const float4*>(&inp[b * n * C + i * C + c]);
            float4 v = *inp4;
            sum.x += hi * v.x;
            sum.y += hi * v.y;
            sum.z += hi * v.z;
            sum.w += hi * v.w;
        }
    }
    __hip_bfloat162* out2 = reinterpret_cast<__hip_bfloat162*>(&out[b * C + c]);
    out2[0] = mhc_floats2bfloat162(sum.x, sum.y);
    out2[1] = mhc_floats2bfloat162(sum.z, sum.w);
}

template<int BLOCK_SIZE, int MAX_N>
__global__ void stream_distribute_mix_add_dynamic_kernel(
    float* __restrict__ out, const float* __restrict__ x_inp, const __hip_bfloat16* __restrict__ y_norm,
    const float* __restrict__ H_post, const float* __restrict__ M, int B, int n, int C) {
    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (idx >= B * n * C)
        return;

    int b = idx / (n * C);
    int remainder = idx % (n * C);
    int i = remainder / C;
    int c = remainder % C;

    const float* h = H_post + b * n;
    const float* m = M + b * n * n;

    float mix_sum = 0.0f;
#pragma unroll
    for (int j = 0; j < MAX_N; j++) {
        if (j < n)
            mix_sum += m[i * n + j] * x_inp[b * n * C + j * C + c];
    }
    out[idx] = mix_sum + h[i] * (float)y_norm[b * C + c];
}

inline void stream_aggregate_bf16_dynamic(__hip_bfloat16* out, const float* inp, const float* H_pre, int B,
                                          int n, int C, hipStream_t stream = nullptr) {
    constexpr int BLOCK = 256;
    bool use_vec4 = (C % 4 == 0) && (C >= 64);

    if (use_vec4) {
        int blocks = (B * (C / 4) + BLOCK - 1) / BLOCK;
#define DISPATCH_AGGREGATE_DYN_VEC4(MAX_N_VAL)                                                     \
    stream_aggregate_bf16_dynamic_vec4_kernel<BLOCK, MAX_N_VAL>                                    \
        <<<blocks, BLOCK, 0, stream>>>(out, inp, H_pre, B, n, C)
        if (n <= 4) {
            DISPATCH_AGGREGATE_DYN_VEC4(4);
        } else if (n <= 8) {
            DISPATCH_AGGREGATE_DYN_VEC4(8);
        } else if (n <= 16) {
            DISPATCH_AGGREGATE_DYN_VEC4(16);
        } else if (n <= 32) {
            DISPATCH_AGGREGATE_DYN_VEC4(32);
        }
#undef DISPATCH_AGGREGATE_DYN_VEC4
    } else {
        int blocks = (B * C + BLOCK - 1) / BLOCK;
#define DISPATCH_AGGREGATE_DYN(MAX_N_VAL)                                                          \
    stream_aggregate_bf16_dynamic_kernel<BLOCK, MAX_N_VAL>                                         \
        <<<blocks, BLOCK, 0, stream>>>(out, inp, H_pre, B, n, C)
        if (n <= 4) {
            DISPATCH_AGGREGATE_DYN(4);
        } else if (n <= 8) {
            DISPATCH_AGGREGATE_DYN(8);
        } else if (n <= 16) {
            DISPATCH_AGGREGATE_DYN(16);
        } else if (n <= 32) {
            DISPATCH_AGGREGATE_DYN(32);
        } else {
            fprintf(stderr, "stream_aggregate_bf16_dynamic: n > 32 not implemented\n");
        }
#undef DISPATCH_AGGREGATE_DYN
    }
}

inline void stream_distribute_mix_add_fused_dynamic(float* out, const float* x_inp,
                                                    const __hip_bfloat16* y_norm, const float* H_post,
                                                    const float* M, int B, int n, int C,
                                                    hipStream_t stream = nullptr) {
    constexpr int BLOCK = 256;
    int blocks = (B * n * C + BLOCK - 1) / BLOCK;

#define DISPATCH_MIX_ADD_DYN(MAX_N_VAL)                                                            \
    stream_distribute_mix_add_dynamic_kernel<BLOCK, MAX_N_VAL>                                     \
        <<<blocks, BLOCK, 0, stream>>>(out, x_inp, y_norm, H_post, M, B, n, C)

    if (n <= 4) {
        DISPATCH_MIX_ADD_DYN(4);
    } else if (n <= 8) {
        DISPATCH_MIX_ADD_DYN(8);
    } else if (n <= 16) {
        DISPATCH_MIX_ADD_DYN(16);
    } else if (n <= 32) {
        DISPATCH_MIX_ADD_DYN(32);
    } else {
        fprintf(stderr, "stream_distribute_mix_add_fused_dynamic: n > 32 not implemented\n");
    }
#undef DISPATCH_MIX_ADD_DYN
}

template<int BLOCK_SIZE, int MAX_N>
__global__ void
stream_aggregate_backward_dx_kernel(float* __restrict__ d_inp, const float* __restrict__ grad,
                                    const float* __restrict__ H_pre, int B, int n, int C) {
    __shared__ float s_H_pre[MAX_N];
    if (threadIdx.x < n)
        s_H_pre[threadIdx.x] = H_pre[threadIdx.x];
    __syncthreads();

    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (idx >= B * n * C)
        return;

    int b = idx / (n * C);
    int remainder = idx % (n * C);
    int i = remainder / C;
    int c = remainder % C;
    d_inp[idx] = grad[b * C + c] * s_H_pre[i];
}

template<int BLOCK_SIZE, int MAX_N>
__global__ void stream_aggregate_backward_dH_partial_kernel(float* __restrict__ partials,
                                                            const float* __restrict__ grad,
                                                            const float* __restrict__ inp, int B,
                                                            int n, int C) {
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    __shared__ float s_warp_sums[MAX_N][BLOCK_SIZE / 32];

    float local_sum[MAX_N] = {0.0f};
    for (int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x; idx < B * C;
         idx += gridDim.x * BLOCK_SIZE) {
        int b = idx / C, c = idx % C;
        float g = grad[idx];
#pragma unroll
        for (int i = 0; i < MAX_N; i++) {
            if (i < n)
                local_sum[i] += g * inp[b * n * C + i * C + c];
        }
    }

    int warp_id = threadIdx.x / 32;
#pragma unroll
    for (int i = 0; i < MAX_N; i++) {
        if (i < n) {
            float warp_sum = cg::reduce(warp, local_sum[i], cg::plus<float>());
            if (warp.thread_rank() == 0)
                s_warp_sums[i][warp_id] = warp_sum;
        }
    }
    block.sync();

    if (threadIdx.x < n) {
        float block_sum = 0.0f;
        for (int w = 0; w < BLOCK_SIZE / 32; w++)
            block_sum += s_warp_sums[threadIdx.x][w];
        partials[blockIdx.x * n + threadIdx.x] = block_sum;
    }
}

template<int MAX_N>
__global__ void reduce_partials_kernel(float* __restrict__ out, const float* __restrict__ partials,
                                       int n, int num_partials) {
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    int i = blockIdx.x;
    if (i >= n)
        return;

    float sum = 0.0f;
    for (int p = threadIdx.x; p < num_partials; p += blockDim.x)
        sum += partials[p * n + i];
    sum = cg::reduce(warp, sum, cg::plus<float>());

    __shared__ float s_warp_sums[8];
    if (warp.thread_rank() == 0)
        s_warp_sums[threadIdx.x / 32] = sum;
    block.sync();

    if (threadIdx.x == 0) {
        float total = 0.0f;
        for (int w = 0; w < (blockDim.x + 31) / 32; w++)
            total += s_warp_sums[w];
        out[i] = total;
    }
}

inline void stream_aggregate_backward(float* d_inp, float* d_H_pre, const float* grad,
                                      const float* inp, const float* H_pre, int B, int n, int C,
                                      float* workspace, int workspace_num_blocks,
                                      hipStream_t stream = nullptr) {
    constexpr int BLOCK = 256;
    int blocks_dx = (B * n * C + BLOCK - 1) / BLOCK;

#define DISPATCH_AGG_BWD(MAX_N_VAL)                                                                \
    stream_aggregate_backward_dx_kernel<BLOCK, MAX_N_VAL>                                          \
        <<<blocks_dx, BLOCK, 0, stream>>>(d_inp, grad, H_pre, B, n, C);                            \
    stream_aggregate_backward_dH_partial_kernel<BLOCK, MAX_N_VAL>                                  \
        <<<workspace_num_blocks, BLOCK, 0, stream>>>(workspace, grad, inp, B, n, C);               \
    reduce_partials_kernel<MAX_N_VAL>                                                              \
        <<<n, 128, 0, stream>>>(d_H_pre, workspace, n, workspace_num_blocks)

    if (n <= 4) {
        DISPATCH_AGG_BWD(4);
    } else if (n <= 8) {
        DISPATCH_AGG_BWD(8);
    } else if (n <= 16) {
        DISPATCH_AGG_BWD(16);
    } else if (n <= 32) {
        DISPATCH_AGG_BWD(32);
    } else {
        fprintf(stderr, "stream_aggregate_backward: n > 32 not implemented\n");
    }
#undef DISPATCH_AGG_BWD
}

template<int BLOCK_SIZE, int MAX_N>
__global__ void stream_distribute_mix_backward_dx_dy_kernel(
    float* __restrict__ d_x, float* __restrict__ d_y_norm, const float* __restrict__ grad,
    const float* __restrict__ M, const float* __restrict__ H_post, int B, int n, int C) {
    __shared__ float s_M[MAX_N * MAX_N];
    __shared__ float s_H[MAX_N];

    if (threadIdx.x < n * n)
        s_M[threadIdx.x] = M[threadIdx.x];
    if (threadIdx.x < n)
        s_H[threadIdx.x] = H_post[threadIdx.x];
    __syncthreads();

    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (idx >= B * n * C)
        return;

    int b = idx / (n * C);
    int remainder = idx % (n * C);
    int j = remainder / C;
    int c = remainder % C;

    float dx_sum = 0.0f;
#pragma unroll
    for (int i = 0; i < MAX_N; i++) {
        if (i < n)
            dx_sum += s_M[i * n + j] * grad[b * n * C + i * C + c];
    }
    d_x[idx] = dx_sum;

    if (j == 0) {
        float dy_sum = 0.0f;
#pragma unroll
        for (int i = 0; i < MAX_N; i++) {
            if (i < n)
                dy_sum += s_H[i] * grad[b * n * C + i * C + c];
        }
        d_y_norm[b * C + c] = dy_sum;
    }
}

template<int BLOCK_SIZE, int MAX_N>
__global__ void stream_distribute_mix_backward_dx_dy_vec4_kernel(
    float* __restrict__ d_x, float* __restrict__ d_y_norm, const float* __restrict__ grad,
    const float* __restrict__ M, const float* __restrict__ H_post, int B, int n, int C) {
    __shared__ float s_M[MAX_N * MAX_N];
    __shared__ float s_H[MAX_N];

    if (threadIdx.x < n * n)
        s_M[threadIdx.x] = M[threadIdx.x];
    if (threadIdx.x < n)
        s_H[threadIdx.x] = H_post[threadIdx.x];
    __syncthreads();

    int vec_idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int C4 = C / 4;
    if (vec_idx >= B * n * C4)
        return;

    int b = vec_idx / (n * C4);
    int remainder = vec_idx % (n * C4);
    int j = remainder / C4;
    int c_base = (remainder % C4) * 4;

    float4 dx_acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 dy_acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

#pragma unroll
    for (int i = 0; i < MAX_N; i++) {
        if (i < n) {
            float4 g = *reinterpret_cast<const float4*>(grad + b * n * C + i * C + c_base);
            float m_ij = s_M[i * n + j];
            dx_acc.x += m_ij * g.x;
            dx_acc.y += m_ij * g.y;
            dx_acc.z += m_ij * g.z;
            dx_acc.w += m_ij * g.w;
            if (j == 0) {
                float h_i = s_H[i];
                dy_acc.x += h_i * g.x;
                dy_acc.y += h_i * g.y;
                dy_acc.z += h_i * g.z;
                dy_acc.w += h_i * g.w;
            }
        }
    }

    *reinterpret_cast<float4*>(d_x + b * n * C + j * C + c_base) = dx_acc;
    if (j == 0)
        *reinterpret_cast<float4*>(d_y_norm + b * C + c_base) = dy_acc;
}

template<int BLOCK_SIZE, int MAX_N>
__global__ void stream_distribute_mix_backward_partials_kernel(
    float* __restrict__ partials_M, float* __restrict__ partials_H, const float* __restrict__ grad,
    const float* __restrict__ x, const float* __restrict__ y_norm, int B, int n, int C) {
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    constexpr int NUM_WARPS = BLOCK_SIZE / 32;
    __shared__ float s_warp_M[MAX_N][MAX_N][NUM_WARPS];
    __shared__ float s_warp_H[MAX_N][NUM_WARPS];

    float local_M[MAX_N][MAX_N];
    float local_H[MAX_N];
#pragma unroll
    for (int i = 0; i < MAX_N; i++) {
        local_H[i] = 0.0f;
#pragma unroll
        for (int j = 0; j < MAX_N; j++)
            local_M[i][j] = 0.0f;
    }

    for (int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x; idx < B * C;
         idx += gridDim.x * BLOCK_SIZE) {
        int b = idx / C, c = idx % C;
        float y_val = y_norm[b * C + c];
#pragma unroll
        for (int i = 0; i < MAX_N; i++) {
            if (i < n) {
                float g = grad[b * n * C + i * C + c];
                local_H[i] += g * y_val;
#pragma unroll
                for (int j = 0; j < MAX_N; j++) {
                    if (j < n)
                        local_M[i][j] += g * x[b * n * C + j * C + c];
                }
            }
        }
    }

    int warp_id = threadIdx.x / 32;
#pragma unroll
    for (int i = 0; i < MAX_N; i++) {
        if (i < n) {
#pragma unroll
            for (int j = 0; j < MAX_N; j++) {
                if (j < n) {
                    float ws = cg::reduce(warp, local_M[i][j], cg::plus<float>());
                    if (warp.thread_rank() == 0)
                        s_warp_M[i][j][warp_id] = ws;
                }
            }
        }
    }
#pragma unroll
    for (int i = 0; i < MAX_N; i++) {
        if (i < n) {
            float ws = cg::reduce(warp, local_H[i], cg::plus<float>());
            if (warp.thread_rank() == 0)
                s_warp_H[i][warp_id] = ws;
        }
    }
    block.sync();

    if (threadIdx.x < n * n) {
        int i = threadIdx.x / n, j = threadIdx.x % n;
        float bs = 0.0f;
        for (int w = 0; w < NUM_WARPS; w++)
            bs += s_warp_M[i][j][w];
        partials_M[blockIdx.x * n * n + threadIdx.x] = bs;
    }
    if (threadIdx.x < n) {
        float bs = 0.0f;
        for (int w = 0; w < NUM_WARPS; w++)
            bs += s_warp_H[threadIdx.x][w];
        partials_H[blockIdx.x * n + threadIdx.x] = bs;
    }
}

template<int MAX_N>
__global__ void reduce_partials_matrix_kernel(float* __restrict__ out,
                                              const float* __restrict__ partials, int n,
                                              int num_partials) {
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    int k = blockIdx.x;
    if (k >= n * n)
        return;

    float sum = 0.0f;
    for (int p = threadIdx.x; p < num_partials; p += blockDim.x)
        sum += partials[p * n * n + k];
    sum = cg::reduce(warp, sum, cg::plus<float>());

    __shared__ float s_warp_sums[8];
    if (warp.thread_rank() == 0)
        s_warp_sums[threadIdx.x / 32] = sum;
    block.sync();

    if (threadIdx.x == 0) {
        float total = 0.0f;
        for (int w = 0; w < (blockDim.x + 31) / 32; w++)
            total += s_warp_sums[w];
        out[k] = total;
    }
}

inline void stream_distribute_mix_backward_fused(float* d_x, float* d_y_norm, float* d_M,
                                                 float* d_H_post, const float* grad, const float* x,
                                                 const float* y_norm, const float* M,
                                                 const float* H_post, int B, int n, int C,
                                                 float* workspace_M, float* workspace_H,
                                                 int workspace_num_blocks,
                                                 hipStream_t stream = nullptr) {
    constexpr int BLOCK = 256;

#define DISPATCH_DIST_BWD(MAX_N_VAL)                                                               \
    do {                                                                                           \
        if (C % 4 == 0 && C >= 64 && n <= 8) {                                                     \
            int blocks = (B * n * (C / 4) + BLOCK - 1) / BLOCK;                                    \
            stream_distribute_mix_backward_dx_dy_vec4_kernel<BLOCK, MAX_N_VAL>                     \
                <<<blocks, BLOCK, 0, stream>>>(d_x, d_y_norm, grad, M, H_post, B, n, C);           \
        } else {                                                                                   \
            int blocks = (B * n * C + BLOCK - 1) / BLOCK;                                          \
            stream_distribute_mix_backward_dx_dy_kernel<BLOCK, MAX_N_VAL>                          \
                <<<blocks, BLOCK, 0, stream>>>(d_x, d_y_norm, grad, M, H_post, B, n, C);           \
        }                                                                                          \
        stream_distribute_mix_backward_partials_kernel<BLOCK, MAX_N_VAL>                           \
            <<<workspace_num_blocks, BLOCK, 0, stream>>>(workspace_M, workspace_H, grad, x,        \
                                                         y_norm, B, n, C);                         \
        reduce_partials_matrix_kernel<MAX_N_VAL>                                                   \
            <<<n * n, 128, 0, stream>>>(d_M, workspace_M, n, workspace_num_blocks);                \
        reduce_partials_kernel<MAX_N_VAL>                                                          \
            <<<n, 128, 0, stream>>>(d_H_post, workspace_H, n, workspace_num_blocks);               \
    } while (0)

    if (n <= 4) {
        DISPATCH_DIST_BWD(4);
    } else if (n <= 8) {
        DISPATCH_DIST_BWD(8);
    } else if (n <= 16) {
        DISPATCH_DIST_BWD(16);
    } else if (n <= 32) {
        DISPATCH_DIST_BWD(32);
    } else {
        fprintf(stderr, "stream_distribute_mix_backward_fused: n > 32 not implemented\n");
    }
#undef DISPATCH_DIST_BWD
}

struct MHCLayerConfig {
    int batch_size;
    int hidden_dim;
    int expansion_rate;
    int sinkhorn_iters;
    float eps;
    float alpha_init;
    bool use_pdl;
    bool use_dynamic_h;

    MHCLayerConfig()
        : batch_size(0), hidden_dim(0), expansion_rate(4), sinkhorn_iters(20), eps(1e-5f),
          alpha_init(0.01f), use_pdl(true), use_dynamic_h(true) {}
};

struct MHCLayerWeights {
    __hip_bfloat16* rmsnorm_weight;

    __hip_bfloat16* phi_combined;
    __hip_bfloat16* phi_pre;
    __hip_bfloat16* phi_post;
    __hip_bfloat16* phi_res;

    float* b_pre;
    float* b_post;
    float* b_res;

    float alpha_pre;
    float alpha_post;
    float alpha_res;

    bool initialized;
    bool dynamic_h;
    int hidden_dim;
    int expansion_rate;

    MHCLayerWeights() : initialized(false), dynamic_h(true), phi_combined(nullptr) {}

    void init(int C, int n, bool use_dynamic = true, float alpha_init = 0.01f) {
        hidden_dim = C;
        expansion_rate = n;
        dynamic_h = use_dynamic;

        HIP_CALL(hipMalloc(&rmsnorm_weight, C * sizeof(__hip_bfloat16)));

        if (dynamic_h) {
            int nC = n * C;
            int total_H_dim = n + n + n * n;
            HIP_CALL(hipMalloc(&phi_combined, total_H_dim * nC * sizeof(__hip_bfloat16)));
            phi_pre = phi_combined;
            phi_post = phi_combined + n * nC;
            phi_res = phi_combined + 2 * n * nC;
        } else {
            phi_combined = nullptr;
            phi_pre = nullptr;
            phi_post = nullptr;
            phi_res = nullptr;
        }

        HIP_CALL(hipMalloc(&b_pre, n * sizeof(float)));
        HIP_CALL(hipMalloc(&b_post, n * sizeof(float)));
        HIP_CALL(hipMalloc(&b_res, n * n * sizeof(float)));

        alpha_pre = alpha_init;
        alpha_post = alpha_init;
        alpha_res = alpha_init;

        initialized = true;
    }

    void destroy() {
        if (!initialized)
            return;

        hipFree(rmsnorm_weight);
        if (dynamic_h) {
            hipFree(phi_combined);
        }
        hipFree(b_pre);
        hipFree(b_post);
        hipFree(b_res);

        initialized = false;
    }
};

struct MHCLayerBuffers {
    float* x_expanded;
    __hip_bfloat16* x_aggregated_bf16;
    float* x_aggregated_f32;
    float* rms_values;
    __hip_bfloat16* layer_out_bf16;
    float* layer_out_f32;
    float* y_distributed;
    float* sinkhorn_M;
    float* x_mixed;
    float* output;

    __hip_bfloat16* x_flat_bf16;
    float* rms_dynamic;
    float* H_proj_raw;

    float* H_pre_activated;
    float* H_post_activated;
    float* H_res_tilde;

    FusedRMSNormMatmul fused_rms_matmul;

    bool initialized;
    bool dynamic_h;
    int batch_size;
    int hidden_dim;
    int expansion_rate;

    MHCLayerBuffers() : initialized(false), x_mixed(nullptr), dynamic_h(true) {}

    void init(int B, int C, int n, bool needs_x_mixed = false, bool use_dynamic_h = true) {
        batch_size = B;
        hidden_dim = C;
        expansion_rate = n;
        dynamic_h = use_dynamic_h;

        HIP_CALL(hipMalloc(&x_expanded, B * n * C * sizeof(float)));
        HIP_CALL(hipMalloc(&x_aggregated_bf16, B * C * sizeof(__hip_bfloat16)));
        HIP_CALL(hipMalloc(&x_aggregated_f32, B * C * sizeof(float)));
        HIP_CALL(hipMalloc(&rms_values, B * sizeof(float)));
        HIP_CALL(hipMalloc(&layer_out_bf16, B * C * sizeof(__hip_bfloat16)));
        HIP_CALL(hipMalloc(&layer_out_f32, B * C * sizeof(float)));
        HIP_CALL(hipMalloc(&y_distributed, B * n * C * sizeof(float)));
        if (needs_x_mixed) {
            HIP_CALL(hipMalloc(&x_mixed, B * n * C * sizeof(float)));
        }
        HIP_CALL(hipMalloc(&output, B * n * C * sizeof(float)));

        if (dynamic_h) {
            int nC = n * C;
            int total_H_dim = n + n + n * n;
            HIP_CALL(hipMalloc(&x_flat_bf16, B * nC * sizeof(__hip_bfloat16)));
            HIP_CALL(hipMalloc(&rms_dynamic, B * sizeof(float)));
            HIP_CALL(hipMalloc(&H_proj_raw, B * total_H_dim * sizeof(float)));
            fused_rms_matmul.init(B, total_H_dim, nC);
            HIP_CALL(hipMalloc(&sinkhorn_M, B * n * n * sizeof(float)));
            HIP_CALL(hipMalloc(&H_pre_activated, B * n * sizeof(float)));
            HIP_CALL(hipMalloc(&H_post_activated, B * n * sizeof(float)));
            HIP_CALL(hipMalloc(&H_res_tilde, B * n * n * sizeof(float)));
        } else {
            x_flat_bf16 = nullptr;
            rms_dynamic = nullptr;
            H_proj_raw = nullptr;
            HIP_CALL(hipMalloc(&sinkhorn_M, n * n * sizeof(float)));
            HIP_CALL(hipMalloc(&H_pre_activated, n * sizeof(float)));
            HIP_CALL(hipMalloc(&H_post_activated, n * sizeof(float)));
            HIP_CALL(hipMalloc(&H_res_tilde, n * n * sizeof(float)));
        }

        initialized = true;
    }

    void destroy() {
        if (!initialized)
            return;

        hipFree(x_expanded);
        hipFree(x_aggregated_bf16);
        hipFree(x_aggregated_f32);
        hipFree(rms_values);
        hipFree(layer_out_bf16);
        hipFree(layer_out_f32);
        hipFree(y_distributed);
        hipFree(sinkhorn_M);
        if (x_mixed)
            hipFree(x_mixed);
        hipFree(output);

        if (dynamic_h) {
            hipFree(x_flat_bf16);
            hipFree(rms_dynamic);
            hipFree(H_proj_raw);
            fused_rms_matmul.destroy();
        }

        hipFree(H_pre_activated);
        hipFree(H_post_activated);
        hipFree(H_res_tilde);

        initialized = false;
    }
};

struct MHCLayerGradients {
    float* d_x_expanded;
    float* d_H_pre;
    float* d_rmsnorm_weight;
    float* d_H_post;
    float* d_H_res;
    float* d_x_aggregated;
    float* d_layer_out;
    float* d_y_distributed;
    float* d_x_mixed;
    float* d_M;

    float* d_H_pre_activated;
    float* d_H_post_activated;
    float* d_H_res_exp;

    float* workspace_dH;
    float* workspace_dM;
    int workspace_num_blocks;

    bool initialized;

    MHCLayerGradients() : initialized(false), workspace_dH(nullptr), workspace_dM(nullptr) {}

    void init(int B, int C, int n) {
        HIP_CALL(hipMalloc(&d_x_expanded, B * n * C * sizeof(float)));
        HIP_CALL(hipMalloc(&d_H_pre, n * sizeof(float)));
        HIP_CALL(hipMalloc(&d_rmsnorm_weight, C * sizeof(float)));
        HIP_CALL(hipMalloc(&d_H_post, n * sizeof(float)));
        HIP_CALL(hipMalloc(&d_H_res, n * n * sizeof(float)));
        HIP_CALL(hipMalloc(&d_x_aggregated, B * C * sizeof(float)));
        HIP_CALL(hipMalloc(&d_layer_out, B * C * sizeof(float)));
        HIP_CALL(hipMalloc(&d_y_distributed, B * n * C * sizeof(float)));
        HIP_CALL(hipMalloc(&d_x_mixed, B * n * C * sizeof(float)));
        HIP_CALL(hipMalloc(&d_M, n * n * sizeof(float)));

        HIP_CALL(hipMalloc(&d_H_pre_activated, n * sizeof(float)));
        HIP_CALL(hipMalloc(&d_H_post_activated, n * sizeof(float)));
        HIP_CALL(hipMalloc(&d_H_res_exp, n * n * sizeof(float)));

        constexpr int BLOCK_SIZE = 256;
        workspace_num_blocks = min(128, (B * C + BLOCK_SIZE - 1) / BLOCK_SIZE);
        HIP_CALL(hipMalloc(&workspace_dH, workspace_num_blocks * n * sizeof(float)));
        HIP_CALL(hipMalloc(&workspace_dM, workspace_num_blocks * n * n * sizeof(float)));

        initialized = true;
    }

    void destroy() {
        if (!initialized)
            return;

        hipFree(d_x_expanded);
        hipFree(d_H_pre);
        hipFree(d_rmsnorm_weight);
        hipFree(d_H_post);
        hipFree(d_H_res);
        hipFree(d_x_aggregated);
        hipFree(d_layer_out);
        hipFree(d_y_distributed);
        hipFree(d_x_mixed);
        hipFree(d_M);

        hipFree(d_H_pre_activated);
        hipFree(d_H_post_activated);
        hipFree(d_H_res_exp);

        hipFree(workspace_dH);
        hipFree(workspace_dM);

        initialized = false;
    }

    void zero_weight_grads(int C, int n, hipStream_t stream = nullptr) {
        HIP_CALL(hipMemsetAsync(d_H_pre, 0, n * sizeof(float), stream));
        HIP_CALL(hipMemsetAsync(d_rmsnorm_weight, 0, C * sizeof(float), stream));
        HIP_CALL(hipMemsetAsync(d_H_post, 0, n * sizeof(float), stream));
        HIP_CALL(hipMemsetAsync(d_H_res, 0, n * n * sizeof(float), stream));
    }
};

template<int BLOCK_SIZE>
__global__ void sigmoid_kernel(float* __restrict__ out, const float* __restrict__ inp, int size) {
    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (idx < size) {
        float x = inp[idx];
        out[idx] = fast_sigmoid(-x);
    }
}

template<int BLOCK_SIZE>
__global__ void sigmoid_scale_kernel(float* __restrict__ out, const float* __restrict__ inp,
                                     float scale, int size) {
    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (idx < size) {
        float x = inp[idx];
        out[idx] = scale / (1.0f + expf(-x));
    }
}

template<int BLOCK_SIZE>
__global__ void exp_kernel(float* __restrict__ out, const float* __restrict__ inp, int size) {
    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (idx < size) {
        out[idx] = fast_exp(inp[idx]);
    }
}

template<int BLOCK_SIZE>
__global__ void sigmoid_backward_kernel(float* __restrict__ d_inp, const float* __restrict__ d_out,
                                        const float* __restrict__ activated, int size) {
    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (idx < size) {
        float s = activated[idx];
        d_inp[idx] = d_out[idx] * s * (1.0f - s);
    }
}

template<int BLOCK_SIZE>
__global__ void
sigmoid_scale_backward_kernel(float* __restrict__ d_inp, const float* __restrict__ d_out,
                              const float* __restrict__ activated, float scale, int size) {
    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (idx < size) {
        float s = activated[idx] / scale;
        d_inp[idx] = d_out[idx] * scale * s * (1.0f - s);
    }
}

template<int BLOCK_SIZE>
__global__ void exp_backward_kernel(float* __restrict__ d_inp, const float* __restrict__ d_out,
                                    const float* __restrict__ exp_val, int size) {
    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (idx < size) {
        d_inp[idx] = d_out[idx] * exp_val[idx];
    }
}

inline void apply_exp(float* out, const float* inp, int size, hipStream_t stream = nullptr) {
    constexpr int BLOCK_SIZE = 256;
    int num_blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    exp_kernel<BLOCK_SIZE><<<num_blocks, BLOCK_SIZE, 0, stream>>>(out, inp, size);
}

inline void sigmoid_backward(float* d_inp, const float* d_out, const float* activated, int size,
                             hipStream_t stream = nullptr) {
    constexpr int BLOCK_SIZE = 256;
    int num_blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    sigmoid_backward_kernel<BLOCK_SIZE>
        <<<num_blocks, BLOCK_SIZE, 0, stream>>>(d_inp, d_out, activated, size);
}

inline void sigmoid_scale_backward(float* d_inp, const float* d_out, const float* activated,
                                   float scale, int size, hipStream_t stream = nullptr) {
    constexpr int BLOCK_SIZE = 256;
    int num_blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    sigmoid_scale_backward_kernel<BLOCK_SIZE>
        <<<num_blocks, BLOCK_SIZE, 0, stream>>>(d_inp, d_out, activated, scale, size);
}

inline void exp_backward(float* d_inp, const float* d_out, const float* exp_val, int size,
                         hipStream_t stream = nullptr) {
    constexpr int BLOCK_SIZE = 256;
    int num_blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    exp_backward_kernel<BLOCK_SIZE>
        <<<num_blocks, BLOCK_SIZE, 0, stream>>>(d_inp, d_out, exp_val, size);
}

template<int BLOCK_SIZE>
__global__ void apply_dynamic_h_activations_kernel(
    float* __restrict__ H_pre_out, float* __restrict__ H_post_out, float* __restrict__ H_res_out,
    const float* __restrict__ H_proj_raw, const float* __restrict__ b_pre,
    const float* __restrict__ b_post, const float* __restrict__ b_res, float alpha_pre,
    float alpha_post, float alpha_res, int B, int n) {
    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int n2 = n * n;
    int total_H_dim = n + n + n2;

    for (int b = idx; b < B; b += gridDim.x * BLOCK_SIZE) {
        const float* proj = H_proj_raw + b * total_H_dim;
        float* h_pre = H_pre_out + b * n;
        float* h_post = H_post_out + b * n;
        float* h_res = H_res_out + b * n2;

        for (int i = 0; i < n; i++) {
            float val = alpha_pre * proj[i] + b_pre[i];
            h_pre[i] = fast_sigmoid(val);
        }

        for (int i = 0; i < n; i++) {
            float val = alpha_post * proj[n + i] + b_post[i];
            h_post[i] = 2.0f * fast_sigmoid(val);
        }

        for (int i = 0; i < n2; i++) {
            float val = alpha_res * proj[2 * n + i] + b_res[i];
            h_res[i] = val;
        }
    }
}

inline void apply_dynamic_h_activations(float* H_pre_out, float* H_post_out, float* H_res_out,
                                        const float* H_proj_raw, const float* b_pre,
                                        const float* b_post, const float* b_res, float alpha_pre,
                                        float alpha_post, float alpha_res, int B, int n,
                                        hipStream_t stream = nullptr) {
    constexpr int BLOCK_SIZE = 256;
    int num_blocks = (B + BLOCK_SIZE - 1) / BLOCK_SIZE;
    apply_dynamic_h_activations_kernel<BLOCK_SIZE><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
        H_pre_out, H_post_out, H_res_out, H_proj_raw, b_pre, b_post, b_res, alpha_pre, alpha_post,
        alpha_res, B, n);
}

struct MHCLayer {
    MHCLayerConfig config;
    MHCLayerWeights weights;
    MHCLayerBuffers buffers;
    MHCLayerGradients grads;

    StreamMixTC stream_mix_tc;
    bool use_tc_mix;
    bool backward_enabled;
    bool use_pipelining;

    hipStream_t stream;
    hipStream_t sinkhorn_stream;
    hipEvent_t sinkhorn_done;
    bool owns_stream;
    bool initialized;

    MHCLayer()
        : stream(nullptr), sinkhorn_stream(nullptr), sinkhorn_done(nullptr), owns_stream(false),
          initialized(false), use_tc_mix(false), backward_enabled(false), use_pipelining(true) {}

    void init(const MHCLayerConfig& cfg, hipStream_t s = nullptr, bool enable_backward = false,
              bool enable_pipelining = true) {
        config = cfg;
        int B = cfg.batch_size;
        int C = cfg.hidden_dim;
        int n = cfg.expansion_rate;

        use_tc_mix = (n >= STREAM_MIX_TC_THRESHOLD);
        backward_enabled = enable_backward;
        // Only use pipelining for large expansion rate (n >= 16) where Sinkhorn-Knopp iteration
        // takes long enough to benefit from overlap
        use_pipelining = enable_pipelining && (n >= 16);

        weights.init(C, n, cfg.use_dynamic_h, cfg.alpha_init);
        buffers.init(B, C, n, use_tc_mix || backward_enabled, cfg.use_dynamic_h);

        if (use_tc_mix) {
            stream_mix_tc.init(B, n, C);
        }

        if (backward_enabled) {
            grads.init(B, C, n);
        }

        if (s == nullptr) {
            HIP_CALL(hipStreamCreate(&stream));
            owns_stream = true;
        } else {
            stream = s;
            owns_stream = false;
        }

        if (use_pipelining) {
            HIP_CALL(hipStreamCreate(&sinkhorn_stream));
            HIP_CALL(hipEventCreate(&sinkhorn_done));
        }

        initialized = true;
    }

    void destroy() {
        if (!initialized)
            return;

        weights.destroy();
        buffers.destroy();
        if (backward_enabled) {
            grads.destroy();
        }

        if (use_tc_mix) {
            stream_mix_tc.destroy();
        }

        if (use_pipelining) {
            if (sinkhorn_stream) {
                hipStreamDestroy(sinkhorn_stream);
                sinkhorn_stream = nullptr;
            }
            if (sinkhorn_done) {
                hipEventDestroy(sinkhorn_done);
                sinkhorn_done = nullptr;
            }
        }

        if (owns_stream && stream) {
            hipStreamDestroy(stream);
            stream = nullptr;
        }

        initialized = false;
    }

    void set_weights_dynamic(const __hip_bfloat16* h_rmsnorm_weight, const __hip_bfloat16* h_phi_pre,
                             const __hip_bfloat16* h_phi_post, const __hip_bfloat16* h_phi_res,
                             const float* h_b_pre, const float* h_b_post, const float* h_b_res,
                             float alpha_pre, float alpha_post, float alpha_res) {
        int C = config.hidden_dim;
        int n = config.expansion_rate;
        int nC = n * C;

        HIP_CALL(hipMemcpyAsync(weights.rmsnorm_weight, h_rmsnorm_weight, C * sizeof(__hip_bfloat16),
                                   hipMemcpyHostToDevice, stream));
        HIP_CALL(hipMemcpyAsync(weights.phi_pre, h_phi_pre, nC * n * sizeof(__hip_bfloat16),
                                   hipMemcpyHostToDevice, stream));
        HIP_CALL(hipMemcpyAsync(weights.phi_post, h_phi_post, nC * n * sizeof(__hip_bfloat16),
                                   hipMemcpyHostToDevice, stream));
        HIP_CALL(hipMemcpyAsync(weights.phi_res, h_phi_res, nC * n * n * sizeof(__hip_bfloat16),
                                   hipMemcpyHostToDevice, stream));
        HIP_CALL(hipMemcpyAsync(weights.b_pre, h_b_pre, n * sizeof(float),
                                   hipMemcpyHostToDevice, stream));
        HIP_CALL(hipMemcpyAsync(weights.b_post, h_b_post, n * sizeof(float),
                                   hipMemcpyHostToDevice, stream));
        HIP_CALL(hipMemcpyAsync(weights.b_res, h_b_res, n * n * sizeof(float),
                                   hipMemcpyHostToDevice, stream));

        weights.alpha_pre = alpha_pre;
        weights.alpha_post = alpha_post;
        weights.alpha_res = alpha_res;
    }

    void set_weights_static(const __hip_bfloat16* h_rmsnorm_weight, const float* h_b_pre,
                            const float* h_b_post, const float* h_b_res) {
        int C = config.hidden_dim;
        int n = config.expansion_rate;

        HIP_CALL(hipMemcpyAsync(weights.rmsnorm_weight, h_rmsnorm_weight, C * sizeof(__hip_bfloat16),
                                   hipMemcpyHostToDevice, stream));
        HIP_CALL(hipMemcpyAsync(weights.b_pre, h_b_pre, n * sizeof(float),
                                   hipMemcpyHostToDevice, stream));
        HIP_CALL(hipMemcpyAsync(weights.b_post, h_b_post, n * sizeof(float),
                                   hipMemcpyHostToDevice, stream));
        HIP_CALL(hipMemcpyAsync(weights.b_res, h_b_res, n * n * sizeof(float),
                                   hipMemcpyHostToDevice, stream));
    }

    void set_weights(const __hip_bfloat16* h_rmsnorm_weight, const float* h_H_pre, const float* h_H_post,
                     const float* h_H_res) {
        set_weights_static(h_rmsnorm_weight, h_H_pre, h_H_post, h_H_res);
    }

    void compute_dynamic_h_internal(int B, int n, int C) {
        int nC = n * C;

        float_to_bf16(buffers.x_flat_bf16, buffers.x_expanded, B * nC, stream);

        buffers.fused_rms_matmul.forward(buffers.H_proj_raw, buffers.x_flat_bf16,
                                         weights.phi_combined, stream);

        apply_dynamic_h_activations(buffers.H_pre_activated, buffers.H_post_activated,
                                    buffers.H_res_tilde, buffers.H_proj_raw, weights.b_pre,
                                    weights.b_post, weights.b_res, weights.alpha_pre,
                                    weights.alpha_post, weights.alpha_res, B, n, stream);

        apply_exp(buffers.H_res_tilde, buffers.H_res_tilde, B * n * n, stream);
        sinkhorn_knopp_forward_batched(buffers.sinkhorn_M, buffers.H_res_tilde, B, n,
                                       config.sinkhorn_iters, config.eps, stream);
    }

    void forward(const float* x_expanded) {
        int B = config.batch_size;
        int C = config.hidden_dim;
        int n = config.expansion_rate;

        HIP_CALL(hipMemcpyAsync(buffers.x_expanded, x_expanded, B * n * C * sizeof(float),
                                   hipMemcpyHostToDevice, stream));

        if (config.use_dynamic_h) {
            compute_dynamic_h_internal(B, n, C);

            stream_aggregate_bf16_dynamic(buffers.x_aggregated_bf16, buffers.x_expanded,
                                          buffers.H_pre_activated, B, n, C, stream);
        } else {
            if (use_pipelining) {
                sinkhorn_knopp_forward_fused_exp(buffers.sinkhorn_M, buffers.H_res_tilde,
                                                 weights.b_res, n, n, config.sinkhorn_iters,
                                                 config.eps, sinkhorn_stream);
                HIP_CALL(hipEventRecord(sinkhorn_done, sinkhorn_stream));
            }

            stream_aggregate_bf16_fused_sigmoid(buffers.x_aggregated_bf16, buffers.H_pre_activated,
                                                buffers.x_expanded, weights.b_pre, B, n, C, stream);

            if (!use_pipelining) {
                sinkhorn_knopp_forward_fused_exp(buffers.sinkhorn_M, buffers.H_res_tilde,
                                                 weights.b_res, n, n, config.sinkhorn_iters,
                                                 config.eps, stream);
            } else {
                HIP_CALL(hipStreamWaitEvent(stream, sinkhorn_done, 0));
            }
        }

        rmsnorm_forward_with_rms(buffers.layer_out_bf16, buffers.rms_values,
                                 buffers.x_aggregated_bf16, weights.rmsnorm_weight, B, C,
                                 config.eps, stream);

        if (config.use_dynamic_h) {
            stream_distribute_mix_add_fused_dynamic(
                buffers.output, buffers.x_expanded, buffers.layer_out_bf16,
                buffers.H_post_activated, buffers.sinkhorn_M, B, n, C, stream);
        } else {
            if (use_tc_mix) {
                stream_mix_tc.forward_fused_distribute_add(
                    buffers.output, buffers.H_post_activated, buffers.x_expanded,
                    buffers.layer_out_bf16, buffers.sinkhorn_M, weights.b_post, buffers.x_mixed,
                    stream);
            } else {
                stream_distribute_mix_add_fused(
                    buffers.output, buffers.H_post_activated, buffers.x_expanded,
                    buffers.layer_out_bf16, weights.b_post, buffers.sinkhorn_M, B, n, C, stream);
            }
        }
    }

    void forward_device(const float* d_x_expanded) {
        int B = config.batch_size;
        int C = config.hidden_dim;
        int n = config.expansion_rate;

        HIP_CALL(hipMemcpyAsync(buffers.x_expanded, d_x_expanded, B * n * C * sizeof(float),
                                   hipMemcpyDeviceToDevice, stream));

        if (config.use_dynamic_h) {
            compute_dynamic_h_internal(B, n, C);

            stream_aggregate_bf16_dynamic(buffers.x_aggregated_bf16, buffers.x_expanded,
                                          buffers.H_pre_activated, B, n, C, stream);
        } else {
            if (use_pipelining) {
                sinkhorn_knopp_forward_fused_exp(buffers.sinkhorn_M, buffers.H_res_tilde,
                                                 weights.b_res, n, n, config.sinkhorn_iters,
                                                 config.eps, sinkhorn_stream);
                HIP_CALL(hipEventRecord(sinkhorn_done, sinkhorn_stream));
            }

            stream_aggregate_bf16_fused_sigmoid(buffers.x_aggregated_bf16, buffers.H_pre_activated,
                                                buffers.x_expanded, weights.b_pre, B, n, C, stream);

            if (!use_pipelining) {
                sinkhorn_knopp_forward_fused_exp(buffers.sinkhorn_M, buffers.H_res_tilde,
                                                 weights.b_res, n, n, config.sinkhorn_iters,
                                                 config.eps, stream);
            } else {
                HIP_CALL(hipStreamWaitEvent(stream, sinkhorn_done, 0));
            }
        }

        rmsnorm_forward_with_rms(buffers.layer_out_bf16, buffers.rms_values,
                                 buffers.x_aggregated_bf16, weights.rmsnorm_weight, B, C,
                                 config.eps, stream);

        if (config.use_dynamic_h) {
            stream_distribute_mix_add_fused_dynamic(
                buffers.output, buffers.x_expanded, buffers.layer_out_bf16,
                buffers.H_post_activated, buffers.sinkhorn_M, B, n, C, stream);
        } else {
            if (use_tc_mix) {
                stream_mix_tc.forward_fused_distribute_add(
                    buffers.output, buffers.H_post_activated, buffers.x_expanded,
                    buffers.layer_out_bf16, buffers.sinkhorn_M, weights.b_post, buffers.x_mixed,
                    stream);
            } else {
                stream_distribute_mix_add_fused(
                    buffers.output, buffers.H_post_activated, buffers.x_expanded,
                    buffers.layer_out_bf16, weights.b_post, buffers.sinkhorn_M, B, n, C, stream);
            }
        }
    }

    float* get_output() { return buffers.output; }

    float* get_rms_values() { return buffers.rms_values; }

    void backward(const float* d_output) {
        if (!backward_enabled) {
            fprintf(stderr, "MHCLayer::backward called but backward not enabled\n");
            return;
        }

        int B = config.batch_size;
        int C = config.hidden_dim;
        int n = config.expansion_rate;

        grads.zero_weight_grads(C, n, stream);

        bf16_to_float(buffers.layer_out_f32, buffers.layer_out_bf16, B * C, stream);

        stream_distribute_mix_backward_fused(
            grads.d_x_mixed, grads.d_layer_out, grads.d_M, grads.d_H_post_activated, d_output,
            buffers.x_expanded, buffers.layer_out_f32, buffers.sinkhorn_M, buffers.H_post_activated,
            B, n, C, grads.workspace_dM, grads.workspace_dH, grads.workspace_num_blocks, stream);

        sinkhorn_knopp_backward(grads.d_H_res_exp, grads.d_M, buffers.sinkhorn_M,
                                buffers.H_res_tilde, n, config.sinkhorn_iters, config.eps, stream);

        exp_backward(grads.d_H_res, grads.d_H_res_exp, buffers.H_res_tilde, n * n, stream);

        sigmoid_scale_backward(grads.d_H_post, grads.d_H_post_activated, buffers.H_post_activated,
                               2.0f, n, stream);

        bf16_to_float(buffers.x_aggregated_f32, buffers.x_aggregated_bf16, B * C, stream);

        rmsnorm_backward(grads.d_x_aggregated, grads.d_rmsnorm_weight, grads.d_layer_out,
                         buffers.x_aggregated_bf16, weights.rmsnorm_weight, buffers.rms_values, B,
                         C, stream);

        stream_aggregate_backward(grads.d_x_expanded, grads.d_H_pre_activated, grads.d_x_aggregated,
                                  buffers.x_expanded, buffers.H_pre_activated, B, n, C,
                                  grads.workspace_dH, grads.workspace_num_blocks, stream);

        sigmoid_backward(grads.d_H_pre, grads.d_H_pre_activated, buffers.H_pre_activated, n,
                         stream);
    }

    void sync() { HIP_CALL(hipStreamSynchronize(stream)); }
};
} // namespace mhc
