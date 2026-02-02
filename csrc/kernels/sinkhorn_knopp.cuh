#pragma once

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <hip/hip_cooperative_groups.h>
#include <hip/hip_cooperative_groups.h>
#include "../include/mhc_layer.h"
#include "../include/utils.cuh"

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

template<int TILE_M, int TILE_N, int BLOCK_SIZE>
__global__ void sinkhorn_knopp_pdl_kernel(float* __restrict__ out, const float* __restrict__ inp,
                                          int M, int N, int num_iters, float eps) {
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

#if __CUDA_ARCH__ >= 900
    if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
        cudaTriggerProgrammaticLaunchCompletion();
    }
#endif

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

#ifdef MHC_ENABLE_PDL
        cudaLaunchAttribute attrs[1];
        attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attrs[0].val.programmaticStreamSerializationAllowed = 1;

        cudaLaunchConfig_t config = {};
        config.numAttrs = 1;
        config.attrs = attrs;
        config.blockDim = {BLOCK_SIZE, 1, 1};
        config.gridDim = {1, 1, 1};
        config.dynamicSmemBytes = smem_size;
        config.stream = stream;

        cudaLaunchKernelEx(&config,
                           sinkhorn_knopp_single_block_fused_exp_kernel<MAX_DIM, BLOCK_SIZE>, out,
                           H_res_exp, inp, M, N, num_iters, eps);
#else
        sinkhorn_knopp_single_block_fused_exp_kernel<MAX_DIM, BLOCK_SIZE>
            <<<1, BLOCK_SIZE, smem_size, stream>>>(out, H_res_exp, inp, M, N, num_iters, eps);
#endif
    } else if (M <= 128 && N <= 128) {
        constexpr int MAX_DIM = 128;
        size_t smem_size =
            MAX_DIM * MAX_DIM * sizeof(float) + MAX_DIM * sizeof(float) + MAX_DIM * sizeof(float);

        auto kernel = sinkhorn_knopp_single_block_fused_exp_kernel<MAX_DIM, BLOCK_SIZE>;
        hipFuncSetAttribute(reinterpret_cast<const void*>(kernel),
                            hipFuncAttributeMaxDynamicSharedMemorySize,
                            smem_size);

#ifdef MHC_ENABLE_PDL
        cudaLaunchAttribute attrs[1];
        attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attrs[0].val.programmaticStreamSerializationAllowed = 1;

        cudaLaunchConfig_t config = {};
        config.numAttrs = 1;
        config.attrs = attrs;
        config.blockDim = {BLOCK_SIZE, 1, 1};
        config.gridDim = {1, 1, 1};
        config.dynamicSmemBytes = smem_size;
        config.stream = stream;

        cudaLaunchKernelEx(&config, kernel, out, H_res_exp, inp, M, N, num_iters, eps);
#else
        kernel<<<1, BLOCK_SIZE, smem_size, stream>>>(out, H_res_exp, inp, M, N, num_iters, eps);
#endif
    } else {
        fprintf(stderr, "sinkhorn_knopp_forward_fused_exp: M > 128 or N > 128 not supported\n");
    }
}

template<int MAX_DIM, int BLOCK_SIZE>
__global__ void sinkhorn_knopp_single_block_pdl_kernel(float* __restrict__ out,
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

#if __CUDA_ARCH__ >= 900
    if (threadIdx.x == 0) {
        cudaTriggerProgrammaticLaunchCompletion();
    }
#endif

    for (int i = threadIdx.x; i < total_elems; i += BLOCK_SIZE) {
        out[i] = tile[i];
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

#ifdef MHC_ENABLE_PDL
        cudaLaunchAttribute attrs[1];
        attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attrs[0].val.programmaticStreamSerializationAllowed = 1;

        cudaLaunchConfig_t config = {};
        config.numAttrs = 1;
        config.attrs = attrs;
        config.blockDim = {THREADS_PER_BLOCK, 1, 1};
        config.gridDim = {(unsigned int)num_blocks, 1, 1};
        config.dynamicSmemBytes = 0;
        config.stream = stream;

        cudaLaunchKernelEx(&config, sinkhorn_knopp_batched_n4_kernel<4>, out, inp, B, num_iters,
                           eps);
#else
        sinkhorn_knopp_batched_n4_kernel<4>
            <<<num_blocks, THREADS_PER_BLOCK, 0, stream>>>(out, inp, B, num_iters, eps);
#endif
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

#ifdef MHC_ENABLE_PDL
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = 1;

    cudaLaunchConfig_t config = {};
    config.numAttrs = 1;
    config.attrs = attrs;
    config.blockDim = {BLOCK_SIZE, 1, 1};
    config.gridDim = {(unsigned int)B, 1, 1};
    config.dynamicSmemBytes = smem_size;
    config.stream = stream;

    cudaLaunchKernelEx(&config, sinkhorn_knopp_batched_kernel<N_MAX, BLOCK_SIZE>, out, inp, B, n,
                       num_iters, eps);
#else
    sinkhorn_knopp_batched_kernel<N_MAX, BLOCK_SIZE>
        <<<B, BLOCK_SIZE, smem_size, stream>>>(out, inp, B, n, num_iters, eps);
#endif
}

} // namespace mhc
