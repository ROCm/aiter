/*
 * Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * MHC (Multi-Head Channel) Layer CUDA/HIP Kernels for aiter
 * Based on mHC implementation
 */

#include <torch/all.h>
#include <ATen/hip/HIPContext.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include "../include/hip_compat.h"
#include "../include/dispatch_utils.h"

typedef __hip_bfloat16 floatX;
typedef __hip_bfloat162 floatX2;

namespace aiter {
namespace mhc {

// ============================================================================
// Utility functions
// ============================================================================

__device__ __forceinline__ float fast_exp(float x) {
    return __expf(x);
}

__device__ __forceinline__ float fast_sigmoid(float x) {
    return 1.0f / (1.0f + fast_exp(-x));
}

__device__ __forceinline__ floatX2 floats2bfloat162(float x, float y) {
    floatX2 out;
    out.x = __float2bfloat16(x);
    out.y = __float2bfloat16(y);
    return out;
}

// Warp reduction using shuffle instructions (HIP compatible)
__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down(val, offset);
    }
    return val;
}

// ============================================================================
// Sinkhorn-Knopp Kernels
// ============================================================================

template<int MAX_DIM, int BLOCK_SIZE>
__global__ void sinkhorn_knopp_kernel(
    float* __restrict__ out,
    const float* __restrict__ inp,
    int M, int N, int num_iters, float eps)
{
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
        // Row normalization
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

        // Column normalization
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
__global__ void sinkhorn_knopp_fused_exp_kernel(
    float* __restrict__ out,
    const float* __restrict__ inp,
    int M, int N, int num_iters, float eps)
{
    extern __shared__ float smem[];
    float* tile = smem;
    float* row_sums = smem + MAX_DIM * MAX_DIM;
    float* col_sums = row_sums + MAX_DIM;

    int total_elems = M * N;

    // Load with exp
    for (int i = threadIdx.x; i < total_elems; i += BLOCK_SIZE) {
        tile[i] = fast_exp(inp[i]);
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

template<int N_MAX, int BLOCK_SIZE>
__global__ void sinkhorn_knopp_batched_kernel(
    float* __restrict__ out,
    const float* __restrict__ inp,
    int B, int n, int num_iters, float eps)
{
    int batch_idx = blockIdx.x;
    if (batch_idx >= B) return;

    extern __shared__ float smem[];
    float* tile = smem;
    float* row_sums = tile + N_MAX * N_MAX;
    float* col_sums = row_sums + N_MAX;

    const float* inp_batch = inp + batch_idx * n * n;
    float* out_batch = out + batch_idx * n * n;

    int total = n * n;

    for (int i = threadIdx.x; i < total; i += BLOCK_SIZE) {
        tile[i] = inp_batch[i];
    }
    __syncthreads();

    for (int iter = 0; iter < num_iters; iter++) {
        for (int r = threadIdx.x; r < n; r += BLOCK_SIZE) {
            float sum = 0.0f;
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

    for (int i = threadIdx.x; i < total; i += BLOCK_SIZE) {
        out_batch[i] = tile[i];
    }
}

// ============================================================================
// Stream Aggregate Kernels
// ============================================================================

template<int BLOCK_SIZE, int MAX_N>
__global__ void stream_aggregate_kernel(
    float* __restrict__ out,
    const float* __restrict__ inp,
    const float* __restrict__ H_pre,
    int B, int n, int C)
{
    __shared__ float s_H_pre[MAX_N];
    if (threadIdx.x < n) {
        s_H_pre[threadIdx.x] = H_pre[threadIdx.x];
    }
    __syncthreads();

    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (idx >= B * C) return;

    int b = idx / C, c = idx % C;
    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < MAX_N; i++) {
        if (i < n)
            sum += s_H_pre[i] * inp[b * n * C + i * C + c];
    }
    out[idx] = sum;
}

template<int BLOCK_SIZE, int MAX_N>
__global__ void stream_aggregate_sigmoid_kernel(
    float* __restrict__ out,
    float* __restrict__ H_pre_activated,
    const float* __restrict__ inp,
    const float* __restrict__ H_pre_raw,
    int B, int n, int C)
{
    __shared__ float s_H_pre[MAX_N];
    if (threadIdx.x < n) {
        float activated = fast_sigmoid(H_pre_raw[threadIdx.x]);
        s_H_pre[threadIdx.x] = activated;
        H_pre_activated[threadIdx.x] = activated;
    }
    __syncthreads();

    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (idx >= B * C) return;

    int b = idx / C, c = idx % C;
    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < MAX_N; i++) {
        if (i < n)
            sum += s_H_pre[i] * inp[b * n * C + i * C + c];
    }
    out[idx] = sum;
}

template<int BLOCK_SIZE, int MAX_N>
__global__ void stream_aggregate_batched_kernel(
    float* __restrict__ out,
    const float* __restrict__ inp,
    const float* __restrict__ H_pre,  // [B, n]
    int B, int n, int C)
{
    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (idx >= B * C) return;

    int b = idx / C, c = idx % C;
    const float* h = H_pre + b * n;

    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < MAX_N; i++) {
        if (i < n)
            sum += h[i] * inp[b * n * C + i * C + c];
    }
    out[idx] = sum;
}

// ============================================================================
// Stream Distribute Mix Add Kernels
// ============================================================================

template<int BLOCK_SIZE, int MAX_N>
__global__ void stream_distribute_mix_add_kernel(
    float* __restrict__ out,
    const float* __restrict__ x_inp,
    const float* __restrict__ y,
    const float* __restrict__ H_post,
    const float* __restrict__ M,
    int B, int n, int C)
{
    __shared__ float s_M[MAX_N * MAX_N];
    __shared__ float s_H_post[MAX_N];

    if (threadIdx.x < n * n)
        s_M[threadIdx.x] = M[threadIdx.x];
    if (threadIdx.x < n)
        s_H_post[threadIdx.x] = H_post[threadIdx.x];
    __syncthreads();

    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (idx >= B * n * C) return;

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
    out[idx] = mix_sum + s_H_post[i] * y[b * C + c];
}

template<int BLOCK_SIZE, int MAX_N>
__global__ void stream_distribute_mix_add_sigmoid_kernel(
    float* __restrict__ out,
    float* __restrict__ H_post_activated,
    const float* __restrict__ x_inp,
    const float* __restrict__ y,
    const float* __restrict__ H_post_raw,
    const float* __restrict__ M,
    int B, int n, int C)
{
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
    if (idx >= B * n * C) return;

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
    out[idx] = mix_sum + s_H_post[i] * y[b * C + c];
}

template<int BLOCK_SIZE, int MAX_N>
__global__ void stream_distribute_mix_add_batched_kernel(
    float* __restrict__ out,
    const float* __restrict__ x_inp,
    const float* __restrict__ y,
    const float* __restrict__ H_post,  // [B, n]
    const float* __restrict__ M,       // [B, n, n]
    int B, int n, int C)
{
    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (idx >= B * n * C) return;

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
    out[idx] = mix_sum + h[i] * y[b * C + c];
}

// ============================================================================
// RMSNorm Kernel
// ============================================================================

template<int BLOCK_SIZE>
__global__ void rmsnorm_kernel(
    float* __restrict__ out,
    float* __restrict__ rms_out,
    const float* __restrict__ inp,
    const float* __restrict__ weight,
    int B, int C, float eps)
{
    constexpr int NUM_WARPS = BLOCK_SIZE / 32;
    
    int b = blockIdx.x;
    if (b >= B) return;
    
    const float* x = inp + b * C;
    float* y = out + b * C;
    
    // Compute sum of squares
    float sum_sq = 0.0f;
    for (int c = threadIdx.x; c < C; c += BLOCK_SIZE) {
        float val = x[c];
        sum_sq += val * val;
    }
    
    // Warp reduction using shuffle
    sum_sq = warp_reduce_sum(sum_sq);
    
    __shared__ float s_warp_sums[NUM_WARPS];
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    
    if (lane_id == 0) {
        s_warp_sums[warp_id] = sum_sq;
    }
    __syncthreads();
    
    // Final reduction in first warp
    if (warp_id == 0) {
        sum_sq = (lane_id < NUM_WARPS) ? s_warp_sums[lane_id] : 0.0f;
        sum_sq = warp_reduce_sum(sum_sq);
    }
    
    __shared__ float s_rms;
    if (threadIdx.x == 0) {
        float rms = rsqrtf(sum_sq / C + eps);
        s_rms = rms;
        if (rms_out) rms_out[b] = rms;
    }
    __syncthreads();
    
    float rms = s_rms;
    for (int c = threadIdx.x; c < C; c += BLOCK_SIZE) {
        y[c] = x[c] * rms * weight[c];
    }
}

// ============================================================================
// Launcher Functions
// ============================================================================

void launch_sinkhorn_knopp(
    float* out, const float* inp,
    int M, int N, int num_iters, float eps,
    hipStream_t stream)
{
    constexpr int BLOCK_SIZE = 256;
    
    if (M <= 64 && N <= 64) {
        constexpr int MAX_DIM = 64;
        size_t smem_size = MAX_DIM * MAX_DIM * sizeof(float) + 2 * MAX_DIM * sizeof(float);
        sinkhorn_knopp_kernel<MAX_DIM, BLOCK_SIZE><<<1, BLOCK_SIZE, smem_size, stream>>>(
            out, inp, M, N, num_iters, eps);
    } else if (M <= 128 && N <= 128) {
        constexpr int MAX_DIM = 128;
        size_t smem_size = MAX_DIM * MAX_DIM * sizeof(float) + 2 * MAX_DIM * sizeof(float);
        (void)hipFuncSetAttribute(
            reinterpret_cast<const void*>(sinkhorn_knopp_kernel<MAX_DIM, BLOCK_SIZE>),
            hipFuncAttributeMaxDynamicSharedMemorySize, smem_size);
        sinkhorn_knopp_kernel<MAX_DIM, BLOCK_SIZE><<<1, BLOCK_SIZE, smem_size, stream>>>(
            out, inp, M, N, num_iters, eps);
    }
}

void launch_sinkhorn_knopp_fused_exp(
    float* out, const float* inp,
    int M, int N, int num_iters, float eps,
    hipStream_t stream)
{
    constexpr int BLOCK_SIZE = 256;
    
    if (M <= 64 && N <= 64) {
        constexpr int MAX_DIM = 64;
        size_t smem_size = MAX_DIM * MAX_DIM * sizeof(float) + 2 * MAX_DIM * sizeof(float);
        sinkhorn_knopp_fused_exp_kernel<MAX_DIM, BLOCK_SIZE><<<1, BLOCK_SIZE, smem_size, stream>>>(
            out, inp, M, N, num_iters, eps);
    } else if (M <= 128 && N <= 128) {
        constexpr int MAX_DIM = 128;
        size_t smem_size = MAX_DIM * MAX_DIM * sizeof(float) + 2 * MAX_DIM * sizeof(float);
        (void)hipFuncSetAttribute(
            reinterpret_cast<const void*>(sinkhorn_knopp_fused_exp_kernel<MAX_DIM, BLOCK_SIZE>),
            hipFuncAttributeMaxDynamicSharedMemorySize, smem_size);
        sinkhorn_knopp_fused_exp_kernel<MAX_DIM, BLOCK_SIZE><<<1, BLOCK_SIZE, smem_size, stream>>>(
            out, inp, M, N, num_iters, eps);
    }
}

void launch_sinkhorn_knopp_batched(
    float* out, const float* inp,
    int B, int n, int num_iters, float eps,
    hipStream_t stream)
{
    constexpr int BLOCK_SIZE = 128;
    constexpr int N_MAX = 32;
    
    if (n > N_MAX) {
        // Fall back to per-batch processing for large n
        for (int b = 0; b < B; b++) {
            launch_sinkhorn_knopp(out + b * n * n, inp + b * n * n, n, n, num_iters, eps, stream);
        }
        return;
    }
    
    size_t smem_size = N_MAX * N_MAX * sizeof(float) + 2 * N_MAX * sizeof(float);
    sinkhorn_knopp_batched_kernel<N_MAX, BLOCK_SIZE><<<B, BLOCK_SIZE, smem_size, stream>>>(
        out, inp, B, n, num_iters, eps);
}

void launch_stream_aggregate(
    float* out, const float* inp, const float* H_pre,
    int B, int n, int C, bool is_batched,
    hipStream_t stream)
{
    constexpr int BLOCK = 256;
    int blocks = (B * C + BLOCK - 1) / BLOCK;
    
#define DISPATCH_AGG(MAX_N_VAL) \
    if (is_batched) { \
        stream_aggregate_batched_kernel<BLOCK, MAX_N_VAL><<<blocks, BLOCK, 0, stream>>>( \
            out, inp, H_pre, B, n, C); \
    } else { \
        stream_aggregate_kernel<BLOCK, MAX_N_VAL><<<blocks, BLOCK, 0, stream>>>( \
            out, inp, H_pre, B, n, C); \
    }
    
    if (n <= 4) { DISPATCH_AGG(4); }
    else if (n <= 8) { DISPATCH_AGG(8); }
    else if (n <= 16) { DISPATCH_AGG(16); }
    else if (n <= 32) { DISPATCH_AGG(32); }
    else if (n <= 64) { DISPATCH_AGG(64); }
    
#undef DISPATCH_AGG
}

void launch_stream_distribute_mix_add(
    float* out, const float* x_inp, const float* y,
    const float* H_post, const float* M,
    int B, int n, int C, bool is_batched,
    hipStream_t stream)
{
    constexpr int BLOCK = 256;
    int blocks = (B * n * C + BLOCK - 1) / BLOCK;
    
#define DISPATCH_DIST(MAX_N_VAL) \
    if (is_batched) { \
        stream_distribute_mix_add_batched_kernel<BLOCK, MAX_N_VAL><<<blocks, BLOCK, 0, stream>>>( \
            out, x_inp, y, H_post, M, B, n, C); \
    } else { \
        stream_distribute_mix_add_kernel<BLOCK, MAX_N_VAL><<<blocks, BLOCK, 0, stream>>>( \
            out, x_inp, y, H_post, M, B, n, C); \
    }
    
    if (n <= 4) { DISPATCH_DIST(4); }
    else if (n <= 8) { DISPATCH_DIST(8); }
    else if (n <= 16) { DISPATCH_DIST(16); }
    else if (n <= 32) { DISPATCH_DIST(32); }
    else if (n <= 64) { DISPATCH_DIST(64); }
    
#undef DISPATCH_DIST
}

void launch_rmsnorm(
    float* out, float* rms_out,
    const float* inp, const float* weight,
    int B, int C, float eps,
    hipStream_t stream)
{
    constexpr int BLOCK = 256;
    rmsnorm_kernel<BLOCK><<<B, BLOCK, 0, stream>>>(out, rms_out, inp, weight, B, C, eps);
}

} // namespace mhc
} // namespace aiter

// ============================================================================
// PyTorch Interface Functions
// ============================================================================

torch::Tensor mhc_layer_forward(
    torch::Tensor& x_expanded,
    torch::Tensor& rmsnorm_weight,
    torch::Tensor& H_pre,
    torch::Tensor& H_post,
    torch::Tensor& H_res,
    float eps,
    int sinkhorn_iters)
{
    TORCH_CHECK(x_expanded.is_cuda(), "x_expanded must be a CUDA tensor");
    TORCH_CHECK(x_expanded.dim() == 3, "x_expanded must be 3D [B, n, C]");
    
    int B = x_expanded.size(0);
    int n = x_expanded.size(1);
    int C = x_expanded.size(2);
    
    auto options = torch::TensorOptions()
        .dtype(torch::kFloat32)
        .device(x_expanded.device());
    
    // Ensure input is float32
    auto x_f32 = x_expanded.to(torch::kFloat32).contiguous();
    auto H_pre_f32 = H_pre.to(torch::kFloat32).contiguous();
    auto H_post_f32 = H_post.to(torch::kFloat32).contiguous();
    auto H_res_f32 = H_res.to(torch::kFloat32).contiguous();
    auto weight_f32 = rmsnorm_weight.to(torch::kFloat32).contiguous();
    
    const hipStream_t stream = at::hip::getCurrentHIPStream();
    
    // Allocate intermediate buffers
    auto x_aggregated = torch::empty({B, C}, options);
    auto M = torch::empty({n, n}, options);
    auto y_norm = torch::empty({B, C}, options);
    auto output = torch::empty({B, n, C}, options);
    
    // Compute H_pre activations (sigmoid) and aggregate
    auto H_pre_act = torch::sigmoid(H_pre_f32);
    
    // Stream aggregate: x_aggregated[b,c] = sum_i(H_pre[i] * x[b,i,c])
    aiter::mhc::launch_stream_aggregate(
        x_aggregated.data_ptr<float>(),
        x_f32.data_ptr<float>(),
        H_pre_act.data_ptr<float>(),
        B, n, C, false,
        stream);
    
    // Sinkhorn-Knopp on H_res (with exp)
    aiter::mhc::launch_sinkhorn_knopp_fused_exp(
        M.data_ptr<float>(),
        H_res_f32.data_ptr<float>(),
        n, n, sinkhorn_iters, eps,
        stream);
    
    // RMSNorm
    aiter::mhc::launch_rmsnorm(
        y_norm.data_ptr<float>(),
        nullptr,
        x_aggregated.data_ptr<float>(),
        weight_f32.data_ptr<float>(),
        B, C, eps,
        stream);
    
    // Compute H_post activations (2 * sigmoid)
    auto H_post_act = 2.0f * torch::sigmoid(H_post_f32);
    
    // Stream distribute mix add: out[b,i,c] = H_post[i] * y[b,c] + sum_j(M[i,j] * x[b,j,c])
    aiter::mhc::launch_stream_distribute_mix_add(
        output.data_ptr<float>(),
        x_f32.data_ptr<float>(),
        y_norm.data_ptr<float>(),
        H_post_act.data_ptr<float>(),
        M.data_ptr<float>(),
        B, n, C, false,
        stream);
    
    return output;
}

torch::Tensor mhc_layer_forward_dynamic(
    torch::Tensor& x_expanded,
    torch::Tensor& rmsnorm_weight,
    torch::Tensor& phi_pre,
    torch::Tensor& phi_post,
    torch::Tensor& phi_res,
    torch::Tensor& b_pre,
    torch::Tensor& b_post,
    torch::Tensor& b_res,
    float alpha_pre,
    float alpha_post,
    float alpha_res,
    float eps,
    int sinkhorn_iters)
{
    TORCH_CHECK(x_expanded.is_cuda(), "x_expanded must be a CUDA tensor");
    TORCH_CHECK(x_expanded.dim() == 3, "x_expanded must be 3D [B, n, C]");
    
    int B = x_expanded.size(0);
    int n = x_expanded.size(1);
    int C = x_expanded.size(2);
    int nC = n * C;
    
    auto options = torch::TensorOptions()
        .dtype(torch::kFloat32)
        .device(x_expanded.device());
    
    auto x_f32 = x_expanded.to(torch::kFloat32).contiguous();
    auto weight_f32 = rmsnorm_weight.to(torch::kFloat32).contiguous();
    
    const hipStream_t stream = at::hip::getCurrentHIPStream();
    
    // Flatten x for projection: [B, n*C]
    auto x_flat = x_f32.view({B, nC});
    
    // Project to get H parameters
    // H_pre_proj: [B, n] = x_flat @ phi_pre.T
    // H_post_proj: [B, n] = x_flat @ phi_post.T
    // H_res_proj: [B, n*n] = x_flat @ phi_res.T
    auto H_pre_proj = torch::mm(x_flat, phi_pre.t());
    auto H_post_proj = torch::mm(x_flat, phi_post.t());
    auto H_res_proj = torch::mm(x_flat, phi_res.t());
    
    // Apply activations with biases and alpha scaling
    auto H_pre_act = torch::sigmoid(alpha_pre * H_pre_proj + b_pre);
    auto H_post_act = 2.0f * torch::sigmoid(alpha_post * H_post_proj + b_post);
    auto H_res_raw = alpha_res * H_res_proj + b_res.view({1, n * n});
    
    // Allocate outputs
    auto x_aggregated = torch::empty({B, C}, options);
    auto M = torch::empty({B, n, n}, options);
    auto y_norm = torch::empty({B, C}, options);
    auto output = torch::empty({B, n, C}, options);
    
    // Stream aggregate with batched H_pre
    aiter::mhc::launch_stream_aggregate(
        x_aggregated.data_ptr<float>(),
        x_f32.data_ptr<float>(),
        H_pre_act.data_ptr<float>(),
        B, n, C, true,
        stream);
    
    // Sinkhorn-Knopp batched
    aiter::mhc::launch_sinkhorn_knopp_batched(
        M.view({B * n * n}).data_ptr<float>(),
        torch::exp(H_res_raw).view({B * n * n}).data_ptr<float>(),
        B, n, sinkhorn_iters, eps,
        stream);
    
    // RMSNorm
    aiter::mhc::launch_rmsnorm(
        y_norm.data_ptr<float>(),
        nullptr,
        x_aggregated.data_ptr<float>(),
        weight_f32.data_ptr<float>(),
        B, C, eps,
        stream);
    
    // Stream distribute mix add batched
    aiter::mhc::launch_stream_distribute_mix_add(
        output.data_ptr<float>(),
        x_f32.data_ptr<float>(),
        y_norm.data_ptr<float>(),
        H_post_act.data_ptr<float>(),
        M.data_ptr<float>(),
        B, n, C, true,
        stream);
    
    return output;
}

torch::Tensor sinkhorn_knopp_forward(
    torch::Tensor& inp,
    int num_iters,
    float eps)
{
    TORCH_CHECK(inp.is_cuda(), "Input must be a CUDA tensor");
    
    auto inp_f32 = inp.to(torch::kFloat32).contiguous();
    auto out = torch::empty_like(inp_f32);
    
    const hipStream_t stream = at::hip::getCurrentHIPStream();
    
    if (inp.dim() == 2) {
        int M = inp.size(0);
        int N = inp.size(1);
        aiter::mhc::launch_sinkhorn_knopp(
            out.data_ptr<float>(),
            inp_f32.data_ptr<float>(),
            M, N, num_iters, eps,
            stream);
    } else if (inp.dim() == 3) {
        int B = inp.size(0);
        int n = inp.size(1);
        TORCH_CHECK(inp.size(2) == n, "Sinkhorn-Knopp expects square matrices");
        aiter::mhc::launch_sinkhorn_knopp_batched(
            out.data_ptr<float>(),
            inp_f32.data_ptr<float>(),
            B, n, num_iters, eps,
            stream);
    } else {
        TORCH_CHECK(false, "Input must be 2D [M, N] or 3D [B, n, n]");
    }
    
    return out;
}

torch::Tensor stream_aggregate_forward(
    torch::Tensor& x_expanded,
    torch::Tensor& H_pre)
{
    TORCH_CHECK(x_expanded.is_cuda(), "x_expanded must be a CUDA tensor");
    TORCH_CHECK(x_expanded.dim() == 3, "x_expanded must be 3D [B, n, C]");
    
    int B = x_expanded.size(0);
    int n = x_expanded.size(1);
    int C = x_expanded.size(2);
    
    auto x_f32 = x_expanded.to(torch::kFloat32).contiguous();
    auto H_pre_f32 = H_pre.to(torch::kFloat32).contiguous();
    
    auto options = torch::TensorOptions()
        .dtype(torch::kFloat32)
        .device(x_expanded.device());
    auto out = torch::empty({B, C}, options);
    
    const hipStream_t stream = at::hip::getCurrentHIPStream();
    
    bool is_batched = (H_pre.dim() == 2);
    
    aiter::mhc::launch_stream_aggregate(
        out.data_ptr<float>(),
        x_f32.data_ptr<float>(),
        H_pre_f32.data_ptr<float>(),
        B, n, C, is_batched,
        stream);
    
    return out;
}

torch::Tensor stream_distribute_mix_add_forward(
    torch::Tensor& x_expanded,
    torch::Tensor& y,
    torch::Tensor& H_post,
    torch::Tensor& M)
{
    TORCH_CHECK(x_expanded.is_cuda(), "x_expanded must be a CUDA tensor");
    TORCH_CHECK(x_expanded.dim() == 3, "x_expanded must be 3D [B, n, C]");
    
    int B = x_expanded.size(0);
    int n = x_expanded.size(1);
    int C = x_expanded.size(2);
    
    auto x_f32 = x_expanded.to(torch::kFloat32).contiguous();
    auto y_f32 = y.to(torch::kFloat32).contiguous();
    auto H_post_f32 = H_post.to(torch::kFloat32).contiguous();
    auto M_f32 = M.to(torch::kFloat32).contiguous();
    
    auto options = torch::TensorOptions()
        .dtype(torch::kFloat32)
        .device(x_expanded.device());
    auto out = torch::empty({B, n, C}, options);
    
    const hipStream_t stream = at::hip::getCurrentHIPStream();
    
    bool is_batched = (M.dim() == 3);
    
    aiter::mhc::launch_stream_distribute_mix_add(
        out.data_ptr<float>(),
        x_f32.data_ptr<float>(),
        y_f32.data_ptr<float>(),
        H_post_f32.data_ptr<float>(),
        M_f32.data_ptr<float>(),
        B, n, C, is_batched,
        stream);
    
    return out;
}
