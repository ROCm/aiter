// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

#include "aiter_hip_common.h"
#include "py_itfs_common.h"
#include "aiter_opus_plus.h"
#include "dispatch_utils.h"
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>

namespace aiter {

/**
 * Optimized Fused Gated RMSNorm + FP8 Group Quantization Kernel
 *
 * Operations:
 * 1. Per-head Gated RMSNorm: norm(x) * silu(z) where:
 *    - norm(x) = x * weight / sqrt(variance + eps) (standard RMSNorm)
 *    - silu(z) = z / (1 + exp(-z))
 * 2. Flatten: [num_tokens, num_heads, head_dim] → [num_tokens, num_heads*head_dim]
 * 3. FP8 group quantization with group_size=128
 *
 * Constraints:
 * - ONLY supports head_dim=128 and group_size=128
 * - Each head is exactly one quantization group
 * - AMD GPU: warp_size=64
 *
 * Template Parameters:
 * - GROUP_SIZE: Quantization group size (compile-time constant, default=128)
 * - BLOCK_SIZE: Number of threads per block (64, 128, or 256)
 *
 * Optimizations:
 * - Grid: (num_tokens, num_heads) - 2D grid
 * - Block: Configurable (64/128/256 threads)
 * - Each thread processes 2 elements using vectorized loads
 * - Warp reduction using __shfl_xor (NO shared memory)
 * - Loop unrolling with #pragma unroll
 * - Coalesced memory access
 */
template <typename DTYPE_I, typename DTYPE_O, int GROUP_SIZE = 128, int THREAD_DATA_SIZE = 16, int BLOCK_SIZE = 256, bool TRANSPOSE_SCALE = false>
__global__ void gated_rmsnorm_fp8_group_quant_kernel(
    DTYPE_O* __restrict__ out,           // [num_tokens, num_heads * head_dim]
    float* __restrict__ scale,           // [num_heads, num_tokens] (transposed) or [num_tokens, num_heads]
    DTYPE_I const* __restrict__ x,       // [num_tokens, num_heads, head_dim] - input to normalize
    DTYPE_I const* __restrict__ z,       // [num_tokens, num_heads, head_dim] - gating tensor
    DTYPE_I const* __restrict__ weight,  // [head_dim] - RMSNorm weight
    double epsilon,
    int num_tokens,
    int num_heads,
    int head_dim,
    int64_t x_token_stride,              // x stride along token dim (elements)
    int64_t x_head_stride,               // x stride along head dim (elements)
    int64_t z_token_stride,              // z stride along token dim (elements)
    int64_t z_head_stride)               // z stride along head dim (elements)
{
    // Compile-time validation
    static_assert(GROUP_SIZE == 128, "Only GROUP_SIZE=128 is supported");
    static_assert(THREAD_DATA_SIZE >= 2 && THREAD_DATA_SIZE <= 32, "THREAD_DATA_SIZE must be 2-32");

    // Calculate groups per warp
    constexpr int WARP_SIZE = 64;
    constexpr int threads_per_group = GROUP_SIZE / THREAD_DATA_SIZE;
    constexpr int groups_per_warp = WARP_SIZE / threads_per_group;
    constexpr int groups_per_block = (BLOCK_SIZE / WARP_SIZE) * groups_per_warp;

    // Grid: (num_tokens, ceil(num_heads / groups_per_block))
    // Each block processes multiple heads/groups
    const int token_id = blockIdx.x;
    const int group_block_id = blockIdx.y;
    const int tid = threadIdx.x;

    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;

    // Which group within this block
    const int thread_group_id = lane_id / threads_per_group;  // 0 to groups_per_warp-1
    const int thread_in_group = lane_id % threads_per_group;

    // Global group/head ID
    const int head_id = group_block_id * groups_per_block + warp_id * groups_per_warp + thread_group_id;

    if (token_id >= num_tokens || head_id >= num_heads) {
        return;
    }

    const int elem_id = thread_in_group * THREAD_DATA_SIZE;

    // Input pointers for this (token, head). x and z may be strided slices of larger
    // tensors, so use the per-tensor (token, head) strides. The inner head_dim must
    // be unit-strided (validated on the host) so vector loads stay coalesced.
    const int64_t x_offset = static_cast<int64_t>(token_id) * x_token_stride
                           + static_cast<int64_t>(head_id) * x_head_stride;
    const int64_t z_offset = static_cast<int64_t>(token_id) * z_token_stride
                           + static_cast<int64_t>(head_id) * z_head_stride;
    const DTYPE_I* x_ptr = x + x_offset;
    const DTYPE_I* z_ptr = z + z_offset;

    // Load THREAD_DATA_SIZE elements per thread
    float x_vals[THREAD_DATA_SIZE];
    float z_vals[THREAD_DATA_SIZE];
    float weight_vals[THREAD_DATA_SIZE];

    #pragma unroll
    for (int i = 0; i < THREAD_DATA_SIZE; i++) {
        x_vals[i] = opus::cast<float>(x_ptr[elem_id + i]);
        z_vals[i] = opus::cast<float>(z_ptr[elem_id + i]);
        weight_vals[i] = opus::cast<float>(weight[elem_id + i]);
    }

    // Step 1: Compute variance for standard RMSNorm (sum of squares of x)
    float sum_sq = 0.0f;
    #pragma unroll
    for (int i = 0; i < THREAD_DATA_SIZE; i++) {
        sum_sq += x_vals[i] * x_vals[i];
    }

    // Group-local reduce sum (only within threads_per_group, not full warp!)
    #pragma unroll
    for (int mask = threads_per_group / 2; mask > 0; mask >>= 1) {
        sum_sq += __shfl_xor(sum_sq, mask);
    }
    // All threads in this group now have the same sum_sq

    // Compute RMS normalization factor
    constexpr float inv_head_dim = 1.0f / static_cast<float>(GROUP_SIZE);
    float variance = sum_sq * inv_head_dim;
    float inv_std = rsqrtf(variance + static_cast<float>(epsilon));

    // Step 2: Apply standard RMSNorm: x * weight / sqrt(variance + eps)
    float normed_vals[THREAD_DATA_SIZE];
    #pragma unroll
    for (int i = 0; i < THREAD_DATA_SIZE; i++) {
        normed_vals[i] = x_vals[i] * weight_vals[i] * inv_std;
    }

    // Step 3: Apply SiLU gating and multiply
    float gated_vals[THREAD_DATA_SIZE];
    #pragma unroll
    for (int i = 0; i < THREAD_DATA_SIZE; i++) {
        float sigmoid_z = 1.0f / (1.0f + expf(-z_vals[i]));
        float silu_z = z_vals[i] * sigmoid_z;
        gated_vals[i] = normed_vals[i] * silu_z;
    }

    // Step 4: Find max absolute value for FP8 quantization
    float local_max = -INFINITY;  // FIX: Initialize to -infinity, not 0
    #pragma unroll
    for (int i = 0; i < THREAD_DATA_SIZE; i++) {
        local_max = fmaxf(local_max, fabsf(gated_vals[i]));
    }

    // Group-local reduce max (only within threads of this group)
    #pragma unroll
    for (int mask = threads_per_group / 2; mask > 0; mask >>= 1) {
        local_max = fmaxf(local_max, __shfl_xor(local_max, mask));
    }

    // Step 5: Compute scale for FP8 quantization
    constexpr float FP8_MAX = static_cast<float>(opus::finfo<DTYPE_O>::max());
    float quant_scale = (local_max > 1e-10f) ? (local_max / FP8_MAX) : 1e-10f;
    float quant_scale_inv = 1.0f / quant_scale;

    // Step 6: Quantize and store
    const int out_base = token_id * (num_heads * head_dim) + head_id * head_dim;
    using DTYPE_O_STORE = typename opus::vector_traits<DTYPE_O>::dtype;
    DTYPE_O_STORE* out_ptr = reinterpret_cast<DTYPE_O_STORE*>(out + out_base);

    #pragma unroll
    for (int i = 0; i < THREAD_DATA_SIZE; i++) {
        float clamped = fminf(fmaxf(gated_vals[i] * quant_scale_inv, -FP8_MAX), FP8_MAX);
        DTYPE_O quantized = opus::cast<DTYPE_O>(clamped);
        out_ptr[elem_id + i] = quantized;
    }

    // Step 7: Thread 0 of each group stores scale
    if (thread_in_group == 0) {
        int scale_idx;
        if constexpr (TRANSPOSE_SCALE) {
            scale_idx = head_id * num_tokens + token_id;
        } else {
            scale_idx = token_id * num_heads + head_id;
        }
        scale[scale_idx] = quant_scale;
    }
}

/**
 * Host function to launch the optimized fused Gated RMSNorm + FP8 group quant kernel
 * with configurable block size for performance tuning.
 *
 * Block size options:
 * - 64 threads (1 warp): Baseline, minimal resource usage
 * - 128 threads (2 warps): Better occupancy, recommended for most cases
 * - 256 threads (4 warps): Maximum occupancy, best for large workloads
 */
template <typename DTYPE_I, typename DTYPE_O, int THREAD_DATA_SIZE, int BLOCK_SIZE, bool TRANSPOSE_SCALE>
void gated_rmsnorm_fp8_group_quant_launcher_impl(
    torch::Tensor& out,
    torch::Tensor& scale,
    torch::Tensor const& x,
    torch::Tensor const& z,
    torch::Tensor const& weight,
    double epsilon,
    int num_tokens,
    int num_heads,
    int head_dim)
{
    constexpr int GROUP_SIZE = 128;
    constexpr int WARP_SIZE = 64;
    constexpr int threads_per_group = GROUP_SIZE / THREAD_DATA_SIZE;
    constexpr int groups_per_warp = WARP_SIZE / threads_per_group;
    constexpr int groups_per_block = (BLOCK_SIZE / WARP_SIZE) * groups_per_warp;

    // Grid: (num_tokens, ceil(num_heads / groups_per_block))
    dim3 grid(num_tokens, (num_heads + groups_per_block - 1) / groups_per_block);
    dim3 block(BLOCK_SIZE);

    hipStream_t stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA();

    // Strides for x and z (token, head). The inner head_dim is required to be
    // unit-stride (validated by the caller) so we don't need to thread it through.
    const int64_t x_token_stride = x.stride(0);
    const int64_t x_head_stride  = x.stride(1);
    const int64_t z_token_stride = z.stride(0);
    const int64_t z_head_stride  = z.stride(1);

    gated_rmsnorm_fp8_group_quant_kernel<DTYPE_I, DTYPE_O, GROUP_SIZE, THREAD_DATA_SIZE, BLOCK_SIZE, TRANSPOSE_SCALE>
        <<<grid, block, 0, stream>>>(
            reinterpret_cast<DTYPE_O*>(out.data_ptr()),
            reinterpret_cast<float*>(scale.data_ptr()),
            reinterpret_cast<DTYPE_I const*>(x.data_ptr()),
            reinterpret_cast<DTYPE_I const*>(z.data_ptr()),
            reinterpret_cast<DTYPE_I const*>(weight.data_ptr()),
            epsilon,
            num_tokens,
            num_heads,
            head_dim,
            x_token_stride,
            x_head_stride,
            z_token_stride,
            z_head_stride
        );
}

template <typename DTYPE_I, typename DTYPE_O>
void gated_rmsnorm_fp8_group_quant_launcher(
    torch::Tensor& out,           // [num_tokens, num_heads * head_dim]
    torch::Tensor& scale,          // [num_heads, num_tokens] (transposed)
    torch::Tensor const& x,        // [num_tokens, num_heads, head_dim] - input to normalize
    torch::Tensor const& z,        // [num_tokens, num_heads, head_dim] - gating tensor
    torch::Tensor const& weight,   // [head_dim] - RMSNorm weight
    double epsilon,
    int group_size,
    bool transpose_scale)
{
    // Validate constraints
    TORCH_CHECK(x.dim() == 3, "Input x must be 3D: [num_tokens, num_heads, head_dim]");
    TORCH_CHECK(z.dim() == 3, "Input z must be 3D: [num_tokens, num_heads, head_dim]");
    const int num_tokens = x.size(0);
    const int num_heads = x.size(1);
    const int head_dim = x.size(2);

    TORCH_CHECK(z.size(0) == num_tokens && z.size(1) == num_heads && z.size(2) == head_dim,
                "Gating tensor z must have same shape as x");
    TORCH_CHECK(head_dim == 128, "ONLY head_dim=128 is supported, got ", head_dim);
    TORCH_CHECK(group_size == 128, "ONLY group_size=128 is supported, got ", group_size);
    TORCH_CHECK(weight.size(0) == head_dim, "Weight size must match head_dim");

    // x and z may be strided slices on the token/head dims, but the inner head_dim
    // must be unit-stride for vectorized loads.
    TORCH_CHECK(x.stride(2) == 1, "x.stride(2) must be 1 (head_dim contiguous), got ", x.stride(2));
    TORCH_CHECK(z.stride(2) == 1, "z.stride(2) must be 1 (head_dim contiguous), got ", z.stride(2));


    // Use THREAD_DATA_SIZE=16 (8 groups/warp) for best bandwidth
    constexpr int thread_data_size = 16;
    if (transpose_scale) {
        gated_rmsnorm_fp8_group_quant_launcher_impl<DTYPE_I, DTYPE_O, thread_data_size, 256, true>(
            out, scale, x, z, weight, epsilon, num_tokens, num_heads, head_dim);
    } else {
        gated_rmsnorm_fp8_group_quant_launcher_impl<DTYPE_I, DTYPE_O, thread_data_size, 256, false>(
            out, scale, x, z, weight, epsilon, num_tokens, num_heads, head_dim);
    }
}

/**
 * Python interface
 */
void gated_rmsnorm_fp8_group_quant(
    torch::Tensor& out,           // [num_tokens, num_heads * head_dim]
    torch::Tensor& scale,          // [num_heads, num_tokens] (transposed)
    torch::Tensor const& x,        // [num_tokens, num_heads, head_dim] - input to normalize
    torch::Tensor const& z,        // [num_tokens, num_heads, head_dim] - gating tensor
    torch::Tensor const& weight,   // [head_dim] - RMSNorm weight
    double epsilon,
    int group_size,
    bool transpose_scale)
{
    // Validate input types
    TORCH_CHECK(x.is_cuda(), "Input x must be on CUDA device");
    TORCH_CHECK(z.is_cuda(), "Input z must be on CUDA device");
    TORCH_CHECK(weight.is_cuda(), "Weight must be on CUDA device");
    TORCH_CHECK(out.is_cuda(), "Output must be on CUDA device");
    TORCH_CHECK(scale.is_cuda(), "Scale must be on CUDA device");

    // Dispatch based on input/output types
    if (x.scalar_type() == at::ScalarType::BFloat16 &&
        (out.scalar_type() == at::ScalarType::Float8_e4m3fnuz || out.scalar_type() == at::ScalarType::Float8_e4m3fn)) {
        gated_rmsnorm_fp8_group_quant_launcher<opus::bf16_t, opus::fp8_t>(
            out, scale, x, z, weight, epsilon, group_size, transpose_scale);
    } else if (x.scalar_type() == at::ScalarType::Half &&
               (out.scalar_type() == at::ScalarType::Float8_e4m3fnuz || out.scalar_type() == at::ScalarType::Float8_e4m3fn)) {
        gated_rmsnorm_fp8_group_quant_launcher<opus::fp16_t, opus::fp8_t>(
            out, scale, x, z, weight, epsilon, group_size, transpose_scale);
    } else {
        TORCH_CHECK(false, "Unsupported dtype combination. Input: ", x.scalar_type(),
                    ", Output: ", out.scalar_type());
    }
}

} // namespace aiter
