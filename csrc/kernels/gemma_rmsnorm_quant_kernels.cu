// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

#include "aiter_hip_common.h"
#include "py_itfs_common.h"
#include "aiter_opus_plus.h"
#include "dispatch_utils.h"
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>

namespace aiter {

/**
 * Fused Gemma RMSNorm + FP8 Group Quantization Kernel
 *
 * Operations:
 * 1. Optional residual add: x = x + residual, residual = x (inplace write-back)
 * 2. Gemma RMSNorm: out = x * rsqrt(mean(x^2) + eps) * (1 + weight)
 *    - Variance computed over full hidden_size
 *    - Gemma-style: (1 + weight) instead of weight
 * 3. FP8 group quantization with group_size=128
 * 4. Optional: also write unquantized normed output (HAS_OUT_NORMED)
 *
 * Constraints:
 * - hidden_size must be a multiple of GROUP_SIZE (128)
 * - AMD GPU: warp_size=64
 *
 * Grid: (num_tokens,) - one block per token
 * Block: BLOCK_SIZE threads
 *
 * Design: Each thread handles THREAD_DATA_SIZE contiguous elements within a
 * group, and loops over multiple groups. Data is cached in registers across
 * the two phases to avoid re-reading from global memory.
 *
 * MAX_GROUPS_PER_THREAD limits the register file usage. For typical Gemma
 * models (hidden_size <= 8192), MAX_GROUPS_PER_THREAD=3 is sufficient
 * with BLOCK_SIZE=256.
 */
template <typename DTYPE_I, typename DTYPE_O, int GROUP_SIZE = 128,
          int THREAD_DATA_SIZE = 16, int BLOCK_SIZE = 256,
          int MAX_GROUPS_PER_THREAD = 3,
          bool TRANSPOSE_SCALE = false, bool HAS_RESIDUAL = false,
          bool HAS_OUT_NORMED = false>
__global__ void gemma_rmsnorm_fp8_group_quant_kernel(
    DTYPE_O* __restrict__ out,           // [num_tokens, hidden_size]
    float* __restrict__ scale,           // [num_tokens, num_groups] or transposed
    DTYPE_I const* __restrict__ x,       // [num_tokens, hidden_size]
    DTYPE_I* __restrict__ residual,      // [num_tokens, hidden_size] (inplace if HAS_RESIDUAL)
    DTYPE_I const* __restrict__ weight,  // [hidden_size]
    DTYPE_I* __restrict__ out_normed,    // [num_tokens, hidden_size] (if HAS_OUT_NORMED)
    double epsilon,
    int num_tokens,
    int hidden_size)
{
    static_assert(GROUP_SIZE == 128, "Only GROUP_SIZE=128 is supported");

    constexpr int WARP_SIZE = 64;
    constexpr int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;
    constexpr int threads_per_group = GROUP_SIZE / THREAD_DATA_SIZE;
    constexpr int groups_per_block = BLOCK_SIZE / threads_per_group;

    const int token_id = blockIdx.x;
    if (token_id >= num_tokens) return;

    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;

    const int slot_group = tid / threads_per_group;
    const int thread_in_group = tid % threads_per_group;
    const int num_groups = hidden_size / GROUP_SIZE;

    const int token_base = token_id * hidden_size;

    // ---- Phase 1: Load ALL data into registers, accumulate sum of squares ----
    // Register cache: x_vals[iter][THREAD_DATA_SIZE]
    float x_cache[MAX_GROUPS_PER_THREAD][THREAD_DATA_SIZE];
    float local_sum_sq = 0.0f;

    #pragma unroll
    for (int iter = 0; iter < MAX_GROUPS_PER_THREAD; iter++) {
        const int g = slot_group + iter * groups_per_block;
        if (g < num_groups) {
            const int elem_base = token_base + g * GROUP_SIZE
                                  + thread_in_group * THREAD_DATA_SIZE;
            #pragma unroll
            for (int i = 0; i < THREAD_DATA_SIZE; i++) {
                float xv = opus::cast<float>(x[elem_base + i]);
                if constexpr (HAS_RESIDUAL) {
                    float rv = opus::cast<float>(residual[elem_base + i]);
                    xv = xv + rv;
                    residual[elem_base + i] = opus::cast<DTYPE_I>(xv);
                }
                x_cache[iter][i] = xv;
                local_sum_sq += xv * xv;
            }
        }
    }

    // ---- Block-wide reduction for variance ----
    #pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        local_sum_sq += __shfl_xor(local_sum_sq, mask);
    }

    __shared__ float warp_sums[NUM_WARPS];
    __shared__ float s_inv_std;

    if (lane_id == 0) {
        warp_sums[warp_id] = local_sum_sq;
    }
    __syncthreads();

    if (warp_id == 0) {
        float sum = (lane_id < NUM_WARPS) ? warp_sums[lane_id] : 0.0f;
        #pragma unroll
        for (int mask = NUM_WARPS / 2; mask > 0; mask >>= 1) {
            sum += __shfl_xor(sum, mask);
        }
        if (lane_id == 0) {
            float variance = sum / static_cast<float>(hidden_size);
            s_inv_std = rsqrtf(variance + static_cast<float>(epsilon));
        }
    }
    __syncthreads();

    const float inv_std = s_inv_std;

    // ---- Phase 2: Apply Gemma RMSNorm + quantize, optionally write normed ----
    constexpr float FP8_MAX = static_cast<float>(opus::finfo<DTYPE_O>::max());

    #pragma unroll
    for (int iter = 0; iter < MAX_GROUPS_PER_THREAD; iter++) {
        const int g = slot_group + iter * groups_per_block;
        if (g < num_groups) {
            const int weight_offset = g * GROUP_SIZE
                                      + thread_in_group * THREAD_DATA_SIZE;

            // Apply Gemma norm from cached x values
            float normed_vals[THREAD_DATA_SIZE];
            #pragma unroll
            for (int i = 0; i < THREAD_DATA_SIZE; i++) {
                float w = opus::cast<float>(weight[weight_offset + i]);
                normed_vals[i] = x_cache[iter][i] * inv_std * (1.0f + w);
            }

            const int elem_base = token_base + g * GROUP_SIZE
                                  + thread_in_group * THREAD_DATA_SIZE;

            // Optionally write unquantized normed output
            if constexpr (HAS_OUT_NORMED) {
                #pragma unroll
                for (int i = 0; i < THREAD_DATA_SIZE; i++) {
                    out_normed[elem_base + i] = opus::cast<DTYPE_I>(normed_vals[i]);
                }
            }

            // FP8 group quantization
            float local_max = -INFINITY;
            #pragma unroll
            for (int i = 0; i < THREAD_DATA_SIZE; i++) {
                local_max = fmaxf(local_max, fabsf(normed_vals[i]));
            }

            #pragma unroll
            for (int mask = threads_per_group / 2; mask > 0; mask >>= 1) {
                local_max = fmaxf(local_max, __shfl_xor(local_max, mask));
            }

            float quant_scale = (local_max > 1e-10f) ? (local_max / FP8_MAX) : 1e-10f;
            float quant_scale_inv = 1.0f / quant_scale;

            using DTYPE_O_STORE = typename opus::vector_traits<DTYPE_O>::dtype;
            DTYPE_O_STORE* out_ptr = reinterpret_cast<DTYPE_O_STORE*>(out + elem_base);

            #pragma unroll
            for (int i = 0; i < THREAD_DATA_SIZE; i++) {
                float clamped = fminf(fmaxf(normed_vals[i] * quant_scale_inv,
                                            -FP8_MAX), FP8_MAX);
                DTYPE_O quantized = opus::cast<DTYPE_O>(clamped);
                out_ptr[i] = quantized;
            }

            if (thread_in_group == 0) {
                int scale_idx;
                if constexpr (TRANSPOSE_SCALE) {
                    scale_idx = g * num_tokens + token_id;
                } else {
                    scale_idx = token_id * num_groups + g;
                }
                scale[scale_idx] = quant_scale;
            }
        }
    }
}

/**
 * Launcher with compile-time parameters
 */
template <typename DTYPE_I, typename DTYPE_O,
          int THREAD_DATA_SIZE, int BLOCK_SIZE, int MAX_GROUPS_PER_THREAD,
          bool TRANSPOSE_SCALE, bool HAS_RESIDUAL, bool HAS_OUT_NORMED>
void gemma_rmsnorm_fp8_group_quant_launcher_impl(
    torch::Tensor& out,
    torch::Tensor& scale,
    torch::Tensor const& x,
    torch::Tensor& residual,
    torch::Tensor const& weight,
    torch::Tensor& out_normed,
    double epsilon,
    int num_tokens,
    int hidden_size)
{
    dim3 grid(num_tokens);
    dim3 block(BLOCK_SIZE);

    hipStream_t stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA();

    DTYPE_I* residual_ptr = HAS_RESIDUAL
        ? reinterpret_cast<DTYPE_I*>(residual.data_ptr())
        : nullptr;

    DTYPE_I* out_normed_ptr = HAS_OUT_NORMED
        ? reinterpret_cast<DTYPE_I*>(out_normed.data_ptr())
        : nullptr;

    gemma_rmsnorm_fp8_group_quant_kernel<
        DTYPE_I, DTYPE_O, 128, THREAD_DATA_SIZE,
        BLOCK_SIZE, MAX_GROUPS_PER_THREAD, TRANSPOSE_SCALE, HAS_RESIDUAL,
        HAS_OUT_NORMED>
        <<<grid, block, 0, stream>>>(
            reinterpret_cast<DTYPE_O*>(out.data_ptr()),
            reinterpret_cast<float*>(scale.data_ptr()),
            reinterpret_cast<DTYPE_I const*>(x.data_ptr()),
            residual_ptr,
            reinterpret_cast<DTYPE_I const*>(weight.data_ptr()),
            out_normed_ptr,
            epsilon,
            num_tokens,
            hidden_size
        );

    C10_HIP_KERNEL_LAUNCH_CHECK();
}

/**
 * Dispatcher
 */
template <typename DTYPE_I, typename DTYPE_O>
void gemma_rmsnorm_fp8_group_quant_launcher(
    torch::Tensor& out,
    torch::Tensor& scale,
    torch::Tensor const& x,
    torch::Tensor& residual,
    torch::Tensor const& weight,
    torch::Tensor& out_normed,
    double epsilon,
    int group_size,
    bool transpose_scale,
    bool has_residual,
    bool has_out_normed)
{
    TORCH_CHECK(x.dim() == 2, "Input x must be 2D: [num_tokens, hidden_size]");
    const int num_tokens = x.size(0);
    const int hidden_size = x.size(1);

    TORCH_CHECK(hidden_size % 128 == 0,
                "hidden_size must be a multiple of 128, got ", hidden_size);
    TORCH_CHECK(group_size == 128,
                "ONLY group_size=128 is supported, got ", group_size);
    TORCH_CHECK(weight.size(0) == hidden_size,
                "Weight size must match hidden_size");

    if (has_residual) {
        TORCH_CHECK(residual.dim() == 2 &&
                    residual.size(0) == num_tokens &&
                    residual.size(1) == hidden_size,
                    "Residual must have same shape as x");
    }

    if (has_out_normed) {
        TORCH_CHECK(out_normed.dim() == 2 &&
                    out_normed.size(0) == num_tokens &&
                    out_normed.size(1) == hidden_size,
                    "out_normed must have same shape as x");
    }

    constexpr int THREAD_DATA_SIZE = 16;
    constexpr int threads_per_group = 128 / THREAD_DATA_SIZE;  // 8
    constexpr int BLOCK_SIZE = 256;
    constexpr int groups_per_block = BLOCK_SIZE / threads_per_group;  // 32
    int num_groups = hidden_size / 128;
    int max_groups_per_thread = (num_groups + groups_per_block - 1) / groups_per_block;

    TORCH_CHECK(max_groups_per_thread <= 3,
                "hidden_size too large: ", hidden_size,
                " requires max_groups_per_thread=", max_groups_per_thread,
                " (max 3, i.e., hidden_size <= ", 3 * groups_per_block * 128, ")");

    // Dispatch on max_groups_per_thread, transpose_scale, has_residual, has_out_normed
    #define DISPATCH_INNER(MGT, OUT_NORMED)                                      \
        if (transpose_scale && has_residual) {                                   \
            gemma_rmsnorm_fp8_group_quant_launcher_impl<                         \
                DTYPE_I, DTYPE_O, THREAD_DATA_SIZE, BLOCK_SIZE,                  \
                MGT, true, true, OUT_NORMED>(                                    \
                out, scale, x, residual, weight, out_normed, epsilon,            \
                num_tokens, hidden_size);                                        \
        } else if (transpose_scale) {                                            \
            gemma_rmsnorm_fp8_group_quant_launcher_impl<                         \
                DTYPE_I, DTYPE_O, THREAD_DATA_SIZE, BLOCK_SIZE,                  \
                MGT, true, false, OUT_NORMED>(                                   \
                out, scale, x, residual, weight, out_normed, epsilon,            \
                num_tokens, hidden_size);                                        \
        } else if (has_residual) {                                               \
            gemma_rmsnorm_fp8_group_quant_launcher_impl<                         \
                DTYPE_I, DTYPE_O, THREAD_DATA_SIZE, BLOCK_SIZE,                  \
                MGT, false, true, OUT_NORMED>(                                   \
                out, scale, x, residual, weight, out_normed, epsilon,            \
                num_tokens, hidden_size);                                        \
        } else {                                                                 \
            gemma_rmsnorm_fp8_group_quant_launcher_impl<                         \
                DTYPE_I, DTYPE_O, THREAD_DATA_SIZE, BLOCK_SIZE,                  \
                MGT, false, false, OUT_NORMED>(                                  \
                out, scale, x, residual, weight, out_normed, epsilon,            \
                num_tokens, hidden_size);                                        \
        }

    #define DISPATCH_MGT(OUT_NORMED)                                             \
        if (max_groups_per_thread <= 1) {                                        \
            DISPATCH_INNER(1, OUT_NORMED)                                        \
        } else if (max_groups_per_thread <= 2) {                                 \
            DISPATCH_INNER(2, OUT_NORMED)                                        \
        } else {                                                                 \
            DISPATCH_INNER(3, OUT_NORMED)                                        \
        }

    if (has_out_normed) {
        DISPATCH_MGT(true)
    } else {
        DISPATCH_MGT(false)
    }

    #undef DISPATCH_MGT
    #undef DISPATCH_INNER
}

/**
 * Python-facing interface
 *
 * Always produces FP8 quantized output (out + scale).
 * Optionally also writes unquantized normed output to out_normed.
 */
void gemma_rmsnorm_fp8_group_quant(
    torch::Tensor& out,
    torch::Tensor& scale,
    torch::Tensor const& x,
    torch::Tensor const& weight,
    double epsilon,
    int group_size,
    bool transpose_scale,
    c10::optional<torch::Tensor> residual,
    c10::optional<torch::Tensor> out_normed)
{
    TORCH_CHECK(x.is_cuda(), "Input x must be on CUDA device");
    TORCH_CHECK(weight.is_cuda(), "Weight must be on CUDA device");
    TORCH_CHECK(out.is_cuda(), "Output must be on CUDA device");
    TORCH_CHECK(scale.is_cuda(), "Scale must be on CUDA device");

    bool has_residual = residual.has_value();
    torch::Tensor residual_tensor = has_residual
        ? residual.value()
        : torch::empty({0}, x.options());

    bool has_out_normed = out_normed.has_value();
    torch::Tensor out_normed_tensor = has_out_normed
        ? out_normed.value()
        : torch::empty({0}, x.options());

    if (has_residual) {
        TORCH_CHECK(residual_tensor.is_cuda(),
                    "Residual must be on CUDA device");
    }

    if (has_out_normed) {
        TORCH_CHECK(out_normed_tensor.is_cuda(),
                    "out_normed must be on CUDA device");
        TORCH_CHECK(out_normed_tensor.scalar_type() == x.scalar_type(),
                    "out_normed dtype must match input dtype. "
                    "Input: ", x.scalar_type(),
                    ", out_normed: ", out_normed_tensor.scalar_type());
    }

    TORCH_CHECK(out.scalar_type() == at::ScalarType::Float8_e4m3fnuz ||
                out.scalar_type() == at::ScalarType::Float8_e4m3fn,
                "Output must be FP8 dtype, got ", out.scalar_type());

    if (x.scalar_type() == at::ScalarType::BFloat16) {
        gemma_rmsnorm_fp8_group_quant_launcher<opus::bf16_t, opus::fp8_t>(
            out, scale, x, residual_tensor, weight, out_normed_tensor,
            epsilon, group_size, transpose_scale, has_residual, has_out_normed);
    } else if (x.scalar_type() == at::ScalarType::Half) {
        gemma_rmsnorm_fp8_group_quant_launcher<opus::fp16_t, opus::fp8_t>(
            out, scale, x, residual_tensor, weight, out_normed_tensor,
            epsilon, group_size, transpose_scale, has_residual, has_out_normed);
    } else {
        TORCH_CHECK(false,
                    "Unsupported input dtype: ", x.scalar_type());
    }
}

} // namespace aiter
