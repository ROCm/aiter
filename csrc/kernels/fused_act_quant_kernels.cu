// SPDX-License-Identifier: MIT
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include <ATen/hip/HIPContext.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <torch/extension.h>

#include <cmath>

#include "aiter_hip_common.h"
#include "ck_tile/core.hpp"
#include "ck_tile/ops/elementwise/unary_element_wise_operation.hpp"
#include "dispatch_utils.h"
#include "hip_compat.h"
#include "hip_reduce.h"
#include "py_itfs_common.h"
#include "quant_common.cuh"
#include "vec_convert.h"
#include <hip/hip_bf16.h>
#include <hipcub/hipcub.hpp>

using fp8_type = ck_tile::fp8_t;

static constexpr int32_t max_vec_size = 8;
static constexpr int32_t max_wave_num = 8;
static constexpr int32_t BlockSize = 256;

namespace aiter {

// Activation function templates
template <typename T>
__device__ __forceinline__ float silu_kernel(const T& x)
{
    // x * sigmoid(x)
    constexpr auto one = ck_tile::type_convert<float>(1);
    float x_           = ck_tile::type_convert<float>(x);
    float y            = x_ * __builtin_amdgcn_rcpf(one + ck_tile::exp(-x_));
    return y;
}

template <typename T>
__device__ __forceinline__ float gelu_kernel(const T& x)
{
    // Equivalent to PyTorch GELU with 'none' approximation.
    const float f         = ck_tile::type_convert<float>(x);
    constexpr float ALPHA = M_SQRT1_2;
    return f * 0.5f * (1.0f + ::erf(f * ALPHA));
}

template <typename T>
__device__ __forceinline__ float gelu_tanh_kernel(const T& x)
{
    // Equivalent to PyTorch GELU with 'tanh' approximation.
    const float f         = ck_tile::type_convert<float>(x);
    constexpr float BETA  = M_SQRT2 * M_2_SQRTPI * 0.5f;
    constexpr float KAPPA = 0.044715;
    float x_cube          = f * f * f;
    float inner           = BETA * (f + KAPPA * x_cube);
    return 0.5f * f * (1.0f + ::tanhf(inner));
}

// Fused activation+mul+per_token_quant kernel
// This kernel combines act_and_mul and dynamic_per_token_scaled_quant
template <typename DTYPE_I, typename DTYPE_O, float (*ACT_FN)(const DTYPE_I&), int32_t VEC_SIZE_I>
__global__ void fused_act_mul_quant_kernel(
    DTYPE_O* __restrict__ out,         // [..., d]
    float* __restrict__ scale,         // [num_tokens]
    const DTYPE_I* __restrict__ input, // [..., 2, d]
    const int d)
{
    const int64_t token_idx = blockIdx.x;

    // Pointers to the two halves of input
    auto const* ptr_x = (input + token_idx * 2 * d);
    auto const* ptr_y = (input + token_idx * 2 * d + d);

    using vec_i = ck_tile::vec_t<DTYPE_I, VEC_SIZE_I>;
    static constexpr int32_t ooba_i = 4 / sizeof(DTYPE_I);
    const int32_t oob_i = (d + ooba_i - 1) / ooba_i * ooba_i;

    auto buffer_x = ck_tile::make_buffer_view<ck_tile::address_space_enum::global>(ptr_x, oob_i);
    auto buffer_y = ck_tile::make_buffer_view<ck_tile::address_space_enum::global>(ptr_y, oob_i);
    buffer_x.init_raw();
    buffer_y.init_raw();

    // Phase 1: Apply activation and multiply, computing absMax
    float absMax = 1e-10f;

    for(int64_t idx = threadIdx.x * VEC_SIZE_I; idx < d; idx += blockDim.x * VEC_SIZE_I)
    {
        vec_i x = buffer_x.template get<vec_i>(idx, 0, true);
        vec_i y = buffer_y.template get<vec_i>(idx, 0, true);

        // Compute activation and multiply, track absMax
#pragma unroll
        for(size_t j = 0; j < VEC_SIZE_I; j++)
        {
            if(idx + j < d)
            {
                float ax = ACT_FN(x[j]);
                float yv = ck_tile::type_convert<float>(y[j]);
                float result = ax * yv;
                absMax = max(absMax, abs(result));
            }
        }
    }

    // Phase 2: Block reduction to find the maximum absolute value
    absMax = block_reduce<float, hipcub::Max, BlockSize, true>(absMax, hipcub::Max());

    // Phase 3: Compute scale
    const float inverted_DTYPE_MAX =
        std::is_same_v<DTYPE_O, ck_tile::fp4x2_t>
            ? 0.25f
            : (1.0f / ck_tile::type_convert<float>(ck_tile::numeric<DTYPE_O>::max()));

    auto fp4_scale = [](float tmp) {
        uint32_t u32 = ck_tile::bit_cast<uint32_t>(tmp);
        uint32_t exponent = (u32 >> 23) & 0b11111111;
        if(exponent == 0b11111111)
        {
            return ck_tile::bit_cast<float>(exponent << 23);
        }
        if(((u32 & 0x400000)) && (((u32 & 0x200000)) || ((u32 & 0x1FFFFF)) || (exponent)))
            exponent += 1;
        return ck_tile::bit_cast<float>(exponent << 23);
    };

    float row_scale = std::is_same_v<DTYPE_O, ck_tile::fp4x2_t>
                          ? fp4_scale(absMax) * inverted_DTYPE_MAX
                          : absMax * inverted_DTYPE_MAX;

    // Thread 0 stores the scale
    if(threadIdx.x == 0)
    {
        if constexpr(std::is_same_v<DTYPE_O, ck_tile::fp4x2_t>)
        {
            auto* tmp = reinterpret_cast<uint8_t*>(scale);
            uint8_t exponent = (ck_tile::bit_cast<uint32_t>(row_scale) >> 23) & 0b11111111;
            tmp[token_idx] = exponent;
        }
        else
        {
            scale[token_idx] = row_scale;
        }
    }

    __syncthreads();

    // Phase 4: Quantize and write output
    const float inverted_scale = std::is_same_v<DTYPE_O, ck_tile::fp4x2_t> ? row_scale : 1.0f / row_scale;

    using DTYPE_STORE = typename ck_tile::vector_traits<DTYPE_O>::scalar_type;
    static constexpr int32_t ooba_o = 4 / sizeof(DTYPE_O);
    const int64_t oob_o = std::is_same_v<DTYPE_O, ck_tile::fp4x2_t>
                              ? ((token_idx + 1) * d / 2 + ooba_o - 1) / ooba_o * ooba_o
                              : ((token_idx + 1) * d + ooba_o - 1) / ooba_o * ooba_o;

    auto* ptr_o = std::is_same_v<DTYPE_O, ck_tile::fp4x2_t>
                      ? reinterpret_cast<DTYPE_STORE*>(out + token_idx * d / 2)
                      : reinterpret_cast<DTYPE_STORE*>(out + token_idx * d);

    auto buffer_o = ck_tile::make_buffer_view<ck_tile::address_space_enum::global>(ptr_o, oob_o);
    buffer_o.init_raw();

    for(int64_t idx = threadIdx.x * VEC_SIZE_I; idx < d; idx += blockDim.x * VEC_SIZE_I)
    {
        vec_i x = buffer_x.template get<vec_i>(idx, 0, true);
        vec_i y = buffer_y.template get<vec_i>(idx, 0, true);

        // Recompute activation and multiply, then quantize
        if constexpr(VEC_SIZE_I == 1)
        {
            // Special case for VEC_SIZE=1: quantize directly without vec_convert
            float ax = ACT_FN(x[0]);
            float yv = ck_tile::type_convert<float>(y[0]);
            float result_fp = ax * yv;

            DTYPE_O quantized;
            if constexpr(std::is_same_v<DTYPE_O, ck_tile::fp4x2_t>)
            {
                // FP4 not supported for scalar case
                assert(false && "FP4 requires VEC_SIZE >= 2");
            }
            else
            {
                quantized = ck_tile::type_convert<DTYPE_O>(result_fp * inverted_scale);
            }

            if(idx < d)
            {
                out[token_idx * d + idx] = quantized;
            }
        }
        else
        {
            // VEC_SIZE >= 2: use vec_convert
            vec_i result;
#pragma unroll
            for(size_t j = 0; j < VEC_SIZE_I; j++)
            {
                float ax = ACT_FN(x[j]);
                float yv = ck_tile::type_convert<float>(y[j]);
                result[j] = ck_tile::type_convert<DTYPE_I>(ax * yv);
            }

            // Quantize - use vec_convert which requires VEC_SIZE >= 2
            static constexpr int32_t vec_size_o =
                std::is_same_v<DTYPE_O, ck_tile::fp4x2_t> ? VEC_SIZE_I / 2 : VEC_SIZE_I;

            auto out_s = ck_tile::vec_convert<DTYPE_O, DTYPE_I, VEC_SIZE_I>(result, inverted_scale)
                             .template get_as<DTYPE_STORE>();

            // Write output
            int64_t out_idx = std::is_same_v<DTYPE_O, ck_tile::fp4x2_t> ? idx / 2 : idx;

            if constexpr(VEC_SIZE_I <= 16)
            {
                buffer_o.template set(out_idx, 0, true, out_s);
            }
            else
            {
                static constexpr int32_t o_step = std::is_same_v<DTYPE_O, ck_tile::fp4x2_t> ? 8 : 16;
                assert(VEC_SIZE_I % 16 == 0);
                using vecT = ck_tile::vec_t<DTYPE_STORE, o_step>;
                auto vec = out_s.template get_as<vecT>();
                static constexpr int32_t num_iter = VEC_SIZE_I / 16;

                for(size_t j = 0; j < num_iter; j++)
                {
                    buffer_o.template set(out_idx + j * o_step, 0, true, vec[j]);
                }
            }
        }
    }
}

} // namespace aiter

static constexpr int nextPow2(unsigned int num)
{
    if(num <= 1)
        return 1;
    return 1 << (CHAR_BIT * sizeof(num) - __builtin_clz(num - 1));
}

// Launch fused activation+mul+quant kernel
#define LAUNCH_FUSED_ACT_MUL_QUANT_KERNEL(KERNEL, DTYPE_O)                                      \
    int d = input.size(-1) / 2;                                                                 \
    int64_t num_tokens = input.numel() / input.size(-1);                                        \
    int vec_size = nextPow2(d / 64);                                                            \
    vec_size = vec_size < 2 ? 2 : vec_size;                                                    \
    vec_size = vec_size > max_vec_size ? max_vec_size : vec_size;                              \
    int num_wave = nextPow2(d / 64 / vec_size);                                                 \
    num_wave = num_wave > max_wave_num ? max_wave_num : num_wave;                              \
    dim3 grid(num_tokens);                                                                      \
    dim3 block(num_wave * 64);                                                                  \
    const at::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(device_of(input));          \
    const hipStream_t stream = at::hip::getCurrentHIPStream();                                  \
    AITER_DISPATCH_FLOATING16_TYPES(input.scalar_type(), "fused_act_mul_quant_kernel", [&] {   \
        using input_dtype = typename t2ck<scalar_t>::type;                                     \
        AITER_DISPATCH_CASE_VEC_SIZE(                                                           \
            vec_size,                                                                           \
            aiter::fused_act_mul_quant_kernel<input_dtype, DTYPE_O, KERNEL<input_dtype>, VEC_SIZE> \
            <<<grid, block, 0, stream>>>(                                                       \
                reinterpret_cast<DTYPE_O*>(out.data_ptr()),                                    \
                scales.data_ptr<float>(),                                                       \
                reinterpret_cast<input_dtype*>(input.data_ptr()),                              \
                d);)                                                                            \
    });

namespace aiter {

void fused_silu_mul_quant(torch::Tensor& out,     // [..., d]
                          torch::Tensor& scales,   // [num_tokens]
                          torch::Tensor& input)    // [..., 2 * d]
{
    if(out.dtype() == torch_fp8)
    {
        LAUNCH_FUSED_ACT_MUL_QUANT_KERNEL(aiter::silu_kernel, FP8_TYPE);
    }
    else if(out.dtype() == torch::kInt8)
    {
        LAUNCH_FUSED_ACT_MUL_QUANT_KERNEL(aiter::silu_kernel, ck_tile::int8_t);
    }
#if defined(__Float4_e2m1fn_x2)
    else if(out.dtype() == torch_fp4x2)
    {
        LAUNCH_FUSED_ACT_MUL_QUANT_KERNEL(aiter::silu_kernel, ck_tile::fp4x2_t);
    }
#endif
    else
    {
        TORCH_CHECK(false, __func__, " not support output type: ", out.dtype());
    }
}

void fused_gelu_mul_quant(torch::Tensor& out,     // [..., d]
                          torch::Tensor& scales,   // [num_tokens]
                          torch::Tensor& input)    // [..., 2 * d]
{
    if(out.dtype() == torch_fp8)
    {
        LAUNCH_FUSED_ACT_MUL_QUANT_KERNEL(aiter::gelu_kernel, FP8_TYPE);
    }
    else if(out.dtype() == torch::kInt8)
    {
        LAUNCH_FUSED_ACT_MUL_QUANT_KERNEL(aiter::gelu_kernel, ck_tile::int8_t);
    }
#if defined(__Float4_e2m1fn_x2)
    else if(out.dtype() == torch_fp4x2)
    {
        LAUNCH_FUSED_ACT_MUL_QUANT_KERNEL(aiter::gelu_kernel, ck_tile::fp4x2_t);
    }
#endif
    else
    {
        TORCH_CHECK(false, __func__, " not support output type: ", out.dtype());
    }
}

void fused_gelu_tanh_mul_quant(torch::Tensor& out,     // [..., d]
                               torch::Tensor& scales,   // [num_tokens]
                               torch::Tensor& input)    // [..., 2 * d]
{
    if(out.dtype() == torch_fp8)
    {
        LAUNCH_FUSED_ACT_MUL_QUANT_KERNEL(aiter::gelu_tanh_kernel, FP8_TYPE);
    }
    else if(out.dtype() == torch::kInt8)
    {
        LAUNCH_FUSED_ACT_MUL_QUANT_KERNEL(aiter::gelu_tanh_kernel, ck_tile::int8_t);
    }
#if defined(__Float4_e2m1fn_x2)
    else if(out.dtype() == torch_fp4x2)
    {
        LAUNCH_FUSED_ACT_MUL_QUANT_KERNEL(aiter::gelu_tanh_kernel, ck_tile::fp4x2_t);
    }
#endif
    else
    {
        TORCH_CHECK(false, __func__, " not support output type: ", out.dtype());
    }
}

} // namespace aiter
