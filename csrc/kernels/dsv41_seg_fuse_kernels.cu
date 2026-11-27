// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

// DeepSeek-V4.1 shared-expert segment fusion kernels (its own JIT module, so
// that nothing here rebuilds module_activation / module_mega_mhc).
// Ported from the DeepSeek-V4.1 DEP8 AgentX PoC (plugin_poc PR #129, Bin Ding).
//
// silu_mul_mxfp8 = act_and_mul_kernel<bf16, bf16, silu, 8, HAS_LIMIT>
//                  (csrc/kernels/activation_kernels.cu) with the group-32
//                  MXFP8 quant of MegaMHCQuantOutput<true>::store
//                  (csrc/kernels/mega_mhc.cu) as its epilogue.
// Both halves are copied statement for statement and this module is built with
// module_activation's flags, so the BF16 intermediate and the codes are the
// ones the two separate launches produce.
#define AITER_NO_TORCH_TYPES
#include <cmath>
#include <limits>

#include "aiter_hip_common.h"
#include "aiter_opus_plus.h"
#include "aiter_stream.h"
#include "aiter_tensor.h"
#include "dsv41_seg_fuse.h"
#include "quant_amax_floor.h"

namespace aiter {
namespace seg_fuse {

// activation_kernels.cu: silu_kernel
template <typename T>
__device__ __forceinline__ float silu_kernel(const T& x)
{
    // x * sigmoid(x)
    constexpr float one = 1.0f;
    float x_            = opus::cast<float>(x);
    float y             = x_ * __builtin_amdgcn_rcpf(one + __ocml_exp_f32(-x_));
    return y;
}

static constexpr int32_t VEC = 8; // 8 bf16 a lane; four adjacent lanes own a group32

template <bool HAS_LIMIT>
__global__ void silu_mul_mxfp8_kernel(uint8_t* __restrict__ out_q,            // [T, d]
                                      uint8_t* __restrict__ out_e,            // [T, d / 32]
                                      const opus::bf16_t* __restrict__ input, // [T, 2, d]
                                      const int d,
                                      const float limit)
{
#if defined(__gfx950__)
    using DTYPE_I                   = opus::bf16_t;
    using DTYPE_O                   = opus::bf16_t;
    using vec_i                     = opus::vector_t<DTYPE_I, VEC>;
    using vec_o                     = opus::vector_t<DTYPE_O, VEC>;
    const int64_t token_idx         = blockIdx.x;
    auto const* ptr_x               = (input + token_idx * 2 * d);
    auto const* ptr_y               = (input + token_idx * 2 * d + d);
    static constexpr int32_t ooba_i = 4 / sizeof(DTYPE_I);
    const int32_t oob_i             = (d + ooba_i - 1) / ooba_i * ooba_i;
    auto buffer_x                   = opus::make_gmem<DTYPE_I>(ptr_x, oob_i * sizeof(DTYPE_I));
    auto buffer_y                   = opus::make_gmem<DTYPE_I>(ptr_y, oob_i * sizeof(DTYPE_I));
    auto codes                      = opus::make_gmem<uint8_t>(out_q + token_idx * d, d);
    auto scales = opus::make_gmem<uint8_t>(out_e + token_idx * (d / 32), d / 32);

    for(int64_t idx = threadIdx.x * VEC; idx < d; idx += blockDim.x * VEC)
    {
        vec_i x{};
        vec_i y{};
        x = load_vector_nbytes<DTYPE_I, VEC, 16, GROUP_NT>(buffer_x, idx);
        y = load_vector_nbytes<DTYPE_I, VEC, 16, GROUP_NT>(buffer_y, idx);

        vec_o r{};

        // ---- activation_kernels.cu: act_and_mul_kernel body ----
#pragma unroll
        for(size_t j = 0; j < VEC; j += 2)
        {
            DTYPE_I x_val0 = x[j];
            float y0       = opus::cast<float>(y[j]);
            if constexpr(HAS_LIMIT)
            {
                float fx0 = opus::cast<float>(x_val0);
                fx0       = fminf(fx0, limit);
                x_val0    = opus::cast<DTYPE_I>(fx0);
                y0        = __builtin_amdgcn_fmed3f(-limit, y0, limit);
            }
            float ax0      = silu_kernel<DTYPE_I>(x_val0);
            DTYPE_I x_val1 = x[j + 1];
            float y1       = opus::cast<float>(y[j + 1]);
            if constexpr(HAS_LIMIT)
            {
                float fx1 = opus::cast<float>(x_val1);
                fx1       = fminf(fx1, limit);
                x_val1    = opus::cast<DTYPE_I>(fx1);
                y1        = __builtin_amdgcn_fmed3f(-limit, y1, limit);
            }
            float ax1        = silu_kernel<DTYPE_I>(x_val1);
            opus::fp32x2_t a = {ax0, ax1};
            opus::fp32x2_t b = {y0, y1};
            opus::fp32x2_t c;
            asm volatile("v_pk_mul_f32 %0, %1, %2" : "=v"(c) : "v"(a), "v"(b));
            r[j]     = opus::cast<DTYPE_O>(c.x);
            r[j + 1] = opus::cast<DTYPE_O>(c.y);
        }

        // ---- mega_mhc.cu: MegaMHCQuantOutput<true>::store ----
        // Quantize the rounded BF16 result. Reinterpret its bits so the
        // compiler cannot bypass that rounding.
        using u32x4      = opus::vector_t<unsigned, 4>;
        const auto words = __builtin_bit_cast(u32x4, r);
        float amax       = aiter::kAmaxFloor;
#pragma unroll
        for(int k = 0; k < 4; ++k)
        {
            amax = fmaxf(amax, __uint_as_float((words[k] & 0x7fff) << 16));
            amax = fmaxf(amax, __uint_as_float(words[k] & 0x7fff0000));
        }
        // The lane quad's maximum: every lane of a quad is in the same
        // iteration (d and the block's columns are multiples of 32).
        amax = fmaxf(
            amax,
            __int_as_float(__builtin_amdgcn_mov_dpp(__float_as_int(amax), 0xb1, 15, 15, false)));
        amax = fmaxf(
            amax,
            __int_as_float(__builtin_amdgcn_mov_dpp(__float_as_int(amax), 0x4e, 15, 15, false)));
        const unsigned bits = __float_as_uint(amax * (1.0f / 448.0f));
        const int exponent  = min(max(static_cast<int>((bits + 0x7fffff) >> 23), 1), 254);
        const float scale   = __uint_as_float(exponent << 23);
        using u16x2         = unsigned short __attribute__((ext_vector_type(2)));
        using i16x2         = short __attribute__((ext_vector_type(2)));
        opus::vector_t<unsigned, 2> packed;
#pragma unroll
        for(int q = 0; q < 2; ++q)
        {
            i16x2 pair{};
            // Materialize scalars: HIP Clang can miscompile bit_cast directly
            // on an ext_vector subscript by using the first element instead.
            const unsigned lo = words[2 * q], hi = words[2 * q + 1];
            pair = __builtin_amdgcn_cvt_scalef32_pk_fp8_bf16(
                pair, __builtin_bit_cast(u16x2, lo), scale, false);
            pair = __builtin_amdgcn_cvt_scalef32_pk_fp8_bf16(
                pair, __builtin_bit_cast(u16x2, hi), scale, true);
            packed[q] = __builtin_bit_cast(unsigned, pair);
        }
        const auto bytes = __builtin_bit_cast(opus::vector_t<uint8_t, 8>, packed);
        const int i      = static_cast<int>(idx);
        store_vector_nbytes<uint8_t, uint8_t, 8, 8>(codes, bytes, i);
        opus::vector_t<uint8_t, 1> e;
        e[0] = exponent;
        // Only the first lane of each quad writes its exponent; buffer bounds
        // discard the other lanes' store.
        store_vector_nbytes<uint8_t, uint8_t, 1, 1>(scales, e, i % 32 == 0 ? i / 32 : d / 32);
    }
#endif
}

static constexpr int next_pow2(unsigned int num)
{
    if(num <= 1)
        return 1;
    return 1 << (CHAR_BIT * sizeof(num) - __builtin_clz(num - 1));
}

} // namespace seg_fuse

void silu_mul_mxfp8(const aiter_tensor_t& out,
                    const aiter_tensor_t& scale,
                    const aiter_tensor_t& input,
                    float limit)
{
    AITER_CHECK(limit >= 0.0f, "silu_mul_mxfp8: limit must be >= 0");
    AITER_CHECK(input.dtype() == AITER_DTYPE_bf16, "silu_mul_mxfp8: input must be BF16");
    AITER_CHECK(out.dtype() == AITER_DTYPE_fp8, "silu_mul_mxfp8: out must be FP8");
    AITER_CHECK(input.dim() >= 1 && input.size(-1) > 0 && input.size(-1) % 64 == 0,
                "silu_mul_mxfp8: input last dim must be 2 * d with d a multiple of 32");
    AITER_CHECK(input.is_contiguous() && out.is_contiguous() && scale.is_contiguous(),
                "silu_mul_mxfp8: tensors must be contiguous");
    const int d              = input.size(-1) / 2;
    const int64_t num_tokens = input.numel() / input.size(-1);
    AITER_CHECK(static_cast<int64_t>(out.numel()) == num_tokens * d,
                "silu_mul_mxfp8: out must be [T, d]");
    AITER_CHECK(static_cast<int64_t>(scale.numel()) == num_tokens * (d / 32),
                "silu_mul_mxfp8: scale must be [T, d / 32] bytes");
    AITER_CHECK(out.device_id == input.device_id && scale.device_id == input.device_id,
                "silu_mul_mxfp8: tensors must be on the same device");
    if(num_tokens == 0)
        return;
    // silu_and_mul's launch geometry at vec_size 8 (COMPUTE_ACTIVATION_KERNEL_PARAMS).
    const int warp_size = static_cast<int>(WARP_SIZE);
    int num_wave        = seg_fuse::next_pow2(d / warp_size / seg_fuse::VEC);
    num_wave            = num_wave > 8 ? 8 : num_wave;
    dim3 grid(num_tokens);
    dim3 block(num_wave * warp_size);
    HipDeviceGuard device_guard(input.device_id);
    const hipStream_t stream = aiter::getCurrentHIPStream();
    auto* q                  = reinterpret_cast<uint8_t*>(out.data_ptr());
    auto* e                  = reinterpret_cast<uint8_t*>(scale.data_ptr());
    auto* in                 = reinterpret_cast<const opus::bf16_t*>(input.data_ptr());
    if(limit > 0.0f)
        seg_fuse::silu_mul_mxfp8_kernel<true><<<grid, block, 0, stream>>>(q, e, in, d, limit);
    else
        seg_fuse::silu_mul_mxfp8_kernel<false><<<grid, block, 0, stream>>>(q, e, in, d, 0.0f);
}

} // namespace aiter
