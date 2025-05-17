// SPDX-License-Identifier: MIT
// Copyright (C) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.
#pragma once
#include "aiter_hip_common.h"

namespace ck_tile
{
    using int8x2_t = ext_vector_t<int8_t, 2>;
    using fp8x2_t = ext_vector_t<fp8_t, 2>;
    using fp8x2_t = ext_vector_t<fp8_t, 2>;
    using fp32x2_t = ext_vector_t<fp32_t, 2>;
    // using fp4x2_t = unsigned _BitInt(8);
    // using fp4x2x2_t = ext_vector_t<fp4x2_t, 2>;

    CK_TILE_DEVICE fp32x2_t amd_assembly_pk_mul_f32(fp32x2_t a, fp32x2_t b)
    {
        fp32x2_t c;
        asm volatile("v_pk_mul_f32 %0, %1, %2" : "=v"(c) : "v"(a), "v"(b));
        return c;
    }
    CK_TILE_DEVICE fp8x2_t amd_assembly_cvt_pk_fp8_f32(fp32_t a, fp32_t b)
    {
        int16x2_t c;
        asm volatile("v_cvt_pk_fp8_f32 %0, %1, %2" : "=v"(c) : "v"(a), "v"(b));
        return bit_cast<fp8x2_t>(c[0]);
    }
    CK_TILE_DEVICE fp8x2_t amd_assembly_cvt_pk_bf8_f32(fp32_t a, fp32_t b)
    {
        int16x2_t c;
        asm volatile("v_cvt_pk_bf8_f32 %0, %1, %2" : "=v"(c) : "v"(a), "v"(b));
        return bit_cast<fp8x2_t>(c[0]);
    }
    //     CK_TILE_DEVICE fp4x2_t amd_assembly_cvt_scalef32_pk_fp4_f32(fp32_t a, fp32_t b, float scale)
    //     {
    // #if defined(__gfx950__)
    //         int8x4_t c;
    //         // permute high bits and low bits to match the order of the original vector
    //         asm volatile("v_cvt_scalef32_pk_fp4_f32 %0, %1, %2" : "=v"(c) : "v"(b), "v"(a), "v"(scale));
    //         return bit_cast<fp4x2_t>(c[0]);
    // #endif
    //     }
    //     CK_TILE_DEVICE fp4x2_t amd_assembly_cvt_scalef32_pk_fp4_f16(fp16_t a, fp16_t b, float scale)
    //     {
    // #if defined(__gfx950__)
    //         int8x4_t c;
    //         // permute high bits and low bits to match the order of the original vector
    //         asm volatile("v_cvt_scalef32_pk_fp4_f16 %0, %1, %2" : "=v"(c) : "v"(b), "v"(a), "v"(scale));
    //         return bit_cast<fp4x2_t>(c[0]);
    // #endif
    //     }
    //     CK_TILE_DEVICE fp4x2_t amd_assembly_cvt_scalef32_pk_fp4_bf16(bf16_t a, bf16_t b, float scale)
    //     {
    // #if defined(__gfx950__)
    //         int8x4_t c;
    //         // permute high bits and low bits to match the order of the original vector
    //         asm volatile("v_cvt_scalef32_pk_fp4_bf16 %0, %1, %2" : "=v"(c) : "v"(b), "v"(a), "v"(scale));
    //         return bit_cast<fp4x2_t>(c[0]);
    // #endif
    //     }

    // convert any to fp32x?_t one by one
    template <typename Y,
              typename X,
              index_t N,
              std::enable_if_t<(std::is_same_v<Y, fp32_t>), bool> = false>
    CK_TILE_HOST_DEVICE constexpr ext_vector_t<Y, N> vec_convert(ext_vector_t<X, N> x)
    {
        using fp32xX_t = ext_vector_t<Y, N>;
        fp32xX_t tmp;
        for (size_t i = 0; i < N; i++)
        {
            tmp[i] = type_convert<Y>(x[i]);
        }
        return tmp;
    }

    template <typename Y,
              typename X,
              index_t N,
              std::enable_if_t<(N % 2 == 0), bool> = false>
    CK_TILE_HOST_DEVICE constexpr ext_vector_t<Y, N> vec_convert(ext_vector_t<X, N> x, float inverted_scale)
    {
        if constexpr (!std::is_same_v<X, fp32_t>)
        {
            using fp32xX_t = ext_vector_t<fp32_t, N>;
            fp32xX_t tmp = vec_convert<fp32_t, X, N>(x);
            return vec_convert<Y, fp32_t, N>(tmp, inverted_scale);
        }
        else
        {
            // fp32->??
            return vec_convert<Y, fp32_t, N>(x, inverted_scale);
        }
    }

    // fp32x2 -> fp8x2
    CK_TILE_HOST_DEVICE constexpr fp8x2_t fp32x2_t_to_fp8x2_t(fp32x2_t x, float inverted_scale)
    {
        using vec_ti = vector_traits<fp32x2_t>;
        constexpr int vec_size = vec_ti::vector_size;
        constexpr auto interpret = numeric_traits<fp8_t>::f8_interpret;
        fp32x2_t tmp = amd_assembly_pk_mul_f32(x, fp32x2_t{inverted_scale, inverted_scale});

        return (interpret == fp8_interpretation::E4M3_FNUZ) || (interpret == fp8_interpretation::E4M3_OCP)
                   ? amd_assembly_cvt_pk_fp8_f32(tmp[0], tmp[1])
                   : amd_assembly_cvt_pk_bf8_f32(tmp[0], tmp[1]);
    }
    // fp32x2 -> int8x2
    CK_TILE_HOST_DEVICE constexpr int8x2_t fp32x2_t_to_int8x2_t(fp32x2_t x, float inverted_scale)
    {
        fp32x2_t tmp = amd_assembly_pk_mul_f32(x, fp32x2_t{inverted_scale, inverted_scale});

        int8x2_t out;
        out[0] = static_cast<int8_t>(tmp[0]);
        out[1] = static_cast<int8_t>(tmp[1]);
        return out;
    }
    // fp32x2 -> int8x2
    CK_TILE_HOST_DEVICE constexpr int8x2_t fp32x2x2_t_to_fp4x2x2_t(fp32x2_t x, float inverted_scale)
    {
        fp32x2_t tmp = amd_assembly_pk_mul_f32(x, fp32x2_t{inverted_scale, inverted_scale});

        int8x2_t out;
        out[0] = static_cast<int8_t>(tmp[0]);
        out[1] = static_cast<int8_t>(tmp[1]);
        return out;
    }
    // // fp32x2 -> fp4x2
    // CK_TILE_HOST_DEVICE constexpr fp4x2_t fp32x2_t_to_fp4x2_t(fp32x2_t x, float inverted_scale)
    // {
    //     return amd_assembly_cvt_scalef32_pk_fp4_f32(x[0], x[1], inverted_scale);
    // }
    // // fp16x2 -> fp4x2
    // CK_TILE_HOST_DEVICE constexpr fp4x2x2_t fp16x2x2_t_to_fp4x2x2_t(fp16x2x2_t x, float inverted_scale)
    // {
    //     return {amd_assembly_cvt_scalef32_pk_fp4_f16(x[0][0], x[0][1], inverted_scale),
    //             amd_assembly_cvt_scalef32_pk_fp4_f16(x[1][0], x[1][1], inverted_scale)};
    // }
    // // bf16x2 -> fp4x2
    // CK_TILE_HOST_DEVICE constexpr fp4x2x2_t bf16x2x2_t_to_fp4x2x2_t(bf16x2x2_t x, float inverted_scale)
    // {
    //     return {amd_assembly_cvt_scalef32_pk_fp4_bf16(x[0][0], x[0][1], inverted_scale),
    //             amd_assembly_cvt_scalef32_pk_fp4_bf16(x[1][0], x[1][1], inverted_scale)};
    // }
#define CK_TILE_TYPE_CONVERT(dtype_, stype_, vec_size_)                                                                                                                           \
    template <>                                                                                                                                                                   \
    CK_TILE_HOST_DEVICE constexpr ext_vector_t<dtype_##_t, vec_size_> vec_convert<dtype_##_t, stype_##_t, vec_size_>(ext_vector_t<stype_##_t, vec_size_> x, float inverted_scale) \
    {                                                                                                                                                                             \
        constexpr int iter_num = vec_size_ / 2;                                                                                                                                   \
        ext_vector_t<dtype_##_t, vec_size_> out;                                                                                                                                  \
        for (size_t i = 0; i < iter_num; i++)                                                                                                                                     \
        {                                                                                                                                                                         \
            auto tmp = stype_##x2##_t_to_##dtype_##x2##_t(stype_##x2##_t{x[i * 2], x[i * 2 + 1]}, inverted_scale);                                                                \
            out[i * 2] = tmp[0];                                                                                                                                                  \
            out[i * 2 + 1] = tmp[1];                                                                                                                                              \
        }                                                                                                                                                                         \
        return out;                                                                                                                                                               \
    }
    CK_TILE_TYPE_CONVERT(fp8, fp32, 2)
    CK_TILE_TYPE_CONVERT(fp8, fp32, 4)
    CK_TILE_TYPE_CONVERT(fp8, fp32, 8)
    CK_TILE_TYPE_CONVERT(fp8, fp32, 16)

    CK_TILE_TYPE_CONVERT(int8, fp32, 2)
    CK_TILE_TYPE_CONVERT(int8, fp32, 4)
    CK_TILE_TYPE_CONVERT(int8, fp32, 8)
    CK_TILE_TYPE_CONVERT(int8, fp32, 16)

    // CK_TILE_TYPE_CONVERT(fp4, fp32, 2)
    // CK_TILE_TYPE_CONVERT(fp4, fp32, 4)
    // CK_TILE_TYPE_CONVERT(fp4, fp32, 8)
    // CK_TILE_TYPE_CONVERT(fp4, fp32, 16)

    // CK_TILE_TYPE_CONVERT(fp4x2, fp16x2, 2)
    // CK_TILE_TYPE_CONVERT(fp4x2, fp16x2, 4)
    // CK_TILE_TYPE_CONVERT(fp4x2, fp16x2, 8)
    // CK_TILE_TYPE_CONVERT(fp4x2, fp16x2, 16)

    // CK_TILE_TYPE_CONVERT(fp4x2, bf16x2, 2)
    // CK_TILE_TYPE_CONVERT(fp4x2, bf16x2, 4)
    // CK_TILE_TYPE_CONVERT(fp4x2, bf16x2, 8)
    // CK_TILE_TYPE_CONVERT(fp4x2, bf16x2, 16)
#undef CK_TILE_TYPE_CONVERT

} // namespace aiter
