// SPDX-License-Identifier: MIT
// Copyright (C) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.
#pragma once
#include "aiter_hip_common.h"

namespace ck_tile {
template <typename T, int N>
using vec_t = thread_buffer<T, N>;
// using vec_t = ext_vector_t<T, N>;

using int8x2_v = vec_t<int8_t, 2>;
using fp8x2_v  = vec_t<fp8_t, 2>;
using fp16x2_v = vec_t<fp16_t, 2>;
using bf16x2_v = vec_t<bf16_t, 2>;
using fp32x2_v = vec_t<fp32_t, 2>;
struct fp4x2_t
{
    using type = uint8_t;
    type data;
    __host__ __device__ constexpr fp4x2_t() : data{type{}} {}
    __host__ __device__ constexpr fp4x2_t(type init) : data{init} {}
};
using fp4x2x2_v = vec_t<fp4x2_t, 2>;
using fp4x2x4_v = vec_t<fp4x2_t, 4>;
using fp4x2x8_v = vec_t<fp4x2_t, 8>;

template <>
struct vector_traits<fp4x2_t>
{
    using scalar_type                    = uint8_t;
    static constexpr index_t vector_size = 1;
};

template <>
struct numeric<fp4x2_t>
{
    // maximum finite value
    CK_TILE_HOST_DEVICE static constexpr fp32_t max() { return 6.0f; }
};
CK_TILE_DEVICE fp32x2_v amd_assembly_pk_mul_f32(fp32x2_v a, fp32x2_t b)
{
    fp32x2_v c;
    asm volatile("v_pk_mul_f32 %0, %1, %2" : "=v"(c) : "v"(a), "v"(b));
    return c;
}
CK_TILE_DEVICE fp8x2_v amd_assembly_cvt_pk_fp8_f32(fp32_t a, fp32_t b)
{
    int16x2_t c;
    static constexpr bool is_e4m3_fnuz =
        constexpr(numeric_traits<fp8_t>::f8_interpret == fp8_interpretation::E4M3_FNUZ);
    static constexpr float d = is_e4m3_fnuz ? 240.0f : 448.0f;
    static constexpr float e = is_e4m3_fnuz ? -240.0f : -448.0f;
    asm volatile("v_med3_f32 %1, %1, %3, %4\n"
                 "v_med3_f32 %2, %2, %3, %4\n"
                 "v_cvt_pk_fp8_f32 %0, %1, %2"
                 : "=v"(c)
                 : "v"(a), "v"(b), "v"(d), "v"(e));
    return bit_cast<fp8x2_v>(c[0]);
}
CK_TILE_DEVICE fp8x2_v amd_assembly_cvt_pk_bf8_f32(fp32_t a, fp32_t b)
{
    int16x2_t c;
    static constexpr float d = 57344.0f;
    static constexpr float e = -57344.0f;
    asm volatile("v_med3_f32 %1, %1, %3, %4\n"
                 "v_med3_f32 %2, %2, %3, %4\n"
                 "v_cvt_pk_bf8_f32 %0, %1, %2"
                 : "=v"(c)
                 : "v"(a), "v"(b), "v"(d), "v"(e));
    return bit_cast<fp8x2_v>(c[0]);
}
CK_TILE_DEVICE fp4x2_t amd_assembly_cvt_scalef32_pk_fp4_f32(fp32_t a, fp32_t b, fp32_t scale)
{
#if defined(__gfx950__)
    int16x2_t c;
    // permute high bits and low bits to match the order of the original vector
    asm volatile("v_cvt_scalef32_pk_fp4_f32 %0, %1, %2, %3" : "=v"(c) : "v"(b), "v"(a), "v"(scale));
    return bit_cast<fp4x2_t>(bit_cast<int8x2_t>(c[0])[0]);
#endif
}
CK_TILE_DEVICE fp4x2_t amd_assembly_cvt_scalef32_pk_fp4_f16(fp16x2_v a, fp32_t scale)
{
#if defined(__gfx950__)
    int16x2_t c;
    // permute high bits and low bits to match the order of the original vector
    asm volatile("v_cvt_scalef32_pk_fp4_f16 %0, %1, %2" : "=v"(c) : "v"(a), "v"(scale));
    return bit_cast<fp4x2_t>(bit_cast<int8x2_t>(c[0])[0]);
#endif
}
CK_TILE_DEVICE fp4x2_t amd_assembly_cvt_scalef32_pk_fp4_bf16(bf16x2_v a, fp32_t scale)
{
#if defined(__gfx950__)
    int16x2_t c;
    // permute high bits and low bits to match the order of the original vector
    asm volatile("v_cvt_scalef32_pk_fp4_bf16 %0, %1, %2" : "=v"(c) : "v"(a), "v"(scale));
    return bit_cast<fp4x2_t>(bit_cast<int8x2_t>(c[0])[0]);
#endif
}

// convert any to fp32x?_t one by one
template <typename Y,
          typename X,
          index_t N,
          std::enable_if_t<(std::is_same_v<Y, fp32_t>), bool> = false>
CK_TILE_HOST_DEVICE constexpr vec_t<Y, N> vec_convert(vec_t<X, N> x)
{
    using fp32xX_t = vec_t<Y, N>;
    fp32xX_t tmp;
    for(size_t i = 0; i < N; i++)
    {
        tmp[i] = type_convert<Y>(x[i]);
    }
    return tmp;
}

template <typename Y,
          typename X,
          index_t N,
          std::enable_if_t<(N % 2 == 0), bool>                    = false,
          std::enable_if_t<(!(std::is_same_v<Y, fp4x2_t>)), bool> = false>
CK_TILE_HOST_DEVICE constexpr vec_t<Y, N> vec_convert(vec_t<X, N> x, fp32_t inverted_scale)
{
    if constexpr(!std::is_same_v<X, fp32_t>)
    {
        using fp32xX_t = vec_t<fp32_t, N>;
        fp32xX_t tmp   = vec_convert<fp32_t, X, N>(x);
        return vec_convert<Y, fp32_t, N>(tmp, inverted_scale);
    }
    else
    {
        // fp32->??
        return vec_convert<Y, fp32_t, N>(x, inverted_scale);
    }
}

// fp32x2 -> fp8x2
CK_TILE_HOST_DEVICE constexpr fp8x2_v fp32x2_t_to_fp8x2_t(fp32x2_v x, fp32_t inverted_scale)
{
    using vec_ti             = vector_traits<fp32x2_v>;
    constexpr int vec_size   = vec_ti::vector_size;
    constexpr auto interpret = numeric_traits<fp8_t>::f8_interpret;
    fp32x2_v tmp             = amd_assembly_pk_mul_f32(x, fp32x2_t{inverted_scale, inverted_scale});

    return (interpret == fp8_interpretation::E4M3_FNUZ) ||
                   (interpret == fp8_interpretation::E4M3_OCP)
               ? amd_assembly_cvt_pk_fp8_f32(tmp[0], tmp[1])
               : amd_assembly_cvt_pk_bf8_f32(tmp[0], tmp[1]);
}
// fp32x2 -> int8x2
CK_TILE_HOST_DEVICE constexpr int8x2_v fp32x2_t_to_int8x2_t(fp32x2_v x, fp32_t inverted_scale)
{
    fp32x2_v tmp = amd_assembly_pk_mul_f32(x, fp32x2_t{inverted_scale, inverted_scale});

    int8x2_v out;
    out[0] = static_cast<int8_t>(tmp[0]);
    out[1] = static_cast<int8_t>(tmp[1]);
    return out;
}
// fp32x2 -> fp4x2
CK_TILE_HOST_DEVICE constexpr fp4x2_t fp32x2_t_to_fp4x2_t(fp32x2_v x, fp32_t inverted_scale)
{
    return amd_assembly_cvt_scalef32_pk_fp4_f32(x[0], x[1], inverted_scale);
}
// fp16x2 -> fp4x2
CK_TILE_HOST_DEVICE constexpr fp4x2_t fp16x2_t_to_fp4x2_t(fp16x2_v x, fp32_t inverted_scale)
{
    return amd_assembly_cvt_scalef32_pk_fp4_f16(x, inverted_scale);
}
// bf16x2 -> fp4x2
CK_TILE_HOST_DEVICE constexpr fp4x2_t bf16x2_t_to_fp4x2_t(bf16x2_v x, fp32_t inverted_scale)
{
    return amd_assembly_cvt_scalef32_pk_fp4_bf16(x, inverted_scale);
}
#define CK_TILE_TYPE_CONVERT(dtype_, stype_, vec_size_)                                     \
    template <>                                                                             \
    CK_TILE_HOST_DEVICE constexpr vec_t<dtype_##_t, vec_size_>                              \
    vec_convert<dtype_##_t, stype_##_t, vec_size_>(vec_t<stype_##_t, vec_size_> x,          \
                                                   fp32_t inverted_scale)                   \
    {                                                                                       \
        constexpr int iter_num = vec_size_ / 2;                                             \
        vec_t<dtype_##_t, vec_size_> out;                                                   \
        using vec_i2 = vec_t<stype_##_t, 2>;                                                \
        using vec_o2 = vec_t<dtype_##_t, 2>;                                                \
        _Pragma("unroll") for(size_t i = 0; i < iter_num; i++)                              \
        {                                                                                   \
            vec_o2 tmp = stype_##x2##_t_to_##dtype_##x2##_t(x.template get_as<vec_i2>()(i), \
                                                            inverted_scale);                \
            out.template get_as<vec_o2>()(i) = tmp;                                         \
        }                                                                                   \
        return out;                                                                         \
    }
CK_TILE_TYPE_CONVERT(fp8, fp32, 2)
CK_TILE_TYPE_CONVERT(fp8, fp32, 4)
CK_TILE_TYPE_CONVERT(fp8, fp32, 8)
CK_TILE_TYPE_CONVERT(fp8, fp32, 16)
CK_TILE_TYPE_CONVERT(fp8, fp32, 32)

CK_TILE_TYPE_CONVERT(int8, fp32, 2)
CK_TILE_TYPE_CONVERT(int8, fp32, 4)
CK_TILE_TYPE_CONVERT(int8, fp32, 8)
CK_TILE_TYPE_CONVERT(int8, fp32, 16)
CK_TILE_TYPE_CONVERT(int8, fp32, 32)
#undef CK_TILE_TYPE_CONVERT

// 4 bit vec convert
// convert any to fp32x?_t one by one
template <typename Y,
          typename X,
          index_t N,
          std::enable_if_t<(N % 2 == 0), bool>                   = false,
          std::enable_if_t<((std::is_same_v<Y, fp4x2_t>)), bool> = false>
CK_TILE_HOST_DEVICE constexpr vec_t<Y, N / 2> vec_convert(vec_t<X, N> x, fp32_t inverted_scale);

#define CK_TILE_TYPE_CONVERT(dtype_, stype_, vec_size_)                                         \
    template <>                                                                                 \
    CK_TILE_HOST_DEVICE constexpr vec_t<dtype_##_t, vec_size_ / 2>                              \
    vec_convert<dtype_##_t, stype_##_t, vec_size_>(vec_t<stype_##_t, vec_size_> x,              \
                                                   fp32_t inverted_scale)                       \
    {                                                                                           \
        constexpr int iter_num = vec_size_ / 2;                                                 \
        vec_t<dtype_##_t, iter_num> out;                                                        \
        using vec_i2 = vec_t<stype_##_t, 2>;                                                    \
        using vec_o2 = dtype_##_t;                                                              \
        _Pragma("unroll") for(size_t i = 0; i < iter_num; i++)                                  \
        {                                                                                       \
            vec_o2 tmp =                                                                        \
                stype_##x2##_t_to_##dtype_##_t(x.template get_as<vec_i2>()(i), inverted_scale); \
            out.template get_as<vec_o2>()(i) = tmp;                                             \
        }                                                                                       \
        return out;                                                                             \
    }
CK_TILE_TYPE_CONVERT(fp4x2, fp32, 4)
CK_TILE_TYPE_CONVERT(fp4x2, fp32, 8)
CK_TILE_TYPE_CONVERT(fp4x2, fp32, 16)
CK_TILE_TYPE_CONVERT(fp4x2, fp32, 32)

CK_TILE_TYPE_CONVERT(fp4x2, fp16, 4)
CK_TILE_TYPE_CONVERT(fp4x2, fp16, 8)
CK_TILE_TYPE_CONVERT(fp4x2, fp16, 16)
CK_TILE_TYPE_CONVERT(fp4x2, fp16, 32)

CK_TILE_TYPE_CONVERT(fp4x2, bf16, 4)
CK_TILE_TYPE_CONVERT(fp4x2, bf16, 8)
CK_TILE_TYPE_CONVERT(fp4x2, bf16, 16)
CK_TILE_TYPE_CONVERT(fp4x2, bf16, 32)
#undef CK_TILE_TYPE_CONVERT

} // namespace ck_tile
