// SPDX-License-Identifier: MIT
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include <hip/hip_bf16.h>
#include <hip/hip_fp8.h>
#include "hip_compat.h"

#include "dtype_fp8.cuh"
#include "quant_utils.cuh"
#include "float.h"

#include <ck_tile/ops/fmha/block/block_masking.hpp>
#include <ck_tile/ops/fmha/block/variants.hpp>

#if defined(NDEBUG)
#undef NDEBUG
#include <assert.h>
#define UNREACHABLE_CODE assert(false);
#define NDEBUG
#else
#define UNREACHABLE_CODE assert(false);
#endif

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define DIVIDE_ROUND_UP(a, b) (((a) + (b)-1) / (b))


using floatx4   = __attribute__((__vector_size__(4 * sizeof(float)))) float;
using float16x4 = __attribute__((__vector_size__(4 * sizeof(_Float16)))) _Float16;
typedef float16x4 _Half4;
using float16x2 = __attribute__((__vector_size__(2 * sizeof(_Float16)))) _Float16;
typedef float16x2 _Half2;
typedef struct _Half8
{
    _Half4 xy[2];
} _Half8;

using bit16x4 = __attribute__((__vector_size__(4 * sizeof(uint16_t)))) uint16_t;
typedef bit16x4 _B16x4;
typedef struct _B16x8
{
    _B16x4 xy[2];
} _B16x8;

using bit16x8 = __attribute__((__vector_size__(8 * sizeof(uint16_t)))) uint16_t;
typedef bit16x8 _B16x8_2;

using _B8x8  = uint2;
using _B8x4  = int32_t; // used in builtins
using bit8_t = uint8_t;

typedef struct _B8x16
{
    _B8x8 xy[2];
} _B8x16;

////// Non temporal loads ///////
template <typename T>
__device__ __forceinline__ T loadnt(T* addr)
{
    return __builtin_nontemporal_load(addr);
}

__device__ __forceinline__ _B16x8 load_ntmprl_16Byte(const _B16x8* addr)
{
    auto addr_alias = reinterpret_cast<const float*>(addr);
    auto dat0       = loadnt(addr_alias);
    auto dat1       = loadnt(addr_alias + 1);
    auto dat2       = loadnt(addr_alias + 2);
    auto dat3       = loadnt(addr_alias + 3);
    auto res        = make_float4(dat0, dat1, dat2, dat3);
    return *reinterpret_cast<_B16x8*>(&res);
}

#if defined(__gfx950__)
template <typename T, int absz, int cbid, int blgp>
__device__ __forceinline__ floatx4 gcn_mfma16x16x32_instr(const _B16x8& inpA,
                                                          const _B16x8& inpB,
                                                          const floatx4& inpC)
{
    _B16x8_2 tmpA = __builtin_shufflevector(inpA.xy[0], inpA.xy[1], 0, 1, 2, 3, 4, 5, 6, 7);
    _B16x8_2 tmpB = __builtin_shufflevector(inpB.xy[0], inpB.xy[1], 0, 1, 2, 3, 4, 5, 6, 7);

    if constexpr(std::is_same<T, _Float16>::value)
    {
        return __builtin_amdgcn_mfma_f32_16x16x32_f16(tmpA, tmpB, inpC, absz, cbid, blgp);
    }
    else if constexpr(std::is_same<T, __hip_bfloat16>::value)
    {
        return __builtin_amdgcn_mfma_f32_16x16x32_bf16(tmpA, tmpB, inpC, absz, cbid, blgp);
    }
    else
    {
        static_assert(false, "unsupported 16b dtype");
    }
}

template <typename T, int absz, int cbid, int blgp>
__device__ __forceinline__ floatx4 gcn_mfma16x16x128_instr(const long& inpA,
                                                           const long& inpB,
                                                           const floatx4& inpC) {
    if constexpr (std::is_same<T, __hip_fp8_e4m3>::value) {
        return __builtin_amdgcn_smfmac_f32_16x16x128_fp8_fp8(inpA, inpB, inpC, absz, cbid, blgp);
    } else if constexpr (std::is_same<T, __hip_fp8_e5m2>::value) {
        return __builtin_amdgcn_smfmac_f32_16x16x128_bf8_bf8(inpA, inpB, inpC, absz, cbid, blgp);
    } else {
        static_assert(false, "unsupported 8b dtype");
    }
}
#else
template <typename T, int absz, int cbid, int blgp>
__device__ __forceinline__ floatx4 gcn_mfma16x16x16_instr(const _B16x4& inpA,
                                                          const _B16x4& inpB,
                                                          const floatx4& inpC)
{
    if constexpr(std::is_same<T, _Float16>::value)
    {
        return __builtin_amdgcn_mfma_f32_16x16x16f16(inpA, inpB, inpC, absz, cbid, blgp);
    }
    else if constexpr(std::is_same<T, __hip_bfloat16>::value)
    {
        return __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(inpA, inpB, inpC, absz, cbid, blgp);
    }
    else
    {
        static_assert(false, "unsupported 16b dtype");
    }
}

template <typename T, int absz, int cbid, int blgp>
__device__ __forceinline__ floatx4 gcn_mfma16x16x32_instr(const long& inpA,
                                                          const long& inpB,
                                                          const floatx4& inpC) {
  if constexpr (std::is_same<T, __hip_fp8_e4m3>::value) {
    return __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(inpA, inpB, inpC, absz, cbid,
                                                 blgp);
  } else if constexpr (std::is_same<T, __hip_fp8_e5m2>::value) {
    return __builtin_amdgcn_mfma_f32_16x16x32_bf8_bf8(inpA, inpB, inpC, absz,
                                                     cbid, blgp);
  } else {
    static_assert(false, "unsupported 8b dtype");
  }
}
#endif

template <typename T>
__device__ __forceinline__ float to_float(const T& inp)
{
    if constexpr(std::is_same<T, _Float16>::value)
    {
        return (float)inp;
    }
    else if constexpr(std::is_same<T, __hip_bfloat16>::value)
    {
        return __bfloat162float(inp);
    }
    else
    {
        static_assert(false, "unsupported 16b dtype");
    }
}


template <typename T>
__device__ __forceinline__ T from_float(const float& inp)
{
    if constexpr(std::is_same<T, _Float16>::value)
    {
        return (_Float16)inp;
    }
    else if constexpr(std::is_same<T, __hip_bfloat16>::value)
    {
        return __float2bfloat16(inp);
    }
    else
    {
        static_assert(false, "unsupported 16b dtype");
    }
}

template <typename T>
__device__ __forceinline__ _B16x4 from_floatx4(const floatx4& inp)
{
    _B16x4 ret;
    if constexpr(std::is_same<T, _Float16>::value)
    {
        union h2cvt
        {
            __half2 h2[2];
            _B16x4 b16x4;
        } u;
        u.h2[0] = __float22half2_rn(make_float2(inp[0], inp[1]));
        u.h2[1] = __float22half2_rn(make_float2(inp[2], inp[3]));
        return u.b16x4;
    }
    else if constexpr(std::is_same<T, __hip_bfloat16>::value)
    {
        for(int i = 0; i < 4; i++)
        {
            union fcvt
            {
                uint32_t u32;
                float f32;
            } u;
            u.f32 = inp[i];
            u.u32 += 0x7fff + ((u.u32 >> 16) & 1); // BF16 RNE with no nan/inf check
            ret[i] = uint16_t(u.u32 >> 16);
        }
        return ret;
    }
    else
    {
        static_assert(false, "unsupported 16b dtype");
    }
}

template <typename T>
__device__ __forceinline__ _B16x4 addx4(const _B16x4& inp1, const _B16x4& inp2)
{
    _B16x4 ret;
    if constexpr(std::is_same<T, _Float16>::value)
    {
        union h2cvt
        {
            _B16x4 b16x4;
            __half2 h2[2];
        } u1, u2, s;
        u1.b16x4 = inp1;
        u2.b16x4 = inp2;
        s.h2[0]  = u1.h2[0] + u2.h2[0];
        s.h2[1]  = u1.h2[1] + u2.h2[1];
        return s.b16x4;
    }
    else if constexpr(std::is_same<T, __hip_bfloat16>::value)
    {
        for(int i = 0; i < 4; i++)
        {
            union fcvt
            {
                float f32;
                uint32_t i32;
            } u1, u2, s;
            u1.i32 = uint32_t(inp1[i]) << 16;
            u2.i32 = uint32_t(inp2[i]) << 16;
            s.f32  = u1.f32 + u2.f32;
            ret[i] = uint16_t(s.i32 >> 16);
        }
        return ret;
    }
    else
    {
        static_assert(false, "unsupported 16b dtype");
    }
}


__device__ __forceinline__ floatx4 to_float_fp8x4(const _B8x4& inp)
{
#if defined(__gfx90a__)
    float4 f32x4 =
        vllm::fp8::vec_conversion<float4, uint32_t>(*reinterpret_cast<const uint32_t*>(&inp));
    return *reinterpret_cast<floatx4*>(&f32x4);
#else // MI3xx+ optimized builtins
    const auto f0 = __builtin_amdgcn_cvt_pk_f32_fp8(inp, false);
    const auto f1 = __builtin_amdgcn_cvt_pk_f32_fp8(inp, true);
    floatx4 ret;
    ret[0] = f0[0];
    ret[1] = f0[1];
    ret[2] = f1[0];
    ret[3] = f1[1];
    return ret;
#endif
}

template <typename T>
__device__ __forceinline__ _B16x4 from_floatx4_rtz(const floatx4& inp)
{
    _B16x4 ret;
    if constexpr(std::is_same<T, _Float16>::value)
    {
        union h2cvt
        {
            _Half2 h2[2];
            _B16x4 b16x4;
        } u;
        u.h2[0] = __builtin_amdgcn_cvt_pkrtz(inp[0], inp[1]);
        u.h2[1] = __builtin_amdgcn_cvt_pkrtz(inp[2], inp[3]);
        return u.b16x4;
    }
    else if constexpr(std::is_same<T, __hip_bfloat16>::value)
    {
        for(int i = 0; i < 4; i++)
        {
            union fcvt
            {
                uint32_t i32;
                float f32;
            } u;
            u.f32  = inp[i];
            ret[i] = uint16_t(u.i32 >> 16);
        }
        return ret;
    }
    else
    {
        static_assert(false, "unsupported 16b dtype");
    }
}

template <typename T>
__device__ __forceinline__ _B16x8 convert_b8x8_custom(const _B8x8 input)
{
    union
    {
        _B8x8 b8x8;
        _B8x4 b8x4[2];
    } tmp;
    tmp.b8x8 = input;
    _B16x8 ret;
    for(int i = 0; i < 2; i++)
    {
        ret.xy[i] = from_floatx4_rtz<T>(to_float_fp8x4(tmp.b8x4[i]));
    }
    return ret;
}

typedef union u64_cvt {
  half f16x4[4];
  int16_t b16x4[4];
  _B8x8 b8x8;
  _B16x4 b64;
  int64_t i64;
} _T8x8;


__device__ float warpReduceMax(float val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val = max(val, __shfl_down(val, offset, warpSize)); // Using max() for reduction
    }
    return val;
}