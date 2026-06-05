// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include "aiter_hip_common.h"
#include <hip/hip_bf16.h>
#include <hip/hip_fp8.h>

#include "dtype_fp8.cuh"
#include "float.h"
#include "quant_utils.cuh"

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
#define DIVIDE_ROUND_UP(a, b) (((a) + (b) - 1) / (b))

using floatx4   = __attribute__((__vector_size__(4 * sizeof(float)))) float;
using float16x4 = __attribute__((__vector_size__(4 * sizeof(_Float16)))) _Float16;
typedef float16x4 _Half4;
using float16x2 = __attribute__((__vector_size__(2 * sizeof(_Float16)))) _Float16;
typedef float16x2 _Half2;
typedef struct _Half8
{
    _Half4 xy[2];
} _Half8;

using bit16x2 = __attribute__((__vector_size__(2 * sizeof(uint16_t)))) uint16_t;
typedef bit16x2 _B16x2;

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

union vec_converter
{
    bit16x4 vec4;
    bit16x2 vec2[2];
};

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

// template <typename T, int absz, int cbid, int blgp>
// __device__ __forceinline__ floatx4 gcn_mfma16x16x128_instr(const long& inpA,
//                                                            const long& inpB,
//                                                            const floatx4& inpC) {
//     if constexpr (std::is_same<T, __hip_fp8_e4m3>::value) {
//         return __builtin_amdgcn_smfmac_f32_16x16x128_fp8_fp8(inpA, inpB, inpC, absz, cbid, blgp);
//     } else if constexpr (std::is_same<T, __hip_fp8_e5m2>::value) {
//         return __builtin_amdgcn_smfmac_f32_16x16x128_bf8_bf8(inpA, inpB, inpC, absz, cbid, blgp);
//     } else {
//         static_assert(false, "unsupported 8b dtype");
//     }
// }
template <typename T, int absz, int cbid, int blgp>
__device__ __forceinline__ floatx4 gcn_mfma16x16x32_instr(const long& inpA,
                                                          const long& inpB,
                                                          const floatx4& inpC)
{
    if constexpr(std::is_same<T, __hip_fp8_e4m3>::value)
    {
        return __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(inpA, inpB, inpC, absz, cbid, blgp);
    }
    else if constexpr(std::is_same<T, __hip_fp8_e5m2>::value)
    {
        return __builtin_amdgcn_mfma_f32_16x16x32_bf8_bf8(inpA, inpB, inpC, absz, cbid, blgp);
    }
    else
    {
        static_assert(false, "unsupported 8b dtype");
    }
}

#elif(defined(__gfx1200__) || defined(__gfx1201__))
// ===========================================================================
// RDNA4 (Navi4x: gfx1200 / gfx1201) WMMA path -- isolated from the gfx9 MFMA
// path above. NOTE: narrowed to gfx1200/gfx1201 on purpose; __GFX12__ is also
// defined for gfx1250, which is a different microarchitecture with its own
// *_gfx1250 WMMA intrinsics and must NOT take this path.
// gfx12 has no Matrix-Arithmetic-Instructions (mai-insts), so MFMA builtins do
// not exist here. Instead we use the 3rd-gen WMMA matrix cores (wave32).
// Verified intrinsics on this toolchain (ROCm 7.2.1 / clang 22):
//   float8 __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(short8 A, short8 B, float8 C)
//   float8 __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12 (half8  A, half8  B, float8 C)
//
// LAYOUT NOTE (important, still TODO at the kernel level):
//   MFMA on gfx9 is wave64 and each lane holds a 4-wide fragment (_B16x4 in,
//   floatx4 out). WMMA on gfx12 is wave32 and each lane holds an 8-wide
//   fragment (short8/half8 in, float8 out). To keep the SAME helper signature
//   the kernel uses (so it compiles and we can advance error-by-error), we pack
//   the incoming 4-wide fragment into the LOW half of the 8-wide WMMA fragment
//   and return the LOW floatx4 of the 8-wide result. The cross-lane fragment
//   redistribution required for full numeric correctness must be done where the
//   kernel loads/stores fragments -- that is the next thing to port.
// ===========================================================================
using _Wf32x8  = __attribute__((__vector_size__(8 * sizeof(float)))) float;
using _Wbf16x8 = __attribute__((__vector_size__(8 * sizeof(short)))) short;
using _Wf16x8  = __attribute__((__vector_size__(8 * sizeof(_Float16)))) _Float16;

template <typename T, int absz, int cbid, int blgp>
__device__ __forceinline__ floatx4 gcn_mfma16x16x16_instr(const _B16x4& inpA,
                                                          const _B16x4& inpB,
                                                          const floatx4& inpC)
{
    _Wf32x8 c8 = {inpC[0], inpC[1], inpC[2], inpC[3], 0.f, 0.f, 0.f, 0.f};
    if constexpr(std::is_same<T, _Float16>::value)
    {
        _Wf16x8 a8{}, b8{};
        const _Float16* ap = reinterpret_cast<const _Float16*>(&inpA);
        const _Float16* bp = reinterpret_cast<const _Float16*>(&inpB);
        for(int i = 0; i < 4; ++i)
        {
            a8[i] = ap[i];
            b8[i] = bp[i];
        }
        _Wf32x8 d8 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a8, b8, c8);
        return floatx4{d8[0], d8[1], d8[2], d8[3]};
    }
    else if constexpr(std::is_same<T, __hip_bfloat16>::value)
    {
        _Wbf16x8 a8{}, b8{};
        const short* ap = reinterpret_cast<const short*>(&inpA);
        const short* bp = reinterpret_cast<const short*>(&inpB);
        for(int i = 0; i < 4; ++i)
        {
            a8[i] = ap[i];
            b8[i] = bp[i];
        }
        _Wf32x8 d8 = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a8, b8, c8);
        return floatx4{d8[0], d8[1], d8[2], d8[3]};
    }
    else
    {
        static_assert(false, "unsupported 16b dtype");
    }
}

template <typename T, int absz, int cbid, int blgp>
__device__ __forceinline__ floatx4 gcn_mfma16x16x32_instr(const long& inpA,
                                                          const long& inpB,
                                                          const floatx4& inpC)
{
    // TODO(gfx12): fp8 KV path. gfx12 fp8 WMMA is
    //   __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12 (K=16), whereas the
    //   MFMA helper this replaces is 16x16x32 (K=32) with a different fragment
    //   layout. Not instantiated for the bf16 config under test, so left as an
    //   identity placeholder until the fp8 KV path is ported.
    static_assert(std::is_same<T, __hip_fp8_e4m3>::value ||
                      std::is_same<T, __hip_fp8_e5m2>::value,
                  "unsupported 8b dtype");
    return inpC;
}

// 8-wide WMMA helper for gfx1200/gfx1201. This is the NUMERICALLY CORRECT
// gfx12 path: the 4-wide gcn_mfma16x16x16_instr above zero-pads the high half
// of the 8-wide WMMA fragment, which is mathematically wrong. The kernel must
// switch to 8-wide loads + this helper on gfx12; see pa_gfx1201_wmma_layout.md.
//
// Verified gfx12 builtin contract (op_tests/wmma_min_test.cpp):
//   builtin(B_frag, A_frag, C) computes D[m=lane%16][n=group*8+e]
//                                       = sum_k A[m][k] * B[n][k]
//   with A_frag[e]=A[m=lane%16][k=group*8+e], B_frag[e]=B[n=lane%16][k=group*8+e].
// QK uses K_frag as A (M=token) and Q_frag as B (N=qhead) -> output D[token][qhead].
// To match the gfx9 MFMA call site (which passes K first, Q second, and gets
// lane=qhead/rowid*4+i=token output), we instead pick M=qhead so the kernel can
// keep its lane=qhead post-processing layout: A_matrix=Q (qhead-major) is passed
// as inpB and B_matrix=K (token-major) as inpA, then builtin first-arg=B=inpA.
// Net: callers keep `helper(Klocal, Qlocal, C)` and we forward `(inpA, inpB, inpC)`
// to the builtin unchanged.
template <typename T>
__device__ __forceinline__ _Wf32x8 gcn_wmma16x16x16_instr(const _B16x8& inpA,
                                                          const _B16x8& inpB,
                                                          const _Wf32x8& inpC)
{
    if constexpr(std::is_same<T, _Float16>::value)
    {
        _Wf16x8 a8, b8;
        __builtin_memcpy(&a8, &inpA, sizeof(a8));
        __builtin_memcpy(&b8, &inpB, sizeof(b8));
        return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a8, b8, inpC);
    }
    else if constexpr(std::is_same<T, __hip_bfloat16>::value)
    {
        _Wbf16x8 a8, b8;
        __builtin_memcpy(&a8, &inpA, sizeof(a8));
        __builtin_memcpy(&b8, &inpB, sizeof(b8));
        return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a8, b8, inpC);
    }
    else
    {
        static_assert(sizeof(T) == 0, "unsupported 16b dtype");
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
                                                          const floatx4& inpC)
{
    if constexpr(std::is_same<T, __hip_fp8_e4m3>::value)
    {
        return __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(inpA, inpB, inpC, absz, cbid, blgp);
    }
    else if constexpr(std::is_same<T, __hip_fp8_e5m2>::value)
    {
        return __builtin_amdgcn_mfma_f32_16x16x32_bf8_bf8(inpA, inpB, inpC, absz, cbid, blgp);
    }
    else
    {
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
    const auto f0 = __builtin_amdgcn_cvt_pk_f32_fp8(inp, false);
    const auto f1 = __builtin_amdgcn_cvt_pk_f32_fp8(inp, true);
    floatx4 ret;
    ret[0] = f0[0];
    ret[1] = f0[1];
    ret[2] = f1[0];
    ret[3] = f1[1];
    return ret;
}

__device__ __forceinline__ floatx4 to_float_bf8x4(const _B8x4& inp)
{
    const auto f0 = __builtin_amdgcn_cvt_pk_f32_bf8(inp, false);
    const auto f1 = __builtin_amdgcn_cvt_pk_f32_bf8(inp, true);
    floatx4 ret;
    ret[0] = f0[0];
    ret[1] = f0[1];
    ret[2] = f1[0];
    ret[3] = f1[1];
    return ret;
}

// Lane-local dequant of a packed 8-fp8 fragment (_B8x8 = uint2 = 8 bytes)
// into an 8-elem bf16/fp16 fragment (_B16x8). Used by the gfx12 fp8-KV PA
// path (Option A): we keep HBM at fp8, but feed the existing bf16/fp16 WMMA
// helper after this in-register dequant. KV_DTYPE selects the fp8 variant
// ABI (e4m3 -> cvt_pk_f32_fp8, e5m2 -> cvt_pk_f32_bf8). No scaling here;
// k_scale/v_scale are folded into post-mfma `scale2 *= *k_scale_ptr` and
// `tmp_out *= *v_scale_ptr` at the kernel level, same as gfx9.
template <typename ScalarT, vllm::Fp8KVCacheDataType KV_DTYPE>
__device__ __forceinline__ _B16x8 dequant_fp8x8_to_b16x8(const _B8x8& in)
{
    static_assert(KV_DTYPE == vllm::Fp8KVCacheDataType::kFp8E4M3 ||
                      KV_DTYPE == vllm::Fp8KVCacheDataType::kFp8E5M2,
                  "dequant_fp8x8_to_b16x8: KV_DTYPE must be a fp8 variant");
    _B8x4 lo = static_cast<_B8x4>(in.x);
    _B8x4 hi = static_cast<_B8x4>(in.y);
    floatx4 f_lo, f_hi;
    if constexpr(KV_DTYPE == vllm::Fp8KVCacheDataType::kFp8E4M3)
    {
        f_lo = to_float_fp8x4(lo);
        f_hi = to_float_fp8x4(hi);
    }
    else
    {
        f_lo = to_float_bf8x4(lo);
        f_hi = to_float_bf8x4(hi);
    }
    _B16x8 out;
    out.xy[0] = from_floatx4<ScalarT>(f_lo);
    out.xy[1] = from_floatx4<ScalarT>(f_hi);
    return out;
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

typedef union u64_cvt
{
    half f16x4[4];
    int16_t b16x4[4];
    _B8x8 b8x8;
    _B16x4 b64;
    int64_t i64;
} _T8x8;

__device__ __forceinline__ float warpReduceMax(float val)
{
    for(int offset = warpSize / 2; offset > 0; offset /= 2)
    {
        val = max(val, __shfl_down(val, offset, warpSize)); // Using max() for reduction
    }
    return val;
}
