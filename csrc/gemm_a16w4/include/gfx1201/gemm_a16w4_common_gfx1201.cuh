#pragma once
// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Shared scalar plumbing for the gfx1201 (Navi 48) a16w4 GEMM kernels.
//
// ── ARCHITECTURE GATE ──────────────────────────────────────────────────────
// Every kernel under gfx1201/ calls __builtin_amdgcn_wmma_*_w32_gfx12, which
// does not exist before RDNA4, and every tile, LDS stride and AGROUP value was
// measured on gfx1201 specifically (see the header comment on each kernel).
//
// aiter builds ONE module for whatever GPU_ARCHS lists, so a hard #error here
// would break any multi-arch build that merely happens to include this module.
// Instead the device bodies compile to a trap on other targets and the host
// entry point refuses to launch -- see aiter::gemm_a16w4() in gemm_a16w4.cu,
// which checks gcnArchName before every dispatch. The two guards are
// deliberately redundant: the host one produces a readable error, the device
// one guarantees that a hole in the host gate cannot silently execute
// whatever the compiler decided a missing builtin should mean.
//
// This mirrors csrc/kernels/chunk_gated_delta_rule_fwd_h.cu, which gates the
// same way on __gfx1200__/__gfx1201__.

#include "gemm_a16w4_launch.h"

#include <hip/hip_runtime.h>

#if defined(__HIP_DEVICE_COMPILE__) && !defined(__gfx1201__)
#define AITER_A16W4_DEVICE_SUPPORTED 0
#else
#define AITER_A16W4_DEVICE_SUPPORTED 1
#endif

// Placed at the top of every a16w4 __global__ body. On a non-gfx1201 device
// target it replaces the body outright, so the WMMA builtins are never parsed
// for an architecture that lacks them.
#if AITER_A16W4_DEVICE_SUPPORTED
#define AITER_A16W4_REQUIRE_GFX1201()
#else
#define AITER_A16W4_REQUIRE_GFX1201() \
    do                                \
    {                                 \
        __builtin_trap();             \
        return;                       \
    } while(0)
#endif

namespace aiter {
namespace a16w4 {

// kGroupSize / kPackK live in gemm_a16w4_launch.h so the host dispatcher can
// see them without pulling in device code.

// +1024 magic bias for the fp16 dequant path. NOT arbitrary: fp16 has a
// 10-bit mantissa, so 0x6400 | n == 1024 + n exactly for any 4-bit n, and
// 1024 + n and 1024 + z both lie in the binade [1024, 2048) where fp16
// subtraction is exact by Sterbenz's lemma. See dequant8_magic() in the fp16
// kernels for why the subtraction must not be folded into the scale.
constexpr float kMagicBias = 1024.0f;
constexpr unsigned kF16Magic = 0x64006400u; // two copies of fp16(1024.0)
constexpr unsigned kNibMask = 0x000F000Fu;  // one nibble in each 16-bit half

using bf16_raw_t = unsigned short;
using fp16_raw_t = unsigned short;

typedef unsigned short u16x4_t __attribute__((ext_vector_type(4)));
typedef unsigned short u16x8_t __attribute__((ext_vector_type(8)));
typedef unsigned int u32x4_t __attribute__((ext_vector_type(4)));
typedef __bf16 frag_bf16x8 __attribute__((ext_vector_type(8)));
typedef _Float16 f16x2 __attribute__((ext_vector_type(2)));
typedef _Float16 f16x8 __attribute__((ext_vector_type(8)));
typedef float frag_f32x8 __attribute__((ext_vector_type(8)));

__device__ __host__ __forceinline__ float bf16_to_f32(bf16_raw_t x)
{
    union
    {
        unsigned u;
        float f;
    } v;
    v.u = (unsigned)x << 16;
    return v.f;
}

__device__ __host__ __forceinline__ bf16_raw_t f32_to_bf16(float x)
{
    union
    {
        float f;
        unsigned u;
    } v;
    v.f = x;
    return (bf16_raw_t)(v.u >> 16);
}

__device__ __host__ __forceinline__ float fp16_to_f32(fp16_raw_t x)
{
    _Float16 h;
    __builtin_memcpy(&h, &x, 2);
    return (float)h;
}

__device__ __host__ __forceinline__ fp16_raw_t f32_to_fp16(float x)
{
    _Float16 h = (_Float16)x;
    fp16_raw_t r;
    __builtin_memcpy(&r, &h, 2);
    return r;
}

__device__ __forceinline__ frag_bf16x8 as_bf16x8(u16x8_t v)
{
    union
    {
        u16x8_t u;
        frag_bf16x8 f;
    } c;
    c.u = v;
    return c.f;
}

__device__ __forceinline__ f16x8 as_f16x8(u16x8_t v)
{
    union
    {
        u16x8_t u;
        f16x8 f;
    } c;
    c.u = v;
    return c.f;
}

__device__ __forceinline__ f16x8 as_f16x8_u32x4(u32x4_t v)
{
    union
    {
        u32x4_t u;
        f16x8 f;
    } c;
    c.u = v;
    return c.f;
}

__device__ __forceinline__ f16x2 as_f16x2(unsigned v)
{
    union
    {
        unsigned u;
        f16x2 f;
    } c;
    c.u = v;
    return c.f;
}

__device__ __forceinline__ unsigned as_u32(f16x2 v)
{
    union
    {
        f16x2 f;
        unsigned u;
    } c;
    c.f = v;
    return c.u;
}

} // namespace a16w4
} // namespace aiter
