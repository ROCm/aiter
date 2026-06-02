// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026 Page_Attetion_GQA_fp8 project
//
// Common types, helpers and MFMA intrinsics for the FP8 paged-attention
// decode kernel.  Targets gfx942 (CDNA3) / MI308.  FP8 dtype is
// `torch.float8_e4m3fnuz` (matches aiter.dtypes.fp8 on gfx942).
//
// Adapted from /opt/PagAttetion_GQA/csrc/pa_gqa_common.h (bf16 v5 kernel)
// and /opt/aiter/csrc/cpp_itfs/pa/pa_common.cuh (aiter HIP fp8 path).

#pragma once

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_fp8.h>
#include <cassert>
#include <cfloat>
#include <cstdint>

#define PAGQA_DIVUP(a, b) (((a) + (b) - 1) / (b))

#ifndef WARP_SIZE
#define WARP_SIZE 64
#endif

// ---------------------------------------------------------------------------
// short vector types
// ---------------------------------------------------------------------------
using floatx2 = __attribute__((__vector_size__(2 * sizeof(float)))) float;
using floatx4 = __attribute__((__vector_size__(4 * sizeof(float)))) float;

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

// fp8 vector packing: 8 fp8 values = 64 bits = one `long`/`uint64_t`.
using _B8x8 = uint2;          // 8 bytes; aliased with uint64_t for MFMA
using _B8x4 = uint32_t;       // 4 bytes
using bit8_t = uint8_t;

typedef struct _B8x16
{
    _B8x8 xy[2];
} _B8x16;

// 8 fp8 as a long for MFMA operand.
typedef union u_fp8x8
{
    __hip_fp8_e4m3_fnuz f8x8[8];   // gfx942 native fp8 type
    bit8_t              b8x8[8];
    _B8x8               u2;
    int64_t             i64;
    uint64_t            u64;
    _B16x4              b16x4;     // when packed pk_fp8
    _B8x4               b8x4[2];
} _T8x8;

// ---------------------------------------------------------------------------
// non-temporal loads
// ---------------------------------------------------------------------------
template <typename T>
__device__ __forceinline__ T loadnt(T* addr)
{
    return __builtin_nontemporal_load(addr);
}

__device__ __forceinline__ _B8x16 load_ntmprl_16Byte_fp8(const _B8x16* addr)
{
    auto addr_alias = reinterpret_cast<const float*>(addr);
    auto dat0       = loadnt(addr_alias);
    auto dat1       = loadnt(addr_alias + 1);
    auto dat2       = loadnt(addr_alias + 2);
    auto dat3       = loadnt(addr_alias + 3);
    auto res        = make_float4(dat0, dat1, dat2, dat3);
    return *reinterpret_cast<_B8x16*>(&res);
}

__device__ __forceinline__ _B8x8 load_ntmprl_8Byte_fp8(const _B8x8* addr)
{
    auto addr_alias = reinterpret_cast<const float*>(addr);
    auto dat0       = loadnt(addr_alias);
    auto dat1       = loadnt(addr_alias + 1);
    auto res        = make_float2(dat0, dat1);
    return *reinterpret_cast<_B8x8*>(&res);
}

// ---------------------------------------------------------------------------
// Cross-lane shuffle helpers (gfx940 / CDNA3 specific) — copy from v5
// ---------------------------------------------------------------------------
//
// ds_swizzle_b32: encoded as `pattern = (K << 10) | 0x1F` for K ∈ {1,2,4,8,16}
// — performs XOR by K within each 32-lane group.  Single-instruction.
template <int K>
__device__ __forceinline__ float pa_shfl_xor_within_32(float v)
{
    static_assert(K == 1 || K == 2 || K == 4 || K == 8 || K == 16,
                  "ds_swizzle xor mask must be a single bit in low 5 bits");
    int   src = __builtin_bit_cast(int, v);
    int   dst = __builtin_amdgcn_ds_swizzle(src, (K << 10) | 0x1F);
    return __builtin_bit_cast(float, dst);
}

__device__ __forceinline__ float pa_shfl_xor_16(float v)
{
    return pa_shfl_xor_within_32<16>(v);
}

__device__ __forceinline__ float pa_shfl_xor_32(float v)
{
    return __shfl_xor(v, 32);
}

// ds_bpermute: per-lane broadcast — lane `dst` receives the value held by
// lane `src` (where `src` is dst's data, i.e. byte-offset = src_lane * 4).
// Single-instruction, latency ~12 cycles, no LDS traffic.
__device__ __forceinline__ float pa_lane_bcast(float val, int src_lane)
{
    const int v_bits = __builtin_bit_cast(int, val);
    const int b_bits = __builtin_amdgcn_ds_bpermute(src_lane << 2, v_bits);
    return __builtin_bit_cast(float, b_bits);
}

// AMDGCN buffer-resource descriptor (V#) builder.  Builds the 4-DWORD
// resource word used by `buffer_load`/`buffer_store` intrinsics.  Bypasses
// L1 ("glc"=1) for streaming workloads where lines are read once.
__device__ __forceinline__ __amdgpu_buffer_rsrc_t pa_make_buffer_rsrc(const void* ptr)
{
    return __builtin_amdgcn_make_buffer_rsrc(
        const_cast<void*>(ptr), 0, 0xffffffff, 0x27000);
}

// Buffer-load 8 bytes (one fp8x8 lane = 1 long).  Hardware-level path:
// emits `buffer_load_dwordx2 vN, voffset, srsrc, 0 offen`.  vs the
// pointer-deref path which emits `global_load_dwordx2` (flat-load).
//
// Buffer-load vs global_load on gfx942: both go to the same VMEM unit
// and reach the same achievable peak.  Buffer-load uses a uniform 4-DWORD
// V# resource (in SGPR) + a 32-bit per-lane voffset (1 VGPR/lane).
// global_load uses a 64-bit per-lane address (2 VGPR/lane).  We prefer
// buffer-load because it saves 1 VGPR/lane and avoids the 64-bit
// add in the address pipeline; the bandwidth difference is < 1% in
// well-pipelined inner loops.
//
// CRITICAL: cache policy dominates bandwidth far more than buffer vs
// global.  Gluon's K/V wins came primarily from `nt` (non-temporal,
// SLC=1) on the V load path — V is single-use streamed and never
// re-read by the same workgroup, so caching it in L2 just evicts
// reusable K lines.  See `pa_buffer_load_b128_nt` below.
__device__ __forceinline__ int64_t pa_buffer_load_b64(
    __amdgpu_buffer_rsrc_t rsrc, unsigned int byte_offset)
{
    // The builtin returns `<2 x i32>` (an HIP vec2 of uint).  Bit-cast to
    // int64_t — same 8-byte payload, fits directly into a long MFMA operand.
    using u32x2 = __attribute__((__vector_size__(2 * sizeof(unsigned int)))) unsigned int;
    const u32x2 v = __builtin_amdgcn_raw_buffer_load_b64(rsrc, byte_offset, 0, 0);
    return __builtin_bit_cast(int64_t, v);
}

// Buffer-load 16 bytes (= 4 dwords = 2 longs) per lane.  Hardware-level
// path: emits `buffer_load_dwordx4 v[N:N+3], voffset, srsrc, 0 offen`.
//
// Used to match gluon's per-lane load width on long-ctx (gluon uses 16B
// per lane = 1024B/wave per instruction, vs our 512B/wave for 8B-per-lane
// loads).  Caller picks the appropriate half (low8 / high8) for MFMA B
// operand based on lane parity.
struct pa_u32x4 { unsigned int x, y, z, w; };
__device__ __forceinline__ pa_u32x4 pa_buffer_load_b128(
    __amdgpu_buffer_rsrc_t rsrc, unsigned int byte_offset)
{
    using u32x4 = __attribute__((__vector_size__(4 * sizeof(unsigned int)))) unsigned int;
    const u32x4 v = __builtin_amdgcn_raw_buffer_load_b128(rsrc, byte_offset, 0, 0);
    pa_u32x4 r;
    r.x = v[0]; r.y = v[1]; r.z = v[2]; r.w = v[3];
    return r;
}

// Non-temporal (L2-bypass) 16-byte buffer load.  Equivalent to gluon's
// `global_load_dwordx4 ... off nt` — sets the SLC bit (bit-1 of
// cachepolicy), which on gfx942/CDNA3 disables L2 allocation for this
// line.  Use for streaming-only data (V cache in paged-attention decode):
// each V byte is read exactly once per workgroup, so caching it in L2
// just evicts more-valuable lines (K data, BT, exp_sums) without ever
// giving a hit back.
//
// Empirical observation on MI308X gfx942: K is dot-product-reused 32×
// per partition (favours L2 caching), V is one-shot streamed (favours
// L2 bypass).  Mirroring gluon's asymmetric policy frees ~20% of L2
// fill bandwidth for K and reduce-kernel exp_sums in long-ctx workloads.
__device__ __forceinline__ pa_u32x4 pa_buffer_load_b128_nt(
    __amdgpu_buffer_rsrc_t rsrc, unsigned int byte_offset)
{
    using u32x4 = __attribute__((__vector_size__(4 * sizeof(unsigned int)))) unsigned int;
    // cachepolicy = 2 → SLC=1 (L2-bypass), GLC=0 (keep L1 coherent path).
    const u32x4 v = __builtin_amdgcn_raw_buffer_load_b128(rsrc, byte_offset, 0, 2);
    pa_u32x4 r;
    r.x = v[0]; r.y = v[1]; r.z = v[2]; r.w = v[3];
    return r;
}

// Helpers to split a 16B lane load into two longs.
__device__ __forceinline__ int64_t pa_u32x4_low_long(const pa_u32x4& v)
{
    unsigned int a[2] = { v.x, v.y };
    return __builtin_bit_cast(int64_t,
        *reinterpret_cast<const __attribute__((__vector_size__(8))) unsigned int*>(a));
}
__device__ __forceinline__ int64_t pa_u32x4_high_long(const pa_u32x4& v)
{
    unsigned int a[2] = { v.z, v.w };
    return __builtin_bit_cast(int64_t,
        *reinterpret_cast<const __attribute__((__vector_size__(8))) unsigned int*>(a));
}

// Buffer-load a bf16/fp16 element (returns 16-bit raw bits → float).
template <typename T>
__device__ __forceinline__ float pa_buffer_load_bf16(__amdgpu_buffer_rsrc_t rsrc,
                                                       int byte_offset)
{
    const short b =
        __builtin_amdgcn_raw_buffer_load_b16(rsrc, byte_offset, 0, 0);
    if constexpr (std::is_same<T, __hip_bfloat16>::value)
    {
        return __builtin_bit_cast(float, ((int32_t)((uint16_t)b)) << 16);
    }
    else if constexpr (std::is_same<T, _Float16>::value)
    {
        return static_cast<float>(__builtin_bit_cast(_Float16, b));
    }
    else
    {
        static_assert(sizeof(T) == 0, "pa_buffer_load_bf16: unsupported");
    }
}

// ---------------------------------------------------------------------------
// FP8 MFMA wrappers (gfx942 / CDNA3): 16x16x32 fp8_fp8
// Each lane provides 8 fp8 (= 64 bits = 1 long) for A and B.
// C is floatx4.  One call does M=16, N=16, K=32 contraction.
// ---------------------------------------------------------------------------
template <int absz = 0, int cbid = 0, int blgp = 0>
__device__ __forceinline__ floatx4 pa_mfma16x16x32_fp8_fp8(
    const int64_t& A, const int64_t& B, const floatx4& C)
{
    return __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(A, B, C, absz, cbid, blgp);
}

// BF16 MFMA fallback for the (rarely used) output conversion path.
template <typename T>
__device__ __forceinline__ floatx4 pa_mfma16x16x16(const _B16x4& A,
                                                   const _B16x4& B,
                                                   const floatx4& C)
{
    if constexpr (std::is_same<T, __hip_bfloat16>::value)
    {
        return __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(A, B, C, 0, 0, 0);
    }
    else if constexpr (std::is_same<T, _Float16>::value)
    {
        return __builtin_amdgcn_mfma_f32_16x16x16f16(A, B, C, 0, 0, 0);
    }
    else
    {
        static_assert(sizeof(T) == 0, "pa_mfma16x16x16: unsupported dtype");
    }
}

// ---------------------------------------------------------------------------
// FP8 conversions (e4m3 FNUZ on gfx942 / FN on gfx950)
// ---------------------------------------------------------------------------
__device__ __forceinline__ float pa_fp8_to_float(const __hip_fp8_e4m3_fnuz& x)
{
    return float(x);
}

__device__ __forceinline__ __hip_fp8_e4m3_fnuz pa_float_to_fp8(float x)
{
    return __hip_fp8_e4m3_fnuz(x);
}

__device__ __forceinline__ float pa_clamp_fp8_e4m3_fnuz(float x)
{
    constexpr float kFp8E4m3FnuzMax = 240.0f;
    return fminf(fmaxf(x, -kFp8E4m3FnuzMax), kFp8E4m3FnuzMax);
}

// pk_fp8_f32: pack 2 fp32 to 2 fp8 using gfx942 hardware intrinsic.
// `__builtin_amdgcn_cvt_pk_fp8_f32` writes 2 fp8 (16 bits) into the
// upper or lower 16 bits of a 32-bit register based on `byte_sel`.
//   cvt_pk_fp8_f32(srcA, srcB, oldVal, byte_sel=false) → lower 16b new
//   cvt_pk_fp8_f32(srcA, srcB, oldVal, byte_sel=true ) → upper 16b new
__device__ __forceinline__ uint32_t pa_pk_fp8x4(float a0, float a1, float a2, float a3)
{
    // Pack 4 fp32 → 4 fp8 e4m3 in one 32-bit register.
    uint32_t r = 0;
    a0 = pa_clamp_fp8_e4m3_fnuz(a0);
    a1 = pa_clamp_fp8_e4m3_fnuz(a1);
    a2 = pa_clamp_fp8_e4m3_fnuz(a2);
    a3 = pa_clamp_fp8_e4m3_fnuz(a3);
    r = __builtin_amdgcn_cvt_pk_fp8_f32(a0, a1, r, false); // lo 16b
    r = __builtin_amdgcn_cvt_pk_fp8_f32(a2, a3, r, true);  // hi 16b
    return r;
}

// Apply Q/K dequant scales to four QK logits held by one lane.
// Scale layout:
//   q_scale is pre-folded by the caller into `qk_base_log2`
//   k_scale: [num_blocks, num_kv_heads, block_size] contiguous fp32
__device__ __forceinline__ void pa_apply_qk_token_scales(
    floatx4& logits,
    const float* __restrict__ k_scale,
    const int* __restrict__ block_table_seq,
    const int token_base,
    const int num_context_blocks,
    const int last_ctx_block,
    const int num_kv_heads,
    const int kv_head_idx,
    const int block_size,
    const float qk_base_log2)
{
    #pragma unroll
    for (int i = 0; i < 4; i++)
    {
        const int token = token_base + i;
        const int logical_block = token / block_size;
        const int block_slot = token - logical_block * block_size;
        const int safe_logical_block =
            (logical_block < num_context_blocks) ? logical_block : last_ctx_block;
        const int physical_block = block_table_seq[safe_logical_block];
        const int64_t scale_idx =
              (static_cast<int64_t>(physical_block) * num_kv_heads + kv_head_idx)
            * block_size + block_slot;
        logits[i] *= qk_base_log2 * k_scale[scale_idx];
    }
}

// Same scale application when the caller already knows the physical cache
// block. This avoids integer div/mod and a repeated block-table lookup in the
// QK hot path; the four logits always sit inside one 16-token cache block.
__device__ __forceinline__ void pa_apply_qk_token_scales_for_block(
    floatx4& logits,
    const float* __restrict__ k_scale,
    const int physical_block,
    const int slot_base,
    const int num_kv_heads,
    const int kv_head_idx,
    const int block_size,
    const float qk_base_log2)
{
    const int64_t scale_base =
        (static_cast<int64_t>(physical_block) * num_kv_heads + kv_head_idx)
        * block_size + slot_base;
    #pragma unroll
    for (int i = 0; i < 4; i++)
        logits[i] *= qk_base_log2 * k_scale[scale_base + i];
}

// ---------------------------------------------------------------------------
// Standard conversions for output dtype (bf16/fp16)
// ---------------------------------------------------------------------------
template <typename T>
__device__ __forceinline__ float pa_to_float(const T& x)
{
    if constexpr (std::is_same<T, __hip_bfloat16>::value)
    {
        return __bfloat162float(x);
    }
    else if constexpr (std::is_same<T, _Float16>::value)
    {
        return static_cast<float>(x);
    }
    else
    {
        static_assert(sizeof(T) == 0, "pa_to_float: unsupported dtype");
    }
}

template <typename T>
__device__ __forceinline__ T pa_from_float(const float& x)
{
    if constexpr (std::is_same<T, __hip_bfloat16>::value)
    {
        return __float2bfloat16(x);
    }
    else if constexpr (std::is_same<T, _Float16>::value)
    {
        return static_cast<_Float16>(x);
    }
    else
    {
        static_assert(sizeof(T) == 0, "pa_from_float: unsupported dtype");
    }
}

// Vectorised float4 → _B16x4 (bf16 RNE / fp16 native).
template <typename T>
__device__ __forceinline__ _B16x4 pa_from_floatx4(const floatx4& inp)
{
    _B16x4 ret;
    if constexpr (std::is_same<T, __hip_bfloat16>::value)
    {
        for (int i = 0; i < 4; i++)
        {
            union fcvt { uint32_t u32; float f32; } u;
            u.f32 = inp[i];
            u.u32 += 0x7fff + ((u.u32 >> 16) & 1); // BF16 RNE (no nan/inf check)
            ret[i] = uint16_t(u.u32 >> 16);
        }
        return ret;
    }
    else if constexpr (std::is_same<T, _Float16>::value)
    {
        union h2cvt
        {
            __half2 h2[2];
            _B16x4  b16x4;
        } u;
        u.h2[0] = __float22half2_rn(make_float2(inp[0], inp[1]));
        u.h2[1] = __float22half2_rn(make_float2(inp[2], inp[3]));
        return u.b16x4;
    }
    else
    {
        static_assert(sizeof(T) == 0, "pa_from_floatx4: unsupported dtype");
    }
}

// ---------------------------------------------------------------------------
// packed bf16 -> fp32 (no v_cvt_pk_f32_bf16 on gfx942 — fused into lshl)
// ---------------------------------------------------------------------------
__device__ __forceinline__ floatx2 pa_bf16x2_to_floatx2(const _B16x2 x)
{
    floatx2 r;
    r[0] = __builtin_bit_cast(float, ((int32_t)x[0]) << 16);
    r[1] = __builtin_bit_cast(float, ((int32_t)x[1]) << 16);
    return r;
}

template <typename T>
__device__ __forceinline__ floatx4 pa_to_floatx4(const _B16x4& x)
{
    floatx4 r;
    if constexpr (std::is_same<T, __hip_bfloat16>::value)
    {
        #pragma unroll
        for (int i = 0; i < 4; i++)
            r[i] = __builtin_bit_cast(float, ((int32_t)x[i]) << 16);
        return r;
    }
    else if constexpr (std::is_same<T, _Float16>::value)
    {
        const __half2* h2 = reinterpret_cast<const __half2*>(&x);
        const float2 f0 = __half22float2(h2[0]);
        const float2 f1 = __half22float2(h2[1]);
        r[0] = f0.x; r[1] = f0.y; r[2] = f1.x; r[3] = f1.y;
        return r;
    }
    else
    {
        static_assert(sizeof(T) == 0, "pa_to_floatx4: unsupported dtype");
    }
}
