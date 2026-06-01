// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <hip/hip_runtime.h>

#include "opus/opus.hpp"

#define DEVICE_INLINE __device__ __forceinline__

namespace aiter::mxfp4_moe::gemm_common {

using buffer_rsrc_t = __amdgpu_buffer_rsrc_t;
using i32x4         = opus::i32x4_t;
using f32x4         = opus::fp32x4_t;

DEVICE_INLINE buffer_rsrc_t make_buffer_rsrc(const void* base, uint32_t num_bytes) {
    return opus::make_buffer_rsrc(base, num_bytes);
}

// asm-declaration form (not the clang builtin) because callers pass a runtime
// `size`; the builtin requires it to be a compile-time constant.
extern "C" __device__ void llvm_amdgcn_raw_ptr_buffer_load_lds(
    buffer_rsrc_t rsrc,
    unsigned char __attribute__((address_space(3)))* lds_ptr,
    int size, int voffset, int soffset, int offset, int aux)
    __asm("llvm.amdgcn.raw.ptr.buffer.load.lds");

DEVICE_INLINE void buffer_load_lds(
    buffer_rsrc_t rsrc, void* lds_ptr, int size,
    int voffset, int soffset, int offset, int aux)
{
    using lds_byte_ptr = unsigned char __attribute__((address_space(3)))*;
    llvm_amdgcn_raw_ptr_buffer_load_lds(
        rsrc, (lds_byte_ptr)lds_ptr,
        size, voffset, soffset, offset, aux);
}

template <int IMM_OFFSET, int AUX = 0>
DEVICE_INLINE void buffer_load_b128_imm_inplace(
    i32x4& dst, buffer_rsrc_t rsrc, int voffset, int soffset)
{
    static_assert(IMM_OFFSET >= 0 && IMM_OFFSET <= 4095,
                  "IMM_OFFSET must fit 12-bit MUBUF inst_offset");
    dst = __builtin_bit_cast(
        i32x4,
        __builtin_amdgcn_raw_buffer_load_b128(
            rsrc, voffset + IMM_OFFSET, soffset, AUX));
}

template <int IMM_OFFSET, int AUX = 0>
DEVICE_INLINE int buffer_load_b32_imm(
    buffer_rsrc_t rsrc, int voffset, int soffset)
{
    static_assert(IMM_OFFSET >= 0 && IMM_OFFSET <= 4095,
                  "IMM_OFFSET must fit 12-bit MUBUF inst_offset");
    return (int)__builtin_amdgcn_raw_buffer_load_b32(
        rsrc, voffset + IMM_OFFSET, soffset, AUX);
}

DEVICE_INLINE float silu_mul_fast(float g, float u) {
    const float e = __expf(-g);
    return g * __builtin_amdgcn_rcpf(1.0f + e) * u;
}

// Load-bearing: 16 rows × 4 dwords = 64 unique 4-bank slots per ds_read_b128.
// Don't change without re-validating.
template <int ROW_BYTES>
DEVICE_INLINE int lds_swizzle_mask(int row) {
    constexpr int kRowMask = ((ROW_BYTES / 16) - 1) << 1;
    return (row & kRowMask) << 3;
}

// Must mirror moe_sort_quant's quant_impl exactly: same bf16→fp4 packing.
DEVICE_INLINE uint8_t inline_quant_encode_e8m0(uint16_t amax_u16) {
    const uint32_t f32bits = (uint32_t)amax_u16 << 16;
    const int bexp = (int)(((f32bits + 0x200000u) >> 23) & 0xFFu);
    return (uint8_t)min(254, max(0, bexp - 2));
}

DEVICE_INLINE uint32_t inline_quant_dpp_quad_amax(uint32_t a32) {
    uint32_t s1 = (uint32_t)__builtin_amdgcn_mov_dpp((int)a32, 0xB1, 0xF, 0xF, true);
    a32 = max(a32, s1);
    uint32_t s2 = (uint32_t)__builtin_amdgcn_mov_dpp((int)a32, 0x4E, 0xF, 0xF, true);
    return max(a32, s2);
}

// v_pk_max_u16: no clang builtin in hipcc 7.2.1.
DEVICE_INLINE uint32_t inline_quant_pkmax_u16(uint32_t a, uint32_t b) {
    uint32_t out;
    asm("v_pk_max_u16 %0, %1, %2" : "=v"(out) : "v"(a), "v"(b));
    return out;
}

template <int NUM_XCDS = 8>
DEVICE_INLINE int remap_xcd(int pid_raw, int total_tiles) {
    const int ids_per_xcd = (total_tiles + NUM_XCDS - 1) / NUM_XCDS;
    int tall_xcds = total_tiles % NUM_XCDS;
    tall_xcds = (tall_xcds == 0) ? NUM_XCDS : tall_xcds;
    const int xcd       = pid_raw % NUM_XCDS;
    const int local_id  = pid_raw / NUM_XCDS;
    if (xcd < tall_xcds) {
        return xcd * ids_per_xcd + local_id;
    } else {
        return tall_xcds * ids_per_xcd
             + (xcd - tall_xcds) * (ids_per_xcd - 1)
             + local_id;
    }
}

}  // namespace aiter::mxfp4_moe::gemm_common
