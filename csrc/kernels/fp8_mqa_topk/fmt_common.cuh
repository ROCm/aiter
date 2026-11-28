// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Fused FP8 MQA logits + top-k (gfx950): shared types and wave-level helpers.
// The pipeline is described in csrc/kernels/fp8_mqa_topk.cu.
#pragma once
#include <hip/hip_runtime.h>
#include <hip/hip_fp8.h>
#include <climits>
#include <cstdint>
#include <type_traits>

namespace mqa_topk {

template <int I> using IC = std::integral_constant<int, I>;
using fp8 = __hip_fp8_storage_t;
constexpr int WARP = 64;
constexpr int NUM_HEADS = 32;
constexpr int HEAD_SIZE = 128;
#define MQA_TOPK_CDIV(a, b) (((a) + (b) - 1) / (b))
#define MQA_TOPK_BUF_RSRC_WORD3 0x00020000

using f8x16 = __attribute__((__vector_size__(16))) fp8;
using f8x32 = __attribute__((__vector_size__(32))) fp8;
using f32x4 = __attribute__((__vector_size__(16))) float;
using f32x2 = __attribute__((__vector_size__(8))) float;
using u32x4 = __attribute__((__vector_size__(16))) unsigned int;
using i32x4 = __attribute__((__vector_size__(16))) int;
using VecOut = __attribute__((__vector_size__(16 * sizeof(float)))) float;

template <int N> __device__ __forceinline__ void wait_vmcnt() {
    static_assert(N >= 0 && N < 64, "vmcnt");
    asm volatile("s_waitcnt vmcnt(%0)" :: "n"(N) : "memory");
}

// NOPS = s_nop wait states issued before the permlane: the caller guarantees >= 2 - NOPS independent instructions
// since the last VALU write of a / b (VALU write -> v_permlane32_swap read hazard; the compiler does not check it for
// inline asm). __builtin_amdgcn_permlane32_swap is unusable on ROCm 7.2.3 clang (2nd result miscompiled).
template <int NOPS>
__device__ __forceinline__ void swap32_reduce2(float& a, float& b) {
    if constexpr (NOPS == 0) asm("v_permlane32_swap_b32 %0, %1" : "+v"(a), "+v"(b));
    else if constexpr (NOPS == 1) asm("s_nop 0\n\tv_permlane32_swap_b32 %0, %1" : "+v"(a), "+v"(b));
    else asm("s_nop 1\n\tv_permlane32_swap_b32 %0, %1" : "+v"(a), "+v"(b));
}

__device__ __forceinline__ f8x32 pack(f8x16 lo, f8x16 hi) {
    f8x32 v; *reinterpret_cast<f8x16*>(&v) = lo; *(reinterpret_cast<f8x16*>(&v) + 1) = hi; return v;
}

// raw buffer V# (stride 0, raw buffer) from a pointer and a byte size; SGPR-uniform
__device__ __forceinline__ i32x4 raw_rsrc(const void* p, int bytes) {
    const uint64_t a = reinterpret_cast<uint64_t>(p);
    i32x4 r;
    r[0] = __builtin_amdgcn_readfirstlane((int)(uint32_t)a);
    r[1] = __builtin_amdgcn_readfirstlane((int)(uint32_t)(a >> 32) & 0xffff);
    r[2] = __builtin_amdgcn_readfirstlane(bytes);
    r[3] = MQA_TOPK_BUF_RSRC_WORD3;
    return r;
}

// order-preserving 32-bit key of an fp32 bit pattern (larger float -> larger key) and its inverse
__device__ __forceinline__ uint32_t okey32(uint32_t u) { return u ^ (((uint32_t)((int32_t)u >> 31)) | 0x80000000u); }
__device__ __forceinline__ uint32_t okey32_inv(uint32_t k) { return (k & 0x80000000u) ? (k ^ 0x80000000u) : ~k; }

__device__ __forceinline__ int wave_incl_scan(int x, int lane) {
    #pragma unroll
    for (int d = 1; d < WARP; d <<= 1) { const int o = __shfl_up(x, d); if (lane >= d) x += o; }
    return x;
}
__device__ __forceinline__ void mem_fence_wave() { asm volatile("" ::: "memory"); }

// candidate slot of row m, split s, index i: interleaved pair segments of 2*CAP u64 (rows 2p, 2p+1), Mp = M rounded
// up to even. The scan writes this layout; the select reads it.
__host__ __device__ __forceinline__ int64_t cand_index(int s, int m, int i, int Mp, int CAP) {
    return ((int64_t)s * Mp + (m & ~1)) * CAP + 2 * (int64_t)i + (m & 1);
}

}  // namespace mqa_topk
