// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Device helpers shared by the gfx1250 MXFP4 MQA-logits kernels. Device pass on gfx1250 only.
#pragma once
#include <opus/opus.hpp>
#include <bit>   // std::bit_cast: __builtin_bit_cast miscompiles the permlane swap to v + v

// relu as max_num, which VOPD issues two at a time where the IEEE `maximum` is one VOP3 an
// element. max_num drops NaN, and NaN logits are a contract (the scale pool's 0xFF rows; top-k
// ranks NaN first), so the kernels put the NaN back from the scales: a logit is NaN exactly when
// its key's or its row's E8M0 scales hold a 0xFF byte -- exactly when the WMMA's scores are NaN.
// The TU must be built with -fno-honor-nans: without it the compiler quiets the WMMA's output
// before every max_num and gains nothing. 0 restores the IEEE maximum (build without the flag).
#ifndef MQA_LOGITS_MXFP4_NUM_RELU
#define MQA_LOGITS_MXFP4_NUM_RELU 1
#endif
// Test only: 8 drops the NaN restore (and the IEEE element that carries a key's NaN), so the op
// test's NaN rows must fail. Every other bit is reserved; 0 is the kernel.
#ifndef MQA_LOGITS_MXFP4_KNOCK
#define MQA_LOGITS_MXFP4_KNOCK 0
#endif
#if MQA_LOGITS_MXFP4_NUM_RELU
#define MQA_LOGITS_RELU(x) __builtin_fmaxf((x), 0.0f)
#else
#define MQA_LOGITS_RELU(x) __builtin_elementwise_maximum((x), 0.0f)
#endif
#define MQA_LOGITS_NAN_RESTORE (MQA_LOGITS_MXFP4_NUM_RELU && !(MQA_LOGITS_MXFP4_KNOCK & 8))

namespace opus_logits::gfx1250 {

// "+s" on purpose: the read-only form is a memory clobber and demotes indexed s_loads. Never
// pin a pointer: the asm divergence spreads to every derived address and costs occupancy.
template<class X>
__device__ inline void pin_sgpr(X& v) { asm("" : "+s"(v)); }

// Fenced so the scheduler cannot move LDS traffic across it; callers owe the preceding wait.
__device__ inline void cta_sync() {
    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);
}

// The cross-group finish for TWO n-tiles at once. v_permlane16_swap trades vdst's high row
// (lanes 16-31) for vsrc's low row (lanes 0-15), so swapping a with b leaves (a.lo, b.lo) and
// (a.hi, b.hi), and their sum holds n-tile a's 16 logits in lanes 0-15 and b's in lanes 16-31:
// one swap and one add for two n-tiles, and every lane a logit to store. Both halves of the
// builtin's result are used (r[0] alone computes v + v).
__device__ inline float pair_head_reduce(float a, float b) {
    opus::vector_t<opus::u32_t, 2> r = __builtin_amdgcn_permlane16_swap(
        std::bit_cast<opus::u32_t>(a), std::bit_cast<opus::u32_t>(b), false, false);
    return std::bit_cast<float>(r[0]) + std::bit_cast<float>(r[1]);
}

// A head sum's last add, (even heads, odd heads) -> one value, as one pk_add of the pair with
// its swap (op_sel) behind an empty asm: as a scalar add the backend packs two accumulators'
// adds into one pk_add and moves the halves into place for it. The asm is a VGPR operand, so
// only for kernels at or under 256 VGPRs (above it the VGPR-MSB mode makes VGPR asm unsafe).
template<bool ASM>
__device__ inline float pair_sum(opus::vector_t<float, 2> s) {
    if constexpr (ASM) {
        opus::vector_t<float, 2> d = s + __builtin_shufflevector(s, s, 1, 0);
        asm volatile("" : "+v"(d));
        return d[0];
    } else {
        return s[0] + s[1];
    }
}

// Does a dword of four E8M0 scales hold a 0xFF (NaN) byte? Each byte's low 7 bits plus one
// carries into its bit 7 only at 0x7F, and never into the next byte; with the byte's own bit 7,
// that is exactly 0xFF. Integer, because -fno-honor-nans folds an FP NaN test to false.
__device__ inline bool e8m0x4_has_nan(unsigned x) {
    return (((x & 0x7f7f7f7fu) + 0x01010101u) & x & 0x80808080u) != 0;
}

// s_setprio 3..0 by the remaining quarter of `chunk_max` tiles: a wave with more of its work
// left asks for the SIMD first, so the CTAs that started late are not starved into a
// latency-bound tail by the hardware's age order.
__device__ inline void set_prio_by_progress(int remaining, int chunk_max) {
    const int rem4 = remaining * 4;   // uniform
    if      (rem4 > 3 * chunk_max) __builtin_amdgcn_s_setprio(3);
    else if (rem4 > 2 * chunk_max) __builtin_amdgcn_s_setprio(2);
    else if (rem4 > chunk_max)     __builtin_amdgcn_s_setprio(1);
    else                           __builtin_amdgcn_s_setprio(0);
}

}  // namespace opus_logits::gfx1250
