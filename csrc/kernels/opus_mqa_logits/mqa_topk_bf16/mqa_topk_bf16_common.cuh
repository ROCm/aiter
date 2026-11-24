// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Arch-neutral device helpers of mqa_topk_bf16 (wave64): sample positions, the order key, wave
// scans and reductions.
#pragma once
#include "mqa_topk_bf16.h"

namespace opus_mqa_topk {

// Sample j stands for keys [32 j, 32 j + 32): a fixed hash picks one of them, so a pattern with a
// period of 32 keys cannot hide from (or crowd) the samples. Non-decreasing in j.
__host__ __device__ inline int sample_key(int j, int keys) {
    const unsigned h = (unsigned)j * 0x9E3779B1u;
    const int c      = j * SAMPLE_STRIDE + (int)(h >> 27);
    return c < keys ? c : keys - 1;
}

// The first sample at or after key x (0 <= x <= keys): [at(lo), at(hi)) are a window's samples.
__device__ inline int sample_at_or_after(int x, int keys) {
    const int j = x / SAMPLE_STRIDE;
    return j < num_samples(keys) && sample_key(j, keys) < x ? j + 1 : j;
}

// A window clamped into [0, keys): lo <= hi.
struct window {
    int lo, hi;
};
__device__ inline window clamp_window(int start, int end, int keys) {
    const int lo = start < 0 ? 0 : start > keys ? keys : start;
    const int hi = end < lo ? lo : end > keys ? keys : end;
    return {lo, hi};
}

// Unsigned keys that order like the scores; -0 orders as +0.
__device__ inline unsigned order_key(float s) {
    unsigned u = __builtin_bit_cast(unsigned, s);
    u          = u == 0x80000000u ? 0u : u;
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}
__device__ inline float order_value(unsigned k) {
    return __builtin_bit_cast(float, (k & 0x80000000u) ? (k & 0x7fffffffu) : ~k);
}

__device__ inline int lane_id() {
    return __builtin_amdgcn_mbcnt_hi(-1, __builtin_amdgcn_mbcnt_lo(-1, 0));
}

// Inclusive running sum over a wave64: row_shr 1/2/4/8 inside each 16-lane row, then row_bcast15
// into rows 1 and 3 and row_bcast31 into rows 2 and 3. `old` is 0 and bound_ctrl off, so a lane
// whose source is out of reach adds 0.
__device__ inline int wave_incl_scan(int v) {
    v += __builtin_amdgcn_update_dpp(0, v, 0x111, 0xF, 0xF, false);
    v += __builtin_amdgcn_update_dpp(0, v, 0x112, 0xF, 0xF, false);
    v += __builtin_amdgcn_update_dpp(0, v, 0x114, 0xF, 0xF, false);
    v += __builtin_amdgcn_update_dpp(0, v, 0x118, 0xF, 0xF, false);
    v += __builtin_amdgcn_update_dpp(0, v, 0x142, 0xA, 0xF, false);
    v += __builtin_amdgcn_update_dpp(0, v, 0x143, 0xC, 0xF, false);
    return v;
}

// The same inside each 16-lane row only.
__device__ inline int row16_incl_scan(int v) {
    v += __builtin_amdgcn_update_dpp(0, v, 0x111, 0xF, 0xF, false);
    v += __builtin_amdgcn_update_dpp(0, v, 0x112, 0xF, 0xF, false);
    v += __builtin_amdgcn_update_dpp(0, v, 0x114, 0xF, 0xF, false);
    v += __builtin_amdgcn_update_dpp(0, v, 0x118, 0xF, 0xF, false);
    return v;
}

__device__ inline int wave_sum(int v) { return __builtin_amdgcn_readlane(wave_incl_scan(v), 63); }

__device__ inline unsigned wave_min(unsigned v) {
    for (int o = 32; o >= 1; o >>= 1) {
        const unsigned y = (unsigned)__shfl_xor((int)v, o, 64);
        v = y < v ? y : v;
    }
    return v;
}
__device__ inline unsigned wave_max(unsigned v) {
    for (int o = 32; o >= 1; o >>= 1) {
        const unsigned y = (unsigned)__shfl_xor((int)v, o, 64);
        v = y > v ? y : v;
    }
    return v;
}

}  // namespace opus_mqa_topk
