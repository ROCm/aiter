// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Device helpers every arch's MXFP4 MQA-logits kernels share. Device passes only.
#pragma once

namespace opus_logits {

// Keep `v` in an SGPR. The "+s" write form on purpose: the read-only form is a memory clobber that
// demotes s_loads to global_load + v_readfirstlane. Losing the pin costs occupancy silently.
template<class X>
__device__ inline void pin_sgpr(X& v) { asm("" : "+s"(v)); }

// The CTA rendezvous, fenced so the scheduler cannot move LDS traffic across it. A bare s_barrier
// orders execution, not memory: the caller's wait gives it something to publish.
__device__ inline void cta_sync() {
    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);
}

// ceil(n / d) for 0 <= n, 1 <= d, with rcp = 1 / d, while n + 2d fits an int. The float quotient
// is within one of n / d while it stays under 2^22 (float's relative error), and the two
// corrections fix that one; for n < 2^25 it is exact at any d (checked exhaustively for d <= 8,
// where the quotient may pass 2^22). The callers divide a call's tiles by its CTA target
// (hundreds or more) and a window's tiles (< 2^25: row_ends is an int) by the chunk size.
__device__ inline int ceil_div_rcp(int n, int d, float rcp) {
    const int m = n + d - 1;
    int q = (int)((float)m * rcp);
    q = q * d > m ? q - 1 : q;
    return (q + 1) * d <= m ? q + 1 : q;
}

// floor(part * tiles / parts) for 0 <= part <= parts <= 65536, in 32 bits whatever tiles is:
// part * tiles = part * q * parts + part * r with q, r = tiles / parts, tiles % parts, and
// part * r < 2^32.
__device__ inline int part_point(int part, int tiles, int parts) {
    const unsigned q = (unsigned)tiles / (unsigned)parts, r = (unsigned)tiles - q * (unsigned)parts;
    return (int)((unsigned)part * q + (unsigned)part * r / (unsigned)parts);
}

}  // namespace opus_logits
