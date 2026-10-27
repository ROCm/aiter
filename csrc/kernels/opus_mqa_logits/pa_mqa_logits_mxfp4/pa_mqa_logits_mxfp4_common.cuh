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

// ceil(n / d) for 0 <= n, 1 <= d, n + d < 2^22, with rcp = 1 / d.
__device__ inline int ceil_div_rcp(int n, int d, float rcp) {
    const int m = n + d - 1;
    int q = (int)((float)m * rcp);
    q = q * d > m ? q - 1 : q;
    return (q + 1) * d <= m ? q + 1 : q;
}

}  // namespace opus_logits
