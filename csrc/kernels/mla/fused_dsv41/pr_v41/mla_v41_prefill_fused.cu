// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// build_co.sh flags: -mllvm -enable-post-misched=1 -mllvm -amdgpu-early-inline-all=false -mllvm -amdgpu-function-calls=true
// The kernel is tuned with post-RA scheduling on (3.5%) and without early inlining (15 VGPRs).
#include <hip/hip_runtime.h>

#include "mla_v41_prefill_fused.hpp"

#if !__has_attribute(amdgpu_pin_vgpr) || !__has_attribute(amdgpu_pin_agpr)
#error "build with the amdgpu-named-reg-pin-std toolchain, see build_co.sh"
#endif

extern "C" __global__ __launch_bounds__(256, 1) void
opus_mla_v41_prefill_a8w8_16mx4_64nx1_kernel(opus_mla_v41_prefill_kargs kargs)
{
    using Traits=opus_mla_v41_prefill_a8w8_16mx4_64nx1_traits<16,64,4,fp8_t,bf16_t,bf16_t>;
    __shared__ __align__(16) char smem[Traits::smem_bytes()+2*68*sizeof(int)];
    if(indices_valid(kargs)) opus_mla_v41_prefill_a8w8_16mx4_64nx1_fwd<Traits,false>(kargs,smem);
    else opus_mla_v41_prefill_a8w8_16mx4_64nx1_fwd<Traits,true>(kargs,smem);
}
