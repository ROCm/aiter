// build_co.sh flags: -fno-slp-vectorize -mllvm -amdgpu-mfma-vgpr-form=1
// No SLP (v_pk_fma_f32 does not co-issue with MFMA); the compiler's MFMAs accumulate in VGPRs, so
// the AGPR file holds only the pinned dK / dV / dQ accumulators.
#include <hip/hip_runtime.h>

#include "fmha_bwd_hd192_v128_bf16.hpp"

#if !__has_attribute(amdgpu_pin_vgpr) || !__has_attribute(amdgpu_pin_agpr)
#error "build with the amdgpu-named-reg-pin-std toolchain, see build_co.sh"
#endif

// clang-format off: the entry points as the other .cu of this directory write them.
// The pre-pass, without and with the output gate.
extern "C" __global__ __launch_bounds__(FMHA_BWD_V128_PRE_BLOCK) void
opus_fmha_bwd_v128_pre_kernel(fmha_bwd_v128_pre_kargs kargs)
{
    fmha_bwd_v128_pre_body<false>(kargs);
}

extern "C" __global__ __launch_bounds__(FMHA_BWD_V128_PRE_BLOCK) void
opus_fmha_bwd_v128_pre_gate_kernel(fmha_bwd_v128_pre_kargs kargs)
{
    fmha_bwd_v128_pre_body<true>(kargs);
}

// The dK / dV pass and the dQ pass.
extern "C" __global__ __launch_bounds__(fmha_bwd_d192_v128_dkdv::BLOCK, 1) void
opus_fmha_bwd_d192_v128_kernel(fmha_bwd_d192_v128_kargs kargs)
{
    fmha_bwd_d192_v128_body(kargs);
}
// clang-format on
