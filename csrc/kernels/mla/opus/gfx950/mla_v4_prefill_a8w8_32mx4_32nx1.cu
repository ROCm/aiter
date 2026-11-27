// build_co.sh flags: -fno-slp-vectorize -mno-amdgpu-ieee -mllvm -enable-post-misched=1 -mllvm -amdgpu-early-inline-all=false -mllvm -amdgpu-function-calls=true
// No SLP (v_pk_fma_f32 does not co-issue with MFMA), no NaN canonicalization in the row max.
#include <hip/hip_runtime.h>

#include "mla_v4_prefill_a8w8_32mx4_32nx1.hpp"

#if !__has_attribute(amdgpu_pin_vgpr) || !__has_attribute(amdgpu_pin_agpr)
#error "build with the amdgpu-named-reg-pin-std toolchain, see build_co.sh"
#endif

template <bool SPLIT>
using opus_mla_v4_prefill_a8w8_32mx4_32nx1_t =
    opus_mla_v4_prefill_a8w8_32mx4_32nx1_traits<32, 32, 4, fp8_t, bf16_t, bf16_t, SPLIT>;

extern "C" __global__ __launch_bounds__(256, 1) void
opus_mla_v4_prefill_a8w8_32mx4_32nx1_kernel(opus_mla_v4_prefill_fp8_kargs kargs)
{
    opus_mla_v4_prefill_a8w8_32mx4_32nx1_fwd<opus_mla_v4_prefill_a8w8_32mx4_32nx1_t<false>>(kargs);
}

extern "C" __global__ __launch_bounds__(256, 1) void opus_mla_v4_prefill_a8w8_32mx4_32nx1_split_kernel(
    opus_mla_prefill_split_kargs<opus_mla_v4_prefill_fp8_kargs> kargs)
{
    opus_mla_v4_prefill_a8w8_32mx4_32nx1_fwd<opus_mla_v4_prefill_a8w8_32mx4_32nx1_t<true>>(kargs);
}
