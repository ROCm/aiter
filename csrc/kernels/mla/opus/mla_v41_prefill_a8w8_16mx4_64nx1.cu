// build_co.sh flags: -mllvm -enable-post-misched=1 -mllvm -amdgpu-early-inline-all=false -mllvm -amdgpu-function-calls=true
// The kernel is tuned with post-RA scheduling on (3.5%) and without early inlining (15 VGPRs).
#include <hip/hip_runtime.h>

#include "mla_v41_prefill_a8w8_16mx4_64nx1.hpp"

#if !__has_attribute(amdgpu_pin_vgpr) || !__has_attribute(amdgpu_pin_agpr)
#error "build with the amdgpu-named-reg-pin-std toolchain, see build_co.sh"
#endif

using opus_mla_v41_prefill_a8w8_16mx4_64nx1_t =
    opus_mla_v41_prefill_a8w8_16mx4_64nx1_traits<16, 64, 4, fp8_t, bf16_t, bf16_t>;

extern "C" __global__ __launch_bounds__(256, 1) void
opus_mla_v41_prefill_a8w8_16mx4_64nx1_kernel(opus_mla_v41_prefill_kargs kargs)
{
    opus_mla_v41_prefill_a8w8_16mx4_64nx1_fwd<opus_mla_v41_prefill_a8w8_16mx4_64nx1_t>(kargs);
}

extern "C" __global__ __launch_bounds__(256, 1) void opus_mla_v41_prefill_a8w8_16mx4_64nx1_split_kernel(
    opus_mla_prefill_split_kargs<opus_mla_v41_prefill_kargs> kargs)
{
    opus_mla_v41_prefill_a8w8_16mx4_64nx1_fwd<opus_mla_v41_prefill_a8w8_16mx4_64nx1_t>(kargs);
}
