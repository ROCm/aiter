#include <hip/hip_runtime.h>

#include "mla_decode_a16w16_32mxt_32nx1.hpp"

#if !__has_attribute(amdgpu_pin_vgpr) || !__has_attribute(amdgpu_pin_agpr)
#error "build with the amdgpu-named-reg-pin-std toolchain, see build_co.sh"
#endif

extern "C" __global__ __launch_bounds__(256, 1) void
opus_mla_decode_a16w16_32mx4_32nx1_kernel(opus_mla_decode_kargs kargs)
{
    opus_mla_decode_a16w16_32mxt_32nx1::mla_decode_fwd_persistent<
        opus_mla_decode_a16w16_32mx4_32nx1_traits<32, 32, 4, bf16_t, bf16_t>>(kargs);
}

extern "C" __global__ __launch_bounds__(256, 1) void
opus_mla_decode_a16w16_32mx3_32nx1_kernel(opus_mla_decode_kargs kargs)
{
    opus_mla_decode_a16w16_32mxt_32nx1::mla_decode_fwd_persistent<
        opus_mla_decode_a16w16_32mx3_32nx1_traits<32, 32, 4, bf16_t, bf16_t>>(kargs);
}
