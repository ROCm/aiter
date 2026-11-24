#include <hip/hip_runtime.h>

#include "mla_decode_a16w16_32mx1_16nx4.hpp"

#if !__has_attribute(amdgpu_pin_vgpr) || !__has_attribute(amdgpu_pin_agpr)
#error "build with the amdgpu-named-reg-pin-std toolchain, see build_co.sh"
#endif

extern "C" __global__ __launch_bounds__(256, 1) void
opus_mla_decode_a16w16_32mx1_16nx4_kernel(opus_mla_decode_kargs kargs)
{
    opus_mla_decode_a16w16_32mx1_16nx4::mla_decode_fwd_persistent<
        opus_mla_decode_a16w16_32mx1_16nx4_traits<32, 64, 4, bf16_t, bf16_t>>(kargs);
}
