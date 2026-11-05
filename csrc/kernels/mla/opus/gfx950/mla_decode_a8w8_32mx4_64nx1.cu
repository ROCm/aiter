#include <hip/hip_runtime.h>

#include "mla_decode_a8w8_32mx4_64nx1.hpp"

#if !__has_attribute(amdgpu_pin_vgpr) || !__has_attribute(amdgpu_pin_agpr)
#error "build with the amdgpu-named-reg-pin-std toolchain, see build_co.sh"
#endif

template <bool CAUSAL, bool LARGE_KV, bool CP>
using mla32mx4_traits_t =
    opus_mla_decode_a8w8_32mx4_64nx1_traits<32, 64, 4, fp8_t, fp8_t, bf16_t, CAUSAL, LARGE_KV, 3,
                                           CP>;

// One entry per (causal, large KV, round-robin CP) specialization.
#define MLA32MX4_CO_ENTRY(suffix, causal, large, cp)                                       \
    extern "C" __global__ __launch_bounds__(256, 1) void                                   \
        opus_mla_decode_a8w8_32mx4_64nx1_##suffix(opus_mla_decode_a8w8_kargs kargs)        \
    {                                                                                      \
        mla_decode_fwd_32mx4_64nx1_a8w8::mla_decode_fwd_persistent<                        \
            mla32mx4_traits_t<causal, large, cp>>(kargs);                                  \
    }

MLA32MX4_CO_ENTRY(nc, false, false, false)
MLA32MX4_CO_ENTRY(causal, true, false, false)
MLA32MX4_CO_ENTRY(nc_large, false, true, false)
MLA32MX4_CO_ENTRY(causal_large, true, true, false)
MLA32MX4_CO_ENTRY(causal_cprr, true, false, true)
MLA32MX4_CO_ENTRY(causal_large_cprr, true, true, true)
