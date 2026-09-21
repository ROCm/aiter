#include <opus/hip_minimal.hpp>
#include "defs.h"

using mla_decode_traits_t = opus_mla_decode_a16w16_32mx1_16nx4_traits<32, 64, 4, bf16_t, bf16_t>;

#ifndef __HIP_DEVICE_COMPILE__
template<typename Traits> __global__ void opus_mla_decode_a16w16_32mx1_16nx4_kernel(opus_mla_decode_kargs kargs) {}
template __global__ void opus_mla_decode_a16w16_32mx1_16nx4_kernel<mla_decode_traits_t>(opus_mla_decode_kargs);
#else
#include "mla_decode_a16w16_32mx1_16nx4_template.hpp"
template __global__ void opus_mla_decode_a16w16_32mx1_16nx4_kernel<mla_decode_traits_t>(opus_mla_decode_kargs);
#endif
