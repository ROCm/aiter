#include <opus/hip_minimal.hpp>
#include "defs.h"

using mla_32mx4_traits_t = opus_mla_decode_a16w16_32mx4_32nx1_traits<32, 32, 4, bf16_t, bf16_t>;
using mla_32mx3_traits_t = opus_mla_decode_a16w16_32mx3_32nx1_traits<32, 32, 4, bf16_t, bf16_t>;

#define MLA_DECODE_32MXN_INSTANTIATIONS                                                                    \
    template __global__ void opus_mla_decode_a16w16_32mxt_32nx1_kernel<mla_32mx4_traits_t>(opus_mla_decode_kargs);  \
    template __global__ void opus_mla_decode_a16w16_32mxt_32nx1_kernel<mla_32mx3_traits_t>(opus_mla_decode_kargs);

#ifndef __HIP_DEVICE_COMPILE__
template<typename Traits> __global__ void opus_mla_decode_a16w16_32mxt_32nx1_kernel(opus_mla_decode_kargs kargs) {}
MLA_DECODE_32MXN_INSTANTIATIONS
#else
#include "mla_decode_a16w16_32mxt_32nx1_template.hpp"
MLA_DECODE_32MXN_INSTANTIATIONS
#endif
