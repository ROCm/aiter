#include <opus/hip_minimal.hpp>
#include "defs.h"

using mxfp8_traits_t         = opus_mla_decode_mxfp8_16mx8_32nx1_traits<16, 32, 8, fp8_t, bf16_t, bf16_t>;
using a16w16_16mx4_traits_t  = opus_mla_decode_a16w16_16mx4_64nx1_traits<16, 64, 4, bf16_t, bf16_t>;
using a16w16_32mx1_traits_t  = opus_mla_decode_a16w16_32mx1_16nx4_traits<32, 64, 4, bf16_t, bf16_t>;
using a16w16_32mx4_traits_t  = opus_mla_decode_a16w16_32mx4_32nx1_traits<32, 32, 4, bf16_t, bf16_t>;
using a16w16_32mx3_traits_t  = opus_mla_decode_a16w16_32mx3_32nx1_traits<32, 32, 4, bf16_t, bf16_t>;

#define MLA_REDUCE_INSTANTIATIONS                                                                             \
    template __global__ void mla_reduce_kernel<mxfp8_traits_t>(opus_mla_decode_reduce_kargs);         \
    template __global__ void mla_reduce_kernel<a16w16_16mx4_traits_t>(opus_mla_decode_reduce_kargs);  \
    template __global__ void mla_reduce_kernel<a16w16_32mx1_traits_t>(opus_mla_decode_reduce_kargs);  \
    template __global__ void mla_reduce_kernel<a16w16_32mx4_traits_t>(opus_mla_decode_reduce_kargs);  \
    template __global__ void mla_reduce_kernel<a16w16_32mx3_traits_t>(opus_mla_decode_reduce_kargs);

#ifndef __HIP_DEVICE_COMPILE__
template<typename Traits, int HEADS_PER_BLOCK = 8> __global__ void mla_reduce_kernel(opus_mla_decode_reduce_kargs kargs) { (void)kargs; }
MLA_REDUCE_INSTANTIATIONS
#else
#include "mla_reduce.hpp"
MLA_REDUCE_INSTANTIATIONS
#endif
