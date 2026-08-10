// GDN K2 instantiation (h update + output)
#include <hip/hip_runtime.h>
#include "opus_gdn/gdn_defs.h"
#ifndef __HIP_DEVICE_COMPILE__
template<typename Traits> __global__ void gdn_k2_kernel(gdn_k2_kargs kargs) {}
#else
#include "opus_gdn/gdn_k2_kernel_template.hpp"
#endif
template __global__ void gdn_k2_kernel<gdn_k2_traits<64, 128, 128, 64, 4>>(gdn_k2_kargs);
template __global__ void gdn_k2_kernel<gdn_k2_traits<64, 128, 128, 32, 4>>(gdn_k2_kargs);
template __global__ void gdn_k2_kernel<gdn_k2_traits<64, 128, 128, 64, 8>>(gdn_k2_kargs);

// gfx942 dense fused W/U A/B variants.  Keep each optimization stage as a
// distinct symbol so OPUS_GDN_WF_VARIANT can bisect both performance and
// numerical regressions without rebuilding the JIT module.
using K2_WF_Base = gdn_k2_traits<64, 128, 128, 64, 8>;
using K2_WF_V1 = gdn_k2_fused_traits<
    K2_WF_Base, true, true, false, false, false>;
using K2_WF_V2 = gdn_k2_fused_traits<
    K2_WF_Base, true, true, false, true, false>;
using K2_WF_V3 = gdn_k2_fused_traits<
    K2_WF_Base, true, true, true, true, false>;
using K2_WF_V4 = gdn_k2_fused_traits<
    K2_WF_Base, true, true, true, true, true>;
using K2_WF_V5 = gdn_k2_fused_traits<
    gdn_k2_traits<64, 128, 128, 128, 16>,
    true, true, true, true, true>;
using K2_WF_V6 = gdn_k2_fused_traits<
    gdn_k2_traits<64, 128, 128, 128, 16>,
    true, true, true, true, false>;
template __global__ void gdn_k2_kernel<K2_WF_V1>(gdn_k2_kargs);
template __global__ void gdn_k2_kernel<K2_WF_V2>(gdn_k2_kargs);
template __global__ void gdn_k2_kernel<K2_WF_V3>(gdn_k2_kargs);
template __global__ void gdn_k2_kernel<K2_WF_V4>(gdn_k2_kargs);
template __global__ void gdn_k2_kernel<K2_WF_V5>(gdn_k2_kargs);
template __global__ void gdn_k2_kernel<K2_WF_V6>(gdn_k2_kargs);

// OCC=2 variant: separate function with forced VGPR limit
using K2_OCC2_Traits = gdn_k2_traits<64, 128, 128, 64, 8, 2>;
#ifdef __HIP_DEVICE_COMPILE__
__global__ void __launch_bounds__(512, 2)
__attribute__((amdgpu_waves_per_eu(4, 4)))
gdn_k2_kernel_occ2(gdn_k2_kargs kargs) {
    gdn_k2_kernel_impl<K2_OCC2_Traits>(kargs);
}
#else
__global__ void gdn_k2_kernel_occ2(gdn_k2_kargs kargs) {}
#endif
template __global__ void gdn_k2_kernel<gdn_k2_traits<64, 128, 128, 32, 8>>(gdn_k2_kargs);
template __global__ void gdn_k2_kernel<gdn_k2_traits<32, 128, 128, 64, 4>>(gdn_k2_kargs);
template __global__ void gdn_k2_kernel<gdn_k2_traits<16, 128, 128, 64, 4>>(gdn_k2_kargs);
template __global__ void gdn_k2_kernel<gdn_k2_traits<128, 128, 128, 64, 4>>(gdn_k2_kargs);
template __global__ void gdn_k2_kernel<gdn_k2_traits<128, 128, 128, 32, 4>>(gdn_k2_kargs);
template __global__ void gdn_k2_kernel<gdn_k2_traits<128, 128, 128, 64, 8>>(gdn_k2_kargs);
template __global__ void gdn_k2_kernel<gdn_k2_traits<128, 128, 128, 32, 8>>(gdn_k2_kargs);
