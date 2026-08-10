// GDN K1 BT=32 instantiation (2-block Neumann + WY factors)
#include <hip/hip_runtime.h>
#include "opus_gdn/gdn_defs.h"
#ifndef __HIP_DEVICE_COMPILE__
template<typename Traits> __global__ void gdn_k1_bt32_kernel(gdn_k1_kargs kargs) {}
template __global__ void gdn_k1_bt32_kernel<gdn_k1_traits<32, 128, 128, 4>>(gdn_k1_kargs);
#else
#include "opus_gdn/gdn_k1_bt32_kernel_template.hpp"
template __global__ void gdn_k1_bt32_kernel<gdn_k1_traits<32, 128, 128, 4>>(gdn_k1_kargs);
#endif
