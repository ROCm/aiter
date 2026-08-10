// Pure-HIP single-warp scan instantiation (split-path, BV=32).
#include <hip/hip_runtime.h>
#include "opus_gdn/gdn_defs.h"
#include "opus_gdn/gdn_k2_scan_hip_kernel_template.hpp"
template __global__ void gdn_k2_scan_hip_kernel<gdn_k2_traits<64, 128, 128, 32, 8>>(gdn_k2_kargs);
