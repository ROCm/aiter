// Faithful triton fwd_h port: 32x32x16 MFMA scan, nw=2.
#include <hip/hip_runtime.h>
#include "opus_gdn/gdn_defs.h"
#include "opus_gdn/gdn_k2_scan32_kernel_template.hpp"
template __global__ void gdn_k2_scan32_kernel<gdn_k2_traits<64, 128, 128, 32, 8>>(gdn_k2_kargs);
