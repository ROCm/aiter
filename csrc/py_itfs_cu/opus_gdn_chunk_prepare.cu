// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// gdn_chunk_prepare kernel instantiation (BT=64, K=V=128).
// Fuses chunk_local_cumsum + chunk_scaled_dot_kkt_fwd + solve_tril + recompute_w_u_fwd.
#include <hip/hip_runtime.h>
#include "opus_gdn/gdn_chunk_prepare_defs.h"
#ifndef __HIP_DEVICE_COMPILE__
template<typename Traits> __global__ void gdn_chunk_prepare_kernel(gdn_chunk_prepare_kargs kargs) {}
template __global__ void gdn_chunk_prepare_kernel<gdn_chunk_prepare_traits<64, 128, 128, 4>>(gdn_chunk_prepare_kargs);
#else
#include "opus_gdn/gdn_chunk_prepare_kernel.hpp"
template __global__ void gdn_chunk_prepare_kernel<gdn_chunk_prepare_traits<64, 128, 128, 4>>(gdn_chunk_prepare_kargs);
#endif
