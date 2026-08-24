// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

// Shared host-side helpers for the a8w8 mxscale BMM kernel families. The
// per-kid host launchers are now fully codegen'd (impl/*.cuh, compiled in
// all_instances_host_gfx950.cu) as inlined template functions, mirroring the
// opus_gemm module. The old hand-written `*_impl` launcher templates that used
// to live here have been removed; only the common shape/dtype check remains.
// The device-kernel definitions are provided by the flatmm split-K pipeline
// header (included ahead of this one in opus_bmm.cu), so no forward
// declarations are needed here. Host pass only.
#ifndef __HIP_DEVICE_COMPILE__

#include "opus_bmm.h"
#include "opus_gemm_arch.cuh"
#include "opus_build_archs.h"
#include "opus_gemm_utils.cuh"  // bf16_t / fp32_t
#include "aiter_stream.h"

#include <optional>

static void opus_bmm_a8w8_common_checks(aiter_tensor_t &O, aiter_tensor_t &wo_a,
                                        aiter_tensor_t &Y, const char *who)
{
  aiter_detail::g_aiter_can_throw = true;
  AITER_CHECK(O.dim() == 3 && wo_a.dim() == 3 && Y.dim() == 3,
              who, ": O/wo_a/Y must be 3D "
              "([M,batch,K] / [batch,N,K] / [M,batch,N])");
  AITER_CHECK(O.dtype() == AITER_DTYPE_fp8 && wo_a.dtype() == AITER_DTYPE_fp8,
              who, ": O and wo_a must be fp8");
  AITER_CHECK(Y.dtype() == AITER_DTYPE_fp32 || Y.dtype() == AITER_DTYPE_bf16,
              who, ": Y must be fp32 or bf16");
  // The kernels index A/B along K with unit stride (kargs carries only M/N/batch
  // strides, never a K stride), so K must be the innermost contiguous dim. The
  // batch axis position is free -- it is fully described by stride_*_batch -- so
  // any batch layout (m-major [M,batch,K], batch-major view, ...) is accepted as
  // long as K stays contiguous. Reject anything else here (host-side, once per
  // launch) rather than silently producing wrong results.
  AITER_CHECK(O.stride(2) == 1, who,
              ": O (x) must be K-contiguous (stride(2)==1); got stride ",
              (long)O.stride(2));
  AITER_CHECK(wo_a.stride(2) == 1, who,
              ": wo_a must be K-contiguous (stride(2)==1); got stride ",
              (long)wo_a.stride(2));
}

#endif // __HIP_DEVICE_COMPILE__
