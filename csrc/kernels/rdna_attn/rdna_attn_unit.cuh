// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// One translation unit of a module_rdna_attn_d<head dim>_<dtype> module:
// RDNA_ATTN_UNIT(kernel) builds the entries of the decode, decode_batch or
// prefill tables with the module's head dim and dtype, and registers them.
//
// The module's flags (aiter/ops/rdna_attn.py):
//   RDNA_ATTN_HEAD_DIM  the head dim, a HeadDim enumerator's value; without
//                       it the unit is empty (the tables-only module)
//   RDNA_ATTN_BF16      1 for bf16, 0 for fp16
//   RDNA_ATTN_GFX1151, RDNA_ATTN_GFX1150, RDNA_ATTN_GFX1201
//                       the GPU families (Arch in rdna_attn_types.h) to build
//
// A family's kernels are compiled in the device passes of its archs only: its
// tables' entries need not build for the others.
#pragma once

#ifdef RDNA_ATTN_HEAD_DIM

#include "rdna_attn_instances.cuh"

namespace rdna {
inline constexpr HeadDim kUnitHeadDim = static_cast<HeadDim>(RDNA_ATTN_HEAD_DIM);
inline constexpr Dtype kUnitDtype = RDNA_ATTN_BF16 ? Dtype::kBf16 : Dtype::kFp16;
}  // namespace rdna

#define RDNA_ATTN_UNIT_OF(kernel, family)                              \
  static const bool kRegistered##family = rdna::register_##kernel<     \
      rdna::Arch::k##family, rdna::kUnitHeadDim, rdna::kUnitDtype>();

#if defined(RDNA_ATTN_GFX1151) && \
    (!defined(__HIP_DEVICE_COMPILE__) || defined(__gfx1151__))
  #define RDNA_ATTN_UNIT_GFX1151(kernel) RDNA_ATTN_UNIT_OF(kernel, Gfx1151)
#else
  #define RDNA_ATTN_UNIT_GFX1151(kernel)
#endif

#if defined(RDNA_ATTN_GFX1150) &&                                     \
    (!defined(__HIP_DEVICE_COMPILE__) || defined(__gfx1150__) ||      \
     defined(__gfx1152__) || defined(__gfx1153__))
  #define RDNA_ATTN_UNIT_GFX1150(kernel) RDNA_ATTN_UNIT_OF(kernel, Gfx1150)
#else
  #define RDNA_ATTN_UNIT_GFX1150(kernel)
#endif

#if defined(RDNA_ATTN_GFX1201) && \
    (!defined(__HIP_DEVICE_COMPILE__) || defined(__gfx1200__) || \
     defined(__gfx1201__))
  #define RDNA_ATTN_UNIT_GFX1201(kernel) RDNA_ATTN_UNIT_OF(kernel, Gfx1201)
#else
  #define RDNA_ATTN_UNIT_GFX1201(kernel)
#endif

#define RDNA_ATTN_UNIT(kernel)   \
  RDNA_ATTN_UNIT_GFX1151(kernel) \
  RDNA_ATTN_UNIT_GFX1150(kernel) \
  RDNA_ATTN_UNIT_GFX1201(kernel)

#else  // !RDNA_ATTN_HEAD_DIM

#define RDNA_ATTN_UNIT(kernel)

#endif
