// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

// Host-side BMM frontends. These expose BMM/grouped-layout APIs while reusing
// the generated opus GEMM backend launcher symbols.
//
// Host pass only, like opus_gemm.cu. Every device symbol this module needs is
// codegen'd: the per-tile compute kernels into <tile>_C{void,bf16,fp32}.device.cu
// and opus_bmm_splitk_reduce_kernel<{__bf16,float}, 8, 128> into
// splitk_reduce_gfx950.device.cu (gen_instances_gfx950.py's splitk_reduce_extra
// hook), which owns both the device kernel and the host __device_stub__ the
// codegen'd split-K launchers reference.
#ifndef __HIP_DEVICE_COMPILE__

#include "opus_bmm.h"
#include "opus_gemm_arch.cuh"
#include "opus_build_archs.h"
#include "opus_gemm_manifest.h"
#include "opus_bmm_mxscale_kid_dispatch.h"
#include "opus_gemm_utils.cuh"  // bf16_t / fp32_t
#include "aiter_stream.h"
#include "gfx950/opus_bmm_launchers_a8w8_mxscale_gfx950.cuh"

#include <unordered_map>

#ifdef OPUS_BUILD_HAS_GFX950
namespace opus_bmm_detail {

// Uniform kid->launcher fn-pointer type. Every kid is codegen'd (no hand-written
// adapters), so this namespace only holds the shared type.
using OpusBmmMxscaleKernel = void (*)(
    aiter_tensor_t &, aiter_tensor_t &, aiter_tensor_t &,
    aiter_tensor_t &, aiter_tensor_t &,
    std::optional<aiter_tensor_t>, int /*split_k*/);
}  // namespace opus_bmm_detail

// Table-driven global exact-kid dispatch. Every registry BMM kid is generated;
// an unknown id is an error rather than a fallback to another kernel.
//
// Throwing beats falling back, and that is worth stating because the fallback
// looks harmless: adding a codegen tag means three hand-maintained lists in
// gen_instances.py on top of the five gfx950 maps, and a kid missed by any of
// them would quietly run another kid's tile, pass the correctness gate, and
// time identically to its siblings. That is exactly how five different blds
// tiles once measured within 0.1% of one another (see the kid224/225 note in
// opus_gemm_common.py).
static opus_bmm_detail::OpusBmmMxscaleKernel
opus_bmm_a8w8_mxscale_exact_dispatch(int kid)
{
  using namespace opus_bmm_detail;
  static const std::unordered_map<int, OpusBmmMxscaleKernel> kDispatch = {
      GENERATE_BMM_MXSCALE_KID_DISPATCH(fp32_t)
  };
  auto it = kDispatch.find(kid);
  AITER_CHECK(it != kDispatch.end(),
              "opus_bmm_a8w8_mxscale: kid ", kid, " has no emitted instance. It is "
              "either not in the catalog or its kernel_tag is missing from one of "
              "the codegen maps; rebuild with AITER_REBUILD=1 after a codegen edit");
  return it->second;
}
#endif  // OPUS_BUILD_HAS_GFX950

void opus_gemm_a8w8_mxscale_bmm_launch(
    aiter_tensor_t &O,
    aiter_tensor_t &wo_a,
    aiter_tensor_t &Y,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    std::optional<aiter_tensor_t> workspace,
    int kid,
    int split_k)
{
  // Common dtype/shape validation + arch gate, done once here so the codegen'd
  // launchers (which omit these to stay lean) and the fused kid 100 wrapper share
  // one check. The _impl still re-checks internally (idempotent).
  opus_bmm_a8w8_common_checks(O, wo_a, Y, x_scale, w_scale,
                              "opus_gemm_a8w8_mxscale_bmm_launch");
#ifndef OPUS_BUILD_HAS_GFX950
  AITER_CHECK(false,
              "opus_gemm_a8w8_mxscale_bmm_launch requires "
              "OPUS_BUILD_HAS_GFX950");
#else
  {
    const auto &arch_info = opus_get_arch_info();
    AITER_CHECK(arch_info.arch == OpusGfxArch::Gfx950,
                "opus_gemm_a8w8_mxscale_bmm_launch is gfx950-only; "
                "current device ", arch_info.dev, " has gcnArchName='",
                arch_info.name, "'");
  }
  opus_bmm_a8w8_mxscale_exact_dispatch(kid)(
      O, wo_a, Y, x_scale, w_scale, workspace, split_k);
#endif  // OPUS_BUILD_HAS_GFX950
}

#endif  // !__HIP_DEVICE_COMPILE__
