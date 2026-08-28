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
#include "opus_bmm_mxscale_tune_lookup.h"  // GENERATE_BMM_MXSCALE_FLATMM_SPLITK_LOOKUP_FP32
#include "opus_gemm_utils.cuh"  // bf16_t / fp32_t
#include "aiter_stream.h"
#include "gfx950/opus_bmm_launchers_a8w8_mxscale_gfx950.cuh"
#ifdef OPUS_BUILD_HAS_GFX1250
// Only under the arch flag: the gfx1250 traits derive TDM window types that need
// clang >= 22, so pulling them into a gfx950-only build would make an unrelated
// toolchain a hard error for a path that build cannot reach anyway.
#include "gfx1250/opus_bmm_launchers_a8w8_mxscale_bpreshuffle_gfx1250.cuh"
#endif

#include <unordered_map>

#ifdef OPUS_BUILD_HAS_GFX950
namespace opus_bmm_detail {

// Uniform kid->launcher fn-pointer type. Every kid is codegen'd (no hand-written
// adapters), so this namespace only holds the shared type.
using OpusBmmMxscaleFlatmmSplitkKernel = void (*)(
    aiter_tensor_t &, aiter_tensor_t &, aiter_tensor_t &,
    aiter_tensor_t &, aiter_tensor_t &, int /*splitK*/);
}  // namespace opus_bmm_detail

// Table-driven kid -> launcher dispatch. Launchers come from the generated
// GENERATE_BMM_MXSCALE_FLATMM_SPLITK_LOOKUP_FP32 macro; unknown / untuned kids
// fall back to the 32x128x128 wg2 baseline.
static opus_bmm_detail::OpusBmmMxscaleFlatmmSplitkKernel
opus_bmm_a8w8_mxscale_tune_dispatch(int id)
{
  using namespace opus_bmm_detail;
  static const std::unordered_map<int, OpusBmmMxscaleFlatmmSplitkKernel> kTune = {
      GENERATE_BMM_MXSCALE_FLATMM_SPLITK_LOOKUP_FP32(fp32_t)
  };
  auto it = kTune.find(id);
  if (it != kTune.end())
    return it->second;
  return &opus_bmm_a8w8_mxscale_flatmm_splitk_256x32x128x128_2x1_16x16x128_1x128x128_wgpcu2<fp32_t>;
}
#endif  // OPUS_BUILD_HAS_GFX950

void opus_bmm_a8w8_mxscale(
    aiter_tensor_t &O,
    aiter_tensor_t &wo_a,
    aiter_tensor_t &Y,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    int splitK,
    int kernelId)
{
  // Common dtype/shape validation + arch gate, done once here so the codegen'd
  // launchers (which omit these to stay lean) and the fused kid 100 wrapper share
  // one check. The _impl still re-checks internally (idempotent).
  opus_bmm_a8w8_common_checks(O, wo_a, Y,
                              "opus_bmm_a8w8_mxscale");
#ifndef OPUS_BUILD_HAS_GFX950
  AITER_CHECK(false,
              "opus_bmm_a8w8_mxscale requires "
              "OPUS_BUILD_HAS_GFX950");
#else
  {
    const auto &arch_info = opus_get_arch_info();
    AITER_CHECK(arch_info.arch == OpusGfxArch::Gfx950,
                "opus_bmm_a8w8_mxscale is gfx950-only; "
                "current device ", arch_info.dev, " has gcnArchName='",
                arch_info.name, "'");
  }
  // Single table lookup instead of a ~40-case switch (see opus_gemm.cu).
  opus_bmm_a8w8_mxscale_tune_dispatch(kernelId)(
      O, wo_a, Y, x_scale, w_scale, splitK);
#endif  // OPUS_BUILD_HAS_GFX950
}

// gfx1250 (MI450) sibling of the above, for a PRESHUFFLED B. Separate entry
// point rather than a kernelId range on opus_bmm_a8w8_mxscale, because the two
// take a different wo_a: this one wants shuffle_weight(w, (16,16)), and feeding
// a row-major weight to it is not a shape error -- it reads in bounds and
// returns wrong numbers. A distinct symbol makes that a call-site decision.
//
// The kernel itself is instantiated in
// opus_bmm_a8w8_mxscale_bpreshuffle_gfx1250.cu (one hand-written tile, no
// codegen kid yet); the launcher header only forward-declares it, so this TU
// stays host-pass-only like the rest of the file.
void opus_bmm_a8w8_mxscale_bpreshuffle(
    aiter_tensor_t &O,
    aiter_tensor_t &wo_a,
    aiter_tensor_t &Y,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    int splitK,
    int kernelId)
{
#ifndef OPUS_BUILD_HAS_GFX1250
  (void)O; (void)wo_a; (void)Y; (void)x_scale; (void)w_scale;
  (void)splitK; (void)kernelId;
  aiter_detail::g_aiter_can_throw = true;
  AITER_CHECK(false,
              "opus_bmm_a8w8_mxscale_bpreshuffle requires "
              "OPUS_BUILD_HAS_GFX1250 (add gfx1250 to GPU_ARCHS)");
#else
  {
    const auto &arch_info = opus_get_arch_info();
    AITER_CHECK(arch_info.arch == OpusGfxArch::Gfx1250,
                "opus_bmm_a8w8_mxscale_bpreshuffle is gfx1250-only; "
                "current device ", arch_info.dev, " has gcnArchName='",
                arch_info.name, "'");
  }
  // One tile so far, so kernelId only ever selects the output dtype path. It
  // stays in the signature because the codegen follow-up turns it into a real
  // kid lookup shaped like opus_bmm_a8w8_mxscale_tune_dispatch above, and
  // changing the ABI later is worse than carrying an unused argument now.
  (void)kernelId;
  if (Y.dtype() == AITER_DTYPE_bf16)
    opus_bmm_a8w8_mxscale_bpreshuffle_launch_gfx1250<
        opus_bmm_a8w8_mxscale_bpreshuffle_tile_gfx1250<bf16_t>>(
        O, wo_a, Y, x_scale, w_scale, splitK);
  else
    opus_bmm_a8w8_mxscale_bpreshuffle_launch_gfx1250<
        opus_bmm_a8w8_mxscale_bpreshuffle_tile_gfx1250<fp32_t>>(
        O, wo_a, Y, x_scale, w_scale, splitK);
#endif  // OPUS_BUILD_HAS_GFX1250
}

#endif  // !__HIP_DEVICE_COMPILE__
