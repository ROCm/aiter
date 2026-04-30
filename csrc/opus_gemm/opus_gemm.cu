// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

// opus_gemm.cu is the host-side dispatcher (lookup table + heuristic
// fallback). It contains no __global__ functions and no `<<<>>>` launches,
// so the device pass has nothing to codegen but still pays the full
// libtorch + HIP runtime + opus.hpp parse (~15s/TU). Skipping the entire
// translation unit on the device pass makes it essentially free, saving
// a second-bottleneck TU's worth of wall time on every rebuild.
//
// Host pass is also torch-free now: the entry points take aiter_tensor_t
// (POD, defined in csrc/include/aiter_tensor.h) instead of torch::Tensor,
// mirroring the refactor in PR #2932 (csrc/kernels/quant_kernels.cu).
// Eliminating <torch/all.h> + <ATen/...> from this TU drops its host-pass
// preprocessed lines from ~440K to ~110K, which feeds straight into a
// shorter critical path in the fused codegen layout (see
// aiter/ops/opus/README.md §7.6).
#ifndef __HIP_DEVICE_COMPILE__

#include "opus_gemm_arch.cuh"                      // OpusGfxArch + opus_get_arch_info / opus_get_gfx_arch
#include "gfx950/opus_gemm_arch_gfx950.cuh"        // opus_dispatch_a16w16_gfx950<T> / opus_a16w16_tune_dispatch_gfx950<T>
#include "opus_gemm_common.cuh"
#include "gfx950/opus_gemm_heuristic_dispatch_gfx950.cuh"  // OpusA16W16NoscaleKernel
#include "opus_gemm_manifest.h"                    // a8w8 launcher symbols
#include "opus_gemm_utils.cuh"                     // bf16_t / fp32_t

#include <functional>
#include <optional>
#include <string>

// ── a8w8 / a8w8_scale launcher signatures ───────────────────────────────────
//
// a8w8 paths bypass the arch-routed dispatcher because there's currently a
// single hardcoded launcher per dtype (no tuned lookup table). The fp8 entry
// in opus_gemm() guards them with an explicit gfx950 AITER_CHECK so callers
// on other archs see the same "pipeline TBD" error as the bf16 path.
using OpusScaleKernel = std::function<
    void(aiter_tensor_t &, aiter_tensor_t &,
         aiter_tensor_t &,
         std::optional<aiter_tensor_t>, std::optional<aiter_tensor_t>)>;

using OpusNoscaleKernel = std::function<
    void(aiter_tensor_t &, aiter_tensor_t &,
         aiter_tensor_t &)>;

template <typename CDataType>
OpusScaleKernel opus_dispatch_scale(int M, int N, int K)
{
  return opus_gemm_512x256x256x128_4x2_16x16x128_1x128x128<CDataType>;
}

template <typename CDataType>
OpusNoscaleKernel opus_dispatch_a8w8(int M, int N, int K)
{
  return opus_gemm_512x256x256x128_2x4_16x16x128_0x0x0<CDataType>;
}

// ── a16w16 arch routers ─────────────────────────────────────────────────────
//
// Both routers share the same shape: switch on opus_get_gfx_arch() and
// dispatch to the per-arch dispatch function for that arch. Adding a new
// arch is local to opus_gemm_arch.cuh + a new opus_gemm_arch_<arch>.cuh +
// one extra `case` in each router below.

template <typename CDataType>
OpusA16W16NoscaleKernel opus_dispatch_a16w16(int M, int N, int K, int batch)
{
  switch (opus_get_gfx_arch())
  {
    case OpusGfxArch::Gfx950:
      return opus_dispatch_a16w16_gfx950<CDataType>(M, N, K, batch);
    // future: case OpusGfxArch::Gfx942: return opus_dispatch_a16w16_gfx942<CDataType>(M, N, K, batch);
    default:
    {
      const auto &info = opus_get_arch_info();
      AITER_CHECK(false,
                  "opus_gemm: a16w16 dispatch is only implemented for gfx950 today; "
                  "current device ", info.dev,
                  " has gcnArchName='", info.name,
                  "'. Other archs (gfx940 / gfx942 / gfx1100 / ...) will be added "
                  "as more pipelines land.");
    }
  }
}

template <typename CDataType>
opus_gfx950_detail::OpusA16W16TuneKernel
opus_a16w16_tune_dispatch(int id)
{
  switch (opus_get_gfx_arch())
  {
    case OpusGfxArch::Gfx950:
      return opus_a16w16_tune_dispatch_gfx950<CDataType>(id);
    // future: case OpusGfxArch::Gfx942: return opus_a16w16_tune_dispatch_gfx942<CDataType>(id);
    default:
    {
      const auto &info = opus_get_arch_info();
      AITER_CHECK(false,
                  "opus_gemm_a16w16_tune: dispatch is only implemented for gfx950 today; "
                  "current device ", info.dev,
                  " has gcnArchName='", info.name,
                  "'. Other archs will be added as more pipelines land.");
    }
  }
}

// ── opus_gemm() — top-level a16w16 / a8w8 entry ─────────────────────────────

void opus_gemm(
  aiter_tensor_t &XQ,
  aiter_tensor_t &WQ,
  aiter_tensor_t &Y,
  std::optional<aiter_tensor_t> group_layout,
  std::optional<aiter_tensor_t> x_scale,
  std::optional<aiter_tensor_t> w_scale,
  std::optional<aiter_tensor_t> bias)
{
  AITER_CHECK(XQ.dim() == 3, "XQ must be 3D [batch, M, K]");
  AITER_CHECK(WQ.dim() == 3, "WQ must be 3D [batch, N, K]");
  AITER_CHECK(Y.dim() == 3, "Y must be 3D [batch, M, N]");

  int M = XQ.size(1);
  int N = WQ.size(1);
  int K = XQ.size(2);

  bool has_scale = x_scale.has_value() && w_scale.has_value();

  if (XQ.dtype() == AITER_DTYPE_fp8)
  {
    // a8w8 / a8w8_scale launchers are gfx950-only today and don't yet flow
    // through the arch-routed dispatcher (they pick a single hardcoded
    // launcher). Guard the entry explicitly so non-gfx950 callers see the
    // same "pipeline TBD for this arch" message as the bf16 path.
    const auto &arch_info = opus_get_arch_info();
    AITER_CHECK(arch_info.arch == OpusGfxArch::Gfx950,
                "opus_gemm: a8w8 path is only implemented for gfx950 today; "
                "current device ", arch_info.dev,
                " has gcnArchName='", arch_info.name,
                "'. Other archs will be added as more pipelines land.");
    // a8w8 / a8w8_scale launchers do not consume bias yet; reject up front
    // rather than silently dropping it.
    AITER_CHECK(!bias.has_value(),
                "opus_gemm: bias is not supported on a8w8 / a8w8_scale paths");
    if (has_scale)
    {
      AITER_CHECK(Y.dtype() == AITER_DTYPE_fp32,
                  "opus_gemm a8w8_scale only supports fp32 output");
      opus_dispatch_scale<fp32_t>(M, N, K)(XQ, WQ, Y, x_scale, w_scale);
    }
    else
    {
      AITER_CHECK(Y.dtype() == AITER_DTYPE_fp32,
                  "opus_gemm a8w8 no-scale only supports fp32 output");
      opus_dispatch_a8w8<fp32_t>(M, N, K)(XQ, WQ, Y);
    }
  }
  else if (XQ.dtype() == AITER_DTYPE_bf16)
  {
    // Two-level dispatch: tuned CSV lookup (baked in at JIT time) first,
    // heuristic if-else tree on miss. splitK is passed 0; splitk kids
    // auto-clamp to pfk anyway so no extra information is needed at this
    // entry point. Kernels that ignore splitK (a16w16 / flatmm) just
    // drop it.
    //
    // Bias routing
    // ------------
    // bias is forwarded to whichever launcher the lookup / heuristic
    // picks. The lookup map only contains bias-aware kids today
    // (a16w16_flatmm_kernels_list is empty -- see opus_gemm_common.py;
    // only split-barrier 4..9 and a16w16_flatmm_splitk 200..299 ever
    // appear), so a non-empty bias is always safe at this entry.
    int batch = XQ.size(0);
    if (Y.dtype() == AITER_DTYPE_bf16)
    {
      opus_dispatch_a16w16<bf16_t>(M, N, K, batch)(XQ, WQ, Y, bias, 0);
    }
    else if (Y.dtype() == AITER_DTYPE_fp32)
    {
      opus_dispatch_a16w16<fp32_t>(M, N, K, batch)(XQ, WQ, Y, bias, 0);
    }
    else
    {
      AITER_CHECK(false, "opus_gemm a16w16: unsupported output dtype, expected bf16 or fp32");
    }
  }
  else
  {
    AITER_CHECK(false, "opus_gemm: unsupported input dtype, expected fp8 or bf16");
  }
}

// ── opus_gemm_a16w16_tune() — id-based tune entry ───────────────────────────
//
// Launcher signature is 4-arg (XQ, WQ, Y, int splitK); all three a16w16-family
// kernels populate the same tune lookup map:
//   * split-barrier a16w16 (kids 4..9)      - ignores splitK
//   * a16w16_flatmm      (kids 100..115)    - ignores splitK
//   * a16w16_flatmm_splitk (kids 200..210)  - interprets splitK as literal KBatch
//
// splitk kids require D_C=fp32 (main kernel writes an fp32 workspace; the
// reduce kernel D_OUT is templated on Y.dtype() and chosen at launch time),
// so the dispatcher forces the <fp32_t> branch for kids >= 200 regardless of
// Y dtype. Both bf16 and fp32 Y are valid.

// splitk kids live in [200, 300). Kept as a range so future splitk tiles don't
// require a dispatcher change.
static constexpr int OPUS_SPLITK_KID_MIN = 200;
static constexpr int OPUS_SPLITK_KID_MAX = 300;
// Split-barrier a16w16 kids live in [4, 10). Used to gate which kids accept
// a non-empty bias (a16w16_flatmm 100..115 keeps HAS_BIAS=false until its
// warp-spec epilogue gets bias support).
static constexpr int OPUS_A16W16_SB_KID_MIN = 4;
static constexpr int OPUS_A16W16_SB_KID_MAX = 10;

static inline bool opus_kid_supports_bias(int kid)
{
  return (kid >= OPUS_A16W16_SB_KID_MIN && kid < OPUS_A16W16_SB_KID_MAX) ||
         (kid >= OPUS_SPLITK_KID_MIN && kid < OPUS_SPLITK_KID_MAX);
}

void opus_gemm_a16w16_tune(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &Y,
    std::optional<aiter_tensor_t> bias,
    int kernelId,
    int splitK)
{
  AITER_CHECK(XQ.dim() == 3, "XQ must be 3D [batch, M, K]");
  AITER_CHECK(WQ.dim() == 3, "WQ must be 3D [batch, N, K]");
  AITER_CHECK(Y.dim() == 3, "Y must be 3D [batch, M, N]");
  AITER_CHECK(XQ.dtype() == WQ.dtype(),
              "XQ and WQ should have the same dtype!");
  // Gate non-bias-capable kids early. The launcher itself will also do the
  // detailed shape/dtype check on a non-empty bias; this guard just gives a
  // clear "wrong kid" error before we even enter the launcher.
  AITER_CHECK(!bias.has_value() || opus_kid_supports_bias(kernelId),
              "opus_gemm_a16w16_tune: bias is currently only supported on "
              "a16w16 split-barrier kids [", OPUS_A16W16_SB_KID_MIN, ", ",
              OPUS_A16W16_SB_KID_MAX, ") or a16w16_flatmm_splitk kids [",
              OPUS_SPLITK_KID_MIN, ", ", OPUS_SPLITK_KID_MAX,
              "); got kid=", kernelId);

  if (XQ.dtype() == AITER_DTYPE_bf16)
  {
    // splitk kids force <fp32_t> because traits static_assert D_C=float.
    // Y can be bf16 or fp32 -- the launcher dispatches the reduce kernel
    // on Y.dtype() at runtime.
    if (kernelId >= OPUS_SPLITK_KID_MIN && kernelId < OPUS_SPLITK_KID_MAX)
    {
      AITER_CHECK(Y.dtype() == AITER_DTYPE_bf16
                  || Y.dtype() == AITER_DTYPE_fp32,
                  "opus_gemm_a16w16_tune splitk kid requires bf16 or fp32 Y "
                  "(reduce kernel writes the correct dtype)");
      opus_a16w16_tune_dispatch<fp32_t>(kernelId)(XQ, WQ, Y, bias, splitK);
    }
    else if (Y.dtype() == AITER_DTYPE_bf16)
    {
      opus_a16w16_tune_dispatch<bf16_t>(kernelId)(XQ, WQ, Y, bias, splitK);
    }
    else if (Y.dtype() == AITER_DTYPE_fp32)
    {
      opus_a16w16_tune_dispatch<fp32_t>(kernelId)(XQ, WQ, Y, bias, splitK);
    }
    else
    {
      AITER_CHECK(false,
                  "opus_gemm_a16w16_tune: unsupported output dtype, expected bf16 or fp32");
    }
  }
  else
  {
    AITER_CHECK(false,
                "opus_gemm_a16w16_tune: unsupported input dtype ",
                AiterDtype_to_str(XQ.dtype()),
                ", expected bf16");
  }
}

#endif // !__HIP_DEVICE_COMPILE__
