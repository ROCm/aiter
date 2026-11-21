// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

// Host-side family routers and strict exact-kid dispatch.
#ifndef __HIP_DEVICE_COMPILE__

#include "opus_gemm_arch.cuh"                      // OpusGfxArch + opus_get_arch_info / opus_get_gfx_arch
#include "opus_build_archs.h"                      // OPUS_BUILD_HAS_GFX942 / OPUS_BUILD_HAS_GFX950
#ifdef OPUS_BUILD_HAS_GFX950
#include "gfx950/opus_gemm_arch_gfx950.cuh"        // generated gfx950 a16w16 kid dispatch
#endif
#ifdef OPUS_BUILD_HAS_GFX942
#include "gfx942/opus_gemm_arch_gfx942.cuh"        // generated gfx942 a16w16 kid dispatch
#endif
#ifdef OPUS_BUILD_HAS_GFX1250
#include "gfx1250/opus_gemm_arch_gfx1250.cuh"      // generated gfx1250 exact-kid dispatch
#endif
#include "opus_gemm_common.cuh"
#include "opus_gemm_manifest.h"                    // a8w8 launcher symbols
#include "opus_gemm_utils.cuh"                     // bf16_t / fp32_t

#include <optional>

#ifndef OPUS_A8W8_DISPATCH_KERNEL_TYPES_DEFINED
#define OPUS_A8W8_DISPATCH_KERNEL_TYPES_DEFINED
using OpusA8W8Kernel = void (*)(
    aiter_tensor_t&, aiter_tensor_t&, aiter_tensor_t&);
using OpusA8W8BlockscaleKernel = void (*)(
    aiter_tensor_t&, aiter_tensor_t&, aiter_tensor_t&,
    aiter_tensor_t&, aiter_tensor_t&);
using OpusA8W8BlockscaleBpreshuffleKernel = void (*)(
    aiter_tensor_t&, aiter_tensor_t&, aiter_tensor_t&,
    aiter_tensor_t&, aiter_tensor_t&);
#endif

static OpusA8W8Kernel opus_a8w8_kid_dispatch(int kid)
{
  const auto &info = opus_get_arch_info();
  switch (info.arch)
  {
    case OpusGfxArch::Gfx950:
#ifdef OPUS_BUILD_HAS_GFX950
      return opus_a8w8_kid_dispatch_gfx950(kid);
#else
      AITER_CHECK(false,
                  "opus_gemm_a8w8_launch: module was not built with gfx950 ",
                  "support for current device ", info.dev);
#endif
      return nullptr;
    case OpusGfxArch::Gfx942:
#ifdef OPUS_BUILD_HAS_GFX942
      AITER_CHECK(false,
                  "no registered kernel for OPUS a8w8 on gfx942");
#else
      AITER_CHECK(false,
                  "opus_gemm_a8w8_launch: module was not built with gfx942 ",
                  "support for current device ", info.dev);
#endif
      return nullptr;
    case OpusGfxArch::Gfx1250:
#ifdef OPUS_BUILD_HAS_GFX1250
      AITER_CHECK(false,
                  "no registered kernel for OPUS a8w8 on gfx1250");
#else
      AITER_CHECK(false,
                  "opus_gemm_a8w8_launch: module was not built with gfx1250 ",
                  "support for current device ", info.dev);
#endif
      return nullptr;
    default:
      AITER_CHECK(false,
                  "opus_gemm_a8w8_launch: unsupported current device ",
                  info.dev, " with gcnArchName='", info.name, "'");
  }
  return nullptr;
}

static OpusA8W8BlockscaleKernel
opus_a8w8_blockscale_kid_dispatch(int kid)
{
  const auto &info = opus_get_arch_info();
  switch (info.arch)
  {
    case OpusGfxArch::Gfx950:
#ifdef OPUS_BUILD_HAS_GFX950
      return opus_a8w8_blockscale_kid_dispatch_gfx950(kid);
#else
      AITER_CHECK(false,
                  "opus_gemm_a8w8_blockscale_launch: module was not built ",
                  "with gfx950 support for current device ", info.dev);
#endif
      return nullptr;
    case OpusGfxArch::Gfx942:
#ifdef OPUS_BUILD_HAS_GFX942
      AITER_CHECK(false,
                  "no registered kernel for OPUS a8w8_blockscale on gfx942");
#else
      AITER_CHECK(false,
                  "opus_gemm_a8w8_blockscale_launch: module was not built ",
                  "with gfx942 support for current device ", info.dev);
#endif
      return nullptr;
    case OpusGfxArch::Gfx1250:
#ifdef OPUS_BUILD_HAS_GFX1250
      AITER_CHECK(false,
                  "no registered kernel for OPUS a8w8_blockscale on gfx1250");
#else
      AITER_CHECK(false,
                  "opus_gemm_a8w8_blockscale_launch: module was not built ",
                  "with gfx1250 support for current device ", info.dev);
#endif
      return nullptr;
    default:
      AITER_CHECK(false,
                  "opus_gemm_a8w8_blockscale_launch: unsupported current device ",
                  info.dev, " with gcnArchName='", info.name, "'");
  }
  return nullptr;
}

template <typename CDataType>
static OpusA8W8BlockscaleBpreshuffleKernel
opus_a8w8_blockscale_bpreshuffle_kid_dispatch(int kid)
{
  const auto &info = opus_get_arch_info();
  switch (info.arch)
  {
    case OpusGfxArch::Gfx950:
#ifdef OPUS_BUILD_HAS_GFX950
      return opus_a8w8_blockscale_bpreshuffle_kid_dispatch_gfx950<CDataType>(kid);
#else
      AITER_CHECK(false,
                  "opus_gemm_a8w8_blockscale_bpreshuffle_launch: module was ",
                  "not built with gfx950 support for current device ", info.dev);
#endif
      return nullptr;
    case OpusGfxArch::Gfx942:
#ifdef OPUS_BUILD_HAS_GFX942
      return opus_a8w8_blockscale_bpreshuffle_kid_dispatch_gfx942<CDataType>(kid);
#else
      AITER_CHECK(false,
                  "opus_gemm_a8w8_blockscale_bpreshuffle_launch: module was ",
                  "not built with gfx942 support for current device ", info.dev);
#endif
      return nullptr;
    case OpusGfxArch::Gfx1250:
#ifdef OPUS_BUILD_HAS_GFX1250
      return opus_a8w8_blockscale_bpreshuffle_kid_dispatch_gfx1250<CDataType>(kid);
#else
      AITER_CHECK(false,
                  "opus_gemm_a8w8_blockscale_bpreshuffle_launch: module was ",
                  "not built with gfx1250 support for current device ", info.dev);
#endif
      return nullptr;
    default:
      AITER_CHECK(false,
                  "opus_gemm_a8w8_blockscale_bpreshuffle_launch: unsupported ",
                  "current device ", info.dev, " with gcnArchName='",
                  info.name, "'");
  }
  return nullptr;
}

template <typename CDataType>
static OpusA16W16Kernel
opus_a16w16_kid_dispatch(int kid)
{
  switch (opus_get_gfx_arch())
  {
#ifdef OPUS_BUILD_HAS_GFX950
    case OpusGfxArch::Gfx950:
      return opus_a16w16_kid_dispatch_gfx950<CDataType>(kid);
#endif
#ifdef OPUS_BUILD_HAS_GFX942
    case OpusGfxArch::Gfx942:
      return opus_a16w16_kid_dispatch_gfx942<CDataType>(kid);
#endif
#ifdef OPUS_BUILD_HAS_GFX1250
    case OpusGfxArch::Gfx1250:
      return opus_a16w16_kid_dispatch_gfx1250<CDataType>(kid);
#endif
    default:
    {
      const auto &info = opus_get_arch_info();
      AITER_CHECK(false,
                  "opus_gemm_a16w16_launch: no non-workspace dispatch table for "
                  "current device ", info.dev,
                  " with gcnArchName='", info.name, "'");
      return nullptr;
    }
  }
}

// Query the direct table separately so an unavailable workspace kid is not
// misclassified as direct.
static bool opus_a16w16_has_non_workspace_kernel(int kid)
{
  switch (opus_get_gfx_arch())
  {
#ifdef OPUS_BUILD_HAS_GFX950
    case OpusGfxArch::Gfx950:
      return opus_a16w16_has_non_workspace_kernel_gfx950(kid);
#endif
#ifdef OPUS_BUILD_HAS_GFX942
    case OpusGfxArch::Gfx942:
      return opus_a16w16_has_non_workspace_kernel_gfx942(kid);
#endif
#ifdef OPUS_BUILD_HAS_GFX1250
    case OpusGfxArch::Gfx1250:
      return opus_a16w16_has_non_workspace_kernel_gfx1250(kid);
#endif
    default:
    {
      const auto &info = opus_get_arch_info();
      AITER_CHECK(false,
                  "opus_gemm_a16w16_launch: no non-workspace dispatch table for "
                  "current device ", info.dev,
                  " with gcnArchName='", info.name, "'");
      return false;
    }
  }
}

// Query the current architecture's generated workspace table.
static bool opus_a16w16_has_workspace_kernel(int kid)
{
  switch (opus_get_gfx_arch())
  {
#ifdef OPUS_BUILD_HAS_GFX950
    case OpusGfxArch::Gfx950:
      return opus_a16w16_has_workspace_kernel_gfx950(kid);
#endif
#ifdef OPUS_BUILD_HAS_GFX942
    case OpusGfxArch::Gfx942:
      return opus_a16w16_has_workspace_kernel_gfx942(kid);
#endif
#ifdef OPUS_BUILD_HAS_GFX1250
    case OpusGfxArch::Gfx1250:
      return opus_a16w16_has_workspace_kernel_gfx1250(kid);
#endif
    default:
    {
      const auto &info = opus_get_arch_info();
      AITER_CHECK(false,
                  "opus_gemm_a16w16_launch: no workspace dispatch table for device ",
                  info.dev, " with gcnArchName='", info.name, "'");
      return false;
    }
  }
}
static OpusA16W16WorkspaceKernel
opus_a16w16_workspace_dispatch(int kid)
{
  switch (opus_get_gfx_arch())
  {
#ifdef OPUS_BUILD_HAS_GFX950
    case OpusGfxArch::Gfx950:
      return opus_a16w16_workspace_dispatch_gfx950(kid);
#endif
#ifdef OPUS_BUILD_HAS_GFX942
    case OpusGfxArch::Gfx942:
      return opus_a16w16_workspace_dispatch_gfx942(kid);
#endif
#ifdef OPUS_BUILD_HAS_GFX1250
    case OpusGfxArch::Gfx1250:
      return opus_a16w16_workspace_dispatch_gfx1250(kid);
#endif
    default:
    {
      const auto &info = opus_get_arch_info();
      AITER_CHECK(false,
                  "opus_gemm_a16w16_launch: no workspace dispatch table for device ",
                  info.dev, " with gcnArchName='", info.name, "'");
      return nullptr;
    }
  }
}

// Validate A16W16 inputs, then call the matching generated launcher table.
static void opus_gemm_a16w16_launch_impl(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &Y,
    std::optional<aiter_tensor_t> bias,
    std::optional<aiter_tensor_t> workspace,
    int kid,
    int split_k)
{
  aiter_detail::g_aiter_can_throw = true;

  AITER_CHECK(XQ.is_gpu() && WQ.is_gpu() && Y.is_gpu(),
              "opus_gemm_a16w16_launch: XQ, WQ, and Y must be GPU tensors");
  AITER_CHECK(XQ.device_id == WQ.device_id && XQ.device_id == Y.device_id,
              "opus_gemm_a16w16_launch: XQ/WQ/Y device ids must match (got ",
              XQ.device_id, "/", WQ.device_id, "/", Y.device_id, ")");
  if (bias.has_value())
  {
    AITER_CHECK(bias->is_gpu() && bias->device_id == XQ.device_id,
                "opus_gemm_a16w16_launch: bias device ", bias->device_id,
                " must match XQ device ", XQ.device_id);
  }
  if (workspace.has_value())
  {
    AITER_CHECK(workspace->is_gpu() && workspace->device_id == XQ.device_id,
                "opus_gemm_a16w16_launch: workspace device ",
                workspace->device_id, " must match input device ", XQ.device_id);
  }
  AITER_CHECK(XQ.dim() == 3,
              "opus_gemm_a16w16_launch: XQ must be 3D [batch, M, K]");
  AITER_CHECK(WQ.dim() == 3,
              "opus_gemm_a16w16_launch: WQ must be 3D [batch, N, K]");
  AITER_CHECK(Y.dim() == 3,
              "opus_gemm_a16w16_launch: Y must be 3D [batch, M, N]");
  AITER_CHECK(XQ.dtype() == WQ.dtype(),
              "opus_gemm_a16w16_launch: XQ and WQ dtype must match");

  if (XQ.dtype() == AITER_DTYPE_bf16)
  {
    const bool uses_workspace = opus_a16w16_has_workspace_kernel(kid);
    const bool has_non_workspace = opus_a16w16_has_non_workspace_kernel(kid);
    AITER_CHECK(!(uses_workspace && has_non_workspace),
                "opus_gemm_a16w16_launch: kid ", kid,
                " appears in both workspace and non-workspace launch tables");
    if (!uses_workspace && !has_non_workspace)
    {
      const auto &info = opus_get_arch_info();
      AITER_CHECK(false,
                  "opus_gemm_a16w16_launch: kid ", kid,
                  " is not compiled in the current OPUS module for device ",
                  info.dev, " with gcnArchName='", info.name, "'");
    }
    if (uses_workspace)
    {
      AITER_CHECK(workspace.has_value(),
                  "opus_gemm_a16w16_launch: workspace kid ", kid,
                  " requires a workspace tensor");
      AITER_CHECK(Y.dtype() == AITER_DTYPE_bf16
                  || Y.dtype() == AITER_DTYPE_fp32,
                  "opus_gemm_a16w16_launch: workspace kid requires bf16 or fp32 Y "
                  "(reduce kernel writes the correct dtype)");
      opus_a16w16_workspace_dispatch(kid)(
          XQ, WQ, Y, workspace.value(), bias, split_k);
    }
    else
    {
      AITER_CHECK(!workspace.has_value(),
                  "opus_gemm_a16w16_launch: non-workspace kid ", kid,
                  " requires workspace=None");
      if (Y.dtype() == AITER_DTYPE_bf16)
      {
        opus_a16w16_kid_dispatch<bf16_t>(kid)(XQ, WQ, Y, bias, split_k);
      }
      else if (Y.dtype() == AITER_DTYPE_fp32)
      {
        opus_a16w16_kid_dispatch<fp32_t>(kid)(XQ, WQ, Y, bias, split_k);
      }
      else
      {
        AITER_CHECK(false,
                    "opus_gemm_a16w16_launch: unsupported output dtype, expected bf16 or fp32");
      }
    }
  }
  else
  {
    AITER_CHECK(false,
                "opus_gemm_a16w16_launch: unsupported input dtype ",
                AiterDtype_to_str(XQ.dtype()),
                ", expected bf16");
  }
}

void opus_gemm_a16w16_launch(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &Y,
    std::optional<aiter_tensor_t> bias,
    std::optional<aiter_tensor_t> workspace,
    int kid,
    int split_k)
{
  opus_gemm_a16w16_launch_impl(
      XQ, WQ, Y, bias, workspace, kid, split_k);
}

static void opus_check_a8_family_tensors(
    const char* entry,
    const aiter_tensor_t &XQ,
    const aiter_tensor_t &WQ,
    const aiter_tensor_t &Y,
    AiterDtype ab_dtype = AITER_DTYPE_fp8)
{
  AITER_CHECK(XQ.is_gpu() && WQ.is_gpu() && Y.is_gpu(),
              entry, ": XQ, WQ, and Y must be GPU tensors");
  AITER_CHECK(XQ.device_id == WQ.device_id && XQ.device_id == Y.device_id,
              entry, ": XQ/WQ/Y device ids must match (got ",
              XQ.device_id, "/", WQ.device_id, "/", Y.device_id, ")");
  int current_device = -1;
  HIP_CALL(hipGetDevice(&current_device));
  AITER_CHECK(current_device == XQ.device_id,
              entry, ": current HIP device ", current_device,
              " does not match tensor device ", XQ.device_id);
  AITER_CHECK(XQ.dtype() == ab_dtype && WQ.dtype() == ab_dtype,
              entry, ": expected ", AiterDtype_to_str(ab_dtype), " XQ/WQ, got ",
              AiterDtype_to_str(XQ.dtype()), "/",
              AiterDtype_to_str(WQ.dtype()));
}

// A8W8 entry points perform common checks before exact-kid table lookup.
static void opus_check_a8_scale_devices(
    const char* entry,
    const aiter_tensor_t &XQ,
    const aiter_tensor_t &x_scale,
    const aiter_tensor_t &w_scale)
{
  AITER_CHECK(x_scale.is_gpu() && w_scale.is_gpu(),
              entry, ": x_scale and w_scale must be GPU tensors");
  AITER_CHECK(x_scale.device_id == XQ.device_id &&
                  w_scale.device_id == XQ.device_id,
              entry, ": scale tensor device ids must match XQ.device_id=",
              XQ.device_id, " (got ", x_scale.device_id, "/",
              w_scale.device_id, ")");
}

void opus_gemm_a8w8_launch(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &Y,
    int kid)
{
  aiter_detail::g_aiter_can_throw = true;
  constexpr const char* entry = "opus_gemm_a8w8_launch";
  opus_check_a8_family_tensors(entry, XQ, WQ, Y);
  AITER_CHECK(Y.dtype() == AITER_DTYPE_fp32,
              entry, ": expected fp32 Y, got ",
              AiterDtype_to_str(Y.dtype()));
  opus_a8w8_kid_dispatch(kid)(XQ, WQ, Y);
}

void opus_gemm_a8w8_blockscale_launch(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &Y,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    int kid)
{
  aiter_detail::g_aiter_can_throw = true;
  constexpr const char* entry = "opus_gemm_a8w8_blockscale_launch";
  opus_check_a8_family_tensors(entry, XQ, WQ, Y);
  opus_check_a8_scale_devices(entry, XQ, x_scale, w_scale);
  AITER_CHECK(Y.dtype() == AITER_DTYPE_fp32,
              entry, ": expected fp32 Y, got ",
              AiterDtype_to_str(Y.dtype()));
  opus_a8w8_blockscale_kid_dispatch(kid)(
      XQ, WQ, Y, x_scale, w_scale);
}

void opus_gemm_a8w8_blockscale_bpreshuffle_launch(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y,
    int kid,
    std::optional<aiter_tensor_t> workspace)
{
  aiter_detail::g_aiter_can_throw = true;
  constexpr const char* entry =
      "opus_gemm_a8w8_blockscale_bpreshuffle_launch";
  opus_check_a8_family_tensors(entry, XQ, WQ, Y);
  opus_check_a8_scale_devices(entry, XQ, x_scale, w_scale);

#ifdef OPUS_BUILD_HAS_GFX950
  if (opus_get_gfx_arch() == OpusGfxArch::Gfx950 &&
      opus_a8w8_blockscale_bpreshuffle_has_workspace_kernel_gfx950(kid))
  {
    AITER_CHECK(workspace.has_value(),
                entry, ": workspace kid ", kid, " requires a workspace tensor");
    AITER_CHECK(Y.dtype() == AITER_DTYPE_bf16,
                entry, ": workspace kid requires bf16 Y, got ",
                AiterDtype_to_str(Y.dtype()));
    AITER_CHECK(workspace->is_gpu() && workspace->device_id == XQ.device_id,
                entry, ": workspace device must match XQ.device_id=", XQ.device_id);
    opus_a8w8_blockscale_bpreshuffle_workspace_kid_dispatch_gfx950(kid)(
        XQ, WQ, x_scale, w_scale, Y, workspace.value());
    return;
  }
#endif
  AITER_CHECK(!workspace.has_value(),
              entry, ": non-workspace kid ", kid, " requires workspace=None");

  if (Y.dtype() == AITER_DTYPE_bf16)
  {
    opus_a8w8_blockscale_bpreshuffle_kid_dispatch<bf16_t>(kid)(
        XQ, WQ, x_scale, w_scale, Y);
  }
  else if (Y.dtype() == AITER_DTYPE_fp32)
  {
    opus_a8w8_blockscale_bpreshuffle_kid_dispatch<fp32_t>(kid)(
        XQ, WQ, x_scale, w_scale, Y);
  }
  else
  {
    AITER_CHECK(false,
                entry, ": unsupported Y dtype ",
                AiterDtype_to_str(Y.dtype()), "; expected bf16 or fp32");
  }
}

// MXFP4 x MXFP4 NT (gfx950), layout="bpreshuffle": inputs as aiter gemm_a4w4(bpreshuffle=True).
// Shapes: XQ [batch, M, K/2], WQ [batch, N, K/2] (fp4x2, WQ in the
// shuffle_weight(16, 16) layout), Y [batch, M, N] bf16; N % 16, K % 256.
// Scales: e8m0_shuffle layout, 2D [batch * rows, cols] or 3D [batch, rows, cols]
// with rows >= M / N padded to 32 (a multiple of 32) and cols = K/32 padded to 8.
void opus_gemm_a4w4_bpreshuffle_launch(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &Y,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    std::optional<aiter_tensor_t> workspace,
    int kid,
    int split_k)
{
  aiter_detail::g_aiter_can_throw = true;
  constexpr const char* entry = "opus_gemm_a4w4_bpreshuffle_launch";
  opus_check_a8_family_tensors(entry, XQ, WQ, Y, AITER_DTYPE_fp4x2);
  opus_check_a8_scale_devices(entry, XQ, x_scale, w_scale);
  AITER_CHECK(Y.dtype() == AITER_DTYPE_bf16,
              entry, ": expected bf16 Y, got ", AiterDtype_to_str(Y.dtype()));
  AITER_CHECK(XQ.dim() == 3 && WQ.dim() == 3 && Y.dim() == 3,
              entry, ": XQ/WQ/Y must be 3D [batch, M, K/2], [batch, N, K/2], [batch, M, N]");
  AITER_CHECK(XQ.is_contiguous() && WQ.is_contiguous() && Y.is_contiguous() &&
                  x_scale.is_contiguous() && w_scale.is_contiguous(),
              entry, ": tensors must be contiguous");
  const int64_t batch = XQ.size(0), M = XQ.size(1), K = XQ.size(2) * 2, N = WQ.size(1);
  AITER_CHECK(batch >= 1 && M >= 1 && WQ.size(0) == batch && WQ.size(2) * 2 == K,
              entry, ": WQ must be [batch, N, K/2] matching XQ");
  AITER_CHECK(Y.size(0) == batch && Y.size(1) == M && Y.size(2) == N,
              entry, ": Y must be [batch, M, N]");
  AITER_CHECK(K % 256 == 0 && K > 0,
              entry, ": K must be a multiple of 256 (got K=", K, ")");
  AITER_CHECK(N % 16 == 0 && N > 0,
              entry, ": N must be a multiple of 16 (got N=", N, ")");
  const int64_t cols = (K / 32 + 7) / 8 * 8;
  auto check_scale = [&](const aiter_tensor_t& s, int64_t rows, const char* name) {
    AITER_CHECK(s.element_size() == 1, entry, ": ", name, " must be e8m0 (1 byte)");
    AITER_CHECK(s.dim() == 2 || (s.dim() == 3 && s.size(0) == batch),
                entry, ": ", name, " must be 2D or [batch, rows, cols]");
    AITER_CHECK(s.dim() == 3 || s.size(0) % batch == 0,
                entry, ": ", name, " rows are not divisible by batch");
    const int64_t r = s.dim() == 3 ? s.size(1) : s.size(0) / batch;
    AITER_CHECK(s.size(-1) == cols, entry, ": ", name, " must have K/32 padded to 8 = ",
                cols, " columns (got ", s.size(-1), ")");
    AITER_CHECK(r >= (rows + 31) / 32 * 32 && r % 32 == 0, entry, ": ", name, " needs >= ",
                (rows + 31) / 32 * 32, " rows per batch, a multiple of 32 (got ", r, ")");
  };
  check_scale(x_scale, M, "x_scale");
  check_scale(w_scale, N, "w_scale");
  if (workspace.has_value())
  {
    AITER_CHECK(workspace->is_gpu() && workspace->device_id == XQ.device_id,
                entry, ": workspace device ", workspace->device_id,
                " must match input device ", XQ.device_id);
  }

  const auto &info = opus_get_arch_info();
  if (info.arch == OpusGfxArch::Gfx950)
  {
#ifdef OPUS_BUILD_HAS_GFX950
    opus_a4w4_kid_dispatch_gfx950(kid)(XQ, WQ, Y, x_scale, w_scale, workspace, split_k);
    return;
#else
    AITER_CHECK(false, entry, ": module was not built with gfx950 support for current device ",
                info.dev);
#endif
  }
  AITER_CHECK(false, "no registered kernel for OPUS a4w4 bpreshuffle on device ", info.dev,
              " with gcnArchName='", info.name, "'");
}

#endif // !__HIP_DEVICE_COMPILE__
