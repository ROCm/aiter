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
#ifdef OPUS_BUILD_HAS_GFX950
#include "gfx950/opus_co_launch_gfx950.cuh"
#include "gfx950/opus_gemm_traits_a8w8_blockscale_grouped_gfx950.cuh"
#include "opus_gemm_a8w8_blockscale_grouped_kid_table.h"   // generated gfx950 FP8 grouped kid table
#endif

#include <algorithm>
#include <climits>
#include <optional>
#include <string>

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
    const aiter_tensor_t &Y)
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
  AITER_CHECK(XQ.dtype() == AITER_DTYPE_fp8 && WQ.dtype() == AITER_DTYPE_fp8,
              entry, ": expected fp8 XQ/WQ, got ",
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

// ---- FP8 blockscale grouped GEMMs (gfx950, pre-built .co images of gen_co/gfx950/a8w8_blockscale_grouped) ----
static void opus_check_same_device(const char* entry, const aiter_tensor_t &XQ,
                                   std::initializer_list<const aiter_tensor_t*> tensors)
{
  for (const auto* t : tensors)
    AITER_CHECK(t->is_gpu() && t->device_id == XQ.device_id, entry,
                ": all tensors must be on the device of XQ (", XQ.device_id, ")");
}

#ifdef OPUS_BUILD_HAS_GFX950
struct OpusA8W8BsGroupedKid
{
  int kid, b_m, b_n;
  const char* name;
};

static constexpr OpusA8W8BsGroupedKid opus_a8w8_bs_m_grouped_kids[] = {
    OPUS_A8W8_BLOCKSCALE_M_GROUPED_KID_TABLE{-1, 0, 0, nullptr}};
static constexpr OpusA8W8BsGroupedKid opus_a8w8_bs_k_grouped_kids[] = {
    OPUS_A8W8_BLOCKSCALE_K_GROUPED_KID_TABLE{-1, 0, 0, nullptr}};

static const OpusA8W8BsGroupedKid&
opus_a8w8_bs_grouped_find_kid(const char* entry, const OpusA8W8BsGroupedKid* kids, int kid)
{
  AITER_CHECK(opus_get_gfx_arch() == OpusGfxArch::Gfx950, entry, ": gfx950 only");
  const OpusA8W8BsGroupedKid* k = kids;
  while (k->name != nullptr && k->kid != kid)
    ++k;
  AITER_CHECK(k->name != nullptr, entry, ": unknown kid ", kid, " (or its images are not built)");
  return *k;
}

static bool opus_is_e8m0(const aiter_tensor_t& t)
{
  return t.dtype() == AITER_DTYPE_fp8_e8m0 || t.dtype() == AITER_DTYPE_u8;
}
#endif

// FP8 blockscale M-grouped GEMM (DeepGEMM m_grouped, fwd NT / dgrad NN): E8M0 x_scale 1x128 (storage [K/128, M]),
// w_scale 128x128; 256-row tiles.
void opus_gemm_a8w8_blockscale_grouped_launch(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &Y,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &group,
    aiter_tensor_t &workspace,
    int kid,
    bool masked)
{
  aiter_detail::g_aiter_can_throw = true;
  const char* entry = "opus_gemm_a8w8_blockscale_grouped_launch";
  opus_check_a8_family_tensors(entry, XQ, WQ, Y);
  opus_check_a8_scale_devices(entry, XQ, x_scale, w_scale);
  opus_check_same_device(entry, XQ, {&group, &workspace});
#ifdef OPUS_BUILD_HAS_GFX950
  const OpusA8W8BsGroupedKid& k = opus_a8w8_bs_grouped_find_kid(entry, opus_a8w8_bs_m_grouped_kids, kid);

  AITER_CHECK(XQ.dtype() == AITER_DTYPE_fp8 && WQ.dtype() == AITER_DTYPE_fp8 &&
                  Y.dtype() == AITER_DTYPE_bf16 && opus_is_e8m0(x_scale) && opus_is_e8m0(w_scale),
              entry, ": XQ and WQ must be fp8, Y bf16, the scales E8M0 (one byte)");
  AITER_CHECK(group.dtype() == AITER_DTYPE_i32 && group.dim() == 1 && group.stride(0) == 1,
              entry, ": the group tensor must be a contiguous 1-D int32 tensor");
  const int rank = masked ? 3 : 2;
  AITER_CHECK(XQ.dim() == rank && Y.dim() == rank && x_scale.dim() == rank && WQ.dim() == 3 &&
                  w_scale.dim() == 3,
              entry,
              masked ? ": masked needs XQ [G, max_m, K], WQ [G, N, K], Y [G, max_m, N] and their scales"
                     : ": contiguous needs XQ [M, K], WQ [G, N, K], Y [M, N] and their scales");
  const int64_t G = WQ.size(0), N = WQ.size(1), K = WQ.size(2);
  const int64_t M = XQ.size(-2);
  AITER_CHECK(XQ.size(-1) == K && Y.size(-2) == M && Y.size(-1) == N &&
                  (!masked || (XQ.size(0) == G && Y.size(0) == G && x_scale.size(0) == G)),
              entry, ": shapes of XQ / WQ / Y do not match");
  AITER_CHECK(x_scale.size(-2) == M && x_scale.size(-1) == K / 128 && w_scale.size(0) == G &&
                  w_scale.size(1) == N / 128 && w_scale.size(2) == K / 128,
              entry, ": x_scale must be [", masked ? "G, max_m" : "M", ", K/128], w_scale [G, N/128, K/128]");
  AITER_CHECK(group.size(0) == (masked ? G : M), entry, ": the group tensor must have ",
              masked ? "G" : "M", " entries");
  AITER_CHECK(K % 128 == 0, entry, ": K=", K, " must be a multiple of 128");
  AITER_CHECK(N % k.b_n == 0, entry, ": N=", N, " must be a multiple of ", k.b_n);
  AITER_CHECK(masked || M % OPUS_A8W8_BS_GROUP_ALIGN == 0, entry, ": contiguous M=", M,
              " must be a multiple of ", OPUS_A8W8_BS_GROUP_ALIGN);
  AITER_CHECK(masked || (G <= OPUS_A8W8_BS_M_GROUPED_PLAN_MAX_GROUPS && workspace.dtype() == AITER_DTYPE_i32 &&
                         workspace.dim() == 1 && workspace.stride(0) == 1 && workspace.size(0) >= 2 * G),
              entry, ": contiguous needs G <= ", OPUS_A8W8_BS_M_GROUPED_PLAN_MAX_GROUPS,
              " and an int32 workspace of 2 G entries");
  if (M == 0 || N == 0 || G == 0)
    return;

  // B: K-contiguous [G, N, K] (NT) or N-contiguous [G, K, N] (NN); its scale likewise.
  const bool b_t = !(WQ.stride(2) == 1 && WQ.stride(1) >= K);
  AITER_CHECK(!b_t || (WQ.stride(1) == 1 && WQ.stride(2) >= N), entry,
              ": WQ must be K-contiguous [G, N, K] or N-contiguous [G, K, N]");
  AITER_CHECK(b_t ? (w_scale.stride(1) == 1 && w_scale.stride(2) >= N / 128)
                  : (w_scale.stride(2) == 1 && w_scale.stride(1) >= K / 128),
              entry, ": w_scale must have the layout of WQ ([G, N/128, K/128] or [G, K/128, N/128])");
  const int64_t ldb = b_t ? WQ.stride(2) : WQ.stride(1);
  const int64_t ldsfb = b_t ? w_scale.stride(2) : w_scale.stride(1);
  const int64_t lda = XQ.stride(-2), ldc = Y.stride(-2);
  AITER_CHECK(XQ.stride(-1) == 1 && lda >= K && Y.stride(-1) == 1 && ldc >= N, entry,
              ": XQ and Y rows must be contiguous");
  const int64_t ldsfa = x_scale.stride(-1);
  AITER_CHECK(x_scale.stride(-2) == 1 && ldsfa >= M, entry,
              ": x_scale must be M-contiguous (storage [K/128, M], e.g. per_1x128 quant with transpose_scale)");
  const int64_t a_batch = masked ? XQ.stride(0) : 0, c_batch = masked ? Y.stride(0) : 0;
  const int64_t sfa_batch = masked ? x_scale.stride(0) : 0;
  AITER_CHECK(lda % 16 == 0 && ldb % 16 == 0 && WQ.stride(0) % 16 == 0 && a_batch % 16 == 0 &&
                  ldc % 8 == 0 && ldsfa % 16 == 0 && sfa_batch % 16 == 0,
              entry, ": XQ / WQ / x_scale pitches must be multiples of 16 bytes, Y's of 8 elements");
  for (const auto* t : {&XQ, &WQ, &Y, &x_scale})
    AITER_CHECK(reinterpret_cast<uintptr_t>(t->data_ptr()) % 16 == 0, entry,
                ": XQ / WQ / Y / x_scale must be 16-byte aligned");
  // The kernel keeps a group of B, its scales, a batch of x_scale and the pitches in 32-bit ints.
  AITER_CHECK((b_t ? K : N) * ldb <= INT_MAX && w_scale.stride(0) <= INT_MAX && K / 128 * ldsfa <= INT_MAX &&
                  sfa_batch <= INT_MAX && lda * 256 <= INT_MAX && ldc * 256 * 2 <= INT_MAX && M <= INT_MAX,
              entry, ": a group of XQ / WQ / Y / scales exceeds the kernel's 32-bit offsets");

  opus_gemm_a8w8_blockscale_grouped_kargs_gfx950 kargs{};
  kargs.ptr_a = XQ.data_ptr();
  kargs.ptr_b = WQ.data_ptr();
  kargs.ptr_c = Y.data_ptr();
  int* const ws = static_cast<int*>(workspace.data_ptr());
  kargs.ptr_rows = masked ? static_cast<const int*>(group.data_ptr()) : ws;
  kargs.ptr_first = masked ? nullptr : ws + G;
  kargs.ptr_sfa = x_scale.data_ptr();
  kargs.ptr_sfb = w_scale.data_ptr();
  kargs.stride_b_group = WQ.stride(0);
  kargs.stride_a_batch = a_batch;
  kargs.stride_c_batch = c_batch;
  kargs.m = static_cast<int>(M);
  kargs.n = static_cast<int>(N);
  kargs.k = static_cast<int>(K);
  kargs.stride_a = static_cast<int>(lda);
  kargs.stride_b = static_cast<int>(ldb);
  kargs.stride_c = static_cast<int>(ldc);
  kargs.stride_sfa = static_cast<int>(ldsfa);
  kargs.stride_sfa_batch = static_cast<int>(sfa_batch);
  kargs.stride_sfb = static_cast<int>(ldsfb);
  kargs.stride_sfb_group = static_cast<int>(w_scale.stride(0));
  kargs.groups = static_cast<int>(G);

  // Grid: an upper bound of the tiles (<= ceil(rows / 256) per group) x N tiles; extra blocks exit.
  const int64_t tiles_n = N / k.b_n;
  const int64_t chunks = (M + OPUS_A8W8_BS_GROUP_ALIGN - 1) / OPUS_A8W8_BS_GROUP_ALIGN;
  const int64_t slots = masked ? G * ((M + 255) / 256) : (chunks + G + 1) / 2;
  AITER_CHECK(slots * tiles_n <= INT_MAX, entry, ": grid too large");
  const std::string symbol = std::string(k.name) + (b_t ? "_nn" : "_nt");
  const std::string image = symbol + ".co";
  const hipStream_t stream = aiter::getCurrentHIPStream();
  if (!masked) {   // the rows / first row of every group of grouped_layout
    opus_gemm_a8w8_blockscale_m_grouped_plan_kargs_gfx950 plan{};
    plan.layout = static_cast<const int*>(group.data_ptr());
    plan.rows = ws;
    plan.first = ws + G;
    plan.m = static_cast<int>(M);
    plan.groups = static_cast<int>(G);
    opus_co_launch_gfx950<sizeof(plan)>(
        opus_gfx950_co::co_kernel(symbol + "_plan", image, OPUS_A8W8_BLOCKSCALE_GROUPED_CO_DIR), dim3(1),
        dim3(OPUS_A8W8_BS_M_GROUPED_PLAN_THREADS), plan, stream);
  }
  opus_co_launch_gfx950<sizeof(kargs)>(
      opus_gfx950_co::co_kernel(symbol, image, OPUS_A8W8_BLOCKSCALE_GROUPED_CO_DIR),
      dim3(static_cast<unsigned>(slots * tiles_n), 1, 1), dim3(256), kargs, stream);
#else
  AITER_CHECK(false, entry, ": the module was built without gfx950");
#endif
}

// FP8 blockscale K-grouped GEMM (DeepGEMM k_grouped, wgrad TN): E8M0 scales 128x1 along the rows.
void opus_gemm_a8w8_blockscale_k_grouped_launch(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &Y,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &grouped_layout,
    std::optional<aiter_tensor_t> C,
    aiter_tensor_t &workspace,
    int kid,
    int seg_len)
{
  aiter_detail::g_aiter_can_throw = true;
  const char* entry = "opus_gemm_a8w8_blockscale_k_grouped_launch";
  opus_check_a8_family_tensors(entry, XQ, WQ, Y);
  opus_check_a8_scale_devices(entry, XQ, x_scale, w_scale);
  opus_check_same_device(entry, XQ, {&grouped_layout, &workspace});
  if (C)
    opus_check_same_device(entry, XQ, {&*C});
#ifdef OPUS_BUILD_HAS_GFX950
  const OpusA8W8BsGroupedKid& k = opus_a8w8_bs_grouped_find_kid(entry, opus_a8w8_bs_k_grouped_kids, kid);

  const bool f32 = Y.dtype() == AITER_DTYPE_fp32;
  AITER_CHECK(XQ.dtype() == AITER_DTYPE_fp8 && WQ.dtype() == AITER_DTYPE_fp8 &&
                  (f32 || Y.dtype() == AITER_DTYPE_bf16) && opus_is_e8m0(x_scale) && opus_is_e8m0(w_scale),
              entry, ": XQ and WQ must be fp8, Y fp32 or bf16, the scales E8M0 (one byte)");
  AITER_CHECK(f32 || (!C && seg_len <= 0), entry,
              ": bf16 Y supports neither C nor a split (seg_len <= 0)");
  AITER_CHECK(grouped_layout.dtype() == AITER_DTYPE_i32 && grouped_layout.dim() == 1 && grouped_layout.stride(0) == 1, entry,
              ": grouped_layout must be a contiguous 1-D int32 tensor");
  AITER_CHECK(workspace.dtype() == AITER_DTYPE_i32 && workspace.dim() == 1 &&
                  workspace.stride(0) == 1,
              entry, ": workspace must be a contiguous 1-D int32 tensor");
  AITER_CHECK(XQ.dim() == 2 && WQ.dim() == 2 && Y.dim() == 3 && x_scale.dim() == 2 && w_scale.dim() == 2,
              entry, ": needs XQ [R, M], WQ [R, N], Y [G, M, N], x_scale [R/128, M], w_scale [R/128, N]");
  const int64_t R = XQ.size(0), M = XQ.size(1), N = WQ.size(1), G = Y.size(0);
  AITER_CHECK(WQ.size(0) == R && Y.size(1) == M && Y.size(2) == N && grouped_layout.size(0) == G, entry,
              ": shapes of XQ / WQ / Y / grouped_layout do not match");
  AITER_CHECK(R % OPUS_A8W8_BS_GROUP_ALIGN == 0 && x_scale.size(0) == R / 128 && x_scale.size(1) == M &&
                  w_scale.size(0) == R / 128 && w_scale.size(1) == N,
              entry, ": R must be a multiple of 128, x_scale [R/128, M], w_scale [R/128, N]");
  if (C)
    AITER_CHECK(C->dtype() == Y.dtype() && C->dim() == 3 && C->size(0) == G &&
                    C->size(1) == M && C->size(2) == N && C->stride(0) == Y.stride(0) &&
                    C->stride(1) == Y.stride(1) && C->stride(2) == Y.stride(2),
                entry, ": C must have the dtype, shape and strides of Y");
  if (M == 0 || N == 0 || G == 0)
    return;

  AITER_CHECK(N % k.b_n == 0, entry, ": N=", N, " must be a multiple of ", k.b_n);
  const int64_t lda = XQ.stride(0), ldb = WQ.stride(0), ldc = Y.stride(1);
  const int64_t ldsfa = x_scale.stride(0), ldsfb = w_scale.stride(0);
  AITER_CHECK(XQ.stride(1) == 1 && lda >= M && WQ.stride(1) == 1 && ldb >= N &&
                  x_scale.stride(1) == 1 && ldsfa >= M && w_scale.stride(1) == 1 && ldsfb >= N &&
                  Y.stride(2) == 1 && ldc >= N && Y.stride(0) >= M * ldc,
              entry, ": XQ, WQ, the scales and the matrices of Y must be row-major");
  AITER_CHECK(lda % 16 == 0 && ldb % 16 == 0 && ldsfa % 16 == 0 && ldsfb % 16 == 0 && ldc % 8 == 0 &&
                  Y.stride(0) % 8 == 0,
              entry, ": the pitches of XQ / WQ / scales must be multiples of 16 bytes, Y's of 8 elements");
  for (const auto* t : {&XQ, &WQ, &Y, &x_scale, &w_scale})
    AITER_CHECK(reinterpret_cast<uintptr_t>(t->data_ptr()) % 16 == 0, entry,
                ": XQ / WQ / Y / scales must be 16-byte aligned");
  if (C)
    AITER_CHECK(reinterpret_cast<uintptr_t>(C->data_ptr()) % 16 == 0, entry,
                ": C must be 16-byte aligned");
  AITER_CHECK(seg_len <= 0 || seg_len % OPUS_A8W8_BS_GROUP_ALIGN == 0, entry, ": seg_len=", seg_len,
              " must be a multiple of ", OPUS_A8W8_BS_GROUP_ALIGN);
  constexpr int NO_SPLIT = INT_MAX / OPUS_A8W8_BS_GROUP_ALIGN * OPUS_A8W8_BS_GROUP_ALIGN;
  const int seg = seg_len > 0 ? seg_len : NO_SPLIT;
  // A segment is addressed with 32-bit byte offsets: at most max_rows rows (the plan kernel traps past it).
  const int64_t max_rows = INT_MAX / std::max(lda, ldb) - 128;
  AITER_CHECK(R <= INT_MAX && max_rows >= OPUS_A8W8_BS_GROUP_ALIGN && ldc * 256 * 4 <= INT_MAX &&
                  R / 128 * std::max(ldsfa, ldsfb) <= INT_MAX && (seg_len <= 0 || seg_len <= max_rows),
              entry, ": seg_len=", seg_len, " exceeds the kernel's 32-bit offsets (at most ", max_rows, " rows)");

  const int64_t tiles_m = (M + k.b_m - 1) / k.b_m, tiles_n = N / k.b_n;
  const int64_t tiles = tiles_m * tiles_n;
  const int64_t full_items = R / seg;
  const int64_t turn_count = full_items > 0 ? G * tiles : 0;
  const int64_t items = full_items + G;
  AITER_CHECK(items * tiles <= INT_MAX && G * tiles <= INT_MAX && full_items < 65536, entry,
              ": grid too large");
  AITER_CHECK(workspace.size(0) >= 4 * items + turn_count, entry, ": workspace needs ",
              4 * items + turn_count, " int32 entries, got ", workspace.size(0));

  int* const ws = static_cast<int*>(workspace.data_ptr());
  opus_gemm_a8w8_blockscale_k_grouped_plan_kargs_gfx950 plan{};
  plan.grouped_layout = static_cast<const int*>(grouped_layout.data_ptr());
  plan.table = ws;
  plan.turn = ws + 4 * items;
  plan.groups = static_cast<int>(G);
  plan.seg_len = seg;
  plan.rows = static_cast<int>(R);
  plan.full_items = static_cast<int>(full_items);
  plan.turn_count = static_cast<int>(turn_count);
  plan.max_rows = static_cast<int>(std::min<int64_t>(max_rows, INT_MAX));

  opus_gemm_a8w8_blockscale_k_grouped_kargs_gfx950 kargs{};
  kargs.ptr_a = XQ.data_ptr();
  kargs.ptr_b = WQ.data_ptr();
  kargs.ptr_c = Y.data_ptr();
  kargs.ptr_src = C ? C->data_ptr() : nullptr;
  kargs.table = ws;
  kargs.turn = plan.turn;
  kargs.ptr_sfa = x_scale.data_ptr();
  kargs.ptr_sfb = w_scale.data_ptr();
  kargs.stride_c_group = Y.stride(0);
  kargs.m = static_cast<int>(M);
  kargs.n = static_cast<int>(N);
  kargs.stride_a = static_cast<int>(lda);
  kargs.stride_b = static_cast<int>(ldb);
  kargs.stride_c = static_cast<int>(ldc);
  kargs.stride_sfa = static_cast<int>(ldsfa);
  kargs.stride_sfb = static_cast<int>(ldsfb);
  kargs.write_empty = !(C && C->data_ptr() == Y.data_ptr());

  const std::string symbol = std::string(k.name) + (f32 ? "_tn_f32" : "_tn");
  const std::string image = symbol + ".co";
  const hipStream_t stream = aiter::getCurrentHIPStream();
  opus_co_launch_gfx950<sizeof(plan)>(
      opus_gfx950_co::co_kernel(symbol + "_plan", image, OPUS_A8W8_BLOCKSCALE_GROUPED_CO_DIR), dim3(1),
      dim3(OPUS_A8W8_BS_K_GROUPED_PLAN_THREADS), plan, stream);
  opus_co_launch_gfx950<sizeof(kargs)>(
      opus_gfx950_co::co_kernel(symbol, image, OPUS_A8W8_BLOCKSCALE_GROUPED_CO_DIR),
      dim3(static_cast<unsigned>(items * tiles)), dim3(256), kargs, stream);
#else
  AITER_CHECK(false, entry, ": the module was built without gfx950");
#endif
}

#endif // !__HIP_DEVICE_COMPILE__
