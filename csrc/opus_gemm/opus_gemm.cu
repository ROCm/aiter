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
#include "aiter_stream.h"                          // aiter::getCurrentHIPStream
#include "aiter_ctypes_error.h"                    // safe exported C ABI

#include <cstdint>
#include <limits>
#include <optional>

AITER_CTYPES_ERROR_DEF

namespace
{
class OpusCabiDeviceStreamGuard
{
public:
  OpusCabiDeviceStreamGuard(int device, hipStream_t stream)
      : previous_stream_(aiter::getCurrentHIPStream())
  {
    HIP_CALL(hipGetDevice(&previous_device_));
    if (previous_device_ != device)
    {
      HIP_CALL(hipSetDevice(device));
      restore_device_ = true;
    }
    aiter::setCurrentHIPStream(stream);
  }

  ~OpusCabiDeviceStreamGuard() noexcept
  {
    aiter::setCurrentHIPStream(previous_stream_);
    if (restore_device_)
    {
      // Never throw while restoring state across the C ABI.
      (void)hipSetDevice(previous_device_);
    }
  }

  OpusCabiDeviceStreamGuard(const OpusCabiDeviceStreamGuard&) = delete;
  OpusCabiDeviceStreamGuard& operator=(const OpusCabiDeviceStreamGuard&) = delete;

private:
  int previous_device_ = -1;
  hipStream_t previous_stream_;
  bool restore_device_ = false;
};
} // namespace

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
#endif // OPUS_BUILD_HAS_GFX950 || OPUS_BUILD_HAS_GFX942

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

// A16W16 exact-kid entry used by pybind and the C ABI.
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

AITER_CTYPES_DEFINE_ENTRYPOINT_VOID(
    opus_gemm_a16w16_launch_cabi,
    (aiter_tensor_t* XQ,
     aiter_tensor_t* WQ,
     aiter_tensor_t* Y,
     aiter_tensor_t* bias,
     aiter_tensor_t* workspace,
     int64_t kid,
     int64_t split_k,
     hipStream_t stream),
    (XQ, WQ, Y, bias, workspace, kid, split_k, stream))
{
  AITER_CHECK(XQ != nullptr,
              "opus_gemm_a16w16_launch_cabi: XQ must not be null");
  AITER_CHECK(WQ != nullptr,
              "opus_gemm_a16w16_launch_cabi: WQ must not be null");
  AITER_CHECK(Y != nullptr,
              "opus_gemm_a16w16_launch_cabi: Y must not be null");
  AITER_CHECK(kid >= static_cast<int64_t>(std::numeric_limits<int>::min())
                  && kid <= static_cast<int64_t>(std::numeric_limits<int>::max()),
              "opus_gemm_a16w16_launch_cabi: kid is outside the C++ int range: ",
              kid);
  AITER_CHECK(split_k >= static_cast<int64_t>(std::numeric_limits<int>::min())
                  && split_k <= static_cast<int64_t>(std::numeric_limits<int>::max()),
              "opus_gemm_a16w16_launch_cabi: split_k is outside the C++ int range: ",
              split_k);

  const std::optional<aiter_tensor_t> optional_bias =
      bias == nullptr ? std::nullopt
                      : std::optional<aiter_tensor_t>{*bias};
  const std::optional<aiter_tensor_t> optional_workspace =
      workspace == nullptr ? std::nullopt
                           : std::optional<aiter_tensor_t>{*workspace};
  const OpusCabiDeviceStreamGuard device_stream_guard(XQ->device_id, stream);
  opus_gemm_a16w16_launch(
      *XQ,
      *WQ,
      *Y,
      optional_bias,
      optional_workspace,
      static_cast<int>(kid),
      static_cast<int>(split_k));
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
    int kid)
{
  aiter_detail::g_aiter_can_throw = true;
  constexpr const char* entry =
      "opus_gemm_a8w8_blockscale_bpreshuffle_launch";
  opus_check_a8_family_tensors(entry, XQ, WQ, Y);
  opus_check_a8_scale_devices(entry, XQ, x_scale, w_scale);

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

// Free everything a single Owner holds: the GPU workspace data buffer (owned via
// the host handle's `ptr`), the host coherent handle itself, and the device
// mirror. Caller must hold the registry mutex and must have synchronized any
// in-flight work that could still reference the buffer.
static void opus_splitk_ws_free_owner_locked(SplitkWsRegistry::Owner* owner)
{
  if (owner == nullptr) return;
  if (owner->host != nullptr)
  {
    if (owner->host->ptr != nullptr)
    {
      HIP_CALL(hipFree(owner->host->ptr));
      owner->host->ptr   = nullptr;
      owner->host->bytes = 0;
    }
#ifdef OPUS_BUILD_HAS_GFX950
    HIP_CALL(hipHostFree(owner->host));  // paired with hipHostMalloc above
#else
    delete owner->host;  // paired with plain `new` for the gfx942/gfx1250 path
#endif
    owner->host = nullptr;
  }
  if (owner->device != nullptr)
  {
    HIP_CALL(hipFree(owner->device));
    owner->device = nullptr;
  }
  delete owner;
}

// Release the splitk workspace (buffer + handles + registry entry) for the
// CURRENT stream. Safe to call when the stream was never registered (no-op).
// Must run in eager mode; frees are stream-capture-illegal.
void opus_gemm_workspace_release()
{
  hipStream_t s = aiter::getCurrentHIPStream();
  hipStreamCaptureStatus cap = hipStreamCaptureStatusNone;
  HIP_CALL(hipStreamIsCapturing(s, &cap));
  AITER_CHECK(cap == hipStreamCaptureStatusNone,
              "opus_gemm_workspace_release must be called in eager mode "
              "(not inside HIP graph capture).");
  // Drain the stream so no in-flight kernel references the buffer being freed.
  HIP_CALL(hipStreamSynchronize(s));
  auto& R = splitk_ws_registry();
  std::lock_guard<std::mutex> g(R.mu);
  auto it = R.map.find(s);
  if (it == R.map.end()) return;
  opus_splitk_ws_free_owner_locked(it->second);
  R.map.erase(it);
}

// Release the splitk workspace for ALL registered streams and clear the
// registry. Intended for explicit teardown (e.g. before a framework tears down
// its stream pool). Must run in eager mode.
void opus_gemm_workspace_release_all()
{
  auto& R = splitk_ws_registry();
  std::lock_guard<std::mutex> g(R.mu);
  if (R.map.empty()) return;
  // Drain all device work before freeing any buffer (buffers belong to many
  // streams; a single device sync covers them all).
  HIP_CALL(hipDeviceSynchronize());
  for (auto& kv : R.map)
  {
    opus_splitk_ws_free_owner_locked(kv.second);
  }
  R.map.clear();
}

#endif // !__HIP_DEVICE_COMPILE__
