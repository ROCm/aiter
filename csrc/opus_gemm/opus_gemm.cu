// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

// Host-side dispatcher (lookup table + heuristic).
#ifndef __HIP_DEVICE_COMPILE__

#include "opus_gemm_arch.cuh"                      // OpusGfxArch + opus_get_arch_info / opus_get_gfx_arch
#include "opus_build_archs.h"                      // OPUS_BUILD_HAS_GFX942 / OPUS_BUILD_HAS_GFX950
// gfx950 dispatcher always included (carries OpusA16W16NoscaleKernel typedef +
// a8w8 launchers used unconditionally below); a8w8 is gfx950-only today.
#include "gfx950/opus_gemm_arch_gfx950.cuh"        // opus_dispatch_a16w16_gfx950<T> / opus_a16w16_tune_dispatch_gfx950<T>
#ifdef OPUS_BUILD_HAS_GFX942
#include "gfx942/opus_gemm_arch_gfx942.cuh"        // opus_dispatch_a16w16_gfx942<T> / opus_a16w16_tune_dispatch_gfx942<T>
#endif
#ifdef OPUS_BUILD_HAS_GFX1250
#include "gfx1250/opus_gemm_arch_gfx1250.cuh"      // opus_a16w16_tune_dispatch_gfx1250<T> (tune-id entry only)
#endif
#include "opus_gemm_common.cuh"
#include "gfx950/opus_gemm_heuristic_dispatch_gfx950.cuh"  // OpusA16W16NoscaleKernel
#ifdef OPUS_BUILD_HAS_GFX942
#include "gfx942/opus_gemm_heuristic_dispatch_gfx942.cuh"
#endif
#include "opus_gemm_manifest.h"                    // a8w8 launcher symbols
#include "opus_gemm_utils.cuh"                     // bf16_t / fp32_t
#include "aiter_stream.h"                          // aiter::getCurrentHIPStream

#include <mutex>
#include <optional>
#include <unordered_map>

// a8w8 / a8w8_scale: single hardcoded launcher per dtype (no tuned table).
// Plain fn ptrs; std::function's type-erasure is pure waste here.
using OpusScaleKernel = void (*)(
    aiter_tensor_t &, aiter_tensor_t &,
    aiter_tensor_t &,
    std::optional<aiter_tensor_t>, std::optional<aiter_tensor_t>);

using OpusNoscaleKernel = void (*)(
    aiter_tensor_t &, aiter_tensor_t &,
    aiter_tensor_t &);

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

// a16w16 arch routers: switch on opus_get_gfx_arch() to per-arch dispatch.
template <typename CDataType>
OpusA16W16NoscaleKernel opus_dispatch_a16w16(int M, int N, int K, int batch, bool has_bias = false)
{
  switch (opus_get_gfx_arch())
  {
    case OpusGfxArch::Gfx950:
      return opus_dispatch_a16w16_gfx950<CDataType>(M, N, K, batch, has_bias);
#ifdef OPUS_BUILD_HAS_GFX942
    case OpusGfxArch::Gfx942:
      return opus_dispatch_a16w16_gfx942<CDataType>(M, N, K, batch, has_bias);
#endif
#ifdef OPUS_BUILD_HAS_GFX1250
    case OpusGfxArch::Gfx1250:
      return opus_dispatch_a16w16_gfx1250<CDataType>(M, N, K, batch, has_bias);
#endif
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
#ifdef OPUS_BUILD_HAS_GFX942
    case OpusGfxArch::Gfx942:
      return opus_a16w16_tune_dispatch_gfx942<CDataType>(id);
#endif
#ifdef OPUS_BUILD_HAS_GFX1250
    case OpusGfxArch::Gfx1250:
      return opus_a16w16_tune_dispatch_gfx1250<CDataType>(id);
#endif
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
  aiter_detail::g_aiter_can_throw = true;
  AITER_CHECK(XQ.dim() == 3, "XQ must be 3D [batch, M, K]");
  AITER_CHECK(WQ.dim() == 3, "WQ must be 3D [batch, N, K]");
  AITER_CHECK(Y.dim() == 3, "Y must be 3D [batch, M, N]");

  int M = XQ.size(1);
  int N = WQ.size(1);
  int K = XQ.size(2);

  bool has_scale = x_scale.has_value() && w_scale.has_value();

  if (XQ.dtype() == AITER_DTYPE_fp8)
  {
    // a8w8 / a8w8_scale launchers are gfx950-only today and don't yet flow through the arch-routed
    // dispatcher (they pick a single har...
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
    // Tuned-lookup-then-heuristic dispatch. splitK=0 = "launcher decides".
    int batch = XQ.size(0);
    const bool has_bias = bias.has_value();
    if (Y.dtype() == AITER_DTYPE_bf16)
    {
      opus_dispatch_a16w16<bf16_t>(M, N, K, batch, has_bias)(XQ, WQ, Y, bias, 0);
    }
    else if (Y.dtype() == AITER_DTYPE_fp32)
    {
      opus_dispatch_a16w16<fp32_t>(M, N, K, batch, has_bias)(XQ, WQ, Y, bias, 0);
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

// opus_gemm_a16w16_tune() — id-based tune entry.

// splitk kids: gfx950 [200,300) + nooob [1200,1300); gfx942 [10200, 10300).
static constexpr int OPUS_SPLITK_KID_MIN = 200;
static constexpr int OPUS_SPLITK_KID_MAX = 300;
static constexpr int OPUS_GFX942_KID_OFFSET = 10000;
static constexpr int OPUS_GFX942_SPLITK_KID_MAX = 300;
// gfx1250 cluster/TDM split-K (fp32 workspace + reduce) kids: [20000, 21000).
static constexpr int OPUS_GFX1250_SPLITK_KID_MIN = 20000;
static constexpr int OPUS_GFX1250_SPLITK_KID_MAX = 21000;
// SB a16w16 kids: gfx950 [4,10) + mirrors at +1000/.../+7000.
static constexpr int OPUS_A16W16_SB_KID_MIN = 4;
static constexpr int OPUS_A16W16_SB_KID_MAX = 10;
// Persistent a16w16 kids: compact [300, 316) = 4 tiles × 4 cpol groups.
static constexpr int OPUS_PERSISTENT_KID_MIN = 300;
static constexpr int OPUS_PERSISTENT_KID_MAX = 316;
// Mono-tile a16w16 kids: [1400, 1500). Mono-tile is intrinsically non-OOB
// (no tail handling in the kernel body), so kids land in the >=1000 band
// directly — there is no base/nooob mirror split for this family. See
// opus_gemm_common.py :: a16w16_mono_tile_kernels_list.
static constexpr int OPUS_MONO_TILE_KID_MIN = 1400;
static constexpr int OPUS_MONO_TILE_KID_MAX = 1500;
// non-OOB kid offset
static constexpr int OPUS_NOOOB_KID_OFFSET = 1000;

static inline bool opus_kid_is_splitk(int kid)
{
  return (kid >= OPUS_SPLITK_KID_MIN && kid < OPUS_SPLITK_KID_MAX) ||
         (kid >= OPUS_SPLITK_KID_MIN + OPUS_NOOOB_KID_OFFSET &&
          kid < OPUS_SPLITK_KID_MAX + OPUS_NOOOB_KID_OFFSET) ||
         (kid >= OPUS_SPLITK_KID_MIN + OPUS_GFX942_KID_OFFSET &&
          kid < OPUS_GFX942_SPLITK_KID_MAX + OPUS_GFX942_KID_OFFSET) ||
         (kid >= OPUS_GFX1250_SPLITK_KID_MIN &&
          kid < OPUS_GFX1250_SPLITK_KID_MAX);
}

static inline bool opus_kid_is_a16w16_sb(int kid)
{
  // SB a16w16 kid bases: 0/1000/2000/.../7000 + [4,10) (cpol mirrors).
  for (int base : {0, 1000, 2000, 3000, 4000, 5000, 6000, 7000})
  {
    if (kid >= base + OPUS_A16W16_SB_KID_MIN && kid < base + OPUS_A16W16_SB_KID_MAX)
      return true;
  }
  return false;
}

static inline bool opus_kid_is_persistent(int kid)
{
  return (kid >= OPUS_PERSISTENT_KID_MIN && kid < OPUS_PERSISTENT_KID_MAX) ||
         (kid >= OPUS_PERSISTENT_KID_MIN + OPUS_NOOOB_KID_OFFSET &&
          kid < OPUS_PERSISTENT_KID_MAX + OPUS_NOOOB_KID_OFFSET);
}

static inline bool opus_kid_is_mono_tile(int kid)
{
  // Mono-tile lives entirely in the non-OOB band [1400, 1500); no mirror.
  return kid >= OPUS_MONO_TILE_KID_MIN && kid < OPUS_MONO_TILE_KID_MAX;
}

static inline bool opus_kid_is_gfx942_splitk(int kid)
{
  return kid >= OPUS_SPLITK_KID_MIN + OPUS_GFX942_KID_OFFSET &&
         kid < OPUS_GFX942_SPLITK_KID_MAX + OPUS_GFX942_KID_OFFSET;
}

static inline bool opus_kid_is_gfx1250_splitk(int kid)
{
  return kid >= OPUS_GFX1250_SPLITK_KID_MIN && kid < OPUS_GFX1250_SPLITK_KID_MAX;
}

static inline bool opus_kid_supports_bias(int kid)
{
  // persistent and mono-tile do not support bias (kargs lacks
  // ptr_bias/stride_bias_batch; launchers reject non-empty bias up front).
  // gfx942 splitk/SB silently ignored bias; exclude explicitly to surface
  // misuse as a clear error.
  // gfx1250 cluster_tdm_splitk_ws DOES support bias (the reduce kernel folds
  // it once, like gfx950 flatmm_splitk).
  return (opus_kid_is_a16w16_sb(kid) || opus_kid_is_splitk(kid))
         && !opus_kid_is_gfx942_splitk(kid);
}

void opus_gemm_a16w16_tune(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &Y,
    std::optional<aiter_tensor_t> bias,
    int kernelId,
    int splitK)
{
  aiter_detail::g_aiter_can_throw = true;
  AITER_CHECK(XQ.dim() == 3, "XQ must be 3D [batch, M, K]");
  AITER_CHECK(WQ.dim() == 3, "WQ must be 3D [batch, N, K]");
  AITER_CHECK(Y.dim() == 3, "Y must be 3D [batch, M, N]");
  AITER_CHECK(XQ.dtype() == WQ.dtype(),
              "XQ and WQ should have the same dtype!");
  // Early-gate non-bias-capable kids for a clean error before launcher entry.
  AITER_CHECK(!bias.has_value() || opus_kid_supports_bias(kernelId),
              "opus_gemm_a16w16_tune: bias is currently only supported on "
              "a16w16 split-barrier kids [", OPUS_A16W16_SB_KID_MIN, ", ",
              OPUS_A16W16_SB_KID_MAX, ") or a16w16_flatmm_splitk kids [",
              OPUS_SPLITK_KID_MIN, ", ", OPUS_SPLITK_KID_MAX,
              "); got kid=", kernelId);

  if (XQ.dtype() == AITER_DTYPE_bf16)
  {
    // All splitk kids (gfx950/gfx942/gfx1250) force <fp32_t>: the main kernel
    // writes an fp32 workspace and a reduce kernel casts it to Y.dtype() at
    // runtime (gfx1250 cluster_tdm_splitk_ws now follows this same pattern).
    if (opus_kid_is_splitk(kernelId))
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

// ──────────────────────────────────────────────────────────────────────────────
// Splitk fp32 workspace: per-stream owner.
//
// Each splitk launcher (generated by gen_instances.py) needs a stable
// `opus_splitk_ws_handle*` to feed into both the main kernel and the reduce
// kernel; captured HIP graphs bake in that pointer. Previously this was a
// `static thread_local` slot — one handle per CPU thread — but under
// vLLM/sglang-style TBO two CPU threads drive two streams concurrently, and
// each captured graph needs its own buffer pointer baked in. The TLS form
// also tripped the in-capture grow guard on the second thread.
//
// Now we own the handle by stream: a process-global mutex-protected map
// keyed by hipStream_t. Eager: lazy-create on first lookup. Capture: caller
// must pre-register the handle via opus_gemm_workspace_init(), otherwise the
// lookup throws (cleaner than the prior SIGABRT). The framework calls
// opus_gemm_workspace_init() once per TBO stream eagerly before capture.
namespace {
struct SplitkWsRegistry {
  std::mutex mu;
  struct Owner {
    opus_splitk_ws_handle* host;
    opus_splitk_ws_handle* device;
  };
  std::unordered_map<hipStream_t, Owner*> map;
};
SplitkWsRegistry& splitk_ws_registry()
{
  static SplitkWsRegistry r;
  return r;
}
} // anonymous

opus_splitk_ws_handle* opus_splitk_ws_get(hipStream_t s, bool allow_create)
{
  auto& R = splitk_ws_registry();
  std::lock_guard<std::mutex> g(R.mu);
  auto it = R.map.find(s);
  if (it != R.map.end()) return it->second->host;
  AITER_CHECK(allow_create,
              "splitk workspace not initialized for the current CUDA stream. "
              "Call aiter.opus_gemm_workspace_init() inside "
              "`with torch.cuda.stream(s):` (and warm with the largest "
              "expected gemm) before HIP graph capture.");
  auto* owner = new SplitkWsRegistry::Owner{};
  opus_splitk_ws_handle* h = nullptr;
  HIP_CALL(hipHostMalloc(reinterpret_cast<void**>(&h),
                         sizeof(opus_splitk_ws_handle),
                         hipHostMallocCoherent));
  h->ptr   = nullptr;
  h->bytes = 0;
  owner->host   = h;
  owner->device = nullptr;
  R.map[s]      = owner;
  return h;
}

const opus_splitk_ws_handle* opus_splitk_ws_device_handle(hipStream_t s, bool allow_create)
{
  (void)opus_splitk_ws_get(s, allow_create);
  auto& R = splitk_ws_registry();
  std::lock_guard<std::mutex> g(R.mu);
  auto it = R.map.find(s);
  AITER_CHECK(it != R.map.end(), "splitk workspace not initialized for the current CUDA stream.");
  if (it->second->device == nullptr)
  {
    AITER_CHECK(allow_create,
                "splitk workspace device handle not initialized for the current CUDA stream. "
                "Warm the opus gfx942 splitK launcher eagerly before HIP graph capture.");
    HIP_CALL(hipMalloc(reinterpret_cast<void**>(&it->second->device),
                       sizeof(opus_splitk_ws_handle)));
    HIP_CALL(hipMemcpy(it->second->device,
                       it->second->host,
                       sizeof(opus_splitk_ws_handle),
                       hipMemcpyHostToDevice));
  }
  return it->second->device;
}

void opus_splitk_ws_sync_to_device(hipStream_t s)
{
  auto& R = splitk_ws_registry();
  std::lock_guard<std::mutex> g(R.mu);
  auto it = R.map.find(s);
  AITER_CHECK(it != R.map.end(), "splitk workspace not initialized for the current CUDA stream.");
  if (it->second->device == nullptr)
  {
    HIP_CALL(hipMalloc(reinterpret_cast<void**>(&it->second->device),
                       sizeof(opus_splitk_ws_handle)));
  }
  HIP_CALL(hipMemcpy(it->second->device,
                     it->second->host,
                     sizeof(opus_splitk_ws_handle),
                     hipMemcpyHostToDevice));
}

void opus_gemm_workspace_init()
{
  hipStream_t s = aiter::getCurrentHIPStream();
  hipStreamCaptureStatus cap = hipStreamCaptureStatusNone;
  HIP_CALL(hipStreamIsCapturing(s, &cap));
  AITER_CHECK(cap == hipStreamCaptureStatusNone,
              "opus_gemm_workspace_init must be called in eager mode "
              "(not inside HIP graph capture).");
  (void)opus_splitk_ws_get(s, /*allow_create=*/true);
}

#endif // !__HIP_DEVICE_COMPILE__
