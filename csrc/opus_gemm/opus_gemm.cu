// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

// Host-side dispatcher (lookup table + heuristic).
#ifndef __HIP_DEVICE_COMPILE__

#include "opus_gemm_arch.cuh"                      // OpusGfxArch + opus_get_arch_info / opus_get_gfx_arch
#include "opus_build_archs.h"                      // OPUS_BUILD_HAS_GFX942 / OPUS_BUILD_HAS_GFX950
#ifdef OPUS_BUILD_HAS_GFX950
#include "gfx950/opus_gemm_arch_gfx950.cuh"        // opus_dispatch_a16w16_gfx950<T> / opus_a16w16_tune_dispatch_gfx950<T>
#endif
#ifdef OPUS_BUILD_HAS_GFX942
#include "gfx942/opus_gemm_arch_gfx942.cuh"        // opus_dispatch_a16w16_gfx942<T> / opus_a16w16_tune_dispatch_gfx942<T>
#endif
#ifdef OPUS_BUILD_HAS_GFX1250
#include "gfx1250/opus_gemm_arch_gfx1250.cuh"      // opus_a16w16_tune_dispatch_gfx1250<T> (tune-id entry only)
#endif
#include "opus_gemm_common.cuh"
#ifdef OPUS_BUILD_HAS_GFX950
#include "gfx950/opus_gemm_heuristic_dispatch_gfx950.cuh"  // OpusA16W16NoscaleKernel
#endif
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
#ifdef OPUS_BUILD_HAS_GFX950
  return opus_gemm_512x256x256x128_4x2_16x16x128_1x128x128<CDataType>;
#else
  (void)M;
  (void)N;
  (void)K;
  return nullptr;
#endif
}

template <typename CDataType>
OpusNoscaleKernel opus_dispatch_a8w8(int M, int N, int K)
{
#ifdef OPUS_BUILD_HAS_GFX950
  return opus_gemm_512x256x256x128_2x4_16x16x128_0x0x0<CDataType>;
#else
  (void)M;
  (void)N;
  (void)K;
  return nullptr;
#endif
}

// a16w16 arch routers (gfx950/gfx942 only; gfx1250 uses its own dispatch with workspace).
#if defined(OPUS_BUILD_HAS_GFX950) || defined(OPUS_BUILD_HAS_GFX942)
template <typename CDataType>
OpusA16W16NoscaleKernel opus_dispatch_a16w16(int M, int N, int K, int batch, bool has_bias = false)
{
  switch (opus_get_gfx_arch())
  {
#ifdef OPUS_BUILD_HAS_GFX950
    case OpusGfxArch::Gfx950:
      return opus_dispatch_a16w16_gfx950<CDataType>(M, N, K, batch, has_bias);
#endif
#ifdef OPUS_BUILD_HAS_GFX942
    case OpusGfxArch::Gfx942:
      return opus_dispatch_a16w16_gfx942<CDataType>(M, N, K, batch, has_bias);
#endif
    default:
    {
      const auto &info = opus_get_arch_info();
      AITER_CHECK(false,
                  "opus_gemm: a16w16 dispatch via this path is only implemented for "
                  "gfx950/gfx942; gfx1250 uses a separate dispatch with workspace. "
                  "current device ", info.dev,
                  " has gcnArchName='", info.name, "'");
    }
  }
}

template <typename CDataType>
OpusA16W16NoscaleKernel
opus_a16w16_tune_dispatch(int id)
{
  switch (opus_get_gfx_arch())
  {
#ifdef OPUS_BUILD_HAS_GFX950
    case OpusGfxArch::Gfx950:
      return opus_a16w16_tune_dispatch_gfx950<CDataType>(id);
#endif
#ifdef OPUS_BUILD_HAS_GFX942
    case OpusGfxArch::Gfx942:
      return opus_a16w16_tune_dispatch_gfx942<CDataType>(id);
#endif
    default:
    {
      const auto &info = opus_get_arch_info();
      AITER_CHECK(false,
                  "opus_gemm_a16w16_tune: dispatch is only implemented for gfx950/gfx942 "
                  "via this path; gfx1250 uses a separate dispatch with workspace. "
                  "current device ", info.dev,
                  " has gcnArchName='", info.name, "'");
    }
  }
}
#endif // OPUS_BUILD_HAS_GFX950 || OPUS_BUILD_HAS_GFX942

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
#ifdef OPUS_BUILD_HAS_GFX950
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
#else
    AITER_CHECK(false,
                "opus_gemm: a8w8 path requires module_deepgemm_opus to be "
                "built with OPUS_BUILD_HAS_GFX950; current device ",
                arch_info.dev, " has gcnArchName='", arch_info.name, "'");
#endif
  }
  else if (XQ.dtype() == AITER_DTYPE_bf16)
  {
    // Tuned-lookup-then-heuristic dispatch. splitK=0 = "launcher decides".
    int batch = XQ.size(0);
    const bool has_bias = bias.has_value();
#ifdef OPUS_BUILD_HAS_GFX1250
    if (opus_get_gfx_arch() == OpusGfxArch::Gfx1250)
    {
      // gfx1250: all kids are split-K and need a workspace. The heuristic
      // dispatch returns a 6-arg function pointer (with workspace). We allocate
      // a temporary workspace here for the auto/heuristic path. For the tuned
      // path, Python allocates via torch.empty.
      auto fn = opus_dispatch_a16w16_gfx1250<fp32_t>(M, N, K, batch, has_bias);
      int padded_M = ((M + 63) / 64) * 64;
      int padded_N = ((N + 63) / 64) * 64;
      size_t ws_elems = (size_t)16 * padded_M * padded_N;
      size_t ws_bytes = ws_elems * sizeof(bf16_t);
      void* ws_ptr = nullptr;
      HIP_CALL(hipMalloc(&ws_ptr, ws_bytes));
      aiter_tensor_t ws_tensor{};
      ws_tensor.ptr = ws_ptr;
      ws_tensor.numel_ = ws_elems;
      ws_tensor.ndim = 1;
      ws_tensor.shape[0] = (int64_t)ws_elems;
      ws_tensor.strides[0] = 1;
      ws_tensor.dtype_ = AITER_DTYPE_bf16;
      ws_tensor.device_id = 0;
      fn(XQ, WQ, Y, ws_tensor, bias, 0);
      HIP_CALL(hipDeviceSynchronize());
      HIP_CALL(hipFree(ws_ptr));
    }
    else
#endif
    {
#if defined(OPUS_BUILD_HAS_GFX950) || defined(OPUS_BUILD_HAS_GFX942)
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
#else
      AITER_CHECK(false, "opus_gemm: no a16w16 dispatch available for this arch");
#endif
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
// gfx1250 split-K kids: [20000, 20100) plain cluster/TDM (fp32 workspace +
// reduce), [20100, 21000) clusterlaunch multicast ws, [21000, 30000) FUSED
// single-kernel in-cluster reduce (no reduce kernel; the fuse family sweeps
// tile x split_k x n_cluster x ws so it spans a wide band). All use the
// <fp32_t> lookup ABI and fold bias; the split-K band spans [20000, 30000).
static constexpr int OPUS_GFX1250_SPLITK_KID_MIN = 20000;
static constexpr int OPUS_GFX1250_SPLITK_KID_MAX = 30000;
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
    std::optional<aiter_tensor_t> workspace,
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
#ifdef OPUS_BUILD_HAS_GFX1250
      if (opus_kid_is_gfx1250_splitk(kernelId))
      {
        AITER_CHECK(workspace.has_value(),
                    "gfx1250 split-K kids require a workspace tensor "
                    "(allocated via torch.empty on the Python side)");
        auto& ws = workspace.value();
        opus_a16w16_tune_dispatch_gfx1250<fp32_t>(kernelId)(XQ, WQ, Y, ws, bias, splitK);
      }
      else
#endif
      {
#if defined(OPUS_BUILD_HAS_GFX950) || defined(OPUS_BUILD_HAS_GFX942)
        opus_a16w16_tune_dispatch<fp32_t>(kernelId)(XQ, WQ, Y, bias, splitK);
#else
        AITER_CHECK(false, "opus_gemm_a16w16_tune: non-gfx1250 splitk dispatch unavailable");
#endif
      }
    }
    else if (Y.dtype() == AITER_DTYPE_bf16)
    {
#if defined(OPUS_BUILD_HAS_GFX950) || defined(OPUS_BUILD_HAS_GFX942)
      opus_a16w16_tune_dispatch<bf16_t>(kernelId)(XQ, WQ, Y, bias, splitK);
#else
      AITER_CHECK(false, "opus_gemm_a16w16_tune: non-splitk bf16 dispatch unavailable for this arch");
#endif
    }
    else if (Y.dtype() == AITER_DTYPE_fp32)
    {
#if defined(OPUS_BUILD_HAS_GFX950) || defined(OPUS_BUILD_HAS_GFX942)
      opus_a16w16_tune_dispatch<fp32_t>(kernelId)(XQ, WQ, Y, bias, splitK);
#else
      AITER_CHECK(false, "opus_gemm_a16w16_tune: non-splitk fp32 dispatch unavailable for this arch");
#endif
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

void opus_gemm_a8w8_blockscale_bpreshuffle_tune(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    std::optional<aiter_tensor_t> x_scale,
    std::optional<aiter_tensor_t> w_scale,
    aiter_tensor_t &Y,
    int kernelId)
{
  aiter_detail::g_aiter_can_throw = true;
  const auto &arch_info = opus_get_arch_info();
  AITER_CHECK(arch_info.arch == OpusGfxArch::Gfx942,
              "opus_gemm_a8w8_blockscale_bpreshuffle_tune is only implemented "
              "for gfx942 today; current device ", arch_info.dev,
              " has gcnArchName='", arch_info.name, "'");
  AITER_CHECK(XQ.dtype() == AITER_DTYPE_fp8 && WQ.dtype() == AITER_DTYPE_fp8,
              "opus_gemm_a8w8_blockscale_bpreshuffle_tune expects fp8 XQ/WQ");
  AITER_CHECK(Y.dtype() == AITER_DTYPE_bf16,
              "opus_gemm_a8w8_blockscale_bpreshuffle_tune expects bf16 Y");
  AITER_CHECK(x_scale.has_value() && w_scale.has_value(),
              "opus_gemm_a8w8_blockscale_bpreshuffle_tune requires x_scale and w_scale");

#ifdef OPUS_BUILD_HAS_GFX942
  opus_a8w8_tune_dispatch_gfx942(kernelId)(XQ, WQ, Y, x_scale, w_scale);
#else
  AITER_CHECK(false,
              "module_deepgemm_opus was not built with OPUS_BUILD_HAS_GFX942");
#endif
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
//
// Teardown: entries are held for the process lifetime unless explicitly freed
// via opus_gemm_workspace_release() (current stream) or
// opus_gemm_workspace_release_all() (all streams). Both run in eager mode and
// synchronize before freeing.
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
#ifdef OPUS_BUILD_HAS_GFX950
  // gfx950 launchers feed the host handle STRAIGHT to the kernel, which
  // dereferences ptr/bytes on the device -- so it must be device-visible
  // pinned/coherent host memory.
  HIP_CALL(hipHostMalloc(reinterpret_cast<void**>(&h),
                         sizeof(opus_splitk_ws_handle),
                         hipHostMallocCoherent));
  h->ptr   = nullptr;
  h->bytes = 0;
#else
  // gfx942/gfx1250 read a device mirror (opus_splitk_ws_sync_to_device); the
  // device never dereferences this host handle. So plain host memory suffices
  // and we avoid pinned/coherent allocations entirely -- the OS reclaims plain
  // host memory at process exit with no dependency on HIP's pinned-memory
  // teardown (which can wedge fragile drivers and hang a subsequent process).
  h = new opus_splitk_ws_handle{nullptr, 0};
#endif
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
