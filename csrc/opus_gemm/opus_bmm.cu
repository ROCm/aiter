// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

// Host-side BMM frontends. These expose BMM/grouped-layout APIs while reusing
// the generated opus GEMM backend launcher symbols.
#include "gfx950/opus_gemm_pipeline_a8w8_scale_gfx950.cuh"

using opus_bmm_a8w8_mxscale_splitk_traits_gfx950 =
    opus_gemm_a8w8_scale_traits_gfx950<512,
      opus::seq<128, 256, 128>,
      opus::tuple<fp8_t, fp8_t, fp32_t, fp32_t, unsigned char>,
      opus::seq<16, 16, 4>,
      opus::seq<1, 128, 128>>;

#ifdef __HIP_DEVICE_COMPILE__
template __global__ void
gemm_a8w8_scale_splitk_kernel<opus_bmm_a8w8_mxscale_splitk_traits_gfx950>(
    opus_gemm_scale_splitk_kargs_gfx950);
#endif

template <typename D_OUT>
__global__ void opus_bmm_splitk_reduce_kernel(
    const opus_splitk_ws_handle* __restrict__ ws_handle,
    D_OUT* __restrict__ out,
    int split_k, int M, int N, int batch,
    int padded_M, int padded_N,
    int stride_c, int stride_c_batch)
{
#ifdef __HIP_DEVICE_COMPILE__
#if defined(__gfx950__)
  using namespace opus;
  constexpr int VEC = 16;
  constexpr int BLOCK = 64;
  constexpr int STEP = 16 / sizeof(D_OUT);
  static_assert(VEC % STEP == 0);

  const int bm_id = int(block_id_y());
  const int n_base = (int(block_id_x()) * BLOCK + int(thread_id_x())) * VEC;
  if (bm_id >= batch * M || n_base >= N) return;
  const int b = bm_id / M;
  const int m = bm_id - b * M;

  const float* __restrict__ workspace =
      reinterpret_cast<const float*>(ws_handle->ptr);
  const long split_stride = (long)batch * padded_M * padded_N;
  const int base = b * padded_M * padded_N + m * padded_N + n_base;
  auto g_ws = make_gmem(workspace, (unsigned int)(split_stride * split_k * sizeof(float)));
  vector_t<float, VEC> acc;
  static_for<VEC>([&](auto i) { acc[i.value] = 0.0f; });
  for (int s = 0; s < split_k; ++s) {
    #pragma unroll
    for (int g = 0; g < VEC / 4; ++g) {
      auto v = g_ws.template load<4>((long)s * split_stride + base + g * 4);
      static_for<4>([&](auto j) { acc[g * 4 + j.value] += v[j.value]; });
    }
  }

  vector_t<D_OUT, VEC> out_v;
  static_for<VEC>([&](auto i) { out_v[i.value] = static_cast<D_OUT>(acc[i.value]); });
  const int c_idx = b * stride_c_batch + m * stride_c + n_base;
  auto g_c = make_gmem(out, (unsigned int)(batch * stride_c_batch * sizeof(D_OUT)));
  if (n_base + VEC <= N) {
    static_for<VEC / STEP>([&](auto group) {
      constexpr int off = group.value * STEP;
      g_c.template store<STEP>(slice(out_v, number<off>{}, number<off + STEP>{}), c_idx + off);
    });
  } else {
    #pragma unroll
    for (int i = 0; i < VEC; ++i) {
      if (n_base + i < N) g_c.template store<1>(out_v[i], c_idx + i);
    }
  }
#endif
#endif
}

#ifdef __HIP_DEVICE_COMPILE__
template __global__ void opus_bmm_splitk_reduce_kernel<__bf16>(
    const opus_splitk_ws_handle*, __bf16*,
    int, int, int, int, int, int, int, int);
template __global__ void opus_bmm_splitk_reduce_kernel<float>(
    const opus_splitk_ws_handle*, float*,
    int, int, int, int, int, int, int, int);
#endif

#ifndef __HIP_DEVICE_COMPILE__

#include "opus_bmm.h"
#include "opus_gemm_arch.cuh"
#include "opus_build_archs.h"
#include "opus_gemm_manifest.h"
#include "opus_gemm_utils.cuh"  // bf16_t / fp32_t
#include "aiter_stream.h"

#include <optional>

// ── opus_bmm_a8w8_scale_mmajor() — zero-copy fp8 block-scale BMM ────────────
// O/Y are [M, batch, *] (dim0=M, dim1=batch); x_scale is
// [M, batch, K/GROUP_K] (per-token M). Weight WQ + w_scale stay batch-major
// ([batch, N, K] / [batch, N/GROUP_N, K/GROUP_K]). No caller-side transpose --
// feeds the DSV4 wo_a activation o=[num_tokens, n_groups, K] directly.
#ifdef OPUS_BUILD_HAS_GFX950
template <typename D_C>
void opus_gemm_512x128x256x128_4x2_16x16x128_1x128x128_mmajor(
    aiter_tensor_t &, aiter_tensor_t &, aiter_tensor_t &,
    std::optional<aiter_tensor_t>, std::optional<aiter_tensor_t>);
template <typename D_C>
void opus_gemm_a8w8_mxscale_512x128x256x128_4x2_16x16x128_1x128x128_mmajor(
    aiter_tensor_t &, aiter_tensor_t &, aiter_tensor_t &,
    std::optional<aiter_tensor_t>, std::optional<aiter_tensor_t>);
template<typename Traits>
__global__ void gemm_a8w8_scale_splitk_kernel(opus_gemm_scale_splitk_kargs_gfx950 kargs);
#endif

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
}

void opus_bmm_a8w8_scale_mmajor(
    aiter_tensor_t &O,
    aiter_tensor_t &wo_a,
    aiter_tensor_t &Y,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale)
{
  opus_bmm_a8w8_common_checks(O, wo_a, Y, "opus_bmm_a8w8_scale_mmajor");
#ifdef OPUS_BUILD_HAS_GFX950
  const auto &arch_info = opus_get_arch_info();
  AITER_CHECK(arch_info.arch == OpusGfxArch::Gfx950,
              "opus_bmm_a8w8_scale_mmajor is gfx950-only; current device ",
              arch_info.dev, " has gcnArchName='", arch_info.name, "'");
  if (Y.dtype() == AITER_DTYPE_bf16) {
    opus_gemm_512x128x256x128_4x2_16x16x128_1x128x128_mmajor<bf16_t>(
        O, wo_a, Y, x_scale, w_scale);
  } else {
    opus_gemm_512x128x256x128_4x2_16x16x128_1x128x128_mmajor<fp32_t>(
        O, wo_a, Y, x_scale, w_scale);
  }
#else
  AITER_CHECK(false,
              "opus_bmm_a8w8_scale_mmajor requires OPUS_BUILD_HAS_GFX950");
#endif
}

void opus_bmm_a8w8_mxscale_mmajor(
    aiter_tensor_t &O,
    aiter_tensor_t &wo_a,
    aiter_tensor_t &Y,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    int kernelId)
{
  opus_bmm_a8w8_common_checks(O, wo_a, Y, "opus_bmm_a8w8_mxscale_mmajor");
#ifdef OPUS_BUILD_HAS_GFX950
  const auto &arch_info = opus_get_arch_info();
  AITER_CHECK(arch_info.arch == OpusGfxArch::Gfx950,
              "opus_bmm_a8w8_mxscale_mmajor is gfx950-only; current device ",
              arch_info.dev, " has gcnArchName='", arch_info.name, "'");
  (void)kernelId;
  if (Y.dtype() == AITER_DTYPE_bf16) {
    opus_gemm_a8w8_mxscale_512x128x256x128_4x2_16x16x128_1x128x128_mmajor<bf16_t>(
        O, wo_a, Y, x_scale, w_scale);
  } else {
    opus_gemm_a8w8_mxscale_512x128x256x128_4x2_16x16x128_1x128x128_mmajor<fp32_t>(
        O, wo_a, Y, x_scale, w_scale);
  }
#else
  AITER_CHECK(false,
              "opus_bmm_a8w8_mxscale_mmajor requires OPUS_BUILD_HAS_GFX950");
#endif
}

void opus_bmm_a8w8_mxscale_splitk_mmajor(
    aiter_tensor_t &O,
    aiter_tensor_t &wo_a,
    aiter_tensor_t &Y,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    int splitK)
{
  opus_bmm_a8w8_common_checks(O, wo_a, Y, "opus_bmm_a8w8_mxscale_splitk_mmajor");
#ifdef OPUS_BUILD_HAS_GFX950
  const auto &arch_info = opus_get_arch_info();
  AITER_CHECK(arch_info.arch == OpusGfxArch::Gfx950,
              "opus_bmm_a8w8_mxscale_splitk_mmajor is gfx950-only; current device ",
              arch_info.dev, " has gcnArchName='", arch_info.name, "'");
  AITER_CHECK(splitK > 1, "splitK must be > 1");

  using Traits = opus_bmm_a8w8_mxscale_splitk_traits_gfx950;

  const int M = O.size(0);
  const int batch = O.size(1);
  const int N = wo_a.size(1);
  const int K = O.size(2);
  const int split_k = splitK;
  const int num_tiles_m = (M + 128 - 1) / 128;
  const int num_tiles_n = (N + 256 - 1) / 256;
  const int padded_M = num_tiles_m * 128;
  const int padded_N = num_tiles_n * 256;

  extern opus_splitk_ws_handle* opus_splitk_ws_get(hipStream_t, bool);
  auto stream = aiter::getCurrentHIPStream();
  hipStreamCaptureStatus capture_status = hipStreamCaptureStatusNone;
  HIP_CALL(hipStreamIsCapturing(stream, &capture_status));
  const bool capturing = (capture_status != hipStreamCaptureStatusNone);
  auto* ws_handle = opus_splitk_ws_get(stream, /*allow_create=*/!capturing);

  size_t ws_bytes = (size_t)split_k * (size_t)batch
                  * (size_t)padded_M * (size_t)padded_N * sizeof(float);
  if (ws_handle->ptr == nullptr || ws_bytes > ws_handle->bytes) {
    AITER_CHECK(!capturing,
                "splitk workspace grow inside HIP graph capture is not supported");
    void* new_ptr = nullptr;
    const size_t kGrowAlign = (size_t)4 * 1024 * 1024;
    size_t grow_bytes = ((ws_bytes + kGrowAlign - 1) / kGrowAlign) * kGrowAlign;
    HIP_CALL(hipMalloc(&new_ptr, grow_bytes));
    if (ws_handle->ptr != nullptr) {
      HIP_CALL(hipDeviceSynchronize());
      HIP_CALL(hipFree(ws_handle->ptr));
    }
    ws_handle->ptr = new_ptr;
    ws_handle->bytes = grow_bytes;
  }

  opus_gemm_scale_splitk_kargs_gfx950 kargs{};
  kargs.ptr_a = O.data_ptr();
  kargs.ptr_b = wo_a.data_ptr();
  kargs.ws_handle = ws_handle;
  kargs.m = M; kargs.n = N; kargs.k = K; kargs.batch = batch;
  kargs.split_k = split_k;
  kargs.stride_a = (int)O.stride(0);
  kargs.stride_b = (int)wo_a.stride(1);
  kargs.stride_ws = padded_N;
  kargs.stride_a_batch = (int)O.stride(1);
  kargs.stride_b_batch = (int)wo_a.stride(0);
  kargs.stride_ws_batch = padded_M * padded_N;
  kargs.ptr_sfa = x_scale.data_ptr();
  kargs.ptr_sfb = w_scale.data_ptr();
  kargs.stride_sfa = (int)x_scale.stride(0);
  kargs.stride_sfa_batch = (int)x_scale.stride(1);
  kargs.stride_sfb = (int)w_scale.stride(1);
  kargs.stride_sfb_batch = (int)w_scale.stride(0);

  dim3 grid_main(num_tiles_m * num_tiles_n * split_k, 1, batch);
  dim3 block_main(512);
  gemm_a8w8_scale_splitk_kernel<Traits><<<grid_main, block_main, 0, stream>>>(kargs);

  constexpr int REDUCE_VEC = 16;
  constexpr int REDUCE_BS = 64;
  dim3 grid_reduce((N + REDUCE_VEC * REDUCE_BS - 1) / (REDUCE_VEC * REDUCE_BS),
                   batch * M, 1);
  dim3 block_reduce(REDUCE_BS);
  const int y_stride_c = (int)Y.stride(0);
  const int y_stride_c_batch = (int)Y.stride(1);
  if (Y.dtype() == AITER_DTYPE_bf16) {
    opus_bmm_splitk_reduce_kernel<__bf16>
        <<<grid_reduce, block_reduce, 0, stream>>>(
            ws_handle, reinterpret_cast<__bf16*>(Y.data_ptr()),
            split_k, M, N, batch, padded_M, padded_N,
            y_stride_c, y_stride_c_batch);
  } else {
    opus_bmm_splitk_reduce_kernel<float>
        <<<grid_reduce, block_reduce, 0, stream>>>(
            ws_handle, reinterpret_cast<float*>(Y.data_ptr()),
            split_k, M, N, batch, padded_M, padded_N,
            y_stride_c, y_stride_c_batch);
  }
#else
  AITER_CHECK(false,
              "opus_bmm_a8w8_mxscale_splitk_mmajor requires OPUS_BUILD_HAS_GFX950");
#endif
}

// ── opus_bmm_a8w8_uniform_scale(_mmajor)() — fp8 block-scale uniform BMM ──────────
// fp8 Route-B low-barrier variant (4-wave full-tile, direct store). Two layout
// surfaces share the same kernel/tiles:
//   * batch-major: O=[batch,M,K], wo_a=[batch,N,K], Y=[batch,M,N].
//   * mmajor: O=[M,batch,K], wo_a=[batch,N,K], Y=[M,batch,N].
// kernelId selects the tile (700=128x128, 701=256x128); Y dtype in {fp32,bf16}.
#ifdef OPUS_BUILD_HAS_GFX950
template <typename D_C>
void opus_gemm_a8w8_uniform_scale_256x128x128x128_2x2_16x16x128_1x128x128(
    aiter_tensor_t &, aiter_tensor_t &, aiter_tensor_t &,
    std::optional<aiter_tensor_t>, std::optional<aiter_tensor_t>);
template <typename D_C>
void opus_gemm_a8w8_uniform_scale_256x256x128x128_2x2_16x16x128_1x128x128(
    aiter_tensor_t &, aiter_tensor_t &, aiter_tensor_t &,
    std::optional<aiter_tensor_t>, std::optional<aiter_tensor_t>);
template <typename D_C>
void opus_gemm_a8w8_uniform_scale_256x128x128x128_2x2_16x16x128_1x128x128_mmajor(
    aiter_tensor_t &, aiter_tensor_t &, aiter_tensor_t &,
    std::optional<aiter_tensor_t>, std::optional<aiter_tensor_t>);
template <typename D_C>
void opus_gemm_a8w8_uniform_scale_256x256x128x128_2x2_16x16x128_1x128x128_mmajor(
    aiter_tensor_t &, aiter_tensor_t &, aiter_tensor_t &,
    std::optional<aiter_tensor_t>, std::optional<aiter_tensor_t>);

template <bool MMAJOR>
static void opus_uniform_scale_dispatch(int kernelId, aiter_tensor_t &O,
                                        aiter_tensor_t &wo_a, aiter_tensor_t &Y,
                                        aiter_tensor_t &x_scale,
                                        aiter_tensor_t &w_scale)
{
  const bool y_bf16 = (Y.dtype() == AITER_DTYPE_bf16);
  switch (kernelId)
  {
    case 701:
      if constexpr (MMAJOR) {
        if (y_bf16)
          opus_gemm_a8w8_uniform_scale_256x256x128x128_2x2_16x16x128_1x128x128_mmajor<bf16_t>(O, wo_a, Y, x_scale, w_scale);
        else
          opus_gemm_a8w8_uniform_scale_256x256x128x128_2x2_16x16x128_1x128x128_mmajor<fp32_t>(O, wo_a, Y, x_scale, w_scale);
      } else {
        if (y_bf16)
          opus_gemm_a8w8_uniform_scale_256x256x128x128_2x2_16x16x128_1x128x128<bf16_t>(O, wo_a, Y, x_scale, w_scale);
        else
          opus_gemm_a8w8_uniform_scale_256x256x128x128_2x2_16x16x128_1x128x128<fp32_t>(O, wo_a, Y, x_scale, w_scale);
      }
      break;
    case 700:
    default:
      if constexpr (MMAJOR) {
        if (y_bf16)
          opus_gemm_a8w8_uniform_scale_256x128x128x128_2x2_16x16x128_1x128x128_mmajor<bf16_t>(O, wo_a, Y, x_scale, w_scale);
        else
          opus_gemm_a8w8_uniform_scale_256x128x128x128_2x2_16x16x128_1x128x128_mmajor<fp32_t>(O, wo_a, Y, x_scale, w_scale);
      } else {
        if (y_bf16)
          opus_gemm_a8w8_uniform_scale_256x128x128x128_2x2_16x16x128_1x128x128<bf16_t>(O, wo_a, Y, x_scale, w_scale);
        else
          opus_gemm_a8w8_uniform_scale_256x128x128x128_2x2_16x16x128_1x128x128<fp32_t>(O, wo_a, Y, x_scale, w_scale);
      }
      break;
  }
}
#endif

static void opus_uniform_scale_common_checks(aiter_tensor_t &O, aiter_tensor_t &wo_a,
                                             aiter_tensor_t &Y, const char *who)
{
  aiter_detail::g_aiter_can_throw = true;
  AITER_CHECK(O.dim() == 3 && wo_a.dim() == 3 && Y.dim() == 3,
              who, ": O/wo_a/Y must be 3D");
  AITER_CHECK(O.dtype() == AITER_DTYPE_fp8 && wo_a.dtype() == AITER_DTYPE_fp8,
              who, ": O and wo_a must be fp8");
  AITER_CHECK(Y.dtype() == AITER_DTYPE_fp32 || Y.dtype() == AITER_DTYPE_bf16,
              who, ": Y must be fp32 or bf16");
}

void opus_bmm_a8w8_uniform_scale(
    aiter_tensor_t &O,
    aiter_tensor_t &wo_a,
    aiter_tensor_t &Y,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    int kernelId)
{
  opus_uniform_scale_common_checks(O, wo_a, Y, "opus_bmm_a8w8_uniform_scale");
#ifdef OPUS_BUILD_HAS_GFX950
  const auto &arch_info = opus_get_arch_info();
  AITER_CHECK(arch_info.arch == OpusGfxArch::Gfx950,
              "opus_bmm_a8w8_uniform_scale is gfx950-only; current device ",
              arch_info.dev, " has gcnArchName='", arch_info.name, "'");
  opus_uniform_scale_dispatch<false>(kernelId, O, wo_a, Y, x_scale, w_scale);
#else
  AITER_CHECK(false, "opus_bmm_a8w8_uniform_scale requires OPUS_BUILD_HAS_GFX950");
#endif
}

void opus_bmm_a8w8_uniform_scale_mmajor(
    aiter_tensor_t &O,
    aiter_tensor_t &wo_a,
    aiter_tensor_t &Y,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    int kernelId)
{
  opus_uniform_scale_common_checks(O, wo_a, Y, "opus_bmm_a8w8_uniform_scale_mmajor");
#ifdef OPUS_BUILD_HAS_GFX950
  const auto &arch_info = opus_get_arch_info();
  AITER_CHECK(arch_info.arch == OpusGfxArch::Gfx950,
              "opus_bmm_a8w8_uniform_scale_mmajor is gfx950-only; current device ",
              arch_info.dev, " has gcnArchName='", arch_info.name, "'");
  opus_uniform_scale_dispatch<true>(kernelId, O, wo_a, Y, x_scale, w_scale);
#else
  AITER_CHECK(false,
              "opus_bmm_a8w8_uniform_scale_mmajor requires OPUS_BUILD_HAS_GFX950");
#endif
}

#endif  // !__HIP_DEVICE_COMPILE__
