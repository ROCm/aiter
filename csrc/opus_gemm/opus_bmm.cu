// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

// Host-side BMM frontends. These expose BMM/grouped-layout APIs while reusing
// the generated opus GEMM backend launcher symbols.
#ifndef __HIP_DEVICE_COMPILE__

#include "opus_bmm.h"
#include "opus_gemm_arch.cuh"
#include "opus_build_archs.h"
#include "opus_gemm_manifest.h"
#include "opus_gemm_utils.cuh"  // bf16_t / fp32_t

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
