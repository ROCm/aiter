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
  // kernelId selects the tile; the Y dtype selects the C path within it. Still
  // a hand-written switch rather than the codegen'd
  // opus_bmm_a8w8_mxscale_tune_dispatch table above, because these tiles are
  // hand-written and have no kid rows yet.
  //   0  128x128x256, 1 WG/CU  -- prefill; 100% occupancy from m >= 2048
  //   1   16x 32x256, 1 WG/CU  -- decode: 4x the workgroups at fixed batch
  //   2   16x 32x256, 2 WG/CU  -- BROKEN, measurement only. Wrong (and
  //       nondeterministic) whenever the grid exceeds the CU count: 2 WG/CU
  //       makes co-resident workgroups share the compile-time-id named
  //       barriers. See the kid2/kid3 warning in the traits header.
  //   3   16x 32x512, 2 WG/CU  -- BROKEN, same cause as kid2
  //   4   16x 64x256, 1 WG/CU, 192 THREADS -- decode; 6 waves (2 producer +
  //       4 consumer) so all 4 SIMDs host a consumer. A/B against kid1, which
  //       it matches on every per-wave quantity.
  //   5   16x 64x256, 1 WG/CU, 128 threads -- kid4's control: kid4's tile at
  //       kid1's wave count, so kid4-vs-kid5 isolates the wave count from B_N.
  //   6   16x128x256, 1 WG/CU, 192 threads -- B_N ladder, kExpN 2
  //   7   16x256x256, 1 WG/CU, 192 threads -- B_N ladder, kExpN 4
  // kid 0..7 all take a PER-COLUMN w_scale ([batch, N, K/GROUP_K]). The two
  // below take the DSV4 128x128 block scale ([batch, N/128, K/GROUP_K]) and the
  // launcher enforces the shape, so they are not drop-in for kid4/kid7:
  //   8   16x 64x256, GROUP_N=128 -- kid4 at the DSV4 scale granularity
  //   9   16x256x256, GROUP_N=128 -- kid7 at the DSV4 scale granularity
  //  10   16x192x256, GROUP_N=128 -- kExpN=3, the only blocked-but-not-uniform
  //                                 tile; exists to test that path, not to win
  //  13   = kid0 + A's scale staged in LDS (SF_A_LDS), so kid13-vs-kid0 is the
  //         pair that measures the panel. 11 and 12 were the kid8/kid9 twins of
  //         the earlier TDM version; ids left unused rather than reassigned,
  //         because old sweep logs name them.
  //  14   = kid13 + B's scale staged too (SF_B_LDS), the gfx950 arrangement.
  //         kid13-vs-kid14 isolates the B side; kid14 is the only tile whose
  //         inner loop issues no global scale load at all.
  //  17   = kid13 with A's panel filled by TDM instead of cooperatively, at a
  //         geometry (width 32, pad 16) that reproduces kid13's pitch exactly,
  //         so kid13-vs-kid17 is the FILL MECHANISM alone. Caps K at 4096.
  //  18   = the same but at the original geometry (width 128, pad 4), so
  //         kid18-vs-kid17 is geometry at fixed mechanism.
  // An out-of-range kid must throw: silently falling back to kid 0 would make a
  // tile sweep report kid 0's timing under a dozen different names.
  AITER_CHECK((kernelId >= 0 && kernelId <= 10) || kernelId == 13 ||
                  kernelId == 14 || kernelId == 17 || kernelId == 18,
              "opus_bmm_a8w8_mxscale_bpreshuffle: kernelId must be 0..10, 13, "
              "14, 17 or 18 (got ", kernelId, "); 0 = prefill 128x128x256, "
              "1..7 = decode tiles (per-column w_scale), 8..10 = decode tiles "
              "(128x128 w_scale), 13 = kid0 with A's scale staged in LDS, 14 = "
              "kid13 with B's staged as well, 17/18 = kid13 with A's panel "
              "filled by TDM (width 32 / width 128)");

#define OPUS_BMM_BPRESHUF_DISPATCH(TILE)                                    \
  do {                                                                      \
    if (Y.dtype() == AITER_DTYPE_bf16)                                      \
      opus_bmm_a8w8_mxscale_bpreshuffle_launch_gfx1250<TILE<bf16_t>>(       \
          O, wo_a, Y, x_scale, w_scale, splitK);                            \
    else                                                                    \
      opus_bmm_a8w8_mxscale_bpreshuffle_launch_gfx1250<TILE<fp32_t>>(       \
          O, wo_a, Y, x_scale, w_scale, splitK);                            \
  } while (0)

  switch (kernelId) {
    case 1:
      OPUS_BMM_BPRESHUF_DISPATCH(
          opus_bmm_a8w8_mxscale_bpreshuffle_tile_dec_n32_gfx1250);
      break;
    case 2:
      OPUS_BMM_BPRESHUF_DISPATCH(
          opus_bmm_a8w8_mxscale_bpreshuffle_tile_dec_n32_wg2_gfx1250);
      break;
    case 3:
      OPUS_BMM_BPRESHUF_DISPATCH(
          opus_bmm_a8w8_mxscale_bpreshuffle_tile_dec_n32_k512_gfx1250);
      break;
    case 4:
      OPUS_BMM_BPRESHUF_DISPATCH(
          opus_bmm_a8w8_mxscale_bpreshuffle_tile_dec_n64_w6_gfx1250);
      break;
    case 5:
      OPUS_BMM_BPRESHUF_DISPATCH(
          opus_bmm_a8w8_mxscale_bpreshuffle_tile_dec_n64_w4_gfx1250);
      break;
    case 6:
      OPUS_BMM_BPRESHUF_DISPATCH(
          opus_bmm_a8w8_mxscale_bpreshuffle_tile_dec_n128_w6_gfx1250);
      break;
    case 7:
      OPUS_BMM_BPRESHUF_DISPATCH(
          opus_bmm_a8w8_mxscale_bpreshuffle_tile_dec_n256_w6_gfx1250);
      break;
    case 8:
      OPUS_BMM_BPRESHUF_DISPATCH(
          opus_bmm_a8w8_mxscale_bpreshuffle_tile_dec_n64_w6_gn128_gfx1250);
      break;
    case 9:
      OPUS_BMM_BPRESHUF_DISPATCH(
          opus_bmm_a8w8_mxscale_bpreshuffle_tile_dec_n256_w6_gn128_gfx1250);
      break;
    case 10:
      OPUS_BMM_BPRESHUF_DISPATCH(
          opus_bmm_a8w8_mxscale_bpreshuffle_tile_dec_n192_w6_gn128_gfx1250);
      break;
    case 13:
      OPUS_BMM_BPRESHUF_DISPATCH(
          opus_bmm_a8w8_mxscale_bpreshuffle_tile_sfa_gfx1250);
      break;
    case 14:
      OPUS_BMM_BPRESHUF_DISPATCH(
          opus_bmm_a8w8_mxscale_bpreshuffle_tile_sfab_gfx1250);
      break;
    case 17:
      OPUS_BMM_BPRESHUF_DISPATCH(
          opus_bmm_a8w8_mxscale_bpreshuffle_tile_sfa_tdm32_gfx1250);
      break;
    case 18:
      OPUS_BMM_BPRESHUF_DISPATCH(
          opus_bmm_a8w8_mxscale_bpreshuffle_tile_sfa_tdm128_gfx1250);
      break;
    default:
      OPUS_BMM_BPRESHUF_DISPATCH(
          opus_bmm_a8w8_mxscale_bpreshuffle_tile_gfx1250);
      break;
  }

#undef OPUS_BMM_BPRESHUF_DISPATCH
#endif  // OPUS_BUILD_HAS_GFX1250
}

#endif  // !__HIP_DEVICE_COMPILE__
