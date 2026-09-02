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
  // Wider prefill tiles. NUM_SLOTS is 3 throughout -- the producer ring only
  // implements that depth; slots=2 deadlocks (see the traits header).
  //  19   256x128x256, 192 thr, consumer grid 2x2 -- M doubled, squared grid
  //  20   128x256x256, 192 thr, grid 1x4          -- N doubled
  //  21   256x256x128, 320 thr, grid 2x4          -- both doubled, kid22's
  //       register footprint at 4x kid22's output tile
  //  22   128x128x256, 192 thr, grid 1x4  CONTROL: kid0's tile with 4 consumer
  //       waves instead of 2, so kid22-vs-kid0 is the wave count alone
  //  23   128x128x128, 192 thr, grid 1x4  CONTROL: kid22 at B_K=128, which
  //       prices the halved K reuse kid21 pays for its LDS
  //  24   128x128x256, 192 thr, grid 2x2  CONTROL: kid22's tile squared, the
  //       clean read on the grid shape by itself
  //  25   256x128x128, 192 thr, grid 2x2 -- kid19 with B_K halved so the tile
  //       fits the 6-wave 512-VGPR ceiling instead of spilling 108 B/lane
  // Validity is the table's membership test, below -- a second hand-kept id
  // range here would drift from it, and did: it silently rejected 23 and 24
  // after they were added to the table.

  // Table-driven kid -> launcher dispatch, mirroring the gfx950 pattern
  // (opus_bmm_a8w8_mxscale_tune_dispatch above). Each entry dispatches on
  // Y.dtype internally so the table holds one fn_ptr per kid, not two.
  using BpreshufLauncher = void (*)(
      aiter_tensor_t &, aiter_tensor_t &, aiter_tensor_t &,
      aiter_tensor_t &, aiter_tensor_t &, int);

#define OPUS_BMM_BPRESHUF_ENTRY(TILE)                                        \
  +[](aiter_tensor_t &O_, aiter_tensor_t &wo_a_, aiter_tensor_t &Y_,         \
      aiter_tensor_t &x_scale_, aiter_tensor_t &w_scale_, int splitK_) {     \
    if (Y_.dtype() == AITER_DTYPE_bf16)                                      \
      opus_bmm_a8w8_mxscale_bpreshuffle_launch_gfx1250<TILE<bf16_t>>(        \
          O_, wo_a_, Y_, x_scale_, w_scale_, splitK_);                        \
    else                                                                     \
      opus_bmm_a8w8_mxscale_bpreshuffle_launch_gfx1250<TILE<fp32_t>>(        \
          O_, wo_a_, Y_, x_scale_, w_scale_, splitK_);                        \
  }

  // The non-specialized tiles are a DIFFERENT kernel symbol, so they need their
  // own entry macro; the table type is shared because the launcher signature is.
#define OPUS_BMM_BPRESHUF_NS_ENTRY(TILE)                                     \
  +[](aiter_tensor_t &O_, aiter_tensor_t &wo_a_, aiter_tensor_t &Y_,         \
      aiter_tensor_t &x_scale_, aiter_tensor_t &w_scale_, int splitK_) {     \
    if (Y_.dtype() == AITER_DTYPE_bf16)                                      \
      opus_bmm_a8w8_mxscale_bpreshuffle_nospec_launch_gfx1250<TILE<bf16_t>>( \
          O_, wo_a_, Y_, x_scale_, w_scale_, splitK_);                       \
    else                                                                     \
      opus_bmm_a8w8_mxscale_bpreshuffle_nospec_launch_gfx1250<TILE<fp32_t>>( \
          O_, wo_a_, Y_, x_scale_, w_scale_, splitK_);                       \
  }

  static const std::unordered_map<int, BpreshufLauncher> kBpreshuf = {
    { 0, OPUS_BMM_BPRESHUF_ENTRY(opus_bmm_a8w8_mxscale_bpreshuffle_tile_gfx1250)},
    { 1, OPUS_BMM_BPRESHUF_ENTRY(opus_bmm_a8w8_mxscale_bpreshuffle_tile_dec_n32_gfx1250)},
    { 2, OPUS_BMM_BPRESHUF_ENTRY(opus_bmm_a8w8_mxscale_bpreshuffle_tile_dec_n32_wg2_gfx1250)},
    { 3, OPUS_BMM_BPRESHUF_ENTRY(opus_bmm_a8w8_mxscale_bpreshuffle_tile_dec_n32_k512_gfx1250)},
    { 4, OPUS_BMM_BPRESHUF_ENTRY(opus_bmm_a8w8_mxscale_bpreshuffle_tile_dec_n64_w6_gfx1250)},
    { 5, OPUS_BMM_BPRESHUF_ENTRY(opus_bmm_a8w8_mxscale_bpreshuffle_tile_dec_n64_w4_gfx1250)},
    { 6, OPUS_BMM_BPRESHUF_ENTRY(opus_bmm_a8w8_mxscale_bpreshuffle_tile_dec_n128_w6_gfx1250)},
    { 7, OPUS_BMM_BPRESHUF_ENTRY(opus_bmm_a8w8_mxscale_bpreshuffle_tile_dec_n256_w6_gfx1250)},
    { 8, OPUS_BMM_BPRESHUF_ENTRY(opus_bmm_a8w8_mxscale_bpreshuffle_tile_dec_n64_w6_gn128_gfx1250)},
    { 9, OPUS_BMM_BPRESHUF_ENTRY(opus_bmm_a8w8_mxscale_bpreshuffle_tile_dec_n256_w6_gn128_gfx1250)},
    {10, OPUS_BMM_BPRESHUF_ENTRY(opus_bmm_a8w8_mxscale_bpreshuffle_tile_dec_n192_w6_gn128_gfx1250)},
    {13, OPUS_BMM_BPRESHUF_ENTRY(opus_bmm_a8w8_mxscale_bpreshuffle_tile_sfa_gfx1250)},
    {14, OPUS_BMM_BPRESHUF_ENTRY(opus_bmm_a8w8_mxscale_bpreshuffle_tile_sfab_gfx1250)},
    {17, OPUS_BMM_BPRESHUF_ENTRY(opus_bmm_a8w8_mxscale_bpreshuffle_tile_sfa_tdm32_gfx1250)},
    {18, OPUS_BMM_BPRESHUF_ENTRY(opus_bmm_a8w8_mxscale_bpreshuffle_tile_sfa_tdm128_gfx1250)},
    {19, OPUS_BMM_BPRESHUF_ENTRY(opus_bmm_a8w8_mxscale_bpreshuffle_tile_pf_m256_gfx1250)},
    {20, OPUS_BMM_BPRESHUF_ENTRY(opus_bmm_a8w8_mxscale_bpreshuffle_tile_pf_n256_gfx1250)},
    {21, OPUS_BMM_BPRESHUF_ENTRY(opus_bmm_a8w8_mxscale_bpreshuffle_tile_pf_m256n256_gfx1250)},
    {22, OPUS_BMM_BPRESHUF_ENTRY(opus_bmm_a8w8_mxscale_bpreshuffle_tile_pf_w6_gfx1250)},
    {23, OPUS_BMM_BPRESHUF_ENTRY(opus_bmm_a8w8_mxscale_bpreshuffle_tile_pf_bk128_gfx1250)},
    {24, OPUS_BMM_BPRESHUF_ENTRY(opus_bmm_a8w8_mxscale_bpreshuffle_tile_pf_w6_2x2_gfx1250)},
    {25, OPUS_BMM_BPRESHUF_ENTRY(opus_bmm_a8w8_mxscale_bpreshuffle_tile_pf_m256_bk128_gfx1250)},
    {26, OPUS_BMM_BPRESHUF_NS_ENTRY(opus_bmm_a8w8_mxscale_bpreshuffle_tile_ns128_gfx1250)},
    {27, OPUS_BMM_BPRESHUF_NS_ENTRY(opus_bmm_a8w8_mxscale_bpreshuffle_tile_ns256_gfx1250)},
    {28, OPUS_BMM_BPRESHUF_NS_ENTRY(opus_bmm_a8w8_mxscale_bpreshuffle_tile_ns256_gn128_gfx1250)},
    {29, OPUS_BMM_BPRESHUF_NS_ENTRY(opus_bmm_a8w8_mxscale_bpreshuffle_tile_ns128_gn128_gfx1250)},
    {30, OPUS_BMM_BPRESHUF_NS_ENTRY(opus_bmm_a8w8_mxscale_bpreshuffle_tile_ns256_gn128_sf_gfx1250)},
    {31, OPUS_BMM_BPRESHUF_NS_ENTRY(opus_bmm_a8w8_mxscale_bpreshuffle_tile_ns128_gn128_sf_gfx1250)},
    {32, OPUS_BMM_BPRESHUF_NS_ENTRY(opus_bmm_a8w8_mxscale_bpreshuffle_tile_fly256_gfx1250)},
    {33, OPUS_BMM_BPRESHUF_NS_ENTRY(opus_bmm_a8w8_mxscale_bpreshuffle_tile_fly256_nb4_gfx1250)},
  };
#undef OPUS_BMM_BPRESHUF_ENTRY
#undef OPUS_BMM_BPRESHUF_NS_ENTRY

  auto it = kBpreshuf.find(kernelId);
  AITER_CHECK(it != kBpreshuf.end(),
              "opus_bmm_a8w8_mxscale_bpreshuffle: unknown kernelId ", kernelId,
              "; valid ids: 0..10, 13, 14, 17..33");
  it->second(O, wo_a, Y, x_scale, w_scale, splitK);
#endif  // OPUS_BUILD_HAS_GFX1250
}

// ---------------------------------------------------------------------------
// CLUSTER-LAUNCH, FUSED SPLIT-K variant of the above.
// ---------------------------------------------------------------------------
// A separate symbol rather than a kernelId range on the entry point above, for
// the same reason that one is separate from opus_bmm_a8w8_mxscale: the CALL
// CONTRACT differs. This one requires a caller-allocated partial workspace
// whenever splitK > 1, and silently ignoring a missing one would produce
// garbage, not an error.
//
// TILE ids are SHARED with the non-cluster entry point: kid 0 is the same
// 128x128x256 prefill tile there and here, so a sweep can compare the two
// directly. Only a SUBSET of those kids is instantiated for the cluster path
// (each one costs |splitK x mClusterWg| kernels, ~2.6 s of build each); the
// switch throws by name on the rest instead of falling back, and
// OPUS_BMM_BPRESHUF_CC_INST in
// opus_bmm_a8w8_mxscale_bpreshuffle_gfx1250.cu is the one list to edit to add
// more.
//
// (splitK, mClusterWg) are CLUSTER DIMENSIONS, hence compile-time. The switch
// below enumerates the instantiated pairs; an unlisted pair throws rather than
// rounding to a neighbour, because rounding splitK changes the numerics and
// rounding mClusterWg changes the grid.
#ifdef OPUS_BUILD_HAS_GFX1250
namespace opus_bmm_cc_detail {

// A 0-element tensor is how the binding spells "not passed" for ws / bias.
static inline aiter_tensor_t* opt(aiter_tensor_t& t)
{
  return (t.numel() == 0 || t.data_ptr() == nullptr) ? nullptr : &t;
}

// mClusterWg > 1 folds adjacent M-TILES into one cluster to multicast B. With a
// single M tile there are no peers to multicast to: every extra workgroup in the
// cluster is out-of-range, runs the whole pipeline against a zero-extent window
// and throws its result away. That is CORRECT but is pure waste, and at decode
// shapes (m <= 16 < B_M) it is the default thing to get wrong -- so say so
// rather than let a sweep read it as "multicast does not help".
template <typename Tile>
static inline void warn_if_mcwg_wasted(int m, int mClusterWg, int kernelId)
{
  if (mClusterWg <= 1) return;
  const int ntm = (m + Tile::kBlockM - 1) / Tile::kBlockM;
  AITER_CHECK(ntm > 1,
              "opus_bmm_a8w8_mxscale_bpreshuffle_clusterclaunch: kernelId=",
              kernelId, " has B_M=", Tile::kBlockM, ", so M=", m,
              " gives ceil(M/B_M)=1 M-tile and mClusterWg=", mClusterWg,
              " would launch ", mClusterWg - 1,
              " out-of-range workgroups per cluster with nothing to multicast "
              "to. Use mClusterWg=1 at this shape.");
}

}  // namespace opus_bmm_cc_detail
#endif  // OPUS_BUILD_HAS_GFX1250

size_t opus_bmm_a8w8_mxscale_bpreshuffle_clusterclaunch_ws_numel(
    int m, int n, int batch, int splitK, int kernelId)
{
#ifndef OPUS_BUILD_HAS_GFX1250
  (void)m; (void)n; (void)batch; (void)splitK; (void)kernelId;
  return 0;
#else
  if (splitK <= 1) return 0;
  // B_M / B_N are all this needs, so the tile only has to be resolved far enough
  // to read its block shape; bf16_t stands in for both C dtypes. The COUNT is
  // dtype-independent -- but the caller must allocate those elements as FP32,
  // because DataWs is fp32 for both C dtypes (see OPUS_BMM_CC_LAUNCH_ONE).
  // Computed from the tile's block shape directly rather than through the
  // launcher's ws_elems helper: that one takes SplitK as a template argument
  // (it is a cluster dimension there), and a sizing call the caller makes
  // BEFORE it has picked a kernel needs splitK to be an ordinary runtime int.
  // The two must agree -- same formula, one factor moved from compile time to
  // run time.
  #define OPUS_BMM_CC_WS(TILE)                                                 \
    do {                                                                       \
      using Tile_ = TILE<bf16_t>;                                              \
      const size_t ntm = (size_t)((m + Tile_::kBlockM - 1) / Tile_::kBlockM);  \
      const size_t ntn = (size_t)((n + Tile_::kBlockN - 1) / Tile_::kBlockN);  \
      return ntm * ntn * (size_t)batch * (size_t)(splitK - 1)                  \
             * (size_t)Tile_::kBlockM * (size_t)Tile_::kBlockN;                \
    } while (0)
  switch (kernelId) {
    case 1:  OPUS_BMM_CC_WS(opus_bmm_a8w8_mxscale_bpreshuffle_tile_dec_n32_gfx1250);
    case 4:  OPUS_BMM_CC_WS(opus_bmm_a8w8_mxscale_bpreshuffle_tile_dec_n64_w6_gfx1250);
    case 13: OPUS_BMM_CC_WS(opus_bmm_a8w8_mxscale_bpreshuffle_tile_sfa_gfx1250);
    default: OPUS_BMM_CC_WS(opus_bmm_a8w8_mxscale_bpreshuffle_tile_gfx1250);
  }
  #undef OPUS_BMM_CC_WS
#endif
}

void opus_bmm_a8w8_mxscale_bpreshuffle_clusterclaunch(
    aiter_tensor_t &O,
    aiter_tensor_t &wo_a,
    aiter_tensor_t &Y,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &ws,
    aiter_tensor_t &bias,
    int splitK,
    int mClusterWg,
    int kernelId)
{
#ifndef OPUS_BUILD_HAS_GFX1250
  (void)O; (void)wo_a; (void)Y; (void)x_scale; (void)w_scale;
  (void)ws; (void)bias; (void)splitK; (void)mClusterWg; (void)kernelId;
  aiter_detail::g_aiter_can_throw = true;
  AITER_CHECK(false,
              "opus_bmm_a8w8_mxscale_bpreshuffle_clusterclaunch requires "
              "OPUS_BUILD_HAS_GFX1250 (add gfx1250 to GPU_ARCHS)");
#else
  {
    const auto &arch_info = opus_get_arch_info();
    AITER_CHECK(arch_info.arch == OpusGfxArch::Gfx1250,
                "opus_bmm_a8w8_mxscale_bpreshuffle_clusterclaunch is "
                "gfx1250-only; current device ", arch_info.dev,
                " has gcnArchName='", arch_info.name, "'");
  }
  aiter_detail::g_aiter_can_throw = true;

  AITER_CHECK(kernelId == 0 || kernelId == 1 || kernelId == 4 || kernelId == 13,
              "opus_bmm_a8w8_mxscale_bpreshuffle_clusterclaunch: kernelId must "
              "be 0, 1, 4 or 13 (got ", kernelId,
              "). Tile ids match the non-cluster entry point, but only these "
              "four are instantiated for the cluster path so far: 0 = prefill "
              "128x128x256, 1 = decode 16x32x256, 4 = decode 16x64x256 6-wave, "
              "13 = kid0 with A's scale staged in LDS. Add more by extending "
              "OPUS_BMM_BPRESHUF_CC_INST in "
              "opus_bmm_a8w8_mxscale_bpreshuffle_gfx1250.cu and this switch.");

  // Instantiated cluster shapes. splitK up to 8; mClusterWg 2 only on the
  // prefill tiles, which are the only ones that can have >1 M-tile at a shape
  // anyone runs. See the header note on why an unlisted pair throws.
  AITER_CHECK(splitK == 1 || splitK == 2 || splitK == 4 || splitK == 8,
              "opus_bmm_a8w8_mxscale_bpreshuffle_clusterclaunch: splitK must be "
              "1, 2, 4 or 8 (got ", splitK, "); it is a cluster dimension and "
              "so is compile-time");
  AITER_CHECK(mClusterWg == 1 || mClusterWg == 2,
              "opus_bmm_a8w8_mxscale_bpreshuffle_clusterclaunch: mClusterWg "
              "must be 1 or 2 (got ", mClusterWg, ")");
  AITER_CHECK(splitK * mClusterWg <= 16,
              "opus_bmm_a8w8_mxscale_bpreshuffle_clusterclaunch: splitK * "
              "mClusterWg must be <= 16 (the cluster workgroup budget); got ",
              splitK, " * ", mClusterWg);

  aiter_tensor_t* p_ws   = opus_bmm_cc_detail::opt(ws);
  aiter_tensor_t* p_bias = opus_bmm_cc_detail::opt(bias);
  const int m = (int)O.size(0);

  // DataWs is fp32 for BOTH C dtypes -- MEASURED, not assumed. Tying it to the C
  // dtype (bf16 C -> bf16 partials) looks free because the reduce accumulator is
  // fp32 either way and the max error barely moves (15 -> 17 on a tile whose
  // absmax is 4924, under one bf16 ULP). It is not free: a cell whose FINAL value
  // is near zero but whose PARTIALS are large has each partial rounded at the
  // partial's magnitude, and the cancellation leaves an absolute error the final
  // magnitude cannot absorb. At splitK 2/4/8 that put 0.05%-1.3% of cells outside
  // this project's own atol=0.5 / rtol=0.02 gate, while fp32 partials are
  // bit-exact against an fp64 reference at every splitK.
  //
  // The cost is 2x workspace bytes. To trade it back for bandwidth on a tile
  // where the accuracy is known to be acceptable, change bf16_t here AND in
  // OPUS_BMM_BPRESHUF_CC_ONE (both must agree -- the launcher validates the
  // workspace in BYTES and will throw if they do not).
  #define OPUS_BMM_CC_LAUNCH_ONE(TILE, SK, MC)                                \
    do {                                                                      \
      opus_bmm_cc_detail::warn_if_mcwg_wasted<TILE<bf16_t>>(m, MC, kernelId);  \
      if (Y.dtype() == AITER_DTYPE_bf16)                                      \
        opus_bmm_a8w8_mxscale_bpreshuffle_clusterclaunch_launch_gfx1250<      \
            TILE<bf16_t>, SK, fp32_t, MC, bf16_t>(                            \
            O, wo_a, Y, x_scale, w_scale, p_ws, p_bias, splitK);              \
      else                                                                    \
        opus_bmm_a8w8_mxscale_bpreshuffle_clusterclaunch_launch_gfx1250<      \
            TILE<fp32_t>, SK, fp32_t, MC, fp32_t>(                            \
            O, wo_a, Y, x_scale, w_scale, p_ws, p_bias, splitK);              \
    } while (0)

  // mClusterWg=2 is instantiated for the prefill tiles only (kid 0 / 13); the
  // decode tiles get mClusterWg=1, which is all a 1-M-tile shape can use.
  #define OPUS_BMM_CC_DISPATCH_SK(TILE, MC)                                   \
    do {                                                                      \
      switch (splitK) {                                                       \
        case 1: OPUS_BMM_CC_LAUNCH_ONE(TILE, 1, MC); break;                   \
        case 2: OPUS_BMM_CC_LAUNCH_ONE(TILE, 2, MC); break;                   \
        case 4: OPUS_BMM_CC_LAUNCH_ONE(TILE, 4, MC); break;                   \
        default: OPUS_BMM_CC_LAUNCH_ONE(TILE, 8, MC); break;                  \
      }                                                                       \
    } while (0)

  #define OPUS_BMM_CC_DISPATCH_PREFILL(TILE)                                  \
    do {                                                                      \
      if (mClusterWg == 2) OPUS_BMM_CC_DISPATCH_SK(TILE, 2);                  \
      else                 OPUS_BMM_CC_DISPATCH_SK(TILE, 1);                  \
    } while (0)

  #define OPUS_BMM_CC_DISPATCH_DECODE(TILE)                                   \
    do {                                                                      \
      AITER_CHECK(mClusterWg == 1,                                            \
                  "opus_bmm_a8w8_mxscale_bpreshuffle_clusterclaunch: kernelId=",\
                  kernelId, " is a decode tile (B_M=16) and is instantiated "  \
                  "for mClusterWg=1 only; a 1-M-tile shape has no peers to "   \
                  "multicast B to");                                          \
      OPUS_BMM_CC_DISPATCH_SK(TILE, 1);                                       \
    } while (0)

  switch (kernelId) {
    case 1:
      OPUS_BMM_CC_DISPATCH_DECODE(
          opus_bmm_a8w8_mxscale_bpreshuffle_tile_dec_n32_gfx1250);
      break;
    case 4:
      OPUS_BMM_CC_DISPATCH_DECODE(
          opus_bmm_a8w8_mxscale_bpreshuffle_tile_dec_n64_w6_gfx1250);
      break;
    case 13:
      OPUS_BMM_CC_DISPATCH_PREFILL(
          opus_bmm_a8w8_mxscale_bpreshuffle_tile_sfa_gfx1250);
      break;
    default:
      OPUS_BMM_CC_DISPATCH_PREFILL(
          opus_bmm_a8w8_mxscale_bpreshuffle_tile_gfx1250);
      break;
  }

  #undef OPUS_BMM_CC_DISPATCH_DECODE
  #undef OPUS_BMM_CC_DISPATCH_PREFILL
  #undef OPUS_BMM_CC_DISPATCH_SK
  #undef OPUS_BMM_CC_LAUNCH_ONE
#endif  // OPUS_BUILD_HAS_GFX1250
}

#endif  // !__HIP_DEVICE_COMPILE__
