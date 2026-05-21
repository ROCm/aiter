// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// AIESW-32176: shared types + device-op instance for the gfx1151 CK W4A16
// b_scale GEMM. Mirrors the layout of csrc/ck_gemm_a4w4_blockscale/include/
// (typedefs hoisted out of the dispatcher .cu).
#pragma once

#ifdef USE_ROCM
#undef __HIP_NO_HALF_OPERATORS__
#undef __HIP_NO_HALF_CONVERSIONS__
#endif

#include <torch/all.h>
#include <torch/extension.h>
#ifdef USE_ROCM
#include <ATen/hip/HIPContext.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <ATen/hip/impl/HIPStreamMasqueradingAsCUDA.h>
#else
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#endif

#include "ck/ck.hpp"
#include "ck/stream_config.hpp"
#include "ck/utility/common_header.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/tensor_operation/gpu/element/unary_element_wise_operation.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_gemm_wmma_cshuffle_v3_b_scale.hpp"

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

// aiter-style type aliases (match csrc/ck_gemm_a4w4_blockscale/include/...)
using F16 = ck::half_t;
using B16 = ck::bhalf_t;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;
using PassThrough = ck::tensor_operation::element_wise::PassThrough;
using DequantPack8WithZp =
    ck::tensor_operation::element_wise::DequantPack8WithZp;
// AIESW-32735 research-only: symmetric int4 dequant carrier (no zp subtract).
// Used by the Baseline_Sym_V1 / Baseline_Sym_V3 tile configs to test the v3
// (Intrawave, deeper K-prefetch) pipeline, which lacks a BElementOpAsym slot.
using DequantPack8 =
    ck::tensor_operation::element_wise::DequantPack8;
// AIESW-32735: folded-dequant carrier. Same 4-arg signature as
// DequantPack8WithZp; the 4th arg is bias_eff = -8*scale - scaled_zp
// (precomputed at weight load) so the dequant collapses to 1 packed-fp16 FMA
// per nibble — same FMA count as the sym path. See v3-sym report:
// /scratch/mgehre/tmp/gemma2b_downproj_v3_sym_report.md.
using DequantPack8WithBias =
    ck::tensor_operation::element_wise::DequantPack8WithBias;
// AIESW-32735 B'': packed-(scale,bias_eff) carrier. ONE per-group fp32 load
// (vs the two per-group loads the bias_eff carrier still needs because the
// gridwise's MakeBScale is invoked separately for scale and zero-point grids).
// Routes through the sym branch of the v1 pipeline (HasBZp=false) so the
// threadwise transfer only loads one buffer; the dequant op bit-extracts both
// scale (low 16 bits) and bias_eff (high 16 bits) from the fp32 carrier.
using DequantPack8WithPackedScaleBias =
    ck::tensor_operation::element_wise::DequantPack8WithPackedScaleBias;

namespace ck_w4a16 {

// Weight + accumulator dtypes are fixed; activation / scale / output dtype is
// templated (T = F16 or B16) to mirror the F16/B16 dispatch pattern in
// csrc/ck_gemm_a4w4_blockscale/.
using BDataType   = ck::pk_i4_t;
using AccDataType = float;

using ALayout = Row;
using BLayout = Col;
using CLayout = Row;

inline constexpr auto GemmDefault =
    ck::tensor_operation::device::GemmSpecialization::Default;
inline constexpr bool PermuteA = false;
inline constexpr bool PermuteB = true;
inline constexpr ck::index_t Scale_Block_N = 1;

// EXP1_FINAL config from gfx1151 sweep (30.0 TFLOPS verified at gate_up_proj
// M=3968 N=19456 K=2560). Holds 28-31 TFLOPS uniformly across M=256-16384 on
// the same column. Same kernel handles all four Qwen3-4B prefill linear
// columns at runtime.
//
// AIESW-32735: KPerBlock is now per-tile-config (see TileConfigKind below).
// The constant below is kept as the *baseline* value so existing call sites
// (re-pack arithmetic in op_tests / vLLM weight loader) keep using K=32.
// The active KPerBlock for a given dispatch is decided at template-instance
// selection time by `DeviceGemmInstanceImpl<T, ScaleBlockK, PreDequantToLDS,
// Tile>::kKPerBlock`.
inline constexpr ck::index_t KPerBlock = 32;

// Tile-config kinds for the Phase-2 tuning experiment. Each kind maps to a
// fixed (MPerBlock, NPerBlock, KPerBlock, MRepeat, NRepeat,
// ABlockTransferThreadClusterLengths, BBlockTransferThreadClusterLengths)
// tuple via partial specialization of DeviceGemmInstanceImpl. BlockSize is
// held at 256 across all kinds so the cluster-length asserts (cluster product
// == BlockSize) stay satisfied.
enum class TileConfigKind : int {
  Baseline      = 0,   // Interwave v1, M=128 N=128 K=32  MRep=4 NRep=2  (current default)
  WideM         = 1,   // Interwave v1, M=256 N=128 K=32  MRep=8 NRep=2  (Phase-2: lost, VGPR cliff)
  LargeK        = 2,   // Interwave v1, M=128 N=128 K=64  MRep=4 NRep=2  (Phase-2: lost, VGPR cliff)
  WideM_LargeK  = 3,   // Interwave v1, M=256 N=128 K=64  MRep=8 NRep=2  (Phase-2: lost, VGPR cliff)
  // Smaller-tile variants (BlockSize=128) — target more WGs/CU to mask vmem latency
  // via inter-WG, not inter-wave, occupancy. Each WG has fewer waves but
  // is much lighter so many can coexist on a CU.
  Tile64        = 4,   // Interwave v1, M=64 N=64 K=32   MRep=2 NRep=2 BlockSize=128
  Tile64_LargeK = 5,   // Interwave v1, M=64 N=64 K=64   MRep=2 NRep=2 BlockSize=128
  NarrowN       = 6,   // Interwave v1, M=128 N=64 K=32  MRep=4 NRep=2 BlockSize=128 (1/2 N-tile)
  // AIESW-32735 research-only: probe whether the CK v3 (Intrawave, deeper
  // K-prefetch) pipeline buys anything on the AWQ shape relative to v1. v3
  // ignores the BZeroPointStruct slot (see static_assert in
  // blockwise_gemm_pipeline_wmmaops_v3.hpp) — i.e. v3 is only correct on the
  // *symmetric* int4 path (scaled_zp == nullptr). The Baseline_Sym_V1 entry
  // is a control to isolate the cost of just swapping the asym dequant op
  // carrier out of the carrier slot; Baseline_Sym_V3 is the experimental
  // arm. Same tile shape (M=128 N=128 K=32 MRep=4 NRep=2) as Baseline.
  Baseline_Sym_V1 = 7,
  Baseline_Sym_V3 = 8,
  // AIESW-32735: folded-dequant. Same tile/pipeline as Baseline (Interwave
  // v1, M=128 N=128 K=32 MRep=4 NRep=2 BlockSize=256), but uses the
  // DequantPack8WithBias carrier so the asym dequant collapses to 1 FMA per
  // nibble instead of (mul + sub). Caller passes precomputed bias_eff in
  // place of scaled_zp (the runtime aiter wrapper documents this — the 4-arg
  // CK signature is unchanged). Target shape is Gemma-2B down_proj (K=16384)
  // where the v3-sym report measured +28.5% headroom from removing the
  // per-group subtract.
  Baseline_Bias = 9,
  // AIESW-32735 B'': packed-(scale,bias_eff) variant. Same tile/pipeline as
  // Baseline (Interwave v1, M=128 N=128 K=32 MRep=4 NRep=2 BlockSize=256),
  // but the gridwise's BScaleType is `float` (fp32 carrier holding fp16/bf16
  // scale in the low 16 bits and bias_eff in the high 16 bits) and the dequant
  // carrier is DequantPack8WithPackedScaleBias. This halves the per-group
  // global-load count vs Baseline_Bias because we ONLY load one buffer (the
  // packed one) instead of two (scale + bias_eff). The host MUST pass the
  // packed buffer in the `in_s` (scale) slot and leave `scaled_zp` as nullptr
  // — the runtime dispatch enforces this (see gemm_w4a16.cu).
  Baseline_PackedSb = 10,
};

// Supported per-group scale-block-K values. ScaleBlockK == KPerBlock (32) gives
// the AWQ group_size=32 variant (one scale per K tile); ScaleBlockK == 128
// gives the AWQ group_size=128 variant (one scale per 4 K tiles). CK's
// gridwise pipeline only requires ScaleBlockK >= KPerWmma (= 16 on RDNA3 WMMA
// fp16), so both 32 and 128 satisfy the static_assert. Adding more values
// would just instantiate another kernel copy; the tile / pipeline params are
// dtype- and ScaleBlockK-independent.

// Templated device-op instance — T is fp16 or bf16 (activation = scale =
// shuffle = output dtype). Tile / pipeline params are dtype- and
// ScaleBlockK-independent.
//
// PreDequantToLDS selects between two implementations:
//   - false (default): the existing fused-dequant path. CK's
//     DeviceGemm_BScale_Wmma_CShuffleV3 dequants int4 inside the WMMA
//     inner loop (per-WMMA-tile VGPR materialization). Per-nibble dequant
//     cost on RDNA 3.5 is dominated by IEEE-correct round-to-bf16
//     (v_add3_u32 + v_cmp_o_f32 + v_cndmask_b16 — see
//     vllm4/notes/ck-w4a16-isa/README.md).
//   - true: pre-dequant-to-LDS. Pay dequant once per K-block into a
//     bf16/fp16 LDS scratch region, then WMMA reads activation-dtype B
//     from LDS directly. Trades ~2x LDS B-tile pressure for amortized
//     dequant cost. Expected to close most of the bf16 CK vs Triton
//     gap on Qwen3-8B-quantized.w4a16 if LDS pressure doesn't break
//     occupancy. Currently STUBBED — see DeviceGemmInstance<...,true>
//     specialization below and TODO(AIESW-32282).
//
// Bf16 dequant rounding: truncate-to-bf16 is now the only behavior. The
// IEEE-correct round-to-nearest-even path was retired after lm_eval verified
// truncate is statistically indistinguishable from Triton on gsm8k 5-shot
// (Orion-zhen/Qwen3-1.7B-AWQ n=500, McNemar p=1.000) and TTFT on Qwen3-8B-
// quantized.w4a16 showed truncate is the only setting where CK beats Triton
// on bf16. The choice is baked into DequantPack8 / DequantPack8WithZp in CK
// (see [CK] AIESW-32282 commit on matthias.gfx11_ck / matthias/threadwise-
// element-op-template), so there's no runtime/template axis to flip.
//
// clang-format off
template <typename T,
          ck::index_t ScaleBlockK,
          bool PreDequantToLDS  = false,
          TileConfigKind Tile   = TileConfigKind::Baseline>
struct DeviceGemmInstanceImpl;

// AIESW-32735: macro that expands to a DeviceGemmInstanceImpl specialization
// for (PreDequantToLDS=false, <Tile>), parametrized by tile-config tuple.
// Holds BlockSize=256, AK1=BK1=8, WMMA tile 16x16x16, Interwave v1 schedule,
// CShuffle epilogue / shuffle params, and the dequant-op carrier constant.
//
// Per-tile cluster lengths must satisfy:
//   AK0 = KPerBlock / AK1 = KPerBlock / 8
//   ABlockTransferThreadClusterLengths = S<AK0, BlockSize/AK0, 1>
// and analogously for B. With BlockSize=256:
//   KPerBlock=32 -> AK0=4 -> cluster S<4, 64, 1>
//   KPerBlock=64 -> AK0=8 -> cluster S<8, 32, 1>
//
// Per-tile CShuffleBlockTransferClusterLengths must shard NPerBlock across
// the cluster's N dim (= 8 here) so 8 * scalar_per_vec(=8) = 64 ≤ NPerBlock.
// We keep S<1, 32, 1, 8> for NPerBlock=128 across all tiles (NPerBlock fixed
// in this experiment), so the C-shuffle params stay valid.
#define _CK_W4A16_DEFINE_TILE_FULL_DEQ(_TILE_KIND_,                                         \
                                       _BLOCKSIZE_,                                         \
                                       _MPB_, _NPB_, _KPB_,                                 \
                                       _MREP_, _NREP_,                                      \
                                       _ACLUSTER_K_, _ACLUSTER_M_,                          \
                                       _BCLUSTER_K_, _BCLUSTER_N_,                          \
                                       _CSHUF_N_,                                           \
                                       _PIPE_SCHED_, _PIPE_VER_,                            \
                                       _DEQ_OP_)                                            \
  template <typename T, ck::index_t ScaleBlockK>                                            \
  struct DeviceGemmInstanceImpl<T, ScaleBlockK,                                             \
                                false,                                                      \
                                _TILE_KIND_> {                                              \
    static constexpr ck::index_t kMPerBlock = _MPB_;                                        \
    static constexpr ck::index_t kNPerBlock = _NPB_;                                        \
    static constexpr ck::index_t kKPerBlock = _KPB_;                                        \
    using BDequantOp = _DEQ_OP_;                                                            \
    using type = ck::tensor_operation::device::DeviceGemm_BScale_Wmma_CShuffleV3<           \
        ALayout, BLayout, CLayout,                                                          \
        T, BDataType, T, T, AccDataType, T,                                                 \
        PassThrough, PassThrough, PassThrough,                                              \
        GemmDefault,                                                                        \
        _BLOCKSIZE_,                                                                        \
        Scale_Block_N, ScaleBlockK,                                                         \
        _MPB_, _NPB_, _KPB_,                                                                \
        8, 8,                                                                               \
        16, 16,                                                                             \
        _MREP_, _NREP_,                                                                     \
        S<_ACLUSTER_K_, _ACLUSTER_M_, 1>,                                                   \
        S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, true,                                              \
        S<_BCLUSTER_K_, _BCLUSTER_N_, 1>,                                                   \
        S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, true,                                              \
        1, 1,                                                                               \
        S<1, 32, 1, _CSHUF_N_>, 8,                                                          \
        ck::BlockGemmPipelineScheduler::_PIPE_SCHED_,                                       \
        ck::BlockGemmPipelineVersion::_PIPE_VER_,                                           \
        T, T,                                                                               \
        PermuteA, PermuteB,                                                                 \
        BDequantOp>;                                                                        \
  }

// Default form keeps the asymmetric AWQ carrier (DequantPack8WithZp) so the
// v1 pipeline resolves the (sym, asym) pair via DequantPolicyFor<>.
#define _CK_W4A16_DEFINE_TILE_FULL(_TILE_KIND_,                                             \
                                   _BLOCKSIZE_,                                             \
                                   _MPB_, _NPB_, _KPB_,                                     \
                                   _MREP_, _NREP_,                                          \
                                   _ACLUSTER_K_, _ACLUSTER_M_,                              \
                                   _BCLUSTER_K_, _BCLUSTER_N_,                              \
                                   _CSHUF_N_,                                               \
                                   _PIPE_SCHED_, _PIPE_VER_)                                \
  _CK_W4A16_DEFINE_TILE_FULL_DEQ(_TILE_KIND_,                                               \
                                 _BLOCKSIZE_,                                               \
                                 _MPB_, _NPB_, _KPB_,                                       \
                                 _MREP_, _NREP_,                                            \
                                 _ACLUSTER_K_, _ACLUSTER_M_,                                \
                                 _BCLUSTER_K_, _BCLUSTER_N_,                                \
                                 _CSHUF_N_,                                                 \
                                 _PIPE_SCHED_, _PIPE_VER_,                                  \
                                 DequantPack8WithZp)

// Shorthand for BlockSize=256 + (Interwave, v1) + CShuf_N=8 — original Phase-2 axis.
#define _CK_W4A16_DEFINE_TILE(_TILE_KIND_,                                                  \
                              _MPB_, _NPB_, _KPB_,                                          \
                              _MREP_, _NREP_,                                               \
                              _ACLUSTER_K_, _ACLUSTER_M_,                                   \
                              _BCLUSTER_K_, _BCLUSTER_N_)                                   \
  _CK_W4A16_DEFINE_TILE_FULL(_TILE_KIND_,                                                   \
                             256,                                                           \
                             _MPB_, _NPB_, _KPB_,                                           \
                             _MREP_, _NREP_,                                                \
                             _ACLUSTER_K_, _ACLUSTER_M_,                                    \
                             _BCLUSTER_K_, _BCLUSTER_N_,                                    \
                             8,                                                             \
                             Interwave, v1)

// Baseline tile (matches the previous unconditional instance exactly).
//   M=128, N=128, K=32, MRep=4, NRep=2; A-cluster S<4,64,1>; B-cluster S<4,64,1>.
_CK_W4A16_DEFINE_TILE(TileConfigKind::Baseline,
                      128, 128, 32, 4, 2,
                      4, 64, 4, 64);

// WideM: larger M tile, more output reuse per B-tile load. Cuts B-traffic
// roughly in half on tall-K narrow-N shapes (Gemma-2B down_proj).
_CK_W4A16_DEFINE_TILE(TileConfigKind::WideM,
                      256, 128, 32, 8, 2,
                      4, 64, 4, 64);

// LargeK: KPerBlock=64 -> AK0=8, cluster S<8,32,1>. Halves inner-K loop
// count, doubles bytes-per-global-access, gives the K pipeline more in-flight
// work to hide HBM latency. A LDS doubles (16 KB at M=128); B LDS doubles to
// 4 KB. Fits well under 64 KB/CU.
_CK_W4A16_DEFINE_TILE(TileConfigKind::LargeK,
                      128, 128, 64, 4, 2,
                      8, 32, 8, 32);

// WideM + LargeK combo. A LDS = 256*64*2 = 32 KB; may push past LDS budget
// with double-buffering. Build may fail; the dispatcher then refuses this
// config at runtime via the IsSupportedArgument check.
_CK_W4A16_DEFINE_TILE(TileConfigKind::WideM_LargeK,
                      256, 128, 64, 8, 2,
                      8, 32, 8, 32);

// AIESW-32735 Phase-2 revision: target the *vmem-latency* bottleneck identified
// in baseline_diagnosis.md. Larger tiles hit the 256-VGPR cap (down to ~4
// waves/SIMD) which makes latency hiding worse. CK WMMA only ships v1 and v3
// pipelines (selector is in blockwise_gemm_pipeline_wmma_selector.hpp), and
// v3 doesn't carry the BElementOpAsym dequant slot, so it cannot run the AWQ-
// asymmetric path. That leaves "more WGs per CU via smaller per-WG resources"
// as the remaining lever — fewer waves per WG, lower VGPR pressure, lower LDS
// per WG → many WGs co-resident on a CU, each making progress on independent
// (M,N) blocks so global-load latency is hidden across WGs instead of across
// waves of one WG.
//
// CShuffleBlockTransferClusterLengths is parameterised on cluster N count
// (S<1, MPerBlock/something, 1, CShufN>); we feed N=4 here so 4*8=32 lanes
// span NPerBlock=64 (this stays compatible with the 8 ScalarPerVector
// assumed in the macro body).
_CK_W4A16_DEFINE_TILE_FULL(TileConfigKind::Tile64,
                           128,                          // BlockSize
                           64, 64, 32, 2, 2,             // M, N, K, MRep, NRep
                           4, 32, 4, 32,                 // A-cluster, B-cluster
                           4,                            // CShuffle N-cluster
                           Interwave, v1);
_CK_W4A16_DEFINE_TILE_FULL(TileConfigKind::Tile64_LargeK,
                           128,
                           64, 64, 64, 2, 2,
                           8, 16, 8, 16,
                           4,
                           Interwave, v1);
_CK_W4A16_DEFINE_TILE_FULL(TileConfigKind::NarrowN,
                           128,
                           128, 64, 32, 4, 2,
                           4, 32, 4, 32,
                           4,
                           Interwave, v1);

// AIESW-32735 research-only: Baseline_Sym_V1 / Baseline_Sym_V3 — same tile
// shape as Baseline (M=128 N=128 K=32 MRep=4 NRep=2 BlockSize=256), but use
// the symmetric dequant carrier (DequantPack8) so the v3 pipeline (which
// ignores the asym slot) can be tested without static_assert. v3 sym is
// functionally correct ONLY when called with scaled_zp == nullptr (the v3
// pipeline drops the zp arg silently — see blockwise_gemm_pipeline_wmmaops_v3.hpp
// AIESW-32176 comment). Baseline_Sym_V1 isolates the cost of just swapping
// the carrier from DequantPack8WithZp to DequantPack8 under v1.
_CK_W4A16_DEFINE_TILE_FULL_DEQ(TileConfigKind::Baseline_Sym_V1,
                               256,
                               128, 128, 32, 4, 2,
                               4, 64, 4, 64,
                               8,
                               Interwave, v1,
                               DequantPack8);
_CK_W4A16_DEFINE_TILE_FULL_DEQ(TileConfigKind::Baseline_Sym_V3,
                               256,
                               128, 128, 32, 4, 2,
                               4, 64, 4, 64,
                               8,
                               Intrawave, v3,
                               DequantPack8);

// AIESW-32735: folded-dequant carrier. Tile shape and pipeline match Baseline
// (so the only difference vs Baseline is which BElementOpAsym the threadwise
// transfer ends up invoking — DequantPack8WithBias instead of
// DequantPack8WithZp). The 4-arg signature into the threadwise transfer is
// unchanged; the host wrapper passes bias_eff in the scaled_zp slot when this
// tile config is selected.
_CK_W4A16_DEFINE_TILE_FULL_DEQ(TileConfigKind::Baseline_Bias,
                               256,
                               128, 128, 32, 4, 2,
                               4, 64, 4, 64,
                               8,
                               Interwave, v1,
                               DequantPack8WithBias);

// AIESW-32735 B'': hand-written specialization for Baseline_PackedSb. Cannot
// use the standard macro because BScaleDataType must be `float` (not `T`) so
// the gridwise issues 4-byte per-group loads carrying both scale (low 16 bits)
// and bias_eff (high 16 bits). Caller's `in_s` slot holds the packed fp32
// buffer; `scaled_zp` MUST be nullptr so the v1 pipeline hits the sym branch
// (single per-group buffer load), and the dequant op
// DequantPack8WithPackedScaleBias bit-extracts both halves at the call site.
template <typename T, ck::index_t ScaleBlockK>
struct DeviceGemmInstanceImpl<T, ScaleBlockK, false, TileConfigKind::Baseline_PackedSb> {
  static constexpr ck::index_t kMPerBlock = 128;
  static constexpr ck::index_t kNPerBlock = 128;
  static constexpr ck::index_t kKPerBlock = 32;
  using BDequantOp = DequantPack8WithPackedScaleBias;
  using type = ck::tensor_operation::device::DeviceGemm_BScale_Wmma_CShuffleV3<
      ALayout, BLayout, CLayout,
      T, BDataType,
      /*BScaleDataType=*/float,   // <-- the key difference vs Baseline_Bias
      T, AccDataType, T,
      PassThrough, PassThrough, PassThrough,
      GemmDefault,
      256,
      Scale_Block_N, ScaleBlockK,
      128, 128, 32,
      8, 8,
      16, 16,
      4, 2,
      S<4, 64, 1>,
      S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, true,
      S<4, 64, 1>,
      S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, true,
      1, 1,
      S<1, 32, 1, 8>, 8,
      ck::BlockGemmPipelineScheduler::Interwave,
      ck::BlockGemmPipelineVersion::v1,
      T, T,
      PermuteA, PermuteB,
      BDequantOp>;
};

#undef _CK_W4A16_DEFINE_TILE
#undef _CK_W4A16_DEFINE_TILE_FULL
#undef _CK_W4A16_DEFINE_TILE_FULL_DEQ

// PreDequantToLDS = true : pre-dequant-to-LDS variant. Currently uses the
// same CK device-op as the false specialization so the template machinery
// compiles cleanly and the test/bench surface is wired end-to-end. The
// runtime dispatcher in gemm_w4a16.cu rejects this path with a TORCH_CHECK
// until the real two-stage / forked-gridwise kernel lands.
//
// TODO(AIESW-32282): implement the actual pre-dequant-to-LDS pipeline. See
// the comment in the previous revision of this file (git blame this region)
// for the two strategies considered (two-stage kernel vs forked gridwise).
template <typename T, ck::index_t ScaleBlockK, TileConfigKind Tile>
struct DeviceGemmInstanceImpl<T, ScaleBlockK,
                              /*PreDequantToLDS=*/true,
                              Tile> {
  // Type aliases match the PreDequantToLDS=false / Baseline specialization
  // so the dispatcher and op_tests link; the runtime path is guarded by a
  // TORCH_CHECK in gemm_w4a16.cu. Tile choice is ignored under this stub.
  using BDequantOp =
      typename DeviceGemmInstanceImpl<T, ScaleBlockK, false, TileConfigKind::Baseline>::BDequantOp;
  using type =
      typename DeviceGemmInstanceImpl<T, ScaleBlockK, false, TileConfigKind::Baseline>::type;
};

template <typename T,
          ck::index_t ScaleBlockK,
          bool PreDequantToLDS  = false,
          TileConfigKind Tile   = TileConfigKind::Baseline>
using DeviceGemmInstance =
    typename DeviceGemmInstanceImpl<T, ScaleBlockK, PreDequantToLDS, Tile>::type;
// clang-format on

}  // namespace ck_w4a16
