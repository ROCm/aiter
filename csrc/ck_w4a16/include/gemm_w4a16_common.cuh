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
inline constexpr ck::index_t KPerBlock = 32;

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
          bool PreDequantToLDS  = false>
struct DeviceGemmInstanceImpl;

// PreDequantToLDS = false : the fused-dequant baseline.
//
// AIESW-32282: BDequantOp (the trailing template slot on
// DeviceGemm_BScale_Wmma_CShuffleV3) carries the dequant element-op down to
// the CK threadwise transfer at the pk_i4 dequant call sites. CK's
// DequantPolicyFor<> trait in unary_element_wise_operation.hpp maps the
// asym carrier to the matching (sym, asym) pair both branches need to
// compile in the same v1 pipeline instantiation. The BElementwiseOperation
// slot is left as PassThrough because the gridwise also applies it at the
// B global->LDS copy (ThreadwiseTensorSliceTransfer_v3r1) which expects a
// 2-arg operator().
template <typename T, ck::index_t ScaleBlockK>
struct DeviceGemmInstanceImpl<T, ScaleBlockK,
                              /*PreDequantToLDS=*/false> {
  using BDequantOp = DequantPack8WithZp;
  using type = ck::tensor_operation::device::DeviceGemm_BScale_Wmma_CShuffleV3<
      // ---- Tensor layouts (A: Row, B: Col = K-major, C: Row) ----
      ALayout,
      BLayout,
      CLayout,
      // ---- Element dtypes ----
      T,                                        // ADataType         (activation: F16 or B16)
      BDataType,
      T,                                        // BScaleDataType    (per-group scale, activation dtype)
      T,                                        // CDataType         (output, activation dtype)
      AccDataType,
      T,                                        // CShuffleDataType  (C-shuffle staging, activation dtype)
      // ---- Elementwise ops applied at A/B global->LDS copy and C epilogue ----
      PassThrough,                              // AElementwiseOperation
      PassThrough,                              // BElementwiseOperation  (B global->LDS copy; dequant
                                                //                         element-op is in BDequantOp slot)
      PassThrough,                              // CElementwiseOperation
      // ---- GemmSpec: no padding (caller guarantees aligned M/N/K) ----
      GemmDefault,                              // GemmSpec
      // ---- Block / wave-group sizing ----
      256,                                      // BlockSize         (threads per wave-group = 4 waves of 64)
      Scale_Block_N,                            // ScaleBlockN       (1 = per-N-column scale)
      ScaleBlockK,
      // ---- Block-level tile (M x N x K) per wave-group ----
      128,                                      // MPerBlock
      128,                                      // NPerBlock
      KPerBlock,
      8,                                        // AK1               (A load width: 8 elements per access)
      8,                                        // BK1               (B load width: 8 nibbles per access)
      // ---- WMMA tile (RDNA3 fp16/bf16 native WMMA is 16x16x16) ----
      16,                                       // MPerWmma
      16,                                       // NPerWmma
      // ---- Per-wave WMMA repeat count (wave-tile = 4*16 x 2*16 = 64x32) ----
      4,                                        // MRepeat
      2,                                        // NRepeat
      // ---- A global->LDS copy descriptors ----
      S<4, 64, 1>,                              // ABlockTransferThreadClusterLengths_AK0_M_AK1 (4*64*1 = 256 = BlockSize)
      S<1, 0, 2>,                               // ABlockTransferThreadClusterArrangeOrder
      S<1, 0, 2>,                               // ABlockTransferSrcAccessOrder
      2,                                        // ABlockTransferSrcVectorDim                  (= AK1 axis)
      8,                                        // ABlockTransferSrcScalarPerVector
      8,                                        // ABlockTransferDstScalarPerVector_AK1
      1,                                        // ABlockLdsExtraM                             (LDS pad row to avoid bank conflicts)
      // ---- B global->LDS copy descriptors (mirror A layout) ----
      S<4, 64, 1>,                              // BBlockTransferThreadClusterLengths_BK0_N_BK1
      S<1, 0, 2>,                               // BBlockTransferThreadClusterArrangeOrder
      S<1, 0, 2>,                               // BBlockTransferSrcAccessOrder
      2,                                        // BBlockTransferSrcVectorDim
      8,                                        // BBlockTransferSrcScalarPerVector
      8,                                        // BBlockTransferDstScalarPerVector_BK1
      1,                                        // BBlockLdsExtraN
      // ---- C wave-shuffle epilogue ----
      1,                                        // CShuffleMRepeatPerShuffle
      1,                                        // CShuffleNRepeatPerShuffle
      S<1, 32, 1, 8>,                           // CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock
      8,                                        // CShuffleBlockTransferScalarPerVector_NPerBlock
      // ---- Pipeline scheduler + version (Interwave v1 = sym+asym pk_i4 dequant) ----
      ck::BlockGemmPipelineScheduler::Interwave, // BlkGemmPipeSched
      ck::BlockGemmPipelineVersion::v1,         // BlkGemmPipelineVer
      // ---- Compute dtypes (no fp32 promotion in inner loop on fp16; bf16 falls back internally) ----
      T,                                        // ComputeTypeA
      T,                                        // ComputeTypeB
      // ---- Layout permute flags ----
      PermuteA,
      PermuteB,
      // ---- AIESW-32282: dequant element-op slot (carrier; trait derives the sym/asym pair) ----
      BDequantOp>;
};

// PreDequantToLDS = true : pre-dequant-to-LDS variant. Currently uses the
// same CK device-op as the false specialization so the template machinery
// compiles cleanly and the test/bench surface is wired end-to-end. The
// runtime dispatcher in gemm_w4a16.cu rejects this path with a TORCH_CHECK
// until the real two-stage / forked-gridwise kernel lands.
//
// TODO(AIESW-32282): implement the actual pre-dequant-to-LDS pipeline. The
// two most plausible strategies are:
//   1. Two-stage kernel: a CK custom kernel that loads packed B + scales
//      (+ optional scaled_zp) from global, dequants into bf16/fp16 in
//      registers, and stores to an LDS scratch region. Stage 2 invokes
//      the standard DeviceGemm_Wmma_CShuffleV3 (non-b_scale variant)
//      reading dequantized B from LDS. Simpler, but pays an extra LDS
//      write+read round-trip.
//   2. Forked gridwise pipeline variant: extend (or fork)
//      GridwiseGemm_wmma_cshuffle_v3_ab_scale so the dequant step runs
//      once before the WMMA loop, materializing bf16/fp16 into an LDS
//      region that the existing B-load path then reads. Larger CK-side
//      change but no extra global-LDS traffic.
//
// Mind RDNA3.5's 64 KB LDS/CU: current per-stage B LDS footprint is
// NPerBlock * KPerBlock * sizeof(BDataType@packed) = 128 * 32 / 2 bytes
// (= 2 KB) and grows to 128 * 32 * sizeof(T) = 8 KB per stage when
// dequantized — should still fit alongside the A tile but check occupancy
// before declaring done.
template <typename T, ck::index_t ScaleBlockK>
struct DeviceGemmInstanceImpl<T, ScaleBlockK,
                              /*PreDequantToLDS=*/true> {
  // Type aliases match the PreDequantToLDS=false specialization so the
  // dispatcher and op_tests link; the runtime path is guarded by a
  // TORCH_CHECK in gemm_w4a16.cu.
  using BDequantOp =
      typename DeviceGemmInstanceImpl<T, ScaleBlockK, false>::BDequantOp;
  using type =
      typename DeviceGemmInstanceImpl<T, ScaleBlockK, false>::type;
};

template <typename T,
          ck::index_t ScaleBlockK,
          bool PreDequantToLDS  = false>
using DeviceGemmInstance =
    typename DeviceGemmInstanceImpl<T, ScaleBlockK, PreDequantToLDS>::type;
// clang-format on

}  // namespace ck_w4a16
