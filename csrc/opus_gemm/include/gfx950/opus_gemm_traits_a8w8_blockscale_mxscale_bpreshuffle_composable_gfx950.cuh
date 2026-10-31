// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "opus_gemm_traits_a8w8_blockscale_mxscale_bpreshuffle_4wave_pin_agpr_gfx950.cuh"

// Geometry keeps M runtime. FixedN=0 also keeps N runtime; a nonzero FixedN
// specializes the column stride independently of the loading policy. Wk is one.
template<int BM,
          int BN,
          int WM,
          int WN,
          int BK,
          int Stages,
          int TotalK,
          int SplitK,
          int FixedN = 0>
struct opus_gemm_composable_geometry
{
    // Launch geometry.
    static constexpr int WG_PER_CU  = 1;
    static constexpr int BLOCK_SIZE = WM * WN * 64;

    // Matrix and wave tiles.
    static constexpr int B_M        = BM;
    static constexpr int B_N        = BN;
    static constexpr int B_K        = BK;
    static constexpr int T_M        = WM;
    static constexpr int T_N        = WN;
    static constexpr int T_K        = 1;
    static constexpr int E_M        = BM / (16 * WM);
    static constexpr int E_N        = BN / (16 * WN);

    // K partition and scheduling.
    static constexpr int TOTAL_K    = TotalK;
    static constexpr int SPLIT_K    = SplitK;
    static constexpr int FIXED_N    = FixedN;
    static constexpr int K          = TotalK / SplitK;
    static constexpr int KS         = BK / 128;
    static constexpr int SC         = K / 128;
    static constexpr int STAGES     = Stages;
    static constexpr int INITIAL    = Stages - 1;
    static constexpr int PANELS     = K / BK;

    static_assert(BM > 0 && BN > 0 && WM > 0 && WN > 0 && WM * WN <= 8);
    static_assert(BM % (16 * WM) == 0 && BN % (16 * WN) == 0);
    static_assert(BK == 128 || BK == 256 || BK == 512);
    static_assert(SplitK > 0);
    static_assert(TotalK > 0 && TotalK % (SplitK * BK) == 0);
    static_assert(Stages >= 2 && PANELS >= INITIAL);
    static_assert(FixedN >= 0 && (FixedN == 0 || FixedN % BN == 0));
};

// Loading, wave ownership and MFMA orientation are independent policies.
// Direct B uses an explicitly unrolled register ring; no LDS storage for B.
template<bool Compact            = true,
          bool PerStage           = true,
          bool BDirect            = false,
          int BAhead              = 1,
          int BCache              = 0,
          bool AsyncGroups        = true,
          bool ContiguousWaves    = true,
          bool NormalMFMA         = true,
          bool SimpleScaleAddress = false,
          int Output              = 3,
          int AddressMode         = 0,
          bool XcdOrder           = false>
struct opus_gemm_composable_policy
{
    static constexpr bool COMPACT = Compact;
    static constexpr bool PER_STAGE = PerStage;
    static constexpr bool B_DIRECT = BDirect;
    static constexpr bool ASYNC_GROUPS = AsyncGroups;
    static constexpr bool CONTIGUOUS = ContiguousWaves;
    static constexpr bool NORMAL = NormalMFMA;
    static constexpr bool SIMPLE_SCALE = SimpleScaleAddress;
    static constexpr int ADDRESS    = AddressMode;
    static constexpr bool XCD_ORDER = XcdOrder;

    // Direct-B lookahead and output policy.
    static constexpr int B_AHEAD    = BAhead;
    static constexpr int B_CACHE    = BCache;
    static constexpr int OUTPUT     = Output;

    static_assert(BCache == 0 || BCache == 1 || BCache == 2 || BCache == 17);
    static_assert(Output >= 0 && Output <= 3);
    static_assert(AddressMode >= 0 && AddressMode <= 2);
};

// Single LDS planner shared by both loading paths and the common epilogue.
template<class Geometry, class Policy>
struct opus_gemm_composable_traits : Geometry, Policy
{
    using G                        = Geometry;
    using P                        = Policy;
    static constexpr int DMA_BYTES = G::BLOCK_SIZE * 16;
    static constexpr int A_BYTES   = ((G::B_M * G::B_K + DMA_BYTES - 1) / DMA_BYTES) * DMA_BYTES;
    static constexpr int B_BYTES =
        P::B_DIRECT ? 0 : ((G::B_N * G::B_K + DMA_BYTES - 1) / DMA_BYTES) * DMA_BYTES;
    static constexpr int A_COPIES      = A_BYTES / DMA_BYTES;
    static constexpr int B_COPIES      = B_BYTES / DMA_BYTES;
    static constexpr int MATRIX_ISSUES = A_COPIES + B_COPIES;

    // Scale copies and storage.
    static constexpr int SCALE_COPY    = P::COMPACT ? 4 : 1;
    static constexpr int SCALE_COLS    = P::PER_STAGE ? G::KS : G::SC;
    static constexpr int SA_COPIES =
        (G::B_M * SCALE_COLS + G::BLOCK_SIZE * SCALE_COPY - 1) / (G::BLOCK_SIZE * SCALE_COPY);
    static constexpr int SA_BYTES    = SA_COPIES * G::BLOCK_SIZE * 4;
    static constexpr int SB_ROWS     = (G::B_N + 127) / 128;
    static constexpr int SB_COPIES   = (SB_ROWS * SCALE_COLS + G::BLOCK_SIZE - 1) / G::BLOCK_SIZE;
    static constexpr int SCALE_BYTES = SA_BYTES + SB_COPIES * G::BLOCK_SIZE * 4;

    // LDS offsets and total allocation.
    static constexpr int STAGE_BYTES = A_BYTES + B_BYTES + (P::PER_STAGE ? SCALE_BYTES : 0);
    static constexpr int OFF_SA      = P::PER_STAGE ? A_BYTES + B_BYTES : G::STAGES * STAGE_BYTES;
    static constexpr int OFF_SB      = OFF_SA + SA_BYTES;
    static constexpr int MAIN_BYTES  = G::STAGES * STAGE_BYTES + (P::PER_STAGE ? 0 : SCALE_BYTES);
    static constexpr int C_BYTES     = G::B_M * (G::B_N + 8) * 2;
    static constexpr int LDS_BYTES   = MAIN_BYTES > C_BYTES ? MAIN_BYTES : C_BYTES;

    // Register-ring scheduling.
    static constexpr int B_SETS      = P::B_DIRECT ? P::B_AHEAD + 1 : 1;
    static constexpr int UNROLL      = P::B_DIRECT ? B_SETS : 1;

    static_assert(G::B_N >= 128 ? G::B_N % 128 == 0 : 128 % G::B_N == 0);
    static_assert(LDS_BYTES + 16 <= 160 * 1024);
    static_assert(!P::B_DIRECT || (P::ASYNC_GROUPS && P::B_AHEAD >= 1 && P::B_AHEAD < G::STAGES));
    static_assert(P::ASYNC_GROUPS || P::PER_STAGE);
};

// M-first traversal is useful when neighbouring workgroups reuse B panels.
// Keep it independent of the loading/compute policy and the default traversal.
template<class Base>
struct opus_gemm_composable_mfirst_traits : Base
{
    static constexpr bool M_FIRST = true;
    static_assert(Base::SPLIT_K == 1);
};
