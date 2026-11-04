// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "opus_gemm_traits_a8w8_blockscale_mxscale_bpreshuffle_4wave_pin_agpr_gfx950.cuh"

#include <opus/opus.hpp>

// Shared block, wave, and MFMA geometry for the M-repeat LDS schedules.
template<int BlockM, int BlockN, int WaveM, int WaveN, int BlockK = 128>
struct opus_gemm_bpreshuffle_m_repeat_geometry {
    static constexpr int BLOCK_SIZE = WaveM * WaveN * 64;
    static constexpr int WARP_SIZE = 64;
    static constexpr int NUM_WAVES = BLOCK_SIZE / WARP_SIZE;

    static constexpr int B_M = BlockM;
    static constexpr int B_N = BlockN;
    static constexpr int B_K = BlockK;
    static constexpr int T_M = WaveM;
    static constexpr int T_N = WaveN;
    static constexpr int T_K = 1;
    static constexpr int W_M = 16;
    static constexpr int W_N = 16;
    static constexpr int W_K = 128;
    static constexpr int E_M = B_M / (T_M * W_M);
    static constexpr int E_N = B_N / (T_N * W_N);
    static constexpr int E_K = B_K / W_K;
};

// M repeats share an A fragment ring, B fragments, and two LDS matrix stages.
template<int BlockM = 192, int BlockN = 256, int ScalePanel = 128, int FixedK = 0,
    int WaveM = 4, int GroupM = 0, int CacheA = 0, int CacheB = 0,
    bool FragmentArrays = false,
    bool XcdOrder = false, bool BoundedOutput = false, int PublishM = 0, int GroupN = 8,
    bool ScaleOnRead = false>
struct opus_gemm_bpreshuffle_m_repeat_traits
    : opus_gemm_bpreshuffle_m_repeat_geometry<BlockM, BlockN, WaveM, 2> {
    using G = opus_gemm_bpreshuffle_m_repeat_geometry<BlockM, BlockN, WaveM, 2>;
    static constexpr bool MULTI_STAGE = false;
    static constexpr int MIN_WGS_PER_CU = 1;

    static constexpr int HALF_B_M = G::B_M;
    static constexpr int HALF_B_N = G::B_N;

    static_assert(G::NUM_WAVES == G::T_M * G::T_N * G::T_K);
    static_assert(G::B_M % (G::T_M * G::W_M) == 0 && G::B_N % (G::T_N * G::W_N) == 0);
    static_assert((G::B_M == 64 || G::B_M == 96 || G::B_M == 128 || G::B_M == 192 || G::B_M == 256) &&
        (G::B_N == 64 || G::B_N == 128 || G::B_N == 256));
    static_assert((WaveM == 2 || WaveM == 4) && G::T_N == 2);

    static constexpr int A_RING = ScaleOnRead ? G::E_M : (G::E_M < 3 ? G::E_M : 3);

    static_assert(G::E_M >= 2 && G::B_N >= G::NUM_WAVES * 16);

    // K specialization, tile grouping, and matrix cache policies.
    static constexpr int FIXED_K = FixedK;
    static constexpr int GROUPED_M = GroupM;
    static constexpr int CACHE_A = CacheA;
    static constexpr int CACHE_B = CacheB;
    static constexpr bool SCALE_ON_READ = ScaleOnRead;

    static_assert(GroupM == 0 || GroupM == 4 || GroupM == 8);
    static_assert(CacheA == 0 || CacheA == 1 || CacheA == 17);
    static_assert(CacheB == 0 || CacheB == 1 || CacheB == 17 || (ScaleOnRead && CacheB == 2));

    // Keep fragments in separate arrays when requested by the JIT configuration.
    static constexpr bool FRAGMENT_ARRAYS = FragmentArrays;

    static constexpr bool XCD_ORDER = XcdOrder;
    static constexpr bool BOUNDED_OUTPUT = BoundedOutput;
    static constexpr int PUBLISH_M = PublishM;

    // The publication index excludes the final M repeat.
    static_assert(PublishM >= 0 && PublishM < BlockM / (WaveM * 16) - 1);

    // Vector widths and K128 block-scale groups.
    static constexpr int VEC_A = 16;
    static constexpr int VEC_B = 16;
    static constexpr int VEC_C = 4;
    static constexpr int VEC_SCALE_A = 16;
    static constexpr int VEC_SF = sizeof(unsigned);
    static constexpr int VEC_OUTPUT = 8;
    static constexpr int A_CHUNKS_PER_FRAGMENT = G::W_M * G::W_K / (G::WARP_SIZE * VEC_A);
    static constexpr int B_CHUNKS_PER_FRAGMENT = G::W_N * G::W_K / (G::WARP_SIZE * VEC_B);

    static constexpr int GROUP_M = 1;
    static constexpr int GROUP_N = 128;
    static constexpr int GROUP_K = 128;
    static constexpr int SCALE_N_HALVES = (G::B_N + GROUP_N - 1) / GROUP_N;

    // Launch-order swizzles for matrix-tile reuse.
    static constexpr int SWIZZLE_GROUP_M = 4;
    static constexpr int SWIZZLE_GROUP_N = GroupN;
    static_assert(GroupN == 4 || GroupN == 8 || GroupN == 16 || GroupN == 32 || GroupN == 64);
    static constexpr int SWIZZLE_MIN_N_TILES = 256;
    static constexpr int SWIZZLE_MAX_N = 2048;
    static constexpr int SWIZZLE_MIN_M = 4096;

    static constexpr int MAX_K = ScalePanel * GROUP_K;
    static constexpr int LOOP_UNROLL = 2;

    // Padded matrix LDS rows and two ping-pong stages.
    static constexpr int smem_linear_wave = G::WARP_SIZE * VEC_A;
    static constexpr int smem_m_rep = G::B_M * G::B_K / smem_linear_wave;
    static constexpr int smem_n_rep = G::B_N * G::B_K / smem_linear_wave;
    static constexpr int smem_padding = 32;
    static constexpr int A_ROWS_PER_WAVE = G::WARP_SIZE / (G::B_K / VEC_A);

    static constexpr int A_STAGE = smem_m_rep * (smem_linear_wave + smem_padding);
    static constexpr int B_STAGE = smem_n_rep * (smem_linear_wave + smem_padding);
    static constexpr int NUM_STAGES = 2;
    static constexpr int MATRIX_LDS_BYTES = NUM_STAGES * (A_STAGE + B_STAGE);
    static constexpr int MATRIX_VMEM_INSTRUCTIONS = G::B_M * G::B_K / (G::BLOCK_SIZE * VEC_A) + G::B_N * G::B_K / (G::BLOCK_SIZE * VEC_B);
    static_assert(NUM_STAGES == 2);

    // Four M-repeat scales per A word; B packs its N128 scales into one word.
    static constexpr int SCALE_PANEL = ScalePanel;
    static constexpr int A_SCALE_PACKS = (G::E_M + VEC_SF - 1) / VEC_SF;
    static constexpr int SFA_M_VECTORS = A_SCALE_PACKS * G::T_M * VEC_SF;
    static constexpr int SFA_K_COLUMNS_PER_WAVE = G::WARP_SIZE / SFA_M_VECTORS;
    static constexpr int SFA_K_COLUMNS_PER_PASS = G::NUM_WAVES * SFA_K_COLUMNS_PER_WAVE;
    static constexpr int SFA_PASSES_PER_CACHE_PANEL = (SCALE_PANEL + SFA_K_COLUMNS_PER_PASS - 1) / SFA_K_COLUMNS_PER_PASS;
    static constexpr int SFA_PANEL_PITCH = A_SCALE_PACKS * G::T_M * G::W_M * VEC_SF;
    static constexpr int SFA_BYTES = ScaleOnRead ? G::B_M * SCALE_PANEL : SFA_PANEL_PITCH * SCALE_PANEL;
    static constexpr int SFB_BYTES = SCALE_PANEL * sizeof(unsigned);

    // Scale-panel passes must cover the K columns and M rows.
    static_assert(G::BLOCK_SIZE % SCALE_PANEL == 0);
    static_assert(G::W_M == VEC_SCALE_A && G::WARP_SIZE % SFA_M_VECTORS == 0);
    static_assert(FIXED_K == 0 || (FIXED_K % G::B_K == 0 && FIXED_K <= MAX_K));
    static_assert(SFA_PASSES_PER_CACHE_PANEL * SFA_K_COLUMNS_PER_PASS >= SCALE_PANEL);

    static constexpr int B_SCALE_PACKS = 1;
    static constexpr int LDS_BYTES = MATRIX_LDS_BYTES + SFA_BYTES + SFB_BYTES;
    static_assert(LDS_BYTES <= 160 * 1024);

    // The BF16 epilogue reuses LDS with a padded row stride.
    static constexpr int C_LDS_ROW_STRIDE_ELEMS = G::B_N + 8;
    static constexpr int OUTPUT_PASSES = G::B_M * G::B_N / (G::BLOCK_SIZE * VEC_OUTPUT);
    static_assert(G::B_M * C_LDS_ROW_STRIDE_ELEMS * sizeof(opus::bf16_t) <= LDS_BYTES);
    static_assert(G::B_M * G::B_N % (G::BLOCK_SIZE * VEC_OUTPUT) == 0);
};

// Compatibility argument order for the raw-scale ping-pong policy.
template<int BlockM = 192, int BlockN = 256, int ScalePanel = 128, int FixedK = 0, int WaveM = 4,
         int GroupM = 0, int CacheA = 0, int CacheB = 0, bool FragmentArrays = false,
         bool PinA = false, bool PinB = false, bool PinC = false, bool XcdOrder = false>
struct opus_gemm_bpreshuffle_m_repeat_scale_traits
    : opus_gemm_bpreshuffle_m_repeat_traits<BlockM, BlockN, ScalePanel, FixedK, WaveM, GroupM,
                                                CacheA, CacheB, FragmentArrays, XcdOrder, false, 0,
                                                8, true> {
    using Base =
        opus_gemm_bpreshuffle_m_repeat_traits<BlockM, BlockN, ScalePanel, FixedK, WaveM, GroupM,
                                                  CacheA, CacheB, FragmentArrays, XcdOrder, false,
                                                  0, 8, true>;
    static_assert(!PinA && !PinB && !PinC, "Raw-scale ping-pong does not support AGPR pinning");
    static_assert(ScalePanel > 0 && (FixedK == 0 || FixedK >= 128));
    static constexpr bool PIN_A = false;
    static constexpr bool PIN_B = false;
    static constexpr bool PIN_C = false;
    static constexpr int C_REGS = BlockM * BlockN / Base::BLOCK_SIZE;
    static constexpr int MIN_K = 128;

    // Raw M-scale vectors are transposed into the MFMA scale words on read.
    static constexpr int SFA_K_PANEL = ScalePanel < 32 ? ScalePanel : 32;
    static constexpr int SFA_K_PASSES = ScalePanel / SFA_K_PANEL;
    static constexpr int SFA_THREADS_PER_GROUP = Base::BLOCK_SIZE / SFA_K_PANEL;
    static constexpr int SFA_K_COLUMNS_PER_WAVE = Base::WARP_SIZE / SFA_THREADS_PER_GROUP;
    static constexpr int SFA_ROWS_PER_PASS = SFA_THREADS_PER_GROUP * Base::VEC_SCALE_A;
    static constexpr int SFA_PASSES = (BlockM + SFA_ROWS_PER_PASS - 1) / SFA_ROWS_PER_PASS;

    static_assert(Base::BLOCK_SIZE % ScalePanel == 0);
    static_assert(SFA_THREADS_PER_GROUP <= Base::WARP_SIZE);
    static_assert(Base::WARP_SIZE % SFA_THREADS_PER_GROUP == 0);
    static_assert(Base::T_N * Base::T_M * SFA_K_COLUMNS_PER_WAVE == SFA_K_PANEL);
    static_assert(SFA_K_PANEL * SFA_K_PASSES == ScalePanel);
    static_assert(SFA_PASSES * SFA_ROWS_PER_PASS >= BlockM);
};

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
struct opus_gemm_bpreshuffle_stage_geometry
    : opus_gemm_bpreshuffle_m_repeat_geometry<BM, BN, WM, WN, BK>
{
    // Launch geometry.
    static constexpr int WG_PER_CU  = 1;

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
struct opus_gemm_bpreshuffle_stage_policy
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
struct opus_gemm_bpreshuffle_stage_traits : Geometry, Policy
{
    static constexpr bool MULTI_STAGE = true;
    static constexpr bool OPERAND_POLICY = true;
    static constexpr int MIN_WGS_PER_CU = 1;
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

// A/B staged operands use the M-repeat pipeline; direct B is selected by B-VGPR.
template<class Geometry, class Policy>
struct opus_gemm_bpreshuffle_m_repeat_operand_traits
    : opus_gemm_bpreshuffle_stage_traits<Geometry, Policy>
{
    static_assert(!Policy::B_DIRECT);
};

// M-first traversal is useful when neighbouring workgroups reuse B panels.
// Keep it independent of the loading/compute policy and the default traversal.
template<class Base>
struct opus_gemm_bpreshuffle_mfirst_traits : Base
{
    static constexpr bool M_FIRST = true;
    static_assert(Base::SPLIT_K == 1);
};

// A/B LDS ring with resident scales. SplitK=1 writes C directly; split-K
// uses the XCD-local partial-result publication and workspace contract.
template<int BM,
          int BN,
          int WN,
          int BK,
          int Stages,
          int TotalK,
          int SplitK,
          int WM            = 1,
          bool CompactScale = false,
          int BCache        = 0,
          bool RolledLoop   = false>
struct opus_gemm_bpreshuffle_m_repeat_stage_traits
    : opus_gemm_bpreshuffle_m_repeat_geometry<BM, BN, WM, WN, BK>,
      opus_gemm_bpreshuffle_stage_policy<CompactScale, false, false, 1, BCache,
                                         true, false, false, false, 0>
{
    using G = opus_gemm_bpreshuffle_m_repeat_geometry<BM, BN, WM, WN, BK>;
    static constexpr bool MULTI_STAGE = true;
    static constexpr bool ROLLED_LOOP = RolledLoop;
    static constexpr bool OPERAND_POLICY = false;
    static constexpr int FIXED_N = 0;
    static constexpr int B_SETS = 1;

    // Launch geometry.
    static constexpr int WG_PER_CU = 1;

    // K partition and the resident scale panel.
    static constexpr int SPLIT_K = SplitK;
    static constexpr int TOTAL_K = TotalK;
    static constexpr int K = TotalK / SplitK;
    static constexpr int PANELS = K / BK;
    static constexpr int KS = BK / 128;
    static constexpr int SC = K / 128;

    // Ring scheduling.
    static constexpr int STAGES = Stages;
    static constexpr int INITIAL = Stages - 1;
    static constexpr int B_CACHE = BCache;

    // Matrix copies and LDS stages.
    static constexpr int A_BYTES = BM * BK;
    static constexpr int B_BYTES = BN * BK;
    static constexpr int STAGE_BYTES = A_BYTES + B_BYTES;
    static constexpr int A_COPIES = A_BYTES / (G::BLOCK_SIZE * 16);
    static constexpr int B_COPIES = B_BYTES / (G::BLOCK_SIZE * 16);
    static constexpr int MATRIX_ISSUES = A_COPIES + B_COPIES;

    // Compact copies group four M scales; the host enforces their alignment.
    static constexpr int SCALE_COLS = SC;
    static constexpr int SB_ROWS = 1;
    static constexpr int SB_COPIES = 1;
    static constexpr int SCALE_COPY = CompactScale ? 4 : 1;
    static constexpr int SA_COPIES = (BM * SC + G::BLOCK_SIZE * SCALE_COPY - 1) / (G::BLOCK_SIZE * SCALE_COPY);
    static constexpr int SA_BYTES = SA_COPIES * G::BLOCK_SIZE * 4;

    // LDS offsets and total allocation.
    static constexpr int OFF_SA = STAGES * STAGE_BYTES;
    static constexpr int OFF_SB = OFF_SA + SA_BYTES;
    static constexpr int LDS_BYTES = OFF_SB + G::BLOCK_SIZE * 4;

    static constexpr int wait(int t)
    {
        int issued = INITIAL * MATRIX_ISSUES + SA_COPIES + 1;
        int ready  = (t + 1) * MATRIX_ISSUES + SA_COPIES + 1;
        for (int i = 0; i < t; ++i)
        {
            if (i + INITIAL < PANELS)
            {
                issued += MATRIX_ISSUES;
                if (i + INITIAL == t)
                    ready = issued;
            }
        }
        return issued - ready;
    }

    // Keep each B tile inside one 128-column scale group.
    static_assert(WM > 0 && WM * WN <= 8);
    static_assert(BM > 0 && BM % (WM * 16) == 0 && BN > 0 && BN % (WN * 16) == 0);
    static_assert(WN == 1 || WN == 2 || WN == 4 || WN == 8);
    static_assert(BCache == 0 || BCache == 1 || BCache == 2 || BCache == 17);
    static_assert(BK == 128 || BK == 256);
    static_assert(SplitK == 1 || SplitK == 2 || SplitK == 4 || SplitK == 8);
    static_assert(TotalK > 0 && TotalK % (SplitK * BK) == 0);
    static_assert(A_BYTES % (G::BLOCK_SIZE * 16) == 0 && B_BYTES % (G::BLOCK_SIZE * 16) == 0);
    static_assert(BN <= 128 && 128 % BN == 0 && SC <= G::BLOCK_SIZE);
    static_assert(Stages >= 2 && PANELS >= INITIAL);
    static_assert(!RolledLoop || (SplitK == 1 && !CompactScale));
    static_assert(LDS_BYTES + sizeof(unsigned) <= 160 * 1024);
};
