// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Traits and kargs for a8w8_scale pipeline (fp8 + block-scale).
// T_M=4, T_N=2 wave mapping. 5-tuple DTYPE with GROUP.
#pragma once

#include "../opus_gemm_utils.cuh"
#include "opus_gemm_traits_a16w16_gfx950.cuh"

template<int BLOCK_SIZE_,
        typename BLOCK_,
        typename DTYPE_,
        typename VEC_,
        typename GROUP_>
struct opus_gemm_a8w8_scale_traits_gfx950 {
    using BLOCK = opus::remove_cvref_t<BLOCK_>;
    using DTYPE = opus::remove_cvref_t<DTYPE_>;
    using VEC   = opus::remove_cvref_t<VEC_>;
    using GROUP = opus::remove_cvref_t<GROUP_>;

    static constexpr int BLOCK_SIZE = BLOCK_SIZE_;

    static constexpr int B_M = opus::get<0>(BLOCK{});
    static constexpr int B_N = opus::get<1>(BLOCK{});
    static constexpr int B_K = opus::get<2>(BLOCK{});

    using D_A   = opus::tuple_element_t<0, DTYPE>;
    using D_B   = opus::tuple_element_t<1, DTYPE>;
    using D_C   = opus::tuple_element_t<2, DTYPE>;
    using D_ACC = opus::tuple_element_t<3, DTYPE>;
    using D_SF  = opus::tuple_element_t<4, DTYPE>;
    static_assert(std::is_same<D_A, D_B>::value);

    static constexpr int T_M = 4;
    static constexpr int T_N = 2;
    static constexpr int T_K = 1;

    // a8w8 is gfx950-only (wave64). On a non-gfx950 device pass the kernel
    // body is stubbed out, but the traits struct is still instantiated for the
    // host launcher; skip the wave-size invariant there (gfx1250 is wave32).
#if !defined(__HIP_DEVICE_COMPILE__) || defined(__gfx950__)
    static_assert(BLOCK_SIZE / opus::get_warp_size() == T_M * T_N * T_K);
#endif
    static_assert(T_K == 1);

    static constexpr int W_M = 16;
    static constexpr int W_N = 16;
    static constexpr int W_K = 128;

    static constexpr int HALF_B_M = B_M / 2;
    static constexpr int HALF_B_N = B_N / 2;

    static_assert(HALF_B_M % (W_M * T_M) == 0);
    static_assert(HALF_B_N % (W_N * T_N) == 0);
    static_assert(B_K % (W_K * T_K) == 0);

    static constexpr int E_M = HALF_B_M / (W_M * T_M);
    static constexpr int E_N = HALF_B_N / (W_N * T_N);
    static constexpr int E_K = B_K / (W_K * T_K);

    static constexpr int VEC_A = opus::get<0>(VEC{});
    static constexpr int VEC_B = opus::get<1>(VEC{});
    static constexpr int VEC_C = opus::get<2>(VEC{});

    static constexpr int GROUP_M = opus::get<0>(GROUP{});
    static constexpr int GROUP_N = opus::get<1>(GROUP{});
    static constexpr int GROUP_K = opus::get<2>(GROUP{});

    static_assert(VEC_A == 16 / sizeof(D_A));
    static constexpr int smem_linear_wave = opus::get_warp_size() * 16 / sizeof(D_A);
    static constexpr int smem_sub = smem_linear_wave / B_K;
    static constexpr int smem_m_rep = HALF_B_M / smem_sub;
    static constexpr int smem_n_rep = HALF_B_N / smem_sub;
    static constexpr int smem_padding = 2 * 16 / sizeof(D_A);

    static constexpr int a_buffer_load_insts = HALF_B_M * B_K / (BLOCK_SIZE * VEC_A);
    static constexpr int b_buffer_load_insts = HALF_B_N * B_K / (BLOCK_SIZE * VEC_B);
    static constexpr int a_ds_read_insts = (E_M * E_K * W_M * W_K) / (opus::get_warp_size() * VEC_A);
    static constexpr int b_ds_read_insts = (E_N * E_K * W_N * W_K) / (opus::get_warp_size() * VEC_B);
    static constexpr int sfa_buffer_load_insts = E_M * (B_K / GROUP_K);
    static constexpr int sfb_buffer_load_insts = (HALF_B_N / GROUP_N) * (B_K / GROUP_K);
};

struct opus_gemm_scale_kargs_gfx950 {
    const void* __restrict__ ptr_a;
    const void* __restrict__ ptr_b;
    void* __restrict__ ptr_c;
    int m;
    int n;
    int k;
    int batch;
    int stride_a;
    int stride_b;
    int stride_c;
    int stride_a_batch;
    int stride_b_batch;
    int stride_c_batch;

    const void* __restrict__ ptr_sfa;
    const void* __restrict__ ptr_sfb;
    int stride_sfa;
    int stride_sfb;
    int stride_sfa_batch;
    int stride_sfb_batch;
};

struct opus_gemm_scale_splitk_kargs_gfx950 {
    const void* __restrict__ ptr_a;
    const void* __restrict__ ptr_b;
    void* __restrict__ ptr_ws;
    int m;
    int n;
    int k;
    int batch;
    int split_k;
    int stride_a;
    int stride_b;
    int stride_ws;
    int stride_a_batch;
    int stride_b_batch;
    int stride_ws_batch;

    const void* __restrict__ ptr_sfa;
    const void* __restrict__ ptr_sfb;
    int stride_sfa;
    int stride_sfb;
    int stride_sfa_batch;
    int stride_sfb_batch;

    void* __restrict__ ptr_c;
    int stride_c;
    int stride_c_batch;
    unsigned long counter_offset_bytes;
};

// 4-wave warp-specialized fp8/e8m0 flatmm split-K traits.
//
// This is intentionally separate from opus_gemm_a8w8_scale_traits_gfx950:
// the existing a8w8_scale pipeline is an 8-wave half-tile kernel, while this
// trait matches the flatmm producer/consumer schedule used for decode-like
// BMM shapes. First version keeps B_M == T_M * W_M and B_K == GROUP_K so each
    // scaled MFMA consumes per-row A scales and one 128x128 B scale.
template<int BLOCK_SIZE_,
        typename BLOCK_,
        typename DTYPE_,
        typename VEC_,
        typename GROUP_,
        int WG_PER_CU_>
struct opus_gemm_a8w8_mxscale_flatmm_splitk_traits_gfx950 {
    using BLOCK = opus::remove_cvref_t<BLOCK_>;
    using DTYPE = opus::remove_cvref_t<DTYPE_>;
    using VEC   = opus::remove_cvref_t<VEC_>;
    using GROUP = opus::remove_cvref_t<GROUP_>;

    static constexpr int BLOCK_SIZE = BLOCK_SIZE_;

    static constexpr int B_M = opus::get<0>(BLOCK{});
    static constexpr int B_N = opus::get<1>(BLOCK{});
    static constexpr int B_K = opus::get<2>(BLOCK{});

    using D_A   = opus::tuple_element_t<0, DTYPE>;
    using D_B   = opus::tuple_element_t<1, DTYPE>;
    using D_C   = opus::tuple_element_t<2, DTYPE>;
    using D_ACC = opus::tuple_element_t<3, DTYPE>;
    using D_SF  = opus::tuple_element_t<4, DTYPE>;
    static_assert(std::is_same<D_A, D_B>::value);
    static_assert(std::is_same_v<D_A, fp8_t>, "mxscale flatmm splitK expects fp8 A/B");
    static_assert(std::is_same_v<D_C, fp32_t>, "mxscale flatmm splitK main writes fp32 workspace");
    static_assert(std::is_same_v<D_ACC, fp32_t>, "mxscale flatmm splitK accumulates in fp32");
    static_assert(std::is_same_v<D_SF, unsigned char>, "mxscale flatmm splitK consumes e8m0 uint8 scales");

    // 4 waves per WG: 2 producer waves + 2 consumer waves.
    //
    // Two consumer-wave layouts, selected at compile time from B_M:
    //   tileM (B_M >= 32): consumers split M (T_M=2, T_N=1). Default; used by
    //     all pre-existing kids (64/128 rows). Bit-identical to the original.
    //   tileN (B_M == 16): consumers split N (T_M=1, T_N=2). A 16-row A tile
    //     maps to a single MFMA M-wave, so small-M / decode BMM shapes stop
    //     over-computing a fat B_M tile (the ~10 us floor on kid320=64x32 for
    //     M<=32 came from computing 64 rows for 8 valid ones). The two consumer
    //     waves instead each own half of B_N.
    static constexpr bool IS_TILE_N = (B_M == 16);
    static constexpr int T_M = IS_TILE_N ? 1 : 2;
    static constexpr int T_N = IS_TILE_N ? 2 : 1;
    static constexpr int T_K = 1;
    static_assert(T_K == 1);
    static_assert(BLOCK_SIZE == 256, "flatmm splitK requires 4 wave64 waves");
#if !defined(__HIP_DEVICE_COMPILE__) || defined(__gfx950__)
    static_assert(BLOCK_SIZE == 4 * opus::get_warp_size(),
                  "flatmm splitK requires exactly four waves");
#endif

    static constexpr int W_M = 16;
    static constexpr int W_N = 16;
    static constexpr int W_K = 128;

    static constexpr int VEC_A = opus::get<0>(VEC{});
    static constexpr int VEC_B = opus::get<1>(VEC{});
    static constexpr int VEC_C = opus::get<2>(VEC{});

    static constexpr int GROUP_M = opus::get<0>(GROUP{});
    static constexpr int GROUP_N = opus::get<1>(GROUP{});
    static constexpr int GROUP_K = opus::get<2>(GROUP{});
    // A and B quantise on the same block: DSv4's 128, or MX's 32, on both axes.
    static_assert(GROUP_M == 1);
    static_assert(GROUP_N == 128 || GROUP_N == 32);
    static_assert(GROUP_K == 128 || GROUP_K == 32);
    static_assert(B_K % GROUP_K == 0,
                  "flatmm K tile must contain whole scale blocks");
    // MX blocks inside one MFMA's K extent, and the reason GROUP_K=32 costs no
    // extra scale instruction: a 16x16x128 fragment hands each lane
    // W_M*W_K/warp_size == 32 elements, which is exactly one MX block, and the
    // lane supplies its own scale byte. So the SF_PER_MFMA_K blocks of one MFMA
    // are told apart by lane_id / W_M in the scale *address*, not by
    // scale_op_sel -- that selects one byte per MFMA for the whole wave (see
    // pack_e8m0x4's note). Measured, not assumed: a probe feeding the four
    // quarters 2^0..2^3 on an all-ones 16x16x128 reads back 32*(1+2+4+8), not
    // the 128 a shared byte would give. At GROUP_K=128 this is 1, every lane
    // term below folds away, and the 128 path stays bit-identical.
    static_assert(W_K % GROUP_K == 0);
    static constexpr int SF_PER_MFMA_K = W_K / GROUP_K;
    // One MFMA fragment splits its K over warp_size / W_M lane quarters, each
    // owning W_K / that many elements -- 32 at W_K=128 on wave64, which is why
    // GROUP_K=32 lands exactly one MX block per lane. SF_LANE_K_DIV is how many
    // quarters share a scale, so the lane's block index is
    // (lane_id / W_M) / SF_LANE_K_DIV, and at GROUP_K=128 the divisor is all
    // four quarters and the term collapses to zero.
    static constexpr int SF_LANE_K_QUARTERS = opus::get_warp_size() / W_M;
    static_assert(SF_LANE_K_QUARTERS % SF_PER_MFMA_K == 0,
                  "an MX block must not span part of a lane's K range");
    static constexpr int SF_LANE_K_DIV = SF_LANE_K_QUARTERS / SF_PER_MFMA_K;

    // async group load geometry; fp8-specific B_K=128 path uses one MFMA per
    // LOAD_GROUP_K, unlike a16w16 flatmm where LOAD_GROUP_K=W_K*2.
    // tileN uses 16-wide A/B async-load groups so that (a) a B_M=16 A tile is a
    // single load group and (b) LOAD_GROUP_M == LOAD_GROUP_N keeps a single
    // `slots` value valid for both A and B (no A/B slot decoupling needed), and
    // B_N=32 splits into two 16-col groups -- one per consumer N-wave.
    static constexpr int LOAD_GROUP_M = IS_TILE_N ? 16 : 32;
    static constexpr int LOAD_GROUP_N = IS_TILE_N ? 16 : 32;
    static constexpr int LOAD_GROUP_K = W_K;
    static constexpr int LOAD_GROUP_M_LANE = 1;
    static constexpr int LOAD_GROUP_N_LANE = 1;
    static constexpr int NUM_LOAD_GROUPS_PER_BM = B_M / LOAD_GROUP_M;
    static constexpr int NUM_LOAD_GROUPS_PER_BN = B_N / LOAD_GROUP_N;
    static constexpr int NUM_LOAD_GROUPS_PER_BK = B_K / LOAD_GROUP_K;
    static_assert(NUM_LOAD_GROUPS_PER_BM * LOAD_GROUP_M == B_M);
    static_assert(NUM_LOAD_GROUPS_PER_BN * LOAD_GROUP_N == B_N);
    // Scale granularity is not load granularity. The two were the same number
    // while GROUP_K == LOAD_GROUP_K == 128, and conflating them is the first
    // thing a GROUP_K=32 instance trips over: its K tile carries SF_PER_MFMA_K
    // times as many scales as it has A/B load groups.
    static_assert(NUM_LOAD_GROUPS_PER_BK * SF_PER_MFMA_K == B_K / GROUP_K);

    static constexpr int COM_REP_M = B_M / (W_M * T_M);
    static constexpr int COM_REP_N = B_N / (W_N * T_N);
    static constexpr int COM_REP_K = B_K / (W_K * T_K);
    static_assert(COM_REP_M == 1 || COM_REP_M == 2 || COM_REP_M == 4,
                  "mxscale flatmm splitK supports 16 (tileN) / 32 / 64 / 128 rows per tile");
    static_assert(COM_REP_N >= 1, "B_N must be a multiple of W_N*T_N");
    // tileN splits B_N across two consumer waves, so B_N must contain 2*W_N cols
    // and every N scale group must be wave-splittable without straddling.
    static_assert(!IS_TILE_N || (B_N % (W_N * T_N) == 0),
                  "tileN requires B_N divisible by W_N*T_N (=32)");
    static_assert(COM_REP_K == NUM_LOAD_GROUPS_PER_BK);
    // B_N <= 2 * GROUP_N used to stand here. Nothing structural was behind it:
    // every consumer loops static_for<N_SCALE_GROUPS> and addresses group ng at
    // ng * stride_sfb, so the group count was already free. It recorded the range
    // that had been exercised, and at GROUP_N=32 it would have capped B_N at 64.
    static_assert(GROUP_N % B_N == 0 || B_N % GROUP_N == 0,
                  "B tile must align with the B scale blocks");
    // Distinct scales a K tile holds, which is what the global buffers and the
    // LDS panels are sized by -- SF_PER_MFMA_K times larger at GROUP_K=32.
    static constexpr int SCALES_PER_BK = B_K / GROUP_K;
    // What one lane needs in registers: one byte per MFMA, whatever GROUP_K is,
    // because the lane owns one MX block of every MFMA it issues. The two
    // coincide at GROUP_K=128, which is why the pipeline sizes its scale vector
    // by SCALES_PER_BK; at 32 that would over-allocate by SF_PER_MFMA_K and
    // index past the lane's own bytes.
    static constexpr int SF_LANE_SCALES_PER_BK = COM_REP_K;
    static_assert(SCALES_PER_BK == SF_LANE_SCALES_PER_BK * SF_PER_MFMA_K,
                  "T_K > 1 would break the one-block-per-lane-per-MFMA identity");
    // How wide the per-lane scale fetch can vectorise. At GROUP_K=128 a lane's
    // COM_REP_K bytes are the tile's whole K-scale run, contiguous, and this
    // stays the single b32 load it has always been. At 32 they are the lane's
    // own block out of each MFMA, so consecutive ones sit SF_PER_MFMA_K apart
    // and the run is not vectorisable -- one ubyte load per MFMA instead. That
    // stride is the buffer's, not the hardware's: a K order grouped by lane
    // quarter would make them adjacent again.
    static constexpr int SF_LANE_LOAD_VEC = SF_PER_MFMA_K == 1 ? SF_LANE_SCALES_PER_BK : 1;
    static constexpr int N_SCALE_GROUPS = (B_N + GROUP_N - 1) / GROUP_N;
    // N subtiles sharing one B scale group: GROUP_N columns per group over W_N
    // per subtile. T_N is deliberately absent. It used to be in this denominator
    // and does not belong -- T_N partitions subtiles across consumer waves, it
    // does not widen a subtile -- and the error was invisible while a tile held
    // one group, because both forms then floor to 0.
    static_assert(GROUP_N % W_N == 0, "a B scale group must be whole subtiles");
    static constexpr int SFB_REP_N = GROUP_N / W_N;
    // A consumer N-wave owns COM_REP_N contiguous subtiles, so it needs this
    // many groups, starting at wave_id_n * COM_REP_N / SFB_REP_N. Holding only
    // its own share is what lets the subtile loop keep indexing v_sfb locally.
    static_assert(COM_REP_N % SFB_REP_N == 0 || SFB_REP_N % COM_REP_N == 0,
                  "an N-wave must not straddle a partial B scale group");
    static constexpr int SFB_GROUPS_PER_WAVE =
        COM_REP_N >= SFB_REP_N ? COM_REP_N / SFB_REP_N : 1;
    // The bound the old B_N <= 2 * GROUP_N was standing in for. What a finer
    // GROUP_N really costs is v_sfb, one byte per (group, MFMA) in the lane:
    // 2 bytes for today's widest 128-column kid, 8 for a B_N=128 tile at
    // GROUP_N=GROUP_K=32. Sized against the lane's share, not SCALES_PER_BK,
    // which is why separating the two mattered.
    static_assert(SFB_GROUPS_PER_WAVE * SF_LANE_SCALES_PER_BK <= 64,
                  "the B scale vector would cost more than 16 VGPRs a lane");

    static_assert(VEC_A == 16 / sizeof(D_A));
    static_assert(VEC_B == 16 / sizeof(D_B));
    static constexpr int smem_linear_wave_per_async_load = opus::get_warp_size() * 16 / sizeof(D_A);
    static constexpr int smem_sub = smem_linear_wave_per_async_load / LOAD_GROUP_K;
    static constexpr int slots = LOAD_GROUP_M / smem_sub;
    static constexpr int smem_padding = 2 * 16 / sizeof(D_A);
    static constexpr int smem_per_group_load_size =
        slots * (smem_linear_wave_per_async_load + smem_padding) * sizeof(D_A);

    static constexpr int WG_PER_CU = WG_PER_CU_;
    static constexpr int LDS_SIZE_TOTAL = 163840;
    static constexpr int max_lds_size_per_wg = LDS_SIZE_TOTAL / WG_PER_CU_;
    static constexpr int per_block_iter_lds_size =
        (NUM_LOAD_GROUPS_PER_BM + NUM_LOAD_GROUPS_PER_BN)
        * NUM_LOAD_GROUPS_PER_BK * smem_per_group_load_size;
    static constexpr int prefetch_k_iter = max_lds_size_per_wg / per_block_iter_lds_size;
    static_assert(prefetch_k_iter >= 3,
                  "flatmm splitK pipeline requires at least 3 LDS prefetch slots");

    static constexpr int a_buffer_load_insts = NUM_LOAD_GROUPS_PER_BM * NUM_LOAD_GROUPS_PER_BK * slots / 2;
    static constexpr int b_buffer_load_insts = NUM_LOAD_GROUPS_PER_BN * NUM_LOAD_GROUPS_PER_BK * slots / 2;
    static constexpr int a_ds_read_insts = (COM_REP_M * COM_REP_K * W_M * W_K) / (opus::get_warp_size() * VEC_A);
    static constexpr int b_ds_read_insts = (COM_REP_N * COM_REP_K * W_N * W_K) / (opus::get_warp_size() * VEC_B);
    static constexpr int mma_insts = COM_REP_M * COM_REP_N * COM_REP_K;
};
