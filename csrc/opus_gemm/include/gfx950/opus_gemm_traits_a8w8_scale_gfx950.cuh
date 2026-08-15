// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Traits and kargs for a8w8_scale pipeline (fp8 + block-scale).
// T_M=4, T_N=2 wave mapping. 5-tuple DTYPE with GROUP.
#pragma once

#include "../opus_gemm_utils.cuh"
#include "opus_gemm_traits_a16w16_gfx950.cuh"  // opus_splitk_ws_handle

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
    // How the quadrant's N-half nests with the 128-column B scale blocks. At
    // B_N=256 a half is exactly one block, so the two halves step one block
    // apart and each consumes its own scale. At B_N=128 a half is 64 columns:
    // both halves sit inside the *same* block, so the stride is 0 while the
    // count is still 1. Taking the count from the stride (as the plain division
    // did) yields zero scales and a zero-length scale vector.
    static_assert(HALF_B_N % GROUP_N == 0 || GROUP_N % HALF_B_N == 0,
                  "B tile halves must nest with the 128-column B scale blocks");
    static constexpr int SFB_HALF_STRIDE     = HALF_B_N / GROUP_N;
    static constexpr int SFB_GROUPS_PER_HALF = SFB_HALF_STRIDE > 0 ? SFB_HALF_STRIDE : 1;
    static constexpr int SFB_TILE_GROUPS     = SFB_HALF_STRIDE > 0 ? 2 * SFB_HALF_STRIDE : 1;

    static constexpr int sfa_buffer_load_insts = E_M * (B_K / GROUP_K);
    static constexpr int sfb_buffer_load_insts = SFB_GROUPS_PER_HALF * (B_K / GROUP_K);

    // Largest K the preload-scale pipelines accept. They stage the whole A-scale
    // panel in LDS, and that panel is (B_M/GROUP_M) rows by (K/B_K) bytes, so it
    // grows with K while the A/B staging already holds 2*(B_M+B_N)*B_K. At
    // B_M=B_N=256, B_K=128 the compiled kernel reports 151,680 of the 163,840
    // bytes a CU has: 11.9 KiB spare against the 16 KiB another 8192 of K would
    // cost. Going higher needs the panel refilled in K chunks, not a bigger
    // buffer. Both the device guard and the launcher check against this, so an
    // unsupported K raises instead of running a kernel that writes nothing.
    //
    // That arithmetic is this traits' -- the 151,680 is kid158/196, and the
    // 2*(B_M+B_N)*B_K staging it turns on is this pipeline's double buffer. The
    // two traits below copy the constant, and for them it is loose rather than
    // tight: their staging is different and the panel scales with B_M, so a
    // B_M=128 wave8 kid measures ~59,000 bytes at K=8192 and has room for
    // ~111,000 of per-split K. Measured per kid by reading
    // .group_segment_fixed_size out of the built code objects. Raising it there is
    // what would let a panel kid run at split_k=1 on large-K machine-filling
    // shapes, worth 17-25% over the shuffle_scale kid that owns that column
    // today -- but the panel array is sized from this constant and not from the
    // runtime K, so a raise costs LDS at every K.
    // It is free only where the CU was not going to give the kernel a second
    // workgroup anyway -- WG_PER_CU==1 is the declaration, not the answer, and a
    // 256-thread workgroup that fits twice loses a fifth of its throughput at
    // machine-filling M when the panel pushes it past half the CU's LDS. The
    // wave8 traits below budgets against that residency for exactly this reason.
    static constexpr int SF_PRELOAD_K_MAX = 8192;

    // B source layout: row-major [N, K] (false) vs shuffle_weight(w, (16,16))
    // (true). Where the LDS-staged path can re-map the load (see
    // b_preshuffle_contig_mxsk) the preshuffle order is staged verbatim and the
    // consumer B layout changes with it; otherwise only the producer's global
    // address math changes and LDS content is identical either way.
    static constexpr bool B_PRESHUFFLE = false;
};

// B-preshuffle sibling of the scale traits: identical tile, wave grid, LDS and
// quadrant schedule, with B read from a preshuffled weight buffer. This is the
// isolation variant -- it changes the preshuffle axis and nothing else, so a
// diff against the plain kid prices the preshuffle on its own.
template<int BLOCK_SIZE_,
        typename BLOCK_,
        typename DTYPE_,
        typename VEC_,
        typename GROUP_>
struct opus_gemm_a8w8_scale_bpreshuffle_traits_gfx950
    : opus_gemm_a8w8_scale_traits_gfx950<BLOCK_SIZE_, BLOCK_, DTYPE_, VEC_, GROUP_> {
    using base = opus_gemm_a8w8_scale_traits_gfx950<BLOCK_SIZE_, BLOCK_, DTYPE_, VEC_, GROUP_>;

    static constexpr bool B_PRESHUFFLE = true;

    static_assert(base::B_N % 16 == 0, "B preshuffle tiles N in blocks of 16");
    static_assert(base::VEC_B == 16,
                  "a preshuffle unit stores 16 contiguous K bytes per column");
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
    const opus_splitk_ws_handle* __restrict__ ws_handle;
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
        int WG_PER_CU_,
        // Consumers fetch B into registers themselves (see the bdirect traits).
        // A template parameter rather than a plain member because it resizes the
        // per-K-tile LDS footprint that prefetch_k_iter is derived from below.
        bool B_DIRECT_REG_ = false,
        // Drop the producer/consumer split: all four waves stage and compute.
        // Changes the wave mapping (T_M) and how many waves share each async
        // copy, so it has to be visible to the geometry below.
        bool ALL_WAVE_ = false>
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
    //
    // ALL_WAVE drops the split entirely: all four waves stage the tile and all
    // four compute it, as a 2x2 (M,N) grid of waves. Two consumer waves cap the
    // tile at ~128x128 -- that fp32 accumulator is already 128 VGPRs per wave and
    // doubling the tile doubles it on top of the A/B double buffers -- so
    // spreading the accumulator over four waves is what lifts the cap.
    //
    // The split is 2x2 rather than 4x1 because the A fragment layout ties T_M to
    // the load group: make_layout_ra_mxsk gives the T_M dimension a stride that
    // has the T_M waves divide one LOAD_GROUP_M of rows, so T_M*W_M must equal
    // LOAD_GROUP_M. With W_M=16 and LOAD_GROUP_M=32 that fixes T_M at 2; T_M=4
    // hands each wave 8 rows where the MFMA needs 16 and the upper two waves
    // read misaligned fragments.
    static constexpr bool ALL_WAVE = ALL_WAVE_;
    static constexpr bool IS_TILE_N = (B_M == 16);
    static constexpr int T_M = (ALL_WAVE || !IS_TILE_N) ? 2 : 1;
    static constexpr int T_N = (ALL_WAVE || IS_TILE_N) ? 2 : 1;
    static constexpr int T_K = 1;
    static_assert(T_K == 1);
    static_assert(!ALL_WAVE || !IS_TILE_N, "ALL_WAVE derives its own 2x2 wave grid");
    // Waves cooperating on each async copy: the producer pair, or all four.
    static constexpr int LOAD_WAVES = ALL_WAVE ? 4 : 2;
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
    static_assert(GROUP_M == 1 && GROUP_N == 128 && GROUP_K == 128);
    static_assert(B_K % GROUP_K == 0,
                  "flatmm K tile must contain whole DSv4 scale blocks");

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
    // make_layout_ra_mxsk gives the T_M dimension a stride that has the T_M waves
    // divide one LOAD_GROUP_M of rows, so the two have to stay in step.
    static_assert(T_M * W_M == LOAD_GROUP_M,
                  "the T_M waves must exactly divide one A load group's rows");
    static_assert(NUM_LOAD_GROUPS_PER_BK == B_K / GROUP_K);

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
    static_assert(B_N <= 2 * GROUP_N,
                  "mxscale flatmm splitK supports up to two 128-column B scale blocks");
    static_assert(GROUP_N % B_N == 0 || B_N % GROUP_N == 0,
                  "B tile must align with 128-column B scale blocks");
    static constexpr int SCALES_PER_BK = B_K / GROUP_K;
    static constexpr int N_SCALE_GROUPS = (B_N + GROUP_N - 1) / GROUP_N;
    // Largest per-split K the LDS scale panels hold, mirroring the pipeline's
    // SFA_K_MAX. The kernel returns without writing anything past it, which a
    // caller cannot distinguish from a GEMM that produced zeros, so launchers
    // check the same bound and raise instead. Split-K raises the reach: the
    // panel only has to cover one split's iterations.
    //
    // Inherited, not derived from this traits' own budget, and loose for most of
    // it: prefetch_k_iter below already spends max_lds_size_per_wg on staging,
    // so the panel lives in that floor division's remainder, which varies with
    // B_M and B_K. Measured, kid324/326 have 1.7-3.5 KiB spare (the bound is
    // real for them) while kid336/342 have 52-109 KiB (it is 13-27x loose). See
    // the base traits' SF_PRELOAD_K_MAX note.
    static constexpr int SF_PRELOAD_K_MAX = 8192;
    // B scale groups spanned by one N-wave's columns. The consumer N-waves read
    // blocked column ranges (nbc, and the matching SPLIT_N_STORE), so wave w owns
    // the contiguous COM_REP_N*W_N columns at w*COM_REP_N*W_N and therefore its
    // own slice of the tile's scale groups. Only ALL_WAVE opts into loading that
    // slice instead of the whole tile's groups -- for every T_N==1 kid the slice
    // is the whole thing and the base is 0, so they are unaffected.
    // The per-wave group *stride* and *count* differ when a wave is narrower than
    // GROUP_N: several N-waves then share one group, so the stride is 0 while the
    // count is still 1. Using the clamped count as the stride walks waves off the
    // end of the tile's groups.
    static constexpr bool SFB_PER_WAVE = ALL_WAVE;
    static constexpr int SFB_GROUP_STRIDE = COM_REP_N * W_N / GROUP_N;
    static constexpr int SFB_GROUPS_PER_WAVE = SFB_GROUP_STRIDE > 0 ? SFB_GROUP_STRIDE : 1;
    static constexpr int SFB_GROUPS = SFB_PER_WAVE ? SFB_GROUPS_PER_WAVE : N_SCALE_GROUPS;

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
    // B_DIRECT_REG stages no B, so only the A groups are budgeted. That headroom
    // is what lets the wider/taller direct-B tiles keep 3 prefetch slots at
    // WG_PER_CU=2; it is deliberately not spent on more slots, since 4 measured
    // ~25% slower than 3 (occupancy here is VGPR-bound, not LDS-bound).
    static constexpr int per_block_iter_lds_size =
        (NUM_LOAD_GROUPS_PER_BM + (B_DIRECT_REG_ ? 0 : NUM_LOAD_GROUPS_PER_BN))
        * NUM_LOAD_GROUPS_PER_BK * smem_per_group_load_size;
    static constexpr int prefetch_k_iter_budget = max_lds_size_per_wg / per_block_iter_lds_size;
    static constexpr int prefetch_k_iter =
        (B_DIRECT_REG_ && prefetch_k_iter_budget > 3) ? 3 : prefetch_k_iter_budget;
    static_assert(prefetch_k_iter >= 3,
                  "flatmm splitK pipeline requires at least 3 LDS prefetch slots");

    // Per-wave async-copy instruction counts: each load group is split across
    // LOAD_WAVES waves (repeat_m/repeat_n in the group-load layouts is
    // slots/LOAD_WAVES), and these drive the vmcnt bookkeeping in the pipeline.
    static_assert(slots % LOAD_WAVES == 0,
                  "each async load group must split evenly across the staging waves");
    static constexpr int a_buffer_load_insts =
        NUM_LOAD_GROUPS_PER_BM * NUM_LOAD_GROUPS_PER_BK * slots / LOAD_WAVES;
    static constexpr int b_buffer_load_insts =
        NUM_LOAD_GROUPS_PER_BN * NUM_LOAD_GROUPS_PER_BK * slots / LOAD_WAVES;
    static constexpr int a_ds_read_insts = (COM_REP_M * COM_REP_K * W_M * W_K) / (opus::get_warp_size() * VEC_A);
    static constexpr int b_ds_read_insts = (COM_REP_N * COM_REP_K * W_N * W_K) / (opus::get_warp_size() * VEC_B);
    static constexpr int mma_insts = COM_REP_M * COM_REP_N * COM_REP_K;

    // B source layout: row-major [N, K] (false) vs the 16x16-tiled preshuffle
    // produced by shuffle_weight(w, layout=(16, 16)) (true). On the LDS-staged
    // path this flips the producer's B global layout, and where the load can be
    // re-mapped to contiguous issues (b_preshuffle_contig_mxsk) it flips the
    // consumer's LDS read layout to match, since LDS then holds preshuffle
    // rather than row-major order.
    static constexpr bool B_PRESHUFFLE = false;
    // Scale packing for the multi-scale-group MFMA path: false keeps the tuned
    // broadcast pack (MXSCALE_ACCUM_MODE), true forces the hardware
    // scale_op_sel byte-select over the COM_REP_K K groups.
    static constexpr bool SCALE_OPSEL = false;
    // Consumer waves read their MFMA B fragments straight from global memory
    // instead of going through the producer/LDS staging. Only meaningful with
    // B_PRESHUFFLE (see the bdirect traits below for why the two are tied).
    static constexpr bool B_DIRECT_REG = B_DIRECT_REG_;
};

// B-preshuffle + scale-op_sel sibling of the flatmm split-K traits.
//
// Same tile geometry / LDS / pipeline as the base traits (so a bpreshuffle kid
// is just its plain kid with a preshuffled weight buffer), with two axes
// flipped: B comes from shuffle_weight(w, layout=(16, 16)) and the per-subtile
// scales are selected with the MFMA scale_op_sel immediate instead of a
// broadcast pack. The 16x16 preshuffle tile requires the B tile and the per-WG
// B load group to be whole 16-column blocks.
template<int BLOCK_SIZE_,
        typename BLOCK_,
        typename DTYPE_,
        typename VEC_,
        typename GROUP_,
        int WG_PER_CU_,
        bool B_DIRECT_REG_ = false,
        bool ALL_WAVE_ = false>
struct opus_gemm_a8w8_mxscale_flatmm_splitk_bpreshuffle_traits_gfx950
    : opus_gemm_a8w8_mxscale_flatmm_splitk_traits_gfx950<BLOCK_SIZE_, BLOCK_, DTYPE_, VEC_, GROUP_, WG_PER_CU_, B_DIRECT_REG_, ALL_WAVE_> {
    using base = opus_gemm_a8w8_mxscale_flatmm_splitk_traits_gfx950<
        BLOCK_SIZE_, BLOCK_, DTYPE_, VEC_, GROUP_, WG_PER_CU_, B_DIRECT_REG_, ALL_WAVE_>;

    static constexpr bool B_PRESHUFFLE = true;
    static constexpr bool SCALE_OPSEL  = true;

    static_assert(base::B_N % 16 == 0,
                  "B preshuffle tiles N in blocks of 16");
    static_assert(base::LOAD_GROUP_N % 16 == 0,
                  "each B load group must cover whole 16-column preshuffle blocks");
    static_assert(base::LOAD_GROUP_K % 32 == 0,
                  "each B load group must cover whole 32-element preshuffle K blocks");
    static_assert(base::VEC_B == 16,
                  "a preshuffle block stores 16 contiguous K bytes per (n, k-half)");
};

// B-preshuffle sibling that keeps the base traits' broadcast scale pack.
//
// The struct above flips two axes at once, so every measurement against a plain
// kid charges the B layout for whatever the op_sel byte-select does as well.
// This one flips only B_PRESHUFFLE, which makes (plain, this, opsel-sibling) a
// three-point split that separates them. The pipeline family already has the
// pure form -- opus_gemm_a8w8_scale_bpreshuffle_traits_gfx950, which kid196
// uses -- and there it measures within noise of its plain twin.
template<int BLOCK_SIZE_,
        typename BLOCK_,
        typename DTYPE_,
        typename VEC_,
        typename GROUP_,
        int WG_PER_CU_,
        bool B_DIRECT_REG_ = false,
        bool ALL_WAVE_ = false>
struct opus_gemm_a8w8_mxscale_flatmm_splitk_bpreshuffle_bcast_traits_gfx950
    : opus_gemm_a8w8_mxscale_flatmm_splitk_traits_gfx950<BLOCK_SIZE_, BLOCK_, DTYPE_, VEC_, GROUP_, WG_PER_CU_, B_DIRECT_REG_, ALL_WAVE_> {
    using base = opus_gemm_a8w8_mxscale_flatmm_splitk_traits_gfx950<
        BLOCK_SIZE_, BLOCK_, DTYPE_, VEC_, GROUP_, WG_PER_CU_, B_DIRECT_REG_, ALL_WAVE_>;

    static constexpr bool B_PRESHUFFLE = true;
    // SCALE_OPSEL stays false, i.e. the base's MXSCALE_ACCUM_MODE pack.

    static_assert(base::B_N % 16 == 0,
                  "B preshuffle tiles N in blocks of 16");
    static_assert(base::LOAD_GROUP_N % 16 == 0,
                  "each B load group must cover whole 16-column preshuffle blocks");
    static_assert(base::LOAD_GROUP_K % 32 == 0,
                  "each B load group must cover whole 32-element preshuffle K blocks");
    static_assert(base::VEC_B == 16,
                  "a preshuffle block stores 16 contiguous K bytes per (n, k-half)");
};

// B-preshuffle sibling that skips LDS for B entirely.
//
// The 16x16 preshuffle order IS the mfma_16x16x128 B fragment order: a 16-column
// block stores byte (n, k) at (k/16)*256 + n*16 + k%16, and the MFMA wants lane l
// to hold n = l%16 over k = rept*64 + (l/16)*16 + 0..15, i.e. block offset
// rept*1024 + (l/16)*256 + (l%16)*16 == rept*1024 + l*16. So one wave's B operand
// for a 16x16x128 tile is 2048 contiguous bytes and every lane's half is a
// naturally aligned dwordx4 -- there is nothing for an LDS round trip to fix up.
// Consumers therefore buffer_load B into the MFMA registers themselves, which
// drops the B half of the producer waves' async copies, the B ds_reads, and the
// B LDS buffers. The cost is that both consumer waves fetch the same B tile (in
// tileM they share one N range), so B is read twice per WG from L1/L2.
template<int BLOCK_SIZE_,
        typename BLOCK_,
        typename DTYPE_,
        typename VEC_,
        typename GROUP_,
        int WG_PER_CU_>
struct opus_gemm_a8w8_mxscale_flatmm_splitk_bpreshuffle_bdirect_traits_gfx950
    : opus_gemm_a8w8_mxscale_flatmm_splitk_bpreshuffle_traits_gfx950<BLOCK_SIZE_, BLOCK_, DTYPE_, VEC_, GROUP_, WG_PER_CU_, true> {
    using base = opus_gemm_a8w8_mxscale_flatmm_splitk_bpreshuffle_traits_gfx950<
        BLOCK_SIZE_, BLOCK_, DTYPE_, VEC_, GROUP_, WG_PER_CU_, true>;

    // buffer_loads per consumer per K tile: one dwordx4 per (n-rep, k-rep, half).
    static constexpr int b_direct_load_insts = base::COM_REP_N * base::COM_REP_K * 2;
    static_assert(b_direct_load_insts == base::b_ds_read_insts,
                  "direct B must fetch exactly what the LDS path used to ds_read");

    // tileN is fine: T_N partitions N across the consumer waves, which the
    // direct-B layout absorbs as a base-offset of the wave's first 16-column
    // block (nbc), the same way the LDS path offsets smem_b_at.
    static_assert(base::T_K == 1,
                  "direct B layout folds away the tile_k mma p-dim");
    static_assert(base::W_N == 16 && base::W_K == 128,
                  "direct B addressing is derived from the 16x16x128 fragment order");
};

// 8-wave all-compute traits for direct-to-register preshuffled B.
//
// Standalone rather than a flag on the traits above, because the two constants
// that have to move -- BLOCK_SIZE and LOAD_GROUP_M -- are what every other
// member there is derived from, and the 4-wave kids must keep their exact
// geometry.
//
// Why 8 waves. The per-CU register file is the binding constraint on tile area:
// a 256x256 fp32 accumulator is 65536 floats == 1024 registers per lane-slot,
// i.e. half of the 4 SIMDs x 512 the CU has, however many waves it is spread
// over. Four waves cannot hold it (256 accumulator + double-buffered fp8
// fragments is 512 per wave, the whole per-wave file, leaving nothing for
// addressing), so the 256x256 tile needs the accumulator spread over eight
// waves at 128 each. Eight waves on 4 SIMDs is 2 waves/SIMD, which caps every
// wave at 256 registers -- affordable here (128 accumulator + 96 of
// single-buffered fragments) and the reason the fragments below are not double
// buffered: the latency hiding comes from the second wave on the SIMD instead.
//
// Why T_M picks the wave grid. LOAD_GROUP_M is T_M*W_M (the T_M waves sharing an
// N range are exactly the ones that split one A load group's rows), and the wave
// grid it implies is what sets how many waves read the same B. Direct-B has no
// LDS to share through, so B's bytes cross L1 once per M-wave: T_M is the read
// amplification on the one operand that is not staged, and 4 vs 2 there is the
// difference between 160KB and 96KB of L1 traffic per K tile against a ~2048
// cycle MFMA budget. Measured at 256x256, M=32768: T_M=4 runs 2.06x the L1
// accesses of the LDS-B kid on the same tile, its L1 return path 93% busy and
// its matrix pipe only 74%; T_M=2 is 1.37x, 84% and 90%, and 16% faster. The
// floor for direct B is 2x on whichever operand it is, since T_M=1 would need
// the whole 256-row A fragment resident (128 registers on top of a 128-register
// accumulator) and does not fit.
//
// T_M also decides how the async copies divide. A group splits across LOAD_WAVES
// waves as slots/LOAD_WAVES, and slots == LOAD_GROUP_M/smem_sub == LOAD_GROUP_M/8
// is 8 at T_M=4 but only 4 at T_M=2, too few for eight staging waves. The extra
// factor comes from M instead: at T_M=2 there are twice as many A load groups, so
// LOAD_M_SPLIT waves take alternate groups and LOAD_WAVES split the slots within
// one. Every wave still issues the same number of copies either way, which is
// what keeps the pipeline's vmcnt immediates wave-invariant.
template<int BLOCK_SIZE_,
        typename BLOCK_,
        typename DTYPE_,
        typename VEC_,
        typename GROUP_,
        int WG_PER_CU_,
        int T_M_>
struct opus_gemm_a8w8_mxscale_bpreshuffle_wave8_traits_gfx950 {
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
    static_assert(std::is_same_v<D_A, fp8_t>);
    static_assert(std::is_same_v<D_C, fp32_t>);
    static_assert(std::is_same_v<D_ACC, fp32_t>);
    static_assert(std::is_same_v<D_SF, unsigned char>);

    // The schedule itself is written against WAVES, not against eight of them, so
    // the wave count follows BLOCK_SIZE: 512 gives the 8-wave grids this file was
    // built for, 256 gives the 4-wave ones. wave64 is assumed (gfx950 only), and
    // the guarded assert below is what catches a target where that is false.
    static constexpr int WAVES = BLOCK_SIZE_ / 64;
    static constexpr bool ALL_WAVE = true;
    static constexpr bool IS_TILE_N = false;
    static constexpr int T_M = T_M_;
    static constexpr int T_N = WAVES / T_M;
    static constexpr int T_K = 1;
#if !defined(__HIP_DEVICE_COMPILE__) || defined(__gfx950__)
    static_assert(BLOCK_SIZE == WAVES * opus::get_warp_size(),
                  "this schedule requires whole wave64 waves");
#endif
    static_assert(WAVES == 4 || WAVES == 8, "only the 4- and 8-wave grids are wired up");
    static_assert(T_M * T_N == WAVES, "the wave grid must cover every wave once");

    static constexpr int W_M = 16;
    static constexpr int W_N = 16;
    static constexpr int W_K = 128;

    static constexpr int VEC_A = opus::get<0>(VEC{});
    static constexpr int VEC_B = opus::get<1>(VEC{});
    static constexpr int VEC_C = opus::get<2>(VEC{});

    static constexpr int GROUP_M = opus::get<0>(GROUP{});
    static constexpr int GROUP_N = opus::get<1>(GROUP{});
    static constexpr int GROUP_K = opus::get<2>(GROUP{});
    static_assert(GROUP_M == 1 && GROUP_N == 128 && GROUP_K == 128);
    static_assert(B_K % GROUP_K == 0);

    static constexpr int LOAD_GROUP_M = T_M * W_M;
    static constexpr int LOAD_GROUP_N = 64;
    static constexpr int LOAD_GROUP_K = W_K;
    static constexpr int LOAD_GROUP_M_LANE = 1;
    static constexpr int LOAD_GROUP_N_LANE = 1;
    static constexpr int NUM_LOAD_GROUPS_PER_BM = B_M / LOAD_GROUP_M;
    static constexpr int NUM_LOAD_GROUPS_PER_BN = B_N / LOAD_GROUP_N;
    static constexpr int NUM_LOAD_GROUPS_PER_BK = B_K / LOAD_GROUP_K;
    static_assert(NUM_LOAD_GROUPS_PER_BM * LOAD_GROUP_M == B_M);
    static_assert(NUM_LOAD_GROUPS_PER_BN * LOAD_GROUP_N == B_N);
    static_assert(T_M * W_M == LOAD_GROUP_M,
                  "the T_M waves must exactly divide one A load group's rows");
    static_assert(NUM_LOAD_GROUPS_PER_BK == B_K / GROUP_K);

    static constexpr int COM_REP_M = B_M / (W_M * T_M);
    static constexpr int COM_REP_N = B_N / (W_N * T_N);
    static constexpr int COM_REP_K = B_K / (W_K * T_K);
    // The M-packed scale panel hands a K group's subtile bytes to the lane as
    // adjacent bytes, which scale_op_sel indexes four at a time.
    static constexpr int SFA_WORDS = COM_REP_M / 4;
    static_assert(COM_REP_M % 4 == 0 && SFA_WORDS >= 1,
                  "the M-packed scale bytes must fill whole op_sel dwords");
    static_assert(COM_REP_N >= 1 && B_N % (W_N * T_N) == 0);
    static_assert(COM_REP_K == NUM_LOAD_GROUPS_PER_BK);
    static_assert(B_N <= 2 * GROUP_N);
    static_assert(GROUP_N % B_N == 0 || B_N % GROUP_N == 0);

    static constexpr int SCALES_PER_BK = B_K / GROUP_K;
    static constexpr int N_SCALE_GROUPS = (B_N + GROUP_N - 1) / GROUP_N;
    // SF_PRELOAD_K_MAX is defined below, once prefetch_k_iter is known -- it is
    // derived from the LDS the staging leaves over rather than being a constant.
    //
    // Every wave owns a contiguous COM_REP_N*W_N column range, so it reads only
    // the B scale groups that range spans -- one group when the range is narrower
    // than GROUP_N (T_N=4), several when it is wider.
    static constexpr bool SFB_PER_WAVE = true;
    static constexpr int SFB_WAVE_COLS = COM_REP_N * W_N;
    static constexpr int SFB_GROUPS = SFB_WAVE_COLS >= GROUP_N ? SFB_WAVE_COLS / GROUP_N : 1;
    static_assert(SFB_WAVE_COLS >= GROUP_N ? SFB_WAVE_COLS % GROUP_N == 0
                                           : GROUP_N % SFB_WAVE_COLS == 0,
                  "a wave's columns must not straddle a partial B scale group");

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
    // B goes straight to registers, so only the A groups are staged.
    static constexpr int per_block_iter_lds_size =
        NUM_LOAD_GROUPS_PER_BM * NUM_LOAD_GROUPS_PER_BK * smem_per_group_load_size;
    static constexpr int prefetch_k_iter_budget = max_lds_size_per_wg / per_block_iter_lds_size;
    // Deeper than 3 buys nothing: vmcnt is one in-order counter, so waiting for
    // this tile's direct-B load also retires every A copy issued before it, and
    // the reachable A lead is two tiles whatever the ring holds.
    static constexpr int prefetch_k_iter = prefetch_k_iter_budget > 3 ? 3 : prefetch_k_iter_budget;
    static_assert(prefetch_k_iter >= 3, "the pipeline requires at least 3 LDS prefetch slots");

    // Largest per-split K the LDS scale panels cover. Past it the kernel returns
    // without writing anything, which a caller cannot tell apart from a GEMM that
    // produced zeros, so the launcher checks the same bound and raises. Split-K
    // extends the reach: a panel only has to hold one split's iterations.
    //
    // Derived from this traits' own budget instead of the flat 8192 it used to
    // copy from the base traits. That 8192 is justified by a 151,680-of-163,840
    // measurement on kid158/196, whose staging is a 2*(B_M+B_N)*B_K double buffer
    // -- a family this constant does not describe. Here the staging is
    // prefetch_k_iter A slots, and the panel gets what that leaves: measured,
    // kid205 sits at 59,012 bytes with room for ~111,000 of per-split K, so the
    // flat bound was 13x short and kept it out of the split_k=1 column, which at
    // K=16384 on a machine-filling shape is where the fastest kernel runs (the
    // panel kids beat the shuffle_scale kid that owns that column today by
    // 17-25% at every split_k they share).
    //
    // Rows are the worst case over the kernel's template parameters, which the
    // traits cannot see: SHUFFLE_SCALE stages no panel at all and SFA_MPACK_GLOBAL
    // stages only the SFB rows, so both get a bound computed for more than they
    // use, which is safe in the direction that matters. The panel is
    // SF_PANEL_ROWS * K / GROUP_K bytes, linear in K, hence the plain division.
    //
    // Capped, because the array is sized from this constant and not from the
    // runtime K: without a cap a kid with a small panel would spend every spare
    // byte of LDS on reach it will never be asked for. 32768 covers the largest K
    // in the tuned table with room to spare.
    // Reserve, because this arithmetic is not the allocator's. The panel array is
    // alignas(16) after the staging array, and comparing this model against
    // .group_segment_fixed_size in the built objects leaves it a few bytes short
    // (58,960 modelled against 58,948 real on kid338). Without a reserve B_M=256
    // lands at 163,812 of 163,840 and a byte of padding would overflow it.
    //
    // Budgeted against the residency the kernel already had rather than against
    // max_lds_size_per_wg, because WG_PER_CU_ is a declared attribute and not
    // what the CU schedules. kid203/kid205 are 256-thread workgroups at 246 VGPR,
    // so two of them fit a CU on waves (2*246 <= 512) and LDS was the binding
    // term: at the flat panel they sat at 59,012 bytes and got two, and spending
    // the spare half on reach took them to 83,972, where 2*83,972 > 163,840 and
    // only one is resident. Measured A/B over the 133-cell table, that costs
    // 1.19x on kid203 and 1.20x on kid205 at m>=1536 -- and nothing at small M,
    // where there are too few workgroups for the second slot to matter. The
    // 512-thread kids are unaffected either way (kid168/kid202 1.011x, kid194 and
    // kid175 1.000x): their footprint was already over half the CU, so the panel
    // spends LDS no second workgroup was going to use.
    //
    // Preserving that residency still leaves most of the reach the derivation is
    // for: kid205 lands at 30,464 of per-split K instead of 32,768, kid194 keeps
    // its 30,848 because it was single-resident to begin with.
    static constexpr int SF_PANEL_LDS_RESERVE = 256;
    static constexpr int SF_PANEL_ROWS = B_M / GROUP_M + N_SCALE_GROUPS;
    static constexpr int SF_PANEL_STAGING_LDS =
        prefetch_k_iter * per_block_iter_lds_size;
    static constexpr int SF_PANEL_FLAT_LDS =
        SF_PANEL_STAGING_LDS + SF_PANEL_ROWS * (8192 / GROUP_K);
    static constexpr int SF_PANEL_RESIDENT_WGS =
        LDS_SIZE_TOTAL / SF_PANEL_FLAT_LDS < 1 ? 1 : LDS_SIZE_TOTAL / SF_PANEL_FLAT_LDS;
    static constexpr int SF_PANEL_LDS_SHARE = LDS_SIZE_TOTAL / SF_PANEL_RESIDENT_WGS;
    static constexpr int SF_PANEL_LDS_CEILING =
        SF_PANEL_LDS_SHARE < max_lds_size_per_wg ? SF_PANEL_LDS_SHARE
                                                 : max_lds_size_per_wg;
    static constexpr int SF_PANEL_LDS_BUDGET =
        SF_PANEL_LDS_CEILING - SF_PANEL_STAGING_LDS - SF_PANEL_LDS_RESERVE;
    static constexpr int SF_PRELOAD_K_FIT =
        (SF_PANEL_LDS_BUDGET / SF_PANEL_ROWS) * GROUP_K;
    static constexpr int SF_PRELOAD_K_CAP = 32768;
    static constexpr int SF_PRELOAD_K_MAX =
        ((SF_PRELOAD_K_FIT < SF_PRELOAD_K_CAP ? SF_PRELOAD_K_FIT : SF_PRELOAD_K_CAP)
         / B_K) * B_K;
    // Never below what the flat constant already promised, so no kid loses reach.
    static_assert(SF_PRELOAD_K_MAX >= 8192,
                  "the scale panel no longer reaches the 8192 of per-split K the "
                  "flat SF_PRELOAD_K_MAX promised; the staging above grew");
    static_assert(SF_PANEL_STAGING_LDS
                      + (long)SF_PANEL_ROWS * SF_PRELOAD_K_MAX / GROUP_K
                  <= SF_PANEL_LDS_CEILING,
                  "A staging plus the scale panel overflow the LDS share that "
                  "keeps this kernel's workgroups resident");

    // All eight waves stage, but a group only has `slots` pieces to hand out, so
    // the waves that do not fit inside one group take a different group instead:
    // LOAD_WAVES waves split the slots of a group, LOAD_M_SPLIT waves take
    // every LOAD_M_SPLIT'th group.
    static constexpr int LOAD_WAVES = slots < WAVES ? slots : WAVES;
    static constexpr int LOAD_M_SPLIT = WAVES / LOAD_WAVES;
    static_assert(LOAD_WAVES * LOAD_M_SPLIT == WAVES);
    static_assert(slots % LOAD_WAVES == 0,
                  "each async load group must split evenly across the staging waves");
    static_assert(NUM_LOAD_GROUPS_PER_BM % LOAD_M_SPLIT == 0,
                  "the A load groups must split evenly across the staging waves");
    static constexpr int a_buffer_load_insts =
        (NUM_LOAD_GROUPS_PER_BM / LOAD_M_SPLIT) * NUM_LOAD_GROUPS_PER_BK * slots / LOAD_WAVES;
    static constexpr int b_buffer_load_insts = 0;
    static constexpr int a_ds_read_insts = (COM_REP_M * COM_REP_K * W_M * W_K) / (opus::get_warp_size() * VEC_A);
    static constexpr int b_ds_read_insts = (COM_REP_N * COM_REP_K * W_N * W_K) / (opus::get_warp_size() * VEC_B);
    static constexpr int mma_insts = COM_REP_M * COM_REP_N * COM_REP_K;

    static constexpr bool B_PRESHUFFLE = true;
    static constexpr bool SCALE_OPSEL = true;
    static constexpr bool B_DIRECT_REG = true;

    static constexpr int b_direct_load_insts = COM_REP_N * COM_REP_K * 2;
    static_assert(b_direct_load_insts == b_ds_read_insts,
                  "direct B must fetch exactly what the LDS path used to ds_read");
    static_assert(B_N % 16 == 0 && LOAD_GROUP_K % 32 == 0 && VEC_B == 16);

    // Registers per lane: fp32 accumulator, A fragment, B fragment. 2 waves/SIMD
    // caps a wave at 256 registers, and what is left over here is what addressing
    // gets. Each MFMA of this shape holds a 4-register fp32 accumulator, which is
    // what makes the whole budget fit in the 224 below rather than the count of
    // MFMAs. A is resident in full, which is what confines this schedule to the
    // (T_M, B_M) pairs whose A fragment fits -- see the T_M=1 note below.
    //
    // K is reduced, not accumulated over: the MMA loop indexes C by
    // im*COM_REP_N + in with no ik, so COM_REP_K belongs in the two fragment
    // terms and not in this one. It read COM_REP_M*COM_REP_N*COM_REP_K*4 while
    // every kid here had COM_REP_K == 1, so the two agreed; the first B_K=256
    // tile is what tells them apart.
    static constexpr int est_acc_vgpr = COM_REP_M * COM_REP_N * 4;
    static constexpr int est_a_vgpr = a_ds_read_insts * 4;
    static constexpr int est_b_vgpr = b_direct_load_insts * 4;
    static_assert(est_acc_vgpr + est_a_vgpr + est_b_vgpr <= 224,
                  "no room left for addressing inside the 256-register wave");
};

// The same 8-wave direct-B schedule on a 2x4 wave grid instead of 4x2.
//
// Only two waves now share an N range, so each of B's bytes crosses L1 twice per
// K tile instead of four times -- the fix for the measured L1 return-path
// saturation that keeps the 4x2 grid's matrix pipe at 74%. The register cost is
// a wash: the per-wave tile turns from 64x128 into 128x64, which trades 32
// registers of B fragment for 32 of A. What it does cost is LDS reads, since A
// is now read by four waves rather than two, and instruction-wise a wave's eight
// M subtiles need two op_sel dwords of A scale rather than one.
template<int BLOCK_SIZE_,
        typename BLOCK_,
        typename DTYPE_,
        typename VEC_,
        typename GROUP_,
        int WG_PER_CU_>
struct opus_gemm_a8w8_mxscale_bpreshuffle_wave8n4_traits_gfx950
    : opus_gemm_a8w8_mxscale_bpreshuffle_wave8_traits_gfx950<
          BLOCK_SIZE_, BLOCK_, DTYPE_, VEC_, GROUP_, WG_PER_CU_, 2> {};

// T_M=1: no wave shares an N range, so each of B's bytes crosses L1 exactly once
// and direct-B finally matches what staging B through LDS costs there. This is
// the far end of the T_M sweep and the fastest thing in the family.
//
// It only exists at B_M=128, because the price of T_M=1 lands on the other
// operand: one wave owns all B_M rows, so at B_M=256 its A fragment is 128
// registers -- as big as the accumulator -- and with B does not fit the 256 a
// 2-waves/SIMD kernel gets. Halving B_M halves it to 64, which does fit, and the
// choice of which operand is redundant then lands the way it does in the
// reference flydsl kernel: the operand every wave shares (A) goes through LDS
// once, and the operand each wave owns alone (B) comes straight from global.
//
// B_M=256 was tried the other way, streaming A from LDS instead of holding it
// (kid195, removed) -- see the kid195 note in opus_gemm_common.py for why the
// schedule that took is this one.
//
// WAVES follows BLOCK_SIZE, so this alias covers both grids that T_M=1 admits:
// 512 threads give 1x8, 256 give the 1x4 that the 4-wave allwave_bdirect family
// cannot express (its traits fix the grid at 2x2).
template<int BLOCK_SIZE_,
        typename BLOCK_,
        typename DTYPE_,
        typename VEC_,
        typename GROUP_,
        int WG_PER_CU_>
struct opus_gemm_a8w8_mxscale_bpreshuffle_wavetm1_traits_gfx950
    : opus_gemm_a8w8_mxscale_bpreshuffle_wave8_traits_gfx950<
          BLOCK_SIZE_, BLOCK_, DTYPE_, VEC_, GROUP_, WG_PER_CU_, 1> {
    static_assert(opus::get<0>(opus::remove_cvref_t<BLOCK_>{}) == 128,
                  "one wave owns all B_M rows here, and A only stays resident at "
                  "B_M=128; a 256-row tile needs a wider grid (wave8n4)");
};

// B-preshuffle sibling with no producer waves: all four waves stage the tile
// and all four compute it (T_M=4).
//
// B stays in LDS here, deliberately. Direct-to-register B and the deep A
// prefetch cannot share one wave: vmcnt is a single in-order counter, and B's
// register double buffer is only one K tile deep while A's LDS pipeline is
// prefetch_k_iter deep, so waiting for B(k) would also drain the A prefetch.
// Keeping the two roles in separate waves is exactly what let the producer count
// only its async copies and the consumer only its B loads. Direct B is worth
// nothing here anyway once the scale panels are preloaded (measured: kid184 vs
// kid325 land within 2% of each other), whereas the fourfold accumulator split
// is what buys the larger tile.
template<int BLOCK_SIZE_,
        typename BLOCK_,
        typename DTYPE_,
        typename VEC_,
        typename GROUP_,
        int WG_PER_CU_>
struct opus_gemm_a8w8_mxscale_flatmm_splitk_bpreshuffle_allwave_traits_gfx950
    : opus_gemm_a8w8_mxscale_flatmm_splitk_bpreshuffle_traits_gfx950<BLOCK_SIZE_, BLOCK_, DTYPE_, VEC_, GROUP_, WG_PER_CU_, false, true> {
    using base = opus_gemm_a8w8_mxscale_flatmm_splitk_bpreshuffle_traits_gfx950<
        BLOCK_SIZE_, BLOCK_, DTYPE_, VEC_, GROUP_, WG_PER_CU_, false, true>;

    static_assert(base::ALL_WAVE && base::T_M == 2 && base::T_N == 2);
    // Each wave owns COM_REP_N*W_N contiguous columns and loads exactly the scale
    // groups they span (SFB_PER_WAVE), so B_N crossing GROUP_N is handled here.
    static_assert(base::COM_REP_N * base::W_N % base::GROUP_N == 0
                  || base::COM_REP_N * base::W_N < base::GROUP_N,
                  "an all-wave N-wave must not straddle a partial B scale group");
    // B_N=256 was measured and rejected: no real gain, plus one accumulator
    // register reads back as 0 through the split-K workspace (see the note next
    // to the all-wave tiles in opus_gemm_common.py). Keep new tiles off that path
    // until that is understood.
    static_assert(base::B_N <= base::GROUP_N,
                  "all-wave B_N>128 is unresolved under split_k>1");
    static_assert(!base::B_DIRECT_REG,
                  "all-wave stages B through LDS so one vmcnt stream serves both operands");
};

// All-wave 2x2 with B read straight into registers. The LDS-staged all-wave
// traits above forbid this pairing because the shared pipeline derives one
// vmcnt bound per tile and it cannot serve an A lead of prefetch_k_iter-1
// tiles and a one-tile B lead at the same time; the dedicated
// allwave_bdirect kernel carries its own two immediates instead, so the
// combination is allowed here.
//
// The pairing was aimed at occupancy: dropping B out of LDS leaves only the A
// double buffer (32 KiB), and the 2x2 grid quarters the tile so the fp32
// accumulator is 64 registers instead of the 128 that a 2x1 grid over a doubled
// M range needs -- both of which the wave4m2 direct-B kid runs out of at
// WG_PER_CU=2. The footprint did come down (120 VGPRs, no spill) and it bought
// nothing, because the smaller tile re-reads B twice as often; the kernel it
// feeds is kept as that control. See the kid 189/197 notes in
// opus_gemm_common.py.
template<int BLOCK_SIZE_,
        typename BLOCK_,
        typename DTYPE_,
        typename VEC_,
        typename GROUP_,
        int WG_PER_CU_>
struct opus_gemm_a8w8_mxscale_flatmm_splitk_bpreshuffle_allwave_bdirect_traits_gfx950
    : opus_gemm_a8w8_mxscale_flatmm_splitk_bpreshuffle_traits_gfx950<BLOCK_SIZE_, BLOCK_, DTYPE_, VEC_, GROUP_, WG_PER_CU_, true, true> {
    using base = opus_gemm_a8w8_mxscale_flatmm_splitk_bpreshuffle_traits_gfx950<
        BLOCK_SIZE_, BLOCK_, DTYPE_, VEC_, GROUP_, WG_PER_CU_, true, true>;

    static constexpr int b_direct_load_insts = base::COM_REP_N * base::COM_REP_K * 2;
    static_assert(b_direct_load_insts == base::b_ds_read_insts,
                  "direct B must fetch exactly what the LDS path used to ds_read");

    static_assert(base::ALL_WAVE && base::T_M == 2 && base::T_N == 2);
    // At 128 columns both N-waves sit inside one B scale group, so
    // SFB_GROUP_STRIDE is 0 and every wave reads group 0. At 256 an N-wave is
    // exactly GROUP_N wide, so the stride is 1 and wave_id_n indexes the group
    // directly -- the same per-wave base the LDS all-wave path carries.
    static_assert(base::B_N <= 2 * base::GROUP_N,
                  "all-wave direct-B spans at most two B scale groups");
    static_assert(base::slots % base::LOAD_WAVES == 0,
                  "each A load group must split evenly across the four staging waves");

    // Accumulator, one A fragment, and the B buffers the schedule alternates --
    // the headroom left is what addressing and the scales get inside a
    // 256-register wave. Doubling B_N doubles both the accumulator and a B
    // buffer, which at 256 columns puts the double-buffered form at 288 and out
    // of the file; that tile runs one B buffer instead (see B_REG_BUFS in the
    // kernel) and lands at exactly 224.
    static constexpr int B_REG_BUFS = base::B_N > base::GROUP_N ? 1 : 2;
    static constexpr int est_acc_vgpr = base::COM_REP_M * base::COM_REP_N * 4;
    static constexpr int est_a_vgpr = base::a_ds_read_insts * 4;
    static constexpr int est_b_vgpr = B_REG_BUFS * b_direct_load_insts * 4;
    static_assert(est_acc_vgpr + est_a_vgpr + est_b_vgpr <= 224,
                  "no room left for addressing inside the 256-register wave");
};
