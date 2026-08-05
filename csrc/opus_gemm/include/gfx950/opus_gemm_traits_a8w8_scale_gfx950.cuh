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
        bool B_DIRECT_REG_ = false>
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

    static constexpr int a_buffer_load_insts = NUM_LOAD_GROUPS_PER_BM * NUM_LOAD_GROUPS_PER_BK * slots / 2;
    static constexpr int b_buffer_load_insts = NUM_LOAD_GROUPS_PER_BN * NUM_LOAD_GROUPS_PER_BK * slots / 2;
    static constexpr int a_ds_read_insts = (COM_REP_M * COM_REP_K * W_M * W_K) / (opus::get_warp_size() * VEC_A);
    static constexpr int b_ds_read_insts = (COM_REP_N * COM_REP_K * W_N * W_K) / (opus::get_warp_size() * VEC_B);
    static constexpr int mma_insts = COM_REP_M * COM_REP_N * COM_REP_K;

    // B source layout: row-major [N, K] (false) vs the 16x16-tiled preshuffle
    // produced by shuffle_weight(w, layout=(16, 16)) (true). Only the B global
    // address math changes; LDS content and every consumer-side layout are
    // identical, so the flag flips one layout helper in the producer.
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
        bool B_DIRECT_REG_ = false>
struct opus_gemm_a8w8_mxscale_flatmm_splitk_bpreshuffle_traits_gfx950
    : opus_gemm_a8w8_mxscale_flatmm_splitk_traits_gfx950<BLOCK_SIZE_, BLOCK_, DTYPE_, VEC_, GROUP_, WG_PER_CU_, B_DIRECT_REG_> {
    using base = opus_gemm_a8w8_mxscale_flatmm_splitk_traits_gfx950<
        BLOCK_SIZE_, BLOCK_, DTYPE_, VEC_, GROUP_, WG_PER_CU_, B_DIRECT_REG_>;

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
