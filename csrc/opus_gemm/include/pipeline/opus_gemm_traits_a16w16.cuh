// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Traits for a16w16 (bf16) pipelines. This header carries two independent
// traits structs and two kargs structs:
//
//   opus_gemm_a16w16_traits<..., TILE, WAVE>
//     Split-barrier pipeline used by opus_gemm_pipeline_a16w16.cuh.
//     Configurable TILE (T_M, T_N, T_K) and WAVE (W_M, W_N, W_K). 4-tuple DTYPE.
//
//   opus_gemm_a16w16_flatmm_traits<..., MFMA, WG_PER_CU, HAS_BIAS>
//     4-wave warp-specialized pipeline (2 producer + 2 consumer) used by
//     opus_gemm_pipeline_a16w16_flatmm.cuh. Derives prefetch depth dynamically
//     from the LDS budget. Locked T_M=2/T_N=1/T_K=1. 5-tuple DTYPE.
//     Ported from gcnasm/opus_fmm/flatmm_a16w16_4wave_wasp.cc.
#pragma once

#include "../opus_gemm_utils.cuh"

// ============================================================================
// Split-barrier a16w16 traits
// ============================================================================

template<int BLOCK_SIZE_,
        typename BLOCK_,
        typename DTYPE_,
        typename VEC_,
        typename TILE_,
        typename WAVE_>
struct opus_gemm_a16w16_traits {
    using BLOCK = opus::remove_cvref_t<BLOCK_>;
    using DTYPE = opus::remove_cvref_t<DTYPE_>;
    using VEC   = opus::remove_cvref_t<VEC_>;
    using TILE  = opus::remove_cvref_t<TILE_>;
    using WAVE  = opus::remove_cvref_t<WAVE_>;

    static constexpr int BLOCK_SIZE = BLOCK_SIZE_;

    static constexpr int B_M = opus::get<0>(BLOCK{});
    static constexpr int B_N = opus::get<1>(BLOCK{});
    static constexpr int B_K = opus::get<2>(BLOCK{});

    using D_A   = opus::tuple_element_t<0, DTYPE>;
    using D_B   = opus::tuple_element_t<1, DTYPE>;
    using D_C   = opus::tuple_element_t<2, DTYPE>;
    using D_ACC = opus::tuple_element_t<3, DTYPE>;
    static_assert(std::is_same<D_A, D_B>::value);

    static constexpr int T_M = opus::get<0>(TILE{});
    static constexpr int T_N = opus::get<1>(TILE{});
    static constexpr int T_K = opus::get<2>(TILE{});

    static_assert(BLOCK_SIZE / opus::get_warp_size() == T_M * T_N * T_K);
    static_assert(T_K == 1);

    static constexpr int W_M = opus::get<0>(WAVE{});
    static constexpr int W_N = opus::get<1>(WAVE{});
    static constexpr int W_K = opus::get<2>(WAVE{});

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
};

#ifndef OPUS_GEMM_NOSCALE_KARGS_DEFINED
#define OPUS_GEMM_NOSCALE_KARGS_DEFINED
struct opus_gemm_noscale_kargs {
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
};
#endif

// ============================================================================
// Flatmm a16w16 traits (4-wave warp-specialized pipeline)
// ============================================================================
//
// 7 template parameters: BLOCK_SIZE, BLOCK, DTYPE, VEC, MFMA, WG_PER_CU, HAS_BIAS.
// The flatmm pipeline uses a warp-specialized 2 producer + 2 consumer layout
// (not split-barrier), requires T_M=2/T_N=1/T_K=1, and derives prefetch depth
// dynamically from the LDS budget.

template<int BLOCK_SIZE_,   // workgroup size (locked to 256 for 4 waves)
        typename BLOCK_,    // opus::seq<B_M, B_N, B_K>
        typename DTYPE_,    // opus::tuple<D_A, D_B, D_C, D_ACC, D_BIAS>
        typename VEC_,      // opus::seq<VEC_A, VEC_B, VEC_C>
        typename MFMA_,     // opus::seq<W_M, W_N, W_K>
        int WG_PER_CU_,
        bool HAS_BIAS_>
struct opus_gemm_a16w16_flatmm_traits {
    using BLOCK = opus::remove_cvref_t<BLOCK_>;
    using DTYPE = opus::remove_cvref_t<DTYPE_>;
    using VEC   = opus::remove_cvref_t<VEC_>;
    using MFMA  = opus::remove_cvref_t<MFMA_>;
    static constexpr int BLOCK_SIZE = BLOCK_SIZE_;

    static constexpr int B_M = opus::get<0>(BLOCK{});
    static constexpr int B_N = opus::get<1>(BLOCK{});
    static constexpr int B_K = opus::get<2>(BLOCK{});

    using D_A    = opus::tuple_element_t<0, DTYPE>;
    using D_B    = opus::tuple_element_t<1, DTYPE>;
    using D_C    = opus::tuple_element_t<2, DTYPE>;
    using D_ACC  = opus::tuple_element_t<3, DTYPE>;
    using D_BIAS = opus::tuple_element_t<4, DTYPE>;

    // Warp-specialized 4-wave layout: 2 producer (async gmem->LDS) + 2 consumer
    // (ds_read + MFMA). Compute uses 2 waves, data load uses 2 waves.
    //
    // NOTE: BLOCK_SIZE / warp_size = 4 waves, NOT T_M * T_N * T_K = 2. The
    // canonical split-barrier check (BLOCK_SIZE == warps * warp_size with
    // warps = T_M*T_N*T_K) does NOT apply here because 2 of the 4 waves are
    // producers that do not contribute to the MMA tile.
    static constexpr int T_M = 2; // compute-wave count along M
    static constexpr int T_N = 1; // compute-wave count along N
    static constexpr int T_K = 1; // compute-wave count along K

    // ── Warp-spec 4-wave pipeline constraints ──
    static_assert(T_K == 1, "flatmm requires T_K=1");
    static_assert(T_M == 2, "flatmm requires T_M=2 (ra layout depends on it)");
    static_assert(T_N == 1, "flatmm requires T_N=1 (consumer waves share N slab)");
    static_assert(BLOCK_SIZE == 256,
                  "flatmm requires BLOCK_SIZE=256 (4 waves: 2 producer + 2 consumer)");
    static_assert(BLOCK_SIZE == 4 * opus::get_warp_size(),
                  "flatmm BLOCK_SIZE must cover exactly 4 waves");

    static constexpr int W_M = opus::get<0>(MFMA{}); // wave gemm size M
    static constexpr int W_N = opus::get<1>(MFMA{}); // wave gemm size N
    static constexpr int W_K = opus::get<2>(MFMA{}); // wave gemm size K

    // ra/rb LDS read layouts are written for LOAD_GROUP_M_LANE=1 (W_M<32).
    // LGML=4 (W_M>=32, e.g. MFMA 32x32x16) requires a different layout not
    // yet implemented; see INTEGRATION.md "MFMA 32x32x16 not supported".
    static_assert(W_M < 32,
                  "flatmm ra layout only implemented for W_M<32 (LOAD_GROUP_M_LANE=1)");

    // async group load geometry
    static constexpr int LOAD_GROUP_M = (W_M >= 32) ? 64 : 32;
    static constexpr int LOAD_GROUP_N = (W_N >= 32) ? 64 : 32;
    static constexpr int LOAD_GROUP_K = W_K * 2; // 2 MFMA per LOAD_GROUP_K
    static constexpr int LOAD_GROUP_M_LANE = (W_M >= 32) ? 4 : 1;
    static constexpr int LOAD_GROUP_N_LANE = (W_N >= 32) ? 4 : 1;
    static constexpr int NUM_LOAD_GROUPS_PER_BM = B_M / LOAD_GROUP_M;
    static constexpr int NUM_LOAD_GROUPS_PER_BN = B_N / LOAD_GROUP_N;
    // K direction: one block-K is made of NUM_LOAD_GROUPS_PER_BK group-loads,
    // each group-load covers LOAD_GROUP_K pixels along K.
    static constexpr int NUM_LOAD_GROUPS_PER_BK = B_K / LOAD_GROUP_K;
    static_assert(NUM_LOAD_GROUPS_PER_BM * LOAD_GROUP_M == B_M);
    static_assert(NUM_LOAD_GROUPS_PER_BN * LOAD_GROUP_N == B_N);
    static_assert(NUM_LOAD_GROUPS_PER_BK * LOAD_GROUP_K == B_K);

    // MFMA inst counts
    static constexpr int COM_REP_M = B_M / (W_M * T_M); // repeat along M
    static constexpr int COM_REP_N = B_N / (W_N * T_N); // repeat along N
    static constexpr int COM_REP_K = B_K / (W_K * T_K); // repeat along K
    static constexpr int VEC_A = opus::get<0>(VEC{});
    static constexpr int VEC_B = opus::get<1>(VEC{});
    static constexpr int VEC_C = opus::get<2>(VEC{});

    static constexpr bool HAS_BIAS = HAS_BIAS_;

    // Compact LDS pixels for one async group_load. smem_sub is defined per
    // LOAD_GROUP_K (not B_K): one group_load copies LOAD_GROUP_M * LOAD_GROUP_K
    // pixels into an independent LDS block regardless of B_K. When
    // B_K > LOAD_GROUP_K multiple LDS blocks are stacked along the K-group
    // axis (see NUM_LOAD_GROUPS_PER_BK above).
    static_assert(VEC_A == 16 / sizeof(D_A));
    static constexpr int smem_linear_wave_per_async_load = opus::get_warp_size() * 16 / sizeof(D_A);
    static constexpr int smem_sub = smem_linear_wave_per_async_load / LOAD_GROUP_K;
    static constexpr int slots = LOAD_GROUP_M / smem_sub;
    static constexpr int smem_padding = (W_M >= 32) ? 16 / sizeof(D_A) : 2 * 16 / sizeof(D_A);
    static constexpr int smem_per_group_load_size = slots * (smem_linear_wave_per_async_load + smem_padding) * sizeof(D_A);

    // Dynamic prefetch K to fill LDS within WG_PER_CU budget.
    // Hardcoded gfx950 LDS size (160 KiB = 163840 B). Cannot use
    // opus::get_smem_size() here because it's guarded by __gfx950__ which is
    // only defined on the device pass; host pass would see 65536 and cause
    // pfk<3 and break static_asserts. All aiter a16w16 kernels are gfx950-only
    // (enforced via --offload-arch=gfx950 in JIT config), so this is safe.
    static constexpr int WG_PER_CU = WG_PER_CU_;
    static constexpr int LDS_SIZE_TOTAL = 163840;
    static constexpr int max_lds_size_per_wg = LDS_SIZE_TOTAL / WG_PER_CU_;
    static constexpr int per_block_iter_lds_size = (NUM_LOAD_GROUPS_PER_BM + NUM_LOAD_GROUPS_PER_BN) * NUM_LOAD_GROUPS_PER_BK * smem_per_group_load_size;
    static constexpr int prefetch_k_iter = max_lds_size_per_wg / per_block_iter_lds_size;
    // Two pipeline modes based on pfk:
    // - pfk >= 4: Depth-2 software-pipelined (main iter i issues K=i+2).
    // - pfk == 3: Depth-1 pipeline (main iter i issues K=i+1). Slot-collision
    //   safe because (i+1)%3 != (i-1)%3 and != i%3.
    // - pfk == 2: Not supported (depth-1 would race onto consumer's slot).
    static_assert(prefetch_k_iter >= 3,
                  "prefetch_k_iter must be >= 3. pfk=3 enters a depth-1 pipeline path; "
                  "pfk>=4 uses the depth-2 pipeline.");

    // Per-wave load counts for 2-wave producer layout (already include the
    // NUM_LOAD_GROUPS_PER_BK inner K-group factor).
    static constexpr int a_buffer_load_insts = NUM_LOAD_GROUPS_PER_BM * NUM_LOAD_GROUPS_PER_BK * slots / 2;
    static constexpr int b_buffer_load_insts = NUM_LOAD_GROUPS_PER_BN * NUM_LOAD_GROUPS_PER_BK * slots / 2;
    static constexpr int a_ds_read_insts = (COM_REP_M * COM_REP_K * W_M * W_K) / (opus::get_warp_size() * VEC_A);
    static constexpr int b_ds_read_insts = (COM_REP_N * COM_REP_K * W_N * W_K) / (opus::get_warp_size() * VEC_B);
    static constexpr int mma_insts = COM_REP_M * COM_REP_N * COM_REP_K;
};

#ifndef OPUS_GEMM_FLATMM_KARGS_DEFINED
#define OPUS_GEMM_FLATMM_KARGS_DEFINED
// Kernel arguments for the a16w16 flatmm pipeline.
// Layout: A[batch, M, K] bf16 row-major, B[batch, N, K] bf16 row-major
// (pre-transposed, no shuffle needed), C[batch, M, N] bf16 row-major.
struct opus_gemm_flatmm_kargs {
    const void* __restrict__ ptr_a;
    const void* __restrict__ ptr_b;
    void*       __restrict__ ptr_c;
    // FIXME: bias not yet implemented. HAS_BIAS=false is hardcoded in all
    // registered instances; this field is reserved for future use and must
    // be passed as nullptr. See gcnasm/opus_fmm/INTEGRATION.md "HAS_BIAS=true
    // not implemented" limitation.
    const void* __restrict__ ptr_bias;
    int m;
    int n;
    int k;
    int batch;
    int stride_a;        // A row stride along K, typically = k
    int stride_b;        // B row stride along K (B[N,K] layout), typically = k
    int stride_c;        // C row stride along N, typically = n
    int stride_a_batch;  // A per-batch element count, typically = m * k
    int stride_b_batch;  // B per-batch element count, typically = n * k
    int stride_c_batch;  // C per-batch element count, typically = m * n
};
#endif
