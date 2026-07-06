// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Traits for a16w16 (bf16) pipelines. This header carries the traits and
// kargs structs for every a16w16 pipeline family on gfx950:
//
//   opus_gemm_a16w16_traits_gfx950<..., TILE, WAVE, HAS_BIAS=false, D_BIAS=void>
//     Split-barrier pipeline used by opus_gemm_pipeline_a16w16_gfx950.cuh.
//     Configurable TILE (T_M, T_N, T_K) and WAVE (W_M, W_N, W_K). 4-tuple DTYPE.
//     HAS_BIAS / D_BIAS default off so existing instantiations remain valid.
//     When HAS_BIAS=true the kernel reads kargs.ptr_bias as a D_BIAS* vector
//     along N (shape [N] or [batch, N]; F.linear convention; selected via
//     kargs.stride_bias_batch).
//
//   opus_gemm_a16w16_flatmm_traits_gfx950<..., MFMA, WG_PER_CU, HAS_BIAS>
//     4-wave warp-specialized pipeline (2 producer + 2 consumer) used by
//     opus_gemm_pipeline_a16w16_flatmm_gfx950.cuh. Derives prefetch depth dynamically
//     from the LDS budget. Locked T_M=2/T_N=1/T_K=1. 5-tuple DTYPE.
//     Ported from gcnasm/opus_fmm/flatmm_a16w16_4wave_wasp.cc.
//
//   opus_flatmm_splitk_traits_gfx950<..., MFMA, WG_PER_CU, HAS_BIAS, HAS_OOB>
//     Split-K variant of the flatmm pipeline. Main kernel writes fp32
//     workspace; a reduce kernel sums splits + casts to bf16 C.
//
//   opus_gemm_a16w16_persistent_traits_gfx950<..., TILE, WAVE, HAS_OOB,
//                                              CACHECTL_A, CACHECTL_B>
//     M-outer + N-fast XCD swizzle persistent pipeline.
//
//   opus_gemm_a16w16_mono_tile_traits_gfx950<..., DTYPE, VEC>
//     Single-MMA-per-K mono-tile pipeline. Locked geometry (T_M=2, T_N=4,
//     T_K=1, W=16x16x32, VEC=8, BLOCK_SIZE=512). Divisible-only; no HAS_BIAS
//     / HAS_OOB template parameters.
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
        typename WAVE_,
        bool HAS_BIAS_ = false,
        typename D_BIAS_ = void,
        bool HAS_OOB_ = true,
        int CACHECTL_A_ = 0,
        int CACHECTL_B_ = 17,
        int SWIZZLE_W_ = 8,
        int SWIZZLE_C_ = 32>
struct opus_gemm_a16w16_traits_gfx950 {
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

    static constexpr bool HAS_BIAS = HAS_BIAS_;
    static constexpr bool HAS_OOB = HAS_OOB_;
    using D_BIAS = std::conditional_t<std::is_same_v<D_BIAS_, void>, D_C, D_BIAS_>;

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

    // Cache policy for A/B loads (CDNA4 ISA Table 49).
    // Values: 0=LRU, 1=SC0(Group,LLC Evict), 2=NT(Stream), 17=SC0+SC1(BYPASS_L2).
    // Tuner selects optimal CPOL per shape from multiple kid variants.
    static constexpr int CACHECTL_A = CACHECTL_A_;
    static constexpr int CACHECTL_B = CACHECTL_B_;

    // HipKittens XCD swizzle parameters (Algorithm 1, MI350 = 8 XCDs)
    static constexpr int NUM_XCD = 8;
    static constexpr int SWIZZLE_W = SWIZZLE_W_;
    static constexpr int SWIZZLE_C = SWIZZLE_C_;
};

#ifndef OPUS_GEMM_NOSCALE_KARGS_GFX950_DEFINED
#define OPUS_GEMM_NOSCALE_KARGS_GFX950_DEFINED
// Shared kargs struct between a16w16 split-barrier and a8w8 noscale launchers.
// Must match the definition in opus_gemm_traits_a8w8_noscale.cuh exactly
// (header-include order chooses which TU gets the canonical definition).
//
// bias fields:
//   ptr_bias          : null when HAS_BIAS=false; otherwise points to D_BIAS
//                       buffer holding the per-output-feature bias vector
//                       (F.linear convention). dtype matches T::D_BIAS
//                       (== T::D_C in default instantiations).
//   stride_bias_batch : in elements of D_BIAS.
//                       0  -> bias is shape [N] and broadcast across batches;
//                       N  -> bias is shape [batch, N], one set per batch.
//                       Reduce / split-barrier kernels read this stride only;
//                       host validates shape <-> stride consistency.
struct opus_gemm_noscale_kargs_gfx950 {
    const void* __restrict__ ptr_a;
    const void* __restrict__ ptr_b;
    void* __restrict__ ptr_c;
    const void* __restrict__ ptr_bias;
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
    int stride_bias_batch;
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
struct opus_gemm_a16w16_flatmm_traits_gfx950 {
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
    // pfk<3 and break static_asserts.
    //
    // All aiter a16w16 kernels are gfx950-only today. Three-layer enforcement:
    //   1. Python: aiter/ops/opus/__init__.py calls _arch._detect_arch({"gfx950"})
    //      at import time. On non-gfx950 the import still succeeds (so it
    //      cannot break the surrounding `from aiter.ops.opus import *` in
    //      aiter/__init__.py) but gemm_a16w16_opus / opus_gemm_a16w16_tune
    //      are replaced with stubs that raise RuntimeError on call, plus a
    //      one-shot RuntimeWarning at import. Helper is reusable for future
    //      opus submodules with different supported sets.
    //   2. Host:   opus_dispatch_a16w16<T> / opus_a16w16_tune_dispatch<T> in
    //      opus_gemm.cu are arch routers built on opus_get_gfx_arch(). Only
    //      the gfx950 branch is wired up today (delegates to
    //      opus_dispatch_a16w16_gfx950<T>); other archs return TORCH_CHECK
    //      fail with a 'pipeline TBD' message. Future archs are added by
    //      extending OpusGfxArch + adding a per-arch dispatch function.
    //   3. Device: each __global__ kernel body wraps real code in
    //      #if defined(__gfx950__) so non-gfx950 device passes (in multi-arch
    //      wheels like GPU_ARCHS='gfx942;gfx950') compile to an empty stub.
    //      Combined with layer 1/2 the empty stub is unreachable at runtime.
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
struct opus_gemm_flatmm_kargs_gfx950 {
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

// ============================================================================
// Split-K FlatMM traits (two-kernel variant: main writes fp32 workspace,
// reduce kernel sums splits + casts to bf16 C)
// ============================================================================
//
// 7 template parameters match opus_gemm_a16w16_flatmm_traits_gfx950, with
// additional static_assert D_C=float (splitk main kernel writes fp32
// partial sums to workspace). Ported from
// gcnasm/opus_fmm/flatmm_a16w16_4wave_wasp_splitk.cc lines 34-143.

template<int BLOCK_SIZE_,   // workgroup size (locked to 256)
        typename BLOCK_,    // opus::seq<B_M, B_N, B_K>
        typename DTYPE_,    // opus::tuple<D_A, D_B, D_C, D_ACC, D_BIAS>, D_C MUST be float
        typename VEC_,      // opus::seq<VEC_A, VEC_B, VEC_C>
        typename MFMA_,     // opus::seq<W_M, W_N, W_K>
        int WG_PER_CU_,
        bool HAS_BIAS_,
        bool HAS_OOB_ = true>
struct opus_flatmm_splitk_traits_gfx950 {
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

    // Split-K writes fp32 partial sums; reduce kernel later casts to bf16.
    static_assert(std::is_same_v<D_C, float>,
                  "splitk kernel requires D_C = float for fp32 workspace");

    // Warp-specialized 4-wave layout: 2 producer + 2 consumer. T_K=1 locked.
    static constexpr int T_M = 2;
    static constexpr int T_N = 1;
    static constexpr int T_K = 1;

    static_assert(T_K == 1, "splitk requires T_K=1");
    static_assert(T_M == 2, "splitk requires T_M=2 (ra layout depends on it)");
    static_assert(T_N == 1, "splitk requires T_N=1 (consumer waves share N slab)");
    static_assert(BLOCK_SIZE == 256,
                  "splitk requires BLOCK_SIZE=256 (4 waves: 2 producer + 2 consumer)");
    static_assert(BLOCK_SIZE == 4 * opus::get_warp_size(),
                  "splitk BLOCK_SIZE must cover exactly 4 waves");

    static constexpr int W_M = opus::get<0>(MFMA{});
    static constexpr int W_N = opus::get<1>(MFMA{});
    static constexpr int W_K = opus::get<2>(MFMA{});

    static_assert(W_M < 32,
                  "splitk ra layout only implemented for W_M<32 (LOAD_GROUP_M_LANE=1)");

    static constexpr int LOAD_GROUP_M = (W_M >= 32) ? 64 : 32;
    static constexpr int LOAD_GROUP_N = (W_N >= 32) ? 64 : 32;
    static constexpr int LOAD_GROUP_K = W_K * 2;
    static constexpr int LOAD_GROUP_M_LANE = (W_M >= 32) ? 4 : 1;
    static constexpr int LOAD_GROUP_N_LANE = (W_N >= 32) ? 4 : 1;
    static constexpr int NUM_LOAD_GROUPS_PER_BM = B_M / LOAD_GROUP_M;
    static constexpr int NUM_LOAD_GROUPS_PER_BN = B_N / LOAD_GROUP_N;
    static constexpr int NUM_LOAD_GROUPS_PER_BK = B_K / LOAD_GROUP_K;
    static_assert(NUM_LOAD_GROUPS_PER_BM * LOAD_GROUP_M == B_M);
    static_assert(NUM_LOAD_GROUPS_PER_BN * LOAD_GROUP_N == B_N);
    static_assert(NUM_LOAD_GROUPS_PER_BK * LOAD_GROUP_K == B_K);

    static constexpr int COM_REP_M = B_M / (W_M * T_M);
    static constexpr int COM_REP_N = B_N / (W_N * T_N);
    static constexpr int COM_REP_K = B_K / (W_K * T_K);
    static constexpr int VEC_A = opus::get<0>(VEC{});
    static constexpr int VEC_B = opus::get<1>(VEC{});
    static constexpr int VEC_C = opus::get<2>(VEC{});

    static constexpr bool HAS_BIAS = HAS_BIAS_;
    static constexpr bool HAS_OOB = HAS_OOB_;

    static_assert(VEC_A == 16 / sizeof(D_A));
    static constexpr int smem_linear_wave_per_async_load = opus::get_warp_size() * 16 / sizeof(D_A);
    static constexpr int smem_sub = smem_linear_wave_per_async_load / LOAD_GROUP_K;
    static constexpr int slots = LOAD_GROUP_M / smem_sub;
    static constexpr int smem_padding = (W_M >= 32) ? 16 / sizeof(D_A) : 2 * 16 / sizeof(D_A);
    static constexpr int smem_per_group_load_size = slots * (smem_linear_wave_per_async_load + smem_padding) * sizeof(D_A);

    // Dynamic prefetch depth from LDS budget. gfx950 = 160 KiB (hardcoded
    // as in opus_gemm_a16w16_flatmm_traits_gfx950 above; host pass cannot use
    // opus::get_smem_size() because it's device-gated).
    static constexpr int WG_PER_CU = WG_PER_CU_;
    static constexpr int LDS_SIZE_TOTAL = 163840;
    static constexpr int max_lds_size_per_wg = LDS_SIZE_TOTAL / WG_PER_CU_;
    static constexpr int per_block_iter_lds_size = (NUM_LOAD_GROUPS_PER_BM + NUM_LOAD_GROUPS_PER_BN) * NUM_LOAD_GROUPS_PER_BK * smem_per_group_load_size;
    static constexpr int prefetch_k_iter = max_lds_size_per_wg / per_block_iter_lds_size;
    static_assert(prefetch_k_iter >= 3,
                  "prefetch_k_iter must be >= 3. pfk=3 enters a depth-1 pipeline path; "
                  "pfk>=4 uses the depth-2 pipeline.");

    static constexpr int a_buffer_load_insts = NUM_LOAD_GROUPS_PER_BM * NUM_LOAD_GROUPS_PER_BK * slots / 2;
    static constexpr int b_buffer_load_insts = NUM_LOAD_GROUPS_PER_BN * NUM_LOAD_GROUPS_PER_BK * slots / 2;
    static constexpr int a_ds_read_insts = (COM_REP_M * COM_REP_K * W_M * W_K) / (opus::get_warp_size() * VEC_A);
    static constexpr int b_ds_read_insts = (COM_REP_N * COM_REP_K * W_N * W_K) / (opus::get_warp_size() * VEC_B);
    static constexpr int mma_insts = COM_REP_M * COM_REP_N * COM_REP_K;
};

// ============================================================================
// Persistent a16w16 traits (M-outer + N-fast XCD swizzle)
// ============================================================================
//
// Pipeline: opus_gemm_pipeline_a16w16_persistent_gfx950.cuh (ported from the
// standalone reference kernel gemm_a16w16_8wave_mouter.cc).
//
// Layout: each WG handles m_per_wg tile_m × 1 tile_n (M outer loop). Within
// one XCD, consecutive launch-wave WGs share the same m_grp and span all 8
// tile_n stripes; this lets the A tile stay resident in L2 across 8 N tiles
// for the duration of one m_grp.
//
// CACHECTL_A / CACHECTL_B default to (0, 17) = (LRU, BYPASS_L2), matching
// the split-barrier traits. Other (cachectl_a, cachectl_b) combos are
// exposed as separate KIDs by the tuner.
//
// NUM_XCD is fixed to 8 (gfx950 = MI350); the swizzle is hard-wired to
// N-fast inside the persistent pipeline body and does NOT take SWIZZLE_W/C
// parameters (those belong to the HipKittens split-barrier swizzle, which
// is a different, orthogonal optimization).
template<int BLOCK_SIZE_,
        typename BLOCK_,
        typename DTYPE_,
        typename VEC_,
        typename TILE_,
        typename WAVE_,
        bool HAS_OOB_ = true,
        int CACHECTL_A_ = 0,
        int CACHECTL_B_ = 17>
struct opus_gemm_a16w16_persistent_traits_gfx950 {
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

    static constexpr bool HAS_OOB = HAS_OOB_;

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

    // Cache policy for A/B loads (CDNA4 ISA Table 49).
    static constexpr int CACHECTL_A = CACHECTL_A_;
    static constexpr int CACHECTL_B = CACHECTL_B_;

    // MI350 = 8 XCDs. Persistent swizzle uses N-fast within each XCD
    // (see kargs.num_tiles_n / kargs.m_grp_per_xcd).
    static constexpr int NUM_XCD = 8;
};

#ifndef OPUS_GEMM_PERSISTENT_KARGS_GFX950_DEFINED
#define OPUS_GEMM_PERSISTENT_KARGS_GFX950_DEFINED
// Kernel arguments for the a16w16 persistent pipeline.
//
// Beyond the usual (M, N, K, batch, stride_*) fields this struct carries
// three persistent-specific values that the launcher computes on the host:
//   m_per_wg       : how many tile_m iters one WG covers in its M outer loop
//                    (= num_tiles_m / split_m, where split_m is the WG grid
//                    extent along M chosen by the launcher heuristic).
//   num_tiles_n    : = grid.x (number of tile_n stripes; the in-kernel
//                    swizzle reads this back instead of using gridDim.x).
//   split_m        : un-padded m_grp count requested by the launcher
//                    (= num_tiles_m / m_per_wg). Used by the kernel's
//                    wave-uniform early-return guard for the over-shoot
//                    WGs introduced when grid.y is padded up to a
//                    NUM_XCD multiple (small-split_m case; no-op when
//                    split_m % NUM_XCD == 0 so zero perf cost on the
//                    large-M shapes the swizzle is tuned for).
//   m_grp_per_xcd  : = ceil_div(split_m, NUM_XCD); used by the in-kernel
//                    swizzle to compute the m_grp owned by this XCD slot.
//                    Combined with grid.y = NUM_XCD * m_grp_per_xcd
//                    (the padded grid the launcher uses), the swizzle
//                    is bijective onto [0, NUM_XCD * m_grp_per_xcd).
//
// NOTE: persistent pipeline does NOT support bias yet (matches the original
// mouter.cc reference); this struct intentionally omits ptr_bias /
// stride_bias_batch to keep the kargs surface minimal.
struct opus_gemm_persistent_kargs_gfx950 {
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
    int m_per_wg;
    int num_tiles_n;
    int split_m;
    int m_grp_per_xcd;
};
#endif

#ifndef OPUS_GEMM_SPLITK_WS_HANDLE_DEFINED
#define OPUS_GEMM_SPLITK_WS_HANDLE_DEFINED
// Indirection slot for the split-K fp32 workspace pointer. Captured HIP
// graphs hold the slot address (stable), not the workspace ptr, so a
// post-capture grow + hipFree of the old buffer doesn't dangle the graph.
struct opus_splitk_ws_handle {
    void*         ptr;    // current backing workspace; null until first grow
    unsigned long bytes;  // current capacity in bytes
};
#endif

#ifndef OPUS_GEMM_FLATMM_SPLITK_KARGS_DEFINED
#define OPUS_GEMM_FLATMM_SPLITK_KARGS_DEFINED
// Kernel arguments for the a16w16 flatmm split-K pipeline.
//
// Main kernel writes fp32 partial results to *ws_handle->ptr, laid out as
// [split_k, B, padded_M, padded_N] (tile-aligned, no per-thread pred on
// store). Reduce kernel consumes it and writes C[B, M, N].
//
// Host must satisfy split_k * pfk * B_K <= K (launcher auto-clamps otherwise).
struct opus_gemm_flatmm_splitk_kargs_gfx950 {
    const void* __restrict__ ptr_a;         // bf16 [B, M, K]
    const void* __restrict__ ptr_b;         // bf16 [B, N, K] (pre-transposed)
    const opus_splitk_ws_handle* __restrict__ ws_handle; // deref at kernel entry
    void*       __restrict__ ptr_c;         // bf16 [B, M, N] (filled by reduce kernel)
    // bias is consumed only by the reduce kernel (main kernel ignores it).
    // ptr_bias = nullptr when HAS_BIAS=false; dtype matches D_BIAS (== D_C
    // for the user-facing matched-dtype convention).
    // stride_bias_batch in elements of D_BIAS:
    //   0  -> bias shape [N], broadcast across batches;
    //   N  -> bias shape [batch, N], per-batch column vector (F.linear).
    const void* __restrict__ ptr_bias;
    int m;
    int n;
    int k;
    int batch;
    int split_k;                            // runtime split factor (literal KBatch)
    int stride_a;                           // stride in unit of pixel (typically = k)
    int stride_b;                           // B[N, K] row stride
    int stride_ws;                          // = padded_N
    int stride_c;                           // = N (for reduce kernel output)
    int stride_a_batch;
    int stride_b_batch;
    int stride_ws_batch;                    // = padded_M * padded_N
    int stride_c_batch;                     // = M * N
    int stride_bias_batch;                  // 0 (broadcast [N]) or N ([batch, N])
};
#endif

// ============================================================================
// Mono-tile a16w16 traits (single MMA across the full B_M x B_N tile per K)
// ============================================================================
//
// Pipeline: opus_gemm_pipeline_a16w16_mono_tile_gfx950.cuh.
//
// Locked geometry, derived in the kernel itself:
//   * T_M = 2, T_N = 4, T_K = 1  -> 8 waves / WG -> BLOCK_SIZE = 8 * 64 = 512.
//   * W_M = 16, W_N = 16, W_K = 32 (MFMA 16x16x32 BF16).
//   * VEC_A = VEC_B = VEC_C = 8.
//
// Constraints (mirror the kernel-internal static_asserts in
// gemm_a16w16_mono_tile_kernel_template.hpp; static_asserts here surface
// invalid tiles at traits instantiation):
//   * BLOCK_SIZE == 512 (T_M * T_N * T_K * warp_size = 2 * 4 * 1 * 64).
//   * B_M % (W_M * T_M) == 0  ->  B_M % 32 == 0.
//   * B_N % (W_N * T_N) == 0  ->  B_N % 64 == 0.
//   * B_K % (W_K * T_K) == 0  ->  B_K % 32 == 0.
//   * smem_linear_wave / B_K must divide evenly (B_K must divide 512).
//   * smem_m_rep = B_M / smem_sub must be >= 8 and divisible by 8 (num_waves).
//   * smem_n_rep = B_N / smem_sub must be >= 8 and divisible by 8.
//   * E_N = B_N / (W_N * T_N) must be divisible by (T_N / T_M) = 2.
//   * ra layout requires (smem_sub / (W_M / T_N)) to divide E_M evenly.
//   * D_A == D_B; D_A locked to bf16_t in current instances.
//   * D_C in { bf16_t, fp32_t }.
//
// No HAS_BIAS / HAS_OOB / CACHECTL template parameters: the mono-tile
// launcher rejects non-empty bias up front and is only instantiated for
// tile-aligned shapes (the launcher enforces M%B_M==N%B_N==K%B_K==0).

template<int BLOCK_SIZE_,
        typename BLOCK_,
        typename DTYPE_,
        typename VEC_>
struct opus_gemm_a16w16_mono_tile_traits_gfx950 {
    using BLOCK = opus::remove_cvref_t<BLOCK_>;
    using DTYPE = opus::remove_cvref_t<DTYPE_>;
    using VEC   = opus::remove_cvref_t<VEC_>;

    static constexpr int BLOCK_SIZE = BLOCK_SIZE_;

    static constexpr int B_M = opus::get<0>(BLOCK{});
    static constexpr int B_N = opus::get<1>(BLOCK{});
    static constexpr int B_K = opus::get<2>(BLOCK{});

    using D_A   = opus::tuple_element_t<0, DTYPE>;
    using D_B   = opus::tuple_element_t<1, DTYPE>;
    using D_C   = opus::tuple_element_t<2, DTYPE>;
    using D_ACC = opus::tuple_element_t<3, DTYPE>;
    static_assert(std::is_same<D_A, D_B>::value,
                  "mono_tile requires D_A == D_B");

    static constexpr int VEC_A = opus::get<0>(VEC{});
    static constexpr int VEC_B = opus::get<1>(VEC{});
    static constexpr int VEC_C = opus::get<2>(VEC{});

    // ── Locked tile/wave geometry (kernel-internal constants) ──
    static constexpr int T_M = 2;
    static constexpr int T_N = 4;
    static constexpr int T_K = 1;
    static constexpr int W_M = 16;
    static constexpr int W_N = 16;
    static constexpr int W_K = 32;

    // BLOCK_SIZE = (T_M * T_N * T_K) * warp_size = 8 * 64 = 512.
    static_assert(BLOCK_SIZE == 512,
                  "mono_tile requires BLOCK_SIZE = 512 (8 waves * 64 lanes)");

    // ── Locked vector widths ──
    static_assert(VEC_A == 8 && VEC_B == 8 && VEC_C == 8,
                  "mono_tile requires VEC_A = VEC_B = VEC_C = 8");
    static_assert(VEC_A == 16 / sizeof(D_A),
                  "mono_tile VEC_A must equal 16 / sizeof(D_A) (= 8 for bf16)");

    // ── Block tile divisibility ──
    static_assert(B_M % (W_M * T_M) == 0,
                  "mono_tile requires B_M divisible by W_M * T_M = 32");
    static_assert(B_N % (W_N * T_N) == 0,
                  "mono_tile requires B_N divisible by W_N * T_N = 64");
    static_assert(B_K % (W_K * T_K) == 0,
                  "mono_tile requires B_K divisible by W_K * T_K = 32");

    // ── Derived MMA repeat counts ──
    static constexpr int E_M = B_M / (W_M * T_M);
    static constexpr int E_N = B_N / (W_N * T_N);
    static constexpr int E_K = B_K / (W_K * T_K);

    // E_N must be divisible by (T_N / T_M) for the rb layout grouping.
    static_assert((E_N * T_M) % T_N == 0,
                  "mono_tile requires E_N divisible by (T_N / T_M) = 2 "
                  "-> B_N % 128 == 0 with the locked T_M=2,T_N=4 geometry");

    // ── LDS layout ──
    static constexpr int smem_linear_wave = 64 * 16 / sizeof(D_A); // 512 for bf16
    static_assert(smem_linear_wave % B_K == 0,
                  "mono_tile requires B_K to divide smem_linear_wave (=512 for bf16)");
    static constexpr int smem_sub = smem_linear_wave / B_K;
    static constexpr int smem_m_rep = B_M / smem_sub;
    static constexpr int smem_n_rep = B_N / smem_sub;
    static constexpr int smem_padding = 2 * 16 / sizeof(D_A);

    static constexpr int num_waves = BLOCK_SIZE / 64;  // 8
    static_assert(B_M % smem_sub == 0,
                  "mono_tile: B_M / smem_sub must be integer (smem_m_rep)");
    static_assert(B_N % smem_sub == 0,
                  "mono_tile: B_N / smem_sub must be integer (smem_n_rep)");
    static_assert(smem_m_rep >= num_waves && (smem_m_rep % num_waves) == 0,
                  "mono_tile: smem_m_rep must be >= 8 and divisible by 8 (num_waves)");
    static_assert(smem_n_rep >= num_waves && (smem_n_rep % num_waves) == 0,
                  "mono_tile: smem_n_rep must be >= 8 and divisible by 8 (num_waves)");

    // ra layout: smem_sub_e_m = smem_sub / (W_M / T_N) ; E_M must divide cleanly.
    static_assert((W_M % T_N) == 0,
                  "mono_tile: W_M must be divisible by T_N (W_M=16, T_N=4)");
    static constexpr int smem_sub_e_m = smem_sub / (W_M / T_N);
    static_assert(smem_sub_e_m > 0 && (E_M % smem_sub_e_m) == 0,
                  "mono_tile: E_M must be divisible by smem_sub / (W_M/T_N)");

    // ── Buffer / ds_read instruction counts (mirror kernel_traits<UT> in the
    //    upstream template; recomputed here so codegen can sanity-print). ──
    static constexpr int a_buffer_load_insts = B_M * B_K / (BLOCK_SIZE * VEC_A);
    static constexpr int b_buffer_load_insts = B_N * B_K / (BLOCK_SIZE * VEC_B);
    static constexpr int a_ds_read_insts = (E_M * E_K * W_M * W_K) / (64 * VEC_A);
    static constexpr int b_ds_read_insts = (E_N * E_K * W_N * W_K) / (64 * VEC_B);
};

#ifndef OPUS_GEMM_MONO_TILE_KARGS_GFX950_DEFINED
#define OPUS_GEMM_MONO_TILE_KARGS_GFX950_DEFINED
// Kernel arguments for the a16w16 mono-tile pipeline.
//
// Mirrors `opus_gemm_kargs` from the upstream standalone reference
// (yk_gcn/opus_gemm/bf16_gemm/gemm_defs.h). The kernel template casts
// ptr_a, ptr_b, ptr_c to the traits-typed pointers internally.
//
// No bias / no splitK / no workspace fields: mono-tile is intrinsically
// non-OOB (launcher enforces M%B_M==N%B_N==K%B_K==0) and the launcher
// rejects any non-empty bias up front.
struct opus_gemm_mono_tile_kargs_gfx950 {
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
