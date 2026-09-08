// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// BF16 persistent a16w16 pipeline (M-outer + N-fast XCD swizzle).
//
// Ported from the standalone reference kernel
// /home/blyu/demon_gcn/opus_gemm/gemm_a16w16_8wave_mouter.cc.
//
// Key differences vs the default split-barrier pipeline
// (opus_gemm_pipeline_a16w16_gfx950.cuh):
//   * Persistent grid: each WG handles m_per_wg tile_m × 1 tile_n. The
//     WG iterates over its m_per_wg tile_m values in an outer loop.
//   * XCD-local N-fast swizzle: within an XCD, consecutive launch-wave
//     WGs share the same m_grp and span all 8 tile_n stripes. This
//     keeps the A tile of a single m_grp resident in L2 across all
//     N stripes for the duration of one m_grp -- standalone bench
//     measures L2 hit rate going from ~50% (default split-barrier
//     with HipKittens W=8,C=32 swizzle) to ~80% on 32K×2K×7K BF16.
//   * Cluster store INSIDE the outer loop; vmcnt(0)+s_barrier between
//     iterations to drain prior store_c traffic before the next iter's
//     async_load. This is fully serial -- no overlap between current
//     iter's store and next iter's prologue load.
//   * No bias support (kargs lacks ptr_bias/stride_bias_batch). Bias
//     plumbing can be added later if a producer needs it; the launcher
//     in opus_gemm_a16w16_persistent_*.cu rejects non-empty bias up front.
//
// The K-loop body (prologue + main loop + 2-chunk epilogue) is
// byte-for-byte the same as the split-barrier reference (modulo
// the kargs type and the absence of the HAS_BIAS prefetch / fold).
//
// One intentional deviation from the mouter reference: the main loop's
// two `v_b[0] = load<VEC_B>(s_b[*][0], u_rb)` sites (at the v_c[1][1]
// mma group of "First tile" and "Second tile") were moved AFTER the
// `s_waitcnt_vmcnt + s_barrier` pair, matching the split-barrier
// pipeline ordering (opus_gemm_pipeline_a16w16_gfx950.cuh:411). The
// mouter "load before wait" ordering is unsafe for the tile-3 traits
// (B_M=B_N=128, b_buffer_load_insts=1) because the vmcnt budget is too
// tight to guarantee the specific s_b[*][0] write has landed before the
// ds_read fires -- silent intermittent corruption only manifests on
// tile-3 because tiles 0/1/2 have larger vmcnt budgets that absorb the
// race. See per-site comments at the two fix sites for details.
#pragma once

#include "opus_gemm_traits_a16w16_gfx950.cuh"

// Layout helpers (make_layout_ga_noscale et al) and CPOL_* macros live
// in opus_gemm_pipeline_a16w16_gfx950.cuh. They're __device__-only and
// the fused host TU never includes either pipeline header, so pulling
// in the split-barrier header here is safe (no ODR clash on the host
// pass).
#include "opus_gemm_pipeline_a16w16_gfx950.cuh"

// ============================================================================
// BF16 persistent GEMM kernel (a16w16)
// ============================================================================
template<typename Traits>
__global__ __launch_bounds__(Traits::BLOCK_SIZE, 2) void
gemm_a16w16_persistent_kernel(opus_gemm_persistent_kargs_gfx950 kargs) {
#ifdef __HIP_DEVICE_COMPILE__
#if defined(__gfx950__)
    using namespace opus;

    using T = opus::remove_cvref_t<Traits>;
    using D_A = typename T::D_A;
    using D_B = typename T::D_B;
    using D_C = typename T::D_C;
    using D_ACC = typename T::D_ACC;

    // ── XCD-local N-fast swizzle ────────────────────────────────────────
    // Default round-robin XCD assignment (fid % 8 == xcd_id) would put 4
    // consecutive WGs on the same XCD but each touching a different
    // m_grp + same tile_n; the per-XCD A working set then balloons to
    // (4 m_grp × m_per_wg × A_tile) which exceeds L2 capacity on M=32K
    // shapes. We re-map (bx, by) so that within an XCD, 8 consecutive
    // launch-wave WGs share the SAME m_grp and span 4 different tile_n
    // each launch wave (8 tile_n total across 2 launch waves), shrinking
    // the per-wave A working set so it fits L2.
    //
    // See plan §2.3 and the standalone reference's swizzle block at
    // gemm_a16w16_8wave_mouter.cc:277..290.
    //
    // Bijectivity & small-split_m correctness
    // ---------------------------------------
    // The swizzle requires grid.y * gx (= grid.y * num_tiles_n) be a
    // multiple of NUM_XCD so that (xcd_id, pos_xcd) covers
    // [0, NUM_XCD) × [0, grid.y*gx/NUM_XCD) bijectively. Equivalently
    // m_grp ∈ [0, NUM_XCD * m_grp_per_xcd). The launcher therefore pads
    // grid.y up to (NUM_XCD * m_grp_per_xcd = NUM_XCD * ceil(split_m/NUM_XCD)).
    // When split_m is already a NUM_XCD multiple (the large-M case the
    // swizzle is tuned for -- mouter's 32K case has split_m=32) the
    // pad is a no-op and there is zero perf overhead. When split_m <
    // NUM_XCD (small-M shapes like M=8192 N=8192 K=256 with B_M=256
    // B_N=128 → split_m=4), the pad multiplies grid.y by NUM_XCD/split_m,
    // and the WGs that land on m_grp >= split_m early-return below.
    // The early-return is a WAVE-UNIFORM branch (every lane in the wave
    // takes the same path -- m_grp is computed via readfirstlane and
    // is identical across the wave), so it lowers to a single
    // `s_cbranch_execz` with no warp divergence cost. Belt-and-
    // suspenders, the gmem buffer resources for A / B / C below also
    // clamp size to 0 on over-shoot, so even if the early-return were
    // ever skipped the buffer hardware would still drop any wayward
    // loads / stores.
    int bx0 = __builtin_amdgcn_readfirstlane(opus::block_id_x());
    int by0 = __builtin_amdgcn_readfirstlane(opus::block_id_y());
    int gx  = __builtin_amdgcn_readfirstlane(kargs.num_tiles_n);
    int fid = by0 * gx + bx0;
    constexpr int NUM_XCD = T::NUM_XCD;
    int xcd_id      = __builtin_amdgcn_readfirstlane(fid % NUM_XCD);
    int pos_xcd     = __builtin_amdgcn_readfirstlane(fid / NUM_XCD);
    int tile_n_id   = __builtin_amdgcn_readfirstlane(pos_xcd % gx);
    int m_grp_local = __builtin_amdgcn_readfirstlane(pos_xcd / gx);
    int m_grp       = __builtin_amdgcn_readfirstlane(
        xcd_id * kargs.m_grp_per_xcd + m_grp_local);

    // Wave-uniform early-return for over-shoot WGs in the small-split_m
    // case (grid.y padded > true split_m). No-op on the large-M case
    // where split_m % NUM_XCD == 0 (kargs.split_m_padded == split_m).
    if (m_grp >= kargs.split_m) return;

    int tile_m_lo = __builtin_amdgcn_readfirstlane(m_grp * kargs.m_per_wg);
    int tile_m_hi = __builtin_amdgcn_readfirstlane(tile_m_lo + kargs.m_per_wg);
    int col       = __builtin_amdgcn_readfirstlane(tile_n_id * T::B_N);

    int batch_id = opus::block_id_z();
    int wave_id = __builtin_amdgcn_readfirstlane(opus::thread_id_x() / get_warp_size());
    int lane_id = opus::thread_id_x() % get_warp_size();

    // B base is tile_n-bound: every M outer iter uses the same B[col:col+B_N, :]
    // sub-matrix. Built once outside the outer loop so the L2 line set stays warm.
    //
    // OOB-clamp protection
    // --------------------
    // The buffer resource (BR) `size` field is a hardware mask -- any
    // buffer_load_* / buffer_store_* with vaddr >= size returns 0 / is
    // silently dropped. We use this to defend against the rare case
    // where the XCD-local swizzle assigns an over-shoot WG (col >=
    // kargs.n in the degenerate split_m < NUM_XCD path). When col >=
    // kargs.n, `(kargs.n - col)` is <= 0; we clamp via std::max so the
    // BR size is 0, guaranteeing every load returns 0 (no fault, no
    // garbage). The over-shoot WG still loops through its M outer
    // iterations doing no-op loads / no-op stores, but produces no
    // visible side effect to global memory.
    int b_rows_remaining = (kargs.n > col) ? (kargs.n - col) : 0;
    auto g_b = make_gmem(
        reinterpret_cast<const D_B*>(kargs.ptr_b)
            + batch_id * kargs.stride_b_batch
            + col * kargs.stride_b,
        (unsigned)b_rows_remaining * kargs.stride_b * sizeof(D_B));

    int wave_id_m = wave_id / T::T_N;
    int wave_id_n = wave_id % T::T_N;

    auto u_ga = make_layout_ga_noscale<T>(lane_id, wave_id_m, wave_id_n, kargs.stride_a);
    auto u_sa = make_layout_sa_noscale<T>(lane_id, wave_id_m, wave_id_n);
    auto u_ra = make_layout_ra_noscale<T>(lane_id, wave_id_m);
    auto u_gb = make_layout_gb_noscale<T>(lane_id, wave_id_m, wave_id_n, kargs.stride_b);
    auto u_sb = make_layout_sb_noscale<T>(lane_id, wave_id_m, wave_id_n);
    auto u_rb = make_layout_rb_noscale<T>(lane_id, wave_id_n);

    // Shared memory: same double-buffer layout as the split-barrier pipeline.
    constexpr int smem_a_byte = T::smem_m_rep * (T::smem_linear_wave + T::smem_padding) * sizeof(D_A);
    __shared__ char smem_a[smem_a_byte * 4];
    smem<D_A> s_a[2][2] = {
        {make_smem(reinterpret_cast<D_A*>(smem_a)),
         make_smem(reinterpret_cast<D_A*>(smem_a + smem_a_byte))},
        {make_smem(reinterpret_cast<D_A*>(smem_a + 2 * smem_a_byte)),
         make_smem(reinterpret_cast<D_A*>(smem_a + 3 * smem_a_byte))}
    };
    constexpr int smem_b_byte = T::smem_n_rep * (T::smem_linear_wave + T::smem_padding) * sizeof(D_B);
    __shared__ char smem_b[smem_b_byte * 4];
    smem<D_B> s_b[2][2] = {
        {make_smem(reinterpret_cast<D_B*>(smem_b)),
         make_smem(reinterpret_cast<D_B*>(smem_b + smem_b_byte))},
        {make_smem(reinterpret_cast<D_B*>(smem_b + 2 * smem_b_byte)),
         make_smem(reinterpret_cast<D_B*>(smem_b + 3 * smem_b_byte))}
    };

    auto mma = make_tiled_mma<D_A, D_B, D_ACC>(
        seq<T::E_M, T::E_N, T::E_K>{},
        seq<T::T_M, T::T_N, T::T_K>{},
        seq<T::W_M, T::W_N, T::W_K>{},
        mfma_adaptor_swap_ab{});

    typename decltype(mma)::vtype_a v_a;
    typename decltype(mma)::vtype_b v_b[2];
    typename decltype(mma)::vtype_c v_c[2][2];
    // v_c clear moved into the outer loop body (per-iter reset).

    auto a_offset = [&](int half_tile_m, int tile_k) {
        return half_tile_m * T::HALF_B_M * kargs.stride_a + tile_k * T::B_K;
    };
    auto b_offset = [&](int half_tile_n, int tile_k) {
        return half_tile_n * T::HALF_B_N * kargs.stride_b + tile_k * T::B_K;
    };

    const int loops = ceil_div(kargs.k, T::B_K);

    // ── M outer loop: each iteration is a full split-barrier-style
    // computation (prologue + main + epilogue + cluster store) for one
    // tile_m. Between iterations, we drain stores fully (vmcnt(0)) and
    // cross-wave sync (s_barrier) before reloading A for the next
    // tile_m. This is fully serial: no overlap between current iter's
    // store and next iter's prologue load.
    for (int tile_m = tile_m_lo; tile_m < tile_m_hi; ++tile_m) {
        int row = __builtin_amdgcn_readfirstlane(tile_m * T::B_M);

        // A base: depends on tile_m, recomputed per outer iter.
        //
        // OOB-clamp protection (same scheme as g_b above): when the XCD
        // swizzle assigns an over-shoot WG with row >= kargs.m, clamp
        // the BR size to 0 so buffer_load_lds returns 0. Without this
        // clamp, (kargs.m - row) wraps negative -> huge unsigned ->
        // hardware lets the load through and faults on the dereference.
        int a_rows_remaining = (kargs.m > row) ? (kargs.m - row) : 0;
        auto g_a = make_gmem(
            reinterpret_cast<const D_A*>(kargs.ptr_a)
                + batch_id * kargs.stride_a_batch
                + row * kargs.stride_a,
            (unsigned)a_rows_remaining * kargs.stride_a * sizeof(D_A));
        // C base: same OOB-clamp story but stricter. g_c carries the
        // store path; an over-shoot WG (row >= kargs.m OR col >= kargs.n)
        // must NOT write any bytes to Y or it would corrupt other tiles
        // or write past Y's allocation entirely (the root cause of the
        // pre-fix M=8192 N=8192 K=256 "Memory access fault: Write
        // access to a read-only page"). BR size = bytes from current
        // base ptr to end of *this WG's valid Y rectangle* (clamped to
        // 0 on over-shoot). The store path's pred (T::HAS_OOB=true) or
        // the unconditional store (T::HAS_OOB=false) both ride on top
        // of this hardware OOB clamp -- store path tightens the bound
        // per-tile, BR loosens it to whole rectangle.
        int c_rows_remaining = (kargs.m > row) ? (kargs.m - row) : 0;
        int c_cols_remaining = (kargs.n > col) ? (kargs.n - col) : 0;
        // C is row-major in [M, N], stride_c = N. The legal byte range
        // from (row, col) onwards is bounded by either rows-remaining
        // (rows * stride) or cols-remaining within the last partial row
        // (cols * 1). To cover BOTH dimensions correctly, we size the BR
        // to cover the rest of THIS WG's row-band: c_rows_remaining
        // full rows minus the leading `col` columns. For the
        // over-shoot WG (rows_remaining == 0) the size is 0 -> hardware
        // drops every store.
        unsigned int c_size_bytes = 0;
        if (c_rows_remaining > 0 && c_cols_remaining > 0) {
            // Bytes from (row, col) to end of WG's row-band:
            //   (c_rows_remaining - 1) * stride_c + c_cols_remaining
            // covers the conservative upper bound of any valid
            // [row, row+B_M) x [col, col+B_N) tile this WG touches.
            c_size_bytes = ((unsigned)(c_rows_remaining - 1) * kargs.stride_c
                          + (unsigned)c_cols_remaining) * sizeof(D_C);
        }
        auto g_c = make_gmem(
            reinterpret_cast<D_C*>(kargs.ptr_c)
                + batch_id * kargs.stride_c_batch
                + row * kargs.stride_c
                + col,
            c_size_bytes);

        // Reset accumulators for this tile.
        clear(v_c[0][0]);
        clear(v_c[0][1]);
        clear(v_c[1][0]);
        clear(v_c[1][1]);

        int tic = 0, toc = 1;

        // ── Prologue (same shape as split-barrier; wave-id-m phase
        // shifter and B/A prefetch are byte-identical). The outer-iter
        // vmcnt(0)+s_barrier at the loop tail resets the wave phase,
        // so every iter re-establishes the shifter.
        async_load<T::VEC_B>(g_b, s_b[tic][0].ptr, u_gb, u_sb, b_offset(0, 0), opus::number<0>{}, opus::number<T::CACHECTL_B>{});
        async_load<T::VEC_A>(g_a, s_a[tic][0].ptr, u_ga, u_sa, a_offset(0, 0), opus::number<0>{}, opus::number<T::CACHECTL_A>{});
        async_load<T::VEC_B>(g_b, s_b[tic][1].ptr, u_gb, u_sb, b_offset(1, 0), opus::number<0>{}, opus::number<T::CACHECTL_B>{});
        async_load<T::VEC_A>(g_a, s_a[tic][1].ptr, u_ga, u_sa, a_offset(1, 0), opus::number<0>{}, opus::number<T::CACHECTL_A>{});

        if (wave_id_m == 1) __builtin_amdgcn_s_barrier();

        s_waitcnt_vmcnt(number<T::a_buffer_load_insts + T::b_buffer_load_insts>{});
        __builtin_amdgcn_s_barrier();

        async_load<T::VEC_B>(g_b, s_b[toc][0].ptr, u_gb, u_sb, b_offset(0, 1), opus::number<0>{}, opus::number<T::CACHECTL_B>{});
        async_load<T::VEC_A>(g_a, s_a[toc][0].ptr, u_ga, u_sa, a_offset(0, 1), opus::number<0>{}, opus::number<T::CACHECTL_A>{});
        async_load<T::VEC_B>(g_b, s_b[toc][1].ptr, u_gb, u_sb, b_offset(1, 1), opus::number<0>{}, opus::number<T::CACHECTL_B>{});

        s_waitcnt_vmcnt(number<T::a_buffer_load_insts + 2 * T::b_buffer_load_insts>{});
        __builtin_amdgcn_s_barrier();

        v_b[0] = load<T::VEC_B>(s_b[tic][0], u_rb);
        __builtin_amdgcn_s_barrier();

        // ── Main loop ──
        // Byte-for-byte matches the mouter reference (mouter.cc:394..483).
        // Differences vs the split-barrier pipeline are intentional and
        // measured (mouter at 1210 TFLOPS on 32K×2K×7K BF16 vs sb at 1149):
        //   * Only 4 sched_barrier(0) per K-tile pair (after v_c[0][0] and
        //     v_c[1][0] mmas), not 8. The other 4 mma sites omit it so the
        //     compiler can hoist v_b[0]/v_a loads above the next mma group.
        //   * v_c[1][1] mma group: the v_b[0] = load(...) for the next
        //     k-tile is emitted BEFORE the async_load + waitcnt + barrier
        //     (not after), letting ds_read overlap buffer_load issue.
        for(int tile = 0; tile < loops - 2; tile += 2) {
            // First tile
            v_a = load<T::VEC_A>(s_a[tic][0], u_ra);
            async_load<T::VEC_A>(g_a, s_a[toc][1].ptr, u_ga, u_sa, a_offset(1, tile + 1), opus::number<0>{}, opus::number<T::CACHECTL_A>{});
            s_waitcnt_lgkmcnt(number<T::a_ds_read_insts>{});
            __builtin_amdgcn_s_barrier();

            s_waitcnt_lgkmcnt(0_I);
            __builtin_amdgcn_s_setprio(1);
            v_c[0][0] = mma(v_a, v_b[0], v_c[0][0]);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);

            v_b[1] = load<T::VEC_B>(s_b[tic][1], u_rb);
            async_load<T::VEC_B>(g_b, s_b[tic][0].ptr, u_gb, u_sb, b_offset(0, tile + 2), opus::number<0>{}, opus::number<T::CACHECTL_B>{});
            __builtin_amdgcn_s_barrier();

            s_waitcnt_lgkmcnt(0_I);
            __builtin_amdgcn_s_setprio(1);
            v_c[0][1] = mma(v_a, v_b[1], v_c[0][1]);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            v_a = load<T::VEC_A>(s_a[tic][1], u_ra);
            async_load<T::VEC_A>(g_a, s_a[tic][0].ptr, u_ga, u_sa, a_offset(0, tile + 2), opus::number<0>{}, opus::number<T::CACHECTL_A>{});
            __builtin_amdgcn_s_barrier();

            s_waitcnt_lgkmcnt(0_I);
            __builtin_amdgcn_s_setprio(1);
            v_c[1][0] = mma(v_a, v_b[0], v_c[1][0]);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);

            // BUG-FIX (was: B_M=B_N=128 tile-3 race, intermittent ~50% rel
            // err on shapes with m_per_wg >= 2 and loops >= 4): move
            // `v_b[0] = load(s_b[toc][0], u_rb)` AFTER the
            // s_waitcnt_vmcnt + s_barrier, matching the split-barrier
            // pipeline ordering (see opus_gemm_pipeline_a16w16_gfx950.cuh:411).
            //
            // The previous "load before wait" ordering was a mouter
            // reference optimization meant to overlap ds_read with
            // buffer_load issue. It's UNSAFE because the next main-loop
            // iteration's pending async_load (at this loop iter's "Second
            // tile" line ~346) writes the SAME s_b[toc][0] LDS region,
            // and the waitcnt(a_buf+2*b_buf) here only bounds in-flight
            // vmem ops (it waits until <=N pending, not "complete this
            // specific load"). For tile-3 (b_buffer_load_insts=1), the
            // wait threshold is just 3 vmem ops -- frequently satisfied
            // by other in-flight loads, leaving the specific s_b[toc][0]
            // write still pending. The ds_read then returns stale data
            // and v_b[0] feeds wrong values into the next iter's v_c
            // mmas -- silent intermittent corruption only the tighter
            // tile-3 vmcnt budget exposes (tiles 0/1/2 have larger
            // budgets that absorb the race without manifesting).
            async_load<T::VEC_B>(g_b, s_b[tic][1].ptr, u_gb, u_sb, b_offset(1, tile + 2), opus::number<0>{}, opus::number<T::CACHECTL_B>{});
            s_waitcnt_vmcnt(number<T::a_buffer_load_insts + 2 * T::b_buffer_load_insts>{});
            __builtin_amdgcn_s_barrier();
            v_b[0] = load<T::VEC_B>(s_b[toc][0], u_rb);

            __builtin_amdgcn_s_setprio(1);
            v_c[1][1] = mma(v_a, v_b[1], v_c[1][1]);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            // Second tile
            v_a = load<T::VEC_A>(s_a[toc][0], u_ra);
            async_load<T::VEC_A>(g_a, s_a[tic][1].ptr, u_ga, u_sa, a_offset(1, tile + 2), opus::number<0>{}, opus::number<T::CACHECTL_A>{});
            s_waitcnt_lgkmcnt(number<T::a_ds_read_insts>{});
            __builtin_amdgcn_s_barrier();

            s_waitcnt_lgkmcnt(0_I);
            __builtin_amdgcn_s_setprio(1);
            v_c[0][0] = mma(v_a, v_b[0], v_c[0][0]);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);

            v_b[1] = load<T::VEC_B>(s_b[toc][1], u_rb);
            async_load<T::VEC_B>(g_b, s_b[toc][0].ptr, u_gb, u_sb, b_offset(0, tile + 3), opus::number<0>{}, opus::number<T::CACHECTL_B>{});
            __builtin_amdgcn_s_barrier();

            s_waitcnt_lgkmcnt(0_I);
            __builtin_amdgcn_s_setprio(1);
            v_c[0][1] = mma(v_a, v_b[1], v_c[0][1]);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            v_a = load<T::VEC_A>(s_a[toc][1], u_ra);
            async_load<T::VEC_A>(g_a, s_a[toc][0].ptr, u_ga, u_sa, a_offset(0, tile + 3), opus::number<0>{}, opus::number<T::CACHECTL_A>{});
            __builtin_amdgcn_s_barrier();

            s_waitcnt_lgkmcnt(0_I);
            __builtin_amdgcn_s_setprio(1);
            v_c[1][0] = mma(v_a, v_b[0], v_c[1][0]);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);

            // BUG-FIX (same as the v_b[0] reorder ~36 lines above): move
            // `v_b[0] = load(s_b[tic][0], u_rb)` AFTER the vmcnt+barrier.
            // See the long comment above for the full rationale -- the
            // "Second tile" half of the unrolled K=2 loop body has the
            // mirror issue: the next loop iter's prologue rewrites
            // s_b[tic][0], so reading it before the vmcnt-bound barrier
            // can return stale data on tile-3 (tighter vmcnt budget).
            async_load<T::VEC_B>(g_b, s_b[toc][1].ptr, u_gb, u_sb, b_offset(1, tile + 3), opus::number<0>{}, opus::number<T::CACHECTL_B>{});
            s_waitcnt_vmcnt(number<T::a_buffer_load_insts + 2 * T::b_buffer_load_insts>{});
            __builtin_amdgcn_s_barrier();
            v_b[0] = load<T::VEC_B>(s_b[tic][0], u_rb);

            __builtin_amdgcn_s_setprio(1);
            v_c[1][1] = mma(v_a, v_b[1], v_c[1][1]);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
        }

        // ── First epilogue chunk (K-tile = loops-2). ──
        // Byte-for-byte matches mouter.cc:485..520. No sched_barrier inside
        // (the inter-mma s_barrier provides sufficient scheduling fence;
        // adding sched_barrier(0) blocks the compiler from overlapping the
        // next group's v_b/v_a ds_read with the prior mma's barrier wait).
        {
            int tile = loops - 2;

            v_a = load<T::VEC_A>(s_a[tic][0], u_ra);
            async_load<T::VEC_A>(g_a, s_a[toc][1].ptr, u_ga, u_sa, a_offset(1, tile + 1), opus::number<0>{}, opus::number<T::CACHECTL_A>{});
            __builtin_amdgcn_s_barrier();
            s_waitcnt_lgkmcnt(0_I);

            __builtin_amdgcn_s_setprio(1);
            v_c[0][0] = mma(v_a, v_b[0], v_c[0][0]);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            v_b[1] = load<T::VEC_B>(s_b[tic][1], u_rb);
            __builtin_amdgcn_s_barrier();

            s_waitcnt_lgkmcnt(0_I);
            __builtin_amdgcn_s_setprio(1);
            v_c[0][1] = mma(v_a, v_b[1], v_c[0][1]);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            v_a = load<T::VEC_A>(s_a[tic][1], u_ra);
            s_waitcnt_vmcnt(number<T::a_buffer_load_insts + T::b_buffer_load_insts>{});
            __builtin_amdgcn_s_barrier();

            s_waitcnt_lgkmcnt(0_I);
            __builtin_amdgcn_s_setprio(1);
            v_c[1][0] = mma(v_a, v_b[0], v_c[1][0]);
            v_c[1][1] = mma(v_a, v_b[1], v_c[1][1]);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            tic ^= 1;
            toc ^= 1;
        }

        // ── Second epilogue chunk (K-tile = loops-1). ──
        // Byte-for-byte matches mouter.cc:546..577.
        {
            v_b[0] = load<T::VEC_B>(s_b[tic][0], u_rb);
            v_a = load<T::VEC_A>(s_a[tic][0], u_ra);
            s_waitcnt_vmcnt(number<T::a_buffer_load_insts>{});
            __builtin_amdgcn_s_barrier();

            s_waitcnt_lgkmcnt(0_I);
            __builtin_amdgcn_s_setprio(1);
            v_c[0][0] = mma(v_a, v_b[0], v_c[0][0]);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            v_b[1] = load<T::VEC_B>(s_b[tic][1], u_rb);
            s_waitcnt_vmcnt(0_I);
            __builtin_amdgcn_s_barrier();

            s_waitcnt_lgkmcnt(0_I);
            __builtin_amdgcn_s_setprio(1);
            v_c[0][1] = mma(v_a, v_b[1], v_c[0][1]);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            v_a = load<T::VEC_A>(s_a[tic][1], u_ra);
            __builtin_amdgcn_s_barrier();

            s_waitcnt_lgkmcnt(0_I);
            __builtin_amdgcn_s_setprio(1);
            v_c[1][0] = mma(v_a, v_b[0], v_c[1][0]);
            v_c[1][1] = mma(v_a, v_b[1], v_c[1][1]);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
        }

        if (wave_id_m == 0) __builtin_amdgcn_s_barrier();

        // ── Cluster store: cast and store all 4 v_c tiles back to gmem. ──
        auto p_coord_c = opus::make_tuple(wave_id_m, lane_id % mma.grpn_c, wave_id_n, lane_id / mma.grpn_c);
        auto u_gc = partition_layout_c<T::VEC_C>(mma, opus::make_tuple(kargs.stride_c, 1_I), p_coord_c);
        auto u_gc_m = partition_layout_c<T::VEC_C>(mma, opus::make_tuple(1_I, 0_I), p_coord_c);
        auto u_gc_n = partition_layout_c<T::VEC_C>(mma, opus::make_tuple(0_I, 1_I), p_coord_c);

        auto c_offset = [&](int half_tile_m, int half_tile_n) {
            return half_tile_m * T::HALF_B_M * kargs.stride_c + half_tile_n * T::HALF_B_N;
        };

        auto store_c = [&](auto& vc, int half_tile_m, int half_tile_n) {
            int g_c_offset = c_offset(half_tile_m, half_tile_n);
            int m_base = row + half_tile_m * T::HALF_B_M;
            int n_base = col + half_tile_n * T::HALF_B_N;

            if constexpr (T::HAS_OOB) {
                auto pred = [&](auto... ids) {
                    return (m_base + u_gc_m(ids...)) < kargs.m && (n_base + u_gc_n(ids...)) < kargs.n;
                };
                if constexpr (std::is_same_v<D_C, D_ACC>) {
                    store_if<T::VEC_C>(g_c, pred, vc, u_gc, g_c_offset, opus::number<CPOL_NT>{});
                } else {
                    auto vc_out = cast<D_C>(vc);
                    store_if<T::VEC_C>(g_c, pred, vc_out, u_gc, g_c_offset, opus::number<CPOL_NT>{});
                }
            } else {
                if constexpr (std::is_same_v<D_C, D_ACC>) {
                    store<T::VEC_C>(g_c, vc, u_gc, g_c_offset, opus::number<CPOL_NT>{});
                } else {
                    auto vc_out = cast<D_C>(vc);
                    store<T::VEC_C>(g_c, vc_out, u_gc, g_c_offset, opus::number<CPOL_NT>{});
                }
            }
        };

        store_c(v_c[0][0], 0, 0);
        store_c(v_c[0][1], 0, 1);
        store_c(v_c[1][0], 1, 0);
        store_c(v_c[1][1], 1, 1);

        // ── M-outer iter barrier: drain all stores fully (vmcnt(0))
        // and cross-wave sync (s_barrier) before reloading A for the
        // next tile_m. Both wave groups hit this same barrier, so the
        // relative phase they entered is preserved (no phase reset).
        s_waitcnt_vmcnt(0_I);
        __builtin_amdgcn_s_barrier();
    }  // end M outer loop
#else
    // Non-gfx950 device pass: empty stub. The Python import guard
    // (aiter/ops/opus/_arch.py) plus the host arch router in
    // opus_gemm.cu prevent runtime dispatch from ever reaching here.
    (void)kargs;
#endif // __gfx950__
#endif // __HIP_DEVICE_COMPILE__
}
