// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

// BMM a8w8 mxscale (e8m0) GEMM pipeline: scale-accumulation helpers plus all
// batched GEMM kernels (main / K1024 / preload-SFA / split-K).
// Reuses the shared a8w8_scale layout infrastructure from the base header.
#include "opus_gemm_pipeline_a8w8_scale_gfx950.cuh"
// pack_e8m0x4 (broadcast e8m0 -> x4 word) is shared via opus_gemm_utils.cuh
// (pulled in transitively), so both this header and the flatmm split-K pipeline
// reference one definition instead of a per-header copy.

#ifdef __HIP_DEVICE_COMPILE__

template<typename T, int ELEM_C, typename Mma, typename VA, typename VB,
         typename VSFA, typename VSFB, typename VC>
OPUS_D void mma_scale_accum(Mma& mma, const VA& v_a, const VB& v_b,
                            const VSFA& v_sfa, const VSFB& v_sfb, VC& v_c) {
    using D_ACC = typename T::D_ACC;
    using D_SF = typename T::D_SF;
    if constexpr (std::is_same_v<D_SF, unsigned char>) {
        // DSV4 scale is 128-block. The gfx950 scaled MFMA consumes 32-block
        // E8M0 scale bytes; replicate one checkpoint byte across all four
        // subblocks in the packed scale word to preserve 128-block semantics.
        static_assert(T::B_K == T::GROUP_K, "e8m0 path assumes one K scale block per B_K");
        static_assert(T::HALF_B_N == T::GROUP_N, "e8m0 path assumes one B scale per half-tile");
        if constexpr (T::E_M == 1) {
            const int scale_a = pack_e8m0x4(v_sfa[0]);
            const int scale_b = pack_e8m0x4(v_sfb[0]);
            v_c = mma(v_a, v_b, v_c, scale_a, scale_b, 0_I, 0_I);
        } else {
            using MMA = typename Mma::MMA;
            constexpr int a_len = Mma::mma_a_len;
            constexpr int b_len = Mma::mma_b_len;
            constexpr int c_len = Mma::mma_c_len;
            constexpr int rep_n_per_scale = T::GROUP_N / (T::W_N * T::T_N);
            static_assert(T::GROUP_N % (T::W_N * T::T_N) == 0);
            opus::static_for<T::E_M>([&](auto im_c) {
                constexpr int im = decltype(im_c)::value;
                opus::static_for<T::E_N>([&](auto in_c) {
                    constexpr int in = decltype(in_c)::value;
                    opus::static_for<T::E_K>([&](auto ik_c) {
                        constexpr int ik = decltype(ik_c)::value;
                        const int scale_a = pack_e8m0x4(v_sfa[im * T::E_K + ik]);
                        const int scale_b =
                            pack_e8m0x4(v_sfb[(in / rep_n_per_scale) * T::E_K + ik]);
                        constexpr int i_tile_a = im * T::E_K + ik;
                        constexpr int i_tile_b = in * T::E_K + ik;
                        constexpr int i_tile_c = im * T::E_N + in;
                        auto s_a = opus::slice(v_a,
                            opus::number<i_tile_a * a_len>{},
                            opus::number<i_tile_a * a_len + a_len>{});
                        auto s_b = opus::slice(v_b,
                            opus::number<i_tile_b * b_len>{},
                            opus::number<i_tile_b * b_len + b_len>{});
                        auto s_c = opus::slice(v_c,
                            opus::number<i_tile_c * c_len>{},
                            opus::number<i_tile_c * c_len + c_len>{});
                        s_c = MMA{}(s_a, s_b, s_c, scale_a, scale_b, 0_I, 0_I);
                        opus::set_slice(v_c, s_c,
                            opus::number<i_tile_c * c_len>{},
                            opus::number<i_tile_c * c_len + c_len>{});
                    });
                });
            });
        }
    } else {
        typename Mma::vtype_c v_mma = mma(v_a, v_b, 0, 0);
        scale_c_tile<T::E_M, T::E_N, ELEM_C, D_ACC, D_SF>(v_mma, v_sfa, v_sfb, v_c);
    }
}

#endif // __HIP_DEVICE_COMPILE__ (scale-accum helpers)

// ============================================================================
// Hand-tuned GEMM kernel with block-scale (a8w8 + scale 1x128x128)
// Kernel definition visible on both passes (host pass needs it for stub generation).
// ============================================================================

template<typename Traits, bool K1024_ONLY, bool PRELOAD_SFA_LDS = false,
         bool PRELOAD_SFB_LDS = false>
__device__ __forceinline__ void gemm_a8w8_scale_kernel_impl(opus_gemm_scale_kargs_gfx950 kargs) {
#ifdef __HIP_DEVICE_COMPILE__
#if defined(__gfx950__)
    using namespace opus;

    using T = opus::remove_cvref_t<Traits>;
    using D_A   = typename T::D_A;
    using D_B   = typename T::D_B;
    using D_C   = typename T::D_C;
    using D_ACC = typename T::D_ACC;
    using D_SF  = typename T::D_SF;

    const int grid_dim_x = opus::grid_size_x() / opus::block_size_x();
    int wgid = (opus::block_id_y() * grid_dim_x) + opus::block_id_x();
    // L2-locality rasterization (Triton-style GROUP_M grouping): process a panel
    // of GROUP_M m-tiles across all n-tiles before advancing, iterating m-tiles
    // fastest within the panel. This keeps each B[n_tile] (~1 MiB weights) hot in
    // L2 across the panel's GROUP_M reuses, recovering high-G / large-M throughput.
    constexpr int GROUP_M = 16;
    const int num_tiles_m = ceil_div(kargs.m, T::B_M);
    const int num_tiles_n = ceil_div(kargs.n, T::B_N);
    const int tiles_per_group = GROUP_M * num_tiles_n;

    // A batch swizzle here (advance batch once per panel, spreading the C drain
    // over more memory channels) was 15% faster in isolation but 1.5% slower in
    // DPA serving: it pays for those channels with the GROUP_M reuse above, and a
    // real step arrives with L2 contended. See opus_bmm.md.
    const int group_id = wgid / tiles_per_group;
    const int first_m = group_id * GROUP_M;
    const int local = wgid - group_id * tiles_per_group;
    const int m_remaining = num_tiles_m - first_m;
    const int group_rows = m_remaining < GROUP_M ? m_remaining : GROUP_M;
    int row = (first_m + (local % group_rows)) * T::B_M;
    int col = (local / group_rows) * T::B_N;

    int batch_id = opus::block_id_z();
    int wave_id = __builtin_amdgcn_readfirstlane(opus::thread_id_x() / get_warp_size());
    int lane_id = opus::thread_id_x() % get_warp_size();

    // Base offsets in 64-bit: with a batch-in-the-middle layout stride_*_batch is
    // M*K (A) / M*N (C), which overflows int32 well before the 4 GiB buffer limit.
    //
    // OOB masking for partial M tiles: bound A / sfa / C to this tile's valid row
    // window so lanes past M read 0 and their stores are dropped by num_records.
    // Any M then runs on a B_M tile (N and K stay divisible, so B and sfb need no
    // bound); the garbage accumulated for masked rows is never stored.
    //
    // Clamp to B_M rather than the full (M - row) span: stride_a = batch*K here,
    // so rows_avail*stride_a would overflow the 32-bit num_records field and wrap
    // on a large-M / high-batch shape. Each WG owns one B_M tile and the base is
    // already at `row`, so the clamp still masks the tail.
    const int rows_left  = kargs.m - row;
    const int rows_avail = rows_left < T::B_M ? rows_left : T::B_M;
    const unsigned int a_bytes =
        (unsigned int)rows_avail * (unsigned int)kargs.stride_a * sizeof(D_A);
    const unsigned int c_bytes =
        (unsigned int)rows_avail * (unsigned int)kargs.stride_c * sizeof(D_C);
    const unsigned int sfa_bytes =
        (unsigned int)ceil_div(rows_avail, T::GROUP_M) * (unsigned int)kargs.stride_sfa * sizeof(D_SF);

    auto g_a = make_gmem(reinterpret_cast<const D_A*>(kargs.ptr_a) + (size_t)batch_id*kargs.stride_a_batch + (size_t)row*kargs.stride_a, a_bytes);
    auto g_b = make_gmem(reinterpret_cast<const D_B*>(kargs.ptr_b) + (size_t)batch_id*kargs.stride_b_batch + (size_t)col*kargs.stride_b);
    auto g_c = make_gmem(reinterpret_cast<D_C*>(kargs.ptr_c) + (size_t)batch_id*kargs.stride_c_batch + (size_t)row*kargs.stride_c + col, c_bytes);

    auto g_sfa = make_gmem(reinterpret_cast<const D_SF*>(kargs.ptr_sfa) + (size_t)batch_id*kargs.stride_sfa_batch + (size_t)(row/T::GROUP_M)*kargs.stride_sfa, sfa_bytes);
    auto g_sfb = make_gmem(reinterpret_cast<const D_SF*>(kargs.ptr_sfb) + (size_t)batch_id*kargs.stride_sfb_batch + (size_t)(col/T::GROUP_N)*kargs.stride_sfb);

    int wave_id_m = wave_id % T::T_M;
    int wave_id_n = wave_id / T::T_M;

    auto u_ga = make_layout_ga<T>(lane_id, wave_id_m, wave_id_n, kargs.stride_a);
    auto u_sa = make_layout_sa<T>(lane_id, wave_id_m, wave_id_n);
    auto u_ra = make_layout_ra<T>(lane_id, wave_id_m);
    auto u_gb = make_layout_gb<T>(lane_id, wave_id_m, wave_id_n, kargs.stride_b);
    auto u_sb = make_layout_sb<T>(lane_id, wave_id_m, wave_id_n);
    auto u_rb = make_layout_rb<T>(lane_id, wave_id_n);

    auto u_sfa = make_layout_sfa<T>(lane_id, wave_id_m, kargs.stride_sfa);

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
    constexpr int ELEM_C = decltype(mma)::elem_c;

    typename decltype(mma)::vtype_a v_a[2];
    typename decltype(mma)::vtype_b v_b;
    typename decltype(mma)::vtype_c v_c[2][2];
    clear(v_c[0][0]);
    clear(v_c[0][1]);
    clear(v_c[1][0]);
    clear(v_c[1][1]);

    using vtype_sfa = vector_t<D_SF, T::E_M * (T::B_K / T::GROUP_K)>;
    using vtype_sfb = vector_t<D_SF, (T::HALF_B_N / T::GROUP_N) * (T::B_K / T::GROUP_K)>;
    vtype_sfa v_sfa[2][2];
    vtype_sfb v_sfb[2][2];

    auto a_offset = [&](int half_tile_m, int tile_k) {
        return half_tile_m * T::HALF_B_M * kargs.stride_a + tile_k * T::B_K;
    };
    auto b_offset = [&](int half_tile_n, int tile_k) {
        return half_tile_n * T::HALF_B_N * kargs.stride_b + tile_k * T::B_K;
    };
    auto sfa_offset = [&](int half_tile_m, int tile_k) {
        return half_tile_m * (T::HALF_B_M / T::GROUP_M) * kargs.stride_sfa + tile_k * (T::B_K / T::GROUP_K);
    };
    auto sfb_offset = [&](int half_tile_n, int tile_k) {
        return half_tile_n * (T::HALF_B_N / T::GROUP_N) * kargs.stride_sfb + tile_k * (T::B_K / T::GROUP_K);
    };

    // kid157: preload the whole A-scale panel into LDS once, then read per-tile
    // A-scale from LDS (ds_read/lgkmcnt) in the main loop instead of a per-tile
    // global buffer_load_b8 (vmcnt) every K iteration. The panel is a compact
    // [B_M/GROUP_M rows][K/B_K K-blocks] row-major byte tile (GROUP_M==1 and
    // B_K==GROUP_K for this traits). The LDS buffer is sized for a compile-time
    // K upper bound (SFA_K_MAX); the actual packed K-tile count is a runtime
    // value so any K<=SFA_K_MAX (and K%B_K==0) works. SFA_K_MAX=8192 keeps the
    // panel <=16 KiB, so total LDS stays 1 WG/CU.
    constexpr int SFA_K_MAX     = 8192;
    constexpr int SFA_K_TILES_MAX = PRELOAD_SFA_LDS ? (SFA_K_MAX / T::B_K) : 1;
    constexpr int SFA_ROWS      = T::B_M / T::GROUP_M;
    constexpr int SFA_LDS_BYTES =
        PRELOAD_SFA_LDS ? (SFA_ROWS * SFA_K_TILES_MAX * (int)sizeof(D_SF)) : 1;
    // 16B-aligned so the panel fill below can land ds_write_b128; a bare char
    // array is only byte-aligned as far as the language is concerned.
    __shared__ alignas(16) char smem_sfa[SFA_LDS_BYTES];
    D_SF* s_sfa_ptr = reinterpret_cast<D_SF*>(smem_sfa);
    // Runtime packed K-tile count (== loops); used as the compact LDS M-row
    // stride so the read layout reuses make_layout_sfa with stride_sfa replaced.
    const int sfa_k_tiles = PRELOAD_SFA_LDS ? (kargs.k / T::B_K) : 1;
    auto u_sfa_lds = make_layout_sfa<T>(lane_id, wave_id_m, sfa_k_tiles);
    auto sfa_lds_offset = [&](int half_tile_m, int tile_k) {
        return half_tile_m * (T::HALF_B_M / T::GROUP_M) * sfa_k_tiles +
               tile_k * (T::B_K / T::GROUP_K);
    };
    auto load_sfa = [&](int half_tile_m, int tile_k) {
        if constexpr (PRELOAD_SFA_LDS) {
            auto s = make_smem(s_sfa_ptr + sfa_lds_offset(half_tile_m, tile_k));
            return load(s, u_sfa_lds);
        } else {
            return load(g_sfa, u_sfa, sfa_offset(half_tile_m, tile_k));
        }
    };

    // kid158: same idea as PRELOAD_SFA_LDS but for the B (block) scale. SFB is
    // tiny (B_N/GROUP_N N-groups * K/B_K K-tiles, block-shared across M) so the
    // panel is a few dozen bytes; the win is purely removing the per-K-tile SFB
    // global buffer_load from the steady-state vmcnt gate. Read layout mirrors
    // sfb_offset with stride_sfb replaced by the compact per-N-group K length.
    constexpr int SFB_K_MAX       = 8192;
    constexpr int SFB_K_TILES_MAX = PRELOAD_SFB_LDS ? (SFB_K_MAX / T::B_K) : 1;
    constexpr int SFB_SPK         = T::B_K / T::GROUP_K;      // scales per K-tile
    constexpr int SFB_NG_PER_HALF = T::HALF_B_N / T::GROUP_N; // N-groups per half-n
    constexpr int SFB_ROWS        = 2 * SFB_NG_PER_HALF;      // N-groups in B_N tile
    constexpr int SFB_LDS_BYTES =
        PRELOAD_SFB_LDS ? (SFB_ROWS * SFB_K_TILES_MAX * SFB_SPK * (int)sizeof(D_SF)) : 1;
    __shared__ char smem_sfb[SFB_LDS_BYTES];
    D_SF* s_sfb_ptr = reinterpret_cast<D_SF*>(smem_sfb);
    const int sfb_k_scales = PRELOAD_SFB_LDS ? ((kargs.k / T::B_K) * SFB_SPK) : 1;
    auto sfb_lds_offset = [&](int half_tile_n, int tile_k) {
        return half_tile_n * SFB_NG_PER_HALF * sfb_k_scales + tile_k * SFB_SPK;
    };
    auto load_sfb = [&](int half_tile_n, int tile_k) {
        if constexpr (PRELOAD_SFB_LDS) {
            auto s = make_smem(s_sfb_ptr + sfb_lds_offset(half_tile_n, tile_k));
            return load<SFB_NG_PER_HALF * SFB_SPK>(s, 0);
        } else {
            return load(g_sfb, sfb_offset(half_tile_n, tile_k));
        }
    };
    // A preloaded panel is read from LDS and issues no vm ops, so it must drop out
    // of every vmcnt threshold below: over-counting retires the wait early and lets
    // the barrier release while the A/B async_loads are still landing.
    constexpr int SFA_VM = PRELOAD_SFA_LDS ? 0 : T::sfa_buffer_load_insts;
    constexpr int SFB_VM = PRELOAD_SFB_LDS ? 0 : T::sfb_buffer_load_insts;

    if constexpr (K1024_ONLY) {
        static_assert(T::B_K == 128, "K1024_ONLY expects eight 128-wide K tiles");
        if (kargs.k != 1024) return;
    }
    if constexpr (PRELOAD_SFA_LDS) {
        if (kargs.k > SFA_K_MAX || (kargs.k % T::B_K) != 0) return;
    }
    if constexpr (PRELOAD_SFB_LDS) {
        if (kargs.k > SFB_K_MAX || (kargs.k % T::B_K) != 0) return;
    }
    const int loops = K1024_ONLY ? 8 : ceil_div(kargs.k, T::B_K);
    int tic = 0, toc = 1;

    // kid158: issue the B-scale fetch before the A panel fill so the two global round
    // trips overlap -- the A fill's own vmcnt(0) retires this load too. The panel is
    // under one byte per thread, so one predicated load covers it and the value can
    // sit in a register across the A fill.
    using sfb_reg_t = decltype(load<1>(g_sfb, 0));
    sfb_reg_t sfb_val{};
    bool sfb_take = false;
    if constexpr (PRELOAD_SFB_LDS) {
        static_assert(SFB_ROWS * SFB_K_TILES_MAX * SFB_SPK <= T::BLOCK_SIZE,
                      "B-scale panel must fit one byte per thread");
        const int tid = opus::thread_id_x();
        sfb_take = tid < SFB_ROWS * sfb_k_scales;
        if (sfb_take) {
            const int ng = tid / sfb_k_scales;
            const int ks = tid - ng * sfb_k_scales;
            sfb_val = load<1>(g_sfb, ng * kargs.stride_sfb + ks);
        }
    }

    // kid157: one-shot cooperative fill of the A-scale panel into LDS, published by
    // the barrier below. Byte-at-a-time was 16 iterations per thread at K=4096, each
    // stalling on its own vmcnt(0). A chunk must not span two M rows nor land
    // unaligned, so the width has to divide both sfa_k_tiles and stride_sfa.
    if constexpr (PRELOAD_SFA_LDS) {
        auto s_sfa = make_smem(s_sfa_ptr);
        const int tid = opus::thread_id_x();
        const int sfa_total = SFA_ROWS * sfa_k_tiles;
        auto fill = [&](auto vec_c) {
            constexpr int VEC = decltype(vec_c)::value;
            for (int idx = tid * VEC; idx < sfa_total; idx += T::BLOCK_SIZE * VEC) {
                const int m  = idx / sfa_k_tiles;
                const int kt = idx - m * sfa_k_tiles;
                s_sfa.template store<VEC>(
                    load<VEC>(g_sfa, m * kargs.stride_sfa + kt), idx);
            }
        };
        const int widths = sfa_k_tiles | kargs.stride_sfa;
        if      ((widths & 15) == 0) fill(number<16>{});
        else if ((widths & 3) == 0)  fill(number<4>{});
        else                         fill(number<1>{});
    }

    // Land the B scale fetched above; its latency is already spent by now.
    if constexpr (PRELOAD_SFB_LDS) {
        if (sfb_take) {
            make_smem(s_sfb_ptr).template store<1>(sfb_val, opus::thread_id_x());
        }
    }

    // One barrier for both panels: draining after each fill in turn cost two full
    // global round trips, since B could not issue until A's barrier released. The
    // panels live in disjoint LDS, so the fills need no ordering between them.
    // s_barrier does not retire LDS traffic, hence the explicit lgkmcnt wait.
    if constexpr (PRELOAD_SFA_LDS || PRELOAD_SFB_LDS) {
        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_s_barrier();
    }

    // Prologue
    v_sfa[tic][0] = load_sfa(0, 0);
    v_sfb[tic][0] = load_sfb(0, 0);
    async_load<T::VEC_A>(g_a, s_a[tic][0].ptr, u_ga, u_sa, a_offset(0, 0));
    async_load<T::VEC_B>(g_b, s_b[tic][0].ptr, u_gb, u_sb, b_offset(0, 0));
    v_sfa[tic][1] = load_sfa(1, 0);
    v_sfb[tic][1] = load_sfb(1, 0);
    async_load<T::VEC_A>(g_a, s_a[tic][1].ptr, u_ga, u_sa, a_offset(1, 0));
    async_load<T::VEC_B>(g_b, s_b[tic][1].ptr, u_gb, u_sb, b_offset(1, 0));

    if (wave_id_n == 1) __builtin_amdgcn_s_barrier();

    s_waitcnt_vmcnt(number<T::b_buffer_load_insts + T::a_buffer_load_insts + SFA_VM + SFB_VM>{});
    __builtin_amdgcn_s_barrier();

    v_sfa[toc][0] = load_sfa(0, 1);
    v_sfb[toc][0] = load_sfb(0, 1);
    async_load<T::VEC_A>(g_a, s_a[toc][0].ptr, u_ga, u_sa, a_offset(0, 1));
    async_load<T::VEC_B>(g_b, s_b[toc][0].ptr, u_gb, u_sb, b_offset(0, 1));
    async_load<T::VEC_A>(g_a, s_a[toc][1].ptr, u_ga, u_sa, a_offset(1, 1));

    s_waitcnt_vmcnt(number<2 * T::a_buffer_load_insts + T::b_buffer_load_insts + SFA_VM + SFB_VM>{});
    __builtin_amdgcn_s_barrier();

    v_a[0] = load<T::VEC_A>(s_a[tic][0], u_ra);
    __builtin_amdgcn_s_barrier();

    // Main loop
    for(int tile = 0; tile < loops - 2; tile += 2) {
        // First tile
        v_sfb[toc][1] = load_sfb(1, tile + 1);
        v_b = load<T::VEC_B>(s_b[tic][0], u_rb);
        async_load<T::VEC_B>(g_b, s_b[toc][1].ptr, u_gb, u_sb, b_offset(1, tile + 1));
        s_waitcnt_lgkmcnt(number<T::b_ds_read_insts>{});
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_s_setprio(1);
        mma_scale_accum<T, ELEM_C>(mma, v_a[0], v_b, v_sfa[tic][0], v_sfb[tic][0], v_c[0][0]);
        sched_barrier_pairs<2, 0, 0>();
        sched_barrier_pairs<1, 2, 0>();
        sched_barrier_pairs<5, 4, 0>();
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        v_sfa[toc][1] = load_sfa(1, tile + 1);
        v_a[1] = load<T::VEC_A>(s_a[tic][1], u_ra);
        async_load<T::VEC_A>(g_a, s_a[tic][0].ptr, u_ga, u_sa, a_offset(0, tile + 2));
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_s_setprio(1);
        mma_scale_accum<T, ELEM_C>(mma, v_a[1], v_b, v_sfa[tic][1], v_sfb[tic][0], v_c[1][0]);
        sched_barrier_pairs<2, 0, 0>();
        sched_barrier_pairs<1, 2, 0>();
        sched_barrier_pairs<5, 4, 0>();
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        v_sfb[tic][0] = load_sfb(0, tile + 2);
        v_b = load<T::VEC_B>(s_b[tic][1], u_rb);
        async_load<T::VEC_B>(g_b, s_b[tic][0].ptr, u_gb, u_sb, b_offset(0, tile + 2));
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_s_setprio(1);
        mma_scale_accum<T, ELEM_C>(mma, v_a[0], v_b, v_sfa[tic][0], v_sfb[tic][1], v_c[0][1]);
        sched_barrier_pairs<2, 0, 0>();
        sched_barrier_pairs<1, 2, 0>();
        sched_barrier_pairs<5, 4, 0>();
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        v_sfa[tic][0] = load_sfa(0, tile + 2);
        v_a[0] = load<T::VEC_A>(s_a[toc][0], u_ra);
        async_load<T::VEC_A>(g_a, s_a[tic][1].ptr, u_ga, u_sa, a_offset(1, tile + 2));
        s_waitcnt_vmcnt(number<
            2 * T::a_buffer_load_insts +
            T::b_buffer_load_insts +
            2 * SFA_VM +
            SFB_VM>{});
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        __builtin_amdgcn_s_setprio(1);
        mma_scale_accum<T, ELEM_C>(mma, v_a[1], v_b, v_sfa[tic][1], v_sfb[tic][1], v_c[1][1]);
        sched_barrier_pairs<2, 0, 0>();
        sched_barrier_pairs<1, 2, 0>();
        sched_barrier_pairs<5, 4, 0>();
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // Second tile
        v_sfb[tic][1] = load_sfb(1, tile + 2);
        v_b = load<T::VEC_B>(s_b[toc][0], u_rb);
        async_load<T::VEC_B>(
            g_b, s_b[tic][1].ptr, u_gb, u_sb,
            b_offset(1, tile + 2));
        s_waitcnt_lgkmcnt(number<T::b_ds_read_insts>{});
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_s_setprio(1);
        mma_scale_accum<T, ELEM_C>(mma, v_a[0], v_b, v_sfa[toc][0], v_sfb[toc][0], v_c[0][0]);
        sched_barrier_pairs<2, 0, 0>();
        sched_barrier_pairs<1, 2, 0>();
        sched_barrier_pairs<5, 4, 0>();
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        v_sfa[tic][1] = load_sfa(1, tile + 2);
        v_a[1] = load<T::VEC_A>(s_a[toc][1], u_ra);
        async_load<T::VEC_A>(g_a, s_a[toc][0].ptr, u_ga, u_sa, a_offset(0, tile + 3));
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_s_setprio(1);
        mma_scale_accum<T, ELEM_C>(mma, v_a[1], v_b, v_sfa[toc][1], v_sfb[toc][0], v_c[1][0]);
        sched_barrier_pairs<2, 0, 0>();
        sched_barrier_pairs<1, 2, 0>();
        sched_barrier_pairs<5, 4, 0>();
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        v_sfb[toc][0] = load_sfb(0, tile + 3);
        v_b = load<T::VEC_B>(s_b[toc][1], u_rb);
        async_load<T::VEC_B>(g_b, s_b[toc][0].ptr, u_gb, u_sb, b_offset(0, tile + 3));
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_s_setprio(1);
        mma_scale_accum<T, ELEM_C>(mma, v_a[0], v_b, v_sfa[toc][0], v_sfb[toc][1], v_c[0][1]);
        sched_barrier_pairs<2, 0, 0>();
        sched_barrier_pairs<1, 2, 0>();
        sched_barrier_pairs<5, 4, 0>();
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        v_sfa[toc][0] = load_sfa(0, tile + 3);
        v_a[0] = load<T::VEC_A>(s_a[tic][0], u_ra);
        async_load<T::VEC_A>(g_a, s_a[toc][1].ptr, u_ga, u_sa, a_offset(1, tile + 3));
        s_waitcnt_vmcnt(number<2 * T::a_buffer_load_insts + T::b_buffer_load_insts + 2 * SFA_VM + SFB_VM>{});
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        __builtin_amdgcn_s_setprio(1);
        mma_scale_accum<T, ELEM_C>(mma, v_a[1], v_b, v_sfa[toc][1], v_sfb[toc][1], v_c[1][1]);
        sched_barrier_pairs<2, 0, 0>();
        sched_barrier_pairs<1, 2, 0>();
        sched_barrier_pairs<5, 4, 0>();
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);
    }

    // Epilogue
    {
        int tile = loops - 2;

        v_sfb[toc][1] = load_sfb(1, tile + 1);
        v_b = load<T::VEC_B>(s_b[tic][0], u_rb);
        async_load<T::VEC_B>(g_b, s_b[toc][1].ptr, u_gb, u_sb, b_offset(1, tile + 1));
        __builtin_amdgcn_s_barrier();

        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_s_setprio(1);
        mma_scale_accum<T, ELEM_C>(mma, v_a[0], v_b, v_sfa[tic][0], v_sfb[tic][0], v_c[0][0]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        v_sfa[toc][1] = load_sfa(1, tile + 1);
        v_a[1] = load<T::VEC_A>(s_a[tic][1], u_ra);
        __builtin_amdgcn_s_barrier();

        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_s_setprio(1);
        mma_scale_accum<T, ELEM_C>(mma, v_a[1], v_b, v_sfa[tic][1], v_sfb[tic][0], v_c[1][0]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        v_b = load<T::VEC_B>(s_b[tic][1], u_rb);
        s_waitcnt_vmcnt(number<T::b_buffer_load_insts + T::a_buffer_load_insts + SFB_VM + 2 * SFA_VM>{});
        __builtin_amdgcn_s_barrier();

        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_s_setprio(1);
        mma_scale_accum<T, ELEM_C>(mma, v_a[0], v_b, v_sfa[tic][0], v_sfb[tic][1], v_c[0][1]);
        mma_scale_accum<T, ELEM_C>(mma, v_a[1], v_b, v_sfa[tic][1], v_sfb[tic][1], v_c[1][1]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        tic ^= 1;
        toc ^= 1;
    }

    {
        v_a[0] = load<T::VEC_A>(s_a[tic][0], u_ra);
        v_b = load<T::VEC_B>(s_b[tic][0], u_rb);
        s_waitcnt_vmcnt(number<T::b_buffer_load_insts + SFB_VM + SFA_VM>{});
        __builtin_amdgcn_s_barrier();

        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_s_setprio(1);
        mma_scale_accum<T, ELEM_C>(mma, v_a[0], v_b, v_sfa[tic][0], v_sfb[tic][0], v_c[0][0]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        v_a[1] = load<T::VEC_A>(s_a[tic][1], u_ra);
        s_waitcnt_vmcnt(0_I);
        __builtin_amdgcn_s_barrier();

        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_s_setprio(1);
        mma_scale_accum<T, ELEM_C>(mma, v_a[1], v_b, v_sfa[tic][1], v_sfb[tic][0], v_c[1][0]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        v_b = load<T::VEC_B>(s_b[tic][1], u_rb);
        __builtin_amdgcn_s_barrier();

        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_s_setprio(1);
        mma_scale_accum<T, ELEM_C>(mma, v_a[0], v_b, v_sfa[tic][0], v_sfb[tic][1], v_c[0][1]);
        mma_scale_accum<T, ELEM_C>(mma, v_a[1], v_b, v_sfa[tic][1], v_sfb[tic][1], v_c[1][1]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);
    }

    if (wave_id_n == 0) __builtin_amdgcn_s_barrier();

    // Store results to global memory
    auto p_coord_c = opus::make_tuple(wave_id_m, lane_id % mma.grpn_c, wave_id_n, lane_id / mma.grpn_c);
    auto u_gc = partition_layout_c<T::VEC_C>(mma, opus::make_tuple(kargs.stride_c, 1_I), p_coord_c);

    auto c_offset = [&](int half_tile_m, int half_tile_n) {
        return half_tile_m * T::HALF_B_M * kargs.stride_c + half_tile_n * T::HALF_B_N;
    };

    store<T::VEC_C>(g_c, v_c[0][0], u_gc, c_offset(0, 0));
    store<T::VEC_C>(g_c, v_c[0][1], u_gc, c_offset(0, 1));
    store<T::VEC_C>(g_c, v_c[1][0], u_gc, c_offset(1, 0));
    store<T::VEC_C>(g_c, v_c[1][1], u_gc, c_offset(1, 1));
#else
    // Non-gfx950 device pass: empty stub. a8w8 is gfx950-only; the host
    // launcher symbol must still exist for the unconditional dispatcher
    // reference, but the body uses gfx950-only intrinsics.
#endif // __gfx950__
#endif // __HIP_DEVICE_COMPILE__
}

template<typename Traits>
__global__ __launch_bounds__(Traits::BLOCK_SIZE, 2) void gemm_a8w8_scale_kernel(opus_gemm_scale_kargs_gfx950 kargs) {
#ifdef __HIP_DEVICE_COMPILE__
#if defined(__gfx950__)
    gemm_a8w8_scale_kernel_impl<Traits, false>(kargs);
#else
    // Non-gfx950 device pass: empty stub.
#endif // __gfx950__
#endif // __HIP_DEVICE_COMPILE__
}

template<typename Traits>
__global__ __launch_bounds__(Traits::BLOCK_SIZE, 2) void gemm_a8w8_scale_k1024_kernel(opus_gemm_scale_kargs_gfx950 kargs) {
#ifdef __HIP_DEVICE_COMPILE__
#if defined(__gfx950__)
    gemm_a8w8_scale_kernel_impl<Traits, true>(kargs);
#else
    // Non-gfx950 device pass: empty stub.
#endif // __gfx950__
#endif // __HIP_DEVICE_COMPILE__
}

template<typename Traits>
__global__ __launch_bounds__(Traits::BLOCK_SIZE, 1) void gemm_a8w8_scale_k1024_lb1_kernel(opus_gemm_scale_kargs_gfx950 kargs) {
#ifdef __HIP_DEVICE_COMPILE__
#if defined(__gfx950__)
    gemm_a8w8_scale_kernel_impl<Traits, true>(kargs);
#else
    // Non-gfx950 device pass: empty stub.
#endif // __gfx950__
#endif // __HIP_DEVICE_COMPILE__
}

// EXPERIMENTAL (kid158): kid150 + both the A (per-token) and B (block) scale panels preloaded into
// LDS, so the steady-state loop reads both SFA and SFB from LDS (ds_read) and the
// per-K-tile SFA/SFB global buffer_loads are removed from the vmcnt gate entirely.
// Supports any K<=8192 (K%B_K==0); LDS panels sized for the compile-time upper
// bound, packed K-tile count resolved at runtime.
template<typename Traits>
__global__ __launch_bounds__(Traits::BLOCK_SIZE, 2)
void gemm_a8w8_scale_preload_sf_kernel(opus_gemm_scale_kargs_gfx950 kargs) {
#ifdef __HIP_DEVICE_COMPILE__
#if defined(__gfx950__)
    gemm_a8w8_scale_kernel_impl<Traits, false, true, true>(kargs);
#endif // __gfx950__
#endif // __HIP_DEVICE_COMPILE__
}

// Split-K main kernel: computes one K partition into fp32 workspace.
template<typename Traits>
__global__ __launch_bounds__(Traits::BLOCK_SIZE, 2) void gemm_a8w8_scale_splitk_kernel(opus_gemm_scale_splitk_kargs_gfx950 kargs) {
#ifdef __HIP_DEVICE_COMPILE__
#if defined(__gfx950__)
    using namespace opus;

    using T = opus::remove_cvref_t<Traits>;
    using D_A   = typename T::D_A;
    using D_B   = typename T::D_B;
    using D_C   = typename T::D_C;
    static_assert(std::is_same_v<D_C, float>, "splitK main writes fp32 workspace");
    using D_ACC = typename T::D_ACC;
    using D_SF  = typename T::D_SF;

    int wgid_full = opus::block_id_x();
    int split_id = wgid_full % kargs.split_k;
    int wgid = wgid_full / kargs.split_k;
    const int num_tiles_n = ceil_div(kargs.n, T::B_N);
    int row = (wgid / num_tiles_n) * T::B_M;
    int col = (wgid % num_tiles_n) * T::B_N;

    const int total_iters = ceil_div(kargs.k, T::B_K);
    const int iters_full = ceil_div(total_iters, kargs.split_k);
    int loops = (split_id < kargs.split_k - 1)
                    ? iters_full
                    : (total_iters - (kargs.split_k - 1) * iters_full);
    if (loops <= 0) return;
    int k_start = split_id * iters_full * T::B_K;
    int sf_start = split_id * iters_full * (T::B_K / T::GROUP_K);

    int batch_id = opus::block_id_z();
    int wave_id = __builtin_amdgcn_readfirstlane(opus::thread_id_x() / get_warp_size());
    int lane_id = opus::thread_id_x() % get_warp_size();

    // 64-bit base offsets (see the non-splitK path above): batch_id*stride_*_batch
    // overflows int32 for large-M batch-in-the-middle layouts.
    auto g_a = make_gmem(reinterpret_cast<const D_A*>(kargs.ptr_a) + (size_t)batch_id*kargs.stride_a_batch + (size_t)row*kargs.stride_a + k_start);
    auto g_b = make_gmem(reinterpret_cast<const D_B*>(kargs.ptr_b) + (size_t)batch_id*kargs.stride_b_batch + (size_t)col*kargs.stride_b + k_start);
    auto g_c = make_gmem(reinterpret_cast<D_C*>(kargs.ws_handle->ptr) + (size_t)split_id * kargs.batch * kargs.stride_ws_batch + (size_t)batch_id * kargs.stride_ws_batch + (size_t)row * kargs.stride_ws + col);

    auto g_sfa = make_gmem(reinterpret_cast<const D_SF*>(kargs.ptr_sfa) + (size_t)batch_id*kargs.stride_sfa_batch + (size_t)(row/T::GROUP_M)*kargs.stride_sfa + sf_start);
    auto g_sfb = make_gmem(reinterpret_cast<const D_SF*>(kargs.ptr_sfb) + (size_t)batch_id*kargs.stride_sfb_batch + (size_t)(col/T::GROUP_N)*kargs.stride_sfb + sf_start);

    int wave_id_m = wave_id % T::T_M;
    int wave_id_n = wave_id / T::T_M;

    auto u_ga = make_layout_ga<T>(lane_id, wave_id_m, wave_id_n, kargs.stride_a);
    auto u_sa = make_layout_sa<T>(lane_id, wave_id_m, wave_id_n);
    auto u_ra = make_layout_ra<T>(lane_id, wave_id_m);
    auto u_gb = make_layout_gb<T>(lane_id, wave_id_m, wave_id_n, kargs.stride_b);
    auto u_sb = make_layout_sb<T>(lane_id, wave_id_m, wave_id_n);
    auto u_rb = make_layout_rb<T>(lane_id, wave_id_n);

    auto u_sfa = make_layout_sfa<T>(lane_id, wave_id_m, kargs.stride_sfa);

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
    constexpr int ELEM_C = decltype(mma)::elem_c;

    typename decltype(mma)::vtype_a v_a[2];
    typename decltype(mma)::vtype_b v_b;
    typename decltype(mma)::vtype_c v_c[2][2];
    clear(v_c[0][0]);
    clear(v_c[0][1]);
    clear(v_c[1][0]);
    clear(v_c[1][1]);

    using vtype_sfa = vector_t<D_SF, T::E_M * (T::B_K / T::GROUP_K)>;
    using vtype_sfb = vector_t<D_SF, (T::HALF_B_N / T::GROUP_N) * (T::B_K / T::GROUP_K)>;
    vtype_sfa v_sfa[2][2];
    vtype_sfb v_sfb[2][2];

    auto a_offset = [&](int half_tile_m, int tile_k) {
        return half_tile_m * T::HALF_B_M * kargs.stride_a + tile_k * T::B_K;
    };
    auto b_offset = [&](int half_tile_n, int tile_k) {
        return half_tile_n * T::HALF_B_N * kargs.stride_b + tile_k * T::B_K;
    };
    auto sfa_offset = [&](int half_tile_m, int tile_k) {
        return half_tile_m * (T::HALF_B_M / T::GROUP_M) * kargs.stride_sfa + tile_k * (T::B_K / T::GROUP_K);
    };
    auto sfb_offset = [&](int half_tile_n, int tile_k) {
        return half_tile_n * (T::HALF_B_N / T::GROUP_N) * kargs.stride_sfb + tile_k * (T::B_K / T::GROUP_K);
    };

    int tic = 0, toc = 1;

    // Prologue
    v_sfa[tic][0] = load(g_sfa, u_sfa, sfa_offset(0, 0));
    v_sfb[tic][0] = load(g_sfb, sfb_offset(0, 0));
    async_load<T::VEC_A>(g_a, s_a[tic][0].ptr, u_ga, u_sa, a_offset(0, 0));
    async_load<T::VEC_B>(g_b, s_b[tic][0].ptr, u_gb, u_sb, b_offset(0, 0));
    v_sfa[tic][1] = load(g_sfa, u_sfa, sfa_offset(1, 0));
    v_sfb[tic][1] = load(g_sfb, sfb_offset(1, 0));
    async_load<T::VEC_A>(g_a, s_a[tic][1].ptr, u_ga, u_sa, a_offset(1, 0));
    async_load<T::VEC_B>(g_b, s_b[tic][1].ptr, u_gb, u_sb, b_offset(1, 0));

    if (wave_id_n == 1) __builtin_amdgcn_s_barrier();

    s_waitcnt_vmcnt(number<T::b_buffer_load_insts + T::a_buffer_load_insts + T::sfa_buffer_load_insts + T::sfb_buffer_load_insts>{});
    __builtin_amdgcn_s_barrier();

    v_sfa[toc][0] = load(g_sfa, u_sfa, sfa_offset(0, 1));
    v_sfb[toc][0] = load(g_sfb, sfb_offset(0, 1));
    async_load<T::VEC_A>(g_a, s_a[toc][0].ptr, u_ga, u_sa, a_offset(0, 1));
    async_load<T::VEC_B>(g_b, s_b[toc][0].ptr, u_gb, u_sb, b_offset(0, 1));
    async_load<T::VEC_A>(g_a, s_a[toc][1].ptr, u_ga, u_sa, a_offset(1, 1));

    s_waitcnt_vmcnt(number<2 * T::a_buffer_load_insts + T::b_buffer_load_insts + T::sfa_buffer_load_insts + T::sfb_buffer_load_insts>{});
    __builtin_amdgcn_s_barrier();

    v_a[0] = load<T::VEC_A>(s_a[tic][0], u_ra);
    __builtin_amdgcn_s_barrier();

    // Main loop
    for(int tile = 0; tile < loops - 2; tile += 2) {
        // First tile
        v_sfb[toc][1] = load(g_sfb, sfb_offset(1, tile + 1));
        v_b = load<T::VEC_B>(s_b[tic][0], u_rb);
        async_load<T::VEC_B>(g_b, s_b[toc][1].ptr, u_gb, u_sb, b_offset(1, tile + 1));
        s_waitcnt_lgkmcnt(number<T::b_ds_read_insts>{});
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_s_setprio(1);
        mma_scale_accum<T, ELEM_C>(mma, v_a[0], v_b, v_sfa[tic][0], v_sfb[tic][0], v_c[0][0]);
        sched_barrier_pairs<2, 0, 0>();
        sched_barrier_pairs<1, 2, 0>();
        sched_barrier_pairs<5, 4, 0>();
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        v_sfa[toc][1] = load(g_sfa, u_sfa, sfa_offset(1, tile + 1));
        v_a[1] = load<T::VEC_A>(s_a[tic][1], u_ra);
        async_load<T::VEC_A>(g_a, s_a[tic][0].ptr, u_ga, u_sa, a_offset(0, tile + 2));
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_s_setprio(1);
        mma_scale_accum<T, ELEM_C>(mma, v_a[1], v_b, v_sfa[tic][1], v_sfb[tic][0], v_c[1][0]);
        sched_barrier_pairs<2, 0, 0>();
        sched_barrier_pairs<1, 2, 0>();
        sched_barrier_pairs<5, 4, 0>();
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        v_sfb[tic][0] = load(g_sfb, sfb_offset(0, tile + 2));
        v_b = load<T::VEC_B>(s_b[tic][1], u_rb);
        async_load<T::VEC_B>(g_b, s_b[tic][0].ptr, u_gb, u_sb, b_offset(0, tile + 2));
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_s_setprio(1);
        mma_scale_accum<T, ELEM_C>(mma, v_a[0], v_b, v_sfa[tic][0], v_sfb[tic][1], v_c[0][1]);
        sched_barrier_pairs<2, 0, 0>();
        sched_barrier_pairs<1, 2, 0>();
        sched_barrier_pairs<5, 4, 0>();
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        v_sfa[tic][0] = load(g_sfa, u_sfa, sfa_offset(0, tile + 2));
        v_a[0] = load<T::VEC_A>(s_a[toc][0], u_ra);
        async_load<T::VEC_A>(g_a, s_a[tic][1].ptr, u_ga, u_sa, a_offset(1, tile + 2));
        s_waitcnt_vmcnt(number<2 * T::a_buffer_load_insts + T::b_buffer_load_insts + 2 * T::sfa_buffer_load_insts + T::sfb_buffer_load_insts>{});
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        __builtin_amdgcn_s_setprio(1);
        mma_scale_accum<T, ELEM_C>(mma, v_a[1], v_b, v_sfa[tic][1], v_sfb[tic][1], v_c[1][1]);
        sched_barrier_pairs<2, 0, 0>();
        sched_barrier_pairs<1, 2, 0>();
        sched_barrier_pairs<5, 4, 0>();
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // Second tile
        v_sfb[tic][1] = load(g_sfb, sfb_offset(1, tile + 2));
        v_b = load<T::VEC_B>(s_b[toc][0], u_rb);
        async_load<T::VEC_B>(g_b, s_b[tic][1].ptr, u_gb, u_sb, b_offset(1, tile + 2));
        s_waitcnt_lgkmcnt(number<T::b_ds_read_insts>{});
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_s_setprio(1);
        mma_scale_accum<T, ELEM_C>(mma, v_a[0], v_b, v_sfa[toc][0], v_sfb[toc][0], v_c[0][0]);
        sched_barrier_pairs<2, 0, 0>();
        sched_barrier_pairs<1, 2, 0>();
        sched_barrier_pairs<5, 4, 0>();
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        v_sfa[tic][1] = load(g_sfa, u_sfa, sfa_offset(1, tile + 2));
        v_a[1] = load<T::VEC_A>(s_a[toc][1], u_ra);
        async_load<T::VEC_A>(g_a, s_a[toc][0].ptr, u_ga, u_sa, a_offset(0, tile + 3));
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_s_setprio(1);
        mma_scale_accum<T, ELEM_C>(mma, v_a[1], v_b, v_sfa[toc][1], v_sfb[toc][0], v_c[1][0]);
        sched_barrier_pairs<2, 0, 0>();
        sched_barrier_pairs<1, 2, 0>();
        sched_barrier_pairs<5, 4, 0>();
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        v_sfb[toc][0] = load(g_sfb, sfb_offset(0, tile + 3));
        v_b = load<T::VEC_B>(s_b[toc][1], u_rb);
        async_load<T::VEC_B>(g_b, s_b[toc][0].ptr, u_gb, u_sb, b_offset(0, tile + 3));
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_s_setprio(1);
        mma_scale_accum<T, ELEM_C>(mma, v_a[0], v_b, v_sfa[toc][0], v_sfb[toc][1], v_c[0][1]);
        sched_barrier_pairs<2, 0, 0>();
        sched_barrier_pairs<1, 2, 0>();
        sched_barrier_pairs<5, 4, 0>();
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        v_sfa[toc][0] = load(g_sfa, u_sfa, sfa_offset(0, tile + 3));
        v_a[0] = load<T::VEC_A>(s_a[tic][0], u_ra);
        async_load<T::VEC_A>(g_a, s_a[toc][1].ptr, u_ga, u_sa, a_offset(1, tile + 3));
        s_waitcnt_vmcnt(number<2 * T::a_buffer_load_insts + T::b_buffer_load_insts + 2 * T::sfa_buffer_load_insts + T::sfb_buffer_load_insts>{});
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        __builtin_amdgcn_s_setprio(1);
        mma_scale_accum<T, ELEM_C>(mma, v_a[1], v_b, v_sfa[toc][1], v_sfb[toc][1], v_c[1][1]);
        sched_barrier_pairs<2, 0, 0>();
        sched_barrier_pairs<1, 2, 0>();
        sched_barrier_pairs<5, 4, 0>();
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);
    }

    // Epilogue
    {
        int tile = loops - 2;

        v_sfb[toc][1] = load(g_sfb, sfb_offset(1, tile + 1));
        v_b = load<T::VEC_B>(s_b[tic][0], u_rb);
        async_load<T::VEC_B>(g_b, s_b[toc][1].ptr, u_gb, u_sb, b_offset(1, tile + 1));
        __builtin_amdgcn_s_barrier();

        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_s_setprio(1);
        mma_scale_accum<T, ELEM_C>(mma, v_a[0], v_b, v_sfa[tic][0], v_sfb[tic][0], v_c[0][0]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        v_sfa[toc][1] = load(g_sfa, u_sfa, sfa_offset(1, tile + 1));
        v_a[1] = load<T::VEC_A>(s_a[tic][1], u_ra);
        __builtin_amdgcn_s_barrier();

        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_s_setprio(1);
        mma_scale_accum<T, ELEM_C>(mma, v_a[1], v_b, v_sfa[tic][1], v_sfb[tic][0], v_c[1][0]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        v_b = load<T::VEC_B>(s_b[tic][1], u_rb);
        s_waitcnt_vmcnt(number<T::b_buffer_load_insts + T::a_buffer_load_insts + T::sfb_buffer_load_insts + 2 * T::sfa_buffer_load_insts>{});
        __builtin_amdgcn_s_barrier();

        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_s_setprio(1);
        mma_scale_accum<T, ELEM_C>(mma, v_a[0], v_b, v_sfa[tic][0], v_sfb[tic][1], v_c[0][1]);
        mma_scale_accum<T, ELEM_C>(mma, v_a[1], v_b, v_sfa[tic][1], v_sfb[tic][1], v_c[1][1]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        tic ^= 1;
        toc ^= 1;
    }

    {
        v_a[0] = load<T::VEC_A>(s_a[tic][0], u_ra);
        v_b = load<T::VEC_B>(s_b[tic][0], u_rb);
        s_waitcnt_vmcnt(number<T::b_buffer_load_insts + T::sfb_buffer_load_insts + T::sfa_buffer_load_insts>{});
        __builtin_amdgcn_s_barrier();

        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_s_setprio(1);
        mma_scale_accum<T, ELEM_C>(mma, v_a[0], v_b, v_sfa[tic][0], v_sfb[tic][0], v_c[0][0]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        v_a[1] = load<T::VEC_A>(s_a[tic][1], u_ra);
        s_waitcnt_vmcnt(0_I);
        __builtin_amdgcn_s_barrier();

        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_s_setprio(1);
        mma_scale_accum<T, ELEM_C>(mma, v_a[1], v_b, v_sfa[tic][1], v_sfb[tic][0], v_c[1][0]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        v_b = load<T::VEC_B>(s_b[tic][1], u_rb);
        __builtin_amdgcn_s_barrier();

        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_s_setprio(1);
        mma_scale_accum<T, ELEM_C>(mma, v_a[0], v_b, v_sfa[tic][0], v_sfb[tic][1], v_c[0][1]);
        mma_scale_accum<T, ELEM_C>(mma, v_a[1], v_b, v_sfa[tic][1], v_sfb[tic][1], v_c[1][1]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);
    }

    if (wave_id_n == 0) __builtin_amdgcn_s_barrier();

    // Store results to global memory
    auto p_coord_c = opus::make_tuple(wave_id_m, lane_id % mma.grpn_c, wave_id_n, lane_id / mma.grpn_c);
    auto u_gc = partition_layout_c<T::VEC_C>(mma, opus::make_tuple(kargs.stride_ws, 1_I), p_coord_c);

    auto c_offset = [&](int half_tile_m, int half_tile_n) {
        return half_tile_m * T::HALF_B_M * kargs.stride_ws + half_tile_n * T::HALF_B_N;
    };

    store<T::VEC_C>(g_c, v_c[0][0], u_gc, c_offset(0, 0));
    store<T::VEC_C>(g_c, v_c[0][1], u_gc, c_offset(0, 1));
    store<T::VEC_C>(g_c, v_c[1][0], u_gc, c_offset(1, 0));
    store<T::VEC_C>(g_c, v_c[1][1], u_gc, c_offset(1, 1));
#else
    // Non-gfx950 device pass: empty stub. a8w8 is gfx950-only; the host
    // launcher symbol must still exist for the unconditional dispatcher
    // reference, but the body uses gfx950-only intrinsics.
#endif // __gfx950__
#endif // __HIP_DEVICE_COMPILE__
}
