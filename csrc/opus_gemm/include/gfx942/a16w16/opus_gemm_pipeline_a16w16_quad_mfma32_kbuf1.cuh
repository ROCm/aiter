// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include "opus_gemm_traits_a16w16.cuh"
#include "splitk_reduce_gfx942.cuh"

#ifdef __HIP_DEVICE_COMPILE__

#include "opus_gemm_helpers_a16w16.cuh"
#include "opus_gemm_asm_mma32x32x8.cuh"

#endif // __HIP_DEVICE_COMPILE__

// gfx942 dedicated 256x256x32 / 4-wave / 32x32x8-MFMA pipeline.
// Each 128x128 quadrant has E_M=E_N=2 and needs four 32x32 accumulators per wave.
template<typename Traits, int COL_MAJOR_GROUP_M = 0, typename Kargs = opus_gemm_noscale_kargs>
// Exact 32KiB LDS relies on the compiler using the 1-block resource contract;
// the 2-block contract was not correct at this AGPR pressure.
__global__ __launch_bounds__(Traits::BLOCK_SIZE, 1) void gemm_a16w16_quad_mfma32_kbuf1_kernel(Kargs kargs) {
#ifdef __HIP_DEVICE_COMPILE__
#if defined(__gfx942__)
    using namespace opus;
    using namespace opus_quad_mfma32_gfx942;

    using T = opus::remove_cvref_t<Traits>;
    constexpr bool IS_SPLITK = std::is_same_v<Kargs, opus_gemm_splitk_kargs>;
    using D_A = typename T::D_A;
    using D_B = typename T::D_B;
    using D_C = typename T::D_C;
    using D_ACC = typename T::D_ACC;

    static_assert(T::BLOCK_SIZE == 256);
    static_assert(T::B_M == 256 && T::B_N == 256 && T::B_K == 32);
    static_assert(T::T_M == 2 && T::T_N == 2 && T::T_K == 1);
    static_assert(T::W_M == 32 && T::W_N == 32 && T::W_K == 8);
    static_assert(T::E_M == 2 && T::E_N == 2 && T::E_K == 4);
    static_assert(T::a_ds_read_insts == T::E_K && T::b_ds_read_insts == T::E_K);

    const int num_tiles_m = ceil_div_constexpr(kargs.m, T::B_M);
    const int num_tiles_n = ceil_div_constexpr(kargs.n, T::B_N);
    const int total_iters = ceil_div(kargs.k, T::B_K);
    int wgid;
    int split_id = 0;
    int k_start = 0;
    int loops = total_iters;
    if constexpr (IS_SPLITK) {
        const int wgid_full = opus::block_id_x();
        split_id = wgid_full % kargs.split_k;
        wgid = wgid_full / kargs.split_k;
        const int iters_full = ceil_div(total_iters, kargs.split_k);
        loops = (split_id < kargs.split_k - 1)
                    ? iters_full
                    : (total_iters - (kargs.split_k - 1) * iters_full);
        k_start = split_id * iters_full * T::B_K;
    } else {
        const int grid_dim_x = num_tiles_m * num_tiles_n;
        wgid = (opus::block_id_y() * grid_dim_x) + opus::block_id_x();
    }
    if (loops <= 0) return;
    int row;
    int col;
    if constexpr (COL_MAJOR_GROUP_M > 0) {
        const int group_span = COL_MAJOR_GROUP_M * num_tiles_n;
        const int group_id = wgid / group_span;
        const int group_m_base = group_id * COL_MAJOR_GROUP_M;
        const int within_group = wgid - group_id * group_span;
        if (group_m_base + COL_MAJOR_GROUP_M <= num_tiles_m) {
            row = (group_m_base + (within_group % COL_MAJOR_GROUP_M)) * T::B_M;
            col = (within_group / COL_MAJOR_GROUP_M) * T::B_N;
        } else {
            const int group_m_size = num_tiles_m - group_m_base;
            row = (group_m_base + (within_group % group_m_size)) * T::B_M;
            col = (within_group / group_m_size) * T::B_N;
        }
    } else {
        row = (wgid / num_tiles_n) * T::B_M;
        col = (wgid % num_tiles_n) * T::B_N;
    }

    int batch_id = opus::block_id_z();
    int wave_id = __builtin_amdgcn_readfirstlane(opus::thread_id_x() / get_warp_size());
    int lane_id = opus::thread_id_x() % get_warp_size();
    int wave_id_m = wave_id / T::T_N;
    int wave_id_n = wave_id % T::T_N;

    auto g_a = make_gmem(
        reinterpret_cast<const D_A*>(kargs.ptr_a)
            + batch_id * kargs.stride_a_batch + row * kargs.stride_a + k_start,
        ((kargs.m - row) * kargs.stride_a - k_start) * sizeof(D_A));
    auto g_b = make_gmem(
        reinterpret_cast<const D_B*>(kargs.ptr_b)
            + batch_id * kargs.stride_b_batch + col * kargs.stride_b + k_start,
        ((kargs.n - col) * kargs.stride_b - k_start) * sizeof(D_B));
    auto g_c = [&]() {
        if constexpr (IS_SPLITK) {
            return make_gmem(opus_splitk_ws_ptr<D_C>(kargs.ws_handle)
                             + (size_t)split_id * kargs.batch * kargs.stride_ws_batch
                             + (size_t)batch_id * kargs.stride_ws_batch
                             + (size_t)row * kargs.stride_ws
                             + (size_t)col);
        } else {
            return make_gmem(reinterpret_cast<D_C*>(kargs.ptr_c)
                             + batch_id * kargs.stride_c_batch
                             + row * kargs.stride_c + col);
        }
    }();

    auto u_ga = make_layout_ga<T>(lane_id, wave_id_m, wave_id_n, kargs.stride_a);
    auto u_sa = make_layout_sa<T, 0>(lane_id, wave_id_m, wave_id_n);
    auto u_ra = make_layout_ra<T, 0>(lane_id, wave_id_m);
    auto u_gb = make_layout_gb<T>(lane_id, wave_id_m, wave_id_n, kargs.stride_b);
    auto u_sb = make_layout_sb<T, 0>(lane_id, wave_id_m, wave_id_n);
    auto u_rb = make_layout_rb_quad_mfma32<T>(lane_id, wave_id_n);

    constexpr int smem_stride = T::smem_linear_wave;
    constexpr int smem_a_byte = T::smem_m_rep * smem_stride * sizeof(D_A);
    constexpr int smem_b_byte = T::smem_n_rep * smem_stride * sizeof(D_B);
    constexpr int ab_stage_byte = 2 * smem_a_byte + 2 * smem_b_byte;
    // 136 bf16 columns keeps C-stage rows 16B aligned while avoiding the
    // slower 128-column LDS drain pattern on this shape.
    constexpr int C_LDS_STRIDE = T::HALF_B_N + 8;
    constexpr int c_stage_byte = T::HALF_B_M * C_LDS_STRIDE * sizeof(D_C);
    constexpr int smem_bytes = ab_stage_byte > c_stage_byte ? ab_stage_byte : c_stage_byte;
    static_assert(smem_bytes <= 64 * 1024);

    __shared__ char smem_storage[smem_bytes];
    char* smem_stage0 = smem_storage;
    char* smem_stage1 = smem_storage;
    char* smem_a0 = smem_stage0;
    char* smem_b0 = smem_stage0 + 2 * smem_a_byte;
    char* smem_a1 = smem_stage1;
    char* smem_b1 = smem_stage1 + 2 * smem_a_byte;

    smem_x1b<D_A, smem_stride> s_a_stage0[2] = {
        smem_x1b<D_A, smem_stride>(reinterpret_cast<D_A*>(smem_a0)),
        smem_x1b<D_A, smem_stride>(reinterpret_cast<D_A*>(smem_a0 + smem_a_byte))
    };
    smem_x1b<D_B, smem_stride> s_b_stage0[2] = {
        smem_x1b<D_B, smem_stride>(reinterpret_cast<D_B*>(smem_b0)),
        smem_x1b<D_B, smem_stride>(reinterpret_cast<D_B*>(smem_b0 + smem_b_byte))
    };
    smem_x1b<D_A, smem_stride> s_a_stage1[2] = {
        smem_x1b<D_A, smem_stride>(reinterpret_cast<D_A*>(smem_a1)),
        smem_x1b<D_A, smem_stride>(reinterpret_cast<D_A*>(smem_a1 + smem_a_byte))
    };
    smem_x1b<D_B, smem_stride> s_b_stage1[2] = {
        smem_x1b<D_B, smem_stride>(reinterpret_cast<D_B*>(smem_b1)),
        smem_x1b<D_B, smem_stride>(reinterpret_cast<D_B*>(smem_b1 + smem_b_byte))
    };

    auto mma = make_tiled_mma<D_A, D_B, D_ACC>(
        seq<T::E_M, T::E_N, T::E_K>{},
        seq<T::T_M, T::T_N, T::T_K>{},
        seq<T::W_M, T::W_N, T::W_K>{},
        mfma_adaptor_swap_ab{});

    float16_acc acc_00[4] = {};
    float16_acc acc_01[4] = {};
    float16_acc acc_10[4] = {};
    float16_acc acc_11[4] = {};

    frag_b128<T::a_ds_read_insts> bank0_a[2], bank1_a[2];
    frag_b128<T::b_ds_read_insts> bank0_b[2], bank1_b[2];

    auto a_offset = [&](int half_tile_m, int tile_k) {
        return half_tile_m * T::HALF_B_M * kargs.stride_a + tile_k * T::B_K;
    };
    auto b_offset = [&](int half_tile_n, int tile_k) {
        return half_tile_n * T::HALF_B_N * kargs.stride_b + tile_k * T::B_K;
    };

    using vgpr_a_t = decltype(load<T::VEC_A>(g_a, u_ga, 0));
    using vgpr_b_t = decltype(load<T::VEC_B>(g_b, u_gb, 0));

    auto vgpr_a0 = load<T::VEC_A>(g_a, u_ga, a_offset(0, 0));
    auto vgpr_a1 = load<T::VEC_A>(g_a, u_ga, a_offset(1, 0));
    auto vgpr_b0 = load<T::VEC_B>(g_b, u_gb, b_offset(0, 0));
    auto vgpr_b1 = load<T::VEC_B>(g_b, u_gb, b_offset(1, 0));

    s_waitcnt_vmcnt(0_I);
    store<T::VEC_A>(s_a_stage0[0], vgpr_a0, u_sa);
    store<T::VEC_A>(s_a_stage0[1], vgpr_a1, u_sa);
    store<T::VEC_B>(s_b_stage0[0], vgpr_b0, u_sb);
    store<T::VEC_B>(s_b_stage0[1], vgpr_b1, u_sb);

    vgpr_a_t pf_a0{}, pf_a1{};
    vgpr_b_t pf_b0{}, pf_b1{};
    if (loops > 1) {
        pf_a0 = load<T::VEC_A>(g_a, u_ga, a_offset(0, 1));
        pf_a1 = load<T::VEC_A>(g_a, u_ga, a_offset(1, 1));
        pf_b0 = load<T::VEC_B>(g_b, u_gb, b_offset(0, 1));
        pf_b1 = load<T::VEC_B>(g_b, u_gb, b_offset(1, 1));
    }

    constexpr int N_RA = T::a_ds_read_insts;
    constexpr int N_RB = T::b_ds_read_insts;
    static_assert(N_RA == 4 && N_RA == N_RB);
    auto ra_offsets = layout_to_offsets<T::VEC_A>(u_ra);
    auto rb_offsets = layout_to_offsets<T::VEC_B>(u_rb);
    unsigned lds_a0_s0[N_RA], lds_a1_s0[N_RA], lds_b0_s0[N_RB], lds_b1_s0[N_RB];
    unsigned lds_a0_s1[N_RA], lds_a1_s1[N_RA], lds_b0_s1[N_RB], lds_b1_s1[N_RB];
    compute_lds_addrs_x1b<smem_x1b<D_A, smem_stride>, N_RA, smem_stride>(lds_a0_s0, s_a_stage0[0], ra_offsets);
    compute_lds_addrs_x1b<smem_x1b<D_A, smem_stride>, N_RA, smem_stride>(lds_a1_s0, s_a_stage0[1], ra_offsets);
    compute_lds_addrs_x1b<smem_x1b<D_B, smem_stride>, N_RB, smem_stride>(lds_b0_s0, s_b_stage0[0], rb_offsets);
    compute_lds_addrs_x1b<smem_x1b<D_B, smem_stride>, N_RB, smem_stride>(lds_b1_s0, s_b_stage0[1], rb_offsets);
    compute_lds_addrs_x1b<smem_x1b<D_A, smem_stride>, N_RA, smem_stride>(lds_a0_s1, s_a_stage1[0], ra_offsets);
    compute_lds_addrs_x1b<smem_x1b<D_A, smem_stride>, N_RA, smem_stride>(lds_a1_s1, s_a_stage1[1], ra_offsets);
    compute_lds_addrs_x1b<smem_x1b<D_B, smem_stride>, N_RB, smem_stride>(lds_b0_s1, s_b_stage1[0], rb_offsets);
    compute_lds_addrs_x1b<smem_x1b<D_B, smem_stride>, N_RB, smem_stride>(lds_b1_s1, s_b_stage1[1], rb_offsets);

    auto load_tile_from_lds = [&](auto& dst_a, auto& dst_b,
                                  const unsigned* lds_a0, const unsigned* lds_a1,
                                  const unsigned* lds_b0, const unsigned* lds_b1) {
        #pragma unroll
        for (int i = 0; i < N_RB; i++) {
            ds_read_b128_frag(dst_b[0], lds_b0, i);
            ds_read_b128_frag(dst_a[0], lds_a0, i);
        }
        #pragma unroll
        for (int i = 0; i < N_RB; i++) {
            ds_read_b128_frag(dst_b[1], lds_b1, i);
            ds_read_b128_frag(dst_a[1], lds_a1, i);
        }
    };

    auto prefetch_next_tile_from_lds = [&](auto& cur_a1, auto& cur_b1,
                                           auto& next_a, auto& next_b,
                                           const unsigned* next_lds_a0, const unsigned* next_lds_a1,
                                           const unsigned* next_lds_b0, const unsigned* next_lds_b1) {
        __builtin_amdgcn_s_barrier();
        phase_ab_prefetch_all_quadrants_2x2_ordered_nowait<T>(
            cur_a1, cur_b1, acc_11,
            next_b[0], next_lds_b0,
            next_a[0], next_lds_a0,
            next_b[1], next_lds_b1,
            next_a[1], next_lds_a1);
    };

    auto run_steady_tile = [&](auto& cur_a, auto& cur_b, auto& next_a, auto& next_b, int tile,
                               auto& s_a_next0, auto& s_a_next1,
                               auto& s_b_next0, auto& s_b_next1,
                               const unsigned* next_lds_a0, const unsigned* next_lds_a1,
                               const unsigned* next_lds_b0, const unsigned* next_lds_b1) {
        vgpr_a_t next_pf_a0{}, next_pf_a1{};
        vgpr_b_t next_pf_b0{}, next_pf_b1{};

        s_waitcnt_lgkmcnt(14_I);
        kstep_compute_2x2<T, 0>(cur_a[0], cur_b[0], acc_00);
        s_waitcnt_lgkmcnt(7_I);
        kstep_compute_2x2<T, 0>(cur_a[0], cur_b[1], acc_01);
        s_waitcnt_vmcnt(6_I);
        s_a_next0.template store_part<T::VEC_A>(pf_a0, u_sa, 0);
        next_pf_a0 = load<T::VEC_A>(g_a, u_ga, a_offset(0, tile + 2));
        s_waitcnt_lgkmcnt(6_I);
        kstep_compute_2x2<T, 0>(cur_a[1], cur_b[0], acc_10);

        s_waitcnt_lgkmcnt(12_I);
        kstep_compute_2x2<T, 1>(cur_a[0], cur_b[0], acc_00);
        s_a_next0.template store_part<T::VEC_A>(pf_a0, u_sa, 1);
        pf_a0 = next_pf_a0;
        s_waitcnt_lgkmcnt(5_I);
        kstep_compute_2x2<T, 1>(cur_a[0], cur_b[1], acc_01);
        s_waitcnt_vmcnt(6_I);
        s_b_next0.template store_part<T::VEC_B>(pf_b0, u_sb, 0);
        next_pf_b0 = load<T::VEC_B>(g_b, u_gb, b_offset(0, tile + 2));
        s_waitcnt_lgkmcnt(4_I);
        kstep_compute_2x2<T, 1>(cur_a[1], cur_b[0], acc_10);
        s_b_next0.template store_part<T::VEC_B>(pf_b0, u_sb, 1);
        pf_b0 = next_pf_b0;

        s_waitcnt_lgkmcnt(10_I);
        kstep_compute_2x2<T, 2>(cur_a[0], cur_b[0], acc_00);
        s_waitcnt_lgkmcnt(3_I);
        kstep_compute_2x2<T, 2>(cur_a[0], cur_b[1], acc_01);
        s_waitcnt_vmcnt(6_I);
        s_a_next1.template store_part<T::VEC_A>(pf_a1, u_sa, 0);
        next_pf_a1 = load<T::VEC_A>(g_a, u_ga, a_offset(1, tile + 2));
        s_waitcnt_lgkmcnt(2_I);
        kstep_compute_2x2<T, 2>(cur_a[1], cur_b[0], acc_10);
        s_waitcnt_lgkmcnt(0_I);
        s_a_next1.template store_part<T::VEC_A>(pf_a1, u_sa, 1);
        pf_a1 = next_pf_a1;

        s_waitcnt_lgkmcnt(8_I);
        kstep_compute_2x2<T, 3>(cur_a[0], cur_b[0], acc_00);
        s_waitcnt_lgkmcnt(1_I);
        kstep_compute_2x2<T, 3>(cur_a[0], cur_b[1], acc_01);
        s_waitcnt_vmcnt(6_I);
        s_b_next1.template store_part<T::VEC_B>(pf_b1, u_sb, 0);
        next_pf_b1 = load<T::VEC_B>(g_b, u_gb, b_offset(1, tile + 2));
        s_waitcnt_lgkmcnt(0_I);
        kstep_compute_2x2<T, 3>(cur_a[1], cur_b[0], acc_10);
        s_b_next1.template store_part<T::VEC_B>(pf_b1, u_sb, 1);
        pf_b1 = next_pf_b1;

        prefetch_next_tile_from_lds(
            cur_a[1], cur_b[1], next_a, next_b,
            next_lds_a0, next_lds_a1, next_lds_b0, next_lds_b1);
    };

    auto run_tail_tile = [&](auto& cur_a, auto& cur_b, auto& next_a, auto& next_b,
                             auto& s_a_next0, auto& s_a_next1,
                             auto& s_b_next0, auto& s_b_next1,
                             const unsigned* next_lds_a0, const unsigned* next_lds_a1,
                             const unsigned* next_lds_b0, const unsigned* next_lds_b1) {
        s_waitcnt_lgkmcnt(14_I);
        kstep_compute_2x2<T, 0>(cur_a[0], cur_b[0], acc_00);
        s_waitcnt_lgkmcnt(7_I);
        kstep_compute_2x2<T, 0>(cur_a[0], cur_b[1], acc_01);
        s_waitcnt_vmcnt(3_I);
        store<T::VEC_A>(s_a_next0, pf_a0, u_sa);
        s_waitcnt_lgkmcnt(6_I);
        kstep_compute_2x2<T, 0>(cur_a[1], cur_b[0], acc_10);

        s_waitcnt_lgkmcnt(12_I);
        kstep_compute_2x2<T, 1>(cur_a[0], cur_b[0], acc_00);
        s_waitcnt_lgkmcnt(5_I);
        kstep_compute_2x2<T, 1>(cur_a[0], cur_b[1], acc_01);
        s_waitcnt_vmcnt(2_I);
        store<T::VEC_B>(s_b_next0, pf_b0, u_sb);
        s_waitcnt_lgkmcnt(4_I);
        kstep_compute_2x2<T, 1>(cur_a[1], cur_b[0], acc_10);

        s_waitcnt_lgkmcnt(10_I);
        kstep_compute_2x2<T, 2>(cur_a[0], cur_b[0], acc_00);
        s_waitcnt_lgkmcnt(3_I);
        kstep_compute_2x2<T, 2>(cur_a[0], cur_b[1], acc_01);
        s_waitcnt_vmcnt(1_I);
        store<T::VEC_A>(s_a_next1, pf_a1, u_sa);
        s_waitcnt_lgkmcnt(2_I);
        kstep_compute_2x2<T, 2>(cur_a[1], cur_b[0], acc_10);
        s_waitcnt_lgkmcnt(0_I);

        s_waitcnt_lgkmcnt(8_I);
        kstep_compute_2x2<T, 3>(cur_a[0], cur_b[0], acc_00);
        s_waitcnt_lgkmcnt(1_I);
        kstep_compute_2x2<T, 3>(cur_a[0], cur_b[1], acc_01);
        s_waitcnt_vmcnt(0_I);
        store<T::VEC_B>(s_b_next1, pf_b1, u_sb);
        s_waitcnt_lgkmcnt(0_I);
        kstep_compute_2x2<T, 3>(cur_a[1], cur_b[0], acc_10);

        prefetch_next_tile_from_lds(
            cur_a[1], cur_b[1], next_a, next_b,
            next_lds_a0, next_lds_a1, next_lds_b0, next_lds_b1);
    };

    __builtin_amdgcn_s_barrier();
    load_tile_from_lds(bank0_a, bank0_b, lds_a0_s0, lds_a1_s0, lds_b0_s0, lds_b1_s0);

    for (int tile = 0; tile < loops - 2; tile += 2) {
        run_steady_tile(bank0_a, bank0_b, bank1_a, bank1_b, tile,
                        s_a_stage1[0], s_a_stage1[1], s_b_stage1[0], s_b_stage1[1],
                        lds_a0_s1, lds_a1_s1, lds_b0_s1, lds_b1_s1);
        run_steady_tile(bank1_a, bank1_b, bank0_a, bank0_b, tile + 1,
                        s_a_stage0[0], s_a_stage0[1], s_b_stage0[0], s_b_stage0[1],
                        lds_a0_s0, lds_a1_s0, lds_b0_s0, lds_b1_s0);
    }

    if (loops >= 2) {
        run_tail_tile(bank0_a, bank0_b, bank1_a, bank1_b,
                      s_a_stage1[0], s_a_stage1[1], s_b_stage1[0], s_b_stage1[1],
                      lds_a0_s1, lds_a1_s1, lds_b0_s1, lds_b1_s1);
    }

    s_waitcnt_lgkmcnt(0_I);
    phase_compute_2x2<T>(bank1_a[0], bank1_b[0], acc_00);
    phase_compute_2x2<T>(bank1_a[0], bank1_b[1], acc_01);
    phase_compute_2x2<T>(bank1_a[1], bank1_b[0], acc_10);
    phase_compute_2x2<T>(bank1_a[1], bank1_b[1], acc_11);

    auto p_coord_c = opus::make_tuple(wave_id_m, lane_id % mma.grpn_c,
                                      wave_id_n, lane_id / mma.grpn_c);
    const int stride_out = [&]() {
        if constexpr (IS_SPLITK) {
            return kargs.stride_ws;
        } else {
            return kargs.stride_c;
        }
    }();
    auto u_gc = partition_layout_c<T::VEC_C>(mma, opus::make_tuple(stride_out, 1_I), p_coord_c);
    auto u_gc_m = partition_layout_c<T::VEC_C>(mma, opus::make_tuple(1_I, 0_I), p_coord_c);
    auto u_gc_n = partition_layout_c<T::VEC_C>(mma, opus::make_tuple(0_I, 1_I), p_coord_c);

    auto c_offset = [&](int half_m, int half_n) {
        return half_m * T::HALF_B_M * stride_out + half_n * T::HALF_B_N;
    };

    auto read_acc_for_store = [](const float16_acc* acc) {
        if constexpr (std::is_same_v<D_C, __bf16>) {
            return agpr_to_bf16_vgpr_trunc<4>(acc);
        } else {
            return cast<D_C>(agpr_to_vgpr<4>(acc));
        }
    };

    auto v_c00 = read_acc_for_store(acc_00);
    auto v_c01 = read_acc_for_store(acc_01);
    auto v_c10 = read_acc_for_store(acc_10);
    auto v_c11 = read_acc_for_store(acc_11);

    auto do_store_if = [&](auto& vc, int g_c_offset, int m_base, int n_base) {
        auto pred = [&](auto... ids) {
            return (m_base + u_gc_m(ids...)) < kargs.m
                && (n_base + u_gc_n(ids...)) < kargs.n;
        };
        store_if<T::VEC_C>(g_c, pred, vc, u_gc, g_c_offset);
    };

    auto do_full_tile_store = [&]() {
        using LT_C = layout_load_traits<decltype(u_gc), T::VEC_C>;
        constexpr auto r_elem_c = LT_C::r_elem;
        constexpr index_t c_chunk = T::VEC_C * vector_traits<D_C>::size();
        constexpr int HALF_TILE_ELEMS = T::HALF_B_M * T::HALF_B_N;
        constexpr int THREAD_TILE_VEC = HALF_TILE_ELEMS / T::BLOCK_SIZE;
        static_assert(THREAD_TILE_VEC * T::BLOCK_SIZE == HALF_TILE_ELEMS);
        constexpr int MAX_STORE_VEC = 16 / sizeof(D_C);
        constexpr int STORE_VEC = THREAD_TILE_VEC < MAX_STORE_VEC ? THREAD_TILE_VEC : MAX_STORE_VEC;
        static_assert(THREAD_TILE_VEC % STORE_VEC == 0);
        constexpr int STORE_ITERS = THREAD_TILE_VEC / STORE_VEC;
        constexpr int LDS_STRIDE = C_LDS_STRIDE;

        smem<D_C> s_c = make_smem(reinterpret_cast<D_C*>(smem_storage));
        auto u_lds_c = partition_layout_c<T::VEC_C>(mma,
            opus::make_tuple(opus::number<LDS_STRIDE>{}, 1_I), p_coord_c);
        auto offsets_lds = layout_to_offsets<T::VEC_C>(u_lds_c);
        const int tid = opus::thread_id_x();
        // Keep the store offsets materialized as int; passing layout offsets
        // directly changes smem store codegen and breaks the splitK path.
        int store_offsets_lds[r_elem_c.value];
        #pragma unroll
        for (index_t i = 0; i < r_elem_c.value; i++) {
            store_offsets_lds[i] = offsets_lds[i];
        }
        int read_offsets_lds[STORE_ITERS];
        #pragma unroll
        for (int si = 0; si < STORE_ITERS; si++) {
            // Iteration-major drain keeps lanes in a wave on contiguous C columns.
            const int linear = (si * T::BLOCK_SIZE + tid) * STORE_VEC;
            const int rd_row = linear / T::HALF_B_N;
            const int rd_col = linear % T::HALF_B_N;
            read_offsets_lds[si] = rd_row * LDS_STRIDE + rd_col;
        }

        auto store_one_quadrant = [&](auto& vc, int hm, int hn) {
            #pragma unroll
            for (index_t i = 0; i < r_elem_c.value; i++) {
                vector_t<D_C, c_chunk> chunk;
                #pragma unroll
                for (index_t j = 0; j < c_chunk; j++) {
                    chunk[j] = vc[i * c_chunk + j];
                }
                s_c.template store<T::VEC_C>(chunk, store_offsets_lds[i]);
            }

            __builtin_amdgcn_s_barrier();

            static_assert((STORE_ITERS % 2) == 0);
            #pragma unroll
            for (int si = 0; si < STORE_ITERS; si += 2) {
                constexpr int LOAD_GROUP = 2;
                auto coal0 = s_c.template load<STORE_VEC>(read_offsets_lds[si + 0]);
                auto coal1 = s_c.template load<STORE_VEC>(read_offsets_lds[si + 1]);
                s_waitcnt_lgkmcnt(0_I);

                #pragma unroll
                for (int li = 0; li < LOAD_GROUP; li++) {
                    const int linear = ((si + li) * T::BLOCK_SIZE + tid) * STORE_VEC;
                    const int rd_row = linear / T::HALF_B_N;
                    const int rd_col = linear % T::HALF_B_N;
                    const int gmem_v_off = rd_row * kargs.stride_c + rd_col;
                    if (li == 0) {
                        g_c.template store<STORE_VEC>(coal0, gmem_v_off,
                            c_offset(hm, hn), opus::number<4>{});
                    } else {
                        g_c.template store<STORE_VEC>(coal1, gmem_v_off,
                            c_offset(hm, hn), opus::number<4>{});
                    }
                }
            }
        };

        store_one_quadrant(v_c00, 0, 0);
        store_one_quadrant(v_c01, 0, 1);
        store_one_quadrant(v_c10, 1, 0);
        store_one_quadrant(v_c11, 1, 1);
    };

    const bool full_tile =
        (row + T::B_M <= kargs.m) && (col + T::B_N <= kargs.n);
    if (full_tile) {
        do_full_tile_store();
    } else {
        do_store_if(v_c00, c_offset(0, 0), row, col);
        do_store_if(v_c01, c_offset(0, 1), row, col + T::HALF_B_N);
        do_store_if(v_c10, c_offset(1, 0), row + T::HALF_B_M, col);
        do_store_if(v_c11, c_offset(1, 1), row + T::HALF_B_M, col + T::HALF_B_N);
    }
#endif // __gfx942__
#endif // __HIP_DEVICE_COMPILE__
}
