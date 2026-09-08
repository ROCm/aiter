// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// BF16 a16w16 flatmm pipeline: 4-wave warp-specialized (2 producer + 2 consumer).
// Ported from gcnasm/opus_fmm/flatmm_a16w16_4wave_wasp.cc.
//
// Uses async gmem->LDS loads on producer waves and ds_read + MFMA on consumer
// waves. Depth-1 pipeline for pfk==3, depth-2 for pfk>=4. B is expected in
// [N, K] row-major (pre-transposed); no shuffle at runtime.
#pragma once

#include "opus_gemm_traits_a16w16_gfx950.cuh"

// ============================================================================
// Layout functions -- device-only; skipping them on the host pass saves ~50%
// of the template-expansion compile time (see aiter skill "opus-kernel-best-practice").
// ============================================================================

#ifdef __HIP_DEVICE_COMPILE__

// Global -> producer thread: per-lane gather addressing for A/B rows.
template<typename T, int WAVES>
inline __device__ auto make_layout_gmem_group_load_flatmm(int lane_id, int wave_id, int stride) {
    constexpr int threads_k = T::LOAD_GROUP_K / T::VEC_A;
    constexpr int threads_m_per_wave = opus::get_warp_size() / threads_k;
    constexpr int interlanegroup_m = threads_m_per_wave / T::LOAD_GROUP_M_LANE;
    constexpr int repeat_m = T::slots / WAVES;

    constexpr auto ga_block_shape = opus::make_tuple(
        opus::number<interlanegroup_m>{},
        opus::number<repeat_m>{},
        opus::number<WAVES>{},
        opus::number<T::LOAD_GROUP_M_LANE>{},
        opus::number<threads_k>{},
        opus::number<T::VEC_A>{});

    constexpr auto ga_block_dim = opus::make_tuple(
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}, opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}));

    return opus::make_layout<0>(
        ga_block_shape,
        opus::unfold_x_stride(ga_block_dim, ga_block_shape, opus::tuple{stride, 1_I}),
        opus::unfold_p_coord(ga_block_dim,
                    opus::tuple{lane_id/threads_k/T::LOAD_GROUP_M_LANE, wave_id % WAVES, (lane_id / threads_k) % T::LOAD_GROUP_M_LANE, lane_id % threads_k}));
}

// LDS store target for producer: mirror of ga within one async group load slab.
template<typename T, int WAVES>
inline __device__ auto make_layout_smem_group_load_flatmm(int lane_id, int wave_id) {
    constexpr int repeat_m = T::slots / WAVES;

    constexpr auto sa_block_shape = opus::make_tuple(
        opus::number<repeat_m>{},
        opus::number<WAVES>{},
        opus::number<opus::get_warp_size()>{},
        opus::number<T::VEC_A>{});

    constexpr auto sa_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}));

    return opus::make_layout<0>(
        sa_block_shape,
        opus::unfold_x_stride(sa_block_dim, sa_block_shape, opus::tuple{T::smem_linear_wave_per_async_load + T::smem_padding, 1_I}),
        opus::unfold_p_coord(sa_block_dim, opus::tuple{wave_id % WAVES, lane_id}));
}

// LDS -> consumer registers: A operand. 9-dim layout that relies on W_M<32 for
// LOAD_GROUP_M_LANE=1 to collapse. W_M>=32 (LGML=4) path is NOT supported here.
template<typename T>
inline __device__ auto make_layout_ra_flatmm(int lane_id, int wave_id_m) {
    constexpr int threads_k = opus::get_warp_size() / T::W_M;
    constexpr int threads_m_per_wave = opus::get_warp_size() / threads_k;
    constexpr int interlanegroup_m = threads_m_per_wave / T::LOAD_GROUP_M_LANE;
    constexpr int per_block_load = T::slots * (T::smem_linear_wave_per_async_load + T::smem_padding);
    constexpr int m_block_stride = T::NUM_LOAD_GROUPS_PER_BK * per_block_load;

    constexpr auto ra_block_shape = opus::make_tuple(
        opus::number<T::COM_REP_M>{},
        opus::number<T::slots>{},
        opus::number<T::NUM_LOAD_GROUPS_PER_BK>{},
        opus::number<T::T_M>{},
        opus::number<interlanegroup_m / T::slots>{},
        opus::number<T::LOAD_GROUP_M_LANE>{},
        opus::number<2>{},
        opus::number<threads_k>{},
        opus::number<T::VEC_A>{});

    constexpr auto ra_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}),
        opus::make_tuple(opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::p_dim{},
                         opus::y_dim{}, opus::p_dim{}, opus::y_dim{}));

    auto lane_id_m = lane_id % T::W_M;

    return opus::make_layout<0>(
        ra_block_shape,
        opus::unfold_x_stride(ra_block_dim, ra_block_shape,
            opus::tuple{opus::number<m_block_stride>{},
                        opus::number<T::smem_linear_wave_per_async_load + T::smem_padding>{},
                        opus::number<per_block_load>{},
                        1_I}),
        opus::unfold_p_coord(ra_block_dim,
            opus::tuple{lane_id_m % T::slots,
                        wave_id_m,
                        lane_id_m / T::slots,
                        lane_id_m % T::LOAD_GROUP_M_LANE,
                        lane_id / T::W_M}));
}

// LDS -> consumer registers: B operand. 5-group layout.
template<typename T>
inline __device__ auto make_layout_rb_flatmm(int lane_id) {
    constexpr int grpk_b = opus::get_warp_size() / T::W_N;
    constexpr int interlanegroup_n = T::W_N / T::LOAD_GROUP_N_LANE;
    constexpr int loops_b = interlanegroup_n / T::slots;
    constexpr int tiles_per_block_n = T::LOAD_GROUP_N / T::W_N;
    constexpr int num_blocks_n = T::COM_REP_N / tiles_per_block_n;
    constexpr int per_block_load = T::slots * (T::smem_linear_wave_per_async_load + T::smem_padding);
    constexpr int n_block_stride = T::NUM_LOAD_GROUPS_PER_BK * per_block_load;
    constexpr int n_intra_stride = T::LOAD_GROUP_N_LANE * 2 * grpk_b * T::VEC_B;

    constexpr auto rb_block_shape = opus::make_tuple(
        opus::number<num_blocks_n>{},
        opus::number<T::slots>{},
        opus::number<tiles_per_block_n>{},
        opus::number<loops_b>{},
        opus::number<T::NUM_LOAD_GROUPS_PER_BK>{},
        opus::number<T::LOAD_GROUP_N_LANE>{},
        opus::number<2>{},
        opus::number<grpk_b>{},
        opus::number<T::VEC_B>{});

    constexpr auto rb_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}),
        opus::make_tuple(opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}, opus::p_dim{}, opus::y_dim{}));

    auto lane_id_n = lane_id % T::W_N;

    return opus::make_layout<0>(
        rb_block_shape,
        opus::unfold_x_stride(rb_block_dim, rb_block_shape,
            opus::tuple{opus::number<n_block_stride>{},
                        opus::number<T::smem_linear_wave_per_async_load + T::smem_padding>{},
                        opus::number<n_intra_stride>{},
                        opus::number<per_block_load>{},
                        1_I}),
        opus::unfold_p_coord(rb_block_dim,
            opus::tuple{lane_id_n % T::slots,
                        lane_id_n / T::slots,
                        lane_id_n % T::LOAD_GROUP_N_LANE,
                        lane_id / T::W_N}));
}

#endif // __HIP_DEVICE_COMPILE__

// ============================================================================
// Kernel: signature visible on both passes (host needs stub), body guarded.
// ============================================================================

template<typename Traits>
__global__ __launch_bounds__(Traits::BLOCK_SIZE, Traits::WG_PER_CU)
void gemm_a16w16_flatmm_kernel(opus_gemm_flatmm_kargs_gfx950 kargs)
{
#ifdef __HIP_DEVICE_COMPILE__
#if defined(__gfx950__)
    // gfx950-only kernel body. Non-gfx950 device passes (multi-arch wheels)
    // fall into the empty #else stub. See opus_gemm_pipeline_a16w16_gfx950.cuh for
    // the full rationale.
    using namespace opus;

    using T = opus::remove_cvref_t<Traits>;
    using D_A = typename T::D_A;
    using D_B = typename T::D_B;
    using D_C = typename T::D_C;
    using D_ACC = typename T::D_ACC;

    // Column-major walk (row fast, col slow): consecutive wgs sweep down a
    // single B column block, maximizing B slice L2 reuse. Beats super-grouping
    // on MI300 empirically.
    int wgid = (opus::block_id_y() * opus::grid_size_x()/opus::block_size_x()) + opus::block_id_x();
    const int num_tiles_m = ceil_div(kargs.m, T::B_M);
    int row = (wgid % num_tiles_m) * T::B_M;
    int col = (wgid / num_tiles_m) * T::B_N;
    int batch_id = opus::block_id_z();
    int wave_id = __builtin_amdgcn_readfirstlane(opus::thread_id_x() / get_warp_size());
    int lane_id = opus::thread_id_x() % get_warp_size();

    // Global memory pointers. B[N, K]: stride_b = K, same as A[M, K].
    auto g_a = make_gmem(reinterpret_cast<const D_A*>(kargs.ptr_a) + batch_id * kargs.stride_a_batch + row * kargs.stride_a, (kargs.m - row) * kargs.stride_a * sizeof(D_A));
    auto g_b = make_gmem(reinterpret_cast<const D_B*>(kargs.ptr_b) + batch_id * kargs.stride_b_batch + col * kargs.stride_b, (kargs.n - col) * kargs.stride_b * sizeof(D_B));
    auto g_c = make_gmem(reinterpret_cast<D_C*>(kargs.ptr_c) + batch_id * kargs.stride_c_batch + row * kargs.stride_c + col);

    // Warp-specialization: role split up front. Producer waves (2) do gmem
    // loads; consumer waves (2) do ds_read + mma. Role swaps every 256 wgs so
    // per-SIMD register layout stays balanced across CUs.
    int role = ((wave_id & 1) ^ ((wgid >> 8) & 1));

    // LDS layout: s_a[pfk][num_m_blocks][num_k_groups],
    //             s_b[pfk][num_n_blocks][num_k_groups].
    // Inside one K iter NUM_LOAD_GROUPS_PER_BK group_loads are stored
    // contiguously along K so ra/rb K-group y_dim strides cleanly across them.
    __shared__ char smem_a[T::prefetch_k_iter * T::NUM_LOAD_GROUPS_PER_BM * T::NUM_LOAD_GROUPS_PER_BK * T::smem_per_group_load_size];
    __shared__ char smem_b[T::prefetch_k_iter * T::NUM_LOAD_GROUPS_PER_BN * T::NUM_LOAD_GROUPS_PER_BK * T::smem_per_group_load_size];

    // Offset helpers. constexpr args -> compile-time constant;
    // runtime slot_k -> single SGPR. Consumer always passes k_group=0 because
    // ra/rb sweeps K-group internally via its dedicated y_dim.
    auto smem_a_at = [&](int slot_k, int m_block, int k_group) -> D_A* {
        return reinterpret_cast<D_A*>(smem_a
            + ((slot_k * T::NUM_LOAD_GROUPS_PER_BM + m_block) * T::NUM_LOAD_GROUPS_PER_BK + k_group)
              * T::smem_per_group_load_size);
    };
    auto smem_b_at = [&](int slot_k, int n_block, int k_group) -> D_B* {
        return reinterpret_cast<D_B*>(smem_b
            + ((slot_k * T::NUM_LOAD_GROUPS_PER_BN + n_block) * T::NUM_LOAD_GROUPS_PER_BK + k_group)
              * T::smem_per_group_load_size);
    };

    // gmem offsets (in D_A/D_B pixels) for (K iter, M/N block, K-group).
    auto a_offset = [&](int loop_k_idx, int group_load_idx, int k_group) {
        return group_load_idx * T::LOAD_GROUP_M * kargs.stride_a
             + (loop_k_idx * T::NUM_LOAD_GROUPS_PER_BK + k_group) * T::LOAD_GROUP_K;
    };
    auto b_offset = [&](int loop_k_idx, int group_load_idx, int k_group) {
        return group_load_idx * T::LOAD_GROUP_N * kargs.stride_b
             + (loop_k_idx * T::NUM_LOAD_GROUPS_PER_BK + k_group) * T::LOAD_GROUP_K;
    };

    // One main-loop K iter consumes B_K pixels (= NUM_LOAD_GROUPS_PER_BK
    // group_loads). Requires loops >= prefetch_k_iter; host validates.
    const int loops = ceil_div(kargs.k, T::B_K);

    constexpr int mb_a = T::a_buffer_load_insts;
    constexpr int mb_b = T::b_buffer_load_insts;
    constexpr int mb   = mb_a + mb_b;

    if (role == 0) {
        // ================= PRODUCER (2 waves) =================
        int wave_id_prod = wave_id / 2;
        auto u_ga = make_layout_gmem_group_load_flatmm<T, 2>(lane_id, wave_id_prod, kargs.stride_a);
        auto u_sa = make_layout_smem_group_load_flatmm<T, 2>(lane_id, wave_id_prod);
        auto u_gb = make_layout_gmem_group_load_flatmm<T, 2>(lane_id, wave_id_prod, kargs.stride_b);
        auto u_sb = make_layout_smem_group_load_flatmm<T, 2>(lane_id, wave_id_prod);

        // Prologue: fill pfk slots (K=0..pfk-1). Each slot adds mb insts to
        // this wave's vmcnt queue (mb already includes NUM_LOAD_GROUPS_PER_BK).
        opus::static_for<T::prefetch_k_iter>([&](auto p_c) {
            constexpr int p = decltype(p_c)::value;
            opus::static_for<T::NUM_LOAD_GROUPS_PER_BK>([&](auto kg_c) {
                constexpr int kg = decltype(kg_c)::value;
                opus::static_for<T::NUM_LOAD_GROUPS_PER_BM>([&](auto m_c) {
                    constexpr int m = decltype(m_c)::value;
                    async_load<T::VEC_A>(g_a, smem_a_at(p, m, kg), u_ga, u_sa, a_offset(p, m, kg));
                });
                opus::static_for<T::NUM_LOAD_GROUPS_PER_BN>([&](auto n_c) {
                    constexpr int n = decltype(n_c)::value;
                    async_load<T::VEC_B>(g_b, smem_b_at(p, n, kg), u_gb, u_sb, b_offset(p, n, kg));
                });
            });
        });

        // Prologue barriers R_0..R_{pfk-3}: decreasing vmcnt, leaving two K's
        // in flight for depth-2 pipeline entry. When pfk==2 this loop is empty.
        opus::static_for<T::prefetch_k_iter - 2>([&](auto i_c) {
            constexpr int p = T::prefetch_k_iter - 1 - decltype(i_c)::value;
            s_waitcnt_vmcnt(number<mb * p>{});
            __builtin_amdgcn_s_barrier();
        });

        if constexpr (T::prefetch_k_iter == 3) {
            // Depth-1 pipeline for pfk=3. Only 1 K in flight at steady state.
            // Slot collision safety: issue K=i+1 -> slot (i+1)%3 !=
            // consumer's current-read slot i%3.
            s_waitcnt_vmcnt(number<mb>{});
            __builtin_amdgcn_s_barrier();   // R_1
            for (int i = T::prefetch_k_iter - 1; i < loops - 1; i++) {
                int issue_k = i + 1;
                int slot = issue_k % T::prefetch_k_iter;
                opus::static_for<T::NUM_LOAD_GROUPS_PER_BK>([&](auto kg_c) {
                    constexpr int kg = decltype(kg_c)::value;
                    opus::static_for<T::NUM_LOAD_GROUPS_PER_BM>([&](auto m_c) {
                        constexpr int m = decltype(m_c)::value;
                        async_load<T::VEC_A>(g_a, smem_a_at(slot, m, kg), u_ga, u_sa, a_offset(issue_k, m, kg));
                    });
                    opus::static_for<T::NUM_LOAD_GROUPS_PER_BN>([&](auto n_c) {
                        constexpr int n = decltype(n_c)::value;
                        async_load<T::VEC_B>(g_b, smem_b_at(slot, n, kg), u_gb, u_sb, b_offset(issue_k, n, kg));
                    });
                });
                s_waitcnt_vmcnt(number<mb>{});
                __builtin_amdgcn_s_barrier();
            }
            s_waitcnt_vmcnt(0_I);
            __builtin_amdgcn_s_barrier();   // R_{loops-1}
        }
        else {
            // Depth-2 pipeline for pfk>=4. Main iter i issues K=i+2 into slot
            // (i+2)%pfk; wait(2*mb) retires the OLDEST pending (K=i).
            for (int i = T::prefetch_k_iter - 2; i < loops - 2; i++) {
                int issue_k = i + 2;
                int slot = issue_k % T::prefetch_k_iter;
                opus::static_for<T::NUM_LOAD_GROUPS_PER_BK>([&](auto kg_c) {
                    constexpr int kg = decltype(kg_c)::value;
                    opus::static_for<T::NUM_LOAD_GROUPS_PER_BM>([&](auto m_c) {
                        constexpr int m = decltype(m_c)::value;
                        async_load<T::VEC_A>(g_a, smem_a_at(slot, m, kg), u_ga, u_sa, a_offset(issue_k, m, kg));
                    });
                    opus::static_for<T::NUM_LOAD_GROUPS_PER_BN>([&](auto n_c) {
                        constexpr int n = decltype(n_c)::value;
                        async_load<T::VEC_B>(g_b, smem_b_at(slot, n, kg), u_gb, u_sb, b_offset(issue_k, n, kg));
                    });
                });
                s_waitcnt_vmcnt(number<2 * mb>{});
                __builtin_amdgcn_s_barrier();
            }

            // Epilogue: drain the last two in-flight K's and emit R_{loops-2},
            // R_{loops-1}.
            s_waitcnt_vmcnt(number<mb>{});
            __builtin_amdgcn_s_barrier();
            s_waitcnt_vmcnt(0_I);
            __builtin_amdgcn_s_barrier();
        }
    }
    else {
        // ================= CONSUMER (2 waves) =================
        // T_N=1, all consumer waves read the same N; wave_id/2 maps {0,1}->0
        // and {2,3}->1 for M-halves.
        int wave_id_m = wave_id / 2;
        int wave_id_n_cons = 0;
        auto u_ra = make_layout_ra_flatmm<T>(lane_id, wave_id_m);
        auto u_rb = make_layout_rb_flatmm<T>(lane_id);

        auto mma = make_tiled_mma<D_A, D_B, D_ACC>(
            seq<T::COM_REP_M, T::COM_REP_N, T::COM_REP_K>{},
            seq<T::T_M, T::T_N, T::T_K>{},
            seq<T::W_M, T::W_N, T::W_K>{},
            mfma_adaptor_swap_ab{});

        // Two named register buffers (not an array) to keep compile-time
        // indices, avoiding cndmask/scratch spill from runtime `v_a[buf]`.
        typename decltype(mma)::vtype_a v_a0, v_a1;
        typename decltype(mma)::vtype_b v_b0, v_b1;
        typename decltype(mma)::vtype_c v_c;
        clear(v_c);

        constexpr int ds_read_insts = T::a_ds_read_insts + T::b_ds_read_insts;

        // Consumer has no outstanding vm ops; barriers are data-ready signals.

        // Prologue Step 0: R_0 -> load K=0.
        __builtin_amdgcn_s_barrier();
        {
            auto sa0 = make_smem(smem_a_at(0, 0, 0));
            auto sb0 = make_smem(smem_b_at(0, 0, 0));
            v_a0 = load<T::VEC_A>(sa0, u_ra);
            v_b0 = load<T::VEC_B>(sb0, u_rb);
        }

        // Prologue Steps 1..pfk-2: R_p -> load K=p, mma K=p-1.
        // Same pattern also serves pfk=3 (emits R_1 + mma K=0).
        opus::static_for<T::prefetch_k_iter - 2>([&](auto i_c) {
            constexpr int p = decltype(i_c)::value + 1;
            constexpr int cur = (p - 1) & 1;
            constexpr int nxt = p & 1;
            __builtin_amdgcn_s_barrier();
            auto sa_p = make_smem(smem_a_at(p, 0, 0));
            auto sb_p = make_smem(smem_b_at(p, 0, 0));
            if constexpr (nxt == 0) {
                v_a0 = load<T::VEC_A>(sa_p, u_ra);
                v_b0 = load<T::VEC_B>(sb_p, u_rb);
            } else {
                v_a1 = load<T::VEC_A>(sa_p, u_ra);
                v_b1 = load<T::VEC_B>(sb_p, u_rb);
            }
            s_waitcnt_lgkmcnt(number<ds_read_insts>{});
            __builtin_amdgcn_s_setprio(1);
            if constexpr (cur == 0) v_c = mma(v_a0, v_b0, v_c);
            else                    v_c = mma(v_a1, v_b1, v_c);
            __builtin_amdgcn_s_setprio(0);
        });

        // L = (pfk - 2) & 1: compile-time buf where the live tile sits
        // entering the main loop. L=0 for even pfk, L=1 for odd pfk.
        constexpr int L = (T::prefetch_k_iter - 2) & 1;

        // Main loop: K from pfk-1 to loops-2, unrolled by 2 so each half-iter
        // uses compile-time register names via `constexpr if`. Iteration count
        // = loops - pfk, which may be odd; a tail half-iter handles that.
        int k = T::prefetch_k_iter - 1;
        for (; k + 1 < loops - 1; k += 2) {
            // half-iter A (mental buf=L): K=k into v_?(L^1), mma K=k-1 from v_?L
            __builtin_amdgcn_s_barrier();
            {
                int slot = k % T::prefetch_k_iter;
                auto sa_k = make_smem(smem_a_at(slot, 0, 0));
                auto sb_k = make_smem(smem_b_at(slot, 0, 0));
                if constexpr (L == 0) {
                    v_a1 = load<T::VEC_A>(sa_k, u_ra);
                    v_b1 = load<T::VEC_B>(sb_k, u_rb);
                } else {
                    v_a0 = load<T::VEC_A>(sa_k, u_ra);
                    v_b0 = load<T::VEC_B>(sb_k, u_rb);
                }
            }
            s_waitcnt_lgkmcnt(number<ds_read_insts>{});
            __builtin_amdgcn_s_setprio(1);
            if constexpr (L == 0) v_c = mma(v_a0, v_b0, v_c);
            else                  v_c = mma(v_a1, v_b1, v_c);
            __builtin_amdgcn_s_setprio(0);

            // half-iter B (mental buf=L^1): K=k+1 into v_?L, mma K=k from v_?(L^1)
            __builtin_amdgcn_s_barrier();
            {
                int slot = (k + 1) % T::prefetch_k_iter;
                auto sa_k = make_smem(smem_a_at(slot, 0, 0));
                auto sb_k = make_smem(smem_b_at(slot, 0, 0));
                if constexpr (L == 0) {
                    v_a0 = load<T::VEC_A>(sa_k, u_ra);
                    v_b0 = load<T::VEC_B>(sb_k, u_rb);
                } else {
                    v_a1 = load<T::VEC_A>(sa_k, u_ra);
                    v_b1 = load<T::VEC_B>(sb_k, u_rb);
                }
            }
            s_waitcnt_lgkmcnt(number<ds_read_insts>{});
            __builtin_amdgcn_s_setprio(1);
            if constexpr (L == 0) v_c = mma(v_a1, v_b1, v_c);
            else                  v_c = mma(v_a0, v_b0, v_c);
            __builtin_amdgcn_s_setprio(0);
        }

        // Tail: one extra half-iter if (loops - pfk) is odd.
        bool last_in_buf1 = (L != 0);
        if (k < loops - 1) {
            __builtin_amdgcn_s_barrier();
            {
                int slot = k % T::prefetch_k_iter;
                auto sa_k = make_smem(smem_a_at(slot, 0, 0));
                auto sb_k = make_smem(smem_b_at(slot, 0, 0));
                if constexpr (L == 0) {
                    v_a1 = load<T::VEC_A>(sa_k, u_ra);
                    v_b1 = load<T::VEC_B>(sb_k, u_rb);
                } else {
                    v_a0 = load<T::VEC_A>(sa_k, u_ra);
                    v_b0 = load<T::VEC_B>(sb_k, u_rb);
                }
            }
            s_waitcnt_lgkmcnt(number<ds_read_insts>{});
            __builtin_amdgcn_s_setprio(1);
            if constexpr (L == 0) v_c = mma(v_a0, v_b0, v_c);
            else                  v_c = mma(v_a1, v_b1, v_c);
            __builtin_amdgcn_s_setprio(0);
            last_in_buf1 = (L == 0);
            k++;
        }

        // Epilogue: load K=loops-1, mma K=loops-2 then K=loops-1.
        __builtin_amdgcn_s_barrier();
        int last_slot = (loops - 1) % T::prefetch_k_iter;
        auto sa_last = make_smem(smem_a_at(last_slot, 0, 0));
        auto sb_last = make_smem(smem_b_at(last_slot, 0, 0));
        if (last_in_buf1) {
            v_a0 = load<T::VEC_A>(sa_last, u_ra);
            v_b0 = load<T::VEC_B>(sb_last, u_rb);
            s_waitcnt_lgkmcnt(number<ds_read_insts>{});
            __builtin_amdgcn_s_setprio(1);
            v_c = mma(v_a1, v_b1, v_c);
            __builtin_amdgcn_s_setprio(0);
            s_waitcnt_lgkmcnt(0_I);
            __builtin_amdgcn_s_setprio(1);
            v_c = mma(v_a0, v_b0, v_c);
            __builtin_amdgcn_s_setprio(0);
        } else {
            v_a1 = load<T::VEC_A>(sa_last, u_ra);
            v_b1 = load<T::VEC_B>(sb_last, u_rb);
            s_waitcnt_lgkmcnt(number<ds_read_insts>{});
            __builtin_amdgcn_s_setprio(1);
            v_c = mma(v_a0, v_b0, v_c);
            __builtin_amdgcn_s_setprio(0);
            s_waitcnt_lgkmcnt(0_I);
            __builtin_amdgcn_s_setprio(1);
            v_c = mma(v_a1, v_b1, v_c);
            __builtin_amdgcn_s_setprio(0);
        }

        // Store C: partition_layout_c auto-traverses COM_REP_M x COM_REP_N.
        auto p_coord_c = opus::make_tuple(wave_id_m, lane_id % mma.grpn_c,
                                          wave_id_n_cons, lane_id / mma.grpn_c);
        auto u_gc = partition_layout_c<T::VEC_C>(mma, opus::make_tuple(kargs.stride_c, 1_I), p_coord_c);
        auto u_gc_m = partition_layout_c<T::VEC_C>(mma, opus::make_tuple(1_I, 0_I), p_coord_c);
        auto u_gc_n = partition_layout_c<T::VEC_C>(mma, opus::make_tuple(0_I, 1_I), p_coord_c);

        auto pred = [&](auto... ids) {
            return (row + u_gc_m(ids...)) < kargs.m && (col + u_gc_n(ids...)) < kargs.n;
        };

        auto v_c_out = cast<D_C>(v_c);
        store_if<T::VEC_C>(g_c, pred, v_c_out, u_gc, 0);
    }
#else
    // Non-gfx950 device pass: empty stub. See gfx950 branch above.
#endif // __gfx950__
#endif // __HIP_DEVICE_COMPILE__
}
