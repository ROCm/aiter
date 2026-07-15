// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// gfx950 Route B: uniform 4-wave + full-tile PGR2 + sched_group_barrier.
//
// Layout/compute skeleton ported from the mono_tile pipeline
// (opus_gemm_pipeline_a16w16_mono_tile_gfx950.cuh) but re-parameterised for
// 4 waves (T_M=T_N=2, BLOCK_SIZE=256) instead of 8 (T_N=4, BLOCK=512), and
// wired to the split-K fp32 workspace path (opus_gemm_flatmm_splitk_kargs).
//
// Why full-tile (not the HALF-quadrant interleave clone): the split-barrier
// ra/rb layout couples E_K with T_N (B_K == T_N*W_K/2), which caps 4-wave at
// B_K=32. The mono_tile ra/rb layout parameterises E_K freely, so 4-wave
// B_K=64 (E_K=2) is expressible. Each wave-row computes a B_M/T_M x B_N slab
// into a single v_c; the two physical wave-rows are differentiated by the ra
// smem-read layout + a store base offset (mono_tile scheme).
//
// Store: native partition_layout_c (fp32 workspace, no permlane16 swap). The
// mono_tile make_layout_gc+permlane store is a bf16-only b128-coalescing
// optimisation matched to the post-swap register layout; for an fp32
// workspace the correctness-first store is the mma's native C partition.
#pragma once

#include <opus/opus.hpp>
#include "opus_gemm_traits_a16w16_gfx950.cuh"
#include "splitk_reduce_gfx950.cuh"

#ifdef __HIP_DEVICE_COMPILE__

namespace opus_uniform_gfx950 {

using opus::operator""_I;

constexpr int UNIFORM_MFMA_MASK    = 0x08;
constexpr int UNIFORM_VALU_MASK    = 0x02;
constexpr int UNIFORM_SALU_MASK    = 0x04;
constexpr int UNIFORM_DS_READ_MASK = 0x100;

// Kernel-internal derived traits (mirrors mono_tile kernel_traits<UT>).
template<typename UT>
struct kernel_traits {
    static constexpr int BLOCK_SIZE = UT::BLOCK_SIZE;
    static constexpr int B_M = UT::B_M;
    static constexpr int B_N = UT::B_N;
    static constexpr int B_K = UT::B_K;

    using D_A   = typename UT::D_A;
    using D_B   = typename UT::D_B;
    using D_C   = typename UT::D_C;
    using D_ACC = typename UT::D_ACC;

    static constexpr int VEC_A = UT::VEC_A;
    static constexpr int VEC_B = UT::VEC_B;
    static constexpr int VEC_C = UT::VEC_C;

    static constexpr int T_M = UT::T_M;
    static constexpr int T_N = UT::T_N;
    static constexpr int T_K = UT::T_K;
    static_assert(BLOCK_SIZE / opus::get_warp_size() == T_M * T_N * T_K);
    static_assert(T_K == 1);

    static constexpr int W_M = UT::W_M;
    static constexpr int W_N = UT::W_N;
    static constexpr int W_K = UT::W_K;
    static_assert(B_K % (W_K * T_K) == 0);

    static constexpr int E_M = UT::E_M;
    static constexpr int E_N = UT::E_N;
    static constexpr int E_K = UT::E_K;

    static constexpr int smem_linear_wave = UT::smem_linear_wave;
    static constexpr int smem_sub = UT::smem_sub;
    static constexpr int smem_m_rep = UT::smem_m_rep;
    static constexpr int smem_n_rep = UT::smem_n_rep;
    static constexpr int smem_padding = UT::smem_padding;
    static constexpr int smem_sub_e_m = UT::smem_sub_e_m;
    static constexpr int smem_sub_e_n = UT::smem_sub_e_n;

    static constexpr int a_buffer_load_insts = UT::a_buffer_load_insts;
    static constexpr int b_buffer_load_insts = UT::b_buffer_load_insts;
    static constexpr int a_ds_read_insts = UT::a_ds_read_insts;
    static constexpr int b_ds_read_insts = UT::b_ds_read_insts;

    static constexpr int CACHECTL_A = UT::CACHECTL_A;
    static constexpr int CACHECTL_B = UT::CACHECTL_B;
};

// sched_group_barrier compute schedule (pa_sparse_prefill style): force the
// compiler to interleave MFMA / ds_read / valu / salu in the em3en4 rhythm.
template<typename T, int Group = 0>
OPUS_D void sched_uniform_compute() {
    opus::static_for<T::E_N>([&](auto) {
        __builtin_amdgcn_sched_group_barrier(UNIFORM_MFMA_MASK, 1, Group);
        __builtin_amdgcn_sched_group_barrier(UNIFORM_DS_READ_MASK, 1, Group);
        opus::static_for<T::E_M>([&](auto) {
            __builtin_amdgcn_sched_group_barrier(UNIFORM_MFMA_MASK, 1, Group);
            __builtin_amdgcn_sched_group_barrier(UNIFORM_DS_READ_MASK, 1, Group);
        });
        __builtin_amdgcn_sched_group_barrier(UNIFORM_VALU_MASK, 1, Group);
    });
    __builtin_amdgcn_sched_group_barrier(UNIFORM_SALU_MASK, 1, Group);
}

template<typename T, typename Mma, typename VA, typename VB, typename VC>
OPUS_D void uniform_mma(VC& v_c, VA& v_a, VB& v_b, Mma& mma) {
    sched_uniform_compute<T>();
    __builtin_amdgcn_s_setprio(1);
    v_c = mma(v_a, v_b, v_c);
    __builtin_amdgcn_s_setprio(0);
    __builtin_amdgcn_sched_barrier(0);
}

template<typename T>
inline __device__ auto make_layout_ga(int lane_id, int wave_id_m, int wave_id_n, int stride_a) {
    constexpr int threads_k = T::B_K / T::VEC_A;
    constexpr int threads_m_per_block = T::BLOCK_SIZE / threads_k;
    constexpr int threads_m_per_wave = opus::get_warp_size() / threads_k;

    constexpr auto ga_block_shape = opus::make_tuple(
        opus::number<T::T_M>{},
        opus::number<T::B_M / threads_m_per_block>{},
        opus::number<threads_m_per_wave>{},
        opus::number<T::T_N>{},
        opus::number<threads_k>{},
        opus::number<T::VEC_A>{});

    constexpr auto ga_block_dim = opus::make_tuple(
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}, opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}));

    return opus::make_layout<T::VEC_A>(
        ga_block_shape,
        opus::unfold_x_stride(ga_block_dim, ga_block_shape, opus::tuple{stride_a, 1_I}),
        opus::unfold_p_coord(ga_block_dim, opus::tuple{wave_id_m, lane_id / threads_k, wave_id_n, lane_id % threads_k}));
}

template<typename T>
inline __device__ auto make_layout_sa(int lane_id, int wave_id_m, int wave_id_n) {
    constexpr int num_waves = T::BLOCK_SIZE / opus::get_warp_size();

    constexpr auto sa_block_shape = opus::make_tuple(
        opus::number<T::T_M>{},
        opus::number<T::smem_m_rep / num_waves>{},
        opus::number<T::T_N>{},
        opus::number<T::VEC_A>{});

    constexpr auto sa_block_dim = opus::make_tuple(
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}));

    return opus::make_layout(
        sa_block_shape,
        opus::unfold_x_stride(sa_block_dim, sa_block_shape, opus::tuple{opus::number<T::smem_linear_wave + T::smem_padding>{}, 1_I}),
        opus::unfold_p_coord(sa_block_dim, opus::tuple{wave_id_m, wave_id_n}));
}

template<typename T>
inline __device__ auto make_layout_ra(int lane_id, int wave_id_m) {
    constexpr int smem_sub_e_m = T::smem_sub_e_m;

    constexpr auto ra_block_shape = opus::make_tuple(
        opus::number<T::T_M>{},
        opus::number<T::E_M / smem_sub_e_m>{},
        opus::number<T::T_N>{},
        opus::number<smem_sub_e_m>{},
        opus::number<T::W_M / T::T_N>{},
        opus::number<T::E_K>{},
        opus::number<opus::get_warp_size() / T::W_M>{},
        opus::number<T::VEC_A>{});

    constexpr auto ra_block_dim = opus::make_tuple(
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::y_dim{}, opus::p_dim{}, opus::y_dim{}));

    auto lane_id_m = lane_id % T::W_M;

    return opus::make_layout(
        ra_block_shape,
        opus::unfold_x_stride(ra_block_dim, ra_block_shape, opus::tuple{opus::number<T::smem_linear_wave + T::smem_padding>{}, 1_I}),
        opus::unfold_p_coord(ra_block_dim, opus::tuple{wave_id_m, lane_id_m % T::T_N, lane_id_m / T::T_N, lane_id / T::W_M}));
}

template<typename T>
inline __device__ auto make_layout_gb(int lane_id, int wave_id_m, int wave_id_n, int stride_b) {
    constexpr int threads_k = T::B_K / T::VEC_B;
    constexpr int threads_n_per_block = T::BLOCK_SIZE / threads_k;
    constexpr int threads_n_per_wave = opus::get_warp_size() / threads_k;

    constexpr auto gb_block_shape = opus::make_tuple(
        opus::number<T::B_N / threads_n_per_block>{},
        opus::number<T::T_M>{},
        opus::number<threads_n_per_wave>{},
        opus::number<T::T_N>{},
        opus::number<threads_k>{},
        opus::number<T::VEC_B>{});

    constexpr auto gb_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}));

    return opus::make_layout<T::VEC_B>(
        gb_block_shape,
        opus::unfold_x_stride(gb_block_dim, gb_block_shape, opus::tuple{stride_b, 1_I}),
        opus::unfold_p_coord(gb_block_dim, opus::tuple{wave_id_m, lane_id / threads_k, wave_id_n, lane_id % threads_k}));
}

template<typename T>
inline __device__ auto make_layout_sb(int lane_id, int wave_id_m, int wave_id_n) {
    constexpr int num_waves = T::BLOCK_SIZE / opus::get_warp_size();

    constexpr auto sb_block_shape = opus::make_tuple(
        opus::number<T::smem_n_rep / num_waves>{},
        opus::number<T::T_M>{},
        opus::number<T::T_N>{},
        opus::number<T::VEC_B>{});

    constexpr auto sb_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}));

    return opus::make_layout(
        sb_block_shape,
        opus::unfold_x_stride(sb_block_dim, sb_block_shape, opus::tuple{opus::number<T::smem_linear_wave + T::smem_padding>{}, 1_I}),
        opus::unfold_p_coord(sb_block_dim, opus::tuple{wave_id_m, wave_id_n}));
}

template<typename T>
inline __device__ auto make_layout_rb(int lane_id, int wave_id_n) {
    auto lane_id_n = lane_id % T::W_N;
    if constexpr (T::smem_sub_e_n == 1) {
        // smem_sub == W_N/T_N (4-wave B_K=64): one smem row holds only
        // W_N/T_N N-values, so a wave_id_n N-slab can't share in-row slots.
        // Derived by reverse-solving store_N from the working B_K=32 case:
        //   store_N(wave_id_n, E_N_rep, lane_n) = 32*E_N_rep + 16*wave_id_n + lane_n,
        // and the B_K=64 physical layout (sb row = 4*iter+2*wm+wn,
        //   N = 32*iter + 16*wm + 2*nslot + wn) gives the bijection
        //   iter=E_N_rep, wm=wave_id_n, nslot=lane_n/T_N, wn=lane_n%T_N:
        //   physical row = 4*E_N_rep + 2*wave_id_n + (lane_n%T_N),
        //   in-row nslot = lane_n/T_N (stride B_K), K = E_K*W_K + (lane/W_N)*8 + vec.
        // Row group order = (E_N[stride 4], wave_id_n[stride 2], parity[stride 1]).
        constexpr auto rb_block_shape = opus::make_tuple(
            opus::number<T::E_N>{},
            opus::number<T::T_N>{},
            opus::number<T::T_N>{},
            opus::number<T::W_N / T::T_N>{},
            opus::number<T::E_K>{},
            opus::number<opus::get_warp_size() / T::W_N>{},
            opus::number<T::VEC_B>{});

        constexpr auto rb_block_dim = opus::make_tuple(
            opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}),
            opus::make_tuple(opus::p_dim{}, opus::y_dim{}, opus::p_dim{}, opus::y_dim{}));

        return opus::make_layout(
            rb_block_shape,
            opus::unfold_x_stride(rb_block_dim, rb_block_shape, opus::tuple{opus::number<T::smem_linear_wave + T::smem_padding>{}, 1_I}),
            opus::unfold_p_coord(rb_block_dim, opus::tuple{wave_id_n, lane_id_n % T::T_N, lane_id_n / T::T_N, lane_id / T::W_N}));
    } else {
        constexpr auto rb_block_shape = opus::make_tuple(
            opus::number<T::E_N>{},
            opus::number<T::T_N / T::T_M>{},
            opus::number<T::T_N>{},
            opus::number<T::T_M>{},
            opus::number<T::W_N / T::T_N>{},
            opus::number<T::E_K>{},
            opus::number<opus::get_warp_size() / T::W_N>{},
            opus::number<T::VEC_B>{});

        constexpr auto rb_block_dim = opus::make_tuple(
            opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}),
            opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::y_dim{}, opus::p_dim{}, opus::y_dim{}));

        return opus::make_layout(
            rb_block_shape,
            opus::unfold_x_stride(rb_block_dim, rb_block_shape, opus::tuple{opus::number<T::smem_linear_wave + T::smem_padding>{}, 1_I}),
            opus::unfold_p_coord(rb_block_dim, opus::tuple{wave_id_n / T::T_M, lane_id_n % T::T_N, wave_id_n % T::T_M, lane_id_n / T::T_N, lane_id / T::W_N}));
    }
}

} // namespace opus_uniform_gfx950

#endif // __HIP_DEVICE_COMPILE__

template<typename Traits>
__global__ __launch_bounds__(Traits::BLOCK_SIZE, 1) void gemm_a16w16_uniform_kernel(opus_gemm_flatmm_splitk_kargs_gfx950 kargs) {
#ifdef __HIP_DEVICE_COMPILE__
#if defined(__gfx950__)
    using namespace opus;
    using namespace opus_uniform_gfx950;

    using T = kernel_traits<opus::remove_cvref_t<Traits>>;
    using D_A = typename T::D_A;
    using D_B = typename T::D_B;
    using D_C = typename T::D_C;
    using D_ACC = typename T::D_ACC;

    static_assert(std::is_same_v<D_C, float>,
                  "uniform splitk main kernel writes fp32 workspace");

    // --- split-K grid: grid.x = split_k * num_tiles_m * num_tiles_n ---
    int wgid_full = block_id_x();
    int split_id = wgid_full % kargs.split_k;
    int wgid = wgid_full / kargs.split_k;

    const int total_iters = ceil_div(kargs.k, T::B_K);
    const int iters_full = ceil_div(total_iters, kargs.split_k);
    int loops = (split_id < kargs.split_k - 1)
                    ? iters_full
                    : (total_iters - (kargs.split_k - 1) * iters_full);
    if (loops <= 0) return;
    int k_start = split_id * iters_full * T::B_K;

    const int num_tiles_m = ceil_div(kargs.m, T::B_M);
    const int num_tiles_n = ceil_div_constexpr(kargs.n, T::B_N);
    int row = (wgid / num_tiles_n) * T::B_M;
    int col = (wgid % num_tiles_n) * T::B_N;

    int batch_id = block_id_z();
    int wave_id = __builtin_amdgcn_readfirstlane(thread_id_x() / get_warp_size());
    int lane_id = thread_id_x() % get_warp_size();
    int wave_id_m = wave_id / T::T_N;
    int wave_id_n = wave_id % T::T_N;

    auto g_a = make_gmem(reinterpret_cast<const D_A*>(kargs.ptr_a)
                         + batch_id * kargs.stride_a_batch + row * kargs.stride_a + k_start,
                         ((kargs.m - row) * kargs.stride_a - k_start) * sizeof(D_A));
    auto g_b = make_gmem(reinterpret_cast<const D_B*>(kargs.ptr_b)
                         + batch_id * kargs.stride_b_batch + col * kargs.stride_b + k_start,
                         ((kargs.n - col) * kargs.stride_b - k_start) * sizeof(D_B));
    // fp32 workspace: [split_k, batch, M, N] padded rows (stride_ws).
    auto g_c = make_gmem(reinterpret_cast<D_C*>(kargs.ws_handle->ptr)
                         + (size_t)split_id * kargs.batch * kargs.stride_ws_batch
                         + (size_t)batch_id * kargs.stride_ws_batch
                         + (size_t)row * kargs.stride_ws
                         + (size_t)col);

    auto u_ga = make_layout_ga<T>(lane_id, wave_id_m, wave_id_n, kargs.stride_a);
    auto u_sa = make_layout_sa<T>(lane_id, wave_id_m, wave_id_n);
    auto u_ra = make_layout_ra<T>(lane_id, wave_id_m);
    auto u_gb = make_layout_gb<T>(lane_id, wave_id_m, wave_id_n, kargs.stride_b);
    auto u_sb = make_layout_sb<T>(lane_id, wave_id_m, wave_id_n);
    auto u_rb = make_layout_rb<T>(lane_id, wave_id_n);

    // PGR1 (prefetch-1) K-loop, 2-deep ping-pong. Correct for any loops >= 1.
    //
    // NOTE: a PGR2+PLR variant (3-deep smem + register double-buffer, 1
    // s_barrier/tile, ds_read hidden behind MFMA) was implemented and measured
    // -- it is SLOWER at the large-M target (kid500 M=1024: 130us -> 165us).
    // The register/LDS blow-up (VGPR 144->212, LDS 70->105KiB) halves occupancy
    // (2 WG/CU -> 1), and for this compute-bound GEMM the occupancy loss beats
    // the fewer-barrier / ds_read-overlap gain. Reverted; PGR1 kept.
    constexpr int smem_a_byte = T::smem_m_rep * (T::smem_linear_wave + T::smem_padding) * sizeof(D_A);
    __shared__ char smem_a[smem_a_byte * 2];
    smem<D_A> s_a[2] = {
        make_smem(reinterpret_cast<D_A*>(smem_a)),
        make_smem(reinterpret_cast<D_A*>(smem_a + smem_a_byte))
    };
    constexpr int smem_b_byte = T::smem_n_rep * (T::smem_linear_wave + T::smem_padding) * sizeof(D_B);
    __shared__ char smem_b[smem_b_byte * 2];
    smem<D_B> s_b[2] = {
        make_smem(reinterpret_cast<D_B*>(smem_b)),
        make_smem(reinterpret_cast<D_B*>(smem_b + smem_b_byte))
    };

    auto mma = make_tiled_mma<D_A, D_B, D_ACC>(
        seq<T::E_M, T::E_N, T::E_K>{},
        seq<1_I, T::T_N, T::T_K>{},
        seq<T::W_M, T::W_N, T::W_K>{},
        mfma_adaptor_swap_ab{});

    typename decltype(mma)::vtype_a v_a;
    typename decltype(mma)::vtype_b v_b;
    typename decltype(mma)::vtype_c v_c;
    clear(v_c);

    auto k_offset = [&](int tile_k) { return tile_k * T::B_K; };

    constexpr int ld_per_tile = T::a_buffer_load_insts + T::b_buffer_load_insts;

    async_load<T::VEC_A>(g_a, s_a[0].ptr, u_ga, u_sa, k_offset(0), opus::number<0>{}, opus::number<T::CACHECTL_A>{});
    async_load<T::VEC_B>(g_b, s_b[0].ptr, u_gb, u_sb, k_offset(0), opus::number<0>{}, opus::number<T::CACHECTL_B>{});

    int cur = 0;
    for (int tile = 0; tile < loops; ++tile) {
        int nxt = cur ^ 1;
        bool has_next = (tile + 1 < loops);
        if (has_next) {
            async_load<T::VEC_A>(g_a, s_a[nxt].ptr, u_ga, u_sa, k_offset(tile + 1), opus::number<0>{}, opus::number<T::CACHECTL_A>{});
            async_load<T::VEC_B>(g_b, s_b[nxt].ptr, u_gb, u_sb, k_offset(tile + 1), opus::number<0>{}, opus::number<T::CACHECTL_B>{});
            s_waitcnt_vmcnt(number<ld_per_tile>{});
        } else {
            s_waitcnt_vmcnt(0_I);
        }
        __builtin_amdgcn_s_barrier();

        v_a = load<T::VEC_A>(s_a[cur], u_ra);
        v_b = load<T::VEC_B>(s_b[cur], u_rb);
        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_sched_barrier(0);

        uniform_mma<T>(v_c, v_a, v_b, mma);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        cur = nxt;
    }

    // fp32 workspace store: native C partition (transposed by swap_ab). The
    // mma's wave-grid M is 1 (seq<1,T_N,T_K>), so wave_id_m selects the M
    // slab via the store base offset, not via p_coord.
    auto p_coord_c = opus::make_tuple(0, lane_id % mma.grpn_c, wave_id_n, lane_id / mma.grpn_c);
    auto u_gc = partition_layout_c<T::VEC_C>(mma, opus::make_tuple(kargs.stride_ws, 1_I), p_coord_c);
    store<T::VEC_C>(g_c, v_c, u_gc, wave_id_m * (T::B_M / T::T_M) * kargs.stride_ws);
#else
    // Non-gfx950 device pass: empty stub. See gfx950 branch above.
    (void)kargs;
#endif // __gfx950__
#endif // __HIP_DEVICE_COMPILE__
}
