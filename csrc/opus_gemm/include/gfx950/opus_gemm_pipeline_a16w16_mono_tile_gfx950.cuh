// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Mono-tile BF16 a16w16 pipeline.
//
// Direct port of the upstream yk_gcn mono-tile BF16 kernel template
// (bf16_gemm/gemm_a16w16_mono_tile_kernel_template.hpp) with two mechanical
// adjustments to fit the aiter codegen contract:
//
//   (1) `opus_gemm_kargs` (yk_gcn) -> `opus_gemm_mono_tile_kargs_gfx950`
//       (aiter; defined in opus_gemm_traits_a16w16_gfx950.cuh alongside
//       the other a16w16 traits/kargs structs).
//   (2) The layout-helper namespace is renamed `gemm_mono_tile` ->
//       `opus_mono_tile_gfx950` to avoid any ODR clash with a separate
//       upstream build that ships the original symbol name.
//
// The kernel body is otherwise byte-for-byte identical to the upstream
// reference. Geometry is locked (T_M=2, T_N=4, T_K=1, W_M=W_N=16, W_K=32,
// VEC=8, BLOCK_SIZE=512); tile-divisibility / smem-rep constraints are
// enforced in the traits header static_asserts and re-validated host-side
// by _validate_a16w16_mono_tile in gen_instances.py.
#pragma once

#include <opus/opus.hpp>
#include "opus_gemm_traits_a16w16_gfx950.cuh"

namespace opus_mono_tile_gfx950 {

using opus::operator""_I;

// Kernel-internal derived traits: wave layout, smem layout, instruction
// counts. Mirrors the upstream kernel_traits<UT> verbatim so the layout
// helpers below resolve their constants the same way.
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

    static constexpr int T_M = 2;
    static constexpr int T_N = 4;
    static constexpr int T_K = 1;
    static_assert(BLOCK_SIZE / opus::get_warp_size() == T_M * T_N * T_K);
    static_assert(T_K == 1);

    static constexpr int W_M = 16;
    static constexpr int W_N = 16;
    static constexpr int W_K = 32;
    static_assert(B_K % (W_K * T_K) == 0);

    static constexpr int E_M = B_M / (W_M * T_M);
    static constexpr int E_N = B_N / (W_N * T_N);
    static constexpr int E_K = B_K / (W_K * T_K);

    static_assert(VEC_A == 16 / sizeof(D_A));
    static constexpr int smem_linear_wave = opus::get_warp_size() * 16 / sizeof(D_A);
    static constexpr int smem_sub = smem_linear_wave / B_K;
    static constexpr int smem_m_rep = B_M / smem_sub;
    static constexpr int smem_n_rep = B_N / smem_sub;
    static constexpr int smem_padding = 2 * 16 / sizeof(D_A);

    static constexpr int a_buffer_load_insts = B_M * B_K / (BLOCK_SIZE * VEC_A);
    static constexpr int b_buffer_load_insts = B_N * B_K / (BLOCK_SIZE * VEC_B);
    static constexpr int a_ds_read_insts = (E_M * E_K * W_M * W_K) / (opus::get_warp_size() * VEC_A);
    static constexpr int b_ds_read_insts = (E_N * E_K * W_N * W_K) / (opus::get_warp_size() * VEC_B);
};

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
    constexpr int smem_sub_e_m = T::smem_sub / (T::W_M / T::T_N);

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

    auto lane_id_n = lane_id % T::W_N;

    return opus::make_layout(
        rb_block_shape,
        opus::unfold_x_stride(rb_block_dim, rb_block_shape, opus::tuple{opus::number<T::smem_linear_wave + T::smem_padding>{}, 1_I}),
        opus::unfold_p_coord(rb_block_dim, opus::tuple{wave_id_n / T::T_M, lane_id_n % T::T_N, wave_id_n % T::T_M, lane_id_n / T::T_N, lane_id / T::W_N}));
}

template<typename T>
inline __device__ auto make_layout_gc(int lane_id, int wave_id_m, int wave_id_n, int stride_c) {
    constexpr auto gc_block_shape = opus::make_tuple(
        opus::number<T::E_M>{},
        opus::number<T::W_M>{},
        opus::number<T::E_N / (opus::get_warp_size() / T::W_M / (T::W_N / T::VEC_C))>{},
        opus::number<opus::get_warp_size() / T::W_M / (T::W_N / T::VEC_C)>{},
        opus::number<T::T_N>{},
        opus::number<T::W_N / T::VEC_C>{},
        opus::number<T::VEC_C>{});

    constexpr auto gc_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}, opus::p_dim{}, opus::y_dim{}));

    return opus::make_layout<T::VEC_C>(
        gc_block_shape,
        opus::unfold_x_stride(gc_block_dim, gc_block_shape, opus::tuple{stride_c, 1_I}),
        opus::unfold_p_coord(gc_block_dim, opus::tuple{lane_id % T::W_M, (lane_id / T::W_M) % (T::W_N / T::VEC_C), wave_id_n, (lane_id / T::W_M) / (T::W_N / T::VEC_C)}));
}

} // namespace opus_mono_tile_gfx950

template<typename UserTraits>
__global__ __launch_bounds__(UserTraits::BLOCK_SIZE, 2)
void gemm_a16w16_mono_tile_kernel_gfx950(opus_gemm_mono_tile_kargs_gfx950 kargs) {
#if defined(__gfx950__)
    using namespace opus;
    using namespace opus_mono_tile_gfx950;
    using opus::operator""_I;
    using T = kernel_traits<opus::remove_cvref_t<UserTraits>>;
    using D_A = typename T::D_A;
    using D_B = typename T::D_B;
    using D_C = typename T::D_C;
    using D_ACC = typename T::D_ACC;

    int wgid = block_id_x();
    const int num_tiles_m = (kargs.m + T::B_M - 1) / T::B_M;
    int row = (wgid % num_tiles_m) * T::B_M;
    int col = (wgid / num_tiles_m) * T::B_N;

    int batch_id = block_id_z();
    int wave_id = __builtin_amdgcn_readfirstlane(thread_id_x() / get_warp_size());
    int lane_id = thread_id_x() % get_warp_size();

    auto g_a = make_gmem(reinterpret_cast<const D_A*>(kargs.ptr_a) + batch_id * kargs.stride_a_batch + row * kargs.stride_a, (kargs.m - row) * kargs.stride_a * sizeof(D_A));
    auto g_b = make_gmem(reinterpret_cast<const D_B*>(kargs.ptr_b) + batch_id * kargs.stride_b_batch + col * kargs.stride_b, (kargs.n - col) * kargs.stride_b * sizeof(D_B));
    auto g_c = make_gmem(reinterpret_cast<D_C*>(kargs.ptr_c) + batch_id * kargs.stride_c_batch + row * kargs.stride_c, (kargs.m - row) * kargs.stride_c * sizeof(D_C));

    int wave_id_m = wave_id / T::T_N;
    int wave_id_n = wave_id % T::T_N;

    auto u_ga = make_layout_ga<T>(lane_id, wave_id_m, wave_id_n, kargs.stride_a);
    auto u_sa = make_layout_sa<T>(lane_id, wave_id_m, wave_id_n);
    auto u_ra = make_layout_ra<T>(lane_id, wave_id_m);
    auto u_gb = make_layout_gb<T>(lane_id, wave_id_m, wave_id_n, kargs.stride_b);
    auto u_sb = make_layout_sb<T>(lane_id, wave_id_m, wave_id_n);
    auto u_rb = make_layout_rb<T>(lane_id, wave_id_n);

    constexpr int smem_a_byte = T::smem_m_rep * (T::smem_linear_wave + T::smem_padding) * sizeof(D_A);
    __shared__ char smem_a[smem_a_byte * 2];
    smem<D_A> s_a[2] = {
        make_smem(reinterpret_cast<D_A*>(smem_a)),
        make_smem(reinterpret_cast<D_A*>(smem_a + smem_a_byte))
    };
    constexpr int smem_b_byte = T::smem_n_rep * (T::smem_linear_wave + T::smem_padding) * sizeof(D_B);
    __shared__ char smem_b[smem_b_byte * 3];
    smem<D_B> sb_r0 = make_smem(reinterpret_cast<D_B*>(smem_b));
    smem<D_B> sb_r1 = make_smem(reinterpret_cast<D_B*>(smem_b + smem_b_byte));
    smem<D_B> sb_w  = make_smem(reinterpret_cast<D_B*>(smem_b + 2 * smem_b_byte));

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

    const int loops = (kargs.k + T::B_K - 1) / T::B_K;
    int tic = 0, toc = 1;

    async_load<T::VEC_A>(g_a, s_a[tic].ptr, u_ga, u_sa, k_offset(0));
    async_load<T::VEC_B>(g_b, sb_r0.ptr, u_gb, u_sb, k_offset(0));
    __builtin_amdgcn_sched_barrier(0);
    async_load<T::VEC_B>(g_b, sb_r1.ptr, u_gb, u_sb, k_offset(1));
    s_waitcnt_vmcnt(number<T::b_buffer_load_insts>{});
    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_s_barrier();

    if (wave_id_m == 1) __builtin_amdgcn_s_barrier();

    for (int tile = 0; tile < loops - 2; tile += 2) {
        async_load<T::VEC_A>(g_a, s_a[toc].ptr, u_ga, u_sa, k_offset(tile + 1));
        v_a = load<T::VEC_A>(s_a[tic], u_ra);
        v_b = load<T::VEC_B>(sb_r0, u_rb);
        async_load<T::VEC_B>(g_b, sb_w.ptr, u_gb, u_sb, k_offset(tile + 2));
        s_waitcnt_lgkmcnt(0_I);
        s_waitcnt_vmcnt(number<T::a_buffer_load_insts + T::b_buffer_load_insts>{});
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        __builtin_amdgcn_s_setprio(1);
        v_c = mma(v_a, v_b, v_c);
        __builtin_amdgcn_sched_barrier(0);
        s_waitcnt_vmcnt(number<T::b_buffer_load_insts>{});
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        async_load<T::VEC_A>(g_a, s_a[tic].ptr, u_ga, u_sa, k_offset(tile + 2));
        v_a = load<T::VEC_A>(s_a[toc], u_ra);
        v_b = load<T::VEC_B>(sb_r1, u_rb);
        async_load<T::VEC_B>(g_b, sb_r0.ptr, u_gb, u_sb, k_offset(tile + 3));
        s_waitcnt_lgkmcnt(0_I);
        s_waitcnt_vmcnt(number<T::a_buffer_load_insts + T::b_buffer_load_insts>{});
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        __builtin_amdgcn_s_setprio(1);
        v_c = mma(v_a, v_b, v_c);
        __builtin_amdgcn_sched_barrier(0);
        s_waitcnt_vmcnt(number<T::b_buffer_load_insts>{});
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        smem<D_B> tmp = sb_w;
        sb_w = sb_r1;
        sb_r1 = sb_r0;
        sb_r0 = tmp;
    }

    {
        int tile = loops - 2;

        v_a = load<T::VEC_A>(s_a[tic], u_ra);
        v_b = load<T::VEC_B>(sb_r0, u_rb);
        async_load<T::VEC_A>(g_a, s_a[toc].ptr, u_ga, u_sa, k_offset(tile + 1));
        s_waitcnt_lgkmcnt(0_I);
        s_waitcnt_vmcnt(number<T::a_buffer_load_insts>{});
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        __builtin_amdgcn_s_setprio(1);
        v_c = mma(v_a, v_b, v_c);
        __builtin_amdgcn_sched_barrier(0);
        s_waitcnt_vmcnt(0_I);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);
    }

    {
        v_a = load<T::VEC_A>(s_a[toc], u_ra);
        v_b = load<T::VEC_B>(sb_r1, u_rb);
        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        v_c = mma(v_a, v_b, v_c);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);
    }

    if (wave_id_m == 0) __builtin_amdgcn_s_barrier();

    auto u_gc = make_layout_gc<T>(lane_id, 0, wave_id_n, kargs.stride_c);
    auto v_c_f16 = cast<D_C>(v_c);
    // For every 8 D_C elements (= 4 u32), swap lane L's upper-half (last 4
    // elems) with lane (L^16)'s lower-half (first 4 elems) using
    // v_permlane16_swap_b32.
    static_assert(sizeof(D_C) * 8 % sizeof(u32_t) == 0);
    constexpr int u32_per_chunk = sizeof(D_C) * 8 / sizeof(u32_t);
    constexpr int num_chunks = sizeof(v_c_f16) / (sizeof(u32_t) * u32_per_chunk);
    auto* p_u32 = reinterpret_cast<u32_t*>(&v_c_f16);
    static_for<num_chunks>([&](auto c) {
        auto* p = p_u32 + c.value * u32_per_chunk;
        auto r0 = __builtin_amdgcn_permlane16_swap(p[0], p[2], false, true);
        auto r1 = __builtin_amdgcn_permlane16_swap(p[1], p[3], false, true);
        p[0] = r0[0]; p[2] = r0[1];
        p[1] = r1[0]; p[3] = r1[1];
    });

    store<T::VEC_C>(g_c, v_c_f16, u_gc, wave_id_m * (T::B_M / T::T_M) * kargs.stride_c + col);
#else
    // Non-gfx950 device pass compiles to an empty stub; host-side arch
    // routing in opus_gemm.cu prevents any non-gfx950 device from
    // reaching this kernel at runtime.
    (void)kargs;
#endif
}
