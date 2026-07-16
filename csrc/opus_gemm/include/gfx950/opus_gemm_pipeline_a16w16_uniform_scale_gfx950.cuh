// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// gfx950 Route B (fp8): uniform 4-wave + full-tile PGR1 + sched_group_barrier,
// with 128x128 block-scale. Forked from
// opus_gemm_pipeline_a16w16_uniform_gfx950.cuh (bf16), changing:
//   * fp8 A/B, MFMA 16x16x128 (W_K=128, VEC_A=VEC_B=16).
//   * ra/rb LDS-read layouts extended with a K-subgroup factor
//       KSUB = (W_M*W_K)/(warp*VEC_A)
//     (= 2 for fp8 16x16x128 vs 1 for bf16 16x16x32). One MFMA's A operand is
//     16(M) x W_K(K); each lane provides W_K/4 K-elements. bf16 needs one
//     b128 ds_read (VEC_A=8 -> 4*8=32=W_K); fp8 needs two (VEC_A=16 ->
//     4*2*16=128=W_K). The KSUB y-dim carries the second sub-read, exactly as
//     opus_gemm_pipeline_a8w8_scale_gfx950.cuh does with W_M*W_K/warp/VEC_A.
//   * per-K-tile scaled accumulate: v_mma = mma(a,b,0,0) fresh, then
//     scale_c_tile applies sfa[m]*sfb before adding into v_c. B_K==GROUP_K so
//     one K-tile is exactly one K-scale-block; B_N==GROUP_N so one sfb value
//     covers the whole tile-N.
//   * DIRECT store to Y (D_C in {bf16,fp32}); no split-K workspace/reduce,
//     mirroring the a8w8_scale store (store<VEC_C> casts fp32 v_c -> D_C).
//
// PGR1 (not PGR2): the bf16 uniform notes PGR2 halves occupancy at large M and
// is slower; fp8 starts from PGR1 too.
//
// PRIMARY HARDWARE-DEBUG TARGETS (if err != 0): make_layout_ra / make_layout_rb
// (the KSUB K-subgroup ordering) and make_layout_sfa (per-token M-row mapping).
// These are the only pieces re-derived vs the proven bf16 uniform + a8w8 scale.
#pragma once

#include <opus/opus.hpp>
#include "opus_gemm_traits_a16w16_gfx950.cuh"
#include "../opus_gemm_utils.cuh"  // scale_c_tile

#ifdef __HIP_DEVICE_COMPILE__

namespace opus_uniform_scale_gfx950 {

using opus::operator""_I;

constexpr int UNIFORM_MFMA_MASK    = 0x08;
constexpr int UNIFORM_VALU_MASK    = 0x02;
constexpr int UNIFORM_SALU_MASK    = 0x04;
constexpr int UNIFORM_DS_READ_MASK = 0x100;

// sched_group_barrier compute schedule: force MFMA / ds_read / valu / salu
// interleave in the em3en4 rhythm (same as bf16 uniform).
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

// K-subgroup factor: extra per-lane ds_read groups needed when one MFMA's K
// (W_K) exceeds warp/W_M * VEC_A. bf16 16x16x32 -> 1; fp8 16x16x128 -> 2.
template<typename T>
constexpr int ksub() {
    return (T::W_M * T::W_K) / (opus::get_warp_size() * T::VEC_A);
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

// A LDS-read layout, extended with the KSUB K-subgroup factor for W_K=128.
// Shape order matches the bf16 uniform ra with KSUB inserted after E_K:
//   (T_M, E_M/smem_sub_e_m, T_N, smem_sub_e_m, W_M/T_N, E_K, KSUB, warp/W_M, VEC_A)
// KSUB=1 degenerates to the bf16 layout exactly.
template<typename T>
inline __device__ auto make_layout_ra(int lane_id, int wave_id_m) {
    constexpr int smem_sub_e_m = T::smem_sub_e_m;
    constexpr int KSUB = ksub<T>();

    constexpr auto ra_block_shape = opus::make_tuple(
        opus::number<T::T_M>{},
        opus::number<T::E_M / smem_sub_e_m>{},
        opus::number<T::T_N>{},
        opus::number<smem_sub_e_m>{},
        opus::number<T::W_M / T::T_N>{},
        opus::number<T::E_K>{},
        opus::number<KSUB>{},
        opus::number<opus::get_warp_size() / T::W_M>{},
        opus::number<T::VEC_A>{});

    constexpr auto ra_block_dim = opus::make_tuple(
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::y_dim{}, opus::y_dim{}, opus::p_dim{}, opus::y_dim{}));

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

// B LDS-read layout. Uniform_scale always hits the smem_sub_e_n==1 regime
// (fp8 B_K=128 -> smem_sub=8=W_N/T_N), so only that branch is provided, with
// the KSUB factor inserted after E_K (mirrors make_layout_ra).
template<typename T>
inline __device__ auto make_layout_rb(int lane_id, int wave_id_n) {
    static_assert(T::smem_sub_e_n == 1,
                  "uniform_scale rb assumes smem_sub_e_n==1 (fp8 B_K=128)");
    constexpr int KSUB = ksub<T>();
    auto lane_id_n = lane_id % T::W_N;

    constexpr auto rb_block_shape = opus::make_tuple(
        opus::number<T::E_N>{},
        opus::number<T::T_N>{},
        opus::number<T::T_N>{},
        opus::number<T::W_N / T::T_N>{},
        opus::number<T::E_K>{},
        opus::number<KSUB>{},
        opus::number<opus::get_warp_size() / T::W_N>{},
        opus::number<T::VEC_B>{});

    constexpr auto rb_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}, opus::y_dim{}, opus::p_dim{}, opus::y_dim{}));

    return opus::make_layout(
        rb_block_shape,
        opus::unfold_x_stride(rb_block_dim, rb_block_shape, opus::tuple{opus::number<T::smem_linear_wave + T::smem_padding>{}, 1_I}),
        opus::unfold_p_coord(rb_block_dim, opus::tuple{wave_id_n, lane_id_n % T::T_N, lane_id_n / T::T_N, lane_id / T::W_N}));
}

// Per-token A scale layout. Mirrors the uniform full-tile store: wave_id_m
// selects a contiguous M-slab of B_M/T_M = E_M*W_M rows (OUTERMOST factor),
// then the E_M block (stride W_M) and the in-block lane row (lane%W_M).
//   M-row(e) = wave_id_m*(E_M*W_M) + e*W_M + (lane%W_M)
// so scale_a[e] (E_M values) lines up with v_c[e][*][*]. stride_sfa is the
// per-M-row stride of the [M, K/GROUP_K] scale tensor; the trailing
// B_K/GROUP_K (=1 here) walks K-scale-blocks with stride 1.
template<typename T>
inline __device__ auto make_layout_sfa(int lane_id, int wave_id_m, int stride_sfa) {
    constexpr auto sfa_block_shape = opus::make_tuple(
        opus::number<T::T_M>{},
        opus::number<T::E_M>{},
        opus::number<T::W_M>{},
        opus::number<T::B_K / T::GROUP_K>{});

    constexpr auto sfa_block_dim = opus::make_tuple(
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}));

    return opus::make_layout(
        sfa_block_shape,
        opus::unfold_x_stride(sfa_block_dim, sfa_block_shape, opus::tuple{stride_sfa, 1_I}),
        opus::unfold_p_coord(sfa_block_dim, opus::tuple{wave_id_m, lane_id % T::W_M}));
}

} // namespace opus_uniform_scale_gfx950

#endif // __HIP_DEVICE_COMPILE__

template<typename Traits>
__global__ __launch_bounds__(Traits::BLOCK_SIZE, Traits::WG_PER_CU) void gemm_a16w16_uniform_scale_kernel(opus_gemm_uniform_scale_kargs_gfx950 kargs) {
#ifdef __HIP_DEVICE_COMPILE__
#if defined(__gfx950__)
    using namespace opus;
    using namespace opus_uniform_scale_gfx950;

    using T = opus::remove_cvref_t<Traits>;
    using D_A   = typename T::D_A;
    using D_B   = typename T::D_B;
    using D_C   = typename T::D_C;
    using D_ACC = typename T::D_ACC;
    using D_SF  = typename T::D_SF;

    const int num_tiles_n = ceil_div_constexpr(kargs.n, T::B_N);
    int wgid = block_id_x();
    int row = (wgid / num_tiles_n) * T::B_M;
    int col = (wgid % num_tiles_n) * T::B_N;

    int batch_id = block_id_z();
    int wave_id = __builtin_amdgcn_readfirstlane(thread_id_x() / get_warp_size());
    int lane_id = thread_id_x() % get_warp_size();
    int wave_id_m = wave_id / T::T_N;
    int wave_id_n = wave_id % T::T_N;

    const int loops = ceil_div(kargs.k, T::B_K);

    auto g_a = make_gmem(reinterpret_cast<const D_A*>(kargs.ptr_a)
                         + batch_id * kargs.stride_a_batch + row * kargs.stride_a,
                         (size_t)(kargs.m - row) * kargs.stride_a * sizeof(D_A));
    auto g_b = make_gmem(reinterpret_cast<const D_B*>(kargs.ptr_b)
                         + batch_id * kargs.stride_b_batch + col * kargs.stride_b,
                         (size_t)(kargs.n - col) * kargs.stride_b * sizeof(D_B));
    auto g_c = make_gmem(reinterpret_cast<D_C*>(kargs.ptr_c)
                         + batch_id * kargs.stride_c_batch + row * kargs.stride_c + col);

    // A scale [M, K/GROUP_K] per token (GROUP_M=1). B scale [N/GROUP_N, K/GROUP_K].
    auto g_sfa = make_gmem(reinterpret_cast<const D_SF*>(kargs.ptr_sfa)
                           + batch_id * kargs.stride_sfa_batch
                           + static_cast<int>(row / T::GROUP_M) * kargs.stride_sfa);
    auto g_sfb = make_gmem(reinterpret_cast<const D_SF*>(kargs.ptr_sfb)
                           + batch_id * kargs.stride_sfb_batch
                           + static_cast<int>(col / T::GROUP_N) * kargs.stride_sfb);

    auto u_ga  = make_layout_ga<T>(lane_id, wave_id_m, wave_id_n, kargs.stride_a);
    auto u_sa  = make_layout_sa<T>(lane_id, wave_id_m, wave_id_n);
    auto u_ra  = make_layout_ra<T>(lane_id, wave_id_m);
    auto u_gb  = make_layout_gb<T>(lane_id, wave_id_m, wave_id_n, kargs.stride_b);
    auto u_sb  = make_layout_sb<T>(lane_id, wave_id_m, wave_id_n);
    auto u_rb  = make_layout_rb<T>(lane_id, wave_id_n);
    auto u_sfa = make_layout_sfa<T>(lane_id, wave_id_m, kargs.stride_sfa);

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
    constexpr int ELEM_C = decltype(mma)::elem_c;

    typename decltype(mma)::vtype_a v_a;
    typename decltype(mma)::vtype_b v_b;
    typename decltype(mma)::vtype_c v_c, v_mma;
    clear(v_c);

    using vtype_sfa = vector_t<D_SF, T::E_M * (T::B_K / T::GROUP_K)>;
    using vtype_sfb = vector_t<D_SF, (T::B_N / T::GROUP_N) * (T::B_K / T::GROUP_K)>;
    vtype_sfa v_sfa[2];
    vtype_sfb v_sfb[2];

    auto k_offset  = [&](int tile_k) { return tile_k * T::B_K; };
    auto sf_offset = [&](int tile_k) { return tile_k * (T::B_K / T::GROUP_K); };

    // Includes scale loads: sfa/sfb are ordinary buffer_loads and bump the VM
    // counter, so the per-tile vmcnt target must count them too (mirrors a8w8).
    constexpr int ld_per_tile = T::a_buffer_load_insts + T::b_buffer_load_insts
                              + T::sfa_buffer_load_insts + T::sfb_buffer_load_insts;

    // Prologue: prefetch K-tile 0 (A/B async + scales).
    async_load<T::VEC_A>(g_a, s_a[0].ptr, u_ga, u_sa, k_offset(0), opus::number<0>{}, opus::number<T::CACHECTL_A>{});
    async_load<T::VEC_B>(g_b, s_b[0].ptr, u_gb, u_sb, k_offset(0), opus::number<0>{}, opus::number<T::CACHECTL_B>{});
    v_sfa[0] = load(g_sfa, u_sfa, sf_offset(0));
    v_sfb[0] = load(g_sfb, sf_offset(0));

    int cur = 0;
    for (int tile = 0; tile < loops; ++tile) {
        int nxt = cur ^ 1;
        bool has_next = (tile + 1 < loops);
        if (has_next) {
            async_load<T::VEC_A>(g_a, s_a[nxt].ptr, u_ga, u_sa, k_offset(tile + 1), opus::number<0>{}, opus::number<T::CACHECTL_A>{});
            async_load<T::VEC_B>(g_b, s_b[nxt].ptr, u_gb, u_sb, k_offset(tile + 1), opus::number<0>{}, opus::number<T::CACHECTL_B>{});
            v_sfa[nxt] = load(g_sfa, u_sfa, sf_offset(tile + 1));
            v_sfb[nxt] = load(g_sfb, sf_offset(tile + 1));
            s_waitcnt_vmcnt(number<ld_per_tile>{});
        } else {
            s_waitcnt_vmcnt(0_I);
        }
        __builtin_amdgcn_s_barrier();

        v_a = load<T::VEC_A>(s_a[cur], u_ra);
        v_b = load<T::VEC_B>(s_b[cur], u_rb);
        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_sched_barrier(0);

        sched_uniform_compute<T>();
        __builtin_amdgcn_s_setprio(1);
        v_mma = mma(v_a, v_b, 0, 0);
        scale_c_tile<T::E_M, T::E_N, ELEM_C, D_ACC, D_SF>(v_mma, v_sfa[cur], v_sfb[cur], v_c);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        cur = nxt;
    }

    // Direct store to Y (cast fp32 v_c -> D_C). wave_id_m picks the M-slab via
    // the store base offset (mma wave-grid M is 1).
    auto p_coord_c = opus::make_tuple(0, lane_id % mma.grpn_c, wave_id_n, lane_id / mma.grpn_c);
    auto u_gc = partition_layout_c<T::VEC_C>(mma, opus::make_tuple(kargs.stride_c, 1_I), p_coord_c);
    store<T::VEC_C>(g_c, v_c, u_gc, wave_id_m * (T::B_M / T::T_M) * kargs.stride_c);
#else
    (void)kargs;
#endif // __gfx950__
#endif // __HIP_DEVICE_COMPILE__
}
