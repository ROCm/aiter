// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include <type_traits>

#include "opus_gemm_traits_a8w8_blockscale_bpreshuffle.cuh"

// ============================================================================
// Layout functions for A/B matrix global/shared/register data movement
// Guarded: these are __device__ functions only needed on the device pass.
// ============================================================================

#ifdef __HIP_DEVICE_COMPILE__

OPUS_D int lds_xor128_swizzle_offset(int elem_offset) {
    constexpr int segment_elem = 128;
    constexpr int swizzle_elem = 16;
    static_assert(segment_elem % swizzle_elem == 0);
    const unsigned offset = static_cast<unsigned>(elem_offset);
    const unsigned segment = offset / segment_elem;
    const unsigned col = offset - segment * segment_elem;
    const unsigned swizzle_segment = segment ^ ((segment >> 3) << 2);
    const unsigned shift =
        (swizzle_segment & static_cast<unsigned>((segment_elem / swizzle_elem) - 1)) * swizzle_elem;
    return static_cast<int>(segment * segment_elem + (col ^ shift));
}

template<typename Layout>
inline __device__ void apply_lds_xor128_swizzle(Layout& u) {
    using layout_t = opus::remove_cvref_t<Layout>;
    static_assert(layout_t::cached_vec > 0);
    #pragma unroll
    for (opus::index_t i = 0; i < layout_t::num_issues; i++) {
        u.offsets[i] = lds_xor128_swizzle_offset(u.offsets[i]);
    }
}

template<typename T>
inline __device__ auto make_layout_ga(int lane_id, int wave_id_m, int wave_id_n, int stride_a) {
    constexpr int threads_k = T::B_K / T::VEC_A;
    constexpr int threads_m_per_block = T::BLOCK_SIZE / threads_k;
    constexpr int threads_m_per_wave = opus::get_warp_size() / threads_k;

    constexpr auto ga_block_shape = opus::make_tuple(
        opus::number<ceil_div_constexpr(T::HALF_B_M, threads_m_per_block)>{},
        opus::number<T::T_M>{},
        opus::number<threads_m_per_wave>{},
        opus::number<T::T_N>{},
        opus::number<threads_k>{},
        opus::number<T::VEC_A>{});

    constexpr auto ga_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}, opus::p_dim{}),
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
        opus::number<ceil_div_constexpr(T::smem_m_rep, num_waves)>{},
        opus::number<T::T_M>{},
        opus::number<T::T_N>{},
        opus::number<opus::get_warp_size()>{},
        opus::number<T::VEC_A>{});

    constexpr auto sa_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}));

    return opus::make_layout<T::VEC_A>(
        sa_block_shape,
        opus::unfold_x_stride(sa_block_dim, sa_block_shape, opus::tuple{T::smem_linear_wave, 1_I}),
        opus::unfold_p_coord(sa_block_dim, opus::tuple{wave_id_m, wave_id_n, lane_id}));
}

template<typename T>
inline __device__ auto make_layout_ra(int lane_id, int wave_id_m) {
    constexpr int total_ds_a_reads = T::E_K * T::W_M * T::W_K / (opus::get_warp_size() * T::VEC_A);
    auto lane_id_m = lane_id % T::W_M;

    constexpr auto ra_block_shape = opus::make_tuple(
        opus::number<T::E_M>{},
        opus::number<T::T_M>{},
        opus::number<2>{},
        opus::number<T::W_M / 2>{},
        opus::number<total_ds_a_reads>{},
        opus::number<opus::get_warp_size() / T::W_M>{},
        opus::number<T::VEC_A>{});

    constexpr auto ra_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}, opus::p_dim{}, opus::y_dim{}));

    return opus::make_layout<T::VEC_A>(
        ra_block_shape,
        opus::unfold_x_stride(ra_block_dim, ra_block_shape, opus::tuple{T::smem_linear_wave, 1_I}),
        opus::unfold_p_coord(
            ra_block_dim,
            opus::tuple{wave_id_m, lane_id_m & 1, lane_id_m >> 1, lane_id / T::W_M}));
}

template<typename T>
inline __device__ auto make_layout_sb_preshuffle(
    int lane_id,
    int wave_id_m,
    int wave_id_n)
{
    constexpr int num_waves = T::BLOCK_SIZE / opus::get_warp_size();

    constexpr auto sb_block_shape = opus::make_tuple(
        opus::number<T::smem_n_rep / num_waves>{},
        opus::number<T::T_M>{},
        opus::number<T::T_N>{},
        opus::number<opus::get_warp_size()>{},
        opus::number<T::VEC_B>{});

    constexpr auto sb_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}));

    return opus::make_layout<T::VEC_B>(
        sb_block_shape,
        opus::unfold_x_stride(sb_block_dim, sb_block_shape, opus::tuple{T::smem_linear_wave, 1_I}),
        opus::unfold_p_coord(sb_block_dim, opus::tuple{wave_id_m, wave_id_n, lane_id}));
}

template<typename T>
inline __device__ auto make_layout_rb(int lane_id, int wave_id_n) {
    constexpr int total_ds_b_reads = T::E_K * T::W_N * T::W_K / (opus::get_warp_size() * T::VEC_B);
    auto lane_id_n = lane_id % T::W_N;

    constexpr auto rb_block_shape = opus::make_tuple(
        opus::number<T::E_N>{},
        opus::number<T::T_N>{},
        opus::number<2>{},
        opus::number<T::W_N / 2>{},
        opus::number<total_ds_b_reads>{},
        opus::number<opus::get_warp_size() / T::W_N>{},
        opus::number<T::VEC_B>{});

    constexpr auto rb_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}, opus::p_dim{}, opus::y_dim{}));

    return opus::make_layout<T::VEC_B>(
        rb_block_shape,
        opus::unfold_x_stride(rb_block_dim, rb_block_shape, opus::tuple{T::smem_linear_wave, 1_I}),
        opus::unfold_p_coord(
            rb_block_dim,
            opus::tuple{wave_id_n, lane_id_n & 1, lane_id_n >> 1, lane_id / T::W_N}));
}

template<typename T>
inline __device__ auto make_layout_gb_preshuffle(
    int lane_id,
    int wave_id_m,
    int wave_id_n,
    int col,
    int half_tile_n,
    int stride_b)
{
    constexpr int threads_k = T::B_K / T::VEC_B;
    constexpr int threads_n_per_block = T::BLOCK_SIZE / threads_k;
    constexpr int threads_n_per_wave = opus::get_warp_size() / threads_k;
    constexpr int loads = T::HALF_B_N / threads_n_per_block;
    static_assert(T::VEC_B == 16);
    static_assert(T::HALF_B_N % threads_n_per_block == 0);
    static_assert(threads_n_per_block % 16 == 0);

    const int lane_k = lane_id % threads_k;
    const int thread_n = wave_id_m * (threads_n_per_wave * T::T_N)
                       + (lane_id / threads_k) * T::T_N
                       + wave_id_n;
    const int n_blk_base = (col + half_tile_n * T::HALF_B_N) / 16;
    const int k_blocks = stride_b / 32;
    const int n_blk = thread_n / 16;
    const int n_in = thread_n - n_blk * 16;
    const int k_blk_sub = lane_k / 2;
    const int k_mid = lane_k & 1;

    constexpr auto gb_block_shape = opus::make_tuple(
        opus::number<loads>{},
        opus::number<T::HALF_B_N / 16>{},
        opus::number<T::B_K / 32>{},
        opus::number<2>{},
        opus::number<16>{},
        opus::number<T::VEC_B>{});

    constexpr auto gb_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::p_dim{}, opus::y_dim{}));

    auto u_gb = opus::make_layout<T::VEC_B>(
        gb_block_shape,
        opus::unfold_x_stride(gb_block_dim, gb_block_shape, opus::tuple{k_blocks * 512, 1_I}),
        opus::unfold_p_coord(gb_block_dim, opus::tuple{n_blk, k_blk_sub, k_mid, n_in}));
    u_gb += n_blk_base * k_blocks * 512;
    return u_gb;
}

template<typename T>
inline __device__ auto make_layout_sfa_bpreshuffle(
    int block_row,
    int half_tile_m,
    int lane_id,
    int wave_id_m)
{
    constexpr auto sfa_block_shape = opus::make_tuple(
        opus::number<T::E_M>{},
        opus::number<T::T_M>{},
        opus::number<T::W_M>{});

    constexpr auto sfa_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}),
        opus::make_tuple(opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}));

    auto u_sfa = opus::make_layout<1>(
        sfa_block_shape,
        opus::unfold_x_stride(
            sfa_block_dim,
            sfa_block_shape,
            opus::tuple{opus::number<T::T_M * T::W_M>{}, opus::number<T::W_M>{}, 1_I}),
        opus::unfold_p_coord(sfa_block_dim, opus::tuple{wave_id_m, lane_id & (T::W_M - 1)}));
    u_sfa += block_row + half_tile_m * T::HALF_B_M;
    return u_sfa;
}

template<typename T, typename GMem, typename Layout>
inline __device__ auto load_sfa_oob_masked(
    GMem& gmem,
    const Layout& u_sfa,
    int base_offset,
    int m)
{
    using D_SF = typename GMem::scalar_type;
    using layout_t = opus::remove_cvref_t<Layout>;
    static_assert(std::is_same_v<D_SF, float>);
    static_assert(layout_t::cached_vec == 1);
    static_assert(layout_t::num_issues == T::E_M);

    opus::vector_t<D_SF, T::E_M> value;
    opus::static_for<T::E_M>([&](auto em) {
        constexpr int em_idx = decltype(em)::value;
        const int global_row = u_sfa.offsets[em_idx];

        D_SF local = D_SF{0};
        if (__builtin_expect(global_row < m, 1)) {
            local = gmem.template load<1>(base_offset + global_row)[0];
        }
        value[em_idx] = local;
    });
    return value;
}

template<typename T, typename GMem, typename Layout>
inline __device__ auto load_sfa_inbounds(
    GMem& gmem,
    const Layout& u_sfa,
    int base_offset)
{
    using D_SF = typename GMem::scalar_type;
    using layout_t = opus::remove_cvref_t<Layout>;
    static_assert(std::is_same_v<D_SF, float>);
    static_assert(layout_t::cached_vec == 1);
    static_assert(layout_t::num_issues == T::E_M);

    opus::vector_t<D_SF, T::E_M> value;
    opus::static_for<T::E_M>([&](auto em) {
        constexpr int em_idx = decltype(em)::value;
        value[em_idx] = gmem.template load<1>(base_offset + u_sfa.offsets[em_idx])[0];
    });
    return value;
}

template<int E_M, typename D_SF>
inline __device__ auto make_row_scale_gfx942_a8w8(
    const opus::vector_t<D_SF, E_M>& scale_a,
    const opus::vector_t<D_SF, 1>& scale_b)
{
    static_assert(std::is_same_v<D_SF, float>);
    opus::vector_t<D_SF, E_M> row_scale;
    opus::static_for<E_M>([&](auto em) {
        constexpr int em_idx = decltype(em)::value;
        row_scale[em_idx] = scale_a[em_idx] * scale_b[0];
    });
    return row_scale;
}

template<int E_M, int E_N, int ELEM_C, typename D_ACC, typename D_SF>
inline __device__ void scale_c_tile_packed_row_scale_gfx942_a8w8(
    const opus::vector_t<D_ACC, E_M * E_N * ELEM_C>& c_mma,
    const opus::vector_t<D_SF, E_M>& row_scale,
    opus::vector_t<D_ACC, E_M * E_N * ELEM_C>& acc)
{
    static_assert(std::is_same_v<D_ACC, float>);
    static_assert(std::is_same_v<D_SF, float>);
    constexpr int row_len = E_N * ELEM_C;
    static_assert(row_len % 2 == 0);

    using f32x2 = float __attribute__((ext_vector_type(2)));
    opus::static_for<E_M>([&](auto em) {
        constexpr int row = decltype(em)::value;
        const D_ACC scale = row_scale[row];
        const f32x2 scale2{scale, scale};
        opus::static_for<row_len / 2>([&](auto j) {
            constexpr int idx = row * row_len + decltype(j)::value * 2;
            f32x2 c_pair{c_mma[idx], c_mma[idx + 1]};
            f32x2 acc_pair{acc[idx], acc[idx + 1]};
            acc_pair = __builtin_elementwise_fma(c_pair, scale2, acc_pair);
            acc[idx] = acc_pair[0];
            acc[idx + 1] = acc_pair[1];
        });
    });
}

template<typename T, typename Mma, typename GC, typename Kargs, typename VC>
inline __device__ void epilogue_store_c_gfx942_a8w8(
    Mma& mma,
    GC& g_c,
    const Kargs& kargs,
    VC& v_c,
    int wave_id_m,
    int wave_id_n,
    int lane_id,
    int row,
    int col,
    char* smem_a_bytes,
    char* smem_b_bytes)
{
    using D_C = typename T::D_C;
    using D_ACC = typename T::D_ACC;
    static_assert(sizeof(D_C) == 2);

    constexpr opus::index_t acc_chunk = T::VEC_C * opus::vector_traits<D_ACC>::size();
    constexpr int half_tile_elems = T::HALF_B_M * T::HALF_B_N;
    constexpr int thread_tile_vec = half_tile_elems / T::BLOCK_SIZE;
    static_assert(thread_tile_vec * T::BLOCK_SIZE == half_tile_elems);
    constexpr int store_vec = thread_tile_vec;
    static_assert(store_vec == 16 / sizeof(D_C));
    constexpr int lds_stride = T::HALF_B_N + 8;

    auto p_coord_c = opus::make_tuple(wave_id_m, lane_id % mma.grpn_c, wave_id_n, lane_id / mma.grpn_c);
    auto u_lds_c = opus::partition_layout_c<T::VEC_C>(
        mma, opus::make_tuple(opus::number<lds_stride>{}, 1_I), p_coord_c);
    auto offsets_lds = opus::layout_to_offsets<T::VEC_C>(u_lds_c);

    using LT_C = opus::layout_load_traits<decltype(u_lds_c), T::VEC_C>;
    constexpr auto r_elem_c = LT_C::r_elem;

    opus::smem<D_C> s_c0 = opus::make_smem(reinterpret_cast<D_C*>(smem_a_bytes));
    opus::smem<D_C> s_c1 = opus::make_smem(reinterpret_cast<D_C*>(smem_b_bytes));
    const int tid = opus::thread_id_x();

    auto c_offset = [&](int half_tile_m, int half_tile_n) {
        return half_tile_m * T::HALF_B_M * kargs.stride_c + half_tile_n * T::HALF_B_N;
    };
    auto stage_hm = [&](int hm) {
        auto& vc0 = v_c[hm][0];
        auto& vc1 = v_c[hm][1];

        #pragma unroll
        for (opus::index_t i = 0; i < r_elem_c.value; i++) {
            opus::vector_t<D_ACC, acc_chunk> chunk0, chunk1;
            #pragma unroll
            for (opus::index_t j = 0; j < acc_chunk; j++) {
                chunk0[j] = vc0[i * acc_chunk + j];
                chunk1[j] = vc1[i * acc_chunk + j];
            }
            s_c0.template store<T::VEC_C>(opus::cast<D_C>(chunk0), offsets_lds[i]);
            s_c1.template store<T::VEC_C>(opus::cast<D_C>(chunk1), offsets_lds[i]);
        }
    };

    auto drain_hm_full = [&](int hm) {
        const int linear = tid * store_vec;
        const int rd_row = linear / T::HALF_B_N;
        const int rd_col = linear % T::HALF_B_N;
        const int lds_rd_off = rd_row * lds_stride + rd_col;
        const int gmem_v_off = rd_row * kargs.stride_c + rd_col;

        auto coal0 = s_c0.template load<store_vec>(lds_rd_off);
        auto coal1 = s_c1.template load<store_vec>(lds_rd_off);
        g_c.template store<store_vec>(coal0, gmem_v_off, c_offset(hm, 0), opus::number<3>{});
        g_c.template store<store_vec>(coal1, gmem_v_off, c_offset(hm, 1), opus::number<3>{});
    };

    auto drain_hm_tail_m = [&](int hm) {
        const int linear = tid * store_vec;
        const int rd_row = linear / T::HALF_B_N;
        const int rd_col = linear % T::HALF_B_N;
        const int m_base = row + hm * T::HALF_B_M;
        const int gmem_v_off = rd_row * kargs.stride_c + rd_col;

        if (__builtin_expect(m_base + rd_row < kargs.m, 1)) {
            auto coal0 = s_c0.template load<store_vec>(rd_row * lds_stride + rd_col);
            auto coal1 = s_c1.template load<store_vec>(rd_row * lds_stride + rd_col);
            g_c.template store<store_vec>(coal0, gmem_v_off, c_offset(hm, 0), opus::number<3>{});
            g_c.template store<store_vec>(coal1, gmem_v_off, c_offset(hm, 1), opus::number<3>{});
        }
    };

    const bool full_m_tile = row + T::B_M <= kargs.m;

    if (__builtin_expect(full_m_tile, 1)) {
        #pragma unroll
        for (int hm = 0; hm < 2; hm++) {
            stage_hm(hm);
            opus::s_waitcnt_lgkmcnt(opus::number<0>{});
            __builtin_amdgcn_s_barrier();
            drain_hm_full(hm);
            opus::s_waitcnt_lgkmcnt(opus::number<0>{});

            if (hm == 0) {
                __builtin_amdgcn_s_barrier();
            }
        }
        return;
    }

    #pragma unroll
    for (int hm = 0; hm < 2; hm++) {
        if (__builtin_expect(row + hm * T::HALF_B_M >= kargs.m, 0)) {
            continue;
        }
        stage_hm(hm);
        opus::s_waitcnt_lgkmcnt(opus::number<0>{});
        __builtin_amdgcn_s_barrier();
        drain_hm_tail_m(hm);
        opus::s_waitcnt_lgkmcnt(opus::number<0>{});
        if (hm == 0) {
            __builtin_amdgcn_s_barrier();
        }
    }
}

#endif // __HIP_DEVICE_COMPILE__ (layout functions)

// ============================================================================
// Single-buffer GEMM kernel with block-scale (a8w8 + scale 1x128x128).
// Kernel definition is visible on both passes; the host pass uses it for stubs.
// ============================================================================

template<typename Traits>
__global__ __launch_bounds__(Traits::BLOCK_SIZE, 2)
void gemm_a8w8_blockscale_bpreshuffle_singlebuf_kernel_gfx942(opus_gemm_a8w8_blockscale_bpreshuffle_kargs_gfx942 kargs) {
#ifdef __HIP_DEVICE_COMPILE__
    using T = opus::remove_cvref_t<Traits>;
    using D_A   = typename T::D_A;
    using D_B   = typename T::D_B;
    using D_C   = typename T::D_C;
    using D_ACC = typename T::D_ACC;
    using D_SF  = typename T::D_SF;

    int wave_id = __builtin_amdgcn_readfirstlane(opus::thread_id_x() / opus::get_warp_size());
    int lane_id = opus::thread_id_x() % opus::get_warp_size();

    auto g_b = opus::make_gmem(reinterpret_cast<const D_B*>(kargs.ptr_b));
    auto g_sfa = opus::make_gmem(reinterpret_cast<const D_SF*>(kargs.ptr_sfa));

    int wave_id_m = wave_id % T::T_M;
    int wave_id_n = wave_id / T::T_M;

    auto u_ga = make_layout_ga<T>(lane_id, wave_id_m, wave_id_n, kargs.stride_a);
    auto u_sa = make_layout_sa<T>(lane_id, wave_id_m, wave_id_n);
    auto u_ra = make_layout_ra<T>(lane_id, wave_id_m);
    auto u_sb = make_layout_sb_preshuffle<T>(lane_id, wave_id_m, wave_id_n);
    auto u_rb = make_layout_rb<T>(lane_id, wave_id_n);
    apply_lds_xor128_swizzle(u_sa);
    apply_lds_xor128_swizzle(u_ra);
    apply_lds_xor128_swizzle(u_sb);
    apply_lds_xor128_swizzle(u_rb);

    constexpr int smem_a_byte = T::smem_m_rep * T::smem_linear_wave * sizeof(D_A);
    __shared__ char smem_a[smem_a_byte * 2];
    static_assert(T::VEC_A == 16 && T::VEC_B == 16);
    using smem_a8w8_t = opus::smem<D_A>;
    using smem_b8w8_t = opus::smem<D_B>;
    smem_a8w8_t s_a[2] = {
        opus::make_smem(reinterpret_cast<D_A*>(smem_a)),
        opus::make_smem(reinterpret_cast<D_A*>(smem_a + smem_a_byte))
    };
    constexpr int smem_b_byte = T::smem_n_rep * T::smem_linear_wave * sizeof(D_B);
    __shared__ char smem_b[smem_b_byte * 2];
    smem_b8w8_t s_b[2] = {
        opus::make_smem(reinterpret_cast<D_B*>(smem_b)),
        opus::make_smem(reinterpret_cast<D_B*>(smem_b + smem_b_byte))
    };
    constexpr int b_tile_stride = (T::B_K / 32) * 512;
    auto mma = opus::make_tiled_mma<D_A, D_B, D_ACC>(
        opus::seq<T::E_M, T::E_N, T::E_K>{},
        opus::seq<T::T_M, T::T_N, T::T_K>{},
        opus::seq<T::W_M, T::W_N, T::W_K>{},
        opus::mfma_adaptor_swap_ab{});
    constexpr int ELEM_C = decltype(mma)::elem_c;

    const int row = opus::block_id_y() * T::B_M;
    const int col = opus::block_id_x() * T::B_N;
    {
        const int valid_m_rows = kargs.m - row < T::B_M ? kargs.m - row : T::B_M;
        const unsigned int g_a_bytes = valid_m_rows == T::B_M
            ? 0xffffffff
            : static_cast<unsigned int>(
                static_cast<size_t>(valid_m_rows) * kargs.stride_a * sizeof(D_A));
        auto g_a = opus::make_gmem(
            reinterpret_cast<const D_A*>(kargs.ptr_a) + row*kargs.stride_a,
            g_a_bytes);
        D_C* ptr_c_base = reinterpret_cast<D_C*>(kargs.ptr_c)
                       + row*kargs.stride_c + col;
        auto g_c = opus::make_gmem(ptr_c_base);
        auto g_sfb = opus::make_gmem(
            reinterpret_cast<const D_SF*>(kargs.ptr_sfb)
            + static_cast<int>(col / T::GROUP_N) * kargs.stride_sfb);
        auto u_gb_0 = make_layout_gb_preshuffle<T>(
            lane_id, wave_id_m, wave_id_n, col, 0, kargs.stride_b);
        auto u_gb_1 = make_layout_gb_preshuffle<T>(
            lane_id, wave_id_m, wave_id_n, col, 1, kargs.stride_b);
        auto u_sfa_0 = make_layout_sfa_bpreshuffle<T>(row, 0, lane_id, wave_id_m);
        auto u_sfa_1 = make_layout_sfa_bpreshuffle<T>(row, 1, lane_id, wave_id_m);

        typename decltype(mma)::vtype_a v_a[2];
        typename decltype(mma)::vtype_b v_b[2];
        typename decltype(mma)::vtype_c v_c[2][2], v_mma;
        opus::clear(v_c[0][0]);
        opus::clear(v_c[0][1]);
        opus::clear(v_c[1][0]);
        opus::clear(v_c[1][1]);

        using vtype_sfa = opus::vector_t<D_SF, T::E_M>;
        using vtype_sfb = opus::vector_t<D_SF, 1>;
        using vtype_scale = opus::vector_t<D_SF, T::E_M>;
        vtype_sfa v_sfa[2];
        vtype_sfb v_sfb, v_sfb_next;
        vtype_sfa v_sfa_next_leader[2];
        vtype_scale v_scale[2];

        const int loops = ceil_div(kargs.k, T::B_K);
        static_assert(T::B_N == T::GROUP_N);

        auto a_offset = [&](int half_tile_m, int tile_k) {
            return half_tile_m * T::HALF_B_M * kargs.stride_a + tile_k * T::B_K;
        };
        auto sfa_scale_k = [&](int tile_k) {
            return (tile_k * T::B_K) / T::GROUP_K;
        };
        auto sfb_offset = [&](int half_tile_n, int tile_k) {
            return (half_tile_n * T::HALF_B_N / T::GROUP_N) * kargs.stride_sfb
                 + (tile_k * T::B_K) / T::GROUP_K;
        };
        auto compute_mma_staged = [&](auto& va, auto& vb) {
            typename decltype(mma)::vtype_c acc;
            opus::clear(acc);
            acc = mma.step_k(opus::number<0>{}, va, vb, acc);
            acc = mma.step_k(opus::number<1>{}, va, vb, acc);
            opus::s_waitcnt_lgkmcnt(1_I);
            acc = mma.step_k(opus::number<2>{}, va, vb, acc);
            acc = mma.step_k(opus::number<3>{}, va, vb, acc);
            return acc;
        };

        if (__builtin_expect(valid_m_rows <= T::HALF_B_M, 0)) {
            v_sfa[0] = load_sfa_oob_masked<T>(g_sfa, u_sfa_0, sfa_scale_k(0) * kargs.m, kargs.m);
            v_sfb = opus::load<1>(g_sfb, sfb_offset(0, 0));
            const auto a_init_0 = opus::load<T::VEC_A>(g_a, u_ga, a_offset(0, 0));
            auto b_init_0 = opus::load<T::VEC_B>(g_b, u_gb_0);
            auto b_init_1 = opus::load<T::VEC_B>(g_b, u_gb_1);
            opus::s_waitcnt_vmcnt(opus::number<2 * T::b_buffer_load_insts>{});
            opus::store<T::VEC_A>(s_a[0], a_init_0, u_sa);
            opus::s_waitcnt_vmcnt(opus::number<T::b_buffer_load_insts>{});
            opus::store<T::VEC_B>(s_b[0], b_init_0, u_sb);
            opus::s_waitcnt_vmcnt(opus::number<0>{});
            opus::store<T::VEC_B>(s_b[1], b_init_1, u_sb);

            const int first_next_tile = loops > 1 ? 1 : 0;
            v_sfa_next_leader[0] =
                load_sfa_oob_masked<T>(g_sfa, u_sfa_0, sfa_scale_k(first_next_tile) * kargs.m, kargs.m);
            opus::s_waitcnt_lgkmcnt(0_I);
            __builtin_amdgcn_s_barrier();

            for(int tile = 0; tile < loops - 1; ++tile) {
                vtype_sfa v_sfa_next2_leader0;
                if (tile + 2 < loops) {
                    v_sfa_next2_leader0 =
                        load_sfa_oob_masked<T>(g_sfa, u_sfa_0, sfa_scale_k(tile + 2) * kargs.m, kargs.m);
                } else {
                    v_sfa_next2_leader0 = v_sfa_next_leader[0];
                }

                v_sfb_next = opus::load<1>(g_sfb, sfb_offset(0, tile + 1));
                const auto a_pref_0 = opus::load<T::VEC_A>(g_a, u_ga, a_offset(0, tile + 1));
                v_scale[0] = make_row_scale_gfx942_a8w8<T::E_M, D_SF>(v_sfa[0], v_sfb);
                const int next_b_offset = (tile + 1) * b_tile_stride;
                auto b_pref_0 = opus::load<T::VEC_B>(g_b, u_gb_0, next_b_offset);

                v_a[0] = opus::load<T::VEC_A>(s_a[0], u_ra);
                v_b[0] = opus::load<T::VEC_B>(s_b[0], u_rb);
                opus::s_waitcnt_lgkmcnt(4_I);
                __builtin_amdgcn_s_setprio(1);
                v_mma = compute_mma_staged(v_a[0], v_b[0]);
                __builtin_amdgcn_s_setprio(0);
                v_b[1] = opus::load<T::VEC_B>(s_b[1], u_rb);
                auto b_pref_1 = opus::load<T::VEC_B>(g_b, u_gb_1, next_b_offset);
                opus::s_waitcnt_vmcnt(opus::number<2 * T::b_buffer_load_insts>{});
                opus::store<T::VEC_A>(s_a[0], a_pref_0, u_sa);
                scale_c_tile_packed_row_scale_gfx942_a8w8<T::E_M, T::E_N, ELEM_C, D_ACC, D_SF>(
                    v_mma, v_scale[0], v_c[0][0]);

                opus::s_waitcnt_lgkmcnt(4_I);
                __builtin_amdgcn_s_setprio(1);
                v_mma = compute_mma_staged(v_a[0], v_b[1]);
                __builtin_amdgcn_s_setprio(0);
                opus::s_waitcnt_vmcnt(opus::number<T::b_buffer_load_insts>{});
                opus::store<T::VEC_B>(s_b[0], b_pref_0, u_sb);
                scale_c_tile_packed_row_scale_gfx942_a8w8<T::E_M, T::E_N, ELEM_C, D_ACC, D_SF>(
                    v_mma, v_scale[0], v_c[0][1]);

                opus::s_waitcnt_vmcnt(opus::number<0>{});
                opus::store<T::VEC_B>(s_b[1], b_pref_1, u_sb);
                auto v_sfa_next0 = v_sfa_next_leader[0];
                v_sfa_next_leader[0] = v_sfa_next2_leader0;
                opus::s_waitcnt_lgkmcnt(1_I);
                __builtin_amdgcn_s_barrier();
                v_sfa[0] = v_sfa_next0;
                v_sfb = v_sfb_next;
            }

            v_scale[0] = make_row_scale_gfx942_a8w8<T::E_M, D_SF>(v_sfa[0], v_sfb);
            v_a[0] = opus::load<T::VEC_A>(s_a[0], u_ra);
            v_b[0] = opus::load<T::VEC_B>(s_b[0], u_rb);
            v_b[1] = opus::load<T::VEC_B>(s_b[1], u_rb);
            opus::s_waitcnt_lgkmcnt(4_I);

            __builtin_amdgcn_s_setprio(1);
            v_mma = compute_mma_staged(v_a[0], v_b[0]);
            __builtin_amdgcn_s_setprio(0);
            scale_c_tile_packed_row_scale_gfx942_a8w8<T::E_M, T::E_N, ELEM_C, D_ACC, D_SF>(
                v_mma, v_scale[0], v_c[0][0]);
            __builtin_amdgcn_s_setprio(1);
            v_mma = compute_mma_staged(v_a[0], v_b[1]);
            __builtin_amdgcn_s_setprio(0);
            scale_c_tile_packed_row_scale_gfx942_a8w8<T::E_M, T::E_N, ELEM_C, D_ACC, D_SF>(
                v_mma, v_scale[0], v_c[0][1]);

            // Epilogue reuses A/B LDS as C scratch; all waves must finish final LDS reads first.
            __builtin_amdgcn_s_barrier();
            epilogue_store_c_gfx942_a8w8<T>(
                mma, g_c, kargs, v_c, wave_id_m, wave_id_n, lane_id, row, col, smem_a, smem_b);
            return;
        }
        v_sfa[0] = load_sfa_inbounds<T>(g_sfa, u_sfa_0, sfa_scale_k(0) * kargs.m);
        v_sfb = opus::load<1>(g_sfb, sfb_offset(0, 0));
        const auto a_init_0 = opus::load<T::VEC_A>(g_a, u_ga, a_offset(0, 0));
        auto b_init_0 = opus::load<T::VEC_B>(g_b, u_gb_0);
        v_sfa[1] = load_sfa_oob_masked<T>(g_sfa, u_sfa_1, sfa_scale_k(0) * kargs.m, kargs.m);
        const auto a_init_1 = opus::load<T::VEC_A>(g_a, u_ga, a_offset(1, 0));
        auto b_init_1 = opus::load<T::VEC_B>(g_b, u_gb_1);
        opus::s_waitcnt_vmcnt(opus::number<T::a_buffer_load_insts + 2 * T::b_buffer_load_insts>{});
        opus::store<T::VEC_A>(s_a[0], a_init_0, u_sa);
        opus::s_waitcnt_vmcnt(opus::number<T::a_buffer_load_insts + T::b_buffer_load_insts>{});
        opus::store<T::VEC_B>(s_b[0], b_init_0, u_sb);
        opus::s_waitcnt_vmcnt(opus::number<T::b_buffer_load_insts>{});
        opus::store<T::VEC_A>(s_a[1], a_init_1, u_sa);
        opus::s_waitcnt_vmcnt(opus::number<0>{});
        opus::store<T::VEC_B>(s_b[1], b_init_1, u_sb);
        const int first_next_tile = loops > 1 ? 1 : 0;
        v_sfa_next_leader[0] =
            load_sfa_inbounds<T>(g_sfa, u_sfa_0, sfa_scale_k(first_next_tile) * kargs.m);
        v_sfa_next_leader[1] =
            load_sfa_oob_masked<T>(g_sfa, u_sfa_1, sfa_scale_k(first_next_tile) * kargs.m, kargs.m);
        opus::s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_s_barrier();

        for(int tile = 0; tile < loops - 1; ++tile) {
            vtype_sfa v_sfa_next2_leader[2];
            if (tile + 2 < loops) {
                v_sfa_next2_leader[0] =
                    load_sfa_inbounds<T>(g_sfa, u_sfa_0, sfa_scale_k(tile + 2) * kargs.m);
                v_sfa_next2_leader[1] =
                    load_sfa_oob_masked<T>(g_sfa, u_sfa_1, sfa_scale_k(tile + 2) * kargs.m, kargs.m);
            } else {
                v_sfa_next2_leader[0] = v_sfa_next_leader[0];
                v_sfa_next2_leader[1] = v_sfa_next_leader[1];
            }

            v_sfb_next = opus::load<1>(g_sfb, sfb_offset(0, tile + 1));
            const auto a_pref_0 = opus::load<T::VEC_A>(g_a, u_ga, a_offset(0, tile + 1));
            v_scale[0] = make_row_scale_gfx942_a8w8<T::E_M, D_SF>(v_sfa[0], v_sfb);
            v_scale[1] = make_row_scale_gfx942_a8w8<T::E_M, D_SF>(v_sfa[1], v_sfb);

            v_a[0] = opus::load<T::VEC_A>(s_a[0], u_ra);
            v_b[0] = opus::load<T::VEC_B>(s_b[0], u_rb);
            opus::s_waitcnt_lgkmcnt(4_I);
            __builtin_amdgcn_s_setprio(1);
            v_mma = compute_mma_staged(v_a[0], v_b[0]);
            __builtin_amdgcn_s_setprio(0);
            v_a[1] = opus::load<T::VEC_A>(s_a[1], u_ra);
            const int next_b_offset = (tile + 1) * b_tile_stride;
            auto b_pref_0 = opus::load<T::VEC_B>(g_b, u_gb_0, next_b_offset);
            opus::s_waitcnt_vmcnt(opus::number<T::a_buffer_load_insts>{});
            opus::store<T::VEC_A>(s_a[0], a_pref_0, u_sa);
            scale_c_tile_packed_row_scale_gfx942_a8w8<T::E_M, T::E_N, ELEM_C, D_ACC, D_SF>(
                v_mma, v_scale[0], v_c[0][0]);

            opus::s_waitcnt_lgkmcnt(4_I);
            __builtin_amdgcn_s_setprio(1);
            v_mma = compute_mma_staged(v_a[1], v_b[0]);
            v_b[1] = opus::load<T::VEC_B>(s_b[1], u_rb);
            __builtin_amdgcn_s_setprio(0);
            const auto a_pref_1 = opus::load<T::VEC_A>(g_a, u_ga, a_offset(1, tile + 1));
            auto b_pref_1 = opus::load<T::VEC_B>(g_b, u_gb_1, next_b_offset);
            scale_c_tile_packed_row_scale_gfx942_a8w8<T::E_M, T::E_N, ELEM_C, D_ACC, D_SF>(
                v_mma, v_scale[1], v_c[1][0]);

            opus::s_waitcnt_lgkmcnt(4_I);
            __builtin_amdgcn_s_setprio(1);
            v_mma = compute_mma_staged(v_a[0], v_b[1]);
            __builtin_amdgcn_s_setprio(0);
            opus::s_waitcnt_vmcnt(opus::number<T::a_buffer_load_insts + T::b_buffer_load_insts>{});
            opus::store<T::VEC_B>(s_b[0], b_pref_0, u_sb);
            scale_c_tile_packed_row_scale_gfx942_a8w8<T::E_M, T::E_N, ELEM_C, D_ACC, D_SF>(
                v_mma, v_scale[0], v_c[0][1]);

            opus::s_waitcnt_vmcnt(opus::number<T::a_buffer_load_insts>{});
            opus::store<T::VEC_A>(s_a[1], a_pref_1, u_sa);

            __builtin_amdgcn_s_setprio(1);
            v_mma = compute_mma_staged(v_a[1], v_b[1]);
            __builtin_amdgcn_s_setprio(0);

            opus::s_waitcnt_vmcnt(opus::number<0>{});
            opus::store<T::VEC_B>(s_b[1], b_pref_1, u_sb);

            auto v_sfa_next0 = v_sfa_next_leader[0];
            auto v_sfa_next1 = v_sfa_next_leader[1];
            v_sfa_next_leader[0] = v_sfa_next2_leader[0];
            v_sfa_next_leader[1] = v_sfa_next2_leader[1];
            opus::s_waitcnt_lgkmcnt(1_I);
            __builtin_amdgcn_s_barrier();
            scale_c_tile_packed_row_scale_gfx942_a8w8<T::E_M, T::E_N, ELEM_C, D_ACC, D_SF>(
                v_mma, v_scale[1], v_c[1][1]);
            v_sfa[0] = v_sfa_next0;
            v_sfa[1] = v_sfa_next1;
            v_sfb = v_sfb_next;
        }

        {
            v_scale[0] = make_row_scale_gfx942_a8w8<T::E_M, D_SF>(v_sfa[0], v_sfb);
            v_scale[1] = make_row_scale_gfx942_a8w8<T::E_M, D_SF>(v_sfa[1], v_sfb);

            v_a[0] = opus::load<T::VEC_A>(s_a[0], u_ra);
            v_b[0] = opus::load<T::VEC_B>(s_b[0], u_rb);
            v_a[1] = opus::load<T::VEC_A>(s_a[1], u_ra);
            v_b[1] = opus::load<T::VEC_B>(s_b[1], u_rb);
            opus::s_waitcnt_lgkmcnt(4_I);

            __builtin_amdgcn_s_setprio(1);
            v_mma = compute_mma_staged(v_a[0], v_b[0]);
            __builtin_amdgcn_s_setprio(0);
            scale_c_tile_packed_row_scale_gfx942_a8w8<T::E_M, T::E_N, ELEM_C, D_ACC, D_SF>(
                v_mma, v_scale[0], v_c[0][0]);
            __builtin_amdgcn_s_setprio(1);
            v_mma = compute_mma_staged(v_a[1], v_b[0]);
            __builtin_amdgcn_s_setprio(0);
            scale_c_tile_packed_row_scale_gfx942_a8w8<T::E_M, T::E_N, ELEM_C, D_ACC, D_SF>(
                v_mma, v_scale[1], v_c[1][0]);
            __builtin_amdgcn_s_setprio(1);
            v_mma = compute_mma_staged(v_a[0], v_b[1]);
            __builtin_amdgcn_s_setprio(0);
            scale_c_tile_packed_row_scale_gfx942_a8w8<T::E_M, T::E_N, ELEM_C, D_ACC, D_SF>(
                v_mma, v_scale[0], v_c[0][1]);
            __builtin_amdgcn_s_setprio(1);
            v_mma = compute_mma_staged(v_a[1], v_b[1]);
            __builtin_amdgcn_s_setprio(0);
            scale_c_tile_packed_row_scale_gfx942_a8w8<T::E_M, T::E_N, ELEM_C, D_ACC, D_SF>(
                v_mma, v_scale[1], v_c[1][1]);
        }

        // Epilogue reuses A/B LDS as C scratch; all waves must finish final LDS reads first.
        __builtin_amdgcn_s_barrier();
        epilogue_store_c_gfx942_a8w8<T>(
            mma, g_c, kargs, v_c, wave_id_m, wave_id_n, lane_id, row, col, smem_a, smem_b);
    }
#endif // __HIP_DEVICE_COMPILE__
}
