// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// gfx942 a16w16 shared helpers: layouts (ga/sa/ra/gb/sb/rb), smem_x1b XOR
// swizzle wrapper, and epilogue store helpers used by all kid families.
#pragma once

#include <type_traits>

#ifdef __HIP_DEVICE_COMPILE__

// X1b row^col XOR swizzle wrapper (BC=0 verified at STRIDE_ELEM=512).
template<typename T_, int STRIDE_ELEM = 512>
struct smem_x1b {
    using inner_t = opus::smem<T_>;
    using T = typename inner_t::T;
    using scalar_type = typename inner_t::scalar_type;
    static constexpr opus::index_t vector_size = inner_t::vector_size;
    template<opus::index_t vec = 1> using vector_type = typename inner_t::template vector_type<vec>;

    inner_t inner;
    OPUS_LDS_ADDR char* ptr;

    OPUS_D smem_x1b(void* p) : inner(p), ptr(inner.ptr) {}

    OPUS_D static int swiz(int v_os) {
        unsigned uv = static_cast<unsigned>(v_os);
        unsigned row = uv / STRIDE_ELEM;
        unsigned col = uv - row * STRIDE_ELEM;
        unsigned xor_col = ((row ^ (col >> 5)) & 7u) << 3;
        return static_cast<int>(row * STRIDE_ELEM + (col ^ xor_col));
    }

    template<opus::index_t vec = 1>
    OPUS_D auto load(int v_os) { return inner.template load<vec>(swiz(v_os)); }

    template<opus::index_t vec = 1, typename V,
             std::enable_if_t<(opus::is_vector_v<V> || opus::is_dtype_v<V> || opus::is_array_v<V>), bool> = true>
    OPUS_D void store(const V& x, int v_os) { inner.template store<vec>(x, swiz(v_os)); }

    template<opus::index_t vec = 1, typename Layout,
             std::enable_if_t<opus::is_layout_v<Layout>, bool> = true>
    OPUS_D auto load(const Layout& u) {
        using LT = opus::layout_load_traits<Layout, vec>;
        constexpr auto r_elem = LT::r_elem;
        auto offsets = opus::layout_to_offsets<vec>(u);
        opus::vector_t<scalar_type, vec * vector_size * r_elem.value> r;
        for (opus::index_t i = 0; i < r_elem.value; i++) {
            auto tmp = inner.template load<vec>(swiz(offsets[i]));
            for (opus::index_t j = 0; j < vec * vector_size; j++)
                r[i * vec * vector_size + j] = tmp[j];
        }
        return r;
    }

    template<opus::index_t vec = 1, typename V, typename Layout,
             std::enable_if_t<((opus::is_array_v<V> || opus::is_dtype_v<V> || opus::is_vector_v<V>)
                                && opus::is_layout_v<Layout>), bool> = true>
    OPUS_D void store(const V& x, const Layout& u) {
        using LT = opus::layout_load_traits<Layout, vec>;
        constexpr auto r_elem = LT::r_elem;
        auto offsets = opus::layout_to_offsets<vec>(u);
        auto a_ = [&]() {
            if constexpr (opus::is_array_v<V>) return opus::to_vector(x);
            else if constexpr (opus::is_dtype_v<V>) return opus::make_repeated_vector(x, opus::number<r_elem.value>{});
            else if constexpr (opus::is_vector_v<V>) return x;
        }();
        for (opus::index_t i = 0; i < r_elem.value; i++) {
            vector_type<vec> v_;
            for (opus::index_t j = 0; j < vec * vector_size; j++)
                v_[j] = a_[i * vec * vector_size + j];
            inner.template store<vec>(v_, swiz(offsets[i]));
        }
    }
};

namespace opus {
template<typename T_, int S> struct is_smem<smem_x1b<T_, S>> : true_type {};
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

template<typename T, int STRIDE_PAD = 16>
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
        opus::unfold_x_stride(sa_block_dim, sa_block_shape, opus::tuple{T::smem_linear_wave + STRIDE_PAD, 1_I}),
        opus::unfold_p_coord(sa_block_dim, opus::tuple{wave_id_m, wave_id_n, lane_id}));
}

template<typename T, int STRIDE_PAD = 16>
inline __device__ auto make_layout_ra(int lane_id, int wave_id_m) {
    constexpr int total_ds_a_reads = T::E_K * T::W_M * T::W_K / (opus::get_warp_size() * T::VEC_A);
    constexpr int num_waves = T::BLOCK_SIZE / opus::get_warp_size();
    constexpr int sa_iters = T::smem_m_rep / num_waves;

    auto lane_id_m = lane_id % T::W_M;

    if constexpr (sa_iters >= T::E_M) {
        constexpr auto ra_block_shape = opus::make_tuple(
            opus::number<T::E_M>{},
            opus::number<T::T_M>{},
            opus::number<T::T_N>{},
            opus::number<T::W_M / T::T_N>{},
            opus::number<total_ds_a_reads>{},
            opus::number<opus::get_warp_size() / T::W_M>{},
            opus::number<T::VEC_A>{});

        constexpr auto ra_block_dim = opus::make_tuple(
            opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}),
            opus::make_tuple(opus::p_dim{}, opus::y_dim{}, opus::p_dim{}, opus::y_dim{}));

        return opus::make_layout<T::VEC_A>(
            ra_block_shape,
            opus::unfold_x_stride(ra_block_dim, ra_block_shape, opus::tuple{T::smem_linear_wave + STRIDE_PAD, 1_I}),
            opus::unfold_p_coord(ra_block_dim, opus::tuple{wave_id_m, lane_id_m % T::T_N, lane_id_m / T::T_N, lane_id / T::W_M}));
    } else {
        constexpr auto ra_block_shape = opus::make_tuple(
            opus::number<T::E_M>{},
            opus::number<T::T_N>{},
            opus::number<T::T_M>{},
            opus::number<T::W_M / T::T_N>{},
            opus::number<total_ds_a_reads>{},
            opus::number<opus::get_warp_size() / T::W_M>{},
            opus::number<T::VEC_A>{});

        constexpr auto ra_block_dim = opus::make_tuple(
            opus::make_tuple(opus::y_dim{}, opus::p_dim{}),
            opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::y_dim{}, opus::p_dim{}, opus::y_dim{}));

        return opus::make_layout<T::VEC_A>(
            ra_block_shape,
            opus::unfold_x_stride(ra_block_dim, ra_block_shape, opus::tuple{T::smem_linear_wave + STRIDE_PAD, 1_I}),
            opus::unfold_p_coord(ra_block_dim, opus::tuple{lane_id_m % T::T_N, wave_id_m, lane_id_m / T::T_N, lane_id / T::W_M}));
    }
}

template<typename T>
inline __device__ auto make_layout_gb(int lane_id, int wave_id_m, int wave_id_n, int stride_b) {
    constexpr int threads_k = T::B_K / T::VEC_B;
    constexpr int threads_n_per_block = T::BLOCK_SIZE / threads_k;
    constexpr int threads_n_per_wave = opus::get_warp_size() / threads_k;

    constexpr auto gb_block_shape = opus::make_tuple(
        opus::number<T::HALF_B_N / threads_n_per_block>{},
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

template<typename T, int STRIDE_PAD = 16>
inline __device__ auto make_layout_sb(int lane_id, int wave_id_m, int wave_id_n) {
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
        opus::unfold_x_stride(sb_block_dim, sb_block_shape, opus::tuple{T::smem_linear_wave + STRIDE_PAD, 1_I}),
        opus::unfold_p_coord(sb_block_dim, opus::tuple{wave_id_m, wave_id_n, lane_id}));
}

// wave_id_n split divisor = T_M (kid 10000 a16w16_kbuf1_large_tile).
template<typename T>
inline __device__ auto make_layout_rb_wave_m_major(int lane_id, int wave_id_n) {
    constexpr int total_ds_b_reads = T::E_K * T::W_N * T::W_K / (opus::get_warp_size() * T::VEC_B);
    constexpr int num_waves = T::BLOCK_SIZE / opus::get_warp_size();
    constexpr int sb_iters = T::smem_n_rep / num_waves;

    auto lane_id_n = lane_id % T::W_N;

    constexpr auto rb_block_shape = opus::make_tuple(
        opus::number<T::E_N>{},
        opus::number<T::T_N / T::T_M>{},
        opus::number<T::T_N>{},
        opus::number<T::T_M>{},
        opus::number<T::W_N / T::T_N>{},
        opus::number<total_ds_b_reads>{},
        opus::number<opus::get_warp_size() / T::W_N>{},
        opus::number<T::VEC_B>{});

    constexpr auto rb_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::y_dim{}, opus::p_dim{}, opus::y_dim{}));

    return opus::make_layout<T::VEC_B>(
        rb_block_shape,
        opus::unfold_x_stride(rb_block_dim, rb_block_shape, opus::tuple{T::smem_linear_wave + 16, 1_I}),
        opus::unfold_p_coord(rb_block_dim, opus::tuple{wave_id_n / T::T_M, lane_id_n % T::T_N, wave_id_n % T::T_M, lane_id_n / T::T_N, lane_id / T::W_N}));
}

// wave_id_n split divisor = T_N/T_M (gfx942 splitK / P1 / WKC families).
template<typename T, int STRIDE_PAD = 0>
inline __device__ auto make_layout_rb_wave_n_major(int lane_id, int wave_id_n) {
    constexpr int total_ds_b_reads = T::E_K * T::W_N * T::W_K / (opus::get_warp_size() * T::VEC_B);
    constexpr int wn_per_wm_grp = T::T_N / T::T_M;

    auto lane_id_n = lane_id % T::W_N;

    constexpr auto rb_block_shape = opus::make_tuple(
        opus::number<T::E_N>{},
        opus::number<wn_per_wm_grp>{},
        opus::number<T::T_N>{},
        opus::number<wn_per_wm_grp>{},
        opus::number<T::W_N / T::T_N>{},
        opus::number<total_ds_b_reads>{},
        opus::number<opus::get_warp_size() / T::W_N>{},
        opus::number<T::VEC_B>{});

    constexpr auto rb_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::y_dim{}, opus::p_dim{}, opus::y_dim{}));

    return opus::make_layout<T::VEC_B>(
        rb_block_shape,
        opus::unfold_x_stride(rb_block_dim, rb_block_shape, opus::tuple{T::smem_linear_wave + STRIDE_PAD, 1_I}),
        opus::unfold_p_coord(rb_block_dim, opus::tuple{wave_id_n / wn_per_wm_grp, lane_id_n % T::T_N, wave_id_n % wn_per_wm_grp, lane_id_n / T::T_N, lane_id / T::W_N}));
}

// ---- epilogue helpers ---- Non-splitk direct store to C with partial-tile pred.
template<typename T, typename Mma, typename GC, typename Kargs, typename VC>
OPUS_D inline void epilogue_store_c_if(
    Mma& mma, GC& g_c, const Kargs& kargs,
    VC& v_c, int wave_id_m, int wave_id_n, int lane_id, int row, int col)
{
    using D_C = typename T::D_C;
    using D_ACC = typename T::D_ACC;

    auto p_coord_c = opus::make_tuple(wave_id_m, lane_id % mma.grpn_c,
                                      wave_id_n, lane_id / mma.grpn_c);
    auto u_gc = opus::partition_layout_c<T::VEC_C>(mma,
                    opus::make_tuple(kargs.stride_c, 1_I), p_coord_c);
    auto u_gc_m = opus::partition_layout_c<T::VEC_C>(mma,
                    opus::make_tuple(1_I, 0_I), p_coord_c);
    auto u_gc_n = opus::partition_layout_c<T::VEC_C>(mma,
                    opus::make_tuple(0_I, 1_I), p_coord_c);

    auto c_offset = [&](int half_m, int half_n) {
        return half_m * T::HALF_B_M * kargs.stride_c + half_n * T::HALF_B_N;
    };

    auto do_store_if = [&](auto& vc, int g_c_offset, int m_base, int n_base) {
        auto pred = [&](auto... ids) {
            return (m_base + u_gc_m(ids...)) < kargs.m
                && (n_base + u_gc_n(ids...)) < kargs.n;
        };
        if constexpr (std::is_same_v<D_C, D_ACC>) {
            opus::store_if<T::VEC_C>(g_c, pred, vc, u_gc, g_c_offset);
        } else {
            auto vc_out = opus::cast<D_C>(vc);
            opus::store_if<T::VEC_C>(g_c, pred, vc_out, u_gc, g_c_offset);
        }
    };

    do_store_if(v_c[0][0], c_offset(0, 0), row + 0 * T::HALF_B_M, col + 0 * T::HALF_B_N);
    do_store_if(v_c[0][1], c_offset(0, 1), row + 0 * T::HALF_B_M, col + 1 * T::HALF_B_N);
    do_store_if(v_c[1][0], c_offset(1, 0), row + 1 * T::HALF_B_M, col + 0 * T::HALF_B_N);
    do_store_if(v_c[1][1], c_offset(1, 1), row + 1 * T::HALF_B_M, col + 1 * T::HALF_B_N);
}

// Nosplit LDS-staged store: stages each half-tile to LDS then reads back as STORE_VEC-wide chunk for
// coalesced gmem write (dwordx...
template<typename T, typename Mma, typename GC, typename Kargs, typename VC>
OPUS_D inline void epilogue_store_c_lds_staged(
    Mma& mma, GC& g_c, const Kargs& kargs,
    VC& v_c, int wave_id_m, int wave_id_n, int lane_id, int row, int col,
    char* smem_a_bytes, char* smem_b_bytes)
{
    using D_C = typename T::D_C;
    using D_ACC = typename T::D_ACC;

    auto p_coord_c = opus::make_tuple(wave_id_m, lane_id % mma.grpn_c,
                                      wave_id_n, lane_id / mma.grpn_c);
    auto u_gc = opus::partition_layout_c<T::VEC_C>(mma,
                    opus::make_tuple(kargs.stride_c, 1_I), p_coord_c);

    auto c_offset = [&](int half_m, int half_n) {
        return half_m * T::HALF_B_M * kargs.stride_c + half_n * T::HALF_B_N;
    };

    const bool full_tile =
        (row + T::B_M <= kargs.m) && (col + T::B_N <= kargs.n);

    if (full_tile) {
        using LT_C = opus::layout_load_traits<decltype(u_gc), T::VEC_C>;
        constexpr auto r_elem_c = LT_C::r_elem;
        constexpr opus::index_t acc_chunk =
            T::VEC_C * opus::vector_traits<D_ACC>::size();

        constexpr int HALF_TILE_ELEMS = T::HALF_B_M * T::HALF_B_N;
        constexpr int STORE_VEC = HALF_TILE_ELEMS / T::BLOCK_SIZE;
        static_assert(STORE_VEC * T::BLOCK_SIZE == HALF_TILE_ELEMS);

        constexpr int LDS_PAD = 8;
        constexpr int LDS_STRIDE = T::HALF_B_N + LDS_PAD;

        D_C* lds_ptr[2] = {
            reinterpret_cast<D_C*>(smem_a_bytes),
            reinterpret_cast<D_C*>(smem_b_bytes)
        };

        auto u_lds_c = opus::partition_layout_c<T::VEC_C>(mma,
            opus::make_tuple(opus::number<LDS_STRIDE>{}, 1_I), p_coord_c);
        auto offsets_lds = opus::layout_to_offsets<T::VEC_C>(u_lds_c);

        const int tid = opus::thread_id_x();
        const int rd_row = (tid * STORE_VEC) / T::HALF_B_N;
        const int rd_col = (tid * STORE_VEC) % T::HALF_B_N;
        const int lds_rd_off = rd_row * LDS_STRIDE + rd_col;
        const int gmem_v_off = rd_row * kargs.stride_c + rd_col;

        #pragma unroll
        for (int hm = 0; hm < 2; hm++) {
            opus::smem<D_C> s_c0 = opus::make_smem(lds_ptr[0]);
            opus::smem<D_C> s_c1 = opus::make_smem(lds_ptr[1]);

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
                if constexpr (std::is_same_v<D_C, D_ACC>) {
                    s_c0.template store<T::VEC_C>(chunk0, offsets_lds[i]);
                    s_c1.template store<T::VEC_C>(chunk1, offsets_lds[i]);
                } else {
                    s_c0.template store<T::VEC_C>(opus::cast<D_C>(chunk0), offsets_lds[i]);
                    s_c1.template store<T::VEC_C>(opus::cast<D_C>(chunk1), offsets_lds[i]);
                }
            }

            __builtin_amdgcn_s_barrier();

            auto coal0 = s_c0.template load<STORE_VEC>(lds_rd_off);
            auto coal1 = s_c1.template load<STORE_VEC>(lds_rd_off);
            g_c.template store<STORE_VEC>(coal0, gmem_v_off,
                c_offset(hm, 0), opus::number<7>{});
            g_c.template store<STORE_VEC>(coal1, gmem_v_off,
                c_offset(hm, 1), opus::number<7>{});

            if (hm == 0) __builtin_amdgcn_s_barrier();
        }
    } else {
        epilogue_store_c_if<T>(mma, g_c, kargs, v_c,
                               wave_id_m, wave_id_n, lane_id, row, col);
    }
}

// SplitK workspace store: sc0+nt (aux=3) write-through to bypass L1
// (cache-line stale-merge caused max_diff 35%->0.03% regression before 2026-06-01).
template<typename T, typename Mma, typename GC, typename Kargs, typename VC>
OPUS_D inline void epilogue_store_workspace_sc0nt(
    Mma& mma, GC& g_c, const Kargs& kargs,
    VC& v_c, int wave_id_m, int wave_id_n, int lane_id)
{
    using D_C = typename T::D_C;
    using D_ACC = typename T::D_ACC;

    auto p_coord_c = opus::make_tuple(wave_id_m, lane_id % mma.grpn_c,
                                      wave_id_n, lane_id / mma.grpn_c);
    auto u_gc = opus::partition_layout_c<T::VEC_C>(mma,
                    opus::make_tuple(kargs.stride_ws, 1_I), p_coord_c);

    auto ws_offset = [&](int half_m, int half_n) {
        return half_m * T::HALF_B_M * kargs.stride_ws + half_n * T::HALF_B_N;
    };

    if constexpr (std::is_same_v<D_C, D_ACC>) {
        opus::store<T::VEC_C>(g_c, v_c[0][0], u_gc, ws_offset(0, 0), opus::number<3>{});
        opus::store<T::VEC_C>(g_c, v_c[0][1], u_gc, ws_offset(0, 1), opus::number<3>{});
        opus::store<T::VEC_C>(g_c, v_c[1][0], u_gc, ws_offset(1, 0), opus::number<3>{});
        opus::store<T::VEC_C>(g_c, v_c[1][1], u_gc, ws_offset(1, 1), opus::number<3>{});
    } else {
        auto c00 = opus::cast<D_C>(v_c[0][0]);
        auto c01 = opus::cast<D_C>(v_c[0][1]);
        auto c10 = opus::cast<D_C>(v_c[1][0]);
        auto c11 = opus::cast<D_C>(v_c[1][1]);
        opus::store<T::VEC_C>(g_c, c00, u_gc, ws_offset(0, 0), opus::number<3>{});
        opus::store<T::VEC_C>(g_c, c01, u_gc, ws_offset(0, 1), opus::number<3>{});
        opus::store<T::VEC_C>(g_c, c10, u_gc, ws_offset(1, 0), opus::number<3>{});
        opus::store<T::VEC_C>(g_c, c11, u_gc, ws_offset(1, 1), opus::number<3>{});
    }
}

#endif // __HIP_DEVICE_COMPILE__
