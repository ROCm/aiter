// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <type_traits>
#include <opus/hip_minimal.hpp>
#include <opus/opus.hpp>
#include "../opus_gemm_utils.cuh"
#include "opus_gemm_traits_a8w8_blockscale_mxscale_bpreshuffle_m_repeat_gfx950.cuh"

#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx950__)
namespace opus_gemm_bpreshuffle_m_repeat_lds_layout {

// Read an A/B fragment without fixing its register allocation.
template<int Vec, int OFFSET_BEGIN, class Fragment, class Mem, class Offsets>
__device__ inline Fragment load_operand_fragment_staged(Mem& smem, const Offsets& offsets) {
    using Chunk = decltype(opus::load<Vec>(smem, offsets[OFFSET_BEGIN]));
    opus::array<Chunk, 2> values;
    opus::static_for<2>([&](auto i) {
        constexpr int chunk = decltype(i)::value;
        values[chunk] = opus::load<Vec>(smem, offsets[OFFSET_BEGIN + chunk]);
    });

    Fragment result;
    opus::set_slice(result, values[0], opus::number<0>{}, opus::number<16>{});
    opus::set_slice(result, values[1], opus::number<16>{}, opus::number<32>{});
    return result;
}

// A layouts: global memory -> LDS -> MFMA registers.
template<class T>
__device__ inline auto make_layout_ga_scale(int lane_id, int wave_id_m, int wave_id_n, int stride_a) {
    constexpr int threads_k = T::B_K / T::VEC_A;
    constexpr int threads_m_per_block = T::BLOCK_SIZE / threads_k;
    constexpr int threads_m_per_wave = T::WARP_SIZE / threads_k;

    constexpr auto ga_block_shape = opus::to_tuple(
        opus::seq<T::HALF_B_M / threads_m_per_block, T::T_N, threads_m_per_wave, T::T_M, threads_k,
                  T::VEC_A>{});

    constexpr auto ga_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}));

    return opus::make_layout<T::VEC_A>(
        ga_block_shape,
        opus::unfold_x_stride(
            ga_block_dim, ga_block_shape,
            opus::tuple{stride_a, 1_I}),
        opus::unfold_p_coord(
            ga_block_dim,
            opus::tuple{wave_id_n, lane_id / threads_k, wave_id_m, lane_id % threads_k}));
}

template<class T>
__device__ inline auto make_layout_sa_scale(int wave_id_m, int wave_id_n) {
    constexpr int num_waves = T::BLOCK_SIZE / T::WARP_SIZE;

    constexpr auto sa_block_shape = opus::to_tuple(
        opus::seq<T::smem_m_rep / num_waves, T::T_N, T::T_M, T::VEC_A>{});

    constexpr auto sa_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}));

    return opus::make_layout(
        sa_block_shape,
        opus::unfold_x_stride(
            sa_block_dim, sa_block_shape,
            opus::tuple{opus::number<T::smem_linear_wave + T::smem_padding>{}, 1_I}),
        opus::unfold_p_coord(
            sa_block_dim,
            opus::tuple{wave_id_n, wave_id_m}));
}

template<class T>
__device__ inline constexpr auto make_layout_ra_scale(int lane_id, int wave_id_m) {
    constexpr auto ra_block_shape = opus::to_tuple(
        opus::seq<T::E_M, T::T_N, T::T_M, T::A_ROWS_PER_WAVE, T::E_K, T::A_CHUNKS_PER_FRAGMENT,
                  T::WARP_SIZE / T::W_M, T::VEC_A>{});

    constexpr auto ra_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}, opus::y_dim{}, opus::p_dim{}, opus::y_dim{}));

    const int lane_id_m = lane_id % T::W_M;
    const int matrix_row = wave_id_m * T::W_M + lane_id_m;
    const int producer_n = matrix_row / (T::A_ROWS_PER_WAVE * T::T_M);
    const int producer_m = matrix_row % T::T_M;
    const int producer_lane_row = (matrix_row % (T::A_ROWS_PER_WAVE * T::T_M)) / T::T_M;

    return opus::make_layout<T::VEC_A>(
        ra_block_shape,
        opus::unfold_x_stride(
            ra_block_dim, ra_block_shape,
            opus::tuple{opus::number<T::smem_linear_wave + T::smem_padding>{}, 1_I}),
        opus::unfold_p_coord(
            ra_block_dim,
            opus::tuple{producer_n, producer_m, producer_lane_row, lane_id / T::W_M}));
}

// Preshuffled B layouts: global memory -> LDS -> MFMA registers.
template<class T>
__device__ inline constexpr auto make_layout_gb_scale(int lane_id, int wave_id_m, int wave_id_n, int stride_b) {
    constexpr int lanes_n = T::W_N;
    constexpr int lanes_k = T::WARP_SIZE / lanes_n;
    constexpr int k_per_wave_load = lanes_k * T::VEC_B;
    constexpr int n_repeats = T::HALF_B_N / (T::NUM_WAVES * lanes_n);
    constexpr int k_repeats = T::B_K / k_per_wave_load;
    if constexpr (T::SCALE_ON_READ) {
        constexpr auto gb_block_shape = opus::to_tuple(
            opus::seq<T::T_N, T::T_M, n_repeats, k_repeats, T::WARP_SIZE, T::VEC_B>{});

        constexpr auto gb_block_dim =
            opus::make_tuple(opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::y_dim{}),
                             opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::y_dim{}));

        return opus::make_layout<T::VEC_B>(
            gb_block_shape,
            opus::unfold_x_stride(gb_block_dim, gb_block_shape, opus::tuple{T::W_N * stride_b, 1_I}),
            opus::unfold_p_coord(gb_block_dim, opus::tuple{wave_id_n, wave_id_m, lane_id}));
    } else {
        constexpr auto gb_block_shape = opus::to_tuple(
            opus::seq<n_repeats, T::T_M, T::T_N, k_repeats, T::WARP_SIZE, T::VEC_B>{});

        constexpr auto gb_block_dim = opus::make_tuple(
            opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}),
            opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::y_dim{}));

        return opus::make_layout<T::VEC_B>(
            gb_block_shape,
            opus::unfold_x_stride(
                gb_block_dim, gb_block_shape,
                opus::tuple{T::W_N * stride_b, 1_I}),
            opus::unfold_p_coord(
                gb_block_dim,
                opus::tuple{wave_id_m, wave_id_n, lane_id}));
    }
}

template<class T>
__device__ inline constexpr auto make_layout_sb_scale(int wave_id_m, int wave_id_n) {
    if constexpr (T::SCALE_ON_READ) {
        constexpr int n_repeats = T::HALF_B_N / (T::NUM_WAVES * T::W_N);
        constexpr int k_per_wave_load = (T::WARP_SIZE / T::W_N) * T::VEC_B;
        constexpr int k_repeats = T::B_K / k_per_wave_load;

        constexpr auto sb_block_shape =
            opus::to_tuple(opus::seq<T::T_N, T::T_M, n_repeats, k_repeats, T::VEC_B>{});

        constexpr auto sb_block_dim = opus::make_tuple(
            opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::y_dim{}, opus::y_dim{}),
            opus::make_tuple(opus::y_dim{}));

        return opus::make_layout(
            sb_block_shape,
            opus::unfold_x_stride(
                sb_block_dim, sb_block_shape,
                opus::tuple{opus::number<T::smem_linear_wave + T::smem_padding>{}, 1_I}),
            opus::unfold_p_coord(sb_block_dim, opus::tuple{wave_id_n, wave_id_m}));
    } else {
        constexpr auto sb_block_shape = opus::to_tuple(
            opus::seq<T::smem_n_rep / T::NUM_WAVES, T::T_M, T::T_N, T::VEC_B>{});

        constexpr auto sb_block_dim = opus::make_tuple(
            opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}),
            opus::make_tuple(opus::y_dim{}));

        return opus::make_layout(
            sb_block_shape,
            opus::unfold_x_stride(
                sb_block_dim, sb_block_shape,
                opus::tuple{opus::number<T::smem_linear_wave + T::smem_padding>{}, 1_I}),
            opus::unfold_p_coord(
                sb_block_dim,
                opus::tuple{wave_id_m, wave_id_n}));
    }
}

template<class T>
__device__ inline constexpr auto make_layout_rb_scale(int lane_id, int wave_id_n) {
    if constexpr (T::SCALE_ON_READ) {
        constexpr int k_vectors = T::W_N * T::W_K / (T::WARP_SIZE * T::VEC_B);

        constexpr auto rb_block_shape = opus::to_tuple(
            opus::seq<T::E_N, T::T_N, T::E_K, k_vectors, T::WARP_SIZE, T::VEC_B>{});

        constexpr auto rb_block_dim = opus::make_tuple(
            opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::y_dim{}, opus::y_dim{}),
            opus::make_tuple(opus::p_dim{}, opus::y_dim{}));

        return opus::make_layout<T::VEC_B>(
            rb_block_shape,
            opus::unfold_x_stride(
                rb_block_dim, rb_block_shape,
                opus::tuple{opus::number<T::smem_linear_wave + T::smem_padding>{}, 1_I}),
            opus::unfold_p_coord(rb_block_dim, opus::tuple{wave_id_n, lane_id}));
    } else {
        constexpr int P = T::smem_linear_wave + T::smem_padding;
        constexpr int n_repeats = T::NUM_WAVES / T::T_N;
        constexpr int k_vectors = T::W_N * T::W_K / (T::WARP_SIZE * T::VEC_B);

        constexpr auto rb_block_shape = opus::to_tuple(
            opus::seq<T::E_N / n_repeats, n_repeats, T::T_N, k_vectors, T::E_K, T::WARP_SIZE, T::VEC_B>{});

        constexpr auto rb_block_dim = opus::make_tuple(
            opus::make_tuple(opus::y_dim{}),
            opus::make_tuple(opus::y_dim{}, opus::p_dim{}),
            opus::make_tuple(opus::y_dim{}),
            opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::y_dim{}));

        return opus::make_layout<T::VEC_B>(
            rb_block_shape,
            opus::unfold_x_stride(
                rb_block_dim, rb_block_shape,
                opus::tuple{opus::number<T::NUM_WAVES * k_vectors * P>{}, opus::number<P>{}, opus::number<T::NUM_WAVES * P>{}, 1_I}),
            opus::unfold_p_coord(
                rb_block_dim,
                opus::tuple{wave_id_n, lane_id}));
    }
}

// A-scale layouts: raw M vectors -> four-repeat packed LDS words -> registers.
template<class T, class Stride>
__device__ inline constexpr auto make_layout_gsfa_scale(int lane_id, int wave_id_m, int wave_id_n, Stride stride_sfa) {
    if constexpr (T::SCALE_ON_READ) {
        // Each thread copies 16 contiguous M scale bytes.
        constexpr auto block_shape = opus::to_tuple(
            opus::seq<T::T_N, T::T_M, T::SFA_K_COLUMNS_PER_WAVE, T::SFA_PASSES,
                      T::SFA_THREADS_PER_GROUP, T::VEC_SCALE_A>{});

        constexpr auto block_dim =
            opus::make_tuple(opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::p_dim{}),
                             opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::y_dim{}));

        return opus::make_layout<T::VEC_SCALE_A>(
            block_shape, opus::unfold_x_stride(block_dim, block_shape, opus::tuple{stride_sfa, 1_I}),
            opus::unfold_p_coord(block_dim,
                                 opus::tuple{wave_id_n, wave_id_m, lane_id / T::SFA_THREADS_PER_GROUP,
                                             lane_id % T::SFA_THREADS_PER_GROUP}));
    } else {
        constexpr auto gsfa_block_shape = opus::to_tuple(
            opus::seq<T::SFA_PASSES_PER_CACHE_PANEL, T::T_N, T::T_M, T::SFA_K_COLUMNS_PER_WAVE,
                      T::SFA_M_VECTORS, T::VEC_SCALE_A>{});

        constexpr auto gsfa_block_dim = opus::make_tuple(
            opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}, opus::p_dim{}),
            opus::make_tuple(opus::p_dim{}, opus::y_dim{}));

        const int lane_m = lane_id % T::SFA_M_VECTORS;
        const int pack = lane_m / (T::T_M * T::VEC_SF);
        const int consumer_wave_m = (lane_m / T::VEC_SF) % T::T_M;
        const int m_vector = (pack * T::VEC_SF + lane_m % T::VEC_SF) * T::T_M + consumer_wave_m;

        return opus::make_layout<T::VEC_SCALE_A>(
            gsfa_block_shape,
            opus::unfold_x_stride(
                gsfa_block_dim, gsfa_block_shape,
                opus::tuple{stride_sfa, 1_I}),
            opus::unfold_p_coord(
                gsfa_block_dim,
                opus::tuple{wave_id_n, wave_id_m, lane_id / T::SFA_M_VECTORS, m_vector}));
    }
}

template<class T>
__device__ inline constexpr auto make_layout_ssfa_scale(int lane_id, int wave_id_m, int wave_id_n) {
    if constexpr (T::SCALE_ON_READ) {
        return make_layout_gsfa_scale<T>(lane_id, wave_id_m, wave_id_n, opus::number<T::B_M>{});
    } else {
        constexpr auto ssfa_block_shape = opus::to_tuple(
            opus::seq<T::SFA_PASSES_PER_CACHE_PANEL, T::T_N, T::T_M, T::SFA_K_COLUMNS_PER_WAVE,
                      T::A_SCALE_PACKS, T::T_M, T::W_M, T::VEC_SF>{});

        constexpr auto ssfa_block_dim = opus::make_tuple(
            opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}, opus::p_dim{}),
            opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::p_dim{}, opus::y_dim{}));

        const int lane_m = lane_id % T::SFA_M_VECTORS;
        return opus::make_layout<T::VEC_SF>(
            ssfa_block_shape,
            opus::unfold_x_stride(
                ssfa_block_dim, ssfa_block_shape,
                opus::tuple{opus::number<T::SFA_PANEL_PITCH>{}, 1_I}),
            opus::unfold_p_coord(
                ssfa_block_dim,
                opus::tuple{wave_id_n, wave_id_m, lane_id / T::SFA_M_VECTORS,
                    lane_m / (T::T_M * T::VEC_SF), (lane_m / T::VEC_SF) % T::T_M, lane_m % T::VEC_SF}));
    }
}

template<class T>
__device__ inline constexpr auto make_layout_rsfa_scale(int lane_id, int wave_id_m) {
    constexpr auto rsfa_block_shape = opus::to_tuple(
        opus::seq<T::A_SCALE_PACKS, T::T_M, T::W_M, T::VEC_SF>{});

    constexpr auto rsfa_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}));

    return opus::make_layout<T::VEC_SF>(
        rsfa_block_shape,
        opus::unfold_x_stride(
            rsfa_block_dim, rsfa_block_shape,
            opus::tuple{opus::number<T::VEC_SF>{}, 1_I}),
        opus::unfold_p_coord(
            rsfa_block_dim,
            opus::tuple{wave_id_m, lane_id % T::W_M}));
}

// B-scale layouts: external K128 bytes -> packed LDS words -> registers.
template<class T>
__device__ inline constexpr auto make_layout_gsfb_scale(int lane_id, int wave_id, int stride_sfb) {
    constexpr auto gsfb_block_shape = opus::to_tuple(
        opus::seq<T::SCALE_N_HALVES, T::NUM_WAVES, T::WARP_SIZE, 1>{});

    constexpr auto gsfb_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::y_dim{}));

    return opus::make_layout<1>(
        gsfb_block_shape,
        opus::unfold_x_stride(
            gsfb_block_dim, gsfb_block_shape,
            opus::tuple{stride_sfb, 1_I}),
        opus::unfold_p_coord(
            gsfb_block_dim,
            opus::tuple{wave_id, lane_id}));
}

template<class T>
__device__ inline constexpr auto make_layout_ssfb_scale(int lane_id, int wave_id) {
    constexpr auto ssfb_block_shape = opus::to_tuple(opus::seq<T::NUM_WAVES, T::WARP_SIZE, 1>{});

    constexpr auto ssfb_block_dim = opus::make_tuple(
        opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::y_dim{}));

    // Offsets count packed uint words, one per K128 group.
    return opus::make_layout<1>(
        ssfb_block_shape,
        opus::unfold_x_stride(
            ssfb_block_dim, ssfb_block_shape,
            opus::tuple{1_I}),
        opus::unfold_p_coord(
            ssfb_block_dim,
            opus::tuple{wave_id, lane_id}));
}

__device__ inline constexpr auto make_layout_rsfb_scale() {
    constexpr auto rsfb_block_shape = opus::to_tuple(opus::seq<1>{});
    constexpr auto rsfb_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}));

    return opus::make_layout<1>(
        rsfb_block_shape,
        opus::unfold_x_stride(
            rsfb_block_dim, rsfb_block_shape,
            opus::tuple{1_I}),
        opus::unfold_p_coord(
            rsfb_block_dim,
            opus::tuple{}));
}

template<class T>
__device__ inline constexpr auto make_layout_rsfa_scale(int lane_id, int wave_id_m, int tile_k) {
    constexpr auto block_shape = opus::to_tuple(
        opus::seq<T::SCALE_PANEL, T::E_M, T::T_M, T::W_M, 1>{}); // one scale byte per M repeat

    constexpr auto block_dim =
        opus::make_tuple(opus::make_tuple(opus::p_dim{}),
                         opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}),
                         opus::make_tuple(opus::y_dim{}));

    return opus::make_layout<1>(
        block_shape,
        opus::unfold_x_stride(block_dim, block_shape,
                              opus::tuple{opus::number<T::B_M>{}, 1_I, 1_I}),
        opus::unfold_p_coord(block_dim, opus::tuple{tile_k, wave_id_m, lane_id % T::W_M}));
}

template<class T> __device__ inline constexpr auto make_layout_rsfb_scale(int tile_k) {
    constexpr auto block_shape = opus::to_tuple(opus::seq<T::SCALE_PANEL, 1>{});
    constexpr auto block_dim = opus::make_tuple(opus::make_tuple(opus::p_dim{}, opus::y_dim{}));

    return opus::make_layout<1>(block_shape,
                                opus::unfold_x_stride(block_dim, block_shape, opus::tuple{1_I}),
                                opus::unfold_p_coord(block_dim, opus::tuple{tile_k}));
}

} // namespace opus_gemm_bpreshuffle_m_repeat_lds_layout

// Two LDS stages feed an A fragment ring and reusable B fragments.
template<class Traits>
__device__ __forceinline__ void opus_gemm_bpreshuffle_m_repeat_lds_body(
    opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 kargs) {
    using namespace opus;
    using namespace opus_gemm_bpreshuffle_m_repeat_lds_layout;

    using T = opus::remove_cvref_t<Traits>;
    using D_A = opus::fp8_t;
    using D_B = opus::fp8_t;
    using D_C = opus::bf16_t;
    using D_ACC = opus::fp32_t;
    using D_SF = unsigned char;
    using D_SF_PACK = unsigned int;

    // Tile and thread coordinates.
    const int wave_id = __builtin_amdgcn_readfirstlane(thread_id_x() / T::WARP_SIZE);
    const int lane_id = thread_id_x() % T::WARP_SIZE;
    const int wave_id_m = wave_id % T::T_M;
    const int wave_id_n = wave_id / T::T_M;
    const int tile_gn = kargs.n / T::B_N;
    int block_m = block_id_y();
    int block_n = block_id_x();
    int tile_linear = block_m * tile_gn + block_n;
    if constexpr (T::XCD_ORDER) {
        const int tile_gm = (kargs.m + T::B_M - 1) / T::B_M;
        const int tile_count = tile_gm * tile_gn;
        const int partition = tile_linear % 8;
        const int remainder = tile_count % 8;
        tile_linear = partition * (tile_count / 8) + (partition < remainder ? partition : remainder) + tile_linear / 8;
        block_m = tile_linear / tile_gn;
        block_n = tile_linear % tile_gn;
    }
    if (kargs.n / T::B_N >= T::SWIZZLE_MIN_N_TILES && kargs.m > T::B_M) {
        // Keep the M tiles for a small panel of B close in the launch order.
        const int grid_m = 1 + (kargs.m - 1) / T::B_M;
        const int grid_n = kargs.n / T::B_N;
        const int linear = tile_linear;
        const int group_size = grid_m * T::SWIZZLE_GROUP_N;
        const int first_n = (linear / group_size) * T::SWIZZLE_GROUP_N;
        const int remaining_n = grid_n - first_n;
        const int actual_n = remaining_n < T::SWIZZLE_GROUP_N ? remaining_n : T::SWIZZLE_GROUP_N;
        const int within_group = linear % group_size;
        block_m = within_group / actual_n;
        block_n = first_n + within_group % actual_n;
    } else if (kargs.n <= T::SWIZZLE_MAX_N && kargs.m >= T::SWIZZLE_MIN_M) {
        const int grid_m = 1 + (kargs.m - 1) / T::B_M;
        const int grid_n = kargs.n / T::B_N;
        const int linear = tile_linear;
        const int group_size = T::SWIZZLE_GROUP_M * grid_n;
        const int first_m = (linear / group_size) * T::SWIZZLE_GROUP_M;
        const int remaining_m = grid_m - first_m;
        const int actual_m = remaining_m < T::SWIZZLE_GROUP_M ? remaining_m : T::SWIZZLE_GROUP_M;
        const int within_group = linear % group_size;
        block_m = first_m + within_group % actual_m;
        block_n = within_group / actual_m;
    }
    if constexpr (T::GROUPED_M > 0) {
        const int gm = (kargs.m + T::B_M - 1) / T::B_M;
        const int gn = kargs.n / T::B_N;
        const int linear = tile_linear;
        const int first = linear / (T::GROUPED_M * gn) * T::GROUPED_M;
        const int rows = gm - first < T::GROUPED_M ? gm - first : T::GROUPED_M;
        const int local = linear % (T::GROUPED_M * gn);
        block_m = first + local % rows;
        block_n = local / rows;
    }
    const int row = block_m * T::B_M;
    const int col = block_n * T::B_N;
    const int batch_id = block_id_z();
    const int loops = T::FIXED_K ? T::FIXED_K / T::B_K : kargs.k / T::B_K;

    // Matrix and scale global-memory views.
    auto g_a = make_gmem(
        reinterpret_cast<const D_A*>(kargs.ptr_a) + batch_id * kargs.stride_a_batch + row * kargs.stride_a,
        static_cast<unsigned>((kargs.m - row) * kargs.stride_a));
    auto g_b = make_gmem(reinterpret_cast<const D_B*>(kargs.ptr_b) +
        batch_id * kargs.stride_b_batch + col * kargs.stride_b);
    const int valid_rows = kargs.m - row < T::B_M ? kargs.m - row : T::B_M;
    auto g_c = make_gmem(
        reinterpret_cast<D_C*>(kargs.ptr_c) + static_cast<int64_t>(batch_id) * kargs.stride_c_batch +
            static_cast<int64_t>(row) * kargs.stride_c + col,
        static_cast<unsigned>((static_cast<int64_t>(valid_rows - 1) * kargs.stride_c + T::B_N) * sizeof(D_C)));
    auto g_sfa = make_gmem(reinterpret_cast<const D_SF*>(kargs.ptr_sfa) + batch_id * kargs.stride_sfa_batch + row);
    auto g_sfb = make_gmem(reinterpret_cast<const D_SF*>(kargs.ptr_sfb) +
        batch_id * kargs.stride_sfb_batch + (col / T::GROUP_N) * kargs.stride_sfb);

    // Matrix and scale layouts: global -> LDS -> registers.
    const auto u_ga = make_layout_ga_scale<T>(lane_id, wave_id_m, wave_id_n, kargs.stride_a);
    const auto u_sa = make_layout_sa_scale<T>(wave_id_m, wave_id_n);
    const auto u_ra = make_layout_ra_scale<T>(lane_id, wave_id_m);
    const auto u_gb = make_layout_gb_scale<T>(lane_id, wave_id_m, wave_id_n, kargs.stride_b);
    const auto u_sb = make_layout_sb_scale<T>(wave_id_m, wave_id_n);
    const auto u_rb = make_layout_rb_scale<T>(lane_id, wave_id_n);
    const auto u_gsfa = make_layout_gsfa_scale<T>(lane_id, wave_id_m, wave_id_n, kargs.stride_sfa);
    const auto u_ssfa = make_layout_ssfa_scale<T>(lane_id, wave_id_m, wave_id_n);
    const auto u_gsfb = make_layout_gsfb_scale<T>(lane_id, wave_id, kargs.stride_sfb);
    const auto u_ssfb = make_layout_ssfb_scale<T>(lane_id, wave_id);

    // Matrix and scale LDS; matrix storage is reused for the C epilogue.
    alignas(16) __shared__ char smem_matrix[T::LDS_BYTES];
    auto s_a = make_smem(reinterpret_cast<D_A*>(smem_matrix));
    auto s_b = make_smem(reinterpret_cast<D_B*>(smem_matrix + T::NUM_STAGES * T::A_STAGE));
    auto s_c = make_smem(reinterpret_cast<D_C*>(smem_matrix));
    auto s_sfa = make_smem(reinterpret_cast<D_SF*>(smem_matrix + T::MATRIX_LDS_BYTES));
    auto s_sfb = make_smem(reinterpret_cast<D_SF_PACK*>(smem_matrix + T::MATRIX_LDS_BYTES + T::SFA_BYTES));

    // MMA and register fragments.
    auto mma = make_tiled_mma<D_A, D_B, D_ACC>(
        seq<T::E_M, T::E_N, T::E_K>{},
        seq<T::T_M, T::T_N, T::T_K>{},
        seq<T::W_M, T::W_N, T::W_K>{},
        mfma_adaptor_swap_ab{});
    constexpr int ELEM_A = decltype(mma)::elem_a;
    constexpr int ELEM_B = decltype(mma)::elem_b;
    constexpr int ELEM_C = decltype(mma)::elem_c;

    using BaseMMA = typename decltype(mma)::MMA;
    using AVector = std::conditional_t<T::SCALE_ON_READ,
        typename decltype(mma)::vtype_a, vector_t<D_A, T::A_RING * ELEM_A>>;
    using AFragments = std::conditional_t<T::FRAGMENT_ARRAYS,
        array<typename BaseMMA::vtype_a, T::A_RING>, AVector>;
    using BFragments = std::conditional_t<T::FRAGMENT_ARRAYS,
        array<typename BaseMMA::vtype_b, T::E_N>, typename decltype(mma)::vtype_b>;
    using CFragments = std::conditional_t<T::FRAGMENT_ARRAYS,
        array<typename BaseMMA::vtype_c, T::E_M * T::E_N>, typename decltype(mma)::vtype_c>;
    AFragments v_a;
    BFragments v_b;
    CFragments v_c{};
    if constexpr (!T::FRAGMENT_ARRAYS)
        clear(v_c);

    // C output layout; BF16 fragments are first staged in LDS.
    const auto p_coord_c = opus::make_tuple(wave_id_m, lane_id % mma.grpn_c, wave_id_n, lane_id / mma.grpn_c);
    const auto u_gc = partition_layout_c<T::VEC_C>(
        mma, opus::to_tuple(opus::seq<T::C_LDS_ROW_STRIDE_ELEMS, 1>{}), p_coord_c);
    const auto gc_offsets = layout_to_offsets<T::VEC_C>(u_gc);
    array<D_SF_PACK, T::A_SCALE_PACKS> v_sfa{}, v_sfa_next{};
    array<D_SF_PACK, T::B_SCALE_PACKS> v_sfb{}, v_sfb_next{};

    // Matrix, scale panel and C staging offsets.
    auto ga_offset = [&](int tile_k) { return tile_k * number<T::B_K>{}; };
    auto gb_offset = [&](int tile_k) { return tile_k * number<T::B_K * T::W_N>{}; };
    auto sa_offset = [&](int stage) { return stage * T::A_STAGE; };
    auto sb_offset = [&](int stage) { return stage * T::B_STAGE; };
    auto gsfa_offset = [&](int panel_k_begin) { return panel_k_begin * kargs.stride_sfa; };
    auto gsfb_offset = [&](int panel_k_begin) { return panel_k_begin; };
    auto ssfa_offset = [&](int tile_k) {
        return tile_k * number<T::SCALE_ON_READ ? T::B_M : T::SFA_PANEL_PITCH>{};
    };

    // s_sfb addresses packed uint words, one word per K128 group.
    auto ssfb_offset = [&](int tile_k) { return tile_k; };
    auto c_offset = [&](int row_c, int col_c) { return row_c * T::C_LDS_ROW_STRIDE_ELEMS + col_c; };

    // A/B global memory -> LDS ping-pong stage.
    auto issue_matrix_prefetch = [&](auto stage_i, int tile_k) {
        constexpr int stage = decltype(stage_i)::value;
        async_load<T::VEC_A>(g_a, s_a.ptr, u_ga, u_sa + sa_offset(stage), ga_offset(tile_k),
            opus::number<0>{}, opus::number<T::CACHE_A>{});
        async_load<T::VEC_B>(g_b, s_b.ptr, u_gb, u_sb + sb_offset(stage), gb_offset(tile_k),
            opus::number<0>{}, opus::number<T::CACHE_B>{});
    };
    const int scale_k_groups = T::FIXED_K ? T::FIXED_K / T::GROUP_K : kargs.k / T::GROUP_K;
    const auto sfa_gmem_offsets = layout_to_offsets<T::VEC_SCALE_A>(u_gsfa);
    const auto sfa_smem_offsets = layout_to_offsets<T::SCALE_ON_READ ? T::VEC_SCALE_A : T::VEC_SF>(u_ssfa);
    const auto sfb_gmem_offsets = layout_to_offsets<1>(u_gsfb);
    const auto sfb_smem_offsets = layout_to_offsets<1>(u_ssfb);
    const auto sfa_row_offsets = layout_to_offsets<T::VEC_SCALE_A>(
        make_layout_gsfa_scale<T>(lane_id, wave_id_m, wave_id_n, 0));

    vector_t<D_SF, T::VEC_SCALE_A> sfa_panel_raw[T::SFA_PASSES_PER_CACHE_PANEL];

    // Load the four input M vectors for each quad before the byte transpose.
    auto load_sfa_panel = [&](int panel_k_begin) {
        if constexpr (T::SCALE_ON_READ) {
            const int local_k_group =
                wave_id * T::SFA_K_COLUMNS_PER_WAVE + lane_id / T::SFA_THREADS_PER_GROUP;
            const int k_group = panel_k_begin + local_k_group;
            static_for<T::SFA_PASSES>([&](auto pass_i) {
                constexpr int pass = decltype(pass_i)::value;
                const int local_row = sfa_row_offsets[pass];
                const int smem_offset = sfa_smem_offsets[pass] + panel_k_begin * T::B_M;
                if (local_row < T::B_M && smem_offset < T::SFA_BYTES && k_group < scale_k_groups) {
                    vector_t<D_SF, T::VEC_SCALE_A> raw;
                    if (row + local_row < kargs.m)
                        raw = load<T::VEC_SCALE_A>(g_sfa,
                                                   sfa_gmem_offsets[pass] + gsfa_offset(panel_k_begin));
                    else
                        static_for<T::VEC_SCALE_A>(
                            [&](auto byte_i) { raw[decltype(byte_i)::value] = 0x7f; });
                    store<T::VEC_SCALE_A>(s_sfa, raw, smem_offset);
                }
            });
        } else {
            const int lane_k = lane_id / T::SFA_M_VECTORS;
            const int lane_m = lane_id % T::SFA_M_VECTORS;
            const int pack = lane_m / (T::T_M * T::VEC_SF);
            const int m_repeat = pack * T::VEC_SF + lane_m % T::VEC_SF;
            const int consumer_wave_m = (lane_m / T::VEC_SF) % T::T_M;
            const int local_row = (m_repeat * T::T_M + consumer_wave_m) * T::VEC_SCALE_A;
            static_for<T::SFA_PASSES_PER_CACHE_PANEL>([&](auto pass_i) {
                constexpr int pass = decltype(pass_i)::value;
                const int local_k_group = pass * T::SFA_K_COLUMNS_PER_PASS
                    + wave_id * T::SFA_K_COLUMNS_PER_WAVE + lane_k;
                const int k_group = panel_k_begin + local_k_group;
                sfa_panel_raw[pass] = {};
                if (lane_k < T::SFA_K_COLUMNS_PER_WAVE && k_group < T::SCALE_PANEL
                    && k_group < scale_k_groups && m_repeat < T::E_M && row + local_row < kargs.m)
                    sfa_panel_raw[pass] = load<T::VEC_SCALE_A>(g_sfa, sfa_gmem_offsets[pass] + gsfa_offset(panel_k_begin));
            });
        }
    };

    // The 4wave 4x4 byte transpose packs repeats separated by T_M * W_M.
    auto publish_sfa_panel = [&](int panel_k_begin) {
        if constexpr (!T::SCALE_ON_READ) {
            const int lane_in_quad = lane_id % T::VEC_SF;
            const unsigned pair_byte_select = lane_in_quad % 2 ? 0x03070105u : 0x06020400u;
            const unsigned quad_byte_select = lane_in_quad / 2 ? 0x03020706u : 0x05040100u;
            auto transpose_sfa_word_4x4 = [&](D_SF_PACK lane_word) {
                const D_SF_PACK adjacent_lane = opus::mov_dpp(lane_word, opus::number<0xb1>{});
                const D_SF_PACK paired_bytes = __builtin_amdgcn_perm(adjacent_lane, lane_word, pair_byte_select);
                const D_SF_PACK opposite_pair = opus::mov_dpp(paired_bytes, opus::number<0x4e>{});
                return __builtin_amdgcn_perm(opposite_pair, paired_bytes, quad_byte_select);
            };
            static_for<T::SFA_PASSES_PER_CACHE_PANEL>([&](auto pass_i) {
                constexpr int pass = decltype(pass_i)::value;
                const int lane_k = lane_id / T::SFA_M_VECTORS;
                const int k_group = panel_k_begin + pass * T::SFA_K_COLUMNS_PER_PASS
                    + wave_id * T::SFA_K_COLUMNS_PER_WAVE + lane_k;
                if (lane_k < T::SFA_K_COLUMNS_PER_WAVE && k_group < T::SCALE_PANEL && k_group < scale_k_groups) {
                    const auto raw_words = __builtin_bit_cast(vector_t<D_SF_PACK, T::VEC_SCALE_A / T::VEC_SF>, sfa_panel_raw[pass]);
                    static_for<T::VEC_SCALE_A / T::VEC_SF>([&](auto word_i) {
                        constexpr int word = decltype(word_i)::value;
                        const D_SF_PACK packed_scales = transpose_sfa_word_4x4(raw_words[word]);
                        store<T::VEC_SF>(s_sfa, __builtin_bit_cast(vector_t<D_SF, T::VEC_SF>, packed_scales),
                            sfa_smem_offsets[pass] + ssfa_offset(panel_k_begin) + word * T::VEC_SF * T::VEC_SF);
                    });
                }
            });
        }
    };

    // Scale B global memory -> VGPR packing -> LDS.
    auto load_sfb_panel = [&](int panel_k_begin) {
        const int k_group = panel_k_begin + wave_id * T::WARP_SIZE + lane_id;
        if (k_group < scale_k_groups) {
            const unsigned lo = load<1>(g_sfb, sfb_gmem_offsets[0] + gsfb_offset(panel_k_begin))[0];
            unsigned hi = 0;
            if constexpr (T::SCALE_N_HALVES == 2)
                hi = load<1>(g_sfb, sfb_gmem_offsets[1] + gsfb_offset(panel_k_begin))[0];
            // One word per K128 group: low/high bytes select the two N128 halves.
            const vector_t<D_SF_PACK, 1> packed{lo | (hi << 8)};
            store<1>(s_sfb, packed, sfb_smem_offsets[0]);
        }
    };

    // Packed scale words feed the MFMA byte selectors directly.
    auto read_scales = [&](int tile_k, auto& scale_a, auto& scale_b) {
        if constexpr (T::SCALE_ON_READ) {
            const auto u_rsfa = make_layout_rsfa_scale<T>(lane_id, wave_id_m, tile_k);
            const auto rsfa_offsets = layout_to_offsets<1>(u_rsfa);
            const auto u_rsfb = make_layout_rsfb_scale<T>(tile_k);
            static_for<T::A_SCALE_PACKS>([&](auto p) { scale_a[decltype(p)::value] = 0; });
            static_for<T::E_M>([&](auto m_i) {
                constexpr int m_repeat = decltype(m_i)::value;
                const unsigned value = load<1>(s_sfa, rsfa_offsets[m_repeat])[0];
                scale_a[m_repeat / 4] |= value << ((m_repeat % 4) * 8);
            });
            scale_b[0] = load<1>(s_sfb, u_rsfb)[0];
        } else {
            const auto rsfa_offsets = layout_to_offsets<T::VEC_SF>(make_layout_rsfa_scale<T>(lane_id, wave_id_m));
            const auto rsfb_offsets = layout_to_offsets<1>(make_layout_rsfb_scale());
            static_for<T::A_SCALE_PACKS>([&](auto pack_i) {
                constexpr int pack = decltype(pack_i)::value;
                scale_a[pack] = __builtin_bit_cast(D_SF_PACK,
                    load<T::VEC_SF>(s_sfa, rsfa_offsets[pack] + ssfa_offset(tile_k)));
            });
            scale_b[0] = load<1>(s_sfb, rsfb_offsets[0] + ssfb_offset(tile_k))[0];
        }
    };

    // Matrix LDS -> registers using the configured fragment representation.
    auto load_a_fragment = [&](auto m_i, auto stage_i) {
        constexpr int m_repeat = decltype(m_i)::value;
        const auto offsets = layout_to_offsets<T::VEC_A>(u_ra + sa_offset(static_cast<int>(stage_i)));
        if constexpr (T::FRAGMENT_ARRAYS) {
            v_a[m_repeat % T::A_RING] = load_operand_fragment_staged<
                T::VEC_A, m_repeat * T::A_CHUNKS_PER_FRAGMENT, typename BaseMMA::vtype_a>(s_a, offsets);
        } else {
            static_for<T::A_CHUNKS_PER_FRAGMENT>([&](auto chunk_i) {
                constexpr int index = m_repeat * T::A_CHUNKS_PER_FRAGMENT + decltype(chunk_i)::value;
                constexpr int slot = (m_repeat % T::A_RING) * T::A_CHUNKS_PER_FRAGMENT + decltype(chunk_i)::value;
                set_slice(v_a, load<T::VEC_A>(s_a, offsets[index]),
                    number<slot * T::VEC_A>{}, number<(slot + 1) * T::VEC_A>{});
            });
        }
    };

    auto load_b_fragment = [&](auto n_i, auto stage_i) {
        constexpr int n_repeat = decltype(n_i)::value;
        const auto offsets = layout_to_offsets<T::VEC_B>(u_rb + sb_offset(decltype(stage_i)::value));
        if constexpr (T::FRAGMENT_ARRAYS) {
            v_b[n_repeat] = load_operand_fragment_staged<
                T::VEC_B, n_repeat * T::B_CHUNKS_PER_FRAGMENT, typename BaseMMA::vtype_b>(s_b, offsets);
        } else {
            static_for<T::B_CHUNKS_PER_FRAGMENT>([&](auto chunk_i) {
                constexpr int index = n_repeat * T::B_CHUNKS_PER_FRAGMENT + decltype(chunk_i)::value;
                set_slice(v_b, load<T::VEC_B>(s_b, offsets[index]),
                    number<index * T::VEC_B>{}, number<(index + 1) * T::VEC_B>{});
            });
        }
    };

    auto mma_scale_fragment = [&](auto m_i, auto n_i) {
        constexpr int mr = decltype(m_i)::value;
        constexpr int nr = decltype(n_i)::value;
        constexpr int ci = mr * T::E_N + nr;
        constexpr int sn = nr / (T::GROUP_N / (T::T_N * T::W_N));
        if constexpr (T::FRAGMENT_ARRAYS) {
            v_c[ci] = BaseMMA{}(v_a[mr % T::A_RING], v_b[nr], v_c[ci],
                static_cast<int>(v_sfa[mr / T::VEC_SF]), static_cast<int>(v_sfb[0]),
                number<mr % T::VEC_SF>{}, number<sn>{});
        } else {
            const auto a = slice(v_a, number<(mr % T::A_RING) * ELEM_A>{},
                number<(mr % T::A_RING + 1) * ELEM_A>{});
            const auto b = slice(v_b, number<nr * ELEM_B>{}, number<(nr + 1) * ELEM_B>{});
            auto c = slice(v_c, number<ci * ELEM_C>{}, number<(ci + 1) * ELEM_C>{});
            c = BaseMMA{}(a, b, c, static_cast<int>(v_sfa[mr / T::VEC_SF]),
                static_cast<int>(v_sfb[0]), number<mr % T::VEC_SF>{}, number<sn>{});
            set_slice(v_c, c, number<ci * ELEM_C>{}, number<(ci + 1) * ELEM_C>{});
        }
    };

    // Long fixed-K fragment tiles can overlap K+2 publication with the last
    // current MFMAs; defer waiting for the next LDS stage until its first read.
    constexpr bool defer_next_stage_wait = T::FIXED_K == 16384 &&
        T::FRAGMENT_ARRAYS && T::E_M == 4 && T::E_N == 8;
    auto advance_tile = [&](auto stage_i, int tile_k) {
        constexpr int stage = decltype(stage_i)::value;
        constexpr int next_stage = 1 - stage;
        read_scales(tile_k + 1, v_sfa_next, v_sfb_next);

        if constexpr (T::SCALE_ON_READ) {
            // Overlap the first M repeat's MFMAs with the pending next matrix tile.
            static_for<T::E_M>([&](auto m_i) {
                static_for<T::E_N>([&](auto n_i) {
                    mma_scale_fragment(m_i, n_i);
                    if constexpr (decltype(m_i)::value == T::E_M - 1)
                        load_b_fragment(n_i, number<next_stage>{});
                });
                if constexpr (decltype(m_i)::value == 0) {
                    __builtin_amdgcn_sched_barrier(0);
                    s_waitcnt_vmcnt(0_I);
                    s_waitcnt_lgkmcnt(0_I);
                    // Publish K+1 and retire all K readers before K+2 reuses the same LDS slot.
                    sync_threads();
                    if (tile_k + 2 < loops)
                        issue_matrix_prefetch(stage_i, tile_k + 2);
                    __builtin_amdgcn_sched_barrier(0);
                }
                // Replace A after its last N use and B after its last M use.
                load_a_fragment(m_i, number<next_stage>{});
            });
            // Complete next-tile reads; the next midpoint barrier protects LDS reuse.
            s_waitcnt_lgkmcnt(0_I);
        } else {
            static_for<T::E_M>([&](auto m_i) {
                constexpr int m = decltype(m_i)::value;
                s_waitcnt_lgkmcnt(0_I);
                if constexpr (m == T::E_M - T::A_RING) {
                    // All current A fragments are resident; release the LDS slot
                    // while the final register fragments still have MFMA consumers.
                    if constexpr (!defer_next_stage_wait)
                        s_waitcnt_vmcnt(0_I);
                    opus::sync_threads();
                    if (tile_k + 2 < loops)
                        issue_matrix_prefetch(stage_i, tile_k + 2);
                    __builtin_amdgcn_sched_barrier(0);
                }
                static_for<T::E_N>([&](auto n_i) {
                    mma_scale_fragment(m_i, n_i);
                    if constexpr (m == T::E_M - 1) {
                        if constexpr (defer_next_stage_wait && decltype(n_i)::value == 0) {
                            if (tile_k + 2 < loops)
                                s_waitcnt_vmcnt(number<T::MATRIX_VMEM_INSTRUCTIONS>{});
                            else
                                s_waitcnt_vmcnt(0_I);
                            opus::sync_threads();
                        }
                        load_b_fragment(n_i, number<next_stage>{});
                    }
                });
                // Reuse the consumed A slot for the M repeat one ring length ahead.
                if constexpr (m + T::A_RING < T::E_M)
                    load_a_fragment(number<m + T::A_RING>{}, stage_i);
            });
            static_for<T::A_RING>([&](auto m_i) { load_a_fragment(m_i, number<next_stage>{}); });
        }
        v_sfa = v_sfa_next;
        v_sfb = v_sfb_next;
    };

    // Final K tile and BF16 output helpers.
    auto finish_tile = [&](auto stage_i) {
        static_for<T::E_M>([&](auto m_i) {
            constexpr int m = decltype(m_i)::value;
            s_waitcnt_lgkmcnt(0_I);
            static_for<T::E_N>([&](auto n_i) { mma_scale_fragment(m_i, n_i); });
            if constexpr (m + T::A_RING < T::E_M)
                load_a_fragment(number<m + T::A_RING>{}, stage_i);
        });
    };

    auto stage_output_fragment = [&](auto c_i) {
        constexpr int c_index = decltype(c_i)::value;
        const auto c = [&] {
            if constexpr (T::FRAGMENT_ARRAYS)
                return v_c[c_index];
            else
                return slice(v_c, number<c_index * ELEM_C>{}, number<(c_index + 1) * ELEM_C>{});
        }();
        store<T::VEC_C>(s_c, cast<D_C>(c), gc_offsets[c_index]);
    };

    const int output_thread_id = wave_id * T::WARP_SIZE + lane_id;
    auto copy_output_bf16 = [&](int copy_index) {
        const int linear = output_thread_id * T::VEC_OUTPUT + copy_index * T::BLOCK_SIZE * T::VEC_OUTPUT;
        const int output_row = linear / T::B_N;
        const int output_col = linear % T::B_N;
        const auto value = load<T::VEC_OUTPUT>(s_c, c_offset(output_row, output_col));
        // The bounded SRD drops complete rows beyond valid_rows; BOUNDED_OUTPUT
        // omits the additional lane guard.
        if (T::BOUNDED_OUTPUT || row + output_row < kargs.m)
            store<T::VEC_OUTPUT>(g_c, value, output_row * kargs.stride_c + output_col, 0, opus::number<2>{});
    };

    // ===== Prologue =====
    // Publish scales and the first matrix tile, then prefetch K+1.
    issue_matrix_prefetch(number<0>{}, 0);
    __builtin_amdgcn_sched_barrier(0);

    if constexpr (T::SCALE_ON_READ) {
        static_for<T::SFA_K_PASSES>([&](auto i) {
            const int first_k = decltype(i)::value * T::SFA_K_PANEL;
            if (first_k < scale_k_groups) load_sfa_panel(first_k);
        });
    } else {
        load_sfa_panel(0);
    }
    load_sfb_panel(0);
    publish_sfa_panel(0);
    __builtin_amdgcn_sched_barrier(0);
    if (loops > 1) {
        issue_matrix_prefetch(number<1>{}, 1);
        __builtin_amdgcn_sched_barrier(0);
        s_waitcnt_vmcnt(number<T::MATRIX_VMEM_INSTRUCTIONS>{});
    } else {
        s_waitcnt_vmcnt(0_I);
    }
    s_waitcnt_lgkmcnt(0_I);
    opus::sync_threads();
    read_scales(0, v_sfa, v_sfb);
    static_for<T::A_RING>([&](auto m_i) { load_a_fragment(m_i, 0_I); });
    static_for<T::E_N>([&](auto n_i) { load_b_fragment(n_i, number<0>{}); });
    s_waitcnt_lgkmcnt(0_I);

    // ===== Main loop =====
    // Alternate LDS tiles while the A fragment ring rolls across M repeats.
    if constexpr (T::SCALE_ON_READ) {
        for (int tile_k = 0; tile_k + 1 < loops; tile_k += T::LOOP_UNROLL) {
            advance_tile(number<0>{}, tile_k);

            if (tile_k + 2 < loops)
                advance_tile(number<1>{}, tile_k + 1);
        }
    } else {
#pragma nounroll
        for (int tile_k = 0; tile_k + 1 < loops; tile_k += T::LOOP_UNROLL) {
            advance_tile(number<0>{}, tile_k);

            if (tile_k + 2 < loops)
                advance_tile(number<1>{}, tile_k + 1);
        }
    }

    // ===== Epilogue =====
    if constexpr (T::SCALE_ON_READ) {
        s_waitcnt_lgkmcnt(0_I);
        sync_threads();

        static_for<T::E_M>([&](auto m_i) {
            static_for<T::E_N>([&](auto n_i) {
                constexpr int c_index = decltype(m_i)::value * T::E_N + decltype(n_i)::value;
                mma_scale_fragment(m_i, n_i);
                stage_output_fragment(number<c_index>{});
            });
        });
    } else {
        if constexpr (T::FRAGMENT_ARRAYS || (T::FIXED_K == 0 && T::E_M >= 3)) {
            // One final MFMA site shortens live ranges when the final LDS stage
            // is selected at runtime.
            finish_tile((loops - 1) % 2);
        } else {
            // Fixed-K and small packed-fragment tiles retain their constant LDS offsets.
            if ((loops - 1) % 2 == 0)
                finish_tile(0_I);
            else
                finish_tile(1_I);
        }

        // ===== Output writeback =====
        // Stage BF16 fragments only after the final tile has released its LDS reads.
        s_waitcnt_lgkmcnt(0_I);
        opus::sync_threads();

        static_for<T::E_M * T::E_N>([&](auto c_i) { stage_output_fragment(c_i); });
    }

    // Write contiguous BF16 vectors, preserving the final partial M tile guard.
    s_waitcnt_lgkmcnt(0_I);
    opus::sync_threads();

    static_for<T::OUTPUT_PASSES>([&](auto copy_i) { copy_output_bf16(decltype(copy_i)::value); });
}

namespace opus_gemm_bpreshuffle_m_repeat_stage_layout {
using opus::operator""_I;

// The ring pipelines count DMA groups, rather than individual VMEM operations.
// Keep the grouped instruction behind the same view/layout API as opus::gmem;
// the ordinary gmem transport is not tracked by asyncmark on gfx950.
template<class D>
struct async_gmem : opus::gmem<D>
{
    using Base = opus::gmem<D>;
    using Base::Base;

    template<opus::index_t Vec, opus::index_t Immediate = 0, opus::index_t Cache = 0>
    __device__ __forceinline__ void _async_load(OPUS_LDS_ADDR void* destination,
                                                int offset,
                                                int scalar_offset       = 0,
                                                opus::number<Immediate> = {},
                                                opus::number<Cache>     = {})
    {
        using Value = typename Base::template vector_type<Vec>;
        __builtin_amdgcn_raw_ptr_buffer_load_async_lds(this->cached_rsrc,
                                                       destination,
                                                       sizeof(Value),
                                                       offset * sizeof(D),
                                                       scalar_offset * sizeof(D),
                                                       Immediate,
                                                       Cache);
    }

    template<opus::index_t Vec, opus::index_t Immediate = 0, opus::index_t Cache = 0>
    __device__ __forceinline__ void async_load(void* destination,
                                               int offset,
                                               int scalar_offset       = 0,
                                               opus::number<Immediate> = {},
                                               opus::number<Cache>     = {})
    {
        auto* target =
            reinterpret_cast<OPUS_LDS_ADDR void*>(reinterpret_cast<__UINTPTR_TYPE__>(destination));
        _async_load<Vec>(
            target, offset, scalar_offset, opus::number<Immediate>{}, opus::number<Cache>{});
    }

    template<opus::index_t Vec,
              class GlobalLayout,
              class SharedLayout,
              opus::index_t Immediate = 0,
              opus::index_t Cache     = 0,
              std::enable_if_t<opus::is_layout_v<GlobalLayout> && opus::is_layout_v<SharedLayout>,
                               bool>  = true>
    __device__ __forceinline__ void async_load(OPUS_LDS_ADDR void* destination,
                                               const GlobalLayout& global_layout,
                                               const SharedLayout& shared_layout,
                                               int scalar_offset       = 0,
                                               opus::number<Immediate> = {},
                                               opus::number<Cache>     = {})
    {
        using Load                = opus::layout_load_traits<GlobalLayout, Vec>;
        const auto global_offsets = opus::layout_to_offsets<Vec>(global_layout);
        const auto shared_offsets = opus::layout_to_offsets<Vec>(shared_layout);
        // Preserve the OPUS shared view's address space through layout offsets.
        // A generic-pointer round trip adds a null-address select at each DMA.
        auto* target = reinterpret_cast<OPUS_LDS_ADDR D*>(destination);
        opus::static_for<Load::r_elem.value>([&](auto i) {
            _async_load<Vec>(target + shared_offsets[i],
                             global_offsets[i],
                             scalar_offset,
                             opus::number<Immediate>{},
                             opus::number<Cache>{});
        });
    }
};

template<class D>
__device__ __forceinline__ auto make_async_gmem(const D* ptr, unsigned int bytes)
{
    return async_gmem<D>(ptr, bytes);
}

__device__ __forceinline__ void commit_async_loads() { __builtin_amdgcn_asyncmark(); }

template<int Pending>
__device__ __forceinline__ void wait_async_loads(opus::number<Pending>)
{
    __builtin_amdgcn_wait_asyncmark(Pending);
}

// Some scale-address policies require a VGPR offset even when all active rows
// happen to share a scalar address. Preserve that requirement at the transport.
template<class Layout>
__device__ __forceinline__ auto vectorize_layout_offsets(Layout layout)
{
#if defined(__AMDGCN__)
    for (int i = 0; i < Layout::num_issues; ++i)
    {
        asm volatile("" : "+v"(layout.offsets[i]));
    }
#endif
    return layout;
}

template<bool Contiguous>
__device__ inline constexpr int repeat_index(int wave_id, int repeat, int repeats, int waves)
{
    if constexpr (Contiguous)
        return wave_id * repeats + repeat;
    else
        return repeat * waves + wave_id;
}

using namespace opus;

template<class T, int Vec, class Copy>
__device__ inline constexpr auto make_layout_shared_copy(Copy copy, int wave_id, int base, int copies)
{
    constexpr int lane_stride = Vec < 4 ? 4 : Vec;
    const auto shared_block_shape =
        opus::make_tuple(copies, opus::number<T::BLOCK_SIZE / 64>{}, 64_I, opus::number<Vec>{});
    constexpr auto shared_block_dim =
        opus::make_tuple(opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::p_dim{}),
                         opus::make_tuple(opus::y_dim{}));
    return opus::make_layout<Vec>(
               shared_block_shape,
               opus::unfold_x_stride(shared_block_dim,
                                     shared_block_shape,
                                     opus::tuple{opus::number<lane_stride>{}, 1_I}),
               opus::unfold_p_coord(shared_block_dim, opus::tuple{int(copy), wave_id, 0_I})) +
           base;
}

// A: global copy, LDS destination, and MFMA operand.
template<class T, class Copy>
__device__ inline constexpr auto make_layout_ga_scale(Copy copy, int lane_id, int wave_id, int stride_a)
{
    constexpr int threads_k       = T::B_K / 16;
    constexpr int rows_per_wave   = 64 / threads_k;
    constexpr int waves           = T::BLOCK_SIZE / 64;
    constexpr int copies          = (T::B_M * T::B_K + T::BLOCK_SIZE * 16 - 1) / (T::BLOCK_SIZE * 16);
    constexpr auto ga_block_shape = opus::to_tuple(opus::seq<copies, waves, rows_per_wave, threads_k, 16>{});
    constexpr auto ga_block_dim =
        opus::make_tuple(opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::p_dim{}),
                         opus::make_tuple(opus::p_dim{}, opus::y_dim{}));
    const auto logical_coord =
        opus::tuple{int(copy), wave_id, lane_id / threads_k, lane_id % threads_k};
    // XOR permutes K-vector ownership using the row coordinate. Compute that
    // coordinate from the same tile decomposition, independently of its pitch.
    const auto row_layout = opus::make_layout<-1>(
        ga_block_shape,
        opus::unfold_x_stride(ga_block_dim, ga_block_shape, opus::tuple{1_I, 0_I}),
        opus::unfold_p_coord(ga_block_dim, logical_coord));
    const int row_swizzle = row_layout(0_I) % threads_k;
    return opus::make_layout<16>(
        ga_block_shape,
        opus::unfold_x_stride(ga_block_dim, ga_block_shape, opus::tuple{stride_a, 1_I}),
        opus::unfold_p_coord(
            ga_block_dim,
            opus::tuple{
                int(copy), wave_id, lane_id / threads_k, (lane_id % threads_k) ^ row_swizzle}));
}

template<class T, class Copy>
__device__ inline constexpr auto make_layout_sa_scale(Copy copy, int wave_id, int base)
{
    return make_layout_shared_copy<T, 16>(copy, wave_id, base, T::A_COPIES);
}

template<class T, bool VectorOffsets = false>
__device__ inline constexpr auto make_layout_ra_scale(int row, int lane_id, int k_slice, int base = 0)
{
    constexpr auto ra_block_shape = opus::to_tuple(opus::seq<T::B_M / 16, 16, T::B_K / 128, 2, 4, 16>{});
    constexpr auto ra_block_dim = opus::make_tuple(opus::make_tuple(opus::p_dim{}, opus::p_dim{}),
                                                   opus::make_tuple(opus::p_dim{}),
                                                   opus::make_tuple(opus::y_dim{}),
                                                   opus::make_tuple(opus::p_dim{}),
                                                   opus::make_tuple(opus::y_dim{}));
    const int swizzle           = row % (T::B_K / 16);
    // XOR flips independent K-slice/lane coordinates. A flipped half axis
    // starts at half one and walks backwards, preserving MFMA operand order.
    const int half_stride = swizzle & 4 ? -64 : 64;
    auto layout =
        opus::make_layout<16>(
            ra_block_shape,
            opus::unfold_x_stride(
                ra_block_dim,
                ra_block_shape,
                opus::tuple{opus::number<T::B_K>{}, 128_I, half_stride, 16_I, 1_I}),
            opus::unfold_p_coord(
                ra_block_dim,
                opus::tuple{
                    row / 16, row % 16, k_slice ^ (swizzle / 8), (lane_id / 16) ^ (swizzle % 4)})) +
        base + (swizzle & 4) * 16;
    if constexpr (VectorOffsets)
        return opus_gemm_bpreshuffle_m_repeat_stage_layout::vectorize_layout_offsets(
            layout);
    else
        return layout;
}

// B operand layouts.
template<class T, class Copy>
__device__ inline constexpr auto make_layout_gb_scale(Copy copy, int lane_id, int wave_id, int stride_b)
{
    constexpr int waves_per_group = T::B_K / 64;
    constexpr auto gb_block_shape =
        opus::to_tuple(opus::seq<T::B_N / 16, waves_per_group, 64, 16>{});
    constexpr auto gb_block_dim =
        opus::make_tuple(opus::make_tuple(opus::p_dim{}),
                         opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::y_dim{}));
    const int copy_wave = int(copy) * (T::BLOCK_SIZE / 64) + wave_id;
    return opus::make_layout<16>(
        gb_block_shape,
        opus::unfold_x_stride(gb_block_dim, gb_block_shape, opus::tuple{16 * stride_b, 1_I}),
        opus::unfold_p_coord(
            gb_block_dim,
            opus::tuple{copy_wave / waves_per_group, copy_wave % waves_per_group, lane_id}));
}

template<class T, class Copy>
__device__ inline constexpr auto make_layout_sb_scale(Copy copy, int wave_id, int base)
{
    return make_layout_shared_copy<T, 16>(copy, wave_id, base, T::B_COPIES);
}

template<class T>
__device__ inline constexpr auto
make_layout_rb_scale(int group_n, int lane_id, int k_slice, int stride_b, int base = 0)
{
    const auto rb_block_shape =
        opus::make_tuple(opus::number<T::B_N / 16>{}, stride_b / 128, 2_I, 64_I, 16_I);
    constexpr auto rb_block_dim = opus::make_tuple(
        opus::make_tuple(opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}, opus::p_dim{}, opus::y_dim{}));
    return opus::make_layout<16>(
               rb_block_shape,
               opus::unfold_x_stride(rb_block_dim, rb_block_shape, opus::tuple{16 * stride_b, 1_I}),
               opus::unfold_p_coord(rb_block_dim, opus::tuple{group_n, k_slice, lane_id})) +
           base;
}

// A scale: global copy, LDS destination, and MFMA scale.
template<class T, bool SimpleAddress = false, class Copy>
__device__ inline auto make_layout_gsfa_scale(Copy copy,
                                        int tid,
                                        int row,
                                        int shape_m,
                                        int stride_sfa,
                                        int first_column,
                                        int columns,
                                        int total_columns)
{
    const auto gsfa_block_shape =
        opus::make_tuple(total_columns, (shape_m + T::SCALE_COPY - 1) / T::SCALE_COPY, opus::number<T::SCALE_COPY>{});
    constexpr auto gsfa_block_dim = opus::make_tuple(
        opus::make_tuple(opus::p_dim{}), opus::make_tuple(opus::p_dim{}, opus::y_dim{}));
    const int copy_lane = int(copy) * T::BLOCK_SIZE + tid;
    const int local_row = (copy_lane % (T::B_M / T::SCALE_COPY)) * T::SCALE_COPY;
    const int column    = copy_lane / (T::B_M / T::SCALE_COPY);
    const bool valid    = column < columns && row + local_row < shape_m;
    int scale_column    = valid ? first_column + column : total_columns - 1;
    int scale_row       = valid ? row + local_row : shape_m;
    if constexpr (SimpleAddress)
    {
        const int bounded_column = column < columns ? column : columns - 1;
        scale_column             = first_column + bounded_column;
        scale_row                = row + local_row;
    }
    auto layout =
        opus::make_layout<T::SCALE_COPY>(
            gsfa_block_shape,
            opus::unfold_x_stride(gsfa_block_dim, gsfa_block_shape, opus::tuple{stride_sfa, 1_I}),
            opus::unfold_p_coord(gsfa_block_dim, opus::tuple{scale_column, scale_row / T::SCALE_COPY})) +
        scale_row % T::SCALE_COPY;
    return opus_gemm_bpreshuffle_m_repeat_stage_layout::vectorize_layout_offsets(layout);
}

template<class T, class Copy>
__device__ inline constexpr auto make_layout_ssfa_scale(Copy copy, int wave_id, int base)
{
    return make_layout_shared_copy<T, T::SCALE_COPY>(copy, wave_id, base, T::SA_COPIES);
}

template<class T>
__device__ inline constexpr auto
make_layout_rsfa_scale(int row, int k_slice, int block_m, int columns, int base = 0)
{
    constexpr int spacing       = 4 / T::SCALE_COPY;
    const auto rsfa_block_shape = opus::make_tuple(columns, block_m / 16, 16_I, 1_I);
    constexpr auto rsfa_block_dim =
        opus::make_tuple(opus::make_tuple(opus::p_dim{}),
                         opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::y_dim{}));
    return opus::make_layout<1>(
               rsfa_block_shape,
               opus::unfold_x_stride(rsfa_block_dim,
                                     rsfa_block_shape,
                                     opus::tuple{block_m * spacing, opus::number<spacing>{}}),
               opus::unfold_p_coord(rsfa_block_dim, opus::tuple{k_slice, row / 16, row % 16})) +
           base;
}

// B scale: global copy, LDS destination, and MFMA scale.
template<class T, class Copy>
__device__ inline constexpr auto make_layout_gsfb_scale(Copy copy,
                                                  int tid,
                                                  int column,
                                                  int stride_sfb,
                                                  int first_column,
                                                  int columns,
                                                  int rows,
                                                  int total_groups)
{
    const auto gsfb_block_shape   = opus::make_tuple(total_groups, stride_sfb, 1_I);
    constexpr auto gsfb_block_dim = opus::make_tuple(
        opus::make_tuple(opus::p_dim{}), opus::make_tuple(opus::p_dim{}, opus::y_dim{}));
    const unsigned copy_lane = int(copy) * T::BLOCK_SIZE + tid;
    const unsigned group_n   = copy_lane / columns < rows ? copy_lane / columns : rows - 1;
    return opus::make_layout<1>(
        gsfb_block_shape,
        opus::unfold_x_stride(gsfb_block_dim, gsfb_block_shape, opus::tuple{stride_sfb, 1_I}),
        opus::unfold_p_coord(
            gsfb_block_dim,
            opus::tuple{column / 128 + group_n, first_column + copy_lane % columns}));
}

template<class T, class Copy>
__device__ inline constexpr auto make_layout_ssfb_scale(Copy copy, int wave_id, int base)
{
    return make_layout_shared_copy<T, 1>(copy, wave_id, base, T::SB_COPIES);
}

template<class T>
__device__ inline constexpr auto
make_layout_rsfb_scale(int group_n, int k_slice, int columns, int base = 0)
{
    const auto rsfb_block_shape   = opus::make_tuple(opus::number<T::SB_ROWS>{}, columns, 1_I);
    constexpr auto rsfb_block_dim = opus::make_tuple(
        opus::make_tuple(opus::p_dim{}), opus::make_tuple(opus::p_dim{}, opus::y_dim{}));
    return opus::make_layout<1>(
               rsfb_block_shape,
               opus::unfold_x_stride(
                   rsfb_block_dim, rsfb_block_shape, opus::tuple{columns * 4, 4_I}),
               opus::unfold_p_coord(rsfb_block_dim, opus::tuple{group_n, k_slice})) +
           base;
}

// C registers use the same wave/lane partition as the tiled MFMA.
template<class T, class MMA>
__device__ inline auto make_layout_rc_scale(MMA& mma, int lane_id, int wave_id_m,
                                          int wave_id_n, int stride_c)
{
    const auto pc =
        opus::make_tuple(wave_id_m, lane_id % mma.grpn_c, wave_id_n, lane_id / mma.grpn_c);
    return partition_layout_c<4>(mma, opus::make_tuple(stride_c, 1_I), pc);
}

// C register and output layouts.
template<class T>
__device__ inline constexpr auto make_layout_rc_scale(int row, int column, int lane_id, int stride_c)
{
    if constexpr (T::NORMAL)
    {
        constexpr auto rc_block_shape = opus::make_tuple(
            opus::number<T::B_M / 16>{}, 4_I, 4_I, opus::number<T::B_N / 16>{}, 16_I);
        constexpr auto rc_block_dim =
            opus::make_tuple(opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::y_dim{}),
                             opus::make_tuple(opus::p_dim{}, opus::p_dim{}));
        return opus::make_layout<1>(
            rc_block_shape,
            opus::unfold_x_stride(rc_block_dim, rc_block_shape, opus::tuple{stride_c, 1_I}),
            opus::unfold_p_coord(rc_block_dim,
                                 opus::tuple{row / 16, lane_id / 16, column / 16, lane_id % 16}));
    }
    else
    {
        constexpr auto rc_block_shape = opus::make_tuple(
            opus::number<T::B_M / 16>{}, 16_I, opus::number<T::B_N / 16>{}, 4_I, 4_I);
        constexpr auto rc_block_dim =
            opus::make_tuple(opus::make_tuple(opus::p_dim{}, opus::p_dim{}),
                             opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::y_dim{}));
        return opus::make_layout<4>(
            rc_block_shape,
            opus::unfold_x_stride(rc_block_dim, rc_block_shape, opus::tuple{stride_c, 1_I}),
            opus::unfold_p_coord(rc_block_dim,
                                 opus::tuple{row / 16, lane_id % 16, column / 16, lane_id / 16}));
    }
}

template<int Vec, int Rows, int Columns>
__device__ inline constexpr auto make_layout_matrix_vector(int row, int column, int stride)
{
    constexpr auto matrix_block_shape =
        opus::make_tuple(opus::number<Rows>{}, opus::number<Columns / Vec>{}, opus::number<Vec>{});
    constexpr auto matrix_block_dim = opus::make_tuple(
        opus::make_tuple(opus::p_dim{}), opus::make_tuple(opus::p_dim{}, opus::y_dim{}));
    return opus::make_layout<Vec>(
        matrix_block_shape,
        opus::unfold_x_stride(matrix_block_dim, matrix_block_shape, opus::tuple{stride, 1_I}),
        opus::unfold_p_coord(matrix_block_dim, opus::tuple{row, column / Vec}));
}

template<class T, int Rows = T::B_M>
__device__ inline constexpr auto make_layout_sc(int row, int column, int stride)
{
    return make_layout_matrix_vector<8, Rows, T::B_N>(row, column, stride);
}

template<class T>
__device__ inline constexpr auto make_layout_gc(int row, int column, int stride)
{
    return make_layout_matrix_vector<8, T::B_M, T::B_N>(row, column, stride);
}

// Split-K fragment slots.
template<class T>
__device__ inline constexpr auto make_layout_partial(int fragment, int tid, int split)
{
    constexpr auto partial_block_shape =
        opus::to_tuple(opus::seq<T::SPLIT_K, (T::B_M * T::B_N) / (T::BLOCK_SIZE * 4), T::BLOCK_SIZE / 64, 64, 4>{});
    constexpr auto partial_block_dim = opus::make_tuple(opus::make_tuple(
        opus::p_dim{}, opus::p_dim{}, opus::p_dim{}, opus::p_dim{}, opus::y_dim{}));
    return opus::make_layout<4>(
        partial_block_shape,
        opus::unfold_x_stride(partial_block_dim, partial_block_shape, opus::tuple{1_I}),
        opus::unfold_p_coord(partial_block_dim, opus::tuple{split, fragment, tid / 64, tid % 64}));
}

} // namespace opus_gemm_bpreshuffle_m_repeat_stage_layout

namespace opus {
template<class D>
struct is_gmem<opus_gemm_bpreshuffle_m_repeat_stage_layout::async_gmem<D>> : true_type {};
} // namespace opus


// Multi-stage LDS transport and publication are shared with direct-B prefetch.
template<class Traits>
__device__ __forceinline__ void opus_gemm_bpreshuffle_m_repeat_stage_body(
    opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 kargs,
    opus::bf16_t* final_output,
    unsigned* counters)
{
    using namespace opus;
    using namespace opus_gemm_bpreshuffle_m_repeat_stage_layout;
    namespace layout = opus_gemm_bpreshuffle_m_repeat_stage_layout;
    using T          = opus::remove_cvref_t<Traits>;
    using D_A        = opus::fp8_t;
    using D_B        = opus::fp8_t;
    using D_C        = opus::bf16_t;
    using D_ACC      = opus::fp32_t;
    using D_SF       = unsigned char;
    // Prepare tile views, operand rings, scales, and accumulators.
    if constexpr (T::OPERAND_POLICY)
    {
        if constexpr (T::FIXED_N > 0)
            kargs.n = T::FIXED_N;
        kargs.k              = T::TOTAL_K;
        kargs.stride_a       = T::TOTAL_K;
        kargs.stride_b       = T::TOTAL_K;
        kargs.stride_c       = kargs.n;
        kargs.stride_sfa     = kargs.m;
        kargs.stride_sfb     = T::TOTAL_K / 128;
    }
    const int m          = kargs.m;
    const int stride_c   = kargs.stride_c;
    const unsigned gm    = (kargs.m + T::B_M - 1) / T::B_M;
    const unsigned gn    = kargs.n / T::B_N;
    const unsigned tiles = gm * gn;
    unsigned tile = 0, split = 0;
    int row, col;
    constexpr bool m_first = [] {
        if constexpr (requires { T::M_FIRST; })
            return T::M_FIRST;
        else
            return false;
    }();
    if constexpr (m_first)
    {
        row = block_id_x() * T::B_M;
        col = block_id_y() * T::B_N;
    }
    else if constexpr (T::SPLIT_K == 1 && !T::XCD_ORDER)
    {
        row = block_id_y() * T::B_M;
        col = block_id_x() * T::B_N;
    }
    else
    {
        const unsigned pid = block_id_x();
        tile               = pid / (8 * T::SPLIT_K) * 8 + pid % 8;
        split              = (pid / 8) % T::SPLIT_K;
        if (tile >= tiles)
            return;
        row = (tile % gm) * T::B_M;
        col = (tile / gm) * T::B_N;
    }
    const int tid       = thread_id_x();
    const int lane_id   = tid % 64;
    const int lane_id_m = lane_id % 16;
    if constexpr (T::OPERAND_POLICY)
        __builtin_assume(static_cast<unsigned>(tid) < T::BLOCK_SIZE);
    const int wave_id = __builtin_amdgcn_readfirstlane(tid / 64);
    if constexpr (T::OPERAND_POLICY)
        __builtin_assume(static_cast<unsigned>(wave_id) < T::T_M * T::T_N);
    const int wave_id_m = wave_id / T::T_N;
    const int wave_id_n = wave_id % T::T_N;
    auto mfrag          = [&](auto mi) {
        return repeat_index<T::CONTIGUOUS>(wave_id_m, int(mi), T::E_M, T::T_M);
    };
    auto nfrag = [&](auto ni) {
        return repeat_index<T::CONTIGUOUS>(wave_id_n, int(ni), T::E_N, T::T_N);
    };
    const int valid_rows = m - row < T::B_M ? m - row : T::B_M;
    // Matrix global-memory views.
    auto g_a   = make_async_gmem(reinterpret_cast<const D_A*>(kargs.ptr_a) + row * kargs.stride_a,
                               valid_rows * kargs.stride_a);
    auto g_b   = make_async_gmem(reinterpret_cast<const D_B*>(kargs.ptr_b) + col * kargs.stride_b,
                               T::B_N * kargs.stride_b);
    auto g_sfa = make_async_gmem(reinterpret_cast<const D_SF*>(kargs.ptr_sfa),
                                 (T::TOTAL_K / 128 - 1) * kargs.stride_sfa + m);
    auto g_sfb = make_async_gmem(reinterpret_cast<const D_SF*>(kargs.ptr_sfb),
                                 (kargs.n / 128 - 1) * kargs.stride_sfb + T::TOTAL_K / 128);
    alignas(16) __shared__ D_SF smem[T::LDS_BYTES];
    auto s_ab = make_smem(reinterpret_cast<D_A*>(smem));
    auto s_sf = make_smem(smem);
    // MMA and register fragments.
    auto mma = [] {
        if constexpr (T::OPERAND_POLICY)
            return make_tiled_mma<D_A, D_B, D_ACC>(
                seq<1, 1, 1>{}, seq<1, 1, 1>{}, seq<16, 16, 128>{},
                std::conditional_t<T::NORMAL, mfma_adaptor, mfma_adaptor_swap_ab>{});
        else
            return make_tiled_mma<D_A, D_B, D_ACC>(
                seq<T::E_M, T::E_N, 1>{}, seq<T::T_M, T::T_N, 1>{},
                seq<16, 16, 128>{}, mfma_adaptor_swap_ab{});
    }();
    using BaseMMA    = typename decltype(mma)::MMA;
    using AFragments = array<array<typename BaseMMA::vtype_a, T::E_M>, T::KS>;
    using BFragments = array<array<typename BaseMMA::vtype_b, T::E_N>, T::KS>;
    struct Stage
    {
        AFragments a;
        BFragments b;
        array<array<unsigned, T::E_M>, T::KS> sa;
        array<array<unsigned, T::OPERAND_POLICY ? T::E_N : 1>, T::KS> sb;
    };
    array<typename BaseMMA::vtype_c, T::E_M * T::E_N> c{};
    array<BFragments, T::B_SETS> b_regs;
    auto prefetch_b = [&](int t, auto phase) {
        if constexpr (T::B_DIRECT)
        {
            static_for<T::KS>([&](auto ki) {
                constexpr int ks = decltype(ki)::value;
                static_for<T::E_N>([&](auto ni) {
                    constexpr int n = decltype(ni)::value;
                    auto& frag      = b_regs[decltype(phase)::value][ks][n];
                    frag            = load<16>(
                        g_b,
                        layout::make_layout_rb_scale<T>(
                            nfrag(ni), lane_id, ks, T::TOTAL_K, (split * T::K + t * T::B_K) * 16),
                        0,
                        number<T::B_CACHE>{});
                });
            });
            __builtin_amdgcn_sched_barrier(0);
        }
    };
    auto prefetch_scales = [&](int base, int kt) {
        static_for<T::SA_COPIES>([&](auto ii) {
            const auto u_gsfa =
                layout::make_layout_gsfa_scale<T, T::SIMPLE_SCALE>(
                    ii,
                    tid,
                    row,
                    m,
                    kargs.stride_sfa,
                    split * T::SC + kt * T::KS,
                    T::SCALE_COLS,
                    T::TOTAL_K / 128);
            const auto u_ssfa = layout::make_layout_ssfa_scale<T>(ii, wave_id, base + T::OFF_SA);
            async_load<T::SCALE_COPY>(g_sfa, s_sf.ptr, u_gsfa, u_ssfa);
        });
        static_for<T::SB_COPIES>([&](auto ii) {
            const auto u_gsfb = layout::make_layout_gsfb_scale<T>(ii,
                                                                        tid,
                                                                        col,
                                                                        kargs.stride_sfb,
                                                                        split * T::SC + kt * T::KS,
                                                                        T::SCALE_COLS,
                                                                        T::SB_ROWS,
                                                                        kargs.n / 128);
            const auto u_ssfb = layout::make_layout_ssfb_scale<T>(ii, wave_id, base + T::OFF_SB);
            async_load<1>(g_sfb, s_sf.ptr, u_gsfb, u_ssfb);
        });
    };
    auto prefetch = [&](int slot, int kt) {
        static_for<T::A_COPIES>([&](auto ii) {
            const auto u_ga = layout::make_layout_ga_scale<T>(
                ii, lane_id, wave_id, kargs.stride_a);
            const auto u_sa = layout::make_layout_sa_scale<T>(ii, wave_id, slot * T::STAGE_BYTES);
            async_load<16>(g_a, s_ab.ptr, u_ga, u_sa, split * T::K + kt * T::B_K);
        });
        static_for<T::B_COPIES>([&](auto ii) {
            const auto u_gb = layout::make_layout_gb_scale<T>(
                ii, lane_id, wave_id, kargs.stride_b);
            const auto u_sb = layout::make_layout_sb_scale<T>(ii, wave_id, slot * T::STAGE_BYTES + T::A_BYTES);
            async_load<16>(g_b,
                           s_ab.ptr,
                           u_gb,
                           u_sb,
                           (split * T::K + kt * T::B_K) * 16,
                           0_I,
                           number<T::B_CACHE>{});
        });
        if constexpr (T::PER_STAGE)
            prefetch_scales(slot * T::STAGE_BYTES, kt);
        commit_async_loads();
        if constexpr (!T::OPERAND_POLICY)
            __builtin_amdgcn_sched_barrier(0);
    };
    // Read all operands before issuing a later stage's DMA. The LLVM backend
    // must see those reads before that DMA to avoid an unnecessary vmcnt(0).
    auto read_stage = [&](int slot, int t, auto phase) {
        Stage f;
        auto& v_a   = f.a;
        auto& v_b   = f.b;
        auto& v_sfa = f.sa;
        auto& v_sfb = f.sb;
        static_for<T::KS>([&](auto ki) {
            constexpr int ks = decltype(ki)::value;
            if constexpr (!T::OPERAND_POLICY)
                v_sfb[ks][0] = load<1>(
                    s_sf, layout::make_layout_rsfb_scale<T>(0, t * T::KS + ks, T::SC, T::OFF_SB))[0];
            else
            {
                static_for<T::E_N>([&](auto ni) {
                    constexpr int n   = decltype(ni)::value;
                    const unsigned bg = nfrag(ni) * 16 / 128;
                    v_sfb[ks][n] =
                        load<1>(s_sf,
                                layout::make_layout_rsfb_scale<T>(
                                    bg,
                                    (T::PER_STAGE ? 0 : t * T::KS) + ks,
                                    T::SCALE_COLS,
                                    (T::PER_STAGE ? slot * T::STAGE_BYTES : 0) + T::OFF_SB))[0];
                });
            }
            // All M repeats differ by whole 16-row panels. Keep their static
            // displacement outside the dynamic address, including the swizzle.
            const int r0     = mfrag(0_I) * 16 + lane_id_m;
            const auto u_ra0 = layout::make_layout_ra_scale<T, T::ADDRESS == 2>(
                r0, lane_id, ks, slot * T::STAGE_BYTES);
            static_for<T::E_M>([&](auto mi) {
                constexpr int m          = decltype(mi)::value;
                const int r              = mfrag(mi) * 16 + lane_id_m;
                constexpr unsigned delta = m * (T::CONTIGUOUS ? 1 : T::T_M) * 16;
                if constexpr (T::ADDRESS > 0 && delta % (T::B_K / 16) == 0)
                {
                    v_a[ks][m] = load<16>(s_ab, u_ra0 + number<delta * T::B_K>{});
                }
                else
                {
                    v_a[ks][m] = load<16>(s_ab,
                                          layout::make_layout_ra_scale<T>(
                                              r, lane_id, ks, slot * T::STAGE_BYTES));
                }
                v_sfa[ks][m] =
                    load<1>(s_sf,
                            layout::make_layout_rsfa_scale<T>(
                                r,
                                (T::PER_STAGE ? 0 : t * T::KS) + ks,
                                T::B_M,
                                T::SCALE_COLS,
                                (T::PER_STAGE ? slot * T::STAGE_BYTES : 0) + T::OFF_SA))[0];
            });
            static_for<T::E_N>([&](auto ni) {
                constexpr int n = decltype(ni)::value;
                if constexpr (T::B_DIRECT)
                    v_b[ks][n] = b_regs[decltype(phase)::value][ks][n];
                else
                {
                    const int ng = nfrag(ni);
                    v_b[ks][n] =
                        load<16>(s_ab,
                                 layout::make_layout_rb_scale<T>(
                                     ng, lane_id, ks, T::B_K, slot * T::STAGE_BYTES + T::A_BYTES));
                }
            });
        });
        return f;
    };
    auto compute = [&](const Stage& f) {
        static_for<T::KS>([&](auto ki) {
            constexpr int ks = decltype(ki)::value;
            static_for<T::E_M>([&](auto mi) {
                constexpr int m = decltype(mi)::value;
                static_for<T::E_N>([&](auto ni) {
                    constexpr int n   = decltype(ni)::value;
                    c[m * T::E_N + n] = BaseMMA{}(f.a[ks][m],
                                                  f.b[ks][n],
                                                  c[m * T::E_N + n],
                                                  int(f.sa[ks][m]),
                                                  int(f.sb[ks][T::OPERAND_POLICY ? n : 0]),
                                                  0_I,
                                                  0_I);
                });
            });
        });
    };
    // Share transport/layouts while retaining the resident-stage issue schedule.
    if constexpr (!T::OPERAND_POLICY)
    {
        auto step = [&](int slot, int t, auto pending, auto refill) {
            constexpr int wait = decltype(pending)::value;
            s_waitcnt_vmcnt(number<(wait > 63 ? 63 : wait)>{});
            s_waitcnt_lgkmcnt(0_I);
            sync_threads();
            array<array<typename BaseMMA::vtype_a, T::E_M>, T::KS> v_a;
            array<array<typename BaseMMA::vtype_b, T::E_N>, T::KS> v_b;
            array<array<unsigned, T::E_M>, T::KS> v_sfa;
            array<unsigned, T::KS> v_sfb;
            static_for<T::KS>([&](auto ki) {
                constexpr int ks = decltype(ki)::value;
                v_sfb[ks] =
                    load<1>(s_sf, layout::make_layout_rsfb_scale<T>(0, t * T::KS + ks, T::SC, T::OFF_SB))[0];
                static_for<T::E_M>([&](auto mi) {
                    constexpr int m = decltype(mi)::value;
                    const int r     = (m * T::T_M + wave_id_m) * 16 + lane_id_m;
                    v_a[ks][m]      = load<16>(
                        s_ab,
                        layout::make_layout_ra_scale<T>(r, lane_id, ks, slot * T::STAGE_BYTES));
                    v_sfa[ks][m] = load<1>(s_sf,
                                           layout::make_layout_rsfa_scale<T>(
                                               r, t * T::KS + ks, T::B_M, T::SC, T::OFF_SA))[0];
                });
                static_for<T::E_N>([&](auto ni) {
                    constexpr int n = decltype(ni)::value;
                    const int ng    = n * T::T_N + wave_id_n;
                    v_b[ks][n] =
                        load<16>(s_ab,
                                 layout::make_layout_rb_scale<T>(
                                     ng, lane_id, ks, T::B_K, slot * T::STAGE_BYTES + T::A_BYTES));
                });
            });
            __builtin_amdgcn_sched_barrier(0);
            if constexpr (decltype(refill)::value)
                prefetch((slot + T::INITIAL) % T::STAGES, t + T::INITIAL);
            s_waitcnt_lgkmcnt(0_I);
            static_for<T::KS>([&](auto ki) {
                constexpr int ks = decltype(ki)::value;
                static_for<T::E_M>([&](auto mi) {
                    constexpr int m = decltype(mi)::value;
                    static_for<T::E_N>([&](auto ni) {
                        constexpr int n   = decltype(ni)::value;
                        c[m * T::E_N + n] = BaseMMA{}(v_a[ks][m],
                                                      v_b[ks][n],
                                                      c[m * T::E_N + n],
                                                      int(v_sfa[ks][m]),
                                                      int(v_sfb[ks]),
                                                      0_I,
                                                      0_I);
                    });
                });
            });
            __builtin_amdgcn_sched_barrier(0);
        };

        // ===== Prologue =====
        // Fill the first data stage and publish the resident scale panel.
        prefetch(0, 0);
        static_for<T::SA_COPIES>([&](auto ii) {
            const auto u_gsfa = layout::make_layout_gsfa_scale<T>(
                ii, tid, row, m, kargs.stride_sfa, split * T::SC, T::SC, T::TOTAL_K / 128);
            const auto u_ssfa = layout::make_layout_ssfa_scale<T>(ii, wave_id, T::OFF_SA);
            async_load<T::SCALE_COPY>(g_sfa, s_sf.ptr, u_gsfa, u_ssfa);
        });
        const auto u_gsfb = layout::make_layout_gsfb_scale<T>(
            0_I, tid, col, kargs.stride_sfb, split * T::SC, T::SC, 1, kargs.n / 128);
        const auto u_ssfb = layout::make_layout_ssfb_scale<T>(0_I, wave_id, T::OFF_SB);
        async_load<1>(g_sfb, s_sf.ptr, u_gsfb, u_ssfb);
        static_for<T::INITIAL - 1>([&](auto ii) {
            constexpr int i = decltype(ii)::value + 1;
            prefetch(i, i);
        });

        // ===== Main loop =====
        constexpr int steady = T::PANELS - T::INITIAL;
        if constexpr (T::ROLLED_LOOP)
        {
#pragma clang loop unroll(disable)
            for (int t = 0; t < steady; ++t)
            {
                step(t % T::STAGES, t, number<(T::INITIAL - 1) * T::MATRIX_ISSUES>{}, 1_I);
            }
        }
        else
        {
            static_for<steady>([&](auto ti) {
                constexpr int t = decltype(ti)::value;
                step(t % T::STAGES, t, number<T::wait(t)>{}, 1_I);
            });
        }

        // ===== Epilogue =====
        // Drain the remaining resident panels without issuing another data stage.
        static_for<T::INITIAL>([&](auto ti) {
            constexpr int j = decltype(ti)::value;
            constexpr int t = steady + j;
            if constexpr (T::ROLLED_LOOP)
                step(t % T::STAGES, t, number<(T::INITIAL - 1 - j) * T::MATRIX_ISSUES>{}, 0_I);
            else
                step(t % T::STAGES, t, number<T::wait(t)>{}, 0_I);
        });

    }
    else
    {
        auto step = [&](int t, auto phase, auto pending, auto refill, auto refill_b) {
            if constexpr (!T::OPERAND_POLICY)
            {
                constexpr int wait = decltype(pending)::value;
                s_waitcnt_vmcnt(number<(wait > 63 ? 63 : wait)>{});
                s_waitcnt_lgkmcnt(0_I);
            }
            else if constexpr (T::ASYNC_GROUPS)
                wait_async_loads(pending);
            else
            {
                constexpr int wait =
                    decltype(pending)::value * (T::MATRIX_ISSUES + T::SA_COPIES + T::SB_COPIES);
                s_waitcnt_vmcnt(number<(wait > 63 ? 63 : wait)>{});
            }
            sync_threads();
            if constexpr (T::B_DIRECT && decltype(refill_b)::value)
                prefetch_b(t + T::B_AHEAD, number<(decltype(phase)::value + T::B_AHEAD) % T::B_SETS>{});
            auto f = read_stage(t % T::STAGES, t, phase);
            __builtin_amdgcn_sched_barrier(0);
            if constexpr (decltype(refill)::value)
            {
                const int slot = T::OPERAND_POLICY ? t : t % T::STAGES;
                prefetch((slot + T::INITIAL) % T::STAGES, t + T::INITIAL);
                if constexpr (T::OPERAND_POLICY)
                    __builtin_amdgcn_sched_barrier(0);
            }
            if constexpr (!T::OPERAND_POLICY)
                s_waitcnt_lgkmcnt(0_I);
            compute(f);
            if constexpr (!T::OPERAND_POLICY)
                __builtin_amdgcn_sched_barrier(0);
        };

        // ===== Prologue =====
        // Fill the register queue and initial LDS stages.
        static_for<T::B_DIRECT ? T::B_AHEAD : 0>([&](auto ti) {
            prefetch_b(decltype(ti)::value, number<decltype(ti)::value % T::B_SETS>{});
        });
        if constexpr (!T::PER_STAGE)
            prefetch_scales(0, 0);
        static_for<T::INITIAL>([&](auto si) { prefetch(decltype(si)::value, decltype(si)::value); });

        // ===== Main loop =====
        constexpr int steady = T::PANELS - T::INITIAL;
#pragma clang loop unroll(disable)
        for (int i = 0; i < steady / T::UNROLL; ++i)
        {
            static_for<T::UNROLL>([&](auto pi) {
                step(i * T::UNROLL + decltype(pi)::value, pi, number<T::INITIAL - 1>{}, 1_I, 1_I);
            });
        }
        static_for<steady % T::UNROLL>([&](auto ii) {
            constexpr int t = steady - steady % T::UNROLL + decltype(ii)::value;
            step(t, number<t % T::B_SETS>{}, number<T::INITIAL - 1>{}, 1_I, 1_I);
        });

        // ===== Epilogue =====
        // Drain the initial LDS queue with decreasing pending counts.
        static_for<T::INITIAL>([&](auto ji) {
            constexpr int j = decltype(ji)::value;
            constexpr int t = steady + j;
            step(t,
                 number<t % T::B_SETS>{},
                 number<T::INITIAL - 1 - j>{},
                 0_I,
                 number<(t + T::B_AHEAD < T::PANELS)>{});
        });

    }

    // ===== Write back =====
    // The direct and fused split-K paths use the same C layouts.
    auto store_output = [&]() {
        if constexpr (!T::OPERAND_POLICY)
        {
            const auto uc = layout::make_layout_rc_scale<T>(mma, lane_id, wave_id_m, wave_id_n, stride_c);
            const auto offsets = layout_to_offsets<4>(uc);
            auto g_c           = make_gmem(final_output + static_cast<int64_t>(row) * stride_c + col,
                                 ((valid_rows - 1) * stride_c + T::B_N) * sizeof(D_C));
            static_for<T::E_M * T::E_N>([&](auto fi) {
                constexpr int f = decltype(fi)::value;
                store<4>(g_c, cast<D_C>(c[f]), offsets[f], 0, number<2>{});
            });
        }
        else
        {
            auto g_c = make_gmem(final_output + static_cast<int64_t>(row) * stride_c + col,
                                 ((valid_rows - 1) * stride_c + T::B_N) * sizeof(D_C));
            if constexpr (T::OUTPUT < 2)
            {
                static_for<T::E_M * T::E_N>([&](auto fi) {
                    constexpr int f  = decltype(fi)::value;
                    constexpr int mi = f / T::E_N;
                    constexpr int ni = f % T::E_N;
                    auto values      = cast<D_C>(c[f]);
                    const auto u_rc  = layout::make_layout_rc_scale<T>(
                        mfrag(number<mi>{}) * 16, nfrag(number<ni>{}) * 16, lane_id, stride_c);
                    store<T::NORMAL ? 1 : 4>(g_c, values, u_rc, 0, 0_I);
                });
            }
            else
            {
                constexpr int pitch = T::B_N + 8;
                auto staged         = make_smem(reinterpret_cast<D_C*>(smem));
                sync_threads();
                static_for<T::E_M * T::E_N>([&](auto fi) {
                    constexpr int f  = decltype(fi)::value;
                    constexpr int mi = f / T::E_N;
                    constexpr int ni = f % T::E_N;
                    auto values      = cast<D_C>(c[f]);
                    const auto u_rc  = layout::make_layout_rc_scale<T>(
                        mfrag(number<mi>{}) * 16, nfrag(number<ni>{}) * 16, lane_id, pitch);
                    store<T::NORMAL ? 1 : 4>(staged, values, u_rc);
                });
                s_waitcnt_lgkmcnt(0_I);
                sync_threads();
                constexpr int chunks = T::B_N / 8;
                static_for<(T::B_M * chunks + T::BLOCK_SIZE - 1) / T::BLOCK_SIZE>([&](auto ii) {
                    const int index = decltype(ii)::value * T::BLOCK_SIZE + tid;
                    const int r     = index / chunks;
                    const int col   = (index % chunks) * 8;
                    if (index < T::B_M * chunks)
                    {
                        const auto u_sc =
                            layout::make_layout_sc<T>(r, col, pitch);
                        const auto u_gc =
                            layout::make_layout_gc<T>(r, col, stride_c);
                        store<8>(
                            g_c, load<8>(staged, u_sc), u_gc, 0, number<(T::OUTPUT == 2 ? 2 : 0)>{});
                    }
                });
            }
        }
    };
    if constexpr (T::SPLIT_K == 1)
    {
        store_output();
    }
    else
    {
        // Continuous fragment slots keep both the producer and final gather
        // coalesced, including partial M tiles. Only final C is bounds-masked.
        constexpr int tile_elements = T::B_M * T::B_N;
        auto g_partials             = make_gmem(reinterpret_cast<D_ACC*>(kargs.ptr_c) +
                                        static_cast<int64_t>(tile) * T::SPLIT_K * tile_elements,
                                    T::SPLIT_K * tile_elements * sizeof(D_ACC));
        static_for<T::E_M * T::E_N>([&](auto fi) {
            constexpr int f = decltype(fi)::value;
            store<4>(g_partials,
                     c[f],
                     layout::make_layout_partial<T>(
                         f, thread_id_x(), split),
                     0,
                     0_I);
        });
        // Local-L2 mode follows the gfx950 XCD-grouped publication protocol:
        // every wave drains its stores, all waves arrive, then the counter updates.
        s_waitcnt_vmcnt(0_I);
        asm volatile("" ::: "memory");
        sync_threads();
        __shared__ unsigned arrived;
        if (thread_id_x() == 0)
        {
            arrived = __hip_atomic_fetch_add(counters + tile, 1u, __ATOMIC_ACQ_REL, 3);
        }
        s_waitcnt_lgkmcnt(0_I);
        sync_threads();
        if (arrived == T::SPLIT_K - 1)
        {
            if constexpr (!T::OPERAND_POLICY)
            {
                const auto uc = layout::make_layout_rc_scale<T>(mma, lane_id, wave_id_m, wave_id_n, stride_c);
                const auto offsets   = layout_to_offsets<4>(uc);
                const int valid_rows = m - row < T::B_M ? m - row : T::B_M;
                auto g_c = make_gmem(final_output + static_cast<int64_t>(row) * stride_c + col,
                                     ((valid_rows - 1) * stride_c + T::B_N) * sizeof(D_C));
                static_for<T::E_M * T::E_N>([&](auto fi) {
                    constexpr int f = decltype(fi)::value;
                    typename BaseMMA::vtype_c sum{};
                    static_for<T::SPLIT_K>([&](auto si) {
                        constexpr int sk = decltype(si)::value;
                        sum += load<4>(
                            g_partials,
                            layout::make_layout_partial<T>(
                                f, thread_id_x(), sk),
                            0,
                            number<17>{});
                    });
                    store<4>(g_c, cast<D_C>(sum), offsets[f], 0, number<2>{});
                });
            }
            else
            {
                static_for<T::E_M * T::E_N>([&](auto fi) {
                    constexpr int f = decltype(fi)::value;
                    typename BaseMMA::vtype_c sum{};
                    static_for<T::SPLIT_K>([&](auto si) {
                        constexpr int sk = decltype(si)::value;
                        sum += load<4>(
                            g_partials,
                            layout::make_layout_partial<T>(
                                f, thread_id_x(), sk),
                            0,
                            number<17>{});
                    });
                    c[f] = sum;
                });
                store_output();
            }
            if (thread_id_x() == 0)
            {
                __hip_atomic_store(counters + tile, 0u, __ATOMIC_RELAXED, 3);
            }
        }
    }
}

// Dispatch the LDS schedule at compile time, reusing the ping-pong scale policy.
template<class Traits>
__device__ __forceinline__ void opus_gemm_bpreshuffle_m_repeat_body(
    opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 args,
    opus::bf16_t* final_output,
    unsigned* counters) {
    if constexpr (Traits::MULTI_STAGE)
        opus_gemm_bpreshuffle_m_repeat_stage_body<Traits>(args, final_output, counters);
    else
        opus_gemm_bpreshuffle_m_repeat_lds_body<Traits>(args);
}

template<class Traits>
__device__ __forceinline__ void opus_gemm_bpreshuffle_m_repeat_body(
    opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 args) {
    if constexpr (Traits::MULTI_STAGE)
        opus_gemm_bpreshuffle_m_repeat_stage_body<Traits>(
            args, reinterpret_cast<opus::bf16_t*>(args.ptr_c), nullptr);
    else
        opus_gemm_bpreshuffle_m_repeat_lds_body<Traits>(args);
}

#endif // gfx950

template<class Traits>
__global__ __launch_bounds__(Traits::BLOCK_SIZE, Traits::MIN_WGS_PER_CU)
void opus_gemm_bpreshuffle_m_repeat_kernel(
    opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 args) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx950__)
    opus_gemm_bpreshuffle_m_repeat_body<Traits>(args);
#endif
}

// Staged configurations retain their direct and split-K argument layout.
template<class Traits>
__global__ __launch_bounds__(Traits::BLOCK_SIZE, Traits::WG_PER_CU) void
opus_gemm_bpreshuffle_m_repeat_kernel(
    opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 args,
    opus::bf16_t* final_output,
    unsigned* counters) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx950__)
    if constexpr (Traits::SPLIT_K == 1)
        opus_gemm_bpreshuffle_m_repeat_body<Traits>(
            args, reinterpret_cast<opus::bf16_t*>(args.ptr_c), nullptr);
    else
        opus_gemm_bpreshuffle_m_repeat_body<Traits>(args, final_output, counters);
#endif
}
