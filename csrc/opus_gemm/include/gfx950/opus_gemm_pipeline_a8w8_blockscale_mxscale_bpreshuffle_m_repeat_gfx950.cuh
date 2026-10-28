// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <opus/hip_minimal.hpp>
#include <opus/opus.hpp>
#include "../opus_gemm_utils.cuh"
#include "opus_gemm_traits_a8w8_blockscale_mxscale_bpreshuffle_m_repeat_gfx950.cuh"

#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx950__)
namespace opus_gemm_bpreshuffle_pingpong_lds_layout {

// Read an A/B fragment directly from LDS into its assigned AGPRs.
template<int Vec, int OFFSET_BEGIN, class Reg, class Fragment, class Mem, class Offsets>
__device__ inline void load_operand_fragment_pinned(Reg reg, Fragment& fragment, Mem& smem, const Offsets& offsets) {
    using Chunk = decltype(opus::load<Vec>(smem, offsets[OFFSET_BEGIN]));
    constexpr int dwords_per_chunk = sizeof(Chunk) / sizeof(opus::u32_t);
    constexpr int physical_reg = decltype(reg)::value;
    static_assert(physical_reg >= 0 && physical_reg < 128);
    opus::array<Chunk, 2> values;

    opus::static_for<2>([&](auto i) {
        constexpr int chunk = decltype(i)::value;
        [[clang::amdgpu_pin_agpr(physical_reg + chunk * dwords_per_chunk)]]
        values[chunk] = opus::load<Vec>(smem, offsets[OFFSET_BEGIN + chunk]);
    });

    Fragment result;
    opus::set_slice(result, values[0], opus::number<0>{}, opus::number<16>{});
    opus::set_slice(result, values[1], opus::number<16>{}, opus::number<32>{});
    [[clang::amdgpu_pin_agpr(physical_reg)]] fragment = result;
}

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

    constexpr auto ga_block_shape = opus::make_tuple(
        opus::number<T::HALF_B_M / threads_m_per_block>{},
        opus::number<T::T_N>{},
        opus::number<threads_m_per_wave>{},
        opus::number<T::T_M>{},
        opus::number<threads_k>{},
        opus::number<T::VEC_A>{});

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

    constexpr auto sa_block_shape = opus::make_tuple(
        opus::number<T::smem_m_rep / num_waves>{},
        opus::number<T::T_N>{},
        opus::number<T::T_M>{},
        opus::number<T::VEC_A>{});

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
    constexpr auto ra_block_shape = opus::make_tuple(
        opus::number<T::E_M>{},
        opus::number<T::T_N>{},
        opus::number<T::T_M>{},
        opus::number<T::A_ROWS_PER_WAVE>{},
        opus::number<T::E_K>{},
        opus::number<T::A_CHUNKS_PER_FRAGMENT>{},
        opus::number<T::WARP_SIZE / T::W_M>{},
        opus::number<T::VEC_A>{});

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

    constexpr auto gb_block_shape = opus::make_tuple(
        opus::number<n_repeats>{},
        opus::number<T::T_M>{},
        opus::number<T::T_N>{},
        opus::number<k_repeats>{},
        opus::number<T::WARP_SIZE>{},
        opus::number<T::VEC_B>{});

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

template<class T>
__device__ inline constexpr auto make_layout_sb_scale(int wave_id_m, int wave_id_n) {
    constexpr auto sb_block_shape = opus::make_tuple(
        opus::number<T::smem_n_rep / T::NUM_WAVES>{},
        opus::number<T::T_M>{},
        opus::number<T::T_N>{},
        opus::number<T::VEC_B>{});

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

template<class T>
__device__ inline constexpr auto make_layout_rb_scale(int lane_id, int wave_id_n) {
    constexpr int P = T::smem_linear_wave + T::smem_padding;
    constexpr int n_repeats = T::NUM_WAVES / T::T_N;
    constexpr int k_vectors = T::W_N * T::W_K / (T::WARP_SIZE * T::VEC_B);

    constexpr auto rb_block_shape = opus::make_tuple(
        opus::number<T::E_N / n_repeats>{},
        opus::number<n_repeats>{},
        opus::number<T::T_N>{},
        opus::number<k_vectors>{},
        opus::number<T::E_K>{},
        opus::number<T::WARP_SIZE>{},
        opus::number<T::VEC_B>{});

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

// A-scale layouts: raw M vectors -> four-repeat packed LDS words -> registers.
template<class T>
__device__ inline constexpr auto make_layout_gsfa_scale(int lane_id, int wave_id_m, int wave_id_n, int stride_sfa) {
    constexpr auto gsfa_block_shape = opus::make_tuple(
        opus::number<T::SFA_PASSES_PER_CACHE_PANEL>{},
        opus::number<T::T_N>{},
        opus::number<T::T_M>{},
        opus::number<T::SFA_K_COLUMNS_PER_WAVE>{},
        opus::number<T::SFA_M_VECTORS>{},
        opus::number<T::VEC_SCALE_A>{});

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

template<class T>
__device__ inline constexpr auto make_layout_ssfa_scale(int lane_id, int wave_id_m, int wave_id_n) {
    constexpr auto ssfa_block_shape = opus::make_tuple(
        opus::number<T::SFA_PASSES_PER_CACHE_PANEL>{},
        opus::number<T::T_N>{},
        opus::number<T::T_M>{},
        opus::number<T::SFA_K_COLUMNS_PER_WAVE>{},
        opus::number<T::A_SCALE_PACKS>{},
        opus::number<T::T_M>{},
        opus::number<T::W_M>{},
        opus::number<T::VEC_SF>{});

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

template<class T>
__device__ inline constexpr auto make_layout_rsfa_scale(int lane_id, int wave_id_m) {
    constexpr auto rsfa_block_shape = opus::make_tuple(
        opus::number<T::A_SCALE_PACKS>{},
        opus::number<T::T_M>{},
        opus::number<T::W_M>{},
        opus::number<T::VEC_SF>{});

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
    constexpr auto gsfb_block_shape = opus::make_tuple(
        opus::number<T::SCALE_N_HALVES>{},
        opus::number<T::NUM_WAVES>{},
        opus::number<T::WARP_SIZE>{},
        1_I);

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
    constexpr auto ssfb_block_shape = opus::make_tuple(
        opus::number<T::NUM_WAVES>{},
        opus::number<T::WARP_SIZE>{},
        1_I);

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
    constexpr auto rsfb_block_shape = opus::make_tuple(
        1_I);
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

} // namespace opus_gemm_bpreshuffle_pingpong_lds_layout

// Two LDS stages feed an A fragment ring and reusable B fragments.
template<class Traits>
__device__ __forceinline__ void opus_gemm_bpreshuffle_pingpong_lds_body(
    opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 kargs) {
    using namespace opus;
    using namespace opus_gemm_bpreshuffle_pingpong_lds_layout;

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
    const auto u_rsfa = make_layout_rsfa_scale<T>(lane_id, wave_id_m);
    const auto u_gsfb = make_layout_gsfb_scale<T>(lane_id, wave_id, kargs.stride_sfb);
    const auto u_ssfb = make_layout_ssfb_scale<T>(lane_id, wave_id);
    const auto u_rsfb = make_layout_rsfb_scale();

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
    using AFragments = std::conditional_t<T::FRAGMENT_ARRAYS,
        array<typename BaseMMA::vtype_a, T::A_RING>, vector_t<D_A, T::A_RING * ELEM_A>>;
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
        mma, opus::make_tuple(number<T::C_LDS_ROW_STRIDE_ELEMS>{}, 1_I), p_coord_c);
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
    auto ssfa_offset = [&](int tile_k) { return tile_k * number<T::SFA_PANEL_PITCH>{}; };

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
    const auto sfa_smem_offsets = layout_to_offsets<T::VEC_SF>(u_ssfa);
    const auto sfb_gmem_offsets = layout_to_offsets<1>(u_gsfb);
    const auto sfb_smem_offsets = layout_to_offsets<1>(u_ssfb);
    const auto rsfa_offsets = layout_to_offsets<T::VEC_SF>(u_rsfa);
    const auto rsfb_offsets = layout_to_offsets<1>(u_rsfb);

    vector_t<D_SF, T::VEC_SCALE_A> sfa_panel_raw[T::SFA_PASSES_PER_CACHE_PANEL];

    // Load the four input M vectors for each quad before the byte transpose.
    auto load_sfa_panel = [&](int panel_k_begin) {
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
    };

    // The 4wave 4x4 byte transpose packs repeats separated by T_M * W_M.
    auto publish_sfa_panel = [&](int panel_k_begin) {
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
        static_for<T::A_SCALE_PACKS>([&](auto pack_i) {
            constexpr int pack = decltype(pack_i)::value;
            scale_a[pack] = __builtin_bit_cast(D_SF_PACK,
                load<T::VEC_SF>(s_sfa, rsfa_offsets[pack] + ssfa_offset(tile_k)));
        });
        scale_b[0] = load<1>(s_sfb, rsfb_offsets[0] + ssfb_offset(tile_k))[0];
    };

    // Matrix LDS -> registers, pinning A/B only when the corresponding trait is enabled.
    auto load_a_fragment = [&](auto m_i, auto stage_i) {
        if constexpr (T::FRAGMENT_ARRAYS) {
            constexpr int m_repeat = decltype(m_i)::value;
            const auto offsets = layout_to_offsets<T::VEC_A>(u_ra + sa_offset(static_cast<int>(stage_i)));
            if constexpr (T::PIN_A) {
                load_operand_fragment_pinned<T::VEC_A, m_repeat * T::A_CHUNKS_PER_FRAGMENT>(number<(m_repeat % T::A_RING) * 8>{}, v_a[m_repeat % T::A_RING], s_a, offsets);
            } else {
                v_a[m_repeat % T::A_RING] = load_operand_fragment_staged<T::VEC_A, m_repeat * T::A_CHUNKS_PER_FRAGMENT, typename BaseMMA::vtype_a>(s_a, offsets);
            }
        } else {
            constexpr int m_repeat = decltype(m_i)::value;
            const int stage = static_cast<int>(stage_i);
            const auto offsets = layout_to_offsets<T::VEC_A>(u_ra + sa_offset(stage));
            static_for<T::A_CHUNKS_PER_FRAGMENT>([&](auto chunk_i) {
                constexpr int index = m_repeat * T::A_CHUNKS_PER_FRAGMENT + decltype(chunk_i)::value;
                set_slice(v_a, load<T::VEC_A>(s_a, offsets[index]),
                    number<((m_repeat % T::A_RING) * T::A_CHUNKS_PER_FRAGMENT + decltype(chunk_i)::value) * T::VEC_A>{},
                    number<((m_repeat % T::A_RING) * T::A_CHUNKS_PER_FRAGMENT + decltype(chunk_i)::value + 1) * T::VEC_A>{});
            });
        }
    };

    auto load_b_fragment = [&](auto n_i, auto stage_i) {
        if constexpr (T::FRAGMENT_ARRAYS) {
            constexpr int n_repeat = decltype(n_i)::value;
            const auto offsets = layout_to_offsets<T::VEC_B>(u_rb + sb_offset(decltype(stage_i)::value));
            if constexpr (T::PIN_B) {
                load_operand_fragment_pinned<T::VEC_B, n_repeat * T::B_CHUNKS_PER_FRAGMENT>(number<64 + n_repeat * 8>{}, v_b[n_repeat], s_b, offsets);
            } else {
                v_b[n_repeat] = load_operand_fragment_staged<T::VEC_B, n_repeat * T::B_CHUNKS_PER_FRAGMENT, typename BaseMMA::vtype_b>(s_b, offsets);
            }
        } else {
            constexpr int n_repeat = decltype(n_i)::value;
            constexpr int stage = decltype(stage_i)::value;
            const auto offsets = layout_to_offsets<T::VEC_B>(u_rb + sb_offset(stage));
            static_for<T::B_CHUNKS_PER_FRAGMENT>([&](auto chunk_i) {
                constexpr int index = n_repeat * T::B_CHUNKS_PER_FRAGMENT + decltype(chunk_i)::value;
                set_slice(v_b, load<T::VEC_B>(s_b, offsets[index]),
                    number<index * T::VEC_B>{}, number<(index + 1) * T::VEC_B>{});
            });
        }
    };

    auto mma_scale_fragment = [&](auto m_i, auto n_i) {
        if constexpr (T::FRAGMENT_ARRAYS) {
            constexpr int mr = decltype(m_i)::value;
            constexpr int nr = decltype(n_i)::value;
            constexpr int ci = mr * T::E_N + nr;
            constexpr int sn = nr / (T::GROUP_N / (T::T_N * T::W_N));
            if constexpr (T::PIN_C) {
                // Pin C and D to the same AGPRs for in-place accumulation.
                [[clang::amdgpu_pin_agpr(128 + ci * 4)]]
                v_c[ci] = v_c[ci];
                [[clang::amdgpu_pin_agpr(128 + ci * 4)]]
                v_c[ci] = BaseMMA{}(v_a[mr % T::A_RING], v_b[nr], v_c[ci],
                    static_cast<int>(v_sfa[mr / T::VEC_SF]), static_cast<int>(v_sfb[0]), number<mr % T::VEC_SF>{}, number<sn>{});
            } else {
                v_c[ci] = BaseMMA{}(v_a[mr % T::A_RING], v_b[nr], v_c[ci],
                    static_cast<int>(v_sfa[mr / T::VEC_SF]), static_cast<int>(v_sfb[0]), number<mr % T::VEC_SF>{}, number<sn>{});
            }
        } else {
            constexpr int m_repeat = decltype(m_i)::value;
            constexpr int n_repeat = decltype(n_i)::value;
            constexpr int c_index = m_repeat * T::E_N + n_repeat;
            constexpr int scale_n_index = n_repeat / (T::GROUP_N / (T::T_N * T::W_N));
            const auto a = slice(v_a, number<(m_repeat % T::A_RING) * ELEM_A>{}, number<(m_repeat % T::A_RING + 1) * ELEM_A>{});
            const auto b = slice(v_b, number<n_repeat * ELEM_B>{}, number<(n_repeat + 1) * ELEM_B>{});
            auto c = slice(v_c, number<c_index * ELEM_C>{}, number<(c_index + 1) * ELEM_C>{});
            c = typename decltype(mma)::MMA{}(a, b, c,
                static_cast<int>(v_sfa[m_repeat / T::VEC_SF]), static_cast<int>(v_sfb[0]),
                number<m_repeat % T::VEC_SF>{}, number<scale_n_index>{});
            set_slice(v_c, c, number<c_index * ELEM_C>{}, number<(c_index + 1) * ELEM_C>{});
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

    load_sfa_panel(0);
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
#pragma nounroll
    for (int tile_k = 0; tile_k + 1 < loops; tile_k += T::LOOP_UNROLL) {
        advance_tile(number<0>{}, tile_k);

        if (tile_k + 2 < loops)
            advance_tile(number<1>{}, tile_k + 1);
    }

    // ===== Epilogue =====
    if constexpr (T::FRAGMENT_ARRAYS || (T::FIXED_K == 0 && T::E_M >= 3)) {
        // One final MFMA site keeps pinned C/D in place and shortens live
        // ranges when the final LDS stage is selected at runtime.
        finish_tile((loops - 1) % 2);
    } else {
        // Fixed-K and small unpinned tiles retain their constant LDS offsets.
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

    // Write contiguous BF16 vectors, preserving the final partial M tile guard.
    s_waitcnt_lgkmcnt(0_I);
    opus::sync_threads();

    static_for<T::OUTPUT_PASSES>([&](auto copy_i) { copy_output_bf16(decltype(copy_i)::value); });
}
#endif // gfx950
