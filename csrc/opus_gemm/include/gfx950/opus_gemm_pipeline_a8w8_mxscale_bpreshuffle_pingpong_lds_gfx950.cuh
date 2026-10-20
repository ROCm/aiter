// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <opus/hip_minimal.hpp>
#include <opus/opus.hpp>
#include "../opus_gemm_utils.cuh"
#include "opus_gemm_kargs_a8w8_mxscale_bpreshuffle_gfx950.cuh"
#include "opus_gemm_traits_a8w8_mxscale_bpreshuffle_pingpong_lds_gfx950.cuh"

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

// A layouts: global memory -> LDS -> registers.
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
        opus::unfold_x_stride(ga_block_dim, ga_block_shape, opus::tuple{stride_a, 1_I}),
        opus::unfold_p_coord(ga_block_dim, opus::tuple{wave_id_n, lane_id / threads_k, wave_id_m, lane_id % threads_k}));
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
        opus::unfold_x_stride(sa_block_dim, sa_block_shape, opus::tuple{opus::number<T::smem_linear_wave + T::smem_padding>{}, 1_I}),
        opus::unfold_p_coord(sa_block_dim, opus::tuple{wave_id_n, wave_id_m}));
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
        opus::unfold_x_stride(ra_block_dim, ra_block_shape, opus::tuple{opus::number<T::smem_linear_wave + T::smem_padding>{}, 1_I}),
        opus::unfold_p_coord(ra_block_dim, opus::tuple{producer_n, producer_m, producer_lane_row, lane_id / T::W_M}));
}

// B-preshuffle layouts: global memory -> LDS -> registers.
template<class T>
__device__ inline constexpr auto make_layout_gb_scale(int lane_id, int wave_id_m, int wave_id_n, int stride_b) {
    constexpr int lanes_n = T::W_N;
    constexpr int lanes_k = T::WARP_SIZE / lanes_n;
    constexpr int k_per_wave_load = lanes_k * T::VEC_B;
    constexpr int n_repeats = T::HALF_B_N / (T::NUM_WAVES * lanes_n);
    constexpr int k_repeats = T::B_K / k_per_wave_load;

    constexpr auto gb_block_shape = opus::make_tuple(
        opus::number<T::T_N>{},
        opus::number<T::T_M>{},
        opus::number<n_repeats>{},
        opus::number<k_repeats>{},
        opus::number<T::WARP_SIZE>{},
        opus::number<T::VEC_B>{});

    constexpr auto gb_block_dim = opus::make_tuple(
        opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::y_dim{}),
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::y_dim{}));

    return opus::make_layout<T::VEC_B>(
        gb_block_shape,
        opus::unfold_x_stride(gb_block_dim, gb_block_shape, opus::tuple{T::W_N * stride_b, 1_I}),
        opus::unfold_p_coord(gb_block_dim, opus::tuple{wave_id_n, wave_id_m, lane_id}));
}

template<class T>
__device__ inline constexpr auto make_layout_sb_scale(int wave_id_m, int wave_id_n) {
    constexpr int n_repeats = T::HALF_B_N / (T::NUM_WAVES * T::W_N);
    constexpr int k_per_wave_load = (T::WARP_SIZE / T::W_N) * T::VEC_B;
    constexpr int k_repeats = T::B_K / k_per_wave_load;

    constexpr auto sb_block_shape = opus::make_tuple(
        opus::number<T::T_N>{},
        opus::number<T::T_M>{},
        opus::number<n_repeats>{},
        opus::number<k_repeats>{},
        opus::number<T::VEC_B>{});

    constexpr auto sb_block_dim = opus::make_tuple(
        opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::y_dim{}, opus::y_dim{}),
        opus::make_tuple(opus::y_dim{}));

    return opus::make_layout(
        sb_block_shape,
        opus::unfold_x_stride(sb_block_dim, sb_block_shape, opus::tuple{opus::number<T::smem_linear_wave + T::smem_padding>{}, 1_I}),
        opus::unfold_p_coord(sb_block_dim, opus::tuple{wave_id_n, wave_id_m}));
}

template<class T>
__device__ inline constexpr auto make_layout_rb_scale(int lane_id, int wave_id_n) {
    constexpr int k_vectors = T::W_N * T::W_K / (T::WARP_SIZE * T::VEC_B);

    constexpr auto rb_block_shape = opus::make_tuple(
        opus::number<T::E_N>{},
        opus::number<T::T_N>{},
        opus::number<T::E_K>{},
        opus::number<k_vectors>{},
        opus::number<T::WARP_SIZE>{},
        opus::number<T::VEC_B>{});

    constexpr auto rb_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::y_dim{}, opus::y_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}));

    return opus::make_layout<T::VEC_B>(
        rb_block_shape,
        opus::unfold_x_stride(rb_block_dim, rb_block_shape, opus::tuple{opus::number<T::smem_linear_wave + T::smem_padding>{}, 1_I}),
        opus::unfold_p_coord(rb_block_dim, opus::tuple{wave_id_n, lane_id}));
}

// A-scale layouts: external K128 bytes -> LDS -> registers.
template<class T>
__device__ inline constexpr auto make_layout_gsfa_scale(int lane_id, int wave_id_m, int wave_id_n, int stride_sfa) {
    constexpr auto block_shape = opus::make_tuple(
        opus::number<T::T_N>{}, // 2 wave_n
        opus::number<T::T_M>{}, // M waves
        opus::number<T::SFA_K_COLUMNS_PER_WAVE>{}, // K128 groups per wave
        opus::number<T::SFA_PASSES>{}, // passes along M
        opus::number<T::SFA_THREADS_PER_GROUP>{}, // threads per K128 group
        opus::number<T::VEC_SCALE_A>{});           // 16 contiguous M scale bytes per thread

    constexpr auto block_dim = opus::make_tuple(
        opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::y_dim{}));

    return opus::make_layout<T::VEC_SCALE_A>(
        block_shape,
        opus::unfold_x_stride(block_dim, block_shape, opus::tuple{stride_sfa, 1_I}),
        opus::unfold_p_coord(block_dim, opus::tuple{wave_id_n, wave_id_m, lane_id / T::SFA_THREADS_PER_GROUP, lane_id % T::SFA_THREADS_PER_GROUP}));
}

template<class T>
__device__ inline constexpr auto make_layout_ssfa_scale(int lane_id, int wave_id_m, int wave_id_n) {
    constexpr auto block_shape = opus::make_tuple(
        opus::number<T::T_N>{}, // 2 wave_n
        opus::number<T::T_M>{}, // M waves
        opus::number<T::SFA_K_COLUMNS_PER_WAVE>{}, // K128 groups per wave
        opus::number<T::SFA_PASSES>{}, // passes along M
        opus::number<T::SFA_THREADS_PER_GROUP>{}, // threads per K128 group
        opus::number<T::VEC_SCALE_A>{});           // 16 contiguous M scale bytes per thread

    constexpr auto block_dim = opus::make_tuple(
        opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::y_dim{}));

    return opus::make_layout<T::VEC_SCALE_A>(
        block_shape,
        opus::unfold_x_stride(block_dim, block_shape, opus::tuple{opus::number<T::B_M>{}, 1_I}),
        opus::unfold_p_coord(block_dim, opus::tuple{wave_id_n, wave_id_m, lane_id / T::SFA_THREADS_PER_GROUP, lane_id % T::SFA_THREADS_PER_GROUP}));
}

template<class T>
__device__ inline constexpr auto make_layout_rsfa_scale(int lane_id, int wave_id_m) {
    constexpr auto block_shape = opus::make_tuple(
        opus::number<T::E_M>{}, // M repeats
        opus::number<T::T_M>{}, // M waves
        opus::number<T::W_M>{}, // 16 M lanes
        1_I);                   // one scale byte per M repeat

    constexpr auto block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}));

    return opus::make_layout<1>(
        block_shape,
        opus::unfold_x_stride(block_dim, block_shape, opus::tuple{1_I, 1_I}),
        opus::unfold_p_coord(block_dim, opus::tuple{wave_id_m, lane_id % T::W_M}));
}

// B-scale layouts: external K128 bytes -> packed LDS words -> registers.
template<class T>
__device__ inline constexpr auto make_layout_gsfb_scale(int lane_id, int wave_id, int stride_sfb) {
    constexpr auto block_shape = opus::make_tuple(
        opus::number<T::SCALE_N_HALVES>{},
        opus::number<T::NUM_WAVES>{},
        opus::number<T::WARP_SIZE>{},
        1_I);

    constexpr auto block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::y_dim{}));

    return opus::make_layout<1>(
        block_shape,
        opus::unfold_x_stride(block_dim, block_shape, opus::tuple{stride_sfb, 1_I}),
        opus::unfold_p_coord(block_dim, opus::tuple{wave_id, lane_id}));
}

template<class T>
__device__ inline constexpr auto make_layout_ssfb_scale(int lane_id, int wave_id) {
    constexpr auto block_shape = opus::make_tuple(
        opus::number<T::NUM_WAVES>{},
        opus::number<T::WARP_SIZE>{},
        1_I);

    constexpr auto block_dim = opus::make_tuple(
        opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::y_dim{}));

    // Offsets count packed uint words, one per K128 group.
    return opus::make_layout<1>(
        block_shape,
        opus::unfold_x_stride(block_dim, block_shape, opus::tuple{1_I}),
        opus::unfold_p_coord(block_dim, opus::tuple{wave_id, lane_id}));
}

__device__ inline constexpr auto make_layout_rsfb_scale() {
    constexpr auto block_shape = opus::make_tuple(
        1_I);
    constexpr auto block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}));

    return opus::make_layout<1>(
        block_shape,
        opus::unfold_x_stride(block_dim, block_shape, opus::tuple{1_I}),
        opus::unfold_p_coord(block_dim, opus::tuple{}));
}

} // namespace opus_gemm_bpreshuffle_pingpong_lds_layout

// Publish the next A/B LDS tile at PUBLISH_M, then replace consumed fragments.
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
    namespace layout_panel = opus_gemm_bpreshuffle_pingpong_lds_layout;

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
    const auto u_ra = layout_panel::make_layout_ra_scale<T>(lane_id, wave_id_m);
    const auto u_gb = make_layout_gb_scale<T>(lane_id, wave_id_m, wave_id_n, kargs.stride_b);
    const auto u_sb = make_layout_sb_scale<T>(wave_id_m, wave_id_n);
    const auto u_rb = make_layout_rb_scale<T>(lane_id, wave_id_n);
    const auto u_gsfa = layout_panel::make_layout_gsfa_scale<T>(lane_id, wave_id_m, wave_id_n, kargs.stride_sfa);
    const auto u_ssfa = layout_panel::make_layout_ssfa_scale<T>(lane_id, wave_id_m, wave_id_n);
    const auto u_rsfa = layout_panel::make_layout_rsfa_scale<T>(lane_id, wave_id_m);
    const auto u_gsfb = layout_panel::make_layout_gsfb_scale<T>(lane_id, wave_id, kargs.stride_sfb);
    const auto u_ssfb = layout_panel::make_layout_ssfb_scale<T>(lane_id, wave_id);
    const auto u_rsfb = layout_panel::make_layout_rsfb_scale();

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
        array<typename BaseMMA::vtype_a, T::E_M>, typename decltype(mma)::vtype_a>;
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
    auto ssfa_offset = [&](int tile_k) { return tile_k * number<T::B_M>{}; };

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
    const auto sfa_smem_offsets = layout_to_offsets<T::VEC_SCALE_A>(u_ssfa);
    const auto sfb_gmem_offsets = layout_to_offsets<1>(u_gsfb);
    const auto sfb_smem_offsets = layout_to_offsets<1>(u_ssfb);
    const auto rsfa_offsets = layout_to_offsets<1>(u_rsfa);
    const auto rsfb_offsets = layout_to_offsets<1>(u_rsfb);

    // Scale A global memory -> VGPR -> LDS, retaining raw E8M0 bytes.
    auto load_sfa_panel = [&](int panel_k_begin) {
        const int local_k_group = wave_id * T::SFA_K_COLUMNS_PER_WAVE + lane_id / T::SFA_THREADS_PER_GROUP;
        const int k_group = panel_k_begin + local_k_group;
        static_for<T::SFA_PASSES>([&](auto pass_i) {
            constexpr int pass = decltype(pass_i)::value;
            const int local_row = sfa_smem_offsets[pass] - local_k_group * T::B_M;
            const int smem_offset = sfa_smem_offsets[pass] + panel_k_begin * T::B_M;
            if (local_row < T::B_M && smem_offset < T::SFA_BYTES && k_group < scale_k_groups) {
                vector_t<D_SF, T::VEC_SCALE_A> raw;
                if (row + local_row < kargs.m)
                    raw = load<T::VEC_SCALE_A>(g_sfa, sfa_gmem_offsets[pass] + gsfa_offset(panel_k_begin));
                else
                    static_for<T::VEC_SCALE_A>([&](auto byte_i) { raw[decltype(byte_i)::value] = 0x7f; });
                store<T::VEC_SCALE_A>(s_sfa, raw, smem_offset);
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

    // Scale LDS -> VGPR; pack A repeats after reading LDS.
    auto read_scales = [&](int tile_k, auto& scale_a, auto& scale_b) {
        static_for<T::A_SCALE_PACKS>([&](auto p) { scale_a[decltype(p)::value] = 0; });
        static_for<T::E_M>([&](auto m_i) {
            constexpr int m_repeat = decltype(m_i)::value;
            const unsigned value = load<1>(s_sfa, rsfa_offsets[m_repeat] + ssfa_offset(tile_k))[0];
            scale_a[m_repeat / 4] |= value << ((m_repeat % 4) * 8);
        });
        scale_b[0] = load<1>(s_sfb, rsfb_offsets[0] + ssfb_offset(tile_k))[0];
    };

    // Matrix LDS -> registers, pinning A/B only when the corresponding trait is enabled.
    auto load_a_fragment = [&](auto m_i, auto stage_i) {
        if constexpr (T::FRAGMENT_ARRAYS) {
            constexpr int m_repeat = decltype(m_i)::value;
            const auto offsets = layout_to_offsets<T::VEC_A>(u_ra + sa_offset(decltype(stage_i)::value));
            if constexpr (T::PIN_A) {
                load_operand_fragment_pinned<T::VEC_A, m_repeat * T::A_CHUNKS_PER_FRAGMENT>( number<m_repeat * 8>{}, v_a[m_repeat], s_a, offsets);
            } else {
                v_a[m_repeat] = load_operand_fragment_staged< T::VEC_A, m_repeat * T::A_CHUNKS_PER_FRAGMENT, typename BaseMMA::vtype_a>(s_a, offsets);
            }
        } else {
            constexpr int m_repeat = decltype(m_i)::value;
            constexpr int stage = decltype(stage_i)::value;
            const auto offsets = layout_to_offsets<T::VEC_A>(u_ra + sa_offset(stage));
            static_for<T::A_CHUNKS_PER_FRAGMENT>([&](auto chunk_i) {
                constexpr int index = m_repeat * T::A_CHUNKS_PER_FRAGMENT + decltype(chunk_i)::value;
                set_slice(v_a, load<T::VEC_A>(s_a, offsets[index]),
                    number<index * T::VEC_A>{}, number<(index + 1) * T::VEC_A>{});
            });
        }
    };

    auto load_b_fragment = [&](auto n_i, auto stage_i) {
        if constexpr (T::FRAGMENT_ARRAYS) {
            constexpr int n_repeat = decltype(n_i)::value;
            const auto offsets = layout_to_offsets<T::VEC_B>(u_rb + sb_offset(decltype(stage_i)::value));
            if constexpr (T::PIN_B) {
                load_operand_fragment_pinned<T::VEC_B, n_repeat * T::B_CHUNKS_PER_FRAGMENT>( number<64 + n_repeat * 8>{}, v_b[n_repeat], s_b, offsets);
            } else {
                v_b[n_repeat] = load_operand_fragment_staged< T::VEC_B, n_repeat * T::B_CHUNKS_PER_FRAGMENT, typename BaseMMA::vtype_b>(s_b, offsets);
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
                v_c[ci] = BaseMMA{}(v_a[mr], v_b[nr], v_c[ci],
                    static_cast<int>(v_sfa[mr / 4]), static_cast<int>(v_sfb[0]), number<mr % 4>{}, number<sn>{});
            } else {
                v_c[ci] = BaseMMA{}(v_a[mr], v_b[nr], v_c[ci],
                    static_cast<int>(v_sfa[mr / 4]), static_cast<int>(v_sfb[0]), number<mr % 4>{}, number<sn>{});
            }
        } else {
            constexpr int m_repeat = decltype(m_i)::value;
            constexpr int n_repeat = decltype(n_i)::value;
            constexpr int c_index = m_repeat * T::E_N + n_repeat;
            constexpr int scale_n_index = n_repeat / (T::GROUP_N / (T::T_N * T::W_N));
            const auto a = slice(v_a, number<m_repeat * ELEM_A>{}, number<(m_repeat + 1) * ELEM_A>{});
            const auto b = slice(v_b, number<n_repeat * ELEM_B>{}, number<(n_repeat + 1) * ELEM_B>{});
            auto c = slice(v_c, number<c_index * ELEM_C>{}, number<(c_index + 1) * ELEM_C>{});
            c = typename decltype(mma)::MMA{}(a, b, c,
                static_cast<int>(v_sfa[m_repeat / 4]), static_cast<int>(v_sfb[0]),
                number<m_repeat % 4>{}, number<scale_n_index>{});
            set_slice(v_c, c, number<c_index * ELEM_C>{}, number<(c_index + 1) * ELEM_C>{});
        }
    };

    auto advance_tile = [&](auto stage_i, int tile_k) {
        constexpr int stage = decltype(stage_i)::value;
        constexpr int next_stage = (stage + 1) % T::NUM_STAGES;
        read_scales(tile_k + 1, v_sfa_next, v_sfb_next);
        // Overlap the selected M repeats with the pending next matrix tile.
        static_for<T::E_M>([&](auto m_i) {
            static_for<T::E_N>([&](auto n_i) {
                mma_scale_fragment(m_i, n_i);
                if constexpr (decltype(m_i)::value == T::E_M - 1)
                    load_b_fragment(n_i, number<next_stage>{});
            });
            if constexpr (decltype(m_i)::value == T::PUBLISH_M) {
                __builtin_amdgcn_sched_barrier(0);
                s_waitcnt_vmcnt(0_I);
                s_waitcnt_lgkmcnt(0_I);
                // Publish K+1 and retire all K readers before K+2 reuses the same LDS slot.
                opus::sync_threads();
                if (tile_k + 2 < loops)
                    issue_matrix_prefetch(stage_i, tile_k + 2);
                __builtin_amdgcn_sched_barrier(0);
                static_for<T::PUBLISH_M>([&](auto earlier_m) { load_a_fragment(earlier_m, number<next_stage>{}); });
            }
            // Earlier A fragments remain intact until K+1 is published.
            if constexpr (decltype(m_i)::value >= T::PUBLISH_M)
                load_a_fragment(m_i, number<next_stage>{});
        });
        // Complete this wave's next-tile reads; the next midpoint barrier protects LDS reuse.
        s_waitcnt_lgkmcnt(0_I);
        v_sfa = v_sfa_next;
        v_sfb = v_sfb_next;
    };

    // ===== Prologue =====
    // Publish scales and the first matrix tile, then prefetch K+1.
    issue_matrix_prefetch(number<0>{}, 0);
    __builtin_amdgcn_sched_barrier(0);

    // Spread short-K scale loads across waves without reserving K-specific kernels.
    static_for<T::SFA_K_PASSES>([&](auto i) {
        const int first_k = decltype(i)::value * T::SFA_K_PANEL;
        if (first_k < scale_k_groups)
            load_sfa_panel(first_k);
    });
    load_sfb_panel(0);
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
    static_for<T::E_M>([&](auto m_i) { load_a_fragment(m_i, number<0>{}); });
    static_for<T::E_N>([&](auto n_i) { load_b_fragment(n_i, number<0>{}); });
    s_waitcnt_lgkmcnt(0_I);

    // ===== Main loop =====
    // Alternate LDS slots and replace operands after their last consumer.
    for (int tile_k = 0; tile_k + 1 < loops; tile_k += T::LOOP_UNROLL) {
        advance_tile(number<0>{}, tile_k);

        if (tile_k + 2 < loops)
            advance_tile(number<1>{}, tile_k + 1);
    }

    // ===== Output writeback =====
    // Finish the last tile and stage BF16 fragments in LDS.
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

    s_waitcnt_lgkmcnt(0_I);
    opus::sync_threads();

    static_for<T::E_M>([&](auto m_i) {
        static_for<T::E_N>([&](auto n_i) {
            constexpr int c_index = decltype(m_i)::value * T::E_N + decltype(n_i)::value;
            mma_scale_fragment(m_i, n_i);
            stage_output_fragment(number<c_index>{});
        });
    });

    // Write contiguous BF16 vectors, preserving the final partial M tile guard.
    s_waitcnt_lgkmcnt(0_I);
    opus::sync_threads();
    const int output_thread_id = wave_id * T::WARP_SIZE + lane_id;
    auto copy_output_bf16 = [&](int copy_index) {
        const int linear = output_thread_id * T::VEC_OUTPUT + copy_index * T::BLOCK_SIZE * T::VEC_OUTPUT;
        const int output_row = linear / T::B_N;
        const int output_col = linear % T::B_N;
        const auto value = load<T::VEC_OUTPUT>(s_c, c_offset(output_row, output_col));
        // The bounded SRD drops complete rows beyond valid_rows. Keep the
        // lane guard as the default; eligible configurations can omit it.
        if (T::BOUNDED_OUTPUT || row + output_row < kargs.m)
            store<T::VEC_OUTPUT>(g_c, value, output_row * kargs.stride_c + output_col, 0, opus::number<2>{});
    };
    static_for<T::OUTPUT_PASSES>([&](auto copy_i) { copy_output_bf16(decltype(copy_i)::value); });
}
#endif
