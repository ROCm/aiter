// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>
#include <type_traits>
#include <opus/hip_minimal.hpp>
#include <opus/opus.hpp>
#include "../opus_gemm_utils.cuh"
#include "opus_gemm_traits_a8w8_blockscale_mxscale_bpreshuffle_staged_b_gfx950.cuh"
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx950__)
namespace opus_gemm_bpreshuffle_staged_b_layout {


template<int CachedVec = 0, class Shape, class Dimensions, class Strides, class Coordinates>
__device__ inline constexpr auto make_layout_from_coordinates(
    const Shape& shape, const Dimensions& dimensions, const Strides& strides, const Coordinates& coordinates) {
    return opus::make_layout<CachedVec>(shape, opus::unfold_x_stride(dimensions, shape, strides), opus::unfold_p_coord(dimensions, coordinates));
}


template<class T>
__device__ inline auto make_layout_ga_scale(int lane_id, int wave_id_m, int wave_id_n, int stride_a) {
    constexpr int threads_k = T::B_K / T::VEC_A;
    constexpr int threads_m_per_block = T::BLOCK_SIZE / threads_k;
    constexpr int threads_m_per_wave = T::WARP_SIZE / threads_k;
    constexpr auto ga_block_shape = opus::make_tuple(opus::number<T::HALF_B_M / threads_m_per_block>{}, opus::number<T::T_N>{},
        opus::number<threads_m_per_wave>{}, opus::number<T::T_M>{}, opus::number<threads_k>{}, opus::number<T::VEC_A>{});
    constexpr auto ga_block_dim = opus::make_tuple(opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}));
    return make_layout_from_coordinates<T::VEC_A>(ga_block_shape, ga_block_dim, opus::tuple{stride_a, 1_I},
        opus::tuple{wave_id_n, lane_id / threads_k, wave_id_m, lane_id % threads_k});
}


template<int Repeats, int WavesFirst, int WavesSecond, int Vector, int Pitch>
__device__ inline constexpr auto make_matrix_stage_layout(int first_wave, int second_wave) {
    constexpr auto shape = opus::make_tuple(opus::number<Repeats>{}, opus::number<WavesFirst>{}, opus::number<WavesSecond>{}, opus::number<Vector>{});
    constexpr auto dimensions = opus::make_tuple(opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}), opus::make_tuple(opus::y_dim{}));
    return make_layout_from_coordinates(shape, dimensions, opus::tuple{opus::number<Pitch>{}, 1_I}, opus::tuple{first_wave, second_wave});
}

template<class T>
__device__ inline auto make_layout_sa_scale(int wave_id_m, int wave_id_n) {
    return make_matrix_stage_layout<T::smem_m_rep / T::NUM_WAVES, T::T_N, T::T_M, T::VEC_A, T::smem_linear_wave + T::smem_padding>(wave_id_n, wave_id_m);
}

template<class T>
__device__ inline constexpr auto make_layout_ra_scale(int lane_id, int wave_id_m) {
    constexpr auto ra_block_shape = opus::make_tuple(opus::number<T::E_M>{}, opus::number<T::W_M / T::A_ROWS_PER_WAVE>{},
        opus::number<T::T_M>{}, opus::number<T::A_ROWS_PER_WAVE>{}, opus::number<T::E_K>{},
        opus::number<T::A_CHUNKS_PER_FRAGMENT>{}, opus::number<T::WARP_SIZE / T::W_M>{}, opus::number<T::VEC_A>{});
    constexpr auto ra_block_dim = opus::make_tuple(opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}, opus::y_dim{}, opus::p_dim{}, opus::y_dim{}));
    const int lane_id_m = lane_id % T::W_M;
    const int matrix_row = wave_id_m * T::W_M + lane_id_m;
    const int producer_n = matrix_row / (T::A_ROWS_PER_WAVE * T::T_M);
    const int producer_m = matrix_row % T::T_M;
    const int producer_lane_row = (matrix_row % (T::A_ROWS_PER_WAVE * T::T_M)) / T::T_M;
    return make_layout_from_coordinates<T::VEC_A>(ra_block_shape, ra_block_dim, opus::tuple{opus::number<T::smem_linear_wave + T::smem_padding>{}, 1_I},
        opus::tuple{producer_n, producer_m, producer_lane_row, lane_id / T::W_M});
}


template<class T>
__device__ inline constexpr auto make_layout_gb_scale(int lane_id, int wave_id_m, int wave_id_n, int stride_b) {
    if constexpr (T::B_N < T::NUM_WAVES * T::W_N) {
        constexpr int waves_n = T::B_N / T::W_N;
        constexpr int waves_k = T::NUM_WAVES / waves_n;
        constexpr int k_repeats = T::B_K / (64 * waves_k);
        const int producer = wave_id_m * T::T_N + wave_id_n;
        constexpr auto shape = opus::make_tuple(1_I, opus::number<waves_n>{},
            opus::number<k_repeats>{}, opus::number<waves_k>{}, opus::number<T::WARP_SIZE>{}, opus::number<T::VEC_B>{});
        constexpr auto dims = opus::make_tuple(opus::make_tuple(opus::y_dim{}, opus::p_dim{}),
            opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}, opus::y_dim{}));
        return make_layout_from_coordinates<T::VEC_B>(shape, dims, opus::tuple{T::W_N * stride_b, 1_I},
            opus::tuple{producer % waves_n, producer / waves_n, lane_id});
    } else {
        constexpr int lanes_n = T::W_N;
        constexpr int lanes_k = T::WARP_SIZE / lanes_n;
        constexpr int k_per_wave_load = lanes_k * T::VEC_B;
        constexpr int n_repeats = T::HALF_B_N / (T::NUM_WAVES * lanes_n);
        constexpr int k_repeats = T::B_K / k_per_wave_load;
        constexpr auto gb_block_shape = opus::make_tuple(opus::number<n_repeats>{}, opus::number<T::T_M>{},
            opus::number<T::T_N>{}, opus::number<k_repeats>{}, opus::number<T::WARP_SIZE>{}, opus::number<T::VEC_B>{});
        constexpr auto gb_block_dim = opus::make_tuple(opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}),
            opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::y_dim{}));
        return make_layout_from_coordinates<T::VEC_B>(gb_block_shape, gb_block_dim, opus::tuple{T::W_N * stride_b, 1_I},
            opus::tuple{wave_id_m, wave_id_n, lane_id});
    }
}

template<class T>
__device__ inline constexpr auto make_layout_sb_scale(int wave_id_m, int wave_id_n) {
    if constexpr (T::B_N < T::NUM_WAVES * T::W_N) {
        constexpr int waves_n = T::B_N / T::W_N;
        constexpr int waves_k = T::NUM_WAVES / waves_n;
        constexpr int k_repeats = T::B_K / (64 * waves_k);
        const int producer = wave_id_m * T::T_N + wave_id_n;
        constexpr auto shape = opus::make_tuple(1_I, opus::number<waves_n>{}, opus::number<k_repeats>{}, opus::number<waves_k>{}, opus::number<T::VEC_B>{});
        constexpr auto dims = opus::make_tuple(opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::y_dim{}, opus::p_dim{}), opus::make_tuple(opus::y_dim{}));
        return make_layout_from_coordinates(shape, dims, opus::tuple{opus::number<T::smem_linear_wave + T::smem_padding>{}, 1_I},
            opus::tuple{producer % waves_n, producer / waves_n});
    } else {
        return make_matrix_stage_layout<T::smem_n_rep / T::NUM_WAVES, T::T_M, T::T_N, T::VEC_B, T::smem_linear_wave + T::smem_padding>(wave_id_m, wave_id_n);
    }
}

template<class T>
__device__ inline constexpr auto make_layout_rb_scale(int lane_id, int wave_id_n) {
    if constexpr (T::B_N < T::NUM_WAVES * T::W_N) {
        constexpr int P = T::smem_linear_wave + T::smem_padding;
        constexpr auto shape = opus::make_tuple(opus::number<T::E_N>{}, opus::number<T::T_N>{},
            opus::number<T::E_K>{}, opus::number<T::B_CHUNKS_PER_FRAGMENT>{}, opus::number<T::WARP_SIZE>{}, opus::number<T::VEC_B>{});
        constexpr auto dims = opus::make_tuple(opus::make_tuple(opus::y_dim{}, opus::p_dim{}), opus::make_tuple(opus::y_dim{}),
            opus::make_tuple(opus::y_dim{}), opus::make_tuple(opus::p_dim{}, opus::y_dim{}));
        return make_layout_from_coordinates<T::VEC_B>(shape, dims, opus::tuple{opus::number<T::B_K / 64 * P>{},
                opus::number<T::B_CHUNKS_PER_FRAGMENT * P>{}, opus::number<P>{}, 1_I}, opus::tuple{wave_id_n, lane_id});
    } else {
        constexpr int P = T::smem_linear_wave + T::smem_padding;
        constexpr int n_repeats = T::NUM_WAVES / T::T_N;
        constexpr int k_vectors = T::W_N * T::W_K / (T::WARP_SIZE * T::VEC_B);
        constexpr auto rb_block_shape = opus::make_tuple(opus::number<T::E_N / n_repeats>{}, opus::number<n_repeats>{},
            opus::number<T::T_N>{}, opus::number<k_vectors>{}, opus::number<T::E_K>{}, opus::number<T::WARP_SIZE>{}, opus::number<T::VEC_B>{});
        constexpr auto rb_block_dim = opus::make_tuple(opus::make_tuple(opus::y_dim{}),
            opus::make_tuple(opus::y_dim{}, opus::p_dim{}), opus::make_tuple(opus::y_dim{}), opus::make_tuple(opus::y_dim{}),
            opus::make_tuple(opus::p_dim{}, opus::y_dim{}));
        return make_layout_from_coordinates<T::VEC_B>(rb_block_shape, rb_block_dim, opus::tuple{opus::number<T::NUM_WAVES * k_vectors * T::E_K * P>{},
                    opus::number<P>{}, opus::number<T::NUM_WAVES * P>{}, opus::number<T::NUM_WAVES * k_vectors * P>{}, 1_I}, opus::tuple{wave_id_n, lane_id});
    }
}


template<class T>
__device__ inline constexpr auto make_layout_gsfa_scale(int lane_id, int wave_id_m, int wave_id_n, int stride_sfa) {
    constexpr auto gsfa_block_shape = opus::make_tuple(opus::number<T::SFA_PASSES_PER_CACHE_PANEL>{}, opus::number<T::T_N>{},
        opus::number<T::T_M>{}, opus::number<T::SFA_K_COLUMNS_PER_WAVE>{}, opus::number<T::SFA_M_VECTORS>{}, opus::number<T::VEC_SCALE_A>{});
    constexpr auto gsfa_block_dim = opus::make_tuple(opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}));
    const int lane_m = lane_id % T::SFA_M_VECTORS;
    const int pack = lane_m / (T::T_M * T::VEC_SF);
    const int consumer_wave_m = (lane_m / T::VEC_SF) % T::T_M;
    const int m_vector = (pack * T::VEC_SF + lane_m % T::VEC_SF) * T::T_M + consumer_wave_m;
    return make_layout_from_coordinates<T::VEC_SCALE_A>(gsfa_block_shape, gsfa_block_dim, opus::tuple{stride_sfa, 1_I},
        opus::tuple{wave_id_n, wave_id_m, lane_id / T::SFA_M_VECTORS, m_vector});
}

template<class T>
__device__ inline constexpr auto make_layout_ssfa_scale(int lane_id, int wave_id_m, int wave_id_n) {
    constexpr auto ssfa_block_shape = opus::make_tuple(opus::number<T::SFA_PASSES_PER_CACHE_PANEL>{}, opus::number<T::T_N>{},
        opus::number<T::T_M>{}, opus::number<T::SFA_K_COLUMNS_PER_WAVE>{}, opus::number<T::A_SCALE_PACKS>{},
        opus::number<T::T_M>{}, opus::number<T::W_M>{}, opus::number<T::VEC_SF>{});
    constexpr auto ssfa_block_dim = opus::make_tuple(opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::p_dim{}, opus::y_dim{}));
    const int lane_m = lane_id % T::SFA_M_VECTORS;
    return make_layout_from_coordinates<T::VEC_SF>(ssfa_block_shape, ssfa_block_dim,
        opus::tuple{opus::number<T::SFA_PANEL_PITCH>{}, 1_I}, opus::tuple{wave_id_n, wave_id_m, lane_id / T::SFA_M_VECTORS,
                lane_m / (T::T_M * T::VEC_SF), (lane_m / T::VEC_SF) % T::T_M, lane_m % T::VEC_SF});
}

template<class T>
__device__ inline constexpr auto make_layout_rsfa_scale(int lane_id, int wave_id_m) {
    constexpr auto rsfa_block_shape = opus::make_tuple(opus::number<T::A_SCALE_PACKS>{}, opus::number<T::T_M>{},
        opus::number<T::W_M>{}, opus::number<T::VEC_SF>{});
    constexpr auto rsfa_block_dim = opus::make_tuple(opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}), opus::make_tuple(opus::y_dim{}));
    return make_layout_from_coordinates<T::VEC_SF>(rsfa_block_shape, rsfa_block_dim,
        opus::tuple{opus::number<T::VEC_SF>{}, 1_I}, opus::tuple{wave_id_m, lane_id % T::W_M});
}



template<int Vec, int OFFSET_BEGIN, class Fragment, int OffsetStride = 1, class Mem, class Offsets>
__device__ inline Fragment load_operand_fragment_staged(Mem& smem, const Offsets& offsets) {
    using Chunk = decltype(opus::load<Vec>(smem, offsets[OFFSET_BEGIN]));
    opus::array<Chunk, 2> values;
    opus::static_for<2>([&](auto i) {
        constexpr int chunk = decltype(i)::value;
        values[chunk] = opus::load<Vec>(smem, offsets[OFFSET_BEGIN + chunk * OffsetStride]);
    });
    Fragment result;
    opus::set_slice(result, values[0], opus::number<0>{}, opus::number<Vec>{});
    opus::set_slice(result, values[1], opus::number<Vec>{}, opus::number<2 * Vec>{});
    return result;
}
} // namespace opus_gemm_bpreshuffle_staged_b_layout


template<class Traits, bool ShortK, int Capacity>
__device__ __forceinline__ void opus_gemm_bpreshuffle_staged_b_body_impl(
    opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 kargs) {
    using namespace opus;
    using namespace opus_gemm_bpreshuffle_staged_b_layout;
    using T = opus::remove_cvref_t<Traits>;
    constexpr int capacity = Capacity;
    static_assert(capacity >= 2 && capacity <= T::RESIDENT_CAPACITY);
    using D_A = opus::fp8_t;
    using D_B = opus::fp8_t;
    using D_C = std::conditional_t<T::SPLIT_K == 1, opus::bf16_t, opus::fp32_t>;
    using D_ACC = opus::fp32_t;
    using D_SF = unsigned char;
    using D_SF_PACK = unsigned int;
    if constexpr (ShortK) kargs.k = __builtin_amdgcn_readfirstlane(kargs.k);
    kargs.stride_a = kargs.k;
    kargs.stride_b = kargs.k;
    kargs.stride_c = kargs.n;
    kargs.stride_sfa = kargs.m;
    kargs.stride_sfb = kargs.k / T::GROUP_K;
    const int wave_id = __builtin_amdgcn_readfirstlane(thread_id_x() / T::WARP_SIZE);
    const int lane_id = thread_id_x() % T::WARP_SIZE;
    if constexpr (!ShortK && T::B_M == 48 && T::B_N == 64 && T::B_K == 128 &&
                  T::T_M == 1 && T::T_N == 2 && T::SCALE_PANEL == 128 &&
                  T::SPLIT_K == 1 && T::CACHE_A == 17 && T::CACHE_B == 1 && T::GROUPED_M == 0) {
        __builtin_assume(static_cast<unsigned>(thread_id_x()) < T::BLOCK_SIZE);
        __builtin_assume(static_cast<unsigned>(wave_id) < T::NUM_WAVES);
    }
    const int wave_id_m = wave_id % T::T_M;
    const int wave_id_n = wave_id / T::T_M;
    const int tile_gn = kargs.n / T::B_N;
    int block_m = block_id_y();
    int block_n = block_id_x();
    int split = T::SPLIT_K == 1 ? 0 : block_id_z();
    const int tile_linear = block_m * tile_gn + block_n;
    if constexpr (!ShortK && T::B_M == 256 && T::B_N == 256 && T::B_K == 128 && T::T_M == 4 && T::T_N == 2 && T::SCALE_PANEL == 64 && T::SPLIT_K == 1 && T::CACHE_A == 0 && T::CACHE_B == 0 && T::GROUPED_M == 0) {
        if (tile_gn >= 256 && tile_gn % 32 == 0 && kargs.m > 3 * T::B_M && kargs.m <= 4 * T::B_M) {
            const int ordered_n = block_m * (tile_gn >> 2) + ((block_n >> 5) << 3) + (block_n & 7);
            block_m = (block_n >> 3) & 3;
            block_n = ordered_n;
        }
    }
    constexpr bool normal_n_panel_order = !ShortK && T::B_M == 192 && T::B_N == 256 && T::B_K == 128 &&
        T::T_M == 4 && T::T_N == 2 && T::SCALE_PANEL == 16 && T::SPLIT_K == 1 &&
        T::CACHE_A == 0 && T::CACHE_B == 0 && T::GROUPED_M == 0;
    if constexpr (normal_n_panel_order || T::GROUPED_M > 0) {
        const bool group_n = normal_n_panel_order && tile_gn >= 256 && kargs.m > T::B_M;
        if (group_n || T::GROUPED_M > 0) {
            const int gm = group_n ? 1 + (kargs.m - 1) / T::B_M : (kargs.m + T::B_M - 1) / T::B_M;
            const int group = group_n ? 8 : T::GROUPED_M, extent = group_n ? tile_gn : gm;
            const int repeated = group_n ? gm : tile_gn, span = group * repeated;
            const int first = tile_linear / span * group, count = extent - first < group ? extent - first : group;
            const int local = tile_linear % span;
            block_m = group_n ? local / count : first + local % count;
            block_n = group_n ? first + local % count : local / count;
        }
    }
    const int row = block_m * T::B_M;
    const int col = block_n * T::B_N;
    const int total_k = kargs.k;
    const int part_k = total_k / T::SPLIT_K;
    const int loops = part_k / T::B_K;
    constexpr bool two_panel_short = ShortK && capacity == 2 && T::B_M == 32 && T::B_N == 32 && T::B_K == 512 && T::NUM_WAVES == 2 && T::E_M == 1;
    const bool small_full_two = two_panel_short && kargs.m > 0 && kargs.m <= 32 && loops == capacity;
    const int part_start = split * part_k;
    const bool ring8 = !ShortK && T::ring8_eligible(kargs.m, total_k);
    constexpr bool normal_stage4_profile = !ShortK && T::B_M == 32 && T::B_N == 32 && T::B_K == 256 &&
        T::T_M == 2 && T::T_N == 1 && T::SCALE_PANEL == 128 && T::SPLIT_K == 1 &&
        T::CACHE_A == 1 && T::CACHE_B == 17 && T::GROUPED_M == 0;
    constexpr bool normal_a1_overlap = !ShortK && T::B_M == 32 && T::B_N == 64 && T::B_K == 128 &&
        T::T_M == 1 && T::T_N == 4 && T::SCALE_PANEL == 256 && T::SPLIT_K == 1 &&
        T::CACHE_A == 17 && T::CACHE_B == 1 && T::GROUPED_M == 0;
    constexpr bool aligned_normal_scales = !ShortK && T::B_M == 32 && T::B_N == 32 && T::B_K == 256 && T::T_M == 2 &&
        ((T::T_N == 2 && T::SCALE_PANEL == 64) || (T::T_N == 1 && T::SCALE_PANEL == 128)) && T::SPLIT_K == 1 &&
        ((T::CACHE_A == 1 && T::CACHE_B == 17) || (T::T_N == 1 && T::SCALE_PANEL == 128 &&
          T::CACHE_A == 0 && T::CACHE_B == 0)) && T::GROUPED_M == 0;
    constexpr bool normal_narrow_raw = !ShortK && T::B_M == 32 && T::B_N == 32 && T::B_K == 256 && T::T_M == 2 && T::T_N == 1 && T::SCALE_PANEL == 128 && T::SPLIT_K == 1 && T::CACHE_A == 0 && T::CACHE_B == 0 && T::GROUPED_M == 0;
    constexpr bool normal_panel128 = !ShortK && T::B_M == 16 && T::B_N == 32 && T::B_K == 128 && T::T_M == 1 && T::T_N == 2 && T::SCALE_PANEL == 128 && T::SPLIT_K == 1 && T::CACHE_A == 0 && T::CACHE_B == 0 && T::GROUPED_M == 0;
    const int stages = ShortK ? loops : (ring8 ? 8 : (normal_stage4_profile ? 4 :
        (T::B_M == 128 && T::B_N == 128 && T::T_M == 4 && T::T_N == 2
            ? T::normal_stage_count(kargs.k) : T::NUM_STAGES)));
    constexpr bool raw_a_stage = !ShortK && ((normal_a1_overlap) ||
        (normal_narrow_raw) ||
        (T::B_M == 80 && T::T_M == 1 && T::T_N == 4) ||
        (T::B_M == 96 && T::B_N == 64) || (T::B_M == 128 && T::B_N == 128 && T::T_M == 2 && T::T_N == 4));
    constexpr bool raw_resident_a = ShortK && T::B_M > 32;
    constexpr int a_stage_bytes = raw_resident_a ? T::A_RESIDENT_STAGE :
        (raw_a_stage ? T::HALF_B_M * T::B_K : T::A_STAGE);
    const int matrix_lds_bytes = stages * (a_stage_bytes + T::B_STAGE);
    __builtin_assume(total_k > 0 && total_k % (T::SPLIT_K * T::B_K) == 0);
    __builtin_assume(part_k / T::GROUP_K <= T::SCALE_PANEL);
    if constexpr (ShortK) __builtin_assume(loops <= capacity);
    else if constexpr (T::resident_capacity(1) != T::resident_capacity(0)) __builtin_assume(ring8 || loops > T::resident_capacity(kargs.m));
    else __builtin_assume(ring8 || loops > capacity);
    // Global strides retain total K; grid.z selects a K partition.
    const int valid_rows = kargs.m - row < T::B_M ? kargs.m - row : T::B_M;
    auto g_a = make_gmem( reinterpret_cast<const D_A*>(kargs.ptr_a) + row * kargs.stride_a + part_start,
        static_cast<unsigned>(valid_rows * kargs.stride_a - part_start));
    auto g_b = make_gmem( reinterpret_cast<const D_B*>(kargs.ptr_b) + col * kargs.stride_b + part_start * T::W_N,
        static_cast<unsigned>(T::B_N * kargs.stride_b - part_start * T::W_N));
    auto g_c = make_gmem( reinterpret_cast<D_C*>(kargs.ptr_c) + static_cast<int64_t>(split) * kargs.m * kargs.n +
            static_cast<int64_t>(row) * kargs.stride_c + col, static_cast<unsigned>(((valid_rows - 1) * kargs.stride_c + T::B_N) * sizeof(D_C)));
    const int scale_k_groups = part_k / T::GROUP_K;
    auto g_sfa = make_gmem( reinterpret_cast<const D_SF*>(kargs.ptr_sfa) + split * scale_k_groups * kargs.stride_sfa + row,
        static_cast<unsigned>(scale_k_groups * kargs.stride_sfa - row));
    auto g_sfb = make_gmem( reinterpret_cast<const D_SF*>(kargs.ptr_sfb) + (col / T::GROUP_N) * kargs.stride_sfb +
            split * scale_k_groups, static_cast<unsigned>(T::SCALE_N_HALVES * kargs.stride_sfb - split * scale_k_groups));
    auto u_ga = make_layout_ga_scale<T>(lane_id, wave_id_m, wave_id_n, kargs.stride_a);
    auto u_sa = [&] {
        const auto logical = make_layout_sa_scale<T>(wave_id_m, wave_id_n);
        if constexpr (raw_a_stage) {
            auto result = make_layout<T::VEC_A>(logical.shape(), logical.stride(), logical.coord());
            static_for<decltype(result)::num_issues>([&](auto i) {
                result.offsets[i] = (int(i) * T::NUM_WAVES + wave_id) * T::smem_linear_wave;
            });
            return result;
        } else
            return logical;
    }();
    auto u_ra = make_layout_ra_scale<T>(lane_id, wave_id_m);
    if constexpr (raw_a_stage) {
        constexpr int threads_k = T::B_K / T::VEC_A;
        static_for<decltype(u_ga)::num_issues>([&](auto i) {
            const int m = (int(i) * T::NUM_WAVES + wave_id) * (T::WARP_SIZE / threads_k) +
                lane_id / threads_k;
            const int k = (lane_id % threads_k) * T::VEC_A;
            if constexpr (T::B_M != 80) u_ga.offsets[i] = m * kargs.stride_a + (k ^ ((m % threads_k) * T::VEC_A));
            else u_ga.offsets[i] += (k ^ ((m % threads_k) * T::VEC_A)) - k;
        });
        static_for<decltype(u_ra)::num_issues>([&](auto i) {
            constexpr int mr = int(i) / (T::E_K * T::A_CHUNKS_PER_FRAGMENT);
            constexpr int ks = int(i) / T::A_CHUNKS_PER_FRAGMENT % T::E_K;
            constexpr int half = int(i) % T::A_CHUNKS_PER_FRAGMENT;
            const int m = (mr * T::T_M + wave_id_m) * T::W_M + lane_id % T::W_M;
            const int kv = ks * (T::W_K / T::VEC_A) +
                half * (T::WARP_SIZE / T::W_M) + lane_id / T::W_M;
            u_ra.offsets[i] = m * T::B_K + (kv ^ (m % threads_k)) * T::VEC_A;
        });
    }
    const auto u_gb = make_layout_gb_scale<T>(lane_id, wave_id_m, wave_id_n, kargs.stride_b);
    const auto u_sb = make_layout_sb_scale<T>(wave_id_m, wave_id_n);
    const auto u_rb = make_layout_rb_scale<T>(lane_id, wave_id_n);
    const auto u_gsfa = make_layout_gsfa_scale<T>(lane_id, wave_id_m, wave_id_n, kargs.stride_sfa);
    const auto u_ssfa = make_layout_ssfa_scale<T>(lane_id, wave_id_m, wave_id_n);
    const auto u_rsfa = make_layout_rsfa_scale<T>(lane_id, wave_id_m);
    const int sfb_thread = wave_id * T::WARP_SIZE + lane_id;
    const auto u_gsfb = make_layout<1>( make_tuple(number<T::SCALE_N_HALVES>{}, 1_I), make_tuple(kargs.stride_sfb, 1_I), make_tuple(underscore{}, sfb_thread));
    const auto u_ssfb = make_layout<1>( make_tuple(1_I, 1_I), make_tuple(1_I, 1_I), make_tuple(sfb_thread, underscore{}));
    const auto u_rsfb = make_layout<1>(make_tuple(1_I), make_tuple(1_I));
    alignas(16) extern __shared__ char smem_matrix[];
    auto s_a = make_smem(reinterpret_cast<D_A*>(smem_matrix));
    auto s_b = make_smem(reinterpret_cast<D_B*>(smem_matrix + stages * a_stage_bytes));
    auto s_c = make_smem(reinterpret_cast<D_C*>(smem_matrix));
    auto s_sfa = make_smem(reinterpret_cast<D_SF*>(smem_matrix + matrix_lds_bytes));
    auto s_sfb = make_smem(reinterpret_cast<D_SF_PACK*>(smem_matrix + matrix_lds_bytes + (!ShortK && T::B_M == 128 && T::B_N == 128 && T::T_M == 4 && T::T_N == 2
        ? T::normal_sfa_bytes(kargs.k) : T::SFA_BYTES)));
    constexpr bool native_short = ShortK && T::B_N == 64 && T::B_K == 128 &&
        T::T_N == 2 && T::SPLIT_K == 1 && T::GROUPED_M == 0 &&
        ((T::T_M == 1 && T::B_M == 48 && T::SCALE_PANEL == 8 && capacity == 6 && T::CACHE_A == 17 && T::CACHE_B == 1) ||
         ((T::T_M == 1 && T::B_M == 32) || (T::T_M == 2 && T::B_M == 64)) &&
            T::SCALE_PANEL == 4 && capacity == 3 && T::CACHE_A == 1 && T::CACHE_B == 17);
    using Adaptor = std::conditional_t<native_short, mfma_adaptor, mfma_adaptor_swap_ab>;
    auto mma = make_tiled_mma<D_A, D_B, D_ACC>( seq<T::E_M, T::E_N, T::E_K>{}, seq<T::T_M, T::T_N, T::T_K>{},
        seq<T::W_M, T::W_N, T::W_K>{}, Adaptor{});
    using BaseMMA = typename decltype(mma)::MMA;
    array<array<typename BaseMMA::vtype_a, T::A_RING>, T::E_K> v_a;
    array<array<typename BaseMMA::vtype_b, T::E_N>, T::E_K> v_b;
    array<typename BaseMMA::vtype_c, T::E_M * T::E_N> v_c{};
    constexpr int store_vec_c = native_short ? 1 : T::VEC_C;
    const auto p_coord_c = opus::make_tuple(wave_id_m,
        native_short ? lane_id / mma.grpn_c : lane_id % mma.grpn_c, wave_id_n,
        native_short ? lane_id % mma.grpn_c : lane_id / mma.grpn_c);
    const auto u_sc = partition_layout_c<store_vec_c>(mma, opus::make_tuple(number<T::C_LDS_ROW_STRIDE_ELEMS>{}, 1_I), p_coord_c);
    const auto sc_offsets = layout_to_offsets<store_vec_c>(u_sc);
    array<array<D_SF_PACK, T::A_SCALE_PACKS>, T::E_K> v_sfa{}, v_sfa_next{};
    array<array<D_SF_PACK, T::B_SCALE_PACKS>, T::E_K> v_sfb{}, v_sfb_next{};
    auto ga_offset = [&](int tile_k) { return tile_k * number<T::B_K>{};
    };
    auto sa_offset = [&](int stage) { return stage * a_stage_bytes;
    };
    auto sb_offset = [&](int stage) { return stage * T::B_STAGE;
    };
    auto ssfa_offset = [&](int tile_k) { return tile_k * number<T::SFA_PANEL_PITCH>{};
    };
    auto issue_matrix_prefetch = [&](auto stage_i, int tile_k) {
        const int stage = static_cast<int>(stage_i);
        constexpr bool typed_resident_six = raw_resident_a && T::B_M == 48 && T::B_N == 32 &&
            T::B_K == 128 && T::T_M == 1 && T::T_N == 2 && T::SCALE_PANEL == 8 && capacity == 6;
        if constexpr ((T::B_M == 16 && T::B_N == 32 && ((ShortK && capacity == 6 && T::B_K == 128 && T::T_M == 1 && T::T_N == 2 && T::SCALE_PANEL == 8) ||
             (T::B_K == 512 && T::NUM_WAVES == 2 && T::E_M == 1) ||
             (!ShortK && T::B_K == 128 && T::T_M == 1 && T::T_N == 2 && T::SCALE_PANEL == 128 &&
              T::SPLIT_K == 1 && T::CACHE_A == 0 && T::CACHE_B == 0 && T::GROUPED_M == 0))) ||
            (ShortK && T::B_M == 32 && T::B_N == 32 && T::B_K == 512 && capacity == 3 && T::NUM_WAVES == 2 && T::E_M == 1) ||
            (ShortK && T::B_M == 32 && T::B_N == 64 && T::B_K == 128 && T::T_M == 1 && T::T_N == 2 &&
             T::SCALE_PANEL == 4 && capacity == 3 && T::SPLIT_K == 1 && T::CACHE_A == 1 && T::CACHE_B == 17 && T::GROUPED_M == 0)) {
            const auto ga = layout_to_offsets<T::VEC_A>(u_ga);
            const auto sa = layout_to_offsets<T::VEC_A>(u_sa);
            const auto gb = layout_to_offsets<T::VEC_B>(u_gb);
            const auto sb = layout_to_offsets<T::VEC_B>(u_sb);
            static_for<decltype(ga)::size()>([&](auto copy_i) {
                constexpr int i = decltype(copy_i)::value;
                g_a.template _async_load<T::VEC_A>(s_a.ptr + sa[i] + sa_offset(stage), ga[i], ga_offset(tile_k), 0_I, number<T::CACHE_A>{});
            });
            static_for<decltype(gb)::size()>([&](auto copy_i) {
                constexpr int i = decltype(copy_i)::value;
                g_b.template _async_load<T::VEC_B>(s_b.ptr + sb[i] + sb_offset(stage), gb[i], tile_k * number<T::B_K * T::W_N>{}, 0_I, number<T::CACHE_B>{});
            });
            return;
        }
        if constexpr (raw_resident_a) {
            static_for<T::HALF_B_M * T::B_K / (T::BLOCK_SIZE * T::VEC_A)>([&](auto copy_i) {
                constexpr int threads_k = T::B_K / T::VEC_A;
                const int copy_wave = decltype(copy_i)::value * T::NUM_WAVES + wave_id;
                const int local_row = copy_wave * (T::WARP_SIZE / threads_k) + lane_id / threads_k;
                const int offset = local_row * kargs.stride_a +
                    ((lane_id % threads_k) ^ (local_row % threads_k)) * T::VEC_A;
                if constexpr (typed_resident_six)
                    g_a.template _async_load<T::VEC_A>(s_a.ptr + sa_offset(stage) + copy_wave * T::smem_linear_wave, offset, ga_offset(tile_k), 0_I, number<T::CACHE_A>{});
                else
                async_load<T::VEC_A>(g_a, s_a.ptr + sa_offset(stage) + copy_wave * T::smem_linear_wave, offset, ga_offset(tile_k), 0_I, number<T::CACHE_A>{});
            });
        } else
            async_load<T::VEC_A>(g_a, s_a.ptr, u_ga, u_sa + sa_offset(stage), ga_offset(tile_k), opus::number<0>{}, opus::number<T::CACHE_A>{});
        if constexpr (typed_resident_six) {
            const auto gb = layout_to_offsets<T::VEC_B>(u_gb);
            const auto sb = layout_to_offsets<T::VEC_B>(u_sb);
            static_for<decltype(gb)::size()>([&](auto copy_i) {
                constexpr int i = decltype(copy_i)::value;
                g_b.template _async_load<T::VEC_B>(s_b.ptr + sb[i] + sb_offset(stage), gb[i], tile_k * number<T::B_K * T::W_N>{}, 0_I, number<T::CACHE_B>{});
            });
        } else
        async_load<T::VEC_B>(g_b, s_b.ptr, u_gb, u_sb + sb_offset(stage), tile_k * number<T::B_K * T::W_N>{}, opus::number<0>{}, opus::number<T::CACHE_B>{});
    };
    constexpr bool full_resident = ShortK && T::B_K == 128 && T::T_N == 2 &&
        (((T::B_M == 64 || T::B_M == 96) && T::B_N == 64 && T::T_M == 2 && T::SCALE_PANEL == 4 && capacity == 3) ||
         (T::B_M == 48 && T::B_N <= 64 && T::T_M == 1 && T::SCALE_PANEL == 8 && capacity == 6) ||
         (T::B_M == 32 && T::B_N == 64 && T::T_M == 1 &&
          ((T::SCALE_PANEL == 8 && capacity == 6) || (T::SCALE_PANEL == 4 && capacity == 3)) &&
          T::SPLIT_K == 1 && T::CACHE_A == 1 && T::CACHE_B == 17 && T::GROUPED_M == 0));
    constexpr bool c_stage_overlap = full_resident && T::B_M > 32 && capacity == 3 && T::SPLIT_K == 1 && T::E_K == 1 && T::E_N == 2 && (T::E_M == 2 || T::E_M == 3);
    constexpr bool resident_overlap = c_stage_overlap || (full_resident &&
        (capacity == 6 || (T::B_M == 32 && capacity == 3)) &&
        T::SPLIT_K == 1 && T::E_K == 1 && T::E_N <= 2 && (T::E_M == 2 || T::E_M == 3));
    static_assert(!c_stage_overlap || T::C_BYTES <= 2 * T::A_RESIDENT_STAGE);
    constexpr bool raw_sfa_lds = T::B_N <= 128 && (T::B_M <= 32 || full_resident ||
        (!ShortK && T::B_M == 64 && T::E_M <= 2 && T::SFA_PASSES_PER_CACHE_PANEL > 1));
    const auto sfa_gmem_offsets = layout_to_offsets<T::VEC_SCALE_A>(u_gsfa);
    const auto sfa_smem_offsets = layout_to_offsets<T::VEC_SF>(u_ssfa);
    const auto sfb_gmem_offsets = layout_to_offsets<1>(u_gsfb);
    const auto sfb_smem_offsets = layout_to_offsets<1>(u_ssfb);
    const auto rsfa_offsets = layout_to_offsets<T::VEC_SF>(u_rsfa);
    const auto rsfb_offsets = layout_to_offsets<1>(u_rsfb);
    constexpr bool register_short_scales = ShortK && T::B_M <= 32;
    constexpr bool deferred_short_a_pack = ShortK && T::B_M == 32 && T::B_N == 64 &&
        T::B_K == 128 && T::T_M == 1 && T::T_N == 2 &&
        ((T::SCALE_PANEL == 4 && capacity == 3) || (T::SCALE_PANEL == 8 && capacity == 6)) &&
        T::SPLIT_K == 1 && T::CACHE_A == 1 && T::CACHE_B == 17 && T::GROUPED_M == 0;
    constexpr bool panel_short = register_short_scales && T::B_K <= 512 && (capacity == 3 || (capacity == 2 && T::B_M == 16 && T::B_N == 32 && T::B_K == 512 && T::NUM_WAVES == 2) || (T::B_M == 16 && T::B_N == 32 && T::B_K == 128 && T::T_M == 1 && T::T_N == 2 && T::SCALE_PANEL == 8 && capacity == 6))
        && T::E_M == 1 && T::SCALE_N_HALVES == 1;
    constexpr bool compact_panel_short = panel_short && T::B_K == 128;
    constexpr bool compact_raw_scales = raw_sfa_lds;
    array<array<D_SF_PACK, (full_resident || normal_a1_overlap) ? T::E_M : 1>, T::E_K> raw_sfa{};
    constexpr bool typed_normal_scales = !ShortK && T::B_M == 16 && T::B_N == 32 && T::B_K == 512 && T::NUM_WAVES == 2 && T::E_M == 1;
    constexpr int resident_groups = register_short_scales ? capacity * T::E_K : 1;
    array<array<D_SF_PACK, T::A_SCALE_PACKS>, resident_groups> resident_sfa{};
    array<array<D_SF_PACK, T::B_SCALE_PACKS>, resident_groups> resident_sfb{};
    array<array<D_SF_PACK, deferred_short_a_pack ? T::E_M : 1>, resident_groups> resident_sfa_raw{};
    vector_t<D_SF, T::VEC_SCALE_A> sfa_panel_raw[T::SFA_PASSES_PER_CACHE_PANEL];
    auto pack_a_scales = [&](auto pack_i, auto read_row) {
        constexpr int pack = decltype(pack_i)::value;
        D_SF_PACK word = 0;
        static_for<T::VEC_SF>([&](auto byte_i) {
            constexpr int byte = decltype(byte_i)::value, repeat = pack * T::VEC_SF + byte;
            if constexpr (repeat < T::E_M) {
                const int local_row = (repeat * T::T_M + wave_id_m) * T::W_M + lane_id % T::W_M;
                word |= static_cast<D_SF_PACK>(read_row(local_row)) << (byte * 8);
            }
        });
        return word;
    };
    auto load_scale_vector = [&](auto& value, int offset, int local_row) {
        constexpr int size = sizeof(value) / sizeof(D_SF);
        value = {};
        if (kargs.m % size == 0) {
            value = load<size>(g_sfa, offset);
            return;
        }
        static_for<size>([&](auto byte_i) {
            constexpr int byte = decltype(byte_i)::value;
            if (row + local_row + byte < kargs.m) value[byte] = load<1>(g_sfa, offset + byte)[0];
        });
    };
    auto load_sfa_panel = [&](int panel_k_begin) {
        if constexpr (register_short_scales) {
            static_for<resident_groups>([&](auto group_i) {
                constexpr int group = decltype(group_i)::value;
                if constexpr (deferred_short_a_pack) {
                    static_for<T::E_M>([&](auto m_i) {
                        constexpr int mr = decltype(m_i)::value;
                        const int local_row = (mr * T::T_M + wave_id_m) * T::W_M + lane_id % T::W_M;
                        resident_sfa_raw[group][mr] = load<1>(g_sfa, group * kargs.stride_sfa + local_row)[0];
                    });
                } else {
                    static_for<T::A_SCALE_PACKS>([&](auto pack_i) {
                        resident_sfa[group][decltype(pack_i)::value] = pack_a_scales(pack_i, [&](int local_row) {
                            return load<1>(g_sfa, group * kargs.stride_sfa + local_row)[0];
                        });
                    });
                }
            });
            return;
        }
        if (compact_raw_scales || (ShortK && T::B_M <= 64 && kargs.m <= 32)) {
            const int thread = wave_id * T::WARP_SIZE + lane_id;
            auto copy_aligned_panel = [&](auto vector_size, auto typed) {
                constexpr int vec = decltype(vector_size)::value, vectors = T::RAW_SFA_ROWS / vec;
                constexpr int groups = ShortK ? capacity : T::SCALE_PANEL;
                static_for<(groups * vectors + T::BLOCK_SIZE - 1) / T::BLOCK_SIZE>([&](auto pass_i) {
                    const int linear = thread + decltype(pass_i)::value * T::BLOCK_SIZE;
                    if (linear < scale_k_groups * vectors) {
                        const int group = linear / vectors, local_row = linear % vectors * vec;
                        const int offset = row + local_row < kargs.m ? group * kargs.stride_sfa + local_row : 0x7fffffff;
                        if constexpr (decltype(typed)::value) g_sfa.template _async_load<vec>(s_sfa.ptr + (linear - lane_id) * vec, offset);
                        else async_load<vec>(g_sfa, s_sfa.ptr + (linear - lane_id) * vec, offset);
                    }
                });
            };
            if constexpr ((full_resident && T::B_M == 48 && T::B_N == 32) ||
                (normal_panel128) ||
                (normal_narrow_raw)) {
                if (kargs.m % 16 == 0) { copy_aligned_panel(16_I, 0_I);
                return;
                }
            }
            if constexpr (aligned_normal_scales) {
                if (kargs.m % 4 == 0) { copy_aligned_panel(4_I, 1_I);
                return;
                }
            }
#pragma clang loop unroll(disable)
            for (int linear = thread; linear < scale_k_groups * (T::RAW_SFA_ROWS / 4);
                 linear += T::BLOCK_SIZE) {
                const int group = linear / (T::RAW_SFA_ROWS / 4);
                const int local_row = linear % (T::RAW_SFA_ROWS / 4) * 4;
                const int offset = row + local_row < kargs.m
                    ? group * kargs.stride_sfa + local_row : 0x7fffffff;
                if (kargs.m % 4 == 0) {
                    if constexpr (typed_normal_scales) g_sfa.template _async_load<4>(s_sfa.ptr + (linear - lane_id) * sizeof(unsigned), offset);
                    else async_load<4>(g_sfa, s_sfa.ptr + (linear - lane_id) * sizeof(unsigned), offset);
                } else if (T::B_M <= 32 && kargs.m == 2) {
                    async_load<2>(g_sfa, s_sfa.ptr + (linear - lane_id) * sizeof(unsigned), offset);
                } else {
                    vector_t<D_SF, 4> value{};
                    load_scale_vector(value, offset, local_row);
                    store<4>(s_sfa, value, linear * sizeof(unsigned));
                }
            }
            return;
        }
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
            if (lane_k < T::SFA_K_COLUMNS_PER_WAVE && k_group < T::SCALE_PANEL && k_group < scale_k_groups && m_repeat < T::E_M && row + local_row < kargs.m) {
                const int offset = sfa_gmem_offsets[pass] + panel_k_begin * kargs.stride_sfa;
                load_scale_vector(sfa_panel_raw[pass], offset, local_row);
            }
        });
    };
    // The 4wave 4x4 byte transpose packs repeats separated by T_M * W_M.
    auto publish_sfa_panel = [&](int panel_k_begin) {
        if constexpr (register_short_scales || compact_raw_scales) return;
        if (ShortK && T::B_M <= 64 && kargs.m <= 32) return;
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
                const auto raw_words = __builtin_bit_cast( vector_t<D_SF_PACK, T::VEC_SCALE_A / T::VEC_SF>, sfa_panel_raw[pass]);
                static_for<T::VEC_SCALE_A / T::VEC_SF>([&](auto word_i) {
                    constexpr int word = decltype(word_i)::value;
                    const D_SF_PACK packed_scales = transpose_sfa_word_4x4(raw_words[word]);
                    store<T::VEC_SF>(s_sfa, __builtin_bit_cast(vector_t<D_SF, T::VEC_SF>, packed_scales),
                        sfa_smem_offsets[pass] + ssfa_offset(panel_k_begin) + word * T::VEC_SF * T::VEC_SF);
                });
            }
        });
    };
    auto load_sfb_panel = [&](int panel_k_begin) {
        if constexpr (register_short_scales) {
            static_for<resident_groups>([&](auto group_i) {
                constexpr int group = decltype(group_i)::value;
                { // Unused resident groups remain protected by the buffer descriptor.
                    const unsigned lo = load<1>(g_sfb, group)[0];
                    unsigned hi = 0;
                    if constexpr (T::SCALE_N_HALVES == 2) hi = load<1>(g_sfb, kargs.stride_sfb + group)[0];
                    resident_sfb[group][0] = lo | (hi << 8);
                }
            });
            return;
        }
        const int k_group = panel_k_begin + wave_id * T::WARP_SIZE + lane_id;
        if (k_group < scale_k_groups) {
            if constexpr (aligned_normal_scales || (normal_panel128)) {
                // smem::ptr uses byte offsets; each K128 scale occupies one LDS dword.
                g_sfb.template _async_load<1>(s_sfb.ptr + wave_id * T::WARP_SIZE * sizeof(D_SF_PACK), sfb_gmem_offsets[0] + panel_k_begin);
                return;
            }
            if constexpr (T::SCALE_N_HALVES == 1) {
                if (full_resident || ((ShortK || T::B_M <= 32) && T::B_M <= 64 && kargs.m <= 32 && (T::B_M > 32 || scale_k_groups <= T::WARP_SIZE))) {
                    if constexpr (typed_normal_scales) g_sfb.template _async_load<1>(s_sfb.ptr + wave_id * T::WARP_SIZE, sfb_gmem_offsets[0] + panel_k_begin);
                    else { async_load<1>(g_sfb, s_sfb.ptr + wave_id * T::WARP_SIZE, sfb_gmem_offsets[0] + panel_k_begin);
                    }
                    return;
                }
            }
            const unsigned lo = load<1>(g_sfb, sfb_gmem_offsets[0] + panel_k_begin)[0];
            unsigned hi = 0;
            if constexpr (T::SCALE_N_HALVES == 2) hi = load<1>(g_sfb, sfb_gmem_offsets[1] + panel_k_begin)[0];
            const vector_t<D_SF_PACK, 1> packed{lo | (hi << 8)};
            store<1>(s_sfb, packed, sfb_smem_offsets[0]);
        }
    };
    auto read_raw_a_scale = [&](auto m_i, int group) { const int local_row = (decltype(m_i)::value * T::T_M + wave_id_m) * T::W_M + lane_id % T::W_M;
    return load<1>(s_sfa, group * T::RAW_SFA_ROWS + local_row)[0];
    };
    auto resident_a_scales = [&](int group) {
        if constexpr (deferred_short_a_pack) {
            // Both byte loads are complete at the resident FIFO wait.
            __builtin_amdgcn_sched_barrier(0);
            array<D_SF_PACK, T::A_SCALE_PACKS> value{};
            value[0] = resident_sfa_raw[group][0] | (resident_sfa_raw[group][1] << 8);
            return value;
        } else return resident_sfa[group];
    };
    auto read_scales = [&](int tile_k, auto& scale_a, auto& scale_b) {
        static_for<T::E_K>([&](auto k_i) {
            constexpr int ks = decltype(k_i)::value;
            const int group = tile_k * T::E_K + ks;
            if constexpr (register_short_scales) {
                scale_a[ks] = resident_a_scales(group);
                scale_b[ks] = resident_sfb[group];
            } else {
                if constexpr (full_resident || normal_a1_overlap) {
                    static_for<T::E_M>([&](auto m_i) {
                        constexpr int mr = decltype(m_i)::value;
                        raw_sfa[ks][mr] = read_raw_a_scale(m_i, group);
                    });
                } else if (compact_raw_scales || (ShortK && T::B_M <= 64 && kargs.m <= 32)) {
                    static_for<T::A_SCALE_PACKS>([&](auto pack_i) {
                        scale_a[ks][decltype(pack_i)::value] = pack_a_scales(pack_i, [&](int local_row) {
                            return local_row < T::RAW_SFA_ROWS ? load<1>(s_sfa, group * T::RAW_SFA_ROWS + local_row)[0] : D_SF{};
                        });
                    });
                } else {
                    static_for<T::A_SCALE_PACKS>([&](auto pack_i) {
                        constexpr int pack = decltype(pack_i)::value;
                        scale_a[ks][pack] = __builtin_bit_cast(D_SF_PACK, load<T::VEC_SF>(s_sfa, rsfa_offsets[pack] + ssfa_offset(group)));
                    });
                }
                scale_b[ks][0] = load<1>(s_sfb, rsfb_offsets[0] + group)[0];
            }
        });
    };
    auto read_a_fragment = [&](auto m_i, auto stage_i, auto k_i) {
        constexpr int mr = decltype(m_i)::value, ks = decltype(k_i)::value;
        if constexpr (raw_resident_a) {
            const int local_row = (mr * T::T_M + wave_id_m) * T::W_M + lane_id % T::W_M;
            const int base = sa_offset(static_cast<int>(stage_i)) + local_row * T::B_K;
            const int swizzle = local_row % (T::B_K / T::VEC_A);
            const int k_vector = ks * (T::W_K / T::VEC_A) + lane_id / T::W_M;
            const array<int, 2> offsets{base + ((k_vector ^ swizzle) * T::VEC_A), base + (((k_vector + 4) ^ swizzle) * T::VEC_A)};
            return load_operand_fragment_staged<T::VEC_A, 0, typename BaseMMA::vtype_a>(s_a, offsets);
        } else {
            const auto offsets = layout_to_offsets<T::VEC_A>(u_ra + sa_offset(static_cast<int>(stage_i)));
            return load_operand_fragment_staged<
                T::VEC_A, (mr * T::E_K + ks) * T::A_CHUNKS_PER_FRAGMENT, typename BaseMMA::vtype_a>(s_a, offsets);
        }
    };
    auto load_a_fragment = [&](auto m_i, auto stage_i, auto k_i) { v_a[decltype(k_i)::value][decltype(m_i)::value % T::A_RING] = read_a_fragment(m_i, stage_i, k_i);
    };
    auto load_b_fragment = [&](auto n_i, auto stage_i, auto k_i) {
        constexpr int nr = decltype(n_i)::value, ks = decltype(k_i)::value;
        const auto offsets = layout_to_offsets<T::VEC_B>(u_rb + sb_offset(static_cast<int>(stage_i)));
        if constexpr (T::B_N < T::NUM_WAVES * T::W_N)
            v_b[ks][nr] = load_operand_fragment_staged<T::VEC_B, (nr * T::E_K + ks) * T::B_CHUNKS_PER_FRAGMENT, typename BaseMMA::vtype_b>(s_b, offsets);
        else
            v_b[ks][nr] = load_operand_fragment_staged<
                T::VEC_B, nr * T::B_CHUNKS_PER_FRAGMENT * T::E_K + ks, typename BaseMMA::vtype_b, T::E_K>(s_b, offsets);
    };
    auto load_tile_fragments = [&](auto stage_i) {
        static_for<T::E_K>([&](auto k_i) {
            static_for<T::A_RING>([&](auto m_i) {
                load_a_fragment(m_i, stage_i, k_i);
            });
            static_for<T::E_N>([&](auto n_i) {
                load_b_fragment(n_i, stage_i, k_i);
            });
        });
    };
    auto mma_a_scale_fragment = [&](auto m_i, auto n_i, auto k_i, const auto& a_fragment) {
        constexpr int mr = decltype(m_i)::value, nr = decltype(n_i)::value;
        constexpr int ks = decltype(k_i)::value, ci = mr * T::E_N + nr;
        constexpr int sn = nr / (T::GROUP_N / (T::T_N * T::W_N));
        if constexpr ((full_resident || normal_a1_overlap) && !register_short_scales)
            v_c[ci] = BaseMMA{}(a_fragment, v_b[ks][nr], v_c[ci], static_cast<int>(raw_sfa[ks][mr]), static_cast<int>(v_sfb[ks][0]), 0_I, number<sn>{});
        else
            v_c[ci] = BaseMMA{}(a_fragment, v_b[ks][nr], v_c[ci], static_cast<int>(v_sfa[ks][mr / T::VEC_SF]), static_cast<int>(v_sfb[ks][0]),
                number<mr % T::VEC_SF>{}, number<sn>{});
    };
    auto mma_scale_fragment = [&](auto m_i, auto n_i, auto k_i) { mma_a_scale_fragment(m_i, n_i, k_i, v_a[decltype(k_i)::value][decltype(m_i)::value % T::A_RING]);
    };
    auto advance_tile = [&](auto stage_i, auto next_stage_i, int tile_k, auto pending) {
        constexpr int next_stage = decltype(next_stage_i)::value;
        if constexpr (!(T::B_M <= 32 && T::B_N <= 128)) read_scales(tile_k + 1, v_sfa_next, v_sfb_next);
        static_for<T::E_M>([&](auto m_i) {
            constexpr int m = decltype(m_i)::value;
            s_waitcnt_lgkmcnt(0_I);
            if constexpr (m == T::E_M - T::A_RING) {
                if (tile_k + stages < loops) s_waitcnt_vmcnt(pending);
                else s_waitcnt_vmcnt(0_I);
                if constexpr (T::NUM_WAVES > 1) sync_threads();
                if (tile_k + stages < loops) issue_matrix_prefetch(stage_i, tile_k + stages);
                __builtin_amdgcn_sched_barrier(0);
            }
            static_for<T::E_K>([&](auto k_i) {
                static_for<T::E_N>([&](auto n_i) {
                    mma_scale_fragment(m_i, n_i, k_i);
                    if constexpr (m == T::E_M - 1) load_b_fragment(n_i, number<next_stage>{}, k_i);
                });
                if constexpr (m + T::A_RING < T::E_M)
                    load_a_fragment(number<m + T::A_RING>{}, stage_i, k_i);
            });
        });
        static_for<T::E_K>([&](auto k_i) {
            static_for<T::A_RING>([&](auto m_i) {
                load_a_fragment(m_i, number<next_stage>{}, k_i);
            });
        });
        if constexpr (T::B_M <= 32 && T::B_N <= 128) read_scales(tile_k + 1, v_sfa_next, v_sfb_next);
        v_sfa = v_sfa_next;
        v_sfb = v_sfb_next;
    };
    auto finish_tile = [&](auto stage_i) {
        static_for<T::E_M>([&](auto m_i) {
            constexpr int m = decltype(m_i)::value;
            constexpr bool first_mfma_wait4 = ShortK && T::B_M == 16 && T::B_N == 32 && T::B_K == 256 && T::E_M == 1 && T::E_N == 1;
            s_waitcnt_lgkmcnt(number<first_mfma_wait4 ? 4 : (normal_a1_overlap && m == 0 ? 2 : 0)>{});
            static_for<T::E_K>([&](auto k_i) {
                if constexpr (first_mfma_wait4 && decltype(k_i)::value > 0) s_waitcnt_lgkmcnt(0_I);
                static_for<T::E_N>([&](auto n_i) {
                    mma_scale_fragment(m_i, n_i, k_i);
                });
                if constexpr (m + T::A_RING < T::E_M) load_a_fragment(number<m + T::A_RING>{}, stage_i, k_i);
            });
        });
    };
    auto stage_output_fragment = [&](auto c_i) { if constexpr (!native_short) { constexpr int c_index = decltype(c_i)::value;
    const auto c_fragment = v_c[c_index];
    store<T::VEC_C>(s_c, cast<D_C>(c_fragment), sc_offsets[c_index]);
    } };
    const int output_thread_id = wave_id * T::WARP_SIZE + lane_id;
    constexpr bool bounded_normal_c = !ShortK && T::B_M == 48 && T::B_N == 64 && T::B_K == 128 &&
        T::T_M == 1 && T::T_N == 2 && T::SCALE_PANEL == 16 && T::SPLIT_K == 1 &&
        T::CACHE_A == 0 && T::CACHE_B == 0 && T::GROUPED_M == 0;
    auto write_output_vector = [&](int pass) {
        const int linear = output_thread_id * T::VEC_OUTPUT + pass * T::BLOCK_SIZE * T::VEC_OUTPUT;
        const int output_row = linear / T::B_N;
        const int output_col = linear % T::B_N;
        if (linear < T::B_M * T::B_N && (bounded_normal_c || (!ShortK && T::B_M == 256 && T::B_N == 256 && T::B_K == 128 && T::T_M == 4 && T::T_N == 2 && T::SCALE_PANEL == 64 && T::SPLIT_K == 1 && T::CACHE_A == 0 && T::CACHE_B == 0 && T::GROUPED_M == 0) || c_stage_overlap || row + output_row < kargs.m)) {
            const auto value = load<T::VEC_OUTPUT>(s_c, output_row * T::C_LDS_ROW_STRIDE_ELEMS + output_col);
            store<T::VEC_OUTPUT>( g_c, value, output_row * kargs.stride_c + output_col, 0, opus::number<(full_resident || bounded_normal_c) ? 0 : 2>{});
        }
    };
    auto write_global_c = [&](auto cache_control) {
        const auto u_gc = partition_layout_c<store_vec_c>(mma, opus::make_tuple(kargs.stride_c, 1_I), p_coord_c);
        const auto gc_offsets = layout_to_offsets<store_vec_c>(u_gc);
        static_for<T::E_M * T::E_N>([&](auto c_i) {
            constexpr int index = decltype(c_i)::value;
            const auto value = cast<D_C>(v_c[index]);
            if constexpr (native_short) {
                static_for<T::VEC_C>([&](auto i) {
                    constexpr int offset = (index / T::E_N * T::VEC_C + decltype(i)::value) * T::E_N + index % T::E_N;
                    store<1>(g_c, value[decltype(i)::value], gc_offsets[offset], 0, cache_control);
                });
            } else store<T::VEC_C>(g_c, value, gc_offsets[index], 0, cache_control);
        });
    };
    if (!small_full_two && (!panel_short || (!compact_panel_short && loops != capacity))) {
    issue_matrix_prefetch(0_I, 0);
    if constexpr (ShortK && T::B_M == 16 && T::B_K >= 256) {
        static_for<capacity>([&](auto stage_i) {
            constexpr int stage = decltype(stage_i)::value;
            if (stage > 0 && stage < loops) issue_matrix_prefetch(stage_i, stage);
        });
    }
    __builtin_amdgcn_sched_barrier(0);
    load_sfa_panel(0);
    load_sfb_panel(0);
    publish_sfa_panel(0);
    __builtin_amdgcn_sched_barrier(0);
    }
    if constexpr (ShortK) {
        auto short_loop = [&](auto full) {
        if constexpr ((decltype(full)::value && register_short_scales && !full_resident) || compact_panel_short) {
            static_for<capacity>([&](auto stage_i) {
                constexpr int stage = decltype(stage_i)::value;
                if (decltype(full)::value || stage < loops) {
                issue_matrix_prefetch(stage_i, stage);
                static_for<T::E_K>([&](auto k_i) {
                    constexpr int group = stage * T::E_K + decltype(k_i)::value;
                    resident_sfa[group][0] = load<1>(g_sfa, group * kargs.stride_sfa + wave_id_m * T::W_M + lane_id % T::W_M)[0];
                    resident_sfb[group][0] = load<1>(g_sfb, group)[0];
                });
                __builtin_amdgcn_sched_barrier(0);
                }
            });
        }
        if constexpr (!compact_panel_short && (!decltype(full)::value || full_resident) && (T::B_M != 16 || T::B_K < 256)) {
        static_for<capacity>([&](auto stage_i) {
            constexpr int stage = decltype(stage_i)::value;
            if (stage > 0 && (decltype(full)::value || stage < loops) && !(full_resident && capacity == 6 && decltype(full)::value && stage == capacity - 1)) issue_matrix_prefetch(stage_i, stage);
        });
        }
        if constexpr (!compact_panel_short && !decltype(full)::value) {
            s_waitcnt_vmcnt(0_I);
            s_waitcnt_lgkmcnt(0_I);
            if constexpr (T::NUM_WAVES > 1) sync_threads();
        }
        if constexpr (resident_overlap && decltype(full)::value) {
            array<array<typename BaseMMA::vtype_b, T::E_N>, T::E_K> v_b_next;
            constexpr int queued = capacity == 6 ? capacity - 2 : capacity - 1;
            s_waitcnt_vmcnt(number<queued * T::MATRIX_VMEM_INSTRUCTIONS>{});
            s_waitcnt_lgkmcnt(0_I);
            sync_threads();
            read_scales(0, v_sfa, v_sfb);
            load_tile_fragments(0_I);
            if constexpr (capacity == 6) {
                __builtin_amdgcn_sched_barrier(0);
                issue_matrix_prefetch(number<capacity - 1>{}, capacity - 1);
            }
            auto stage_output_row = [&](auto m_i) {
                static_for<T::E_N>([&](auto n_i) {
                    stage_output_fragment(number<decltype(m_i)::value * T::E_N + decltype(n_i)::value>{});
                });
            };
            auto read_next = [&](auto next_i) {
                constexpr int next = decltype(next_i)::value;
                constexpr int pending = (capacity - next - 1) * T::MATRIX_VMEM_INSTRUCTIONS;
                __builtin_amdgcn_sched_barrier(0);
                s_waitcnt_vmcnt(number<pending>{});
                sync_threads();
                if constexpr (T::E_M == 3) {
                    const auto current_b = v_b;
                    static_for<T::E_N>([&](auto n_i) {
                        load_b_fragment(n_i, next_i, 0_I);
                    });
                    v_b_next = v_b;
                    v_b = current_b;
                }
                load_a_fragment(0_I, next_i, 0_I);
                if constexpr (register_short_scales) {
                    v_sfa_next[0] = resident_a_scales(next);
                    v_sfb_next[0] = resident_sfb[next];
                } else {
                    raw_sfa[0][0] = read_raw_a_scale(0_I, next);
                    v_sfb_next[0][0] = load<1>(s_sfb, next)[0];
                }
                __builtin_amdgcn_sched_barrier(0);
            };
            static_for<capacity>([&](auto stage_i) {
                constexpr int stage = decltype(stage_i)::value, next = stage + 1;
                s_waitcnt_lgkmcnt(number<deferred_short_a_pack && capacity == 3 ? (stage == 0 ? 2 : 4) : 0>{});
                static_for<T::E_M>([&](auto m_i) {
                    constexpr int mr = decltype(m_i)::value;
                    static_for<T::E_N>([&](auto n_i) {
                        constexpr int nr = decltype(n_i)::value;
                        if constexpr (deferred_short_a_pack && capacity == 3) {
                            if constexpr (mr == 0 && nr == 1) s_waitcnt_lgkmcnt(number<stage == 0 ? 0 : 2>{});
                            if constexpr (mr == 1 && nr == 0) s_waitcnt_lgkmcnt(number<next < capacity ? 2 : 0>{});
                        }
                        mma_scale_fragment(m_i, n_i, 0_I);
                        if constexpr (next < capacity && T::E_M == 3 && mr * T::E_N + nr == (capacity == 6 && T::B_M == 48 && T::B_N == 32 && T::B_K == 128 && T::T_M == 1 && T::T_N == 2 && T::SCALE_PANEL == 8 ? (T::E_M - 1) * T::E_N - 1 : T::E_M * T::E_N / 2 - 1))
                            read_next(number<next>{});
                        if constexpr (c_stage_overlap && stage == capacity - 1 && mr > 0 && nr == 0) {
                            if (kargs.m > 32) stage_output_row(number<mr - 1>{});
                        }
                        if constexpr (next < capacity && T::E_M == 2 && mr == 1) {
                            load_b_fragment(n_i, number<next>{}, 0_I);
                            __builtin_amdgcn_sched_barrier(0);
                        }
                    });
                    if constexpr (next < capacity && T::E_M == 2 && mr == 0)
                        read_next(number<next>{});
                    if constexpr (next < capacity && mr > 0) {
                        load_a_fragment(m_i, number<next>{}, 0_I);
                        if constexpr (!register_short_scales) raw_sfa[0][mr] = read_raw_a_scale(m_i, next);
                        __builtin_amdgcn_sched_barrier(0);
                    }
                });
                if constexpr (next < capacity) {
                    if constexpr (T::E_M == 3) v_b = v_b_next;
                    if constexpr (register_short_scales) v_sfa = v_sfa_next;
                    v_sfb = v_sfb_next;
                }
                if constexpr (c_stage_overlap && stage == capacity - 1) {
                    if (kargs.m > 32) stage_output_row(number<T::E_M - 1>{});
                }
            });
            return;
        }
        static_for<capacity>([&](auto stage_i) {
            constexpr int stage = decltype(stage_i)::value;
            if (decltype(full)::value || stage < loops) {
                if constexpr (decltype(full)::value || compact_panel_short) {
                    if constexpr (decltype(full)::value) {
                        constexpr int queued = full_resident && capacity == 6 && stage == 0
                            ? capacity - 2 : capacity - stage - 1;
                        constexpr int pending = queued
                            * (T::MATRIX_VMEM_INSTRUCTIONS + (register_short_scales ? 2 * T::E_K : 0));
                        s_waitcnt_vmcnt(number<(pending > 63 ? 63 : pending)>{});
                    } else s_waitcnt_vmcnt(0_I);
                    s_waitcnt_lgkmcnt(0_I);
                    if constexpr (T::NUM_WAVES > 1) sync_threads();
                }
                read_scales(stage, v_sfa, v_sfb);
                if constexpr (T::B_M == 32 && T::B_N == 32 && T::B_K == 512 && T::NUM_WAVES == 2 && T::E_M == 1) {
                    static_for<T::E_K>([&](auto k_i) {
                        constexpr int ks = decltype(k_i)::value;
                        static_for<T::A_RING>([&](auto m_i) {
                            load_a_fragment(m_i, stage_i, k_i);
                        });
                        static_for<T::E_N>([&](auto n_i) {
                            load_b_fragment(n_i, stage_i, k_i);
                        });
                        if constexpr (ks > 0) {
                            __builtin_amdgcn_sched_barrier(0);
                            s_waitcnt_lgkmcnt(number<T::A_CHUNKS_PER_FRAGMENT * T::A_RING + T::B_CHUNKS_PER_FRAGMENT * T::E_N>{});
                            static_for<T::E_N>([&](auto n_i) {
                                mma_scale_fragment(0_I, n_i, number<ks - 1>{});
                            });
                        }
                    });
                    s_waitcnt_lgkmcnt(0_I);
                    static_for<T::E_N>([&](auto n_i) {
                        mma_scale_fragment(0_I, n_i, number<T::E_K - 1>{});
                    });
                } else {
                    load_tile_fragments(stage_i);
                    if constexpr (full_resident && capacity == 6 && decltype(full)::value && stage == 0) {
                        __builtin_amdgcn_sched_barrier(0);
                        issue_matrix_prefetch(number<capacity - 1>{}, capacity - 1);
                        __builtin_amdgcn_sched_barrier(0);
                    }
                    if constexpr (T::B_M <= 32) __builtin_amdgcn_sched_barrier(0);
                    finish_tile(stage_i);
                }
            }
        });
        };
        constexpr bool prefer_full = (compact_panel_short && capacity == 6) ||
            (T::B_M == 16 && T::B_N == 32 && T::B_K == 512 && (capacity == 2 || capacity == 3) && T::NUM_WAVES == 2 && T::E_M == 1);
        if constexpr (two_panel_short || prefer_full || panel_short || full_resident) {
            const bool full = two_panel_short ? small_full_two : loops == capacity;
            if (prefer_full ? __builtin_expect(full, 1) : full) short_loop(1_I);
            else short_loop(0_I);
        } else short_loop(0_I);
    } else {
        auto normal_loop = [&](auto stage_count) {
            constexpr int count = decltype(stage_count)::value;
            if constexpr ((T::B_M <= 32 || T::B_M == 64) && T::B_N <= 256) {
                // Read the current tile before DMA fills the unused LDS slot.
                constexpr int initial = count - 1;
                static_for<initial>([&](auto stage_i) {
                    if constexpr (decltype(stage_i)::value > 0) issue_matrix_prefetch(stage_i, decltype(stage_i)::value);
                });
                array<array<typename BaseMMA::vtype_a, T::E_M>, T::E_K> full_a;
                auto step = [&](auto stage_i, int tile_k, auto pending, auto refill) {
                    s_waitcnt_vmcnt(pending);
                    s_waitcnt_lgkmcnt(0_I);
                    if constexpr (T::NUM_WAVES > 1) sync_threads();
                    read_scales(tile_k, v_sfa, v_sfb);
                    static_for<T::E_K>([&](auto k_i) {
                        if constexpr (normal_a1_overlap)
                            static_for<T::E_N>([&](auto n_i) {
                                load_b_fragment(n_i, stage_i, k_i);
                            });
                        static_for<T::E_M>([&](auto m_i) {
                            if constexpr (T::E_M <= T::A_RING) load_a_fragment(m_i, stage_i, k_i);
                            else {
                                constexpr int mr = decltype(m_i)::value, ks = decltype(k_i)::value;
                                full_a[ks][mr] = read_a_fragment(m_i, stage_i, k_i);
                            }
                        });
                        if constexpr (!normal_a1_overlap)
                            static_for<T::E_N>([&](auto n_i) {
                                load_b_fragment(n_i, stage_i, k_i);
                            });
                    });
                    __builtin_amdgcn_sched_barrier(0);
                    if constexpr (decltype(refill)::value) {
                        constexpr int slot = (decltype(stage_i)::value + initial) % count;
                        issue_matrix_prefetch(number<slot>{}, tile_k + initial);
                    }
                    if constexpr (T::E_M <= T::A_RING) finish_tile(stage_i);
                    else {
                        s_waitcnt_lgkmcnt(0_I);
                        static_for<T::E_M>([&](auto m_i) {
                            static_for<T::E_K>([&](auto k_i) {
                                static_for<T::E_N>([&](auto n_i) {
                                    mma_a_scale_fragment(m_i, n_i, k_i, full_a[decltype(k_i)::value][decltype(m_i)::value]);
                                });
                            });
                        });
                    }
                    __builtin_amdgcn_sched_barrier(0);
                };
                constexpr int queued = (initial - 1) * T::MATRIX_VMEM_INSTRUCTIONS;
                constexpr int pending = queued <= 63 ? queued : 63;
                const int steady = loops - initial;
                int base = 0;
                constexpr int loop_unroll = (normal_a1_overlap || (T::B_M == 64 && T::B_N == 64 && T::B_K == 256 && T::T_M == 2 && T::T_N == 4 && T::SCALE_PANEL == 64 && T::SPLIT_K == 1 && T::CACHE_A == 17 && T::CACHE_B == 1 && T::GROUPED_M == 0) || (T::B_M == 32 && T::B_N == 32 && T::B_K == 128 && T::T_M == 2 && T::T_N == 1 && T::SCALE_PANEL == 128 && T::SPLIT_K == 1 && T::CACHE_A == 0 && T::CACHE_B == 0 && T::GROUPED_M == 0) || (T::B_M == 64 && T::B_N == 128 && T::B_K == 128 && T::T_M == 2 && T::T_N == 2 && T::SCALE_PANEL == 16 && T::SPLIT_K == 1 && T::CACHE_A == 0 && T::CACHE_B == 0 && T::GROUPED_M == 0)) ? 1 : (T::E_K <= 2 ? 4 : 1);
#pragma clang loop unroll_count(loop_unroll)
                for (; base + count <= steady; base += count) {
                    static_for<count>([&](auto stage_i) {
                        const int tile_k = base + decltype(stage_i)::value;
                        step(stage_i, tile_k, number<pending>{}, 1_I);
                    });
                }
                static_for<count>([&](auto stage_i) {
                    const int tile_k = base + decltype(stage_i)::value;
                    if (tile_k < steady) step(stage_i, tile_k, number<pending>{}, 1_I);
                });
                static_for<initial>([&](auto drain_i) {
                    constexpr int d = decltype(drain_i)::value;
                    constexpr int queued_drain = (initial - 1 - d) * T::MATRIX_VMEM_INSTRUCTIONS;
                    const int tile_k = steady + d;
                    step(tile_k % count, tile_k, number<queued_drain <= 63 ? queued_drain : 63>{}, 0_I);
                });
                return;
            }
            static_for<count>([&](auto stage_i) {
                constexpr int stage = decltype(stage_i)::value;
                if constexpr (stage > 0) issue_matrix_prefetch(stage_i, stage);
            });
            constexpr int pending = (count - 1) * T::MATRIX_VMEM_INSTRUCTIONS;
            s_waitcnt_vmcnt(number<pending <= 63 ? pending : 63>{});
            s_waitcnt_lgkmcnt(0_I);
            if constexpr (T::NUM_WAVES > 1) sync_threads();
            read_scales(0, v_sfa, v_sfb);
            load_tile_fragments(0_I);
            s_waitcnt_lgkmcnt(0_I);
            constexpr int live_fragments = 4 * T::E_M * T::E_N +
                8 * T::E_K * (T::A_RING + T::E_N);
            int base = 0;
            if constexpr (live_fragments < 160) {
                __builtin_assume(stages == count);
                constexpr int loop_unroll = (T::B_M == 48 && T::B_N == 64 && T::B_K == 128 && T::T_M == 1 && T::T_N == 2 &&
                    ((T::SCALE_PANEL == 16 && T::CACHE_A == 0 && T::CACHE_B == 0) || (T::SCALE_PANEL == 128 && T::CACHE_A == 17 && T::CACHE_B == 1)) && T::SPLIT_K == 1 && T::GROUPED_M == 0) || (T::B_M == 128 && T::B_N == 128 && T::B_K == 128 && T::T_M == 2 && T::T_N == 4 && T::SCALE_PANEL == 256 && T::SPLIT_K == 1 && T::CACHE_A == 17 && T::CACHE_B == 1 && T::GROUPED_M == 0) ? 1 : (T::E_K <= 2 ? 4 : 1);
#pragma clang loop unroll_count(loop_unroll)
                for (; base + 2 * count <= loops; base += count) {
                    static_for<count>([&](auto stage_i) {
                        constexpr int stage = decltype(stage_i)::value;
                        const int tile_k = base + stage;
                        __builtin_assume(tile_k + count < loops);
                        advance_tile(stage_i, number<(stage + 1) % count>{}, tile_k, number<((count - 2) * T::MATRIX_VMEM_INSTRUCTIONS <= 63
                            ? (count - 2) * T::MATRIX_VMEM_INSTRUCTIONS : 63)>{});
                    });
                }
            }
#pragma clang loop unroll(disable)
            for (; base + 1 < loops; base += count) {
                static_for<count>([&](auto stage_i) {
                    constexpr int stage = decltype(stage_i)::value;
                    const int tile_k = base + stage;
                    if (tile_k + 1 < loops)
                        advance_tile(stage_i, number<(stage + 1) % count>{}, tile_k,
                            number<live_fragments >= 160 ? 0 : ((count - 2) * T::MATRIX_VMEM_INSTRUCTIONS <= 63
                                ? (count - 2) * T::MATRIX_VMEM_INSTRUCTIONS : 63)>{});
                });
            }
            finish_tile((loops - 1) % count);
        };
        if (ring8)
            normal_loop(number<8>{});
        else if constexpr (T::B_M == 128 && T::B_N == 128 && T::T_M == 4 && T::T_N == 2) {
            if (stages == 3)
                normal_loop(number<3>{});
            else
                normal_loop(number<2>{});
        } else if constexpr (normal_stage4_profile)
            normal_loop(number<4>{});
        else normal_loop(number<T::NUM_STAGES>{});
    }
    if constexpr (c_stage_overlap && !native_short) {
        if (loops == capacity && kargs.m > 32) {
            s_waitcnt_lgkmcnt(0_I);
            sync_threads();
            static_for<T::OUTPUT_PASSES>([&](auto pass_i) {
                write_output_vector(decltype(pass_i)::value);
            });
            return;
        }
    }
    if constexpr (native_short || (!(normal_a1_overlap || (normal_narrow_raw)) &&
        (T::B_M <= 32 || (ShortK && T::B_M <= 64)))) {
        if (native_short || T::B_M <= 32 || kargs.m <= 32 || (T::B_M == 48 && T::B_N == 32 && T::B_K == 128 && T::T_M == 1 && T::T_N == 2 && T::SCALE_PANEL == 8 && capacity == 6)) {
            write_global_c(number<ShortK ? 0 : 2>{});
            return;
        }
    }
    s_waitcnt_lgkmcnt(0_I);
    sync_threads();
    if constexpr (T::SPLIT_K > 1) {
        // Each split publishes FP32 [SplitK,M,N] for the caller-owned reducer.
        write_global_c(number<2>{});
    } else {
        static_for<T::E_M * T::E_N>([&](auto c_i) {
            stage_output_fragment(c_i);
        });
        s_waitcnt_lgkmcnt(0_I);
        sync_threads();
        static_for<T::OUTPUT_PASSES>([&](auto pass_i) {
            write_output_vector(decltype(pass_i)::value);
        });
    }
}
#endif // gfx950
template<class Traits, bool ShortK>
__global__ __attribute__((amdgpu_flat_work_group_size(1, Traits::BLOCK_SIZE), amdgpu_waves_per_eu(Traits::WG_PER_CU, (ShortK && Traits::B_M <= 32 && Traits::B_N == 32 && Traits::B_K == 512 && Traits::NUM_WAVES == 2 ? 2 : 0))))
void opus_gemm_bpreshuffle_staged_b_kernel( opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 args) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx950__)
    if constexpr (ShortK && Traits::B_M == 16 && Traits::B_N == 32 && Traits::B_K == 128 && Traits::T_M == 1 && Traits::T_N == 2 && Traits::SCALE_PANEL == 8 && Traits::RESIDENT_CAPACITY >= 6) {
        if (__builtin_expect(args.m > 0 && args.m <= 32, 1)) opus_gemm_bpreshuffle_staged_b_body_impl<Traits, ShortK, 6>(args);
        else opus_gemm_bpreshuffle_staged_b_body_impl<Traits, ShortK, Traits::RESIDENT_CAPACITY>(args);
    } else opus_gemm_bpreshuffle_staged_b_body_impl<Traits, ShortK, Traits::RESIDENT_CAPACITY>(args);
#endif
}
