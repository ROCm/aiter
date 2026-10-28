// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <opus/hip_minimal.hpp>
#include <opus/opus.hpp>
#include "../opus_gemm_utils.cuh"
#include "opus_gemm_traits_a8w8_blockscale_mxscale_bpreshuffle_interleave_gfx950.cuh"

#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx950__)
namespace opus_gemm_bpreshuffle_wave_interleave_layout {

using opus::operator""_I;

// The XOR exchanges 16-byte K vectors inside each group of 16 A rows.
template<class T>
__device__ inline int a_offset(int row, int col) {
    const int offset = row * T::K_STAGE + col;
    return offset ^ (((offset % (T::W_M * T::K_STAGE)) >> 8) << 4);
}

template<class T>
__device__ inline int b_offset(int row, int col) {
    constexpr int n_lanes = T::WARP_SIZE / (T::K_STAGE / T::VEC_B);
    return (row / n_lanes) * (T::WARP_SIZE * T::VEC_B) +
        (row % n_lanes) * T::VEC_B + (col / T::VEC_B) * (n_lanes * T::VEC_B);
}

// A global-memory vectors use the LDS XOR swizzle.
template<class T>
__device__ inline auto make_layout_ga(int lane_id, int wave_id, int stride_a) {
    constexpr int threads_k = T::K_STAGE / T::VEC_A;
    constexpr int threads_m = T::WARP_SIZE / threads_k;

    constexpr auto ga_block_shape = opus::make_tuple(
        opus::number<T::A_STEPS>{},
        opus::number<T::BLOCK_SIZE / T::WARP_SIZE>{},
        opus::number<threads_m>{},
        opus::number<threads_k>{},
        opus::number<T::VEC_A>{});
    constexpr auto ga_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}));

    return opus::make_layout<T::VEC_A>(
        ga_block_shape,
        opus::unfold_x_stride(ga_block_dim, ga_block_shape, opus::tuple{stride_a, 1_I}),
        opus::unfold_p_coord(ga_block_dim, opus::tuple{wave_id, lane_id / threads_k, lane_id % threads_k}));
}

// LDS producer offsets exclude the lane offset supplied by async_load.
template<class T>
__device__ inline auto make_layout_sa(int wave_id) {
    constexpr auto sa_block_shape = opus::make_tuple(
        opus::number<T::A_STEPS>{},
        opus::number<T::BLOCK_SIZE / T::WARP_SIZE>{},
        opus::number<T::VEC_A>{});
    constexpr auto sa_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}));

    return opus::make_layout<T::VEC_A>(
        sa_block_shape,
        opus::unfold_x_stride(sa_block_dim, sa_block_shape, opus::tuple{opus::number<T::WARP_SIZE * T::VEC_A>{}, 1_I}),
        opus::unfold_p_coord(sa_block_dim, opus::tuple{wave_id}));
}

// A LDS -> MFMA register layouts.
template<class T>
__device__ inline auto make_layout_ra(int lane_id, int wave_id_m) {
    constexpr int lanes_k = T::WARP_SIZE / T::W_M;
    constexpr int chunks = T::W_M * T::W_K / (T::WARP_SIZE * T::VEC_A);

    constexpr auto ra_block_shape = opus::make_tuple(
        opus::number<T::E_M>{},
        opus::number<T::T_M>{},
        opus::number<T::W_M>{},
        opus::number<chunks>{},
        opus::number<lanes_k>{},
        opus::number<T::VEC_A>{});
    constexpr auto ra_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}),
        opus::make_tuple(opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::y_dim{}));

    return opus::make_layout<T::VEC_A>(
        ra_block_shape,
        opus::unfold_x_stride(ra_block_dim, ra_block_shape,
            opus::tuple{opus::number<T::W_M * T::W_K>{}, opus::number<T::E_M * T::W_M * T::W_K>{}, opus::number<T::W_K>{}, 1_I}),
        opus::unfold_p_coord(ra_block_dim, opus::tuple{wave_id_m, lane_id % T::W_M, lane_id / T::W_M}));
}

// B global-memory coordinates follow the N16/K64 preshuffle.
template<class T>
__device__ inline auto make_layout_gb(int lane_id, int wave_id, int stride_b) {
    constexpr int n_lanes = T::WARP_SIZE / (T::K_STAGE / T::VEC_B);
    constexpr int waves_per_n_group = T::W_N / n_lanes;
    constexpr int k_vectors_per_half = T::WARP_SIZE / T::W_N;
    constexpr int lanes_per_k_half = n_lanes * k_vectors_per_half;

    constexpr auto gb_block_shape = opus::make_tuple(
        opus::number<T::B_STEPS>{},
        opus::number<T::BLOCK_SIZE / (waves_per_n_group * T::WARP_SIZE)>{},
        opus::number<waves_per_n_group>{},
        opus::number<T::WARP_SIZE / lanes_per_k_half>{},
        opus::number<k_vectors_per_half>{},
        opus::number<n_lanes>{},
        opus::number<T::VEC_B>{});
    constexpr auto gb_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}));

    return opus::make_layout<T::VEC_B>(
        gb_block_shape,
        opus::unfold_x_stride(gb_block_dim, gb_block_shape,
            opus::tuple{stride_b * T::W_N, opus::number<n_lanes * T::VEC_B>{}, opus::number<T::W_N * T::VEC_B>{}, 1_I}),
        opus::unfold_p_coord(gb_block_dim,
            opus::tuple{wave_id / waves_per_n_group, wave_id % waves_per_n_group, lane_id / lanes_per_k_half, (lane_id / n_lanes) % k_vectors_per_half, lane_id % n_lanes}));
}

template<class T>
__device__ inline auto make_layout_sb(int wave_id) {
    constexpr auto sb_block_shape = opus::make_tuple(
        opus::number<T::B_STEPS>{},
        opus::number<T::BLOCK_SIZE / T::WARP_SIZE>{},
        opus::number<T::VEC_B>{});
    constexpr auto sb_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}));

    return opus::make_layout<T::VEC_B>(
        sb_block_shape,
        opus::unfold_x_stride(sb_block_dim, sb_block_shape, opus::tuple{opus::number<T::WARP_SIZE * T::VEC_B>{}, 1_I}),
        opus::unfold_p_coord(sb_block_dim, opus::tuple{wave_id}));
}

template<class T>
__device__ inline auto make_layout_rb(int lane_id, int wave_id_n) {
    constexpr int n_lanes = T::WARP_SIZE / (T::W_K / T::VEC_B);
    constexpr int lanes_k = T::WARP_SIZE / T::W_N;
    constexpr int chunks = T::W_N * T::W_K / (T::WARP_SIZE * T::VEC_B);

    constexpr auto rb_block_shape = opus::make_tuple(
        opus::number<T::E_N>{},
        opus::number<T::T_N>{},
        opus::number<T::W_N / n_lanes>{},
        opus::number<n_lanes>{},
        opus::number<chunks>{},
        opus::number<lanes_k>{},
        opus::number<T::VEC_B>{});
    constexpr auto rb_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}),
        opus::make_tuple(opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}));

    return opus::make_layout<T::VEC_B>(
        rb_block_shape,
        opus::unfold_x_stride(rb_block_dim, rb_block_shape,
            opus::tuple{opus::number<T::W_N * T::W_K>{}, opus::number<T::E_N * T::W_N * T::W_K>{}, opus::number<n_lanes * T::W_K>{}, opus::number<T::VEC_B>{}, opus::number<n_lanes * T::VEC_B>{}, 1_I}),
        opus::unfold_p_coord(rb_block_dim,
            opus::tuple{wave_id_n, (lane_id % T::W_N) / n_lanes, lane_id % n_lanes, lane_id / T::W_N}));
}

// Each wave reads one scale byte for every M repeat in both matrix halves.
template<class T>
__device__ inline auto make_layout_gsfa_first(int lane_id, int wave_id_m) {
    constexpr auto gsfa_first_block_shape = opus::make_tuple(
        opus::number<T::M_HALVES>{},
        opus::number<T::T_M>{},
        opus::number<T::E_M>{},
        opus::number<T::W_M>{},
        1_I);
    constexpr auto gsfa_first_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::y_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}));

    return opus::make_layout<1>(
        gsfa_first_block_shape,
        opus::unfold_x_stride(gsfa_first_block_dim, gsfa_first_block_shape, opus::tuple{1_I, 1_I}),
        opus::unfold_p_coord(gsfa_first_block_dim, opus::tuple{wave_id_m, lane_id % T::W_M}));
}

template<class T>
__device__ inline auto make_layout_gsfa(int lane_id, int wave_id, int stride_sfa) {
    constexpr int copies = (T::SFA_BYTES + T::BLOCK_SIZE * T::VEC_SCALE_A - 1) / (T::BLOCK_SIZE * T::VEC_SCALE_A);
    constexpr auto gsfa_block_shape = opus::make_tuple(
        opus::number<copies>{},
        opus::number<T::BLOCK_SIZE / T::WARP_SIZE>{},
        opus::number<T::WARP_SIZE>{},
        opus::number<T::VEC_SCALE_A>{});
    constexpr auto gsfa_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}));

    auto layout = opus::make_layout<T::VEC_SCALE_A>(
        gsfa_block_shape,
        opus::unfold_x_stride(gsfa_block_dim, gsfa_block_shape, opus::tuple{opus::number<T::VEC_SCALE_A>{}, 1_I}),
        opus::unfold_p_coord(gsfa_block_dim, opus::tuple{wave_id, lane_id}));
    // BlockM=192 can put successive producer copies in different scale rows.
    opus::static_for<copies>([&](auto i) {
        constexpr int copy = decltype(i)::value;
        const int linear = layout.offsets[copy];
        layout.offsets[copy] = (linear / T::B_M + 1) * stride_sfa + linear % T::B_M;
    });
    return layout;
}

template<class T>
__device__ inline auto make_layout_ssfa(int wave_id) {
    constexpr int copies = (T::SFA_BYTES + T::BLOCK_SIZE * T::VEC_SCALE_A - 1) / (T::BLOCK_SIZE * T::VEC_SCALE_A);
    constexpr auto ssfa_block_shape = opus::make_tuple(
        opus::number<copies>{},
        opus::number<T::BLOCK_SIZE / T::WARP_SIZE>{},
        opus::number<T::VEC_SCALE_A>{});
    constexpr auto ssfa_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}));

    return opus::make_layout<T::VEC_SCALE_A>(
        ssfa_block_shape,
        opus::unfold_x_stride(ssfa_block_dim, ssfa_block_shape, opus::tuple{opus::number<T::WARP_SIZE * T::VEC_SCALE_A>{}, 1_I}),
        opus::unfold_p_coord(ssfa_block_dim, opus::tuple{wave_id}));
}

template<class T>
__device__ inline auto make_layout_rsfa(int lane_id, int wave_id_m) {
    constexpr auto rsfa_block_shape = opus::make_tuple(
        opus::number<T::M_HALVES>{},
        opus::number<T::T_M>{},
        opus::number<T::E_M>{},
        opus::number<T::W_M>{},
        1_I);
    constexpr auto rsfa_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::y_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}));

    return opus::make_layout<1>(
        rsfa_block_shape,
        opus::unfold_x_stride(rsfa_block_dim, rsfa_block_shape, opus::tuple{1_I, 1_I}),
        opus::unfold_p_coord(rsfa_block_dim, opus::tuple{wave_id_m, lane_id % T::W_M}));
}

// PACKED_SCALE reads four M16 bytes per output word; offsets stay in bytes.
template<class T>
__device__ inline auto make_layout_sfa_repack(int lane_id, int wave_id) {
    constexpr int words_per_group = T::B_M / T::SCALE_WORD_BYTES;
    constexpr int rounds = (T::SCALE_GROUPS - 1) * words_per_group / T::BLOCK_SIZE + 1;
    constexpr auto sfa_repack_block_shape = opus::make_tuple(
        opus::number<rounds>{},
        opus::number<T::BLOCK_SIZE / T::WARP_SIZE>{},
        opus::number<T::WARP_SIZE / T::W_M>{},
        opus::number<T::W_M>{},
        opus::number<T::SCALE_WORD_BYTES>{});
    constexpr auto sfa_repack_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}));

    return opus::make_layout(
        sfa_repack_block_shape,
        opus::unfold_x_stride(sfa_repack_block_dim, sfa_repack_block_shape,
            opus::tuple{opus::number<T::B_M>{}, opus::number<T::W_M * T::SCALE_WORD_BYTES>{}, 1_I, opus::number<T::W_M>{}}),
        opus::unfold_p_coord(sfa_repack_block_dim, opus::tuple{wave_id, lane_id / T::W_M, lane_id % T::W_M}));
}

// Packed-scale writes and reads count dwords rather than raw scale bytes.
template<class T>
__device__ inline auto make_layout_psfa(int lane_id, int wave_id) {
    constexpr int words_per_group = T::B_M / T::SCALE_WORD_BYTES;
    constexpr int rounds = (T::SCALE_GROUPS - 1) * words_per_group / T::BLOCK_SIZE + 1;
    constexpr auto psfa_block_shape = opus::make_tuple(
        opus::number<rounds>{},
        opus::number<T::BLOCK_SIZE / T::WARP_SIZE>{},
        opus::number<T::WARP_SIZE>{},
        1_I);
    constexpr auto psfa_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}));

    return opus::make_layout(
        psfa_block_shape,
        opus::unfold_x_stride(psfa_block_dim, psfa_block_shape, opus::tuple{1_I, 1_I}),
        opus::unfold_p_coord(psfa_block_dim, opus::tuple{wave_id, lane_id}));
}

template<class T>
__device__ inline auto make_layout_rsfa_packed(int lane_id, int wave_id_m) {
    // Only the PACKED_SCALE branch consumes this E_M=4 mapping.
    constexpr auto rsfa_packed_block_shape = opus::make_tuple(
        opus::number<T::M_HALVES>{},
        opus::number<T::T_M>{},
        opus::number<T::W_M>{},
        1_I);
    constexpr auto rsfa_packed_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}),
        opus::make_tuple(opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}));

    return opus::make_layout(
        rsfa_packed_block_shape,
        opus::unfold_x_stride(rsfa_packed_block_dim, rsfa_packed_block_shape,
            opus::tuple{opus::number<T::HALF_M / T::SCALE_WORD_BYTES>{}, opus::number<T::E_M * T::SCALE_WORD_BYTES>{}, 1_I}),
        opus::unfold_p_coord(rsfa_packed_block_dim, opus::tuple{wave_id_m, lane_id % T::W_M}));
}

template<class T>
__device__ inline auto make_layout_gsfb(int lane_id, int wave_id, int stride_sfb) {
    constexpr auto gsfb_block_shape = opus::make_tuple(
        opus::number<T::N_HALVES>{},
        opus::number<T::BLOCK_SIZE / T::WARP_SIZE>{},
        opus::number<T::WARP_SIZE>{},
        1_I);
    constexpr auto gsfb_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::y_dim{}));

    return opus::make_layout<1>(
        gsfb_block_shape,
        opus::unfold_x_stride(gsfb_block_dim, gsfb_block_shape, opus::tuple{stride_sfb, 1_I}),
        opus::unfold_p_coord(gsfb_block_dim, opus::tuple{wave_id, lane_id}));
}

template<class T>
__device__ inline auto make_layout_rsfb() {
    constexpr auto rsfb_block_shape = opus::make_tuple(
        opus::number<T::SCALE_GROUPS>{},
        1_I);
    constexpr auto rsfb_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::y_dim{}));

    return opus::make_layout(
        rsfb_block_shape,
        opus::unfold_x_stride(rsfb_block_dim, rsfb_block_shape, opus::tuple{1_I}),
        opus::unfold_p_coord(rsfb_block_dim, opus::tuple{}));
}

// All quadrant/M/N repeats expose their four BF16 row elements to LDS.
template<class T>
__device__ inline auto make_layout_sc(int lane_id, int wave_id_m, int wave_id_n) {
    constexpr auto sc_block_shape = opus::make_tuple(
        opus::number<T::M_HALVES>{},
        opus::number<T::N_HALVES>{},
        opus::number<T::E_M>{},
        opus::number<T::E_N>{},
        opus::number<T::T_M>{},
        opus::number<T::T_N>{},
        opus::number<T::WARP_SIZE / T::W_M>{},
        opus::number<T::W_M>{},
        opus::number<T::VEC_C>{});
    constexpr auto sc_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}),
        opus::make_tuple(opus::y_dim{}),
        opus::make_tuple(opus::y_dim{}),
        opus::make_tuple(opus::y_dim{}),
        opus::make_tuple(opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}));

    return opus::make_layout(
        sc_block_shape,
        opus::unfold_x_stride(sc_block_dim, sc_block_shape,
            opus::tuple{opus::number<T::HALF_M * T::C_STRIDE>{}, opus::number<T::HALF_N>{},
                opus::number<T::W_M * T::C_STRIDE>{}, opus::number<T::W_N>{}, opus::number<T::E_M * T::W_M * T::C_STRIDE>{},
                opus::number<T::E_N * T::W_N>{}, opus::number<T::VEC_C * T::C_STRIDE>{}, 1_I, opus::number<T::C_STRIDE>{}}),
        opus::unfold_p_coord(sc_block_dim, opus::tuple{wave_id_m, wave_id_n, lane_id / T::W_M, lane_id % T::W_M}));
}

template<class T>
__device__ inline auto make_layout_c_output(int lane_id, int wave_id, int stride_c) {
    constexpr int threads_n = T::B_N / T::VEC_OUTPUT;
    constexpr int threads_m = T::WARP_SIZE / threads_n;

    constexpr auto c_output_block_shape = opus::make_tuple(
        opus::number<T::OUTPUT_PASSES>{},
        opus::number<T::BLOCK_SIZE / T::WARP_SIZE>{},
        opus::number<threads_m>{},
        opus::number<threads_n>{},
        opus::number<T::VEC_OUTPUT>{});
    constexpr auto c_output_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}));

    return opus::make_layout(
        c_output_block_shape,
        opus::unfold_x_stride(c_output_block_dim, c_output_block_shape,
            opus::tuple{(T::BLOCK_SIZE * T::VEC_OUTPUT / T::B_N) * stride_c, stride_c, 1_I}),
        opus::unfold_p_coord(c_output_block_dim, opus::tuple{wave_id, lane_id / threads_n, lane_id % threads_n}));
}

template<class Fragment, class Mem, class Layout, class Index>
__device__ inline Fragment read_operand_fragment(Mem& smem, int base, const Layout& layout, Index) {
    constexpr int index = Index::value;
    const int lo = layout.offsets[index * 2];
    const int hi = layout.offsets[index * 2 + 1];
    Fragment result;
    opus::set_slice(result, opus::load<16>(smem, base + lo), opus::number<0>{}, opus::number<16>{});
    opus::set_slice(result, opus::load<16>(smem, base + hi), opus::number<16>{}, opus::number<32>{});
    return result;
}

} // namespace opus_gemm_bpreshuffle_wave_interleave_layout

template<class Traits>
__device__ __forceinline__ void opus_gemm_bpreshuffle_wave_interleave_body(
    opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 kargs) {
    using namespace opus;
    using namespace opus_gemm_bpreshuffle_wave_interleave_layout;
    using T = Traits;
    using D_A = fp8_t;
    using D_C = bf16_t;

    if constexpr (T::FIXED_K > 0) {
        kargs.k = T::FIXED_K;
        kargs.stride_a = T::FIXED_K;
        kargs.stride_b = T::FIXED_K;
        kargs.stride_sfb = T::FIXED_K / 128;
    }

    // Tile and thread coordinates.
    const int lane_id = thread_id_x() % T::WARP_SIZE;
    const int wave_id = __builtin_amdgcn_readfirstlane(thread_id_x() / T::WARP_SIZE);
    const int wave_id_m = wave_id / T::T_N;
    const int wave_id_n = wave_id % T::T_N;
    const int gn = kargs.n / T::B_N;
    const int gm = (kargs.m + T::B_M - 1) / T::B_M;
    int block_m = block_id_y();
    int block_n = block_id_x();
    if constexpr (T::GROUP_M > 0) {
        int linear = block_m * gn + block_n;
        if constexpr (T::XCD_ORDER) {
            const int xcd = linear % 8;
            const int intra = linear / 8;
            const int base = gm * gn / 8;
            const int extra = gm * gn % 8;
            linear = xcd * base + (xcd < extra ? xcd : extra) + intra;
        }
        const int first = linear / (T::GROUP_M * gn) * T::GROUP_M;
        const int rows = gm - first < T::GROUP_M ? gm - first : T::GROUP_M;
        const int local = linear % (T::GROUP_M * gn);
        block_m = first + local % rows;
        block_n = local / rows;
    }
    const int row = block_m * T::B_M;
    const int col = block_n * T::B_N;
    const int batch = block_id_z();
    const int loops = (T::FIXED_K ? T::FIXED_K : kargs.k) / T::K_STAGE;
    const int valid_rows = kargs.m - row < T::B_M ? kargs.m - row : T::B_M;

    // Matrix and scale global-memory views, bounded for the final partial M tile.
    auto g_a = make_gmem(
        reinterpret_cast<const D_A*>(kargs.ptr_a) + batch * kargs.stride_a_batch + row * kargs.stride_a,
        static_cast<unsigned>((kargs.m - row) * kargs.stride_a));
    auto g_b = make_gmem(reinterpret_cast<const D_A*>(kargs.ptr_b) +
        batch * kargs.stride_b_batch + col * kargs.stride_b);
    auto g_c = make_gmem(
        reinterpret_cast<D_C*>(kargs.ptr_c) + static_cast<int64_t>(batch) * kargs.stride_c_batch +
            static_cast<int64_t>(row) * kargs.stride_c + col,
        static_cast<unsigned>((static_cast<int64_t>(valid_rows - 1) * kargs.stride_c + T::B_N) * sizeof(D_C)));
    auto g_sfa = make_gmem(reinterpret_cast<const unsigned char*>(kargs.ptr_sfa) + batch * kargs.stride_sfa_batch + row,
        static_cast<unsigned>(kargs.stride_sfa_batch - row));
    auto g_sfb = make_gmem(reinterpret_cast<const unsigned char*>(kargs.ptr_sfb) +
        batch * kargs.stride_sfb_batch + (col / 128) * kargs.stride_sfb);

    // Matrix and scale layouts: global -> LDS -> register fragments.
    constexpr int a_threads_k = T::K_STAGE / T::VEC_A;
    constexpr int a_rows_per_wave = T::WARP_SIZE / a_threads_k;
    auto u_ga = make_layout_ga<T>(lane_id, wave_id, kargs.stride_a);
    opus::static_for<T::A_STEPS>([&](auto i) {
        constexpr int step = decltype(i)::value;
        const int row = lane_id / a_threads_k + wave_id * a_rows_per_wave + step * (T::BLOCK_SIZE / a_threads_k);
        const int col = lane_id % a_threads_k * T::VEC_A;
        u_ga.offsets[step] += a_offset<T>(row, col) % T::K_STAGE - col;
    });
    const auto u_sa = make_layout_sa<T>(wave_id);
    constexpr int a_chunks = T::W_M * T::W_K / (T::WARP_SIZE * T::VEC_A);
    constexpr int a_k_lane_groups = T::WARP_SIZE / T::W_M;
    auto u_ra = make_layout_ra<T>(lane_id, wave_id_m);
    opus::static_for<T::E_M>([&](auto m_i) {
        constexpr int m = decltype(m_i)::value;
        const int row = (wave_id_m * T::E_M + m) * T::W_M + lane_id % T::W_M;
        opus::static_for<a_chunks>([&](auto chunk_i) {
            constexpr int chunk = decltype(chunk_i)::value;
            u_ra.offsets[m * a_chunks + chunk] = a_offset<T>(row,
                lane_id / T::W_M * T::VEC_A + chunk * a_k_lane_groups * T::VEC_A);
        });
    });
    const auto u_gb = make_layout_gb<T>(lane_id, wave_id, kargs.stride_b);
    const auto u_sb = make_layout_sb<T>(wave_id);
    const auto u_rb = make_layout_rb<T>(lane_id, wave_id_n);
    const auto u_gsfa_first = make_layout_gsfa_first<T>(lane_id, wave_id_m);
    const auto u_gsfa = make_layout_gsfa<T>(lane_id, wave_id, kargs.stride_sfa);
    const auto u_ssfa = make_layout_ssfa<T>(wave_id);
    const auto u_rsfa = make_layout_rsfa<T>(lane_id, wave_id_m);
    const auto u_sfa_repack = make_layout_sfa_repack<T>(lane_id, wave_id);
    const auto u_psfa = make_layout_psfa<T>(lane_id, wave_id);
    const auto u_rsfa_packed = make_layout_rsfa_packed<T>(lane_id, wave_id_m);
    const auto u_gsfb = make_layout_gsfb<T>(lane_id, wave_id, kargs.stride_sfb);
    const auto u_rsfb = make_layout_rsfb<T>();
    const auto u_sc = make_layout_sc<T>(lane_id, wave_id_m, wave_id_n);
    const auto u_gc = make_layout_c_output<T>(lane_id, wave_id, kargs.stride_c);
    const auto u_rc = make_layout_c_output<T>(lane_id, wave_id, T::C_STRIDE);

    // Matrix and scale LDS; matrix storage is reused for the BF16 epilogue.
    alignas(16) __shared__ char storage[T::LDS_BYTES];
    auto s_a = make_smem(reinterpret_cast<D_A*>(storage));
    auto s_b = make_smem(reinterpret_cast<D_A*>(storage + 4 * T::A_SLOT));
    auto s_sfa = make_smem(reinterpret_cast<unsigned char*>(storage + T::MATRIX_BYTES));
    auto s_sfb = make_smem(reinterpret_cast<unsigned short*>(storage + T::MATRIX_BYTES + T::SFA_BYTES));
    auto s_c = make_smem(reinterpret_cast<D_C*>(storage));

    // MFMA and register fragments.
    auto mma = make_tiled_mma<D_A, D_A, fp32_t>(
        seq<T::E_M, T::E_N, 1>{},
        seq<1, 1, 1>{},
        seq<T::W_M, T::W_N, T::W_K>{},
        mfma_adaptor{});
    using MMA = typename decltype(mma)::MMA;
    array<array<typename MMA::vtype_a, T::E_M>, 2> v_a;
    array<typename MMA::vtype_b, 2 * T::E_N> v_b;
    array<typename MMA::vtype_c, 4 * T::E_M * T::E_N> v_c{};
    array<unsigned, 2> first_scale_a{};
    array<unsigned, 2> scale_a{};
    unsigned scale_b = 0;

    // Matrix and scale producer offsets; register reads use their own layouts.
    const auto& ga_offsets = u_ga.offsets;
    const auto& sa_offsets = u_sa.offsets;
    const auto& gb_offsets = u_gb.offsets;
    const auto& sb_offsets = u_sb.offsets;
    const auto& gsfa_offsets = u_gsfa.offsets;
    const auto& ssfa_offsets = u_ssfa.offsets;
    const auto& gsfb_offsets = u_gsfb.offsets;

    // A/B global memory -> LDS slots, one wave of vectors per request.
    auto load_a = [&](auto slot_i, int k) {
        constexpr int slot = decltype(slot_i)::value;
        static_for<T::A_STEPS>([&](auto i) {
            constexpr int step = decltype(i)::value;
            async_load<16>(g_a, s_a.ptr + slot * T::A_SLOT + sa_offsets[step],
                ga_offsets[step], (slot % 2) * T::HALF_M * kargs.stride_a + k * 128, number<0>{}, number<T::CACHE_A>{});
        });
    };
    auto load_b = [&](auto slot_i, int k) {
        constexpr int slot = decltype(slot_i)::value;
        static_for<T::B_STEPS>([&](auto i) {
            constexpr int step = decltype(i)::value;
            async_load<16>(g_b, s_b.ptr + slot * T::B_SLOT + sb_offsets[step],
                gb_offsets[step], (slot % 2) * T::HALF_N * kargs.stride_b + k * 2048, number<0>{}, number<T::CACHE_B>{});
        });
    };

    // Keep K=0 scales in registers. The remaining 127 A columns plus packed
    // B scales fit exactly beside the matrix slots at the maximum K=16384.
    auto load_sfa_first = [&] {
        static_for<2>([&](auto half_i) {
            constexpr int half = decltype(half_i)::value;
            static_for<T::E_M>([&](auto i) {
                constexpr int m = decltype(i)::value;
                first_scale_a[half] |= static_cast<unsigned>(load<1>(g_sfa, u_gsfa_first.offsets[half * T::E_M + m])[0]) << (m * 8);
            });
        });
    };
    auto load_sfa_panel = [&] {
        constexpr int bytes_per_copy = T::BLOCK_SIZE * 16;
        static_for<(T::SFA_BYTES + bytes_per_copy - 1) / bytes_per_copy>([&](auto i) {
            constexpr int copy = decltype(i)::value;
            const int linear = ssfa_offsets[copy] + lane_id * 16;
            const int group = linear / T::B_M;
            if (group + 1 < loops && group < T::SCALE_GROUPS - 1)
                async_load<16>(g_sfa, s_sfa.ptr + ssfa_offsets[copy], gsfa_offsets[copy]);
        });
    };
    auto load_sfb_panel = [&] {
        const int sf_group = wave_id * 64 + lane_id;
        if (sf_group < loops) {
            const unsigned lo = load<1>(g_sfb, gsfb_offsets[0])[0];
            const unsigned hi = load<1>(g_sfb, gsfb_offsets[1])[0];
            store<1>(s_sfb, vector_t<unsigned short, 1>{static_cast<unsigned short>(lo | (hi << 8))}, u_rsfb(sf_group, 0_I));
        }
    };
    auto repack_sfa_panel = [&] {
        if constexpr (T::PACKED_SCALE) {
            // Repack four M16 scale bytes into a dword once, before accumulators
            // become live. A main-loop scale read then feeds four MFMAs directly.
            constexpr int words_per_group = T::B_M / 4;
            constexpr int rounds = (T::SCALE_GROUPS - 1) * words_per_group / T::BLOCK_SIZE + 1;
            array<unsigned, rounds> packed{};
            s_waitcnt_vmcnt(number<0>{});
            s_waitcnt_lgkmcnt(number<0>{});
            sync_threads();
            static_for<rounds>([&](auto i) {
                constexpr int step = decltype(i)::value;
                const int word = u_psfa(step, 0_I);
                const int group = word / words_per_group;
                if (group + 1 < loops && group < T::SCALE_GROUPS - 1)
                    static_for<4>([&](auto b) {
                        packed[step] |=
                            static_cast<unsigned>(load<1>(s_sfa, u_sfa_repack(step, b))[0]) <<
                            (decltype(b)::value * 8);
                    });
            });
            s_waitcnt_lgkmcnt(number<0>{});
            sync_threads();
            auto s_packed = make_smem(reinterpret_cast<unsigned*>(s_sfa.ptr));
            static_for<rounds>([&](auto i) {
                constexpr int step = decltype(i)::value;
                const int word = u_psfa(step, 0_I);
                const int group = word / words_per_group;
                if (group + 1 < loops && group < T::SCALE_GROUPS - 1)
                    store<1>(s_packed, vector_t<unsigned, 1>{packed[step]}, word);
            });
            s_waitcnt_lgkmcnt(number<0>{});
            sync_threads();
        }
    };

    auto read_scale_a = [&](int k, auto half_i) {
        constexpr int half = decltype(half_i)::value;
        unsigned result = 0;
        if (k == 0)
            return first_scale_a[half];
        if constexpr (T::PACKED_SCALE) {
            auto s_packed = make_smem(reinterpret_cast<unsigned*>(s_sfa.ptr));
            result = load<1>(s_packed, (k - 1) * (T::B_M / 4) + u_rsfa_packed(half, 0_I))[0];
        } else {
            static_for<T::E_M>([&](auto i) {
                constexpr int m = decltype(i)::value;
                result |= static_cast<unsigned>(load<1>(s_sfa, (k - 1) * T::B_M + u_rsfa.offsets[half * T::E_M + m])[0]) << (m * 8);
            });
        }
        return result;
    };
    auto issue_tile = [&](auto stage_i, int k) {
        constexpr int first = decltype(stage_i)::value * 2;
        load_a(number<first>{}, k);
        load_a(number<first + 1>{}, k);
        load_b(number<first>{}, k);
        load_b(number<first + 1>{}, k);
    };
    auto read_b = [&](auto stage_i, auto n_i) {
        constexpr int n = decltype(n_i)::value;
        constexpr int half = n / T::E_N;
        constexpr int slot = decltype(stage_i)::value * 2 + half;
        v_b[n] = read_operand_fragment<typename MMA::vtype_b>(
            s_b, slot * T::B_SLOT, u_rb, number<n % T::E_N>{});
    };
    auto consume_tile = [&](auto stage_i, int k) {
        constexpr int stage = decltype(stage_i)::value;
        s_waitcnt_vmcnt(0_I);
        s_waitcnt_lgkmcnt(0_I);
        sync_threads();
        if (k + 1 < loops)
            issue_tile(number<1 - stage>{}, k + 1);

        static_for<2>([&](auto half_i) {
            constexpr int half = decltype(half_i)::value;
            scale_a[half] = read_scale_a(k, half_i);
            static_for<T::E_M>([&](auto m_i) {
                constexpr int m = decltype(m_i)::value;
                v_a[half][m] = read_operand_fragment<typename MMA::vtype_a>(
                    s_a, (stage * 2 + half) * T::A_SLOT, u_ra, m_i);
            });
        });
        scale_b = load<1>(s_sfb, u_rsfb(k, 0_I))[0];
        static_for<2 * T::E_N>([&](auto n_i) { read_b(stage_i, n_i); });
        s_waitcnt_lgkmcnt(0_I);

        static_for<2 * T::E_N>([&](auto n_i) {
            constexpr int n = decltype(n_i)::value;
            constexpr int half_n = n / T::E_N;
            static_for<2>([&](auto half_i) {
                constexpr int half_m = decltype(half_i)::value;
                static_for<T::E_M>([&](auto m_i) {
                    constexpr int m = decltype(m_i)::value;
                    constexpr int ci = ((half_m * 2 + half_n) * T::E_M + m) * T::E_N + n % T::E_N;
                    v_c[ci] = MMA{}(v_a[half_m][m], v_b[n], v_c[ci],
                        static_cast<int>(scale_a[half_m]), static_cast<int>(scale_b), m_i, number<half_n>{});
                });
            });
        });
    };

    // ===== Prologue =====
    // Publish scales, then issue the first A/B tile.
    load_sfa_first();
    load_sfa_panel();
    load_sfb_panel();
    repack_sfa_panel();
    issue_tile(0_I, 0);

    // ===== Main loop =====
    // Alternate the two LDS stages, including the final K tile.
    // All waves publish together; each B fragment is reused across both M halves.
#pragma nounroll
    for (int k = 0; k < loops; k += 2) {
        consume_tile(0_I, k);
        if (k + 1 < loops)
            consume_tile(1_I, k + 1);
    }

    // ===== Epilogue =====
    // Drain matrix reads before reusing their LDS slots for C.
    s_waitcnt_vmcnt(0_I);
    s_waitcnt_lgkmcnt(0_I);
    sync_threads();

    // ===== Output writeback =====
    // Reuse matrix LDS for BF16 output.
    // Stage the MFMA's row fragments, then write contiguous BF16 vectors.
    static_for<4 * T::E_M * T::E_N>([&](auto ci) {
        constexpr int index = decltype(ci)::value;
        constexpr int quadrant = index / (T::E_M * T::E_N);
        constexpr int mr = index / T::E_N % T::E_M;
        constexpr int nr = index % T::E_N;
        const auto converted = cast<D_C>(v_c[index]);
        static_for<4>([&](auto e) {
            const vector_t<D_C, 1> value{converted[decltype(e)::value]};
            store<1>(s_c, value, u_sc(quadrant / 2, quadrant % 2, mr, nr, e));
        });
    });

    s_waitcnt_lgkmcnt(number<0>{});
    sync_threads();
    static_for<T::OUTPUT_PASSES>([&](auto i) {
        constexpr int copy = decltype(i)::value;
        store<8>(g_c, load<8>(s_c, u_rc(copy, 0_I)), u_gc(copy, 0_I), 0, number<2>{});
    });
}
#endif // gfx950
