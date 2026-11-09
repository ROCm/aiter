// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>
#include <opus/hip_minimal.hpp>
#include <opus/opus.hpp>
#include "../opus_gemm_utils.cuh"
#include "opus_gemm_traits_a8w8_blockscale_mxscale_bpreshuffle_direct_b_gfx950.cuh"
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx950__)
namespace opus_gemm_bpreshuffle_direct_b_layout {
using opus::operator""_I;
using namespace opus;
// Complete global/LDS A tiles; the XOR ownership matches the normal B pipeline.
// A: global copy, LDS stage and MFMA fragment layouts.
template<class T>
__device__ inline auto make_layout_ga(int lane_id, int wave_id, int stride_a) {
    constexpr int threads_k = T::B_K / T::VEC_A;
    constexpr int rows_per_wave = T::WARP_SIZE / threads_k;
    constexpr auto ga_block_shape = opus::make_tuple(
        opus::number<T::A_REPEATS>{},
        opus::number<T::A_COPY_N>{},
        opus::number<T::A_COPY_M>{},
        opus::number<rows_per_wave>{},
        opus::number<threads_k>{},
        opus::number<T::VEC_A>{});
    constexpr auto ga_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}));
    return opus::make_layout<T::VEC_A>(
        ga_block_shape,
        opus::unfold_x_stride(ga_block_dim, ga_block_shape, opus::tuple{stride_a, 1_I}),
        opus::unfold_p_coord(ga_block_dim, opus::tuple{wave_id % T::A_COPY_N, wave_id / T::A_COPY_N,
            lane_id / threads_k, lane_id % threads_k}));
}
template<class T>
__device__ inline auto make_layout_sa(int wave_id) {
    constexpr int smem_linear_wave = T::WARP_SIZE * T::VEC_A;
    constexpr auto sa_block_shape = opus::make_tuple(
        opus::number<T::A_REPEATS>{},
        opus::number<T::A_COPY_N>{},
        opus::number<T::A_COPY_M>{},
        opus::number<T::VEC_A>{});
    constexpr auto sa_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}));
    return opus::make_layout<T::VEC_A>(
        sa_block_shape,
        opus::unfold_x_stride(sa_block_dim, sa_block_shape, opus::tuple{opus::number<smem_linear_wave>{}, 1_I}),
        opus::unfold_p_coord(sa_block_dim, opus::tuple{wave_id % T::A_COPY_N, wave_id / T::A_COPY_N}));
}
// All A MFMA offsets for this wave, including both K64 halves of every K128.
template<class T>
__device__ inline auto make_layout_ra(int lane_id, int wave_id_m) {
    constexpr auto ra_block_shape = opus::make_tuple(
        opus::number<T::MR>{},
        opus::number<T::T_M>{},
        opus::number<T::W_M>{},
        opus::number<T::KS>{},
        opus::number<T::A_CHUNKS_PER_FRAGMENT>{},
        opus::number<T::WARP_SIZE / T::W_M>{},
        opus::number<T::VEC_A>{});
    constexpr auto ra_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}, opus::y_dim{}, opus::p_dim{}, opus::y_dim{}));
    return opus::make_layout<T::VEC_A>(
        ra_block_shape,
        opus::unfold_x_stride(ra_block_dim, ra_block_shape, opus::tuple{opus::number<T::B_K>{}, 1_I}),
        opus::unfold_p_coord(ra_block_dim, opus::tuple{wave_id_m, lane_id % T::W_M, lane_id / T::W_M}));
}
// B: preshuffled global panels map directly to the VGPR queue.
template<class T>
__device__ inline auto make_layout_rb(int lane_id, int wave_id_n, int stride_b) {
    constexpr auto rb_block_shape = opus::make_tuple(
        opus::number<T::B_K / T::W_K>{},
        opus::number<T::T_N>{},
        opus::number<T::NR>{},
        opus::number<T::B_CHUNKS_PER_FRAGMENT>{},
        opus::number<T::WARP_SIZE / T::W_N>{},
        opus::number<T::W_N>{},
        opus::number<T::VEC_B>{});
    constexpr auto rb_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}),
        opus::make_tuple(opus::y_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::y_dim{}));
    return opus::make_layout<T::VEC_B>(
        rb_block_shape,
        opus::unfold_x_stride(
            rb_block_dim, rb_block_shape,
            opus::tuple{opus::number<T::W_N * T::W_K>{}, T::W_N * stride_b,
                opus::number<T::WARP_SIZE * T::VEC_B>{}, opus::number<1>{}}),
        opus::unfold_p_coord(rb_block_dim, opus::tuple{wave_id_n, lane_id / T::W_N, lane_id % T::W_N}));
}
// Full scale tiles. Runtime row bounds and K-window movement belong to the load.
// SFA: global columns, LDS bytes and MFMA scale operands.
template<class T>
__device__ inline auto make_layout_gsfa(int lane_id, int wave_id) {
    constexpr auto gsfa_block_shape = opus::make_tuple(
        opus::number<T::SA_PASSES>{},
        opus::number<T::NUM_WAVES>{},
        opus::number<T::WARP_SIZE>{},
        opus::number<T::VEC_SCALE_A>{});
    constexpr auto gsfa_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}, opus::y_dim{}));
    return opus::make_layout<T::VEC_SCALE_A>(
        gsfa_block_shape,
        opus::unfold_x_stride(gsfa_block_dim, gsfa_block_shape, opus::tuple{1_I}),
        opus::unfold_p_coord(gsfa_block_dim, opus::tuple{wave_id, lane_id}));
}
template<class T>
__device__ inline auto make_layout_ssfa(int wave_id) {
    constexpr int spacing = T::VEC_SCALE_A < T::VEC_SCALE_B ? T::VEC_SCALE_B : T::VEC_SCALE_A;
    constexpr auto ssfa_block_shape = opus::make_tuple(
        opus::number<T::SA_PASSES>{},
        opus::number<T::NUM_WAVES>{},
        opus::number<T::WARP_SIZE>{},
        opus::number<T::VEC_SCALE_A>{});
    constexpr auto ssfa_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}));
    return opus::make_layout<T::VEC_SCALE_A>(
        ssfa_block_shape,
        opus::unfold_x_stride(ssfa_block_dim, ssfa_block_shape, opus::tuple{opus::number<spacing>{}, 1_I}),
        opus::unfold_p_coord(ssfa_block_dim, opus::tuple{wave_id, 0_I}));
}
template<class T>
__device__ inline auto make_layout_rsfa(int lane_id, int wave_id_m) {
    constexpr auto rsfa_block_shape = opus::make_tuple(
        opus::number<T::KS>{},
        opus::number<T::MR>{},
        opus::number<T::T_M>{},
        opus::number<T::W_M>{},
        opus::number<T::VEC_SF>{});
    constexpr auto rsfa_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}),
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}, opus::y_dim{}));
    return opus::make_layout<T::VEC_SF>(
        rsfa_block_shape,
        opus::unfold_x_stride(rsfa_block_dim, rsfa_block_shape,
            opus::tuple{opus::number<T::B_M * T::SA_SPACING>{}, opus::number<T::SA_SPACING>{}}),
        opus::unfold_p_coord(rsfa_block_dim, opus::tuple{wave_id_m, lane_id % T::W_M}));
}
// SFB: global scale groups and LDS reads for each wave.
template<class T>
__device__ inline auto make_layout_gsfb(int stride_sfb) {
    constexpr auto gsfb_block_shape = opus::make_tuple(
        opus::number<T::SB_ROWS>{},
        opus::number<T::SCALE_COLS>{});
    constexpr auto gsfb_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}),
        opus::make_tuple(opus::y_dim{}));
    return opus::make_layout(
        gsfb_block_shape,
        opus::unfold_x_stride(gsfb_block_dim, gsfb_block_shape, opus::tuple{stride_sfb, opus::number<1>{}}),
        opus::unfold_p_coord(gsfb_block_dim, opus::tuple{}));
}
template<class T>
__device__ inline constexpr auto make_layout_ssfb() {
    constexpr auto ssfb_block_shape = opus::make_tuple(
        opus::number<T::NUM_WAVES>{},
        opus::number<T::WARP_SIZE>{},
        opus::number<T::VEC_SCALE_B>{});
    constexpr auto ssfb_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::y_dim{}),
        opus::make_tuple(opus::y_dim{}));
    return opus::make_layout(
        ssfb_block_shape,
        opus::unfold_x_stride(ssfb_block_dim, ssfb_block_shape, opus::tuple{opus::number<T::VEC_SCALE_B>{}, opus::number<1>{}}),
        opus::unfold_p_coord(ssfb_block_dim, opus::tuple{}));
}
template<class T>
__device__ inline auto make_layout_rsfb(int wave_id_n) {
    constexpr int groups = (T::B_N / T::T_N + T::GROUP_N - 1) / T::GROUP_N;
    constexpr auto rsfb_block_shape = opus::make_tuple(
        opus::number<T::KS>{},
        opus::number<groups>{},
        opus::number<T::SB_ROWS>{});
    constexpr auto rsfb_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}),
        opus::make_tuple(opus::y_dim{}),
        opus::make_tuple(opus::p_dim{}));
    return opus::make_layout(
        rsfb_block_shape,
        opus::unfold_x_stride(rsfb_block_dim, rsfb_block_shape,
            opus::tuple{opus::number<T::VEC_SCALE_B>{},
                opus::number<T::SCALE_COLS * T::VEC_SCALE_B>{},
                opus::number<T::SCALE_COLS * T::VEC_SCALE_B>{}}),
        opus::unfold_p_coord(rsfb_block_dim, opus::tuple{wave_id_n * (T::B_N / T::T_N) / T::GROUP_N}));
}
// MFMA C coordinates mapped to global output addresses, including FP32 partials.
// C: MFMA fragments, LDS stripes and contiguous output vectors.
template<class T>
__device__ inline auto make_layout_gc_fragment(int lane_id, int wave_id_m, int wave_id_n, int stride_c) {
    constexpr auto gc_block_shape = opus::make_tuple(
        opus::number<T::MR>{},
        opus::number<T::T_M>{},
        opus::number<T::W_M>{},
        opus::number<T::T_N>{},
        opus::number<T::NR>{},
        opus::number<T::WARP_SIZE / T::W_M>{},
        opus::number<T::VEC_C>{});
    constexpr auto gc_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}));
    return opus::make_layout<T::VEC_C>(
        gc_block_shape,
        opus::unfold_x_stride(gc_block_dim, gc_block_shape, opus::tuple{stride_c, opus::number<T::W_N>{}, 1_I}),
        opus::unfold_p_coord(
            gc_block_dim, opus::tuple{wave_id_m, lane_id % T::W_M,
                wave_id_n, lane_id / T::W_M}));
}
// The same MFMA coordinates map to the padded LDS output stripe.
template<class T>
__device__ inline auto make_layout_sc_fragment(int lane_id, int wave_id_m, int wave_id_n) {
    return make_layout_gc_fragment<T>(lane_id, wave_id_m, wave_id_n, T::B_N + T::C_PADDING);
}
template<class T, int Rows>
__device__ inline auto make_layout_sc_vector(int lane_id, int wave_id) {
    constexpr int copies = (Rows * (T::B_N / T::VEC_OUTPUT) + T::BLOCK_SIZE - 1) / T::BLOCK_SIZE;
    constexpr auto sc_block_shape = opus::make_tuple(
        opus::number<copies>{},
        opus::number<T::NUM_WAVES>{},
        opus::number<T::WARP_SIZE>{},
        opus::number<T::VEC_OUTPUT>{});
    constexpr auto sc_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}, opus::y_dim{}));
    auto u = opus::make_layout<T::VEC_OUTPUT>(
        sc_block_shape,
        opus::unfold_x_stride(sc_block_dim, sc_block_shape, opus::tuple{opus::number<1>{}}),
        opus::unfold_p_coord(sc_block_dim, opus::tuple{wave_id, lane_id}));
    opus::static_for<copies>([&](auto i) {
        const int index = decltype(i)::value * T::BLOCK_SIZE + wave_id * T::WARP_SIZE + lane_id;
        u.offsets[i] =
            (index / (T::B_N / T::VEC_OUTPUT)) * (T::B_N + T::C_PADDING) +
            (index % (T::B_N / T::VEC_OUTPUT)) * T::VEC_OUTPUT;
    });
    return u;
}
template<class T, int Rows>
__device__ inline auto make_layout_gc_vector(int lane_id, int wave_id, int stride_c) {
    constexpr int copies = (Rows * (T::B_N / T::VEC_OUTPUT) + T::BLOCK_SIZE - 1) / T::BLOCK_SIZE;
    constexpr auto gc_block_shape = opus::make_tuple(
        opus::number<copies>{},
        opus::number<T::NUM_WAVES>{},
        opus::number<T::WARP_SIZE>{},
        opus::number<T::VEC_OUTPUT>{});
    constexpr auto gc_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}, opus::y_dim{}));
    auto u = opus::make_layout<T::VEC_OUTPUT>(
        gc_block_shape,
        opus::unfold_x_stride(gc_block_dim, gc_block_shape, opus::tuple{opus::number<1>{}}),
        opus::unfold_p_coord(gc_block_dim, opus::tuple{wave_id, lane_id}));
    opus::static_for<copies>([&](auto i) {
        const int index = decltype(i)::value * T::BLOCK_SIZE + wave_id * T::WARP_SIZE + lane_id;
        u.offsets[i] =
            (index / (T::B_N / T::VEC_OUTPUT)) * stride_c +
            (index % (T::B_N / T::VEC_OUTPUT)) * T::VEC_OUTPUT;
    });
    return u;
}
} // namespace opus_gemm_bpreshuffle_direct_b_layout
template<class Traits, bool ShortK>
__device__ __forceinline__ void opus_gemm_bpreshuffle_direct_b_body_impl(
    opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 kargs) {
    using namespace opus;
    namespace layout = opus_gemm_bpreshuffle_direct_b_layout;
    using T = opus::remove_cvref_t<Traits>;
    using D_A = fp8_t;
    using D_B = fp8_t;
    using D_SF = unsigned char;
    using D_C = std::conditional_t<T::SPLIT_K == 1, bf16_t, fp32_t>;
    using Index = unsigned;
    // Tile geometry and K-panel resource bounds.
    constexpr int BM = T::B_M;
    constexpr int BN = T::B_N;
    constexpr int NB = T::STAGES;
    constexpr int BD = T::B_AHEAD;
    constexpr int BS = T::B_SETS;
    constexpr int MR = T::MR;
    constexpr int NR = T::NR;
    constexpr int SC = T::SCALE_COLS;
    constexpr int BK = T::B_K;
    constexpr int KS = T::KS;
    constexpr bool resident_six = ShortK && BM == 48 && BN == 256 &&
        T::T_M == 1 && T::T_N == 4 && BD == 1 &&
        ((BK == 128 && NB == 9 && SC == 8) || (BK == 256 && NB == 7 && SC == 12)) &&
        T::VEC_SCALE_A == 16 && T::WG_PER_CU == 1 && T::SPLIT_K == 1 &&
        T::CACHECTL_A == 0 && T::B_CACHE == 0;
    constexpr bool resident_three = ShortK && BM == 32 && BN == 64 && BK == 512 &&
        T::T_M == 1 && T::T_N == 4 && NB == 5 && BD == 1 && SC == 16 &&
        T::VEC_SCALE_A == 16 && T::WG_PER_CU == 1 && T::SPLIT_K == 1 &&
        T::CACHECTL_A == 0 && T::B_CACHE == 0;
    constexpr bool resident362 = ShortK && BM == 48 && BN == 64 && BK == 256 &&
        T::T_M == 1 && T::T_N == 4 && NB == 7 && BD == 1 && SC == 12 &&
        T::VEC_SCALE_A == 4 && T::WG_PER_CU == 2 && T::SPLIT_K == 1 &&
        T::CACHECTL_A == 17 && T::B_CACHE == 1;
    constexpr bool defer_k_slice = ShortK && BM == 64 && BN == 64 && BK == 256 && NB == 7 && BD == 1 && SC == 12 &&
        T::T_M == 1 && T::T_N == 4 && T::WG_PER_CU == 1 && T::SPLIT_K == 1 &&
        T::VEC_SCALE_A == 4 && T::CACHECTL_A == 0 && T::B_CACHE == 2;
    constexpr bool defer_a_slice = defer_k_slice || (ShortK && BM == 64 && BN == 64 && BK == 256 &&
        NB == 5 && BD == 1 && SC == 8 && T::T_M == 1 && T::T_N == 4 && T::WG_PER_CU == 1 &&
        T::SPLIT_K == 1 && T::VEC_SCALE_A == 16 && T::CACHECTL_A == 1 && T::B_CACHE == 17);
    constexpr bool full_resident = ShortK && (resident_six || resident_three || resident362 || T::RESIDENT_K128_SIX ||
        (BM == 64 && BN == 64 && BK == 256 && T::T_M == 1 && T::T_N == 4 &&
         NB == 7 && BD == 1 && SC == 12 && T::VEC_SCALE_A == 4 &&
         T::WG_PER_CU == 1 && T::SPLIT_K == 1 && T::CACHECTL_A == 0 && T::B_CACHE == 2) ||
        (BM == 64 && BN == 64 && BK == 256 && T::T_M == 1 && T::T_N == 4 &&
         NB == 5 && BD == 1 && SC == 8 && T::VEC_SCALE_A == 16 &&
         T::WG_PER_CU == 1 && T::SPLIT_K == 1 && T::CACHECTL_A == 1 && T::B_CACHE == 17) ||
        (BM == 64 && BN == 128 && BK == 128 && NB == 4 && BD == 1 && SC == 4 &&
         T::T_M == 1 && T::T_N == 4 && T::SPLIT_K == 1 && T::CACHECTL_A == 0 && T::B_CACHE == 0 &&
         ((T::WG_PER_CU == 1 && T::VEC_SCALE_A == 4) || (T::WG_PER_CU == 2 && T::VEC_SCALE_A == 16))) ||
        (BM == 48 && BN == 64 && BK == 256 && NB == 7 && BD == 1 && SC == 12 &&
         T::T_M == 1 && T::T_N == 4 && T::WG_PER_CU == 2 && T::SPLIT_K == 1 &&
         T::VEC_SCALE_A == 4 && T::CACHECTL_A == 0 && T::B_CACHE == 0));
    static_assert(T::SPLIT_K == 1 || T::SPLIT_K == 2 || T::SPLIT_K == 4);
    // Runtime dimensions and contiguous tensor strides.
    const int total_k = ShortK ? __builtin_amdgcn_readfirstlane(kargs.k) : kargs.k;
    __builtin_assume(total_k > 0 && total_k % (T::SPLIT_K * BK) == 0);
    __builtin_assume(static_cast<unsigned>(kargs.m) - 1 < 1024);
    __builtin_assume(kargs.n > 0 && kargs.n % T::GROUP_N == 0);
    kargs.stride_a   = kargs.stride_b = total_k;
    kargs.stride_c   = kargs.n;
    kargs.stride_sfa = kargs.m;
    kargs.stride_sfb = total_k / T::GROUP_K;
    const int part_k     = total_k / T::SPLIT_K;
    const int P          = part_k / BK;
    const int part_sc    = part_k / T::GROUP_K;
    const int split      = T::SPLIT_K > 1 ? block_id_z() : 0;
    const int part_start = split * part_k;
    if constexpr (ShortK)
        __builtin_assume(P <= T::RESIDENT_BYTES / T::STAGE_BYTES);
    else
        __builtin_assume(P > T::RESIDENT_BYTES / T::STAGE_BYTES);
    const int M = kargs.m;
    const int N = kargs.n;
    const int stride_a = total_k;
    const int stride_b = total_k;
    const int stride_c = kargs.stride_c;
    static_assert(BK == 128 || BK == 256 || BK == 512);
    static_assert(T::T_M * T::T_N * T::WARP_SIZE == T::BLOCK_SIZE);
    static_assert(T::MR * T::T_M * T::W_M == T::B_M && T::NR * T::T_N * T::W_N == T::B_N);
    // Tile and thread coordinates.
    const Index tid       = thread_id_x();
    const Index lane_id   = tid % T::WARP_SIZE;
    const Index wave_id   = __builtin_amdgcn_readfirstlane(tid / T::WARP_SIZE);
    const Index wave_id_n = wave_id % T::T_N;
    const Index wave_id_m = T::T_M == 1 ? 0 : wave_id / T::T_N;
    __builtin_assume(static_cast<unsigned>(tid) < T::BLOCK_SIZE);
    __builtin_assume(static_cast<unsigned>(wave_id) < T::NUM_WAVES);
    const int row        = block_id_y() * BM;
    const int tile_n = [&] {
        if constexpr (BM == 16 && BN == 16 && T::NUM_WAVES == 1 && T::SPLIT_K == 1) {
            // The original tile order distributes adjacent groups across gfx950 XCDs.
            const int groups = N / (8 * BN);
            return block_id_x() % 8 * groups + block_id_x() / 8;
        } else {
            return block_id_x();
        }
    }();
    const int col        = tile_n * BN;
    const int valid_rows = M - row < BM ? M - row : BM;
    // Global memory views bound the final partial-M tile.
    auto g_a = make_gmem(
        reinterpret_cast<const D_A*>(kargs.ptr_a) + row * stride_a + part_start,
        static_cast<unsigned>(valid_rows * stride_a - part_start));
    auto g_b = make_gmem(
        reinterpret_cast<const D_B*>(kargs.ptr_b) + col * stride_b + part_start * T::W_N,
        static_cast<unsigned>(BN * stride_b - part_start * T::W_N));
    const auto* ptr_sfa =
        reinterpret_cast<const D_SF*>(kargs.ptr_sfa) + split * part_sc * kargs.stride_sfa;
    auto g_sfb = make_gmem(
        reinterpret_cast<const D_SF*>(kargs.ptr_sfb) + split * part_sc,
        static_cast<unsigned>((N / T::GROUP_N - 1) * kargs.stride_sfb + part_sc));
    auto g_c = make_gmem(
        reinterpret_cast<D_C*>(kargs.ptr_c) + static_cast<int64_t>(split) * M * stride_c +
            static_cast<int64_t>(row) * stride_c + col,
        static_cast<unsigned>(((valid_rows - 1) * stride_c + BN) * sizeof(D_C)));
    // One dynamic LDS arena; C reuses the matrix storage after the main loop.
    alignas(16) extern __shared__ char dynamic_matrix[];
    char* const smem_matrix = [] {
        if constexpr (T::UNIFORM_SHORT_QUEUE) {
            alignas(16) __shared__ char fixed_matrix[T::LDS_BYTES];
            return fixed_matrix;
        } else {
            return dynamic_matrix;
        }
    }();
    const int off_sa = T::UNIFORM_SHORT_QUEUE ? T::OFF_SA : T::off_sa(total_k);
    const int off_sb = T::UNIFORM_SHORT_QUEUE ? T::OFF_SB : T::off_sb(total_k);
    auto s_a = make_smem(reinterpret_cast<D_A*>(smem_matrix));
    auto s_sfa = make_smem(reinterpret_cast<D_SF*>(smem_matrix + off_sa));
    auto s_sfb = make_smem(reinterpret_cast<D_SF*>(smem_matrix + off_sb));
    auto s_c = make_smem(reinterpret_cast<D_C*>(smem_matrix));
    using Base =
        decltype(make_mfma<D_A, D_B, fp32_t>(
            number<T::W_M>{}, number<T::W_N>{}, number<T::W_K>{}, mfma_adaptor_swap_ab{}));
    array<typename Base::vtype_c, MR * NR> c{};
    array<array<array<typename Base::vtype_b, NR>, KS>, BS> b_regs;
    constexpr int VecScaleA = T::VEC_SCALE_A;
    // Apply the A LDS XOR permutation after constructing the logical layouts.
    constexpr int a_threads_k = T::B_K / T::VEC_A;
    constexpr int a_rows_per_wave = T::WARP_SIZE / a_threads_k;
    const int a_wave_n = wave_id % T::A_COPY_N;
    const int a_wave_m = T::A_COPY_M == 1 ? 0 : wave_id / T::A_COPY_N;
    auto u_ga = layout::make_layout_ga<T>(lane_id, wave_id, stride_a);
    static_for<T::A_REPEATS>([&](auto i) {
        const int row = ((int(i) * T::A_COPY_N + a_wave_n) * T::A_COPY_M + a_wave_m) *
            a_rows_per_wave + lane_id / a_threads_k;
        const int k = (lane_id % a_threads_k) * T::VEC_A;
        u_ga.offsets[i] += (k ^ ((row % a_threads_k) * T::VEC_A)) - k;
    });
    const auto u_sa = layout::make_layout_sa<T>(wave_id);
    auto u_ra = layout::make_layout_ra<T>(lane_id, wave_id_m);
    static_for<T::MR * T::KS * T::A_CHUNKS_PER_FRAGMENT>([&](auto i) {
        constexpr int m = decltype(i)::value / (T::KS * T::A_CHUNKS_PER_FRAGMENT);
        constexpr int ks = decltype(i)::value / T::A_CHUNKS_PER_FRAGMENT % T::KS;
        constexpr int half = decltype(i)::value % T::A_CHUNKS_PER_FRAGMENT;
        const int row = (m * T::T_M + wave_id_m) * T::W_M + lane_id % T::W_M;
        const int k_vector = ks * (T::W_K / T::VEC_A) +
            half * (T::WARP_SIZE / T::W_M) + lane_id / T::W_M;
        u_ra.offsets[i] +=
            ((k_vector ^ (row % a_threads_k)) - k_vector) * T::VEC_A;
    });
    const auto u_rb = layout::make_layout_rb<T>(lane_id, wave_id_n, stride_b);
    const auto u_gsfa = layout::make_layout_gsfa<T>(lane_id, wave_id);
    const auto u_ssfa = layout::make_layout_ssfa<T>(wave_id);
    const auto u_rsfa = layout::make_layout_rsfa<T>(lane_id, wave_id_m);
    auto u_gsfb = layout::make_layout_gsfb<T>(kargs.stride_sfb);
    const auto u_ssfb = layout::make_layout_ssfb<T>();
    const auto u_rsfb = layout::make_layout_rsfb<T>(wave_id_n);
    const auto u_gc = layout::make_layout_gc_fragment<T>(lane_id, wave_id_m, wave_id_n, stride_c);
    const auto u_sc = layout::make_layout_sc_fragment<T>(lane_id, wave_id_m, wave_id_n);
    // Prefetch one B panel directly into its VGPR queue slot.
    auto issue_b = [&](auto bi, int kt) {
        if constexpr ((ShortK && T::UNIFORM_SHORT_QUEUE) ||
                      (!ShortK && !(BM == 64 && BN == 128 && BK == 128 && T::T_M == 1 && T::T_N == 2 && NB == 4 && BD == 3 && T::WG_PER_CU == 1 && T::VEC_SCALE_A == 16 && T::SPLIT_K == 1 && T::CACHECTL_A == 1 && T::B_CACHE == 17) &&
                       ((BM == 64 && T::T_M == 1 && T::T_N == 2 && KS == 1 && NB == 4 && BD == 3) ||
                        (BM == 32 && BN == 64 && T::T_M == 1 && T::T_N == 4 && KS == 4 && NB == 3 && BD == 1)))) {
            g_b = make_gmem(
                reinterpret_cast<const D_B*>(kargs.ptr_b) + col * stride_b + part_start * T::W_N,
                kt < P ? static_cast<unsigned>(BN * stride_b - part_start * T::W_N) : 0u);
        }
        if constexpr (!ShortK && !(BM == 64 && BN == 128 && BK == 128 && T::T_M == 1 && T::T_N == 2 && NB == 4 && BD == 3 && T::WG_PER_CU == 1 && T::VEC_SCALE_A == 16 && T::SPLIT_K == 1 && T::CACHECTL_A == 1 && T::B_CACHE == 17))
            kt = kt < P ? kt : P - 1;
        constexpr int set = decltype(bi)::value;
        static_for<KS>([&](auto ki) {
            constexpr int ks = decltype(ki)::value;
            static_for<NR>([&](auto ni) {
                constexpr int n = decltype(ni)::value;
                constexpr int offset = (ks * NR + n) * 2;
                set_slice(b_regs[set][ks][n], load<16>(g_b, u_rb.offsets[offset],
                    kt * (BK * 16), number<T::B_CACHE>{}), 0_I, 16_I);
                set_slice(b_regs[set][ks][n], load<16>(g_b, u_rb.offsets[offset + 1],
                    kt * (BK * 16), number<T::B_CACHE>{}), 16_I, 32_I);
            });
        });
        __builtin_amdgcn_sched_barrier(0);
    };
    // Prefetch one A panel into a padded LDS stage.
    auto issue_a = [&](int slot, int kt) {
        if constexpr ((ShortK && T::UNIFORM_SHORT_QUEUE) ||
                      (!ShortK && !(BM == 64 && BN == 128 && BK == 128 && T::T_M == 1 && T::T_N == 2 && NB == 4 && BD == 3 && T::WG_PER_CU == 1 && T::VEC_SCALE_A == 16 && T::SPLIT_K == 1 && T::CACHECTL_A == 1 && T::B_CACHE == 17) &&
                       ((BM == 64 && T::T_M == 1 && T::T_N == 2 && KS == 1 && NB == 4 && BD == 3) ||
                        (BM == 32 && BN == 64 && T::T_M == 1 && T::T_N == 4 && KS == 4 && NB == 3 && BD == 1)))) {
            g_a = make_gmem(
                reinterpret_cast<const D_A*>(kargs.ptr_a) + row * stride_a + part_start,
                kt < P ? static_cast<unsigned>(valid_rows * stride_a - part_start) : 0u);
        }
        if constexpr (!ShortK && !(BM == 64 && BN == 128 && BK == 128 && T::T_M == 1 && T::T_N == 2 && NB == 4 && BD == 3 && T::WG_PER_CU == 1 && T::VEC_SCALE_A == 16 && T::SPLIT_K == 1 && T::CACHECTL_A == 1 && T::B_CACHE == 17))
            kt = kt < P ? kt : P - 1;
        if (T::A_COPY_THREADS == T::BLOCK_SIZE || wave_id < T::A_COPY_THREADS / T::WARP_SIZE) {
            if constexpr (ShortK && T::RESIDENT_K128_SIX) {
                const auto ga = layout_to_offsets<T::VEC_A>(u_ga);
                const auto sa = layout_to_offsets<T::VEC_A>(u_sa);
                static_for<decltype(ga)::size()>([&](auto copy_i) {
                    constexpr int i = decltype(copy_i)::value;
                    g_a.template _async_load<T::VEC_A>(s_a.ptr + sa[i] + slot * T::STAGE_BYTES,
                        ga[i], kt * BK, 0_I, number<T::CACHECTL_A>{});
                });
            } else {
                async_load<16>(g_a, s_a.ptr + slot * T::STAGE_BYTES,
                    u_ga, u_sa, kt * BK, 0_I, number<T::CACHECTL_A>{});
            }
        }
        __builtin_amdgcn_sched_barrier(0);
    };
    // Move the scale view; row and final-window bounds stay at the load site.
    auto issue_scale_panel = [&](int first_column) {
        const int remaining = part_sc - first_column;
        const int available = remaining < SC ? remaining : SC;
        auto window_sfa = make_gmem(
            ptr_sfa + first_column * kargs.stride_sfa,
            static_cast<unsigned>(available * kargs.stride_sfa));
        auto current_sfa = u_gsfa;
        static_for<T::SA_PASSES>([&](auto i) {
            const int index = current_sfa.offsets[i];
            const int local_row = index % BM;
            const int column = index / BM;
            current_sfa.offsets[i] = row + local_row < M ?
                column * kargs.stride_sfa + row + local_row : available * kargs.stride_sfa;
        });
        // Constrain scale offsets to VGPRs after all row bounds are applied.
#if defined(__AMDGCN__)
        for (int i = 0; i < decltype(current_sfa)::num_issues; ++i) {
            asm volatile("" : "+v"(current_sfa.offsets[i]));
        }
#endif
        if constexpr ((ShortK && (T::RESIDENT_K128_SIX || (full_resident && BK == 128))) ||
                      (!ShortK && BM == 64 && BN == 128 && BK == 256 &&
                       T::T_M == 1 && T::T_N == 8 && NB == 3 && BD == 1 &&
                       T::VEC_SCALE_A == 4)) {
            const auto gs = layout_to_offsets<VecScaleA>(current_sfa);
            const auto ss = layout_to_offsets<VecScaleA>(u_ssfa);
            static_for<decltype(gs)::size()>([&](auto copy_i) {
                constexpr int i = decltype(copy_i)::value;
                window_sfa.template _async_load<VecScaleA>(s_sfa.ptr + ss[i], gs[i], 0, 0_I);
            });
        } else {
            async_load<VecScaleA>(window_sfa,
                s_sfa.ptr, current_sfa, u_ssfa);
        }
        constexpr bool live_b = !ShortK &&
            (KS == 4 || (KS == 1 && BN >= 128) || BM <= 32);
        // Limit B producers only in the tile schedules that benefit from it.
        if (!live_b || tid < T::SB_ROWS * SC) {
            const int index = live_b ? tid : tid % (T::SB_ROWS * SC);
            const int group = index / SC;
            const int column = index % SC < available ? index % SC : available - 1;
            if constexpr (ShortK && (T::RESIDENT_K128_SIX || (full_resident && BK == 128)))
                g_sfb.template _async_load<1>(s_sfb.ptr + u_ssfb(wave_id, 0, 0),
                    u_gsfb(group, column), col / 128 * kargs.stride_sfb + first_column, 0_I);
            else
                async_load<1>(g_sfb, s_sfb.ptr + u_ssfb(wave_id, 0, 0), u_gsfb(group, column),
                    col / 128 * kargs.stride_sfb + first_column);
        }
    };
    // Consume one panel, overlapping its reads and MFMA with queue refills.
    auto consume_panel = [&](int t, int slot, auto bi, auto wai, auto wbi, auto fai, auto fbi) {
        constexpr int set   = decltype(bi)::value;
        constexpr int wa    = decltype(wai)::value;
        constexpr int wb    = decltype(wbi)::value;
        const bool future_a = fai;
        const bool future_b = fbi;
        // Wait for the A stage and replenish the B queue.
        __builtin_amdgcn_sched_barrier(0);
        if constexpr (wa >= 0)
            s_waitcnt_vmcnt(number<(wa > 63 ? 63 : wa)>{});
        if constexpr (ShortK && (T::RESIDENT_K128_SIX || (full_resident && (T::VEC_SCALE_A == 16 || BK == 128 || BM == 48)))) {
            if (P != T::RESIDENT_BYTES / T::STAGE_BYTES) s_waitcnt_lgkmcnt(0_I);
        } else if constexpr (!(ShortK && T::UNIFORM_SHORT_QUEUE) && !full_resident)
            s_waitcnt_lgkmcnt(0_I);
        if constexpr (decltype(fai)::value)
            if constexpr (T::NUM_WAVES > 1)
                sync_threads();
        if (future_b)
            issue_b(number<(set + BD) % BS>{}, t + BD);
        // Read A fragments and their A/B scales from the current LDS window.
        array<array<typename Base::vtype_a, MR>, KS> af;
        array<array<unsigned, MR>, KS> as;
        array<array<unsigned, NR>, KS> bs;
        array<unsigned, KS> panel_bs;
        if constexpr (resident_three) {
            static_assert(KS == 4 && NR == 1 && T::VEC_SCALE_B == 4);
            const auto raw = __builtin_bit_cast(array<unsigned, KS>,
                load<16>(s_sfb, u_rsfb(0, 0) + t * KS * 4));
            static_for<KS>([&](auto ki) { panel_bs[ki] = raw[ki]; });
        }
        constexpr bool deferred_a = !ShortK && BM == 64 && BN == 128 && BK == 256 &&
            T::T_M == 1 && ((T::T_N == 8 && NB == 3 && BD == 1 && T::WG_PER_CU == 1) || (T::T_N == 4 && NB == 3 && BD == 2 && T::WG_PER_CU == 2)) &&
            T::SPLIT_K == 1 && T::VEC_SCALE_A == 4 && T::CACHECTL_A == 0 && T::B_CACHE == 0;
        constexpr bool partial_a = deferred_a || (ShortK && (T::RESIDENT_K128_SIX || (resident_six && BK == 256))) ||
            (!ShortK && BM == 96 && BN == 128 && BK == 128 &&
            T::T_M == 1 && T::T_N == 4 && BD == 2 && T::VEC_SCALE_A == 16 &&
            T::WG_PER_CU == 1 && T::SPLIT_K == 1 && T::B_CACHE == 1 &&
            ((NB == 4 && SC == 64 && T::CACHECTL_A == 1) ||
             (NB == 3 && SC == 32 && T::CACHECTL_A == 17)));
        constexpr int initial_m = deferred_a ? 2 : (ShortK ? 2 : 3);
        auto read_a = [&](auto ki, auto mi) {
            constexpr int ks = decltype(ki)::value, m = decltype(mi)::value;
            constexpr int offset = (m * KS + ks) * 2;
            set_slice(af[ks][m], load<16>(s_a, slot * T::STAGE_BYTES + u_ra.offsets[offset]), 0_I, 16_I);
            set_slice(af[ks][m], load<16>(s_a, slot * T::STAGE_BYTES + u_ra.offsets[offset + 1]), 16_I, 32_I);
        };
        static_for<KS>([&](auto ki) {
            constexpr int ks    = decltype(ki)::value;
            const int scale_col =
                (ShortK && T::RESIDENT_BYTES / T::STAGE_BYTES <= SC / KS ? t * KS : t * KS % SC) + ks;
            static_for<NR>([&](auto ni) {
                constexpr int nr = decltype(ni)::value;
                const int group = (wave_id_n * NR * 16 + nr * 16) / 128;
                const int first_group = wave_id_n * NR * 16 / 128;
                if constexpr (resident_three)
                    bs[ks][nr] = panel_bs[ks];
                else
                    bs[ks][nr] = load<1>(s_sfb,
                        u_rsfb(ks, group - first_group) + (scale_col - ks) * 4)[0];
            });
            static_for<MR>([&](auto mi) {
                constexpr int m = decltype(mi)::value;
                if constexpr ((!partial_a || m < initial_m) && (!(resident362 || defer_a_slice || (resident_six && BK == 256)) || ks == 0) && (!resident_three || ks < 2)) read_a(ki, mi);
                as[ks][m] = load<1>(s_sfa,
                    u_rsfa.offsets[ks * MR + m] + (scale_col - ks) * BM * T::SA_SPACING)[0];
            });
        });
        __builtin_amdgcn_sched_barrier(0);
        // Refill the consumed A stage before computing this panel.
        if (future_a)
            issue_a(slot == 0 ? NB - 1 : slot - 1, t + NB - 1);
        __builtin_amdgcn_sched_barrier(0);
        if constexpr (NB > 2 && wb >= 0)
            s_waitcnt_vmcnt(number<(wb > 63 ? 63 : wb)>{});
        s_waitcnt_lgkmcnt(0_I);
        // Compute the K128 MFMA slices for this panel.
        static_for<KS>([&](auto ki) {
            constexpr int ks = decltype(ki)::value;
            auto compute = [&](auto mi, auto ni) {
                constexpr int m = decltype(mi)::value, n = decltype(ni)::value, ci = m * NR + n;
                c[ci] = Base{}(af[ks][m], b_regs[set][ks][n], c[ci],
                    static_cast<int>(as[ks][m]), static_cast<int>(bs[ks][n]), 0_I, 0_I);
                if constexpr (partial_a && n == (deferred_a ? NR - 1 : 0) && m + initial_m < MR) {
                    __builtin_amdgcn_sched_barrier(0);
                    read_a(ki, number<m + initial_m>{});
                    __builtin_amdgcn_sched_barrier(0);
                } else if constexpr (resident362 && n == 0 && m == 0 && ks + 1 < KS) {
                    static_assert(KS == 2 && MR == 3);
                    __builtin_amdgcn_sched_barrier(0);
                    static_for<MR>([&](auto next_m) { read_a(number<ks + 1>{}, next_m); });
                    __builtin_amdgcn_sched_barrier(0);
                } else if constexpr (defer_a_slice && ks == 0 && m == 1 && n == NR - 1) {
                    static_assert(KS == 2 && MR == 4);
                    __builtin_amdgcn_sched_barrier(0);
                    static_for<MR>([&](auto next_m) { read_a(1_I, next_m); });
                    __builtin_amdgcn_sched_barrier(0);
                } else if constexpr (resident_three && n == 0 && m == 0 && ks + 2 < KS) {
                    __builtin_amdgcn_sched_barrier(0);
                    static_for<MR>([&](auto next_m) { read_a(number<ks + 2>{}, next_m); });
                    __builtin_amdgcn_sched_barrier(0);
                }
                if constexpr (resident_six && BK == 256 && ks == 0 && m == MR - 2 && n == NR - 1) {
                    __builtin_amdgcn_sched_barrier(0);
                    static_for<2>([&](auto next_m) { read_a(1_I, next_m); });
                    __builtin_amdgcn_sched_barrier(0);
                }
            };
            if constexpr (partial_a && !deferred_a && !(resident_six && BK == 256))
                static_for<NR>([&](auto ni) {
                    static_for<MR>([&](auto mi) { compute(mi, ni); });
                });
            else
                static_for<MR>([&](auto mi) {
                    static_for<NR>([&](auto ni) { compute(mi, ni); });
                });
        });
        __builtin_amdgcn_sched_barrier(0);
    };
    if constexpr (full_resident) {
        constexpr bool early_b0 = BM == 64 && BN == 64 && BK == 256 && NB == 7 && BD == 1 && SC == 12 &&
            T::T_M == 1 && T::T_N == 4 && T::WG_PER_CU == 1 && T::SPLIT_K == 1 &&
            T::VEC_SCALE_A == 4 && T::CACHECTL_A == 0 && T::B_CACHE == 2;
        static_assert(T::RESIDENT_BYTES / T::STAGE_BYTES <= SC / KS);
        auto roll_scales = [](int) {};
        auto begin_panels = [&] {
            // Prologue.
            // Issue B, the first A stage, both scale panels, then the remaining A stages.
            if constexpr (!early_b0) static_for<T::INITIAL_B>([&](auto i) {
                if constexpr (ShortK && T::UNIFORM_SHORT_QUEUE)
                    issue_b(i, decltype(i)::value);
                else if (decltype(i)::value < P)
                    issue_b(i, decltype(i)::value);
            });
            issue_a(0, 0);
            issue_scale_panel(0);
            __builtin_amdgcn_sched_barrier(0);
            static_for<T::INITIAL_A - 1>([&](auto i) {
                if constexpr (ShortK && T::UNIFORM_SHORT_QUEUE)
                    issue_a(decltype(i)::value + 1, decltype(i)::value + 1);
                else if (decltype(i)::value + 1 < P)
                    issue_a(decltype(i)::value + 1, decltype(i)::value + 1);
            });
            if constexpr (ShortK && T::RESIDENT_BYTES / T::STAGE_BYTES > T::INITIAL_A) {
        #pragma clang loop unroll(disable)
                for (int slot = T::INITIAL_A; slot < P; ++slot)
                    issue_a(slot, slot);
            }
        };
        auto run_short = [&](auto full) {
            // Short-K main loop: fixed resident A slots.
            constexpr int capacity = T::RESIDENT_BYTES / T::STAGE_BYTES;
            constexpr int full_panels = capacity - (resident_six && BK == 128 ? 2 : (resident_three ? 1 : (resident362 ? 3 : 0)));
            const int panels = decltype(full)::value ? full_panels : P;
            // Subgroup copiers publish all resident A before the other waves consume it.
            constexpr bool publish_all = T::A_COPY_THREADS != T::BLOCK_SIZE;
            if constexpr (publish_all) {
                s_waitcnt_vmcnt(0_I);
                s_waitcnt_lgkmcnt(0_I);
                if constexpr (T::NUM_WAVES > 1)
                    sync_threads();
            }
            static_for<capacity>([&](auto phase) {
                constexpr int slot = decltype(phase)::value;
                const int t        = slot;
                if (t < panels) {
                    if constexpr (capacity > SC / KS)
                        roll_scales(t);
                    if constexpr (T::UNIFORM_SHORT_QUEUE) {
                        constexpr int queued = capacity > BD ?
                            (slot < capacity - BD ? slot : capacity - BD) : 0;
                        constexpr int tail = capacity - slot - 1 < BD ? capacity - slot - 1 : BD - 1;
                        constexpr int wait = slot < BD ?
                            (capacity - slot - 1) * T::A_REPEATS + queued * T::B_ISSUES :
                            (tail + 1) * T::B_ISSUES;
                        s_waitcnt_vmcnt(number<(wait > 63 ? 63 : wait)>{});
                        s_waitcnt_lgkmcnt(0_I);
                        if constexpr (T::NUM_WAVES > 1)
                            sync_threads();
                    } else if constexpr (!publish_all && slot < BD) {
                        // VMEM wait operands are immediate; select them from the uniform queue length.
                        const int uniform_panels = __builtin_amdgcn_readfirstlane(panels);
                        static_for<capacity - slot>([&](auto remaining) {
                            constexpr int panels = slot + 1 + decltype(remaining)::value;
                            if (uniform_panels == panels) {
                                constexpr int queued_b = decltype(full)::value && full_panels == capacity && slot == 0 && T::A_REPEATS >= 4 && capacity == 6 ? 1 :
                                    (panels > BD ? (slot < panels - BD ? slot : panels - BD) : 0);
                                constexpr int wait =
                                    (panels - slot - 1) * T::A_REPEATS + queued_b * T::B_ISSUES;
                                s_waitcnt_vmcnt(number<(wait > 63 ? 63 : wait)>{});
                            }
                        });
                        s_waitcnt_lgkmcnt(0_I);
                        if constexpr (T::NUM_WAVES > 1)
                            sync_threads();
                    } else {
                        if constexpr (slot >= BD) {
                            const int pending = panels - t - 1 < BD ? panels - t - 1 : BD - 1;
                            const int uniform_pending = __builtin_amdgcn_readfirstlane(pending);
                            static_for<BD>([&](auto queued) {
                                constexpr int count = decltype(queued)::value;
                                if (uniform_pending == count) {
                                    constexpr int wait = count * T::B_ISSUES;
                                    s_waitcnt_vmcnt(number<(wait > 63 ? 63 : wait)>{});
                                }
                            });
                        }
                        // B[BD] follows all A DMA; publish the full resident
                        // tile once before later panels omit the barrier.
                        if constexpr (!publish_all && slot == BD) {
                            s_waitcnt_lgkmcnt(0_I);
                            if constexpr (T::NUM_WAVES > 1)
                                sync_threads();
                        }
                    }
                    if constexpr (T::UNIFORM_SHORT_QUEUE)
                        consume_panel(t, slot, number<slot % BS>{},
                             number<-1>{}, number<(slot >= BD ? 0 : -1)>{},
                             number<false>{}, number<(slot + BD < capacity)>{});
                    else
                        consume_panel(t, slot, number<slot % BS>{},
                             number<-1>{}, number<-1>{}, number<false>{},
                             t + BD < panels && !(decltype(full)::value && full_panels == capacity && slot == 0 && T::A_REPEATS >= 4 && capacity == 6));
                }
            });
        };
        constexpr int capacity = T::RESIDENT_BYTES / T::STAGE_BYTES;
        constexpr int full_panels = capacity - (resident_six && BK == 128 ? 2 : (resident_three ? 1 : (resident362 ? 3 : 0)));
        if constexpr (early_b0) issue_b(0_I, 0);
        if (P == full_panels) {
            if constexpr ((BM == 64 && BN == 64 && BK == 256 && NB == 7 && BD == 1 && SC == 12 &&
                          T::T_M == 1 && T::T_N == 4 && T::WG_PER_CU == 1 && T::SPLIT_K == 1 &&
                          T::VEC_SCALE_A == 4 && T::CACHECTL_A == 0 && T::B_CACHE == 2) || (resident_six && BK == 256)) {
                // Runtime full-capacity selection proves these descriptor extents.
                constexpr int full_k = full_panels * BK * T::SPLIT_K;
                g_a = make_gmem(reinterpret_cast<const D_A*>(kargs.ptr_a) + row * full_k,
                                static_cast<unsigned>(valid_rows * full_k));
                g_sfb = make_gmem(reinterpret_cast<const D_SF*>(kargs.ptr_sfb),
                                 static_cast<unsigned>(N / T::GROUP_N * (full_k / T::GROUP_K)));
                s_sfa = make_smem(reinterpret_cast<D_SF*>(smem_matrix + T::off_sa(full_k)));
                s_sfb = make_smem(reinterpret_cast<D_SF*>(smem_matrix + T::off_sb(full_k)));
                u_ga = layout::make_layout_ga<T>(lane_id, wave_id, full_k);
                static_for<T::A_REPEATS>([&](auto i) {
                    const int r = ((int(i) * T::A_COPY_N + a_wave_n) * T::A_COPY_M + a_wave_m) *
                        a_rows_per_wave + lane_id / a_threads_k;
                    const int k = (lane_id % a_threads_k) * T::VEC_A;
                    u_ga.offsets[i] += (k ^ ((r % a_threads_k) * T::VEC_A)) - k;
                });
                u_gsfb = layout::make_layout_gsfb<T>(full_k / T::GROUP_K);
            }
            if constexpr (!early_b0) static_for<T::INITIAL_B>([&](auto i) { issue_b(i, decltype(i)::value); });
            issue_a(0, 0);
            issue_scale_panel(0);
            if constexpr (full_panels == capacity && T::A_REPEATS >= 4 && capacity == 6) issue_b(1_I, 1);
            __builtin_amdgcn_sched_barrier(0);
            static_for<full_panels - 1>([&](auto i) {
                issue_a(decltype(i)::value + 1, decltype(i)::value + 1);
            });
            run_short(number<true>{});
        } else {
            begin_panels();
            run_short(number<false>{});
        }
    } else {
        // Prologue.
        // Issue B, the first A stage, both scale panels, then the remaining A stages.
        static_for<T::INITIAL_B>([&](auto i) {
            if constexpr (ShortK && T::UNIFORM_SHORT_QUEUE)
                issue_b(i, decltype(i)::value);
            else if (decltype(i)::value < P)
                issue_b(i, decltype(i)::value);
        });
        issue_a(0, 0);
        issue_scale_panel(0);
        __builtin_amdgcn_sched_barrier(0);
        static_for<T::INITIAL_A - 1>([&](auto i) {
            if constexpr (ShortK && T::UNIFORM_SHORT_QUEUE)
                issue_a(decltype(i)::value + 1, decltype(i)::value + 1);
            else if (decltype(i)::value + 1 < P)
                issue_a(decltype(i)::value + 1, decltype(i)::value + 1);
        });
        if constexpr (ShortK && T::RESIDENT_BYTES / T::STAGE_BYTES > T::INITIAL_A) {
    #pragma clang loop unroll(disable)
            for (int slot = T::INITIAL_A; slot < P; ++slot)
                issue_a(slot, slot);
        }
        // Scale-window rollover waits for the previous copies before reuse.
        constexpr int WA = (NB - 2) * (T::A_REPEATS + T::B_ISSUES);
        constexpr int WB = BD * T::B_ISSUES + (BD + 1) * T::A_REPEATS;
        auto roll_scales = [&](int t) {
            if (t > 0 && t * KS % SC == 0) {
                s_waitcnt_vmcnt(0_I);
                s_waitcnt_lgkmcnt(0_I);
                if constexpr (T::NUM_WAVES > 1)
                    sync_threads();
                issue_scale_panel(t * KS);
                s_waitcnt_vmcnt(0_I);
                if constexpr (T::NUM_WAVES > 1)
                    sync_threads();
            }
        };
        if constexpr (ShortK) {
            // Short-K main loop: fixed resident A slots.
            constexpr int capacity = T::RESIDENT_BYTES / T::STAGE_BYTES;
            // Subgroup copiers publish all resident A before the other waves consume it.
            constexpr bool publish_all = T::A_COPY_THREADS != T::BLOCK_SIZE;
            if constexpr (publish_all) {
                s_waitcnt_vmcnt(0_I);
                s_waitcnt_lgkmcnt(0_I);
                if constexpr (T::NUM_WAVES > 1)
                    sync_threads();
            }
            static_for<capacity>([&](auto phase) {
                constexpr int slot = decltype(phase)::value;
                const int t        = slot;
                if (t < P) {
                    if constexpr (capacity > SC / KS)
                        roll_scales(t);
                    if constexpr (T::UNIFORM_SHORT_QUEUE) {
                        constexpr int queued = capacity > BD ?
                            (slot < capacity - BD ? slot : capacity - BD) : 0;
                        constexpr int tail = capacity - slot - 1 < BD ? capacity - slot - 1 : BD - 1;
                        constexpr int wait = slot < BD ?
                            (capacity - slot - 1) * T::A_REPEATS + queued * T::B_ISSUES :
                            (tail + 1) * T::B_ISSUES;
                        s_waitcnt_vmcnt(number<(wait > 63 ? 63 : wait)>{});
                        s_waitcnt_lgkmcnt(0_I);
                        if constexpr (T::NUM_WAVES > 1)
                            sync_threads();
                    } else if constexpr (!publish_all && slot < BD) {
                        // VMEM wait operands are immediate; select them from the uniform queue length.
                        const int uniform_panels = __builtin_amdgcn_readfirstlane(P);
                        static_for<capacity - slot>([&](auto remaining) {
                            constexpr int panels = slot + 1 + decltype(remaining)::value;
                            if (uniform_panels == panels) {
                                constexpr int queued_b = panels > BD ?
                                    (slot < panels - BD ? slot : panels - BD) : 0;
                                constexpr int wait =
                                    (panels - slot - 1) * T::A_REPEATS + queued_b * T::B_ISSUES;
                                s_waitcnt_vmcnt(number<(wait > 63 ? 63 : wait)>{});
                            }
                        });
                        s_waitcnt_lgkmcnt(0_I);
                        if constexpr (T::NUM_WAVES > 1)
                            sync_threads();
                    } else {
                        if constexpr (slot >= BD) {
                            const int pending = P - t - 1 < BD ? P - t - 1 : BD - 1;
                            const int uniform_pending = __builtin_amdgcn_readfirstlane(pending);
                            static_for<BD>([&](auto queued) {
                                constexpr int count = decltype(queued)::value;
                                if (uniform_pending == count) {
                                    constexpr int wait = count * T::B_ISSUES;
                                    s_waitcnt_vmcnt(number<(wait > 63 ? 63 : wait)>{});
                                }
                            });
                        }
                        // B[BD] follows all A DMA; publish the full resident
                        // tile once before later panels omit the barrier.
                        if constexpr (!publish_all && slot == BD) {
                            s_waitcnt_lgkmcnt(0_I);
                            if constexpr (T::NUM_WAVES > 1)
                                sync_threads();
                        }
                    }
                    if constexpr (T::UNIFORM_SHORT_QUEUE)
                        consume_panel(t, slot, number<slot % BS>{},
                             number<-1>{}, number<(slot >= BD ? 0 : -1)>{},
                             number<false>{}, number<(slot + BD < capacity)>{});
                    else
                        consume_panel(t, slot, number<slot % BS>{},
                             number<-1>{}, number<-1>{}, number<false>{}, t + BD < P);
                }
            });
        } else {
            // Normal-K main loop: rolling A stages and B VGPR slots.
            // Keep one computation site per B register set in the runtime loop.
            // Clamped lookahead preserves the steady A/B wait ledger.
            consume_panel(0, 0, 0_I, number<(T::INITIAL_A - 1) * T::A_REPEATS>{}, number<-1>{},
                 number<true>{}, number<true>{});
            static_for<T::INITIAL_A - 1>([&](auto si) {
                constexpr int t = decltype(si)::value + 1;
                constexpr int wait = (NB - 2) * T::A_REPEATS + t * T::B_ISSUES;
                roll_scales(t);
                s_waitcnt_vmcnt(number<(wait > 63 ? 63 : wait)>{});
                consume_panel(t, t % NB, number<t % BS>{}, number<-1>{}, number<WB>{},
                     number<true>{}, number<true>{});
            });
            constexpr int phases = [] {
                int a = NB, b = BS;
                while (b) {
                    int remainder = a % b;
                    a = b;
                    b = remainder;
                }
                return NB / a * BS;
            }();
    #pragma clang loop unroll(disable)
            for (int base = T::INITIAL_A; base < P; base += phases) {
                static_for<phases>([&](auto phase) {
                    constexpr int offset = decltype(phase)::value;
                    const int t = base + offset;
                    if (t >= P)
                        return;
                    roll_scales(t);
                    s_waitcnt_vmcnt(number<(WA > 63 ? 63 : WA)>{});
                    consume_panel(t, (T::INITIAL_A + offset) % NB,
                         number<(T::INITIAL_A + offset) % BS>{},
                         number<-1>{}, number<WB>{}, number<true>{}, number<true>{});
                });
            }
        }
    }
    // Drain outstanding global copies before the output arena is reused.
    // Writeback and split-K partials.
    if constexpr (T::OUTPUT == 0)
        s_waitcnt_vmcnt(0_I);
    if constexpr (T::SPLIT_K > 1) {
        // Write FP32 partials directly to caller-owned [SplitK, M, N].
        static_for<MR>([&](auto mi) {
            constexpr int m = decltype(mi)::value;
            static_for<NR>([&](auto ni) {
                constexpr int n  = decltype(ni)::value;
                constexpr int ci = m * NR + n;
                store<4>(g_c, c[ci], u_gc.offsets[ci], 0, number<T::STORE_CACHE>{});
            });
        });
    } else {
        // Stage C in LDS and gather contiguous BF16 stores.
        constexpr int MAX_ROWS = BM;
        static_for<(MAX_ROWS + T::C_ROWS - 1) / T::C_ROWS>([&](auto parti) {
            constexpr int part      = decltype(parti)::value;
            constexpr int PART_ROWS =
                MAX_ROWS - part * T::C_ROWS < T::C_ROWS ? MAX_ROWS - part * T::C_ROWS : T::C_ROWS;
            // Store one BF16 stripe from the MFMA register tile.
            s_waitcnt_lgkmcnt(0_I);
            if constexpr (T::NUM_WAVES > 1)
                sync_threads();
            static_for<MR>([&](auto mi) {
                constexpr int m = decltype(mi)::value;
                const int rr = (m * T::T_M + wave_id_m) * 16;
                if (rr / T::C_ROWS == part && rr < MAX_ROWS)
                    static_for<NR>([&](auto ni) {
                        constexpr int n  = decltype(ni)::value;
                        constexpr int ci = m * NR + n;
                        store<4>(s_c, cast<bf16_t>(c[ci]),
                            u_sc.offsets[ci] - part * T::C_ROWS * (BN + T::C_PADDING));
                    });
            });
            s_waitcnt_lgkmcnt(0_I);
            if constexpr (T::NUM_WAVES > 1)
                sync_threads();
            // Gather contiguous BF16 vectors from the LDS stripe.
            constexpr int CHUNKS = (PART_ROWS * (BN / 8) + T::BLOCK_SIZE - 1) / T::BLOCK_SIZE;
            array<vector_t<bf16_t, 8>, CHUNKS> output;
            const auto u_sc_out = layout::make_layout_sc_vector<T, PART_ROWS>(lane_id, wave_id);
            const auto u_gc_out = layout::make_layout_gc_vector<T, PART_ROWS>(lane_id, wave_id, stride_c);
            static_for<CHUNKS>([&](auto ii) {
                constexpr int i = decltype(ii)::value;
                const int index = i * T::BLOCK_SIZE + tid;
                const int rr    = index / (BN / 8);
                if constexpr (PART_ROWS * (BN / 8) % T::BLOCK_SIZE == 0)
                    output[i] = load<8>(s_c, u_sc_out.offsets[i]);
                else if (rr < PART_ROWS)
                    output[i] = load<8>(s_c, u_sc_out.offsets[i]);
            });
            // Write only the valid M rows to global memory.
            __builtin_amdgcn_sched_barrier(0);
            s_waitcnt_lgkmcnt(0_I);
            static_for<CHUNKS>([&](auto ii) {
                constexpr int i = decltype(ii)::value;
                const int index = i * T::BLOCK_SIZE + tid;
                const int rr    = index / (BN / 8);
                if (rr < PART_ROWS)
                    if (defer_k_slice || part * T::C_ROWS + rr < valid_rows)
                        store<8>(g_c, output[i],
                                 part * T::C_ROWS * stride_c + u_gc_out.offsets[i], 0,
                                 number<T::STORE_CACHE>{});
            });
        });
    }
    if constexpr (!T::UNIFORM_SHORT_QUEUE)
        s_waitcnt_vmcnt(0_I);
}
#endif // gfx950
// A uses its LDS ring; B is loaded directly into its VGPR prefetch queue.
// Only the K draining mode is an entry parameter.
template<class Traits, bool ShortK>
__global__ __launch_bounds__(Traits::BLOCK_SIZE, Traits::WG_PER_CU)
void opus_gemm_bpreshuffle_direct_b_kernel(
    opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 args) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx950__)
    static_assert(Traits::SPLIT_K == 1 || Traits::SPLIT_K == 2 || Traits::SPLIT_K == 4);
    using T = std::conditional_t<ShortK,
        opus_gemm_bpreshuffle_direct_b_short_traits<Traits>, Traits>;
    opus_gemm_bpreshuffle_direct_b_body_impl<T, ShortK>(args);
#endif
}
