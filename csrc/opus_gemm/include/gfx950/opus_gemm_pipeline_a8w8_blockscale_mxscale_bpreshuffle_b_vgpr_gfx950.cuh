// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <opus/hip_minimal.hpp>
#include <opus/opus.hpp>
#include "../opus_gemm_utils.cuh"
#include "opus_gemm_traits_a8w8_blockscale_mxscale_bpreshuffle_b_vgpr_gfx950.cuh"
#include "opus_gemm_traits_a8w8_blockscale_mxscale_bpreshuffle_b_vgpr_overlap_gfx950.cuh"

#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx950__)
namespace opus_gemm_prefetch_b_vgpr_detail {

// Global A copies preserve the row-dependent XOR permutation used by the LDS ring.
template<class T>
__device__ inline auto make_layout_ga(int lane_id, int wave_id_n, int stride_a) {
    constexpr int threads_k = T::B_K / T::VEC_A;
    constexpr int threads_m_per_block = T::BLOCK_SIZE / threads_k;
    constexpr int threads_m_per_wave = T::WARP_SIZE / threads_k;

    constexpr auto ga_block_shape = opus::make_tuple(
        opus::number<T::B_M / threads_m_per_block>{},
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
        opus::unfold_p_coord(ga_block_dim, opus::tuple{wave_id_n, lane_id / threads_k, 0_I, lane_id % threads_k}));
}

// VMEM-to-LDS adds the lane displacement; the layout contains each wave base.
template<class T>
__device__ inline auto make_layout_sa(int wave_id_n) {
    constexpr int smem_linear_wave = T::WARP_SIZE * T::VEC_A;
    constexpr int smem_m_rep = T::B_M * T::B_K / smem_linear_wave;

    constexpr auto sa_block_shape = opus::make_tuple(
        opus::number<smem_m_rep / T::NUM_WAVES>{},
        opus::number<T::T_N>{},
        opus::number<T::T_M>{},
        opus::number<T::VEC_A>{});

    constexpr auto sa_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}));

    return opus::make_layout<T::VEC_A>(
        sa_block_shape,
        opus::unfold_x_stride(sa_block_dim, sa_block_shape, opus::tuple{opus::number<smem_linear_wave>{}, 1_I}),
        opus::unfold_p_coord(sa_block_dim, opus::tuple{wave_id_n, 0_I}));
}

// Construct only the current K128 A fragment's offsets at its consumption point.
template<class T>
__device__ inline auto make_layout_ra(int lane_id) {
    constexpr int smem_linear_wave = T::WARP_SIZE * T::VEC_A;
    constexpr int rows_per_slot = smem_linear_wave / T::B_K;
    constexpr int slots_per_m = T::W_M / rows_per_slot;
    constexpr int k_slices = T::B_K / T::W_K;

    constexpr auto ra_block_shape = opus::make_tuple(
        opus::number<T::E_M>{},
        opus::number<slots_per_m>{},
        opus::number<rows_per_slot>{},
        opus::number<k_slices>{},
        opus::number<T::A_CHUNKS_PER_FRAGMENT>{},
        opus::number<T::WARP_SIZE / T::W_M>{},
        opus::number<T::VEC_A>{});

    constexpr auto ra_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::y_dim{}, opus::p_dim{}, opus::y_dim{}));

    const int lane_id_m = lane_id % T::W_M;

    return opus::make_layout<T::VEC_A>(
        ra_block_shape,
        opus::unfold_x_stride(ra_block_dim, ra_block_shape, opus::tuple{opus::number<smem_linear_wave>{}, 1_I}),
        opus::unfold_p_coord(ra_block_dim, opus::tuple{lane_id_m / rows_per_slot, lane_id_m % rows_per_slot, 0_I, lane_id / T::W_M}));
}

// Each B fragment has two K64 halves and is loaded directly into its VGPR slot.
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
        opus::unfold_x_stride(rb_block_dim, rb_block_shape, opus::tuple{opus::number<T::W_N * T::W_K>{}, T::W_N * stride_b, opus::number<T::WARP_SIZE * T::VEC_B>{}, opus::number<1>{}}),
        opus::unfold_p_coord(rb_block_dim, opus::tuple{wave_id_n, lane_id / T::W_N, lane_id % T::W_N}));
}

// A scale vectors cover four M16 repeats per quad before the VGPR transpose.
template<class T>
__device__ inline auto make_layout_gsfa(int lane_id, int wave_id_n, int stride_sfa) {
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

    return opus::make_layout<T::VEC_SCALE_A>(
        gsfa_block_shape,
        opus::unfold_x_stride(gsfa_block_dim, gsfa_block_shape, opus::tuple{stride_sfa, 1_I}),
        opus::unfold_p_coord(gsfa_block_dim, opus::tuple{wave_id_n, 0_I, lane_id / T::SFA_M_VECTORS, lane_id % T::SFA_M_VECTORS}));
}

template<class T>
__device__ inline auto make_layout_ssfa(int lane_id, int wave_id_n) {
    constexpr auto ssfa_block_shape = opus::make_tuple(
        opus::number<T::SFA_PASSES_PER_CACHE_PANEL>{},
        opus::number<T::T_N>{},
        opus::number<T::T_M>{},
        opus::number<T::SFA_K_COLUMNS_PER_WAVE>{},
        opus::number<T::SFA_PACKED_M_GROUPS>{},
        opus::number<T::W_M>{},
        opus::number<T::VEC_SF>{});

    constexpr auto ssfa_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::y_dim{}));

    return opus::make_layout<T::VEC_SF>(
        ssfa_block_shape,
        opus::unfold_x_stride(ssfa_block_dim, ssfa_block_shape, opus::tuple{opus::number<T::SFA_PANEL_PITCH>{}, 1_I}),
        opus::unfold_p_coord(ssfa_block_dim, opus::tuple{wave_id_n, 0_I, lane_id / T::SFA_M_VECTORS,
            (lane_id % T::SFA_M_VECTORS) / T::VEC_SF, lane_id % T::VEC_SF}));
}

template<class T>
__device__ inline auto make_layout_rsfa(int lane_id) {
    constexpr auto rsfa_block_shape = opus::make_tuple(
        opus::number<T::B_K / T::GROUP_K>{},
        opus::number<T::SFA_PACKED_M_GROUPS>{},
        opus::number<T::W_M>{},
        opus::number<T::VEC_SF>{});

    constexpr auto rsfa_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}),
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::y_dim{}));

    return opus::make_layout<T::VEC_SF>(
        rsfa_block_shape,
        opus::unfold_x_stride(rsfa_block_dim, rsfa_block_shape, opus::tuple{opus::number<T::SFA_PANEL_PITCH>{}, 1_I}),
        opus::unfold_p_coord(rsfa_block_dim, opus::tuple{lane_id % T::W_M}));
}

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
    constexpr int scale_stride = T::SB_SLOTS ? T::VEC_SCALE_B : 1;
    constexpr auto rsfb_block_shape = opus::make_tuple(
        opus::number<T::B_K / T::GROUP_K>{},
        opus::number<groups>{},
        opus::number<T::SB_ROWS>{});

    constexpr auto rsfb_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}),
        opus::make_tuple(opus::y_dim{}),
        opus::make_tuple(opus::p_dim{}));

    return opus::make_layout(
        rsfb_block_shape,
        opus::unfold_x_stride(rsfb_block_dim, rsfb_block_shape, opus::tuple{opus::number<scale_stride>{}, opus::number<T::SCALE_COLS * scale_stride>{}, opus::number<T::SCALE_COLS * scale_stride>{}}),
        opus::unfold_p_coord(rsfb_block_dim, opus::tuple{wave_id_n * (T::B_N / T::T_N) / T::GROUP_N}));
}

// MFMA C fragments scatter into the padded LDS output stripe.
template<class T>
__device__ inline auto make_layout_sc(int lane_id, int wave_id_n) {
    constexpr auto sc_block_shape = opus::make_tuple(
        opus::number<T::MR>{},
        opus::number<T::W_M>{},
        opus::number<T::T_N>{},
        opus::number<T::NR>{},
        opus::number<T::WARP_SIZE / T::W_M>{},
        opus::number<T::VEC_C>{});

    constexpr auto sc_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}));

    return opus::make_layout<T::VEC_C>(
        sc_block_shape,
        opus::unfold_x_stride(sc_block_dim, sc_block_shape, opus::tuple{opus::number<T::B_N + T::C_PADDING>{}, opus::number<T::W_N>{}, opus::number<1>{}}),
        opus::unfold_p_coord(sc_block_dim, opus::tuple{lane_id % T::W_M, wave_id_n, lane_id / T::W_M}));
}

template<class T, int Rows>
__device__ inline auto make_layout_sc_copy(int lane_id, int wave_id_n) {
    constexpr int copies = (Rows * (T::B_N / T::VEC_OUTPUT) + T::BLOCK_SIZE - 1) / T::BLOCK_SIZE;
    constexpr auto sc_copy_block_shape = opus::make_tuple(
        opus::number<copies>{},
        opus::number<T::NUM_WAVES>{},
        opus::number<T::WARP_SIZE>{},
        opus::number<T::VEC_OUTPUT>{});

    constexpr auto sc_copy_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}, opus::y_dim{}));

    auto u = opus::make_layout<T::VEC_OUTPUT>(
        sc_copy_block_shape,
        opus::unfold_x_stride(sc_copy_block_dim, sc_copy_block_shape, opus::tuple{opus::number<1>{}}),
        opus::unfold_p_coord(sc_copy_block_dim, opus::tuple{wave_id_n, lane_id}));
    opus::static_for<copies>([&](auto i) {
        const int index = decltype(i)::value * T::BLOCK_SIZE + wave_id_n * T::WARP_SIZE + lane_id;
        u.offsets[i] = (index / (T::B_N / T::VEC_OUTPUT)) * (T::B_N + T::C_PADDING) + (index % (T::B_N / T::VEC_OUTPUT)) * T::VEC_OUTPUT;
    });
    return u;
}

template<class T, int Rows>
__device__ inline auto make_layout_gc_copy(int lane_id, int wave_id_n, int stride_c) {
    constexpr int copies = (Rows * (T::B_N / T::VEC_OUTPUT) + T::BLOCK_SIZE - 1) / T::BLOCK_SIZE;
    constexpr auto gc_copy_block_shape = opus::make_tuple(
        opus::number<copies>{},
        opus::number<T::NUM_WAVES>{},
        opus::number<T::WARP_SIZE>{},
        opus::number<T::VEC_OUTPUT>{});

    constexpr auto gc_copy_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}, opus::y_dim{}));

    auto u = opus::make_layout<T::VEC_OUTPUT>(
        gc_copy_block_shape,
        opus::unfold_x_stride(gc_copy_block_dim, gc_copy_block_shape, opus::tuple{opus::number<1>{}}),
        opus::unfold_p_coord(gc_copy_block_dim, opus::tuple{wave_id_n, lane_id}));
    opus::static_for<copies>([&](auto i) {
        const int index = decltype(i)::value * T::BLOCK_SIZE + wave_id_n * T::WARP_SIZE + lane_id;
        u.offsets[i] = (index / (T::B_N / T::VEC_OUTPUT)) * stride_c + (index % (T::B_N / T::VEC_OUTPUT)) * T::VEC_OUTPUT;
    });
    return u;
}

template<class T, bool ContiguousWaveN>
__device__ inline auto make_layout_gc_packed(int lane_id, int wave_id_m, int wave_id_n, int stride_c) {
    constexpr int lanes_per_packed_group = (T::VEC_OUTPUT / T::VEC_C) * T::W_M;
    constexpr auto gc_packed_block_shape = opus::make_tuple(
        opus::number<T::E_M>{},
        opus::number<T::T_M>{},
        opus::number<T::W_M>{},
        opus::number<T::E_N>{},
        opus::number<T::T_N>{},
        opus::number<T::WARP_SIZE / lanes_per_packed_group>{},
        opus::number<T::VEC_OUTPUT>{});

    constexpr auto gc_packed_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}),
        opus::make_tuple(opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}));

    return opus::make_layout<T::VEC_OUTPUT>(
        gc_packed_block_shape,
        opus::unfold_x_stride(gc_packed_block_dim, gc_packed_block_shape, opus::tuple{stride_c, opus::number<(ContiguousWaveN ? 1 : T::T_N) * T::W_N>{}, opus::number<(ContiguousWaveN ? T::E_N : 1) * T::W_N>{}, opus::number<1>{}}),
        opus::unfold_p_coord(gc_packed_block_dim, opus::tuple{wave_id_m, lane_id % T::W_M, wave_id_n, lane_id / lanes_per_packed_group}));
}

// Read an A fragment directly from LDS into its assigned AGPRs.
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

// Pack adjacent MFMA output fragments into contiguous BF16 stores.
// Use lane_id directly; opus::shfl adds a lane-ID query here.
template<class T, bool ContiguousWaveN = false, class Accumulators, class Gmem>
__device__ __forceinline__ void store_mfma16_packed(
    const Accumulators& c, Gmem g_c, int lane_id, int wave_id_m, int wave_id_n, int stride_c) {
    using namespace opus;
    static_assert(T::W_M == 16 && T::W_N == 16);
    const int side = (lane_id / 16) % 2;
    const auto u_gc = make_layout_gc_packed<T, ContiguousWaveN>(lane_id, wave_id_m, wave_id_n, stride_c);
    static_for<T::E_M * T::E_N / 2>([&](auto pair) {
        constexpr int ci = decltype(pair)::value * 2;
        const auto first = __builtin_bit_cast(vector_t<unsigned, 2>, cast<bf16_t>(c[ci]));
        const auto second = __builtin_bit_cast(vector_t<unsigned, 2>, cast<bf16_t>(c[ci + 1]));
        const unsigned own0 = side ? second[0] : first[0];
        const unsigned own1 = side ? second[1] : first[1];
        const unsigned src0 = side ? first[0] : second[0];
        const unsigned src1 = side ? first[1] : second[1];
        const unsigned other0 = __builtin_amdgcn_ds_bpermute((lane_id ^ 16) * 4, src0);
        const unsigned other1 = __builtin_amdgcn_ds_bpermute((lane_id ^ 16) * 4, src1);
        vector_t<unsigned, 4> packed;
        packed[0] = side ? other0 : own0;
        packed[1] = side ? other1 : own1;
        packed[2] = side ? own0 : other0;
        packed[3] = side ? own1 : other1;
        store<8>(g_c, __builtin_bit_cast(vector_t<bf16_t, 8>, packed),
            u_gc.offsets[ci + side], 0, number<T::STORE_CACHE>{});
    });
    if constexpr (T::E_M * T::E_N % 2 != 0) {
        constexpr int ci = T::E_M * T::E_N - 1;
        const auto words = __builtin_bit_cast(vector_t<unsigned, 2>, cast<bf16_t>(c[ci]));
        const unsigned other0 = __builtin_amdgcn_ds_bpermute((lane_id ^ 16) * 4, words[0]);
        const unsigned other1 = __builtin_amdgcn_ds_bpermute((lane_id ^ 16) * 4, words[1]);
        if (side == 0) {
            vector_t<unsigned, 4> packed{words[0], words[1], other0, other1};
            store<8>(g_c, __builtin_bit_cast(vector_t<bf16_t, 8>, packed),
                u_gc.offsets[ci], 0, number<T::STORE_CACHE>{});
        }
    }
}

// Choose an output stripe that fits the LDS allocation reused by the epilogue.
__device__ inline constexpr int output_rows(int bm, int bn, int pipeline_bytes) {
    return opus_gemm_blockscale_mxscale_bpreshuffle_output_rows(bm, bn, pipeline_bytes);
}

// Fixed-shape entries use contiguous matrix and scale strides.
#if defined(OPUS_BP_FIXED_M) && defined(OPUS_BP_FIXED_N) && defined(OPUS_BP_FIXED_K)
__device__ __forceinline__ void apply_fixed_args(opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950& a) {
    a.m = OPUS_BP_FIXED_M;
    a.n = OPUS_BP_FIXED_N;
    a.k = OPUS_BP_FIXED_K;
    a.batch = 1;
    a.stride_a = OPUS_BP_FIXED_K;
    a.stride_b = OPUS_BP_FIXED_K;
    a.stride_c = OPUS_BP_FIXED_N;
    a.stride_a_batch = static_cast<int>(int64_t(OPUS_BP_FIXED_M) * OPUS_BP_FIXED_K);
    a.stride_b_batch = static_cast<int>(int64_t(OPUS_BP_FIXED_N) * OPUS_BP_FIXED_K);
    a.stride_c_batch = static_cast<int>(int64_t(OPUS_BP_FIXED_M) * OPUS_BP_FIXED_N);
    a.stride_sfa = OPUS_BP_FIXED_M;
    a.stride_sfb = OPUS_BP_FIXED_K / 128;
    a.stride_sfa_batch = OPUS_BP_FIXED_M * (OPUS_BP_FIXED_K / 128);
    a.stride_sfb_batch = (OPUS_BP_FIXED_N / 128) * (OPUS_BP_FIXED_K / 128);
}
#endif

// Stream individual B fragments across N, reusing each fragment for all M repeats.
template<class Traits, int PublishM = -1>
__device__ __forceinline__ void fragment_prefetch_body(opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 args) {
#if defined(OPUS_BP_FIXED_M) && defined(OPUS_BP_FIXED_N) && defined(OPUS_BP_FIXED_K)
    opus_gemm_prefetch_b_vgpr_detail::apply_fixed_args(args);
#endif
    using namespace opus;

    using T = opus::remove_cvref_t<Traits>;
    using Index = unsigned;
    constexpr int WaveColumns = T::T_N;

    // Tile geometry and runtime strides.
    constexpr int BM = T::B_M;
    constexpr int BN = T::B_N;
    constexpr int NB = T::STAGES;
    constexpr int BD = T::B_AHEAD;
    constexpr int BS = T::B_SETS;
    constexpr int P = T::PANELS;
    constexpr int MR = T::MR;
    constexpr int NR = T::NR;
    constexpr int K = T::K;
    constexpr int SC = T::SCALE_COLS;
    constexpr int BK = T::B_K;
    constexpr int KS = BK / T::W_K;
    constexpr bool ROTATE_RING = T::ROTATE_B && !T::STATIC_B_RING;
    constexpr bool STREAM_A = T::ROTATE_B || T::STREAM_A;

    const int M = args.m;
    const int N = args.n;
    const int stride_a = T::CONTIGUOUS ? K : args.stride_a;
    const int stride_b = T::CONTIGUOUS ? K : args.stride_b;
    const int stride_c = args.stride_c;

    static_assert((BK == 128 || BK == 256) && SC == K / 128);
    static_assert(T::T_M == 1 && T::T_N * T::WARP_SIZE == T::BLOCK_SIZE);
    static_assert(T::MR * T::W_M == T::B_M && T::NR * T::T_N * T::W_N == T::B_N);

    // Tile and thread coordinates.
    const Index thread_id = thread_id_x();
    const Index lane_id = thread_id % T::WARP_SIZE;
    const Index wave_id = __builtin_amdgcn_readfirstlane(thread_id / T::WARP_SIZE);
    const Index wave_id_n = wave_id;
    const unsigned pid = block_id_x();
    const unsigned gm = (M + BM - 1) / BM;
    const unsigned gn = N / BN;
    const unsigned total = gm * gn;
    unsigned tile = T::XCD_ORDER ? pid % 8 * ((total + 7) / 8) + pid / 8 : pid;
    if constexpr (T::XCD_ORDER)
        if (tile >= total)
            return;
    if constexpr (T::GROUPED_M > 0) {
        const unsigned first = tile / (T::GROUPED_M * gn) * T::GROUPED_M;
        const unsigned rows = gm - first < T::GROUPED_M ? gm - first : T::GROUPED_M;
        const unsigned local = tile % (T::GROUPED_M * gn);
        tile = (first + local % rows) * gn + local / rows;
    }
    const int row = M <= BM ? 0 : (tile / gn) * BM;
    const int col = M <= BM ? tile * BN : (tile % gn) * BN;
    const int valid_rows = M % BM == 0 ? BM : (M <= BM ? M : (row == (gm - 1) * BM ? M % BM : BM));

    constexpr int F = KS * NR;
    constexpr int MAX_ROWS = BM;
    constexpr int C_ROWS = output_rows(BM, BN, T::PIPELINE_BYTES);
    constexpr int STEADY = P - T::INITIAL_A;
    constexpr int START = T::INITIAL_A < STEADY ? T::INITIAL_A : STEADY;
    constexpr int PERIOD = []() constexpr {
        if constexpr (ROTATE_RING) return 1;
        if constexpr (T::COMPACT_B_RING) {
            int a = BS, b = F;
            while (b) { int r = a % b; a = b; b = r; }
            const int b_panels = BS / a;
            a = NB; b = b_panels;
            while (b) { int r = a % b; a = b; b = r; }
            return NB / a * b_panels;
        } else {
            int a = NB, b = BS;
            while (b) { int r = a % b; a = b; b = r; }
            return NB / a * BS;
        }
    }();
    constexpr int GROUPS = (STEADY - START) / PERIOD;
    constexpr int END = START + GROUPS * PERIOD;

    // Matrix and scale global-memory views, bounded for the final partial M tile.
    auto g_a = make_gmem(reinterpret_cast<const fp8_t*>(args.ptr_a) + row * stride_a,
        static_cast<unsigned>(valid_rows * stride_a));
    auto g_b = make_gmem(reinterpret_cast<const fp8_t*>(args.ptr_b) + col * stride_b,
        static_cast<unsigned>(BN * stride_b));
    auto g_sfa = make_gmem(reinterpret_cast<const unsigned char*>(args.ptr_sfa),
        static_cast<unsigned>((SC - 1) * args.stride_sfa + M));
    auto g_sfb = make_gmem(reinterpret_cast<const unsigned char*>(args.ptr_sfb),
        static_cast<unsigned>((N / 128 - 1) * args.stride_sfb + SC));
    auto g_c = make_gmem(reinterpret_cast<bf16_t*>(args.ptr_c) + static_cast<int64_t>(row) * stride_c + col,
        static_cast<unsigned>(((valid_rows - 1) * stride_c + BN) * 2));

    // Matrix, scale, and output layouts.
    constexpr int a_threads_k = T::B_K / T::VEC_A;
    constexpr int a_rows_per_wave = T::WARP_SIZE / a_threads_k;
    auto u_ga = make_layout_ga<T>(lane_id, wave_id_n, stride_a);
    opus::static_for<T::A_COPIES>([&](auto i) {
        const int row = (decltype(i)::value * T::NUM_WAVES + wave_id_n) * a_rows_per_wave + lane_id / a_threads_k;
        const int k = (lane_id % a_threads_k) * T::VEC_A;
        u_ga.offsets[i] += (k ^ ((row % a_threads_k) * T::VEC_A)) - k;
    });
    const auto u_sa = make_layout_sa<T>(wave_id_n);
    const auto u_rb = make_layout_rb<T>(lane_id, wave_id_n, stride_b);
    const auto u_gsfa = make_layout_gsfa<T>(lane_id, wave_id_n, args.stride_sfa);
    const auto u_ssfa = make_layout_ssfa<T>(lane_id, wave_id_n);
    const auto u_rsfa = make_layout_rsfa<T>(lane_id);
    const auto u_gsfb = make_layout_gsfb<T>(args.stride_sfb);
    constexpr auto u_ssfb = make_layout_ssfb<T>();
    const auto u_rsfb = make_layout_rsfb<T>(wave_id_n);
    const auto u_sc = make_layout_sc<T>(lane_id, wave_id_n);
    const auto u_sc_copy = make_layout_sc_copy<T, C_ROWS>(lane_id, wave_id_n);
    const auto u_gc_copy = make_layout_gc_copy<T, C_ROWS>(lane_id, wave_id_n, stride_c);

    // A and scale LDS; the allocation is reused for the BF16 epilogue.
    alignas(16) __shared__ unsigned char arena[T::LDS_BYTES];
    auto s_a = make_smem(reinterpret_cast<fp8_t*>(arena));
    auto s_scale = make_smem(arena);
    auto s_c = make_smem(reinterpret_cast<bf16_t*>(arena));

    // MMA accumulators and the B register ring.
    using Base = decltype(make_mfma<fp8_t, fp8_t, fp32_t>(16_I, 16_I, 128_I, mfma_adaptor_swap_ab{}));
    array<typename Base::vtype_c, MR * NR> r_c{};
    array<typename Base::vtype_b, BS> r_b;

    auto issue_b = [&](auto slot_i, auto fragment_i, int kt) {
        constexpr int set = decltype(slot_i)::value;
        constexpr int fragment = decltype(fragment_i)::value;
        set_slice(r_b[set],
            load<T::VEC_B>(g_b, u_rb.offsets[fragment * T::B_CHUNKS_PER_FRAGMENT],
                kt * (BK * T::W_N), number<T::B_CACHE>{}),
            0_I, number<T::VEC_B>{});
        set_slice(r_b[set],
            load<T::VEC_B>(g_b, u_rb.offsets[fragment * T::B_CHUNKS_PER_FRAGMENT + 1],
                kt * (BK * T::W_N), number<T::B_CACHE>{}),
            number<T::VEC_B>{}, number<T::B_CHUNKS_PER_FRAGMENT * T::VEC_B>{});
    };
    auto issue_a = [&](int slot, int kt) {
        __builtin_amdgcn_sched_barrier(0);
        static_for<T::A_COPIES>([&](auto ii) {
            constexpr int i = decltype(ii)::value;
            async_load<16>(g_a, arena + slot * T::STAGE_BYTES + u_sa.offsets[i],
                u_ga.offsets[i], kt * BK, 0_I, number<T::CACHECTL_A>{});
        });
        __builtin_amdgcn_sched_barrier(0);
    };

    vector_t<unsigned char, T::VEC_SCALE_A> sfa_panel_raw[T::SFA_PASSES_PER_CACHE_PANEL];
    auto issue_scale_panels = [&]() {
        static_for<T::SFA_PASSES_PER_CACHE_PANEL>([&](auto pass_i) {
            constexpr int pass = decltype(pass_i)::value;
            const int lane_k = lane_id / T::SFA_M_VECTORS;
            const int scale_col = pass * T::SFA_K_COLUMNS_PER_PASS
                + wave_id_n * T::SFA_K_COLUMNS_PER_WAVE + lane_k;
            const int scale_row = (lane_id % T::SFA_M_VECTORS) * T::VEC_SCALE_A;
            sfa_panel_raw[pass] = {};
            if (lane_k < T::SFA_K_COLUMNS_PER_WAVE && scale_col < SC
                && scale_row < BM && row + scale_row < M)
                sfa_panel_raw[pass] = load<T::VEC_SCALE_A>(g_sfa, u_gsfa.offsets[pass] + row);
        });
        auto* sbdst = arena + T::OFF_SB + u_ssfb(wave_id, 0, 0);
        if constexpr (T::SB_SLOTS) {
            const int index = thread_id % (T::SB_ROWS * SC);
            const int sr = index / SC;
            const int scale_col = index % SC;
            async_load<1>(g_sfb, sbdst, u_gsfb(col / 128 + sr, scale_col));
        } else {
            const int index = thread_id % (T::SB_ROWS * (SC / 4));
            const int sr = index / (SC / 4);
            const int word = index % (SC / 4);
            async_load<4>(g_sfb, sbdst, u_gsfb(col / 128 + sr, word * 4));
        }
    };

    // The same 4x4 byte transpose as 4wave; TM=1 packs M,M+16,M+32,M+48.
    auto publish_sfa_panel = [&]() {
        const int lane_in_quad = lane_id % T::VEC_SF;
        const unsigned pair_byte_select = lane_in_quad % 2 ? 0x03070105u : 0x06020400u;
        const unsigned quad_byte_select = lane_in_quad / 2 ? 0x03020706u : 0x05040100u;
        auto transpose_sfa_word_4x4 = [&](unsigned lane_word) {
            const unsigned adjacent_lane = opus::mov_dpp(lane_word, opus::number<0xb1>{});
            const unsigned paired_bytes = __builtin_amdgcn_perm(adjacent_lane, lane_word, pair_byte_select);
            const unsigned opposite_pair = opus::mov_dpp(paired_bytes, opus::number<0x4e>{});
            return __builtin_amdgcn_perm(opposite_pair, paired_bytes, quad_byte_select);
        };
        static_for<T::SFA_PASSES_PER_CACHE_PANEL>([&](auto pass_i) {
            constexpr int pass = decltype(pass_i)::value;
            const int lane_k = lane_id / T::SFA_M_VECTORS;
            const int scale_col = pass * T::SFA_K_COLUMNS_PER_PASS
                + wave_id_n * T::SFA_K_COLUMNS_PER_WAVE + lane_k;
            if (lane_k < T::SFA_K_COLUMNS_PER_WAVE && scale_col < SC) {
                const auto raw_words = __builtin_bit_cast(vector_t<unsigned, T::VEC_SCALE_A / T::VEC_SF>, sfa_panel_raw[pass]);
                static_for<T::VEC_SCALE_A / T::VEC_SF>([&](auto word_i) {
                    constexpr int word = decltype(word_i)::value;
                    const unsigned packed_scales = transpose_sfa_word_4x4(raw_words[word]);
                    store<T::VEC_SF>(s_scale, __builtin_bit_cast(vector_t<unsigned char, T::VEC_SF>, packed_scales),
                        T::OFF_SA + u_ssfa.offsets[pass] + word * T::VEC_SF * T::VEC_SF);
                });
            }
        });
    };

    // Prepare the panel consumer before issuing the prologue requests.
    auto consume_panel = [&](int t, auto phase_i) {
        constexpr int phase = decltype(phase_i)::value;
        constexpr bool future_a = phase + T::INITIAL_A < P;
        constexpr int issued_b = (phase * F + BD < P * F ? phase * F + BD : P * F);
        constexpr int before_a = BD + (phase - T::INITIAL_A) * F + 1;
        constexpr int newer_b_a = phase < T::INITIAL_A ? issued_b : (issued_b > before_a ? issued_b - before_a : 0);
        constexpr int newer_a = phase < T::INITIAL_A ? T::INITIAL_A - phase - 1 : (P - phase - 1 < T::INITIAL_A - 1 ? P - phase - 1 : T::INITIAL_A - 1);
        constexpr int wa = newer_b_a * 2 + newer_a * T::A_COPIES;
        const int slot = ROTATE_RING ? t % NB : phase % NB;
        __builtin_amdgcn_sched_barrier(0);
        s_waitcnt_vmcnt(number<(wa > 63 ? 63 : wa)>{});
        s_waitcnt_lgkmcnt(0_I);
        opus::sync_threads();

        array<array<typename Base::vtype_a, MR>, STREAM_A ? 1 : KS> r_a;
        array<array<unsigned, T::SFA_PACKED_M_GROUPS>, KS> r_sfa;
        array<unsigned, KS> r_sfb{};
        static_for<KS>([&](auto ki) {
            constexpr int ks = decltype(ki)::value;
            static_for<(BN / WaveColumns + 127) / 128>([&](auto group_i) {
                constexpr int group = decltype(group_i)::value;
                const unsigned scale = load<1>(s_scale,
                    T::OFF_SB + u_rsfb(ks, group) + t * KS * (T::SB_SLOTS ? 4 : 1))[0];
                r_sfb[ks] |= scale << (group * 8);
            });
            static_for<T::SFA_PACKED_M_GROUPS>([&](auto pack_i) {
                constexpr int pack = decltype(pack_i)::value;
                r_sfa[ks][pack] = __builtin_bit_cast(unsigned, load<T::VEC_SF>(s_scale,
                    T::OFF_SA + u_rsfa.offsets[ks * T::SFA_PACKED_M_GROUPS + pack]
                        + t * KS * T::SFA_PANEL_PITCH));
            });
        });
        auto read_a = [&](auto ki) {
            constexpr int ks = decltype(ki)::value;
            auto u_ra = make_layout_ra<T>(lane_id);
            u_ra += ks * T::W_K;
            opus::static_for<T::MR * T::A_CHUNKS_PER_FRAGMENT>([&](auto i) {
                constexpr int m = decltype(i)::value / T::A_CHUNKS_PER_FRAGMENT;
                constexpr int half = decltype(i)::value % T::A_CHUNKS_PER_FRAGMENT;
                const int row = m * T::W_M + lane_id % T::W_M;
                const int k_vector = ks * (T::W_K / T::VEC_A) + lane_id / T::W_M + half * (T::WARP_SIZE / T::W_M);
                u_ra.offsets[i] += ((k_vector ^ (row % (T::B_K / T::VEC_A))) - k_vector) * T::VEC_A;
            });
            static_for<MR>([&](auto mi) {
                constexpr int m = decltype(mi)::value;
                const int lo = slot * T::STAGE_BYTES + u_ra.offsets[m * 2];
                const int hi = slot * T::STAGE_BYTES + u_ra.offsets[m * 2 + 1];
                if constexpr (T::PIN_A) {
                    array<int, 2> offsets;
                    offsets[0] = lo;
                    offsets[1] = hi;
                    load_operand_fragment_pinned<16, 0>(number<(ks * MR + m) * 8>{}, r_a[STREAM_A ? 0 : ks][m], s_a, offsets);
                } else {
                    set_slice(r_a[STREAM_A ? 0 : ks][m], load<16>(s_a, lo), 0_I, 16_I);
                    set_slice(r_a[STREAM_A ? 0 : ks][m], load<16>(s_a, hi), 16_I, 32_I);
                }
            });
        };
        if constexpr (!STREAM_A)
            static_for<KS>([&](auto ki) { read_a(ki); });
        static_for<KS * NR>([&](auto fragment_i) {
            constexpr int fragment = decltype(fragment_i)::value;
            constexpr int ks = fragment / NR;
            constexpr int n = fragment % NR;
            constexpr int absolute = phase * F + fragment;
            if constexpr (STREAM_A && n == 0)
                read_a(number<ks>{});
            if constexpr (absolute + BD < P * F)
                issue_b(number<((ROTATE_RING ? fragment : absolute) + BD) % BS>{}, number<(fragment + BD) % F>{}, t + (fragment + BD) / F);
            constexpr int newer_b = P * F - absolute - 1 < BD ? P * F - absolute - 1 : BD;
            constexpr int first_a = absolute < BD ? 0 : (absolute - BD + F - 1) / F;
            constexpr int last_a = (phase - (fragment == 0 ? 1 : 0) < P - T::INITIAL_A - 1)
                ? phase - (fragment == 0 ? 1 : 0) : P - T::INITIAL_A - 1;
            constexpr int pending_a = last_a >= first_a ? (last_a - first_a + 1) * T::A_COPIES : 0;
            constexpr int wait = newer_b * 2 + pending_a;
            __builtin_amdgcn_sched_barrier(0);
            s_waitcnt_vmcnt(number<(wait > 63 ? 63 : wait)>{});
            if constexpr (fragment == 0 || (STREAM_A && n == 0))
                s_waitcnt_lgkmcnt(0_I);
            if constexpr (future_a && fragment == 0 && PublishM < 0)
                issue_a((ROTATE_RING ? t + NB - 1 : phase + NB - 1) % NB, t + NB - 1);

            static_for<MR>([&](auto mi) {
                constexpr int m = decltype(mi)::value;
                constexpr int ci = m * NR + n;
                if constexpr (T::PIN_C) {
                    [[clang::amdgpu_pin_agpr(128 + ci * 4)]] r_c[ci] = r_c[ci];
                    [[clang::amdgpu_pin_agpr(128 + ci * 4)]]
                    r_c[ci] = Base{}(r_a[STREAM_A ? 0 : ks][m], r_b[(ROTATE_RING ? fragment : phase * F + fragment) % BS], r_c[ci],
                        static_cast<int>(r_sfa[ks][m / T::VEC_SF]), static_cast<int>(r_sfb[ks]), number<m % T::VEC_SF>{}, number<n / 8>{});
                } else {
                    r_c[ci] = Base{}(r_a[STREAM_A ? 0 : ks][m], r_b[(ROTATE_RING ? fragment : phase * F + fragment) % BS], r_c[ci],
                        static_cast<int>(r_sfa[ks][m / T::VEC_SF]), static_cast<int>(r_sfb[ks]), number<m % T::VEC_SF>{}, number<n / 8>{});
                }
                if constexpr (future_a && fragment == 0 && m == PublishM)
                    issue_a((ROTATE_RING ? t + NB - 1 : phase + NB - 1) % NB, t + NB - 1);
            });
        });
        if constexpr (ROTATE_RING) {
            // Advance the bounded queue without expanding its ring period.
            auto next_b = r_b;
            static_for<BS>([&](auto i) {
                constexpr int slot_b = decltype(i)::value;
                next_b[slot_b] = r_b[(slot_b + F) % BS];
            });
            r_b = next_b;
        }
    };
    // ===== Prologue =====
    // Issue initial scale/matrix loads and consume startup panels.
    issue_scale_panels();
    static_for<T::INITIAL_A>([&](auto i) { issue_a(decltype(i)::value, decltype(i)::value); });
    static_for<BD>([&](auto i) {
        constexpr int fragment = decltype(i)::value;
        issue_b(number<fragment % BS>{}, number<fragment % F>{}, fragment / F);
    });
    publish_sfa_panel();
    static_for<START>([&](auto i) {
        constexpr int t = decltype(i)::value;
        consume_panel(t, number<t>{});
    });
    // ===== Main loop =====
    // Consume the steady A-stage/B-queue period.
    if constexpr (GROUPS > 0) {
#pragma nounroll
        for (int group = 0; group < GROUPS; ++group) {
            static_for<PERIOD>([&](auto i) {
                constexpr int phase = START + decltype(i)::value;
                consume_panel(phase + group * PERIOD, number<phase>{});
            });
        }
    }
    // ===== Epilogue =====
    // Drain the remaining panels before output writeback.
    static_for<P - END>([&](auto i) {
        constexpr int t = END + decltype(i)::value;
        consume_panel(t, number<t>{});
    });

    // ===== Output writeback =====
    // Pack BF16 directly or reuse LDS for contiguous stripes.
    if constexpr (T::OUTPUT == 2) {
        opus_gemm_prefetch_b_vgpr_detail::store_mfma16_packed<T, true>(r_c, g_c, lane_id, 0, wave_id_n, stride_c);
    } else {
        static_for<(MAX_ROWS + C_ROWS - 1) / C_ROWS>([&](auto parti) {
            constexpr int part = decltype(parti)::value;
            constexpr int PART_ROWS = MAX_ROWS - part * C_ROWS < C_ROWS ? MAX_ROWS - part * C_ROWS : C_ROWS;
            s_waitcnt_lgkmcnt(0_I);
            opus::sync_threads();
            static_for<MR>([&](auto mi) {
                constexpr int m = decltype(mi)::value;
                if constexpr (m * 16 / C_ROWS == part && m * 16 < MAX_ROWS)
                    static_for<NR>([&](auto ni) {
                        constexpr int n = decltype(ni)::value;
                        constexpr int ci = m * NR + n;
                        store<4>(s_c, cast<bf16_t>(r_c[ci]),
                            u_sc.offsets[ci] - part * C_ROWS * (BN + 8));
                    });
            });
            s_waitcnt_lgkmcnt(0_I);
            opus::sync_threads();
            constexpr int CHUNKS = (PART_ROWS * (BN / 8) + T::BLOCK_SIZE - 1) / T::BLOCK_SIZE;
            array<vector_t<bf16_t, 8>, CHUNKS> output;
            static_for<CHUNKS>([&](auto ii) {
                constexpr int i = decltype(ii)::value;
                const int index = i * T::BLOCK_SIZE + thread_id;
                const int rr = index / (BN / 8);
                if constexpr (PART_ROWS * (BN / 8) % T::BLOCK_SIZE == 0)
                    output[i] = load<8>(s_c, u_sc_copy.offsets[i]);
                else if (rr < PART_ROWS)
                    output[i] = load<8>(s_c, u_sc_copy.offsets[i]);
            });
            __builtin_amdgcn_sched_barrier(0);
            s_waitcnt_lgkmcnt(0_I);
            static_for<CHUNKS>([&](auto ii) {
                constexpr int i = decltype(ii)::value;
                const int index = i * T::BLOCK_SIZE + thread_id;
                const int rr = index / (BN / 8);
                if (rr < PART_ROWS && part * C_ROWS + rr < valid_rows)
                    store<8>(g_c, output[i], part * C_ROWS * stride_c + u_gc_copy.offsets[i], 0, number<T::STORE_CACHE>{});
            });
        });
    }
    s_waitcnt_vmcnt(0_I);
}

template<class Traits>
__device__ __forceinline__ void bulk_panel_overlap_body(opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 args) {
    fragment_prefetch_body<Traits, Traits::PUBLISH_M>(args);
}

} // namespace opus_gemm_prefetch_b_vgpr_detail

namespace opus_gemm_prefetch_b_vgpr_detail {
template<class Traits, int PublishM = -1>
__device__ __forceinline__ void bulk_panel_body(
    opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 args);
} // namespace opus_gemm_prefetch_b_vgpr_detail

namespace opus_gemm_b_direct_detail {
using opus::operator""_I;

// The ring pipelines count DMA groups, rather than individual VMEM operations.
// Keep the grouped instruction behind the same view/layout API as opus::gmem;
// the ordinary gmem transport is not tracked by asyncmark on gfx950.
template<class D>
struct async_gmem : opus::gmem<D> {
    using Base = opus::gmem<D>;
    using Base::Base;

    template<opus::index_t Vec, opus::index_t Immediate = 0, opus::index_t Cache = 0>
    __device__ __forceinline__ void _async_load(OPUS_LDS_ADDR void* destination, int offset,
                                                int scalar_offset = 0, opus::number<Immediate> = {},
                                                opus::number<Cache> = {}) {
        using Value = typename Base::template vector_type<Vec>;
        __builtin_amdgcn_raw_ptr_buffer_load_async_lds(this->cached_rsrc, destination,
                                                       sizeof(Value), offset * sizeof(D),
                                                       scalar_offset * sizeof(D), Immediate, Cache);
    }

    template<opus::index_t Vec, opus::index_t Immediate = 0, opus::index_t Cache = 0>
    __device__ __forceinline__ void async_load(void* destination, int offset, int scalar_offset = 0,
                                               opus::number<Immediate> = {},
                                               opus::number<Cache> = {}) {
        auto* target =
            reinterpret_cast<OPUS_LDS_ADDR void*>(reinterpret_cast<__UINTPTR_TYPE__>(destination));
        _async_load<Vec>(target, offset, scalar_offset, opus::number<Immediate>{},
                         opus::number<Cache>{});
    }

    template<opus::index_t Vec, class GlobalLayout, class SharedLayout, opus::index_t Immediate = 0,
             opus::index_t Cache = 0,
             std::enable_if_t<opus::is_layout_v<GlobalLayout> && opus::is_layout_v<SharedLayout>,
                              bool> = true>
    __device__ __forceinline__ void
    async_load(OPUS_LDS_ADDR void* destination, const GlobalLayout& global_layout,
               const SharedLayout& shared_layout, int scalar_offset = 0,
               opus::number<Immediate> = {}, opus::number<Cache> = {}) {
        using Load = opus::layout_load_traits<GlobalLayout, Vec>;
        const auto global_offsets = opus::layout_to_offsets<Vec>(global_layout);
        const auto shared_offsets = opus::layout_to_offsets<Vec>(shared_layout);
        // Preserve the OPUS shared view's address space through layout offsets.
        // A generic-pointer round trip adds a null-address select at each DMA.
        auto* target = reinterpret_cast<OPUS_LDS_ADDR D*>(destination);
        opus::static_for<Load::r_elem.value>([&](auto i) {
            _async_load<Vec>(target + shared_offsets[i], global_offsets[i], scalar_offset,
                             opus::number<Immediate>{}, opus::number<Cache>{});
        });
    }
};

template<class D>
__device__ __forceinline__ auto make_async_gmem(const D* ptr, unsigned int bytes) {
    return async_gmem<D>(ptr, bytes);
}

__device__ __forceinline__ void commit_async_loads() { __builtin_amdgcn_asyncmark(); }

// Some scale-address policies require a VGPR offset even when all active rows
// happen to share a scalar address. Preserve that requirement at the transport.
template<class Layout> __device__ __forceinline__ auto vectorize_layout_offsets(Layout layout) {
#if defined(__AMDGCN__)
    for (int i = 0; i < Layout::num_issues; ++i) {
        asm volatile("" : "+v"(layout.offsets[i]));
    }
#endif
    return layout;
}

// Assemble the two MFMA chunks directly, as in the pinned-AGPR pipeline.
// Static chunk ownership avoids a register-array index in the LDS load path.
template<class Fragment, class Memory, class Layout>
__device__ __forceinline__ Fragment load_operand_fragment(Memory& memory, const Layout& layout) {
    using Load = opus::layout_load_traits<Layout, 16>;
    static_assert(Load::r_elem.value == 2);
    const auto offsets = opus::layout_to_offsets<16>(layout);
    using Chunk = decltype(opus::load<16>(memory, offsets[0]));
    opus::array<Chunk, 2> values;
    opus::static_for<2>([&](auto i) { values[i] = opus::load<16>(memory, offsets[i]); });
    Fragment result;
    opus::set_slice(result, values[0], 0_I, 16_I);
    opus::set_slice(result, values[1], 16_I, 32_I);
    return result;
}

template<class T, bool ContiguousWaveN = false, class Accumulators, class Gmem>
__device__ __forceinline__ void store_mfma16_packed(const Accumulators& c, Gmem gc, int lane,
                                                    int wm, int wn, int stride_c) {
    using namespace opus;
    static_assert(T::W_M == 16 && T::W_N == 16);
    const int side = (lane / 16) % 2;
    static_for<T::E_M * T::E_N / 2>([&](auto pair) {
        constexpr int ci = decltype(pair)::value * 2;
        const auto first = __builtin_bit_cast(vector_t<unsigned, 2>, cast<bf16_t>(c[ci]));
        const auto second = __builtin_bit_cast(vector_t<unsigned, 2>, cast<bf16_t>(c[ci + 1]));
        const unsigned own0 = side ? second[0] : first[0];
        const unsigned own1 = side ? second[1] : first[1];
        const unsigned src0 = side ? first[0] : second[0];
        const unsigned src1 = side ? first[1] : second[1];
        const unsigned other0 = __builtin_amdgcn_ds_bpermute((lane ^ 16) * 4, src0);
        const unsigned other1 = __builtin_amdgcn_ds_bpermute((lane ^ 16) * 4, src1);
        vector_t<unsigned, 4> packed;
        packed[0] = side ? other0 : own0;
        packed[1] = side ? other1 : own1;
        packed[2] = side ? own0 : other0;
        packed[3] = side ? own1 : other1;
        const int m = (ci + side) / T::E_N, n = (ci + side) % T::E_N;
        const int rr = (m * T::T_M + wm) * 16 + lane % 16;
        const int nn = (ContiguousWaveN ? n + wn * T::E_N : n * T::T_N + wn) * 16 + (lane / 32) * 8;
        store<8>(gc, __builtin_bit_cast(vector_t<bf16_t, 8>, packed), rr * stride_c + nn, 0,
                 number<T::STORE_CACHE>{});
    });
    if constexpr (T::E_M * T::E_N % 2 != 0) {
        constexpr int ci = T::E_M * T::E_N - 1;
        const auto words = __builtin_bit_cast(vector_t<unsigned, 2>, cast<bf16_t>(c[ci]));
        const unsigned other0 = __builtin_amdgcn_ds_bpermute((lane ^ 16) * 4, words[0]);
        const unsigned other1 = __builtin_amdgcn_ds_bpermute((lane ^ 16) * 4, words[1]);
        if (side == 0) {
            vector_t<unsigned, 4> packed{words[0], words[1], other0, other1};
            const int rr = ((ci / T::E_N) * T::T_M + wm) * 16 + lane % 16;
            const int nn =
                (ContiguousWaveN ? (ci % T::E_N) + wn * T::E_N : (ci % T::E_N) * T::T_N + wn) * 16 +
                (lane / 32) * 8;
            store<8>(gc, __builtin_bit_cast(vector_t<bf16_t, 8>, packed), rr * stride_c + nn, 0,
                     number<T::STORE_CACHE>{});
        }
    }
}

using namespace opus;

template<class T, int Vec, class Copy>
__device__ inline constexpr auto make_layout_shared_copy(Copy copy, int wave_id, int base,
                                                         int copies) {
    constexpr int lane_stride = Vec < 4 ? 4 : Vec;
    const auto shared_block_shape =
        opus::make_tuple(copies, opus::number<T::BLOCK_SIZE / 64>{}, 64_I, opus::number<Vec>{});
    constexpr auto shared_block_dim =
        opus::make_tuple(opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::p_dim{}),
                         opus::make_tuple(opus::y_dim{}));
    return opus::make_layout<Vec>(
               shared_block_shape,
               opus::unfold_x_stride(shared_block_dim, shared_block_shape,
                                     opus::tuple{opus::number<lane_stride>{}, 1_I}),
               opus::unfold_p_coord(shared_block_dim, opus::tuple{int(copy), wave_id, 0_I})) +
           base;
}

// A: global copy, LDS destination, and MFMA operand.
template<class T, class Copy>
__device__ inline constexpr auto make_layout_ga_scale(Copy copy, int lane_id, int wave_id,
                                                      int stride_a) {
    constexpr int threads_k = T::B_K / 16;
    constexpr int rows_per_wave = 64 / threads_k;
    constexpr int waves = T::BLOCK_SIZE / 64;
    constexpr int copies = (T::B_M * T::B_K + T::BLOCK_SIZE * 16 - 1) / (T::BLOCK_SIZE * 16);
    constexpr auto ga_block_shape =
        opus::make_tuple(opus::number<copies>{}, opus::number<waves>{},
                         opus::number<rows_per_wave>{}, opus::number<threads_k>{}, 16_I);
    constexpr auto ga_block_dim =
        opus::make_tuple(opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::p_dim{}),
                         opus::make_tuple(opus::p_dim{}, opus::y_dim{}));
    const auto logical_coord =
        opus::tuple{int(copy), wave_id, lane_id / threads_k, lane_id % threads_k};
    // XOR permutes K-vector ownership using the row coordinate. Compute that
    // coordinate from the same tile decomposition, independently of its pitch.
    const auto row_layout = opus::make_layout<-1>(
        ga_block_shape, opus::unfold_x_stride(ga_block_dim, ga_block_shape, opus::tuple{1_I, 0_I}),
        opus::unfold_p_coord(ga_block_dim, logical_coord));
    const int row_swizzle = row_layout(0_I) % threads_k;
    return opus::make_layout<16>(
        ga_block_shape,
        opus::unfold_x_stride(ga_block_dim, ga_block_shape, opus::tuple{stride_a, 1_I}),
        opus::unfold_p_coord(ga_block_dim, opus::tuple{int(copy), wave_id, lane_id / threads_k,
                                                       (lane_id % threads_k) ^ row_swizzle}));
}

template<class T, class Copy>
__device__ inline constexpr auto make_layout_sa_scale(Copy copy, int wave_id, int base) {
    return make_layout_shared_copy<T, 16>(copy, wave_id, base, T::A_COPIES);
}

template<class T, bool VectorOffsets = false>
__device__ inline constexpr auto make_layout_ra_scale(int row, int lane_id, int k_slice,
                                                      int base = 0) {
    constexpr auto ra_block_shape = opus::make_tuple(opus::number<T::B_M / 16>{}, 16_I,
                                                     opus::number<T::B_K / 128>{}, 2_I, 4_I, 16_I);
    constexpr auto ra_block_dim =
        opus::make_tuple(opus::make_tuple(opus::p_dim{}, opus::p_dim{}),
                         opus::make_tuple(opus::p_dim{}), opus::make_tuple(opus::y_dim{}),
                         opus::make_tuple(opus::p_dim{}), opus::make_tuple(opus::y_dim{}));
    const int swizzle = row % (T::B_K / 16);
    // XOR flips independent K-slice/lane coordinates. A flipped half axis
    // starts at half one and walks backwards, preserving MFMA operand order.
    const int half_stride = swizzle & 4 ? -64 : 64;
    auto layout = opus::make_layout<16>(
                      ra_block_shape,
                      opus::unfold_x_stride(
                          ra_block_dim, ra_block_shape,
                          opus::tuple{opus::number<T::B_K>{}, 128_I, half_stride, 16_I, 1_I}),
                      opus::unfold_p_coord(ra_block_dim,
                                           opus::tuple{row / 16, row % 16, k_slice ^ (swizzle / 8),
                                                       (lane_id / 16) ^ (swizzle % 4)})) +
                  base + (swizzle & 4) * 16;
    if constexpr (VectorOffsets)
        return opus_gemm_b_direct_detail::vectorize_layout_offsets(layout);
    else
        return layout;
}

// B operand layouts.
template<class T>
__device__ inline constexpr auto make_layout_rb_scale(int group_n, int lane_id, int k_slice,
                                                      int stride_b, int base = 0) {
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
template<class T, class Copy>
__device__ inline auto make_layout_gsfa_scale(Copy copy, int tid, int row, int m, int stride_sfa) {
    constexpr int Vec = T::BYTE_SCALE ? 1 : 16;
    const int copy_lane = int(copy) * T::BLOCK_SIZE + tid;
    const int local_row = copy_lane % (T::SA_ROWS / Vec) * Vec;
    const int column = copy_lane / (T::SA_ROWS / Vec);
    const int bounded_column = column < T::SCALE_COLS ? column : T::SCALE_COLS - 1;
    const bool valid_row = local_row < T::B_M && row + local_row < m;
    const bool valid = valid_row && (!T::BYTE_SCALE || column < T::SCALE_COLS);
    const int scale_column = valid ? bounded_column : T::SCALE_COLS - 1;
    const int scale_row = valid ? row + local_row : m;

    const auto block_shape =
        opus::make_tuple(opus::number<T::SCALE_COLS>{}, (m + Vec - 1) / Vec, opus::number<Vec>{});
    constexpr auto block_dim = opus::make_tuple(opus::make_tuple(opus::p_dim{}),
                                                opus::make_tuple(opus::p_dim{}, opus::y_dim{}));
    auto layout = opus::make_layout<Vec>(
                      block_shape,
                      opus::unfold_x_stride(block_dim, block_shape, opus::tuple{stride_sfa, 1_I}),
                      opus::unfold_p_coord(block_dim, opus::tuple{scale_column, scale_row / Vec})) +
                  scale_row % Vec;
    return opus_gemm_b_direct_detail::vectorize_layout_offsets(layout);
}

template<class T, class Copy>
__device__ inline constexpr auto make_layout_ssfa_scale(Copy copy, int wave_id, int base) {
    return make_layout_shared_copy<T, T::BYTE_SCALE ? 1 : 16>(copy, wave_id, base, T::SA_COPIES);
}

template<class T>
__device__ inline constexpr auto make_layout_rsfa_scale(int row, int k_slice, int block_m,
                                                        int columns, int base = 0) {
    constexpr int spacing = T::SA_SCALE;
    const auto rsfa_block_shape = opus::make_tuple(columns, block_m / 16, 16_I, 1_I);
    constexpr auto rsfa_block_dim =
        opus::make_tuple(opus::make_tuple(opus::p_dim{}),
                         opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::y_dim{}));
    return opus::make_layout<1>(
               rsfa_block_shape,
               opus::unfold_x_stride(rsfa_block_dim, rsfa_block_shape,
                                     opus::tuple{block_m * spacing, opus::number<spacing>{}}),
               opus::unfold_p_coord(rsfa_block_dim, opus::tuple{k_slice, row / 16, row % 16})) +
           base;
}

// B scale: global copy, LDS destination, and MFMA scale.
template<class T>
__device__ inline constexpr auto make_layout_gsfb_scale(int tid, int column, int groups,
                                                        int stride_sfb) {
    constexpr int Vec = T::SB_SLOTS ? 1 : 4;
    constexpr int columns = T::SCALE_COLS / Vec;
    const int copy_lane = tid % (T::SB_ROWS * columns);
    const auto block_shape = opus::make_tuple(groups, opus::number<columns>{}, opus::number<Vec>{});
    constexpr auto block_dim = opus::make_tuple(opus::make_tuple(opus::p_dim{}),
                                                opus::make_tuple(opus::p_dim{}, opus::y_dim{}));
    return opus::make_layout<Vec>(
        block_shape, opus::unfold_x_stride(block_dim, block_shape, opus::tuple{stride_sfb, 1_I}),
        opus::unfold_p_coord(block_dim,
                             opus::tuple{column / 128 + copy_lane / columns, copy_lane % columns}));
}

template<class T, class Copy>
__device__ inline constexpr auto make_layout_ssfb_scale(Copy copy, int wave_id, int base) {
    return make_layout_shared_copy<T, T::SB_SLOTS ? 1 : 4>(copy, wave_id, base, 1);
}

template<class T>
__device__ inline constexpr auto make_layout_rsfb_scale(int group_n, int k_column) {
    constexpr int spacing = T::SB_SLOTS ? 4 : 1;
    constexpr auto block_shape =
        opus::make_tuple(opus::number<T::SB_ROWS>{}, opus::number<T::SCALE_COLS>{}, 1_I);
    constexpr auto block_dim = opus::make_tuple(opus::make_tuple(opus::p_dim{}),
                                                opus::make_tuple(opus::p_dim{}, opus::y_dim{}));
    return opus::make_layout<1>(
               block_shape,
               opus::unfold_x_stride(
                   block_dim, block_shape,
                   opus::tuple{opus::number<T::SCALE_COLS * spacing>{}, opus::number<spacing>{}}),
               opus::unfold_p_coord(block_dim, opus::tuple{group_n, k_column})) +
           T::OFF_SB;
}

// C register and output layouts.
template<class T, bool Normal = false>
__device__ inline constexpr auto make_layout_rc_scale(int row, int column, int lane_id,
                                                      int stride_c) {
    if constexpr (Normal) {
        constexpr auto rc_block_shape = opus::make_tuple(opus::number<T::B_M / 16>{}, 4_I, 4_I,
                                                         opus::number<T::B_N / 16>{}, 16_I);
        constexpr auto rc_block_dim =
            opus::make_tuple(opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::y_dim{}),
                             opus::make_tuple(opus::p_dim{}, opus::p_dim{}));
        return opus::make_layout<1>(
            rc_block_shape,
            opus::unfold_x_stride(rc_block_dim, rc_block_shape, opus::tuple{stride_c, 1_I}),
            opus::unfold_p_coord(rc_block_dim,
                                 opus::tuple{row / 16, lane_id / 16, column / 16, lane_id % 16}));
    } else {
        constexpr auto rc_block_shape = opus::make_tuple(opus::number<T::B_M / 16>{}, 16_I,
                                                         opus::number<T::B_N / 16>{}, 4_I, 4_I);
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
__device__ inline constexpr auto make_layout_matrix_vector(int row, int column, int stride) {
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
__device__ inline constexpr auto make_layout_sc(int row, int column, int stride) {
    return make_layout_matrix_vector<8, Rows, T::B_N>(row, column, stride);
}

template<class T> __device__ inline constexpr auto make_layout_gc(int row, int column, int stride) {
    return make_layout_matrix_vector<8, T::B_M, T::B_N>(row, column, stride);
}

} // namespace opus_gemm_b_direct_detail

namespace opus {
template<class D> struct is_gmem<opus_gemm_b_direct_detail::async_gmem<D>> : true_type {};
} // namespace opus

template<class Traits>
__device__ __forceinline__ void opus_gemm_blockscale_mxscale_bpreshuffle_b_direct(
    opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 kargs) {
#if defined(OPUS_BP_FIXED_M) && defined(OPUS_BP_FIXED_N) && defined(OPUS_BP_FIXED_K)
    opus_gemm_prefetch_b_vgpr_detail::apply_fixed_args(kargs);
#endif
    using namespace opus;
    using namespace opus_gemm_b_direct_detail;
    namespace layout = opus_gemm_b_direct_detail;
    using T = opus::remove_cvref_t<Traits>;
    constexpr bool native_c = [] {
        if constexpr (requires { T::NATIVE_C; })
            return T::NATIVE_C;
        else
            return false;
    }();
    using Index = unsigned;

    // Prepare tile views, operand rings, scales, and accumulators.
    constexpr int BM = T::B_M, BN = T::B_N, NB = T::STAGES, BD = T::B_AHEAD, BS = T::B_SETS,
                  P = T::PANELS;
    constexpr int MR = T::MR, NR = T::NR, K = T::K, SC = T::SCALE_COLS, BK = T::B_K, KS = BK / 128;
    const int M = kargs.m, N = kargs.n;
    const int split = T::SPLIT_K > 1 ? block_id_z() : 0;
    using D_C = std::conditional_t<T::SPLIT_K == 1, bf16_t, float>;
    const int stride_a = T::CONTIGUOUS ? T::TOTAL_K : kargs.stride_a,
              stride_b = T::CONTIGUOUS ? T::TOTAL_K : kargs.stride_b, stride_c = kargs.stride_c;
    static_assert((BK == 128 || BK == 256) && SC == K / 128);
    static_assert(T::T_M == 1 && T::T_N * 64 == T::BLOCK_SIZE);
    static_assert(T::MR * 16 == T::B_M && T::NR * T::T_N * 16 == T::B_N);
    const Index tid = thread_id_x(), lane_id = tid % 64, lane_id_m = lane_id % 16;
    const Index wave_id = __builtin_amdgcn_readfirstlane(tid / 64), wave_id_n = wave_id;
    if constexpr (requires { T::THREAD_RANGE_KNOWN; }) {
        __builtin_assume(static_cast<unsigned>(tid) < T::BLOCK_SIZE);
        __builtin_assume(static_cast<unsigned>(wave_id) < T::BLOCK_SIZE / 64);
    }
    const unsigned pid = block_id_x(), gm = (M + BM - 1) / BM, gn = N / BN, total = gm * gn;
    const unsigned tile = T::XCD_ORDER ? pid % 8 * ((total + 7) / 8) + pid / 8 : pid;
    if constexpr (T::XCD_ORDER)
        if (tile >= total)
            return;
    const int row = T::GRID_2D ? block_id_y() * BM : (M <= BM ? 0 : (tile / gn) * BM);
    const int col = T::GRID_2D ? block_id_x() * BN : (M <= BM ? tile * BN : (tile % gn) * BN);
    int valid_rows = M % BM == 0 ? BM : (M <= BM ? M : (row == (gm - 1) * BM ? M % BM : BM));
    if constexpr (requires { T::SIMPLE_TAIL_ROWS; })
        valid_rows = M - row < BM ? M - row : BM;
    auto g_a =
        make_async_gmem(reinterpret_cast<const fp8_t*>(kargs.ptr_a) + row * stride_a + split * K,
                        static_cast<unsigned>(valid_rows * stride_a - split * K));
    auto g_b =
        make_gmem(reinterpret_cast<const fp8_t*>(kargs.ptr_b) + col * stride_b + split * K * 16,
                  static_cast<unsigned>(BN * stride_b - split * K * 16));
    auto g_sfa = make_async_gmem(reinterpret_cast<const unsigned char*>(kargs.ptr_sfa) +
                                     split * SC * kargs.stride_sfa,
                                 static_cast<unsigned>((SC - 1) * kargs.stride_sfa + M));
    auto g_sfb = make_async_gmem(reinterpret_cast<const unsigned char*>(kargs.ptr_sfb) + split * SC,
                                 static_cast<unsigned>((N / 128 - 1) * kargs.stride_sfb + SC));
    auto g_c =
        make_gmem(reinterpret_cast<D_C*>(kargs.ptr_c) + static_cast<int64_t>(split) * M * stride_c +
                      static_cast<int64_t>(row) * stride_c + col,
                  static_cast<unsigned>(((valid_rows - 1) * stride_c + BN) * sizeof(D_C)));
    alignas(16) __shared__ unsigned char arena[T::LDS_BYTES];
    auto s_a = make_smem(reinterpret_cast<fp8_t*>(arena));
    auto s_sf = make_smem(arena);
    using Base =
        decltype(make_mfma<fp8_t, fp8_t, fp32_t>(16_I, 16_I, 128_I, mfma_adaptor_swap_ab{}));
    array<typename Base::vtype_c, MR * NR> c{};
    array<array<array<typename Base::vtype_b, NR>, KS>, BS> b_regs;
    constexpr int VecScaleA = T::BYTE_SCALE ? 1 : 16;
    constexpr int VecScaleB = T::SB_SLOTS ? 1 : 4;
    auto issue_b = [&](auto bi, int kt) {
        constexpr int set = decltype(bi)::value;
        static_for<KS>([&](auto ki) {
            constexpr int ks = decltype(ki)::value;
            static_for<NR>([&](auto ni) {
                constexpr int n = decltype(ni)::value;
                const auto u_rb =
                    layout::make_layout_rb_scale<T>(wave_id_n * NR + n, lane_id, ks, stride_b);
                b_regs[set][ks][n] = load<16>(g_b, u_rb, kt * (BK * 16), number<T::B_CACHE>{});
            });
        });
        __builtin_amdgcn_sched_barrier(0);
    };
    auto issue_a = [&](int slot, int kt) {
        static_for<T::A_COPIES>([&](auto ii) {
            const auto u_ga = layout::make_layout_ga_scale<T>(ii, lane_id, wave_id, stride_a);
            const auto u_sa = layout::make_layout_sa_scale<T>(ii, wave_id, slot * T::STAGE_BYTES);
            async_load<16>(g_a, s_a.ptr, u_ga, u_sa, kt * BK, 0_I, number<T::CACHECTL_A>{});
        });
        commit_async_loads();
        __builtin_amdgcn_sched_barrier(0);
    };
    // Retire one A/B pair with the original VMEM ledger and pinning order.
    auto step = [&](int t, int slot, auto bi, auto wai, auto wbi, auto fai, auto fbi) {
        constexpr int set = decltype(bi)::value, wa = decltype(wai)::value,
                      wb = decltype(wbi)::value;
        constexpr bool future_a = decltype(fai)::value, future_b = decltype(fbi)::value;
        __builtin_amdgcn_sched_barrier(0);
        s_waitcnt_vmcnt(number<(wa > 63 ? 63 : wa)>{});
        s_waitcnt_lgkmcnt(0_I);
        sync_threads();
        if constexpr (future_b)
            issue_b(number<(set + BD) % BS>{}, t + BD);
        array<array<typename Base::vtype_a, MR>, KS> af;
        array<array<unsigned, MR>, KS> as;
        array<array<unsigned, NR>, KS> bs;
        static_for<KS>([&](auto ki) {
            constexpr int ks = decltype(ki)::value;
            const int scale_col = t * KS + ks;
            static_for<NR>([&](auto ni) {
                constexpr int nr = decltype(ni)::value;
                const int group_n = (wave_id_n * NR * 16 + nr * 16) / 128;
                const auto u_rsfb = layout::make_layout_rsfb_scale<T>(group_n, scale_col);
                bs[ks][nr] = load<1>(s_sf, u_rsfb)[0];
            });
            static_for<MR>([&](auto mi) {
                constexpr int m = decltype(mi)::value;
                const int ar = m * 16 + lane_id_m;
                const auto u_ra =
                    layout::make_layout_ra_scale<T>(ar, lane_id, ks, slot * T::STAGE_BYTES);
                {
                    af[ks][m] = load_operand_fragment<typename Base::vtype_a>(s_a, u_ra);
                }
                const auto u_rsfa =
                    layout::make_layout_rsfa_scale<T>(ar, scale_col, T::SA_ROWS, SC, T::OFF_SA);
                as[ks][m] = load<1>(s_sf, u_rsfa)[0];
            });
        });
        __builtin_amdgcn_sched_barrier(0);
        if constexpr (future_a)
            issue_a(slot == 0 ? NB - 1 : slot - 1, t + NB - 1);
        __builtin_amdgcn_sched_barrier(0);
        s_waitcnt_vmcnt(number<(wb > 63 ? 63 : wb)>{});
        s_waitcnt_lgkmcnt(0_I);
        static_for<KS>([&](auto ki) {
            constexpr int ks = decltype(ki)::value;
            static_for<MR>([&](auto mi) {
                constexpr int m = decltype(mi)::value;
                static_for<NR>([&](auto ni) {
                    constexpr int n = decltype(ni)::value, ci = m * NR + n;
                    {
                        c[ci] = Base{}(af[ks][m], b_regs[set][ks][n], c[ci],
                                       static_cast<int>(as[ks][m]), static_cast<int>(bs[ks][n]),
                                       0_I, 0_I);
                    }
                });
            });
        });
        if constexpr (!native_c && !T::C_AGPR && !T::PIN_C)
            static_for<MR * NR>([&](auto ci) {
                auto& v = c[decltype(ci)::value];
                asm volatile("" : "+v"(v));
            });
        __builtin_amdgcn_sched_barrier(0);
    };

    // ===== Prologue =====
    // Issue B, the first A stage, both scale panels, then the remaining A stages.
    static_for<T::INITIAL_B>([&](auto i) { issue_b(i, decltype(i)::value); });
    issue_a(0, 0);
    static_for<T::SA_COPIES>([&](auto ii) {
        const auto u_gsfa = layout::make_layout_gsfa_scale<T>(ii, tid, row, M, kargs.stride_sfa);
        const auto u_ssfa = layout::make_layout_ssfa_scale<T>(ii, wave_id, T::OFF_SA);
        async_load<VecScaleA>(g_sfa, s_sf.ptr, u_gsfa, u_ssfa);
    });
    const auto u_gsfb = layout::make_layout_gsfb_scale<T>(tid, col, N / 128, kargs.stride_sfb);
    const auto u_ssfb = layout::make_layout_ssfb_scale<T>(0_I, wave_id, T::OFF_SB);
    async_load<VecScaleB>(g_sfb, s_sf.ptr, u_gsfb, u_ssfb);
    static_for<T::INITIAL_A - 1>(
        [&](auto i) { issue_a(decltype(i)::value + 1, decltype(i)::value + 1); });

    // ===== Main loop =====
    constexpr int START = T::STEADY < NB - 1 ? T::STEADY : NB - 1;
    static_for<START>([&](auto ti) {
        constexpr int t = decltype(ti)::value;
        step(t, t % NB, number<t % BS>{}, number<T::wait_a(t)>{}, number<T::wait_b(t)>{},
             number<true>{}, number<(t + BD < P)>{});
    });
    constexpr int PHASES = []() constexpr {
        int a = NB, b = BS;
        while (b) {
            int r = a % b;
            a = b;
            b = r;
        }
        return NB / a * BS;
    }();
    constexpr int GROUPS = (T::STEADY - START) / PHASES, REGULAR_END = START + GROUPS * PHASES;
    if constexpr (GROUPS > 0) {
#pragma clang loop unroll(disable)
        for (int group = 0; group < GROUPS; ++group) {
            static_for<PHASES>([&](auto phase) {
                constexpr int ph = decltype(phase)::value, t0 = START + ph;
                step(START + group * PHASES + ph, t0 % NB, number<t0 % BS>{},
                     number<T::wait_a(t0)>{}, number<T::wait_b(t0)>{}, number<true>{},
                     number<true>{});
            });
        }
    }

    // ===== Epilogue =====
    // Retire the final panels with their existing refill conditions and fixed waits.
    static_for<P - REGULAR_END>([&](auto ti) {
        constexpr int t = REGULAR_END + decltype(ti)::value;
        step(t, t % NB, number<t % BS>{}, number<T::wait_a(t)>{}, number<T::wait_b(t)>{},
             number<(t + NB - 1 < P)>{}, number<(t + BD < P)>{});
    });

    // ===== Write back =====
    if constexpr (T::SPLIT_K > 1) {
        static_for<MR>([&](auto mi) {
            constexpr int m = decltype(mi)::value;
            static_for<NR>([&](auto ni) {
                constexpr int n = decltype(ni)::value, ci = m * NR + n;
                const auto u_gc = layout::make_layout_rc_scale<T>(
                    m * 16, wave_id_n * NR * 16 + n * 16, lane_id, stride_c);
                store<4>(g_c, c[ci], u_gc, 0, number<T::STORE_CACHE>{});
            });
        });
    } else if constexpr (T::OUTPUT == 2) {
        opus_gemm_b_direct_detail::store_mfma16_packed<T, true>(c, g_c, lane_id, 0, wave_id_n,
                                                                stride_c);
    } else {
        // Stage C in LDS and gather contiguous BF16 stores.
        auto s_c = make_smem(reinterpret_cast<bf16_t*>(arena));
        constexpr int MAX_ROWS = BM;
        static_for<(MAX_ROWS + T::C_ROWS - 1) / T::C_ROWS>([&](auto parti) {
            constexpr int part = decltype(parti)::value;
            constexpr int PART_ROWS =
                MAX_ROWS - part * T::C_ROWS < T::C_ROWS ? MAX_ROWS - part * T::C_ROWS : T::C_ROWS;
            s_waitcnt_lgkmcnt(0_I);
            sync_threads();
            static_for<MR>([&](auto mi) {
                constexpr int m = decltype(mi)::value;
                if constexpr (m * 16 / T::C_ROWS == part && m * 16 < MAX_ROWS)
                    static_for<NR>([&](auto ni) {
                        constexpr int n = decltype(ni)::value, ci = m * NR + n;
                        const auto u_sc = layout::make_layout_rc_scale<T>(
                            m * 16 - part * T::C_ROWS, wave_id_n * NR * 16 + n * 16, lane_id,
                            BN + 8);
                        store<4>(s_c, cast<bf16_t>(c[ci]), u_sc);
                    });
            });
            s_waitcnt_lgkmcnt(0_I);
            sync_threads();
            constexpr int CHUNKS = (PART_ROWS * (BN / 8) + T::BLOCK_SIZE - 1) / T::BLOCK_SIZE;
            array<vector_t<bf16_t, 8>, CHUNKS> output;
            static_for<CHUNKS>([&](auto ii) {
                constexpr int i = decltype(ii)::value;
                const int index = i * T::BLOCK_SIZE + tid, rr = index / (BN / 8),
                          nn = index % (BN / 8) * 8;
                const auto u_sc = layout::make_layout_sc<T, PART_ROWS>(rr, nn, BN + 8);
                if constexpr (PART_ROWS * (BN / 8) % T::BLOCK_SIZE == 0)
                    output[i] = load<8>(s_c, u_sc);
                else if (rr < PART_ROWS)
                    output[i] = load<8>(s_c, u_sc);
            });
            __builtin_amdgcn_sched_barrier(0);
            s_waitcnt_lgkmcnt(0_I);
            static_for<CHUNKS>([&](auto ii) {
                constexpr int i = decltype(ii)::value;
                const int index = i * T::BLOCK_SIZE + tid, rr = index / (BN / 8),
                          nn = index % (BN / 8) * 8;
                if (rr < PART_ROWS && part * T::C_ROWS + rr < valid_rows)
                    store<8>(g_c, output[i],
                             layout::make_layout_gc<T>(part * T::C_ROWS + rr, nn, stride_c), 0,
                             number<T::STORE_CACHE>{});
            });
        });
    }
    s_waitcnt_vmcnt(0_I);
}

// Existing generated CO stubs retain their registered compute entry.
namespace opus_gemm_b_direct_detail {
template<class Traits>
__device__ __forceinline__ void
bulk_panel_body(opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 kargs) {
    opus_gemm_prefetch_b_vgpr_detail::bulk_panel_body<Traits>(kargs);
}
} // namespace opus_gemm_b_direct_detail

namespace opus_gemm_prefetch_b_vgpr_detail {
// Select the prefetch unit from traits; no runtime shape-dependent branch.
template<class Traits, int PublishM>
__device__ __forceinline__ void
bulk_panel_body(opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 args) {
    if constexpr (Traits::B_PANEL_PREFETCH)
        opus_gemm_blockscale_mxscale_bpreshuffle_b_direct<Traits>(args);
    else
        fragment_prefetch_body<Traits, PublishM>(args);
}
} // namespace opus_gemm_prefetch_b_vgpr_detail

#endif

template<class Traits>
__global__ __launch_bounds__(Traits::BLOCK_SIZE, Traits::WG_PER_CU)
void opus_gemm_blockscale_mxscale_bpreshuffle_prefetch_b_vgpr_bulk_panel_kernel(
    opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 args) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx950__)
    opus_gemm_prefetch_b_vgpr_detail::bulk_panel_body<Traits>(args);
#endif
}

template<class Traits>
__global__ __launch_bounds__(Traits::BLOCK_SIZE, Traits::WG_PER_CU)
void opus_gemm_blockscale_mxscale_bpreshuffle_prefetch_b_vgpr_overlap_bulk_panel_kernel(
    opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 args) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx950__)
    opus_gemm_prefetch_b_vgpr_detail::bulk_panel_overlap_body<Traits>(args);
#endif
}

template<class Traits>
__global__ __launch_bounds__(Traits::BLOCK_SIZE, Traits::WG_PER_CU)
void opus_gemm_blockscale_mxscale_bpreshuffle_b_direct_bulk_panel_kernel(
    opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 kargs) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx950__)
    opus_gemm_prefetch_b_vgpr_detail::bulk_panel_body<Traits>(kargs);
#endif
}
