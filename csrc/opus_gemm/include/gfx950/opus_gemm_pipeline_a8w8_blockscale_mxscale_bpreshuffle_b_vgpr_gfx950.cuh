// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <opus/hip_minimal.hpp>
#include <opus/opus.hpp>
#include "../opus_gemm_utils.cuh"
#include "opus_gemm_traits_a8w8_blockscale_mxscale_bpreshuffle_b_vgpr_gfx950.cuh"
#include "opus_gemm_traits_a8w8_blockscale_mxscale_bpreshuffle_b_vgpr_overlap_gfx950.cuh"

#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx950__)
namespace opus_gemm_prefetch_b_vgpr_detail {

// Global A copies preserve the row-dependent XOR permutation used by the LDS ring.
template<class T>
__device__ inline auto make_layout_ga(int lane_id, int wave_id_n) {
    constexpr auto ga_block_shape = opus::make_tuple(
        opus::number<T::A_COPIES>{},
        opus::number<T::NUM_WAVES>{},
        opus::number<T::WARP_SIZE>{},
        opus::number<T::VEC_A>{});

    constexpr auto ga_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}, opus::y_dim{}));

    return opus::make_layout<T::VEC_A>(
        ga_block_shape,
        opus::unfold_x_stride(ga_block_dim, ga_block_shape, opus::tuple{opus::number<1>{}}),
        opus::unfold_p_coord(ga_block_dim, opus::tuple{wave_id_n, lane_id}));
}

// VMEM-to-LDS adds the lane displacement; the layout contains each wave base.
template<class T>
__device__ inline auto make_layout_sa(int wave_id_n) {
    constexpr auto sa_block_shape = opus::make_tuple(
        opus::number<T::A_COPIES>{},
        opus::number<T::NUM_WAVES>{},
        opus::number<T::VEC_A>{});

    constexpr auto sa_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}));

    return opus::make_layout<T::VEC_A>(
        sa_block_shape,
        opus::unfold_x_stride(sa_block_dim, sa_block_shape, opus::tuple{opus::number<T::WARP_SIZE * T::VEC_A>{}, opus::number<1>{}}),
        opus::unfold_p_coord(sa_block_dim, opus::tuple{wave_id_n}));
}

// Construct only the current K128 A fragment's offsets at its consumption point.
template<class T>
__device__ inline auto make_layout_ra(int lane_id) {
    constexpr auto ra_block_shape = opus::make_tuple(
        opus::number<T::MR>{},
        opus::number<T::W_M>{},
        opus::number<T::A_CHUNKS_PER_FRAGMENT>{},
        opus::number<T::WARP_SIZE / T::W_M>{},
        opus::number<T::VEC_A>{});

    constexpr auto ra_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::y_dim{}));

    return opus::make_layout<T::VEC_A>(
        ra_block_shape,
        opus::unfold_x_stride(ra_block_dim, ra_block_shape, opus::tuple{opus::number<T::B_K>{}, opus::number<1>{}}),
        opus::unfold_p_coord(ra_block_dim, opus::tuple{lane_id % T::W_M, lane_id / T::W_M}));
}

// Each B fragment has two K64 halves and is loaded directly into its VGPR slot.
template<class T>
__device__ inline auto make_layout_gb(int lane_id, int wave_id_n, int stride_b) {
    constexpr auto gb_block_shape = opus::make_tuple(
        opus::number<T::B_K / T::W_K>{},
        opus::number<T::T_N>{},
        opus::number<T::NR>{},
        opus::number<T::B_CHUNKS_PER_FRAGMENT>{},
        opus::number<T::WARP_SIZE / T::W_N>{},
        opus::number<T::W_N>{},
        opus::number<T::VEC_B>{});

    constexpr auto gb_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}),
        opus::make_tuple(opus::y_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::y_dim{}));

    return opus::make_layout<T::VEC_B>(
        gb_block_shape,
        opus::unfold_x_stride(gb_block_dim, gb_block_shape, opus::tuple{opus::number<T::W_N * T::W_K>{}, T::W_N * stride_b, opus::number<T::WARP_SIZE * T::VEC_B>{}, opus::number<1>{}}),
        opus::unfold_p_coord(gb_block_dim, opus::tuple{wave_id_n, lane_id / T::W_N, lane_id % T::W_N}));
}

// Raw E8M0 scale panels retain their physical K-major A and N-major B strides.
template<class T>
__device__ inline auto make_layout_gsfa(int stride_sfa) {
    constexpr auto gsfa_block_shape = opus::make_tuple(
        opus::number<T::SCALE_COLS>{},
        opus::number<T::SA_ROWS>{});

    constexpr auto gsfa_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}),
        opus::make_tuple(opus::y_dim{}));

    return opus::make_layout(
        gsfa_block_shape,
        opus::unfold_x_stride(gsfa_block_dim, gsfa_block_shape, opus::tuple{stride_sfa, opus::number<1>{}}),
        opus::unfold_p_coord(gsfa_block_dim, opus::tuple{}));
}

template<class T>
__device__ inline auto make_layout_ssfa(int wave_id_n) {
    constexpr auto ssfa_block_shape = opus::make_tuple(
        opus::number<T::SA_COPIES>{},
        opus::number<T::NUM_WAVES>{},
        opus::number<T::VEC_SCALE_A>{});

    constexpr auto ssfa_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}));

    return opus::make_layout<T::VEC_SCALE_A>(
        ssfa_block_shape,
        opus::unfold_x_stride(ssfa_block_dim, ssfa_block_shape, opus::tuple{opus::number<T::WARP_SIZE * T::VEC_SCALE_A>{}, opus::number<1>{}}),
        opus::unfold_p_coord(ssfa_block_dim, opus::tuple{wave_id_n}));
}

template<class T>
__device__ inline auto make_layout_rsfa(int lane_id) {
    constexpr auto rsfa_block_shape = opus::make_tuple(
        opus::number<T::B_K / T::GROUP_K>{},
        opus::number<T::MR>{},
        opus::number<T::W_M>{});

    constexpr auto rsfa_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}),
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}));

    return opus::make_layout(
        rsfa_block_shape,
        opus::unfold_x_stride(rsfa_block_dim, rsfa_block_shape, opus::tuple{opus::number<T::SA_ROWS>{}, opus::number<1>{}}),
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
    for (int rows = bm; rows >= 16; rows -= 16)
        if (bm % rows == 0 && rows * (bn + 8) * 2 <= pipeline_bytes)
            return rows;
    return 16;
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
__device__ __forceinline__ void bulk_panel_body(opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 args) {
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
        if constexpr (T::ROTATE_B) return 1;
        int a = NB, b = BS;
        while (b) { int r = a % b; a = b; b = r; }
        return NB / a * BS;
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
    auto u_ga = make_layout_ga<T>(lane_id, wave_id_n);
    opus::static_for<T::A_COPIES>([&](auto i) {
        const int linear = u_ga.offsets[i];
        const int row = linear / T::B_K;
        const int k = (linear % T::B_K) ^ ((row % (T::B_K / T::VEC_A)) * T::VEC_A);
        u_ga.offsets[i] = row * stride_a + k;
    });
    const auto u_sa = make_layout_sa<T>(wave_id_n);
    const auto u_gb = make_layout_gb<T>(lane_id, wave_id_n, stride_b);
    const auto u_gsfa = make_layout_gsfa<T>(args.stride_sfa);
    const auto u_ssfa = make_layout_ssfa<T>(wave_id_n);
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
            load<16>(g_b, u_gb.offsets[fragment * 2], kt * (BK * 16), number<T::B_CACHE>{}), 0_I, 16_I);
        set_slice(r_b[set],
            load<16>(g_b, u_gb.offsets[fragment * 2 + 1], kt * (BK * 16), number<T::B_CACHE>{}), 16_I, 32_I);
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

    auto issue_scale_panels = [&]() {
        static_for<T::SA_COPIES>([&](auto ii) {
            constexpr int i = decltype(ii)::value;
            const int index = (i * T::BLOCK_SIZE + thread_id) * 16;
            const int sr = index % T::SA_ROWS;
            const int scol = index / T::SA_ROWS;
            const int scale_col = scol < SC ? scol : SC - 1;
            int source = sr < BM && row + sr < M ? u_gsfa(scale_col, sr) + row : (SC - 1) * args.stride_sfa + M;
            async_load<16>(g_sfa, arena + T::OFF_SA + u_ssfa.offsets[i], source);
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

    // Prepare the panel consumer before issuing the prologue requests.
    auto consume_panel = [&](int t, auto phase_i) {
        constexpr int phase = decltype(phase_i)::value;
        constexpr bool future_a = phase + T::INITIAL_A < P;
        constexpr int issued_b = (phase * F + BD < P * F ? phase * F + BD : P * F);
        constexpr int before_a = BD + (phase - T::INITIAL_A) * F + 1;
        constexpr int newer_b_a = phase < T::INITIAL_A ? issued_b : (issued_b > before_a ? issued_b - before_a : 0);
        constexpr int newer_a = phase < T::INITIAL_A ? T::INITIAL_A - phase - 1 : (P - phase - 1 < T::INITIAL_A - 1 ? P - phase - 1 : T::INITIAL_A - 1);
        constexpr int wa = newer_b_a * 2 + newer_a * T::A_COPIES;
        const int slot = T::ROTATE_B ? t % NB : phase % NB;
        __builtin_amdgcn_sched_barrier(0);
        s_waitcnt_vmcnt(number<(wa > 63 ? 63 : wa)>{});
        s_waitcnt_lgkmcnt(0_I);
        opus::sync_threads();

        array<array<typename Base::vtype_a, MR>, T::ROTATE_B ? 1 : KS> r_a;
        array<array<unsigned, (MR + 3) / 4>, KS> r_sfa{};
        array<unsigned, KS> r_sfb{};
        static_for<KS>([&](auto ki) {
            constexpr int ks = decltype(ki)::value;
            static_for<(BN / WaveColumns + 127) / 128>([&](auto group_i) {
                constexpr int group = decltype(group_i)::value;
                const unsigned scale = load<1>(s_scale,
                    T::OFF_SB + u_rsfb(ks, group) + t * KS * (T::SB_SLOTS ? 4 : 1))[0];
                r_sfb[ks] |= scale << (group * 8);
            });
            static_for<MR>([&](auto mi) {
                constexpr int m = decltype(mi)::value;
                const unsigned scale = load<1>(s_scale, T::OFF_SA + u_rsfa(ks, m) + t * KS * T::SA_ROWS)[0];
                r_sfa[ks][m / 4] |= scale << ((m % 4) * 8);
            });
        });
        auto read_a = [&](auto ki) {
            constexpr int ks = decltype(ki)::value;
            auto u_ra = make_layout_ra<T>(lane_id);
            opus::static_for<T::MR * T::A_CHUNKS_PER_FRAGMENT>([&](auto i) {
                constexpr int m = decltype(i)::value / T::A_CHUNKS_PER_FRAGMENT;
                constexpr int half = decltype(i)::value % T::A_CHUNKS_PER_FRAGMENT;
                const int row = m * T::W_M + lane_id % T::W_M;
                const int k_vector = (ks * (T::W_K / T::VEC_A) + lane_id / T::W_M + half * (T::WARP_SIZE / T::W_M)) ^ (row % (T::B_K / T::VEC_A));
                u_ra.offsets[i] = row * T::B_K + k_vector * T::VEC_A;
            });
            static_for<MR>([&](auto mi) {
                constexpr int m = decltype(mi)::value;
                const int lo = slot * T::STAGE_BYTES + u_ra.offsets[m * 2];
                const int hi = slot * T::STAGE_BYTES + u_ra.offsets[m * 2 + 1];
                if constexpr (T::PIN_A) {
                    array<int, 2> offsets;
                    offsets[0] = lo;
                    offsets[1] = hi;
                    load_operand_fragment_pinned<16, 0>(number<(ks * MR + m) * 8>{}, r_a[T::ROTATE_B ? 0 : ks][m], s_a, offsets);
                } else {
                    set_slice(r_a[T::ROTATE_B ? 0 : ks][m], load<16>(s_a, lo), 0_I, 16_I);
                    set_slice(r_a[T::ROTATE_B ? 0 : ks][m], load<16>(s_a, hi), 16_I, 32_I);
                }
            });
        };
        if constexpr (!T::ROTATE_B)
            static_for<KS>([&](auto ki) { read_a(ki); });
        static_for<KS * NR>([&](auto fragment_i) {
            constexpr int fragment = decltype(fragment_i)::value;
            constexpr int ks = fragment / NR;
            constexpr int n = fragment % NR;
            constexpr int absolute = phase * F + fragment;
            if constexpr (T::ROTATE_B && n == 0)
                read_a(number<ks>{});
            if constexpr (absolute + BD < P * F)
                issue_b(number<((T::ROTATE_B ? fragment : absolute) + BD) % BS>{}, number<(fragment + BD) % F>{}, t + (fragment + BD) / F);
            constexpr int newer_b = P * F - absolute - 1 < BD ? P * F - absolute - 1 : BD;
            constexpr int first_a = absolute < BD ? 0 : (absolute - BD + F - 1) / F;
            constexpr int last_a = (phase - (fragment == 0 ? 1 : 0) < P - T::INITIAL_A - 1)
                ? phase - (fragment == 0 ? 1 : 0) : P - T::INITIAL_A - 1;
            constexpr int pending_a = last_a >= first_a ? (last_a - first_a + 1) * T::A_COPIES : 0;
            constexpr int wait = newer_b * 2 + pending_a;
            __builtin_amdgcn_sched_barrier(0);
            s_waitcnt_vmcnt(number<(wait > 63 ? 63 : wait)>{});
            if constexpr (fragment == 0 || (T::ROTATE_B && n == 0))
                s_waitcnt_lgkmcnt(0_I);
            if constexpr (future_a && fragment == 0 && PublishM < 0)
                issue_a((T::ROTATE_B ? t + NB - 1 : phase + NB - 1) % NB, t + NB - 1);

            static_for<MR>([&](auto mi) {
                constexpr int m = decltype(mi)::value;
                constexpr int ci = m * NR + n;
                if constexpr (T::PIN_C) {
                    [[clang::amdgpu_pin_agpr(128 + ci * 4)]] r_c[ci] = r_c[ci];
                    [[clang::amdgpu_pin_agpr(128 + ci * 4)]]
                    r_c[ci] = Base{}(r_a[T::ROTATE_B ? 0 : ks][m], r_b[(T::ROTATE_B ? fragment : phase * F + fragment) % BS], r_c[ci],
                        static_cast<int>(r_sfa[ks][m / 4]), static_cast<int>(r_sfb[ks]), number<m % 4>{}, number<n / 8>{});
                } else {
                    r_c[ci] = Base{}(r_a[T::ROTATE_B ? 0 : ks][m], r_b[(T::ROTATE_B ? fragment : phase * F + fragment) % BS], r_c[ci],
                        static_cast<int>(r_sfa[ks][m / 4]), static_cast<int>(r_sfb[ks]), number<m % 4>{}, number<n / 8>{});
                }
                if constexpr (future_a && fragment == 0 && m == PublishM)
                    issue_a((T::ROTATE_B ? t + NB - 1 : phase + NB - 1) % NB, t + NB - 1);
            });
        });
        if constexpr (T::ROTATE_B) {
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
    bulk_panel_body<Traits, Traits::PUBLISH_M>(args);
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
