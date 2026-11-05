// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <opus/hip_minimal.hpp>
#include <opus/opus.hpp>
#include "../opus_gemm_utils.cuh"
#include "opus_gemm_traits_a8w8_blockscale_mxscale_bpreshuffle_b_vgpr_gfx950.cuh"

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
        opus::number<T::MR>{},
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
        opus::number<T::MR>{},
        opus::number<T::T_M>{},
        opus::number<T::W_M>{},
        opus::number<T::NR>{},
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
        opus::unfold_x_stride(gc_packed_block_dim, gc_packed_block_shape, opus::tuple{stride_c, opus::number<(ContiguousWaveN ? 1 : T::T_N) * T::W_N>{}, opus::number<(ContiguousWaveN ? T::NR : 1) * T::W_N>{}, opus::number<1>{}}),
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
    static_for<T::MR * T::NR / 2>([&](auto pair) {
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
    if constexpr (T::MR * T::NR % 2 != 0) {
        constexpr int ci = T::MR * T::NR - 1;
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

// Reuse the same LDS arena in both runtime scale-cache ranges.
template<class T>
__device__ __forceinline__ unsigned char* b_pipeline_arena() {
    alignas(16) __shared__ unsigned char arena[T::LDS_BYTES];
    return arena;
}

// Stream individual B fragments across N, reusing each fragment for all M repeats.
template<class Traits, int PublishM = -1, bool Short = false, bool SingleCache = false, class ArenaTraits = Traits>
__device__ __forceinline__ void bulk_panel_body(opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 args) {
    using namespace opus;

    using T = opus::remove_cvref_t<Traits>;
    using Index = unsigned;
    constexpr int WaveColumns = T::T_N;
    using Schedule = opus_gemm_bpreshuffle_b_schedule<T, PublishM, Short, SingleCache>;
    if constexpr (Schedule::SMALL_SCALE_CACHE) {
        if (args.k > 0 && args.k / T::GROUP_K <= 4) {
            using SmallScale = typename T::template scale_panel_traits<4>;
            bulk_panel_body<SmallScale, PublishM, Short, SingleCache, ArenaTraits>(args);
            return;
        }
    }
    if constexpr (Schedule::PINNED_SCALE_CACHE) {
        if (args.k > 0 && args.k / T::GROUP_K <= T::SCALE_COLS) {
            bulk_panel_body<T, PublishM, false, true>(args);
            return;
        }
    }

    // A finite runtime K range needs one scale cache for this tile layout.
    if constexpr (Schedule::W2_SCALE_CACHE) {
        if (args.k > 0 && args.k / T::GROUP_K <= T::SCALE_COLS) {
            bulk_panel_body<T, PublishM, false, true>(args);
            return;
        }
    }

    // Tile geometry and the schedule derived by traits.
    constexpr int BM = T::B_M;
    constexpr int BN = T::B_N;
    constexpr int MR = T::MR;
    constexpr int NR = T::NR;
    constexpr int BK = T::B_K;
    constexpr int KS = BK / T::W_K;
    constexpr int A_READ_LEAD = Schedule::A_READ_LEAD;
    constexpr bool NARROW = Schedule::MEMORY_HEAVY;
    constexpr bool PARTIAL_A = Schedule::PARTIAL_A;
    constexpr bool STREAM_A = Schedule::STREAM_A;
    constexpr bool THREE_A = Schedule::THREE_A;
    constexpr bool LATE_A = Schedule::LATE_A;
    constexpr int PUBLISH_A = Schedule::PUBLISH_A;
    constexpr int BD = Schedule::B_AHEAD;
    constexpr int BS = Schedule::B_SLOTS;
    constexpr int NB = Schedule::A_STAGES;
    constexpr int INITIAL_A = Schedule::INITIAL_A;
    constexpr int SCALE_K_TILES = Schedule::SCALE_K_TILES;
    constexpr bool ROTATE_RING = Schedule::ROTATE_RING;
    constexpr bool STAGED_A = Schedule::STAGED_A;

    const int M = args.m;
    const int N = args.n;
    if (args.k <= 0 || args.k % BK) return;
    const unsigned K = args.k;
    const unsigned P = K / BK;
    const unsigned SC = K / T::GROUP_K;
    if constexpr (Short)
        if (SC > T::SCALE_COLS) return;
    const int stride_a = T::CONTIGUOUS ? K : args.stride_a;
    const int stride_b = T::CONTIGUOUS ? K : args.stride_b;
    const int stride_c = args.stride_c;

    static_assert((BK == 128 || BK == 256) && T::SCALE_COLS % KS == 0);
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
    const int row = M <= BM ? 0 : (tile / gn) * BM;
    const int col = M <= BM ? tile * BN : (tile % gn) * BN;
    const int valid_rows = M % BM == 0 ? BM : (M <= BM ? M : (row == (gm - 1) * BM ? M % BM : BM));

    constexpr int F = KS * NR;
    constexpr int MAX_ROWS = BM;
    constexpr int C_ROWS = output_rows(BM, BN, T::PIPELINE_BYTES);
    constexpr int PERIOD = []() constexpr {
        if constexpr (ROTATE_RING) return 1;
        int a = BS, b = F;
        while (b) { int r = a % b; a = b; b = r; }
        const int period = BS / a;
        if constexpr (NARROW && (!Short || T::PIN_C)) {
            a = NB; b = period;
            while (b) { int r = a % b; a = b; b = r; }
            const int cycle = NB / a * period;
            // Align both rings when this keeps the unrolled matrix work bounded.
            // The number of groups is still calculated from runtime K.
            if (cycle * F * MR <= 256) return cycle;
        }
        return Short || NARROW ? period : (period == 1 ? 2 : period);
    }();
    constexpr int SCALE_PHASE_DIVISOR = []() constexpr {
        int a = PERIOD, b = SCALE_K_TILES;
        while (b) { int r = a % b; a = b; b = r; }
        return a;
    }();

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
    auto* arena = b_pipeline_arena<ArenaTraits>();
    auto s_a = make_smem(reinterpret_cast<fp8_t*>(arena));
    auto s_scale = make_smem(arena);
    auto s_c = make_smem(reinterpret_cast<bf16_t*>(arena));

    // MMA accumulators and the B register ring.
    using Base = decltype(make_mfma<fp8_t, fp8_t, fp32_t>(16_I, 16_I, 128_I, mfma_adaptor_swap_ab{}));
    array<typename Base::vtype_c, MR * NR> r_c{};
    array<typename Base::vtype_b, BS> r_b;

    auto issue_b = [&](auto slot_i, auto fragment_i, int kt, auto check_i) {
        constexpr int set = decltype(slot_i)::value;
        constexpr int fragment = decltype(fragment_i)::value;
        constexpr bool checked = decltype(check_i)::value;
        // Keep vector offsets and request counts unchanged through the tail.
        // Small memory-bound short tiles zero unused B requests through bounds.
        const bool invalid = checked && kt >= P;
        const int offset = invalid ? 0 : kt * (BK * T::W_N);
        auto g_request = make_gmem(reinterpret_cast<const fp8_t*>(args.ptr_b) + col * stride_b,
            Schedule::ZERO_UNUSED_B_PREFETCH && invalid
                ? 0u : static_cast<unsigned>(BN * stride_b));
        set_slice(r_b[set],
            load<T::VEC_B>(g_request, u_rb.offsets[fragment * T::B_CHUNKS_PER_FRAGMENT],
                offset, number<T::B_CACHE>{}),
            0_I, number<T::VEC_B>{});
        set_slice(r_b[set],
            load<T::VEC_B>(g_request, u_rb.offsets[fragment * T::B_CHUNKS_PER_FRAGMENT + 1],
                offset, number<T::B_CACHE>{}),
            number<T::VEC_B>{}, number<T::B_CHUNKS_PER_FRAGMENT * T::VEC_B>{});
    };
    auto issue_a = [&](int slot, int kt, bool checked = false) {
        __builtin_amdgcn_sched_barrier(0);
        const int offset = checked && kt >= P ? valid_rows * stride_a : kt * BK;
        static_for<T::A_COPIES>([&](auto ii) {
            constexpr int i = decltype(ii)::value;
            async_load<16>(g_a, arena + slot * T::STAGE_BYTES + u_sa.offsets[i],
                u_ga.offsets[i] + (checked ? offset : 0), checked ? 0 : offset,
                0_I, number<T::CACHECTL_A>{});
        });
        __builtin_amdgcn_sched_barrier(0);
    };

    vector_t<unsigned char, T::VEC_SCALE_A> sfa_panel_raw[T::SFA_PASSES_PER_CACHE_PANEL];
    auto issue_scale_panel = [&](int begin) {
        static_for<T::SFA_PASSES_PER_CACHE_PANEL>([&](auto pass_i) {
            constexpr int pass = decltype(pass_i)::value;
            const int lane_k = lane_id / T::SFA_M_VECTORS;
            const int scale_col = begin + pass * T::SFA_K_COLUMNS_PER_PASS
                + wave_id_n * T::SFA_K_COLUMNS_PER_WAVE + lane_k;
            const int scale_row = (lane_id % T::SFA_M_VECTORS) * T::VEC_SCALE_A;
            const bool valid = lane_k < T::SFA_K_COLUMNS_PER_WAVE && scale_col < SC
                && scale_col < begin + T::SCALE_COLS
                && scale_row < BM && row + scale_row < M;
            const unsigned offset = valid
                ? scale_col * args.stride_sfa + row + scale_row
                : (SC - 1) * args.stride_sfa + M;
            sfa_panel_raw[pass] = load<T::VEC_SCALE_A>(g_sfa, offset);
        });
        auto* sbdst = arena + T::OFF_SB + u_ssfb(wave_id, 0, 0);
        if constexpr (T::SB_SLOTS) {
            const int index = thread_id % (T::SB_ROWS * T::SCALE_COLS);
            const int sr = index / T::SCALE_COLS;
            const int scale_col = begin + index % T::SCALE_COLS;
            const unsigned offset = scale_col < SC
                ? (col / 128 + sr) * args.stride_sfb + scale_col
                : (N / 128 - 1) * args.stride_sfb + SC;
            async_load<1>(g_sfb, sbdst, offset);
        } else {
            const int index = thread_id % (T::SB_ROWS * (T::SCALE_COLS / 4));
            const int sr = index / (T::SCALE_COLS / 4);
            const int first = begin + index % (T::SCALE_COLS / 4) * 4;
            if (first + 3 < SC)
                async_load<4>(g_sfb, sbdst, u_gsfb(col / 128 + sr, first));
            else if (first < SC) {
                vector_t<unsigned char, 4> bytes{};
                static_for<4>([&](auto byte_i) {
                    constexpr int byte = decltype(byte_i)::value;
                    if (first + byte < SC)
                        bytes[byte] = load<1>(g_sfb,
                            u_gsfb(col / 128 + sr, first + byte))[0];
                });
                store<4>(s_scale, bytes, T::OFF_SB + u_ssfb(wave_id, lane_id, 0));
            }
        }
    };

    // The same 4x4 byte transpose as 4wave; TM=1 packs M,M+16,M+32,M+48.
    auto publish_sfa_panel = [&](int begin) {
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
            if (lane_k < T::SFA_K_COLUMNS_PER_WAVE && scale_col < T::SCALE_COLS
                && begin + scale_col < SC) {
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
    auto consume_panel = [&](unsigned t, auto phase_i, auto tail_i) {
        constexpr int phase = decltype(phase_i)::value;
        constexpr bool tail = decltype(tail_i)::value != 0;
        constexpr bool last_panel = decltype(tail_i)::value == 2;
        const unsigned remaining = last_panel ? 1 : P - t;
        const unsigned scale_tile = Short || SingleCache ? t : t % SCALE_K_TILES;
        // Fixed ring phases cannot all cross a scale-panel boundary.
        if constexpr (!Short && !SingleCache && (ROTATE_RING || phase % SCALE_PHASE_DIVISOR == 0)) {
            if (t && scale_tile == 0) {
                s_waitcnt_lgkmcnt(0_I);
                opus::sync_threads();
                issue_scale_panel(t * KS);
                publish_sfa_panel(t * KS);
                s_waitcnt_vmcnt(0_I);
                s_waitcnt_lgkmcnt(0_I);
                opus::sync_threads();
            }
        }
        // Equal wait counts share one path; other drain states use a short
        // decision tree without duplicating the MFMA body.
        auto wait_for = [&](auto count) {
            constexpr int steady = decltype(count(number<INITIAL_A + 1>{}))::value;
            constexpr int first = decltype(count(number<1>{}))::value;
            constexpr int last = decltype(count(number<INITIAL_A>{}))::value;
            if constexpr (last_panel) {
                s_waitcnt_vmcnt(number<(first > 63 ? 63 : first)>{});
            } else if constexpr ((Short && T::PIN_C) || !tail || (steady == first && first == last)) {
                s_waitcnt_vmcnt(number<(steady > 63 ? 63 : steady)>{});
            } else {
                if (remaining > INITIAL_A)
                    s_waitcnt_vmcnt(number<(steady > 63 ? 63 : steady)>{});
                else {
                    auto drain = [&]<int Lo, int Hi>(auto&& self) {
                        constexpr int low = decltype(count(number<Lo>{}))::value;
                        constexpr int high = decltype(count(number<Hi>{}))::value;
                        if constexpr (low == high) {
                            s_waitcnt_vmcnt(number<(low > 63 ? 63 : low)>{});
                        } else {
                            constexpr int middle = (Lo + Hi) / 2;
                            if (remaining <= middle)
                                self.template operator()<Lo, middle>(self);
                            else
                                self.template operator()<middle + 1, Hi>(self);
                        }
                    };
                    drain.template operator()<1, INITIAL_A>(drain);
                }
            }
        };
        const int slot = !ROTATE_RING && PERIOD % NB == 0 ? phase % NB : t % NB;
        __builtin_amdgcn_sched_barrier(0);
        wait_for([](auto rest_i) {
            constexpr int panels = phase + decltype(rest_i)::value;
            constexpr int issued_b = phase * F + BD;
            constexpr int before_a = BD + (phase - INITIAL_A) * F + 1;
            constexpr int newer_b_a = phase < INITIAL_A ? issued_b
                : (issued_b > before_a ? issued_b - before_a : 0);
            constexpr int initial_a = panels < INITIAL_A ? panels : INITIAL_A;
            constexpr int newer_a = phase < INITIAL_A ? initial_a - phase - 1
                : (panels - phase - 1 < INITIAL_A - 1 ? panels - phase - 1 : INITIAL_A - 1);
            return number<newer_b_a * 2 + newer_a * T::A_COPIES>{};
        });
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
                    T::OFF_SB + u_rsfb(ks, group)
                        + scale_tile * KS * (T::SB_SLOTS ? 4 : 1))[0];
                r_sfb[ks] |= scale << (group * 8);
            });
            static_for<T::SFA_PACKED_M_GROUPS>([&](auto pack_i) {
                constexpr int pack = decltype(pack_i)::value;
                r_sfa[ks][pack] = __builtin_bit_cast(unsigned, load<T::VEC_SF>(s_scale,
                    T::OFF_SA + u_rsfa.offsets[ks * T::SFA_PACKED_M_GROUPS + pack]
                        + scale_tile * KS * T::SFA_PANEL_PITCH));
            });
        });
        auto read_a_fragment = [&](auto ki, auto mi) {
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
        };
        auto read_a = [&](auto ki, auto fragment_m) {
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
                if constexpr (decltype(fragment_m)::value < 0 || decltype(fragment_m)::value == m) {
                if constexpr (T::PIN_A) {
                    array<int, 2> offsets;
                    offsets[0] = lo;
                    offsets[1] = hi;
                    load_operand_fragment_pinned<16, 0>(number<(ks * MR + m) * 8>{}, r_a[STREAM_A ? 0 : ks][m], s_a, offsets);
                } else {
                    set_slice(r_a[STREAM_A ? 0 : ks][m], load<16>(s_a, lo), 0_I, 16_I);
                    set_slice(r_a[STREAM_A ? 0 : ks][m], load<16>(s_a, hi), 16_I, 32_I);
                }
                }
            });
        };
        if constexpr (PARTIAL_A)
            static_for<A_READ_LEAD>([&](auto mi) { read_a_fragment(number<0>{}, mi); });
        else if constexpr (STAGED_A)
            read_a(number<0>{}, number<-1>{});
        else if constexpr (!STREAM_A)
            static_for<KS>([&](auto ki) { read_a(ki, number<-1>{}); });
        static_for<KS * NR>([&](auto fragment_i) {
            constexpr int fragment = decltype(fragment_i)::value;
            constexpr int ks = fragment / NR;
            constexpr int n = fragment % NR;
            constexpr int absolute = phase * F + fragment;
            if constexpr (STREAM_A && n == 0)
                read_a(number<ks>{}, number<-1>{});
            if constexpr (!last_panel || fragment + BD < F)
                issue_b(number<((ROTATE_RING ? fragment : absolute) + BD) % BS>{},
                    number<(fragment + BD) % F>{}, t + (fragment + BD) / F,
                    number<(!last_panel && (tail || BD > INITIAL_A * F))>{});
            __builtin_amdgcn_sched_barrier(0);
            if constexpr (Short && NARROW && !T::PIN_C && !last_panel)
                s_waitcnt_vmcnt(number<(BD * 2 > 63 ? 63 : BD * 2)>{});
            else wait_for([](auto rest_i) {
                constexpr int panels = phase + decltype(rest_i)::value;
                constexpr int newer_b = last_panel && F - fragment - 1 < BD
                    ? F - fragment - 1 : BD;
                constexpr int first_a = absolute < BD ? 0 : (absolute - BD + F - 1) / F;
                constexpr int last_a = phase - (fragment == 0 ? 1 : 0) < panels - INITIAL_A - 1
                    ? phase - (fragment == 0 ? 1 : 0) : panels - INITIAL_A - 1;
                constexpr int pending_a = last_a >= first_a ? (last_a - first_a + 1) * T::A_COPIES : 0;
                return number<newer_b * 2 + pending_a>{};
            });

            if constexpr (!last_panel && fragment == 0 && PublishM < 0)
                if ((Short && T::PIN_C) || !tail || remaining > INITIAL_A)
                    issue_a(!ROTATE_RING && PERIOD % NB == 0 ? (phase + NB - 1) % NB
                        : (t + NB - 1) % NB, t + NB - 1, Short && T::PIN_C && tail);

            auto compute_b = [&](auto bank_i) {
                static_for<MR>([&](auto mi) {
                    constexpr int m = decltype(mi)::value;
                    constexpr int ci = m * NR + n;
                    if constexpr (T::PIN_C) {
                        [[clang::amdgpu_pin_agpr(128 + ci * 4)]] r_c[ci] = r_c[ci];
                        [[clang::amdgpu_pin_agpr(128 + ci * 4)]]
                        r_c[ci] = Base{}(r_a[STREAM_A ? 0 : ks][m], r_b[decltype(bank_i)::value], r_c[ci],
                            static_cast<int>(r_sfa[ks][m / T::VEC_SF]), static_cast<int>(r_sfb[ks]), number<m % T::VEC_SF>{}, number<n / 8>{});
                    } else {
                        r_c[ci] = Base{}(r_a[STREAM_A ? 0 : ks][m], r_b[decltype(bank_i)::value], r_c[ci],
                            static_cast<int>(r_sfa[ks][m / T::VEC_SF]), static_cast<int>(r_sfb[ks]), number<m % T::VEC_SF>{}, number<n / 8>{});
                    }
                    if constexpr (PARTIAL_A && fragment == 0 && m + A_READ_LEAD < MR) {
                        __builtin_amdgcn_sched_barrier(0);
                        read_a_fragment(number<ks>{}, number<m + A_READ_LEAD>{});
                        __builtin_amdgcn_sched_barrier(0);
                    }
                    if constexpr (STAGED_A && fragment == 0) {
                        __builtin_amdgcn_sched_barrier(0);
                        read_a(number<1>{}, mi);
                        __builtin_amdgcn_sched_barrier(0);
                    }
                    if constexpr (!last_panel && fragment == 0 && m == PUBLISH_A)
                        if ((Short && T::PIN_C) || !tail || remaining > INITIAL_A)
                            issue_a(!ROTATE_RING && PERIOD % NB == 0 ? (phase + NB - 1) % NB
                        : (t + NB - 1) % NB, t + NB - 1, Short && T::PIN_C && tail);
                });
            };
            compute_b(number<(ROTATE_RING ? fragment : absolute) % BS>{});
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
    // Keep the B queue full, suppressing memory traffic beyond K via bounds.
    // Bound startup requests in the descriptor, then stop new A copies at P.
    issue_scale_panel(0);
    static_for<INITIAL_A>([&](auto i) {
        constexpr int t = decltype(i)::value;
        issue_a(t, t, true);
    });
    static_for<BD>([&](auto i) {
        constexpr int fragment = decltype(i)::value;
        issue_b(number<fragment % BS>{}, number<fragment % F>{}, fragment / F, number<1>{});
    });
    publish_sfa_panel(0);
    // ===== Main loop =====
    if constexpr (Short && T::PIN_C && PERIOD == 1 && INITIAL_A == 1) {
        consume_panel(0, number<0>{}, number<1>{});
        if (P > 1) {
#pragma nounroll
            for (unsigned t = 1; t + 1 < P; ++t)
                consume_panel(t, number<1>{}, number<0>{});
            consume_panel(P - 1, number<1>{}, number<2>{});
        }
    } else if constexpr (Short && !T::PIN_C && PERIOD == 1 && INITIAL_A == 1) {
        if constexpr (T::PIN_A && KS == 1 && NR >= 4) {
            consume_panel(0, number<0>{}, number<0>{});
            if (P > 1) {
#pragma nounroll
                for (unsigned t = 1; t + 1 < P; ++t)
                    consume_panel(t, number<1>{}, number<0>{});
                consume_panel(P - 1, number<1>{}, number<2>{});
            } else {
                s_waitcnt_vmcnt(0_I);
            }
        } else {
        if (P == 1) {
            consume_panel(0, number<0>{}, number<2>{});
        } else {
            consume_panel(0, number<0>{}, number<0>{});
#pragma nounroll
            for (unsigned t = 1; t + 1 < P; ++t)
                consume_panel(t, number<1>{}, number<0>{});
            consume_panel(P - 1, number<1>{}, number<2>{});
        }
        }
    } else if constexpr (Short && !T::PIN_C) {
        static_for<INITIAL_A>([&](auto i) {
            constexpr int t = decltype(i)::value;
            if (t < P) consume_panel(t, number<t>{}, number<1>{});
        });
#pragma nounroll
        for (unsigned t = INITIAL_A; t < P; t += PERIOD) {
            static_for<PERIOD>([&](auto i) {
                constexpr int step = decltype(i)::value;
                if (step == 0 || t + step < P)
                    consume_panel(t + step, number<INITIAL_A + step>{}, number<1>{});
            });
        }
    } else {
        static_for<INITIAL_A>([&](auto i) {
            constexpr int t = decltype(i)::value;
            if (t < P) consume_panel(t, number<t>{}, number<1>{});
        });
        // Only the B register-slot cycle is unrolled; A LDS stages advance at runtime.
        const unsigned start = P < INITIAL_A ? P : INITIAL_A;
        const unsigned groups = P > 2 * INITIAL_A
            ? (P - 2 * INITIAL_A) / PERIOD : 0;
#pragma nounroll
        for (unsigned group = 0; group < groups; ++group) {
            static_for<PERIOD>([&](auto i) {
                constexpr int phase = INITIAL_A + decltype(i)::value;
                consume_panel(start + group * PERIOD + decltype(i)::value,
                    number<phase>{}, number<0>{});
            });
        }
        const unsigned k_tile = start + groups * PERIOD;
        // ===== Epilogue =====
        // Drain the remaining panels before output writeback.
        auto drain = [&]<int Step>(auto&& self) {
            const unsigned t = k_tile + Step;
            if (t >= P) return;
            if constexpr ((LATE_A && T::STREAM_A) || (THREE_A && (WaveColumns == 4 || (WaveColumns == 2 && PublishM == 0))) || STAGED_A
                || (!Short && !STREAM_A && !T::PIN_A && !T::PIN_C && T::OUTPUT == 0
                    && BK == 256 && MR == 6 && NR == 1 && BN == 128
                    && T::STAGES == 3 && WaveColumns == 8 && PublishM == 3)) {
                if (t + 1 == P)
                    consume_panel(t, number<INITIAL_A + Step>{}, number<2>{});
                else
                    consume_panel(t, number<INITIAL_A + Step>{}, number<1>{});
            } else
                consume_panel(t, number<INITIAL_A + Step>{}, number<1>{});
            if constexpr (Step + 1 < PERIOD + INITIAL_A)
                self.template operator()<Step + 1>(self);
        };
        drain.template operator()<0>(drain);

    }

    constexpr int C_STORE_CACHE = Schedule::C_STORE_CACHE;
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
            if constexpr (Short && T::PIN_A && !T::PIN_C && KS == 1
                && (NR >= 4 || ((MR == 4 || MR == 8) && NR == 2 && T::STAGES == 4))) {
                constexpr int OUTPUT_GROUP = NR == 2 ? 2 : 8;
                static_for<(CHUNKS + OUTPUT_GROUP - 1) / OUTPUT_GROUP>([&](auto group_i) {
                    constexpr int begin = decltype(group_i)::value * OUTPUT_GROUP;
                    constexpr int copies = CHUNKS - begin < OUTPUT_GROUP ? CHUNKS - begin : OUTPUT_GROUP;
                    s_waitcnt_lgkmcnt(number<CHUNKS - begin - copies>{});
                    static_for<copies>([&](auto copy_i) {
                        constexpr int i = begin + decltype(copy_i)::value;
                        const int index = i * T::BLOCK_SIZE + thread_id;
                        const int rr = index / (BN / 8);
                        if (rr < PART_ROWS && part * C_ROWS + rr < valid_rows)
                            store<8>(g_c, output[i], part * C_ROWS * stride_c + u_gc_copy.offsets[i], 0, number<C_STORE_CACHE>{});
                    });
                });
            } else {
            s_waitcnt_lgkmcnt(0_I);
            static_for<CHUNKS>([&](auto ii) {
                constexpr int i = decltype(ii)::value;
                const int index = i * T::BLOCK_SIZE + thread_id;
                const int rr = index / (BN / 8);
                if (rr < PART_ROWS && part * C_ROWS + rr < valid_rows)
                    store<8>(g_c, output[i], part * C_ROWS * stride_c + u_gc_copy.offsets[i], 0, number<C_STORE_CACHE>{});
            });
            }
        });
    }
    s_waitcnt_vmcnt(0_I);
}

template<class Traits>
__device__ __forceinline__ void bulk_panel_overlap_body(opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 args) {
    using Schedule = opus_gemm_bpreshuffle_b_schedule<Traits, Traits::PUBLISH_M, false>;
    if constexpr (Schedule::OVERLAP_SCALE_CACHE) {
        // The total K and loop count remain runtime; this range needs no rollover.
        if (args.k / Traits::GROUP_K <= Traits::SCALE_COLS)
            bulk_panel_body<Traits, Traits::PUBLISH_M, false, true>(args);
        else
            bulk_panel_body<Traits, Traits::PUBLISH_M>(args);
    } else
        bulk_panel_body<Traits, Traits::PUBLISH_M>(args);
}

} // namespace opus_gemm_prefetch_b_vgpr_detail
#endif

template<class Traits, bool Short = false>
__global__ __launch_bounds__(Traits::BLOCK_SIZE, Traits::WG_PER_CU)
void opus_gemm_blockscale_mxscale_bpreshuffle_prefetch_b_vgpr_bulk_panel_kernel(
    opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 args) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx950__)
    using T = std::conditional_t<Short, opus_gemm_bpreshuffle_b_short_traits<Traits>, Traits>;
    opus_gemm_prefetch_b_vgpr_detail::bulk_panel_body<T, -1, Short>(args);
#endif
}

template<class Traits, bool Short = false>
__global__ __launch_bounds__(Traits::BLOCK_SIZE, Traits::WG_PER_CU)
void opus_gemm_blockscale_mxscale_bpreshuffle_prefetch_b_vgpr_overlap_bulk_panel_kernel(
    opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 args) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx950__)
    using T = std::conditional_t<Short, opus_gemm_bpreshuffle_b_short_traits<Traits>, Traits>;
    if constexpr (Short)
        opus_gemm_prefetch_b_vgpr_detail::bulk_panel_body<T, Traits::PUBLISH_M, Short>(args);
    else
        opus_gemm_prefetch_b_vgpr_detail::bulk_panel_overlap_body<T>(args);
#endif
}
