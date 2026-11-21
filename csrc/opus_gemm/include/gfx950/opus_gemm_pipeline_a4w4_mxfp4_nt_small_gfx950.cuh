// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// MXFP4 x MXFP4 NT 1x4-wave small-tile pipeline, plain row-major inputs (A / B / scales).  A / B: STAGES-deep LDS-DMA
// ring; scales: one K-deep LDS panel per ring cycle.  KS > 1 splits K over the waves; split-K via splitk_reduce_kernel.
#pragma once
#include "opus_gemm_traits_a4w4_mxfp4_nt_small_gfx950.cuh"

namespace opus_mxfp4_small {
using opus::operator""_I;
using fp4x32_t = opus::vector_t<opus::i32_t, 4>;   // one fp4 MFMA operand: 32 fp4 = 16 bytes per lane
using i32x8_t = opus::vector_t<opus::i32_t, 8>;
using sf8_t = opus::vector_t<opus::u32_t, 2>;      // the 8 scale bytes of one row and K tile

// ===========================================================================
// A / B: global -> LDS (LDS-DMA) -> VGPR
// ===========================================================================
// Create layout for loading A matrix from global memory (wave w: NI chunks of 8 rows x 128 B of K tile w / WQ; 16 B
// pieces XOR-swizzled per row pair).
template<class T>
__device__ inline auto make_layout_ga(int lane_id, int wave_id, int stride_a) {
    constexpr int threads_k = T::B_K_BYTES / T::VEC_A;
    constexpr int NI = T::a_buffer_load_insts, WQ = T::A_CHUNKS / NI;

    constexpr auto ga_block_shape = opus::make_tuple(
        opus::number<T::KS>{},
        opus::number<WQ>{},
        opus::number<NI>{},
        opus::number<T::smem_sub>{},
        opus::number<threads_k>{},
        opus::number<T::VEC_A>{});

    constexpr auto ga_block_dim = opus::make_tuple(
        opus::make_tuple(opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}));

    const int lane_id_r = lane_id / threads_k;

    return opus::make_layout<T::VEC_A>(
        ga_block_shape,
        opus::unfold_x_stride(ga_block_dim, ga_block_shape, opus::tuple{opus::number<T::B_K_BYTES>{}, stride_a, 1_I}),
        opus::unfold_p_coord(ga_block_dim, opus::tuple{wave_id / WQ, wave_id % WQ, lane_id_r, (lane_id % threads_k) ^ ((lane_id_r >> 1) & 3)}));
}

// Create layout for storing A matrix to shared memory (chunk c at c * CHUNK_BYTES).
template<class T>
__device__ inline auto make_layout_sa(int wave_id) {
    constexpr int NI = T::a_buffer_load_insts, WQ = T::A_CHUNKS / NI;

    constexpr auto sa_block_shape = opus::make_tuple(
        opus::number<T::KS>{},
        opus::number<WQ>{},
        opus::number<NI>{},
        opus::number<T::VEC_A>{});

    constexpr auto sa_block_dim = opus::make_tuple(
        opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::y_dim{}),
        opus::make_tuple(opus::y_dim{}));

    return opus::make_layout(
        sa_block_shape,
        opus::unfold_x_stride(sa_block_dim, sa_block_shape, opus::tuple{opus::number<T::CHUNK_BYTES>{}, 1_I}),
        opus::unfold_p_coord(sa_block_dim, opus::tuple{wave_id / WQ, wave_id % WQ}));
}

// Create layout for reading A matrix from shared memory to registers (offsets[e * E_K + kk], K tile wave_id_k).
template<class T>
__device__ inline auto make_layout_ra(int lane_id, int wave_id_k) {
    constexpr auto ra_block_shape = opus::make_tuple(
        opus::number<T::T_K>{},
        opus::number<T::T_M>{},
        opus::number<T::E_M>{},
        opus::number<T::W_M / T::smem_sub>{},
        opus::number<T::smem_sub>{},
        opus::number<T::E_K>{},
        opus::number<T::WARP_SIZE / T::W_M>{},
        opus::number<T::VEC_A>{});

    constexpr auto ra_block_dim = opus::make_tuple(
        opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::y_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}, opus::p_dim{}, opus::y_dim{}));

    const int lane_id_m = lane_id % T::W_M;
    const int r = lane_id_m % T::smem_sub;

    return opus::make_layout<T::VEC_A>(
        ra_block_shape,
        opus::unfold_x_stride(ra_block_dim, ra_block_shape, opus::tuple{opus::number<T::CHUNK_BYTES>{}, 1_I}),
        opus::unfold_p_coord(ra_block_dim, opus::tuple{wave_id_k, 0, lane_id_m / T::smem_sub, r, (lane_id / T::W_M) ^ ((r >> 1) & 3)}));
}

// Create layout for loading B matrix from global memory (as A).
template<class T>
__device__ inline auto make_layout_gb(int lane_id, int wave_id, int stride_b) {
    constexpr int threads_k = T::B_K_BYTES / T::VEC_B;
    constexpr int NI = T::b_buffer_load_insts, WQ = T::B_CHUNKS / NI;

    constexpr auto gb_block_shape = opus::make_tuple(
        opus::number<T::KS>{},
        opus::number<WQ>{},
        opus::number<NI>{},
        opus::number<T::smem_sub>{},
        opus::number<threads_k>{},
        opus::number<T::VEC_B>{});

    constexpr auto gb_block_dim = opus::make_tuple(
        opus::make_tuple(opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}));

    const int lane_id_r = lane_id / threads_k;

    return opus::make_layout<T::VEC_B>(
        gb_block_shape,
        opus::unfold_x_stride(gb_block_dim, gb_block_shape, opus::tuple{opus::number<T::B_K_BYTES>{}, stride_b, 1_I}),
        opus::unfold_p_coord(gb_block_dim, opus::tuple{wave_id / WQ, wave_id % WQ, lane_id_r, (lane_id % threads_k) ^ ((lane_id_r >> 1) & 3)}));
}

// Create layout for storing B matrix to shared memory (as A).
template<class T>
__device__ inline auto make_layout_sb(int wave_id) {
    constexpr int NI = T::b_buffer_load_insts, WQ = T::B_CHUNKS / NI;

    constexpr auto sb_block_shape = opus::make_tuple(
        opus::number<T::KS>{},
        opus::number<WQ>{},
        opus::number<NI>{},
        opus::number<T::VEC_B>{});

    constexpr auto sb_block_dim = opus::make_tuple(
        opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::y_dim{}),
        opus::make_tuple(opus::y_dim{}));

    return opus::make_layout(
        sb_block_shape,
        opus::unfold_x_stride(sb_block_dim, sb_block_shape, opus::tuple{opus::number<T::CHUNK_BYTES>{}, 1_I}),
        opus::unfold_p_coord(sb_block_dim, opus::tuple{wave_id / WQ, wave_id % WQ}));
}

// Create layout for reading B matrix from shared memory to registers (offsets[n * E_K + kk]).
template<class T>
__device__ inline auto make_layout_rb(int lane_id, int wave_id_n, int wave_id_k) {
    constexpr auto rb_block_shape = opus::make_tuple(
        opus::number<T::T_K>{},
        opus::number<T::T_N>{},
        opus::number<T::E_N>{},
        opus::number<T::W_N / T::smem_sub>{},
        opus::number<T::smem_sub>{},
        opus::number<T::E_K>{},
        opus::number<T::WARP_SIZE / T::W_N>{},
        opus::number<T::VEC_B>{});

    constexpr auto rb_block_dim = opus::make_tuple(
        opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::y_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}, opus::p_dim{}, opus::y_dim{}));

    const int lane_id_n = lane_id % T::W_N;
    const int r = lane_id_n % T::smem_sub;

    return opus::make_layout<T::VEC_B>(
        rb_block_shape,
        opus::unfold_x_stride(rb_block_dim, rb_block_shape, opus::tuple{opus::number<T::CHUNK_BYTES>{}, 1_I}),
        opus::unfold_p_coord(rb_block_dim, opus::tuple{wave_id_k, wave_id_n, lane_id_n / T::smem_sub, r, (lane_id / T::W_N) ^ ((r >> 1) & 3)}));
}

// ===========================================================================
// Scales: global -> LDS (K-deep panel per ring cycle) -> VGPR
// ===========================================================================
// Create layout for loading scale factors from global memory (wave w: NI copies of SF_PANEL_ROWS rows x SF_ROW_BYTES;
// lands as the row-major [row][SF_ROW_BYTES] panel).
template<class T, int ROWS>
__device__ inline auto make_layout_gsf(int lane_id, int wave_id, int stride_sf) {
    constexpr int lanes_k = T::SF_ROW_BYTES / T::VEC_SF_DMA;
    constexpr int NI = ROWS / T::SF_PANEL_ROWS / T::NUM_WAVES;

    constexpr auto gsf_block_shape = opus::make_tuple(
        opus::number<T::NUM_WAVES>{},
        opus::number<NI>{},
        opus::number<T::SF_PANEL_ROWS>{},
        opus::number<lanes_k>{},
        opus::number<T::VEC_SF_DMA>{});

    constexpr auto gsf_block_dim = opus::make_tuple(
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}));

    return opus::make_layout<T::VEC_SF_DMA>(
        gsf_block_shape,
        opus::unfold_x_stride(gsf_block_dim, gsf_block_shape, opus::tuple{stride_sf, 1_I}),
        opus::unfold_p_coord(gsf_block_dim, opus::tuple{wave_id, lane_id / lanes_k, lane_id % lanes_k}));
}
template<class T> __device__ inline auto make_layout_gsfa(int lane_id, int wave_id, int stride_sfa) { return make_layout_gsf<T, T::B_M>(lane_id, wave_id, stride_sfa); }
template<class T> __device__ inline auto make_layout_gsfb(int lane_id, int wave_id, int stride_sfb) { return make_layout_gsf<T, T::B_N>(lane_id, wave_id, stride_sfb); }

// Create layout for storing scale factors to shared memory (copy d at d * SF_DMA_BYTES).
template<class T, int ROWS>
__device__ inline auto make_layout_ssf(int wave_id) {
    constexpr int NI = ROWS / T::SF_PANEL_ROWS / T::NUM_WAVES;

    constexpr auto ssf_block_shape = opus::make_tuple(
        opus::number<T::NUM_WAVES>{},
        opus::number<NI>{},
        opus::number<T::VEC_SF_DMA>{});

    constexpr auto ssf_block_dim = opus::make_tuple(
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}),
        opus::make_tuple(opus::y_dim{}));

    return opus::make_layout(
        ssf_block_shape,
        opus::unfold_x_stride(ssf_block_dim, ssf_block_shape, opus::tuple{opus::number<T::SF_DMA_BYTES>{}, 1_I}),
        opus::unfold_p_coord(ssf_block_dim, opus::tuple{wave_id}));
}
template<class T> __device__ inline auto make_layout_ssfa(int wave_id) { return make_layout_ssf<T, T::B_M>(wave_id); }
template<class T> __device__ inline auto make_layout_ssfb(int wave_id) { return make_layout_ssf<T, T::B_N>(wave_id); }

// Create layout for reading scales from shared memory: the 8 bytes of row 16 (w, e) + lane % 16, K tile wave_id_k.
template<class T, int W, int E>
__device__ inline auto make_layout_rsf(int lane_id, int wave_id, int wave_id_k) {
    constexpr auto rsf_block_shape = opus::make_tuple(
        opus::number<T::T_K>{},
        opus::number<W>{},
        opus::number<E>{},
        opus::number<T::W_M>{},
        opus::number<T::VEC_SF>{});

    constexpr auto rsf_block_dim = opus::make_tuple(
        opus::make_tuple(opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}));

    return opus::make_layout<T::VEC_SF>(
        rsf_block_shape,
        opus::unfold_x_stride(rsf_block_dim, rsf_block_shape,
                              opus::tuple{opus::number<T::SF_K_TILE_BYTES>{}, opus::number<T::SF_ROW_BYTES>{}, 1_I}),
        opus::unfold_p_coord(rsf_block_dim, opus::tuple{wave_id_k, wave_id, lane_id % T::W_M}));
}
template<class T> __device__ inline auto make_layout_rsfa(int lane_id, int wave_id_k) { return make_layout_rsf<T, T::T_M, T::E_M>(lane_id, 0, wave_id_k); }
template<class T> __device__ inline auto make_layout_rsfb(int lane_id, int wave_id_n, int wave_id_k) { return make_layout_rsf<T, T::T_N, T::E_N>(lane_id, wave_id_n, wave_id_k); }

// v_perm selector: scale bytes g and K_HALF + g of the row (g = lane / 16) -> bytes 0 / 1 (opsel kk), 0 above.
template<class T>
__device__ __forceinline__ unsigned sf_perm_sel(int lane_id) {
    constexpr unsigned K_HALF = T::SF_K_TILE_BYTES / T::E_K;   // 4 scale bytes per kk
    const unsigned g = lane_id / T::W_M;
    return g | ((K_HALF + g) << 8) | 0x0c0c0000u;
}
__device__ __forceinline__ unsigned sf_pack(const sf8_t& v, unsigned sel) { return __builtin_amdgcn_perm(v[1], v[0], sel); }

// ===========================================================================
// C
// ===========================================================================
// Create layout for storing C to global memory (offsets[e * E_N + n], elements; also the split-K partial tile).
template<class T>
__device__ inline auto make_layout_gc(int lane_id, int stride_c) {
    constexpr auto gc_block_shape = opus::make_tuple(
        opus::number<T::E_M>{},
        opus::number<T::W_M>{},
        opus::number<T::E_N>{},
        opus::number<T::WARP_SIZE / T::W_M>{},
        opus::number<T::VEC_C>{});

    constexpr auto gc_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::y_dim{}));

    return opus::make_layout<T::VEC_C>(
        gc_block_shape,
        opus::unfold_x_stride(gc_block_dim, gc_block_shape, opus::tuple{stride_c, 1_I}),
        opus::unfold_p_coord(gc_block_dim, opus::tuple{lane_id % T::W_M, lane_id / T::W_M}));
}

// Like opus async_load(layout), but the per-issue LDS offsets go into the MUBUF immediate (one m0); the immediate also
// shifts the global address, so it is subtracted there.
template<int VEC, class G, class LG, class LS>
__device__ inline void async_load_imm(G& g, void* smem_base, const LG& u_g, const LS& u_s, int s_os) {
    const auto gofs = opus::layout_to_offsets<VEC>(u_g);
    const auto sofs = opus::layout_to_offsets<VEC>(u_s);
    constexpr auto imm = opus::layout_imm_offsets_v<opus::remove_cvref_t<LS>, VEC>;
    constexpr int n = static_cast<int>(sizeof(imm) / sizeof(imm[0]));
    char* const base = reinterpret_cast<char*>(smem_base) + sofs[0];
    opus::static_for<n>([&](auto i) {
        constexpr int io = imm[decltype(i)::value] - imm[0];
        static_assert(io >= 0 && io <= 4095, "LDS issue stride must fit the 12-bit MUBUF offset");
        g.template async_load<VEC>(base, gofs[decltype(i)::value] - io, s_os, opus::number<io>{});
    });
}

// sched_group_barrier masks (LLVM AMDGPU)
constexpr int SCHED_MFMA = 0x008;
constexpr int SCHED_VMEM_READ = 0x820;   // 0x020 VMEM read | 0x800 LDS DMA (newer LLVM classifies buffer_load ... lds as LDS DMA)
constexpr int SCHED_DS_READ = 0x100;

template<int MASK, int CNT = 1>
__device__ inline void sched_group() {
    if constexpr (CNT > 0) __builtin_amdgcn_sched_group_barrier(MASK, CNT, 0);
}

// Ring step schedule: DS reads first, then MFMAs with the VMEM copies interleaved.
template<class T, int DS, int VMEM>
__device__ inline void sched_step() {
    constexpr int MFMAS = T::E_K * T::E_M * T::E_N;
    sched_group<SCHED_DS_READ, DS>();
    opus::static_for<MFMAS>([&](auto m) {
        sched_group<SCHED_MFMA>();
        sched_group<SCHED_VMEM_READ, (decltype(m)::value < VMEM)>();
    });
    sched_group<SCHED_VMEM_READ, (VMEM > MFMAS ? VMEM - MFMAS : 0)>();
}

// One scaled fp4 MFMA (16 B operands widened to the 32 B builtin operand; upper half not read).
template<int OPSEL_A, int OPSEL_B, class MMA, class CF>
__device__ __forceinline__ void mma_scale_one(MMA&, const fp4x32_t& a, const fp4x32_t& b, CF& c, unsigned v_sfa, unsigned v_sfb) {
    using base_mma = typename opus::remove_cvref_t<MMA>::MMA;
    const i32x8_t a8 = __builtin_shufflevector(a, a, 0, 1, 2, 3, -1, -1, -1, -1);
    const i32x8_t b8 = __builtin_shufflevector(b, b, 0, 1, 2, 3, -1, -1, -1, -1);
    c = base_mma{}(a8, b8, c, static_cast<int>(v_sfa), static_cast<int>(v_sfb), opus::number<OPSEL_A>{}, opus::number<OPSEL_B>{});
}

// C += A x B of one ring step (scale byte kk = opsel kk).
template<class T, class MMA, class CF>
__device__ __forceinline__ void mma_step(MMA& mma, const fp4x32_t (&v_a)[T::E_K][T::E_M], const fp4x32_t (&v_b)[T::E_K][T::E_N],
                                         CF (&v_c)[T::E_M * T::E_N], const opus::u32_t (&v_sfa)[T::E_M], const opus::u32_t (&v_sfb)[T::E_N]) {
    opus::static_for<T::E_K>([&](auto kk) { opus::static_for<T::E_M>([&](auto e) { opus::static_for<T::E_N>([&](auto n) {
        constexpr int K = decltype(kk)::value, E = decltype(e)::value, N = decltype(n)::value;
        mma_scale_one<K, K>(mma, v_a[K][E], v_b[K][N], v_c[E * T::E_N + N], v_sfa[E], v_sfb[N]);
    }); }); });
}

// ===========================================================================
// Kernel
// ===========================================================================
// Cache policy of the C stores: nt (C is written once).
constexpr int C_AUX = 2;
template<class Traits>
__device__ __forceinline__ void gemm_a4w4_mxfp4_nt_small_body(const opus_gemm_mxfp4_small_kargs_gfx950& kargs) {
#if defined(__gfx950__)
    using namespace opus;

    using T = opus::remove_cvref_t<Traits>;
    using D_A = opus::fp4_t;
    using D_B = opus::fp4_t;
    using D_ACC = opus::fp32_t;
    using D_C = opus::bf16_t;
    using D_AB = opus::u8_t;     // fp4x2 bytes
    using D_SF = opus::e8m0_t;
    constexpr int STAGES = T::STAGES;

    const int lane_id = static_cast<int>(opus::lane_id());
    const int wave_id = __builtin_amdgcn_readfirstlane(thread_id_x() / T::WARP_SIZE);
    const int wave_id_n = wave_id % T::T_N;
    const int wave_id_k = wave_id / T::T_N;
    const int split_k = kargs.split_k;
    const int batch_id = block_id_z() / split_k;
    const int k_split = block_id_z() % split_k;
    const int row = block_id_y() * T::B_M;
    const int col = block_id_x() * T::B_N;                 // the block's first column (B / scale copies)
    const int col_w = col + wave_id_n * T::WAVE_N;         // this wave's first column (C)
    // S > 1: load the workspace pointer early.
    float* ws = nullptr;
    if (split_k > 1) ws = static_cast<float*>(kargs.ws_handle->ptr);
    const int k_tiles = kargs.k / T::B_K, k_per = (k_tiles + split_k - 1) / split_k;
    const int k_begin = k_split * k_per;
    const int k_end = k_tiles < k_begin + k_per ? k_tiles : k_begin + k_per;
    const int steps = k_end > k_begin ? (k_end - k_begin + T::KS - 1) / T::KS : 0;   // ring steps (KS K tiles each)

    // M / N boundary: rows past the matrix are outside the buffer range (loads 0, stores dropped); 64-bit tile base.
    const int rows_in = kargs.m - row < T::B_M ? kargs.m - row : T::B_M;
    const int cols_in = kargs.n - col < T::B_N ? kargs.n - col : T::B_N;
    auto g_a = make_gmem(reinterpret_cast<const D_AB*>(kargs.ptr_a) + batch_id * kargs.stride_a_batch + static_cast<long long>(row) * kargs.stride_a,
                         static_cast<unsigned>(rows_in * kargs.stride_a));
    auto g_b = make_gmem(reinterpret_cast<const D_AB*>(kargs.ptr_b) + batch_id * kargs.stride_b_batch + static_cast<long long>(col) * kargs.stride_b,
                         static_cast<unsigned>(cols_in * kargs.stride_b));
    auto g_c = make_gmem(reinterpret_cast<D_C*>(kargs.ptr_c) + batch_id * kargs.stride_c_batch + static_cast<long long>(row) * kargs.stride_c + col_w,
                         static_cast<unsigned>(rows_in * kargs.stride_c * sizeof(D_C)));
    auto g_sfa = make_gmem(reinterpret_cast<const D_SF*>(kargs.ptr_sfa) + batch_id * kargs.stride_sfa_batch + static_cast<long long>(row) * kargs.stride_sfa,
                           static_cast<unsigned>(rows_in * kargs.stride_sfa));
    auto g_sfb = make_gmem(reinterpret_cast<const D_SF*>(kargs.ptr_sfb) + batch_id * kargs.stride_sfb_batch + static_cast<long long>(col) * kargs.stride_sfb,
                           static_cast<unsigned>(cols_in * kargs.stride_sfb));

    const auto u_ga = make_layout_ga<T>(lane_id, wave_id, kargs.stride_a);
    const auto u_gb = make_layout_gb<T>(lane_id, wave_id, kargs.stride_b);
    const auto u_sa = make_layout_sa<T>(wave_id);
    const auto u_sb = make_layout_sb<T>(wave_id);
    const auto u_ra = make_layout_ra<T>(lane_id, wave_id_k);
    const auto u_rb = make_layout_rb<T>(lane_id, wave_id_n, wave_id_k);
    const auto u_gsfa = make_layout_gsfa<T>(lane_id, wave_id, kargs.stride_sfa);
    const auto u_gsfb = make_layout_gsfb<T>(lane_id, wave_id, kargs.stride_sfb);
    const auto u_ssfa = make_layout_ssfa<T>(wave_id);
    const auto u_ssfb = make_layout_ssfb<T>(wave_id);
    const auto u_rsfa = make_layout_rsfa<T>(lane_id, wave_id_k);
    const auto u_rsfb = make_layout_rsfb<T>(lane_id, wave_id_n, wave_id_k);

    // LDS: STAGES ring stages [A][B] (later the KS partial tiles), 2 scale panels per operand.
    __shared__ __attribute__((aligned(16))) char smem_ab[STAGES * T::STAGE_BYTES];
    char* const smem_a = smem_ab;
    char* const smem_b = smem_ab + T::STAGE_B_OFF;
    __shared__ __attribute__((aligned(16))) char smem_sfa[T::SCALE_PANEL_BUFFERS * T::SFA_PANEL_BYTES];
    __shared__ __attribute__((aligned(16))) char smem_sfb[T::SCALE_PANEL_BUFFERS * T::SFB_PANEL_BYTES];
    auto s_a = make_smem(reinterpret_cast<D_AB*>(smem_a));
    auto s_b = make_smem(reinterpret_cast<D_AB*>(smem_b));
    auto s_sfa = make_smem(reinterpret_cast<D_SF*>(smem_sfa));
    auto s_sfb = make_smem(reinterpret_cast<D_SF*>(smem_sfb));

    auto mma = make_tiled_mma<D_A, D_B, D_ACC>(
        seq<T::E_M, T::E_N, 1>{},          // E_K = 2 steps are v_a[.][kk]
        seq<T::T_M, T::T_N, T::T_K>{},
        seq<T::W_M, T::W_N, T::W_K>{},
        mfma_adaptor_swap_ab{});
    using MMA = decltype(mma);
    using CFrag = opus::vector_t<D_ACC, MMA::mma_c_len>;   // one 16x16 C fragment
    CFrag v_c[T::E_M * T::E_N];                            // [fragment]
    opus::static_for<T::E_M * T::E_N>([&](auto f) { clear(v_c[f.value]); });

    // Operands, double-buffered by step parity.
    fp4x32_t v_a[2][T::E_K][T::E_M];
    fp4x32_t v_b[2][T::E_K][T::E_N];
    sf8_t v_sfa_next[2][T::E_M];
    sf8_t v_sfb_next[2][T::E_N];
    // MFMA scale dwords (byte kk = opsel kk).
    using D_SF_PACK = opus::u32_t;
    D_SF_PACK v_sfa[T::E_M];
    D_SF_PACK v_sfb[T::E_N];

    auto ga_offset = [&](int tile_k) { return tile_k * T::B_K_BYTES; };
    auto gb_offset = [&](int tile_k) { return tile_k * T::B_K_BYTES; };
    auto gsf_offset = [&](int tile_k) { return tile_k * T::SF_K_TILE_BYTES; };
    auto sa_offset = [&](int stage) { return stage * T::STAGE_BYTES; };
    auto sb_offset = [&](int stage) { return stage * T::STAGE_BYTES; };
    auto ssfa_offset = [&](int buf) { return buf * T::SFA_PANEL_BYTES; };
    auto ssfb_offset = [&](int buf) { return buf * T::SFB_PANEL_BYTES; };

    // Global -> LDS.  K tiles past k_end read 0 through an out-of-range voffset (the bounds check skips soffset).
    constexpr int G_OOB = 0x40000000;
    const int a_t = wave_id / (T::A_CHUNKS / T::a_buffer_load_insts);
    const int b_t = wave_id / (T::B_CHUNKS / T::b_buffer_load_insts);
    auto prefetch_a = [&](int stage, int tile) {
        async_load_imm<T::VEC_A>(g_a, s_a.ptr, u_ga + (tile + a_t < k_end ? 0 : G_OOB), u_sa + sa_offset(stage), ga_offset(tile));
    };
    auto prefetch_b = [&](int stage, int tile) {
        async_load_imm<T::VEC_B>(g_b, s_b.ptr, u_gb + (tile + b_t < k_end ? 0 : G_OOB), u_sb + sb_offset(stage), gb_offset(tile));
    };
    // Scales: global -> LDS directly, one panel per ring cycle.  Unlike the big pipeline (global -> VGPR -> quad transpose
    // -> LDS), the MFMA dword packs one row's two K halves, an in-lane v_perm after the LDS read (read_sf).
    const int sf_lane_kt = (lane_id % (T::SF_ROW_BYTES / T::VEC_SF_DMA)) * T::VEC_SF_DMA / T::SF_K_TILE_BYTES;
    auto prefetch_sfa = [&](int buf, int tile) {
        async_load_imm<T::VEC_SF_DMA>(g_sfa, s_sfa.ptr, u_gsfa + (tile + sf_lane_kt < k_end ? 0 : G_OOB), u_ssfa + ssfa_offset(buf), gsf_offset(tile));
    };
    auto prefetch_sfb = [&](int buf, int tile) {
        async_load_imm<T::VEC_SF_DMA>(g_sfb, s_sfb.ptr, u_gsfb + (tile + sf_lane_kt < k_end ? 0 : G_OOB), u_ssfb + ssfb_offset(buf), gsf_offset(tile));
    };
    // All copies of a ring step (+ the scale panel at the cycle's first step).
    auto prefetch_step = [&](auto stage_, int tile, int buf) {
        prefetch_a(decltype(stage_)::value, tile);
        prefetch_b(decltype(stage_)::value, tile);
        if constexpr (decltype(stage_)::value == 0) {
            prefetch_sfa(buf, tile);
            prefetch_sfb(buf, tile);
        }
    };

    // LDS -> VGPR.
    const auto ra_offs = opus::layout_to_offsets<T::VEC_A>(u_ra);
    const auto rb_offs = opus::layout_to_offsets<T::VEC_B>(u_rb);
    const auto rsfa_offs = opus::layout_to_offsets<T::VEC_SF>(u_rsfa);
    const auto rsfb_offs = opus::layout_to_offsets<T::VEC_SF>(u_rsfb);
    auto load_a = [&](auto& sa_, auto cur_) {
        constexpr int cur = decltype(cur_)::value;
        opus::static_for<T::E_K>([&](auto kk) { opus::static_for<T::E_M>([&](auto e) {
            v_a[cur][kk.value][e.value] = __builtin_bit_cast(fp4x32_t, load<T::VEC_A>(sa_, ra_offs[e.value * T::E_K + kk.value]));
        }); });
    };
    auto load_b = [&](auto& sb_, auto cur_) {
        constexpr int cur = decltype(cur_)::value;
        opus::static_for<T::E_K>([&](auto kk) { opus::static_for<T::E_N>([&](auto n) {
            v_b[cur][kk.value][n.value] = __builtin_bit_cast(fp4x32_t, load<T::VEC_B>(sb_, rb_offs[n.value * T::E_K + kk.value]));
        }); });
    };
    auto load_sfa = [&](auto& rd_sfa_, auto cur_, int stage) {
        constexpr int cur = decltype(cur_)::value;
        opus::static_for<T::E_M>([&](auto e) {
            v_sfa_next[cur][e.value] = __builtin_bit_cast(sf8_t, load<T::VEC_SF>(rd_sfa_, rsfa_offs[e.value] + gsf_offset(stage * T::KS)));
        });
    };
    auto load_sfb = [&](auto& rd_sfb_, auto cur_, int stage) {
        constexpr int cur = decltype(cur_)::value;
        opus::static_for<T::E_N>([&](auto n) {
            v_sfb_next[cur][n.value] = __builtin_bit_cast(sf8_t, load<T::VEC_SF>(rd_sfb_, rsfb_offs[n.value] + gsf_offset(stage * T::KS)));
        });
    };
    // Scale bytes -> MFMA scale dwords.
    const unsigned sf_sel = sf_perm_sel<T>(lane_id);
    auto read_sf = [&](auto cur_) {
        constexpr int cur = decltype(cur_)::value;
        opus::static_for<T::E_M>([&](auto e) { v_sfa[e.value] = sf_pack(v_sfa_next[cur][e.value], sf_sel); });
        opus::static_for<T::E_N>([&](auto n) { v_sfb[n.value] = sf_pack(v_sfb_next[cur][n.value], sf_sel); });
    };

    // __restrict__ read views (function parameters): otherwise the reads wait for all in-flight LDS-DMA (vmcnt(0)).
    auto views = [&](OPUS_LDS_ADDR char* __restrict__ read_a, OPUS_LDS_ADDR char* __restrict__ read_b,
                     OPUS_LDS_ADDR char* __restrict__ read_sfa, OPUS_LDS_ADDR char* __restrict__ read_sfb, auto&& body) {
        auto rd_a = s_a;     rd_a.ptr = read_a;
        auto rd_b = s_b;     rd_b.ptr = read_b;
        auto rd_sfa = s_sfa; rd_sfa.ptr = read_sfa;
        auto rd_sfb = s_sfb; rd_sfb.ptr = read_sfb;
        body(rd_a, rd_b, rd_sfa, rd_sfb);
    };

    // Per-wave instruction counts.
    constexpr int BUFFER_A = T::a_buffer_load_insts, BUFFER_B = T::b_buffer_load_insts;   // buffer_load ... lds per step
    constexpr int BUFFER_SF = T::sfa_buffer_load_insts + T::sfb_buffer_load_insts;       // per panel (first step of a cycle)
    constexpr int DS_A = T::E_K * T::E_M, DS_B = T::E_K * T::E_N;                        // ds_read_b128 per step
    constexpr int DS_SF = T::E_M + T::E_N;                                               // ds_read_b64 per step
    constexpr int VMEM_STEP = BUFFER_A + BUFFER_B;
    static_assert(VMEM_STEP == T::VMEM_PER_STEP && DS_A + DS_B + DS_SF == T::DS_READS_PER_STEP);

    // Prologue: copy steps 0 .. STAGES-1, read step 0.  Waits are asm with a "memory" clobber (compiler barrier).
    static_for<STAGES>([&](auto t) { prefetch_step(t, k_begin + T::KS * decltype(t)::value, 0); __builtin_amdgcn_sched_barrier(0); });
    // Drain the workspace s_load here (else every ds_read wait becomes lgkmcnt(0)).
    s_waitcnt_lgkmcnt(0_I);
    asm volatile("s_waitcnt vmcnt(%0)" :: "n"((STAGES - 1) * VMEM_STEP) : "memory");
    views(s_a.ptr + sa_offset(0), s_b.ptr + sb_offset(0), s_sfa.ptr + ssfa_offset(0), s_sfb.ptr + ssfb_offset(0),
          [&](auto& rd_a, auto& rd_b, auto& rd_sfa, auto& rd_sfb) {
        sync_threads();
        __builtin_amdgcn_sched_barrier(0);
        load_a(rd_a, 0_I);
        load_b(rd_b, 0_I);
        load_sfa(rd_sfa, 0_I, 0);
        load_sfb(rd_sfb, 0_I, 0);
        __builtin_amdgcn_sched_barrier(0);
    });

    // Main loop: step t = read t+1, copy t + STAGES, MFMAs of t.  Two ring cycles per trip (panel parity PAR).
    int step = 0;
    auto main_trip = [&](auto par_) {
        constexpr int PAR = decltype(par_)::value;
        const int t0 = step + PAR * STAGES;
        static_for<STAGES>([&](auto u) {
            constexpr int stage = decltype(u)::value, next_stage = (stage + 1) % STAGES;
            constexpr int cur = stage % 2, next = 1 - cur;
            constexpr int sf_next = stage == STAGES - 1 ? 1 - PAR : PAR;   // panel of step t+1
            // In flight: steps t+2 .. t+STAGES-1 (+ the next panel).
            constexpr int VMCNT = (STAGES - 2) * VMEM_STEP + (stage >= 1 && stage <= STAGES - 2 ? BUFFER_SF : 0);
            constexpr int VMEM = VMEM_STEP + (stage == 0 ? BUFFER_SF : 0);
            views(s_a.ptr + sa_offset(next_stage), s_b.ptr + sb_offset(next_stage), s_sfa.ptr + ssfa_offset(sf_next), s_sfb.ptr + ssfb_offset(sf_next),
                  [&](auto& rd_a, auto& rd_b, auto& rd_sfa, auto& rd_sfb) {
                // step t+1 landed, stage(t) free.
                asm volatile("s_waitcnt vmcnt(%0) lgkmcnt(0)" :: "n"(VMCNT) : "memory");
                sync_threads();
                __builtin_amdgcn_sched_barrier(0);

                // MFMAs of t + read t+1, copy t + STAGES.
                load_a(rd_a, number<next>{});
                load_b(rd_b, number<next>{});
                load_sfa(rd_sfa, number<next>{}, next_stage);
                load_sfb(rd_sfb, number<next>{}, next_stage);
                prefetch_step(number<stage>{}, k_begin + T::KS * (t0 + stage + STAGES), 1 - PAR);
                read_sf(number<cur>{});
                mma_step<T>(mma, v_a[cur], v_b[cur], v_c, v_sfa, v_sfb);
                sched_step<T, DS_A + DS_B + DS_SF, VMEM>();
                __builtin_amdgcn_sched_barrier(0);
            });
        });
    };
#pragma clang loop unroll(disable)
    for (; step < steps; step += 2 * STAGES) {
        main_trip(0_I);
        if (step + STAGES < steps) main_trip(1_I);
    }
    s_waitcnt_vmcnt(0_I);

    // Epilogue (KS > 1): groups 1..KS-1 park partial tiles in LDS, group 0 adds them.
    if constexpr (T::KS > 1) {
        constexpr int FRAGS = T::E_M * T::E_N;
        auto s_red = make_smem(reinterpret_cast<D_ACC*>(smem_ab));
        auto red_off = [&](int g, int f) { return (((g * T::T_N + wave_id_n) * FRAGS + f) * T::WARP_SIZE + lane_id) * T::VEC_C; };
        s_waitcnt_lgkmcnt(0_I);
        sync_threads();
        if (wave_id_k > 0)
            static_for<FRAGS>([&](auto f) { store<T::VEC_C>(s_red, v_c[f.value], red_off(wave_id_k - 1, f.value)); });
        __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup");
        sync_threads();
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup");
        if (wave_id_k > 0) return;
        static_for<T::KS - 1>([&](auto g) { static_for<FRAGS>([&](auto f) {
            v_c[f.value] += load<T::VEC_C>(s_red, red_off(g.value, f.value));
        }); });
    }

    // Write back C (S > 1: fp32 partial tile to the workspace).
    if (split_k > 1) {
        const int padded_m = (kargs.m + T::B_M - 1) / T::B_M * T::B_M, padded_n = (kargs.n + T::B_N - 1) / T::B_N * T::B_N;
        auto g_w = make_gmem(ws + (static_cast<long long>(k_split * kargs.batch + batch_id) * padded_m + row) * padded_n + col_w,
                             static_cast<unsigned>((T::B_M * padded_n) * sizeof(float)));
        const auto u_gw = make_layout_gc<T>(lane_id, padded_n);
        static_for<T::E_M * T::E_N>([&](auto f) { store<T::VEC_C>(g_w, v_c[f.value], u_gw.offsets[f.value]); });
        return;
    }
    // Columns past N: out-of-range offset.
    constexpr int C_OOB = 0x40000000;
    const auto u_gc = make_layout_gc<T>(lane_id, kargs.stride_c);
    const int c_col = wave_id_n * T::WAVE_N + T::VEC_C * (lane_id / T::W_M);   // this lane's first column in the tile
    static_for<T::E_M>([&](auto e) { static_for<T::E_N>([&](auto n) {
        const int oob = c_col + T::W_N * n.value < cols_in ? 0 : C_OOB;
        store<T::VEC_C>(g_c, opus::cast<D_C>(v_c[e.value * T::E_N + n.value]), u_gc.offsets[e.value * T::E_N + n.value] + oob, 0, opus::number<C_AUX>{});
    }); });
#endif   // __gfx950__
}

}   // namespace opus_mxfp4_small

// grid = ((N + B_N - 1) / B_N, (M + B_M - 1) / B_M, batch * split_k), block = 256.
template<class Traits>
__global__ __launch_bounds__(Traits::BLOCK_SIZE, 1) void gemm_a4w4_mxfp4_nt_small_kernel(opus_gemm_mxfp4_small_kargs_gfx950 kargs) {
    opus_mxfp4_small::gemm_a4w4_mxfp4_nt_small_body<Traits>(kargs);
}
