// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// MXFP4 x MXFP4 NT 1x4-wave small-tile pipeline, aiter bpreshuffle inputs (A row-major, B shuffle_weight(16, 16),
// e8m0_shuffle scales).  A: STAGES-deep LDS-DMA ring; B and the raw scale dwords (already in the MFMA lane layout):
// STAGES-deep register ring.  KS > 1 splits K over the waves; split-K via splitk_reduce_kernel.
#pragma once
#include "opus_gemm_traits_a4w4_mxfp4_nt_bpreshuffle_small_gfx950.cuh"

namespace opus_mxfp4_bpreshuffle_small {
using opus::operator""_I;
using fp4x32_t = opus::vector_t<opus::i32_t, 4>;   // one fp4 MFMA operand: 32 fp4 = 16 bytes per lane
using i32x8_t = opus::vector_t<opus::i32_t, 8>;

// ===========================================================================
// A: global -> LDS (LDS-DMA) -> VGPR
// ===========================================================================
// Create layout for loading A matrix from global memory (wave w: NI chunks of 8 rows x 128 B of K tile w / WQ; 16 B
// pieces XOR-swizzled per row pair).
template<class T>
__device__ inline auto make_layout_ga(int lane_id, int wave_id_a, int stride_a) {
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
        opus::unfold_p_coord(ga_block_dim, opus::tuple{wave_id_a / WQ, wave_id_a % WQ, lane_id_r, (lane_id % threads_k) ^ ((lane_id_r >> 1) & 3)}));
}

// Create layout for storing A matrix to shared memory (chunk c at c * CHUNK_BYTES).
template<class T>
__device__ inline auto make_layout_sa(int wave_id_a) {
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
        opus::unfold_p_coord(sa_block_dim, opus::tuple{wave_id_a / WQ, wave_id_a % WQ}));
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

// ===========================================================================
// B: global -> VGPR (shuffle_weight(16, 16): 1 KB = 16 columns x 128 K per 64 lanes x 16 B)
// ===========================================================================
// Create layout for loading B matrix from global memory to registers (offsets[n * E_K + kk]: the wave's 16-column block
// n, K half kk; the K tile goes into the soffset).
template<class T>
__device__ inline auto make_layout_gb(int lane_id, int wave_id_n, int stride_b) {
    constexpr auto gb_block_shape = opus::make_tuple(
        opus::number<T::T_N>{},
        opus::number<T::E_N>{},
        opus::number<T::E_K>{},
        opus::number<T::WARP_SIZE>{},
        opus::number<T::VEC_B>{});

    constexpr auto gb_block_dim = opus::make_tuple(
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}),
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::y_dim{}));

    return opus::make_layout<T::VEC_B>(
        gb_block_shape,
        opus::unfold_x_stride(gb_block_dim, gb_block_shape, opus::tuple{T::W_N * stride_b, 1_I}),
        opus::unfold_p_coord(gb_block_dim, opus::tuple{wave_id_n, lane_id}));
}

// ===========================================================================
// Scales: raw e8m0_shuffle dwords, global -> VGPR
// ===========================================================================
// Create layout for loading scale factors from global memory: lane l reads dword l of each of LOADS 32-row blocks from
// block block0 on (offsets[j]; the K tile goes into the soffset).
template<class T, int ROWS, int LOADS>
__device__ inline auto make_layout_gsf(int lane_id, int block0, int stride_sf) {
    constexpr auto gsf_block_shape = opus::make_tuple(
        opus::number<(ROWS + T::SF_ROWS - 1) / T::SF_ROWS>{},
        opus::number<LOADS>{},
        opus::number<T::WARP_SIZE>{},
        opus::number<T::VEC_SF>{});

    constexpr auto gsf_block_dim = opus::make_tuple(
        opus::make_tuple(opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}));

    return opus::make_layout<T::VEC_SF>(
        gsf_block_shape,
        opus::unfold_x_stride(gsf_block_dim, gsf_block_shape, opus::tuple{stride_sf, stride_sf, 1_I}),
        opus::unfold_p_coord(gsf_block_dim, opus::tuple{block0, lane_id}));
}
template<class T> __device__ inline auto make_layout_gsfa(int lane_id, int stride_sfa) { return make_layout_gsf<T, T::B_M, T::sfa_buffer_load_insts>(lane_id, 0, stride_sfa); }
template<class T> __device__ inline auto make_layout_gsfb(int lane_id, int wave_id_n, int stride_sfb) {
    return make_layout_gsf<T, T::B_N, T::sfb_buffer_load_insts>(lane_id, wave_id_n * T::WAVE_N / T::SF_ROWS, stride_sfb);
}

// Register side: the 16-row block e (16-column block n) of K half kk is dword e / 2 (n / 2), byte opsel 2 kk + e % 2
// (n % 2).  A single 16-row (16-column) block uses the half of its 32-row block chosen by a runtime shift.
template<class T> __device__ inline constexpr int rsfa_dword(int e) { return e / 2; }
template<class T> __device__ inline constexpr int rsfa_opsel(int e, int kk) { return 2 * kk + (T::SFA_HALF ? 0 : e % 2); }
template<class T> __device__ inline constexpr int rsfb_dword(int n) { return n / 2; }
template<class T> __device__ inline constexpr int rsfb_opsel(int n, int kk) { return 2 * kk + (T::SFB_HALF ? 0 : n % 2); }

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

// C += A x B of one ring step (raw scale dwords: dword e / 2 (n / 2), opsel 2 kk + e % 2 (n % 2); a single 16-row
// (16-column) block shifts its half of the 32-row block into place).
template<class T, class MMA, class CF>
__device__ __forceinline__ void mma_step(MMA& mma, const fp4x32_t (&v_a)[T::E_K][T::E_M], const fp4x32_t (&v_b)[T::E_N][T::E_K],
                                         CF (&v_c)[T::E_M * T::E_N], const opus::u32_t (&v_sfa)[T::sfa_buffer_load_insts],
                                         const opus::u32_t (&v_sfb)[T::sfb_buffer_load_insts], unsigned sh_a, unsigned sh_b) {
    opus::static_for<T::E_K>([&](auto kk) { opus::static_for<T::E_M>([&](auto e) { opus::static_for<T::E_N>([&](auto n) {
        constexpr int K = decltype(kk)::value, E = decltype(e)::value, N = decltype(n)::value;
        const unsigned sfa = T::SFA_HALF ? (v_sfa[rsfa_dword<T>(E)] >> sh_a) : v_sfa[rsfa_dword<T>(E)];
        const unsigned sfb = T::SFB_HALF ? (v_sfb[rsfb_dword<T>(N)] >> sh_b) : v_sfb[rsfb_dword<T>(N)];
        mma_scale_one<rsfa_opsel<T>(E, K), rsfb_opsel<T>(N, K)>(mma, v_a[K][E], v_b[N][K], v_c[E * T::E_N + N], sfa, sfb);
    }); }); });
}

// ===========================================================================
// Kernel
// ===========================================================================
// Cache policy of the C stores: nt (C is written once).
constexpr int C_AUX = 2;
template<class Traits>
__device__ __forceinline__ void gemm_a4w4_mxfp4_nt_bpreshuffle_small_body(const opus_gemm_mxfp4_bpreshuffle_small_kargs_gfx950& kargs) {
#if defined(__gfx950__)
    using namespace opus;

    using T = opus::remove_cvref_t<Traits>;
    using D_A = opus::fp4_t;
    using D_B = opus::fp4_t;
    using D_ACC = opus::fp32_t;
    using D_C = opus::bf16_t;
    using D_AB = opus::u8_t;     // fp4x2 bytes
    using D_SF = opus::u8_t;     // e8m0 bytes, read as raw dwords
    constexpr int STAGES = T::STAGES;

    const int lane_id = static_cast<int>(opus::lane_id());
    const int wave_id = __builtin_amdgcn_readfirstlane(thread_id_x() / T::WARP_SIZE);
    const int wave_id_n = wave_id % T::T_N;
    const int wave_id_k = wave_id / T::T_N;
    const int split_k = kargs.split_k;
    const int batch_id = block_id_z() / split_k;
    const int k_split = block_id_z() % split_k;
    const int row = block_id_y() * T::B_M;
    const int col = block_id_x() * T::B_N;                 // the block's first column
    const int col_w = col + wave_id_n * T::WAVE_N;         // this wave's first column (B / scale copies, C)
    // S > 1: load the workspace pointer early.
    float* ws = nullptr;
    if (split_k > 1) ws = static_cast<float*>(kargs.ws_handle->ptr);
    const int k_tiles = kargs.k / T::B_K, k_per = (k_tiles + split_k - 1) / split_k;
    const int k_begin = k_split * k_per;
    const int k_end = k_tiles < k_begin + k_per ? k_tiles : k_begin + k_per;
    const int steps = k_end > k_begin ? (k_end - k_begin + T::KS - 1) / T::KS : 0;   // ring steps (KS K tiles each)

    // M / N boundary: rows past the matrix are outside the buffer range (loads 0, stores dropped); 64-bit tile base.
    // B / scales start at this wave's columns; scale blocks past the matrix read 0.
    const int rows_in = kargs.m - row < T::B_M ? kargs.m - row : T::B_M;
    const int cols_in = kargs.n - col < T::B_N ? kargs.n - col : T::B_N;
    auto g_a = make_gmem(reinterpret_cast<const D_AB*>(kargs.ptr_a) + batch_id * kargs.stride_a_batch + static_cast<long long>(row) * kargs.stride_a,
                         static_cast<unsigned>(rows_in * kargs.stride_a));
    auto g_b = make_gmem(reinterpret_cast<const D_AB*>(kargs.ptr_b) + batch_id * kargs.stride_b_batch + static_cast<long long>(col) * kargs.stride_b,
                         static_cast<unsigned>(cols_in * kargs.stride_b));
    auto g_c = make_gmem(reinterpret_cast<D_C*>(kargs.ptr_c) + batch_id * kargs.stride_c_batch + static_cast<long long>(row) * kargs.stride_c + col_w,
                         static_cast<unsigned>(rows_in * kargs.stride_c * sizeof(D_C)));
    auto g_sfa = make_gmem(reinterpret_cast<const D_SF*>(kargs.ptr_sfa) + batch_id * kargs.stride_sfa_batch + static_cast<long long>(row / T::SF_ROWS) * kargs.stride_sfa,
                           static_cast<unsigned>(((kargs.m + T::SF_ROWS - 1) / T::SF_ROWS - row / T::SF_ROWS) * kargs.stride_sfa));
    auto g_sfb = make_gmem(reinterpret_cast<const D_SF*>(kargs.ptr_sfb) + batch_id * kargs.stride_sfb_batch + static_cast<long long>(col / T::SF_ROWS) * kargs.stride_sfb,
                           static_cast<unsigned>(((kargs.n + T::SF_ROWS - 1) / T::SF_ROWS - col / T::SF_ROWS) * kargs.stride_sfb));
    // A single 16-row (16-column) block: its half of the 32-row scale block.
    const unsigned sh_a = T::SFA_HALF ? 8u * ((row / 16) & 1) : 0u;
    const unsigned sh_b = T::SFB_HALF ? 8u * ((col_w / 16) & 1) : 0u;

    // Fewer A chunks per step than waves: the last waves repeat the last chunk.
    const int wave_id_a = T::A_DMA_EXACT ? wave_id : (wave_id < T::A_DMA_CHUNKS / T::a_buffer_load_insts ? wave_id : T::A_DMA_CHUNKS / T::a_buffer_load_insts - 1);
    const auto u_ga = make_layout_ga<T>(lane_id, wave_id_a, kargs.stride_a);
    const auto u_sa = make_layout_sa<T>(wave_id_a);
    const auto u_ra = make_layout_ra<T>(lane_id, wave_id_k);
    const auto u_gb = make_layout_gb<T>(lane_id, wave_id_n, kargs.stride_b);
    const auto u_gsfa = make_layout_gsfa<T>(lane_id, kargs.stride_sfa);
    const auto u_gsfb = make_layout_gsfb<T>(lane_id, wave_id_n, kargs.stride_sfb);

    // LDS: STAGES ring stages of A (later the KS partial tiles).
    __shared__ __attribute__((aligned(16))) char smem_ab[T::LDS_BYTES];
    char* const smem_a = smem_ab;
    auto s_a = make_smem(reinterpret_cast<D_AB*>(smem_a));

    auto mma = make_tiled_mma<D_A, D_B, D_ACC>(
        seq<T::E_M, T::E_N, 1>{},          // E_K = 2 steps are v_a[kk]
        seq<T::T_M, T::T_N, T::T_K>{},
        seq<T::W_M, T::W_N, T::W_K>{},
        mfma_adaptor_swap_ab{});
    using MMA = decltype(mma);
    using CFrag = opus::vector_t<D_ACC, MMA::mma_c_len>;   // one 16x16 C fragment
    CFrag v_c[T::E_M * T::E_N];                            // [fragment]
    opus::static_for<T::E_M * T::E_N>([&](auto f) { clear(v_c[f.value]); });

    // Register ring: slot s holds the B operands and raw scale dwords of ring step s (mod STAGES).
    fp4x32_t v_b[STAGES][T::E_N][T::E_K];
    opus::u32_t v_sfa[STAGES][T::sfa_buffer_load_insts];
    opus::u32_t v_sfb[STAGES][T::sfb_buffer_load_insts];

    auto ga_offset = [&](int tile_k) { return tile_k * T::B_K_BYTES; };
    auto gb_offset = [&](int tile_k) { return tile_k * T::B_BLOCK_BYTES * T::E_K; };
    auto gsf_offset = [&](int tile_k) { return tile_k * T::SF_K_TILE_BYTES; };
    auto sa_offset = [&](int stage) { return stage * T::STAGE_BYTES; };

    // Global -> LDS / VGPR.  K tiles past k_end read 0 through an out-of-range voffset (the bounds check skips soffset),
    // so the copy count per step is fixed (counted vmcnt) and the loop has no branch.
    constexpr int G_OOB = 0x40000000;
    const int a_t = wave_id_a / (T::A_CHUNKS / T::a_buffer_load_insts);
    auto prefetch_a = [&](int stage, int tile) {
        async_load_imm<T::VEC_A>(g_a, s_a.ptr, u_ga + (tile + a_t < k_end ? 0 : G_OOB), u_sa + sa_offset(stage), ga_offset(tile));
    };
    // This wave's B / scales: K tile tile + wave_id_k.
    auto prefetch_b = [&](auto slot_, int tile) {
        constexpr int slot = decltype(slot_)::value;
        const int oob = tile + wave_id_k < k_end ? 0 : G_OOB;
        opus::static_for<T::E_N>([&](auto n) { opus::static_for<T::E_K>([&](auto kk) {
            v_b[slot][n.value][kk.value] = __builtin_bit_cast(fp4x32_t,
                load<T::VEC_B>(g_b, u_gb.offsets[n.value * T::E_K + kk.value] + oob, gb_offset(tile + wave_id_k)));
        }); });
    };
    auto prefetch_sfa = [&](auto slot_, int tile) {
        constexpr int slot = decltype(slot_)::value;
        const int oob = tile + wave_id_k < k_end ? 0 : G_OOB;
        opus::static_for<T::sfa_buffer_load_insts>([&](auto j) {
            v_sfa[slot][j.value] = __builtin_bit_cast(opus::u32_t, load<T::VEC_SF>(g_sfa, u_gsfa.offsets[j.value] + oob, gsf_offset(tile + wave_id_k)));
        });
    };
    auto prefetch_sfb = [&](auto slot_, int tile) {
        constexpr int slot = decltype(slot_)::value;
        const int oob = tile + wave_id_k < k_end ? 0 : G_OOB;
        opus::static_for<T::sfb_buffer_load_insts>([&](auto j) {
            v_sfb[slot][j.value] = __builtin_bit_cast(opus::u32_t, load<T::VEC_SF>(g_sfb, u_gsfb.offsets[j.value] + oob, gsf_offset(tile + wave_id_k)));
        });
    };
    // All copies of a ring step.
    auto prefetch_step = [&](auto stage_, int tile) {
        prefetch_a(decltype(stage_)::value, tile);
        prefetch_sfa(stage_, tile);
        prefetch_b(stage_, tile);
        prefetch_sfb(stage_, tile);
    };

    // LDS -> VGPR.
    const auto ra_offs = opus::layout_to_offsets<T::VEC_A>(u_ra);
    auto load_a = [&](auto& sa_, fp4x32_t (&v)[T::E_K][T::E_M]) {
        opus::static_for<T::E_K>([&](auto kk) { opus::static_for<T::E_M>([&](auto e) {
            v[kk.value][e.value] = __builtin_bit_cast(fp4x32_t, load<T::VEC_A>(sa_, ra_offs[e.value * T::E_K + kk.value]));
        }); });
    };

    // __restrict__ read view (function parameter): otherwise the reads wait for all in-flight LDS-DMA (vmcnt(0)).
    auto views = [&](OPUS_LDS_ADDR char* __restrict__ read_a, auto&& body) {
        auto rd_a = s_a; rd_a.ptr = read_a;
        body(rd_a);
    };

    // Per-wave instruction counts.
    constexpr int DS_A = T::E_K * T::E_M;                                                // ds_read_b128 per step
    constexpr int VMEM_STEP = T::VMEM_PER_STEP;                                          // A copies + B / scale loads

    // Prologue: copy steps 0 .. STAGES-1, read A of step 0.  Waits are asm with a "memory" clobber (compiler barrier).
    static_for<STAGES>([&](auto t) { prefetch_step(t, k_begin + T::KS * decltype(t)::value); __builtin_amdgcn_sched_barrier(0); });
    // Drain the workspace s_load here (else every ds_read wait becomes lgkmcnt(0)).
    s_waitcnt_lgkmcnt(0_I);
    asm volatile("s_waitcnt vmcnt(%0)" :: "n"((STAGES - 1) * VMEM_STEP) : "memory");
    fp4x32_t v_a[T::E_K][T::E_M];   // A operands of the current step
    views(s_a.ptr + sa_offset(0), [&](auto& rd_a) {
        sync_threads();
        load_a(rd_a, v_a);
        __builtin_amdgcn_sched_barrier(0);
    });

    // Main loop: step t = read A of t+1, MFMAs of t, copy t + STAGES into t's slot.
#pragma clang loop unroll(disable)
    for (int step = 0; step < steps; step += STAGES) {
        static_for<STAGES>([&](auto u) {
            constexpr int stage = decltype(u)::value, next_stage = (stage + 1) % STAGES;
            const int tile = k_begin + T::KS * (step + stage);
            views(s_a.ptr + sa_offset(next_stage), [&](auto& rd_a) {
                // step t+1 landed (STAGES-2 younger steps in flight), stage(t) free (every wave read A(t)).
                asm volatile("s_waitcnt vmcnt(%0) lgkmcnt(0)" :: "n"((STAGES - 2) * VMEM_STEP) : "memory");
                sync_threads();
                __builtin_amdgcn_sched_barrier(0);

                // MFMAs of t + read A of t+1, copy t + STAGES.
                fp4x32_t v_an[T::E_K][T::E_M];
                load_a(rd_a, v_an);
                mma_step<T>(mma, v_a, v_b[stage], v_c, v_sfa[stage], v_sfb[stage], sh_a, sh_b);
                prefetch_step(number<stage>{}, tile + T::KS * STAGES);
                sched_step<T, DS_A, VMEM_STEP>();
                __builtin_amdgcn_sched_barrier(0);
                opus::static_for<T::E_K>([&](auto kk) { opus::static_for<T::E_M>([&](auto e) { v_a[kk.value][e.value] = v_an[kk.value][e.value]; }); });
            });
        });
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

}   // namespace opus_mxfp4_bpreshuffle_small

// grid = ((N + B_N - 1) / B_N, (M + B_M - 1) / B_M, batch * split_k), block = 256.
template<class Traits>
__global__ __launch_bounds__(Traits::BLOCK_SIZE, 1) void gemm_a4w4_mxfp4_nt_bpreshuffle_small_kernel(opus_gemm_mxfp4_bpreshuffle_small_kargs_gfx950 kargs) {
    opus_mxfp4_bpreshuffle_small::gemm_a4w4_mxfp4_nt_bpreshuffle_small_body<Traits>(kargs);
}
