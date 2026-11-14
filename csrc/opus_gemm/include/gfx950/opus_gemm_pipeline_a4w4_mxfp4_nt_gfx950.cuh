#pragma once

#include <opus/hip_minimal.hpp>
#include <opus/opus.hpp>

#include "opus_gemm_traits_a4w4_mxfp4_nt_gfx950.cuh"

#ifndef AB_AUX   // cache policy of the A / B LDS-DMA loads
#define AB_AUX 0
#endif

using opus::operator""_I;

// Create layout for loading A matrix from global memory (LDS-DMA, 16 B pieces XOR-swizzled per row pair).
template<class T>
__device__ inline auto make_layout_ga(int lane_id, int wave_id_m, int wave_id_n, int stride_a) {
    constexpr int threads_k = T::B_K_BYTES / T::VEC_A;

    constexpr auto ga_block_shape = opus::make_tuple(
        opus::number<T::T_N>{},
        opus::number<T::T_M>{},
        opus::number<T::a_buffer_load_insts / 2>{},
        opus::number<T::WARP_SIZE / threads_k>{},
        opus::number<2>{},
        opus::number<threads_k>{},
        opus::number<T::VEC_A>{});

    constexpr auto ga_block_dim = opus::make_tuple(
        opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::y_dim{}, opus::p_dim{}, opus::y_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}));

    const int lane_id_m = lane_id / threads_k;

    return opus::make_layout<T::VEC_A>(
        ga_block_shape,
        opus::unfold_x_stride(ga_block_dim, ga_block_shape, opus::tuple{stride_a, 1_I}),
        opus::unfold_p_coord(ga_block_dim, opus::tuple{wave_id_n, wave_id_m, lane_id_m, (lane_id % threads_k) ^ (((lane_id_m >> 1) & 1) << 1)}));
}

// Create layout for storing A matrix to shared memory.
template<class T>
__device__ inline auto make_layout_sa(int wave_id_m, int wave_id_n) {
    constexpr auto sa_block_shape = opus::make_tuple(
        opus::number<T::T_N>{},
        opus::number<T::T_M>{},
        opus::number<T::a_buffer_load_insts>{},
        opus::number<T::VEC_A>{});

    constexpr auto sa_block_dim = opus::make_tuple(
        opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::y_dim{}),
        opus::make_tuple(opus::y_dim{}));

    return opus::make_layout(
        sa_block_shape,
        opus::unfold_x_stride(sa_block_dim, sa_block_shape, opus::tuple{opus::number<T::smem_linear_wave_a + T::smem_padding>{}, 1_I}),
        opus::unfold_p_coord(sa_block_dim, opus::tuple{wave_id_n, wave_id_m}));
}

// Create layout for reading A matrix from shared memory to registers (kk = 0; kk = 1 is + W_K_BYTES).
template<class T>
__device__ inline auto make_layout_ra(int lane_id, int wave_id_m) {
    constexpr auto ra_block_shape = opus::make_tuple(
        opus::number<T::E_M>{},
        opus::number<T::T_M>{},
        opus::number<T::T_M>{},
        opus::number<T::W_M / T::T_M>{},
        opus::number<T::E_K>{},
        opus::number<T::W_M * T::W_K_BYTES / T::WARP_SIZE / T::VEC_A>{},
        opus::number<T::WARP_SIZE / T::W_M>{},
        opus::number<T::VEC_A>{});

    constexpr auto ra_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::y_dim{}, opus::p_dim{}, opus::y_dim{}));

    const int lane_id_m = lane_id % T::W_M;

    return opus::make_layout<T::VEC_A>(
        ra_block_shape,
        opus::unfold_x_stride(ra_block_dim, ra_block_shape, opus::tuple{opus::number<T::smem_linear_wave_a + T::smem_padding>{}, 1_I}),
        opus::unfold_p_coord(ra_block_dim, opus::tuple{wave_id_m, lane_id_m % T::T_M, lane_id_m / T::T_M, 0_I, (lane_id / T::W_M) ^ (((lane_id_m >> 2) & 1) << 1)}));
}

// Create layout for loading B matrix (shuffle_weight(16, 16): 1 KB per 16 rows x 128 K bytes) from global memory.
template<class T>
__device__ inline auto make_layout_gb(int lane_id, int wave_id_m, int wave_id_n, int stride_b) {
    constexpr auto gb_block_shape = opus::make_tuple(
        opus::number<T::T_M>{},
        opus::number<T::T_N>{},
        opus::number<T::b_buffer_load_insts / 2>{},
        opus::number<2>{},
        opus::number<T::WARP_SIZE>{},
        opus::number<T::VEC_B>{});

    constexpr auto gb_block_dim = opus::make_tuple(
        opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::y_dim{}),
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::y_dim{}));

    return opus::make_layout<T::VEC_B>(
        gb_block_shape,
        opus::unfold_x_stride(gb_block_dim, gb_block_shape, opus::tuple{T::W_N * stride_b, 1_I}),
        opus::unfold_p_coord(gb_block_dim, opus::tuple{wave_id_m, wave_id_n, lane_id}));
}

// Create layout for storing B matrix to shared memory.
template<class T>
__device__ inline auto make_layout_sb(int wave_id_m, int wave_id_n) {
    constexpr auto sb_block_shape = opus::make_tuple(
        opus::number<T::T_M>{},
        opus::number<T::T_N>{},
        opus::number<T::b_buffer_load_insts>{},
        opus::number<T::VEC_B>{});

    constexpr auto sb_block_dim = opus::make_tuple(
        opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::y_dim{}),
        opus::make_tuple(opus::y_dim{}));

    return opus::make_layout(
        sb_block_shape,
        opus::unfold_x_stride(sb_block_dim, sb_block_shape, opus::tuple{opus::number<T::smem_linear_wave_b + T::smem_padding>{}, 1_I}),
        opus::unfold_p_coord(sb_block_dim, opus::tuple{wave_id_m, wave_id_n}));
}

// Create layout for reading B matrix from shared memory to registers (kk = 0; kk = 1 is the next chunk).
template<class T>
__device__ inline auto make_layout_rb(int lane_id, int wave_id_n) {
    constexpr auto rb_block_shape = opus::make_tuple(
        opus::number<T::E_N>{},
        opus::number<T::T_N>{},
        opus::number<T::WARP_SIZE>{},
        opus::number<T::VEC_B>{});

    constexpr auto rb_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}));

    return opus::make_layout<T::VEC_B>(
        rb_block_shape,
        opus::unfold_x_stride(rb_block_dim, rb_block_shape, opus::tuple{opus::number<2 * (T::smem_linear_wave_b + T::smem_padding)>{}, 1_I}),
        opus::unfold_p_coord(rb_block_dim, opus::tuple{wave_id_n, lane_id}));
}

// Like opus async_load(layout), but the per-issue LDS offsets go into the MUBUF immediate (one LDS base in m0); the
// immediate also shifts the global address, so it is subtracted there.
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
        g.template async_load<VEC>(base, gofs[decltype(i)::value] - io, s_os, opus::number<io>{}, opus::number<AB_AUX>{});
    });
}

// sched_group_barrier masks (LLVM AMDGPU)
constexpr int SCHED_VALU = 0x002;
constexpr int SCHED_MFMA = 0x008;
constexpr int SCHED_VMEM_READ = 0x820;   // 0x020 VMEM read | 0x800 LDS DMA (newer LLVM classifies buffer_load ... lds as LDS DMA)
constexpr int SCHED_VMEM_WRITE = 0x040;
constexpr int SCHED_DS_READ = 0x100;

template<int MASK, int CNT = 1>
__device__ inline void sched_group() {
    if constexpr (CNT > 0) __builtin_amdgcn_sched_group_barrier(MASK, CNT, 0);
}

// One MFMA, then DS reads and VALU.
template<int DS, int VALU>
__device__ inline void sched_m() {
    sched_group<SCHED_MFMA>();
    sched_group<SCHED_DS_READ, DS>();
    sched_group<SCHED_VALU, VALU>();
}

// REPS x 4 MFMAs with up to 2 DS reads, 1 VMEM read and VALU between them.
template<int REPS, int DS, int VMEM, int VALU>
__device__ inline void sched_x4() {
    static_assert(DS <= 2 && VMEM <= 1);
    opus::static_for<REPS>([&](auto) {
        sched_group<SCHED_MFMA>();
        sched_group<SCHED_DS_READ, DS >= 1>();
        sched_group<SCHED_MFMA>();
        sched_group<SCHED_VMEM_READ, VMEM>();
        sched_group<SCHED_MFMA>();
        sched_group<SCHED_DS_READ, DS >= 2>();
        sched_group<SCHED_MFMA>();
        sched_group<SCHED_VALU, VALU>();
    });
}

// Generic quadrant schedule: Q MFMAs with DS reads / VMEM reads / VALU / VMEM writes spread evenly after them.
template<int Q, int DS, int VMEM, int VALU, int STORE>
__device__ inline void sched_spread() {
    opus::static_for<Q>([&](auto i_) {
        constexpr int i = decltype(i_)::value;
        sched_group<SCHED_MFMA>();
        sched_group<SCHED_DS_READ, DS * (i + 1) / Q - DS * i / Q>();
        sched_group<SCHED_VMEM_READ, VMEM * (i + 1) / Q - VMEM * i / Q>();
        sched_group<SCHED_VALU, VALU * (i + 1) / Q - VALU * i / Q>();
        sched_group<SCHED_VMEM_WRITE, STORE * (i + 1) / Q - STORE * i / Q>();
    });
}

// Hand-tuned quadrant schedules of the 256 x 256 tile (32 MFMAs each).
// C10: A0 / B0 copies of tile t+2 (8 VMEM).
__device__ inline void sched_c10() { sched_x4<8, 0, 1, 0>(); }
// C10 of the last K tile without copies.
__device__ inline void sched_c10_last() { sched_x4<8, 0, 0, 0>(); }
// C01: 24 DS (scales, B0 of t+1) + 8 VMEM (A1 / B1 copies of t+2).
__device__ inline void sched_c01() { sched_x4<4, 2, 1, 0>(); sched_x4<4, 1, 1, 0>(); }
// C01 of the last K tiles: 24 DS, no VMEM.
__device__ inline void sched_c01_last() { sched_x4<4, 2, 0, 0>(); sched_x4<4, 1, 0, 0>(); }
// C01 with the A1 / B1 copies moved to C11 / C00: 16 DS, no VMEM.
__device__ inline void sched_c01_nov() {
    opus::static_for<8>([&](auto) {
        sched_group<SCHED_MFMA>();
        sched_group<SCHED_DS_READ>();
        sched_group<SCHED_MFMA, 2>();
        sched_group<SCHED_DS_READ>();
        sched_group<SCHED_MFMA>();
    });
}
// C11: 8 DS (A0 of t+1) + 24 VALU (scale shifts of half 0) + 1 VMEM (scale panel).
__device__ inline void sched_c11() {
    opus::static_for<7>([&](auto) { sched_m<1, 1>(); sched_m<0, 1>(); sched_m<0, 1>(); sched_m<0, 0>(); });
    sched_m<1, 1>();
    sched_group<SCHED_VMEM_READ>();
    sched_m<0, 1>(); sched_m<0, 1>(); sched_m<0, 0>();
}
// C11 + 5 VMEM (4 A1 copies + scale panel).
__device__ inline void sched_c11_pf() {
    opus::static_for<8>([&](auto r) {
        sched_group<SCHED_MFMA>();
        sched_group<SCHED_DS_READ>();
        sched_group<SCHED_VALU>();
        sched_group<SCHED_MFMA>();
        sched_group<SCHED_VMEM_READ, decltype(r)::value % 2 == 0 || decltype(r)::value == 7>();
        sched_group<SCHED_VALU>();
        sched_group<SCHED_MFMA>();
        sched_group<SCHED_VALU>();
        sched_group<SCHED_MFMA>();
    });
}
// C11 of the last K tile + scale panel 0 of the next output tile (3 VMEM).
__device__ inline void sched_c11_next() { sched_x4<2, 0, 1, 0>(); sched_x4<6, 0, 0, 0>(); }
// C00: 16 DS (A1 / B1 of t+1) in the first 24 MFMAs + 24 VALU (scale shifts of half 1).
__device__ inline void sched_c00() {
    opus::static_for<8>([&](auto) { sched_m<1, 1>(); sched_m<1, 1>(); sched_m<0, 1>(); });
    opus::static_for<8>([&](auto) { sched_m<0, 0>(); });
}
// C00 + 4 VMEM (B1 copies).
__device__ inline void sched_c00_pf() {
    opus::static_for<8>([&](auto r) {
        sched_group<SCHED_MFMA>();
        sched_group<SCHED_DS_READ>();
        sched_group<SCHED_VALU>();
        sched_group<SCHED_MFMA>();
        sched_group<SCHED_DS_READ>();
        sched_group<SCHED_VALU>();
        sched_group<SCHED_MFMA>();
        sched_group<SCHED_VMEM_READ, decltype(r)::value % 2 == 1>();
        sched_group<SCHED_VALU>();
    });
    opus::static_for<8>([&](auto) { sched_group<SCHED_MFMA>(); });
}
// Prologue C00 + 16 VMEM writes (bottom half of the previous output tile).
__device__ inline void sched_c00_store() {
    opus::static_for<8>([&](auto) {
        sched_group<SCHED_MFMA>();
        sched_group<SCHED_DS_READ>();
        sched_group<SCHED_MFMA>();
        sched_group<SCHED_VMEM_WRITE>();
        sched_group<SCHED_MFMA>();
        sched_group<SCHED_DS_READ>();
        sched_group<SCHED_MFMA>();
        sched_group<SCHED_VMEM_WRITE>();
    });
}

// The 256 x 256 tile uses the hand-tuned schedule TUNED; other tiles spread the instructions evenly.
template<class T, void (*TUNED)(), int DS, int VMEM, int VALU, int STORE = 0>
__device__ inline void sched_quadrant() {
    if constexpr (T::E_M == 4 && T::E_N == 4) TUNED();
    else sched_spread<2 * T::E_M * T::E_N, DS, VMEM, VALU, STORE>();
}

// One fp4 MFMA operand per lane: 32 fp4 = 16 bytes = 4 registers, held as 4 x i32 (opus::vector_t<fp4_t, 32> is a
// bit-packed struct, not a register vector).
using fp4x32_t = opus::vector_t<opus::i32_t, 4>;

// One scaled fp4 MFMA into the C fragment pinned at C_REG (>= 0: AGPR, <= -2: VGPR -C_REG - 2, -1: not pinned); ZC
// starts from C = 0.  The 16 B operands are widened to the 32 B builtin operand (the upper half is not read).
template<int OPSEL_A, int OPSEL_B, int C_REG, bool ZC = false, class MMA, class CF>
__device__ inline void mma_scale_one(MMA&, const fp4x32_t& a, const fp4x32_t& b, CF& c, unsigned int v_sfa, unsigned int v_sfb) {
    using base_mma = typename opus::remove_cvref_t<MMA>::MMA;
    using i32x8_t = opus::vector_t<opus::i32_t, 8>;
    const i32x8_t a8 = __builtin_shufflevector(a, a, 0, 1, 2, 3, -1, -1, -1, -1);
    const i32x8_t b8 = __builtin_shufflevector(b, b, 0, 1, 2, 3, -1, -1, -1, -1);
    if constexpr (C_REG >= 0) {
        [[clang::amdgpu_pin_agpr(C_REG)]] c = base_mma{}(a8, b8, ZC ? CF{} : c, static_cast<int>(v_sfa), static_cast<int>(v_sfb), opus::number<OPSEL_A>{}, opus::number<OPSEL_B>{});
    } else if constexpr (C_REG <= -2) {
        [[clang::amdgpu_pin_vgpr(-C_REG - 2)]] c = base_mma{}(a8, b8, ZC ? CF{} : c, static_cast<int>(v_sfa), static_cast<int>(v_sfb), opus::number<OPSEL_A>{}, opus::number<OPSEL_B>{});
    } else {
        c = base_mma{}(a8, b8, ZC ? CF{} : c, static_cast<int>(v_sfa), static_cast<int>(v_sfb), opus::number<OPSEL_A>{}, opus::number<OPSEL_B>{});
    }
}

// Pinned register of C fragment (m, n) of quadrant (HM, HN): HN = 0 in AGPR 128.., HN = 1 in VGPR 128...
template<class T, int HM, int HN, int M, int N>
constexpr int c_reg() {
    constexpr int frag = (M * T::E_N + N) * 4;
    if constexpr (HN == 0) return 128 + HM * 64 + frag;
    else return -(128 + HM * 64 + frag) - 2;
}

// ===========================================================================
// Kernel
// ===========================================================================
#ifndef SK_AUX   // cache policy of the split-K partial stores / loads: sc0 sc1 (system coherent, bypass L2)
#define SK_AUX 17
#endif
#ifndef C_AUX    // cache policy of the C stores: nt
#define C_AUX 2
#endif

// Kernel body as a device function: it backs the template __global__ below and the extern "C" entry points of the
// pre-compiled .co images (gen_co/build_co_gfx950.py).
template<class Traits>
__device__ __forceinline__ void gemm_a4w4_mxfp4_nt_body(opus_gemm_mxfp4_scale_kargs_gfx950 kargs) {
#if defined(__gfx950__)
    using namespace opus;

    using T = opus::remove_cvref_t<Traits>;
    using D_A = opus::fp4_t;
    using D_B = opus::fp4_t;
    using D_AB = opus::u8_t;     // fp4x2 bytes
    using D_C = opus::bf16_t;
    using D_ACC = opus::fp32_t;
    using D_SF = opus::e8m0_t;
    using D_SF_PACK = opus::u32_t;

    const int num_tiles_n = opus_mxfp4_ceil_div(kargs.n, T::B_N);
    const int wave_id = __builtin_amdgcn_readfirstlane(thread_id_x() / T::WARP_SIZE);
    const int tiles_per_batch = opus_mxfp4_ceil_div(kargs.m, T::B_M) * num_tiles_n;
    const int total_tiles = tiles_per_batch * kargs.batch;
    // Persistent: block b runs tiles b, b + grid_blocks, ...; one tile (split-K: tile x K segment) per block otherwise.
    const int split_k = T::SPLITK ? (kargs.split_k < 0 ? -kargs.split_k : kargs.split_k) : 1;
    const int grid_blocks = T::PERSIST ? kargs.grid_blocks : total_tiles;
    const int first_tile = T::SPLITK ? static_cast<int>(block_id_x()) / split_k : static_cast<int>(block_id_x());
    const int k_split = T::SPLITK ? static_cast<int>(block_id_x()) % split_k : 0;

    // Persistent: K tile 0 (+ scale panel 0) of the next output tile is copied in the last K tile of the current one,
    // and the bottom half of C (bf16) is stored in the next tile's prologue.
    int prefetched = 0;
    constexpr int H1_VEC = 16 / sizeof(D_C);
    constexpr int H1_COPIES = T::HALF_B_M * T::B_N / (T::BLOCK_SIZE * H1_VEC);
    opus::vector_t<D_C, H1_VEC> v_pend_c[H1_COPIES];
    D_C* pend_c_ptr = reinterpret_cast<D_C*>(kargs.ptr_c);
    int pend_c = 0;
    int pend_rows = 0, pend_cols = 0;
    // Out-of-range element offset for masked C stores (byte offset 2^31 > any buffer range we set).
    constexpr int C_OOB = 0x40000000;
    auto store_pend_c = [&](int tid) {
        auto g_cp = make_gmem(pend_c_ptr, pend_c ? static_cast<unsigned int>(pend_rows) * static_cast<unsigned int>(kargs.stride_c) * 2u : 0u);
        if constexpr (T::B_N == 192) {   // copy i -> rows (i / 3) * 32 + tid / 8, columns (i % 3) * 64 + (tid % 8) * 8
            opus::static_for<H1_COPIES>([&](auto i) {
                constexpr int I = decltype(i)::value;
                const int r = (I / 3) * 32 + tid / 8 + T::HALF_B_M, c = (I % 3) * 64 + (tid % 8) * 8;
                store<H1_VEC>(g_cp, v_pend_c[I], r * kargs.stride_c + c + (c < pend_cols ? 0 : C_OOB), 0, opus::number<C_AUX>{});
            });
        } else {   // a thread's column is the same in every copy (copies step by whole rows)
            const int oob = (tid * H1_VEC) % T::B_N < pend_cols ? 0 : C_OOB;
            opus::static_for<H1_COPIES>([&](auto i) {
                const int linear = tid * H1_VEC + decltype(i)::value * T::BLOCK_SIZE * H1_VEC;
                const int r = linear / T::B_N + T::HALF_B_M, c = linear % T::B_N;
                store<H1_VEC>(g_cp, v_pend_c[i.value], r * kargs.stride_c + c + oob, 0, opus::number<C_AUX>{});
            });
        }
    };

    // A / B operands live across output tiles (the last K tile of one tile overlaps the next tile's prologue).
    fp4x32_t v_a[2][T::E_K][T::E_M];   // [half][kk][repeat]
    fp4x32_t v_b[2][T::E_K][T::E_N];
#pragma clang loop unroll(disable)
    for (int tile_id = first_tile; tile_id < total_tiles; tile_id += grid_blocks) {
    const int batch_id = tile_id / tiles_per_batch;
    const int tile_in_batch = tile_id % tiles_per_batch;
    const int row = (tile_in_batch / num_tiles_n) * T::B_M;
    const int col = (tile_in_batch % num_tiles_n) * T::B_N;
    // Split-K: this block's K segment starts at K tile kt0; the A / B / scale bases below are moved there.
    const int k_seg = T::SPLITK ? kargs.k / split_k : kargs.k;
    const int kt0 = T::SPLITK ? k_split * (k_seg / T::B_K) : 0;
    const int loops = k_seg / T::B_K;

    // Lane id from mbcnt over an opaque all-ones mask: the compiler still knows it is in [0, 64) but cannot hoist the
    // lane-dependent offsets out of the tile loop (kept live through the write-back, they spill).  Written out at each
    // use: from a helper function the early passes lose the [0, 64) range and the generated code changes.
    unsigned int lane_mask = ~0u;
    asm volatile("" : "+s"(lane_mask));
    const int lane_id = static_cast<int>(__builtin_amdgcn_mbcnt_hi(lane_mask, __builtin_amdgcn_mbcnt_lo(lane_mask, 0u)));
    const int wave_id_m = wave_id % T::T_M;
    const int wave_id_n = wave_id / T::T_M;

    // M / N boundary: rows of A / B / C past the matrix are outside the buffer range (loads return 0, stores are
    // dropped); C columns past N are masked per 16 B copy.
    const int rows_in = T::BOUNDARY ? kargs.m - row : T::B_M, cols_in = T::BOUNDARY ? kargs.n - col : T::B_N;
    const int a_k0 = kt0 * T::B_K_BYTES, b_k0 = kt0 * T::B_K_BYTES * T::W_N, sf_k0 = kt0 * 256;
    auto g_a = make_gmem(reinterpret_cast<const D_AB*>(kargs.ptr_a) + batch_id * kargs.stride_a_batch + row * kargs.stride_a + a_k0,
                         T::BOUNDARY ? static_cast<unsigned int>(rows_in) * static_cast<unsigned int>(kargs.stride_a) - a_k0 : 0xffffffffu);
    auto g_b = make_gmem(reinterpret_cast<const D_AB*>(kargs.ptr_b) + batch_id * kargs.stride_b_batch + col * kargs.stride_b + b_k0,
                         T::BOUNDARY ? static_cast<unsigned int>(cols_in) * static_cast<unsigned int>(kargs.stride_b) - b_k0 : 0xffffffffu);
    auto g_c = make_gmem(reinterpret_cast<D_C*>(kargs.ptr_c) + batch_id * kargs.stride_c_batch + row * kargs.stride_c + col,
                         T::BOUNDARY ? static_cast<unsigned int>(rows_in) * static_cast<unsigned int>(kargs.stride_c) * 2u : 0xffffffffu);
    auto g_sfa = make_gmem(reinterpret_cast<const D_SF*>(kargs.ptr_sfa) + batch_id * kargs.stride_sfa_batch + (row / 32) * kargs.stride_sfa + sf_k0,
                           static_cast<unsigned int>(opus_mxfp4_ceil_div(kargs.m - row, 32) * kargs.stride_sfa - sf_k0));
    auto g_sfb = make_gmem(reinterpret_cast<const D_SF*>(kargs.ptr_sfb) + batch_id * kargs.stride_sfb_batch + (col / 32) * kargs.stride_sfb + sf_k0,
                           static_cast<unsigned int>(opus_mxfp4_ceil_div(kargs.n - col, 32) * kargs.stride_sfb - sf_k0));

    const auto u_ga = make_layout_ga<T>(lane_id, wave_id_m, wave_id_n, kargs.stride_a);
    const auto u_gb = make_layout_gb<T>(lane_id, wave_id_m, wave_id_n, kargs.stride_b);
    const auto u_sa = make_layout_sa<T>(wave_id_m, wave_id_n);
    const auto u_sb = make_layout_sb<T>(wave_id_m, wave_id_n);
    const auto u_ra = make_layout_ra<T>(lane_id, wave_id_m);
    const auto u_rb = make_layout_rb<T>(lane_id, wave_id_n);

    // LDS: [A stage 0][B stage 0][A stage 1][B stage 1]; stage 1 (66 KB) also stages half of C in the write-back while
    // stage 0 receives the next tile's K tile 0.
    constexpr int smem_a_elem = T::smem_m_rep * (T::smem_linear_wave_a + T::smem_padding);
    constexpr int smem_b_elem = T::smem_n_rep * (T::smem_linear_wave_b + T::smem_padding);
    constexpr int smem_a_bytes = smem_a_elem * 4 * sizeof(D_A);
    constexpr int smem_b_bytes = smem_b_elem * 4 * sizeof(D_B);
    __shared__ __attribute__((aligned(16))) char smem_ab[smem_a_bytes + smem_b_bytes];
    char* const smem_a = smem_ab;
    char* const smem_b = smem_ab + smem_a_bytes / 2;
    __shared__ char smem_sfa[T::SCALE_PANEL_BUFFERS * T::SFA_PANEL_BYTES * sizeof(D_SF)];
    __shared__ char smem_sfb[T::SCALE_PANEL_BUFFERS * T::SFB_PANEL_BYTES * sizeof(D_SF)];
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
    CFrag v_c[2][2][T::E_M * T::E_N];                      // [half m][half n][fragment]
    opus::static_for<T::E_M * T::E_N>([&](auto f) { clear(v_c[0][0][f.value]); });

    // Per-half scale dwords of the MFMAs (opsel 2 kk picks the byte).
    D_SF_PACK v_ssa[2][T::E_M], v_ssb[2][T::E_N];

    auto ga_offset = [&](int half_tile_m, int tile_k) { return half_tile_m * T::HALF_B_M * kargs.stride_a + tile_k * T::B_K_BYTES; };
    auto gb_offset = [&](int half_tile_n, int tile_k) { return half_tile_n * T::HALF_B_N * kargs.stride_b + tile_k * (T::B_K_BYTES * T::W_N); };
    constexpr int stage_elems = 2 * smem_a_elem + 2 * smem_b_elem;
    auto sa_offset = [&](int stage, int half_tile_m) { return stage * stage_elems + half_tile_m * smem_a_elem; };
    auto sb_offset = [&](int stage, int half_tile_n) { return stage * stage_elems + half_tile_n * smem_b_elem; };
    // Scales (e8m0_shuffle, 256 B per 32-row block and K tile): a panel of 2 K tiles x B_M rows is B_M / 32 runs of
    // 512 B; wave w copies blocks 2w, 2w + 1 (waves past B_M / 64 repeat a copy).
    static_assert(T::SCALE_PANEL_TILES == 2 && T::NUM_KGROUPS == 8 && T::SF_WAVES_A <= T::NUM_WAVES && T::SF_WAVES_B <= T::NUM_WAVES);
    const int wave_sfa = wave_id % T::SF_WAVES_A, wave_sfb = wave_id % T::SF_WAVES_B;
    const int sfa_g_lane = (2 * wave_sfa + lane_id / 32) * kargs.stride_sfa + (lane_id % 32) * T::VEC_GLOBAL_SCALE;
    const int sfb_g_lane = (2 * wave_sfb + lane_id / 32) * kargs.stride_sfb + (lane_id % 32) * T::VEC_GLOBAL_SCALE;
    auto gsfa_offset = [&](int panel_k_begin) { return sfa_g_lane + panel_k_begin * 256; };
    auto gsfb_offset = [&](int panel_k_begin) { return sfb_g_lane + panel_k_begin * 256; };

    // Global -> LDS.  Odd issue counts (B_M / B_N = 192): wave w copies chunks w * NI + i, so a chunk pair can straddle
    // two waves and the chunk -> rows map is computed per issue (A chunk c = rows (c / 2) * 16 + 2 * (lane / 8) + c % 2,
    // B chunk c = 16-row block c / 2, K half c % 2).
    constexpr int DMA_CHUNK = T::smem_linear_wave_a + T::smem_padding;
    const int a_kcol = ((lane_id % 8) ^ ((((lane_id / 8) >> 1) & 1) << 1)) * T::VEC_A;
    auto dma_odd_a = [&](auto& g, int lds_off, int s_os) {
        constexpr int NI = T::a_buffer_load_insts;
        const int w = wave_id_n * 2 + wave_id_m;
        char* const base = reinterpret_cast<char*>(s_a.ptr) + lds_off + w * NI * DMA_CHUNK;
        opus::static_for<NI>([&](auto i_) {
            constexpr int i = decltype(i_)::value, io = i * DMA_CHUNK;
            const int c = w * NI + i;
            const int goff = ((c >> 1) * 16 + 2 * (lane_id / 8) + (c & 1)) * kargs.stride_a + a_kcol;
            g.template async_load<T::VEC_A>(base, goff - io, s_os, opus::number<io>{}, opus::number<AB_AUX>{});
        });
    };
    auto dma_odd_b = [&](auto& g, int lds_off, int s_os) {
        constexpr int NI = T::b_buffer_load_insts;
        const int w = wave_id_m * 2 + wave_id_n;
        char* const base = reinterpret_cast<char*>(s_b.ptr) + lds_off + w * NI * DMA_CHUNK;
        opus::static_for<NI>([&](auto i_) {
            constexpr int i = decltype(i_)::value, io = i * DMA_CHUNK;
            const int c = w * NI + i;
            const int goff = (c >> 1) * T::W_N * kargs.stride_b + (c & 1) * 1024 + lane_id * T::VEC_B;
            g.template async_load<T::VEC_B>(base, goff - io, s_os, opus::number<io>{}, opus::number<AB_AUX>{});
        });
    };
    auto prefetch_a = [&](auto& g, int stage, int tile, int half_tile_m) {
        if constexpr (T::A_ODD) dma_odd_a(g, sa_offset(stage, half_tile_m), ga_offset(half_tile_m, tile));
        else async_load_imm<T::VEC_A>(g, s_a.ptr, u_ga, u_sa + sa_offset(stage, half_tile_m), ga_offset(half_tile_m, tile));
    };
    auto prefetch_b = [&](auto& g, int stage, int tile, int half_tile_n) {
        if constexpr (T::B_ODD) dma_odd_b(g, sb_offset(stage, half_tile_n), gb_offset(half_tile_n, tile));
        else async_load_imm<T::VEC_B>(g, s_b.ptr, u_gb, u_sb + sb_offset(stage, half_tile_n), gb_offset(half_tile_n, tile));
    };
    auto dma_sfa = [&](int panel_k_begin, int buf) {
        g_sfa.template async_load<T::VEC_GLOBAL_SCALE>(reinterpret_cast<char*>(s_sfa.ptr) + buf * T::SFA_PANEL_BYTES + wave_sfa * (T::WARP_SIZE * T::VEC_GLOBAL_SCALE), gsfa_offset(panel_k_begin));
    };
    auto dma_sfb = [&](int panel_k_begin, int buf) {
        g_sfb.template async_load<T::VEC_GLOBAL_SCALE>(reinterpret_cast<char*>(s_sfb.ptr) + buf * T::SFB_PANEL_BYTES + wave_sfb * (T::WARP_SIZE * T::VEC_GLOBAL_SCALE), gsfb_offset(panel_k_begin));
    };

    // LDS -> VGPR (A / B operands pinned: A in AGPR 0..63, B in AGPR 64..127).
    const auto ra_offs = opus::layout_to_offsets<T::VEC_A>(u_ra);
    const auto rb_offs = opus::layout_to_offsets<T::VEC_B>(u_rb);
    auto load_a = [&](auto& sa_, auto half_) {
        constexpr int half = decltype(half_)::value;
        opus::static_for<T::E_K>([&](auto kk) { opus::static_for<T::E_M>([&](auto e) {
            constexpr int agpr = 0 + half * 32 + (kk.value * T::E_M + e.value) * 4;
            [[clang::amdgpu_pin_agpr(agpr)]] v_a[half][kk.value][e.value] =
                __builtin_bit_cast(fp4x32_t, load<T::VEC_A>(sa_, ra_offs[e.value] + sa_offset(0, half) + kk.value * T::W_K_BYTES));
        }); });
    };
    auto load_b = [&](auto& sb_, auto half_) {
        constexpr int half = decltype(half_)::value;
        opus::static_for<T::E_K>([&](auto kk) { opus::static_for<T::E_N>([&](auto e) {
            constexpr int agpr = 64 + half * 32 + (kk.value * T::E_N + e.value) * 4;
            [[clang::amdgpu_pin_agpr(agpr)]] v_b[half][kk.value][e.value] =
                __builtin_bit_cast(fp4x32_t, load<T::VEC_B>(sb_, rb_offs[e.value] + sb_offset(0, half) + kk.value * (T::smem_linear_wave_b + T::smem_padding)));
        }); });
    };
    // Scales LDS -> VGPR: load_sf* issues the raw dword reads (in C01); read_sf picks this wave's bytes (in C11 / C00,
    // long after the data landed, so no lgkmcnt wait lands in the main loop).  Half h, repeat e = 32-row block
    // h * E + e of the panel (512 B per block); dword = [kk0 w0, kk0 w1, kk1 w0, kk1 w1], w = wave_id_m (A) / wave_id_n (B).
    using SF_RAW_A = opus::vector_t<opus::u32_t, T::E_M>;
    using SF_RAW_B = opus::vector_t<opus::u32_t, T::E_N>;
    SF_RAW_A v_sfa_raw[2];
    SF_RAW_B v_sfb_raw[2];
    const int sf_rd_base = lane_id * 4;
    const unsigned int sfa_sh = 8u * static_cast<unsigned int>(wave_id_m), sfb_sh = 8u * static_cast<unsigned int>(wave_id_n);
    auto load_sf_raw = [&](auto& rd_, auto h_, auto e_cnt_, int read_off) {
        constexpr int h = decltype(h_)::value, E = decltype(e_cnt_)::value;
        opus::vector_t<opus::u32_t, E> r;
        opus::static_for<E>([&](auto e) {
            r[e.value] = __builtin_bit_cast(unsigned int, load<4>(rd_, sf_rd_base + (h * E + e.value) * 512 + read_off));
        });
        return r;
    };
    auto load_sf = [&](auto& rd_sfa_, auto& rd_sfb_, int buf, int tile_in_panel) {
        const int read_off_a = buf * T::SFA_PANEL_BYTES + tile_in_panel * 256;
        const int read_off_b = buf * T::SFB_PANEL_BYTES + tile_in_panel * 256;
        opus::static_for<2>([&](auto h) { v_sfa_raw[h.value] = load_sf_raw(rd_sfa_, h, opus::number<T::E_M>{}, read_off_a); });
        opus::static_for<2>([&](auto h) { v_sfb_raw[h.value] = load_sf_raw(rd_sfb_, h, opus::number<T::E_N>{}, read_off_b); });
    };
    auto read_sf = [&](auto h_) {
        constexpr int h = decltype(h_)::value;
        auto shr2 = [](unsigned int lo, unsigned int hi, unsigned int sh, unsigned int& o0, unsigned int& o1) {
            const unsigned long long v = ((static_cast<unsigned long long>(hi) << 32) | lo) >> sh;
            o0 = static_cast<unsigned int>(v); o1 = static_cast<unsigned int>(v >> 32);
        };
        opus::static_for<T::E_M / 2>([&](auto q) { shr2(v_sfa_raw[h][2 * q.value], v_sfa_raw[h][2 * q.value + 1], sfa_sh, v_ssa[h][2 * q.value], v_ssa[h][2 * q.value + 1]); });
        opus::static_for<T::E_N / 2>([&](auto q) { shr2(v_sfb_raw[h][2 * q.value], v_sfb_raw[h][2 * q.value + 1], sfb_sh, v_ssb[h][2 * q.value], v_ssb[h][2 * q.value + 1]); });
        if constexpr (T::E_M % 2) v_ssa[h][T::E_M - 1] = v_sfa_raw[h][T::E_M - 1] >> sfa_sh;   // odd E: last dword alone
        if constexpr (T::E_N % 2) v_ssb[h][T::E_N - 1] = v_sfb_raw[h][T::E_N - 1] >> sfb_sh;
    };

    // LDS reads go through __restrict__ copies of the views, passed as function parameters (clang ignores __restrict__
    // on locals): without them the compiler cannot tell the reads from the in-flight LDS-DMA copies and puts a vmcnt(0)
    // before the reads.
    auto views = [&](OPUS_LDS_ADDR char* __restrict__ read_a, OPUS_LDS_ADDR char* __restrict__ read_b,
                     OPUS_LDS_ADDR char* __restrict__ read_sfa, OPUS_LDS_ADDR char* __restrict__ read_sfb, auto&& body) {
        auto rd_a = s_a;     rd_a.ptr = read_a;
        auto rd_b = s_b;     rd_b.ptr = read_b;
        auto rd_sfa = s_sfa; rd_sfa.ptr = read_sfa;
        auto rd_sfb = s_sfb; rd_sfb.ptr = read_sfb;
        body(rd_a, rd_b, rd_sfa, rd_sfb);
    };

    // Quadrant (hm, hn) of C: E_M x E_N fragments x 2 MFMAs (kk = 0, 1) of A half hm and B half hn; zc: the kk = 0 MFMA
    // starts from C = 0.  Kept a lambda: a free function template changes the generated code.
    constexpr int frags = T::E_M * T::E_N;
    auto mma_quadrant = [&](auto hm_, auto hn_, auto zc_) __attribute__((always_inline)) {
        constexpr int hm = decltype(hm_)::value, hn = decltype(hn_)::value;
        opus::static_for<frags>([&](auto i) {
            constexpr int m = decltype(i)::value / T::E_N, n = decltype(i)::value % T::E_N;
            mma_scale_one<0, 0, c_reg<T, hm, hn, m, n>(), decltype(zc_)::value>(mma, v_a[hm][0][m], v_b[hn][0][n], v_c[hm][hn][i.value], v_ssa[hm][m], v_ssb[hn][n]);
            mma_scale_one<2, 2, c_reg<T, hm, hn, m, n>()>(mma, v_a[hm][1][m], v_b[hn][1][n], v_c[hm][hn][i.value], v_ssa[hm][m], v_ssb[hn][n]);
        });
    };
    constexpr auto no_zc = opus::bool_constant<false>{};

    // Per-quadrant instruction counts for the generic schedules (per wave).
    constexpr int QA = T::a_buffer_load_insts, QB = T::b_buffer_load_insts;   // A / B LDS-DMA issues per half tile
    constexpr int RA = T::E_K * T::E_M, RB = T::E_K * T::E_N;                  // A / B ds_read_b128 per half tile
    constexpr int SFR = T::E_M + T::E_N;                                       // scale ds_reads per K tile
    constexpr int SV = (T::E_M + 1) / 2 + (T::E_N + 1) / 2;                    // scale shifts per half
    constexpr int BB_VMCNT = T::a_buffer_load_insts + T::b_buffer_load_insts;  // C10's A0 + B0 copies

    // Prologue: issue K tile 0 + scale panel 0 (+ sfb of panel 1) unless prefetched, then K tile 1; wait for tile 0,
    // read A0 / B0 and the scales of tile 0, and compute C00 of tile 0 while A1 / B1 are read.  The main loop then starts
    // with "C00 of tile t is done".  Persistent: the counted wait lets the previous tile's top-half C stores and tile 1
    // stay in flight.
    if (!prefetched) {
        prefetch_a(g_a, 0, 0, 0);
        prefetch_b(g_b, 0, 0, 0);
        prefetch_a(g_a, 0, 0, 1);
        prefetch_b(g_b, 0, 0, 1);
        dma_sfa(0, 0);
        dma_sfb(0, 0);
        dma_sfb(T::SCALE_PANEL_TILES, 1);
    }
    if (loops > 1) {
        prefetch_a(g_a, 1, 1, 0);
        prefetch_b(g_b, 1, 1, 0);
        prefetch_a(g_a, 1, 1, 1);
        prefetch_b(g_b, 1, 1, 1);
        if (prefetched) s_waitcnt_vmcnt(number<2 * (T::a_buffer_load_insts + T::b_buffer_load_insts) + T::C_COPIES>{});
        else            s_waitcnt_vmcnt(number<2 * (T::a_buffer_load_insts + T::b_buffer_load_insts)>{});
    } else {
        s_waitcnt_vmcnt(0_I);
    }
    views(s_a.ptr, s_b.ptr, s_sfa.ptr, s_sfb.ptr, [&](auto& rd_a, auto& rd_b, auto& rd_sfa, auto& rd_sfb) {
        // A0 / B0 reads go out first; their latency is covered by the scale reads, which need a second barrier.
        sync_threads();
        __builtin_amdgcn_sched_barrier(0);
        load_a(rd_a, number<0>{});
        load_b(rd_b, number<0>{});
        sched_group<SCHED_DS_READ, T::E_K * (T::E_M + T::E_N)>();
        __builtin_amdgcn_sched_barrier(0);
        s_waitcnt_lgkmcnt(0_I);
        sync_threads();
        __builtin_amdgcn_sched_barrier(0);
        load_sf(rd_sfa, rd_sfb, 0, 0);
        read_sf(number<0>{});
        read_sf(number<1>{});
        __builtin_amdgcn_sched_barrier(0);

        // C00(0) + A1, B1 of tile 0 (+ the previous tile's bottom-half C stores).
        load_a(rd_a, number<1>{});
        load_b(rd_b, number<1>{});
        if constexpr (T::PERSIST) store_pend_c(wave_id * T::WARP_SIZE + lane_id);
        mma_quadrant(0_I, 0_I, no_zc);
        sched_quadrant<T, sched_c00_store, RA + RB, 0, 0, H1_COPIES>();
        __builtin_amdgcn_sched_barrier(0);
    });

    // Main loop, two K tiles per trip (stage and panel parity compile-time).  Each K tile t: C10(t), C01(t), C11(t),
    // C00(t+1), with the copies of tile t+2 into stage(t) and the reads of tile t+1 from next_stage.  The first trip is a
    // separate instance whose C10 / C01 / C11 start from C = 0 (ZFIRST).  Scale panel buffers: sf_rd holds the panel of
    // tile t+1, sf_wr = sf_rd + 1 (mod 3) receives the panel copied in K tile t; sf_rd advances after an even t.
    int tile = 0;
    int sf_rd = 0;
    auto main_trip = [&](auto zfirst_) {
        constexpr auto zc = opus::bool_constant<decltype(zfirst_)::value>{};
        {   // even tile
            constexpr int stage = 0, next_stage = 1;
            const int sf_wr = sf_rd == T::SCALE_PANEL_BUFFERS - 1 ? 0 : sf_rd + 1;
            views(s_a.ptr + sa_offset(next_stage, 0), s_b.ptr + sb_offset(next_stage, 0), s_sfa.ptr, s_sfb.ptr,
                  [&](auto& rd_a, auto& rd_b, auto& rd_sfa, auto& rd_sfb) {
                // Every wave is done reading stage(t) (A1 / B1 of tile t in C00): tile t+2 may be copied into it.
                s_waitcnt_lgkmcnt(0_I);
                sync_threads();
                __builtin_amdgcn_sched_barrier(0);

                // C10(t) + A0, B0 of tile t+2 -> stage(t).
                prefetch_a(g_a, stage, tile + 2, 0);
                prefetch_b(g_b, stage, tile + 2, 0);
                mma_quadrant(1_I, 0_I, zc);
                sched_quadrant<T, sched_c10, 0, QA + QB, 0>();
                __builtin_amdgcn_sched_barrier(0);
                // Tile t+1 (and its scale panel) landed; the copies just issued may still fly.
                s_waitcnt_vmcnt(number<BB_VMCNT>{});
                sync_threads();
                __builtin_amdgcn_sched_barrier(0);

                // C01(t) + scales and B0 of tile t+1.
                load_sf(rd_sfa, rd_sfb, sf_rd, 1);
                load_b(rd_b, number<0>{});
                mma_quadrant(0_I, 1_I, zc);
                sched_quadrant<T, sched_c01_nov, RB + SFR, 0, 0>();
                __builtin_amdgcn_sched_barrier(0);
                __builtin_amdgcn_sched_barrier(0);   // empty region (the scale panel copy used to sit here): keeps the schedule

                // C11(t) + A0 of tile t+1 + scales of half 0 + A1 of tile t+2 + sfa of the scale panel starting at t+2.
                load_a(rd_a, number<0>{});
                read_sf(number<0>{});
                prefetch_a(g_a, stage, tile + 2, 1);
                dma_sfa(tile + 2, sf_wr);
                mma_quadrant(1_I, 1_I, zc);
                sched_quadrant<T, sched_c11_pf, RA, QA + 1, SV>();
                __builtin_amdgcn_sched_barrier(0);

                // C00(t+1) + A1, B1 of tile t+1 + scales of half 1 + B1 of tile t+2.
                load_a(rd_a, number<1>{});
                load_b(rd_b, number<1>{});
                read_sf(number<1>{});
                prefetch_b(g_b, stage, tile + 2, 1);
                mma_quadrant(0_I, 0_I, no_zc);
                sched_quadrant<T, sched_c00_pf, RA + RB, QB, SV>();
                __builtin_amdgcn_sched_barrier(0);
            });
            sf_rd = sf_wr;
        }
        {   // odd tile
            const int tile_odd = tile + 1;
            const int tile = tile_odd;
            constexpr int stage = 1, next_stage = 0;
            const int sf_wr = sf_rd == T::SCALE_PANEL_BUFFERS - 1 ? 0 : sf_rd + 1;
            views(s_a.ptr + sa_offset(next_stage, 0), s_b.ptr + sb_offset(next_stage, 0), s_sfa.ptr, s_sfb.ptr,
                  [&](auto& rd_a, auto& rd_b, auto& rd_sfa, auto& rd_sfb) {
                // Every wave is done reading stage(t) (A1 / B1 of tile t in C00): tile t+2 may be copied into it.
                s_waitcnt_lgkmcnt(0_I);
                sync_threads();
                __builtin_amdgcn_sched_barrier(0);

                // C10(t) + A0, B0 of tile t+2 -> stage(t).
                prefetch_a(g_a, stage, tile + 2, 0);
                prefetch_b(g_b, stage, tile + 2, 0);
                mma_quadrant(1_I, 0_I, no_zc);
                sched_quadrant<T, sched_c10, 0, QA + QB, 0>();
                __builtin_amdgcn_sched_barrier(0);
                // Tile t+1 (and its scale panel) landed; the copies just issued may still fly.
                s_waitcnt_vmcnt(number<BB_VMCNT>{});
                sync_threads();
                __builtin_amdgcn_sched_barrier(0);

                // C01(t) + scales and B0 of tile t+1.
                load_sf(rd_sfa, rd_sfb, sf_rd, 0);
                load_b(rd_b, number<0>{});
                mma_quadrant(0_I, 1_I, no_zc);
                sched_quadrant<T, sched_c01_nov, RB + SFR, 0, 0>();
                __builtin_amdgcn_sched_barrier(0);
                __builtin_amdgcn_sched_barrier(0);   // empty region (the scale panel copy used to sit here): keeps the schedule

                // C11(t) + A0 of tile t+1 + scales of half 0 + A1 of tile t+2 + sfb of the scale panel starting at t+3.
                load_a(rd_a, number<0>{});
                read_sf(number<0>{});
                prefetch_a(g_a, stage, tile + 2, 1);
                dma_sfb(tile + 3, sf_wr);
                mma_quadrant(1_I, 1_I, no_zc);
                sched_quadrant<T, sched_c11_pf, RA, QA + 1, SV>();
                __builtin_amdgcn_sched_barrier(0);

                // C00(t+1) + A1, B1 of tile t+1 + scales of half 1 + B1 of tile t+2.
                load_a(rd_a, number<1>{});
                load_b(rd_b, number<1>{});
                read_sf(number<1>{});
                prefetch_b(g_b, stage, tile + 2, 1);
                mma_quadrant(0_I, 0_I, no_zc);
                sched_quadrant<T, sched_c00_pf, RA + RB, QB, SV>();
                __builtin_amdgcn_sched_barrier(0);
            });
        }
    };
    if (tile + 3 < loops) {
        main_trip(opus::bool_constant<true>{});
        tile += 2;
    } else {
        opus::static_for<2>([&](auto hm) { opus::static_for<2>([&](auto hn) { opus::static_for<T::E_M * T::E_N>([&](auto f) {
            if constexpr (hm.value + hn.value > 0) clear(v_c[hm.value][hn.value][f.value]);
        }); }); });
    }
#pragma clang loop unroll(disable)
    for (; tile + 3 < loops; tile += 2) main_trip(no_zc);
    // At most one more steady iteration (odd number of them): runtime stage, all copies in C01.
#pragma clang loop unroll(disable)
    for (; tile + 2 < loops; ++tile) {
        const int stage = tile & 1;
        const int next_stage = stage ^ 1;
        const int sf_wr = sf_rd == T::SCALE_PANEL_BUFFERS - 1 ? 0 : sf_rd + 1;
        views(s_a.ptr + sa_offset(next_stage, 0), s_b.ptr + sb_offset(next_stage, 0), s_sfa.ptr, s_sfb.ptr,
              [&](auto& rd_a, auto& rd_b, auto& rd_sfa, auto& rd_sfb) {
            s_waitcnt_lgkmcnt(0_I);
            sync_threads();
            __builtin_amdgcn_sched_barrier(0);

            // C10(t) + A0, B0 of tile t+2 -> stage(t).
            prefetch_a(g_a, stage, tile + 2, 0);
            prefetch_b(g_b, stage, tile + 2, 0);
            mma_quadrant(1_I, 0_I, no_zc);
            sched_quadrant<T, sched_c10, 0, QA + QB, 0>();
            __builtin_amdgcn_sched_barrier(0);
            s_waitcnt_vmcnt(number<BB_VMCNT>{});
            sync_threads();
            __builtin_amdgcn_sched_barrier(0);

            // C01(t) + scales and B0 of tile t+1 + A1, B1 of tile t+2.
            load_sf(rd_sfa, rd_sfb, sf_rd, (tile + 1) & 1);
            load_b(rd_b, number<0>{});
            prefetch_a(g_a, stage, tile + 2, 1);
            prefetch_b(g_b, stage, tile + 2, 1);
            mma_quadrant(0_I, 1_I, no_zc);
            sched_quadrant<T, sched_c01, RB + SFR, QA + QB, 0>();
            __builtin_amdgcn_sched_barrier(0);

            // Scale panel copy (runtime parity, so a branch on a region boundary).  The two branches end in different
            // (empty) asm so the compiler cannot merge them into one copy that selects between the sfa / sfb layouts
            // (that select goes through scratch, whose vmcnt wait drains every copy in flight).
            if (tile & 1) {
                dma_sfb(tile + 3, sf_wr);
                asm volatile("; panel copy: sfb" ::: "memory");
            } else {
                dma_sfa(tile + 2, sf_wr);
                asm volatile("; panel copy: sfa" ::: "memory");
            }
            __builtin_amdgcn_sched_barrier(0);

            // C11(t) + A0 of tile t+1 + scales of half 0.
            load_a(rd_a, number<0>{});
            read_sf(number<0>{});
            mma_quadrant(1_I, 1_I, no_zc);
            sched_quadrant<T, sched_c11, RA, 1, SV>();
            __builtin_amdgcn_sched_barrier(0);

            // C00(t+1) + A1, B1 of tile t+1 + scales of half 1.
            load_a(rd_a, number<1>{});
            load_b(rd_b, number<1>{});
            read_sf(number<1>{});
            mma_quadrant(0_I, 0_I, no_zc);
            sched_quadrant<T, sched_c00, RA + RB, 0, SV>();
            __builtin_amdgcn_sched_barrier(0);
        });
        sf_rd = (tile & 1) ? sf_rd : sf_wr;
    }

    // Epilogue 1: second-to-last K tile (C10, C01, C11 of it, C00 of the last one), no more copies.
    if (tile + 1 < loops) {
        const int next_stage = (tile & 1) ^ 1;
        views(s_a.ptr + sa_offset(next_stage, 0), s_b.ptr + sb_offset(next_stage, 0), s_sfa.ptr, s_sfb.ptr,
              [&](auto& rd_a, auto& rd_b, auto& rd_sfa, auto& rd_sfb) {
            mma_quadrant(1_I, 0_I, no_zc);
            sched_quadrant<T, sched_c10_last, 0, 0, 0>();
            __builtin_amdgcn_sched_barrier(0);
            s_waitcnt_vmcnt(0_I);
            sync_threads();
            __builtin_amdgcn_sched_barrier(0);

            load_sf(rd_sfa, rd_sfb, sf_rd, (tile + 1) & 1);
            load_b(rd_b, number<0>{});
            mma_quadrant(0_I, 1_I, no_zc);
            sched_quadrant<T, sched_c01_last, RB + SFR, 0, 0>();
            __builtin_amdgcn_sched_barrier(0);

            load_a(rd_a, number<0>{});
            read_sf(number<0>{});
            mma_quadrant(1_I, 1_I, no_zc);
            sched_quadrant<T, sched_c11, RA, 1, SV>();
            __builtin_amdgcn_sched_barrier(0);

            load_a(rd_a, number<1>{});
            load_b(rd_b, number<1>{});
            read_sf(number<1>{});
            mma_quadrant(0_I, 0_I, no_zc);
            sched_quadrant<T, sched_c00, RA + RB, 0, SV>();
            __builtin_amdgcn_sched_barrier(0);
        });
        ++tile;
    }

    // Write-back helpers.  bf16 C is staged through LDS (stage 1, rows padded to B_N + 8 elements) half a tile at a
    // time, as PR 6696 (Output::lds): a wave holds only 16 columns of a row per fragment, so direct fragment stores
    // would leave every 128 B line partially written.
    constexpr int c_lds_stride = T::B_N + 8;
    constexpr int VEC_OUT = 16 / sizeof(D_C);
    static_assert(T::HALF_B_M * c_lds_stride * sizeof(D_C) <= stage_elems * sizeof(D_AB));
    auto s_c = make_smem(reinterpret_cast<D_C*>(smem_ab + stage_elems));
    // Thread position re-derived here (opaque lane id) so the write-back offsets are not kept live (and spilled) from
    // the prologue.
    unsigned int lane_mask_e = ~0u;
    asm volatile("" : "+s"(lane_mask_e));
    const int lane_e = static_cast<int>(__builtin_amdgcn_mbcnt_hi(lane_mask_e, __builtin_amdgcn_mbcnt_lo(lane_mask_e, 0u)));
    const int tid_e = wave_id * T::WARP_SIZE + lane_e;
    auto p_coord_c = opus::make_tuple(wave_id % T::T_M, lane_e % mma.grpn_c, wave_id / T::T_M, lane_e / mma.grpn_c);
    auto u_sc = partition_layout_c<T::VEC_C>(mma, opus::make_tuple(opus::number<c_lds_stride>{}, 1_I), p_coord_c);
    // B_N = 256: rows with bit 3 set store their 8 B pieces with the two halves of each 16 B chunk swapped (the 528 B
    // row stride maps rows r and r + 8 to the same bank); odd reads swap them back in registers.
    const int c_swz = T::C_SWZ ? ((lane_e >> 3) & 1) * (4 - 8 * ((lane_e / mma.grpn_c) & 1)) : 0;
    auto c_unswz = [](auto v, auto i_) {
        if constexpr (!T::C_SWZ || (decltype(i_)::value & 1) == 0) return v;
        else {
            using U4 = opus::vector_t<opus::u32_t, 4>;
            const U4 u = __builtin_bit_cast(U4, v);
            return __builtin_bit_cast(decltype(v), U4{u[2], u[3], u[0], u[1]});
        }
    };
    // Half tile LDS -> global: all LDS reads before the stores (the compiler does not move a ds_read above a
    // buffer_store, so a read / store loop would expose every read latency).
    constexpr int c_copies = T::HALF_B_M * T::B_N / (T::BLOCK_SIZE * VEC_OUT);
    static_assert(T::B_N == 192 || T::BLOCK_SIZE * 8 % T::B_N == 0, "a thread's C column is the same in every copy");
    static_assert(T::B_N != 192 || (c_copies % 3 == 0 && T::HALF_B_M % 32 == 0));
    // B_N = 192: copy i of thread t -> staged row (i / 3) * 32 + t / 8, column (i % 3) * 64 + (t % 8) * 8.
    auto c_row = [&](int t, auto i_) { constexpr int I = decltype(i_)::value; return (I / 3) * 32 + t / 8; };
    auto c_col = [&](int t, auto i_) { constexpr int I = decltype(i_)::value; return (I % 3) * 64 + (t % 8) * 8; };
    auto copy_output_half = [&](int half_tile_m) {
        if constexpr (T::B_N == 192) {
            opus::vector_t<D_C, VEC_OUT> v[c_copies];
            opus::static_for<c_copies>([&](auto i) {
                v[i.value] = load<VEC_OUT>(s_c, c_row(tid_e, i) * c_lds_stride + c_col(tid_e, i));
            });
            opus::static_for<c_copies>([&](auto i) {
                const int r = c_row(tid_e, i) + half_tile_m * T::HALF_B_M, c = c_col(tid_e, i);
                store<VEC_OUT>(g_c, v[i.value], r * kargs.stride_c + c + (c < cols_in ? 0 : C_OOB), 0, opus::number<C_AUX>{});
            });
        } else {
            const int oob = (tid_e * VEC_OUT) % T::B_N < cols_in ? 0 : C_OOB;
            opus::vector_t<D_C, VEC_OUT> v[c_copies];
            opus::static_for<c_copies>([&](auto i) {
                const int linear = tid_e * VEC_OUT + decltype(i)::value * T::BLOCK_SIZE * VEC_OUT;
                v[i.value] = c_unswz(load<VEC_OUT>(s_c, (linear / T::B_N) * c_lds_stride + linear % T::B_N), i);
            });
            opus::static_for<c_copies>([&](auto i) {
                const int linear = tid_e * VEC_OUT + decltype(i)::value * T::BLOCK_SIZE * VEC_OUT;
                const int r = linear / T::B_N + half_tile_m * T::HALF_B_M, c = linear % T::B_N;
                store<VEC_OUT>(g_c, v[i.value], r * kargs.stride_c + c + oob, 0, opus::number<C_AUX>{});
            });
        }
    };
    // fp32 quadrant -> bf16 -> LDS (columns half_tile_n * HALF_B_N of the staged half).
    auto stage_quadrant = [&](const auto& acc, int half_tile_n) {
        typename MMA::vtype_c quad;
        opus::static_for<T::E_M * T::E_N>([&](auto f) {
            opus::set_slice(quad, acc[f.value], opus::number<f.value * MMA::mma_c_len>{}, opus::number<(f.value + 1) * MMA::mma_c_len>{});
        });
        store<T::VEC_C>(s_c, cast<D_C>(quad), u_sc + (half_tile_n * T::HALF_B_N + c_swz));
    };
    auto sync_lds = [&]() {
        s_waitcnt_lgkmcnt(0_I);
        sync_threads();
        __builtin_amdgcn_sched_barrier(0);
    };

    // Epilogue 2: last K tile (C10, C01, C11).  Every LDS read of this output tile is done after the barrier, so the
    // persistent kernel copies K tile 0 and scale panel 0 of the block's next output tile into stage 0 / panel buffer 0
    // now; without a next tile the copies use an empty buffer range (no memory traffic), so no branch is needed.
    {
        const int next_id = tile_id + grid_blocks;
        const bool has_next = next_id < total_tiles;
        const int nid = has_next ? next_id : tile_id;
        const int n_batch = nid / tiles_per_batch;
        const int n_in_batch = nid % tiles_per_batch;
        const int n_row = (n_in_batch / num_tiles_n) * T::B_M;
        const int n_col = (n_in_batch % num_tiles_n) * T::B_N;
        auto ng_a = make_gmem(reinterpret_cast<const D_AB*>(kargs.ptr_a) + n_batch * kargs.stride_a_batch + n_row * kargs.stride_a,
                              has_next ? static_cast<unsigned int>(kargs.m - n_row) * static_cast<unsigned int>(kargs.stride_a) : 0u);
        auto ng_b = make_gmem(reinterpret_cast<const D_AB*>(kargs.ptr_b) + n_batch * kargs.stride_b_batch + n_col * kargs.stride_b,
                              has_next ? static_cast<unsigned int>(kargs.n - n_col) * static_cast<unsigned int>(kargs.stride_b) : 0u);
        auto ng_sfa = make_gmem(reinterpret_cast<const D_SF*>(kargs.ptr_sfa) + n_batch * kargs.stride_sfa_batch + (n_row / 32) * kargs.stride_sfa,
                                has_next ? static_cast<unsigned int>(opus_mxfp4_ceil_div(kargs.m - n_row, 32) * kargs.stride_sfa) : 0u);
        auto ng_sfb = make_gmem(reinterpret_cast<const D_SF*>(kargs.ptr_sfb) + n_batch * kargs.stride_sfb_batch + (n_col / 32) * kargs.stride_sfb,
                                has_next ? static_cast<unsigned int>(opus_mxfp4_ceil_div(kargs.n - n_col, 32) * kargs.stride_sfb) : 0u);
        prefetched = T::PERSIST ? has_next : false;

        s_waitcnt_lgkmcnt(0_I);
        sync_threads();
        __builtin_amdgcn_sched_barrier(0);

        // C10(t) + A0, B0 of the next tile's K tile 0.
        if constexpr (T::PERSIST) {
            prefetch_a(ng_a, 0, 0, 0);
            prefetch_b(ng_b, 0, 0, 0);
        }
        mma_quadrant(1_I, 0_I, no_zc);
        sched_quadrant<T, sched_c10, 0, QA + QB, 0>();
        __builtin_amdgcn_sched_barrier(0);

        // C01(t) + A1, B1 of the next tile's K tile 0.
        if constexpr (T::PERSIST) {
            prefetch_a(ng_a, 0, 0, 1);
            prefetch_b(ng_b, 0, 0, 1);
        }
        mma_quadrant(0_I, 1_I, no_zc);
        sched_quadrant<T, sched_c10, 0, QA + QB, 0>();
        __builtin_amdgcn_sched_barrier(0);

        // C11(t) + scale panel 0 (+ sfb of panel 1) of the next tile (lane offsets re-derived from an opaque lane id
        // so the per-tile ones need not stay live).
        if constexpr (T::PERSIST) {
            unsigned int lane_mask_n = ~0u;
            asm volatile("" : "+s"(lane_mask_n));
            const int ln = static_cast<int>(__builtin_amdgcn_mbcnt_hi(lane_mask_n, __builtin_amdgcn_mbcnt_lo(lane_mask_n, 0u)));
            const int ws_a = wave_id % T::SF_WAVES_A, ws_b = wave_id % T::SF_WAVES_B;
            const int blk = 2 * ws_a + ln / 32, blk_b = 2 * ws_b + ln / 32, in_blk = (ln % 32) * T::VEC_GLOBAL_SCALE;
            char* const sfa_dst = reinterpret_cast<char*>(s_sfa.ptr) + ws_a * (T::WARP_SIZE * T::VEC_GLOBAL_SCALE);
            char* const sfb_dst = reinterpret_cast<char*>(s_sfb.ptr) + ws_b * (T::WARP_SIZE * T::VEC_GLOBAL_SCALE);
            ng_sfa.template async_load<T::VEC_GLOBAL_SCALE>(sfa_dst, blk * kargs.stride_sfa + in_blk);
            ng_sfb.template async_load<T::VEC_GLOBAL_SCALE>(sfb_dst, blk_b * kargs.stride_sfb + in_blk);
            ng_sfb.template async_load<T::VEC_GLOBAL_SCALE>(sfb_dst + T::SFB_PANEL_BYTES, blk_b * kargs.stride_sfb + in_blk + T::SCALE_PANEL_TILES * 256);
        }
        mma_quadrant(1_I, 1_I, no_zc);
        sched_quadrant<T, sched_c11_next, 0, 3, 0>();
        __builtin_amdgcn_sched_barrier(0);
    }

    // Split-K, S > 1: store this segment's fp32 partial tile (C fragment layout, row stride B_N) to the workspace; the
    // last block to arrive for this tile (atomic counter, reset by that block) adds the other S - 1 partials and writes
    // C below.  kargs.split_k < 0: partials only, gemm_a4w4_mxfp4_nt_sk_reduce sums them.
    if constexpr (T::SPLITK) {
        int* const sk_last_p = reinterpret_cast<int*>(smem_sfa);   // broadcast flag (scale panels are no longer read)
        const bool sk_separate = kargs.split_k < 0;
        if (split_k > 1 || sk_separate) {
            const long tile_elems = static_cast<long>(T::B_M) * T::B_N;
            float* const ws_tiles = reinterpret_cast<float*>(reinterpret_cast<char*>(kargs.ptr_ws) + opus_mxfp4_sk_counter_bytes(total_tiles));
            int* const counters = reinterpret_cast<int*>(kargs.ptr_ws);
            auto u_wc = partition_layout_c<T::VEC_C>(mma, opus::make_tuple(opus::number<T::B_N>{}, 1_I), p_coord_c);
            auto part = [&](int s_) { return make_gmem(ws_tiles + (static_cast<long>(tile_id) * split_k + s_) * tile_elems,
                                                       static_cast<unsigned int>(tile_elems * 4)); };
            auto g_own = part(k_split);
            const auto wc_offs = opus::layout_to_offsets<T::VEC_C>(u_wc);
            static_assert(MMA::mma_c_len == T::VEC_C);
            auto wc_offset = [&](auto hm, auto hn, auto f) {
                return wc_offs[f.value] + (hm.value * T::HALF_B_M * T::B_N + hn.value * T::HALF_B_N);
            };
            if (sk_separate) {   // the reduce kernel runs after this one: plain (L2-cached) stores
                opus::static_for<2>([&](auto hm) { opus::static_for<2>([&](auto hn) { opus::static_for<T::E_M * T::E_N>([&](auto f) {
                    store<T::VEC_C>(g_own, v_c[hm.value][hn.value][f.value], wc_offset(hm, hn, f));
                }); }); });
                return;
            }
            opus::static_for<2>([&](auto hm) { opus::static_for<2>([&](auto hn) { opus::static_for<T::E_M * T::E_N>([&](auto f) {
                store<T::VEC_C>(g_own, v_c[hm.value][hn.value][f.value], wc_offset(hm, hn, f), 0, opus::number<SK_AUX>{});
            }); }); });
            // No agent-scope fences (on gfx950 they write back / invalidate the whole L2): the partials use system-coherent
            // stores / loads that bypass the non-coherent L2, so vmcnt(0) before the relaxed device-scope counter is enough.
            s_waitcnt_vmcnt(0_I);
            sync_threads();
            if (thread_id_x() == 0) {
                const int arrived = __scoped_atomic_fetch_add(&counters[tile_id], 1, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE);
                const int last = arrived == split_k - 1;
                if (last) __scoped_atomic_store_n(&counters[tile_id], 0, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE);
                *sk_last_p = last;
            }
            s_waitcnt_lgkmcnt(0_I);
            sync_threads();
            if (!*sk_last_p) return;
            for (int s2 = 0; s2 < split_k; ++s2) {
                if (s2 == k_split) continue;
                auto g_p = part(s2);
                opus::static_for<2>([&](auto hm) { opus::static_for<2>([&](auto hn) { opus::static_for<T::E_M * T::E_N>([&](auto f) {
                    const auto q = load<T::VEC_C>(g_p, wc_offset(hm, hn, f), 0, opus::number<SK_AUX>{});
                    opus::static_for<MMA::mma_c_len>([&](auto j) { v_c[hm.value][hn.value][f.value][j.value] += q[j.value]; });
                }); }); });
            }
        }
    }

    // Write back C: top half straight out; bottom half straight out (one tile per block) or into registers, stored in
    // the next tile's prologue (persistent).  No vmcnt waits: the next tile's K tile 0 copies are in flight.
    stage_quadrant(v_c[0][0], 0);
    stage_quadrant(v_c[0][1], 1);
    sync_lds();
    copy_output_half(0);
    sync_lds();
    stage_quadrant(v_c[1][0], 0);
    stage_quadrant(v_c[1][1], 1);
    sync_lds();
    if constexpr (!T::PERSIST) {
        copy_output_half(1);
    } else {
        opus::static_for<c_copies>([&](auto i) {
            if constexpr (T::B_N == 192) {
                v_pend_c[i.value] = load<VEC_OUT>(s_c, c_row(tid_e, i) * c_lds_stride + c_col(tid_e, i));
            } else {
                const int linear = tid_e * VEC_OUT + decltype(i)::value * T::BLOCK_SIZE * VEC_OUT;
                v_pend_c[i.value] = c_unswz(load<VEC_OUT>(s_c, (linear / T::B_N) * c_lds_stride + linear % T::B_N), i);
            }
        });
        pend_c_ptr = reinterpret_cast<D_C*>(kargs.ptr_c) + batch_id * kargs.stride_c_batch + row * kargs.stride_c + col;
        pend_c = 1;
        pend_rows = rows_in;
        pend_cols = cols_in;
    }
    // The next tile's prologue copies overwrite the staged C: every wave must have finished reading it.
    s_waitcnt_lgkmcnt(0_I);
    sync_threads();
    __builtin_amdgcn_sched_barrier(0);
    }   // tile loop
    if (T::PERSIST && pend_c) {   // the last tile's bottom half
        unsigned int lane_mask_l = ~0u;
        asm volatile("" : "+s"(lane_mask_l));
        const int ln = static_cast<int>(__builtin_amdgcn_mbcnt_hi(lane_mask_l, __builtin_amdgcn_mbcnt_lo(lane_mask_l, 0u)));
        s_waitcnt_lgkmcnt(0_I);
        store_pend_c(wave_id * T::WARP_SIZE + ln);
    }
#endif   // __gfx950__
}

template<class Traits>
__global__ __launch_bounds__(Traits::BLOCK_SIZE, 1)
void gemm_a4w4_mxfp4_nt_kernel(opus_gemm_mxfp4_scale_kargs_gfx950 kargs) {
    gemm_a4w4_mxfp4_nt_body<Traits>(kargs);
}

// Split-K reduce for kargs.split_k < 0: C[tile] = bf16(sum_s partial[tile][s]) with the M / N boundary.  One block of 256
// threads per (tile, row slab); each thread owns 8 consecutive columns of one row and loads 2 x 16 B per segment.
template<class Traits>
__device__ __forceinline__ void gemm_a4w4_mxfp4_nt_sk_reduce_body(opus_gemm_mxfp4_scale_kargs_gfx950 kargs) {
#if defined(__gfx950__)
    using T = opus::remove_cvref_t<Traits>;
    const int S = kargs.split_k < 0 ? -kargs.split_k : kargs.split_k;
    const int num_tiles_n = opus_mxfp4_ceil_div(kargs.n, T::B_N);
    const int tiles_per_batch = opus_mxfp4_ceil_div(kargs.m, T::B_M) * num_tiles_n;
    const int total_tiles = tiles_per_batch * kargs.batch;
    constexpr int COLS8 = T::B_N / 8;                         // threads per row
    constexpr int ROWS = T::BLOCK_SIZE / COLS8;               // rows per block (B_N 256: 8, 192: 10 with 240 threads, 128: 16)
    constexpr int SLABS = (T::B_M + ROWS - 1) / ROWS;
    const int bid = static_cast<int>(opus::block_id_x());
    const int tile_id = bid / SLABS, slab = bid % SLABS;
    const int tid = static_cast<int>(opus::thread_id_x());
    const int r = slab * ROWS + tid / COLS8, c = (tid % COLS8) * 8;
    if (tid >= ROWS * COLS8 || r >= T::B_M || tile_id >= total_tiles) return;
    const int batch_id = tile_id / tiles_per_batch, tib = tile_id % tiles_per_batch;
    const int row = (tib / num_tiles_n) * T::B_M + r, col = (tib % num_tiles_n) * T::B_N + c;
    if (row >= kargs.m || col >= kargs.n) return;
    const long tile_elems = static_cast<long>(T::B_M) * T::B_N;
    // Buffer base per tile (wave-uniform: a lane-dependent base would turn every load into a readfirstlane waterfall loop).
    const float* ws = reinterpret_cast<const float*>(reinterpret_cast<const char*>(kargs.ptr_ws) + opus_mxfp4_sk_counter_bytes(total_tiles))
                      + static_cast<long>(tile_id) * S * tile_elems;
    auto g = opus::make_gmem(ws, static_cast<unsigned int>(S * tile_elems * 4));
    const int lane_off = r * T::B_N + c;
    opus::vector_t<float, 8> acc;
    opus::clear(acc);
    for (int s = 0; s < S; ++s) {
        const auto lo = opus::load<4>(g, lane_off, static_cast<int>(s * tile_elems));
        const auto hi = opus::load<4>(g, lane_off + 4, static_cast<int>(s * tile_elems));
        opus::static_for<4>([&](auto j) { acc[j.value] += lo[j.value]; acc[4 + j.value] += hi[j.value]; });
    }
    opus::vector_t<opus::bf16_t, 8> out;
    opus::static_for<8>([&](auto j) { out[j.value] = static_cast<opus::bf16_t>(acc[j.value]); });
    auto gc = opus::make_gmem(reinterpret_cast<opus::bf16_t*>(kargs.ptr_c) + batch_id * kargs.stride_c_batch);
    opus::store<8>(gc, out, row * kargs.stride_c + col);
#endif
}

template<class Traits>
__global__ __launch_bounds__(Traits::BLOCK_SIZE, 1)
void gemm_a4w4_mxfp4_nt_sk_reduce(opus_gemm_mxfp4_scale_kargs_gfx950 kargs) {
    gemm_a4w4_mxfp4_nt_sk_reduce_body<Traits>(kargs);
}
