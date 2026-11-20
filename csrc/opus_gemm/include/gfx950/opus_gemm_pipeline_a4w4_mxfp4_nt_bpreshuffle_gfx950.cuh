#pragma once
// MXFP4 x MXFP4 NT 2x2 4-wave pipeline, aiter bpreshuffle inputs (A row-major, B shuffle_weight(16, 16), e8m0_shuffle scales).

#include <opus/hip_minimal.hpp>
#include <opus/opus.hpp>

#include "opus_gemm_traits_a4w4_mxfp4_nt_bpreshuffle_gfx950.cuh"

using opus::operator""_I;

// Everything below lives in its own namespace (opus_mxfp4_bpreshuffle).
namespace opus_mxfp4_bpreshuffle {

// Create layout for loading A matrix from global memory (LDS-DMA, 16 B pieces XOR-swizzled per row pair).
template<class T>
__device__ inline auto make_layout_ga(int lane_id, int wave_id_m, int wave_id_n, int stride_a) {
    constexpr int threads_k = T::B_K_BYTES / T::VEC_A;

    constexpr auto ga_block_shape = opus::make_tuple(
        opus::number<T::T_N>{},
        opus::number<T::T_M>{},
        opus::number<T::a_buffer_load_insts / T::ROW_PAIR>{},
        opus::number<T::WARP_SIZE / threads_k>{},
        opus::number<T::ROW_PAIR>{},
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
        opus::number<T::b_buffer_load_insts / T::ROW_PAIR>{},
        opus::number<T::ROW_PAIR>{},
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

// Scales (e8m0_shuffle): a 32-row block holds 256 B per K tile; lane l of the block's 256 B is the MFMA lane l
// ([kk0 r, kk0 r + 16, kk1 r, kk1 r + 16], r = l % 16), so panels are copied global -> LDS verbatim.
// Create layout for loading scale factors from global memory: wave w (w < SF_WAVES) copies 32-row blocks 2 w, 2 w + 1
// of a panel (2 K tiles = 512 B per block), 16 B per lane.
template<class T>
__device__ inline auto make_layout_gsf(int lane_id, int wave_id_sf, int stride_sf) {
    constexpr int lanes_block = T::SCALE_PANEL_TILES * 256 / T::VEC_GLOBAL_SCALE;   // 32 lanes per 512 B block

    constexpr auto gsf_block_shape = opus::make_tuple(
        opus::number<T::NUM_WAVES>{},
        opus::number<T::WARP_SIZE / lanes_block>{},
        opus::number<lanes_block>{},
        opus::number<T::VEC_GLOBAL_SCALE>{});

    constexpr auto gsf_block_dim = opus::make_tuple(
        opus::make_tuple(opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}));

    return opus::make_layout<T::VEC_GLOBAL_SCALE>(
        gsf_block_shape,
        opus::unfold_x_stride(gsf_block_dim, gsf_block_shape, opus::tuple{stride_sf, 1_I}),
        opus::unfold_p_coord(gsf_block_dim, opus::tuple{wave_id_sf, lane_id / lanes_block, lane_id % lanes_block}));
}
template<class T> __device__ inline auto make_layout_gsfa(int lane_id, int wave_id, int stride_sfa) { return make_layout_gsf<T>(lane_id, wave_id % T::SF_WAVES_A, stride_sfa); }
template<class T> __device__ inline auto make_layout_gsfb(int lane_id, int wave_id, int stride_sfb) { return make_layout_gsf<T>(lane_id, wave_id % T::SF_WAVES_B, stride_sfb); }

// Create layout for storing scale factors to shared memory: the panel image is [block][512 B], so wave w's 1 KB copy
// lands at w * 1 KB.
template<class T>
__device__ inline auto make_layout_ssf(int wave_id_sf) {
    constexpr auto ssf_block_shape = opus::make_tuple(
        opus::number<T::NUM_WAVES>{},
        opus::number<T::VEC_GLOBAL_SCALE>{});

    constexpr auto ssf_block_dim = opus::make_tuple(
        opus::make_tuple(opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}));

    return opus::make_layout(
        ssf_block_shape,
        opus::unfold_x_stride(ssf_block_dim, ssf_block_shape, opus::tuple{opus::number<T::WARP_SIZE * T::VEC_GLOBAL_SCALE>{}, 1_I}),
        opus::unfold_p_coord(ssf_block_dim, opus::tuple{wave_id_sf}));
}
template<class T> __device__ inline auto make_layout_ssfa(int wave_id) { return make_layout_ssf<T>(wave_id % T::SF_WAVES_A); }
template<class T> __device__ inline auto make_layout_ssfb(int wave_id) { return make_layout_ssf<T>(wave_id % T::SF_WAVES_B); }

// Create layout for reading scales from shared memory to registers: lane l reads dword l of 32-row block h * E + e of
// the panel (the block's [kk0 w0, kk0 w1, kk1 w0, kk1 w1] for row l % 16); offsets[h * E + e].  The K tile of the panel
// (256 B) is added at the read.
template<class T, int E>
__device__ inline auto make_layout_rsf(int lane_id) {
    constexpr auto rsf_block_shape = opus::make_tuple(
        opus::number<T::HALVES>{},
        opus::number<E>{},
        opus::number<T::WARP_SIZE>{},
        opus::number<4>{});

    constexpr auto rsf_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::y_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}));

    return opus::make_layout<4>(
        rsf_block_shape,
        opus::unfold_x_stride(rsf_block_dim, rsf_block_shape, opus::tuple{opus::number<T::SCALE_PANEL_TILES * 256>{}, 1_I}),
        opus::unfold_p_coord(rsf_block_dim, opus::tuple{lane_id}));
}
template<class T> __device__ inline auto make_layout_rsfa(int lane_id) { return make_layout_rsf<T, T::E_M>(lane_id); }
template<class T> __device__ inline auto make_layout_rsfb(int lane_id) { return make_layout_rsf<T, T::E_N>(lane_id); }

// Like opus async_load(layout), but the per-issue LDS offsets go into the MUBUF immediate (one LDS base in m0); the
// immediate also shifts the global address, so it is subtracted there.  opus async_load(layout) writes m0 per issue
// and keeps one LDS address per issue live in SGPRs.
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
constexpr int SCHED_VALU = 0x002;
constexpr int SCHED_MFMA = 0x008;
constexpr int SCHED_VMEM_READ = 0x820;   // 0x020 VMEM read | 0x800 LDS DMA (newer LLVM classifies buffer_load ... lds as LDS DMA)
constexpr int SCHED_VMEM_WRITE = 0x040;
constexpr int SCHED_DS_READ = 0x100;

template<int MASK, int CNT = 1>
__device__ inline void sched_group() {
    if constexpr (CNT > 0) __builtin_amdgcn_sched_group_barrier(MASK, CNT, 0);
}

// Quadrant schedule: the Q = 2 * E_M * E_N MFMAs of a quadrant with its DS reads / VMEM reads / VALU / VMEM writes
// spread evenly between them.
template<class T, int DS, int VMEM, int VALU, int STORE = 0>
__device__ inline void sched_quadrant() {
    constexpr int Q = 2 * T::E_M * T::E_N;
    opus::static_for<Q>([&](auto i_) {
        constexpr int i = decltype(i_)::value;
        sched_group<SCHED_MFMA>();
        sched_group<SCHED_DS_READ, DS * (i + 1) / Q - DS * i / Q>();
        sched_group<SCHED_VMEM_READ, VMEM * (i + 1) / Q - VMEM * i / Q>();
        sched_group<SCHED_VALU, VALU * (i + 1) / Q - VALU * i / Q>();
        sched_group<SCHED_VMEM_WRITE, STORE * (i + 1) / Q - STORE * i / Q>();
    });
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

// Quadrant (HM, HN) of C: E_M x E_N fragments x 2 MFMAs (kk = 0, 1) of A half HM and B half HN; ZC: the kk = 0 MFMA
// starts from C = 0.  Scale dword e of half h holds repeat e; opsel 2 kk picks the byte of K tile half kk.
template<class T, int HM, int HN, bool ZC = false, class MMA, class CF>
__device__ __forceinline__ void mma_quadrant(MMA& mma, const fp4x32_t (&v_a)[2][T::E_K][T::E_M], const fp4x32_t (&v_b)[2][T::E_K][T::E_N],
                                             CF (&v_c)[2][2][T::E_M * T::E_N], const opus::u32_t (&v_sfa)[2][T::E_M], const opus::u32_t (&v_sfb)[2][T::E_N]) {
    opus::static_for<T::E_M * T::E_N>([&](auto i) {
        constexpr int m = decltype(i)::value / T::E_N, n = decltype(i)::value % T::E_N;
        mma_scale_one<0, 0, c_reg<T, HM, HN, m, n>(), ZC>(mma, v_a[HM][0][m], v_b[HN][0][n], v_c[HM][HN][i.value], v_sfa[HM][m], v_sfb[HN][n]);
        mma_scale_one<2, 2, c_reg<T, HM, HN, m, n>()>(mma, v_a[HM][1][m], v_b[HN][1][n], v_c[HM][HN][i.value], v_sfa[HM][m], v_sfb[HN][n]);
    });
}

// ===========================================================================
// Kernel
// ===========================================================================
// Cache policy of the C stores: nt (non-temporal).  C is written once and never re-read, so it should not evict the
// A / B tiles other blocks still read from L2; A / B keep the default policy.
constexpr int C_AUX = 2;
// Cache policy of the split-K partial stores / loads: sc0 sc1 (system coherent, bypass the non-coherent L2).
constexpr int SK_AUX = 17;
// Kernel body as a device function: it backs the template __global__ below and the extern "C" entry points of the
// pre-compiled .co images (gen_co/build_co.py --arch gfx950).
template<class Traits>
__device__ __forceinline__ void gemm_a4w4_mxfp4_nt_bpreshuffle_body(opus_gemm_mxfp4_bpreshuffle_kargs_gfx950 kargs) {
#if defined(__gfx950__)
    using namespace opus;

    using T = opus::remove_cvref_t<Traits>;
    using D_A = opus::fp4_t;
    using D_B = opus::fp4_t;
    using D_ACC = opus::fp32_t;
    using D_C = opus::bf16_t;
    using D_AB = opus::u8_t;     // fp4x2 bytes
    using D_SF = opus::e8m0_t;

    const int num_tiles_n = opus_mxfp4_bpreshuffle_ceil_div(kargs.n, T::B_N);
    const int wave_id = __builtin_amdgcn_readfirstlane(thread_id_x() / T::WARP_SIZE);
    const int tiles_per_batch = opus_mxfp4_bpreshuffle_ceil_div(kargs.m, T::B_M) * num_tiles_n;
    const int total_tiles = tiles_per_batch * kargs.batch;
    // Persistent: block b runs tiles b, b + grid_blocks, ...; one tile (split-K: tile x K segment) per block otherwise.
    const int split_k = T::SPLITK ? kargs.split_k : 1;
    const int grid_blocks = T::PERSIST ? kargs.grid_blocks : total_tiles;
    const int first_tile = T::SPLITK ? static_cast<int>(block_id_x()) / split_k : static_cast<int>(block_id_x());
    const int k_split = T::SPLITK ? static_cast<int>(block_id_x()) % split_k : 0;

    // Persistent: K tile 0 (+ scale panel 0) of the next output tile is loaded in the last K tile of the current one,
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
    const int block_m = tile_in_batch / num_tiles_n;
    const int block_n = tile_in_batch % num_tiles_n;
    const int row = block_m * T::B_M;
    const int col = block_n * T::B_N;
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
    // dropped); C columns past N are masked per 16 B copy.  The tile base is 64-bit and every range spans this tile
    // only (rows_in <= B_M, cols_in <= B_N), so neither the base nor a 32-bit range overflows on a > 4 GiB operand.
    const int rows_in = T::BOUNDARY ? opus_mxfp4_bpreshuffle_min(kargs.m - row, T::B_M) : T::B_M;
    const int cols_in = T::BOUNDARY ? opus_mxfp4_bpreshuffle_min(kargs.n - col, T::B_N) : T::B_N;
    const int a_k0 = kt0 * T::B_K_BYTES, b_k0 = kt0 * T::B_K_BYTES * T::W_N, sf_k0 = kt0 * 256;
    auto g_a = make_gmem(reinterpret_cast<const D_AB*>(kargs.ptr_a) + batch_id * kargs.stride_a_batch + static_cast<long long>(row) * kargs.stride_a + a_k0,
                         T::BOUNDARY ? static_cast<unsigned int>(rows_in) * static_cast<unsigned int>(kargs.stride_a) - a_k0 : 0xffffffffu);
    auto g_b = make_gmem(reinterpret_cast<const D_AB*>(kargs.ptr_b) + batch_id * kargs.stride_b_batch + static_cast<long long>(col) * kargs.stride_b + b_k0,
                         T::BOUNDARY ? static_cast<unsigned int>(cols_in) * static_cast<unsigned int>(kargs.stride_b) - b_k0 : 0xffffffffu);
    auto g_c = make_gmem(reinterpret_cast<D_C*>(kargs.ptr_c) + batch_id * kargs.stride_c_batch + static_cast<long long>(row) * kargs.stride_c + col,
                         T::BOUNDARY ? static_cast<unsigned int>(rows_in) * static_cast<unsigned int>(kargs.stride_c) * 2u : 0xffffffffu);
    // e8m0_shuffle scales: the tile's 32-row blocks; blocks past the matrix are outside the range (read as 0).
    auto g_sfa = make_gmem(reinterpret_cast<const D_SF*>(kargs.ptr_sfa) + batch_id * kargs.stride_sfa_batch + static_cast<long long>(row / 32) * kargs.stride_sfa + sf_k0,
                           static_cast<unsigned int>(opus_mxfp4_bpreshuffle_ceil_div(rows_in, 32) * kargs.stride_sfa - sf_k0));
    auto g_sfb = make_gmem(reinterpret_cast<const D_SF*>(kargs.ptr_sfb) + batch_id * kargs.stride_sfb_batch + static_cast<long long>(col / 32) * kargs.stride_sfb + sf_k0,
                           static_cast<unsigned int>(opus_mxfp4_bpreshuffle_ceil_div(cols_in, 32) * kargs.stride_sfb - sf_k0));

    const auto u_ga = make_layout_ga<T>(lane_id, wave_id_m, wave_id_n, kargs.stride_a);
    const auto u_gb = make_layout_gb<T>(lane_id, wave_id_m, wave_id_n, kargs.stride_b);
    const auto u_sa = make_layout_sa<T>(wave_id_m, wave_id_n);
    const auto u_sb = make_layout_sb<T>(wave_id_m, wave_id_n);
    const auto u_ra = make_layout_ra<T>(lane_id, wave_id_m);
    const auto u_rb = make_layout_rb<T>(lane_id, wave_id_n);
    const auto u_gsfa = make_layout_gsfa<T>(lane_id, wave_id, kargs.stride_sfa);
    const auto u_gsfb = make_layout_gsfb<T>(lane_id, wave_id, kargs.stride_sfb);
    const auto u_ssfa = make_layout_ssfa<T>(wave_id);
    const auto u_ssfb = make_layout_ssfb<T>(wave_id);
    const auto u_rsfa = make_layout_rsfa<T>(lane_id);
    const auto u_rsfb = make_layout_rsfb<T>(lane_id);

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

    // Per-half scale dwords of the MFMAs: dword e holds repeat e, byte 2 kk + w (opsel 2 kk after the wave shift).
    using D_SF_PACK = opus::u32_t;
    D_SF_PACK v_sfa[2][T::E_M];
    D_SF_PACK v_sfb[2][T::E_N];

    auto ga_offset = [&](int half_tile_m, int tile_k) { return half_tile_m * T::HALF_B_M * kargs.stride_a + tile_k * T::B_K_BYTES; };
    auto gb_offset = [&](int half_tile_n, int tile_k) { return half_tile_n * T::HALF_B_N * kargs.stride_b + tile_k * (T::B_K_BYTES * T::W_N); };
    constexpr int stage_elems = 2 * smem_a_elem + 2 * smem_b_elem;
    auto sa_offset = [&](int stage, int half_tile_m) { return stage * stage_elems + half_tile_m * smem_a_elem; };
    auto sb_offset = [&](int stage, int half_tile_n) { return stage * stage_elems + half_tile_n * smem_b_elem; };
    static_assert(T::SCALE_PANEL_TILES == 2 && T::NUM_KGROUPS == 8 && T::SF_WAVES_A <= T::NUM_WAVES && T::SF_WAVES_B <= T::NUM_WAVES);
    auto gsf_offset = [&](int panel_k_begin) { return panel_k_begin * 256; };
    auto ssfa_offset = [&](int buf) { return buf * T::SFA_PANEL_BYTES; };
    auto ssfb_offset = [&](int buf) { return buf * T::SFB_PANEL_BYTES; };

    // Global -> LDS.  A chunk is 8 rows x 128 B (8 lanes x 16 B per row); chunk pair (2 c, 2 c + 1) interleaves into the
    // 16-row block c: chunk c holds rows (c / 2) * 16 + 2 * (lane / 8) + c % 2.  A B chunk (shuffle_weight) is K half c % 2
    // of the 16-row block c / 2 (1 KB, lane l at + 16 l).  The layout path (async_load_imm) needs whole chunk pairs per
    // wave; with B_M / B_N = 192 (A_SPLIT_PAIR / B_SPLIT_PAIR) wave w copies chunks w * NI + i, a pair straddles two
    // waves, and async_load_rows / async_load_blocks compute each issue's chunk directly.
    constexpr int CHUNK_BYTES = T::smem_linear_wave_a + T::smem_padding;   // LDS bytes of one chunk (incl. padding)
    constexpr int threads_k = T::B_K_BYTES / T::VEC_A;   // 8 lanes per 128-byte row
    const int a_kcol = ((lane_id % threads_k) ^ ((((lane_id / threads_k) >> 1) & 1) << 1)) * T::VEC_A;
    auto async_load_rows = [&](auto& g, int lds_off, int s_os) {
        constexpr int NI = T::a_buffer_load_insts;
        const int w = wave_id_n * 2 + wave_id_m;
        char* const base = reinterpret_cast<char*>(s_a.ptr) + lds_off + w * NI * CHUNK_BYTES;
        opus::static_for<NI>([&](auto i_) {
            constexpr int i = decltype(i_)::value, io = i * CHUNK_BYTES;
            const int c = w * NI + i;
            const int goff = ((c >> 1) * T::W_M + T::ROW_PAIR * (lane_id / threads_k) + (c & 1)) * kargs.stride_a + a_kcol;
            g.template async_load<T::VEC_A>(base, goff - io, s_os, opus::number<io>{});
        });
    };
    auto async_load_blocks = [&](auto& g, int lds_off, int s_os) {
        constexpr int NI = T::b_buffer_load_insts;
        const int w = wave_id_m * 2 + wave_id_n;
        char* const base = reinterpret_cast<char*>(s_b.ptr) + lds_off + w * NI * CHUNK_BYTES;
        opus::static_for<NI>([&](auto i_) {
            constexpr int i = decltype(i_)::value, io = i * CHUNK_BYTES;
            const int c = w * NI + i;
            const int goff = (c >> 1) * T::W_N * kargs.stride_b + (c & 1) * T::smem_linear_wave_b + lane_id * T::VEC_B;
            g.template async_load<T::VEC_B>(base, goff - io, s_os, opus::number<io>{});
        });
    };
    auto prefetch_a = [&](auto& g, int stage, int tile, int half_tile_m) {
        if constexpr (T::A_SPLIT_PAIR) async_load_rows(g, sa_offset(stage, half_tile_m), ga_offset(half_tile_m, tile));
        else async_load_imm<T::VEC_A>(g, s_a.ptr, u_ga, u_sa + sa_offset(stage, half_tile_m), ga_offset(half_tile_m, tile));
    };
    auto prefetch_b = [&](auto& g, int stage, int tile, int half_tile_n) {
        if constexpr (T::B_SPLIT_PAIR) async_load_blocks(g, sb_offset(stage, half_tile_n), gb_offset(half_tile_n, tile));
        else async_load_imm<T::VEC_B>(g, s_b.ptr, u_gb, u_sb + sb_offset(stage, half_tile_n), gb_offset(half_tile_n, tile));
    };
    // Scales: global -> LDS verbatim (LDS-DMA), one panel (2 K tiles) per copy.
    auto prefetch_sfa = [&](int panel_k_begin, int buf) {
        async_load<T::VEC_GLOBAL_SCALE>(g_sfa, s_sfa.ptr, u_gsfa, u_ssfa + ssfa_offset(buf), gsf_offset(panel_k_begin));
    };
    auto prefetch_sfb = [&](int panel_k_begin, int buf) {
        async_load<T::VEC_GLOBAL_SCALE>(g_sfb, s_sfb.ptr, u_gsfb, u_ssfb + ssfb_offset(buf), gsf_offset(panel_k_begin));
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
    // Scales LDS -> VGPR, issued early (C01) into v_sf*_next; read_sf shifts this wave's bytes of half h into v_sfa /
    // v_sfb (in C11 / C00, long after the data landed, so no lgkmcnt wait lands in the main loop).  Byte 2 kk + w of the
    // dword is K tile half kk of this wave's 16 rows (w = wave_id_m (A) / wave_id_n (B)): a shift by 8 w leaves it at the
    // opsel 2 kk byte.
    using SF_E_A = opus::vector_t<opus::u32_t, T::E_M>;
    using SF_E_B = opus::vector_t<opus::u32_t, T::E_N>;
    SF_E_A v_sfa_next[2];
    SF_E_B v_sfb_next[2];
    const auto rsfa_offs = opus::layout_to_offsets<4>(u_rsfa);   // [h * E_M + e]
    const auto rsfb_offs = opus::layout_to_offsets<4>(u_rsfb);
    const unsigned int sfa_sh = 8u * static_cast<unsigned int>(wave_id_m), sfb_sh = 8u * static_cast<unsigned int>(wave_id_n);
    auto load_sfa = [&](auto& rd_sfa_, int buf, int tile_in_panel) {
        const int off = ssfa_offset(buf) + tile_in_panel * 256;
        opus::static_for<2>([&](auto h) { opus::static_for<T::E_M>([&](auto e) {
            v_sfa_next[h.value][e.value] = __builtin_bit_cast(opus::u32_t, load<4>(rd_sfa_, rsfa_offs[h.value * T::E_M + e.value] + off));
        }); });
    };
    auto load_sfb = [&](auto& rd_sfb_, int buf, int tile_in_panel) {
        const int off = ssfb_offset(buf) + tile_in_panel * 256;
        opus::static_for<2>([&](auto h) { opus::static_for<T::E_N>([&](auto e) {
            v_sfb_next[h.value][e.value] = __builtin_bit_cast(opus::u32_t, load<4>(rd_sfb_, rsfb_offs[h.value * T::E_N + e.value] + off));
        }); });
    };
    // Two dwords shifted as one 64-bit value (one v_lshrrev_b64); an odd E shifts its last dword alone.
    auto shr2 = [](unsigned int lo, unsigned int hi, unsigned int sh, unsigned int& o0, unsigned int& o1) {
        const unsigned long long v = ((static_cast<unsigned long long>(hi) << 32) | lo) >> sh;
        o0 = static_cast<unsigned int>(v); o1 = static_cast<unsigned int>(v >> 32);
    };
    auto read_sf = [&](auto h_) {
        constexpr int h = decltype(h_)::value;
        opus::static_for<T::E_M / 2>([&](auto q) { shr2(v_sfa_next[h][2 * q.value], v_sfa_next[h][2 * q.value + 1], sfa_sh, v_sfa[h][2 * q.value], v_sfa[h][2 * q.value + 1]); });
        opus::static_for<T::E_N / 2>([&](auto q) { shr2(v_sfb_next[h][2 * q.value], v_sfb_next[h][2 * q.value + 1], sfb_sh, v_sfb[h][2 * q.value], v_sfb[h][2 * q.value + 1]); });
        if constexpr (T::E_M % 2) v_sfa[h][T::E_M - 1] = v_sfa_next[h][T::E_M - 1] >> sfa_sh;
        if constexpr (T::E_N % 2) v_sfb[h][T::E_N - 1] = v_sfb_next[h][T::E_N - 1] >> sfb_sh;
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

    // Per-wave instruction counts for sched_quadrant.
    constexpr int BUFFER_A = T::a_buffer_load_insts, BUFFER_B = T::b_buffer_load_insts;      // buffer_load ... lds per half tile
    constexpr int BUFFER_SF = 1;                                                             // scale panel buffer_load ... lds (sfa or sfb)
    constexpr int DS_A = T::E_K * T::E_M, DS_B = T::E_K * T::E_N;                            // ds_read_b128 per half tile
    constexpr int DS_SF = T::E_M + T::E_N;                                                   // scale ds_read2st64_b32 per K tile (A, B; half 0 / 1 paired)
    constexpr int VALU_READ_SF = (T::E_M + 1) / 2 + (T::E_N + 1) / 2;                        // read_sf: 64-bit shifts per half
    constexpr int VMCNT_AB0 = BUFFER_A + BUFFER_B;                                           // C10's A0 / B0 copies left in flight

    // Prologue: load K tiles 0-1 and scale panel 0 (+ sfb of panel 1), compute C00 of tile 0 (+ store the previous
    // tile's bottom-half C).  Persistent: K tile 0 and panel 0 were loaded in the previous tile's last K tile, and its
    // top-half C stores may still fly.
    if (!prefetched) {
        prefetch_a(g_a, 0, 0, 0);
        prefetch_b(g_b, 0, 0, 0);
        prefetch_a(g_a, 0, 0, 1);
        prefetch_b(g_b, 0, 0, 1);
        prefetch_sfa(0, 0);
        prefetch_sfb(0, 0);
        prefetch_sfb(T::SCALE_PANEL_TILES, 1);
    }
    if (loops > 1) {
        prefetch_a(g_a, 1, 1, 0);
        prefetch_b(g_b, 1, 1, 0);
        prefetch_a(g_a, 1, 1, 1);
        prefetch_b(g_b, 1, 1, 1);
        if (prefetched) s_waitcnt_vmcnt(number<2 * (BUFFER_A + BUFFER_B) + T::C_COPIES>{});   // tile 1 + top-half C may fly
        else            s_waitcnt_vmcnt(number<2 * (BUFFER_A + BUFFER_B)>{});                 // tile 1 may fly
    } else {
        s_waitcnt_vmcnt(0_I);
    }
    views(s_a.ptr, s_b.ptr, s_sfa.ptr, s_sfb.ptr, [&](auto& rd_a, auto& rd_b, auto& rd_sfa, auto& rd_sfb) {
        // A0 / B0 reads first; the scale reads need a second barrier and cover their latency.
        sync_threads();
        __builtin_amdgcn_sched_barrier(0);
        load_a(rd_a, number<0>{});
        load_b(rd_b, number<0>{});
        sched_group<SCHED_DS_READ, T::E_K * (T::E_M + T::E_N)>();   // A0, B0 first
        __builtin_amdgcn_sched_barrier(0);
        s_waitcnt_lgkmcnt(0_I);
        sync_threads();
        __builtin_amdgcn_sched_barrier(0);
        load_sfa(rd_sfa, 0, 0);
        load_sfb(rd_sfb, 0, 0);
        read_sf(number<0>{});
        read_sf(number<1>{});
        __builtin_amdgcn_sched_barrier(0);
        // C00(0) + read A1 / B1 (+ store the previous tile's bottom-half C).
        load_a(rd_a, number<1>{});
        load_b(rd_b, number<1>{});
        if constexpr (T::PERSIST) store_pend_c(wave_id * T::WARP_SIZE + lane_id);
        mma_quadrant<T, 0, 0>(mma, v_a, v_b, v_c, v_sfa, v_sfb);
        sched_quadrant<T, DS_A + DS_B, 0, 0, H1_COPIES>();
        __builtin_amdgcn_sched_barrier(0);
    });

    // Main loop: K tile t = C10, C01, C11 of t and C00 of t+1, reading t+1 and copying t+2.  Two K tiles per trip
    // (compile-time stage / panel parity); the first trip zeroes C (zc).  Scale buffers: read sf_rd, write sf_wr (the
    // panel of t+2 / the sfb panel of t+3, one panel ahead of the sfa ones).
    int tile = 0;
    int sf_rd = 0;
    auto main_trip = [&](auto first_) {
        constexpr auto zc = opus::bool_constant<decltype(first_)::value>{};
        {   // even tile
            constexpr int stage = 0, next_stage = 1;
            const int sf_wr = sf_rd == T::SCALE_PANEL_BUFFERS - 1 ? 0 : sf_rd + 1;
            views(s_a.ptr + sa_offset(next_stage, 0), s_b.ptr + sb_offset(next_stage, 0), s_sfa.ptr, s_sfb.ptr,
                  [&](auto& rd_a, auto& rd_b, auto& rd_sfa, auto& rd_sfb) {
                // stage(t) is free for tile t+2.
                s_waitcnt_lgkmcnt(0_I);
                sync_threads();
                __builtin_amdgcn_sched_barrier(0);

                // C10 + copy A0 / B0 of t+2.
                prefetch_a(g_a, stage, tile + 2, 0);
                prefetch_b(g_b, stage, tile + 2, 0);
                mma_quadrant<T, 1, 0, decltype(zc)::value>(mma, v_a, v_b, v_c, v_sfa, v_sfb);
                sched_quadrant<T, 0, BUFFER_A + BUFFER_B, 0>();
                __builtin_amdgcn_sched_barrier(0);
                // tile t+1 (and its scale panel) landed (the copies above may still fly).
                s_waitcnt_vmcnt(number<VMCNT_AB0>{});
                sync_threads();
                __builtin_amdgcn_sched_barrier(0);

                // C01 + read scales / B0 of t+1, copy A1 / B1 of t+2.
                load_sfa(rd_sfa, sf_rd, 1);
                load_sfb(rd_sfb, sf_rd, 1);
                load_b(rd_b, number<0>{});
                prefetch_a(g_a, stage, tile + 2, 1);
                prefetch_b(g_b, stage, tile + 2, 1);
                mma_quadrant<T, 0, 1, decltype(zc)::value>(mma, v_a, v_b, v_c, v_sfa, v_sfb);
                sched_quadrant<T, DS_B + DS_SF, BUFFER_A + BUFFER_B, 0>();
                __builtin_amdgcn_sched_barrier(0);

                // C11 + read A0 of t+1, scales of half 0, panel step.
                load_a(rd_a, number<0>{});
                read_sf(number<0>{});
                prefetch_sfa(tile + 2, sf_wr);                    // panel step: sfa of t+2
                mma_quadrant<T, 1, 1, decltype(zc)::value>(mma, v_a, v_b, v_c, v_sfa, v_sfb);
                sched_quadrant<T, DS_A, BUFFER_SF, VALU_READ_SF>();
                __builtin_amdgcn_sched_barrier(0);

                // C00 of t+1 + read A1 / B1 of t+1, scales of half 1.
                load_a(rd_a, number<1>{});
                load_b(rd_b, number<1>{});
                read_sf(number<1>{});
                mma_quadrant<T, 0, 0>(mma, v_a, v_b, v_c, v_sfa, v_sfb);
                sched_quadrant<T, DS_A + DS_B, 0, VALU_READ_SF>();
                __builtin_amdgcn_sched_barrier(0);
            });
            sf_rd = sf_wr;                                     // after an even tile
        }
        {   // odd tile
            const int tile_odd = tile + 1;
            const int tile = tile_odd;
            constexpr int stage = 1, next_stage = 0;
            const int sf_wr = sf_rd == T::SCALE_PANEL_BUFFERS - 1 ? 0 : sf_rd + 1;
            views(s_a.ptr + sa_offset(next_stage, 0), s_b.ptr + sb_offset(next_stage, 0), s_sfa.ptr, s_sfb.ptr,
                  [&](auto& rd_a, auto& rd_b, auto& rd_sfa, auto& rd_sfb) {
                // stage(t) is free for tile t+2.
                s_waitcnt_lgkmcnt(0_I);
                sync_threads();
                __builtin_amdgcn_sched_barrier(0);

                // C10 + copy A0 / B0 of t+2.
                prefetch_a(g_a, stage, tile + 2, 0);
                prefetch_b(g_b, stage, tile + 2, 0);
                mma_quadrant<T, 1, 0>(mma, v_a, v_b, v_c, v_sfa, v_sfb);
                sched_quadrant<T, 0, BUFFER_A + BUFFER_B, 0>();
                __builtin_amdgcn_sched_barrier(0);
                // tile t+1 (and its scale panel) landed (the copies above may still fly).
                s_waitcnt_vmcnt(number<VMCNT_AB0>{});
                sync_threads();
                __builtin_amdgcn_sched_barrier(0);

                // C01 + read scales / B0 of t+1, copy A1 / B1 of t+2.
                load_sfa(rd_sfa, sf_rd, 0);
                load_sfb(rd_sfb, sf_rd, 0);
                load_b(rd_b, number<0>{});
                prefetch_a(g_a, stage, tile + 2, 1);
                prefetch_b(g_b, stage, tile + 2, 1);
                mma_quadrant<T, 0, 1>(mma, v_a, v_b, v_c, v_sfa, v_sfb);
                sched_quadrant<T, DS_B + DS_SF, BUFFER_A + BUFFER_B, 0>();
                __builtin_amdgcn_sched_barrier(0);

                // C11 + read A0 of t+1, scales of half 0, panel step.
                load_a(rd_a, number<0>{});
                read_sf(number<0>{});
                prefetch_sfb(tile + 3, sf_wr);                    // panel step: sfb of t+3
                mma_quadrant<T, 1, 1>(mma, v_a, v_b, v_c, v_sfa, v_sfb);
                sched_quadrant<T, DS_A, BUFFER_SF, VALU_READ_SF>();
                __builtin_amdgcn_sched_barrier(0);

                // C00 of t+1 + read A1 / B1 of t+1, scales of half 1.
                load_a(rd_a, number<1>{});
                load_b(rd_b, number<1>{});
                read_sf(number<1>{});
                mma_quadrant<T, 0, 0>(mma, v_a, v_b, v_c, v_sfa, v_sfb);
                sched_quadrant<T, DS_A + DS_B, 0, VALU_READ_SF>();
                __builtin_amdgcn_sched_barrier(0);
            });
        }
    };
    if (tile + 3 < loops) { main_trip(opus::bool_constant<true>{}); tile += 2; }
    else {
        opus::static_for<2>([&](auto hm) { opus::static_for<2>([&](auto hn) { opus::static_for<T::E_M * T::E_N>([&](auto f) {
            if constexpr (hm.value + hn.value > 0) clear(v_c[hm.value][hn.value][f.value]);
        }); }); });
    }
#pragma clang loop unroll(disable)
    for (; tile + 3 < loops; tile += 2) main_trip(opus::bool_constant<false>{});
    // Odd K tile count: one more even step.
    if (tile + 2 < loops) {
        constexpr int stage = 0, next_stage = 1;
        const int sf_wr = sf_rd == T::SCALE_PANEL_BUFFERS - 1 ? 0 : sf_rd + 1;
        views(s_a.ptr + sa_offset(next_stage, 0), s_b.ptr + sb_offset(next_stage, 0), s_sfa.ptr, s_sfb.ptr,
              [&](auto& rd_a, auto& rd_b, auto& rd_sfa, auto& rd_sfb) {
            // stage(t) is free for tile t+2.
            s_waitcnt_lgkmcnt(0_I);
            sync_threads();
            __builtin_amdgcn_sched_barrier(0);

            // C10 + copy A0 / B0 of t+2.
            prefetch_a(g_a, stage, tile + 2, 0);
            prefetch_b(g_b, stage, tile + 2, 0);
            mma_quadrant<T, 1, 0>(mma, v_a, v_b, v_c, v_sfa, v_sfb);
            sched_quadrant<T, 0, BUFFER_A + BUFFER_B, 0>();
            __builtin_amdgcn_sched_barrier(0);
            // tile t+1 landed (the copies above may still fly).
            s_waitcnt_vmcnt(number<VMCNT_AB0>{});
            sync_threads();
            __builtin_amdgcn_sched_barrier(0);

            // C01 + read scales / B0 of t+1, copy A1 / B1 of t+2.
            load_sfa(rd_sfa, sf_rd, 1);
            load_sfb(rd_sfb, sf_rd, 1);
            load_b(rd_b, number<0>{});
            prefetch_a(g_a, stage, tile + 2, 1);
            prefetch_b(g_b, stage, tile + 2, 1);
            mma_quadrant<T, 0, 1>(mma, v_a, v_b, v_c, v_sfa, v_sfb);
            sched_quadrant<T, DS_B + DS_SF, BUFFER_A + BUFFER_B, 0>();
            __builtin_amdgcn_sched_barrier(0);

            // panel step: sfa of t+2.
            prefetch_sfa(tile + 2, sf_wr);
            __builtin_amdgcn_sched_barrier(0);

            // C11 + read A0 of t+1, scales of half 0.
            load_a(rd_a, number<0>{});
            read_sf(number<0>{});
            mma_quadrant<T, 1, 1>(mma, v_a, v_b, v_c, v_sfa, v_sfb);
            sched_quadrant<T, DS_A, 0, VALU_READ_SF>();
            __builtin_amdgcn_sched_barrier(0);

            // C00 of t+1 + read A1 / B1 of t+1, scales of half 1.
            load_a(rd_a, number<1>{});
            load_b(rd_b, number<1>{});
            read_sf(number<1>{});
            mma_quadrant<T, 0, 0>(mma, v_a, v_b, v_c, v_sfa, v_sfb);
            sched_quadrant<T, DS_A + DS_B, 0, VALU_READ_SF>();
            __builtin_amdgcn_sched_barrier(0);
        });
        sf_rd = sf_wr;
        ++tile;
    }

    // Epilogue 1: second-to-last K tile, no more copies.
    if (tile + 1 < loops) {
        const int next_stage = (tile & 1) ^ 1;
        views(s_a.ptr + sa_offset(next_stage, 0), s_b.ptr + sb_offset(next_stage, 0), s_sfa.ptr, s_sfb.ptr,
              [&](auto& rd_a, auto& rd_b, auto& rd_sfa, auto& rd_sfb) {
            mma_quadrant<T, 1, 0>(mma, v_a, v_b, v_c, v_sfa, v_sfb);
            sched_quadrant<T, 0, 0, 0>();
            __builtin_amdgcn_sched_barrier(0);
            // tile t+1 landed.
            s_waitcnt_vmcnt(0_I);
            sync_threads();
            __builtin_amdgcn_sched_barrier(0);

            // C01 + read scales / B0 of t+1.
            load_sfa(rd_sfa, sf_rd, (tile + 1) & 1);
            load_sfb(rd_sfb, sf_rd, (tile + 1) & 1);
            load_b(rd_b, number<0>{});
            mma_quadrant<T, 0, 1>(mma, v_a, v_b, v_c, v_sfa, v_sfb);
            sched_quadrant<T, DS_B + DS_SF, 0, 0>();
            __builtin_amdgcn_sched_barrier(0);

            // C11 + read A0 of t+1, scales of half 0.
            load_a(rd_a, number<0>{});
            read_sf(number<0>{});
            mma_quadrant<T, 1, 1>(mma, v_a, v_b, v_c, v_sfa, v_sfb);
            sched_quadrant<T, DS_A, 0, VALU_READ_SF>();
            __builtin_amdgcn_sched_barrier(0);

            // C00 of t+1 + read A1 / B1 of t+1, scales of half 1.
            load_a(rd_a, number<1>{});
            load_b(rd_b, number<1>{});
            read_sf(number<1>{});
            mma_quadrant<T, 0, 0>(mma, v_a, v_b, v_c, v_sfa, v_sfb);
            sched_quadrant<T, DS_A + DS_B, 0, VALU_READ_SF>();
            __builtin_amdgcn_sched_barrier(0);
        });
        ++tile;
    }

    // Write-back: bf16 C goes through LDS half a tile at a time (rows padded to B_N + 8) so stores fill whole 128 B lines.
    constexpr int c_lds_stride = T::B_N + 8;
    constexpr int VEC_OUT = 16 / sizeof(D_C);
    static_assert(T::HALF_B_M * c_lds_stride * sizeof(D_C) <= stage_elems * sizeof(D_AB));
    auto s_c = make_smem(reinterpret_cast<D_C*>(smem_ab + stage_elems));
    // Opaque lane id: the write-back offsets are recomputed here, not kept live from the prologue.
    unsigned int lane_mask_e = ~0u;
    asm volatile("" : "+s"(lane_mask_e));
    const int lane_e = static_cast<int>(__builtin_amdgcn_mbcnt_hi(lane_mask_e, __builtin_amdgcn_mbcnt_lo(lane_mask_e, 0u)));
    const int wave_e = wave_id;
    const int tid_e = wave_e * T::WARP_SIZE + lane_e;
    auto p_coord_c = opus::make_tuple(wave_e % T::T_M, lane_e % mma.grpn_c, wave_e / T::T_M, lane_e / mma.grpn_c);
    auto u_sc = partition_layout_c<T::VEC_C>(mma, opus::make_tuple(opus::number<c_lds_stride>{}, 1_I), p_coord_c);
    const int output_thread_id = tid_e;
    // B_N = 256: rows with bit 3 set swap the halves of each 16 B chunk (rows r / r + 8 share a bank); reads swap back.
    const int c_swz = T::C_SWZ ? ((lane_e >> 3) & 1) * (4 - 8 * ((lane_e / mma.grpn_c) & 1)) : 0;
    auto c_unswz = [](auto v, auto i_) {
        if constexpr (!T::C_SWZ || (decltype(i_)::value & 1) == 0) return v;
        else {
            using U4 = opus::vector_t<opus::u32_t, 4>;
            const U4 u = __builtin_bit_cast(U4, v);
            return __builtin_bit_cast(decltype(v), U4{u[2], u[3], u[0], u[1]});
        }
    };
    // Half tile LDS -> global: all reads before the stores (a ds_read is never moved above a buffer_store).
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
                v[i.value] = load<VEC_OUT>(s_c, c_row(output_thread_id, i) * c_lds_stride + c_col(output_thread_id, i));
            });
            opus::static_for<c_copies>([&](auto i) {
                const int r = c_row(output_thread_id, i) + half_tile_m * T::HALF_B_M, c = c_col(output_thread_id, i);
                store<VEC_OUT>(g_c, v[i.value], r * kargs.stride_c + c + (c < cols_in ? 0 : C_OOB), 0, opus::number<C_AUX>{});
            });
        } else {
            const int oob = (output_thread_id * VEC_OUT) % T::B_N < cols_in ? 0 : C_OOB;
            opus::vector_t<D_C, VEC_OUT> v[c_copies];
            opus::static_for<c_copies>([&](auto i) {
                const int linear = output_thread_id * VEC_OUT + decltype(i)::value * T::BLOCK_SIZE * VEC_OUT;
                v[i.value] = c_unswz(load<VEC_OUT>(s_c, (linear / T::B_N) * c_lds_stride + linear % T::B_N), i);
            });
            opus::static_for<c_copies>([&](auto i) {
                const int linear = output_thread_id * VEC_OUT + decltype(i)::value * T::BLOCK_SIZE * VEC_OUT;
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

    // Epilogue 2: last K tile.  Persistent: also copy K tile 0 and scale panel 0 of the next output tile (empty
    // buffer range when there is none, so no branch).
    {
        const int next_id = tile_id + grid_blocks;
        const bool has_next = next_id < total_tiles;
        const int nid = has_next ? next_id : tile_id;
        const int n_batch = nid / tiles_per_batch;
        const int n_in_batch = nid % tiles_per_batch;
        const int n_row = (n_in_batch / num_tiles_n) * T::B_M;
        const int n_col = (n_in_batch % num_tiles_n) * T::B_N;
        const int n_rows_in = opus_mxfp4_bpreshuffle_min(kargs.m - n_row, T::B_M);
        const int n_cols_in = opus_mxfp4_bpreshuffle_min(kargs.n - n_col, T::B_N);
        auto ng_a = make_gmem(reinterpret_cast<const D_AB*>(kargs.ptr_a) + n_batch * kargs.stride_a_batch + static_cast<long long>(n_row) * kargs.stride_a,
                              has_next ? static_cast<unsigned int>(n_rows_in) * static_cast<unsigned int>(kargs.stride_a) : 0u);
        auto ng_b = make_gmem(reinterpret_cast<const D_AB*>(kargs.ptr_b) + n_batch * kargs.stride_b_batch + static_cast<long long>(n_col) * kargs.stride_b,
                              has_next ? static_cast<unsigned int>(n_cols_in) * static_cast<unsigned int>(kargs.stride_b) : 0u);
        auto ng_sfa = make_gmem(reinterpret_cast<const D_SF*>(kargs.ptr_sfa) + n_batch * kargs.stride_sfa_batch + static_cast<long long>(n_row / 32) * kargs.stride_sfa,
                                has_next ? static_cast<unsigned int>(opus_mxfp4_bpreshuffle_ceil_div(n_rows_in, 32) * kargs.stride_sfa) : 0u);
        auto ng_sfb = make_gmem(reinterpret_cast<const D_SF*>(kargs.ptr_sfb) + n_batch * kargs.stride_sfb_batch + static_cast<long long>(n_col / 32) * kargs.stride_sfb,
                                has_next ? static_cast<unsigned int>(opus_mxfp4_bpreshuffle_ceil_div(n_cols_in, 32) * kargs.stride_sfb) : 0u);
        prefetched = T::PERSIST ? has_next : false;

        s_waitcnt_lgkmcnt(0_I);
        sync_threads();
        __builtin_amdgcn_sched_barrier(0);

        // C10 + copy A0 / B0 of the next tile.
        if constexpr (T::PERSIST) {
            prefetch_a(ng_a, 0, 0, 0);
            prefetch_b(ng_b, 0, 0, 0);
        }
        mma_quadrant<T, 1, 0>(mma, v_a, v_b, v_c, v_sfa, v_sfb);
        sched_quadrant<T, 0, BUFFER_A + BUFFER_B, 0>();
        __builtin_amdgcn_sched_barrier(0);

        // C01 + copy A1 / B1 of the next tile.
        if constexpr (T::PERSIST) {
            prefetch_a(ng_a, 0, 0, 1);
            prefetch_b(ng_b, 0, 0, 1);
        }
        mma_quadrant<T, 0, 1>(mma, v_a, v_b, v_c, v_sfa, v_sfb);
        sched_quadrant<T, 0, BUFFER_A + BUFFER_B, 0>();
        __builtin_amdgcn_sched_barrier(0);

        // C11 + scale panel 0 (+ sfb of panel 1) of the next tile (offsets from an opaque lane id).
        if constexpr (T::PERSIST) {
            unsigned int lm = ~0u; asm volatile("" : "+s"(lm));
            const int ln = static_cast<int>(__builtin_amdgcn_mbcnt_hi(lm, __builtin_amdgcn_mbcnt_lo(lm, 0u)));
            // make_layout_gsf* block map as one sum x stride (the layout form keeps more per-tile offsets live).
            const int ws_a = wave_id % T::SF_WAVES_A, ws_b = wave_id % T::SF_WAVES_B;
            const int blk = 2 * ws_a + ln / 32, blk_b = 2 * ws_b + ln / 32, in_blk = (ln % 32) * T::VEC_GLOBAL_SCALE;
            char* const sfa_dst = reinterpret_cast<char*>(s_sfa.ptr) + ws_a * (T::WARP_SIZE * T::VEC_GLOBAL_SCALE);
            char* const sfb_dst = reinterpret_cast<char*>(s_sfb.ptr) + ws_b * (T::WARP_SIZE * T::VEC_GLOBAL_SCALE);
            ng_sfa.template async_load<T::VEC_GLOBAL_SCALE>(sfa_dst, blk * kargs.stride_sfa + in_blk);
            ng_sfb.template async_load<T::VEC_GLOBAL_SCALE>(sfb_dst, blk_b * kargs.stride_sfb + in_blk);
            ng_sfb.template async_load<T::VEC_GLOBAL_SCALE>(sfb_dst + T::SFB_PANEL_BYTES, blk_b * kargs.stride_sfb + in_blk + T::SCALE_PANEL_TILES * 256);
        }
        mma_quadrant<T, 1, 1>(mma, v_a, v_b, v_c, v_sfa, v_sfb);
        sched_quadrant<T, 0, 3 * BUFFER_SF, 0>();
        __builtin_amdgcn_sched_barrier(0);
    }

    // Split-K, S > 1: store this segment's fp32 partial tile (C fragment layout, row stride B_N) to the workspace; the
    // last block to arrive for this tile (atomic counter, reset by that block) adds the other S - 1 partials and writes
    // C below.
    if constexpr (T::SPLITK) {
        int* const sk_last_p = reinterpret_cast<int*>(smem_sfa);   // broadcast flag (scale panels are no longer read)
        if (split_k > 1) {
            const long tile_elems = static_cast<long>(T::B_M) * T::B_N;
            float* const ws_tiles = reinterpret_cast<float*>(reinterpret_cast<char*>(kargs.ptr_ws) + opus_mxfp4_bpreshuffle_sk_counter_bytes(total_tiles));
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

    // Write back C: top half now; bottom half now (np) or in the next tile's prologue (persistent).  No vmcnt waits:
    // the next tile's K tile 0 copies are in flight.
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
                v_pend_c[i.value] = load<VEC_OUT>(s_c, c_row(output_thread_id, i) * c_lds_stride + c_col(output_thread_id, i));
            } else {
                const int linear = output_thread_id * VEC_OUT + decltype(i)::value * T::BLOCK_SIZE * VEC_OUT;
                v_pend_c[i.value] = c_unswz(load<VEC_OUT>(s_c, (linear / T::B_N) * c_lds_stride + linear % T::B_N), i);
            }
        });
        pend_c_ptr = reinterpret_cast<D_C*>(kargs.ptr_c) + batch_id * kargs.stride_c_batch + static_cast<long long>(row) * kargs.stride_c + col;
        pend_c = 1;
        pend_rows = rows_in;
        pend_cols = cols_in;
    }
    // The next tile's copies overwrite the staged C.
    s_waitcnt_lgkmcnt(0_I);
    sync_threads();
    __builtin_amdgcn_sched_barrier(0);
    }   // tile loop
    if (T::PERSIST && pend_c) {   // the last tile's bottom half
        unsigned int lm = ~0u; asm volatile("" : "+s"(lm));
        const int ln = static_cast<int>(__builtin_amdgcn_mbcnt_hi(lm, __builtin_amdgcn_mbcnt_lo(lm, 0u)));
        s_waitcnt_lgkmcnt(0_I);
        store_pend_c(wave_id * T::WARP_SIZE + ln);
    }
#endif   // __gfx950__
}

template<class Traits>
__global__ __launch_bounds__(Traits::BLOCK_SIZE, 1)
void gemm_a4w4_mxfp4_nt_bpreshuffle_kernel(opus_gemm_mxfp4_bpreshuffle_kargs_gfx950 kargs) {
    gemm_a4w4_mxfp4_nt_bpreshuffle_body<Traits>(kargs);
}


}   // namespace opus_mxfp4_bpreshuffle
