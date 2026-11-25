#pragma once
// MXFP4 x MXFP4 NT 2x2 4-wave pipeline, plain row-major inputs (A / B / scales).

#include <opus/hip_minimal.hpp>
#include <opus/opus.hpp>

#include "opus_gemm_traits_a4w4_mxfp4_nt_gfx950.cuh"

using opus::operator""_I;

// Everything below lives in its own namespace (opus_mxfp4).
namespace opus_mxfp4 {

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

// Create layout for loading B matrix (row-major, as A with wave_m / wave_n swapped) from global memory.
template<class T>
__device__ inline auto make_layout_gb(int lane_id, int wave_id_m, int wave_id_n, int stride_b) {
    constexpr int threads_k = T::B_K_BYTES / T::VEC_B;

    constexpr auto gb_block_shape = opus::make_tuple(
        opus::number<T::T_M>{},
        opus::number<T::T_N>{},
        opus::number<T::b_buffer_load_insts / T::ROW_PAIR>{},
        opus::number<T::WARP_SIZE / threads_k>{},
        opus::number<T::ROW_PAIR>{},
        opus::number<threads_k>{},
        opus::number<T::VEC_B>{});

    constexpr auto gb_block_dim = opus::make_tuple(
        opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::y_dim{}, opus::p_dim{}, opus::y_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}));

    const int lane_id_n = lane_id / threads_k;

    return opus::make_layout<T::VEC_B>(
        gb_block_shape,
        opus::unfold_x_stride(gb_block_dim, gb_block_shape, opus::tuple{stride_b, 1_I}),
        opus::unfold_p_coord(gb_block_dim, opus::tuple{wave_id_m, wave_id_n, lane_id_n, (lane_id % threads_k) ^ (((lane_id_n >> 1) & 1) << 1)}));
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

// Create layout for reading B matrix from shared memory to registers (kk = 0; kk = 1 is + W_K_BYTES).
template<class T>
__device__ inline auto make_layout_rb(int lane_id, int wave_id_n) {
    constexpr auto rb_block_shape = opus::make_tuple(
        opus::number<T::E_N>{},
        opus::number<T::T_N>{},
        opus::number<T::T_N>{},
        opus::number<T::W_N / T::T_N>{},
        opus::number<T::E_K>{},
        opus::number<T::W_N * T::W_K_BYTES / T::WARP_SIZE / T::VEC_B>{},
        opus::number<T::WARP_SIZE / T::W_N>{},
        opus::number<T::VEC_B>{});

    constexpr auto rb_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::y_dim{}, opus::p_dim{}, opus::y_dim{}));

    const int lane_id_n = lane_id % T::W_N;

    return opus::make_layout<T::VEC_B>(
        rb_block_shape,
        opus::unfold_x_stride(rb_block_dim, rb_block_shape, opus::tuple{opus::number<T::smem_linear_wave_b + T::smem_padding>{}, 1_I}),
        opus::unfold_p_coord(rb_block_dim, opus::tuple{wave_id_n, lane_id_n % T::T_N, lane_id_n / T::T_N, 0_I, (lane_id / T::W_N) ^ (((lane_id_n >> 2) & 1) << 1)}));
}

// Scales (row-major [rows, K/32]): global -> VGPR -> 4x4 byte transpose -> LDS -> VGPR.  Writer wave (h, w): for A
// h = wave_n, w = wave_m (B: h = wave_m, w = wave_n); its lane 4 r + e handles row h * HALF_B + e * 32 + w * 16 + r.
// Rows e >= E feed bytes no MFMA selects.
// Create layout for loading scale factors from global memory: 16 B (16 K groups = one scale panel of 2 K tiles) of the
// lane's row (the panel goes into the offset at the load).
template<class T, int HALF_B, int T_H, int T_W>
__device__ inline auto make_layout_gsf(int lane_id, int wave_id_h, int wave_id_w, int stride_sf) {
    constexpr auto gsf_block_shape = opus::make_tuple(
        opus::number<T_H>{},
        opus::number<T::WARP_SIZE / T::W_M>{},
        opus::number<T_W>{},
        opus::number<T::W_M>{},
        opus::number<T::VEC_GLOBAL_SCALE>{});

    constexpr auto gsf_block_dim = opus::make_tuple(
        opus::make_tuple(opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}));

    return opus::make_layout<T::VEC_GLOBAL_SCALE>(
        gsf_block_shape,
        opus::unfold_x_stride(gsf_block_dim, gsf_block_shape, opus::tuple{HALF_B * stride_sf, stride_sf, 1_I}),
        opus::unfold_p_coord(gsf_block_dim, opus::tuple{wave_id_h, lane_id % (T::WARP_SIZE / T::W_M), wave_id_w, lane_id / (T::WARP_SIZE / T::W_M)}));
}
template<class T> __device__ inline auto make_layout_gsfa(int lane_id, int wave_id_m, int wave_id_n, int stride_sfa) { return make_layout_gsf<T, T::HALF_B_M, T::T_N, T::T_M>(lane_id, wave_id_n, wave_id_m, stride_sfa); }
template<class T> __device__ inline auto make_layout_gsfb(int lane_id, int wave_id_m, int wave_id_n, int stride_sfb) { return make_layout_gsf<T, T::HALF_B_N, T::T_M, T::T_N>(lane_id, wave_id_m, wave_id_n, stride_sfb); }

// Create layout for storing the transposed scales to shared memory: panel image [K tile][h][w][slot][8 B]; lane 4 r + g
// writes the slot its reader lane g * 16 + r reads, slots XOR-swizzled within a wave's 64: s -> s ^ (((s >> 4) & 3) << 2)
// (bank-conflict free).  offsets[t] = K tile t of the panel.
template<class T, int T_H, int T_W>
__device__ inline auto make_layout_ssf(int lane_id, int wave_id_h, int wave_id_w) {
    constexpr auto ssf_block_shape = opus::make_tuple(
        opus::number<T::SCALE_PANEL_TILES>{},
        opus::number<T_H>{},
        opus::number<T_W>{},
        opus::number<T::WARP_SIZE>{},
        opus::number<T::VEC_LDS_SCALE>{});

    constexpr auto ssf_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}, opus::p_dim{}, opus::y_dim{}));

    constexpr int lane_groups = T::WARP_SIZE / T::W_M;   // 4: lane = lane_groups * r + g
    const int g = lane_id % lane_groups, r = lane_id / lane_groups;
    const int slot = g * T::W_M + (r ^ (g * lane_groups));   // the reader lane g * W_M + r, XOR-swizzled

    return opus::make_layout<T::VEC_LDS_SCALE>(
        ssf_block_shape,
        opus::unfold_x_stride(ssf_block_dim, ssf_block_shape, opus::tuple{1_I}),
        opus::unfold_p_coord(ssf_block_dim, opus::tuple{wave_id_h, wave_id_w, slot}));
}
template<class T> __device__ inline auto make_layout_ssfa(int lane_id, int wave_id_m, int wave_id_n) { return make_layout_ssf<T, T::T_N, T::T_M>(lane_id, wave_id_n, wave_id_m); }
template<class T> __device__ inline auto make_layout_ssfb(int lane_id, int wave_id_m, int wave_id_n) { return make_layout_ssf<T, T::T_M, T::T_N>(lane_id, wave_id_m, wave_id_n); }

// Create layout for reading scales from shared memory to registers: lane reads the 2 dwords (kk = 0, 1) of half h at
// slot (h * T_W + w) * 64 + lane (swizzled as stored); offsets[h].  The K tile of the panel is added at the read.
template<class T, int T_W>
__device__ inline auto make_layout_rsf(int lane_id, int wave_id_w) {
    constexpr int lane_groups = T::WARP_SIZE / T::W_M;   // 4: the slot XOR (make_layout_ssf) undone per 16-lane row
    constexpr auto rsf_block_shape = opus::make_tuple(
        opus::number<T::HALVES>{},
        opus::number<T_W>{},
        opus::number<T::WARP_SIZE>{},
        opus::number<T::VEC_LDS_SCALE>{});

    constexpr auto rsf_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}, opus::y_dim{}));

    return opus::make_layout<T::VEC_LDS_SCALE>(
        rsf_block_shape,
        opus::unfold_x_stride(rsf_block_dim, rsf_block_shape, opus::tuple{1_I}),
        opus::unfold_p_coord(rsf_block_dim, opus::tuple{wave_id_w, lane_id ^ static_cast<int>(static_cast<unsigned>(lane_id) / T::W_M % lane_groups * lane_groups)}));
}
template<class T> __device__ inline auto make_layout_rsfa(int lane_id, int wave_id_m) { return make_layout_rsf<T, T::T_M>(lane_id, wave_id_m); }
template<class T> __device__ inline auto make_layout_rsfb(int lane_id, int wave_id_n) { return make_layout_rsf<T, T::T_N>(lane_id, wave_id_n); }

// Like opus async_load(layout), but the per-issue LDS offsets go into the MUBUF immediate (one LDS base in m0); the
// immediate also shifts the global address, so it is subtracted there.  opus async_load(layout) writes m0 per issue
// and keeps one LDS address per issue live in SGPRs: 4x the m0 writes and ~2x the SGPR spills here, ~1% slower over
// all 13 .co kids.
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
// spread evenly between them.  (Hand-tuned 256 x 256 schedules were measured equal to this and dropped.)
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
// starts from C = 0.  Scale dword kk of half h holds repeat e in byte e (opsel).  A function here, not a lambda:
// the form changes the generated code, and this one reproduces the committed .co.
//
template<class T, int HM, int HN, bool ZC = false, class MMA, class CF>
__device__ __forceinline__ void mma_quadrant(MMA& mma, const fp4x32_t (&v_a)[2][T::E_K][T::E_M], const fp4x32_t (&v_b)[2][T::E_K][T::E_N],
                                             CF (&v_c)[2][2][T::E_M * T::E_N], const opus::u32_t (&v_sfa)[2][T::E_K], const opus::u32_t (&v_sfb)[2][T::E_K]) {
    opus::static_for<T::E_M * T::E_N>([&](auto i) {
        constexpr int m = decltype(i)::value / T::E_N, n = decltype(i)::value % T::E_N;
        mma_scale_one<m, n, c_reg<T, HM, HN, m, n>(), ZC>(mma, v_a[HM][0][m], v_b[HN][0][n], v_c[HM][HN][i.value], v_sfa[HM][0], v_sfb[HN][0]);
        mma_scale_one<m, n, c_reg<T, HM, HN, m, n>()>(mma, v_a[HM][1][m], v_b[HN][1][n], v_c[HM][HN][i.value], v_sfa[HM][1], v_sfb[HN][1]);
    });
}

// ===========================================================================
// Kernel
// ===========================================================================
// Cache policy of the C stores: nt (non-temporal).  C is written once and never re-read, so it should not evict the
// A / B tiles other blocks still read from L2 (~1-1.5% at M >= 4096); A / B keep the default policy (nt there is
// 20-25% slower).
constexpr int C_AUX = 2;
// Kernel body as a device function: it backs the template __global__ below and the extern "C" entry points of the
// pre-compiled .co images (gen_co/build_co.py --arch gfx950 --family a4w4).
template<class Traits>
__device__ __forceinline__ void gemm_a4w4_mxfp4_nt_body(opus_gemm_mxfp4_kargs_gfx950 kargs) {
#if defined(__gfx950__)
    using namespace opus;

    using T = opus::remove_cvref_t<Traits>;
    using D_A = opus::fp4_t;
    using D_B = opus::fp4_t;
    using D_ACC = opus::fp32_t;
    using D_C = opus::bf16_t;
    using D_AB = opus::u8_t;     // fp4x2 bytes
    using D_SF = opus::e8m0_t;

    const int num_tiles_n = opus_mxfp4_ceil_div(kargs.n, T::B_N);
    const int wave_id = __builtin_amdgcn_readfirstlane(thread_id_x() / T::WARP_SIZE);
    const int tiles_per_batch = opus_mxfp4_ceil_div(kargs.m, T::B_M) * num_tiles_n;
    const int total_tiles = tiles_per_batch * kargs.batch;
    // Persistent: block b runs tiles b, b + grid_blocks, ...; one tile per block otherwise.
    const int grid_blocks = T::PERSIST ? kargs.grid_blocks : total_tiles;
    const int first_tile = static_cast<int>(block_id_x());
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
    opus::vector_t<opus::e8m0_t, Traits::VEC_GLOBAL_SCALE> v_nsfa0, v_nsfb0;   // scale panel 0 of the next output tile
    // A / B operands live across output tiles (the last K tile of one tile overlaps the next tile's prologue).
    fp4x32_t v_a[2][T::E_K][T::E_M];   // [half][kk][repeat]
    fp4x32_t v_b[2][T::E_K][T::E_N];
#pragma clang loop unroll(disable)
    for (int tile_id = first_tile; tile_id < total_tiles; tile_id += grid_blocks) {
    const int batch_id = tile_id / tiles_per_batch;
    const int tile_in_batch = tile_id % tiles_per_batch;
    const int block_m = tile_in_batch / num_tiles_n;
    const int block_n = tile_in_batch % num_tiles_n;
    const int loops = kargs.k / T::B_K;
    const int row = block_m * T::B_M;
    const int col = block_n * T::B_N;
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
    const int rows_in = T::BOUNDARY ? opus_mxfp4_min(kargs.m - row, T::B_M) : T::B_M;
    const int cols_in = T::BOUNDARY ? opus_mxfp4_min(kargs.n - col, T::B_N) : T::B_N;
    auto g_a = make_gmem(reinterpret_cast<const D_AB*>(kargs.ptr_a) + batch_id * kargs.stride_a_batch + static_cast<long long>(row) * kargs.stride_a,
                         T::BOUNDARY ? static_cast<unsigned int>(rows_in) * static_cast<unsigned int>(kargs.stride_a) : 0xffffffffu);
    auto g_b = make_gmem(reinterpret_cast<const D_AB*>(kargs.ptr_b) + batch_id * kargs.stride_b_batch + static_cast<long long>(col) * kargs.stride_b,
                         T::BOUNDARY ? static_cast<unsigned int>(cols_in) * static_cast<unsigned int>(kargs.stride_b) : 0xffffffffu);
    auto g_c = make_gmem(reinterpret_cast<D_C*>(kargs.ptr_c) + batch_id * kargs.stride_c_batch + static_cast<long long>(row) * kargs.stride_c + col,
                         T::BOUNDARY ? static_cast<unsigned int>(rows_in) * static_cast<unsigned int>(kargs.stride_c) * 2u : 0xffffffffu);
    // Row-major scales [rows, K/32]: rows past the matrix are outside the range (read as 0).
    auto g_sfa = make_gmem(reinterpret_cast<const D_SF*>(kargs.ptr_sfa) + batch_id * kargs.stride_sfa_batch + static_cast<long long>(row) * kargs.stride_sfa,
                           static_cast<unsigned int>(opus_mxfp4_min(kargs.m - row, T::B_M) * kargs.stride_sfa));
    auto g_sfb = make_gmem(reinterpret_cast<const D_SF*>(kargs.ptr_sfb) + batch_id * kargs.stride_sfb_batch + static_cast<long long>(col) * kargs.stride_sfb,
                           static_cast<unsigned int>(opus_mxfp4_min(kargs.n - col, T::B_N) * kargs.stride_sfb));

    const auto u_ga = make_layout_ga<T>(lane_id, wave_id_m, wave_id_n, kargs.stride_a);
    const auto u_gb = make_layout_gb<T>(lane_id, wave_id_m, wave_id_n, kargs.stride_b);
    const auto u_sa = make_layout_sa<T>(wave_id_m, wave_id_n);
    const auto u_sb = make_layout_sb<T>(wave_id_m, wave_id_n);
    const auto u_ra = make_layout_ra<T>(lane_id, wave_id_m);
    const auto u_rb = make_layout_rb<T>(lane_id, wave_id_n);
    const auto u_gsfa = make_layout_gsfa<T>(lane_id, wave_id_m, wave_id_n, kargs.stride_sfa);
    const auto u_gsfb = make_layout_gsfb<T>(lane_id, wave_id_m, wave_id_n, kargs.stride_sfb);
    const auto u_ssfa = make_layout_ssfa<T>(lane_id, wave_id_m, wave_id_n);
    const auto u_ssfb = make_layout_ssfb<T>(lane_id, wave_id_m, wave_id_n);
    const auto u_rsfa = make_layout_rsfa<T>(lane_id, wave_id_m);
    const auto u_rsfb = make_layout_rsfb<T>(lane_id, wave_id_n);

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

    // Per-half scale dwords of the MFMAs: dword kk holds repeat e in byte e (opsel e).
    using D_SF_PACK = opus::u32_t;
    D_SF_PACK v_sfa[2][T::E_K];
    D_SF_PACK v_sfb[2][T::E_K];

    auto ga_offset = [&](int half_tile_m, int tile_k) { return half_tile_m * T::HALF_B_M * kargs.stride_a + tile_k * T::B_K_BYTES; };
    auto gb_offset = [&](int half_tile_n, int tile_k) { return half_tile_n * T::HALF_B_N * kargs.stride_b + tile_k * T::B_K_BYTES; };
    constexpr int stage_elems = 2 * smem_a_elem + 2 * smem_b_elem;
    auto sa_offset = [&](int stage, int half_tile_m) { return stage * stage_elems + half_tile_m * smem_a_elem; };
    auto sb_offset = [&](int stage, int half_tile_n) { return stage * stage_elems + half_tile_n * smem_b_elem; };

    // Global -> LDS.  A chunk is 8 rows x 128 B (8 lanes x 16 B per row); chunk pair (2 c, 2 c + 1) interleaves into the
    // 16-row block c: chunk c holds rows (c / 2) * 16 + 2 * (lane / 8) + c % 2.  The layout path (async_load_imm) needs
    // whole chunk pairs per wave; with B_M / B_N = 192 (A_SPLIT_PAIR / B_SPLIT_PAIR) wave w copies chunks w * NI + i,
    // a pair straddles two waves, and async_load_rows computes each issue's rows directly.
    constexpr int CHUNK_BYTES = T::smem_linear_wave_a + T::smem_padding;   // LDS bytes of one chunk (incl. padding)
    constexpr int threads_k = T::B_K_BYTES / T::VEC_A;   // 8 lanes per 128-byte row
    const int a_kcol = ((lane_id % threads_k) ^ ((((lane_id / threads_k) >> 1) & 1) << 1)) * T::VEC_A;
    auto async_load_rows = [&](auto& g, auto ni_, char* smem, int w, int stride, int lds_off, int s_os) {
        constexpr int NI = decltype(ni_)::value;
        char* const base = smem + lds_off + w * NI * CHUNK_BYTES;
        opus::static_for<NI>([&](auto i_) {
            constexpr int i = decltype(i_)::value, io = i * CHUNK_BYTES;
            const unsigned c = w * NI + i;   // unsigned: / and % stay shifts and masks
            const int goff = (static_cast<int>(c / T::ROW_PAIR * T::W_M + c % T::ROW_PAIR) + T::ROW_PAIR * (lane_id / threads_k)) * stride + a_kcol;
            g.template async_load<T::VEC_A>(base, goff - io, s_os, opus::number<io>{});
        });
    };
    auto prefetch_a = [&](auto& g, int stage, int tile, int half_tile_m) {
        if constexpr (T::A_SPLIT_PAIR)
            async_load_rows(g, opus::number<T::a_buffer_load_insts>{}, reinterpret_cast<char*>(s_a.ptr), wave_id_n * 2 + wave_id_m,
                            kargs.stride_a, sa_offset(stage, half_tile_m), ga_offset(half_tile_m, tile));
        else async_load_imm<T::VEC_A>(g, s_a.ptr, u_ga, u_sa + sa_offset(stage, half_tile_m), ga_offset(half_tile_m, tile));
    };
    auto prefetch_b = [&](auto& g, int stage, int tile, int half_tile_n) {
        if constexpr (T::B_SPLIT_PAIR)
            async_load_rows(g, opus::number<T::b_buffer_load_insts>{}, reinterpret_cast<char*>(s_b.ptr), wave_id_m * 2 + wave_id_n,
                            kargs.stride_b, sb_offset(stage, half_tile_n), gb_offset(half_tile_n, tile));
        else async_load_imm<T::VEC_B>(g, s_b.ptr, u_gb, u_sb + sb_offset(stage, half_tile_n), gb_offset(half_tile_n, tile));
    };
    // Scales: global -> VGPR -> quad transpose -> LDS (MFMA layout).  A panel (2 K tiles) is loaded in tile t, stored in t + 1.
    using SF_PANEL = opus::vector_t<D_SF, T::VEC_GLOBAL_SCALE>;
    using SF_DW4 = opus::vector_t<opus::u32_t, 4>;
    SF_PANEL v_sf_pend;
    auto fetch_sfa = [&](int panel_k_begin) { return load<T::VEC_GLOBAL_SCALE>(g_sfa, u_gsfa.offsets[0] + panel_k_begin * T::NUM_KGROUPS); };
    auto fetch_sfb = [&](int panel_k_begin) { return load<T::VEC_GLOBAL_SCALE>(g_sfb, u_gsfb.offsets[0] + panel_k_begin * T::NUM_KGROUPS); };
    // 4x4 byte transpose in each lane quad (lane 4 r + e: [group g] -> [repeat e]): swap with e ^ 1, then e ^ 2.
    const unsigned int sf_q = static_cast<unsigned int>(lane_id % 4);
    const unsigned int sf_sel1 = (sf_q & 1) ? 0x03070105u : 0x06020400u;
    const unsigned int sf_sel2 = (sf_q & 2) ? 0x03020706u : 0x05040100u;
    auto quad_transpose = [&](unsigned int x) {
        const unsigned int y = mov_dpp(x, opus::number<0xB1>{}, opus::number<0xF>{}, opus::number<0xF>{}, opus::bool_constant<true>{});   // quad_perm [1,0,3,2]
        const unsigned int t = __builtin_amdgcn_perm(y, x, sf_sel1);
        const unsigned int z = mov_dpp(t, opus::number<0x4E>{}, opus::number<0xF>{}, opus::number<0xF>{}, opus::bool_constant<true>{});   // quad_perm [2,3,0,1]
        return __builtin_amdgcn_perm(z, t, sf_sel2);
    };
    // Dwords 0-1 / 2-3 = K tile 0 / 1 of the panel.
    auto put_sf = [&](auto& s_sf_, const auto& u_ssf, const SF_PANEL& raw, int buf) {
        const auto dw = __builtin_bit_cast(SF_DW4, raw);
        SF_DW4 t;
        opus::static_for<4>([&](auto d) { t[d.value] = quad_transpose(dw[d.value]); });
        using SF_DW2 = opus::vector_t<opus::u32_t, 2>;
        using SF_B8 = opus::vector_t<D_SF, T::VEC_LDS_SCALE>;
        const int o = buf * T::SFA_PANEL_BYTES;
        store<T::VEC_LDS_SCALE>(s_sf_, __builtin_bit_cast(SF_B8, SF_DW2{t[0], t[1]}), o + u_ssf.offsets[0]);
        store<T::VEC_LDS_SCALE>(s_sf_, __builtin_bit_cast(SF_B8, SF_DW2{t[2], t[3]}), o + u_ssf.offsets[1]);
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
                __builtin_bit_cast(fp4x32_t, load<T::VEC_B>(sb_, rb_offs[e.value] + sb_offset(0, half) + kk.value * T::W_K_BYTES));
        }); });
    };
    // Scales LDS -> VGPR, issued early (C01) into v_sf*_next; read_sf moves half h into v_sfa / v_sfb once the current
    // tile's MFMAs no longer read it.
    using SF_KK = opus::vector_t<D_SF_PACK, T::E_K>;
    SF_KK v_sfa_next[2], v_sfb_next[2];
    auto load_sfa = [&](auto& rd_sfa_, int buf, int tile_in_panel) {
        const int off = buf * T::SFA_PANEL_BYTES + tile_in_panel * (T::SFA_PANEL_BYTES / 2);
        opus::static_for<2>([&](auto h) { v_sfa_next[h.value] = __builtin_bit_cast(SF_KK, load<T::VEC_LDS_SCALE>(rd_sfa_, u_rsfa.offsets[h.value] + off)); });
    };
    auto load_sfb = [&](auto& rd_sfb_, int buf, int tile_in_panel) {
        const int off = buf * T::SFB_PANEL_BYTES + tile_in_panel * (T::SFB_PANEL_BYTES / 2);
        opus::static_for<2>([&](auto h) { v_sfb_next[h.value] = __builtin_bit_cast(SF_KK, load<T::VEC_LDS_SCALE>(rd_sfb_, u_rsfb.offsets[h.value] + off)); });
    };
    auto read_sf = [&](auto h_) {
        constexpr int h = decltype(h_)::value;
        opus::static_for<T::E_K>([&](auto kk) { v_sfa[h][kk.value] = v_sfa_next[h][kk.value]; v_sfb[h][kk.value] = v_sfb_next[h][kk.value]; });
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
    constexpr int DS_A = T::E_K * T::E_M, DS_B = T::E_K * T::E_N;                            // ds_read_b128 per half tile
    constexpr int DS_SF = T::HALVES * 2;                                                     // scale ds_read_b64 per K tile (A, B)
    constexpr int VALU_READ_SF = 0;                                                          // read_sf: register moves only
    constexpr int VALU_PUT_SF = T::VEC_GLOBAL_SCALE / 4 * (2 + 2);                           // put_sf: per dword 2 DPP + 2 v_perm
    constexpr int VMCNT_AB0 = BUFFER_A + BUFFER_B;                                           // C10's A0 / B0 copies left in flight
    // Prologue: load K tiles 0-1 and scale panels 0-1, store panel 0 to LDS, compute C00 of tile 0 (sfa of panel 1 stays in v_sf_pend).
    if (!prefetched) {
        prefetch_a(g_a, 0, 0, 0);
        prefetch_b(g_b, 0, 0, 0);
        prefetch_a(g_a, 0, 0, 1);
        prefetch_b(g_b, 0, 0, 1);
        v_nsfa0 = fetch_sfa(0);
        v_nsfb0 = fetch_sfb(0);
    }
    SF_PANEL v_sfb1;
    if (loops > T::SCALE_PANEL_TILES) {
        v_sfb1 = fetch_sfb(T::SCALE_PANEL_TILES);
        v_sf_pend = fetch_sfa(T::SCALE_PANEL_TILES);
    }
    if (loops > 1) {
        prefetch_a(g_a, 1, 1, 0);
        prefetch_b(g_b, 1, 1, 0);
        prefetch_a(g_a, 1, 1, 1);
        prefetch_b(g_b, 1, 1, 1);
        s_waitcnt_vmcnt(number<2 * (BUFFER_A + BUFFER_B) + 1>{});   // tile 1 + pending sfa may fly
    } else {
        s_waitcnt_vmcnt(0_I);
    }
    views(s_a.ptr, s_b.ptr, s_sfa.ptr, s_sfb.ptr, [&](auto& rd_a, auto& rd_b, auto& rd_sfa, auto& rd_sfb) {
        // A0 / B0 reads first; the scale transposes cover their latency.
        sync_threads();
        __builtin_amdgcn_sched_barrier(0);
        load_a(rd_a, number<0>{});
        load_b(rd_b, number<0>{});
        put_sf(rd_sfa, u_ssfa, v_nsfa0, 0);
        put_sf(rd_sfb, u_ssfb, v_nsfb0, 0);
        if (loops > T::SCALE_PANEL_TILES) put_sf(rd_sfb, u_ssfb, v_sfb1, 1);
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
        load_b(rd_b, opus::number<1>{});
        if constexpr (T::PERSIST) store_pend_c(wave_id * T::WARP_SIZE + lane_id);
        mma_quadrant<T, 0, 0>(mma, v_a, v_b, v_c, v_sfa, v_sfb);
        sched_quadrant<T, DS_A + DS_B, 0, 0, H1_COPIES>();
        __builtin_amdgcn_sched_barrier(0);
    });

    // Main loop: K tile t = C10, C01, C11 of t and C00 of t+1, reading t+1 and copying t+2.  Two K tiles per trip
    // (compile-time stage / panel parity); the first trip zeroes C (zc).  Scale buffers: read sf_rd, write sf_wr.
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
                // tile t+1 landed (the copies above may still fly).
                s_waitcnt_vmcnt(number<VMCNT_AB0>{});
                sync_threads();
                __builtin_amdgcn_sched_barrier(0);

                // C01 + read scales / B0 of t+1, copy A1 / B1 of t+2.
                load_sfa(rd_sfa, sf_rd, 1);
                load_sfb(rd_sfb, sf_rd, 1);
                load_b(rd_b, opus::number<0>{});
                prefetch_a(g_a, stage, tile + 2, 1);
                prefetch_b(g_b, stage, tile + 2, 1);
                mma_quadrant<T, 0, 1, decltype(zc)::value>(mma, v_a, v_b, v_c, v_sfa, v_sfb);
                sched_quadrant<T, DS_B + DS_SF, BUFFER_A + BUFFER_B, 0>();
                __builtin_amdgcn_sched_barrier(0);

                __builtin_amdgcn_sched_barrier(0);   // empty region: keeps the schedule

                // C11 + read A0 of t+1, scales of half 0.
                load_a(rd_a, number<0>{});
                read_sf(number<0>{});
                put_sf(rd_sfa, u_ssfa, v_sf_pend, sf_wr);   // panel step: sfa of t+2
                v_sf_pend = fetch_sfb(tile + 4);                 // and sfb of t+4
                mma_quadrant<T, 1, 1, decltype(zc)::value>(mma, v_a, v_b, v_c, v_sfa, v_sfb);
                sched_quadrant<T, DS_A, 1, VALU_PUT_SF>();
                __builtin_amdgcn_sched_barrier(0);

                // C00 of t+1 + read A1 / B1 of t+1, scales of half 1.
                load_a(rd_a, number<1>{});
                load_b(rd_b, opus::number<1>{});
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
                // tile t+1 landed (the copies above may still fly).
                s_waitcnt_vmcnt(number<VMCNT_AB0>{});
                sync_threads();
                __builtin_amdgcn_sched_barrier(0);

                // C01 + read scales / B0 of t+1, copy A1 / B1 of t+2.
                load_sfa(rd_sfa, sf_rd, 0);
                load_sfb(rd_sfb, sf_rd, 0);
                load_b(rd_b, opus::number<0>{});
                prefetch_a(g_a, stage, tile + 2, 1);
                prefetch_b(g_b, stage, tile + 2, 1);
                mma_quadrant<T, 0, 1>(mma, v_a, v_b, v_c, v_sfa, v_sfb);
                sched_quadrant<T, DS_B + DS_SF, BUFFER_A + BUFFER_B, 0>();
                __builtin_amdgcn_sched_barrier(0);

                __builtin_amdgcn_sched_barrier(0);   // empty region: keeps the schedule

                // C11 + read A0 of t+1, scales of half 0.
                load_a(rd_a, number<0>{});
                read_sf(number<0>{});
                put_sf(rd_sfb, u_ssfb, v_sf_pend, sf_wr);   // panel step: sfb of t+3
                v_sf_pend = fetch_sfa(tile + 3);                 // and sfa of t+3
                mma_quadrant<T, 1, 1>(mma, v_a, v_b, v_c, v_sfa, v_sfb);
                sched_quadrant<T, DS_A, 1, VALU_PUT_SF>();
                __builtin_amdgcn_sched_barrier(0);

                // C00 of t+1 + read A1 / B1 of t+1, scales of half 1.
                load_a(rd_a, number<1>{});
                load_b(rd_b, opus::number<1>{});
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
            load_b(rd_b, opus::number<0>{});
            prefetch_a(g_a, stage, tile + 2, 1);
            prefetch_b(g_b, stage, tile + 2, 1);
            mma_quadrant<T, 0, 1>(mma, v_a, v_b, v_c, v_sfa, v_sfb);
            sched_quadrant<T, DS_B + DS_SF, BUFFER_A + BUFFER_B, 0>();
            __builtin_amdgcn_sched_barrier(0);

            // panel step: sfa of t+2 and sfb of t+4.
            put_sf(rd_sfa, u_ssfa, v_sf_pend, sf_wr);
            v_sf_pend = fetch_sfb(tile + 4);
            __builtin_amdgcn_sched_barrier(0);

            // C11 + read A0 of t+1, scales of half 0.
            load_a(rd_a, number<0>{});
            read_sf(number<0>{});
            mma_quadrant<T, 1, 1>(mma, v_a, v_b, v_c, v_sfa, v_sfb);
            sched_quadrant<T, DS_A, 1, VALU_READ_SF>();
            __builtin_amdgcn_sched_barrier(0);

            // C00 of t+1 + read A1 / B1 of t+1, scales of half 1.
            load_a(rd_a, number<1>{});
            load_b(rd_b, opus::number<1>{});
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
            load_b(rd_b, opus::number<0>{});
            mma_quadrant<T, 0, 1>(mma, v_a, v_b, v_c, v_sfa, v_sfb);
            sched_quadrant<T, DS_B + DS_SF, 0, 0>();
            __builtin_amdgcn_sched_barrier(0);

            // C11 + read A0 of t+1, scales of half 0.
            load_a(rd_a, number<0>{});
            read_sf(number<0>{});
            mma_quadrant<T, 1, 1>(mma, v_a, v_b, v_c, v_sfa, v_sfb);
            sched_quadrant<T, DS_A, 1, VALU_READ_SF>();
            __builtin_amdgcn_sched_barrier(0);

            // C00 of t+1 + read A1 / B1 of t+1, scales of half 1.
            load_a(rd_a, number<1>{});
            load_b(rd_b, opus::number<1>{});
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
        const int n_rows_in = opus_mxfp4_min(kargs.m - n_row, T::B_M);
        const int n_cols_in = opus_mxfp4_min(kargs.n - n_col, T::B_N);
        auto ng_a = make_gmem(reinterpret_cast<const D_AB*>(kargs.ptr_a) + n_batch * kargs.stride_a_batch + static_cast<long long>(n_row) * kargs.stride_a,
                              has_next ? static_cast<unsigned int>(n_rows_in) * static_cast<unsigned int>(kargs.stride_a) : 0u);
        auto ng_b = make_gmem(reinterpret_cast<const D_AB*>(kargs.ptr_b) + n_batch * kargs.stride_b_batch + static_cast<long long>(n_col) * kargs.stride_b,
                              has_next ? static_cast<unsigned int>(n_cols_in) * static_cast<unsigned int>(kargs.stride_b) : 0u);
        auto ng_sfa = make_gmem(reinterpret_cast<const D_SF*>(kargs.ptr_sfa) + n_batch * kargs.stride_sfa_batch + static_cast<long long>(n_row) * kargs.stride_sfa,
                                has_next ? static_cast<unsigned int>(n_rows_in * kargs.stride_sfa) : 0u);
        auto ng_sfb = make_gmem(reinterpret_cast<const D_SF*>(kargs.ptr_sfb) + n_batch * kargs.stride_sfb_batch + static_cast<long long>(n_col) * kargs.stride_sfb,
                                has_next ? static_cast<unsigned int>(n_cols_in * kargs.stride_sfb) : 0u);
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

        // C11 + scale panel 0 of the next tile (offsets from an opaque lane id).
        if constexpr (T::PERSIST) {
            unsigned int lm = ~0u; asm volatile("" : "+s"(lm));
            const int ln = static_cast<int>(__builtin_amdgcn_mbcnt_hi(lm, __builtin_amdgcn_mbcnt_lo(lm, 0u)));
            // make_layout_gsf* row map as one row sum x stride (the layout form costs 4-5 more SGPR spills).
            const int e = ln % 4, r = ln / 4;
            v_nsfa0 = load<T::VEC_GLOBAL_SCALE>(ng_sfa, (wave_id_n * T::HALF_B_M + e * 32 + wave_id_m * 16 + r) * kargs.stride_sfa);
            v_nsfb0 = load<T::VEC_GLOBAL_SCALE>(ng_sfb, (wave_id_m * T::HALF_B_N + e * 32 + wave_id_n * 16 + r) * kargs.stride_sfb);
        }
        mma_quadrant<T, 1, 1>(mma, v_a, v_b, v_c, v_sfa, v_sfb);
        sched_quadrant<T, 0, 2, 0>();
        __builtin_amdgcn_sched_barrier(0);
    }

    // Write back C: top half now; bottom half now (np) or in the next tile's prologue (persistent).
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
void gemm_a4w4_mxfp4_nt_kernel(opus_gemm_mxfp4_kargs_gfx950 kargs) {
    gemm_a4w4_mxfp4_nt_body<Traits>(kargs);
}


}   // namespace opus_mxfp4
