// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// a16w16 (bf16) 4-wave pipeline with pinned registers, all four layouts (NT / NN / TT / TN).
// Traits / kargs: opus_gemm_a16w16_4wave_traits_gfx950 / opus_gemm_a16w16_4wave_kargs_gfx950.
//
// Structure (one workgroup = one B_M x B_N tile, 4 waves in 2 x 2):
//   * The tile is split into half M x half N quadrants C00, C10, C01, C11; every wave computes
//     E_M x E_N 16x16 fragments of each quadrant (2 MFMAs 16x16x32 per fragment and K tile).
//   * K tiles of 64 go global -> LDS by buffer_load ... lds (16 B per lane), double-buffered
//     (2 stages x 2 halves of A and B); LDS -> registers once per half and K tile.
//   * Per K tile t the quadrants run C10(t), C01(t), C11(t), C00(t+1); the copies of tile t+2 and
//     the LDS reads of tile t+1 are interleaved with the MFMAs by sched_group_barrier.
//   * Operands: a K-contiguous operand ([rows, K]) is copied with a 16-byte slot XOR swizzle and
//     read with ds_read_b128; an M/N-contiguous one ([K, rows]) is copied row-wise with a 32-byte
//     window swizzle and read with ds_read_b64_tr_b16 (4 per operand).
//   * PIN: A operands live in a[0:63], B operands in a[64:127], C in a[128:255] + v[128:...] (pinned;
//     needed by the large tiles). PIN = false leaves registers to the allocator (small tiles: fewer
//     registers -> WG_PER_CU workgroups per CU) and builds with stock ROCm clang.
//   * C (bf16, + optional bias) is written back through LDS (image rows padded by 8), one quadrant at a
//     time. Split-K: each split writes its fp32 partial tile into the workspace of
//     splitk_reduce_gfx950.cuh, whose reduce kernel produces C.
#pragma once

#include "opus_gemm_traits_a16w16_gfx950.cuh"

#ifdef __HIP_DEVICE_COMPILE__
#if defined(__gfx950__)
// PIN instances need the AGPR-pinning toolchain; PIN = false instances build with stock ROCm clang.
#if __has_cpp_attribute(clang::amdgpu_pin_agpr)
#define OPUS_A16W16_4WAVE_HAS_PIN 1
#else
#define OPUS_A16W16_4WAVE_HAS_PIN 0
#endif

namespace opus_a16w16_4wave {

using opus::operator""_I;

// ----------------------------------------------------------------------------
// K-contiguous operand ([rows, K], 128 bytes per row and K tile).
// ----------------------------------------------------------------------------
// global -> LDS: one wave issue copies 8 rows x 128 bytes; lane l writes LDS slot l % 8 of row l / 8 and
// fetches K slot (l % 8) ^ x ^ (wave_m << 2) of it (x = row in the chunk): the 16 lanes reading one
// fragment column then hit 16 different banks.
template<class T, int HALF_ROWS>
__device__ inline auto make_layout_g_k(int lane_id, int wave_id_m, int wave_id_n, int stride) {
    constexpr int threads_k = T::B_K_BYTES / T::VEC_A;                 // 8
    constexpr int threads_m_per_block = T::BLOCK_SIZE / threads_k;     // 32
    constexpr int threads_m_per_wave = T::WARP_SIZE / threads_k;       // 8
    static_assert(threads_k == T::B_K / T::VEC_A_ELEMS);

    constexpr auto shape = opus::make_tuple(
        opus::number<HALF_ROWS / threads_m_per_block>{},               // y issue
        opus::number<T::T_N>{},                                        // p wave_n
        opus::number<threads_m_per_wave>{},                            // p lane / 8
        opus::number<T::T_M>{},                                        // p wave_m
        opus::number<threads_k>{},                                     // p lane % 8
        opus::number<T::VEC_A>{});                                     // y 16 bytes
    constexpr auto dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}));
    return opus::make_layout<T::VEC_A>(
        shape,
        opus::unfold_x_stride(dim, shape, opus::tuple{stride, 1_I}),
        opus::unfold_p_coord(dim, opus::tuple{wave_id_n, lane_id / threads_k, wave_id_m,
                             (lane_id % threads_k) ^ (((lane_id / threads_k) ^ (wave_id_m << 2)) & 7)}));
}

// LDS side of the copy: chunk (wave_n * 2 + wave_m) * E + issue, so a wave's E issues are adjacent
// (one m0, MUBUF immediate offsets).
template<class T, int HALF_ROWS>
__device__ inline auto make_layout_s_k(int wave_id_m, int wave_id_n) {
    static_assert(T::smem_sub_a == T::WARP_SIZE / (T::B_K_BYTES / T::VEC_A), "one wave issue = one chunk");
    constexpr int E = HALF_ROWS / T::smem_sub_a / T::NUM_WAVES;
    constexpr auto shape = opus::make_tuple(
        opus::number<E>{},                                             // y issue
        opus::number<T::T_N>{},                                        // p wave_n
        opus::number<T::T_M>{},                                        // p wave_m
        opus::number<T::VEC_A>{});                                     // y 16 bytes
    constexpr auto dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}));
    return opus::make_layout(
        shape,
        opus::make_tuple(opus::number<T::smem_linear_wave_a>{}, opus::number<2 * E * T::smem_linear_wave_a>{},
                         opus::number<E * T::smem_linear_wave_a>{}, 1_I),
        opus::unfold_p_coord(dim, opus::tuple{wave_id_n, wave_id_m}));
}

// LDS -> registers: lane (row r = lane % 16, group g = lane / 16) reads 2 chunks of 16 bytes at bytes
// 64 c + 16 g of its 128-byte row (K-step c of the two 16x16x32 MFMAs); chunk c = 1 is load_*_k's
// c1_delta away (the swizzle flips bit 2 of the slot).
template<class T, int E>
__device__ inline auto make_layout_r_k(int lane_id, int wave_id) {
    static_assert(T::smem_sub_a * T::T_M == T::W_M && T::T_M == 2);
    static_assert(T::WARP_SIZE / T::W_M * T::E_C * T::VEC_A == T::B_K_BYTES);
    constexpr auto shape = opus::make_tuple(
        opus::number<E>{},                                             // y repeat
        opus::number<T::T_M>{},                                        // p wave
        opus::number<T::T_M>{},                                        // p r % 2
        opus::number<T::W_M / T::T_M>{},                               // p r / 2
        opus::number<T::E_C>{},                                        // p chunk, fixed to 0
        opus::number<T::WARP_SIZE / T::W_M>{},                         // p g
        opus::number<T::VEC_A>{});                                     // y 16 bytes
    constexpr auto dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::p_dim{}, opus::y_dim{}));
    const int lane_r = lane_id % T::W_M;
    return opus::make_layout<T::VEC_A>(
        shape,
        opus::make_tuple(opus::number<T::smem_linear_wave_a>{}, opus::number<2 * E * T::smem_linear_wave_a>{},
                         opus::number<E * T::smem_linear_wave_a>{}, opus::number<T::B_K_BYTES>{},
                         opus::number<T::B_K_BYTES / T::E_C>{}, opus::number<T::VEC_A>{}, 1_I),
        opus::unfold_p_coord(dim, opus::tuple{wave_id, lane_r % T::T_M, lane_r / T::T_M, 0_I,
                             (lane_id / T::W_M) ^ (((lane_r / T::T_M) ^ ((lane_r % T::T_M) << 2)) & 7)}));
}

// ----------------------------------------------------------------------------
// M/N-contiguous operand ([K, rows]): half tile = 64 k rows x W = 64 E bytes.
// ----------------------------------------------------------------------------
// 32-byte window swizzle: LDS 16-byte slot t of k row k holds global window (t / 2) ^ mn_swz<E>(k); the
// 8 rows of one TR16 pass (k = 8 g + 4 h + b) then land on 8 distinct 32-byte slots of the 256-byte
// bank period.
template<int E>
__device__ __host__ inline constexpr int mn_swz(int k) {
    if constexpr (E == 4) return (k & 7) ^ (((k >> 3) & 1) * 4);
    else if constexpr (E == 2) return ((k >> 1) & 1) | (((k >> 3) & 1) << 1);
    else return (k >> 3) & 1;
}

// ----------------------------------------------------------------------------
// Helpers
// ----------------------------------------------------------------------------
// LDS-DMA of all issues of a layout with one m0: the issue's LDS offset goes into the MUBUF immediate
// (it is added to the global address too, so it is subtracted from the per-issue voffset).
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
constexpr int SCHED_VMEM_READ = 0x820;   // 0x020 VMEM read | 0x800 LDS DMA (buffer_load ... lds)
constexpr int SCHED_DS_READ = 0x100;

// Schedule of one quadrant: SLOTS MFMA pairs; DS LDS reads front-loaded, at most CAP per pair; VM
// buffer_loads spread evenly from the first pair.
template<int SLOTS, int DS, int CAP, int VM>
__device__ inline void sched_gen() {
    opus::static_for<SLOTS>([&](auto j_) {
        constexpr int j = decltype(j_)::value;
        constexpr int left = DS - CAP * j;
        constexpr int ds = CAP == 0 ? 0 : (left <= 0 ? 0 : (left > CAP ? CAP : left));
        constexpr int vm = ((j + 1) * VM + SLOTS - 1) / SLOTS - (j * VM + SLOTS - 1) / SLOTS;
        __builtin_amdgcn_sched_group_barrier(SCHED_MFMA, 2, 0);
        if constexpr (ds > 0) __builtin_amdgcn_sched_group_barrier(SCHED_DS_READ, ds, 0);
        if constexpr (vm > 0) __builtin_amdgcn_sched_group_barrier(SCHED_VMEM_READ, vm, 0);
    });
}

// One MFMA operand per lane: 2 K-steps x 8 bf16 = 8 registers, pinned as ONE value (pinning two
// 4-register halves leaves their concatenation unpinned, and the compiler then copies every operand).
using chunk16_t = opus::vector_t<opus::i32_t, 4>;
using op8_t = opus::vector_t<opus::i32_t, 8>;

// 2 MFMAs 16x16x32 on one fragment: K-step s is registers 4 s .. 4 s + 3 of the operands.
// PIN: C_REG >= 0: accumulator pinned at a[C_REG]; C_REG <= -2: at v[-C_REG - 2]. Else left to the allocator.
template<bool PIN, int C_REG, class MMA, class CF>
__device__ inline void mma_bf16(MMA& mma, const op8_t& a8, const op8_t& b8, CF& c) {
    (void)mma;
    using base_mma = typename opus::remove_cvref_t<MMA>::MMA;
    using bf16x8 = opus::vector_t<opus::bf16_t, 8>;
    opus::static_for<2>([&](auto s_) {
        constexpr int s = decltype(s_)::value;
        const bf16x8 a = __builtin_bit_cast(bf16x8, chunk16_t{a8[4 * s], a8[4 * s + 1], a8[4 * s + 2], a8[4 * s + 3]});
        const bf16x8 b = __builtin_bit_cast(bf16x8, chunk16_t{b8[4 * s], b8[4 * s + 1], b8[4 * s + 2], b8[4 * s + 3]});
        if constexpr (!PIN) {
            c = base_mma{}(a, b, c);
        } else {
#if OPUS_A16W16_4WAVE_HAS_PIN
            if constexpr (C_REG >= 0) {
                [[clang::amdgpu_pin_agpr(C_REG)]] c = base_mma{}(a, b, c);
            } else {
                [[clang::amdgpu_pin_vgpr(-C_REG - 2)]] c = base_mma{}(a, b, c);
            }
#endif
        }
    });
}

// Register of fragment (m, n) of quadrant (HM, HN): linear index in quadrant order C00, C10, C01, C11,
// the first 128 in a[128:255], the rest in v[128:].
template<class T, int HM, int HN, int M, int N>
constexpr int c_reg() {
    constexpr int idx = (HN * 2 + HM) * T::E_M * T::E_N * 4 + (M * T::E_N + N) * 4;
    if constexpr (idx < 128) return 128 + idx;
    else return -(128 + idx - 128) - 2;
}

// Wave-uniform copy of a pointer: buffer descriptors built from it stay in SGPRs (no waterfall loops).
template<class P>
__device__ inline P uniform_ptr(P ptr) {
    const unsigned long long v = reinterpret_cast<unsigned long long>(ptr);
    const unsigned lo = __builtin_amdgcn_readfirstlane(static_cast<unsigned>(v));
    const unsigned hi = __builtin_amdgcn_readfirstlane(static_cast<unsigned>(v >> 32));
    return reinterpret_cast<P>((static_cast<unsigned long long>(hi) << 32) | lo);
}

} // namespace opus_a16w16_4wave
#endif // __gfx950__
#endif // __HIP_DEVICE_COMPILE__

// ============================================================================
// Kernel (definition visible on both passes; the host pass only needs the stub)
// ============================================================================
template<typename Traits>
__global__ __launch_bounds__(Traits::BLOCK_SIZE, Traits::WG_PER_CU) void gemm_a16w16_4wave_kernel(opus_gemm_a16w16_4wave_kargs_gfx950 kargs) {
#ifdef __HIP_DEVICE_COMPILE__
#if defined(__gfx950__)
    using namespace opus;
    using namespace opus_a16w16_4wave;
    using T = opus::remove_cvref_t<Traits>;
    using D_C = typename T::D_C;
    using D_ACC = typename T::D_ACC;
    using D_OUT = typename T::D_OUT;    // bf16 C, or fp32 partial sums (SPLITK)
    using D_BIAS = typename T::D_BIAS;
    using D_AB = opus::u8_t;            // operands are addressed in bytes
    static_assert(!T::PIN || OPUS_A16W16_4WAVE_HAS_PIN, "PIN instances need the AGPR-pinning toolchain ([[clang::amdgpu_pin_agpr]])");

    // Workgroup -> (tile, split): splits of one tile are adjacent (as the aiter flatmm split-K kernels).
    const int num_tiles_n = ceil_div(kargs.n, T::B_N);
    int tile_id = block_id_x(), split_id = 0;
    if constexpr (T::SPLITK) {
        split_id = tile_id % kargs.split_k;
        tile_id = tile_id / kargs.split_k;
    }
    const int block_m = tile_id / num_tiles_n;
    const int block_n = tile_id % num_tiles_n;
    // K tiles of this split: k_iters / split_k each, the first k_iters % split_k one more (>= 2 needed).
    const int k_iters = kargs.k / T::B_K;
    const int ks_base = T::SPLITK ? k_iters / kargs.split_k : k_iters;
    const int ks_rem = T::SPLITK ? k_iters % kargs.split_k : 0;
    const int loops = ks_base + (split_id < ks_rem ? 1 : 0);
    const int k_tile0 = split_id * ks_base + (split_id < ks_rem ? split_id : ks_rem);
    const int row = block_m * T::B_M;
    const int col = block_n * T::B_N;
    const int batch_id = block_id_z();
    const int wave_id = __builtin_amdgcn_readfirstlane(thread_id_x() / T::WARP_SIZE);
    const int lane_id = thread_id_x() % T::WARP_SIZE;
    const int wave_id_m = wave_id % T::T_M;
    const int wave_id_n = wave_id / T::T_M;
    const int stride_a = kargs.stride_a * 2;   // bytes
    const int stride_b = kargs.stride_b * 2;

    // Operands, bounded so that reads past the matrix return 0: [rows, K] -> rows < M (N),
    // [K, rows] -> k < K (columns >= M only feed C rows that are never stored).
    // The split's K range starts k_tile0 tiles in (64 k rows of [K, rows], 128 bytes of [rows, K]).
    const int a_k0 = T::A_T ? k_tile0 * T::B_K * stride_a : k_tile0 * T::B_K_BYTES;
    const int b_k0 = T::B_T ? k_tile0 * T::B_K * stride_b : k_tile0 * T::B_K_BYTES;
    auto g_a = make_gmem(uniform_ptr(reinterpret_cast<const D_AB*>(kargs.ptr_a) + batch_id * kargs.stride_a_batch * 2 + row * (T::A_T ? 2 : stride_a) + a_k0),
                         __builtin_amdgcn_readfirstlane(static_cast<unsigned int>(T::A_T ? kargs.k * stride_a - row * 2 - a_k0 : (kargs.m - row) * stride_a - a_k0)));
    auto g_b = make_gmem(uniform_ptr(reinterpret_cast<const D_AB*>(kargs.ptr_b) + batch_id * kargs.stride_b_batch * 2 + col * (T::B_T ? 2 : stride_b) + b_k0),
                         __builtin_amdgcn_readfirstlane(static_cast<unsigned int>(T::B_T ? kargs.k * stride_b - col * 2 - b_k0 : (kargs.n - col) * stride_b - b_k0)));
    // Output: C (rows >= M dropped by the descriptor, columns >= N by the store offset), or the split's
    // fp32 workspace tile [split][batch][padded_M][padded_N] written whole.
    const int out_stride = T::SPLITK ? kargs.stride_ws : kargs.stride_c;
    const int c_m_rem = T::SPLITK ? T::B_M : kargs.m - row;
    const int c_n_rem = T::SPLITK ? T::B_N : kargs.n - col;
    D_OUT* const out_ptr = T::SPLITK
        ? reinterpret_cast<D_OUT*>(kargs.ptr_ws) + (static_cast<long>(split_id) * kargs.batch + batch_id) * kargs.stride_ws_batch + row * out_stride + col
        : reinterpret_cast<D_OUT*>(kargs.ptr_c) + batch_id * kargs.stride_c_batch + row * out_stride + col;
    auto g_c = make_gmem(out_ptr, __builtin_amdgcn_readfirstlane(static_cast<unsigned int>(c_m_rem * out_stride * sizeof(D_OUT))));

    // K-contiguous copy / read layouts (B: A's with the waves swapped).
    const auto u_ga = make_layout_g_k<T, T::HALF_B_M>(lane_id, wave_id_m, wave_id_n, stride_a);
    const auto u_gb = make_layout_g_k<T, T::HALF_B_N>(lane_id, wave_id_n, wave_id_m, stride_b);
    const auto u_sa = make_layout_s_k<T, T::HALF_B_M>(wave_id_m, wave_id_n);
    const auto u_sb = make_layout_s_k<T, T::HALF_B_N>(wave_id_n, wave_id_m);
    const auto u_ra = make_layout_r_k<T, T::E_M>(lane_id, wave_id_m);
    const auto u_rb = make_layout_r_k<T, T::E_N>(lane_id, wave_id_n);

    constexpr int smem_a_elem = T::smem_m_rep * (T::smem_linear_wave_a + T::smem_padding);   // one half, bytes
    constexpr int smem_b_elem = T::smem_n_rep * (T::smem_linear_wave_b + T::smem_padding);
    constexpr int smem_a_bytes = smem_a_elem * 4;                                            // 2 stages x 2 halves
    constexpr int smem_b_bytes = smem_b_elem * 4;
    static_assert(T::LDS_BYTES >= smem_a_bytes + smem_b_bytes);
    __shared__ __attribute__((aligned(16))) char smem_all[T::LDS_BYTES];
    auto s_a = make_smem(reinterpret_cast<D_AB*>(smem_all));
    auto s_b = make_smem(reinterpret_cast<D_AB*>(smem_all + smem_a_bytes));

    auto mma16 = make_tiled_mma<opus::bf16_t, opus::bf16_t, D_ACC>(
        seq<T::E_M, T::E_N, 1>{},
        seq<T::T_M, T::T_N, T::T_K>{},
        seq<T::W_M, T::W_N, T::W_K>{},
        mfma_adaptor_swap_ab{});
    using MMA = decltype(mma16);
    op8_t v_a[2][T::E_M];   // [half][repeat]
    op8_t v_b[2][T::E_N];
    using CFrag = opus::vector_t<D_ACC, MMA::mma_c_len>;   // one 16x16 C fragment, 4 fp32 per lane
    CFrag v_c[2][2][T::E_M * T::E_N];
    opus::static_for<2>([&](auto hm) { opus::static_for<2>([&](auto hn) { opus::static_for<T::E_M * T::E_N>([&](auto f) {
        clear(v_c[hm.value][hn.value][f.value]);
    }); }); });

    // ---- offsets ----
    auto ga_offset = [&](int half_tile, int tile_k) {
        if constexpr (T::A_T) return tile_k * 64 * stride_a + half_tile * T::HALF_B_M * 2;    // 64 k rows per tile
        else return half_tile * T::HALF_B_M * stride_a + tile_k * T::B_K_BYTES;
    };
    auto gb_offset = [&](int half_tile, int tile_k) {
        if constexpr (T::B_T) return tile_k * 64 * stride_b + half_tile * T::HALF_B_N * 2;
        else return half_tile * T::HALF_B_N * stride_b + tile_k * T::B_K_BYTES;
    };
    auto sa_offset = [&](int stage, int half_tile) { return (stage * 2 + half_tile) * smem_a_elem; };
    auto sb_offset = [&](int stage, int half_tile) { return (stage * 2 + half_tile) * smem_b_elem; };

    // ---- global -> LDS ----
    // M/N-contiguous: wave w = wave_m * 2 + wave_n copies k rows 16 w .. 16 w + 15 of the half in E
    // issues of 1 KB (16 / E rows of 64 E bytes), LDS chunk w E + i.
    const int mn_w = wave_id_m * 2 + wave_id_n;
    auto prefetch_mn = [&](auto& g, auto& s, auto e_, int stride, int s_ofs, int g_ofs) {
        constexpr int EE = decltype(e_)::value, RPI = 16 / EE;
        char* const base = reinterpret_cast<char*>(s.ptr) + s_ofs + mn_w * EE * 1024;
        const int kr = lane_id / (4 * EE), t16 = lane_id % (4 * EE);
        opus::static_for<EE>([&](auto i_) {
            constexpr int i = decltype(i_)::value;
            constexpr int io = i * 1024;
            const int kk = 16 * mn_w + i * RPI + kr;
            const int gofs = kk * stride + (((t16 >> 1) ^ mn_swz<EE>(kk)) * 32 + (t16 & 1) * 16);
            g.template async_load<16>(base, gofs - io, g_ofs, opus::number<io>{});
        });
    };
    auto prefetch_a = [&](int stage, int tile, int half_tile) {
        if constexpr (T::A_T) prefetch_mn(g_a, s_a, number<T::E_M>{}, stride_a, sa_offset(stage, half_tile), ga_offset(half_tile, tile));
        else async_load_imm<T::VEC_A>(g_a, s_a.ptr, u_ga, u_sa + sa_offset(stage, half_tile), ga_offset(half_tile, tile));
    };
    auto prefetch_b = [&](int stage, int tile, int half_tile) {
        if constexpr (T::B_T) prefetch_mn(g_b, s_b, number<T::E_N>{}, stride_b, sb_offset(stage, half_tile), gb_offset(half_tile, tile));
        else async_load_imm<T::VEC_B>(g_b, s_b.ptr, u_gb, u_sb + sb_offset(stage, half_tile), gb_offset(half_tile, tile));
    };

    // ---- LDS -> registers ----
    // K-contiguous: 2 ds_read_b128 per operand (chunk 1 = slot ^ 4: c1_delta).
    const auto ra_offs = opus::layout_to_offsets<T::VEC_A>(u_ra);   // [repeat]
    const auto rb_offs = opus::layout_to_offsets<T::VEC_B>(u_rb);
    const int slot0 = ((lane_id / T::W_M) ^ (((lane_id % T::W_M) / 2) ^ (((lane_id % T::W_M) % 2) << 2))) & 7;
    const int c1_delta = ((slot0 ^ 4) - slot0) * T::VEC_A;
    auto load_k = [&](auto& sm_, const auto& r_offs, auto& v, int s_ofs, auto agpr_) {
        constexpr int E = sizeof(v) / sizeof(v[0]);
        opus::static_for<E>([&](auto e) {
            constexpr int agpr = decltype(agpr_)::value + e.value * 8;
            const auto c0 = __builtin_bit_cast(chunk16_t, load<T::VEC_A>(sm_, r_offs[e.value] + s_ofs));
            const auto c1 = __builtin_bit_cast(chunk16_t, load<T::VEC_A>(sm_, r_offs[e.value] + s_ofs + c1_delta));
            if constexpr (!T::PIN) {
                v[e.value] = __builtin_shufflevector(c0, c1, 0, 1, 2, 3, 4, 5, 6, 7);
            } else {
#if OPUS_A16W16_4WAVE_HAS_PIN
                [[clang::amdgpu_pin_agpr(agpr)]] v[e.value] = __builtin_shufflevector(c0, c1, 0, 1, 2, 3, 4, 5, 6, 7);
#endif
            }
        });
    };
    // M/N-contiguous: operand e (rows e * 32 + wave * 16 + r -> 32-byte window 2 e + wave), K-step s, pass h
    // reads k rows 32 s + 8 g + 4 h + b (b = (lane % 16) / 4 supplies the row, 8 bytes at (lane % 4) * 8)
    // with ds_read_b64_tr_b16; per-lane bases rb[e][h], K-step and half are immediates.
    using i32x2_t = opus::vector_t<opus::i32_t, 2>;
    auto tr16 = [&](auto& sm_, int byte_ofs) {
        // pointer arithmetic on the view's (restrict) LDS pointer keeps the no-alias fact (no vmcnt(0) before it)
        return __builtin_bit_cast(i32x2_t, __builtin_amdgcn_ds_read_tr16_b64_v4i16(
            reinterpret_cast<OPUS_LDS_ADDR opus::vector_t<short, 4>*>(sm_.ptr + byte_ofs)));
    };
    auto mn_bases = [&](auto e_, int wave) {
        constexpr int EE = decltype(e_)::value, W = 64 * EE;
        opus::array<int, 2 * EE> rb;
        opus::static_for<EE>([&](auto e) { opus::static_for<2>([&](auto h) {
            const int kk = 8 * (lane_id / 16) + 4 * h.value + ((lane_id % 16) >> 2);
            rb[e.value * 2 + h.value] = kk * W + (((2 * e.value + wave) ^ mn_swz<EE>(kk)) * 32) + (lane_id & 3) * 8;
        }); });
        return rb;
    };
    const auto mn_rb_a = mn_bases(number<T::E_M>{}, wave_id_m);
    const auto mn_rb_b = mn_bases(number<T::E_N>{}, wave_id_n);
    auto load_mn = [&](auto& sm_, const auto& rb, auto& v, auto half_, auto agpr_) {
        constexpr int EE = sizeof(v) / sizeof(v[0]), W = 64 * EE;
        constexpr int o = decltype(half_)::value * 64 * W;              // the half's offset in bytes
        opus::static_for<EE>([&](auto e) {
            constexpr int agpr = decltype(agpr_)::value + e.value * 8;
            const int b0 = rb[e.value * 2], b1 = rb[e.value * 2 + 1];
            const i32x2_t q0 = tr16(sm_, b0 + o), q1 = tr16(sm_, b1 + o);
            const i32x2_t q2 = tr16(sm_, b0 + o + 32 * W), q3 = tr16(sm_, b1 + o + 32 * W);
            const chunk16_t c0 = __builtin_shufflevector(q0, q1, 0, 1, 2, 3), c1 = __builtin_shufflevector(q2, q3, 0, 1, 2, 3);
            if constexpr (!T::PIN) {
                v[e.value] = __builtin_shufflevector(c0, c1, 0, 1, 2, 3, 4, 5, 6, 7);
            } else {
#if OPUS_A16W16_4WAVE_HAS_PIN
                [[clang::amdgpu_pin_agpr(agpr)]] v[e.value] = __builtin_shufflevector(c0, c1, 0, 1, 2, 3, 4, 5, 6, 7);
#endif
            }
        });
    };
    auto load_a = [&](auto& sm_, auto half_) {
        constexpr int half = decltype(half_)::value;
        if constexpr (T::A_T) load_mn(sm_, mn_rb_a, v_a[half], half_, number<half * 32>{});
        else load_k(sm_, ra_offs, v_a[half], sa_offset(0, half), number<half * 32>{});
    };
    auto load_b = [&](auto& sm_, auto half_) {
        constexpr int half = decltype(half_)::value;
        if constexpr (T::B_T) load_mn(sm_, mn_rb_b, v_b[half], half_, number<64 + half * 32>{});
        else load_k(sm_, rb_offs, v_b[half], sb_offset(0, half), number<64 + half * 32>{});
    };

    // LDS reads go through __restrict__ copies of the views passed as function parameters (clang ignores
    // __restrict__ on locals): otherwise the reads cannot be told apart from the in-flight LDS-DMA copies
    // and the compiler puts a vmcnt(0) before them.
    auto views = [&](OPUS_LDS_ADDR char* __restrict__ read_a, OPUS_LDS_ADDR char* __restrict__ read_b, auto&& body) {
        auto rd_a = s_a; rd_a.ptr = read_a;
        auto rd_b = s_b; rd_b.ptr = read_b;
        body(rd_a, rd_b);
    };

    // One quadrant (HM, HN): 2 MFMAs per fragment, accumulators pinned (c_reg).
    constexpr int frags = T::E_M * T::E_N;
    auto quadrant = [&](auto hm_, auto hn_) {
        constexpr int hm = decltype(hm_)::value, hn = decltype(hn_)::value;
        opus::static_for<frags>([&](auto i) {
            constexpr int m = decltype(i)::value / T::E_N, n = decltype(i)::value % T::E_N;
            mma_bf16<T::PIN, c_reg<T, hm, hn, m, n>()>(mma16, v_a[hm][m], v_b[hn][n], v_c[hm][hn][m * T::E_N + n]);
        });
    };

    // Schedule: SL MFMA pairs per quadrant; DSA / DSB LDS reads per half of A / B; VMH copies per half (A + B).
    constexpr int SL = frags;
    constexpr int DSA = (T::A_T ? 4 : 2) * T::E_M, DSB = (T::B_T ? 4 : 2) * T::E_N;
    constexpr int VMH = T::a_buffer_load_insts + T::b_buffer_load_insts;

    // One K tile t with tile t+1 in stage next_stage: C10(t) | C01(t) | C11(t) | C00(t+1). PREFETCH: copy
    // tile t+2 into stage(t) (B_a: every wave is done reading stage(t)); B_b: tile t+1 has landed.
    auto k_tile = [&](auto& rd_a, auto& rd_b, int stage, int tile, auto prefetch_) {
        constexpr bool PREFETCH = decltype(prefetch_)::value;
        if constexpr (PREFETCH) {
            s_waitcnt_lgkmcnt(0_I);
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);
            prefetch_a(stage, tile + 2, 0);
            prefetch_b(stage, tile + 2, 0);
        }
        // C10(t) + A0, B0 of tile t+2
        quadrant(number<1>{}, number<0>{});
        sched_gen<SL, 0, 0, PREFETCH ? VMH : 0>();
        __builtin_amdgcn_sched_barrier(0);
        if constexpr (PREFETCH) s_waitcnt_vmcnt(number<VMH>{});   // the copies just issued may still fly
        else s_waitcnt_vmcnt(0_I);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);
        // C01(t) + B0 of tile t+1 + A1, B1 of tile t+2
        load_b(rd_b, number<0>{});
        if constexpr (PREFETCH) {
            prefetch_a(stage, tile + 2, 1);
            prefetch_b(stage, tile + 2, 1);
        }
        quadrant(number<0>{}, number<1>{});
        sched_gen<SL, DSB, T::SCHED_C01CAP, PREFETCH ? VMH : 0>();
        __builtin_amdgcn_sched_barrier(0);
        // C11(t) + A0 of tile t+1
        load_a(rd_a, number<0>{});
        quadrant(number<1>{}, number<1>{});
        sched_gen<SL, DSA, T::SCHED_C11CAP, 0>();
        __builtin_amdgcn_sched_barrier(0);
        // C00(t+1) + A1, B1 of tile t+1
        load_a(rd_a, number<1>{});
        load_b(rd_b, number<1>{});
        quadrant(number<0>{}, number<0>{});
        sched_gen<SL, DSA + DSB, T::SCHED_C00CAP, 0>();
        __builtin_amdgcn_sched_barrier(0);
    };

    // ===== Prologue: copy tiles 0 and 1, wait for tile 0, read it and compute C00(0) =====
    prefetch_a(0, 0, 0);
    prefetch_b(0, 0, 0);
    prefetch_a(0, 0, 1);
    prefetch_b(0, 0, 1);
    if (loops > 1) {
        prefetch_a(1, 1, 0);
        prefetch_b(1, 1, 0);
        prefetch_a(1, 1, 1);
        prefetch_b(1, 1, 1);
    }
    if (loops > 1) {
        s_waitcnt_vmcnt(number<2 * VMH>{});   // tile 1 may fly
    } else {
        s_waitcnt_vmcnt(0_I);
    }
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);
    views(s_a.ptr, s_b.ptr, [&](auto& rd_a, auto& rd_b) {   // tile 1 may still be in flight
        load_a(rd_a, number<0>{});
        load_b(rd_b, number<0>{});
        __builtin_amdgcn_sched_barrier(0);
        load_a(rd_a, number<1>{});
        load_b(rd_b, number<1>{});
        quadrant(number<0>{}, number<0>{});
        sched_gen<SL, DSA + DSB, 2, 0>();
        __builtin_amdgcn_sched_barrier(0);
    });

    // ===== Main loop: 2 K tiles per trip, stages compile-time =====
    // Keep the control flow straight (once-loops, plain blocks): pinned values crossing loop exits or
    // branch joins lose their AGPR assignment.
    int tile = 0;
#pragma clang loop unroll(disable)
    for (; tile + 3 < loops; tile += 2) {
        views(s_a.ptr + sa_offset(1, 0), s_b.ptr + sb_offset(1, 0), [&](auto& rd_a, auto& rd_b) {
            k_tile(rd_a, rd_b, 0, tile, true_type{});
        });
        views(s_a.ptr + sa_offset(0, 0), s_b.ptr + sb_offset(0, 0), [&](auto& rd_a, auto& rd_b) {
            k_tile(rd_a, rd_b, 1, tile + 1, true_type{});
        });
    }
    // At most one more tile with a prefetch (odd number of them).
#pragma clang loop unroll(disable)
    for (int once = 0; once < 1 && tile + 2 < loops; ++once, ++tile) {
        const int stage = tile & 1;
        views(s_a.ptr + sa_offset(stage ^ 1, 0), s_b.ptr + sb_offset(stage ^ 1, 0), [&](auto& rd_a, auto& rd_b) {
            k_tile(rd_a, rd_b, stage, tile, true_type{});
        });
    }

    // ===== Epilogue =====
    {   // second-to-last tile (loops >= 2): no more copies
        const int next_stage = (tile & 1) ^ 1;
        views(s_a.ptr + sa_offset(next_stage, 0), s_b.ptr + sb_offset(next_stage, 0), [&](auto& rd_a, auto& rd_b) {
            k_tile(rd_a, rd_b, 0, tile, false_type{});
        });
        ++tile;
    }
    {   // last tile: C10, C01, C11
        quadrant(number<1>{}, number<0>{});
        sched_gen<SL, 0, 0, 0>();
        __builtin_amdgcn_sched_barrier(0);
        quadrant(number<0>{}, number<1>{});
        sched_gen<SL, 0, 0, 0>();
        __builtin_amdgcn_sched_barrier(0);
        quadrant(number<1>{}, number<1>{});
        sched_gen<SL, 0, 0, 0>();
        __builtin_amdgcn_sched_barrier(0);
    }

    // ===== Write back C through LDS =====
    // A wave owns only 16 columns of a row per fragment, so direct fragment stores would leave every
    // 128-byte line partially written. C is staged in the idle LDS, rows padded by 8 elements (the 16
    // rows of a fragment step across the banks): as one image of the tile when it fits (one region per
    // quadrant, no reuse), else quadrant by quadrant in 2 regions (C00 / C01 -> 0, C10 / C11 -> 1).
    // Per quadrant: stage -> sync -> 16-byte copies (non-temporal), columns >= N dropped.
    constexpr bool FULL = T::C_STAGE_FULL;
    constexpr int c_lds_stride = FULL ? T::B_N + 8 : T::HALF_B_N + 8;
    constexpr int VEC_OUT = 16 / sizeof(D_OUT);
    static_assert((FULL ? T::B_M : 2 * T::HALF_B_M) * c_lds_stride * sizeof(D_OUT) <= T::LDS_BYTES);
    auto s_c = make_smem(reinterpret_cast<D_OUT*>(smem_all));
    auto p_coord_c = opus::make_tuple(wave_id_m, lane_id % mma16.grpn_c, wave_id_n, lane_id / mma16.grpn_c);
    auto u_sc = partition_layout_c<T::VEC_C>(mma16, opus::make_tuple(opus::number<c_lds_stride>{}, 1_I), p_coord_c);
    auto sc_offset = [&](int half_tile_m, int half_tile_n) {
        return half_tile_m * T::HALF_B_M * c_lds_stride + (FULL ? half_tile_n * T::HALF_B_N : 0);
    };
    const int output_thread_id = static_cast<int>(thread_id_x());
    auto copy_output = [&](int half_tile_m, int half_tile_n) {
        opus::static_for<T::HALF_B_M * T::HALF_B_N / (T::BLOCK_SIZE * VEC_OUT)>([&](auto i) {
            const int linear = output_thread_id * VEC_OUT + decltype(i)::value * T::BLOCK_SIZE * VEC_OUT;
            const int r = linear / T::HALF_B_N + half_tile_m * T::HALF_B_M;
            const int cl = linear % T::HALF_B_N, c = cl + half_tile_n * T::HALF_B_N;
            const auto value = load<VEC_OUT>(s_c, r * c_lds_stride + (FULL ? c : cl));
            // columns >= N: 2^31 bytes, far outside the descriptor, dropped by the hardware (no branch;
            // a larger offset would wrap the 32-bit byte offset)
            constexpr int OOB = static_cast<int>((1u << 31) / sizeof(D_OUT));
            store<VEC_OUT>(g_c, value, c < c_n_rem ? r * out_stride + c : OOB, 0, opus::number<2>{});
        });
    };
    // Bias (per output column, fp32 add before the cast): the E_N column groups of this lane in both
    // halves of N, loaded up front (columns >= N read 0).
    using BiasV = opus::vector_t<D_BIAS, T::VEC_C>;
    BiasV v_bias[2][T::E_N];
    if constexpr (T::HAS_BIAS) {
        auto g_bias = make_gmem(uniform_ptr(reinterpret_cast<const D_BIAS*>(kargs.ptr_bias) + batch_id * kargs.stride_bias_batch + col),
                                __builtin_amdgcn_readfirstlane(static_cast<unsigned int>((kargs.n - col) * sizeof(D_BIAS))));
        const auto sc_offs = opus::layout_to_offsets<T::VEC_C>(u_sc);   // fragment f = m * E_N + n
        opus::static_for<2>([&](auto hn) { opus::static_for<T::E_N>([&](auto n) {
            v_bias[hn.value][n.value] = g_bias.template load<T::VEC_C>(hn.value * T::HALF_B_N + sc_offs[n.value] % c_lds_stride);
        }); });
    }
    auto store_quadrant = [&](const auto& acc, auto half_tile_m_, auto half_tile_n_) {
        constexpr int hm = decltype(half_tile_m_)::value, hn = decltype(half_tile_n_)::value;
        typename MMA::vtype_c quad;
        opus::static_for<T::E_M * T::E_N>([&](auto f) {
            auto frag = acc[f.value];
            if constexpr (T::HAS_BIAS) {
                opus::static_for<T::VEC_C>([&](auto j) {
                    frag[j.value] += static_cast<float>(v_bias[hn][f.value % T::E_N][j.value]);
                });
            }
            opus::set_slice(quad, frag, opus::number<f.value * MMA::mma_c_len>{}, opus::number<(f.value + 1) * MMA::mma_c_len>{});
        });
        if constexpr (!FULL && hn == 1) {   // region hm is reused: every wave is done copying (hm, 0) out
            s_waitcnt_lgkmcnt(0_I);
            __builtin_amdgcn_s_barrier();
        }
        store<T::VEC_C>(s_c, cast<D_OUT>(quad), u_sc + sc_offset(hm, hn));
        __builtin_amdgcn_sched_barrier(0);
        if constexpr (!(hm == 1 && hn == 0)) s_waitcnt_vmcnt(0_I);
        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);
        copy_output(hm, hn);
    };
    // all waves are done with their last operand reads
    s_waitcnt_vmcnt(0_I);
    s_waitcnt_lgkmcnt(0_I);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);
    store_quadrant(v_c[0][0], number<0>{}, number<0>{});
    store_quadrant(v_c[1][0], number<1>{}, number<0>{});
    store_quadrant(v_c[0][1], number<0>{}, number<1>{});
    store_quadrant(v_c[1][1], number<1>{}, number<1>{});
#endif // __gfx950__
#endif // __HIP_DEVICE_COMPILE__
}
