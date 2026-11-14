// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Small-tile MXFP4 x MXFP4 NT GEMM (gfx950), shuffle layout (aiter gemm_a4w4(bpreshuffle=True)): A [M, K/2] fp4x2
// row-major, B shuffle_weight(16, 16), scales e8m0_shuffle (1x32 UE8M0), C [M, N] bf16.  M <= ~512.  opus, standard
// ROCm toolchain (no pin attributes).  Traits / kargs: opus_gemm_traits_a4w4_mxfp4_nt_small_gfx950.cuh.
//
// 1x4 waves (T_M = 1, T_N = 4): the 4 waves split the tile along N and share A.
//   A: global -> LDS by LDS-DMA into a STAGES-deep ring (one __shared__ object per stage), ds_read one step ahead;
//   B (each wave's E_N 16-column blocks) and the raw e8m0_shuffle scale dwords: global -> a STAGES-deep register ring.
//   KS = 2 / 4 splits the 4 waves KS ways along K instead (T_N = 4 / KS); the partial tiles are summed through LDS once.
// MFMA v_mfma_scale_f32_16x16x128_f8f6f4 with B as srcA (each lane holds 4 consecutive columns of one C row); the scale
//   byte is picked by a compile-time opsel from the raw dword ([kk0 r, kk0 r+16, kk1 r, kk1 r+16] of a 32-row block).
// Main loop (asm style): step t waits until step t+1 landed (counted vmcnt), barrier, ds_reads A(t+1), and runs the MFMAs
//   of t interleaved with the copies of step t+STAGES into t's slot (sched_group_barrier).  K tiles past the end load
//   through zero-range buffers (no branch in the loop).
// Tails: M / N by buffer ranges (A rows, B columns, scales) + a 16-column C mask; no host padding.
// Split-K (split_k = S > 1): each block writes its fp32 partial tile to the workspace, splitk_reduce_kernel
//   (splitk_reduce_gfx950.cuh) sums the S partials into C.
#pragma once
#include "opus_gemm_traits_a4w4_mxfp4_nt_small_gfx950.cuh"

namespace opus_mxfp4_small {
using opus::operator""_I;
using fp4x32_t = opus::vector_t<opus::i32_t, 4>;   // one fp4 MFMA operand: 32 fp4 = 16 bytes per lane
using i32x8_t = opus::vector_t<opus::i32_t, 8>;

// ===========================================================================
// A: global -> LDS (LDS-DMA) -> VGPR
// ===========================================================================
// Create layout for loading A matrix from global memory: one LDS-DMA issue copies an 8-row x 128 B chunk (lane l lands
// at l * 16 B), so slot s of row r fetches the 16 B piece s ^ ((r >> 1) & 3) (conflict-free ds_read_b128 below).
template<class T>
__device__ inline auto make_layout_ga(int lane_id, int stride_a) {
    constexpr int threads_k = T::B_K_BYTES / T::VEC_A;

    constexpr auto ga_block_shape = opus::make_tuple(
        opus::number<T::smem_sub_a>{},
        opus::number<threads_k>{},
        opus::number<T::VEC_A>{});

    constexpr auto ga_block_dim = opus::make_tuple(
        opus::make_tuple(opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}));

    const int lane_id_m = lane_id / threads_k;

    return opus::make_layout<T::VEC_A>(
        ga_block_shape,
        opus::unfold_x_stride(ga_block_dim, ga_block_shape, opus::tuple{stride_a, 1_I}),
        opus::unfold_p_coord(ga_block_dim, opus::tuple{lane_id_m, (lane_id % threads_k) ^ ((lane_id_m >> 1) & 3)}));
}

// Create layout for storing A matrix to shared memory: chunk (t, r) of a ring step = K tile t (KS > 1), rows [8 r, 8 r + 8).
template<class T>
__device__ inline constexpr auto make_layout_sa() {
    return opus::make_layout<-1>(opus::make_tuple(opus::number<T::KS>{}, opus::number<T::A_CHUNKS>{}),
                                 opus::make_tuple(opus::number<T::A_TILE_BYTES>{}, opus::number<T::A_CHUNK_BYTES>{}));
}
// Create layout for the wave-uniform global offset of chunk (t, r) (passed as the buffer soffset).
template<class T>
__device__ inline auto make_layout_gas(int stride_a) {
    return opus::make_layout<-1>(opus::make_tuple(opus::number<T::KS>{}, opus::number<T::A_CHUNKS>{}),
                                 opus::make_tuple(opus::number<T::B_K_BYTES>{}, T::smem_sub_a * stride_a));
}

// Create layout for reading A matrix from shared memory to registers: lane holds row 16 e + lane % 16, K bytes
// [64 kk + 16 (lane / 16), +16) of the K tile of its wave group (wave_id_k); offsets[e * E_K + kk].
template<class T>
__device__ inline auto make_layout_ra(int lane_id, int wave_id_k) {
    constexpr auto ra_block_shape = opus::make_tuple(
        opus::number<T::T_K>{},
        opus::number<T::E_M>{},
        opus::number<T::W_M / T::smem_sub_a>{},
        opus::number<T::smem_sub_a>{},
        opus::number<T::E_K>{},
        opus::number<T::WARP_SIZE / T::W_M>{},
        opus::number<T::VEC_A>{});

    constexpr auto ra_block_dim = opus::make_tuple(
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}, opus::p_dim{}, opus::y_dim{}));

    const int lane_id_m = lane_id % T::W_M;
    const int r = lane_id_m % T::smem_sub_a;

    return opus::make_layout<T::VEC_A>(
        ra_block_shape,
        opus::unfold_x_stride(ra_block_dim, ra_block_shape, opus::tuple{opus::number<T::A_CHUNK_BYTES>{}, 1_I}),
        opus::unfold_p_coord(ra_block_dim, opus::tuple{wave_id_k, lane_id_m / T::smem_sub_a, r, (lane_id / T::W_M) ^ ((r >> 1) & 3)}));
}

// ===========================================================================
// B: global -> VGPR (shuffle_weight(16, 16): 1 KB = 16 columns x 128 K per 64 lanes x 16 B)
// ===========================================================================
// Create layout for loading B matrix from global memory to registers: offsets[e * E_K + kk] = the wave's 16-column
// block e, K half kk (the K tile goes into the buffer soffset).
template<class T>
__device__ inline auto make_layout_gb(int lane_id, int stride_b) {
    constexpr auto gb_block_shape = opus::make_tuple(
        opus::number<T::E_N>{},
        opus::number<T::E_K>{},
        opus::number<T::WARP_SIZE>{},
        opus::number<T::VEC_B>{});

    constexpr auto gb_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}),
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::y_dim{}));

    return opus::make_layout<T::VEC_B>(
        gb_block_shape,
        opus::unfold_x_stride(gb_block_dim, gb_block_shape, opus::tuple{T::W_N * stride_b, 1_I}),
        opus::unfold_p_coord(gb_block_dim, opus::tuple{lane_id}));
}

// ===========================================================================
// Scales: raw e8m0_shuffle dwords, global -> VGPR
// ===========================================================================
// Create layout for loading scale factors from global memory: per K tile each 32-row block holds 256 B, lane l reads
// dword l = [kk0 r, kk0 r+16, kk1 r, kk1 r+16], r = l % 16; offsets[j] = 32-row block j (K tile in the soffset).
template<class T, int LOADS>
__device__ inline auto make_layout_gsf(int lane_id, int stride_sf) {
    constexpr auto gsf_block_shape = opus::make_tuple(
        opus::number<LOADS>{},
        opus::number<T::WARP_SIZE>{},
        opus::number<T::VEC_SF>{});

    constexpr auto gsf_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}));

    return opus::make_layout<T::VEC_SF>(
        gsf_block_shape,
        opus::unfold_x_stride(gsf_block_dim, gsf_block_shape, opus::tuple{stride_sf, 1_I}),
        opus::unfold_p_coord(gsf_block_dim, opus::tuple{lane_id}));
}
template<class T> __device__ inline auto make_layout_gsfa(int lane_id, int stride_sfa) { return make_layout_gsf<T, T::SFA_LOADS>(lane_id, stride_sfa); }
template<class T> __device__ inline auto make_layout_gsfb(int lane_id, int stride_sfb) { return make_layout_gsf<T, T::SFB_LOADS>(lane_id, stride_sfb); }

// Register side: the 16-row block e (16-column block n) of K half kk is dword e / 2 (n / 2), byte opsel 2 kk + e % 2
// (n % 2).  A wave with a single 16-row (16-column) block uses the half of its 32-row block chosen by a runtime shift.
template<class T> __device__ inline constexpr int rsfa_dword(int e) { return e / 2; }
template<class T> __device__ inline constexpr int rsfa_opsel(int e, int kk) { return 2 * kk + (T::SFA_HALF ? 0 : e % 2); }
template<class T> __device__ inline constexpr int rsfb_dword(int n) { return n / 2; }
template<class T> __device__ inline constexpr int rsfb_opsel(int n, int kk) { return 2 * kk + (T::SFB_HALF ? 0 : n % 2); }

// ===========================================================================
// C
// ===========================================================================
// Create layout for storing C to global memory: lane holds row 16 e + lane % 16, columns 16 n + 4 (lane / 16) .. +3 of
// its wave's columns; offsets[e * E_N + n] (elements).  Also the layout of the fp32 split-K partial tile.
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

// One __shared__ object per ring stage (kept explicit instead of one opus smem arena): the compiler's waitcnt pass then
// knows an LDS-DMA into stage j cannot alias a ds_read of stage i, instead of waiting for every in-flight copy before
// each read (one array = everything aliases).  LDS accesses below use address_space(3) pointers derived from these
// objects (not opus::smem, whose integer round trip loses the object), for the same reason.
template<int I, int BYTES>
__device__ __forceinline__ char* lds_stage() {
    __shared__ __attribute__((aligned(16))) char buf[BYTES];
    return buf;
}
__device__ __forceinline__ OPUS_LDS_ADDR char* lds_ptr(char* p) { return (OPUS_LDS_ADDR char*)p; }

// sched_group_barrier masks (LLVM AMDGPU)
constexpr int SCHED_MFMA = 0x008;
constexpr int SCHED_VMEM_READ = 0x820;   // 0x020 VMEM read | 0x800 LDS DMA (newer LLVM classifies buffer_load ... lds as LDS DMA)
constexpr int SCHED_DS_READ = 0x100;

template<int MASK, int CNT = 1>
__device__ inline void sched_group() {
    if constexpr (CNT > 0) __builtin_amdgcn_sched_group_barrier(MASK, CNT, 0);
}

// One scaled fp4 MFMA with B as srcA: C fragment row = lane % 16, columns 4 (lane / 16) .. +3.
template<int OPSEL_B, int OPSEL_A, class CF>
__device__ __forceinline__ void mma_scale_one(CF& c, const fp4x32_t& b, const fp4x32_t& a, unsigned v_sfb, unsigned v_sfa) {
    using mma = opus::mfma_f32_16x16x128_fp4_fp4;   // fp4 x fp4 -> fp32, 16x16x128 (only the low 4 dwords of srcA/B are read)
    const i32x8_t b8 = __builtin_shufflevector(b, b, 0, 1, 2, 3, -1, -1, -1, -1);
    const i32x8_t a8 = __builtin_shufflevector(a, a, 0, 1, 2, 3, -1, -1, -1, -1);
    c = mma{}(b8, a8, c, static_cast<int>(v_sfb), static_cast<int>(v_sfa), opus::number<OPSEL_B>{}, opus::number<OPSEL_A>{});
}

// ===========================================================================
// Kernel body
// ===========================================================================
template<class Traits>
__device__ __forceinline__ void gemm_a4w4_mxfp4_nt_small_body(const opus_gemm_mxfp4_small_kargs_gfx950& kargs) {
    using namespace opus;
    using T = opus::remove_cvref_t<Traits>;
    using D_AB = opus::u8_t;     // fp4x2 bytes
    using D_SF = opus::u8_t;     // e8m0 bytes, read as raw dwords
    using D_ACC = opus::fp32_t;
    using D_C = opus::bf16_t;
    using CFrag = opus::vector_t<D_ACC, T::VEC_C>;
    constexpr int D = T::STAGES, G = T::VMEM_PER_STEP;

    const int lane_id = static_cast<int>(opus::lane_id());
    const int wave_id = __builtin_amdgcn_readfirstlane(thread_id_x() / T::WARP_SIZE);
    const int wave_id_n = wave_id % T::T_N, wave_id_k = wave_id / T::T_N;
    const int split_k = kargs.split_k;
    const int row0 = block_id_y() * T::B_M;
    const int col0 = block_id_x() * T::B_N + wave_id_n * T::WAVE_N;   // this wave's first column
    const int batch_id = block_id_z() / split_k, k_split = block_id_z() % split_k;
    // S > 1: read the workspace slot now, so the (host-coherent) load is long done when the partial tile is written.
    float* ws = nullptr;
    if (split_k > 1) ws = static_cast<float*>(kargs.ws_handle->ptr);
    const int k_tiles = kargs.k / T::B_K, k_per = (k_tiles + split_k - 1) / split_k;
    const int k_begin = k_split * k_per;
    const int k_end = k_tiles < k_begin + k_per ? k_tiles : k_begin + k_per;
    const int stride_b = kargs.k / 2;   // bytes per B row (shuffle_weight keeps the row-major row stride)

    // M / N tails: rows / columns past the matrix lie outside the buffer ranges (loads return 0).
    auto clampb = [](long long v) { return v <= 0 ? 0u : (v > 0x7fffffffLL ? 0x7fffffffu : static_cast<unsigned>(v)); };
    const D_AB* pa = reinterpret_cast<const D_AB*>(kargs.ptr_a) + batch_id * kargs.stride_a_batch + static_cast<long long>(row0) * kargs.stride_a;
    const D_AB* pb = reinterpret_cast<const D_AB*>(kargs.ptr_b) + batch_id * kargs.stride_b_batch + static_cast<long long>(col0) * stride_b;
    const D_SF* psfa = reinterpret_cast<const D_SF*>(kargs.ptr_sfa) + batch_id * kargs.stride_sfa_batch + static_cast<long long>(row0 / 32) * kargs.stride_sfa;
    const D_SF* psfb = reinterpret_cast<const D_SF*>(kargs.ptr_sfb) + batch_id * kargs.stride_sfb_batch + static_cast<long long>(col0 / 32) * kargs.stride_sfb;
    const unsigned a_bytes = clampb(static_cast<long long>(kargs.m - row0) * kargs.stride_a);
    const unsigned b_bytes = clampb(static_cast<long long>(kargs.n - col0) * stride_b);
    const unsigned sfa_bytes = clampb(static_cast<long long>((kargs.m + 31) / 32 - row0 / 32) * kargs.stride_sfa);
    const unsigned sfb_bytes = clampb(static_cast<long long>((kargs.n + 31) / 32 - col0 / 32) * kargs.stride_sfb);
    // One 16-row (16-column) block: its half of the 32-row scale block.
    const unsigned sh_a = T::SFA_HALF ? 8u * ((row0 / 16) & 1) : 0u;
    const unsigned sh_b = T::SFB_HALF ? 8u * ((col0 / 16) & 1) : 0u;

    const auto u_ga = make_layout_ga<T>(lane_id, kargs.stride_a);
    constexpr auto u_sa = make_layout_sa<T>();
    const auto u_gas = make_layout_gas<T>(kargs.stride_a);
    const auto u_ra = make_layout_ra<T>(lane_id, wave_id_k);
    const auto u_gb = make_layout_gb<T>(lane_id, stride_b);
    const auto u_gsfa = make_layout_gsfa<T>(lane_id, kargs.stride_sfa);
    const auto u_gsfb = make_layout_gsfb<T>(lane_id, kargs.stride_sfb);

    // Register ring: slot s holds the B operands and scale dwords of ring step s (mod STAGES).
    fp4x32_t v_b[D][T::E_N][T::E_K];
    unsigned v_sfa[D][T::SFA_LOADS], v_sfb[D][T::SFB_LOADS];

    // Copies of ring step `slot` (K tiles kt .. kt + KS - 1; this wave's B / scales: K tile kt + wave_id_k).  K tiles past
    // k_end read through zero-range buffers, so the copy count per step is fixed (counted vmcnt) and the loop has no branch.
    auto issue = [&](auto slot_i, int kt) {
        constexpr int slot = decltype(slot_i)::value;
        const int kw = kt + wave_id_k;
        const bool in = kw < k_end;
        char* const stage = lds_stage<slot, T::STAGE_BYTES>();
        static_for<T::a_buffer_load_insts>([&](auto i) {
            const int b0 = wave_id * T::a_buffer_load_insts + decltype(i)::value;
            const int b = T::A_DMA_EXACT ? b0 : __builtin_amdgcn_readfirstlane(b0 < T::A_DMA_CHUNKS ? b0 : T::A_DMA_CHUNKS - 1);
            const int t = b / T::A_CHUNKS, r = b % T::A_CHUNKS;   // K tile kt + t, rows [8 r, 8 r + 8)
            auto g_a = make_gmem(pa, kt + t < k_end ? a_bytes : 0u);
            g_a.template _async_load<T::VEC_A>(lds_ptr(stage + u_sa(t, r)), u_ga.offsets[0], u_gas(t, r) + kt * T::B_K_BYTES);
        });
        auto g_sfa = make_gmem(psfa, in ? sfa_bytes : 0u);
        static_for<T::SFA_LOADS>([&](auto j) {
            v_sfa[slot][j.value] = __builtin_bit_cast(unsigned, load<T::VEC_SF>(g_sfa, u_gsfa.offsets[j.value], kw * T::SF_K_TILE_BYTES));
        });
        auto g_b = make_gmem(pb, in ? b_bytes : 0u);
        static_for<T::E_N>([&](auto n) { static_for<T::E_K>([&](auto kk) {
            v_b[slot][n.value][kk.value] = __builtin_bit_cast(fp4x32_t,
                load<T::VEC_B>(g_b, u_gb.offsets[n.value * T::E_K + kk.value], kw * T::B_K_BYTES * T::W_N));
        }); });
        auto g_sfb = make_gmem(psfb, in ? sfb_bytes : 0u);
        static_for<T::SFB_LOADS>([&](auto j) {
            v_sfb[slot][j.value] = __builtin_bit_cast(unsigned, load<T::VEC_SF>(g_sfb, u_gsfb.offsets[j.value], kw * T::SF_K_TILE_BYTES));
        });
    };
    auto read_a = [&](fp4x32_t (&v)[T::E_K][T::E_M], auto slot_i) {
        OPUS_LDS_ADDR char* const stage = lds_ptr(lds_stage<decltype(slot_i)::value, T::STAGE_BYTES>());
        static_for<T::E_K>([&](auto kk) { static_for<T::E_M>([&](auto e) {
            v[kk.value][e.value] = *reinterpret_cast<OPUS_LDS_ADDR const fp4x32_t*>(stage + u_ra.offsets[e.value * T::E_K + kk.value]);
        }); });
    };

    CFrag v_c[T::E_M][T::E_N];
    static_for<T::E_M>([&](auto e) { static_for<T::E_N>([&](auto n) { clear(v_c[e.value][n.value]); }); });
    const int steps = k_end > k_begin ? (k_end - k_begin + T::KS - 1) / T::KS : 0;   // ring steps (KS K tiles each)

    // Ring step t lives in slot t % STAGES.  Prologue: issue steps 0 .. STAGES-1, wait for step 0, read A(0).
    // The waits are asm with a "memory" clobber (vmcnt and lgkmcnt in one instruction, and a compiler barrier for the
    // LDS reads that follow), counted in VMEM_PER_STEP units: opus::s_waitcnt_vmcnt is not a compiler memory barrier.
    static_for<D>([&](auto t) { issue(t, k_begin + T::KS * decltype(t)::value); __builtin_amdgcn_sched_barrier(0); });
    fp4x32_t v_a[T::E_K][T::E_M];   // A operands of the current step
    // The workspace-slot s_load (S > 1) must not be pending in the loop: the waitcnt pass cannot see the asm waits and,
    // with an SMEM load outstanding, would turn its ds_read waits into lgkmcnt(0).  Drained here (builtin: visible to it).
    s_waitcnt_lgkmcnt(0_I);
    asm volatile("s_waitcnt vmcnt(%0)" :: "n"((D - 1) * G) : "memory");
    sync_threads();
    read_a(v_a, 0_I);
    __builtin_amdgcn_sched_barrier(0);
#pragma clang loop unroll(disable)
    for (int i0 = 0; i0 < steps; i0 += D) {
        static_for<D>([&](auto u) {
            constexpr int s = decltype(u)::value, sn = (s + 1) % D;
            const int kt = k_begin + T::KS * (i0 + s);
            // Step t+1 landed (D-2 younger steps in flight) and this wave's A(t) reads are done; the barrier makes A(t+1)
            // visible and frees A(t)'s stage (every wave has read it) for the copies of step t+D.
            asm volatile("s_waitcnt vmcnt(%0) lgkmcnt(0)" :: "n"((D - 2) * G) : "memory");
            sync_threads();
            __builtin_amdgcn_sched_barrier(0);
            fp4x32_t v_an[T::E_K][T::E_M];
            read_a(v_an, number<sn>{});
            static_for<T::E_K>([&](auto kk) { static_for<T::E_M>([&](auto e) { static_for<T::E_N>([&](auto n) {
                constexpr int K = decltype(kk)::value, E = decltype(e)::value, N = decltype(n)::value;
                const unsigned sfa = T::SFA_HALF ? (v_sfa[s][rsfa_dword<T>(E)] >> sh_a) : v_sfa[s][rsfa_dword<T>(E)];
                const unsigned sfb = T::SFB_HALF ? (v_sfb[s][rsfb_dword<T>(N)] >> sh_b) : v_sfb[s][rsfb_dword<T>(N)];
                mma_scale_one<rsfb_opsel<T>(N, K), rsfa_opsel<T>(E, K)>(v_c[E][N], v_b[s][N][K], v_a[K][E], sfb, sfa);
            }); }); });
            issue(number<s>{}, kt + T::KS * D);
            // Schedule (kept explicit): the next step's ds_reads first, then MFMA / VMEM alternating.
            constexpr int MFMAS = T::E_K * T::E_M * T::E_N;
            sched_group<SCHED_DS_READ, T::E_K * T::E_M>();
            static_for<MFMAS>([&](auto m) {
                sched_group<SCHED_MFMA>();
                sched_group<SCHED_VMEM_READ, (decltype(m)::value < G)>();
            });
            sched_group<SCHED_VMEM_READ, G - MFMAS>();
            __builtin_amdgcn_sched_barrier(0);
            static_for<T::E_K>([&](auto kk) { static_for<T::E_M>([&](auto e) { v_a[kk.value][e.value] = v_an[kk.value][e.value]; }); });
        });
    }
    s_waitcnt_vmcnt(0_I);

    if constexpr (T::KS > 1) {   // wave groups 1..KS-1 park their partial tiles in LDS, group 0 adds them
        constexpr int FRAGS = T::E_M * T::E_N;
        __shared__ __attribute__((aligned(16))) char red[T::RED_BYTES];
        auto s_red = make_smem(reinterpret_cast<D_ACC*>(red));
        auto red_off = [&](int g, int f) { return (((g * T::T_N + wave_id_n) * FRAGS + f) * T::WARP_SIZE + lane_id) * T::VEC_C; };
        if (wave_id_k > 0)
            static_for<FRAGS>([&](auto f) { store<T::VEC_C>(s_red, v_c[f.value / T::E_N][f.value % T::E_N], red_off(wave_id_k - 1, f.value)); });
        __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup");
        sync_threads();
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup");
        if (wave_id_k > 0) return;
        static_for<T::KS - 1>([&](auto g) { static_for<FRAGS>([&](auto f) {
            v_c[f.value / T::E_N][f.value % T::E_N] += load<T::VEC_C>(s_red, red_off(g.value, f.value));
        }); });
    }

    if (split_k > 1) {   // fp32 partial tile -> workspace [S][batch][padded_M][padded_N]; splitk_reduce_kernel writes C
        const int padded_m = (kargs.m + T::B_M - 1) / T::B_M * T::B_M, padded_n = (kargs.n + T::B_N - 1) / T::B_N * T::B_N;
        auto g_w = make_gmem(ws + (static_cast<long long>(k_split * kargs.batch + batch_id) * padded_m + row0) * padded_n + col0,
                             static_cast<unsigned>((T::B_M * padded_n) * sizeof(float)));
        const auto u_gw = make_layout_gc<T>(lane_id, padded_n);
        static_for<T::E_M * T::E_N>([&](auto f) { store<T::VEC_C>(g_w, v_c[f.value / T::E_N][f.value % T::E_N], u_gw.offsets[f.value]); });
        return;
    }
    auto g_c = make_gmem(reinterpret_cast<D_C*>(kargs.ptr_c) + batch_id * kargs.stride_c_batch + static_cast<long long>(row0) * kargs.stride_c + col0,
                         clampb(static_cast<long long>(kargs.m - row0) * kargs.stride_c * 2));
    const auto u_gc = make_layout_gc<T>(lane_id, kargs.stride_c);
    static_for<T::E_M>([&](auto e) { static_for<T::E_N>([&](auto n) {
        if (col0 + 16 * n.value + 4 * (lane_id / 16) < kargs.n)   // N tail: whole 16-column blocks (N % 16 == 0)
            store<T::VEC_C>(g_c, opus::cast<D_C>(v_c[e.value][n.value]), u_gc.offsets[e.value * T::E_N + n.value]);
    }); });
}

}   // namespace opus_mxfp4_small

// grid = ((N + B_N - 1) / B_N, (M + B_M - 1) / B_M, batch * split_k), block = 256.
template<class Traits>
__global__ __launch_bounds__(Traits::BLOCK_SIZE, 1) void gemm_a4w4_mxfp4_nt_small_kernel(opus_gemm_mxfp4_small_kargs_gfx950 kargs) {
#if defined(__gfx950__)
    opus_mxfp4_small::gemm_a4w4_mxfp4_nt_small_body<Traits>(kargs);
#endif   // __gfx950__
}
