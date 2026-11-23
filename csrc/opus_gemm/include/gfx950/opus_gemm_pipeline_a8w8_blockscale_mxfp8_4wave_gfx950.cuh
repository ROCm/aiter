// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <opus/hip_minimal.hpp>
#include <opus/opus.hpp>
#include "opus_gemm_traits_a8w8_blockscale_mxscale_bpreshuffle_4wave_pin_agpr_gfx950.cuh"
#include "opus_gemm_traits_a8w8_blockscale_mxfp8_4wave_gfx950.cuh"

#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx950__)
namespace opus_gemm_mxfp8_detail {
using opus::operator""_I;

// sched_group_barrier masks (LLVM AMDGPU)
constexpr int SCHED_MFMA = 0x008;
constexpr int SCHED_VALU = 0x002;
constexpr int SCHED_DS_READ = 0x100;
constexpr int SCHED_LDS_DMA = 0x800;   // buffer_load ... lds

// Source reads target already-retired fragments. Group the independent
// MFMA, LDSDMA, DS and address instructions inside one issue window.
template<int Mfmas, int Copies, int Reads, int Valus = 0>
__device__ __forceinline__ void schedule_step() {
    __builtin_amdgcn_sched_group_barrier(SCHED_MFMA, Mfmas, 0);
    if constexpr (Valus > 0)
        __builtin_amdgcn_sched_group_barrier(SCHED_VALU, Valus, 0);
    if constexpr (Copies > 0)
        __builtin_amdgcn_sched_group_barrier(SCHED_LDS_DMA, Copies, 0);
    if constexpr (Reads > 0)
        __builtin_amdgcn_sched_group_barrier(SCHED_DS_READ, Reads, 0);
    // Compiler ordering only; no hardware stall is emitted.
    __builtin_amdgcn_sched_barrier(0);
}

// ============================================================================
// A [M, K] / B [N, K] (K-contiguous): the NT copy and LDS shuffle, plain ds_read_b128.
// The reader follows the copy's even / odd row interleaving. B calls these with N as M
// (HALF_B_N == HALF_B_M, E_N == E_M, ...), so its fragments map like A's. Slots are
// SMEM_SLOT_K apart and chunks XOR-swizzled by the kernel (see k_copy_lane_id).
// ============================================================================
// Create layout for loading A matrix from global memory.
template<class T>
__device__ inline auto make_layout_ga_k(int lane_id, int wave_id_m, int wave_id_n, int stride_a) {
    constexpr int threads_k = T::B_K / T::VEC_A;
    constexpr int threads_m_per_block = T::BLOCK_SIZE / threads_k;
    constexpr int threads_m_per_wave = T::WARP_SIZE / threads_k;

    constexpr auto ga_block_shape = opus::make_tuple(
        opus::number<T::HALF_B_M / threads_m_per_block>{},
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
        opus::unfold_p_coord(ga_block_dim, opus::tuple{wave_id_n, lane_id / threads_k, wave_id_m, lane_id % threads_k}));
}

// Create layout for storing A matrix to shared memory.
template<class T>
__device__ inline auto make_layout_sa_k(int wave_id_m, int wave_id_n) {
    constexpr auto sa_block_shape = opus::make_tuple(
        opus::number<T::smem_m_rep / T::NUM_WAVES>{},
        opus::number<T::T_N>{},
        opus::number<T::T_M>{},
        opus::number<T::VEC_A>{});

    constexpr auto sa_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}));

    return opus::make_layout<T::VEC_A>(
        sa_block_shape,
        opus::unfold_x_stride(sa_block_dim, sa_block_shape, opus::tuple{opus::number<T::SMEM_SLOT_K>{}, 1_I}),
        opus::unfold_p_coord(sa_block_dim, opus::tuple{wave_id_n, wave_id_m}));
}

// Create layout for reading A matrix from shared memory to registers.
template<class T>
__device__ inline auto make_layout_ra_k(int lane_id, int wave_id_m) {
    constexpr auto ra_block_shape = opus::make_tuple(
        opus::number<T::E_M>{},
        opus::number<T::T_M>{},
        opus::number<T::T_M>{},
        opus::number<T::W_M / T::T_M>{},
        opus::number<T::E_K>{},
        opus::number<T::W_M * T::W_K / T::WARP_SIZE / T::VEC_A>{},
        opus::number<T::WARP_SIZE / T::W_M>{},
        opus::number<T::VEC_A>{});

    constexpr auto ra_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}, opus::y_dim{}, opus::p_dim{}, opus::y_dim{}));

    const int lane_id_m = lane_id % T::W_M;

    return opus::make_layout<T::VEC_A>(
        ra_block_shape,
        opus::unfold_x_stride(ra_block_dim, ra_block_shape, opus::tuple{opus::number<T::SMEM_SLOT_K>{}, 1_I}),
        opus::unfold_p_coord(ra_block_dim, opus::tuple{wave_id_m, lane_id_m % T::T_M, lane_id_m / T::T_M, lane_id / T::W_M}));
}

// ============================================================================
// A [K, M] / B [K, N] (A_T / B_T, M/N-contiguous): K8 x M128 LDS slots, ds_read_b64_tr_b8
// ============================================================================
// Create layout for loading A matrix from global memory.
template<class T>
__device__ inline auto make_layout_ga_mn(int lane_id, int wave_id_m, int wave_id_n, int stride_a) {
    constexpr int lanes_m = T::HALF_B_M / T::VEC_A;
    constexpr int lanes_k = T::WARP_SIZE / lanes_m;
    constexpr int k_repeats = T::B_K / (T::NUM_WAVES * lanes_k);

    constexpr auto ga_block_shape = opus::make_tuple(
        opus::number<k_repeats>{},
        opus::number<T::T_N>{},
        opus::number<T::T_M>{},
        opus::number<lanes_k>{},
        opus::number<lanes_m>{},
        opus::number<T::VEC_A>{});

    constexpr auto ga_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}));

    return opus::make_layout<T::VEC_A>(
        ga_block_shape,
        opus::unfold_x_stride(ga_block_dim, ga_block_shape, opus::tuple{stride_a, 1_I}),
        opus::unfold_p_coord(ga_block_dim, opus::tuple{wave_id_n, wave_id_m, lane_id / lanes_m, lane_id % lanes_m}));
}

// Create layout for storing A matrix to shared memory.
template<class T>
__device__ inline auto make_layout_sa_mn(int wave_id_m, int wave_id_n) {
    constexpr auto sa_block_shape = opus::make_tuple(
        opus::number<T::smem_m_rep / T::NUM_WAVES>{},
        opus::number<T::T_N>{},
        opus::number<T::T_M>{},
        opus::number<T::VEC_A>{});

    constexpr auto sa_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}));

    // The LDS DMA adds lane * 16 itself: the layout selects only the wave slot.
    return opus::make_layout(
        sa_block_shape,
        opus::unfold_x_stride(sa_block_dim, sa_block_shape, opus::tuple{opus::number<T::smem_linear_wave + T::smem_padding>{}, 1_I}),
        opus::unfold_p_coord(sa_block_dim, opus::tuple{wave_id_n, wave_id_m}));
}

// Create layout for reading A matrix from shared memory to registers.
template<class T>
__device__ inline auto make_layout_ra_mn(int lane_id, int wave_id_m) {
    constexpr int P = T::smem_linear_wave + T::smem_padding;
    constexpr int VEC_TR = 8;  // ds_read_b64_tr_b8 bytes per lane
    constexpr int k_groups = T::WARP_SIZE / T::W_M;
    constexpr int k64_repeats = T::W_K / T::WARP_SIZE;
    constexpr int tr_reads_per_k16 = 16 / VEC_TR;
    static_assert(T::E_K == 1 && k_groups == 4);
    static_assert(k64_repeats == 2 && tr_reads_per_k16 == 2 && T::W_M == 2 * VEC_TR);
    // K8 x M128 slots: LDS(k, m) = (k / 8) * P + (k % 8) * 128 + m. Reads go in
    // repeat -> K64 -> TR8 order, the MMA register order.
    constexpr auto ra_block_shape = opus::make_tuple(
        opus::number<T::E_M>{},
        opus::number<T::T_M>{},
        opus::number<k64_repeats>{},
        opus::number<tr_reads_per_k16>{},
        opus::number<k_groups>{},
        opus::number<T::smem_sub>{},
        opus::number<16 / VEC_TR>{},
        opus::number<VEC_TR>{});

    constexpr auto ra_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}),
        opus::make_tuple(opus::y_dim{}),
        opus::make_tuple(opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}));

    return opus::make_layout<VEC_TR>(
        ra_block_shape,
        opus::unfold_x_stride(ra_block_dim, ra_block_shape, opus::tuple{opus::number<T::W_M>{}, opus::number<8 * P>{}, opus::number<P>{}, opus::number<2 * P>{}, opus::number<T::HALF_B_M>{}, 1_I}),
        opus::unfold_p_coord(ra_block_dim, opus::tuple{wave_id_m, lane_id / T::W_M, (lane_id % T::W_M) / 2, lane_id % 2}));
}

// ============================================================================
// Scale factors for B 128x128, by B's storage
// ============================================================================
// Create layout for loading scale factors for B from global memory.
// scaleB [N/128, K/128]: waves 0 / 1 publish one N128 half each.
template<class T>
__device__ inline auto make_layout_gsfb_k(int lane_id, int wave_id, int stride_sfb) {
    constexpr auto gsfb_block_shape = opus::make_tuple(
        opus::number<T::SCALE_N_HALVES>{},
        opus::number<T::SCALE_PANEL_K_CAPACITY>{},
        opus::number<1>{});

    constexpr auto gsfb_block_dim = opus::make_tuple(
        opus::make_tuple(opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}));

    return opus::make_layout<1>(
        gsfb_block_shape,
        opus::unfold_x_stride(gsfb_block_dim, gsfb_block_shape, opus::tuple{stride_sfb, 1_I}),
        opus::unfold_p_coord(gsfb_block_dim, opus::tuple{wave_id, lane_id}));
}

// Create layout for loading scale factors for B from global memory.
// scaleB [K/128, N/128]: waves 0 / 1 publish one N128 half each.
template<class T>
__device__ inline auto make_layout_gsfb_mn(int lane_id, int wave_id, int stride_sfb) {
    constexpr auto gsfb_block_shape = opus::make_tuple(
        opus::number<T::SCALE_PANEL_K_CAPACITY>{},
        opus::number<T::SCALE_N_HALVES>{},
        opus::number<1>{});

    constexpr auto gsfb_block_dim = opus::make_tuple(
        opus::make_tuple(opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}));

    return opus::make_layout<1>(
        gsfb_block_shape,
        opus::unfold_x_stride(gsfb_block_dim, gsfb_block_shape, opus::tuple{stride_sfb, 1_I}),
        opus::unfold_p_coord(gsfb_block_dim, opus::tuple{lane_id, wave_id}));
}

// ============================================================================
// Scale factors: A 1x128 packed in an LDS panel of 64 K columns, B 128x128
// ============================================================================
// Create layout for loading scale factors for A from global memory.
template<class T>
__device__ inline auto make_layout_gsfa(int lane_id, int wave_id_m, int wave_id_n, int stride_sfa) {
    constexpr auto gsfa_block_shape = opus::make_tuple(
        opus::number<T::SFA_PASSES_PER_CACHE_PANEL>{},  // cache-fill passes
        opus::number<T::T_N>{},
        opus::number<T::T_M>{},
        opus::number<T::SFA_K_COLUMNS_PER_WAVE>{},
        opus::number<T::SFA_LANES_PER_K_COLUMN>{},      // lanes per K column
        opus::number<T::VEC_SCALE_A>{});                // contiguous M scales per lane

    constexpr auto gsfa_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}));

    // A 16-row M vector is row block hm * 8 + mi * T_M + wave_id_m. Lanes of a quad carry
    // mi = 0..3 so the quad transpose packs one dword per row: lane bits are
    // [hm][wave_id_m][mi], i.e. vectors 0,2,4,6,1,3,5,7,8,10,...
    const int lane_id_m = (lane_id & 8) | ((lane_id & 3) << 1) | ((lane_id & 4) >> 2);

    return opus::make_layout<T::VEC_SCALE_A>(
        gsfa_block_shape,
        opus::unfold_x_stride(gsfa_block_dim, gsfa_block_shape, opus::tuple{stride_sfa, 1_I}),
        opus::unfold_p_coord(gsfa_block_dim, opus::tuple{wave_id_n, wave_id_m, lane_id / T::SFA_LANES_PER_K_COLUMN, lane_id_m}));
}

// Create layout for storing scale factors for A to shared memory.
template<class T>
__device__ inline auto make_layout_ssfa(int lane_id, int wave_id_m, int wave_id_n) {
    constexpr auto ssfa_block_shape = opus::make_tuple(
        opus::number<T::SFA_PASSES_PER_CACHE_PANEL>{},  // cache-fill passes
        opus::number<T::T_N>{},
        opus::number<T::T_M>{},
        opus::number<T::SFA_K_COLUMNS_PER_WAVE>{},
        opus::number<T::B_M / T::VEC_SF>{},             // packed M dwords per K column
        opus::number<T::VEC_SF>{});                     // M-repeat scale bytes per dword

    constexpr auto ssfa_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}));

    // LDS byte of (wave_id_m, row l16, hm, mi) is wave_id_m * 16 * H_M * 4 + l16 * H_M * 4 + hm * 4 + mi.
    // Quad lane q holds row l16 = 4 * word + q after the transpose, so its first packed dword is
    // wave_id_m * 16 * H_M + q * H_M + hm and successive words are 4 * H_M dwords apart.
    const int lane_id_m = ((lane_id & 4) << 3) | ((lane_id & 3) << 1) | ((lane_id & 8) >> 3);

    return opus::make_layout<T::VEC_SF>(
        ssfa_block_shape,
        opus::unfold_x_stride(ssfa_block_dim, ssfa_block_shape, opus::tuple{opus::number<T::SFA_PANEL_PITCH>{}, 1_I}),
        opus::unfold_p_coord(ssfa_block_dim, opus::tuple{wave_id_n, wave_id_m, lane_id / T::SFA_LANES_PER_K_COLUMN, lane_id_m}));
}

// Create layout for reading scale factors for A from shared memory.
template<class T, int Vec = T::VEC_SCALE_SF>
__device__ inline auto make_layout_rsfa(int lane_id, int wave_id_m) {
    static_assert(Vec == T::VEC_SCALE_SF || Vec == T::SFA_READ_BYTES);

    constexpr auto rsfa_block_shape = opus::make_tuple(
        opus::number<T::T_M>{},
        opus::number<T::W_M>{},
        opus::number<Vec>{});

    constexpr auto rsfa_block_dim = opus::make_tuple(
        opus::make_tuple(opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}));

    // One dword per M quadrant for each row: H_M * 4 bytes per (wave_id_m, row).
    return opus::make_layout<Vec>(
        rsfa_block_shape,
        opus::unfold_x_stride(rsfa_block_dim, rsfa_block_shape, opus::tuple{opus::number<T::SFA_READ_BYTES>{}, 1_I}),
        opus::unfold_p_coord(rsfa_block_dim, opus::tuple{wave_id_m, lane_id % T::W_M}));
}

// Create layout for storing scale factors for B to shared memory.
template<class T>
__device__ inline auto make_layout_ssfb(int lane_id, int wave_id) {
    constexpr auto ssfb_block_shape = opus::make_tuple(
        opus::number<T::SCALE_N_HALVES>{},          // N128 halves
        opus::number<T::SCALE_PANEL_K_CAPACITY>{},  // K128 slots
        opus::number<T::VEC_SF>{});                 // the low byte holds the N128 scale (op_sel_B 0)

    constexpr auto ssfb_block_dim = opus::make_tuple(
        opus::make_tuple(opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}));

    return opus::make_layout<T::VEC_SF>(
        ssfb_block_shape,
        opus::unfold_x_stride(ssfb_block_dim, ssfb_block_shape, opus::tuple{opus::number<T::VEC_SF>{}, opus::number<T::SCALE_N_HALVES * T::VEC_SF>{}, 1_I}),
        opus::unfold_p_coord(ssfb_block_dim, opus::tuple{wave_id, lane_id}));
}

// Create layout for reading scale factors for B from shared memory.
template<class T, int Vec = T::VEC_SCALE_SF>
__device__ inline auto make_layout_rsfb() {
    static_assert(Vec == T::VEC_SCALE_SF || Vec == T::SFB_READ_BYTES);

    constexpr auto rsfb_block_shape = opus::make_tuple(opus::number<Vec>{});

    constexpr auto rsfb_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}));

    return opus::make_layout<Vec>(
        rsfb_block_shape,
        opus::unfold_x_stride(rsfb_block_dim, rsfb_block_shape, opus::tuple{1_I}),
        opus::unfold_p_coord(rsfb_block_dim, opus::tuple{}));
}

// ============================================================================
// A / B layouts: each operand takes the _k or _mn builder by its own storage; B passes
// the same wave ids to the copy and store (the slot filled does not depend on which
// wave copies it) and wave_id_n to the reader.
// ============================================================================
// Create layout for loading A matrix from global memory.
template<class T>
__device__ inline auto make_layout_ga(int lane_id, int wave_id_m, int wave_id_n, int stride_a) {
    if constexpr (T::A_T) return make_layout_ga_mn<T>(lane_id, wave_id_m, wave_id_n, stride_a);
    else return make_layout_ga_k<T>(lane_id, wave_id_m, wave_id_n, stride_a);
}

// Create layout for storing A matrix to shared memory.
template<class T>
__device__ inline auto make_layout_sa(int wave_id_m, int wave_id_n) {
    if constexpr (T::A_T) return make_layout_sa_mn<T>(wave_id_m, wave_id_n);
    else return make_layout_sa_k<T>(wave_id_m, wave_id_n);
}

// Create layout for reading A matrix from shared memory to registers.
template<class T>
__device__ inline auto make_layout_ra(int lane_id, int wave_id_m) {
    if constexpr (T::A_T) return make_layout_ra_mn<T>(lane_id, wave_id_m);
    else return make_layout_ra_k<T>(lane_id, wave_id_m);
}

// Create layout for loading B matrix from global memory.
template<class T>
__device__ inline auto make_layout_gb(int lane_id, int wave_id_m, int wave_id_n, int stride_b) {
    if constexpr (T::B_T) return make_layout_ga_mn<T>(lane_id, wave_id_m, wave_id_n, stride_b);
    else return make_layout_ga_k<T>(lane_id, wave_id_m, wave_id_n, stride_b);
}

// Create layout for storing B matrix to shared memory.
template<class T>
__device__ inline auto make_layout_sb(int wave_id_m, int wave_id_n) {
    if constexpr (T::B_T) return make_layout_sa_mn<T>(wave_id_m, wave_id_n);
    else return make_layout_sa_k<T>(wave_id_m, wave_id_n);
}

// Create layout for reading B matrix from shared memory to registers.
template<class T>
__device__ inline auto make_layout_rb(int lane_id, int wave_id_n) {
    if constexpr (T::B_T) return make_layout_ra_mn<T>(lane_id, wave_id_n);
    else return make_layout_ra_k<T>(lane_id, wave_id_n);
}

// Create layout for loading scale factors for B from global memory.
template<class T>
__device__ inline auto make_layout_gsfb(int lane_id, int wave_id, int stride_sfb) {
    if constexpr (T::B_T) return make_layout_gsfb_mn<T>(lane_id, wave_id, stride_sfb);
    else return make_layout_gsfb_k<T>(lane_id, wave_id, stride_sfb);
}

} // namespace opus_gemm_mxfp8_detail

// NT / NN / TT / TN. Each operand takes its own layouts by its storage (A_T / B_T); the
// schedule details that were tuned per direction are kept as such (see NN_SCHEDULE,
// A_FIRST below).
template<class Traits>
__device__ __forceinline__ void opus_gemm_mxfp8_4wave(
    opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 kargs) {
    using namespace opus;
    using namespace opus_gemm_mxfp8_detail;

    using T = opus::remove_cvref_t<Traits>;
    using D_A = opus::fp8_t;
    using D_B = opus::fp8_t;
    using D_C = opus::bf16_t;
    using D_ACC = opus::fp32_t;
    using D_SF = opus::u8_t;        // E8M0
    using D_SF_PACK = opus::u32_t;  // 4 E8M0 scales

    // NN / NT (A plain) keep their own copy issue (per-pass layouts), C11 staging and
    // scale-read point; TT / NT (B plain) consume A0 first in the 2x2 schedule.
    constexpr bool NN_SCHEDULE = !T::A_T;
    constexpr bool A_FIRST = !T::B_T;
    static_assert(T::BLOCK_SIZE == 256 && T::B_K == 128);
    // 256x256 (2x2 quadrants) keeps its hand schedule; 256x128 (one B quadrant)
    // uses the row / diagonal quadrant schedule below.
    static_assert(T::H_M == 2, "MXFP8 direction tiles are 256x256 or 256x128");
    static_assert(T::T_M == 2 && T::T_N == 2 && T::T_K == 1 && T::E_M == 4 && T::E_N == 4);
    static_assert(T::VEC_A == 16 && T::VEC_B == 16);

    const int lane_id = thread_id_x() % T::WARP_SIZE;
    const int wave_id = __builtin_amdgcn_readfirstlane(thread_id_x() / T::WARP_SIZE);
    const int wave_id_m = wave_id % T::T_M;
    const int wave_id_n = wave_id / T::T_M;

    // Workgroup raster as NT. GROUPED_M: walk GROUPED_M M tiles down each N column
    // before the next column. Otherwise swap the low tile-index bits when both grid
    // dimensions are even (2x2 raster).
    int block_m = block_id_y();
    int block_n = block_id_x();
    if constexpr (T::GROUPED_M != 0) {
        const int gm = (kargs.m + T::B_M - 1) / T::B_M;
        const int gn = kargs.n / T::B_N;
        const int linear = block_m * gn + block_n;
        const int first_m = (linear / (T::GROUPED_M * gn)) * T::GROUPED_M;
        const int group_rows = gm - first_m < T::GROUPED_M ? gm - first_m : T::GROUPED_M;
        const int local = linear % (T::GROUPED_M * gn);
        block_m = first_m + local % group_rows;
        block_n = local / group_rows;
    } else {
        const bool even_grid = T::B_M == T::B_N ? ((kargs.m | kargs.n) & (2 * T::B_M - 1)) == 0
            : ((kargs.m & (2 * T::B_M - 1)) | (kargs.n & (2 * T::B_N - 1))) == 0;
        if (even_grid) {
            const int old_m = block_m;
            block_m = (block_m & ~1) | (block_n & 1);
            block_n = (block_n & ~1) | (old_m & 1);
        }
    }
    const int row = block_m * T::B_M;
    const int col = block_n * T::B_N;
    const int batch_id = block_id_z();

    // Global views, each a 64-bit base at this tile and a buffer range ending with its matrix.
    // A_T: A [K, M], the tile starts at column row; else A [M, K], at row row.
    // B_T: B [K, N], the tile starts at column col; else B [N, K], at row col.
    // scaleB is [K128, N128] (B_T) or [N128, K128].
    // C: the range ends at the tile's last valid row. Padded-M rows fall outside it and their
    // stores are dropped by the hardware, so the epilogue needs no per-store branch.
    const int64_t a_offset = T::A_T ? int64_t(row) : int64_t(row) * kargs.stride_a;
    const int64_t b_offset = T::B_T ? int64_t(col) : int64_t(col) * kargs.stride_b;
    const int64_t sfb_offset = T::B_T ? int64_t(col / T::GROUP_N) : int64_t(col / T::GROUP_N) * kargs.stride_sfb;
    const int c_rows = kargs.m - row < T::B_M ? kargs.m - row : T::B_M;
    auto g_a = make_gmem(reinterpret_cast<const D_A*>(kargs.ptr_a) + int64_t(batch_id) * kargs.stride_a_batch + a_offset,
        static_cast<unsigned>((T::A_T ? int64_t(kargs.k) : int64_t(kargs.m)) * kargs.stride_a - a_offset));
    auto g_b = make_gmem(reinterpret_cast<const D_B*>(kargs.ptr_b) + int64_t(batch_id) * kargs.stride_b_batch + b_offset,
        static_cast<unsigned>((T::B_T ? int64_t(kargs.k) : int64_t(kargs.n)) * kargs.stride_b - b_offset));
    auto g_c = make_gmem(reinterpret_cast<D_C*>(kargs.ptr_c) +
        int64_t(batch_id) * kargs.stride_c_batch + int64_t(row) * kargs.stride_c + col,
        static_cast<unsigned>((int64_t(c_rows - 1) * kargs.stride_c + T::B_N) * sizeof(D_C)));
    auto g_sfa = make_gmem(reinterpret_cast<const D_SF*>(kargs.ptr_sfa) + int64_t(batch_id) * kargs.stride_sfa_batch + row,
        static_cast<unsigned>(kargs.stride_sfa_batch - row));
    auto g_sfb = make_gmem(reinterpret_cast<const D_SF*>(kargs.ptr_sfb) + int64_t(batch_id) * kargs.stride_sfb_batch + sfb_offset,
        static_cast<unsigned>(kargs.stride_sfb_batch - sfb_offset));

    constexpr int smem_a_elem = T::SMEM_HALF_A_ELEMS;  // one 128 x B_K quadrant
    constexpr int smem_b_elem = T::SMEM_HALF_B_ELEMS;
    __shared__ __attribute__((aligned(16))) char smem_ab[T::SMEM_A_ELEMS + T::SMEM_B_ELEMS];
    __shared__ __attribute__((aligned(16))) char smem_sfa[T::SFA_PANEL_BYTES];
    __shared__ __attribute__((aligned(16))) char smem_sfb[T::SFB_PANEL_BYTES];
    auto s_a = make_smem(reinterpret_cast<D_A*>(smem_ab));
    auto s_b = make_smem(reinterpret_cast<D_B*>(smem_ab + T::SMEM_A_ELEMS));
    auto s_sfa = make_smem(reinterpret_cast<D_SF*>(smem_sfa));
    auto s_sfb = make_smem(reinterpret_cast<D_SF*>(smem_sfb));

    auto mma = make_tiled_mma<D_A, D_B, D_ACC>(
        seq<T::E_M, T::E_N, T::E_K>{},
        seq<T::T_M, T::T_N, T::T_K>{},
        seq<T::W_M, T::W_N, T::W_K>{},
        mfma_adaptor_swap_ab{});
    using MMA = typename decltype(mma)::MMA;
    using AFragment = typename MMA::vtype_a;
    using BFragment = typename MMA::vtype_b;
    constexpr int ELEM_C = MMA::elem_c;
    static_assert(sizeof(typename decltype(mma)::vtype_c) == T::E_M * T::E_N * ELEM_C * sizeof(D_ACC));
    constexpr int a_chunks = sizeof(AFragment) / T::VEC_LDS_A;   // LDS reads per operand fragment
    constexpr int b_chunks = sizeof(BFragment) / T::VEC_LDS_B;

    AFragment v_a[T::H_M][T::E_M];
    BFragment v_b[T::H_N][T::E_N];
    opus::array<typename MMA::vtype_c, T::E_M * T::E_N> v_c[T::H_M][T::H_N];

    // Global -> LDS: A / B copies (LDS DMA) into the two K slots.
    // XOR swizzles of the LDS slots, which hold 8 rows of 128 bytes each, so rows land on
    // the same banks. A_T / B_T (K rows of M / N, TR8 reads): rows k and k+2 are 256 bytes
    // apart and the 16 lanes that read one chunk from 8 rows conflict 4-way; store chunk c
    // of row k at c ^ (k / 2). K-contiguous (M / N rows of K, ds_read_b128): store chunk c
    // of row r at c ^ (r % 4) (see SMEM_SLOT_K). The LDS DMA writes lane l to row l / 8,
    // position l % 8, so that lane fetches source chunk (l % 8) ^ f(l / 8).
    const int tr_copy_lane_id = (lane_id & ~7) | ((lane_id & 7) ^ (lane_id >> 4));
    const int k_copy_lane_id = (lane_id & ~7) | ((lane_id & 7) ^ ((lane_id >> 3) & 3));
    auto u_ga = make_layout_ga<T>(T::A_T ? tr_copy_lane_id : k_copy_lane_id, wave_id_m, wave_id_n, kargs.stride_a);
    auto u_gb = make_layout_gb<T>(T::B_T ? tr_copy_lane_id : k_copy_lane_id, wave_id_m, wave_id_n, kargs.stride_b);
    auto u_sa = make_layout_sa<T>(wave_id_m, wave_id_n);
    auto u_sb = make_layout_sb<T>(wave_id_m, wave_id_n);
    auto ga_offset = [&](int half_tile_m, int tile_k) {
        if constexpr (T::A_T) return tile_k * T::B_K * kargs.stride_a + half_tile_m * T::HALF_B_M;
        else return half_tile_m * T::HALF_B_M * kargs.stride_a + tile_k * T::B_K;
    };
    // PAD_M: the last K row's columns past M must be visible to the buffer bounds check,
    // which ignores SOFFSET, so the whole A offset goes in the VGPR offset and they read
    // zero instead of past the end of A. Otherwise the tile offset rides in SOFFSET.
    auto ga_voffset = [&](int lane_offset, int half_tile_m, int tile_k) {
        return T::PAD_M ? lane_offset + ga_offset(half_tile_m, tile_k) : lane_offset;
    };
    auto ga_soffset = [&](int half_tile_m, int tile_k) { return T::PAD_M ? 0 : ga_offset(half_tile_m, tile_k); };
    auto gb_offset = [&](int half_tile_n, int tile_k) {
        if constexpr (T::B_T) return tile_k * T::B_K * kargs.stride_b + half_tile_n * T::HALF_B_N;
        else return half_tile_n * T::HALF_B_N * kargs.stride_b + tile_k * T::B_K;
    };
    auto sa_offset = [&](int stage, int half_tile_m) { return (stage * T::H_M + half_tile_m) * smem_a_elem; };
    auto sb_offset = [&](int stage, int half_tile_n) { return (stage * T::H_N + half_tile_n) * smem_b_elem; };
    const auto ga_copy_offsets = layout_to_offsets<T::VEC_A>(u_ga);
    const auto sa_copy_offsets = layout_to_offsets<T::VEC_A>(u_sa);
    // One B copy pass with its repeat coordinate fixed, leaving one contiguous vector for
    // async_load; the uniform tile offset goes to SOFFSET, so each copy keeps a
    // loop-invariant VGPR address and B's buffer resource stays in SGPRs.
    auto copy_b_pass = [&](OPUS_LDS_ADDR char* base, auto pass_i, int half, int tile) {
        constexpr int pass = decltype(pass_i)::value;
        const auto u_gb_copy = make_layout<T::VEC_B>(u_gb.shape(), u_gb.stride(),
            merge_peepholed_tuple(u_gb.coord(), make_tuple(number<pass>{}, underscore{})));
        const auto u_sb_copy = make_layout<T::VEC_B>(u_sb.shape(), u_sb.stride(),
            merge_peepholed_tuple(u_sb.coord(), make_tuple(number<pass>{}, underscore{})));
        static_assert(layout_load_traits<decltype(u_gb_copy), T::VEC_B>::r_elem.value == 1);
        async_load<T::VEC_B>(g_b, base, u_gb_copy, u_sb_copy + sb_offset(0, half), gb_offset(half, tile));
    };
    // NN prologue: every 128-row/column quadrant with the complete layouts.
    auto prefetch_tile = [&](int tile, int stage) {
        static_for<T::H_M>([&](auto half_i) {
            constexpr int half = decltype(half_i)::value;
            if constexpr (T::PAD_M)
                async_load<T::VEC_A>(g_a, s_a.ptr, u_ga + ga_offset(half, tile), u_sa + sa_offset(stage, half));
            else
                async_load<T::VEC_A>(g_a, s_a.ptr, u_ga, u_sa + sa_offset(stage, half), ga_offset(half, tile));
        });
        static_for<T::H_N>([&](auto half_i) {
            constexpr int half = decltype(half_i)::value;
            async_load<T::VEC_B>(g_b, s_b.ptr,
                u_gb, u_sb + sb_offset(stage, half), gb_offset(half, tile));
        });
    };
    // Spread each half's four copies between independent MFMA pairs.
    auto prefetch_copy = [&](int tile, int stage, auto part_i, auto pass_i) {
        constexpr int part = decltype(part_i)::value;
        constexpr int pass = decltype(pass_i)::value;
        // Parts 0..H_M-1 copy the A quadrants, the rest the B quadrants.
        constexpr int half = part < T::H_M ? part : part - T::H_M;
        if constexpr (part < T::H_M)
            async_load<T::VEC_A>(g_a,
                reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(s_a.ptr)
                    + sa_copy_offsets[pass] + sa_offset(stage, half)),
                ga_voffset(ga_copy_offsets[pass], half, tile), ga_soffset(half, tile));
        else
            copy_b_pass(s_b.ptr + sb_offset(stage, 0), pass_i, half, tile);
    };

    // LDS -> registers: A / B operand fragment reads.
    auto u_ra = make_layout_ra<T>(lane_id, wave_id_m);
    auto u_rb = make_layout_rb<T>(lane_id, wave_id_n);
    // Lane l reads row (l % 16) / 2 of its slot. A_T / B_T (TR8): chunk c = mi * T_M +
    // wave_id_m (ni * T_N + wave_id_n) moved to c ^ (row / 2). K-contiguous: the lane reads
    // K chunks l / 16 and l / 16 + 4, moved to c ^ (row % 4); the XOR (< 4) leaves the + 4
    // alone. Either way the shift is constant per lane and fragment.
    const int tr_read_xor = (lane_id % 16) / 4;
    const int k_chunk = lane_id / 16;
    const int k_shift = ((k_chunk ^ ((lane_id % 16) / 2 % 4)) - k_chunk) * 16;
    auto ra_shift = [&](int mi) {
        const int chunk = mi * T::T_M + wave_id_m;
        return T::A_T ? ((chunk ^ tr_read_xor) - chunk) * 16 : k_shift;
    };
    auto rb_shift = [&](int ni) {
        const int chunk = ni * T::T_N + wave_id_n;
        return T::B_T ? ((chunk ^ tr_read_xor) - chunk) * 16 : k_shift;
    };
    // LDS reads of operand fragment chunks [first, first + count). A_T / B_T read with
    // ds_read_tr8 through the builtin (opus' tr_load is inline asm), so the result can
    // go straight to its AGPRs and be scheduled among the MFMAs. Each read is addressed
    // as base + delta: the per-lane base is computed once and the read's compile-time
    // delta (layout_imm_offsets_v) stays the ds_read offset immediate.
    const int ra_base = layout_to_offsets<T::VEC_LDS_A>(u_ra)[0];
    const int rb_base = layout_to_offsets<T::VEC_LDS_B>(u_rb)[0];
    constexpr auto ra_deltas = layout_imm_offsets_v<decltype(u_ra), T::VEC_LDS_A>;
    constexpr auto rb_deltas = layout_imm_offsets_v<decltype(u_rb), T::VEC_LDS_B>;
    using i32x2_t = opus::vector_t<opus::i32_t, 2>;
    auto tr8 = [&](auto& sm_, int base, int delta) {
        return __builtin_amdgcn_ds_read_tr8_b64_v2i32(reinterpret_cast<OPUS_LDS_ADDR i32x2_t*>(sm_.ptr + base + delta));
    };
    auto read_a_chunks = [&](auto& sm_, auto mi_i, auto first_i, auto count_i, int ofs) {
        constexpr int mi = decltype(mi_i)::value, first = decltype(first_i)::value, count = decltype(count_i)::value;
        vector_t<D_A, count * T::VEC_LDS_A> raw;
        static_for<count>([&](auto c) {
            const int base = ra_base + ra_shift(mi) + ofs;
            constexpr int delta = ra_deltas[mi * a_chunks + first + c.value] - ra_deltas[0];
            if constexpr (T::A_T)
                set_slice(raw, __builtin_bit_cast(vector_t<D_A, T::VEC_LDS_A>, tr8(sm_, base, delta)),
                    number<c.value * T::VEC_LDS_A>{}, number<(c.value + 1) * T::VEC_LDS_A>{});
            else
                set_slice(raw, load<T::VEC_LDS_A>(sm_, base + delta),
                    number<c.value * T::VEC_LDS_A>{}, number<(c.value + 1) * T::VEC_LDS_A>{});
        });
        return raw;
    };
    auto read_b_chunks = [&](auto& sm_, auto ni_i, auto first_i, auto count_i, int ofs) {
        constexpr int ni = decltype(ni_i)::value, first = decltype(first_i)::value, count = decltype(count_i)::value;
        vector_t<D_B, count * T::VEC_LDS_B> raw;
        static_for<count>([&](auto c) {
            const int base = rb_base + rb_shift(ni) + ofs;
            constexpr int delta = rb_deltas[ni * b_chunks + first + c.value] - rb_deltas[0];
            if constexpr (T::B_T)
                set_slice(raw, __builtin_bit_cast(vector_t<D_B, T::VEC_LDS_B>, tr8(sm_, base, delta)),
                    number<c.value * T::VEC_LDS_B>{}, number<(c.value + 1) * T::VEC_LDS_B>{});
            else
                set_slice(raw, load<T::VEC_LDS_B>(sm_, base + delta),
                    number<c.value * T::VEC_LDS_B>{}, number<(c.value + 1) * T::VEC_LDS_B>{});
        });
        return raw;
    };
    auto read_a_fragment = [&](auto& sm_, auto mi_i, int ofs) {
        return __builtin_bit_cast(AFragment, read_a_chunks(sm_, mi_i, number<0>{}, number<a_chunks>{}, ofs));
    };
    auto read_b_fragment = [&](auto& sm_, auto ni_i, int ofs) {
        return __builtin_bit_cast(BFragment, read_b_chunks(sm_, ni_i, number<0>{}, number<b_chunks>{}, ofs));
    };

    // Scales: global -> LDS panel -> registers. Cache the same 64 K128 columns as NT.
    // SFA is packed in VGPRs before publication; consumers read both M/N halves in one
    // LDS pair.
    const int loops = kargs.k / T::B_K;
    constexpr int scale_panel_mask = T::SCALE_PANEL_K_CAPACITY - 1;
    static_assert((T::SCALE_PANEL_K_CAPACITY & scale_panel_mask) == 0);
    const auto u_gsfa = make_layout_gsfa<T>(lane_id, wave_id_m, wave_id_n, kargs.stride_sfa);
    const auto u_ssfa = make_layout_ssfa<T>(lane_id, wave_id_m, wave_id_n);
    const auto u_rsfa = make_layout_rsfa<T, T::SFA_READ_BYTES>(lane_id, wave_id_m);
    const auto u_gsfb = make_layout_gsfb<T>(lane_id, wave_id, kargs.stride_sfb);
    const auto u_ssfb = make_layout_ssfb<T>(lane_id, wave_id);
    const auto u_rsfb = make_layout_rsfb<T, T::SFB_READ_BYTES>();
    const auto sfa_gmem_offsets = layout_to_offsets<T::VEC_SCALE_A>(u_gsfa);
    const auto sfa_smem_offsets = layout_to_offsets<T::VEC_SF>(u_ssfa);
    constexpr int sfa_lds_word_stride = T::VEC_SF * (T::B_M / T::HALF_B_M) * T::E_M;
    const int lane_in_quad = lane_id % 4;
    const unsigned pair_byte_select = (lane_in_quad % 2) ? 0x03070105u : 0x06020400u;
    const unsigned quad_byte_select = (lane_in_quad / 2) ? 0x03020706u : 0x05040100u;

    auto fill_scale_panel = [&](int panel_begin) {
        vector_t<D_SF, T::VEC_SCALE_A> sfa_panel_raw[T::SFA_PASSES_PER_CACHE_PANEL];
        static_for<T::SFA_PASSES_PER_CACHE_PANEL>([&](auto pass_i) {
            constexpr int pass = decltype(pass_i)::value;
            const int first_k_group = panel_begin + pass * T::SFA_K_COLUMNS_PER_PASS
                + wave_id * T::SFA_K_COLUMNS_PER_WAVE;
            sfa_panel_raw[pass] = {};
            // The bounded resource returns zero for the final partial pass's
            // K columns past loops, without reading into the next batch.
            if (first_k_group < loops)
                sfa_panel_raw[pass] = load<T::VEC_SCALE_A>(g_sfa,
                    sfa_gmem_offsets[pass] + panel_begin * kargs.stride_sfa);
        });
        D_SF_PACK raw_b = 0;
        if (wave_id < T::SCALE_N_HALVES && panel_begin + lane_id < loops)
            raw_b = static_cast<D_SF_PACK>(load<1>(g_sfb,
                u_gsfb + (T::B_T ? panel_begin * kargs.stride_sfb : panel_begin))[0]);

        // Global reads must finish before DPP/byte permutation and LDS stores.
        s_waitcnt_vmcnt(number<0>{});
        static_for<T::SFA_PASSES_PER_CACHE_PANEL>([&](auto pass_i) {
            constexpr int pass = decltype(pass_i)::value;
            const int first_k_group = panel_begin + pass * T::SFA_K_COLUMNS_PER_PASS
                + wave_id * T::SFA_K_COLUMNS_PER_WAVE;
            if (first_k_group < loops) {
                const auto raw_words = __builtin_bit_cast(vector_t<D_SF_PACK, 4>, sfa_panel_raw[pass]);
                static_for<4>([&](auto word_i) {
                    constexpr int word = decltype(word_i)::value;
                    const D_SF_PACK adjacent = mov_dpp(raw_words[word], number<0xb1>{});
                    const D_SF_PACK paired = __builtin_amdgcn_perm(adjacent, raw_words[word], pair_byte_select);
                    const D_SF_PACK opposite = mov_dpp(paired, number<0x4e>{});
                    const D_SF_PACK packed = __builtin_amdgcn_perm(opposite, paired, quad_byte_select);
                    store<T::VEC_SF>(s_sfa,
                        __builtin_bit_cast(vector_t<D_SF, T::VEC_SF>, packed),
                        sfa_smem_offsets[pass] + word * sfa_lds_word_stride);
                });
            }
        });
        if (wave_id < T::SCALE_N_HALVES)
            store<T::VEC_SF>(s_sfb,
                __builtin_bit_cast(vector_t<D_SF, T::VEC_SF>, raw_b), u_ssfb);
        s_waitcnt_lgkmcnt(number<0>{});
        sync_threads();
    };

    vector_t<D_SF_PACK, T::H_M> v_sfa;
    vector_t<D_SF_PACK, T::H_N> v_sfb;
    auto load_scales = [&](int tile) {
        v_sfa = __builtin_bit_cast(vector_t<D_SF_PACK, T::H_M>,
            load<T::SFA_READ_BYTES>(s_sfa,
                u_rsfa + (tile & scale_panel_mask) * T::SFA_PANEL_PITCH));
        v_sfb = __builtin_bit_cast(vector_t<D_SF_PACK, T::H_N>,
            load<T::SFB_READ_BYTES>(s_sfb,
                u_rsfb + (tile & scale_panel_mask) * T::SCALE_N_HALVES * T::VEC_SF));
    };

    // Prologue
    // Ki computes entirely from registers, Ki+1 is read from one LDS slot,
    // and Ki+2 is copied into the other. Prime both slots before computing.
    // fill_scale_panel's waits/barrier also complete the matrix copies.
    if constexpr (NN_SCHEDULE) {
        prefetch_tile(0, 0);
        if (loops > 1) prefetch_tile(1, 1);
    } else {
        static_for<T::H_M + T::H_N>([&](auto part) {
            static_for<4>([&](auto pass) { prefetch_copy(0, 0, part, pass); });
        });
        if (loops > 1) {
            static_for<T::H_M + T::H_N>([&](auto part) {
                static_for<4>([&](auto pass) { prefetch_copy(1, 1, part, pass); });
            });
        }
    }
    fill_scale_panel(0);

    static_for<2>([&](auto half_i) {
        constexpr int half = decltype(half_i)::value;
        if constexpr (half < T::H_M)
        static_for<T::E_M>([&](auto mi_i) {
            constexpr int mi = decltype(mi_i)::value;
            v_a[half][mi] = read_a_fragment(s_a, mi_i, sa_offset(0, half));
        });
        if constexpr (half < T::H_N)
        static_for<T::E_N>([&](auto ni_i) {
            constexpr int ni = decltype(ni_i)::value;
            v_b[half][ni] = read_b_fragment(s_b, ni_i, sb_offset(0, half));
        });
    });
    load_scales(0);
    // Every wave must finish reading K0 before its LDS slot can hold K2.
    s_waitcnt_lgkmcnt(number<0>{});
    sync_threads();
    int stage = 0;

    // Main loop
    // One K128 iteration: MFMA K_i, read K_i+1, and prefetch K_i+2.
    auto main_loop_iteration = [&](int tile, auto first_tile, auto next_tile, auto later_tile,
        OPUS_LDS_ADDR char* __restrict__ read_a,
        OPUS_LDS_ADDR char* __restrict__ read_b,
        OPUS_LDS_ADDR char* __restrict__ write_a,
        OPUS_LDS_ADDR char* __restrict__ write_b) {
        constexpr bool first_k = decltype(first_tile)::value;
        auto next_s_a = s_a;
        auto next_s_b = s_b;
        next_s_a.ptr = read_a;
        next_s_b.ptr = read_b;
        const int future_tile = tile + 2;
        auto prefetch = [&](auto part_i, auto pass_i) {
            if constexpr (first_k) {
                if (loops < 3) return;
            }
            constexpr int part = decltype(part_i)::value;
            constexpr int pass = decltype(pass_i)::value;
            constexpr int half = part < T::H_M ? part : part - T::H_M;
            if constexpr (NN_SCHEDULE) {
                // NN: fix the repeat coordinates, leaving one contiguous vector for
                // async_load; the uniform tile offset goes to SOFFSET (as NT), so each
                // copy keeps a loop-invariant VGPR address.
                if constexpr (part < T::H_M) {
                    const auto u_ga_copy = make_layout<T::VEC_A>(u_ga.shape(), u_ga.stride(),
                        merge_peepholed_tuple(u_ga.coord(), make_tuple(number<pass>{}, underscore{})));
                    const auto u_sa_copy = make_layout<T::VEC_A>(u_sa.shape(), u_sa.stride(),
                        merge_peepholed_tuple(u_sa.coord(), make_tuple(number<pass>{}, underscore{})));
                    static_assert(layout_load_traits<decltype(u_ga_copy), T::VEC_A>::r_elem.value == 1);
                    if constexpr (T::PAD_M)
                        async_load<T::VEC_A>(g_a, s_a.ptr,
                            u_ga_copy + ga_offset(half, future_tile), u_sa_copy + sa_offset(stage, half));
                    else
                        async_load<T::VEC_A>(g_a, s_a.ptr,
                            u_ga_copy, u_sa_copy + sa_offset(stage, half), ga_offset(half, future_tile));
                } else {
                    const auto u_gb_copy = make_layout<T::VEC_B>(u_gb.shape(), u_gb.stride(),
                        merge_peepholed_tuple(u_gb.coord(), make_tuple(number<pass>{}, underscore{})));
                    const auto u_sb_copy = make_layout<T::VEC_B>(u_sb.shape(), u_sb.stride(),
                        merge_peepholed_tuple(u_sb.coord(), make_tuple(number<pass>{}, underscore{})));
                    static_assert(layout_load_traits<decltype(u_gb_copy), T::VEC_B>::r_elem.value == 1);
                    async_load<T::VEC_B>(g_b, s_b.ptr,
                        u_gb_copy, u_sb_copy + sb_offset(stage, half), gb_offset(half, future_tile));
                }
            } else if constexpr (part < T::H_M)
                async_load<T::VEC_A>(g_a,
                    reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(write_a)
                        + sa_copy_offsets[pass] + sa_offset(0, half)),
                    ga_voffset(ga_copy_offsets[pass], half, future_tile), ga_soffset(half, future_tile));
            else
                copy_b_pass(write_b, pass_i, half, future_tile);
        };
        constexpr bool has_next = decltype(next_tile)::value;
        constexpr bool has_later = decltype(later_tile)::value;
        // The preceding tile-end publication barrier completed all old
        // scale-panel readers before its LDS cache is refilled at K64. The
        // refill's LGKM wait also completes any carried matrix reads.
        if (tile != 0 && (tile & scale_panel_mask) == 0) {
            fill_scale_panel(tile);
            load_scales(tile);
            s_waitcnt_lgkmcnt(number<0>{});
            __builtin_amdgcn_sched_barrier(0);
        }
        D_SF_PACK sfa[T::H_M], sfb[T::H_N];
        static_for<T::H_M>([&](auto i) { sfa[decltype(i)::value] = v_sfa[decltype(i)::value]; });
        static_for<T::H_N>([&](auto i) { sfb[decltype(i)::value] = v_sfb[decltype(i)::value]; });
        __builtin_amdgcn_sched_barrier(0);

        // As in NT: pin C01 to a[148:211] and the C11 tail (indices 11-15) to a[128:147].
        auto mfma = [&](auto hm, auto hn, auto mi, auto ni) {
            constexpr int index = decltype(mi)::value * T::E_N + decltype(ni)::value;
            constexpr int c_dwords = sizeof(typename MMA::vtype_c) / sizeof(opus::u32_t);
            constexpr int c01_agpr_base = 148, c11_tail_agpr_base = 128, c11_tail_begin = 11;
            constexpr bool c01 = decltype(hm)::value == 0 && decltype(hn)::value == 1;
            constexpr bool c11_tail = decltype(hm)::value == 1 && decltype(hn)::value == 1 &&
                index >= c11_tail_begin;
            constexpr int pin = c01 ? c01_agpr_base + index * c_dwords
                : c11_tail ? c11_tail_agpr_base + (index - c11_tail_begin) * c_dwords : -1;
            // Bind A / B to AGPRs at the MFMA and keep them there for later MFMAs.
            // [[clang::amdgpu_pin_agpr]] is only a hint on the producing instruction, and
            // the staged A1[3] / B[3] reach their MFMAs through a copy, which has none.
            static_assert(sizeof(AFragment) == 32 && sizeof(BFragment) == 32);
            auto a = __builtin_bit_cast(i32x8_t, v_a[hm][mi]);
            auto b = __builtin_bit_cast(i32x8_t, v_b[hn][ni]);
            asm volatile("" : "+a"(a), "+a"(b));
            v_a[hm][mi] = __builtin_bit_cast(AFragment, a);
            v_b[hn][ni] = __builtin_bit_cast(BFragment, b);
            const int sa = static_cast<int>(sfa[hm]), sb = static_cast<int>(sfb[hn]);
            if constexpr (first_k && pin >= 0) {
                [[clang::amdgpu_pin_agpr(pin)]] v_c[hm][hn][index] = MMA{}(a, b, sa, sb, mi, number<0>{});
            } else if constexpr (first_k) {
                v_c[hm][hn][index] = MMA{}(a, b, sa, sb, mi, number<0>{});
            } else if constexpr (pin >= 0) {
                [[clang::amdgpu_pin_agpr(pin)]] v_c[hm][hn][index] = MMA{}(a, b, v_c[hm][hn][index], sa, sb, mi, number<0>{});
            } else {
                v_c[hm][hn][index] = MMA{}(a, b, v_c[hm][hn][index], sa, sb, mi, number<0>{});
            }
        };
        auto load_a = [&](auto half_i, auto repeat_i, auto begin_i, auto count_i) {
            if constexpr (first_k) {
                if (loops < 2) return;
            }
            constexpr int half = decltype(half_i)::value, repeat = decltype(repeat_i)::value;
            constexpr int begin = decltype(begin_i)::value, count = decltype(count_i)::value;
            // Ki+1 scales are read here (TT / TN) while MFMA still uses the Ki snapshot.
            if constexpr (!NN_SCHEDULE && T::H_N == 2 && half == 0 && repeat == 3 && begin == 0) {
                if (((tile + 1) & scale_panel_mask) != 0) load_scales(tile + 1);
            }
            set_slice(v_a[half][repeat], read_a_chunks(next_s_a, repeat_i, begin_i, count_i, sa_offset(0, half)),
                number<begin * T::VEC_LDS_A>{}, number<(begin + count) * T::VEC_LDS_A>{});
        };
        auto load_b = [&](auto half_i, auto repeat_i, auto begin_i, auto count_i) {
            if constexpr (first_k) {
                if (loops < 2) return;
            }
            constexpr int half = decltype(half_i)::value, repeat = decltype(repeat_i)::value;
            constexpr int begin = decltype(begin_i)::value, count = decltype(count_i)::value;
            // NN reads the Ki+1 scales here instead (2x2 schedule).
            if constexpr (NN_SCHEDULE && T::H_N == 2 && half == 0 && repeat == 3 && begin == 0) {
                if (((tile + 1) & scale_panel_mask) != 0) load_scales(tile + 1);
            }
            set_slice(v_b[half][repeat], read_b_chunks(next_s_b, repeat_i, begin_i, count_i, sb_offset(0, half)),
                number<begin * T::VEC_LDS_B>{}, number<(begin + count) * T::VEC_LDS_B>{});
        };
        if constexpr (T::H_N == 2) {
            if constexpr (A_FIRST) {
                // TT / NT: C[0][0] and C[0][1] consume A0 across both opposite halves.
                // Each retired fragment is reloaded over the four MFMA pairs of the next row.
                static_for<4>([&](auto mi) {
                    static_for<4>([&](auto ni) {
                        constexpr int m = decltype(mi)::value, n = decltype(ni)::value;
                        constexpr int begin = n * a_chunks / 4;
                        constexpr int end = (n + 1) * a_chunks / 4;
                        mfma(number<0>{}, number<0>{}, mi, ni);
                        if constexpr (has_next && m > 0 && end > begin)
                            load_a(number<0>{}, number<m - 1>{},
                                number<begin>{}, number<end - begin>{});
                        schedule_step<1, 0, (has_next && m > 0 ? end - begin : 0),
                            has_next && m == 1 && n == 0 ? 1 : 0>();
                        mfma(number<0>{}, number<1>{}, mi, ni);
                        if constexpr (has_later) prefetch(mi, ni);
                        schedule_step<1, has_later ? 1 : 0, 0>();
                    });
                });

                // C[1][0]: finish A0 reloads, then reload retired B0 fragments.
                static_for<4>([&](auto ni) {
                    static_for<4>([&](auto mi) {
                        constexpr int m = decltype(mi)::value, n = decltype(ni)::value;
                        constexpr int begin = m * b_chunks / 4;
                        constexpr int end = (m + 1) * b_chunks / 4;
                        constexpr int a_begin = m * a_chunks / 4;
                        constexpr int a_end = (m + 1) * a_chunks / 4;
                        mfma(number<1>{}, number<0>{}, mi, ni);
                        if constexpr (has_next && n == 0 && a_end > a_begin)
                            load_a(number<0>{}, number<3>{},
                                number<a_begin>{}, number<a_end - a_begin>{});
                        if constexpr (has_next && n > 0 && end > begin)
                            load_b(number<0>{}, number<n - 1>{},
                                number<begin>{}, number<end - begin>{});
                        schedule_step<1, 0, has_next ? (n == 0 ?
                            (m == 0 ? a_end - a_begin + 2 : a_end - a_begin) : end - begin) : 0>();
                    });
                });
            } else {
                // C[0][0] and C[1][0]: consume B0 across both opposite halves.
                // Each retired fragment is reloaded one DS chunk per MFMA pair.
                static_for<4>([&](auto ni) {
                    static_for<4>([&](auto mi) {
                        constexpr int m = decltype(mi)::value, n = decltype(ni)::value;
                        mfma(number<0>{}, number<0>{}, mi, ni);
                        if constexpr (has_next && n > 0)
                            load_b(number<0>{}, number<n - 1>{},
                                mi, number<1>{});
                        schedule_step<1, 0, (has_next && n > 0 ? 1 : 0),
                            has_next && n == 1 && m == 0 ? 1 : 0>();
                        mfma(number<1>{}, number<0>{}, mi, ni);
                        if constexpr (has_later) prefetch(ni, mi);
                        schedule_step<1, has_later ? 1 : 0, 0>();
                    });
                });

                // C[0][1]: finish B0 reloads, then reload retired A0 fragments.
                static_for<4>([&](auto mi) {
                    static_for<4>([&](auto ni) {
                        constexpr int m = decltype(mi)::value, n = decltype(ni)::value;
                        constexpr int begin = n * a_chunks / 4;
                        constexpr int end = (n + 1) * a_chunks / 4;
                        mfma(number<0>{}, number<1>{}, mi, ni);
                        if constexpr (has_next && m == 0)
                            load_b(number<0>{}, number<3>{},
                                ni, number<1>{});
                        if constexpr (has_next && m > 0 && end > begin)
                            load_a(number<0>{}, number<m - 1>{},
                                number<begin>{}, number<end - begin>{});
                        schedule_step<1, 0,
                            has_next ? (m == 0 ? (n == 0 ? 3 : 1) : end - begin) : 0>();
                    });
                });
            }

            // C[1][1]: retire one row/column pair at a time and load its next fragments.
            // A1[3] and B1[3] retire with the last MFMA, so (as in NT) their next values
            // are staged early in spare registers and moved in after it; the tile-end
            // LGKM wait then finds every Ki+1 read long complete.
            AFragment a1_m3_next;
            BFragment b1_n3_next;
            static_for<4>([&](auto diagonal_i) {
                constexpr int diagonal = decltype(diagonal_i)::value;
                constexpr int rows = 4 - diagonal, columns = 3 - diagonal;
                static_for<rows>([&](auto step_i) {
                    constexpr int step = decltype(step_i)::value;
                    constexpr int begin = step * b_chunks / rows;
                    constexpr int end = (step + 1) * b_chunks / rows;
                    if constexpr (has_next && diagonal == 3) {
                        // B1[2] has retired; its read can precede the last MFMA.
                        load_b(number<1>{}, number<2>{},
                            number<0>{}, number<b_chunks>{});
                        __builtin_amdgcn_sched_group_barrier(SCHED_DS_READ, b_chunks, 0);
                        __builtin_amdgcn_sched_barrier(0);
                    }
                    mfma(number<1>{}, number<1>{}, number<diagonal>{},
                        number<diagonal + step>{});
                    if constexpr (has_next && diagonal > 0 && diagonal < 3 && end > begin)
                        load_b(number<1>{}, number<diagonal - 1>{},
                            number<begin>{}, number<end - begin>{});
                    // The other operand's last half-0 fragment retired in C[0][1] / C[1][0].
                    constexpr int last0_chunks = A_FIRST ? b_chunks : a_chunks;
                    if constexpr (has_next && diagonal == 0 && step < last0_chunks) {
                        if constexpr (A_FIRST)
                            load_b(number<0>{}, number<3>{}, number<step>{}, number<1>{});
                        else
                            load_a(number<0>{}, number<3>{}, number<step>{}, number<1>{});
                    }
                    // A1[3] / B1[3] staging: NN early in diagonal 0, TT / TN in diagonals 1 / 2.
                    constexpr bool stage_a = NN_SCHEDULE ? diagonal == 0 && step == 2 : diagonal == 1 && step == 0;
                    constexpr bool stage_b = NN_SCHEDULE ? diagonal == 0 && step == 3 : diagonal == 2 && step == 0;
                    if constexpr (has_next && stage_a)
                        a1_m3_next = read_a_fragment(next_s_a, number<3>{}, sa_offset(0, 1));
                    if constexpr (has_next && stage_b)
                        b1_n3_next = read_b_fragment(next_s_b, number<3>{}, sb_offset(0, 1));
                    if constexpr (NN_SCHEDULE) {
                        schedule_step<1, 0, has_next ? (diagonal > 0 ?
                            (diagonal == 3 ? 0 : end - begin) : (step < a_chunks ? 1 :
                            step == 2 ? a_chunks : step == 3 ? b_chunks : 0)) : 0>();
                    } else {
                        constexpr int staged_reads = stage_a ? a_chunks : stage_b ? b_chunks : 0;
                        schedule_step<1, 0, has_next ? staged_reads + (diagonal > 0 ?
                            (diagonal == 3 ? 0 : end - begin) : (step < last0_chunks ? 1 : 0)) : 0>();
                    }
                });
                static_for<columns>([&](auto step_i) {
                    constexpr int step = decltype(step_i)::value;
                    constexpr int begin = step * a_chunks / columns;
                    constexpr int end = (step + 1) * a_chunks / columns;
                    mfma(number<1>{}, number<1>{}, number<diagonal + 1 + step>{},
                        number<diagonal>{});
                    if constexpr (has_next && end > begin)
                        load_a(number<1>{}, number<diagonal>{},
                            number<begin>{}, number<end - begin>{});
                    schedule_step<1, 0, has_next ? end - begin : 0>();
                });
                if constexpr (has_next && diagonal == 3) {
                    v_a[1][3] = a1_m3_next;
                    v_b[1][3] = b1_n3_next;
                }
            });
        } else {
            // One B quadrant (256x128): B0 feeds every MFMA, so it retires
            // only in the last quadrant. 256x128 runs C00 row by row (A0[m] retires
            // with its row and is reloaded during the next one), then C10 along
            // diagonals. In the diagonal quadrant
            // row d retires A1[d] (reloaded during column d) and column d retires
            // B0[d] (reloaded during diagonal d + 1); A1[3] and B0[3] retire with
            // the last MFMA, so their next values are staged early and moved in after.
            // The K_i+2 copies go one per MFMA from the first MFMA of the tile.
            constexpr int copies = (T::H_M + T::H_N) * 4;
            constexpr int g0 = T::E_M * T::E_N;
            auto copy_at = [&](auto g_i) {
                constexpr int g = decltype(g_i)::value;
                if constexpr (has_later && g < copies)
                    prefetch(number<g / 4>{}, number<g % 4>{});
            };
            static_for<4>([&](auto mi) {
                static_for<4>([&](auto ni) {
                    constexpr int m = decltype(mi)::value, n = decltype(ni)::value;
                    constexpr int begin = n * a_chunks / 4;
                    constexpr int end = (n + 1) * a_chunks / 4;
                    constexpr int g = m * 4 + n;
                    mfma(number<0>{}, number<0>{}, mi, ni);
                    if constexpr (has_next && m > 0 && end > begin)
                        load_a(number<0>{}, number<m - 1>{},
                            number<begin>{}, number<end - begin>{});
                    copy_at(number<g>{});
                    schedule_step<1, (has_later && g < copies) ? 1 : 0,
                        (has_next && m > 0) ? end - begin : 0>();
                });
            });
            // Ki+1 scales; MFMAs keep using the Ki snapshot in sfa / sfb.
            if constexpr (has_next) {
                if (!first_k || loops > 1)
                    if (((tile + 1) & scale_panel_mask) != 0) load_scales(tile + 1);
            }
            AFragment a_last_next;
            BFragment b_last_next;
            static_for<4>([&](auto diagonal_i) {
                constexpr int diagonal = decltype(diagonal_i)::value;
                constexpr int rows = 4 - diagonal, columns = 3 - diagonal;
                // MFMAs before this diagonal: sum over d < diagonal of (7 - 2d).
                constexpr int gd = g0 + 7 * diagonal - diagonal * (diagonal - 1);
                static_for<rows>([&](auto step_i) {
                    constexpr int step = decltype(step_i)::value;
                    constexpr int begin = step * b_chunks / rows;
                    constexpr int end = (step + 1) * b_chunks / rows;
                    constexpr int g = gd + step;
                    if constexpr (has_next && diagonal == 3) {
                        // B0[2] has retired; its read can precede the last MFMA.
                        load_b(number<0>{}, number<2>{}, number<0>{}, number<b_chunks>{});
                        __builtin_amdgcn_sched_group_barrier(SCHED_DS_READ, b_chunks, 0);
                        __builtin_amdgcn_sched_barrier(0);
                    }
                    mfma(number<1>{}, number<0>{}, number<diagonal>{}, number<diagonal + step>{});
                    if constexpr (has_next && diagonal > 0 && diagonal < 3 && end > begin)
                        load_b(number<0>{}, number<diagonal - 1>{},
                            number<begin>{}, number<end - begin>{});
                    if constexpr (has_next && diagonal == 0 && step < a_chunks)
                        load_a(number<0>{}, number<3>{}, number<step>{}, number<1>{});
                    if constexpr (has_next && diagonal == 0 && step == 2)
                        a_last_next = read_a_fragment(next_s_a, number<3>{}, sa_offset(0, 1));
                    if constexpr (has_next && diagonal == 0 && step == 3)
                        b_last_next = read_b_fragment(next_s_b, number<3>{}, sb_offset(0, 0));
                    copy_at(number<g>{});
                    constexpr int reads = !has_next ? 0
                        : (diagonal > 0 && diagonal < 3 ? end - begin : 0)
                        + (diagonal == 0 && step < a_chunks ? 1 : 0)
                        + (diagonal == 0 && step == 2 ? a_chunks : 0)
                        + (diagonal == 0 && step == 3 ? b_chunks : 0)
                        + (diagonal == 0 && step == 0 ? 2 : 0);
                    schedule_step<1, (has_later && g < copies) ? 1 : 0, reads>();
                });
                static_for<columns>([&](auto step_i) {
                    constexpr int step = decltype(step_i)::value;
                    constexpr int begin = step * a_chunks / columns;
                    constexpr int end = (step + 1) * a_chunks / columns;
                    constexpr int g = gd + rows + step;
                    mfma(number<1>{}, number<0>{}, number<diagonal + 1 + step>{}, number<diagonal>{});
                    if constexpr (has_next && end > begin)
                        load_a(number<1>{}, number<diagonal>{},
                            number<begin>{}, number<end - begin>{});
                    copy_at(number<g>{});
                    schedule_step<1, (has_later && g < copies) ? 1 : 0,
                        has_next ? end - begin : 0>();
                });
                if constexpr (has_next && diagonal == 3) {
                    v_a[1][3] = a_last_next;
                    v_b[0][3] = b_last_next;
                }
            });
        }

        if constexpr (has_next) {
            // Finish every wave's Ki+1 reads before its slot holds Ki+3.
            // Publish Ki+2 here while preserving the original copy lead time.
            if (!first_k || loops > 1) {
                s_waitcnt_lgkmcnt(number<0>{});
                if constexpr (has_later) {
                    if (!first_k || loops > 2) s_waitcnt_vmcnt(number<0>{});
                }
                sync_threads();
                __builtin_amdgcn_sched_barrier(0);
            }
        }
        stage ^= 1;
    };

    // Steady tiles read Ki+1 and copy Ki+2. Drain the final two without
    // issuing out-of-range copies; K=128 and K=256 use only these tails.
    // Initialize every C fragment with its first MFMA's literal C=0.
    main_loop_iteration(0, number<true>{}, number<true>{}, number<true>{},
        s_a.ptr + sa_offset(1, 0), s_b.ptr + sb_offset(1, 0),
        s_a.ptr + sa_offset(0, 0), s_b.ptr + sb_offset(0, 0));
    int tile = 1;
#pragma clang loop unroll_count(2)
    for (; tile + 2 < loops; ++tile)
        main_loop_iteration(tile, number<false>{}, number<true>{}, number<true>{},
            s_a.ptr + sa_offset(stage ^ 1, 0), s_b.ptr + sb_offset(stage ^ 1, 0),
            s_a.ptr + sa_offset(stage, 0), s_b.ptr + sb_offset(stage, 0));
    if (tile + 1 < loops) {
        main_loop_iteration(tile, number<false>{}, number<true>{}, number<false>{},
            s_a.ptr + sa_offset(stage ^ 1, 0), s_b.ptr + sb_offset(stage ^ 1, 0),
            s_a.ptr + sa_offset(stage, 0), s_b.ptr + sb_offset(stage, 0));
        ++tile;
    }
    if (tile < loops)
        main_loop_iteration(tile, number<false>{}, number<false>{}, number<false>{},
            s_a.ptr + sa_offset(stage ^ 1, 0), s_b.ptr + sb_offset(stage ^ 1, 0),
            s_a.ptr + sa_offset(stage, 0), s_b.ptr + sb_offset(stage, 0));

    // Preserve the validated accumulator-to-epilogue scheduling gap.
    __builtin_amdgcn_sched_barrier(0);
    asm volatile("s_nop 7" : : : "memory");
    __builtin_amdgcn_sched_barrier(0);

    // Epilogue
    // Stage each 128x128 BF16 quadrant in the idle matrix LDS, then copy it
    // out in full 256-byte row segments, as NT's pin_output::lds does. A wave
    // owns only 32 contiguous bytes of each C row (the other wave_id_n holds the
    // neighbouring 32 bytes), so direct fragment stores leave every 128-byte
    // line partially written and cannot overlap the next workgroup.
    // Rows are padded to B_N + 8 elements: the 16 rows of a fragment then step
    // by 16 bytes across the banks and the staging writes are conflict-free.
    constexpr int c_lds_stride = T::B_N + 8;
    static_assert(T::B_M * c_lds_stride * sizeof(D_C) <= T::SMEM_A_ELEMS + T::SMEM_B_ELEMS);
    constexpr int VEC_OUT = 8;  // 16 bytes per lane
    auto s_c = make_smem(reinterpret_cast<D_C*>(smem_ab));
    const int output_lane_id = opus::lane_id();
    const int output_thread_id = wave_id * T::WARP_SIZE + output_lane_id;
    // All waves have finished their last matrix LDS reads.
    s_waitcnt_lgkmcnt(number<0>{});
    sync_threads();
    __builtin_amdgcn_sched_barrier(0);
    // MFMA C fragment: lane l holds row l%16 and columns 4*(l/16)..+3.
    auto stage_quadrant = [&](const auto& accumulators, int half_tile_m, int half_tile_n) {
        static_for<T::E_M * T::E_N>([&](auto fi) {
            constexpr int f = decltype(fi)::value;
            const int r = half_tile_m * T::HALF_B_M + ((f / T::E_N) * T::T_M + wave_id_m) * T::W_M + output_lane_id % T::W_M;
            const int c = half_tile_n * T::HALF_B_N + ((f % T::E_N) * T::T_N + wave_id_n) * T::W_N + (output_lane_id / T::W_M) * T::VEC_C;
            store<T::VEC_C>(s_c, cast<D_C>(accumulators[f]), r * c_lds_stride + c);
        });
    };
    // Each wave store covers four complete 256-byte quadrant rows.
    auto copy_quadrant = [&](int half_tile_m, int half_tile_n) {
        __builtin_amdgcn_sched_barrier(0);
        s_waitcnt_lgkmcnt(number<0>{});
        sync_threads();
        __builtin_amdgcn_sched_barrier(0);
        static_for<T::HALF_B_M * T::HALF_B_N / (T::BLOCK_SIZE * VEC_OUT)>([&](auto i) {
            const int linear = output_thread_id * VEC_OUT + decltype(i)::value * T::BLOCK_SIZE * VEC_OUT;
            const int r = linear / T::HALF_B_N + half_tile_m * T::HALF_B_M;
            const int c = linear % T::HALF_B_N + half_tile_n * T::HALF_B_N;
            const auto value = load<VEC_OUT>(s_c, r * c_lds_stride + c);
            store<VEC_OUT>(g_c, value, r * kargs.stride_c + c, 0, number<2>{});
        });
    };
    // Quadrant order C00, C10, C01, C11 (those present in the tile).
    stage_quadrant(v_c[0][0], 0, 0);
    copy_quadrant(0, 0);
    stage_quadrant(v_c[1][0], 1, 0);
    copy_quadrant(1, 0);
    if constexpr (T::H_N == 2) {
        stage_quadrant(v_c[0][T::H_N - 1], 0, 1);
        copy_quadrant(0, 1);
        stage_quadrant(v_c[1][T::H_N - 1], 1, 1);
        copy_quadrant(1, 1);
    }
}
#endif // gfx950 device
