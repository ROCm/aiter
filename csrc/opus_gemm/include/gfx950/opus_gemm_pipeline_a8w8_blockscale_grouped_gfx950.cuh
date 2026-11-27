// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// gfx950 FP8 blockscale grouped GEMMs: M-grouped NT / NN and K-grouped TN on the MXFP8 1x128 4-wave pipeline.
#pragma once

#include <cstdint>
#include <opus/hip_minimal.hpp>
#include <opus/opus.hpp>
#include "opus_gemm_traits_a8w8_blockscale_grouped_gfx950.cuh"

#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx950__)
namespace opus_a8w8_bs_grouped {
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
// A [M, K] (K-contiguous): the NT copy and LDS shuffle, plain ds_read_b128.
// The reader follows the copy's even / odd row interleaving.
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
        opus::unfold_x_stride(sa_block_dim, sa_block_shape, opus::tuple{opus::number<T::smem_linear_wave + T::smem_padding>{}, 1_I}),
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
        opus::unfold_x_stride(ra_block_dim, ra_block_shape, opus::tuple{opus::number<T::smem_linear_wave + T::smem_padding>{}, 1_I}),
        opus::unfold_p_coord(ra_block_dim, opus::tuple{wave_id_m, lane_id_m % T::T_M, lane_id_m / T::T_M, lane_id / T::W_M}));
}

// ============================================================================
// A [K, M] (A_T, M-contiguous): K8 x M128 LDS slots, ds_read_b64_tr_b8
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
template<class T, int VEC_LDS = T::VEC_LDS_A>
__device__ inline auto make_layout_ra_mn(int lane_id, int wave_id_m) {
    constexpr int P = T::smem_linear_wave + T::smem_padding;
    constexpr int k_groups = T::WARP_SIZE / T::W_M;
    constexpr int k64_repeats = T::W_K / T::WARP_SIZE;
    constexpr int tr_reads_per_k16 = 16 / VEC_LDS;
    static_assert(T::E_K == 1 && k_groups == 4);
    static_assert(k64_repeats == 2 && tr_reads_per_k16 == 2 && T::W_M == 2 * VEC_LDS);
    // K8 x M128 slots: LDS(k, m) = (k / 8) * P + (k % 8) * 128 + m. Reads go in
    // repeat -> K64 -> TR8 order, the MMA register order.
    constexpr auto ra_block_shape = opus::make_tuple(
        opus::number<T::E_M>{},
        opus::number<T::T_M>{},
        opus::number<k64_repeats>{},
        opus::number<tr_reads_per_k16>{},
        opus::number<k_groups>{},
        opus::number<T::smem_sub>{},
        opus::number<16 / VEC_LDS>{},
        opus::number<VEC_LDS>{});

    constexpr auto ra_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}),
        opus::make_tuple(opus::y_dim{}),
        opus::make_tuple(opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}));

    return opus::make_layout<VEC_LDS>(
        ra_block_shape,
        opus::unfold_x_stride(ra_block_dim, ra_block_shape, opus::tuple{opus::number<T::W_M>{}, opus::number<8 * P>{}, opus::number<P>{}, opus::number<2 * P>{}, opus::number<T::HALF_B_M>{}, 1_I}),
        opus::unfold_p_coord(ra_block_dim, opus::tuple{wave_id_m, lane_id / T::W_M, (lane_id % T::W_M) / 2, lane_id % 2}));
}

// ============================================================================
// B 128x128 scales (!SFB_COL); the per-column B scales (SFB_COL) take A's builders
// ============================================================================
// Create layout for loading scale factors for B from global memory.
// scaleB [N/128, K/128] (NT): waves 0 / 1 publish one N128 half each.
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

// scaleB [K/128, N/128] (NN): waves 0 / 1 publish one N128 half each.
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
// A / B layouts: each operand takes the _k or _mn builder by its own storage
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

// B takes A's builders by its own storage: [N, K] as an [M, K] A, [K, N] as a [K, M] A
// (HALF_B_N / E_N / T_N / W_N / VEC_B equal their M / A counterparts). The copy layouts keep
// (wave_id_m, wave_id_n): any split of the copy works if its LDS layout matches.
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

// Create layout for reading B matrix from shared memory to registers: fragment ni of wave
// wave_id_n holds columns (ni * T_N + wave_id_n) * W_N + lane % 16 of the quadrant.
template<class T>
__device__ inline auto make_layout_rb(int lane_id, int wave_id_n) {
    if constexpr (T::B_T) return make_layout_ra_mn<T, T::VEC_LDS_B>(lane_id, wave_id_n);
    else return make_layout_ra_k<T>(lane_id, wave_id_n);
}

// Create layout for loading 128x128 scale factors for B from global memory.
template<class T>
__device__ inline auto make_layout_gsfb(int lane_id, int wave_id, int stride_sfb) {
    if constexpr (T::B_T) return make_layout_gsfb_mn<T>(lane_id, wave_id, stride_sfb);
    else return make_layout_gsfb_k<T>(lane_id, wave_id, stride_sfb);
}


// ============================================================================
// Grouped tile mapping
// ============================================================================


// The tile of a block (row < 0: none): C rows [row, m_end), B group, A / C batch; K-grouped: A / B rows
// [k_row0, k_row0 + k_rows), segment seg of nseg.
struct tile_t {
    int row, m_end, group, batch;
    int k_row0 = 0, k_rows = 0, seg = 0, nseg = 1;
};

// K-grouped tile of slot = item * tiles_m + m tile, item = plan table entry {group, row0, rows, seg | nseg << 16}.
template<class T, class K>
__device__ inline tile_t k_grouped_tile(const K& kargs, int slot) {
    const int tiles_m = ceil_div(kargs.m, T::B_M);
    const int* const e = kargs.table + 4 * (slot / tiles_m);
    const int group = e[0], rows = e[2];
    if (group < 0 || (rows == 0 && !kargs.write_empty)) return {-1, 0, 0, 0};   // empty: C_g = src_g (or 0)
    return {(slot % tiles_m) * T::B_M, kargs.m, group, 0, e[1], rows, e[3] & 0xffff, e[3] >> 16};
}

// K-grouped plan (one workgroup): table = full seg_len segments [0, full_items), then the remainder of each group
// (full_items + g), so segment s launches before s + 1; zeroes the turn counters.
__device__ inline void k_grouped_plan_body(const opus_gemm_a8w8_blockscale_k_grouped_plan_kargs_gfx950& k) {
    constexpr int NT = OPUS_A8W8_BS_K_GROUPED_PLAN_THREADS;
    constexpr int ALIGN = OPUS_A8W8_BS_GROUP_ALIGN;
    __shared__ int s_rows[NT], s_full[NT];
    const int tid = opus::thread_id_x();
    auto entry = [&](int i, int group, int row0, int rows, int seg, int nseg) {
        const int in = row0 < k.rows ? k.rows - row0 : 0;   // never past row R
        int* const e = k.table + 4 * i;
        e[0] = group;
        e[1] = row0;
        e[2] = rows < in ? rows : in;
        if (e[2] > k.max_rows) __builtin_trap();   // a segment past the 32-bit offsets: fail loudly
        e[3] = seg | (nseg << 16);
    };
    int off = 0, full = 0;   // rows / full segments of the previous chunks of NT groups
    for (int base = 0; base < k.groups; base += NT) {
        const int g = base + tid;
        const int size = g < k.groups ? k.grouped_layout[g] : 0;
        const int rows = size < 0 ? 0 : size < k.rows ? size : k.rows;
        const int f = rows / k.seg_len, padded = (rows + ALIGN - 1) / ALIGN * ALIGN;
        int sum_rows = padded, sum_full = f;   // inclusive prefix sums over the chunk
        for (int d = 1; d < NT; d <<= 1) {
            s_rows[tid] = sum_rows;
            s_full[tid] = sum_full;
            opus::sync_threads();
            if (tid >= d) {
                sum_rows += s_rows[tid - d];
                sum_full += s_full[tid - d];
            }
            opus::sync_threads();
        }
        if (g < k.groups) {
            const int off_g = off + sum_rows - padded, full_g = full + sum_full - f;
            const int rem = rows - f * k.seg_len;
            const int nseg = f + (rem > 0 || rows == 0 ? 1 : 0);
            for (int s = 0; s < f && full_g + s < k.full_items; ++s)
                entry(full_g + s, g, off_g + s * k.seg_len, k.seg_len, s, nseg);
            entry(k.full_items + g, rem > 0 || rows == 0 ? g : -1, off_g + f * k.seg_len, rem, f, nseg);
        }
        if (tid == NT - 1) {
            s_rows[0] = sum_rows;
            s_full[0] = sum_full;
        }
        opus::sync_threads();
        off += s_rows[0];
        full += s_full[0];
        opus::sync_threads();
    }
    for (int i = full + tid; i < k.full_items; i += NT)
        k.table[4 * i] = -1;
    for (int i = tid; i < k.turn_count; i += NT)
        k.turn[i] = 0;
}

// M-grouped plan (one workgroup, contiguous): the rows and the first row of each group of grouped_layout.
__device__ inline void m_grouped_plan_body(const opus_gemm_a8w8_blockscale_m_grouped_plan_kargs_gfx950& k) {
    __shared__ int s_rows[OPUS_A8W8_BS_M_GROUPED_PLAN_MAX_GROUPS], s_first[OPUS_A8W8_BS_M_GROUPED_PLAN_MAX_GROUPS];
    const int tid = opus::thread_id_x(), lane_id = tid % 64;
    auto add = [](int* p, int v) { __hip_atomic_fetch_add(p, v, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WORKGROUP); };
    auto lower = [](int* p, int v) { __hip_atomic_fetch_min(p, v, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WORKGROUP); };
    for (int g = tid; g < k.groups; g += OPUS_A8W8_BS_M_GROUPED_PLAN_THREADS) {
        s_rows[g] = 0;
        s_first[g] = k.m;
    }
    opus::sync_threads();
    for (int base = tid - lane_id; base < k.m; base += OPUS_A8W8_BS_M_GROUPED_PLAN_THREADS) {
        const int r = base + lane_id;
        const int g = r < k.m ? k.layout[r] : -1;
        const bool valid = g >= 0 && g < k.groups;
        const int g0 = __builtin_amdgcn_readfirstlane(g);
        if (__builtin_amdgcn_ballot_w64(g != g0) == 0) {   // one group (or padding) in the 64 rows
            if (lane_id == 0 && valid) {
                add(&s_rows[g], min(64, k.m - base));
                lower(&s_first[g], r);
            }
        } else if (valid) {
            add(&s_rows[g], 1);
            lower(&s_first[g], r);
        }
    }
    opus::sync_threads();
    for (int g = tid; g < k.groups; g += OPUS_A8W8_BS_M_GROUPED_PLAN_THREADS) {
        k.rows[g] = s_rows[g];
        k.first[g] = s_rows[g] > 0 ? s_first[g] : 0;
    }
}

// Inclusive prefix sum over the 64 lanes of a wave.
__device__ inline int wave_inclusive_sum(int v, int lane_id) {
    for (int d = 1; d < 64; d <<= 1) {
        const int u = __builtin_amdgcn_ds_bpermute(((lane_id - d) & 63) << 2, v);
        if (lane_id >= d) v += u;
    }
    return v;
}

// Block -> (M tile, N tile): bands of BAND M tiles, M fastest, so consecutive blocks share a B tile.
constexpr int BAND = 4;

__device__ inline void band_order(int wgid, int tiles, int tiles_n, int& tile, int& n) {
    const int per_band = BAND * tiles_n;
    const int first = wgid / per_band * BAND, i = wgid % per_band;
    const int size = tiles - first < BAND ? tiles - first : BAND;
    tile = first + i % size;
    n = i / size;
}

// M-grouped tile of block wgid: B_M-row tiles of all groups numbered in group order (masked: ptr_first
// nullptr). One wave: lane l sums the tiles of groups [l c, l c + c), c = ceil(G / 64).
template<int B_M, class K>
__device__ inline tile_t m_grouped_tile(const K& kargs, int wgid, int tiles_n, int& n_tile) {
    constexpr int ALIGN = OPUS_A8W8_BS_GROUP_ALIGN;
    constexpr tile_t none = {-1, 0, 0, 0};
    const int lane_id = opus::thread_id_x() % 64;
    const int groups = kargs.groups, c = ceil_div(groups, 64);
    auto group_rows = [&](int g) { const int m = kargs.ptr_rows[g]; return m < 0 ? 0 : m < kargs.m ? m : kargs.m; };
    int lane_tiles = 0;
#pragma clang loop unroll_count(4)
    for (int i = 0; i < c; ++i) {
        const int g = lane_id * c + i;
        lane_tiles += ceil_div(g < groups ? group_rows(g) : 0, B_M);
    }
    const int tiles_incl = wave_inclusive_sum(lane_tiles, lane_id);
    const int tiles = __builtin_amdgcn_readlane(tiles_incl, 63);
    if (wgid >= tiles * tiles_n) return none;
    int t;
    band_order(wgid, tiles, tiles_n, t, n_tile);
    const int l = __builtin_ctzll(__builtin_amdgcn_ballot_w64(t < tiles_incl));
    int tile_base = __builtin_amdgcn_readlane(tiles_incl, l) - __builtin_amdgcn_readlane(lane_tiles, l);
    for (int b = 0; b < c; b += 64) {   // lane l's groups, 64 at a time (one step for G <= 4096)
        const int i = b + lane_id, g = l * c + i;
        const int rows = i < c && g < groups ? group_rows(g) : 0;
        const int n = ceil_div(rows, B_M);
        const int t_incl = wave_inclusive_sum(n, lane_id) + tile_base;
        const unsigned long long hit = __builtin_amdgcn_ballot_w64(t < t_incl);
        if (hit != 0) {
            const int q = __builtin_ctzll(hit);   // the group holding tile t
            const int group = l * c + b + q;
            const int grows = __builtin_amdgcn_readlane(rows, q);
            const int local = t - (__builtin_amdgcn_readlane(t_incl, q) - __builtin_amdgcn_readlane(n, q));
            const bool masked = kargs.ptr_first == nullptr;
            const int row0 = masked ? 0 : kargs.ptr_first[group];
            const int m_end = masked ? grows : min(row0 + (grows + ALIGN - 1) / ALIGN * ALIGN, kargs.m);
            return {row0 + local * B_M, m_end, group, masked ? group : 0};
        }
        tile_base = __builtin_amdgcn_readlane(t_incl, 63);
    }
    return none;
}

} // namespace opus_a8w8_bs_grouped

// The views of one B_M x B_N tile: each starts at the tile, its range (bytes) ends where its data ends.
struct a8w8_bs_tile_view_t {
    const void* a;
    const void* b;
    const void* sfa;
    const void* sfb;
    void* c;
    unsigned a_range, b_range, sfa_range, sfb_range, c_range;
    int stride_a, stride_b, stride_sfa, stride_sfb, stride_c;
    int loops;   // K tiles (B_K deep)
    // fp32 C: C = src + the tile's sum; segment seg of nseg waits for *turn == seg, then sets seg + 1.
    const void* src;
    unsigned src_range;
    int* turn;
    int seg, nseg;
};

// One tile of C, operand layouts by storage (A_T / B_T); VOFS: whole offset bounds-checked (K-grouped).
template<class Traits, bool VOFS = false>
__device__ __forceinline__ void opus_gemm_a8w8_bs_grouped_tile(const a8w8_bs_tile_view_t& v, char* smem) {
    using namespace opus;
    using namespace opus_a8w8_bs_grouped;

    using T = opus::remove_cvref_t<Traits>;
    using D_A = opus::fp8_t;
    using D_B = opus::fp8_t;
    using D_C = typename T::D_C;
    using D_ACC = opus::fp32_t;
    using D_SF = opus::u8_t;        // E8M0
    using D_SF_PACK = opus::u32_t;  // 4 E8M0 scales

    // NN (A plain, B TR8) keeps its C11 staging and scale-read point; B plain (NT) consumes A0
    // first in the 2x2 schedule, as TT.
    constexpr bool NN_SCHEDULE = !T::A_T && T::B_T;
    constexpr bool A_FIRST = !T::B_T;
    static_assert(T::BLOCK_SIZE == 256 && T::B_K == 128);
    static_assert(T::H_M == 2, "FP8 grouped tiles are 256x256 or 256x128");
    static_assert(T::T_M == 2 && T::T_N == 2 && T::T_K == 1 && T::E_M == 4 && T::E_N == 4);
    static_assert(T::VEC_A == 16 && T::VEC_B == 16);

    const int lane_id = thread_id_x() % T::WARP_SIZE;
    const int wave_id = __builtin_amdgcn_readfirstlane(thread_id_x() / T::WARP_SIZE);
    const int wave_id_m = wave_id % T::T_M;
    const int wave_id_n = wave_id / T::T_M;

    auto g_a = make_gmem(reinterpret_cast<const D_A*>(v.a), v.a_range);
    auto g_b = make_gmem(reinterpret_cast<const D_B*>(v.b), v.b_range);
    auto g_c = make_gmem(reinterpret_cast<D_C*>(v.c), v.c_range);
    auto g_sfa = make_gmem(reinterpret_cast<const D_SF*>(v.sfa), v.sfa_range);
    auto g_sfb = make_gmem(reinterpret_cast<const D_SF*>(v.sfb), v.sfb_range);

    constexpr int smem_a_elem = T::SMEM_HALF_A_ELEMS;  // one 128 x B_K quadrant
    constexpr int smem_b_elem = T::SMEM_HALF_B_ELEMS;
    char* const smem_ab = smem;
    char* const smem_sfa = smem + T::SMEM_A_ELEMS + T::SMEM_B_ELEMS;
    char* const smem_sfb = smem_sfa + T::SFA_PANEL_BYTES;
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
    static_assert(sizeof(typename decltype(mma)::vtype_c) == T::E_M * T::E_N * MMA::elem_c * sizeof(D_ACC));
    constexpr int a_chunks = sizeof(AFragment) / T::VEC_LDS_A;   // LDS reads per operand fragment
    constexpr int b_chunks = sizeof(BFragment) / T::VEC_LDS_B;

    AFragment v_a[T::H_M][T::E_M];
    BFragment v_b[T::H_N][T::E_N];
    opus::array<typename MMA::vtype_c, T::E_M * T::E_N> v_c[T::H_M][T::H_N];

    // Global -> LDS: A / B copies (LDS DMA) into the two K slots.
    // [K, M] / [K, N] operands: XOR swizzle for the TR8 reads. A slot holds 8 K rows of 128 M bytes,
    // so rows k and k+2 are 256 bytes apart and land on the same banks: the 16 lanes that read one
    // M chunk from 8 rows conflict 4-way. Store chunk c of row k at chunk position c ^ (k / 2). The
    // LDS DMA writes lane l to row l / 8, position l % 8, so that lane fetches source chunk
    // (l % 8) ^ (l / 16).
    const int tr_copy_lane_id = (lane_id & ~7) | ((lane_id & 7) ^ (lane_id >> 4));
    auto u_ga = make_layout_ga<T>(T::A_T ? tr_copy_lane_id : lane_id, wave_id_m, wave_id_n, v.stride_a);
    auto u_gb = make_layout_gb<T>(T::B_T ? tr_copy_lane_id : lane_id, wave_id_m, wave_id_n, v.stride_b);
    auto u_sa = make_layout_sa<T>(wave_id_m, wave_id_n);
    auto u_sb = make_layout_sb<T>(wave_id_m, wave_id_n);
    auto ga_offset = [&](int half_tile_m, int tile_k) {
        if constexpr (T::A_T) return tile_k * T::B_K * v.stride_a + half_tile_m * T::HALF_B_M;
        else return half_tile_m * T::HALF_B_M * v.stride_a + tile_k * T::B_K;
    };
    auto gb_offset = [&](int half_tile_n, int tile_k) {
        if constexpr (T::B_T) return tile_k * T::B_K * v.stride_b + half_tile_n * T::HALF_B_N;
        else return half_tile_n * T::HALF_B_N * v.stride_b + tile_k * T::B_K;
    };
    auto sa_offset = [&](int stage, int half_tile_m) { return (stage * T::H_M + half_tile_m) * smem_a_elem; };
    auto sb_offset = [&](int stage, int half_tile_n) { return (stage * T::H_N + half_tile_n) * smem_b_elem; };
    const auto ga_copy_offsets = layout_to_offsets<T::VEC_A>(u_ga);
    const auto gb_copy_offsets = layout_to_offsets<T::VEC_B>(u_gb);
    const auto sa_copy_offsets = layout_to_offsets<T::VEC_A>(u_sa);
    const auto sb_copy_offsets = layout_to_offsets<T::VEC_B>(u_sb);
    // Spread each half's four copies between independent MFMA pairs.
    auto copy_pass = [&](uintptr_t lds_stage_base_a, uintptr_t lds_stage_base_b, int tile, auto part_i, auto pass_i) {
        constexpr int part = decltype(part_i)::value;
        constexpr int pass = decltype(pass_i)::value;
        // Parts 0..H_M-1 copy the A quadrants, the rest the B quadrants.
        constexpr int half = part < T::H_M ? part : part - T::H_M;
        if constexpr (part < T::H_M) {
            void* const dst = reinterpret_cast<void*>(lds_stage_base_a + sa_copy_offsets[pass] + sa_offset(0, half));
            if constexpr (VOFS) async_load<T::VEC_A>(g_a, dst, ga_copy_offsets[pass] + ga_offset(half, tile), 0);
            else async_load<T::VEC_A>(g_a, dst, ga_copy_offsets[pass], ga_offset(half, tile));
        } else {
            void* const dst = reinterpret_cast<void*>(lds_stage_base_b + sb_copy_offsets[pass] + sb_offset(0, half));
            if constexpr (VOFS) async_load<T::VEC_B>(g_b, dst, gb_copy_offsets[pass] + gb_offset(half, tile), 0);
            else async_load<T::VEC_B>(g_b, dst, gb_copy_offsets[pass], gb_offset(half, tile));
        }
    };
    static_assert(T::HALF_B_M * T::B_K / (T::BLOCK_SIZE * T::VEC_A) == 4, "four copy passes per quadrant");

    // LDS -> registers: A / B operand fragment reads.
    auto u_ra = make_layout_ra<T>(lane_id, wave_id_m);
    auto u_rb = make_layout_rb<T>(lane_id, wave_id_n);
    // TR8 lane l reads row (l % 16) / 2 of its slot; chunk c = i * 2 + wave_id moved to c ^ (row / 2).
    // The shift is constant per lane and fragment.
    const int tr_read_xor = (lane_id % 16) / 4;
    auto ra_shift = [&](int mi) {
        const int chunk = mi * T::T_M + wave_id_m;
        return T::A_T ? ((chunk ^ tr_read_xor) - chunk) * 16 : 0;
    };
    auto rb_shift = [&](int ni) {
        const int chunk = ni * T::T_N + wave_id_n;
        return T::B_T ? ((chunk ^ tr_read_xor) - chunk) * 16 : 0;
    };
    // LDS reads of operand fragment chunks [first, first + count). TR8 reads go through the builtin
    // (opus' tr_load is inline asm), so the result can go straight to its AGPRs and be scheduled
    // among the MFMAs. Each read is addressed as base + delta: the per-lane base is computed once and
    // the read's compile-time delta (layout_imm_offsets_v) stays the ds_read offset immediate.
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

    // Scales: global -> LDS panel -> registers, SCALE_PANEL_K_CAPACITY K128 columns per panel.
    // Per-row scales (A, and B with SFB_COL) are packed in VGPRs before publication: one dword per
    // quadrant row holds the scales of its 4 fragments (op_sel = fragment); consumers read both
    // halves in one LDS pair. A 128x128 B scale is one byte per N quadrant (op_sel 0).
    const int loops = v.loops;
    constexpr int scale_panel_mask = T::SCALE_PANEL_K_CAPACITY - 1;
    static_assert((T::SCALE_PANEL_K_CAPACITY & scale_panel_mask) == 0);
    const auto u_gsfa = make_layout_gsfa<T>(lane_id, wave_id_m, wave_id_n, v.stride_sfa);
    const auto u_ssfa = make_layout_ssfa<T>(lane_id, wave_id_m, wave_id_n);
    const auto u_rsfa = make_layout_rsfa<T, T::SFA_READ_BYTES>(lane_id, wave_id_m);
    const auto sfa_gmem_offsets = layout_to_offsets<T::VEC_SCALE_A>(u_gsfa);
    const auto sfa_smem_offsets = layout_to_offsets<T::VEC_SF>(u_ssfa);
    // SFB_COL: B_N == B_M, so A's scale layouts serve B (wave_id_n reads).
    const auto u_gsfb_col = make_layout_gsfa<T>(lane_id, wave_id_m, wave_id_n, v.stride_sfb);
    const auto u_rsfb = [&] {
        if constexpr (T::SFB_COL) return make_layout_rsfa<T, T::SFB_READ_BYTES>(lane_id, wave_id_n);
        else return make_layout_rsfb<T, T::SFB_READ_BYTES>();
    }();
    const auto u_gsfb = make_layout_gsfb<T>(lane_id, wave_id, v.stride_sfb);
    const auto u_ssfb = make_layout_ssfb<T>(lane_id, wave_id);
    constexpr int sfa_lds_word_stride = T::VEC_SF * (T::B_M / T::HALF_B_M) * T::E_M;
    const int lane_in_quad = lane_id % 4;
    const unsigned pair_byte_select = (lane_in_quad % 2) ? 0x03070105u : 0x06020400u;
    const unsigned quad_byte_select = (lane_in_quad / 2) ? 0x03020706u : 0x05040100u;

    auto fill_scale_panel = [&](int panel_begin) {
        // Per-row scales [K128][rows]: each pass loads 16 contiguous row scales per lane.
        auto load_rows = [&](auto& g, int stride, auto& raw) {
            static_for<T::SFA_PASSES_PER_CACHE_PANEL>([&](auto pass_i) {
                constexpr int pass = decltype(pass_i)::value;
                const int first_k_group = panel_begin + pass * T::SFA_K_COLUMNS_PER_PASS
                    + wave_id * T::SFA_K_COLUMNS_PER_WAVE;
                raw[pass] = {};
                // The bounded resource returns zero for the final partial pass's
                // K columns past loops, without reading past the view.
                if (first_k_group < loops)
                    raw[pass] = load<T::VEC_SCALE_A>(g, sfa_gmem_offsets[pass] + panel_begin * stride);
            });
        };
        // Quad transpose: one packed dword per row (bytes = its 4 fragments), stored to the panel.
        auto store_rows = [&](auto& s, auto& raw) {
            static_for<T::SFA_PASSES_PER_CACHE_PANEL>([&](auto pass_i) {
                constexpr int pass = decltype(pass_i)::value;
                const int first_k_group = panel_begin + pass * T::SFA_K_COLUMNS_PER_PASS
                    + wave_id * T::SFA_K_COLUMNS_PER_WAVE;
                if (first_k_group < loops) {
                    const auto raw_words = __builtin_bit_cast(vector_t<D_SF_PACK, 4>, raw[pass]);
                    static_for<4>([&](auto word_i) {
                        constexpr int word = decltype(word_i)::value;
                        const D_SF_PACK adjacent = mov_dpp(raw_words[word], number<0xb1>{});
                        const D_SF_PACK paired = __builtin_amdgcn_perm(adjacent, raw_words[word], pair_byte_select);
                        const D_SF_PACK opposite = mov_dpp(paired, number<0x4e>{});
                        const D_SF_PACK packed = __builtin_amdgcn_perm(opposite, paired, quad_byte_select);
                        store<T::VEC_SF>(s,
                            __builtin_bit_cast(vector_t<D_SF, T::VEC_SF>, packed),
                            sfa_smem_offsets[pass] + word * sfa_lds_word_stride);
                    });
                }
            });
        };
        vector_t<D_SF, T::VEC_SCALE_A> sfa_panel_raw[T::SFA_PASSES_PER_CACHE_PANEL];
        load_rows(g_sfa, v.stride_sfa, sfa_panel_raw);
        if constexpr (T::SFB_COL) {
            static_assert(T::B_N == T::B_M);
            vector_t<D_SF, T::VEC_SCALE_A> sfb_panel_raw[T::SFA_PASSES_PER_CACHE_PANEL];
            static_for<T::SFA_PASSES_PER_CACHE_PANEL>([&](auto pass_i) {
                constexpr int pass = decltype(pass_i)::value;
                const int first_k_group = panel_begin + pass * T::SFA_K_COLUMNS_PER_PASS
                    + wave_id * T::SFA_K_COLUMNS_PER_WAVE;
                sfb_panel_raw[pass] = {};
                if (first_k_group < loops)
                    sfb_panel_raw[pass] = load<T::VEC_SCALE_A>(g_sfb,
                        layout_to_offsets<T::VEC_SCALE_A>(u_gsfb_col)[pass] + panel_begin * v.stride_sfb);
            });
            // Global reads must finish before DPP/byte permutation and LDS stores.
            s_waitcnt_vmcnt(number<0>{});
            store_rows(s_sfa, sfa_panel_raw);
            store_rows(s_sfb, sfb_panel_raw);
        } else {
            D_SF_PACK raw_b = 0;
            if (wave_id < T::SCALE_N_HALVES && panel_begin + lane_id < loops)
                raw_b = static_cast<D_SF_PACK>(load<1>(g_sfb,
                    u_gsfb + (T::B_T ? panel_begin * v.stride_sfb : panel_begin))[0]);
            // Global reads must finish before DPP/byte permutation and LDS stores.
            s_waitcnt_vmcnt(number<0>{});
            store_rows(s_sfa, sfa_panel_raw);
            if (wave_id < T::SCALE_N_HALVES)
                store<T::VEC_SF>(s_sfb,
                    __builtin_bit_cast(vector_t<D_SF, T::VEC_SF>, raw_b), u_ssfb);
        }
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
                u_rsfb + (tile & scale_panel_mask) * T::SFB_PANEL_PITCH));
    };

    // Prologue
    // Ki computes entirely from registers, Ki+1 is read from one LDS slot,
    // and Ki+2 is copied into the other. Prime both slots before computing.
    // fill_scale_panel's waits/barrier also complete the matrix copies.
    const uintptr_t lds_a = reinterpret_cast<uintptr_t>(s_a.ptr), lds_b = reinterpret_cast<uintptr_t>(s_b.ptr);
    static_for<T::H_M + T::H_N>([&](auto part) {
        static_for<4>([&](auto pass) { copy_pass(lds_a + sa_offset(0, 0), lds_b + sb_offset(0, 0), 0, part, pass); });
    });
    if (loops > 1) {
        static_for<T::H_M + T::H_N>([&](auto part) {
            static_for<4>([&](auto pass) {
                copy_pass(lds_a + sa_offset(1, 0), lds_b + sb_offset(1, 0), 1, part, pass);
            });
        });
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
            copy_pass(reinterpret_cast<uintptr_t>(write_a), reinterpret_cast<uintptr_t>(write_b),
                future_tile, part_i, pass_i);
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
            // op_sel: A (and per-column B) scales pack the 4 fragments of a row in one dword.
            constexpr auto sel_b = number<T::SFB_COL ? decltype(ni)::value : 0>{};
            if constexpr (first_k && pin >= 0) {
                [[clang::amdgpu_pin_agpr(pin)]] v_c[hm][hn][index] = MMA{}(a, b, sa, sb, mi, sel_b);
            } else if constexpr (first_k) {
                v_c[hm][hn][index] = MMA{}(a, b, sa, sb, mi, sel_b);
            } else if constexpr (pin >= 0) {
                [[clang::amdgpu_pin_agpr(pin)]] v_c[hm][hn][index] = MMA{}(a, b, v_c[hm][hn][index], sa, sb, mi, sel_b);
            } else {
                v_c[hm][hn][index] = MMA{}(a, b, v_c[hm][hn][index], sa, sb, mi, sel_b);
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
                // TT: C[0][0] and C[0][1] consume A0 across both opposite halves.
                // Each retired fragment is reloaded one DS chunk per MFMA pair.
                static_for<4>([&](auto mi) {
                    static_for<4>([&](auto ni) {
                        constexpr int m = decltype(mi)::value, n = decltype(ni)::value;
                        mfma(number<0>{}, number<0>{}, mi, ni);
                        // one chunk per MFMA: a plain A fragment (2 chunks) is done after 2
                        if constexpr (has_next && m > 0 && n < a_chunks)
                            load_a(number<0>{}, number<m - 1>{},
                                ni, number<1>{});
                        schedule_step<1, 0, (has_next && m > 0 && n < a_chunks ? 1 : 0),
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
                        mfma(number<1>{}, number<0>{}, mi, ni);
                        if constexpr (has_next && n == 0 && m < a_chunks)
                            load_a(number<0>{}, number<3>{},
                                mi, number<1>{});
                        if constexpr (has_next && n > 0 && end > begin)
                            load_b(number<0>{}, number<n - 1>{},
                                number<begin>{}, number<end - begin>{});
                        schedule_step<1, 0,
                            has_next ? (n == 0 ? (m < a_chunks ? (m == 0 ? 3 : 1) : 0) : end - begin) : 0>();
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

    // Epilogue: C staged in LDS (rows padded by 8; bf16 one image, fp32 2 quadrant regions), full-row copies.
    constexpr bool FULL = T::B_M * (T::B_N + 8) * sizeof(D_C) <= T::SMEM_A_ELEMS + T::SMEM_B_ELEMS;
    constexpr int c_lds_stride = FULL ? T::B_N + 8 : T::HALF_B_N + 8;
    static_assert(T::B_M * c_lds_stride * sizeof(D_C) <= T::LDS_BYTES);
    constexpr int VEC_OUT = 16 / sizeof(D_C);  // 16 bytes per lane
    constexpr int OUT_ITERS = T::HALF_B_M * T::HALF_B_N / (T::BLOCK_SIZE * VEC_OUT);
    constexpr bool RMW = T::C_F32;
    auto s_c = make_smem(reinterpret_cast<D_C*>(smem_ab));
    auto g_src = make_gmem(reinterpret_cast<const D_C*>(RMW ? v.src : v.c), RMW ? v.src_range : 0u);
    const int output_lane_id = opus::lane_id();
    const int output_thread_id = wave_id * T::WARP_SIZE + output_lane_id;
    auto sc_col = [&](int c, int half_tile_n) { return FULL ? c : c - half_tile_n * T::HALF_B_N; };
    // RMW: segment s waits for turn == s (segment s - 1 is dispatched earlier), then acquires C.
    if constexpr (RMW) {
        if (v.seg > 0) {
            if (thread_id_x() == 0) {
                while (__hip_atomic_load(v.turn, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) != v.seg)
                    __builtin_amdgcn_s_sleep(2);
            }
            __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
        }
    }
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
            store<T::VEC_C>(s_c, cast<D_C>(accumulators[f]), r * c_lds_stride + sc_col(c, half_tile_n));
        });
    };
    // Full 256-byte quadrant rows per wave store; RMW src loaded before the staging.
    using out_vec_t = vector_t<D_C, VEC_OUT>;
    auto store_quadrant = [&](const auto& accumulators, int half_tile_m, int half_tile_n) {
        out_vec_t src[RMW ? OUT_ITERS : 1];
        if constexpr (RMW) {
            static_for<OUT_ITERS>([&](auto i) {
                const int linear = output_thread_id * VEC_OUT + decltype(i)::value * T::BLOCK_SIZE * VEC_OUT;
                const int r = linear / T::HALF_B_N + half_tile_m * T::HALF_B_M;
                const int c = linear % T::HALF_B_N + half_tile_n * T::HALF_B_N;
                src[decltype(i)::value] = load<VEC_OUT>(g_src, r * v.stride_c + c);
            });
        }
        stage_quadrant(accumulators, half_tile_m, half_tile_n);
        __builtin_amdgcn_sched_barrier(0);
        if constexpr (RMW) s_waitcnt_vmcnt(number<0>{});
        s_waitcnt_lgkmcnt(number<0>{});
        sync_threads();
        __builtin_amdgcn_sched_barrier(0);
        static_for<OUT_ITERS>([&](auto i) {
            const int linear = output_thread_id * VEC_OUT + decltype(i)::value * T::BLOCK_SIZE * VEC_OUT;
            const int r = linear / T::HALF_B_N + half_tile_m * T::HALF_B_M;
            const int c = linear % T::HALF_B_N + half_tile_n * T::HALF_B_N;
            const auto value = load<VEC_OUT>(s_c, r * c_lds_stride + sc_col(c, half_tile_n));
            if constexpr (RMW)
                store<VEC_OUT>(g_c, value + src[decltype(i)::value], r * v.stride_c + c, 0, number<2>{});
            else
                store<VEC_OUT>(g_c, value, r * v.stride_c + c, 0, number<2>{});
        });
    };
    // Quadrant order C00, C10, C01, C11 (those present in the tile).
    store_quadrant(v_c[0][0], 0, 0);
    store_quadrant(v_c[1][0], 1, 0);
    if constexpr (T::H_N == 2) {
        store_quadrant(v_c[0][T::H_N - 1], 0, 1);
        store_quadrant(v_c[1][T::H_N - 1], 1, 1);
    }
    // RMW: hand the tile to the next segment once every wave's C stores are visible (release).
    if constexpr (RMW) {
        if (v.seg + 1 < v.nseg) {
            __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
            sync_threads();
            if (thread_id_x() == 0)
                __hip_atomic_store(v.turn, v.seg + 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        }
    }
}

// M-grouped kernel (fwd NT / dgrad NN): 256-row tiles from m_grouped_tile.
template<typename Traits>
__device__ __forceinline__ void
gemm_a8w8_bs_m_grouped_body(const opus_gemm_a8w8_blockscale_grouped_kargs_gfx950& kargs) {
    using namespace opus_a8w8_bs_grouped;
    using T = opus::remove_cvref_t<Traits>;
    static_assert(!T::A_T && !T::SFB_COL, "M-grouped: A [M, K], 128x128 B scales");
    const int num_tiles_n = kargs.n / T::B_N;
    const int wgid = opus::block_id_x();
    int n_tile;
    const tile_t tile = m_grouped_tile<T::B_M>(kargs, wgid, num_tiles_n, n_tile);
    if (tile.row < 0) return;
    const int row = tile.row, col = n_tile * T::B_N, group = tile.group, batch = tile.batch;
    const int rows_in = tile.m_end - row < T::B_M ? tile.m_end - row : T::B_M;
    const int k128 = kargs.k / T::B_K;

    // Views at the tile, 64-bit bases, ranges ending at the tile (A / C rows) or the group (B / scales).
    a8w8_bs_tile_view_t v;
    const long long b_ofs = T::B_T ? col : static_cast<long long>(col) * kargs.stride_b;
    const int sfb_ofs = T::B_T ? col / 128 : col / 128 * kargs.stride_sfb;
    v.a = reinterpret_cast<const char*>(kargs.ptr_a) + batch * kargs.stride_a_batch
        + static_cast<long long>(row) * kargs.stride_a;
    v.a_range = __builtin_amdgcn_readfirstlane(static_cast<unsigned>(rows_in * kargs.stride_a));
    v.b = reinterpret_cast<const char*>(kargs.ptr_b) + group * kargs.stride_b_group + b_ofs;
    v.b_range = __builtin_amdgcn_readfirstlane(static_cast<unsigned>(
        (T::B_T ? static_cast<long long>(kargs.k) : static_cast<long long>(kargs.n)) * kargs.stride_b - b_ofs));
    v.sfa = reinterpret_cast<const char*>(kargs.ptr_sfa) + static_cast<long long>(batch) * kargs.stride_sfa_batch + row;
    v.sfa_range = __builtin_amdgcn_readfirstlane(static_cast<unsigned>(k128 * kargs.stride_sfa - row));
    v.sfb = reinterpret_cast<const char*>(kargs.ptr_sfb) + static_cast<long long>(group) * kargs.stride_sfb_group
        + sfb_ofs;
    v.sfb_range = __builtin_amdgcn_readfirstlane(static_cast<unsigned>(kargs.stride_sfb_group - sfb_ofs));
    v.c = reinterpret_cast<opus::bf16_t*>(kargs.ptr_c) + batch * kargs.stride_c_batch
        + static_cast<long long>(row) * kargs.stride_c + col;
    v.c_range = __builtin_amdgcn_readfirstlane(
        static_cast<unsigned>(((rows_in - 1) * kargs.stride_c + T::B_N) * sizeof(opus::bf16_t)));
    v.stride_a = kargs.stride_a;
    v.stride_b = kargs.stride_b;
    v.stride_sfa = kargs.stride_sfa;
    v.stride_sfb = kargs.stride_sfb;
    v.stride_c = kargs.stride_c;
    v.loops = k128;
    v.src = nullptr;
    v.src_range = 0;
    v.turn = nullptr;
    v.seg = 0;
    v.nseg = 1;
    __shared__ __attribute__((aligned(16))) char smem[T::LDS_BYTES];
    opus_gemm_a8w8_bs_grouped_tile<T>(v, smem);
}

// K-grouped kernel (wgrad TN): block -> (work item, M tile, N tile), the item's tiles in band_order.
template<typename Traits>
__device__ __forceinline__ void
gemm_a8w8_bs_k_grouped_body(const opus_gemm_a8w8_blockscale_k_grouped_kargs_gfx950& kargs) {
    using namespace opus_a8w8_bs_grouped;
    using T = opus::remove_cvref_t<Traits>;
    using D_C = typename T::D_C;
    static_assert(T::A_T && T::B_T && T::SFB_COL, "K-grouped: TN, per-column B scales");
    const int wgid = opus::block_id_x();
    const int num_tiles_n = kargs.n / T::B_N;
    const int tiles_m = ceil_div(kargs.m, T::B_M);
    const int item = wgid / (tiles_m * num_tiles_n);
    int m_tile, n_tile;
    band_order(wgid % (tiles_m * num_tiles_n), tiles_m, num_tiles_n, m_tile, n_tile);
    const tile_t c_tile = k_grouped_tile<T>(kargs, item * tiles_m + m_tile);
    if (c_tile.row < 0) return;
    const int row = c_tile.row, col = n_tile * T::B_N;
    const int rows_in = kargs.m - row < T::B_M ? kargs.m - row : T::B_M;
    // the segment's K tiles; an empty group computes C_g = src_g (or 0) from one tile of zeros
    const int loops = c_tile.k_rows > 0 ? ceil_div(c_tile.k_rows, T::B_K) : 1;

    // Views from the segment's first row, ranges ending at its last row.
    a8w8_bs_tile_view_t v;
    const long long r0 = c_tile.k_row0;
    v.a = reinterpret_cast<const char*>(kargs.ptr_a) + r0 * kargs.stride_a + row;
    v.a_range = __builtin_amdgcn_readfirstlane(static_cast<unsigned>(max(c_tile.k_rows * kargs.stride_a - row, 0)));
    v.b = reinterpret_cast<const char*>(kargs.ptr_b) + r0 * kargs.stride_b + col;
    v.b_range = __builtin_amdgcn_readfirstlane(static_cast<unsigned>(max(c_tile.k_rows * kargs.stride_b - col, 0)));
    const int k128 = ceil_div(c_tile.k_rows, T::B_K);
    v.sfa = reinterpret_cast<const char*>(kargs.ptr_sfa) + r0 / T::B_K * kargs.stride_sfa + row;
    v.sfa_range = __builtin_amdgcn_readfirstlane(static_cast<unsigned>(max(k128 * kargs.stride_sfa - row, 0)));
    v.sfb = reinterpret_cast<const char*>(kargs.ptr_sfb) + r0 / T::B_K * kargs.stride_sfb + col;
    v.sfb_range = __builtin_amdgcn_readfirstlane(static_cast<unsigned>(max(k128 * kargs.stride_sfb - col, 0)));
    const long long c_ofs = c_tile.group * kargs.stride_c_group + static_cast<long long>(row) * kargs.stride_c + col;
    D_C* const c = reinterpret_cast<D_C*>(kargs.ptr_c) + c_ofs;
    v.c = c;
    v.c_range = __builtin_amdgcn_readfirstlane(
        static_cast<unsigned>(((rows_in - 1) * kargs.stride_c + T::B_N) * sizeof(D_C)));
    v.stride_a = kargs.stride_a;
    v.stride_b = kargs.stride_b;
    v.stride_sfa = kargs.stride_sfa;
    v.stride_sfb = kargs.stride_sfb;
    v.stride_c = kargs.stride_c;
    v.loops = loops;
    // fp32 C: segment 0 adds src (none: 0), later ones C itself
    const void* const src = c_tile.seg > 0 ? static_cast<const void*>(c)
        : kargs.ptr_src != nullptr ? static_cast<const void*>(reinterpret_cast<const D_C*>(kargs.ptr_src) + c_ofs)
                                   : nullptr;
    v.src = src != nullptr ? src : c;
    v.src_range = src != nullptr ? v.c_range : 0u;
    v.turn = kargs.turn + (c_tile.group * tiles_m + row / T::B_M) * num_tiles_n + col / T::B_N;
    v.seg = c_tile.seg;
    v.nseg = c_tile.nseg;
    __shared__ __attribute__((aligned(16))) char smem[T::LDS_BYTES];
    opus_gemm_a8w8_bs_grouped_tile<T, true>(v, smem);
}
#endif // gfx950 device
