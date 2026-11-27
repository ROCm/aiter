// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// gfx950 bf16 grouped GEMMs: M-grouped NT / NN and K-grouped TN on the bf16 4-wave pipeline.
#pragma once

#include "opus_gemm_traits_a16w16_grouped_gfx950.cuh"

// ============================================================================
// The 4-wave pipeline's layouts, swizzles, schedule, operand pinning and MFMA
// ============================================================================
#ifdef __HIP_DEVICE_COMPILE__
#if defined(__gfx950__)
// The operands and accumulators are pinned to fixed AGPR / VGPR ranges: needs the AGPR-pinning
// toolchain (gen_co/build_co.py).
#if !__has_cpp_attribute(clang::amdgpu_pin_agpr)
#error "opus_gemm a16w16 grouped: [[clang::amdgpu_pin_agpr]] needs the AGPR-pinning toolchain"
#endif

namespace opus_a16w16_grouped {

using opus::operator""_I;

template<int E>
__device__ __host__ inline constexpr int mn_swz(int k) {
    if constexpr (E == 4) return (k & 7) ^ (((k >> 3) & 1) * 4);
    else if constexpr (E == 2) return ((k >> 1) & 1) | (((k >> 3) & 1) << 1);
    else return (k >> 3) & 1;
}

template<class T, int E, int T_W>
__device__ inline int mn_r_xor(int lane_id, int wave, int e, int h) {
    constexpr int tr_cols = T::W_M / T::VEC_TR;
    const int k = (lane_id / T::W_M) * (T::W_K / (T::WARP_SIZE / T::W_M)) + h * T::VEC_TR + (lane_id % T::W_M) / tr_cols;
    const int window = T_W * e + wave;
    return ((window ^ mn_swz<E>(k)) - window) * T::W_M;              // window = W_M elements
}

enum { OPERAND_A = 0, OPERAND_B = 1 };

// ============================================================================
// K-contiguous operand (_nt): A [M, K], B [N, K]
// ============================================================================
template<class T, int OP>
__device__ inline auto make_layout_g_nt(int lane_id, int wave_id_m, int wave_id_n, int stride) {
    constexpr int E = OP == OPERAND_A ? T::E_M : T::E_N;
    constexpr int threads_k = T::B_K / T::VEC_A;

    const int wave_x = OP == OPERAND_A ? wave_id_m : wave_id_n;
    const int wave_y = OP == OPERAND_A ? wave_id_n : wave_id_m;

    constexpr auto g_block_shape = opus::make_tuple(
        opus::number<E>{},
        opus::number<T::T_N>{},
        opus::number<T::WARP_SIZE / threads_k>{},
        opus::number<T::T_M>{},
        opus::number<threads_k>{},
        opus::number<T::VEC_A>{});

    constexpr auto g_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}));

    return opus::make_layout<T::VEC_A>(
        g_block_shape,
        opus::unfold_x_stride(g_block_dim, g_block_shape, opus::tuple{stride, 1_I}),
        opus::unfold_p_coord(g_block_dim, opus::tuple{wave_y, lane_id / threads_k, wave_x, (lane_id % threads_k) ^ (((lane_id / threads_k) ^ (wave_x << 2)) & 7)}));
}

template<class T, int OP>
__device__ inline auto make_layout_s_nt(int wave_id_m, int wave_id_n) {
    constexpr int E = OP == OPERAND_A ? T::E_M : T::E_N;

    const int wave_x = OP == OPERAND_A ? wave_id_m : wave_id_n;
    const int wave_y = OP == OPERAND_A ? wave_id_n : wave_id_m;

    constexpr auto s_block_shape = opus::make_tuple(
        opus::number<T::T_N>{},
        opus::number<T::T_M>{},
        opus::number<E>{},
        opus::number<T::VEC_A>{});

    constexpr auto s_block_dim = opus::make_tuple(
        opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::y_dim{}),
        opus::make_tuple(opus::y_dim{}));

    return opus::make_layout(
        s_block_shape,
        opus::unfold_x_stride(s_block_dim, s_block_shape, opus::tuple{opus::number<T::smem_linear_wave_a>{}, 1_I}),
        opus::unfold_p_coord(s_block_dim, opus::tuple{wave_y, wave_x}));
}

template<class T, int OP>
__device__ inline auto make_layout_r_nt(int lane_id, int wave_id_m, int wave_id_n, int kk) {
    constexpr int E = OP == OPERAND_A ? T::E_M : T::E_N;

    const int wave = OP == OPERAND_A ? wave_id_m : wave_id_n;
    const int lane_r = lane_id % T::W_M;

    constexpr auto r_block_shape = opus::make_tuple(
        opus::number<T::T_M>{},
        opus::number<T::T_M>{},
        opus::number<E>{},
        opus::number<T::W_M / T::T_M>{},
        opus::number<T::E_C>{},
        opus::number<T::WARP_SIZE / T::W_M>{},
        opus::number<T::VEC_A>{});

    constexpr auto r_block_dim = opus::make_tuple(
        opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::y_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::p_dim{}, opus::y_dim{}));

    return opus::make_layout<T::VEC_A>(
        r_block_shape,
        opus::unfold_x_stride(r_block_dim, r_block_shape, opus::tuple{opus::number<T::smem_linear_wave_a>{}, 1_I}),
        opus::unfold_p_coord(r_block_dim, opus::tuple{wave, lane_r % T::T_M, lane_r / T::T_M, 0_I, (4 * kk + lane_id / T::W_M) ^ (((lane_r / T::T_M) ^ ((lane_r % T::T_M) << 2)) & 7)}));
}

// ============================================================================
// M/N-contiguous operand (_tn): A [K, M], B [K, N]
// ============================================================================
template<class T, int OP>
__device__ inline auto make_layout_g_tn(int lane_id, int wave_id_m, int wave_id_n, int stride) {
    constexpr int E = OP == OPERAND_A ? T::E_M : T::E_N;
    constexpr int row = OP == OPERAND_A ? T::HALF_B_M : T::HALF_B_N;
    constexpr int threads_mn = row / T::VEC_A;
    constexpr int rows_per_issue = T::WARP_SIZE / threads_mn;

    const int kr = lane_id / threads_mn;
    const int swz = mn_swz<E>(rows_per_issue * (wave_id_m * T::T_N + wave_id_n) + kr) * (T::W_M / T::VEC_A);

    constexpr auto g_block_shape = opus::make_tuple(
        opus::number<E>{},
        opus::number<T::T_M>{},
        opus::number<T::T_N>{},
        opus::number<rows_per_issue>{},
        opus::number<threads_mn>{},
        opus::number<T::VEC_A>{});

    constexpr auto g_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}));

    return opus::make_layout<T::VEC_A>(
        g_block_shape,
        opus::unfold_x_stride(g_block_dim, g_block_shape, opus::tuple{stride, 1_I}),
        opus::unfold_p_coord(g_block_dim, opus::tuple{wave_id_m, wave_id_n, kr, (lane_id % threads_mn) ^ swz}));
}

template<class T, int OP>
__device__ inline auto make_layout_s_tn(int wave_id_m, int wave_id_n) {
    constexpr int E = OP == OPERAND_A ? T::E_M : T::E_N;

    constexpr auto s_block_shape = opus::make_tuple(
        opus::number<E>{},
        opus::number<T::T_M>{},
        opus::number<T::T_N>{},
        opus::number<T::VEC_A>{});

    constexpr auto s_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}));

    return opus::make_layout(
        s_block_shape,
        opus::unfold_x_stride(s_block_dim, s_block_shape, opus::tuple{opus::number<T::smem_linear_wave_a>{}, 1_I}),
        opus::unfold_p_coord(s_block_dim, opus::tuple{wave_id_m, wave_id_n}));
}

template<class T, int OP>
__device__ inline auto make_layout_r_tn(int lane_id, int wave_id_m, int wave_id_n) {
    constexpr int E = OP == OPERAND_A ? T::E_M : T::E_N;
    constexpr int T_W = OP == OPERAND_A ? T::T_M : T::T_N;
    constexpr int row = OP == OPERAND_A ? T::HALF_B_M : T::HALF_B_N;

    const int wave = OP == OPERAND_A ? wave_id_m : wave_id_n;

    constexpr auto r_block_shape = opus::make_tuple(
        opus::number<T::B_K / T::W_K>{},
        opus::number<T::WARP_SIZE / T::W_M>{},
        opus::number<T::VEC_A / T::VEC_TR>{},
        opus::number<T::VEC_TR>{},
        opus::number<E>{},
        opus::number<T_W>{},
        opus::number<T::W_M / T::VEC_TR>{},
        opus::number<T::VEC_TR>{});

    constexpr auto r_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::y_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}, opus::y_dim{}));

    return opus::make_layout<T::VEC_TR>(
        r_block_shape,
        opus::unfold_x_stride(r_block_dim, r_block_shape, opus::tuple{opus::number<row>{}, 1_I}),
        opus::unfold_p_coord(r_block_dim, opus::tuple{lane_id / T::W_M, (lane_id % T::W_M) / (T::W_M / T::VEC_TR), wave, lane_id % (T::W_M / T::VEC_TR)}));
}

// ============================================================================
// A / B layouts: each operand takes the _nt or _tn builder by its own storage.
// ============================================================================
// Create layout for loading A matrix from global memory.
template<class T>
__device__ inline auto make_layout_ga(int lane_id, int wave_id_m, int wave_id_n, int stride_a) {
    if constexpr (T::A_T) return make_layout_g_tn<T, OPERAND_A>(lane_id, wave_id_m, wave_id_n, stride_a);
    else return make_layout_g_nt<T, OPERAND_A>(lane_id, wave_id_m, wave_id_n, stride_a);
}

// Create layout for storing A matrix to shared memory.
template<class T>
__device__ inline auto make_layout_sa(int wave_id_m, int wave_id_n) {
    if constexpr (T::A_T) return make_layout_s_tn<T, OPERAND_A>(wave_id_m, wave_id_n);
    else return make_layout_s_nt<T, OPERAND_A>(wave_id_m, wave_id_n);
}

// Create layout for loading B matrix from global memory.
template<class T>
__device__ inline auto make_layout_gb(int lane_id, int wave_id_m, int wave_id_n, int stride_b) {
    if constexpr (T::B_T) return make_layout_g_tn<T, OPERAND_B>(lane_id, wave_id_m, wave_id_n, stride_b);
    else return make_layout_g_nt<T, OPERAND_B>(lane_id, wave_id_m, wave_id_n, stride_b);
}

// Create layout for storing B matrix to shared memory.
template<class T>
__device__ inline auto make_layout_sb(int wave_id_m, int wave_id_n) {
    if constexpr (T::B_T) return make_layout_s_tn<T, OPERAND_B>(wave_id_m, wave_id_n);
    else return make_layout_s_nt<T, OPERAND_B>(wave_id_m, wave_id_n);
}

// Create layout for reading A matrix from shared memory to registers (K step kk; _tn: kk is an immediate).
template<class T>
__device__ inline auto make_layout_ra(int lane_id, int wave_id_m, int wave_id_n, int kk) {
    if constexpr (T::A_T) return make_layout_r_tn<T, OPERAND_A>(lane_id, wave_id_m, wave_id_n);
    else return make_layout_r_nt<T, OPERAND_A>(lane_id, wave_id_m, wave_id_n, kk);
}

// Create layout for reading B matrix from shared memory to registers (K step kk; _tn: kk is an immediate).
template<class T>
__device__ inline auto make_layout_rb(int lane_id, int wave_id_m, int wave_id_n, int kk) {
    if constexpr (T::B_T) return make_layout_r_tn<T, OPERAND_B>(lane_id, wave_id_m, wave_id_n);
    else return make_layout_r_nt<T, OPERAND_B>(lane_id, wave_id_m, wave_id_n, kk);
}

// sched_group_barrier masks (LLVM AMDGPU)
constexpr int SCHED_MFMA = 0x008;
constexpr int SCHED_VMEM_READ = 0x820;   // 0x020 VMEM read | 0x800 LDS DMA (buffer_load ... lds)
constexpr int SCHED_DS_READ = 0x100;

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

// One MFMA operand per lane: 8 bf16 (16 bytes) for one 16x16x32 MFMA.
using operand_t = opus::vector_t<opus::bf16_t, 8>;

// The operand is pinned at a[AGPR .. AGPR + 3].
template<int AGPR>
__device__ inline void set_operand(operand_t& v, const operand_t& x) {
    [[clang::amdgpu_pin_agpr(AGPR)]] v = x;
}

// One MFMA 16x16x32. C_REG >= 0: accumulator pinned at a[C_REG]; C_REG <= -2: at v[-C_REG - 2].
template<int C_REG, class MMA, class CF>
__device__ inline void mma_bf16(const operand_t& a, const operand_t& b, CF& c) {
    using base_mma = typename MMA::MMA;
    if constexpr (C_REG >= 0) {
        [[clang::amdgpu_pin_agpr(C_REG)]] c = base_mma{}(a, b, c);
    } else {
        [[clang::amdgpu_pin_vgpr(-C_REG - 2)]] c = base_mma{}(a, b, c);
    }
}

// Pinned registers, packed: A operands a[0 .. A_REGS), B operands a[A_REGS .. A_REGS + B_REGS), then the C
// fragments (quadrant order C00, C10, C01, C11) up to a[255] and on in v[128 ..]. A_REGS = 2 halves x 2 kk x E_M x 4.
template<class T> constexpr int A_REGS = 16 * T::E_M;
template<class T> constexpr int B_REGS = 16 * T::E_N;

template<class T, int HM, int HN, int M, int N>
constexpr int c_reg() {
    constexpr int idx = A_REGS<T> + B_REGS<T> + (HN * 2 + HM) * T::E_M * T::E_N * 4 + (M * T::E_N + N) * 4;
    if constexpr (idx < 256) return idx;
    else return -(idx - 128) - 2;                                     // v[idx - 128]
}


// ============================================================================
// Grouped tile mapping
// ============================================================================

// The tile of a block (row < 0: none): C rows [row, m_end), B group, A / C batch, tile height; K-grouped:
// A / B rows [k_row0, k_row0 + k_rows), segment seg of nseg.
struct tile_t {
    int row, m_end, group, batch, height = 0;
    int k_row0 = 0, k_rows = 0, seg = 0, nseg = 1;
};

// K-grouped tile of slot = item * tiles_m + m tile, item = plan table entry {group, row0, rows, seg | nseg << 16}.
template<class T, class K>
__device__ inline tile_t k_grouped_tile(const K& kargs, int slot) {
    const int tiles_m = ceil_div(kargs.m, T::B_M);
    const int* const e = kargs.table + 4 * (slot / tiles_m);
    const int group = e[0], rows = e[2];
    if (group < 0 || (rows == 0 && !kargs.write_empty)) return {-1, 0, 0, 0};   // empty: C_g = src_g (or 0)
    return {(slot % tiles_m) * T::B_M, kargs.m, group, 0, T::B_M, e[1], rows, e[3] & 0xffff, e[3] >> 16};
}

// K-grouped plan (one workgroup): table = full seg_len segments [0, full_items), then the remainder of each group
// (full_items + g), so segment s launches before s + 1; zeroes the turn counters.
// opus::sync_threads() is a bare s_barrier: every LDS write / read waits (lgkmcnt) before it.
__device__ inline void k_grouped_plan_body(const opus_gemm_a16w16_k_grouped_plan_kargs_gfx950& k) {
    constexpr int NT = OPUS_K_GROUPED_PLAN_THREADS;
    constexpr int ALIGN = OPUS_A16W16_GROUP_ALIGN;
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
            opus::s_waitcnt_lgkmcnt(opus::number<0>{});
            opus::sync_threads();
            if (tid >= d) {
                sum_rows += s_rows[tid - d];
                sum_full += s_full[tid - d];
            }
            opus::s_waitcnt_lgkmcnt(opus::number<0>{});
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
        opus::s_waitcnt_lgkmcnt(opus::number<0>{});
        opus::sync_threads();
        off += s_rows[0];
        full += s_full[0];
        opus::s_waitcnt_lgkmcnt(opus::number<0>{});
        opus::sync_threads();
    }
    for (int i = full + tid; i < k.full_items; i += NT)
        k.table[4 * i] = -1;
    for (int i = tid; i < k.turn_count; i += NT)
        k.turn[i] = 0;
}

// M-grouped plan (one workgroup, contiguous): the rows and the first row of each group of grouped_layout.
__device__ inline void m_grouped_plan_body(const opus_gemm_a16w16_m_grouped_plan_kargs_gfx950& k) {
    __shared__ int s_rows[OPUS_M_GROUPED_PLAN_MAX_GROUPS], s_first[OPUS_M_GROUPED_PLAN_MAX_GROUPS];
    const int tid = opus::thread_id_x(), lane_id = tid % 64;
    auto add = [](int* p, int v) { __hip_atomic_fetch_add(p, v, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WORKGROUP); };
    auto lower = [](int* p, int v) { __hip_atomic_fetch_min(p, v, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WORKGROUP); };
    for (int g = tid; g < k.groups; g += OPUS_M_GROUPED_PLAN_THREADS) {
        s_rows[g] = 0;
        s_first[g] = k.m;
    }
    opus::s_waitcnt_lgkmcnt(opus::number<0>{});
    opus::sync_threads();
    for (int base = tid - lane_id; base < k.m; base += OPUS_M_GROUPED_PLAN_THREADS) {
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
    opus::s_waitcnt_lgkmcnt(opus::number<0>{});
    opus::sync_threads();
    for (int g = tid; g < k.groups; g += OPUS_M_GROUPED_PLAN_THREADS) {
        k.rows[g] = s_rows[g];
        k.first[g] = s_rows[g] > 0 ? s_first[g] : 0;
    }
}

// Run time of a 256 / 192 / 128-row tile (gfx950, % of 256); keeps a group <= ceil(rows / 256) tiles (grid bound).
constexpr int MIXED_COST[3] = {100, 91, 65};

// counts[h]: the H[h]-row tiles of the cheapest tiling of `rows` rows.
template<int H0, int H1, int H2>
__device__ inline void mixed_counts(int rows, int (&counts)[3]) {
    int best = 0x7fffffff;
    for (int a2 = 0; a2 <= 1; ++a2) {                                  // 2 H2 tiles: one H0 is cheaper
        for (int a1 = 0; a1 <= 3; ++a1) {                              // 4 H1 = 3 H0
            const int rest = rows - a1 * H1 - a2 * H2;
            const int a0 = rest > 0 ? (rest + H0 - 1) / H0 : 0;
            const int cost = a0 * MIXED_COST[0] + a1 * MIXED_COST[1] + a2 * MIXED_COST[2];
            if (cost < best) {
                best = cost;
                counts[0] = a0;
                counts[1] = a1;
                counts[2] = a2;
            }
        }
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

// Block -> (M tile, N tile): bands of MIXED_BAND M tiles, M fastest, so blocks share a B tile (+8% / +3.5%).
constexpr int MIXED_BAND = 4;

__device__ inline void mixed_band_order(int wgid, int tiles, int tiles_n, int& tile, int& n) {
    const int per_band = MIXED_BAND * tiles_n;
    const int first = wgid / per_band * MIXED_BAND, i = wgid % per_band;
    const int size = tiles - first < MIXED_BAND ? tiles - first : MIXED_BAND;
    tile = first + i % size;
    n = i / size;
}

// M-grouped tile of block wgid: the tiles of all groups numbered in group order (masked: ptr_first nullptr).
template<int H0, int H1, int H2, class K>
__device__ inline tile_t m_grouped_tile(const K& kargs, int wgid, int tiles_n, int& n_tile) {
    constexpr int ALIGN = OPUS_A16W16_GROUP_ALIGN;
    constexpr tile_t none = {-1, 0, 0, 0};
    constexpr int H[3] = {H0, H1, H2};
    const int lane_id = opus::thread_id_x() % 64;
    const int groups = kargs.groups, c = ceil_div(groups, 64);
    auto group_rows = [&](int g) { const int m = kargs.ptr_rows[g]; return m < 0 ? 0 : m < kargs.m ? m : kargs.m; };
    auto group_tiles = [&](int rows) {
        int counts[3];
        mixed_counts<H0, H1, H2>(rows, counts);
        return rows > 0 ? counts[0] + counts[1] + counts[2] : 0;
    };
    int lane_tiles = 0;
#pragma clang loop unroll_count(4)
    for (int i = 0; i < c; ++i) {
        const int g = lane_id * c + i;
        lane_tiles += group_tiles(g < groups ? group_rows(g) : 0);
    }
    const int tiles_incl = wave_inclusive_sum(lane_tiles, lane_id);
    const int tiles = __builtin_amdgcn_readlane(tiles_incl, 63);
    if (wgid >= tiles * tiles_n) return none;
    int t;
    mixed_band_order(wgid, tiles, tiles_n, t, n_tile);
    const int l = __builtin_ctzll(__builtin_amdgcn_ballot_w64(t < tiles_incl));
    int tile_base = __builtin_amdgcn_readlane(tiles_incl, l) - __builtin_amdgcn_readlane(lane_tiles, l);
    for (int b = 0; b < c; b += 64) {   // lane l's groups, 64 at a time (one step for G <= 4096)
        const int i = b + lane_id, g = l * c + i;
        const int rows = i < c && g < groups ? group_rows(g) : 0;
        const int n = group_tiles(rows);
        const int t_incl = wave_inclusive_sum(n, lane_id) + tile_base;
        const unsigned long long hit = __builtin_amdgcn_ballot_w64(t < t_incl);
        if (hit != 0) {
            const int q = __builtin_ctzll(hit);   // the group holding tile t
            const int group = l * c + b + q;
            const int grows = __builtin_amdgcn_readlane(rows, q);
            int local = t - (__builtin_amdgcn_readlane(t_incl, q) - __builtin_amdgcn_readlane(n, q));
            const bool masked = kargs.ptr_first == nullptr;
            const int row0 = masked ? 0 : kargs.ptr_first[group];
            const int m_end = masked ? grows : min(row0 + (grows + ALIGN - 1) / ALIGN * ALIGN, kargs.m);
            int cnt[3];
            mixed_counts<H0, H1, H2>(grows, cnt);
            int ofs = 0;
            for (int h = 0; h < 3; ++h) {
                if (local < cnt[h]) return {row0 + ofs + local * H[h], m_end, group, masked ? group : 0, H[h]};
                local -= cnt[h];
                ofs += cnt[h] * H[h];
            }
            return none;
        }
        tile_base = __builtin_amdgcn_readlane(t_incl, 63);
    }
    return none;
}

} // namespace opus_a16w16_grouped
#endif // __gfx950__
#endif // __HIP_DEVICE_COMPILE__

// ============================================================================
// One B_M x B_N tile of C (c_tile, columns from col), LDS smem_ab of at least Traits::LDS_BYTES:
// 2 x 2 waves, quadrants C00 / C10 / C01 / C11.
template<typename Traits, typename Kargs>
__device__ __forceinline__ void gemm_a16w16_grouped_tile_body(const Kargs& kargs,
                                                              const opus_a16w16_grouped::tile_t& c_tile, int col,
                                                              char* const smem_ab) {
#ifdef __HIP_DEVICE_COMPILE__
#if defined(__gfx950__)
    using namespace opus;
    using namespace opus_a16w16_grouped;

    using T = opus::remove_cvref_t<Traits>;
    using D_A = typename T::D_A;
    using D_B = typename T::D_B;
    using D_ACC = typename T::D_ACC;
    using D_OUT = typename T::D_OUT;

    const int num_tiles_n = ceil_div(kargs.n, T::B_N);
    const int row = c_tile.row;

    const int batch_id = c_tile.batch;
    const int wave_id = __builtin_amdgcn_readfirstlane(thread_id_x() / T::WARP_SIZE);
    const int lane_id = thread_id_x() % T::WARP_SIZE;

    // K-grouped: the segment's K tiles, at least 2 (rows past k_rows read 0).
    const int loops = [&] {
        if constexpr (T::K_GROUPED) return max(2, ceil_div(c_tile.k_rows, T::B_K));
        else return kargs.k / T::B_K;
    }();

    // Buffer views of A / B / C: 64-bit tile bases, ranges ending at the tile.
    const int stride_out = kargs.stride_c;
    const int c_n_rem = kargs.n - col;
    const int rows_in = c_tile.m_end - row < T::B_M ? c_tile.m_end - row : T::B_M;
    auto g_a = [&] {
        if constexpr (T::K_GROUPED)   // A [R][M]: the segment's rows, the range ends at its last row
            return make_gmem(reinterpret_cast<const D_A*>(kargs.ptr_a)
                                 + static_cast<long long>(c_tile.k_row0) * kargs.stride_a + row,
                             __builtin_amdgcn_readfirstlane(static_cast<unsigned int>(
                                 max(c_tile.k_rows * kargs.stride_a - row, 0) * sizeof(D_A))));
        else   // A [M][K] (masked: + the batch of the group)
            return make_gmem(reinterpret_cast<const D_A*>(kargs.ptr_a)
                                 + static_cast<long long>(batch_id) * kargs.stride_a_batch
                                 + static_cast<long long>(row) * kargs.stride_a,
                             __builtin_amdgcn_readfirstlane(static_cast<unsigned int>(
                                 rows_in * kargs.stride_a * sizeof(D_A))));
    }();
    auto g_b = [&] {
        if constexpr (T::K_GROUPED)   // B [R][N]
            return make_gmem(reinterpret_cast<const D_B*>(kargs.ptr_b)
                                 + static_cast<long long>(c_tile.k_row0) * kargs.stride_b + col,
                             __builtin_amdgcn_readfirstlane(static_cast<unsigned int>(
                                 max(c_tile.k_rows * kargs.stride_b - col, 0) * sizeof(D_B))));
        else   // B[g]: [N][K] (B_T: [K][N])
            return make_gmem(reinterpret_cast<const D_B*>(kargs.ptr_b) + c_tile.group * kargs.stride_b_group
                                 + col * (T::B_T ? 1 : kargs.stride_b),
                             __builtin_amdgcn_readfirstlane(static_cast<unsigned int>(
                                 (T::B_T ? kargs.k * kargs.stride_b - col : (kargs.n - col) * kargs.stride_b)
                                 * sizeof(D_B))));
    }();
    const int c_m_rem = rows_in;
    const long long c_tile_ofs = [&] {   // K-grouped: C [G][M][N]
        if constexpr (T::K_GROUPED)
            return c_tile.group * kargs.stride_c_group + static_cast<long long>(row) * stride_out + col;
        else return 0ll;
    }();
    D_OUT* const ptr_out = [&] {
        if constexpr (T::K_GROUPED) return reinterpret_cast<D_OUT*>(kargs.ptr_c) + c_tile_ofs;
        else return reinterpret_cast<D_OUT*>(kargs.ptr_c) + static_cast<long long>(batch_id) * kargs.stride_c_batch
                    + static_cast<long long>(row) * stride_out + col;
    }();
    const unsigned int c_range =
        __builtin_amdgcn_readfirstlane(static_cast<unsigned int>(c_m_rem * stride_out * sizeof(D_OUT)));
    auto g_c = make_gmem(ptr_out, c_range);
    // K-grouped fp32 C = src + sum of segments: segment 0 reads src (none: 0), later ones C itself.
    constexpr bool RMW = T::K_GROUPED && T::C_F32;
    auto g_src = [&] {
        if constexpr (RMW) {
            const D_OUT* const src = c_tile.seg > 0 ? ptr_out
                : kargs.ptr_src != nullptr ? reinterpret_cast<const D_OUT*>(kargs.ptr_src) + c_tile_ofs : nullptr;
            return make_gmem(src != nullptr ? src : ptr_out, src != nullptr ? c_range : 0u);
        } else {
            return 0;
        }
    }();

    const int wave_id_m = wave_id % T::T_M;
    const int wave_id_n = wave_id / T::T_M;

    auto u_ga = make_layout_ga<T>(lane_id, wave_id_m, wave_id_n, kargs.stride_a);
    auto u_sa = make_layout_sa<T>(wave_id_m, wave_id_n);
    auto u_gb = make_layout_gb<T>(lane_id, wave_id_m, wave_id_n, kargs.stride_b);
    auto u_sb = make_layout_sb<T>(wave_id_m, wave_id_n);
    const auto u_ra0 = make_layout_ra<T>(lane_id, wave_id_m, wave_id_n, 0);
    const auto u_ra1 = make_layout_ra<T>(lane_id, wave_id_m, wave_id_n, 1);
    const auto u_rb0 = make_layout_rb<T>(lane_id, wave_id_m, wave_id_n, 0);
    const auto u_rb1 = make_layout_rb<T>(lane_id, wave_id_m, wave_id_n, 1);

    constexpr int smem_a_elem = T::smem_m_rep * (T::smem_linear_wave_a + T::smem_padding);
    constexpr int smem_b_elem = T::smem_n_rep * (T::smem_linear_wave_b + T::smem_padding);
    constexpr int smem_a_bytes = smem_a_elem * 4 * sizeof(D_A);
    constexpr int smem_b_bytes = smem_b_elem * 4 * sizeof(D_B);
    static_assert(smem_a_bytes + smem_b_bytes <= T::LDS_BYTES);
    char* const smem_a = smem_ab;
    char* const smem_b = smem_ab + smem_a_bytes;
    auto s_a = make_smem(reinterpret_cast<D_A*>(smem_a));
    auto s_b = make_smem(reinterpret_cast<D_B*>(smem_b));

    auto mma = make_tiled_mma<D_A, D_B, D_ACC>(
        seq<T::E_M, T::E_N, 1>{},
        seq<T::T_M, T::T_N, T::T_K>{},
        seq<T::W_M, T::W_N, T::W_K>{},
        mfma_adaptor_swap_ab{});
    using MMA = decltype(mma);

    // Operands [half][kk][repeat]: one MFMA operand (8 bf16) per lane, kk = K 0..31 / 32..63 of the K tile;
    // accumulators [half_m][half_n][fragment].
    operand_t v_a[2][2][T::E_M];   // [half][kk][repeat]
    operand_t v_b[2][2][T::E_N];
    using vtype_c = opus::vector_t<D_ACC, MMA::mma_c_len>;
    vtype_c v_c[2][2][T::E_M * T::E_N];
    opus::static_for<2>([&](auto hm) { opus::static_for<2>([&](auto hn) { opus::static_for<T::E_M * T::E_N>([&](auto f) {
        clear(v_c[hm.value][hn.value][f.value]);
    }); }); });

    auto ga_offset = [&](int half_tile_m, int tile_k) {
        if constexpr (T::A_T) return tile_k * T::B_K * kargs.stride_a + half_tile_m * T::HALF_B_M;
        else return half_tile_m * T::HALF_B_M * kargs.stride_a + tile_k * T::B_K;
    };
    auto gb_offset = [&](int half_tile_n, int tile_k) {
        if constexpr (T::B_T) return tile_k * T::B_K * kargs.stride_b + half_tile_n * T::HALF_B_N;
        else return half_tile_n * T::HALF_B_N * kargs.stride_b + tile_k * T::B_K;
    };
    auto sa_offset = [&](int stage, int half_tile_m) { return (stage * 2 + half_tile_m) * smem_a_elem; };
    auto sb_offset = [&](int stage, int half_tile_n) { return (stage * 2 + half_tile_n) * smem_b_elem; };
    auto lds_a = [&](int stage) { return reinterpret_cast<OPUS_LDS_ADDR D_A*>(s_a.ptr) + sa_offset(stage, 0); };
    auto lds_b = [&](int stage) { return reinterpret_cast<OPUS_LDS_ADDR D_B*>(s_b.ptr) + sb_offset(stage, 0); };

    // Global -> LDS: half tile half_tile_m / half_tile_n of K tile tile_k into stage.
    auto prefetch_a = [&](int stage, int tile_k, int half_tile_m) {
        async_load<T::VEC_A>(g_a, s_a.ptr, u_ga, u_sa + sa_offset(stage, half_tile_m), ga_offset(half_tile_m, tile_k));
    };
    auto prefetch_b = [&](int stage, int tile_k, int half_tile_n) {
        async_load<T::VEC_B>(g_b, s_b.ptr, u_gb, u_sb + sb_offset(stage, half_tile_n), gb_offset(half_tile_n, tile_k));
    };

    // LDS -> registers: v[kk][e] = the operand of MFMA kk (K 32 kk .. 32 kk + 31) of repeat e, 8 bf16 per
    // lane, pinned at a[agpr + 8 e + 4 kk]. half_ofs: the half tile in the stage.
    //   K-contiguous:   1 ds_read_b128 per operand, by the read layout of K step kk
    //   M/N-contiguous: 2 ds_read_b64_tr_b16 (4 bf16 each) per operand at the swizzled (e, h) bases + 32 kk k rows
    using i32x2_t = opus::vector_t<opus::i32_t, 2>;
    auto tr16 = [&](auto& sm_, int ofs) {
        // pointer arithmetic on the view's (restrict) LDS pointer keeps the no-alias fact (no vmcnt(0) before it)
        return __builtin_bit_cast(i32x2_t, __builtin_amdgcn_ds_read_tr16_b64_v4i16(
            reinterpret_cast<OPUS_LDS_ADDR opus::vector_t<short, 4>*>(reinterpret_cast<OPUS_LDS_ADDR short*>(sm_.ptr) + ofs)));
    };
    auto load_k = [&](auto& sm_, const auto& u_r0, const auto& u_r1, auto& v, int half_ofs, auto agpr_) {
        constexpr int E = sizeof(v[0]) / sizeof(v[0][0]);
        const auto x0 = load<T::VEC_A>(sm_, u_r0 + half_ofs);         // E operands of K step 0
        const auto x1 = load<T::VEC_A>(sm_, u_r1 + half_ofs);         // E operands of K step 1
        opus::static_for<E>([&](auto e) {
            constexpr int agpr = decltype(agpr_)::value + 8 * e.value;
            set_operand<agpr>(v[0][e.value], opus::slice(x0, opus::number<8 * e.value>{}, opus::number<8 * e.value + 8>{}));
            set_operand<agpr + 4>(v[1][e.value], opus::slice(x1, opus::number<8 * e.value>{}, opus::number<8 * e.value + 8>{}));
        });
    };
    // M/N-contiguous read bases [e][h] (K step 0, swizzled), built once
    constexpr int tr_per_mfma = T::VEC_A / T::VEC_TR;
    auto mn_bases = [&](const auto& u_r, int wave, auto e_, auto t_w_) {
        constexpr int E = decltype(e_)::value;
        const auto o = opus::layout_to_offsets<T::VEC_TR>(u_r);        // [kk][h][e]
        opus::array<int, tr_per_mfma * E> b;
        opus::static_for<E>([&](auto e) { opus::static_for<tr_per_mfma>([&](auto h) {
            b[tr_per_mfma * e.value + h.value] = o[h.value * E + e.value]
                + mn_r_xor<T, E, decltype(t_w_)::value>(lane_id, wave, e.value, h.value);
        }); });
        return b;
    };
    const auto ra_mn = [&] { if constexpr (T::A_T) return mn_bases(u_ra0, wave_id_m, number<T::E_M>{}, number<T::T_M>{}); else return 0; }();
    const auto rb_mn = [&] { if constexpr (T::B_T) return mn_bases(u_rb0, wave_id_n, number<T::E_N>{}, number<T::T_N>{}); else return 0; }();
    auto load_mn = [&](auto& sm_, const auto& b, auto& v, int half_ofs, auto row_, auto agpr_) {
        constexpr int E = sizeof(v[0]) / sizeof(v[0][0]);
        constexpr int row = decltype(row_)::value;                       // one k row of the half tile
        opus::static_for<E>([&](auto e) { opus::static_for<2>([&](auto kk) {
            constexpr int agpr = decltype(agpr_)::value + 8 * e.value + 4 * kk.value;
            const int k_ofs = half_ofs + kk.value * T::W_K * row;        // K step kk: W_K k rows further
            set_operand<agpr>(v[kk.value][e.value], __builtin_bit_cast(operand_t, opus::concat_vector(
                tr16(sm_, b[tr_per_mfma * e.value] + k_ofs), tr16(sm_, b[tr_per_mfma * e.value + 1] + k_ofs))));
        }); });
    };
    auto load_a = [&](auto& sm_, auto half_) {
        constexpr int half = decltype(half_)::value;
        if constexpr (T::A_T) load_mn(sm_, ra_mn, v_a[half], sa_offset(0, half), number<T::HALF_B_M>{}, number<half * A_REGS<T> / 2>{});
        else load_k(sm_, u_ra0, u_ra1, v_a[half], sa_offset(0, half), number<half * A_REGS<T> / 2>{});
    };
    auto load_b = [&](auto& sm_, auto half_) {
        constexpr int half = decltype(half_)::value;
        if constexpr (T::B_T) load_mn(sm_, rb_mn, v_b[half], sb_offset(0, half), number<T::HALF_B_N>{}, number<A_REGS<T> + half * B_REGS<T> / 2>{});
        else load_k(sm_, u_rb0, u_rb1, v_b[half], sb_offset(0, half), number<A_REGS<T> + half * B_REGS<T> / 2>{});
    };

    // LDS reads go through __restrict__ copies of the views passed as function parameters (clang ignores
    // __restrict__ on locals): otherwise the reads cannot be told apart from the in-flight LDS-DMA copies
    // and the compiler puts a vmcnt(0) before them.
    auto views = [&](OPUS_LDS_ADDR D_A* __restrict__ read_a, OPUS_LDS_ADDR D_B* __restrict__ read_b, auto&& body) {
        auto rd_a = s_a; rd_a.ptr = reinterpret_cast<OPUS_LDS_ADDR char*>(read_a);
        auto rd_b = s_b; rd_b.ptr = reinterpret_cast<OPUS_LDS_ADDR char*>(read_b);
        body(rd_a, rd_b);
    };

    // One quadrant (HM, HN): 2 MFMAs per fragment, accumulators pinned (c_reg).
    constexpr int frags = T::E_M * T::E_N;
    auto quadrant = [&](auto hm_, auto hn_) {
        constexpr int hm = decltype(hm_)::value, hn = decltype(hn_)::value;
        opus::static_for<frags>([&](auto i) {
            constexpr int m = decltype(i)::value / T::E_N, n = decltype(i)::value % T::E_N;
            mma_bf16<c_reg<T, hm, hn, m, n>(), MMA>(v_a[hm][0][m], v_b[hn][0][n], v_c[hm][hn][m * T::E_N + n]);   // kk = 0
            mma_bf16<c_reg<T, hm, hn, m, n>(), MMA>(v_a[hm][1][m], v_b[hn][1][n], v_c[hm][hn][m * T::E_N + n]);   // kk = 1
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
            sync_threads();
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
        sync_threads();
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

    // Prologue: copy tiles 0 and 1, wait for tile 0, read it and compute C00(0)
    prefetch_a(0, 0, 0);
    prefetch_b(0, 0, 0);
    prefetch_a(0, 0, 1);
    prefetch_b(0, 0, 1);
    if (loops > 1) {
        prefetch_a(1, 1, 0);
        prefetch_b(1, 1, 0);
        prefetch_a(1, 1, 1);
        prefetch_b(1, 1, 1);
        s_waitcnt_vmcnt(number<2 * VMH>{});   // tile 1 may fly
    } else {
        s_waitcnt_vmcnt(0_I);
    }
    sync_threads();
    __builtin_amdgcn_sched_barrier(0);
    views(lds_a(0), lds_b(0), [&](auto& rd_a, auto& rd_b) {   // tile 1 may still be in flight
        load_a(rd_a, number<0>{});
        load_b(rd_b, number<0>{});
        __builtin_amdgcn_sched_barrier(0);
        load_a(rd_a, number<1>{});
        load_b(rd_b, number<1>{});
        quadrant(number<0>{}, number<0>{});
        sched_gen<SL, DSA + DSB, 2, 0>();
        __builtin_amdgcn_sched_barrier(0);
    });

    // Main loop: 2 K tiles per trip, stages compile-time
    // Keep the control flow straight (once-loops, plain blocks): pinned values crossing loop exits or
    // branch joins lose their AGPR assignment.
    int tile = 0;
#pragma clang loop unroll(disable)
    for (; tile + 3 < loops; tile += 2) {
        views(lds_a(1), lds_b(1), [&](auto& rd_a, auto& rd_b) {
            k_tile(rd_a, rd_b, 0, tile, true_type{});
        });
        views(lds_a(0), lds_b(0), [&](auto& rd_a, auto& rd_b) {
            k_tile(rd_a, rd_b, 1, tile + 1, true_type{});
        });
    }
    // At most one more tile with a prefetch (odd number of them).
#pragma clang loop unroll(disable)
    for (int once = 0; once < 1 && tile + 2 < loops; ++once, ++tile) {
        const int stage = tile & 1;
        views(lds_a(stage ^ 1), lds_b(stage ^ 1), [&](auto& rd_a, auto& rd_b) {
            k_tile(rd_a, rd_b, stage, tile, true_type{});
        });
    }

    // Epilogue: the last two K tiles
    {   // second-to-last tile (loops >= 2): no more copies
        const int next_stage = (tile & 1) ^ 1;
        views(lds_a(next_stage), lds_b(next_stage), [&](auto& rd_a, auto& rd_b) {
            k_tile(rd_a, rd_b, 0, tile, false_type{});
        });
        ++tile;
    }
    {   // last tile: C10, C01, C11 (its operands are already in registers)
        auto last = [&](auto hm_, auto hn_) {
            quadrant(hm_, hn_);
            sched_gen<SL, 0, 0, 0>();
            __builtin_amdgcn_sched_barrier(0);
        };
        last(number<1>{}, number<0>{});
        last(number<0>{}, number<1>{});
        last(number<1>{}, number<1>{});
    }

    // Write back C through LDS
    // A wave owns only 16 columns of a row per fragment, so direct fragment stores would leave every
    // 128-byte line partially written. C is staged in the idle LDS, rows padded by 8 elements (the 16
    // rows of a fragment step across the banks): as one image of the tile when it fits (one region per
    // quadrant, no reuse), else quadrant by quadrant in 2 regions (C00 / C01 -> 0, C10 / C11 -> 1).
    // Per quadrant: stage -> sync -> 16-byte copies (non-temporal), columns >= N dropped.
    constexpr bool FULL = T::C_STAGE_FULL;
    constexpr int c_lds_stride = FULL ? T::B_N + 8 : T::HALF_B_N + 8;
    constexpr int VEC_OUT = 16 / sizeof(D_OUT);
    static_assert((FULL ? T::B_M : 2 * T::HALF_B_M) * c_lds_stride * sizeof(D_OUT) <= T::LDS_BYTES);
    auto s_c = make_smem(reinterpret_cast<D_OUT*>(smem_ab));
    auto p_coord_c = opus::make_tuple(wave_id_m, lane_id % mma.grpn_c, wave_id_n, lane_id / mma.grpn_c);
    auto u_sc = partition_layout_c<T::VEC_C>(mma, opus::make_tuple(opus::number<c_lds_stride>{}, 1_I), p_coord_c);
    auto sc_offset = [&](int half_tile_m, int half_tile_n) {
        return half_tile_m * T::HALF_B_M * c_lds_stride + (FULL ? half_tile_n * T::HALF_B_N : 0);
    };
    // K-grouped: thread id recomputed here (opaque mbcnt), not kept live through the main loop (spills).
    const int output_thread_id = [&] {
        if constexpr (T::K_GROUPED) {
            unsigned int lane_mask = ~0u;
            asm volatile("" : "+s"(lane_mask));
            return wave_id * T::WARP_SIZE
                + static_cast<int>(__builtin_amdgcn_mbcnt_hi(lane_mask, __builtin_amdgcn_mbcnt_lo(lane_mask, 0u)));
        } else {
            return static_cast<int>(thread_id_x());
        }
    }();
    // Last tile column with N % VEC_OUT != 0: per-element stores (K-grouped: N % 8 == 0, never).
    const bool ragged = !T::K_GROUPED && c_n_rem < T::B_N && (c_n_rem % VEC_OUT) != 0;   // wave-uniform
    // columns >= N: offset 2^31 bytes, dropped by the buffer descriptor
    constexpr int OOB = static_cast<int>((1u << 31) / sizeof(D_OUT));
    constexpr int OUT_ITERS = T::HALF_B_M * T::HALF_B_N / (T::BLOCK_SIZE * VEC_OUT);
    using out_vec_t = opus::vector_t<D_OUT, VEC_OUT>;
    auto copy_output = [&](int half_tile_m, int half_tile_n, const auto& src) {
        if (!ragged) {
            opus::static_for<OUT_ITERS>([&](auto i) {
                const int linear = output_thread_id * VEC_OUT + decltype(i)::value * T::BLOCK_SIZE * VEC_OUT;
                const int r = linear / T::HALF_B_N + half_tile_m * T::HALF_B_M;
                const int cl = linear % T::HALF_B_N, c = cl + half_tile_n * T::HALF_B_N;
                const auto value = load<VEC_OUT>(s_c, r * c_lds_stride + (FULL ? c : cl));
                if constexpr (RMW)
                    store<VEC_OUT>(g_c, value + src[decltype(i)::value], c < c_n_rem ? r * stride_out + c : OOB, 0,
                                   opus::number<2>{});
                else
                    store<VEC_OUT>(g_c, value, c < c_n_rem ? r * stride_out + c : OOB, 0, opus::number<2>{});
            });
        } else {
#pragma clang loop unroll(disable)
            for (int linear = output_thread_id; linear < T::HALF_B_M * T::HALF_B_N; linear += T::BLOCK_SIZE) {
                const int r = linear / T::HALF_B_N + half_tile_m * T::HALF_B_M;
                const int cl = linear % T::HALF_B_N, c = cl + half_tile_n * T::HALF_B_N;
                g_c.template store<1>(load<1>(s_c, r * c_lds_stride + (FULL ? c : cl)),
                                      c < c_n_rem ? r * stride_out + c : OOB, 0, opus::number<2>{});
            }
        }
    };
    auto store_quadrant = [&](const auto& acc, auto half_tile_m_, auto half_tile_n_) {
        constexpr int hm = decltype(half_tile_m_)::value, hn = decltype(half_tile_n_)::value;
        typename MMA::vtype_c quad;
        opus::static_for<T::E_M * T::E_N>([&](auto f) {
            opus::set_slice(quad, acc[f.value], opus::number<f.value * MMA::mma_c_len>{}, opus::number<(f.value + 1) * MMA::mma_c_len>{});
        });
        if constexpr (!FULL && hn == 1) {   // region hm is reused: every wave is done copying (hm, 0) out
            s_waitcnt_lgkmcnt(0_I);
            sync_threads();
        }
        // RMW: the src values of the quadrant, loaded before the staging to hide their latency
        out_vec_t src[RMW ? OUT_ITERS : 1];
        if constexpr (RMW) {
            opus::static_for<OUT_ITERS>([&](auto i) {   // the elements copy_output stores
                const int linear = output_thread_id * VEC_OUT + decltype(i)::value * T::BLOCK_SIZE * VEC_OUT;
                const int r = linear / T::HALF_B_N + hm * T::HALF_B_M;
                const int c = linear % T::HALF_B_N + hn * T::HALF_B_N;
                src[decltype(i)::value] = load<VEC_OUT>(g_src, c < c_n_rem ? r * stride_out + c : OOB);
            });
        }
        store<T::VEC_C>(s_c, cast<D_OUT>(quad), u_sc + sc_offset(hm, hn));
        __builtin_amdgcn_sched_barrier(0);
        if constexpr (!(hm == 1 && hn == 0)) s_waitcnt_vmcnt(0_I);
        s_waitcnt_lgkmcnt(0_I);
        sync_threads();
        __builtin_amdgcn_sched_barrier(0);
        copy_output(hm, hn, src);
    };
    // RMW: segment s waits for turn == s (segment s - 1 is dispatched earlier), then acquires C.
    [[maybe_unused]] int* const turn = [&] {
        if constexpr (RMW)
            return kargs.turn + (c_tile.group * ceil_div(kargs.m, T::B_M) + row / T::B_M) * num_tiles_n + col / T::B_N;
        else return nullptr;
    }();
    if constexpr (RMW) {
        if (c_tile.seg > 0) {
            if (thread_id_x() == 0) {
                while (__hip_atomic_load(turn, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) != c_tile.seg)
                    __builtin_amdgcn_s_sleep(2);
            }
            __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
        }
    }
    // all waves are done with their last operand reads
    s_waitcnt_vmcnt(0_I);
    s_waitcnt_lgkmcnt(0_I);
    sync_threads();
    __builtin_amdgcn_sched_barrier(0);
    store_quadrant(v_c[0][0], number<0>{}, number<0>{});
    store_quadrant(v_c[1][0], number<1>{}, number<0>{});
    store_quadrant(v_c[0][1], number<0>{}, number<1>{});
    store_quadrant(v_c[1][1], number<1>{}, number<1>{});
    // RMW: hand the tile to the next segment once every wave's C stores are visible (release)
    if constexpr (RMW) {
        if (c_tile.seg + 1 < c_tile.nseg) {
            __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
            sync_threads();
            if (thread_id_x() == 0)
                __hip_atomic_store(turn, c_tile.seg + 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        }
    }
#endif // __gfx950__
#endif // __HIP_DEVICE_COMPILE__
}

// K-grouped kernel: block -> (work item, M tile, N tile), the item's tiles in mixed_band_order.
template<typename Traits, typename Kargs>
__device__ __forceinline__ void gemm_a16w16_k_grouped_body(const Kargs& kargs) {
#ifdef __HIP_DEVICE_COMPILE__
#if defined(__gfx950__)
    using namespace opus_a16w16_grouped;
    using T = opus::remove_cvref_t<Traits>;
    static_assert(T::K_GROUPED);
    const int wgid = opus::block_id_x();
    const int num_tiles_n = ceil_div(kargs.n, T::B_N);
    const int tiles_m = ceil_div(kargs.m, T::B_M);
    const int item = wgid / (tiles_m * num_tiles_n);
    int m_tile, n_tile;
    mixed_band_order(wgid % (tiles_m * num_tiles_n), tiles_m, num_tiles_n, m_tile, n_tile);
    const tile_t c_tile = k_grouped_tile<T>(kargs, item * tiles_m + m_tile);
    if (c_tile.row < 0) return;
    __shared__ __attribute__((aligned(16))) char smem_ab[T::LDS_BYTES];
    gemm_a16w16_grouped_tile_body<Traits>(kargs, c_tile, n_tile * T::B_N, smem_ab);
#endif // __gfx950__
#endif // __HIP_DEVICE_COMPILE__
}

// M-grouped kernel: tile bodies T0 / T1 / T2 (B_M 256 / 192 / 128) picked per group, one shared LDS buffer.
template<typename T0, typename T1, typename T2, typename Kargs>
__device__ __forceinline__ void gemm_a16w16_m_grouped_body(const Kargs& kargs) {
#ifdef __HIP_DEVICE_COMPILE__
#if defined(__gfx950__)
    using namespace opus_a16w16_grouped;
    static_assert(T0::B_N == T1::B_N && T1::B_N == T2::B_N && T0::B_T == T1::B_T && T1::B_T == T2::B_T
                  && !T0::K_GROUPED && !T1::K_GROUPED && !T2::K_GROUPED, "M-grouped tiles of one B_N and layout");
    constexpr int LDS_01 = T0::LDS_BYTES > T1::LDS_BYTES ? T0::LDS_BYTES : T1::LDS_BYTES;
    constexpr int LDS_BYTES = LDS_01 > T2::LDS_BYTES ? LDS_01 : T2::LDS_BYTES;
    const int num_tiles_n = ceil_div(kargs.n, T0::B_N);
    const int wgid = opus::block_id_x();
    int n_tile;
    const tile_t tile = m_grouped_tile<T0::B_M, T1::B_M, T2::B_M>(kargs, wgid, num_tiles_n, n_tile);
    if (tile.row < 0) return;
    __shared__ __attribute__((aligned(16))) char smem_ab[LDS_BYTES];
    const int col = n_tile * T0::B_N;
    // each body runs to the end: no pinned value crosses the branches
    if (tile.height == T0::B_M) gemm_a16w16_grouped_tile_body<T0>(kargs, tile, col, smem_ab);
    else if (tile.height == T1::B_M) gemm_a16w16_grouped_tile_body<T1>(kargs, tile, col, smem_ab);
    else gemm_a16w16_grouped_tile_body<T2>(kargs, tile, col, smem_ab);
#endif // __gfx950__
#endif // __HIP_DEVICE_COMPILE__
}

// Kernel arguments of a grouped traits.
template<typename Traits>
using opus_gemm_a16w16_grouped_kargs_t = std::conditional_t<Traits::K_GROUPED, opus_gemm_a16w16_k_grouped_kargs_gfx950,
                                                            opus_gemm_a16w16_grouped_kargs_gfx950>;
