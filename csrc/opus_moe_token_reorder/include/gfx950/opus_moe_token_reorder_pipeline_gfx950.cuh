// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

// __HIP_DEVICE_COMPILE__.
#include "opus_moe_token_reorder_traits_gfx950.cuh"
#include <hip/hip_runtime.h>

// K4 / K5 round a product to fp32 before anything is added to it: never contract a * b + c into an FMA.
#pragma clang fp contract(off)

namespace opus_moe_token_reorder {


// ============================================================================
// Shared
// ============================================================================

// Workgroup barrier between LDS phases: s_barrier, and the fences make it wait for the outstanding LDS accesses.
__device__ inline void lds_barrier() {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup");
    opus::sync_threads();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup");
}

// View of one row of a [rows, H] buffer: 64-bit base (the buffer can pass 4 GiB), 32-bit offsets inside the row.
// The row is in the base, never in the s_os of a store: a 16-byte buffer store with a register s_os gets no
// hazard nop, and an instruction right behind it that writes its data register changes what is stored.
template<class T>
__device__ inline auto row_view(const char* rows, size_t row, int chunks) {
    return opus::make_gmem(
        reinterpret_cast<const typename T::D_ROW*>(rows) + row * chunks * T::VEC,
        static_cast<unsigned>(chunks) * T::CHUNK_BYTES);
}

// ============================================================================
// K2 / K3 / K4 / K5: one workgroup per destination row
// ============================================================================
// K2 permute fwd:    x_perm [R, H] = x [N, H] by sorted_token_ids [R], padding rows 0
// K3 permute bwd:    dx [N, H]     = d_perm [R, H] by reverse_sorted [N]
// K4 unpermute fwd:  out [N, H]    = weights [N] * y_perm [R, H] by reverse_sorted [N]
// K5 unpermute bwd:  d_perm [R, H] = weights [N] * d_out [N, H] by sorted_token_ids [R], padding rows 0
//                    d_weights [N] = dot(d_out [N, H], y_perm [R, H])

// one row [H]: [NUM_WAVES][WARP_SIZE][VEC] -> [4][64][8], H > 2048: + 2048 per pass
template<class T>
__device__ inline auto make_layout_global_row(int wave_id, int lane_id) {
    constexpr auto shape = opus::make_tuple(
        opus::number<T::NUM_WAVES>{}, // 4
        opus::number<T::WARP_SIZE>{}, // 64
        opus::number<T::VEC>{});      // 8 bf16 = 16 bytes

    constexpr auto dim = opus::make_tuple(
        opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::y_dim{}));

    return opus::make_layout<T::VEC>(
        shape,
        opus::unfold_x_stride(dim, shape, opus::tuple{opus::number<1>{}}),
        opus::unfold_p_coord(dim, opus::tuple{wave_id, lane_id}));
}

// K5 LDS partial sums of d_weights: [NUM_WAVES][WARP_SIZE] -> [256]
template<class T>
__device__ inline auto make_layout_lds_partial(int wave_id, int lane_id) {
    constexpr auto shape = opus::make_tuple(
        opus::number<T::NUM_WAVES>{}, // 4
        opus::number<T::WARP_SIZE>{}, // 64
        opus::number<1>{});

    constexpr auto dim = opus::make_tuple(
        opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::y_dim{}));

    return opus::make_layout<1>(
        shape,
        opus::unfold_x_stride(dim, shape, opus::tuple{opus::number<1>{}}),
        opus::unfold_p_coord(dim, opus::tuple{wave_id, lane_id}));
}

// K2 / K3 / K4: dst[b] = src[index[b]] (WEIGHTED: times weights[b], one rounding to bf16); index[b] < 0: dst[b] = 0.
// CPOL_DST: cache policy of the stores. NT: the loads are non-temporal.
template<bool WEIGHTED, int CPOL_DST, bool NT>
__device__ __forceinline__ void move_row_body(const opus_moe_permute_rows_kargs& k) {
    using T = row_traits;
    constexpr int CPOL_SRC = NT ? T::CPOL_NT : T::CPOL_DEFAULT;

    // 1 thread; the destination row and its source row (the same in every lane)
    const int tid = opus::thread_id_x();
    const int wave_id = tid / T::WARP_SIZE, lane_id = tid % T::WARP_SIZE;
    const size_t row = static_cast<unsigned>(opus::block_id_x());
    const int src = __builtin_amdgcn_readfirstlane(k.index[row]);

    // 2 views of the two rows; layout of one pass over a row
    auto g_dst = row_view<T>(k.dst, row, k.chunks);
    auto g_src = row_view<T>(k.src, src < 0 ? 0 : src, k.chunks);
    const auto u_row = make_layout_global_row<T>(wave_id, lane_id);

    // 3 padding row: 0. Thread i has the 16-byte chunks i, i + 256, ...
    if (src < 0) {
        opus::vector_t<T::D_ROW, T::VEC> zero;
        opus::clear(zero);
        for (int chunk = tid; chunk < k.chunks; chunk += T::BLOCK_SIZE)
            opus::store<T::VEC>(g_dst, zero, u_row + (chunk - tid) * T::VEC, 0, opus::number<CPOL_DST>{});
        return;
    }

    // 4 copy: global -> vgpr -> global
    const float w = WEIGHTED
        ? __builtin_bit_cast(float, __builtin_amdgcn_readfirstlane(reinterpret_cast<const int*>(k.weights)[row]))
        : 1.0f;
    for (int chunk = tid; chunk < k.chunks; chunk += T::BLOCK_SIZE) {
        const auto u = u_row + (chunk - tid) * T::VEC;
        const auto v = opus::load<T::VEC>(g_src, u, 0, opus::number<CPOL_SRC>{});
        if constexpr (WEIGHTED) {   // bf16 -> fp32 (exact), times w, fp32 -> bf16 (nearest even)
            const opus::vector_t<T::D_ACC, T::VEC> f = opus::cast<T::D_ACC>(v) * w;
            opus::store<T::VEC>(g_dst, opus::cast<T::D_ROW>(f), u, 0, opus::number<CPOL_DST>{});
        } else {
            opus::store<T::VEC>(g_dst, v, u, 0, opus::number<CPOL_DST>{});
        }
    }
}

// K2: x_perm is written once: non-temporal stores.
__global__ void __launch_bounds__(row_traits::BLOCK_SIZE) moe_permute_fwd(opus_moe_permute_rows_kargs k) {
    if (k.stream)
        move_row_body<false, row_traits::CPOL_NT, true>(k);
    else
        move_row_body<false, row_traits::CPOL_NT, false>(k);
}

// K3: the store has system scope (sc0 sc1): it goes past the L2, which the gather loads of the same wave are
// streaming through (measured against a plain store: K3 -2.7%, K4 -2.5%).
__global__ void __launch_bounds__(row_traits::BLOCK_SIZE) moe_permute_bwd(opus_moe_permute_rows_kargs k) {
    if (k.stream)
        move_row_body<false, row_traits::CPOL_SYSTEM, true>(k);
    else
        move_row_body<false, row_traits::CPOL_SYSTEM, false>(k);
}

// K4
__global__ void __launch_bounds__(row_traits::BLOCK_SIZE) moe_unpermute_fwd(opus_moe_permute_rows_kargs k) {
    if (k.stream)
        move_row_body<true, row_traits::CPOL_SYSTEM, true>(k);
    else
        move_row_body<true, row_traits::CPOL_SYSTEM, false>(k);
}

// K5: row r of token i, d = d_out[i]: d_perm[r] = weights[i] * d, d_weights[i] = dot(d, y_perm[r]); padding rows 0.
template<bool NT, class Smem>
__device__ __forceinline__ void unpermute_bwd_body(const opus_moe_unpermute_bwd_kargs& k, Smem& s_partial) {
    using T = row_traits;
    constexpr int CPOL_SRC = NT ? T::CPOL_NT : T::CPOL_DEFAULT;

    // 1 thread; the row and its token (the same in every lane)
    const int tid = opus::thread_id_x();
    const int wave_id = tid / T::WARP_SIZE, lane_id = tid % T::WARP_SIZE;
    const size_t row = static_cast<unsigned>(opus::block_id_x());
    const int token = __builtin_amdgcn_readfirstlane(k.sorted_token_ids[row]);

    // 2 views of the three rows; layouts
    auto g_dp = row_view<T>(k.d_perm, row, k.chunks);
    auto g_y = row_view<T>(k.y_perm, row, k.chunks);
    auto g_d = row_view<T>(k.d_out, token < 0 ? 0 : token, k.chunks);
    const auto u_row = make_layout_global_row<T>(wave_id, lane_id);
    const auto u_partial = make_layout_lds_partial<T>(wave_id, lane_id);

    // 3 padding row: 0
    if (token < 0) {
        opus::vector_t<T::D_ROW, T::VEC> zero;
        opus::clear(zero);
        for (int chunk = tid; chunk < k.chunks; chunk += T::BLOCK_SIZE)
            opus::store<T::VEC>(g_dp, zero, u_row + (chunk - tid) * T::VEC, 0, opus::number<T::CPOL_NT>{});
        return;
    }

    // 4 d_perm (written once: non-temporal), and the thread's partial sum of the dot product: the products of
    // its chunks i, i + 256, ... in ascending element order, in fp32 (the reference repeats this order).
    // global -> vgpr -> global / LDS
    const float w = __builtin_bit_cast(float, __builtin_amdgcn_readfirstlane(reinterpret_cast<const int*>(k.weights)[token]));
    if (tid < k.chunks) {
        float sum;
        for (int chunk = tid; chunk < k.chunks; chunk += T::BLOCK_SIZE) {
            const auto u = u_row + (chunk - tid) * T::VEC;
            const opus::vector_t<T::D_ACC, T::VEC> d = opus::cast<T::D_ACC>(opus::load<T::VEC>(g_d, u, 0, opus::number<CPOL_SRC>{}));
            const opus::vector_t<T::D_ACC, T::VEC> p = d * opus::cast<T::D_ACC>(opus::load<T::VEC>(g_y, u, 0, opus::number<CPOL_SRC>{}));
            const bool first = chunk == tid;
            if (first)
                sum = p[0];
            for (int e = first ? 1 : 0; e < T::VEC; e++)
                sum = sum + p[e];
            const opus::vector_t<T::D_ACC, T::VEC> dw = d * w;
            opus::store<T::VEC>(g_dp, opus::cast<T::D_ROW>(dw), u, 0, opus::number<T::CPOL_NT>{});
        }
        opus::store<1>(s_partial, opus::make_vector(sum), u_partial);
    }
    lds_barrier();

    // 5 d_weights[token]: thread 0 adds the partial sums in ascending thread order. LDS -> vgpr -> global
    if (tid == 0) {
        const int n = k.chunks < T::BLOCK_SIZE ? k.chunks : T::BLOCK_SIZE;
        float acc = opus::load<1>(s_partial, 0)[0];
        for (int i = 1; i < n; i++)
            acc = acc + opus::load<1>(s_partial, i)[0];
        auto g_dw = opus::make_gmem(k.d_weights + token);
        opus::store<1>(g_dw, acc, 0);
    }
}

__global__ void __launch_bounds__(row_traits::BLOCK_SIZE) moe_unpermute_bwd(opus_moe_unpermute_bwd_kargs k) {
    __shared__ float partial[row_traits::BLOCK_SIZE];
    auto s_partial = opus::make_smem(partial);
    if (k.stream)
        unpermute_bwd_body<true>(k, s_partial);
    else
        unpermute_bwd_body<false>(k, s_partial);
}

// ============================================================================
// Sort
// ============================================================================
// in:  expert_ids [N]
// out: reverse_sorted [N], sorted_token_ids [R], grouped_layout [R], group_rows [E], num_valid_ids [1]
// R = roundup(N, 128) + 128 * (E - 1)
// rows in use = num_valid_ids <= R
//
// Two launches, one workgroup per block of 256 tokens (one thread per token), E <= 256 (one thread per expert).
// counts [E][blocks] (global) carries the count of every expert in every block from launch 1 to launch 2.

// expert_ids / reverse_sorted of the block: [NUM_WAVES][WARP_SIZE] -> [256]
template<class T>
__device__ inline auto make_layout_global_token(int wave_id, int lane_id) {
    constexpr auto shape = opus::make_tuple(
        opus::number<T::NUM_WAVES>{}, // 4
        opus::number<T::WARP_SIZE>{}, // 64
        opus::number<1>{});

    constexpr auto dim = opus::make_tuple(
        opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::y_dim{}));

    return opus::make_layout<1>(
        shape,
        opus::unfold_x_stride(dim, shape, opus::tuple{opus::number<1>{}}),
        opus::unfold_p_coord(dim, opus::tuple{wave_id, lane_id}));
}

// LDS bitmap [EXPERTS][TOKENS] bits -> [32][256]: bit (e, t) = token t of the block has expert e
template<class T>
__device__ inline auto make_layout_lds_expert_bitmap(int expert) {
    constexpr auto shape = opus::make_tuple(
        opus::number<T::EXPERTS>{}, // 32
        opus::number<T::WORDS>{});  // 8 x 32 bits

    constexpr auto dim = opus::make_tuple(
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}));

    return opus::make_layout<T::WORDS>(
        shape,
        opus::unfold_x_stride(dim, shape, opus::tuple{opus::number<1>{}}),
        opus::unfold_p_coord(dim, opus::tuple{expert}));
}

// counts [EXPERTS][blocks] -> [32][blocks]: the row of one expert. block_stride: blocks rounded up to COUNT_BATCH
template<class T>
__device__ inline auto make_layout_global_count(int expert, int block_stride) {
    constexpr auto shape = opus::make_tuple(
        opus::number<T::EXPERTS>{}, // 32
        opus::number<1>{});

    constexpr auto dim = opus::make_tuple(
        opus::make_tuple(opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}));

    return opus::make_layout<1>(
        shape,
        opus::unfold_x_stride(dim, shape, opus::tuple{block_stride, opus::number<1>{}}),
        opus::unfold_p_coord(dim, opus::tuple{expert}));
}

// LDS total / first [EXPERTS] -> [32]
template<class T>
__device__ inline auto make_layout_lds_expert(int expert) {
    constexpr auto shape = opus::make_tuple(
        opus::number<T::EXPERTS>{}, // 32
        opus::number<1>{});

    constexpr auto dim = opus::make_tuple(
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}));

    return opus::make_layout<1>(
        shape,
        opus::unfold_x_stride(dim, shape, opus::tuple{opus::number<1>{}}),
        opus::unfold_p_coord(dim, opus::tuple{expert}));
}

// Launch 1: counts[e][block] = tokens of the block with expert e.
__device__ __forceinline__ void sort_count_body(const opus_moe_sort_kargs& k) {
    using T = sort_traits;
    const int tid = opus::thread_id_x();
    const int wave_id = tid / T::WARP_SIZE, lane_id = tid % T::WARP_SIZE;
    const unsigned block = opus::block_id_x();
    const int block_stride = (k.blocks + T::COUNT_BATCH - 1) / T::COUNT_BATCH * T::COUNT_BATCH;

    // views
    __shared__ unsigned bits[T::EXPERTS * T::WORDS];
    auto s_bits = opus::make_smem(bits);
    auto words = reinterpret_cast<OPUS_LDS_ADDR unsigned*>(s_bits.ptr);   // the bitmap storage, for the atomic
    auto g_ids = opus::make_gmem(k.expert_ids + static_cast<size_t>(block) * T::TOKENS);   // the block
    auto g_counts = opus::make_gmem(k.counts);

    // layouts: the thread's token; thread i -> expert i (its bitmap row, its counts row)
    const auto u_token = make_layout_global_token<T>(wave_id, lane_id);
    const auto u_i_bits = make_layout_lds_expert_bitmap<T>(tid);
    const auto u_i_count = make_layout_global_count<T>(tid, block_stride);

    // 1 expert id of the thread's token. global -> vgpr
    const int token = block * T::TOKENS + tid;
    int expert = -1;   // no token (past N)
    if (token < k.tokens)
        expert = opus::load<1>(g_ids, u_token)[0];
    const bool valid = static_cast<unsigned>(expert) < static_cast<unsigned>(k.experts);   // 0 <= expert < E

    // 2 bitmap = 0. vgpr -> LDS
    for (int word = tid; word < k.experts * T::WORDS; word += T::BLOCK_SIZE)
        opus::store<1>(s_bits, 0u, word);
    lds_barrier();

    // 3 bit (expert, token) = 1: several threads write one word, atomic or. vgpr -> LDS
    if (valid) {
        const auto u_bits = make_layout_lds_expert_bitmap<T>(expert);   // the token's expert row
        __hip_atomic_fetch_or(words + u_bits(0) + tid / T::WORD_BITS, 1u << (tid % T::WORD_BITS),
                              __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WORKGROUP);
    }
    lds_barrier();

    // 4 thread i has expert i: counts[i][block] = the 1 bits of row i. LDS -> vgpr -> global
    if (tid < k.experts) {
        const auto row = opus::load<T::WORDS>(s_bits, u_i_bits);
        int count = 0;
        for (int w = 0; w < T::WORDS; w++)
            count += __builtin_popcount(row[w]);
        opus::store<1>(g_counts, opus::make_vector(count), u_i_count + block);
    }
}

__global__ void __launch_bounds__(workgroup_traits::BLOCK_SIZE) moe_sort_count(opus_moe_sort_kargs k) {
    sort_count_body(k);
}

// Launch 2: the row of every token. The workgroups behind the token blocks are fill workgroups (8 experts each):
// group_rows and the -1 of the padding rows.
__device__ __forceinline__ void sort_rows_body(const opus_moe_sort_kargs& k) {
    using T = sort_traits;
    const int tid = opus::thread_id_x();
    const int wave_id = tid / T::WARP_SIZE, lane_id = tid % T::WARP_SIZE;
    const unsigned block = opus::block_id_x();
    const bool token_wg = block < static_cast<unsigned>(k.blocks);   // else: a fill workgroup
    const int block_stride = (k.blocks + T::COUNT_BATCH - 1) / T::COUNT_BATCH * T::COUNT_BATCH;
    auto padded = [](int rows) { return (rows + T::GROUP_ALIGN - 1) & ~(T::GROUP_ALIGN - 1); };   // up to 128

    // views
    __shared__ unsigned bits[T::EXPERTS * T::WORDS];
    __shared__ int total[T::EXPERTS];   // tokens of the expert in all blocks
    __shared__ int first[T::EXPERTS];   // row of the expert's first token of this block
    auto s_bits = opus::make_smem(bits);
    auto s_total = opus::make_smem(total);
    auto s_first = opus::make_smem(first);
    auto words = reinterpret_cast<OPUS_LDS_ADDR unsigned*>(s_bits.ptr);   // the bitmap storage, for the atomic
    auto g_counts = opus::make_gmem(k.counts);
    auto g_sorted = opus::make_gmem(k.sorted_token_ids);
    auto g_layout = opus::make_gmem(k.grouped_layout);

    // layouts: the thread's token; thread i -> expert i (its counts row, its total / first slot)
    const auto u_token = make_layout_global_token<T>(wave_id, lane_id);
    const auto u_i_count = make_layout_global_count<T>(tid, block_stride);
    const auto u_i_expert = make_layout_lds_expert<T>(tid);

    // 5 as 1 and 2 (LDS does not survive launch 1): expert id of the thread's token, bitmap = 0
    const int token = block * T::TOKENS + tid;
    const bool live = token_wg && token < k.tokens;
    int expert = -1;
    if (live) {
        auto g_ids = opus::make_gmem(k.expert_ids + static_cast<size_t>(block) * T::TOKENS);   // the block
        expert = opus::load<1>(g_ids, u_token)[0];
    }
    const bool valid = static_cast<unsigned>(expert) < static_cast<unsigned>(k.experts);
    if (token_wg)
        for (int word = tid; word < k.experts * T::WORDS; word += T::BLOCK_SIZE)
            opus::store<1>(s_bits, 0u, word);

    // 6 thread i has expert i: total[i] = its tokens in all blocks, first[i] = in the blocks before this one
    // (fill workgroup: all blocks). global -> vgpr -> LDS
    if (tid < k.experts) {
        const int before_blocks = token_wg ? static_cast<int>(block) : k.blocks;
        const int row = u_i_count(0);   // counts[tid][0]
        int before = 0, all = 0;
        for (int b0 = 0; b0 < k.blocks; b0 += T::COUNT_BATCH) {   // the loads of COUNT_BATCH blocks go out together
            opus::vector_t<int, 4> count[T::COUNT_BATCH / 4];     // 4 blocks per load
            for (int q = 0; q < T::COUNT_BATCH / 4; q++)
                count[q] = opus::load<4>(g_counts, row + b0 + 4 * q);
            for (int j = 0; j < T::COUNT_BATCH; j++) {
                const int c = b0 + j < k.blocks ? count[j / 4][j % 4] : 0;   // past the blocks: not written
                all += c;
                before += b0 + j < before_blocks ? c : 0;
            }
        }
        opus::store<1>(s_total, opus::make_vector(all), u_i_expert);
        opus::store<1>(s_first, opus::make_vector(before), u_i_expert);
    }
    lds_barrier();

    // as 3: bit (expert, token) = 1
    if (valid) {
        const auto u_bits = make_layout_lds_expert_bitmap<T>(expert);   // the token's expert row
        __hip_atomic_fetch_or(words + u_bits(0) + tid / T::WORD_BITS, 1u << (tid % T::WORD_BITS),
                              __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WORKGROUP);
    }

    // 7 first[i] += first row of expert i's group = the padded totals of the experts before i; used = the rows
    // in use = the padded totals of all experts. LDS -> vgpr -> LDS
    int start = 0, used = 0;
    for (int e0 = 0; e0 < k.experts; e0 += 4) {   // 4 experts per LDS load
        const auto total4 = opus::load<4>(s_total, e0);
        for (int j = 0; j < 4; j++) {
            const int rows = e0 + j < k.experts ? padded(total4[j]) : 0;
            used += rows;
            start += e0 + j < tid ? rows : 0;
        }
    }
    if (tid < k.experts)
        opus::store<1>(s_first, opus::make_vector(opus::load<1>(s_first, u_i_expert)[0] + start), u_i_expert);
    if (block == 0 && tid == 0) {   // num_valid_ids: one thread of the launch
        auto g_valid = opus::make_gmem(k.num_valid_ids);
        opus::store<1>(g_valid, used, 0);
    }
    lds_barrier();

    // 8 token workgroup: row of the token = first[expert] + the tokens of the block with the expert before it
    // (the 1 bits below its own). Writes reverse_sorted[token], sorted_token_ids[row], grouped_layout[row].
    // LDS -> vgpr -> global
    if (token_wg) {
        if (!live)
            return;
        const int e = valid ? expert : 0;
        const auto u_e_bits = make_layout_lds_expert_bitmap<T>(e);   // the token's expert row
        const auto u_e_expert = make_layout_lds_expert<T>(e);        // first[the token's expert]
        const auto row_bits = opus::load<T::WORDS>(s_bits, u_e_bits);
        int rank = 0;
        for (int w = 0; w < T::WORDS; w++) {
            const int my_word = tid / T::WORD_BITS;
            const unsigned below = w < my_word ? ~0u : w == my_word ? (1u << (tid % T::WORD_BITS)) - 1u : 0u;
            rank += __builtin_popcount(row_bits[w] & below);
        }
        const int row = valid ? opus::load<1>(s_first, u_e_expert)[0] + rank : -1;
        auto g_rows = opus::make_gmem(k.reverse_sorted + static_cast<size_t>(block) * T::TOKENS);   // the block
        opus::store<1>(g_rows, opus::make_vector(row), u_token);
        if (valid) {
            opus::store<1>(g_sorted, token, row);
            opus::store<1>(g_layout, expert, row);
        }
        return;
    }

    // 9 fill workgroup `fill`, FILL_THREADS (32) threads per expert: group_rows[e], -1 in the expert's padding
    // rows; then -1 in a piece of the rows past the last group. LDS -> vgpr -> global
    const int fill = block - k.blocks;
    const int e = fill * T::FILL_EXPERTS + tid / T::FILL_THREADS, member = tid % T::FILL_THREADS;
    if (e < k.experts) {
        const auto u_e_expert = make_layout_lds_expert<T>(e);
        const int count = opus::load<1>(s_total, u_e_expert)[0];
        const int pad = opus::load<1>(s_first, u_e_expert)[0];   // behind the expert's last token
        if (member == 0) {
            auto g_group_rows = opus::make_gmem(k.group_rows);
            opus::store<1>(g_group_rows, count, e);
        }
        for (int i = member; i < padded(count) - count; i += T::FILL_THREADS) {
            opus::store<1>(g_sorted, -1, pad + i);
            opus::store<1>(g_layout, -1, pad + i);
        }
    }
    for (int r = used + fill * T::FILL_TAIL_ROWS + tid; r < used + (fill + 1) * T::FILL_TAIL_ROWS && r < k.rows; r += T::BLOCK_SIZE) {
        opus::store<1>(g_sorted, -1, r);
        opus::store<1>(g_layout, -1, r);
    }
}

__global__ void __launch_bounds__(workgroup_traits::BLOCK_SIZE) moe_sort_rows(opus_moe_sort_kargs k) {
    sort_rows_body(k);
}

} // namespace opus_moe_token_reorder
