// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// OPUS-based paged-attention decode for gfx950.
//
// Algorithm follows the sp3 kernel PA_A16W16_*_1TG_4W_16mx1_64nx4: one thread
// group of 4 waves per (batch, kv-head), 16 query rows, waves split along the
// KV axis for Q*K and along the head axis for P*V.
//
// Single header: the public API is always visible, host plumbing and the device
// kernel sit behind PA_DECODE_OPUS_IMPL, and the host pass gets an empty stub so
// the `__device_stub__` symbols still resolve.

#pragma once
#include "aiter_tensor.h"

// Public API: paged-attention decode over a block table.
//
// Tensor expectations (row-major, last dim contiguous):
//   q            : [batch, num_heads, 128]                          bf16
//   k_cache      : [num_blocks, num_kv_heads, 128/8, 16, 8]         bf16 (vLLM packing, x=8)
//   v_cache      : [num_blocks, num_kv_heads, 128, 16]              bf16
//   block_tables : [batch, max_blocks_per_batch_row]                int32
//   context_lens : [batch]                                          int32
//   out          : [batch, num_heads, 128]                          bf16 (caller-allocated)
//
// `softmax_scale` is forwarded as-is (no implicit 1/sqrt(D)).
// Requires head_dim == 128, page size == 16, and num_heads/num_kv_heads <= 16.
void pa_decode_opus_fwd(aiter_tensor_t& q,
                        aiter_tensor_t& k_cache,
                        aiter_tensor_t& v_cache,
                        aiter_tensor_t& block_tables,
                        aiter_tensor_t& context_lens,
                        aiter_tensor_t& out,
                        float softmax_scale);

#ifdef PA_DECODE_OPUS_IMPL
// ============================================================================
// Implementation section - only compiled in the .cu translation unit
// ============================================================================

using bf16_t = __bf16;

// KV tile depth: how many KV tokens one main-loop iteration consumes.
//
// K and V go straight from the cache into the matrix-core fragments, so the tile
// is paid for in registers, not LDS, and occupancy tracks it directly. On gfx950
// (VGPR+AGPR, waves/SIMD): 64 -> 89+8, 4;  128 -> 129+8, 3;  256 -> 217+16, 2;
// 512 -> 256+161, 1; LDS stays under 17KB throughout and never binds. HBM
// streaming is flat over 64..256 (two waves already saturate it) while a
// cache-resident working set follows occupancy, so 128 is the compromise:
// fastest of the four streaming, within 2.5% of the best cache-resident.
// Re-measure with op_tests/sweep_kv_tile.sh and op_tests/kv_tile_resources.sh.
#ifndef PA_DECODE_KV_TILE
#define PA_DECODE_KV_TILE 128
#endif

// Kernel arguments.
struct pa_decode_kargs
{
    const void* __restrict__ q_ptr;       // [batch, num_heads, D]
    const void* __restrict__ k_ptr;       // [num_blocks, num_kv_heads, D/K_PACK, PAGE, K_PACK]
    const void* __restrict__ v_ptr;       // [num_blocks, num_kv_heads, D, PAGE]
    void* __restrict__ out_ptr;           // [batch, num_heads, D]
    const int* __restrict__ block_tables; // [batch, max_blocks_per_batch_row]
    const int* __restrict__ context_lens; // [batch]
    // Split-KV scratch, only used when num_splits > 1.
    float* __restrict__ partial_o;  // [batch][num_kv_heads][num_splits][Q_TILE][D]
    float* __restrict__ partial_ml; // [batch][num_kv_heads][num_splits][2][Q_TILE]
    // One arrival counter per (batch, kv-head). Zeroed at allocation, and left
    // zeroed by whichever split arrives last, so no per-call memset is needed.
    unsigned int* __restrict__ split_counters;
    int num_splits;
    int batch;
    int num_heads;
    int num_kv_heads;
    int gqa_ratio; // num_heads / num_kv_heads
    int max_blocks_per_batch_row;
    int stride_q_b; // elements, one batch row of q ([batch, num_heads, D])
    int stride_q_h; // elements, one q head
    int stride_o_b;
    int stride_o_h;
    int stride_k_blk; // elements, one page of all kv heads
    int stride_k_h;   // elements, one kv head inside a page
    int stride_v_blk;
    int stride_v_h;
    float softmax_scale;
};

// Tile shape / MFMA configuration.
//
// A workgroup owns one (batch, kv-head), or one split of its context, and walks
// that range one KV_TILE at a time. Per tile, with M the query rows throughout:
//
//   S [Q_TILE, KV_TILE] = Q [Q_TILE, D_HEAD] * K^T    GEMM0, contracts D_HEAD
//   P [Q_TILE, KV_TILE] = softmax(S)                  also rescales the running O
//   O [Q_TILE, D_HEAD] += P * V^T                     GEMM1, contracts KV_TILE
//
// Q is loaded once and O accumulates across tiles, so only K and V stream. Q_TILE
// is the MFMA's M and also the largest GQA ratio supported, so one tile carries
// every query head that shares this kv-head.
//
// The `16mx1_16nx4` namespace names GEMM0's wave tiling: `16mx1` = W_M=16 query
// rows by T_M=1 wave, the whole Q_TILE; `16nx4` = W_N=16 tokens by T_N=4 waves,
// one 64-token step along KV repeated GEMM0_E_N = KV_TILE/(W_N*T_N) times. The
// name is thus independent of KV_TILE, unlike sp3's `64nx4`, which counts a
// wave's KV columns.
template<int D_HEAD_ = 128, int KV_TILE_ = 128, int PAGE_SIZE_ = 16, int NUM_WARPS_ = 4>
struct pa_decode_traits
{
    using D_ATTN = bf16_t;
    using D_ACC  = float;

    static constexpr int D_HEAD    = D_HEAD_;
    static constexpr int KV_TILE   = KV_TILE_;
    static constexpr int PAGE_SIZE = PAGE_SIZE_;
    static constexpr int NUM_WARPS = NUM_WARPS_;
    static constexpr int WARP_SIZE = 64;
    static constexpr int BLOCK_SIZE = NUM_WARPS * WARP_SIZE;
    static constexpr int Q_TILE     = 16; // MFMA M, also the max supported GQA ratio

    // Upper bound on KV splits: num_splits is a runtime value, but the fused
    // merge indexes LDS by split, so this statically sizes that buffer. Raising
    // it past the point where smem_reduce_bytes() overtakes the attention side
    // grows LDS for every shape, including the ones that never split.
    static constexpr int MAX_SPLITS = 64;

    // Wave tiling: 1 wave along M, all waves along N.
    static constexpr int T_M = 1;
    static constexpr int T_N = NUM_WARPS;
    static constexpr int T_K = 1;

    // gfx950 native bf16 matrix core.
    static constexpr int W_M = 16;
    static constexpr int W_N = 16;
    static constexpr int W_K = 32;

    static constexpr int ELEM_A = W_M * W_K / WARP_SIZE; // 8
    static constexpr int ELEM_B = W_N * W_K / WARP_SIZE; // 8
    static constexpr int ELEM_C = W_M * W_N / WARP_SIZE; // 4, consecutive along N

    // GEMM0: S[Q_TILE, KV_TILE] = Q[Q_TILE, D_HEAD] * K^T
    static constexpr int GEMM0_E_M = Q_TILE / (W_M * T_M);
    static constexpr int GEMM0_E_N = KV_TILE / (W_N * T_N);
    static constexpr int GEMM0_E_K = D_HEAD / (W_K * T_K);

    // GEMM1: O[Q_TILE, D_HEAD] = P[Q_TILE, KV_TILE] * V^T
    static constexpr int GEMM1_E_M = Q_TILE / (W_M * T_M);
    static constexpr int GEMM1_E_N = D_HEAD / (W_N * T_N);
    static constexpr int GEMM1_E_K = KV_TILE / (W_K * T_K);

    static constexpr int PAGES_PER_TILE = KV_TILE / PAGE_SIZE;
    // vLLM packs the K cache as [D/x, PAGE, x] with x = 16 bytes / sizeof(dtype).
    static constexpr int K_PACK = 16 / sizeof(D_ATTN);
    static constexpr int VEC    = 16 / sizeof(D_ATTN); // widest global vector, 8 bf16

    // Lane counts, not thread groups: GRP is opus's name for the dims of an MFMA
    // operand spread across lanes (the <p> dims of
    // `B:[(grpn_b<p>), (rept_b, grpk_b<p>, pack_b)]` in opus.hpp). These restate
    // mfma_adaptor::grpn_b/::grpk_b, unreachable here because the static_asserts
    // below run before any mma object exists. The 64 lanes form a GRP_K by GRP_N
    // grid: lane % GRP_N picks the N position, lane / GRP_N the contraction
    // slice, VEC elements each, so the slices span W_K.
    static constexpr int GRP_N = W_N;             // == mfma_adaptor::grpn_b
    static constexpr int GRP_K = WARP_SIZE / W_N; // == mfma_adaptor::grpk_b
    // How many lane groups share one page when walking V along tokens.
    static constexpr int KGRP_PER_PAGE = PAGE_SIZE / VEC;

    // Only P is staged through LDS: K and V go straight from global into the
    // matrix-core fragments and S never leaves registers, so LDS is needed only
    // where a wave reads what another wave produced -- for P that is every
    // element, since GEMM1 contracts the token dim that splits the waves. The
    // tile is [Q_TILE][KV_TILE]; PAD staggers rows off each other's banks.
    static constexpr int PAD          = 8;
    static constexpr int P_ROW_STRIDE = KV_TILE + PAD;

    static constexpr int smem_p_elems = Q_TILE * P_ROW_STRIDE;
    // Cross-wave row reduction scratch, laid out [row][wave] so a row's T_N
    // contributions fold with one 16B LDS read. Used twice at the same
    // addresses: each tile's max fold, then the final l sum.
    static constexpr int smem_row_reduce_elems = Q_TILE * T_N;

    static_assert(Q_TILE == W_M, "one MFMA tile along M");
    static_assert(KV_TILE % (W_N * T_N) == 0);
    static_assert(KV_TILE % PAGE_SIZE == 0);
    static_assert(D_HEAD % (W_N * T_N) == 0);
    static_assert(KV_TILE % W_K == 0);
    static_assert(D_HEAD % W_K == 0);
    // Softmax runs on the GEMM0 C fragment, where a query row is spread over the
    // lanes {r, r+16, r+32, r+48} of each wave. That is exactly the reach of a
    // permlane32/permlane16 swap pair, which is what lets S stay in registers.
    static_assert(W_M == 16, "a C fragment row must span four 16-lane groups");
    static_assert(WARP_SIZE == 64);

    // Contract that lets the KV fragments be read straight from the cache:
    // one K MFMA tile must cover exactly one page, and one lane's slice of the
    // contraction dim must be exactly the 16B the cache stores contiguously.
    static_assert(GRP_N == PAGE_SIZE, "a K MFMA tile must line up with one page");
    static_assert(GRP_K * VEC == W_K, "a lane's K slice must be one 16B vector");
    static_assert(K_PACK == VEC);
    static_assert(PAGE_SIZE % VEC == 0);
    static_assert(ELEM_B == VEC, "one B fragment is one 16B global vector");
    static_assert(GEMM0_E_N * T_N == PAGES_PER_TILE, "K's N tiles must cover the tile's pages");
    static_assert(GEMM1_E_K * (GRP_K / KGRP_PER_PAGE) == PAGES_PER_TILE,
                  "V's K steps must walk exactly the tile's pages");

    // The split-KV merge reuses this allocation -- the P tile and the row-reduce
    // scratch are both dead by the time it runs -- so the union costs it no LDS.
    // At KV_TILE=128 the attention side is the larger.
    static constexpr size_t smem_reduce_bytes()
    {
        return static_cast<size_t>(MAX_SPLITS * Q_TILE + Q_TILE) * sizeof(D_ACC);
    }

    static constexpr size_t smem_size_bytes()
    {
        const size_t attn = smem_p_elems * sizeof(D_ATTN) // bf16 P handed to GEMM1
                            + smem_row_reduce_elems * sizeof(D_ACC); // cross-wave row fold
        const size_t reduce = smem_reduce_bytes();
        return attn > reduce ? attn : reduce;
    }
};

using pa_decode_traits_d128 = pa_decode_traits<128, PA_DECODE_KV_TILE, 16, 4>;

template<class Traits>
__global__ void pa_decode_opus_kernel(pa_decode_kargs kargs);

#ifdef __HIP_DEVICE_COMPILE__
// ---------------------------------------------------------------------------
// Device pass
// ---------------------------------------------------------------------------
#include "opus/opus.hpp"

#if defined(__gfx950__)

namespace pa_decode_16mx1_16nx4 {

using opus::operator""_I;

// Scratch addressing for the split-KV path: one (Q_TILE x D) accumulator plus a
// (m, l) pair per query row, for every (batch, kv-head, split).
template<class T>
__device__ inline int64_t partial_slot(const pa_decode_kargs& kargs, int b, int kvh, int split)
{
    return (static_cast<int64_t>(b) * kargs.num_kv_heads + kvh) * kargs.num_splits + split;
}

template<class T>
__device__ inline float* partial_ml_ptr(const pa_decode_kargs& kargs, int b, int kvh, int split)
{
    return kargs.partial_ml + partial_slot<T>(kargs, b, kvh, split) * 2 * T::Q_TILE;
}

template<class T>
__device__ inline float* partial_o_ptr(const pa_decode_kargs& kargs, int b, int kvh, int split)
{
    return kargs.partial_o + partial_slot<T>(kargs, b, kvh, split) * T::Q_TILE * T::D_HEAD;
}

// Where this lane's K and V fragments sit inside a page. Only the page base
// moves per tile, so this is computed once.
//
// The cache layouts hand the matrix cores exactly what they want: a B fragment
// needs lane l to hold column l%16 and 8 consecutive contraction positions, and
// both caches store those 8 values in 16 contiguous bytes (K is
// [D/K_PACK][PAGE][K_PACK], a lane walks one token's head dims; V is [D][PAGE],
// a lane walks 8 tokens of one head dim). Every fragment is one dwordx4 straight
// from the cache, and the KV tile never goes through LDS. K lines one MFMA tile
// up with one page (GRP_N == PAGE_SIZE) so its page index is wave-uniform; V
// spans PAGE_SIZE/VEC lane groups per page, so its index alternates on the upper
// half of the wave.
template<class T>
struct kv_frag_slice
{
    int k_off;      // element offset of the K fragment inside its page
    int v_off;      // element offset of the V fragment inside its page
    int v_page_odd; // 0/1: which page of the pair this lane reads V from

    __device__ kv_frag_slice(int lane_id, int warp_id)
    {
        const int n_lane = lane_id % T::GRP_N; // token for K, head dim for V
        const int k_grp  = lane_id / T::GRP_N; // slice of the contraction dim

        k_off = k_grp * T::PAGE_SIZE * T::K_PACK + n_lane * T::K_PACK;

        v_off = (warp_id * T::GRP_N + n_lane) * T::PAGE_SIZE
                + (k_grp % T::KGRP_PER_PAGE) * T::VEC;
        v_page_odd = k_grp / T::KGRP_PER_PAGE;
    }
};

// Page indices for one KV tile, kept in registers a full iteration ahead of the
// KV loads that consume them.
template<class T>
struct page_ids_t
{
    int id[T::PAGES_PER_TILE];
};

// Read one tile's worth of block-table entries.
//
// The buffer's num_records stops at this split's last page, so a slot past the
// end reads 0 with no bounds check in the instruction stream. Page 0 is a valid
// index, but the tokens it stands in for sit past the context length, so the
// tail tile masks their scores anyway. The buffer (rather than a raw pointer) is
// what keeps the reads wide: an explicit guard -- branch or clamp -- makes every
// slot its own expression, costing a dword load plus a 64-bit address chain per
// page against two dwordx4 for the whole tile here.
//
// Hoisting the wave-uniform ids into SGPRs with readfirstlane measured as a wash:
// readfirstlane has to wait on the load, which anchors the block-table round trip
// in front of GEMM0 instead of letting the address math sink in among the MFMAs.
template<class T, class G>
__device__ inline page_ids_t<T> load_page_ids(G& g_bt, int tile_idx)
{
    static_assert(T::PAGES_PER_TILE % 4 == 0, "block-table reads are dwordx4");
    page_ids_t<T> pid;
    const int base = tile_idx * T::PAGES_PER_TILE * static_cast<int>(sizeof(int));
    opus::static_for<T::PAGES_PER_TILE / 4>([&](auto ig) {
        const auto v = g_bt.template _load<4>(base + ig.value * 16);
        opus::static_for<4>([&](auto j) { pid.id[ig.value * 4 + j.value] = v[j.value]; });
    });
    return pid;
}

// Page indices for the K fragments of one wave. A K MFMA tile is exactly one
// page, so wave w only ever reads pages {i_n * T_N + w}. Reading just those
// keeps the array indices compile-time constant; indexing the whole-tile array
// by a runtime wave id becomes an s_cselect chain or spills to scratch.
template<class T>
struct k_page_ids_t
{
    int id[T::GEMM0_E_N];
};

template<class T, class G>
__device__ inline k_page_ids_t<T> load_k_page_ids(G& g_bt, int tile_idx, int warp_id)
{
    k_page_ids_t<T> pid;
    const int base = (tile_idx * T::PAGES_PER_TILE + warp_id) * static_cast<int>(sizeof(int));
    opus::static_for<T::GEMM0_E_N>([&](auto in) {
        const auto v     = g_bt.template _load<1>(base + in.value * T::T_N * static_cast<int>(sizeof(int)));
        pid.id[in.value] = v[0];
    });
    return pid;
}

// Read one tile's K fragments straight from the cache into the matrix-core
// operand. Deliberately does not wait on the results: the caller issues these
// right after the GEMM that consumed the previous tile, so the latency hides
// behind the rest of the loop body.
//
// Addressed by pointer, not buffer: a buffer offset is 32-bit and would cap the
// cache at 4GiB, and no scalar base can lift that since V's page index varies
// per lane (the halves of a wave read adjacent pages). Buffer addressing measured
// identical where it fits -- the widening multiply and 64-bit add per page
// disappear into the memory latency.
template<class T, class VB>
__device__ inline void load_k_frags(const typename T::D_ATTN* __restrict__ p_k,
                                    const k_page_ids_t<T>& pid,
                                    int stride_k_blk,
                                    const kv_frag_slice<T>& s,
                                    VB& v_k)
{
    using D_ATTN      = typename T::D_ATTN;
    using vec_t       = opus::vector_t<D_ATTN, T::VEC>;
    constexpr int E_N = T::GEMM0_E_N;
    constexpr int E_K = T::GEMM0_E_K;

    opus::static_for<E_N>([&](auto in) {
        constexpr int i_n = in.value;
        const D_ATTN* g_n = p_k + static_cast<int64_t>(pid.id[i_n]) * stride_k_blk + s.k_off;

        opus::static_for<E_K>([&](auto ik) {
            constexpr int i_k   = ik.value;
            constexpr int d_off = i_k * T::GRP_K * T::PAGE_SIZE * T::K_PACK;
            const vec_t frag    = *reinterpret_cast<const vec_t*>(g_n + d_off);
            opus::static_for<T::ELEM_B>(
                [&](auto j) { v_k[(i_n * E_K + i_k) * T::ELEM_B + j.value] = frag[j.value]; });
        });
    });
}

// Same for V. Here the contraction dim is tokens, so a K step walks pages while
// N walks head dims. Each step of GRP_K lane groups spans GRP_K/KGRP_PER_PAGE
// pages, and the upper lane groups sit on the odd page of that span.
template<class T, class VB>
__device__ inline void load_v_frags(const typename T::D_ATTN* __restrict__ p_v,
                                    const page_ids_t<T>& pid,
                                    int stride_v_blk,
                                    const kv_frag_slice<T>& s,
                                    VB& v_v)
{
    using D_ATTN      = typename T::D_ATTN;
    using vec_t       = opus::vector_t<D_ATTN, T::VEC>;
    constexpr int E_N = T::GEMM1_E_N;
    constexpr int E_K = T::GEMM1_E_K;

    opus::static_for<E_K>([&](auto ik) {
        constexpr int i_k       = ik.value;
        constexpr int page_base = i_k * (T::GRP_K / T::KGRP_PER_PAGE);
        // Both indices stay compile time; the runtime part is a lane select.
        const int page    = s.v_page_odd ? pid.id[page_base + 1] : pid.id[page_base];
        const D_ATTN* g_k = p_v + static_cast<int64_t>(page) * stride_v_blk + s.v_off;

        opus::static_for<E_N>([&](auto in) {
            constexpr int i_n   = in.value;
            constexpr int d_off = i_n * T::T_N * T::GRP_N * T::PAGE_SIZE;
            const vec_t frag    = *reinterpret_cast<const vec_t*>(g_k + d_off);
            opus::static_for<T::ELEM_B>(
                [&](auto j) { v_v[(i_n * E_K + i_k) * T::ELEM_B + j.value] = frag[j.value]; });
        });
    });
}

// All-reduce a query row inside one wave.
//
// After swap_ab the C fragment puts query row r on lanes {r, r+16, r+32, r+48},
// so folding a row is a pair swap at distance 32 then one at distance 16 -- two
// gfx950 VALU ops, no LDS and no barrier, which is what lets the score tile stay
// in registers. A lane covers only a quarter of a row, so every row-wide
// quantity has to pass through here to mean anything.
//
// Must be inline asm, not __builtin_amdgcn_permlane*_swap: the intrinsic returns
// both halves as one vector, and when both operands hold the same value -- how an
// all-reduce uses it -- the compiler folds them into one register. The second
// half is lost and the fold collapses to op(v, v): invisible for max, badly
// wrong for a sum.
//
// Inline asm also costs us the hazard recognizer, so the wait states are supplied
// here. Per LLVM's GCNHazardRecognizer for gfx950 a VALU write to an operand
// needs 2 wait states before the swap reads it and a v_cmpx writing exec needs 4;
// s_nop 3 covers both. Reading the result afterwards needs none.
#define PA_SWAP_HAZARD "s_nop 3\n\t"

__device__ inline void permlane32_swap(float& a, float& b)
{
    asm volatile(PA_SWAP_HAZARD "v_permlane32_swap_b32 %0, %1" : "+v"(a), "+v"(b));
}

__device__ inline void permlane16_swap(float& a, float& b)
{
    asm volatile(PA_SWAP_HAZARD "v_permlane16_swap_b32 %0, %1" : "+v"(a), "+v"(b));
}

template<class OP>
__device__ inline float wave_row_fold(float v, OP op)
{
    float a = v, b = v;
    permlane32_swap(a, b);
    a = op(a, b);
    b = a;
    permlane16_swap(a, b);
    return op(a, b);
}

// Publish this wave's row value and fold the T_N of them, through LDS. Each wave
// holds a partial for every query row, so the exchange is T_N contiguous floats
// per row -- one 16B read. This barrier is the only one the whole softmax needs.
//
// s_row_reduce is deliberately not __restrict__, and the barrier is the fenced
// __syncthreads rather than a bare s_barrier: every wave writes and reads this
// scratch, twice per kernel at the same addresses, and a plain s_barrier only
// orders execution -- the compiler would be free to carry the first fold's
// loaded values across it and reuse them for the second.
template<class T, class OP>
__device__ inline typename T::D_ACC
fold_across_waves(typename T::D_ACC* s_row_reduce,
                  typename T::D_ACC v,
                  int q_row,
                  int warp_id,
                  OP op)
{
    s_row_reduce[q_row * T::T_N + warp_id] = v;
    __syncthreads();

    typename T::D_ACC acc = s_row_reduce[q_row * T::T_N];
    opus::static_for<T::T_N - 1>(
        [&](auto iw) { acc = op(acc, s_row_reduce[q_row * T::T_N + iw.value + 1]); });
    return acc;
}

// Masked online softmax over the GEMM0 C fragment, in registers. Returns the
// rescale factor for the caller's O accumulator.
//
// S never reaches LDS: lane l of wave w owns query row l % W_M and, per N
// fragment, ELEM_C consecutive tokens, so folding a row is a permlane pair
// inside the wave plus one float between waves -- against a full
// [Q_TILE][KV_TILE] fp32 round trip if the tile went through LDS.
//
// m_run must stay identical across waves, since the P they all write feeds one
// GEMM and needs one common scale. l_run needs no such agreement, so it stays an
// unfolded per-lane partial, reduced once after the last tile.
//
// MASKED is false for every tile but the last, the only one that can run past the
// context. The check is 3 VALU per C fragment element, worth specialising rather
// than covering with a uniform branch.
template<class T, bool MASKED, class VC>
__device__ inline typename T::D_ACC
online_softmax_frag(VC& v_s,
                    typename T::D_ACC* __restrict__ s_row_reduce,
                    typename T::D_ACC& m_run,
                    typename T::D_ACC& l_run,
                    typename T::D_ACC scale,
                    int valid_kv,
                    int tile_idx,
                    int lane_id,
                    int warp_id)
{
    using D_ACC       = typename T::D_ACC;
    constexpr int E_N = T::GEMM0_E_N;
    constexpr int E_C = T::ELEM_C;

    const int q_row    = lane_id % T::W_M;
    const int tok_lane = (lane_id / T::W_M) * E_C;

    D_ACC local_max = opus::numeric_limits<D_ACC>::lowest();
    opus::static_for<E_N>([&](auto in) {
        constexpr int i_n = in.value;
        const int kv_base = tile_idx * T::KV_TILE + (i_n * T::T_N + warp_id) * T::W_N + tok_lane;
        opus::static_for<E_C>([&](auto ic) {
            constexpr int c = ic.value;
            D_ACC s         = v_s[i_n * E_C + c] * scale;
            if constexpr(MASKED)
                s = (kv_base + c) < valid_kv ? s : opus::numeric_limits<D_ACC>::lowest();
            v_s[i_n * E_C + c] = s;
            local_max          = s > local_max ? s : local_max;
        });
    });

    const D_ACC tile_max =
        fold_across_waves<T>(s_row_reduce,
                             wave_row_fold(local_max, [](D_ACC a, D_ACC b) { return a > b ? a : b; }),
                             q_row,
                             warp_id,
                             [](D_ACC a, D_ACC b) { return a > b ? a : b; });

    const D_ACC m_prev  = m_run;
    m_run               = tile_max > m_prev ? tile_max : m_prev;
    const D_ACC rescale = __builtin_amdgcn_exp2f(m_prev - m_run);

    D_ACC local_sum = 0.0f;
    opus::static_for<E_N * E_C>([&](auto i) {
        // exp2 of lowest() underflows to 0, which is exactly the mask we want.
        const D_ACC e = __builtin_amdgcn_exp2f(v_s[i.value] - m_run);
        v_s[i.value]  = e;
        local_sum += e;
    });
    l_run = l_run * rescale + local_sum;

    return rescale;
}

// Announce that this split is done, and report whether it was the last one.
//
// The merge is fused rather than a second kernel: that launch cost 4-5us of
// almost pure dispatch, more than the split-KV path was buying back. Nothing
// ever waits on the counter -- it only tells the last arrival, which already
// has the machine, that every partial is in memory.
template<class T>
__device__ inline bool split_arrive_is_last(const pa_decode_kargs& kargs, int b, int kvh, int tid)
{
    __shared__ int s_last;

    // Release for the whole block: the barrier orders every thread's partial
    // stores ahead of the counter bump, the waitcnt makes sure they retired.
    // The textbook __threadfence() cost 28us of a 41us kernel -- on a multi-XCD
    // part an agent-scope release writes back all of L2, evicting the KV stream.
    // Uncached scratch removes the need: the stores retire at the coherence point.
    __syncthreads();
    __builtin_amdgcn_s_waitcnt(0);

    if(tid == 0)
    {
        unsigned int* slot = kargs.split_counters + b * kargs.num_kv_heads + kvh;
        const unsigned int prev = atomicAdd(slot, 1u);
        const bool last = prev + 1u == static_cast<unsigned int>(kargs.num_splits);
        // The buffer is zeroed once at allocation and never again, so the last
        // arrival is what leaves it clean for the next launch.
        if(last) atomicExch(slot, 0u);
        s_last = last ? 1 : 0;
    }
    __syncthreads();
    return s_last != 0;
}

// Merge this (batch, kv-head)'s per-split partials into the final output.
template<class T>
__device__ inline void reduce_splits(const pa_decode_kargs& kargs, int b, int kvh, int tid,
                                     char* smem)
{
    using D_ATTN = typename T::D_ATTN;
    using D_ACC  = typename T::D_ACC;

    const int splits  = kargs.num_splits;
    const D_ACC* p_ml = partial_ml_ptr<T>(kargs, b, kvh, 0);
    const D_ACC* p_o  = partial_o_ptr<T>(kargs, b, kvh, 0);

    // Carved out of the attention body's buffer, which is dead by now.
    auto* s_scale = reinterpret_cast<D_ACC*>(smem);
    auto* s_inv_l = s_scale + T::MAX_SPLITS * T::Q_TILE;

    if(tid < T::Q_TILE)
    {
        D_ACC m_max = -opus::numeric_limits<D_ACC>::max();
        for(int i = 0; i < splits; ++i)
        {
            const D_ACC m = p_ml[static_cast<int64_t>(i) * 2 * T::Q_TILE + tid];
            if(m > m_max) m_max = m;
        }

        // The softmax runs in log2 space, so the cross-split rescale is exp2 too.
        D_ACC l_sum = 0.0f;
        for(int i = 0; i < splits; ++i)
        {
            const int64_t base = static_cast<int64_t>(i) * 2 * T::Q_TILE;
            const D_ACC c      = __builtin_amdgcn_exp2f(p_ml[base + tid] - m_max);
            s_scale[i * T::Q_TILE + tid] = c;
            l_sum += p_ml[base + T::Q_TILE + tid] * c;
        }
        s_inv_l[tid] = l_sum > D_ACC(0.0f) ? D_ACC(1.0f) / l_sum : D_ACC(0.0f);
    }
    __syncthreads();

    auto* out = reinterpret_cast<D_ATTN*>(kargs.out_ptr)
                + static_cast<int64_t>(b) * kargs.stride_o_b
                + static_cast<int64_t>(kvh) * kargs.gqa_ratio * kargs.stride_o_h;

    // Four dims per thread, over only the rows the GQA group has: the partials are
    // contiguous along the head dim, and bounding by gqa_ratio keeps gqa<16 from
    // spending iterations on rows it would then discard.
    constexpr int VEC      = 4;
    constexpr int ROW_VECS = T::D_HEAD / VEC;
    const int vec_total    = kargs.gqa_ratio * ROW_VECS;

    for(int idx = tid; idx < vec_total; idx += T::BLOCK_SIZE)
    {
        const int row       = idx / ROW_VECS;
        const int dim       = (idx - row * ROW_VECS) * VEC;
        const int64_t o_row = static_cast<int64_t>(row) * T::D_HEAD + dim;

        D_ACC acc[VEC] = {0.0f, 0.0f, 0.0f, 0.0f};
        for(int i = 0; i < splits; ++i)
        {
            const D_ACC c = s_scale[i * T::Q_TILE + row];
            const auto v  = *reinterpret_cast<const opus::vector_t<D_ACC, VEC>*>(
                p_o + static_cast<int64_t>(i) * T::Q_TILE * T::D_HEAD + o_row);
            opus::static_for<VEC>([&](auto j) { acc[j.value] += v[j.value] * c; });
        }

        // Adjacent by construction, so these fold into one wide store.
        const D_ACC inv_l = s_inv_l[row];
        auto* dst = out + static_cast<int64_t>(row) * kargs.stride_o_h + dim;
        opus::static_for<VEC>(
            [&](auto j) { dst[j.value] = static_cast<D_ATTN>(acc[j.value] * inv_l); });
    }
}

} // namespace pa_decode_16mx1_16nx4

template<class Traits>
__global__ __launch_bounds__(Traits::BLOCK_SIZE, 1) void pa_decode_opus_kernel(pa_decode_kargs kargs)
{
    using namespace opus;
    using namespace pa_decode_16mx1_16nx4;
    using T      = opus::remove_cvref_t<Traits>;
    using D_ATTN = typename T::D_ATTN;
    using D_ACC  = typename T::D_ACC;

    const int kv_head_idx = block_id_x();
    const int batch_idx   = block_id_y();
    const int split_idx   = block_id_z();

    const int tid     = thread_id_x();
    const int lane_id = tid % T::WARP_SIZE;
    const int warp_id = __builtin_amdgcn_readfirstlane(tid / T::WARP_SIZE);

    // ── LDS layout: bf16 P | cross-wave reduction scratch ──
    //
    // Everything else stays in registers. P is the one operand that genuinely has
    // to cross lanes: GEMM0 produces it in the C layout, GEMM1 consumes it in the
    // A layout, and it contracts the token dim -- exactly the dim split across
    // the waves. Declared this early because the split-KV merge borrows it too,
    // and an empty split can reach the merge without running the attention body.
    __shared__ __align__(16) char smem[T::smem_size_bytes()];

    // Split-KV: each workgroup owns a contiguous run of whole tiles. Splitting on
    // tile (not token) boundaries keeps kv_start page-aligned, so the block table
    // can simply be re-based and the rest of the kernel stays split-agnostic.
    const int context_len = kargs.context_lens[batch_idx];
    const int tiles_total = (context_len + T::KV_TILE - 1) / T::KV_TILE;
    const int tiles_per_split = (tiles_total + kargs.num_splits - 1) / kargs.num_splits;
    const int tile_begin      = split_idx * tiles_per_split;
    const int tile_end        = min(tile_begin + tiles_per_split, tiles_total);

    if(tile_begin >= tile_end)
    {
        if(kargs.num_splits > 1)
        {
            // Empty split: publish an identity partial so the reduction can ignore it.
            if(tid < T::Q_TILE)
            {
                float* p_ml = partial_ml_ptr<T>(kargs, batch_idx, kv_head_idx, split_idx);
                p_ml[tid]             = -opus::numeric_limits<float>::max();
                p_ml[T::Q_TILE + tid] = 0.0f;
            }
            // Still has to arrive, or the split that does finish last would be
            // waiting on a count that never completes.
            if(split_arrive_is_last<T>(kargs, batch_idx, kv_head_idx, tid))
                reduce_splits<T>(kargs, batch_idx, kv_head_idx, tid, smem);
        }
        return;
    }

    const int kv_start    = tile_begin * T::KV_TILE;
    const int split_len   = min(context_len - kv_start, (tile_end - tile_begin) * T::KV_TILE);
    const int num_pages   = (split_len + T::PAGE_SIZE - 1) / T::PAGE_SIZE;
    const int num_tiles   = tile_end - tile_begin;

    auto* p_smem  = reinterpret_cast<char*>(smem);
    auto* s_p_raw = reinterpret_cast<D_ATTN*>(p_smem);
    p_smem += T::smem_p_elems * sizeof(D_ATTN);
    auto* s_row_reduce = reinterpret_cast<D_ACC*>(p_smem);

    auto s_p = make_smem(s_p_raw);

    // ── Matrix cores. swap_ab keeps the query row on `lane % W_M` for both GEMMs. ──
    auto mma0 = make_tiled_mma<D_ATTN, D_ATTN, D_ACC>(
        seq<T::GEMM0_E_M, T::GEMM0_E_N, T::GEMM0_E_K>{},
        seq<T::T_M, T::T_N, T::T_K>{},
        seq<T::W_M, T::W_N, T::W_K>{},
        mfma_adaptor_swap_ab{});
    auto mma1 = make_tiled_mma<D_ATTN, D_ATTN, D_ACC>(
        seq<T::GEMM1_E_M, T::GEMM1_E_N, T::GEMM1_E_K>{},
        seq<T::T_M, T::T_N, T::T_K>{},
        seq<T::W_M, T::W_N, T::W_K>{},
        mfma_adaptor_swap_ab{});

    // Every operand is a plain row-major matrix, so the register fragment layout
    // comes straight from the mma adaptor instead of being hand-derived.
    const int q_head_base = kv_head_idx * kargs.gqa_ratio;
    auto u_q  = partition_layout_a<T::ELEM_A>(
        mma0, opus::make_tuple(kargs.stride_q_h, 1_I),
        opus::make_tuple(0_I, lane_id % mma0.grpm_a, 0_I, lane_id / mma0.grpm_a));
    // Writes P out of the C fragment: the mapping the score tile used to be stored
    // with, now applied once, to bf16, after softmax has run in registers.
    auto u_p  = partition_layout_c(
        mma0, opus::make_tuple(number<T::P_ROW_STRIDE>{}, 1_I),
        opus::make_tuple(0_I, lane_id % mma0.grpn_c, warp_id, lane_id / mma0.grpn_c));
    auto u_rp = partition_layout_a<T::ELEM_A>(
        mma1, opus::make_tuple(number<T::P_ROW_STRIDE>{}, 1_I),
        opus::make_tuple(0_I, lane_id % mma1.grpm_a, 0_I, lane_id / mma1.grpm_a));
    auto u_o  = partition_layout_c(
        mma1, opus::make_tuple(kargs.stride_o_h, 1_I),
        opus::make_tuple(0_I, lane_id % mma1.grpn_c, warp_id, lane_id / mma1.grpn_c));

    // ── Load Q once. Rows past the GQA group fall outside the buffer and read 0. ──
    const int64_t q_offset = static_cast<int64_t>(batch_idx) * kargs.stride_q_b
                             + static_cast<int64_t>(q_head_base) * kargs.stride_q_h;
    auto g_q = make_gmem(reinterpret_cast<const D_ATTN*>(kargs.q_ptr) + q_offset,
                         kargs.gqa_ratio * kargs.stride_q_h * sizeof(D_ATTN));
    typename decltype(mma0)::vtype_a v_q = load<T::ELEM_A>(g_q, u_q);

    typename decltype(mma0)::vtype_b v_k;
    typename decltype(mma0)::vtype_c v_s;
    typename decltype(mma1)::vtype_a v_p;
    typename decltype(mma1)::vtype_b v_v;
    typename decltype(mma1)::vtype_c v_o;
    clear(v_o);

    constexpr D_ACC LOG2_E = 1.44269504089f;
    const D_ACC temperature_scale = kargs.softmax_scale * LOG2_E;

    // Online softmax state, per lane rather than per LDS row. m is held equal
    // across waves by the per-tile fold; l is this wave's own partial sum.
    D_ACC m_run = opus::numeric_limits<D_ACC>::lowest();
    D_ACC l_run = 0.0f;

    const D_ATTN* p_k = reinterpret_cast<const D_ATTN*>(kargs.k_ptr) + kv_head_idx * kargs.stride_k_h;
    const D_ATTN* p_v = reinterpret_cast<const D_ATTN*>(kargs.v_ptr) + kv_head_idx * kargs.stride_v_h;
    // Rebased on this split's first page, and bounded to its last one so that
    // the tail tile's over-read comes back as 0 instead of needing a guard.
    auto g_bt = make_gmem(kargs.block_tables
                              + static_cast<int64_t>(batch_idx) * kargs.max_blocks_per_batch_row
                              + kv_start / T::PAGE_SIZE,
                          num_pages * static_cast<unsigned int>(sizeof(int)));

    // Single-buffered KV and page ids, both prefetched one tile ahead. Each KV
    // operand is refetched right after the GEMM that consumed it, so its load
    // stays in flight for the rest of the body: K hides behind softmax and GEMM1,
    // V behind the next tile's GEMM0. A fragment dies as its GEMM issues, so
    // deeper buffering only buys occupancy loss -- prefetch distance two
    // (+64 VGPRs) and double-buffered ids (+53, from unrolling by two, not from
    // the 10 the ids need) both drop 3 waves/SIMD to 2 and measured slower.
    //
    // The last tile is peeled out: only it can run past the context, so the
    // interior body needs neither the per-token mask nor a `has_next` guard. The
    // guard cost more -- with the loads on one path only, the compiler cannot know
    // how many are in flight at the loop header, and fell back to vmcnt(0) before
    // the last GEMM0 MFMA, draining V's prefetch.
    const kv_frag_slice<T> kv_slice(lane_id, warp_id);
    k_page_ids_t<T> k_pid = load_k_page_ids<T>(g_bt, 0, warp_id);
    page_ids_t<T>   v_pid = load_page_ids<T>(g_bt, 0);
    load_k_frags<T>(p_k, k_pid, kargs.stride_k_blk, kv_slice, v_k);
    load_v_frags<T>(p_v, v_pid, kargs.stride_v_blk, kv_slice, v_v);

    // MASKED and PREFETCH are compile time, so the tail instantiation shares no
    // code with the loop body and neither branch survives into either.
    auto run_tile = [&](int tile_idx, auto masked, auto prefetch) {
        constexpr bool MASKED   = decltype(masked)::value != 0;
        constexpr bool PREFETCH = decltype(prefetch)::value != 0;

        if constexpr(PREFETCH)
        {
            k_pid = load_k_page_ids<T>(g_bt, tile_idx + 1, warp_id);
            v_pid = load_page_ids<T>(g_bt, tile_idx + 1);
        }
        // Keep the block-table reads ahead of GEMM0 instead of letting the
        // scheduler sink them among the MFMAs: every KV address for the next tile
        // depends on them, and left alone the first address computation lands a
        // dozen instructions behind the load and stalls on it.
        __builtin_amdgcn_sched_barrier(0);

        v_s = mma0(v_q, v_k);
        if constexpr(PREFETCH)
            load_k_frags<T>(p_k, k_pid, kargs.stride_k_blk, kv_slice, v_k);

        // Softmax happens in place on v_s, so this is also the P the store below
        // hands to GEMM1. The one barrier inside is the cross-wave max fold.
        const D_ACC rescale = online_softmax_frag<T, MASKED>(
            v_s, s_row_reduce, m_run, l_run, temperature_scale, split_len, tile_idx,
            lane_id, warp_id);
        opus::static_for<vector_traits<decltype(v_o)>::size()>(
            [&](auto i) { v_o[i.value] *= rescale; });

        auto v_p_out = cast<D_ATTN>(v_s);
        store<4>(s_p, v_p_out, u_p);
        opus::s_waitcnt_lgkmcnt(opus::number<0>{});
        __builtin_amdgcn_s_barrier();

        v_p = load<T::ELEM_A>(s_p, u_rp);
        v_o = mma1(v_p, v_v, v_o);
        if constexpr(PREFETCH)
            load_v_frags<T>(p_v, v_pid, kargs.stride_v_blk, kv_slice, v_v);

        // No trailing barrier: the next tile cannot overwrite s_p or s_row_reduce
        // before its own max fold, by which point every wave has issued its P reads.
    };

    for (int tile_idx = 0; tile_idx + 1 < num_tiles; ++tile_idx)
        run_tile(tile_idx, 0_I, 1_I);
    run_tile(num_tiles - 1, 1_I, 0_I);

    // l is still a per-lane partial: a lane summed only its own quarter of the row
    // and only its own wave's tokens. Both folds can wait until here because
    // summing is linear and a row's rescale is common to all its lanes. Reusing
    // s_row_reduce is safe -- the last tile read it before that tile's P barrier.
    const D_ACC l_final = fold_across_waves<T>(
        s_row_reduce,
        wave_row_fold(l_run, [](D_ACC a, D_ACC b) { return a + b; }),
        lane_id % T::W_M,
        warp_id,
        [](D_ACC a, D_ACC b) { return a + b; });

    if(kargs.num_splits > 1)
    {
        // Publish the unnormalized accumulator with its (m, l) so the reducer can
        // rescale across splits. Rows past the GQA group fall outside the bound.
        auto u_po = partition_layout_c(
            mma1, opus::make_tuple(number<T::D_HEAD>{}, 1_I),
            opus::make_tuple(0_I, lane_id % mma1.grpn_c, warp_id, lane_id / mma1.grpn_c));
        auto g_po = make_gmem(partial_o_ptr<T>(kargs, batch_idx, kv_head_idx, split_idx),
                              kargs.gqa_ratio * T::D_HEAD * sizeof(D_ACC));
        store<4>(g_po, v_o, u_po);

        // Lanes 0..Q_TILE-1 of wave 0 sit on query rows 0..Q_TILE-1, so their
        // register copies of m/l are exactly the per-row values to publish.
        if(tid < T::Q_TILE)
        {
            float* p_ml           = partial_ml_ptr<T>(kargs, batch_idx, kv_head_idx, split_idx);
            p_ml[tid]             = m_run;
            p_ml[T::Q_TILE + tid] = l_final;
        }

        if(split_arrive_is_last<T>(kargs, batch_idx, kv_head_idx, tid))
            reduce_splits<T>(kargs, batch_idx, kv_head_idx, tid, smem);
        return;
    }

    // ── Normalize and write back. Rows past the GQA group are dropped by the buffer bound. ──
    const D_ACC o_scale = l_final > D_ACC(0.0f) ? D_ACC(1.0f) / l_final : D_ACC(0.0f);
    opus::static_for<vector_traits<decltype(v_o)>::size()>(
        [&](auto i) { v_o[i.value] *= o_scale; });

    const int64_t o_offset = static_cast<int64_t>(batch_idx) * kargs.stride_o_b
                             + static_cast<int64_t>(q_head_base) * kargs.stride_o_h;
    auto g_o = make_gmem(reinterpret_cast<D_ATTN*>(kargs.out_ptr) + o_offset,
                         kargs.gqa_ratio * kargs.stride_o_h * sizeof(D_ATTN));
    auto v_o_attn = cast<D_ATTN>(v_o);
    store<4>(g_o, v_o_attn, u_o);
}

#else // !__gfx950__
template<class Traits>
__global__ void pa_decode_opus_kernel(pa_decode_kargs kargs) {}
#endif

#else // !__HIP_DEVICE_COMPILE__
// Host pass only needs the launch stub; opus.hpp stays out of this translation pass.
template<class Traits>
__global__ void pa_decode_opus_kernel(pa_decode_kargs kargs) {}
#endif // __HIP_DEVICE_COMPILE__

#endif // PA_DECODE_OPUS_IMPL
