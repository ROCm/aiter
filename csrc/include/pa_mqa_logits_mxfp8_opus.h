// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// MXFP8 paged MQA logits (gfx950), OPUS implementation.
//
// Semantics (per query row r, window [s, e)):
//   out[r, s:e] = sum_H( relu(Q[r] . Kᵀ) * weight[r] ) * weight_scale
//
// This op is a thin launcher: Q/KV must already be MXFP8-quantized and
// preshuffled into the kernel ABI, and the persistent-grid schedule
// (`cta_info[n_ctas, CTA_INFO_WIDTH]`) must already be built (see the schedule
// builders below). Quant/preshuffle is the caller's responsibility.

#pragma once
#include "aiter_tensor.h"
#include <cstdint>

// cta_info packed fields per CTA ([n_ctas, CTA_INFO_WIDTH] int32) — public ABI:
//   0: row_id       -- output row / flat query-token index
//   1: batch_id     -- sequence index (indexes block_tables)
//   2: chunk_start  -- first KV chunk (block_k units) this CTA handles
//   3: chunk_count  -- number of chunks this CTA handles
//   4: local_start  -- window lower bound [start, end)
//   5: local_end    -- window upper bound
static constexpr int CTA_INFO_WIDTH = 6;

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

// Launch the MXFP8 MQA logits kernel over a prebuilt persistent-grid schedule.
//
// Tensor expectations (row-major, last dim contiguous):
//   q            : [total_tokens, H, D]                               fp8  (E4M3)
//   q_scale      : [total_tokens, K_TILES, K_CHUNKS, 16, QS_PAD]      uint8 (E8M0, preshuffled)
//   kv_cache     : [num_blocks, K_TILES, 8, PAGE, 16]                 fp8  (E4M3, paged+preshuffled)
//   kv_scale     : [num_blocks, K_TILES, K_CHUNKS, PAGE]             uint8 (E8M0)
//   block_tables : [batch, max_blocks_per_seq]                        int32
//   weights      : [total_tokens, H]                                  bf16
//   cta_info     : [n_ctas, CTA_INFO_WIDTH]                           int32 (persistent-grid schedule)
//   out          : [total_tokens, max_seq_len]                        fp32  (caller-allocated; OOB left as-is)
// `weight_scale` is folded into the per-head weighting; `block_k` / `kv_block_size`
// must match the compiled traits (256 / 64 for the 4-wave variant).
void pa_mqa_logits_mxfp8_fwd(aiter_tensor_t& q,
                          aiter_tensor_t& q_scale,
                          aiter_tensor_t& kv_cache,
                          aiter_tensor_t& kv_scale,
                          aiter_tensor_t& block_tables,
                          aiter_tensor_t& weights,
                          aiter_tensor_t& cta_info,
                          aiter_tensor_t& out,
                          int n_ctas,
                          float weight_scale,
                          int block_k,
                          int kv_block_size,
                          int max_seq_len);

// Build the persistent-grid schedule `cta_info[P, CTA_INFO_WIDTH]` from ragged
// windows (device, cudagraph-safe: all inputs/outputs are caller-allocated
// device buffers; grid is derived from static shapes, not from window values).
//   row_to_batch / local_starts / local_ends : [total_tokens] int32
//   scratch                                  : [>= total_tokens + 2] int32 workspace
//   cta_info                                 : [parallel_unit_num, CTA_INFO_WIDTH] int32 (out)
void pa_mqa_logits_mxfp8_prefill_schedule(aiter_tensor_t& row_to_batch,
                                       aiter_tensor_t& local_starts,
                                       aiter_tensor_t& local_ends,
                                       aiter_tensor_t& scratch,
                                       aiter_tensor_t& cta_info,
                                       int total_tokens,
                                       int parallel_unit_num,
                                       int block_k,
                                       int max_seq_len);

// Build ragged-row metadata for per-batch variable query length (MTP decode).
// MTP tail-causal: batch b's n-th query token sees [0, context_len[b] - (qlen-1-n)).
//   cu_seq_q     : [B+1]      int32 (prefix sum of per-batch qlen)
//   context_lens : [B]        int32
//   row_to_batch / local_starts / local_ends : [total_q] int32 (out)
void pa_mqa_logits_mxfp8_varqlen_windows(aiter_tensor_t& cu_seq_q,
                                      aiter_tensor_t& context_lens,
                                      aiter_tensor_t& row_to_batch,
                                      aiter_tensor_t& local_starts,
                                      aiter_tensor_t& local_ends,
                                      int total_q);

#ifdef PA_MQA_LOGITS_MXFP8_IMPL
// ============================================================================
// Implementation section - only compiled in the .cu translation unit(s).
// Kernel-arg / traits plumbing (ABI-identical to the standalone
// gcnasm/opus_logits/logits_defs.h) + the device kernel template.
// ============================================================================

// Scalar data types (host+device safe; identical to opus::bf16_t / opus::fp8_t).
using mqa_logits_bf16_t = __bf16;
using mqa_logits_fp8_t  = _BitInt(8);   // fp8 E4M3 storage (== opus::fp8_t)

// Kernel arguments. Pointers are opaque; the kernel reinterprets per its ABI.
struct opus_mqa_logits_kargs {
    const void* __restrict__ ptr_q;         // [total_tokens, H, D]                          fp8 (E4M3)
    const void* __restrict__ ptr_q_scale;   // [total_tokens, K_TILES, K_CHUNKS, 16, QS_PAD] e8m0
    const void* __restrict__ ptr_kv;        // [num_blocks, K_TILES, 8, PAGE, 16]            fp8 (E4M3)
    const void* __restrict__ ptr_kv_scale;  // [num_blocks, K_TILES, K_CHUNKS, PAGE]         e8m0
    const int*  __restrict__ ptr_block_tables; // [batch, max_blocks_per_seq] int32
    const void* __restrict__ ptr_weights;   // [total_tokens, H] bf16
    const int*  __restrict__ ptr_cta_info;   // [n_ctas, CTA_INFO_WIDTH] int32
    float* __restrict__ ptr_out;             // [total_tokens, max_seq_len] fp32

    int   max_seq_len;
    int   stride_out_row;      // out row stride in elements (== max_seq_len for dense out)
    float weight_scale;
    int   block_k;             // KV tile size along seq_kv (== Traits::KV_TILE_SIZE)
    int   kv_block_size;       // paged block (page) size (== Traits::PAGE_SIZE)
    int   max_blocks_per_seq;  // block_tables row stride
};

// Compile-time config. Defaults target the 4-wave MXFP8 variant:
//   NUM_WARPS=4 -> BLOCK=256, KV_TILE=256 (block_k), PAGE=64, D=128, H=64.
template<int KV_TILE_SIZE_ = 256,
         int PAGE_SIZE_     = 64,
         int HEAD_DIM_      = 128,
         int N_HEADS_       = 64,
         int NUM_WARPS_     = 4>
struct opus_mqa_logits_traits {
    static constexpr int KV_TILE_SIZE = KV_TILE_SIZE_;  // block_k
    static constexpr int PAGE_SIZE    = PAGE_SIZE_;     // kv_block_size
    static constexpr int HEAD_DIM     = HEAD_DIM_;
    static constexpr int N_HEADS      = N_HEADS_;
    static constexpr int NUM_WARPS    = NUM_WARPS_;

    static constexpr int WARP_SIZE  = 64;
    static constexpr int BLOCK_SIZE = NUM_WARPS * WARP_SIZE;  // 256 (4-wave)

    // Data-type aliases the kernel refers to.
    using D_DATA   = mqa_logits_fp8_t;    // Q / KV data (E4M3), MFMA A/B operand element
    using D_WEIGHT = mqa_logits_bf16_t;   // per-head weights
    using D_ACC    = float;               // MFMA accumulator (C)
    using D_OUT    = float;               // output logits
    using D_SCALE  = int;                 // E8M0 blockscale, packed as an int32 dword per lane

    // MFMA: scaled f8f6f4, 16x16x128 (M x N x K). K covers 128 head_dim in one shot.
    static constexpr int MFMA_M = 16;
    static constexpr int MFMA_N = 16;
    static constexpr int MFMA_K = 128;

    static constexpr int SCALE_BLOCK = 32;                 // E8M0 blockscale granularity
    static constexpr int K_CHUNKS    = MFMA_K / SCALE_BLOCK; // 4 (32-elem chunks per 128-K)

    static constexpr int m_tiles = N_HEADS / MFMA_M;       // 4  (head tiles along M)
    static constexpr int k_tiles = HEAD_DIM / MFMA_K;      // 1  (outer K loop)

    static constexpr int N_TILES          = KV_TILE_SIZE / MFMA_N;    // 16
    static constexpr int N_TILES_PER_WARP = N_TILES / NUM_WARPS;      // 4
    static constexpr int TILES_PER_BLOCK  = PAGE_SIZE / MFMA_N;       // 4 (MFMA_N tiles per page)
    // #physical blocks a warp's NTPW tiles span (1 => all NTPW share one page).
    static constexpr int N_PHYS = (N_TILES_PER_WARP + TILES_PER_BLOCK - 1) / TILES_PER_BLOCK; // 1

    static constexpr int ELEM_BYTES = 1;   // fp8 = 1 byte/elem

    // MI350 fp8 16x16x128 register layout: lane g=lane/16 holds
    //   vgpr0-3 = K[g*16 : +16]  and  vgpr4-7 = K[64 + g*16 : +16]  (two 16-K groups).
    // (fp4 differs: lane g holds one contiguous K[g*32:+32].)
    static constexpr int KV_GRP_ELEMS = 16;                   // 16 fp8 per vgpr group
    static constexpr int KV_GRP_BYTES = KV_GRP_ELEMS;         // 16 bytes
    static constexpr int A_BYTES_PER_LANE = (MFMA_M * MFMA_K / WARP_SIZE) * ELEM_BYTES; // 32 (i32x8)
    static constexpr int VGRP_K_OFFSET = 64;                  // K gap between the two groups (Q gather)

    static constexpr int C_FRAG        = MFMA_M * MFMA_N / WARP_SIZE;      // 4: C frag / head rows per lane per m-tile (== weight VEC)
    static constexpr int LANE_M_GROUPS = WARP_SIZE / MFMA_M;              // 4: 16-lane groups per wave (lane_div_16 range)
    static constexpr int VGRP_PER_LANE = A_BYTES_PER_LANE / KV_GRP_BYTES;  // 2: 16-elem K groups per A/B fragment

    // q_scale preshuffle: [T, K_TILES, 4, 16, QS_PAD]; QS_PAD = round_up(m_tiles,4).
    static constexpr int QS_DW  = (m_tiles + 3) / 4;   // 1
    static constexpr int QS_PAD = QS_DW * 4;           // 4

    // ---- byte strides for paged KV cache [num_blocks, K_TILES, 8(c), PAGE, 16] ----
    // chunk c = contiguous K[c*16 : +16] within a 128-K tile (c = 0..7). Lane g loads
    // c=g (vgpr0-3) and c=g+4 (vgpr4-7). page kept after c so consecutive tokens are contiguous.
    static constexpr int K16_PER_TILE   = MFMA_K / 16;                  // 8
    static constexpr int VGRP_CHUNK_HI  = 4;                            // vgpr4-7 chunk = g + 4
    static constexpr int stride_kv_chunk = PAGE_SIZE * KV_GRP_BYTES;    // one 16-K chunk across page
    static constexpr int stride_kv_ktile = K16_PER_TILE * stride_kv_chunk;
    static constexpr int stride_kv_block = k_tiles * stride_kv_ktile;
    // ---- byte strides for kv_scale [num_blocks, K_TILES, 4, PAGE] (e8m0, 1 byte) ----
    static constexpr int stride_kvs_ktile = K_CHUNKS * PAGE_SIZE;
    static constexpr int stride_kvs_block = k_tiles  * stride_kvs_ktile;

    static_assert(HEAD_DIM % MFMA_K == 0, "HEAD_DIM must be a multiple of MFMA_K (128)");
    static_assert(N_HEADS % MFMA_M == 0, "N_HEADS must be a multiple of MFMA_M (16)");
    static_assert(m_tiles <= 8, "m_tiles > 8 unsupported (use N_HEADS <= 128)");
    static_assert(KV_TILE_SIZE % MFMA_N == 0, "KV_TILE must be a multiple of MFMA_N");
    static_assert(N_TILES % NUM_WARPS == 0, "N_TILES must be a multiple of NUM_WARPS");
    static_assert(PAGE_SIZE % MFMA_N == 0, "PAGE must be a multiple of MFMA_N");
    static_assert(KV_TILE_SIZE % PAGE_SIZE == 0, "KV_TILE must be a multiple of PAGE");
    static_assert(N_TILES_PER_WARP == 4, "kernel currently assumes N_TILES_PER_WARP == 4");
    static_assert(N_PHYS == 1, "kernel currently assumes N_PHYS == 1 (NTPW tiles share one page)");
};

__host__ __device__ inline int mqa_logits_ceil_div(int a, int b) { return (a + b - 1) / b; }

#ifndef __HIP_DEVICE_COMPILE__
// Host pass: empty stub so the __device_stub__ symbol resolves for the launcher.
namespace opus_logits {
template<class T> __global__ void pa_mqa_logits_mxfp8_kernel(opus_mqa_logits_kargs) {}
}
#else
// ============================================================================
// Device kernel (compute unchanged; ported verbatim from
// gcnasm/opus_logits/mqa_logits_mxfp8_kernel_template.hpp).
// ============================================================================
#include <opus/opus.hpp>

// Minimum waves/SIMD hint for __launch_bounds__. With the 4-wave (BLOCK=256) variant, a bare
// __launch_bounds__(256) relaxes the VGPR ceiling to 512 and clang balloons to ~304 VGPR (occ 1);
// pinning min-occupancy 2 caps VGPR near 256 to keep 2 waves/SIMD. Set 0/1 to let clang choose.
#ifndef OPUS_LOGITS_MIN_WAVES
#define OPUS_LOGITS_MIN_WAVES 2
#endif

namespace opus_logits {

// ── sched_group_barrier co-exec control (mha idiom) ──────────────────────────────────────────
// LLVM SchedGroup masks: pin which instruction classes land in each ordered group so the
// scaled-MFMA shadow gets filled with exactly the ops that CAN co-execute with it.
constexpr int VALU_MASK = 0x02;   // v_max / v_add / v_pk_fma ... (no MFMA/TRANS/mem)
constexpr int MFMA_MASK = 0x08;   // v_mfma_*

// Pairs × (1 MFMA, CNT of OTHER_MASK) in program order, same SyncID = Group. Interleaves each MFMA
// with CNT co-exec ops issued in its shadow. The mxfp8 scaled MFMA can only co-exec with 4 insts,
// so keep CNT<=4.
template<int Pairs, int OTHER_MASK, int CNT, int Group>
__device__ inline void sched_mfma_pairs() {
    __builtin_amdgcn_sched_group_barrier(MFMA_MASK, 1, Group);
    __builtin_amdgcn_sched_group_barrier(OTHER_MASK, CNT, Group);
    if constexpr (Pairs > 1) sched_mfma_pairs<Pairs - 1, OTHER_MASK, CNT, Group>();
}

// Butterfly add across the XOR-partner lane (head reduction step).
__device__ inline float bperm_xor_add(float v, int lane, int sh) {
    int peer_byte = (lane ^ sh) << 2;   // ds_bpermute addresses in bytes (lane*4)
    int peer = __builtin_amdgcn_ds_bpermute(peer_byte, __builtin_bit_cast(int, v));
    return v + __builtin_bit_cast(float, peer);
}

// ---------------------------------------------------------------------------
// Per-lane make_layout builders (byte/element units; fp8 elem = 1 byte).
// Convention: p-dims carry a fixed thread coord; y-dims are iterated by load/store,
// with the trailing y-dim acting as the contiguous VEC dimension.
// ---------------------------------------------------------------------------

// Q (A operand), natural [T,H,128] fp8; row folded into the gmem base pointer.
// Per m-tile the loaded 32 bytes are [group0=K[g*16:+16], group1=K[64+g*16:+16]] = i32x8.
template<class T>
__device__ inline auto make_layout_q(int lane_mod_16, int lane_div_16) {
    constexpr auto shape = opus::make_tuple(
        opus::number<T::m_tiles>{},        // m-tile              (y, outer)
        opus::number<T::VGRP_PER_LANE>{},  // 16-K group (hi/lo)  (y)
        opus::number<T::MFMA_M>{},         // head row in tile    (p: lane_mod_16)
        opus::number<T::K_CHUNKS>{},       // K group g           (p: lane_div_16)
        opus::number<T::KV_GRP_ELEMS>{});  // VEC = 16 fp8        (y, contiguous)
    constexpr auto dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}),                    // m-tile
        opus::make_tuple(opus::y_dim{}),                    // 16-K group
        opus::make_tuple(opus::p_dim{}),                    // head row
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}));    // g + VEC (contiguous: g=16, VEC=1)
    return opus::make_layout(
        shape,
        opus::unfold_x_stride(dim, shape, opus::make_tuple(
            opus::number<T::MFMA_M * T::HEAD_DIM>{}, // m-tile: 16 heads
            opus::number<T::VGRP_K_OFFSET>{},        // group: +64 in K
            opus::number<T::HEAD_DIM>{},             // head row
            opus::number<1>{})),                     // (g, VEC) base=1 -> g=KV_GRP_ELEMS(16), VEC=1
        opus::unfold_p_coord(dim, opus::make_tuple(lane_mod_16, lane_div_16)));
}

// Per-nt KV (B operand), paged [num_blocks,K_TILES,8(c),page,16] fp8: ONE token-tile's i32x8
// (== Mma::vtype_b). The caller adds nt*(MFMA_N*KV_GRP_BYTES) to the offset. Splitting the tile load
// into NTPW independent per-nt buffer_loads lets the cross-tile nt=0 gemm wait for just nt=0's load.
template<class T>
__device__ inline auto make_layout_kv_nt(int lane_mod_16, int lane_div_16) {
    constexpr auto shape = opus::make_tuple(
        opus::number<T::VGRP_PER_LANE>{},    // vgpr group (c=g / c=g+4) (y)
        opus::number<T::MFMA_N>{},           // token o in page          (p: lane_mod_16)
        opus::number<T::K_CHUNKS>{},         // chunk g                   (p: lane_div_16)
        opus::number<T::KV_GRP_ELEMS>{});    // VEC = 16 fp8              (y, contiguous)
    constexpr auto dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}),
        opus::make_tuple(opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}));
    return opus::make_layout(
        shape,
        opus::unfold_x_stride(dim, shape, opus::make_tuple(
            opus::number<T::VGRP_CHUNK_HI * T::stride_kv_chunk>{},
            opus::number<T::KV_GRP_BYTES>{},
            opus::number<T::stride_kv_chunk>{},
            opus::number<1>{})),
        opus::unfold_p_coord(dim, opus::make_tuple(lane_mod_16, lane_div_16)));
}

// Scale word (q_scale / kv_scale): one E8M0 dword per lane (int units). The 4 bytes
// pack the 4 m-tiles (q) / token-tiles (kv); MFMA op_sel selects the byte. Trailing
// size-1 y-dim provides the (single) load issue.
template<class T>
__device__ inline auto make_layout_scale(int lane_div_16, int lane_mod_16) {
    constexpr auto shape = opus::make_tuple(
        opus::number<T::K_CHUNKS>{},   // block/group g   (p: lane_div_16)
        opus::number<T::MFMA_M>{},     // row/col in tile (p: lane_mod_16)
        opus::number<1>{});            // single dword    (y)
    constexpr auto dim = opus::make_tuple(
        opus::make_tuple(opus::p_dim{}, opus::p_dim{}),   // g + lane (contiguous: g=16, lane=1)
        opus::make_tuple(opus::y_dim{}));                 // single dword
    return opus::make_layout(
        shape,
        opus::unfold_x_stride(dim, shape, opus::make_tuple(
            opus::number<1>{},          // (g, lane) base=1 -> g=MFMA_M(16), lane=1
            opus::number<0>{})),        // y (size 1)
        opus::unfold_p_coord(dim, opus::make_tuple(lane_div_16, lane_mod_16)));
}

// block_tables (page table): int32 [batch, max_blocks_per_seq]; batch folded into base.
// Warp w reads page index bi = (chunk_start+tile_idx)*(block_k/PAGE) + warp_id; the per-tile term
// is added at the load site, so the layout contributes just the warp partition (stride 1).
template<class T>
__device__ inline auto make_layout_bt(int warp_id) {
    constexpr auto shape = opus::make_tuple(
        opus::number<T::NUM_WARPS>{},  // page-in-chunk = warp id (p)
        opus::number<1>{});            // single int      (y)
    constexpr auto dim = opus::make_tuple(
        opus::make_tuple(opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}));
    return opus::make_layout(
        shape,
        opus::unfold_x_stride(dim, shape, opus::make_tuple(
            opus::number<1>{},   // warp stride: bi += 1 per warp
            opus::number<0>{})), // y (size 1)
        opus::unfold_p_coord(dim, opus::make_tuple(warp_id)));
}

// Weights: bf16 [T,H]; row folded into base. Per m-tile 4 contiguous bf16 at
// head0 = mi*16 + lane_div_16*4.
template<class T>
__device__ inline auto make_layout_w(int lane_div_16) {
    constexpr auto shape = opus::make_tuple(
        opus::number<T::m_tiles>{},        // m-tile          (y, outer)
        opus::number<T::LANE_M_GROUPS>{},  // lane_div_16 grp (p)
        opus::number<T::C_FRAG>{});        // VEC = C_FRAG bf16 (y, contiguous, one per head)
    constexpr auto dim = opus::make_tuple(
        // m-tile + lane_div_16 grp + VEC all contiguous (16 = 4 grp x 4 VEC) -> one group, base=1.
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::y_dim{}));
    return opus::make_layout(
        shape,
        opus::unfold_x_stride(dim, shape, opus::make_tuple(
            opus::number<1>{})),       // base=1 -> m-tile=16, lane_div_16=4, VEC=1
        opus::unfold_p_coord(dim, opus::make_tuple(lane_div_16)));
}

// Output: fp32 [T, max_seq_len]; row+local_start folded into the gmem base. Whole-tile layout
// covering ALL warps: token-within-chunk = warp_id*(NTPW*MFMA_N) + nt*MFMA_N + lane_mod_16, so
// the store only adds the chunk base -- no warp_id / NTPW arithmetic at the store site.
template<class T>
__device__ inline auto make_layout_out(int warp_id, int lane_mod_16) {
    constexpr auto shape = opus::make_tuple(
        opus::number<T::N_TILES_PER_WARP>{}, // token-tile nt              (y)
        opus::number<T::NUM_WARPS>{},        // warp = NTPW*MFMA_N tokens   (p: warp_id)
        opus::number<T::MFMA_N>{});          // token in tile              (p: lane_mod_16)
    constexpr auto dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}),
        opus::make_tuple(opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}));
    return opus::make_layout(
        shape,
        opus::unfold_x_stride(dim, shape, opus::make_tuple(
            opus::number<T::MFMA_N>{},                        // nt stride
            opus::number<T::N_TILES_PER_WARP * T::MFMA_N>{},  // warp stride (== PAGE tokens)
            opus::number<1>{})),                              // lane_mod_16 stride
        opus::unfold_p_coord(dim, opus::make_tuple(warp_id, lane_mod_16)));
}

template<class T>
__global__ __launch_bounds__(T::BLOCK_SIZE, OPUS_LOGITS_MIN_WAVES) void pa_mqa_logits_mxfp8_kernel(opus_mqa_logits_kargs kargs) {
    // ---- data-type aliases ----
    using D_DATA   = typename T::D_DATA;
    using D_WEIGHT = typename T::D_WEIGHT;
    using D_SCALE  = typename T::D_SCALE;
    using D_ACC    = typename T::D_ACC;
    using D_OUT    = typename T::D_OUT;

    constexpr int WARP     = T::WARP_SIZE;          // 64
    constexpr int NTPW     = T::N_TILES_PER_WARP;   // 4
    constexpr int MT       = T::m_tiles;            // 4
    constexpr int PAGE     = T::PAGE_SIZE;          // 64
    constexpr int HEAD_DIM = T::HEAD_DIM;           // 128
    constexpr int N_HEADS  = T::N_HEADS;            // 64
    // mxfp8 scaled MFMA can co-exec with at most this many VALU in its shadow (mha idiom).
    constexpr int MFMA_VALU_COEXEC = 4;

    const int block_k = kargs.block_k;              // 256 (4-wave)
    const int max_blk = kargs.max_blocks_per_seq;

    const int tid = opus::thread_id_x();
    const int pid = opus::block_id_x();
    const int warp_id     = tid >> 6;
    const int lane_id     = tid & (WARP - 1);
    const int lane_mod_16 = lane_id & (T::MFMA_M - 1);               // position within a 16-lane group
    const int lane_div_16 = (lane_id >> 4) & (T::LANE_M_GROUPS - 1); // which 16-lane group (0..LANE_M_GROUPS-1)

    // ---- per-CTA assignment (uniform across block) ----
    const int* ci = kargs.ptr_cta_info + (size_t)pid * CTA_INFO_WIDTH;
    const int row_id      = ci[0];
    const int batch_id    = ci[1];
    const int chunk_start = ci[2];
    const int tile_count  = ci[3];
    const int local_start = ci[4];
    const int local_end   = ci[5];

    // ---- GEMM object (scaled fp8 16x16x128); use its single-tile MMA + accumulator type ----
    auto mma = opus::make_tiled_mma<D_DATA, D_DATA, D_ACC>(
        opus::seq<MT, NTPW, T::k_tiles>{},                        // expand (E_M,E_N,E_K)
        opus::seq<1, T::NUM_WARPS, 1>{},                          // waves  (T_M,T_N,T_K)
        opus::seq<T::MFMA_M, T::MFMA_N, T::MFMA_K>{});            // wave   (W_M,W_N,W_K)
    using Mma = typename decltype(mma)::MMA;
    Mma mma_op;

    // ---- buffer resources (row folded into base pointers to keep voffsets small) ----
    const D_DATA*   q_base  = reinterpret_cast<const D_DATA*>(kargs.ptr_q) + (size_t)row_id * N_HEADS * HEAD_DIM;
    const D_SCALE*  qs_base = reinterpret_cast<const D_SCALE*>(kargs.ptr_q_scale) + (size_t)row_id * (T::K_CHUNKS * T::MFMA_M);
    const D_WEIGHT* w_base  = reinterpret_cast<const D_WEIGHT*>(kargs.ptr_weights) + (size_t)row_id * N_HEADS;
    const int*      bt_base = kargs.ptr_block_tables + (size_t)batch_id * max_blk;   // fold batch into base
    // Out base folds ONLY the row (16B-safe: never depends on local_start; fixes the leading-token
    // drop for local_start % 4 != 0 -- see gcnasm/opus_logits/KNOWN_ISSUE_out_store_alignment.md).
    // Store offsets are absolute token indices; masking is by forcing a < 0 (OOB) offset (see do_store).
    D_OUT*          out_base  = kargs.ptr_out + (size_t)row_id * kargs.stride_out_row;
    const unsigned  out_bytes = (unsigned)(local_end > 0 ? local_end : 0) * (unsigned)sizeof(D_OUT);

    auto g_q   = opus::make_gmem(q_base);
    auto g_qs  = opus::make_gmem(qs_base);
    auto g_kv  = opus::make_gmem(reinterpret_cast<const D_DATA*>(kargs.ptr_kv));
    auto g_kvs = opus::make_gmem(reinterpret_cast<const D_SCALE*>(kargs.ptr_kv_scale));
    auto g_bt  = opus::make_gmem(bt_base);
    auto g_w   = opus::make_gmem(w_base);
    auto g_out = opus::make_gmem(out_base, out_bytes);   // OOB size == local_end: drops out-of-window / non-writer (< 0) offsets

    // ---- Hoist Q (i32x8 per m-tile) ----
    auto u_q = make_layout_q<T>(lane_mod_16, lane_div_16);
    auto v_q = opus::load<T::KV_GRP_ELEMS>(g_q, u_q);
    auto* q_a = reinterpret_cast<typename Mma::vtype_a*>(&v_q);   // vector_t<D_DATA, 32> per m-tile

    // ---- Hoist Q scale (one dword; byte mi = e8m0 for head-tile mi) ----
    auto u_qs = make_layout_scale<T>(lane_div_16, lane_mod_16);
    const int q_scale = opus::load<1>(g_qs, u_qs)[0];

    // ---- Hoist weights (C_FRAG f32 per m-tile, one per head row) with weight_scale folded in ----
    auto u_w = make_layout_w<T>(lane_div_16);
    auto v_w = opus::load<T::C_FRAG>(g_w, u_w);
    auto* w4 = reinterpret_cast<opus::vector_t<D_WEIGHT, T::C_FRAG>*>(&v_w);
    opus::vector_t<float, T::C_FRAG> w_pl[MT];
    opus::static_for<MT>([&](auto mic) {
        constexpr int mi = mic.value;
        opus::static_for<T::C_FRAG>([&](auto ec) {
            constexpr int e = ec.value;
            w_pl[mi][e] = (float)w4[mi][e] * kargs.weight_scale;
        });
    });

    auto u_kv_nt = make_layout_kv_nt<T>(lane_mod_16, lane_div_16);   // per-nt split KV load
    auto u_kvs = make_layout_scale<T>(lane_div_16, lane_mod_16);
    auto u_bt  = make_layout_bt<T>(warp_id);
    const int tok_lane_base = warp_id * (NTPW * T::MFMA_N) + lane_mod_16;   // row r's abs out-token = c*block_k + nt*MFMA_N + this
    // Single-writer lane via a large-negative ADDRESS bias (no compare): only lane_div_16 == 0 stores;
    // the head reduction leaves all 4 groups with the same out[nt], so the other 3 get a < 0 offset
    // (below the CTA's smallest out-token base) that the OOB size drops.
    const int lane_bias = (lane_div_16 != 0) ? -(chunk_start + tile_count) * block_k : 0;
    constexpr int pages_per_tile = T::KV_TILE_SIZE / PAGE;   // warps per tile (== NUM_WARPS)

    // Per-tile paged lookups (block-table page id, then that page's packed KV scale word).
    // page index within seq: bi = (chunk_start+tile_idx)*pages_per_tile + warp_id (uniform in warp).
    auto load_page_id  = [&](int tile_idx) {
        return opus::load<1>(g_bt, u_bt + (chunk_start + tile_idx) * pages_per_tile)[0];
    };
    // kv_scale word: page_id folded via layout + (int units = stride_kvs_block/4).
    auto load_kv_scale = [&](int page_id) {
        return opus::load<1>(g_kvs, u_kvs + page_id * (T::stride_kvs_block / (int)sizeof(int)))[0];
    };

    constexpr int EC = Mma::elem_c;   // per-(mi,nt) C fragment length (== 4)
    using kv_nt_t = decltype(opus::load<T::KV_GRP_ELEMS>(g_kv, u_kv_nt));    // one nt's i32x8 (== vtype_b)

    auto clamp_tile = [&](int t) { return t < tile_count ? t : tile_count - 1; };
    // Split KV load: NTPW independent per-nt loads, nt=0 first.
    auto issue_ks = [&](kv_nt_t (&kv)[NTPW], int& kvs_w, int page_id) {
        kvs_w = load_kv_scale(page_id);
        opus::static_for<NTPW>([&](auto ntc) {
            constexpr int nt = ntc.value;
            kv[nt] = opus::load<T::KV_GRP_ELEMS>(
                g_kv, u_kv_nt + page_id * T::stride_kv_block + nt * (T::MFMA_N * T::KV_GRP_BYTES));
        });
    };
    // One nt's score = ONE CONTIGUOUS MT*EC-wide accumulator (a slice of the conceptual whole-S).
    // A single vector (not an MT-array of EC-wide frags) keeps its (mi,e) layout contiguous & even-
    // aligned, so the packed reduce reads v_pk_fma operands as free sub-slices (no v_mov to assemble).
    using sfrag = opus::vector_t<float, MT * EC>;
    // gemm for one nt -> the sfrag (MT scaled MFMAs written via set_slice; op_sel_a=mi, op_sel_b=nt).
    auto gemm_nt = [&](sfrag& accs, kv_nt_t& kvnt, int kvs_w, auto ntc) {
        constexpr int nt = ntc.value;
        auto& kv_b = *reinterpret_cast<typename Mma::vtype_b*>(&kvnt);
        opus::static_for<MT>([&](auto mic) {
            constexpr int mi = mic.value;
            typename Mma::vtype_c c0{};
            auto acc = mma_op(q_a[mi], kv_b, c0, q_scale, kvs_w, opus::number<mi>{}, opus::number<nt>{});
            opus::set_slice(accs, acc, opus::number<mi * EC>{}, opus::number<mi * EC + EC>{});
        });
    };
    // relu split OUT of the reduce: standalone in-place v_max over the 16 accumulators. No w / bperm
    // dependency, so the scheduler is free to slot these v_max into an MFMA shadow (mfma+max co-exec).
    auto relu_nt = [&](sfrag& accs) {
        opus::static_for<MT * EC>([&](auto ic) {
            constexpr int i = ic.value;
            accs[i] = accs[i] > 0.0f ? accs[i] : 0.0f;
        });
    };

    // Batched reduce of ALL NTPW nt at once: one 2-wide accumulator PER OUTPUT nt, nt INNERMOST, so
    // consecutive v_pk_fma hit different regs (no s_nop) and the final combine is a trivial per-output
    // 2-lane add (no cross-mi tree -> few v_mov). Consumes the ALREADY-relu'd score frags.
    auto reduce_all = [&](sfrag& s0, sfrag& s1, sfrag& s2, sfrag& s3, opus::vector_t<float, NTPW>& out) {
        sfrag* sp[NTPW] = { &s0, &s1, &s2, &s3 };
        opus::vector_t<float, 2> acc[NTPW];
        opus::static_for<NTPW>([&](auto ntc) { acc[ntc.value] = opus::vector_t<float, 2>{0.0f, 0.0f}; });
        opus::static_for<MT>([&](auto mic) {
            constexpr int mi = mic.value;
            opus::static_for<EC / 2>([&](auto pc) {
                constexpr int p = pc.value;
                opus::vector_t<float, 2> w{ w_pl[mi][2 * p], w_pl[mi][2 * p + 1] };
                opus::static_for<NTPW>([&](auto ntc) {   // nt INNERMOST -> consecutive pk_fma hit acc[0..3]
                    constexpr int nt = ntc.value;
                    sfrag& sn = *sp[nt];
                    opus::vector_t<float, 2> s{ sn[mi * EC + 2 * p], sn[mi * EC + 2 * p + 1] };
                    acc[nt] = s * w + acc[nt];
                });
            });
        });
        opus::static_for<NTPW>([&](auto ntc) {
            constexpr int nt = ntc.value;
            float ts = acc[nt][0] + acc[nt][1];
            // head reduce across the LANE_M_GROUPS 16-lane groups: xor MFMA_M then 2*MFMA_M.
            ts = bperm_xor_add(ts, lane_id, T::MFMA_M);
            ts = bperm_xor_add(ts, lane_id, 2 * T::MFMA_M);
            out[nt] = ts;
        });
    };
    // FlyDSL-style masked store (branch-free): per nt, keep the absolute out-token offset if in-window
    // (with the single-writer bias folded in), else force -1. Out-of-window / non-writer become a < 0
    // offset that the buffer OOB size drops -- one v_cndmask + one UNCONDITIONAL store per nt, no store_if.
    const unsigned win = (unsigned)(local_end > local_start ? local_end - local_start : 0);
    auto do_store = [&](opus::vector_t<float, NTPW>& out, int t) {
        const int tile_off = (chunk_start + t) * block_k;
        opus::static_for<NTPW>([&](auto ntc) {
            constexpr int nt = ntc.value;
            const int  abs_tok = tile_off + nt * T::MFMA_N + tok_lane_base;
            const bool in_win  = (unsigned)(abs_tok - local_start) < win;
            const int  off     = in_win ? (abs_tok + lane_bias) : -1;
            g_out.template store<1>(out[nt], off);
        });
    };

    // Empty assignment (0 tiles): nothing to load/compute/store (uniform across the block).
    if (tile_count <= 0) return;

    auto compute_phase = [&](kv_nt_t (&kv_c)[NTPW], int& kvs_c, sfrag& a0_in,
                             kv_nt_t (&kv_p)[NTPW], int& kvs_p, sfrag& a0_out,
                             int& pg_c, int pf_tile, opus::vector_t<float, NTPW>& out_cur) {
        sfrag s1, s2, s3;
        __builtin_amdgcn_sched_barrier(0);
        gemm_nt(s1, kv_c[1], kvs_c, opus::number<1>{});
        relu_nt(a0_in);
        sched_mfma_pairs<MT, VALU_MASK, MFMA_VALU_COEXEC, 1>();
        __builtin_amdgcn_sched_barrier(0);
        gemm_nt(s2, kv_c[2], kvs_c, opus::number<2>{});
        relu_nt(s1);
        sched_mfma_pairs<MT, VALU_MASK, MFMA_VALU_COEXEC, 2>();
        __builtin_amdgcn_sched_barrier(0);
        gemm_nt(s3, kv_c[3], kvs_c, opus::number<3>{});
        relu_nt(s2);
        sched_mfma_pairs<MT, VALU_MASK, MFMA_VALU_COEXEC, 3>();
        __builtin_amdgcn_sched_barrier(0);
        issue_ks(kv_c, kvs_c, pg_c);                       // reload kv_c with chunk pf_tile (page prefetched)
        pg_c = load_page_id(clamp_tile(pf_tile + 2));      // prefetch page id for kv_c's next reload
        gemm_nt(a0_out, kv_p[0], kvs_p, opus::number<0>{});// kv_p's nt=0 (loaded a phase ago) -> carry
        relu_nt(s3);
        sched_mfma_pairs<MT, VALU_MASK, MFMA_VALU_COEXEC, 4>();
        __builtin_amdgcn_sched_barrier(0);
        // batched reduce of ALL 4 nt (nt0=a0_in carried, 1=s1, 2=s2, 3=s3): clean, no s_nop / no v_mov.
        reduce_all(a0_in, s1, s2, s3, out_cur);
    };
    // Lagged store: emitted right after the NEXT phase's compute, so out_prev (computed one phase
    // ago) is stored while the current phase's MFMAs / KV reload are in flight (out double-buffered).
    auto lagged_store = [&](opus::vector_t<float, NTPW>& out_prev, int t_prev) {
        __builtin_amdgcn_sched_barrier(0);
        do_store(out_prev, t_prev);
        __builtin_amdgcn_sched_barrier(0);
    };

    kv_nt_t kvA[NTPW], kvB[NTPW];
    int     kvsA, kvsB;
    sfrag   acc0A, acc0B;
    int     pgA, pgB;
    opus::vector_t<float, NTPW> outA, outB;   // double-buffered output (phase i even->outA, odd->outB)

    // ---- prologue: chunk-0 -> kvA (+nt=0 -> acc0A), chunk-1 -> kvB; page ids for first reloads (2,3). ----
    issue_ks(kvA, kvsA, load_page_id(clamp_tile(0)));
    gemm_nt(acc0A, kvA[0], kvsA, opus::number<0>{});
    issue_ks(kvB, kvsB, load_page_id(clamp_tile(1)));
    pgA = load_page_id(clamp_tile(2));
    pgB = load_page_id(clamp_tile(3));
    __builtin_amdgcn_sched_barrier(0);

    // ---- PEELED phase 0 (chunk 0 -> outA), no store yet (stored during phase 1). ----
    compute_phase(kvA, kvsA, acc0A, kvB, kvsB, acc0B, pgA, 2, outA);
    __builtin_amdgcn_sched_barrier(0);

    // ---- PEELED phase 1 (chunk 1 -> outB), store outA(chunk 0). ----
    if (tile_count >= 2) {
        compute_phase(kvB, kvsB, acc0B, kvA, kvsA, acc0A, pgB, 3, outB);
        lagged_store(outA, 0);
    }

    // ---- main loop: 2 chunks / iter. Each compute stores the OTHER buffer from the prior phase. ----
    int t = 2;
    for (; t + 1 < tile_count; t += 2) {
        compute_phase(kvA, kvsA, acc0A,  kvB, kvsB, acc0B, pgA, t + 2, outA);
        lagged_store(outB, t - 1);
        compute_phase(kvB, kvsB, acc0B, kvA, kvsA, acc0A,  pgB, t + 3, outB);
        lagged_store(outA, t);
    }
    if (t < tile_count) {   // leftover even chunk t (kvA -> outA), store the prior odd chunk (outB).
        compute_phase(kvA, kvsA, acc0A, kvB, kvsB, acc0B, pgA, t + 2, outA);
        lagged_store(outB, t - 1);
    }

    // ---- epilogue: store the last computed chunk (never lagged into a following phase). ----
    const int last = tile_count - 1;
    if ((last & 1) == 0) do_store(outA, last);
    else                 do_store(outB, last);
}

} // namespace opus_logits
#endif

#endif  // PA_MQA_LOGITS_MXFP8_IMPL
