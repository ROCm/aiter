// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// MXFP4 paged MQA logits (gfx950), OPUS implementation.
//
// Semantics (per query row r, window [s, e)):
//   out[r, s:e] = sum_H( relu(Q[r] . Kᵀ) * weight[r] ) * weight_scale
//
// Same op as pa_mqa_logits_mxfp8_opus.h with FP4 (E2M1) data instead of FP8
// (E4M3); the E8M0 block-scale path (block=32) is identical. This is a thin
// launcher: Q/KV must already be MXFP4-quantized and preshuffled into the kernel
// ABI (byte-identical to the FlyDSL fp4 ABI). The per-CTA assignment is derived
// entirely in-kernel from blockIdx + per-row windows / per-batch metadata
// (schedule-free, cudagraph-safe); quant/preshuffle is the caller's responsibility.
//
// Deliberately a separate kernel from the fp8 one rather than a shared template:
// the fp8 kernel sits at the VGPR=218 / occupancy-2 ceiling, while fp4 runs at
// VGPR=162 / occupancy-3 with zero LDS and is tuned independently.

#pragma once
#include "aiter_tensor.h"
#include <cstdint>

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

// PREFILL launch: 1D grid (num_rows), one CTA per query row covering its whole
// window, read from per-row [num_rows] int32 arrays (row_to_batch/local_starts/
// local_ends, e.g. built by pa_mqa_logits_mxfp4_prefill_windows). No context split.
//
// block_k picks the compiled variant: 256 -> 4-wave, 64 -> 1-wave. It is a
// caller-supplied performance knob, not a shape: both variants produce identical
// results. See the python wrapper for which one to pick.
void pa_mqa_logits_mxfp4_fwd_prefill(aiter_tensor_t& q,
                          aiter_tensor_t& q_scale,
                          aiter_tensor_t& kv_cache,
                          aiter_tensor_t& kv_scale,
                          aiter_tensor_t& block_tables,
                          aiter_tensor_t& weights,
                          aiter_tensor_t& row_to_batch,
                          aiter_tensor_t& local_starts,
                          aiter_tensor_t& local_ends,
                          aiter_tensor_t& out,
                          int num_rows,
                          float weight_scale,
                          int block_k,
                          int kv_block_size,
                          int max_seq_len);

// DECODE launch: 3D grid (batch, next_n_max, split_kv). Q/weights/out PACKED
// ([total_q, ...]); the packed row (cu_seq_q[batch]+n) and MTP tail-causal window
// are derived inline from per-batch cu_seq_q / context_lens (no window arrays).
// cudagraph-safe (grid from static shapes; context_lens only read in-kernel).
void pa_mqa_logits_mxfp4_fwd_decode(aiter_tensor_t& q,
                          aiter_tensor_t& q_scale,
                          aiter_tensor_t& kv_cache,
                          aiter_tensor_t& kv_scale,
                          aiter_tensor_t& block_tables,
                          aiter_tensor_t& weights,
                          aiter_tensor_t& cu_seq_q,
                          aiter_tensor_t& context_lens,
                          aiter_tensor_t& out,
                          int batch,
                          int next_n_max,
                          int split_kv,
                          float weight_scale,
                          int block_k,
                          int kv_block_size,
                          int max_seq_len);

// Build the per-row [local_start, local_end) window arrays the PREFILL launch
// consumes, from cu_seq_q [B+1] + context_lens [B] (MTP tail-causal: batch b's
// n-th row sees [0, context_len[b] - (qlen-1-n)); plain causal when qlen == ctx).
// Outputs row_to_batch / local_starts / local_ends, each [total_q] int32. Device-side.
//
// Identical rule to pa_mqa_logits_mxfp8_prefill_windows -- the window arrays carry
// no element type, so either builder feeds either kernel.
void pa_mqa_logits_mxfp4_prefill_windows(aiter_tensor_t& cu_seq_q,
                                      aiter_tensor_t& context_lens,
                                      aiter_tensor_t& row_to_batch,
                                      aiter_tensor_t& local_starts,
                                      aiter_tensor_t& local_ends,
                                      int total_q);

#ifdef PA_MQA_LOGITS_MXFP4_IMPL
// ==== Implementation (compiled only in the .cu TU): kargs + traits + kernel. ====
//
// The kargs struct and the mqa_logits_sched enum below are byte-identical to the
// fp8 header's -- the ABI does not depend on the element width. They are spelled
// out again rather than shared because each header is self-contained behind its
// own _IMPL guard, and no TU defines both.

using mqa_logits_bf16_t = __bf16;

// Kernel arguments. Pointers are opaque; the kernel reinterprets per its ABI.
struct opus_mqa_logits_kargs {
    const void* __restrict__ ptr_q;         // [total_tokens, H, D/2]                        fp4 (E2M1, 2/byte)
    const void* __restrict__ ptr_q_scale;   // [total_tokens, K_TILES, K_CHUNKS, 16, QS_PAD] e8m0
    const void* __restrict__ ptr_kv;        // [num_blocks, K_TILES, 4, PAGE, 16]            fp4 (E2M1, 2/byte)
    const void* __restrict__ ptr_kv_scale;  // [num_blocks, K_TILES, K_CHUNKS, PAGE]         e8m0
    const int*  __restrict__ ptr_block_tables; // [batch, max_blocks_per_seq] int32
    const void* __restrict__ ptr_weights;   // [total_tokens, H] bf16
    float* __restrict__ ptr_out;             // [total_tokens, max_seq_len] fp32

    // Prefill (SCHED Prefill): per-row window arrays, each [num_rows] int32.
    const int* __restrict__ ptr_row_to_batch;
    const int* __restrict__ ptr_local_starts;
    const int* __restrict__ ptr_local_ends;
    // Decode (SCHED Decode): per-batch arrays; windows derived inline in-kernel.
    const int* __restrict__ ptr_cu_seq_q;       // [batch+1] int32 (packed qlen prefix sum)
    const int* __restrict__ ptr_context_lens;   // [batch]   int32
    int   split_kv;            // context splits per row (>= 1); SCHED Decode only
    int   num_rows;            // total query rows (prefill grid.x)

    int   max_seq_len;
    int   stride_out_row;      // out row stride in elements (== max_seq_len for dense out)
    float weight_scale;
    int   block_k;             // KV tile size (== Traits::KV_TILE_SIZE)
    int   kv_block_size;       // paged block (page) size (== Traits::PAGE_SIZE)
    int   max_blocks_per_seq;  // block_tables row stride
};

// UNITS WARNING (the whole reason the fp4 traits are separate from the fp8 ones):
//   opus gmem offsets are BYTES (see gmem::_load, "os in unit of byte"). Under fp8 one element
//   is one byte, so the fp8 traits could use element counts and byte counts interchangeably.
//   Under fp4 an element is HALF a byte, so every stride below is spelled out in bytes and the
//   element counts are kept separate. Do not reintroduce a `sizeof(D_DATA)`-style ELEM_BYTES:
//   opus's fp4_t is one logical element with sizeof()==1 but sizeof_bits()==4, so sizeof would
//   silently give 2x every stride.
//
// D_DATA is deliberately NOT declared here: fp4 is opus::fp4_t, a device-side packed type, and
// this struct must stay host-compilable (the launcher reads BLOCK_SIZE / KV_TILE_SIZE etc.).
// The kernel template aliases it.
template<int KV_TILE_SIZE_ = 256,
         int PAGE_SIZE_     = 64,
         int HEAD_DIM_      = 128,
         int N_HEADS_       = 64,
         int NUM_WARPS_     = 4>
struct opus_mqa_logits_fp4_traits {
    static constexpr int KV_TILE_SIZE = KV_TILE_SIZE_;  // block_k
    static constexpr int PAGE_SIZE    = PAGE_SIZE_;     // kv_block_size
    static constexpr int HEAD_DIM     = HEAD_DIM_;
    static constexpr int N_HEADS      = N_HEADS_;
    static constexpr int NUM_WARPS    = NUM_WARPS_;

    static constexpr int WARP_SIZE  = 64;
    static constexpr int BLOCK_SIZE = NUM_WARPS * WARP_SIZE;

    // Non-D_DATA type aliases (identical to fp8).
    using D_WEIGHT = mqa_logits_bf16_t;   // per-head weights
    using D_ACC    = float;               // MFMA accumulator (C)
    using D_OUT    = float;               // output logits
    using D_SCALE  = int;                 // E8M0 blockscale, packed as an int32 dword per lane

    // MFMA: scaled f8f6f4, 16x16x128. Same instruction as fp8; opus picks format code 4 (fp4)
    // from dtype_a/dtype_b, so no cbsz/blgp plumbing is needed here.
    static constexpr int MFMA_M = 16;
    static constexpr int MFMA_N = 16;
    static constexpr int MFMA_K = 128;

    static constexpr int ELEM_BITS = 4;                       // fp4: half a byte per element
    static constexpr int SCALE_BLOCK = 32;                    // E8M0 blockscale granularity
    static constexpr int K_CHUNKS    = MFMA_K / SCALE_BLOCK;  // 4 scale blocks along K=128

    static constexpr int M_TILES = N_HEADS / MFMA_M;   // 4  (head tiles along M)
    static constexpr int K_TILES = HEAD_DIM / MFMA_K;  // 1  (outer K loop)

    static constexpr int N_TOTAL_TILES   = KV_TILE_SIZE / MFMA_N;      // 16
    static constexpr int N_TILES         = N_TOTAL_TILES / NUM_WARPS;  // 4 (NTPW)
    static constexpr int TILES_PER_BLOCK = PAGE_SIZE / MFMA_N;         // 4 (MFMA_N tiles per page)
    static constexpr int N_PHYS = (N_TILES + TILES_PER_BLOCK - 1) / TILES_PER_BLOCK; // 1

    static constexpr int DWORDx4_BYTES = 16;

    // ---- fp4 register layout (THE difference vs fp8) ----
    // MI350 fp4: lane g (=lane/16) holds ONE contiguous 32-K block, K[g*32 : +32] = 32 fp4
    // = 16 bytes in vgpr0-3 (i32x4). fp8 instead holds two 16-K groups (K[g*16:+16] and
    // K[64+g*16:+16]) = 32 bytes in vgpr0-7. So fp4 does a single 16B load per lane and the
    // fp8 traits VGRP_PER_LANE / VGRP_K_OFFSET / VGRP_CHUNK_HI have no fp4 counterpart.
    // Bonus: a lane's 32-K data block coincides exactly with its 32-K scale block, so the
    // data chunk index and the scale block index are the same `g`.
    static constexpr int KV_GRP_ELEMS = SCALE_BLOCK;                  // 32 fp4 per lane
    static constexpr int KV_GRP_BYTES = KV_GRP_ELEMS * ELEM_BITS / 8; // 16 bytes
    // PAYLOAD bytes a lane actually loads. Note this is NOT the MFMA operand register width:
    // opus always passes 256-bit (i32x8, 32 B) operands regardless of element type, and fp4
    // (format code 4) reads only the low 16 B -- vgpr4-7 are ignored by the hardware. So the
    // kernel loads 16 B and fills the low half of a 32 B operand.
    static constexpr int A_BYTES_PER_LANE = (MFMA_M * MFMA_K / WARP_SIZE) * ELEM_BITS / 8; // 16

    static constexpr int C_FRAG        = MFMA_M * MFMA_N / WARP_SIZE; // 4: C frag rows per lane per m-tile
    static constexpr int LANE_M_GROUPS = WARP_SIZE / MFMA_M;          // 4: 16-lane groups per wave

    // q_scale preshuffle: [T, K_TILES, 4, 16, QS_PAD]; QS_PAD = round_up(M_TILES,4).
    // IDENTICAL to fp8 -- the scale ABI does not depend on the data width.
    static constexpr int QS_DW  = (M_TILES + 3) / 4;   // 1
    static constexpr int QS_PAD = QS_DW * 4;           // 4

    // ---- Q byte geometry (direct global read; no LDS staging in this variant) ----
    // q is natural [T, H, D/2] fp4 => D/2 bytes per head row.
    static constexpr int Q_ROW_BYTES  = HEAD_DIM * ELEM_BITS / 8;        // 64
    static constexpr int Q_TILE_BYTES = MFMA_M * Q_ROW_BYTES;            // 1024 (16 heads)

    // ---- byte strides for the paged KV cache ----
    // Layout [num_blocks, K_TILES, K_CHUNKS(4), PAGE, 16]: chunk g holds the contiguous 32-K
    // block K[g*32 : +32] as 16 packed bytes, matching FlyDSL's fp4 writer
    //   kv_cache[p, kt, kc, o, :] = 16 bytes for K[(kt*4+kc)*32 : +32]
    // (fp8 uses 8 chunks of 16 K each instead.) `page` stays after the chunk so consecutive
    // tokens are contiguous.
    static constexpr int KCHUNK_PER_TILE = MFMA_K / SCALE_BLOCK;         // 4
    static constexpr int stride_kv_chunk = PAGE_SIZE * KV_GRP_BYTES;     // 1024: one chunk across a page
    static constexpr int stride_kv_ktile = KCHUNK_PER_TILE * stride_kv_chunk;
    static constexpr int stride_kv_block = K_TILES * stride_kv_ktile;
    // ---- byte strides for kv_scale [num_blocks, K_TILES, 4, PAGE] (e8m0, 1 byte) ----
    // Identical to fp8.
    static constexpr int stride_kvs_ktile = K_CHUNKS * PAGE_SIZE;
    static constexpr int stride_kvs_block = K_TILES  * stride_kvs_ktile;

    static_assert(ELEM_BITS == 4, "this traits set is fp4-only");
    static_assert(HEAD_DIM % MFMA_K == 0, "HEAD_DIM must be a multiple of MFMA_K (128)");
    static_assert(N_HEADS % MFMA_M == 0, "N_HEADS must be a multiple of MFMA_M (16)");
    static_assert(M_TILES <= 8, "M_TILES > 8 unsupported (use N_HEADS <= 128)");
    static_assert(KV_TILE_SIZE % MFMA_N == 0, "KV_TILE must be a multiple of MFMA_N");
    static_assert(N_TOTAL_TILES % NUM_WARPS == 0, "N_TOTAL_TILES must be a multiple of NUM_WARPS");
    static_assert(PAGE_SIZE % MFMA_N == 0, "PAGE must be a multiple of MFMA_N");
    static_assert(KV_TILE_SIZE % PAGE_SIZE == 0, "KV_TILE must be a multiple of PAGE");
    static_assert(N_TILES == 4, "kernel currently assumes N_TILES == 4");
    static_assert(N_PHYS == 1, "kernel currently assumes N_PHYS == 1 (NTPW tiles share one page)");
    static_assert(KV_GRP_BYTES == 16, "a lane's fp4 block must be exactly one 16B dwordx4");
    static_assert(A_BYTES_PER_LANE == 16, "fp4 payload per lane is 16 B (low half of the operand)");
};

namespace opus_logits {
enum class mqa_logits_sched {
    Prefill,
    Decode,
};
}

#ifndef __HIP_DEVICE_COMPILE__
// Host pass: empty stub so the __device_stub__ symbol resolves for the launcher.
namespace opus_logits {
template<class T, mqa_logits_sched SCHED = mqa_logits_sched::Prefill>
__global__ void pa_mqa_logits_mxfp4_kernel(opus_mqa_logits_kargs) {}
}
#else
// ============================================================================
// Device kernel, ported verbatim from
// gcnasm/opus_logits/mqa_logits_mxfp4_kernel_template.hpp.
//
// That file is itself derived from the fp8 kernel: the compute core (scaled MFMA
// sequence, relu, batched packed reduce, permlane head reduce, out double buffer
// and the whole software pipeline) is a copy and only the DATA path differs.
// What differs from fp8:
//  1. Register layout. MI350 fp4 gives each lane ONE contiguous 32-K block
//     K[g*32:+32] = 32 fp4 = 16 B (vgpr0-3 / i32x4). fp8 instead gives two 16-K
//     groups = 32 B (vgpr0-7 / i32x8). So every lane does a SINGLE 16 B load and
//     the fp8 VGRP_PER_LANE / VGRP_K_OFFSET / VGRP_CHUNK_HI dimension disappears.
//  2. Byte-addressed gmem. opus gmem offsets are bytes and an fp4 element is half
//     a byte, so a layout in element units cannot express the stride. Both Q and
//     KV are read through uint8_t gmem with byte strides.
//  3. Q is read straight from global (no LDS staging), so this kernel uses zero
//     LDS. fp8's async-Q geometry assumes a head row fills a 128 B async granule;
//     an fp4 head row is only 64 B. Re-adding async-Q for fp4 needs a 64 B granule
//     and is left as a follow-up.
//  4. The scale path is UNCHANGED (E8M0 per 32-K block, op_sel byte select).
// ============================================================================
#include <opus/opus.hpp>

// Minimum waves/SIMD hint for __launch_bounds__ (see the fp8 header for the reasoning).
#ifndef OPUS_LOGITS_MIN_WAVES
#define OPUS_LOGITS_MIN_WAVES 2
#endif

namespace opus_logits {

using opus::operator""_I;

// ─── scheduling helpers (copies of the fp8 ones, suffixed to avoid an ODR clash) ───
constexpr int VALU_MASK_FP4 = 0x02;   // v_max / v_add / v_pk_fma ... (no MFMA/TRANS/mem)
constexpr int MFMA_MASK_FP4 = 0x08;   // v_mfma_*

// Interleave `Pairs` (MFMA, OTHER) groups so the VALU ops land in the MFMA shadow.
template<int Pairs, int OTHER_MASK, int CNT, int Group>
__device__ inline void sched_mfma_pairs_fp4() {
    opus::static_for<Pairs>([&](auto) {
        __builtin_amdgcn_sched_group_barrier(MFMA_MASK_FP4, 1, Group);
        __builtin_amdgcn_sched_group_barrier(OTHER_MASK, CNT, Group);
    });
}

// Force `v` to be available in a scalar register at this exact point in the program. The asm
// emits nothing; being an in-out operand it cannot be moved, so the s_load that produced `v`
// cannot sink past it. Left alone the sinker pushes every kernarg load down to its first use,
// which lands them on the far side of the early-exit branches and turns what should be one
// scalar-cache round trip into four (Prefill) or five (Decode) serialized ones ahead of the
// prologue -- worth ~7% on decode.
//
// THIS IS LOAD-BEARING. Constraint choice is not free here, and both alternatives are worse:
//   - read-only (`asm volatile("" :: "s"(v))`) makes the asm a memory clobber, so the indexed
//     loads further down can no longer be proven noclobber and decay from s_load to
//     global_load + v_readfirstlane (VGPR 162 -> 170, occupancy 3 -> 2).
//   - pinning a POINTER fails outright: LLVM calls an inline-asm result divergent, which
//     propagates into every address derived from it (VGPR 162 -> 178, occupancy 3 -> 2).
// So only scalars are pinned, with an in-out constraint.
template<class X>
__device__ inline void pin_sgpr(X& v) { asm("" : "+s"(v)); }

// Sum a per-lane value across the LANE_M_GROUPS 16-lane groups (lane^32, then lane^16).
// Verbatim from the fp8 kernel: the v_mov seeds b with a copy of a, then the swap leaves
// a=self / b=partner, so a+b is the butterfly add. Do NOT switch to
// __builtin_amdgcn_permlane16_swap here -- the builtin mis-lowers (one swap result gets
// dropped), which silently scales the head reduction. The asm form is the proven path.
__device__ inline float permlane_head_reduce_fp4(float v) {
    int a = __builtin_bit_cast(int, v), b = a;
    asm("v_mov_b32_e32 %0, %1\n\t" : "=v"(a) : "v"(b));
    asm("v_permlane32_swap_b32 %0, %1\n\t" : "+v"(a), "+v"(b));
    v = __builtin_bit_cast(float, a) + __builtin_bit_cast(float, b);   // += lane^32
    a = __builtin_bit_cast(int, v); b = a;
    asm("v_mov_b32_e32 %0, %1\n\t" : "=v"(a) : "v"(b));
    asm("v_permlane16_swap_b32 %0, %1\n\t" : "+v"(a), "+v"(b));
    return __builtin_bit_cast(float, a) + __builtin_bit_cast(float, b); // += lane^16
}

// ─── Q partition layout: direct global read, BYTE strides ───────────────────
// q is natural [T, H, D/2] fp4 -> Q_ROW_BYTES (=D/2) per head row, no data preshuffle.
// Lane (lane_mod_16 = head row within tile, lane_div_16 = g) reads the 16 B holding
// K[g*32 : +32] of head (mi*16 + lane_mod_16). The y dims are m-tile (outer, 4 issues)
// and the 16 contiguous bytes.
template<class T>
__device__ inline auto make_layout_q_fp4(int lane_mod_16, int lane_div_16) {
    constexpr auto shape = opus::make_tuple(
        opus::number<T::M_TILES>{},        // m-tile            (y, outer)
        opus::number<T::MFMA_M>{},         // head row in tile  (p: lane_mod_16)
        opus::number<T::K_CHUNKS>{},       // 32-K block g      (p: lane_div_16)
        opus::number<T::KV_GRP_BYTES>{});  // 16 B = 32 fp4     (y, contiguous)
    constexpr auto dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}),                    // m-tile
        opus::make_tuple(opus::p_dim{}),                    // head row
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}));    // g + bytes (contiguous)
    return opus::make_layout(
        shape,
        opus::unfold_x_stride(dim, shape, opus::make_tuple(
            opus::number<T::MFMA_M * T::Q_ROW_BYTES>{},  // m-tile: 16 heads
            opus::number<T::Q_ROW_BYTES>{},              // head row
            opus::number<1>{})),                         // (g, byte) base=1 -> g=KV_GRP_BYTES(16), byte=1
        opus::unfold_p_coord(dim, opus::make_tuple(lane_mod_16, lane_div_16)));
}

// ─── KV partition layout (one MFMA_N token-tile), BYTE strides ──────────────
// kv_cache [num_blocks, K_TILES, K_CHUNKS(4), PAGE, 16]: chunk g = the 32-K block
// K[g*32:+32] as 16 packed bytes. Single group per lane (vs fp8's c=g and c=g+4).
template<class T>
__device__ inline auto make_layout_kv_nt_fp4(int lane_mod_16, int lane_div_16) {
    constexpr auto shape = opus::make_tuple(
        opus::number<T::MFMA_N>{},         // token o in page  (p: lane_mod_16)
        opus::number<T::K_CHUNKS>{},       // chunk g          (p: lane_div_16)
        opus::number<T::KV_GRP_BYTES>{});  // 16 B = 32 fp4    (y, contiguous)
    constexpr auto dim = opus::make_tuple(
        opus::make_tuple(opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}));
    return opus::make_layout(
        shape,
        opus::unfold_x_stride(dim, shape, opus::make_tuple(
            opus::number<T::KV_GRP_BYTES>{},      // token: consecutive tokens are 16 B apart
            opus::number<T::stride_kv_chunk>{},   // chunk g: one chunk across the page
            opus::number<1>{})),                  // bytes
        opus::unfold_p_coord(dim, opus::make_tuple(lane_mod_16, lane_div_16)));
}

// ─── Scale word (q_scale / kv_scale) — IDENTICAL to fp8 ────────────────────
// One E8M0 dword per lane; the 4 bytes pack the 4 m-tiles (q) / token-tiles (kv) and MFMA
// op_sel selects the byte. Trailing size-1 y-dim provides the single load issue.
template<class T>
__device__ inline auto make_layout_scale_fp4(int lane_div_16, int lane_mod_16) {
    constexpr auto shape = opus::make_tuple(
        opus::number<T::K_CHUNKS>{},   // block g         (p: lane_div_16)
        opus::number<T::MFMA_M>{},     // row/col in tile (p: lane_mod_16)
        opus::number<1>{});            // single dword    (y)
    constexpr auto dim = opus::make_tuple(
        opus::make_tuple(opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}));
    return opus::make_layout(
        shape,
        opus::unfold_x_stride(dim, shape, opus::make_tuple(
            opus::number<1>{},          // (g, lane) base=1 -> g=MFMA_M(16), lane=1
            opus::number<0>{})),
        opus::unfold_p_coord(dim, opus::make_tuple(lane_div_16, lane_mod_16)));
}

// ─── block_tables / weights / out layouts — IDENTICAL to fp8 ───────────────
template<class T>
__device__ inline auto make_layout_bt_fp4(int warp_id) {
    constexpr auto shape = opus::make_tuple(
        opus::number<T::NUM_WARPS>{},  // page-in-chunk = warp id (p)
        opus::number<1>{});            // single int              (y)
    constexpr auto dim = opus::make_tuple(
        opus::make_tuple(opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}));
    return opus::make_layout(
        shape,
        opus::unfold_x_stride(dim, shape, opus::make_tuple(
            opus::number<1>{},
            opus::number<0>{})),
        opus::unfold_p_coord(dim, opus::make_tuple(warp_id)));
}

// Weights: bf16 [T,H]; row folded into base. Per m-tile 4 contiguous bf16 at
// head0 = mi*16 + lane_div_16*4.
template<class T>
__device__ inline auto make_layout_w_fp4(int lane_div_16) {
    constexpr auto shape = opus::make_tuple(
        opus::number<T::M_TILES>{},        // m-tile          (y, outer)
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

// ─── kernel ────────────────────────────────────────────────────────────────
// SCHED (mqa_logits_sched): Prefill = 1D grid, per-row window arrays;
//                           Decode  = 3D grid (batch, next_n_max, split_kv), inline MTP windows.
template<class T, mqa_logits_sched SCHED = mqa_logits_sched::Prefill>
__global__ __launch_bounds__(T::BLOCK_SIZE, OPUS_LOGITS_MIN_WAVES) void pa_mqa_logits_mxfp4_kernel(opus_mqa_logits_kargs kargs) {
    // ---- data-type aliases ----
    using D_DATA   = opus::fp4_t;              // MFMA A/B element (E2M1); opus picks format code 4
    using D_BYTE   = uint8_t;                  // gmem scalar: fp4 is addressed in bytes
    using D_WEIGHT = typename T::D_WEIGHT;
    using D_SCALE  = typename T::D_SCALE;
    using D_ACC    = typename T::D_ACC;
    using D_OUT    = typename T::D_OUT;

    constexpr int WARP     = T::WARP_SIZE;          // 64
    constexpr int NTPW     = T::N_TILES;            // 4
    constexpr int MT       = T::M_TILES;            // 4
    constexpr int PAGE     = T::PAGE_SIZE;          // 64
    constexpr int N_HEADS  = T::N_HEADS;            // 64
    // mxfp4 scaled MFMA can co-exec with at most this many VALU in its shadow (mha idiom).
    constexpr int MFMA_VALU_COEXEC = 4;

    // ---- kernel arguments: one scalar round trip ----
    // Every kernarg field the kernel needs is read HERE, ahead of the early-exit branches
    // below, and pinned so it stays here. All of it comes off the same kernarg base and none of
    // it depends on anything, so it can all be in flight at once; a branch in between is what
    // turns it into several dependent trips through the scalar cache. Pinned, it issues as one
    // batch at cycle 0 behind a single `s_waitcnt lgkmcnt(0)`, which is worth ~+7% on decode
    // and ~+1% on prefill: prefill saturates the machine so the latency hides, while decode's
    // short shapes sit on a launch floor where the prologue is the critical path.
    //
    // Reading the fields into locals is NOT enough on its own: without the pins the sinker
    // moves every load back down to its first use and the ISA comes out byte-identical.
    // SGPRs are not an occupancy limiter on gfx9+, so holding the block live is free.
    const void* p_q        = kargs.ptr_q;
    const void* p_q_scale  = kargs.ptr_q_scale;
    const void* p_kv       = kargs.ptr_kv;
    const void* p_kv_scale = kargs.ptr_kv_scale;
    const int*  p_bt       = kargs.ptr_block_tables;
    const void* p_weights  = kargs.ptr_weights;
    float*      p_out      = kargs.ptr_out;
    int   stride_out   = kargs.stride_out_row;
    float weight_scale = kargs.weight_scale;
    int   block_k = kargs.block_k;                  // 256 (4-wave) / 64 (1-wave)
    int   max_blk = kargs.max_blocks_per_seq;
    pin_sgpr(p_q); pin_sgpr(p_q_scale); pin_sgpr(p_kv); pin_sgpr(p_kv_scale);
    pin_sgpr(p_bt); pin_sgpr(p_weights); pin_sgpr(p_out);
    pin_sgpr(stride_out); pin_sgpr(weight_scale); pin_sgpr(block_k); pin_sgpr(max_blk);

    // The index-array pointers are deliberately NOT pinned. Doing so does fold them into the
    // group above (3 round trips -> 2), but pinning a pointer propagates LLVM's inline-asm
    // divergence marking into every address derived from it and costs VGPR 162 -> 178,
    // occupancy 3 -> 2. Their round trip is a scalar-cache hit anyway: the group above has
    // already pulled both 64 B lines of the kernarg block in, so what is left is a cheap
    // re-read, not the cold miss that the merge above removed.
    const int* p_row_to_batch = kargs.ptr_row_to_batch;
    const int* p_local_starts = kargs.ptr_local_starts;
    const int* p_local_ends   = kargs.ptr_local_ends;
    const int* p_cu_seq_q     = kargs.ptr_cu_seq_q;
    const int* p_context_lens = kargs.ptr_context_lens;
    // Decode's split count is the last kernarg scalar read on that path; pinned so it rides in
    // the group above instead of trailing a round trip of its own after the mtp_pos branch.
    int split_kv = kargs.split_kv;
    if constexpr(SCHED == mqa_logits_sched::Decode) { pin_sgpr(split_kv); }

    const int tid = opus::thread_id_x();
    const int warp_id     = tid >> 6;
    const int lane_id     = tid & (WARP - 1);
    const int lane_mod_16 = lane_id & (T::MFMA_M - 1);
    const int lane_div_16 = (lane_id >> 4) & (T::LANE_M_GROUPS - 1);

    // ---- per-CTA assignment (uniform across block) ----
    // Identical to the fp8 kernel: everything derived from blockIdx, no host-side cta_info.
    //
    // The indexed arrays are a pointer chase off the kernarg block, so they are the one
    // genuinely dependent round trip left. Within a schedule they are all indexed the same way
    // and issue as one group.
    int row_id, batch_id, chunk_start, tile_count, local_start, local_end;
    constexpr int kv_tile_size = T::KV_TILE_SIZE;
    if constexpr(SCHED == mqa_logits_sched::Prefill) {
        // 1D grid: one CTA per query row, covering its whole ragged window (no split).
        const int query_row = opus::block_id_x();
        // num_rows came up with the kernarg block above, so this branch costs no round trip of
        // its own. Do NOT hoist the reads below it behind a clamped index: making the index
        // conditional is enough for LLVM to call it divergent, which turns these three
        // s_loads into global_load + v_readfirstlane and is far worse than the branch.
        if(query_row >= kargs.num_rows) return;
        row_id      = query_row;
        batch_id    = p_row_to_batch[query_row];
        local_start = p_local_starts[query_row];
        local_end   = p_local_ends[query_row];
        pin_sgpr(batch_id);
        const int first_kv_tile = local_start / kv_tile_size;
        const int end_kv_tile   = (local_end > 0) ? ((local_end + kv_tile_size - 1) / kv_tile_size) : 0;
        chunk_start = first_kv_tile;
        tile_count  = end_kv_tile - first_kv_tile;
    } else {
        // 3D grid (batch, next_n_max, split_kv): packed row + MTP tail-causal window.
        const int num_splits = split_kv;
        const int batch      = opus::block_id_x();
        const int mtp_pos    = opus::block_id_y();
        const int split_idx  = opus::block_id_z();
        // All three are indexed by `batch` alone, so they issue as one group. `batch` is
        // grid.x, hence always in bounds — no clamp needed on this path.
        int q_start = p_cu_seq_q[batch];
        int q_next  = p_cu_seq_q[batch + 1];
        int ctx_len = p_context_lens[batch];
        pin_sgpr(ctx_len);
        const int qlen = q_next - q_start;
        if(mtp_pos >= qlen) return;
        row_id      = q_start + mtp_pos;
        batch_id    = batch;
        local_start = 0;
        local_end   = ctx_len - (qlen - 1 - mtp_pos);
        const int window_tiles    = (local_end > 0) ? ((local_end + kv_tile_size - 1) / kv_tile_size) : 0;
        const int tiles_per_split = window_tiles / num_splits;
        const int remainder       = window_tiles - tiles_per_split * num_splits;
        const int my_first_tile   = split_idx * tiles_per_split + (split_idx < remainder ? split_idx : remainder);
        chunk_start = my_first_tile;
        tile_count  = (my_first_tile < window_tiles) ? (tiles_per_split + (split_idx < remainder ? 1 : 0)) : 0;
    }

    // ---- GEMM object (scaled fp4 16x16x128) ----
    auto mma = opus::make_tiled_mma<D_DATA, D_DATA, D_ACC>(
        opus::seq<MT, NTPW, T::K_TILES>{},
        opus::seq<1, T::NUM_WARPS, 1>{},
        opus::seq<T::MFMA_M, T::MFMA_N, T::MFMA_K>{});
    using Mma = typename decltype(mma)::MMA;
    Mma mma_op;
    // opus always hands the scaled MFMA a 256-bit operand regardless of element type; fp4
    // (format code 4) consumes only the low 16 B, so we load 16 B and leave the high half alone.
    static_assert(sizeof(typename Mma::vtype_a) == 2 * T::A_BYTES_PER_LANE,
                  "expected a 256-bit MFMA operand with the fp4 payload in its low half");

    // ---- buffer resources (row folded into base pointers to keep voffsets small) ----
    // Q/KV are byte-addressed (fp4): Q row stride is N_HEADS*Q_ROW_BYTES bytes.
    const D_BYTE*   q_base  = reinterpret_cast<const D_BYTE*>(p_q) + (size_t)row_id * N_HEADS * T::Q_ROW_BYTES;
    const D_SCALE*  qs_base = reinterpret_cast<const D_SCALE*>(p_q_scale) + (size_t)row_id * (T::K_CHUNKS * T::MFMA_M);
    const D_WEIGHT* w_base  = reinterpret_cast<const D_WEIGHT*>(p_weights) + (size_t)row_id * N_HEADS;
    const int*      bt_base = p_bt + (size_t)batch_id * max_blk;
    D_OUT*          out_base  = p_out + (size_t)row_id * stride_out;
    const unsigned  out_bytes = (unsigned)(local_end > 0 ? local_end : 0) * (unsigned)sizeof(D_OUT);

    auto g_q   = opus::make_gmem(q_base);
    auto g_qs  = opus::make_gmem(qs_base);
    auto g_kv  = opus::make_gmem(reinterpret_cast<const D_BYTE*>(p_kv));
    auto g_kvs = opus::make_gmem(reinterpret_cast<const D_SCALE*>(p_kv_scale));
    auto g_bt  = opus::make_gmem(bt_base);
    auto g_w   = opus::make_gmem(w_base);
    auto g_out = opus::make_gmem(out_base, out_bytes);   // OOB size == local_end: masks upper bound + neg-biased lanes

    // ---- Q / scale / weight partitions (loads deferred to the prologue) ----
    // Direct global read: MT issues x 16 B payload, each filling the low half of one m-tile's
    // 256-bit A operand. Zero-initialized so the ignored high half is never undefined.
    auto u_q = make_layout_q_fp4<T>(lane_mod_16, lane_div_16);
    typename Mma::vtype_a q_a[MT]{};

    auto u_qs = make_layout_scale_fp4<T>(lane_div_16, lane_mod_16);
    int q_scale;

    auto u_w = make_layout_w_fp4<T>(lane_div_16);
    opus::vector_t<float, T::C_FRAG> w_pl[MT];

    auto u_kv_nt = make_layout_kv_nt_fp4<T>(lane_mod_16, lane_div_16);
    auto u_kvs   = make_layout_scale_fp4<T>(lane_div_16, lane_mod_16);
    auto u_bt    = make_layout_bt_fp4<T>(warp_id);
    const int tok_lane_base = warp_id * (NTPW * T::MFMA_N) + lane_mod_16;
    const int lane_bias = (lane_div_16 != 0) ? -(chunk_start + tile_count) * block_k : 0;
    constexpr int pages_per_tile = T::KV_TILE_SIZE / PAGE;

    auto load_page_id  = [&](int tile_idx) {
        return opus::load<1>(g_bt, u_bt + (chunk_start + tile_idx) * pages_per_tile)[0];
    };
    auto load_kv_scale = [&](int page_id) {
        return opus::load<1>(g_kvs, u_kvs + page_id * (T::stride_kvs_block / (int)sizeof(int)))[0];
    };

    constexpr int EC = Mma::elem_c;   // per-(mi,nt) C fragment length (== 4)
    // One nt's KV operand: a 256-bit register whose low 16 B is the lane's 32-K fp4 block.
    using kv_nt_t = typename Mma::vtype_b;

    auto clamp_tile = [&](int t) { return t < tile_count ? t : tile_count - 1; };
    // Split KV load: NTPW independent per-nt loads of 16 B each (fp8 needs two 16 B groups).
    auto issue_ks = [&](kv_nt_t (&kv)[NTPW], int& kvs_w, int page_id) {
        kvs_w = load_kv_scale(page_id);
        opus::static_for<NTPW>([&](auto ntc) {
            constexpr int nt = ntc.value;
            auto v = opus::load<T::KV_GRP_BYTES>(
                g_kv, u_kv_nt + page_id * T::stride_kv_block + nt * (T::MFMA_N * T::KV_GRP_BYTES));
            opus::set_slice(kv[nt], v, opus::number<0>{}, opus::number<T::KV_GRP_BYTES>{});
        });
    };

    using sfrag = opus::vector_t<float, MT * EC>;
    auto gemm_nt = [&](sfrag& accs, kv_nt_t& kvnt, int kvs_w, auto ntc) {
        constexpr int nt = ntc.value;
        opus::static_for<MT>([&](auto mic) {
            constexpr int mi = mic.value;
            typename Mma::vtype_c c0{};
            auto acc = mma_op(q_a[mi], kvnt, c0, q_scale, kvs_w, opus::number<mi>{}, opus::number<nt>{});
            opus::set_slice(accs, acc, opus::number<mi * EC>{}, opus::number<mi * EC + EC>{});
        });
    };
    auto relu_nt = [&](sfrag& accs) {
        opus::static_for<MT * EC>([&](auto ic) {
            constexpr int i = ic.value;
            accs[i] = accs[i] > 0.0f ? accs[i] : 0.0f;
        });
    };

    auto reduce_all = [&](sfrag& s0, sfrag& s1, sfrag& s2, sfrag& s3, opus::vector_t<float, NTPW>& out) {
        sfrag* sp[NTPW] = { &s0, &s1, &s2, &s3 };
        opus::vector_t<float, 2> acc[NTPW];
        opus::static_for<NTPW>([&](auto ntc) { acc[ntc.value] = opus::vector_t<float, 2>{0.0f, 0.0f}; });
        opus::static_for<MT>([&](auto mic) {
            constexpr int mi = mic.value;
            opus::static_for<EC / 2>([&](auto pc) {
                constexpr int p = pc.value;
                opus::vector_t<float, 2> w{ w_pl[mi][2 * p], w_pl[mi][2 * p + 1] };
                opus::static_for<NTPW>([&](auto ntc) {
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
            out[nt] = permlane_head_reduce_fp4(ts);
        });
    };

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
        sched_mfma_pairs_fp4<MT, VALU_MASK_FP4, MFMA_VALU_COEXEC, 1>();
        __builtin_amdgcn_sched_barrier(0);
        gemm_nt(s2, kv_c[2], kvs_c, opus::number<2>{});
        relu_nt(s1);
        sched_mfma_pairs_fp4<MT, VALU_MASK_FP4, MFMA_VALU_COEXEC, 2>();
        __builtin_amdgcn_sched_barrier(0);
        gemm_nt(s3, kv_c[3], kvs_c, opus::number<3>{});
        relu_nt(s2);
        sched_mfma_pairs_fp4<MT, VALU_MASK_FP4, MFMA_VALU_COEXEC, 3>();
        __builtin_amdgcn_sched_barrier(0);
        issue_ks(kv_c, kvs_c, pg_c);
        pg_c = load_page_id(clamp_tile(pf_tile + 2));
        gemm_nt(a0_out, kv_p[0], kvs_p, opus::number<0>{});
        relu_nt(s3);
        sched_mfma_pairs_fp4<MT, VALU_MASK_FP4, MFMA_VALU_COEXEC, 4>();
        __builtin_amdgcn_sched_barrier(0);
        reduce_all(a0_in, s1, s2, s3, out_cur);
    };
    // do_store pinned between two barriers so the scheduler cannot sink the stores into the
    // surrounding MFMA groups (which would undo sched_mfma_pairs_fp4's interleaving).
    auto fenced_store = [&](opus::vector_t<float, NTPW>& out_v, int t) {
        __builtin_amdgcn_sched_barrier(0);
        do_store(out_v, t);
        __builtin_amdgcn_sched_barrier(0);
    };

    // Zero-initialized: fp4 fills only the low 16 B of each 256-bit operand, and leaving the
    // high half undefined lets garbage reach the MFMA.
    kv_nt_t kvA[NTPW]{}, kvB[NTPW]{};
    int     kvsA, kvsB;
    sfrag   acc0A, acc0B;
    int     pgA, pgB;
    opus::vector_t<float, NTPW> outA, outB;

    // ---- prologue: Q (direct global) + q_scale + weights, then chunk-0 -> kvA, chunk-1 -> kvB.
    // No LDS staging and hence no s_barrier: Q lands straight in registers.
    {   // MT back-to-back 16 B loads -> the low half of each m-tile's A operand.
        auto vq = opus::load<T::KV_GRP_BYTES>(g_q, u_q);
        auto* q16 = reinterpret_cast<opus::vector_t<D_BYTE, T::KV_GRP_BYTES>*>(&vq);
        opus::static_for<MT>([&](auto mic) {
            constexpr int mi = mic.value;
            opus::set_slice(q_a[mi], q16[mi], opus::number<0>{}, opus::number<T::KV_GRP_BYTES>{});
        });
    }
    q_scale = opus::load<1>(g_qs, u_qs)[0];
    {
        auto v_w = opus::load<T::C_FRAG>(g_w, u_w);
        auto* w4 = reinterpret_cast<opus::vector_t<D_WEIGHT, T::C_FRAG>*>(&v_w);
        opus::static_for<MT>([&](auto mic) {
            constexpr int mi = mic.value;
            opus::static_for<T::C_FRAG>([&](auto ec) {
                constexpr int e = ec.value;
                w_pl[mi][e] = (float)w4[mi][e] * weight_scale;
            });
        });
    }

    issue_ks(kvA, kvsA, load_page_id(clamp_tile(0)));
    gemm_nt(acc0A, kvA[0], kvsA, opus::number<0>{});
    issue_ks(kvB, kvsB, load_page_id(clamp_tile(1)));
    pgA = load_page_id(clamp_tile(2));
    pgB = load_page_id(clamp_tile(3));
    __builtin_amdgcn_sched_barrier(0);

    // ---- PEELED phase 0 (chunk 0 -> outA). ----
    compute_phase(kvA, kvsA, acc0A, kvB, kvsB, acc0B, pgA, 2, outA);
    __builtin_amdgcn_sched_barrier(0);

    // Each phase stores its own result as soon as reduce_all has it, so the loop body ends with
    // a compute rather than with the four out stores. Storing a phase behind (the fp8 schedule,
    // which this used to inherit) puts those stores last, where the next iteration's first MFMA
    // overwrites the very registers they read; that WAR pinned the loop-head s_waitcnt tighter
    // than the KV dependency needed and cost 1.5-6.7% depending on shape.
    fenced_store(outA, 0);
    if (tile_count >= 2) {
        compute_phase(kvB, kvsB, acc0B, kvA, kvsA, acc0A, pgB, 3, outB);
        fenced_store(outB, 1);
    }

    int t = 2;
    for (; t + 1 < tile_count; t += 2) {
        compute_phase(kvA, kvsA, acc0A,  kvB, kvsB, acc0B, pgA, t + 2, outA);
        fenced_store(outA, t);
        compute_phase(kvB, kvsB, acc0B, kvA, kvsA, acc0A,  pgB, t + 3, outB);
        fenced_store(outB, t + 1);
    }
    if (t < tile_count) {   // t == tile_count-1 here, so this is the last chunk
        compute_phase(kvA, kvsA, acc0A, kvB, kvsB, acc0B, pgA, t + 2, outA);
        fenced_store(outA, t);
    }
    // no epilogue store: every chunk was stored by the phase that computed it
}

} // namespace opus_logits
#endif
#endif  // PA_MQA_LOGITS_MXFP4_IMPL
