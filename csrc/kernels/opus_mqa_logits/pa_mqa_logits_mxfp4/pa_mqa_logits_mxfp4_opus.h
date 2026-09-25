// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// MXFP4 paged MQA logits (OPUS): one schedule, two arch-selected device bodies.
//
//   gfx950 : MFMA 32x32x64, wave64. Scales/kv PERMUTED:
//            q_scale [T,2,32,4]   kv_cache [nb,4,PAGE,16]   kv_scale [nb,2,32,4]
//   gfx1250: WMMA 32x16x128, wave32 / TDM. Scales/kv NATURAL:
//            q_scale [T,H,4]      kv_cache [nb,PAGE,64]     kv_scale [nb,PAGE,4]
//   shared : q [T,H,D/2] u8, weights [T,H] bf16, out [T,max_seq_len] fp32.
//
// Both arches share the kargs and the per-tile schedule; fwd_sched dispatches by runtime arch.
// Namespaces: opus_logits holds the shared schedule; opus_logits::gfx950 / ::gfx1250 each hold
// that arch's pa_mqa_logits_mxfp4_traits and pa_mqa_logits_mxfp4_kernel.
// gfx950 runs at q_per_block == 1, so a tile is one row.
//
// Passing one arch's scale arrays to the other is SILENT: the byte counts match, only the
// permutation differs. Only a comparison against a dequantized reference catches it.
//
// `num_rows` (the plan's total_q) is checked against every per-row array and q / weights / out.
//
// THE THINGS THE CALLER OWES (not checkable here):
//   1. the window is non-decreasing in row index within a tile, else the CTA's waves disagree
//      on the trip count and DEADLOCK on the phase barrier;
//   2. the store is bounded by the window, not max_seq_len: local_ends past out.size(1)
//      writes past the row;
//   3. a row with no live sequence has an empty window (local_ends <= local_starts); only such
//      a row may have a negative row_to_batch. Cells outside a row's window are never written.
#pragma once
#include "aiter_tensor.h"
#include <cstdint>
#include <cstddef>

// -- public API: three ops for BOTH arches; the instance is named by (q_per_block, block_k) --
void pa_mqa_logits_mxfp4_build_tiles(aiter_tensor_t& cu_seq_q,
                                     aiter_tensor_t& cu_tiles,
                                     int total_q,
                                     int max_tiles,
                                     int q_per_block);

void pa_mqa_logits_mxfp4_build_sched(aiter_tensor_t& cu_tiles,
                                     aiter_tensor_t& local_starts,
                                     aiter_tensor_t& local_ends,
                                     aiter_tensor_t& row_to_batch,
                                     aiter_tensor_t& cta_info,
                                     int num_tiles,
                                     int num_rows,
                                     int num_ctas,
                                     int cta_resident,
                                     int block_k,
                                     // 1: a tile is a row; cu_tiles is unread, may be empty.
                                     int q_per_block);

void pa_mqa_logits_mxfp4_fwd_sched(aiter_tensor_t& q,
                                   aiter_tensor_t& q_scale,
                                   aiter_tensor_t& kv_cache,
                                   aiter_tensor_t& kv_scale,
                                   aiter_tensor_t& block_tables,
                                   aiter_tensor_t& weights,
                                   aiter_tensor_t& local_starts,
                                   aiter_tensor_t& local_ends,
                                   aiter_tensor_t& cta_info,
                                   aiter_tensor_t& out,
                                   int num_rows,
                                   int num_ctas,
                                   float weight_scale,
                                   int kv_block_size,
                                   int max_seq_len,
                                   int q_per_block,
                                   int block_k);

#ifdef PA_MQA_LOGITS_MXFP4_IMPL
// ==== Implementation (compiled only in the .cu TU) ====

#include <opus/dtypes.hpp>
using bf16_t            = opus::dtypes::bf16;   // gfx1250 traits' per-head weight type
using mqa_logits_bf16_t = __bf16;               // gfx950 traits' per-head weight type

// relu must PROPAGATE NaN (a 0xFF E8M0 scale), so it is IEEE-754-2019 `maximum`, not a
// compare-and-select. Requires -fno-finite-math-only.
#define OPUS_LOGITS_RELU(x) __builtin_elementwise_maximum((x), 0.0f)

// -- shared record + kargs -------------------------------------------------------------
// One 32-byte record per CTA slot, loaded by one s_load_dwordx8.
// local_start / local_end (the tile's union window) set ONLY the loop bound: the trip count
// fixes the barrier count, so anything per-row reaching it deadlocks the CTA. The store mask
// is per row.
struct opus_mqa_cta_record {
    int row_id;       // FIRST packed query row of the tile
    int batch_id;     // block_tables row
    int chunk_start;  // first KV tile (block_k units), absolute, not relative to local_start
    int chunk_count;  // KV tiles it covers; 0 = surplus slot
    int local_start;  // the tile's UNION window start
    int local_end;    // the tile's UNION window end
    int group_rows;   // query rows in the tile, 1..Q_PER_BLOCK
    int _pad;
};
static_assert(sizeof(opus_mqa_cta_record) == 32,
              "the record must stay one s_load_dwordx8");
static_assert(alignof(opus_mqa_cta_record) == 4);

// Scale / cache layouts differ per arch (see the table at the top). Not layout-compatible with
// the opus-ops standalone kargs.
struct opus_mqa_logits_kargs {
    const void* __restrict__ ptr_q;         // [total_tokens, H, D/2]                fp4 (E2M1)
    const void* __restrict__ ptr_q_scale;   // [total_tokens, ...] e8m0, per-arch layout
    const void* __restrict__ ptr_kv;        // [num_blocks, ...]   fp4, per-arch layout
    const void* __restrict__ ptr_kv_scale;  // [num_blocks, ...]   e8m0, per-arch layout
    const int*  __restrict__ ptr_block_tables; // [batch, max_blocks_per_seq] int32
    const void* __restrict__ ptr_weights;   // [total_tokens, H] bf16
    float* __restrict__ ptr_out;             // [total_tokens, >= max window] fp32

    // Per-row windows, [num_rows] int32, for the store mask. gfx1250 only: local_ends required,
    // local_starts may be NULL (start 0). NULL on gfx950 (window comes off the record).
    const int* __restrict__ ptr_local_starts;
    const int* __restrict__ ptr_local_ends;

    int   num_rows;            // the schedule's row count; both kernels bound row_id by it
    int   stride_out_row;      // out row stride in elements
    float weight_scale;
    int   block_k;             // KV tile size along seq_kv (== Traits::KV_TILE_SIZE)
    int   max_blocks_per_seq;  // block_tables row stride

    const opus_mqa_cta_record* __restrict__ ptr_cta_info;  // [num_ctas]
    int   num_ctas;            // grid.x of the launch; slots past the work idle
};

// Both kernels read these by offset; the gfx950 kernel's ISA is pinned against them.
static_assert(sizeof(opus_mqa_logits_kargs)  == 112);
static_assert(alignof(opus_mqa_logits_kargs) == 8);
static_assert(offsetof(opus_mqa_logits_kargs, ptr_q             ) ==   0);
static_assert(offsetof(opus_mqa_logits_kargs, ptr_q_scale       ) ==   8);
static_assert(offsetof(opus_mqa_logits_kargs, ptr_kv            ) ==  16);
static_assert(offsetof(opus_mqa_logits_kargs, ptr_kv_scale      ) ==  24);
static_assert(offsetof(opus_mqa_logits_kargs, ptr_block_tables  ) ==  32);
static_assert(offsetof(opus_mqa_logits_kargs, ptr_weights       ) ==  40);
static_assert(offsetof(opus_mqa_logits_kargs, ptr_out           ) ==  48);
static_assert(offsetof(opus_mqa_logits_kargs, ptr_local_starts  ) ==  56);
static_assert(offsetof(opus_mqa_logits_kargs, ptr_local_ends    ) ==  64);
static_assert(offsetof(opus_mqa_logits_kargs, num_rows          ) ==  72);
static_assert(offsetof(opus_mqa_logits_kargs, stride_out_row    ) ==  76);
static_assert(offsetof(opus_mqa_logits_kargs, weight_scale      ) ==  80);
static_assert(offsetof(opus_mqa_logits_kargs, block_k           ) ==  84);
static_assert(offsetof(opus_mqa_logits_kargs, max_blocks_per_seq) ==  88);
static_assert(offsetof(opus_mqa_logits_kargs, ptr_cta_info      ) ==  96);
static_assert(offsetof(opus_mqa_logits_kargs, num_ctas          ) == 104);

// -- the schedule builder (arch-agnostic) --
#include "pa_mqa_logits_mxfp4_sched.cuh"

namespace opus_logits {
// Only `Table` is implemented here (both kernels static_assert it); Prefill / Decode keep the
// enum in sync with the opus-ops standalone build.
enum class mqa_logits_sched {
    Prefill,
    Decode,
    Table,
};
}

// -- gfx950 traits (MFMA 32x32x64) ----------------------------------------------------
// No D_DATA: opus::fp4_t is device-only and this struct must stay host-compilable.
namespace opus_logits::gfx950 {
template<int KV_TILE_SIZE_ = 256,
         int PAGE_SIZE_    = 64,
         int HEAD_DIM_     = 128,
         int N_HEADS_      = 64,
         int NUM_WARPS_    = 4>
struct pa_mqa_logits_mxfp4_traits {
    static constexpr int KV_TILE_SIZE = KV_TILE_SIZE_;  // block_k
    static constexpr int PAGE_SIZE    = PAGE_SIZE_;     // kv_block_size
    static constexpr int HEAD_DIM     = HEAD_DIM_;
    static constexpr int N_HEADS      = N_HEADS_;
    static constexpr int NUM_WARPS    = NUM_WARPS_;

    static constexpr int WARP_SIZE  = 64;
    static constexpr int BLOCK_SIZE = NUM_WARPS * WARP_SIZE;

    using D_WEIGHT = mqa_logits_bf16_t;   // per-head weights
    using D_ACC    = float;    // MFMA accumulator (C)
    using D_OUT    = float;    // output logits
    using D_SCALE  = int;      // E8M0 blockscale, packed as one int32 dword per lane

    // ---- MFMA shape ----
    static constexpr int MFMA_M = 32;
    static constexpr int MFMA_N = 32;
    static constexpr int MFMA_K = 64;

    static constexpr int ELEM_BITS   = 4;    // fp4: half a byte per element
    static constexpr int SCALE_BLOCK = 32;   // E8M0 blockscale granularity

    // ---- derived tile counts ----
    static constexpr int M_TILES  = N_HEADS / MFMA_M;    // 2  (head tiles along M)
    static constexpr int K_TILES  = HEAD_DIM / MFMA_K;   // 2  (outer K loop)
    static constexpr int K_CHUNKS = MFMA_K / SCALE_BLOCK;// 2  (32-K scale blocks per k-tile)
    static constexpr int SCALE_BLOCKS_ROW = HEAD_DIM / SCALE_BLOCK;  // 4 == K_TILES * K_CHUNKS

    static constexpr int N_TOTAL_TILES   = KV_TILE_SIZE / MFMA_N;      // 8 (bk=256) / 2 (bk=64)
    static constexpr int N_TILES         = N_TOTAL_TILES / NUM_WARPS;  // 2 = NTPW, both variants
    static constexpr int TILES_PER_BLOCK = PAGE_SIZE / MFMA_N;         // 2 (MFMA_N tiles per page)
    static constexpr int N_PHYS = (N_TILES + TILES_PER_BLOCK - 1) / TILES_PER_BLOCK;  // 1

    // ---- C fragment geometry ----
    // C is [(rept_c<y>, grpm_c<p>, pack_c<y>), (grpn_c<p>)]: m = rept*8 + (L/32)*4 + pack.
    static constexpr int C_FRAG = MFMA_M * MFMA_N / WARP_SIZE;  // 16 floats per lane per m-tile
    static constexpr int GRPN_C = MFMA_N;                       // 32: n == lane % 32
    static constexpr int GRPM_C = WARP_SIZE / GRPN_C;           // 2:  M spread over 2 lane groups
    static constexpr int PACK_C = 4;                            // 16 B of f32 per contiguous run
    static constexpr int REPT_C = C_FRAG / PACK_C;              // 4
    // A lane holds 32 of the 64 heads; the partner lane group's 32 arrive via one permlane32.
    static constexpr int HEADS_PER_LANE = M_TILES * C_FRAG;     // 32

    static constexpr int DWORDx4_BYTES = 16;

    // ---- per-lane payload ----
    // A lane holds one 32-K block = 16 B: the low half of opus's 256-bit operand (fp4 reads
    // only the low 16 B).
    static constexpr int KV_GRP_ELEMS = SCALE_BLOCK;                  // 32 fp4 per lane
    static constexpr int KV_GRP_BYTES = KV_GRP_ELEMS * ELEM_BITS / 8; // 16 bytes
    static constexpr int A_BYTES_PER_LANE = (MFMA_M * MFMA_K / WARP_SIZE) * ELEM_BITS / 8; // 16

    // ---- op_sel byte packing ----
    // One scale dword per lane covers the (kt, tile) loop:
    //     A: byte = kt * M_TILES + mi     B: byte = kt * N_TILES + nt
    static constexpr int QS_BYTES  = K_TILES * M_TILES;   // 4
    static constexpr int KVS_BYTES = K_TILES * N_TILES;   // 4
    static_assert(QS_BYTES == 4 && KVS_BYTES == 4,
                  "the op_sel byte packing assumes exactly 4 (kt, tile) pairs per dword");

    // -- LAYOUTS -------------------------------------------------------------
    // ---- q: natural [T][head][D/2] (NO preshuffle) ----
    // stride_q_* are unused by the kernel; they only derive Q_ROW_BYTES.
    static constexpr int stride_q_m     = KV_GRP_BYTES;                    // 16
    static constexpr int stride_q_g     = MFMA_M * stride_q_m;             // 512
    static constexpr int stride_q_ktile = K_CHUNKS * stride_q_g;           // 1024
    static constexpr int stride_q_mtile = K_TILES * stride_q_ktile;        // 2048
    static constexpr int Q_ROW_BYTES    = M_TILES * stride_q_mtile;        // 4096 == H * D/2

    // ---- kv_cache: [num_blocks][kt(K_TILES)][g(K_CHUNKS)][nt(TILES_PER_BLOCK)][m(MFMA_N)][16 B]
    // holds KV[page token = nt*MFMA_N + m][K = (kt*K_CHUNKS + g)*32 : +32], i.e. the standard
    // paged fp4 layout [num_blocks][b(4)][o(PAGE)][16 B] (pinned by the static_assert below).
    static constexpr int stride_kv_m     = KV_GRP_BYTES;                      // 16
    static constexpr int stride_kv_ntile = MFMA_N * stride_kv_m;              // 512
    static constexpr int stride_kv_g     = TILES_PER_BLOCK * stride_kv_ntile; // 1024 == PAGE*16
    static constexpr int stride_kv_ktile = K_CHUNKS * stride_kv_g;            // 2048
    static constexpr int stride_kv_block = K_TILES * stride_kv_ktile;         // 4096
    static_assert(stride_kv_g == PAGE_SIZE * KV_GRP_BYTES,
                  "kv_cache must stay the standard paged [b(4)][o(PAGE)][16 B] layout: "
                  "the 32-K block stride is one block across a whole page");

    // ---- q_scale: [T][g(K_CHUNKS)][m(MFMA_N)][byte(QS_BYTES)] (e8m0) ----
    // byte (kt*M_TILES + mi) of lane L's dword = e8m0(head mi*32 + m, block kt*K_CHUNKS + g).
    // In dwords: index = g*MFMA_N + m == L; one load per lane.
    static constexpr int stride_qs_g_dw = MFMA_M;                          // 32 dwords
    static constexpr int QS_ROW_BYTES   = K_CHUNKS * MFMA_M * QS_BYTES;    // 256

    // ---- kv_scale: [num_blocks][g(K_CHUNKS)][m(MFMA_N)][byte(KVS_BYTES)] (e8m0) ----
    // byte (kt*N_TILES + nt) = e8m0(page token nt*32 + m, block kt*K_CHUNKS + g).
    static constexpr int stride_kvs_g_dw  = MFMA_N;                        // 32 dwords
    static constexpr int stride_kvs_block = K_CHUNKS * MFMA_N * KVS_BYTES; // 256 bytes

    // ---- byte counts the launcher checks (names shared with the gfx1250 traits) ----
    static constexpr int KV_PAGE_BYTES  = stride_kv_block;                  // 4096
    static constexpr int KVS_PAGE_BYTES = stride_kvs_block;                 // 256
    static constexpr int PAGES_PER_TILE = KV_TILE_SIZE / PAGE_SIZE;         // 4 (bk=256) / 1
    static constexpr bool READS_ROW_WINDOWS = false;

    // ---- weights: natural [T, H] bf16 ----
    static constexpr int stride_w_lg   = HEADS_PER_LANE;                   // 32 bf16
    static constexpr int W_ROW_ELEMS   = GRPM_C * HEADS_PER_LANE;          // 64 == N_HEADS

    static_assert(ELEM_BITS == 4, "this traits set is fp4-only");
    static_assert(HEAD_DIM % MFMA_K == 0, "HEAD_DIM must be a multiple of MFMA_K (64)");
    static_assert(N_HEADS % MFMA_M == 0, "N_HEADS must be a multiple of MFMA_M (32)");
    static_assert(KV_TILE_SIZE % MFMA_N == 0, "KV_TILE must be a multiple of MFMA_N");
    static_assert(N_TOTAL_TILES % NUM_WARPS == 0, "N_TOTAL_TILES must be a multiple of NUM_WARPS");
    static_assert(PAGE_SIZE % MFMA_N == 0, "PAGE must be a multiple of MFMA_N");
    static_assert(KV_TILE_SIZE % PAGE_SIZE == 0, "KV_TILE must be a multiple of PAGE");
    static_assert(N_TILES == 2, "kernel currently assumes N_TILES (NTPW) == 2");
    static_assert(M_TILES == 2, "kernel currently assumes M_TILES == 2");
    static_assert(K_TILES == 2, "kernel currently assumes K_TILES == 2 (the outer K loop)");
    static_assert(N_PHYS == 1, "kernel assumes N_PHYS == 1 (a warp's NTPW tiles share one page)");
    static_assert(N_TILES == TILES_PER_BLOCK,
                  "the kv layout uses TILES_PER_BLOCK as its nt extent; NTPW must match it");
    static_assert(KV_GRP_BYTES == 16, "a lane's fp4 block must be exactly one 16B dwordx4");
    static_assert(A_BYTES_PER_LANE == 16, "fp4 payload per lane is 16 B (low half of the operand)");
    static_assert(GRPM_C == 2, "the head reduction emits exactly one permlane32 swap");
    static_assert(Q_ROW_BYTES == N_HEADS * HEAD_DIM * ELEM_BITS / 8, "q preshuffle must be size-preserving");
    static_assert(W_ROW_ELEMS == N_HEADS, "weight preshuffle must be size-preserving");
};
}  // namespace opus_logits::gfx950

// -- gfx1250 traits (WMMA 32x16x128) --------------------------------------------------
// Natural kv_cache layout. KV_TILE_SIZE sizes the accumulator (ACC_VGPR), which sets waves per
// SIMD (3 at 64, 1 at 128). The caller's `cta_resident` must be re-tuned if it changes: a stale
// value silently under-fills the GPU.
namespace opus_logits::gfx1250 {
template<int Q_PER_BLOCK_  = 4,
         int LDS_STAGES_   = 2,
         int KV_TILE_SIZE_ = 64,
         int PAGE_SIZE_    = 64,
         int HEAD_DIM_     = 128,
         int N_HEADS_      = 64>
struct pa_mqa_logits_mxfp4_traits {
    static constexpr int KV_TILE_SIZE = KV_TILE_SIZE_;  // block_k
    static constexpr int PAGE_SIZE    = PAGE_SIZE_;     // kv_block_size
    static constexpr int HEAD_DIM     = HEAD_DIM_;
    static constexpr int N_HEADS      = N_HEADS_;

    // Constant, not opus::get_warp_size(): that returns 64 in the host pass, silently building
    // the wave64 fragment layout.
    static constexpr int WAVE_SIZE   = 32;
    static constexpr int Q_PER_BLOCK = Q_PER_BLOCK_;               // query rows per CTA == waves
    static constexpr int NUM_WAVES   = Q_PER_BLOCK;
    static constexpr int BLOCK_SIZE  = NUM_WAVES * WAVE_SIZE;      // 128

    using D_WEIGHT = bf16_t;   // per-head weights
    using D_ACC    = float;    // WMMA accumulator (C)
    using D_OUT    = float;    // output logits
    using D_SCALE  = int;      // E8M0 blockscale, packed as one int32 dword per lane (BX32)

    static constexpr int MMA_M = 32;   // heads
    static constexpr int MMA_N = 16;   // tokens
    static constexpr int MMA_K = 128;  // head_dim

    static constexpr int ELEM_BITS   = 4;    // fp4: half a byte per element
    static constexpr int SCALE_BLOCK = 32;   // E8M0 blockscale granularity

    static constexpr int M_TILES  = N_HEADS / MMA_M;      // 2  (head tiles along M)
    static constexpr int K_TILES  = HEAD_DIM / MMA_K;     // 1  (no outer K loop)
    static constexpr int K_CHUNKS = MMA_K / SCALE_BLOCK;  // 4  (32-K scale blocks per instruction)
    static constexpr int SCALE_BLOCKS_ROW = HEAD_DIM / SCALE_BLOCK;  // 4 == K_TILES * K_CHUNKS

    static constexpr int N_TILES         = KV_TILE_SIZE / MMA_N;   // 4
    static constexpr int TILES_PER_PAGE  = PAGE_SIZE / MMA_N;      // 4
    static constexpr int PAGES_PER_TILE  = KV_TILE_SIZE / PAGE_SIZE;  // 1

    static constexpr int C_FRAG = MMA_M * MMA_N / WAVE_SIZE;   // 16 floats per lane per m-tile
    static constexpr int GRPN_C = MMA_N;                       // 16: n == lane % 16
    static constexpr int GRPM_C = WAVE_SIZE / GRPN_C;          // 2:  M spread over 2 lane groups
    static constexpr int PACK_C = C_FRAG / 2;                  // 8:  c[0..7] is m = base + i
    static constexpr int REPT_C = C_FRAG / PACK_C;             // 2:  and c[8..15] is base + 16
    static constexpr int HEADS_PER_LANE = M_TILES * C_FRAG;    // 32
    static constexpr int SWAP_DISTANCE  = GRPN_C;              // 16 == permlane16_swap's distance

    static constexpr int KV_GRP_ELEMS = SCALE_BLOCK;                     // 32 fp4 per block
    static constexpr int KV_GRP_BYTES = KV_GRP_ELEMS * ELEM_BITS / 8;    // 16 B == one ds_load_b128
    static constexpr int A_BYTES_PER_LANE = MMA_M * MMA_K / WAVE_SIZE * ELEM_BITS / 8;  // 64
    static constexpr int B_BYTES_PER_LANE = MMA_N * MMA_K / WAVE_SIZE * ELEM_BITS / 8;  // 32
    static constexpr int A_VGPR_GROUPS = A_BYTES_PER_LANE / KV_GRP_BYTES;   // 4
    static constexpr int B_VGPR_GROUPS = B_BYTES_PER_LANE / KV_GRP_BYTES;   // 2

    static constexpr int SCALE_BYTES_PER_DWORD = K_CHUNKS;   // 4
    static_assert(SCALE_BYTES_PER_DWORD == 4,
                  "a BX32 scale operand is one dword; byte b must be block b with nothing left over");

    static constexpr int TOKEN_BYTES  = HEAD_DIM * ELEM_BITS / 8;     // 64 == one token's fp4 row
    static constexpr int Q_ROW_BYTES  = N_HEADS * TOKEN_BYTES;        // 4096 == H * D/2
    static constexpr int QS_ROW_BYTES = N_HEADS * SCALE_BYTES_PER_DWORD;   // 256
    static constexpr int KV_PAGE_BYTES  = PAGE_SIZE * TOKEN_BYTES;    // 4096
    static constexpr int KVS_PAGE_BYTES = PAGE_SIZE * SCALE_BYTES_PER_DWORD;  // 256
    static constexpr int W_ROW_ELEMS    = N_HEADS;                    // natural [T, H] bf16
    static constexpr bool READS_ROW_WINDOWS = true;

    static constexpr int KV_CHUNK_BYTES = PAGE_SIZE * KV_GRP_BYTES;   // 1024

    // -- the ABI, as byte offsets within one row / one page --
    // `block` is the 32-element E8M0 block inside a token: the 16 B at `b * 16` of its 64 B.
    static constexpr int q_byte(int head, int block) {
        return head * TOKEN_BYTES + block * KV_GRP_BYTES;
    }
    static constexpr int q_scale_byte(int head) {
        return head * SCALE_BYTES_PER_DWORD;
    }
    static constexpr int kv_byte(int token, int block) {
        return token * TOKEN_BYTES + block * KV_GRP_BYTES;
    }
    static constexpr int kv_scale_byte(int token) {
        return token * SCALE_BYTES_PER_DWORD;
    }

    // -- measured WMMA fragment maps (also used by the op test's reference) --
    static constexpr int lane_m0(int lane) { return lane % GRPN_C; }   // L % 16
    static constexpr int lane_g (int lane) { return lane / GRPN_C; }   // L / 16

    static constexpr int frag_a_row  (int lane, int v) { return lane_m0(lane) + (v >= 2 ? 16 : 0); }
    static constexpr int frag_a_block(int lane, int v) { return lane_g(lane) + (v % 2 ? 2 : 0); }
    static constexpr int frag_b_col  (int lane)        { return lane_m0(lane); }
    static constexpr int frag_b_block(int lane, int v) { return lane_g(lane) + (v ? 2 : 0); }
    static constexpr int frag_c_n    (int lane)        { return lane_m0(lane); }
    static constexpr int frag_c_m    (int lane, int reg) {
        return (reg < PACK_C ? 0 : 16) + lane_g(lane) * PACK_C + reg % PACK_C;
    }
    static constexpr bool frag_a_is_bijection() {
        bool seen[MMA_M][K_CHUNKS] = {};
        for (int lane = 0; lane < WAVE_SIZE; ++lane)
            for (int v = 0; v < A_VGPR_GROUPS; ++v) {
                const int r = frag_a_row(lane, v), b = frag_a_block(lane, v);
                if (r < 0 || r >= MMA_M || b < 0 || b >= K_CHUNKS || seen[r][b]) return false;
                seen[r][b] = true;
            }
        for (int r = 0; r < MMA_M; ++r)
            for (int b = 0; b < K_CHUNKS; ++b) if (!seen[r][b]) return false;
        return true;
    }
    static constexpr bool frag_b_is_bijection() {
        bool seen[MMA_N][K_CHUNKS] = {};
        for (int lane = 0; lane < WAVE_SIZE; ++lane)
            for (int v = 0; v < B_VGPR_GROUPS; ++v) {
                const int n = frag_b_col(lane), b = frag_b_block(lane, v);
                if (n < 0 || n >= MMA_N || b < 0 || b >= K_CHUNKS || seen[n][b]) return false;
                seen[n][b] = true;
            }
        for (int n = 0; n < MMA_N; ++n)
            for (int b = 0; b < K_CHUNKS; ++b) if (!seen[n][b]) return false;
        return true;
    }
    static constexpr bool frag_c_is_bijection() {
        bool seen[MMA_M][MMA_N] = {};
        for (int lane = 0; lane < WAVE_SIZE; ++lane)
            for (int reg = 0; reg < C_FRAG; ++reg) {
                const int m = frag_c_m(lane, reg), n = frag_c_n(lane);
                if (m < 0 || m >= MMA_M || n < 0 || n >= MMA_N || seen[m][n]) return false;
                seen[m][n] = true;
            }
        for (int m = 0; m < MMA_M; ++m)
            for (int n = 0; n < MMA_N; ++n) if (!seen[m][n]) return false;
        return true;
    }
    static_assert(frag_a_is_bijection(), "A's (lane, VGPR group) -> (row, block) map is not a bijection");
    static_assert(frag_b_is_bijection(), "B's (lane, VGPR group) -> (col, block) map is not a bijection");
    static_assert(frag_c_is_bijection(), "C's (lane, reg) -> (m, n) map is not a bijection");

    static constexpr int scale_a_row(int lane) { return lane; }
    static constexpr int scale_b_sel (int n_tile) { return n_tile % 2; }
    static constexpr int scale_b_lane(int n_tile, int n) { return scale_b_sel(n_tile) * GRPN_C + n; }

    // -- LDS tile image. TDM copies pages verbatim and WMMA fixes the lane->token map, so the
    // padding is the only way to break the bank conflict. --
    static constexpr int LDS_PAD_INTERVAL = 128;
    static constexpr int LDS_PAD_AMOUNT   =  16;
    static constexpr int LDS_PAD_STRIDE = LDS_PAD_INTERVAL ? LDS_PAD_INTERVAL : 1;

    static constexpr int lds_expand(int page_byte) {
        return page_byte + (page_byte / LDS_PAD_STRIDE) * LDS_PAD_AMOUNT;
    }
    static constexpr int lds_byte(int token_in_page, int block) {
        return lds_expand(kv_byte(token_in_page, block));
    }
    static constexpr int LDS_PAGE_BYTES = lds_expand(KV_PAGE_BYTES);  // 4608 == 4096 + one 16 B pad per 128 B
    static constexpr int LDS_TILE_BYTES  = PAGES_PER_TILE * LDS_PAGE_BYTES;   // 4608
    static constexpr int LDS_SCALE_BYTES = KV_TILE_SIZE * SCALE_BYTES_PER_DWORD;   // 256
    static constexpr int LDS_STAGE_BYTES = LDS_TILE_BYTES + LDS_SCALE_BYTES;
    static constexpr int LDS_STAGES      = LDS_STAGES_;
    static constexpr int LDS_BYTES       = LDS_STAGES * LDS_STAGE_BYTES;
    static constexpr size_t smem_size_bytes() { return (size_t)LDS_BYTES; }
    static_assert((LDS_STAGES & (LDS_STAGES - 1)) == 0,
                  "LDS_STAGES must be a power of two: the stage index is a mask, so a non-power "
                  "turns one `and` into a division in the phase's address arithmetic");

    static constexpr int lds_stage_byte(int tile) { return (tile & (LDS_STAGES - 1)) * LDS_STAGE_BYTES; }
    static constexpr int lds_page_byte (int page) { return page * LDS_PAGE_BYTES; }
    static constexpr int lds_scale_byte(int token_in_tile) {
        return LDS_TILE_BYTES + token_in_tile * SCALE_BYTES_PER_DWORD;
    }

    static constexpr int LDS_NTILE_BYTES = lds_byte(MMA_N, 0) - lds_byte(0, 0);   // 1152
    static constexpr bool lds_ntile_step_is_uniform() {
        for (int t = 0; t + MMA_N <= PAGE_SIZE - MMA_N; ++t)
            for (int b = 0; b < K_CHUNKS; ++b)
                if (lds_byte(t + MMA_N, b) - lds_byte(t, b) != LDS_NTILE_BYTES) return false;
        return true;
    }
    static_assert(lds_ntile_step_is_uniform(),
                  "the n-tile step in LDS is not lane-independent, so the B reads cannot share an "
                  "address register. The pad interval must divide MMA_N * TOKEN_BYTES.");

    static constexpr int LDS_BLOCK_BYTES = lds_byte(0, 1) - lds_byte(0, 0);   // 16
    static constexpr bool lds_block_step_is_uniform() {
        for (int t = 0; t < PAGE_SIZE; ++t)
            for (int b = 0; b + 1 < K_CHUNKS; ++b)
                if (lds_byte(t, b + 1) - lds_byte(t, b) != LDS_BLOCK_BYTES) return false;
        return true;
    }
    static_assert(lds_block_step_is_uniform(),
                  "the K-block step in LDS is not uniform -- under the natural layout that means "
                  "the pad is cutting a token in half -- so the B reads' block offset cannot be "
                  "an immediate");

    static constexpr int lds_bank_unit(int lane, int v) {
        return (lds_byte(frag_b_col(lane), frag_b_block(lane, v)) >> 4) % 8;
    }
    static constexpr bool lds_conflict_free() {
        for (int v = 0; v < B_VGPR_GROUPS; ++v)
            for (int base = 0; base < WAVE_SIZE; base += 8) {
                int seen = 0;
                for (int i = 0; i < 8; ++i) seen |= 1 << lds_bank_unit(base + i, v);
                if (seen != 0xFF) return false;
            }
        return true;
    }
    static constexpr bool lds_conflict_free_unpadded_control() {
        for (int base = 0; base < WAVE_SIZE; base += 8) {
            int seen = 0;
            for (int i = 0; i < 8; ++i) {
                const int lane = base + i;
                const int linear = frag_b_col(lane) * TOKEN_BYTES + frag_b_block(lane, 0) * KV_GRP_BYTES;
                seen |= 1 << ((linear >> 4) % 8);
            }
            if (seen != 0xFF) return false;
        }
        return true;
    }
    static_assert(!lds_conflict_free_unpadded_control(),
                  "the unpadded natural layout is supposed to CONFLICT; if it does not, this "
                  "checker cannot detect a conflict and the assert below proves nothing");
    static_assert(lds_conflict_free(),
                  "the LDS padding policy leaves a bank conflict. Unpadded, the natural layout's "
                  "64 B token stride puts every bank in {0-3} or {16-19} -- a 4-way conflict; the "
                  "pad of one read vector per 128 B is what makes the 8 lanes tile all 32 banks.");
    static_assert(LDS_PAD_INTERVAL != 0,
                  "the natural layout's lanes are 64 B apart and do NOT tile the banks unpadded, "
                  "so this layout may not run the pad off: see the control above.");

    // -- TDM: every wave issues (keeps the steady loop branch-free). A page splits into
    // contiguous byte pieces; wave w takes piece (w % SPLIT) of page (w / SPLIT). --
    static constexpr int TDM_ISSUE_WAVES = NUM_WAVES;
    static constexpr int TDM_PAGE_SPLIT  = TDM_ISSUE_WAVES / PAGES_PER_TILE;   // 4
    static constexpr int TDM_OPS_PER_TILE = 2 * TDM_ISSUE_WAVES;               // 8
    static constexpr int TDM_INFLIGHT_PER_WAVE = 3;                   // hardware, per opus.hpp
    static constexpr int TDM_OPS_PER_ISSUING_WAVE = TDM_OPS_PER_TILE / TDM_ISSUE_WAVES;   // 2
    static constexpr int TDM_INFLIGHT_CAP = TDM_INFLIGHT_PER_WAVE / TDM_OPS_PER_ISSUING_WAVE;

    static constexpr int TDM_ROWS        = PAGE_SIZE / TDM_PAGE_SPLIT;         // 16
    static constexpr int TDM_PIECE_BYTES = KV_PAGE_BYTES / TDM_PAGE_SPLIT;     // 1024, global side
    static constexpr int LDS_PIECE_BYTES = lds_expand(TDM_PIECE_BYTES);        // 1152
    static constexpr int TDM_SCALE_PIECE_BYTES = TDM_ROWS * SCALE_BYTES_PER_DWORD;   // 64

    static_assert(TDM_ISSUE_WAVES <= NUM_WAVES,
                  "there are not enough waves to spread a tile's TDM issues over");
    static_assert(TDM_ISSUE_WAVES % PAGES_PER_TILE == 0,
                  "the issuing waves must divide into whole pages, or a wave straddles two pages "
                  "and needs two page ids");
    static_assert(PAGE_SIZE % TDM_PAGE_SPLIT == 0 && KV_PAGE_BYTES % TDM_PAGE_SPLIT == 0,
                  "a page must split into equal pieces");
    static_assert(LDS_PIECE_BYTES * TDM_PAGE_SPLIT == LDS_PAGE_BYTES,
                  "the pieces must tile the padded LDS page exactly. They do not when a piece's "
                  "byte count is not a whole number of pad intervals, and the pieces then overlap "
                  "by the rounding -- silently, because each DMA on its own is still in range.");
    static_assert(TDM_OPS_PER_ISSUING_WAVE <= TDM_INFLIGHT_PER_WAVE,
                  "one tile's share already exceeds a wave's TDM queue depth");

    // -- pipeline depth, derived from the stage count --
    static constexpr int TILES_IN_FLIGHT = LDS_STAGES - 1;
    static constexpr int ISSUE_LEAD      = TILES_IN_FLIGHT;   // phase t issues tile t + this
    static_assert(LDS_STAGES >= 2, "need at least a double buffer");
    static_assert(TILES_IN_FLIGHT <= TDM_INFLIGHT_CAP,
                  "LDS_STAGES asks for more tiles in flight than the per-wave TDM queue admits. "
                  "Splitting a page more finely does NOT raise the cap: however finely it is "
                  "divided, every issuing wave still carries one data op and one scale op, so "
                  "the depth of 3 admits one tile. Reaching more needs the data and scale "
                  "descriptors MERGED, not a finer split.");
    static constexpr int TENSORCNT_KEEP = (TILES_IN_FLIGHT - 1) * TDM_OPS_PER_ISSUING_WAVE;

    // -- accumulator: two ping-ponged sets so tile t+1's WMMAs overlap tile t's reduction.
    // ACC_VGPR scales with the KV tile and sets occupancy. --
    static constexpr int ACC_SETS = 2;
    static constexpr int ACC_VGPR_PER_SET = M_TILES * N_TILES * C_FRAG;   // 128
    static constexpr int ACC_VGPR = ACC_SETS * ACC_VGPR_PER_SET;          // 256
    static constexpr int WMMA_PER_TILE = M_TILES * N_TILES;               // 8
    static constexpr int WAVES_PER_EU = 1;
    static constexpr int VGPR_ADDRESSABLE = 1024;                    // wave32, per lane
    static_assert(ACC_VGPR < VGPR_ADDRESSABLE,
                  "the accumulator alone must not fill the register file");

    static_assert(ELEM_BITS == 4, "this traits set is fp4-only");
    static_assert(HEAD_DIM == MMA_K,
                  "K = 128 is the whole head_dim in one instruction; a HEAD_DIM past MMA_K needs "
                  "the outer kt loop back, and with it the kt term in both scale byte indices");
    static_assert(N_HEADS % MMA_M == 0, "N_HEADS must be a multiple of MMA_M (32)");
    static_assert(KV_TILE_SIZE % MMA_N == 0, "KV_TILE must be a multiple of MMA_N");
    static_assert(KV_TILE_SIZE % PAGE_SIZE == 0, "KV_TILE must be a whole number of pages");
    static_assert(PAGE_SIZE % MMA_N == 0, "PAGE must be a multiple of MMA_N");
    static_assert(M_TILES == 2, "the head reduction assumes 64 heads over 2 m-tiles");
    static_assert(GRPM_C == 2, "the head reduction emits exactly one permlane16_swap");
    static_assert(HEADS_PER_LANE * GRPM_C == N_HEADS,
                  "a lane and its swap partner must together hold every head, or the reduction "
                  "drops heads without any count changing");
    static_assert(A_BYTES_PER_LANE == 64 && B_BYTES_PER_LANE == 32,
                  "the dedicated f4 instruction takes i32x16 / i32x8 and uses all of both");
    static_assert(KV_GRP_BYTES == 16, "a 32-element fp4 block must be exactly one ds_load_b128");
    static_assert(Q_ROW_BYTES == N_HEADS * HEAD_DIM * ELEM_BITS / 8, "q must be size-preserving");
    static_assert(QS_ROW_BYTES == N_HEADS * SCALE_BLOCKS_ROW, "q_scale must be size-preserving");
    static_assert(KV_PAGE_BYTES == PAGE_SIZE * HEAD_DIM * ELEM_BITS / 8, "kv must be size-preserving");
    static_assert(Q_PER_BLOCK >= 1, "Q_PER_BLOCK must be positive");
};
}  // namespace opus_logits::gfx1250


// -- device bodies, one per arch. Each self-stubs on the wrong arch, so both launch symbols
// resolve in this one TU. --
#include "pa_mqa_logits_mxfp4_gfx950.cuh"
#include "pa_mqa_logits_mxfp4_gfx1250.cuh"


#endif  // PA_MQA_LOGITS_MXFP4_IMPL
