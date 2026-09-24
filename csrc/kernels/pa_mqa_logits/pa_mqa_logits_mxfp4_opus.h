// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// MXFP4 paged MQA logits (OPUS) -- ONE module, ONE schedule, TWO arch-selected device bodies.
//
//   gfx950 : MFMA 32x32x64, wave64. Scales/kv PERMUTED:
//            q_scale [T,2,32,4]   kv_cache [nb,4,PAGE,16]   kv_scale [nb,2,32,4]
//   gfx1250: WMMA 32x16x128, wave32 / TDM. Scales/kv NATURAL:
//            q_scale [T,H,4]      kv_cache [nb,PAGE,64]     kv_scale [nb,PAGE,4]
//   shared : q [T,H,D/2] u8, weights [T,H] bf16, out [T,max_seq_len] fp32.
//
// Both arches take the SAME kernel args (opus_mqa_logits_kargs) and the SAME per-tile schedule
// (build_tiles + build_sched); a `#if defined(__gfx950__)` / `#if defined(__gfx1250__)` version
// macro selects which device body compiles, and fwd_sched dispatches by runtime arch. gfx950 is
// driven at q_per_block == 1, so each "tile" is one row and the tile-union window is that row's
// window -- exactly the per-row record its kernel already reads.
//
// HANDING ONE ARCH THE OTHER'S ARRAYS IS SILENT: every fp4 scale layout has the same byte count,
// so the launcher catches a wrong array and never a wrong permutation. Only a random-data
// comparison against a dequantized reference does; the python entry points check the ndim.
//
// THE TWO THINGS THE CALLER OWES, neither checkable here (broken -> the CTA's waves disagree on
// the trip count and DEADLOCK on the phase barrier; they do not answer wrong):
//   1. the window rule is NON-DECREASING in the row index within a tile;
//   2. the store is bounded by the WINDOW, not by max_seq_len -- a `local_ends` entry past
//      out.size(1) writes past the row.
#pragma once
#include "aiter_tensor.h"
#include <cstdint>
#include <cstddef>

// ── public API: three ops for BOTH arches; the instance is named by (q_per_block, block_k) ──
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
                                     int num_ctas,
                                     int cta_resident,
                                     int block_k,
                                     // 1 is the IDENTITY cut -- a tile is a row -- and then
                                     // `cu_tiles` is not read and may be empty.
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
// ==== Implementation (compiled only in the .cu TU): kargs + schedule + traits + kernels. ====

#include <opus/dtypes.hpp>
using bf16_t            = opus::dtypes::bf16;   // gfx1250 traits' per-head weight type
using mqa_logits_bf16_t = __bf16;               // gfx950 traits' per-head weight type

// The relu must be the IEEE-754-2019 `maximum`, which PROPAGATES NaN (a 0xFF E8M0 scale), not a
// compare-and-select, which returns 0 for it. REQUIRES -fno-finite-math-only (the #error below
// enforces it). gfx950's body inlines the same builtin; gfx1250's body uses this macro.
#define OPUS_LOGITS_RELU(x) __builtin_elementwise_maximum((x), 0.0f)

// ── shared record + kargs (byte-identical across arches) ─────────────────────────────
// One 32-byte record per CTA slot, arriving in one `s_load_dwordx8` -- do not shrink it.
// `chunk_count == 0` marks a surplus slot; a slot is 1..Q_PER_BLOCK rows, `group_rows` many.
//
// `chunk_start` is ABSOLUTE, a tile index into the sequence and not an offset from
// `local_start`; the store's out-of-range proof depends on it.
//
// `local_start` / `local_end` are the tile's UNION window and set the LOOP BOUND, the only
// thing they may set: the trip count fixes the barrier count, so anything per-row reaching it
// deadlocks the CTA. The STORE MASK is separate and per row.
struct opus_mqa_cta_record {
    int row_id;       // FIRST packed query row of the tile
    int batch_id;     // block_tables row
    int chunk_start;  // first KV tile (block_k units) this CTA covers, absolute
    int chunk_count;  // KV tiles it covers; 0 = surplus slot
    int local_start;  // the tile's UNION window start
    int local_end;    // the tile's UNION window end
    int group_rows;   // query rows in the tile, 1..Q_PER_BLOCK
    int _pad;
};
static_assert(sizeof(opus_mqa_cta_record) == 32,
              "the record must stay one s_load_dwordx8");
static_assert(alignof(opus_mqa_cta_record) == 4);

struct opus_mqa_logits_kargs {
    const void* __restrict__ ptr_q;         // [total_tokens, H, D/2]                fp4 (E2M1)
    const void* __restrict__ ptr_q_scale;   // [total_tokens, H, 4]                  e8m0
    const void* __restrict__ ptr_kv;        // [num_blocks, PAGE, D/2]               fp4 (E2M1)
    const void* __restrict__ ptr_kv_scale;  // [num_blocks, PAGE, 4]                 e8m0
    const int*  __restrict__ ptr_block_tables; // [batch, max_blocks_per_seq] int32
    const void* __restrict__ ptr_weights;   // [total_tokens, H] bf16
    float* __restrict__ ptr_out;             // [total_tokens, max_seq_len] fp32

    // PER-ROW window arrays, each [total_q] int32, read for the STORE MASK only -- the loop bound
    // comes from the record's union. `ptr_local_starts` MAY BE NULL (every row starts at 0);
    // `ptr_local_ends` may not, being the store's upper bound.
    const int* __restrict__ ptr_row_to_batch;   // builder only; the record carries batch_id
    const int* __restrict__ ptr_local_starts;
    const int* __restrict__ ptr_local_ends;
    // Read by neither the kernel nor the schedule. `ptr_cu_seq_q` is the BATCH prefix sum and is
    // an input to the tile cut, not to the launch.
    const int* __restrict__ ptr_cu_seq_q;
    int   split_kv;            // context splits per row (>= 1); unused here
    int   num_rows;            // total query rows
    int   num_batches;         // real batch count

    int   max_seq_len;
    int   stride_out_row;      // out row stride in elements (== max_seq_len for dense out)
    float weight_scale;
    int   block_k;             // KV tile size along seq_kv (== Traits::KV_TILE_SIZE)
    int   kv_block_size;       // paged block (page) size (== Traits::PAGE_SIZE)
    int   max_blocks_per_seq;  // block_tables row stride

    // APPENDED, not grouped with the pointers above: every field before this keeps the kernarg
    // offset it has on gfx950, which the offsetof block below enforces.
    const opus_mqa_cta_record* __restrict__ ptr_cta_info;  // [num_ctas]
    int   num_ctas;            // grid.x of the launch; slots past the work idle
};

// A field reordered or retyped on either target fails here instead of producing a launcher that
// reads the wrong dword.
static_assert(sizeof(opus_mqa_logits_kargs)  == 144);   // 128 + the appended pair
static_assert(alignof(opus_mqa_logits_kargs) == 8);
static_assert(offsetof(opus_mqa_logits_kargs, ptr_q             ) ==   0);
static_assert(offsetof(opus_mqa_logits_kargs, ptr_q_scale       ) ==   8);
static_assert(offsetof(opus_mqa_logits_kargs, ptr_kv            ) ==  16);
static_assert(offsetof(opus_mqa_logits_kargs, ptr_kv_scale      ) ==  24);
static_assert(offsetof(opus_mqa_logits_kargs, ptr_block_tables  ) ==  32);
static_assert(offsetof(opus_mqa_logits_kargs, ptr_weights       ) ==  40);
static_assert(offsetof(opus_mqa_logits_kargs, ptr_out           ) ==  48);
static_assert(offsetof(opus_mqa_logits_kargs, ptr_row_to_batch  ) ==  56);
static_assert(offsetof(opus_mqa_logits_kargs, ptr_local_starts  ) ==  64);
static_assert(offsetof(opus_mqa_logits_kargs, ptr_local_ends    ) ==  72);
static_assert(offsetof(opus_mqa_logits_kargs, ptr_cu_seq_q      ) ==  80);
static_assert(offsetof(opus_mqa_logits_kargs, split_kv          ) ==  88);
static_assert(offsetof(opus_mqa_logits_kargs, num_rows          ) ==  92);
static_assert(offsetof(opus_mqa_logits_kargs, num_batches       ) ==  96);
static_assert(offsetof(opus_mqa_logits_kargs, max_seq_len       ) == 100);
static_assert(offsetof(opus_mqa_logits_kargs, stride_out_row    ) == 104);
static_assert(offsetof(opus_mqa_logits_kargs, weight_scale      ) == 108);
static_assert(offsetof(opus_mqa_logits_kargs, block_k           ) == 112);
static_assert(offsetof(opus_mqa_logits_kargs, kv_block_size     ) == 116);
static_assert(offsetof(opus_mqa_logits_kargs, max_blocks_per_seq) == 120);
static_assert(offsetof(opus_mqa_logits_kargs, ptr_cta_info      ) == 128);
static_assert(offsetof(opus_mqa_logits_kargs, num_ctas          ) == 136);

__host__ __device__ inline int ceil_div_i(int a, int b) { return (a + b - 1) / b; }

// ── the schedule builder: split into its own file (arch-agnostic, shared) ──
#include "pa_mqa_logits_mxfp4_sched.cuh"

// Kept here, after the builder, for the reason the ordering note above gives -- it belongs with
// the ABI and cannot sit there.
namespace opus_logits {
// gfx950's two schedule-free modes are names only here; this target compiles `Table` and nothing
// else. Prefill gave every tile one CTA whatever its window length, which is the load imbalance
// the table exists to fix.
enum class mqa_logits_sched {
    Prefill,
    Decode,
    Table,
};
}

// ── gfx950 traits (MFMA 32x32x64) ────────────────────────────────────────────────────
// Traits for the MXFP4 paged MQA logits kernel on `mfma_scale_f32_32x32x64_f8f6f4`.
//
// D_DATA is deliberately not declared here: fp4 is opus::fp4_t, a device-side packed type,
// and this struct must stay host-compilable. The kernel template aliases it.
template<int KV_TILE_SIZE_ = 256,
         int PAGE_SIZE_    = 64,
         int HEAD_DIM_     = 128,
         int N_HEADS_      = 64,
         int NUM_WARPS_    = 4>
struct opus_mqa_logits_fp4_traits {
    static constexpr int KV_TILE_SIZE = KV_TILE_SIZE_;  // block_k
    static constexpr int PAGE_SIZE    = PAGE_SIZE_;     // kv_block_size
    static constexpr int HEAD_DIM     = HEAD_DIM_;
    static constexpr int N_HEADS      = N_HEADS_;
    static constexpr int NUM_WARPS    = NUM_WARPS_;

    static constexpr int WARP_SIZE  = 64;
    static constexpr int BLOCK_SIZE = NUM_WARPS * WARP_SIZE;

    using D_WEIGHT = mqa_logits_bf16_t;   // per-head weights (standalone spells this bf16_t)
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
    static constexpr int K_TILES  = HEAD_DIM / MFMA_K;   // 2  (outer K loop -- ACTIVE here)
    static constexpr int K_CHUNKS = MFMA_K / SCALE_BLOCK;// 2  (32-K scale blocks per k-tile)
    // Scale blocks across a whole head row, i.e. what the host quantizer produces per row.
    static constexpr int SCALE_BLOCKS_ROW = HEAD_DIM / SCALE_BLOCK;  // 4 == K_TILES * K_CHUNKS

    static constexpr int N_TOTAL_TILES   = KV_TILE_SIZE / MFMA_N;      // 8 (bk=256) / 2 (bk=64)
    static constexpr int N_TILES         = N_TOTAL_TILES / NUM_WARPS;  // 2 = NTPW, both variants
    static constexpr int TILES_PER_BLOCK = PAGE_SIZE / MFMA_N;         // 2 (MFMA_N tiles per page)
    static constexpr int N_PHYS = (N_TILES + TILES_PER_BLOCK - 1) / TILES_PER_BLOCK;  // 1

    // ---- C fragment geometry (from the probe / opus mfma_adaptor) ----
    // C is [(rept_c<y>, grpm_c<p>, pack_c<y>), (grpn_c<p>)]: m = rept*8 + (L/32)*4 + pack.
    static constexpr int C_FRAG = MFMA_M * MFMA_N / WARP_SIZE;  // 16 floats per lane per m-tile
    static constexpr int GRPN_C = MFMA_N;                       // 32: n == lane % 32
    static constexpr int GRPM_C = WARP_SIZE / GRPN_C;           // 2:  M spread over 2 lane groups
    static constexpr int PACK_C = 4;                            // 16 B of f32 per contiguous run
    static constexpr int REPT_C = C_FRAG / PACK_C;              // 4
    // Heads a single lane accumulates, across all m-tiles: M_TILES * C_FRAG = 32 of the 64.
    // The other 32 live in the partner lane group and arrive via one permlane32 swap.
    static constexpr int HEADS_PER_LANE = M_TILES * C_FRAG;     // 32

    static constexpr int DWORDx4_BYTES = 16;

    // ---- per-lane payload ----
    // A lane holds one contiguous 32-K block = 32 fp4 = 16 B. This is NOT the MFMA
    // operand width: opus always passes 256-bit operands and fp4 reads only the low 16 B,
    // so the kernel loads 16 B and fills the low half.
    static constexpr int KV_GRP_ELEMS = SCALE_BLOCK;                  // 32 fp4 per lane
    static constexpr int KV_GRP_BYTES = KV_GRP_ELEMS * ELEM_BITS / 8; // 16 bytes
    static constexpr int A_BYTES_PER_LANE = (MFMA_M * MFMA_K / WARP_SIZE) * ELEM_BITS / 8; // 16

    // ---- op_sel byte packing ----
    // One scale dword per lane serves the whole (kt, tile) double loop, since the dword
    // has 4 bytes and both products are 4:
    //     A: byte index = kt * M_TILES + mi     B: byte index = kt * N_TILES + nt
    static constexpr int QS_BYTES  = K_TILES * M_TILES;   // 4
    static constexpr int KVS_BYTES = K_TILES * N_TILES;   // 4
    static_assert(QS_BYTES == 4 && KVS_BYTES == 4,
                  "the op_sel byte packing assumes exactly 4 (kt, tile) pairs per dword");

    // ── LAYOUTS ─────────────────────────────────────────────────────────────
    // ---- q: natural [T][head][D/2] (NO preshuffle) ----
    // The stride_q_* below are vestigial (the natural make_layout_q does not use them); kept only
    // so Q_ROW_BYTES stays the size of one query row (H * D/2 = 4096 B, unchanged by the ABI).
    static constexpr int stride_q_m     = KV_GRP_BYTES;                    // 16
    static constexpr int stride_q_g     = MFMA_M * stride_q_m;             // 512
    static constexpr int stride_q_ktile = K_CHUNKS * stride_q_g;           // 1024
    static constexpr int stride_q_mtile = K_TILES * stride_q_ktile;        // 2048
    static constexpr int Q_ROW_BYTES    = M_TILES * stride_q_mtile;        // 4096 == H * D/2

    // ---- kv_cache: [num_blocks][kt(K_TILES)][g(K_CHUNKS)][nt(TILES_PER_BLOCK)][m(MFMA_N)][16 B]
    // holds KV[page token = nt*MFMA_N + m][K = (kt*K_CHUNKS + g)*32 : +32].
    //
    // Substituting b = kt*K_CHUNKS + g and o = nt*MFMA_N + m makes this the standard
    // paged fp4 layout [num_blocks][b(4)][o(PAGE)][16 B], so callers need no fp4-specific
    // kv writer. The static_assert below pins that equivalence.
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
    // In dwords: index = g*MFMA_N + m == L. One load per lane for the whole kernel.
    static constexpr int stride_qs_g_dw = MFMA_M;                          // 32 dwords
    static constexpr int QS_ROW_BYTES   = K_CHUNKS * MFMA_M * QS_BYTES;    // 256

    // ---- kv_scale: [num_blocks][g(K_CHUNKS)][m(MFMA_N)][byte(KVS_BYTES)] (e8m0) ----
    // byte (kt*N_TILES + nt) = e8m0(page token nt*32 + m, block kt*K_CHUNKS + g).
    static constexpr int stride_kvs_g_dw  = MFMA_N;                        // 32 dwords
    static constexpr int stride_kvs_block = K_CHUNKS * MFMA_N * KVS_BYTES; // 256 bytes

    // ---- weights: natural [T, H] bf16 ----
    static constexpr int stride_w_lg   = HEADS_PER_LANE;                   // 32 bf16 (vestigial)
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

// ── gfx1250 traits (WMMA 32x16x128) ──────────────────────────────────────────────────
// ONE kv_cache layout, the natural one: with no FlyDSL kernel on this target there is nothing
// to be byte-compatible with, so the LDS address formulas below are that layout's, once.
//
// KV_TILE_SIZE = 64 is ONE PAGE, and it is the width the accumulator is sized by: N_TILES is
// KV_TILE_SIZE / MMA_N, so the tile decides ACC_VGPR and ACC_VGPR decides how many waves the
// allocator fits per SIMD. At 64 that is three, at 128 it is one. **The caller's
// `cta_resident` follows that occupancy and has to be re-tuned when this number moves** -- a
// stale value under-sizes the grid, the split cannot fill the part, and nothing reports it.
template<int Q_PER_BLOCK_  = 4,
         int LDS_STAGES_   = 2,
         int KV_TILE_SIZE_ = 64,
         int PAGE_SIZE_    = 64,
         int HEAD_DIM_     = 128,
         int N_HEADS_      = 64>
struct opus_mqa_logits_fp4_qshare_traits {
    static constexpr int KV_TILE_SIZE = KV_TILE_SIZE_;  // block_k
    static constexpr int PAGE_SIZE    = PAGE_SIZE_;     // kv_block_size
    static constexpr int HEAD_DIM     = HEAD_DIM_;
    static constexpr int N_HEADS      = N_HEADS_;

    // A CONSTANT, never `opus::get_warp_size()`: that returns 64 in the host pass, which would
    // build the wave64 fragment layout with every byte count still matching.
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

    static constexpr int KV_CHUNK_BYTES = PAGE_SIZE * KV_GRP_BYTES;   // 1024

    // ── the ABI, as byte offsets within one row / one page ──
    // `block` is the 32-element E8M0 block index, and it sits INSIDE the token: a token's 128 fp4
    // elements are 64 contiguous bytes and block `b` is the 16 B at `b * 16`. LDS_BLOCK_BYTES below
    // takes it from here rather than a literal at the read site -- reading the wrong 16 B is
    // invisible in the resource report.
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

    // ── the measured fragment maps, shared by the kernel and the op test's reference ──
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

    // ── LDS: the tile image, and the one padding policy that breaks the bank conflict ──
    // TDM copies a page as it is, so the pad is the ONLY LDS-side freedom: WMMA fixes which lane
    // reads which token, so the lane mapping cannot be rearranged instead.
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

    // ── TDM. EVERY wave issues, and that is why the predicate is a compile-time constant and
    // the steady loop keeps no branch for it: two issuing waves out of four showed up as an
    // `s_cbranch_vccnz` per phase, splitting the loop into basic blocks. A page splits by
    // CONTIGUOUS BYTES -- not by token or by K block -- and wave w takes piece (w % SPLIT) of
    // page (w / SPLIT). ──
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

    // ── pipeline depth: it follows from the stage count, not the other way round ──
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

    // ── the accumulator: TWO copies, ping-ponged, so tile t+1's WMMAs issue while tile t is
    // reduced -- the two sets have no data dependency, so the scheduler may interleave a tile's
    // WMMAs with the previous tile's relu, weight and permlane work. Its width is what sets
    // occupancy: ACC_VGPR is M_TILES x N_TILES x C_FRAG per set, so the KV tile decides both. ──
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

using logits_fp4_qshare_traits_4q = opus_mqa_logits_fp4_qshare_traits<4>;

// ── device bodies: one file per arch, split out of this header. Each self-stubs on the
// wrong arch, so both launch symbols resolve in this one TU. ─────────────────────────
#include "pa_mqa_logits_mxfp4_gfx950.cuh"
#include "pa_mqa_logits_mxfp4_gfx1250.cuh"


#endif  // PA_MQA_LOGITS_MXFP4_IMPL
