// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// gfx950 (MFMA 32x32x64, wave64) traits for the MXFP4 MQA-logits kernels. Host-safe: the gfx950
// TU's host pass sizes the launch from them. No D_DATA: opus::fp4_t is device-only.
//
// MFMA-permuted layouts (the KV cache writer's), 64-row pages:
//   q [T, H, D/2] natural, q_scale [T, 2, 32, 4], kv [pages, 4, 64, 16], kv_scale [pages, 2, 32, 4].
#pragma once
#include <cstddef>

namespace opus_logits::gfx950 {

// One query row a CTA (the stock kernel): NUM_WARPS waves split one KV_TILE_SIZE-key tile, two
// 32-key n-tiles each, so a 64-key tile is one wave and a 256-key tile four.
template<int KV_TILE_SIZE_ = 64,
         int NUM_WARPS_    = 1,
         int N_HEADS_      = 64,
         int PAGE_SIZE_    = 64,
         int HEAD_DIM_     = 128>
struct mqa_logits_traits {
    static constexpr int KV_TILE_SIZE = KV_TILE_SIZE_;  // block_k
    static constexpr int PAGE_SIZE    = PAGE_SIZE_;     // kv_block_size
    static constexpr int HEAD_DIM     = HEAD_DIM_;
    static constexpr int N_HEADS      = N_HEADS_;
    static constexpr int NUM_WARPS    = NUM_WARPS_;

    static constexpr int WARP_SIZE  = 64;
    static constexpr int BLOCK_SIZE = NUM_WARPS * WARP_SIZE;

    using D_WEIGHT = __bf16;   // per-head weights
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
    static constexpr int Q_ROW_BYTES = N_HEADS * HEAD_DIM * ELEM_BITS / 8;   // 4096

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
    static constexpr int QS_ROW_BYTES = K_CHUNKS * MFMA_M * QS_BYTES;      // 256

    // ---- kv_scale: [num_blocks][g(K_CHUNKS)][m(MFMA_N)][byte(KVS_BYTES)] (e8m0) ----
    // byte (kt*N_TILES + nt) = e8m0(page token nt*32 + m, block kt*K_CHUNKS + g).
    static constexpr int stride_kvs_block = K_CHUNKS * MFMA_N * KVS_BYTES; // 256 bytes

    static constexpr int PAGES_PER_TILE = KV_TILE_SIZE / PAGE_SIZE;         // 4 (bk=256) / 1

    // ---- weights: natural [T, H] bf16 ----
    static constexpr int W_ROW_ELEMS = GRPM_C * HEADS_PER_LANE;            // 64 == N_HEADS

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
    static_assert(W_ROW_ELEMS == N_HEADS, "weights must stay natural [T, H]");
};

// QPB query rows of one sequence a CTA, one a wave, sharing each 64-key tile through LDS (qshare,
// opus-ops opus_logits/gfx950 FP4_HANDOFF §19, §26). A wave is the one-wave stock geometry over
// the WHOLE tile (MT = 2, NTPW = 2, KT = 2): only BLOCK_SIZE and the LDS pipe are new.
//
// THE PAGE COPY. A 64-key page is 4096 B, one buffer_load_lds of 16 B a lane per 4 waves. The
// first ISSUE_WAVES waves (the largest power of two <= QPB, at most 4) copy it, ASYNC_ISSUES
// issues each, into a byte-exact LDS image: 1 / 2 / 4 rows a CTA all of them, 6 / 8 rows the first
// four. Every wave copies its own image of the page's 256 B of scales (64 lanes x 4 B: the 1-byte
// form lands lane l at M0 + 2l on gfx950), so it reads one page id a phase.
template<int QPB_, int LDS_STAGES_ = 4>
struct qshare_traits : mqa_logits_traits<64, 1> {
    using Base = mqa_logits_traits<64, 1>;

    static constexpr int Q_PER_BLOCK = QPB_;               // query rows a CTA == waves
    static constexpr int NUM_WAVES   = Q_PER_BLOCK;
    // Shadows Base::BLOCK_SIZE (the one-wave value).
    static constexpr int BLOCK_SIZE  = NUM_WAVES * Base::WARP_SIZE;

    static constexpr int LDS_STAGES     = LDS_STAGES_;
    static constexpr int LDS_TILE_BYTES = Base::stride_kv_block;   // 4096 == one page

    static constexpr int ISSUE_WAVES     = Q_PER_BLOCK >= 4 ? 4 : Q_PER_BLOCK >= 2 ? 2 : 1;
    static constexpr int ASYNC_VEC_BYTES = Base::KV_GRP_BYTES;     // 16 (one dwordx4)
    static constexpr int ASYNC_WAVE_BYTES = Base::WARP_SIZE * ASYNC_VEC_BYTES;   // 1024
    static constexpr int ASYNC_ISSUES    = LDS_TILE_BYTES / (ISSUE_WAVES * ASYNC_WAVE_BYTES);
    static_assert(LDS_TILE_BYTES == ASYNC_ISSUES * ISSUE_WAVES * ASYNC_WAVE_BYTES,
                  "the issuing waves must tile the page exactly with 16 B a lane an issue");

    static constexpr int SCALE_PAGE_BYTES = Base::stride_kvs_block;               // 256
    static constexpr int SCALE_VEC_BYTES  = 4;
    static_assert(SCALE_PAGE_BYTES == Base::WARP_SIZE * SCALE_VEC_BYTES,
                  "a wave must copy a page's scales in exactly one dword-a-lane issue");
    static constexpr int LDS_SCALE_BYTES = NUM_WAVES * SCALE_PAGE_BYTES;   // a stage's images

    // Data region first so its addressing is untouched, scales appended.
    static constexpr int LDS_DATA_BYTES = LDS_STAGES * LDS_TILE_BYTES;
    static constexpr int LDS_BYTES      = LDS_DATA_BYTES + LDS_STAGES * LDS_SCALE_BYTES;

    // ── the pipe ──
    // Phase t ds_reads tile t + READ_LEAD into the registers the next phase's gemms use, and
    // issues the copy of tile t + ASYNC_LEAD. Writing stage (t + ASYNC_LEAD) % S clobbers the tile
    // read at phase t + ASYNC_LEAD - S - READ_LEAD, which must be an earlier phase so the barrier
    // at the top of phase t separates them: ASYNC_LEAD <= S + READ_LEAD - 1, the copy's shadow
    // S - 1 phases.
    static constexpr int READ_LEAD  = 2;
    static constexpr int ASYNC_LEAD = LDS_STAGES + READ_LEAD - 1;
    static_assert(LDS_STAGES >= 3 && (LDS_STAGES & (LDS_STAGES - 1)) == 0,
                  "LDS_STAGES: a power of two (the stage index is a mask), at least 3 (the "
                  "peeled phases' waits below assume tile READ_LEAD is a fill stage)");

    // ── vmcnt ledgers, a wave's own vmem ops (gfx9 counts the stores too) ──
    // The prologue copies tiles 0 .. S - 1, then tile S into the stage tile 0 vacated; phase p
    // copies tile p + S + 1 and ends with N_TILES stores. A copy is the page (ASYNC_ISSUES,
    // issuing waves only) and the scale (1). The wait at the top of phase t must retire tile
    // t + READ_LEAD's copies. Behind them sit the copies of the S - 2 tiles after it and the
    // stores of every phase since -- min(t, S - 1) phases, as the prologue stores nothing. The
    // loop's first phase, t = 2, has the fewest: (S - 2) copies + 2 phases' stores. Counting the
    // copying phase's stores as well (S - 2 whole phases + its stores, true only from
    // t = S - 1) let phase 2 read tile 4 before it landed: qshare-2's fifth tile wrong under
    // load. At S = 4 this equals (S - 2) * (COPIES + STORES_PER_PHASE); deeper rings pay the
    // first phase's count on every phase, so they would want phases 2 .. S - 2 peeled.
    // Undercounting is safe (it waits for more), so with waves that do not copy the page every
    // wave takes their count.
    static constexpr int STORES_PER_PHASE = Base::N_TILES;
    static constexpr int COPIES = (ISSUE_WAVES == NUM_WAVES ? ASYNC_ISSUES : 0) + 1;
    static constexpr int KEEP_STEADY = (LDS_STAGES - 2) * COPIES + 2 * STORES_PER_PHASE;
    // The peeled phases 0 and 1: no store issued before phase 0's wait, so after tile
    // READ_LEAD's copies come the rest of the fill and the prologue's refill of stage 0:
    // S - 2 tiles' copies. Phase 1 has strictly more behind its target (phase 0's copies and
    // stores), so the same count undercounts it.
    static constexpr int KEEP_PEELED = (LDS_STAGES - 2) * COPIES;
    // The prologue's two waits. Q / q_scale / weights go out before any copy, so retiring the
    // first copy retires them too. 0: after the fill, retire tile 0's. 1: tile 1's.
    static constexpr int PROLOGUE_KEEP0 = (LDS_STAGES - 1) * COPIES;
    static constexpr int PROLOGUE_KEEP1 = (LDS_STAGES - 2) * COPIES;

    static_assert(Base::PAGE_SIZE == Base::KV_TILE_SIZE,
                  "one tile must be one page: the copy addresses a single block_tables entry");
    static_assert(Q_PER_BLOCK >= 1 && Q_PER_BLOCK <= 16, "Q_PER_BLOCK out of range");
};

}  // namespace opus_logits::gfx950
