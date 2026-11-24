// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// gfx950's MXFP4 MQA-logits instances: their traits and registry entries. DeepSeek-V4's 64 index
// heads and V4.1's 32, on one KV layout: the plane permuted 64 rows at a time, a 64-key tile one
// such group. The kernels are in pa_mqa_logits_mxfp4_gfx950_kv.cuh,
// pa_mqa_logits_mxfp4_gfx950_qshare.cuh and pa_mqa_logits_mxfp4_gfx950_rows.cuh.
//
// Built with `-mllvm -enable-post-misched=1` (optCompilerConfig.json's per-source flags),
// overriding aiter's default of 0: the kernels' MFMA / VALU interleave depends on it.
#include "pa_mqa_logits_mxfp4.h"

#include <type_traits>

namespace opus_logits::gfx950 {

// -- traits ---------------------------------------------------------------------------------------
// Host-safe: the host pass sizes the launch from them. No D_DATA: opus::fp4_t is device-only.
// D = 128, the KV cache writer's MFMA-permuted layouts: q [T, H, D/2] natural,
// q_scale [T, 2, 32, H / 16], kv [groups, 4, 64, 16], kv_scale [groups, 2, 32, 4]. A 64-row group
// is a V4 page, or eight V4.1 8-row pages (page p: rows (p % 8) * 8 .. + 7 of group p / 8); the
// table names whole groups (V4) or runs of them (V4.1 full layers, an entry 2^tpb_shift groups).

// What both kernels share: one wave computes a whole 64-key tile (one group) for one query row of
// N_HEADS heads, M_TILES MFMA m-tiles of 32 heads.
template<int N_HEADS_ = 64>
struct mqa_logits_traits {
    static constexpr int N_HEADS       = N_HEADS_;
    static constexpr int ROWS_PER_WAVE = 1;
    static constexpr int HEAD_DIM      = 128;
    static constexpr int PAGE_SIZE     = 64;   // the permuted group
    static constexpr int KV_TILE_SIZE  = PAGE_SIZE;
    static constexpr int WAVE_SIZE     = 64;
    static constexpr int SLOT_WAVES    = 1;    // a CTA is one schedule slot

    // A table entry names one tile; V4.1's logits may go to a flat out (out_offsets).
    static constexpr bool ENTRY_RUNS = false;
    static constexpr bool FLAT_OUT   = N_HEADS == 32;
    static constexpr bool KV_NT      = false;
    // V4.1 candidate layers (GATHER, kv_traits): a tile gathers TILE_PAGES 8-row pages, and each
    // n-tile holds GATHER_NT_KEYS consecutive keys of a lane's page.
    static constexpr bool GATHER         = false;
    static constexpr int  PAGE_ROWS      = 8;
    static constexpr int  TILE_PAGES     = KV_TILE_SIZE / PAGE_ROWS;   // 8

    using D_WEIGHT = __bf16;   // per-head weights
    using D_ACC    = float;    // MFMA accumulator
    using D_OUT    = float;    // output logits
    using D_SCALE  = int;      // a key's four E8M0 bytes, one dword a lane

    static constexpr int MMA_M = 32;   // heads
    static constexpr int MMA_N = 32;   // keys
    static constexpr int MMA_K = 64;   // head_dim

    static constexpr int ELEM_BITS   = 4;
    static constexpr int SCALE_BLOCK = 32;

    static constexpr int M_TILES        = N_HEADS / MMA_M;        // 2 (V4) / 1 (V4.1)
    static constexpr int N_TILES        = KV_TILE_SIZE / MMA_N;   // 2
    static constexpr int GATHER_NT_KEYS = PAGE_ROWS / N_TILES;    // 4
    static constexpr int K_TILES        = HEAD_DIM / MMA_K;       // 2: the outer K loop
    static constexpr int K_CHUNKS       = MMA_K / SCALE_BLOCK;    // 2: scale blocks a k-tile

    // C fragment of one m-tile: n = L % 32, m = rept * 8 + (L / 32) * 4 + pack.
    static constexpr int C_FRAG         = MMA_M * MMA_N / WAVE_SIZE;   // 16
    static constexpr int GRPN_C         = MMA_N;                        // 32
    static constexpr int GRPM_C         = WAVE_SIZE / GRPN_C;           // 2
    static constexpr int PACK_C         = 4;
    static constexpr int REPT_C         = C_FRAG / PACK_C;              // 4
    static constexpr int HEADS_PER_LANE = M_TILES * C_FRAG;             // 32 / 16

    // A lane holds one 32-element block, 16 B: the low half of the 256-bit MFMA operand.
    static constexpr int KV_GRP_BYTES     = SCALE_BLOCK * ELEM_BITS / 8;                 // 16
    static constexpr int A_BYTES_PER_LANE = MMA_M * MMA_K / WAVE_SIZE * ELEM_BITS / 8;   // 16

    static constexpr int Q_ROW_BYTES = N_HEADS * HEAD_DIM * ELEM_BITS / 8;   // 4096
    static constexpr int W_ROW_ELEMS = N_HEADS;                              // natural [T, H]

    // kv_cache [pages][kt][g][nt][m][16 B] holds block kt * K_CHUNKS + g of key nt * 32 + m:
    // the standard paged layout [pages][block (4)][key (64)][16 B].
    static constexpr int STRIDE_KV_M     = KV_GRP_BYTES;              // 16
    static constexpr int STRIDE_KV_NTILE = MMA_N * STRIDE_KV_M;       // 512
    static constexpr int STRIDE_KV_G     = N_TILES * STRIDE_KV_NTILE; // 1024
    static constexpr int STRIDE_KV_KTILE = K_CHUNKS * STRIDE_KV_G;    // 2048
    static constexpr int KV_PAGE_BYTES   = K_TILES * STRIDE_KV_KTILE; // 4096

    // q_scale [T][g][m][KT * M_TILES] / kv_scale [groups][g][m][4]: lane L = g * 32 + m reads
    // element L (a row's D_QSCALE, a group's dword), whose byte kt * tiles + tile is the E8M0 of
    // (head / key tile * 32 + m, block kt * K_CHUNKS + g). The MFMA's op_sel picks the byte:
    // kt * M_TILES + mi for A, kt * N_TILES + nt for B.
    using D_QSCALE = std::conditional_t<M_TILES == 2, int, unsigned short>;
    static constexpr int KVS_PAGE_BYTES = K_CHUNKS * MMA_N * 4;   // 256

    static_assert(M_TILES >= 1 && M_TILES <= 2, "a row is one or two m-tiles of heads");
    static_assert(N_TILES == 2 && K_TILES == 2, "the kernels unroll two n-tiles and two k-tiles");
    static_assert(K_TILES * M_TILES == (int)sizeof(D_QSCALE) && K_TILES * N_TILES == 4,
                  "a lane's scale element holds exactly its (kt, tile) bytes");
    static_assert(STRIDE_KV_G == PAGE_SIZE * KV_GRP_BYTES,
                  "a 32-element block's stride must span the whole page");
    static_assert(KV_GRP_BYTES == 16 && A_BYTES_PER_LANE == 16,
                  "a lane's fp4 payload is one 16 B load, the low half of the operand");
    static_assert(GRPM_C == 2 && HEADS_PER_LANE * GRPM_C == N_HEADS,
                  "a lane and its permlane32 partner must together hold every head");
};

// One query row a CTA, the KV read straight from global into a register double buffer. GATHER:
// a candidate layer's row (V4.1), a tile 8 arbitrary 8-row pages.
template<int N_HEADS_ = 64, bool GATHER_ = false>
struct kv_traits : mqa_logits_traits<N_HEADS_> {
    using base = mqa_logits_traits<N_HEADS_>;
    static constexpr bool GATHER    = GATHER_;
    static_assert(!GATHER || N_HEADS_ == 32, "a gathered tile is a candidate row's (V4.1)");
    static constexpr int NUM_WAVES  = 1;
    static constexpr int BLOCK_SIZE = NUM_WAVES * base::WAVE_SIZE;
    static constexpr int STAGES     = 2;
    static constexpr int SCALE_TDM  = 0;
};

// QPB query rows of one sequence a CTA, one a wave, sharing each 64-key tile through a ring of
// STAGES LDS stages (V4).
//
// The page copy: a 4096 B page is one buffer_load_lds of 16 B a lane per 4 waves. The first
// ISSUE_WAVES waves (the largest power of two <= QPB, at most 4) copy it, ASYNC_ISSUES issues
// each, into a byte-exact LDS image. Every wave copies its own image of the page's 256 B of
// scales (64 lanes x 4 B), so it reads one page id a phase.
template<int QPB_, int STAGES_ = 4>
struct qshare_traits : mqa_logits_traits<> {
    static constexpr int Q_PER_BLOCK = QPB_;
    static constexpr int NUM_WAVES   = Q_PER_BLOCK;
    static constexpr int BLOCK_SIZE  = NUM_WAVES * WAVE_SIZE;
    static constexpr int STAGES      = STAGES_;
    static constexpr int SCALE_TDM   = 1;   // the key scales are copied with the tile

    static constexpr int LDS_TILE_BYTES   = KV_PAGE_BYTES;
    static constexpr int ISSUE_WAVES      = Q_PER_BLOCK >= 4 ? 4 : Q_PER_BLOCK >= 2 ? 2 : 1;
    static constexpr int ASYNC_VEC_BYTES  = KV_GRP_BYTES;                  // 16
    static constexpr int ASYNC_WAVE_BYTES = WAVE_SIZE * ASYNC_VEC_BYTES;   // 1024
    static constexpr int ASYNC_ISSUES     = LDS_TILE_BYTES / (ISSUE_WAVES * ASYNC_WAVE_BYTES);
    static_assert(LDS_TILE_BYTES == ASYNC_ISSUES * ISSUE_WAVES * ASYNC_WAVE_BYTES,
                  "the issuing waves must tile the page exactly with 16 B a lane an issue");

    static constexpr int SCALE_PAGE_BYTES = KVS_PAGE_BYTES;   // 256
    static_assert(SCALE_PAGE_BYTES == WAVE_SIZE * 4,
                  "a wave must copy a page's scales in exactly one dword-a-lane issue");
    static constexpr int LDS_SCALE_BYTES = NUM_WAVES * SCALE_PAGE_BYTES;   // a stage's images

    // The data region first, the scale images after it.
    static constexpr int LDS_DATA_BYTES = STAGES * LDS_TILE_BYTES;
    static constexpr int LDS_BYTES      = LDS_DATA_BYTES + STAGES * LDS_SCALE_BYTES;

    // The pipe. Phase t ds_reads tile t + READ_LEAD into the registers the next phase's gemms use,
    // and issues the copy of tile t + ASYNC_LEAD. Writing stage (t + ASYNC_LEAD) % S clobbers the
    // tile read at phase t + ASYNC_LEAD - S - READ_LEAD, which must be an earlier phase so the
    // barrier at the top of phase t separates them: ASYNC_LEAD <= S + READ_LEAD - 1.
    static constexpr int READ_LEAD  = 2;
    static constexpr int ASYNC_LEAD = STAGES + READ_LEAD - 1;
    static_assert(STAGES >= 3 && (STAGES & (STAGES - 1)) == 0,
                  "STAGES: a power of two (the stage index is a mask), at least 3 (the peeled "
                  "phases' waits assume tile READ_LEAD is a fill stage)");

    // vmcnt, a wave's own vmem ops (gfx9 counts the stores too). The prologue copies tiles
    // 0 .. S - 1, then tile S into the stage tile 0 vacated; phase p copies tile p + S + 1 and
    // ends with N_TILES stores. A copy is the page (ASYNC_ISSUES, issuing waves only) and the
    // scales (1). The wait at the top of phase t must retire tile t + READ_LEAD's copies; behind
    // them sit the copies of the S - 2 tiles after it and the stores of every phase since:
    // min(t, S - 1) phases, as the prologue stores nothing. The loop's first phase, t = 2, has
    // the fewest. A smaller keep waits for more, so with waves that do not copy the page every
    // wave takes their count.
    static constexpr int STORES_PER_PHASE = N_TILES;
    static constexpr int COPIES           = (ISSUE_WAVES == NUM_WAVES ? ASYNC_ISSUES : 0) + 1;
    static constexpr int KEEP_STEADY      = (STAGES - 2) * COPIES + 2 * STORES_PER_PHASE;
    // Phases 0 and 1 (peeled): no store before phase 0's wait, so behind tile READ_LEAD's copies
    // are the rest of the fill and the prologue's refill of stage 0. Phase 1 has more behind it.
    static constexpr int KEEP_PEELED = (STAGES - 2) * COPIES;
    // The prologue's waits for tiles 0 and 1. Q / q_scale / weights go out before any copy, so
    // retiring the first copy retires them too.
    static constexpr int PROLOGUE_KEEP0 = (STAGES - 1) * COPIES;
    static constexpr int PROLOGUE_KEEP1 = (STAGES - 2) * COPIES;

    static_assert(Q_PER_BLOCK >= 1 && Q_PER_BLOCK <= 16, "Q_PER_BLOCK out of range");
};

// V4.1 full layers (32 heads): ROWS_PER_WAVE query rows of one sequence a wave, MFMA 32x32x64,
// STAGES KV tiles in registers (rows_kernel). The same layouts: q_scale [T, 2, 32, 2],
// kv [groups, 4, 64, 16], kv_scale [groups, 2, 32, 4]. SLOT_WAVES waves a CTA, each a schedule
// slot of its own (instance_kernels counts grids and residency in slots).
template<int ROWS_PER_WAVE_, bool KV_NT_, int STAGES_ = 3, int SLOT_WAVES_ = 1, int MIN_WAVES_ = 2>
struct rows_traits {
    static constexpr int  N_HEADS       = 32;
    static constexpr int  ROWS_PER_WAVE = ROWS_PER_WAVE_;
    static constexpr int  HEAD_DIM      = 128;
    static constexpr int  PAGE_SIZE     = 64;   // the permuted group, a tile
    static constexpr int  KV_TILE_SIZE  = PAGE_SIZE;
    static constexpr int  PAGE_ROWS     = 8;    // V4.1's page
    static constexpr int  WAVE_SIZE     = 64;
    static constexpr int  NUM_WAVES     = 1;   // the rows a slot: rows_per_cta = ROWS_PER_WAVE
    static constexpr int  SLOT_WAVES    = SLOT_WAVES_;
    static constexpr int  BLOCK_SIZE    = WAVE_SIZE * SLOT_WAVES;
    static constexpr int  STAGES        = STAGES_;
    static constexpr int  SCALE_TDM     = 0;
    static constexpr bool GATHER        = false;
    static constexpr bool ENTRY_RUNS    = true;
    static constexpr bool FLAT_OUT      = true;
    // Nontemporal KV loads (buffer aux 2) where no other CTA reads the walk again (decode): they
    // stream past the caches. Prefill re-reads a sequence's keys group after group from L2.
    static constexpr bool KV_NT         = KV_NT_;
    static constexpr int  KV_AUX        = KV_NT ? 2 : 0;
    static constexpr int  MIN_WAVES     = MIN_WAVES_;   // a SIMD's, __launch_bounds__

    using D_WEIGHT = __bf16;
    using D_OUT    = float;

    static constexpr int MMA_M = 32;   // heads
    static constexpr int MMA_N = 32;   // keys
    static constexpr int MMA_K = 64;   // head_dim

    static constexpr int N_TILES = KV_TILE_SIZE / MMA_N;        // 2
    static constexpr int K_TILES = HEAD_DIM / MMA_K;            // 2
    static constexpr int C_FRAG  = MMA_M * MMA_N / WAVE_SIZE;   // 16

    static constexpr int KV_GRP_BYTES   = 16;                          // a lane's 32 fp4
    static constexpr int Q_HEAD_BYTES   = HEAD_DIM / 2;                // 64
    static constexpr int Q_ROW_BYTES    = N_HEADS * Q_HEAD_BYTES;      // 2048
    static constexpr int W_ROW_ELEMS    = N_HEADS;
    static constexpr int QS_ROW_SHORTS  = 2 * N_HEADS;                 // [g][m] shorts (kt0, kt1)
    static constexpr int STRIDE_KV_M    = KV_GRP_BYTES;                // key
    static constexpr int STRIDE_KV_G    = PAGE_SIZE * KV_GRP_BYTES;    // a 32-element block: 1 KB
    static constexpr int KV_PAGE_BYTES  = HEAD_DIM / 32 * STRIDE_KV_G; // 4096
    static constexpr int KVS_PAGE_BYTES = PAGE_SIZE * 4;               // 256

    static_assert(ROWS_PER_WAVE >= 1 && ROWS_PER_WAVE <= 8 && STAGES >= 2, "rows / stages");
    static_assert(N_TILES == 2 && K_TILES == 2 && C_FRAG == 16, "the kernel's item shape");
};

}  // namespace opus_logits::gfx950

// The device pass on gfx950 gets the kernels; every other pass gets empty stubs so the launch
// symbols resolve.
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx950__)
#include "pa_mqa_logits_mxfp4_gfx950_kv.cuh"
#include "pa_mqa_logits_mxfp4_gfx950_qshare.cuh"
#include "pa_mqa_logits_mxfp4_gfx950_rows.cuh"
#else
namespace opus_logits::gfx950 {
template<class T, int SCHED> __global__ void kv_kernel(mqa_logits_mxfp4_kargs) {}
template<class T, int SCHED> __global__ void qshare_kernel(mqa_logits_mxfp4_kargs) {}
template<class T, int SCHED> __global__ void rows_kernel(mqa_logits_mxfp4_kargs) {}
}  // namespace opus_logits::gfx950
#endif

namespace opus_logits::gfx950 {
namespace {

// V4: six- and eight-row qshare CTAs are not registered: one such CTA fills a CU and its waves
// stall at the per-phase barrier together, where two or four smaller CTAs a CU overlap theirs.
using h64_r1w1s2L = kv_traits<>;        // decode, 1 row a sequence; prefill
using h64_r1w2s4T = qshare_traits<2>;   // decode, 2 rows a sequence; 5-6 (three groups)
using h64_r1w4s4T = qshare_traits<4>;   // decode, 3-4 rows; 7-8 (two groups)
// V4.1 full layers: a wave a sequence's rows (up to eight), nontemporal for decode (no other CTA
// reads a walk again), cached for prefill (the groups of a sequence re-read its keys). Decode
// also has four-wave CTAs, a slot a wave, for large batches: a CTA's slots are cut by one wave
// and the grid is a quarter the CTAs.
using h32_r1w1s3Ln   = rows_traits<1, true>;
using h32_r2w1s3Ln   = rows_traits<2, true>;
using h32_r3w1s3Ln   = rows_traits<3, true>;
using h32_r4w1s2Ln   = rows_traits<4, true, 2>;
using h32_r6w1s3Ln   = rows_traits<6, true>;
using h32_r8w1s3Ln   = rows_traits<8, true>;
using h32_r1w1s3Lnx4 = rows_traits<1, true, 3, 4>;
using h32_r2w1s3Lnx4 = rows_traits<2, true, 3, 4>;
using h32_r3w1s3Lnx4 = rows_traits<3, true, 3, 4>;
using h32_r4w1s2Lnx4 = rows_traits<4, true, 2, 4>;
using h32_r6w1s3Lnx4 = rows_traits<6, true, 3, 4>;
using h32_r8w1s3Lnx4 = rows_traits<8, true, 3, 4>;
using h32_r4w1s3L    = rows_traits<4, false>;
// V4.1 candidate layers.
using h32g_r1w1s2L = kv_traits<32, true>;

// A table entry is one 64-row group (V4: a page; pages_per_block 1), or (V4.1 full layers) a
// power-of-two run of up to 2^MAX_TPB_SHIFT of them: the MFMA-permuted kv_cache / kv_scale are laid
// out a group at a time. V4.1 candidate layers: an 8-row page.
// A V# a page (GATHER: 64-bit addresses), so no buffer resource bounds the kv_cache. GATHER clamps
// its page ids into the plane by kv_last_row, a 32-bit key index: 2^32 keys.
template<class T, class K>
constexpr mqa_logits_instance entry() {
    constexpr int run_max = T::ENTRY_RUNS ? T::PAGE_SIZE << MAX_TPB_SHIFT : T::PAGE_SIZE;
    return {.arch           = "gfx950",
            .heads          = T::N_HEADS,
            .rows_per_wave  = T::ROWS_PER_WAVE,
            .waves          = T::NUM_WAVES,
            .stages         = T::STAGES,
            .scale_tdm      = T::SCALE_TDM,
            .gather         = T::GATHER,
            .entry_rows_min = T::GATHER ? T::PAGE_ROWS : T::PAGE_SIZE,
            .entry_rows_max = T::GATHER ? T::PAGE_ROWS : run_max,
            .tile_keys      = T::KV_TILE_SIZE,
            .modes          = K::modes,
            .flat_out       = T::FLAT_OUT,
            .kv_nt          = T::KV_NT,
            .slot_waves     = T::SLOT_WAVES,
            .max_kv_bytes   = T::GATHER ? (1LL << 32) * T::KV_PAGE_BYTES / T::PAGE_SIZE : 0,
            .launch         = K::launch,
            .occupancy      = K::occupancy};
}

template<class T>
constexpr mqa_logits_instance kv_entry() {
    using K = instance_kernels<T::BLOCK_SIZE, kv_kernel<T, SCHED_TABLE>, kv_kernel<T, SCHED_FREE>,
                               kv_kernel<T, SCHED_PREFILL>>;
    return entry<T, K>();
}

template<class T>
constexpr mqa_logits_instance rows_entry() {
    using K = instance_kernels<T::BLOCK_SIZE, rows_kernel<T, SCHED_TABLE>,
                               rows_kernel<T, SCHED_FREE>, rows_kernel<T, SCHED_PREFILL>,
                               T::SLOT_WAVES>;
    return entry<T, K>();
}

template<class T>
constexpr mqa_logits_instance qshare_entry() {
    using K = instance_kernels<T::BLOCK_SIZE, qshare_kernel<T, SCHED_TABLE>,
                               qshare_kernel<T, SCHED_FREE>, qshare_kernel<T, SCHED_PREFILL>>;
    return entry<T, K>();
}

}  // namespace
}  // namespace opus_logits::gfx950

namespace opus_logits {

const mqa_logits_instance* mqa_logits_gfx950_instances(int* count) {
    using namespace gfx950;
    static const mqa_logits_instance table[] = {
        kv_entry<h64_r1w1s2L>(),
        qshare_entry<h64_r1w2s4T>(),
        qshare_entry<h64_r1w4s4T>(),
        rows_entry<h32_r1w1s3Ln>(),
        rows_entry<h32_r2w1s3Ln>(),
        rows_entry<h32_r3w1s3Ln>(),
        rows_entry<h32_r4w1s2Ln>(),
        rows_entry<h32_r6w1s3Ln>(),
        rows_entry<h32_r8w1s3Ln>(),
        rows_entry<h32_r1w1s3Lnx4>(),
        rows_entry<h32_r2w1s3Lnx4>(),
        rows_entry<h32_r3w1s3Lnx4>(),
        rows_entry<h32_r4w1s2Lnx4>(),
        rows_entry<h32_r6w1s3Lnx4>(),
        rows_entry<h32_r8w1s3Lnx4>(),
        rows_entry<h32_r4w1s3L>(),
        kv_entry<h32g_r1w1s2L>(),
    };
    *count = static_cast<int>(sizeof(table) / sizeof(table[0]));
    return table;
}

}  // namespace opus_logits
