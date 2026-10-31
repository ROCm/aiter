// SPDX-License-Identifier: MIT
// LG-QSDMA-r6x poster path (device pass only; UNCOMPILED): r6b poster + indexer/reuse items (from r5x) + puts-per-peer NP as a TEMPLATE parameter (entries n1/n2/n4 in one HSACO).
// Differences vs r6b (everything else is the r6b text, copied by build step from candidates/LG-QSDMA-r6/device/lg_r6_poster.hpp r6b+seams):
//  * NP (puts per peer) is a template parameter instead of a build constant; the group target is (chunks/NP) * LG_ITEMS_PER_GROUP (9 dense, 11 indexer).
//  * item order r6x: dense = r6b order (g-major, slot-major); indexer = per (put-group g) [per chunk-in-group: IK, IQ0, IQ1] then the dense slot-major remote tiles; self tiles last.
//  * publish is multi-destination: ONE wait/barrier/release fence per WG, then ONE relaxed (or acquire, LG_R6_ARRIVAL_ACQ) RMW per distinct REMOTE destination counter (review r5x (1)(b)).
//  * X-1 immunity: see the comment block at r6_poster_block (put only after the bounded wait for the full group count; arrival only after ALL groups were put and drained).
#pragma once
#include "lg_comm_device.hpp"
#include "layout/quad_fp8_fieldmap.hpp"

namespace lgc {
static_assert(LG_R6_POSTER == 1 && LG_REMOTE_HALF == 0, "r6x: poster path with full tiles only");
static_assert(LG_L2A_BYPASS == 0 && !LG_NEGCTRL_NOWAIT && !LG_NEGCTRL_ISO && !LG_L1C_SWIZZLE, "r6x: L2b only; no L2a/NOWAIT/ISO/L1c");
enum { T_NORMAL = 0, T_IQ = 1, T_IK = 2, T_HN = 3, T_HQ = 4, T_HK = 5, T_QN = 6 };   // T_H*: r7 half-N (256x128) items of the half-first plan
struct XItem { int type; int chunk; int dslot; int nloc; int nsel; };   // nsel: half selector of T_HN (0/1); T_QN: hn = nsel & 1 (column half), hm = nsel >> 1 (row half); 0 elsewhere      // NORMAL: dslot 0..2 remote peers, 3 = self, nloc 0..8; IQ: nloc = pair j (0,1); IK: -
constexpr int XEXTRA = LG_X_INDEXER ? 5 : 0;                      // IK + IQ_0..IQ_3 (one full tile each, ONE destination) per chunk completing remote groups
constexpr int r6x_group_items(int chunks, int np) { return (chunks / np) * static_cast<int>(LG_ITEMS_PER_GROUP); }
constexpr int grid_items_x(int chunks) { return static_cast<int>(LG_X_ITEMS_PER_CHUNK) * chunks; }
// remote items per put-group g: cpp * (XEXTRA + 27); remote phase = groups g = 0..np-1 (g-major); self phase = chunk-major 9 normal self tiles
constexpr XItem select_item_r6x(int t, int chunks, int np) {
    const int cpp = chunks / np, per_g = cpp * (XEXTRA + NPEER * NPD), Rn = per_g * np;
    if (t < Rn) {
        const int g = t / per_g, w = t % per_g;
        if (LG_X_INDEXER && w < cpp * XEXTRA) {                  // IK/IQ items first within the group, chunk-in-group major
            const int c = g * cpp + w / XEXTRA, k = w % XEXTRA;
            return k == 0 ? XItem{T_IK, c, 0, 0} : XItem{T_IQ, c, 0, k - 1};   // IQ: nloc = destination RANK d (0..3); IQ_rank is self-only (no arrival)
        }
        const int v = w - cpp * XEXTRA;                            // dense r6b order: slot-major, then (chunk-in-group, tile column)
        const int S = cpp * NPD, slot = v / S, u = v % S;
        return {T_NORMAL, g * cpp + u / NPD, slot, u % NPD};
    }
    const int s = t - Rn;
    return {T_NORMAL, s / NPD, NPEER, s % NPD};
}
constexpr bool r6x_order_ok(int chunks, int np) {                // bijection + coverage + arrivals per (remote dst, group); rank-0 view
    bool seen_n[32][4][9] = {}; bool seen_q[32][4] = {}; bool seen_k[32] = {};
    int arr[4][4] = {}; int raw_cnt[80] = {}; int pk_cnt[4][20] = {};
    const int cpp = chunks / np;
    for (int t = 0; t < grid_items_x(chunks); ++t) {
        const XItem x = select_item_r6x(t, chunks, np);
        if (x.chunk < 0 || x.chunk >= chunks) return false;
        const int g = x.chunk / cpp;
        if (x.type == T_NORMAL) {
            if (x.dslot < 0 || x.dslot > NPEER || x.nloc < 0 || x.nloc >= NPD) return false;
            const int dst = x.dslot < NPEER ? (x.dslot + 1) : 0;
            if (seen_n[x.chunk][dst][x.nloc]) return false;
            seen_n[x.chunk][dst][x.nloc] = true;
            for (int hn = 0; hn < 2; ++hn) {
                const int hv = (dst * NPD + x.nloc) * 2 + hn;
                if (x.chunk == 0) { ++raw_cnt[lgq::ht_raw(hv) / 128]; ++pk_cnt[lgq::ht_dest(hv)][lgq::ht_packed(hv) / 128]; }
                if (lgq::ht_dest(hv) != dst) return false;
            }
            if (dst != 0) ++arr[dst][g];
            if (dst != 0 && x.dslot >= 0 && g != (x.chunk / cpp)) return false;
        } else if (x.type == T_IQ) {
            if (!LG_X_INDEXER || x.nloc < 0 || x.nloc > 3 || seen_q[x.chunk][x.nloc]) return false;
            seen_q[x.chunk][x.nloc] = true;
            { const int d = x.nloc; if (x.chunk == 0) { ++raw_cnt[(LG_X_IQ_RAW + d * 128) / 128]; ++pk_cnt[d][LG_X_IQ_PACKED / 128]; } if (d != 0) ++arr[d][g]; }
        } else {
            if (!LG_X_INDEXER || seen_k[x.chunk]) return false;
            seen_k[x.chunk] = true;
            if (x.chunk == 0) { ++raw_cnt[LG_X_IK_RAW / 128]; for (int d = 0; d < 4; ++d) ++pk_cnt[d][LG_X_IK_PACKED / 128]; }
            for (int d = 1; d < 4; ++d) ++arr[d][g];
        }
    }
    const int nseg = LG_X_INDEXER ? 77 : 72, npk = LG_X_INDEXER ? 20 : 18;
    for (int i = 0; i < nseg; ++i) if (raw_cnt[i] != 1) return false;
    for (int d = 0; d < 4; ++d) for (int p = 0; p < npk; ++p) if (pk_cnt[d][p] != 1) return false;
    for (int d = 1; d < 4; ++d) for (int g = 0; g < np; ++g) if (arr[d][g] != r6x_group_items(chunks, np)) return false;
    return true;
}
static_assert(r6x_order_ok(8, 1) && r6x_order_ok(8, 2) && r6x_order_ok(8, 4) && r6x_order_ok(16, 1) && r6x_order_ok(16, 2) && r6x_order_ok(16, 4) &&
              r6x_order_ok(32, 1) && r6x_order_ok(32, 2) && r6x_order_ok(32, 4), "r6x order: bijection, coverage, (chunks/np)*items_per_group arrivals per remote (dst, group), chunks 8/16/32 x np 1/2/4");
static_assert(select_item_r6x(0, 8, 2).dslot == 0 && select_item_r6x(grid_items_x(8) - 1, 8, 2).dslot == NPEER && (LG_X_INDEXER || select_item_r6x(3, 8, 1).type == T_NORMAL), "remote first, self last");

// ======================= r7: half-N FIRST group (HG) =======================================================================================================================
// At M=2048 every remote full tile runs in wave 1, so the first put waits for wave 1 (~100 us). With HG, group 0 of every peer is made of HALF-N (256x128) items (each ~1/2 tile time):
// group 0 completes after ~1/2 tile time and its put starts earlier. Later groups and self tiles stay full tiles. IK/IQ items are HALF items (one 128-col segment each, no duplicate) in ALL groups.
// Group g arrivals per remote (dst): g==0: 9 tiles x 2 halves (+ IQ_dst + IK) per chunk; g>=1: 9 (+2) per chunk.
constexpr int r7_want(int chunks, int np, int g) { return (chunks / np) * (static_cast<int>(LG_ITEMS_PER_GROUP) + (g == 0 ? static_cast<int>(NPD) : 0)); }
constexpr int grid_items_h(int chunks, int np, int hg) { return static_cast<int>(LG_X_ITEMS_PER_CHUNK) * chunks + (hg ? NPEER * NPD * (chunks / np) : 0); }
constexpr XItem select_item_r7(int t, int chunks, int np) {
    const int cpp = chunks / np, g0 = cpp * (XEXTRA + 2 * NPEER * NPD), gN = cpp * (XEXTRA + NPEER * NPD), Rn = g0 + (np - 1) * gN;
    if (t < Rn) {
        const int g = (t < g0) ? 0 : 1 + (t - g0) / gN, w = (t < g0) ? t : (t - g0) % gN;
        if (LG_X_INDEXER && w < cpp * XEXTRA) {                  // IK_H / IQ_H_d first within the group, chunk-in-group major
            const int c = g * cpp + w / XEXTRA, k = w % XEXTRA;
            return k == 0 ? XItem{T_HK, c, 0, 0, 0} : XItem{T_HQ, c, 0, k - 1, 0};
        }
        const int v = w - cpp * XEXTRA;
        if (g == 0) {                                              // halves: slot-major, then (chunk-in-group, tile column, nsel); the two halves of a tile are adjacent
            const int S = cpp * NPD * 2, slot = v / S, u = v % S, u2 = u / 2;
            return {T_HN, g * cpp + u2 / NPD, slot, u2 % NPD, u % 2};
        }
        const int S = cpp * NPD, slot = v / S, u = v % S;           // full tiles (r6b order)
        return {T_NORMAL, g * cpp + u / NPD, slot, u % NPD, 0};
    }
    const int s = t - Rn;
    return {T_NORMAL, s / NPD, NPEER, s % NPD, 0};
}
constexpr bool r7_order_ok(int chunks, int np) {
    bool seen_n[32][4][9][2] = {}; bool seen_q[32][4] = {}; bool seen_k[32] = {};
    int arr[4][4] = {}; int raw_cnt[80] = {}; int pk_cnt[4][20] = {};
    const int cpp = chunks / np;
    for (int t = 0; t < grid_items_h(chunks, np, 1); ++t) {
        const XItem x = select_item_r7(t, chunks, np);
        if (x.chunk < 0 || x.chunk >= chunks) return false;
        const int g = x.chunk / cpp;
        if (x.type == T_NORMAL || x.type == T_HN) {
            if (x.dslot < 0 || x.dslot > NPEER || x.nloc < 0 || x.nloc >= NPD) return false;
            if (x.type == T_HN && (g != 0 || x.dslot == NPEER || x.nsel < 0 || x.nsel > 1)) return false;   // halves only in group 0, remote only
            if (x.type == T_NORMAL && g == 0 && x.dslot != NPEER) return false;                              // group 0 remote tiles must be halves
            const int dst = x.dslot < NPEER ? (x.dslot + 1) : 0;
            for (int hn = 0; hn < 2; ++hn) {
                if (x.type == T_HN && hn != x.nsel) continue;
                if (seen_n[x.chunk][dst][x.nloc][hn]) return false;
                seen_n[x.chunk][dst][x.nloc][hn] = true;
                const int hv = (dst * NPD + x.nloc) * 2 + hn;
                ++raw_cnt[lgq::ht_raw(hv) / 128]; ++pk_cnt[lgq::ht_dest(hv)][lgq::ht_packed(hv) / 128];
                if (lgq::ht_dest(hv) != dst) return false;
            }
            if (dst != 0) ++arr[dst][g];
        } else if (x.type == T_HQ) {
            if (!LG_X_INDEXER || x.nloc < 0 || x.nloc > 3 || seen_q[x.chunk][x.nloc]) return false;
            seen_q[x.chunk][x.nloc] = true;
            { const int d = x.nloc; ++raw_cnt[(LG_X_IQ_RAW + d * 128) / 128]; ++pk_cnt[d][LG_X_IQ_PACKED / 128]; if (d != 0) ++arr[d][g]; }
        } else if (x.type == T_HK) {
            if (!LG_X_INDEXER || seen_k[x.chunk]) return false;
            seen_k[x.chunk] = true;
            ++raw_cnt[LG_X_IK_RAW / 128]; for (int d = 0; d < 4; ++d) ++pk_cnt[d][LG_X_IK_PACKED / 128];
            for (int d = 1; d < 4; ++d) ++arr[d][g];
        } else { return false; }
    }
    const int nseg = LG_X_INDEXER ? 77 : 72, npk = LG_X_INDEXER ? 20 : 18;
    for (int i = 0; i < nseg; ++i) if (raw_cnt[i] != chunks) return false;                  // every raw 128-col segment read exactly once per chunk
    for (int d = 0; d < 4; ++d) for (int p = 0; p < npk; ++p) if (pk_cnt[d][p] != chunks) return false;
    for (int d = 1; d < 4; ++d) for (int g = 0; g < np; ++g) if (arr[d][g] != r7_want(chunks, np, g)) return false;
    return true;
}
static_assert(r7_order_ok(8, 2) && r7_order_ok(8, 4) && r7_order_ok(16, 2) && r7_order_ok(16, 4) && r7_order_ok(32, 2) && r7_order_ok(32, 4), "r7 half-first order: bijection, coverage, per-group arrivals (chunks 8/16/32 x n 2/4)");
static_assert(grid_items_h(8, 2, 1) == (LG_X_INDEXER ? 436 : 396) && grid_items_h(8, 2, 0) == (LG_X_INDEXER ? 328 : 288), "M2048 n2 grid sizes: 436/396 half-first, 328/288 plain");
// ======================= r9: QUARTER-first group 0 (dense, M=2048, n=2; plan bit 0x800, host half_first = 2) ==========================================================================
// A quarter item is 128 rows x 128 cols (one hm half of one hn half of a 256x256 tile): it runs the PROVEN half pipeline with ONE A half (half the MFMAs and A traffic, same B layout), so group 0 completes
// after ~0.6x the half-first time. (A true 256x64 quarter-N needs a new 64-column B layout; not attempted: the B layouts are hard-wired to 128 columns.) Group 0 = the first LG_R9_QC chunk rows of every peer
// as quarter items; group 1 = chunk 2 (full tiles, ready while the group-0 put is still on the wire); group 2 = the remaining chunks (full tiles); self tiles last (full). THREE puts per peer (NP = 3). Dense only.
// Pipe model (inference, tests/pipe_model_r9.py): a 2-group split [2 quarter][6 full] leaves the link idle ~15 us before the second put (group 1 not ready when put 0 finishes) and is +6..+17 us WORSE than n2h;
// the 3-group split [2][1][5] keeps the pipe continuous: -6 us (quarter item = 0.6 tile) .. +5 us (0.8 tile) vs n2h.
constexpr int R9_QC = static_cast<int>(LG_R9_QC), R9_G1 = static_cast<int>(LG_R9_G1);
constexpr int R9_NG = 3;                                           // put groups per peer in the quarter-first plan: [0,QC) quarters, [QC,QC+G1) full, [QC+G1,chunks) full
constexpr int r9_g_first(int g) { return g == 0 ? 0 : (g == 1 ? R9_QC : R9_QC + R9_G1); }
constexpr int r9_g_n(int g, int chunks) { return g == 0 ? R9_QC : (g == 1 ? R9_G1 : chunks - R9_QC - R9_G1); }
constexpr int r9_group_of(int c) { return c < R9_QC ? 0 : (c < R9_QC + R9_G1 ? 1 : 2); }
constexpr int r9_want(int chunks, int g) { return g == 0 ? R9_QC * NPD * 4 : r9_g_n(g, chunks) * NPD; }
constexpr int grid_items_q(int chunks) { return NPEER * NPD * (R9_QC * 4 + (chunks - R9_QC)) + NPD * chunks; }
constexpr XItem select_item_r9(int t, int chunks) {
    const int S0 = R9_QC * NPD * 4, g0 = NPEER * S0, S1 = R9_G1 * NPD, g1 = NPEER * S1, S2 = (chunks - R9_QC - R9_G1) * NPD, Rn = g0 + g1 + NPEER * S2;
    if (t < g0) {                                                  // group 0: slot-major, then (chunk, tile column, quarter); the 4 quarters of a tile are adjacent
        const int slot = t / S0, u = t % S0, q = u % 4, u2 = u / 4;
        return {T_QN, u2 / NPD, slot, u2 % NPD, q};
    }
    if (t < g0 + g1) {                                             // group 1: full tiles (one chunk row), slot-major
        const int v = t - g0, slot = v / S1, u = v % S1;
        return {T_NORMAL, R9_QC + u / NPD, slot, u % NPD, 0};
    }
    if (t < Rn) {                                                  // group 2: the remaining full-tile chunk rows, slot-major
        const int v = t - g0 - g1, slot = v / S2, u = v % S2;
        return {T_NORMAL, R9_QC + R9_G1 + u / NPD, slot, u % NPD, 0};
    }
    const int s = t - Rn;
    return {T_NORMAL, s / NPD, NPEER, s % NPD, 0};
}
constexpr bool r9_order_ok(int chunks) {
    bool seen_t[32][4][9] = {}; bool cov[32][4][9][2][2] = {};
    int arr[4][3] = {};
    for (int t = 0; t < grid_items_q(chunks); ++t) {
        const XItem x = select_item_r9(t, chunks);
        if (x.chunk < 0 || x.chunk >= chunks || x.dslot < 0 || x.dslot > NPEER || x.nloc < 0 || x.nloc >= NPD) return false;
        const int dst = x.dslot < NPEER ? (x.dslot + 1) : 0, g = r9_group_of(x.chunk);
        if (x.type == T_QN) {
            if (g != 0 || x.dslot == NPEER || x.nsel < 0 || x.nsel > 3) return false;                      // quarters only in group 0 and only remote
            const int hn = x.nsel & 1, hm = x.nsel >> 1;
            if (cov[x.chunk][dst][x.nloc][hn][hm]) return false;
            cov[x.chunk][dst][x.nloc][hn][hm] = true;
        } else if (x.type == T_NORMAL) {
            if (g == 0 && x.dslot != NPEER) return false;                                                    // group-0 remote tiles must be quarters
            if (seen_t[x.chunk][dst][x.nloc]) return false;
            seen_t[x.chunk][dst][x.nloc] = true;
            for (int hn = 0; hn < 2; ++hn) for (int hm = 0; hm < 2; ++hm) { if (cov[x.chunk][dst][x.nloc][hn][hm]) return false; cov[x.chunk][dst][x.nloc][hn][hm] = true; }
        } else { return false; }
        if (dst != 0) ++arr[dst][g];
    }
    for (int c = 0; c < chunks; ++c) for (int d = 0; d < 4; ++d) for (int n = 0; n < NPD; ++n) for (int hn = 0; hn < 2; ++hn) for (int hm = 0; hm < 2; ++hm) if (!cov[c][d][n][hn][hm]) return false;   // every 128x128 quarter of every tile once
    for (int d = 1; d < 4; ++d) for (int g = 0; g < R9_NG; ++g) if (arr[d][g] != r9_want(chunks, g)) return false;
    return true;
}
#if !LG_X_INDEXER
static_assert(r9_order_ok(8) && grid_items_q(8) == 450 && r9_want(8, 0) == 72 && r9_want(8, 1) == 9 && r9_want(8, 2) == 45, "r9 quarter-first plan (M2048, 3 groups 2/1/5 chunks): bijection, quarter coverage, per-group arrivals (72 / 9 / 45), 450 items");
#endif
// group helpers: chunk -> put group, group first chunk / chunk count (uneven for HG == 2: 2 / 1 / chunks-3 with NP = 3; even chunks/NP otherwise)
template<int NP, int HG> __device__ constexpr int group_of_chunk(int c, int chunks) { return HG == 2 ? r9_group_of(c) : c / (chunks / NP); }
template<int NP, int HG> __device__ constexpr int group_first(int g, int chunks) { return HG == 2 ? r9_g_first(g) : g * (chunks / NP); }
template<int NP, int HG> __device__ constexpr int group_nchunks(int g, int chunks) { return HG == 2 ? r9_g_n(g, chunks) : chunks / NP; }
template<int NP, int HG> __device__ constexpr int want_of(int chunks, int g) { return HG == 2 ? r9_want(chunks, g) : (HG ? r7_want(chunks, NP, g) : r6x_group_items(chunks, NP)); }
template<int NP, int HG> __device__ constexpr int grid_of(int chunks) { return HG == 2 ? grid_items_q(chunks) : grid_items_h(chunks, NP, HG); }
template<int NP, int HG> __device__ constexpr XItem select_item_h(int t, int chunks) { return HG == 2 ? select_item_r9(t, chunks) : (HG ? select_item_r7(t, chunks, NP) : select_item_r6x(t, chunks, NP)); }
// r8c: EXPLICIT release form (Evidence ISA gate, review 32f95e line 67): a bare ACQ_REL RMW / release store was lowered as buffer_wbl2 -> lgkm wait -> atomic with NO s_waitcnt vmcnt(0) between the
// writeback and the atomic/store. Explicit sequence: system release fence (wbl2) -> asm vmcnt(0) (compiler-opaque, memory clobber) -> RELAXED system atomic/store [-> acquire fence for RMWs that acquire].
__device__ __forceinline__ void rel_fence_vm0() {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "");
    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
}
__device__ __forceinline__ void st_u64_rel_x(uint8_t* p, uint64_t v) {      // release store, explicit form
    rel_fence_vm0();
    __hip_atomic_store(reinterpret_cast<uint64_t*>(p), v, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}
__device__ __forceinline__ uint32_t* gdone_ptr(const LgCommArgsV2& a, int dst, int g) { return reinterpret_cast<uint32_t*>(cw(a, LGC_DONE_OFF + 4u * (dst * LG_MAX_CHUNKS + g))); }

// ---- producer publication: items that wrote to 0..3 REMOTE destinations (d0..d2, first nd valid, WG-uniform; nd == 0: self-only, return).
// ONE per-wave wait + barrier + thread-0 release fence (one WBL2 covers every destination's staging stores), then one RMW(+1) on gdone[dst][g] per distinct remote destination (review r5x (1)(b):
// the release fence precedes EACH RMW => each destination's poster acquire synchronizes with this WG). r6 has NO last-arriver read of `old`, so the RMW needs no acquire part (default RELAXED).
template<int NP, int HG>
__device__ __forceinline__ void x_publish(const LgCommArgsV2& a, int d0, int d1, int d2, int nd, uint32_t chunk) {   // scalar destinations (no runtime-indexed array => no LDS promotion)
    if (nd == 0) return;
#if LG_STAMPS
    const uint64_t tcc0 = stamp_now();
#endif
    opus::s_waitcnt_vmcnt(opus::number<0>{});
    opus::s_waitcnt_lgkmcnt(opus::number<0>{});
    asm volatile("s_waitcnt vmcnt(0) lgkmcnt(0)" ::: "memory");                // compiler-opaque copy (W-1), every wave before the barrier
    mori::cco::impl::syncThreads();
    if (opus::thread_id_x() == 0) {
        const uint32_t g = static_cast<uint32_t>(group_of_chunk<NP, HG>(static_cast<int>(chunk), static_cast<int>(a.chunks)));
        const uint32_t cap = static_cast<uint32_t>(want_of<NP, HG>(static_cast<int>(a.chunks), static_cast<int>(g)));
#if LG_NEGCTRL_NOWB
        // NEGATIVE CONTROL: no release fence (RELAXED RMW): stale S must be detectable
#else
        rel_fence_vm0();                                                       // ONE system release: buffer_wbl2 sc0 sc1 + explicit s_waitcnt vmcnt(0), BEFORE the first RMW (W4-C1; r8c: explicit asm vmcnt(0))
#endif
        auto rmw = [&](int dd) {
#if LG_R6_ARRIVAL_ACQ
            const uint32_t old = __hip_atomic_fetch_add(gdone_ptr(a, dd, static_cast<int>(g)), 1u, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
#else
            const uint32_t old = __hip_atomic_fetch_add(gdone_ptr(a, dd, static_cast<int>(g)), 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
#endif
            if (old + 1u > cap) set_poison(a, LGC_POISON_DUP_CHUNK);           // overcount: never silent
        };
        if (nd > 0) rmw(d0);                                                   // one RMW per distinct remote destination, after the single release fence
        if (nd > 1) rmw(d1);
        if (nd > 2) rmw(d2);
#if LG_STAMPS
        uint64_t* rec = stamp_rec(a);
        rec[1] = tcc0; rec[3] = stamp_now(); rec[4] = rec[3];
#endif
    }
}

// ---- one put of group g of `peer`: rows [g*cpp*256, (g+1)*cpp*256) of S_peer -> R_peer[rank]; queue 0; no signal; byte count derived from a.F ----
template<int NP, int HG>
__device__ __forceinline__ void r6x_put_group(const LgCommArgsV2& a, int peer, uint32_t g) {
    const uint64_t c0 = static_cast<uint64_t>(group_first<NP, HG>(static_cast<int>(g), static_cast<int>(a.chunks)));      // first chunk of group g
    const uint64_t cpp = static_cast<uint64_t>(group_nchunks<NP, HG>(static_cast<int>(g), static_cast<int>(a.chunks)));   // chunks in group g (uneven in the r9 plan)
    const uint64_t slot = 2ull * a.M * a.F;
    const uint64_t off = c0 * 2ull * LG_CHUNK_ROWS * a.F, bytes = cpp * 2ull * LG_CHUNK_ROWS * a.F;
    ccoSdma{a.dev}.template put<ccoCoopThread, false, false, ccoSdmaOptFlagsDefault, ccoSdmaThreadIndependent>(
        peer, a.Rwin, static_cast<size_t>(a.rank) * slot + off, a.Swin, static_cast<size_t>(peer) * slot + off, static_cast<size_t>(bytes), 0u);
#if LG_DEBUG_POSTCOUNT
    __hip_atomic_fetch_add(reinterpret_cast<uint32_t*>(cw(a, LGC_POSTCNT_OFF + 4u * (peer * LG_MAX_CHUNKS + g))), 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);   // expect exactly +1 per (peer, group) per epoch
#endif
}
// ---- r6b waits: RELAXED system-scope spin + ONE acquire fence after success (review P-C3: an ACQUIRE load per spin would issue a buffer_inv per spin for the whole GEMM). Budgets are in
// MICROSECONDS measured with s_memrealtime (100 MHz => 100 ticks/us), not iteration counts (review P-C9/L-1): args.reserved[1] = budget per wait (default 50 ms), must be non-zero.
__device__ __forceinline__ uint64_t r6_budget_ticks(const LgCommArgsV2& a) { return a.reserved[1] * 100ull; }
__device__ __forceinline__ bool r6_poll_u32_ge(const uint32_t* p, uint32_t want, uint64_t budget_ticks) {
    const uint64_t t0 = __builtin_amdgcn_s_memrealtime();
    for (;;) {
        if (__hip_atomic_load(p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM) >= want) { __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, ""); return true; }   // ONE system acquire fence after success
        if (__builtin_amdgcn_s_memrealtime() - t0 >= budget_ticks) return false;
        __builtin_amdgcn_s_sleep(1);
    }
}
__device__ __forceinline__ bool r6_poll_u64_ge(const uint64_t* p, uint64_t want, uint64_t budget_ticks) {
    const uint64_t t0 = __builtin_amdgcn_s_memrealtime();
    for (;;) {
        if (__hip_atomic_load(p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM) >= want) { __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, ""); return true; }
        if (__builtin_amdgcn_s_memrealtime() - t0 >= budget_ticks) return false;
        __builtin_amdgcn_s_sleep(1);
    }
}

// X-1 IMMUNITY (review lg-r5x-review-r1.md (3): "a missing arrival is a silent stale pass" in r5b/r5x quiet_ready). In r6/r6x the ARRIVAL store is the LAST statement of r6_poster_block and is
// reachable ONLY through the loop below: for every group g < NP the poster has (1) passed the bounded wait `r6_poll_u32_ge(gdone[peer][g] >= want)` with want = (chunks/NP)*LG_ITEMS_PER_GROUP (every
// arrival of every item of that group, incl. IK/IQ in the indexer plan), (2) passed the credit wait, (3) issued the put for exactly that group's rows. Any timeout returns through fail() BEFORE the
// arrival store (no arrival is published), and quiet_ready's own W2 then times out on the peer => poison. A missing item arrival therefore can never produce "arrival without complete groups".
// There is no state in which a group is skipped: g runs 0..NP-1 and every iteration either puts or returns poisoned.
#if LG_R8_FOLD
// ---- r8 A2: W2 + reset + epoch advance folded into the poster (replaces the quiet_ready kernel) ----
// Each of the 3 poster waves (one per peer p) after its own quiet + arrival release: (1) bounded wait for p's arrival[p] == e in THIS rank's control window (relaxed spin + one acquire fence),
// (2) fold counter: release fence -> vmcnt(0) -> RELAXED RMW -> acquire fence; the LAST of the 3 (old == 2) resets the device-side state and advances the epoch (only if poison == 0).
// ONLY waves that reach this function fold: a W2 timeout / epoch mismatch still folds (counter reaches 3, reset runs, epoch NOT advanced because poison != 0). **A put-phase failure (r6_poster_block fail(): group wait,
// credit wait, forced seams) returns BEFORE this call: that wave never folds, so the fold counter stays < 3 and gdone/state words are left stale. Such an epoch REQUIRES host recovery before the next launch (review A2-H):
// drain_all on every rank -> barrier -> init_control on every rank (zeroes the whole control window incl. fold counter, gdone, state) -> barrier. LgComm.epoch() refuses to launch while a poison is latched; LgComm.recover() performs the sequence.
__device__ __forceinline__ void r8_fold_wait_reset(const LgCommArgsV2& a, int peer, uint64_t e, uint64_t budget) {
    const uint64_t t0 = __builtin_amdgcn_s_memrealtime();
    bool got = false;
    for (;;) {                                                               // W2: peer's arrival (its puts into MY R are complete) - relaxed spin, ONE acquire fence on success
        const uint64_t v = __hip_atomic_load(reinterpret_cast<const uint64_t*>(cw(a, LGC_ARRIVAL_OFF + 8u * peer)), __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
        if (v == e) { __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, ""); got = true; break; }
        if (v > e) { set_poison(a, LGC_POISON_EPOCH_MISMATCH); break; }
        if (__builtin_amdgcn_s_memrealtime() - t0 >= budget) break;
        __builtin_amdgcn_s_sleep(1);
    }
    if (!got && ld_u32(cw(a, LGC_POISON_OFF)) == 0u) set_poison(a, LGC_POISON_W2_TIMEOUT);
    // fold: the last waiter of the 3 resets + advances. Its acquire fence (below) pairs with the other two waves' release fences (their W2 acquires + all earlier accesses precede their RMW).
    // r8c fold RMW, explicit form (Lead/Evidence): release fence -> asm vmcnt(0) -> RELAXED system fetch_add -> acquire fence. (The earlier ACQ_REL RMW lowered as wbl2 -> lgkm wait -> atomic with no vmcnt(0);
    // the acquire fence after the RMW keeps the acquire half: it pairs with the other two waves' release fences through the counter's release sequence.)
    rel_fence_vm0();
    const uint32_t old = __hip_atomic_fetch_add(reinterpret_cast<uint32_t*>(cw(a, LGC_FOLD_CNT_OFF)), 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "");
    if (old == 2u) {
        for (int i = 0; i < 4 * static_cast<int>(LG_MAX_CHUNKS); ++i) {      // device-side reset of the group counters + debug state words (all 3 posters are past their group waits)
            __hip_atomic_store(reinterpret_cast<uint32_t*>(cw(a, LGC_DONE_OFF + 4u * i)), 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
            __hip_atomic_store(reinterpret_cast<uint32_t*>(cw(a, LGC_STATE_OFF + 4u * i)), 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
        }
#if LG_STAMPS
        __hip_atomic_store(reinterpret_cast<uint64_t*>(cw(a, LGC_FIRSTPUT_OFF)), 0ull, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
#endif
        __hip_atomic_store(reinterpret_cast<uint32_t*>(cw(a, LGC_FOLD_CNT_OFF)), 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);   // reset the counter for the next epoch (before the epoch release below)
        mori::cco::impl::threadFenceSystem();
        if (ld_u32(cw(a, LGC_POISON_OFF)) == 0u) {                           // poisoned epochs are NOT advanced
            add_report(a, 5, 1);
            st_u64_rel_x(cw(a, LGC_EP_OFF), e + 1ull);                       // epoch advance: explicit release (after the reset stores + system fence)
        }
    }
}
#endif

// ---- the poster block (block 0): waves 0..2 active, one issuing lane each; all other threads exit uniformly (no syncThreads in this path) ----
template<int NP, int HG>
__device__ __forceinline__ void r6_poster_block(const LgCommArgsV2& a) {
    const int tid = static_cast<int>(opus::thread_id_x()), wave = tid / 64, lane = tid % 64;
    if (wave >= 3 || lane != 0) return;
    const int rank = static_cast<int>(a.rank), peer = peer_of(rank, wave);
    const uint64_t e = ld_u64(cw(a, LGC_EP_OFF));
    const uint64_t budget = r6_budget_ticks(a);
    uint32_t issued = 0;                                                          // puts already posted to this peer in this epoch
    // F-A (review): on ANY timeout after a put was issued, drain that peer's queue before returning poisoned (the engine may still be copying into the peer's R).
    auto fail = [&](uint32_t reason) {
        set_poison(a, reason);
        if (issued != 0u) ccoSdma{a.dev}.template quietQueue<ccoCoopThread>(peer, 0u);
    };
#if LG_STAMPS
    uint64_t* ps = reinterpret_cast<uint64_t*>(cw(a, LGC_PSTAMP_OFF + 128u * static_cast<uint32_t>(wave)));
    ps[0] = stamp_now(); ps[14] = static_cast<uint64_t>(peer);
#endif
    st_u64_rel_x(cw_peer(a, peer, LGC_CREDIT_OFF + 8u * a.rank), e);             // credit: "this rank began epoch e" -> the peer's poster may PUT into this rank's R
#if LG_STAMPS
    ps[1] = stamp_now();
#endif
    bool credit_ok = false;
    // want(g) = arrivals completing (peer, group g): (chunks/NP) * LG_ITEMS_PER_GROUP [+ (chunks/NP)*9 extra half arrivals in group 0 of the r7 half-first plan]; the put is NEVER issued before this count (X-1 immunity)
    for (uint32_t g = 0; g < static_cast<uint32_t>(NP); ++g) {
        const uint32_t want = static_cast<uint32_t>(want_of<NP, HG>(static_cast<int>(a.chunks), static_cast<int>(g)));
        if (!r6_poll_u32_ge(gdone_ptr(a, peer, g), want, budget)) { fail(LGC_POISON_POSTER_WAIT); return; }   // no further put, no arrival: the peer's W2 times out and poisons too
#if LG_STAMPS
        ps[3 + g] = stamp_now();
#endif
        if (!credit_ok) {                                                         // bounded wait for the PEER's credit (peer began epoch e => its R is free); first group only
            const uint64_t t0 = __builtin_amdgcn_s_memrealtime();
            credit_ok = r6_poll_u64_ge(reinterpret_cast<const uint64_t*>(cw(a, LGC_CREDIT_OFF + 8u * peer)), e, budget);
            add_report(a, 1, __builtin_amdgcn_s_memrealtime() - t0);
            if (!credit_ok) { fail(LGC_POISON_W1_TIMEOUT); return; }
#if LG_STAMPS
            ps[2] = stamp_now();
#endif
        }
#if LG_STAMPS
        if (g == 0) { uint64_t z0 = 0ull; __hip_atomic_compare_exchange_strong(reinterpret_cast<uint64_t*>(cw(a, LGC_FIRSTPUT_OFF)), &z0, __builtin_amdgcn_s_memrealtime(), __ATOMIC_RELAXED, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM); ps[13] = stamp_now(); }
#endif
#if LG_DEBUG_POSTCOUNT
        { uint32_t* st = reinterpret_cast<uint32_t*>(cw(a, LGC_STATE_OFF + 4u * (peer * LG_MAX_CHUNKS + g)));   // debug assertion of exactly-once: EMPTY -> POSTING by CAS
          uint32_t ex = LGC_REC_EMPTY;
          if (!__hip_atomic_compare_exchange_strong(st, &ex, static_cast<uint32_t>(LGC_REC_POSTING), __ATOMIC_ACQ_REL, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM)) set_poison(a, LGC_POISON_DUP_CHUNK); }
#endif
#if LG_X_FORCE_TIMEOUT_G0
        if (g == 0u) { fail(LGC_POISON_POSTER_WAIT); return; }                    // Y4 seam: forced group-0 timeout BEFORE the first put (issued == 0 => fail() does no quietQueue); peers' W2 must time out and poison
#endif
#if LG_X_FORCE_TIMEOUT_G1
        if (g == 1u) { fail(LGC_POISON_POSTER_WAIT); return; }                    // X-A test hook (debug build only): forced g=1 timeout AFTER put(0) was issued
#endif
        r6x_put_group<NP, HG>(a, peer, g);
#if LG_X_NEG_DOUBLE_PUT
        static_assert(LG_DEBUG_POSTCOUNT, "LG_X_NEG_DOUBLE_PUT needs LG_DEBUG_POSTCOUNT=1");
        if (g == 0u && wave == 1) {                                               // X1-neg: duplicate put of (peer slot 1, group 0): post_count -> 2, second EMPTY->POSTING CAS must fail => poison DUP_CHUNK
            r6x_put_group<NP, HG>(a, peer, g);
            uint32_t* st2 = reinterpret_cast<uint32_t*>(cw(a, LGC_STATE_OFF + 4u * (peer * LG_MAX_CHUNKS + g)));
            uint32_t ex2 = LGC_REC_EMPTY;
            if (!__hip_atomic_compare_exchange_strong(st2, &ex2, static_cast<uint32_t>(LGC_REC_POSTING), __ATOMIC_ACQ_REL, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM)) set_poison(a, LGC_POISON_DUP_CHUNK);
        }
#endif
        ++issued;
        add_report(a, 2, 1);                                                      // inline_posts: expect 3*n per epoch
#if LG_STAMPS
        ps[7 + g] = stamp_now();
#endif
    }
    ccoSdma{a.dev}.template quietQueue<ccoCoopThread>(peer, 0u);                 // drain: all n puts of this peer completed at the destination (CCO quiet semantics, claim C6)
#if LG_STAMPS
    ps[11] = stamp_now();
#endif
    st_u64_rel_x(cw_peer(a, peer, LGC_ARRIVAL_OFF + 8u * a.rank), e);            // RELEASE arrival (explicit form: vmcnt(0) also drains quietQueue's rptr loads): "all of rank's epoch-e puts into peer's R are complete"
#if LG_STAMPS
    ps[12] = stamp_now();
#endif
#if LG_R8_FOLD
    r8_fold_wait_reset(a, peer, e, budget);
#endif
}

// ---- quiet_ready for r6: W2 + reset + epoch advance (no posting, no W1). ONE WG, threads 0..2 = peers. ----
__device__ __forceinline__ void r6_quiet_ready(const LgCommArgsV2& a) {
    const int tid = static_cast<int>(opus::thread_id_x());
    const uint64_t e = ld_u64(cw(a, LGC_EP_OFF));
    if (tid < 3) {
        const int d = peer_of(static_cast<int>(a.rank), tid);
        bool got = false;
        const uint64_t t0 = __builtin_amdgcn_s_memrealtime(), budget = r6_budget_ticks(a);
        for (;;) {                                                                  // relaxed spin on OWN arrival[d]; one acquire fence on success; budget in us
            const uint64_t v = __hip_atomic_load(reinterpret_cast<const uint64_t*>(cw(a, LGC_ARRIVAL_OFF + 8u * d)), __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
            if (v == e) { __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, ""); got = true; break; }
            if (v > e) { set_poison(a, LGC_POISON_EPOCH_MISMATCH); break; }
            if (__builtin_amdgcn_s_memrealtime() - t0 >= budget) break;
            __builtin_amdgcn_s_sleep(1);
        }
        if (!got && ld_u32(cw(a, LGC_POISON_OFF)) == 0u) set_poison(a, LGC_POISON_W2_TIMEOUT);
    }
    mori::cco::impl::syncThreads();
    for (int i = tid; i < 4 * static_cast<int>(LG_MAX_CHUNKS); i += static_cast<int>(opus::block_size_x())) {   // device-side reset (no host memset): group counters + debug states
        __hip_atomic_store(reinterpret_cast<uint32_t*>(cw(a, LGC_DONE_OFF + 4u * i)), 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
        __hip_atomic_store(reinterpret_cast<uint32_t*>(cw(a, LGC_STATE_OFF + 4u * i)), 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
    }
#if LG_STAMPS
    if (tid == 0) __hip_atomic_store(reinterpret_cast<uint64_t*>(cw(a, LGC_FIRSTPUT_OFF)), 0ull, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
#endif
    mori::cco::impl::syncThreads();
    if (tid == 0) {
        mori::cco::impl::threadFenceSystem();
        if (ld_u32(cw(a, LGC_POISON_OFF)) == 0u) {                               // poisoned epochs are NOT advanced
            add_report(a, 5, 1);
            st_u64_rel_x(cw(a, LGC_EP_OFF), e + 1ull);
        }
    }
}

// ---- host recovery step (F-A): quietQueue on EVERY peer queue of this rank; run + synchronize before any re-init / next epoch after a poisoned epoch. 1 WG, threads 0..2 ----
__device__ __forceinline__ void r6_drain_all(const LgCommArgsV2& a) {
    const int tid = static_cast<int>(opus::thread_id_x());
    if (tid < 3) {
        const int peer = peer_of(static_cast<int>(a.rank), tid);
        for (uint32_t q = 0; q < a.sdma_queues; ++q) ccoSdma{a.dev}.template quietQueue<ccoCoopThread>(peer, q);
    }
    mori::cco::impl::syncThreads();
}} // namespace lgc
