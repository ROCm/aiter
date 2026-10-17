// SPDX-License-Identifier: MIT
// LG-QSDMA-r2 device-side communication helpers + quiet_ready body (device pass only; UNCOMPILED source).
// Protocol: large-amend-r1.md §1 (single R, credit released at GEMM entry, bounded waits only in quiet_ready).
// CCO API usage mirrors PR695-M4 support.cu l.422-449 (put<ccoCoopThread,false,false,ccoSdmaOptFlagsDefault,
// ccoSdmaThreadIndependent>, quietQueue<ccoCoopThread>) and gcnasm sdma_a2a_kernel.cc (ccoGetLsaPeerPtr).
#pragma once
#include <opus/opus.hpp>
#include "lg_comm_abi_v2.h"

namespace lgc {
using namespace mori::cco;

__device__ __forceinline__ uint8_t* cw(const LgCommArgsV2& a, uint32_t off) {
    return static_cast<uint8_t*>(ccoGetLocalPtr(a.Cwin, off));
}
__device__ __forceinline__ uint8_t* cw_peer(const LgCommArgsV2& a, int peer, uint32_t off) {
    return static_cast<uint8_t*>(ccoGetLsaPeerPtr(a.Cwin, peer, off));
}
__device__ __forceinline__ int peer_of(int rank, int i) { return (rank + 1 + i) & 3; }   // i in 0..2: the 3 remote peers

__device__ __forceinline__ uint64_t ld_u64(const uint8_t* p) {
    return __hip_atomic_load(reinterpret_cast<const uint64_t*>(p), __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
}
__device__ __forceinline__ uint32_t ld_u32(const uint8_t* p) {
    return __hip_atomic_load(reinterpret_cast<const uint32_t*>(p), __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
}
__device__ __forceinline__ void st_u64_rel(uint8_t* p, uint64_t v) {
    __hip_atomic_store(reinterpret_cast<uint64_t*>(p), v, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
}
__device__ __forceinline__ void st_u32_rel(uint8_t* p, uint32_t v) {
    __hip_atomic_store(reinterpret_cast<uint32_t*>(p), v, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
}
__device__ __forceinline__ void add_report(const LgCommArgsV2& a, int idx, uint64_t v) {
    __hip_atomic_fetch_add(reinterpret_cast<uint64_t*>(cw(a, LGC_REPORT_OFF + 8u * idx)), v, __ATOMIC_RELAXED,
                           __HIP_MEMORY_SCOPE_SYSTEM);
}
// First poison reason wins (CAS from 0); sticky.
__device__ __forceinline__ void set_poison(const LgCommArgsV2& a, uint32_t reason) {
    uint32_t expected = 0u;
    __hip_atomic_compare_exchange_strong(reinterpret_cast<uint32_t*>(cw(a, LGC_POISON_OFF)), &expected, reason,
                                         __ATOMIC_RELEASE, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
    __hip_atomic_store(reinterpret_cast<uint64_t*>(cw(a, LGC_REPORT_OFF + 8u * 4)), static_cast<uint64_t>(reason),
                       __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}

#ifdef __HIP_DEVICE_COMPILE__
static_assert(LG_SDMA_RING_BYTES == mori::cco::CCO_SDMA_QUEUE_SIZE && LG_SDMA_UNIT_BYTES == mori::cco::CCO_SDMA_UNIT, "ring constants must match the pinned cco.hpp");
// B-1: a no-signal put = ONE 32 B COPY unit (count field 30 bits, so 2*256*F = 1.18 MB is a single packet; no ccoSdmaDispatchChunked split). Ring = 8192 units.
static_assert(LG_MAX_INLINE_PER_HOLDER * 4u * LG_SDMA_UNIT_BYTES * 2u <= LG_SDMA_RING_BYTES, "inline cap leaves >2x ring headroom even if no packet was consumed");
#endif
// READY->POSTING claim: exactly one claimant wins (CAS). LG_NEGCTRL_NOCAS=1 replaces it by load+store (NEGATIVE CONTROL: duplicates must appear).
__device__ __forceinline__ bool claim_chunk(uint32_t* st) {
#if LG_NEGCTRL_NOCAS
    if (__hip_atomic_load(st, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM) == LGC_REC_READY) {
        __hip_atomic_store(st, static_cast<uint32_t>(LGC_REC_POSTING), __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
        return true;
    }
    return false;
#else
    uint32_t exp = LGC_REC_READY;
    return __hip_atomic_compare_exchange_strong(st, &exp, static_cast<uint32_t>(LGC_REC_POSTING), __ATOMIC_ACQ_REL,
                                                __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
#endif
}
// ONE no-signal SDMA put of chunk c: S_d[c] -> R_d[rank][c] on queue c % Q, exactly 2*256*F bytes (window-relative offsets).
__device__ __forceinline__ void put_chunk(const LgCommArgsV2& a, int d, uint32_t c) {
#if LG_STAMPS
    { uint64_t z0 = 0ull; __hip_atomic_compare_exchange_strong(reinterpret_cast<uint64_t*>(cw(a, LGC_FIRSTPUT_OFF)), &z0, __builtin_amdgcn_s_memrealtime(),
                                                              __ATOMIC_RELAXED, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM); }   // r5 stamp: first put of the epoch
#endif
#if LG_DEBUG_POSTCOUNT
    __hip_atomic_fetch_add(reinterpret_cast<uint32_t*>(cw(a, LGC_POSTCNT_OFF + 4u * (d * LG_MAX_CHUNKS + c))), 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);   // debug: per-(dst,chunk) post count (expected +1 per epoch)
#endif
    const uint64_t slot = 2ull * a.M * a.F;                       // bytes of one [M,F] BF16 slab
    const uint64_t off = static_cast<uint64_t>(c) * 2ull * LG_CHUNK_ROWS * a.F;
    ccoSdma{a.dev}.template put<ccoCoopThread, false, false, ccoSdmaOptFlagsDefault, ccoSdmaThreadIndependent>(
        d, a.Rwin, static_cast<size_t>(a.rank) * slot + off,
        a.Swin, static_cast<size_t>(d) * slot + off,
        static_cast<size_t>(2ull * LG_CHUNK_ROWS * a.F), c % a.sdma_queues);   // r4: queue = chunk % Q
}

// r5 stamps (LG_STAMPS=1). Records are written by thread 0 only; XCC = blockIdx % 8 (measured round-robin) plus the raw HW_REG_XCC_ID (encoding unverified).
#if LG_STAMPS
__device__ __forceinline__ uint64_t* stamp_rec(const LgCommArgsV2& a) { return reinterpret_cast<uint64_t*>(cw(a, LGC_STAMP_OFF + 64u * static_cast<uint32_t>(opus::block_id_x()))); }
__device__ __forceinline__ uint64_t stamp_now() { return __builtin_amdgcn_s_memrealtime(); }
__device__ __forceinline__ void stamp_entry(const LgCommArgsV2& a) { if (opus::thread_id_x() == 0) stamp_rec(a)[0] = stamp_now(); }
__device__ __forceinline__ void stamp_self_end(const LgCommArgsV2& a, int chunk) {   // self tiles: [1] = begin proxy (entry), [4] = end, [5] info with dst = self marker
    if (opus::thread_id_x() == 0) { uint64_t* r = stamp_rec(a); r[4] = stamp_now(); r[5] = (static_cast<uint64_t>(opus::block_id_x()) & 7ull) | (static_cast<uint64_t>(chunk & 63) << 6) | (static_cast<uint64_t>(opus::block_id_x()) << 32) | (1ull << 13); }
}
#else
__device__ __forceinline__ void stamp_entry(const LgCommArgsV2&) {}
__device__ __forceinline__ void stamp_self_end(const LgCommArgsV2&, int) {}
#endif

// ---- GEMM entry: release credit for THIS epoch to the 3 peers; NO wait. Block 0, threads 0..2 only. ----
// (Correctness never assumes block 0 runs first: all remote waits live in quiet_ready, stream-after the whole GEMM.)
__device__ __forceinline__ uint64_t gemm_entry_epoch_and_credit(const LgCommArgsV2& a) {
    const uint64_t e = ld_u64(cw(a, LGC_EP_OFF));
    if (opus::block_id_x() == 0 && opus::thread_id_x() < 3) {
        const int d = peer_of(static_cast<int>(a.rank), static_cast<int>(opus::thread_id_x()));
        st_u64_rel(cw_peer(a, d, LGC_CREDIT_OFF + 8u * a.rank), e);   // credit_d[rank] = e ("rank began epoch e")
    }
    return e;
}

// ---- work-item order (r4): REMOTE items first, SELF tiles last; remote tiles optionally split into two half-N (256x128) items ----
// LG_REMOTE_HALF=1: 432 remote half items at M=2048 (3 dst x 8 chunks x 18 items per (dst,chunk)); the first wave (256 CUs) then covers
// 14 complete (dst,chunk) groups after ~T/2 (instead of 0 after T), so SDMA starts early. LG_REMOTE_HALF=0: r3 order (full remote tiles).
// Items are ordered group-major: g = chunk*LG_NUM_PEERS + slot, LG_TILES_N_PER_DEST tiles per group (u = g*NPD + nloc); NPD = F/256 (9 dense, 10 indexer). With half items, blocks t and t+LG_NUM_XCD are the two
// halves of the same tile (same A rows) so that, under round-robin dispatch over XCDs, both land on the same XCD/L2 (inference; correctness
// never depends on it). Self tiles stay full 256x256 and are placed last (written straight into R[rank], no PUT).
struct ItemSel { int chunk; int dslot; int nloc; int nsel; bool half; };
constexpr int dest_of_slot(int rank, int slot) { return (rank + 1 + slot) & 3; }   // slots 0..2 = 3 peers, slot 3 = self
constexpr int items_per_group() { return static_cast<int>(LG_ITEMS_PER_GROUP); }
constexpr int NPD = static_cast<int>(LG_TILES_N_PER_DEST);   // r4b: F-derived tiles per destination (was literal 9)
constexpr int NPEER = static_cast<int>(LG_NUM_PEERS);          // r4b: remote destinations (was literal 3)
constexpr int remote_tiles(int chunks) { return NPEER * NPD * chunks; }
constexpr int remote_items(int chunks) { return remote_tiles(chunks) * (LG_REMOTE_HALF ? 2 : 1); }
constexpr int grid_items(int chunks) { return remote_items(chunks) + NPD * chunks; }
constexpr ItemSel select_item(int t, int chunks) {
    const int Rn = remote_items(chunks);
    if (t < Rn) {
        int u = t, nsel = 0;
        if (LG_REMOTE_HALF) {
            if (remote_tiles(chunks) % LG_NUM_XCD == 0) { u = (t / (2 * LG_NUM_XCD)) * LG_NUM_XCD + (t % LG_NUM_XCD); nsel = (t / LG_NUM_XCD) & 1; }
            else { u = t >> 1; nsel = t & 1; }
        }
        const int g = u / NPD;
        return {g / NPEER, g % NPEER, u % NPD, nsel, LG_REMOTE_HALF != 0};
    }
    const int s = t - Rn;
    return {s / NPD, NPEER, s % NPD, 0, false};   // self slot = NPEER
}
constexpr bool item_order_is_bijection(int chunks) {
    bool seen[32 * 4 * 10 * 2] = {};   // sized for NPD <= 10 (indexer)
    int count = 0;
    for (int t = 0; t < grid_items(chunks); ++t) {
        const ItemSel s = select_item(t, chunks);
        if (s.chunk < 0 || s.chunk >= chunks || s.dslot < 0 || s.dslot > NPEER || s.nloc < 0 || s.nloc >= NPD || s.nsel < 0 || s.nsel > 1) return false;
        if (s.dslot == NPEER && (s.half || s.nsel != 0)) return false;
        if (s.dslot < NPEER && !s.half && s.nsel != 0) return false;
        const int k = ((s.chunk * (NPEER + 1) + s.dslot) * 10 + s.nloc) * 2 + s.nsel;
        if (seen[k]) return false;
        seen[k] = true; ++count;
    }
    return count == grid_items(chunks);
}
static_assert(item_order_is_bijection(1) && item_order_is_bijection(2) && item_order_is_bijection(4) && item_order_is_bijection(8) &&
              item_order_is_bijection(16) && item_order_is_bijection(32), "item order bijection");
static_assert(select_item(0, 8).dslot == 0 && select_item(remote_items(8) - 1, 8).dslot == 2 && select_item(remote_items(8), 8).dslot == 3, "remote first, self last");
static_assert(dest_of_slot(0, 3) == 0 && dest_of_slot(1, 3) == 1 && dest_of_slot(2, 3) == 2 && dest_of_slot(3, 3) == 3, "slot 3 == self");
#if LG_NEGCTRL_NOWAIT
static_assert(LG_L2A_BYPASS, "N-C4 NOWAIT control is defined for the L2a bypass path (LG_L2A_BYPASS=1)");
#endif
static_assert(NPD <= 10 && (NPD + (LG_X_INDEXER ? 1 : 0)) * static_cast<int>(LG_TILE_N) == static_cast<int>(LG_COMM_F), "r6x: NPD normal tiles (+ one IQ|IK tile column when indexer) == F");
static_assert(!(LG_COMM_F == 2304u && LG_REMOTE_HALF) || (grid_items(8) == 504 && remote_items(8) == 432), "dense M=2048 half plan: 432 remote half items + 72 self tiles");

// ---- L1c (r5, LG_L1C_SWIZZLE=1): XCC-blocked W n-band order -------------------------------------------------------------------------
// Round-robin dispatch puts block b on XCC b % 8 (measured). m-fast (r4) makes every XCC read ALL W bands (8x W fabric traffic). Here XCC x owns the
// contiguous bands {x*K .. x*K+K-1} of each phase (K = NB/8), for ALL chunks m, plus a balanced TAIL: the T = NB%8 leftover bands are distributed by the
// flat tail index lambda = m*T + k -> XCC lambda%8. Slots on one XCC are ordered m-major (all of row m before row m+1), so every XCC finishes chunk m at about
// the same time and the (dst,chunk) PUT groups still become ready chunk-by-chunk (remote phase first, self phase last). wgid = 8*j + x; with half items the
// two halves of a logical tile are j and j+1 (wgids differ by 8 => same XCC). Supported when chunks % 8 == 0 (so Mt*T % 8 == 0: exact balance);
// otherwise the r4 order is used at runtime. Remote phase: NB = NPEER*NPD (27 dense / 30 indexer); self phase: NB = NPD (9 / 10).
constexpr int NXCC = static_cast<int>(LG_NUM_XCD);
constexpr int tail_mod(int x, int m, int T) { return (((x - m * T) % NXCC) + NXCC) % NXCC; }          // (x - m*T) mod 8, in [0,8)
constexpr int phase_row_tail(int NB, int m, int x) { const int T = NB % NXCC; return (T != 0 && tail_mod(x, m, T) < T) ? 1 : 0; }
constexpr int phase_count(int NB, int Mt, int x) { (void)x; return (NB / NXCC) * Mt + (((NB % NXCC) * Mt) / NXCC); }   // exact when Mt % 8 == 0 for every XCC
struct PhaseSel { int m; int band; };
constexpr PhaseSel phase_decode(int NB, int Mt, int x, int i) {
    const int K = NB / NXCC, T = NB % NXCC;
    int rem = i;
    for (int m = 0; m < Mt; ++m) {
        const int rows = K + phase_row_tail(NB, m, x);
        if (rem < rows) return rem < K ? PhaseSel{m, x * K + rem} : PhaseSel{m, NXCC * K + tail_mod(x, m, T)};
        rem -= rows;
    }
    return PhaseSel{-1, -1};
}
constexpr ItemSel select_item_l1c(int t, int chunks) {
    const int x = t % NXCC, j = t / NXCC;
    const int mult = LG_REMOTE_HALF ? 2 : 1;
    const int NBR = NPEER * NPD;
    const int cntR = phase_count(NBR, chunks, x);
    if (j < mult * cntR) {
        const int ell = j / mult, nsel = LG_REMOTE_HALF ? (j % mult) : 0;
        const PhaseSel p = phase_decode(NBR, chunks, x, ell);
        return {p.m, p.band / NPD, p.band % NPD, nsel, LG_REMOTE_HALF != 0};
    }
    const PhaseSel p = phase_decode(NPD, chunks, x, j - mult * cntR);
    return {p.m, NPEER, p.band, 0, false};
}
constexpr bool l1c_order_is_bijection(int chunks) {      // bijection onto (chunk, slot, nloc, nsel) + exact per-XCC balance
    if (chunks % NXCC != 0) return false;
    bool seen[32 * 4 * 10 * 2] = {};
    int perx[NXCC] = {};
    for (int t = 0; t < grid_items(chunks); ++t) {
        const ItemSel s = select_item_l1c(t, chunks);
        if (s.chunk < 0 || s.chunk >= chunks || s.dslot < 0 || s.dslot > NPEER || s.nloc < 0 || s.nloc >= NPD || s.nsel < 0 || s.nsel > 1) return false;
        if (s.dslot == NPEER && (s.half || s.nsel != 0)) return false;
        if (s.dslot < NPEER && !s.half && s.nsel != 0) return false;
        const int k = ((s.chunk * (NPEER + 1) + s.dslot) * 10 + s.nloc) * 2 + s.nsel;
        if (seen[k]) return false;
        seen[k] = true; perx[t % NXCC] += 1;
    }
    for (int x = 1; x < NXCC; ++x) if (perx[x] != perx[0]) return false;
    return true;
}
static_assert(l1c_order_is_bijection(8) && l1c_order_is_bijection(16) && l1c_order_is_bijection(32), "L1c swizzle: bijection + per-XCC balance for chunks 8/16/32");
static_assert(select_item_l1c(1, 8).chunk == 0 && select_item_l1c(1, 8).dslot == 0 && select_item_l1c(1, 8).nloc == 3, "L1c: XCC 1 owns W bands 3,4,5 first (m=0)");
static_assert(select_item_l1c(grid_items(8) - 1, 8).dslot == NPEER, "L1c: self tiles last");
__device__ __forceinline__ ItemSel select_item_rt(int t, int chunks) {     // runtime dispatch (chunks is a kernel argument)
#if LG_L1C_SWIZZLE
    if (chunks % NXCC == 0) return select_item_l1c(t, chunks);
#endif
    return select_item(t, chunks);
}
// ---- GEMM chunk completion with LOCK COMBINING per (dst, queue) (r3 + r4) ----
// Record states: EMPTY(0) -> READY(1) -> POSTING(2) -> SUBMITTED(3); READY->POSTING is a CAS => each chunk is put EXACTLY ONCE.
// r4: chunk c uses queue c % Q; the local lock and the combining scan are per (dst, queue) so queues drain independently.
// Correctness never depends on combining: quiet_ready posts every chunk still READY. No GEMM WG waits on another GPU; a failed lock
// CAS never spins; the lock holder waits at most on its own SDMA queue space.
__device__ __forceinline__ void post_ready_row(const LgCommArgsV2& a, int dst, uint32_t qid) {
    uint32_t posted = 0;
    for (uint32_t c = qid; c < a.chunks; c += a.sdma_queues) {
        uint32_t* st = reinterpret_cast<uint32_t*>(cw(a, LGC_STATE_OFF + 4u * (dst * LG_MAX_CHUNKS + c)));
        if (posted >= LG_MAX_INLINE_PER_HOLDER) break;           // B-1 cap: leave the rest READY (next round / quiet_ready); never risk blocking on ring space
        if (claim_chunk(st)) {
            put_chunk(a, dst, c);
            st_u32_rel(reinterpret_cast<uint8_t*>(st), LGC_REC_SUBMITTED);
            add_report(a, 2, 1);                                  // inline_posts (all puts issued during the GEMM, incl. combined)
            ++posted;
        }
    }
}
__device__ __forceinline__ bool any_ready_row(const LgCommArgsV2& a, int dst, uint32_t qid) {
    for (uint32_t c = qid; c < a.chunks; c += a.sdma_queues)
        if (ld_u32(cw(a, LGC_STATE_OFF + 4u * (dst * LG_MAX_CHUNKS + c))) == LGC_REC_READY) return true;
    return false;
}

// Called by EVERY thread after its item's stores; WG-uniform until the single-thread tail. dst != rank only.
__device__ __forceinline__ void gemm_chunk_complete(const LgCommArgsV2& a, uint64_t e, int dst, uint32_t chunk) {
    if (dst == static_cast<int>(a.rank)) return;                 // WG-uniform: self slab went straight to R, no PUT, no counter
#if LG_STAMPS
    const uint64_t tcc0 = stamp_now();                           // chunk-complete begin (all threads read; thread 0 stores)
#endif
#if !LG_NEGCTRL_NOWAIT
    opus::s_waitcnt_vmcnt(opus::number<0>{});                    // own S stores completed (NEVER make this conditional on production flags; N-C4 control removes it on purpose)
    opus::s_waitcnt_lgkmcnt(opus::number<0>{});
#endif
#if LG_NEGCTRL_NOWAIT || LG_NEGCTRL_NOWB
    // NEGATIVE CONTROL: no writeback/release at all (per-wave fence removed, RMW below is RELAXED). Stale S must be detectable.
#elif LG_L2B_ONE_RELEASE
    // r5 L2b (needs reviewer W-4): NO per-wave threadFenceSystem. Each wave drains its own stores (vmcnt(0) above + the asm wait below); the WG barrier
    // orders them before thread 0's SYSTEM ACQ_REL fetch_add, whose release writes back this XCD's dirty L2 lines ONCE per WG (the L2a bypass stores need no
    // writeback at all). Per-WG release is kept: this is NOT a last-arriver-only release.
#else
    mori::cco::impl::threadFenceSystem();                        // SDMA engine reads S: system-scope release (buffer_wbl2 sc0 sc1) - every wave
#endif
    // W-1 (review lg-r3-cas-review-r1): explicit wait the compiler cannot move across (volatile asm + memory clobber), in EVERY thread BEFORE the barrier.
#if !LG_NEGCTRL_NOWAIT
    asm volatile("s_waitcnt vmcnt(0) lgkmcnt(0)" ::: "memory");  // L2a-W: the ONLY ordering of the asm bypass stores; unconditional in every production build
#endif
    mori::cco::impl::syncThreads();                              // all producers of this WG drained
#if LG_STAMPS
    const uint64_t tcc1 = stamp_now();
#endif
    if (opus::thread_id_x() == 0) {
        uint32_t* done = reinterpret_cast<uint32_t*>(cw(a, LGC_DONE_OFF + 4u * (dst * LG_MAX_CHUNKS + chunk)));
#if LG_NEGCTRL_NOWAIT || LG_NEGCTRL_NOWB
        const uint32_t old = __hip_atomic_fetch_add(done, 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
#elif LG_L2B_ONE_RELEASE
        // W4-C1: explicit thread-0 SYSTEM release fence FIRST (lowers to buffer_wbl2 sc0 sc1 + s_waitcnt vmcnt(0) in the RAN ISA, so the writeback is COMPLETE
        // before the atomic), then the RMW with ACQUIRE only (the release is the fence: fence-release + atomic RMW synchronizes with the last arriver's acquire).
        // This keeps exactly ONE release (one WBL2) per WG; an ACQ_REL RMW would emit a second WBL2. Evidence ISA gate W4-C1/W4-C2 checks wbl2 -> wait -> atomic.
        __builtin_amdgcn_fence(__ATOMIC_RELEASE, "");
        const uint32_t old = __hip_atomic_fetch_add(done, 1u, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
#else
        const uint32_t old = __hip_atomic_fetch_add(done, 1u, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_SYSTEM);
#endif
#if LG_STAMPS
        const uint64_t tcc2 = stamp_now();
        const uint32_t xcc = static_cast<uint32_t>(opus::block_id_x()) & 7u;
        const uint32_t qpos = __hip_atomic_fetch_add(reinterpret_cast<uint32_t*>(cw(a, LGC_XCCQ_OFF + 4u * xcc)), 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
        uint64_t* rec = stamp_rec(a);
        rec[1] = tcc0; rec[2] = tcc1; rec[3] = tcc2;
        rec[6] = static_cast<uint64_t>(__builtin_amdgcn_s_getreg(20 | (3 << 11)));          // raw XCC_ID (encoding UNVERIFIED on this toolchain)
        rec[5] = xcc | (static_cast<uint64_t>(dst) << 4) | (static_cast<uint64_t>(chunk & 63) << 6) | (static_cast<uint64_t>(old + 1u == LG_ITEMS_PER_GROUP) << 12) |
                 (static_cast<uint64_t>(qpos & 0xFFFF) << 16) | (static_cast<uint64_t>(opus::block_id_x()) << 32);
#endif
        if (old + 1u == LG_ITEMS_PER_GROUP) {                    // unique last arriver of (dst, chunk): 9 full or 18 half items (traits-derived)
#if LG_NEGCTRL_ISO
            // N-C3 NOWB-ISO control: inline posting OFF and NO last-arriver / holder fences, so no incidental WBL2 runs between the producers' stores and the put;
            // every put is issued by quiet_ready (its fences run on quiet_ready's own XCD only). Debug delay knob: LG_ISO_DELAY_ITERS (s_sleep loops in quiet_ready before posting).
            st_u32_rel(cw(a, LGC_STATE_OFF + 4u * (dst * LG_MAX_CHUNKS + chunk)), LGC_REC_READY);
            if (true) {                                          // ISO: always take the 'leave READY' branch (deferred/credit_miss counters are meaningless in this build)
#else
            st_u32_rel(cw(a, LGC_STATE_OFF + 4u * (dst * LG_MAX_CHUNKS + chunk)), LGC_REC_READY);
            mori::cco::impl::threadFenceSystem();                // seq_cst: READY visible before the lock attempt (W4-C4: Dekker fence, keep)
            if (ld_u64(cw(a, LGC_CREDIT_OFF + 8u * dst)) < e) {
#endif  // peer has not begun epoch e: R_dst not free yet
                add_report(a, 0, 1); add_report(a, 6, 1);        // deferred += 1 ; credit_miss += 1 ; chunk stays READY
            } else {
                const uint32_t qid = chunk % a.sdma_queues;
                uint32_t* lock = reinterpret_cast<uint32_t*>(cw(a, LGC_LOCK_OFF + 4u * (static_cast<uint32_t>(dst) * 4u + qid)));
                for (int round = 0; round < 3; ++round) {        // bounded: <= 3 lock attempts, never an unbounded spin
                    uint32_t exp0 = 0u;
                    if (!__hip_atomic_compare_exchange_strong(lock, &exp0, 1u, __ATOMIC_ACQUIRE, __ATOMIC_RELAXED,
                                                              __HIP_MEMORY_SCOPE_SYSTEM)) {
                        if (round == 0) { add_report(a, 0, 1); add_report(a, 7, 1); }   // lock_miss: holder / quiet_ready will post us
                        break;
                    }
                    mori::cco::impl::threadFenceSystem();
                    post_ready_row(a, dst, qid);                 // post EVERY READY chunk of this (dst, queue)
                    __hip_atomic_store(lock, 0u, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
                    mori::cco::impl::threadFenceSystem();        // seq_cst between unlock and rescan (lost-wakeup guard)
                    if (!any_ready_row(a, dst, qid)) break;
                }
#if LG_STAMPS
                stamp_rec(a)[7] = stamp_now();                   // time after this last-arriver finished posting (combining) its row
#endif
            }
        } else if (old + 1u > LG_ITEMS_PER_GROUP) {
            set_poison(a, LGC_POISON_DUP_CHUNK);
        }
#if LG_STAMPS
        stamp_rec(a)[4] = stamp_now();                           // end of gemm_chunk_complete (thread 0)
#endif
    }
}
// ---- quiet_ready(e): ONE WG (>= 64 threads). Threads 0..2 = remote peers d. All remote waits are bounded (poison). ----
__device__ __forceinline__ void quiet_ready_body(const LgCommArgsV2& a) {
    const int tid = static_cast<int>(opus::thread_id_x());
    const int rank = static_cast<int>(a.rank);
    const uint64_t e = ld_u64(cw(a, LGC_EP_OFF));
#if LG_STAMPS
    uint64_t* qs = reinterpret_cast<uint64_t*>(cw(a, LGC_QSTAMP_OFF + 64u * (tid < 3 ? tid : 0)));
    if (tid < 3) { qs[0] = __builtin_amdgcn_s_memrealtime(); qs[7] = static_cast<uint64_t>(peer_of(rank, tid)); }
    if (tid == 0) reinterpret_cast<uint64_t*>(cw(a, LGC_QSTAMP_OFF + 192u))[0] = static_cast<uint64_t>(__builtin_amdgcn_s_getreg(20 | (3 << 11)));   // N-C3: raw XCC_ID of the quiet_ready WG (encoding UNVERIFIED; blockIdx 0 => expect XCC 0): compare with producers' XCC
#endif
    if (tid < 3) {
        const int d = peer_of(rank, tid);
        // (a) W1: bounded acquire-poll credit_me[d] >= e (d has begun epoch e => R_d is free), then post READY leftovers.
        const uint64_t t0 = __builtin_amdgcn_s_memrealtime();
        bool credit_ok = false;
        for (uint64_t i = 0; i < a.poll_iterations; ++i) {
            if (ld_u64(cw(a, LGC_CREDIT_OFF + 8u * d)) >= e) { credit_ok = true; break; }
            __builtin_amdgcn_s_sleep(1);
        }
        add_report(a, 1, __builtin_amdgcn_s_memrealtime() - t0);
        if (!credit_ok) set_poison(a, LGC_POISON_W1_TIMEOUT);
#if LG_STAMPS
        qs[1] = __builtin_amdgcn_s_memrealtime();
#endif
#if LG_NEGCTRL_ISO && LG_ISO_DELAY_ITERS
        for (uint32_t i = 0; i < LG_ISO_DELAY_ITERS; ++i) __builtin_amdgcn_s_sleep(100);       // debug delay before the posts (N-C3); sweep {0, small}
#endif
        if (credit_ok) {
            for (uint32_t c = 0; c < a.chunks; ++c) {            // GEMM is complete (stream order): no lock contention
                uint8_t* st = cw(a, LGC_STATE_OFF + 4u * (d * LG_MAX_CHUNKS + c));
                if (claim_chunk(reinterpret_cast<uint32_t*>(st))) {   // exactly-once (CAS; negative-control build: load+store)
                    put_chunk(a, d, c);
                    st_u32_rel(st, LGC_REC_SUBMITTED);
                    add_report(a, 3, 1);                          // quiet_posts (leftovers)
                }
            }
        }
#if LG_STAMPS
        qs[2] = __builtin_amdgcn_s_memrealtime();
#endif
        // (b) drain every issued put on this peer's queue (unconditional).
        for (uint32_t qq = 0; qq < a.sdma_queues; ++qq) ccoSdma{a.dev}.template quietQueue<ccoCoopThread>(d, qq);   // r4: drain ALL queues
#if LG_STAMPS
        qs[3] = __builtin_amdgcn_s_memrealtime();
#endif
        // (c) publish arrival only if all of this source's data for d was really posted+drained.
        if (credit_ok) st_u64_rel(cw_peer(a, d, LGC_ARRIVAL_OFF + 8u * rank), e);
#if LG_STAMPS
        qs[4] = __builtin_amdgcn_s_memrealtime();
#endif
        // (d) W2: bounded acquire-poll own arrival[d] == e (> e or other mismatch => poison).
        bool got = false;
        for (uint64_t i = 0; i < a.poll_iterations; ++i) {
            const uint64_t v = ld_u64(cw(a, LGC_ARRIVAL_OFF + 8u * d));
            if (v == e) { got = true; break; }
            if (v > e) { set_poison(a, LGC_POISON_EPOCH_MISMATCH); break; }
            __builtin_amdgcn_s_sleep(1);
        }
        if (!got && ld_u32(cw(a, LGC_POISON_OFF)) == 0u) set_poison(a, LGC_POISON_W2_TIMEOUT);
#if LG_STAMPS
        qs[5] = __builtin_amdgcn_s_memrealtime();
        qs[6] = __hip_atomic_load(reinterpret_cast<uint64_t*>(cw(a, LGC_FIRSTPUT_OFF)), __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);   // first put of this epoch (same value in all 3 records)
#endif
    }
    mori::cco::impl::syncThreads();                              // all arrivals acquired by their polling thread
    // (e) device-side reset for the next replay (no host memset): chunk_done, state, locks. credit/arrival stay monotonic.
    for (int i = tid; i < 4 * static_cast<int>(LG_MAX_CHUNKS); i += static_cast<int>(opus::block_size_x())) {
        __hip_atomic_store(reinterpret_cast<uint32_t*>(cw(a, LGC_DONE_OFF + 4u * i)), 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
        __hip_atomic_store(reinterpret_cast<uint32_t*>(cw(a, LGC_STATE_OFF + 4u * i)), 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
    }
    if (tid < 16)                                                // r4: lock per (dst, queue) = 4*4 words
        __hip_atomic_store(reinterpret_cast<uint32_t*>(cw(a, LGC_LOCK_OFF + 4u * tid)), 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
#if LG_STAMPS
    if (tid < 8) __hip_atomic_store(reinterpret_cast<uint32_t*>(cw(a, LGC_XCCQ_OFF + 4u * tid)), 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);   // per-XCC queue-position counters
    if (tid == 0) __hip_atomic_store(reinterpret_cast<uint64_t*>(cw(a, LGC_FIRSTPUT_OFF)), 0ull, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);       // first-put stamp (copied into qs[6] above)
#endif
    mori::cco::impl::syncThreads();
    if (tid == 0) {
        mori::cco::impl::threadFenceSystem();
        if (ld_u32(cw(a, LGC_POISON_OFF)) == 0u) {               // poisoned epochs are NOT advanced (sticky, never a silent pass)
            add_report(a, 5, 1);
            st_u64_rel(cw(a, LGC_EP_OFF), e + 1ull);
        }
    }
}
} // namespace lgc
