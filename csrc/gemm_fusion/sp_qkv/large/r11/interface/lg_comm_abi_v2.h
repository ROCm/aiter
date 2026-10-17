// SPDX-License-Identifier: MIT
// LG-QSDMA-r2 communication-slice ABI v2 (new, versioned; no reuse of the 336B SP4C struct or the r1 120B struct).
// Linux LP64. Passed BY VALUE to both kernels. Uses the ACTUAL pinned CCO native types (mori/cco/cco.hpp).
// Windows (all MORI-managed, size-only ccoWindowRegister overload, registered BEFORE ccoDevCommCreate):
//   Swin : staging S_d = [4][M][F] BF16 (slab d at d*2MF bytes; slab `rank` unused: self goes straight to R)
//   Rwin : single receive buffer R = [4 sources][M][F] BF16 (== token-major [4M, F], row = s*M + m); registered once
//   Cwin : control window (offsets below); epoch/credit/arrival words are monotonic u64 and never reset
#pragma once
#include <stdint.h>
#include <stddef.h>
#include <type_traits>
#include <mori/cco/cco.hpp>

#define LG_COMM_ABI_VERSION 3u          /* unchanged in r5: all r5 options are compile-time flags; struct/layout of LgCommArgsV2 identical to r4b */
#define LG_CHUNK_ROWS 256u            // one chunk = one 256-row M tile
#define LG_MAX_CHUNKS 32u             // M <= 8192
#define LG_TILE_N 256u                // GEMM tile width (B_N)
#ifndef LG_X_INDEXER
#define LG_X_INDEXER 1                /* r6x: 1 = qkv_indexer/reuse slice (N=9856, F=2560, 39 items/chunk); 0 = dense (N=9216, F=2304, 36 items/chunk; same order as r6b). HOST AND DEVICE MUST AGREE (plan code bit 8). */
#endif
#define LG_X_N_RAW (LG_X_INDEXER ? 9856u : 9216u)
#define LG_X_IQ_RAW 9216u             /* IQ raw cols 9216..9727: IQ_d = 9216 + d*128 */
#define LG_X_IK_RAW 9728u             /* IK raw cols 9728..9855: ONE segment replicated to all 4 destinations */
#define LG_X_IQ_PACKED 2304u
#define LG_X_IK_PACKED 2432u
#ifndef LG_COMM_F
#define LG_COMM_F (LG_X_INDEXER ? 2560u : 2304u)               // slab columns per destination (dense qkv: 2048 Q + 128 K + 128 V; indexer slice: 2560 => 10 tiles)
#endif
#define LG_TILES_N_PER_DEST ((LG_COMM_F - (LG_X_INDEXER ? 256u : 0u)) / LG_TILE_N)   /* NORMAL Q|K|V tiles per destination = 9 (the 10th tile column of F=2560 = IQ pair-tile + IK half item) */   // r4b: F-derived (9 dense, 10 indexer) == full-tile arrivals per (dst,chunk)
#define LG_NUM_PEERS 3u               // remote destinations per rank (4 ranks)
static_assert(LG_COMM_F % LG_TILE_N == 0, "slab columns must be a whole number of 256-wide tiles");
#define LG_Q_ID 0u                    // SDMA queue 0
#ifndef LG_REMOTE_HALF                   /* r4: 1 = remote tiles run as two 256x128 half-N items (default); 0 = r3 full remote tiles. Host and device MUST use the same value. */
#define LG_REMOTE_HALF 0              /* r6 default: full 256x256 remote tiles (V0 was best in r4b); 1 = half-N items (group ordering supports it) */
#endif
#define LG_ITEMS_PER_GROUP (LG_TILES_N_PER_DEST + (LG_X_INDEXER ? 2u : 0u))   /* arrivals per (dst, chunk): 9 normal (+ 1 IQ pair-tile + 1 IK = 11 for indexer) */
#define LG_X_ITEMS_PER_CHUNK (4u * LG_TILES_N_PER_DEST + (LG_X_INDEXER ? 5u : 0u))   /* work items per 256-row chunk: 36 (dense) / 41 (36 normal + 4 IQ_d full tiles + 1 IK full tile; r6x4: no IQ pair tiles) */   /* work items completing one (dst,chunk) */
#define LG_NUM_XCD 8                  /* MI355X XCDs (inference): blocks t and t+8 share an XCD/L2 under round-robin dispatch */               // slab columns per destination (dense qkv: 2048 Q + 128 K + 128 V)

// ---- control window offsets (bytes) ----
#define LGC_EP_OFF 0u                 // u64 local epoch ep_d (host init = 1; advanced only by quiet_ready last step)
#define LGC_POISON_OFF 8u             // u32 sticky reason (0 = ok)
#define LGC_MAGIC_OFF 16u             // u64 LGC_MAGIC
#define LGC_CREDIT_OFF 64u            // credit[4] u64 (+8*d): written BY d into THIS window; value e = "d began epoch e"
#define LGC_ARRIVAL_OFF 128u          // arrival[4] u64 (+8*s): written BY s into THIS window; value e = "s's epoch-e PUTs done"
#define LGC_DONE_OFF 512u             // chunk_done[4][32] u32 (+4*(dst*32+c)), device-reset each epoch
#define LGC_STATE_OFF 1024u           // chunk record state[4][32] u32 (EMPTY/READY/POSTING/SUBMITTED), device-reset each epoch
#define LGC_LOCK_OFF 1536u            // submit lock[4] u32 (+4*dst), device-reset each epoch
#define LGC_REPORT_OFF 2048u          // report u64[8]: [0]=deferred_chunks (GEMM-side credit/lock miss) [1]=credit_wait_ticks (100 MHz)
                                      //   [2]=inline_posts [3]=quiet_posts (leftover) [4]=poison_reason [5]=epochs_done
// r3 report slots: [0]=deferred (= [6]+[7]) [2]=inline_posts (all puts issued during the GEMM incl. combined) [3]=quiet_posts (leftovers)
//   [4]=poison_reason [5]=epochs_done [6]=credit_miss (last-arriver saw credit<e) [7]=lock_miss (last-arriver lost the lock CAS)
//   invariant per epoch: inline_posts + quiet_posts == 3*chunks
#define LGC_POSTCNT_OFF 2112u         // DEBUG (LG_DEBUG_POSTCOUNT=1): post_count[4][32] u32 (+4*(dst*32+c)), CUMULATIVE over epochs, never device-reset (host reads/zeroes between epochs)
#define LGC_BYTES 8192u
#ifndef LG_DEBUG_POSTCOUNT
#define LG_DEBUG_POSTCOUNT 0          // 1 = count every put per (dst,chunk) (expected exactly 1 per epoch)
#endif
#ifndef LG_NEGCTRL_NOCAS
#define LG_NEGCTRL_NOCAS 0            // 1 = NEGATIVE CONTROL: READY->POSTING claim is load+store (NOT atomic); post_count must then show duplicates under contention. NEVER for measurement.
#endif
#define LG_SDMA_RING_BYTES 262144u    // CCO_SDMA_QUEUE_SIZE (pinned cco.hpp cce15596 l.1184), device pass static_asserts equality
#define LG_SDMA_UNIT_BYTES 32u        // CCO_SDMA_UNIT (l.1355): one COPY packet (no signals) = 1 unit
#ifndef LG_R6_POSTER
#define LG_R6_POSTER 1                /* r6: ONE resident poster block (blockIdx 0, 3 waves = 3 peers) posts n big puts per peer; 0 = r5b last-arriver CAS/combining path (kept as the control) */
#endif
/* r6x: puts per peer is selected by ENTRY (n1/n2/n4) and checked via LG_R6_PLAN_CODE_N(n); there is no build-time n. */
#ifndef LG_X_NEG_DOUBLE_PUT
#define LG_X_NEG_DOUBLE_PUT 0         /* X1 negative control (debug only, NEVER production; needs LG_DEBUG_POSTCOUNT=1): wave 1 (peer slot 1) issues group 0's put TWICE => post_count 2 + DUP_CHUNK poison */
#endif
#ifndef LG_X_POSTER_LAST
#define LG_X_POSTER_LAST 0            /* X-L test hook (debug only): poster = the LAST blockIdx (grid_items), items = blockIdx 0.. (poster dispatched last) */
#endif
#ifndef LG_NO_GRIDREAD
#define LG_NO_GRIDREAD 1              /* r10: 1 = the entry config guard uses the COMPILE-TIME plan grid (no AQL dispatch-packet grid_size_x s_load in any WG entry path); ONE item block (block id 1) compares the real packet value at the END of its work and poisons CONFIG on mismatch. 2 = no packet read at all (dispatch_ptr dropped from the descriptor; the host grid is the only check). 0 = r8g behaviour (every block reads grid_size_x at entry), for A/B. */
#endif
#ifndef LG_R8_FOLD
#define LG_R8_FOLD 0                  /* r8 A2: the poster block folds W2 (wait own arrivals) + device reset + epoch advance; the quiet_ready kernel is NOT launched. 0 = r7 behaviour (quiet_ready kernel). HOST AND DEVICE MUST AGREE (plan code bit 0x400). */
#endif
#define LGC_FOLD_CNT_OFF (LGC_LOCK_OFF + 60u)   /* u32 fold counter: lock word 15 (unused by the r6 poster path; r4 locks use words 0..15 only in the r5b CAS build); device-reset by the last folded waiter */
#define LG_R_PAD_BYTES 4096u          /* r8 A1: R window padding so a [T,2560] view with row stride 2304 (reuse) stays inside the allocation */
#ifndef LG_X_FORCE_TIMEOUT_G0
#define LG_X_FORCE_TIMEOUT_G0 0       /* r7 Y4 seam (debug only, NEVER production): the poster behaves as if its GROUP-0 wait timed out BEFORE any put (issued == 0): POSTER_WAIT poison, no put, no arrival (works for every n incl. the half-first entries) */
#endif
#ifndef LG_X_FORCE_TIMEOUT_G1
#define LG_X_FORCE_TIMEOUT_G1 0       /* X-A test hook (debug builds only, NEVER production): the poster behaves as if its g=1 wait timed out right after put(0) (needs LG_R6_PUTS_PER_PEER >= 2) */
#endif
#ifndef LG_R6_ARRIVAL_ACQ
#define LG_R6_ARRIVAL_ACQ 0           /* r6b default 0: producers' counter RMW is RELAXED after the release fence (review P-C2b: release sequence headed by the fence); 1 = ACQUIRE (the r5b-measured form) kept as an option */
#endif
#define LG_R6_PLAN_CODE_N(n, hg) ((LG_R6_POSTER ? 0x10000ull : 0ull) | (static_cast<uint64_t>(n) << 4) | (LG_REMOTE_HALF ? 1ull : 0ull) | (LG_X_INDEXER ? 0x100ull : 0ull) | (((hg) == 1) ? 0x200ull : (((hg) == 2) ? 0x800ull : 0ull)) | (LG_R8_FOLD ? 0x400ull : 0ull))   /* args.reserved[0] must equal this for the entry (poster, n, half, indexer, r7 half-first-group bit 0x200) */
#ifndef LG_L1C_SWIZZLE
#define LG_L1C_SWIZZLE 0              /* r5 L1c: XCC-blocked W n-band order (wgid = 8*j + x, XCC x owns contiguous bands, m-major slots). 0 = r4 order. Needs chunks % 8 == 0, else r4 order at runtime. */
#endif
#ifndef LG_L2A_BYPASS
#define LG_L2A_BYPASS 0               /* r5 L2a: staging (remote) slab stores as 16 B global_store_dwordx4 SC0|SC1 (L2 write-through). Self slab (R[rank]) stays cached. */
#endif
#ifndef LG_L2B_ONE_RELEASE
#define LG_L2B_ONE_RELEASE 1          /* r5 L2b: NO per-wave threadFenceSystem in gemm_chunk_complete; ONE system release per WG = thread 0 ACQ_REL fetch_add after the barrier (+ per-wave vmcnt(0)/lgkm(0) kept). Needs reviewer sign-off (W-4). */
#endif
#ifndef LG_NEGCTRL_NOWB
#define LG_NEGCTRL_NOWB 0             /* NEGATIVE CONTROL: no per-wave fence and a RELAXED chunk_done RMW (no writeback/release); with LG_L2A_BYPASS=0 stale S reads must show as R mismatches. NEVER for timing. */
#endif
#ifndef LG_NEGCTRL_NOWAIT
#define LG_NEGCTRL_NOWAIT 0           /* N-C4 NEGATIVE CONTROL (needs LG_L2A_BYPASS=1): BOTH per-wave vmcnt(0) waits removed + RELAXED chunk_done RMW + no fences; premature-completion bypass bug must show as R mismatches. NEVER timing. */
#endif
#ifndef LG_NEGCTRL_ISO
#define LG_NEGCTRL_ISO 0              /* N-C3 NOWB-ISO (combine with LG_NEGCTRL_NOWB=1, LG_L2A_BYPASS=0): inline posting OFF, no last-arriver/holder fences; all puts by quiet_ready. NEVER timing. */
#endif
#ifndef LG_ISO_DELAY_ITERS
#define LG_ISO_DELAY_ITERS 0          /* N-C3 debug delay (s_sleep(100) iterations in quiet_ready before posting); sweep {0, 10, 100}; NOTE a delay also gives natural L2 eviction time, so 0 is the strictest */
#endif
#ifndef LG_STAMPS
#define LG_STAMPS 0                   /* r5 richer stamps (s_memrealtime, 100 MHz): per-WG chunk-complete records + per-peer quiet_ready stamps + first-put. Debug build only. */
#endif
#define LGC_QSTAMP_OFF 4096u          /* quiet_ready stamps u64[3 peers][8] (+64*i): [0] start [1] credit ok [2] leftovers posted [3] quietQueue done [4] arrival released [5] own arrival seen [6] first-put time [7] peer rank */
#define LGC_XCCQ_OFF 4352u            /* per-XCC chunk-complete arrival counters u32[8] (queue position); device-reset by quiet_ready (LG_STAMPS only) */
#define LGC_FIRSTPUT_OFF 4416u        /* u64 time of the first put of the epoch (CAS 0->t); device-cleared by quiet_ready after it is copied (LG_STAMPS only) */
/* LGC_QSTAMP_OFF + 192: u64 raw XCC_ID of the quiet_ready WG (LG_STAMPS only) */
#define LGC_PSTAMP_OFF 5120u          /* r6 poster stamps u64[3 peers][16] (+128*peer), LG_STAMPS only: [0] poster start [1] credit released [2] credit observed [3..6] group g ready-seen [7..10] group g put issued [11] last quiet done [12] arrival released [13] first put [14] peer rank */
#define LGC_STAMP_OFF 65536u          /* per-WG records u64[8] at +64*blockIdx: [0] kernel entry [1] chunk-complete begin [2] after per-wave waits+barrier [3] after chunk_done RMW [4] end
                                         [5] info = xcc(4)|dst<<4(2)|chunk<<6(6)|last<<12(1)|qpos<<16(16)|bid<<32 [6] raw HW_REG_XCC_ID (unverified encoding) [7] time after this WG POSTED a chunk (last arriver only). Needs Cwin capacity >= 65536+64*grid. */
#define LG_MAX_INLINE_PER_HOLDER 64u  // B-1 cap: a lock holder posts at most this many chunks per scan; the rest stay READY for the next round / quiet_ready
#define LGC_MAGIC 0x3243474C514D4147ULL
enum { LGC_REC_EMPTY = 0, LGC_REC_READY = 1, LGC_REC_POSTING = 2, LGC_REC_SUBMITTED = 3 };   /* r3: READY->POSTING by CAS = exactly-once put */
enum { LGC_POISON_W1_TIMEOUT = 1, LGC_POISON_W2_TIMEOUT = 2, LGC_POISON_EPOCH_MISMATCH = 3, LGC_POISON_DUP_CHUNK = 4, LGC_POISON_CONFIG = 5, LGC_POISON_POSTER_WAIT = 6 };   /* 6: r6 poster bounded wait for a chunk group timed out */

#define LG_ENTRY_GEMM  "lg_gemm_quad_fp8_g1comm_v2"
#define LG_ENTRY_QUIET "lg_quiet_ready_v2"
#define LG_ENTRY_GEMM_N1 "lg_r6x_gemm_n1_v1"   /* r6x: puts per peer selected by ENTRY (1/2/4), one HSACO */
#define LG_ENTRY_GEMM_N2 "lg_r6x_gemm_n2_v1"
#define LG_ENTRY_GEMM_N4 "lg_r6x_gemm_n4_v1"
#define LG_ENTRY_GEMM_N2H "lg_r7_gemm_n2h_v1"   /* r7: half-N first group (group 0 remote tiles as 256x128 items), n = 2 / 4 */
#define LG_ENTRY_GEMM_N4H "lg_r7_gemm_n4h_v1"
#define LG_ENTRY_GEMM_N3Q "lg_r9_gemm_n3q_v1"   /* r9: quarter-first, THREE put groups per peer (chunks [0,2) as 128x128 quarter items, [2,3) and [3,8) as full tiles), dense M=2048 only; plan code bit 0x800 (host half_first = 2) */
#define LG_R9_QC 2                              /* chunks in group 0 of the quarter-first plan (the first LG_R9_QC chunk rows run as quarter items) */
#define LG_R9_G1 1                              /* chunks in group 1 (full tiles); group 2 = the remaining chunks - LG_R9_QC - LG_R9_G1 */
#define LG_ENTRY_DRAIN "lg_drain_all_v2"           /* r6b host recovery step: quietQueue on every peer queue (run + sync before any re-init after a poisoned epoch) */
#define LG_POLL_BUDGET_US_DEFAULT 50000u          /* r6b: poster/quiet_ready wait budget in microseconds (s_memrealtime, 100 MHz); passed in args.reserved[1] (must be non-zero). Must exceed worst-case GEMM time + rank skew + late poster dispatch at M=8192 (GEMM ~0.26-1.3 ms there). */

struct LgCommArgsV2 {                         /* size 320, align 8 */
    mori::cco::ccoDevComm dev;                /* 0   (200 B) */
    mori::cco::ccoWindow_t Swin;              /* 200 */
    mori::cco::ccoWindow_t Rwin;              /* 208 */
    mori::cco::ccoWindow_t Cwin;              /* 216 */
    const void* A;                            /* 224 FP8 [M,K] */
    const void* W;                            /* 232 FP8 [N,K] shuffle(16,16) */
    const float* Ascale;                      /* 240 [M] */
    const float* Wscale;                      /* 248 [N] raw order */
    uint32_t M;                               /* 256 multiple of 256, <= 8192 */
    uint32_t N;                               /* 260 9216 */
    uint32_t K;                               /* 264 6144 */
    uint32_t strideA;                         /* 268 */
    uint32_t F;                               /* 272 2304 */
    uint32_t rank;                            /* 276 == lsaRank == hipDev */
    uint32_t chunks;                          /* 280 M/256 */
    uint32_t abi_version;                     /* 284 == 3 */
    uint64_t poll_iterations;                 /* 288 bound for W1/W2 polls (each iteration s_sleep(1)) */
    uint32_t sdma_queues;                     /* 296 Q in {1,2,4}: chunk c of a peer uses queue c%Q; must equal ccoDevComm sdmaNumQueue */
    uint32_t items_per_group;                 /* 300 == LG_ITEMS_PER_GROUP (9 full / 18 half-N remote items per (dst,chunk)); kernel poisons on mismatch */
    uint64_t reserved[2];                     /* 304 must be 0 */
};
static_assert(sizeof(mori::cco::ccoDevComm) == 200, "DevComm 200B");
static_assert(std::is_trivially_copyable<mori::cco::ccoDevComm>::value, "DevComm by value");
static_assert(sizeof(LgCommArgsV2) == 320 && alignof(LgCommArgsV2) == 8, "LgCommArgsV2 320B");
static_assert(offsetof(LgCommArgsV2, Swin) == 200 && offsetof(LgCommArgsV2, A) == 224 &&
              offsetof(LgCommArgsV2, M) == 256 && offsetof(LgCommArgsV2, rank) == 276 &&
              offsetof(LgCommArgsV2, poll_iterations) == 288 && offsetof(LgCommArgsV2, sdma_queues) == 296 &&
              offsetof(LgCommArgsV2, items_per_group) == 300 && offsetof(LgCommArgsV2, reserved) == 304, "LgCommArgsV2 offsets");
