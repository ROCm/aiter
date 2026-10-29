// SPDX-License-Identifier: MIT
// LG-QSDMA-r19 communication-slice ABI v7 (large/r19 = the r13/large path extended to local M <= 16384; INTERNAL-ONLY).
// Linux LP64. Passed BY VALUE to both kernels. Uses the ACTUAL pinned CCO native types (mori/cco/cco.hpp).
// Windows (all MORI-managed, size-only ccoWindowRegister overload, registered BEFORE ccoDevCommCreate):
//   Swin : staging S_d = [4][M][F] BF16 (slab d at d*2MF bytes; slab `rank` unused on a FULL bucket: self goes straight to R)
//   Rwin : single receive buffer R = [4 sources][m_real][F] BF16 (== token-major [4*m_real, F], row = s*m_real + m); registered once
//   Cwin : control window (offsets below); epoch/credit/arrival words are monotonic u64 and never reset
// ABI 7 vs the r13 ABI 5 (the args struct stays 360 B):
//   * LG_MAX_CHUNKS 32 -> 64 => local M <= 16384 (chunks = M/256 <= 64; the geometry/data split of R16 "the kernel only gets the valid m" is unchanged);
//   * the 32-bit tag low word is repartitioned: (m_real - 1) gets 14 bits at LG_TAG_M_SHIFT 18, call_id keeps its LOW 18 bits (mask 0x3FFFF);
//   * the C window is re-laid out for the [4][64] arrays (DONE/STATE/POSTCNT grow to 4x64 u32; REPORT/CTAG/ATAG move; the EP/CREDIT/ARRIVAL words keep their r11..r13 offsets);
//   * protocol, order proofs, numerics, knobs and the host entry points are r13's (see R19-LARGE-NOTE.md for the tables and the proof strategy).
#pragma once
#include <stdint.h>
#include <stddef.h>
#include <type_traits>
#include <mori/cco/cco.hpp>

#define LG_COMM_ABI_VERSION 7u          /* r19 (ABI 7): r13's ABI 5 extended to M <= 16384: LG_MAX_CHUNKS 64, tag 14-bit m_real / 18-bit call_id, C-window arrays [4][64]. r14 (ABI 6, split-K `sk` in the old pad word) is a different, disjoint generation: none of its structures/offsets are reused here; r11 (ABI 3), r12 (4), r13 (5), r18 (5) are untouched. */
#define LG_CHUNK_ROWS 256u            // one chunk = one 256-row M tile
#define LG_MAX_CHUNKS 64u             // M <= 16384
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

// ---- control window offsets (bytes); ABI 7 layout for LG_MAX_CHUNKS == 64 (the ABI-5/r13 offsets are 512/1024/1536/2048/2112/3072/3104) ----
// The words the R12 protocol touches every epoch (EP / POISON / MAGIC / CREDIT / ARRIVAL) keep their r11..r13 offsets: ARRIVAL stays at 128 so a receiver written against
// the r13 contract still finds it (r19 itself uses only the in-tree quiet_ready). Everything sized by the chunk count moves; the exact partition is enforced by the
// static asserts at the end of this header:
//   [512, 1536)  chunk_done[4][64]         u32 (+4*(dst*LG_MAX_CHUNKS+c))  arrival counters of the r6x plan (group-indexed on the poster side), device-reset each epoch
//   [1536, 2560) chunk record state[4][64] u32 (r5b-combining path only: EMPTY/READY/POSTING/SUBMITTED), device-reset each epoch
//   [2560, 2576) submit lock[4]            u32 (r5b path only)
//   [2620, 2624) fold counter              u32 (LG_R8_FOLD path only; = LGC_LOCK_OFF + 60)
//   [2624, 2688) report u64[8]
//   [2688, 3712) DEBUG post_count[4][64]   u32 (LG_DEBUG_POSTCOUNT=1; cumulative, never device-reset)
//   [3712, 3744) credit tag u64[4]         (+8*d)
//   [3744, 3776) arrival tag u64[4]        (+8*s)
// The remaining areas (quiet-ready stamps at 4096, per-WG stamp records at 65536) are debug-only and do not collide with the above.
#define LGC_EP_OFF 0u                 // u64 local epoch ep_d (host init = 1; advanced only by quiet_ready last step)
#define LGC_POISON_OFF 8u             // u32 sticky reason (0 = ok)
#define LGC_MAGIC_OFF 16u             // u64 LGC_MAGIC
#define LGC_CREDIT_OFF 64u            // credit[4] u64 (+8*d): written BY d into THIS window; value e = "d began epoch e"
#define LGC_ARRIVAL_OFF 128u          // arrival[4] u64 (+8*s): written BY s into THIS window; value e = "s's epoch-e PUTs done"
#define LGC_DONE_OFF 512u             // chunk_done[4][64] u32 (+4*(dst*LG_MAX_CHUNKS+c)), device-reset each epoch
#define LGC_STATE_OFF 1536u           // chunk record state[4][64] u32 (EMPTY/READY/POSTING/SUBMITTED), device-reset each epoch
#define LGC_LOCK_OFF 2560u            // submit lock[4] u32 (+4*dst), device-reset each epoch
#define LGC_REPORT_OFF 2624u          // report u64[8]: [0]=deferred_chunks (GEMM-side credit/lock miss) [1]=credit_wait_ticks (100 MHz)
                                      //   [2]=inline_posts [3]=quiet_posts (leftover) [4]=poison_reason [5]=epochs_done
// r3 report slots: [0]=deferred (= [6]+[7]) [2]=inline_posts (all puts issued during the GEMM incl. combined) [3]=quiet_posts (leftovers)
//   [4]=poison_reason [5]=epochs_done [6]=credit_miss (last-arriver saw credit<e) [7]=lock_miss (last-arriver lost the lock CAS)
//   invariant per epoch: inline_posts + quiet_posts == 3*chunks
#define LGC_POSTCNT_OFF 2688u         // DEBUG (LG_DEBUG_POSTCOUNT=1): post_count[4][64] u32 (+4*(dst*LG_MAX_CHUNKS+c)), CUMULATIVE over epochs, never device-reset (host reads/zeroes between epochs)
#define LGC_CTAG_OFF 3712u            /* r12: credit tag u64[4] (+8*d): written BY d into THIS window BEFORE its credit release; value E = (STEP<<32)|((m_real-1)<<18)|call_id[17:0] of d's launch */
#define LGC_ATAG_OFF 3744u            /* r12: arrival tag u64[4] (+8*s): written BY s into THIS window BEFORE its arrival release; same E */
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
#ifndef LG_TAG_EARLY
#define LG_TAG_EARLY 1                /* r12b (overhead study F6/F1): 1 = the ARRIVAL tag is stored right after the credit-tag check, i.e. in the batch of the first put (before the puts), by the poster lane that later releases the arrival: by the time of the arrival release (after quietQueue) it has long retired, so the release fence has no outstanding remote store. 0 = r12 behaviour (tag store immediately before the arrival release). The credit tag cannot move earlier (it is the poster's first action; its RTT is hidden behind GEMM group 0). Safe: the store follows the peer's credit e, which the peer issues only after its quiet_ready e-1 read atag[us]. */
#endif
#ifndef LG_POLL_BATCH
#define LG_POLL_BATCH 1               /* r12b (F2): 1 = W1/W2/credit/gdone polls load the target word and ERR back-to-back (one wait, one RTT per iteration); 2 = ERR only every 8th iteration; 0 = r12 behaviour (ERR loaded after the target compare). ERR is a liveness signal: a late observation only delays the sticky skip (bounded). */
#endif
#ifndef LG_R13_POSTER
#define LG_R13_POSTER 1               /* r13f attribution knob: 1 = r13 poster/tag/put code (live-row puts, skip-put path, m_real in the tag, unrolled-off group loop); 0 = r12's poster/tag/put code EXACTLY (full-bucket puts, tag = (STEP<<32)|call_id, slot with M): correct only at m_real == M; A/B only, never production */
#endif
/* NOTE (r19): r13's LG_R13_ARGS attribution knob is GONE. Its `=0` arm was the 352 B V3 args layout with no m_real/pad4; the ABI-7 generation defines m_real
   unconditionally (the tag/m_real geometry is part of the generation), so the knob cannot be expressed and the V3 layout is not reachable in this tree. */
#ifndef LG_R13_ABOUND
#define LG_R13_ABOUND 1               /* r13d attribution knob: 1 = A buffer num_records = max(m_real - row, 0) * strideA; 0 = r12's EXACT expression (M - row) * strideA (bucket M; correct only at m_real == M; A/B only, never production) */
#endif
#ifndef LG_R13_MR_SGPR
#define LG_R13_MR_SGPR 1              /* r13d attribution knob: 1 = m_real through __builtin_amdgcn_readfirstlane (SGPR); 0 = plain kernarg read like r12 reads M (A/B only) */
#endif
#ifndef LG_R13_RSTRIDE
#define LG_R13_RSTRIDE 1              /* r13d attribution knob: 1 = R/S slab stride and put slot use m_real rows; 0 = M rows like r12 (correct only at m_real == M; A/B only, never production) */
#endif
#ifndef LG_R13_STORE_MASK
#define LG_R13_STORE_MASK 1           /* r13 (ISA attribution knob): 1 = dead-row stores (row >= m_real) are redirected branchlessly to the dump line; 0 = no redirect (CORRECT ONLY when m_real == M; for register/ISA A/B only, never production) */
#endif
#ifndef LG_R13_ASCALE_CLAMP
#define LG_R13_ASCALE_CLAMP 1         /* r13 (ISA attribution knob): 1 = Ascale index clamped to m_real-1 (no OOB read of the caller's Ascale); 0 = r12 load (OOB-unsafe unless m_real == M; A/B only) */
#endif
#ifndef LG_X_TEST_BAD_CTAG
#define LG_X_TEST_BAD_CTAG 0          /* r12 seam (debug only, NEVER production): the poster of peer slot 0 stores E ^ 1 as its credit tag => the receiving peer must set ERR TAG_CREDIT (0x31) before any put */
#endif
#ifndef LG_X_TEST_BAD_ATAG
#define LG_X_TEST_BAD_ATAG 0          /* r12 seam (debug only): the poster of peer slot 0 stores E ^ 1 as its arrival tag => the receiving peer's quiet_ready must set ERR TAG_ARRIVAL (0x32) */
#endif
#ifndef LG_X_TEST_CTAG_AFTER_RELEASE
#define LG_X_TEST_CTAG_AFTER_RELEASE 0 /* r12 seam (debug only): the credit tag store is issued AFTER the credit release (+ LG_X_TEST_TAG_DELAY_ITERS s_sleep(100) iterations): with a delay the consumer sees a stale tag => TAG_CREDIT mismatches MUST appear (negative control of the store-before-release rule) */
#endif
#ifndef LG_X_TEST_ATAG_AFTER_RELEASE
#define LG_X_TEST_ATAG_AFTER_RELEASE 0 /* r12 seam (debug only): same for the arrival tag => TAG_ARRIVAL mismatches MUST appear */
#endif
#ifndef LG_X_TEST_TAG_DELAY_ITERS
#define LG_X_TEST_TAG_DELAY_ITERS 0   /* r12 seam: s_sleep(100) iterations before the late tag store of the AFTER_RELEASE seams */
#endif
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
/* ---- r12 per-comm ERR page (global, owned by the aiter FusionResources; layout of gf_global_abi.h, card D2.1) ----
   page base = `err` (u32 ERR at +0), `step` = page base + 8 (u64 STEP, ATOM-incremented once per forward), DIAG u64[8] at +64 (written once by the CAS winner):
   DIAG[0] = expected tag E, [1] = tag/value seen, [2] = auxiliary word (credit/arrival value), [3] = plan code (args.reserved[0]), [4] = epoch e, [5] = rank, [6..7] = 0.
   ERR encoding: reason[7:0] | origin_rank[9:8] | kind[11:10] (1 small, 2 large) | propagated[12]. The first CAS (0 -> code) wins. */
#define LG_GF_ERR_OFF 0u
#define LG_GF_STEP_OFF 8u
#define LG_GF_DIAG_OFF 64u
#define LG_ERR_REASON_MASK 0xFFu
#define LG_ERR_ORIGIN_SHIFT 8u
#define LG_ERR_KIND_SHIFT 10u
#define LG_ERR_KIND_LARGE 2u
#define LG_ERR_PROPAGATED (1u << 12)
struct LgGfPeers { uint32_t* err[4]; };       /* device-resident: err[r] = the PEER rank r's ERR word through the flat-VA map (ccoGetPeerPtr); err[own rank] is ignored. Built by the host that owns the page (card D2.6). */
static_assert(sizeof(LgGfPeers) == 32, "LgGfPeers 32B");
enum { LGC_REC_EMPTY = 0, LGC_REC_READY = 1, LGC_REC_POSTING = 2, LGC_REC_SUBMITTED = 3 };   /* r3: READY->POSTING by CAS = exactly-once put */
enum { LGC_POISON_W1_TIMEOUT = 1, LGC_POISON_W2_TIMEOUT = 2, LGC_POISON_EPOCH_MISMATCH = 3, LGC_POISON_DUP_CHUNK = 4, LGC_POISON_CONFIG = 5, LGC_POISON_POSTER_WAIT = 6, LGC_POISON_NO_GPAGE = 8, LGC_POISON_TAG_CREDIT = 0x31, LGC_POISON_TAG_ARRIVAL = 0x32 };   /* r12: 8 = launch without an ERR/STEP page (mirror only), 0x31 / 0x32 = ctag / atag != E (card: tag mismatch 0x3x) */   /* 6: r6 poster bounded wait for a chunk group timed out */

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
/* r12c (overhead-r2 A1): NO separate NaN-fill node. quiet_ready runs LG_QR_WGS workgroups (x 256 threads): WG0 = the W2/reset/epoch role (unchanged); every WG reads ERR ONCE at entry; WG >= 1 exits when ERR == 0, else grid-stride fills its share of
   R (+ the reuse pad) with BF16 NaN; if WG0 itself sees ERR during this launch (it was clear at its entry) WG0 fills ALL of R alone. No inter-WG wait or flag. 1 = A1' (single WG: identical to the r11 healthy path; the fill is done by WG0 alone / its stride). */
#ifndef LG_QR_WGS
#define LG_QR_WGS 64u
#endif
static_assert(LG_QR_WGS >= 1u && LG_QR_WGS <= 256u, "LG_QR_WGS in [1, 256]");
#define LG_ENTRY_DRAIN "lg_drain_all_v2"           /* r6b host recovery step: quietQueue on every peer queue (run + sync before any re-init after a poisoned epoch) */
#define LG_POLL_BUDGET_US_DEFAULT 50000u          /* r6b: poster/quiet_ready wait budget in microseconds (s_memrealtime, 100 MHz); passed in args.reserved[1] (must be non-zero). Must exceed worst-case GEMM time + rank skew + late poster dispatch at M=16384 (GEMM ~0.55-1.45 ms there on the r13 recipe; the profile passes 5 s). */

struct LgCommArgsV6 {                         /* size 360, align 8 (the r13 V4 layout: V3 352 B + m_real/pad4; ABI 7 repartitions the tag word and the C window only) */
    mori::cco::ccoDevComm dev;                /* 0   (200 B) */
    mori::cco::ccoWindow_t Swin;              /* 200 */
    mori::cco::ccoWindow_t Rwin;              /* 208 */
    mori::cco::ccoWindow_t Cwin;              /* 216 */
    const void* A;                            /* 224 FP8 [m_real,K] */
    const void* W;                            /* 232 FP8 [N,K] shuffle(16,16) */
    const float* Ascale;                      /* 240 [m_real] */
    const float* Wscale;                      /* 248 [N] raw order */
    uint32_t M;                               /* 256 multiple of 256, <= 16384 (the plan's bucket; geometry comes from M) */
    uint32_t N;                               /* 260 9216 */
    uint32_t K;                               /* 264 6144 */
    uint32_t strideA;                         /* 268 */
    uint32_t F;                               /* 272 2304 */
    uint32_t rank;                            /* 276 == lsaRank == hipDev */
    uint32_t chunks;                          /* 280 M/256, <= LG_MAX_CHUNKS (64) */
    uint32_t abi_version;                     /* 284 == LG_COMM_ABI_VERSION (7) */
    uint64_t poll_iterations;                 /* 288 bound for W1/W2 polls (each iteration s_sleep(1)) */
    uint32_t sdma_queues;                     /* 296 Q in {1,2,4}: chunk c of a peer uses queue c%Q; must equal ccoDevComm sdmaNumQueue */
    uint32_t items_per_group;                 /* 300 == LG_ITEMS_PER_GROUP (9 full / 18 half-N remote items per (dst,chunk)); kernel poisons on mismatch */
    uint64_t reserved[2];                     /* 304 [0] plan code, [1] wait budget in us (non-zero) */
    const uint64_t* step;                     /* 320 STEP word (page base + 8); read once at kernel entry (agent-scope relaxed) */
    uint32_t* err;                            /* 328 per-comm ERR word (page base); CAS-set by the first failing kernel, read at entry and in every poll */
    const LgGfPeers* gpeers;                  /* 336 peer ERR words for propagation (cold path), may be null (no propagation) */
    uint32_t call_id;                         /* 344 31-bit rank-uniform key of (site, tokens, vid, n_fused); only its LOW 18 bits reach the tag */
    uint32_t qr_wgs;                          /* 348 == LG_QR_WGS of the HOST build; the device compares it with its own LG_QR_WGS (guard / quiet_ready WG0 => CONFIG): a host launching 64 WGs into a 1-WG device image is refused */
    uint32_t m_real;                          /* 352 r19: VALID rows of this launch, 1..M (M = the plan's bucket, multiple of 256). Geometry (tiles, chunks, grid, windows, K counts) comes from M; DATA from m_real: A/Ascale rows >= m_real are never read, results of rows >= m_real are never stored or sent, R row base = src * m_real + m (a contiguous R[:4*m_real]), put length = live rows, tag packs m_real */
    uint32_t pad4;                            /* 356 must be 0 */
};
static_assert(sizeof(mori::cco::ccoDevComm) == 200, "DevComm 200B");
static_assert(std::is_trivially_copyable<mori::cco::ccoDevComm>::value, "DevComm by value");
static_assert(sizeof(LgCommArgsV6) == 360 && alignof(LgCommArgsV6) == 8, "LgCommArgsV6 360B");
#define LG_MREAL(a) ((a).m_real)
static_assert(offsetof(LgCommArgsV6, Swin) == 200 && offsetof(LgCommArgsV6, A) == 224 &&
              offsetof(LgCommArgsV6, M) == 256 && offsetof(LgCommArgsV6, rank) == 276 &&
              offsetof(LgCommArgsV6, poll_iterations) == 288 && offsetof(LgCommArgsV6, sdma_queues) == 296 &&
              offsetof(LgCommArgsV6, items_per_group) == 300 && offsetof(LgCommArgsV6, reserved) == 304 &&
              offsetof(LgCommArgsV6, step) == 320 && offsetof(LgCommArgsV6, err) == 328 && offsetof(LgCommArgsV6, gpeers) == 336 && offsetof(LgCommArgsV6, call_id) == 344 && offsetof(LgCommArgsV6, qr_wgs) == 348 &&
              offsetof(LgCommArgsV6, m_real) == 352 && offsetof(LgCommArgsV6, pad4) == 356, "LgCommArgsV6 offsets");
/* r19 tag packing: E = (STEP << 32) | ((m_real - 1) << LG_TAG_M_SHIFT) | (call_id & LG_TAG_CALLID_MASK). m_real in 1..16384 -> 14 bits at shift 18;
   call_id keeps its LOW 18 bits (the runtime's static uniqueness check — the callid gate — must hold on the 18-bit ids). */
#define LG_TAG_CALLID_BITS 18u
#define LG_TAG_CALLID_MASK 0x3FFFFu
#define LG_TAG_M_SHIFT 18u
static_assert(LG_MAX_CHUNKS * LG_CHUNK_ROWS <= (1u << (32u - LG_TAG_M_SHIFT)), "m_real - 1 must fit the tag's 14 bits");
static_assert(LG_TAG_M_SHIFT == LG_TAG_CALLID_BITS && (1u << LG_TAG_CALLID_BITS) - 1u == LG_TAG_CALLID_MASK && LG_TAG_M_SHIFT + 14u == 32u, "tag low word = 18-bit call_id | 14-bit m_real");
/* ABI-7 control-window layout (bytes): every array is sized by LG_MAX_CHUNKS, the protocol words keep their offsets, nothing overlaps; the debug areas are clear. */
static_assert(LGC_DONE_OFF == 512u && LGC_STATE_OFF == LGC_DONE_OFF + 4u * 4u * LG_MAX_CHUNKS &&
              LGC_LOCK_OFF == LGC_STATE_OFF + 4u * 4u * LG_MAX_CHUNKS &&
              LGC_LOCK_OFF + 16u <= LGC_FOLD_CNT_OFF && LGC_FOLD_CNT_OFF + 4u <= LGC_REPORT_OFF &&
              LGC_REPORT_OFF + 64u <= LGC_POSTCNT_OFF && LGC_POSTCNT_OFF + 4u * 4u * LG_MAX_CHUNKS == LGC_CTAG_OFF &&
              LGC_CTAG_OFF + 32u == LGC_ATAG_OFF && LGC_ATAG_OFF + 32u <= LGC_QSTAMP_OFF &&
              LGC_QSTAMP_OFF + 192u <= LGC_XCCQ_OFF && LGC_FIRSTPUT_OFF + 8u <= LGC_PSTAMP_OFF &&
              LGC_PSTAMP_OFF + 3u * 16u * 8u <= LGC_STAMP_OFF && LGC_BYTES <= LGC_STAMP_OFF, "r19 ABI-7 control layout");
