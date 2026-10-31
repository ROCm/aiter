// SPDX-License-Identifier: MIT
// SM-SK2-RAN-r2 private ABI v2 (adds dbg pointer; SM_STAMPS layout) (small-M dense64). NEW header; does not reuse or alter Sp4cDeviceArgsV1.
// UNCOMPILED. Window pointers are obtained ONLY via mori::cco ccoGetLocalPtr / ccoGetLsaPeerPtr.
#pragma once
#include <stdint.h>
#include <stddef.h>
#include <type_traits>
#include <mori/cco/cco.hpp>

#define SM_ABI_VERSION 2u
// Fixed dense64 geometry for this slice.
// r14 (D9 P3): split-K factor, compile-time. 2 = r13 (A/B), 3 = default (K=6144 = 48 K128-blocks -> 16 blocks per split; 3 x 144 = 432 GEMM WGs (462 indexer/reuse) <= 512 slots at 2 WGs/CU). SK=4 does not fit
// (576 WGs > 512 slots at 2 WG/CU; 3 WG/CU needs 221 KB LDS > 160 KB), see the note. The partial-sum order becomes ((0+p0)+p1)+p2: numerics gate = original .02 + K scale/cache BYTE-EXACT vs golden.
#ifndef SM_SK_CFG
#define SM_SK_CFG 3
#endif
#if SM_SK_CFG == 2
#define SM_SK        2u
#else
#define SM_SK        3u
#endif
static_assert(SM_SK_CFG == 2 || SM_SK_CFG == 3, "SM_SK_CFG in {2,3}");
#define SM_MP        32u            // padded rows per source (live rows M<=16 for T=64)
// r10: SM_DENSE=1 = qkv_dense (Q16,K,V per destination; no IQ/IK, no index cache): F=2304 (Q2048 K128 V128) = 72 units, N=9216, 18 heads per source. Implies SM_R5X_INDEXER=0 (reuse numerics path).
#ifndef SM_DENSE
#define SM_DENSE 0
#endif
#if SM_DENSE
#define SM_N         9216u
#define SM_F         2304u
#else
#define SM_N         9856u            // r5x: F=2560 packed columns (Q2048 K128 V128 IQ128 IK128) = 77 units of 128
#define SM_F         2560u
#endif
#define SM_UNIT_COLS 128u
#if SM_DENSE
#define SM_NA        72u            // units = N/128
#define SM_HEADS_SRC 18u            // 16 Q + K + V heads per source (dense)
#define SM_NB        72u            // 4 sources * 18 heads
#else
#define SM_NA        77u            // units = N/128
#define SM_HEADS_SRC 20u            // 16 Q + K + V + IQ + IK heads per source (indexer)
#define SM_NB        80u            // 4 sources * 20 heads
#endif

// Control window layout (bytes, own rank's Cwin; all u64/u32 naturally aligned).
#define SM_C_EPOCH_G 40u            // u64: epoch for the launch pair, WRITTEN by sm_gemm (= ep_r+1), READ by RAN. SM_C_EPOCH (below) = ep_r: written by RAN, read by GEMM.
#define SM_C_EPOCH   0u             // u64: completed epochs; current e = value+1
#define SM_C_TICKA   8u             // u32
#define SM_C_TICKB   12u            // u32
#define SM_C_DONE    16u            // u32
#define SM_C_POISON  20u            // u32 (sticky)
#define SM_C_SRCDONE 24u            // u32[4]
#define SM_C_CREDIT  64u            // u64[4][2]: credit written BY dest d into source's window, [d][par]
#define SM_C_FLAGS   256u           // u64[2][4][SM_NA]: flag[par][src][unit] = epoch (written remotely by src); parity-indexed like Rx
#define SM_C_FLAG_OFF(par,src,u) (SM_C_FLAGS + 8u*(((par)*4u + (src))*SM_NA + (u)))
#define SM_C_BYTES   (SM_C_FLAGS + 8u*2u*4u*SM_NA)

struct SmArgsV1 {
    mori::cco::ccoDevComm dev;
    mori::cco::ccoWindow_t Rwin;    // receive window: [par][src][SM_MP][F] bf16
    mori::cco::ccoWindow_t Cwin;    // control window (SM_C_BYTES)
    const float* P;                 // partials [SM_SK][SM_MP][SM_N] fp32, private, 256B-aligned
    const float* Ascale;            // source's own per-token scale [M]
    const float* Wscale;            // source's own per-channel scale [N]
    // dest-owned receiver inputs (golden dense numerics)
    const float* qw; const float* kw;               // [128]
    const uint16_t* cos_cache; const uint16_t* sin_cache; // BF16 bits [max_pos][32] (golden operators.py:93 .bfloat16()); loaded BF16->fp32
    int64_t cos_stride_pos;
    const int64_t* positions;       // [T]
    const int64_t* slot_mapping;    // [T]
    void* k_cache; void* v_cache;   // fp8, shuffle layouts (strides below)
    float* k_scale; float* v_scale;
    int64_t kc_sb, kc_sh, kc_sdx, kc_ss, kc_sx;
    int64_t vc_sb, vc_sh, vc_ssc, vc_sd, vc_sx;
    int64_t ks_sb, ks_sh, vs_sb, vs_sh;
    uint16_t* q_out;                // bf16 [T][16*128]
    uint16_t* k_out;                // bf16 [T][128]
    uint16_t* v_out;                // bf16 [T][128] (raw V, dense/reuse path; may be nullptr in indexer mode - golden returns no v)
    const float* iqw; const float* ikw;   // [128] FP32 exact copies of the BF16 index_q / index_k norm weights (indexer)
    uint16_t* index_q_out;          // bf16 [T][128] (indexer)
    void* index_cache;              // fp8 [slots][128] flat (page 128), indexer only; reuse never touches it
    const int64_t* index_slot_mapping;  // [T] (golden passes slot_mapping)
    uint32_t M;                     // live rows per source, runtime 1..SM_MP(32); T must equal 4*M (kernel poisons 0x40 otherwise)
    uint32_t T;                     // total tokens = 4*M
    uint32_t rank;
    uint32_t block_size;            // 128 (dense)
    uint32_t x_size;
    uint32_t pad0;
    float eps;                      // 1e-6
    float fp8_amax;                 // finfo(k_cache.dtype).max
    uint64_t poll_iterations;
    // U2: host-resolved raw window pointers (local + the 4 flat peer addresses of the Cwin/Rwin windows, via ccoGetPeerPtr in the driver)
    uint8_t* cwin_loc; uint8_t* rwin_loc;
    uint8_t* cwin_pr0; uint8_t* cwin_pr1; uint8_t* cwin_pr2; uint8_t* cwin_pr3;
    uint8_t* rwin_pr0; uint8_t* rwin_pr1; uint8_t* rwin_pr2; uint8_t* rwin_pr3;
    uint64_t* dbg;                  // SM_STAMPS build only (else nullptr): see SM_DBG_* layout
};
static_assert(alignof(SmArgsV1) == 8, "SmArgsV1 align");
// GEMM kernel args (plain POD; no CCO). Grid: (N/64, SK, 1), 256 threads. K = 6144.
struct SmGemmArgsV1 {
    const void* A;          // fp8 E4M3FN [M][K], row stride strideA bytes; only M (<=16) live rows exist in memory
    const void* W;          // fp8 E4M3FN preshuffled (16,16) [N][K]
    float* P;               // [SM_SK][SM_MP][N] fp32 partials, 256B aligned
    uint32_t M, N, K, strideA;
    uint32_t sk;            // must equal SM_SK
    uint32_t pad0;
    mori::cco::ccoWindow_t Cwin;   // control window (GEMM writes ep_g = ep_r + 1; T2 also credit/flags)
    uint64_t* dbg;          // SM_STAMPS build only (else nullptr)
};
static_assert(alignof(SmGemmArgsV1) == 8, "SmGemmArgsV1 align");

// ---- SM_STAMPS debug buffer (diagnostic build only; separate binary). u64 words, written by thread 0 of each WG ----
// dbg[0] = replay counter (advanced by the last RAN WG after all stamps); record base = SM_DBG_HDR + ((cnt % SM_DBG_SLOTS)*(SM_DBG_RAN_MAXWG+SM_DBG_GEMM_MAXWG) + kind_off + wg) * SM_DBG_WORDS
#define SM_DBG_SLOTS       4u
#define SM_DBG_WORDS       64u
#define SM_DBG_RAN_MAXWG   128u
#define SM_DBG_GEMM_MAXWG  480u
#define SM_DBG_HDR         8u
#define SM_RX_BYTES        (2u*4u*SM_MP*SM_F*2u)   // Rx window bytes [par][src][Mp][F] bf16 (must be 16B-aligned base)
#define SM_P3_GRID         15u                     // SM_P3 static roles: 3 push + 3 self + 9 B WGs
#define SM_DBG_TOTAL_WORDS (SM_DBG_HDR + SM_DBG_SLOTS*(SM_DBG_RAN_MAXWG+SM_DBG_GEMM_MAXWG)*SM_DBG_WORDS)
// RAN record word map: 0 entry,1 first A ticket,2 A compute done,3 push retired,4 release done,5 A exit,6 first B ticket,
//  7 first poll start,8 flag observed,9 B math done,10 B stores retired,11 credit/SRCDONE done,12 B exit,13 tail done,14 exit,
//  15 poll iterations (first B unit), 16 hw_id, 17 first A unit id, 18 first B unit id, 19 n A units, 20 n B units, 21..23 reserved
// GEMM record word map: 0 entry, 1 first stage ready, 2 loop exit, 3 partial stores retired, 4 exit, 16 hw_id
static_assert(sizeof(void*) == 8, "64-bit");

// r3 additions (STAMPS, 64-word records). RAN: 21 spare, 24..31 A-unit k publish-done (k<8), 32..39 B-unit k flag seen,
//  40..47 B-unit k body done, 48..55 B-unit k credit done, 56 max credit-poll iterations, 57 sum flag-poll iterations,
//  58 XCC_ID, 59 hw_id. P3: 24 push start, 25 push retired, 26 flag released (push WGs); 27 self start, 28 self done.
// GEMM: 0 entry, 1 producer first stage, 2 producer loop exit, 3 consumer partial stores retired, 4 producer exit,
//  5 consumer first stage, 6 consumer loop exit, 7 consumer exit, 16 hw_id, 58 XCC_ID.

// ---- SM-T1-r4 (Track 1) control-window words: release-sequence counters (monotonic u64, never reset; host zeroes the window once) ----
// flag[src] lives in the DESTINATION window: K_sd pushers each add 1 (system release) per epoch; the receiver waits flag >= K_sd*e and
//   poisons only if flag > K_sd*(e+1) (a source may run at most one epoch ahead). credit[d] lives in the SOURCE window: each of the 9 B WGs of (src=me, dst=d) adds 1 per epoch;
//   the source requires credit[d] >= 9*(e-2) (e>2) before touching Rx parity (e&1).
// ---- SM-T1-r5 static roles: 24 push (3 per XCC, XCC = wgid%8) + 18 self + 3*B_PER_PAIR B ----
#ifndef SM_T1_B_HEADS
#define SM_T1_B_HEADS       2u    // U3 A/B flag: heads per B WG (1 -> 54 B WGs, 2 -> 27 B WGs pipelined)
#endif
#ifndef SM_U2_HOST_PTRS
#define SM_U2_HOST_PTRS     1     // U2: kernel uses host-resolved window pointers from kernargs (0 = r4 ccoGetLocalPtr/ccoGetLsaPeerPtr in the kernel)
#endif
// r15 (D10): push-role shape switches (default 0 = r14 byte-identical). SM_PUSH8 (Q1, M <= 16 only): 8 push WGs, ONE per XCC, 512 threads, the two 256-thread halves split the WG's 6..8 units; a WG may straddle two
// destinations (one release per touched destination, issued by two different waves). SM_PUSH_HALVES (Q3, M up to 32): 24 push WGs at 512 threads, half 0 owns row pass 0 and half 1 row pass 1. Either switch makes the whole RAN launch
// 512 threads per WG; the self/IK/B WGs retire waves 4-7 at kernel start. SM_NT_STORES (Q2): non-temporal stores for the GEMM P partials and the B/self outputs, caches, scales. All three are independent build flags.
#ifndef SM_PUSH8
#define SM_PUSH8 0
#endif
#ifndef SM_PUSH_HALVES
#define SM_PUSH_HALVES 0
#endif
#ifndef SM_NT_STORES
#define SM_NT_STORES 0
#endif
#if SM_PUSH8 && SM_PUSH_HALVES
#error "SM_PUSH8 and SM_PUSH_HALVES are exclusive"
#endif
#if SM_PUSH8 || SM_PUSH_HALVES
#define SM_RAN_THREADS      512u
#else
#define SM_RAN_THREADS      256u
#endif
#if SM_PUSH8
#define SM_T1_PUSH_WGS      8u
#else
#define SM_T1_PUSH_WGS      24u
#endif
#ifndef SM_R5X_INDEXER
#if SM_DENSE
#define SM_R5X_INDEXER      0
#else
#define SM_R5X_INDEXER      1
#endif
#endif
#if SM_DENSE && SM_R5X_INDEXER
#error "SM_DENSE requires SM_R5X_INDEXER=0"
#endif
#ifndef SM_R5X_GEMM_SKIP_IDX
#define SM_R5X_GEMM_SKIP_IDX 0    // reuse only: 1 = GEMM grid skips the IQ/IK N tiles (144 instead of 154). NOT in the approved scope unless the lead says so.
#endif
#if SM_R5X_INDEXER && SM_R5X_GEMM_SKIP_IDX
#error "SM_R5X_GEMM_SKIP_IDX is reuse-only"
#endif
#if SM_R5X_INDEXER
#define SM_T1_UPD           19u   // units per remote destination (Q16 + K + V + IQ)
#define SM_T1_HEADS_SRC     20u   // heads per source seen by a destination (+ IK)
#define SM_T1_IK_WGS        1u    // ONE IK owner WG per source rank
#else
#define SM_T1_UPD           18u
#define SM_T1_HEADS_SRC     18u
#define SM_T1_IK_WGS        0u
#endif
#define SM_T1_SELF_WGS      SM_T1_UPD
#define SM_T1_S0            (SM_T1_PUSH_WGS + SM_T1_IK_WGS)                      // first SELF WG
#define SM_T1_B0            (SM_T1_PUSH_WGS + SM_T1_IK_WGS + SM_T1_SELF_WGS)     // first B WG
#define SM_GEMM_NTILES      ((SM_R5X_GEMM_SKIP_IDX) ? 144u : (SM_N / 64u))
#define SM_T1_B_PER_PAIR    (SM_T1_HEADS_SRC / SM_T1_B_HEADS)
#define SM_T1_B_WGS         (3u * SM_T1_B_PER_PAIR)
#define SM_T1_GRID          (SM_T1_B0 + SM_T1_B_WGS)   // indexer: 74 / 104; reuse: 69 / 96
#define SM_T1_FLAG(src)     (SM_C_FLAGS + 8u*(src))     // ONE monotone u64 per (src,dst) in the DESTINATION window (not parity-indexed)
#define SM_T1_CREDIT(d)     (SM_C_CREDIT + 16u*(d))

// ---- SM-T1-r7 (D7 K1): RAN kernarg = { const SmDev* blk, uint32 M, uint32 rank } (16 B). SmDev is a DEVICE-RESIDENT IMMUTABLE block written once per bind (outside capture).
// Role-first layout, 64 B groups (one s_load_dwordx16 each): [0] push set, [1] window/pointer set, [2] head-common set, [3] cache set, [4] stride set, [5] misc.
struct SmDev {
    // ---- group 0 (push path): P, Wscale, Ascale, local control window, the 4 peer Rx windows
    const float* P; const float* Wscale; const float* Ascale; uint8_t* cwin_loc;
    uint8_t* rwin_pr0; uint8_t* rwin_pr1; uint8_t* rwin_pr2; uint8_t* rwin_pr3;
    // ---- group 1: control peers + misc pointers
    uint8_t* cwin_pr0; uint8_t* cwin_pr1; uint8_t* cwin_pr2; uint8_t* cwin_pr3;
    uint64_t poll_iterations; uint64_t* dbg; uint8_t* rwin_loc; int64_t cos_stride_pos;
    // ---- group 2: head-common (self/B)
    const float* qw; const float* kw; const uint16_t* cos_cache; const uint16_t* sin_cache;
    const int64_t* positions; const int64_t* slot_mapping; uint16_t* q_out; uint16_t* k_out;
    // ---- group 3: caches
    uint16_t* v_out; void* k_cache; void* v_cache; float* k_scale; float* v_scale; int64_t kc_sb; int64_t kc_sdx; int64_t kc_ss;
    // ---- group 4: strides
    int64_t kc_sx; int64_t vc_sb; int64_t vc_ssc; int64_t vc_sd; int64_t vc_sx; int64_t ks_sb; int64_t vs_sb; int64_t kc_sh_unused;
    // ---- group 5: misc
    uint32_t block_size; uint32_t x_size; uint32_t T; uint32_t pad0; float eps; float fp8_amax;
    // ---- r7x: indexer/reuse fields (formerly pad1[5]); group 5 stays 64 B
    const float* iqw; const float* ikw; uint16_t* index_q_out; void* index_cache; const int64_t* index_slot_mapping;
};
static_assert(sizeof(SmDev) == 384 && alignof(SmDev) == 8, "SmDev must be 6 x 64 B groups");
static_assert(offsetof(SmDev, rwin_pr3) + sizeof(uint8_t*) == 64, "push set (P,Wscale,Ascale,cwin_loc,rwin_pr0..3) must be exactly group 0");
static_assert(offsetof(SmDev, qw) == 128 && offsetof(SmDev, v_out) == 192 && offsetof(SmDev, kc_sx) == 256 && offsetof(SmDev, block_size) == 320, "group boundaries");
// Kernarg for sm_ran_v1 (r7): the three scalars are the only kernarg the kernel reads before the first vmem load.
struct SmRanArgs { const SmDev* blk; uint32_t M; uint32_t rank; };
static_assert(sizeof(SmRanArgs) == 16, "kernarg = 16 B");
static_assert((SM_F * 2u) % 16u == 0u, "Rx row stride must keep 16 B alignment for the 16 B SC0|SC1 push stores");
static_assert(SM_N == SM_NA * 128u, "N = units * 128");
static_assert(SM_DENSE ? (SM_F == 2304u && SM_NA == 72u) : (SM_F == 2560u && SM_NA == 77u), "F geometry per mode");
