import sys

r = sys.argv[1]


def rd(p):
    with open(p, encoding="utf-8", newline="") as _f:
        t = _f.read()
    nl = "\r\n" if "\r\n" in t else "\n"
    return t.replace("\r\n", "\n"), nl


def wr(p, t, nl):
    with open(p, "w", encoding="utf-8", newline="") as _f:
        _f.write(t.replace("\n", nl))


def rep(t, old, new, cnt=1):
    assert t.count(old) == cnt, (f"count {t.count(old):d} != {cnt:d}: ") + old[:100]
    return t.replace(old, new)


# ================= ABI =================
p = r + "/interface/sm_abi_v3.h"
t, nl = rd(p)
t = rep(
    t,
    "#define SM_RAN_THREADS      512u\n",
    """// r22 (tuned MP128; R14 F1) option switches, each default ON once correct (old behaviour selectable for the A/B), byte-identical outputs to r21:
//  SM_B_QUAD (1): B WG = (src, head, 4-pass group) on all 8 waves (halves = 2 passes each, one round trip): B_SPLIT = NPASS/4 = 2 => dense 228 / indexer 248 RAN WGs (<= 256 resident at 1 WG/CU). 0 = r21 (one pass per B WG, 552 / 608).
//  SM_GEMM_BN128 (1): GEMM BM128 x BN128, SK3, PFK4 (32 KiB stage, 128 KiB LDS, 1 WG/CU), 72 / 77 N-tiles x 3 splits = 216 / 231 WGs; each output keeps the r18 MFMA sequence => P byte-identical. 0 = r21 (BN64, PFK3, 2 WG/CU).
//  SM_RX3 (1): Rx indexed by REMOTE ORDINAL (src < dst ? src : src - 1): 3 slots, the never-used self slot is dropped: Rx [2][3][MP][F] = 3,538,944 B dense / 3,932,160 B indexer+reuse (fits the 4 MiB view). 0 = 4 slots.
//  SM_PUSH_PAR (0, behind the D-9 recount): 96 push WGs x 256 threads (the 2 halves become separate WGs) with per-(XCC, destination) release AGGREGATION (agent-scope counter; the last arriver does the single system release): <= 3 releases per XCC, K = 8 + IK_WGS.
#ifndef SM_B_QUAD
#define SM_B_QUAD 1
#endif
#ifndef SM_GEMM_BN128
#define SM_GEMM_BN128 1
#endif
#ifndef SM_RX3
#define SM_RX3 1
#endif
#ifndef SM_PUSH_PAR
#define SM_PUSH_PAR 0
#endif
#define SM_GEMM_BN          (SM_GEMM_BN128 ? 128u : 64u)
#define SM_RX_SLOTS         (SM_RX3 ? 3u : 4u)
#define SM_RAN_THREADS      512u
""",
)
t = rep(
    t,
    "#define SM_T1_PUSH_WGS      (24u * (SM_NPASS / 4u))",
    "#define SM_T1_PUSH_WGS      (24u * (SM_NPASS / 4u) * (SM_PUSH_PAR ? 2u : 1u))",
)
t = rep(
    t,
    "#define SM_B_SPLIT SM_NPASS            // r20: one B WG = one head-group x ONE 16-row pass (dead passes still acquire)",
    "#define SM_B_SPLIT (SM_B_QUAD ? SM_NPASS / 4u : SM_NPASS)   // r22: B quad = 4 passes per WG (2 per 256-thread half); r21: one 16-row pass per B WG. Dead passes / dead quads still acquire.",
)
t = rep(
    t,
    "#define SM_RX_BYTES        (2u*4u*SM_MP*SM_F*2u)",
    "#define SM_RX_BYTES        (2u*SM_RX_SLOTS*SM_MP*SM_F*2u)",
)
t = rep(
    t,
    "#define SM_GEMM_NTILES      ((SM_R5X_GEMM_SKIP_IDX) ? 144u : (SM_N / 64u))",
    "#define SM_GEMM_NTILES      ((((SM_R5X_GEMM_SKIP_IDX) ? 9216u : SM_N)) / SM_GEMM_BN)   // r22: N-tiles of BN columns (BN128: dense/reuse-no-idx 72, indexer/reuse 77)",
)
t = rep(
    t,
    "#define SM_C_BYTES   (SM_C_FLAGS + 8u*2u*4u*SM_NA)",
    "#define SM_C_AGG(wu) (SM_C_FLAGS + 384u + 8u*(wu))   // r22 SM_PUSH_PAR: u64 release-aggregation counter per unit WG (24 words = bytes 640-831), zeroed with the control window by reset()\n#define SM_C_BYTES   (SM_C_FLAGS + 8u*2u*4u*SM_NA)",
)
wr(p, t, nl)
# ================= roles =================
with open(r + "/device/layout/t1_roles.hpp", "w", encoding="utf-8", newline="") as _f:
    _f.write("""// SPDX-License-Identifier: MIT
// SM-T1 static role table, r22 (tuned MP128). Single source of truth for sender and receiver counts. Host-pass safe (constexpr only).
// Remote units per destination: UPD = 19 (indexer: Q16,K,V,IQ) or 18 (dense/reuse: Q16,K,V). UNIT_WGS = 24 unit WGs cover 3*UPD units (2..3 units of ONE destination each). The push role has ROWGROUPS = NPASS/4 row groups of them
// (WG w: unit WG w % 24, row group (w / 24) % ROWGROUPS) and, with SM_PUSH_PAR, 2 "halves" (w / (24 * ROWGROUPS)) as separate 256-thread WGs. Default (PAR = 0): every (src,dst) pair is pushed by 8 * ROWGROUPS unit pushers that each
// release once. With PAR = 1 the AGG_N = ROWGROUPS * 2 WGs that share a unit WG (same XCC, same destination) release ONCE through an aggregation counter => 8 releasers per (src,dst).
// The indexer adds IK_WGS = NPASS/2 IK owners that each push their rows of the IK head to all 3 destinations: every (src,dst) flag counts kcount = releasers + IK_WGS adds per epoch.
#pragma once
#include <stdint.h>
#include "sm_abi_v3.h"
namespace t1r {
constexpr uint32_t PAR = SM_PUSH_PAR ? 2u : 1u, UNIT_WGS = 24u, ROWGROUPS = SM_NPASS / 4u, PUSH_WGS = SM_T1_PUSH_WGS, HEADS = SM_T1_UPD, UNITS = 3u * SM_T1_UPD, B_PER_PAIR = SM_T1_B_PER_PAIR, IK_WGS = SM_T1_IK_WGS;
constexpr uint32_t AGG_N = SM_PUSH_PAR ? ROWGROUPS * PAR : 1u;                  // push WGs per aggregated release (same unit WG)
constexpr uint32_t u0(uint32_t wu) { return (wu * UNITS) / UNIT_WGS; }          // unit WG wu owns remote units [u0(wu), u0(wu+1))
constexpr uint32_t g0(uint32_t w) { return u0(w % UNIT_WGS); }                  // push WG w (any row group / half) owns the units of its unit WG
constexpr bool touches(uint32_t w, uint32_t r) { return u0(w % UNIT_WGS) < (r + 1u) * HEADS && u0(w % UNIT_WGS + 1u) > r * HEADS; }
constexpr uint32_t unit_pushers(uint32_t r) { uint32_t k = 0; for (uint32_t w = 0; w < PUSH_WGS; ++w) if (touches(w, r)) ++k; return k; }   // push WGs touching destination r
constexpr uint32_t releasers(uint32_t r) { return unit_pushers(r) / AGG_N; }                                                            // releases per epoch toward r from the push role
constexpr uint32_t kcount(uint32_t r) { return releasers(r) + IK_WGS; }                                                                 // total adds per (src, dst r) per epoch
constexpr bool partition_ok() { for (uint32_t wu = 0; wu < UNIT_WGS; ++wu) if (u0(wu + 1u) <= u0(wu)) return false; return u0(0) == 0 && u0(UNIT_WGS) == UNITS; }
constexpr uint32_t touch_pairs() { uint32_t n = 0; for (uint32_t wu = 0; wu < UNIT_WGS; ++wu) { uint32_t m = 0; for (uint32_t g = u0(wu); g < u0(wu + 1u); ++g) m |= 1u << (g / HEADS); n += (m & 1u) + ((m >> 1) & 1u) + ((m >> 2) & 1u); } return n * ROWGROUPS * PAR; }
static_assert(partition_ok(), "push partition must be non-empty and cover all remote units");
static_assert(unit_pushers(0) + unit_pushers(1) + unit_pushers(2) == touch_pairs(), "K table must match the push partition");
constexpr bool single_dest() { for (uint32_t wu = 0; wu < UNIT_WGS; ++wu) { if (u0(wu + 1u) - u0(wu) < 2u || u0(wu + 1u) - u0(wu) > 3u) return false; if (u0(wu) / HEADS != (u0(wu + 1u) - 1u) / HEADS) return false; } return true; }
static_assert(single_dest(), "every unit WG owns 2..3 units of exactly ONE destination");
static_assert(PUSH_WGS == UNIT_WGS * ROWGROUPS * PAR && ROWGROUPS * 4u == SM_NPASS, "push role: NPASS/4 row groups of 24 unit WGs (x2 with SM_PUSH_PAR: halves as separate WGs)");
static_assert(unit_pushers(0) == 8u * ROWGROUPS * PAR && unit_pushers(1) == 8u * ROWGROUPS * PAR && unit_pushers(2) == 8u * ROWGROUPS * PAR, "8 * ROWGROUPS * PAR push WGs per (src,dst)");
static_assert(releasers(0) == (SM_PUSH_PAR ? 8u : 8u * ROWGROUPS) && releasers(1) == releasers(0) && releasers(2) == releasers(0), "releasers per (src,dst): 16 (default) / 8 (SM_PUSH_PAR aggregation)");
static_assert(kcount(0) == releasers(0) + SM_T1_IK_WGS && kcount(1) == kcount(0) && kcount(2) == kcount(0), "flag K = releasers + IK_WGS");
static_assert(PUSH_WGS % 8 == 0 && UNIT_WGS % 8 == 0, "PUSH_WGS / 8 push WGs per XCC (XCC = wgid % 8); the WGs of one unit WG share an XCC (w % 24 and the group stride 24 are multiples of 8)");
// C-1 (no-credit review): the no-credit WAR argument needs B coverage of EVERY (src, dst) pair: B WG bi = (r, pair) polls flag[s(r)]; pair = hp * SM_B_SPLIT + rs (head group hp, row pass or 4-pass quad rs); every (head, rs) is covered exactly
// once per (src, dst) pair, so every pair has B WGs that acquire flag[src] every epoch (dead passes / dead quads included).
static_assert(SM_T1_HEADS_SRC % SM_T1_B_HEADS == 0u && B_PER_PAIR >= 1u && B_PER_PAIR % SM_B_SPLIT == 0u, "C-1: heads per source must split evenly over the B WGs of a pair");
constexpr bool b_cover_ok() { uint32_t seen[24][SM_B_SPLIT] = {}; for (uint32_t pair = 0; pair < B_PER_PAIR; ++pair) { const uint32_t hp = pair / SM_B_SPLIT, rs = pair % SM_B_SPLIT; for (uint32_t k = 0; k < SM_T1_B_HEADS; ++k) { const uint32_t h = hp * SM_T1_B_HEADS + k; if (h >= SM_T1_HEADS_SRC) return false; ++seen[h][rs]; } }
    for (uint32_t h = 0; h < SM_T1_HEADS_SRC; ++h) for (uint32_t rs = 0; rs < SM_B_SPLIT; ++rs) if (seen[h][rs] != 1u) return false; return true; }
static_assert(b_cover_ok(), "C-1: the B WGs of one (src, dst) pair cover every (head, pass-or-quad) exactly once (and each acquires flag[src])");
static_assert(SM_T1_B_WGS == 3u * B_PER_PAIR, "C-1: 3 sources x B_PER_PAIR B WGs: every (src, dst) pair has B coverage");
static_assert(B_PER_PAIR * SM_T1_B_HEADS == SM_T1_HEADS_SRC * SM_B_SPLIT, "B WGs per (src,dst) cover every head the source sends x B_SPLIT");
static_assert(SM_T1_GRID == SM_T1_PUSH_WGS + SM_T1_IK_WGS + SM_T1_SELF_WGS + 3u * B_PER_PAIR, "grid = push + IK + self + 3*B_PER_PAIR");
static_assert(HEADS == 19 || HEADS == 18, "mode geometry");
// ---- r22 (tuned MP128) role table checks ----
static_assert(SM_NPASS == 8u && SM_MP == 128u, "r22 is the MP128 build: 8 row passes of 16 rows");
static_assert(SM_B_SPLIT == (SM_B_QUAD ? SM_NPASS / 4u : SM_NPASS), "B WGs: one 4-pass quad (SM_B_QUAD) or one pass each; dead ones still acquire flag[src]");
static_assert(SM_SELF_SPLIT * 2u == SM_NPASS, "self WGs: 2 passes each, NPASS/2 row groups");
static_assert(IK_WGS == 0u || IK_WGS * 2u == SM_NPASS, "IK owners (indexer): 2 passes each => IK_WGS == NPASS/2; dense/reuse have none");
static_assert(SM_R5X_INDEXER ? (IK_WGS == 4u) : (IK_WGS == 0u), "IK owners only in the indexer build");
static_assert(SM_T1_GRID <= SM_DBG_RAN_MAXWG, "grid fits the stamp records");
static_assert(SM_T1_B_HEADS != 1u || SM_T1_GRID == SM_T1_PUSH_WGS + SM_T1_IK_WGS + SM_T1_SELF_WGS + (SM_B_QUAD ? 3u * SM_T1_HEADS_SRC * 2u : 3u * SM_T1_HEADS_SRC * 8u), "NH1 grid sums");
static_assert(!(SM_B_QUAD && !SM_PUSH_PAR && SM_T1_B_HEADS == 1u) || SM_T1_GRID == (SM_R5X_INDEXER ? 248u : 228u), "R14 F1 (1): B-quad grids 228 (dense/reuse) / 248 (indexer)");
static_assert(!(SM_B_QUAD && !SM_PUSH_PAR && SM_T1_B_HEADS == 1u) || SM_T1_GRID <= 256u, "R14 F1 (1): the RAN grid fits residency at 1 WG/CU (256 CUs)");
} // namespace t1r
""")
# ================= layouts =================
p = r + "/device/layout/ran_layouts.hpp"
t, nl = rd(p)
t = rep(
    t,
    "constexpr int kF = static_cast<int>(SM_F), kN = static_cast<int>(SM_N), kMP = static_cast<int>(SM_MP);",
    "constexpr int kF = static_cast<int>(SM_F), kN = static_cast<int>(SM_N), kMP = static_cast<int>(SM_MP), kRxS = static_cast<int>(SM_RX_SLOTS);",
)
t = (
    rep(
        t,
        "// Rx window: [par=2][src=4][Mp=SM_MP=64][F] bf16 (element offsets)",
        "// Rx window: [par=2][slot=SM_RX_SLOTS (3 = remote ordinal, 4 = source rank)][Mp=SM_MP=128][F] bf16 (element offsets)",
    )
    if "// Rx window: [par=2][src=4][Mp=SM_MP=64][F] bf16 (element offsets)" in t
    else t
)
t = rep(
    t,
    "opus::make_tuple(2, 4, kMP, kF), opus::make_tuple(4 * kMP * kF, kMP * kF, kF, 1)",
    "opus::make_tuple(2, kRxS, kMP, kF), opus::make_tuple(kRxS * kMP * kF, kMP * kF, kF, 1)",
)
t = rep(
    t,
    "1ull * (4ull * SM_MP * SM_F) + 3ull * (1ull * SM_MP * SM_F)",
    "1ull * (1ull * SM_RX_SLOTS * SM_MP * SM_F) + (SM_RX_SLOTS - 1ull) * (1ull * SM_MP * SM_F)",
)
t = rep(
    t,
    'static_assert(SM_RX_BYTES == 2ull * 4ull * SM_MP * SM_F * 2ull, "window bytes = [par 2][src 4][MP][F] bf16");',
    'static_assert(SM_RX_BYTES == 2ull * SM_RX_SLOTS * SM_MP * SM_F * 2ull, "window bytes = [par 2][slot SM_RX_SLOTS][MP][F] bf16");',
)
a = t.index("static_assert(SM_RX_BYTES > 4ull * 1024ull * 1024ull")
b = t.index("\n", t.index("static_assert(SM_DENSE ? (SM_RX_BYTES == 4718592ull)"))
t = (
    t[:a]
    + """static_assert(SM_RX_BYTES <= 8ull * 1024ull * 1024ull && (SM_RX3 ? SM_RX_BYTES <= 4ull * 1024ull * 1024ull : SM_RX_BYTES > 4ull * 1024ull * 1024ull), "R14 F1 (4): with 3 Rx slots the MP128 window fits the 4 MiB view (dense 3,538,944 B, indexer/reuse 3,932,160 B); with 4 slots it needs the 8 MiB view");
static_assert(SM_RX3 ? (SM_DENSE ? (SM_RX_BYTES == 3538944ull) : (SM_RX_BYTES == 3932160ull)) : (SM_DENSE ? (SM_RX_BYTES == 4718592ull) : (SM_RX_BYTES == 5242880ull)), "X-5 recomputed for the Rx slot count");"""
    + t[b:]
)
wr(p, t, nl)
print("abi+roles+layouts ok")
