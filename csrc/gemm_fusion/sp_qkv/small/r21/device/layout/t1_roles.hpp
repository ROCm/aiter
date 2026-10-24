// SPDX-License-Identifier: MIT
// SM-T1 static role table, r21 (MP128). Single source of truth for sender and receiver counts. Host-pass safe (constexpr only).
// Remote units per destination: UPD = 19 (indexer: Q16,K,V,IQ) or 18 (dense/reuse: Q16,K,V). UNIT_WGS = 24 unit WGs cover 3*UPD units (2..3 units of ONE destination each); the push role has ROWGROUPS = NPASS/4 row groups of them
// (WG w = unit WG w % 24, row group w / 24), so every (src,dst) pair is pushed by 8 * ROWGROUPS unit pushers. The indexer adds IK_WGS = NPASS/2 IK owners that each push their rows of the IK head to all 3 destinations:
// every (src,dst) flag counts 8 * ROWGROUPS + IK_WGS adds per epoch (dense/reuse 16, indexer 20).
#pragma once
#include <stdint.h>
#include "sm_abi_v3.h"
namespace t1r {
constexpr uint32_t PUSH_WGS = SM_T1_PUSH_WGS, UNIT_WGS = 24u, ROWGROUPS = SM_T1_PUSH_WGS / 24u, HEADS = SM_T1_UPD, UNITS = 3u * SM_T1_UPD, B_PER_PAIR = SM_T1_B_PER_PAIR, IK_WGS = SM_T1_IK_WGS;
constexpr uint32_t u0(uint32_t wu) { return (wu * UNITS) / UNIT_WGS; }          // unit WG wu owns remote units [u0(wu), u0(wu+1))
constexpr uint32_t g0(uint32_t w) { return u0(w % UNIT_WGS); }                  // push WG w (any row group) owns the units of its unit WG
constexpr bool touches(uint32_t w, uint32_t r) { return u0(w % UNIT_WGS) < (r + 1u) * HEADS && u0(w % UNIT_WGS + 1u) > r * HEADS; }
constexpr uint32_t unit_pushers(uint32_t r) { uint32_t k = 0; for (uint32_t w = 0; w < PUSH_WGS; ++w) if (touches(w, r)) ++k; return k; }
constexpr uint32_t kcount(uint32_t r) { return unit_pushers(r) + IK_WGS; }      // total adds per (src, dst r) per epoch
constexpr bool partition_ok() { for (uint32_t wu = 0; wu < UNIT_WGS; ++wu) if (u0(wu + 1u) <= u0(wu)) return false; return u0(0) == 0 && u0(UNIT_WGS) == UNITS; }
// independent recount from the unit side: every (unit WG, destination) pair it touches, times the row groups
constexpr uint32_t touch_pairs() { uint32_t n = 0; for (uint32_t wu = 0; wu < UNIT_WGS; ++wu) { uint32_t m = 0; for (uint32_t g = u0(wu); g < u0(wu + 1u); ++g) m |= 1u << (g / HEADS); n += (m & 1u) + ((m >> 1) & 1u) + ((m >> 2) & 1u); } return n * ROWGROUPS; }
static_assert(partition_ok(), "push partition must be non-empty and cover all remote units");
static_assert(unit_pushers(0) + unit_pushers(1) + unit_pushers(2) == touch_pairs(), "K table must match the push partition");
constexpr bool single_dest() { for (uint32_t wu = 0; wu < UNIT_WGS; ++wu) { if (u0(wu + 1u) - u0(wu) < 2u || u0(wu + 1u) - u0(wu) > 3u) return false; if (u0(wu) / HEADS != (u0(wu + 1u) - 1u) / HEADS) return false; } return true; }
static_assert(single_dest(), "every unit WG owns 2..3 units of exactly ONE destination");
static_assert(ROWGROUPS * 4u == SM_NPASS && PUSH_WGS == UNIT_WGS * ROWGROUPS, "push role: NPASS/4 row groups of 24 unit WGs (each WG = 2 halves x 2 passes)");
static_assert(unit_pushers(0) == 8u * ROWGROUPS && unit_pushers(1) == 8u * ROWGROUPS && unit_pushers(2) == 8u * ROWGROUPS, "8 * ROWGROUPS unit pushers per (src,dst)");
static_assert(kcount(0) == 8u * ROWGROUPS + SM_T1_IK_WGS && kcount(1) == 8u * ROWGROUPS + SM_T1_IK_WGS && kcount(2) == 8u * ROWGROUPS + SM_T1_IK_WGS, "flag K = 8 * ROWGROUPS unit pushers + IK_WGS");
static_assert(PUSH_WGS % 8 == 0, "PUSH_WGS / 8 push WGs per XCC (XCC = wgid % 8)");
// C-1 (no-credit review): the no-credit WAR argument needs B coverage of EVERY (src, dst) pair: B WG bi = (r, pair) polls flag[s(r)]; pair = hp * SM_B_SPLIT + rs (head group hp, row pass rs); every (head, row pass) is covered exactly once per
// (src, dst) pair, so every pair has B WGs that acquire flag[src] every epoch (dead passes included).
static_assert(SM_T1_HEADS_SRC % SM_T1_B_HEADS == 0u && B_PER_PAIR >= 1u && B_PER_PAIR % SM_B_SPLIT == 0u, "C-1: heads per source must split evenly over the B WGs of a pair");
constexpr bool b_cover_ok() { uint32_t seen[24][SM_B_SPLIT] = {}; for (uint32_t pair = 0; pair < B_PER_PAIR; ++pair) { const uint32_t hp = pair / SM_B_SPLIT, rs = pair % SM_B_SPLIT; for (uint32_t k = 0; k < SM_T1_B_HEADS; ++k) { const uint32_t h = hp * SM_T1_B_HEADS + k; if (h >= SM_T1_HEADS_SRC) return false; ++seen[h][rs]; } }
    for (uint32_t h = 0; h < SM_T1_HEADS_SRC; ++h) for (uint32_t rs = 0; rs < SM_B_SPLIT; ++rs) if (seen[h][rs] != 1u) return false; return true; }
static_assert(b_cover_ok(), "C-1: the B WGs of one (src, dst) pair cover every (head, row pass) exactly once (and each acquires flag[src])");
static_assert(SM_T1_B_WGS == 3u * B_PER_PAIR, "C-1: 3 sources x B_PER_PAIR B WGs: every (src, dst) pair has B coverage");
static_assert(B_PER_PAIR * SM_T1_B_HEADS == SM_T1_HEADS_SRC * SM_B_SPLIT, "B WGs per (src,dst) cover every head the source sends (indexer: 20 incl. IK; reuse: 18) x row passes");
static_assert(SM_T1_GRID == SM_T1_PUSH_WGS + SM_T1_IK_WGS + SM_T1_SELF_WGS + 3u * B_PER_PAIR, "grid = push + IK + self + 3*B_PER_PAIR");
static_assert(HEADS == 19 || HEADS == 18, "mode geometry");
// ---- r21 (MP128) role table checks ----
static_assert(SM_NPASS == 8u && SM_MP == 128u, "r21 is the MP128 build: 8 row passes of 16 rows");
static_assert(SM_B_SPLIT == SM_NPASS, "B WGs: one row pass each (dead passes still acquire flag[src])");
static_assert(SM_SELF_SPLIT * 2u == SM_NPASS, "self WGs: 2 passes each, NPASS/2 row groups");
static_assert(IK_WGS == 0u || IK_WGS * 2u == SM_NPASS, "IK owners (indexer): 2 passes each => IK_WGS == NPASS/2; dense/reuse have none");
static_assert(SM_R5X_INDEXER ? (IK_WGS == 4u && kcount(0) == 20u && kcount(1) == 20u && kcount(2) == 20u) : (IK_WGS == 0u && kcount(0) == 16u && kcount(1) == 16u && kcount(2) == 16u), "D-9 at MP128: K = 16 unit pushers + IK_WGS (indexer 20, dense/reuse 16) for EVERY (src, dst) pair");
static_assert(SM_T1_GRID <= SM_DBG_RAN_MAXWG, "grid fits the stamp records");
static_assert(SM_T1_GRID == (SM_R5X_INDEXER ? 608u : 552u) || SM_T1_B_HEADS != 1u, "NH1 grids: dense/reuse 552, indexer 608");
} // namespace t1r
