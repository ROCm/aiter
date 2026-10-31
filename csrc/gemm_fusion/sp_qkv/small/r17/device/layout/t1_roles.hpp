// SPDX-License-Identifier: MIT
// SM-T1-r5x static role table (single source of truth for sender and receiver counts). Host-pass safe (constexpr only).
// Remote units per destination: UPD = 19 (indexer: Q16,K,V,IQ) or 18 (reuse: Q16,K,V). 24 push WGs cover 3*UPD units; the indexer adds ONE IK owner WG
// that pushes the byte-identical IK row to all 3 remote destinations, so every (src,dst) flag counts 8 unit pushers + SM_T1_IK_WGS.
#pragma once
#include <stdint.h>
#include "sm_abi_v2.h"
namespace t1r {
constexpr uint32_t PUSH_WGS = SM_T1_PUSH_WGS, HEADS = SM_T1_UPD, UNITS = 3u * SM_T1_UPD, B_PER_PAIR = SM_T1_B_PER_PAIR, IK_WGS = SM_T1_IK_WGS;
constexpr uint32_t g0(uint32_t w) { return (w * UNITS) / PUSH_WGS; }          // push WG w owns remote units [g0(w), g0(w+1))
constexpr bool touches(uint32_t w, uint32_t r) { return g0(w) < (r + 1u) * HEADS && g0(w + 1u) > r * HEADS; }
constexpr uint32_t unit_pushers(uint32_t r) { uint32_t k = 0; for (uint32_t w = 0; w < PUSH_WGS; ++w) if (touches(w, r)) ++k; return k; }
constexpr uint32_t kcount(uint32_t r) { return unit_pushers(r) + IK_WGS; }      // total adds per (src, dst r) per epoch
constexpr bool partition_ok() { for (uint32_t w = 0; w < PUSH_WGS; ++w) if (g0(w + 1u) <= g0(w)) return false; return g0(0) == 0 && g0(PUSH_WGS) == UNITS; }
// independent recount from the unit side
constexpr uint32_t touch_pairs() { uint32_t n = 0; for (uint32_t w = 0; w < PUSH_WGS; ++w) { uint32_t m = 0; for (uint32_t g = g0(w); g < g0(w + 1u); ++g) m |= 1u << (g / HEADS); n += (m & 1u) + ((m >> 1) & 1u) + ((m >> 2) & 1u); } return n; }
static_assert(partition_ok(), "push partition must be non-empty and cover all remote units");
static_assert(unit_pushers(0) + unit_pushers(1) + unit_pushers(2) == touch_pairs(), "K table must match the push partition");
#if !SM_PUSH8
constexpr bool single_dest() { for (uint32_t w = 0; w < PUSH_WGS; ++w) { if (g0(w + 1u) - g0(w) < 2u || g0(w + 1u) - g0(w) > 3u) return false; if (g0(w) / HEADS != (g0(w + 1u) - 1u) / HEADS) return false; } return true; }
static_assert(single_dest(), "every push WG owns 2..3 units of exactly ONE destination");
static_assert(PUSH_WGS == 24 && unit_pushers(0) == 8 && unit_pushers(1) == 8 && unit_pushers(2) == 8, "8 unit pushers per (src,dst) for 24 push WGs");
static_assert(kcount(0) == 8u + SM_T1_IK_WGS && kcount(1) == 8u + SM_T1_IK_WGS && kcount(2) == 8u + SM_T1_IK_WGS, "flag K = 8 unit pushers + IK owner (indexer: 9, reuse: 8)");
static_assert(PUSH_WGS % 8 == 0, "3 push WGs per XCC (XCC = wgid % 8)");
#else
// r15 Q1: 8 push WGs (one per XCC, wgid = XCC), 3*UPD units in contiguous blocks of 6..8 per WG; a WG touches 1 or 2 destinations (never 3: a block is shorter than UPD), K per destination = number of touching WGs (+ IK owner).
constexpr uint32_t units(uint32_t w) { return g0(w + 1u) - g0(w); }
constexpr uint32_t ndest(uint32_t w) { uint32_t n = 0; for (uint32_t r = 0; r < 3u; ++r) if (touches(w, r)) ++n; return n; }
constexpr bool q1_shape_ok() { for (uint32_t w = 0; w < PUSH_WGS; ++w) { if (units(w) < 5u || units(w) > 8u || ndest(w) < 1u || ndest(w) > 2u) return false; } return true; }
constexpr uint32_t total_releases() { uint32_t n = 0; for (uint32_t w = 0; w < PUSH_WGS; ++w) n += ndest(w); return n; }
static_assert(PUSH_WGS == 8 && q1_shape_ok(), "Q1: 8 push WGs of 5..8 units touching 1..2 destinations");
static_assert(unit_pushers(0) + unit_pushers(1) + unit_pushers(2) == total_releases(), "K table == total releases (every touching WG releases once per touched destination)");
static_assert(unit_pushers(0) >= 3u && unit_pushers(1) >= 3u && unit_pushers(2) >= 3u, "every destination gets at least 3 unit pushers");
static_assert(PUSH_WGS == 8, "one push WG per XCC (XCC = wgid % 8)");
#endif
// r17 C-1 (no-credit review): the no-credit WAR argument needs B coverage of EVERY (src, dst) pair in EVERY variant (indexer/reuse/dense, NH1/NH2, PUSH8/HALVES/neither): B WG bi = (r, pair) polls flag[s(r)], r < 3 sources, pair < B_PER_PAIR heads-groups of NH heads each.
static_assert(SM_T1_HEADS_SRC % SM_T1_B_HEADS == 0u && B_PER_PAIR >= 1u, "C-1: heads per source must split evenly over the B WGs of a pair");
constexpr bool b_cover_ok() { uint32_t seen[24] = {}; for (uint32_t pair = 0; pair < B_PER_PAIR; ++pair) for (uint32_t k = 0; k < SM_T1_B_HEADS; ++k) { const uint32_t h = pair * SM_T1_B_HEADS + k; if (h >= SM_T1_HEADS_SRC) return false; ++seen[h]; } for (uint32_t h = 0; h < SM_T1_HEADS_SRC; ++h) if (seen[h] != 1u) return false; return true; }
static_assert(b_cover_ok(), "C-1: the B WGs of one (src, dst) pair cover every head the source sends exactly once (and each acquires flag[src])");
static_assert(SM_T1_B_WGS == 3u * B_PER_PAIR, "C-1: 3 sources x B_PER_PAIR B WGs: every (src, dst) pair has B coverage");
static_assert(B_PER_PAIR * SM_T1_B_HEADS == SM_T1_HEADS_SRC, "B WGs per (src,dst) cover every head the source sends (indexer: 20 incl. IK; reuse: 18)");
static_assert(SM_T1_GRID == SM_T1_PUSH_WGS + SM_T1_IK_WGS + SM_T1_UPD + 3u * B_PER_PAIR, "grid = push + IK + self + 3*B_PER_PAIR");
static_assert(HEADS == 19 || HEADS == 18, "mode geometry");
} // namespace t1r