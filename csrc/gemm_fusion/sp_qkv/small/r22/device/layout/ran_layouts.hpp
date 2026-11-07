// SPDX-License-Identifier: MIT
// SM-SK2-RAN-r1 RAN addressing as Opus layouts (design §4): a 128-column unit row is 16 adjacent lanes x
// 16 B (8 bf16) = 256 B contiguous in Rx (and 16 lanes x 8 fp32 = 512 B contiguous reads in P per row).
// UNCOMPILED. Used for offsets in ran_kernel.cu (P read, Rx write/read). NOT retrofitted (known deviation):
// control-window/flag addressing, K/V cache scatter (strides come from the golden tensors at runtime),
// and the Q/K/V output pointers remain pointer arithmetic.
#pragma once
#include <opus/opus.hpp>
#include "sm_abi_v3.h"

namespace sml {
constexpr int kF = static_cast<int>(SM_F), kN = static_cast<int>(SM_N), kMP = static_cast<int>(SM_MP), kRxS = static_cast<int>(SM_RX_SLOTS);
// Rx window: [par=2][slot=SM_RX_SLOTS (3 = remote ordinal, 4 = source rank)][Mp=SM_MP=128][F] bf16 (element offsets)
OPUS_D auto rx_view() { return opus::make_layout<-1>(opus::make_tuple(2, kRxS, kMP, kF), opus::make_tuple(kRxS * kMP * kF, kMP * kF, kF, 1)); }
// Partials: [SK][Mp=SM_MP=64][N] fp32 (SK = SM_SK, 2 or 3)
OPUS_D auto p_view() { return opus::make_layout<-1>(opus::make_tuple(2, kMP, kN), opus::make_tuple(kMP * kN, kN, 1)); }
} // namespace sml

// r18b X-5: out-of-range buffer loads silently return 0, so every Rx access (bypass loads use the SM_RX_BYTES buffer size) must stay inside the window: the largest byte touched is
// 2 * rx_view(par=1, src=3, row=MP-1, col=F-8) + 16 (one 16 B vector per lane at the last 8-column group). Mirrors the layout formula above with plain integers (host-safe).
namespace rxchk {
// rx_view strides: par * (4*MP*F) + src * (MP*F) + row * F + col (elements, bf16). Largest access: par = 1, src = 3, row = MP-1, col = F-8 (one 16 B vector = 8 bf16 per lane).
constexpr unsigned long long kMaxElem = 1ull * (1ull * SM_RX_SLOTS * SM_MP * SM_F) + (SM_RX_SLOTS - 1ull) * (1ull * SM_MP * SM_F) + (SM_MP - 1ull) * SM_F + (SM_F - 8ull);
constexpr unsigned long long kMaxByte = 2ull * kMaxElem + 16ull;
static_assert(SM_MP == 128u, "r21 layouts are written for MP == 128");
static_assert(SM_RX_BYTES == 2ull * SM_RX_SLOTS * SM_MP * SM_F * 2ull, "window bytes = [par 2][slot SM_RX_SLOTS][MP][F] bf16");
static_assert(SM_RX_BYTES <= 8ull * 1024ull * 1024ull && (SM_RX3 ? SM_RX_BYTES <= 4ull * 1024ull * 1024ull : SM_RX_BYTES > 4ull * 1024ull * 1024ull), "R14 F1 (4): with 3 Rx slots the MP128 window fits the 4 MiB view (dense 3,538,944 B, indexer/reuse 3,932,160 B); with 4 slots it needs the 8 MiB view");
static_assert(SM_RX3 ? (SM_DENSE ? (SM_RX_BYTES == 3538944ull) : (SM_RX_BYTES == 3932160ull)) : (SM_DENSE ? (SM_RX_BYTES == 4718592ull) : (SM_RX_BYTES == 5242880ull)), "X-5 recomputed for the Rx slot count");
static_assert(kMaxByte <= SM_RX_BYTES, "X-5: the highest Rx vector access (par 1, src 3, row MP-1, col F-8) must fit in SM_RX_BYTES (out-of-range buffer loads silently return 0)");
// review C-1: the ordinal map must be a bijection {s != d} -> {0, 1, 2} for EVERY destination d (12 ordered pairs); mirrors rx_ord() in ran_kernel.cu (RX3) and the identity for 4 slots.
constexpr unsigned rxord(unsigned s, unsigned d) { return SM_RX3 ? (s < d ? s : s - 1u) : s; }
constexpr bool rxord_bijective() {
    for (unsigned d = 0; d < 4u; ++d) { unsigned seen = 0u; for (unsigned s = 0; s < 4u; ++s) { if (s == d) continue; const unsigned o = rxord(s, d); if (o >= SM_RX_SLOTS || (seen & (1u << o))) return false; seen |= 1u << o; } if (SM_RX3 && seen != 7u) return false; }
    return true;
}
static_assert(rxord_bijective(), "C-1: rx_ord(s, d) is injective into the Rx slots for all 12 (s, d) pairs (RX3: a bijection onto {0,1,2})");
}
