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
constexpr int kF = static_cast<int>(SM_F), kN = static_cast<int>(SM_N), kMP = static_cast<int>(SM_MP);
// Rx window: [par=2][src=4][Mp=SM_MP=64][F] bf16 (element offsets)
OPUS_D auto rx_view() { return opus::make_layout<-1>(opus::make_tuple(2, 4, kMP, kF), opus::make_tuple(4 * kMP * kF, kMP * kF, kF, 1)); }
// Partials: [SK][Mp=SM_MP=64][N] fp32 (SK = SM_SK, 2 or 3)
OPUS_D auto p_view() { return opus::make_layout<-1>(opus::make_tuple(2, kMP, kN), opus::make_tuple(kMP * kN, kN, 1)); }
} // namespace sml

// r18b X-5: out-of-range buffer loads silently return 0, so every Rx access (bypass loads use the SM_RX_BYTES buffer size) must stay inside the window: the largest byte touched is
// 2 * rx_view(par=1, src=3, row=MP-1, col=F-8) + 16 (one 16 B vector per lane at the last 8-column group). Mirrors the layout formula above with plain integers (host-safe).
namespace rxchk {
// rx_view strides: par * (4*MP*F) + src * (MP*F) + row * F + col (elements, bf16). Largest access: par = 1, src = 3, row = MP-1, col = F-8 (one 16 B vector = 8 bf16 per lane).
constexpr unsigned long long kMaxElem = 1ull * (4ull * SM_MP * SM_F) + 3ull * (1ull * SM_MP * SM_F) + (SM_MP - 1ull) * SM_F + (SM_F - 8ull);
constexpr unsigned long long kMaxByte = 2ull * kMaxElem + 16ull;
static_assert(SM_MP == 128u, "r21 layouts are written for MP == 128");
static_assert(SM_RX_BYTES == 2ull * 4ull * SM_MP * SM_F * 2ull, "window bytes = [par 2][src 4][MP][F] bf16");
static_assert(SM_RX_BYTES > 4ull * 1024ull * 1024ull && SM_RX_BYTES <= 8ull * 1024ull * 1024ull, "R13 MP128: the Rx window (dense 4,718,592 B = 4.5 MiB, indexer/reuse 5,242,880 B = 5.0 MiB) needs the 8 MiB arena view (the 4 MiB view of MP64 is too small)");
static_assert(SM_DENSE ? (SM_RX_BYTES == 4718592ull) : (SM_RX_BYTES == 5242880ull), "X-5 recomputed for MP = 128");
static_assert(kMaxByte <= SM_RX_BYTES, "X-5: the highest Rx vector access (par 1, src 3, row MP-1, col F-8) must fit in SM_RX_BYTES (out-of-range buffer loads silently return 0)");
}
