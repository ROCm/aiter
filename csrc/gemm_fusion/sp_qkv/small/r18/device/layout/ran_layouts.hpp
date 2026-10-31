// SPDX-License-Identifier: MIT
// SM-SK2-RAN-r1 RAN addressing as Opus layouts (design §4): a 128-column unit row is 16 adjacent lanes x
// 16 B (8 bf16) = 256 B contiguous in Rx (and 16 lanes x 8 fp32 = 512 B contiguous reads in P per row).
// UNCOMPILED. Used for offsets in ran_kernel.cu (P read, Rx write/read). NOT retrofitted (known deviation):
// control-window/flag addressing, K/V cache scatter (strides come from the golden tensors at runtime),
// and the Q/K/V output pointers remain pointer arithmetic.
#pragma once
#include <opus/opus.hpp>
#include "sm_abi_v2.h"

namespace sml {
constexpr int kF = static_cast<int>(SM_F), kN = static_cast<int>(SM_N);
// Rx window: [par=2][src=4][Mp=32][F=2560] bf16 (element offsets)
OPUS_D auto rx_view() { return opus::make_layout<-1>(opus::make_tuple(2, 4, 32, kF), opus::make_tuple(4 * 32 * kF, 32 * kF, kF, 1)); }
// Partials: [SK][Mp=32][N] fp32 (SK = SM_SK, 2 or 3)
OPUS_D auto p_view() { return opus::make_layout<-1>(opus::make_tuple(2, 32, kN), opus::make_tuple(32 * kN, kN, 1)); }
} // namespace sml

// r18b X-5: out-of-range buffer loads silently return 0, so every Rx access (bypass loads use the SM_RX_BYTES buffer size) must stay inside the window: the largest byte touched is
// 2 * rx_view(par=1, src=3, row=MP-1, col=F-8) + 16 (one 16 B vector per lane at the last 8-column group). Mirrors the layout formula above with plain integers (host-safe).
namespace rxchk {
// rx_view strides: par * (4*32*F) + src * (32*F) + row * F + col (elements, bf16). Largest access: par = 1, src = 3, row = MP-1, col = F-8 (one 16 B vector = 8 bf16 per lane).
constexpr unsigned long long kMaxElem = 1ull * (4ull * 32ull * SM_F) + 3ull * (32ull * SM_F) + (SM_MP - 1ull) * SM_F + (SM_F - 8ull);
constexpr unsigned long long kMaxByte = 2ull * kMaxElem + 16ull;
static_assert(SM_MP == 32u, "rx_view is written for MP == 32");
static_assert(kMaxByte <= SM_RX_BYTES, "X-5: the highest Rx vector access (par 1, src 3, row MP-1, col F-8) must fit in SM_RX_BYTES (out-of-range buffer loads silently return 0)");
}
