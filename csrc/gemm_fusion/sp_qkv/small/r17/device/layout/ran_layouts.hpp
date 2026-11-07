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
