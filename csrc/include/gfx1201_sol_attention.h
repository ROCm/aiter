// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include <cstdint>

// gfx1201 Sol sparse attention (MiniMax-H3): routing and Phase-A state for the list-driven ASM core
// (hsa/gfx1201/sage_attention/attention_sol.hsaco). Pointers are device addresses, `stream` a hipStream_t.
void launch_gfx1201_sol_route(int64_t Q, int64_t QS, int64_t K, int64_t KS, int64_t V, int64_t VS, int64_t kbar,
                              int64_t vsum, int64_t kb, int64_t kb_scale, int64_t vb, int64_t vb_scale, int64_t mu,
                              int64_t var, int64_t list, int64_t count, int64_t mask, int64_t Sp, int64_t valid,
                              int64_t H, int64_t prefix, double tau, int64_t stream);
void launch_gfx1201_sol_phase_a(int64_t Q, int64_t QScale, int64_t VScale, int64_t KB, int64_t VB, int64_t KBScale,
                                int64_t VBScale, int64_t Cnt, int64_t Mask, int64_t State, int64_t padded_seq_len,
                                int64_t valid_seq_len, int64_t num_heads, int64_t nkb, int64_t nkbp, int64_t n_wg,
                                int64_t stream);
