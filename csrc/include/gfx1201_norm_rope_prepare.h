// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include <cstdint>

// Fused QK RMSNorm + RoPE + Sage INT8 Q/K and FP8 V preparation for gfx1201, heads in {7, 14, 28, 56}.
// Pointers are device addresses. `stream` is a hipStream_t passed as an integer.
void launch_gfx1201_norm_rope_prepare(int64_t query,
                                      int64_t key,
                                      int64_t value,
                                      int64_t cosine,
                                      int64_t sine,
                                      int64_t maximum,
                                      int64_t query_weight,
                                      int64_t key_weight,
                                      int64_t query_out,
                                      int64_t key_out,
                                      int64_t value_out,
                                      int64_t query_scale,
                                      int64_t key_scale,
                                      int64_t value_scale,
                                      int64_t rows,
                                      int64_t padded_rows,
                                      double sm_scale,
                                      int64_t stream,
                                      int64_t parts,
                                      int64_t heads);
