// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include <cstdint>

// gfx1201 SageAttention core: INT8 QK, FP8 PV, BM512/BN32.
// Pointers are device addresses. `stream` is a hipStream_t passed as an integer.
void launch_gfx1201_sage_attention(int64_t q,
                                   int64_t k,
                                   int64_t v,
                                   int64_t q_scale,
                                   int64_t k_scale,
                                   int64_t v_scale,
                                   int64_t out,
                                   int64_t batch_size,
                                   int64_t padded_seq_len,
                                   int64_t valid_seq_len,
                                   int64_t num_heads,
                                   int64_t stream);
