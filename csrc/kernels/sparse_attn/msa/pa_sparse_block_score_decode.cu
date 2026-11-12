// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Every prebuilt variant of the uniform-row scoring pass with one query tile
// per wave (two: pa_sparse_block_score_decode_q2.cu).
#include "pa_sparse_block_select.hpp"

namespace aiter {
namespace sparse_attn {
SPARSE_DECODE_TABLE_Q1(SPARSE_DECODE_DEFINE)
} // namespace sparse_attn
} // namespace aiter
