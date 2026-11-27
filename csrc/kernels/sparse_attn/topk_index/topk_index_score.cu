// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Instantiation only; the table and the launchers live in topk_index_score.hpp.
// Builds 14 specialisations: 7 (H, Q) cells x AUX_K in {0, 3}.
#include "topk_index_score.hpp"

namespace aiter {
namespace sparse_attn {
OPUS_IDX_SCORE_TABLE(OPUS_IDX_SCORE_DEFINE)
} // namespace sparse_attn
} // namespace aiter
