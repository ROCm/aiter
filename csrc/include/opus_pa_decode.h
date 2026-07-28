// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include "aiter_tensor.h"

#include <optional>

// Opus-based paged attention decode (BF16 Q + FP8 KV, asm MFMA path).
// API mirrors pa_fwd_asm / asm_pa.cu.
void pa_opus_fwd(aiter_tensor_t& Q,
                 aiter_tensor_t& K,
                 aiter_tensor_t& V,
                 aiter_tensor_t& block_tables,
                 aiter_tensor_t& context_lens,
                 int block_tables_stride0,
                 int max_qlen,
                 std::optional<aiter_tensor_t> K_QScale,
                 std::optional<aiter_tensor_t> V_QScale,
                 std::optional<aiter_tensor_t> out_,
                 std::optional<aiter_tensor_t> qo_indptr,
                 int high_precision);
