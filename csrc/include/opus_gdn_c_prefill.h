// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include <torch/extension.h>

// gfx942 C-input GDN prefill, dense or packed varlen.
//
// c_mode:
//   0 = conservative measured auto policy (dense); CF for packed varlen
//   1 = CF, fused recurrence and output
//   2 = CS, split recurrence followed by the shared K6
//
// An empty cu_seqlens selects the dense [B, T, ...] layout.  Otherwise q/k/v
// are packed as [1, total_tokens, ...], the state carries N entries, and every
// sequence length must be a multiple of BT=64 because the C-input kernels emit
// no token-tail predicates.
void opus_gdn_c_prefill_fwd(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor g,
    torch::Tensor beta,
    torch::Tensor o,
    float scale,
    torch::Tensor initial_state,
    torch::Tensor final_state,
    torch::Tensor cu_seqlens,     // [N + 1] int32; empty => dense
    torch::Tensor chunk_indices,  // [total_chunks, 2] int32
    torch::Tensor chunk_offsets,  // [N + 1] int32
    bool has_initial_state,
    bool output_final_state,
    int c_mode,
    bool use_env_overrides = true);
