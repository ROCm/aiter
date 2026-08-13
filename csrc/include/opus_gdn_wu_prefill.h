// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once
#include <torch/extension.h>

void opus_gdn_wu_prefill_fwd(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor g,
    torch::Tensor beta,
    torch::Tensor o,
    float scale,
    torch::Tensor initial_state,
    torch::Tensor final_state,
    torch::Tensor cu_seqlens,
    torch::Tensor chunk_indices,
    torch::Tensor chunk_offsets,
    int varlen_max_chunks,
    bool has_initial_state,
    bool output_final_state,
    int BT,
    int BV,
    int num_warps,
    int k1_algo,
    int k2_mode,
    bool use_env_overrides = true);
