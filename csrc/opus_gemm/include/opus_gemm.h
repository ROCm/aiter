// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

// Exact-kid OPUS entry points. aiter_tensor_t keeps this header torch-free.
#include "aiter_tensor.h"
#include <optional>

void opus_gemm_a16w16_launch(aiter_tensor_t& XQ,
                             aiter_tensor_t& WQ,
                             aiter_tensor_t& Y,
                             std::optional<aiter_tensor_t> bias,
                             std::optional<aiter_tensor_t> workspace,
                             int kid,
                             int split_k);

void opus_gemm_a8w8_launch(aiter_tensor_t& XQ,
                           aiter_tensor_t& WQ,
                           aiter_tensor_t& Y,
                           int kid);

// Blockscale interfaces require both scale tensors.
void opus_gemm_a8w8_blockscale_launch(aiter_tensor_t& XQ,
                                      aiter_tensor_t& WQ,
                                      aiter_tensor_t& Y,
                                      aiter_tensor_t& x_scale,
                                      aiter_tensor_t& w_scale,
                                      int kid);

void opus_gemm_a8w8_blockscale_bpreshuffle_launch(
    aiter_tensor_t& XQ,
    aiter_tensor_t& WQ,
    aiter_tensor_t& x_scale,
    aiter_tensor_t& w_scale,
    aiter_tensor_t& Y,
    int kid,
    std::optional<aiter_tensor_t> workspace = std::nullopt);

// FP8 blockscale grouped GEMMs (gfx950, DeepGEMM interfaces), E8M0 scales, nothing shuffled.
// M-grouped (fwd NT / dgrad NN): group = grouped_layout [M] (contiguous, workspace int32 [2 G]) or masked_m [G].
void opus_gemm_a8w8_blockscale_grouped_launch(aiter_tensor_t& XQ,
                                              aiter_tensor_t& WQ,
                                              aiter_tensor_t& Y,
                                              aiter_tensor_t& x_scale,
                                              aiter_tensor_t& w_scale,
                                              aiter_tensor_t& group,
                                              aiter_tensor_t& workspace,
                                              int kid,
                                              bool masked);

// K-grouped (wgrad TN): Y[g] (+)= XQ[rows of g]^T WQ[rows of g], grouped_layout int32 [G] rows per group.
void opus_gemm_a8w8_blockscale_k_grouped_launch(aiter_tensor_t& XQ,
                                                aiter_tensor_t& WQ,
                                                aiter_tensor_t& Y,
                                                aiter_tensor_t& x_scale,
                                                aiter_tensor_t& w_scale,
                                                aiter_tensor_t& grouped_layout,
                                                std::optional<aiter_tensor_t> C,
                                                aiter_tensor_t& workspace,
                                                int kid,
                                                int seg_len);
