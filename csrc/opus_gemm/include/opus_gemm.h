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

// MXFP4 x MXFP4 (gfx950): fp4x2 XQ [B, M, K/2], shuffle_weight WQ [B, N, K/2],
// e8m0_shuffle scales, bf16 Y [B, M, N]; split-K kids take a caller workspace.
void opus_gemm_a4w4_bpreshuffle_launch(aiter_tensor_t& XQ,
                           aiter_tensor_t& WQ,
                           aiter_tensor_t& Y,
                           aiter_tensor_t& x_scale,
                           aiter_tensor_t& w_scale,
                           std::optional<aiter_tensor_t> workspace,
                           int kid,
                           int split_k);
