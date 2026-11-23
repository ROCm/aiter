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

// gfx950 MXFP8 1x128 blockscale with plain A / B in the NT / NN / TT / TN layouts;
// the layout is read from the XQ / WQ strides. BF16 Y.
void opus_gemm_a8w8_blockscale_mxfp8_launch(aiter_tensor_t& XQ,
                                            aiter_tensor_t& WQ,
                                            aiter_tensor_t& x_scale,
                                            aiter_tensor_t& w_scale,
                                            aiter_tensor_t& Y,
                                            int kid);

// MXFP8 x MXFP8 (gfx950): fp8 XQ [B, M, K] and WQ [B, N, K] views, row-major
// [B * rows, K/32] e8m0 scales, bf16 Y [B, M, N]; split-K kids take a caller
// workspace. a8w8_mxfp8: plain NT / NN / TT / TN from the strides;
// a8w8_mxfp8_bpreshuffle: NT with WQ = shuffle_weight(w, (16, 16)).
void opus_gemm_a8w8_mxfp8_launch(aiter_tensor_t& XQ,
                                 aiter_tensor_t& WQ,
                                 aiter_tensor_t& Y,
                                 aiter_tensor_t& x_scale,
                                 aiter_tensor_t& w_scale,
                                 std::optional<aiter_tensor_t> workspace,
                                 int kid,
                                 int split_k);
void opus_gemm_a8w8_mxfp8_bpreshuffle_launch(aiter_tensor_t& XQ,
                                             aiter_tensor_t& WQ,
                                             aiter_tensor_t& Y,
                                             aiter_tensor_t& x_scale,
                                             aiter_tensor_t& w_scale,
                                             std::optional<aiter_tensor_t> workspace,
                                             int kid,
                                             int split_k);
