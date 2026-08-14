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

// Production A16W16 C ABI. Optional tensors use nullptr; stream is caller-owned.
AITER_C_ITFS int opus_gemm_a16w16_launch_cabi(aiter_tensor_t* XQ,
                                              aiter_tensor_t* WQ,
                                              aiter_tensor_t* Y,
                                              aiter_tensor_t* bias,
                                              aiter_tensor_t* workspace,
                                              int64_t kid,
                                              int64_t split_k,
                                              hipStream_t stream);

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
    int kid);
