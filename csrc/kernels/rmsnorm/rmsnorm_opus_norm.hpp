// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Per-input-dtype split of the plain norm launcher. Each dtype (bf16 / fp16 / fp32) is
// defined in its own .cu so the be+generic norm kernel families instantiate in parallel
// (norm is the compile bottleneck; its kernels are the expensive ones to instantiate).
// The entrypoints (rmsnorm_opus_norm_entry.cu) dispatch the dtype code to these. The
// kernels/launcher (launch_norm in rmsnorm.h) are unchanged -- this only moves each
// dtype's instantiation into a separate translation unit.
#pragma once
#include "rmsnorm.h"

// Common parameter/argument lists so every per-dtype norm launcher shares one signature.
// residual == nullptr -> no add; residual_out == residual -> in-place; distinct -> out-of-place.
#define OPUS_NORM_PARAMS                                                                            \
    void *out, const void *in, const void *weight, void *residual, void *residual_out,             \
        float epsilon, int rows, int hidden, int in_s, int model_sensitive, int gemma,             \
        hipStream_t s
#define OPUS_NORM_ARGS                                                                              \
    out, in, weight, residual, residual_out, epsilon, rows, hidden, in_s, model_sensitive, gemma, s

#define OPUS_NORM_DEFINE(FN, T)                                                                     \
    void FN(OPUS_NORM_PARAMS) { launch_norm<T>(OPUS_NORM_ARGS); }

namespace aiter {
void opus_norm_bf16(OPUS_NORM_PARAMS);
void opus_norm_fp16(OPUS_NORM_PARAMS);
void opus_norm_fp32(OPUS_NORM_PARAMS);
} // namespace aiter
