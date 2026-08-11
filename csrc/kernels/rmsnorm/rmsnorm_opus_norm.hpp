// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Plain norm launcher (launch_norm) split per input dtype (bf16/fp16/fp32), each in its own
// .cu so the be+generic kernels compile in parallel (norm is the compile bottleneck).
// Kernels/launcher unchanged; rmsnorm_opus_norm_entry.cu dispatches the dtype code.
#pragma once
#include "rmsnorm.h"

// Shared param/arg lists. residual==null: no-add; residual_out==residual: in-place; else oop.
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
