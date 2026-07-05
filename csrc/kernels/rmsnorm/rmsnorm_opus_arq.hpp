// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Per-output-dtype split of the add_rmsnorm_quant (arq) launcher. Each out dtype
// (int8 / fp8 / fp4 / no-quant) is defined in its own .cu so the arq kernel families
// compile in parallel; the C entrypoint (arq_entry.cu) dispatches out_code to these.
// The kernels/launcher (launch_arq_io in rmsnorm.h) are unchanged -- this only moves
// the instantiation of each out dtype into a separate translation unit.
#pragma once
#include "rmsnorm.h"

// Common parameter/argument lists so every arq launcher shares one signature.
#define OPUS_ARQ_PARAMS                                                                             \
    int in_code, void *out, void *rout, void *scale, const void *in, const void *rin,              \
        const void *w, const void *xsc, float epsilon, int m, int n, float qmax, int in_s,         \
        int rin_s, int rout_s, int out_s, int group, int shuffle, int gemma, int cu_num,           \
        hipStream_t s
#define OPUS_ARQ_ARGS                                                                               \
    out, rout, scale, in, rin, w, xsc, epsilon, m, n, qmax, in_s, rin_s, rout_s, out_s, group,      \
        shuffle, gemma, cu_num, s

// Defines one per-out-dtype launcher: dispatches in_code (bf16/fp16) into launch_arq_io.
#define OPUS_ARQ_DEFINE(FN, OUT_T)                                                                  \
    void FN(OPUS_ARQ_PARAMS)                                                                        \
    {                                                                                              \
        if(in_code == 1)                                                                           \
            launch_arq_io<bf16_t, OUT_T>(OPUS_ARQ_ARGS);                                            \
        else                                                                                       \
            launch_arq_io<fp16_t, OUT_T>(OPUS_ARQ_ARGS);                                            \
    }

namespace aiter {
void opus_arq_i8(OPUS_ARQ_PARAMS);
void opus_arq_fp8(OPUS_ARQ_PARAMS);
void opus_arq_fp4(OPUS_ARQ_PARAMS);
void opus_arq_noquant(OPUS_ARQ_PARAMS); // out dtype == in dtype
} // namespace aiter
