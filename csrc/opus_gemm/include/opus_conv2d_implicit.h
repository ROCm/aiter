// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Conv2D implicit GEMM (opus-fused, gfx942) — C++ entry point.
// Uses aiter_tensor_t (torch-free POD) like opus_gemm.h.
#pragma once

#include "aiter_tensor.h"

// Conv2D forward using implicit GEMM fused into opus asm pipeline.
//
// Caller responsibilities (Python op layer handles these):
//   - input:  [N, Hi, Wi, C_pad] bf16 NHWC, C_pad = group * ceil(Cpg, 8)*8
//   - weight: [group, Kpg_pad, GEMM_K_pad] bf16 (RSC order, pre-packed)
//             Kpg_pad = ceil(K/group, 16)*16, GEMM_K_pad = ceil(R*S*Cpg_pad, 128)*128
//   - output: [M, group*Kpg_pad] bf16, M = N * Ho * Wo
//
// Conv parameters are passed explicitly (not inferred from tensor shapes)
// because padding changes the tensor dimensions.
void opus_conv2d_implicit(
    aiter_tensor_t& input,
    aiter_tensor_t& weight,
    aiter_tensor_t& output,
    int N_batch, int C, int K,
    int Hi, int Wi,
    int R, int S,
    int pad_h, int pad_w,
    int stride_h, int stride_w,
    int dil_h, int dil_w,
    int group);
