// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Fused rmsnorm + dynamic/smooth quant (int8/fp8) via launch_quant. Own TU so the
// per-token quant kernels compile in parallel with norm/arq.

#include "rmsnorm.h"

#define OPUS_EXPORT extern "C" __attribute__((visibility("default")))

// Fused rmsnorm + quant. residual/residual_out/xscale/unquant = 0 to disable.
// residual_out != residual => out-of-place fused add (no host staging copy). out_code: 0=int8,1=fp8.
OPUS_EXPORT void rms_norm_quant_opus(size_t out,
                                     size_t yscale,
                                     size_t unquant,
                                     size_t in,
                                     size_t weight,
                                     size_t residual,
                                     size_t residual_out,
                                     size_t xscale,
                                     float epsilon,
                                     int rows,
                                     int hidden,
                                     float qmax,
                                     int in_code,
                                     int out_code,
                                     int model_sensitive,
                                     size_t stream)
{
    using namespace aiter;
    if(rows <= 0 || hidden <= 0)
        return;
    launch_quant(reinterpret_cast<void*>(out),
                 reinterpret_cast<void*>(yscale),
                 reinterpret_cast<void*>(unquant),
                 reinterpret_cast<const void*>(in),
                 reinterpret_cast<const void*>(weight),
                 reinterpret_cast<void*>(residual),
                 reinterpret_cast<void*>(residual_out),
                 reinterpret_cast<const void*>(xscale),
                 epsilon,
                 rows,
                 hidden,
                 qmax,
                 in_code,
                 out_code,
                 model_sensitive,
                 reinterpret_cast<hipStream_t>(stream));
}
