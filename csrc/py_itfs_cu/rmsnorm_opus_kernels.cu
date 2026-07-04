// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// OPUS RMSNorm C ABI (ctypes): raw int64 pointers + dims, validated Python-side.
// dtype: 0=fp16, 1=bf16, 2=fp32.

#include "rmsnorm.h"

#define OPUS_EXPORT extern "C" __attribute__((visibility("default")))

// Dispatch a norm launch on the dtype code.
#define OPUS_NORM_DISPATCH(DTYPE, O, I, W, R)                                                        \
    do                                                                                               \
    {                                                                                                \
        if((DTYPE) == 2)                                                                             \
            launch_norm<fp32_t>((O), (I), (W), (R), epsilon, rows, hidden, in_s, model_sensitive, gemma, s); \
        else if((DTYPE) == 1)                                                                        \
            launch_norm<bf16_t>((O), (I), (W), (R), epsilon, rows, hidden, in_s, model_sensitive, gemma, s); \
        else                                                                                         \
            launch_norm<fp16_t>((O), (I), (W), (R), epsilon, rows, hidden, in_s, model_sensitive, gemma, s); \
    } while(0)

OPUS_EXPORT void rms_norm_opus(size_t out,
                               size_t in,
                               size_t weight,
                               float epsilon,
                               int rows,
                               int hidden,
                               int in_s, // input row stride (elements); == hidden if contiguous
                               int dtype,
                               int model_sensitive,
                               int gemma,
                               size_t stream)
{
    using namespace aiter;
    if(rows <= 0 || hidden <= 0)
        return;
    auto s  = reinterpret_cast<hipStream_t>(stream);
    auto* o = reinterpret_cast<void*>(out);
    auto* i = reinterpret_cast<const void*>(in);
    auto* w = reinterpret_cast<const void*>(weight);
    OPUS_NORM_DISPATCH(dtype, o, i, w, nullptr);
}

OPUS_EXPORT void fused_add_rms_norm_opus(size_t inout,
                                         size_t residual,
                                         size_t weight,
                                         float epsilon,
                                         int rows,
                                         int hidden,
                                         int dtype,
                                         int model_sensitive,
                                         int gemma,
                                         size_t stream)
{
    using namespace aiter;
    if(rows <= 0 || hidden <= 0)
        return;
    const int in_s = hidden; // in-place fused-add always operates on contiguous rows
    auto s   = reinterpret_cast<hipStream_t>(stream);
    auto* io = reinterpret_cast<void*>(inout);
    auto* r  = reinterpret_cast<void*>(residual);
    auto* w  = reinterpret_cast<const void*>(weight);
    OPUS_NORM_DISPATCH(dtype, io, io, w, r); // in-place: out == in == inout
}

// Fused rmsnorm + quant. residual/xscale/unquant = 0 to disable. out_code: 0=int8,1=fp8.
OPUS_EXPORT void rms_norm_quant_opus(size_t out,
                                     size_t yscale,
                                     size_t unquant,
                                     size_t in,
                                     size_t weight,
                                     size_t residual,
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
