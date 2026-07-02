// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// OPUS RMSNorm C ABI (ctypes). Single torch-free TU with no HIP-runtime / CK /
// pybind / aiter_tensor.h dependency: tensors arrive as raw pointers + dims, so
// the whole module is a handful of preprocessed lines under -D__HIPCC_RTC__.
//
// Args are passed by aiter's _ctypes_call as int64 (pointers/dims) + float; the
// Python wrapper validates dtype/shape and supplies the current HIP stream.
// is_bf16: 1 = bf16, 0 = fp16.

#include "rmsnorm_opus.h"

#define OPUS_EXPORT extern "C" __attribute__((visibility("default")))

OPUS_EXPORT void rms_norm_opus(size_t out,
                               size_t in,
                               size_t weight,
                               float epsilon,
                               int rows,
                               int hidden,
                               int is_bf16,
                               int model_sensitive,
                               size_t stream)
{
    using namespace aiter::rmsnorm_opus;
    if(rows <= 0 || hidden <= 0)
        return;
    auto s  = reinterpret_cast<hipStream_t>(stream);
    auto* o = reinterpret_cast<void*>(out);
    auto* i = reinterpret_cast<const void*>(in);
    auto* w = reinterpret_cast<const void*>(weight);
    if(is_bf16)
        launch_norm<bf16_t>(o, i, w, nullptr, epsilon, rows, hidden, model_sensitive, s);
    else
        launch_norm<fp16_t>(o, i, w, nullptr, epsilon, rows, hidden, model_sensitive, s);
}

OPUS_EXPORT void fused_add_rms_norm_opus(size_t inout,
                                         size_t residual,
                                         size_t weight,
                                         float epsilon,
                                         int rows,
                                         int hidden,
                                         int is_bf16,
                                         int model_sensitive,
                                         size_t stream)
{
    using namespace aiter::rmsnorm_opus;
    if(rows <= 0 || hidden <= 0)
        return;
    auto s   = reinterpret_cast<hipStream_t>(stream);
    auto* io = reinterpret_cast<void*>(inout);
    auto* r  = reinterpret_cast<void*>(residual);
    auto* w  = reinterpret_cast<const void*>(weight);
    if(is_bf16) // in-place: out == in == inout
        launch_norm<bf16_t>(io, io, w, r, epsilon, rows, hidden, model_sensitive, s);
    else
        launch_norm<fp16_t>(io, io, w, r, epsilon, rows, hidden, model_sensitive, s);
}

// Fused rmsnorm + dynamic/smooth quant. residual/xscale/unquant = 0 to disable
// fused-add / smooth / save-unquant. in_code: 0=fp16,1=bf16; out_code: 0=int8,1=fp8.
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
    using namespace aiter::rmsnorm_opus;
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
