// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// OPUS RMSNorm C ABI (ctypes): raw int64 pointers + dims, validated Python-side.
// dtype: 0=fp16, 1=bf16, 2=fp32.
//
// Compile unit 1/N of module_rmsnorm: the plain norm entrypoints (no quant), all
// backed by launch_norm (be/generic kernels). Split into its own translation unit so
// the norm kernel family compiles in parallel with the quant/arq families.

#include "rmsnorm.h"

#define OPUS_EXPORT extern "C" __attribute__((visibility("default")))

// Dispatch a norm launch on the dtype code. R = residual_in, RO = residual_out.
#define OPUS_NORM_DISPATCH(DTYPE, O, I, W, R, RO)                                                    \
    do                                                                                               \
    {                                                                                                \
        if((DTYPE) == 2)                                                                             \
            launch_norm<fp32_t>((O), (I), (W), (R), (RO), epsilon, rows, hidden, in_s, model_sensitive, gemma, s); \
        else if((DTYPE) == 1)                                                                        \
            launch_norm<bf16_t>((O), (I), (W), (R), (RO), epsilon, rows, hidden, in_s, model_sensitive, gemma, s); \
        else                                                                                         \
            launch_norm<fp16_t>((O), (I), (W), (R), (RO), epsilon, rows, hidden, in_s, model_sensitive, gemma, s); \
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
    OPUS_NORM_DISPATCH(dtype, o, i, w, nullptr, nullptr);
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
    OPUS_NORM_DISPATCH(dtype, io, io, w, r, r); // in-place: out==in==inout, residual_out==residual
}

// Out-of-place fused add + rmsnorm: out = rmsnorm(input + residual_in) * weight,
// residual_out = input + residual_in. Reads input/residual_in, writes out/residual_out in
// one pass (no host-side staging copies). in_s = input row stride (elements).
OPUS_EXPORT void add_rms_norm_opus(size_t out,
                                   size_t in,
                                   size_t residual_in,
                                   size_t residual_out,
                                   size_t weight,
                                   float epsilon,
                                   int rows,
                                   int hidden,
                                   int in_s,
                                   int dtype,
                                   int model_sensitive,
                                   int gemma,
                                   size_t stream)
{
    using namespace aiter;
    if(rows <= 0 || hidden <= 0)
        return;
    auto s   = reinterpret_cast<hipStream_t>(stream);
    auto* o  = reinterpret_cast<void*>(out);
    auto* i  = reinterpret_cast<const void*>(in);
    auto* ri = reinterpret_cast<void*>(residual_in);
    auto* ro = reinterpret_cast<void*>(residual_out);
    auto* w  = reinterpret_cast<const void*>(weight);
    OPUS_NORM_DISPATCH(dtype, o, i, w, ri, ro);
}
