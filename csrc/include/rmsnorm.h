// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// OPUS RMSNorm host-side launch helpers (torch-free; raw pointers + dims).
#pragma once
#include "opus/hip_minimal.hpp" // dim3, hipStream_t, hipLaunchKernelGGL (both passes)
#include "opus/rmsnorm_opus_kernel.hpp"
#include <utility> // std::pair (structured-binding launch geometry)

namespace aiter {

// Element types for instantiating the kernels, sourced from opus.
using bf16_t = opus::bf16_t;
using fp16_t = opus::fp16_t;
using fp32_t = opus::fp32_t;
using i8_t   = opus::i8_t;
using fp8_t  = opus::fp8_t;

// Launch geometry (block, grid): x = threads/row, y = rows/block; ~2 vecs/thread.
inline std::pair<dim3, dim3> pick_dims(int rows, int vhid)
{
    const int budget = (rows < 256) ? 1024 : 256; // total threads per block
    const int want   = (vhid + 1) / 2;            // ~2 vectors per thread
    // cap threads/row at 256; wider LDS reductions misbehave on gfx942
    const int tpr_cap = 256;
    int tpr           = 64;
    while(tpr < want && tpr < budget && tpr < tpr_cap)
        tpr <<= 1;
    if(tpr > tpr_cap)
        tpr = tpr_cap;
    // pack small-hidden rows, but never more row slots than rows present.
    int rpb = budget / tpr;
    if(rpb < 1)
        rpb = 1;
    if(rpb > rows)
        rpb = rows;
    const int nblocks = (rows + rpb - 1) / rpb;
    return {dim3((unsigned)tpr, (unsigned)rpb), dim3((unsigned)nblocks)};
}


// rmsnorm (+ fused add when residual != nullptr), generic kernel (<=2 ulp).
template <typename scalar_t>
inline void launch_norm(void* out,
                        const void* in,
                        const void* weight,
                        void* residual,
                        void* residual_out,
                        float epsilon,
                        int rows,
                        int hidden,
                        int in_s,
                        int model_sensitive,
                        int gemma,
                        hipStream_t stream)
{
    constexpr int VW = 16 / (int)sizeof(scalar_t); // 8 for bf16/fp16, 4 for fp32
    // oop: out-of-place add (residual_out != residual). In-place / no-add use OOP=false.
    const bool oop = (residual != nullptr) && (residual_out != residual);
    // no pointer-alignment gate: AMDGPU handles misaligned 128-bit access.
    const bool vec           = (hidden % VW == 0);
    const auto [block, grid] = pick_dims(rows, vec ? hidden / VW : hidden);
    // gemma and OOP are compile-time template args (no runtime cost when false).
#define OPUS_LAUNCH_FWD(WIDTH, G, OOP)                                              \
    hipLaunchKernelGGL((rmsnorm_opus_kernel<rmsnorm_opus_traits<scalar_t, WIDTH, G>, OOP>), \
                       grid, block, 0, stream, out, in, weight, residual, residual_out, \
                       epsilon, rows, hidden, in_s, model_sensitive)
#define OPUS_LAUNCH_FWD_OOP(WIDTH, G)                                               \
    do { if(oop) OPUS_LAUNCH_FWD(WIDTH, G, true); else OPUS_LAUNCH_FWD(WIDTH, G, false); } while(0)
    if(vec)
    {
        if(gemma)
            OPUS_LAUNCH_FWD_OOP(VW, true);
        else
            OPUS_LAUNCH_FWD_OOP(VW, false);
    }
    else
    {
        if(gemma)
            OPUS_LAUNCH_FWD_OOP(1, true);
        else
            OPUS_LAUNCH_FWD_OOP(1, false);
    }
#undef OPUS_LAUNCH_FWD
#undef OPUS_LAUNCH_FWD_OOP
}

template <typename in_t, typename out_t>
inline void launch_quant_t(void* out,
                           void* yscale,
                           void* unquant,
                           const void* in,
                           const void* weight,
                           void* residual,
                           void* residual_out,
                           const void* xscale,
                           float epsilon,
                           int rows,
                           int hidden,
                           float qmax,
                           int model_sensitive,
                           hipStream_t stream)
{
    constexpr int VW = 16 / (int)sizeof(in_t); // 8 for bf16/fp16, 4 for fp32
    const bool vec = (hidden % VW == 0);
    // oop: out-of-place add (residual_out != residual). In-place / no-add use OOP=false.
    const bool oop = (residual != nullptr) && (residual_out != residual);
    const auto [block, grid] = pick_dims(rows, vec ? hidden / VW : hidden);
#define OPUS_LAUNCH_QUANT(WIDTH, OOP)                                                       \
    hipLaunchKernelGGL((rmsnorm_quant_opus<rmsnorm_quant_opus_traits<in_t, out_t, WIDTH>, OOP>), \
                       grid, block, 0, stream, out, yscale, unquant, in, weight, residual, \
                       residual_out, xscale, epsilon, rows, hidden, qmax, model_sensitive)
    if(vec)
    {
        if(oop) OPUS_LAUNCH_QUANT(VW, true); else OPUS_LAUNCH_QUANT(VW, false);
    }
    else
    {
        if(oop) OPUS_LAUNCH_QUANT(1, true); else OPUS_LAUNCH_QUANT(1, false);
    }
#undef OPUS_LAUNCH_QUANT
}

// in_code: 0=fp16, 1=bf16, 2=fp32 ; out_code: 0=int8, 1=fp8
inline void launch_quant(void* out,
                         void* yscale,
                         void* unquant,
                         const void* in,
                         const void* weight,
                         void* residual,
                         void* residual_out,
                         const void* xscale,
                         float epsilon,
                         int rows,
                         int hidden,
                         float qmax,
                         int in_code,
                         int out_code,
                         int model_sensitive,
                         hipStream_t stream)
{
#define OPUS_QUANT(IN_T, OUT_T)                                                                     \
    launch_quant_t<IN_T, OUT_T>(out, yscale, unquant, in, weight, residual, residual_out, xscale,   \
                                epsilon, rows, hidden, qmax, model_sensitive, stream)
    if(in_code == 2)
    {
        if(out_code)
            OPUS_QUANT(fp32_t, fp8_t);
        else
            OPUS_QUANT(fp32_t, i8_t);
    }
    else if(in_code == 1)
    {
        if(out_code)
            OPUS_QUANT(bf16_t, fp8_t);
        else
            OPUS_QUANT(bf16_t, i8_t);
    }
    else
    {
        if(out_code)
            OPUS_QUANT(fp16_t, fp8_t);
        else
            OPUS_QUANT(fp16_t, i8_t);
    }
#undef OPUS_QUANT
}

} // namespace aiter
