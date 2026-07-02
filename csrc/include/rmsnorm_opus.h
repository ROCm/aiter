// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// OPUS RMSNorm host-side launch helpers. Fully torch-free / HIP-runtime-free:
// only opus/hip_minimal.hpp (host launch) + opus.hpp (device, via the kernel
// header). Tensors cross the ctypes boundary as raw pointers + dims, so nothing
// pulls <hip/hip_runtime.h> or aiter_tensor.h.
#pragma once
#include "opus/hip_minimal.hpp" // dim3, hipStream_t, hipLaunchKernelGGL (both passes)
#include "opus/rmsnorm_opus_kernel.hpp"

namespace aiter {
namespace rmsnorm_opus {

// 2D launch geometry: blockDim.x = threads-per-row (power of two), blockDim.y =
// rows-per-block. For large hidden this is 1 row/block; for small hidden it packs
// several rows so tiny rows are not launch/occupancy-bound (vhid = vector work per
// row = hidden/width). tpr targets ~2 vectors/thread so large hidden is unchanged.
struct launch_dims
{
    dim3 block;
    dim3 grid;
};

inline launch_dims pick_dims(int rows, int vhid)
{
    const int budget = (rows < 256) ? 1024 : 256; // total threads per block
    const int want   = (vhid + 1) / 2;            // ~2 vectors per thread
    int tpr          = 64;
    while(tpr < want && tpr < budget)
        tpr <<= 1;
    if(tpr > budget)
        tpr = budget;
    int rpb = budget / tpr;
    if(rpb < 1)
        rpb = 1;
    const int nblocks = (rows + rpb - 1) / rpb;
    return {dim3(tpr, rpb), dim3(nblocks)};
}

inline bool aligned16(const void* p) { return (reinterpret_cast<size_t>(p) % 16) == 0; }

template <typename scalar_t>
inline void launch_rms(void* out,
                       const void* in,
                       const void* weight,
                       float epsilon,
                       int rows,
                       int hidden,
                       int model_sensitive,
                       hipStream_t stream)
{
    const bool vec8     = (hidden % 8 == 0) && aligned16(out) && aligned16(in) && aligned16(weight);
    const launch_dims d = pick_dims(rows, vec8 ? hidden / 8 : hidden);
    if(vec8)
        hipLaunchKernelGGL((rmsnorm2d_fwd_kernel<scalar_t, 8>),
                           d.grid,
                           d.block,
                           0,
                           stream,
                           out,
                           in,
                           weight,
                           epsilon,
                           rows,
                           hidden,
                           model_sensitive);
    else
        hipLaunchKernelGGL((rmsnorm2d_fwd_kernel<scalar_t, 1>),
                           d.grid,
                           d.block,
                           0,
                           stream,
                           out,
                           in,
                           weight,
                           epsilon,
                           rows,
                           hidden,
                           model_sensitive);
}

template <typename scalar_t>
inline void launch_fused_add(void* inout,
                             void* residual,
                             const void* weight,
                             float epsilon,
                             int rows,
                             int hidden,
                             int model_sensitive,
                             hipStream_t stream)
{
    const bool vec8 =
        (hidden % 8 == 0) && aligned16(inout) && aligned16(residual) && aligned16(weight);
    const launch_dims d = pick_dims(rows, vec8 ? hidden / 8 : hidden);
    if(vec8)
        hipLaunchKernelGGL((fused_add_rmsnorm2d_fwd_kernel<scalar_t, 8>),
                           d.grid,
                           d.block,
                           0,
                           stream,
                           inout,
                           residual,
                           weight,
                           epsilon,
                           rows,
                           hidden,
                           model_sensitive);
    else
        hipLaunchKernelGGL((fused_add_rmsnorm2d_fwd_kernel<scalar_t, 1>),
                           d.grid,
                           d.block,
                           0,
                           stream,
                           inout,
                           residual,
                           weight,
                           epsilon,
                           rows,
                           hidden,
                           model_sensitive);
}

template <typename in_t, typename out_t>
inline void launch_quant_t(void* out,
                           void* yscale,
                           void* unquant,
                           const void* in,
                           const void* weight,
                           void* residual,
                           const void* xscale,
                           float epsilon,
                           int rows,
                           int hidden,
                           float qmax,
                           int model_sensitive,
                           hipStream_t stream)
{
    const bool vec8 = (hidden % 8 == 0) && aligned16(out) && aligned16(in) && aligned16(weight) &&
                      (residual == nullptr || aligned16(residual)) &&
                      (unquant == nullptr || aligned16(unquant));
    const launch_dims d = pick_dims(rows, vec8 ? hidden / 8 : hidden);
    if(vec8)
        hipLaunchKernelGGL((rmsnorm2d_quant_kernel<in_t, out_t, 8>),
                           d.grid,
                           d.block,
                           0,
                           stream,
                           out,
                           yscale,
                           unquant,
                           in,
                           weight,
                           residual,
                           xscale,
                           epsilon,
                           rows,
                           hidden,
                           qmax,
                           model_sensitive);
    else
        hipLaunchKernelGGL((rmsnorm2d_quant_kernel<in_t, out_t, 1>),
                           d.grid,
                           d.block,
                           0,
                           stream,
                           out,
                           yscale,
                           unquant,
                           in,
                           weight,
                           residual,
                           xscale,
                           epsilon,
                           rows,
                           hidden,
                           qmax,
                           model_sensitive);
}

// in_code: 0=fp16, 1=bf16 ; out_code: 0=int8, 1=fp8
inline void launch_quant(void* out,
                         void* yscale,
                         void* unquant,
                         const void* in,
                         const void* weight,
                         void* residual,
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
    if(in_code)
    {
        if(out_code)
            launch_quant_t<bf16_t, fp8_t>(out,
                                          yscale,
                                          unquant,
                                          in,
                                          weight,
                                          residual,
                                          xscale,
                                          epsilon,
                                          rows,
                                          hidden,
                                          qmax,
                                          model_sensitive,
                                          stream);
        else
            launch_quant_t<bf16_t, i8_t>(out,
                                         yscale,
                                         unquant,
                                         in,
                                         weight,
                                         residual,
                                         xscale,
                                         epsilon,
                                         rows,
                                         hidden,
                                         qmax,
                                         model_sensitive,
                                         stream);
    }
    else
    {
        if(out_code)
            launch_quant_t<fp16_t, fp8_t>(out,
                                          yscale,
                                          unquant,
                                          in,
                                          weight,
                                          residual,
                                          xscale,
                                          epsilon,
                                          rows,
                                          hidden,
                                          qmax,
                                          model_sensitive,
                                          stream);
        else
            launch_quant_t<fp16_t, i8_t>(out,
                                         yscale,
                                         unquant,
                                         in,
                                         weight,
                                         residual,
                                         xscale,
                                         epsilon,
                                         rows,
                                         hidden,
                                         qmax,
                                         model_sensitive,
                                         stream);
    }
}

} // namespace rmsnorm_opus
} // namespace aiter
