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

// Bit-exact (vs CK) launch for one CK tile geometry (TN threads/row, RN vecs).
template <typename scalar_t, int TN, int RN>
inline void launch_be(void* out,
                      const void* in,
                      const void* weight,
                      void* residual,
                      float epsilon,
                      int rows,
                      int hidden,
                      int model_sensitive,
                      hipStream_t stream)
{
    const int tm = (TN > 64) ? 1 : (256 / TN); // rows/block; TN>64 needs 1 row/block
    const dim3 block(TN, tm);
    const dim3 grid((rows + tm - 1) / tm);
    hipLaunchKernelGGL((rmsnorm2d_fwd_be_kernel<scalar_t, TN, RN>),
                       grid,
                       block,
                       0,
                       stream,
                       out,
                       in,
                       weight,
                       residual,
                       epsilon,
                       rows,
                       hidden,
                       model_sensitive);
}

// Dispatch to the bit-exact kernel for CK's vn=8 tile buckets; returns false if
// this hidden size has no bit-exact geometry (caller uses the generic kernel).
template <typename scalar_t>
inline bool launch_norm_be(void* out,
                           const void* in,
                           const void* weight,
                           void* residual,
                           float epsilon,
                           int rows,
                           int hidden,
                           int model_sensitive,
                           hipStream_t stream)
{
#define OPUS_BE(N, TN, RN)                                                              \
    case N:                                                                             \
        launch_be<scalar_t, TN, RN>(                                                    \
            out, in, weight, residual, epsilon, rows, hidden, model_sensitive, stream); \
        return true
    switch(hidden)
    {
        OPUS_BE(64, 8, 1);
        OPUS_BE(128, 16, 1);
        OPUS_BE(512, 64, 1);
        OPUS_BE(1024, 64, 2);
        OPUS_BE(1536, 64, 3);
        OPUS_BE(2048, 256, 1);
        OPUS_BE(3072, 128, 3);
        OPUS_BE(4096, 256, 2);
        OPUS_BE(6144, 256, 3);
        OPUS_BE(8192, 256, 4);
    default: return false;
    }
#undef OPUS_BE
}

// rmsnorm (+ fused residual add when residual != nullptr; in-place when out == in).
// Bit-exact vs CK on the vn=8 tile buckets; generic (formula-exact, <=2 ulp) otherwise.
template <typename scalar_t>
inline void launch_norm(void* out,
                        const void* in,
                        const void* weight,
                        void* residual,
                        float epsilon,
                        int rows,
                        int hidden,
                        int model_sensitive,
                        hipStream_t stream)
{
    const bool aligned = aligned16(out) && aligned16(in) && aligned16(weight) &&
                         (residual == nullptr || aligned16(residual));
    if(aligned && (hidden % 8 == 0) &&
       launch_norm_be<scalar_t>(
           out, in, weight, residual, epsilon, rows, hidden, model_sensitive, stream))
        return;
    const bool vec8 = (hidden % 8 == 0) && aligned16(out) && aligned16(in) && aligned16(weight) &&
                      (residual == nullptr || aligned16(residual));
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
                           residual,
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
                           residual,
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
