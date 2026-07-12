// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Host-side dispatcher for conv2d implicit GEMM (opus-fused, gfx942).

#ifndef __HIP_DEVICE_COMPILE__

#include "opus_conv2d_implicit.h"
#include "gfx942/opus_conv2d_implicit_kernel.cuh"
#include "opus_gemm_utils.cuh"
#include "aiter_hip_common.h"
#include "aiter_stream.h"

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
    int group)
{
    int Ho = (Hi + 2 * pad_h - dil_h * (R - 1) - 1) / stride_h + 1;
    int Wo = (Wi + 2 * pad_w - dil_w * (S - 1) - 1) / stride_w + 1;
    AITER_CHECK(Ho > 0 && Wo > 0, "opus_conv2d_implicit: invalid output size Ho=", Ho, " Wo=", Wo);

    int Cpg = C / group;
    int Kpg = K / group;
    int Cpg_pad = ceil_div(Cpg, 8) * 8;
    int C_pad   = group * Cpg_pad;
    int Kpg_pad = ceil_div(Kpg, 16) * 16;
    int GEMM_K_real = R * S * Cpg_pad;
    int GEMM_K_pad  = ceil_div(GEMM_K_real, 128) * 128;
    int M = N_batch * Ho * Wo;
    int stride_out = group * Kpg_pad;

    conv_implicit_kargs ka{};
    ka.ptr_in  = input.ptr;
    ka.ptr_wei = weight.ptr;
    ka.ptr_out = output.ptr;
    ka.M = M;
    ka.Kpg_pad = Kpg_pad;
    ka.GEMM_K_pad = GEMM_K_pad;
    ka.GEMM_K_real = GEMM_K_real;
    ka.group = group;
    ka.Hi = Hi;
    ka.Wi = Wi;
    ka.C_pad = C_pad;
    ka.Cpg_pad = Cpg_pad;
    ka.Ho = Ho;
    ka.Wo = Wo;
    ka.stride_h = stride_h;
    ka.stride_w = stride_w;
    ka.pad_h = pad_h;
    ka.pad_w = pad_w;
    ka.dil_h = dil_h;
    ka.dil_w = dil_w;
    ka.stride_out = stride_out;

    ka.div_GEMMK = opus::mdiv(GEMM_K_pad);
    ka.div_HoWo  = opus::mdiv(Ho * Wo);
    ka.div_Wo    = opus::mdiv(Wo);
    ka.div_SC    = opus::mdiv(S * Cpg_pad);
    ka.div_C     = opus::mdiv(Cpg_pad);

    int ntm = ceil_div(M, 128);
    int ntn = ceil_div(Kpg_pad, 128);
    dim3 grid(ntm * ntn, 1, group);
    dim3 block(512);

    hipStream_t stream = aiter::getCurrentHIPStream();
    conv_implicit_gemm_kernel<ConvImplicitTraits><<<grid, block, 0, stream>>>(ka);
}

#endif // !__HIP_DEVICE_COMPILE__
