// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>

struct Args {
    const hip_bfloat16 *x, *weight, *dy;
    const int* offsets;
    hip_bfloat16* dx;
    float* dw;
    int experts, n, k;
};
#ifdef DGRAD
extern "C" __global__ void opus_gemm_gfx950_bf16_latent_moe_bwd_scalar_nn(Args a) {
    int row = blockIdx.x, lo = 0, hi = a.experts;
    while (lo < hi) {
        int mid = (lo+hi)/2;
        if (a.offsets[mid+1] <= row) lo = mid+1;
        else hi = mid;
    }
    for (int k = threadIdx.x; k < a.k; k += blockDim.x) {
        float sum = 0;
        for (int n = 0; n < a.n; ++n)
            sum = fmaf(float(a.dy[(long long)row*a.n+n]),
                       float(a.weight[((long long)lo*a.n+n)*a.k+k]), sum);
        a.dx[(long long)row*a.k+k] = hip_bfloat16(sum);
    }
}
#else
extern "C" __global__ void opus_gemm_gfx950_bf16_latent_moe_bwd_scalar_tn_f32(Args a) {
    int e = blockIdx.y, n = blockIdx.x;
    for (int k = threadIdx.x; k < a.k; k += blockDim.x) {
        float sum = 0;
        for (int r = a.offsets[e]; r < a.offsets[e+1]; ++r)
            sum = fmaf(float(a.dy[(long long)r*a.n+n]), float(a.x[(long long)r*a.k+k]), sum);
        a.dw[((long long)e*a.n+n)*a.k+k] = sum;
    }
}
#endif
