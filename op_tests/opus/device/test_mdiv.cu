// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Device test for opus::mdiv magic division.
// Each thread computes quotient and remainder via mdiv::divmod and writes to output.

#ifdef __HIP_DEVICE_COMPILE__
// ── Device pass ─────────────────────────────────────────────────────────────
#include "opus/opus.hpp"
using namespace opus;
static constexpr int BLOCK_SIZE = 256;

__global__ void mdiv_kernel(const unsigned int* __restrict__ dividends,
                            unsigned int* __restrict__ out_q,
                            unsigned int* __restrict__ out_r,
                            mdiv magic, int n)
{
    int idx = __builtin_amdgcn_workgroup_id_x() * BLOCK_SIZE + __builtin_amdgcn_workitem_id_x();
    if (idx < n) {
        unsigned int q, r;
        magic.divmod(dividends[idx], q, r);
        out_q[idx] = q;
        out_r[idx] = r;
    }
}

#else
// ── Host pass ───────────────────────────────────────────────────────────────
// #include <hip/hip_runtime.h>   // replaced by hip_host_minimal.h for faster builds
#include "hip_host_minimal.h"
#include <cstdio>
#include "opus/opus.hpp"
using namespace opus;
static constexpr int BLOCK_SIZE = 256;

__global__ void mdiv_kernel(const unsigned int* __restrict__ dividends,
                            unsigned int* __restrict__ out_q,
                            unsigned int* __restrict__ out_r,
                            mdiv magic, int n) {}

extern "C" void run_mdiv(const void* d_dividends, void* d_out_q, void* d_out_r,
                          int divisor, int n)
{
    mdiv magic(static_cast<unsigned int>(divisor));
    int grid = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    mdiv_kernel<<<grid, BLOCK_SIZE>>>(
        static_cast<const unsigned int*>(d_dividends),
        static_cast<unsigned int*>(d_out_q),
        static_cast<unsigned int*>(d_out_r),
        magic, n);
    hipError_t err = hipDeviceSynchronize();
    if (err != hipSuccess) {
        fprintf(stderr, "mdiv_kernel failed: %s\n", hipGetErrorString(err));
    }
}
#endif
