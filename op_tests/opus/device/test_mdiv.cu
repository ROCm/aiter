// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Device test for opus::mdiv magic division.
// Each thread computes quotient and remainder via mdiv::divmod and writes to output.

#include <hip/hip_runtime.h>
#include "opus/opus.hpp"
#include "test_mdiv.h"

using namespace opus;

static constexpr int BLOCK_SIZE = 256;

__global__ void mdiv_kernel(const uint32_t* __restrict__ dividends,
                            uint32_t* __restrict__ out_q,
                            uint32_t* __restrict__ out_r,
                            mdiv magic, int n)
{
    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (idx < n) {
        uint32_t q, r;
        magic.divmod(dividends[idx], q, r);
        out_q[idx] = q;
        out_r[idx] = r;
    }
}

extern "C" void run_mdiv(const void* d_dividends, void* d_out_q, void* d_out_r,
                          int divisor, int n)
{
    mdiv magic(static_cast<uint32_t>(divisor));
    int grid = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    mdiv_kernel<<<grid, BLOCK_SIZE>>>(
        static_cast<const uint32_t*>(d_dividends),
        static_cast<uint32_t*>(d_out_q),
        static_cast<uint32_t*>(d_out_r),
        magic, n);
    hipError_t err = hipDeviceSynchronize();
    if (err != hipSuccess) {
        fprintf(stderr, "mdiv_kernel failed: %s\n", hipGetErrorString(err));
    }
}
