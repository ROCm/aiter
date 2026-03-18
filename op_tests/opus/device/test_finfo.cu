// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Device test for opus::finfo.
// Single-thread kernel writes eps/max/min/tiny as float and bits as int.

#ifdef __HIP_DEVICE_COMPILE__
// ── Device pass ─────────────────────────────────────────────────────────────
#include "opus/opus.hpp"
using namespace opus;
namespace {

// Each type writes 5 floats: eps, max, min, tiny, __int_as_float(bits)
template<typename T>
__device__ void write_finfo(float* out) {
    out[0] = finfo<T>::eps();
    out[1] = finfo<T>::max();
    out[2] = finfo<T>::min();
    out[3] = finfo<T>::tiny();
    out[4] = __builtin_bit_cast(float, finfo<T>::bits);
}

__global__ void finfo_kernel(float* out) {
    if (__builtin_amdgcn_workitem_id_x() != 0) return;
    write_finfo<fp32_t>(out +  0);
    write_finfo<fp16_t>(out +  5);
    write_finfo<bf16_t>(out + 10);
    write_finfo<fp8_t >(out + 15);
    write_finfo<bf8_t >(out + 20);
    write_finfo<fp4_t >(out + 25);
    write_finfo<e8m0_t>(out + 30);
}
} // anonymous namespace

#else
// ── Host pass ───────────────────────────────────────────────────────────────
#include "hip_host_minimal.h"
#include <cstdio>

namespace {
__global__ void finfo_kernel(float* out) {}
}

extern "C" void run_finfo(void* d_out) {
    finfo_kernel<<<1, 1>>>(static_cast<float*>(d_out));
    hipError_t err = hipDeviceSynchronize();
    if (err != hipSuccess) {
        fprintf(stderr, "finfo_kernel failed: %s\n", hipGetErrorString(err));
    }
}
#endif
