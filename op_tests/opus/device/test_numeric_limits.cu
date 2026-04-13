// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Device test for opus::numeric_limits.
// Single-thread kernel writes min/max/lowest/quiet_nan/infinity as uint32 bit patterns.

#ifdef __HIP_DEVICE_COMPILE__
// ── Device pass ─────────────────────────────────────────────────────────────
#include "opus/opus.hpp"
using namespace opus;
namespace {

template<typename T>
__device__ unsigned int to_bits(T v) {
    if constexpr (sizeof(T) == 1) return static_cast<unsigned int>(__builtin_bit_cast(unsigned char, v));
    else if constexpr (sizeof(T) == 2) return static_cast<unsigned int>(__builtin_bit_cast(unsigned short, v));
    else return __builtin_bit_cast(unsigned int, v);
}

template<typename T>
__device__ void write_limits(unsigned int* out) {
    out[0] = to_bits(numeric_limits<T>::min());
    out[1] = to_bits(numeric_limits<T>::max());
    out[2] = to_bits(numeric_limits<T>::lowest());
    out[3] = to_bits(numeric_limits<T>::quiet_nan());
    out[4] = to_bits(numeric_limits<T>::infinity());
}

__global__ void numeric_limits_kernel(unsigned int* out) {
    if (__builtin_amdgcn_workitem_id_x() != 0) return;
    write_limits<fp32_t>(out +  0);
    write_limits<fp16_t>(out +  5);
    write_limits<bf16_t>(out + 10);
    write_limits<fp8_t >(out + 15);
    write_limits<bf8_t >(out + 20);
    write_limits<i32_t >(out + 25);
    write_limits<u32_t >(out + 30);
    write_limits<i16_t >(out + 35);
#if __clang_major__ >= 20
    write_limits<u16_t >(out + 40);
#endif
    write_limits<i8_t  >(out + 45);
    write_limits<u8_t  >(out + 50);
}
} // anonymous namespace

#else
// ── Host pass ───────────────────────────────────────────────────────────────
// #include <hip/hip_runtime.h>   // replaced by hip_host_minimal.h for faster builds
#include "hip_host_minimal.h"
#include <cstdio>

namespace {
__global__ void numeric_limits_kernel(unsigned int* out) {}
}

extern "C" void run_numeric_limits(void* d_out) {
    numeric_limits_kernel<<<1, 1>>>(static_cast<unsigned int*>(d_out));
    hipError_t err = hipDeviceSynchronize();
    if (err != hipSuccess) {
        fprintf(stderr, "numeric_limits_kernel failed: %s\n", hipGetErrorString(err));
    }
}
#endif
