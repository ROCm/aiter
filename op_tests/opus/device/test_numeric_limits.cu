// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Device test for opus::numeric_limits.
// Single-thread kernel writes min/max/lowest/quiet_nan/infinity as uint32 bit patterns.

#include <hip/hip_runtime.h>
#include "opus/opus.hpp"
#include "test_numeric_limits.h"

using namespace opus;

namespace {

template<typename T>
__device__ uint32_t to_bits(T v) {
    if constexpr (sizeof(T) == 1) return static_cast<uint32_t>(__builtin_bit_cast(uint8_t, v));
    else if constexpr (sizeof(T) == 2) return static_cast<uint32_t>(__builtin_bit_cast(uint16_t, v));
    else return __builtin_bit_cast(uint32_t, v);
}

template<typename T>
__device__ void write_limits(uint32_t* out) {
    out[0] = to_bits(numeric_limits<T>::min());
    out[1] = to_bits(numeric_limits<T>::max());
    out[2] = to_bits(numeric_limits<T>::lowest());
    out[3] = to_bits(numeric_limits<T>::quiet_nan());
    out[4] = to_bits(numeric_limits<T>::infinity());
}

__global__ void numeric_limits_kernel(uint32_t* out) {
    if (threadIdx.x != 0) return;
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

extern "C" void run_numeric_limits(void* d_out) {
    numeric_limits_kernel<<<1, 1>>>(static_cast<uint32_t*>(d_out));
    hipError_t err = hipDeviceSynchronize();
    if (err != hipSuccess) {
        fprintf(stderr, "numeric_limits_kernel failed: %s\n", hipGetErrorString(err));
    }
}
