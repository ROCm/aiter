#pragma once

#include <cstdio>
#include <cmath>
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include "mhc_layer.h"

namespace mhc {

template<int BLOCK_SIZE>
__global__ void float_to_bf16_kernel(__hip_bfloat16* __restrict__ out, const float* __restrict__ inp,
                                     int size) {
    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (idx < size) {
        out[idx] = (__hip_bfloat16)inp[idx];
    }
}

template<int BLOCK_SIZE>
__global__ void bf16_to_float_kernel(float* __restrict__ out, const __hip_bfloat16* __restrict__ inp,
                                     int size) {
    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (idx < size) {
        out[idx] = (float)inp[idx];
    }
}

inline void float_to_bf16(__hip_bfloat16* out, const float* inp, int size, hipStream_t stream = nullptr) {
    constexpr int BLOCK_SIZE = 256;
    int num_blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    float_to_bf16_kernel<BLOCK_SIZE><<<num_blocks, BLOCK_SIZE, 0, stream>>>(out, inp, size);
}

inline void bf16_to_float(float* out, const __hip_bfloat16* inp, int size, hipStream_t stream = nullptr) {
    constexpr int BLOCK_SIZE = 256;
    int num_blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    bf16_to_float_kernel<BLOCK_SIZE><<<num_blocks, BLOCK_SIZE, 0, stream>>>(out, inp, size);
}

__device__ __forceinline__ float fast_exp(float x) {
    return __expf(x);
}

__device__ __forceinline__ float fast_sigmoid(float x) {
    return 1.0f / (1.0f + fast_exp(-x));
}

__device__ __forceinline__ __hip_bfloat162 mhc_floats2bfloat162(float x, float y) {
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__) || defined(__HIP_DEVICE_COMPILE__)
    __hip_bfloat162 out;
    out.x = __float2bfloat16(x);
    out.y = __float2bfloat16(y);
    return out;
#else
    return mhc_floats2bfloat162(x, y);
#endif
}


} // namespace mhc
