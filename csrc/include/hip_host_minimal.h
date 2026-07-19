// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

#ifndef HIP_HOST_MINIMAL_H
#define HIP_HOST_MINIMAL_H

/**
 * @file hip_host_minimal.h
 * @brief Minimal HIP host-side declarations for kernel launch and device management.
 *
 * Build-time optimization: replaces the full <hip/hip_runtime.h> (~100K+ lines
 * after preprocessing) with only the dozen declarations actually needed by the
 * host launcher code.  This saves ~500ms per translation unit on the host pass.
 *
 * If you need more HIP APIs than declared here, either add them below or
 * switch back to the full header:
 *     // #include "hip_host_minimal.h"    // fast: minimal declarations
 *     // #include <hip/hip_runtime.h>     // slow: full HIP runtime (~100K lines)
 */

#include <cstddef>   // size_t

// ---------- Error handling ----------
typedef int hipError_t;
#define hipSuccess 0
extern "C" hipError_t hipGetLastError();
extern "C" hipError_t hipDeviceSynchronize();
extern "C" const char* hipGetErrorString(hipError_t error);

// ---------- Memory management ----------
extern "C" hipError_t hipMalloc(void** ptr, size_t size);
extern "C" hipError_t hipFree(void* ptr);
extern "C" hipError_t hipMemset(void* dst, int value, size_t sizeBytes);

// Typed overload so callers can pass e.g. unsigned int** without casting
template <typename T>
inline hipError_t hipMalloc(T** ptr, size_t size) {
    return hipMalloc(reinterpret_cast<void**>(ptr), size);
}

// ---------- dim3 ----------
struct dim3 {
    unsigned int x, y, z;
    constexpr dim3(unsigned int _x = 1, unsigned int _y = 1, unsigned int _z = 1)
        : x(_x), y(_y), z(_z) {}
};

// ---------- Kernel launch ----------
typedef void* hipStream_t;

// The <<<>>> syntax is lowered by the HIP compiler into calls to these two
// internal functions.  Declaring them here is enough -- they are defined in
// the HIP runtime library that we link against.
extern "C" hipError_t __hipPushCallConfiguration(dim3 gridDim, dim3 blockDim,
                                                  size_t sharedMem = 0,
                                                  hipStream_t stream = nullptr);
extern "C" hipError_t __hipPopCallConfiguration(dim3* gridDim, dim3* blockDim,
                                                 size_t* sharedMem,
                                                 hipStream_t* stream);

extern "C" hipError_t hipLaunchKernel(const void* function_address,
                                      dim3 numBlocks, dim3 dimBlocks,
                                      void** args, size_t sharedMemBytes,
                                      hipStream_t stream);

#ifndef hipLaunchKernelGGL
#define hipLaunchKernelGGL(kernel, numBlocks, dimBlocks, sharedMemBytes, stream, ...) \
    kernel<<<numBlocks, dimBlocks, sharedMemBytes, stream>>>(__VA_ARGS__)
#endif

#endif // HIP_HOST_MINIMAL_H
