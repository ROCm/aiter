// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

#ifndef HIP_MINIMAL_HPP
#define HIP_MINIMAL_HPP

/**
 * @file opus/hip_minimal.hpp
 * @brief Minimal HIP declarations for both host and device compilation.
 *
 * Replaces <hip/hip_runtime.h> (~100K+ preprocessed lines) with only the
 * declarations actually needed, saving ~500ms per translation unit.
 *
 * - Device pass: __launch_bounds__, __shared__/__device__/__global__,
 *                __all() warp vote intrinsic
 * - Host pass:   dim3, hipLaunchKernelGGL, hipMalloc/hipFree, error handling
 * - Both passes: attribute keyword fallbacks
 *
 * Compile: hipcc kernel.cu -I<aiter_root>/csrc/include -D__HIPCC_RTC__ ...
 */

// ========== Both passes: attribute keyword fallbacks ==========
// Provided by hipcc's implicit wrapper, but define as fallbacks for --genco or standalone builds
#ifndef __launch_bounds__
#define __launch_bounds_impl0__(requiredMaxThreadsPerBlock) \
    __attribute__((amdgpu_flat_work_group_size(1, requiredMaxThreadsPerBlock)))
#define __launch_bounds_impl1__(requiredMaxThreadsPerBlock, minBlocksPerMultiprocessor) \
    __attribute__((amdgpu_flat_work_group_size(1, requiredMaxThreadsPerBlock), \
                   amdgpu_waves_per_eu(minBlocksPerMultiprocessor)))
#define __launch_bounds_select__(_1, _2, impl_, ...) impl_
#define __launch_bounds__(...) \
    __launch_bounds_select__(__VA_ARGS__, __launch_bounds_impl1__, __launch_bounds_impl0__, )(__VA_ARGS__)
#endif

#ifndef __shared__
#define __shared__ __attribute__((shared))
#endif
#ifndef __device__
#define __device__ __attribute__((device))
#endif
#ifndef __global__
#define __global__ __attribute__((global))
#endif
#ifndef __host__
#define __host__ __attribute__((host))
#endif

// ========== Device pass only ==========
#if defined(__HIP_DEVICE_COMPILE__)

// Warp vote intrinsic — __all(predicate) returns non-zero iff predicate is
// non-zero for every active lane in the wavefront
extern "C" __device__ int __ockl_wfall_i32(int);
__device__ inline int __all(int predicate) { return __ockl_wfall_i32(predicate); }

#endif // __HIP_DEVICE_COMPILE__

// ========== Host pass only ==========
#if !defined(__HIP_DEVICE_COMPILE__)

#include <cstddef>   // size_t

// ---------- Error handling ----------
typedef int hipError_t;
typedef void* hipStream_t;
#define hipSuccess 0
extern "C" hipError_t hipGetLastError();
extern "C" hipError_t hipDeviceSynchronize();
extern "C" const char* hipGetErrorString(hipError_t error);

// ---------- Memory management ----------
extern "C" hipError_t hipMalloc(void** ptr, size_t size);
extern "C" hipError_t hipFree(void* ptr);
extern "C" hipError_t hipMemset(void* dst, int value, size_t sizeBytes);
enum hipMemcpyKind { hipMemcpyHostToHost = 0, hipMemcpyHostToDevice = 1, hipMemcpyDeviceToHost = 2, hipMemcpyDeviceToDevice = 3, hipMemcpyDefault = 4 };
extern "C" hipError_t hipMemcpy(void* dst, const void* src, size_t sizeBytes, hipMemcpyKind kind);

template <typename T>
inline hipError_t hipMalloc(T** ptr, size_t size) {
    return hipMalloc(reinterpret_cast<void**>(ptr), size);
}

// ---------- Events (timing) ----------
typedef void* hipEvent_t;
extern "C" hipError_t hipEventCreate(hipEvent_t* event);
extern "C" hipError_t hipEventDestroy(hipEvent_t event);
extern "C" hipError_t hipEventRecord(hipEvent_t event, hipStream_t stream = nullptr);
extern "C" hipError_t hipEventSynchronize(hipEvent_t event);
extern "C" hipError_t hipEventElapsedTime(float* ms, hipEvent_t start, hipEvent_t stop);

// ---------- dim3 ----------
struct dim3 {
    unsigned int x, y, z;
    constexpr dim3(unsigned int _x = 1, unsigned int _y = 1, unsigned int _z = 1)
        : x(_x), y(_y), z(_z) {}
};

// ---------- Kernel launch ----------
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

#endif // !defined(__HIP_DEVICE_COMPILE__)

#endif // HIP_MINIMAL_HPP
