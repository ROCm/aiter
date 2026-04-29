// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

#ifndef HIP_MINIMAL_HPP
#define HIP_MINIMAL_HPP

/**
 * @file opus/hip_minimal.hpp
 * @brief Minimal HIP replacement for <hip/hip_runtime.h>.
 *
 * Replaces <hip/hip_runtime.h> (~100K+ preprocessed lines) on BOTH passes
 * for opus-based kernels:
 *   * host pass: dim3, hipError_t, hipMalloc, hipLaunchKernelGGL, etc.
 *   * device pass: __launch_bounds__ / __global__ / __device__ /
 *     __forceinline__ keyword fallbacks, plus the HIP "magic" globals
 *     threadIdx / blockIdx / blockDim / gridDim implemented via
 *     __builtin_amdgcn_* (no HIP runtime header dependency).
 *
 * Usage:
 *   #include <opus/hip_minimal.hpp>   // both passes — drop-in replacement
 *
 * For higher-level device intrinsics (cross-lane shuffle, lane_id, etc.)
 * include <opus/opus.hpp>.
 *
 * Compile: hipcc kernel.cu -I<aiter_root>/csrc/include -D__HIPCC_RTC__ ...
 */

// ========== Attribute keyword fallbacks (both passes) ==========
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
#ifndef __forceinline__
#define __forceinline__ inline __attribute__((always_inline))
#endif
#ifndef __noinline__
#define __noinline__ __attribute__((noinline))
#endif

// ========== Device-side "magic" globals (threadIdx / blockIdx / blockDim / gridDim) ==========
//
// HIP's <hip/hip_runtime.h> exposes threadIdx.x / blockIdx.x / etc. via a
// fairly large __declspec(property) trick. That header alone is ~100K
// preprocessed lines, which dominates the device-pass parse time on
// every instance TU. Replicating just the four magic globals here lets
// us skip the full runtime header on the device pass.
//
// Backed by __builtin_amdgcn_* directly so no HIP runtime symbols are
// referenced. Behavior is identical to HIP's: `blockIdx.x` returns
// the workgroup id along the X dim, etc.
//
// Guarded on HIP_INCLUDE_HIP_HIP_RUNTIME_API_H so callers that already
// pulled in <hip/hip_runtime.h> get the real HIP definitions instead.
#if defined(__HIP_DEVICE_COMPILE__) && !defined(HIP_INCLUDE_HIP_HIP_RUNTIME_API_H)
namespace opus_hip_minimal_detail {

struct __opus_threadIdx_t {
    __device__ __forceinline__ unsigned int get_x() const { return __builtin_amdgcn_workitem_id_x(); }
    __device__ __forceinline__ unsigned int get_y() const { return __builtin_amdgcn_workitem_id_y(); }
    __device__ __forceinline__ unsigned int get_z() const { return __builtin_amdgcn_workitem_id_z(); }
    __declspec(property(get = get_x)) unsigned int x;
    __declspec(property(get = get_y)) unsigned int y;
    __declspec(property(get = get_z)) unsigned int z;
};

struct __opus_blockIdx_t {
    __device__ __forceinline__ unsigned int get_x() const { return __builtin_amdgcn_workgroup_id_x(); }
    __device__ __forceinline__ unsigned int get_y() const { return __builtin_amdgcn_workgroup_id_y(); }
    __device__ __forceinline__ unsigned int get_z() const { return __builtin_amdgcn_workgroup_id_z(); }
    __declspec(property(get = get_x)) unsigned int x;
    __declspec(property(get = get_y)) unsigned int y;
    __declspec(property(get = get_z)) unsigned int z;
};

struct __opus_blockDim_t {
    __device__ __forceinline__ unsigned int get_x() const { return __builtin_amdgcn_workgroup_size_x(); }
    __device__ __forceinline__ unsigned int get_y() const { return __builtin_amdgcn_workgroup_size_y(); }
    __device__ __forceinline__ unsigned int get_z() const { return __builtin_amdgcn_workgroup_size_z(); }
    __declspec(property(get = get_x)) unsigned int x;
    __declspec(property(get = get_y)) unsigned int y;
    __declspec(property(get = get_z)) unsigned int z;
};

struct __opus_gridDim_t {
    __device__ __forceinline__ unsigned int get_x() const { return __builtin_amdgcn_grid_size_x() / __builtin_amdgcn_workgroup_size_x(); }
    __device__ __forceinline__ unsigned int get_y() const { return __builtin_amdgcn_grid_size_y() / __builtin_amdgcn_workgroup_size_y(); }
    __device__ __forceinline__ unsigned int get_z() const { return __builtin_amdgcn_grid_size_z() / __builtin_amdgcn_workgroup_size_z(); }
    __declspec(property(get = get_x)) unsigned int x;
    __declspec(property(get = get_y)) unsigned int y;
    __declspec(property(get = get_z)) unsigned int z;
};

} // namespace opus_hip_minimal_detail

// Match HIP runtime exactly: external linkage + weak so multiple TUs that
// each reach this header don't ODR-clash. const because the underlying
// builtins are pure reads of intrinsic state. The empty-brace initializer
// is required by the language for const variables; the structs have no
// state so no actual storage is generated.
extern const __device__ __attribute__((weak)) opus_hip_minimal_detail::__opus_threadIdx_t threadIdx;
extern const __device__ __attribute__((weak)) opus_hip_minimal_detail::__opus_blockIdx_t  blockIdx;
extern const __device__ __attribute__((weak)) opus_hip_minimal_detail::__opus_blockDim_t  blockDim;
extern const __device__ __attribute__((weak)) opus_hip_minimal_detail::__opus_gridDim_t   gridDim;

#endif // __HIP_DEVICE_COMPILE__ && !HIP_INCLUDE_HIP_HIP_RUNTIME_API_H

// ========== Host-side declarations (guarded to coexist with <hip/hip_runtime.h>) ==========
#if !defined(HIP_INCLUDE_HIP_HIP_RUNTIME_API_H)

#include <cstddef>   // size_t

typedef int hipError_t;
typedef void* hipStream_t;
#define hipSuccess 0

struct dim3 {
    unsigned int x, y, z;
    constexpr dim3(unsigned int _x = 1, unsigned int _y = 1, unsigned int _z = 1)
        : x(_x), y(_y), z(_z) {}
};

// Error handling
extern "C" hipError_t hipGetLastError();
extern "C" hipError_t hipDeviceSynchronize();
extern "C" const char* hipGetErrorString(hipError_t error);

// Memory management
extern "C" hipError_t hipMalloc(void** ptr, size_t size);
extern "C" hipError_t hipFree(void* ptr);
extern "C" hipError_t hipMemset(void* dst, int value, size_t sizeBytes);
enum hipMemcpyKind { hipMemcpyHostToHost = 0, hipMemcpyHostToDevice = 1, hipMemcpyDeviceToHost = 2, hipMemcpyDeviceToDevice = 3, hipMemcpyDefault = 4 };
extern "C" hipError_t hipMemcpy(void* dst, const void* src, size_t sizeBytes, hipMemcpyKind kind);
template <typename T> inline hipError_t hipMalloc(T** ptr, size_t size) { return hipMalloc(reinterpret_cast<void**>(ptr), size); }

// Events (timing)
typedef void* hipEvent_t;
extern "C" hipError_t hipEventCreate(hipEvent_t* event);
extern "C" hipError_t hipEventDestroy(hipEvent_t event);
extern "C" hipError_t hipEventRecord(hipEvent_t event, hipStream_t stream = nullptr);
extern "C" hipError_t hipEventSynchronize(hipEvent_t event);
extern "C" hipError_t hipEventElapsedTime(float* ms, hipEvent_t start, hipEvent_t stop);

// Kernel launch (<<<>>> syntax)
extern "C" hipError_t __hipPushCallConfiguration(dim3 gridDim, dim3 blockDim, size_t sharedMem = 0, hipStream_t stream = nullptr);
extern "C" hipError_t __hipPopCallConfiguration(dim3* gridDim, dim3* blockDim, size_t* sharedMem, hipStream_t* stream);
extern "C" hipError_t hipLaunchKernel(const void* function_address, dim3 numBlocks, dim3 dimBlocks, void** args, size_t sharedMemBytes, hipStream_t stream);
#ifndef hipLaunchKernelGGL
#define hipLaunchKernelGGL(kernel, numBlocks, dimBlocks, sharedMemBytes, stream, ...) \
    kernel<<<numBlocks, dimBlocks, sharedMemBytes, stream>>>(__VA_ARGS__)
#endif

#endif // !HIP_INCLUDE_HIP_HIP_RUNTIME_API_H

#endif // HIP_MINIMAL_HPP
