/**
 * !!! NB !!!
 * This approach is borrowed and extended from the 3rd-party Eigen wrapper/implementation in PyTorch.
 * It is NOT smart, NOT extensible, NOT flexiable, NOT adaptive,
 * but straightforward and good for early iterations of development.
 * Smarter ones are expected to be implemented in futuer iterations.
 */
#pragma once

// To workaround some unexpected HIPify behavior
#if defined(__HIPCC__) || (defined(__clang__) && defined(__HIP__)) || defined(__HIPCC_RTC__)

#define gpuStream_t hipStream_t
#define gpuDeviceProp_t hipDeviceProp_t
#define gpuError_t hipError_t
#define gpuSuccess hipSuccess
#define gpuErrorNotReady hipErrorNotReady
#define gpuGetDeviceCount hipGetDeviceCount
#define gpuGetLastError hipGetLastError
#define gpuPeekAtLastError hipPeekAtLastError
#define gpuGetErrorName hipGetErrorName
#define gpuGetErrorString hipGetErrorString
#define gpuGetDeviceProperties hipGetDeviceProperties
#define gpuStreamDefault hipStreamDefault
#define gpuGetDevice hipGetDevice
#define gpuSetDevice hipSetDevice
#define gpuMalloc hipMalloc
#define gpuFree hipFree
#define gpuMemsetAsync hipMemsetAsync
#define gpuMemcpyAsync hipMemcpyAsync
#define gpuMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#define gpuMemcpyDeviceToHost hipMemcpyDeviceToHost
#define gpuMemcpyHostToDevice hipMemcpyHostToDevice
#define gpuStreamQuery hipStreamQuery
#define gpuSharedMemConfig hipSharedMemConfig
#define gpuDeviceSetSharedMemConfig hipDeviceSetSharedMemConfig
#define gpuStreamSynchronize hipStreamSynchronize
#define gpuDeviceSynchronize hipDeviceSynchronize
#define gpuMemcpy hipMemcpy
#define gpuDeviceGetAttribute hipDeviceGetAttribute
#define gpuFuncSetAttribute hipFuncSetAttribute
#define gpuLaunchKernel hipLaunchKernel
#define gpuFreeHost hipHostFree
#define gpuMallocHost hipHostMalloc
#define gpuOccupancyMaxActiveBlocksPerMultiprocessor hipOccupancyMaxActiveBlocksPerMultiprocessor
#define gpuDevAttrMaxSharedMemoryPerMultiprocessor hipDeviceAttributeMaxSharedMemoryPerMultiprocessor
#define gpuFuncAttributeMaxDynamicSharedMemorySize hipFuncAttributeMaxDynamicSharedMemorySize
#define gpuDevAttrMultiProcessorCount hipDeviceAttributeMultiprocessorCount
#define gpuDevAttrComputeCapabilityMajor hipDeviceAttributeComputeCapabilityMajor
#define gpuDevAttrComputeCapabilityMinor hipDeviceAttributeComputeCapabilityMinor
// float8 Precision Device types
#define __gpu_fp8_e4m3 __hip_fp8_e4m3_fnuz
#define __gpu_fp8_e5m2 __hip_fp8_e5m2_fnuz
#define __gpu_fp8x2_e4m3 __hip_fp8x2_e4m3_fnuz
#define __gpu_fp8x2_e5m2 __hip_fp8x2_e5m2_fnuz
#define __gpu_fp8x2_storage_t __hip_fp8x2_storage_t
#define __gpu_fp8x4_storage_t __hip_fp8x4_storage_t
#define __gpu_fp8x4_e4m3 __hip_fp8x4_e4m3_fnuz
#define __gpu_fp8x4_e5m2 __hip_fp8x4_e5m2_fnuz
// Bfloat16 Precision Device types
// https://github.com/ROCm/ROCm/issues/2534
// #define gpu_bfloat16 hip_bfloat16
#define gpu_bfloat16 __hip_bfloat16
#define __gpu_bfloat16 __hip_bfloat16
#define gpu_bfloat162 __hip_bfloat162
#define __gpu_bfloat162 __hip_bfloat162

// #elif defined(__CUDACC__) || defined(__NVCC__) || (defined(__clang__) && defined(__CUDA__)) || defined(__CUDACC_RTC__)
#else

#define gpuStream_t cudaStream_t
#define gpuDeviceProp_t cudaDeviceProp
#define gpuError_t cudaError_t
#define gpuSuccess cudaSuccess
#define gpuErrorNotReady cudaErrorNotReady
#define gpuGetDeviceCount cudaGetDeviceCount
#define gpuGetLastError cudaGetLastError
#define gpuPeekAtLastError cudaPeekAtLastError
#define gpuGetErrorName cudaGetErrorName
#define gpuGetErrorString cudaGetErrorString
#define gpuGetDeviceProperties cudaGetDeviceProperties
#define gpuStreamDefault cudaStreamDefault
#define gpuGetDevice cudaGetDevice
#define gpuSetDevice cudaSetDevice
#define gpuMalloc cudaMalloc
#define gpuFree cudaFree
#define gpuMemsetAsync cudaMemsetAsync
#define gpuMemcpyAsync cudaMemcpyAsync
#define gpuMemcpyDeviceToDevice cudaMemcpyDeviceToDevice
#define gpuMemcpyDeviceToHost cudaMemcpyDeviceToHost
#define gpuMemcpyHostToDevice cudaMemcpyHostToDevice
#define gpuStreamQuery cudaStreamQuery
#define gpuSharedMemConfig cudaSharedMemConfig
#define gpuDeviceSetSharedMemConfig cudaDeviceSetSharedMemConfig
#define gpuStreamSynchronize cudaStreamSynchronize
#define gpuDeviceSynchronize cudaDeviceSynchronize
#define gpuMemcpy cudaMemcpy
#define gpuDeviceGetAttribute cudaDeviceGetAttribute
#define gpuFuncSetAttribute cudaFuncSetAttribute
#define gpuLaunchKernel cudaLaunchKernel
#define gpuFreeHost cudaFreeHost
#define gpuMallocHost cudaMallocHost
#define gpuOccupancyMaxActiveBlocksPerMultiprocessor cudaOccupancyMaxActiveBlocksPerMultiprocessor
#define gpuDevAttrMaxSharedMemoryPerMultiprocessor cudaDevAttrMaxSharedMemoryPerMultiprocessor
#define gpuFuncAttributeMaxDynamicSharedMemorySize cudaFuncAttributeMaxDynamicSharedMemorySize
#define gpuDevAttrMultiProcessorCount cudaDevAttrMultiProcessorCount
#define gpuDevAttrComputeCapabilityMajor cudaDevAttrComputeCapabilityMajor
#define gpuDevAttrComputeCapabilityMinor cudaDevAttrComputeCapabilityMinor
// float8 Precision Device types
#define __gpu_fp8_e4m3 __nv_fp8_e4m3
#define __gpu_fp8_e5m2 __nv_fp8_e5m2
#define __gpu_fp8x2_e4m3 __nv_fp8x2_e4m3
#define __gpu_fp8x2_e5m2 __nv_fp8x2_e5m2
#define __gpu_fp8x2_storage_t __nv_fp8x2_storage_t
#define __gpu_fp8x4_storage_t __nv_fp8x4_storage_t
#define __gpu_fp8x4_e4m3 __nv_fp8x4_e4m3
#define __gpu_fp8x4_e5m2 __nv_fp8x4_e5m2
// Bfloat16 Precision Device types
#define gpu_bfloat16 nv_bfloat16
#define __gpu_bfloat16 __nv_bfloat16
#define gpu_bfloat162 nv_bfloat162
#define __gpu_bfloat162 __nv_bfloat162

#endif  // __CUDACC__ or __HIPCC__

// `gpu_assert` can be overridden.
#ifndef gpu_assert

#ifdef __HIP_DEVICE_COMPILE__
// HIPCC does not support the use of assert on the GPU side.
#define gpu_assert(COND)
#else
#define gpu_assert(COND) assert(COND)
#endif

#endif  // gpu_assert
