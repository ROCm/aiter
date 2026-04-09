// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#ifdef USE_ROCM

#include "aiter_hip_common.h"
#include <functional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>

// ---------------------------------------------------------------------------
// GemmDispatchHash
//
// Hash for the (gfx, cu_num, M, N, K) 5-tuple used as the C++ runtime
// dispatch key in all CK GEMM modules.  The gfx arch string (e.g. "gfx942")
// is included so that multi-arch .so files containing kernels for two
// architectures that share the same cu_num do not collide.  Uses boost-style
// mixing with the golden-ratio constant (0x9e3779b9) for a non-commutative,
// low-collision hash.
// ---------------------------------------------------------------------------
struct GemmDispatchHash
{
    size_t operator()(const std::tuple<std::string, int, int, int, int>& t) const
    {
        size_t h = std::hash<std::string>{}(std::get<0>(t));
        h ^= std::hash<int>{}(std::get<1>(t)) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int>{}(std::get<2>(t)) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int>{}(std::get<3>(t)) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int>{}(std::get<4>(t)) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};

// ---------------------------------------------------------------------------
// get_device_cu_num
//
// Returns the multiProcessorCount of the current HIP device, cached for the
// lifetime of the process.  Safe to call at first dispatch: the GPU device and
// its CU count are fixed once the process starts.
//
// Note: multi-GPU processes where different devices have different
// multiProcessorCounts are not a supported inference scenario; the cached value
// reflects whichever device is current at the time of the first dispatch call.
// ---------------------------------------------------------------------------
inline int get_device_cu_num()
{
    static const int cu_num = []() {
        int device = -1;
        HIP_CALL(hipGetDevice(&device));
        hipDeviceProp_t prop{};
        HIP_CALL(hipGetDeviceProperties(&prop, device));
        return prop.multiProcessorCount;
    }();
    return cu_num;
}

// ---------------------------------------------------------------------------
// get_device_gfx
//
// Returns the GCN arch name of the current HIP device (e.g. "gfx942"),
// cached for the lifetime of the process.  Delegates to get_gpu_arch() from
// aiter_hip_common.h which strips any :sramecc+:xnack- suffix from gcnArchName.
// ---------------------------------------------------------------------------
inline const std::string& get_device_gfx()
{
    static const std::string gfx = get_gpu_arch();
    return gfx;
}

// ---------------------------------------------------------------------------
// GemmDispatchMap
//
// Convenience alias for the (gfx, cu_num, M, N, K)-keyed dispatch map type.
// Each module instantiates this with its own RowwiseKernel / BlockwiseKernel
// function type:
//
//   using RowwiseKernelMap = GemmDispatchMap<RowwiseKernel>;
// ---------------------------------------------------------------------------
template <typename KernelFn>
using GemmDispatchMap =
    std::unordered_map<std::tuple<std::string, int, int, int, int>, KernelFn, GemmDispatchHash>;

// ---------------------------------------------------------------------------
// BatchedGemmDispatchHash
//
// Hash for the (gfx, cu_num, B, M, N, K) 6-tuple used as the C++ runtime
// dispatch key in batched CK GEMM modules.  Same boost-style mixing as
// GemmDispatchHash.
// ---------------------------------------------------------------------------
struct BatchedGemmDispatchHash
{
    size_t operator()(const std::tuple<std::string, int, int, int, int, int>& t) const
    {
        size_t h = std::hash<std::string>{}(std::get<0>(t));
        h ^= std::hash<int>{}(std::get<1>(t)) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int>{}(std::get<2>(t)) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int>{}(std::get<3>(t)) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int>{}(std::get<4>(t)) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int>{}(std::get<5>(t)) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};

// ---------------------------------------------------------------------------
// BatchedGemmDispatchMap
//
// Convenience alias for the (gfx, cu_num, B, M, N, K)-keyed dispatch map type.
// Used by batched GEMM modules:
//
//   using BatchedRowwiseKernelMap = BatchedGemmDispatchMap<BatchedRowwiseKernel>;
// ---------------------------------------------------------------------------
template <typename KernelFn>
using BatchedGemmDispatchMap =
    std::unordered_map<std::tuple<std::string, int, int, int, int, int>,
                       KernelFn,
                       BatchedGemmDispatchHash>;

#endif // USE_ROCM
