// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#ifdef USE_ROCM

#include <hip/hip_runtime.h>
#include <functional>
#include <tuple>
#include <unordered_map>

// ---------------------------------------------------------------------------
// GemmDispatchHash
//
// Hash for the (cu_num, M, N, K) 4-tuple used as the C++ runtime dispatch key
// in all CK GEMM modules.  Replaces the per-module IntTupleHash that used plain
// XOR (commutative — shape permutations collide).  Uses boost-style mixing with
// the golden-ratio constant (0x9e3779b9) for a non-commutative, low-collision hash.
// ---------------------------------------------------------------------------
struct GemmDispatchHash
{
    size_t operator()(const std::tuple<int, int, int, int>& t) const
    {
        size_t h = std::hash<int>{}(std::get<0>(t));
        h ^= std::hash<int>{}(std::get<1>(t)) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int>{}(std::get<2>(t)) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int>{}(std::get<3>(t)) + 0x9e3779b9 + (h << 6) + (h >> 2);
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
// Used as the first element of the (cu_num, M, N, K) dispatch key so that a
// multi-arch .so (built for e.g. gfx942 and gfx950) can select the correct
// tuned kernel at runtime without recompilation.
//
// Note: multi-GPU processes where different devices have different
// multiProcessorCounts are not a supported inference scenario; the cached value
// reflects whichever device is current at the time of the first dispatch call.
// ---------------------------------------------------------------------------
inline int get_device_cu_num()
{
    static const int cu_num = []() {
        hipDeviceProp_t prop{};
        int device = 0;
        hipGetDevice(&device);
        hipGetDeviceProperties(&prop, device);
        return prop.multiProcessorCount;
    }();
    return cu_num;
}

// ---------------------------------------------------------------------------
// GemmDispatchMap
//
// Convenience alias for the (cu_num, M, N, K)-keyed dispatch map type.
// Each module instantiates this with its own RowwiseKernel / BlockwiseKernel
// function type:
//
//   using RowwiseKernelMap = GemmDispatchMap<RowwiseKernel>;
// ---------------------------------------------------------------------------
template <typename KernelFn>
using GemmDispatchMap =
    std::unordered_map<std::tuple<int, int, int, int>, KernelFn, GemmDispatchHash>;

// ---------------------------------------------------------------------------
// BatchedGemmDispatchHash
//
// Hash for the (cu_num, B, M, N, K) 5-tuple used as the C++ runtime dispatch
// key in batched CK GEMM modules.  Same boost-style mixing as GemmDispatchHash
// (fixes the commutative plain-XOR bug in the old IntTupleHash).
// ---------------------------------------------------------------------------
struct BatchedGemmDispatchHash
{
    size_t operator()(const std::tuple<int, int, int, int, int>& t) const
    {
        size_t h = std::hash<int>{}(std::get<0>(t));
        h ^= std::hash<int>{}(std::get<1>(t)) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int>{}(std::get<2>(t)) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int>{}(std::get<3>(t)) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int>{}(std::get<4>(t)) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};

// ---------------------------------------------------------------------------
// BatchedGemmDispatchMap
//
// Convenience alias for the (cu_num, B, M, N, K)-keyed dispatch map type.
// Used by batched GEMM modules:
//
//   using BatchedRowwiseKernelMap = BatchedGemmDispatchMap<BatchedRowwiseKernel>;
// ---------------------------------------------------------------------------
template <typename KernelFn>
using BatchedGemmDispatchMap =
    std::unordered_map<std::tuple<int, int, int, int, int>, KernelFn, BatchedGemmDispatchHash>;

#endif // USE_ROCM
