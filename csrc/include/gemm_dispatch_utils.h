// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#ifdef USE_ROCM

#include "aiter_hip_common.h"
#include <cstddef>
#include <functional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>

// ---------------------------------------------------------------------------
// GemmLookupKey
//
// POD dispatch key keyed on (gfx, cu_num, M, N, K).  Keeping it trivially
// destructible and standard-layout lets the generated lookup tables be
// constant-initialized into .data.rel.ro — no per-entry constructor code,
// no exception-cleanup chain in the dispatch lambda.
//
// gfx views must point into storage that outlives the table.  In practice
// table entries point to string literals ("gfx950"); runtime keys point
// into get_device_gfx()'s permanently-cached std::string.
// ---------------------------------------------------------------------------
struct GemmLookupKey
{
    std::string_view gfx;
    int cu_num;
    int M;
    int N;
    int K;
};

static_assert(std::is_trivially_destructible_v<GemmLookupKey>);
static_assert(std::is_standard_layout_v<GemmLookupKey>);

struct GemmLookupKeyHash
{
    size_t operator()(const GemmLookupKey& k) const noexcept
    {
        size_t h = std::hash<std::string_view>{}(k.gfx);
        h ^= std::hash<int>{}(k.cu_num) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int>{}(k.M) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int>{}(k.N) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int>{}(k.K) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};

struct GemmLookupKeyEq
{
    bool operator()(const GemmLookupKey& a, const GemmLookupKey& b) const noexcept
    { return a.cu_num == b.cu_num && a.M == b.M && a.N == b.N && a.K == b.K && a.gfx == b.gfx; }
};

// ---------------------------------------------------------------------------
// get_device_cu_num
//
// Returns the multiProcessorCount of the current HIP device.  Cached per
// device ID via SynchronizedCache so that processes calling hipSetDevice()
// across GPUs with different CU counts always get the correct value.
// ---------------------------------------------------------------------------
inline int get_device_cu_num()
{
    static SynchronizedCache<int, int> cache;
    int device = -1;
    HIP_CALL(hipGetDevice(&device));
    return cache.get_or_create(device, [device]() {
        hipDeviceProp_t prop{};
        HIP_CALL(hipGetDeviceProperties(&prop, device));
        return prop.multiProcessorCount;
    });
}

// ---------------------------------------------------------------------------
// get_device_gfx
//
// Returns the GCN arch name of the current HIP device (e.g. "gfx942").
// Cached per device ID via SynchronizedCache so that processes calling
// hipSetDevice() across GPUs of different architectures always get the
// correct arch string.  Strips any :sramecc+:xnack- suffix from gcnArchName.
//
// Returned by std::string_view because the cached std::string lives for the
// program's lifetime (the cache is a function-local static unordered_map
// that is never erased), so the view is permanently valid.
// ---------------------------------------------------------------------------
inline std::string_view get_device_gfx()
{
    static SynchronizedCache<int, std::string> cache;
    int device = -1;
    HIP_CALL(hipGetDevice(&device));
    return cache.get_or_create(device, [device]() {
        hipDeviceProp_t prop{};
        HIP_CALL(hipGetDeviceProperties(&prop, device));
        std::string arch_full = prop.gcnArchName;
        size_t colon_pos      = arch_full.find(':');
        return colon_pos != std::string::npos ? arch_full.substr(0, colon_pos) : arch_full;
    });
}

// ---------------------------------------------------------------------------
// GemmDispatchMap
//
// Convenience alias for the (gfx, cu_num, M, N, K)-keyed dispatch map type.
// Each module instantiates this with its own raw-function-pointer kernel
// type:
//
//   using RowwiseKernel    = torch::Tensor (*)(torch::Tensor&, ...);
//   using RowwiseKernelMap = GemmDispatchMap<RowwiseKernel>;
//
// KernelFn must be trivially destructible (use a function pointer, not
// std::function) for the constant-init / .rodata optimization to apply.
// ---------------------------------------------------------------------------
template <typename KernelFn>
using GemmDispatchMap =
    std::unordered_map<GemmLookupKey, KernelFn, GemmLookupKeyHash, GemmLookupKeyEq>;

// ---------------------------------------------------------------------------
// BatchedGemmLookupKey
//
// POD dispatch key keyed on (gfx, cu_num, B, M, N, K) — used by batched
// GEMM modules.  Same trivial-destructibility / standard-layout properties
// as GemmLookupKey.
// ---------------------------------------------------------------------------
struct BatchedGemmLookupKey
{
    std::string_view gfx;
    int cu_num;
    int B;
    int M;
    int N;
    int K;
};

static_assert(std::is_trivially_destructible_v<BatchedGemmLookupKey>);
static_assert(std::is_standard_layout_v<BatchedGemmLookupKey>);

struct BatchedGemmLookupKeyHash
{
    size_t operator()(const BatchedGemmLookupKey& k) const noexcept
    {
        size_t h = std::hash<std::string_view>{}(k.gfx);
        h ^= std::hash<int>{}(k.cu_num) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int>{}(k.B) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int>{}(k.M) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int>{}(k.N) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int>{}(k.K) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};

struct BatchedGemmLookupKeyEq
{
    bool operator()(const BatchedGemmLookupKey& a, const BatchedGemmLookupKey& b) const noexcept
    {
        return a.cu_num == b.cu_num && a.B == b.B && a.M == b.M && a.N == b.N && a.K == b.K &&
               a.gfx == b.gfx;
    }
};

// ---------------------------------------------------------------------------
// BatchedGemmDispatchMap
//
// Convenience alias for the (gfx, cu_num, B, M, N, K)-keyed dispatch map
// used by batched GEMM modules:
//
//   using BatchedRowwiseKernelMap = BatchedGemmDispatchMap<BatchedRowwiseKernel>;
// ---------------------------------------------------------------------------
template <typename KernelFn>
using BatchedGemmDispatchMap = std::
    unordered_map<BatchedGemmLookupKey, KernelFn, BatchedGemmLookupKeyHash, BatchedGemmLookupKeyEq>;

#endif // USE_ROCM
