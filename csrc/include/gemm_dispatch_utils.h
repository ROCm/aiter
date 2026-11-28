// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#ifdef USE_ROCM

#include "aiter_hip_common.h"
#include <atomic>
#include <cstddef>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <vector>

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
// get_tuned_cu_nums
//
// CU counts whose tuned rows serve the current device, in lookup order: the
// device CU count, then the CU count of the full chip when the device is a
// compute partition (DPX, QPX, CPX). HIP does not report the partition, so
// Python publishes it. When aiter loads a module that defines
// set_tuned_cu_fallback, it calls set_tuned_cu_fallback(gfx, cu_num,
// full_chip_cu_num) for each visible partition (get_tuned_cu_fallbacks in
// aiter/jit/utils/chip_info.py). An entry is keyed by gfx and CU count, so
// each device finds its own entry. A device without an entry uses only its
// own CU count. Each thread resolves a device once, and again after a new
// entry, so a dispatch takes no lock.
// ---------------------------------------------------------------------------
struct TunedCuFallback
{
    std::string gfx;
    int cu_num;
    int full_chip_cu_num;
};

struct TunedCuFallbacks
{
    std::mutex mu;
    std::vector<TunedCuFallback> entries;
    std::atomic<int> version{0};
};

inline TunedCuFallbacks& tuned_cu_fallbacks()
{
    static TunedCuFallbacks fallbacks;
    return fallbacks;
}

inline void set_tuned_cu_fallback(const std::string& gfx, int cu_num, int full_chip_cu_num)
{
    auto& fallbacks = tuned_cu_fallbacks();
    std::lock_guard<std::mutex> guard(fallbacks.mu);
    for(auto& entry : fallbacks.entries)
    {
        if(entry.gfx == gfx && entry.cu_num == cu_num)
        {
            entry.full_chip_cu_num = full_chip_cu_num;
            fallbacks.version.fetch_add(1, std::memory_order_release);
            return;
        }
    }
    fallbacks.entries.push_back({gfx, cu_num, full_chip_cu_num});
    fallbacks.version.fetch_add(1, std::memory_order_release);
}

struct TunedCuNums
{
    int cu_nums[2];
    int size;

    const int* begin() const { return cu_nums; }
    const int* end() const { return cu_nums + size; }
};

inline TunedCuNums get_tuned_cu_nums()
{
    struct Resolved
    {
        int version = -1;
        TunedCuNums cu_nums{};
    };
    thread_local std::unordered_map<int, Resolved> resolved;
    int device = -1;
    HIP_CALL(hipGetDevice(&device));
    auto& fallbacks   = tuned_cu_fallbacks();
    const int version = fallbacks.version.load(std::memory_order_acquire);
    Resolved& cached  = resolved[device];
    if(cached.version == version)
        return cached.cu_nums;

    const int cu_num           = get_device_cu_num();
    const std::string_view gfx = get_device_gfx();
    TunedCuNums result{{cu_num, 0}, 1};
    {
        std::lock_guard<std::mutex> guard(fallbacks.mu);
        for(const auto& entry : fallbacks.entries)
        {
            if(entry.gfx == gfx && entry.cu_num == cu_num && entry.full_chip_cu_num > 0 &&
               entry.full_chip_cu_num != cu_num)
            {
                result.cu_nums[1] = entry.full_chip_cu_num;
                result.size       = 2;
                break;
            }
        }
    }
    cached = {version, result};
    return result;
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

// ---------------------------------------------------------------------------
// find_tuned_row
//
// The row of the first CU count in cu_nums that has a row for this shape
// and that accept(row) accepts, or lookup.end(). shape is the key of the map
// without gfx and cu_num.
// ---------------------------------------------------------------------------
template <typename Map, typename Accept, typename... Shape>
typename Map::const_iterator find_tuned_row_if(const Map& lookup,
                                               const TunedCuNums& cu_nums,
                                               Accept&& accept,
                                               std::string_view gfx,
                                               Shape... shape)
{
    for(const int cu_num : cu_nums)
    {
        auto it = lookup.find(typename Map::key_type{gfx, cu_num, shape...});
        if(it != lookup.end() && accept(it->second))
            return it;
    }
    return lookup.end();
}

template <typename Map, typename... Shape>
typename Map::const_iterator
find_tuned_row(const Map& lookup, const TunedCuNums& cu_nums, std::string_view gfx, Shape... shape)
{
    return find_tuned_row_if(lookup, cu_nums, [](const auto&) { return true; }, gfx, shape...);
}

#endif // USE_ROCM
