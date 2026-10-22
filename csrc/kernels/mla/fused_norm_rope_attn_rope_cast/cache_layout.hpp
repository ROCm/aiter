// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once
#include <cstdint>
#include <hip/hip_runtime.h>

// page_size == 0 preserves the legacy tagged byte-address format.
// PageSize: 0 = legacy, -1 = runtime page size, positive = specialized page size.
// Paged mode consumes int32 slots. The two bases may alias one allocation.
struct Fp8CacheLayout
{
    const uint8_t* scales    = nullptr;
    int64_t data_page_stride = 0, scale_page_stride = 0;
    int page_size = 0, num_rows = 0;

    template <int PageSize>
    __device__ __forceinline__ int64_t row(const void* indices, int i) const
    {
        if constexpr(PageSize)
            return static_cast<const int*>(indices)[i];
        else
            return static_cast<const int64_t*>(indices)[i];
    }
    template <int PageSize>
    __device__ __forceinline__ bool valid(int64_t r, int64_t data_bytes) const
    {
        if constexpr(!PageSize)
            return r >= 0 && (r & 7) == 1 && data_bytes >= 528 && (r >> 1) <= data_bytes - 528;
        // Host validation caps num_rows by both byte views before launch.
        return r >= 0 && r < num_rows;
    }
    template <int PageSize>
    __device__ __forceinline__ const uint8_t* data(const void* base, int64_t r) const
    {
        const int block = PageSize > 0 ? PageSize : page_size;
        const int64_t offset =
            PageSize ? (int(r) / block) * data_page_stride + int64_t(int(r) % block) * 512 : r >> 1;
        return static_cast<const uint8_t*>(base) + offset;
    }
    template <int PageSize>
    __device__ __forceinline__ const uint8_t* scale(const void* base, int64_t r) const
    {
        if constexpr(!PageSize)
            return static_cast<const uint8_t*>(base) + (r >> 1) + 512;
        const int block = PageSize > 0 ? PageSize : page_size;
        return scales + (int(r) / block) * scale_page_stride + int64_t(int(r) % block) * 16;
    }
};
