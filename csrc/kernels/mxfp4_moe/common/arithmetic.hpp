// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <hip/hip_runtime.h>

template <int x, int y> constexpr __device__ __host__ inline int exact_div() {
    static_assert(x % y == 0);
    static_assert(x >= y);
    return x / y;
}

constexpr __device__ __host__ inline int ceil_div(int x, int y) { return (x + y - 1) / y; }

template <int x, int y> constexpr __device__ __host__ inline int round_up() {
    static_assert(y > 0 && (y & (y - 1)) == 0, "round_up: y must be a power of 2");
    constexpr int mask = y - 1;
    return (x + mask) & ~mask;
}

__device__ __host__ inline int round_up(int x, int y) {
    int mask = y - 1;
    return (x + mask) & ~mask;
}

template <int x, int y> constexpr __device__ __host__ inline int round_down() {
    static_assert(y > 0 && (y & (y - 1)) == 0, "round_down: y must be a power of 2");
    constexpr int mask = y - 1;
    return x & ~mask;
}

__device__ __host__ inline int round_down(int x, int y) {
    int mask = y - 1;
    return x & ~mask;
}

__device__ __host__ static constexpr int compute_k_shift(int K_TILES) {
    int s = 0;
    while ((1 << s) < K_TILES)
        s++;
    return s;
};