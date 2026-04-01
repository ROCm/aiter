// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp8.h>

#include <opus/opus.hpp>

using fp8_t = opus::fp8_t;
using bf16_t = opus::bf16_t;
using fp32_t = opus::fp32_t;
using opus::operator""_I;

#define CHECK_HIP(call)                                                                                   \
    do {                                                                                                  \
        hipError_t status_ = call;                                                                        \
        if (status_ != hipSuccess) {                                                                      \
            fprintf(stderr, "HIP error (%s:%d): %s\n", __FILE__, __LINE__, hipGetErrorString(status_));   \
            exit(1);                                                                                      \
        }                                                                                                 \
    } while(0)

#define CHECK_HIP_KERNEL_LAUNCH() CHECK_HIP(hipGetLastError())

#define MFMA_MASK 0x08
#define VALU_MASK 0x02

#define SCHED_BARRIER(mask, cnt, group) __builtin_amdgcn_sched_group_barrier(mask, cnt, group)

template<int Pairs, int VALU_CNT, int Group>
__device__ __forceinline__ void sched_barrier_pairs() {
    SCHED_BARRIER(MFMA_MASK, 1, Group);
    SCHED_BARRIER(VALU_MASK, VALU_CNT, Group);
    if constexpr (Pairs > 1) sched_barrier_pairs<Pairs - 1, VALU_CNT, Group>();
}

__host__ __device__ inline int ceil_div(int a, int b) {
    return (a + b - 1) / b;
}

__host__ __device__ constexpr inline int ceil_div_constexpr(int a, int b) {
    return (a + b - 1) / b;
}

template<int E_M, int E_N, int ELEM_C, typename D_ACC, typename D_SF>
inline __device__ void scale_c_tile(
    const opus::vector_t<D_ACC, E_M * E_N * ELEM_C>& c_mma,
    const opus::vector_t<D_SF, E_M>& scale_a,
    const opus::vector_t<D_SF, 1_I>& scale_b,
    opus::vector_t<D_ACC, E_M * E_N * ELEM_C>& acc) {
    constexpr int row_len = E_N * ELEM_C;
    D_SF sfb = opus::get<0>(scale_b);
    opus::vector_t<D_ACC, E_M> row_scales;
    opus::static_for<E_M>([&](auto row) {
        row_scales[decltype(row)::value] = opus::get<decltype(row)::value>(scale_a) * sfb;
    });

    opus::static_for<E_M>([&](auto row) {
        constexpr int start = decltype(row)::value * row_len;
        D_ACC row_scale = opus::get<decltype(row)::value>(row_scales);
        opus::static_for<row_len>([&](auto j) {
            acc[start + j.value] += c_mma[start + j.value] * row_scale;
        });
    });
}
