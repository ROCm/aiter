// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

// HIP runtime split. Three include modes:
//
//   * __HIP_DEVICE_COMPILE__ (device pass, any TU): hip_minimal only.
//   * __HIPCC_RTC__ (RTC mode, both passes of a TU built with
//     -D__HIPCC_RTC__, i.e. the codegen-emitted .device.cu files):
//     hip_minimal only. Their host pass has no kernel body and only
//     needs the bare keyword fallbacks; the full <hip/hip_runtime.h>
//     would be a 100K-line waste and would also pull in
//     <hip/amd_detail/hip_fp16.h>, which depends on the wrapper that
//     RTC short-circuits.
//   * Otherwise (host pass of a non-RTC TU, e.g. all_instances_host.cu
//     or opus_gemm.cu): full runtime, because ATen / ck_tile /
//     pybind11 expect it.
//
// hip_bf16 / hip_fp8 stay in the full-runtime branch only: opus.hpp
// defines bf16_t / fp8_t via compiler built-ins (__bf16, _BitInt(8))
// so the legacy HIP type aliases are dead code on the device side.
#if defined(__HIP_DEVICE_COMPILE__) || defined(__HIPCC_RTC__)
#include <opus/hip_minimal.hpp>
#else
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp8.h>
#endif

#include <opus/opus.hpp>

using fp8_t = opus::fp8_t;
using bf16_t = opus::bf16_t;
using fp32_t = opus::fp32_t;
using opus::operator""_I;

// CHECK_HIP / CHECK_HIP_KERNEL_LAUNCH are host-only diagnostics; the
// macros are only expanded at host call sites (no kernel body uses
// them) so we can leave the macros visible to both passes without
// hurting device-side parsing.
#define CHECK_HIP(call)                                                                                   \
    do {                                                                                                  \
        hipError_t status_ = call;                                                                        \
        if (status_ != hipSuccess) {                                                                      \
            fprintf(stderr, "HIP error (%s:%d): %s\n", __FILE__, __LINE__, hipGetErrorString(status_));   \
            exit(1);                                                                                      \
        }                                                                                                 \
    } while(0)

#define CHECK_HIP_KERNEL_LAUNCH() CHECK_HIP(hipGetLastError())

__host__ __device__ inline int ceil_div(int a, int b) {
    return (a + b - 1) / b;
}

__host__ __device__ constexpr inline int ceil_div_constexpr(int a, int b) {
    return (a + b - 1) / b;
}

#ifdef __HIP_DEVICE_COMPILE__
// ── Device-only utilities (skipped on host pass to reduce parse time) ──

#define MFMA_MASK 0x08
#define VALU_MASK 0x02

#define SCHED_BARRIER(mask, cnt, group) __builtin_amdgcn_sched_group_barrier(mask, cnt, group)

template<int Pairs, int VALU_CNT, int Group>
__device__ __forceinline__ void sched_barrier_pairs() {
    SCHED_BARRIER(MFMA_MASK, 1, Group);
    SCHED_BARRIER(VALU_MASK, VALU_CNT, Group);
    if constexpr (Pairs > 1) sched_barrier_pairs<Pairs - 1, VALU_CNT, Group>();
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

#endif // __HIP_DEVICE_COMPILE__
