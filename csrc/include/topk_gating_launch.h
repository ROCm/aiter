// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Declaration-only interface between the topk_gating entry point and the
// precompiled kernel instantiations.
//
// The kernel templates live in topk_gating_kernels.cuh and are explicitly
// instantiated by the translation units under csrc/kernels/topk_gating_inst/,
// one per (gating dtype, score function) pair. Including this header costs
// nothing, so the entry TU stays cheap to compile while the 27
// (gating dtype x bias dtype x score function) instantiations build in
// parallel across those TUs.
#pragma once

#include <hip/hip_runtime.h>
#include <cstddef>

namespace aiter {

enum { SCORE_SQRTSOFTPLUS = 0, SCORE_SIGMOID = 1, SCORE_SOFTMAX = 2 };

// Launch arguments in type-erased form: the element types are carried by the
// topk_gating_launch template parameters, not by these pointers.
struct topk_gating_params
{
    const void* gating;      // [num_tokens, num_experts], DTYPE_I
    const void* bias;        // [num_experts], DTYPE_B; nullptr when unbiased
    float*      weights;     // [num_tokens, topk]
    int*        ids;         // [num_tokens, topk]
    size_t      stride_tk;
    int         num_experts;
    int         topk;
    int         num_tokens;
    float       routed_scaling_factor;
    bool        need_renorm;
    hipStream_t stream;
};

// Selects and launches the fastest kernel variant for p. Defined in
// topk_gating_kernels.cuh; only the instantiations listed in
// csrc/kernels/topk_gating_inst/ are available to link against.
template <typename DTYPE_I, typename DTYPE_B, int SF>
void topk_gating_launch(const topk_gating_params& p);

} // namespace aiter
