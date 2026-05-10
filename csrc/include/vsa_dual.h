// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Public C++ entry point for the FP4 VSA dual-warp-set attention kernel
// (loads /home/aiter/hsa/gfx950/vsa/vsa_dual_setprio_dropB.co at first call).

#pragma once

#include <torch/extension.h>

// Launches `fp4_vsa_dual_kernel` (the canonical optC1 build of the
// dual-warp-set FP4 attention kernel).  All tensors must live on the
// current CUDA/HIP device; the kernel writes into `out`, `lse`, and
// `counters` (the latter is auto-zeroed on every call).
//
// Numerical contract vs the reference `setprio_exp` baseline:
//   - cosine similarity:  1.000000
//   - max |diff|:         0.03125  (= 1 bf16 ULP)
//   - within 1 bf16 ULP:  >= 99.97 % of elements
//
// Layout convention:  inputs are flattened to (BH, T, D); pass BH = B * H.
//
// Caller responsibilities:
//   - `lim` and `n_dense` are the L2-aware tile-ordering hints — build
//     them ONCE per (q2k_idx, q2k_num) shape via the Python helper
//     `aiter.build_l2_aware_lim_vsa(...)`.
//   - `counters` must be a contiguous int32 tensor with at least 2
//     elements (used by the kernel as two atomic dispatch counters).
void vsa_dual_dropB(const torch::Tensor& q,
                    const torch::Tensor& k,
                    const torch::Tensor& v,
                    const torch::Tensor& qscale,
                    const torch::Tensor& kscale,
                    const torch::Tensor& vmean,
                    const torch::Tensor& vscale,
                    const torch::Tensor& q2k_idx,
                    const torch::Tensor& q2k_num,
                    const torch::Tensor& vbs,
                    const torch::Tensor& lim,
                    const torch::Tensor& out,
                    const torch::Tensor& lse,
                    const torch::Tensor& counters,
                    int64_t B,
                    int64_t T,
                    int64_t num_q_blks,
                    int64_t max_kv,
                    int64_t n_dense);
