// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
// Torch-free TU: define AITER_NO_TORCH_TYPES before topk_gating_kernels.cuh so its
// aiter_opus_plus.h include does not pull in the c10 half/bfloat16 headers.
#define AITER_NO_TORCH_TYPES
#include "topk_gating_kernels.cuh"

AITER_TOPK_GATING_INSTANTIATE(aiter::SCORE_SIGMOID)
