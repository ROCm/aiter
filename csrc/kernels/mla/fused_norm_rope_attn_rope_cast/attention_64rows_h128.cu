// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// Compile the H128 kernels independently from H64, using the same launch templates.
#include "attention_64rows.cuh"

template hipError_t run_fused_norm_rope_attn_rope_cast_heads<128>(
    const FusedNormRopeAttnRopeCastParams& args, bool raw_q, hipStream_t stream);
