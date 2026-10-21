// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// gfx950 block-FP8 MoE: D4096, I1024, top-k 8/9, 1..1024 experts.
//
// out[t] = sum_j topk_w[t][j] * Expert_{topk_ids[t][j]}(x[t]).
// Expert(x) = W2 * (silu(min(gate,L)) * clamp(up,-L,L)), gate/up = W13*x.
// Clamp only when swiglu_limit L > 0. Invalid expert IDs contribute zero.
//
// x/out: BF16 [T,D]; topk_ids/topk_w: int32/FP32 [T,topk].
// w13: E4M3 [E,2I,D], gate rows then up; w2: E4M3 [E,D,I].
// Weights use shuffle_weight(w,(16,16)); s13/s2 are FP32 power-of-two 128x128 scales
// shaped [E,2I/128,D/128] and [E,D/128,I/128]. Only scale exponents are consumed.
// Activations use power-of-two 1x128 scales. Workspace size: hk_moe_workspace_bytes.
// Graph-capturable on the supplied stream.
#pragma once
#include <hip/hip_runtime.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

enum hk_moe_status
{
    HK_MOE_OK            = 0,
    HK_MOE_ERR_LAUNCH    = -1,
    HK_MOE_ERR_EXPERTS   = -2,
    HK_MOE_ERR_WORKSPACE = -3,
    HK_MOE_ERR_TOPK      = -4,
    HK_MOE_ERR_SHAPE     = -5,
};

size_t hk_moe_workspace_bytes(int T, int topk, int num_experts);

int hk_moe_forward(const void* x,
                   void* out,
                   int T,
                   int hidden,
                   int inter,
                   const int* topk_ids,
                   const float* topk_w,
                   int topk,
                   const void* w13,
                   const float* s13,
                   const void* w2,
                   const float* s2,
                   int num_experts,
                   float swiglu_limit,
                   void* workspace,
                   size_t ws_bytes,
                   hipStream_t stream);

#ifdef __cplusplus
}
#endif
