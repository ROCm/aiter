// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// HipKittens block-FP8 MoE forward for gfx950.
//
//   out[t] = sum_j topk_w[t][j] * Expert_{topk_ids[t][j]}(x[t])
//   Expert_e(x) = W2_e (act(W13_e x))
//   act = silu(min(gate, L)) * clamp(up, -L, L) for swiglu_limit L > 0, else silu(gate) * up.
//
// Supported shapes: hidden 4096, intermediate 1024, top-k 8 or 9, up to 1024 experts.
//   x, out        bf16 [T][hidden]
//   topk_ids      int32 [T][topk]   (out-of-range ids contribute nothing)
//   topk_w        fp32  [T][topk]
//   w13           fp8 e4m3 [E][2*inter][hidden]  rows [0, inter) = gate, [inter, 2*inter) = up
//   w2            fp8 e4m3 [E][hidden][inter]
//   s13           fp32 128x128 block scales [E][2*inter/128][hidden/128]
//   s2            fp32 128x128 block scales [E][hidden/128][inter/128]
// Weights use aiter's shuffle_weight(w, (16, 16)) layout. Weight scales must be powers of two
// (UE8M0); only the exponent is used. Activations use power-of-two scales per (token, 128 columns).
// The workspace must hold hk_moe_workspace_bytes(T, topk, E) bytes. All work is enqueued on
// stream and supports graph capture.
#pragma once
#include <hip/hip_runtime.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

enum hk_moe_status
{
    HK_MOE_OK            = 0,
    HK_MOE_ERR_LAUNCH    = -1, // a kernel launch failed (hipGetLastError)
    HK_MOE_ERR_EXPERTS   = -2, // num_experts > 1024
    HK_MOE_ERR_WORKSPACE = -3, // ws_bytes < hk_moe_workspace_bytes(T, topk, num_experts)
    HK_MOE_ERR_TOPK      = -4, // topk not 8 or 9
    HK_MOE_ERR_SHAPE     = -5, // hidden / inter not 4096 / 1024
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
