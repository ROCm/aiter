// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// gfx950 sparse MLA: H32/D512, shared latent K/V, no separate RoPE.
// out[t,h] = softmax(q[t,h] @ kv[idx[t]].T * sm_scale) @ kv[idx[t]].
//
// q: E4M3 [T,32,512]; out: BF16 [T,32,512].
// kv: E4M3 rows at slot*kv_stride_bytes; first 512 bytes are latent, rows aligned to 16.
// idx: int32 [T,topk]; topk is a positive multiple of 128, <=2304.
// Slots outside [0,num_slots) are masked. No workspace is needed.
#pragma once
#include <hip/hip_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

enum HkSparseMlaStatus : int
{
    HK_SPARSE_MLA_OK            = 0,
    HK_SPARSE_MLA_BAD_SHAPE     = -1,
    HK_SPARSE_MLA_LAUNCH_FAILED = -2,
};

// Returns HkSparseMlaStatus; graph-capturable on stream.
int hk_sparse_mla_fwd(const void* q,
                      const void* kv,
                      const int* idx,
                      void* out,
                      int T,
                      int topk,
                      long kv_stride_bytes,
                      long num_slots,
                      float sm_scale,
                      hipStream_t stream);

#ifdef __cplusplus
}
#endif
