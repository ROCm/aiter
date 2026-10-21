// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// ctypes binding; aiter/ops/hk_sparse_mla.py validates tensors.
#include "aiter_hip_common.h"

#include "aiter_ctypes_error.h"
#include "sparse_mla.h"

AITER_CTYPES_ERROR_DEF

AITER_CTYPES_DEFINE_ENTRYPOINT_VOID(
    aiter_hk_sparse_mla_fwd,
    (size_t q,
     size_t kv,
     size_t indices,
     size_t out,
     int64_t num_tokens,
     int64_t topk,
     int64_t kv_stride_bytes,
     int64_t num_slots,
     float sm_scale,
     hipStream_t stream),
    (q, kv, indices, out, num_tokens, topk, kv_stride_bytes, num_slots, sm_scale, stream))
{
    const int rc = hk_sparse_mla_fwd(reinterpret_cast<const void*>(q),
                                     reinterpret_cast<const void*>(kv),
                                     reinterpret_cast<const int*>(indices),
                                     reinterpret_cast<void*>(out),
                                     static_cast<int>(num_tokens),
                                     static_cast<int>(topk),
                                     static_cast<long>(kv_stride_bytes),
                                     static_cast<long>(num_slots),
                                     sm_scale,
                                     stream);
    AITER_CHECK(rc != HK_SPARSE_MLA_BAD_SHAPE,
                "hk_sparse_mla_fwd: topk must be a positive multiple of 128 (<= 2304), kv rows "
                "16-byte aligned and "
                "num_slots in [1, 2^31)");
    AITER_CHECK(rc == HK_SPARSE_MLA_OK, "hk_sparse_mla_fwd failed with status ", rc);
}
