// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// ctypes bindings; aiter/ops/hk_sparse_mla.py validates tensors and allocates workspace.
#include "aiter_hip_common.h"

#include "aiter_ctypes_error.h"
#include "sparse_mla.h"

AITER_CTYPES_ERROR_DEF

// Workspace in 256-byte units; zero for one split.
AITER_CTYPES_DEFINE_ENTRYPOINT(aiter_hk_sparse_mla_workspace_units,
                               (int64_t num_tokens, int64_t topk, hipStream_t stream),
                               (num_tokens, topk, stream))
{
    return static_cast<int>(
        (hk_sparse_mla_workspace_bytes(static_cast<int>(num_tokens), static_cast<int>(topk)) +
         255) /
        256);
}

AITER_CTYPES_DEFINE_ENTRYPOINT_VOID(aiter_hk_sparse_mla_fwd,
                                    (size_t q,
                                     size_t kv,
                                     size_t indices,
                                     size_t out,
                                     int64_t num_tokens,
                                     int64_t topk,
                                     int64_t kv_stride_bytes,
                                     int64_t num_slots,
                                     float sm_scale,
                                     size_t workspace,
                                     int64_t workspace_bytes,
                                     hipStream_t stream),
                                    (q,
                                     kv,
                                     indices,
                                     out,
                                     num_tokens,
                                     topk,
                                     kv_stride_bytes,
                                     num_slots,
                                     sm_scale,
                                     workspace,
                                     workspace_bytes,
                                     stream))
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
                                     reinterpret_cast<void*>(workspace),
                                     static_cast<size_t>(workspace_bytes),
                                     stream);
    AITER_CHECK(rc != HK_SPARSE_MLA_BAD_SHAPE,
                "hk_sparse_mla_fwd: topk must be a positive multiple of 128 (<= 2304), kv rows "
                "16-byte aligned and "
                "num_slots in [1, 2^31)");
    AITER_CHECK(rc != HK_SPARSE_MLA_SMALL_WORKSPACE, "hk_sparse_mla_fwd: workspace too small");
    AITER_CHECK(rc == HK_SPARSE_MLA_OK, "hk_sparse_mla_fwd failed with status ", rc);
}
