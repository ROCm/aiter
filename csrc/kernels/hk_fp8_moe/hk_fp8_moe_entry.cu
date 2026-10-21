// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// ctypes bindings; aiter/ops/hk_fp8_moe.py validates tensors and allocates workspace.
#include "aiter_hip_common.h"

#include "aiter_ctypes_error.h"
#include "hk_moe.h"

AITER_CTYPES_ERROR_DEF

namespace {
const char* moe_status_message(int rc)
{
    switch(rc)
    {
    case HK_MOE_ERR_LAUNCH: return "kernel launch failed";
    case HK_MOE_ERR_EXPERTS: return "more than 1024 experts";
    case HK_MOE_ERR_WORKSPACE: return "workspace too small";
    case HK_MOE_ERR_TOPK: return "top-k must be 8 or 9";
    case HK_MOE_ERR_SHAPE: return "hidden / intermediate must be 4096 / 1024";
    default: return "unknown status";
    }
}
} // namespace

// Use 256-byte units to fit workspace sizes in the int return.
AITER_CTYPES_DEFINE_ENTRYPOINT(
    aiter_hk_fp8_moe_workspace_units,
    (int64_t num_tokens, int64_t topk, int64_t num_experts, hipStream_t stream),
    (num_tokens, topk, num_experts, stream))
{
    return static_cast<int>((hk_moe_workspace_bytes(static_cast<int>(num_tokens),
                                                    static_cast<int>(topk),
                                                    static_cast<int>(num_experts)) +
                             255) /
                            256);
}

AITER_CTYPES_DEFINE_ENTRYPOINT_VOID(aiter_hk_fp8_moe_fwd,
                                    (size_t x,
                                     size_t out,
                                     int64_t num_tokens,
                                     size_t topk_ids,
                                     size_t topk_weights,
                                     int64_t topk,
                                     size_t w13,
                                     size_t w13_scale,
                                     size_t w2,
                                     size_t w2_scale,
                                     int64_t num_experts,
                                     float swiglu_limit,
                                     size_t workspace,
                                     int64_t workspace_bytes,
                                     hipStream_t stream),
                                    (x,
                                     out,
                                     num_tokens,
                                     topk_ids,
                                     topk_weights,
                                     topk,
                                     w13,
                                     w13_scale,
                                     w2,
                                     w2_scale,
                                     num_experts,
                                     swiglu_limit,
                                     workspace,
                                     workspace_bytes,
                                     stream))
{
    const int rc = hk_moe_forward(reinterpret_cast<const void*>(x),
                                  reinterpret_cast<void*>(out),
                                  static_cast<int>(num_tokens),
                                  4096,
                                  1024,
                                  reinterpret_cast<const int*>(topk_ids),
                                  reinterpret_cast<const float*>(topk_weights),
                                  static_cast<int>(topk),
                                  reinterpret_cast<const void*>(w13),
                                  reinterpret_cast<const float*>(w13_scale),
                                  reinterpret_cast<const void*>(w2),
                                  reinterpret_cast<const float*>(w2_scale),
                                  static_cast<int>(num_experts),
                                  swiglu_limit,
                                  reinterpret_cast<void*>(workspace),
                                  static_cast<size_t>(workspace_bytes),
                                  stream);
    AITER_CHECK(rc == HK_MOE_OK, "hk_fp8_moe: ", moe_status_message(rc), " (status ", rc, ")");
}
