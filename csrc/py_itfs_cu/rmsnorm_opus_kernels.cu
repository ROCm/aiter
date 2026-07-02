// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Opus RMSNorm: torch-free, pybind-free C ABI (ctypes). One TU, no CK.
// The trailing hipStream_t is the stream auto-appended by aiter's _ctypes_call.

#include "rmsnorm_opus.h" // brings aiter_hip_common.h (AITER_C_ITFS, aiter_detail)

#include "aiter_ctypes_error.h"

AITER_CTYPES_ERROR_DEF

// out = rmsnorm(input) * weight  (bf16/fp16, fp32 accumulate)
AITER_CTYPES_DEFINE_ENTRYPOINT_VOID(rms_norm_opus,
                                    (aiter_tensor_t * out,
                                     aiter_tensor_t* input,
                                     aiter_tensor_t* weight,
                                     float epsilon,
                                     hipStream_t stream),
                                    (out, input, weight, epsilon, stream))
{
    using namespace aiter::rmsnorm_opus;
    check_common(*input, *weight);
    AITER_CHECK(out->dtype() == input->dtype(), "rms_norm_opus: out dtype must match input");
    AITER_CHECK(out->is_contiguous(), "rms_norm_opus: out must be contiguous");
    AITER_CHECK(out->numel() == input->numel(), "rms_norm_opus: out must match input shape");

    const int hidden = static_cast<int>(input->size(-1));
    const int rows   = hidden == 0 ? 0 : static_cast<int>(input->numel() / hidden);
    if(rows == 0 || hidden == 0)
        return;

    HipDeviceGuard guard(input->device_id);
    if(input->dtype() == AITER_DTYPE_bf16)
        launch_rms<bf16_t>(*out, *input, *weight, epsilon, rows, hidden, stream);
    else
        launch_rms<fp16_t>(*out, *input, *weight, epsilon, rows, hidden, stream);
}

// In-place fused residual-add + rmsnorm:
//   x = input + residual;  residual = x;  input = rmsnorm(x) * weight
AITER_CTYPES_DEFINE_ENTRYPOINT_VOID(fused_add_rms_norm_opus,
                                    (aiter_tensor_t * input,
                                     aiter_tensor_t* residual,
                                     aiter_tensor_t* weight,
                                     float epsilon,
                                     hipStream_t stream),
                                    (input, residual, weight, epsilon, stream))
{
    using namespace aiter::rmsnorm_opus;
    check_common(*input, *weight);
    AITER_CHECK(residual->dtype() == input->dtype(),
                "fused_add_rms_norm_opus: residual dtype must match input");
    AITER_CHECK(residual->is_contiguous(), "fused_add_rms_norm_opus: residual must be contiguous");
    AITER_CHECK(residual->numel() == input->numel(),
                "fused_add_rms_norm_opus: residual must match input shape");

    const int hidden = static_cast<int>(input->size(-1));
    const int rows   = hidden == 0 ? 0 : static_cast<int>(input->numel() / hidden);
    if(rows == 0 || hidden == 0)
        return;

    HipDeviceGuard guard(input->device_id);
    if(input->dtype() == AITER_DTYPE_bf16)
        launch_fused_add<bf16_t>(*input, *residual, *weight, epsilon, rows, hidden, stream);
    else
        launch_fused_add<fp16_t>(*input, *residual, *weight, epsilon, rows, hidden, stream);
}
