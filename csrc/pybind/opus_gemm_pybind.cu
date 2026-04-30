// SPDX-License-Identifier: MIT
// Copyright (c) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// pybind glue is host-only. Skip the entire TU on the device pass so we
// don't pay the libtorch + pybind11 + HIP runtime parse (~15s) for code
// that has no GPU side at all.
#ifndef __HIP_DEVICE_COMPILE__

#include "rocm_ops.hpp"
#include "aiter_stream.h"
#include "opus_gemm.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    AITER_SET_STREAM_PYBIND
    OPUS_GEMM_PYBIND;
    OPUS_GEMM_A16W16_TUNE_PYBIND;
}

#endif // !__HIP_DEVICE_COMPILE__
