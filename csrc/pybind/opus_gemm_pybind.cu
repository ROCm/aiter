// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Register the four OPUS launch interfaces on the host pass only.
#ifndef __HIP_DEVICE_COMPILE__

#include "rocm_ops.hpp"
#include "aiter_stream.h"
#include "opus_bmm.h"
#include "opus_gemm.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    AITER_SET_STREAM_PYBIND
    OPUS_GEMM_A16W16_LAUNCH_PYBIND;
    OPUS_GEMM_A8W8_LAUNCH_PYBIND;
    OPUS_GEMM_A8W8_BLOCKSCALE_LAUNCH_PYBIND;
    OPUS_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE_LAUNCH_PYBIND;
    OPUS_GEMM_A8W8_MXSCALE_BMM_LAUNCH_PYBIND;
}

#endif // !__HIP_DEVICE_COMPILE__
