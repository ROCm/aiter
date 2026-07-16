// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

#ifndef __HIP_DEVICE_COMPILE__

#include "rocm_ops.hpp"
#include "aiter_stream.h"
#include "opus_moe.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    AITER_SET_STREAM_PYBIND
    OPUS_MOE_PYBIND;
}

#endif // !__HIP_DEVICE_COMPILE__
