// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// pybind glue is host-only. Skip the entire TU on the device pass so we
// don't pay the libtorch + pybind11 + HIP runtime parse (~15s) for code
// that has no GPU side at all.
#ifndef __HIP_DEVICE_COMPILE__

#include "rocm_ops.hpp"
#include "aiter_hip_common.h"
#include "aiter_stream.h"
#include "opus_gemm.h"
#include "opus_bmm.h"

PYBIND11_MODULE(AITER_EXTENSION_NAME, m)
{
    py::register_exception<aiter_detail::unbaked_kernel_error>(
        m, "UnbakedKernelError", PyExc_RuntimeError);
    AITER_SET_STREAM_PYBIND
    OPUS_GEMM_PYBIND;
    OPUS_GEMM_A16W16_TUNE_PYBIND;
    OPUS_BMM_A8W8_MXSCALE_PYBIND;
    OPUS_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE_TUNE_PYBIND;
    OPUS_GEMM_WORKSPACE_INIT_PYBIND;
    OPUS_GEMM_WORKSPACE_RELEASE_PYBIND;
}

#endif // !__HIP_DEVICE_COMPILE__
