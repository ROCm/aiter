// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

// Torch-free TU. The bound ops take aiter_tensor_t, registered by
// module_aiter_core in the torch-free pybind ABI family, so this TU must not
// pull in torch/ATen headers -- doing so would link libtorch and switch the
// module to torch's PYBIND11_INTERNALS_ID, which never matches.
#include "rocm_ops.hpp"
#include "aiter_stream.h"
#include "gemm_a4w4_blockscale.h"

PYBIND11_MODULE(AITER_EXTENSION_NAME, m)
{
    // Required by the develop=True ops: lets the Python marshalling push the
    // current HIP stream into this TU.
    AITER_SET_STREAM_PYBIND
    GEMM_A4W4_BLOCKSCALE_PYBIND;
}
