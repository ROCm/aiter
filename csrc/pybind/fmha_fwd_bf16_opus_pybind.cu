// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#include "rocm_ops.hpp"
#include "torch/fmha_fwd_bf16_opus.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    FMHA_FWD_BF16_OPUS_PYBIND;
}
