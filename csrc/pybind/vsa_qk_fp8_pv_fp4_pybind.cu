// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#include "rocm_ops.hpp"
#include "vsa_qk_fp8_pv_fp4.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { VSA_QK_FP8_PV_FP4_PYBIND; }
