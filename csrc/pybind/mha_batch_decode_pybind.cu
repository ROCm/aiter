// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
#include "rocm_ops.hpp"
#include "mha_batch_decode.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { MHA_BATCH_DECODE_PYBIND; }