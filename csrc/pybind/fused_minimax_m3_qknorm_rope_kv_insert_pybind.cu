// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2024-2026, The vLLM team.
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

#include "fused_minimax_m3_qknorm_rope_kv_insert.h"
#include "rocm_ops.hpp"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { FUSED_MINIMAX_M3_QKNORM_ROPE_KV_INSERT_PYBIND; }
