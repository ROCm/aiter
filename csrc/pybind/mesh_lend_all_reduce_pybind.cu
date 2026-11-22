// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#include "aiter_stream.h"
#include "mesh_lend_all_reduce.h"
#include "rocm_ops.hpp"

PYBIND11_MODULE(AITER_EXTENSION_NAME, m) { MESH_LEND_ALL_REDUCE_PYBIND; }
