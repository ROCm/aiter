// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include "rope.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("rope_cached_positions_2c_fwd_impl", &rope_cached_positions_2c_fwd_impl);
    m.def("rope_cached_positions_offsets_2c_fwd_impl", &rope_cached_positions_offsets_2c_fwd_impl);
}
