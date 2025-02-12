// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include "rope.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("rope_fwd", &rope_fwd);
    m.def("rope_bwd", &rope_bwd);
    m.def("rope_cached_fwd", &rope_cached_fwd);
    m.def("rope_cached_bwd", &rope_cached_bwd);
    m.def("rope_thd_fwd", &rope_thd_fwd);
    m.def("rope_thd_bwd", &rope_thd_bwd);
    m.def("rope_2d_fwd", &rope_2d_fwd);
    m.def("rope_2d_bwd", &rope_2d_bwd);
}
