// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include "rope.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("rope_fwd_impl", &rope_fwd_impl);
    m.def("rope_bwd_impl", &rope_bwd_impl);
    m.def("rope_cached_fwd_impl", &rope_cached_fwd_impl);
    m.def("rope_cached_bwd_impl", &rope_cached_bwd_impl);
    m.def("rope_thd_fwd_impl", &rope_thd_fwd_impl);
    m.def("rope_thd_bwd_impl", &rope_thd_bwd_impl);
    m.def("rope_2d_fwd_impl", &rope_2d_fwd_impl);
    m.def("rope_2d_bwd_impl", &rope_2d_bwd_impl);
}
