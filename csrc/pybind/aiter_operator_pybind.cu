// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
#include "aiter_operator.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("add", &aiter_add, "apply for add with transpose and broadcast.");
    m.def("mul", &aiter_mul, "apply for mul with transpose and broadcast.");
    m.def("sub", &aiter_sub, "apply for sub with transpose and broadcast.");
    m.def("div", &aiter_div, "apply for div with transpose and broadcast.");
    m.def("add_", &aiter_add_, "apply for add_ with transpose and broadcast.");
    m.def("mul_", &aiter_mul_, "apply for mul_ with transpose and broadcast.");
    m.def("sub_", &aiter_sub_, "apply for sub_ with transpose and broadcast.");
    m.def("div_", &aiter_div_, "apply for div_ with transpose and broadcast.");
    m.def("sigmoid", &aiter_sigmoid, "apply for sigmoid.");
    m.def("tanh", &aiter_tanh, "apply for tanh.");
}
