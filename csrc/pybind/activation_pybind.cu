// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
#include "activation.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("silu_and_mul", &silu_and_mul, "Activation function used in SwiGLU.");
    m.def("gelu_and_mul", &gelu_and_mul, "Activation function used in GELU.");
    m.def("gelu_tanh_and_mul", &gelu_tanh_and_mul, "Activation function used in GELU tanh.");
}