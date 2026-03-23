// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#include <pybind11/pybind11.h>
#include "aiter_enum.h"

PYBIND11_MODULE(module_aiter_enum, m)
{
    pybind11::enum_<QuantType>(m, "QuantType")
        .value("No", QuantType::No)
        .value("per_Tensor", QuantType::per_Tensor)
        .value("per_Token", QuantType::per_Token)
        .value("per_1x32", QuantType::per_1x32)
        .value("per_1x128", QuantType::per_1x128)
        .value("per_128x128", QuantType::per_128x128)
        .value("per_256x128", QuantType::per_256x128)
        .value("per_1024x128", QuantType::per_1024x128)
        .export_values();
    pybind11::enum_<ActivationType>(m, "ActivationType")
        .value("No", ActivationType::No)
        .value("Silu", ActivationType::Silu)
        .value("Gelu", ActivationType::Gelu)
        .value("Swiglu", ActivationType::Swiglu)
        .export_values();
    pybind11::implicitly_convertible<int, QuantType>();
    pybind11::implicitly_convertible<int, ActivationType>();
}
