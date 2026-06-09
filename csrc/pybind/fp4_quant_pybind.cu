// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#include "fp4_quant.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("_quantize_fp4_e8m0_per_channel_kblock_hip",
          &aiter::quantize_fp4_e8m0_per_channel_kblock_hip,
          "Quantize V to packed FP4 E2M1 with per-channel per-kblock E8M0 scale.",
          py::arg("packed"),
          py::arg("scale_byte"),
          py::arg("v"),
          py::arg("kblock_size") = 32);
}
