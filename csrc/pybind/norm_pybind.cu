/*
 * Copyright (c) 2024 Advanced Micro Devices, Inc.  All rights reserved.
 *
 * @Script: norm_pybind.cu
 * @Author: valarLip
 * @Email: lingpeng.jin@amd.com
 * @Create At: 2024-12-02 16:00:01
 * @Last Modified By: valarLip
 * @Last Modified At: 2025-01-03 16:34:45
 * @Description: This is description.
 */

#include "norm.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("layernorm2d_fwd", &layernorm2d,
          py::arg("input"), py::arg("weight"), py::arg("bias"),
          py::arg("epsilon"), py::arg("x_bias") = std::nullopt);
    m.def("layernorm2d_fwd_with_add", &layernorm2d_with_add,
          py::arg("out"), py::arg("input"),
          py::arg("residual_in"), py::arg("residual_out"),
          py::arg("weight"), py::arg("bias"),
          py::arg("epsilon"), py::arg("x_bias") = std::nullopt);
    m.def("layernorm2d_fwd_with_smoothquant", &layernorm2d_with_smoothquant,
          py::arg("out"), py::arg("input"),
          py::arg("xscale"), py::arg("yscale"),
          py::arg("weight"), py::arg("bias"),
          py::arg("epsilon"), py::arg("x_bias") = std::nullopt);
    m.def("layernorm2d_fwd_with_add_smoothquant", &layernorm2d_with_add_smoothquant,
          py::arg("out"), py::arg("input"),
          py::arg("residual_in"), py::arg("residual_out"),
          py::arg("xscale"), py::arg("yscale"),
          py::arg("weight"), py::arg("bias"),
          py::arg("epsilon"), py::arg("x_bias") = std::nullopt);
    m.def("layernorm2d_fwd_with_dynamicquant", &layernorm2d_with_dynamicquant,
          py::arg("out"), py::arg("input"), 
          py::arg("yscale"), py::arg("weight"), py::arg("bias"),
          py::arg("epsilon"), py::arg("x_bias") = std::nullopt);
    m.def("layernorm2d_fwd_with_add_dynamicquant", &layernorm2d_with_add_dynamicquant,
          py::arg("out"), py::arg("input"),
          py::arg("residual_in"), py::arg("residual_out"),
          py::arg("yscale"), py::arg("weight"), py::arg("bias"),
          py::arg("epsilon"), py::arg("x_bias") = std::nullopt);
    // following are asm kernels
    m.def("layernorm2d_with_add_asm", &layernorm2d_with_add_asm);
    m.def("layernorm2d_with_add_smoothquant_asm", &layernorm2d_with_add_smoothquant_asm);
}