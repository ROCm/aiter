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
    m.def("layernorm2d_fwd", &layernorm2d);
    m.def("layernorm2d_fwd_with_add", &layernorm2d_with_add);
    m.def("layernorm2d_fwd_with_smoothquant", &layernorm2d_with_smoothquant);
    m.def("layernorm2d_fwd_with_add_smoothquant", &layernorm2d_with_add_smoothquant);
    m.def("layernorm2d_fwd_with_dynamicquant", &layernorm2d_with_dynamicquant);
    m.def("layernorm2d_fwd_with_add_dynamicquant", &layernorm2d_with_add_dynamicquant);
    // following are asm kernels
    m.def("layernorm2d_with_add_asm", &layernorm2d_with_add_asm);
    m.def("layernorm2d_with_add_smoothquant_asm", &layernorm2d_with_add_smoothquant_asm);
}