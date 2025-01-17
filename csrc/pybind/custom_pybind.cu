// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
#include "custom.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("wvSpltK", &wvSpltK, "wvSpltK(Tensor in_a, Tensor in_b, Tensor! out_c, int N_in,"
                               "        int CuCount) -> ()");
    m.def("LLMM1", &LLMM1, "LLMM1(Tensor in_a, Tensor in_b, Tensor! out_c, int rows_per_block) -> "
                           "()");
}