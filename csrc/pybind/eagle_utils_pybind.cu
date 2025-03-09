// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
#include "eagle_utils.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("build_tree_kernel_efficient", build_tree_kernel_efficient, "build_tree_kernel_efficient");

  m.def("build_tree_kernel", build_tree_kernel, "build_tree_kernel");
}