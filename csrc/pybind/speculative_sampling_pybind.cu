// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#include "speculative_sampling.h"
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("tree_speculative_sampling_target_only", &tree_speculative_sampling_target_only);
}
