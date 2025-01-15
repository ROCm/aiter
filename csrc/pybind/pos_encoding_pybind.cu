// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
#include "pos_encoding.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("rotary_embedding_fwd", &rotary_embedding, "rotary_embedding");
    m.def("batched_rotary_embedding", &batched_rotary_embedding, "batched_rotary_embedding");
}