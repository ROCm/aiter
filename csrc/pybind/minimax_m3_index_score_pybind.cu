// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

#include "aiter_stream.h"
#include "minimax_m3_index_score.h"
#include "rocm_ops.hpp"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    AITER_SET_STREAM_PYBIND
    m.def("minimax_m3_decode_index_score",
          &aiter::minimax_m3_decode_index_score,
          "MiniMax-M3 decode index block-score kernel");
}
