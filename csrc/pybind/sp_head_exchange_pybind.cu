// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#include "aiter_stream.h"
#include "rocm_ops.hpp"
#include <optional>
namespace aiter {
void all_to_all(int64_t handle,
                const aiter_tensor_t& input,
                const aiter_tensor_t& output,
                int64_t registered_buffer,
                int64_t registered_bytes,
                bool stage,
                bool gather_last,
                int64_t blocks,
                std::optional<aiter_tensor_t> scales,
                std::optional<aiter_tensor_t> scales_out);
void sp_head_exchange(int64_t handle,
                      const aiter_tensor_t& input,
                      const aiter_tensor_t& output,
                      int64_t registered_buffer,
                      int64_t registered_bytes,
                      bool stage,
                      int64_t blocks);
} // namespace aiter
PYBIND11_MODULE(AITER_EXTENSION_NAME, m)
{
    AITER_SET_STREAM_PYBIND;
    m.def("sp_head_exchange", &aiter::sp_head_exchange);
    m.def("all_to_all",
          &aiter::all_to_all,
          py::arg("handle"),
          py::arg("input"),
          py::arg("output"),
          py::arg("registered_buffer"),
          py::arg("registered_bytes"),
          py::arg("stage"),
          py::arg("gather_last"),
          py::arg("blocks"),
          py::arg("scales")     = std::nullopt,
          py::arg("scales_out") = std::nullopt);
}
