// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

#include <ATen/hip/HIPContext.h>
#include <torch/extension.h>

#include "aiter_stream.h"
#include "rocm_ops.hpp"

torch::Tensor vsa_sparse_attention(const torch::Tensor& q,
                                   const torch::Tensor& k,
                                   const torch::Tensor& v,
                                   const torch::Tensor& block_lut,
                                   const torch::Tensor& block_counts);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    AITER_SET_STREAM_PYBIND;
    py::options options;
    options.disable_function_signatures();
    m.def("vsa_sparse_attention",
          [](const torch::Tensor& q,
             const torch::Tensor& k,
             const torch::Tensor& v,
             const torch::Tensor& block_lut,
             const torch::Tensor& block_counts) {
              aiter::setCurrentHIPStream(at::hip::getCurrentHIPStream());
              return vsa_sparse_attention(q, k, v, block_lut, block_counts);
          },
          "vsa_sparse_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, "
          "block_lut: torch.Tensor, block_counts: torch.Tensor) -> torch.Tensor",
          py::arg("q"),
          py::arg("k"),
          py::arg("v"),
          py::arg("block_lut"),
          py::arg("block_counts"));
}
