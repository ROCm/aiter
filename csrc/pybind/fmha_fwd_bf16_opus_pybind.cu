// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#include "fmha_fwd_host_scale.hpp"
#include "torch/fmha_fwd_bf16_opus.h"

namespace py = pybind11;

PYBIND11_MODULE(AITER_EXTENSION_NAME, m)
{
    // Accept the Python double before the default float caster can flush a subnormal scale.
    m.def("fmha_fwd_bf16_opus_fwd",
          [](at::Tensor& q, at::Tensor& k, at::Tensor& v, at::Tensor& out,
             bool causal, double softmax_scale, const std::optional<at::Tensor>& lse,
             const std::optional<at::Tensor>& seqstart_q, const std::optional<at::Tensor>& seqstart_k,
             const std::optional<at::Tensor>& seqstart_q_pad, const std::optional<at::Tensor>& seqstart_k_pad,
             int max_seqlen_q, int max_seqlen_k, int window_size_left) {
              fmha_fwd_bf16_opus_fwd(q, k, v, out, causal, aiter::fmha::narrow_scale(softmax_scale),
                                     lse, seqstart_q, seqstart_k, seqstart_q_pad, seqstart_k_pad,
                                     max_seqlen_q, max_seqlen_k, window_size_left);
          },
          py::arg("q"),
          py::arg("k"),
          py::arg("v"),
          py::arg("out"),
          py::arg("causal"),
          py::arg("softmax_scale"),
          py::arg("lse") = std::nullopt,
          py::arg("seqstart_q") = std::nullopt,
          py::arg("seqstart_k") = std::nullopt,
          py::arg("seqstart_q_pad") = std::nullopt,
          py::arg("seqstart_k_pad") = std::nullopt,
          py::arg("max_seqlen_q") = 0,
          py::arg("max_seqlen_k") = 0,
          py::arg("window_size_left") = -1);
}
