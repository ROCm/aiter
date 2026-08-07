#pragma once
// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

#include <torch/extension.h>

namespace aiter {
namespace torch_itfs {

at::Tensor fmha_v4_fwd(const at::Tensor& q,
                       const at::Tensor& k,
                       const at::Tensor& v,
                       const at::Tensor& q_descale,
                       const at::Tensor& k_descale,
                       const at::Tensor& v_descale,
                       at::Tensor out,
                       int64_t q_format,
                       int64_t k_format,
                       int64_t v_format,
                       int64_t q_scale_mode,
                       int64_t k_scale_mode,
                       int64_t v_scale_mode,
                       double softmax_scale);

} // namespace torch_itfs
} // namespace aiter