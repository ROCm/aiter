// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <ATen/ATen.h>
#include <torch/extension.h>

#include <optional>

void hipb_grouped_mm(
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& cu_seqlens,
    torch::Tensor out,
    bool a_is_transposed = false,
    std::optional<torch::Tensor> bias = std::nullopt,
    int solution_index = -1);

void hipb_multistream_mm(
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& cu_seqlens,
    torch::Tensor out,
    bool a_is_transposed = false,
    std::optional<torch::Tensor> bias = std::nullopt,
    bool b_is_transposed = false);
