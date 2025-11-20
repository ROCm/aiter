// SPDX-License-Identifier: MIT
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <torch/extension.h>

using namespace at;

using fptr_t = int64_t;

fptr_t init_trtllm_ar_fusion(int64_t rank, int64_t world_size, int64_t max_size_in_bytes);
void destroy_trtllm_ar_fusion(fptr_t fptr);
Tensor get_trtllm_ar_fusion_handle(fptr_t fptr);
void open_trtllm_ar_fusion_handles(fptr_t fptr, std::vector<Tensor> handles);
Tensor get_trtllm_ar_fusion_workspace(fptr_t fptr, const Tensor& ref);

void trtllm_allreduce_rms(int64_t rank,
                          int64_t nranks,
                          at::Tensor& allreduce_in,
                          at::Tensor& residual_in,
                          at::Tensor& rms_gamma,
                          at::Tensor& residual_out,
                          at::Tensor& norm_out,
                          at::Tensor& scale_out,
                          double eps,
                          bool fp8_out,
                          Tensor& workspace);
