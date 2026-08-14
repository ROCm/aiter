/* SPDX-License-Identifier: MIT
   Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
*/
#include "rocm_ops.hpp"
#include <torch/all.h>

// topk_sigmoid is still a torch-based op (module_moe_topk_ck); declared here so it
// does not pull a torch dependency into the now torch-free moe_op.h.
namespace aiter {
void topk_sigmoid(torch::Tensor topk_weights,   // [tokens, topk]
                  torch::Tensor topk_indices,   // [tokens, topk]
                  torch::Tensor gating_output); // [tokens, experts]
} // namespace aiter

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    MOE_TOPK_CK_PYBIND;
}
