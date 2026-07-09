#pragma once
#include <torch/extension.h>
#include <optional>
#include <vector>

namespace aiter {
// Returns {o, lse}. lse is an empty (0,) fp32 tensor when return_lse == false.
std::vector<at::Tensor> mha_fwd_native_splitkv(
    at::Tensor q, at::Tensor k, at::Tensor v,
    std::optional<at::Tensor> out,
    double softmax_scale, bool causal, bool return_lse, int64_t num_splits);
}  // namespace aiter
