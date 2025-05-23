#pragma once

#include <torch/torch.h>
#include <optional>

namespace aiter{
void mla_decode_fwd(
    torch::Tensor& q,
    torch::Tensor& kv_buffer,
    torch::Tensor& output,
    torch::Tensor& kv_indptr,
    torch::Tensor& kv_indices,
    torch::Tensor& kv_last_page_lens,
    std::optional<float> softmax_scale=std::nullopt,
    float logit_cap=0.0,
    std::optional<int> num_kv_splits=std::nullopt,
    std::optional<torch::Tensor> logits=std::nullopt,
    std::optional<torch::Tensor> attn_lse=std::nullopt
);
}
