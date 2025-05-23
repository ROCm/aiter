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
    torch::Tensor& logits,
    torch::Tensor& attn_lse,
    std::optional<float> sm_scale=std::nullopt,
    float logit_cap=0.0
);
}