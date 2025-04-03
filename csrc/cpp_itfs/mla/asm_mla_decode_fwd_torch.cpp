#include <stdexcept>
#include <cmath>
#include "asm_mla_decode_fwd_torch.h"
#include "asm_mla_decode_fwd.h"
#include <ATen/hip/HIPContext.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <torch/all.h>


namespace aiter{
void mla_decode_fwd(
    torch::Tensor& q,
    torch::Tensor& kv_buffer,
    torch::Tensor& output,
    torch::Tensor& kv_indptr,
    torch::Tensor& kv_indices,
    torch::Tensor& kv_last_page_lens,
    std::optional<float> softmax_scale,
    float logit_cap,
    std::optional<int> num_kv_splits,
    std::optional<torch::Tensor> logits,
    std::optional<torch::Tensor> attn_lse
){
    const at::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(device_of(q));
    const hipStream_t stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA();

    if (q.dtype() != torch::kBFloat16) {
        throw std::invalid_argument("mla_decode_fwd: only support dtype == torch.bfloat16 for now");
    }
    int num_kv_heads = kv_buffer.size(2);
    int num_seqs = q.size(0);
    int num_heads = q.size(1);
    int head_size = q.size(2);
    int page_size = kv_buffer.size(1);
    int v_head_dim = output.size(2);

    if (num_kv_heads != 1) {
        throw std::invalid_argument("mla_decode_fwd: only support num_kv_heads==1 for now");
    }

    if (head_size != kv_buffer.size(3)) {
        throw std::invalid_argument("mla_decode_fwd: only support head_size == KV.size(3) for now");
    }

    if (logit_cap > 0) {
        throw std::invalid_argument("mla_decode_fwd: only support logit_cap==0 for now");
    }

    if (softmax_scale == std::nullopt) {
        softmax_scale = 1.0f / std::sqrt(static_cast<float>(v_head_dim));
    }

    // Calculate num_kv_splits if not provided
    if (num_kv_splits == std::nullopt) {
        // Note: This would need hipGetDeviceProperties equivalent
        auto cu_num = at::cuda::getDeviceProperties(q.device().index())->multiProcessorCount;
        num_kv_splits = std::min(16, std::max(1, cu_num / num_seqs));
    }

    if(logits == std::nullopt){
        logits = torch::empty({num_seqs, num_kv_splits.value(), num_heads, v_head_dim}, q.options().dtype(torch::kFloat32));
    }
    if(attn_lse == std::nullopt){
        attn_lse = torch::empty({num_seqs, num_kv_splits.value(), num_heads, 1}, q.options().dtype(torch::kFloat32));
    }

    if(num_kv_splits.value() != logits.value().size(1)){
        throw std::invalid_argument("mla_decode_fwd: num_kv_splits != logits.size(1)");
    }
    
    asm_mla_decode_fwd(std::nullopt, q.data_ptr(), kv_buffer.data_ptr(), kv_indptr.data_ptr(), kv_indices.data_ptr(), kv_last_page_lens.data_ptr(), softmax_scale.value(), logits.value().data_ptr(), attn_lse.value().data_ptr(), output.data_ptr(), num_seqs, num_heads, num_kv_heads, q.stride(0), kv_buffer.stride(0), attn_lse.value().stride(0), attn_lse.value().stride(1), attn_lse.value().stride(2), output.stride(0), output.stride(1), page_size, "__hip_bfloat16", "__hip_bfloat16", num_kv_splits.value(), v_head_dim, stream);
}
}