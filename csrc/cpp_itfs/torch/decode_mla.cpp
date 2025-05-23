#include <stdexcept>
#include <cmath>
#include "attention_asm_mla.h"
#include "decode_mla_stage2_asm.f51b00db_0d1d2d3d45678910.h"
#include <ATen/hip/HIPContext.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>


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
){
    const at::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(device_of(q));
    const hipStream_t stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA();
    if(logit_cap > 0){
        throw std::invalid_argument("logit_cap is not supported yet");
    }
    if(sm_scale == std::nullopt){
        sm_scale = 1.0/sqrt(kv_buffer.size(3));
    }
    int bs = output.size(0);
    int nhead = output.size(1);

    mla_stage1_asm_fwd(q, kv_buffer, kv_indptr, kv_indices, kv_last_page_lens, sm_scale.value(), logits, attn_lse);

    decode_mla_stage2_asm_f51b00db_0d1d2d3d45678910(stream, logits.data_ptr(), attn_lse.data_ptr(), output.data_ptr(), kv_indptr.data_ptr(), attn_lse.stride(0), attn_lse.stride(2), attn_lse.stride(1), output.stride(0), output.stride(1), bs, nhead);
}
}