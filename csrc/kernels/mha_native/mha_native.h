#pragma once
#include "aiter_tensor.h"

namespace aiter {
// Torch-free entry: all buffers (o, lse, scratch_o, scratch_lse) are
// pre-allocated by the Python side and passed in; this writes into o (and lse
// when return_lse). lse / scratch_* are sized by the caller (see
// aiter/ops/mha.py). scratch_o = [G,B,Hq,Sq,D] fp32, scratch_lse = [G,B,Hq,Sq]
// fp32, lse = [B,Hq,Sq] fp32 (or empty when return_lse == false).
void mha_fwd_native_splitkv(
    aiter_tensor_t& q, aiter_tensor_t& k, aiter_tensor_t& v,
    aiter_tensor_t& o, aiter_tensor_t& lse,
    aiter_tensor_t& scratch_o, aiter_tensor_t& scratch_lse,
    double softmax_scale, bool causal, bool return_lse, int64_t num_splits);
}  // namespace aiter
