#pragma once

#include <torch/extension.h>

torch::Tensor pa_fwd_naive(torch::Tensor &Q, //   [num_seqs, num_heads, head_size]
                           torch::Tensor &K, //   [num_blocks, num_kv_heads, head_size/x, block_size, x]
                                             // or[num_batch, seqlen, num_kv_heads, head_size]
                           torch::Tensor &V, //   [num_blocks, num_kv_heads, head_size, block_size]
                                             // or[num_batch*seqlen, num_kv_heads, head_size]
                           torch::Tensor &block_tables,
                           torch::Tensor &context_lens,
                           torch::Tensor &k_dequant_scales,
                           torch::Tensor &v_dequant_scales,
                           const int max_seq_len,
                           const int num_kv_heads,
                           const float scale_s,
                           const float scale_k,
                           const float scale_v,
                           const int block_size,
                           const int quant_algo,
                           std::optional<torch::Tensor> kv_qscale = std::nullopt
                           // above are input
);