// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include "py_itfs_common.h"
#include "ck_tile/ref/mla.hpp"

void mla_decode_fwd_ck(torch::Tensor &Q,    //   [batch_size, num_heads, kv_lora_rank + qk_rope_head_dim]
    torch::Tensor &K,                       //   [num_page * page_size, num_kv_heads, kv_lora_rank + qk_rope_head_dim]
    std::optional<torch::Tensor> &v_,        //   [num_page * page_size, num_kv_heads, v_head_dim]
    std::optional<torch::Tensor> &out_,        //   [batch_size, num_heads, v_head_dim]
    int head_size_v,
    torch::Tensor &kv_indptr,               //   [batch_size+1]
    torch::Tensor &kv_page_indices,         //   [num_page_used]
    torch::Tensor &kv_last_page_lens,       //   [batch_size]
    float softmax_scale)
{
    const at::cuda::OptionalCUDAGuard device_guard(device_of(Q));
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    torch::Tensor V = v_.value_or(K);
    torch::Tensor O = out_.value_or(torch::empty_like(Q));

    int batch = Q.size(0);
    int nhead = Q.size(1);
    int qk_head_dim = Q.size(2);
    int page_size = K.size(1);
    int kv_lora_rank = O.size(2);
    int qk_rope_head_dim = qk_head_dim - kv_lora_rank;

    // int batch = Q.size(0);
    // int nhead = Q.size(1);
    // int nhead_k = V.size(1);
    // int hdim_q = Q.size(2);
    // int hdim_v = V.size(2);
    // int max_num_blocks_per_seq = block_tables.size(1);
    // int max_kv_tokens = k_dequant_scales.numel() == 0 ? 0 : k_dequant_scales.size(1);

    ck_tile::naive_attention_fwd_traits naive_t;
    naive_t.q_type = torchDTypeToStr(Q.dtype());
    naive_t.k_type = torchDTypeToStr(K.dtype());
    naive_t.v_type = torchDTypeToStr(K.dtype());
    naive_t.o_type = torchDTypeToStr(O.dtype());
    // naive_t.q_layout = "bhsd";
    // naive_t.k_layout = "phdsx"; // TODO
    // naive_t.v_layout = "phds";  // TODO
    // naive_t.o_layout = "bhsd";
    // naive_t.variation = 2; // decode variation
    // naive_t.quant_algo = quant_algo;

    ck_tile::naive_attention_fwd_args naive_a;
    // naive_a.q_ptr = Q.data_ptr();
    // naive_a.k_ptr = K.data_ptr();
    // naive_a.v_ptr = V.data_ptr();
    // naive_a.o_ptr = out.data_ptr();
    // naive_a.scale_s = scale_s;
    // naive_a.context_len_ptr = context_lens.data_ptr(); // used when seqlen kv come from a pointer
    // naive_a.page_table_ptr = block_tables.data_ptr();  // [batch, num_blocks] seqlen_kv is in different block(paged attn)
    // naive_a.hdim = hdim_q;
    // naive_a.hdim_v = hdim_v; // could be cross-attn, where V and Q/K hdim are different
    // naive_a.batch_q = batch;
    // naive_a.batch_kv = 1;           // decode case batch-kv always 1
    // naive_a.batch_ratio_kv = batch; // batch_q / batch_kv
    // naive_a.seqlen_q = 1;           // in decode case, this should be 1
    // naive_a.seqlen_kv = 0;          // if context_len_ptr is not nullptr, ignore this field
    // naive_a.nhead_q = nhead;
    // naive_a.nhead_kv = nhead_k;
    // naive_a.nhead_ratio_kv = naive_a.nhead_q / naive_a.nhead_kv; // nhead_q / nhead_kv
    // naive_a.page_size = block_size;                              // if paged, the seqlen-kv for each block

    // naive_a.kscale_ptr = k_dequant_scales.data_ptr();
    // naive_a.vscale_ptr = v_dequant_scales.data_ptr();
    // naive_a.max_pages_per_seq = max_num_blocks_per_seq;
    // naive_a.max_kv_tokens = max_kv_tokens;

    ck_tile::stream_config naive_s{stream};

    printf("caroo: %s, %d, %s\n", __FILE__, __LINE__, __FUNCTION__);

    mla_fwd(naive_t, naive_a, naive_s);

    O.fill_(7);
}