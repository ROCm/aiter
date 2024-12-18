/*
 * Copyright (c) 2024 Advanced Micro Devices, Inc.  All rights reserved.
 *
 * @Script: attention_kernels.cu
 * @Author: valarLip
 * @Email: lingpeng.jin@amd.com
 * @Create At: 2024-12-04 20:28:50
 * @Last Modified By: valarLip
 * @Last Modified At: 2024-12-05 17:56:48
 * @Description: This is description.
 */

#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include "py_itfs_common.h"
#include "ck_tile/ref/naive_attention.hpp"

torch::Tensor pa_fwd_naive(torch::Tensor &Q, //   [num_seqs, num_heads, head_size]
                           torch::Tensor &K, //   [num_blocks, num_kv_heads, head_size/x, block_size, x]
                                             // or[num_batch, seqlen, num_kv_heads, head_size]
                           torch::Tensor &V, //   [num_blocks, num_kv_heads, head_size, block_size]
                                             // or[num_batch*seqlen, num_kv_heads, head_size]
                           torch::Tensor &block_tables, // [num_seqs, max_num_blocks_per_seq]
                           torch::Tensor &context_lens,
                           torch::Tensor &k_dequant_scales, // [num_heads, max_kv_tokens]
                           torch::Tensor &v_dequant_scales, // [num_heads, max_kv_tokens]
                           const int max_seq_len,
                           const int num_kv_heads,
                           const float scale_s,
                           const float scale_k,
                           const float scale_v,
                           const int block_size,
                           const int quant_algo    // 0: no quant, 1: per-token FP8 quant
)
{
    // TORCH_CHECK(scale_k == 1. && scale_v == 1., "only support 1.0 for now")
    torch::Tensor out = torch::empty_like(Q);
    int batch = Q.size(0);
    int nhead = Q.size(1);
    int nhead_k = V.size(1);
    int hdim_q = Q.size(2);
    int hdim_v = V.size(2);
    int max_num_blocks_per_seq = block_tables.size(1);
    int max_kv_tokens = k_dequant_scales.numel() == 0? 0 : k_dequant_scales.size(1);

    ck_tile::naive_attention_fwd_traits naive_t;
    naive_t.q_type = torchDTypeToStr(Q.dtype());
    naive_t.k_type = torchDTypeToStr(K.dtype());
    naive_t.v_type = torchDTypeToStr(V.dtype());
    naive_t.o_type = torchDTypeToStr(out.dtype());
    naive_t.q_layout = "bhsd";
    naive_t.k_layout = "phdsx"; // TODO
    naive_t.v_layout = "phds";  // TODO
    naive_t.o_layout = "bhsd";
    naive_t.variation = 2; // decode variation
    naive_t.quant_algo = quant_algo;

    ck_tile::naive_attention_fwd_args naive_a;
    naive_a.q_ptr = Q.data_ptr();
    naive_a.k_ptr = K.data_ptr();
    naive_a.v_ptr = V.data_ptr();
    naive_a.o_ptr = out.data_ptr();
    naive_a.scale_s = scale_s;
    naive_a.context_len_ptr = context_lens.data_ptr(); // used when seqlen kv come from a pointer
    naive_a.page_table_ptr = block_tables.data_ptr();  // [batch, num_blocks] seqlen_kv is in different block(paged attn)
    naive_a.hdim = hdim_q;
    naive_a.hdim_v = hdim_v; // could be cross-attn, where V and Q/K hdim are different
    naive_a.batch_q = batch;
    naive_a.batch_kv = 1;           // decode case batch-kv always 1
    naive_a.batch_ratio_kv = batch; // batch_q / batch_kv
    naive_a.seqlen_q = 1;           // in decode case, this should be 1
    naive_a.seqlen_kv = 0;          // if context_len_ptr is not nullptr, ignore this field
    naive_a.nhead_q = nhead;
    naive_a.nhead_kv = nhead_k;
    naive_a.nhead_ratio_kv = naive_a.nhead_q / naive_a.nhead_kv; // nhead_q / nhead_kv
    naive_a.page_size = block_size;                              // if paged, the seqlen-kv for each block

    naive_a.kscale_ptr = k_dequant_scales.data_ptr();
    naive_a.vscale_ptr = v_dequant_scales.data_ptr();
    naive_a.max_pages_per_seq = max_num_blocks_per_seq;
    naive_a.max_kv_tokens = max_kv_tokens;

    ck_tile::stream_config naive_s{};

    naive_attention_fwd(naive_t, naive_a, naive_s);
    return out;
}