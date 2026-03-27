// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#include "py_itfs_common.h"
#include <ATen/hip/HIPContext.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <torch/all.h>

#include "unified_attention.hpp"

void unified_attention_fwd(
    torch::Tensor& output,          // [num_tokens, num_heads_q, head_size]
    torch::Tensor& query,           // [num_tokens, num_heads_q, head_size]
    torch::Tensor& key_cache,       // [num_blks, blk_size, num_kv_heads, head_size]
    torch::Tensor& value_cache,     // [num_blks, blk_size, num_kv_heads, head_size]
    torch::Tensor& block_tables,    // [num_seqs, max_num_blocks_per_seq]
    torch::Tensor& seq_lens,        // [num_seqs]
    torch::Tensor& query_start_len, // [num_seqs + 1]
    int mask_type,
    float scale_s,
    float scale,
    float scale_k,
    float scale_v,
    float scale_out)
{
    auto dtype = query.dtype();
    TORCH_CHECK(dtype == torch::kFloat16 || dtype == torch::kBFloat16,
                "unified_attention only supports fp16 and bf16 data types");
    TORCH_CHECK(block_tables.dtype() == torch::kInt32, "block_tables must be int32");
    TORCH_CHECK(seq_lens.dtype() == torch::kInt32, "seq_lens must be int32");
    TORCH_CHECK(query_start_len.dtype() == torch::kInt32, "query_start_len must be int32");

    ck_tile::unified_attention_args::data_type_enum dt;
    if (dtype == torch::kFloat16)
        dt = ck_tile::unified_attention_args::data_type_enum::fp16;
    else
        dt = ck_tile::unified_attention_args::data_type_enum::bf16;

    ck_tile::index_t num_tokens   = query.size(0);
    ck_tile::index_t num_head_q   = query.size(1);
    ck_tile::index_t hdim         = query.size(2);
    ck_tile::index_t num_kv_heads = key_cache.size(2);
    ck_tile::index_t num_blks     = key_cache.size(0);
    ck_tile::index_t page_blk_size = key_cache.size(1);
    ck_tile::index_t num_seqs     = seq_lens.size(0);
    ck_tile::index_t num_queries_per_kv = num_head_q / num_kv_heads;

    const at::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(device_of(query));
    const hipStream_t stream = at::hip::getCurrentHIPStream();

    ck_tile::unified_attention_args args;
    args.data_type          = dt;
    args.mask_type          = mask_type;
    args.num_tokens         = num_tokens;
    args.num_blks           = num_blks;
    args.num_head_q         = num_head_q;
    args.num_queries_per_kv = num_queries_per_kv;
    args.page_blk_size      = page_blk_size;
    args.hdim               = hdim;
    args.scale_s            = scale_s;
    args.scale              = scale;
    args.scale_k            = scale_k;
    args.scale_v            = scale_v;
    args.scale_out          = scale_out;

    args.q_ptr          = query.data_ptr();
    args.query_stride_0 = query.stride(0);
    args.query_stride_1 = query.stride(1);

    args.k_ptr            = key_cache.data_ptr();
    args.stride_k_cache_0 = key_cache.stride(0);
    args.stride_k_cache_1 = key_cache.stride(1);
    args.stride_k_cache_2 = key_cache.stride(2);
    args.stride_k_cache_3 = key_cache.stride(3);

    args.v_ptr            = value_cache.data_ptr();
    args.stride_v_cache_0 = value_cache.stride(0);
    args.stride_v_cache_1 = value_cache.stride(1);
    args.stride_v_cache_2 = value_cache.stride(2);
    args.stride_v_cache_3 = value_cache.stride(3);

    args.o_ptr           = output.data_ptr();
    args.output_stride_0 = output.stride(0);
    args.output_stride_1 = output.stride(1);

    args.block_tables_ptr  = block_tables.data_ptr<int32_t>();
    args.block_table_stride = block_tables.stride(0);
    args.seq_lens_ptr       = seq_lens.data_ptr<int32_t>();
    args.query_start_len_ptr = query_start_len.data_ptr<int32_t>();
    args.num_seqs            = num_seqs;

    auto [launched, elapsed] = ck_tile::unified_attention(args, {stream});
    TORCH_CHECK(launched,
                "unified_attention: no matching kernel for hdim=", hdim,
                " num_queries_per_kv=", num_queries_per_kv,
                " mask_type=", mask_type);
}
