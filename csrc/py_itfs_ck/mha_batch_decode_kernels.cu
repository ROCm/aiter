// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include "py_itfs_common.h"
#include "mha_common.h"

#include "fmha_fwd.hpp"
#include "mask.hpp"

fmha_batch_decode_traits get_ck_fmha_batch_decode_traits(std::string dtype,
                                                         int head_size_q,
                                                         int head_size_v,
                                                         bool has_logits_soft_cap,
                                                         bool has_lse,
                                                         bool enable_alibi)
{
    return fmha_batch_decode_traits{head_size_q,
                                    head_size_v,
                                    dtype,
                                    false, // is_group_mode
                                    true,  // is_v_rowmajor
                                    has_logits_soft_cap,
                                    mask_enum::no_mask,
                                    enable_alibi ? bias_enum::alibi : bias_enum::no_bias,
                                    has_lse,
                                    false}; // do_fp8_static_quant
}

fmha_batch_decode_args get_ck_fmha_batch_decode_args(bool has_lse,
                                                     const int b,
                                                     const int h,
                                                     const int h_k,
                                                     const int d,
                                                     const int d_v,
                                                     const int num_splits,
                                                     float softmax_scale,
                                                     float logits_soft_cap,
                                                     // device pointers
                                                     const at::Tensor q,
                                                     const at::Tensor k,
                                                     const at::Tensor v,
                                                     const at::Tensor kv_indptr,
                                                     const at::Tensor kv_page_indices,
                                                     std::optional<const at::Tensor>& alibi_slopes_,
                                                     at::Tensor out,
                                                     at::Tensor lse,
                                                     at::Tensor lse_acc,
                                                     at::Tensor out_acc)
{
    // q: (batch_size, seqlen_q, nheads, d)
    // k: (num_blocks, page_block_size, num_heads_k, d)
    // v: (num_blocks, page_block_size, num_heads_k, d_v)
    // o: (batch_size, seqlen_q, nheads, d_v)

    // alibi_slopes:(batch_size, nheads) or (nhead)
    // lse: (nheads, seqlen_q)
    // lse_acc: (nheads, split, seqlen_q)
    // o_acc: (nheads, split, seqlen_q, d_v)
    // block_table: (batch_size, max_num_blocks_per_seq)

    fmha_batch_decode_args args;
    args.q_ptr       = q.data_ptr();
    args.k_ptr       = k.data_ptr();
    args.v_ptr       = v.data_ptr();
    args.bias_ptr    = nullptr;
    args.lse_acc_ptr = lse_acc.data_ptr();
    args.o_acc_ptr   = out_acc.data_ptr();
    args.lse_ptr     = nullptr;
    args.o_ptr       = out.data_ptr();

    args.num_total_pages = k.size(0);
    args.kv_indptr       = kv_indptr.data_ptr();
    args.kv_page_indices = kv_page_indices.data_ptr();

    args.seqstart_q_ptr = nullptr;

    args.seqlen_q = args.max_seqlen_q = 1;
    args.batch                        = b;
    args.hdim_q                       = d;
    args.hdim_v                       = d_v;
    args.nhead_q                      = h;
    args.nhead_k                      = h_k;
    args.num_splits                   = num_splits;

    args.scale_s = softmax_scale;
    args.scale_p = 1;
    args.scale_o = 1;

    args.logits_soft_cap = logits_soft_cap;

    args.batch_stride_q = q.stride(0);
    args.stride_q       = q.stride(0);
    args.nhead_stride_q = q.stride(1);

    args.batch_stride_k = k.stride(0);
    args.stride_k       = k.stride(0);
    args.nhead_stride_k = k.stride(1);

    args.batch_stride_v = v.stride(0);
    args.stride_v       = v.stride(0);
    args.nhead_stride_v = v.stride(1);

    args.batch_stride_o = out.stride(0);
    args.stride_o       = out.stride(0);
    args.nhead_stride_o = out.stride(1);

    args.batch_stride_bias = 0;
    args.stride_bias       = 0;
    args.nhead_stride_bias = 0;

    args.batch_stride_lse = 0;
    args.nhead_stride_lse = 0;

    args.batch_stride_lse_acc = lse_acc.stride(0);
    args.nhead_stride_lse_acc = lse_acc.stride(1);
    args.split_stride_lse_acc = lse_acc.stride(2);

    args.batch_stride_o_acc = out_acc.stride(0);
    args.nhead_stride_o_acc = out_acc.stride(1);
    args.split_stride_o_acc = out_acc.stride(2);
    args.stride_o_acc       = out_acc.stride(3);

    if(has_lse)
    {
        args.lse_ptr          = lse.data_ptr();
        args.batch_stride_lse = lse.stride(0);
        args.nhead_stride_lse = lse.stride(1);
    }

    if(alibi_slopes_.has_value())
    {
        auto alibi_slopes = alibi_slopes_.value();
        CHECK_DEVICE(alibi_slopes);
        TORCH_CHECK(alibi_slopes.stride(-1) == 1,
                    "ALiBi slopes tensor must have contiguous last dimension");
        TORCH_CHECK(alibi_slopes.sizes() == torch::IntArrayRef({h}) ||
                    alibi_slopes.sizes() == torch::IntArrayRef({b, h}));
        args.bias_ptr    = alibi_slopes.data_ptr();
        args.stride_bias = alibi_slopes.dim() == 2 ? alibi_slopes.stride(0) : 0;
    }

    args.window_size_left  = -1;
    args.window_size_right = -1;
    args.mask_type         = static_cast<ck_tile::index_t>(mask_enum::no_mask);

    return args;
}

std::vector<at::Tensor>
mha_batch_decode(at::Tensor& q,               // [b, hq, d]
                 const at::Tensor& k,         // [total_k, hk, d]
                 const at::Tensor& v,         // [total_k, hk, d]
                 const at::Tensor& kv_indptr, // [b+1]
                 const at::Tensor& kv_page_indices,
                 float softmax_scale,
                 float logits_soft_cap,
                 bool zero_tensors,
                 bool return_softmax_lse,
                 std::optional<at::Tensor> out_,               // [b, hq, d]
                 std::optional<const at::Tensor> alibi_slopes_ // [hq] or [b, hq])
)
{
    auto q_dtype = q.dtype();
    TORCH_CHECK(q_dtype == torch::kFloat16 || q_dtype == torch::kBFloat16,
                "FlashAttention only support fp16 and bf16 data type");

    TORCH_CHECK(k.dtype() == q_dtype, "query and key must have the same dtype");
    TORCH_CHECK(v.dtype() == q_dtype, "query and value must have the same dtype");
    TORCH_CHECK(kv_indptr.dtype() == torch::kInt32, "kv_indptr must have dtype int32");

    std::string q_dtype_str = q_dtype == torch::kFloat16 ? "fp16" : "bf16";

    CHECK_DEVICE(q);
    CHECK_DEVICE(k);
    CHECK_DEVICE(v);
    CHECK_DEVICE(kv_indptr);

    CHECK_DEVICE(kv_page_indices);
    TORCH_CHECK(kv_page_indices.dtype() == torch::kInt32,
                "kv_page_indices must have dtype torch.int32");
    TORCH_CHECK(kv_page_indices.stride(-1) == 1,
                "kv_page_indices must have contiguous last dimension");

    TORCH_CHECK(q.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(k.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(v.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    CHECK_CONTIGUOUS(kv_indptr);

    const auto sizes = q.sizes();

    const int batch_size  = sizes[0];
    int num_heads         = sizes[1];
    const int head_size_q = sizes[2];
    const int head_size_v = v.size(2);
    const int num_heads_k = k.size(1);

    const int max_num_blocks_per_seq = kv_page_indices.size(0);
    const int num_blocks             = k.size(0);
    const int page_block_size        = 1;

    // TODO
    // Faster to transpose q from (b, 1, (nheads_kv ngroups), d) to (b, ngroups, nheads_kv, d) in
    // this case H/t Daniel Haziza

    const int seqlen_q = 1;

    TORCH_CHECK(batch_size > 0, "batch size must be postive");
    TORCH_CHECK(head_size_q <= 256, "CK only supports head dimension at most 256");
    TORCH_CHECK(head_size_v <= 256, "CK only supports head dimension at most 256");
    TORCH_CHECK(head_size_q % 8 == 0,
                "query, key, value, and out_ must have a head_size that is a multiple of 8");
    TORCH_CHECK(head_size_v % 8 == 0,
                "query, key, value, and out_ must have a head_size that is a multiple of 8");
    TORCH_CHECK(num_heads % num_heads_k == 0,
                "Number of heads in key/value must divide number of heads in query");

    CHECK_SHAPE(q, batch_size, num_heads, head_size_q);
    CHECK_SHAPE(k, num_blocks, num_heads_k, head_size_q);
    CHECK_SHAPE(v, num_blocks, num_heads_k, head_size_v);

    CHECK_SHAPE(kv_indptr, batch_size + 1);
    auto opts = q.options();

    at::Tensor out;
    if(out_.has_value())
    {
        out = out_.value();
        TORCH_CHECK(out.dtype() == q_dtype, "Output must have the same dtype as inputs");
        CHECK_DEVICE(out);
        TORCH_CHECK(out.stride(-1) == 1, "Output tensor must have contiguous last dimension");
        CHECK_SHAPE(out, batch_size, num_heads, head_size_v);
    }
    else
    {
        out = torch::empty({batch_size, num_heads, head_size_v}, opts.dtype(q_dtype));
    }

    // Otherwise the kernel will be launched from cuda:0 device
    at::cuda::CUDAGuard device_guard{q.device()};

    bool has_logits_soft_cap = logits_soft_cap > 0.0f;
    bool has_lse             = return_softmax_lse;

    at::Tensor softmax_lse;
    if(return_softmax_lse)
    {
        softmax_lse = torch::empty({num_heads, seqlen_q}, opts.dtype(torch::kFloat32));
    }
    else
    {
        softmax_lse = torch::empty({0}, opts.dtype(torch::kFloat32));
    }

    if(zero_tensors)
    {
        out.zero_();
        softmax_lse.fill_(-std::numeric_limits<float>::infinity());
    }

    int num_splits = 0;
    // make sure the kM0 here is same as the one in codegen script
    num_splits = aiter::override_num_splits_if_necessary(
        batch_size, num_heads, seqlen_q, head_size_v, 0, num_splits, /*kM0=*/16);
    TORCH_CHECK(num_splits > 0, "num_splits should greater than 0");
    TORCH_CHECK(num_splits <= 128, "num_splits greater than 128 is not supported");

    auto softmax_lse_accum =
        torch::empty({batch_size, num_heads, num_splits, seqlen_q}, opts.dtype(at::kFloat));
    auto out_accum = torch::empty({batch_size, num_heads, num_splits, seqlen_q, head_size_v},
                                  opts.dtype(at::kFloat));

    auto stream = at::cuda::getCurrentHIPStream().stream();
    ck_tile::stream_config stream_config{stream};

    auto traits = get_ck_fmha_batch_decode_traits(q_dtype_str,
                                                  head_size_q,
                                                  head_size_v,
                                                  has_logits_soft_cap,
                                                  has_lse,
                                                  alibi_slopes_.has_value());

    auto args = get_ck_fmha_batch_decode_args(has_lse,
                                              batch_size,
                                              num_heads,
                                              num_heads_k,
                                              head_size_q,
                                              head_size_v,
                                              num_splits,
                                              softmax_scale,
                                              logits_soft_cap,
                                              q,
                                              k,
                                              v,
                                              kv_indptr,
                                              kv_page_indices,
                                              alibi_slopes_,
                                              out,
                                              softmax_lse,
                                              softmax_lse_accum,
                                              out_accum);

    float t = fmha_batch_decode(traits, args, stream_config);
    TORCH_CHECK(t >= 0, "invalid argument for fmha_batch_decode");

    return {out, softmax_lse};
}
