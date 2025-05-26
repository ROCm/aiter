// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include "py_itfs_common.h"
#include "mha_common.h"

#include "mha_fwd.h"

namespace aiter {
namespace torch_itfs {
fmha_batch_prefill_args
get_ck_fmha_batch_prefill_args(bool has_lse,
                               bool has_dropout_randval,
                               const mask_info& mask,
                               // sizes
                               const int b,
                               const int max_seqlen_q,
                               const int h,
                               const int h_k,
                               const int d,
                               const int d_v,
                               // device pointers
                               const at::Tensor q,
                               const at::Tensor k,
                               const at::Tensor v,
                               const at::Tensor seqlens_q,
                               const at::Tensor kv_indptr,
                               const at::Tensor kv_page_indices,
                               std::optional<const at::Tensor>& bias_,
                               std::optional<const at::Tensor>& alibi_slopes_,
                               at::Tensor out,
                               at::Tensor softmax_lse,
                               at::Tensor dropout_randval,
                               float softmax_scale,
                               float logits_soft_cap,
                               float p_dropout,
                               std::pair<uint64_t*, uint64_t*> drop_seed_offset)
{
    // q: (total_q, nheads, d)
    // k: (total_k, nheads_k, d)
    // v: (total_k, nheads_k, d_v)
    // o: (total_q, nheads, d_v)

    // bias:(total_q, max_seqlen_k)
    // alibi_slopes:(batch, nheads) or (nhead)
    // lse: (nheads, total_q)
    // randval: (nheads, total_q, max_seqlen_k)

    ck_tile::index_t total_q = q.size(0);
    ck_tile::index_t total_k = k.size(0);

    ck_tile::index_t stride_q       = q.stride(0);
    ck_tile::index_t stride_k       = k.stride(0);
    ck_tile::index_t stride_v       = v.stride(0);
    ck_tile::index_t stride_o       = out.stride(0);
    ck_tile::index_t stride_randval = has_dropout_randval ? dropout_randval.stride(1) : 0;

    ck_tile::index_t nhead_stride_q       = q.stride(1);
    ck_tile::index_t nhead_stride_k       = k.stride(1);
    ck_tile::index_t nhead_stride_v       = v.stride(1);
    ck_tile::index_t nhead_stride_o       = out.stride(1);
    ck_tile::index_t nhead_stride_lse     = has_lse ? softmax_lse.stride(0) : 0;
    ck_tile::index_t nhead_stride_randval = has_dropout_randval ? dropout_randval.stride(0) : 0;

    ck_tile::index_t batch_stride_q       = 0;
    ck_tile::index_t batch_stride_k       = 0;
    ck_tile::index_t batch_stride_v       = 0;
    ck_tile::index_t batch_stride_o       = 0;
    ck_tile::index_t batch_stride_lse     = 0;
    ck_tile::index_t batch_stride_randval = 0;

    void* bias_ptr                     = nullptr;
    ck_tile::index_t stride_bias       = 0;
    ck_tile::index_t nhead_stride_bias = 0;
    ck_tile::index_t batch_stride_bias = 0;

    if(bias_.has_value())
    {
        auto bias = bias_.value();
        CHECK_DEVICE(bias);
        TORCH_CHECK(bias.stride(-1) == 1, "bias tensor must have contiguous last dimension");
        TORCH_CHECK(bias.dim() == 2, "only support 2d bias");
        bias_ptr = bias.data_ptr();
        if(bias.dim() == 2)
            stride_bias = bias.stride(0);
    }
    else if(alibi_slopes_.has_value())
    {
        auto alibi_slopes = alibi_slopes_.value();
        CHECK_DEVICE(alibi_slopes);
        TORCH_CHECK(alibi_slopes.stride(-1) == 1,
                    "ALiBi slopes tensor must have contiguous last dimension");
        TORCH_CHECK(alibi_slopes.sizes() == torch::IntArrayRef({h}) ||
                    alibi_slopes.sizes() == torch::IntArrayRef({b, h}));
        bias_ptr    = alibi_slopes.data_ptr();
        stride_bias = alibi_slopes.dim() == 2 ? alibi_slopes.stride(0) : 0;
    }

    fmha_batch_prefill_args args;

    args.q_ptr           = q.data_ptr();
    args.k_ptr           = k.data_ptr();
    args.v_ptr           = v.data_ptr();
    args.bias_ptr        = bias_ptr;
    args.rand_val_ptr    = has_dropout_randval ? dropout_randval.data_ptr() : nullptr;
    args.lse_ptr         = has_lse ? softmax_lse.data_ptr() : nullptr;
    args.o_ptr           = out.data_ptr();
    args.seqstart_q_ptr  = seqlens_q.data_ptr();
    args.seqlen_q        = total_q;
    args.seqlen_k        = total_k;
    args.batch           = b;
    args.max_seqlen_q    = max_seqlen_q;
    args.hdim_q          = d;
    args.hdim_v          = d_v;
    args.nhead_q         = h;
    args.nhead_k         = h_k;
    args.num_total_pages = total_k;
    args.kv_indptr       = kv_indptr.data_ptr();
    args.kv_page_indices = kv_page_indices.data_ptr();
    args.scale_s         = softmax_scale;
    args.scale_p         = 1;
    args.scale_o         = 1;

    args.logits_soft_cap = logits_soft_cap;

    args.stride_q             = stride_q;
    args.stride_k             = stride_k;
    args.stride_v             = stride_v;
    args.stride_bias          = stride_bias;
    args.stride_randval       = stride_randval;
    args.stride_o             = stride_o;
    args.nhead_stride_q       = nhead_stride_q;
    args.nhead_stride_k       = nhead_stride_k;
    args.nhead_stride_v       = nhead_stride_v;
    args.nhead_stride_bias    = nhead_stride_bias;
    args.nhead_stride_randval = nhead_stride_randval;
    args.nhead_stride_lse     = nhead_stride_lse;
    args.nhead_stride_o       = nhead_stride_o;
    args.batch_stride_q       = batch_stride_q;
    args.batch_stride_k       = batch_stride_k;
    args.batch_stride_v       = batch_stride_v;
    args.batch_stride_bias    = batch_stride_bias;
    args.batch_stride_randval = batch_stride_randval;
    args.batch_stride_lse     = batch_stride_lse;
    args.batch_stride_o       = batch_stride_o;
    args.window_size_left     = mask.left;
    args.window_size_right    = mask.right;
    args.mask_type            = static_cast<ck_tile::index_t>(mask.type);
    args.p_drop               = p_dropout;
    args.s_randval            = has_dropout_randval;
    args.drop_seed_offset     = drop_seed_offset;

    return args;
}

std::vector<at::Tensor>
mha_batch_prefill(at::Tensor& q,                  // [total_q, hq, d]
                  const at::Tensor& k,            // [total_k, hk, d]
                  const at::Tensor& v,            // [total_k, hk, d]
                  const at::Tensor& cu_seqlens_q, // [b+1]
                  const at::Tensor& kv_indptr,    // [b+1]
                  const at::Tensor& kv_page_indices,
                  int max_seqlen_q,
                  int max_seqlen_k,
                  float p_dropout,
                  float softmax_scale,
                  float logits_soft_cap,
                  bool zero_tensors,
                  bool is_causal,
                  int window_size_left,
                  int window_size_right,
                  bool return_softmax_lse,
                  bool return_dropout_randval,
                  std::optional<at::Tensor> out_,                // [total_q, hq, d]
                  std::optional<const at::Tensor> bias_,         // [total_q, max_seqlen_k]
                  std::optional<const at::Tensor> alibi_slopes_, // [hq] or [b, hq]
                  std::optional<at::Generator> gen_)
{
    auto q_dtype = q.dtype();
    TORCH_CHECK(q_dtype == torch::kFloat16 || q_dtype == torch::kBFloat16,
                "FlashAttention only support fp16 and bf16 data type");

    TORCH_CHECK(k.dtype() == q_dtype, "query and key must have the same dtype");
    TORCH_CHECK(v.dtype() == q_dtype, "query and value must have the same dtype");
    TORCH_CHECK(cu_seqlens_q.dtype() == torch::kInt32, "cu_seqlens_q must have dtype int32");
    TORCH_CHECK(kv_indptr.dtype() == torch::kInt32, "kv_indptr must have dtype int32");

    std::string q_dtype_str = q_dtype == torch::kFloat16 ? "fp16" : "bf16";

    CHECK_DEVICE(q);
    CHECK_DEVICE(k);
    CHECK_DEVICE(v);
    CHECK_DEVICE(cu_seqlens_q);
    CHECK_DEVICE(kv_indptr);

    CHECK_DEVICE(kv_page_indices);
    TORCH_CHECK(kv_page_indices.dtype() == torch::kInt32,
                "kv_page_indices must have dtype torch.int32");
    TORCH_CHECK(kv_page_indices.stride(-1) == 1,
                "kv_page_indices must have contiguous last dimension");

    TORCH_CHECK(q.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(k.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(v.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    CHECK_CONTIGUOUS(cu_seqlens_q);
    CHECK_CONTIGUOUS(kv_indptr);

    const auto sizes = q.sizes();

    const int batch_size  = cu_seqlens_q.numel() - 1;
    int num_heads         = sizes[1];
    const int head_size_q = sizes[2];
    const int head_size_v = v.size(2);
    const int num_heads_k = k.size(1);

    const int num_blocks = k.size(0);

    if(max_seqlen_q == 1 && !alibi_slopes_.has_value())
    {
        is_causal = false;
    } // causal=true is the same as causal=false in this case

    TORCH_CHECK(!(bias_.has_value() && alibi_slopes_.has_value()),
                "cannot apply bias and alibi at the same time");
    bias_enum bias_type = bias_.has_value()           ? bias_enum::elementwise_bias
                          : alibi_slopes_.has_value() ? bias_type = bias_enum::alibi
                                                      : bias_enum::no_bias;

    // TODO
    // Faster to transpose q from (b, 1, (nheads_kv ngroups), d) to (b, ngroups, nheads_kv, d) in
    // this case H/t Daniel Haziza

    const int total_q = q.size(0);

    TORCH_CHECK(batch_size > 0, "batch size must be postive");
    TORCH_CHECK(head_size_q <= 256, "CK only supports head dimension at most 256");
    TORCH_CHECK(head_size_v <= 256, "CK only supports head dimension at most 256");
    TORCH_CHECK(head_size_q % 8 == 0,
                "query, key, value, and out_ must have a head_size that is a multiple of 8");
    TORCH_CHECK(head_size_v % 8 == 0,
                "query, key, value, and out_ must have a head_size that is a multiple of 8");
    TORCH_CHECK(num_heads % num_heads_k == 0,
                "Number of heads in key/value must divide number of heads in query");

    if(window_size_left >= max_seqlen_k)
    {
        window_size_left = -1;
    }
    if(window_size_right >= max_seqlen_k)
    {
        window_size_right = -1;
    }

    mask_info mask;

    if(is_causal)
    {
        // Causal is the special case where window_size_right == 0 and window_size_left < 0.
        window_size_right         = 0;
        std::string mask_identify = "b:" + std::to_string(window_size_left) + "," + "0";
        mask = mask_info::decode(mask_identify, max_seqlen_q, max_seqlen_k); // casual
    }
    else if(window_size_left == -1 && window_size_right == -1)
    {
        mask = mask_info::decode("0", max_seqlen_q, max_seqlen_k); // no mask
    }
    else
    {
        // Local is the more general case where window_size_right >= 0 or window_size_left >= 0.
        std::string mask_identify =
            "b:" + std::to_string(window_size_left) + "," + std::to_string(window_size_right);
        mask = mask_info::decode(mask_identify, max_seqlen_q, max_seqlen_k); // local
    }

    CHECK_SHAPE(q, total_q, num_heads, head_size_q);
    CHECK_SHAPE(k, num_blocks, num_heads_k, head_size_q);
    CHECK_SHAPE(v, num_blocks, num_heads_k, head_size_v);

    CHECK_SHAPE(cu_seqlens_q, batch_size + 1);
    CHECK_SHAPE(kv_indptr, batch_size + 1);
    auto opts = q.options();

    at::Tensor out;
    if(out_.has_value())
    {
        out = out_.value();
        TORCH_CHECK(out.dtype() == q_dtype, "Output must have the same dtype as inputs");
        CHECK_DEVICE(out);
        TORCH_CHECK(out.stride(-1) == 1, "Output tensor must have contiguous last dimension");
        CHECK_SHAPE(out, total_q, num_heads, head_size_v);
    }
    else
    {
        out = torch::empty({total_q, num_heads, head_size_v}, opts.dtype(q_dtype));
    }

    // Otherwise the kernel will be launched from cuda:0 device
    at::cuda::CUDAGuard device_guard{q.device()};

    bool has_lse     = return_softmax_lse;
    bool has_dropout = p_dropout > 0.0f;

    at::Tensor softmax_lse;
    if(return_softmax_lse)
    {
        softmax_lse = torch::empty({num_heads, total_q}, opts.dtype(torch::kFloat32));
    }
    else
    {
        softmax_lse = torch::empty({0}, opts.dtype(torch::kFloat32));
    }

    at::Tensor p;
    if(return_dropout_randval)
    {
        TORCH_CHECK(has_dropout, "return_dropout_randval require p_dropout > 0");
        p = torch::empty({num_heads, total_q, max_seqlen_k}, opts.dtype(torch::kUInt8));
    }
    else
    {
        p = torch::empty({0}, opts);
    }

    if(zero_tensors)
    {
        out.zero_();
        softmax_lse.fill_(-std::numeric_limits<float>::infinity());
        if(return_dropout_randval)
        {
            p.zero_();
        }
    }

    int64_t counter_offset = batch_size * num_heads * ck_tile::get_warp_size();
    auto rng_state         = torch::empty({2}, opts.dtype(torch::kInt64));
    auto rng_state_ptr     = reinterpret_cast<uint64_t*>(rng_state.data_ptr());

    if(p_dropout > 0.0)
    {
        auto gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(
            gen_, at::cuda::detail::getDefaultCUDAGenerator());
        // See Note [Acquire lock when using random generators]
        std::lock_guard<std::mutex> lock(gen->mutex_);
        auto philox_args = gen->philox_cuda_state(counter_offset);
        hipLaunchKernelGGL(
            aiter::ParsePhiloxCudaState, dim3(1), dim3(64), 0, 0, philox_args, rng_state_ptr);
    }

    if(max_seqlen_k > 0)
    {
        auto stream = at::cuda::getCurrentHIPStream().stream();
        ck_tile::stream_config stream_config{stream};

        auto drop_seed_offset = std::make_pair(rng_state_ptr, rng_state_ptr + 1);

        auto args = get_ck_fmha_batch_prefill_args(has_lse,
                                                   return_dropout_randval,
                                                   mask,
                                                   batch_size,
                                                   max_seqlen_q,
                                                   num_heads,
                                                   num_heads_k,
                                                   head_size_q,
                                                   head_size_v,
                                                   q,
                                                   k,
                                                   v,
                                                   cu_seqlens_q,
                                                   kv_indptr,
                                                   kv_page_indices,
                                                   bias_,
                                                   alibi_slopes_,
                                                   out,
                                                   softmax_lse,
                                                   p,
                                                   softmax_scale,
                                                   logits_soft_cap,
                                                   p_dropout,
                                                   drop_seed_offset);

        float t = aiter::mha_batch_prefill(args,
                                           stream_config,
                                           q_dtype_str,
                                           true, // is_group_mode
                                           mask.type,
                                           bias_type,
                                           has_lse,
                                           false);
        TORCH_CHECK(t >= 0, "invalid argument for fmha_fwd_splitkv");
    }
    else
    {
        // If seqlen_k == 0, then we have an empty tensor. We need to set the output to 0.
        out.zero_();
        softmax_lse.fill_(std::numeric_limits<float>::infinity());
    }

    return {out, softmax_lse, p, rng_state};
}

} // namespace torch_itfs
} // namespace aiter
