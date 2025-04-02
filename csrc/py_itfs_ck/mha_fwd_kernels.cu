// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include "py_itfs_common.h"
#include "mha_common.h"

#include "fmha_fwd.hpp"
#include "mask.hpp"

namespace aiter {
namespace torch_itfs {
fmha_fwd_traits get_ck_fmha_fwd_traits(const mask_info &mask,
                                       std::string dtype,
                                       int head_size_q,
                                       int head_size_v,
                                       bool has_dropout,
                                       bool has_lse,
                                       bias_enum bias_type)
{
    return fmha_fwd_traits{head_size_q,
                           head_size_v,
                           dtype,
                           false, // is_group_mode
                           true,  // is_v_rowmajor
                           mask.type,
                           bias_type,
                           has_lse,
                           has_dropout,
                           false}; // do_fp8_static_quant
}

fmha_fwd_args get_ck_fmha_fwd_args(bool has_lse,
                                   bool has_dropout_randval,
                                   const mask_info &mask,
                                   // sizes
                                   const int b,
                                   const int seqlen_q,
                                   const int seqlen_k,
                                   const int h,
                                   const int h_k,
                                   const int d,
                                   const int d_v,
                                   // device pointers
                                   const at::Tensor q,
                                   const at::Tensor k,
                                   const at::Tensor v,
                                   std::optional<const at::Tensor> &bias_,
                                   std::optional<const at::Tensor> &alibi_slopes_,
                                   at::Tensor out,
                                   at::Tensor softmax_lse,
                                   at::Tensor dropout_randval,
                                   float softmax_scale,
                                   float p_dropout,
                                   std::pair<uint64_t*, uint64_t*> drop_seed_offset)
{
    // q: (batch_size, seqlen_q, nheads, d)
    // k: (batch_size, seqlen_k, nheads_k, d)
    // v: (batch_size, seqlen_k, nheads_k, d_v)
    // o: (batch_size, seqlen_q, nheads, d_v)

    // bias:(seqlen_q, seqlen_k)
    // alibi_slopes:(batch_size, nheads) or (nhead)
    // lse: (batch_size, nheads, seqlen_q)
    // randval: (batch_size, nheads, seqlen_q, seqlen_k)

    ck_tile::index_t stride_q = q.stride(1);
    ck_tile::index_t stride_k = k.stride(1);
    ck_tile::index_t stride_v = v.stride(1);
    ck_tile::index_t stride_o = out.stride(1);
    ck_tile::index_t stride_randval = has_dropout_randval ? dropout_randval.stride(2) : 0;

    ck_tile::index_t nhead_stride_q = q.stride(2);
    ck_tile::index_t nhead_stride_k = k.stride(2);
    ck_tile::index_t nhead_stride_v = v.stride(2);
    ck_tile::index_t nhead_stride_o = out.stride(2);
    ck_tile::index_t nhead_stride_lse = has_lse ? softmax_lse.stride(1) : 0;
    ck_tile::index_t nhead_stride_randval = has_dropout_randval ? dropout_randval.stride(1) : 0;

    ck_tile::index_t batch_stride_q = q.stride(0);
    ck_tile::index_t batch_stride_k = k.stride(0);
    ck_tile::index_t batch_stride_v = v.stride(0);
    ck_tile::index_t batch_stride_o = out.stride(0);

    ck_tile::index_t batch_stride_lse = has_lse ? softmax_lse.stride(0) : 0;
    ck_tile::index_t batch_stride_randval = has_dropout_randval ? dropout_randval.stride(0) : 0;

    void *bias_ptr = nullptr;
    ck_tile::index_t stride_bias = 0;

    if (bias_.has_value()) {
        auto bias = bias_.value();
        CHECK_DEVICE(bias);
        TORCH_CHECK(bias.stride(-1) == 1, "bias tensor must have contiguous last dimension");
        TORCH_CHECK(bias.sizes() == torch::IntArrayRef({seqlen_q, seqlen_k}), "bias shape should be [sq, sk]");
        bias_ptr = bias.data_ptr();
        stride_bias = bias.stride(0);
    }
    else if (alibi_slopes_.has_value()) {
        auto alibi_slopes = alibi_slopes_.value();
        CHECK_DEVICE(alibi_slopes);
        TORCH_CHECK(alibi_slopes.stride(-1) == 1, "ALiBi slopes tensor must have contiguous last dimension");
        TORCH_CHECK(alibi_slopes.sizes() == torch::IntArrayRef({h}) || alibi_slopes.sizes() == torch::IntArrayRef({b, h}));
        bias_ptr = alibi_slopes.data_ptr();
        stride_bias = alibi_slopes.dim() == 2 ? alibi_slopes.stride(0) : 0;
    }

    return fmha_fwd_args{q.data_ptr(),
                         k.data_ptr(),
                         v.data_ptr(),
                         bias_ptr,
                         has_dropout_randval ? dropout_randval.data_ptr() : nullptr,
                         has_lse ? softmax_lse.data_ptr() : nullptr,
                         out.data_ptr(),
                         nullptr, // seqstart_q
                         nullptr, // seqstart_k
                         nullptr,
                         seqlen_q,
                         seqlen_k,
                         b,
                         seqlen_q,      // max_seqlen_q
                         d,             // hdim_q
                         d_v,           // hdim_v
                         h,             // nhead
                         h_k,           // nhead_k
                         softmax_scale, // scale_s
                         1,             // scale_p
                         1,             // scale_o
                         stride_q,
                         stride_k,
                         stride_v,
                         stride_bias,
                         stride_randval,
                         stride_o,
                         nhead_stride_q,
                         nhead_stride_k,
                         nhead_stride_v,
                         0, // nhead_stride_bias
                         nhead_stride_randval,
                         nhead_stride_lse,
                         nhead_stride_o,
                         batch_stride_q,
                         batch_stride_k,
                         batch_stride_v,
                         0, // batch_stride_bias
                         batch_stride_randval,
                         batch_stride_lse,
                         batch_stride_o,
                         mask.left,
                         mask.right,
                         static_cast<ck_tile::index_t>(mask.type),
                         p_dropout,
                         has_dropout_randval,
                         drop_seed_offset};
}

std::vector<at::Tensor>
mha_fwd(at::Tensor &q, // [b, sq, hq, d]
        const at::Tensor &k, // [b, sk, hk, d]
        const at::Tensor &v, // [b, sk, hk, d_v]
        float p_dropout,
        float softmax_scale,
        bool is_causal,
        int window_size_left,
        int window_size_right,
        bool return_softmax_lse,
        bool return_dropout_randval,
        std::optional<at::Tensor> out_,          // [b, sq, hq, d_v]
        std::optional<const at::Tensor> bias_,   // [sq, sk]
        std::optional<const at::Tensor> alibi_slopes_, // [hq] or [b, hq]
        std::optional<at::Generator> gen_)
{
    auto q_dtype = q.dtype();
    TORCH_CHECK(q_dtype == torch::kFloat16 || q_dtype == torch::kBFloat16,
                "FlashAttention only support fp16 and bf16 data type");

    TORCH_CHECK(k.dtype() == q_dtype, "query and key must have the same dtype");
    TORCH_CHECK(v.dtype() == q_dtype, "query and value must have the same dtype");

    std::string q_dtype_str = q_dtype == torch::kFloat16 ? "fp16" : "bf16";

    CHECK_DEVICE(q); CHECK_DEVICE(k); CHECK_DEVICE(v);

    TORCH_CHECK(q.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(k.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(v.stride(-1) == 1, "Input tensor must have contiguous last dimension");

    const auto sizes = q.sizes();

    const int batch_size = sizes[0];
    int seqlen_q = sizes[1];
    int num_heads = sizes[2];
    const int head_size_q = sizes[3];
    const int head_size_v = v.sizes()[3];
    const int seqlen_k = k.size(1);
    const int num_heads_k = k.size(2);
    TORCH_CHECK(batch_size > 0, "batch size must be positive");
    TORCH_CHECK(head_size_q <= 256, "CK only supports head dimension at most 256");
    TORCH_CHECK(head_size_v <= 256, "CK only supports head dimension at most 256");
    TORCH_CHECK(head_size_q % 8 == 0, "query, key, value, and out_ must have a head_size_q that is a multiple of 8");
    TORCH_CHECK(head_size_v % 8 == 0, "query, key, value, and out_ must have a head_size_q that is a multiple of 8");
    TORCH_CHECK(num_heads % num_heads_k == 0, "Number of heads in key/value must divide number of heads in query");

    if (window_size_left >= seqlen_k) { window_size_left = -1; }
    if (window_size_right >= seqlen_k) { window_size_right = -1; }

    // causal=true is the same as causal=false in this case
    if (seqlen_q == 1 && !alibi_slopes_.has_value()) { is_causal = false; }

    mask_info mask;
    if (is_causal) {
        // Causal is the special case where window_size_right == 0 and window_size_left < 0.
        window_size_right = 0;
        std::string mask_identify = "b:" + std::to_string(window_size_left) + "," + "0";
        mask = mask_info::decode(mask_identify, seqlen_q, seqlen_k); // casual
    }
    else if (window_size_left == -1 && window_size_right == -1) {
        mask = mask_info::decode("0", seqlen_q, seqlen_k); // no mask
    }
    else {
        // Local is the more general case where window_size_right >= 0 or window_size_left >= 0.
        std::string mask_identify = "b:" + std::to_string(window_size_left) + "," + std::to_string(window_size_right);
        mask = mask_info::decode(mask_identify, seqlen_q, seqlen_k); // local
    }

    TORCH_CHECK(!(bias_.has_value() && alibi_slopes_.has_value()), "cannot apply bias and alibi at the same time");
    bias_enum bias_type = bias_.has_value() ? bias_enum::elementwise_bias :
        alibi_slopes_.has_value() ? bias_type = bias_enum::alibi : bias_enum::no_bias;

    // Faster to transpose q from (b, 1, (nheads_kv ngroups), d) to (b, ngroups, nheads_kv, d) in this case
    // H/t Daniel Haziza
    const int seqlenq_ngroups_swapped = seqlen_q == 1 && num_heads > num_heads_k &&
        window_size_left < 0 && window_size_right < 0 && p_dropout == 0.f && head_size_q % 8 == 0 &&
        !alibi_slopes_.has_value() && !bias_.has_value();
    const int ngroups = num_heads / num_heads_k;
    if (seqlenq_ngroups_swapped) {
        q = q.reshape({batch_size, num_heads_k, ngroups, head_size_q}).transpose(1, 2);
        seqlen_q = ngroups;
        num_heads = num_heads_k;
    }

    CHECK_SHAPE(q, batch_size, seqlen_q, num_heads, head_size_q);
    CHECK_SHAPE(k, batch_size, seqlen_k, num_heads_k, head_size_q);
    CHECK_SHAPE(v, batch_size, seqlen_k, num_heads_k, head_size_v);

    auto opts = q.options();
    at::Tensor out;
    if (out_.has_value()) {
        out = out_.value();
        TORCH_CHECK(out.dtype() == q_dtype, "Output must have the same dtype as inputs");
        CHECK_DEVICE(out);
        TORCH_CHECK(out.stride(-1) == 1, "Output tensor must have contiguous last dimension");
        CHECK_SHAPE(out, batch_size, sizes[1], sizes[2], head_size_v);
        if (seqlenq_ngroups_swapped) {
            out = out.reshape({batch_size, num_heads_k, ngroups, head_size_v}).transpose(1, 2);
        }
    }
    else {
        out = torch::empty({batch_size, seqlen_q, num_heads, head_size_v}, opts.dtype(q_dtype));
    }

    // Otherwise the kernel will be launched from cuda:0 device
    at::cuda::CUDAGuard device_guard{q.device()};

    bool has_lse = return_softmax_lse;
    bool has_dropout = p_dropout > 0.0f;

    at::Tensor softmax_lse;
    if (return_softmax_lse) {
        softmax_lse = torch::empty({batch_size, num_heads, seqlen_q}, opts.dtype(torch::kFloat32));
    }
    else {
        softmax_lse = torch::empty({ 0 }, opts.dtype(torch::kFloat32));
    }

    at::Tensor p;
    if (return_dropout_randval) {
        TORCH_CHECK(has_dropout, "return_dropout_randval require p_dropout > 0");
        p = torch::empty({batch_size, num_heads, seqlen_q, seqlen_k}, opts.dtype(torch::kUInt8));
    }
    else {
        p = torch::empty({ 0 }, opts);
    }

    int64_t counter_offset = batch_size * num_heads * ck_tile::get_warp_size();
    auto rng_state = torch::empty({2}, opts.dtype(torch::kInt64));
    auto rng_state_ptr = reinterpret_cast<uint64_t*>(rng_state.data_ptr());

    if (p_dropout > 0.0)  {
        auto gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(
            gen_, at::cuda::detail::getDefaultCUDAGenerator());
        // See Note [Acquire lock when using random generators]
        std::lock_guard<std::mutex> lock(gen->mutex_);
        auto philox_args = gen->philox_cuda_state(counter_offset);
        hipLaunchKernelGGL(
            aiter::ParsePhiloxCudaState, dim3(1), dim3(64), 0, 0, philox_args, rng_state_ptr);
    }

    if (seqlen_k > 0) {
        auto drop_seed_offset = std::make_pair(rng_state_ptr, rng_state_ptr + 1);
        auto stream = at::cuda::getCurrentHIPStream().stream();
        ck_tile::stream_config stream_config{stream};

        auto traits =
            get_ck_fmha_fwd_traits(
                mask,
                q_dtype_str,
                head_size_q,
                head_size_v,
                has_dropout,
                has_lse,
                bias_type);

        auto args =
            get_ck_fmha_fwd_args(
                has_lse,
                return_dropout_randval,
                mask,
                batch_size,
                seqlen_q,
                seqlen_k,
                num_heads,
                num_heads_k,
                head_size_q,
                head_size_v,
                q,
                k,
                v,
                bias_,
                alibi_slopes_,
                out,
                softmax_lse,
                p,
                softmax_scale,
                p_dropout,
                drop_seed_offset);

        float t = fmha_fwd(traits, args, stream_config);
        TORCH_CHECK(t >= 0, "invalid argument for fmha_fwd");
    }
    else {
        // If seqlen_k == 0, then we have an empty tensor. We need to set the output to 0.
        out.zero_();
        softmax_lse.fill_(std::numeric_limits<float>::infinity());
    }

    if (seqlenq_ngroups_swapped) {
        out = out.transpose(1, 2).reshape({batch_size, 1, num_heads_k * seqlen_q, head_size_q});
        q = q.transpose(1, 2).reshape({batch_size, 1, num_heads_k * seqlen_q, head_size_q});
        if (has_lse) {
            softmax_lse = softmax_lse.reshape({batch_size, num_heads_k * seqlen_q, 1});
        }
    }
    return {out, softmax_lse, p, rng_state};
}

} // namespace torch_itfs
} // namespace aiter
