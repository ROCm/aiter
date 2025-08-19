#include "mha_common.h"
#include "py_itfs_common.h"
#include <ATen/cuda/CUDAContext.h>
#include <torch/all.h>

#include <string>
#include <type_traits>
#include <utility>

#include "fmha_fwd_v3.hpp"

#define DEBUG_DTYPE_FP16 0
#define DEBUG_DTYPE_BF16 1
#define DEBUG_MASK_NONE 0
#define DEBUG_MASK_CAUSAL 1

#define DEBUG_SINGLE_INST 0
#define DEBUG_SINGLE_INST_DTYPE DEBUG_DTYPE_BF16
#define DEBUG_SINGLE_INST_MASK DEBUG_MASK_NONE

namespace aiter {
namespace torch_itfs {
namespace {

float fmha_fwd_v3(const ck_tile::fmha_fwd_v3_args& args, const ck_tile::stream_config& config)
{
    float time = 0.0;

    // TODO: compile fp16/bf16, masking=true/false kernels separately
    if(args.data_type == ck_tile::fmha_fwd_v3_args::data_type_enum::fp16)
    {
        if(args.mask_type == static_cast<int>(mask_enum::no_mask))
        {
#if !DEBUG_SINGLE_INST || \
    (DEBUG_SINGLE_INST_DTYPE == DEBUG_DTYPE_FP16 && DEBUG_SINGLE_INST_MASK == DEBUG_MASK_NONE)
            time = fmha_fwd_v3_dispatch<
                type_tag<ck_tile::fmha_fwd_v3_args::data_type_enum::fp16, false>>(args, config);
#endif
        }
        else
        {
#if !DEBUG_SINGLE_INST || \
    (DEBUG_SINGLE_INST_DTYPE == DEBUG_DTYPE_FP16 && DEBUG_SINGLE_INST_MASK == DEBUG_MASK_CAUSAL)
            time = fmha_fwd_v3_dispatch<
                type_tag<ck_tile::fmha_fwd_v3_args::data_type_enum::fp16, true>>(args, config);
#endif
        }
    }
    else if(args.data_type == ck_tile::fmha_fwd_v3_args::data_type_enum::bf16)
    {
        if(args.mask_type == static_cast<int>(mask_enum::no_mask))
        {
#if !DEBUG_SINGLE_INST || \
    (DEBUG_SINGLE_INST_DTYPE == DEBUG_DTYPE_BF16 && DEBUG_SINGLE_INST_MASK == DEBUG_MASK_NONE)
            time = fmha_fwd_v3_dispatch<
                type_tag<ck_tile::fmha_fwd_v3_args::data_type_enum::bf16, false>>(args, config);
#endif
        }
        else
        {
#if !DEBUG_SINGLE_INST || \
    (DEBUG_SINGLE_INST_DTYPE == DEBUG_DTYPE_BF16 && DEBUG_SINGLE_INST_MASK == DEBUG_MASK_CAUSAL)
            time = fmha_fwd_v3_dispatch<
                type_tag<ck_tile::fmha_fwd_v3_args::data_type_enum::bf16, true>>(args, config);
#endif
        }
    }

    return 0.0;
}
} // namespace
//////////////////////////////////////////////////////////////////////////////////////

std::vector<at::Tensor> fmha_v3_fwd_ck(const at::Tensor& q, // [b, sq, hq, d]
                                       const at::Tensor& k, // [b, sk, hk, d]
                                       const at::Tensor& v, // [b, sk, hk, d_v]
                                       float softmax_scale,
                                       bool is_causal,
                                       int window_size_left,
                                       int window_size_right,
                                       bool return_softmax_lse)
{
    auto q_dtype = q.dtype();
    TORCH_CHECK(q_dtype == at::ScalarType::Half || q_dtype == at::ScalarType::BFloat16,
                "FlashAttention only support fp16 and bf16 data type");

    TORCH_CHECK(k.dtype() == q_dtype, "query and key must have the same dtype");
    TORCH_CHECK(v.dtype() == q_dtype, "query and value must have the same dtype");

    CHECK_DEVICE(q);
    CHECK_DEVICE(k);
    CHECK_DEVICE(v);

    TORCH_CHECK(q.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(k.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(v.stride(-1) == 1, "Input tensor must have contiguous last dimension");

    const auto sizes = q.sizes();

    const int batch_size  = sizes[0];
    int seqlen_q          = sizes[1];
    int num_heads         = sizes[2];
    const int head_size_q = sizes[3];
    const int head_size_v = v.sizes()[3];
    const int seqlen_k    = k.size(1);
    const int num_heads_k = k.size(2);
    TORCH_CHECK(batch_size > 0, "batch size must be positive");
    TORCH_CHECK(head_size_q <= 256, "CK only supports head dimension at most 256");
    TORCH_CHECK(head_size_v <= 256, "CK only supports head dimension at most 256");
    TORCH_CHECK(head_size_q % 8 == 0,
                "query, key, value, and out_ must have a head_size_q that is a multiple of 8");
    TORCH_CHECK(head_size_v % 8 == 0,
                "query, key, value, and out_ must have a head_size_q that is a multiple of 8");
    TORCH_CHECK(
        num_heads % num_heads_k == 0,
        "ck_tile::number of heads in key/value must divide ck_tile::number of heads in query");

    CHECK_SHAPE(q, batch_size, seqlen_q, num_heads, head_size_q);
    CHECK_SHAPE(k, batch_size, seqlen_k, num_heads_k, head_size_q);
    CHECK_SHAPE(v, batch_size, seqlen_k, num_heads_k, head_size_v);

    if(window_size_left >= seqlen_k)
    {
        window_size_left = -1;
    }
    if(window_size_right >= seqlen_k)
    {
        window_size_right = -1;
    }

    mask_info mask;
    if(is_causal)
    {
        // Causal is the special case where window_size_right == 0 and window_size_left < 0.
        window_size_right         = 0;
        std::string mask_identify = "b:" + std::to_string(window_size_left) + "," + "0";
        mask                      = mask_info::decode(mask_identify, seqlen_q, seqlen_k); // casual
    }
    else if(window_size_left == -1 && window_size_right == -1)
    {
        mask = mask_info::decode("0", seqlen_q, seqlen_k); // no mask
    }
    else
    {
        // Local is the more general case where window_size_right >= 0 or window_size_left >= 0.
        std::string mask_identify =
            "b:" + std::to_string(window_size_left) + "," + std::to_string(window_size_right);
        mask = mask_info::decode(mask_identify, seqlen_q, seqlen_k); // local
    }

    ck_tile::fmha_fwd_v3_args args;

    args.data_type = (q_dtype == at::ScalarType::Half)
                         ? ck_tile::fmha_fwd_v3_args::data_type_enum::fp16
                         : ck_tile::fmha_fwd_v3_args::data_type_enum::bf16;

    args.batch    = batch_size;
    args.seqlen_q = seqlen_q;
    args.seqlen_k = seqlen_k;
    args.hdim_q   = head_size_q;
    args.hdim_v   = head_size_v;
    args.nhead_q  = num_heads;
    args.nhead_k  = num_heads_k;

    args.scale_s = softmax_scale;

    args.window_size_left  = mask.left;
    args.window_size_right = mask.right;
    args.mask_type         = static_cast<ck_tile::index_t>(mask.type);

    args.q_ptr          = q.data_ptr();
    args.batch_stride_q = q.stride(0);
    args.stride_q       = q.stride(1);
    args.nhead_stride_q = q.stride(2);

    args.k_ptr          = k.data_ptr();
    args.batch_stride_k = k.stride(0);
    args.stride_k       = k.stride(1);
    args.nhead_stride_k = k.stride(2);

    args.v_ptr          = v.data_ptr();
    args.batch_stride_v = v.stride(0);
    args.stride_v       = v.stride(1);
    args.nhead_stride_v = v.stride(2);

    auto opts = q.options();
    at::Tensor out =
        torch::empty({batch_size, seqlen_q, num_heads, head_size_v}, opts.dtype(q_dtype));

    args.o_ptr          = out.data_ptr();
    args.batch_stride_o = out.stride(0);
    args.stride_o       = out.stride(1);
    args.nhead_stride_o = out.stride(2);

    auto stream = at::cuda::getCurrentHIPStream().stream();
    ck_tile::stream_config stream_config{stream};

    fmha_fwd_v3(args, stream_config);

    return {out};
}

} // namespace torch_itfs
} // namespace aiter
