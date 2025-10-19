#include "mha_common.h"
#include "py_itfs_common.h"
#include <ATen/cuda/CUDAContext.h>
#include <torch/all.h>

#include <string>
#include <type_traits>
#include <utility>

#include "fmha_fwd_v3.hpp"
#include "mask.hpp"

namespace aiter {
namespace torch_itfs {

std::vector<at::Tensor> fmha_v3_fwd_ck(const at::Tensor& q, // [b, sq, hq, d]
                                       const at::Tensor& k, // [b, sk, hk, d]
                                       const at::Tensor& v, // [b, sk, hk, d_v]
                                       float softmax_scale,
                                       bool is_causal)
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

    mask_info mask;
    if(is_causal)
    {
        // Causal is the special case where window_size_right == 0 and window_size_left < 0.
        std::string mask_identify = "b:-1,0";
        mask                      = mask_info::decode(mask_identify, seqlen_q, seqlen_k); // casual
    }
    else
    {
        mask = mask_info::decode("0", seqlen_q, seqlen_k); // no mask
    }

    ck_tile::fmha_fwd_v3_args args;

    args.data_type = (q_dtype == at::ScalarType::Half)
                         ? ck_tile::fmha_fwd_v3_args::data_type_enum::fp16
                         : ck_tile::fmha_fwd_v3_args::data_type_enum::bf16;

    args.batch        = batch_size;
    args.max_seqlen_q = seqlen_q;
    args.seqlen_q     = seqlen_q;
    args.seqlen_k     = seqlen_k;
    args.hdim_qk      = head_size_q;
    args.hdim_v       = head_size_v;
    args.nhead_q      = num_heads;
    args.nhead_kv     = num_heads_k;

    args.softmax_scale = softmax_scale;

    args.seqstart_q_ptr = nullptr;
    args.seqstart_k_ptr = nullptr;
    args.seqlen_k_ptr   = nullptr;

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

    ck_tile::fmha_fwd_v3(args, stream_config);

    return {out};
}

} // namespace torch_itfs
} // namespace aiter
