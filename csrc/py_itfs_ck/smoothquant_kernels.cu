#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include "py_itfs_common.h"

#include "smoothquant.hpp"
#include "moe_smoothquant.hpp"

void smoothquant_fwd(torch::Tensor &out,     // [m ,n]
                     torch::Tensor &input,   // [m ,n]
                     torch::Tensor &x_scale, // [1 ,n]
                     torch::Tensor &y_scale) // [m ,1]
{
    auto dtype = input.dtype();
    TORCH_CHECK(dtype == torch::kFloat16 || dtype == torch::kBFloat16,
                "ck layernorm2d only support fp16 and bf16 data type");

    std::string dtype_str = torchDTypeToStr(input.dtype());
    int n = input.size(-1);
    int m = input.numel() / n;
    int stride = n;
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    smoothquant({
                    dtype_str // input  dtype
                },
                {input.data_ptr(),   // p_x
                 x_scale.data_ptr(), // p_x_scale
                 y_scale.data_ptr(), // p_y
                 out.data_ptr(),     // p_y_scale
                 m, n, stride},
                {stream});
}

void moe_smoothquant_fwd(torch::Tensor &out,      // [topk * tokens, hidden_size]
                         torch::Tensor &input,    // [tokens, hidden_size]
                         torch::Tensor &x_scale,  // [experts, hidden_size]
                         torch::Tensor &topk_ids, // [tokens, topk]
                         torch::Tensor &y_scale)  // [topk * tokens,  1]
{
    auto dtype = input.dtype();
    TORCH_CHECK(dtype == torch::kFloat16 || dtype == torch::kBFloat16,
                "ck layernorm2d only support fp16 and bf16 data type");

    std::string dtype_str = torchDTypeToStr(input.dtype());
    int n = input.size(-1);
    int m = input.numel() / n;
    int experts = x_scale.size(0);
    int topk = topk_ids.size(1);
    int stride = n;
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    moe_smoothquant({
                        dtype_str // input  dtype
                    },
                    {input.data_ptr(),    // [tokens, hidden_size], input, fp16/bf16
                     x_scale.data_ptr(),  // [experts, hidden_size], input, columnwise scale, fp32
                     topk_ids.data_ptr(), // [tokens, topk]

                     y_scale.data_ptr(), // [topk * tokens,  1], output, rowwise quant scale
                     out.data_ptr(),     // [topk * tokens, hidden_size], output
                     m, n, experts, topk, stride, stride},
                    {stream});
}
