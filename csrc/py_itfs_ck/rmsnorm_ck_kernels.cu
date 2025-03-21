// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
#include <torch/all.h>
#include <ATen/hip/HIPContext.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include "py_itfs_common.h"

#include "rmsnorm2d_fwd.hpp"

void rmsnorm2d(torch::Tensor &out,    // [m, n]
               torch::Tensor &input,  // [m, n]
               torch::Tensor &weight, // [1, n]
               double epsilon)
{
    auto dtype = input.dtype();
    TORCH_CHECK(dtype == torch::kFloat16 || dtype == torch::kBFloat16,
                "ck rmsnorm2d only support fp16 and bf16 data type");

    std::string dtype_str = torchDTypeToStr(dtype);
    int n = input.size(-1);
    int m = input.numel() / n;
    int stride = input.stride(0);
    int xr_stride = -1;
    int y_stride = out.stride(0);
    int yr_stride = -1;
    bool SaveRms = false;
    const at::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(device_of(input));
    const hipStream_t stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA();

    rmsnorm2d_fwd({
                      dtype_str, // input precision
                      dtype_str, // output precision
                      dtype_str, // x-scale, used for [1*N] input smooth quant
                      dtype_str, // y-scale, used for [M*1] output for next layer
                      SaveRms,
                      0, // fused_add
                      0  // fused_quant
                  },
                  {input.data_ptr(),
                   nullptr, // p_x_residual
                   nullptr, // p_x_scale
                   weight.data_ptr(), out.data_ptr(),
                   nullptr, // p_y_residual
                   nullptr, // p_y_scale
                   nullptr, // p_invRms
                   static_cast<float>(epsilon), m, n, stride, xr_stride, y_stride, yr_stride},
                  {stream});
}

torch::Tensor rmsnorm2d(torch::Tensor &input,  // [m, n]
                        torch::Tensor &weight, // [1, n]
                        double epsilon)
{
    torch::Tensor out = torch::empty_like(input);
    rmsnorm2d(out, input, weight, epsilon);

    return out;
}

void rmsnorm2d_with_add(torch::Tensor &out,          // [m ,n]
                        torch::Tensor &input,        // [m ,n]
                        torch::Tensor &residual_in,  // [m ,n]
                        torch::Tensor &residual_out, // [m ,n]
                        torch::Tensor &weight,       // [1 ,n]
                        double epsilon)
{
    auto dtype = input.dtype();
    TORCH_CHECK(dtype == torch::kFloat16 || dtype == torch::kBFloat16,
                "ck rmsnorm2d only support fp16 and bf16 data type");

    std::string dtype_str = torchDTypeToStr(input.dtype());
    int n = input.size(-1);
    int m = input.numel() / n;
    int stride = input.stride(0);
    int xr_stride = residual_in.stride(0);
    int y_stride = out.stride(0);
    int yr_stride = residual_out.stride(0);
    bool SaveRms = false;
    const at::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(device_of(input));
    const hipStream_t stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA();

    rmsnorm2d_fwd({
                      dtype_str, // input precision
                      dtype_str, // output precision
                      dtype_str, // x-scale, used for [1*N] input smooth quant
                      dtype_str, // y-scale, used for [M*1] output for next layer
                      SaveRms,
                      1, // fused_add
                      0  // fused_quant
                  },
                  {input.data_ptr(),        // p_x
                   residual_in.data_ptr(),  // p_x_residual
                   nullptr,                 // p_x_scale
                   weight.data_ptr(),       // p_gamma
                   out.data_ptr(),          // p_y
                   residual_out.data_ptr(), // p_y_residual
                   nullptr,                 // p_y_scale
                   nullptr,                 // p_invRms
                   static_cast<float>(epsilon), m, n, stride, xr_stride, y_stride, yr_stride},
                  {stream});
}

void rmsnorm2d_with_smoothquant(torch::Tensor &out,    // [m ,n]
                                torch::Tensor &input,  // [m ,n]
                                torch::Tensor &xscale, // [1 ,n]
                                torch::Tensor &yscale, // [m ,1]
                                torch::Tensor &weight, // [1 ,n]
                                double epsilon)
{
    auto dtype = input.dtype();
    TORCH_CHECK(dtype == torch::kFloat16 || dtype == torch::kBFloat16,
                "ck rmsnorm2d only support fp16 and bf16 data type");

    std::string dtype_str = torchDTypeToStr(input.dtype());
    std::string out_dtype_str = torchDTypeToStr(out.dtype());
    std::string xscale_dtype_str = torchDTypeToStr(xscale.dtype());
    std::string yscale_dtype_str = torchDTypeToStr(yscale.dtype());
    int n = input.size(-1);
    int m = input.numel() / n;
    int stride = input.stride(0);
    int xr_stride = -1;
    int y_stride = out.stride(0);
    int yr_stride = -1;
    bool SaveRms = false;
    const at::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(device_of(input));
    const hipStream_t stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA();

    rmsnorm2d_fwd({
                      dtype_str,        // input precision
                      out_dtype_str,    // output precision
                      xscale_dtype_str, // x-scale, used for [1*N] input smooth quant
                      yscale_dtype_str, // y-scale, used for [M*1] output for next layer
                      SaveRms,
                      0, // fused_add
                      1  // fused_quant
                  },
                  {input.data_ptr(),  // p_x
                   nullptr,           // p_x_residual
                   xscale.data_ptr(), // p_x_scale
                   weight.data_ptr(), // p_gamma
                   out.data_ptr(),    // p_y
                   nullptr,           // p_y_residual
                   yscale.data_ptr(), // p_y_scale
                   nullptr,           // p_invRms
                   static_cast<float>(epsilon), m, n, stride, xr_stride, y_stride, yr_stride},
                  {stream});
}

void rmsnorm2d_with_add_smoothquant(torch::Tensor &out,          // [m ,n]
                                    torch::Tensor &input,        // [m ,n]
                                    torch::Tensor &residual_in,  // [m ,n]
                                    torch::Tensor &residual_out, // [m ,n]
                                    torch::Tensor &xscale,       // [1 ,n]
                                    torch::Tensor &yscale,       // [m ,1]
                                    torch::Tensor &weight,       // [1 ,n]
                                    double epsilon)
{
    auto dtype = input.dtype();
    TORCH_CHECK(dtype == torch::kFloat16 || dtype == torch::kBFloat16,
                "ck rmsnorm2d only support fp16 and bf16 data type");

    std::string dtype_str = torchDTypeToStr(input.dtype());
    std::string out_dtype_str = torchDTypeToStr(out.dtype());
    std::string xscale_dtype_str = torchDTypeToStr(xscale.dtype());
    std::string yscale_dtype_str = torchDTypeToStr(yscale.dtype());
    int n = input.size(-1);
    int m = input.numel() / n;
    int stride = input.stride(0);
    int xr_stride = residual_in.stride(0);
    int y_stride = out.stride(0);
    int yr_stride = residual_out.stride(0);
    bool SaveRms = false;
    const at::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(device_of(input));
    const hipStream_t stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA();

    rmsnorm2d_fwd({
                      dtype_str,        // input precision
                      out_dtype_str,    // output precision
                      xscale_dtype_str, // x-scale, used for [1*N] input smooth quant
                      yscale_dtype_str, // y-scale, used for [M*1] output for next layer
                      SaveRms,
                      1, // fused_add
                      1  // fused_quant
                  },
                  {input.data_ptr(),        // p_x
                   residual_in.data_ptr(),  // p_x_residual
                   xscale.data_ptr(),       // p_x_scale
                   weight.data_ptr(),       // p_gamma
                   out.data_ptr(),          // p_y
                   residual_out.data_ptr(), // p_y_residual
                   yscale.data_ptr(),       // p_y_scale
                   nullptr,                 // p_invRms
                   static_cast<float>(epsilon), m, n, stride, xr_stride, y_stride, yr_stride},
                  {stream});
}

void rmsnorm2d_with_dynamicquant(torch::Tensor &out,    // [m ,n]
                                 torch::Tensor &input,  // [m ,n]
                                 torch::Tensor &yscale, // [m ,1]
                                 torch::Tensor &weight, // [1 ,n]
                                 double epsilon)
{
    auto dtype = input.dtype();
    TORCH_CHECK(dtype == torch::kFloat16 || dtype == torch::kBFloat16,
                "ck rmsnorm2d only support fp16 and bf16 data type");

    std::string dtype_str = torchDTypeToStr(input.dtype());
    std::string out_dtype_str = torchDTypeToStr(out.dtype());
    std::string yscale_dtype_str = torchDTypeToStr(yscale.dtype());
    int n = input.size(-1);
    int m = input.numel() / n;
    int stride = input.stride(0);
    int xr_stride = -1;
    int y_stride = out.stride(0);
    int yr_stride = -1;
    bool SaveRms = false;
    const at::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(device_of(input));
    const hipStream_t stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA();

    rmsnorm2d_fwd({
                      dtype_str,        // input precision
                      out_dtype_str,    // output precision
                      dtype_str,        // x-scale, used for [1*N] input smooth quant
                      yscale_dtype_str, // y-scale, used for [M*1] output for next layer
                      SaveRms,
                      0, // fused_add
                      2  // fused_quant
                  },
                  {input.data_ptr(),  // p_x
                   nullptr,           // p_x_residual
                   nullptr,           // p_x_scale
                   weight.data_ptr(), // p_gamma
                   out.data_ptr(),    // p_y
                   nullptr,           // p_y_residual
                   yscale.data_ptr(), // p_y_scale
                   nullptr,           // p_invRms
                   static_cast<float>(epsilon), m, n, stride, xr_stride, y_stride, yr_stride},
                  {stream});
}

void rmsnorm2d_with_add_dynamicquant(torch::Tensor &out,          // [m ,n]
                                     torch::Tensor &input,        // [m ,n]
                                     torch::Tensor &residual_in,  // [m ,n]
                                     torch::Tensor &residual_out, // [m ,n]
                                     torch::Tensor &yscale,       // [m ,1]
                                     torch::Tensor &weight,       // [1 ,n]
                                     double epsilon)
{
    auto dtype = input.dtype();
    TORCH_CHECK(dtype == torch::kFloat16 || dtype == torch::kBFloat16,
                "ck rmsnorm2d only support fp16 and bf16 data type");

    std::string dtype_str = torchDTypeToStr(input.dtype());
    std::string out_dtype_str = torchDTypeToStr(out.dtype());
    std::string yscale_dtype_str = torchDTypeToStr(yscale.dtype());
    int n = input.size(-1);
    int m = input.numel() / n;
    int stride = input.stride(0);
    int xr_stride = residual_in.stride(0);
    int y_stride = out.stride(0);
    int yr_stride = residual_out.stride(0);
    bool SaveRms = false;
    const at::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(device_of(input));
    const hipStream_t stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA();

    rmsnorm2d_fwd({
                      dtype_str,        // input precision
                      out_dtype_str,    // output precision
                      dtype_str,        // x-scale, used for [1*N] input smooth quant
                      yscale_dtype_str, // y-scale, used for [M*1] output for next layer
                      SaveRms,
                      1, // fused_add
                      2  // fused_quant
                  },
                  {input.data_ptr(),        // p_x
                   residual_in.data_ptr(),  // p_x_residual
                   nullptr,                 // p_x_scale
                   weight.data_ptr(),       // p_gamma
                   out.data_ptr(),          // p_y
                   residual_out.data_ptr(), // p_y_residual
                   yscale.data_ptr(),       // p_y_scale
                   nullptr,                 // p_invRms
                   static_cast<float>(epsilon), m, n, stride, xr_stride, y_stride, yr_stride},
                  {stream});
}