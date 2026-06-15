// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

#include "fused_ar_mhc_rmsnorm.h"
#include "custom_all_reduce.cuh"
#include "mhc.h"
#include "aiter_enum.h"
#include "aiter_hip_common.h"
#include "py_itfs_common.h"
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>

namespace aiter {

namespace {

aiter_tensor_t make_aiter_tensor(const torch::Tensor& t)
{
    AITER_CHECK(t.is_cuda(), "fused AR+MHC requires CUDA tensors");
    AITER_CHECK(t.dim() <= 8, "tensor rank too large for aiter_tensor_t");

    aiter_tensor_t at{};
    at.ptr     = t.data_ptr();
    at.numel_  = static_cast<size_t>(t.numel());
    at.ndim    = t.dim();
    for(int i = 0; i < t.dim(); ++i)
    {
        at.shape[i]   = t.size(i);
        at.strides[i] = t.stride(i);
    }
    at.device_id = static_cast<int>(t.device().index());
    switch(t.scalar_type())
    {
    case at::ScalarType::Float: at.dtype_ = AITER_DTYPE_fp32; break;
    case at::ScalarType::Half: at.dtype_ = AITER_DTYPE_fp16; break;
    case at::ScalarType::BFloat16: at.dtype_ = AITER_DTYPE_bf16; break;
    default: throw std::runtime_error("fused AR+MHC only supports fp32/fp16/bf16");
    }
    return at;
}

void copy_input_to_registered_buffer(const aiter_tensor_t& inp,
                                     int m,
                                     int input_n,
                                     hipStream_t stream,
                                     int64_t reg_ptr,
                                     int64_t reg_bytes)
{
    int64_t data_bytes = inp.numel() * inp.element_size();
    if(reg_ptr == 0)
        return;
    if(data_bytes > reg_bytes)
        throw std::runtime_error("registered buffer is too small to contain the input");
    HIP_CALL(hipMemcpyAsync((void*)reg_ptr, inp.data_ptr(), data_bytes,
                            hipMemcpyDeviceToDevice, stream));
}

template <typename T>
void run_ar_input_prototype(CustomAllreduce* fa,
                            hipStream_t stream,
                            T* inp_ptr,
                            T* layer_input_ptr,
                            int m,
                            int input_n,
                            int n,
                            int64_t reg_ptr,
                            int64_t reg_bytes)
{
    void* actual_inp = inp_ptr;
    if(reg_ptr != 0)
        actual_inp = (void*)reg_ptr;
    fa->dispatchAllReduceGatherLayerInputPrototype<T>(
        stream,
        reinterpret_cast<T*>(actual_inp),
        layer_input_ptr,
        m,
        input_n,
        n);
}

template <typename T>
void run_ar_mhc_post_large_m(CustomAllreduce* fa,
                             hipStream_t stream,
                             T* inp_ptr,
                             T* next_residual_ptr,
                             T* residual_ptr,
                             float* post_layer_mix_ptr,
                             float* comb_res_mix_ptr,
                             int m,
                             int input_n,
                             int hidden_size,
                             int residual_stride,
                             int64_t reg_ptr,
                             int64_t reg_bytes)
{
    void* actual_inp = inp_ptr;
    if(reg_ptr != 0)
        actual_inp = (void*)reg_ptr;
    fa->dispatchAllReduceMhcPostLargeM<T>(stream,
                                          reinterpret_cast<T*>(actual_inp),
                                          next_residual_ptr,
                                          residual_ptr,
                                          post_layer_mix_ptr,
                                          comb_res_mix_ptr,
                                          m,
                                          input_n,
                                          hidden_size,
                                          residual_stride);
}

template <typename T>
void dispatch_ar_input(fptr_t _fa,
                       CustomAllreduce* fa,
                       hipStream_t stream,
                       const aiter_tensor_t& inp,
                       const aiter_tensor_t& layer_input,
                       bool use_ar_mhc_post_epilogue,
                       bool use_new,
                       bool open_fp8_quant,
                       int64_t reg_ptr,
                       int64_t reg_bytes)
{
    int input_n  = static_cast<int>(inp.size(-1));
    int n        = static_cast<int>(layer_input.size(-1));
    int m        = static_cast<int>(inp.numel() / input_n);
    T* inp_ptr   = reinterpret_cast<T*>(inp.data_ptr());
    T* layer_ptr = reinterpret_cast<T*>(layer_input.data_ptr());

    if(use_ar_mhc_post_epilogue && input_n == n)
    {
        copy_input_to_registered_buffer(inp, m, input_n, stream, reg_ptr, reg_bytes);
        run_ar_input_prototype<T>(fa, stream, inp_ptr, layer_ptr, m, input_n, n, reg_ptr,
                                  reg_bytes);
        return;
    }

    all_reduce(_fa, inp, layer_input, use_new, open_fp8_quant, reg_ptr, reg_bytes);
}

void run_mhc_pre_large_m(torch::Tensor& post_mix,
                         torch::Tensor& comb_mix,
                         torch::Tensor& layer_input_out,
                         torch::Tensor& gemm_out,
                         torch::Tensor& gemm_out_sqrsum,
                         torch::Tensor& next_residual,
                         torch::Tensor& fn,
                         torch::Tensor& hc_scale,
                         torch::Tensor& hc_base,
                         torch::Tensor& norm_weight,
                         int pre_tile_k,
                         float rms_eps,
                         float hc_pre_eps,
                         float hc_sinkhorn_eps,
                         float norm_eps,
                         float hc_post_mult_value,
                         int sinkhorn_repeat)
{
    mhc_pre_gemm_sqrsum(gemm_out, gemm_out_sqrsum, next_residual, fn, pre_tile_k);
    mhc_pre_big_fuse_rmsnorm(post_mix,
                               comb_mix,
                               layer_input_out,
                               gemm_out,
                               gemm_out_sqrsum,
                               hc_scale,
                               hc_base,
                               next_residual,
                               norm_weight,
                               rms_eps,
                               hc_pre_eps,
                               hc_sinkhorn_eps,
                               norm_eps,
                               hc_post_mult_value,
                               sinkhorn_repeat);
}

} // namespace

void fused_allreduce_mhc_fused_post_pre_rmsnorm(
    fptr_t _fa,
    torch::Tensor& inp,
    torch::Tensor& layer_input,
    torch::Tensor& residual_in,
    torch::Tensor& post_layer_mix,
    torch::Tensor& comb_res_mix,
    torch::Tensor& fn,
    torch::Tensor& hc_scale,
    torch::Tensor& hc_base,
    torch::Tensor& norm_weight,
    torch::Tensor& gemm_out,
    torch::Tensor& gemm_out_sqrsum,
    torch::Tensor& next_residual,
    torch::Tensor& post_mix,
    torch::Tensor& comb_mix,
    torch::Tensor& layer_input_out,
    float rms_eps,
    float hc_pre_eps,
    float hc_sinkhorn_eps,
    float norm_eps,
    float hc_post_mult_value,
    int sinkhorn_repeat,
    int tile_m,
    int tile_n,
    int tile_k,
    int pre_tile_k,
    int post_store_nt,
    bool use_large_m,
    bool use_ar_mhc_full_fusion,
    bool use_ar_mhc_post_epilogue,
    bool use_large_m_post_epilogue,
    bool use_new,
    bool open_fp8_quant,
    int64_t reg_ptr,
    int64_t reg_bytes)
{
    const at::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(device_of(inp));
    hipStream_t stream = at::hip::getCurrentHIPStream();
    auto fa            = reinterpret_cast<CustomAllreduce*>(_fa);
    auto inp_at        = make_aiter_tensor(inp);
    auto layer_at      = make_aiter_tensor(layer_input);

    if(use_ar_mhc_full_fusion)
    {
        copy_input_to_registered_buffer(inp_at,
                                        static_cast<int>(inp_at.numel() / inp_at.size(-1)),
                                        static_cast<int>(inp_at.size(-1)),
                                        stream,
                                        reg_ptr,
                                        reg_bytes);
        launch_fused_ar_mhc_gemm_sqrsum_unified(_fa,
                                                inp,
                                                gemm_out,
                                                gemm_out_sqrsum,
                                                next_residual,
                                                residual_in,
                                                post_layer_mix,
                                                comb_res_mix,
                                                fn,
                                                post_mix,
                                                comb_mix,
                                                layer_input_out,
                                                hc_scale,
                                                hc_base,
                                                norm_weight,
                                                rms_eps,
                                                hc_pre_eps,
                                                hc_sinkhorn_eps,
                                                norm_eps,
                                                hc_post_mult_value,
                                                sinkhorn_repeat,
                                                tile_m,
                                                tile_n,
                                                tile_k,
                                                reg_ptr);
        return;
    }

    if(use_large_m && use_large_m_post_epilogue)
    {
        const int m       = static_cast<int>(inp_at.numel() / inp_at.size(-1));
        const int input_n = static_cast<int>(inp_at.size(-1));
        const int hidden  = static_cast<int>(residual_in.size(-1));
        const int stride  = static_cast<int>(residual_in.stride(0));
        copy_input_to_registered_buffer(inp_at, m, input_n, stream, reg_ptr, reg_bytes);

        switch(inp.scalar_type())
        {
        case at::ScalarType::BFloat16: {
            run_ar_mhc_post_large_m<opus::bf16_t>(
                fa,
                stream,
                reinterpret_cast<opus::bf16_t*>(inp.data_ptr()),
                reinterpret_cast<opus::bf16_t*>(next_residual.data_ptr()),
                reinterpret_cast<opus::bf16_t*>(residual_in.data_ptr()),
                reinterpret_cast<float*>(post_layer_mix.data_ptr()),
                reinterpret_cast<float*>(comb_res_mix.data_ptr()),
                m,
                input_n,
                hidden,
                stride,
                reg_ptr,
                reg_bytes);
            break;
        }
        case at::ScalarType::Half: {
            run_ar_mhc_post_large_m<opus::fp16_t>(
                fa,
                stream,
                reinterpret_cast<opus::fp16_t*>(inp.data_ptr()),
                reinterpret_cast<opus::fp16_t*>(next_residual.data_ptr()),
                reinterpret_cast<opus::fp16_t*>(residual_in.data_ptr()),
                reinterpret_cast<float*>(post_layer_mix.data_ptr()),
                reinterpret_cast<float*>(comb_res_mix.data_ptr()),
                m,
                input_n,
                hidden,
                stride,
                reg_ptr,
                reg_bytes);
            break;
        }
        default:
            throw std::runtime_error("fused AR+MHC only supports fp16/bf16 activations");
        }

        run_mhc_pre_large_m(post_mix,
                            comb_mix,
                            layer_input_out,
                            gemm_out,
                            gemm_out_sqrsum,
                            next_residual,
                            fn,
                            hc_scale,
                            hc_base,
                            norm_weight,
                            pre_tile_k,
                            rms_eps,
                            hc_pre_eps,
                            hc_sinkhorn_eps,
                            norm_eps,
                            hc_post_mult_value,
                            sinkhorn_repeat);
        return;
    }

    switch(inp.scalar_type())
    {
    case at::ScalarType::BFloat16: {
        dispatch_ar_input<opus::bf16_t>(_fa,
                                        fa,
                                        stream,
                                        inp_at,
                                        layer_at,
                                        use_ar_mhc_post_epilogue,
                                        use_new,
                                        open_fp8_quant,
                                        reg_ptr,
                                        reg_bytes);
        break;
    }
    case at::ScalarType::Half: {
        dispatch_ar_input<opus::fp16_t>(_fa,
                                        fa,
                                        stream,
                                        inp_at,
                                        layer_at,
                                        use_ar_mhc_post_epilogue,
                                        use_new,
                                        open_fp8_quant,
                                        reg_ptr,
                                        reg_bytes);
        break;
    }
    default:
        throw std::runtime_error("fused AR+MHC only supports fp16/bf16 activations");
    }

    if(use_large_m)
    {
        mhc_post(next_residual,
                 layer_input,
                 residual_in,
                 post_layer_mix,
                 comb_res_mix,
                 post_store_nt);
        run_mhc_pre_large_m(post_mix,
                            comb_mix,
                            layer_input_out,
                            gemm_out,
                            gemm_out_sqrsum,
                            next_residual,
                            fn,
                            hc_scale,
                            hc_base,
                            norm_weight,
                            pre_tile_k,
                            rms_eps,
                            hc_pre_eps,
                            hc_sinkhorn_eps,
                            norm_eps,
                            hc_post_mult_value,
                            sinkhorn_repeat);
        return;
    }

    if(use_ar_mhc_post_epilogue)
    {
        mhc_post(next_residual,
                 layer_input,
                 residual_in,
                 post_layer_mix,
                 comb_res_mix,
                 post_store_nt);
        run_mhc_pre_large_m(post_mix,
                            comb_mix,
                            layer_input_out,
                            gemm_out,
                            gemm_out_sqrsum,
                            next_residual,
                            fn,
                            hc_scale,
                            hc_base,
                            norm_weight,
                            pre_tile_k,
                            rms_eps,
                            hc_pre_eps,
                            hc_sinkhorn_eps,
                            norm_eps,
                            hc_post_mult_value,
                            sinkhorn_repeat);
        return;
    }

    mhc_fused_post_pre_gemm_sqrsum(gemm_out,
                                   gemm_out_sqrsum,
                                   next_residual,
                                   layer_input,
                                   residual_in,
                                   post_layer_mix,
                                   comb_res_mix,
                                   fn,
                                   tile_m,
                                   tile_n,
                                   tile_k);

    mhc_pre_big_fuse_rmsnorm(post_mix,
                               comb_mix,
                               layer_input_out,
                               gemm_out,
                               gemm_out_sqrsum,
                               hc_scale,
                               hc_base,
                               next_residual,
                               norm_weight,
                               rms_eps,
                               hc_pre_eps,
                               hc_sinkhorn_eps,
                               norm_eps,
                               hc_post_mult_value,
                               sinkhorn_repeat);
}

} // namespace aiter
