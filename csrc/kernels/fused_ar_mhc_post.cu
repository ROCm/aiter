// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

#include "fused_ar_mhc_post.h"
#include "custom_all_reduce.cuh"
#include "mhc.h"
#include "aiter_enum.h"
#include "aiter_hip_common.h"
#include "py_itfs_common.h"
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <c10/core/ScalarType.h>

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
c10::ScalarType scalar_type_for_ar_mhc_post()
{
    if constexpr(std::is_same_v<T, opus::bf16_t>)
        return c10::ScalarType::BFloat16;
    else
        return c10::ScalarType::Half;
}

template <typename T>
void run_ar_mhc_post_split(CustomAllreduce* fa,
                           hipStream_t stream,
                           c10::ScalarType dtype,
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
                           int64_t reg_bytes);

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
    const int64_t size_bytes = static_cast<int64_t>(m) * input_n * static_cast<int64_t>(sizeof(T));
    constexpr int64_t kSplitMinBytes = 512 * 1024;
    if(fa->world_size_ >= 4 && size_bytes > kSplitMinBytes)
    {
        run_ar_mhc_post_split(fa,
                              stream,
                              scalar_type_for_ar_mhc_post<T>(),
                              inp_ptr,
                              next_residual_ptr,
                              residual_ptr,
                              post_layer_mix_ptr,
                              comb_res_mix_ptr,
                              m,
                              input_n,
                              hidden_size,
                              residual_stride,
                              reg_ptr,
                              reg_bytes);
        return;
    }

    void* actual_inp = inp_ptr;
    if(reg_ptr != 0)
        actual_inp = (void*)reg_ptr;
    fa->dispatchAllReduceMhcPost1Stage<T>(stream,
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
void run_ar_mhc_post_split(CustomAllreduce* fa,
                           hipStream_t stream,
                           c10::ScalarType dtype,
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
    fa->dispatchAllReduceMhcPostSplit<T>(stream,
                                         reinterpret_cast<T*>(actual_inp),
                                         next_residual_ptr,
                                         residual_ptr,
                                         post_layer_mix_ptr,
                                         comb_res_mix_ptr,
                                         m,
                                         input_n,
                                         hidden_size,
                                         residual_stride);
    launch_mhc_post_raw(stream,
                        dtype,
                        next_residual_ptr,
                        fa->local_tmp_reduced_ptr<T>(),
                        residual_ptr,
                        post_layer_mix_ptr,
                        comb_res_mix_ptr,
                        m,
                        hidden_size,
                        hidden_size,
                        residual_stride);
}

template <typename T>
void run_ar_mhc_post_1stage(CustomAllreduce* fa,
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
    fa->dispatchAllReduceMhcPost1Stage<T>(stream,
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

} // namespace

void fused_allreduce_mhc_post_only(fptr_t _fa,
                                   torch::Tensor& inp,
                                   torch::Tensor& next_residual,
                                   torch::Tensor& residual_in,
                                   torch::Tensor& post_layer_mix,
                                   torch::Tensor& comb_res_mix,
                                   bool use_new,
                                   bool open_fp8_quant,
                                   int64_t reg_ptr,
                                   int64_t reg_bytes)
{
    (void)use_new;
    (void)open_fp8_quant;
    const at::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(device_of(inp));
    hipStream_t stream = at::hip::getCurrentHIPStream();
    auto fa            = reinterpret_cast<CustomAllreduce*>(_fa);
    auto inp_at        = make_aiter_tensor(inp);

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
        throw std::runtime_error("fused AR+MHC post only supports fp16/bf16 activations");
    }
}

void fused_allreduce_mhc_post_one_stage(fptr_t _fa,
                                        torch::Tensor& inp,
                                        torch::Tensor& next_residual,
                                        torch::Tensor& residual_in,
                                        torch::Tensor& post_layer_mix,
                                        torch::Tensor& comb_res_mix,
                                        bool use_new,
                                        bool open_fp8_quant,
                                        int64_t reg_ptr,
                                        int64_t reg_bytes)
{
    (void)use_new;
    (void)open_fp8_quant;
    const at::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(device_of(inp));
    hipStream_t stream = at::hip::getCurrentHIPStream();
    auto fa            = reinterpret_cast<CustomAllreduce*>(_fa);
    auto inp_at        = make_aiter_tensor(inp);

    const int m       = static_cast<int>(inp_at.numel() / inp_at.size(-1));
    const int input_n = static_cast<int>(inp_at.size(-1));
    const int hidden  = static_cast<int>(residual_in.size(-1));
    const int stride  = static_cast<int>(residual_in.stride(0));

    copy_input_to_registered_buffer(inp_at, m, input_n, stream, reg_ptr, reg_bytes);

    switch(inp.scalar_type())
    {
    case at::ScalarType::BFloat16: {
        run_ar_mhc_post_1stage<opus::bf16_t>(
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
        run_ar_mhc_post_1stage<opus::fp16_t>(
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
        throw std::runtime_error("fused AR+MHC post one-stage supports fp16/bf16 activations");
    }
}

void fused_allreduce_mhc_post_split(fptr_t _fa,
                                    torch::Tensor& inp,
                                    torch::Tensor& next_residual,
                                    torch::Tensor& residual_in,
                                    torch::Tensor& post_layer_mix,
                                    torch::Tensor& comb_res_mix,
                                    bool use_new,
                                    bool open_fp8_quant,
                                    int64_t reg_ptr,
                                    int64_t reg_bytes)
{
    (void)use_new;
    (void)open_fp8_quant;
    const at::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(device_of(inp));
    hipStream_t stream = at::hip::getCurrentHIPStream();
    auto fa            = reinterpret_cast<CustomAllreduce*>(_fa);
    auto inp_at        = make_aiter_tensor(inp);

    const int m       = static_cast<int>(inp_at.numel() / inp_at.size(-1));
    const int input_n = static_cast<int>(inp_at.size(-1));
    const int hidden  = static_cast<int>(residual_in.size(-1));
    const int stride  = static_cast<int>(residual_in.stride(0));

    copy_input_to_registered_buffer(inp_at, m, input_n, stream, reg_ptr, reg_bytes);

    switch(inp.scalar_type())
    {
    case at::ScalarType::BFloat16: {
        run_ar_mhc_post_split<opus::bf16_t>(
            fa,
            stream,
            at::ScalarType::BFloat16,
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
        run_ar_mhc_post_split<opus::fp16_t>(
            fa,
            stream,
            at::ScalarType::Half,
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
        throw std::runtime_error("fused AR+MHC post split supports fp16/bf16 activations");
    }
}

} // namespace aiter
