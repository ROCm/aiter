// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

#include "fused_ar_mhc_post.h"
#include "custom_all_reduce.cuh"
#include "mhc.h"
#include "aiter_enum.h"
#include "aiter_hip_common.h"
#include "aiter_stream.h"

namespace aiter {

namespace {

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
AiterDtype aiter_dtype_for_ar_mhc_post()
{
    if constexpr(std::is_same_v<T, opus::bf16_t>)
        return AITER_DTYPE_bf16;
    else
        return AITER_DTYPE_fp16;
}

template <typename T>
void run_ar_mhc_post_split(CustomAllreduce* fa,
                           hipStream_t stream,
                           AiterDtype dtype,
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
                              aiter_dtype_for_ar_mhc_post<T>(),
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
                           AiterDtype dtype,
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
                                   aiter_tensor_t& inp,
                                   aiter_tensor_t& next_residual,
                                   aiter_tensor_t& residual_in,
                                   aiter_tensor_t& post_layer_mix,
                                   aiter_tensor_t& comb_res_mix,
                                   bool use_new,
                                   bool open_fp8_quant,
                                   int64_t reg_ptr,
                                   int64_t reg_bytes)
{
    (void)use_new;
    (void)open_fp8_quant;
    HipDeviceGuard device_guard(inp.device_id);
    hipStream_t stream = aiter::getCurrentHIPStream();
    auto fa            = reinterpret_cast<CustomAllreduce*>(_fa);

    const int m       = static_cast<int>(inp.numel() / inp.size(-1));
    const int input_n = static_cast<int>(inp.size(-1));
    const int hidden  = static_cast<int>(residual_in.size(-1));
    const int stride  = static_cast<int>(residual_in.stride(0));

    copy_input_to_registered_buffer(inp, m, input_n, stream, reg_ptr, reg_bytes);

    switch(inp.dtype())
    {
    case AITER_DTYPE_bf16: {
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
    case AITER_DTYPE_fp16: {
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
                                        aiter_tensor_t& inp,
                                        aiter_tensor_t& next_residual,
                                        aiter_tensor_t& residual_in,
                                        aiter_tensor_t& post_layer_mix,
                                        aiter_tensor_t& comb_res_mix,
                                        bool use_new,
                                        bool open_fp8_quant,
                                        int64_t reg_ptr,
                                        int64_t reg_bytes)
{
    (void)use_new;
    (void)open_fp8_quant;
    HipDeviceGuard device_guard(inp.device_id);
    hipStream_t stream = aiter::getCurrentHIPStream();
    auto fa            = reinterpret_cast<CustomAllreduce*>(_fa);

    const int m       = static_cast<int>(inp.numel() / inp.size(-1));
    const int input_n = static_cast<int>(inp.size(-1));
    const int hidden  = static_cast<int>(residual_in.size(-1));
    const int stride  = static_cast<int>(residual_in.stride(0));

    copy_input_to_registered_buffer(inp, m, input_n, stream, reg_ptr, reg_bytes);

    switch(inp.dtype())
    {
    case AITER_DTYPE_bf16: {
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
    case AITER_DTYPE_fp16: {
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
                                    aiter_tensor_t& inp,
                                    aiter_tensor_t& next_residual,
                                    aiter_tensor_t& residual_in,
                                    aiter_tensor_t& post_layer_mix,
                                    aiter_tensor_t& comb_res_mix,
                                    bool use_new,
                                    bool open_fp8_quant,
                                    int64_t reg_ptr,
                                    int64_t reg_bytes)
{
    (void)use_new;
    (void)open_fp8_quant;
    HipDeviceGuard device_guard(inp.device_id);
    hipStream_t stream = aiter::getCurrentHIPStream();
    auto fa            = reinterpret_cast<CustomAllreduce*>(_fa);

    const int m       = static_cast<int>(inp.numel() / inp.size(-1));
    const int input_n = static_cast<int>(inp.size(-1));
    const int hidden  = static_cast<int>(residual_in.size(-1));
    const int stride  = static_cast<int>(residual_in.stride(0));

    copy_input_to_registered_buffer(inp, m, input_n, stream, reg_ptr, reg_bytes);

    switch(inp.dtype())
    {
    case AITER_DTYPE_bf16: {
        run_ar_mhc_post_split<opus::bf16_t>(
            fa,
            stream,
            AITER_DTYPE_bf16,
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
    case AITER_DTYPE_fp16: {
        run_ar_mhc_post_split<opus::fp16_t>(
            fa,
            stream,
            AITER_DTYPE_fp16,
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
