/*
 * Copyright © Advanced Micro Devices, Inc. All rights reserved.
 * Copyright (C) 2024-2026, The vLLM team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "custom_all_reduce.cuh"
#include <cstring>

using fp8_type = ck_tile::fp8_t;

// fake pointer type, must match fptr_t type in custom_all_reduce.h
using fptr_t = int64_t;
static_assert(sizeof(void*) == sizeof(fptr_t));

namespace aiter {

// Dtype codes – keep in sync with Python-side _DTYPE_MAP
static constexpr int64_t DTYPE_FLOAT32  = 0;
static constexpr int64_t DTYPE_FLOAT16  = 1;
static constexpr int64_t DTYPE_BFLOAT16 = 2;

inline int64_t dtype_element_size(int64_t dtype)
{
    switch(dtype)
    {
    case DTYPE_FLOAT32:  return 4;
    case DTYPE_FLOAT16:  return 2;
    case DTYPE_BFLOAT16: return 2;
    default: throw std::runtime_error("unsupported dtype: " + std::to_string(dtype));
    }
}

// ---- init / dispose / meta_size ----

fptr_t init_custom_ar(int64_t meta_ptr,
                      int64_t rank_data_ptr,
                      int64_t rank_data_sz,
                      const std::vector<int64_t>& ipc_handle_ptrs,
                      const std::vector<int64_t>& offsets,
                      int64_t rank,
                      bool fully_connected)
{
    int world_size = offsets.size();
    if(world_size > 8)
        throw std::invalid_argument("world size > 8 is not supported");
    if(world_size % 2 != 0)
        throw std::invalid_argument("Odd num gpus is not supported for now");
    if(world_size != (int)ipc_handle_ptrs.size())
        throw std::invalid_argument("handles length should equal to offsets length");
    if(rank < 0 || rank >= world_size)
        throw std::invalid_argument("invalid rank passed in");

    hipIpcMemHandle_t ipc_handles[8];
    for(int i = 0; i < world_size; i++)
    {
        std::memcpy(&ipc_handles[i], (void*)ipc_handle_ptrs[i], sizeof(hipIpcMemHandle_t));
    }
    return (fptr_t) new aiter::CustomAllreduce(reinterpret_cast<aiter::Signal*>(meta_ptr),
                                               (void*)rank_data_ptr,
                                               rank_data_sz,
                                               ipc_handles,
                                               offsets,
                                               rank,
                                               fully_connected);
}

void dispose(fptr_t _fa)
{
    auto fa = reinterpret_cast<aiter::CustomAllreduce*>(_fa);
    delete fa;
}

int64_t meta_size() { return sizeof(aiter::Signal); }

// ---- Internal dispatch helpers ----

static void _all_reduce(fptr_t _fa, void* inp, void* out,
                        int64_t numel, int64_t dtype, hipStream_t stream,
                        bool use_new, bool open_fp8_quant, bool is_broadcast_reg_outptr)
{
    auto fa = reinterpret_cast<aiter::CustomAllreduce*>(_fa);
    switch(dtype)
    {
    case DTYPE_FLOAT32: {
        fa->allreduce<opus::fp32_t>(stream,
                             reinterpret_cast<opus::fp32_t*>(inp),
                             reinterpret_cast<opus::fp32_t*>(out),
                             numel, use_new, is_broadcast_reg_outptr);
        break;
    }
    case DTYPE_FLOAT16: {
        if(open_fp8_quant && numel >= 128 * 2048)
        {
            fa->runFp8QuantKernel<opus::fp16_t>(stream,
                                        reinterpret_cast<opus::fp16_t*>(inp),
                                        reinterpret_cast<opus::fp16_t*>(out),
                                        numel);
        }
        else
        {
            fa->allreduce<opus::fp16_t>(stream,
                                reinterpret_cast<opus::fp16_t*>(inp),
                                reinterpret_cast<opus::fp16_t*>(out),
                                numel, use_new, is_broadcast_reg_outptr);
        }
        break;
    }
#if (__CUDA_ARCH__ >= 800 || !defined(__CUDA_ARCH__))
    case DTYPE_BFLOAT16: {
        fa->allreduce<opus::bf16_t>(stream,
                                      reinterpret_cast<opus::bf16_t*>(inp),
                                      reinterpret_cast<opus::bf16_t*>(out),
                                      numel, use_new);
        break;
    }
#endif
    default:
        throw std::runtime_error("custom allreduce only supports float32, float16 and bfloat16");
    }
}

static void _reduce_scatter(fptr_t _fa, void* inp, void* out,
                            int64_t size, int64_t dtype, hipStream_t stream)
{
    auto fa = reinterpret_cast<aiter::CustomAllreduce*>(_fa);
    switch(dtype)
    {
    case DTYPE_FLOAT32: {
        fa->dispatchReduceScatter<opus::fp32_t>(stream,
                                     reinterpret_cast<opus::fp32_t*>(inp),
                                     reinterpret_cast<opus::fp32_t*>(out),
                                     size);
        break;
    }
    case DTYPE_FLOAT16: {
        fa->dispatchReduceScatter<opus::fp16_t>(stream,
                                    reinterpret_cast<opus::fp16_t*>(inp),
                                    reinterpret_cast<opus::fp16_t*>(out),
                                    size);
        break;
    }
#if (__CUDA_ARCH__ >= 800 || !defined(__CUDA_ARCH__))
    case DTYPE_BFLOAT16: {
        fa->dispatchReduceScatter<opus::bf16_t>(stream,
                                              reinterpret_cast<opus::bf16_t*>(inp),
                                              reinterpret_cast<opus::bf16_t*>(out),
                                              size);
        break;
    }
#endif
    default:
        throw std::runtime_error("custom allreduce only supports float32, float16 and bfloat16");
    }
}

static void _all_gather(fptr_t _fa, void* inp, void* out,
                        int64_t size, int64_t dtype,
                        int64_t last_dim_size, int64_t gather_dim,
                        hipStream_t stream)
{
    auto fa = reinterpret_cast<aiter::CustomAllreduce*>(_fa);
    switch(dtype)
    {
    case DTYPE_FLOAT32: {
        fa->dispatchAllGather<opus::fp32_t>(stream,
                                     reinterpret_cast<opus::fp32_t*>(inp),
                                     reinterpret_cast<opus::fp32_t*>(out),
                                     size, last_dim_size, gather_dim);
        break;
    }
    case DTYPE_FLOAT16: {
        fa->dispatchAllGather<opus::fp16_t>(stream,
                                    reinterpret_cast<opus::fp16_t*>(inp),
                                    reinterpret_cast<opus::fp16_t*>(out),
                                    size, last_dim_size, gather_dim);
        break;
    }
#if (__CUDA_ARCH__ >= 800 || !defined(__CUDA_ARCH__))
    case DTYPE_BFLOAT16: {
        fa->dispatchAllGather<opus::bf16_t>(stream,
                                    reinterpret_cast<opus::bf16_t*>(inp),
                                    reinterpret_cast<opus::bf16_t*>(out),
                                    size, last_dim_size, gather_dim);
        break;
    }
#endif
    default:
        throw std::runtime_error("custom allreduce only supports float32, float16 and bfloat16");
    }
}

static void _fused_allreduce_rmsnorm(fptr_t _fa,
                                     void* inp, void* residual_inp,
                                     void* residual_out, void* out,
                                     void* scale_out, void* w,
                                     int64_t dtype, float eps,
                                     int m, int n,
                                     bool use_1stage,
                                     hipStream_t stream)
{
    auto fa = reinterpret_cast<aiter::CustomAllreduce*>(_fa);
    bool use_fp8_per_token_quant = (scale_out != nullptr);

#define DISPATCH_AR_FUSION(DTYPE)                                \
    if(!use_fp8_per_token_quant)                                 \
    {                                                            \
        fa->dispatchFusedAllReduceRMSNorm<DTYPE>(                \
            stream,                                              \
            reinterpret_cast<DTYPE*>(inp),                       \
            reinterpret_cast<DTYPE*>(residual_inp),              \
            reinterpret_cast<DTYPE*>(residual_out),              \
            reinterpret_cast<DTYPE*>(out),                       \
            reinterpret_cast<DTYPE*>(w),                         \
            eps,                                                 \
            m,                                                   \
            n,                                                   \
            use_1stage);                                         \
    }                                                            \
    else                                                         \
    {                                                            \
        fa->dispatchFusedAllReduceRMSNormQuant<DTYPE, fp8_type>( \
            stream,                                              \
            reinterpret_cast<DTYPE*>(inp),                       \
            reinterpret_cast<DTYPE*>(residual_inp),              \
            reinterpret_cast<DTYPE*>(residual_out),              \
            reinterpret_cast<fp8_type*>(out),                    \
            reinterpret_cast<float*>(scale_out),                 \
            reinterpret_cast<DTYPE*>(w),                         \
            eps,                                                 \
            m,                                                   \
            n,                                                   \
            use_1stage);                                         \
    }

    switch(dtype)
    {
    case DTYPE_FLOAT32: {
        DISPATCH_AR_FUSION(opus::fp32_t)
        break;
    }
    case DTYPE_FLOAT16: {
        DISPATCH_AR_FUSION(opus::fp16_t)
        break;
    }
#if(__CUDA_ARCH__ >= 800 || !defined(__CUDA_ARCH__))
    case DTYPE_BFLOAT16: {
        DISPATCH_AR_FUSION(opus::bf16_t)
        break;
    }
#endif
    default:
        throw std::runtime_error("custom allreduce only supports float32, float16 and bfloat16");
    }

#undef DISPATCH_AR_FUSION
}

// ---- Public collective APIs ----

void all_reduce(fptr_t _fa, int64_t inp, int64_t out,
                int64_t numel, int64_t dtype,
                bool use_new, bool open_fp8_quant,
                int64_t reg_inp_ptr, int64_t reg_inp_bytes,
                int64_t reg_out_ptr, int64_t reg_out_bytes,
                int64_t stream_ptr)
{
    auto stream    = (hipStream_t)stream_ptr;
    int64_t elem_sz    = dtype_element_size(dtype);
    int64_t data_bytes = numel * elem_sz;

    void* actual_inp = (void*)inp;
    void* actual_out = (void*)out;

    bool is_broadcast_reg_outptr = (reg_out_ptr == 0);

    if(reg_inp_ptr == 0 && reg_out_ptr == 0)
    {
        _all_reduce(_fa, actual_inp, actual_out, numel, dtype, stream,
                    use_new, open_fp8_quant, is_broadcast_reg_outptr);
        return;
    }

    if(reg_inp_ptr != 0)
    {
        if(data_bytes > reg_inp_bytes)
            throw std::runtime_error("registered buffer is too small to contain the input");
        HIP_CALL(hipMemcpyAsync((void*)reg_inp_ptr, (void*)inp, data_bytes,
                                hipMemcpyDeviceToDevice, stream));
        actual_inp = (void*)reg_inp_ptr;
    }

    if(reg_out_ptr != 0 && is_broadcast_reg_outptr)
    {
        if(data_bytes > reg_out_bytes)
            throw std::runtime_error("registered output buffer is too small to contain the output");
        actual_out = (void*)reg_out_ptr;
    }

    _all_reduce(_fa, actual_inp, actual_out, numel, dtype, stream,
                use_new, open_fp8_quant, is_broadcast_reg_outptr);

    if(reg_out_ptr != 0 && is_broadcast_reg_outptr)
    {
        HIP_CALL(hipMemcpyAsync((void*)out, (void*)reg_out_ptr, data_bytes,
                                hipMemcpyDeviceToDevice, stream));
    }
}

void reduce_scatter(fptr_t _fa, int64_t inp, int64_t out,
                    int64_t inp_numel, int64_t dtype,
                    int64_t reg_ptr, int64_t reg_bytes,
                    int64_t stream_ptr)
{
    auto stream    = (hipStream_t)stream_ptr;
    int64_t elem_sz    = dtype_element_size(dtype);
    int64_t data_bytes = inp_numel * elem_sz;

    if(reg_ptr != 0)
    {
        if(data_bytes > reg_bytes)
            throw std::runtime_error("registered buffer is too small to contain the input");
        HIP_CALL(hipMemcpyAsync((void*)reg_ptr, (void*)inp, data_bytes,
                                hipMemcpyDeviceToDevice, stream));
        _reduce_scatter(_fa, (void*)reg_ptr, (void*)out, inp_numel, dtype, stream);
    }
    else
    {
        _reduce_scatter(_fa, (void*)inp, (void*)out, inp_numel, dtype, stream);
    }
}

void all_gather_reg(fptr_t _fa, int64_t inp, int64_t out,
                    int64_t inp_numel, int64_t dtype,
                    int64_t last_dim_size, int64_t dim,
                    int64_t stream_ptr)
{
    auto stream = (hipStream_t)stream_ptr;
    _all_gather(_fa, (void*)inp, (void*)out, inp_numel, dtype,
                last_dim_size, dim, stream);
}

void all_gather_unreg(fptr_t _fa, int64_t inp, int64_t reg_buffer,
                      int64_t out, int64_t inp_numel, int64_t dtype,
                      int64_t reg_bytes,
                      int64_t last_dim_size, int64_t dim,
                      int64_t stream_ptr)
{
    auto stream    = (hipStream_t)stream_ptr;
    int64_t elem_sz    = dtype_element_size(dtype);
    int64_t data_bytes = inp_numel * elem_sz;

    if(data_bytes > reg_bytes)
        throw std::runtime_error("registered buffer is too small to contain the input");
    HIP_CALL(hipMemcpyAsync((void*)reg_buffer, (void*)inp, data_bytes,
                            hipMemcpyDeviceToDevice, stream));
    _all_gather(_fa, (void*)reg_buffer, (void*)out, inp_numel, dtype,
                last_dim_size, dim, stream);
}

void fused_allreduce_rmsnorm(fptr_t _fa,
                             int64_t inp, int64_t res_inp, int64_t res_out,
                             int64_t out, int64_t w,
                             int64_t numel, int64_t w_numel, int64_t dtype,
                             double eps,
                             int64_t reg_ptr, int64_t reg_bytes,
                             bool use_1stage,
                             int64_t stream_ptr)
{
    auto stream    = (hipStream_t)stream_ptr;
    int64_t elem_sz    = dtype_element_size(dtype);
    int64_t data_bytes = numel * elem_sz;
    int n = (int)w_numel;
    int m = (int)(numel / w_numel);

    if(reg_ptr != 0)
    {
        if(data_bytes > reg_bytes)
            throw std::runtime_error("registered buffer is too small to contain the input");
        HIP_CALL(hipMemcpyAsync((void*)reg_ptr, (void*)inp, data_bytes,
                                hipMemcpyDeviceToDevice, stream));
        _fused_allreduce_rmsnorm(_fa,
                                 (void*)reg_ptr, (void*)res_inp, (void*)res_out,
                                 (void*)out, nullptr, (void*)w,
                                 dtype, (float)eps, m, n, use_1stage, stream);
    }
    else
    {
        _fused_allreduce_rmsnorm(_fa,
                                 (void*)inp, (void*)res_inp, (void*)res_out,
                                 (void*)out, nullptr, (void*)w,
                                 dtype, (float)eps, m, n, use_1stage, stream);
    }
}

void fused_allreduce_rmsnorm_quant(fptr_t _fa,
                                   int64_t inp, int64_t res_inp, int64_t res_out,
                                   int64_t out, int64_t scale_out, int64_t w,
                                   int64_t numel, int64_t w_numel, int64_t dtype,
                                   double eps,
                                   int64_t reg_ptr, int64_t reg_bytes,
                                   bool use_1stage,
                                   int64_t stream_ptr)
{
    auto stream    = (hipStream_t)stream_ptr;
    int64_t elem_sz    = dtype_element_size(dtype);
    int64_t data_bytes = numel * elem_sz;
    int n = (int)w_numel;
    int m = (int)(numel / w_numel);

    if(reg_ptr != 0)
    {
        if(data_bytes > reg_bytes)
            throw std::runtime_error("registered buffer is too small to contain the input");
        HIP_CALL(hipMemcpyAsync((void*)reg_ptr, (void*)inp, data_bytes,
                                hipMemcpyDeviceToDevice, stream));
        _fused_allreduce_rmsnorm(_fa,
                                 (void*)reg_ptr, (void*)res_inp, (void*)res_out,
                                 (void*)out, (void*)scale_out, (void*)w,
                                 dtype, (float)eps, m, n, use_1stage, stream);
    }
    else
    {
        _fused_allreduce_rmsnorm(_fa,
                                 (void*)inp, (void*)res_inp, (void*)res_out,
                                 (void*)out, (void*)scale_out, (void*)w,
                                 dtype, (float)eps, m, n, use_1stage, stream);
    }
}

// ---- Buffer registration ----

void register_input_buffer(fptr_t _fa,
                           int64_t self_ptr,
                           const std::vector<int64_t>& ipc_handle_ptrs,
                           const std::vector<int64_t>& offsets)
{
    auto fa = reinterpret_cast<aiter::CustomAllreduce*>(_fa);
    int world_size = ipc_handle_ptrs.size();
    std::vector<hipIpcMemHandle_t> ipc_handles(world_size);
    for(int i = 0; i < world_size; i++)
    {
        std::memcpy(&ipc_handles[i], (void*)ipc_handle_ptrs[i], sizeof(hipIpcMemHandle_t));
    }
    fa->register_input_buffer(ipc_handles.data(), offsets.data(), (void*)self_ptr);
}

void register_output_buffer(fptr_t _fa,
                            int64_t self_ptr,
                            const std::vector<int64_t>& ipc_handle_ptrs,
                            const std::vector<int64_t>& offsets)
{
    auto fa = reinterpret_cast<aiter::CustomAllreduce*>(_fa);
    int world_size = ipc_handle_ptrs.size();
    std::vector<hipIpcMemHandle_t> ipc_handles(world_size);
    for(int i = 0; i < world_size; i++)
    {
        std::memcpy(&ipc_handles[i], (void*)ipc_handle_ptrs[i], sizeof(hipIpcMemHandle_t));
    }
    fa->register_output_buffer(ipc_handles.data(), offsets.data(), (void*)self_ptr);
}

int64_t get_graph_buffer_count(fptr_t _fa)
{
    auto fa = reinterpret_cast<aiter::CustomAllreduce*>(_fa);
    return (int64_t)(fa->graph_unreg_input_buffers_.size() +
                     fa->graph_unreg_output_buffers_.size());
}

void get_graph_buffer_ipc_meta(fptr_t _fa,
                               int64_t handle_out,
                               int64_t offset_out)
{
    auto fa = reinterpret_cast<aiter::CustomAllreduce*>(_fa);
    auto [handle_bytes, offsets] = fa->get_graph_buffer_ipc_meta();
    std::memcpy((void*)handle_out, handle_bytes.data(), handle_bytes.size());
    std::memcpy((void*)offset_out, offsets.data(), offsets.size() * sizeof(int64_t));
}

void register_graph_buffers(fptr_t _fa,
                            const std::vector<int64_t>& handle_ptrs,
                            const std::vector<int64_t>& offset_ptrs)
{
    auto fa = reinterpret_cast<aiter::CustomAllreduce*>(_fa);
    int world_size = handle_ptrs.size();
    std::vector<const void*> handles(world_size);
    std::vector<const int64_t*> offsets(world_size);
    for(int i = 0; i < world_size; i++)
    {
        handles[i] = (const void*)handle_ptrs[i];
        offsets[i] = (const int64_t*)offset_ptrs[i];
    }
    fa->register_graph_buffers(handles.data(), offsets.data());
}

// ---- ROCm-specific utilities ----

#ifdef USE_ROCM

int64_t allocate_meta_buffer(int64_t size, int64_t stream_ptr)
{
    auto stream = (hipStream_t)stream_ptr;
    void* buffer;
    hipStreamCaptureMode mode = hipStreamCaptureModeRelaxed;
    HIP_CALL(hipThreadExchangeStreamCaptureMode(&mode));
    HIP_CALL(hipExtMallocWithFlags((void**)&buffer, size, hipDeviceMallocUncached));
    HIP_CALL(hipMemsetAsync(buffer, 0, size, stream));
    HIP_CALL(hipStreamSynchronize(stream));
    HIP_CALL(hipThreadExchangeStreamCaptureMode(&mode));
    return (int64_t)buffer;
}

void free_meta_buffer(int64_t ptr)
{
    HIP_CALL(hipFree((void*)ptr));
}

void get_meta_buffer_ipc_handle(int64_t inp_ptr, int64_t out_handle_ptr)
{
    HIP_CALL(hipIpcGetMemHandle((hipIpcMemHandle_t*)out_handle_ptr, (void*)inp_ptr));
}

#endif

} // namespace aiter
