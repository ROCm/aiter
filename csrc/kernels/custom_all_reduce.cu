/*
 * Copyright © Advanced Micro Devices, Inc. All rights reserved.
 * Copyright (C) 2024-2025, The vLLM team.
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
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <ATen/hip/impl/HIPStreamMasqueradingAsCUDA.h>
#include <torch/all.h>

#include "custom_all_reduce.cuh"

// fake pointer type, must match fptr_t type in ops.h
using fptr_t = int64_t;
static_assert(sizeof(void*) == sizeof(fptr_t));

namespace aiter {

fptr_t init_custom_ar(torch::Tensor& meta,
                      torch::Tensor& rank_data,
                      const std::vector<torch::Tensor>& handles,
                      const std::vector<int64_t>& offsets,
                      int64_t rank,
                      bool fully_connected)
{
    int world_size = offsets.size();
    if(world_size > 8)
        throw std::invalid_argument("world size > 8 is not supported");
    if(world_size % 2 != 0)
        throw std::invalid_argument("Odd num gpus is not supported for now");
    if(world_size != handles.size())
        throw std::invalid_argument("handles length should equal to offsets length");
    if(rank < 0 || rank >= world_size)
        throw std::invalid_argument("invalid rank passed in");

    hipIpcMemHandle_t ipc_handles[8];
    for(int i = 0; i < world_size; i++)
    {
        hipIpcMemHandle_t* ipc_handle_ptr = (hipIpcMemHandle_t*)handles[i].data_ptr();
        std::memcpy(&ipc_handles[i], ipc_handle_ptr, sizeof(hipIpcMemHandle_t));
    }
    return (fptr_t) new aiter::CustomAllreduce(reinterpret_cast<aiter::Signal*>(meta.data_ptr()),
                                               rank_data.data_ptr(),
                                               rank_data.numel(),
                                               ipc_handles,
                                               offsets,
                                               rank,
                                               fully_connected);
}

/**
 * Make sure tensor t's data lies completely within ((char)t.data_ptr()) +
 * t.numel() * t.element_size(). This is slightly weaker than t.is_contiguous()
 * because it allows transpose of contiguous slice (i.e. slicing the first
 * dimension). Currently, we require this because stride information is not
 * passed into the kernels and we treat input tensors as flat.
 *
 * Examples
 * A = torch.zeros(3, 3, 3)
 * 1. A: OK
 * 2. A[1:]: OK
 * 3. A.permute(2, 0, 1): OK
 * 4. A[1:].permute(2, 0, 1): OK
 * 5. A[None].expand(2, -1, -1, -1): Not OK
 * 6. A[:, 1:, 1:]: Not OK
 */
bool _is_weak_contiguous(torch::Tensor& t)
{
    return t.is_contiguous() || (t.storage().nbytes() - t.storage_offset() * t.element_size() ==
                                 t.numel() * t.element_size());
}

void _all_reduce(
    fptr_t _fa, torch::Tensor& inp, torch::Tensor& out, hipStream_t stream, bool open_fp8_quant)
{
    auto fa = reinterpret_cast<aiter::CustomAllreduce*>(_fa);
    TORCH_CHECK(_is_weak_contiguous(out));
    switch(out.scalar_type())
    {
    case at::ScalarType::Float: {
        fa->allreduce<float>(stream,
                             reinterpret_cast<float*>(inp.data_ptr()),
                             reinterpret_cast<float*>(out.data_ptr()),
                             out.numel());
        break;
    }
    case at::ScalarType::Half: {
        /*
         * By default, hidden_dim is a multiple of 128
         * Obvious effects can only be achieved when the data scale reaches a certain level
         * */
        if(open_fp8_quant && out.numel() >= 128 * 2048)
        {
            fa->runFp8QuantKernel<half>(stream,
                                        reinterpret_cast<half*>(inp.data_ptr()),
                                        reinterpret_cast<half*>(out.data_ptr()),
                                        out.numel());
        }
        else
        {
            fa->allreduce<half>(stream,
                                reinterpret_cast<half*>(inp.data_ptr()),
                                reinterpret_cast<half*>(out.data_ptr()),
                                out.numel());
        }
        break;
    }
#if (__CUDA_ARCH__ >= 800 || !defined(__CUDA_ARCH__))
    case at::ScalarType::BFloat16: {
        fa->allreduce<__hip_bfloat16>(stream,
                                      reinterpret_cast<__hip_bfloat16*>(inp.data_ptr()),
                                      reinterpret_cast<__hip_bfloat16*>(out.data_ptr()),
                                      out.numel());
        break;
    }
#endif
    default:
        throw std::runtime_error("custom allreduce only supports float32, float16 and bfloat16");
    }
}

void all_reduce(fptr_t _fa,
                torch::Tensor& inp,
                torch::Tensor& out,
                bool open_fp8_quant,
                std::optional<torch::Tensor> reg_buffer)
{
    const at::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(device_of(inp));
    auto stream = c10::hip::getCurrentHIPStreamMasqueradingAsCUDA().stream();
    TORCH_CHECK_EQ(inp.scalar_type(), out.scalar_type());
    TORCH_CHECK_EQ(inp.numel(), out.numel());

    if(reg_buffer.has_value())
    {
        auto input_size = inp.numel() * inp.element_size();
        TORCH_CHECK(input_size <= reg_buffer.value().numel() * reg_buffer.value().element_size(),
                    "registered buffer is too small to contain the input");
        HIP_CALL(hipMemcpyAsync(reg_buffer.value().data_ptr(),
                                inp.data_ptr(),
                                input_size,
                                hipMemcpyDeviceToDevice,
                                stream));
        _all_reduce(_fa, reg_buffer.value(), out, stream, open_fp8_quant);
    }
    else
    {
        _all_reduce(_fa, inp, out, stream, open_fp8_quant);
    }
    

}

void _all_gather(fptr_t _fa, torch::Tensor& inp, torch::Tensor& out, int size, hipStream_t stream)
{
    auto fa = reinterpret_cast<aiter::CustomAllreduce*>(_fa);
    TORCH_CHECK(_is_weak_contiguous(out));
    switch(out.scalar_type())
    {
    case at::ScalarType::Float: {
        fa->dispatchAllGather<float>(stream,
                                     reinterpret_cast<float*>(inp.data_ptr()),
                                     reinterpret_cast<float*>(out.data_ptr()),
                                     size);
        break;
    }
    case at::ScalarType::Half: {
        fa->dispatchAllGather<half>(stream,
                                    reinterpret_cast<half*>(inp.data_ptr()),
                                    reinterpret_cast<half*>(out.data_ptr()),
                                    size);
        break;
    }
#if (__CUDA_ARCH__ >= 800 || !defined(__CUDA_ARCH__))
    case at::ScalarType::BFloat16: {
        fa->dispatchAllGather<__hip_bfloat16>(stream,
                                              reinterpret_cast<__hip_bfloat16*>(inp.data_ptr()),
                                              reinterpret_cast<__hip_bfloat16*>(out.data_ptr()),
                                              size);
        break;
    }
#endif
    default:
        throw std::runtime_error("custom allreduce only supports float32, float16 and bfloat16");
    }
}

void all_gather_reg(fptr_t _fa, torch::Tensor& inp, torch::Tensor& out)
{
    const at::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(device_of(inp));
    auto stream = c10::hip::getCurrentHIPStreamMasqueradingAsCUDA().stream();
    TORCH_CHECK_EQ(inp.scalar_type(), out.scalar_type());
    _all_gather(_fa, inp, out, inp.numel(), stream);
}

void all_gather_unreg(fptr_t _fa, torch::Tensor& inp, torch::Tensor& reg_buffer, torch::Tensor& out)
{
    const at::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(device_of(inp));
    auto stream = c10::hip::getCurrentHIPStreamMasqueradingAsCUDA().stream();

    auto input_size = inp.numel() * inp.element_size();
    TORCH_CHECK_EQ(inp.scalar_type(), out.scalar_type());
    TORCH_CHECK(input_size <= reg_buffer.numel() * reg_buffer.element_size(),
                "registered buffer is too small to contain the input");
    HIP_CALL(hipMemcpyAsync(
        reg_buffer.data_ptr(), inp.data_ptr(), input_size, hipMemcpyDeviceToDevice, stream));
    _all_gather(_fa, reg_buffer, out, inp.numel(), stream);
}

void _fused_allreduce_rmsnorm(
    fptr_t _fa, torch::Tensor& inp, torch::Tensor& out, torch::Tensor& w, int eps, int m, int n, hipStream_t stream)
{
    auto fa = reinterpret_cast<aiter::CustomAllreduce*>(_fa);
    TORCH_CHECK(_is_weak_contiguous(out));
    switch(out.scalar_type())
    {
    case at::ScalarType::Float: {
        fa->dispatchFusedAllReduceRMSNorm<float>(stream,
                             reinterpret_cast<float*>(inp.data_ptr()),
                             reinterpret_cast<float*>(out.data_ptr()),
                             reinterpret_cast<float*>(w.data_ptr()),
                             eps, m, n);
        break;
    }
    case at::ScalarType::Half: {
        fa->dispatchFusedAllReduceRMSNorm<half>(stream,
                             reinterpret_cast<half*>(inp.data_ptr()),
                             reinterpret_cast<half*>(out.data_ptr()),
                             reinterpret_cast<half*>(w.data_ptr()),
                             eps, m, n);
        break;
    }
#if (__CUDA_ARCH__ >= 800 || !defined(__CUDA_ARCH__))
    case at::ScalarType::BFloat16: {
        fa->dispatchFusedAllReduceRMSNorm<__hip_bfloat16>(stream,
                             reinterpret_cast<__hip_bfloat16*>(inp.data_ptr()),
                             reinterpret_cast<__hip_bfloat16*>(out.data_ptr()),
                             reinterpret_cast<__hip_bfloat16*>(w.data_ptr()),
                             eps, m, n);
        break;
    }
#endif
    default:
        throw std::runtime_error("custom allreduce only supports float32, float16 and bfloat16");
    }
}

void fused_allreduce_rmsnorm(fptr_t _fa,
                torch::Tensor& inp,
                torch::Tensor& out,
                torch::Tensor& w,
                float eps,
                std::optional<torch::Tensor> reg_buffer)
{
    const at::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(device_of(inp));
    auto stream = c10::hip::getCurrentHIPStreamMasqueradingAsCUDA().stream();
    TORCH_CHECK_EQ(inp.scalar_type(), out.scalar_type());
    TORCH_CHECK_EQ(inp.numel(), out.numel());
    int n = w.numel();
    int m = inp.numel() / n;

    if(reg_buffer.has_value())
    {
        auto input_size = inp.numel() * inp.element_size();
        TORCH_CHECK(input_size <= reg_buffer.value().numel() * reg_buffer.value().element_size(),
                    "registered buffer is too small to contain the input");
        HIP_CALL(hipMemcpyAsync(reg_buffer.value().data_ptr(),
                                inp.data_ptr(),
                                input_size,
                                hipMemcpyDeviceToDevice,
                                stream));
        _fused_allreduce_rmsnorm(_fa, reg_buffer.value(), out, w, eps, m, n, stream);
    }
    else
    {
      _fused_allreduce_rmsnorm(_fa, inp, out, w, eps, m, n, stream);
    }
}

void dispose(fptr_t _fa)
{
    auto fa = reinterpret_cast<aiter::CustomAllreduce*>(_fa);
    delete fa;
}

int64_t meta_size() { return sizeof(aiter::Signal); }

void register_buffer(fptr_t _fa,
                     torch::Tensor& t,
                     const std::vector<torch::Tensor>& handles,
                     const std::vector<int64_t>& offsets)
{
    auto fa = reinterpret_cast<aiter::CustomAllreduce*>(_fa);
    fa->register_buffer(handles, offsets, t.data_ptr());
}

std::vector<at::Tensor> get_graph_buffer_ipc_meta(fptr_t _fa)
{
    auto fa                      = reinterpret_cast<aiter::CustomAllreduce*>(_fa);
    auto [handle_bytes, offsets] = fa->get_graph_buffer_ipc_meta();
    auto options                 = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU);
    auto handles = torch::empty({static_cast<int64_t>(handle_bytes.size())}, options);
    std::memcpy(handles.data_ptr(), handle_bytes.data(), handle_bytes.size());

    torch::Tensor offset_tensor =
        torch::from_blob(offsets.data(), {static_cast<int64_t>(offsets.size())}, torch::kInt64)
            .clone();
    return {handles, offset_tensor};
}

void register_graph_buffers(fptr_t _fa,
                            const std::vector<torch::Tensor>& handles,
                            const std::vector<torch::Tensor>& offsets)
{
    auto fa = reinterpret_cast<aiter::CustomAllreduce*>(_fa);
    fa->register_graph_buffers(handles, offsets);
}

#ifdef USE_ROCM

void free_meta_buffer(void* buffer) { HIP_CALL(hipFree(buffer)); }

torch::Tensor get_meta_buffer_ipc_handle(torch::Tensor& inp)
{
    auto options     = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU);
    auto data_handle = torch::empty({static_cast<int64_t>(sizeof(hipIpcMemHandle_t))}, options);
    HIP_CALL(hipIpcGetMemHandle((hipIpcMemHandle_t*)data_handle.data_ptr(), inp.data_ptr()));
    return data_handle;
}

torch::Tensor allocate_meta_buffer(int64_t size)
{
    auto device_index = c10::hip::current_device();
    at::DeviceGuard device_guard(at::Device(at::DeviceType::CUDA, device_index));
    void* buffer;
    hipStreamCaptureMode mode = hipStreamCaptureModeRelaxed;
    auto stream               = c10::hip::getCurrentHIPStreamMasqueradingAsCUDA().stream();
    HIP_CALL(hipThreadExchangeStreamCaptureMode(&mode));
    HIP_CALL(hipExtMallocWithFlags((void**)&buffer, size, hipDeviceMallocUncached));
    HIP_CALL(hipMemsetAsync(buffer, 0, size, stream));
    HIP_CALL(hipStreamSynchronize(stream));
    HIP_CALL(hipThreadExchangeStreamCaptureMode(&mode));
    auto options = torch::TensorOptions().dtype(torch::kI8).device(torch::kCUDA, device_index);
    return torch::from_blob(buffer, {size}, free_meta_buffer, options);
}

#endif

} // namespace aiter
