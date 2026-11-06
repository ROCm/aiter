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
// Instances split out of custom_all_reduce.cu so that they compile in parallel
// with it: the residual all-reduce (HAS_RESIDUAL kernels), and the row-option
// local RMSNorm and single-kernel 2-stage fused AR+RMSNorm that
// custom_all_reduce.cuh declares extern (AITER_CUSTOM_AR_EXT_INSTANCES).
#include "custom_all_reduce.cuh"
#include "aiter_stream.h"
#include "aiter_tensor.h"

// fake pointer type, must match fptr_t type in custom_all_reduce.h
using fptr_t = int64_t;

namespace aiter {

void all_reduce_residual(fptr_t _fa,
                         const aiter_tensor_t& inp,
                         const aiter_tensor_t& out,
                         const aiter_tensor_t& residual,
                         int64_t reg_inp_ptr,
                         int64_t reg_inp_bytes)
{
    HipDeviceGuard device_guard(inp.device_id);
    hipStream_t stream = aiter::getCurrentHIPStream();
    auto dtype         = inp.dtype();
    int64_t numel      = inp.numel();
    int64_t data_bytes = numel * inp.element_size();

    // The copy and the kernels read inp densely; they index out and residual
    // densely by inp's extent, on inp's device.
    if(!inp.is_contiguous())
        throw std::runtime_error("all_reduce_residual: inp must be contiguous");
    for(auto [t, name] : {std::pair{&residual, "residual"}, std::pair{&out, "out"}})
    {
        if(t->numel() != numel || t->dtype() != dtype || t->device_id != inp.device_id)
            throw std::runtime_error(std::string("all_reduce_residual: ") + name +
                                     " must match the input's numel, dtype and device");
        if(!t->is_contiguous())
            throw std::runtime_error(std::string("all_reduce_residual: ") + name +
                                     " must be contiguous");
    }

    void* actual_inp = inp.data_ptr();
    // As in all_reduce: a registered input (graph mode) writes to peer outputs.
    bool is_broadcast_reg_outptr = (reg_inp_ptr == 0);
    if(reg_inp_ptr != 0)
    {
        if(data_bytes > reg_inp_bytes)
            throw std::runtime_error("registered buffer is too small to contain the input");
        HIP_CALL(hipMemcpyAsync(
            (void*)reg_inp_ptr, actual_inp, data_bytes, hipMemcpyDeviceToDevice, stream));
        actual_inp = (void*)reg_inp_ptr;
    }

    auto fa = reinterpret_cast<aiter::CustomAllreduce*>(_fa);
#define DISPATCH_AR_RESIDUAL(DTYPE)                                                    \
    fa->allreduce_residual<DTYPE>(stream,                                              \
                                  reinterpret_cast<DTYPE*>(actual_inp),                \
                                  reinterpret_cast<DTYPE*>(out.data_ptr()),            \
                                  reinterpret_cast<const DTYPE*>(residual.data_ptr()), \
                                  numel,                                               \
                                  is_broadcast_reg_outptr)
    switch(dtype)
    {
    case AITER_DTYPE_fp32: DISPATCH_AR_RESIDUAL(opus::fp32_t); break;
    case AITER_DTYPE_fp16: DISPATCH_AR_RESIDUAL(opus::fp16_t); break;
#if (__CUDA_ARCH__ >= 800 || !defined(__CUDA_ARCH__))
    case AITER_DTYPE_bf16: DISPATCH_AR_RESIDUAL(opus::bf16_t); break;
#endif
    default:
        throw std::runtime_error(
            "custom allreduce residual only supports float32, float16 and bfloat16");
    }
#undef DISPATCH_AR_RESIDUAL
}

AITER_CUSTOM_AR_EXT_INSTANCES(, opus::fp32_t)
AITER_CUSTOM_AR_EXT_INSTANCES(, opus::fp16_t)
AITER_CUSTOM_AR_EXT_INSTANCES(, opus::bf16_t)

} // namespace aiter
