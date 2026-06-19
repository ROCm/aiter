/*
 * Copyright (C) Advanced Micro Devices, Inc. All rights reserved.
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
 *
 * Host-side dispatch for gfx1250 (MI450) custom allreduce.
 * Pointer-based API — hipIpc is not available on gfx1250.
 */
#include "custom_all_reduce_gfx1250.cuh"
#include "aiter_stream.h"
#include "aiter_tensor.h"
#include <cstring>

using fptr_t = int64_t;
static_assert(sizeof(void*) == sizeof(fptr_t));

namespace aiter {

// ---- init / dispose / meta_size ----

fptr_t init_custom_ar(int64_t meta_ptr,
                      int64_t rank_data_ptr,
                      int64_t rank_data_sz,
                      const std::vector<int64_t>& all_meta_ptrs,
                      int64_t rank,
                      bool fully_connected)
{
    int world_size = all_meta_ptrs.size();
    if(world_size > 4)
        throw std::invalid_argument("gfx1250 custom allreduce: world size > 4 is not supported");
    if(world_size % 2 != 0)
        throw std::invalid_argument("Odd num gpus is not supported for now");
    if(rank < 0 || rank >= world_size)
        throw std::invalid_argument("invalid rank passed in");

    return (fptr_t) new aiter::CustomAllreduce(reinterpret_cast<aiter::Signal*>(meta_ptr),
                                               (void*)rank_data_ptr,
                                               rank_data_sz,
                                               all_meta_ptrs,
                                               rank,
                                               fully_connected);
}

void dispose(fptr_t _fa)
{
    auto fa = reinterpret_cast<aiter::CustomAllreduce*>(_fa);
    delete fa;
}

int64_t meta_size() { return sizeof(aiter::Signal); }

// ---- Internal dispatch helper ----

static void _all_reduce(fptr_t _fa, void* inp, void* out,
                        int64_t numel, AiterDtype dtype,
                        bool use_new, bool is_broadcast_reg_outptr)
{
    hipStream_t stream = aiter::getCurrentHIPStream();
    auto fa = reinterpret_cast<aiter::CustomAllreduce*>(_fa);
    switch(dtype)
    {
    case AITER_DTYPE_fp32: {
        fa->allreduce<opus::fp32_t>(stream,
                             reinterpret_cast<opus::fp32_t*>(inp),
                             reinterpret_cast<opus::fp32_t*>(out),
                             numel, use_new, is_broadcast_reg_outptr);
        break;
    }
    case AITER_DTYPE_fp16: {
        fa->allreduce<opus::fp16_t>(stream,
                                reinterpret_cast<opus::fp16_t*>(inp),
                                reinterpret_cast<opus::fp16_t*>(out),
                                numel, use_new, is_broadcast_reg_outptr);
        break;
    }
    case AITER_DTYPE_bf16: {
        fa->allreduce<opus::bf16_t>(stream,
                                      reinterpret_cast<opus::bf16_t*>(inp),
                                      reinterpret_cast<opus::bf16_t*>(out),
                                      numel, use_new);
        break;
    }
    default:
        throw std::runtime_error("gfx1250 custom allreduce only supports float32, float16 and bfloat16");
    }
}

// ---- Buffer registration (pointer-based) ----

void register_input_buffer(fptr_t _fa,
                           int64_t self_ptr,
                           const std::vector<int64_t>& all_ptrs)
{
    auto fa = reinterpret_cast<aiter::CustomAllreduce*>(_fa);
    fa->register_input_buffer(all_ptrs, (void*)self_ptr);
}

void register_output_buffer(fptr_t _fa,
                            int64_t self_ptr,
                            const std::vector<int64_t>& all_ptrs)
{
    auto fa = reinterpret_cast<aiter::CustomAllreduce*>(_fa);
    fa->register_output_buffer(all_ptrs, (void*)self_ptr);
}

int64_t get_graph_buffer_count(fptr_t _fa)
{
    auto fa = reinterpret_cast<aiter::CustomAllreduce*>(_fa);
    return (int64_t)(fa->graph_unreg_input_buffers_.size() +
                     fa->graph_unreg_output_buffers_.size());
}

void get_graph_buffer_ptrs(fptr_t _fa, int64_t ptrs_out)
{
    auto fa = reinterpret_cast<aiter::CustomAllreduce*>(_fa);
    auto ptrs = fa->get_graph_buffer_ptrs();
    std::memcpy((void*)ptrs_out, ptrs.data(), ptrs.size() * sizeof(int64_t));
}

void register_graph_buffers(fptr_t _fa,
                            const std::vector<int64_t>& ptrs_per_rank)
{
    auto fa = reinterpret_cast<aiter::CustomAllreduce*>(_fa);
    int world_size = fa->world_size_;
    int total_buffers = fa->graph_unreg_input_buffers_.size() +
                        fa->graph_unreg_output_buffers_.size();
    // ptrs_per_rank is a flat list: [rank0_buf0, rank0_buf1, ..., rank1_buf0, ...]
    std::vector<const int64_t*> per_rank(world_size);
    for(int i = 0; i < world_size; i++)
        per_rank[i] = &ptrs_per_rank[i * total_buffers];
    fa->register_graph_buffers(per_rank.data());
}

// ---- Public collective API ----

void all_reduce(fptr_t _fa,
                const aiter_tensor_t& inp,
                const aiter_tensor_t& out,
                bool use_new, bool open_fp8_quant,
                int64_t reg_inp_ptr, int64_t reg_inp_bytes)
{
    HipDeviceGuard device_guard(inp.device_id);
    hipStream_t stream = aiter::getCurrentHIPStream();
    auto dtype     = inp.dtype();
    int64_t numel  = inp.numel();
    int64_t data_bytes = numel * inp.element_size();

    void* actual_inp = inp.data_ptr();
    void* actual_out = out.data_ptr();

    bool is_broadcast_reg_outptr = (reg_inp_ptr == 0);

    if(reg_inp_ptr != 0)
    {
        if(data_bytes > reg_inp_bytes)
            throw std::runtime_error("registered buffer is too small to contain the input");
        HIP_CALL(hipMemcpyAsync((void*)reg_inp_ptr, actual_inp, data_bytes,
                                hipMemcpyDeviceToDevice, stream));
        actual_inp = (void*)reg_inp_ptr;
    }

    _all_reduce(_fa, actual_inp, actual_out, numel, dtype,
                use_new, is_broadcast_reg_outptr);
}

} // namespace aiter
