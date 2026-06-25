#pragma once
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
 * Self-contained gfx1250 (MI450) custom allreduce.
 * Does NOT include aiter_hip_common.h (avoids CK dependency).
 */
#include "opus/opus.hpp"
#include "hip_float8.h"
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <array>
#include <cstdint>
#include <iostream>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

// ---------------------------------------------------------------------------
// Utilities copied from aiter_hip_common.h to stay CK-free
// ---------------------------------------------------------------------------
#ifndef HIP_CALL
#define HIP_CALL(call)                                                       \
    do                                                                       \
    {                                                                        \
        hipError_t err = call;                                               \
        if(err != hipSuccess) [[unlikely]]                                   \
        {                                                                    \
            std::cerr << "[AITER] " << __FILE__ << ":" << __LINE__           \
                      << " fail to call " #call " ---> [HIP error]("        \
                      << hipGetErrorString(err) << ')' << std::endl;         \
            std::abort();                                                    \
        }                                                                    \
    } while(0)
#endif

#ifndef DINLINE
#define DINLINE __device__ __forceinline__
#endif

namespace aiter {

// ---------------------------------------------------------------------------
// Constants & data structures
// ---------------------------------------------------------------------------
constexpr int kMaxBlocks = 512;

struct Signal
{
    alignas(128) uint32_t start[kMaxBlocks][8];
    alignas(128) uint32_t end[kMaxBlocks][8];
    alignas(128) uint32_t _flag[kMaxBlocks];
};

struct __align__(16) RankData
{
    const void* ptrs[8];
};

struct __align__(16) RankSignals
{
    Signal* signals[8];
};

// ---------------------------------------------------------------------------
// Scalar cast helpers
// ---------------------------------------------------------------------------
template <typename inp_dtype>
DINLINE opus::fp32_t upcast_s(inp_dtype val)
{ return opus::cast<opus::fp32_t>(val); }

template <>
DINLINE opus::fp32_t upcast_s<opus::fp32_t>(opus::fp32_t val)
{ return val; }

template <typename out_dtype>
DINLINE out_dtype downcast_s(opus::fp32_t val)
{ return opus::cast<out_dtype>(val); }

template <>
DINLINE opus::fp32_t downcast_s<opus::fp32_t>(opus::fp32_t val)
{ return val; }

// ---------------------------------------------------------------------------
// Synchronisation primitives (ROCm path only)
// ---------------------------------------------------------------------------
template <int ngpus>
DINLINE void start_sync(const RankSignals& sg, Signal* self_sg, int rank)
{
    uint32_t flag = self_sg->_flag[blockIdx.x] + 1;
    if(threadIdx.x < ngpus)
    {
        __scoped_atomic_store_n(&sg.signals[threadIdx.x]->start[blockIdx.x][rank],
                                flag,
                                __ATOMIC_RELAXED,
                                __MEMORY_SCOPE_SYSTEM);
        while(__scoped_atomic_load_n(&self_sg->start[blockIdx.x][threadIdx.x],
                                     __ATOMIC_RELAXED,
                                     __MEMORY_SCOPE_DEVICE) < flag)
            ;
    }
    __syncthreads();
    if(threadIdx.x == 0)
        self_sg->_flag[blockIdx.x] = flag;
}

template <int ngpus, bool final_sync = false>
DINLINE void end_sync(const RankSignals& sg, Signal* self_sg, int rank)
{
    __syncthreads();
    uint32_t flag = self_sg->_flag[blockIdx.x] + 1;
    if(threadIdx.x < ngpus)
    {
        __scoped_atomic_store_n(&sg.signals[threadIdx.x]->end[blockIdx.x][rank],
                                flag,
                                final_sync ? __ATOMIC_RELAXED : __ATOMIC_RELEASE,
                                __MEMORY_SCOPE_SYSTEM);
        while(__scoped_atomic_load_n(&self_sg->end[blockIdx.x][threadIdx.x],
                                     final_sync ? __ATOMIC_RELAXED : __ATOMIC_ACQUIRE,
                                     __MEMORY_SCOPE_DEVICE) < flag)
            ;
    }
    __syncthreads();
    if(threadIdx.x == 0)
        self_sg->_flag[blockIdx.x] = flag;
}

// ---------------------------------------------------------------------------
// gfx1250 allreduce kernel
// ---------------------------------------------------------------------------
template <typename T, int ngpus, bool is_broadcast_reg_outptr = false>
__global__ void __launch_bounds__(256, 2) ar_gfx1250_naive_unroll4(
    RankData* _input_dp,
    RankData* _output_dp,
    RankSignals sg,
    Signal* self_sg,
    T* __restrict__ result,
    int rank,
    int size)
{
    constexpr int pack_size = 16 / sizeof(T);
    constexpr int unroll    = 4;
    using P                 = typename opus::vector_t<T, pack_size>;
    using A                 = typename opus::vector_t<opus::fp32_t, pack_size>;
    int tid    = blockIdx.x * blockDim.x + threadIdx.x;
    int nthds  = blockDim.x * gridDim.x;
    const P* ptrs[ngpus];
#pragma unroll
    for(int i = 0; i < ngpus; i++)
    {
        int target = (rank + i) % ngpus;
        ptrs[i]    = (const P*)_input_dp->ptrs[target];
    }
    start_sync<ngpus>(sg, self_sg, rank);
    int aligned_size = (size / unroll) * unroll;
    for(int base = tid * unroll; base < aligned_size; base += nthds * unroll)
    {
      P inp_reg[ngpus][unroll];
#pragma unroll
      for (int i = 0; i < ngpus; ++i)
      {
#pragma unroll
        for (int j = 0; j < unroll; ++j)
          inp_reg[i][j] = ptrs[i][base + j];
      }
      A rslt_tmp[unroll];
      P rslt_reg[unroll];
#pragma unroll
      for (int u = 0; u < unroll; ++u)
      {
#pragma unroll
        for (int j = 0; j < pack_size; ++j)
          rslt_tmp[u][j] = upcast_s(inp_reg[0][u][j]);
#pragma unroll
        for (int g = 1; g < ngpus; ++g)
        {
#pragma unroll
          for (int j = 0; j < pack_size; ++j)
            rslt_tmp[u][j] += upcast_s(inp_reg[g][u][j]);
        }
      }
#pragma unroll
      for (int u = 0; u < unroll; ++u)
      {
#pragma unroll
        for (int j = 0; j < pack_size; ++j)
          rslt_reg[u][j] = downcast_s<T>(rslt_tmp[u][j]);
        *(reinterpret_cast<P*>(result) + base + u) = rslt_reg[u];
      }
    }
    for(int idx = aligned_size + tid; idx < size; idx += nthds)
    {
      A acc;
#pragma unroll
      for (int j = 0; j < pack_size; ++j)
        acc[j] = upcast_s(ptrs[0][idx][j]);
#pragma unroll
      for (int i = 1; i < ngpus; ++i)
      {
#pragma unroll
        for (int j = 0; j < pack_size; ++j)
          acc[j] += upcast_s(ptrs[i][idx][j]);
      }
      P out_val;
#pragma unroll
      for (int j = 0; j < pack_size; ++j)
        out_val[j] = downcast_s<T>(acc[j]);
      *(reinterpret_cast<P*>(result) + idx) = out_val;
    }
    end_sync<ngpus, true>(sg, self_sg, rank);
}

// ---------------------------------------------------------------------------
// gfx1250 allgather kernel
// ---------------------------------------------------------------------------
template <typename T, int ngpus>
__global__ void __launch_bounds__(256, 2) ag_gfx1250_naive_unroll4(
    RankData* _input_dp,
    RankSignals sg,
    Signal* self_sg,
    T* __restrict__ result,
    int rank,
    int size)
{
    constexpr int pack_size = 16 / sizeof(T);
    constexpr int unroll    = 4;
    using P                 = typename opus::vector_t<T, pack_size>;
    int index    = blockIdx.x * blockDim.x * unroll + threadIdx.x;
    int stride  = blockDim.x * gridDim.x * unroll;
    const P* ptrs[ngpus];
#pragma unroll
    for(int i = 0; i < ngpus; i++)
    {
        ptrs[i] = (const P*)_input_dp->ptrs[i];
    }
    start_sync<ngpus>(sg, self_sg, rank);
    for (int idx = index; idx + blockDim.x * (unroll - 1) < size; idx += stride)
    {
#pragma unroll
      for (int i = 0; i < ngpus; ++i)
      {
        int rank_idx = (rank + i) % ngpus;
#pragma unroll
        for (int j = 0; j < unroll; ++j)
        {
          *(reinterpret_cast<P*>(result) + size * rank_idx + idx + j * blockDim.x) = ptrs[rank_idx][idx + j * blockDim.x];
        }
      }
    }
    end_sync<ngpus, true>(sg, self_sg, rank);
}

template <typename T, int ngpus>
__global__ void __launch_bounds__(256, 2) ag_gfx1250_warpsplit_unroll4(
    RankData* _input_dp,
    RankSignals sg,
    Signal* self_sg,
    T* __restrict__ result,
    int rank,
    int size)
{
    constexpr int pack_size = 16 / sizeof(T);
    constexpr int unroll    = 4;
    constexpr int tnum_gpu = 256 / ngpus;
    using P                 = typename opus::vector_t<T, pack_size>;
    int warp_id = threadIdx.x / tnum_gpu;
    int lane_id = threadIdx.x % tnum_gpu;
    int index    = blockIdx.x * tnum_gpu * unroll + lane_id;
    int stride  = blockDim.x * tnum_gpu * unroll;
    const P* ptrs[ngpus];
#pragma unroll
    for(int i = 0; i < ngpus; i++)
    {
        ptrs[i] = (const P*)_input_dp->ptrs[i];
    }
    start_sync<ngpus>(sg, self_sg, rank);
    for (int idx = index; idx + tnum_gpu * (unroll - 1) < size; idx += stride)
    {
#pragma unroll
      for (int i = 0; i < unroll; ++i)
      {
        P* rslt_addr = reinterpret_cast<P*>(result) + warp_id * size + idx + tnum_gpu * i;
        *rslt_addr = ptrs[warp_id][idx + i * tnum_gpu];
      }
    }
    end_sync<ngpus, true>(sg, self_sg, rank);
}

// ---------------------------------------------------------------------------
// gfx1250 bandwidth test kernel
// ---------------------------------------------------------------------------
template <typename T, int unroll>
__global__ void  p2p_bandwidth_test_kernel(
    RankData* _input_dp,
    RankSignals sg,
    Signal* self_sg,
    T* __restrict__ result,
    int rank,
    int size)
{
    constexpr int pack_size = 16 / sizeof(T);
    using P                 = typename opus::vector_t<T, pack_size>;
    int index    = blockIdx.x * blockDim.x * unroll + threadIdx.x;
    int stride  = blockDim.x * gridDim.x * unroll;
    const P* ptrs[2];
#pragma unroll
    for(int i = 0; i < 2; i++)
    {
        ptrs[i] = (const P*)_input_dp->ptrs[i];
    }
    start_sync<2>(sg, self_sg, rank);
    for (int idx = index; idx < size; idx += stride)
    {
      P reg[unroll];
#pragma unroll
      for (int i = 0; i < unroll; ++i)
      {
        reg[i] = ptrs[(rank + 1) % 2][idx + i * blockDim.x];
      }
#pragma unroll
      for (int i = 0; i < unroll; ++i)
      {
        *(reinterpret_cast<P*>(result) + idx + i * blockDim.x) = reg[i];
      }
    }
    // end_sync<2, true>(sg, self_sg, rank);
}

// ---------------------------------------------------------------------------
// CustomAllreduce class (gfx1250-only, simplified)
// ---------------------------------------------------------------------------
// gfx1250: hipIpc is not available. Buffer sharing uses torch's
// cross-process CUDA tensor sharing; the C++ layer receives direct
// device pointers that torch already mapped into each process's VA.

class CustomAllreduce
{
public:
    int rank_;
    int world_size_;
    bool full_nvlink_;

    RankSignals sg_;
    std::unordered_map<void*, RankData*> input_buffer;
    std::unordered_map<void*, RankData*> output_buffers_;
    Signal* self_sg_;

    RankData *d_rank_data_base_, *d_rank_data_end_;
    std::vector<void*> graph_unreg_input_buffers_;
    std::vector<void*> graph_unreg_output_buffers_;

    // gfx1250: hipIpc is not available. Instead, each rank's Signal buffer
    // is a torch-shared tensor whose device pointer is exchanged via the
    // distributed store.  The constructor receives the remote pointers
    // directly (torch already mapped them into this process's VA space).
    CustomAllreduce(Signal* meta,
                    void* rank_data,
                    size_t rank_data_sz,
                    const std::vector<int64_t>& all_meta_ptrs,
                    int rank,
                    bool fully_connected = true)
        : rank_(rank),
          world_size_(all_meta_ptrs.size()),
          full_nvlink_(fully_connected),
          self_sg_(meta),
          d_rank_data_base_(reinterpret_cast<RankData*>(rank_data)),
          d_rank_data_end_(d_rank_data_base_ + rank_data_sz / sizeof(RankData))
    {
        for(int i = 0; i < world_size_; i++)
        {
            sg_.signals[i] = reinterpret_cast<Signal*>(all_meta_ptrs[i]);
        }
    }

    // gfx1250: return raw device pointers (no hipIpc handles).
    std::vector<int64_t> get_graph_buffer_ptrs()
    {
        auto num_input_buffers  = graph_unreg_input_buffers_.size();
        auto num_output_buffers = graph_unreg_output_buffers_.size();
        auto num_buffers        = num_input_buffers + num_output_buffers;
        std::vector<int64_t> ptrs(num_buffers);
        for(size_t i = 0; i < num_input_buffers; i++)
            ptrs[i] = (int64_t)graph_unreg_input_buffers_[i];
        for(size_t i = 0; i < num_output_buffers; i++)
            ptrs[num_input_buffers + i] = (int64_t)graph_unreg_output_buffers_[i];
        return ptrs;
    }

    void check_rank_data_capacity(size_t num = 1)
    {
        if(d_rank_data_base_ + num > d_rank_data_end_)
            throw std::runtime_error("Rank data buffer is overflowed by " +
                                     std::to_string(d_rank_data_base_ + num - d_rank_data_end_));
    }

    // gfx1250: receive direct device pointers instead of IPC handles.
    void register_input_buffer(const std::vector<int64_t>& all_ptrs, void* self)
    {
        check_rank_data_capacity();
        RankData data;
        for(int i = 0; i < world_size_; i++)
            data.ptrs[i] = (i != rank_) ? (void*)all_ptrs[i] : self;
        auto d_data = d_rank_data_base_++;
        HIP_CALL(hipMemcpy(d_data, &data, sizeof(RankData), hipMemcpyHostToDevice));
        input_buffer[self] = d_data;
    }

    void register_output_buffer(const std::vector<int64_t>& all_ptrs, void* self)
    {
        check_rank_data_capacity();
        RankData data;
        for(int i = 0; i < world_size_; i++)
            data.ptrs[i] = (i != rank_) ? (void*)all_ptrs[i] : self;
        auto d_data = d_rank_data_base_++;
        HIP_CALL(hipMemcpy(d_data, &data, sizeof(RankData), hipMemcpyHostToDevice));
        output_buffers_[self] = d_data;
    }

    RankData* get_buffer_RD(hipStream_t stream, void* input)
    {
        auto it = input_buffer.find(input);
        if(it != input_buffer.end())
            return it->second;
        hipStreamCaptureStatus status;
        HIP_CALL(hipStreamIsCapturing(stream, &status));
        if(status == hipStreamCaptureStatusActive)
        {
            auto ptrs = d_rank_data_base_ + graph_unreg_input_buffers_.size();
            graph_unreg_input_buffers_.push_back(input);
            return ptrs;
        }
        throw std::runtime_error("buffer address " +
                                 std::to_string(reinterpret_cast<uint64_t>(input)) +
                                 " is not registered!");
    }

    RankData* get_output_buffer_RD(hipStream_t stream, void* output)
    {
        auto it = output_buffers_.find(output);
        if(it != output_buffers_.end())
            return it->second;
        hipStreamCaptureStatus status;
        HIP_CALL(hipStreamIsCapturing(stream, &status));
        if(status == hipStreamCaptureStatusActive)
        {
            auto ptrs = d_rank_data_base_ + graph_unreg_input_buffers_.size() +
                        graph_unreg_output_buffers_.size();
            graph_unreg_output_buffers_.push_back(output);
            return ptrs;
        }
        throw std::runtime_error("output buffer address " +
                                 std::to_string(reinterpret_cast<uint64_t>(output)) +
                                 " is not registered!");
    }

    // gfx1250: receive direct device pointers per rank per buffer.
    // ptrs_per_rank[rank_j] points to a flat array of int64_t device pointers,
    // one per buffer (inputs first, then outputs), in the same order as
    // graph_unreg_input_buffers_ + graph_unreg_output_buffers_.
    void register_graph_buffers(const int64_t* const* ptrs_per_rank)
    {
        auto num_input_buffers  = graph_unreg_input_buffers_.size();
        auto num_output_buffers = graph_unreg_output_buffers_.size();
        auto total_buffers      = num_input_buffers + num_output_buffers;
        check_rank_data_capacity(total_buffers);
        std::vector<RankData> rank_data(total_buffers);
        for(size_t i = 0; i < num_input_buffers; i++)
        {
            auto self_ptr = graph_unreg_input_buffers_[i];
            auto& rd      = rank_data[i];
            for(int j = 0; j < world_size_; j++)
                rd.ptrs[j] = (j != rank_) ? (void*)ptrs_per_rank[j][i] : self_ptr;
        }
        for(size_t i = 0; i < num_output_buffers; i++)
        {
            auto self_ptr = graph_unreg_output_buffers_[i];
            auto& rd      = rank_data[num_input_buffers + i];
            for(int j = 0; j < world_size_; j++)
                rd.ptrs[j] = (j != rank_) ? (void*)ptrs_per_rank[j][num_input_buffers + i]
                                          : self_ptr;
            output_buffers_[self_ptr] = d_rank_data_base_ + num_input_buffers + i;
        }
        HIP_CALL(hipMemcpy(d_rank_data_base_,
                           rank_data.data(),
                           sizeof(RankData) * total_buffers,
                           hipMemcpyHostToDevice));
        d_rank_data_base_ += total_buffers;
        graph_unreg_input_buffers_.clear();
        graph_unreg_output_buffers_.clear();
    }

    template <typename T>
    void allgather_naive(hipStream_t stream,
                         T* input,
                         T* output,
                         int size)
    {
        auto d = 16 / sizeof(T);
        if(size % d != 0)
            throw std::runtime_error(
                "allgather requires input length to be multiple of " + std::to_string(d));

        RankData* input_ptrs = get_buffer_RD(stream, input);
        size /= d;

        constexpr int threads = 256;
        int blocks = std::min(kMaxBlocks,
                              (size + threads * 4 - 1) / (threads * 4));
        if(world_size_ == 2)
            ag_gfx1250_naive_unroll4<T, 2><<<blocks, threads, 0, stream>>>(
                input_ptrs, sg_, self_sg_, output, rank_, size);
        else
            ag_gfx1250_naive_unroll4<T, 4><<<blocks, threads, 0, stream>>>(
                input_ptrs, sg_, self_sg_, output, rank_, size);
    }

    template <typename T>
    void allgather_warpsplit(hipStream_t stream,
                             T* input,
                             T* output,
                             int size)
    {
        auto d = 16 / sizeof(T);
        if(size % d != 0)
            throw std::runtime_error(
                "allgather requires input length to be multiple of " + std::to_string(d));

        RankData* input_ptrs = get_buffer_RD(stream, input);
        size /= d;

        constexpr int threads = 256;
        int blocks = std::min(kMaxBlocks,
                              (size + threads * 4 - 1) / (threads * 4));
        if(world_size_ == 2)
            ag_gfx1250_warpsplit_unroll4<T, 2><<<blocks, threads, 0, stream>>>(
                input_ptrs, sg_, self_sg_, output, rank_, size);
        else
            ag_gfx1250_warpsplit_unroll4<T, 4><<<blocks, threads, 0, stream>>>(
                input_ptrs, sg_, self_sg_, output, rank_, size);
    }

    template <typename T, int unroll>
    void p2p_bw_test(hipStream_t stream,
                     T* input,
                     T* output,
                     int size,
                     int threads,
                     int blocks)
    {
        auto d = 16 / sizeof(T);
        if(size % d != 0)
            throw std::runtime_error(
                "p2p_bw_test requires input length to be multiple of " + std::to_string(d));

        RankData* input_ptrs = get_buffer_RD(stream, input);
        size /= d;

        p2p_bandwidth_test_kernel<T, unroll><<<blocks, threads, 0, stream>>>(
            input_ptrs, sg_, self_sg_, output, rank_, size);
    }

    template <typename T>
    void allreduce(hipStream_t stream,
                   T* input,
                   T* output,
                   int size,
                   bool use_new                 = true,
                   bool is_broadcast_reg_outptr = false)
    {
        auto d = 16 / sizeof(T);
        if(size % d != 0)
            throw std::runtime_error(
                "custom allreduce requires input length to be multiple of " + std::to_string(d));

        RankData* input_ptrs = get_buffer_RD(stream, input);
        RankData* output_ptrs = nullptr;
        if(is_broadcast_reg_outptr)
            output_ptrs = get_output_buffer_RD(stream, output);

        size /= d;

        if(world_size_ > 4)
            throw std::runtime_error(
                "gfx1250 custom allreduce only supports world_size <= 4, got " +
                std::to_string(world_size_));

        constexpr int threads = 256;
        int blocks = std::min(kMaxBlocks,
                              (size + threads * 4 - 1) / (threads * 4));
        if(world_size_ == 2)
        {
            ar_gfx1250_naive_unroll4<T, 2><<<blocks, threads, 0, stream>>>(
                input_ptrs, output_ptrs, sg_, self_sg_, output, rank_, size);
        }
        else
        {
            ar_gfx1250_naive_unroll4<T, 4><<<blocks, threads, 0, stream>>>(
                input_ptrs, output_ptrs, sg_, self_sg_, output, rank_, size);
        }
    }
};

} // namespace aiter
