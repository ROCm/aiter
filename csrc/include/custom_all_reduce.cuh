#pragma once
/*
 * Copyright (C) Advanced Micro Devices, Inc. All rights reserved.
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
#include "aiter_hip_common.h"
#include "opus/opus.hpp"
#include "communication_asm.h"
#include "hip_float8.h"
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <iostream>
#include <limits>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

namespace aiter {

constexpr int kMaxBlocks = 80;
// note: we don't want to use atomics for signals because peer atomics are no
// supported on PCIe links
struct Signal
{
    alignas(128) uint32_t start[kMaxBlocks][8];
    alignas(128) uint32_t end[kMaxBlocks][8];
    alignas(128) uint32_t _flag[kMaxBlocks]; // incremental flags for each rank
};

#ifdef USE_ROCM
struct __align__(16) RankData { const void* ptrs[8]; };
#else
struct __align__(16) RankData { const void* __restrict__ ptrs[8]; };
#endif

struct __align__(16) RankSignals
{
#ifndef USE_ROCM
    volatile
#endif
        Signal* signals[8];
};

#define DINLINE __device__ __forceinline__

// scalar cast functions
template <typename inp_dtype>
DINLINE opus::fp32_t upcast_s(inp_dtype val)
{
    return opus::cast<opus::fp32_t>(val);
}

template <>
DINLINE opus::fp32_t upcast_s<opus::fp32_t>(opus::fp32_t val)
{
    return val;
}

template <typename out_dtype>
DINLINE out_dtype downcast_s(opus::fp32_t val)
{
    return opus::cast<out_dtype>(val);
}

template <>
DINLINE opus::fp32_t downcast_s<opus::fp32_t>(opus::fp32_t val)
{
    return val;
}

// scalar add functions
// for some reason when compiling with Pytorch, the + operator for half and
// bfloat is disabled so we call the intrinsics directly
template <typename T, int N>
DINLINE opus::vector_t<T, N>& packed_assign_add(opus::vector_t<T, N>& a, opus::vector_t<T, N> b)
{
    if constexpr(std::is_same<T, opus::fp32_t>::value)
    {
        a += b;
    }
    else
    {
#pragma unroll
        for(int i = 0; i < N; i++)
        {
            a[i] = downcast_s<T>(upcast_s(a[i]) + upcast_s(b[i]));
        }
    }
    return a;
}

// not support fp8 pack convert
template <typename V, std::enable_if_t<opus::is_vector_v<V>, bool> = true>
DINLINE auto upcast(V val) -> opus::vector_t<float, opus::vector_traits<V>::size()>
{
    using T = typename opus::vector_traits<V>::dtype;
    constexpr int N = opus::vector_traits<V>::size();
    if constexpr(std::is_same<T, opus::fp32_t>::value)
    {
        return val;
    }
    else
    {
        opus::vector_t<float, N> out;
#pragma unroll
        for(int i = 0; i < N; i++)
        {
            out[i] = upcast_s(val[i]);
        }
        return out;
    }
}

template <typename O, typename V, std::enable_if_t<opus::is_vector_v<V>, bool> = true>
DINLINE O downcast(V val)
{
    using T = typename opus::vector_traits<O>::dtype;
    constexpr int N = opus::vector_traits<O>::size();
    if constexpr(std::is_same<T, float>::value)
    {
        return val;
    }
    else
    {
        O out;
#pragma unroll
        for(int i = 0; i < N; i++)
        {
            out[i] = downcast_s<T>(val[i]);
        }
        return out;
    }
}

// This function is meant to be used as the first synchronization in the all
// reduce kernel. Thus, it doesn't need to make any visibility guarantees for
// prior memory accesses. Note: volatile writes will not be reordered against
// other volatile writes.
template <int ngpus>
DINLINE void start_sync(const RankSignals& sg,
#ifndef USE_ROCM
                        volatile
#endif
                        Signal* self_sg,
                        int rank)
{
#ifdef USE_ROCM
    uint32_t flag = self_sg->_flag[blockIdx.x] + 1;
    if(threadIdx.x < ngpus)
    {
        // simultaneously write to the corresponding flag of all ranks.
        // Latency = 1 p2p write
        __scoped_atomic_store_n(&sg.signals[threadIdx.x]->start[blockIdx.x][rank],
                                flag,
                                __ATOMIC_RELAXED,
                                __MEMORY_SCOPE_SYSTEM);
        // wait until we got true from all ranks
        while(__scoped_atomic_load_n(&self_sg->start[blockIdx.x][threadIdx.x],
                                     __ATOMIC_RELAXED,
                                     __MEMORY_SCOPE_DEVICE) < flag)
            ;
    }
    __syncthreads();
    // use one thread to update flag
    if(threadIdx.x == 0)
        self_sg->_flag[blockIdx.x] = flag;
#else
    if(threadIdx.x < ngpus)
    {
        // reset flag for next time
        self_sg->end[blockIdx.x][threadIdx.x] = 0;
        // simultaneously write to the corresponding flag of all ranks.
        // Latency = 1 p2p write
        sg.signals[threadIdx.x]->start[blockIdx.x][rank] = 1;
        // wait until we got true from all ranks
        while(!self_sg->start[blockIdx.x][threadIdx.x])
            ;
    }
    __syncthreads();
#endif
}

// This function is meant to be used as the second or the final synchronization
// barrier in the all reduce kernel. If it's the final synchronization barrier,
// we don't need to make any visibility guarantees for prior memory accesses.
template <int ngpus, bool final_sync = false>
DINLINE void end_sync(const RankSignals& sg,
#ifndef USE_ROCM
                      volatile
#endif
                      Signal* self_sg,
                      int rank)
{
#ifdef USE_ROCM
    __syncthreads();
    // eliminate the case that prior writes are not visible after signals become
    // visible. Note that I did not managed to make this happen through a lot of
    // testing. Might be the case that hardware provides stronger guarantee than
    // the memory model.
    uint32_t flag = self_sg->_flag[blockIdx.x] + 1;
    if(threadIdx.x < ngpus)
    {
        // simultaneously write to the corresponding flag of all ranks.
        // Latency = 1 p2p write
        __scoped_atomic_store_n(&sg.signals[threadIdx.x]->end[blockIdx.x][rank],
                                flag,
                                final_sync ? __ATOMIC_RELAXED : __ATOMIC_RELEASE,
                                __MEMORY_SCOPE_SYSTEM);
        // wait until we got true from all ranks
        while(__scoped_atomic_load_n(&self_sg->end[blockIdx.x][threadIdx.x],
                                     final_sync ? __ATOMIC_RELAXED : __ATOMIC_ACQUIRE,
                                     __MEMORY_SCOPE_DEVICE) < flag)
            ;
    }
    __syncthreads();
    // use one thread to update flag
    if(threadIdx.x == 0)
        self_sg->_flag[blockIdx.x] = flag;
#else
    __syncthreads();
    // eliminate the case that prior writes are not visible after signals become
    // visible. Note that I did not managed to make this happen through a lot of
    // testing. Might be the case that hardware provides stronger guarantee than
    // the memory model.
    if constexpr(!final_sync)
        __threadfence_system();
    if(threadIdx.x < ngpus)
    {
        // reset flag for next time
        self_sg->start[blockIdx.x][threadIdx.x] = 0;
        // simultaneously write to the corresponding flag of all ranks.
        // Latency = 1 p2p write
        sg.signals[threadIdx.x]->end[blockIdx.x][rank] = 1;
        // wait until we got true from all ranks
        while(!self_sg->end[blockIdx.x][threadIdx.x])
            ;
    }
    if constexpr(!final_sync)
        __syncthreads();
#endif
}

template <typename P, int ngpus, typename A>
DINLINE P packed_reduce(const P* ptrs[], int idx)
{
    A tmp = upcast(ptrs[0][idx]);
#pragma unroll
    for(int i = 1; i < ngpus; i++)
    {
        packed_assign_add<typename opus::vector_traits<A>::dtype, opus::vector_traits<A>::size()>(tmp, upcast(ptrs[i][idx]));
    }
    return downcast<P>(tmp);
}

template <typename T, int ngpus, bool is_broadcast_reg_outptr = false>
__global__ void __launch_bounds__(512, 1) cross_device_reduce_1stage_naive(RankData* _input_dp,
                                                                           RankData* _output_dp,
                                                                           RankSignals sg,
#ifndef USE_ROCM
                                                                           volatile
#endif
                                                                           Signal* self_sg,
                                                                           T* __restrict__ result,
                                                                           int rank,
                                                                           int size)
{
    constexpr int pack_size = 16 / sizeof(T);
    using P = typename opus::vector_t<T, pack_size>;
    using A = typename opus::vector_t<opus::fp32_t, pack_size>;
    // note: we don't reorder the address so the accumulation order is the same
    // for all ranks, ensuring bitwise identical results
    auto dp = *_input_dp;
    start_sync<ngpus>(sg, self_sg, rank);
    // do the actual reduction
    for(int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < size; idx += gridDim.x * blockDim.x)
    {
        ((P*)result)[idx] = packed_reduce<P, ngpus, A>((const P**)&dp.ptrs[0], idx);
    }
    end_sync<ngpus, true>(sg, self_sg, rank);
}

template <typename P>
#ifdef USE_ROCM
DINLINE P* get_tmp_buf(Signal* sg)
{
#else
DINLINE P* get_tmp_buf(volatile Signal* sg)
{
#endif
    return (P*)(((Signal*)sg) + 1);
}

template <typename T, int ngpus, bool is_broadcast_reg_outptr = false>
__global__ void __launch_bounds__(512, 1) cross_device_reduce_2stage_naive(RankData* _input_dp,
                                                                           RankData* _output_dp,
                                                                           RankSignals sg,
#ifndef USE_ROCM
                                                                           volatile
#endif
                                                                           Signal* self_sg,
                                                                           T* __restrict__ result,
                                                                           int rank,
                                                                           int size)
{
    constexpr int pack_size = 16 / sizeof(T);
    int tid                 = blockIdx.x * blockDim.x + threadIdx.x;
    int stride              = gridDim.x * blockDim.x;
    using P                 = typename opus::vector_t<T, pack_size>;
    using A                 = typename opus::vector_t<opus::fp32_t, pack_size>;
    int part                = size / ngpus;
    int start               = rank * part;
    int end                 = rank == ngpus - 1 ? size : start + part;
    int largest_part        = part + size % ngpus;
    const P* ptrs[ngpus];
    P* tmps[ngpus];
#pragma unroll
    for(int i = 0; i < ngpus; i++)
    {
        int target = (rank + i) % ngpus;
        ptrs[i]    = (const P*)_input_dp->ptrs[target];
        tmps[i]    = get_tmp_buf<P>(sg.signals[target]);
    }
    auto tmp_out = tmps[0];
    start_sync<ngpus>(sg, self_sg, rank);
    // stage 1: reduce scatter
    for(int idx = start + tid; idx < end; idx += stride)
    {
        tmp_out[idx - start] = packed_reduce<P, ngpus, A>(ptrs, idx);
    }
    end_sync<ngpus>(sg, self_sg, rank);

    // stage 2: allgather. Note: it's important to match the tid between
    // the two stages, because visibility across devices is only guaranteed
    // between threads that have the same tid. If thread i computes the sum of
    // start + i in the first stage, then thread i also gathers start + i from all
    // ranks.
    for(int idx = tid; idx < largest_part; idx += stride)
    {
#pragma unroll
        for(int i = 0; i < ngpus; i++)
        {
            int gather_from_rank = ((rank + i) % ngpus);
            if(gather_from_rank == ngpus - 1 || idx < part)
            {
                int dst_idx           = gather_from_rank * part + idx;
                ((P*)result)[dst_idx] = tmps[i][idx];
            }
        }
    }
}

#define THREAD_NUM 512

template <typename T, int ngpus, bool is_broadcast_reg_outptr = false>
__global__ void __launch_bounds__(512, 1) cross_device_reduce_1stage(RankData* _input_dp,
                                                                     RankData* _output_dp,
                                                                     RankSignals sg,
#ifndef USE_ROCM
                                                                     volatile
#endif
                                                                     Signal* self_sg,
                                                                     T* __restrict__ result,
                                                                     int rank,
                                                                     int size)
{
    constexpr int pack_size = 16 / sizeof(T);
    using P = typename opus::vector_t<T, pack_size>;
    using A = typename opus::vector_t<opus::fp32_t, pack_size>;
    constexpr int tnum_gpu = THREAD_NUM / ngpus;
    __shared__ T tmp_smem[tnum_gpu * ngpus * pack_size];
    // note: we don't reorder the address so the accumulation order is the same
    // for all ranks, ensuring bitwise identical results
    auto dp = *_input_dp;

    // load one gpu data each wave
    int warp_id = threadIdx.x / tnum_gpu;
    int lane_id = threadIdx.x % tnum_gpu;
    start_sync<ngpus>(sg, self_sg, rank);
    // do the actual reduction
    for(int idx = blockIdx.x * tnum_gpu + lane_id; idx < size; idx += gridDim.x * tnum_gpu)
    {
        *(reinterpret_cast<P*>(&tmp_smem[0]) + threadIdx.x) =
            ((const P**)&dp.ptrs[0])[warp_id][idx];
        __syncthreads();
        if(warp_id == 0)
        {
            A add_reg;
#pragma unroll
            for(int i = 0; i < pack_size; ++i)
            {
                add_reg[i] =
                    upcast_s(tmp_smem[threadIdx.x * pack_size + i]);
            }
            constexpr int smem_gpu_loop_stride = tnum_gpu * pack_size;
#pragma unroll
            for(int i = 1; i < ngpus; ++i)
            {
#pragma unroll
                for(int j = 0; j < pack_size; ++j)
                {
                    add_reg[j] += upcast_s(
                        tmp_smem[smem_gpu_loop_stride * i + threadIdx.x * pack_size + j]);
                }
            }
            P write_reg;
#pragma unroll
            for(int i = 0; i < pack_size; ++i)
            {
                write_reg[i] = downcast_s<T>(add_reg[i]);
            }
            ((P*)result)[idx] = write_reg;
        }
        __syncthreads();
    }
    // maybe do not need device sync
    // end_sync<ngpus, true>(sg, self_sg, rank);
}

template <typename T, int ngpus, bool is_broadcast_reg_outptr = false>
__global__ void __launch_bounds__(512, 1) cross_device_reduce_2stage(RankData* _input_dp,
                                                                     RankData* _output_dp,
                                                                     RankSignals sg,
#ifndef USE_ROCM
                                                                     volatile
#endif
                                                                     Signal* self_sg,
                                                                     T* __restrict__ result,
                                                                     int rank,
                                                                     int size)
{
    constexpr int pack_size = 16 / sizeof(T);
    constexpr int tnum_gpu  = THREAD_NUM / ngpus;
    using P                 = typename opus::vector_t<T, pack_size>;
    using A                 = typename opus::vector_t<opus::fp32_t, pack_size>;
    int warp_id             = threadIdx.x / tnum_gpu;
    int lane_id             = threadIdx.x % tnum_gpu;
    int tid                 = blockIdx.x * tnum_gpu + lane_id;
    int stride              = gridDim.x * tnum_gpu;
    int part                = size / ngpus;
    int start               = rank * part;
    int end                 = rank == ngpus - 1 ? size : start + part;
    int largest_part        = part + size % ngpus;
    __shared__ T tmp_smem[tnum_gpu * ngpus * pack_size];
    const P* ptrs[ngpus];
    P* tmps[ngpus];
#pragma unroll
    for(int i = 0; i < ngpus; i++)
    {
        int target = (rank + i) % ngpus;
        ptrs[i]    = (const P*)_input_dp->ptrs[target];
        tmps[i]    = get_tmp_buf<P>(sg.signals[target]);
    }
    auto tmp_out = tmps[0];
    start_sync<ngpus>(sg, self_sg, rank);
    // stage 1: reduce scatter
    for(int idx = start + tid; idx < end; idx += stride)
    {
        *(reinterpret_cast<P*>(&tmp_smem[0]) + threadIdx.x) = ptrs[warp_id][idx];
        __syncthreads();
        // cal add in first 64 threads
        if(warp_id == 0)
        {
            A add_reg;
#pragma unroll
            for(int i = 0; i < pack_size; ++i)
            {
                add_reg[i] =
                    upcast_s(tmp_smem[pack_size * threadIdx.x + i]);
            }
            constexpr int smem_gpu_loop_stride = tnum_gpu * pack_size;
#pragma unroll
            for(int i = 1; i < ngpus; ++i)
            {
#pragma unroll
                for(int j = 0; j < pack_size; ++j)
                {
                    add_reg[j] += upcast_s(
                        tmp_smem[i * smem_gpu_loop_stride + pack_size * threadIdx.x + j]);
                }
            }
            P write_reg;
#pragma unroll
            for(int i = 0; i < pack_size; ++i)
            {
                write_reg[i] = downcast_s<T>(add_reg[i]);
            }
            tmp_out[idx - start] = write_reg;
        }
        __syncthreads();
    }
    end_sync<ngpus>(sg, self_sg, rank);

    // stage 2: allgather. Note: it's important to match the tid between
    // the two stages, because visibility across devices is only guaranteed
    // between threads that have the same tid. If thread i computes the sum of
    // start + i in the first stage, then thread i also gathers start + i from all
    // ranks.
    for(int idx = tid; idx < largest_part; idx += stride)
    {
        int dst_idx           = (warp_id + rank) % ngpus * part + idx;
        ((P*)result)[dst_idx] = tmps[warp_id][idx];
    }
}

template <typename T, int ngpus, bool is_broadcast_reg_outptr = false>
__global__ void __launch_bounds__(512, 1) cross_device_reduce_2stage_write_mode(RankData* _input_dp,
                                                                                RankData* _output_dp,
                                                                                RankSignals sg,
#ifndef USE_ROCM
                                                                                volatile
#endif
                                                                                Signal* self_sg,
                                                                                T* __restrict__ result,
                                                                                int rank,
                                                                                int size)
{
    constexpr int pack_size = 16 / sizeof(T);
    constexpr int tnum_gpu  = THREAD_NUM / ngpus;
    using P                 = typename opus::vector_t<T, pack_size>;
    using A                 = typename opus::vector_t<opus::fp32_t, pack_size>;
    __shared__ T tmp_smem[tnum_gpu * ngpus * pack_size];
    __shared__ T res_smem[tnum_gpu * pack_size];
    int warp_id       = threadIdx.x / tnum_gpu;
    int lane_id       = threadIdx.x % tnum_gpu;
    int tid           = blockIdx.x * tnum_gpu + lane_id;
    int stride        = gridDim.x * tnum_gpu;
    int part          = size / ngpus;
    P* output_ptrs[ngpus];
    P* tmps[ngpus];
#pragma unroll
    for(int i = 0; i < ngpus; i++)
    {
        tmps[i]           = get_tmp_buf<P>(sg.signals[i]);
    }
    if (is_broadcast_reg_outptr)
    {
#pragma unroll
        for(int i = 0; i < ngpus; i++)
        {
            output_ptrs[i] = (P*)_output_dp->ptrs[i];
        }  
    }
    const P* input_ptr = (const P*)_input_dp->ptrs[rank];
    auto tmp_out = tmps[rank];
    int stage3_offset = size;
 
    // stage1: write local rank data to remote rank
    int start        = warp_id * part;
    int end          = warp_id == ngpus - 1 ? size : start + part;
    for(int idx = start + tid; idx < end; idx += stride)
    {
        tmps[warp_id][rank * part + idx - start] = input_ptr[idx];
    }
    end_sync<ngpus>(sg, self_sg, rank);
 
    // stage 2: reduce scatter & write result to remote rank
    end = rank != ngpus - 1 ? part : size - part * (ngpus-1);
    for(int idx = tid; idx < end; idx += stride)
    {
        *(reinterpret_cast<P*>(&tmp_smem[0]) + threadIdx.x) = tmp_out[warp_id * part + idx];
        __syncthreads();
        // cal add in first 64 threads
        if(warp_id == 0)
        {
            A add_reg;
#pragma unroll
            for(int i = 0; i < pack_size; ++i)
            {
                add_reg[i] =
                    upcast_s(tmp_smem[pack_size * threadIdx.x + i]);
            }
            constexpr int smem_gpu_loop_stride = tnum_gpu * pack_size;
#pragma unroll
            for(int i = 1; i < ngpus; ++i)
            {
#pragma unroll
                for(int j = 0; j < pack_size; ++j)
                {
                    add_reg[j] += upcast_s(
                        tmp_smem[i * smem_gpu_loop_stride + pack_size * threadIdx.x + j]);
                }
            }
            P write_reg;
#pragma unroll
            for(int i = 0; i < pack_size; ++i)
            {
                write_reg[i] = downcast_s<T>(add_reg[i]);
            }
            *(reinterpret_cast<P*>(&res_smem[0]) + lane_id) = write_reg;
        }
        __syncthreads();
        // send data to remote rank
        if (is_broadcast_reg_outptr)
        {
            P temp_val = *(reinterpret_cast<P*>(&res_smem[0]) + lane_id);
            auto src_addr = (reinterpret_cast<int*>(&temp_val));
            auto dst_addr = (reinterpret_cast<int*>(&output_ptrs[warp_id][rank * part + idx]));
            __builtin_nontemporal_store(*src_addr, dst_addr);
            __builtin_nontemporal_store(*(src_addr + 1), dst_addr + 1);
            __builtin_nontemporal_store(*(src_addr + 2), dst_addr + 2);
            __builtin_nontemporal_store(*(src_addr + 3), dst_addr + 3);
        } else {
            tmps[warp_id][rank * part + idx + stage3_offset] = *(reinterpret_cast<P*>(&res_smem[0]) + lane_id);
        }
    }
    end_sync<ngpus>(sg, self_sg, rank);

    if (!is_broadcast_reg_outptr)
    {
        // stage 3: get the output from tmp_buffer
        end = warp_id == ngpus - 1 ? size : start + part;
        for(int idx = start + tid; idx < end; idx += stride)
        {
            ((P*)result)[idx] = tmp_out[idx + stage3_offset];
        }
    }
}

/*
 * naive allgather
 * for case: input(1345,)
 * */
template <typename T, int ngpus>
__global__ void __launch_bounds__(512, 1) allgather_naive(
    RankData* _dp, RankSignals sg, Signal* self_sg, T* __restrict__ result, int rank, int size)
{
    constexpr int tnum_gpu = THREAD_NUM / ngpus;
    int warp_id            = threadIdx.x / tnum_gpu;
    int lane_id            = threadIdx.x % tnum_gpu;
    int tid                = blockIdx.x * tnum_gpu + lane_id;
    int stride             = gridDim.x * tnum_gpu;
    const T* ptrs[ngpus];

#pragma unroll
    for(int i = 0; i < ngpus; ++i)
    {
        ptrs[i] = (const T*)_dp->ptrs[i];
    }
    start_sync<ngpus>(sg, self_sg, rank);

    for(int idx = tid; idx < size; idx += stride)
    {
        int write_idx     = warp_id * size + idx;
        result[write_idx] = ptrs[warp_id][idx];
    }
}

template <typename T, int ngpus>
__global__ void __launch_bounds__(512, 1) allgather_vec(
    RankData* _dp, RankSignals sg, Signal* self_sg, T* __restrict__ result, int rank, int size)
{
    constexpr int tnum_gpu = THREAD_NUM / ngpus;
    constexpr int pack_size = 16 / sizeof(T);
    using P                = typename opus::vector_t<T, pack_size>;
    int warp_id            = threadIdx.x / tnum_gpu;
    int lane_id            = threadIdx.x % tnum_gpu;
    int tid                = blockIdx.x * tnum_gpu + lane_id;
    int stride             = gridDim.x * tnum_gpu;
    const P* ptrs[ngpus];

#pragma unroll
    for(int i = 0; i < ngpus; ++i)
    {
        ptrs[i] = (const P*)_dp->ptrs[i];
    }
    start_sync<ngpus>(sg, self_sg, rank);

    for(int idx = tid; idx < size; idx += stride)
    {
        int write_idx                                   = warp_id * size + idx;
        *(reinterpret_cast<P*>(&result[0]) + write_idx) = ptrs[warp_id][idx];
    }
}

/*
 * reduce_scatter, at first dim
 * range = size / (pack_size * ngpu)
 * for case:
 *  input:(ngpus * n) -> output:(n)
 *  input:(ngpus * m, n, ...) -> output(m, n, ...)
 * cond: size % (pack_size * ngpus) == 0
 * */
template <typename T, int ngpus>
__global__ void __launch_bounds__(512, 1) reduce_scatter_first_dim(
    RankData* _dp, RankSignals sg, Signal* self_sg, T* __restrict__ result, int rank, int range)
{
    int tid    = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    constexpr int pack_size = 16 / sizeof(T);
    using P    = typename opus::vector_t<T, pack_size>;
    using A    = typename opus::vector_t<opus::fp32_t, pack_size>;
    const P* ptrs[ngpus];
#pragma unroll
    for(int i = 0; i < ngpus; i++)
    {
        int target = (rank + i) % ngpus;
        ptrs[i]    = (const P*)_dp->ptrs[target];
    }
    start_sync<ngpus>(sg, self_sg, rank);

    for(int idx = tid; idx < range; idx += stride)
    {
        int load_index  = rank * range + idx;
        int store_index = idx;
        *(reinterpret_cast<P*>(result) + store_index) =
            packed_reduce<P, ngpus, A>(ptrs, load_index);
    }
}

// fp8 quant all-reduce code start
template <typename T>
struct Fp16Filter
{
    static const bool value = false;
};

template <>
struct Fp16Filter<opus::fp16_t>
{
    static const bool value = true;
};

template <typename T>
struct Bf16Filter
{
    static const bool value = false;
};

template <>
struct Bf16Filter<opus::bf16_t>
{
    static const bool value = true;
};

// dtypes only support half and bf16 now
#define FP16_FILTER typename std::enable_if<Fp16Filter<T>::value, void>::type* = nullptr

#define BF16_FILTER typename std::enable_if<Bf16Filter<T>::value, void>::type* = nullptr

template <template <typename> class functor, typename T, int size>
DINLINE T packReduce(opus::vector_t<T, size> pack)
{
    auto op   = functor<T>();
    T ret_val = pack[0];
#pragma unroll
    for(int i = 1; i < size; ++i)
    {
        ret_val = op(ret_val, pack[i]);
    }
    return ret_val;
}

template <template <typename> class functor, typename T, int size>
DINLINE opus::vector_t<T, size> packOp(opus::vector_t<T, size> a, opus::vector_t<T, size> b)
{
    auto op = functor<T>();
    opus::vector_t<T, size> ret_pack;
#pragma unroll
    for(int i = 0; i < size; ++i)
    {
        ret_pack[i] = op(a[i], b[i]);
    }
    return ret_pack;
}

template <typename T>
struct AddFunctor
{
    DINLINE T operator()(T a, T b)
    {
        opus::fp32_t a_fp32 = upcast_s(a);
        opus::fp32_t b_fp32 = upcast_s(b);
        return downcast_s<T>(a_fp32 + b_fp32);
    }
};

template <>
struct AddFunctor<opus::fp32_t>
{
    DINLINE opus::fp32_t operator()(opus::fp32_t a, opus::fp32_t b)
    {
        return a + b;
    }
};

// MLA metadata used this specialisation
template <>
struct AddFunctor<int>
{
    DINLINE int operator()(int a, int b)
    {
        return a + b;
    }
};

template <typename T>
struct MaxFunctor
{
    DINLINE T operator()(T a, T b) { return max(a, b); }
};

/*
 * todo:
 * static_cast may not safe
 * need a convert dtype template function defined by myself
 *
 * done
 * */
template <typename T>
struct AbsMaxFunctor
{
    DINLINE T operator()(T a, T b)
    {
        T zero_t = downcast_s<T>(0.0f);
        a        = a > zero_t ? a : zero_t - a;
        b        = b > zero_t ? b : zero_t - b;
        return max(a, b);
    }
};

template <template <typename> class functor, typename T, int reduce_range>
DINLINE T warpReduce(T val)
{
    auto op = functor<T>();
#pragma unroll
    for(int stride = reduce_range / 2; stride > 0; stride >>= 1)
    {
        T tmp = __shfl_xor(val, stride, reduce_range);
        val   = op(val, tmp);
    }
    return val;
}

// the following code only support bf16 and fp16
// pack_size must be divisible by 4
// TODO: check if pack_size is divisible by 4
template <typename T, int pack_size>
DINLINE opus::vector_t<opus::fp8_t, pack_size> packQuant(opus::vector_t<T, pack_size> inp_pack, T scale_functor)
{
    opus::vector_t<opus::fp8_t, pack_size> ret_val;
#pragma unroll
    for(int i = 0; i < pack_size / 4; ++i)
    {
        opus::fp32x4_t tmp;
#pragma unroll
        for (int j = 0; j < 4; ++j)
        {
            tmp[j] = upcast_s(inp_pack[i * 4 + j]);
        }
        *(reinterpret_cast<opus::fp8x4_t*>(&ret_val) + i) = opus::cast<opus::fp8_t>(tmp / upcast_s(scale_functor));
    }
    return ret_val;
}

template <typename T, int pack_size>
DINLINE opus::vector_t<T, pack_size> packDequant(opus::vector_t<opus::fp8_t, pack_size> inp_pack, T scale_functor)
{
    opus::vector_t<T, pack_size> ret_val;
#pragma unroll
    for (int i = 0; i < pack_size / 4; ++i)
    {
        opus::fp32x4_t tmp = opus::cast<opus::fp32_t>(*(reinterpret_cast<opus::fp8x4_t*>(&inp_pack) + i));
        tmp *= upcast_s(scale_functor);
#pragma unroll
        for (int j = 0; j < 4; ++j)
        {
            ret_val[i * 4 + j] = downcast_s<T>(tmp[j]);
        }
    }
    return ret_val;
}

template <typename T, int pack_size, int ngpus>
DINLINE opus::vector_t<T, pack_size> multiGPUPackReduce(const opus::vector_t<T, pack_size>* ptrs[ngpus],
                                                 int index)
{
    opus::vector_t<opus::fp32_t, pack_size> ret_val = upcast(ptrs[0][index]);
#pragma unroll
    for(int gpu_id = 1; gpu_id < ngpus; ++gpu_id)
    {
        opus::vector_t<opus::fp32_t, pack_size> tmp = upcast(ptrs[gpu_id][index]);
        ret_val += tmp;
    }
    return downcast<opus::vector_t<T, pack_size>>(ret_val);
}

// bf16 quant fp8 kernel function
// too slow need to be optimized
// fp16
template <typename T, int quant_scale, int pack_size, int ngpus, FP16_FILTER>
__global__ __forceinline__ void __launch_bounds__(512, 1) allReduceQuantFp8(
    RankData* _dp, RankSignals sg, Signal* self_sg, T* __restrict__ result, int rank, int size)
{
    float FP8_UPBOUND = opus::cast<opus::fp32_t>(opus::numeric_limits<opus::fp8_t>::max());
    int tid           = blockIdx.x * blockDim.x + threadIdx.x;
    int stride        = gridDim.x * blockDim.x;
    using inp_pack    = opus::vector_t<T, pack_size>;
    using fp8_pack    = opus::vector_t<opus::fp8_t, pack_size>;
    int part          = size / ngpus;
    int start         = rank * part;
    int end           = rank == ngpus - 1 ? size : start + part;
    int largest_part  = part + size % ngpus;
    const inp_pack* ptrs[ngpus];
    fp8_pack* tmps[ngpus];
#pragma unroll
    for(int i = 0; i < ngpus; i++)
    {
        int target = (rank + i) % ngpus;
        ptrs[i]    = (const inp_pack*)_dp->ptrs[target];
        tmps[i]    = get_tmp_buf<fp8_pack>(sg.signals[target]);
    }
    auto tmp_out = tmps[0];
    start_sync<ngpus>(sg, self_sg, rank);
    // stage 1: reduce scatter
    for(int idx = start + tid; idx < end; idx += stride)
    {
        inp_pack half8_reg;
        // half8_reg = packed_reduce<P, ngpus, A>(ptrs, idx);
        half8_reg                = multiGPUPackReduce<T, pack_size, ngpus>(ptrs, idx);
        ((inp_pack*)result)[idx] = half8_reg;
        // quant
        T thread_max = packReduce<AbsMaxFunctor, T, pack_size>(half8_reg);
        thread_max   = warpReduce<MaxFunctor, T, quant_scale / pack_size>(thread_max);
        T scale_factor =
            downcast_s<T>(upcast_s(thread_max) / FP8_UPBOUND);
        tmp_out[idx - start] = packQuant<T, pack_size>(half8_reg, scale_factor);
        if(threadIdx.x % (quant_scale / pack_size) == 0)
        {
            *(reinterpret_cast<T*>(&tmp_out[part]) + (idx - start) / (quant_scale / pack_size)) =
                scale_factor;
        }
    }
    end_sync<ngpus>(sg, self_sg, rank);

    // stage 2: all-gather
    for(int idx = tid; idx < largest_part; idx += stride)
    {
#pragma unroll
        for(int i = 1; i < ngpus; i++)
        {
            int gather_from_rank = ((rank + i) % ngpus);
            if(gather_from_rank == ngpus - 1 || idx < part)
            {
                // dequant
                T scale_factor;
                int factor_stride = quant_scale / pack_size;
                if(threadIdx.x % factor_stride == 0)
                {
                    scale_factor = *(reinterpret_cast<T*>(&tmps[i][part]) + idx / factor_stride);
                }
                scale_factor = __shfl(scale_factor, (threadIdx.x / factor_stride) * factor_stride);
                inp_pack half8_reg = packDequant<T, pack_size>(tmps[i][idx], scale_factor);
                int dst_idx        = gather_from_rank * part + idx;
                ((inp_pack*)result)[dst_idx] = half8_reg;
            }
        }
    }
}

// fused allreduce rmsnorm first step
template <typename T, int ngpus>
__global__ void __launch_bounds__(512, 1) reduce_scatter_cross_device_store(
    RankData* _dp, RankSignals sg, Signal* self_sg, int rank, int size)
{
    constexpr int pack_size = 16 / sizeof(T);
    constexpr int tnum_gpu  = THREAD_NUM / ngpus;
    using P                 = typename opus::vector_t<T, pack_size>;
    using A                 = typename opus::vector_t<opus::fp32_t, pack_size>;
    __shared__ T tmp_smem[tnum_gpu * ngpus * pack_size];
    int warp_id = threadIdx.x / tnum_gpu;
    int lane_id = threadIdx.x % tnum_gpu;
    int tid     = blockIdx.x * tnum_gpu + lane_id;
    const P* ptrs[ngpus];
    P* tmps[ngpus];
#pragma unroll
    for(int i = 0; i < ngpus; ++i)
    {
        ptrs[i] = (const P*)_dp->ptrs[i];
        tmps[i] = get_tmp_buf<P>(sg.signals[i]);
    }
    start_sync<ngpus>(sg, self_sg, rank);

    int part = size / (pack_size * ngpus);
    for(int idx = tid; idx < part; idx += gridDim.x * tnum_gpu)
    {
        // cross device read by all warp
        P input_reg                                         = ptrs[warp_id][rank * part + idx];
        *(reinterpret_cast<P*>(&tmp_smem[0]) + threadIdx.x) = input_reg;
        __syncthreads();
        // calculate and save in first warp
        if(warp_id == 0)
        {
            A add_reg;
#pragma unroll
            for(int i = 0; i < pack_size; ++i)
            {
                add_reg[i] =
                    upcast_s(tmp_smem[pack_size * threadIdx.x + i]);
            }
#pragma unroll
            for(int i = 1; i < ngpus; ++i)
            {
#pragma unroll
                for(int j = 0; j < pack_size; ++j)
                {
                    add_reg[j] += upcast_s(
                        tmp_smem[i * pack_size * tnum_gpu + pack_size * threadIdx.x + j]);
                }
            }
            P add_rslt;
#pragma unroll
            for(int i = 0; i < pack_size; ++i)
            {
                add_rslt[i] = downcast_s<T>(add_reg[i]);
            }
            *(reinterpret_cast<P*>(&tmp_smem[0]) + lane_id) = add_rslt;
        }
        __syncthreads();

        // cross device store
        P rslt                           = *(reinterpret_cast<P*>(&tmp_smem[0]) + lane_id);
        tmps[warp_id][rank * part + idx] = rslt;
    }
    end_sync<ngpus, true>(sg, self_sg, rank);
}

template <int reduce_range>
DINLINE void smemReduceSum(float* smem_addr)
{
    // a warp executes the same instruction
#pragma unroll
    for(int stride = reduce_range / 2; stride > 32; stride >>= 1)
    {
        if(threadIdx.x < stride)
        {
            smem_addr[threadIdx.x] += smem_addr[threadIdx.x + stride];
        }
        __syncthreads();
    }
    volatile float* v_smem = &smem_addr[0];
    if(threadIdx.x < 32)
    {
        v_smem[threadIdx.x] += v_smem[threadIdx.x + 32];
        v_smem[threadIdx.x] += v_smem[threadIdx.x + 16];
        v_smem[threadIdx.x] += v_smem[threadIdx.x + 8];
        v_smem[threadIdx.x] += v_smem[threadIdx.x + 4];
        v_smem[threadIdx.x] += v_smem[threadIdx.x + 2];
        v_smem[threadIdx.x] += v_smem[threadIdx.x + 1];
    }
    __syncthreads();
}

/*
 * input case n dim should be divided by 4096 with dtype bf16
 * and should be divided by 2048 with dtype fp32
 * */
template <typename T, int tnum, int n_loop>
__global__ void __launch_bounds__(tnum, 1)
    local_device_load_rmsnorm_naive(RankSignals sg,
                                    T* __restrict__ residual_inp,
                                    T* __restrict__ residual_out,
                                    T* __restrict__ results,
                                    T* __restrict__ weight,
                                    float eps,
                                    int rank,
                                    int m,
                                    int n)
{
    constexpr int pack_size = 16 / sizeof(T);
    using P                 = typename opus::vector_t<T, pack_size>;
    using A                 = typename opus::vector_t<opus::fp32_t, pack_size>;
    __shared__ float smem[tnum];
    P* tmps = get_tmp_buf<P>(sg.signals[rank]);

    for(int bid = blockIdx.x; bid < m; bid += gridDim.x)
    {
        float square_sum = 0.0f;
        A rms_inp_f32[n_loop];
        P w_arr[n_loop];
#pragma unroll
        for(int n_iter = 0; n_iter < n_loop; ++n_iter)
        {
            int read_idx        = bid * n_loop * blockDim.x + n_iter * blockDim.x + threadIdx.x;
            P reduce_out_pack   = tmps[read_idx];
            P residual_inp_pack = *(reinterpret_cast<P*>(residual_inp) + read_idx);
            w_arr[n_iter] = *(reinterpret_cast<P*>(weight) + n_iter * blockDim.x + threadIdx.x);
            A reduce_pack;
#pragma unroll
            for(int i = 0; i < pack_size; ++i)
            {
                float res_inp = upcast_s(residual_inp_pack[i]);
                float ar_out  = upcast_s(reduce_out_pack[i]);
                float rms_inp = res_inp + ar_out;
                rms_inp_f32[n_iter][i] = rms_inp;
                reduce_pack[i]         = rms_inp * rms_inp;
            }
            square_sum += packReduce<AddFunctor, float, pack_size>(reduce_pack);
        }
        smem[threadIdx.x] = square_sum;
        __syncthreads();
        smemReduceSum<tnum>(&smem[0]);
        square_sum  = smem[0];
        float denom = rsqrtf(square_sum / n + eps);
#pragma unroll
        for(int n_iter = 0; n_iter < n_loop; ++n_iter)
        {
            P rmsnorm_rslt;
            P rmsnorm_inp;
#pragma unroll
            for(int i = 0; i < pack_size; ++i)
            {
                float x_f32          = rms_inp_f32[n_iter][i];
                float w_f32          = upcast_s(w_arr[n_iter][i]);
                rmsnorm_inp[i]  = downcast_s<T>(x_f32);
                rmsnorm_rslt[i] = downcast_s<T>(x_f32 * w_f32 * denom);
            }
            int write_idx = bid * n_loop * blockDim.x + n_iter * blockDim.x + threadIdx.x;
            *(reinterpret_cast<P*>(results) + write_idx)      = rmsnorm_rslt;
            *(reinterpret_cast<P*>(residual_out) + write_idx) = rmsnorm_inp;
        }
    }
}

/*
 * block size can be 256 and 512
 * corresponding 2048 and 4096 elem per block
 * */
template <typename T, int tnum, int n_loop>
__global__ void __launch_bounds__(tnum, 1) local_device_load_rmsnorm(RankSignals sg,
                                                                     T* __restrict__ residual_inp,
                                                                     T* __restrict__ residual_out,
                                                                     T* __restrict__ results,
                                                                     T* __restrict__ weight,
                                                                     float eps,
                                                                     int rank,
                                                                     int m,
                                                                     int n)
{
    constexpr int pack_size = 16 / sizeof(T);
    using P                 = typename opus::vector_t<T, pack_size>;
    using A                 = typename opus::vector_t<opus::fp32_t, pack_size>;
    __shared__ float smem[tnum];
    P* tmps = get_tmp_buf<P>(sg.signals[rank]);

    for(int bid = blockIdx.x; bid < m; bid += gridDim.x)
    {
        float square_sum = 0.0f;
        A rms_inp_f32[n_loop];
        P w_arr[n_loop];
#pragma unroll
        for(int n_iter = 0; n_iter < n_loop; ++n_iter)
        {
            if(n_iter * tnum + threadIdx.x < (n / pack_size))
            {
                int read_idx        = bid * (n / pack_size) + n_iter * tnum + threadIdx.x;
                P reduce_out_pack   = tmps[read_idx];
                P residual_inp_pack = *(reinterpret_cast<P*>(residual_inp) + read_idx);
                w_arr[n_iter]       = *(reinterpret_cast<P*>(weight) + n_iter * tnum + threadIdx.x);
                A reduce_pack;
#pragma unroll
                for(int i = 0; i < pack_size; ++i)
                {
                    float ar_out  = upcast_s(reduce_out_pack[i]);
                    float res_inp = upcast_s(residual_inp_pack[i]);
                    float rms_inp = ar_out + res_inp;
                    rms_inp_f32[n_iter][i] = rms_inp;
                    reduce_pack[i]         = rms_inp * rms_inp;
                }
                square_sum += packReduce<AddFunctor, float, pack_size>(reduce_pack);
            }
        }
        smem[threadIdx.x] = square_sum;
        __syncthreads();
        smemReduceSum<tnum>(&smem[0]);
        square_sum  = smem[0];
        float denom = rsqrtf(square_sum / n + eps);
#pragma unroll
        for(int n_iter = 0; n_iter < n_loop; ++n_iter)
        {
            if(n_iter * tnum + threadIdx.x < (n / pack_size))
            {
                P rmsnorm_rslt;
                P rmsnorm_inp;
#pragma unroll
                for(int i = 0; i < pack_size; ++i)
                {
                    float x_f32          = rms_inp_f32[n_iter][i];
                    float w_f32          = upcast_s(w_arr[n_iter][i]);
                    rmsnorm_inp[i]  = downcast_s<T>(x_f32);
                    rmsnorm_rslt[i] = downcast_s<T>(x_f32 * w_f32 * denom);
                }
                int write_idx = bid * (n / pack_size) + n_iter * tnum + threadIdx.x;
                *(reinterpret_cast<P*>(results) + write_idx)      = rmsnorm_rslt;
                *(reinterpret_cast<P*>(residual_out) + write_idx) = rmsnorm_inp;
            }
        }
    }
}

template <typename T, int n_loop>
__global__ void __launch_bounds__(256, 1)
    local_device_load_rmsnorm_512n(RankSignals sg,
                                   T* __restrict__ residual_inp,
                                   T* __restrict__ residual_out,
                                   T* __restrict__ results,
                                   T* __restrict__ weight,
                                   float eps,
                                   int rank,
                                   int m,
                                   int n)
{
    constexpr int pack_size = 16 / sizeof(T);
    using P                 = typename opus::vector_t<T, pack_size>;
    using A                 = typename opus::vector_t<opus::fp32_t, pack_size>;
    P* tmps                 = get_tmp_buf<P>(sg.signals[rank]);
    int warp_id             = threadIdx.x / 64;
    int lane_id             = threadIdx.x % 64;
    int warp_num            = blockDim.x / 64;

    for(int bid = blockIdx.x * warp_num + warp_id; bid < m; bid += gridDim.x * warp_num)
    {
        float square_sum = 0.0f;
        A rms_inp_f32[n_loop];
        P w_arr[n_loop];
#pragma unroll
        for(int n_iter = 0; n_iter < n_loop; ++n_iter)
        {
            int read_idx        = bid * 64 * n_loop + n_iter * 64 + lane_id;
            P reduce_out_pack   = tmps[read_idx];
            P residual_inp_pack = *(reinterpret_cast<P*>(residual_inp) + read_idx);
            w_arr[n_iter]       = *(reinterpret_cast<P*>(weight) + n_iter * 64 + lane_id);
            A reduce_pack;
#pragma unroll
            for(int i = 0; i < pack_size; ++i)
            {
                float ar_out  = upcast_s(reduce_out_pack[i]);
                float res_inp = upcast_s(residual_inp_pack[i]);
                float rms_inp = ar_out + res_inp;
                rms_inp_f32[n_iter][i] = rms_inp;
                reduce_pack[i]         = rms_inp * rms_inp;
            }
            float tmp_sum = packReduce<AddFunctor, float, pack_size>(reduce_pack);
            square_sum += tmp_sum;
        }
        square_sum  = warpReduce<AddFunctor, float, 64>(square_sum);
        float denom = rsqrtf(square_sum / n + eps);
#pragma unroll
        for(int n_iter = 0; n_iter < n_loop; ++n_iter)
        {
            P rmsnorm_rslt;
            P rmsnorm_inp;
#pragma unroll
            for(int i = 0; i < pack_size; ++i)
            {
                float x_f32          = rms_inp_f32[n_iter][i];
                float w_f32          = upcast_s(w_arr[n_iter][i]);
                rmsnorm_inp[i]  = downcast_s<T>(x_f32);
                rmsnorm_rslt[i] = downcast_s<T>(x_f32 * w_f32 * denom);
            }
            int write_idx = bid * 64 * n_loop + n_iter * 64 + lane_id;
            *(reinterpret_cast<P*>(results) + write_idx)      = rmsnorm_rslt;
            *(reinterpret_cast<P*>(residual_out) + write_idx) = rmsnorm_inp;
        }
    }
}

template <template <typename> class functor, typename T, int BLOCK_SIZE, int WARP_SIZE>
__device__ __forceinline__ T ar_fusion_epilogue_block_reduce(T val)
{
    static __shared__ T shared[BLOCK_SIZE / WARP_SIZE];
    const int tid = threadIdx.x;
    const int w_tid = tid % WARP_SIZE;
    const int wid = tid / WARP_SIZE;
    val = warpReduce<functor, T, WARP_SIZE>(val);
    if (w_tid == 0) {
        shared[wid] = val;
    }
    __syncthreads();
    val = shared[w_tid];
    __syncthreads();
    val = warpReduce<functor, T, BLOCK_SIZE / WARP_SIZE>(val);
    return val;
}

template <typename P, typename A, typename O, typename OT, int PACK_SIZE, int BLOCK_SIZE, int WARP_SIZE = 32>
__device__ __forceinline__ void ar_fusion_epilogue_rms_norm(O &out, A &in, P &weight, float eps, int hidden_dim)
{
    __shared__ float s_val;
    float acc = 0.f;
#pragma unroll
    for (int i = 0; i < PACK_SIZE; ++i) {
        float v = upcast_s(in[i]);
        acc += v * v;
    }
    acc = ar_fusion_epilogue_block_reduce<AddFunctor, float, BLOCK_SIZE, WARP_SIZE>(acc);
    if (threadIdx.x == 0) {
        s_val = rsqrtf(acc / hidden_dim + eps);
    }
    __syncthreads();
#pragma unroll
    for (int i = 0; i < PACK_SIZE; ++i) {
        float out_ = in[i] * s_val * upcast_s(weight[i]);
        out[i] = downcast_s<OT>(out_);
    }
}

template <typename A, int PACK_SIZE, int BLOCK_SIZE, int WARP_SIZE = 32>
__device__ __forceinline__ float ar_fusion_epilogue_reduce_abs_max(A &data)
{
    __shared__ float s_val;
    auto fn = [](float a, float b) { return a > b ? a : b; };
    float acc = -1.f;
#pragma unroll
    for (int i = 0; i < PACK_SIZE; ++i) {
        float v = upcast_s(data[i]);
        acc = fn(acc, std::abs(v));
    }
    acc = ar_fusion_epilogue_block_reduce<MaxFunctor, float, BLOCK_SIZE, WARP_SIZE>(acc);
    if (threadIdx.x == 0) {
        s_val = acc;
    }
    __syncthreads();
    acc = s_val;
    return acc;
}

template <typename P, typename A, typename T, typename OutT, int PACK_SIZE, int BLOCK_SIZE>
__device__ __forceinline__ void ar_fusion_epilogue(
    A &in,
    P &weight,
    int hidden_dim,
    float eps,
    int idx,
    int tidx,
    OutT* __restrict__ output,
    float* __restrict__ scale_out)
{
    if constexpr (std::is_same_v<T, OutT>) {
        P out;
        ar_fusion_epilogue_rms_norm<P, A, P, T, PACK_SIZE, BLOCK_SIZE>(out, in, weight, eps, hidden_dim);
        *reinterpret_cast<P *>(output + idx) = out;
    } else {
        float FP8_UPBOUND = opus::cast<opus::fp32_t>(opus::numeric_limits<opus::fp8_t>::max());
        using OP = opus::vector_t<OutT, PACK_SIZE>;
        OP out_quant;
        A out;
        ar_fusion_epilogue_rms_norm<P, A, A, float, PACK_SIZE, BLOCK_SIZE>(out, in, weight, eps, hidden_dim);
        float amax = ar_fusion_epilogue_reduce_abs_max<A, PACK_SIZE, BLOCK_SIZE>(out);
        float scale = amax == 0.f ? 1.f : amax / FP8_UPBOUND;
        out_quant = packQuant<opus::fp32_t, PACK_SIZE>(out, scale);
        *reinterpret_cast<OP *>(output + idx) = out_quant;
        if (threadIdx.x == 0)
            scale_out[tidx] = scale;
    }
}

template <typename T, typename OutT, int ngpus, int BLOCK_SIZE>
__global__ void __launch_bounds__(BLOCK_SIZE, 1) allreduce_fusion_kernel_1stage(
    RankData* _dp,
    RankSignals sg,
    Signal* self_sg,
    int rank,
    T* __restrict__ residual_inp,
    T* __restrict__ residual_out,
    OutT* __restrict__ output,
    T* __restrict__ weight,
    float* __restrict__ scale_out,
    int size,
    int hidden_dim,
    float eps)
{
    constexpr int pack_size = 16 / sizeof(T);
    constexpr int tnum_gpu  = BLOCK_SIZE / ngpus;
    using P                 = typename opus::vector_t<T, pack_size>;
    using A                 = typename opus::vector_t<opus::fp32_t, pack_size>;
    int tidx = blockIdx.x;
    int access_id_in_token = threadIdx.x * pack_size;
    int idx = tidx * hidden_dim + access_id_in_token;
    const P* ptrs[ngpus];
    P* tmps[ngpus];
#pragma unroll
    for(int i = 0; i < ngpus; ++i)
    {
        ptrs[i] = (const P*)_dp->ptrs[i];
        tmps[i] = get_tmp_buf<P>(sg.signals[i]);
    }
    start_sync<ngpus>(sg, self_sg, rank);

    A acc;
    P vec = ptrs[0][idx / pack_size];
#pragma unroll
    for (int v = 0; v < pack_size; ++v) {
        acc[v] = upcast_s(vec[v]);
    }

#pragma unroll
    for (int r = 1; r < ngpus; ++r) {
        vec = ptrs[r][idx / pack_size];
#pragma unroll
        for (int v = 0; v < pack_size; ++v) {
            acc[v] += upcast_s(vec[v]);
        }
    }

    P res = *reinterpret_cast<P *>(residual_inp + idx);

#pragma unroll
    for (int v = 0; v < pack_size; ++v) {
        acc[v] += upcast_s(res[v]);
    }

#pragma unroll
    for (int v = 0; v < pack_size; ++v) {
        vec[v] = downcast_s<T>(acc[v]);
    }

    *reinterpret_cast<P *>(residual_out + idx) = vec;
    P weight_p = *reinterpret_cast<P *>(weight + access_id_in_token);
    ar_fusion_epilogue<P, A, T, OutT, pack_size, BLOCK_SIZE>(
        acc, weight_p, hidden_dim, eps, idx, tidx, output, scale_out);
}

template <typename T, typename OutT, int NGPUS, int HIDDEN_DIM>
void allreduce_fusion_kernel_1stage_launcher(
    RankData* _dp,
    RankSignals sg,
    Signal* self_sg,
    int rank,
    T* residual_inp,
    T* residual_out,
    OutT* output,
    T* weight,
    float *scale_out,
    int size,
    float eps,
    hipStream_t stream)
{
    constexpr int PACK_SIZE = 16 / sizeof(T);
    constexpr int BLOCK_SIZE = HIDDEN_DIM / PACK_SIZE;
    int token_num = size / HIDDEN_DIM;
    if(token_num > kMaxBlocks)
        throw std::runtime_error("Token number is too large for allreduce_fusion_kernel_1stage kernel");
    dim3 threadsPerBlock(BLOCK_SIZE);
    dim3 numBlocks(token_num);
    allreduce_fusion_kernel_1stage<T, OutT, NGPUS, BLOCK_SIZE><<<numBlocks, threadsPerBlock, 0, stream>>>(
        _dp, sg, self_sg, rank, residual_inp, residual_out, output, weight, scale_out, size, HIDDEN_DIM, eps);
}

template <typename T, typename OutT, int ngpus, int BLOCK_SIZE>
__global__ void __launch_bounds__(BLOCK_SIZE, 1)
    allreduce_fusion_kernel_2stage(RankData* _dp,
                                   RankSignals sg,
                                   Signal* self_sg,
                                   int rank,
                                   T* __restrict__ residual_inp,
                                   T* __restrict__ residual_out,
                                   OutT* __restrict__ output,
                                   T* __restrict__ weight,
                                   float* __restrict__ scale_out,
                                   int size,
                                   int hidden_dim,
                                   float eps)
{
    constexpr int pack_size = 16 / sizeof(T);
    constexpr int tnum_gpu  = BLOCK_SIZE / ngpus;
    using P                 = typename opus::vector_t<T, pack_size>;
    using OP                = opus::vector_t<OutT, 16 / sizeof(T)>;
    using A                 = typename opus::vector_t<opus::fp32_t, pack_size>;
    __shared__ P tmp_smem[tnum_gpu * ngpus];
    int warp_id = threadIdx.x / tnum_gpu;
    int lane_id = threadIdx.x % tnum_gpu;
    const P* ptrs[ngpus];
    P* tmps[ngpus];
#pragma unroll
    for(int i = 0; i < ngpus; ++i)
    {
        ptrs[i] = (const P*)_dp->ptrs[i];
        tmps[i] = get_tmp_buf<P>(sg.signals[i]);
    }
    A acc;
    start_sync<ngpus>(sg, self_sg, rank);

    for(int idx = ((blockIdx.x * ngpus + rank) * tnum_gpu + lane_id) * pack_size; idx < size;
        idx += gridDim.x * ngpus * tnum_gpu * pack_size)
    {
        P vec                 = ptrs[warp_id][idx / pack_size];
        tmp_smem[threadIdx.x] = vec;
        __syncthreads();
        if(warp_id == 0)
        {
#pragma unroll
            for(int v = 0; v < pack_size; ++v)
            {
                acc[v] = upcast_s(vec[v]);
            }
#pragma unroll
            for(int r = 1; r < ngpus; ++r)
            {
                vec = tmp_smem[r * tnum_gpu + lane_id];
#pragma unroll
                for(int v = 0; v < pack_size; ++v)
                {
                    acc[v] += upcast_s(vec[v]);
                }
            }
#pragma unroll
            for(int v = 0; v < pack_size; ++v)
            {
                vec[v] = downcast_s<T>(acc[v]);
            }
            tmp_smem[lane_id] = vec;
        }
        __syncthreads();
        vec                            = tmp_smem[lane_id];
        tmps[warp_id][idx / pack_size] = vec;
    }

    int access_id_in_token = threadIdx.x * pack_size;
    P weight_p             = *reinterpret_cast<P*>(weight + access_id_in_token);
    end_sync<ngpus>(sg, self_sg, rank);
    for(int idx = blockIdx.x * hidden_dim + access_id_in_token, tidx = blockIdx.x; idx < size;
        idx += gridDim.x * hidden_dim, tidx += gridDim.x)
    {
        P vec = tmps[rank][idx / pack_size];
        P res = *reinterpret_cast<P*>(residual_inp + idx);
#pragma unroll
        for(int v = 0; v < pack_size; ++v)
        {
            vec[v] += res[v];
        }
        *reinterpret_cast<P*>(residual_out + idx) = vec;
#pragma unroll
        for(int v = 0; v < pack_size; ++v)
        {
            acc[v] = upcast_s(vec[v]);
        }
        ar_fusion_epilogue<P, A, T, OutT, pack_size, BLOCK_SIZE>(
            acc, weight_p, hidden_dim, eps, idx, tidx, output, scale_out);
    }
}

template <typename T, typename OutT, int NGPUS, int HIDDEN_DIM>
void allreduce_fusion_kernel_2stage_launcher(RankData* _dp,
                                             RankSignals sg,
                                             Signal* self_sg,
                                             int rank,
                                             T* residual_inp,
                                             T* residual_out,
                                             OutT* output,
                                             T* weight,
                                             float* scale_out,
                                             int size,
                                             float eps,
                                             hipStream_t stream)
{
    constexpr int PACK_SIZE  = 16 / sizeof(T);
    constexpr int BLOCK_SIZE = HIDDEN_DIM / PACK_SIZE;
    int token_num            = size / HIDDEN_DIM;
    dim3 threadsPerBlock(BLOCK_SIZE);
    token_num = std::min(token_num, kMaxBlocks);
    dim3 numBlocks(token_num);
    allreduce_fusion_kernel_2stage<T, OutT, NGPUS, BLOCK_SIZE>
        <<<numBlocks, threadsPerBlock, 0, stream>>>(_dp,
                                                    sg,
                                                    self_sg,
                                                    rank,
                                                    residual_inp,
                                                    residual_out,
                                                    output,
                                                    weight,
                                                    scale_out,
                                                    size,
                                                    HIDDEN_DIM,
                                                    eps);
}

template <typename T, typename OutT, int BLOCK_SIZE>
__global__ void __launch_bounds__(BLOCK_SIZE, 1)
    local_device_load_rmsnorm_quant_naive(RankSignals sg,
                                          int rank,
                                          T* __restrict__ residual_inp,
                                          T* __restrict__ residual_out,
                                          OutT* __restrict__ output,
                                          T* __restrict__ weight,
                                          float* __restrict__ scale_out,
                                          int size,
                                          int hidden_dim,
                                          float eps)
{
    constexpr int pack_size = 16 / sizeof(T);
    using P                 = typename opus::vector_t<T, pack_size>;
    using A                 = typename opus::vector_t<opus::fp32_t, pack_size>;
    P* tmps                 = get_tmp_buf<P>(sg.signals[rank]);
    int access_id_in_token  = threadIdx.x * pack_size;
    P weight_p              = *reinterpret_cast<P*>(weight + access_id_in_token);
    int idx                 = blockIdx.x * hidden_dim + access_id_in_token;
    int tidx                = blockIdx.x;
    {
        A acc;
        P vec = tmps[idx / pack_size];
        P res = *reinterpret_cast<P*>(residual_inp + idx);
#pragma unroll
        for(int v = 0; v < pack_size; ++v)
        {
            vec[v] += res[v];
        }
        *reinterpret_cast<P*>(residual_out + idx) = vec;
#pragma unroll
        for(int v = 0; v < pack_size; ++v)
        {
            acc[v] = upcast_s(vec[v]);
        }
        ar_fusion_epilogue<P, A, T, OutT, pack_size, BLOCK_SIZE>(
            acc, weight_p, hidden_dim, eps, idx, tidx, output, scale_out);
    }
}

template <typename T, typename OutT, int NGPUS, int HIDDEN_DIM>
void allreduce_fusion_kernel_split_launcher(RankData* _dp,
                                            RankSignals sg,
                                            Signal* self_sg,
                                            int rank,
                                            T* residual_inp,
                                            T* residual_out,
                                            OutT* output,
                                            T* weight,
                                            float* scale_out,
                                            int size,
                                            float eps,
                                            hipStream_t stream)
{
    // step 1, run reduce-scatter + allgather cross device save
    dim3 block(512);
    int block_num = ((size / NGPUS) + 512 - 1) / 512;
    dim3 grid(std::min(block_num, 80));
    switch(NGPUS)
    {
    case 8:
        reduce_scatter_cross_device_store<T, 8>
            <<<grid, block, 0, stream>>>(_dp, sg, self_sg, rank, size);
        break;
    case 4:
        reduce_scatter_cross_device_store<T, 4>
            <<<grid, block, 0, stream>>>(_dp, sg, self_sg, rank, size);
        break;
    case 2:
        reduce_scatter_cross_device_store<T, 2>
            <<<grid, block, 0, stream>>>(_dp, sg, self_sg, rank, size);
        break;
    default: printf("fused allreduce rmsnorm world size error\n");
    }
    // step 2, run allgather local device load + rmsnorm + quant
    constexpr int PACK_SIZE  = 16 / sizeof(T);
    constexpr int BLOCK_SIZE = HIDDEN_DIM / PACK_SIZE;
    int nblocks              = size / HIDDEN_DIM;
    dim3 threadsPerBlock(BLOCK_SIZE);
    dim3 numBlocks(nblocks);
    local_device_load_rmsnorm_quant_naive<T, OutT, BLOCK_SIZE>
        <<<numBlocks, threadsPerBlock, 0, stream>>>(
            sg, rank, residual_inp, residual_out, output, weight, scale_out, size, HIDDEN_DIM, eps);
}

using IPC_KEY = std::array<uint8_t, sizeof(hipIpcMemHandle_t)>;
static_assert(sizeof(IPC_KEY) == sizeof(hipIpcMemHandle_t));
static_assert(alignof(IPC_KEY) == alignof(hipIpcMemHandle_t));

class CustomAllreduce
{
    public:
    int rank_;
    int world_size_;
    bool full_nvlink_;

    // below are device pointers
    RankSignals sg_;
    std::unordered_map<void *, RankData *> input_buffer;
    std::unordered_map<void *, RankData *> output_buffers_;
    Signal *self_sg_;

    // stores the registered device pointers from all ranks
    RankData *d_rank_data_base_, *d_rank_data_end_;
    std::vector<void *> graph_unreg_input_buffers_;
    std::vector<void *> graph_unreg_output_buffers_;
    // a map from IPC handles to opened IPC pointers
    std::map<IPC_KEY, char*> ipc_handles_;

    /**
     * meta is a pointer to device metadata and temporary buffer for allreduce.
     *
     * There's a total of sizeof(Signal) of prefix before the actual data,
     * so meta + 1 points to actual temporary buffer.
     *
     * note: this class does not own any device memory. Any required buffers
     * are passed in from the constructor
     */
    CustomAllreduce(Signal* meta,
                    void* rank_data,
                    size_t rank_data_sz,
                    const hipIpcMemHandle_t* handles,
                    const std::vector<int64_t>& offsets,
                    int rank,
                    bool fully_connected = true)
        : rank_(rank),
          world_size_(offsets.size()),
          full_nvlink_(fully_connected),
          self_sg_(meta),
          d_rank_data_base_(reinterpret_cast<RankData*>(rank_data)),
          d_rank_data_end_(d_rank_data_base_ + rank_data_sz / sizeof(RankData))
    {
        for(int i = 0; i < world_size_; i++)
        {
            Signal* rank_sg;
            if(i != rank_)
            {
                char* handle = open_ipc_handle(&handles[i]);
                handle += offsets[i];
                rank_sg = (Signal*)handle;
            }
            else
            {
                rank_sg = self_sg_;
            }
            sg_.signals[i] = rank_sg;
        }
    }

    char* open_ipc_handle(const void* ipc_handle)
    {
        auto [it, new_handle] = ipc_handles_.insert({*((IPC_KEY*)ipc_handle), nullptr});
        if(new_handle)
        {
            char* ipc_ptr;
            HIP_CALL(hipIpcOpenMemHandle((void**)&ipc_ptr,
                                         *((const hipIpcMemHandle_t*)ipc_handle),
                                         hipIpcMemLazyEnablePeerAccess));
            it->second = ipc_ptr;
        }
        return it->second;
    }

    std::pair<std::vector<uint8_t>, std::vector<int64_t>> get_graph_buffer_ipc_meta()
    {
        auto num_input_buffers = graph_unreg_input_buffers_.size();
        auto num_output_buffers = graph_unreg_output_buffers_.size();
        auto num_buffers = num_input_buffers + num_output_buffers;
        auto handle_sz   = sizeof(hipIpcMemHandle_t);
        std::vector<uint8_t> handles(handle_sz * num_buffers, 0);
        std::vector<int64_t> offsets(num_buffers);
        for(int i = 0; i < num_input_buffers; i++)
        {
            auto ptr = graph_unreg_input_buffers_[i];
            void* base_ptr;
            // note: must share the base address of each allocation, or we get wrong
            // address
            if(hipPointerGetAttribute(&base_ptr,
#ifdef USE_ROCM
                                      HIP_POINTER_ATTRIBUTE_RANGE_START_ADDR,
#else
                                      CU_POINTER_ATTRIBUTE_RANGE_START_ADDR,
#endif
                                  (hipDeviceptr_t)ptr) != CUDA_SUCCESS)
                throw std::runtime_error("failed to get pointer attr");
            HIP_CALL(hipIpcGetMemHandle(
                (hipIpcMemHandle_t *)&handles[i * handle_sz], base_ptr));
            offsets[i] = ((char *)ptr) - ((char *)base_ptr);
        }
      
        // Process output buffers
        for (int i = 0; i < num_output_buffers; i++)
        {
            auto ptr = graph_unreg_output_buffers_[i];
            void *base_ptr;
            if (hipPointerGetAttribute(&base_ptr,
#ifdef USE_ROCM
                                      HIP_POINTER_ATTRIBUTE_RANGE_START_ADDR,
#else
                                      CU_POINTER_ATTRIBUTE_RANGE_START_ADDR,
#endif
                                  (hipDeviceptr_t)ptr) != CUDA_SUCCESS)
                throw std::runtime_error("failed to get pointer attr for output");
            HIP_CALL(hipIpcGetMemHandle(
                (hipIpcMemHandle_t *)&handles[(num_input_buffers + i) * handle_sz], base_ptr));
            offsets[num_input_buffers + i] = ((char *)ptr) - ((char *)base_ptr);
        }
      
      return std::make_pair(handles, offsets);
    }

    void check_rank_data_capacity(size_t num = 1)
    {
      if (d_rank_data_base_ + num > d_rank_data_end_)
        throw std::runtime_error(
            "Rank data buffer is overflowed by " +
            std::to_string(d_rank_data_base_ + num - d_rank_data_end_));
    }

    void register_input_buffer(const std::vector<torch::Tensor> &handles,
                         const std::vector<int64_t> &offsets, void *self)
    {
      check_rank_data_capacity();
      RankData data;
      for (int i = 0; i < world_size_; i++)
      {
        if (i != rank_)
        {
          hipIpcMemHandle_t* ipc_handle_ptr = (hipIpcMemHandle_t*)handles[i].data_ptr();
          char *handle = open_ipc_handle((void*)ipc_handle_ptr);
          handle += offsets[i];
          data.ptrs[i] = handle;
        }
        else
        {
          data.ptrs[i] = self;
        }
      }
      auto d_data = d_rank_data_base_++;
      HIP_CALL(
          hipMemcpy(d_data, &data, sizeof(RankData), hipMemcpyHostToDevice));
      input_buffer[self] = d_data;
    }

    void register_output_buffer(const std::vector<torch::Tensor> &handles,
                                const std::vector<int64_t> &offsets, void *self)
    {
      check_rank_data_capacity();
      RankData data;
      // Setup output_ptrs
      for (int i = 0; i < world_size_; i++)
      {
        if (i != rank_)
        {
          hipIpcMemHandle_t* ipc_handle_ptr = (hipIpcMemHandle_t*)handles[i].data_ptr();
          char *handle = open_ipc_handle((void*)ipc_handle_ptr);
          handle += offsets[i];
          data.ptrs[i] = handle;
        }
        else
        {
          data.ptrs[i] = self;
        }
      }
      auto d_data = d_rank_data_base_++;
      HIP_CALL(
          hipMemcpy(d_data, &data, sizeof(RankData), hipMemcpyHostToDevice));
      output_buffers_[self] = d_data;
    }

    RankData *get_buffer_RD(hipStream_t stream, void *input)
    {
      RankData *ptrs;
      auto it = input_buffer.find(input);
      if (it != input_buffer.end())
      {
        ptrs = it->second;
      }
      else
      {
        hipStreamCaptureStatus status;
        HIP_CALL(hipStreamIsCapturing(stream, &status));
        if (status == hipStreamCaptureStatusActive)
        {
          ptrs = d_rank_data_base_ + graph_unreg_input_buffers_.size();
          graph_unreg_input_buffers_.push_back(input);
        }
        else
        {
          throw std::runtime_error(
              "buffer address " +
              std::to_string(reinterpret_cast<uint64_t>(input)) +
              " is not registered!");
        }
      }

      return ptrs;
    }

    RankData *get_output_buffer_RD(hipStream_t stream, void *output)
    {
      RankData *ptrs;
      auto it = output_buffers_.find(output);
      if (it != output_buffers_.end())
      {
        ptrs = it->second;
      }
      else
      {
        hipStreamCaptureStatus status;
        HIP_CALL(hipStreamIsCapturing(stream, &status));
        if (status == hipStreamCaptureStatusActive)
        {
          // For graph mode, collect output addresses
          ptrs = d_rank_data_base_ + graph_unreg_input_buffers_.size() + graph_unreg_output_buffers_.size();
          graph_unreg_output_buffers_.push_back(output);
        }
        else
        {
          throw std::runtime_error(
              "output buffer address " +
              std::to_string(reinterpret_cast<uint64_t>(output)) +
              " is not registered!");
        }
      }

      return ptrs;
    }

    // note: when registering graph buffers, we intentionally choose to not
    // deduplicate the addresses. That means if the allocator reuses some
    // addresses, they will be registered again. This is to account for the remote
    // possibility of different allocation patterns between ranks. For example,
    // rank 1 may get the same input address for the second allreduce, but rank 2
    // got a different address. IPC handles have internal reference counting
    // mechanism so overhead should be small.
    void register_graph_buffers(
        const std::vector<torch::Tensor> &handles,
        const std::vector<torch::Tensor> &offsets)
    {
      auto num_input_buffers = graph_unreg_input_buffers_.size();
      auto num_output_buffers = graph_unreg_output_buffers_.size();
      auto total_buffers = num_input_buffers + num_output_buffers;
      check_rank_data_capacity(total_buffers);
      std::vector<RankData> rank_data(total_buffers);
      
      // Register input buffers
      for (int i = 0; i < num_input_buffers; i++)
      {
        auto self_ptr = graph_unreg_input_buffers_[i];
        auto &rd = rank_data[i];
        for (int j = 0; j < world_size_; j++)
        {
            if (j != rank_)
            {
                hipIpcMemHandle_t* ipc_handle_ptr = (hipIpcMemHandle_t*)handles[j].data_ptr() + i;
                char *handle = open_ipc_handle(ipc_handle_ptr);
                handle += *((int64_t*)offsets[j].data_ptr() + i);
                rd.ptrs[j] = handle;
            }
            else
            {
                rd.ptrs[j] = self_ptr;
            }
        }
      }
      // Register output buffers
      for (int i = 0; i < num_output_buffers; i++)
      {
        auto self_ptr = graph_unreg_output_buffers_[i];
        auto &rd = rank_data[num_input_buffers + i];
        for (int j = 0; j < world_size_; j++)
        {
          if (j != rank_)
          {
            hipIpcMemHandle_t* ipc_handle_ptr = (hipIpcMemHandle_t*)handles[j].data_ptr() + num_input_buffers + i;
            char *handle = open_ipc_handle(ipc_handle_ptr);
            handle += *((int64_t*)offsets[j].data_ptr() + num_input_buffers + i);
            rd.ptrs[j] = handle;
          }
          else
          {
            rd.ptrs[j] = self_ptr;
          }
        }
        output_buffers_[self_ptr] = d_rank_data_base_ + num_input_buffers + i;
      }
      
      HIP_CALL(hipMemcpy(d_rank_data_base_, rank_data.data(),
                           sizeof(RankData) * total_buffers,
                           hipMemcpyHostToDevice));
      d_rank_data_base_ += total_buffers;
      graph_unreg_input_buffers_.clear();
      graph_unreg_output_buffers_.clear();
    }

    /*
     * call all reduce fp8 kernel
     * case size in single gpu: (128, 8192)
     * support 8 gpu only
     * should make ngpus as template param
     * should quant scale match hidden_dim when hidden_dim less than 128?
     * */
    template <typename T>
    void runFp8QuantKernel(hipStream_t stream, T* input, T* output, int size)
    {
        RankData* ptrs = get_buffer_RD(stream, input);
        // 32 block 512 thread or 64 block 256 thread
#define DISPATHC_UNIT(pack_size, quant_scale, ngpus)                                \
    do                                                                              \
    {                                                                               \
    case ngpus: {                                                                   \
        allReduceQuantFp8<T, quant_scale, pack_size, ngpus>                         \
            <<<grid, block, 0, stream>>>(ptrs, sg_, self_sg_, output, rank_, size); \
        return;                                                                     \
    }                                                                               \
    } while(0)

#define DISPATCH_CALL(pack_size, block_size, quant_scale)                                     \
    do                                                                                        \
    {                                                                                         \
        block.x = block_size;                                                                 \
        grid.x  = min((16384 / block_size), (single_device_size / (pack_size * block_size))); \
        size /= pack_size;                                                                    \
        switch(world_size_)                                                                   \
        {                                                                                     \
            DISPATHC_UNIT(pack_size, quant_scale, 2);                                         \
            DISPATHC_UNIT(pack_size, quant_scale, 4);                                         \
            DISPATHC_UNIT(pack_size, quant_scale, 6);                                         \
            DISPATHC_UNIT(pack_size, quant_scale, 8);                                         \
        }                                                                                     \
    } while(0)

        int single_device_size          = size / world_size_;
        constexpr int max_thread_num    = 512;
        constexpr int max_pack_size     = 8;
        constexpr int max_elem_perblock = max_thread_num * max_pack_size;
        dim3 grid, block;
        if(single_device_size % 128 == 0)
        {
            DISPATCH_CALL(8, 256, 128);
        }
        else if(single_device_size % 64 == 0)
        {
            DISPATCH_CALL(8, 256, 64);
        }
        else if(single_device_size % 32 == 0)
        {
            DISPATCH_CALL(8, 256, 32);
        }
        else if(single_device_size % 16 == 0)
        {
            DISPATCH_CALL(8, 256, 16);
        }
        else // 512
        {
            DISPATCH_CALL(8, 256, 8);
        }
    }

    /**
     * This is the result after careful grid search. Using 36 blocks give the best
     * or close to the best runtime on the devices I tried: A100, A10, A30, T4,
     * V100. You'll notice that NCCL kernels also only take a small amount of SMs.
     * Not quite sure the underlying reason, but my guess is that too many SMs
     * will cause contention on NVLink bus.
     */
    template <typename T>
    void allreduce(hipStream_t stream,
                   T* input,
                   T* output,
                   int size,
                   bool use_new = true,
                   bool is_broadcast_reg_outptr = false,
#ifndef USE_ROCM
                   int threads     = 512,
                   int block_limit = 20){
#else
                   int threads     = 512,
                   int block_limit = 16)
    {
#endif
        auto d = 16 / sizeof(T);
    if(size % d != 0)
        throw std::runtime_error("custom allreduce currently requires input length to be multiple "
                                 "of " +
                                 std::to_string(d));
    if(block_limit > kMaxBlocks)
        throw std::runtime_error("max supported block limit is " + std::to_string(kMaxBlocks) +
                                 ". Got " + std::to_string(block_limit));

    RankData *input_ptrs = get_buffer_RD(stream, input);
    RankData *output_ptrs = nullptr;
    if (is_broadcast_reg_outptr)
    {
        output_ptrs = get_output_buffer_RD(stream, output);
    }

    auto bytes = size * sizeof(T);
    size /= d;

    // use new version of allreduce kernel
    if(use_new)
    {
        hipDevice_t dev;
        hipDeviceProp_t dev_prop;
        hipGetDevice(&dev);
        hipGetDeviceProperties(&dev_prop, dev);
        std::string arch = dev_prop.gcnArchName;
        bool use_write_mode = false;

        int blocks       = 16;
        bool call_1stage = false;
        bool call_2stage = false;
        if(world_size_ == 2)
        {
            call_1stage = true;
        }
        else if(full_nvlink_)
        {
            if((world_size_ <= 4 && bytes < 160 * 1024) || (world_size_ <= 8 && bytes < 80 * 1024))
            {
                call_1stage = true;
            }
            else
            {
                call_2stage = true;
            }
        }
        if(call_1stage)
        {
            blocks = std::min(kMaxBlocks,
                              (size + (threads / world_size_) - 1) / (threads / world_size_));
        }
        else if(call_2stage)
        {
            blocks = std::min(kMaxBlocks,   
                              (size / world_size_ + (threads / world_size_) - 1) /
                                  (threads / world_size_));
            if (world_size_ == 8 && bytes > 512 * 4096 * 2 && arch.find("gfx942") != std::string::npos) {
                use_write_mode = true;
            }
        }

#define KL(ngpus, name) \
    do { \
        if(is_broadcast_reg_outptr) { \
            name<T, ngpus, true><<<blocks, threads, 0, stream>>>(input_ptrs, output_ptrs, sg_, self_sg_, output, rank_, size); \
        } else { \
            name<T, ngpus, false><<<blocks, threads, 0, stream>>>(input_ptrs, output_ptrs, sg_, self_sg_, output, rank_, size); \
        } \
    } while(0)

#define dispatch(ngpus, name)                             \
    do                                                    \
    {                                                     \
        if(bytes % (ngpus * 16) == 0 && world_size_ != 6) \
        {                                                 \
            if (use_write_mode) {                         \
                KL(ngpus, name##_write_mode);             \
            }                                             \
            else {                                        \
                KL(ngpus, name);                          \
            }                                             \
        }                                                 \
        else                                              \
        {                                                 \
            KL(ngpus, name##_naive);                      \
        }                                                 \
    } while(0)

#define REDUCE_CASE(ngpus)                               \
    case ngpus: {                                        \
        if(call_1stage)                                  \
        {                                                \
            KL(ngpus, cross_device_reduce_1stage);       \
        }                                                \
        else if(call_2stage)                             \
        {                                                \
            dispatch(ngpus, cross_device_reduce_2stage); \
        }                                                \
        break;                                           \
    }

        switch(world_size_)
        {
            REDUCE_CASE(2)
            REDUCE_CASE(4)
            REDUCE_CASE(6)
            REDUCE_CASE(8)
        default:
            throw std::runtime_error(
                "custom allreduce only supports num gpus in (2,4,6,8). Actual num "
                "gpus = " +
                std::to_string(world_size_));
        }
    }
    else // use vllm allreduce kernel
    {
        int blocks = std::min(block_limit, (size + threads - 1) / threads);
#define VLLM_REDUCE_CASE(ngpus)                              \
    case ngpus: {                                            \
        if(world_size_ == 2)                                 \
        {                                                    \
            KL(ngpus, cross_device_reduce_1stage);           \
        }                                                    \
        else if(full_nvlink_)                                \
        {                                                    \
            if((world_size_ <= 4 && bytes < 512 * 1024) ||   \
               (world_size_ <= 8 && bytes < 256 * 1024))     \
            {                                                \
                KL(ngpus, cross_device_reduce_1stage_naive); \
            }                                                \
            else                                             \
            {                                                \
                KL(ngpus, cross_device_reduce_2stage_naive); \
            }                                                \
        }                                                    \
        break;                                               \
    }

        switch(world_size_)
        {
            VLLM_REDUCE_CASE(2)
            VLLM_REDUCE_CASE(4)
            VLLM_REDUCE_CASE(6)
            VLLM_REDUCE_CASE(8)
        default:
            throw std::runtime_error(
                "custom allreduce only supports num gpus in (2,4,6,8). Actual num "
                "gpus = " +
                std::to_string(world_size_));
        }
    }
#undef REDUCE_CASE
#undef KL
}

template <typename T>
void dispatchReduceScatter(hipStream_t stream, T* input, T* output, int size)
{
    RankData* ptrs = get_buffer_RD(stream, input);
    auto d         = 16 / sizeof(T);
    int range      = size / (world_size_ * d);
    dim3 block(512);
    int block_num = (range + 511) / 512;
    dim3 grid(std::min(16, block_num));
    switch(world_size_)
    {
    case 8:
        reduce_scatter_first_dim<T, 8>
            <<<grid, block, 0, stream>>>(ptrs, sg_, self_sg_, output, rank_, range);
        break;
    case 4:
        reduce_scatter_first_dim<T, 4>
            <<<grid, block, 0, stream>>>(ptrs, sg_, self_sg_, output, rank_, range);
        break;
    case 2:
        reduce_scatter_first_dim<T, 2>
            <<<grid, block, 0, stream>>>(ptrs, sg_, self_sg_, output, rank_, range);
        break;
    default: printf("reduce_scatter world_size error!\n");
    }
}

template <typename T>
void dispatchAllGather(hipStream_t stream, T* input, T* output, int size)
{
    RankData* ptrs = get_buffer_RD(stream, input);
    auto d         = 16 / sizeof(T);
    dim3 block(512);
    if(size % d != 0)
    {
        int block_num = (size + 512 - 1) / 512;
        dim3 grid(std::min(block_num, 80));
        switch(world_size_)
        {
        case 8:
            allgather_naive<T, 8>
                <<<grid, block, 0, stream>>>(ptrs, sg_, self_sg_, output, rank_, size);
            break;
        case 4:
            allgather_naive<T, 4>
                <<<grid, block, 0, stream>>>(ptrs, sg_, self_sg_, output, rank_, size);
            break;
        case 2:
            allgather_naive<T, 2>
                <<<grid, block, 0, stream>>>(ptrs, sg_, self_sg_, output, rank_, size);
            break;
        default: printf("allgather world_size error\n");
        }
    }
    else
    {
        size /= d;
        int tnum_per_block = 512 / world_size_;
        int block_num      = (size + tnum_per_block - 1) / tnum_per_block;
        dim3 grid(std::min(block_num, 80));
        switch(world_size_)
        {
        case 8:
            allgather_vec<T, 8>
                <<<grid, block, 0, stream>>>(ptrs, sg_, self_sg_, output, rank_, size);
            break;
        case 4:
            allgather_vec<T, 4>
                <<<grid, block, 0, stream>>>(ptrs, sg_, self_sg_, output, rank_, size);
            break;
        case 2:
            allgather_vec<T, 2>
                <<<grid, block, 0, stream>>>(ptrs, sg_, self_sg_, output, rank_, size);
            break;
        default: printf("allgather world_size error\n");
        }
    }
}

template <typename T>
void dispatchFusedAllReduceRMSNorm(hipStream_t stream,
                                   T* input,
                                   T* residual_inp,
                                   T* residual_out,
                                   T* output,
                                   T* weight,
                                   float eps,
                                   int m,
                                   int n,
                                   bool use_1stage)
{
    auto d   = 16 / sizeof(T);
    int size = m * n;
    if(size % d != 0)
    {
        throw std::runtime_error("custom allreduce currently requires input length to be multiple "
                                 "of " +
                                 std::to_string(d));
    }
    RankData* ptrs = get_buffer_RD(stream, input);
    hipDevice_t dev;
    hipDeviceProp_t dev_prop;
    hipGetDevice(&dev);
    hipGetDeviceProperties(&dev_prop, dev);
    uint32_t num_cu = dev_prop.multiProcessorCount;

    use_1stage = (use_1stage && (n == 4096 || n == 2048 || n == 1024 || n == 512));
#define DISPATCH_1S_KERNEL(NGPUS, N)                                          \
    case N: {                                                                 \
        allreduce_fusion_kernel_1stage_launcher<T, T, NGPUS, N>(ptrs,         \
                                                                sg_,          \
                                                                self_sg_,     \
                                                                rank_,        \
                                                                residual_inp, \
                                                                residual_out, \
                                                                output,       \
                                                                weight,       \
                                                                nullptr,      \
                                                                size,         \
                                                                eps,          \
                                                                stream);      \
        return;                                                               \
    }
#define MAYBE_DISPATCH_1S_KERNEL(NGPUS)                                  \
    if(use_1stage)                                                       \
    {                                                                    \
        switch(n)                                                        \
        {                                                                \
            DISPATCH_1S_KERNEL(NGPUS, 4096)                              \
            DISPATCH_1S_KERNEL(NGPUS, 2048)                              \
            DISPATCH_1S_KERNEL(NGPUS, 1024)                              \
            DISPATCH_1S_KERNEL(NGPUS, 512)                               \
        default: printf("fused 1stage allreduce rmsnorm N-dim error\n"); \
        }                                                                \
    }

    // step 1, run reduce-scatter + allgather cross device save
    dim3 block(512);
    int block_num = ((size / world_size_) + 512 - 1) / 512;
    dim3 grid(std::min(block_num, 80));
    switch(world_size_)
    {
    case 8:
        MAYBE_DISPATCH_1S_KERNEL(8);
        reduce_scatter_cross_device_store<T, 8>
            <<<grid, block, 0, stream>>>(ptrs, sg_, self_sg_, rank_, size);
        break;
    case 4:
        MAYBE_DISPATCH_1S_KERNEL(4);
        reduce_scatter_cross_device_store<T, 4>
            <<<grid, block, 0, stream>>>(ptrs, sg_, self_sg_, rank_, size);
        break;
    case 2:
        MAYBE_DISPATCH_1S_KERNEL(2);
        reduce_scatter_cross_device_store<T, 2>
            <<<grid, block, 0, stream>>>(ptrs, sg_, self_sg_, rank_, size);
        break;
    default: printf("fused allreduce rmsnorm world size error\n");
    }

#undef MAYBE_DISPATCH_1S_KERNEL

    // step 2, run allgather local device load + rmsnorm
    int n_bytes  = n * sizeof(T);
    auto setGrid = [&](int naive_grid_size, const void* kernel_ptr) {
        int occupancy;
        hipOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy, kernel_ptr, block.x, 0);
        grid.x = naive_grid_size < num_cu * occupancy ? naive_grid_size : num_cu * occupancy;
    };

#define launch_fused_allreduce_rmsnorm(template_kernel)                         \
    do                                                                          \
    {                                                                           \
        auto kernel_ptr = reinterpret_cast<const void*>(template_kernel);       \
        setGrid(naive_grid_size, kernel_ptr);                                   \
        template_kernel<<<grid, block, 0, stream>>>(                            \
            sg_, residual_inp, residual_out, output, weight, eps, rank_, m, n); \
    } while(0)

    if(n_bytes % 1024 == 0)
    {
        if(8192 <= n_bytes && n_bytes <= 32768)
        {
            int naive_grid_size = m;
            int n_loop          = n_bytes / 8192; // 1, 2, 3, 4
            if(n_bytes % 8192 == 0)
            {
                switch(n_loop)
                {
                case 1:
                    launch_fused_allreduce_rmsnorm((local_device_load_rmsnorm_naive<T, 512, 1>));
                    break;
                case 2:
                    launch_fused_allreduce_rmsnorm((local_device_load_rmsnorm_naive<T, 512, 2>));
                    break;
                case 3:
                    launch_fused_allreduce_rmsnorm((local_device_load_rmsnorm_naive<T, 512, 3>));
                    break;
                case 4:
                    launch_fused_allreduce_rmsnorm((local_device_load_rmsnorm_naive<T, 512, 4>));
                    break;
                }
            }
            else
            {
                n_loop += 1;
                switch(n_loop)
                {
                case 2:
                    launch_fused_allreduce_rmsnorm((local_device_load_rmsnorm<T, 512, 2>));
                    break;
                case 3:
                    launch_fused_allreduce_rmsnorm((local_device_load_rmsnorm<T, 512, 3>));
                    break;
                case 4:
                    launch_fused_allreduce_rmsnorm((local_device_load_rmsnorm<T, 512, 4>));
                    break;
                }
            }
        }
        else if(4096 <= n_bytes && n_bytes < 8192)
        {
            block.x             = 256;
            int naive_grid_size = m;
            if(n_bytes == 4096)
            {
                // naive n_loop = 1
                launch_fused_allreduce_rmsnorm((local_device_load_rmsnorm_naive<T, 256, 1>));
            }
            else
            {
                // n_loop = 2
                launch_fused_allreduce_rmsnorm((local_device_load_rmsnorm<T, 256, 2>));
            }
        }
        else if(1024 <= n_bytes && n_bytes < 4096)
        {
            block.x             = 256;
            int naive_grid_size = (m + 3) / 4;
            int n_loop          = n_bytes / 1024;
            switch(n_loop)
            {
            case 1: launch_fused_allreduce_rmsnorm((local_device_load_rmsnorm_512n<T, 1>)); break;
            case 2: launch_fused_allreduce_rmsnorm((local_device_load_rmsnorm_512n<T, 2>)); break;
            case 3: launch_fused_allreduce_rmsnorm((local_device_load_rmsnorm_512n<T, 3>)); break;
            }
        }
        else
        {
            printf("fused allreduce rmsnorm shape size error\n");
        }
    }
    else
    {
        printf("fused allreduce rmsnorm shape error\n");
    }
}

template <typename T, typename QT>
void dispatchFusedAllReduceRMSNormQuant(hipStream_t stream,
                                        T* input,
                                        T* residual_inp,
                                        T* residual_out,
                                        QT* output,
                                        float* scale_out,
                                        T* weight,
                                        float eps,
                                        int m,
                                        int n,
                                        bool use_1stage)
{
    auto d   = 16 / sizeof(T);
    int size = m * n;
    if(size % d != 0)
    {
        throw std::runtime_error("custom allreduce currently requires input length to be multiple "
                                 "of " +
                                 std::to_string(d));
    }
    RankData* ptrs = get_buffer_RD(stream, input);

    bool n_constrain = (n == 4096 || n == 2048 || n == 1024 || n == 512);
    use_1stage       = use_1stage && n_constrain;
#define DISPATCH_AR_FUSION_KERNEL_(NGPUS, N, KERNEL) \
    case N: {                                        \
        KERNEL<T, QT, NGPUS, N>(ptrs,                \
                                sg_,                 \
                                self_sg_,            \
                                rank_,               \
                                residual_inp,        \
                                residual_out,        \
                                output,              \
                                weight,              \
                                scale_out,           \
                                size,                \
                                eps,                 \
                                stream);             \
        return;                                      \
    }
#define DISPATCH_AR_FUSION_KERNEL(NGPUS)                                                     \
    if(use_1stage)                                                                           \
    {                                                                                        \
        switch(n)                                                                            \
        {                                                                                    \
            DISPATCH_AR_FUSION_KERNEL_(NGPUS, 4096, allreduce_fusion_kernel_1stage_launcher) \
            DISPATCH_AR_FUSION_KERNEL_(NGPUS, 2048, allreduce_fusion_kernel_1stage_launcher) \
            DISPATCH_AR_FUSION_KERNEL_(NGPUS, 1024, allreduce_fusion_kernel_1stage_launcher) \
            DISPATCH_AR_FUSION_KERNEL_(NGPUS, 512, allreduce_fusion_kernel_1stage_launcher)  \
        default: printf("fused 1stage allreduce rmsnorm quant N-dim error\n");               \
        }                                                                                    \
    }                                                                                        \
    else if(n_constrain && (size * sizeof(T) <= 512 * 1024))                                 \
    {                                                                                        \
        switch(n)                                                                            \
        {                                                                                    \
            DISPATCH_AR_FUSION_KERNEL_(NGPUS, 4096, allreduce_fusion_kernel_2stage_launcher) \
            DISPATCH_AR_FUSION_KERNEL_(NGPUS, 2048, allreduce_fusion_kernel_2stage_launcher) \
            DISPATCH_AR_FUSION_KERNEL_(NGPUS, 1024, allreduce_fusion_kernel_2stage_launcher) \
            DISPATCH_AR_FUSION_KERNEL_(NGPUS, 512, allreduce_fusion_kernel_2stage_launcher)  \
        default: printf("fused 2stage allreduce rmsnorm quant N-dim error\n");               \
        }                                                                                    \
    }                                                                                        \
    else                                                                                     \
    {                                                                                        \
        switch(n)                                                                            \
        {                                                                                    \
            DISPATCH_AR_FUSION_KERNEL_(NGPUS, 4096, allreduce_fusion_kernel_split_launcher)  \
            DISPATCH_AR_FUSION_KERNEL_(NGPUS, 2048, allreduce_fusion_kernel_split_launcher)  \
            DISPATCH_AR_FUSION_KERNEL_(NGPUS, 1024, allreduce_fusion_kernel_split_launcher)  \
            DISPATCH_AR_FUSION_KERNEL_(NGPUS, 512, allreduce_fusion_kernel_split_launcher)   \
        default: printf("fused allreduce rmsnorm quant N-dim error\n");                      \
        }                                                                                    \
    }

    switch(world_size_)
    {
    case 8: DISPATCH_AR_FUSION_KERNEL(8); break;
    case 4: DISPATCH_AR_FUSION_KERNEL(4); break;
    case 2: DISPATCH_AR_FUSION_KERNEL(2); break;
    default: printf("fused allreduce rmsnorm world size error\n");
    }
}

~CustomAllreduce()
{
    for(auto [_, ptr] : ipc_handles_)
    {
        HIP_CALL(hipIpcCloseMemHandle(ptr));
    }
}
}; // namespace aiter
/**
 * To inspect PTX/SASS, copy paste this header file to compiler explorer and add
 a template instantiation:
 * template void aiter::CustomAllreduce::allreduce<half>(hipStream_t, half *,
 half *, int, int, int);
*/
} // namespace aiter
