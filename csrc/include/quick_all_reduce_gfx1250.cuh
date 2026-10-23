#pragma once
// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

// gfx1250-specific PullQ4 layout, kernels, route selection, and launch helpers.
// This implementation header is included by quick_all_reduce.cuh after the
// shared codecs and legacy QuickReduce kernels have been defined.

#include "quick_all_reduce_base.h"

namespace aiter {

// PullQ4 communicator data region, starting at DeviceComms::data_offset:
//   [0, pq)            epoch-parity-0 compressed payload
//   [pq, 2 * pq)       epoch-parity-1 compressed payload
//   [2 * pq, + 8)      two uint32 publication flags, selected by epoch parity
// where pq = quick_reduce_pull_q4_size_per_phase(max_problem_size).
inline int64_t quick_reduce_pull_q4_size_per_phase(int64_t max_problem_size)
{
    return divceil(max_problem_size, kTileSize) * CodecQ4<half, 1>::kTransmittedTileSize;
}

inline int64_t quick_reduce_pull_q4_flags_offset(int64_t max_problem_size)
{
    return 2 * quick_reduce_pull_q4_size_per_phase(max_problem_size);
}

inline int64_t quick_reduce_pull_q4_buffer_size(int64_t max_problem_size)
{
    return quick_reduce_pull_q4_flags_offset(max_problem_size) + 2 * sizeof(uint32_t);
}

// Only the gfx1250 prefix is enabled; other gfx125x parts have not been validated.
inline bool quick_reduce_use_pull_q4(const hipDeviceProp_t& prop, int world_size, int quant_level)
{
    const bool is_gfx1250 = std::string(prop.gcnArchName).rfind("gfx1250", 0) == 0;
    return is_gfx1250 && quant_level == static_cast<int>(QuickReduceQuantLevel::INT4) &&
           (world_size == 2 || world_size == 4 || world_size == 8);
}

// gfx1250 peer-pull INT4 all-reduce in three same-stream dispatches:
//   1. quantize: every rank compresses each 32 KiB logical tile once into its
//      own IPC buffer.
//   2. publish: one CTA release-publishes the local epoch and acquire-waits
//      for every peer's epoch.
//   3. reduce: every rank pulls all compressed tiles and reduces in fixed rank
//      order, so outputs are bit-identical across ranks.
// This removes the two remote-scatter phases and the second quantization of
// the two-shot protocol, and the dispatch boundary removes per-CTA polling.
template <typename T, int world_size, bool cast_bf2half>
struct AllReducePullQ4
{
    static_assert(sizeof(T) == 2);
    static_assert(world_size == 2 || world_size == 4 || world_size == 8);

    using Codec = CodecQ4<T, 1>;

    __device__ static int32x4_t load_input_atom(T const* __restrict__ input,
                                                int32x4_t src_descriptor,
                                                uint32_t N,
                                                uint32_t src_offset)
    {
        int32x4_t input_atom;
        if(src_offset + sizeof(int32x4_t) <= N * sizeof(T))
        {
            input_atom = buffer_load_dwordx4(src_descriptor, src_offset, 0, 0);
        }
        else
        {
            int32x4_t tail_atom      = {};
            uint16_t* tail           = reinterpret_cast<uint16_t*>(&tail_atom);
            const uint16_t* src_bits = reinterpret_cast<const uint16_t*>(input);
            const uint32_t elem      = src_offset / sizeof(T);
#pragma unroll
            for(int j = 0; j < sizeof(int32x4_t) / sizeof(uint16_t); ++j)
            {
                if(elem + j < N)
                    tail[j] = src_bits[elem + j];
            }
            input_atom = tail_atom;
        }
        if constexpr(cast_bf2half)
        {
            const nv_bfloat162* bf_buf = reinterpret_cast<const nv_bfloat162*>(&input_atom);
            half2 half_buf[4];
#pragma unroll
            for(int j = 0; j < 4; ++j)
                half_buf[j] = __float22half2_rn(__bfloat1622float2(bf_buf[j]));
            input_atom = *reinterpret_cast<const int32x4_t*>(half_buf);
        }
        return input_atom;
    }

    __device__ static void store_output_atom(T* __restrict__ output,
                                             int32x4_t dst_descriptor,
                                             uint32_t N,
                                             uint32_t dst_offset,
                                             int32x4_t reduced)
    {
        int32x4_t output_atom;
        if constexpr(cast_bf2half)
        {
            const half2* half_buf = reinterpret_cast<const half2*>(&reduced);
            nv_bfloat162 bf16_buf[4];
#pragma unroll
            for(int j = 0; j < 4; ++j)
                bf16_buf[j] = __float22bfloat162_rn(__half22float2(half_buf[j]));
            output_atom = *reinterpret_cast<const int32x4_t*>(bf16_buf);
        }
        else
        {
            output_atom = reduced;
        }

        if(dst_offset + sizeof(int32x4_t) <= N * sizeof(T))
        {
            buffer_store_dwordx4(output_atom, dst_descriptor, dst_offset, 0, 0);
        }
        else
        {
            uint16_t* dst_bits       = reinterpret_cast<uint16_t*>(output);
            const uint16_t* out_bits = reinterpret_cast<const uint16_t*>(&output_atom);
            const uint32_t elem      = dst_offset / sizeof(T);
#pragma unroll
            for(int j = 0; j < sizeof(int32x4_t) / sizeof(uint16_t); ++j)
            {
                if(elem + j < N)
                    dst_bits[elem + j] = out_bits[j];
            }
        }
    }

    __device__ static void quantize(T const* __restrict__ input,
                                    uint32_t N,
                                    int block,
                                    int rank,
                                    uint8_t** __restrict__ buffer_list,
                                    uint32_t data_offset,
                                    uint32_t epoch,
                                    int64_t size_per_phase)
    {
        const int thread           = threadIdx.x + threadIdx.y * blockDim.x;
        const int64_t phase_offset = (epoch & 1u) * size_per_phase;
        int32x4_t* local_send      = reinterpret_cast<int32x4_t*>(
            buffer_list[rank] + data_offset + phase_offset + block * Codec::kTransmittedTileSize);
        BufferResource src_buffer(const_cast<T*>(input), N * sizeof(T));
        uint32_t src_offset = block * kTileSize + thread * sizeof(int32x4_t);
        Codec codec(thread, rank);

        if constexpr(world_size == 2)
        {
#pragma unroll
            for(int i = 0; i < kAtoms; ++i)
            {
                int32x4_t input_atom = load_input_atom(input, src_buffer.descriptor, N, src_offset);
                codec.send_atom(local_send, &input_atom);
                local_send += Codec::kRankBufferTileStride;
                src_offset += kAtomStride * sizeof(int32x4_t);
            }
        }
        else
        {
            int32x4_t input_atoms[kAtoms];
#pragma unroll
            for(int i = 0; i < kAtoms; ++i)
            {
                input_atoms[i] = load_input_atom(input, src_buffer.descriptor, N, src_offset);
                src_offset += kAtomStride * sizeof(int32x4_t);
            }
            codec.send(local_send, input_atoms);
        }
    }

    __device__ static void reduce(T* __restrict__ output,
                                  uint32_t N,
                                  int block,
                                  int rank,
                                  uint8_t** __restrict__ buffer_list,
                                  uint32_t data_offset,
                                  uint32_t epoch,
                                  int64_t size_per_phase)
    {
        const int thread           = threadIdx.x + threadIdx.y * blockDim.x;
        const int64_t phase_offset = (epoch & 1u) * size_per_phase;
        Codec codec(thread, rank);
        int32x4_t reduced[kAtoms] = {};

#pragma unroll
        for(int r = 0; r < world_size; ++r)
        {
            int32x4_t* peer_recv = reinterpret_cast<int32x4_t*>(
                buffer_list[r] + data_offset + phase_offset + block * Codec::kTransmittedTileSize);
#pragma unroll
            for(int i = 0; i < kAtoms; ++i)
            {
                int32x4_t peer_atom;
                codec.recv_atom(peer_recv, &peer_atom);
                packed_assign_add<T>(&reduced[i], &peer_atom);
                peer_recv += Codec::kRankBufferTileStride;
            }
        }

        BufferResource dst_buffer(output, N * sizeof(T));
        uint32_t dst_offset = block * kTileSize + thread * sizeof(int32x4_t);
#pragma unroll
        for(int i = 0; i < kAtoms; ++i)
        {
            store_output_atom(output, dst_buffer.descriptor, N, dst_offset, reduced[i]);
            dst_offset += kAtomStride * sizeof(int32x4_t);
        }
    }
};

template <typename AllReduceKernel, typename T>
__global__ __quickreduce_launch_bounds_two_shot__ static void
allreduce_prototype_pull_q4_quantize(T const* A,
                                     uint32_t N,
                                     uint32_t num_blocks,
                                     int rank,
                                     uint8_t** dbuffer_list,
                                     uint32_t data_offset,
                                     uint32_t* d_epoch,
                                     int64_t size_per_phase)
{
    const uint32_t epoch = d_epoch[0] + 1u;
    for(int block = blockIdx.x; block < num_blocks; block += gridDim.x)
        AllReduceKernel::quantize(
            A, N, block, rank, dbuffer_list, data_offset, epoch, size_per_phase);
}

// Peers observe the quantize kernel's nontemporal payload stores after this
// release only because dbuffer is allocated with hipDeviceMallocUncached.
__global__ static void allreduce_prototype_pull_q4_publish(int rank,
                                                           int world_size,
                                                           uint8_t** dbuffer_list,
                                                           uint32_t data_offset,
                                                           uint32_t* d_epoch,
                                                           int64_t size_per_phase)
{
    const uint32_t epoch       = d_epoch[0] + 1u;
    const int64_t flags_offset = data_offset + 2 * size_per_phase + (epoch & 1u) * sizeof(uint32_t);
    if(threadIdx.x == 0)
        set_sync_flag(reinterpret_cast<uint32_t*>(dbuffer_list[rank] + flags_offset), epoch);
    __syncthreads();
    if(threadIdx.x < world_size)
        wait_sync_flag(reinterpret_cast<uint32_t*>(dbuffer_list[threadIdx.x] + flags_offset),
                       epoch);
    __syncthreads();
    if(threadIdx.x == 0)
        d_epoch[0] = epoch;
}

template <typename AllReduceKernel, typename T>
__global__ __quickreduce_launch_bounds_two_shot__ static void
allreduce_prototype_pull_q4_reduce(T* B,
                                   uint32_t N,
                                   uint32_t num_blocks,
                                   int rank,
                                   uint8_t** dbuffer_list,
                                   uint32_t data_offset,
                                   uint32_t* d_epoch,
                                   int64_t size_per_phase)
{
    const uint32_t epoch = d_epoch[0];
    for(int block = blockIdx.x; block < num_blocks; block += gridDim.x)
        AllReduceKernel::reduce(
            B, N, block, rank, dbuffer_list, data_offset, epoch, size_per_phase);
}

template <typename T, int world_size, bool cast_bf2half>
void quick_reduce_launch_pull_q4(T const* A,
                                 T* B,
                                 uint32_t N,
                                 uint32_t num_blocks,
                                 uint32_t grid,
                                 int rank,
                                 uint8_t** dbuffer_list,
                                 uint32_t data_offset,
                                 uint32_t* d_epoch,
                                 int64_t size_per_phase,
                                 hipStream_t stream)
{
    using AllReduceKernel = AllReducePullQ4<T, world_size, cast_bf2half>;
    hipLaunchKernelGGL((allreduce_prototype_pull_q4_quantize<AllReduceKernel, T>),
                       dim3(grid),
                       dim3(kBlockTwoShot),
                       0,
                       stream,
                       A,
                       N,
                       num_blocks,
                       rank,
                       dbuffer_list,
                       data_offset,
                       d_epoch,
                       size_per_phase);
    hipLaunchKernelGGL(allreduce_prototype_pull_q4_publish,
                       dim3(1),
                       dim3(32),
                       0,
                       stream,
                       rank,
                       world_size,
                       dbuffer_list,
                       data_offset,
                       d_epoch,
                       size_per_phase);
    hipLaunchKernelGGL((allreduce_prototype_pull_q4_reduce<AllReduceKernel, T>),
                       dim3(grid),
                       dim3(kBlockTwoShot),
                       0,
                       stream,
                       B,
                       N,
                       num_blocks,
                       rank,
                       dbuffer_list,
                       data_offset,
                       d_epoch,
                       size_per_phase);
}

template <typename T, bool cast_bf2half>
void quick_reduce_dispatch_pull_q4(int world_size,
                                   T const* A,
                                   T* B,
                                   uint32_t N,
                                   uint32_t num_blocks,
                                   uint32_t grid,
                                   int rank,
                                   uint8_t** dbuffer_list,
                                   uint32_t data_offset,
                                   uint32_t* d_epoch,
                                   int64_t size_per_phase,
                                   hipStream_t stream)
{
    switch(world_size)
    {
    case 2:
        quick_reduce_launch_pull_q4<T, 2, cast_bf2half>(A,
                                                        B,
                                                        N,
                                                        num_blocks,
                                                        grid,
                                                        rank,
                                                        dbuffer_list,
                                                        data_offset,
                                                        d_epoch,
                                                        size_per_phase,
                                                        stream);
        break;
    case 4:
        quick_reduce_launch_pull_q4<T, 4, cast_bf2half>(A,
                                                        B,
                                                        N,
                                                        num_blocks,
                                                        grid,
                                                        rank,
                                                        dbuffer_list,
                                                        data_offset,
                                                        d_epoch,
                                                        size_per_phase,
                                                        stream);
        break;
    case 8:
        quick_reduce_launch_pull_q4<T, 8, cast_bf2half>(A,
                                                        B,
                                                        N,
                                                        num_blocks,
                                                        grid,
                                                        rank,
                                                        dbuffer_list,
                                                        data_offset,
                                                        d_epoch,
                                                        size_per_phase,
                                                        stream);
        break;
    default: throw std::runtime_error("PullQ4 supports only world sizes 2, 4, and 8");
    }
}

} // namespace aiter
