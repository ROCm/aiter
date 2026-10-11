#pragma once
// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

// gfx1250-specific PullQ4 layout, kernels, route selection, and launch helpers.
// This implementation header is included by quick_all_reduce.cuh after the
// shared codecs and legacy QuickReduce kernels have been defined.

#include "quick_all_reduce_base.h"

namespace aiter {

inline int64_t quick_reduce_pull_q4_size_per_phase(int64_t max_problem_size)
{
    return divceil(max_problem_size, kTileSize) * CodecQ4<half, 1>::kTransmittedTileSize;
}

inline bool quick_reduce_env_enabled(const char* name)
{
    const char* value = std::getenv(name);
    if(value == nullptr)
        return false;
    std::string normalized(value);
    std::transform(normalized.begin(), normalized.end(), normalized.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return normalized == "1" || normalized == "true" || normalized == "yes" || normalized == "on";
}

struct QuickReduceGfx1250Route
{
    bool use_pull_q4;
    bool use_pull_q4_bulk;
};

inline QuickReduceGfx1250Route quick_reduce_gfx1250_route(const hipDeviceProp_t& prop,
                                                          int configured_quant_level)
{
    const bool is_gfx1250 = std::string(prop.gcnArchName).rfind("gfx1250", 0) == 0;
    const bool configured_int4 =
        configured_quant_level == static_cast<int>(QuickReduceQuantLevel::INT4);
    const bool legacy_force =
        configured_quant_level < 0 && quick_reduce_env_enabled("AITER_QUICK_REDUCE_FORCE_PULL_Q4");
    const bool use_pull_q4 = is_gfx1250 && (configured_int4 || legacy_force);
    return {use_pull_q4,
            use_pull_q4 && (configured_int4 ||
                            quick_reduce_env_enabled("AITER_QUICK_REDUCE_FORCE_PULL_Q4_BULK"))};
}

// gfx1250 peer-pull INT4 all-reduce. Each rank publishes one compressed copy
// of the complete logical tile in its own IPC buffer. Every rank then pulls
// the compressed tile from all peers and reduces locally. This removes the
// two remote-scatter phases and the second quantization from the original
// two-shot protocol.
template <typename T, int world_size, bool cast_bf2half>
struct AllReducePullQ4
{
    static_assert(sizeof(T) == 2);
    static_assert(world_size == 2 || world_size == 4);

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

    __device__ static void run(T const* __restrict__ input,
                               T* __restrict__ output,
                               uint32_t const N,
                               int const block,
                               int const rank,
                               uint8_t** __restrict__ buffer_list,
                               uint32_t const data_offset,
                               uint32_t flag_color,
                               int64_t data_size_per_phase)
    {
        const int thread = threadIdx.x + threadIdx.y * blockDim.x;
        uint8_t* buffer_ptr[world_size];
#pragma unroll
        for(int r = 0; r < world_size; ++r)
            buffer_ptr[r] = buffer_list[r];

        Codec codec(thread, rank);
        const int64_t phase_offset = (flag_color & 1u) * data_size_per_phase;
        int32x4_t* local_send      = reinterpret_cast<int32x4_t*>(
            buffer_ptr[rank] + data_offset + phase_offset + block * Codec::kTransmittedTileSize);
        BufferResource src_buffer(const_cast<T*>(input), N * sizeof(T));
        uint32_t src_offset = block * kTileSize + thread * sizeof(int32x4_t);
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
#pragma unroll
            for(int i = 0; i < kAtoms; ++i)
            {
                codec.send_atom(local_send, &input_atoms[i]);
                local_send += Codec::kRankBufferTileStride;
            }
        }

        // The release flag makes all preceding compressed payload stores
        // visible to peer GPUs before they begin pulling this logical tile.
        __syncthreads();
        const uint32_t flag_phase_size = data_offset / 2;
        const uint32_t pull_flag_phase_size =
            (data_size_per_phase / Codec::kTransmittedTileSize) * sizeof(uint32_t);
        const uint32_t flag_phase_offset =
            (flag_color & 1u) * flag_phase_size + flag_phase_size - pull_flag_phase_size;
        if(thread == 0)
        {
            uint32_t* local_flag = reinterpret_cast<uint32_t*>(
                buffer_ptr[rank] + flag_phase_offset + block * sizeof(uint32_t));
            set_sync_flag(local_flag, flag_color);
        }

        // CAR-style parallel peer polling: one thread per rank, followed by a
        // single CTA barrier instead of one barrier for every peer.
        if(thread < world_size)
        {
            uint32_t* peer_flag = reinterpret_cast<uint32_t*>(
                buffer_ptr[thread] + flag_phase_offset + block * sizeof(uint32_t));
            wait_sync_flag(peer_flag, flag_color);
        }
        __syncthreads();

        int32x4_t reduced[kAtoms] = {};
#pragma unroll
        for(int r = 0; r < world_size; ++r)
        {
            int32x4_t* peer_recv = reinterpret_cast<int32x4_t*>(
                buffer_ptr[r] + data_offset + phase_offset + block * Codec::kTransmittedTileSize);
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
            int32x4_t output_atom;
            if constexpr(cast_bf2half)
            {
                const half2* half_buf = reinterpret_cast<const half2*>(&reduced[i]);
                nv_bfloat162 bf16_buf[4];
#pragma unroll
                for(int j = 0; j < 4; ++j)
                    bf16_buf[j] = __float22bfloat162_rn(__half22float2(half_buf[j]));
                output_atom = *reinterpret_cast<const int32x4_t*>(bf16_buf);
            }
            else
            {
                output_atom = reduced[i];
            }

            if(dst_offset + sizeof(int32x4_t) <= N * sizeof(T))
            {
                buffer_store_dwordx4(output_atom, dst_buffer.descriptor, dst_offset, 0, 0);
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
            dst_offset += kAtomStride * sizeof(int32x4_t);
        }
    }
};

template <typename T, int world_size, bool cast_bf2half>
struct AllReducePullQ4Bulk
{
    static_assert(sizeof(T) == 2);
    static_assert(world_size == 2 || world_size == 4);

    using Codec = CodecQ4<T, 1>;
    using Fused = AllReducePullQ4<T, world_size, cast_bf2half>;

    __device__ static void quantize(T const* __restrict__ input,
                                    uint32_t N,
                                    int block,
                                    int rank,
                                    uint8_t** __restrict__ buffer_list,
                                    uint32_t data_offset,
                                    uint32_t epoch,
                                    int64_t data_size_per_phase)
    {
        const int thread           = threadIdx.x + threadIdx.y * blockDim.x;
        const int64_t bulk_base    = 2 * data_size_per_phase;
        const int64_t phase_offset = (epoch & 1u) * data_size_per_phase;
        int32x4_t* local_send =
            reinterpret_cast<int32x4_t*>(buffer_list[rank] + data_offset + bulk_base +
                                         phase_offset + block * Codec::kTransmittedTileSize);
        BufferResource src_buffer(const_cast<T*>(input), N * sizeof(T));
        uint32_t src_offset = block * kTileSize + thread * sizeof(int32x4_t);
        Codec codec(thread, rank);

        if constexpr(world_size == 2)
        {
#pragma unroll
            for(int i = 0; i < kAtoms; ++i)
            {
                int32x4_t input_atom =
                    Fused::load_input_atom(input, src_buffer.descriptor, N, src_offset);
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
                input_atoms[i] =
                    Fused::load_input_atom(input, src_buffer.descriptor, N, src_offset);
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
                                  int64_t data_size_per_phase)
    {
        const int thread           = threadIdx.x + threadIdx.y * blockDim.x;
        const int64_t bulk_base    = 2 * data_size_per_phase;
        const int64_t phase_offset = (epoch & 1u) * data_size_per_phase;
        Codec codec(thread, rank);
        int32x4_t reduced[kAtoms] = {};

#pragma unroll
        for(int r = 0; r < world_size; ++r)
        {
            int32x4_t* peer_recv =
                reinterpret_cast<int32x4_t*>(buffer_list[r] + data_offset + bulk_base +
                                             phase_offset + block * Codec::kTransmittedTileSize);
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
            int32x4_t output_atom;
            if constexpr(cast_bf2half)
            {
                const half2* half_buf = reinterpret_cast<const half2*>(&reduced[i]);
                nv_bfloat162 bf16_buf[4];
#pragma unroll
                for(int j = 0; j < 4; ++j)
                    bf16_buf[j] = __float22bfloat162_rn(__half22float2(half_buf[j]));
                output_atom = *reinterpret_cast<const int32x4_t*>(bf16_buf);
            }
            else
            {
                output_atom = reduced[i];
            }

            if(dst_offset + sizeof(int32x4_t) <= N * sizeof(T))
            {
                buffer_store_dwordx4(output_atom, dst_buffer.descriptor, dst_offset, 0, 0);
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
            dst_offset += kAtomStride * sizeof(int32x4_t);
        }
    }
};

template <typename AllReduceKernel, typename T>
__global__ __quickreduce_launch_bounds_two_shot__ static void
allreduce_prototype_pull_q4(T const* A,
                            T* B,
                            uint32_t N,
                            uint32_t num_blocks,
                            int rank,
                            uint8_t** dbuffer_list,
                            uint32_t data_offset,
                            uint32_t* d_flag_color,
                            int64_t data_size_per_phase)
{
    int block      = blockIdx.x;
    const int grid = gridDim.x;

    while(block < num_blocks)
    {
        // Use one epoch per logical tile so changing grid sizes cannot make a
        // newly activated tile inherit another tile's wrapped epoch.
        uint32_t flag_color = d_flag_color[block];
        AllReduceKernel::run(
            A, B, N, block, rank, dbuffer_list, data_offset, flag_color, data_size_per_phase);
        if(threadIdx.x == 0 && threadIdx.y == 0)
            d_flag_color[block] = flag_color + 1;
        block += grid;
    }
}

template <typename AllReduceKernel, typename T>
__global__ __quickreduce_launch_bounds_two_shot__ static void
allreduce_prototype_pull_q4_bulk_quantize(T const* A,
                                          uint32_t N,
                                          uint32_t num_blocks,
                                          int rank,
                                          uint8_t** dbuffer_list,
                                          uint32_t data_offset,
                                          uint32_t* d_epoch,
                                          int64_t data_size_per_phase)
{
    const uint32_t epoch = d_epoch[0] + 1u;
    for(int block = blockIdx.x; block < num_blocks; block += gridDim.x)
        AllReduceKernel::quantize(
            A, N, block, rank, dbuffer_list, data_offset, epoch, data_size_per_phase);
}

__global__ static void allreduce_prototype_pull_q4_bulk_publish(int rank,
                                                                int world_size,
                                                                uint8_t** dbuffer_list,
                                                                uint32_t data_offset,
                                                                uint32_t* d_epoch,
                                                                int64_t data_size_per_phase)
{
    const uint32_t epoch = d_epoch[0] + 1u;
    if(threadIdx.x == 0)
    {
        uint32_t* flag =
            reinterpret_cast<uint32_t*>(dbuffer_list[rank] + data_offset + 4 * data_size_per_phase +
                                        (epoch & 1u) * sizeof(uint32_t));
        set_sync_flag(flag, epoch);
    }
    __syncthreads();
    if(threadIdx.x < world_size)
    {
        uint32_t* peer_flag =
            reinterpret_cast<uint32_t*>(dbuffer_list[threadIdx.x] + data_offset +
                                        4 * data_size_per_phase + (epoch & 1u) * sizeof(uint32_t));
        wait_sync_flag(peer_flag, epoch);
    }
    __syncthreads();
    if(threadIdx.x == 0)
        d_epoch[0] = epoch;
}

template <typename AllReduceKernel, typename T>
__global__ __quickreduce_launch_bounds_two_shot__ static void
allreduce_prototype_pull_q4_bulk_reduce(T* B,
                                        uint32_t N,
                                        uint32_t num_blocks,
                                        int rank,
                                        uint8_t** dbuffer_list,
                                        uint32_t data_offset,
                                        uint32_t* d_epoch,
                                        int64_t data_size_per_phase)
{
    const uint32_t epoch = d_epoch[0];

    for(int block = blockIdx.x; block < num_blocks; block += gridDim.x)
        AllReduceKernel::reduce(
            B, N, block, rank, dbuffer_list, data_offset, epoch, data_size_per_phase);
}

#define PULL_Q4_DISPATCH()                                                    \
    if(world_size == 2)                                                       \
    {                                                                         \
        using AllReduceKernel = AllReducePullQ4<T, 2, cast_bf2half>;          \
        hipLaunchKernelGGL((allreduce_prototype_pull_q4<AllReduceKernel, T>), \
                           dim3(grid),                                        \
                           dim3(kBlockTwoShot),                               \
                           0,                                                 \
                           stream,                                            \
                           A,                                                 \
                           B,                                                 \
                           N,                                                 \
                           num_blocks,                                        \
                           rank,                                              \
                           dbuffer_list,                                      \
                           data_offset,                                       \
                           d_pull_q4_flag_color,                              \
                           this->pull_q4_size_per_phase);                     \
    }                                                                         \
    else if(world_size == 4)                                                  \
    {                                                                         \
        using AllReduceKernel = AllReducePullQ4<T, 4, cast_bf2half>;          \
        hipLaunchKernelGGL((allreduce_prototype_pull_q4<AllReduceKernel, T>), \
                           dim3(grid),                                        \
                           dim3(kBlockTwoShot),                               \
                           0,                                                 \
                           stream,                                            \
                           A,                                                 \
                           B,                                                 \
                           N,                                                 \
                           num_blocks,                                        \
                           rank,                                              \
                           dbuffer_list,                                      \
                           data_offset,                                       \
                           d_pull_q4_flag_color,                              \
                           this->pull_q4_size_per_phase);                     \
    }

#define PULL_Q4_BULK_DISPATCH()                                                             \
    if(world_size == 2)                                                                     \
    {                                                                                       \
        using AllReduceKernel = AllReducePullQ4Bulk<T, 2, cast_bf2half>;                    \
        hipLaunchKernelGGL((allreduce_prototype_pull_q4_bulk_quantize<AllReduceKernel, T>), \
                           dim3(grid),                                                      \
                           dim3(kBlockTwoShot),                                             \
                           0,                                                               \
                           stream,                                                          \
                           A,                                                               \
                           N,                                                               \
                           num_blocks,                                                      \
                           rank,                                                            \
                           dbuffer_list,                                                    \
                           data_offset,                                                     \
                           d_pull_q4_bulk_epoch,                                            \
                           this->pull_q4_size_per_phase);                                   \
        hipLaunchKernelGGL(allreduce_prototype_pull_q4_bulk_publish,                        \
                           dim3(1),                                                         \
                           dim3(32),                                                        \
                           0,                                                               \
                           stream,                                                          \
                           rank,                                                            \
                           world_size,                                                      \
                           dbuffer_list,                                                    \
                           data_offset,                                                     \
                           d_pull_q4_bulk_epoch,                                            \
                           this->pull_q4_size_per_phase);                                   \
        hipLaunchKernelGGL((allreduce_prototype_pull_q4_bulk_reduce<AllReduceKernel, T>),   \
                           dim3(grid),                                                      \
                           dim3(kBlockTwoShot),                                             \
                           0,                                                               \
                           stream,                                                          \
                           B,                                                               \
                           N,                                                               \
                           num_blocks,                                                      \
                           rank,                                                            \
                           dbuffer_list,                                                    \
                           data_offset,                                                     \
                           d_pull_q4_bulk_epoch,                                            \
                           this->pull_q4_size_per_phase);                                   \
    }                                                                                       \
    else if(world_size == 4)                                                                \
    {                                                                                       \
        using AllReduceKernel = AllReducePullQ4Bulk<T, 4, cast_bf2half>;                    \
        hipLaunchKernelGGL((allreduce_prototype_pull_q4_bulk_quantize<AllReduceKernel, T>), \
                           dim3(grid),                                                      \
                           dim3(kBlockTwoShot),                                             \
                           0,                                                               \
                           stream,                                                          \
                           A,                                                               \
                           N,                                                               \
                           num_blocks,                                                      \
                           rank,                                                            \
                           dbuffer_list,                                                    \
                           data_offset,                                                     \
                           d_pull_q4_bulk_epoch,                                            \
                           this->pull_q4_size_per_phase);                                   \
        hipLaunchKernelGGL(allreduce_prototype_pull_q4_bulk_publish,                        \
                           dim3(1),                                                         \
                           dim3(32),                                                        \
                           0,                                                               \
                           stream,                                                          \
                           rank,                                                            \
                           world_size,                                                      \
                           dbuffer_list,                                                    \
                           data_offset,                                                     \
                           d_pull_q4_bulk_epoch,                                            \
                           this->pull_q4_size_per_phase);                                   \
        hipLaunchKernelGGL((allreduce_prototype_pull_q4_bulk_reduce<AllReduceKernel, T>),   \
                           dim3(grid),                                                      \
                           dim3(kBlockTwoShot),                                             \
                           0,                                                               \
                           stream,                                                          \
                           B,                                                               \
                           N,                                                               \
                           num_blocks,                                                      \
                           rank,                                                            \
                           dbuffer_list,                                                    \
                           data_offset,                                                     \
                           d_pull_q4_bulk_epoch,                                            \
                           this->pull_q4_size_per_phase);                                   \
    }

} // namespace aiter
