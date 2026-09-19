// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

#include "aiter_hip_common.h"
#include "aiter_stream.h"
#include "iq2r.h"
#include "opus/opus.hpp"

#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <optional>

namespace aiter {
namespace {

constexpr int kCodebookBytes       = 4096;
constexpr int kTileN               = 16;
constexpr int kTileK               = 128;
constexpr int kScaleBlock          = 32;
constexpr int kLaneRecordBytes     = 8;
constexpr int kAtomsPerTriplet     = 3;
constexpr int kNBlocksPerGroup     = 6;
constexpr int kAtomPairBytes       = 1024;
constexpr int kAtomTwoRecordBytes  = 12;
constexpr int kAtomTwoOffset       = 1024;
constexpr int kAtomTwoMetadata     = 8;
constexpr int kTripletBytes        = 1792;
// A gfx950 12-byte buffer_load_lds advances the destination by a 16-byte
// lane slot.  Keep the third-atom records padded in LDS while retaining the
// compact 12-byte global layout.
constexpr int kAtomTwoLDSRecordBytes = 16;
constexpr int kTripletLDSBytes       = 2048;
constexpr int kGroupBytes          = 3584;

__device__ __forceinline__ int64_t physical_tile(int n_block,
                                                  int k_tile,
                                                  int k_tiles)
{
    return (static_cast<int64_t>(n_block / kNBlocksPerGroup) * k_tiles + k_tile) *
               kNBlocksPerGroup +
           n_block % kNBlocksPerGroup;
}

__device__ __forceinline__ int64_t triplet_base(int64_t tile)
{
    const int block_in_group = static_cast<int>(tile % kNBlocksPerGroup);
    return (tile / kNBlocksPerGroup) * kGroupBytes +
           (block_in_group / kAtomsPerTriplet) * kTripletBytes;
}

__device__ __forceinline__ int64_t lane_record_offset(int64_t tile, int lane)
{
    const int atom = static_cast<int>(tile % kNBlocksPerGroup) % kAtomsPerTriplet;
    const int64_t base = triplet_base(tile);
    return atom < 2 ? base + lane * (2 * kLaneRecordBytes) + atom * kLaneRecordBytes
                    : base + kAtomTwoOffset + lane * kAtomTwoRecordBytes;
}

__device__ __forceinline__ int64_t metadata_offset(int64_t tile, int lane)
{
    return triplet_base(tile) + kAtomTwoOffset + lane * kAtomTwoRecordBytes +
           kAtomTwoMetadata;
}

__device__ __forceinline__ uint32_t load_u32(const uint8_t* pointer)
{
    return *reinterpret_cast<const uint32_t*>(pointer);
}

__device__ __forceinline__ uint64_t load_u64(const uint8_t* pointer)
{
    uint64_t result;
    auto* words = reinterpret_cast<uint32_t*>(&result);
    words[0] = load_u32(pointer);
    words[1] = load_u32(pointer + 4);
    return result;
}

__device__ __forceinline__ uint64_t apply_signs(uint64_t magnitude,
                                                 uint32_t signs)
{
    constexpr uint32_t spread = 0x10204080u;
    constexpr uint32_t sign_mask = 0x80808080u;
    const uint32_t low = ((signs & 0x0fu) * spread) & sign_mask;
    const uint32_t high = (((signs >> 4) & 0x0fu) * spread) & sign_mask;
    return magnitude ^ (static_cast<uint64_t>(high) << 32) ^ low;
}

__device__ __forceinline__ void wave_min(float& error, int& index)
{
#pragma unroll
    for(int offset = 32; offset > 0; offset >>= 1)
    {
        const float other_error = __shfl_down(error, offset);
        const int other_index = __shfl_down(index, offset);
        if(other_error < error || (other_error == error && other_index < index))
        {
            error = other_error;
            index = other_index;
        }
    }
}

__global__ __launch_bounds__(64) void iq2r_encode_assign_kernel(
    const float* __restrict__ weight,
    const float* __restrict__ importance,
    const float* __restrict__ codebook,
    float codebook_max,
    int N,
    int storage_K,
    int valid_K,
    int exponent_radius,
    uint16_t* __restrict__ indices,
    uint8_t* __restrict__ data,
    uint8_t* __restrict__ scales)
{
    const int logical = static_cast<int>(blockIdx.x);
    const int lane = static_cast<int>(threadIdx.x);
    const int blocks_per_row = valid_K / kScaleBlock;
    const int row = logical / blocks_per_row;
    const int block_k = logical % blocks_per_row;
    if(row >= N)
        return;
    const int k_base = block_k * kScaleBlock;
    const float value =
        lane < kScaleBlock ? weight[static_cast<int64_t>(row) * storage_K + k_base + lane]
                           : 0.0f;
    float maximum = fabsf(value);
#pragma unroll
    for(int offset = 32; offset > 0; offset >>= 1)
        maximum = fmaxf(maximum, __shfl_down(maximum, offset));
    maximum = __shfl(maximum, 0);
    const int center = static_cast<int>(nearbyintf(
        log2f(fmaxf(maximum, 0x1p-126f) / fmaxf(codebook_max, 1.0e-30f))));

    float best_total = INFINITY;
    int best_exponent = 0;
    int best_indices[4] = {0, 0, 0, 0};
    const int candidate_count = 2 * exponent_radius + 1;
    for(int candidate = 0; candidate < candidate_count; ++candidate)
    {
        const int exponent = min(127, max(-126, center + candidate - exponent_radius));
        const float scale = exp2f(static_cast<float>(exponent));
        float total = 0.0f;
        int candidate_indices[4] = {0, 0, 0, 0};
#pragma unroll
        for(int group = 0; group < 4; ++group)
        {
            float local_error = INFINITY;
            int local_index = 0;
#pragma unroll
            for(int slot = 0; slot < 8; ++slot)
            {
                const int codebook_index = lane + slot * 64;
                float error = 0.0f;
#pragma unroll
                for(int element = 0; element < 8; ++element)
                {
                    const int k = k_base + group * 8 + element;
                    const float delta =
                        fabsf(weight[static_cast<int64_t>(row) * storage_K + k]) -
                        codebook[codebook_index * 8 + element] * scale;
                    error = fmaf(delta * delta, importance[k], error);
                }
                if(error < local_error)
                {
                    local_error = error;
                    local_index = codebook_index;
                }
            }
            wave_min(local_error, local_index);
            if(lane == 0)
            {
                total += local_error;
                candidate_indices[group] = local_index;
            }
        }
        if(lane == 0 && total < best_total)
        {
            best_total = total;
            best_exponent = exponent;
#pragma unroll
            for(int group = 0; group < 4; ++group)
                best_indices[group] = candidate_indices[group];
        }
    }
    if(lane != 0)
        return;

    const int row_in_tile = row % kTileN;
    const int n_block = row / kTileN;
    const int k_tile = block_k / 4;
    const int logical_block = block_k % 4;
    const int k_tiles = storage_K / kTileK;
    const int64_t tile = physical_tile(n_block, k_tile, k_tiles);
    scales[tile * 64 + logical_block * 16 + row_in_tile] =
        static_cast<uint8_t>(best_exponent + 127);
#pragma unroll
    for(int group = 0; group < 4; ++group)
    {
        const int physical_group = 2 * (logical_block % 2) + group / 2;
        const int physical_slot = 2 * (logical_block / 2) + group % 2;
        const int physical_lane = physical_group * 16 + row_in_tile;
        indices[(tile * 64 + physical_lane) * 4 + physical_slot] =
            static_cast<uint16_t>(best_indices[group]);
        uint8_t sign = 0;
#pragma unroll
        for(int element = 0; element < 8; ++element)
        {
            const float item = weight[static_cast<int64_t>(row) * storage_K +
                                      k_base + group * 8 + element];
            sign |= static_cast<uint8_t>(signbit(item)) << element;
        }
        data[lane_record_offset(tile, physical_lane) + 4 + physical_slot] = sign;
    }
}

__global__ __launch_bounds__(256) void iq2r_encode_base_kernel(
    const uint8_t* __restrict__ scales,
    int N,
    int storage_K,
    int valid_K,
    uint8_t* __restrict__ base_exponents)
{
    const int n_block = static_cast<int>(blockIdx.x);
    const int thread = static_cast<int>(threadIdx.x);
    const int n_blocks = N / kTileN;
    if(n_block >= n_blocks)
    {
        if(thread == 0)
            base_exponents[n_block] = 127;
        return;
    }
    const int k_tiles = storage_K / kTileK;
    const int valid_blocks = valid_K / kScaleBlock;
    uint32_t minimum = 255;
    for(int item = thread; item < valid_blocks * 16; item += blockDim.x)
    {
        const int block_k = item / 16;
        const int row_in_tile = item % 16;
        const int k_tile = block_k / 4;
        const int logical_block = block_k % 4;
        const int64_t tile = physical_tile(n_block, k_tile, k_tiles);
        minimum = min(minimum,
                      static_cast<uint32_t>(
                          scales[tile * 64 + logical_block * 16 + row_in_tile]));
    }
#pragma unroll
    for(int offset = 32; offset > 0; offset >>= 1)
        minimum = min(minimum, __shfl_down(minimum, offset));
    __shared__ uint32_t wave_minimum[4];
    if((thread & 63) == 0)
        wave_minimum[thread / 64] = minimum;
    __syncthreads();
    if(thread < 64)
    {
        minimum = thread < 4 ? wave_minimum[thread] : 255;
#pragma unroll
        for(int offset = 32; offset > 0; offset >>= 1)
            minimum = min(minimum, __shfl_down(minimum, offset));
        if(thread == 0)
            base_exponents[n_block] = static_cast<uint8_t>(minimum);
    }
}

__global__ __launch_bounds__(256) void iq2r_encode_pack_kernel(
    const uint16_t* __restrict__ indices,
    const uint8_t* __restrict__ scales,
    const uint8_t* __restrict__ base_exponents,
    int64_t triplets,
    int N,
    int storage_K,
    int valid_K,
    uint8_t* __restrict__ data,
    int32_t* __restrict__ scale_delta_overflow)
{
    const int64_t item = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if(item >= triplets * 64)
        return;
    const int64_t triplet_index = item / 64;
    const int lane = static_cast<int>(item % 64);
    const int k_tiles = storage_K / kTileK;
    const int64_t group_k = triplet_index / 2;
    const int triplet_in_group = static_cast<int>(triplet_index % 2);
    const int n_group = static_cast<int>(group_k / k_tiles);
    const int k_tile = static_cast<int>(group_k % k_tiles);
    const int64_t base = group_k * kGroupBytes + triplet_in_group * kTripletBytes;
    uint32_t metadata = 0;
#pragma unroll
    for(int atom = 0; atom < kAtomsPerTriplet; ++atom)
    {
        const int64_t tile = group_k * kNBlocksPerGroup +
                             triplet_in_group * kAtomsPerTriplet + atom;
        uint32_t high_bits = 0;
#pragma unroll
        for(int slot = 0; slot < 4; ++slot)
        {
            const uint32_t value = indices[(tile * 64 + lane) * 4 + slot];
            data[lane_record_offset(tile, lane) + slot] = static_cast<uint8_t>(value);
            high_bits |= ((value >> 8) & 1u) << slot;
        }
        metadata |= high_bits << (atom * 8);
        const int n_block = n_group * kNBlocksPerGroup +
                            triplet_in_group * kAtomsPerTriplet + atom;
        const int block_k = k_tile * 4 + lane / 16;
        if(n_block < N / kTileN && block_k < valid_K / kScaleBlock)
        {
            const int scale = scales[tile * 64 + lane];
            const int base_exponent = base_exponents[n_block];
            const int delta = scale - base_exponent;
            if(delta > 15)
                atomicExch(scale_delta_overflow, 1);
            metadata |= static_cast<uint32_t>(min(max(delta, 0), 15))
                        << (atom * 8 + 4);
        }
    }
    const int64_t destination = base + kAtomTwoOffset + lane * kAtomTwoRecordBytes +
                                kAtomTwoMetadata;
    data[destination] = static_cast<uint8_t>(metadata);
    data[destination + 1] = static_cast<uint8_t>(metadata >> 8);
    data[destination + 2] = static_cast<uint8_t>(metadata >> 16);
}

__device__ __forceinline__ float decode_fp8(uint8_t bits)
{
    const opus::fp8_t value = __builtin_bit_cast(opus::fp8_t, bits);
    return opus::fp8_to_fp32(value);
}

__global__ void iq2r_materialize_kernel(const uint8_t* __restrict__ data,
                                        const uint8_t* __restrict__ auxiliary,
                                        float* __restrict__ output,
                                        int N,
                                        int K)
{
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t elements = static_cast<int64_t>(N) * K;
    if(index >= elements)
        return;

    const int row = static_cast<int>(index / K);
    const int column = static_cast<int>(index % K);
    const int n_block = row / kTileN;
    const int row_in_block = row % kTileN;
    const int k_tile = column / kTileK;
    const int local_k = column % kTileK;
    const int logical_block = local_k / kScaleBlock;
    const int within_block = local_k % kScaleBlock;
    const int lane_group = (logical_block % 2) * 2 + within_block / 16;
    const int lane = lane_group * 16 + row_in_block;
    const int fragment_index = (logical_block / 2) * 16 + within_block % 16;
    const int codeword = fragment_index / 8;
    const int element = fragment_index % 8;
    const int k_tiles = (K + kTileK - 1) / kTileK;
    const int64_t tile = physical_tile(n_block, k_tile, k_tiles);
    const uint8_t* record = data + lane_record_offset(tile, lane);
    const uint32_t metadata = load_u32(data + metadata_offset(tile, lane));
    const int atom = n_block % kAtomsPerTriplet;
    const int high_bits = (metadata >> (atom * 8)) & 0x0f;
    const int codebook_index = record[codeword] |
                               (((high_bits >> codeword) & 1) << 8);
    const uint8_t magnitude = auxiliary[codebook_index * 8 + element];
    const uint8_t signed_value = magnitude ^
                                 (((record[4 + codeword] >> element) & 1) << 7);
    const int scale_lane = logical_block * 16 + row_in_block;
    const uint32_t scale_metadata =
        load_u32(data + metadata_offset(tile, scale_lane));
    const int delta = (scale_metadata >> (atom * 8 + 4)) & 0x0f;
    const int exponent = static_cast<int>(auxiliary[kCodebookBytes + n_block]) + delta;
    const float scale = exponent == 0 ? 0.0f : ldexpf(1.0f, exponent - 127);
    output[index] = decode_fp8(signed_value) * scale;
}

template<int TileN>
__global__ __launch_bounds__(256) void iq2r_gemm_kernel(
    const opus::fp8_t* __restrict__ activations,
    const uint8_t* __restrict__ activation_scales,
    const uint8_t* __restrict__ all_data,
    const uint8_t* __restrict__ all_auxiliary,
    const __hip_bfloat16* __restrict__ all_bias,
    __hip_bfloat16* __restrict__ output,
    int M,
    int N,
    int K,
    int data_bytes,
    int auxiliary_bytes,
    int expert_index)
{
#if defined(__gfx950__)
    constexpr int kThreads = 256;
    constexpr int kWaves = kThreads / 64;
    constexpr int kAtomsPerWave = TileN / (kWaves * 16);
    static_assert(TileN == 64 || TileN == 128);
    static_assert(kAtomsPerWave == 1 || kAtomsPerWave == 2);

    __shared__ alignas(16) uint8_t codebook[kCodebookBytes];
    const uint8_t* data = all_data + static_cast<int64_t>(expert_index) * data_bytes;
    const uint8_t* auxiliary =
        all_auxiliary + static_cast<int64_t>(expert_index) * auxiliary_bytes;
    for(int byte = threadIdx.x * 16; byte < kCodebookBytes; byte += kThreads * 16)
        *reinterpret_cast<uint4*>(codebook + byte) =
            *reinterpret_cast<const uint4*>(auxiliary + byte);
    __syncthreads();

    const int wave = threadIdx.x / 64;
    const int lane = threadIdx.x % 64;
    const int lane_row = lane % 16;
    const int lane_group = lane / 16;
    const int row_base = blockIdx.y * 16;
    const int input_row = row_base + lane_row;
    const int n_block_base = blockIdx.x * (TileN / 16) + wave * kAtomsPerWave;
    const int k_tiles = (K + kTileK - 1) / kTileK;
    opus::vector_t<float, 4> accumulators[kAtomsPerWave] = {};
    auto mma = opus::mfma<opus::fp8_t, opus::fp8_t, opus::fp32_t, 16, 16, 128>{};

    for(int k_tile = 0; k_tile < k_tiles; ++k_tile)
    {
        union
        {
            opus::i32x8_t words;
            uint8_t bytes[32];
        } activation_fragment;
#pragma unroll
        for(int item = 0; item < 16; ++item)
        {
            const int k0 = k_tile * kTileK + lane_group * 16 + item;
            const int k1 = k0 + 64;
            activation_fragment.bytes[item] =
                input_row < M && k0 < K
                    ? __builtin_bit_cast(uint8_t,
                          activations[static_cast<int64_t>(input_row) * K + k0])
                    : 0;
            activation_fragment.bytes[16 + item] =
                input_row < M && k1 < K
                    ? __builtin_bit_cast(uint8_t,
                          activations[static_cast<int64_t>(input_row) * K + k1])
                    : 0;
        }
        uint32_t scale_a = 127u * 0x01010101u;
        if(input_row < M)
        {
            const uint32_t exponent = activation_scales[
                static_cast<int64_t>(input_row) * (K / kScaleBlock) +
                k_tile * 4 + lane_group];
            scale_a = exponent * 0x01010101u;
        }

        uint32_t scale_b = 0;
        opus::i32x8_t weight_fragments[kAtomsPerWave];
#pragma unroll
        for(int atom_index = 0; atom_index < kAtomsPerWave; ++atom_index)
        {
            const int n_block = n_block_base + atom_index;
            const int64_t tile = physical_tile(n_block, k_tile, k_tiles);
            const uint8_t* record = data + lane_record_offset(tile, lane);
            const uint32_t index_lows = load_u32(record);
            const uint32_t signs = load_u32(record + 4);
            const uint32_t metadata = load_u32(data + metadata_offset(tile, lane));
            const int atom = n_block % kAtomsPerTriplet;
            const uint32_t index_highs = (metadata >> (atom * 8)) & 0x0fu;
            union
            {
                opus::i32x8_t words;
                uint64_t codewords[4];
            } decoded;
#pragma unroll
            for(int codeword = 0; codeword < 4; ++codeword)
            {
                const int codebook_index =
                    ((index_lows >> (codeword * 8)) & 0xffu) |
                    (((index_highs >> codeword) & 1u) << 8);
                decoded.codewords[codeword] = apply_signs(
                    *reinterpret_cast<const uint64_t*>(codebook + codebook_index * 8),
                    (signs >> (codeword * 8)) & 0xffu);
            }
            weight_fragments[atom_index] = decoded.words;
            const uint32_t base = auxiliary[kCodebookBytes + n_block];
            const uint32_t delta = (metadata >> (atom * 8 + 4)) & 0x0fu;
            scale_b |= (base + delta) << (atom_index * 8);
        }

#pragma unroll
        for(int atom_index = 0; atom_index < kAtomsPerWave; ++atom_index)
        {
            if constexpr(kAtomsPerWave == 1)
            {
                accumulators[atom_index] = mma(activation_fragment.words,
                                                weight_fragments[atom_index],
                                                accumulators[atom_index],
                                                scale_a,
                                                scale_b,
                                                opus::number<0>{},
                                                opus::number<0>{});
            }
            else if(atom_index == 0)
            {
                accumulators[atom_index] = mma(activation_fragment.words,
                                                weight_fragments[atom_index],
                                                accumulators[atom_index],
                                                scale_a,
                                                scale_b,
                                                opus::number<0>{},
                                                opus::number<0>{});
            }
            else
            {
                accumulators[atom_index] = mma(activation_fragment.words,
                                                weight_fragments[atom_index],
                                                accumulators[atom_index],
                                                scale_a,
                                                scale_b,
                                                opus::number<0>{},
                                                opus::number<1>{});
            }
        }
    }

#pragma unroll
    for(int atom_index = 0; atom_index < kAtomsPerWave; ++atom_index)
    {
        const int output_column = (n_block_base + atom_index) * 16 + lane_row;
#pragma unroll
        for(int item = 0; item < 4; ++item)
        {
            const int output_row = row_base + lane_group * 4 + item;
            if(output_row < M && output_column < N)
            {
                float value = accumulators[atom_index][item];
                if(all_bias != nullptr)
                    value += __bfloat162float(
                        all_bias[static_cast<int64_t>(expert_index) * N + output_column]);
                output[static_cast<int64_t>(output_row) * N + output_column] =
                    __float2bfloat16(value);
            }
        }
    }
#endif
}

template<int TileN>
__global__ __launch_bounds__(256) void iq2r_task_gemm_kernel(
    const opus::fp8_t* __restrict__ activations,
    const uint8_t* __restrict__ activation_scales,
    const uint8_t* __restrict__ all_data,
    const uint8_t* __restrict__ all_auxiliary,
    const int32_t* __restrict__ tasks,
    const int32_t* __restrict__ task_count,
    const __hip_bfloat16* __restrict__ all_bias,
    __hip_bfloat16* __restrict__ output,
    int M,
    int N,
    int K,
    int expert_count,
    int data_bytes,
    int auxiliary_bytes)
{
#if defined(__gfx950__)
    constexpr int kThreads = 256;
    constexpr int kWaves = kThreads / 64;
    constexpr int kAtomsPerWave = TileN / (kWaves * 16);
    static_assert(TileN == 64 || TileN == 128);

    const int task_index = blockIdx.y;
    if(task_index >= task_count[0])
        return;
    const int row_begin = tasks[task_index * 3];
    const int row_count = tasks[task_index * 3 + 1];
    const int expert_index = tasks[task_index * 3 + 2];
    if(row_begin < 0 || row_count <= 0 || row_begin + row_count > M ||
       expert_index < 0 || expert_index >= expert_count)
        return;

    __shared__ alignas(16) uint8_t codebook[kCodebookBytes];
    const uint8_t* data = all_data + static_cast<int64_t>(expert_index) * data_bytes;
    const uint8_t* auxiliary =
        all_auxiliary + static_cast<int64_t>(expert_index) * auxiliary_bytes;
    for(int byte = threadIdx.x * 16; byte < kCodebookBytes; byte += kThreads * 16)
        *reinterpret_cast<uint4*>(codebook + byte) =
            *reinterpret_cast<const uint4*>(auxiliary + byte);
    __syncthreads();

    const int wave = threadIdx.x / 64;
    const int lane = threadIdx.x % 64;
    const int lane_row = lane % 16;
    const int lane_group = lane / 16;
    const int n_block_base = blockIdx.x * (TileN / 16) + wave * kAtomsPerWave;
    const int k_tiles = (K + kTileK - 1) / kTileK;
    auto mma = opus::mfma<opus::fp8_t, opus::fp8_t, opus::fp32_t, 16, 16, 128>{};

    for(int row_base = row_begin; row_base < row_begin + row_count; row_base += 16)
    {
        const int row_end = min(row_base + 16, row_begin + row_count);
        const int input_row = row_base + lane_row;
        opus::vector_t<float, 4> accumulators[kAtomsPerWave] = {};
        for(int k_tile = 0; k_tile < k_tiles; ++k_tile)
        {
            union
            {
                opus::i32x8_t words;
                uint8_t bytes[32];
            } activation_fragment;
#pragma unroll
            for(int item = 0; item < 16; ++item)
            {
                const int k0 = k_tile * kTileK + lane_group * 16 + item;
                const int k1 = k0 + 64;
                activation_fragment.bytes[item] =
                    input_row < row_end && k0 < K
                        ? __builtin_bit_cast(uint8_t,
                              activations[static_cast<int64_t>(input_row) * K + k0])
                        : 0;
                activation_fragment.bytes[16 + item] =
                    input_row < row_end && k1 < K
                        ? __builtin_bit_cast(uint8_t,
                              activations[static_cast<int64_t>(input_row) * K + k1])
                        : 0;
            }
            uint32_t scale_a = 127u * 0x01010101u;
            if(input_row < row_end)
            {
                const uint32_t exponent = activation_scales[
                    static_cast<int64_t>(input_row) * (K / kScaleBlock) +
                    k_tile * 4 + lane_group];
                scale_a = exponent * 0x01010101u;
            }

            uint32_t scale_b = 0;
            opus::i32x8_t weight_fragments[kAtomsPerWave];
#pragma unroll
            for(int atom_index = 0; atom_index < kAtomsPerWave; ++atom_index)
            {
                const int n_block = n_block_base + atom_index;
                const int64_t tile = physical_tile(n_block, k_tile, k_tiles);
                const uint8_t* record = data + lane_record_offset(tile, lane);
                const uint32_t index_lows = load_u32(record);
                const uint32_t signs = load_u32(record + 4);
                const uint32_t metadata = load_u32(data + metadata_offset(tile, lane));
                const int atom = n_block % kAtomsPerTriplet;
                const uint32_t index_highs = (metadata >> (atom * 8)) & 0x0fu;
                union
                {
                    opus::i32x8_t words;
                    uint64_t codewords[4];
                } decoded;
#pragma unroll
                for(int codeword = 0; codeword < 4; ++codeword)
                {
                    const int codebook_index =
                        ((index_lows >> (codeword * 8)) & 0xffu) |
                        (((index_highs >> codeword) & 1u) << 8);
                    decoded.codewords[codeword] = apply_signs(
                        *reinterpret_cast<const uint64_t*>(
                            codebook + codebook_index * 8),
                        (signs >> (codeword * 8)) & 0xffu);
                }
                weight_fragments[atom_index] = decoded.words;
                const uint32_t base = auxiliary[kCodebookBytes + n_block];
                const uint32_t delta = (metadata >> (atom * 8 + 4)) & 0x0fu;
                scale_b |= (base + delta) << (atom_index * 8);
            }

#pragma unroll
            for(int atom_index = 0; atom_index < kAtomsPerWave; ++atom_index)
            {
                if constexpr(kAtomsPerWave == 1)
                    accumulators[atom_index] = mma(activation_fragment.words,
                                                    weight_fragments[atom_index],
                                                    accumulators[atom_index],
                                                    scale_a,
                                                    scale_b,
                                                    opus::number<0>{},
                                                    opus::number<0>{});
                else if(atom_index == 0)
                    accumulators[atom_index] = mma(activation_fragment.words,
                                                    weight_fragments[atom_index],
                                                    accumulators[atom_index],
                                                    scale_a,
                                                    scale_b,
                                                    opus::number<0>{},
                                                    opus::number<0>{});
                else
                    accumulators[atom_index] = mma(activation_fragment.words,
                                                    weight_fragments[atom_index],
                                                    accumulators[atom_index],
                                                    scale_a,
                                                    scale_b,
                                                    opus::number<0>{},
                                                    opus::number<1>{});
            }
        }

#pragma unroll
        for(int atom_index = 0; atom_index < kAtomsPerWave; ++atom_index)
        {
            const int output_column = (n_block_base + atom_index) * 16 + lane_row;
#pragma unroll
            for(int item = 0; item < 4; ++item)
            {
                const int output_row = row_base + lane_group * 4 + item;
                if(output_row < row_end && output_column < N)
                {
                    float value = accumulators[atom_index][item];
                    if(all_bias != nullptr)
                        value += __bfloat162float(all_bias[
                            static_cast<int64_t>(expert_index) * N + output_column]);
                    output[static_cast<int64_t>(output_row) * N + output_column] =
                        __float2bfloat16(value);
                }
            }
        }
    }
#endif
}

template<int Atom>
__device__ __forceinline__ void iq2r_cooperative_mfma(
    const opus::i32x8_t& activation,
    const opus::i32x8_t* weights,
    opus::vector_t<float, 4>* accumulators,
    uint32_t scale_a,
    const uint32_t* scale_b)
{
    auto mma = opus::mfma<opus::fp8_t, opus::fp8_t, opus::fp32_t, 16, 16, 128>{};
    accumulators[Atom] = mma(activation,
                             weights[Atom],
                             accumulators[Atom],
                             scale_a,
                             scale_b[Atom / 4],
                             opus::number<0>{},
                             opus::number<Atom % 4>{});
}

struct IQ2RCompressedTriplet
{
    uint4 paired;
    uint4 third;
};

__device__ __forceinline__ void iq2r_issue_direct_lds_triplet(
    opus::gmem<uint8_t>& data_buffer,
    uint8_t* weight_cache,
    int lane,
    int triplet,
    int source_base)
{
    const int scalar_base = __builtin_amdgcn_readfirstlane(source_base);
    const int cache_base = triplet * kTripletLDSBytes;
    // Use the public wrapper so Opus performs the generic-to-LDS address-space
    // conversion in device compilation while keeping the host pass valid.
    data_buffer.template async_load<16>(weight_cache + cache_base,
                                        lane * 16,
                                        scalar_base,
                                        opus::number<0>{},
                                        opus::number<2>{});
    data_buffer.template async_load<12>(weight_cache + cache_base + kAtomTwoOffset,
                                        lane * kAtomTwoRecordBytes,
                                        scalar_base + kAtomTwoOffset,
                                        opus::number<0>{},
                                        opus::number<2>{});
    asm volatile("" ::: "memory");
}

__device__ __forceinline__ IQ2RCompressedTriplet iq2r_read_direct_lds_triplet(
    const uint8_t* weight_cache, int lane, int triplet)
{
    const int cache_base = triplet * kTripletLDSBytes;
    IQ2RCompressedTriplet result;
    result.third = *reinterpret_cast<const uint4*>(
        weight_cache + cache_base + kAtomTwoOffset +
        lane * kAtomTwoLDSRecordBytes);
    result.paired = *reinterpret_cast<const uint4*>(
        weight_cache + cache_base + lane * 16);
    return result;
}

__device__ __forceinline__ uint32_t iq2r_decode_direct_lds_triplet(
    const IQ2RCompressedTriplet& compressed,
    const uint64_t* codebook,
    uint32_t packed_bases,
    opus::i32x8_t* weight_fragments)
{
    const uint32_t index_lows[kAtomsPerTriplet] = {
        compressed.paired.x, compressed.paired.z, compressed.third.x};
    const uint32_t signs[kAtomsPerTriplet] = {
        compressed.paired.y, compressed.paired.w, compressed.third.y};
    const uint32_t metadata = compressed.third.z;

#pragma unroll
    for(int local_atom = 0; local_atom < kAtomsPerTriplet; ++local_atom)
    {
        const uint32_t index_highs =
            (metadata >> (local_atom * 8)) & 0x0fu;
        union
        {
            opus::i32x8_t words;
            uint64_t codewords[4];
        } decoded;
#pragma unroll
        for(int codeword = 0; codeword < 4; ++codeword)
        {
            const int codebook_index =
                ((index_lows[local_atom] >> (codeword * 8)) & 0xffu) |
                (((index_highs >> codeword) & 1u) << 8);
            decoded.codewords[codeword] = apply_signs(
                codebook[codebook_index],
                (signs[local_atom] >> (codeword * 8)) & 0xffu);
        }
        weight_fragments[local_atom] = decoded.words;
    }
    // Valid base+delta values are at most 254, so the packed byte add cannot
    // carry into a neighboring scale.
    return packed_bases + ((metadata >> 4) & 0x000f0f0fu);
}

template<int AccumulatorBase>
__device__ __forceinline__ void iq2r_cooperative_triplet_mfma(
    const opus::i32x8_t& activation,
    const opus::i32x8_t* weights,
    opus::vector_t<float, 4>* accumulators,
    uint32_t scale_a,
    uint32_t scale_b)
{
    auto mma = opus::mfma<opus::fp8_t, opus::fp8_t, opus::fp32_t, 16, 16, 128>{};
    accumulators[AccumulatorBase] =
        mma(activation,
            weights[0],
            accumulators[AccumulatorBase],
            scale_a,
            scale_b,
            opus::number<0>{},
            opus::number<0>{});
    accumulators[AccumulatorBase + 1] =
        mma(activation,
            weights[1],
            accumulators[AccumulatorBase + 1],
            scale_a,
            scale_b,
            opus::number<0>{},
            opus::number<1>{});
    accumulators[AccumulatorBase + 2] =
        mma(activation,
            weights[2],
            accumulators[AccumulatorBase + 2],
            scale_a,
            scale_b,
            opus::number<0>{},
            opus::number<2>{});
}

template<int OutputAtoms, int PhysicalWaves>
__global__ __launch_bounds__(64 * PhysicalWaves, PhysicalWaves == 4 ? 2 : 1)
void iq2r_task_gemm_cooperative_kernel(
    const opus::fp8_t* __restrict__ activations,
    const uint8_t* __restrict__ activation_scales,
    const uint8_t* __restrict__ all_data,
    const uint8_t* __restrict__ all_auxiliary,
    const int32_t* __restrict__ tasks,
    const int32_t* __restrict__ task_count,
    const __hip_bfloat16* __restrict__ all_bias,
    __hip_bfloat16* __restrict__ output,
    int M,
    int N,
    int K,
    int expert_count,
    int data_bytes,
    int auxiliary_bytes)
{
#if defined(__gfx950__)
    static_assert(OutputAtoms == 3 || OutputAtoms == 6);
    static_assert(PhysicalWaves == 4 || PhysicalWaves == 8);
    constexpr int kOutputColumns = OutputAtoms * kTileN;
    constexpr int kDirectTriplets = OutputAtoms / kAtomsPerTriplet;
    constexpr int kStoredAtoms = OutputAtoms < 3 ? OutputAtoms : 3;

    struct SharedStorage
    {
        alignas(16) uint64_t codebook[kCodebookBytes / sizeof(uint64_t)];
        alignas(16) uint8_t
            weight_cache[PhysicalWaves][kDirectTriplets * kTripletLDSBytes];
        alignas(16) opus::vector_t<float, 4>
            partial[PhysicalWaves][kStoredAtoms][64];
    };
    __shared__ SharedStorage shared;

    const int lane = static_cast<int>(threadIdx.x);
    const int wave = static_cast<int>(threadIdx.y);
    const int linear_thread = wave * 64 + lane;
    const int linear_threads = PhysicalWaves * 64;
    const int num_tasks = task_count[0];
    if(num_tasks <= 0)
        return;
    const int n_tiles = (N + kOutputColumns - 1) / kOutputColumns;
    const int total_tiles = num_tasks * n_tiles;
    const int k_tiles = (K + kTileK - 1) / kTileK;
    const int base_iterations = k_tiles / PhysicalWaves;
    const int extra_iterations = k_tiles % PhysicalWaves;
    const int k_begin = wave * base_iterations + min(wave, extra_iterations);
    const int iterations = base_iterations + (wave < extra_iterations ? 1 : 0);
    int previous_expert = -1;

    for(int work_index = static_cast<int>(blockIdx.x); work_index < total_tiles;
        work_index += static_cast<int>(gridDim.x))
    {
        // N-major traversal lets a resident workgroup reuse an expert's
        // codebook when its grid-stride lands on another output tile for the
        // same task.  All task fields are read uniformly by the workgroup.
        int task_index;
        int n_tile_index;
        if constexpr(OutputAtoms == 6 ||
                     (OutputAtoms == 3 && PhysicalWaves == 8))
        {
            task_index = work_index % num_tasks;
            n_tile_index = work_index / num_tasks;
        }
        else
        {
            task_index = work_index / n_tiles;
            n_tile_index = work_index % n_tiles;
        }
        const int row_begin = tasks[task_index * 3];
        const int row_count = tasks[task_index * 3 + 1];
        const int expert_index = tasks[task_index * 3 + 2];
        if(row_begin < 0 || row_count <= 0 || row_begin + row_count > M ||
           expert_index < 0 || expert_index >= expert_count)
            continue;

        const uint8_t* data =
            all_data + static_cast<int64_t>(expert_index) * data_bytes;
        const uint8_t* auxiliary =
            all_auxiliary + static_cast<int64_t>(expert_index) * auxiliary_bytes;
        opus::gmem<uint8_t> data_buffer(data, static_cast<unsigned int>(data_bytes));
        if(expert_index != previous_expert)
        {
            for(int entry = linear_thread;
                entry < kCodebookBytes / static_cast<int>(sizeof(uint64_t));
                entry += linear_threads)
                shared.codebook[entry] =
                    reinterpret_cast<const uint64_t*>(auxiliary)[entry];
            __syncthreads();
            previous_expert = expert_index;
        }

        const int n_block_base = n_tile_index * OutputAtoms;
        uint32_t packed_bases[kDirectTriplets] = {};
#pragma unroll
        for(int triplet = 0; triplet < kDirectTriplets; ++triplet)
        {
#pragma unroll
            for(int atom = 0; atom < kAtomsPerTriplet; ++atom)
                packed_bases[triplet] |=
                    static_cast<uint32_t>(
                        auxiliary[kCodebookBytes + n_block_base +
                                  triplet * kAtomsPerTriplet + atom])
                    << (atom * 8);
        }
        const int row_end = row_begin + row_count;
        for(int row_base = row_begin; row_base < row_end; row_base += 16)
        {
            const int sub_end = min(row_base + 16, row_end);
            const int input_row = row_base + lane % 16;
            const int load_row = min(input_row, sub_end - 1);
            const int lane_group = lane / 16;
            opus::vector_t<float, 4> accumulators[OutputAtoms] = {};

            int next_data_base = static_cast<int>(triplet_base(
                physical_tile(n_block_base, k_begin, k_tiles)));
#pragma unroll
            for(int triplet = 0; triplet < kDirectTriplets; ++triplet)
                iq2r_issue_direct_lds_triplet(data_buffer,
                                               shared.weight_cache[wave],
                                               lane,
                                               triplet,
                                               next_data_base +
                                                   triplet * kTripletBytes);
            next_data_base += kGroupBytes;

            for(int iteration = 0; iteration < iterations; ++iteration)
            {
                const int k_tile = k_begin + iteration;
                union
                {
                    opus::i32x8_t words;
                    uint8_t bytes[32];
                } activation_fragment{};
                const int activation_k = k_tile * kTileK + lane_group * 16;
                if(activation_k + 15 < K)
                {
                    const auto* activation_row = reinterpret_cast<const uint8_t*>(
                        activations + static_cast<int64_t>(load_row) * K);
                    *reinterpret_cast<uint4*>(activation_fragment.bytes) =
                        *reinterpret_cast<const uint4*>(activation_row + activation_k);
                    if(activation_k + 64 + 15 < K)
                        *reinterpret_cast<uint4*>(activation_fragment.bytes + 16) =
                            *reinterpret_cast<const uint4*>(activation_row +
                                                           activation_k + 64);
                }

                uint32_t scale_a = 127u * 0x01010101u;
                const int scale_column = k_tile * 4 + lane_group;
                if(scale_column < K / kScaleBlock)
                {
                    const uint32_t exponent = activation_scales[
                        static_cast<int64_t>(load_row) * (K / kScaleBlock) +
                        scale_column];
                    scale_a = exponent * 0x01010101u;
                }

                // The direct-to-LDS requests precede two or three activation
                // requests.  Retire only the older weight requests first so
                // their LDS reads and IQ2R decode overlap the activation tail.
                if(k_tile + 1 < k_tiles)
                    __builtin_amdgcn_s_waitcnt(0x0F73); // vmcnt(3)
                else
                    __builtin_amdgcn_s_waitcnt(0x0F72); // vmcnt(2)

                IQ2RCompressedTriplet compressed =
                    iq2r_read_direct_lds_triplet(
                        shared.weight_cache[wave], lane, 0);
                asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
                __builtin_amdgcn_s_waitcnt(0x0F70); // vmcnt(0)
                if(iteration + 1 < iterations)
                {
                    iq2r_issue_direct_lds_triplet(data_buffer,
                                                   shared.weight_cache[wave],
                                                   lane,
                                                   0,
                                                   next_data_base);
                }
                opus::i32x8_t weight_fragments[kAtomsPerTriplet];
                uint32_t scale_b = iq2r_decode_direct_lds_triplet(
                    compressed,
                    shared.codebook,
                    packed_bases[0],
                    weight_fragments);
                iq2r_cooperative_triplet_mfma<0>(activation_fragment.words,
                                                  weight_fragments,
                                                  accumulators,
                                                  scale_a,
                                                  scale_b);
                if constexpr(OutputAtoms == 6)
                {
                    compressed = iq2r_read_direct_lds_triplet(
                        shared.weight_cache[wave], lane, 1);
                    asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
                    if(iteration + 1 < iterations)
                    {
                        iq2r_issue_direct_lds_triplet(
                            data_buffer,
                            shared.weight_cache[wave],
                            lane,
                            1,
                            next_data_base + kTripletBytes);
                        next_data_base += kGroupBytes;
                    }
                    scale_b = iq2r_decode_direct_lds_triplet(compressed,
                                                             shared.codebook,
                                                             packed_bases[1],
                                                             weight_fragments);
                    iq2r_cooperative_triplet_mfma<3>(activation_fragment.words,
                                                      weight_fragments,
                                                      accumulators,
                                                      scale_a,
                                                      scale_b);
                }
                else if(iteration + 1 < iterations)
                    next_data_base += kGroupBytes;
            }

            const int output_row = row_base + (lane / 16) * 4 + wave;
#pragma unroll
            for(int atom = 0; atom < 3; ++atom)
                shared.partial[wave][atom][lane] = accumulators[atom];
            __syncthreads();
#pragma unroll
            for(int atom = 0; atom < 3; ++atom)
            {
                const int output_column =
                    n_tile_index * kOutputColumns + atom * 16 + lane % 16;
                if(wave < 4 && output_row < sub_end && output_column < N)
                {
                    float value = 0.0f;
#pragma unroll
                    for(int source_wave = 0; source_wave < PhysicalWaves;
                        ++source_wave)
                        value += shared.partial[source_wave][atom][lane][wave];
                    if(all_bias != nullptr)
                        value += __bfloat162float(
                            all_bias[static_cast<int64_t>(expert_index) * N +
                                     output_column]);
                    output[static_cast<int64_t>(output_row) * N + output_column] =
                        __float2bfloat16(value);
                }
            }
            __syncthreads();

            if constexpr(OutputAtoms == 6)
            {
#pragma unroll
                for(int atom = 0; atom < 3; ++atom)
                    shared.partial[wave][atom][lane] = accumulators[atom + 3];
                __syncthreads();
#pragma unroll
                for(int local_atom = 0; local_atom < 3; ++local_atom)
                {
                    const int atom = local_atom + 3;
                    const int output_column =
                        n_tile_index * kOutputColumns + atom * 16 + lane % 16;
                    if(wave < 4 && output_row < sub_end && output_column < N)
                    {
                        float value = 0.0f;
#pragma unroll
                        for(int source_wave = 0; source_wave < PhysicalWaves;
                            ++source_wave)
                            value +=
                                shared.partial[source_wave][local_atom][lane][wave];
                        if(all_bias != nullptr)
                            value += __bfloat162float(
                                all_bias[static_cast<int64_t>(expert_index) * N +
                                         output_column]);
                        output[static_cast<int64_t>(output_row) * N + output_column] =
                            __float2bfloat16(value);
                    }
                }
                __syncthreads();
            }
        }
    }
#endif
}

int64_t expected_data_bytes(int64_t n, int64_t k)
{
    const int64_t n_blocks = n / kTileN;
    const int64_t physical_n_blocks = ((n_blocks + 5) / 6) * 6;
    const int64_t k_tiles = (k + kTileK - 1) / kTileK;
    return (physical_n_blocks / 6) * k_tiles * kGroupBytes;
}

int64_t expected_auxiliary_bytes(int64_t n)
{
    const int64_t n_blocks = n / kTileN;
    const int64_t physical_n_blocks = ((n_blocks + 5) / 6) * 6;
    return kCodebookBytes + physical_n_blocks + 3;
}

void validate_weights(const aiter_tensor_t& data,
                      const aiter_tensor_t& auxiliary,
                      int64_t logical_n,
                      int64_t logical_k)
{
    AITER_CHECK(data.is_gpu() && auxiliary.is_gpu(), "IQ2R weights must be GPU tensors");
    AITER_CHECK(data.device_id == auxiliary.device_id,
                "IQ2R data and auxiliary must be on the same device");
    AITER_CHECK(data.dtype() == AITER_DTYPE_u8 && auxiliary.dtype() == AITER_DTYPE_u8,
                "IQ2R data and auxiliary must be uint8");
    AITER_CHECK(data.dim() == 2 && auxiliary.dim() == 2 &&
                    data.size(0) == auxiliary.size(0),
                "IQ2R data and auxiliary must be matching [experts,bytes]");
    AITER_CHECK(data.is_contiguous() && auxiliary.is_contiguous(),
                "IQ2R data and auxiliary must be contiguous");
    AITER_CHECK(logical_n > 0 && logical_n % 16 == 0 && logical_k > 0 &&
                    logical_k % 32 == 0,
                "IQ2R requires N%16==0 and K%32==0");
    AITER_CHECK(data.size(1) == expected_data_bytes(logical_n, logical_k),
                "IQ2R data byte count does not match logical dimensions");
    AITER_CHECK(auxiliary.size(1) == expected_auxiliary_bytes(logical_n),
                "IQ2R auxiliary byte count does not match logical dimensions");
}

} // namespace

void iq2r_encode_out(const aiter_tensor_t& weight,
                     const aiter_tensor_t& importance,
                     const aiter_tensor_t& codebook,
                     aiter_tensor_t& indices,
                     aiter_tensor_t& scales,
                     aiter_tensor_t& data,
                     aiter_tensor_t& auxiliary,
                     aiter_tensor_t& scale_delta_overflow,
                     int64_t valid_k,
                     int64_t exponent_radius,
                     double codebook_max)
{
    AITER_CHECK(weight.is_gpu() && importance.is_gpu() && codebook.is_gpu() &&
                    indices.is_gpu() && scales.is_gpu() && data.is_gpu() &&
                    auxiliary.is_gpu() && scale_delta_overflow.is_gpu(),
                "IQ2R encoder requires GPU tensors");
    const int device = weight.device_id;
    AITER_CHECK(importance.device_id == device && codebook.device_id == device &&
                    indices.device_id == device && scales.device_id == device &&
                    data.device_id == device && auxiliary.device_id == device &&
                    scale_delta_overflow.device_id == device,
                "IQ2R encoder tensors must be on the same device");
    AITER_CHECK(weight.dtype() == AITER_DTYPE_fp32 &&
                    importance.dtype() == AITER_DTYPE_fp32 &&
                    codebook.dtype() == AITER_DTYPE_fp32,
                "IQ2R encoder inputs must be float32");
    AITER_CHECK(indices.dtype() == AITER_DTYPE_i16 && scales.dtype() == AITER_DTYPE_u8 &&
                    data.dtype() == AITER_DTYPE_u8 && auxiliary.dtype() == AITER_DTYPE_u8 &&
                    scale_delta_overflow.dtype() == AITER_DTYPE_i32,
                "IQ2R encoder outputs have invalid dtypes");
    AITER_CHECK(weight.dim() == 2 && importance.dim() == 1 &&
                    importance.size(0) == weight.size(1) && codebook.dim() == 2 &&
                    codebook.size(0) == 512 && codebook.size(1) == 8,
                "IQ2R encoder expects weight [N,K], importance [K], codebook [512,8]");
    AITER_CHECK(weight.is_contiguous() && importance.is_contiguous() &&
                    codebook.is_contiguous() && indices.is_contiguous() &&
                    scales.is_contiguous() && data.is_contiguous() &&
                    auxiliary.is_contiguous() && scale_delta_overflow.is_contiguous(),
                "IQ2R encoder tensors must be contiguous");
    const int64_t N = weight.size(0);
    const int64_t storage_k = weight.size(1);
    AITER_CHECK(N > 0 && N % 16 == 0 && storage_k > 0 && storage_k % 128 == 0 &&
                    valid_k > 0 && valid_k <= storage_k && valid_k % 32 == 0,
                "IQ2R encoder requires N%16==0, storage_K%128==0, valid_K%32==0");
    AITER_CHECK(exponent_radius >= 0 && exponent_radius <= 16,
                "IQ2R exponent_radius must be in [0,16]");
    const int64_t physical_n_blocks = ((N / 16 + 5) / 6) * 6;
    const int64_t k_tiles = storage_k / 128;
    const int64_t tiles = physical_n_blocks * k_tiles;
    AITER_CHECK(indices.numel() == tiles * 64 * 4 && scales.numel() == tiles * 64,
                "IQ2R encoder scratch sizes are invalid");
    AITER_CHECK(data.numel() == (physical_n_blocks / 6) * k_tiles * kGroupBytes &&
                    auxiliary.numel() == kCodebookBytes + physical_n_blocks + 3,
                "IQ2R encoder output sizes are invalid");
    AITER_CHECK(scale_delta_overflow.numel() == 1,
                "IQ2R scale_delta_overflow must have one element");

    const auto stream = getCurrentHIPStream();
    const int64_t logical_blocks = N * (valid_k / 32);
    hipLaunchKernelGGL(iq2r_encode_assign_kernel,
                       dim3(static_cast<uint32_t>(logical_blocks)),
                       dim3(64),
                       0,
                       stream,
                       static_cast<const float*>(weight.data_ptr()),
                       static_cast<const float*>(importance.data_ptr()),
                       static_cast<const float*>(codebook.data_ptr()),
                       static_cast<float>(codebook_max),
                       static_cast<int>(N),
                       static_cast<int>(storage_k),
                       static_cast<int>(valid_k),
                       static_cast<int>(exponent_radius),
                       reinterpret_cast<uint16_t*>(indices.data_ptr()),
                       static_cast<uint8_t*>(data.data_ptr()),
                       static_cast<uint8_t*>(scales.data_ptr()));
    hipLaunchKernelGGL(iq2r_encode_base_kernel,
                       dim3(static_cast<uint32_t>(physical_n_blocks)),
                       dim3(256),
                       0,
                       stream,
                       static_cast<const uint8_t*>(scales.data_ptr()),
                       static_cast<int>(N),
                       static_cast<int>(storage_k),
                       static_cast<int>(valid_k),
                       static_cast<uint8_t*>(auxiliary.data_ptr()) + kCodebookBytes);
    const int64_t triplets = tiles / kAtomsPerTriplet;
    hipLaunchKernelGGL(iq2r_encode_pack_kernel,
                       dim3(static_cast<uint32_t>((triplets * 64 + 255) / 256)),
                       dim3(256),
                       0,
                       stream,
                       reinterpret_cast<const uint16_t*>(indices.data_ptr()),
                       static_cast<const uint8_t*>(scales.data_ptr()),
                       static_cast<const uint8_t*>(auxiliary.data_ptr()) + kCodebookBytes,
                       triplets,
                       static_cast<int>(N),
                       static_cast<int>(storage_k),
                       static_cast<int>(valid_k),
                       static_cast<uint8_t*>(data.data_ptr()),
                       static_cast<int32_t*>(scale_delta_overflow.data_ptr()));
    HIP_CALL_LAUNCH(hipGetLastError());
}

void iq2r_materialize_out(const aiter_tensor_t& data,
                          const aiter_tensor_t& auxiliary,
                          aiter_tensor_t& output,
                          int64_t logical_n,
                          int64_t logical_k,
                          int64_t expert_index)
{
    validate_weights(data, auxiliary, logical_n, logical_k);
    AITER_CHECK(output.is_gpu() && output.device_id == data.device_id,
                "IQ2R materialized output must be on the same GPU");
    AITER_CHECK(output.dtype() == AITER_DTYPE_fp32 && output.dim() == 2 &&
                    output.size(0) == logical_n && output.size(1) == logical_k,
                "IQ2R materialized output must be float32 [N,K]");
    AITER_CHECK(output.is_contiguous(), "IQ2R materialized output must be contiguous");
    AITER_CHECK(expert_index >= 0 && expert_index < data.size(0),
                "IQ2R expert_index is out of range");
    const int64_t elements = logical_n * logical_k;
    constexpr int threads = 256;
    hipLaunchKernelGGL(iq2r_materialize_kernel,
                       dim3(static_cast<uint32_t>((elements + threads - 1) / threads)),
                       dim3(threads),
                       0,
                       getCurrentHIPStream(),
                       static_cast<const uint8_t*>(data.data_ptr()) +
                           expert_index * data.size(1),
                       static_cast<const uint8_t*>(auxiliary.data_ptr()) +
                           expert_index * auxiliary.size(1),
                       static_cast<float*>(output.data_ptr()),
                       static_cast<int>(logical_n),
                       static_cast<int>(logical_k));
    HIP_CALL_LAUNCH(hipGetLastError());
}

void iq2r_gemm_out(const aiter_tensor_t& activations,
                   const aiter_tensor_t& activation_scales,
                   const aiter_tensor_t& data,
                   const aiter_tensor_t& auxiliary,
                   std::optional<aiter_tensor_t> bias,
                   aiter_tensor_t& output,
                   int64_t logical_n,
                   int64_t logical_k,
                   int64_t tile_n,
                   int64_t expert_index)
{
    validate_weights(data, auxiliary, logical_n, logical_k);
    AITER_CHECK(activations.is_gpu() && activation_scales.is_gpu() && output.is_gpu(),
                "IQ2R GEMM requires GPU tensors");
    AITER_CHECK(activations.device_id == data.device_id &&
                    activation_scales.device_id == data.device_id &&
                    output.device_id == data.device_id,
                "IQ2R GEMM tensors must be on the same device");
    AITER_CHECK(activations.dtype() == AITER_DTYPE_fp8,
                "IQ2R activations must be float8_e4m3fn");
    AITER_CHECK(activation_scales.dtype() == AITER_DTYPE_u8,
                "IQ2R activation scales must be uint8 E8M0");
    AITER_CHECK(output.dtype() == AITER_DTYPE_bf16,
                "initial IQ2R GEMM output must be bfloat16");
    AITER_CHECK(activations.dim() == 2 && activations.size(1) == logical_k,
                "IQ2R activations must have shape [M,K]");
    AITER_CHECK(activation_scales.dim() == 2 &&
                    activation_scales.size(0) == activations.size(0) &&
                    activation_scales.size(1) == logical_k / kScaleBlock,
                "IQ2R activation scales must have shape [M,K/32]");
    AITER_CHECK(output.dim() == 2 && output.size(0) == activations.size(0) &&
                    output.size(1) == logical_n,
                "IQ2R output must have shape [M,N]");
    AITER_CHECK(activations.is_contiguous() && activation_scales.is_contiguous() &&
                    output.is_contiguous(),
                "IQ2R GEMM tensors must be contiguous");
    AITER_CHECK(tile_n == 64 || tile_n == 128, "IQ2R tile_n must be 64 or 128");
    AITER_CHECK(logical_n % tile_n == 0, "IQ2R logical_n must be divisible by tile_n");
    AITER_CHECK(expert_index >= 0 && expert_index < data.size(0),
                "IQ2R expert_index is out of range");

    const __hip_bfloat16* bias_pointer = nullptr;
    if(bias.has_value())
    {
        const auto& value = *bias;
        AITER_CHECK(value.is_gpu() && value.device_id == data.device_id,
                    "IQ2R bias must be on the same GPU");
        AITER_CHECK(value.dtype() == AITER_DTYPE_bf16,
                    "initial IQ2R GEMM bias must be bfloat16");
        AITER_CHECK(value.dim() == 2 && value.size(0) == data.size(0) &&
                        value.size(1) == logical_n && value.is_contiguous(),
                    "IQ2R bias must be contiguous [experts,N]");
        bias_pointer = static_cast<const __hip_bfloat16*>(value.data_ptr());
    }

    const dim3 grid(static_cast<uint32_t>(logical_n / tile_n),
                    static_cast<uint32_t>((activations.size(0) + 15) / 16));
    if(tile_n == 64)
        hipLaunchKernelGGL((iq2r_gemm_kernel<64>),
                           grid,
                           dim3(256),
                           0,
                           getCurrentHIPStream(),
                           static_cast<const opus::fp8_t*>(activations.data_ptr()),
                           static_cast<const uint8_t*>(activation_scales.data_ptr()),
                           static_cast<const uint8_t*>(data.data_ptr()),
                           static_cast<const uint8_t*>(auxiliary.data_ptr()),
                           bias_pointer,
                           static_cast<__hip_bfloat16*>(output.data_ptr()),
                           static_cast<int>(activations.size(0)),
                           static_cast<int>(logical_n),
                           static_cast<int>(logical_k),
                           static_cast<int>(data.size(1)),
                           static_cast<int>(auxiliary.size(1)),
                           static_cast<int>(expert_index));
    else
        hipLaunchKernelGGL((iq2r_gemm_kernel<128>),
                           grid,
                           dim3(256),
                           0,
                           getCurrentHIPStream(),
                           static_cast<const opus::fp8_t*>(activations.data_ptr()),
                           static_cast<const uint8_t*>(activation_scales.data_ptr()),
                           static_cast<const uint8_t*>(data.data_ptr()),
                           static_cast<const uint8_t*>(auxiliary.data_ptr()),
                           bias_pointer,
                           static_cast<__hip_bfloat16*>(output.data_ptr()),
                           static_cast<int>(activations.size(0)),
                           static_cast<int>(logical_n),
                           static_cast<int>(logical_k),
                           static_cast<int>(data.size(1)),
                           static_cast<int>(auxiliary.size(1)),
                           static_cast<int>(expert_index));
    HIP_CALL_LAUNCH(hipGetLastError());
}

void iq2r_task_gemm_out(const aiter_tensor_t& activations,
                        const aiter_tensor_t& activation_scales,
                        const aiter_tensor_t& data,
                        const aiter_tensor_t& auxiliary,
                        const aiter_tensor_t& tasks,
                        const aiter_tensor_t& task_count,
                        std::optional<aiter_tensor_t> bias,
                        aiter_tensor_t& output,
                        int64_t logical_n,
                        int64_t logical_k,
                        int64_t tile_n)
{
    validate_weights(data, auxiliary, logical_n, logical_k);
    AITER_CHECK(activations.is_gpu() && activation_scales.is_gpu() && tasks.is_gpu() &&
                    task_count.is_gpu() && output.is_gpu(),
                "IQ2R task GEMM requires GPU tensors");
    AITER_CHECK(activations.device_id == data.device_id &&
                    activation_scales.device_id == data.device_id &&
                    tasks.device_id == data.device_id && task_count.device_id == data.device_id &&
                    output.device_id == data.device_id,
                "IQ2R task GEMM tensors must be on the same device");
    AITER_CHECK(activations.dtype() == AITER_DTYPE_fp8 &&
                    activation_scales.dtype() == AITER_DTYPE_u8 &&
                    output.dtype() == AITER_DTYPE_bf16,
                "IQ2R task GEMM expects FP8 activations, uint8 scales, and BF16 output");
    AITER_CHECK(tasks.dtype() == AITER_DTYPE_i32 && task_count.dtype() == AITER_DTYPE_i32,
                "IQ2R tasks and task_count must be int32");
    AITER_CHECK(tasks.dim() == 2 && tasks.size(1) == 3 &&
                    task_count.dim() == 1 && task_count.size(0) == 1,
                "IQ2R tasks must be [capacity,3] and task_count [1]");
    AITER_CHECK(activations.dim() == 2 && activations.size(1) == logical_k &&
                    activation_scales.dim() == 2 &&
                    activation_scales.size(0) == activations.size(0) &&
                    activation_scales.size(1) == logical_k / kScaleBlock,
                "IQ2R task GEMM activation shapes are invalid");
    AITER_CHECK(output.dim() == 2 && output.size(0) == activations.size(0) &&
                    output.size(1) == logical_n,
                "IQ2R task GEMM output must be [M,N]");
    AITER_CHECK(activations.is_contiguous() && activation_scales.is_contiguous() &&
                    tasks.is_contiguous() && task_count.is_contiguous() &&
                    output.is_contiguous(),
                "IQ2R task GEMM tensors must be contiguous");
    AITER_CHECK(tile_n == 64 || tile_n == 128, "IQ2R tile_n must be 64 or 128");
    AITER_CHECK(logical_n % tile_n == 0, "IQ2R logical_n must be divisible by tile_n");

    const __hip_bfloat16* bias_pointer = nullptr;
    if(bias.has_value())
    {
        const auto& value = *bias;
        AITER_CHECK(value.is_gpu() && value.device_id == data.device_id &&
                        value.dtype() == AITER_DTYPE_bf16 && value.dim() == 2 &&
                        value.size(0) == data.size(0) && value.size(1) == logical_n &&
                        value.is_contiguous(),
                    "IQ2R bias must be contiguous BF16 [experts,N] on the same GPU");
        bias_pointer = static_cast<const __hip_bfloat16*>(value.data_ptr());
    }

    const bool gpt_oss_shape = logical_k == 2880 &&
                               (logical_n == 2880 || logical_n == 5760);
    if(gpt_oss_shape)
    {
        const int routed_m = static_cast<int>(activations.size(0));
        const int expert_count = static_cast<int>(data.size(0));
        const int active_experts = min(expert_count, routed_m);
        const int cu_count = static_cast<int>(get_num_cu_func());
        const int base_grid = 2 * cu_count;
        const bool use_wide_eight_wave_down =
            logical_n == 2880 && routed_m > 4 && routed_m <= 8;
        // GPT-OSS top-k=4 maps the production M=16 decode graph to 64 routed
        // rows.  Captured ATOM routes consistently favor the narrower 3x4
        // family here while retaining the existing 5xCU grid.
        const bool use_narrow_four_wave_down =
            logical_n == 2880 && routed_m == 64;
        const bool use_narrow = routed_m < 16 && !use_wide_eight_wave_down;
        const int output_columns = use_narrow ? 48 : 96;
        const int estimated_tiles =
            active_experts *
            ((static_cast<int>(logical_n) + output_columns - 1) / output_columns);
        int launch_grid = base_grid * (estimated_tiles > base_grid ? 2 : 1);

        if(!use_narrow && logical_n == 2880)
        {
            // Once GPT-OSS reaches 64 decoded tokens (256 routed rows at
            // top-k=4), four workgroups per CU consistently outperform the
            // previous five-CU multiplier across uniform and concentrated
            // expert distributions. Keep the decode-tuned launch below that
            // boundary.
            launch_grid = (routed_m >= 256 ? 4 : 5) * cu_count;
        }

        // Profiling-only launch controls used to tune GPT-OSS decode shapes.
        // They are read on the host before launch, so separate CUDA/HIP graphs
        // can capture different candidates without rebuilding this module. An
        // unset variable preserves the production heuristic above.
        const char* family_override = std::getenv(
            logical_n == 5760 ? "IQ2R_GEMM_GATE_UP_FAMILY"
                              : "IQ2R_GEMM_DOWN_FAMILY");
        const char* grid_override = std::getenv(
            logical_n == 5760 ? "IQ2R_GEMM_GATE_UP_GRID_MULTIPLIER"
                              : "IQ2R_GEMM_DOWN_GRID_MULTIPLIER");
        if(grid_override != nullptr && grid_override[0] != '\0')
        {
            const int multiplier = std::atoi(grid_override);
            AITER_CHECK(multiplier >= 1 && multiplier <= 8,
                        "IQ2R grid multiplier override must be in [1,8]");
            launch_grid = multiplier * cu_count;
        }

#define IQ2R_LAUNCH_COOPERATIVE(OUTPUT_ATOMS, PHYSICAL_WAVES)                 \
    hipLaunchKernelGGL(                                                        \
        (iq2r_task_gemm_cooperative_kernel<OUTPUT_ATOMS, PHYSICAL_WAVES>),    \
        dim3(static_cast<uint32_t>(launch_grid)),                              \
        dim3(64, PHYSICAL_WAVES),                                              \
        0,                                                                     \
        getCurrentHIPStream(),                                                 \
        static_cast<const opus::fp8_t*>(activations.data_ptr()),               \
        static_cast<const uint8_t*>(activation_scales.data_ptr()),             \
        static_cast<const uint8_t*>(data.data_ptr()),                          \
        static_cast<const uint8_t*>(auxiliary.data_ptr()),                     \
        static_cast<const int32_t*>(tasks.data_ptr()),                         \
        static_cast<const int32_t*>(task_count.data_ptr()),                    \
        bias_pointer,                                                          \
        static_cast<__hip_bfloat16*>(output.data_ptr()),                       \
        routed_m,                                                              \
        static_cast<int>(logical_n),                                           \
        static_cast<int>(logical_k),                                           \
        expert_count,                                                          \
        static_cast<int>(data.size(1)),                                        \
        static_cast<int>(auxiliary.size(1)))

        if(family_override != nullptr && family_override[0] != '\0')
        {
            if(std::strcmp(family_override, "3x4") == 0)
                IQ2R_LAUNCH_COOPERATIVE(3, 4);
            else if(std::strcmp(family_override, "3x8") == 0)
                IQ2R_LAUNCH_COOPERATIVE(3, 8);
            else if(std::strcmp(family_override, "6x4") == 0)
                IQ2R_LAUNCH_COOPERATIVE(6, 4);
            else if(std::strcmp(family_override, "6x8") == 0)
                IQ2R_LAUNCH_COOPERATIVE(6, 8);
            else
                AITER_CHECK(false,
                            "IQ2R launch family override must be one of "
                            "3x4, 3x8, 6x4, or 6x8");
        }
        else if(use_narrow)
        {
            if(routed_m <= 4)
                IQ2R_LAUNCH_COOPERATIVE(3, 8);
            else
                IQ2R_LAUNCH_COOPERATIVE(3, 4);
        }
        else if(use_wide_eight_wave_down)
        {
            // The GPT-OSS down projection benefits from the wider output tile
            // while still splitting its 23 K tiles across eight physical waves.
            IQ2R_LAUNCH_COOPERATIVE(6, 8);
        }
        else if(use_narrow_four_wave_down)
        {
            IQ2R_LAUNCH_COOPERATIVE(3, 4);
        }
        else
        {
            IQ2R_LAUNCH_COOPERATIVE(6, 4);
        }
#undef IQ2R_LAUNCH_COOPERATIVE
        HIP_CALL_LAUNCH(hipGetLastError());
        return;
    }

    const dim3 grid(static_cast<uint32_t>(logical_n / tile_n),
                    static_cast<uint32_t>(tasks.size(0)));
    if(tile_n == 64)
        hipLaunchKernelGGL((iq2r_task_gemm_kernel<64>),
                           grid,
                           dim3(256),
                           0,
                           getCurrentHIPStream(),
                           static_cast<const opus::fp8_t*>(activations.data_ptr()),
                           static_cast<const uint8_t*>(activation_scales.data_ptr()),
                           static_cast<const uint8_t*>(data.data_ptr()),
                           static_cast<const uint8_t*>(auxiliary.data_ptr()),
                           static_cast<const int32_t*>(tasks.data_ptr()),
                           static_cast<const int32_t*>(task_count.data_ptr()),
                           bias_pointer,
                           static_cast<__hip_bfloat16*>(output.data_ptr()),
                           static_cast<int>(activations.size(0)),
                           static_cast<int>(logical_n),
                           static_cast<int>(logical_k),
                           static_cast<int>(data.size(0)),
                           static_cast<int>(data.size(1)),
                           static_cast<int>(auxiliary.size(1)));
    else
        hipLaunchKernelGGL((iq2r_task_gemm_kernel<128>),
                           grid,
                           dim3(256),
                           0,
                           getCurrentHIPStream(),
                           static_cast<const opus::fp8_t*>(activations.data_ptr()),
                           static_cast<const uint8_t*>(activation_scales.data_ptr()),
                           static_cast<const uint8_t*>(data.data_ptr()),
                           static_cast<const uint8_t*>(auxiliary.data_ptr()),
                           static_cast<const int32_t*>(tasks.data_ptr()),
                           static_cast<const int32_t*>(task_count.data_ptr()),
                           bias_pointer,
                           static_cast<__hip_bfloat16*>(output.data_ptr()),
                           static_cast<int>(activations.size(0)),
                           static_cast<int>(logical_n),
                           static_cast<int>(logical_k),
                           static_cast<int>(data.size(0)),
                           static_cast<int>(data.size(1)),
                           static_cast<int>(auxiliary.size(1)));
    HIP_CALL_LAUNCH(hipGetLastError());
}

} // namespace aiter
