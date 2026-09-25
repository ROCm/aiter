#include <cstdio>
// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

#include "aiter_hip_common.h"
#include "aiter_stream.h"
#include "iq2r.h"
#include "mx_quant_utils.h"
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
    if(row_begin < 0 || row_count <= 0 || row_begin + row_count > M)
        return;
    if(expert_index < 0 || expert_index >= expert_count)
    {
        // Expert-parallel callers remap non-local global routes to -1. The
        // reduction still visits every original route, so materialize exact
        // zeros for skipped tasks rather than leaving stale workspace values.
        const int output_column_begin = static_cast<int>(blockIdx.x) * TileN;
        const int task_elements = row_count * TileN;
        for(int element = threadIdx.x; element < task_elements;
            element += blockDim.x)
        {
            const int row = row_begin + element / TileN;
            const int column = output_column_begin + element % TileN;
            if(column < N)
                output[static_cast<int64_t>(row) * N + column] =
                    __float2bfloat16(0.0f);
        }
        return;
    }

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

union IQ2RActivationFragment
{
    opus::i32x8_t words;
    uint8_t bytes[32];
};

template<int N>
__device__ __forceinline__ void iq2r_wait_vmcnt()
{
    static_assert(N >= 0 && N <= 63);
    constexpr unsigned waitcnt = 0x0f70u | static_cast<unsigned>(N & 0x0f) |
                                 (static_cast<unsigned>(N & 0x30) << 10);
    __builtin_amdgcn_s_waitcnt(waitcnt);
}

template<bool TiledScales>
__device__ __forceinline__ int64_t iq2r_activation_scale_offset(
    int input_row, int scale_column, int rows, int groups_per_row)
{
    if constexpr(TiledScales)
        return (static_cast<int64_t>(scale_column / 4) * ((rows + 15) / 16) +
                input_row / 16) *
                   64 +
               (scale_column % 4) * 16 + input_row % 16;
    return static_cast<int64_t>(input_row) * groups_per_row + scale_column;
}

// Plain GLM's compiled gate/up weights are interleaved by output column.  One
// 64-column IQ2R tile therefore contains exactly one 32-value SwiGLU/MXFP8
// group.  Keeping that tile inside the GEMM workgroup removes the 8 KiB BF16
// gate/up round trip per routed row while preserving both BF16 rounding points
// used by the established two-kernel path.
template<bool TiledActivationScales>
__global__ __launch_bounds__(256) void iq2r_task_gemm_swiglu_quant_kernel(
    const opus::fp8_t* __restrict__ activations,
    const uint8_t* __restrict__ activation_scales,
    const uint8_t* __restrict__ all_data,
    const uint8_t* __restrict__ all_auxiliary,
    const int32_t* __restrict__ tasks,
    const int32_t* __restrict__ task_count,
    const __hip_bfloat16* __restrict__ all_bias,
    opus::fp8_t* __restrict__ output,
    uint8_t* __restrict__ output_scales,
    int M,
    int N,
    int K,
    int expert_count,
    int data_bytes,
    int auxiliary_bytes,
    float limit,
    float alpha,
    float up_offset)
{
#if defined(__gfx950__)
    constexpr int kThreads = 256;
    constexpr int kWaves = kThreads / 64;
    constexpr int kRawColumns = 64;
    constexpr int kOutputColumns = kRawColumns / 2;
    constexpr int kElementsPerQuantLane = 8;
    static_assert(kWaves == 4);

    struct SharedStorage
    {
        alignas(16) uint8_t codebook[kCodebookBytes];
        alignas(16) __hip_bfloat16 gate_up[16][kRawColumns];
    };
    __shared__ SharedStorage shared;

    const int num_tasks = task_count[0];
    if(num_tasks <= 0)
        return;
    const int n_tiles = N / kRawColumns;
    const int total_tiles = num_tasks * n_tiles;
    const int wave = static_cast<int>(threadIdx.x) / 64;
    const int lane = static_cast<int>(threadIdx.x) % 64;
    const int lane_row = lane % 16;
    const int lane_group = lane / 16;
    const int k_tiles = (K + kTileK - 1) / kTileK;
    const int input_scale_groups = K / kScaleBlock;
    const int output_scale_groups = N / 64;
    auto mma = opus::mfma<opus::fp8_t, opus::fp8_t, opus::fp32_t, 16, 16, 128>{};

    for(int work_index = static_cast<int>(blockIdx.x); work_index < total_tiles;
        work_index += static_cast<int>(gridDim.x))
    {
        const int task_index = work_index / n_tiles;
        const int n_tile = work_index % n_tiles;
        const int row_begin = tasks[task_index * 3];
        const int row_count = tasks[task_index * 3 + 1];
        const int expert_index = tasks[task_index * 3 + 2];
        if(row_begin < 0 || row_count <= 0 || row_begin + row_count > M)
            continue;

        if(expert_index < 0 || expert_index >= expert_count)
        {
            const int task_elements = row_count * kOutputColumns;
            for(int element = static_cast<int>(threadIdx.x); element < task_elements;
                element += kThreads)
            {
                const int row = row_begin + element / kOutputColumns;
                const int column = n_tile * kOutputColumns + element % kOutputColumns;
                output[static_cast<int64_t>(row) * (N / 2) + column] =
                    opus::fp32_to_fp8(0.0f);
            }
            if(threadIdx.x < row_count)
                output_scales[static_cast<int64_t>(row_begin + threadIdx.x) *
                                  output_scale_groups +
                              n_tile] = 127;
            continue;
        }

        const uint8_t* data =
            all_data + static_cast<int64_t>(expert_index) * data_bytes;
        const uint8_t* auxiliary =
            all_auxiliary + static_cast<int64_t>(expert_index) * auxiliary_bytes;
        for(int byte = static_cast<int>(threadIdx.x) * 16; byte < kCodebookBytes;
            byte += kThreads * 16)
            *reinterpret_cast<uint4*>(shared.codebook + byte) =
                *reinterpret_cast<const uint4*>(auxiliary + byte);
        __syncthreads();

        const int n_block = n_tile * kWaves + wave;
        for(int row_base = row_begin; row_base < row_begin + row_count;
            row_base += 16)
        {
            const int row_end = min(row_base + 16, row_begin + row_count);
            const int input_row = row_base + lane_row;
            opus::vector_t<float, 4> accumulators = {};
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
                            ? __builtin_bit_cast(
                                  uint8_t,
                                  activations[static_cast<int64_t>(input_row) * K + k0])
                            : 0;
                    activation_fragment.bytes[16 + item] =
                        input_row < row_end && k1 < K
                            ? __builtin_bit_cast(
                                  uint8_t,
                                  activations[static_cast<int64_t>(input_row) * K + k1])
                            : 0;
                }
                uint32_t scale_a = 127u * 0x01010101u;
                if(input_row < row_end)
                {
                    const int scale_column = k_tile * 4 + lane_group;
                    const uint32_t exponent = activation_scales[
                        iq2r_activation_scale_offset<TiledActivationScales>(
                            input_row,
                            scale_column,
                            M,
                            input_scale_groups)];
                    scale_a = exponent * 0x01010101u;
                }

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
                            shared.codebook + codebook_index * 8),
                        (signs >> (codeword * 8)) & 0xffu);
                }
                const uint32_t base = auxiliary[kCodebookBytes + n_block];
                const uint32_t delta = (metadata >> (atom * 8 + 4)) & 0x0fu;
                const uint32_t scale_b = base + delta;
                accumulators = mma(activation_fragment.words,
                                   decoded.words,
                                   accumulators,
                                   scale_a,
                                   scale_b,
                                   opus::number<0>{},
                                   opus::number<0>{});
            }

#pragma unroll
            for(int item = 0; item < 4; ++item)
            {
                const int local_row = lane_group * 4 + item;
                const int output_row = row_base + local_row;
                float value = accumulators[item];
                if(all_bias != nullptr && output_row < row_end)
                    value += __bfloat162float(
                        all_bias[static_cast<int64_t>(expert_index) * N +
                                 n_tile * kRawColumns + wave * 16 + lane_row]);
                shared.gate_up[local_row][wave * 16 + lane_row] =
                    output_row < row_end ? __float2bfloat16(value)
                                         : __float2bfloat16(0.0f);
            }
            __syncthreads();

            if(threadIdx.x < 64)
            {
                const int local_row = static_cast<int>(threadIdx.x) / 4;
                const int quant_lane = static_cast<int>(threadIdx.x) % 4;
                const int output_row = row_base + local_row;
                const int local_column = quant_lane * kElementsPerQuantLane;
                using bf16_vector =
                    opus::vector_t<opus::bf16_t, kElementsPerQuantLane>;
                using fp8_vector =
                    opus::vector_t<opus::fp8_t, kElementsPerQuantLane>;
                bf16_vector values;
                float abs_max = 1.0e-10f;
#pragma unroll
                for(int element = 0; element < kElementsPerQuantLane; ++element)
                {
                    const int column = local_column + element;
                    const float gate = __bfloat162float(
                        shared.gate_up[local_row][2 * column]);
                    const float up = __bfloat162float(
                        shared.gate_up[local_row][2 * column + 1]);
                    const bool clamp = limit > 0.0f;
                    const float clipped_gate =
                        clamp && gate > limit ? limit : gate;
                    const float clipped_up =
                        clamp && up < -limit
                            ? -limit
                            : (clamp && up > limit ? limit : up);
                    const float swish =
                        clipped_gate / (1.0f + __expf(-alpha * clipped_gate));
                    const __hip_bfloat16 rounded =
                        __float2bfloat16(swish * (clipped_up + up_offset));
                    values[element] = __builtin_bit_cast(opus::bf16_t, rounded);
                    abs_max =
                        fmaxf(abs_max, fabsf(__bfloat162float(rounded)));
                }
                abs_max = fmaxf(abs_max, __shfl_xor(abs_max, 1));
                abs_max = fmaxf(abs_max, __shfl_xor(abs_max, 2));
                if(output_row < row_end)
                {
                    const auto block_scale = fp_f32_to_e8m0_block_scale<
                        kDefaultMxScaleRoundMode,
                        MxDtype::FP8_E4M3>(abs_max);
                    const float inverse_scale = 1.0f / block_scale.dq_scale;
                    fp8_vector quantized;
#pragma unroll
                    for(int element = 0; element < kElementsPerQuantLane;
                        ++element)
                        quantized[element] = opus::fp32_to_fp8(
                            static_cast<float>(values[element]) * inverse_scale);
                    const int output_column =
                        n_tile * kOutputColumns + local_column;
                    *reinterpret_cast<fp8_vector*>(
                        output + static_cast<int64_t>(output_row) * (N / 2) +
                        output_column) = quantized;
                    if(quant_lane == 0)
                        output_scales[
                            static_cast<int64_t>(output_row) * output_scale_groups +
                            n_tile] = block_scale.byte;
                }
            }
            __syncthreads();
        }
    }
#endif
}

template<bool StagedScales, bool TiledScales>
__device__ __forceinline__ void iq2r_issue_buffered_activation_fragment(
    opus::gmem<uint8_t>& activation_buffer,
    opus::gmem<uint8_t>& scale_buffer,
    const uint8_t* staged_scales,
    int input_row,
    int row_base,
    int k_tile,
    int lane_group,
    int M,
    int K,
    IQ2RActivationFragment& fragment,
    uint32_t& scale_a)
{
    uint32_t exponent;
    if constexpr(StagedScales)
        exponent = staged_scales[k_tile * 64 + lane_group * 16 +
                                 input_row - row_base];
    else
    {
        const int scale_offset = iq2r_activation_scale_offset<TiledScales>(
            input_row, k_tile * 4 + lane_group, M, K / kScaleBlock);
        exponent = scale_buffer.template load<1>(scale_offset)[0];
    }
    scale_a = exponent * 0x01010101u;

    const int activation_offset =
        input_row * K + k_tile * kTileK + lane_group * 16;
    const auto lower = activation_buffer.template load<16>(activation_offset);
    const auto upper =
        activation_buffer.template load<16>(activation_offset + 64);
    *reinterpret_cast<uint4*>(fragment.bytes) =
        __builtin_bit_cast(uint4, lower);
    *reinterpret_cast<uint4*>(fragment.bytes + 16) =
        __builtin_bit_cast(uint4, upper);
    asm volatile("" ::: "memory");
}

template<int LoadAux = 2>
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
                                        opus::number<LoadAux>{});
    data_buffer.template async_load<12>(weight_cache + cache_base + kAtomTwoOffset,
                                        lane * kAtomTwoRecordBytes,
                                        scalar_base + kAtomTwoOffset,
                                        opus::number<0>{},
                                        opus::number<LoadAux>{});
    asm volatile("" ::: "memory");
}

template<int LoadAux = 2>
__device__ __forceinline__ void iq2r_issue_direct_lds_fused_first(
    opus::gmem<uint8_t>& data_buffer,
    uint8_t* weight_cache,
    int lane,
    int atom_offset,
    int source_base)
{
    const int base = __builtin_amdgcn_readfirstlane(source_base);
    if(atom_offset == 0)
        iq2r_issue_direct_lds_triplet<LoadAux>(
            data_buffer, weight_cache, lane, 0, base);
    else if(atom_offset == 1)
    {
        data_buffer.template async_load<8>(weight_cache,
                                            lane * 16,
                                            base,
                                            opus::number<kLaneRecordBytes>{},
                                            opus::number<LoadAux>{});
        data_buffer.template async_load<12>(weight_cache + kAtomTwoOffset,
                                             lane * kAtomTwoRecordBytes,
                                             base + kAtomTwoOffset,
                                             opus::number<0>{},
                                             opus::number<LoadAux>{});
    }
    else
        data_buffer.template async_load<12>(weight_cache + kAtomTwoOffset,
                                             lane * kAtomTwoRecordBytes,
                                             base + kAtomTwoOffset,
                                             opus::number<0>{},
                                             opus::number<LoadAux>{});
    asm volatile("" ::: "memory");
}

template<int LoadAux = 2>
__device__ __forceinline__ void iq2r_issue_direct_lds_fused_second(
    opus::gmem<uint8_t>& data_buffer,
    uint8_t* weight_cache,
    int lane,
    int atom_offset,
    int source_base)
{
    constexpr int cache_base = kTripletLDSBytes;
    const int base = __builtin_amdgcn_readfirstlane(source_base);
    if(atom_offset == 0)
    {
        data_buffer.template async_load<8>(weight_cache + cache_base,
                                            lane * 16,
                                            base,
                                            opus::number<0>{},
                                            opus::number<LoadAux>{});
        data_buffer.template async_load<4>(
            weight_cache + cache_base + kAtomTwoOffset,
            lane * kAtomTwoRecordBytes,
            base + kAtomTwoOffset,
            opus::number<kAtomTwoMetadata>{},
            opus::number<LoadAux>{});
    }
    else if(atom_offset == 1)
    {
        data_buffer.template async_load<16>(weight_cache + cache_base,
                                             lane * 16,
                                             base,
                                             opus::number<0>{},
                                             opus::number<LoadAux>{});
        data_buffer.template async_load<4>(
            weight_cache + cache_base + kAtomTwoOffset,
            lane * kAtomTwoRecordBytes,
            base + kAtomTwoOffset,
            opus::number<kAtomTwoMetadata>{},
            opus::number<LoadAux>{});
    }
    else
        iq2r_issue_direct_lds_triplet<LoadAux>(
            data_buffer, weight_cache, lane, 1, base);
    asm volatile("" ::: "memory");
}

template<int LoadAux = 2>
__device__ __forceinline__ void iq2r_issue_direct_lds_fused_atoms(
    opus::gmem<uint8_t>& data_buffer,
    uint8_t* weight_cache,
    int lane,
    int atom_offset,
    int first_source_base,
    int second_source_base)
{
    iq2r_issue_direct_lds_fused_first<LoadAux>(
        data_buffer, weight_cache, lane, atom_offset, first_source_base);
    iq2r_issue_direct_lds_fused_second<LoadAux>(
        data_buffer, weight_cache, lane, atom_offset, second_source_base);
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

template<int Atom>
__device__ __forceinline__ void iq2r_decode_direct_lds_atom(
    const IQ2RCompressedTriplet& compressed,
    const uint64_t* codebook,
    uint32_t packed_bases,
    opus::i32x8_t& weight_fragment,
    uint32_t& scale_b)
{
    static_assert(Atom >= 0 && Atom < kAtomsPerTriplet);
    const uint32_t index_lows = Atom == 0 ? compressed.paired.x
                                : Atom == 1 ? compressed.paired.z
                                            : compressed.third.x;
    const uint32_t signs = Atom == 0 ? compressed.paired.y
                           : Atom == 1 ? compressed.paired.w
                                       : compressed.third.y;
    const uint32_t metadata = compressed.third.z;
    const uint32_t index_highs = (metadata >> (Atom * 8)) & 0x0fu;
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
        decoded.codewords[codeword] =
            apply_signs(codebook[codebook_index],
                        (signs >> (codeword * 8)) & 0xffu);
    }
    weight_fragment = decoded.words;
    const uint32_t base = (packed_bases >> (Atom * 8)) & 0xffu;
    const uint32_t delta = (metadata >> (Atom * 8 + 4)) & 0x0fu;
    scale_b = base + delta;
}

template<bool PinToAgpr>
__device__ __forceinline__ void
iq2r_pin_accumulator(opus::vector_t<float, 4>& accumulator)
{
    if constexpr(PinToAgpr)
        asm volatile("" : "+a"(accumulator));
}

template<int AccumulatorBase, bool PinToAgpr = false>
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
    iq2r_pin_accumulator<PinToAgpr>(accumulators[AccumulatorBase]);
    accumulators[AccumulatorBase + 1] =
        mma(activation,
            weights[1],
            accumulators[AccumulatorBase + 1],
            scale_a,
            scale_b,
            opus::number<0>{},
            opus::number<1>{});
    iq2r_pin_accumulator<PinToAgpr>(accumulators[AccumulatorBase + 1]);
    accumulators[AccumulatorBase + 2] =
        mma(activation,
            weights[2],
            accumulators[AccumulatorBase + 2],
            scale_a,
            scale_b,
            opus::number<0>{},
            opus::number<2>{});
    iq2r_pin_accumulator<PinToAgpr>(accumulators[AccumulatorBase + 2]);
}

template<int Accumulator, int ScaleSlot = 0>
__device__ __forceinline__ void iq2r_cooperative_single_mfma(
    const opus::i32x8_t& activation,
    const opus::i32x8_t& weight,
    opus::vector_t<float, 4>* accumulators,
    uint32_t scale_a,
    uint32_t scale_b)
{
    auto mma = opus::mfma<opus::fp8_t, opus::fp8_t, opus::fp32_t, 16, 16, 128>{};
    accumulators[Accumulator] = mma(activation,
                                    weight,
                                    accumulators[Accumulator],
                                    scale_a,
                                    scale_b,
                                    opus::number<0>{},
                                    opus::number<ScaleSlot>{});
}

__device__ __forceinline__ IQ2RCompressedTriplet
iq2r_load_compact_triplet(const uint8_t* data, int source_base, int lane)
{
    IQ2RCompressedTriplet result;
    result.paired = *reinterpret_cast<const uint4*>(
        data + source_base + lane * 16);
    const uint8_t* third = data + source_base + kAtomTwoOffset +
                           lane * kAtomTwoRecordBytes;
    result.third.x = load_u32(third);
    result.third.y = load_u32(third + 4);
    result.third.z = load_u32(third + 8);
    result.third.w = 0;
    return result;
}

// GLM TP8 gate diagnostic: preserve the established eight-wave K partition
// and 48-column output tile while reusing each decoded IQ2R triplet across two
// independent 16-row activation slabs.
template<int PhysicalWaves>
__global__ __launch_bounds__(64 * PhysicalWaves, PhysicalWaves == 2 ? 4 : 1)
void iq2r_task_gemm_cooperative_m32_kernel(
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
    static_assert(PhysicalWaves == 2 || PhysicalWaves == 4 || PhysicalWaves == 8);
    constexpr int kMAtoms = 2;
    constexpr int kOutputAtoms = 3;
    constexpr int kOutputColumns = kOutputAtoms * kTileN;

    struct SharedStorage
    {
        alignas(16) uint64_t codebook[kCodebookBytes / sizeof(uint64_t)];
        alignas(16) uint8_t
            weight_cache[PhysicalWaves][kTripletLDSBytes];
        alignas(16) opus::vector_t<float, 4>
            partial[PhysicalWaves][kOutputAtoms][64];
    };
    __shared__ SharedStorage shared;

    const int lane = static_cast<int>(threadIdx.x);
    const int wave = static_cast<int>(threadIdx.y);
    const int linear_thread = wave * 64 + lane;
    const int linear_threads = PhysicalWaves * 64;
    const int lane_row = lane % 16;
    const int lane_group = lane / 16;
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
        const int task_index = work_index % num_tasks;
        const int n_tile_index = work_index / num_tasks;
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

        const int n_block_base = n_tile_index * kOutputAtoms;
        uint32_t packed_bases = 0;
#pragma unroll
        for(int atom = 0; atom < kOutputAtoms; ++atom)
            packed_bases |=
                static_cast<uint32_t>(
                    auxiliary[kCodebookBytes + n_block_base + atom])
                << (atom * 8);

        const int row_end = row_begin + row_count;
        for(int row_base = row_begin; row_base < row_end;
            row_base += kMAtoms * kTileN)
        {
            const int sub_end = min(row_base + kMAtoms * kTileN, row_end);
            opus::vector_t<float, 4>
                accumulators[kMAtoms][kOutputAtoms] = {};
            int next_data_base = static_cast<int>(triplet_base(
                physical_tile(n_block_base, k_begin, k_tiles)));
            iq2r_issue_direct_lds_triplet<0>(data_buffer,
                                              shared.weight_cache[wave],
                                              lane,
                                              0,
                                              next_data_base);
            next_data_base += kGroupBytes;

            for(int iteration = 0; iteration < iterations; ++iteration)
            {
                const int k_tile = k_begin + iteration;
                IQ2RActivationFragment activation_fragments[kMAtoms] = {};
                uint32_t scale_a[kMAtoms];
#pragma unroll
                for(int m_atom = 0; m_atom < kMAtoms; ++m_atom)
                {
                    const int input_row = min(row_base + m_atom * kTileN +
                                                  lane_row,
                                              sub_end - 1);
                    const int activation_k =
                        k_tile * kTileK + lane_group * 16;
                    const auto* activation_row = reinterpret_cast<const uint8_t*>(
                        activations + static_cast<int64_t>(input_row) * K);
                    *reinterpret_cast<uint4*>(activation_fragments[m_atom].bytes) =
                        *reinterpret_cast<const uint4*>(activation_row +
                                                        activation_k);
                    *reinterpret_cast<uint4*>(
                        activation_fragments[m_atom].bytes + 16) =
                        *reinterpret_cast<const uint4*>(activation_row +
                                                        activation_k + 64);
                    const uint32_t exponent = activation_scales[
                        static_cast<int64_t>(input_row) * (K / kScaleBlock) +
                        k_tile * 4 + lane_group];
                    scale_a[m_atom] = exponent * 0x01010101u;
                }

                iq2r_wait_vmcnt<3 * kMAtoms>();
                const IQ2RCompressedTriplet compressed =
                    iq2r_read_direct_lds_triplet(
                        shared.weight_cache[wave], lane, 0);
                asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
                iq2r_wait_vmcnt<0>();
                if(iteration + 1 < iterations)
                {
                    iq2r_issue_direct_lds_triplet<0>(data_buffer,
                                                      shared.weight_cache[wave],
                                                      lane,
                                                      0,
                                                      next_data_base);
                    next_data_base += kGroupBytes;
                }

                opus::i32x8_t weight_fragments[kOutputAtoms];
                const uint32_t scale_b = iq2r_decode_direct_lds_triplet(
                    compressed,
                    shared.codebook,
                    packed_bases,
                    weight_fragments);
#pragma unroll
                for(int m_atom = 0; m_atom < kMAtoms; ++m_atom)
                    iq2r_cooperative_triplet_mfma<0>(
                        activation_fragments[m_atom].words,
                        weight_fragments,
                        accumulators[m_atom],
                        scale_a[m_atom],
                        scale_b);
            }

#pragma unroll
            for(int m_atom = 0; m_atom < kMAtoms; ++m_atom)
            {
#pragma unroll
                for(int atom = 0; atom < kOutputAtoms; ++atom)
                    shared.partial[wave][atom][lane] =
                        accumulators[m_atom][atom];
                __syncthreads();
#pragma unroll
                for(int atom = 0; atom < kOutputAtoms; ++atom)
                {
                    const int output_column =
                        n_tile_index * kOutputColumns + atom * kTileN + lane_row;
                    if constexpr(PhysicalWaves == 2)
                    {
#pragma unroll
                        for(int writer = 0; writer < 2; ++writer)
                        {
                            const int row_component =
                                wave + writer * PhysicalWaves;
                            const int output_row =
                                row_base + m_atom * kTileN +
                                lane_group * 4 + row_component;
                            if(output_row < sub_end && output_column < N)
                            {
                                const float* p0 = reinterpret_cast<const float*>(
                                    &shared.partial[0][atom][lane]);
                                const float* p1 = reinterpret_cast<const float*>(
                                    &shared.partial[1][atom][lane]);
                                float value =
                                    p0[row_component] + p1[row_component];
                                if(all_bias != nullptr)
                                    value += __bfloat162float(all_bias[
                                        static_cast<int64_t>(expert_index) * N +
                                        output_column]);
                                output[static_cast<int64_t>(output_row) * N +
                                       output_column] = __float2bfloat16(value);
                            }
                        }
                    }
                    else
                    {
                        const int output_row = row_base + m_atom * kTileN +
                                               lane_group * 4 + wave;
                        if(wave < 4 && output_row < sub_end && output_column < N)
                        {
                            float value = 0.0f;
#pragma unroll
                            for(int source_wave = 0;
                                source_wave < PhysicalWaves;
                                ++source_wave)
                                value += shared.partial[source_wave][atom]
                                                       [lane][wave];
                            if(all_bias != nullptr)
                                value += __bfloat162float(all_bias[
                                    static_cast<int64_t>(expert_index) * N +
                                    output_column]);
                            output[static_cast<int64_t>(output_row) * N +
                                   output_column] = __float2bfloat16(value);
                        }
                    }
                }
                __syncthreads();
            }
        }
    }
#endif
}

// E063 experimental token-major gate; baseline core above is unchanged.
template<int PhysicalWaves>
__global__ __launch_bounds__(64 * PhysicalWaves, PhysicalWaves == 2 ? 4 : 1)
void iq2r_task_gemm_indexed_m32_kernel(
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
    int auxiliary_bytes,
    const int32_t* __restrict__ gather_indices)
{
#if defined(__gfx950__)
    static_assert(PhysicalWaves == 2 || PhysicalWaves == 8);
    constexpr int kMAtoms = 2;
    constexpr int kOutputAtoms = 3;
    constexpr int kOutputColumns = kOutputAtoms * kTileN;

    struct SharedStorage
    {
        alignas(16) uint64_t codebook[kCodebookBytes / sizeof(uint64_t)];
        alignas(16) uint8_t
            weight_cache[PhysicalWaves][kTripletLDSBytes];
        alignas(16) opus::vector_t<float, 4>
            partial[PhysicalWaves][kOutputAtoms][64];
    };
    __shared__ SharedStorage shared;

    const int lane = static_cast<int>(threadIdx.x);
    const int wave = static_cast<int>(threadIdx.y);
    const int linear_thread = wave * 64 + lane;
    const int linear_threads = PhysicalWaves * 64;
    const int lane_row = lane % 16;
    const int lane_group = lane / 16;
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
        const int task_index = work_index % num_tasks;
        const int n_tile_index = work_index / num_tasks;
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

        const int n_block_base = n_tile_index * kOutputAtoms;
        uint32_t packed_bases = 0;
#pragma unroll
        for(int atom = 0; atom < kOutputAtoms; ++atom)
            packed_bases |=
                static_cast<uint32_t>(
                    auxiliary[kCodebookBytes + n_block_base + atom])
                << (atom * 8);

        const int row_end = row_begin + row_count;
        for(int row_base = row_begin; row_base < row_end;
            row_base += kMAtoms * kTileN)
        {
            const int sub_end = min(row_base + kMAtoms * kTileN, row_end);
            // Route permutation is trusted sorter output. Read it once per
            // row slab, outside the K loop; output remains in routed-row order.
            int source_rows[kMAtoms];
#pragma unroll
            for(int m_atom = 0; m_atom < kMAtoms; ++m_atom)
            {
                const int route_row = min(row_base + m_atom * kTileN + lane_row,
                                          sub_end - 1);
                source_rows[m_atom] = gather_indices[route_row] / 9;
            }
            opus::vector_t<float, 4>
                accumulators[kMAtoms][kOutputAtoms] = {};
            int next_data_base = static_cast<int>(triplet_base(
                physical_tile(n_block_base, k_begin, k_tiles)));
            iq2r_issue_direct_lds_triplet<0>(data_buffer,
                                              shared.weight_cache[wave],
                                              lane,
                                              0,
                                              next_data_base);
            next_data_base += kGroupBytes;

            for(int iteration = 0; iteration < iterations; ++iteration)
            {
                const int k_tile = k_begin + iteration;
                IQ2RActivationFragment activation_fragments[kMAtoms] = {};
                uint32_t scale_a[kMAtoms];
#pragma unroll
                for(int m_atom = 0; m_atom < kMAtoms; ++m_atom)
                {
                    const int input_row = source_rows[m_atom];
                    const int activation_k =
                        k_tile * kTileK + lane_group * 16;
                    const auto* activation_row = reinterpret_cast<const uint8_t*>(
                        activations + static_cast<int64_t>(input_row) * K);
                    *reinterpret_cast<uint4*>(activation_fragments[m_atom].bytes) =
                        *reinterpret_cast<const uint4*>(activation_row +
                                                        activation_k);
                    *reinterpret_cast<uint4*>(
                        activation_fragments[m_atom].bytes + 16) =
                        *reinterpret_cast<const uint4*>(activation_row +
                                                        activation_k + 64);
                    const uint32_t exponent = activation_scales[
                        static_cast<int64_t>(input_row) * (K / kScaleBlock) +
                        k_tile * 4 + lane_group];
                    scale_a[m_atom] = exponent * 0x01010101u;
                }

                iq2r_wait_vmcnt<3 * kMAtoms>();
                const IQ2RCompressedTriplet compressed =
                    iq2r_read_direct_lds_triplet(
                        shared.weight_cache[wave], lane, 0);
                asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
                iq2r_wait_vmcnt<0>();
                if(iteration + 1 < iterations)
                {
                    iq2r_issue_direct_lds_triplet<0>(data_buffer,
                                                      shared.weight_cache[wave],
                                                      lane,
                                                      0,
                                                      next_data_base);
                    next_data_base += kGroupBytes;
                }

                opus::i32x8_t weight_fragments[kOutputAtoms];
                const uint32_t scale_b = iq2r_decode_direct_lds_triplet(
                    compressed,
                    shared.codebook,
                    packed_bases,
                    weight_fragments);
#pragma unroll
                for(int m_atom = 0; m_atom < kMAtoms; ++m_atom)
                    iq2r_cooperative_triplet_mfma<0>(
                        activation_fragments[m_atom].words,
                        weight_fragments,
                        accumulators[m_atom],
                        scale_a[m_atom],
                        scale_b);
            }

#pragma unroll
            for(int m_atom = 0; m_atom < kMAtoms; ++m_atom)
            {
#pragma unroll
                for(int atom = 0; atom < kOutputAtoms; ++atom)
                    shared.partial[wave][atom][lane] =
                        accumulators[m_atom][atom];
                __syncthreads();
#pragma unroll
                for(int atom = 0; atom < kOutputAtoms; ++atom)
                {
                    const int output_column =
                        n_tile_index * kOutputColumns + atom * kTileN + lane_row;
                    if constexpr(PhysicalWaves == 2)
                    {
#pragma unroll
                        for(int writer = 0; writer < 2; ++writer)
                        {
                            const int row_component =
                                wave + writer * PhysicalWaves;
                            const int output_row =
                                row_base + m_atom * kTileN +
                                lane_group * 4 + row_component;
                            if(output_row < sub_end && output_column < N)
                            {
                                const float* p0 = reinterpret_cast<const float*>(
                                    &shared.partial[0][atom][lane]);
                                const float* p1 = reinterpret_cast<const float*>(
                                    &shared.partial[1][atom][lane]);
                                float value =
                                    p0[row_component] + p1[row_component];
                                if(all_bias != nullptr)
                                    value += __bfloat162float(all_bias[
                                        static_cast<int64_t>(expert_index) * N +
                                        output_column]);
                                output[static_cast<int64_t>(output_row) * N +
                                       output_column] = __float2bfloat16(value);
                            }
                        }
                    }
                    else
                    {
                        const int output_row = row_base + m_atom * kTileN +
                                               lane_group * 4 + wave;
                        if(wave < 4 && output_row < sub_end && output_column < N)
                        {
                            float value = 0.0f;
#pragma unroll
                            for(int source_wave = 0;
                                source_wave < PhysicalWaves;
                                ++source_wave)
                                value += shared.partial[source_wave][atom]
                                                       [lane][wave];
                            if(all_bias != nullptr)
                                value += __bfloat162float(all_bias[
                                    static_cast<int64_t>(expert_index) * N +
                                    output_column]);
                            output[static_cast<int64_t>(output_row) * N +
                                   output_column] = __float2bfloat16(value);
                        }
                    }
                }
                __syncthreads();
            }
        }
    }
#endif
}

// Large routed-M diagnostic family.  Multiple M atoms share each compact IQ2R
// weight triplet, so every global weight record and codebook lookup feeds four
// independent 16x16 output tiles instead of being reloaded for four serial
// row slabs. The counted wait staircase retires the direct-to-LDS weight
// transfer while activation loads remain outstanding, then keeps the next
// triplet in flight across the current tile's decode and MFMAs.
template<int MAtoms, bool NMajor>
__global__ __launch_bounds__(256, MAtoms == 2 ? 2 : 1)
void iq2r_task_gemm_large_m_x192_kernel(
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
    constexpr int kPhysicalWaves = 4;
    static_assert(MAtoms == 2 || MAtoms == 4);
    constexpr int kMAtoms = MAtoms;
    constexpr int kNAtomsPerWave = 3;
    constexpr int kOutputColumns =
        kPhysicalWaves * kNAtomsPerWave * kTileN;

    struct SharedStorage
    {
        alignas(16) uint64_t codebook[kCodebookBytes / sizeof(uint64_t)];
        alignas(16) uint8_t weight_cache[kPhysicalWaves][kTripletLDSBytes];
    };
    __shared__ SharedStorage shared;

    const int lane = static_cast<int>(threadIdx.x);
    const int wave = static_cast<int>(threadIdx.y);
    const int linear_thread = wave * 64 + lane;
    const int lane_row = lane % 16;
    const int lane_group = lane / 16;
    const int num_tasks = task_count[0];
    if(num_tasks <= 0)
        return;

    const int n_tiles = (N + kOutputColumns - 1) / kOutputColumns;
    const int total_tiles = num_tasks * n_tiles;
    const int k_tiles = (K + kTileK - 1) / kTileK;
    int previous_expert = -1;

    for(int work_index = static_cast<int>(blockIdx.x); work_index < total_tiles;
        work_index += static_cast<int>(gridDim.x))
    {
        const int task_index =
            NMajor ? work_index % num_tasks : work_index / n_tiles;
        const int n_tile_index =
            NMajor ? work_index / num_tasks : work_index % n_tiles;
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
            // Protect readers of the old expert before reusing the codebook.
            if(previous_expert >= 0)
                __syncthreads();
            for(int entry = linear_thread;
                entry < kCodebookBytes / static_cast<int>(sizeof(uint64_t));
                entry += 256)
                shared.codebook[entry] =
                    reinterpret_cast<const uint64_t*>(auxiliary)[entry];
            __syncthreads();
            previous_expert = expert_index;
        }

        const int n_block_base =
            n_tile_index * (kOutputColumns / kTileN) +
            wave * kNAtomsPerWave;
        uint32_t packed_bases = 0;
#pragma unroll
        for(int atom = 0; atom < kNAtomsPerWave; ++atom)
            packed_bases |=
                static_cast<uint32_t>(
                    auxiliary[kCodebookBytes + n_block_base + atom])
                << (atom * 8);

        const int row_end = row_begin + row_count;
        for(int row_base = row_begin; row_base < row_end;
            row_base += kMAtoms * 16)
        {
            const int sub_end = min(row_base + kMAtoms * 16, row_end);
            opus::vector_t<float, 4>
                accumulators[kMAtoms][kNAtomsPerWave] = {};

            int next_data_base = static_cast<int>(triplet_base(
                physical_tile(n_block_base, 0, k_tiles)));
            iq2r_issue_direct_lds_triplet(data_buffer,
                                           shared.weight_cache[wave],
                                           lane,
                                           0,
                                           next_data_base);
            next_data_base += kGroupBytes;

            for(int k_tile = 0; k_tile < k_tiles; ++k_tile)
            {
                union ActivationFragment
                {
                    opus::i32x8_t words;
                    uint8_t bytes[32];
                } activation_fragments[kMAtoms] = {};
                uint32_t scale_a[kMAtoms];
#pragma unroll
                for(int m_atom = 0; m_atom < kMAtoms; ++m_atom)
                    scale_a[m_atom] = 127u * 0x01010101u;

#pragma unroll
                for(int m_atom = 0; m_atom < kMAtoms; ++m_atom)
                {
                    const int input_row = row_base + m_atom * 16 + lane_row;
                    if(input_row < sub_end)
                    {
                        const int activation_k =
                            k_tile * kTileK + lane_group * 16;
                        const auto* activation_row =
                            reinterpret_cast<const uint8_t*>(
                                activations + static_cast<int64_t>(input_row) * K);
                        if(activation_k + 15 < K)
                            *reinterpret_cast<uint4*>(
                                activation_fragments[m_atom].bytes) =
                                *reinterpret_cast<const uint4*>(
                                    activation_row + activation_k);
                        if(activation_k + 64 + 15 < K)
                            *reinterpret_cast<uint4*>(
                                activation_fragments[m_atom].bytes + 16) =
                                *reinterpret_cast<const uint4*>(
                                    activation_row + activation_k + 64);

                        const int scale_column = k_tile * 4 + lane_group;
                        if(scale_column < K / kScaleBlock)
                        {
                            const uint32_t exponent = activation_scales[
                                static_cast<int64_t>(input_row) *
                                    (K / kScaleBlock) +
                                scale_column];
                            scale_a[m_atom] = exponent * 0x01010101u;
                        }
                    }
                }

                // Two weight requests precede eight activation-vector and four
                // activation-scale requests on full K tiles. Wait only for the
                // older weight pair, leaving that activation tail in flight
                // while the compact record moves out of LDS. The padded final
                // K tile omits the second vector load and two scale groups.
                // Partial slabs issue fewer loads than the fixed wait allowance.
                if(sub_end - row_base < kMAtoms * 16)
                    __builtin_amdgcn_s_waitcnt(0x0F70);
                else if(k_tile + 1 < k_tiles)
                    __builtin_amdgcn_s_waitcnt(0x0F7C); // vmcnt(12)
                else
                    __builtin_amdgcn_s_waitcnt(0x0F78); // vmcnt(8)
                const IQ2RCompressedTriplet compressed =
                    iq2r_read_direct_lds_triplet(
                        shared.weight_cache[wave], lane, 0);
                asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
                const bool has_next = k_tile + 1 < k_tiles;
                if(has_next)
                {
                    iq2r_issue_direct_lds_triplet(data_buffer,
                                                   shared.weight_cache[wave],
                                                   lane,
                                                   0,
                                                   next_data_base);
                    next_data_base += kGroupBytes;
                }
                opus::i32x8_t weight_fragments[kNAtomsPerWave];
                const uint32_t scale_b = iq2r_decode_direct_lds_triplet(
                    compressed,
                    shared.codebook,
                    packed_bases,
                    weight_fragments);
                if(has_next)
                    __builtin_amdgcn_s_waitcnt(0x0F72); // vmcnt(2)
                else
                    __builtin_amdgcn_s_waitcnt(0x0F70); // vmcnt(0)
#pragma unroll
                for(int m_atom = 0; m_atom < kMAtoms; ++m_atom)
                    iq2r_cooperative_triplet_mfma<0>(
                        activation_fragments[m_atom].words,
                        weight_fragments,
                        accumulators[m_atom],
                        scale_a[m_atom],
                        scale_b);
            }

#pragma unroll
            for(int m_atom = 0; m_atom < kMAtoms; ++m_atom)
            {
#pragma unroll
                for(int atom = 0; atom < kNAtomsPerWave; ++atom)
                {
                    const int output_column =
                        (n_block_base + atom) * kTileN + lane_row;
#pragma unroll
                    for(int item = 0; item < 4; ++item)
                    {
                        const int output_row =
                            row_base + m_atom * 16 + lane_group * 4 + item;
                        if(output_row < sub_end && output_column < N)
                        {
                            float value = accumulators[m_atom][atom][item];
                            if(all_bias != nullptr)
                                value += __bfloat162float(all_bias[
                                    static_cast<int64_t>(expert_index) * N +
                                    output_column]);
                            output[static_cast<int64_t>(output_row) * N +
                                   output_column] = __float2bfloat16(value);
                        }
                    }
                }
            }
        }
    }
#endif
}

template<int MAtoms, bool NMajor>
__global__ __launch_bounds__(256, MAtoms == 2 ? 2 : 1)
void iq2r_down_sparse_large32_kernel(
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
    constexpr int kPhysicalWaves = 4;
    static_assert(MAtoms == 2 || MAtoms == 4);
    constexpr int kMAtoms = MAtoms;
    constexpr int kNAtomsPerWave = 3;
    constexpr int kOutputColumns =
        kPhysicalWaves * kNAtomsPerWave * kTileN;

    struct SharedStorage
    {
        alignas(16) uint64_t codebook[kCodebookBytes / sizeof(uint64_t)];
        alignas(16) uint8_t weight_cache[kPhysicalWaves][kTripletLDSBytes];
    };
    __shared__ SharedStorage shared;

    const int lane = static_cast<int>(threadIdx.x);
    const int wave = static_cast<int>(threadIdx.y);
    const int linear_thread = wave * 64 + lane;
    const int lane_row = lane % 16;
    const int lane_group = lane / 16;
    const int num_tasks = task_count[0];
    if(num_tasks <= 0)
        return;

    const int n_tiles = (N + kOutputColumns - 1) / kOutputColumns;
    const int total_tiles = num_tasks * n_tiles;
    const int k_tiles = (K + kTileK - 1) / kTileK;
    int previous_expert = -1;

    for(int work_index = static_cast<int>(blockIdx.x); work_index < total_tiles;
        work_index += static_cast<int>(gridDim.x))
    {
        const int task_index =
            NMajor ? work_index % num_tasks : work_index / n_tiles;
        const int n_tile_index =
            NMajor ? work_index / num_tasks : work_index % n_tiles;
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
            // Protect readers of the old expert before reusing the codebook.
            if(previous_expert >= 0)
                __syncthreads();
            for(int entry = linear_thread;
                entry < kCodebookBytes / static_cast<int>(sizeof(uint64_t));
                entry += 256)
                shared.codebook[entry] =
                    reinterpret_cast<const uint64_t*>(auxiliary)[entry];
            __syncthreads();
            previous_expert = expert_index;
        }

        const int n_block_base =
            n_tile_index * (kOutputColumns / kTileN) +
            wave * kNAtomsPerWave;
        uint32_t packed_bases = 0;
#pragma unroll
        for(int atom = 0; atom < kNAtomsPerWave; ++atom)
            packed_bases |=
                static_cast<uint32_t>(
                    auxiliary[kCodebookBytes + n_block_base + atom])
                << (atom * 8);

        const int row_end = row_begin + row_count;
        for(int row_base = row_begin; row_base < row_end;
            row_base += kMAtoms * 16)
        {
            const int sub_end = min(row_base + kMAtoms * 16, row_end);
            opus::vector_t<float, 4>
                accumulators[kMAtoms][kNAtomsPerWave] = {};

            int next_data_base = static_cast<int>(triplet_base(
                physical_tile(n_block_base, 0, k_tiles)));
            iq2r_issue_direct_lds_triplet(data_buffer,
                                           shared.weight_cache[wave],
                                           lane,
                                           0,
                                           next_data_base);
            next_data_base += kGroupBytes;

            for(int k_tile = 0; k_tile < k_tiles; ++k_tile)
            {
                union ActivationFragment
                {
                    opus::i32x8_t words;
                    uint8_t bytes[32];
                } activation_fragments[kMAtoms] = {};
                uint32_t scale_a[kMAtoms];
#pragma unroll
                for(int m_atom = 0; m_atom < kMAtoms; ++m_atom)
                    scale_a[m_atom] = 127u * 0x01010101u;

#pragma unroll
                for(int m_atom = 0; m_atom < kMAtoms; ++m_atom)
                {
                    const int input_row = row_base + m_atom * 16 + lane_row;
                    if(input_row < sub_end)
                    {
                        const int activation_k =
                            k_tile * kTileK + lane_group * 16;
                        const auto* activation_row =
                            reinterpret_cast<const uint8_t*>(
                                activations + static_cast<int64_t>(input_row) * K);
                        if(activation_k + 15 < K)
                            *reinterpret_cast<uint4*>(
                                activation_fragments[m_atom].bytes) =
                                *reinterpret_cast<const uint4*>(
                                    activation_row + activation_k);
                        if(activation_k + 64 + 15 < K)
                            *reinterpret_cast<uint4*>(
                                activation_fragments[m_atom].bytes + 16) =
                                *reinterpret_cast<const uint4*>(
                                    activation_row + activation_k + 64);

                        const int scale_column = k_tile * 4 + lane_group;
                        if(scale_column < K / kScaleBlock)
                        {
                            const uint32_t exponent = activation_scales[
                                static_cast<int64_t>(input_row) *
                                    (K / kScaleBlock) +
                                scale_column];
                            scale_a[m_atom] = exponent * 0x01010101u;
                        }
                    }
                }

                // Two weight requests precede eight activation-vector and four
                // activation-scale requests on full K tiles. Wait only for the
                // older weight pair, leaving that activation tail in flight
                // while the compact record moves out of LDS. The padded final
                // K tile omits the second vector load and two scale groups.
                // Partial slabs issue fewer loads than the fixed wait allowance.
                if(sub_end - row_base < kMAtoms * 16)
                    __builtin_amdgcn_s_waitcnt(0x0F70);
                else if(k_tile + 1 < k_tiles)
                    __builtin_amdgcn_s_waitcnt(0x0F7C); // vmcnt(12)
                else
                    __builtin_amdgcn_s_waitcnt(0x0F78); // vmcnt(8)
                const IQ2RCompressedTriplet compressed =
                    iq2r_read_direct_lds_triplet(
                        shared.weight_cache[wave], lane, 0);
                asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
                const bool has_next = k_tile + 1 < k_tiles;
                if(has_next)
                {
                    iq2r_issue_direct_lds_triplet(data_buffer,
                                                   shared.weight_cache[wave],
                                                   lane,
                                                   0,
                                                   next_data_base);
                    next_data_base += kGroupBytes;
                }
                opus::i32x8_t weight_fragments[kNAtomsPerWave];
                const uint32_t scale_b = iq2r_decode_direct_lds_triplet(
                    compressed,
                    shared.codebook,
                    packed_bases,
                    weight_fragments);
                if(has_next)
                    __builtin_amdgcn_s_waitcnt(0x0F72); // vmcnt(2)
                else
                    __builtin_amdgcn_s_waitcnt(0x0F70); // vmcnt(0)
#pragma unroll
                for(int m_atom = 0; m_atom < kMAtoms; ++m_atom)
                    if(row_base + m_atom * 16 < sub_end)
                    iq2r_cooperative_triplet_mfma<0>(
                        activation_fragments[m_atom].words,
                        weight_fragments,
                        accumulators[m_atom],
                        scale_a[m_atom],
                        scale_b);
            }

#pragma unroll
            for(int m_atom = 0; m_atom < kMAtoms; ++m_atom)
            {
                if(row_base + m_atom * 16 >= sub_end) continue;
#pragma unroll
                for(int atom = 0; atom < kNAtomsPerWave; ++atom)
                {
                    const int output_column =
                        (n_block_base + atom) * kTileN + lane_row;
#pragma unroll
                    for(int item = 0; item < 4; ++item)
                    {
                        const int output_row =
                            row_base + m_atom * 16 + lane_group * 4 + item;
                        if(output_row < sub_end && output_column < N)
                        {
                            float value = accumulators[m_atom][atom][item];
                            if(all_bias != nullptr)
                                value += __bfloat162float(all_bias[
                                    static_cast<int64_t>(expert_index) * N +
                                    output_column]);
                            output[static_cast<int64_t>(output_row) * N +
                                   output_column] = __float2bfloat16(value);
                        }
                    }
                }
            }
        }
    }
#endif
}

// Two-stage high-M pipeline for the GPT-OSS K=2880 expert shapes.  Each wave
// keeps two compact IQ2R weight triplets in a register ring while the whole
// workgroup stages two 32x128 activation tiles through LDS.  The counted VMEM
// staircase retires only the current tile, leaving the next tile outstanding
// across codebook expansion and the six MFMAs per wave.
template<int MAtoms, bool TaskPersistent, bool PinToAgpr = false>
__global__ __launch_bounds__(256, MAtoms == 2 ? 2 : 1)
void iq2r_task_gemm_prefetch_x192_kernel(
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
    int auxiliary_bytes,
    int task_splits)
{
#if defined(__gfx950__)
    constexpr int kPhysicalWaves = 4;
    static_assert(MAtoms == 2 || MAtoms == 4);
    constexpr int kMAtoms = MAtoms;
    constexpr int kNAtomsPerWave = 3;
    constexpr int kOutputColumns = 192;
    constexpr int kKTiles = 23;
    constexpr int kActivationColumns = kTileK / sizeof(uint64_t);
    constexpr int kActivationElements = kMAtoms * kTileN * kActivationColumns;

    struct SharedStorage
    {
        alignas(16) uint64_t codebook[kCodebookBytes / sizeof(uint64_t)];
        alignas(16) uint64_t activation_cache[2][kActivationElements];
    };
    struct ActivationSlot
    {
        opus::vector_t<uint8_t, 16> data[kMAtoms / 2];
        uint32_t scales[kMAtoms];
    };
    __shared__ SharedStorage shared;

    const int lane = static_cast<int>(threadIdx.x);
    const int wave = static_cast<int>(threadIdx.y);
    const int linear_thread = wave * 64 + lane;
    const int lane_row = lane % 16;
    const int lane_group = lane / 16;
    const int load_column = linear_thread % 8;
    const int load_row = linear_thread / 8;
    const int num_tasks = task_count[0];
    if(num_tasks <= 0 || K != 2880)
        return;

    const int n_tiles = (N + kOutputColumns - 1) / kOutputColumns;
    const int total_tiles = num_tasks * n_tiles;
    const int fixed_task = TaskPersistent
                               ? static_cast<int>(blockIdx.x) / task_splits
                               : -1;
    if constexpr(TaskPersistent)
    {
        if(fixed_task >= num_tasks)
            return;
    }
    const int work_begin = TaskPersistent
                               ? static_cast<int>(blockIdx.x) % task_splits
                               : static_cast<int>(blockIdx.x);
    const int work_end = TaskPersistent ? n_tiles : total_tiles;
    const int work_step = TaskPersistent ? task_splits
                                         : static_cast<int>(gridDim.x);
    int previous_expert = -1;

    for(int work_index = work_begin; work_index < work_end;
        work_index += work_step)
    {
        const int task_index =
            TaskPersistent ? fixed_task : work_index / n_tiles;
        const int n_tile_index = TaskPersistent ? work_index
                                                : work_index % n_tiles;
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
        opus::gmem<uint8_t> activation_buffer(
            activations, static_cast<unsigned int>(M * K));
        opus::gmem<uint8_t> scale_buffer(
            activation_scales,
            static_cast<unsigned int>(M * (K / kScaleBlock)));
        if(expert_index != previous_expert)
        {
            for(int entry = linear_thread;
                entry < kCodebookBytes / static_cast<int>(sizeof(uint64_t));
                entry += 256)
                shared.codebook[entry] =
                    reinterpret_cast<const uint64_t*>(auxiliary)[entry];
            __syncthreads();
            previous_expert = expert_index;
        }

        const int n_block_base =
            n_tile_index * (kOutputColumns / kTileN) +
            wave * kNAtomsPerWave;
        uint32_t packed_bases = 0;
#pragma unroll
        for(int atom = 0; atom < kNAtomsPerWave; ++atom)
            packed_bases |=
                static_cast<uint32_t>(
                    auxiliary[kCodebookBytes + n_block_base + atom])
                << (atom * 8);

        const int row_end = row_begin + row_count;
        for(int row_base = row_begin; row_base < row_end;
            row_base += kMAtoms * kTileN)
        {
            const int sub_end = min(row_base + kMAtoms * kTileN, row_end);
            opus::vector_t<float, 4>
                accumulators[kMAtoms][kNAtomsPerWave] = {};
            ActivationSlot activation_ring[2] = {};
            IQ2RCompressedTriplet weight_ring[2] = {};
            int next_data_base = static_cast<int>(triplet_base(
                physical_tile(n_block_base, 0, kKTiles)));

            auto issue_tile = [&](int k_tile, int slot) {
                const int input_column = k_tile * kTileK + load_column * 16;
#pragma unroll
                for(int load_atom = 0; load_atom < kMAtoms / 2; ++load_atom)
                {
                    const int input_row =
                        row_base + load_row + load_atom * 32;
                    if(input_row < sub_end && input_column + 15 < K)
                        activation_ring[slot].data[load_atom] =
                            activation_buffer.template load<16>(
                                input_row * K + input_column);
                    else
                        activation_ring[slot].data[load_atom] = {};
                }

#pragma unroll
                for(int m_atom = 0; m_atom < kMAtoms; ++m_atom)
                {
                    const int scale_row = row_base + m_atom * kTileN + lane_row;
                    if(scale_row < sub_end)
                        activation_ring[slot].scales[m_atom] =
                            scale_buffer.template load<1>(
                                scale_row * (K / kScaleBlock) +
                                k_tile * 4 + lane_group)[0];
                    else
                        activation_ring[slot].scales[m_atom] = 127;
                }

                const auto paired = data_buffer.template load<16>(
                    next_data_base + lane * 16);
                weight_ring[slot].paired =
                    __builtin_bit_cast(uint4, paired);
                const int third_base = next_data_base + kAtomTwoOffset +
                                       lane * kAtomTwoRecordBytes;
                const auto third_xy = data_buffer.template load<8>(third_base);
                const auto third_z = data_buffer.template load<4>(third_base + 8);
                const uint64_t packed_xy =
                    __builtin_bit_cast(uint64_t, third_xy);
                weight_ring[slot].third.x = static_cast<uint32_t>(packed_xy);
                weight_ring[slot].third.y =
                    static_cast<uint32_t>(packed_xy >> 32);
                weight_ring[slot].third.z =
                    __builtin_bit_cast(uint32_t, third_z);
                weight_ring[slot].third.w = 0;
                next_data_base += kGroupBytes;
            };

            issue_tile(0, 0);
            issue_tile(1, 1);
#pragma unroll
            for(int k_tile = 0; k_tile < kKTiles; ++k_tile)
            {
                const int slot = k_tile & 1;
                if(k_tile + 1 < kKTiles)
                    iq2r_wait_vmcnt<3 + kMAtoms / 2 + kMAtoms>();
                else
                    iq2r_wait_vmcnt<0>();

                const int pair_column = load_column * 2;
                const int pair = pair_column >> 1;
                const int bit = pair_column & 1;
#pragma unroll
                for(int load_atom = 0; load_atom < kMAtoms / 2; ++load_atom)
                {
                    const int row = load_row + load_atom * 32;
                    const int swizzled_column =
                        ((pair ^ (row & (kActivationColumns / 2 - 1))) << 1) +
                        bit;
                    const auto staged = __builtin_bit_cast(
                        uint4, activation_ring[slot].data[load_atom]);
                    *reinterpret_cast<uint4*>(
                        &shared.activation_cache[slot]
                                                 [row * kActivationColumns +
                                                  swizzled_column]) = staged;
                }
                __syncthreads();

                opus::i32x8_t activation_fragments[kMAtoms];
#pragma unroll
                for(int m_atom = 0; m_atom < kMAtoms; ++m_atom)
                {
                    const int row = m_atom * kTileN + lane_row;
                    const int column0 = lane_group * 2;
                    const int pair0 = column0 >> 1;
                    const int bit0 = column0 & 1;
                    const int swizzled0 =
                        ((pair0 ^ (row & (kActivationColumns / 2 - 1))) << 1) +
                        bit0;
                    const int column1 = 8 + lane_group * 2;
                    const int pair1 = column1 >> 1;
                    const int bit1 = column1 & 1;
                    const int swizzled1 =
                        ((pair1 ^ (row & (kActivationColumns / 2 - 1))) << 1) +
                        bit1;
                    auto* halves = reinterpret_cast<uint4*>(
                        &activation_fragments[m_atom]);
                    halves[0] = *reinterpret_cast<const uint4*>(
                        &shared.activation_cache[slot]
                                                 [row * kActivationColumns +
                                                  swizzled0]);
                    halves[1] = *reinterpret_cast<const uint4*>(
                        &shared.activation_cache[slot]
                                                 [row * kActivationColumns +
                                                  swizzled1]);
                }
                asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");

                opus::i32x8_t weight_fragments[kNAtomsPerWave];
                const uint32_t scale_b = iq2r_decode_direct_lds_triplet(
                    weight_ring[slot],
                    shared.codebook,
                    packed_bases,
                    weight_fragments);
#pragma unroll
                for(int m_atom = 0; m_atom < kMAtoms; ++m_atom)
                {
                    const uint32_t scale_a =
                        activation_ring[slot].scales[m_atom] * 0x01010101u;
                    iq2r_cooperative_triplet_mfma<0, PinToAgpr>(
                        activation_fragments[m_atom],
                        weight_fragments,
                        accumulators[m_atom],
                        scale_a,
                        scale_b);
                }

                if(k_tile + 2 < kKTiles)
                    issue_tile(k_tile + 2, slot);
            }

#pragma unroll
            for(int m_atom = 0; m_atom < kMAtoms; ++m_atom)
            {
#pragma unroll
                for(int atom = 0; atom < kNAtomsPerWave; ++atom)
                {
                    const int output_column =
                        (n_block_base + atom) * kTileN + lane_row;
#pragma unroll
                    for(int item = 0; item < 4; ++item)
                    {
                        const int output_row =
                            row_base + m_atom * kTileN + lane_group * 4 + item;
                        if(output_row < sub_end && output_column < N)
                        {
                            float value = accumulators[m_atom][atom][item];
                            if(all_bias != nullptr)
                                value += __bfloat162float(all_bias[
                                    static_cast<int64_t>(expert_index) * N +
                                    output_column]);
                            output[static_cast<int64_t>(output_row) * N +
                                   output_column] = __float2bfloat16(value);
                        }
                    }
                }
            }
        }
    }
#endif
}

template<int OutputAtoms,
         int PhysicalWaves,
         bool ActivationLookahead = false,
         bool StagedScales = false,
         bool TiledActivationScales = false,
         bool VectorCodebookCopy = true,
         int LoadAux = 2,
         bool FusedSwiGLU = false>
__global__ __launch_bounds__(64 * PhysicalWaves,
                             PhysicalWaves == 2 ? 4
                                                : (PhysicalWaves == 4 ? 2 : 1))
void iq2r_task_gemm_cooperative_kernel(
    const opus::fp8_t* __restrict__ activations,
    const uint8_t* __restrict__ activation_scales,
    const uint8_t* __restrict__ all_data,
    const uint8_t* __restrict__ all_auxiliary,
    const int32_t* __restrict__ tasks,
    const int32_t* __restrict__ task_count,
    const __hip_bfloat16* __restrict__ all_bias,
    __hip_bfloat16* __restrict__ output,
    opus::fp8_t* __restrict__ fused_output,
    uint8_t* __restrict__ fused_output_scales,
    int M,
    int N,
    int K,
    int expert_count,
    int data_bytes,
    int auxiliary_bytes,
    float swiglu_limit,
    float swiglu_alpha,
    float swiglu_up_offset)
{
#if defined(__gfx950__)
    static_assert(OutputAtoms == 3 || OutputAtoms == 4 || OutputAtoms == 6);
    static_assert(PhysicalWaves == 2 || PhysicalWaves == 4 || PhysicalWaves == 8);
    static_assert(!ActivationLookahead || OutputAtoms == 6);
    static_assert(!(StagedScales && TiledActivationScales));
    static_assert(!FusedSwiGLU ||
                  (OutputAtoms == 4 && PhysicalWaves == 4 &&
                   !ActivationLookahead && !StagedScales));
    constexpr int kOutputColumns = OutputAtoms * kTileN;
    constexpr int kDirectTriplets =
        (OutputAtoms + kAtomsPerTriplet - 1) / kAtomsPerTriplet;
    constexpr int kStoredAtoms = OutputAtoms < 3 ? OutputAtoms : 3;
    constexpr int kScaleGroups = 2880 / kScaleBlock;
    constexpr int kPaddedScaleGroups = ((kScaleGroups + 3) / 4) * 4;
    constexpr int kStagedScaleBytes = 16 * kPaddedScaleGroups;

    struct SharedStorage
    {
        alignas(16) uint64_t codebook[kCodebookBytes / sizeof(uint64_t)];
        alignas(16) uint8_t
            weight_cache[PhysicalWaves][kDirectTriplets * kTripletLDSBytes];
        union
        {
            alignas(16) opus::vector_t<float, 4>
                partial[PhysicalWaves][kStoredAtoms][64];
            alignas(16) uint8_t scale_cache[kStagedScaleBytes];
        } reusable;
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
        if(row_begin < 0 || row_count <= 0 || row_begin + row_count > M)
            continue;
        if(expert_index < 0 || expert_index >= expert_count)
        {
            // Expert-parallel callers retain non-local routes in the sorted
            // permutation so the final indexed reduction can preserve the
            // original top-k order. Materialize their contribution exactly as
            // zero, matching the generic task kernel instead of leaving stale
            // workspace values behind.
            const int output_column_begin =
                n_tile_index * (FusedSwiGLU ? kOutputColumns / 2
                                            : kOutputColumns);
            const int output_width = FusedSwiGLU ? N / 2 : N;
            const int output_columns = min(FusedSwiGLU ? kOutputColumns / 2
                                                       : kOutputColumns,
                                           output_width - output_column_begin);
            const int task_elements = row_count * output_columns;
            for(int element = linear_thread; element < task_elements;
                element += linear_threads)
            {
                const int row = row_begin + element / output_columns;
                const int column = output_column_begin + element % output_columns;
                if constexpr(FusedSwiGLU)
                    fused_output[static_cast<int64_t>(row) * output_width + column] =
                        opus::fp32_to_fp8(0.0f);
                else
                    output[static_cast<int64_t>(row) * output_width + column] =
                        __float2bfloat16(0.0f);
            }
            if constexpr(FusedSwiGLU)
            {
                if(linear_thread < row_count)
                    fused_output_scales[
                        static_cast<int64_t>(row_begin + linear_thread) *
                            (N / 64) +
                        n_tile_index] = 127;
            }
            continue;
        }

        const uint8_t* data =
            all_data + static_cast<int64_t>(expert_index) * data_bytes;
        const uint8_t* auxiliary =
            all_auxiliary + static_cast<int64_t>(expert_index) * auxiliary_bytes;
        opus::gmem<uint8_t> data_buffer(data, static_cast<unsigned int>(data_bytes));
        if(expert_index != previous_expert)
        {
            if constexpr(VectorCodebookCopy && PhysicalWaves == 4)
            {
                reinterpret_cast<uint4*>(shared.codebook)[linear_thread] =
                    reinterpret_cast<const uint4*>(auxiliary)[linear_thread];
            }
            else
            {
                for(int entry = linear_thread;
                    entry < kCodebookBytes / static_cast<int>(sizeof(uint64_t));
                    entry += linear_threads)
                    shared.codebook[entry] =
                        reinterpret_cast<const uint64_t*>(auxiliary)[entry];
            }
            __syncthreads();
            previous_expert = expert_index;
        }

        const int n_block_base = n_tile_index * OutputAtoms;
        // Four-atom tiles advance by 64 columns and are not generally aligned
        // to the three-atom (48-column) physical IQ2R packing groups. Load the
        // two enclosing triplets and decode the requested four consecutive
        // atoms, matching the established fused-SwiGLU path.
        const int loaded_n_block_base =
            OutputAtoms == 4 ? n_block_base - n_block_base % kAtomsPerTriplet
                             : n_block_base;
        const int fused_atom_offset = n_block_base - loaded_n_block_base;
        uint32_t packed_bases[kDirectTriplets] = {};
#pragma unroll
        for(int triplet = 0; triplet < kDirectTriplets; ++triplet)
        {
#pragma unroll
            for(int atom = 0; atom < kAtomsPerTriplet; ++atom)
                packed_bases[triplet] |=
                    static_cast<uint32_t>(
                        auxiliary[kCodebookBytes + loaded_n_block_base +
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

            if constexpr(StagedScales)
            {
                const int valid_rows = sub_end - row_base;
                for(int index = linear_thread; index < kStagedScaleBytes;
                    index += linear_threads)
                {
                    const int row = index / kPaddedScaleGroups;
                    const int group = index % kPaddedScaleGroups;
                    uint8_t exponent = 127;
                    if(group < kScaleGroups)
                    {
                        const int source_row = row_base + min(row, valid_rows - 1);
                        exponent = activation_scales[
                            iq2r_activation_scale_offset<TiledActivationScales>(
                                source_row, group, M, kScaleGroups)];
                    }
                    const int scale_k_tile = group / 4;
                    const int scale_group = group % 4;
                    shared.reusable.scale_cache[scale_k_tile * 64 +
                                                scale_group * 16 + row] = exponent;
                }
                __syncthreads();
            }
            opus::vector_t<float, 4> accumulators[OutputAtoms] = {};

            int next_data_base = static_cast<int>(triplet_base(
                physical_tile(loaded_n_block_base, k_begin, k_tiles)));
            int next_second_data_base = static_cast<int>(triplet_base(
                physical_tile(loaded_n_block_base + kAtomsPerTriplet,
                              k_begin,
                              k_tiles)));

            if constexpr(ActivationLookahead)
            {
                // Keep the next activation fragment in VGPRs while the second
                // weight triplet for the current K tile is decoded and
                // consumed.  Every steady-state iteration has this request
                // order: next activation, next triplet 0, next triplet 1.
                // Counted waits retire only the current triplet and leave that
                // younger lookahead in flight.
                const int uniform_iterations =
                    __builtin_amdgcn_readfirstlane(iterations);
                opus::gmem<uint8_t> activation_buffer(
                    activations, static_cast<unsigned int>(M * K));
                opus::gmem<uint8_t> scale_buffer(
                    activation_scales,
                    static_cast<unsigned int>(
                        TiledActivationScales
                            ? ((K / kScaleBlock + 3) / 4) * ((M + 15) / 16) *
                                  64
                            : M * (K / kScaleBlock)));
                IQ2RActivationFragment current_activation{};
                uint32_t current_scale_a = 127u * 0x01010101u;
                iq2r_issue_buffered_activation_fragment<StagedScales,
                                                         TiledActivationScales>(
                    activation_buffer,
                    scale_buffer,
                    shared.reusable.scale_cache,
                    load_row,
                    row_base,
                    k_begin,
                    lane_group,
                    M,
                    K,
                    current_activation,
                    current_scale_a);
                iq2r_issue_direct_lds_triplet<LoadAux>(data_buffer,
                                               shared.weight_cache[wave],
                                               lane,
                                               0,
                                               next_data_base);
                iq2r_issue_direct_lds_triplet<LoadAux>(data_buffer,
                                               shared.weight_cache[wave],
                                               lane,
                                               1,
                                               next_data_base + kTripletBytes);
                next_data_base += kGroupBytes;

                for(int iteration = 0; iteration < uniform_iterations; ++iteration)
                {
                    // Activation requests precede both current weight
                    // triplets. Retire the activation and triplet 0 while the
                    // two triplet-1 requests remain outstanding.
                    iq2r_wait_vmcnt<2>();
                    IQ2RCompressedTriplet compressed =
                        iq2r_read_direct_lds_triplet(
                            shared.weight_cache[wave], lane, 0);
                    asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
                    const opus::i32x8_t current_words = current_activation.words;

                    const bool has_next = iteration + 1 < uniform_iterations;
                    uint32_t next_scale_a = 127u * 0x01010101u;
                    if(has_next)
                    {
                        iq2r_issue_buffered_activation_fragment<
                            StagedScales,
                            TiledActivationScales>(
                            activation_buffer,
                            scale_buffer,
                            shared.reusable.scale_cache,
                            load_row,
                            row_base,
                            k_begin + iteration + 1,
                            lane_group,
                            M,
                            K,
                            current_activation,
                            next_scale_a);
                        iq2r_issue_direct_lds_triplet<LoadAux>(data_buffer,
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
                    iq2r_cooperative_triplet_mfma<0>(
                        current_words,
                        weight_fragments,
                        accumulators,
                        current_scale_a,
                        scale_b);

                    if(has_next)
                    {
                        // Two activation requests and two triplet-0 requests
                        // remain younger when scales are staged in LDS. The
                        // row-major variant has one additional scale request.
                        if constexpr(StagedScales)
                            iq2r_wait_vmcnt<4>();
                        else
                            iq2r_wait_vmcnt<5>();
                    }
                    else
                        iq2r_wait_vmcnt<0>();

                    compressed = iq2r_read_direct_lds_triplet(
                        shared.weight_cache[wave], lane, 1);
                    asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
                    if(has_next)
                    {
                        iq2r_issue_direct_lds_triplet<LoadAux>(data_buffer,
                                                       shared.weight_cache[wave],
                                                       lane,
                                                       1,
                                                       next_data_base +
                                                           kTripletBytes);
                        next_data_base += kGroupBytes;
                    }
                    scale_b = iq2r_decode_direct_lds_triplet(compressed,
                                                             shared.codebook,
                                                             packed_bases[1],
                                                             weight_fragments);
                    iq2r_cooperative_triplet_mfma<3>(
                        current_words,
                        weight_fragments,
                        accumulators,
                        current_scale_a,
                        scale_b);
                    if(has_next)
                        current_scale_a = next_scale_a;
                }
            }
            else
            {
                if constexpr(OutputAtoms == 4)
                {
                    iq2r_issue_direct_lds_triplet<LoadAux>(
                        data_buffer,
                        shared.weight_cache[wave],
                        lane,
                        0,
                        next_data_base);
                    iq2r_issue_direct_lds_triplet<LoadAux>(
                        data_buffer,
                        shared.weight_cache[wave],
                        lane,
                        1,
                        next_second_data_base);
                }
                else
                {
#pragma unroll
                    for(int triplet = 0; triplet < kDirectTriplets; ++triplet)
                        iq2r_issue_direct_lds_triplet<LoadAux>(
                            data_buffer,
                            shared.weight_cache[wave],
                            lane,
                            triplet,
                            next_data_base + triplet * kTripletBytes);
                }
                next_data_base += kGroupBytes;
                next_second_data_base += kGroupBytes;
                int tiled_scale_offset =
                    (k_begin * ((M + 15) / 16) + load_row / 16) * 64 +
                    lane_group * 16 + load_row % 16;
                const int tiled_scale_stride = ((M + 15) / 16) * 64;

                for(int iteration = 0; iteration < iterations; ++iteration)
                {
                    const int k_tile = k_begin + iteration;
                    IQ2RActivationFragment activation_fragment{};
                    const int activation_k = k_tile * kTileK + lane_group * 16;
                    if(activation_k + 15 < K)
                    {
                        const auto* activation_row =
                            reinterpret_cast<const uint8_t*>(
                                activations + static_cast<int64_t>(load_row) * K);
                        *reinterpret_cast<uint4*>(activation_fragment.bytes) =
                            *reinterpret_cast<const uint4*>(activation_row +
                                                           activation_k);
                        if(activation_k + 64 + 15 < K)
                            *reinterpret_cast<uint4*>(activation_fragment.bytes +
                                                     16) =
                                *reinterpret_cast<const uint4*>(activation_row +
                                                               activation_k + 64);
                    }

                    uint32_t scale_a = 127u * 0x01010101u;
                    if constexpr(StagedScales)
                    {
                        const uint32_t exponent = shared.reusable.scale_cache[
                            k_tile * 64 + lane_group * 16 + load_row - row_base];
                        scale_a = exponent * 0x01010101u;
                    }
                    else
                    {
                        const int scale_column = k_tile * 4 + lane_group;
                        if(scale_column < K / kScaleBlock)
                        {
                            uint32_t exponent;
                            if constexpr(TiledActivationScales)
                            {
                                exponent = activation_scales[tiled_scale_offset];
                                tiled_scale_offset += tiled_scale_stride;
                            }
                            else
                                exponent = activation_scales[
                                    static_cast<int64_t>(load_row) *
                                            (K / kScaleBlock) +
                                        scale_column];
                            scale_a = exponent * 0x01010101u;
                        }
                    }

                    // The direct-to-LDS requests precede two or three activation
                    // requests. Retire only the older weight requests first so
                    // their LDS reads and IQ2R decode overlap the activation tail.
                    if(k_tile + 1 < k_tiles)
                    {
                        if constexpr(StagedScales)
                            iq2r_wait_vmcnt<2>();
                        else
                            iq2r_wait_vmcnt<3>();
                    }
                    else
                    {
                        if constexpr(StagedScales)
                            iq2r_wait_vmcnt<1>();
                        else
                            iq2r_wait_vmcnt<2>();
                    }

                    IQ2RCompressedTriplet compressed =
                        iq2r_read_direct_lds_triplet(
                            shared.weight_cache[wave], lane, 0);
                    asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
                    iq2r_wait_vmcnt<0>();
                    if(iteration + 1 < iterations)
                    {
                        if constexpr(OutputAtoms == 4)
                            iq2r_issue_direct_lds_triplet<LoadAux>(
                                data_buffer,
                                shared.weight_cache[wave],
                                lane,
                                0,
                                next_data_base);
                        else
                            iq2r_issue_direct_lds_triplet<LoadAux>(
                                data_buffer,
                                shared.weight_cache[wave],
                                lane,
                                0,
                                next_data_base);
                    }
                    if constexpr(OutputAtoms == 4)
                    {
                        opus::i32x8_t weight_fragment;
                        uint32_t scale_b;
                        if(fused_atom_offset == 0)
                        {
                            iq2r_decode_direct_lds_atom<0>(compressed,
                                                           shared.codebook,
                                                           packed_bases[0],
                                                           weight_fragment,
                                                           scale_b);
                            iq2r_cooperative_single_mfma<0>(
                                activation_fragment.words, weight_fragment,
                                accumulators, scale_a, scale_b);
                            iq2r_decode_direct_lds_atom<1>(compressed,
                                                           shared.codebook,
                                                           packed_bases[0],
                                                           weight_fragment,
                                                           scale_b);
                            iq2r_cooperative_single_mfma<1>(
                                activation_fragment.words, weight_fragment,
                                accumulators, scale_a, scale_b);
                            iq2r_decode_direct_lds_atom<2>(compressed,
                                                           shared.codebook,
                                                           packed_bases[0],
                                                           weight_fragment,
                                                           scale_b);
                            iq2r_cooperative_single_mfma<2>(
                                activation_fragment.words, weight_fragment,
                                accumulators, scale_a, scale_b);
                        }
                        else if(fused_atom_offset == 1)
                        {
                            iq2r_decode_direct_lds_atom<1>(compressed,
                                                           shared.codebook,
                                                           packed_bases[0],
                                                           weight_fragment,
                                                           scale_b);
                            iq2r_cooperative_single_mfma<0>(
                                activation_fragment.words, weight_fragment,
                                accumulators, scale_a, scale_b);
                            iq2r_decode_direct_lds_atom<2>(compressed,
                                                           shared.codebook,
                                                           packed_bases[0],
                                                           weight_fragment,
                                                           scale_b);
                            iq2r_cooperative_single_mfma<1>(
                                activation_fragment.words, weight_fragment,
                                accumulators, scale_a, scale_b);
                        }
                        else
                        {
                            iq2r_decode_direct_lds_atom<2>(compressed,
                                                           shared.codebook,
                                                           packed_bases[0],
                                                           weight_fragment,
                                                           scale_b);
                            iq2r_cooperative_single_mfma<0>(
                                activation_fragment.words, weight_fragment,
                                accumulators, scale_a, scale_b);
                        }
                    }
                    else
                    {
                        opus::i32x8_t weight_fragments[kAtomsPerTriplet];
                        const uint32_t scale_b = iq2r_decode_direct_lds_triplet(
                            compressed,
                            shared.codebook,
                            packed_bases[0],
                            weight_fragments);
                        iq2r_cooperative_triplet_mfma<0>(
                            activation_fragment.words,
                            weight_fragments,
                            accumulators,
                            scale_a,
                            scale_b);
                    }
                    if constexpr(OutputAtoms > 3)
                    {
                        compressed = iq2r_read_direct_lds_triplet(
                            shared.weight_cache[wave], lane, 1);
                        asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
                        if(iteration + 1 < iterations)
                        {
                            if constexpr(OutputAtoms == 4)
                                iq2r_issue_direct_lds_triplet<LoadAux>(
                                    data_buffer,
                                    shared.weight_cache[wave],
                                    lane,
                                    1,
                                    next_second_data_base);
                            else
                                iq2r_issue_direct_lds_triplet<LoadAux>(
                                    data_buffer,
                                    shared.weight_cache[wave],
                                    lane,
                                    1,
                                    next_data_base + kTripletBytes);
                            next_data_base += kGroupBytes;
                            next_second_data_base += kGroupBytes;
                        }
                        if constexpr(OutputAtoms == 4)
                        {
                            opus::i32x8_t weight_fragment;
                            uint32_t scale_b;
                            if(fused_atom_offset == 0)
                            {
                                iq2r_decode_direct_lds_atom<0>(compressed,
                                                               shared.codebook,
                                                               packed_bases[1],
                                                               weight_fragment,
                                                               scale_b);
                                iq2r_cooperative_single_mfma<3>(
                                    activation_fragment.words, weight_fragment,
                                    accumulators, scale_a, scale_b);
                            }
                            else if(fused_atom_offset == 1)
                            {
                                iq2r_decode_direct_lds_atom<0>(compressed,
                                                               shared.codebook,
                                                               packed_bases[1],
                                                               weight_fragment,
                                                               scale_b);
                                iq2r_cooperative_single_mfma<2>(
                                    activation_fragment.words, weight_fragment,
                                    accumulators, scale_a, scale_b);
                                iq2r_decode_direct_lds_atom<1>(compressed,
                                                               shared.codebook,
                                                               packed_bases[1],
                                                               weight_fragment,
                                                               scale_b);
                                iq2r_cooperative_single_mfma<3>(
                                    activation_fragment.words, weight_fragment,
                                    accumulators, scale_a, scale_b);
                            }
                            else
                            {
                                iq2r_decode_direct_lds_atom<0>(compressed,
                                                               shared.codebook,
                                                               packed_bases[1],
                                                               weight_fragment,
                                                               scale_b);
                                iq2r_cooperative_single_mfma<1>(
                                    activation_fragment.words, weight_fragment,
                                    accumulators, scale_a, scale_b);
                                iq2r_decode_direct_lds_atom<1>(compressed,
                                                               shared.codebook,
                                                               packed_bases[1],
                                                               weight_fragment,
                                                               scale_b);
                                iq2r_cooperative_single_mfma<2>(
                                    activation_fragment.words, weight_fragment,
                                    accumulators, scale_a, scale_b);
                                iq2r_decode_direct_lds_atom<2>(compressed,
                                                               shared.codebook,
                                                               packed_bases[1],
                                                               weight_fragment,
                                                               scale_b);
                                iq2r_cooperative_single_mfma<3>(
                                    activation_fragment.words, weight_fragment,
                                    accumulators, scale_a, scale_b);
                            }
                        }
                        else
                        {
                            opus::i32x8_t weight_fragments[kAtomsPerTriplet];
                            const uint32_t scale_b = iq2r_decode_direct_lds_triplet(
                                compressed,
                                shared.codebook,
                                packed_bases[1],
                                weight_fragments);
                            iq2r_cooperative_triplet_mfma<3>(
                                activation_fragment.words,
                                weight_fragments,
                                accumulators,
                                scale_a,
                                scale_b);
                        }
                    }
                    else if(iteration + 1 < iterations)
                        next_data_base += kGroupBytes;
                }
            }

            if constexpr(StagedScales)
                __syncthreads();

            const int output_row = row_base + (lane / 16) * 4 + wave;
            if constexpr(FusedSwiGLU)
            {
                float raw_values[4] = {};
#pragma unroll
                for(int atom = 0; atom < 3; ++atom)
                    shared.reusable.partial[wave][atom][lane] = accumulators[atom];
                __syncthreads();
                if(wave < 4 && output_row < sub_end)
                {
#pragma unroll
                    for(int atom = 0; atom < 3; ++atom)
                    {
                        const float* p0 = reinterpret_cast<const float*>(
                            &shared.reusable.partial[0][atom][lane]);
                        const float* p1 = reinterpret_cast<const float*>(
                            &shared.reusable.partial[1][atom][lane]);
                        const float* p2 = reinterpret_cast<const float*>(
                            &shared.reusable.partial[2][atom][lane]);
                        const float* p3 = reinterpret_cast<const float*>(
                            &shared.reusable.partial[3][atom][lane]);
                        float value = p0[wave] + p1[wave] + p2[wave] + p3[wave];
                        if(all_bias != nullptr)
                            value += __bfloat162float(
                                all_bias[static_cast<int64_t>(expert_index) * N +
                                         n_tile_index * kOutputColumns + atom * 16 +
                                         lane % 16]);
                        raw_values[atom] = __bfloat162float(__float2bfloat16(value));
                    }
                }
                __syncthreads();

                shared.reusable.partial[wave][0][lane] = accumulators[3];
                __syncthreads();
                if(wave < 4 && output_row < sub_end)
                {
                    const float* p0 = reinterpret_cast<const float*>(
                        &shared.reusable.partial[0][0][lane]);
                    const float* p1 = reinterpret_cast<const float*>(
                        &shared.reusable.partial[1][0][lane]);
                    const float* p2 = reinterpret_cast<const float*>(
                        &shared.reusable.partial[2][0][lane]);
                    const float* p3 = reinterpret_cast<const float*>(
                        &shared.reusable.partial[3][0][lane]);
                    float value = p0[wave] + p1[wave] + p2[wave] + p3[wave];
                    if(all_bias != nullptr)
                        value += __bfloat162float(
                            all_bias[static_cast<int64_t>(expert_index) * N +
                                     n_tile_index * kOutputColumns + 3 * 16 +
                                     lane % 16]);
                    raw_values[3] = __bfloat162float(__float2bfloat16(value));
                }
                __syncthreads();

                if(wave < 4 && output_row < sub_end)
                {
                    const bool even_column = (lane % 2) == 0;
                    opus::vector_t<opus::bf16_t, 4> activated;
                    float abs_max = 1.0e-10f;
#pragma unroll
                    for(int atom = 0; atom < 4; ++atom)
                    {
                        const float paired = __shfl_xor(raw_values[atom], 1);
                        if(even_column)
                        {
                            const float gate = raw_values[atom];
                            const float up = paired;
                            const bool clamp = swiglu_limit > 0.0f;
                            const float clipped_gate =
                                clamp && gate > swiglu_limit ? swiglu_limit : gate;
                            const float clipped_up =
                                clamp && up < -swiglu_limit
                                    ? -swiglu_limit
                                    : (clamp && up > swiglu_limit
                                           ? swiglu_limit
                                           : up);
                            const float swish =
                                clipped_gate /
                                (1.0f + __expf(-swiglu_alpha * clipped_gate));
                            const __hip_bfloat16 rounded = __float2bfloat16(
                                swish * (clipped_up + swiglu_up_offset));
                            activated[atom] =
                                __builtin_bit_cast(opus::bf16_t, rounded);
                            abs_max =
                                fmaxf(abs_max, fabsf(__bfloat162float(rounded)));
                        }
                    }
                    abs_max = fmaxf(abs_max, __shfl_xor(abs_max, 2));
                    abs_max = fmaxf(abs_max, __shfl_xor(abs_max, 4));
                    abs_max = fmaxf(abs_max, __shfl_xor(abs_max, 8));
                    if(even_column)
                    {
                        const auto block_scale = fp_f32_to_e8m0_block_scale<
                            kDefaultMxScaleRoundMode,
                            MxDtype::FP8_E4M3>(abs_max);
                        const float inverse_scale = 1.0f / block_scale.dq_scale;
                        const int output_width = N / 2;
                        const int output_column_base =
                            n_tile_index * (kOutputColumns / 2) + (lane % 16) / 2;
#pragma unroll
                        for(int atom = 0; atom < 4; ++atom)
                            fused_output[static_cast<int64_t>(output_row) *
                                             output_width +
                                         output_column_base + atom * 8] =
                                opus::fp32_to_fp8(
                                    static_cast<float>(activated[atom]) *
                                    inverse_scale);
                        if(lane % 16 == 0)
                            fused_output_scales[
                                static_cast<int64_t>(output_row) * (N / 64) +
                                n_tile_index] = block_scale.byte;
                    }
                }
                __syncthreads();
            }
            else
            {
#pragma unroll
                for(int atom = 0; atom < 3; ++atom)
                    shared.reusable.partial[wave][atom][lane] = accumulators[atom];
                __syncthreads();
#pragma unroll
                for(int atom = 0; atom < 3; ++atom)
                {
                    const int output_column =
                        n_tile_index * kOutputColumns + atom * 16 + lane % 16;
                    if constexpr(PhysicalWaves == 2)
                    {
#pragma unroll
                        for(int writer = 0; writer < 2; ++writer)
                        {
                            const int row_component = wave + writer * PhysicalWaves;
                            const int writer_output_row =
                                row_base + (lane / 16) * 4 + row_component;
                            if(writer_output_row < sub_end && output_column < N)
                            {
                                const float* p0 = reinterpret_cast<const float*>(
                                    &shared.reusable.partial[0][atom][lane]);
                                const float* p1 = reinterpret_cast<const float*>(
                                    &shared.reusable.partial[1][atom][lane]);
                                float value = p0[row_component] + p1[row_component];
                                if(all_bias != nullptr)
                                    value += __bfloat162float(
                                        all_bias[static_cast<int64_t>(expert_index) * N +
                                                 output_column]);
                                output[static_cast<int64_t>(writer_output_row) * N +
                                       output_column] = __float2bfloat16(value);
                            }
                        }
                    }
                    else if(wave < 4 && output_row < sub_end && output_column < N)
                    {
                        float value = 0.0f;
                        if constexpr(PhysicalWaves == 4)
                        {
                            const float* p0 = reinterpret_cast<const float*>(
                                &shared.reusable.partial[0][atom][lane]);
                            const float* p1 = reinterpret_cast<const float*>(
                                &shared.reusable.partial[1][atom][lane]);
                            const float* p2 = reinterpret_cast<const float*>(
                                &shared.reusable.partial[2][atom][lane]);
                            const float* p3 = reinterpret_cast<const float*>(
                                &shared.reusable.partial[3][atom][lane]);
                            value = p0[wave] + p1[wave] + p2[wave] + p3[wave];
                        }
                        else
                        {
#pragma unroll
                            for(int source_wave = 0;
                                source_wave < PhysicalWaves;
                                ++source_wave)
                                value += shared.reusable
                                             .partial[source_wave][atom][lane][wave];
                        }
                        if(all_bias != nullptr)
                            value += __bfloat162float(
                                all_bias[static_cast<int64_t>(expert_index) * N +
                                         output_column]);
                        output[static_cast<int64_t>(output_row) * N + output_column] =
                            __float2bfloat16(value);
                    }
                }
                __syncthreads();

                if constexpr(OutputAtoms > 3)
                {
#pragma unroll
                    for(int atom = 0; atom < OutputAtoms - 3; ++atom)
                        shared.reusable.partial[wave][atom][lane] = accumulators[atom + 3];
                    __syncthreads();
#pragma unroll
                    for(int local_atom = 0; local_atom < OutputAtoms - 3;
                        ++local_atom)
                    {
                        const int atom = local_atom + 3;
                        const int output_column =
                            n_tile_index * kOutputColumns + atom * 16 + lane % 16;
                        if constexpr(PhysicalWaves == 2)
                        {
#pragma unroll
                            for(int writer = 0; writer < 2; ++writer)
                            {
                                const int row_component =
                                    wave + writer * PhysicalWaves;
                                const int writer_output_row =
                                    row_base + (lane / 16) * 4 + row_component;
                                if(writer_output_row < sub_end && output_column < N)
                                {
                                    const float* p0 =
                                        reinterpret_cast<const float*>(
                                            &shared.reusable
                                                 .partial[0][local_atom][lane]);
                                    const float* p1 =
                                        reinterpret_cast<const float*>(
                                            &shared.reusable
                                                 .partial[1][local_atom][lane]);
                                    float value =
                                        p0[row_component] + p1[row_component];
                                    if(all_bias != nullptr)
                                        value += __bfloat162float(all_bias[
                                            static_cast<int64_t>(expert_index) * N +
                                            output_column]);
                                    output[static_cast<int64_t>(writer_output_row) * N +
                                           output_column] = __float2bfloat16(value);
                                }
                            }
                        }
                        else if(wave < 4 && output_row < sub_end && output_column < N)
                        {
                            float value = 0.0f;
                            if constexpr(PhysicalWaves == 4)
                            {
                                const float* p0 = reinterpret_cast<const float*>(
                                    &shared.reusable.partial[0][local_atom][lane]);
                                const float* p1 = reinterpret_cast<const float*>(
                                    &shared.reusable.partial[1][local_atom][lane]);
                                const float* p2 = reinterpret_cast<const float*>(
                                    &shared.reusable.partial[2][local_atom][lane]);
                                const float* p3 = reinterpret_cast<const float*>(
                                    &shared.reusable.partial[3][local_atom][lane]);
                                value = p0[wave] + p1[wave] + p2[wave] + p3[wave];
                            }
                            else
                            {
#pragma unroll
                                for(int source_wave = 0;
                                    source_wave < PhysicalWaves;
                                    ++source_wave)
                                    value += shared.reusable
                                                 .partial[source_wave][local_atom]
                                                         [lane][wave];
                            }
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

// Four raw atoms form exactly one 32-value SwiGLU/FP8 scale group.
// Keep the retained eight-way contiguous K partition and its reduction order.
template<int Offset, int MAtoms>
__device__ __forceinline__ void iq2r_aligned_decode_mfma(
    const IQ2RCompressedTriplet& first, const IQ2RCompressedTriplet& second,
    const uint64_t* codebook, uint32_t bases0, uint32_t bases1,
    const IQ2RActivationFragment* activation, const uint32_t* scale_a,
    opus::vector_t<float,4> (&accumulators)[MAtoms][4])
{
#pragma unroll
    for(int atom=0;atom<4;++atom)
    {
        // Clang unrolls atom, so every decoder and accumulator index is static.
        opus::i32x8_t b; uint32_t sb;
        if(atom+Offset==0) iq2r_decode_direct_lds_atom<0>(first,codebook,bases0,b,sb);
        else if(atom+Offset==1) iq2r_decode_direct_lds_atom<1>(first,codebook,bases0,b,sb);
        else if(atom+Offset==2) iq2r_decode_direct_lds_atom<2>(first,codebook,bases0,b,sb);
        else if(atom+Offset==3) iq2r_decode_direct_lds_atom<0>(second,codebook,bases1,b,sb);
        else if(atom+Offset==4) iq2r_decode_direct_lds_atom<1>(second,codebook,bases1,b,sb);
        else iq2r_decode_direct_lds_atom<2>(second,codebook,bases1,b,sb);
#pragma unroll
        for(int m=0;m<MAtoms;++m)
        {
            auto mma=opus::mfma<opus::fp8_t,opus::fp8_t,opus::fp32_t,16,16,128>{};
            accumulators[m][atom]=mma(activation[m].words,b,accumulators[m][atom],
                                      scale_a[m],sb,opus::number<0>{},opus::number<0>{});
        }
    }
}

template<int MAtoms>
__global__ __launch_bounds__(512,1) void iq2r_gate_aligned_fused_kernel(
    const opus::fp8_t* __restrict__ activations,
    const uint8_t* __restrict__ scales,
    const uint8_t* __restrict__ all_data,
    const uint8_t* __restrict__ all_auxiliary,
    const int32_t* __restrict__ tasks,
    const int32_t* __restrict__ task_count,
    const int32_t* __restrict__ gather,
    opus::fp8_t* __restrict__ output,
    uint8_t* __restrict__ output_scales,
    int routes,int data_bytes,int auxiliary_bytes)
{
#if defined(__gfx950__)
    constexpr int K=6144, Rows=16*MAtoms, Subtasks=2/MAtoms;
    struct Storage {
        alignas(16) uint64_t codebook[kCodebookBytes/8];
        union {
            alignas(16) uint8_t cache[8][2*kTripletLDSBytes];
            alignas(16) opus::vector_t<float,4> partial[8][4][64];
        } reuse;
        alignas(16) __hip_bfloat16 gate[Rows][64];
    };
    __shared__ Storage shared;
    const int lane=threadIdx.x,wave=threadIdx.y,linear=wave*64+lane;
    const int lane_row=lane%16,lane_group=lane/16;
    const int num_tasks=task_count[0];
    const int total=num_tasks*Subtasks*8;
    for(int work=blockIdx.x;work<total;work+=gridDim.x)
    {
        const int task=(work/Subtasks)%num_tasks;
        const int sub=work%Subtasks;
        const int n_tile=work/(num_tasks*Subtasks);
        const int row_begin=tasks[task*3]+sub*Rows;
        const int row_end=min(row_begin+Rows,tasks[task*3]+tasks[task*3+1]);
        const int expert=tasks[task*3+2];
        if(row_begin>=row_end || row_begin<0 || row_end>routes || expert<0 || expert>=257) continue;
        const uint8_t* data=all_data+static_cast<int64_t>(expert)*data_bytes;
        const uint8_t* aux=all_auxiliary+static_cast<int64_t>(expert)*auxiliary_bytes;
        // Previous work ends with a barrier protecting both union and codebook.
        shared.codebook[linear]=reinterpret_cast<const uint64_t*>(aux)[linear];
        __syncthreads();
        const int nbase=n_tile*4, offset=nbase%3, triplet0=nbase-offset;
        uint32_t bases0=0,bases1=0;
#pragma unroll
        for(int atom=0;atom<3;++atom)
        {
            const int b0=triplet0+atom,b1=triplet0+3+atom;
            bases0|=static_cast<uint32_t>(b0<32?aux[kCodebookBytes+b0]:0)<<(atom*8);
            bases1|=static_cast<uint32_t>(b1<32?aux[kCodebookBytes+b1]:0)<<(atom*8);
        }
        int rows[MAtoms];
#pragma unroll
        for(int m=0;m<MAtoms;++m) rows[m]=gather[min(row_begin+m*16+lane_row,row_end-1)]/9;
        opus::vector_t<float,4> accumulators[MAtoms][4]={};
        opus::gmem<uint8_t> buffer(data,static_cast<unsigned int>(data_bytes));
        int base0=static_cast<int>(triplet_base(physical_tile(triplet0,wave*6,48)));
        int base1=static_cast<int>(triplet_base(physical_tile(triplet0+3,wave*6,48)));
        iq2r_issue_direct_lds_triplet<0>(buffer,shared.reuse.cache[wave],lane,0,base0);
        iq2r_issue_direct_lds_triplet<0>(buffer,shared.reuse.cache[wave],lane,1,base1);
        for(int iteration=0;iteration<6;++iteration)
        {
            const int kt=wave*6+iteration;
            IQ2RActivationFragment a[MAtoms]={};uint32_t sa[MAtoms];
#pragma unroll
            for(int m=0;m<MAtoms;++m)
            {
                const auto* input=reinterpret_cast<const uint8_t*>(activations+static_cast<int64_t>(rows[m])*K);
                const int ak=kt*128+lane_group*16;
                *reinterpret_cast<uint4*>(a[m].bytes)=*reinterpret_cast<const uint4*>(input+ak);
                *reinterpret_cast<uint4*>(a[m].bytes+16)=*reinterpret_cast<const uint4*>(input+ak+64);
                sa[m]=scales[static_cast<int64_t>(rows[m])*192+kt*4+lane_group]*0x01010101u;
            }
            iq2r_wait_vmcnt<0>();
            const auto first=iq2r_read_direct_lds_triplet(shared.reuse.cache[wave],lane,0);
            const auto second=iq2r_read_direct_lds_triplet(shared.reuse.cache[wave],lane,1);
            asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
            if(iteration<5)
            {
                base0+=kGroupBytes;base1+=kGroupBytes;
                iq2r_issue_direct_lds_triplet<0>(buffer,shared.reuse.cache[wave],lane,0,base0);
        iq2r_issue_direct_lds_triplet<0>(buffer,shared.reuse.cache[wave],lane,1,base1);
            }
            if(offset==0) iq2r_aligned_decode_mfma<0,MAtoms>(first,second,shared.codebook,bases0,bases1,a,sa,accumulators);
            else if(offset==1) iq2r_aligned_decode_mfma<1,MAtoms>(first,second,shared.codebook,bases0,bases1,a,sa,accumulators);
            else iq2r_aligned_decode_mfma<2,MAtoms>(first,second,shared.codebook,bases0,bases1,a,sa,accumulators);
        }
        // No wave may overwrite another wave's cache before its final read.
        __syncthreads();
#pragma unroll
        for(int m=0;m<MAtoms;++m)
        {
#pragma unroll
            for(int atom=0;atom<4;++atom) shared.reuse.partial[wave][atom][lane]=accumulators[m][atom];
            __syncthreads();
            if(wave<4)
            {
                const int local_row=m*16+lane_group*4+wave;
#pragma unroll
                for(int atom=0;atom<4;++atom)
                {
                    float value=0.0f;
#pragma unroll
                    for(int source=0;source<8;++source) value+=shared.reuse.partial[source][atom][lane][wave];
                    shared.gate[local_row][atom*16+lane_row]=__float2bfloat16(value);
                }
            }
            __syncthreads();
        }
        if(linear<Rows*4)
        {
            const int local_row=linear/4,quant_lane=linear%4;
            const int output_row=row_begin+local_row;
            using B=opus::vector_t<opus::bf16_t,8>;
            using Q=opus::vector_t<opus::fp8_t,8>;
            B values;float abs_max=1.0e-10f;
#pragma unroll
            for(int element=0;element<8;++element)
            {
                const int column=quant_lane*8+element;
                const float gate=__bfloat162float(shared.gate[local_row][2*column]);
                const float up=__bfloat162float(shared.gate[local_row][2*column+1]);
                const float swish=gate/(1.0f+__expf(-gate));
                const __hip_bfloat16 rounded=__float2bfloat16(swish*(up+0.0f));
                values[element]=__builtin_bit_cast(opus::bf16_t,rounded);
                abs_max=fmaxf(abs_max,fabsf(__bfloat162float(rounded)));
            }
            abs_max=fmaxf(abs_max,__shfl_xor(abs_max,1));
            abs_max=fmaxf(abs_max,__shfl_xor(abs_max,2));
            if(output_row<row_end)
            {
                const auto bs=fp_f32_to_e8m0_block_scale<kDefaultMxScaleRoundMode,MxDtype::FP8_E4M3>(abs_max);
                const float inverse=1.0f/bs.dq_scale;
                Q q;
#pragma unroll
                for(int element=0;element<8;++element) q[element]=opus::fp32_to_fp8(static_cast<float>(values[element])*inverse);
                *reinterpret_cast<Q*>(output+static_cast<int64_t>(output_row)*256+n_tile*32+quant_lane*8)=q;
                if(quant_lane==0) output_scales[static_cast<int64_t>(output_row)*8+n_tile]=bs.byte;
            }
        }
        __syncthreads();
    }
#endif
}

constexpr int kQuadBytes=2304;
struct IQ2RCompressedQuad { uint4 first; uint4 second; uint32_t metadata; };

__device__ __forceinline__ void iq2r_issue_quad(
    opus::gmem<uint8_t>& buffer,uint8_t* cache,int lane,int source_base)
{
    const int base=__builtin_amdgcn_readfirstlane(source_base);
    buffer.template async_load<16>(cache,lane*16,base,opus::number<0>{},opus::number<0>{});
    buffer.template async_load<16>(cache+1024,lane*16,base+1024,opus::number<0>{},opus::number<0>{});
    buffer.template async_load<4>(cache+2048,lane*4,base+2048,opus::number<0>{},opus::number<0>{});
    asm volatile("" ::: "memory");
}

__device__ __forceinline__ IQ2RCompressedQuad iq2r_read_quad(const uint8_t* cache,int lane)
{
    return {*reinterpret_cast<const uint4*>(cache+lane*16),
            *reinterpret_cast<const uint4*>(cache+1024+lane*16),
            *reinterpret_cast<const uint32_t*>(cache+2048+lane*4)};
}

template<int MAtoms>
__device__ __forceinline__ void iq2r_quad_decode_mfma(
    const IQ2RCompressedQuad& compressed,const uint64_t* codebook,uint32_t bases,
    const IQ2RActivationFragment* activation,const uint32_t* scale_a,
    opus::vector_t<float,4> (&accumulators)[MAtoms][4])
{
#pragma unroll
    for(int atom=0;atom<4;++atom)
    {
        const uint32_t lows=atom==0?compressed.first.x:atom==1?compressed.first.z:atom==2?compressed.second.x:compressed.second.z;
        const uint32_t signs=atom==0?compressed.first.y:atom==1?compressed.first.w:atom==2?compressed.second.y:compressed.second.w;
        const uint32_t highs=(compressed.metadata>>(atom*8))&15u;
        union { opus::i32x8_t words;uint64_t codewords[4]; } decoded;
#pragma unroll
        for(int word=0;word<4;++word)
        {
            const int index=((lows>>(word*8))&255u)|(((highs>>word)&1u)<<8);
            decoded.codewords[word]=apply_signs(codebook[index],(signs>>(word*8))&255u);
        }
        const uint32_t sb=((bases>>(atom*8))&255u)+((compressed.metadata>>(atom*8+4))&15u);
#pragma unroll
        for(int m=0;m<MAtoms;++m)
        {
            auto mma=opus::mfma<opus::fp8_t,opus::fp8_t,opus::fp32_t,16,16,128>{};
            accumulators[m][atom]=mma(activation[m].words,decoded.words,accumulators[m][atom],
                                      scale_a[m],sb,opus::number<0>{},opus::number<0>{});
        }
    }
}

template<int MAtoms>
__global__ __launch_bounds__(512,1) void iq2r_gate_quad_fused_kernel(
    const opus::fp8_t* __restrict__ activations,
    const uint8_t* __restrict__ scales,
    const uint8_t* __restrict__ all_data,
    const uint8_t* __restrict__ all_auxiliary,
    const int32_t* __restrict__ tasks,
    const int32_t* __restrict__ task_count,
    const int32_t* __restrict__ gather,
    opus::fp8_t* __restrict__ output,
    uint8_t* __restrict__ output_scales,
    int routes,int data_bytes,int auxiliary_bytes)
{
#if defined(__gfx950__)
    constexpr int K=6144, Rows=16*MAtoms, Subtasks=2/MAtoms;
    struct Storage {
        alignas(16) uint64_t codebook[kCodebookBytes/8];
        union {
            alignas(16) uint8_t cache[8][kQuadBytes];
            alignas(16) opus::vector_t<float,4> partial[8][4][64];
        } reuse;
        alignas(16) __hip_bfloat16 gate[Rows][64];
    };
    __shared__ Storage shared;
    const int lane=threadIdx.x,wave=threadIdx.y,linear=wave*64+lane;
    const int lane_row=lane%16,lane_group=lane/16;
    const int num_tasks=task_count[0];
    const int total=num_tasks*Subtasks*8;
    for(int work=blockIdx.x;work<total;work+=gridDim.x)
    {
        const int task=(work/Subtasks)%num_tasks;
        const int sub=work%Subtasks;
        const int n_tile=work/(num_tasks*Subtasks);
        const int row_begin=tasks[task*3]+sub*Rows;
        const int row_end=min(row_begin+Rows,tasks[task*3]+tasks[task*3+1]);
        const int expert=tasks[task*3+2];
        if(row_begin>=row_end || row_begin<0 || row_end>routes || expert<0 || expert>=257) continue;
        const uint8_t* data=all_data+static_cast<int64_t>(expert)*data_bytes;
        const uint8_t* aux=all_auxiliary+static_cast<int64_t>(expert)*auxiliary_bytes;
        // Previous work ends with a barrier protecting both union and codebook.
        shared.codebook[linear]=reinterpret_cast<const uint64_t*>(aux)[linear];
        __syncthreads();
        const int nbase=n_tile*4;
        uint32_t bases=0;
#pragma unroll
        for(int atom=0;atom<4;++atom)
            bases|=static_cast<uint32_t>(aux[kCodebookBytes+nbase+atom])<<(atom*8);
        int rows[MAtoms];
#pragma unroll
        for(int m=0;m<MAtoms;++m) rows[m]=gather[min(row_begin+m*16+lane_row,row_end-1)]/9;
        opus::vector_t<float,4> accumulators[MAtoms][4]={};
        opus::gmem<uint8_t> buffer(data,static_cast<unsigned int>(data_bytes));
        int base=(n_tile*48+wave*6)*kQuadBytes;
        iq2r_issue_quad(buffer,shared.reuse.cache[wave],lane,base);
        for(int iteration=0;iteration<6;++iteration)
        {
            const int kt=wave*6+iteration;
            IQ2RActivationFragment a[MAtoms]={};uint32_t sa[MAtoms];
#pragma unroll
            for(int m=0;m<MAtoms;++m)
            {
                const auto* input=reinterpret_cast<const uint8_t*>(activations+static_cast<int64_t>(rows[m])*K);
                const int ak=kt*128+lane_group*16;
                *reinterpret_cast<uint4*>(a[m].bytes)=*reinterpret_cast<const uint4*>(input+ak);
                *reinterpret_cast<uint4*>(a[m].bytes+16)=*reinterpret_cast<const uint4*>(input+ak+64);
                sa[m]=scales[static_cast<int64_t>(rows[m])*192+kt*4+lane_group]*0x01010101u;
            }
            iq2r_wait_vmcnt<0>();
            const auto compressed=iq2r_read_quad(shared.reuse.cache[wave],lane);
            asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
            if(iteration<5)
            {
                base+=kQuadBytes;
                iq2r_issue_quad(buffer,shared.reuse.cache[wave],lane,base);
            }
            iq2r_quad_decode_mfma<MAtoms>(compressed,shared.codebook,bases,a,sa,accumulators);
        }
        // No wave may overwrite another wave's cache before its final read.
        __syncthreads();
#pragma unroll
        for(int m=0;m<MAtoms;++m)
        {
#pragma unroll
            for(int atom=0;atom<4;++atom) shared.reuse.partial[wave][atom][lane]=accumulators[m][atom];
            __syncthreads();
            if(wave<4)
            {
                const int local_row=m*16+lane_group*4+wave;
#pragma unroll
                for(int atom=0;atom<4;++atom)
                {
                    float value=0.0f;
#pragma unroll
                    for(int source=0;source<8;++source) value+=shared.reuse.partial[source][atom][lane][wave];
                    shared.gate[local_row][atom*16+lane_row]=__float2bfloat16(value);
                }
            }
            __syncthreads();
        }
        if(linear<Rows*4)
        {
            const int local_row=linear/4,quant_lane=linear%4;
            const int output_row=row_begin+local_row;
            using B=opus::vector_t<opus::bf16_t,8>;
            using Q=opus::vector_t<opus::fp8_t,8>;
            B values;float abs_max=1.0e-10f;
#pragma unroll
            for(int element=0;element<8;++element)
            {
                const int column=quant_lane*8+element;
                const float gate=__bfloat162float(shared.gate[local_row][2*column]);
                const float up=__bfloat162float(shared.gate[local_row][2*column+1]);
                const float swish=gate/(1.0f+__expf(-gate));
                const __hip_bfloat16 rounded=__float2bfloat16(swish*(up+0.0f));
                values[element]=__builtin_bit_cast(opus::bf16_t,rounded);
                abs_max=fmaxf(abs_max,fabsf(__bfloat162float(rounded)));
            }
            abs_max=fmaxf(abs_max,__shfl_xor(abs_max,1));
            abs_max=fmaxf(abs_max,__shfl_xor(abs_max,2));
            if(output_row<row_end)
            {
                const auto bs=fp_f32_to_e8m0_block_scale<kDefaultMxScaleRoundMode,MxDtype::FP8_E4M3>(abs_max);
                const float inverse=1.0f/bs.dq_scale;
                Q q;
#pragma unroll
                for(int element=0;element<8;++element) q[element]=opus::fp32_to_fp8(static_cast<float>(values[element])*inverse);
                *reinterpret_cast<Q*>(output+static_cast<int64_t>(output_row)*256+n_tile*32+quant_lane*8)=q;
                if(quant_lane==0) output_scales[static_cast<int64_t>(output_row)*8+n_tile]=bs.byte;
            }
        }
        __syncthreads();
    }
#endif
}

template<int MAtoms>
__global__ __launch_bounds__(512,1) void iq2r_gate_quad_route_fused_kernel(
    const opus::fp8_t* __restrict__ activations,
    const uint8_t* __restrict__ scales,
    const uint8_t* __restrict__ all_data,
    const uint8_t* __restrict__ all_auxiliary,
    const int32_t* __restrict__ tasks,
    const int32_t* __restrict__ task_count,
    const int32_t* __restrict__ gather,
    opus::fp8_t* __restrict__ output,
    uint8_t* __restrict__ output_scales,
    int routes,int data_bytes,int auxiliary_bytes)
{
#if defined(__gfx950__)
    constexpr int K=6144, Rows=16*MAtoms, Subtasks=1;
    static_assert(MAtoms==1);
    struct Storage {
        alignas(16) uint64_t codebook[kCodebookBytes/8];
        union {
            alignas(16) uint8_t cache[8][kQuadBytes];
            alignas(16) opus::vector_t<float,4> partial[8][4][64];
        } reuse;
        alignas(16) __hip_bfloat16 gate[Rows][64];
    };
    __shared__ Storage shared;
    const int lane=threadIdx.x,wave=threadIdx.y,linear=wave*64+lane;
    const int lane_row=lane%16,lane_group=lane/16;
    const int num_tasks=task_count[0];
    const int total=num_tasks*Subtasks*8;
    for(int work=blockIdx.x;work<total;work+=gridDim.x)
    {
        const int task=(work/Subtasks)%num_tasks;
        const int sub=work%Subtasks;
        const int n_tile=work/(num_tasks*Subtasks);
        const int row_begin=tasks[task*3]+sub*Rows;
        const int row_end=min(row_begin+Rows,tasks[task*3]+tasks[task*3+1]);
        const int expert=tasks[task*3+2];
        if(row_begin>=row_end || row_begin<0 || row_end>routes || expert<0 || expert>=257) continue;
        const uint8_t* data=all_data+static_cast<int64_t>(expert)*data_bytes;
        const uint8_t* aux=all_auxiliary+static_cast<int64_t>(expert)*auxiliary_bytes;
        // Previous work ends with a barrier protecting both union and codebook.
        shared.codebook[linear]=reinterpret_cast<const uint64_t*>(aux)[linear];
        __syncthreads();
        const int nbase=n_tile*4;
        uint32_t bases=0;
#pragma unroll
        for(int atom=0;atom<4;++atom)
            bases|=static_cast<uint32_t>(aux[kCodebookBytes+nbase+atom])<<(atom*8);
        int rows[MAtoms];
#pragma unroll
        for(int m=0;m<MAtoms;++m) rows[m]=min(row_begin+m*16+lane_row,row_end-1);
        opus::vector_t<float,4> accumulators[MAtoms][4]={};
        opus::gmem<uint8_t> buffer(data,static_cast<unsigned int>(data_bytes));
        int base=(n_tile*48+wave*6)*kQuadBytes;
        iq2r_issue_quad(buffer,shared.reuse.cache[wave],lane,base);
        for(int iteration=0;iteration<6;++iteration)
        {
            const int kt=wave*6+iteration;
            IQ2RActivationFragment a[MAtoms]={};uint32_t sa[MAtoms];
#pragma unroll
            for(int m=0;m<MAtoms;++m)
            {
                const auto* input=reinterpret_cast<const uint8_t*>(activations+static_cast<int64_t>(rows[m])*K);
                const int ak=kt*128+lane_group*16;
                *reinterpret_cast<uint4*>(a[m].bytes)=*reinterpret_cast<const uint4*>(input+ak);
                *reinterpret_cast<uint4*>(a[m].bytes+16)=*reinterpret_cast<const uint4*>(input+ak+64);
                sa[m]=scales[static_cast<int64_t>(rows[m])*192+kt*4+lane_group]*0x01010101u;
            }
            iq2r_wait_vmcnt<0>();
            const auto compressed=iq2r_read_quad(shared.reuse.cache[wave],lane);
            asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
            if(iteration<5)
            {
                base+=kQuadBytes;
                iq2r_issue_quad(buffer,shared.reuse.cache[wave],lane,base);
            }
            iq2r_quad_decode_mfma<MAtoms>(compressed,shared.codebook,bases,a,sa,accumulators);
        }
        // No wave may overwrite another wave's cache before its final read.
        __syncthreads();
#pragma unroll
        for(int m=0;m<MAtoms;++m)
        {
#pragma unroll
            for(int atom=0;atom<4;++atom) shared.reuse.partial[wave][atom][lane]=accumulators[m][atom];
            __syncthreads();
            if(wave<4)
            {
                const int local_row=m*16+lane_group*4+wave;
#pragma unroll
                for(int atom=0;atom<4;++atom)
                {
                    float value=0.0f;
#pragma unroll
                    for(int source=0;source<8;++source) value+=shared.reuse.partial[source][atom][lane][wave];
                    shared.gate[local_row][atom*16+lane_row]=__float2bfloat16(value);
                }
            }
            __syncthreads();
        }
        if(linear<Rows*4)
        {
            const int local_row=linear/4,quant_lane=linear%4;
            const int output_row=row_begin+local_row;
            using B=opus::vector_t<opus::bf16_t,8>;
            using Q=opus::vector_t<opus::fp8_t,8>;
            B values;float abs_max=1.0e-10f;
#pragma unroll
            for(int element=0;element<8;++element)
            {
                const int column=quant_lane*8+element;
                const float gate=__bfloat162float(shared.gate[local_row][2*column]);
                const float up=__bfloat162float(shared.gate[local_row][2*column+1]);
                const float swish=gate/(1.0f+__expf(-gate));
                const __hip_bfloat16 rounded=__float2bfloat16(swish*(up+0.0f));
                values[element]=__builtin_bit_cast(opus::bf16_t,rounded);
                abs_max=fmaxf(abs_max,fabsf(__bfloat162float(rounded)));
            }
            abs_max=fmaxf(abs_max,__shfl_xor(abs_max,1));
            abs_max=fmaxf(abs_max,__shfl_xor(abs_max,2));
            if(output_row<row_end)
            {
                const auto bs=fp_f32_to_e8m0_block_scale<kDefaultMxScaleRoundMode,MxDtype::FP8_E4M3>(abs_max);
                const float inverse=1.0f/bs.dq_scale;
                Q q;
#pragma unroll
                for(int element=0;element<8;++element) q[element]=opus::fp32_to_fp8(static_cast<float>(values[element])*inverse);
                *reinterpret_cast<Q*>(output+static_cast<int64_t>(output_row)*256+n_tile*32+quant_lane*8)=q;
                if(quant_lane==0) output_scales[static_cast<int64_t>(output_row)*8+n_tile]=bs.byte;
            }
        }
        __syncthreads();
    }
#endif
}

template<int MAtoms>
__device__ __forceinline__ void iq2r_quad_sparse_decode_mfma(
    const IQ2RCompressedQuad& compressed,const uint64_t* codebook,uint32_t bases,
    const IQ2RActivationFragment* activation,const uint32_t* scale_a,
    opus::vector_t<float,4> (&accumulators)[MAtoms][4], int active_m)
{
#pragma unroll
    for(int atom=0;atom<4;++atom)
    {
        const uint32_t lows=atom==0?compressed.first.x:atom==1?compressed.first.z:atom==2?compressed.second.x:compressed.second.z;
        const uint32_t signs=atom==0?compressed.first.y:atom==1?compressed.first.w:atom==2?compressed.second.y:compressed.second.w;
        const uint32_t highs=(compressed.metadata>>(atom*8))&15u;
        union { opus::i32x8_t words;uint64_t codewords[4]; } decoded;
#pragma unroll
        for(int word=0;word<4;++word)
        {
            const int index=((lows>>(word*8))&255u)|(((highs>>word)&1u)<<8);
            decoded.codewords[word]=apply_signs(codebook[index],(signs>>(word*8))&255u);
        }
        const uint32_t sb=((bases>>(atom*8))&255u)+((compressed.metadata>>(atom*8+4))&15u);
#pragma unroll
        for(int m=0;m<MAtoms;++m)
        {
            if(m>=active_m) continue;
            auto mma=opus::mfma<opus::fp8_t,opus::fp8_t,opus::fp32_t,16,16,128>{};
            accumulators[m][atom]=mma(activation[m].words,decoded.words,accumulators[m][atom],
                                      scale_a[m],sb,opus::number<0>{},opus::number<0>{});
        }
    }
}
template<int MAtoms>
__global__ __launch_bounds__(512,1) void iq2r_gate_quad_sparse_kernel(
    const opus::fp8_t* __restrict__ activations,
    const uint8_t* __restrict__ scales,
    const uint8_t* __restrict__ all_data,
    const uint8_t* __restrict__ all_auxiliary,
    const int32_t* __restrict__ tasks,
    const int32_t* __restrict__ task_count,
    const int32_t* __restrict__ gather,
    opus::fp8_t* __restrict__ output,
    uint8_t* __restrict__ output_scales,
    int routes,int data_bytes,int auxiliary_bytes)
{
#if defined(__gfx950__)
    constexpr int K=6144, Rows=16*MAtoms, Subtasks=2/MAtoms;
    struct Storage {
        alignas(16) uint64_t codebook[kCodebookBytes/8];
        union {
            alignas(16) uint8_t cache[8][kQuadBytes];
            alignas(16) opus::vector_t<float,4> partial[8][4][64];
        } reuse;
        alignas(16) __hip_bfloat16 gate[Rows][64];
    };
    __shared__ Storage shared;
    const int lane=threadIdx.x,wave=threadIdx.y,linear=wave*64+lane;
    const int lane_row=lane%16,lane_group=lane/16;
    const int num_tasks=task_count[0];
    const int total=num_tasks*Subtasks*8;
    for(int work=blockIdx.x;work<total;work+=gridDim.x)
    {
        const int task=(work/Subtasks)%num_tasks;
        const int sub=work%Subtasks;
        const int n_tile=work/(num_tasks*Subtasks);
        const int row_begin=tasks[task*3]+sub*Rows;
        const int row_end=min(row_begin+Rows,tasks[task*3]+tasks[task*3+1]);
        const int expert=tasks[task*3+2];
        const int active_m=(row_end-row_begin+15)/16;
        if(row_begin>=row_end || row_begin<0 || row_end>routes || expert<0 || expert>=257) continue;
        const uint8_t* data=all_data+static_cast<int64_t>(expert)*data_bytes;
        const uint8_t* aux=all_auxiliary+static_cast<int64_t>(expert)*auxiliary_bytes;
        // Previous work ends with a barrier protecting both union and codebook.
        shared.codebook[linear]=reinterpret_cast<const uint64_t*>(aux)[linear];
        __syncthreads();
        const int nbase=n_tile*4;
        uint32_t bases=0;
#pragma unroll
        for(int atom=0;atom<4;++atom)
            bases|=static_cast<uint32_t>(aux[kCodebookBytes+nbase+atom])<<(atom*8);
        int rows[MAtoms];
#pragma unroll
        for(int m=0;m<MAtoms;++m) rows[m]=gather[min(row_begin+m*16+lane_row,row_end-1)]/9;
        opus::vector_t<float,4> accumulators[MAtoms][4]={};
        opus::gmem<uint8_t> buffer(data,static_cast<unsigned int>(data_bytes));
        int base=(n_tile*48+wave*6)*kQuadBytes;
        iq2r_issue_quad(buffer,shared.reuse.cache[wave],lane,base);
        for(int iteration=0;iteration<6;++iteration)
        {
            const int kt=wave*6+iteration;
            IQ2RActivationFragment a[MAtoms]={};uint32_t sa[MAtoms];
#pragma unroll
            for(int m=0;m<MAtoms;++m)
            {
                if(m>=active_m) continue;
                const auto* input=reinterpret_cast<const uint8_t*>(activations+static_cast<int64_t>(rows[m])*K);
                const int ak=kt*128+lane_group*16;
                *reinterpret_cast<uint4*>(a[m].bytes)=*reinterpret_cast<const uint4*>(input+ak);
                *reinterpret_cast<uint4*>(a[m].bytes+16)=*reinterpret_cast<const uint4*>(input+ak+64);
                sa[m]=scales[static_cast<int64_t>(rows[m])*192+kt*4+lane_group]*0x01010101u;
            }
            iq2r_wait_vmcnt<0>();
            const auto compressed=iq2r_read_quad(shared.reuse.cache[wave],lane);
            asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
            if(iteration<5)
            {
                base+=kQuadBytes;
                iq2r_issue_quad(buffer,shared.reuse.cache[wave],lane,base);
            }
            iq2r_quad_sparse_decode_mfma<MAtoms>(compressed,shared.codebook,bases,a,sa,accumulators,active_m);
        }
        // No wave may overwrite another wave's cache before its final read.
        __syncthreads();
#pragma unroll
        for(int m=0;m<MAtoms;++m)
        {
            if(m>=active_m) continue;
#pragma unroll
            for(int atom=0;atom<4;++atom) shared.reuse.partial[wave][atom][lane]=accumulators[m][atom];
            __syncthreads();
            if(wave<4)
            {
                const int local_row=m*16+lane_group*4+wave;
#pragma unroll
                for(int atom=0;atom<4;++atom)
                {
                    float value=0.0f;
#pragma unroll
                    for(int source=0;source<8;++source) value+=shared.reuse.partial[source][atom][lane][wave];
                    shared.gate[local_row][atom*16+lane_row]=__float2bfloat16(value);
                }
            }
            __syncthreads();
        }
        if(linear<active_m*16*4)
        {
            const int local_row=linear/4,quant_lane=linear%4;
            const int output_row=row_begin+local_row;
            using B=opus::vector_t<opus::bf16_t,8>;
            using Q=opus::vector_t<opus::fp8_t,8>;
            B values;float abs_max=1.0e-10f;
#pragma unroll
            for(int element=0;element<8;++element)
            {
                const int column=quant_lane*8+element;
                const float gate=__bfloat162float(shared.gate[local_row][2*column]);
                const float up=__bfloat162float(shared.gate[local_row][2*column+1]);
                const float swish=gate/(1.0f+__expf(-gate));
                const __hip_bfloat16 rounded=__float2bfloat16(swish*(up+0.0f));
                values[element]=__builtin_bit_cast(opus::bf16_t,rounded);
                abs_max=fmaxf(abs_max,fabsf(__bfloat162float(rounded)));
            }
            abs_max=fmaxf(abs_max,__shfl_xor(abs_max,1));
            abs_max=fmaxf(abs_max,__shfl_xor(abs_max,2));
            if(output_row<row_end)
            {
                const auto bs=fp_f32_to_e8m0_block_scale<kDefaultMxScaleRoundMode,MxDtype::FP8_E4M3>(abs_max);
                const float inverse=1.0f/bs.dq_scale;
                Q q;
#pragma unroll
                for(int element=0;element<8;++element) q[element]=opus::fp32_to_fp8(static_cast<float>(values[element])*inverse);
                *reinterpret_cast<Q*>(output+static_cast<int64_t>(output_row)*256+n_tile*32+quant_lane*8)=q;
                if(quant_lane==0) output_scales[static_cast<int64_t>(output_row)*8+n_tile]=bs.byte;
            }
        }
        __syncthreads();
    }
#endif
}
// Split the existing eight K partitions across CTAs without changing their sums.
template<int PhysicalWaves>
__global__ __launch_bounds__(64*PhysicalWaves) void iq2r_gate_quad_splitk_kernel(
    const opus::fp8_t* __restrict__ input, const uint8_t* __restrict__ scales,
    const uint8_t* __restrict__ all_data, const uint8_t* __restrict__ all_aux,
    const int32_t* __restrict__ tasks, const int32_t* __restrict__ task_count,
    float* __restrict__ partials, int routes, int data_bytes, int aux_bytes)
{
#if defined(__gfx950__)
    static_assert(PhysicalWaves==1 || PhysicalWaves==2 || PhysicalWaves==4);
    constexpr int Parts=8/PhysicalWaves;
    __shared__ alignas(16) uint64_t codebook[512];
    __shared__ alignas(16) uint8_t cache[PhysicalWaves][kQuadBytes];
    const int lane=threadIdx.x, wave=threadIdx.y;
    const int linear=wave*64+lane;
    const int count=task_count[0];
    for(int work=blockIdx.x;work<count*8*Parts;work+=gridDim.x)
    {
        const int task=work%count, n_tile=(work/count)%8, part=work/(count*8);
        const int row_begin=tasks[task*3], row_end=row_begin+tasks[task*3+1];
        const int expert=tasks[task*3+2];
        if(row_begin<0 || row_end>routes || row_end<=row_begin || row_end-row_begin>16 || expert<0 || expert>=257) continue;
        const auto* data=all_data+static_cast<int64_t>(expert)*data_bytes;
        const auto* aux=all_aux+static_cast<int64_t>(expert)*aux_bytes;
        for(int i=linear;i<512;i+=64*PhysicalWaves)
            codebook[i]=reinterpret_cast<const uint64_t*>(aux)[i];
        __syncthreads();
        const int global_wave=part*PhysicalWaves+wave;
        const int input_row=min(row_begin+lane%16,row_end-1);
        const auto* row=reinterpret_cast<const uint8_t*>(input+static_cast<int64_t>(input_row)*6144);
        uint32_t bases=0;
#pragma unroll
        for(int atom=0;atom<4;++atom) bases|=static_cast<uint32_t>(aux[kCodebookBytes+n_tile*4+atom])<<(atom*8);
        opus::vector_t<float,4> accumulator[1][4]={};
        opus::gmem<uint8_t> buffer(data,static_cast<unsigned int>(data_bytes));
        int base=(n_tile*48+global_wave*6)*kQuadBytes;
        iq2r_issue_quad(buffer,cache[wave],lane,base);
        for(int iteration=0;iteration<6;++iteration)
        {
            const int kt=global_wave*6+iteration, ak=kt*128+(lane/16)*16;
            IQ2RActivationFragment activation[1]={};
            *reinterpret_cast<uint4*>(activation[0].bytes)=*reinterpret_cast<const uint4*>(row+ak);
            *reinterpret_cast<uint4*>(activation[0].bytes+16)=*reinterpret_cast<const uint4*>(row+ak+64);
            uint32_t sa[1]={scales[static_cast<int64_t>(input_row)*192+kt*4+lane/16]*0x01010101u};
            iq2r_wait_vmcnt<0>();
            const auto compressed=iq2r_read_quad(cache[wave],lane);
            asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
            if(iteration<5)
            {
                base+=kQuadBytes;
                iq2r_issue_quad(buffer,cache[wave],lane,base);
            }
            iq2r_quad_decode_mfma<1>(compressed,codebook,bases,activation,sa,accumulator);
        }
#pragma unroll
        for(int atom=0;atom<4;++atom)
#pragma unroll
            for(int item=0;item<4;++item)
            {
                const int out_row=row_begin+(lane/16)*4+item;
                const int out_col=n_tile*64+atom*16+lane%16;
                if(out_row<row_end)
                    partials[(static_cast<int64_t>(global_wave)*routes+out_row)*512+out_col]=accumulator[0][atom][item];
            }
        __syncthreads();
    }
#endif
}

__global__ __launch_bounds__(64) void iq2r_gate_splitk_swiglu_quant_kernel(
    const float* __restrict__ partials, opus::fp8_t* __restrict__ output,
    uint8_t* __restrict__ scales, int routes)
{
#if defined(__gfx950__)
    const int index=blockIdx.x*64+threadIdx.x, row=index/256, col=index%256;
    float gate=0.0f, up=0.0f;
#pragma unroll
    for(int wave=0;wave<8;++wave)
    {
        const int64_t base=(static_cast<int64_t>(wave)*routes+row)*512+col*2;
        gate+=partials[base];up+=partials[base+1];
    }
    gate=__bfloat162float(__float2bfloat16(gate));
    up=__bfloat162float(__float2bfloat16(up));
    const float value=__bfloat162float(__float2bfloat16((gate/(1.0f+__expf(-gate)))*(up+0.0f)));
    float abs_max=fmaxf(fabsf(value),1.0e-10f);
#pragma unroll
    for(int delta=16;delta>0;delta/=2) abs_max=fmaxf(abs_max,__shfl_xor(abs_max,delta,32));
    const auto bs=fp_f32_to_e8m0_block_scale<kDefaultMxScaleRoundMode,MxDtype::FP8_E4M3>(abs_max);
    const float inverse=1.0f/bs.dq_scale;
    output[index]=opus::fp32_to_fp8(value*inverse);
    if(col%32==0) scales[row*8+col/32]=bs.byte;
#endif
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

using iq2r_scheduled_i32x3 = int __attribute__((ext_vector_type(3)));
__device__ iq2r_scheduled_i32x3 iq2r_scheduled_load_dwordx3(opus::i32x4_t, int, int, int)
    __asm("llvm.amdgcn.raw.buffer.load.v3i32");

__device__ __forceinline__ IQ2RCompressedTriplet iq2r_scheduled_load_compact(
    opus::gmem<uint8_t>& buffer, int base, int lane)
{
    IQ2RCompressedTriplet out;
    out.paired = __builtin_bit_cast(uint4, buffer.template load<16>(lane*16, base, opus::number<2>{}));
    opus::i32x4_t rsrc;
    __builtin_memcpy(&rsrc, &buffer.cached_rsrc, sizeof(rsrc));
    const auto third = iq2r_scheduled_load_dwordx3(rsrc, lane*12+kAtomTwoOffset, base, 2);
    out.third = make_uint4(third[0], third[1], third[2], 0);
    asm volatile("" ::: "memory");
    return out;
}

__device__ opus::i32x4_t iq2r_scheduled_load_dwordx4(opus::i32x4_t, int, int, int)
    __asm("llvm.amdgcn.raw.buffer.load.v4i32");

__device__ __forceinline__ IQ2RCompressedTriplet iq2r_scheduled_load_compact_uniform(
    const uint8_t* data, int data_bytes, int base, int lane)
{
    const uint64_t address = reinterpret_cast<uint64_t>(data);
    const opus::i32x4_t rsrc = {
        static_cast<int>(__builtin_amdgcn_readfirstlane(static_cast<uint32_t>(address))),
        static_cast<int>(__builtin_amdgcn_readfirstlane(static_cast<uint32_t>(address >> 32))),
        __builtin_amdgcn_readfirstlane(data_bytes),
        static_cast<int>(opus::buffer_default_config())};
    const int scalar_base = __builtin_amdgcn_readfirstlane(base);
    IQ2RCompressedTriplet out;
    out.paired = __builtin_bit_cast(uint4, iq2r_scheduled_load_dwordx4(rsrc, lane*16, scalar_base, 2));
    const auto third = iq2r_scheduled_load_dwordx3(rsrc, lane*12+kAtomTwoOffset, scalar_base, 2);
    out.third = make_uint4(third[0], third[1], third[2], 0);
    asm volatile("" ::: "memory");
    return out;
}
__device__ opus::i32x4_t iq2r_scheduled_buffer_load4(opus::i32x4_t, int, int, int)
    __asm("llvm.amdgcn.raw.buffer.load.v4i32");
__device__ int iq2r_scheduled_buffer_load1(opus::i32x4_t, int, int, int)
    __asm("llvm.amdgcn.raw.buffer.load.i32");

__device__ __forceinline__ IQ2RCompressedQuad iq2r_scheduled_load_quad(
    const uint8_t* data, int data_bytes, int base, int lane)
{
    const uint64_t address=reinterpret_cast<uint64_t>(data);
    const opus::i32x4_t rsrc={
        static_cast<int>(__builtin_amdgcn_readfirstlane(static_cast<uint32_t>(address))),
        static_cast<int>(__builtin_amdgcn_readfirstlane(static_cast<uint32_t>(address>>32))),
        __builtin_amdgcn_readfirstlane(data_bytes),
        static_cast<int>(opus::buffer_default_config())};
    const int sb=__builtin_amdgcn_readfirstlane(base);
    IQ2RCompressedQuad out;
    out.first=__builtin_bit_cast(uint4,iq2r_scheduled_buffer_load4(rsrc,lane*16,sb,0));
    out.second=__builtin_bit_cast(uint4,iq2r_scheduled_buffer_load4(rsrc,lane*16+1024,sb,0));
    out.metadata=iq2r_scheduled_buffer_load1(rsrc,lane*4+2048,sb,0);
    asm volatile("" ::: "memory");
    return out;
}

__global__ __launch_bounds__(256, 2) void iq2r_scheduled_large_fixed_kn_kernel(
    const opus::fp8_t* __restrict__ activations,
    const uint8_t* __restrict__ activation_scales,
    const uint8_t* __restrict__ all_data,
    const uint8_t* __restrict__ all_auxiliary,
    const int32_t* __restrict__ tasks,
    const int32_t* __restrict__ task_count,
    const __hip_bfloat16* __restrict__ all_bias,
    __hip_bfloat16* __restrict__ output,
    int M,
    int unused_N,
    int unused_K,
    int expert_count,
    int data_bytes,
    int auxiliary_bytes)
{
#if defined(__gfx950__)
    constexpr int N = 6144;
    constexpr int K = 256;
    constexpr int MAtoms = 2;
    constexpr bool NMajor = false;
    constexpr int kPhysicalWaves = 4;
    static_assert(MAtoms == 2 || MAtoms == 4);
    constexpr int kMAtoms = MAtoms;
    constexpr int kNAtomsPerWave = 3;
    constexpr int kOutputColumns =
        kPhysicalWaves * kNAtomsPerWave * kTileN;

    struct SharedStorage
    {
        alignas(16) uint64_t codebook[kCodebookBytes / sizeof(uint64_t)];
        alignas(16) uint8_t weight_cache[kPhysicalWaves][kTripletLDSBytes];
    };
    __shared__ SharedStorage shared;

    const int lane = static_cast<int>(threadIdx.x);
    const int wave = static_cast<int>(threadIdx.y);
    const int linear_thread = wave * 64 + lane;
    const int lane_row = lane % 16;
    const int lane_group = lane / 16;
    const int num_tasks = task_count[0];
    if(num_tasks <= 0)
        return;

    const int n_tiles = (N + kOutputColumns - 1) / kOutputColumns;
    const int total_tiles = num_tasks * n_tiles;
    const int k_tiles = (K + kTileK - 1) / kTileK;
    int previous_expert = -1;

    for(int work_index = static_cast<int>(blockIdx.x); work_index < total_tiles;
        work_index += static_cast<int>(gridDim.x))
    {
        const int task_index =
            NMajor ? work_index % num_tasks : work_index / n_tiles;
        const int n_tile_index =
            NMajor ? work_index / num_tasks : work_index % n_tiles;
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
            // Protect readers of the old expert before reusing the codebook.
            if(previous_expert >= 0)
                __syncthreads();
            for(int entry = linear_thread;
                entry < kCodebookBytes / static_cast<int>(sizeof(uint64_t));
                entry += 256)
                shared.codebook[entry] =
                    reinterpret_cast<const uint64_t*>(auxiliary)[entry];
            __syncthreads();
            previous_expert = expert_index;
        }

        const int n_block_base =
            n_tile_index * (kOutputColumns / kTileN) +
            wave * kNAtomsPerWave;
        uint32_t packed_bases = 0;
#pragma unroll
        for(int atom = 0; atom < kNAtomsPerWave; ++atom)
            packed_bases |=
                static_cast<uint32_t>(
                    auxiliary[kCodebookBytes + n_block_base + atom])
                << (atom * 8);

        const int row_end = row_begin + row_count;
        for(int row_base = row_begin; row_base < row_end;
            row_base += kMAtoms * 16)
        {
            const int sub_end = min(row_base + kMAtoms * 16, row_end);
            opus::vector_t<float, 4>
                accumulators[kMAtoms][kNAtomsPerWave] = {};

            int next_data_base = static_cast<int>(triplet_base(
                physical_tile(n_block_base, 0, k_tiles)));
            iq2r_issue_direct_lds_triplet(data_buffer,
                                           shared.weight_cache[wave],
                                           lane,
                                           0,
                                           next_data_base);
            next_data_base += kGroupBytes;

            for(int k_tile = 0; k_tile < k_tiles; ++k_tile)
            {
                union ActivationFragment
                {
                    opus::i32x8_t words;
                    uint8_t bytes[32];
                } activation_fragments[kMAtoms] = {};
                uint32_t scale_a[kMAtoms];
#pragma unroll
                for(int m_atom = 0; m_atom < kMAtoms; ++m_atom)
                    scale_a[m_atom] = 127u * 0x01010101u;

#pragma unroll
                for(int m_atom = 0; m_atom < kMAtoms; ++m_atom)
                {
                    const int input_row = row_base + m_atom * 16 + lane_row;
                    if(input_row < sub_end)
                    {
                        const int activation_k =
                            k_tile * kTileK + lane_group * 16;
                        const auto* activation_row =
                            reinterpret_cast<const uint8_t*>(
                                activations + static_cast<int64_t>(input_row) * K);
                        if(activation_k + 15 < K)
                            *reinterpret_cast<uint4*>(
                                activation_fragments[m_atom].bytes) =
                                *reinterpret_cast<const uint4*>(
                                    activation_row + activation_k);
                        if(activation_k + 64 + 15 < K)
                            *reinterpret_cast<uint4*>(
                                activation_fragments[m_atom].bytes + 16) =
                                *reinterpret_cast<const uint4*>(
                                    activation_row + activation_k + 64);

                        const int scale_column = k_tile * 4 + lane_group;
                        if(scale_column < K / kScaleBlock)
                        {
                            const uint32_t exponent = activation_scales[
                                static_cast<int64_t>(input_row) *
                                    (K / kScaleBlock) +
                                scale_column];
                            scale_a[m_atom] = exponent * 0x01010101u;
                        }
                    }
                }

                // Two weight requests precede eight activation-vector and four
                // activation-scale requests on full K tiles. Wait only for the
                // older weight pair, leaving that activation tail in flight
                // while the compact record moves out of LDS. The padded final
                // K tile omits the second vector load and two scale groups.
                // Partial slabs issue fewer loads than the fixed wait allowance.
                if(sub_end - row_base < kMAtoms * 16)
                    __builtin_amdgcn_s_waitcnt(0x0F70);
                else if(k_tile + 1 < k_tiles)
                    __builtin_amdgcn_s_waitcnt(0x0F7C); // vmcnt(12)
                else
                    __builtin_amdgcn_s_waitcnt(0x0F78); // vmcnt(8)
                const IQ2RCompressedTriplet compressed =
                    iq2r_read_direct_lds_triplet(
                        shared.weight_cache[wave], lane, 0);
                asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
                const bool has_next = k_tile + 1 < k_tiles;
                if(has_next)
                {
                    iq2r_issue_direct_lds_triplet(data_buffer,
                                                   shared.weight_cache[wave],
                                                   lane,
                                                   0,
                                                   next_data_base);
                    next_data_base += kGroupBytes;
                }
                opus::i32x8_t weight_fragments[kNAtomsPerWave];
                const uint32_t scale_b = iq2r_decode_direct_lds_triplet(
                    compressed,
                    shared.codebook,
                    packed_bases,
                    weight_fragments);
                if(has_next)
                    __builtin_amdgcn_s_waitcnt(0x0F72); // vmcnt(2)
                else
                    __builtin_amdgcn_s_waitcnt(0x0F70); // vmcnt(0)
#pragma unroll
                for(int m_atom = 0; m_atom < kMAtoms; ++m_atom)
                    if(row_base + m_atom * 16 < sub_end)
                    iq2r_cooperative_triplet_mfma<0>(
                        activation_fragments[m_atom].words,
                        weight_fragments,
                        accumulators[m_atom],
                        scale_a[m_atom],
                        scale_b);
            }

#pragma unroll
            for(int m_atom = 0; m_atom < kMAtoms; ++m_atom)
            {
                if(row_base + m_atom * 16 >= sub_end) continue;
#pragma unroll
                for(int atom = 0; atom < kNAtomsPerWave; ++atom)
                {
                    const int output_column =
                        (n_block_base + atom) * kTileN + lane_row;
#pragma unroll
                    for(int item = 0; item < 4; ++item)
                    {
                        const int output_row =
                            row_base + m_atom * 16 + lane_group * 4 + item;
                        if(output_row < sub_end && output_column < N)
                        {
                            float value = accumulators[m_atom][atom][item];
                            if(all_bias != nullptr)
                                value += __bfloat162float(all_bias[
                                    static_cast<int64_t>(expert_index) * N +
                                    output_column]);
                            output[static_cast<int64_t>(output_row) * N +
                                   output_column] = __float2bfloat16(value);
                        }
                    }
                }
            }
        }
    }
#endif
}



__global__ __launch_bounds__(256, 2) void iq2r_scheduled_large_register_prefetch_uniform_kernel(
    const opus::fp8_t* __restrict__ activations,
    const uint8_t* __restrict__ activation_scales,
    const uint8_t* __restrict__ all_data,
    const uint8_t* __restrict__ all_auxiliary,
    const int32_t* __restrict__ tasks,
    const int32_t* __restrict__ task_count,
    const __hip_bfloat16* __restrict__ all_bias,
    __hip_bfloat16* __restrict__ output,
    int M,
    int unused_N,
    int unused_K,
    int expert_count,
    int data_bytes,
    int auxiliary_bytes)
{
#if defined(__gfx950__)
    constexpr int N = 6144;
    constexpr int K = 256;
    constexpr int MAtoms = 2;
    constexpr bool NMajor = false;
    constexpr int kPhysicalWaves = 4;
    static_assert(MAtoms == 2 || MAtoms == 4);
    constexpr int kMAtoms = MAtoms;
    constexpr int kNAtomsPerWave = 3;
    constexpr int kOutputColumns =
        kPhysicalWaves * kNAtomsPerWave * kTileN;

    struct SharedStorage
    {
        alignas(16) uint64_t codebook[kCodebookBytes / sizeof(uint64_t)];
    };
    __shared__ SharedStorage shared;

    const int lane = static_cast<int>(threadIdx.x);
    const int wave = static_cast<int>(threadIdx.y);
    const int linear_thread = wave * 64 + lane;
    const int lane_row = lane % 16;
    const int lane_group = lane / 16;
    const int num_tasks = task_count[0];
    if(num_tasks <= 0)
        return;

    const int n_tiles = (N + kOutputColumns - 1) / kOutputColumns;
    const int total_tiles = num_tasks * n_tiles;
    const int k_tiles = (K + kTileK - 1) / kTileK;
    int previous_expert = -1;

    for(int work_index = static_cast<int>(blockIdx.x); work_index < total_tiles;
        work_index += static_cast<int>(gridDim.x))
    {
        const int task_index =
            NMajor ? work_index % num_tasks : work_index / n_tiles;
        const int n_tile_index =
            NMajor ? work_index / num_tasks : work_index % n_tiles;
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
            // Protect readers of the old expert before reusing the codebook.
            if(previous_expert >= 0)
                __syncthreads();
            for(int entry = linear_thread;
                entry < kCodebookBytes / static_cast<int>(sizeof(uint64_t));
                entry += 256)
                shared.codebook[entry] =
                    reinterpret_cast<const uint64_t*>(auxiliary)[entry];
            __syncthreads();
            previous_expert = expert_index;
        }

        const int n_block_base =
            n_tile_index * (kOutputColumns / kTileN) +
            wave * kNAtomsPerWave;
        uint32_t packed_bases = 0;
#pragma unroll
        for(int atom = 0; atom < kNAtomsPerWave; ++atom)
            packed_bases |=
                static_cast<uint32_t>(
                    auxiliary[kCodebookBytes + n_block_base + atom])
                << (atom * 8);

        const int row_end = row_begin + row_count;
        for(int row_base = row_begin; row_base < row_end;
            row_base += kMAtoms * 16)
        {
            const int sub_end = min(row_base + kMAtoms * 16, row_end);
            opus::vector_t<float, 4>
                accumulators[kMAtoms][kNAtomsPerWave] = {};

            int next_data_base = static_cast<int>(triplet_base(
                physical_tile(n_block_base, 0, k_tiles)));
            IQ2RCompressedTriplet pending = iq2r_scheduled_load_compact_uniform(data, data_bytes, next_data_base, lane);
            next_data_base += kGroupBytes;

#pragma unroll 1
            for(int k_tile = 0; k_tile < k_tiles; ++k_tile)
            {
                union ActivationFragment
                {
                    opus::i32x8_t words;
                    uint8_t bytes[32];
                } activation_fragments[kMAtoms] = {};
                uint32_t scale_a[kMAtoms];
#pragma unroll
                for(int m_atom = 0; m_atom < kMAtoms; ++m_atom)
                    scale_a[m_atom] = 127u * 0x01010101u;

#pragma unroll
                for(int m_atom = 0; m_atom < kMAtoms; ++m_atom)
                {
                    const int input_row = row_base + m_atom * 16 + lane_row;
                    if(input_row < sub_end)
                    {
                        const int activation_k =
                            k_tile * kTileK + lane_group * 16;
                        const auto* activation_row =
                            reinterpret_cast<const uint8_t*>(
                                activations + static_cast<int64_t>(input_row) * K);
                        if(activation_k + 15 < K)
                            *reinterpret_cast<uint4*>(
                                activation_fragments[m_atom].bytes) =
                                *reinterpret_cast<const uint4*>(
                                    activation_row + activation_k);
                        if(activation_k + 64 + 15 < K)
                            *reinterpret_cast<uint4*>(
                                activation_fragments[m_atom].bytes + 16) =
                                *reinterpret_cast<const uint4*>(
                                    activation_row + activation_k + 64);

                        const int scale_column = k_tile * 4 + lane_group;
                        if(scale_column < K / kScaleBlock)
                        {
                            const uint32_t exponent = activation_scales[
                                static_cast<int64_t>(input_row) *
                                    (K / kScaleBlock) +
                                scale_column];
                            scale_a[m_atom] = exponent * 0x01010101u;
                        }
                    }
                }

                const IQ2RCompressedTriplet compressed = pending;
                if(k_tile + 1 < k_tiles)
                {
                    pending = iq2r_scheduled_load_compact_uniform(data, data_bytes, next_data_base, lane);
                    next_data_base += kGroupBytes;
                }
                opus::i32x8_t weight_fragments[kNAtomsPerWave];
                const uint32_t scale_b = iq2r_decode_direct_lds_triplet(
                    compressed,
                    shared.codebook,
                    packed_bases,
                    weight_fragments);
#pragma unroll
                for(int m_atom = 0; m_atom < kMAtoms; ++m_atom)
                    if(row_base + m_atom * 16 < sub_end)
                    iq2r_cooperative_triplet_mfma<0>(
                        activation_fragments[m_atom].words,
                        weight_fragments,
                        accumulators[m_atom],
                        scale_a[m_atom],
                        scale_b);
            }

#pragma unroll
            for(int m_atom = 0; m_atom < kMAtoms; ++m_atom)
            {
                if(row_base + m_atom * 16 >= sub_end) continue;
#pragma unroll
                for(int atom = 0; atom < kNAtomsPerWave; ++atom)
                {
                    const int output_column =
                        (n_block_base + atom) * kTileN + lane_row;
#pragma unroll
                    for(int item = 0; item < 4; ++item)
                    {
                        const int output_row =
                            row_base + m_atom * 16 + lane_group * 4 + item;
                        if(output_row < sub_end && output_column < N)
                        {
                            float value = accumulators[m_atom][atom][item];
                            if(all_bias != nullptr)
                                value += __bfloat162float(all_bias[
                                    static_cast<int64_t>(expert_index) * N +
                                    output_column]);
                            output[static_cast<int64_t>(output_row) * N +
                                   output_column] = __float2bfloat16(value);
                        }
                    }
                }
            }
        }
    }
#endif
}



__global__ __launch_bounds__(256, 2) void iq2r_scheduled_large_fixed_kn_deferred_scale_kernel(
    const opus::fp8_t* __restrict__ activations,
    const uint8_t* __restrict__ activation_scales,
    const uint8_t* __restrict__ all_data,
    const uint8_t* __restrict__ all_auxiliary,
    const int32_t* __restrict__ tasks,
    const int32_t* __restrict__ task_count,
    const __hip_bfloat16* __restrict__ all_bias,
    __hip_bfloat16* __restrict__ output,
    int M,
    int unused_N,
    int unused_K,
    int expert_count,
    int data_bytes,
    int auxiliary_bytes)
{
#if defined(__gfx950__)
    constexpr int N = 6144;
    constexpr int K = 256;
    constexpr int MAtoms = 2;
    constexpr bool NMajor = false;
    constexpr int kPhysicalWaves = 4;
    static_assert(MAtoms == 2 || MAtoms == 4);
    constexpr int kMAtoms = MAtoms;
    constexpr int kNAtomsPerWave = 3;
    constexpr int kOutputColumns =
        kPhysicalWaves * kNAtomsPerWave * kTileN;

    struct SharedStorage
    {
        alignas(16) uint64_t codebook[kCodebookBytes / sizeof(uint64_t)];
        alignas(16) uint8_t weight_cache[kPhysicalWaves][kTripletLDSBytes];
    };
    __shared__ SharedStorage shared;

    const int lane = static_cast<int>(threadIdx.x);
    const int wave = static_cast<int>(threadIdx.y);
    const int linear_thread = wave * 64 + lane;
    const int lane_row = lane % 16;
    const int lane_group = lane / 16;
    const int num_tasks = task_count[0];
    if(num_tasks <= 0)
        return;

    const int n_tiles = (N + kOutputColumns - 1) / kOutputColumns;
    const int total_tiles = num_tasks * n_tiles;
    const int k_tiles = (K + kTileK - 1) / kTileK;
    int previous_expert = -1;

    for(int work_index = static_cast<int>(blockIdx.x); work_index < total_tiles;
        work_index += static_cast<int>(gridDim.x))
    {
        const int task_index =
            NMajor ? work_index % num_tasks : work_index / n_tiles;
        const int n_tile_index =
            NMajor ? work_index / num_tasks : work_index % n_tiles;
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
            // Protect readers of the old expert before reusing the codebook.
            if(previous_expert >= 0)
                __syncthreads();
            for(int entry = linear_thread;
                entry < kCodebookBytes / static_cast<int>(sizeof(uint64_t));
                entry += 256)
                shared.codebook[entry] =
                    reinterpret_cast<const uint64_t*>(auxiliary)[entry];
            __syncthreads();
            previous_expert = expert_index;
        }

        const int n_block_base =
            n_tile_index * (kOutputColumns / kTileN) +
            wave * kNAtomsPerWave;
        uint32_t packed_bases = 0;
#pragma unroll
        for(int atom = 0; atom < kNAtomsPerWave; ++atom)
            packed_bases |=
                static_cast<uint32_t>(
                    auxiliary[kCodebookBytes + n_block_base + atom])
                << (atom * 8);

        const int row_end = row_begin + row_count;
        for(int row_base = row_begin; row_base < row_end;
            row_base += kMAtoms * 16)
        {
            const int sub_end = min(row_base + kMAtoms * 16, row_end);
            opus::vector_t<float, 4>
                accumulators[kMAtoms][kNAtomsPerWave] = {};

            int next_data_base = static_cast<int>(triplet_base(
                physical_tile(n_block_base, 0, k_tiles)));
            iq2r_issue_direct_lds_triplet(data_buffer,
                                           shared.weight_cache[wave],
                                           lane,
                                           0,
                                           next_data_base);
            next_data_base += kGroupBytes;

            for(int k_tile = 0; k_tile < k_tiles; ++k_tile)
            {
                union ActivationFragment
                {
                    opus::i32x8_t words;
                    uint8_t bytes[32];
                } activation_fragments[kMAtoms] = {};
                uint32_t scale_a[kMAtoms];
#pragma unroll
                for(int m_atom = 0; m_atom < kMAtoms; ++m_atom)
                    scale_a[m_atom] = 127u;

#pragma unroll
                for(int m_atom = 0; m_atom < kMAtoms; ++m_atom)
                {
                    const int input_row = min(row_base + m_atom * 16 + lane_row, sub_end - 1);
                    if(row_base + m_atom * 16 < sub_end)
                    {
                        const int activation_k =
                            k_tile * kTileK + lane_group * 16;
                        const auto* activation_row =
                            reinterpret_cast<const uint8_t*>(
                                activations + static_cast<int64_t>(input_row) * K);
                        if(activation_k + 15 < K)
                            *reinterpret_cast<uint4*>(
                                activation_fragments[m_atom].bytes) =
                                *reinterpret_cast<const uint4*>(
                                    activation_row + activation_k);
                        if(activation_k + 64 + 15 < K)
                            *reinterpret_cast<uint4*>(
                                activation_fragments[m_atom].bytes + 16) =
                                *reinterpret_cast<const uint4*>(
                                    activation_row + activation_k + 64);

                        const int scale_column = k_tile * 4 + lane_group;
                        if(scale_column < K / kScaleBlock)
                        {
                            const uint32_t exponent = activation_scales[
                                static_cast<int64_t>(input_row) *
                                    (K / kScaleBlock) +
                                scale_column];
                            scale_a[m_atom] = exponent;
                        }
                    }
                }

                // Two weight requests precede eight activation-vector and four
                // activation-scale requests on full K tiles. Wait only for the
                // older weight pair, leaving that activation tail in flight
                // while the compact record moves out of LDS. The padded final
                // K tile omits the second vector load and two scale groups.
                // Partial slabs issue fewer loads than the fixed wait allowance.
                if(sub_end - row_base < kMAtoms * 16)
                    __builtin_amdgcn_s_waitcnt(0x0F70);
                else if(k_tile + 1 < k_tiles)
                    __builtin_amdgcn_s_waitcnt(0x0F7C); // vmcnt(12)
                else
                    __builtin_amdgcn_s_waitcnt(0x0F78); // vmcnt(8)
                const IQ2RCompressedTriplet compressed =
                    iq2r_read_direct_lds_triplet(
                        shared.weight_cache[wave], lane, 0);
                asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
                const bool has_next = k_tile + 1 < k_tiles;
                if(has_next)
                {
                    iq2r_issue_direct_lds_triplet(data_buffer,
                                                   shared.weight_cache[wave],
                                                   lane,
                                                   0,
                                                   next_data_base);
                    next_data_base += kGroupBytes;
                }
                opus::i32x8_t weight_fragments[kNAtomsPerWave];
                const uint32_t scale_b = iq2r_decode_direct_lds_triplet(
                    compressed,
                    shared.codebook,
                    packed_bases,
                    weight_fragments);
                if(has_next)
                    __builtin_amdgcn_s_waitcnt(0x0F72); // vmcnt(2)
                else
                    __builtin_amdgcn_s_waitcnt(0x0F70); // vmcnt(0)
#pragma unroll
                for(int m_atom = 0; m_atom < kMAtoms; ++m_atom)
                    asm volatile("" : "+v"(scale_a[m_atom]));
#pragma unroll
                for(int m_atom = 0; m_atom < kMAtoms; ++m_atom)
                    if(row_base + m_atom * 16 < sub_end)
                    iq2r_cooperative_triplet_mfma<0>(
                        activation_fragments[m_atom].words,
                        weight_fragments,
                        accumulators[m_atom],
                        scale_a[m_atom] * 0x01010101u,
                        scale_b);
            }

#pragma unroll
            for(int m_atom = 0; m_atom < kMAtoms; ++m_atom)
            {
                if(row_base + m_atom * 16 >= sub_end) continue;
#pragma unroll
                for(int atom = 0; atom < kNAtomsPerWave; ++atom)
                {
                    const int output_column =
                        (n_block_base + atom) * kTileN + lane_row;
#pragma unroll
                    for(int item = 0; item < 4; ++item)
                    {
                        const int output_row =
                            row_base + m_atom * 16 + lane_group * 4 + item;
                        if(output_row < sub_end && output_column < N)
                        {
                            float value = accumulators[m_atom][atom][item];
                            if(all_bias != nullptr)
                                value += __bfloat162float(all_bias[
                                    static_cast<int64_t>(expert_index) * N +
                                    output_column]);
                            output[static_cast<int64_t>(output_row) * N +
                                   output_column] = __float2bfloat16(value);
                        }
                    }
                }
            }
        }
    }
#endif
}



__global__ __launch_bounds__(256, 2) void iq2r_scheduled_large_register_prefetch_uniform_deferred_scale_kernel(
    const opus::fp8_t* __restrict__ activations,
    const uint8_t* __restrict__ activation_scales,
    const uint8_t* __restrict__ all_data,
    const uint8_t* __restrict__ all_auxiliary,
    const int32_t* __restrict__ tasks,
    const int32_t* __restrict__ task_count,
    const __hip_bfloat16* __restrict__ all_bias,
    __hip_bfloat16* __restrict__ output,
    int M,
    int unused_N,
    int unused_K,
    int expert_count,
    int data_bytes,
    int auxiliary_bytes)
{
#if defined(__gfx950__)
    constexpr int N = 6144;
    constexpr int K = 256;
    constexpr int MAtoms = 2;
    constexpr bool NMajor = false;
    constexpr int kPhysicalWaves = 4;
    static_assert(MAtoms == 2 || MAtoms == 4);
    constexpr int kMAtoms = MAtoms;
    constexpr int kNAtomsPerWave = 3;
    constexpr int kOutputColumns =
        kPhysicalWaves * kNAtomsPerWave * kTileN;

    struct SharedStorage
    {
        alignas(16) uint64_t codebook[kCodebookBytes / sizeof(uint64_t)];
    };
    __shared__ SharedStorage shared;

    const int lane = static_cast<int>(threadIdx.x);
    const int wave = static_cast<int>(threadIdx.y);
    const int linear_thread = wave * 64 + lane;
    const int lane_row = lane % 16;
    const int lane_group = lane / 16;
    const int num_tasks = task_count[0];
    if(num_tasks <= 0)
        return;

    const int n_tiles = (N + kOutputColumns - 1) / kOutputColumns;
    const int total_tiles = num_tasks * n_tiles;
    const int k_tiles = (K + kTileK - 1) / kTileK;
    int previous_expert = -1;

    for(int work_index = static_cast<int>(blockIdx.x); work_index < total_tiles;
        work_index += static_cast<int>(gridDim.x))
    {
        const int task_index =
            NMajor ? work_index % num_tasks : work_index / n_tiles;
        const int n_tile_index =
            NMajor ? work_index / num_tasks : work_index % n_tiles;
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
            // Protect readers of the old expert before reusing the codebook.
            if(previous_expert >= 0)
                __syncthreads();
            for(int entry = linear_thread;
                entry < kCodebookBytes / static_cast<int>(sizeof(uint64_t));
                entry += 256)
                shared.codebook[entry] =
                    reinterpret_cast<const uint64_t*>(auxiliary)[entry];
            __syncthreads();
            previous_expert = expert_index;
        }

        const int n_block_base =
            n_tile_index * (kOutputColumns / kTileN) +
            wave * kNAtomsPerWave;
        uint32_t packed_bases = 0;
#pragma unroll
        for(int atom = 0; atom < kNAtomsPerWave; ++atom)
            packed_bases |=
                static_cast<uint32_t>(
                    auxiliary[kCodebookBytes + n_block_base + atom])
                << (atom * 8);

        const int row_end = row_begin + row_count;
        for(int row_base = row_begin; row_base < row_end;
            row_base += kMAtoms * 16)
        {
            const int sub_end = min(row_base + kMAtoms * 16, row_end);
            opus::vector_t<float, 4>
                accumulators[kMAtoms][kNAtomsPerWave] = {};

            int next_data_base = static_cast<int>(triplet_base(
                physical_tile(n_block_base, 0, k_tiles)));
            IQ2RCompressedTriplet pending = iq2r_scheduled_load_compact_uniform(data, data_bytes, next_data_base, lane);
            next_data_base += kGroupBytes;

#pragma unroll 1
            for(int k_tile = 0; k_tile < k_tiles; ++k_tile)
            {
                union ActivationFragment
                {
                    opus::i32x8_t words;
                    uint8_t bytes[32];
                } activation_fragments[kMAtoms] = {};
                uint32_t scale_a[kMAtoms];
#pragma unroll
                for(int m_atom = 0; m_atom < kMAtoms; ++m_atom)
                    scale_a[m_atom] = 127u;

#pragma unroll
                for(int m_atom = 0; m_atom < kMAtoms; ++m_atom)
                {
                    const int input_row = min(row_base + m_atom * 16 + lane_row, sub_end - 1);
                    if(row_base + m_atom * 16 < sub_end)
                    {
                        const int activation_k =
                            k_tile * kTileK + lane_group * 16;
                        const auto* activation_row =
                            reinterpret_cast<const uint8_t*>(
                                activations + static_cast<int64_t>(input_row) * K);
                        if(activation_k + 15 < K)
                            *reinterpret_cast<uint4*>(
                                activation_fragments[m_atom].bytes) =
                                *reinterpret_cast<const uint4*>(
                                    activation_row + activation_k);
                        if(activation_k + 64 + 15 < K)
                            *reinterpret_cast<uint4*>(
                                activation_fragments[m_atom].bytes + 16) =
                                *reinterpret_cast<const uint4*>(
                                    activation_row + activation_k + 64);

                        const int scale_column = k_tile * 4 + lane_group;
                        if(scale_column < K / kScaleBlock)
                        {
                            const uint32_t exponent = activation_scales[
                                static_cast<int64_t>(input_row) *
                                    (K / kScaleBlock) +
                                scale_column];
                            scale_a[m_atom] = exponent;
                        }
                    }
                }

                const IQ2RCompressedTriplet compressed = pending;
                if(k_tile + 1 < k_tiles)
                {
                    pending = iq2r_scheduled_load_compact_uniform(data, data_bytes, next_data_base, lane);
                    next_data_base += kGroupBytes;
                }
                opus::i32x8_t weight_fragments[kNAtomsPerWave];
                const uint32_t scale_b = iq2r_decode_direct_lds_triplet(
                    compressed,
                    shared.codebook,
                    packed_bases,
                    weight_fragments);
#pragma unroll
                for(int m_atom = 0; m_atom < kMAtoms; ++m_atom)
                    asm volatile("" : "+v"(scale_a[m_atom]));
#pragma unroll
                for(int m_atom = 0; m_atom < kMAtoms; ++m_atom)
                    if(row_base + m_atom * 16 < sub_end)
                    iq2r_cooperative_triplet_mfma<0>(
                        activation_fragments[m_atom].words,
                        weight_fragments,
                        accumulators[m_atom],
                        scale_a[m_atom] * 0x01010101u,
                        scale_b);
            }

#pragma unroll
            for(int m_atom = 0; m_atom < kMAtoms; ++m_atom)
            {
                if(row_base + m_atom * 16 >= sub_end) continue;
#pragma unroll
                for(int atom = 0; atom < kNAtomsPerWave; ++atom)
                {
                    const int output_column =
                        (n_block_base + atom) * kTileN + lane_row;
#pragma unroll
                    for(int item = 0; item < 4; ++item)
                    {
                        const int output_row =
                            row_base + m_atom * 16 + lane_group * 4 + item;
                        if(output_row < sub_end && output_column < N)
                        {
                            float value = accumulators[m_atom][atom][item];
                            if(all_bias != nullptr)
                                value += __bfloat162float(all_bias[
                                    static_cast<int64_t>(expert_index) * N +
                                    output_column]);
                            output[static_cast<int64_t>(output_row) * N +
                                   output_column] = __float2bfloat16(value);
                        }
                    }
                }
            }
        }
    }
#endif
}


__global__ __launch_bounds__(256,2) void iq2r_scheduled_small_single_epilogue_kernel(
    const opus::fp8_t* __restrict__ activations,
    const uint8_t* __restrict__ activation_scales,
    const uint8_t* __restrict__ all_data,
    const uint8_t* __restrict__ all_auxiliary,
    const int32_t* __restrict__ tasks,
    const int32_t* __restrict__ task_count,
    const __hip_bfloat16* __restrict__ all_bias,
    __hip_bfloat16* __restrict__ output,
    opus::fp8_t* __restrict__ fused_output,
    uint8_t* __restrict__ fused_output_scales,
    int M,
    int unused_N,
    int unused_K,
    int expert_count,
    int data_bytes,
    int auxiliary_bytes,
    float swiglu_limit,
    float swiglu_alpha,
    float swiglu_up_offset)
{
#if defined(__gfx950__)
    constexpr int N = 6144, K = 256;
    constexpr int OutputAtoms = 6, PhysicalWaves = 4, LoadAux = 2;
    constexpr bool ActivationLookahead = false, StagedScales = false;
    constexpr bool TiledActivationScales = false, VectorCodebookCopy = true, FusedSwiGLU = false;
    static_assert(OutputAtoms == 3 || OutputAtoms == 4 || OutputAtoms == 6);
    static_assert(PhysicalWaves == 2 || PhysicalWaves == 4 || PhysicalWaves == 8);
    static_assert(!ActivationLookahead || OutputAtoms == 6);
    static_assert(!(StagedScales && TiledActivationScales));
    static_assert(!FusedSwiGLU ||
                  (OutputAtoms == 4 && PhysicalWaves == 4 &&
                   !ActivationLookahead && !StagedScales));
    constexpr int kOutputColumns = OutputAtoms * kTileN;
    constexpr int kDirectTriplets =
        (OutputAtoms + kAtomsPerTriplet - 1) / kAtomsPerTriplet;
    constexpr int kStoredAtoms = OutputAtoms < 3 ? OutputAtoms : 3;
    constexpr int kScaleGroups = 2880 / kScaleBlock;
    constexpr int kPaddedScaleGroups = ((kScaleGroups + 3) / 4) * 4;
    constexpr int kStagedScaleBytes = 16 * kPaddedScaleGroups;

    struct SharedStorage
    {
        alignas(16) uint64_t codebook[kCodebookBytes / sizeof(uint64_t)];
        alignas(16) uint8_t
            weight_cache[2][kDirectTriplets * kTripletLDSBytes];
        union
        {
            alignas(16) opus::vector_t<float, 4>
                partial[2][6][64];
            alignas(16) uint8_t scale_cache[kStagedScaleBytes];
        } reusable;
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
        if(row_begin < 0 || row_count <= 0 || row_begin + row_count > M)
            continue;
        if(expert_index < 0 || expert_index >= expert_count)
        {
            // Expert-parallel callers retain non-local routes in the sorted
            // permutation so the final indexed reduction can preserve the
            // original top-k order. Materialize their contribution exactly as
            // zero, matching the generic task kernel instead of leaving stale
            // workspace values behind.
            const int output_column_begin =
                n_tile_index * (FusedSwiGLU ? kOutputColumns / 2
                                            : kOutputColumns);
            const int output_width = FusedSwiGLU ? N / 2 : N;
            const int output_columns = min(FusedSwiGLU ? kOutputColumns / 2
                                                       : kOutputColumns,
                                           output_width - output_column_begin);
            const int task_elements = row_count * output_columns;
            for(int element = linear_thread; element < task_elements;
                element += linear_threads)
            {
                const int row = row_begin + element / output_columns;
                const int column = output_column_begin + element % output_columns;
                if constexpr(FusedSwiGLU)
                    fused_output[static_cast<int64_t>(row) * output_width + column] =
                        opus::fp32_to_fp8(0.0f);
                else
                    output[static_cast<int64_t>(row) * output_width + column] =
                        __float2bfloat16(0.0f);
            }
            if constexpr(FusedSwiGLU)
            {
                if(linear_thread < row_count)
                    fused_output_scales[
                        static_cast<int64_t>(row_begin + linear_thread) *
                            (N / 64) +
                        n_tile_index] = 127;
            }
            continue;
        }

        const uint8_t* data =
            all_data + static_cast<int64_t>(expert_index) * data_bytes;
        const uint8_t* auxiliary =
            all_auxiliary + static_cast<int64_t>(expert_index) * auxiliary_bytes;
        opus::gmem<uint8_t> data_buffer(data, static_cast<unsigned int>(data_bytes));
        if(expert_index != previous_expert)
        {
            if constexpr(VectorCodebookCopy && PhysicalWaves == 4)
            {
                reinterpret_cast<uint4*>(shared.codebook)[linear_thread] =
                    reinterpret_cast<const uint4*>(auxiliary)[linear_thread];
            }
            else
            {
                for(int entry = linear_thread;
                    entry < kCodebookBytes / static_cast<int>(sizeof(uint64_t));
                    entry += linear_threads)
                    shared.codebook[entry] =
                        reinterpret_cast<const uint64_t*>(auxiliary)[entry];
            }
            __syncthreads();
            previous_expert = expert_index;
        }

        const int n_block_base = n_tile_index * OutputAtoms;
        // Four-atom tiles advance by 64 columns and are not generally aligned
        // to the three-atom (48-column) physical IQ2R packing groups. Load the
        // two enclosing triplets and decode the requested four consecutive
        // atoms, matching the established fused-SwiGLU path.
        const int loaded_n_block_base =
            OutputAtoms == 4 ? n_block_base - n_block_base % kAtomsPerTriplet
                             : n_block_base;
        const int fused_atom_offset = n_block_base - loaded_n_block_base;
        uint32_t packed_bases[kDirectTriplets] = {};
#pragma unroll
        for(int triplet = 0; triplet < kDirectTriplets; ++triplet)
        {
#pragma unroll
            for(int atom = 0; atom < kAtomsPerTriplet; ++atom)
                packed_bases[triplet] |=
                    static_cast<uint32_t>(
                        auxiliary[kCodebookBytes + loaded_n_block_base +
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

            if constexpr(StagedScales)
            {
                const int valid_rows = sub_end - row_base;
                for(int index = linear_thread; index < kStagedScaleBytes;
                    index += linear_threads)
                {
                    const int row = index / kPaddedScaleGroups;
                    const int group = index % kPaddedScaleGroups;
                    uint8_t exponent = 127;
                    if(group < kScaleGroups)
                    {
                        const int source_row = row_base + min(row, valid_rows - 1);
                        exponent = activation_scales[
                            iq2r_activation_scale_offset<TiledActivationScales>(
                                source_row, group, M, kScaleGroups)];
                    }
                    const int scale_k_tile = group / 4;
                    const int scale_group = group % 4;
                    shared.reusable.scale_cache[scale_k_tile * 64 +
                                                scale_group * 16 + row] = exponent;
                }
                __syncthreads();
            }
            opus::vector_t<float, 4> accumulators[OutputAtoms] = {};

            int next_data_base = static_cast<int>(triplet_base(
                physical_tile(loaded_n_block_base, k_begin, k_tiles)));
            int next_second_data_base = static_cast<int>(triplet_base(
                physical_tile(loaded_n_block_base + kAtomsPerTriplet,
                              k_begin,
                              k_tiles)));

            if constexpr(ActivationLookahead)
            {
                // Keep the next activation fragment in VGPRs while the second
                // weight triplet for the current K tile is decoded and
                // consumed.  Every steady-state iteration has this request
                // order: next activation, next triplet 0, next triplet 1.
                // Counted waits retire only the current triplet and leave that
                // younger lookahead in flight.
                const int uniform_iterations =
                    __builtin_amdgcn_readfirstlane(iterations);
                opus::gmem<uint8_t> activation_buffer(
                    activations, static_cast<unsigned int>(M * K));
                opus::gmem<uint8_t> scale_buffer(
                    activation_scales,
                    static_cast<unsigned int>(
                        TiledActivationScales
                            ? ((K / kScaleBlock + 3) / 4) * ((M + 15) / 16) *
                                  64
                            : M * (K / kScaleBlock)));
                IQ2RActivationFragment current_activation{};
                uint32_t current_scale_a = 127u * 0x01010101u;
                iq2r_issue_buffered_activation_fragment<StagedScales,
                                                         TiledActivationScales>(
                    activation_buffer,
                    scale_buffer,
                    shared.reusable.scale_cache,
                    load_row,
                    row_base,
                    k_begin,
                    lane_group,
                    M,
                    K,
                    current_activation,
                    current_scale_a);
                iq2r_issue_direct_lds_triplet<LoadAux>(data_buffer,
                                               shared.weight_cache[wave],
                                               lane,
                                               0,
                                               next_data_base);
                iq2r_issue_direct_lds_triplet<LoadAux>(data_buffer,
                                               shared.weight_cache[wave],
                                               lane,
                                               1,
                                               next_data_base + kTripletBytes);
                next_data_base += kGroupBytes;

                for(int iteration = 0; iteration < uniform_iterations; ++iteration)
                {
                    // Activation requests precede both current weight
                    // triplets. Retire the activation and triplet 0 while the
                    // two triplet-1 requests remain outstanding.
                    iq2r_wait_vmcnt<2>();
                    IQ2RCompressedTriplet compressed =
                        iq2r_read_direct_lds_triplet(
                            shared.weight_cache[wave], lane, 0);
                    asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
                    const opus::i32x8_t current_words = current_activation.words;

                    const bool has_next = iteration + 1 < uniform_iterations;
                    uint32_t next_scale_a = 127u * 0x01010101u;
                    if(has_next)
                    {
                        iq2r_issue_buffered_activation_fragment<
                            StagedScales,
                            TiledActivationScales>(
                            activation_buffer,
                            scale_buffer,
                            shared.reusable.scale_cache,
                            load_row,
                            row_base,
                            k_begin + iteration + 1,
                            lane_group,
                            M,
                            K,
                            current_activation,
                            next_scale_a);
                        iq2r_issue_direct_lds_triplet<LoadAux>(data_buffer,
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
                    iq2r_cooperative_triplet_mfma<0>(
                        current_words,
                        weight_fragments,
                        accumulators,
                        current_scale_a,
                        scale_b);

                    if(has_next)
                    {
                        // Two activation requests and two triplet-0 requests
                        // remain younger when scales are staged in LDS. The
                        // row-major variant has one additional scale request.
                        if constexpr(StagedScales)
                            iq2r_wait_vmcnt<4>();
                        else
                            iq2r_wait_vmcnt<5>();
                    }
                    else
                        iq2r_wait_vmcnt<0>();

                    compressed = iq2r_read_direct_lds_triplet(
                        shared.weight_cache[wave], lane, 1);
                    asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
                    if(has_next)
                    {
                        iq2r_issue_direct_lds_triplet<LoadAux>(data_buffer,
                                                       shared.weight_cache[wave],
                                                       lane,
                                                       1,
                                                       next_data_base +
                                                           kTripletBytes);
                        next_data_base += kGroupBytes;
                    }
                    scale_b = iq2r_decode_direct_lds_triplet(compressed,
                                                             shared.codebook,
                                                             packed_bases[1],
                                                             weight_fragments);
                    iq2r_cooperative_triplet_mfma<3>(
                        current_words,
                        weight_fragments,
                        accumulators,
                        current_scale_a,
                        scale_b);
                    if(has_next)
                        current_scale_a = next_scale_a;
                }
            }
            else
            {
                if(iterations > 0)
                {
                if constexpr(OutputAtoms == 4)
                {
                    iq2r_issue_direct_lds_triplet<LoadAux>(
                        data_buffer,
                        shared.weight_cache[wave],
                        lane,
                        0,
                        next_data_base);
                    iq2r_issue_direct_lds_triplet<LoadAux>(
                        data_buffer,
                        shared.weight_cache[wave],
                        lane,
                        1,
                        next_second_data_base);
                }
                else
                {
#pragma unroll
                    for(int triplet = 0; triplet < kDirectTriplets; ++triplet)
                        iq2r_issue_direct_lds_triplet<LoadAux>(
                            data_buffer,
                            shared.weight_cache[wave],
                            lane,
                            triplet,
                            next_data_base + triplet * kTripletBytes);
                }
                }
                next_data_base += kGroupBytes;
                next_second_data_base += kGroupBytes;
                int tiled_scale_offset =
                    (k_begin * ((M + 15) / 16) + load_row / 16) * 64 +
                    lane_group * 16 + load_row % 16;
                const int tiled_scale_stride = ((M + 15) / 16) * 64;

                for(int iteration = 0; iteration < iterations; ++iteration)
                {
                    const int k_tile = k_begin + iteration;
                    IQ2RActivationFragment activation_fragment{};
                    const int activation_k = k_tile * kTileK + lane_group * 16;
                    if(activation_k + 15 < K)
                    {
                        const auto* activation_row =
                            reinterpret_cast<const uint8_t*>(
                                activations + static_cast<int64_t>(load_row) * K);
                        *reinterpret_cast<uint4*>(activation_fragment.bytes) =
                            *reinterpret_cast<const uint4*>(activation_row +
                                                           activation_k);
                        if(activation_k + 64 + 15 < K)
                            *reinterpret_cast<uint4*>(activation_fragment.bytes +
                                                     16) =
                                *reinterpret_cast<const uint4*>(activation_row +
                                                               activation_k + 64);
                    }

                    uint32_t scale_a = 127u * 0x01010101u;
                    if constexpr(StagedScales)
                    {
                        const uint32_t exponent = shared.reusable.scale_cache[
                            k_tile * 64 + lane_group * 16 + load_row - row_base];
                        scale_a = exponent * 0x01010101u;
                    }
                    else
                    {
                        const int scale_column = k_tile * 4 + lane_group;
                        if(scale_column < K / kScaleBlock)
                        {
                            uint32_t exponent;
                            if constexpr(TiledActivationScales)
                            {
                                exponent = activation_scales[tiled_scale_offset];
                                tiled_scale_offset += tiled_scale_stride;
                            }
                            else
                                exponent = activation_scales[
                                    static_cast<int64_t>(load_row) *
                                            (K / kScaleBlock) +
                                        scale_column];
                            scale_a = exponent * 0x01010101u;
                        }
                    }

                    // The direct-to-LDS requests precede two or three activation
                    // requests. Retire only the older weight requests first so
                    // their LDS reads and IQ2R decode overlap the activation tail.
                    if(k_tile + 1 < k_tiles)
                    {
                        if constexpr(StagedScales)
                            iq2r_wait_vmcnt<2>();
                        else
                            iq2r_wait_vmcnt<3>();
                    }
                    else
                    {
                        if constexpr(StagedScales)
                            iq2r_wait_vmcnt<1>();
                        else
                            iq2r_wait_vmcnt<2>();
                    }

                    IQ2RCompressedTriplet compressed =
                        iq2r_read_direct_lds_triplet(
                            shared.weight_cache[wave], lane, 0);
                    asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
                    iq2r_wait_vmcnt<0>();
                    if(iteration + 1 < iterations)
                    {
                        if constexpr(OutputAtoms == 4)
                            iq2r_issue_direct_lds_triplet<LoadAux>(
                                data_buffer,
                                shared.weight_cache[wave],
                                lane,
                                0,
                                next_data_base);
                        else
                            iq2r_issue_direct_lds_triplet<LoadAux>(
                                data_buffer,
                                shared.weight_cache[wave],
                                lane,
                                0,
                                next_data_base);
                    }
                    if constexpr(OutputAtoms == 4)
                    {
                        opus::i32x8_t weight_fragment;
                        uint32_t scale_b;
                        if(fused_atom_offset == 0)
                        {
                            iq2r_decode_direct_lds_atom<0>(compressed,
                                                           shared.codebook,
                                                           packed_bases[0],
                                                           weight_fragment,
                                                           scale_b);
                            iq2r_cooperative_single_mfma<0>(
                                activation_fragment.words, weight_fragment,
                                accumulators, scale_a, scale_b);
                            iq2r_decode_direct_lds_atom<1>(compressed,
                                                           shared.codebook,
                                                           packed_bases[0],
                                                           weight_fragment,
                                                           scale_b);
                            iq2r_cooperative_single_mfma<1>(
                                activation_fragment.words, weight_fragment,
                                accumulators, scale_a, scale_b);
                            iq2r_decode_direct_lds_atom<2>(compressed,
                                                           shared.codebook,
                                                           packed_bases[0],
                                                           weight_fragment,
                                                           scale_b);
                            iq2r_cooperative_single_mfma<2>(
                                activation_fragment.words, weight_fragment,
                                accumulators, scale_a, scale_b);
                        }
                        else if(fused_atom_offset == 1)
                        {
                            iq2r_decode_direct_lds_atom<1>(compressed,
                                                           shared.codebook,
                                                           packed_bases[0],
                                                           weight_fragment,
                                                           scale_b);
                            iq2r_cooperative_single_mfma<0>(
                                activation_fragment.words, weight_fragment,
                                accumulators, scale_a, scale_b);
                            iq2r_decode_direct_lds_atom<2>(compressed,
                                                           shared.codebook,
                                                           packed_bases[0],
                                                           weight_fragment,
                                                           scale_b);
                            iq2r_cooperative_single_mfma<1>(
                                activation_fragment.words, weight_fragment,
                                accumulators, scale_a, scale_b);
                        }
                        else
                        {
                            iq2r_decode_direct_lds_atom<2>(compressed,
                                                           shared.codebook,
                                                           packed_bases[0],
                                                           weight_fragment,
                                                           scale_b);
                            iq2r_cooperative_single_mfma<0>(
                                activation_fragment.words, weight_fragment,
                                accumulators, scale_a, scale_b);
                        }
                    }
                    else
                    {
                        opus::i32x8_t weight_fragments[kAtomsPerTriplet];
                        const uint32_t scale_b = iq2r_decode_direct_lds_triplet(
                            compressed,
                            shared.codebook,
                            packed_bases[0],
                            weight_fragments);
                        iq2r_cooperative_triplet_mfma<0>(
                            activation_fragment.words,
                            weight_fragments,
                            accumulators,
                            scale_a,
                            scale_b);
                    }
                    if constexpr(OutputAtoms > 3)
                    {
                        compressed = iq2r_read_direct_lds_triplet(
                            shared.weight_cache[wave], lane, 1);
                        asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
                        if(iteration + 1 < iterations)
                        {
                            if constexpr(OutputAtoms == 4)
                                iq2r_issue_direct_lds_triplet<LoadAux>(
                                    data_buffer,
                                    shared.weight_cache[wave],
                                    lane,
                                    1,
                                    next_second_data_base);
                            else
                                iq2r_issue_direct_lds_triplet<LoadAux>(
                                    data_buffer,
                                    shared.weight_cache[wave],
                                    lane,
                                    1,
                                    next_data_base + kTripletBytes);
                            next_data_base += kGroupBytes;
                            next_second_data_base += kGroupBytes;
                        }
                        if constexpr(OutputAtoms == 4)
                        {
                            opus::i32x8_t weight_fragment;
                            uint32_t scale_b;
                            if(fused_atom_offset == 0)
                            {
                                iq2r_decode_direct_lds_atom<0>(compressed,
                                                               shared.codebook,
                                                               packed_bases[1],
                                                               weight_fragment,
                                                               scale_b);
                                iq2r_cooperative_single_mfma<3>(
                                    activation_fragment.words, weight_fragment,
                                    accumulators, scale_a, scale_b);
                            }
                            else if(fused_atom_offset == 1)
                            {
                                iq2r_decode_direct_lds_atom<0>(compressed,
                                                               shared.codebook,
                                                               packed_bases[1],
                                                               weight_fragment,
                                                               scale_b);
                                iq2r_cooperative_single_mfma<2>(
                                    activation_fragment.words, weight_fragment,
                                    accumulators, scale_a, scale_b);
                                iq2r_decode_direct_lds_atom<1>(compressed,
                                                               shared.codebook,
                                                               packed_bases[1],
                                                               weight_fragment,
                                                               scale_b);
                                iq2r_cooperative_single_mfma<3>(
                                    activation_fragment.words, weight_fragment,
                                    accumulators, scale_a, scale_b);
                            }
                            else
                            {
                                iq2r_decode_direct_lds_atom<0>(compressed,
                                                               shared.codebook,
                                                               packed_bases[1],
                                                               weight_fragment,
                                                               scale_b);
                                iq2r_cooperative_single_mfma<1>(
                                    activation_fragment.words, weight_fragment,
                                    accumulators, scale_a, scale_b);
                                iq2r_decode_direct_lds_atom<1>(compressed,
                                                               shared.codebook,
                                                               packed_bases[1],
                                                               weight_fragment,
                                                               scale_b);
                                iq2r_cooperative_single_mfma<2>(
                                    activation_fragment.words, weight_fragment,
                                    accumulators, scale_a, scale_b);
                                iq2r_decode_direct_lds_atom<2>(compressed,
                                                               shared.codebook,
                                                               packed_bases[1],
                                                               weight_fragment,
                                                               scale_b);
                                iq2r_cooperative_single_mfma<3>(
                                    activation_fragment.words, weight_fragment,
                                    accumulators, scale_a, scale_b);
                            }
                        }
                        else
                        {
                            opus::i32x8_t weight_fragments[kAtomsPerTriplet];
                            const uint32_t scale_b = iq2r_decode_direct_lds_triplet(
                                compressed,
                                shared.codebook,
                                packed_bases[1],
                                weight_fragments);
                            iq2r_cooperative_triplet_mfma<3>(
                                activation_fragment.words,
                                weight_fragments,
                                accumulators,
                                scale_a,
                                scale_b);
                        }
                    }
                    else if(iteration + 1 < iterations)
                        next_data_base += kGroupBytes;
                }
            }

            if constexpr(StagedScales)
                __syncthreads();

            const int output_row = row_base + (lane / 16) * 4 + wave;
            if constexpr(FusedSwiGLU)
            {
                float raw_values[4] = {};
#pragma unroll
                for(int atom = 0; atom < 3; ++atom)
                    shared.reusable.partial[wave][atom][lane] = accumulators[atom];
                __syncthreads();
                if(wave < 4 && output_row < sub_end)
                {
#pragma unroll
                    for(int atom = 0; atom < 3; ++atom)
                    {
                        const float* p0 = reinterpret_cast<const float*>(
                            &shared.reusable.partial[0][atom][lane]);
                        const float* p1 = reinterpret_cast<const float*>(
                            &shared.reusable.partial[1][atom][lane]);
                        const float* p2 = reinterpret_cast<const float*>(
                            &shared.reusable.partial[2][atom][lane]);
                        const float* p3 = reinterpret_cast<const float*>(
                            &shared.reusable.partial[3][atom][lane]);
                        float value = p0[wave] + p1[wave] + p2[wave] + p3[wave];
                        if(all_bias != nullptr)
                            value += __bfloat162float(
                                all_bias[static_cast<int64_t>(expert_index) * N +
                                         n_tile_index * kOutputColumns + atom * 16 +
                                         lane % 16]);
                        raw_values[atom] = __bfloat162float(__float2bfloat16(value));
                    }
                }
                __syncthreads();

                shared.reusable.partial[wave][0][lane] = accumulators[3];
                __syncthreads();
                if(wave < 4 && output_row < sub_end)
                {
                    const float* p0 = reinterpret_cast<const float*>(
                        &shared.reusable.partial[0][0][lane]);
                    const float* p1 = reinterpret_cast<const float*>(
                        &shared.reusable.partial[1][0][lane]);
                    const float* p2 = reinterpret_cast<const float*>(
                        &shared.reusable.partial[2][0][lane]);
                    const float* p3 = reinterpret_cast<const float*>(
                        &shared.reusable.partial[3][0][lane]);
                    float value = p0[wave] + p1[wave] + p2[wave] + p3[wave];
                    if(all_bias != nullptr)
                        value += __bfloat162float(
                            all_bias[static_cast<int64_t>(expert_index) * N +
                                     n_tile_index * kOutputColumns + 3 * 16 +
                                     lane % 16]);
                    raw_values[3] = __bfloat162float(__float2bfloat16(value));
                }
                __syncthreads();

                if(wave < 4 && output_row < sub_end)
                {
                    const bool even_column = (lane % 2) == 0;
                    opus::vector_t<opus::bf16_t, 4> activated;
                    float abs_max = 1.0e-10f;
#pragma unroll
                    for(int atom = 0; atom < 4; ++atom)
                    {
                        const float paired = __shfl_xor(raw_values[atom], 1);
                        if(even_column)
                        {
                            const float gate = raw_values[atom];
                            const float up = paired;
                            const bool clamp = swiglu_limit > 0.0f;
                            const float clipped_gate =
                                clamp && gate > swiglu_limit ? swiglu_limit : gate;
                            const float clipped_up =
                                clamp && up < -swiglu_limit
                                    ? -swiglu_limit
                                    : (clamp && up > swiglu_limit
                                           ? swiglu_limit
                                           : up);
                            const float swish =
                                clipped_gate /
                                (1.0f + __expf(-swiglu_alpha * clipped_gate));
                            const __hip_bfloat16 rounded = __float2bfloat16(
                                swish * (clipped_up + swiglu_up_offset));
                            activated[atom] =
                                __builtin_bit_cast(opus::bf16_t, rounded);
                            abs_max =
                                fmaxf(abs_max, fabsf(__bfloat162float(rounded)));
                        }
                    }
                    abs_max = fmaxf(abs_max, __shfl_xor(abs_max, 2));
                    abs_max = fmaxf(abs_max, __shfl_xor(abs_max, 4));
                    abs_max = fmaxf(abs_max, __shfl_xor(abs_max, 8));
                    if(even_column)
                    {
                        const auto block_scale = fp_f32_to_e8m0_block_scale<
                            kDefaultMxScaleRoundMode,
                            MxDtype::FP8_E4M3>(abs_max);
                        const float inverse_scale = 1.0f / block_scale.dq_scale;
                        const int output_width = N / 2;
                        const int output_column_base =
                            n_tile_index * (kOutputColumns / 2) + (lane % 16) / 2;
#pragma unroll
                        for(int atom = 0; atom < 4; ++atom)
                            fused_output[static_cast<int64_t>(output_row) *
                                             output_width +
                                         output_column_base + atom * 8] =
                                opus::fp32_to_fp8(
                                    static_cast<float>(activated[atom]) *
                                    inverse_scale);
                        if(lane % 16 == 0)
                            fused_output_scales[
                                static_cast<int64_t>(output_row) * (N / 64) +
                                n_tile_index] = block_scale.byte;
                    }
                }
                __syncthreads();
            }
            else
            {
                if(wave < 2)
                {
#pragma unroll
                    for(int atom = 0; atom < 6; ++atom)
                        shared.reusable.partial[wave][atom][lane] = accumulators[atom];
                }
                __syncthreads();
                if(output_row < sub_end)
                {
#pragma unroll
                    for(int atom = 0; atom < 6; ++atom)
                    {
                        const int column = n_tile_index*96 + atom*16 + lane%16;
                        const float* p0 = reinterpret_cast<const float*>(&shared.reusable.partial[0][atom][lane]);
                        const float* p1 = reinterpret_cast<const float*>(&shared.reusable.partial[1][atom][lane]);
                        float value = ((p0[wave] + p1[wave]) + 0.0f) + 0.0f;
                        if(all_bias != nullptr)
                            value += __bfloat162float(all_bias[static_cast<int64_t>(expert_index)*N + column]);
                        output[static_cast<int64_t>(output_row)*N + column] = __float2bfloat16(value);
                    }
                }
                __syncthreads();
            }

        }
    }
#endif
}



__global__ __launch_bounds__(256,2) void iq2r_scheduled_small_single_epilogue_deferred_kernel(
    const opus::fp8_t* __restrict__ activations,
    const uint8_t* __restrict__ activation_scales,
    const uint8_t* __restrict__ all_data,
    const uint8_t* __restrict__ all_auxiliary,
    const int32_t* __restrict__ tasks,
    const int32_t* __restrict__ task_count,
    const __hip_bfloat16* __restrict__ all_bias,
    __hip_bfloat16* __restrict__ output,
    opus::fp8_t* __restrict__ fused_output,
    uint8_t* __restrict__ fused_output_scales,
    int M,
    int unused_N,
    int unused_K,
    int expert_count,
    int data_bytes,
    int auxiliary_bytes,
    float swiglu_limit,
    float swiglu_alpha,
    float swiglu_up_offset)
{
#if defined(__gfx950__)
    constexpr int N = 6144, K = 256;
    constexpr int OutputAtoms = 6, PhysicalWaves = 4, LoadAux = 2;
    constexpr bool ActivationLookahead = false, StagedScales = false;
    constexpr bool TiledActivationScales = false, VectorCodebookCopy = true, FusedSwiGLU = false;
    static_assert(OutputAtoms == 3 || OutputAtoms == 4 || OutputAtoms == 6);
    static_assert(PhysicalWaves == 2 || PhysicalWaves == 4 || PhysicalWaves == 8);
    static_assert(!ActivationLookahead || OutputAtoms == 6);
    static_assert(!(StagedScales && TiledActivationScales));
    static_assert(!FusedSwiGLU ||
                  (OutputAtoms == 4 && PhysicalWaves == 4 &&
                   !ActivationLookahead && !StagedScales));
    constexpr int kOutputColumns = OutputAtoms * kTileN;
    constexpr int kDirectTriplets =
        (OutputAtoms + kAtomsPerTriplet - 1) / kAtomsPerTriplet;
    constexpr int kStoredAtoms = OutputAtoms < 3 ? OutputAtoms : 3;
    constexpr int kScaleGroups = 2880 / kScaleBlock;
    constexpr int kPaddedScaleGroups = ((kScaleGroups + 3) / 4) * 4;
    constexpr int kStagedScaleBytes = 16 * kPaddedScaleGroups;

    struct SharedStorage
    {
        alignas(16) uint64_t codebook[kCodebookBytes / sizeof(uint64_t)];
        alignas(16) uint8_t
            weight_cache[2][kDirectTriplets * kTripletLDSBytes];
        union
        {
            alignas(16) opus::vector_t<float, 4>
                partial[2][6][64];
            alignas(16) uint8_t scale_cache[kStagedScaleBytes];
        } reusable;
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
        if(row_begin < 0 || row_count <= 0 || row_begin + row_count > M)
            continue;
        if(expert_index < 0 || expert_index >= expert_count)
        {
            // Expert-parallel callers retain non-local routes in the sorted
            // permutation so the final indexed reduction can preserve the
            // original top-k order. Materialize their contribution exactly as
            // zero, matching the generic task kernel instead of leaving stale
            // workspace values behind.
            const int output_column_begin =
                n_tile_index * (FusedSwiGLU ? kOutputColumns / 2
                                            : kOutputColumns);
            const int output_width = FusedSwiGLU ? N / 2 : N;
            const int output_columns = min(FusedSwiGLU ? kOutputColumns / 2
                                                       : kOutputColumns,
                                           output_width - output_column_begin);
            const int task_elements = row_count * output_columns;
            for(int element = linear_thread; element < task_elements;
                element += linear_threads)
            {
                const int row = row_begin + element / output_columns;
                const int column = output_column_begin + element % output_columns;
                if constexpr(FusedSwiGLU)
                    fused_output[static_cast<int64_t>(row) * output_width + column] =
                        opus::fp32_to_fp8(0.0f);
                else
                    output[static_cast<int64_t>(row) * output_width + column] =
                        __float2bfloat16(0.0f);
            }
            if constexpr(FusedSwiGLU)
            {
                if(linear_thread < row_count)
                    fused_output_scales[
                        static_cast<int64_t>(row_begin + linear_thread) *
                            (N / 64) +
                        n_tile_index] = 127;
            }
            continue;
        }

        const uint8_t* data =
            all_data + static_cast<int64_t>(expert_index) * data_bytes;
        const uint8_t* auxiliary =
            all_auxiliary + static_cast<int64_t>(expert_index) * auxiliary_bytes;
        opus::gmem<uint8_t> data_buffer(data, static_cast<unsigned int>(data_bytes));
        if(expert_index != previous_expert)
        {
            if constexpr(VectorCodebookCopy && PhysicalWaves == 4)
            {
                reinterpret_cast<uint4*>(shared.codebook)[linear_thread] =
                    reinterpret_cast<const uint4*>(auxiliary)[linear_thread];
            }
            else
            {
                for(int entry = linear_thread;
                    entry < kCodebookBytes / static_cast<int>(sizeof(uint64_t));
                    entry += linear_threads)
                    shared.codebook[entry] =
                        reinterpret_cast<const uint64_t*>(auxiliary)[entry];
            }
            __syncthreads();
            previous_expert = expert_index;
        }

        const int n_block_base = n_tile_index * OutputAtoms;
        // Four-atom tiles advance by 64 columns and are not generally aligned
        // to the three-atom (48-column) physical IQ2R packing groups. Load the
        // two enclosing triplets and decode the requested four consecutive
        // atoms, matching the established fused-SwiGLU path.
        const int loaded_n_block_base =
            OutputAtoms == 4 ? n_block_base - n_block_base % kAtomsPerTriplet
                             : n_block_base;
        const int fused_atom_offset = n_block_base - loaded_n_block_base;
        uint32_t packed_bases[kDirectTriplets] = {};
#pragma unroll
        for(int triplet = 0; triplet < kDirectTriplets; ++triplet)
        {
#pragma unroll
            for(int atom = 0; atom < kAtomsPerTriplet; ++atom)
                packed_bases[triplet] |=
                    static_cast<uint32_t>(
                        auxiliary[kCodebookBytes + loaded_n_block_base +
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

            if constexpr(StagedScales)
            {
                const int valid_rows = sub_end - row_base;
                for(int index = linear_thread; index < kStagedScaleBytes;
                    index += linear_threads)
                {
                    const int row = index / kPaddedScaleGroups;
                    const int group = index % kPaddedScaleGroups;
                    uint8_t exponent = 127;
                    if(group < kScaleGroups)
                    {
                        const int source_row = row_base + min(row, valid_rows - 1);
                        exponent = activation_scales[
                            iq2r_activation_scale_offset<TiledActivationScales>(
                                source_row, group, M, kScaleGroups)];
                    }
                    const int scale_k_tile = group / 4;
                    const int scale_group = group % 4;
                    shared.reusable.scale_cache[scale_k_tile * 64 +
                                                scale_group * 16 + row] = exponent;
                }
                __syncthreads();
            }
            opus::vector_t<float, 4> accumulators[OutputAtoms] = {};

            int next_data_base = static_cast<int>(triplet_base(
                physical_tile(loaded_n_block_base, k_begin, k_tiles)));
            int next_second_data_base = static_cast<int>(triplet_base(
                physical_tile(loaded_n_block_base + kAtomsPerTriplet,
                              k_begin,
                              k_tiles)));

            if constexpr(ActivationLookahead)
            {
                // Keep the next activation fragment in VGPRs while the second
                // weight triplet for the current K tile is decoded and
                // consumed.  Every steady-state iteration has this request
                // order: next activation, next triplet 0, next triplet 1.
                // Counted waits retire only the current triplet and leave that
                // younger lookahead in flight.
                const int uniform_iterations =
                    __builtin_amdgcn_readfirstlane(iterations);
                opus::gmem<uint8_t> activation_buffer(
                    activations, static_cast<unsigned int>(M * K));
                opus::gmem<uint8_t> scale_buffer(
                    activation_scales,
                    static_cast<unsigned int>(
                        TiledActivationScales
                            ? ((K / kScaleBlock + 3) / 4) * ((M + 15) / 16) *
                                  64
                            : M * (K / kScaleBlock)));
                IQ2RActivationFragment current_activation{};
                uint32_t current_scale_a = 127u * 0x01010101u;
                iq2r_issue_buffered_activation_fragment<StagedScales,
                                                         TiledActivationScales>(
                    activation_buffer,
                    scale_buffer,
                    shared.reusable.scale_cache,
                    load_row,
                    row_base,
                    k_begin,
                    lane_group,
                    M,
                    K,
                    current_activation,
                    current_scale_a);
                iq2r_issue_direct_lds_triplet<LoadAux>(data_buffer,
                                               shared.weight_cache[wave],
                                               lane,
                                               0,
                                               next_data_base);
                iq2r_issue_direct_lds_triplet<LoadAux>(data_buffer,
                                               shared.weight_cache[wave],
                                               lane,
                                               1,
                                               next_data_base + kTripletBytes);
                next_data_base += kGroupBytes;

                for(int iteration = 0; iteration < uniform_iterations; ++iteration)
                {
                    // Activation requests precede both current weight
                    // triplets. Retire the activation and triplet 0 while the
                    // two triplet-1 requests remain outstanding.
                    iq2r_wait_vmcnt<2>();
                    IQ2RCompressedTriplet compressed =
                        iq2r_read_direct_lds_triplet(
                            shared.weight_cache[wave], lane, 0);
                    asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
                    const opus::i32x8_t current_words = current_activation.words;

                    const bool has_next = iteration + 1 < uniform_iterations;
                    uint32_t next_scale_a = 127u * 0x01010101u;
                    if(has_next)
                    {
                        iq2r_issue_buffered_activation_fragment<
                            StagedScales,
                            TiledActivationScales>(
                            activation_buffer,
                            scale_buffer,
                            shared.reusable.scale_cache,
                            load_row,
                            row_base,
                            k_begin + iteration + 1,
                            lane_group,
                            M,
                            K,
                            current_activation,
                            next_scale_a);
                        iq2r_issue_direct_lds_triplet<LoadAux>(data_buffer,
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
                    iq2r_cooperative_triplet_mfma<0>(
                        current_words,
                        weight_fragments,
                        accumulators,
                        current_scale_a,
                        scale_b);

                    if(has_next)
                    {
                        // Two activation requests and two triplet-0 requests
                        // remain younger when scales are staged in LDS. The
                        // row-major variant has one additional scale request.
                        if constexpr(StagedScales)
                            iq2r_wait_vmcnt<4>();
                        else
                            iq2r_wait_vmcnt<5>();
                    }
                    else
                        iq2r_wait_vmcnt<0>();

                    compressed = iq2r_read_direct_lds_triplet(
                        shared.weight_cache[wave], lane, 1);
                    asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
                    if(has_next)
                    {
                        iq2r_issue_direct_lds_triplet<LoadAux>(data_buffer,
                                                       shared.weight_cache[wave],
                                                       lane,
                                                       1,
                                                       next_data_base +
                                                           kTripletBytes);
                        next_data_base += kGroupBytes;
                    }
                    scale_b = iq2r_decode_direct_lds_triplet(compressed,
                                                             shared.codebook,
                                                             packed_bases[1],
                                                             weight_fragments);
                    iq2r_cooperative_triplet_mfma<3>(
                        current_words,
                        weight_fragments,
                        accumulators,
                        current_scale_a,
                        scale_b);
                    if(has_next)
                        current_scale_a = next_scale_a;
                }
            }
            else
            {
                if(iterations > 0)
                {
                if constexpr(OutputAtoms == 4)
                {
                    iq2r_issue_direct_lds_triplet<LoadAux>(
                        data_buffer,
                        shared.weight_cache[wave],
                        lane,
                        0,
                        next_data_base);
                    iq2r_issue_direct_lds_triplet<LoadAux>(
                        data_buffer,
                        shared.weight_cache[wave],
                        lane,
                        1,
                        next_second_data_base);
                }
                else
                {
#pragma unroll
                    for(int triplet = 0; triplet < kDirectTriplets; ++triplet)
                        iq2r_issue_direct_lds_triplet<LoadAux>(
                            data_buffer,
                            shared.weight_cache[wave],
                            lane,
                            triplet,
                            next_data_base + triplet * kTripletBytes);
                }
                }
                next_data_base += kGroupBytes;
                next_second_data_base += kGroupBytes;
                int tiled_scale_offset =
                    (k_begin * ((M + 15) / 16) + load_row / 16) * 64 +
                    lane_group * 16 + load_row % 16;
                const int tiled_scale_stride = ((M + 15) / 16) * 64;

                for(int iteration = 0; iteration < iterations; ++iteration)
                {
                    const int k_tile = k_begin + iteration;
                    IQ2RActivationFragment activation_fragment{};
                    const int activation_k = k_tile * kTileK + lane_group * 16;
                    if(activation_k + 15 < K)
                    {
                        const auto* activation_row =
                            reinterpret_cast<const uint8_t*>(
                                activations + static_cast<int64_t>(load_row) * K);
                        *reinterpret_cast<uint4*>(activation_fragment.bytes) =
                            *reinterpret_cast<const uint4*>(activation_row +
                                                           activation_k);
                        if(activation_k + 64 + 15 < K)
                            *reinterpret_cast<uint4*>(activation_fragment.bytes +
                                                     16) =
                                *reinterpret_cast<const uint4*>(activation_row +
                                                               activation_k + 64);
                    }

                    uint32_t scale_a = 127u;
                    if constexpr(StagedScales)
                    {
                        const uint32_t exponent = shared.reusable.scale_cache[
                            k_tile * 64 + lane_group * 16 + load_row - row_base];
                        scale_a = exponent;
                    }
                    else
                    {
                        const int scale_column = k_tile * 4 + lane_group;
                        if(scale_column < K / kScaleBlock)
                        {
                            uint32_t exponent;
                            if constexpr(TiledActivationScales)
                            {
                                exponent = activation_scales[tiled_scale_offset];
                                tiled_scale_offset += tiled_scale_stride;
                            }
                            else
                                exponent = activation_scales[
                                    static_cast<int64_t>(load_row) *
                                            (K / kScaleBlock) +
                                        scale_column];
                            scale_a = exponent;
                        }
                    }

                    // The direct-to-LDS requests precede two or three activation
                    // requests. Retire only the older weight requests first so
                    // their LDS reads and IQ2R decode overlap the activation tail.
                    if(k_tile + 1 < k_tiles)
                    {
                        if constexpr(StagedScales)
                            iq2r_wait_vmcnt<2>();
                        else
                            iq2r_wait_vmcnt<3>();
                    }
                    else
                    {
                        if constexpr(StagedScales)
                            iq2r_wait_vmcnt<1>();
                        else
                            iq2r_wait_vmcnt<2>();
                    }

                    IQ2RCompressedTriplet compressed =
                        iq2r_read_direct_lds_triplet(
                            shared.weight_cache[wave], lane, 0);
                    asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
                    asm volatile("" : "+v"(scale_a));
                    if(iteration + 1 < iterations)
                    {
                        if constexpr(OutputAtoms == 4)
                            iq2r_issue_direct_lds_triplet<LoadAux>(
                                data_buffer,
                                shared.weight_cache[wave],
                                lane,
                                0,
                                next_data_base);
                        else
                            iq2r_issue_direct_lds_triplet<LoadAux>(
                                data_buffer,
                                shared.weight_cache[wave],
                                lane,
                                0,
                                next_data_base);
                    }
                    if constexpr(OutputAtoms == 4)
                    {
                        opus::i32x8_t weight_fragment;
                        uint32_t scale_b;
                        if(fused_atom_offset == 0)
                        {
                            iq2r_decode_direct_lds_atom<0>(compressed,
                                                           shared.codebook,
                                                           packed_bases[0],
                                                           weight_fragment,
                                                           scale_b);
                            iq2r_cooperative_single_mfma<0>(
                                activation_fragment.words, weight_fragment,
                                accumulators, scale_a, scale_b);
                            iq2r_decode_direct_lds_atom<1>(compressed,
                                                           shared.codebook,
                                                           packed_bases[0],
                                                           weight_fragment,
                                                           scale_b);
                            iq2r_cooperative_single_mfma<1>(
                                activation_fragment.words, weight_fragment,
                                accumulators, scale_a, scale_b);
                            iq2r_decode_direct_lds_atom<2>(compressed,
                                                           shared.codebook,
                                                           packed_bases[0],
                                                           weight_fragment,
                                                           scale_b);
                            iq2r_cooperative_single_mfma<2>(
                                activation_fragment.words, weight_fragment,
                                accumulators, scale_a, scale_b);
                        }
                        else if(fused_atom_offset == 1)
                        {
                            iq2r_decode_direct_lds_atom<1>(compressed,
                                                           shared.codebook,
                                                           packed_bases[0],
                                                           weight_fragment,
                                                           scale_b);
                            iq2r_cooperative_single_mfma<0>(
                                activation_fragment.words, weight_fragment,
                                accumulators, scale_a, scale_b);
                            iq2r_decode_direct_lds_atom<2>(compressed,
                                                           shared.codebook,
                                                           packed_bases[0],
                                                           weight_fragment,
                                                           scale_b);
                            iq2r_cooperative_single_mfma<1>(
                                activation_fragment.words, weight_fragment,
                                accumulators, scale_a, scale_b);
                        }
                        else
                        {
                            iq2r_decode_direct_lds_atom<2>(compressed,
                                                           shared.codebook,
                                                           packed_bases[0],
                                                           weight_fragment,
                                                           scale_b);
                            iq2r_cooperative_single_mfma<0>(
                                activation_fragment.words, weight_fragment,
                                accumulators, scale_a, scale_b);
                        }
                    }
                    else
                    {
                        opus::i32x8_t weight_fragments[kAtomsPerTriplet];
                        const uint32_t scale_b = iq2r_decode_direct_lds_triplet(
                            compressed,
                            shared.codebook,
                            packed_bases[0],
                            weight_fragments);
                        iq2r_cooperative_triplet_mfma<0>(
                            activation_fragment.words,
                            weight_fragments,
                            accumulators,
                            scale_a * 0x01010101u,
                            scale_b);
                    }
                    if constexpr(OutputAtoms > 3)
                    {
                        compressed = iq2r_read_direct_lds_triplet(
                            shared.weight_cache[wave], lane, 1);
                        asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
                        if(iteration + 1 < iterations)
                        {
                            if constexpr(OutputAtoms == 4)
                                iq2r_issue_direct_lds_triplet<LoadAux>(
                                    data_buffer,
                                    shared.weight_cache[wave],
                                    lane,
                                    1,
                                    next_second_data_base);
                            else
                                iq2r_issue_direct_lds_triplet<LoadAux>(
                                    data_buffer,
                                    shared.weight_cache[wave],
                                    lane,
                                    1,
                                    next_data_base + kTripletBytes);
                            next_data_base += kGroupBytes;
                            next_second_data_base += kGroupBytes;
                        }
                        if constexpr(OutputAtoms == 4)
                        {
                            opus::i32x8_t weight_fragment;
                            uint32_t scale_b;
                            if(fused_atom_offset == 0)
                            {
                                iq2r_decode_direct_lds_atom<0>(compressed,
                                                               shared.codebook,
                                                               packed_bases[1],
                                                               weight_fragment,
                                                               scale_b);
                                iq2r_cooperative_single_mfma<3>(
                                    activation_fragment.words, weight_fragment,
                                    accumulators, scale_a, scale_b);
                            }
                            else if(fused_atom_offset == 1)
                            {
                                iq2r_decode_direct_lds_atom<0>(compressed,
                                                               shared.codebook,
                                                               packed_bases[1],
                                                               weight_fragment,
                                                               scale_b);
                                iq2r_cooperative_single_mfma<2>(
                                    activation_fragment.words, weight_fragment,
                                    accumulators, scale_a, scale_b);
                                iq2r_decode_direct_lds_atom<1>(compressed,
                                                               shared.codebook,
                                                               packed_bases[1],
                                                               weight_fragment,
                                                               scale_b);
                                iq2r_cooperative_single_mfma<3>(
                                    activation_fragment.words, weight_fragment,
                                    accumulators, scale_a, scale_b);
                            }
                            else
                            {
                                iq2r_decode_direct_lds_atom<0>(compressed,
                                                               shared.codebook,
                                                               packed_bases[1],
                                                               weight_fragment,
                                                               scale_b);
                                iq2r_cooperative_single_mfma<1>(
                                    activation_fragment.words, weight_fragment,
                                    accumulators, scale_a, scale_b);
                                iq2r_decode_direct_lds_atom<1>(compressed,
                                                               shared.codebook,
                                                               packed_bases[1],
                                                               weight_fragment,
                                                               scale_b);
                                iq2r_cooperative_single_mfma<2>(
                                    activation_fragment.words, weight_fragment,
                                    accumulators, scale_a, scale_b);
                                iq2r_decode_direct_lds_atom<2>(compressed,
                                                               shared.codebook,
                                                               packed_bases[1],
                                                               weight_fragment,
                                                               scale_b);
                                iq2r_cooperative_single_mfma<3>(
                                    activation_fragment.words, weight_fragment,
                                    accumulators, scale_a, scale_b);
                            }
                        }
                        else
                        {
                            opus::i32x8_t weight_fragments[kAtomsPerTriplet];
                            const uint32_t scale_b = iq2r_decode_direct_lds_triplet(
                                compressed,
                                shared.codebook,
                                packed_bases[1],
                                weight_fragments);
                            iq2r_cooperative_triplet_mfma<3>(
                                activation_fragment.words,
                                weight_fragments,
                                accumulators,
                                scale_a,
                                scale_b);
                        }
                    }
                    else if(iteration + 1 < iterations)
                        next_data_base += kGroupBytes;
                }
            }

            if constexpr(StagedScales)
                __syncthreads();

            const int output_row = row_base + (lane / 16) * 4 + wave;
            if constexpr(FusedSwiGLU)
            {
                float raw_values[4] = {};
#pragma unroll
                for(int atom = 0; atom < 3; ++atom)
                    shared.reusable.partial[wave][atom][lane] = accumulators[atom];
                __syncthreads();
                if(wave < 4 && output_row < sub_end)
                {
#pragma unroll
                    for(int atom = 0; atom < 3; ++atom)
                    {
                        const float* p0 = reinterpret_cast<const float*>(
                            &shared.reusable.partial[0][atom][lane]);
                        const float* p1 = reinterpret_cast<const float*>(
                            &shared.reusable.partial[1][atom][lane]);
                        const float* p2 = reinterpret_cast<const float*>(
                            &shared.reusable.partial[2][atom][lane]);
                        const float* p3 = reinterpret_cast<const float*>(
                            &shared.reusable.partial[3][atom][lane]);
                        float value = p0[wave] + p1[wave] + p2[wave] + p3[wave];
                        if(all_bias != nullptr)
                            value += __bfloat162float(
                                all_bias[static_cast<int64_t>(expert_index) * N +
                                         n_tile_index * kOutputColumns + atom * 16 +
                                         lane % 16]);
                        raw_values[atom] = __bfloat162float(__float2bfloat16(value));
                    }
                }
                __syncthreads();

                shared.reusable.partial[wave][0][lane] = accumulators[3];
                __syncthreads();
                if(wave < 4 && output_row < sub_end)
                {
                    const float* p0 = reinterpret_cast<const float*>(
                        &shared.reusable.partial[0][0][lane]);
                    const float* p1 = reinterpret_cast<const float*>(
                        &shared.reusable.partial[1][0][lane]);
                    const float* p2 = reinterpret_cast<const float*>(
                        &shared.reusable.partial[2][0][lane]);
                    const float* p3 = reinterpret_cast<const float*>(
                        &shared.reusable.partial[3][0][lane]);
                    float value = p0[wave] + p1[wave] + p2[wave] + p3[wave];
                    if(all_bias != nullptr)
                        value += __bfloat162float(
                            all_bias[static_cast<int64_t>(expert_index) * N +
                                     n_tile_index * kOutputColumns + 3 * 16 +
                                     lane % 16]);
                    raw_values[3] = __bfloat162float(__float2bfloat16(value));
                }
                __syncthreads();

                if(wave < 4 && output_row < sub_end)
                {
                    const bool even_column = (lane % 2) == 0;
                    opus::vector_t<opus::bf16_t, 4> activated;
                    float abs_max = 1.0e-10f;
#pragma unroll
                    for(int atom = 0; atom < 4; ++atom)
                    {
                        const float paired = __shfl_xor(raw_values[atom], 1);
                        if(even_column)
                        {
                            const float gate = raw_values[atom];
                            const float up = paired;
                            const bool clamp = swiglu_limit > 0.0f;
                            const float clipped_gate =
                                clamp && gate > swiglu_limit ? swiglu_limit : gate;
                            const float clipped_up =
                                clamp && up < -swiglu_limit
                                    ? -swiglu_limit
                                    : (clamp && up > swiglu_limit
                                           ? swiglu_limit
                                           : up);
                            const float swish =
                                clipped_gate /
                                (1.0f + __expf(-swiglu_alpha * clipped_gate));
                            const __hip_bfloat16 rounded = __float2bfloat16(
                                swish * (clipped_up + swiglu_up_offset));
                            activated[atom] =
                                __builtin_bit_cast(opus::bf16_t, rounded);
                            abs_max =
                                fmaxf(abs_max, fabsf(__bfloat162float(rounded)));
                        }
                    }
                    abs_max = fmaxf(abs_max, __shfl_xor(abs_max, 2));
                    abs_max = fmaxf(abs_max, __shfl_xor(abs_max, 4));
                    abs_max = fmaxf(abs_max, __shfl_xor(abs_max, 8));
                    if(even_column)
                    {
                        const auto block_scale = fp_f32_to_e8m0_block_scale<
                            kDefaultMxScaleRoundMode,
                            MxDtype::FP8_E4M3>(abs_max);
                        const float inverse_scale = 1.0f / block_scale.dq_scale;
                        const int output_width = N / 2;
                        const int output_column_base =
                            n_tile_index * (kOutputColumns / 2) + (lane % 16) / 2;
#pragma unroll
                        for(int atom = 0; atom < 4; ++atom)
                            fused_output[static_cast<int64_t>(output_row) *
                                             output_width +
                                         output_column_base + atom * 8] =
                                opus::fp32_to_fp8(
                                    static_cast<float>(activated[atom]) *
                                    inverse_scale);
                        if(lane % 16 == 0)
                            fused_output_scales[
                                static_cast<int64_t>(output_row) * (N / 64) +
                                n_tile_index] = block_scale.byte;
                    }
                }
                __syncthreads();
            }
            else
            {
                if(wave < 2)
                {
#pragma unroll
                    for(int atom = 0; atom < 6; ++atom)
                        shared.reusable.partial[wave][atom][lane] = accumulators[atom];
                }
                __syncthreads();
                if(output_row < sub_end)
                {
#pragma unroll
                    for(int atom = 0; atom < 6; ++atom)
                    {
                        const int column = n_tile_index*96 + atom*16 + lane%16;
                        const float* p0 = reinterpret_cast<const float*>(&shared.reusable.partial[0][atom][lane]);
                        const float* p1 = reinterpret_cast<const float*>(&shared.reusable.partial[1][atom][lane]);
                        float value = ((p0[wave] + p1[wave]) + 0.0f) + 0.0f;
                        if(all_bias != nullptr)
                            value += __bfloat162float(all_bias[static_cast<int64_t>(expert_index)*N + column]);
                        output[static_cast<int64_t>(output_row)*N + column] = __float2bfloat16(value);
                    }
                }
                __syncthreads();
            }

        }
    }
#endif
}



__global__ __launch_bounds__(256,2) void iq2r_scheduled_small_register_single_epilogue_kernel(
    const opus::fp8_t* __restrict__ activations,
    const uint8_t* __restrict__ activation_scales,
    const uint8_t* __restrict__ all_data,
    const uint8_t* __restrict__ all_auxiliary,
    const int32_t* __restrict__ tasks,
    const int32_t* __restrict__ task_count,
    const __hip_bfloat16* __restrict__ all_bias,
    __hip_bfloat16* __restrict__ output,
    opus::fp8_t* __restrict__ fused_output,
    uint8_t* __restrict__ fused_output_scales,
    int M,
    int unused_N,
    int unused_K,
    int expert_count,
    int data_bytes,
    int auxiliary_bytes,
    float swiglu_limit,
    float swiglu_alpha,
    float swiglu_up_offset)
{
#if defined(__gfx950__)
    constexpr int N = 6144, K = 256;
    constexpr int OutputAtoms = 6, PhysicalWaves = 4, LoadAux = 2;
    constexpr bool ActivationLookahead = false, StagedScales = false;
    constexpr bool TiledActivationScales = false, VectorCodebookCopy = true, FusedSwiGLU = false;
    static_assert(OutputAtoms == 3 || OutputAtoms == 4 || OutputAtoms == 6);
    static_assert(PhysicalWaves == 2 || PhysicalWaves == 4 || PhysicalWaves == 8);
    static_assert(!ActivationLookahead || OutputAtoms == 6);
    static_assert(!(StagedScales && TiledActivationScales));
    static_assert(!FusedSwiGLU ||
                  (OutputAtoms == 4 && PhysicalWaves == 4 &&
                   !ActivationLookahead && !StagedScales));
    constexpr int kOutputColumns = OutputAtoms * kTileN;
    constexpr int kDirectTriplets =
        (OutputAtoms + kAtomsPerTriplet - 1) / kAtomsPerTriplet;
    constexpr int kStoredAtoms = OutputAtoms < 3 ? OutputAtoms : 3;
    constexpr int kScaleGroups = 2880 / kScaleBlock;
    constexpr int kPaddedScaleGroups = ((kScaleGroups + 3) / 4) * 4;
    constexpr int kStagedScaleBytes = 16 * kPaddedScaleGroups;

    struct SharedStorage
    {
        alignas(16) uint64_t codebook[kCodebookBytes / sizeof(uint64_t)];
        union
        {
            alignas(16) opus::vector_t<float, 4>
                partial[2][6][64];
            alignas(16) uint8_t scale_cache[kStagedScaleBytes];
        } reusable;
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
        if(row_begin < 0 || row_count <= 0 || row_begin + row_count > M)
            continue;
        if(expert_index < 0 || expert_index >= expert_count)
        {
            // Expert-parallel callers retain non-local routes in the sorted
            // permutation so the final indexed reduction can preserve the
            // original top-k order. Materialize their contribution exactly as
            // zero, matching the generic task kernel instead of leaving stale
            // workspace values behind.
            const int output_column_begin =
                n_tile_index * (FusedSwiGLU ? kOutputColumns / 2
                                            : kOutputColumns);
            const int output_width = FusedSwiGLU ? N / 2 : N;
            const int output_columns = min(FusedSwiGLU ? kOutputColumns / 2
                                                       : kOutputColumns,
                                           output_width - output_column_begin);
            const int task_elements = row_count * output_columns;
            for(int element = linear_thread; element < task_elements;
                element += linear_threads)
            {
                const int row = row_begin + element / output_columns;
                const int column = output_column_begin + element % output_columns;
                if constexpr(FusedSwiGLU)
                    fused_output[static_cast<int64_t>(row) * output_width + column] =
                        opus::fp32_to_fp8(0.0f);
                else
                    output[static_cast<int64_t>(row) * output_width + column] =
                        __float2bfloat16(0.0f);
            }
            if constexpr(FusedSwiGLU)
            {
                if(linear_thread < row_count)
                    fused_output_scales[
                        static_cast<int64_t>(row_begin + linear_thread) *
                            (N / 64) +
                        n_tile_index] = 127;
            }
            continue;
        }

        const uint8_t* data =
            all_data + static_cast<int64_t>(expert_index) * data_bytes;
        const uint8_t* auxiliary =
            all_auxiliary + static_cast<int64_t>(expert_index) * auxiliary_bytes;
        opus::gmem<uint8_t> data_buffer(data, static_cast<unsigned int>(data_bytes));
        if(expert_index != previous_expert)
        {
            if constexpr(VectorCodebookCopy && PhysicalWaves == 4)
            {
                reinterpret_cast<uint4*>(shared.codebook)[linear_thread] =
                    reinterpret_cast<const uint4*>(auxiliary)[linear_thread];
            }
            else
            {
                for(int entry = linear_thread;
                    entry < kCodebookBytes / static_cast<int>(sizeof(uint64_t));
                    entry += linear_threads)
                    shared.codebook[entry] =
                        reinterpret_cast<const uint64_t*>(auxiliary)[entry];
            }
            __syncthreads();
            previous_expert = expert_index;
        }

        const int n_block_base = n_tile_index * OutputAtoms;
        // Four-atom tiles advance by 64 columns and are not generally aligned
        // to the three-atom (48-column) physical IQ2R packing groups. Load the
        // two enclosing triplets and decode the requested four consecutive
        // atoms, matching the established fused-SwiGLU path.
        const int loaded_n_block_base =
            OutputAtoms == 4 ? n_block_base - n_block_base % kAtomsPerTriplet
                             : n_block_base;
        const int fused_atom_offset = n_block_base - loaded_n_block_base;
        uint32_t packed_bases[kDirectTriplets] = {};
#pragma unroll
        for(int triplet = 0; triplet < kDirectTriplets; ++triplet)
        {
#pragma unroll
            for(int atom = 0; atom < kAtomsPerTriplet; ++atom)
                packed_bases[triplet] |=
                    static_cast<uint32_t>(
                        auxiliary[kCodebookBytes + loaded_n_block_base +
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

            if constexpr(StagedScales)
            {
                const int valid_rows = sub_end - row_base;
                for(int index = linear_thread; index < kStagedScaleBytes;
                    index += linear_threads)
                {
                    const int row = index / kPaddedScaleGroups;
                    const int group = index % kPaddedScaleGroups;
                    uint8_t exponent = 127;
                    if(group < kScaleGroups)
                    {
                        const int source_row = row_base + min(row, valid_rows - 1);
                        exponent = activation_scales[
                            iq2r_activation_scale_offset<TiledActivationScales>(
                                source_row, group, M, kScaleGroups)];
                    }
                    const int scale_k_tile = group / 4;
                    const int scale_group = group % 4;
                    shared.reusable.scale_cache[scale_k_tile * 64 +
                                                scale_group * 16 + row] = exponent;
                }
                __syncthreads();
            }
            opus::vector_t<float, 4> accumulators[OutputAtoms] = {};

            int next_data_base = static_cast<int>(triplet_base(
                physical_tile(loaded_n_block_base, k_begin, k_tiles)));
            int next_second_data_base = static_cast<int>(triplet_base(
                physical_tile(loaded_n_block_base + kAtomsPerTriplet,
                              k_begin,
                              k_tiles)));

            if(wave < 2)
            {
                const auto first = iq2r_scheduled_load_compact_uniform(data, data_bytes, next_data_base, lane);
                const auto second = iq2r_scheduled_load_compact_uniform(data, data_bytes, next_data_base + kTripletBytes, lane);
                IQ2RActivationFragment a;
                const uint8_t* input = reinterpret_cast<const uint8_t*>(activations) + static_cast<int64_t>(load_row)*256 + wave*128 + lane_group*16;
                *reinterpret_cast<uint4*>(a.bytes) = *reinterpret_cast<const uint4*>(input);
                *reinterpret_cast<uint4*>(a.bytes+16) = *reinterpret_cast<const uint4*>(input+64);
                uint32_t sa = activation_scales[static_cast<int64_t>(load_row)*8 + wave*4 + lane_group];
                opus::i32x8_t b[3];
                uint32_t sb = iq2r_decode_direct_lds_triplet(first, shared.codebook, packed_bases[0], b);
                asm volatile("" : "+v"(sa));
                iq2r_cooperative_triplet_mfma<0>(a.words, b, accumulators, sa*0x01010101u, sb);
                sb = iq2r_decode_direct_lds_triplet(second, shared.codebook, packed_bases[1], b);
                iq2r_cooperative_triplet_mfma<3>(a.words, b, accumulators, sa*0x01010101u, sb);
            }
            if constexpr(StagedScales)
                __syncthreads();

            const int output_row = row_base + (lane / 16) * 4 + wave;
            if constexpr(FusedSwiGLU)
            {
                float raw_values[4] = {};
#pragma unroll
                for(int atom = 0; atom < 3; ++atom)
                    shared.reusable.partial[wave][atom][lane] = accumulators[atom];
                __syncthreads();
                if(wave < 4 && output_row < sub_end)
                {
#pragma unroll
                    for(int atom = 0; atom < 3; ++atom)
                    {
                        const float* p0 = reinterpret_cast<const float*>(
                            &shared.reusable.partial[0][atom][lane]);
                        const float* p1 = reinterpret_cast<const float*>(
                            &shared.reusable.partial[1][atom][lane]);
                        const float* p2 = reinterpret_cast<const float*>(
                            &shared.reusable.partial[2][atom][lane]);
                        const float* p3 = reinterpret_cast<const float*>(
                            &shared.reusable.partial[3][atom][lane]);
                        float value = p0[wave] + p1[wave] + p2[wave] + p3[wave];
                        if(all_bias != nullptr)
                            value += __bfloat162float(
                                all_bias[static_cast<int64_t>(expert_index) * N +
                                         n_tile_index * kOutputColumns + atom * 16 +
                                         lane % 16]);
                        raw_values[atom] = __bfloat162float(__float2bfloat16(value));
                    }
                }
                __syncthreads();

                shared.reusable.partial[wave][0][lane] = accumulators[3];
                __syncthreads();
                if(wave < 4 && output_row < sub_end)
                {
                    const float* p0 = reinterpret_cast<const float*>(
                        &shared.reusable.partial[0][0][lane]);
                    const float* p1 = reinterpret_cast<const float*>(
                        &shared.reusable.partial[1][0][lane]);
                    const float* p2 = reinterpret_cast<const float*>(
                        &shared.reusable.partial[2][0][lane]);
                    const float* p3 = reinterpret_cast<const float*>(
                        &shared.reusable.partial[3][0][lane]);
                    float value = p0[wave] + p1[wave] + p2[wave] + p3[wave];
                    if(all_bias != nullptr)
                        value += __bfloat162float(
                            all_bias[static_cast<int64_t>(expert_index) * N +
                                     n_tile_index * kOutputColumns + 3 * 16 +
                                     lane % 16]);
                    raw_values[3] = __bfloat162float(__float2bfloat16(value));
                }
                __syncthreads();

                if(wave < 4 && output_row < sub_end)
                {
                    const bool even_column = (lane % 2) == 0;
                    opus::vector_t<opus::bf16_t, 4> activated;
                    float abs_max = 1.0e-10f;
#pragma unroll
                    for(int atom = 0; atom < 4; ++atom)
                    {
                        const float paired = __shfl_xor(raw_values[atom], 1);
                        if(even_column)
                        {
                            const float gate = raw_values[atom];
                            const float up = paired;
                            const bool clamp = swiglu_limit > 0.0f;
                            const float clipped_gate =
                                clamp && gate > swiglu_limit ? swiglu_limit : gate;
                            const float clipped_up =
                                clamp && up < -swiglu_limit
                                    ? -swiglu_limit
                                    : (clamp && up > swiglu_limit
                                           ? swiglu_limit
                                           : up);
                            const float swish =
                                clipped_gate /
                                (1.0f + __expf(-swiglu_alpha * clipped_gate));
                            const __hip_bfloat16 rounded = __float2bfloat16(
                                swish * (clipped_up + swiglu_up_offset));
                            activated[atom] =
                                __builtin_bit_cast(opus::bf16_t, rounded);
                            abs_max =
                                fmaxf(abs_max, fabsf(__bfloat162float(rounded)));
                        }
                    }
                    abs_max = fmaxf(abs_max, __shfl_xor(abs_max, 2));
                    abs_max = fmaxf(abs_max, __shfl_xor(abs_max, 4));
                    abs_max = fmaxf(abs_max, __shfl_xor(abs_max, 8));
                    if(even_column)
                    {
                        const auto block_scale = fp_f32_to_e8m0_block_scale<
                            kDefaultMxScaleRoundMode,
                            MxDtype::FP8_E4M3>(abs_max);
                        const float inverse_scale = 1.0f / block_scale.dq_scale;
                        const int output_width = N / 2;
                        const int output_column_base =
                            n_tile_index * (kOutputColumns / 2) + (lane % 16) / 2;
#pragma unroll
                        for(int atom = 0; atom < 4; ++atom)
                            fused_output[static_cast<int64_t>(output_row) *
                                             output_width +
                                         output_column_base + atom * 8] =
                                opus::fp32_to_fp8(
                                    static_cast<float>(activated[atom]) *
                                    inverse_scale);
                        if(lane % 16 == 0)
                            fused_output_scales[
                                static_cast<int64_t>(output_row) * (N / 64) +
                                n_tile_index] = block_scale.byte;
                    }
                }
                __syncthreads();
            }
            else
            {
                if(wave < 2)
                {
#pragma unroll
                    for(int atom = 0; atom < 6; ++atom)
                        shared.reusable.partial[wave][atom][lane] = accumulators[atom];
                }
                __syncthreads();
                if(output_row < sub_end)
                {
#pragma unroll
                    for(int atom = 0; atom < 6; ++atom)
                    {
                        const int column = n_tile_index*96 + atom*16 + lane%16;
                        const float* p0 = reinterpret_cast<const float*>(&shared.reusable.partial[0][atom][lane]);
                        const float* p1 = reinterpret_cast<const float*>(&shared.reusable.partial[1][atom][lane]);
                        float value = ((p0[wave] + p1[wave]) + 0.0f) + 0.0f;
                        if(all_bias != nullptr)
                            value += __bfloat162float(all_bias[static_cast<int64_t>(expert_index)*N + column]);
                        output[static_cast<int64_t>(output_row)*N + column] = __float2bfloat16(value);
                    }
                }
                __syncthreads();
            }

        }
    }
#endif
}


__global__ __launch_bounds__(512,1) void iq2r_scheduled_gate_wait3_kernel(
    const opus::fp8_t* __restrict__ activations,
    const uint8_t* __restrict__ scales,
    const uint8_t* __restrict__ all_data,
    const uint8_t* __restrict__ all_auxiliary,
    const int32_t* __restrict__ tasks,
    const int32_t* __restrict__ task_count,
    const int32_t* __restrict__ gather,
    opus::fp8_t* __restrict__ output,
    uint8_t* __restrict__ output_scales,
    int routes,int data_bytes,int auxiliary_bytes)
{
#if defined(__gfx950__)
    constexpr int MAtoms=1;
    constexpr int K=6144, Rows=16*MAtoms, Subtasks=1;
    static_assert(MAtoms==1);
    struct Storage {
        alignas(16) uint64_t codebook[kCodebookBytes/8];
        union {
            alignas(16) uint8_t cache[8][kQuadBytes];
            alignas(16) opus::vector_t<float,4> partial[8][4][64];
        } reuse;
        alignas(16) __hip_bfloat16 gate[Rows][64];
    };
    __shared__ Storage shared;
    const int lane=threadIdx.x,wave=threadIdx.y,linear=wave*64+lane;
    const int lane_row=lane%16,lane_group=lane/16;
    const int num_tasks=task_count[0];
    const int total=num_tasks*Subtasks*8;
    for(int work=blockIdx.x;work<total;work+=gridDim.x)
    {
        const int task=(work/Subtasks)%num_tasks;
        const int sub=work%Subtasks;
        const int n_tile=work/(num_tasks*Subtasks);
        const int row_begin=tasks[task*3]+sub*Rows;
        const int row_end=min(row_begin+Rows,tasks[task*3]+tasks[task*3+1]);
        const int expert=tasks[task*3+2];
        if(row_begin>=row_end || row_begin<0 || row_end>routes || expert<0 || expert>=257) continue;
        const uint8_t* data=all_data+static_cast<int64_t>(expert)*data_bytes;
        const uint8_t* aux=all_auxiliary+static_cast<int64_t>(expert)*auxiliary_bytes;
        // Previous work ends with a barrier protecting both union and codebook.
        shared.codebook[linear]=reinterpret_cast<const uint64_t*>(aux)[linear];
        __syncthreads();
        const int nbase=n_tile*4;
        uint32_t bases=0;
#pragma unroll
        for(int atom=0;atom<4;++atom)
            bases|=static_cast<uint32_t>(aux[kCodebookBytes+nbase+atom])<<(atom*8);
        int rows[MAtoms];
#pragma unroll
        for(int m=0;m<MAtoms;++m) rows[m]=min(row_begin+m*16+lane_row,row_end-1);
        opus::vector_t<float,4> accumulators[MAtoms][4]={};
        opus::gmem<uint8_t> buffer(data,static_cast<unsigned int>(data_bytes));
        int base=(n_tile*48+wave*6)*kQuadBytes;
        iq2r_issue_quad(buffer,shared.reuse.cache[wave],lane,base);
        for(int iteration=0;iteration<6;++iteration)
        {
            const int kt=wave*6+iteration;
            IQ2RActivationFragment a[MAtoms]={};uint32_t sa[MAtoms];
#pragma unroll
            for(int m=0;m<MAtoms;++m)
            {
                const auto* input=reinterpret_cast<const uint8_t*>(activations+static_cast<int64_t>(rows[m])*K);
                const int ak=kt*128+lane_group*16;
                *reinterpret_cast<uint4*>(a[m].bytes)=*reinterpret_cast<const uint4*>(input+ak);
                *reinterpret_cast<uint4*>(a[m].bytes+16)=*reinterpret_cast<const uint4*>(input+ak+64);
                sa[m]=scales[static_cast<int64_t>(rows[m])*192+kt*4+lane_group]*0x01010101u;
            }
            iq2r_wait_vmcnt<3>();
            const auto compressed=iq2r_read_quad(shared.reuse.cache[wave],lane);
            asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
            if(iteration<5)
            {
                base+=kQuadBytes;
                iq2r_issue_quad(buffer,shared.reuse.cache[wave],lane,base);
            }
            iq2r_quad_decode_mfma<MAtoms>(compressed,shared.codebook,bases,a,sa,accumulators);
        }
        // No wave may overwrite another wave's cache before its final read.
        __syncthreads();
#pragma unroll
        for(int m=0;m<MAtoms;++m)
        {
#pragma unroll
            for(int atom=0;atom<4;++atom) shared.reuse.partial[wave][atom][lane]=accumulators[m][atom];
            __syncthreads();
            if(wave<4)
            {
                const int local_row=m*16+lane_group*4+wave;
#pragma unroll
                for(int atom=0;atom<4;++atom)
                {
                    float value=0.0f;
#pragma unroll
                    for(int source=0;source<8;++source) value+=shared.reuse.partial[source][atom][lane][wave];
                    shared.gate[local_row][atom*16+lane_row]=__float2bfloat16(value);
                }
            }
            __syncthreads();
        }
        if(linear<Rows*4)
        {
            const int local_row=linear/4,quant_lane=linear%4;
            const int output_row=row_begin+local_row;
            using B=opus::vector_t<opus::bf16_t,8>;
            using Q=opus::vector_t<opus::fp8_t,8>;
            B values;float abs_max=1.0e-10f;
#pragma unroll
            for(int element=0;element<8;++element)
            {
                const int column=quant_lane*8+element;
                const float gate=__bfloat162float(shared.gate[local_row][2*column]);
                const float up=__bfloat162float(shared.gate[local_row][2*column+1]);
                const float swish=gate/(1.0f+__expf(-gate));
                const __hip_bfloat16 rounded=__float2bfloat16(swish*(up+0.0f));
                values[element]=__builtin_bit_cast(opus::bf16_t,rounded);
                abs_max=fmaxf(abs_max,fabsf(__bfloat162float(rounded)));
            }
            abs_max=fmaxf(abs_max,__shfl_xor(abs_max,1));
            abs_max=fmaxf(abs_max,__shfl_xor(abs_max,2));
            if(output_row<row_end)
            {
                const auto bs=fp_f32_to_e8m0_block_scale<kDefaultMxScaleRoundMode,MxDtype::FP8_E4M3>(abs_max);
                const float inverse=1.0f/bs.dq_scale;
                Q q;
#pragma unroll
                for(int element=0;element<8;++element) q[element]=opus::fp32_to_fp8(static_cast<float>(values[element])*inverse);
                *reinterpret_cast<Q*>(output+static_cast<int64_t>(output_row)*256+n_tile*32+quant_lane*8)=q;
                if(quant_lane==0) output_scales[static_cast<int64_t>(output_row)*8+n_tile]=bs.byte;
            }
        }
        __syncthreads();
    }
#endif
}



__global__ __launch_bounds__(512,1) void iq2r_scheduled_gate_register_unroll6_kernel(
    const opus::fp8_t* __restrict__ activations,
    const uint8_t* __restrict__ scales,
    const uint8_t* __restrict__ all_data,
    const uint8_t* __restrict__ all_auxiliary,
    const int32_t* __restrict__ tasks,
    const int32_t* __restrict__ task_count,
    const int32_t* __restrict__ gather,
    opus::fp8_t* __restrict__ output,
    uint8_t* __restrict__ output_scales,
    int routes,int data_bytes,int auxiliary_bytes)
{
#if defined(__gfx950__)
    constexpr int MAtoms=1;
    constexpr int K=6144, Rows=16*MAtoms, Subtasks=1;
    static_assert(MAtoms==1);
    struct Storage {
        alignas(16) uint64_t codebook[kCodebookBytes/8];
        union {
            alignas(16) uint8_t cache[8][kQuadBytes];
            alignas(16) opus::vector_t<float,4> partial[8][4][64];
        } reuse;
        alignas(16) __hip_bfloat16 gate[Rows][64];
    };
    __shared__ Storage shared;
    const int lane=threadIdx.x,wave=threadIdx.y,linear=wave*64+lane;
    const int lane_row=lane%16,lane_group=lane/16;
    const int num_tasks=task_count[0];
    const int total=num_tasks*Subtasks*8;
    for(int work=blockIdx.x;work<total;work+=gridDim.x)
    {
        const int task=(work/Subtasks)%num_tasks;
        const int sub=work%Subtasks;
        const int n_tile=work/(num_tasks*Subtasks);
        const int row_begin=tasks[task*3]+sub*Rows;
        const int row_end=min(row_begin+Rows,tasks[task*3]+tasks[task*3+1]);
        const int expert=tasks[task*3+2];
        if(row_begin>=row_end || row_begin<0 || row_end>routes || expert<0 || expert>=257) continue;
        const uint8_t* data=all_data+static_cast<int64_t>(expert)*data_bytes;
        const uint8_t* aux=all_auxiliary+static_cast<int64_t>(expert)*auxiliary_bytes;
        // Previous work ends with a barrier protecting both union and codebook.
        shared.codebook[linear]=reinterpret_cast<const uint64_t*>(aux)[linear];
        __syncthreads();
        const int nbase=n_tile*4;
        uint32_t bases=0;
#pragma unroll
        for(int atom=0;atom<4;++atom)
            bases|=static_cast<uint32_t>(aux[kCodebookBytes+nbase+atom])<<(atom*8);
        int rows[MAtoms];
#pragma unroll
        for(int m=0;m<MAtoms;++m) rows[m]=min(row_begin+m*16+lane_row,row_end-1);
        opus::vector_t<float,4> accumulators[MAtoms][4]={};
        opus::gmem<uint8_t> buffer(data,static_cast<unsigned int>(data_bytes));
        int base=(n_tile*48+wave*6)*kQuadBytes;
        IQ2RCompressedQuad pending=iq2r_scheduled_load_quad(data,data_bytes,base,lane);
#pragma unroll 6
        for(int iteration=0;iteration<6;++iteration)
        {
            const int kt=wave*6+iteration;
            IQ2RActivationFragment a[MAtoms]={};uint32_t sa[MAtoms];
#pragma unroll
            for(int m=0;m<MAtoms;++m)
            {
                const auto* input=reinterpret_cast<const uint8_t*>(activations+static_cast<int64_t>(rows[m])*K);
                const int ak=kt*128+lane_group*16;
                *reinterpret_cast<uint4*>(a[m].bytes)=*reinterpret_cast<const uint4*>(input+ak);
                *reinterpret_cast<uint4*>(a[m].bytes+16)=*reinterpret_cast<const uint4*>(input+ak+64);
                sa[m]=scales[static_cast<int64_t>(rows[m])*192+kt*4+lane_group]*0x01010101u;
            }
            const auto compressed=pending;
            if(iteration<5)
            {
                base+=kQuadBytes;
                pending=iq2r_scheduled_load_quad(data,data_bytes,base,lane);
            }
            iq2r_quad_decode_mfma<MAtoms>(compressed,shared.codebook,bases,a,sa,accumulators);
        }
        // No wave may overwrite another wave's cache before its final read.
        __syncthreads();
#pragma unroll
        for(int m=0;m<MAtoms;++m)
        {
#pragma unroll
            for(int atom=0;atom<4;++atom) shared.reuse.partial[wave][atom][lane]=accumulators[m][atom];
            __syncthreads();
            if(wave<4)
            {
                const int local_row=m*16+lane_group*4+wave;
#pragma unroll
                for(int atom=0;atom<4;++atom)
                {
                    float value=0.0f;
#pragma unroll
                    for(int source=0;source<8;++source) value+=shared.reuse.partial[source][atom][lane][wave];
                    shared.gate[local_row][atom*16+lane_row]=__float2bfloat16(value);
                }
            }
            __syncthreads();
        }
        if(linear<Rows*4)
        {
            const int local_row=linear/4,quant_lane=linear%4;
            const int output_row=row_begin+local_row;
            using B=opus::vector_t<opus::bf16_t,8>;
            using Q=opus::vector_t<opus::fp8_t,8>;
            B values;float abs_max=1.0e-10f;
#pragma unroll
            for(int element=0;element<8;++element)
            {
                const int column=quant_lane*8+element;
                const float gate=__bfloat162float(shared.gate[local_row][2*column]);
                const float up=__bfloat162float(shared.gate[local_row][2*column+1]);
                const float swish=gate/(1.0f+__expf(-gate));
                const __hip_bfloat16 rounded=__float2bfloat16(swish*(up+0.0f));
                values[element]=__builtin_bit_cast(opus::bf16_t,rounded);
                abs_max=fmaxf(abs_max,fabsf(__bfloat162float(rounded)));
            }
            abs_max=fmaxf(abs_max,__shfl_xor(abs_max,1));
            abs_max=fmaxf(abs_max,__shfl_xor(abs_max,2));
            if(output_row<row_end)
            {
                const auto bs=fp_f32_to_e8m0_block_scale<kDefaultMxScaleRoundMode,MxDtype::FP8_E4M3>(abs_max);
                const float inverse=1.0f/bs.dq_scale;
                Q q;
#pragma unroll
                for(int element=0;element<8;++element) q[element]=opus::fp32_to_fp8(static_cast<float>(values[element])*inverse);
                *reinterpret_cast<Q*>(output+static_cast<int64_t>(output_row)*256+n_tile*32+quant_lane*8)=q;
                if(quant_lane==0) output_scales[static_cast<int64_t>(output_row)*8+n_tile]=bs.byte;
            }
        }
        __syncthreads();
    }
#endif
}


__global__ __launch_bounds__(384,1) void iq2r_down_token_fused48_kernel(
    const opus::fp8_t* __restrict__ activations,
    const uint8_t* __restrict__ scales,
    const uint8_t* __restrict__ all_data,
    const uint8_t* __restrict__ all_auxiliary,
    const int32_t* __restrict__ expert_ids,
    const int32_t* __restrict__ scatter,
    const float* __restrict__ route_weights,
    __hip_bfloat16* __restrict__ output,
    int data_bytes, int auxiliary_bytes)
{
#if defined(__gfx950__)
    constexpr int Atoms=3,N=6144,K=256;
    struct Storage {
        alignas(16) uint64_t codebook[3][kCodebookBytes/8];
        alignas(16) float partial[6][Atoms][16];
        alignas(16) __hip_bfloat16 routes[9][Atoms*16];
    };
    __shared__ Storage shared;
    const int lane=threadIdx.x,wave=threadIdx.y;
    const int route_group=wave/2,kwave=wave%2;
    const int lane_row=lane%16,lane_group=lane/16;
    const int token=blockIdx.y,n_block=static_cast<int>(blockIdx.x)*Atoms;
    const int triplet=n_block/3*3,atom_offset=n_block%3;
    for(int group=0;group<3;++group)
    {
        const int route=group*3+route_group,original=token*9+route;
        const int expert=expert_ids[original],row=scatter[original];
        const uint8_t* data=all_data+static_cast<int64_t>(expert)*data_bytes;
        const uint8_t* aux=all_auxiliary+static_cast<int64_t>(expert)*auxiliary_bytes;
        for(int entry=kwave*64+lane;entry<kCodebookBytes/16;entry+=128)
            reinterpret_cast<uint4*>(shared.codebook[route_group])[entry]=reinterpret_cast<const uint4*>(aux)[entry];
        __syncthreads();
        opus::gmem<uint8_t> buffer(data,static_cast<unsigned int>(data_bytes));
        const int base=static_cast<int>(triplet_base(physical_tile(triplet,kwave,2)));
        const auto compressed=iq2r_scheduled_load_compact_uniform(data,data_bytes,base,lane);
        IQ2RActivationFragment a={};
        const auto* input=reinterpret_cast<const uint8_t*>(activations+static_cast<int64_t>(row)*K);
        const int ak=kwave*128+lane_group*16;
        *reinterpret_cast<uint4*>(a.bytes)=*reinterpret_cast<const uint4*>(input+ak);
        *reinterpret_cast<uint4*>(a.bytes+16)=*reinterpret_cast<const uint4*>(input+ak+64);
        uint32_t sa=scales[static_cast<int64_t>(row)*8+kwave*4+lane_group];
        uint32_t bases=0;
#pragma unroll
        for(int atom=0;atom<3;++atom) bases|=static_cast<uint32_t>(aux[kCodebookBytes+triplet+atom])<<(atom*8);

        opus::vector_t<float,4> sums[Atoms]={};
        if constexpr(Atoms==3)
        {
            opus::i32x8_t b[3];
            const uint32_t sb=iq2r_decode_direct_lds_triplet(compressed,shared.codebook[route_group],bases,b);
            asm volatile("" : "+v"(sa));
            iq2r_cooperative_triplet_mfma<0>(a.words,b,sums,sa*0x01010101u,sb);
        }
        else
        {
            opus::i32x8_t b;uint32_t sb;
            if(atom_offset==0) iq2r_decode_direct_lds_atom<0>(compressed,shared.codebook[route_group],bases,b,sb);
            else if(atom_offset==1) iq2r_decode_direct_lds_atom<1>(compressed,shared.codebook[route_group],bases,b,sb);
            else iq2r_decode_direct_lds_atom<2>(compressed,shared.codebook[route_group],bases,b,sb);
            asm volatile("" : "+v"(sa));
            iq2r_cooperative_single_mfma<0>(a.words,b,sums,sa*0x01010101u,sb);
        }
#pragma unroll
        for(int atom=0;atom<Atoms;++atom) if(lane_group==0) shared.partial[wave][atom][lane_row]=sums[atom][0];
        __syncthreads();
        if(kwave==0 && lane_group==0)
        {
#pragma unroll
            for(int atom=0;atom<Atoms;++atom)
            {
                const float value=shared.partial[wave][atom][lane_row]+shared.partial[wave+1][atom][lane_row];
                shared.routes[route][atom*16+lane_row]=__float2bfloat16(value);
            }
        }
        __syncthreads();
    }
    if(wave==0 && lane_group==0)
    {
#pragma unroll
        for(int atom=0;atom<Atoms;++atom)
        {
            float combined=0.0f;
#pragma unroll
            for(int route=0;route<9;++route)
                combined=fmaf(__bfloat162float(shared.routes[route][atom*16+lane_row]),route_weights[token*9+route],combined);
            output[static_cast<int64_t>(token)*N+(n_block+atom)*16+lane_row]=__float2bfloat16(combined);
        }
    }
#endif
}
__global__ __launch_bounds__(576,1) void iq2r_down_token_route9_kernel(
    const opus::fp8_t* __restrict__ activations,
    const uint8_t* __restrict__ scales,
    const uint8_t* __restrict__ all_data,
    const uint8_t* __restrict__ all_auxiliary,
    const int32_t* __restrict__ expert_ids,
    const int32_t* __restrict__ scatter,
    const float* __restrict__ route_weights,
    __hip_bfloat16* __restrict__ output,
    int data_bytes, int auxiliary_bytes)
{
#if defined(__gfx950__)
    constexpr int Groups=9,KWaves=1,Atoms=3,N=6144,K=256;
    struct Storage {
        alignas(16) uint64_t codebook[Groups][kCodebookBytes/8];
        alignas(16) __hip_bfloat16 routes[9][Atoms*16];
    };
    __shared__ Storage shared;
    const int lane=threadIdx.x,wave=threadIdx.y;
    const int route_group=wave/KWaves,kwave=wave%KWaves;
    const int lane_row=lane%16,lane_group=lane/16;
    const int token=blockIdx.y,n_block=static_cast<int>(blockIdx.x)*Atoms;
    const int triplet=n_block/3*3,atom_offset=n_block%3;
    for(int group=0;group<9/Groups;++group)
    {
        const int route=group*Groups+route_group,original=token*9+route;
        const int expert=expert_ids[original],row=scatter[original];
        const bool valid=expert>=0 && expert<257 && row>=0 && row<static_cast<int>(gridDim.y)*9;
        const uint8_t* data=all_data+static_cast<int64_t>(expert)*data_bytes;
        const uint8_t* aux=all_auxiliary+static_cast<int64_t>(expert)*auxiliary_bytes;
        if(valid)
        for(int entry=kwave*64+lane;entry<kCodebookBytes/16;entry+=KWaves*64)
            reinterpret_cast<uint4*>(shared.codebook[route_group])[entry]=reinterpret_cast<const uint4*>(aux)[entry];
        __syncthreads();
        if(valid)
        {
        opus::vector_t<float,4> k_sums[2][Atoms]={};
#pragma unroll
        for(int ktile=0;ktile<2;++ktile)
        {
            const int base=static_cast<int>(triplet_base(physical_tile(triplet,ktile,2)));
            const auto compressed=iq2r_scheduled_load_compact_uniform(data,data_bytes,base,lane);
            IQ2RActivationFragment a={};
            const auto* input=reinterpret_cast<const uint8_t*>(activations+static_cast<int64_t>(row)*K);
            const int ak=ktile*128+lane_group*16;
            *reinterpret_cast<uint4*>(a.bytes)=*reinterpret_cast<const uint4*>(input+ak);
            *reinterpret_cast<uint4*>(a.bytes+16)=*reinterpret_cast<const uint4*>(input+ak+64);
            uint32_t sa=scales[static_cast<int64_t>(row)*8+ktile*4+lane_group];
            uint32_t bases=0;
#pragma unroll
            for(int atom=0;atom<3;++atom) bases|=static_cast<uint32_t>(aux[kCodebookBytes+triplet+atom])<<(atom*8);
            opus::i32x8_t b[3];
            const uint32_t sb=iq2r_decode_direct_lds_triplet(compressed,shared.codebook[route_group],bases,b);
            asm volatile("" : "+v"(sa));
            iq2r_cooperative_triplet_mfma<0>(a.words,b,k_sums[ktile],sa*0x01010101u,sb);
        }
        if(lane_group==0)
        {
#pragma unroll
            for(int atom=0;atom<Atoms;++atom)
            {
                const float value=k_sums[0][atom][0]+k_sums[1][atom][0];
                shared.routes[route][atom*16+lane_row]=__float2bfloat16(value);
            }
        }
        }
        else if(lane_group==0)
        {
#pragma unroll
            for(int atom=0;atom<Atoms;++atom)
                shared.routes[route][atom*16+lane_row]=__float2bfloat16(0.0f);
        }
        __syncthreads();
    }
    if(wave==0 && lane_group==0)
    {
#pragma unroll
        for(int atom=0;atom<Atoms;++atom)
        {
            float combined=0.0f;
#pragma unroll
            for(int route=0;route<9;++route)
                combined=fmaf(__bfloat162float(shared.routes[route][atom*16+lane_row]),route_weights[token*9+route],combined);
            output[static_cast<int64_t>(token)*N+(n_block+atom)*16+lane_row]=__float2bfloat16(combined);
        }
    }
#endif
}
__global__ __launch_bounds__(512,1) void iq2r_scheduled_gate_wait3_direct_kernel(
    const opus::fp8_t* __restrict__ activations,
    const uint8_t* __restrict__ scales,
    const uint8_t* __restrict__ all_data,
    const uint8_t* __restrict__ all_auxiliary,
    const int32_t* __restrict__ tasks,
    const int32_t* __restrict__ task_count,
    const int32_t* __restrict__ gather,
    opus::fp8_t* __restrict__ output,
    uint8_t* __restrict__ output_scales,
    int routes,int data_bytes,int auxiliary_bytes)
{
#if defined(__gfx950__)
    constexpr int MAtoms=1;
    constexpr int K=6144, Rows=16*MAtoms, Subtasks=1;
    static_assert(MAtoms==1);
    struct Storage {
        alignas(16) uint64_t codebook[kCodebookBytes/8];
        union {
            alignas(16) uint8_t cache[8][kQuadBytes];
            alignas(16) opus::vector_t<float,4> partial[8][4][64];
        } reuse;
    };
    __shared__ Storage shared;
    const int lane=threadIdx.x,wave=threadIdx.y,linear=wave*64+lane;
    const int lane_row=lane%16,lane_group=lane/16;
    const int num_tasks=task_count[0];
    const int total=num_tasks*Subtasks*8;
    for(int work=blockIdx.x;work<total;work+=gridDim.x)
    {
        const int task=(work/Subtasks)%num_tasks;
        const int sub=work%Subtasks;
        const int n_tile=work/(num_tasks*Subtasks);
        const int row_begin=tasks[task*3]+sub*Rows;
        const int row_end=min(row_begin+Rows,tasks[task*3]+tasks[task*3+1]);
        const int expert=tasks[task*3+2];
        if(row_begin>=row_end || row_begin<0 || row_end>routes || expert<0 || expert>=257) continue;
        const uint8_t* data=all_data+static_cast<int64_t>(expert)*data_bytes;
        const uint8_t* aux=all_auxiliary+static_cast<int64_t>(expert)*auxiliary_bytes;
        // Previous work ends with a barrier protecting both union and codebook.
        shared.codebook[linear]=reinterpret_cast<const uint64_t*>(aux)[linear];
        __syncthreads();
        const int nbase=n_tile*4;
        uint32_t bases=0;
#pragma unroll
        for(int atom=0;atom<4;++atom)
            bases|=static_cast<uint32_t>(aux[kCodebookBytes+nbase+atom])<<(atom*8);
        int rows[MAtoms];
#pragma unroll
        for(int m=0;m<MAtoms;++m) rows[m]=min(row_begin+m*16+lane_row,row_end-1);
        opus::vector_t<float,4> accumulators[MAtoms][4]={};
        opus::gmem<uint8_t> buffer(data,static_cast<unsigned int>(data_bytes));
        int base=(n_tile*48+wave*6)*kQuadBytes;
        iq2r_issue_quad(buffer,shared.reuse.cache[wave],lane,base);
        for(int iteration=0;iteration<6;++iteration)
        {
            const int kt=wave*6+iteration;
            IQ2RActivationFragment a[MAtoms]={};uint32_t sa[MAtoms];
#pragma unroll
            for(int m=0;m<MAtoms;++m)
            {
                const auto* input=reinterpret_cast<const uint8_t*>(activations+static_cast<int64_t>(rows[m])*K);
                const int ak=kt*128+lane_group*16;
                *reinterpret_cast<uint4*>(a[m].bytes)=*reinterpret_cast<const uint4*>(input+ak);
                *reinterpret_cast<uint4*>(a[m].bytes+16)=*reinterpret_cast<const uint4*>(input+ak+64);
                sa[m]=scales[static_cast<int64_t>(rows[m])*192+kt*4+lane_group]*0x01010101u;
            }
            iq2r_wait_vmcnt<3>();
            const auto compressed=iq2r_read_quad(shared.reuse.cache[wave],lane);
            asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
            if(iteration<5)
            {
                base+=kQuadBytes;
                iq2r_issue_quad(buffer,shared.reuse.cache[wave],lane,base);
            }
            iq2r_quad_decode_mfma<MAtoms>(compressed,shared.codebook,bases,a,sa,accumulators);
        }
        // No wave may overwrite another wave's cache before its final read.
        __syncthreads();
#pragma unroll
        for(int m=0;m<MAtoms;++m)
        {
#pragma unroll
            for(int atom=0;atom<4;++atom)
                shared.reuse.partial[wave][atom][lane]=accumulators[m][atom];
            __syncthreads();
            if(wave<4)
            {
                const int local_row=m*16+lane_group*4+wave;
                const int output_row=row_begin+local_row;
                float values[4];
#pragma unroll
                for(int atom=0;atom<4;++atom)
                {
                    float value=0.0f;
#pragma unroll
                    for(int source=0;source<8;++source)
                        value+=shared.reuse.partial[source][atom][lane][wave];
                    values[atom]=__bfloat162float(__float2bfloat16(value));
                }
                // Adjacent lanes hold the interleaved gate and up columns.
                // Read the odd lane before restricting execution to even lanes.
                float up[4];
#pragma unroll
                for(int atom=0;atom<4;++atom) up[atom]=__shfl_xor(values[atom],1);
                if(lane_row%2==0)
                {
                    float activated[4];float abs_max=1.0e-10f;
#pragma unroll
                    for(int atom=0;atom<4;++atom)
                    {
                        const float swish=values[atom]/(1.0f+__expf(-values[atom]));
                        activated[atom]=__bfloat162float(__float2bfloat16(swish*(up[atom]+0.0f)));
                        abs_max=fmaxf(abs_max,fabsf(activated[atom]));
                    }
                    abs_max=fmaxf(abs_max,__shfl_xor(abs_max,2));
                    abs_max=fmaxf(abs_max,__shfl_xor(abs_max,4));
                    abs_max=fmaxf(abs_max,__shfl_xor(abs_max,8));
                    if(output_row<row_end)
                    {
                        const auto bs=fp_f32_to_e8m0_block_scale<kDefaultMxScaleRoundMode,MxDtype::FP8_E4M3>(abs_max);
                        const float inverse=1.0f/bs.dq_scale;
#pragma unroll
                        for(int atom=0;atom<4;++atom)
                            output[static_cast<int64_t>(output_row)*256+n_tile*32+atom*8+lane_row/2]=opus::fp32_to_fp8(activated[atom]*inverse);
                        if(lane_row==0) output_scales[static_cast<int64_t>(output_row)*8+n_tile]=bs.byte;
                    }
                }
            }
            __syncthreads();
        }
    }
#endif
}

__global__ __launch_bounds__(512,1) void iq2r_scheduled_gate_register_unroll6_direct_kernel(
    const opus::fp8_t* __restrict__ activations,
    const uint8_t* __restrict__ scales,
    const uint8_t* __restrict__ all_data,
    const uint8_t* __restrict__ all_auxiliary,
    const int32_t* __restrict__ tasks,
    const int32_t* __restrict__ task_count,
    const int32_t* __restrict__ gather,
    opus::fp8_t* __restrict__ output,
    uint8_t* __restrict__ output_scales,
    int routes,int data_bytes,int auxiliary_bytes)
{
#if defined(__gfx950__)
    constexpr int MAtoms=1;
    constexpr int K=6144, Rows=16*MAtoms, Subtasks=1;
    static_assert(MAtoms==1);
    struct Storage {
        alignas(16) uint64_t codebook[kCodebookBytes/8];
        union {
            alignas(16) uint8_t cache[8][kQuadBytes];
            alignas(16) opus::vector_t<float,4> partial[8][4][64];
        } reuse;
    };
    __shared__ Storage shared;
    const int lane=threadIdx.x,wave=threadIdx.y,linear=wave*64+lane;
    const int lane_row=lane%16,lane_group=lane/16;
    const int num_tasks=task_count[0];
    const int total=num_tasks*Subtasks*8;
    for(int work=blockIdx.x;work<total;work+=gridDim.x)
    {
        const int task=(work/Subtasks)%num_tasks;
        const int sub=work%Subtasks;
        const int n_tile=work/(num_tasks*Subtasks);
        const int row_begin=tasks[task*3]+sub*Rows;
        const int row_end=min(row_begin+Rows,tasks[task*3]+tasks[task*3+1]);
        const int expert=tasks[task*3+2];
        if(row_begin>=row_end || row_begin<0 || row_end>routes || expert<0 || expert>=257) continue;
        const uint8_t* data=all_data+static_cast<int64_t>(expert)*data_bytes;
        const uint8_t* aux=all_auxiliary+static_cast<int64_t>(expert)*auxiliary_bytes;
        // Previous work ends with a barrier protecting both union and codebook.
        shared.codebook[linear]=reinterpret_cast<const uint64_t*>(aux)[linear];
        __syncthreads();
        const int nbase=n_tile*4;
        uint32_t bases=0;
#pragma unroll
        for(int atom=0;atom<4;++atom)
            bases|=static_cast<uint32_t>(aux[kCodebookBytes+nbase+atom])<<(atom*8);
        int rows[MAtoms];
#pragma unroll
        for(int m=0;m<MAtoms;++m) rows[m]=min(row_begin+m*16+lane_row,row_end-1);
        opus::vector_t<float,4> accumulators[MAtoms][4]={};
        opus::gmem<uint8_t> buffer(data,static_cast<unsigned int>(data_bytes));
        int base=(n_tile*48+wave*6)*kQuadBytes;
        IQ2RCompressedQuad pending=iq2r_scheduled_load_quad(data,data_bytes,base,lane);
#pragma unroll 6
        for(int iteration=0;iteration<6;++iteration)
        {
            const int kt=wave*6+iteration;
            IQ2RActivationFragment a[MAtoms]={};uint32_t sa[MAtoms];
#pragma unroll
            for(int m=0;m<MAtoms;++m)
            {
                const auto* input=reinterpret_cast<const uint8_t*>(activations+static_cast<int64_t>(rows[m])*K);
                const int ak=kt*128+lane_group*16;
                *reinterpret_cast<uint4*>(a[m].bytes)=*reinterpret_cast<const uint4*>(input+ak);
                *reinterpret_cast<uint4*>(a[m].bytes+16)=*reinterpret_cast<const uint4*>(input+ak+64);
                sa[m]=scales[static_cast<int64_t>(rows[m])*192+kt*4+lane_group]*0x01010101u;
            }
            const auto compressed=pending;
            if(iteration<5)
            {
                base+=kQuadBytes;
                pending=iq2r_scheduled_load_quad(data,data_bytes,base,lane);
            }
            iq2r_quad_decode_mfma<MAtoms>(compressed,shared.codebook,bases,a,sa,accumulators);
        }
        // No wave may overwrite another wave's cache before its final read.
        __syncthreads();
#pragma unroll
        for(int m=0;m<MAtoms;++m)
        {
#pragma unroll
            for(int atom=0;atom<4;++atom)
                shared.reuse.partial[wave][atom][lane]=accumulators[m][atom];
            __syncthreads();
            if(wave<4)
            {
                const int local_row=m*16+lane_group*4+wave;
                const int output_row=row_begin+local_row;
                float values[4];
#pragma unroll
                for(int atom=0;atom<4;++atom)
                {
                    float value=0.0f;
#pragma unroll
                    for(int source=0;source<8;++source)
                        value+=shared.reuse.partial[source][atom][lane][wave];
                    values[atom]=__bfloat162float(__float2bfloat16(value));
                }
                // Adjacent lanes hold the interleaved gate and up columns.
                // Read the odd lane before restricting execution to even lanes.
                float up[4];
#pragma unroll
                for(int atom=0;atom<4;++atom) up[atom]=__shfl_xor(values[atom],1);
                if(lane_row%2==0)
                {
                    float activated[4];float abs_max=1.0e-10f;
#pragma unroll
                    for(int atom=0;atom<4;++atom)
                    {
                        const float swish=values[atom]/(1.0f+__expf(-values[atom]));
                        activated[atom]=__bfloat162float(__float2bfloat16(swish*(up[atom]+0.0f)));
                        abs_max=fmaxf(abs_max,fabsf(activated[atom]));
                    }
                    abs_max=fmaxf(abs_max,__shfl_xor(abs_max,2));
                    abs_max=fmaxf(abs_max,__shfl_xor(abs_max,4));
                    abs_max=fmaxf(abs_max,__shfl_xor(abs_max,8));
                    if(output_row<row_end)
                    {
                        const auto bs=fp_f32_to_e8m0_block_scale<kDefaultMxScaleRoundMode,MxDtype::FP8_E4M3>(abs_max);
                        const float inverse=1.0f/bs.dq_scale;
#pragma unroll
                        for(int atom=0;atom<4;++atom)
                            output[static_cast<int64_t>(output_row)*256+n_tile*32+atom*8+lane_row/2]=opus::fp32_to_fp8(activated[atom]*inverse);
                        if(lane_row==0) output_scales[static_cast<int64_t>(output_row)*8+n_tile]=bs.byte;
                    }
                }
            }
            __syncthreads();
        }
    }
#endif
}
template<int MAtoms,int Batch,bool Prepared=false>
__device__ __forceinline__ void e117_decode(
    const IQ2RCompressedQuad& compressed,const uint64_t* codebook,uint32_t bases,
    const IQ2RActivationFragment* activation,const uint32_t* scale_a,
    opus::vector_t<float,4> (&accumulators)[MAtoms][4],
    const uint32_t* prepared_highs=nullptr,uint32_t prepared_scales=0)
{
#pragma unroll
    for(int atom=0;atom<4;++atom)
    {
        const uint32_t lows=atom==0?compressed.first.x:atom==1?compressed.first.z:atom==2?compressed.second.x:compressed.second.z;
        const uint32_t signs=atom==0?compressed.first.y:atom==1?compressed.first.w:atom==2?compressed.second.y:compressed.second.w;
        const uint32_t highs=Prepared?prepared_highs[atom]:(compressed.metadata>>(atom*8))&15u;
        union { opus::i32x8_t words;uint64_t codewords[4]; } decoded;
        if constexpr(Batch==0)
        {
#pragma unroll
            for(int word=0;word<4;++word)
            {
                const int index=((lows>>(word*8))&255u)|(((highs>>word)&1u)<<8);
                decoded.codewords[word]=apply_signs(codebook[index],(signs>>(word*8))&255u);
            }
        }
        else
        {
            uint64_t magnitudes[4];
#pragma unroll
            for(int word=0;word<4;++word)
            {
                const int index=((lows>>(word*8))&255u)|(((highs>>word)&1u)<<8);
                magnitudes[word]=codebook[index];
            }
            // Batch=1 is the source-level Redline ordering. Batch=2 forces
            // all four independent results to be live before applying signs.
            if constexpr(Batch==2)
                asm volatile("" : "+v"(magnitudes[0]), "+v"(magnitudes[1]),
                                  "+v"(magnitudes[2]), "+v"(magnitudes[3]));
#pragma unroll
            for(int word=0;word<4;++word)
                decoded.codewords[word]=apply_signs(magnitudes[word],(signs>>(word*8))&255u);
        }
        const uint32_t sb=Prepared?((prepared_scales>>(atom*8))&255u):((bases>>(atom*8))&255u)+((compressed.metadata>>(atom*8+4))&15u);
#pragma unroll
        for(int m=0;m<MAtoms;++m)
        {
            auto mma=opus::mfma<opus::fp8_t,opus::fp8_t,opus::fp32_t,16,16,128>{};
            accumulators[m][atom]=mma(activation[m].words,decoded.words,accumulators[m][atom],
                                      scale_a[m],sb,opus::number<0>{},opus::number<0>{});
        }
    }
}



template<int MAtoms>
__device__ __forceinline__ void e117_wide2_forced(
    const IQ2RCompressedQuad& compressed,const uint64_t* codebook,uint32_t bases,
    const IQ2RActivationFragment* activation,const uint32_t* scale_a,
    opus::vector_t<float,4> (&accumulators)[MAtoms][4])
{
#pragma unroll
    for(int group=0;group<4;group+=2)
    {
        uint64_t magnitudes[2][4];
        uint32_t signs[2];
#pragma unroll
        for(int local=0;local<2;++local)
        {
            const int atom=group+local;
            const uint32_t lows=atom==0?compressed.first.x:atom==1?compressed.first.z:atom==2?compressed.second.x:compressed.second.z;
            signs[local]=atom==0?compressed.first.y:atom==1?compressed.first.w:atom==2?compressed.second.y:compressed.second.w;
            const uint32_t highs=(compressed.metadata>>(atom*8))&15u;
#pragma unroll
            for(int word=0;word<4;++word)
            {
                const int index=((lows>>(word*8))&255u)|(((highs>>word)&1u)<<8);
                magnitudes[local][word]=codebook[index];
            }
        }
        asm volatile("" : "+v"(magnitudes[0][0]), "+v"(magnitudes[0][1]), "+v"(magnitudes[0][2]), "+v"(magnitudes[0][3]), "+v"(magnitudes[1][0]), "+v"(magnitudes[1][1]), "+v"(magnitudes[1][2]), "+v"(magnitudes[1][3]));
#pragma unroll
        for(int local=0;local<2;++local)
        {
            const int atom=group+local;
            union {opus::i32x8_t words;uint64_t codewords[4];} decoded;
#pragma unroll
            for(int word=0;word<4;++word)
                decoded.codewords[word]=apply_signs(magnitudes[local][word],(signs[local]>>(word*8))&255u);
            const uint32_t sb=((bases>>(atom*8))&255u)+((compressed.metadata>>(atom*8+4))&15u);
#pragma unroll
            for(int m=0;m<MAtoms;++m)
            {
                auto mma=opus::mfma<opus::fp8_t,opus::fp8_t,opus::fp32_t,16,16,128>{};
                accumulators[m][atom]=mma(activation[m].words,decoded.words,accumulators[m][atom],scale_a[m],sb,opus::number<0>{},opus::number<0>{});
            }
        }
    }
}



__global__ __launch_bounds__(512,1) void iq2r_scheduled_gate_wait3_direct_batch_kernel(
    const opus::fp8_t* __restrict__ activations,
    const uint8_t* __restrict__ scales,
    const uint8_t* __restrict__ all_data,
    const uint8_t* __restrict__ all_auxiliary,
    const int32_t* __restrict__ tasks,
    const int32_t* __restrict__ task_count,
    const int32_t* __restrict__ gather,
    opus::fp8_t* __restrict__ output,
    uint8_t* __restrict__ output_scales,
    int routes,int data_bytes,int auxiliary_bytes)
{
#if defined(__gfx950__)
    constexpr int MAtoms=1;
    constexpr int K=6144, Rows=16*MAtoms, Subtasks=1;
    static_assert(MAtoms==1);
    struct Storage {
        alignas(16) uint64_t codebook[kCodebookBytes/8];
        union {
            alignas(16) uint8_t cache[8][kQuadBytes];
            alignas(16) opus::vector_t<float,4> partial[8][4][64];
        } reuse;
    };
    __shared__ Storage shared;
    const int lane=threadIdx.x,wave=threadIdx.y,linear=wave*64+lane;
    const int lane_row=lane%16,lane_group=lane/16;
    const int num_tasks=task_count[0];
    const int total=num_tasks*Subtasks*8;
    for(int work=blockIdx.x;work<total;work+=gridDim.x)
    {
        const int task=(work/Subtasks)%num_tasks;
        const int sub=work%Subtasks;
        const int n_tile=work/(num_tasks*Subtasks);
        const int row_begin=tasks[task*3]+sub*Rows;
        const int row_end=min(row_begin+Rows,tasks[task*3]+tasks[task*3+1]);
        const int expert=tasks[task*3+2];
        if(row_begin>=row_end || row_begin<0 || row_end>routes || expert<0 || expert>=257) continue;
        const uint8_t* data=all_data+static_cast<int64_t>(expert)*data_bytes;
        const uint8_t* aux=all_auxiliary+static_cast<int64_t>(expert)*auxiliary_bytes;
        // Previous work ends with a barrier protecting both union and codebook.
        shared.codebook[linear]=reinterpret_cast<const uint64_t*>(aux)[linear];
        __syncthreads();
        const int nbase=n_tile*4;
        uint32_t bases=0;
#pragma unroll
        for(int atom=0;atom<4;++atom)
            bases|=static_cast<uint32_t>(aux[kCodebookBytes+nbase+atom])<<(atom*8);
        int rows[MAtoms];
#pragma unroll
        for(int m=0;m<MAtoms;++m) rows[m]=min(row_begin+m*16+lane_row,row_end-1);
        opus::vector_t<float,4> accumulators[MAtoms][4]={};
        opus::gmem<uint8_t> buffer(data,static_cast<unsigned int>(data_bytes));
        int base=(n_tile*48+wave*6)*kQuadBytes;
        iq2r_issue_quad(buffer,shared.reuse.cache[wave],lane,base);
        for(int iteration=0;iteration<6;++iteration)
        {
            const int kt=wave*6+iteration;
            IQ2RActivationFragment a[MAtoms]={};uint32_t sa[MAtoms];
#pragma unroll
            for(int m=0;m<MAtoms;++m)
            {
                const auto* input=reinterpret_cast<const uint8_t*>(activations+static_cast<int64_t>(rows[m])*K);
                const int ak=kt*128+lane_group*16;
                *reinterpret_cast<uint4*>(a[m].bytes)=*reinterpret_cast<const uint4*>(input+ak);
                *reinterpret_cast<uint4*>(a[m].bytes+16)=*reinterpret_cast<const uint4*>(input+ak+64);
                sa[m]=scales[static_cast<int64_t>(rows[m])*192+kt*4+lane_group]*0x01010101u;
            }
            iq2r_wait_vmcnt<3>();
            const auto compressed=iq2r_read_quad(shared.reuse.cache[wave],lane);
            asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
            if(iteration<5)
            {
                base+=kQuadBytes;
                iq2r_issue_quad(buffer,shared.reuse.cache[wave],lane,base);
            }
            e117_decode<MAtoms,2>(compressed,shared.codebook,bases,a,sa,accumulators);
        }
        // No wave may overwrite another wave's cache before its final read.
        __syncthreads();
#pragma unroll
        for(int m=0;m<MAtoms;++m)
        {
#pragma unroll
            for(int atom=0;atom<4;++atom)
                shared.reuse.partial[wave][atom][lane]=accumulators[m][atom];
            __syncthreads();
            if(wave<4)
            {
                const int local_row=m*16+lane_group*4+wave;
                const int output_row=row_begin+local_row;
                float values[4];
#pragma unroll
                for(int atom=0;atom<4;++atom)
                {
                    float value=0.0f;
#pragma unroll
                    for(int source=0;source<8;++source)
                        value+=shared.reuse.partial[source][atom][lane][wave];
                    values[atom]=__bfloat162float(__float2bfloat16(value));
                }
                // Adjacent lanes hold the interleaved gate and up columns.
                // Read the odd lane before restricting execution to even lanes.
                float up[4];
#pragma unroll
                for(int atom=0;atom<4;++atom) up[atom]=__shfl_xor(values[atom],1);
                if(lane_row%2==0)
                {
                    float activated[4];float abs_max=1.0e-10f;
#pragma unroll
                    for(int atom=0;atom<4;++atom)
                    {
                        const float swish=values[atom]/(1.0f+__expf(-values[atom]));
                        activated[atom]=__bfloat162float(__float2bfloat16(swish*(up[atom]+0.0f)));
                        abs_max=fmaxf(abs_max,fabsf(activated[atom]));
                    }
                    abs_max=fmaxf(abs_max,__shfl_xor(abs_max,2));
                    abs_max=fmaxf(abs_max,__shfl_xor(abs_max,4));
                    abs_max=fmaxf(abs_max,__shfl_xor(abs_max,8));
                    if(output_row<row_end)
                    {
                        const auto bs=fp_f32_to_e8m0_block_scale<kDefaultMxScaleRoundMode,MxDtype::FP8_E4M3>(abs_max);
                        const float inverse=1.0f/bs.dq_scale;
#pragma unroll
                        for(int atom=0;atom<4;++atom)
                            output[static_cast<int64_t>(output_row)*256+n_tile*32+atom*8+lane_row/2]=opus::fp32_to_fp8(activated[atom]*inverse);
                        if(lane_row==0) output_scales[static_cast<int64_t>(output_row)*8+n_tile]=bs.byte;
                    }
                }
            }
            __syncthreads();
        }
    }
#endif
}

__global__ __launch_bounds__(512,1) void iq2r_scheduled_gate_register_unroll6_direct_batch_kernel(
    const opus::fp8_t* __restrict__ activations,
    const uint8_t* __restrict__ scales,
    const uint8_t* __restrict__ all_data,
    const uint8_t* __restrict__ all_auxiliary,
    const int32_t* __restrict__ tasks,
    const int32_t* __restrict__ task_count,
    const int32_t* __restrict__ gather,
    opus::fp8_t* __restrict__ output,
    uint8_t* __restrict__ output_scales,
    int routes,int data_bytes,int auxiliary_bytes)
{
#if defined(__gfx950__)
    constexpr int MAtoms=1;
    constexpr int K=6144, Rows=16*MAtoms, Subtasks=1;
    static_assert(MAtoms==1);
    struct Storage {
        alignas(16) uint64_t codebook[kCodebookBytes/8];
        union {
            alignas(16) uint8_t cache[8][kQuadBytes];
            alignas(16) opus::vector_t<float,4> partial[8][4][64];
        } reuse;
    };
    __shared__ Storage shared;
    const int lane=threadIdx.x,wave=threadIdx.y,linear=wave*64+lane;
    const int lane_row=lane%16,lane_group=lane/16;
    const int num_tasks=task_count[0];
    const int total=num_tasks*Subtasks*8;
    for(int work=blockIdx.x;work<total;work+=gridDim.x)
    {
        const int task=(work/Subtasks)%num_tasks;
        const int sub=work%Subtasks;
        const int n_tile=work/(num_tasks*Subtasks);
        const int row_begin=tasks[task*3]+sub*Rows;
        const int row_end=min(row_begin+Rows,tasks[task*3]+tasks[task*3+1]);
        const int expert=tasks[task*3+2];
        if(row_begin>=row_end || row_begin<0 || row_end>routes || expert<0 || expert>=257) continue;
        const uint8_t* data=all_data+static_cast<int64_t>(expert)*data_bytes;
        const uint8_t* aux=all_auxiliary+static_cast<int64_t>(expert)*auxiliary_bytes;
        // Previous work ends with a barrier protecting both union and codebook.
        shared.codebook[linear]=reinterpret_cast<const uint64_t*>(aux)[linear];
        __syncthreads();
        const int nbase=n_tile*4;
        uint32_t bases=0;
#pragma unroll
        for(int atom=0;atom<4;++atom)
            bases|=static_cast<uint32_t>(aux[kCodebookBytes+nbase+atom])<<(atom*8);
        int rows[MAtoms];
#pragma unroll
        for(int m=0;m<MAtoms;++m) rows[m]=min(row_begin+m*16+lane_row,row_end-1);
        opus::vector_t<float,4> accumulators[MAtoms][4]={};
        opus::gmem<uint8_t> buffer(data,static_cast<unsigned int>(data_bytes));
        int base=(n_tile*48+wave*6)*kQuadBytes;
        IQ2RCompressedQuad pending=iq2r_scheduled_load_quad(data,data_bytes,base,lane);
#pragma unroll 6
        for(int iteration=0;iteration<6;++iteration)
        {
            const int kt=wave*6+iteration;
            IQ2RActivationFragment a[MAtoms]={};uint32_t sa[MAtoms];
#pragma unroll
            for(int m=0;m<MAtoms;++m)
            {
                const auto* input=reinterpret_cast<const uint8_t*>(activations+static_cast<int64_t>(rows[m])*K);
                const int ak=kt*128+lane_group*16;
                *reinterpret_cast<uint4*>(a[m].bytes)=*reinterpret_cast<const uint4*>(input+ak);
                *reinterpret_cast<uint4*>(a[m].bytes+16)=*reinterpret_cast<const uint4*>(input+ak+64);
                sa[m]=scales[static_cast<int64_t>(rows[m])*192+kt*4+lane_group]*0x01010101u;
            }
            const auto compressed=pending;
            if(iteration<5)
            {
                base+=kQuadBytes;
                pending=iq2r_scheduled_load_quad(data,data_bytes,base,lane);
            }
            e117_wide2_forced<MAtoms>(compressed,shared.codebook,bases,a,sa,accumulators);
        }
        // No wave may overwrite another wave's cache before its final read.
        __syncthreads();
#pragma unroll
        for(int m=0;m<MAtoms;++m)
        {
#pragma unroll
            for(int atom=0;atom<4;++atom)
                shared.reuse.partial[wave][atom][lane]=accumulators[m][atom];
            __syncthreads();
            if(wave<4)
            {
                const int local_row=m*16+lane_group*4+wave;
                const int output_row=row_begin+local_row;
                float values[4];
#pragma unroll
                for(int atom=0;atom<4;++atom)
                {
                    float value=0.0f;
#pragma unroll
                    for(int source=0;source<8;++source)
                        value+=shared.reuse.partial[source][atom][lane][wave];
                    values[atom]=__bfloat162float(__float2bfloat16(value));
                }
                // Adjacent lanes hold the interleaved gate and up columns.
                // Read the odd lane before restricting execution to even lanes.
                float up[4];
#pragma unroll
                for(int atom=0;atom<4;++atom) up[atom]=__shfl_xor(values[atom],1);
                if(lane_row%2==0)
                {
                    float activated[4];float abs_max=1.0e-10f;
#pragma unroll
                    for(int atom=0;atom<4;++atom)
                    {
                        const float swish=values[atom]/(1.0f+__expf(-values[atom]));
                        activated[atom]=__bfloat162float(__float2bfloat16(swish*(up[atom]+0.0f)));
                        abs_max=fmaxf(abs_max,fabsf(activated[atom]));
                    }
                    abs_max=fmaxf(abs_max,__shfl_xor(abs_max,2));
                    abs_max=fmaxf(abs_max,__shfl_xor(abs_max,4));
                    abs_max=fmaxf(abs_max,__shfl_xor(abs_max,8));
                    if(output_row<row_end)
                    {
                        const auto bs=fp_f32_to_e8m0_block_scale<kDefaultMxScaleRoundMode,MxDtype::FP8_E4M3>(abs_max);
                        const float inverse=1.0f/bs.dq_scale;
#pragma unroll
                        for(int atom=0;atom<4;++atom)
                            output[static_cast<int64_t>(output_row)*256+n_tile*32+atom*8+lane_row/2]=opus::fp32_to_fp8(activated[atom]*inverse);
                        if(lane_row==0) output_scales[static_cast<int64_t>(output_row)*8+n_tile]=bs.byte;
                    }
                }
            }
            __syncthreads();
        }
    }
#endif
}
__global__ __launch_bounds__(512,1) void iq2r_scheduled_gate_wait3_kernel_tp4(
    const opus::fp8_t* __restrict__ activations,
    const uint8_t* __restrict__ scales,
    const uint8_t* __restrict__ all_data,
    const uint8_t* __restrict__ all_auxiliary,
    const int32_t* __restrict__ tasks,
    const int32_t* __restrict__ task_count,
    const int32_t* __restrict__ gather,
    opus::fp8_t* __restrict__ output,
    uint8_t* __restrict__ output_scales,
    int routes,int data_bytes,int auxiliary_bytes)
{
#if defined(__gfx950__)
    constexpr int MAtoms=1;
    constexpr int K=6144, Rows=16*MAtoms, Subtasks=1;
    static_assert(MAtoms==1);
    struct Storage {
        alignas(16) uint64_t codebook[kCodebookBytes/8];
        union {
            alignas(16) uint8_t cache[8][kQuadBytes];
            alignas(16) opus::vector_t<float,4> partial[8][4][64];
        } reuse;
        alignas(16) __hip_bfloat16 gate[Rows][64];
    };
    __shared__ Storage shared;
    const int lane=threadIdx.x,wave=threadIdx.y,linear=wave*64+lane;
    const int lane_row=lane%16,lane_group=lane/16;
    const int num_tasks=task_count[0];
    const int total=num_tasks*Subtasks*16;
    for(int work=blockIdx.x;work<total;work+=gridDim.x)
    {
        const int task=(work/Subtasks)%num_tasks;
        const int sub=work%Subtasks;
        const int n_tile=work/(num_tasks*Subtasks);
        const int row_begin=tasks[task*3]+sub*Rows;
        const int row_end=min(row_begin+Rows,tasks[task*3]+tasks[task*3+1]);
        const int expert=tasks[task*3+2];
        if(row_begin>=row_end || row_begin<0 || row_end>routes || expert<0 || expert>=257) continue;
        const uint8_t* data=all_data+static_cast<int64_t>(expert)*data_bytes;
        const uint8_t* aux=all_auxiliary+static_cast<int64_t>(expert)*auxiliary_bytes;
        // Previous work ends with a barrier protecting both union and codebook.
        shared.codebook[linear]=reinterpret_cast<const uint64_t*>(aux)[linear];
        __syncthreads();
        const int nbase=n_tile*4;
        uint32_t bases=0;
#pragma unroll
        for(int atom=0;atom<4;++atom)
            bases|=static_cast<uint32_t>(aux[kCodebookBytes+nbase+atom])<<(atom*8);
        int rows[MAtoms];
#pragma unroll
        for(int m=0;m<MAtoms;++m) rows[m]=min(row_begin+m*16+lane_row,row_end-1);
        opus::vector_t<float,4> accumulators[MAtoms][4]={};
        opus::gmem<uint8_t> buffer(data,static_cast<unsigned int>(data_bytes));
        int base=(n_tile*48+wave*6)*kQuadBytes;
        iq2r_issue_quad(buffer,shared.reuse.cache[wave],lane,base);
        for(int iteration=0;iteration<6;++iteration)
        {
            const int kt=wave*6+iteration;
            IQ2RActivationFragment a[MAtoms]={};uint32_t sa[MAtoms];
#pragma unroll
            for(int m=0;m<MAtoms;++m)
            {
                const auto* input=reinterpret_cast<const uint8_t*>(activations+static_cast<int64_t>(rows[m])*K);
                const int ak=kt*128+lane_group*16;
                *reinterpret_cast<uint4*>(a[m].bytes)=*reinterpret_cast<const uint4*>(input+ak);
                *reinterpret_cast<uint4*>(a[m].bytes+16)=*reinterpret_cast<const uint4*>(input+ak+64);
                sa[m]=scales[static_cast<int64_t>(rows[m])*192+kt*4+lane_group]*0x01010101u;
            }
            iq2r_wait_vmcnt<3>();
            const auto compressed=iq2r_read_quad(shared.reuse.cache[wave],lane);
            asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
            if(iteration<5)
            {
                base+=kQuadBytes;
                iq2r_issue_quad(buffer,shared.reuse.cache[wave],lane,base);
            }
            iq2r_quad_decode_mfma<MAtoms>(compressed,shared.codebook,bases,a,sa,accumulators);
        }
        // No wave may overwrite another wave's cache before its final read.
        __syncthreads();
#pragma unroll
        for(int m=0;m<MAtoms;++m)
        {
#pragma unroll
            for(int atom=0;atom<4;++atom) shared.reuse.partial[wave][atom][lane]=accumulators[m][atom];
            __syncthreads();
            if(wave<4)
            {
                const int local_row=m*16+lane_group*4+wave;
#pragma unroll
                for(int atom=0;atom<4;++atom)
                {
                    float value=0.0f;
#pragma unroll
                    for(int source=0;source<8;++source) value+=shared.reuse.partial[source][atom][lane][wave];
                    shared.gate[local_row][atom*16+lane_row]=__float2bfloat16(value);
                }
            }
            __syncthreads();
        }
        if(linear<Rows*4)
        {
            const int local_row=linear/4,quant_lane=linear%4;
            const int output_row=row_begin+local_row;
            using B=opus::vector_t<opus::bf16_t,8>;
            using Q=opus::vector_t<opus::fp8_t,8>;
            B values;float abs_max=1.0e-10f;
#pragma unroll
            for(int element=0;element<8;++element)
            {
                const int column=quant_lane*8+element;
                const float gate=__bfloat162float(shared.gate[local_row][2*column]);
                const float up=__bfloat162float(shared.gate[local_row][2*column+1]);
                const float swish=gate/(1.0f+__expf(-gate));
                const __hip_bfloat16 rounded=__float2bfloat16(swish*(up+0.0f));
                values[element]=__builtin_bit_cast(opus::bf16_t,rounded);
                abs_max=fmaxf(abs_max,fabsf(__bfloat162float(rounded)));
            }
            abs_max=fmaxf(abs_max,__shfl_xor(abs_max,1));
            abs_max=fmaxf(abs_max,__shfl_xor(abs_max,2));
            if(output_row<row_end)
            {
                const auto bs=fp_f32_to_e8m0_block_scale<kDefaultMxScaleRoundMode,MxDtype::FP8_E4M3>(abs_max);
                const float inverse=1.0f/bs.dq_scale;
                Q q;
#pragma unroll
                for(int element=0;element<8;++element) q[element]=opus::fp32_to_fp8(static_cast<float>(values[element])*inverse);
                *reinterpret_cast<Q*>(output+static_cast<int64_t>(output_row)*512+n_tile*32+quant_lane*8)=q;
                if(quant_lane==0) output_scales[static_cast<int64_t>(output_row)*16+n_tile]=bs.byte;
            }
        }
        __syncthreads();
    }
#endif
}

__global__ __launch_bounds__(512,1) void iq2r_scheduled_gate_register_unroll6_kernel_tp4(
    const opus::fp8_t* __restrict__ activations,
    const uint8_t* __restrict__ scales,
    const uint8_t* __restrict__ all_data,
    const uint8_t* __restrict__ all_auxiliary,
    const int32_t* __restrict__ tasks,
    const int32_t* __restrict__ task_count,
    const int32_t* __restrict__ gather,
    opus::fp8_t* __restrict__ output,
    uint8_t* __restrict__ output_scales,
    int routes,int data_bytes,int auxiliary_bytes)
{
#if defined(__gfx950__)
    constexpr int MAtoms=1;
    constexpr int K=6144, Rows=16*MAtoms, Subtasks=1;
    static_assert(MAtoms==1);
    struct Storage {
        alignas(16) uint64_t codebook[kCodebookBytes/8];
        union {
            alignas(16) uint8_t cache[8][kQuadBytes];
            alignas(16) opus::vector_t<float,4> partial[8][4][64];
        } reuse;
        alignas(16) __hip_bfloat16 gate[Rows][64];
    };
    __shared__ Storage shared;
    const int lane=threadIdx.x,wave=threadIdx.y,linear=wave*64+lane;
    const int lane_row=lane%16,lane_group=lane/16;
    const int num_tasks=task_count[0];
    const int total=num_tasks*Subtasks*16;
    for(int work=blockIdx.x;work<total;work+=gridDim.x)
    {
        const int task=(work/Subtasks)%num_tasks;
        const int sub=work%Subtasks;
        const int n_tile=work/(num_tasks*Subtasks);
        const int row_begin=tasks[task*3]+sub*Rows;
        const int row_end=min(row_begin+Rows,tasks[task*3]+tasks[task*3+1]);
        const int expert=tasks[task*3+2];
        if(row_begin>=row_end || row_begin<0 || row_end>routes || expert<0 || expert>=257) continue;
        const uint8_t* data=all_data+static_cast<int64_t>(expert)*data_bytes;
        const uint8_t* aux=all_auxiliary+static_cast<int64_t>(expert)*auxiliary_bytes;
        // Previous work ends with a barrier protecting both union and codebook.
        shared.codebook[linear]=reinterpret_cast<const uint64_t*>(aux)[linear];
        __syncthreads();
        const int nbase=n_tile*4;
        uint32_t bases=0;
#pragma unroll
        for(int atom=0;atom<4;++atom)
            bases|=static_cast<uint32_t>(aux[kCodebookBytes+nbase+atom])<<(atom*8);
        int rows[MAtoms];
#pragma unroll
        for(int m=0;m<MAtoms;++m) rows[m]=min(row_begin+m*16+lane_row,row_end-1);
        opus::vector_t<float,4> accumulators[MAtoms][4]={};
        opus::gmem<uint8_t> buffer(data,static_cast<unsigned int>(data_bytes));
        int base=(n_tile*48+wave*6)*kQuadBytes;
        IQ2RCompressedQuad pending=iq2r_scheduled_load_quad(data,data_bytes,base,lane);
#pragma unroll 6
        for(int iteration=0;iteration<6;++iteration)
        {
            const int kt=wave*6+iteration;
            IQ2RActivationFragment a[MAtoms]={};uint32_t sa[MAtoms];
#pragma unroll
            for(int m=0;m<MAtoms;++m)
            {
                const auto* input=reinterpret_cast<const uint8_t*>(activations+static_cast<int64_t>(rows[m])*K);
                const int ak=kt*128+lane_group*16;
                *reinterpret_cast<uint4*>(a[m].bytes)=*reinterpret_cast<const uint4*>(input+ak);
                *reinterpret_cast<uint4*>(a[m].bytes+16)=*reinterpret_cast<const uint4*>(input+ak+64);
                sa[m]=scales[static_cast<int64_t>(rows[m])*192+kt*4+lane_group]*0x01010101u;
            }
            const auto compressed=pending;
            if(iteration<5)
            {
                base+=kQuadBytes;
                pending=iq2r_scheduled_load_quad(data,data_bytes,base,lane);
            }
            iq2r_quad_decode_mfma<MAtoms>(compressed,shared.codebook,bases,a,sa,accumulators);
        }
        // No wave may overwrite another wave's cache before its final read.
        __syncthreads();
#pragma unroll
        for(int m=0;m<MAtoms;++m)
        {
#pragma unroll
            for(int atom=0;atom<4;++atom) shared.reuse.partial[wave][atom][lane]=accumulators[m][atom];
            __syncthreads();
            if(wave<4)
            {
                const int local_row=m*16+lane_group*4+wave;
#pragma unroll
                for(int atom=0;atom<4;++atom)
                {
                    float value=0.0f;
#pragma unroll
                    for(int source=0;source<8;++source) value+=shared.reuse.partial[source][atom][lane][wave];
                    shared.gate[local_row][atom*16+lane_row]=__float2bfloat16(value);
                }
            }
            __syncthreads();
        }
        if(linear<Rows*4)
        {
            const int local_row=linear/4,quant_lane=linear%4;
            const int output_row=row_begin+local_row;
            using B=opus::vector_t<opus::bf16_t,8>;
            using Q=opus::vector_t<opus::fp8_t,8>;
            B values;float abs_max=1.0e-10f;
#pragma unroll
            for(int element=0;element<8;++element)
            {
                const int column=quant_lane*8+element;
                const float gate=__bfloat162float(shared.gate[local_row][2*column]);
                const float up=__bfloat162float(shared.gate[local_row][2*column+1]);
                const float swish=gate/(1.0f+__expf(-gate));
                const __hip_bfloat16 rounded=__float2bfloat16(swish*(up+0.0f));
                values[element]=__builtin_bit_cast(opus::bf16_t,rounded);
                abs_max=fmaxf(abs_max,fabsf(__bfloat162float(rounded)));
            }
            abs_max=fmaxf(abs_max,__shfl_xor(abs_max,1));
            abs_max=fmaxf(abs_max,__shfl_xor(abs_max,2));
            if(output_row<row_end)
            {
                const auto bs=fp_f32_to_e8m0_block_scale<kDefaultMxScaleRoundMode,MxDtype::FP8_E4M3>(abs_max);
                const float inverse=1.0f/bs.dq_scale;
                Q q;
#pragma unroll
                for(int element=0;element<8;++element) q[element]=opus::fp32_to_fp8(static_cast<float>(values[element])*inverse);
                *reinterpret_cast<Q*>(output+static_cast<int64_t>(output_row)*512+n_tile*32+quant_lane*8)=q;
                if(quant_lane==0) output_scales[static_cast<int64_t>(output_row)*16+n_tile]=bs.byte;
            }
        }
        __syncthreads();
    }
#endif
}

__global__ __launch_bounds__(512,1) void iq2r_scheduled_gate_wait3_direct_kernel_tp4(
    const opus::fp8_t* __restrict__ activations,
    const uint8_t* __restrict__ scales,
    const uint8_t* __restrict__ all_data,
    const uint8_t* __restrict__ all_auxiliary,
    const int32_t* __restrict__ tasks,
    const int32_t* __restrict__ task_count,
    const int32_t* __restrict__ gather,
    opus::fp8_t* __restrict__ output,
    uint8_t* __restrict__ output_scales,
    int routes,int data_bytes,int auxiliary_bytes)
{
#if defined(__gfx950__)
    constexpr int MAtoms=1;
    constexpr int K=6144, Rows=16*MAtoms, Subtasks=1;
    static_assert(MAtoms==1);
    struct Storage {
        alignas(16) uint64_t codebook[kCodebookBytes/8];
        union {
            alignas(16) uint8_t cache[8][kQuadBytes];
            alignas(16) opus::vector_t<float,4> partial[8][4][64];
        } reuse;
    };
    __shared__ Storage shared;
    const int lane=threadIdx.x,wave=threadIdx.y,linear=wave*64+lane;
    const int lane_row=lane%16,lane_group=lane/16;
    const int num_tasks=task_count[0];
    const int total=num_tasks*Subtasks*16;
    for(int work=blockIdx.x;work<total;work+=gridDim.x)
    {
        const int task=(work/Subtasks)%num_tasks;
        const int sub=work%Subtasks;
        const int n_tile=work/(num_tasks*Subtasks);
        const int row_begin=tasks[task*3]+sub*Rows;
        const int row_end=min(row_begin+Rows,tasks[task*3]+tasks[task*3+1]);
        const int expert=tasks[task*3+2];
        if(row_begin>=row_end || row_begin<0 || row_end>routes || expert<0 || expert>=257) continue;
        const uint8_t* data=all_data+static_cast<int64_t>(expert)*data_bytes;
        const uint8_t* aux=all_auxiliary+static_cast<int64_t>(expert)*auxiliary_bytes;
        // Previous work ends with a barrier protecting both union and codebook.
        shared.codebook[linear]=reinterpret_cast<const uint64_t*>(aux)[linear];
        __syncthreads();
        const int nbase=n_tile*4;
        uint32_t bases=0;
#pragma unroll
        for(int atom=0;atom<4;++atom)
            bases|=static_cast<uint32_t>(aux[kCodebookBytes+nbase+atom])<<(atom*8);
        int rows[MAtoms];
#pragma unroll
        for(int m=0;m<MAtoms;++m) rows[m]=min(row_begin+m*16+lane_row,row_end-1);
        opus::vector_t<float,4> accumulators[MAtoms][4]={};
        opus::gmem<uint8_t> buffer(data,static_cast<unsigned int>(data_bytes));
        int base=(n_tile*48+wave*6)*kQuadBytes;
        iq2r_issue_quad(buffer,shared.reuse.cache[wave],lane,base);
        for(int iteration=0;iteration<6;++iteration)
        {
            const int kt=wave*6+iteration;
            IQ2RActivationFragment a[MAtoms]={};uint32_t sa[MAtoms];
#pragma unroll
            for(int m=0;m<MAtoms;++m)
            {
                const auto* input=reinterpret_cast<const uint8_t*>(activations+static_cast<int64_t>(rows[m])*K);
                const int ak=kt*128+lane_group*16;
                *reinterpret_cast<uint4*>(a[m].bytes)=*reinterpret_cast<const uint4*>(input+ak);
                *reinterpret_cast<uint4*>(a[m].bytes+16)=*reinterpret_cast<const uint4*>(input+ak+64);
                sa[m]=scales[static_cast<int64_t>(rows[m])*192+kt*4+lane_group]*0x01010101u;
            }
            iq2r_wait_vmcnt<3>();
            const auto compressed=iq2r_read_quad(shared.reuse.cache[wave],lane);
            asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
            if(iteration<5)
            {
                base+=kQuadBytes;
                iq2r_issue_quad(buffer,shared.reuse.cache[wave],lane,base);
            }
            iq2r_quad_decode_mfma<MAtoms>(compressed,shared.codebook,bases,a,sa,accumulators);
        }
        // No wave may overwrite another wave's cache before its final read.
        __syncthreads();
#pragma unroll
        for(int m=0;m<MAtoms;++m)
        {
#pragma unroll
            for(int atom=0;atom<4;++atom)
                shared.reuse.partial[wave][atom][lane]=accumulators[m][atom];
            __syncthreads();
            if(wave<4)
            {
                const int local_row=m*16+lane_group*4+wave;
                const int output_row=row_begin+local_row;
                float values[4];
#pragma unroll
                for(int atom=0;atom<4;++atom)
                {
                    float value=0.0f;
#pragma unroll
                    for(int source=0;source<8;++source)
                        value+=shared.reuse.partial[source][atom][lane][wave];
                    values[atom]=__bfloat162float(__float2bfloat16(value));
                }
                // Adjacent lanes hold the interleaved gate and up columns.
                // Read the odd lane before restricting execution to even lanes.
                float up[4];
#pragma unroll
                for(int atom=0;atom<4;++atom) up[atom]=__shfl_xor(values[atom],1);
                if(lane_row%2==0)
                {
                    float activated[4];float abs_max=1.0e-10f;
#pragma unroll
                    for(int atom=0;atom<4;++atom)
                    {
                        const float swish=values[atom]/(1.0f+__expf(-values[atom]));
                        activated[atom]=__bfloat162float(__float2bfloat16(swish*(up[atom]+0.0f)));
                        abs_max=fmaxf(abs_max,fabsf(activated[atom]));
                    }
                    abs_max=fmaxf(abs_max,__shfl_xor(abs_max,2));
                    abs_max=fmaxf(abs_max,__shfl_xor(abs_max,4));
                    abs_max=fmaxf(abs_max,__shfl_xor(abs_max,8));
                    if(output_row<row_end)
                    {
                        const auto bs=fp_f32_to_e8m0_block_scale<kDefaultMxScaleRoundMode,MxDtype::FP8_E4M3>(abs_max);
                        const float inverse=1.0f/bs.dq_scale;
#pragma unroll
                        for(int atom=0;atom<4;++atom)
                            output[static_cast<int64_t>(output_row)*512+n_tile*32+atom*8+lane_row/2]=opus::fp32_to_fp8(activated[atom]*inverse);
                        if(lane_row==0) output_scales[static_cast<int64_t>(output_row)*16+n_tile]=bs.byte;
                    }
                }
            }
            __syncthreads();
        }
    }
#endif
}

__global__ __launch_bounds__(512,1) void iq2r_scheduled_gate_register_unroll6_direct_kernel_tp4(
    const opus::fp8_t* __restrict__ activations,
    const uint8_t* __restrict__ scales,
    const uint8_t* __restrict__ all_data,
    const uint8_t* __restrict__ all_auxiliary,
    const int32_t* __restrict__ tasks,
    const int32_t* __restrict__ task_count,
    const int32_t* __restrict__ gather,
    opus::fp8_t* __restrict__ output,
    uint8_t* __restrict__ output_scales,
    int routes,int data_bytes,int auxiliary_bytes)
{
#if defined(__gfx950__)
    constexpr int MAtoms=1;
    constexpr int K=6144, Rows=16*MAtoms, Subtasks=1;
    static_assert(MAtoms==1);
    struct Storage {
        alignas(16) uint64_t codebook[kCodebookBytes/8];
        union {
            alignas(16) uint8_t cache[8][kQuadBytes];
            alignas(16) opus::vector_t<float,4> partial[8][4][64];
        } reuse;
    };
    __shared__ Storage shared;
    const int lane=threadIdx.x,wave=threadIdx.y,linear=wave*64+lane;
    const int lane_row=lane%16,lane_group=lane/16;
    const int num_tasks=task_count[0];
    const int total=num_tasks*Subtasks*16;
    for(int work=blockIdx.x;work<total;work+=gridDim.x)
    {
        const int task=(work/Subtasks)%num_tasks;
        const int sub=work%Subtasks;
        const int n_tile=work/(num_tasks*Subtasks);
        const int row_begin=tasks[task*3]+sub*Rows;
        const int row_end=min(row_begin+Rows,tasks[task*3]+tasks[task*3+1]);
        const int expert=tasks[task*3+2];
        if(row_begin>=row_end || row_begin<0 || row_end>routes || expert<0 || expert>=257) continue;
        const uint8_t* data=all_data+static_cast<int64_t>(expert)*data_bytes;
        const uint8_t* aux=all_auxiliary+static_cast<int64_t>(expert)*auxiliary_bytes;
        // Previous work ends with a barrier protecting both union and codebook.
        shared.codebook[linear]=reinterpret_cast<const uint64_t*>(aux)[linear];
        __syncthreads();
        const int nbase=n_tile*4;
        uint32_t bases=0;
#pragma unroll
        for(int atom=0;atom<4;++atom)
            bases|=static_cast<uint32_t>(aux[kCodebookBytes+nbase+atom])<<(atom*8);
        int rows[MAtoms];
#pragma unroll
        for(int m=0;m<MAtoms;++m) rows[m]=min(row_begin+m*16+lane_row,row_end-1);
        opus::vector_t<float,4> accumulators[MAtoms][4]={};
        opus::gmem<uint8_t> buffer(data,static_cast<unsigned int>(data_bytes));
        int base=(n_tile*48+wave*6)*kQuadBytes;
        IQ2RCompressedQuad pending=iq2r_scheduled_load_quad(data,data_bytes,base,lane);
#pragma unroll 6
        for(int iteration=0;iteration<6;++iteration)
        {
            const int kt=wave*6+iteration;
            IQ2RActivationFragment a[MAtoms]={};uint32_t sa[MAtoms];
#pragma unroll
            for(int m=0;m<MAtoms;++m)
            {
                const auto* input=reinterpret_cast<const uint8_t*>(activations+static_cast<int64_t>(rows[m])*K);
                const int ak=kt*128+lane_group*16;
                *reinterpret_cast<uint4*>(a[m].bytes)=*reinterpret_cast<const uint4*>(input+ak);
                *reinterpret_cast<uint4*>(a[m].bytes+16)=*reinterpret_cast<const uint4*>(input+ak+64);
                sa[m]=scales[static_cast<int64_t>(rows[m])*192+kt*4+lane_group]*0x01010101u;
            }
            const auto compressed=pending;
            if(iteration<5)
            {
                base+=kQuadBytes;
                pending=iq2r_scheduled_load_quad(data,data_bytes,base,lane);
            }
            iq2r_quad_decode_mfma<MAtoms>(compressed,shared.codebook,bases,a,sa,accumulators);
        }
        // No wave may overwrite another wave's cache before its final read.
        __syncthreads();
#pragma unroll
        for(int m=0;m<MAtoms;++m)
        {
#pragma unroll
            for(int atom=0;atom<4;++atom)
                shared.reuse.partial[wave][atom][lane]=accumulators[m][atom];
            __syncthreads();
            if(wave<4)
            {
                const int local_row=m*16+lane_group*4+wave;
                const int output_row=row_begin+local_row;
                float values[4];
#pragma unroll
                for(int atom=0;atom<4;++atom)
                {
                    float value=0.0f;
#pragma unroll
                    for(int source=0;source<8;++source)
                        value+=shared.reuse.partial[source][atom][lane][wave];
                    values[atom]=__bfloat162float(__float2bfloat16(value));
                }
                // Adjacent lanes hold the interleaved gate and up columns.
                // Read the odd lane before restricting execution to even lanes.
                float up[4];
#pragma unroll
                for(int atom=0;atom<4;++atom) up[atom]=__shfl_xor(values[atom],1);
                if(lane_row%2==0)
                {
                    float activated[4];float abs_max=1.0e-10f;
#pragma unroll
                    for(int atom=0;atom<4;++atom)
                    {
                        const float swish=values[atom]/(1.0f+__expf(-values[atom]));
                        activated[atom]=__bfloat162float(__float2bfloat16(swish*(up[atom]+0.0f)));
                        abs_max=fmaxf(abs_max,fabsf(activated[atom]));
                    }
                    abs_max=fmaxf(abs_max,__shfl_xor(abs_max,2));
                    abs_max=fmaxf(abs_max,__shfl_xor(abs_max,4));
                    abs_max=fmaxf(abs_max,__shfl_xor(abs_max,8));
                    if(output_row<row_end)
                    {
                        const auto bs=fp_f32_to_e8m0_block_scale<kDefaultMxScaleRoundMode,MxDtype::FP8_E4M3>(abs_max);
                        const float inverse=1.0f/bs.dq_scale;
#pragma unroll
                        for(int atom=0;atom<4;++atom)
                            output[static_cast<int64_t>(output_row)*512+n_tile*32+atom*8+lane_row/2]=opus::fp32_to_fp8(activated[atom]*inverse);
                        if(lane_row==0) output_scales[static_cast<int64_t>(output_row)*16+n_tile]=bs.byte;
                    }
                }
            }
            __syncthreads();
        }
    }
#endif
}

__global__ __launch_bounds__(512,1) void iq2r_scheduled_gate_wait3_direct_batch_kernel_tp4(
    const opus::fp8_t* __restrict__ activations,
    const uint8_t* __restrict__ scales,
    const uint8_t* __restrict__ all_data,
    const uint8_t* __restrict__ all_auxiliary,
    const int32_t* __restrict__ tasks,
    const int32_t* __restrict__ task_count,
    const int32_t* __restrict__ gather,
    opus::fp8_t* __restrict__ output,
    uint8_t* __restrict__ output_scales,
    int routes,int data_bytes,int auxiliary_bytes)
{
#if defined(__gfx950__)
    constexpr int MAtoms=1;
    constexpr int K=6144, Rows=16*MAtoms, Subtasks=1;
    static_assert(MAtoms==1);
    struct Storage {
        alignas(16) uint64_t codebook[kCodebookBytes/8];
        union {
            alignas(16) uint8_t cache[8][kQuadBytes];
            alignas(16) opus::vector_t<float,4> partial[8][4][64];
        } reuse;
    };
    __shared__ Storage shared;
    const int lane=threadIdx.x,wave=threadIdx.y,linear=wave*64+lane;
    const int lane_row=lane%16,lane_group=lane/16;
    const int num_tasks=task_count[0];
    const int total=num_tasks*Subtasks*16;
    for(int work=blockIdx.x;work<total;work+=gridDim.x)
    {
        const int task=(work/Subtasks)%num_tasks;
        const int sub=work%Subtasks;
        const int n_tile=work/(num_tasks*Subtasks);
        const int row_begin=tasks[task*3]+sub*Rows;
        const int row_end=min(row_begin+Rows,tasks[task*3]+tasks[task*3+1]);
        const int expert=tasks[task*3+2];
        if(row_begin>=row_end || row_begin<0 || row_end>routes || expert<0 || expert>=257) continue;
        const uint8_t* data=all_data+static_cast<int64_t>(expert)*data_bytes;
        const uint8_t* aux=all_auxiliary+static_cast<int64_t>(expert)*auxiliary_bytes;
        // Previous work ends with a barrier protecting both union and codebook.
        shared.codebook[linear]=reinterpret_cast<const uint64_t*>(aux)[linear];
        __syncthreads();
        const int nbase=n_tile*4;
        uint32_t bases=0;
#pragma unroll
        for(int atom=0;atom<4;++atom)
            bases|=static_cast<uint32_t>(aux[kCodebookBytes+nbase+atom])<<(atom*8);
        int rows[MAtoms];
#pragma unroll
        for(int m=0;m<MAtoms;++m) rows[m]=min(row_begin+m*16+lane_row,row_end-1);
        opus::vector_t<float,4> accumulators[MAtoms][4]={};
        opus::gmem<uint8_t> buffer(data,static_cast<unsigned int>(data_bytes));
        int base=(n_tile*48+wave*6)*kQuadBytes;
        iq2r_issue_quad(buffer,shared.reuse.cache[wave],lane,base);
        for(int iteration=0;iteration<6;++iteration)
        {
            const int kt=wave*6+iteration;
            IQ2RActivationFragment a[MAtoms]={};uint32_t sa[MAtoms];
#pragma unroll
            for(int m=0;m<MAtoms;++m)
            {
                const auto* input=reinterpret_cast<const uint8_t*>(activations+static_cast<int64_t>(rows[m])*K);
                const int ak=kt*128+lane_group*16;
                *reinterpret_cast<uint4*>(a[m].bytes)=*reinterpret_cast<const uint4*>(input+ak);
                *reinterpret_cast<uint4*>(a[m].bytes+16)=*reinterpret_cast<const uint4*>(input+ak+64);
                sa[m]=scales[static_cast<int64_t>(rows[m])*192+kt*4+lane_group]*0x01010101u;
            }
            iq2r_wait_vmcnt<3>();
            const auto compressed=iq2r_read_quad(shared.reuse.cache[wave],lane);
            asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
            if(iteration<5)
            {
                base+=kQuadBytes;
                iq2r_issue_quad(buffer,shared.reuse.cache[wave],lane,base);
            }
            e117_decode<MAtoms,2>(compressed,shared.codebook,bases,a,sa,accumulators);
        }
        // No wave may overwrite another wave's cache before its final read.
        __syncthreads();
#pragma unroll
        for(int m=0;m<MAtoms;++m)
        {
#pragma unroll
            for(int atom=0;atom<4;++atom)
                shared.reuse.partial[wave][atom][lane]=accumulators[m][atom];
            __syncthreads();
            if(wave<4)
            {
                const int local_row=m*16+lane_group*4+wave;
                const int output_row=row_begin+local_row;
                float values[4];
#pragma unroll
                for(int atom=0;atom<4;++atom)
                {
                    float value=0.0f;
#pragma unroll
                    for(int source=0;source<8;++source)
                        value+=shared.reuse.partial[source][atom][lane][wave];
                    values[atom]=__bfloat162float(__float2bfloat16(value));
                }
                // Adjacent lanes hold the interleaved gate and up columns.
                // Read the odd lane before restricting execution to even lanes.
                float up[4];
#pragma unroll
                for(int atom=0;atom<4;++atom) up[atom]=__shfl_xor(values[atom],1);
                if(lane_row%2==0)
                {
                    float activated[4];float abs_max=1.0e-10f;
#pragma unroll
                    for(int atom=0;atom<4;++atom)
                    {
                        const float swish=values[atom]/(1.0f+__expf(-values[atom]));
                        activated[atom]=__bfloat162float(__float2bfloat16(swish*(up[atom]+0.0f)));
                        abs_max=fmaxf(abs_max,fabsf(activated[atom]));
                    }
                    abs_max=fmaxf(abs_max,__shfl_xor(abs_max,2));
                    abs_max=fmaxf(abs_max,__shfl_xor(abs_max,4));
                    abs_max=fmaxf(abs_max,__shfl_xor(abs_max,8));
                    if(output_row<row_end)
                    {
                        const auto bs=fp_f32_to_e8m0_block_scale<kDefaultMxScaleRoundMode,MxDtype::FP8_E4M3>(abs_max);
                        const float inverse=1.0f/bs.dq_scale;
#pragma unroll
                        for(int atom=0;atom<4;++atom)
                            output[static_cast<int64_t>(output_row)*512+n_tile*32+atom*8+lane_row/2]=opus::fp32_to_fp8(activated[atom]*inverse);
                        if(lane_row==0) output_scales[static_cast<int64_t>(output_row)*16+n_tile]=bs.byte;
                    }
                }
            }
            __syncthreads();
        }
    }
#endif
}

__global__ __launch_bounds__(512,1) void iq2r_scheduled_gate_register_unroll6_direct_batch_kernel_tp4(
    const opus::fp8_t* __restrict__ activations,
    const uint8_t* __restrict__ scales,
    const uint8_t* __restrict__ all_data,
    const uint8_t* __restrict__ all_auxiliary,
    const int32_t* __restrict__ tasks,
    const int32_t* __restrict__ task_count,
    const int32_t* __restrict__ gather,
    opus::fp8_t* __restrict__ output,
    uint8_t* __restrict__ output_scales,
    int routes,int data_bytes,int auxiliary_bytes)
{
#if defined(__gfx950__)
    constexpr int MAtoms=1;
    constexpr int K=6144, Rows=16*MAtoms, Subtasks=1;
    static_assert(MAtoms==1);
    struct Storage {
        alignas(16) uint64_t codebook[kCodebookBytes/8];
        union {
            alignas(16) uint8_t cache[8][kQuadBytes];
            alignas(16) opus::vector_t<float,4> partial[8][4][64];
        } reuse;
    };
    __shared__ Storage shared;
    const int lane=threadIdx.x,wave=threadIdx.y,linear=wave*64+lane;
    const int lane_row=lane%16,lane_group=lane/16;
    const int num_tasks=task_count[0];
    const int total=num_tasks*Subtasks*16;
    for(int work=blockIdx.x;work<total;work+=gridDim.x)
    {
        const int task=(work/Subtasks)%num_tasks;
        const int sub=work%Subtasks;
        const int n_tile=work/(num_tasks*Subtasks);
        const int row_begin=tasks[task*3]+sub*Rows;
        const int row_end=min(row_begin+Rows,tasks[task*3]+tasks[task*3+1]);
        const int expert=tasks[task*3+2];
        if(row_begin>=row_end || row_begin<0 || row_end>routes || expert<0 || expert>=257) continue;
        const uint8_t* data=all_data+static_cast<int64_t>(expert)*data_bytes;
        const uint8_t* aux=all_auxiliary+static_cast<int64_t>(expert)*auxiliary_bytes;
        // Previous work ends with a barrier protecting both union and codebook.
        shared.codebook[linear]=reinterpret_cast<const uint64_t*>(aux)[linear];
        __syncthreads();
        const int nbase=n_tile*4;
        uint32_t bases=0;
#pragma unroll
        for(int atom=0;atom<4;++atom)
            bases|=static_cast<uint32_t>(aux[kCodebookBytes+nbase+atom])<<(atom*8);
        int rows[MAtoms];
#pragma unroll
        for(int m=0;m<MAtoms;++m) rows[m]=min(row_begin+m*16+lane_row,row_end-1);
        opus::vector_t<float,4> accumulators[MAtoms][4]={};
        opus::gmem<uint8_t> buffer(data,static_cast<unsigned int>(data_bytes));
        int base=(n_tile*48+wave*6)*kQuadBytes;
        IQ2RCompressedQuad pending=iq2r_scheduled_load_quad(data,data_bytes,base,lane);
#pragma unroll 6
        for(int iteration=0;iteration<6;++iteration)
        {
            const int kt=wave*6+iteration;
            IQ2RActivationFragment a[MAtoms]={};uint32_t sa[MAtoms];
#pragma unroll
            for(int m=0;m<MAtoms;++m)
            {
                const auto* input=reinterpret_cast<const uint8_t*>(activations+static_cast<int64_t>(rows[m])*K);
                const int ak=kt*128+lane_group*16;
                *reinterpret_cast<uint4*>(a[m].bytes)=*reinterpret_cast<const uint4*>(input+ak);
                *reinterpret_cast<uint4*>(a[m].bytes+16)=*reinterpret_cast<const uint4*>(input+ak+64);
                sa[m]=scales[static_cast<int64_t>(rows[m])*192+kt*4+lane_group]*0x01010101u;
            }
            const auto compressed=pending;
            if(iteration<5)
            {
                base+=kQuadBytes;
                pending=iq2r_scheduled_load_quad(data,data_bytes,base,lane);
            }
            e117_wide2_forced<MAtoms>(compressed,shared.codebook,bases,a,sa,accumulators);
        }
        // No wave may overwrite another wave's cache before its final read.
        __syncthreads();
#pragma unroll
        for(int m=0;m<MAtoms;++m)
        {
#pragma unroll
            for(int atom=0;atom<4;++atom)
                shared.reuse.partial[wave][atom][lane]=accumulators[m][atom];
            __syncthreads();
            if(wave<4)
            {
                const int local_row=m*16+lane_group*4+wave;
                const int output_row=row_begin+local_row;
                float values[4];
#pragma unroll
                for(int atom=0;atom<4;++atom)
                {
                    float value=0.0f;
#pragma unroll
                    for(int source=0;source<8;++source)
                        value+=shared.reuse.partial[source][atom][lane][wave];
                    values[atom]=__bfloat162float(__float2bfloat16(value));
                }
                // Adjacent lanes hold the interleaved gate and up columns.
                // Read the odd lane before restricting execution to even lanes.
                float up[4];
#pragma unroll
                for(int atom=0;atom<4;++atom) up[atom]=__shfl_xor(values[atom],1);
                if(lane_row%2==0)
                {
                    float activated[4];float abs_max=1.0e-10f;
#pragma unroll
                    for(int atom=0;atom<4;++atom)
                    {
                        const float swish=values[atom]/(1.0f+__expf(-values[atom]));
                        activated[atom]=__bfloat162float(__float2bfloat16(swish*(up[atom]+0.0f)));
                        abs_max=fmaxf(abs_max,fabsf(activated[atom]));
                    }
                    abs_max=fmaxf(abs_max,__shfl_xor(abs_max,2));
                    abs_max=fmaxf(abs_max,__shfl_xor(abs_max,4));
                    abs_max=fmaxf(abs_max,__shfl_xor(abs_max,8));
                    if(output_row<row_end)
                    {
                        const auto bs=fp_f32_to_e8m0_block_scale<kDefaultMxScaleRoundMode,MxDtype::FP8_E4M3>(abs_max);
                        const float inverse=1.0f/bs.dq_scale;
#pragma unroll
                        for(int atom=0;atom<4;++atom)
                            output[static_cast<int64_t>(output_row)*512+n_tile*32+atom*8+lane_row/2]=opus::fp32_to_fp8(activated[atom]*inverse);
                        if(lane_row==0) output_scales[static_cast<int64_t>(output_row)*16+n_tile]=bs.byte;
                    }
                }
            }
            __syncthreads();
        }
    }
#endif
}

__global__ __launch_bounds__(256,2) void iq2r_scheduled_small_register_single_epilogue_kernel_tp4(
    const opus::fp8_t* __restrict__ activations,
    const uint8_t* __restrict__ activation_scales,
    const uint8_t* __restrict__ all_data,
    const uint8_t* __restrict__ all_auxiliary,
    const int32_t* __restrict__ tasks,
    const int32_t* __restrict__ task_count,
    const __hip_bfloat16* __restrict__ all_bias,
    __hip_bfloat16* __restrict__ output,
    opus::fp8_t* __restrict__ fused_output,
    uint8_t* __restrict__ fused_output_scales,
    int M,
    int unused_N,
    int unused_K,
    int expert_count,
    int data_bytes,
    int auxiliary_bytes,
    float swiglu_limit,
    float swiglu_alpha,
    float swiglu_up_offset)
{
#if defined(__gfx950__)
    constexpr int N = 6144, K = 512;
    constexpr int OutputAtoms = 6, PhysicalWaves = 4, LoadAux = 2;
    constexpr bool ActivationLookahead = false, StagedScales = false;
    constexpr bool TiledActivationScales = false, VectorCodebookCopy = true, FusedSwiGLU = false;
    static_assert(OutputAtoms == 3 || OutputAtoms == 4 || OutputAtoms == 6);
    static_assert(PhysicalWaves == 2 || PhysicalWaves == 4 || PhysicalWaves == 8);
    static_assert(!ActivationLookahead || OutputAtoms == 6);
    static_assert(!(StagedScales && TiledActivationScales));
    static_assert(!FusedSwiGLU ||
                  (OutputAtoms == 4 && PhysicalWaves == 4 &&
                   !ActivationLookahead && !StagedScales));
    constexpr int kOutputColumns = OutputAtoms * kTileN;
    constexpr int kDirectTriplets =
        (OutputAtoms + kAtomsPerTriplet - 1) / kAtomsPerTriplet;
    constexpr int kStoredAtoms = OutputAtoms < 3 ? OutputAtoms : 3;
    constexpr int kScaleGroups = 2880 / kScaleBlock;
    constexpr int kPaddedScaleGroups = ((kScaleGroups + 3) / 4) * 4;
    constexpr int kStagedScaleBytes = 16 * kPaddedScaleGroups;

    struct SharedStorage
    {
        alignas(16) uint64_t codebook[kCodebookBytes / sizeof(uint64_t)];
        union
        {
            alignas(16) opus::vector_t<float, 4>
                partial[4][6][64];
            alignas(16) uint8_t scale_cache[kStagedScaleBytes];
        } reusable;
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
        if(row_begin < 0 || row_count <= 0 || row_begin + row_count > M)
            continue;
        if(expert_index < 0 || expert_index >= expert_count)
        {
            // Expert-parallel callers retain non-local routes in the sorted
            // permutation so the final indexed reduction can preserve the
            // original top-k order. Materialize their contribution exactly as
            // zero, matching the generic task kernel instead of leaving stale
            // workspace values behind.
            const int output_column_begin =
                n_tile_index * (FusedSwiGLU ? kOutputColumns / 2
                                            : kOutputColumns);
            const int output_width = FusedSwiGLU ? N / 2 : N;
            const int output_columns = min(FusedSwiGLU ? kOutputColumns / 2
                                                       : kOutputColumns,
                                           output_width - output_column_begin);
            const int task_elements = row_count * output_columns;
            for(int element = linear_thread; element < task_elements;
                element += linear_threads)
            {
                const int row = row_begin + element / output_columns;
                const int column = output_column_begin + element % output_columns;
                if constexpr(FusedSwiGLU)
                    fused_output[static_cast<int64_t>(row) * output_width + column] =
                        opus::fp32_to_fp8(0.0f);
                else
                    output[static_cast<int64_t>(row) * output_width + column] =
                        __float2bfloat16(0.0f);
            }
            if constexpr(FusedSwiGLU)
            {
                if(linear_thread < row_count)
                    fused_output_scales[
                        static_cast<int64_t>(row_begin + linear_thread) *
                            (N / 64) +
                        n_tile_index] = 127;
            }
            continue;
        }

        const uint8_t* data =
            all_data + static_cast<int64_t>(expert_index) * data_bytes;
        const uint8_t* auxiliary =
            all_auxiliary + static_cast<int64_t>(expert_index) * auxiliary_bytes;
        opus::gmem<uint8_t> data_buffer(data, static_cast<unsigned int>(data_bytes));
        if(expert_index != previous_expert)
        {
            if constexpr(VectorCodebookCopy && PhysicalWaves == 4)
            {
                reinterpret_cast<uint4*>(shared.codebook)[linear_thread] =
                    reinterpret_cast<const uint4*>(auxiliary)[linear_thread];
            }
            else
            {
                for(int entry = linear_thread;
                    entry < kCodebookBytes / static_cast<int>(sizeof(uint64_t));
                    entry += linear_threads)
                    shared.codebook[entry] =
                        reinterpret_cast<const uint64_t*>(auxiliary)[entry];
            }
            __syncthreads();
            previous_expert = expert_index;
        }

        const int n_block_base = n_tile_index * OutputAtoms;
        // Four-atom tiles advance by 64 columns and are not generally aligned
        // to the three-atom (48-column) physical IQ2R packing groups. Load the
        // two enclosing triplets and decode the requested four consecutive
        // atoms, matching the established fused-SwiGLU path.
        const int loaded_n_block_base =
            OutputAtoms == 4 ? n_block_base - n_block_base % kAtomsPerTriplet
                             : n_block_base;
        const int fused_atom_offset = n_block_base - loaded_n_block_base;
        uint32_t packed_bases[kDirectTriplets] = {};
#pragma unroll
        for(int triplet = 0; triplet < kDirectTriplets; ++triplet)
        {
#pragma unroll
            for(int atom = 0; atom < kAtomsPerTriplet; ++atom)
                packed_bases[triplet] |=
                    static_cast<uint32_t>(
                        auxiliary[kCodebookBytes + loaded_n_block_base +
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

            if constexpr(StagedScales)
            {
                const int valid_rows = sub_end - row_base;
                for(int index = linear_thread; index < kStagedScaleBytes;
                    index += linear_threads)
                {
                    const int row = index / kPaddedScaleGroups;
                    const int group = index % kPaddedScaleGroups;
                    uint8_t exponent = 127;
                    if(group < kScaleGroups)
                    {
                        const int source_row = row_base + min(row, valid_rows - 1);
                        exponent = activation_scales[
                            iq2r_activation_scale_offset<TiledActivationScales>(
                                source_row, group, M, kScaleGroups)];
                    }
                    const int scale_k_tile = group / 4;
                    const int scale_group = group % 4;
                    shared.reusable.scale_cache[scale_k_tile * 64 +
                                                scale_group * 16 + row] = exponent;
                }
                __syncthreads();
            }
            opus::vector_t<float, 4> accumulators[OutputAtoms] = {};

            int next_data_base = static_cast<int>(triplet_base(
                physical_tile(loaded_n_block_base, k_begin, k_tiles)));
            int next_second_data_base = static_cast<int>(triplet_base(
                physical_tile(loaded_n_block_base + kAtomsPerTriplet,
                              k_begin,
                              k_tiles)));

            if(wave < 4)
            {
                const auto first = iq2r_scheduled_load_compact_uniform(data, data_bytes, next_data_base, lane);
                const auto second = iq2r_scheduled_load_compact_uniform(data, data_bytes, next_data_base + kTripletBytes, lane);
                IQ2RActivationFragment a;
                const uint8_t* input = reinterpret_cast<const uint8_t*>(activations) + static_cast<int64_t>(load_row)*512 + wave*128 + lane_group*16;
                *reinterpret_cast<uint4*>(a.bytes) = *reinterpret_cast<const uint4*>(input);
                *reinterpret_cast<uint4*>(a.bytes+16) = *reinterpret_cast<const uint4*>(input+64);
                uint32_t sa = activation_scales[static_cast<int64_t>(load_row)*16 + wave*4 + lane_group];
                opus::i32x8_t b[3];
                uint32_t sb = iq2r_decode_direct_lds_triplet(first, shared.codebook, packed_bases[0], b);
                asm volatile("" : "+v"(sa));
                iq2r_cooperative_triplet_mfma<0>(a.words, b, accumulators, sa*0x01010101u, sb);
                sb = iq2r_decode_direct_lds_triplet(second, shared.codebook, packed_bases[1], b);
                iq2r_cooperative_triplet_mfma<3>(a.words, b, accumulators, sa*0x01010101u, sb);
            }
            if constexpr(StagedScales)
                __syncthreads();

            const int output_row = row_base + (lane / 16) * 4 + wave;
            if constexpr(FusedSwiGLU)
            {
                float raw_values[4] = {};
#pragma unroll
                for(int atom = 0; atom < 3; ++atom)
                    shared.reusable.partial[wave][atom][lane] = accumulators[atom];
                __syncthreads();
                if(wave < 4 && output_row < sub_end)
                {
#pragma unroll
                    for(int atom = 0; atom < 3; ++atom)
                    {
                        const float* p0 = reinterpret_cast<const float*>(
                            &shared.reusable.partial[0][atom][lane]);
                        const float* p1 = reinterpret_cast<const float*>(
                            &shared.reusable.partial[1][atom][lane]);
                        const float* p2 = reinterpret_cast<const float*>(
                            &shared.reusable.partial[2][atom][lane]);
                        const float* p3 = reinterpret_cast<const float*>(
                            &shared.reusable.partial[3][atom][lane]);
                        float value = p0[wave] + p1[wave] + p2[wave] + p3[wave];
                        if(all_bias != nullptr)
                            value += __bfloat162float(
                                all_bias[static_cast<int64_t>(expert_index) * N +
                                         n_tile_index * kOutputColumns + atom * 16 +
                                         lane % 16]);
                        raw_values[atom] = __bfloat162float(__float2bfloat16(value));
                    }
                }
                __syncthreads();

                shared.reusable.partial[wave][0][lane] = accumulators[3];
                __syncthreads();
                if(wave < 4 && output_row < sub_end)
                {
                    const float* p0 = reinterpret_cast<const float*>(
                        &shared.reusable.partial[0][0][lane]);
                    const float* p1 = reinterpret_cast<const float*>(
                        &shared.reusable.partial[1][0][lane]);
                    const float* p2 = reinterpret_cast<const float*>(
                        &shared.reusable.partial[2][0][lane]);
                    const float* p3 = reinterpret_cast<const float*>(
                        &shared.reusable.partial[3][0][lane]);
                    float value = p0[wave] + p1[wave] + p2[wave] + p3[wave];
                    if(all_bias != nullptr)
                        value += __bfloat162float(
                            all_bias[static_cast<int64_t>(expert_index) * N +
                                     n_tile_index * kOutputColumns + 3 * 16 +
                                     lane % 16]);
                    raw_values[3] = __bfloat162float(__float2bfloat16(value));
                }
                __syncthreads();

                if(wave < 4 && output_row < sub_end)
                {
                    const bool even_column = (lane % 2) == 0;
                    opus::vector_t<opus::bf16_t, 4> activated;
                    float abs_max = 1.0e-10f;
#pragma unroll
                    for(int atom = 0; atom < 4; ++atom)
                    {
                        const float paired = __shfl_xor(raw_values[atom], 1);
                        if(even_column)
                        {
                            const float gate = raw_values[atom];
                            const float up = paired;
                            const bool clamp = swiglu_limit > 0.0f;
                            const float clipped_gate =
                                clamp && gate > swiglu_limit ? swiglu_limit : gate;
                            const float clipped_up =
                                clamp && up < -swiglu_limit
                                    ? -swiglu_limit
                                    : (clamp && up > swiglu_limit
                                           ? swiglu_limit
                                           : up);
                            const float swish =
                                clipped_gate /
                                (1.0f + __expf(-swiglu_alpha * clipped_gate));
                            const __hip_bfloat16 rounded = __float2bfloat16(
                                swish * (clipped_up + swiglu_up_offset));
                            activated[atom] =
                                __builtin_bit_cast(opus::bf16_t, rounded);
                            abs_max =
                                fmaxf(abs_max, fabsf(__bfloat162float(rounded)));
                        }
                    }
                    abs_max = fmaxf(abs_max, __shfl_xor(abs_max, 2));
                    abs_max = fmaxf(abs_max, __shfl_xor(abs_max, 4));
                    abs_max = fmaxf(abs_max, __shfl_xor(abs_max, 8));
                    if(even_column)
                    {
                        const auto block_scale = fp_f32_to_e8m0_block_scale<
                            kDefaultMxScaleRoundMode,
                            MxDtype::FP8_E4M3>(abs_max);
                        const float inverse_scale = 1.0f / block_scale.dq_scale;
                        const int output_width = N / 2;
                        const int output_column_base =
                            n_tile_index * (kOutputColumns / 2) + (lane % 16) / 2;
#pragma unroll
                        for(int atom = 0; atom < 4; ++atom)
                            fused_output[static_cast<int64_t>(output_row) *
                                             output_width +
                                         output_column_base + atom * 8] =
                                opus::fp32_to_fp8(
                                    static_cast<float>(activated[atom]) *
                                    inverse_scale);
                        if(lane % 16 == 0)
                            fused_output_scales[
                                static_cast<int64_t>(output_row) * (N / 64) +
                                n_tile_index] = block_scale.byte;
                    }
                }
                __syncthreads();
            }
            else
            {
                if(wave < 4)
                {
#pragma unroll
                    for(int atom = 0; atom < 6; ++atom)
                        shared.reusable.partial[wave][atom][lane] = accumulators[atom];
                }
                __syncthreads();
                if(output_row < sub_end)
                {
#pragma unroll
                    for(int atom = 0; atom < 6; ++atom)
                    {
                        const int column = n_tile_index*96 + atom*16 + lane%16;
                        const float* p0 = reinterpret_cast<const float*>(&shared.reusable.partial[0][atom][lane]);
                        const float* p1 = reinterpret_cast<const float*>(&shared.reusable.partial[1][atom][lane]);
                        const float* p2 = reinterpret_cast<const float*>(&shared.reusable.partial[2][atom][lane]);
                        const float* p3 = reinterpret_cast<const float*>(&shared.reusable.partial[3][atom][lane]);
                        float value = ((p0[wave] + p1[wave]) + p2[wave]) + p3[wave];
                        if(all_bias != nullptr)
                            value += __bfloat162float(all_bias[static_cast<int64_t>(expert_index)*N + column]);
                        output[static_cast<int64_t>(output_row)*N + column] = __float2bfloat16(value);
                    }
                }
                __syncthreads();
            }

        }
    }
#endif
}

__global__ __launch_bounds__(576,1) void iq2r_down_token_route9_kernel_tp4(
    const opus::fp8_t* __restrict__ activations,
    const uint8_t* __restrict__ scales,
    const uint8_t* __restrict__ all_data,
    const uint8_t* __restrict__ all_auxiliary,
    const int32_t* __restrict__ expert_ids,
    const int32_t* __restrict__ scatter,
    const float* __restrict__ route_weights,
    __hip_bfloat16* __restrict__ output,
    int data_bytes, int auxiliary_bytes)
{
#if defined(__gfx950__)
    constexpr int Groups=9,KWaves=1,Atoms=3,N=6144,K=512;
    struct Storage {
        alignas(16) uint64_t codebook[Groups][kCodebookBytes/8];
        alignas(16) __hip_bfloat16 routes[9][Atoms*16];
    };
    __shared__ Storage shared;
    const int lane=threadIdx.x,wave=threadIdx.y;
    const int route_group=wave/KWaves,kwave=wave%KWaves;
    const int lane_row=lane%16,lane_group=lane/16;
    const int token=blockIdx.y,n_block=static_cast<int>(blockIdx.x)*Atoms;
    const int triplet=n_block/3*3,atom_offset=n_block%3;
    for(int group=0;group<9/Groups;++group)
    {
        const int route=group*Groups+route_group,original=token*9+route;
        const int expert=expert_ids[original],row=scatter[original];
        const bool valid=expert>=0 && expert<257 && row>=0 && row<static_cast<int>(gridDim.y)*9;
        const uint8_t* data=all_data+static_cast<int64_t>(expert)*data_bytes;
        const uint8_t* aux=all_auxiliary+static_cast<int64_t>(expert)*auxiliary_bytes;
        if(valid)
        for(int entry=kwave*64+lane;entry<kCodebookBytes/16;entry+=KWaves*64)
            reinterpret_cast<uint4*>(shared.codebook[route_group])[entry]=reinterpret_cast<const uint4*>(aux)[entry];
        __syncthreads();
        if(valid)
        {
        opus::vector_t<float,4> k_sums[4][Atoms]={};
#pragma unroll
        for(int ktile=0;ktile<4;++ktile)
        {
            const int base=static_cast<int>(triplet_base(physical_tile(triplet,ktile,4)));
            const auto compressed=iq2r_scheduled_load_compact_uniform(data,data_bytes,base,lane);
            IQ2RActivationFragment a={};
            const auto* input=reinterpret_cast<const uint8_t*>(activations+static_cast<int64_t>(row)*K);
            const int ak=ktile*128+lane_group*16;
            *reinterpret_cast<uint4*>(a.bytes)=*reinterpret_cast<const uint4*>(input+ak);
            *reinterpret_cast<uint4*>(a.bytes+16)=*reinterpret_cast<const uint4*>(input+ak+64);
            uint32_t sa=scales[static_cast<int64_t>(row)*16+ktile*4+lane_group];
            uint32_t bases=0;
#pragma unroll
            for(int atom=0;atom<3;++atom) bases|=static_cast<uint32_t>(aux[kCodebookBytes+triplet+atom])<<(atom*8);
            opus::i32x8_t b[3];
            const uint32_t sb=iq2r_decode_direct_lds_triplet(compressed,shared.codebook[route_group],bases,b);
            asm volatile("" : "+v"(sa));
            iq2r_cooperative_triplet_mfma<0>(a.words,b,k_sums[ktile],sa*0x01010101u,sb);
        }
        if(lane_group==0)
        {
#pragma unroll
            for(int atom=0;atom<Atoms;++atom)
            {
                const float value=((k_sums[0][atom][0]+k_sums[1][atom][0])+k_sums[2][atom][0])+k_sums[3][atom][0];
                shared.routes[route][atom*16+lane_row]=__float2bfloat16(value);
            }
        }
        }
        else if(lane_group==0)
        {
#pragma unroll
            for(int atom=0;atom<Atoms;++atom)
                shared.routes[route][atom*16+lane_row]=__float2bfloat16(0.0f);
        }
        __syncthreads();
    }
    if(wave==0 && lane_group==0)
    {
#pragma unroll
        for(int atom=0;atom<Atoms;++atom)
        {
            float combined=0.0f;
#pragma unroll
            for(int route=0;route<9;++route)
                combined=fmaf(__bfloat162float(shared.routes[route][atom*16+lane_row]),route_weights[token*9+route],combined);
            output[static_cast<int64_t>(token)*N+(n_block+atom)*16+lane_row]=__float2bfloat16(combined);
        }
    }
#endif
}
template<int MAtoms,int TaskRows>
__global__ __launch_bounds__(512,1) void iq2r_gate_quad_sparse_kernel_tp4(
    const opus::fp8_t* __restrict__ activations,
    const uint8_t* __restrict__ scales,
    const uint8_t* __restrict__ all_data,
    const uint8_t* __restrict__ all_auxiliary,
    const int32_t* __restrict__ tasks,
    const int32_t* __restrict__ task_count,
    const int32_t* __restrict__ gather,
    opus::fp8_t* __restrict__ output,
    uint8_t* __restrict__ output_scales,
    int routes,int data_bytes,int auxiliary_bytes)
{
#if defined(__gfx950__)
    constexpr int K=6144, Rows=16*MAtoms, Subtasks=TaskRows/(16*MAtoms);
    struct Storage {
        alignas(16) uint64_t codebook[kCodebookBytes/8];
        union {
            alignas(16) uint8_t cache[8][kQuadBytes];
            alignas(16) opus::vector_t<float,4> partial[8][4][64];
        } reuse;
        alignas(16) __hip_bfloat16 gate[Rows][64];
    };
    __shared__ Storage shared;
    const int lane=threadIdx.x,wave=threadIdx.y,linear=wave*64+lane;
    const int lane_row=lane%16,lane_group=lane/16;
    const int num_tasks=task_count[0];
    const int total=num_tasks*Subtasks*16;
    for(int work=blockIdx.x;work<total;work+=gridDim.x)
    {
        const int task=(work/Subtasks)%num_tasks;
        const int sub=work%Subtasks;
        const int n_tile=work/(num_tasks*Subtasks);
        const int row_begin=tasks[task*3]+sub*Rows;
        const int row_end=min(row_begin+Rows,tasks[task*3]+tasks[task*3+1]);
        const int expert=tasks[task*3+2];
        const int active_m=(row_end-row_begin+15)/16;
        if(row_begin>=row_end || row_begin<0 || row_end>routes || expert<0 || expert>=257) continue;
        const uint8_t* data=all_data+static_cast<int64_t>(expert)*data_bytes;
        const uint8_t* aux=all_auxiliary+static_cast<int64_t>(expert)*auxiliary_bytes;
        // Previous work ends with a barrier protecting both union and codebook.
        shared.codebook[linear]=reinterpret_cast<const uint64_t*>(aux)[linear];
        __syncthreads();
        const int nbase=n_tile*4;
        uint32_t bases=0;
#pragma unroll
        for(int atom=0;atom<4;++atom)
            bases|=static_cast<uint32_t>(aux[kCodebookBytes+nbase+atom])<<(atom*8);
        int rows[MAtoms];
#pragma unroll
        for(int m=0;m<MAtoms;++m) rows[m]=gather[min(row_begin+m*16+lane_row,row_end-1)]/9;
        opus::vector_t<float,4> accumulators[MAtoms][4]={};
        opus::gmem<uint8_t> buffer(data,static_cast<unsigned int>(data_bytes));
        int base=(n_tile*48+wave*6)*kQuadBytes;
        iq2r_issue_quad(buffer,shared.reuse.cache[wave],lane,base);
        for(int iteration=0;iteration<6;++iteration)
        {
            const int kt=wave*6+iteration;
            IQ2RActivationFragment a[MAtoms]={};uint32_t sa[MAtoms];
#pragma unroll
            for(int m=0;m<MAtoms;++m)
            {
                if(m>=active_m) continue;
                const auto* input=reinterpret_cast<const uint8_t*>(activations+static_cast<int64_t>(rows[m])*K);
                const int ak=kt*128+lane_group*16;
                *reinterpret_cast<uint4*>(a[m].bytes)=*reinterpret_cast<const uint4*>(input+ak);
                *reinterpret_cast<uint4*>(a[m].bytes+16)=*reinterpret_cast<const uint4*>(input+ak+64);
                sa[m]=scales[static_cast<int64_t>(rows[m])*192+kt*4+lane_group]*0x01010101u;
            }
            iq2r_wait_vmcnt<0>();
            const auto compressed=iq2r_read_quad(shared.reuse.cache[wave],lane);
            asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
            if(iteration<5)
            {
                base+=kQuadBytes;
                iq2r_issue_quad(buffer,shared.reuse.cache[wave],lane,base);
            }
            iq2r_quad_sparse_decode_mfma<MAtoms>(compressed,shared.codebook,bases,a,sa,accumulators,active_m);
        }
        // No wave may overwrite another wave's cache before its final read.
        __syncthreads();
#pragma unroll
        for(int m=0;m<MAtoms;++m)
        {
            if(m>=active_m) continue;
#pragma unroll
            for(int atom=0;atom<4;++atom) shared.reuse.partial[wave][atom][lane]=accumulators[m][atom];
            __syncthreads();
            if(wave<4)
            {
                const int local_row=m*16+lane_group*4+wave;
#pragma unroll
                for(int atom=0;atom<4;++atom)
                {
                    float value=0.0f;
#pragma unroll
                    for(int source=0;source<8;++source) value+=shared.reuse.partial[source][atom][lane][wave];
                    shared.gate[local_row][atom*16+lane_row]=__float2bfloat16(value);
                }
            }
            __syncthreads();
        }
        if(linear<active_m*16*4)
        {
            const int local_row=linear/4,quant_lane=linear%4;
            const int output_row=row_begin+local_row;
            using B=opus::vector_t<opus::bf16_t,8>;
            using Q=opus::vector_t<opus::fp8_t,8>;
            B values;float abs_max=1.0e-10f;
#pragma unroll
            for(int element=0;element<8;++element)
            {
                const int column=quant_lane*8+element;
                const float gate=__bfloat162float(shared.gate[local_row][2*column]);
                const float up=__bfloat162float(shared.gate[local_row][2*column+1]);
                const float swish=gate/(1.0f+__expf(-gate));
                const __hip_bfloat16 rounded=__float2bfloat16(swish*(up+0.0f));
                values[element]=__builtin_bit_cast(opus::bf16_t,rounded);
                abs_max=fmaxf(abs_max,fabsf(__bfloat162float(rounded)));
            }
            abs_max=fmaxf(abs_max,__shfl_xor(abs_max,1));
            abs_max=fmaxf(abs_max,__shfl_xor(abs_max,2));
            if(output_row<row_end)
            {
                const auto bs=fp_f32_to_e8m0_block_scale<kDefaultMxScaleRoundMode,MxDtype::FP8_E4M3>(abs_max);
                const float inverse=1.0f/bs.dq_scale;
                Q q;
#pragma unroll
                for(int element=0;element<8;++element) q[element]=opus::fp32_to_fp8(static_cast<float>(values[element])*inverse);
                *reinterpret_cast<Q*>(output+static_cast<int64_t>(output_row)*512+n_tile*32+quant_lane*8)=q;
                if(quant_lane==0) output_scales[static_cast<int64_t>(output_row)*16+n_tile]=bs.byte;
            }
        }
        __syncthreads();
    }
#endif
}

__global__ __launch_bounds__(256, 2) void iq2r_scheduled_large_fixed_kn_kernel_tp4(
    const opus::fp8_t* __restrict__ activations,
    const uint8_t* __restrict__ activation_scales,
    const uint8_t* __restrict__ all_data,
    const uint8_t* __restrict__ all_auxiliary,
    const int32_t* __restrict__ tasks,
    const int32_t* __restrict__ task_count,
    const __hip_bfloat16* __restrict__ all_bias,
    __hip_bfloat16* __restrict__ output,
    int M,
    int unused_N,
    int unused_K,
    int expert_count,
    int data_bytes,
    int auxiliary_bytes)
{
#if defined(__gfx950__)
    constexpr int N = 6144;
    constexpr int K = 512;
    constexpr int MAtoms = 2;
    constexpr bool NMajor = false;
    constexpr int kPhysicalWaves = 4;
    static_assert(MAtoms == 2 || MAtoms == 4);
    constexpr int kMAtoms = MAtoms;
    constexpr int kNAtomsPerWave = 3;
    constexpr int kOutputColumns =
        kPhysicalWaves * kNAtomsPerWave * kTileN;

    struct SharedStorage
    {
        alignas(16) uint64_t codebook[kCodebookBytes / sizeof(uint64_t)];
        alignas(16) uint8_t weight_cache[kPhysicalWaves][kTripletLDSBytes];
    };
    __shared__ SharedStorage shared;

    const int lane = static_cast<int>(threadIdx.x);
    const int wave = static_cast<int>(threadIdx.y);
    const int linear_thread = wave * 64 + lane;
    const int lane_row = lane % 16;
    const int lane_group = lane / 16;
    const int num_tasks = task_count[0];
    if(num_tasks <= 0)
        return;

    const int n_tiles = (N + kOutputColumns - 1) / kOutputColumns;
    const int total_tiles = num_tasks * n_tiles;
    const int k_tiles = (K + kTileK - 1) / kTileK;
    int previous_expert = -1;

    for(int work_index = static_cast<int>(blockIdx.x); work_index < total_tiles;
        work_index += static_cast<int>(gridDim.x))
    {
        const int task_index =
            NMajor ? work_index % num_tasks : work_index / n_tiles;
        const int n_tile_index =
            NMajor ? work_index / num_tasks : work_index % n_tiles;
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
            // Protect readers of the old expert before reusing the codebook.
            if(previous_expert >= 0)
                __syncthreads();
            for(int entry = linear_thread;
                entry < kCodebookBytes / static_cast<int>(sizeof(uint64_t));
                entry += 256)
                shared.codebook[entry] =
                    reinterpret_cast<const uint64_t*>(auxiliary)[entry];
            __syncthreads();
            previous_expert = expert_index;
        }

        const int n_block_base =
            n_tile_index * (kOutputColumns / kTileN) +
            wave * kNAtomsPerWave;
        uint32_t packed_bases = 0;
#pragma unroll
        for(int atom = 0; atom < kNAtomsPerWave; ++atom)
            packed_bases |=
                static_cast<uint32_t>(
                    auxiliary[kCodebookBytes + n_block_base + atom])
                << (atom * 8);

        const int row_end = row_begin + row_count;
        for(int row_base = row_begin; row_base < row_end;
            row_base += kMAtoms * 16)
        {
            const int sub_end = min(row_base + kMAtoms * 16, row_end);
            opus::vector_t<float, 4>
                accumulators[kMAtoms][kNAtomsPerWave] = {};

            int next_data_base = static_cast<int>(triplet_base(
                physical_tile(n_block_base, 0, k_tiles)));
            iq2r_issue_direct_lds_triplet(data_buffer,
                                           shared.weight_cache[wave],
                                           lane,
                                           0,
                                           next_data_base);
            next_data_base += kGroupBytes;

            for(int k_tile = 0; k_tile < k_tiles; ++k_tile)
            {
                union ActivationFragment
                {
                    opus::i32x8_t words;
                    uint8_t bytes[32];
                } activation_fragments[kMAtoms] = {};
                uint32_t scale_a[kMAtoms];
#pragma unroll
                for(int m_atom = 0; m_atom < kMAtoms; ++m_atom)
                    scale_a[m_atom] = 127u * 0x01010101u;

#pragma unroll
                for(int m_atom = 0; m_atom < kMAtoms; ++m_atom)
                {
                    const int input_row = row_base + m_atom * 16 + lane_row;
                    if(input_row < sub_end)
                    {
                        const int activation_k =
                            k_tile * kTileK + lane_group * 16;
                        const auto* activation_row =
                            reinterpret_cast<const uint8_t*>(
                                activations + static_cast<int64_t>(input_row) * K);
                        if(activation_k + 15 < K)
                            *reinterpret_cast<uint4*>(
                                activation_fragments[m_atom].bytes) =
                                *reinterpret_cast<const uint4*>(
                                    activation_row + activation_k);
                        if(activation_k + 64 + 15 < K)
                            *reinterpret_cast<uint4*>(
                                activation_fragments[m_atom].bytes + 16) =
                                *reinterpret_cast<const uint4*>(
                                    activation_row + activation_k + 64);

                        const int scale_column = k_tile * 4 + lane_group;
                        if(scale_column < K / kScaleBlock)
                        {
                            const uint32_t exponent = activation_scales[
                                static_cast<int64_t>(input_row) *
                                    (K / kScaleBlock) +
                                scale_column];
                            scale_a[m_atom] = exponent * 0x01010101u;
                        }
                    }
                }

                // Two weight requests precede eight activation-vector and four
                // activation-scale requests on full K tiles. Wait only for the
                // older weight pair, leaving that activation tail in flight
                // while the compact record moves out of LDS. The padded final
                // K tile omits the second vector load and two scale groups.
                // Partial slabs issue fewer loads than the fixed wait allowance.
                if(sub_end - row_base < kMAtoms * 16)
                    __builtin_amdgcn_s_waitcnt(0x0F70);
                else if(k_tile + 1 < k_tiles)
                    __builtin_amdgcn_s_waitcnt(0x0F7C); // vmcnt(12)
                else
                    __builtin_amdgcn_s_waitcnt(0x0F78); // vmcnt(8)
                const IQ2RCompressedTriplet compressed =
                    iq2r_read_direct_lds_triplet(
                        shared.weight_cache[wave], lane, 0);
                asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
                const bool has_next = k_tile + 1 < k_tiles;
                if(has_next)
                {
                    iq2r_issue_direct_lds_triplet(data_buffer,
                                                   shared.weight_cache[wave],
                                                   lane,
                                                   0,
                                                   next_data_base);
                    next_data_base += kGroupBytes;
                }
                opus::i32x8_t weight_fragments[kNAtomsPerWave];
                const uint32_t scale_b = iq2r_decode_direct_lds_triplet(
                    compressed,
                    shared.codebook,
                    packed_bases,
                    weight_fragments);
                if(has_next)
                    __builtin_amdgcn_s_waitcnt(0x0F72); // vmcnt(2)
                else
                    __builtin_amdgcn_s_waitcnt(0x0F70); // vmcnt(0)
#pragma unroll
                for(int m_atom = 0; m_atom < kMAtoms; ++m_atom)
                    if(row_base + m_atom * 16 < sub_end)
                    iq2r_cooperative_triplet_mfma<0>(
                        activation_fragments[m_atom].words,
                        weight_fragments,
                        accumulators[m_atom],
                        scale_a[m_atom],
                        scale_b);
            }

#pragma unroll
            for(int m_atom = 0; m_atom < kMAtoms; ++m_atom)
            {
                if(row_base + m_atom * 16 >= sub_end) continue;
#pragma unroll
                for(int atom = 0; atom < kNAtomsPerWave; ++atom)
                {
                    const int output_column =
                        (n_block_base + atom) * kTileN + lane_row;
#pragma unroll
                    for(int item = 0; item < 4; ++item)
                    {
                        const int output_row =
                            row_base + m_atom * 16 + lane_group * 4 + item;
                        if(output_row < sub_end && output_column < N)
                        {
                            float value = accumulators[m_atom][atom][item];
                            if(all_bias != nullptr)
                                value += __bfloat162float(all_bias[
                                    static_cast<int64_t>(expert_index) * N +
                                    output_column]);
                            output[static_cast<int64_t>(output_row) * N +
                                   output_column] = __float2bfloat16(value);
                        }
                    }
                }
            }
        }
    }
#endif
}

__global__ __launch_bounds__(256, 2) void iq2r_scheduled_large_register_prefetch_uniform_kernel_tp4(
    const opus::fp8_t* __restrict__ activations,
    const uint8_t* __restrict__ activation_scales,
    const uint8_t* __restrict__ all_data,
    const uint8_t* __restrict__ all_auxiliary,
    const int32_t* __restrict__ tasks,
    const int32_t* __restrict__ task_count,
    const __hip_bfloat16* __restrict__ all_bias,
    __hip_bfloat16* __restrict__ output,
    int M,
    int unused_N,
    int unused_K,
    int expert_count,
    int data_bytes,
    int auxiliary_bytes)
{
#if defined(__gfx950__)
    constexpr int N = 6144;
    constexpr int K = 512;
    constexpr int MAtoms = 2;
    constexpr bool NMajor = false;
    constexpr int kPhysicalWaves = 4;
    static_assert(MAtoms == 2 || MAtoms == 4);
    constexpr int kMAtoms = MAtoms;
    constexpr int kNAtomsPerWave = 3;
    constexpr int kOutputColumns =
        kPhysicalWaves * kNAtomsPerWave * kTileN;

    struct SharedStorage
    {
        alignas(16) uint64_t codebook[kCodebookBytes / sizeof(uint64_t)];
    };
    __shared__ SharedStorage shared;

    const int lane = static_cast<int>(threadIdx.x);
    const int wave = static_cast<int>(threadIdx.y);
    const int linear_thread = wave * 64 + lane;
    const int lane_row = lane % 16;
    const int lane_group = lane / 16;
    const int num_tasks = task_count[0];
    if(num_tasks <= 0)
        return;

    const int n_tiles = (N + kOutputColumns - 1) / kOutputColumns;
    const int total_tiles = num_tasks * n_tiles;
    const int k_tiles = (K + kTileK - 1) / kTileK;
    int previous_expert = -1;

    for(int work_index = static_cast<int>(blockIdx.x); work_index < total_tiles;
        work_index += static_cast<int>(gridDim.x))
    {
        const int task_index =
            NMajor ? work_index % num_tasks : work_index / n_tiles;
        const int n_tile_index =
            NMajor ? work_index / num_tasks : work_index % n_tiles;
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
            // Protect readers of the old expert before reusing the codebook.
            if(previous_expert >= 0)
                __syncthreads();
            for(int entry = linear_thread;
                entry < kCodebookBytes / static_cast<int>(sizeof(uint64_t));
                entry += 256)
                shared.codebook[entry] =
                    reinterpret_cast<const uint64_t*>(auxiliary)[entry];
            __syncthreads();
            previous_expert = expert_index;
        }

        const int n_block_base =
            n_tile_index * (kOutputColumns / kTileN) +
            wave * kNAtomsPerWave;
        uint32_t packed_bases = 0;
#pragma unroll
        for(int atom = 0; atom < kNAtomsPerWave; ++atom)
            packed_bases |=
                static_cast<uint32_t>(
                    auxiliary[kCodebookBytes + n_block_base + atom])
                << (atom * 8);

        const int row_end = row_begin + row_count;
        for(int row_base = row_begin; row_base < row_end;
            row_base += kMAtoms * 16)
        {
            const int sub_end = min(row_base + kMAtoms * 16, row_end);
            opus::vector_t<float, 4>
                accumulators[kMAtoms][kNAtomsPerWave] = {};

            int next_data_base = static_cast<int>(triplet_base(
                physical_tile(n_block_base, 0, k_tiles)));
            IQ2RCompressedTriplet pending = iq2r_scheduled_load_compact_uniform(data, data_bytes, next_data_base, lane);
            next_data_base += kGroupBytes;

#pragma unroll 1
            for(int k_tile = 0; k_tile < k_tiles; ++k_tile)
            {
                union ActivationFragment
                {
                    opus::i32x8_t words;
                    uint8_t bytes[32];
                } activation_fragments[kMAtoms] = {};
                uint32_t scale_a[kMAtoms];
#pragma unroll
                for(int m_atom = 0; m_atom < kMAtoms; ++m_atom)
                    scale_a[m_atom] = 127u * 0x01010101u;

#pragma unroll
                for(int m_atom = 0; m_atom < kMAtoms; ++m_atom)
                {
                    const int input_row = row_base + m_atom * 16 + lane_row;
                    if(input_row < sub_end)
                    {
                        const int activation_k =
                            k_tile * kTileK + lane_group * 16;
                        const auto* activation_row =
                            reinterpret_cast<const uint8_t*>(
                                activations + static_cast<int64_t>(input_row) * K);
                        if(activation_k + 15 < K)
                            *reinterpret_cast<uint4*>(
                                activation_fragments[m_atom].bytes) =
                                *reinterpret_cast<const uint4*>(
                                    activation_row + activation_k);
                        if(activation_k + 64 + 15 < K)
                            *reinterpret_cast<uint4*>(
                                activation_fragments[m_atom].bytes + 16) =
                                *reinterpret_cast<const uint4*>(
                                    activation_row + activation_k + 64);

                        const int scale_column = k_tile * 4 + lane_group;
                        if(scale_column < K / kScaleBlock)
                        {
                            const uint32_t exponent = activation_scales[
                                static_cast<int64_t>(input_row) *
                                    (K / kScaleBlock) +
                                scale_column];
                            scale_a[m_atom] = exponent * 0x01010101u;
                        }
                    }
                }

                const IQ2RCompressedTriplet compressed = pending;
                if(k_tile + 1 < k_tiles)
                {
                    pending = iq2r_scheduled_load_compact_uniform(data, data_bytes, next_data_base, lane);
                    next_data_base += kGroupBytes;
                }
                opus::i32x8_t weight_fragments[kNAtomsPerWave];
                const uint32_t scale_b = iq2r_decode_direct_lds_triplet(
                    compressed,
                    shared.codebook,
                    packed_bases,
                    weight_fragments);
#pragma unroll
                for(int m_atom = 0; m_atom < kMAtoms; ++m_atom)
                    if(row_base + m_atom * 16 < sub_end)
                    iq2r_cooperative_triplet_mfma<0>(
                        activation_fragments[m_atom].words,
                        weight_fragments,
                        accumulators[m_atom],
                        scale_a[m_atom],
                        scale_b);
            }

#pragma unroll
            for(int m_atom = 0; m_atom < kMAtoms; ++m_atom)
            {
                if(row_base + m_atom * 16 >= sub_end) continue;
#pragma unroll
                for(int atom = 0; atom < kNAtomsPerWave; ++atom)
                {
                    const int output_column =
                        (n_block_base + atom) * kTileN + lane_row;
#pragma unroll
                    for(int item = 0; item < 4; ++item)
                    {
                        const int output_row =
                            row_base + m_atom * 16 + lane_group * 4 + item;
                        if(output_row < sub_end && output_column < N)
                        {
                            float value = accumulators[m_atom][atom][item];
                            if(all_bias != nullptr)
                                value += __bfloat162float(all_bias[
                                    static_cast<int64_t>(expert_index) * N +
                                    output_column]);
                            output[static_cast<int64_t>(output_row) * N +
                                   output_column] = __float2bfloat16(value);
                        }
                    }
                }
            }
        }
    }
#endif
}

__global__ __launch_bounds__(256, 2) void iq2r_scheduled_large_fixed_kn_deferred_scale_kernel_tp4(
    const opus::fp8_t* __restrict__ activations,
    const uint8_t* __restrict__ activation_scales,
    const uint8_t* __restrict__ all_data,
    const uint8_t* __restrict__ all_auxiliary,
    const int32_t* __restrict__ tasks,
    const int32_t* __restrict__ task_count,
    const __hip_bfloat16* __restrict__ all_bias,
    __hip_bfloat16* __restrict__ output,
    int M,
    int unused_N,
    int unused_K,
    int expert_count,
    int data_bytes,
    int auxiliary_bytes)
{
#if defined(__gfx950__)
    constexpr int N = 6144;
    constexpr int K = 512;
    constexpr int MAtoms = 2;
    constexpr bool NMajor = false;
    constexpr int kPhysicalWaves = 4;
    static_assert(MAtoms == 2 || MAtoms == 4);
    constexpr int kMAtoms = MAtoms;
    constexpr int kNAtomsPerWave = 3;
    constexpr int kOutputColumns =
        kPhysicalWaves * kNAtomsPerWave * kTileN;

    struct SharedStorage
    {
        alignas(16) uint64_t codebook[kCodebookBytes / sizeof(uint64_t)];
        alignas(16) uint8_t weight_cache[kPhysicalWaves][kTripletLDSBytes];
    };
    __shared__ SharedStorage shared;

    const int lane = static_cast<int>(threadIdx.x);
    const int wave = static_cast<int>(threadIdx.y);
    const int linear_thread = wave * 64 + lane;
    const int lane_row = lane % 16;
    const int lane_group = lane / 16;
    const int num_tasks = task_count[0];
    if(num_tasks <= 0)
        return;

    const int n_tiles = (N + kOutputColumns - 1) / kOutputColumns;
    const int total_tiles = num_tasks * n_tiles;
    const int k_tiles = (K + kTileK - 1) / kTileK;
    int previous_expert = -1;

    for(int work_index = static_cast<int>(blockIdx.x); work_index < total_tiles;
        work_index += static_cast<int>(gridDim.x))
    {
        const int task_index =
            NMajor ? work_index % num_tasks : work_index / n_tiles;
        const int n_tile_index =
            NMajor ? work_index / num_tasks : work_index % n_tiles;
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
            // Protect readers of the old expert before reusing the codebook.
            if(previous_expert >= 0)
                __syncthreads();
            for(int entry = linear_thread;
                entry < kCodebookBytes / static_cast<int>(sizeof(uint64_t));
                entry += 256)
                shared.codebook[entry] =
                    reinterpret_cast<const uint64_t*>(auxiliary)[entry];
            __syncthreads();
            previous_expert = expert_index;
        }

        const int n_block_base =
            n_tile_index * (kOutputColumns / kTileN) +
            wave * kNAtomsPerWave;
        uint32_t packed_bases = 0;
#pragma unroll
        for(int atom = 0; atom < kNAtomsPerWave; ++atom)
            packed_bases |=
                static_cast<uint32_t>(
                    auxiliary[kCodebookBytes + n_block_base + atom])
                << (atom * 8);

        const int row_end = row_begin + row_count;
        for(int row_base = row_begin; row_base < row_end;
            row_base += kMAtoms * 16)
        {
            const int sub_end = min(row_base + kMAtoms * 16, row_end);
            opus::vector_t<float, 4>
                accumulators[kMAtoms][kNAtomsPerWave] = {};

            int next_data_base = static_cast<int>(triplet_base(
                physical_tile(n_block_base, 0, k_tiles)));
            iq2r_issue_direct_lds_triplet(data_buffer,
                                           shared.weight_cache[wave],
                                           lane,
                                           0,
                                           next_data_base);
            next_data_base += kGroupBytes;

            for(int k_tile = 0; k_tile < k_tiles; ++k_tile)
            {
                union ActivationFragment
                {
                    opus::i32x8_t words;
                    uint8_t bytes[32];
                } activation_fragments[kMAtoms] = {};
                uint32_t scale_a[kMAtoms];
#pragma unroll
                for(int m_atom = 0; m_atom < kMAtoms; ++m_atom)
                    scale_a[m_atom] = 127u;

#pragma unroll
                for(int m_atom = 0; m_atom < kMAtoms; ++m_atom)
                {
                    const int input_row = min(row_base + m_atom * 16 + lane_row, sub_end - 1);
                    if(row_base + m_atom * 16 < sub_end)
                    {
                        const int activation_k =
                            k_tile * kTileK + lane_group * 16;
                        const auto* activation_row =
                            reinterpret_cast<const uint8_t*>(
                                activations + static_cast<int64_t>(input_row) * K);
                        if(activation_k + 15 < K)
                            *reinterpret_cast<uint4*>(
                                activation_fragments[m_atom].bytes) =
                                *reinterpret_cast<const uint4*>(
                                    activation_row + activation_k);
                        if(activation_k + 64 + 15 < K)
                            *reinterpret_cast<uint4*>(
                                activation_fragments[m_atom].bytes + 16) =
                                *reinterpret_cast<const uint4*>(
                                    activation_row + activation_k + 64);

                        const int scale_column = k_tile * 4 + lane_group;
                        if(scale_column < K / kScaleBlock)
                        {
                            const uint32_t exponent = activation_scales[
                                static_cast<int64_t>(input_row) *
                                    (K / kScaleBlock) +
                                scale_column];
                            scale_a[m_atom] = exponent;
                        }
                    }
                }

                // Two weight requests precede eight activation-vector and four
                // activation-scale requests on full K tiles. Wait only for the
                // older weight pair, leaving that activation tail in flight
                // while the compact record moves out of LDS. The padded final
                // K tile omits the second vector load and two scale groups.
                // Partial slabs issue fewer loads than the fixed wait allowance.
                if(sub_end - row_base < kMAtoms * 16)
                    __builtin_amdgcn_s_waitcnt(0x0F70);
                else if(k_tile + 1 < k_tiles)
                    __builtin_amdgcn_s_waitcnt(0x0F7C); // vmcnt(12)
                else
                    __builtin_amdgcn_s_waitcnt(0x0F78); // vmcnt(8)
                const IQ2RCompressedTriplet compressed =
                    iq2r_read_direct_lds_triplet(
                        shared.weight_cache[wave], lane, 0);
                asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
                const bool has_next = k_tile + 1 < k_tiles;
                if(has_next)
                {
                    iq2r_issue_direct_lds_triplet(data_buffer,
                                                   shared.weight_cache[wave],
                                                   lane,
                                                   0,
                                                   next_data_base);
                    next_data_base += kGroupBytes;
                }
                opus::i32x8_t weight_fragments[kNAtomsPerWave];
                const uint32_t scale_b = iq2r_decode_direct_lds_triplet(
                    compressed,
                    shared.codebook,
                    packed_bases,
                    weight_fragments);
                if(has_next)
                    __builtin_amdgcn_s_waitcnt(0x0F72); // vmcnt(2)
                else
                    __builtin_amdgcn_s_waitcnt(0x0F70); // vmcnt(0)
#pragma unroll
                for(int m_atom = 0; m_atom < kMAtoms; ++m_atom)
                    asm volatile("" : "+v"(scale_a[m_atom]));
#pragma unroll
                for(int m_atom = 0; m_atom < kMAtoms; ++m_atom)
                    if(row_base + m_atom * 16 < sub_end)
                    iq2r_cooperative_triplet_mfma<0>(
                        activation_fragments[m_atom].words,
                        weight_fragments,
                        accumulators[m_atom],
                        scale_a[m_atom] * 0x01010101u,
                        scale_b);
            }

#pragma unroll
            for(int m_atom = 0; m_atom < kMAtoms; ++m_atom)
            {
                if(row_base + m_atom * 16 >= sub_end) continue;
#pragma unroll
                for(int atom = 0; atom < kNAtomsPerWave; ++atom)
                {
                    const int output_column =
                        (n_block_base + atom) * kTileN + lane_row;
#pragma unroll
                    for(int item = 0; item < 4; ++item)
                    {
                        const int output_row =
                            row_base + m_atom * 16 + lane_group * 4 + item;
                        if(output_row < sub_end && output_column < N)
                        {
                            float value = accumulators[m_atom][atom][item];
                            if(all_bias != nullptr)
                                value += __bfloat162float(all_bias[
                                    static_cast<int64_t>(expert_index) * N +
                                    output_column]);
                            output[static_cast<int64_t>(output_row) * N +
                                   output_column] = __float2bfloat16(value);
                        }
                    }
                }
            }
        }
    }
#endif
}

__global__ __launch_bounds__(256, 2) void iq2r_scheduled_large_register_prefetch_uniform_deferred_scale_kernel_tp4(
    const opus::fp8_t* __restrict__ activations,
    const uint8_t* __restrict__ activation_scales,
    const uint8_t* __restrict__ all_data,
    const uint8_t* __restrict__ all_auxiliary,
    const int32_t* __restrict__ tasks,
    const int32_t* __restrict__ task_count,
    const __hip_bfloat16* __restrict__ all_bias,
    __hip_bfloat16* __restrict__ output,
    int M,
    int unused_N,
    int unused_K,
    int expert_count,
    int data_bytes,
    int auxiliary_bytes)
{
#if defined(__gfx950__)
    constexpr int N = 6144;
    constexpr int K = 512;
    constexpr int MAtoms = 2;
    constexpr bool NMajor = false;
    constexpr int kPhysicalWaves = 4;
    static_assert(MAtoms == 2 || MAtoms == 4);
    constexpr int kMAtoms = MAtoms;
    constexpr int kNAtomsPerWave = 3;
    constexpr int kOutputColumns =
        kPhysicalWaves * kNAtomsPerWave * kTileN;

    struct SharedStorage
    {
        alignas(16) uint64_t codebook[kCodebookBytes / sizeof(uint64_t)];
    };
    __shared__ SharedStorage shared;

    const int lane = static_cast<int>(threadIdx.x);
    const int wave = static_cast<int>(threadIdx.y);
    const int linear_thread = wave * 64 + lane;
    const int lane_row = lane % 16;
    const int lane_group = lane / 16;
    const int num_tasks = task_count[0];
    if(num_tasks <= 0)
        return;

    const int n_tiles = (N + kOutputColumns - 1) / kOutputColumns;
    const int total_tiles = num_tasks * n_tiles;
    const int k_tiles = (K + kTileK - 1) / kTileK;
    int previous_expert = -1;

    for(int work_index = static_cast<int>(blockIdx.x); work_index < total_tiles;
        work_index += static_cast<int>(gridDim.x))
    {
        const int task_index =
            NMajor ? work_index % num_tasks : work_index / n_tiles;
        const int n_tile_index =
            NMajor ? work_index / num_tasks : work_index % n_tiles;
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
            // Protect readers of the old expert before reusing the codebook.
            if(previous_expert >= 0)
                __syncthreads();
            for(int entry = linear_thread;
                entry < kCodebookBytes / static_cast<int>(sizeof(uint64_t));
                entry += 256)
                shared.codebook[entry] =
                    reinterpret_cast<const uint64_t*>(auxiliary)[entry];
            __syncthreads();
            previous_expert = expert_index;
        }

        const int n_block_base =
            n_tile_index * (kOutputColumns / kTileN) +
            wave * kNAtomsPerWave;
        uint32_t packed_bases = 0;
#pragma unroll
        for(int atom = 0; atom < kNAtomsPerWave; ++atom)
            packed_bases |=
                static_cast<uint32_t>(
                    auxiliary[kCodebookBytes + n_block_base + atom])
                << (atom * 8);

        const int row_end = row_begin + row_count;
        for(int row_base = row_begin; row_base < row_end;
            row_base += kMAtoms * 16)
        {
            const int sub_end = min(row_base + kMAtoms * 16, row_end);
            opus::vector_t<float, 4>
                accumulators[kMAtoms][kNAtomsPerWave] = {};

            int next_data_base = static_cast<int>(triplet_base(
                physical_tile(n_block_base, 0, k_tiles)));
            IQ2RCompressedTriplet pending = iq2r_scheduled_load_compact_uniform(data, data_bytes, next_data_base, lane);
            next_data_base += kGroupBytes;

#pragma unroll 1
            for(int k_tile = 0; k_tile < k_tiles; ++k_tile)
            {
                union ActivationFragment
                {
                    opus::i32x8_t words;
                    uint8_t bytes[32];
                } activation_fragments[kMAtoms] = {};
                uint32_t scale_a[kMAtoms];
#pragma unroll
                for(int m_atom = 0; m_atom < kMAtoms; ++m_atom)
                    scale_a[m_atom] = 127u;

#pragma unroll
                for(int m_atom = 0; m_atom < kMAtoms; ++m_atom)
                {
                    const int input_row = min(row_base + m_atom * 16 + lane_row, sub_end - 1);
                    if(row_base + m_atom * 16 < sub_end)
                    {
                        const int activation_k =
                            k_tile * kTileK + lane_group * 16;
                        const auto* activation_row =
                            reinterpret_cast<const uint8_t*>(
                                activations + static_cast<int64_t>(input_row) * K);
                        if(activation_k + 15 < K)
                            *reinterpret_cast<uint4*>(
                                activation_fragments[m_atom].bytes) =
                                *reinterpret_cast<const uint4*>(
                                    activation_row + activation_k);
                        if(activation_k + 64 + 15 < K)
                            *reinterpret_cast<uint4*>(
                                activation_fragments[m_atom].bytes + 16) =
                                *reinterpret_cast<const uint4*>(
                                    activation_row + activation_k + 64);

                        const int scale_column = k_tile * 4 + lane_group;
                        if(scale_column < K / kScaleBlock)
                        {
                            const uint32_t exponent = activation_scales[
                                static_cast<int64_t>(input_row) *
                                    (K / kScaleBlock) +
                                scale_column];
                            scale_a[m_atom] = exponent;
                        }
                    }
                }

                const IQ2RCompressedTriplet compressed = pending;
                if(k_tile + 1 < k_tiles)
                {
                    pending = iq2r_scheduled_load_compact_uniform(data, data_bytes, next_data_base, lane);
                    next_data_base += kGroupBytes;
                }
                opus::i32x8_t weight_fragments[kNAtomsPerWave];
                const uint32_t scale_b = iq2r_decode_direct_lds_triplet(
                    compressed,
                    shared.codebook,
                    packed_bases,
                    weight_fragments);
#pragma unroll
                for(int m_atom = 0; m_atom < kMAtoms; ++m_atom)
                    asm volatile("" : "+v"(scale_a[m_atom]));
#pragma unroll
                for(int m_atom = 0; m_atom < kMAtoms; ++m_atom)
                    if(row_base + m_atom * 16 < sub_end)
                    iq2r_cooperative_triplet_mfma<0>(
                        activation_fragments[m_atom].words,
                        weight_fragments,
                        accumulators[m_atom],
                        scale_a[m_atom] * 0x01010101u,
                        scale_b);
            }

#pragma unroll
            for(int m_atom = 0; m_atom < kMAtoms; ++m_atom)
            {
                if(row_base + m_atom * 16 >= sub_end) continue;
#pragma unroll
                for(int atom = 0; atom < kNAtomsPerWave; ++atom)
                {
                    const int output_column =
                        (n_block_base + atom) * kTileN + lane_row;
#pragma unroll
                    for(int item = 0; item < 4; ++item)
                    {
                        const int output_row =
                            row_base + m_atom * 16 + lane_group * 4 + item;
                        if(output_row < sub_end && output_column < N)
                        {
                            float value = accumulators[m_atom][atom][item];
                            if(all_bias != nullptr)
                                value += __bfloat162float(all_bias[
                                    static_cast<int64_t>(expert_index) * N +
                                    output_column]);
                            output[static_cast<int64_t>(output_row) * N +
                                   output_column] = __float2bfloat16(value);
                        }
                    }
                }
            }
        }
    }
#endif
}
template<int Tokens>
__global__ __launch_bounds__(576,1) void e141_down_grouped_routes(
    const opus::fp8_t* __restrict__ activations,const uint8_t* __restrict__ scales,
    const uint8_t* __restrict__ all_data,const uint8_t* __restrict__ all_auxiliary,
    const int32_t* __restrict__ expert_ids,const int32_t* __restrict__ scatter,
    const float* __restrict__ route_weights,__hip_bfloat16* __restrict__ output,
    int data_bytes,int auxiliary_bytes,int total_tokens)
{
#if defined(__gfx950__)
    static_assert(Tokens==2 || Tokens==4);
    constexpr int Atoms=3,K=256,N=6144;
    struct Storage {
        alignas(16) uint64_t codebook[9][kCodebookBytes/8];
        alignas(16) __hip_bfloat16 routes[Tokens][9][48];
        int experts[Tokens*9];
        int rows[Tokens*9];
    };
    __shared__ Storage shared;
    const int lane=threadIdx.x,wave=threadIdx.y,linear=wave*64+lane;
    const int lane_row=lane%16,lane_group=lane/16;
    const int group=blockIdx.y,group_stride=total_tokens/Tokens;
    const int n_block=static_cast<int>(blockIdx.x)*Atoms;
    if(linear<Tokens*9)
    {
        const int original=(group+(linear/9)*group_stride)*9+linear%9;
        shared.experts[linear]=expert_ids[original];
        shared.rows[linear]=scatter[original];
    }
    __syncthreads();
    const int lane_expert=lane<Tokens*9?shared.experts[lane]:-1;
#pragma unroll
    for(int phase=0;phase<Tokens;++phase)
    {
        const int expert=shared.experts[phase*9+wave];
        const uint64_t matches=__ballot(lane<Tokens*9 && lane_expert==expert);
        const bool active=(matches & ((uint64_t(1)<<(phase*9))-1))==0;
        int route_for_token[Tokens];
#pragma unroll
        for(int m=0;m<Tokens;++m)
        {
            const uint32_t mask=static_cast<uint32_t>((matches>>(m*9))&511u);
            route_for_token[m]=mask?__ffs(mask)-1:-1;
        }
        const uint8_t* data=all_data+static_cast<int64_t>(expert)*data_bytes;
        const uint8_t* aux=all_auxiliary+static_cast<int64_t>(expert)*auxiliary_bytes;
        if(active)
            for(int entry=lane;entry<kCodebookBytes/16;entry+=64)
                reinterpret_cast<uint4*>(shared.codebook[wave])[entry]=reinterpret_cast<const uint4*>(aux)[entry];
        __syncthreads();
        if(active)
        {
            const int m=min(lane_row,Tokens-1);
            const int route=route_for_token[m];
            const int row=route>=0?shared.rows[m*9+route]:shared.rows[phase*9+wave];
            opus::vector_t<float,4> sums[2][Atoms]={};
#pragma unroll
            for(int ktile=0;ktile<2;++ktile)
            {
                const int base=static_cast<int>(triplet_base(physical_tile(n_block,ktile,2)));
                const auto compressed=iq2r_scheduled_load_compact_uniform(data,data_bytes,base,lane);
                IQ2RActivationFragment a={};
                const auto* input=reinterpret_cast<const uint8_t*>(activations+static_cast<int64_t>(row)*K);
                const int ak=ktile*128+lane_group*16;
                *reinterpret_cast<uint4*>(a.bytes)=*reinterpret_cast<const uint4*>(input+ak);
                *reinterpret_cast<uint4*>(a.bytes+16)=*reinterpret_cast<const uint4*>(input+ak+64);
                uint32_t sa=scales[static_cast<int64_t>(row)*8+ktile*4+lane_group],bases=0;
#pragma unroll
                for(int atom=0;atom<3;++atom)bases|=static_cast<uint32_t>(aux[kCodebookBytes+n_block+atom])<<(atom*8);
                opus::i32x8_t b[3];
                const uint32_t sb=iq2r_decode_direct_lds_triplet(compressed,shared.codebook[wave],bases,b);
                asm volatile("" : "+v"(sa));
                iq2r_cooperative_triplet_mfma<0>(a.words,b,sums[ktile],sa*0x01010101u,sb);
            }
            if(lane_group==0)
            {
#pragma unroll
                for(int m=0;m<Tokens;++m)
                {
                    const int route=route_for_token[m];
                    if(route>=0)
                    {
#pragma unroll
                        for(int atom=0;atom<Atoms;++atom)
                            shared.routes[m][route][atom*16+lane_row]=__float2bfloat16(sums[0][atom][m]+sums[1][atom][m]);
                    }
                }
            }
        }
        __syncthreads();
    }
    if(wave<Tokens && lane_group==0)
    {
        const int token=group+wave*group_stride;
#pragma unroll
        for(int atom=0;atom<Atoms;++atom)
        {
            float combined=0.0f;
#pragma unroll
            for(int route=0;route<9;++route)
                combined=fmaf(__bfloat162float(shared.routes[wave][route][atom*16+lane_row]),route_weights[token*9+route],combined);
            output[static_cast<int64_t>(token)*N+(n_block+atom)*16+lane_row]=__float2bfloat16(combined);
        }
    }
#endif
}

__device__ __forceinline__ int e150_remap8(int index,int total)
{
 const int per=(total+7)/8,tall=total%8==0?8:total%8;
 const int xcd=index%8,local=index/8;
 return xcd<tall?xcd*per+local:tall*per+(xcd-tall)*(per-1)+local;
}

template<int MAtoms>
__global__ __launch_bounds__(512,1) void iq2r_gate_quad_sparse_kernel_xcd(
    const opus::fp8_t* __restrict__ activations,
    const uint8_t* __restrict__ scales,
    const uint8_t* __restrict__ all_data,
    const uint8_t* __restrict__ all_auxiliary,
    const int32_t* __restrict__ tasks,
    const int32_t* __restrict__ task_count,
    const int32_t* __restrict__ gather,
    opus::fp8_t* __restrict__ output,
    uint8_t* __restrict__ output_scales,
    int routes,int data_bytes,int auxiliary_bytes)
{
#if defined(__gfx950__)
    constexpr int K=6144, Rows=16*MAtoms, Subtasks=2/MAtoms;
    struct Storage {
        alignas(16) uint64_t codebook[kCodebookBytes/8];
        union {
            alignas(16) uint8_t cache[8][kQuadBytes];
            alignas(16) opus::vector_t<float,4> partial[8][4][64];
        } reuse;
        alignas(16) __hip_bfloat16 gate[Rows][64];
    };
    __shared__ Storage shared;
    const int lane=threadIdx.x,wave=threadIdx.y,linear=wave*64+lane;
    const int lane_row=lane%16,lane_group=lane/16;
    const int num_tasks=task_count[0];
    const int total=num_tasks*Subtasks*8;
    for(int work=blockIdx.x;work<total;work+=gridDim.x)
    {
        const int index=e150_remap8(work,total);
        const int task=index/(8*Subtasks);
        const int sub=(index/8)%Subtasks;
        const int n_tile=index%8;
        const int row_begin=tasks[task*3]+sub*Rows;
        const int row_end=min(row_begin+Rows,tasks[task*3]+tasks[task*3+1]);
        const int expert=tasks[task*3+2];
        const int active_m=(row_end-row_begin+15)/16;
        if(row_begin>=row_end || row_begin<0 || row_end>routes || expert<0 || expert>=257) continue;
        const uint8_t* data=all_data+static_cast<int64_t>(expert)*data_bytes;
        const uint8_t* aux=all_auxiliary+static_cast<int64_t>(expert)*auxiliary_bytes;
        // Previous work ends with a barrier protecting both union and codebook.
        shared.codebook[linear]=reinterpret_cast<const uint64_t*>(aux)[linear];
        __syncthreads();
        const int nbase=n_tile*4;
        uint32_t bases=0;
#pragma unroll
        for(int atom=0;atom<4;++atom)
            bases|=static_cast<uint32_t>(aux[kCodebookBytes+nbase+atom])<<(atom*8);
        int rows[MAtoms];
#pragma unroll
        for(int m=0;m<MAtoms;++m) rows[m]=gather[min(row_begin+m*16+lane_row,row_end-1)]/9;
        opus::vector_t<float,4> accumulators[MAtoms][4]={};
        opus::gmem<uint8_t> buffer(data,static_cast<unsigned int>(data_bytes));
        int base=(n_tile*48+wave*6)*kQuadBytes;
        iq2r_issue_quad(buffer,shared.reuse.cache[wave],lane,base);
        for(int iteration=0;iteration<6;++iteration)
        {
            const int kt=wave*6+iteration;
            IQ2RActivationFragment a[MAtoms]={};uint32_t sa[MAtoms];
#pragma unroll
            for(int m=0;m<MAtoms;++m)
            {
                if(m>=active_m) continue;
                const auto* input=reinterpret_cast<const uint8_t*>(activations+static_cast<int64_t>(rows[m])*K);
                const int ak=kt*128+lane_group*16;
                *reinterpret_cast<uint4*>(a[m].bytes)=*reinterpret_cast<const uint4*>(input+ak);
                *reinterpret_cast<uint4*>(a[m].bytes+16)=*reinterpret_cast<const uint4*>(input+ak+64);
                sa[m]=scales[static_cast<int64_t>(rows[m])*192+kt*4+lane_group]*0x01010101u;
            }
            iq2r_wait_vmcnt<0>();
            const auto compressed=iq2r_read_quad(shared.reuse.cache[wave],lane);
            asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
            if(iteration<5)
            {
                base+=kQuadBytes;
                iq2r_issue_quad(buffer,shared.reuse.cache[wave],lane,base);
            }
            iq2r_quad_sparse_decode_mfma<MAtoms>(compressed,shared.codebook,bases,a,sa,accumulators,active_m);
        }
        // No wave may overwrite another wave's cache before its final read.
        __syncthreads();
#pragma unroll
        for(int m=0;m<MAtoms;++m)
        {
            if(m>=active_m) continue;
#pragma unroll
            for(int atom=0;atom<4;++atom) shared.reuse.partial[wave][atom][lane]=accumulators[m][atom];
            __syncthreads();
            if(wave<4)
            {
                const int local_row=m*16+lane_group*4+wave;
#pragma unroll
                for(int atom=0;atom<4;++atom)
                {
                    float value=0.0f;
#pragma unroll
                    for(int source=0;source<8;++source) value+=shared.reuse.partial[source][atom][lane][wave];
                    shared.gate[local_row][atom*16+lane_row]=__float2bfloat16(value);
                }
            }
            __syncthreads();
        }
        if(linear<active_m*16*4)
        {
            const int local_row=linear/4,quant_lane=linear%4;
            const int output_row=row_begin+local_row;
            using B=opus::vector_t<opus::bf16_t,8>;
            using Q=opus::vector_t<opus::fp8_t,8>;
            B values;float abs_max=1.0e-10f;
#pragma unroll
            for(int element=0;element<8;++element)
            {
                const int column=quant_lane*8+element;
                const float gate=__bfloat162float(shared.gate[local_row][2*column]);
                const float up=__bfloat162float(shared.gate[local_row][2*column+1]);
                const float swish=gate/(1.0f+__expf(-gate));
                const __hip_bfloat16 rounded=__float2bfloat16(swish*(up+0.0f));
                values[element]=__builtin_bit_cast(opus::bf16_t,rounded);
                abs_max=fmaxf(abs_max,fabsf(__bfloat162float(rounded)));
            }
            abs_max=fmaxf(abs_max,__shfl_xor(abs_max,1));
            abs_max=fmaxf(abs_max,__shfl_xor(abs_max,2));
            if(output_row<row_end)
            {
                const auto bs=fp_f32_to_e8m0_block_scale<kDefaultMxScaleRoundMode,MxDtype::FP8_E4M3>(abs_max);
                const float inverse=1.0f/bs.dq_scale;
                Q q;
#pragma unroll
                for(int element=0;element<8;++element) q[element]=opus::fp32_to_fp8(static_cast<float>(values[element])*inverse);
                *reinterpret_cast<Q*>(output+static_cast<int64_t>(output_row)*256+n_tile*32+quant_lane*8)=q;
                if(quant_lane==0) output_scales[static_cast<int64_t>(output_row)*8+n_tile]=bs.byte;
            }
        }
        __syncthreads();
    }
#endif
}
template<int MAtoms,int TaskRows>
__global__ __launch_bounds__(512,1) void iq2r_gate_quad_sparse_kernel_tp4_xcd(
    const opus::fp8_t* __restrict__ activations,
    const uint8_t* __restrict__ scales,
    const uint8_t* __restrict__ all_data,
    const uint8_t* __restrict__ all_auxiliary,
    const int32_t* __restrict__ tasks,
    const int32_t* __restrict__ task_count,
    const int32_t* __restrict__ gather,
    opus::fp8_t* __restrict__ output,
    uint8_t* __restrict__ output_scales,
    int routes,int data_bytes,int auxiliary_bytes)
{
#if defined(__gfx950__)
    constexpr int K=6144, Rows=16*MAtoms, Subtasks=TaskRows/(16*MAtoms);
    struct Storage {
        alignas(16) uint64_t codebook[kCodebookBytes/8];
        union {
            alignas(16) uint8_t cache[8][kQuadBytes];
            alignas(16) opus::vector_t<float,4> partial[8][4][64];
        } reuse;
        alignas(16) __hip_bfloat16 gate[Rows][64];
    };
    __shared__ Storage shared;
    const int lane=threadIdx.x,wave=threadIdx.y,linear=wave*64+lane;
    const int lane_row=lane%16,lane_group=lane/16;
    const int num_tasks=task_count[0];
    const int total=num_tasks*Subtasks*16;
    for(int work=blockIdx.x;work<total;work+=gridDim.x)
    {
        const int index=e150_remap8(work,total);
        const int task=index/(16*Subtasks);
        const int sub=(index/16)%Subtasks;
        const int n_tile=index%16;
        const int row_begin=tasks[task*3]+sub*Rows;
        const int row_end=min(row_begin+Rows,tasks[task*3]+tasks[task*3+1]);
        const int expert=tasks[task*3+2];
        const int active_m=(row_end-row_begin+15)/16;
        if(row_begin>=row_end || row_begin<0 || row_end>routes || expert<0 || expert>=257) continue;
        const uint8_t* data=all_data+static_cast<int64_t>(expert)*data_bytes;
        const uint8_t* aux=all_auxiliary+static_cast<int64_t>(expert)*auxiliary_bytes;
        // Previous work ends with a barrier protecting both union and codebook.
        shared.codebook[linear]=reinterpret_cast<const uint64_t*>(aux)[linear];
        __syncthreads();
        const int nbase=n_tile*4;
        uint32_t bases=0;
#pragma unroll
        for(int atom=0;atom<4;++atom)
            bases|=static_cast<uint32_t>(aux[kCodebookBytes+nbase+atom])<<(atom*8);
        int rows[MAtoms];
#pragma unroll
        for(int m=0;m<MAtoms;++m) rows[m]=gather[min(row_begin+m*16+lane_row,row_end-1)]/9;
        opus::vector_t<float,4> accumulators[MAtoms][4]={};
        opus::gmem<uint8_t> buffer(data,static_cast<unsigned int>(data_bytes));
        int base=(n_tile*48+wave*6)*kQuadBytes;
        iq2r_issue_quad(buffer,shared.reuse.cache[wave],lane,base);
        for(int iteration=0;iteration<6;++iteration)
        {
            const int kt=wave*6+iteration;
            IQ2RActivationFragment a[MAtoms]={};uint32_t sa[MAtoms];
#pragma unroll
            for(int m=0;m<MAtoms;++m)
            {
                if(m>=active_m) continue;
                const auto* input=reinterpret_cast<const uint8_t*>(activations+static_cast<int64_t>(rows[m])*K);
                const int ak=kt*128+lane_group*16;
                *reinterpret_cast<uint4*>(a[m].bytes)=*reinterpret_cast<const uint4*>(input+ak);
                *reinterpret_cast<uint4*>(a[m].bytes+16)=*reinterpret_cast<const uint4*>(input+ak+64);
                sa[m]=scales[static_cast<int64_t>(rows[m])*192+kt*4+lane_group]*0x01010101u;
            }
            iq2r_wait_vmcnt<0>();
            const auto compressed=iq2r_read_quad(shared.reuse.cache[wave],lane);
            asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
            if(iteration<5)
            {
                base+=kQuadBytes;
                iq2r_issue_quad(buffer,shared.reuse.cache[wave],lane,base);
            }
            iq2r_quad_sparse_decode_mfma<MAtoms>(compressed,shared.codebook,bases,a,sa,accumulators,active_m);
        }
        // No wave may overwrite another wave's cache before its final read.
        __syncthreads();
#pragma unroll
        for(int m=0;m<MAtoms;++m)
        {
            if(m>=active_m) continue;
#pragma unroll
            for(int atom=0;atom<4;++atom) shared.reuse.partial[wave][atom][lane]=accumulators[m][atom];
            __syncthreads();
            if(wave<4)
            {
                const int local_row=m*16+lane_group*4+wave;
#pragma unroll
                for(int atom=0;atom<4;++atom)
                {
                    float value=0.0f;
#pragma unroll
                    for(int source=0;source<8;++source) value+=shared.reuse.partial[source][atom][lane][wave];
                    shared.gate[local_row][atom*16+lane_row]=__float2bfloat16(value);
                }
            }
            __syncthreads();
        }
        if(linear<active_m*16*4)
        {
            const int local_row=linear/4,quant_lane=linear%4;
            const int output_row=row_begin+local_row;
            using B=opus::vector_t<opus::bf16_t,8>;
            using Q=opus::vector_t<opus::fp8_t,8>;
            B values;float abs_max=1.0e-10f;
#pragma unroll
            for(int element=0;element<8;++element)
            {
                const int column=quant_lane*8+element;
                const float gate=__bfloat162float(shared.gate[local_row][2*column]);
                const float up=__bfloat162float(shared.gate[local_row][2*column+1]);
                const float swish=gate/(1.0f+__expf(-gate));
                const __hip_bfloat16 rounded=__float2bfloat16(swish*(up+0.0f));
                values[element]=__builtin_bit_cast(opus::bf16_t,rounded);
                abs_max=fmaxf(abs_max,fabsf(__bfloat162float(rounded)));
            }
            abs_max=fmaxf(abs_max,__shfl_xor(abs_max,1));
            abs_max=fmaxf(abs_max,__shfl_xor(abs_max,2));
            if(output_row<row_end)
            {
                const auto bs=fp_f32_to_e8m0_block_scale<kDefaultMxScaleRoundMode,MxDtype::FP8_E4M3>(abs_max);
                const float inverse=1.0f/bs.dq_scale;
                Q q;
#pragma unroll
                for(int element=0;element<8;++element) q[element]=opus::fp32_to_fp8(static_cast<float>(values[element])*inverse);
                *reinterpret_cast<Q*>(output+static_cast<int64_t>(output_row)*512+n_tile*32+quant_lane*8)=q;
                if(quant_lane==0) output_scales[static_cast<int64_t>(output_row)*16+n_tile]=bs.byte;
            }
        }
        __syncthreads();
    }
#endif
}
template<int TaskRows>
__global__ __launch_bounds__(512,1) void e127_m32_register(
    const opus::fp8_t* __restrict__ activations,
    const uint8_t* __restrict__ scales,
    const uint8_t* __restrict__ all_data,
    const uint8_t* __restrict__ all_auxiliary,
    const int32_t* __restrict__ tasks,
    const int32_t* __restrict__ task_count,
    const int32_t* __restrict__ gather,
    opus::fp8_t* __restrict__ output,
    uint8_t* __restrict__ output_scales,
    int routes,int data_bytes,int auxiliary_bytes)
{
#if defined(__gfx950__)
    constexpr int MAtoms=2;
    constexpr int K=6144, Rows=16*MAtoms, Subtasks=TaskRows/(16*MAtoms);
    struct Storage {
        alignas(16) uint64_t codebook[kCodebookBytes/8];
        union {
            alignas(16) uint8_t cache[8][kQuadBytes];
            alignas(16) opus::vector_t<float,4> partial[8][4][64];
        } reuse;
        alignas(16) __hip_bfloat16 gate[Rows][64];
    };
    __shared__ Storage shared;
    const int lane=threadIdx.x,wave=threadIdx.y,linear=wave*64+lane;
    const int lane_row=lane%16,lane_group=lane/16;
    const int num_tasks=task_count[0];
    const int total=num_tasks*Subtasks*8;
    for(int work=blockIdx.x;work<total;work+=gridDim.x)
    {
        const int task=(work/Subtasks)%num_tasks;
        const int sub=work%Subtasks;
        const int n_tile=work/(num_tasks*Subtasks);
        const int row_begin=tasks[task*3]+sub*Rows;
        const int row_end=min(row_begin+Rows,tasks[task*3]+tasks[task*3+1]);
        const int expert=tasks[task*3+2];
        const int active_m=(row_end-row_begin+15)/16;
        if(row_begin>=row_end || row_begin<0 || row_end>routes || expert<0 || expert>=257) continue;
        const uint8_t* data=all_data+static_cast<int64_t>(expert)*data_bytes;
        const uint8_t* aux=all_auxiliary+static_cast<int64_t>(expert)*auxiliary_bytes;
        // Previous work ends with a barrier protecting both union and codebook.
        shared.codebook[linear]=reinterpret_cast<const uint64_t*>(aux)[linear];
        __syncthreads();
        const int nbase=n_tile*4;
        uint32_t bases=0;
#pragma unroll
        for(int atom=0;atom<4;++atom)
            bases|=static_cast<uint32_t>(aux[kCodebookBytes+nbase+atom])<<(atom*8);
        int rows[MAtoms];
#pragma unroll
        for(int m=0;m<MAtoms;++m) rows[m]=gather[min(row_begin+m*16+lane_row,row_end-1)]/9;
        opus::vector_t<float,4> accumulators[MAtoms][4]={};
        opus::gmem<uint8_t> buffer(data,static_cast<unsigned int>(data_bytes));
        int base=(n_tile*48+wave*6)*kQuadBytes;
        IQ2RCompressedQuad pending=iq2r_scheduled_load_quad(data,data_bytes,base,lane);
        for(int iteration=0;iteration<6;++iteration)
        {
            const int kt=wave*6+iteration;
            IQ2RActivationFragment a[MAtoms]={};uint32_t sa[MAtoms];
#pragma unroll
            for(int m=0;m<MAtoms;++m)
            {
                if(m>=active_m) continue;
                const auto* input=reinterpret_cast<const uint8_t*>(activations+static_cast<int64_t>(rows[m])*K);
                const int ak=kt*128+lane_group*16;
                *reinterpret_cast<uint4*>(a[m].bytes)=*reinterpret_cast<const uint4*>(input+ak);
                *reinterpret_cast<uint4*>(a[m].bytes+16)=*reinterpret_cast<const uint4*>(input+ak+64);
                sa[m]=scales[static_cast<int64_t>(rows[m])*192+kt*4+lane_group]*0x01010101u;
            }
            const auto compressed=pending;
            if(iteration<5)
            {
                base+=kQuadBytes;
                pending=iq2r_scheduled_load_quad(data,data_bytes,base,lane);
            }
            iq2r_quad_sparse_decode_mfma<MAtoms>(compressed,shared.codebook,bases,a,sa,accumulators,active_m);
        }
        // No wave may overwrite another wave's cache before its final read.
        __syncthreads();
#pragma unroll
        for(int m=0;m<MAtoms;++m)
        {
            if(m>=active_m) continue;
#pragma unroll
            for(int atom=0;atom<4;++atom) shared.reuse.partial[wave][atom][lane]=accumulators[m][atom];
            __syncthreads();
            if(wave<4)
            {
                const int local_row=m*16+lane_group*4+wave;
#pragma unroll
                for(int atom=0;atom<4;++atom)
                {
                    float value=0.0f;
#pragma unroll
                    for(int source=0;source<8;++source) value+=shared.reuse.partial[source][atom][lane][wave];
                    shared.gate[local_row][atom*16+lane_row]=__float2bfloat16(value);
                }
            }
            __syncthreads();
        }
        if(linear<active_m*16*4)
        {
            const int local_row=linear/4,quant_lane=linear%4;
            const int output_row=row_begin+local_row;
            using B=opus::vector_t<opus::bf16_t,8>;
            using Q=opus::vector_t<opus::fp8_t,8>;
            B values;float abs_max=1.0e-10f;
#pragma unroll
            for(int element=0;element<8;++element)
            {
                const int column=quant_lane*8+element;
                const float gate=__bfloat162float(shared.gate[local_row][2*column]);
                const float up=__bfloat162float(shared.gate[local_row][2*column+1]);
                const float swish=gate/(1.0f+__expf(-gate));
                const __hip_bfloat16 rounded=__float2bfloat16(swish*(up+0.0f));
                values[element]=__builtin_bit_cast(opus::bf16_t,rounded);
                abs_max=fmaxf(abs_max,fabsf(__bfloat162float(rounded)));
            }
            abs_max=fmaxf(abs_max,__shfl_xor(abs_max,1));
            abs_max=fmaxf(abs_max,__shfl_xor(abs_max,2));
            if(output_row<row_end)
            {
                const auto bs=fp_f32_to_e8m0_block_scale<kDefaultMxScaleRoundMode,MxDtype::FP8_E4M3>(abs_max);
                const float inverse=1.0f/bs.dq_scale;
                Q q;
#pragma unroll
                for(int element=0;element<8;++element) q[element]=opus::fp32_to_fp8(static_cast<float>(values[element])*inverse);
                *reinterpret_cast<Q*>(output+static_cast<int64_t>(output_row)*256+n_tile*32+quant_lane*8)=q;
                if(quant_lane==0) output_scales[static_cast<int64_t>(output_row)*8+n_tile]=bs.byte;
            }
        }
        __syncthreads();
    }
#endif
}

template<int TaskRows>
__global__ __launch_bounds__(512,1) void e127_m64_register(
    const opus::fp8_t* __restrict__ activations,
    const uint8_t* __restrict__ scales,
    const uint8_t* __restrict__ all_data,
    const uint8_t* __restrict__ all_auxiliary,
    const int32_t* __restrict__ tasks,
    const int32_t* __restrict__ task_count,
    const int32_t* __restrict__ gather,
    opus::fp8_t* __restrict__ output,
    uint8_t* __restrict__ output_scales,
    int routes,int data_bytes,int auxiliary_bytes)
{
#if defined(__gfx950__)
    constexpr int MAtoms=4;
    constexpr int K=6144, Rows=16*MAtoms, Subtasks=TaskRows/(16*MAtoms);
    struct Storage {
        alignas(16) uint64_t codebook[kCodebookBytes/8];
        union {
            alignas(16) uint8_t cache[8][kQuadBytes];
            alignas(16) opus::vector_t<float,4> partial[8][4][64];
        } reuse;
        alignas(16) __hip_bfloat16 gate[Rows][64];
    };
    __shared__ Storage shared;
    const int lane=threadIdx.x,wave=threadIdx.y,linear=wave*64+lane;
    const int lane_row=lane%16,lane_group=lane/16;
    const int num_tasks=task_count[0];
    const int total=num_tasks*Subtasks*8;
    for(int work=blockIdx.x;work<total;work+=gridDim.x)
    {
        const int task=(work/Subtasks)%num_tasks;
        const int sub=work%Subtasks;
        const int n_tile=work/(num_tasks*Subtasks);
        const int row_begin=tasks[task*3]+sub*Rows;
        const int row_end=min(row_begin+Rows,tasks[task*3]+tasks[task*3+1]);
        const int expert=tasks[task*3+2];
        const int active_m=(row_end-row_begin+15)/16;
        if(row_begin>=row_end || row_begin<0 || row_end>routes || expert<0 || expert>=257) continue;
        const uint8_t* data=all_data+static_cast<int64_t>(expert)*data_bytes;
        const uint8_t* aux=all_auxiliary+static_cast<int64_t>(expert)*auxiliary_bytes;
        // Previous work ends with a barrier protecting both union and codebook.
        shared.codebook[linear]=reinterpret_cast<const uint64_t*>(aux)[linear];
        __syncthreads();
        const int nbase=n_tile*4;
        uint32_t bases=0;
#pragma unroll
        for(int atom=0;atom<4;++atom)
            bases|=static_cast<uint32_t>(aux[kCodebookBytes+nbase+atom])<<(atom*8);
        int rows[MAtoms];
#pragma unroll
        for(int m=0;m<MAtoms;++m) rows[m]=gather[min(row_begin+m*16+lane_row,row_end-1)]/9;
        opus::vector_t<float,4> accumulators[MAtoms][4]={};
        opus::gmem<uint8_t> buffer(data,static_cast<unsigned int>(data_bytes));
        int base=(n_tile*48+wave*6)*kQuadBytes;
        IQ2RCompressedQuad pending=iq2r_scheduled_load_quad(data,data_bytes,base,lane);
        for(int iteration=0;iteration<6;++iteration)
        {
            const int kt=wave*6+iteration;
            IQ2RActivationFragment a[MAtoms]={};uint32_t sa[MAtoms];
#pragma unroll
            for(int m=0;m<MAtoms;++m)
            {
                if(m>=active_m) continue;
                const auto* input=reinterpret_cast<const uint8_t*>(activations+static_cast<int64_t>(rows[m])*K);
                const int ak=kt*128+lane_group*16;
                *reinterpret_cast<uint4*>(a[m].bytes)=*reinterpret_cast<const uint4*>(input+ak);
                *reinterpret_cast<uint4*>(a[m].bytes+16)=*reinterpret_cast<const uint4*>(input+ak+64);
                sa[m]=scales[static_cast<int64_t>(rows[m])*192+kt*4+lane_group]*0x01010101u;
            }
            const auto compressed=pending;
            if(iteration<5)
            {
                base+=kQuadBytes;
                pending=iq2r_scheduled_load_quad(data,data_bytes,base,lane);
            }
            iq2r_quad_sparse_decode_mfma<MAtoms>(compressed,shared.codebook,bases,a,sa,accumulators,active_m);
        }
        // No wave may overwrite another wave's cache before its final read.
        __syncthreads();
#pragma unroll
        for(int m=0;m<MAtoms;++m)
        {
            if(m>=active_m) continue;
#pragma unroll
            for(int atom=0;atom<4;++atom) shared.reuse.partial[wave][atom][lane]=accumulators[m][atom];
            __syncthreads();
            if(wave<4)
            {
                const int local_row=m*16+lane_group*4+wave;
#pragma unroll
                for(int atom=0;atom<4;++atom)
                {
                    float value=0.0f;
#pragma unroll
                    for(int source=0;source<8;++source) value+=shared.reuse.partial[source][atom][lane][wave];
                    shared.gate[local_row][atom*16+lane_row]=__float2bfloat16(value);
                }
            }
            __syncthreads();
        }
        if(linear<active_m*16*4)
        {
            const int local_row=linear/4,quant_lane=linear%4;
            const int output_row=row_begin+local_row;
            using B=opus::vector_t<opus::bf16_t,8>;
            using Q=opus::vector_t<opus::fp8_t,8>;
            B values;float abs_max=1.0e-10f;
#pragma unroll
            for(int element=0;element<8;++element)
            {
                const int column=quant_lane*8+element;
                const float gate=__bfloat162float(shared.gate[local_row][2*column]);
                const float up=__bfloat162float(shared.gate[local_row][2*column+1]);
                const float swish=gate/(1.0f+__expf(-gate));
                const __hip_bfloat16 rounded=__float2bfloat16(swish*(up+0.0f));
                values[element]=__builtin_bit_cast(opus::bf16_t,rounded);
                abs_max=fmaxf(abs_max,fabsf(__bfloat162float(rounded)));
            }
            abs_max=fmaxf(abs_max,__shfl_xor(abs_max,1));
            abs_max=fmaxf(abs_max,__shfl_xor(abs_max,2));
            if(output_row<row_end)
            {
                const auto bs=fp_f32_to_e8m0_block_scale<kDefaultMxScaleRoundMode,MxDtype::FP8_E4M3>(abs_max);
                const float inverse=1.0f/bs.dq_scale;
                Q q;
#pragma unroll
                for(int element=0;element<8;++element) q[element]=opus::fp32_to_fp8(static_cast<float>(values[element])*inverse);
                *reinterpret_cast<Q*>(output+static_cast<int64_t>(output_row)*256+n_tile*32+quant_lane*8)=q;
                if(quant_lane==0) output_scales[static_cast<int64_t>(output_row)*8+n_tile]=bs.byte;
            }
        }
        __syncthreads();
    }
#endif
}

template<int TaskRows>
__global__ __launch_bounds__(512,1) void e127_m64_ab_unroll1(
    const opus::fp8_t* __restrict__ activations,
    const uint8_t* __restrict__ scales,
    const uint8_t* __restrict__ all_data,
    const uint8_t* __restrict__ all_auxiliary,
    const int32_t* __restrict__ tasks,
    const int32_t* __restrict__ task_count,
    const int32_t* __restrict__ gather,
    opus::fp8_t* __restrict__ output,
    uint8_t* __restrict__ output_scales,
    int routes,int data_bytes,int auxiliary_bytes)
{
#if defined(__gfx950__)
    constexpr int MAtoms=4;
    constexpr int K=6144, Rows=16*MAtoms, Subtasks=TaskRows/(16*MAtoms);
    struct Storage {
        alignas(16) uint64_t codebook[kCodebookBytes/8];
        union {
            alignas(16) uint8_t cache[8][kQuadBytes];
            alignas(16) opus::vector_t<float,4> partial[8][4][64];
        } reuse;
        alignas(16) __hip_bfloat16 gate[Rows][64];
    };
    __shared__ Storage shared;
    const int lane=threadIdx.x,wave=threadIdx.y,linear=wave*64+lane;
    const int lane_row=lane%16,lane_group=lane/16;
    const int num_tasks=task_count[0];
    const int total=num_tasks*Subtasks*8;
    for(int work=blockIdx.x;work<total;work+=gridDim.x)
    {
        const int task=(work/Subtasks)%num_tasks;
        const int sub=work%Subtasks;
        const int n_tile=work/(num_tasks*Subtasks);
        const int row_begin=tasks[task*3]+sub*Rows;
        const int row_end=min(row_begin+Rows,tasks[task*3]+tasks[task*3+1]);
        const int expert=tasks[task*3+2];
        const int active_m=(row_end-row_begin+15)/16;
        if(row_begin>=row_end || row_begin<0 || row_end>routes || expert<0 || expert>=257) continue;
        const uint8_t* data=all_data+static_cast<int64_t>(expert)*data_bytes;
        const uint8_t* aux=all_auxiliary+static_cast<int64_t>(expert)*auxiliary_bytes;
        // Previous work ends with a barrier protecting both union and codebook.
        shared.codebook[linear]=reinterpret_cast<const uint64_t*>(aux)[linear];
        __syncthreads();
        const int nbase=n_tile*4;
        uint32_t bases=0;
#pragma unroll
        for(int atom=0;atom<4;++atom)
            bases|=static_cast<uint32_t>(aux[kCodebookBytes+nbase+atom])<<(atom*8);
        int rows[MAtoms];
#pragma unroll
        for(int m=0;m<MAtoms;++m) rows[m]=gather[min(row_begin+m*16+lane_row,row_end-1)]/9;
        opus::vector_t<float,4> accumulators[MAtoms][4]={};
        opus::gmem<uint8_t> buffer(data,static_cast<unsigned int>(data_bytes));
        int base=(n_tile*48+wave*6)*kQuadBytes;
        IQ2RCompressedQuad pending=iq2r_scheduled_load_quad(data,data_bytes,base,lane);
        IQ2RActivationFragment pending_a[MAtoms]={};uint32_t pending_sa[MAtoms]={};
#pragma unroll
        for(int m=0;m<MAtoms;++m)
        {
            if(m>=active_m) continue;
            const auto* input=reinterpret_cast<const uint8_t*>(activations+static_cast<int64_t>(rows[m])*K);
            const int ak=wave*6*128+lane_group*16;
            *reinterpret_cast<uint4*>(pending_a[m].bytes)=*reinterpret_cast<const uint4*>(input+ak);
            *reinterpret_cast<uint4*>(pending_a[m].bytes+16)=*reinterpret_cast<const uint4*>(input+ak+64);
            pending_sa[m]=scales[static_cast<int64_t>(rows[m])*192+wave*6*4+lane_group];
        }
#pragma unroll 1
        for(int iteration=0;iteration<6;++iteration)
        {
            const auto compressed=pending;
            IQ2RActivationFragment a[MAtoms];uint32_t sa[MAtoms];
#pragma unroll
            for(int m=0;m<MAtoms;++m) {a[m]=pending_a[m];sa[m]=pending_sa[m]*0x01010101u;}
            if(iteration<5)
            {
                base+=kQuadBytes;
                pending=iq2r_scheduled_load_quad(data,data_bytes,base,lane);
#pragma unroll
                for(int m=0;m<MAtoms;++m)
                {
                    if(m>=active_m) continue;
                    const auto* input=reinterpret_cast<const uint8_t*>(activations+static_cast<int64_t>(rows[m])*K);
                    const int kt=wave*6+iteration+1,ak=kt*128+lane_group*16;
                    *reinterpret_cast<uint4*>(pending_a[m].bytes)=*reinterpret_cast<const uint4*>(input+ak);
                    *reinterpret_cast<uint4*>(pending_a[m].bytes+16)=*reinterpret_cast<const uint4*>(input+ak+64);
                    pending_sa[m]=scales[static_cast<int64_t>(rows[m])*192+kt*4+lane_group];
                }
            }
            iq2r_quad_sparse_decode_mfma<MAtoms>(compressed,shared.codebook,bases,a,sa,accumulators,active_m);
        }
        // No wave may overwrite another wave's cache before its final read.
        __syncthreads();
#pragma unroll
        for(int m=0;m<MAtoms;++m)
        {
            if(m>=active_m) continue;
#pragma unroll
            for(int atom=0;atom<4;++atom) shared.reuse.partial[wave][atom][lane]=accumulators[m][atom];
            __syncthreads();
            if(wave<4)
            {
                const int local_row=m*16+lane_group*4+wave;
#pragma unroll
                for(int atom=0;atom<4;++atom)
                {
                    float value=0.0f;
#pragma unroll
                    for(int source=0;source<8;++source) value+=shared.reuse.partial[source][atom][lane][wave];
                    shared.gate[local_row][atom*16+lane_row]=__float2bfloat16(value);
                }
            }
            __syncthreads();
        }
        if(linear<active_m*16*4)
        {
            const int local_row=linear/4,quant_lane=linear%4;
            const int output_row=row_begin+local_row;
            using B=opus::vector_t<opus::bf16_t,8>;
            using Q=opus::vector_t<opus::fp8_t,8>;
            B values;float abs_max=1.0e-10f;
#pragma unroll
            for(int element=0;element<8;++element)
            {
                const int column=quant_lane*8+element;
                const float gate=__bfloat162float(shared.gate[local_row][2*column]);
                const float up=__bfloat162float(shared.gate[local_row][2*column+1]);
                const float swish=gate/(1.0f+__expf(-gate));
                const __hip_bfloat16 rounded=__float2bfloat16(swish*(up+0.0f));
                values[element]=__builtin_bit_cast(opus::bf16_t,rounded);
                abs_max=fmaxf(abs_max,fabsf(__bfloat162float(rounded)));
            }
            abs_max=fmaxf(abs_max,__shfl_xor(abs_max,1));
            abs_max=fmaxf(abs_max,__shfl_xor(abs_max,2));
            if(output_row<row_end)
            {
                const auto bs=fp_f32_to_e8m0_block_scale<kDefaultMxScaleRoundMode,MxDtype::FP8_E4M3>(abs_max);
                const float inverse=1.0f/bs.dq_scale;
                Q q;
#pragma unroll
                for(int element=0;element<8;++element) q[element]=opus::fp32_to_fp8(static_cast<float>(values[element])*inverse);
                *reinterpret_cast<Q*>(output+static_cast<int64_t>(output_row)*256+n_tile*32+quant_lane*8)=q;
                if(quant_lane==0) output_scales[static_cast<int64_t>(output_row)*8+n_tile]=bs.byte;
            }
        }
        __syncthreads();
    }
#endif
}
__global__ __launch_bounds__(768,1) void e172_compact_down_routes(
    const opus::fp8_t* __restrict__ activations,const uint8_t* __restrict__ scales,
    const uint8_t* __restrict__ all_data,const uint8_t* __restrict__ all_auxiliary,
    const int32_t* __restrict__ expert_ids,const int32_t* __restrict__ scatter,
    const float* __restrict__ route_weights,__hip_bfloat16* __restrict__ output,
    int data_bytes,int auxiliary_bytes,int total_tokens,const int32_t* task_count,const int32_t* task_table)
{
#if defined(__gfx950__)
    const int tasks=task_count[0];
    // At most twelve global experts fit one simultaneous wave per expert.
    // Other patterns retain two-token reuse with inactive extra waves.
    const bool Compact=tasks<=12;
    const int Tokens=Compact?4:2;
    const int Phases=Compact?1:2;
    constexpr int MaxTokens=4;
    constexpr int Atoms=3,K=256,N=6144;
    struct Storage {
        alignas(16) uint64_t codebook[12][kCodebookBytes/8];
        alignas(16) __hip_bfloat16 routes[MaxTokens][9][48];
        int experts[MaxTokens*9];
        int rows[MaxTokens*9];
    };
    __shared__ Storage shared;
    const int lane=threadIdx.x,wave=threadIdx.y,linear=wave*64+lane;
    const int lane_row=lane%16,lane_group=lane/16;
    const int group=blockIdx.y,group_stride=total_tokens/Tokens;
    if(group>=group_stride)return;
    const int n_block=static_cast<int>(blockIdx.x)*Atoms;
    if(linear<Tokens*9)
    {
        const int original=(group+(linear/9)*group_stride)*9+linear%9;
        shared.experts[linear]=expert_ids[original];
        if(shared.experts[linear]<0 || shared.experts[linear]>=257)
            for(int column=0;column<48;++column)
                shared.routes[linear/9][linear%9][column]=__float2bfloat16(0.0f);
        shared.rows[linear]=scatter[original];
    }
    __syncthreads();
    const int lane_expert=lane<Tokens*9?shared.experts[lane]:-1;
#pragma unroll
    for(int phase=0;phase<MaxTokens;++phase)
    {
        if(phase>=Phases)break;
        const int expert=Compact?(wave<tasks?task_table[wave*3+2]:-1):
                                     (wave<9?shared.experts[phase*9+wave]:-1);
        const uint64_t matches=__ballot(lane<Tokens*9 && lane_expert==expert);
        const bool active=expert>=0 && expert<257 &&
            (Compact?(wave<tasks && matches!=0):
                     (wave<9 && (matches & ((uint64_t(1)<<(phase*9))-1))==0));
        int route_for_token[MaxTokens];
#pragma unroll
        for(int m=0;m<MaxTokens;++m)
        {
            if(m>=Tokens)continue;
            const uint32_t mask=static_cast<uint32_t>((matches>>(m*9))&511u);
            route_for_token[m]=mask?__ffs(mask)-1:-1;
        }
        const uint8_t* data=all_data+static_cast<int64_t>(expert)*data_bytes;
        const uint8_t* aux=all_auxiliary+static_cast<int64_t>(expert)*auxiliary_bytes;
        if(active)
            for(int entry=lane;entry<kCodebookBytes/16;entry+=64)
                reinterpret_cast<uint4*>(shared.codebook[wave])[entry]=reinterpret_cast<const uint4*>(aux)[entry];
        __syncthreads();
        if(active)
        {
            const int m=min(lane_row,Tokens-1);
            const int route=route_for_token[m];
            const int row=route>=0?shared.rows[m*9+route]:shared.rows[phase*9+wave];
            opus::vector_t<float,4> sums[2][Atoms]={};
#pragma unroll
            for(int ktile=0;ktile<2;++ktile)
            {
                const int base=static_cast<int>(triplet_base(physical_tile(n_block,ktile,2)));
                const auto compressed=iq2r_scheduled_load_compact_uniform(data,data_bytes,base,lane);
                IQ2RActivationFragment a={};
                const auto* input=reinterpret_cast<const uint8_t*>(activations+static_cast<int64_t>(row)*K);
                const int ak=ktile*128+lane_group*16;
                *reinterpret_cast<uint4*>(a.bytes)=*reinterpret_cast<const uint4*>(input+ak);
                *reinterpret_cast<uint4*>(a.bytes+16)=*reinterpret_cast<const uint4*>(input+ak+64);
                uint32_t sa=scales[static_cast<int64_t>(row)*8+ktile*4+lane_group],bases=0;
#pragma unroll
                for(int atom=0;atom<3;++atom)bases|=static_cast<uint32_t>(aux[kCodebookBytes+n_block+atom])<<(atom*8);
                opus::i32x8_t b[3];
                const uint32_t sb=iq2r_decode_direct_lds_triplet(compressed,shared.codebook[wave],bases,b);
                asm volatile("" : "+v"(sa));
                iq2r_cooperative_triplet_mfma<0>(a.words,b,sums[ktile],sa*0x01010101u,sb);
            }
            if(lane_group==0)
            {
#pragma unroll
                for(int m=0;m<MaxTokens;++m)
                {
                    if(m>=Tokens)continue;
                    const int route=route_for_token[m];
                    if(route>=0)
                    {
#pragma unroll
                        for(int atom=0;atom<Atoms;++atom)
                            shared.routes[m][route][atom*16+lane_row]=__float2bfloat16(sums[0][atom][m]+sums[1][atom][m]);
                    }
                }
            }
        }
        __syncthreads();
    }
    if(wave<Tokens && lane_group==0)
    {
        const int token=group+wave*group_stride;
#pragma unroll
        for(int atom=0;atom<Atoms;++atom)
        {
            float combined=0.0f;
#pragma unroll
            for(int route=0;route<9;++route)
                combined=fmaf(__bfloat162float(shared.routes[wave][route][atom*16+lane_row]),route_weights[token*9+route],combined);
            output[static_cast<int64_t>(token)*N+(n_block+atom)*16+lane_row]=__float2bfloat16(combined);
        }
    }
#endif
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
    const HipDeviceGuard device_guard(device);
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

void iq2r_task_gemm_indexed_out(const aiter_tensor_t& activations,
                        const aiter_tensor_t& activation_scales,
                        const aiter_tensor_t& data,
                        const aiter_tensor_t& auxiliary,
                        const aiter_tensor_t& tasks,
                        const aiter_tensor_t& task_count,
                        std::optional<aiter_tensor_t> bias,
                        aiter_tensor_t& output,
                        int64_t logical_n,
                        int64_t logical_k,
                        int64_t tile_n,
                        const aiter_tensor_t& gather_indices)
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
    const bool tiled_activation_scales = activation_scales.dim() == 4;
    const bool valid_activation_scale_shape =
        (activation_scales.dim() == 2 &&
         activation_scales.size(0) == activations.size(0) &&
         activation_scales.size(1) == logical_k / kScaleBlock) ||
        (tiled_activation_scales &&
         activation_scales.size(0) ==
             (logical_k / kScaleBlock + 3) / 4 &&
         activation_scales.size(1) >= (activations.size(0) + 15) / 16 &&
         activation_scales.size(2) == 4 && activation_scales.size(3) == 16);
    AITER_CHECK(activations.dim() == 2 && activations.size(1) == logical_k &&
                    valid_activation_scale_shape,
                "IQ2R task GEMM activation shapes are invalid");
    AITER_CHECK(output.dim() == 2 && output.size(0) == activations.size(0) * 9 &&
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

    AITER_CHECK(gather_indices.is_gpu() && gather_indices.device_id == data.device_id &&
                    gather_indices.dtype() == AITER_DTYPE_i32 && gather_indices.dim() == 1 &&
                    gather_indices.is_contiguous() && gather_indices.size(0) == output.size(0),
                "E063 gather_indices must be contiguous int32 [routes] on the weight GPU");
    AITER_CHECK(logical_k == 6144 && logical_n == 512 && data.size(0) == 257 &&
                    !tiled_activation_scales,
                "E063 supports only GLM TP8 gate with 257 experts and row-major scales");
    const int routed_m = static_cast<int>(output.size(0));
    const int token_m = static_cast<int>(activations.size(0));
    const bool decode = (token_m >= 64 && token_m <= 227) || token_m == 256;
    const bool prefill = token_m == 1536 || (token_m >= 2048 && token_m <= 4096);
    const int task_rows = prefill ? 64 : 32;
    const int capacity = (routed_m + task_rows - 1) / task_rows + min(routed_m, 258);
    AITER_CHECK((decode || prefill) && tasks.size(0) == capacity,
                "E063 requires the E060 cooperative M32 task policy");
    const char* family = std::getenv("IQ2R_GEMM_GATE_UP_FAMILY");
    AITER_CHECK(family == nullptr || family[0] == '\0' || std::strcmp(family, "3x8m2") == 0,
                "E063 indexed gate supports only the 3x8m2 family");
    int multiplier = prefill ? 10 : 2;
    const char* grid = std::getenv("IQ2R_GEMM_GATE_UP_GRID_MULTIPLIER");
    if(grid != nullptr && grid[0] != '\0')
        multiplier = std::atoi(grid);
    AITER_CHECK(multiplier >= 1 && multiplier <= 16, "E063 grid must be in [1,16]");
    hipLaunchKernelGGL(
        (iq2r_task_gemm_indexed_m32_kernel<8>),
        dim3(multiplier * static_cast<int>(get_num_cu_func())),
        dim3(64, 8), 0, getCurrentHIPStream(),
        static_cast<const opus::fp8_t*>(activations.data_ptr()),
        static_cast<const uint8_t*>(activation_scales.data_ptr()),
        static_cast<const uint8_t*>(data.data_ptr()),
        static_cast<const uint8_t*>(auxiliary.data_ptr()),
        static_cast<const int32_t*>(tasks.data_ptr()),
        static_cast<const int32_t*>(task_count.data_ptr()),
        bias_pointer, static_cast<__hip_bfloat16*>(output.data_ptr()),
        routed_m, static_cast<int>(logical_n), static_cast<int>(logical_k),
        static_cast<int>(data.size(0)), static_cast<int>(data.size(1)),
        static_cast<int>(auxiliary.size(1)),
        static_cast<const int32_t*>(gather_indices.data_ptr()));
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
    const bool tiled_activation_scales = activation_scales.dim() == 4;
    const bool valid_activation_scale_shape =
        (activation_scales.dim() == 2 &&
         activation_scales.size(0) == activations.size(0) &&
         activation_scales.size(1) == logical_k / kScaleBlock) ||
        (tiled_activation_scales &&
         activation_scales.size(0) ==
             (logical_k / kScaleBlock + 3) / 4 &&
         activation_scales.size(1) >= (activations.size(0) + 15) / 16 &&
         activation_scales.size(2) == 4 && activation_scales.size(3) == 16);
    AITER_CHECK(activations.dim() == 2 && activations.size(1) == logical_k &&
                    valid_activation_scale_shape,
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
    const bool glm53_gate_shape = logical_k == 6144 && logical_n == 4096;
    const bool glm53_down_shape = logical_k == 2048 && logical_n == 6144;
    const char* tp4_env=std::getenv("IQ2R_GLM53_TP4");
    const bool tp4_enabled=tp4_env && std::strcmp(tp4_env,"1")==0;
    const bool glm53_tp_gate_shape = logical_k == 6144 && (logical_n == 512 || (tp4_enabled && logical_n == 1024));
    const bool glm53_tp_down_shape = (logical_k == 256 || (tp4_enabled && logical_k == 512)) && logical_n == 6144;
    const bool cooperative_shape =
        gpt_oss_shape || glm53_gate_shape || glm53_down_shape ||
        glm53_tp_gate_shape || glm53_tp_down_shape;
    if(cooperative_shape)
    {
        const int routed_m = static_cast<int>(activations.size(0));
        const int expert_count = static_cast<int>(data.size(0));
        const int active_experts = min(expert_count, routed_m);
        const int m32_task_capacity =
            (routed_m + 31) / 32 + min(routed_m, expert_count + 1);
        const int m64_task_capacity =
            (routed_m + 63) / 64 + min(routed_m, expert_count + 1);
        const bool use_glm53_tp_m32 =
            !tiled_activation_scales && routed_m >= 512 &&
            (routed_m <= 2048 || (routed_m == 2304 && expert_count == 257)) &&
            tasks.size(0) == m32_task_capacity;
        const bool use_glm53_tp_prefill_m64 =
            !tiled_activation_scales && expert_count == 257 &&
            (routed_m == 13824 ||
             (routed_m >= 18432 && routed_m <= 36864)) &&
            tasks.size(0) == m64_task_capacity;
        const int cu_count = static_cast<int>(get_num_cu_func());
        const int base_grid = 2 * cu_count;
        const bool use_wide_eight_wave_down =
            gpt_oss_shape && logical_n == 2880 && routed_m > 4 && routed_m <= 8;
        // GPT-OSS top-k=4 maps token counts 4..8 and 16 to 16..32 and 64
        // routed rows. Captured M=5/M=8 routes plus synthetic M=6/M=7 route
        // distributions consistently favor the narrower 3x4 down-projection
        // family in this decode band while retaining the existing 5xCU grid.
        const bool use_narrow_four_wave_down =
            gpt_oss_shape && logical_n == 2880 &&
            (routed_m == 16 || (routed_m >= 20 && routed_m <= 32) ||
             routed_m == 64);
        const bool use_two_wave_glm53_tp_down =
            glm53_tp_down_shape && logical_k == 256 && !tiled_activation_scales && routed_m >= 512;
        const bool use_narrow = routed_m < 16 && !use_wide_eight_wave_down;
        const int output_columns = use_narrow ? 48 : 96;
        const int estimated_tiles =
            active_experts *
            ((static_cast<int>(logical_n) + output_columns - 1) / output_columns);
        int launch_grid = base_grid * (estimated_tiles > base_grid ? 2 : 1);

        // A TP8 plain-GLM rank owns only 1/8 of each expert's intermediate
        // width. The resulting N=512 gate/up shape has enough independent
        // output work at the base 2xCU grid. The K=256 down projection favors
        // four waves through 256 routed rows. At 512 routed rows and above,
        // two waves map exactly to its two K tiles and an 8xCU grid wins across
        // independent route captures while reducing LDS from 32 KiB to 18 KiB.
        if(glm53_tp_gate_shape || glm53_tp_down_shape)
            launch_grid = base_grid;
        if(glm53_tp_down_shape && routed_m >= 16)
            launch_grid = 4 * cu_count;
        if(use_two_wave_glm53_tp_down)
            launch_grid = 8 * cu_count;
        if(glm53_tp_gate_shape && use_glm53_tp_prefill_m64)
            launch_grid = 10 * cu_count;

        if(gpt_oss_shape && !use_narrow && logical_n == 2880)
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
        const bool gate_up_projection =
            (gpt_oss_shape && logical_n == 5760) || glm53_gate_shape ||
            glm53_tp_gate_shape;
        const char* family_override = std::getenv(
            gate_up_projection ? "IQ2R_GEMM_GATE_UP_FAMILY"
                               : "IQ2R_GEMM_DOWN_FAMILY");
        const char* grid_override = std::getenv(
            gate_up_projection ? "IQ2R_GEMM_GATE_UP_GRID_MULTIPLIER"
                               : "IQ2R_GEMM_DOWN_GRID_MULTIPLIER");
        if(grid_override != nullptr && grid_override[0] != '\0')
        {
            const int multiplier = std::atoi(grid_override);
            AITER_CHECK(multiplier >= 1 && multiplier <= 16,
                        "IQ2R grid multiplier override must be in [1,16]");
            launch_grid = multiplier * cu_count;
        }
        const bool use_prefetch32a_auto =
            (family_override == nullptr || family_override[0] == '\0') &&
            logical_k == 2880 && routed_m == 4096 &&
            (logical_n == 5760 || logical_n == 2880);
        const bool glm53_tp_m256_down_large32_shape =
            glm53_tp_down_shape && !tiled_activation_scales &&
            routed_m == 2304 && expert_count == 257 &&
            tasks.size(0) == m32_task_capacity;
        const bool use_glm53_tp_m256_down_large32_auto =
            (family_override == nullptr || family_override[0] == '\0') &&
            glm53_tp_m256_down_large32_shape;
        const bool use_prefetch_family =
            use_prefetch32a_auto ||
            (family_override != nullptr &&
             std::strncmp(family_override, "prefetch", 8) == 0);
        if(use_prefetch32a_auto &&
           (grid_override == nullptr || grid_override[0] == '\0'))
            launch_grid = (logical_n == 5760 ? 11 : 10) * cu_count;
        if(use_glm53_tp_m256_down_large32_auto &&
           (grid_override == nullptr || grid_override[0] == '\0'))
            launch_grid = 4 * cu_count;

#define IQ2R_LAUNCH_DATA_PARALLEL(TILE_N)                                    \
    hipLaunchKernelGGL(                                                       \
        (iq2r_task_gemm_kernel<TILE_N>),                                     \
        dim3(static_cast<uint32_t>(logical_n / TILE_N),                       \
             static_cast<uint32_t>(tasks.size(0))),                           \
        dim3(256),                                                            \
        0,                                                                    \
        getCurrentHIPStream(),                                                \
        static_cast<const opus::fp8_t*>(activations.data_ptr()),              \
        static_cast<const uint8_t*>(activation_scales.data_ptr()),            \
        static_cast<const uint8_t*>(data.data_ptr()),                         \
        static_cast<const uint8_t*>(auxiliary.data_ptr()),                    \
        static_cast<const int32_t*>(tasks.data_ptr()),                        \
        static_cast<const int32_t*>(task_count.data_ptr()),                   \
        bias_pointer,                                                         \
        static_cast<__hip_bfloat16*>(output.data_ptr()),                      \
        routed_m,                                                             \
        static_cast<int>(logical_n),                                          \
        static_cast<int>(logical_k),                                          \
        expert_count,                                                         \
        static_cast<int>(data.size(1)),                                       \
        static_cast<int>(auxiliary.size(1)))

        // The generic task kernel is deliberately exposed as a profiling-only
        // GPT-OSS family.  It assigns one CTA to each (task, N tile) and then
        // walks every 16-row slab in that task, which is a more natural shape
        // for large routed-M/prefill than the persistent decode scheduler.
        // Keep production dispatch unchanged until a measured crossover is
        // established for both projections and representative route skew.
        if(family_override != nullptr &&
           std::strcmp(family_override, "data64") == 0)
        {
            IQ2R_LAUNCH_DATA_PARALLEL(64);
            HIP_CALL_LAUNCH(hipGetLastError());
            return;
        }
        if(family_override != nullptr &&
           std::strcmp(family_override, "data128") == 0)
        {
            AITER_CHECK(logical_n % 128 == 0,
                        "IQ2R data128 family requires N divisible by 128");
            IQ2R_LAUNCH_DATA_PARALLEL(128);
            HIP_CALL_LAUNCH(hipGetLastError());
            return;
        }
        if(family_override != nullptr &&
           ((!gpt_oss_shape &&
             (std::strcmp(family_override, "3x4s") == 0 ||
              std::strcmp(family_override, "6x4s") == 0 ||
              std::strcmp(family_override, "6x4as") == 0 ||
              std::strncmp(family_override, "prefetch", 8) == 0)) ||
            (!(gpt_oss_shape || glm53_tp_gate_shape ||
               glm53_tp_m256_down_large32_shape) &&
             std::strncmp(family_override, "large", 5) == 0)))
            AITER_CHECK(false,
                        "this IQ2R launch family is currently restricted to "
                        "a supported GPT-OSS or GLM-5.3 shape");
        if(use_prefetch32a_auto ||
           use_glm53_tp_m256_down_large32_auto ||
           (family_override != nullptr &&
            (std::strcmp(family_override, "large32") == 0 ||
            std::strcmp(family_override, "large32n") == 0 ||
            std::strcmp(family_override, "large64") == 0 ||
            std::strcmp(family_override, "large64n") == 0 ||
            std::strcmp(family_override, "prefetch32") == 0 ||
            std::strcmp(family_override, "prefetch32a") == 0 ||
            std::strcmp(family_override, "prefetch32t") == 0 ||
            std::strcmp(family_override, "prefetch64") == 0 ||
            std::strcmp(family_override, "prefetch64a") == 0 ||
            std::strcmp(family_override, "large192") == 0)))
        {
#define IQ2R_LAUNCH_LARGE_M(M_ATOMS, N_MAJOR)                                 \
    hipLaunchKernelGGL(                                                       \
        (iq2r_task_gemm_large_m_x192_kernel<M_ATOMS, N_MAJOR>),              \
        dim3(static_cast<uint32_t>(launch_grid)),                             \
        dim3(64, 4),                                                          \
        0,                                                                    \
        getCurrentHIPStream(),                                                \
        static_cast<const opus::fp8_t*>(activations.data_ptr()),              \
        static_cast<const uint8_t*>(activation_scales.data_ptr()),            \
        static_cast<const uint8_t*>(data.data_ptr()),                         \
        static_cast<const uint8_t*>(auxiliary.data_ptr()),                    \
        static_cast<const int32_t*>(tasks.data_ptr()),                        \
        static_cast<const int32_t*>(task_count.data_ptr()),                   \
        bias_pointer,                                                         \
        static_cast<__hip_bfloat16*>(output.data_ptr()),                      \
        routed_m,                                                             \
        static_cast<int>(logical_n),                                          \
        static_cast<int>(logical_k),                                          \
        expert_count,                                                         \
        static_cast<int>(data.size(1)),                                       \
        static_cast<int>(auxiliary.size(1)))
            if(use_prefetch_family)
            {
                AITER_CHECK(logical_k == 2880,
                            "IQ2R prefetch family requires K=2880");
                const bool task_persistent =
                    !use_prefetch32a_auto &&
                    std::strcmp(family_override, "prefetch32t") == 0;
                const int task_splits =
                    grid_override != nullptr && grid_override[0] != '\0'
                        ? std::atoi(grid_override)
                        : 4;
                const int prefetch_grid = task_persistent
                                              ? static_cast<int>(tasks.size(0)) *
                                                    task_splits
                                              : launch_grid;
#define IQ2R_LAUNCH_PREFETCH(M_ATOMS, TASK_PERSISTENT, PIN_TO_AGPR)           \
    hipLaunchKernelGGL(                                                       \
        (iq2r_task_gemm_prefetch_x192_kernel<M_ATOMS,                         \
                                              TASK_PERSISTENT,                \
                                              PIN_TO_AGPR>),                  \
        dim3(static_cast<uint32_t>(prefetch_grid)),                           \
        dim3(64, 4),                                                          \
        0,                                                                    \
        getCurrentHIPStream(),                                                \
        static_cast<const opus::fp8_t*>(activations.data_ptr()),              \
        static_cast<const uint8_t*>(activation_scales.data_ptr()),            \
        static_cast<const uint8_t*>(data.data_ptr()),                         \
        static_cast<const uint8_t*>(auxiliary.data_ptr()),                    \
        static_cast<const int32_t*>(tasks.data_ptr()),                        \
        static_cast<const int32_t*>(task_count.data_ptr()),                   \
        bias_pointer,                                                         \
        static_cast<__hip_bfloat16*>(output.data_ptr()),                      \
        routed_m,                                                             \
        static_cast<int>(logical_n),                                          \
        static_cast<int>(logical_k),                                          \
        expert_count,                                                         \
        static_cast<int>(data.size(1)),                                       \
        static_cast<int>(auxiliary.size(1)),                                  \
        task_splits)
                if(!use_prefetch32a_auto &&
                   std::strcmp(family_override, "prefetch32") == 0)
                    IQ2R_LAUNCH_PREFETCH(2, false, false);
                else if(use_prefetch32a_auto ||
                        std::strcmp(family_override, "prefetch32a") == 0)
                    IQ2R_LAUNCH_PREFETCH(2, false, true);
                else if(task_persistent)
                    IQ2R_LAUNCH_PREFETCH(2, true, false);
                else if(std::strcmp(family_override, "prefetch64") == 0)
                    IQ2R_LAUNCH_PREFETCH(4, false, false);
                else
                    IQ2R_LAUNCH_PREFETCH(4, false, true);
#undef IQ2R_LAUNCH_PREFETCH
            }
            else if(use_glm53_tp_m256_down_large32_auto ||
                    std::strcmp(family_override, "large32") == 0)
                IQ2R_LAUNCH_LARGE_M(2, false);
            else if(std::strcmp(family_override, "large32n") == 0)
                IQ2R_LAUNCH_LARGE_M(2, true);
            else if(std::strcmp(family_override, "large64n") == 0)
                IQ2R_LAUNCH_LARGE_M(4, true);
            else
                IQ2R_LAUNCH_LARGE_M(4, false);
#undef IQ2R_LAUNCH_LARGE_M
            HIP_CALL_LAUNCH(hipGetLastError());
            return;
        }
#undef IQ2R_LAUNCH_DATA_PARALLEL

#define IQ2R_LAUNCH_COOPERATIVE(OUTPUT_ATOMS, PHYSICAL_WAVES, LOOKAHEAD,      \
                                STAGED_SCALES, TILED_SCALES,                  \
                                VECTOR_CODEBOOK_COPY, LOAD_AUX)               \
    hipLaunchKernelGGL(                                                        \
        (iq2r_task_gemm_cooperative_kernel<OUTPUT_ATOMS,                      \
                                            PHYSICAL_WAVES,                   \
                                            LOOKAHEAD,                        \
                                            STAGED_SCALES,                    \
                                            TILED_SCALES,                     \
                                            VECTOR_CODEBOOK_COPY,             \
                                            LOAD_AUX,                          \
                                            false>),                           \
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
        nullptr,                                                               \
        nullptr,                                                               \
        routed_m,                                                              \
        static_cast<int>(logical_n),                                           \
        static_cast<int>(logical_k),                                           \
        expert_count,                                                          \
        static_cast<int>(data.size(1)),                                        \
        static_cast<int>(auxiliary.size(1)),                                   \
        0.0f,                                                                  \
        1.0f,                                                                  \
        0.0f)
        if(family_override != nullptr && family_override[0] != '\0')
        {
            if(std::strcmp(family_override, "3x8m2") == 0)
            {
                hipLaunchKernelGGL(
                    (iq2r_task_gemm_cooperative_m32_kernel<8>),
                    dim3(static_cast<uint32_t>(launch_grid)),
                    dim3(64, 8),
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
                    routed_m,
                    static_cast<int>(logical_n),
                    static_cast<int>(logical_k),
                    expert_count,
                    static_cast<int>(data.size(1)),
                    static_cast<int>(auxiliary.size(1)));
            }
            else if(std::strcmp(family_override, "3x4") == 0)
                IQ2R_LAUNCH_COOPERATIVE(3, 4, false, false, false, true, 2);
            else if(std::strcmp(family_override, "3x4scalar") == 0)
                IQ2R_LAUNCH_COOPERATIVE(3, 4, false, false, false, false, 2);
            else if(std::strcmp(family_override, "3x4t") == 0)
                IQ2R_LAUNCH_COOPERATIVE(3, 4, false, false, true, true, 2);
            else if(std::strcmp(family_override, "3x4s") == 0)
                IQ2R_LAUNCH_COOPERATIVE(3, 4, false, true, false, true, 2);
            else if(std::strcmp(family_override, "3x8") == 0)
                IQ2R_LAUNCH_COOPERATIVE(3, 8, false, false, false, true, 2);
            else if(std::strcmp(family_override, "3x8l0") == 0)
                IQ2R_LAUNCH_COOPERATIVE(3, 8, false, false, false, true, 0);
            else if(std::strcmp(family_override, "3x8t") == 0)
                IQ2R_LAUNCH_COOPERATIVE(3, 8, false, false, true, true, 2);
            else if(std::strcmp(family_override, "4x4") == 0)
                IQ2R_LAUNCH_COOPERATIVE(4, 4, false, false, false, true, 2);
            else if(std::strcmp(family_override, "4x8") == 0)
                IQ2R_LAUNCH_COOPERATIVE(4, 8, false, false, false, true, 2);
            else if(std::strcmp(family_override, "3x2m2") == 0)
            {
                AITER_CHECK(glm53_tp_down_shape,
                            "3x2m2 requires the GLM-5.3 TP8 down shape");
                hipLaunchKernelGGL(
                    (iq2r_task_gemm_cooperative_m32_kernel<2>),
                    dim3(static_cast<uint32_t>(launch_grid)),
                    dim3(64, 2),
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
                    routed_m,
                    static_cast<int>(logical_n),
                    static_cast<int>(logical_k),
                    expert_count,
                    static_cast<int>(data.size(1)),
                    static_cast<int>(auxiliary.size(1)));
            }
            else if(std::strcmp(family_override, "6x4") == 0)
                IQ2R_LAUNCH_COOPERATIVE(6, 4, false, false, false, true, 2);
            else if(std::strcmp(family_override, "6x2") == 0)
                IQ2R_LAUNCH_COOPERATIVE(6, 2, false, false, false, true, 2);
            else if(std::strcmp(family_override, "6x4scalar") == 0)
                IQ2R_LAUNCH_COOPERATIVE(6, 4, false, false, false, false, 2);
            else if(std::strcmp(family_override, "6x4t") == 0)
                IQ2R_LAUNCH_COOPERATIVE(6, 4, false, false, true, true, 2);
            else if(std::strcmp(family_override, "6x4a") == 0)
                IQ2R_LAUNCH_COOPERATIVE(6, 4, true, false, false, true, 2);
            else if(std::strcmp(family_override, "6x4at") == 0)
                IQ2R_LAUNCH_COOPERATIVE(6, 4, true, false, true, true, 2);
            else if(std::strcmp(family_override, "6x4s") == 0)
                IQ2R_LAUNCH_COOPERATIVE(6, 4, false, true, false, true, 2);
            else if(std::strcmp(family_override, "6x4as") == 0)
                IQ2R_LAUNCH_COOPERATIVE(6, 4, true, true, false, true, 2);
            else if(std::strcmp(family_override, "6x8") == 0)
                IQ2R_LAUNCH_COOPERATIVE(6, 8, false, false, false, true, 2);
            else if(std::strcmp(family_override, "6x8t") == 0)
                IQ2R_LAUNCH_COOPERATIVE(6, 8, false, false, true, true, 2);
            else if(std::strcmp(family_override, "6x8a") == 0)
                IQ2R_LAUNCH_COOPERATIVE(6, 8, true, false, false, true, 2);
            else
                AITER_CHECK(false,
                            "IQ2R launch family override must be one of "
                            "3x4, 3x4scalar, 3x4t, 3x4s, 3x8, 3x8l0, 3x8m2, "
                            "3x8t, "
                            "4x4, 4x8, 3x2m2, 6x4, 6x2, 6x4scalar, 6x4t, 6x4a, "
                            "6x4at, 6x4s, 6x4as, 6x8, 6x8t, 6x8a, data64, "
                            "data128, or "
                            "large32, large32n, large64, large64n, "
                            "prefetch32, prefetch32a, prefetch32t, "
                            "prefetch64, prefetch64a, or "
                            "large192");
        }
        else if(glm53_tp_gate_shape)
        {
            if(use_glm53_tp_m32 || use_glm53_tp_prefill_m64)
                hipLaunchKernelGGL(
                    (iq2r_task_gemm_cooperative_m32_kernel<8>),
                    dim3(static_cast<uint32_t>(launch_grid)),
                    dim3(64, 8),
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
                    routed_m,
                    static_cast<int>(logical_n),
                    static_cast<int>(logical_k),
                    expert_count,
                    static_cast<int>(data.size(1)),
                    static_cast<int>(auxiliary.size(1)));
            else if(tiled_activation_scales)
                IQ2R_LAUNCH_COOPERATIVE(3, 8, false, false, true, true, 0);
            else
                IQ2R_LAUNCH_COOPERATIVE(3, 8, false, false, false, true, 0);
        }
        else if(glm53_gate_shape)
        {
            // Real EP4 route sweeps from M=1 through M=32 favor the narrow
            // 48-column family. Four physical waves divide the 48 K tiles
            // evenly and leave enough workgroups to fill all CUs when only a
            // handful of this rank's experts are active.
            if(tiled_activation_scales)
                IQ2R_LAUNCH_COOPERATIVE(3, 4, false, false, true, true, 2);
            else
                IQ2R_LAUNCH_COOPERATIVE(3, 4, false, false, false, true, 2);
        }
        else if(glm53_tp_down_shape)
        {
            if((use_glm53_tp_m32 || use_glm53_tp_prefill_m64) && logical_k == 512)
                hipLaunchKernelGGL(
                    (iq2r_task_gemm_cooperative_m32_kernel<4>),
                    dim3(static_cast<uint32_t>(launch_grid)),
                    dim3(64, 4),
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
                    routed_m,
                    static_cast<int>(logical_n),
                    static_cast<int>(logical_k),
                    expert_count,
                    static_cast<int>(data.size(1)),
                    static_cast<int>(auxiliary.size(1)));
            else if(use_glm53_tp_m32 || use_glm53_tp_prefill_m64)
                hipLaunchKernelGGL(
                    (iq2r_task_gemm_cooperative_m32_kernel<2>),
                    dim3(static_cast<uint32_t>(launch_grid)),
                    dim3(64, 2),
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
                    routed_m,
                    static_cast<int>(logical_n),
                    static_cast<int>(logical_k),
                    expert_count,
                    static_cast<int>(data.size(1)),
                    static_cast<int>(auxiliary.size(1)));
            else if(use_two_wave_glm53_tp_down)
                IQ2R_LAUNCH_COOPERATIVE(6, 2, false, false, false, true, 2);
            else if(tiled_activation_scales)
                IQ2R_LAUNCH_COOPERATIVE(6, 4, false, false, true, true, 2);
            else
                IQ2R_LAUNCH_COOPERATIVE(6, 4, false, false, false, true, 2);
        }
        else if(glm53_down_shape)
        {
            // The down projection has only 16 K tiles. Four waves divide them
            // evenly, while the narrow output tile provides enough parallel
            // work across the 6144-column result for sparse EP4 decode routes.
            if(tiled_activation_scales)
                IQ2R_LAUNCH_COOPERATIVE(3, 4, false, false, true, true, 2);
            else
                IQ2R_LAUNCH_COOPERATIVE(3, 4, false, false, false, true, 2);
        }
        else if(use_narrow)
        {
            if(routed_m <= 4)
            {
                if(tiled_activation_scales)
                    IQ2R_LAUNCH_COOPERATIVE(
                        3, 8, false, false, true, true, 2);
                else
                    IQ2R_LAUNCH_COOPERATIVE(
                        3, 8, false, false, false, true, 2);
            }
            else
            {
                if(tiled_activation_scales)
                    IQ2R_LAUNCH_COOPERATIVE(
                        3, 4, false, false, true, true, 2);
                else
                    IQ2R_LAUNCH_COOPERATIVE(
                        3, 4, false, false, false, true, 2);
            }
        }
        else if(use_wide_eight_wave_down)
        {
            // The GPT-OSS down projection benefits from the wider output tile
            // while still splitting its 23 K tiles across eight physical waves.
            if(tiled_activation_scales)
                IQ2R_LAUNCH_COOPERATIVE(6, 8, false, false, true, true, 2);
            else
                IQ2R_LAUNCH_COOPERATIVE(6, 8, false, false, false, true, 2);
        }
        else if(use_narrow_four_wave_down)
        {
            if(tiled_activation_scales)
                IQ2R_LAUNCH_COOPERATIVE(3, 4, false, false, true, true, 2);
            else
                IQ2R_LAUNCH_COOPERATIVE(3, 4, false, false, false, true, 2);
        }
        else
        {
            if(tiled_activation_scales)
                IQ2R_LAUNCH_COOPERATIVE(6, 4, false, false, true, true, 2);
            else
                IQ2R_LAUNCH_COOPERATIVE(6, 4, false, false, false, true, 2);
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

void iq2r_task_gemm_swiglu_quant_out(const aiter_tensor_t& activations,
                                     const aiter_tensor_t& activation_scales,
                                     const aiter_tensor_t& data,
                                     const aiter_tensor_t& auxiliary,
                                     const aiter_tensor_t& tasks,
                                     const aiter_tensor_t& task_count,
                                     std::optional<aiter_tensor_t> bias,
                                     aiter_tensor_t& output,
                                     aiter_tensor_t& output_scales,
                                     int64_t logical_n,
                                     int64_t logical_k,
                                     double limit,
                                     double alpha,
                                     double up_offset)
{
    validate_weights(data, auxiliary, logical_n, logical_k);
    AITER_CHECK(activations.is_gpu() && activation_scales.is_gpu() && tasks.is_gpu() &&
                    task_count.is_gpu() && output.is_gpu() && output_scales.is_gpu(),
                "fused IQ2R gate/up requires GPU tensors");
    const int device = data.device_id;
    AITER_CHECK(activations.device_id == device && activation_scales.device_id == device &&
                    tasks.device_id == device && task_count.device_id == device &&
                    output.device_id == device && output_scales.device_id == device,
                "fused IQ2R gate/up tensors must share a GPU");
    AITER_CHECK(activations.dtype() == AITER_DTYPE_fp8 &&
                    activation_scales.dtype() == AITER_DTYPE_u8 &&
                    tasks.dtype() == AITER_DTYPE_i32 &&
                    task_count.dtype() == AITER_DTYPE_i32 &&
                    output.dtype() == AITER_DTYPE_fp8 &&
                    output_scales.dtype() == AITER_DTYPE_u8,
                "fused IQ2R gate/up tensor dtypes are invalid");
    AITER_CHECK(tasks.dim() == 2 && tasks.size(1) == 3 && task_count.dim() == 1 &&
                    task_count.size(0) == 1,
                "IQ2R tasks must be [capacity,3] and task_count [1]");
    const bool tiled_activation_scales = activation_scales.dim() == 4;
    const bool valid_activation_scale_shape =
        (activation_scales.dim() == 2 &&
         activation_scales.size(0) == activations.size(0) &&
         activation_scales.size(1) == logical_k / kScaleBlock) ||
        (tiled_activation_scales &&
         activation_scales.size(0) ==
             (logical_k / kScaleBlock + 3) / 4 &&
         activation_scales.size(1) >= (activations.size(0) + 15) / 16 &&
         activation_scales.size(2) == 4 && activation_scales.size(3) == 16);
    AITER_CHECK(activations.dim() == 2 && activations.size(1) == logical_k &&
                    valid_activation_scale_shape,
                "fused IQ2R gate/up activation shapes are invalid");
    AITER_CHECK(logical_n % 64 == 0 && output.dim() == 2 &&
                    output.size(0) == activations.size(0) &&
                    output.size(1) == logical_n / 2 && output_scales.dim() == 2 &&
                    output_scales.size(0) == activations.size(0) &&
                    output_scales.size(1) == logical_n / 64,
                "fused IQ2R gate/up output shapes are invalid");
    AITER_CHECK(activations.is_contiguous() && activation_scales.is_contiguous() &&
                    tasks.is_contiguous() && task_count.is_contiguous() &&
                    output.is_contiguous() && output_scales.is_contiguous(),
                "fused IQ2R gate/up tensors must be contiguous");

    const __hip_bfloat16* bias_pointer = nullptr;
    if(bias.has_value())
    {
        const auto& value = *bias;
        AITER_CHECK(value.is_gpu() && value.device_id == device &&
                        value.dtype() == AITER_DTYPE_bf16 && value.dim() == 2 &&
                        value.size(0) == data.size(0) && value.size(1) == logical_n &&
                        value.is_contiguous(),
                    "IQ2R bias must be contiguous BF16 [experts,N] on the same GPU");
        bias_pointer = static_cast<const __hip_bfloat16*>(value.data_ptr());
    }

    int grid_multiplier = 2;
    const char* grid_override = std::getenv("IQ2R_FUSED_GATE_GRID_MULTIPLIER");
    if(grid_override != nullptr && grid_override[0] != '\0')
    {
        grid_multiplier = std::atoi(grid_override);
        AITER_CHECK(grid_multiplier >= 1 && grid_multiplier <= 16,
                    "IQ2R fused gate grid multiplier must be in [1,16]");
    }
    const int launch_grid = grid_multiplier * static_cast<int>(get_num_cu_func());
#define IQ2R_LAUNCH_FUSED_GATE(TILED_SCALES)                                  \
    hipLaunchKernelGGL(                                                        \
        (iq2r_task_gemm_cooperative_kernel<4,                                 \
                                            4,                                 \
                                            false,                             \
                                            false,                             \
                                            TILED_SCALES,                      \
                                            true,                              \
                                            2,                                 \
                                            true>),                            \
        dim3(static_cast<uint32_t>(launch_grid)),                              \
        dim3(64, 4),                                                           \
        0,                                                                     \
        getCurrentHIPStream(),                                                 \
        static_cast<const opus::fp8_t*>(activations.data_ptr()),               \
        static_cast<const uint8_t*>(activation_scales.data_ptr()),             \
        static_cast<const uint8_t*>(data.data_ptr()),                          \
        static_cast<const uint8_t*>(auxiliary.data_ptr()),                     \
        static_cast<const int32_t*>(tasks.data_ptr()),                         \
        static_cast<const int32_t*>(task_count.data_ptr()),                    \
        bias_pointer,                                                          \
        nullptr,                                                               \
        static_cast<opus::fp8_t*>(output.data_ptr()),                          \
        static_cast<uint8_t*>(output_scales.data_ptr()),                       \
        static_cast<int>(activations.size(0)),                                 \
        static_cast<int>(logical_n),                                           \
        static_cast<int>(logical_k),                                           \
        static_cast<int>(data.size(0)),                                        \
        static_cast<int>(data.size(1)),                                        \
        static_cast<int>(auxiliary.size(1)),                                   \
        static_cast<float>(limit),                                             \
        static_cast<float>(alpha),                                             \
        static_cast<float>(up_offset))
    if(tiled_activation_scales)
        IQ2R_LAUNCH_FUSED_GATE(true);
    else
        IQ2R_LAUNCH_FUSED_GATE(false);
#undef IQ2R_LAUNCH_FUSED_GATE
    HIP_CALL_LAUNCH(hipGetLastError());
}

void iq2r_gate_aligned_fused_out(
    const aiter_tensor_t& activations, const aiter_tensor_t& scales,
    const aiter_tensor_t& data, const aiter_tensor_t& auxiliary,
    const aiter_tensor_t& tasks, const aiter_tensor_t& task_count,
    const aiter_tensor_t& gather, aiter_tensor_t& output,
    aiter_tensor_t& output_scales, int64_t rows_per_cta)
{
    validate_weights(data,auxiliary,512,6144);
    const aiter_tensor_t* tensors[]={&activations,&scales,&data,&auxiliary,&tasks,&task_count,&gather,&output,&output_scales};
    for(const auto* tensor:tensors)
        AITER_CHECK(tensor->is_gpu() && tensor->device_id==data.device_id && tensor->is_contiguous(),
                    "E066 requires contiguous tensors on the weight GPU");
    AITER_CHECK(activations.dim()==2 && activations.size(1)==6144 && activations.dtype()==AITER_DTYPE_fp8 && data.size(0)==257,
                "E066 requires token-major FP8 GLM TP8 inputs");
    const int tokens=static_cast<int>(activations.size(0)), routes=tokens*9;
    AITER_CHECK((tokens>=64 && tokens<=227) || tokens==256,"E066 requires qualified M32 decode");
    AITER_CHECK(scales.dim()==2 && scales.size(0)==tokens && scales.size(1)==192 && scales.dtype()==AITER_DTYPE_u8,
                "E066 requires row-major uint8 scales");
    AITER_CHECK(tasks.dtype()==AITER_DTYPE_i32 && tasks.dim()==2 && tasks.size(1)==3 &&
                tasks.size(0)==(routes+31)/32+min(routes,258),"E066 requires M32 tasks");
    AITER_CHECK(task_count.dtype()==AITER_DTYPE_i32 && task_count.dim()==1 && task_count.size(0)==1,
                "E066 requires scalar int32 task count");
    AITER_CHECK(gather.dtype()==AITER_DTYPE_i32 && gather.dim()==1 && gather.size(0)==routes,
                "E066 requires int32 gather [routes]");
    AITER_CHECK(output.dtype()==AITER_DTYPE_fp8 && output.dim()==2 && output.size(0)==routes && output.size(1)==256 &&
                output_scales.dtype()==AITER_DTYPE_u8 && output_scales.dim()==2 && output_scales.size(0)==routes && output_scales.size(1)==8,
                "E066 requires FP8 intermediate [routes,256], scales [routes,8]");
    AITER_CHECK(rows_per_cta==16 || rows_per_cta==32,"E066 rows_per_cta must be 16 or 32");
#define E066_LAUNCH(MATOMS) \
    hipLaunchKernelGGL((iq2r_gate_aligned_fused_kernel<MATOMS>), \
      dim3(2*static_cast<int>(get_num_cu_func())),dim3(64,8),0,getCurrentHIPStream(), \
      static_cast<const opus::fp8_t*>(activations.data_ptr()),static_cast<const uint8_t*>(scales.data_ptr()), \
      static_cast<const uint8_t*>(data.data_ptr()),static_cast<const uint8_t*>(auxiliary.data_ptr()), \
      static_cast<const int32_t*>(tasks.data_ptr()),static_cast<const int32_t*>(task_count.data_ptr()), \
      static_cast<const int32_t*>(gather.data_ptr()),static_cast<opus::fp8_t*>(output.data_ptr()), \
      static_cast<uint8_t*>(output_scales.data_ptr()),routes,static_cast<int>(data.size(1)),static_cast<int>(auxiliary.size(1)))
    if(rows_per_cta==16) E066_LAUNCH(1);
    else E066_LAUNCH(2);
#undef E066_LAUNCH
    HIP_CALL_LAUNCH(hipGetLastError());
}

void iq2r_gate_quad_fused_out(
    const aiter_tensor_t& activations, const aiter_tensor_t& scales,
    const aiter_tensor_t& data, const aiter_tensor_t& auxiliary,
    const aiter_tensor_t& tasks, const aiter_tensor_t& task_count,
    const aiter_tensor_t& gather, aiter_tensor_t& output,
    aiter_tensor_t& output_scales, int64_t rows_per_cta)
{
    AITER_CHECK(data.dtype()==AITER_DTYPE_u8 && data.dim()==2 &&
                data.size(0)==257 && data.size(1)==8*48*kQuadBytes,
                "E069 requires quad-packed uint8 gate [257,884736]");
    AITER_CHECK(auxiliary.dtype()==AITER_DTYPE_u8 && auxiliary.dim()==2 &&
                auxiliary.size(0)==257 && auxiliary.size(1)==expected_auxiliary_bytes(512),
                "E069 requires unchanged canonical gate auxiliary bytes");
    const aiter_tensor_t* tensors[]={&activations,&scales,&data,&auxiliary,&tasks,&task_count,&gather,&output,&output_scales};
    for(const auto* tensor:tensors)
        AITER_CHECK(tensor->is_gpu() && tensor->device_id==data.device_id && tensor->is_contiguous(),
                    "E069 requires contiguous tensors on the weight GPU");
    AITER_CHECK(activations.dim()==2 && activations.size(1)==6144 && activations.dtype()==AITER_DTYPE_fp8 && data.size(0)==257,
                "E069 requires token-major FP8 GLM TP8 inputs");
    const int tokens=static_cast<int>(activations.size(0)), routes=tokens*9;
    AITER_CHECK((tokens>=64 && tokens<=227) || tokens==256,"E069 requires qualified M32 decode");
    AITER_CHECK(scales.dim()==2 && scales.size(0)==tokens && scales.size(1)==192 && scales.dtype()==AITER_DTYPE_u8,
                "E069 requires row-major uint8 scales");
    AITER_CHECK(tasks.dtype()==AITER_DTYPE_i32 && tasks.dim()==2 && tasks.size(1)==3 &&
                tasks.size(0)==(routes+31)/32+min(routes,258),"E069 requires M32 tasks");
    AITER_CHECK(task_count.dtype()==AITER_DTYPE_i32 && task_count.dim()==1 && task_count.size(0)==1,
                "E069 requires scalar int32 task count");
    AITER_CHECK(gather.dtype()==AITER_DTYPE_i32 && gather.dim()==1 && gather.size(0)==routes,
                "E069 requires int32 gather [routes]");
    AITER_CHECK(output.dtype()==AITER_DTYPE_fp8 && output.dim()==2 && output.size(0)==routes && output.size(1)==256 &&
                output_scales.dtype()==AITER_DTYPE_u8 && output_scales.dim()==2 && output_scales.size(0)==routes && output_scales.size(1)==8,
                "E069 requires FP8 intermediate [routes,256], scales [routes,8]");
    AITER_CHECK(rows_per_cta==16 || rows_per_cta==32,"E069 rows_per_cta must be 16 or 32");
#define E069_LAUNCH(MATOMS) \
    hipLaunchKernelGGL((iq2r_gate_quad_fused_kernel<MATOMS>), \
      dim3(2*static_cast<int>(get_num_cu_func())),dim3(64,8),0,getCurrentHIPStream(), \
      static_cast<const opus::fp8_t*>(activations.data_ptr()),static_cast<const uint8_t*>(scales.data_ptr()), \
      static_cast<const uint8_t*>(data.data_ptr()),static_cast<const uint8_t*>(auxiliary.data_ptr()), \
      static_cast<const int32_t*>(tasks.data_ptr()),static_cast<const int32_t*>(task_count.data_ptr()), \
      static_cast<const int32_t*>(gather.data_ptr()),static_cast<opus::fp8_t*>(output.data_ptr()), \
      static_cast<uint8_t*>(output_scales.data_ptr()),routes,static_cast<int>(data.size(1)),static_cast<int>(auxiliary.size(1)))
    if(rows_per_cta==16) E069_LAUNCH(1);
    else E069_LAUNCH(2);
#undef E069_LAUNCH
    HIP_CALL_LAUNCH(hipGetLastError());
}

void iq2r_gate_quad_sparse_out(
    const aiter_tensor_t& activations, const aiter_tensor_t& scales,
    const aiter_tensor_t& data, const aiter_tensor_t& auxiliary,
    const aiter_tensor_t& tasks, const aiter_tensor_t& task_count,
    const aiter_tensor_t& gather, aiter_tensor_t& output,
    aiter_tensor_t& output_scales, int64_t rows_per_cta)
{
    AITER_CHECK(data.dtype()==AITER_DTYPE_u8 && data.dim()==2 &&
                data.size(0)==257 && data.size(1)==8*48*kQuadBytes,
                "E071 requires quad-packed uint8 gate [257,884736]");
    AITER_CHECK(auxiliary.dtype()==AITER_DTYPE_u8 && auxiliary.dim()==2 &&
                auxiliary.size(0)==257 && auxiliary.size(1)==expected_auxiliary_bytes(512),
                "E071 requires unchanged canonical gate auxiliary bytes");
    const aiter_tensor_t* tensors[]={&activations,&scales,&data,&auxiliary,&tasks,&task_count,&gather,&output,&output_scales};
    for(const auto* tensor:tensors)
        AITER_CHECK(tensor->is_gpu() && tensor->device_id==data.device_id && tensor->is_contiguous(),
                    "E071 requires contiguous tensors on the weight GPU");
    AITER_CHECK(activations.dim()==2 && activations.size(1)==6144 && activations.dtype()==AITER_DTYPE_fp8 && data.size(0)==257,
                "E071 requires token-major FP8 GLM TP8 inputs");
    const int tokens=static_cast<int>(activations.size(0)), routes=tokens*9;
    AITER_CHECK(tokens==32 || (tokens>=64 && tokens<=227) || tokens==256,"E071 requires qualified M32 decode");
    AITER_CHECK(scales.dim()==2 && scales.size(0)==tokens && scales.size(1)==192 && scales.dtype()==AITER_DTYPE_u8,
                "E071 requires row-major uint8 scales");
    AITER_CHECK(tasks.dtype()==AITER_DTYPE_i32 && tasks.dim()==2 && tasks.size(1)==3 &&
                tasks.size(0)==(routes+31)/32+min(routes,258),"E071 requires M32 tasks");
    AITER_CHECK(task_count.dtype()==AITER_DTYPE_i32 && task_count.dim()==1 && task_count.size(0)==1,
                "E071 requires scalar int32 task count");
    AITER_CHECK(gather.dtype()==AITER_DTYPE_i32 && gather.dim()==1 && gather.size(0)==routes,
                "E071 requires int32 gather [routes]");
    AITER_CHECK(output.dtype()==AITER_DTYPE_fp8 && output.dim()==2 && output.size(0)==routes && output.size(1)==256 &&
                output_scales.dtype()==AITER_DTYPE_u8 && output_scales.dim()==2 && output_scales.size(0)==routes && output_scales.size(1)==8,
                "E071 requires FP8 intermediate [routes,256], scales [routes,8]");
    AITER_CHECK(rows_per_cta==32,"E071 rows_per_cta must be 32");
    const char* direct_setting=std::getenv("IQ2R_GLM53_XCD_SPARSE_GATE");
    const bool direct=direct_setting && std::strcmp(direct_setting,"1")==0 && tokens==128;
    if(direct) {
        static thread_local bool audited=false;
        const char* audit=std::getenv("ATOM_IQ2R_AUDIT");
        if(audit && std::strcmp(audit,"1")==0 && !audited)
        {{
            audited=true;
            std::fprintf(stderr,"IQ2R_XCD_SPARSE_GATE device=%d tokens=%d tp=8 kernel=iq2r_gate_quad_sparse_kernel_xcd\n",data.device_id,tokens);
        }}
#define E071_XCD_LAUNCH(MATOMS) \
    hipLaunchKernelGGL((iq2r_gate_quad_sparse_kernel_xcd<MATOMS>), \
      dim3(2*static_cast<int>(get_num_cu_func())),dim3(64,8),0,getCurrentHIPStream(), \
      static_cast<const opus::fp8_t*>(activations.data_ptr()),static_cast<const uint8_t*>(scales.data_ptr()), \
      static_cast<const uint8_t*>(data.data_ptr()),static_cast<const uint8_t*>(auxiliary.data_ptr()), \
      static_cast<const int32_t*>(tasks.data_ptr()),static_cast<const int32_t*>(task_count.data_ptr()), \
      static_cast<const int32_t*>(gather.data_ptr()),static_cast<opus::fp8_t*>(output.data_ptr()), \
      static_cast<uint8_t*>(output_scales.data_ptr()),routes,static_cast<int>(data.size(1)),static_cast<int>(auxiliary.size(1)))
    if(rows_per_cta==16) E071_XCD_LAUNCH(1);
    else E071_XCD_LAUNCH(2);
#undef E071_XCD_LAUNCH
    } else {
#define E071_LAUNCH(MATOMS) \
    hipLaunchKernelGGL((iq2r_gate_quad_sparse_kernel<MATOMS>), \
      dim3(2*static_cast<int>(get_num_cu_func())),dim3(64,8),0,getCurrentHIPStream(), \
      static_cast<const opus::fp8_t*>(activations.data_ptr()),static_cast<const uint8_t*>(scales.data_ptr()), \
      static_cast<const uint8_t*>(data.data_ptr()),static_cast<const uint8_t*>(auxiliary.data_ptr()), \
      static_cast<const int32_t*>(tasks.data_ptr()),static_cast<const int32_t*>(task_count.data_ptr()), \
      static_cast<const int32_t*>(gather.data_ptr()),static_cast<opus::fp8_t*>(output.data_ptr()), \
      static_cast<uint8_t*>(output_scales.data_ptr()),routes,static_cast<int>(data.size(1)),static_cast<int>(auxiliary.size(1)))
    if(rows_per_cta==16) E071_LAUNCH(1);
    else E071_LAUNCH(2);
#undef E071_LAUNCH
    }
    HIP_CALL_LAUNCH(hipGetLastError());
}
void iq2r_gate_quad_splitk_out(
    const aiter_tensor_t& input, const aiter_tensor_t& scales,
    const aiter_tensor_t& data, const aiter_tensor_t& auxiliary,
    const aiter_tensor_t& tasks, const aiter_tensor_t& task_count,
    aiter_tensor_t& partials, aiter_tensor_t& output,
    aiter_tensor_t& output_scales, int64_t physical_waves)
{
    const aiter_tensor_t* tensors[]={&input,&scales,&data,&auxiliary,&tasks,&task_count,&partials,&output,&output_scales};
    for(const auto* t:tensors)
        AITER_CHECK(t->is_gpu() && t->device_id==data.device_id && t->is_contiguous(),"E073 tensors must be contiguous on the weight GPU");
    AITER_CHECK(data.dtype()==AITER_DTYPE_u8 && data.dim()==2 && data.size(0)==257 && data.size(1)==8*48*kQuadBytes,
                "E073 requires quad-packed gate bytes");
    AITER_CHECK(auxiliary.dtype()==AITER_DTYPE_u8 && auxiliary.dim()==2 && auxiliary.size(0)==257 && auxiliary.size(1)==expected_auxiliary_bytes(512),
                "E073 requires canonical auxiliary bytes");
    AITER_CHECK(input.dtype()==AITER_DTYPE_fp8 && input.dim()==2 && input.size(1)==6144,"E073 requires route-major FP8 inputs");
    const int routes=static_cast<int>(input.size(0));
    AITER_CHECK(routes>=9 && routes<=144 && routes%9==0,"E073 targets 1..16 tokens");
    AITER_CHECK(scales.dtype()==AITER_DTYPE_u8 && scales.dim()==2 && scales.size(0)==routes && scales.size(1)==192,"E073 requires row-major input scales");
    AITER_CHECK(tasks.dtype()==AITER_DTYPE_i32 && tasks.dim()==2 && tasks.size(0)==routes && tasks.size(1)==3 &&
                task_count.dtype()==AITER_DTYPE_i32 && task_count.dim()==1 && task_count.size(0)==1,"E073 requires direct/grouped GLM tasks");
    AITER_CHECK(partials.dtype()==AITER_DTYPE_fp32 && partials.dim()==3 && partials.size(0)==8 && partials.size(1)==routes && partials.size(2)==512,
                "E073 requires FP32 partials [8,routes,512]");
    AITER_CHECK(output.dtype()==AITER_DTYPE_fp8 && output.dim()==2 && output.size(0)==routes && output.size(1)==256 &&
                output_scales.dtype()==AITER_DTYPE_u8 && output_scales.dim()==2 && output_scales.size(0)==routes && output_scales.size(1)==8,
                "E073 requires FP8 intermediate and E8M0 scales");
    AITER_CHECK(physical_waves==1 || physical_waves==2 || physical_waves==4,"E073 supports 1,2,4 waves");
#define E073_LAUNCH(W) \
    hipLaunchKernelGGL((iq2r_gate_quad_splitk_kernel<W>),dim3(2*static_cast<int>(get_num_cu_func())),dim3(64,W),0,getCurrentHIPStream(), \
      static_cast<const opus::fp8_t*>(input.data_ptr()),static_cast<const uint8_t*>(scales.data_ptr()), \
      static_cast<const uint8_t*>(data.data_ptr()),static_cast<const uint8_t*>(auxiliary.data_ptr()), \
      static_cast<const int32_t*>(tasks.data_ptr()),static_cast<const int32_t*>(task_count.data_ptr()), \
      static_cast<float*>(partials.data_ptr()),routes,static_cast<int>(data.size(1)),static_cast<int>(auxiliary.size(1)))
    if(physical_waves==1) E073_LAUNCH(1);
    else if(physical_waves==2) E073_LAUNCH(2);
    else E073_LAUNCH(4);
#undef E073_LAUNCH
    HIP_CALL_LAUNCH(hipGetLastError());
    hipLaunchKernelGGL(iq2r_gate_splitk_swiglu_quant_kernel,dim3(routes*4),dim3(64),0,getCurrentHIPStream(),
        static_cast<const float*>(partials.data_ptr()),static_cast<opus::fp8_t*>(output.data_ptr()),
        static_cast<uint8_t*>(output_scales.data_ptr()),routes);
    HIP_CALL_LAUNCH(hipGetLastError());
}
void iq2r_gate_quad_route_fused_out(
    const aiter_tensor_t& activations, const aiter_tensor_t& scales,
    const aiter_tensor_t& data, const aiter_tensor_t& auxiliary,
    const aiter_tensor_t& tasks, const aiter_tensor_t& task_count,
    aiter_tensor_t& output, aiter_tensor_t& output_scales)
{
    const aiter_tensor_t* tensors[]={&activations,&scales,&data,&auxiliary,
                                    &tasks,&task_count,&output,&output_scales};
    for(const auto* tensor:tensors)
        AITER_CHECK(tensor->is_gpu() && tensor->device_id==data.device_id && tensor->is_contiguous(),
                    "E079 requires contiguous tensors on the weight GPU");
    AITER_CHECK(data.dtype()==AITER_DTYPE_u8 && data.dim()==2 && data.size(0)==257 &&
                data.size(1)==8*48*kQuadBytes, "E079 requires quad-packed GLM gate weights");
    AITER_CHECK(auxiliary.dtype()==AITER_DTYPE_u8 && auxiliary.dim()==2 &&
                auxiliary.size(0)==257 && auxiliary.size(1)==expected_auxiliary_bytes(512),
                "E079 requires canonical gate auxiliary bytes");
    AITER_CHECK(activations.dtype()==AITER_DTYPE_fp8 && activations.dim()==2 &&
                activations.size(1)==6144 && activations.size(0)>=9 &&
                activations.size(0)<=144 && activations.size(0)%9==0,
                "E079 requires route-major FP8 for 1..16 top-9 tokens");
    const int routes=static_cast<int>(activations.size(0));
    AITER_CHECK(scales.dtype()==AITER_DTYPE_u8 && scales.dim()==2 &&
                scales.size(0)==routes && scales.size(1)==192,
                "E079 requires route-major E8M0 input scales");
    AITER_CHECK(tasks.dtype()==AITER_DTYPE_i32 && tasks.dim()==2 &&
                tasks.size(0)==routes && tasks.size(1)==3 &&
                task_count.dtype()==AITER_DTYPE_i32 && task_count.dim()==1 && task_count.size(0)==1,
                "E079 requires small-token direct/grouped tasks");
    AITER_CHECK(output.dtype()==AITER_DTYPE_fp8 && output.dim()==2 &&
                output.size(0)==routes && output.size(1)==256 &&
                output_scales.dtype()==AITER_DTYPE_u8 && output_scales.dim()==2 &&
                output_scales.size(0)==routes && output_scales.size(1)==8,
                "E079 requires FP8 intermediate [routes,256] and E8M0 [routes,8]");
    HipDeviceGuard device_guard(data.device_id);
    hipLaunchKernelGGL((iq2r_gate_quad_route_fused_kernel<1>),
        dim3(2*static_cast<int>(get_num_cu_func())),dim3(64,8),0,getCurrentHIPStream(),
        static_cast<const opus::fp8_t*>(activations.data_ptr()),
        static_cast<const uint8_t*>(scales.data_ptr()),static_cast<const uint8_t*>(data.data_ptr()),
        static_cast<const uint8_t*>(auxiliary.data_ptr()),static_cast<const int32_t*>(tasks.data_ptr()),
        static_cast<const int32_t*>(task_count.data_ptr()),nullptr,
        static_cast<opus::fp8_t*>(output.data_ptr()),static_cast<uint8_t*>(output_scales.data_ptr()),
        routes,static_cast<int>(data.size(1)),static_cast<int>(auxiliary.size(1)));
    HIP_CALL_LAUNCH(hipGetLastError());
}

void iq2r_down_sparse_large32_out(
    const aiter_tensor_t& activations, const aiter_tensor_t& scales,
    const aiter_tensor_t& data, const aiter_tensor_t& auxiliary,
    const aiter_tensor_t& tasks, const aiter_tensor_t& task_count,
    aiter_tensor_t& output, int64_t grid_multiplier)
{
    validate_weights(data, auxiliary, 6144, 256);
    const aiter_tensor_t* tensors[]={&activations,&scales,&data,&auxiliary,&tasks,&task_count,&output};
    for(const auto* tensor:tensors)
        AITER_CHECK(tensor->is_gpu() && tensor->device_id==data.device_id && tensor->is_contiguous(),
                    "E085 tensors must be contiguous on the weight GPU");
    AITER_CHECK(data.size(0)==257 && activations.dtype()==AITER_DTYPE_fp8 &&
                activations.dim()==2 && activations.size(0)==2304 && activations.size(1)==256,
                "E085 requires GLM TP8 M256 intermediate FP8 [2304,256]");
    AITER_CHECK(scales.dtype()==AITER_DTYPE_u8 && scales.dim()==2 && scales.size(0)==2304 && scales.size(1)==8,
                "E085 requires row-major E8M0 scales [2304,8]");
    AITER_CHECK(tasks.dtype()==AITER_DTYPE_i32 && tasks.dim()==2 && tasks.size(0)==330 && tasks.size(1)==3 &&
                task_count.dtype()==AITER_DTYPE_i32 && task_count.dim()==1 && task_count.size(0)==1,
                "E085 requires GLM M32 tasks [330,3]");
    AITER_CHECK(output.dtype()==AITER_DTYPE_bf16 && output.dim()==2 && output.size(0)==2304 && output.size(1)==6144,
                "E085 requires BF16 route output [2304,6144]");
    AITER_CHECK(grid_multiplier==2 || grid_multiplier==4 || grid_multiplier==8,
                "E085 grid multiplier must be 2, 4 or 8");
    HipDeviceGuard device_guard(data.device_id);
    hipLaunchKernelGGL((iq2r_down_sparse_large32_kernel<2,false>),
        dim3(grid_multiplier*static_cast<int>(get_num_cu_func())),dim3(64,4),0,getCurrentHIPStream(),
        static_cast<const opus::fp8_t*>(activations.data_ptr()),static_cast<const uint8_t*>(scales.data_ptr()),
        static_cast<const uint8_t*>(data.data_ptr()),static_cast<const uint8_t*>(auxiliary.data_ptr()),
        static_cast<const int32_t*>(tasks.data_ptr()),static_cast<const int32_t*>(task_count.data_ptr()),
        nullptr,static_cast<__hip_bfloat16*>(output.data_ptr()),2304,6144,256,257,
        static_cast<int>(data.size(1)),static_cast<int>(auxiliary.size(1)));
    HIP_CALL_LAUNCH(hipGetLastError());
}
void iq2r_down_sparse_scheduled_out(
    const aiter_tensor_t& activations, const aiter_tensor_t& scales,
    const aiter_tensor_t& data, const aiter_tensor_t& auxiliary,
    const aiter_tensor_t& tasks, const aiter_tensor_t& task_count,
    aiter_tensor_t& output, int64_t grid_multiplier, int64_t variant)
{
    validate_weights(data, auxiliary, 6144, 256);
    const aiter_tensor_t* tensors[]={&activations,&scales,&data,&auxiliary,&tasks,&task_count,&output};
    for(const auto* tensor:tensors)
        AITER_CHECK(tensor->is_gpu() && tensor->device_id==data.device_id && tensor->is_contiguous(),
                    "E085 tensors must be contiguous on the weight GPU");
    AITER_CHECK(data.size(0)==257 && activations.dtype()==AITER_DTYPE_fp8 &&
                activations.dim()==2 && activations.size(0)==2304 && activations.size(1)==256,
                "E085 requires GLM TP8 M256 intermediate FP8 [2304,256]");
    AITER_CHECK(scales.dtype()==AITER_DTYPE_u8 && scales.dim()==2 && scales.size(0)==2304 && scales.size(1)==8,
                "E085 requires row-major E8M0 scales [2304,8]");
    AITER_CHECK(tasks.dtype()==AITER_DTYPE_i32 && tasks.dim()==2 && tasks.size(0)==330 && tasks.size(1)==3 &&
                task_count.dtype()==AITER_DTYPE_i32 && task_count.dim()==1 && task_count.size(0)==1,
                "E085 requires GLM M32 tasks [330,3]");
    AITER_CHECK(output.dtype()==AITER_DTYPE_bf16 && output.dim()==2 && output.size(0)==2304 && output.size(1)==6144,
                "E085 requires BF16 route output [2304,6144]");
    AITER_CHECK(grid_multiplier==2 || grid_multiplier==4 || grid_multiplier==8,
                "E085 grid multiplier must be 2, 4 or 8");
    AITER_CHECK(variant>=0 && variant<4,"Invalid scheduled down variant");
    HipDeviceGuard device_guard(data.device_id);
    if(variant==0)
    {
    hipLaunchKernelGGL(iq2r_scheduled_large_fixed_kn_kernel,
        dim3(grid_multiplier*static_cast<int>(get_num_cu_func())),dim3(64,4),0,getCurrentHIPStream(),
        static_cast<const opus::fp8_t*>(activations.data_ptr()),static_cast<const uint8_t*>(scales.data_ptr()),
        static_cast<const uint8_t*>(data.data_ptr()),static_cast<const uint8_t*>(auxiliary.data_ptr()),
        static_cast<const int32_t*>(tasks.data_ptr()),static_cast<const int32_t*>(task_count.data_ptr()),
        nullptr,static_cast<__hip_bfloat16*>(output.data_ptr()),2304,6144,256,257,
        static_cast<int>(data.size(1)),static_cast<int>(auxiliary.size(1)));
    }
    if(variant==1)
    {
    hipLaunchKernelGGL(iq2r_scheduled_large_register_prefetch_uniform_kernel,
        dim3(grid_multiplier*static_cast<int>(get_num_cu_func())),dim3(64,4),0,getCurrentHIPStream(),
        static_cast<const opus::fp8_t*>(activations.data_ptr()),static_cast<const uint8_t*>(scales.data_ptr()),
        static_cast<const uint8_t*>(data.data_ptr()),static_cast<const uint8_t*>(auxiliary.data_ptr()),
        static_cast<const int32_t*>(tasks.data_ptr()),static_cast<const int32_t*>(task_count.data_ptr()),
        nullptr,static_cast<__hip_bfloat16*>(output.data_ptr()),2304,6144,256,257,
        static_cast<int>(data.size(1)),static_cast<int>(auxiliary.size(1)));
    }
    if(variant==2)
    {
    hipLaunchKernelGGL(iq2r_scheduled_large_fixed_kn_deferred_scale_kernel,
        dim3(grid_multiplier*static_cast<int>(get_num_cu_func())),dim3(64,4),0,getCurrentHIPStream(),
        static_cast<const opus::fp8_t*>(activations.data_ptr()),static_cast<const uint8_t*>(scales.data_ptr()),
        static_cast<const uint8_t*>(data.data_ptr()),static_cast<const uint8_t*>(auxiliary.data_ptr()),
        static_cast<const int32_t*>(tasks.data_ptr()),static_cast<const int32_t*>(task_count.data_ptr()),
        nullptr,static_cast<__hip_bfloat16*>(output.data_ptr()),2304,6144,256,257,
        static_cast<int>(data.size(1)),static_cast<int>(auxiliary.size(1)));
    }
    if(variant==3)
    {
    hipLaunchKernelGGL(iq2r_scheduled_large_register_prefetch_uniform_deferred_scale_kernel,
        dim3(grid_multiplier*static_cast<int>(get_num_cu_func())),dim3(64,4),0,getCurrentHIPStream(),
        static_cast<const opus::fp8_t*>(activations.data_ptr()),static_cast<const uint8_t*>(scales.data_ptr()),
        static_cast<const uint8_t*>(data.data_ptr()),static_cast<const uint8_t*>(auxiliary.data_ptr()),
        static_cast<const int32_t*>(tasks.data_ptr()),static_cast<const int32_t*>(task_count.data_ptr()),
        nullptr,static_cast<__hip_bfloat16*>(output.data_ptr()),2304,6144,256,257,
        static_cast<int>(data.size(1)),static_cast<int>(auxiliary.size(1)));
    }
    HIP_CALL_LAUNCH(hipGetLastError());
}

void iq2r_down_shortk_out(
    const aiter_tensor_t& activations, const aiter_tensor_t& scales,
    const aiter_tensor_t& data, const aiter_tensor_t& auxiliary,
    const aiter_tensor_t& tasks, const aiter_tensor_t& task_count,
    aiter_tensor_t& output, int64_t grid_multiplier, int64_t variant)
{
    const int routes=static_cast<int>(activations.size(0));
    AITER_CHECK(((routes>=9 && routes<=144 && routes%9==0) || routes==288),"Short-K down requires 1..16 or 32 top-9 tokens");
    AITER_CHECK(variant>=0 && variant<3,"Invalid short-K down variant");
    validate_weights(data, auxiliary, 6144, 256);
    const aiter_tensor_t* tensors[]={&activations,&scales,&data,&auxiliary,&tasks,&task_count,&output};
    for(const auto* tensor:tensors)
        AITER_CHECK(tensor->is_gpu() && tensor->device_id==data.device_id && tensor->is_contiguous(),
                    "Short-K tensors must be contiguous on the weight GPU");
    AITER_CHECK(data.size(0)==257 && activations.dtype()==AITER_DTYPE_fp8 &&
                activations.dim()==2 && activations.size(0)==routes && activations.size(1)==256,
                "Short-K requires GLM TP8 intermediate FP8 [routes,256]");
    AITER_CHECK(scales.dtype()==AITER_DTYPE_u8 && scales.dim()==2 && scales.size(0)==routes && scales.size(1)==8,
                "Short-K requires row-major E8M0 scales [routes,8]");
    AITER_CHECK(tasks.dtype()==AITER_DTYPE_i32 && tasks.dim()==2 && tasks.size(0)==(routes==288 ? 267 : routes) && tasks.size(1)==3 &&
                task_count.dtype()==AITER_DTYPE_i32 && task_count.dim()==1 && task_count.size(0)==1,
                "Short-K requires small-token tasks [routes,3]");
    AITER_CHECK(output.dtype()==AITER_DTYPE_bf16 && output.dim()==2 && output.size(0)==routes && output.size(1)==6144,
                "Short-K requires BF16 route output [routes,6144]");
    AITER_CHECK(grid_multiplier==2 || grid_multiplier==4 || grid_multiplier==8,
                "Short-K grid multiplier must be 2, 4 or 8");
    HipDeviceGuard device_guard(data.device_id);
    if(variant==0)
    {
    hipLaunchKernelGGL(iq2r_scheduled_small_single_epilogue_kernel,
        dim3(grid_multiplier*static_cast<int>(get_num_cu_func())),dim3(64,4),0,getCurrentHIPStream(),
        static_cast<const opus::fp8_t*>(activations.data_ptr()),static_cast<const uint8_t*>(scales.data_ptr()),
        static_cast<const uint8_t*>(data.data_ptr()),static_cast<const uint8_t*>(auxiliary.data_ptr()),
        static_cast<const int32_t*>(tasks.data_ptr()),static_cast<const int32_t*>(task_count.data_ptr()),
        nullptr,static_cast<__hip_bfloat16*>(output.data_ptr()),nullptr,nullptr,routes,6144,256,257,
        static_cast<int>(data.size(1)),static_cast<int>(auxiliary.size(1)),0.0f,1.0f,0.0f);
    }
    if(variant==1)
    {
    hipLaunchKernelGGL(iq2r_scheduled_small_single_epilogue_deferred_kernel,
        dim3(grid_multiplier*static_cast<int>(get_num_cu_func())),dim3(64,4),0,getCurrentHIPStream(),
        static_cast<const opus::fp8_t*>(activations.data_ptr()),static_cast<const uint8_t*>(scales.data_ptr()),
        static_cast<const uint8_t*>(data.data_ptr()),static_cast<const uint8_t*>(auxiliary.data_ptr()),
        static_cast<const int32_t*>(tasks.data_ptr()),static_cast<const int32_t*>(task_count.data_ptr()),
        nullptr,static_cast<__hip_bfloat16*>(output.data_ptr()),nullptr,nullptr,routes,6144,256,257,
        static_cast<int>(data.size(1)),static_cast<int>(auxiliary.size(1)),0.0f,1.0f,0.0f);
    }
    if(variant==2)
    {
    hipLaunchKernelGGL(iq2r_scheduled_small_register_single_epilogue_kernel,
        dim3(grid_multiplier*static_cast<int>(get_num_cu_func())),dim3(64,4),0,getCurrentHIPStream(),
        static_cast<const opus::fp8_t*>(activations.data_ptr()),static_cast<const uint8_t*>(scales.data_ptr()),
        static_cast<const uint8_t*>(data.data_ptr()),static_cast<const uint8_t*>(auxiliary.data_ptr()),
        static_cast<const int32_t*>(tasks.data_ptr()),static_cast<const int32_t*>(task_count.data_ptr()),
        nullptr,static_cast<__hip_bfloat16*>(output.data_ptr()),nullptr,nullptr,routes,6144,256,257,
        static_cast<int>(data.size(1)),static_cast<int>(auxiliary.size(1)),0.0f,1.0f,0.0f);
    }
    HIP_CALL_LAUNCH(hipGetLastError());
}

void iq2r_gate_quad_scheduled_out(
    const aiter_tensor_t& activations, const aiter_tensor_t& scales,
    const aiter_tensor_t& data, const aiter_tensor_t& auxiliary,
    const aiter_tensor_t& tasks, const aiter_tensor_t& task_count,
    aiter_tensor_t& output, aiter_tensor_t& output_scales, int64_t variant)
{
    const aiter_tensor_t* tensors[]={&activations,&scales,&data,&auxiliary,
                                    &tasks,&task_count,&output,&output_scales};
    for(const auto* tensor:tensors)
        AITER_CHECK(tensor->is_gpu() && tensor->device_id==data.device_id && tensor->is_contiguous(),
                    "E079 requires contiguous tensors on the weight GPU");
    AITER_CHECK(data.dtype()==AITER_DTYPE_u8 && data.dim()==2 && data.size(0)==257 &&
                data.size(1)==8*48*kQuadBytes, "E079 requires quad-packed GLM gate weights");
    AITER_CHECK(auxiliary.dtype()==AITER_DTYPE_u8 && auxiliary.dim()==2 &&
                auxiliary.size(0)==257 && auxiliary.size(1)==expected_auxiliary_bytes(512),
                "E079 requires canonical gate auxiliary bytes");
    AITER_CHECK(activations.dtype()==AITER_DTYPE_fp8 && activations.dim()==2 &&
                activations.size(1)==6144 && activations.size(0)>=9 &&
                activations.size(0)<=144 && activations.size(0)%9==0,
                "E079 requires route-major FP8 for 1..16 top-9 tokens");
    const int routes=static_cast<int>(activations.size(0));
    AITER_CHECK(scales.dtype()==AITER_DTYPE_u8 && scales.dim()==2 &&
                scales.size(0)==routes && scales.size(1)==192,
                "E079 requires route-major E8M0 input scales");
    AITER_CHECK(tasks.dtype()==AITER_DTYPE_i32 && tasks.dim()==2 &&
                tasks.size(0)==routes && tasks.size(1)==3 &&
                task_count.dtype()==AITER_DTYPE_i32 && task_count.dim()==1 && task_count.size(0)==1,
                "E079 requires small-token direct/grouped tasks");
    AITER_CHECK(output.dtype()==AITER_DTYPE_fp8 && output.dim()==2 &&
                output.size(0)==routes && output.size(1)==256 &&
                output_scales.dtype()==AITER_DTYPE_u8 && output_scales.dim()==2 &&
                output_scales.size(0)==routes && output_scales.size(1)==8,
                "E079 requires FP8 intermediate [routes,256] and E8M0 [routes,8]");
    AITER_CHECK(variant>=0 && variant<6,"Invalid scheduled gate variant");
    HipDeviceGuard device_guard(data.device_id);
    if(variant==0)
    {
    hipLaunchKernelGGL(iq2r_scheduled_gate_wait3_kernel,
        dim3(2*static_cast<int>(get_num_cu_func())),dim3(64,8),0,getCurrentHIPStream(),
        static_cast<const opus::fp8_t*>(activations.data_ptr()),
        static_cast<const uint8_t*>(scales.data_ptr()),static_cast<const uint8_t*>(data.data_ptr()),
        static_cast<const uint8_t*>(auxiliary.data_ptr()),static_cast<const int32_t*>(tasks.data_ptr()),
        static_cast<const int32_t*>(task_count.data_ptr()),nullptr,
        static_cast<opus::fp8_t*>(output.data_ptr()),static_cast<uint8_t*>(output_scales.data_ptr()),
        routes,static_cast<int>(data.size(1)),static_cast<int>(auxiliary.size(1)));
    }
    if(variant==1)
    {
    hipLaunchKernelGGL(iq2r_scheduled_gate_register_unroll6_kernel,
        dim3(2*static_cast<int>(get_num_cu_func())),dim3(64,8),0,getCurrentHIPStream(),
        static_cast<const opus::fp8_t*>(activations.data_ptr()),
        static_cast<const uint8_t*>(scales.data_ptr()),static_cast<const uint8_t*>(data.data_ptr()),
        static_cast<const uint8_t*>(auxiliary.data_ptr()),static_cast<const int32_t*>(tasks.data_ptr()),
        static_cast<const int32_t*>(task_count.data_ptr()),nullptr,
        static_cast<opus::fp8_t*>(output.data_ptr()),static_cast<uint8_t*>(output_scales.data_ptr()),
        routes,static_cast<int>(data.size(1)),static_cast<int>(auxiliary.size(1)));
    }
    if(variant==2)
    {
    hipLaunchKernelGGL(iq2r_scheduled_gate_wait3_direct_kernel,
        dim3(2*static_cast<int>(get_num_cu_func())),dim3(64,8),0,getCurrentHIPStream(),
        static_cast<const opus::fp8_t*>(activations.data_ptr()),
        static_cast<const uint8_t*>(scales.data_ptr()),static_cast<const uint8_t*>(data.data_ptr()),
        static_cast<const uint8_t*>(auxiliary.data_ptr()),static_cast<const int32_t*>(tasks.data_ptr()),
        static_cast<const int32_t*>(task_count.data_ptr()),nullptr,
        static_cast<opus::fp8_t*>(output.data_ptr()),static_cast<uint8_t*>(output_scales.data_ptr()),
        routes,static_cast<int>(data.size(1)),static_cast<int>(auxiliary.size(1)));
    }
    if(variant==3)
    {
    hipLaunchKernelGGL(iq2r_scheduled_gate_register_unroll6_direct_kernel,
        dim3(2*static_cast<int>(get_num_cu_func())),dim3(64,8),0,getCurrentHIPStream(),
        static_cast<const opus::fp8_t*>(activations.data_ptr()),
        static_cast<const uint8_t*>(scales.data_ptr()),static_cast<const uint8_t*>(data.data_ptr()),
        static_cast<const uint8_t*>(auxiliary.data_ptr()),static_cast<const int32_t*>(tasks.data_ptr()),
        static_cast<const int32_t*>(task_count.data_ptr()),nullptr,
        static_cast<opus::fp8_t*>(output.data_ptr()),static_cast<uint8_t*>(output_scales.data_ptr()),
        routes,static_cast<int>(data.size(1)),static_cast<int>(auxiliary.size(1)));
    }
    if(variant==4)
    {
    hipLaunchKernelGGL(iq2r_scheduled_gate_wait3_direct_batch_kernel,
        dim3(2*static_cast<int>(get_num_cu_func())),dim3(64,8),0,getCurrentHIPStream(),
        static_cast<const opus::fp8_t*>(activations.data_ptr()),
        static_cast<const uint8_t*>(scales.data_ptr()),static_cast<const uint8_t*>(data.data_ptr()),
        static_cast<const uint8_t*>(auxiliary.data_ptr()),static_cast<const int32_t*>(tasks.data_ptr()),
        static_cast<const int32_t*>(task_count.data_ptr()),nullptr,
        static_cast<opus::fp8_t*>(output.data_ptr()),static_cast<uint8_t*>(output_scales.data_ptr()),
        routes,static_cast<int>(data.size(1)),static_cast<int>(auxiliary.size(1)));
    }
    if(variant==5)
    {
    hipLaunchKernelGGL(iq2r_scheduled_gate_register_unroll6_direct_batch_kernel,
        dim3(2*static_cast<int>(get_num_cu_func())),dim3(64,8),0,getCurrentHIPStream(),
        static_cast<const opus::fp8_t*>(activations.data_ptr()),
        static_cast<const uint8_t*>(scales.data_ptr()),static_cast<const uint8_t*>(data.data_ptr()),
        static_cast<const uint8_t*>(auxiliary.data_ptr()),static_cast<const int32_t*>(tasks.data_ptr()),
        static_cast<const int32_t*>(task_count.data_ptr()),nullptr,
        static_cast<opus::fp8_t*>(output.data_ptr()),static_cast<uint8_t*>(output_scales.data_ptr()),
        routes,static_cast<int>(data.size(1)),static_cast<int>(auxiliary.size(1)));
    }
    HIP_CALL_LAUNCH(hipGetLastError());
}

void iq2r_fused48_down_check(
    const aiter_tensor_t& activations, const aiter_tensor_t& scales,
    const aiter_tensor_t& data, const aiter_tensor_t& auxiliary,
    const aiter_tensor_t& route_map, const aiter_tensor_t& route_weights,
    const aiter_tensor_t& output)
{
    validate_weights(data,auxiliary,6144,256);
    const aiter_tensor_t* tensors[]={&activations,&scales,&data,&auxiliary,&route_map,&route_weights,&output};
    for(const auto* tensor:tensors)
        AITER_CHECK(tensor->is_gpu() && tensor->device_id==data.device_id && tensor->is_contiguous(),
                    "Fused IQ2R down/reduction requires contiguous tensors on the weight GPU");
    AITER_CHECK(data.size(0)==257 && activations.dim()==2 && activations.size(1)==256 &&
                activations.size(0)%9==0 && activations.dtype()==AITER_DTYPE_fp8,
                "Fused IQ2R down/reduction requires GLM TP8 intermediate FP8 [tokens*9,256]");
    const int routes=static_cast<int>(activations.size(0));
    AITER_CHECK(scales.dtype()==AITER_DTYPE_u8 && scales.dim()==2 && scales.size(0)==routes && scales.size(1)==8,
                "Fused IQ2R down/reduction requires row-major uint8 scales");
    AITER_CHECK(route_map.dtype()==AITER_DTYPE_i32 && route_map.dim()==1 && route_map.size(0)==routes,
                "Fused IQ2R down/reduction route map must be int32 [routes]");
    AITER_CHECK(route_weights.dtype()==AITER_DTYPE_fp32 && route_weights.dim()==2 &&
                route_weights.size(0)==routes/9 && route_weights.size(1)==9,
                "Fused IQ2R down/reduction weights must be float32 [tokens,9]");
    AITER_CHECK(output.dim()==2 && output.size(0)==routes/9 && output.size(1)==6144 &&
                output.dtype()==AITER_DTYPE_bf16,
                "Fused IQ2R down/reduction output has wrong shape or dtype");
}

void iq2r_down_token_fused48_out(
    const aiter_tensor_t& activations, const aiter_tensor_t& scales,
    const aiter_tensor_t& data, const aiter_tensor_t& auxiliary,
    const aiter_tensor_t& expert_ids, const aiter_tensor_t& scatter,
    const aiter_tensor_t& route_weights, aiter_tensor_t& output)
{
    iq2r_fused48_down_check(activations,scales,data,auxiliary,scatter,route_weights,output);
    const int tokens=static_cast<int>(output.size(0));
    AITER_CHECK((tokens==1 || tokens==2 || tokens==4),"Fused IQ2R down/reduction token-owned prototype targets small decode");
    AITER_CHECK(expert_ids.is_gpu() && expert_ids.device_id==data.device_id && expert_ids.dtype()==AITER_DTYPE_i32 &&
                expert_ids.dim()==2 && expert_ids.size(0)==tokens && expert_ids.size(1)==9 && expert_ids.is_contiguous(),
                "Fused IQ2R down/reduction expert IDs must be contiguous int32 [tokens,9]");
    HipDeviceGuard device_guard(data.device_id);
    hipLaunchKernelGGL(iq2r_down_token_fused48_kernel,dim3(128,tokens),dim3(64,6),0,getCurrentHIPStream(),
      static_cast<const opus::fp8_t*>(activations.data_ptr()),static_cast<const uint8_t*>(scales.data_ptr()),
      static_cast<const uint8_t*>(data.data_ptr()),static_cast<const uint8_t*>(auxiliary.data_ptr()),
      static_cast<const int32_t*>(expert_ids.data_ptr()),static_cast<const int32_t*>(scatter.data_ptr()),
      static_cast<const float*>(route_weights.data_ptr()),static_cast<__hip_bfloat16*>(output.data_ptr()),
      static_cast<int>(data.size(1)),static_cast<int>(auxiliary.size(1)));
    HIP_CALL_LAUNCH(hipGetLastError());
}

void iq2r_down_token_route9_out(
    const aiter_tensor_t& activations, const aiter_tensor_t& scales,
    const aiter_tensor_t& data, const aiter_tensor_t& auxiliary,
    const aiter_tensor_t& expert_ids, const aiter_tensor_t& scatter,
    const aiter_tensor_t& route_weights, aiter_tensor_t& output)
{
    iq2r_fused48_down_check(activations,scales,data,auxiliary,scatter,route_weights,output);
    const int tokens=static_cast<int>(output.size(0));
    AITER_CHECK((tokens==1 || tokens==2 || tokens==4),"Fused IQ2R down/reduction token-owned prototype targets small decode");
    AITER_CHECK(expert_ids.is_gpu() && expert_ids.device_id==data.device_id && expert_ids.dtype()==AITER_DTYPE_i32 &&
                expert_ids.dim()==2 && expert_ids.size(0)==tokens && expert_ids.size(1)==9 && expert_ids.is_contiguous(),
                "Fused IQ2R down/reduction expert IDs must be contiguous int32 [tokens,9]");
    HipDeviceGuard device_guard(data.device_id);
    hipLaunchKernelGGL(iq2r_down_token_route9_kernel,dim3(128,tokens),dim3(64,9),0,getCurrentHIPStream(),
      static_cast<const opus::fp8_t*>(activations.data_ptr()),static_cast<const uint8_t*>(scales.data_ptr()),
      static_cast<const uint8_t*>(data.data_ptr()),static_cast<const uint8_t*>(auxiliary.data_ptr()),
      static_cast<const int32_t*>(expert_ids.data_ptr()),static_cast<const int32_t*>(scatter.data_ptr()),
      static_cast<const float*>(route_weights.data_ptr()),static_cast<__hip_bfloat16*>(output.data_ptr()),
      static_cast<int>(data.size(1)),static_cast<int>(auxiliary.size(1)));
    HIP_CALL_LAUNCH(hipGetLastError());
}

void iq2r_fused48_down_check_tp4(
    const aiter_tensor_t& activations, const aiter_tensor_t& scales,
    const aiter_tensor_t& data, const aiter_tensor_t& auxiliary,
    const aiter_tensor_t& route_map, const aiter_tensor_t& route_weights,
    const aiter_tensor_t& output)
{
    validate_weights(data,auxiliary,6144,512);
    const aiter_tensor_t* tensors[]={&activations,&scales,&data,&auxiliary,&route_map,&route_weights,&output};
    for(const auto* tensor:tensors)
        AITER_CHECK(tensor->is_gpu() && tensor->device_id==data.device_id && tensor->is_contiguous(),
                    "Fused IQ2R down/reduction requires contiguous tensors on the weight GPU");
    AITER_CHECK(data.size(0)==257 && activations.dim()==2 && activations.size(1)==512 &&
                activations.size(0)%9==0 && activations.dtype()==AITER_DTYPE_fp8,
                "Fused IQ2R down/reduction requires GLM TP8 intermediate FP8 [tokens*9,256]");
    const int routes=static_cast<int>(activations.size(0));
    AITER_CHECK(scales.dtype()==AITER_DTYPE_u8 && scales.dim()==2 && scales.size(0)==routes && scales.size(1)==16,
                "Fused IQ2R down/reduction requires row-major uint8 scales");
    AITER_CHECK(route_map.dtype()==AITER_DTYPE_i32 && route_map.dim()==1 && route_map.size(0)==routes,
                "Fused IQ2R down/reduction route map must be int32 [routes]");
    AITER_CHECK(route_weights.dtype()==AITER_DTYPE_fp32 && route_weights.dim()==2 &&
                route_weights.size(0)==routes/9 && route_weights.size(1)==9,
                "Fused IQ2R down/reduction weights must be float32 [tokens,9]");
    AITER_CHECK(output.dim()==2 && output.size(0)==routes/9 && output.size(1)==6144 &&
                output.dtype()==AITER_DTYPE_bf16,
                "Fused IQ2R down/reduction output has wrong shape or dtype");
}

void iq2r_glm53_tp4_gate_out(
    const aiter_tensor_t& activations, const aiter_tensor_t& scales,
    const aiter_tensor_t& data, const aiter_tensor_t& auxiliary,
    const aiter_tensor_t& tasks, const aiter_tensor_t& task_count,
    aiter_tensor_t& output, aiter_tensor_t& output_scales, int64_t variant)
{
    const aiter_tensor_t* tensors[]={&activations,&scales,&data,&auxiliary,
                                    &tasks,&task_count,&output,&output_scales};
    for(const auto* tensor:tensors)
        AITER_CHECK(tensor->is_gpu() && tensor->device_id==data.device_id && tensor->is_contiguous(),
                    "E079 requires contiguous tensors on the weight GPU");
    AITER_CHECK(data.dtype()==AITER_DTYPE_u8 && data.dim()==2 && data.size(0)==257 &&
                data.size(1)==16*48*kQuadBytes, "E079 requires quad-packed GLM gate weights");
    AITER_CHECK(auxiliary.dtype()==AITER_DTYPE_u8 && auxiliary.dim()==2 &&
                auxiliary.size(0)==257 && auxiliary.size(1)==expected_auxiliary_bytes(1024),
                "E079 requires canonical gate auxiliary bytes");
    AITER_CHECK(activations.dtype()==AITER_DTYPE_fp8 && activations.dim()==2 &&
                activations.size(1)==6144 && activations.size(0)>=9 &&
                activations.size(0)<=144 && activations.size(0)%9==0,
                "E079 requires route-major FP8 for 1..16 top-9 tokens");
    const int routes=static_cast<int>(activations.size(0));
    AITER_CHECK(scales.dtype()==AITER_DTYPE_u8 && scales.dim()==2 &&
                scales.size(0)==routes && scales.size(1)==192,
                "E079 requires route-major E8M0 input scales");
    AITER_CHECK(tasks.dtype()==AITER_DTYPE_i32 && tasks.dim()==2 &&
                tasks.size(0)==routes && tasks.size(1)==3 &&
                task_count.dtype()==AITER_DTYPE_i32 && task_count.dim()==1 && task_count.size(0)==1,
                "E079 requires small-token direct/grouped tasks");
    AITER_CHECK(output.dtype()==AITER_DTYPE_fp8 && output.dim()==2 &&
                output.size(0)==routes && output.size(1)==512 &&
                output_scales.dtype()==AITER_DTYPE_u8 && output_scales.dim()==2 &&
                output_scales.size(0)==routes && output_scales.size(1)==16,
                "E079 requires FP8 intermediate [routes,256] and E8M0 [routes,8]");
    AITER_CHECK(variant>=0 && variant<6,"Invalid scheduled gate variant");
    HipDeviceGuard device_guard(data.device_id);
    if(variant==0)
    {
    hipLaunchKernelGGL(iq2r_scheduled_gate_wait3_kernel_tp4,
        dim3(2*static_cast<int>(get_num_cu_func())),dim3(64,8),0,getCurrentHIPStream(),
        static_cast<const opus::fp8_t*>(activations.data_ptr()),
        static_cast<const uint8_t*>(scales.data_ptr()),static_cast<const uint8_t*>(data.data_ptr()),
        static_cast<const uint8_t*>(auxiliary.data_ptr()),static_cast<const int32_t*>(tasks.data_ptr()),
        static_cast<const int32_t*>(task_count.data_ptr()),nullptr,
        static_cast<opus::fp8_t*>(output.data_ptr()),static_cast<uint8_t*>(output_scales.data_ptr()),
        routes,static_cast<int>(data.size(1)),static_cast<int>(auxiliary.size(1)));
    }
    if(variant==1)
    {
    hipLaunchKernelGGL(iq2r_scheduled_gate_register_unroll6_kernel_tp4,
        dim3(2*static_cast<int>(get_num_cu_func())),dim3(64,8),0,getCurrentHIPStream(),
        static_cast<const opus::fp8_t*>(activations.data_ptr()),
        static_cast<const uint8_t*>(scales.data_ptr()),static_cast<const uint8_t*>(data.data_ptr()),
        static_cast<const uint8_t*>(auxiliary.data_ptr()),static_cast<const int32_t*>(tasks.data_ptr()),
        static_cast<const int32_t*>(task_count.data_ptr()),nullptr,
        static_cast<opus::fp8_t*>(output.data_ptr()),static_cast<uint8_t*>(output_scales.data_ptr()),
        routes,static_cast<int>(data.size(1)),static_cast<int>(auxiliary.size(1)));
    }
    if(variant==2)
    {
    hipLaunchKernelGGL(iq2r_scheduled_gate_wait3_direct_kernel_tp4,
        dim3(2*static_cast<int>(get_num_cu_func())),dim3(64,8),0,getCurrentHIPStream(),
        static_cast<const opus::fp8_t*>(activations.data_ptr()),
        static_cast<const uint8_t*>(scales.data_ptr()),static_cast<const uint8_t*>(data.data_ptr()),
        static_cast<const uint8_t*>(auxiliary.data_ptr()),static_cast<const int32_t*>(tasks.data_ptr()),
        static_cast<const int32_t*>(task_count.data_ptr()),nullptr,
        static_cast<opus::fp8_t*>(output.data_ptr()),static_cast<uint8_t*>(output_scales.data_ptr()),
        routes,static_cast<int>(data.size(1)),static_cast<int>(auxiliary.size(1)));
    }
    if(variant==3)
    {
    hipLaunchKernelGGL(iq2r_scheduled_gate_register_unroll6_direct_kernel_tp4,
        dim3(2*static_cast<int>(get_num_cu_func())),dim3(64,8),0,getCurrentHIPStream(),
        static_cast<const opus::fp8_t*>(activations.data_ptr()),
        static_cast<const uint8_t*>(scales.data_ptr()),static_cast<const uint8_t*>(data.data_ptr()),
        static_cast<const uint8_t*>(auxiliary.data_ptr()),static_cast<const int32_t*>(tasks.data_ptr()),
        static_cast<const int32_t*>(task_count.data_ptr()),nullptr,
        static_cast<opus::fp8_t*>(output.data_ptr()),static_cast<uint8_t*>(output_scales.data_ptr()),
        routes,static_cast<int>(data.size(1)),static_cast<int>(auxiliary.size(1)));
    }
    if(variant==4)
    {
    hipLaunchKernelGGL(iq2r_scheduled_gate_wait3_direct_batch_kernel_tp4,
        dim3(2*static_cast<int>(get_num_cu_func())),dim3(64,8),0,getCurrentHIPStream(),
        static_cast<const opus::fp8_t*>(activations.data_ptr()),
        static_cast<const uint8_t*>(scales.data_ptr()),static_cast<const uint8_t*>(data.data_ptr()),
        static_cast<const uint8_t*>(auxiliary.data_ptr()),static_cast<const int32_t*>(tasks.data_ptr()),
        static_cast<const int32_t*>(task_count.data_ptr()),nullptr,
        static_cast<opus::fp8_t*>(output.data_ptr()),static_cast<uint8_t*>(output_scales.data_ptr()),
        routes,static_cast<int>(data.size(1)),static_cast<int>(auxiliary.size(1)));
    }
    if(variant==5)
    {
    hipLaunchKernelGGL(iq2r_scheduled_gate_register_unroll6_direct_batch_kernel_tp4,
        dim3(2*static_cast<int>(get_num_cu_func())),dim3(64,8),0,getCurrentHIPStream(),
        static_cast<const opus::fp8_t*>(activations.data_ptr()),
        static_cast<const uint8_t*>(scales.data_ptr()),static_cast<const uint8_t*>(data.data_ptr()),
        static_cast<const uint8_t*>(auxiliary.data_ptr()),static_cast<const int32_t*>(tasks.data_ptr()),
        static_cast<const int32_t*>(task_count.data_ptr()),nullptr,
        static_cast<opus::fp8_t*>(output.data_ptr()),static_cast<uint8_t*>(output_scales.data_ptr()),
        routes,static_cast<int>(data.size(1)),static_cast<int>(auxiliary.size(1)));
    }
    HIP_CALL_LAUNCH(hipGetLastError());
}

void iq2r_glm53_tp4_down_out(
    const aiter_tensor_t& activations, const aiter_tensor_t& scales,
    const aiter_tensor_t& data, const aiter_tensor_t& auxiliary,
    const aiter_tensor_t& tasks, const aiter_tensor_t& task_count,
    aiter_tensor_t& output, int64_t grid_multiplier, int64_t variant)
{
    const int routes=static_cast<int>(activations.size(0));
    AITER_CHECK(((routes>=9 && routes<=144 && routes%9==0) || routes==288),"Short-K down requires 1..16 or 32 top-9 tokens");
    AITER_CHECK(variant==2,"Invalid short-K down variant");
    validate_weights(data, auxiliary, 6144, 512);
    const aiter_tensor_t* tensors[]={&activations,&scales,&data,&auxiliary,&tasks,&task_count,&output};
    for(const auto* tensor:tensors)
        AITER_CHECK(tensor->is_gpu() && tensor->device_id==data.device_id && tensor->is_contiguous(),
                    "Short-K tensors must be contiguous on the weight GPU");
    AITER_CHECK(data.size(0)==257 && activations.dtype()==AITER_DTYPE_fp8 &&
                activations.dim()==2 && activations.size(0)==routes && activations.size(1)==512,
                "Short-K requires GLM TP8 intermediate FP8 [routes,256]");
    AITER_CHECK(scales.dtype()==AITER_DTYPE_u8 && scales.dim()==2 && scales.size(0)==routes && scales.size(1)==16,
                "Short-K requires row-major E8M0 scales [routes,8]");
    AITER_CHECK(tasks.dtype()==AITER_DTYPE_i32 && tasks.dim()==2 && tasks.size(0)==(routes==288 ? 267 : routes) && tasks.size(1)==3 &&
                task_count.dtype()==AITER_DTYPE_i32 && task_count.dim()==1 && task_count.size(0)==1,
                "Short-K requires small-token tasks [routes,3]");
    AITER_CHECK(output.dtype()==AITER_DTYPE_bf16 && output.dim()==2 && output.size(0)==routes && output.size(1)==6144,
                "Short-K requires BF16 route output [routes,6144]");
    AITER_CHECK(grid_multiplier==2 || grid_multiplier==4 || grid_multiplier==8,
                "Short-K grid multiplier must be 2, 4 or 8");
    HipDeviceGuard device_guard(data.device_id);
    if(variant==2)
    {
    hipLaunchKernelGGL(iq2r_scheduled_small_register_single_epilogue_kernel_tp4,
        dim3(grid_multiplier*static_cast<int>(get_num_cu_func())),dim3(64,4),0,getCurrentHIPStream(),
        static_cast<const opus::fp8_t*>(activations.data_ptr()),static_cast<const uint8_t*>(scales.data_ptr()),
        static_cast<const uint8_t*>(data.data_ptr()),static_cast<const uint8_t*>(auxiliary.data_ptr()),
        static_cast<const int32_t*>(tasks.data_ptr()),static_cast<const int32_t*>(task_count.data_ptr()),
        nullptr,static_cast<__hip_bfloat16*>(output.data_ptr()),nullptr,nullptr,routes,6144,512,257,
        static_cast<int>(data.size(1)),static_cast<int>(auxiliary.size(1)),0.0f,1.0f,0.0f);
    }
    HIP_CALL_LAUNCH(hipGetLastError());
}

void iq2r_glm53_tp4_route9_out(
    const aiter_tensor_t& activations, const aiter_tensor_t& scales,
    const aiter_tensor_t& data, const aiter_tensor_t& auxiliary,
    const aiter_tensor_t& expert_ids, const aiter_tensor_t& scatter,
    const aiter_tensor_t& route_weights, aiter_tensor_t& output)
{
    iq2r_fused48_down_check_tp4(activations,scales,data,auxiliary,scatter,route_weights,output);
    const int tokens=static_cast<int>(output.size(0));
    AITER_CHECK((tokens==1 || tokens==2 || tokens==4),"Fused IQ2R down/reduction token-owned prototype targets small decode");
    AITER_CHECK(expert_ids.is_gpu() && expert_ids.device_id==data.device_id && expert_ids.dtype()==AITER_DTYPE_i32 &&
                expert_ids.dim()==2 && expert_ids.size(0)==tokens && expert_ids.size(1)==9 && expert_ids.is_contiguous(),
                "Fused IQ2R down/reduction expert IDs must be contiguous int32 [tokens,9]");
    HipDeviceGuard device_guard(data.device_id);
    hipLaunchKernelGGL(iq2r_down_token_route9_kernel_tp4,dim3(128,tokens),dim3(64,9),0,getCurrentHIPStream(),
      static_cast<const opus::fp8_t*>(activations.data_ptr()),static_cast<const uint8_t*>(scales.data_ptr()),
      static_cast<const uint8_t*>(data.data_ptr()),static_cast<const uint8_t*>(auxiliary.data_ptr()),
      static_cast<const int32_t*>(expert_ids.data_ptr()),static_cast<const int32_t*>(scatter.data_ptr()),
      static_cast<const float*>(route_weights.data_ptr()),static_cast<__hip_bfloat16*>(output.data_ptr()),
      static_cast<int>(data.size(1)),static_cast<int>(auxiliary.size(1)));
    HIP_CALL_LAUNCH(hipGetLastError());
}
void iq2r_glm53_tp4_indexed_gate_out(
    const aiter_tensor_t& activations, const aiter_tensor_t& scales,
    const aiter_tensor_t& data, const aiter_tensor_t& auxiliary,
    const aiter_tensor_t& tasks, const aiter_tensor_t& task_count,
    const aiter_tensor_t& gather, aiter_tensor_t& output,
    aiter_tensor_t& output_scales, int64_t rows_per_cta)
{
    AITER_CHECK(data.dtype()==AITER_DTYPE_u8 && data.dim()==2 &&
                data.size(0)==257 && data.size(1)==16*48*kQuadBytes,
                "E071 requires quad-packed uint8 gate [257,884736]");
    AITER_CHECK(auxiliary.dtype()==AITER_DTYPE_u8 && auxiliary.dim()==2 &&
                auxiliary.size(0)==257 && auxiliary.size(1)==expected_auxiliary_bytes(1024),
                "E071 requires unchanged canonical gate auxiliary bytes");
    const aiter_tensor_t* tensors[]={&activations,&scales,&data,&auxiliary,&tasks,&task_count,&gather,&output,&output_scales};
    for(const auto* tensor:tensors)
        AITER_CHECK(tensor->is_gpu() && tensor->device_id==data.device_id && tensor->is_contiguous(),
                    "E071 requires contiguous tensors on the weight GPU");
    AITER_CHECK(activations.dim()==2 && activations.size(1)==6144 && activations.dtype()==AITER_DTYPE_fp8 && data.size(0)==257,
                "E071 requires token-major FP8 GLM TP8 inputs");
    const int tokens=static_cast<int>(activations.size(0)), routes=tokens*9;
    AITER_CHECK(tokens==32 || (tokens>=64 && tokens<=227) || tokens==256 || tokens==1536 || (tokens>=2048 && tokens<=4096),"E071 requires qualified M32 decode");
    AITER_CHECK(scales.dim()==2 && scales.size(0)==tokens && scales.size(1)==192 && scales.dtype()==AITER_DTYPE_u8,
                "E071 requires row-major uint8 scales");
    AITER_CHECK(tasks.dtype()==AITER_DTYPE_i32 && tasks.dim()==2 && tasks.size(1)==3 &&
                tasks.size(0)==(routes+rows_per_cta-1)/rows_per_cta+min(routes,258),"E071 requires M32 tasks");
    AITER_CHECK(task_count.dtype()==AITER_DTYPE_i32 && task_count.dim()==1 && task_count.size(0)==1,
                "E071 requires scalar int32 task count");
    AITER_CHECK(gather.dtype()==AITER_DTYPE_i32 && gather.dim()==1 && gather.size(0)==routes,
                "E071 requires int32 gather [routes]");
    AITER_CHECK(output.dtype()==AITER_DTYPE_fp8 && output.dim()==2 && output.size(0)==routes && output.size(1)==512 &&
                output_scales.dtype()==AITER_DTYPE_u8 && output_scales.dim()==2 && output_scales.size(0)==routes && output_scales.size(1)==16,
                "E071 requires FP8 intermediate [routes,256], scales [routes,8]");
    AITER_CHECK((rows_per_cta==32 && tokens<=256) || (rows_per_cta==64 && tokens>=1536),"E071 rows_per_cta must be 32");
    const char* direct_setting=std::getenv("IQ2R_GLM53_XCD_SPARSE_GATE");
    const bool direct=direct_setting && std::strcmp(direct_setting,"1")==0 && tokens==128;
    if(direct) {
        static thread_local bool audited=false;
        const char* audit=std::getenv("ATOM_IQ2R_AUDIT");
        if(audit && std::strcmp(audit,"1")==0 && !audited)
        {{
            audited=true;
            std::fprintf(stderr,"IQ2R_XCD_SPARSE_GATE device=%d tokens=%d tp=4 kernel=iq2r_gate_quad_sparse_kernel_tp4_xcd\n",data.device_id,tokens);
        }}
#define E125_XCD_LAUNCH(MATOMS, TASKROWS) \
    hipLaunchKernelGGL((iq2r_gate_quad_sparse_kernel_tp4_xcd<MATOMS,TASKROWS>), \
      dim3(2*static_cast<int>(get_num_cu_func())),dim3(64,8),0,getCurrentHIPStream(), \
      static_cast<const opus::fp8_t*>(activations.data_ptr()),static_cast<const uint8_t*>(scales.data_ptr()), \
      static_cast<const uint8_t*>(data.data_ptr()),static_cast<const uint8_t*>(auxiliary.data_ptr()), \
      static_cast<const int32_t*>(tasks.data_ptr()),static_cast<const int32_t*>(task_count.data_ptr()), \
      static_cast<const int32_t*>(gather.data_ptr()),static_cast<opus::fp8_t*>(output.data_ptr()), \
      static_cast<uint8_t*>(output_scales.data_ptr()),routes,static_cast<int>(data.size(1)),static_cast<int>(auxiliary.size(1)))
    if(rows_per_cta==32) E125_XCD_LAUNCH(2,32);
    else E125_XCD_LAUNCH(4,64);
#undef E125_XCD_LAUNCH
    } else {
#define E125_LAUNCH(MATOMS, TASKROWS) \
    hipLaunchKernelGGL((iq2r_gate_quad_sparse_kernel_tp4<MATOMS,TASKROWS>), \
      dim3(2*static_cast<int>(get_num_cu_func())),dim3(64,8),0,getCurrentHIPStream(), \
      static_cast<const opus::fp8_t*>(activations.data_ptr()),static_cast<const uint8_t*>(scales.data_ptr()), \
      static_cast<const uint8_t*>(data.data_ptr()),static_cast<const uint8_t*>(auxiliary.data_ptr()), \
      static_cast<const int32_t*>(tasks.data_ptr()),static_cast<const int32_t*>(task_count.data_ptr()), \
      static_cast<const int32_t*>(gather.data_ptr()),static_cast<opus::fp8_t*>(output.data_ptr()), \
      static_cast<uint8_t*>(output_scales.data_ptr()),routes,static_cast<int>(data.size(1)),static_cast<int>(auxiliary.size(1)))
    if(rows_per_cta==32) E125_LAUNCH(2,32);
    else E125_LAUNCH(4,64);
#undef E125_LAUNCH
    }
    HIP_CALL_LAUNCH(hipGetLastError());
}

void iq2r_glm53_tp4_large_down_out(
    const aiter_tensor_t& activations, const aiter_tensor_t& scales,
    const aiter_tensor_t& data, const aiter_tensor_t& auxiliary,
    const aiter_tensor_t& tasks, const aiter_tensor_t& task_count,
    aiter_tensor_t& output, int64_t grid_multiplier, int64_t variant)
{
    validate_weights(data, auxiliary, 6144, 512);
    const aiter_tensor_t* tensors[]={&activations,&scales,&data,&auxiliary,&tasks,&task_count,&output};
    for(const auto* tensor:tensors)
        AITER_CHECK(tensor->is_gpu() && tensor->device_id==data.device_id && tensor->is_contiguous(),
                    "E085 tensors must be contiguous on the weight GPU");
    AITER_CHECK(data.size(0)==257 && activations.dtype()==AITER_DTYPE_fp8 &&
                activations.dim()==2 && activations.size(0)==2304 && activations.size(1)==512,
                "E085 requires GLM TP8 M256 intermediate FP8 [2304,256]");
    AITER_CHECK(scales.dtype()==AITER_DTYPE_u8 && scales.dim()==2 && scales.size(0)==2304 && scales.size(1)==16,
                "E085 requires row-major E8M0 scales [2304,8]");
    AITER_CHECK(tasks.dtype()==AITER_DTYPE_i32 && tasks.dim()==2 && tasks.size(0)==330 && tasks.size(1)==3 &&
                task_count.dtype()==AITER_DTYPE_i32 && task_count.dim()==1 && task_count.size(0)==1,
                "E085 requires GLM M32 tasks [330,3]");
    AITER_CHECK(output.dtype()==AITER_DTYPE_bf16 && output.dim()==2 && output.size(0)==2304 && output.size(1)==6144,
                "E085 requires BF16 route output [2304,6144]");
    AITER_CHECK(grid_multiplier==2 || grid_multiplier==4 || grid_multiplier==8,
                "E085 grid multiplier must be 2, 4 or 8");
    AITER_CHECK(variant>=0 && variant<4,"Invalid scheduled down variant");
    HipDeviceGuard device_guard(data.device_id);
    if(variant==0)
    {
    hipLaunchKernelGGL(iq2r_scheduled_large_fixed_kn_kernel_tp4,
        dim3(grid_multiplier*static_cast<int>(get_num_cu_func())),dim3(64,4),0,getCurrentHIPStream(),
        static_cast<const opus::fp8_t*>(activations.data_ptr()),static_cast<const uint8_t*>(scales.data_ptr()),
        static_cast<const uint8_t*>(data.data_ptr()),static_cast<const uint8_t*>(auxiliary.data_ptr()),
        static_cast<const int32_t*>(tasks.data_ptr()),static_cast<const int32_t*>(task_count.data_ptr()),
        nullptr,static_cast<__hip_bfloat16*>(output.data_ptr()),2304,6144,512,257,
        static_cast<int>(data.size(1)),static_cast<int>(auxiliary.size(1)));
    }
    if(variant==1)
    {
    hipLaunchKernelGGL(iq2r_scheduled_large_register_prefetch_uniform_kernel_tp4,
        dim3(grid_multiplier*static_cast<int>(get_num_cu_func())),dim3(64,4),0,getCurrentHIPStream(),
        static_cast<const opus::fp8_t*>(activations.data_ptr()),static_cast<const uint8_t*>(scales.data_ptr()),
        static_cast<const uint8_t*>(data.data_ptr()),static_cast<const uint8_t*>(auxiliary.data_ptr()),
        static_cast<const int32_t*>(tasks.data_ptr()),static_cast<const int32_t*>(task_count.data_ptr()),
        nullptr,static_cast<__hip_bfloat16*>(output.data_ptr()),2304,6144,512,257,
        static_cast<int>(data.size(1)),static_cast<int>(auxiliary.size(1)));
    }
    if(variant==2)
    {
    hipLaunchKernelGGL(iq2r_scheduled_large_fixed_kn_deferred_scale_kernel_tp4,
        dim3(grid_multiplier*static_cast<int>(get_num_cu_func())),dim3(64,4),0,getCurrentHIPStream(),
        static_cast<const opus::fp8_t*>(activations.data_ptr()),static_cast<const uint8_t*>(scales.data_ptr()),
        static_cast<const uint8_t*>(data.data_ptr()),static_cast<const uint8_t*>(auxiliary.data_ptr()),
        static_cast<const int32_t*>(tasks.data_ptr()),static_cast<const int32_t*>(task_count.data_ptr()),
        nullptr,static_cast<__hip_bfloat16*>(output.data_ptr()),2304,6144,512,257,
        static_cast<int>(data.size(1)),static_cast<int>(auxiliary.size(1)));
    }
    if(variant==3)
    {
    hipLaunchKernelGGL(iq2r_scheduled_large_register_prefetch_uniform_deferred_scale_kernel_tp4,
        dim3(grid_multiplier*static_cast<int>(get_num_cu_func())),dim3(64,4),0,getCurrentHIPStream(),
        static_cast<const opus::fp8_t*>(activations.data_ptr()),static_cast<const uint8_t*>(scales.data_ptr()),
        static_cast<const uint8_t*>(data.data_ptr()),static_cast<const uint8_t*>(auxiliary.data_ptr()),
        static_cast<const int32_t*>(tasks.data_ptr()),static_cast<const int32_t*>(task_count.data_ptr()),
        nullptr,static_cast<__hip_bfloat16*>(output.data_ptr()),2304,6144,512,257,
        static_cast<int>(data.size(1)),static_cast<int>(auxiliary.size(1)));
    }
    HIP_CALL_LAUNCH(hipGetLastError());
}
void iq2r_down_token_pair9_out(
    const aiter_tensor_t& activations, const aiter_tensor_t& scales,
    const aiter_tensor_t& data, const aiter_tensor_t& auxiliary,
    const aiter_tensor_t& expert_ids, const aiter_tensor_t& scatter,
    const aiter_tensor_t& route_weights, aiter_tensor_t& output,int64_t group_tokens)
{
    iq2r_fused48_down_check(activations,scales,data,auxiliary,scatter,route_weights,output);
    const int tokens=static_cast<int>(output.size(0));
    AITER_CHECK((tokens==4 || tokens==8 || tokens==16),"Fused IQ2R down/reduction token-owned prototype targets small decode");
    AITER_CHECK(expert_ids.is_gpu() && expert_ids.device_id==data.device_id && expert_ids.dtype()==AITER_DTYPE_i32 &&
                expert_ids.dim()==2 && expert_ids.size(0)==tokens && expert_ids.size(1)==9 && expert_ids.is_contiguous(),
                "Fused IQ2R down/reduction expert IDs must be contiguous int32 [tokens,9]");
    HipDeviceGuard device_guard(data.device_id);
    AITER_CHECK(group_tokens==2 || group_tokens==4,"E141 group must be two or four tokens");
    if(group_tokens==2) {
    hipLaunchKernelGGL((e141_down_grouped_routes<2>),dim3(128,tokens/2),dim3(64,9),0,getCurrentHIPStream(),
      static_cast<const opus::fp8_t*>(activations.data_ptr()),static_cast<const uint8_t*>(scales.data_ptr()),
      static_cast<const uint8_t*>(data.data_ptr()),static_cast<const uint8_t*>(auxiliary.data_ptr()),
      static_cast<const int32_t*>(expert_ids.data_ptr()),static_cast<const int32_t*>(scatter.data_ptr()),
      static_cast<const float*>(route_weights.data_ptr()),static_cast<__hip_bfloat16*>(output.data_ptr()),
      static_cast<int>(data.size(1)),static_cast<int>(auxiliary.size(1)),tokens);
    } else {
    hipLaunchKernelGGL((e141_down_grouped_routes<4>),dim3(128,tokens/4),dim3(64,9),0,getCurrentHIPStream(),
      static_cast<const opus::fp8_t*>(activations.data_ptr()),static_cast<const uint8_t*>(scales.data_ptr()),
      static_cast<const uint8_t*>(data.data_ptr()),static_cast<const uint8_t*>(auxiliary.data_ptr()),
      static_cast<const int32_t*>(expert_ids.data_ptr()),static_cast<const int32_t*>(scatter.data_ptr()),
      static_cast<const float*>(route_weights.data_ptr()),static_cast<__hip_bfloat16*>(output.data_ptr()),
      static_cast<int>(data.size(1)),static_cast<int>(auxiliary.size(1)),tokens);
    }
    HIP_CALL_LAUNCH(hipGetLastError());
}
void iq2r_glm53_dense_gate_out(
    const aiter_tensor_t& activations, const aiter_tensor_t& scales,
    const aiter_tensor_t& data, const aiter_tensor_t& auxiliary,
    const aiter_tensor_t& tasks, const aiter_tensor_t& task_count,
    const aiter_tensor_t& gather, aiter_tensor_t& output,
    aiter_tensor_t& output_scales, int64_t rows_per_cta,int64_t variant)
{
    AITER_CHECK(data.dtype()==AITER_DTYPE_u8 && data.dim()==2 &&
                data.size(0)==257 && data.size(1)==8*48*kQuadBytes,
                "E071 requires quad-packed uint8 gate [257,884736]");
    AITER_CHECK(auxiliary.dtype()==AITER_DTYPE_u8 && auxiliary.dim()==2 &&
                auxiliary.size(0)==257 && auxiliary.size(1)==expected_auxiliary_bytes(512),
                "E071 requires unchanged canonical gate auxiliary bytes");
    const aiter_tensor_t* tensors[]={&activations,&scales,&data,&auxiliary,&tasks,&task_count,&gather,&output,&output_scales};
    for(const auto* tensor:tensors)
        AITER_CHECK(tensor->is_gpu() && tensor->device_id==data.device_id && tensor->is_contiguous(),
                    "E071 requires contiguous tensors on the weight GPU");
    AITER_CHECK(activations.dim()==2 && activations.size(1)==6144 && activations.dtype()==AITER_DTYPE_fp8 && data.size(0)==257,
                "E071 requires token-major FP8 GLM TP8 inputs");
    const int tokens=static_cast<int>(activations.size(0)), routes=tokens*9;
    AITER_CHECK(tokens==32 || (tokens>=64 && tokens<=227) || tokens==256 || tokens==1536 || (tokens>=2048 && tokens<=4096),"E071 requires qualified M32 decode");
    AITER_CHECK(scales.dim()==2 && scales.size(0)==tokens && scales.size(1)==192 && scales.dtype()==AITER_DTYPE_u8,
                "E071 requires row-major uint8 scales");
    AITER_CHECK(tasks.dtype()==AITER_DTYPE_i32 && tasks.dim()==2 && tasks.size(1)==3 &&
                tasks.size(0)==(routes+rows_per_cta-1)/rows_per_cta+min(routes,258),"E071 requires M32 tasks");
    AITER_CHECK(task_count.dtype()==AITER_DTYPE_i32 && task_count.dim()==1 && task_count.size(0)==1,
                "E071 requires scalar int32 task count");
    AITER_CHECK(gather.dtype()==AITER_DTYPE_i32 && gather.dim()==1 && gather.size(0)==routes,
                "E071 requires int32 gather [routes]");
    AITER_CHECK(output.dtype()==AITER_DTYPE_fp8 && output.dim()==2 && output.size(0)==routes && output.size(1)==256 &&
                output_scales.dtype()==AITER_DTYPE_u8 && output_scales.dim()==2 && output_scales.size(0)==routes && output_scales.size(1)==8,
                "E071 requires FP8 intermediate [routes,256], scales [routes,8]");
    AITER_CHECK((rows_per_cta==32 && tokens<=256) || (rows_per_cta==64 && tokens>=1536),"E127 invalid task rows");
    AITER_CHECK(variant==0 || (variant==1 && rows_per_cta==64),"E127 invalid pipeline variant");
#define E127_LAUNCH(KERNEL, TASKROWS) \
    hipLaunchKernelGGL((KERNEL<TASKROWS>), \
      dim3(2*static_cast<int>(get_num_cu_func())),dim3(64,8),0,getCurrentHIPStream(), \
      static_cast<const opus::fp8_t*>(activations.data_ptr()),static_cast<const uint8_t*>(scales.data_ptr()), \
      static_cast<const uint8_t*>(data.data_ptr()),static_cast<const uint8_t*>(auxiliary.data_ptr()), \
      static_cast<const int32_t*>(tasks.data_ptr()),static_cast<const int32_t*>(task_count.data_ptr()), \
      static_cast<const int32_t*>(gather.data_ptr()),static_cast<opus::fp8_t*>(output.data_ptr()), \
      static_cast<uint8_t*>(output_scales.data_ptr()),routes,static_cast<int>(data.size(1)),static_cast<int>(auxiliary.size(1)))
    if(rows_per_cta==32) E127_LAUNCH(e127_m32_register,32);
    else if(variant==0) E127_LAUNCH(e127_m64_register,64);
    else E127_LAUNCH(e127_m64_ab_unroll1,64);
#undef E127_LAUNCH
    HIP_CALL_LAUNCH(hipGetLastError());
}
void iq2r_down_token_adaptive9_out(
    const aiter_tensor_t& activations, const aiter_tensor_t& scales,
    const aiter_tensor_t& data, const aiter_tensor_t& auxiliary,
    const aiter_tensor_t& expert_ids, const aiter_tensor_t& scatter,
    const aiter_tensor_t& route_weights, aiter_tensor_t& output,const aiter_tensor_t& task_count,const aiter_tensor_t& task_table)
{
    iq2r_fused48_down_check(activations,scales,data,auxiliary,scatter,route_weights,output);
    const int tokens=static_cast<int>(output.size(0));
    AITER_CHECK(tokens==8,"Fused IQ2R down/reduction token-owned prototype targets small decode");
    AITER_CHECK(expert_ids.is_gpu() && expert_ids.device_id==data.device_id && expert_ids.dtype()==AITER_DTYPE_i32 &&
                expert_ids.dim()==2 && expert_ids.size(0)==tokens && expert_ids.size(1)==9 && expert_ids.is_contiguous(),
                "Fused IQ2R down/reduction expert IDs must be contiguous int32 [tokens,9]");
    HipDeviceGuard device_guard(data.device_id);
    AITER_CHECK(task_count.is_gpu() && task_count.device_id==data.device_id && task_count.dtype()==AITER_DTYPE_i32 && task_count.numel()==1,"E161 invalid task count");
    AITER_CHECK(task_table.is_gpu() && task_table.device_id==data.device_id &&
                task_table.is_contiguous() && task_table.dtype()==AITER_DTYPE_i32 &&
                task_table.dim()==2 && task_table.size(0)>=tokens*9 && task_table.size(1)==3,
                "E172 requires the compact task table");
    hipLaunchKernelGGL(e172_compact_down_routes,dim3(128,tokens/2),dim3(64,12),0,getCurrentHIPStream(),
      static_cast<const opus::fp8_t*>(activations.data_ptr()),static_cast<const uint8_t*>(scales.data_ptr()),
      static_cast<const uint8_t*>(data.data_ptr()),static_cast<const uint8_t*>(auxiliary.data_ptr()),
      static_cast<const int32_t*>(expert_ids.data_ptr()),static_cast<const int32_t*>(scatter.data_ptr()),
      static_cast<const float*>(route_weights.data_ptr()),static_cast<__hip_bfloat16*>(output.data_ptr()),
      static_cast<int>(data.size(1)),static_cast<int>(auxiliary.size(1)),tokens,static_cast<const int32_t*>(task_count.data_ptr()),static_cast<const int32_t*>(task_table.data_ptr()));
    HIP_CALL_LAUNCH(hipGetLastError());
}
} // namespace aiter
