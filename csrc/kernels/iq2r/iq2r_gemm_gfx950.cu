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

// Reorder logical tiles so neighboring ranges stay on the same XCD despite
// the hardware's round-robin workgroup placement.  This improves reuse of an
// expert's compact weight/codebook working set without changing the logical
// task order or output layout.
__device__ __forceinline__ int iq2r_remap_xcd(int block_id,
                                               int total_tiles,
                                               int num_xcds = 8)
{
    const int ids_per_xcd = (total_tiles + num_xcds - 1) / num_xcds;
    int tall_xcds = total_tiles % num_xcds;
    tall_xcds = tall_xcds == 0 ? num_xcds : tall_xcds;
    const int xcd = block_id % num_xcds;
    const int local_id = block_id / num_xcds;
    return xcd < tall_xcds
               ? xcd * ids_per_xcd + local_id
               : tall_xcds * ids_per_xcd +
                     (xcd - tall_xcds) * (ids_per_xcd - 1) + local_id;
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
                if(k_tile + 1 < k_tiles)
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

// Two-stage high-M pipeline for the GPT-OSS K=2880 expert shapes.  Each wave
// keeps two compact IQ2R weight triplets in a register ring while the whole
// workgroup stages two (MAtoms*16)x128 activation tiles through LDS. The
// counted VMEM staircase retires only the current tile, leaving the next tile
// outstanding across codebook expansion and the six MFMAs per wave.
template<int MAtoms,
         int PhysicalWaves,
         bool StagedScales,
         bool TaskPersistent,
         bool PinToAgpr = false,
         bool UseBias = false>
__global__ __launch_bounds__(64 * PhysicalWaves,
                             PhysicalWaves == 4 && MAtoms == 2 ? 2 : 1)
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
    static_assert(MAtoms == 2 || MAtoms == 4);
    static_assert(PhysicalWaves == 4 || PhysicalWaves == 8);
    static_assert(PhysicalWaves == 4 || MAtoms == 4);
    constexpr int kMAtoms = MAtoms;
    constexpr int kNAtomsPerWave = 3;
    constexpr int kOutputColumns =
        PhysicalWaves * kNAtomsPerWave * kTileN;
    constexpr int kKTiles = 23;
    constexpr int kActivationColumns = kTileK / sizeof(uint64_t);
    constexpr int kActivationElements = kMAtoms * kTileN * kActivationColumns;
    constexpr int kThreads = PhysicalWaves * 64;
    constexpr int kActivationColumnsPerLoad = 16;
    constexpr int kActivationLoadColumns =
        kTileK / kActivationColumnsPerLoad;
    constexpr int kActivationLoadRows = kThreads / kActivationLoadColumns;
    static_assert(kThreads % kActivationLoadColumns == 0);
    static_assert((kMAtoms * kTileN) % kActivationLoadRows == 0);
    constexpr int kActivationLoadsPerThread =
        (kMAtoms * kTileN) / kActivationLoadRows;
    constexpr int kScaleGroupsPerTile = kTileK / kScaleBlock;
    constexpr int kScaleElements = kMAtoms * kTileN * kScaleGroupsPerTile;

    struct InputPayload
    {
        alignas(16) uint64_t activation_cache[2][kActivationElements];
        alignas(16) uint8_t
            scale_cache[StagedScales ? 2 * kScaleElements : 1];
    };

    struct SharedStorage
    {
        alignas(16) uint64_t codebook[kCodebookBytes / sizeof(uint64_t)];
        union
        {
            InputPayload input;
            alignas(16) __hip_bfloat16
                epilogue_cache[2][kTileN * kOutputColumns];
        } payload;
    };
    struct ActivationSlot
    {
        opus::vector_t<uint8_t, 16> data[kActivationLoadsPerThread];
        uint32_t scales[StagedScales ? 1 : kMAtoms];
        uint8_t staged_scale;
    };
    __shared__ SharedStorage shared;

    const int lane = static_cast<int>(threadIdx.x);
    const int wave = static_cast<int>(threadIdx.y);
    const int linear_thread = wave * 64 + lane;
    const int lane_row = lane % 16;
    const int lane_group = lane / 16;
    const int load_column = linear_thread % kActivationLoadColumns;
    const int load_row = linear_thread / kActivationLoadColumns;
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
        const int remapped_work_index = TaskPersistent
                                            ? work_index
                                            : iq2r_remap_xcd(work_index,
                                                             total_tiles);
        const int task_index =
            TaskPersistent ? fixed_task : remapped_work_index / n_tiles;
        const int n_tile_index = TaskPersistent
                                     ? remapped_work_index
                                     : remapped_work_index % n_tiles;
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
                entry += kThreads)
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
                for(int load_atom = 0;
                    load_atom < kActivationLoadsPerThread;
                    ++load_atom)
                {
                    const int input_row =
                        row_base + load_row +
                        load_atom * kActivationLoadRows;
                    if(input_row < sub_end && input_column + 15 < K)
                        activation_ring[slot].data[load_atom] =
                            activation_buffer.template load<16>(
                                input_row * K + input_column);
                    else
                        activation_ring[slot].data[load_atom] = {};
                }

                if constexpr(StagedScales)
                {
                    // Keep the scale request count uniform across all waves so
                    // the counted VMEM wait below remains valid. Eight-wave
                    // workgroups load each unique scale twice; four-wave
                    // workgroups load it exactly once.
                    const int scale_index = linear_thread % kScaleElements;
                    const int scale_row =
                        row_base + scale_index / kScaleGroupsPerTile;
                    const int scale_group = scale_index % kScaleGroupsPerTile;
                    if(scale_row < sub_end)
                        activation_ring[slot].staged_scale =
                            scale_buffer.template load<1>(
                                scale_row * (K / kScaleBlock) +
                                k_tile * kScaleGroupsPerTile + scale_group)[0];
                    else
                        activation_ring[slot].staged_scale = 127;
                }
                else
                {
#pragma unroll
                    for(int m_atom = 0; m_atom < kMAtoms; ++m_atom)
                    {
                        const int scale_row =
                            row_base + m_atom * kTileN + lane_row;
                        if(scale_row < sub_end)
                            activation_ring[slot].scales[m_atom] =
                                scale_buffer.template load<1>(
                                    scale_row * (K / kScaleBlock) +
                                    k_tile * kScaleGroupsPerTile + lane_group)[0];
                        else
                            activation_ring[slot].scales[m_atom] = 127;
                    }
                }

                const auto paired = data_buffer.template load<16>(
                    next_data_base + lane * 16);
                weight_ring[slot].paired =
                    __builtin_bit_cast(uint4, paired);
                const int third_base = next_data_base + kAtomTwoOffset +
                                       lane * kAtomTwoRecordBytes;
                const auto third_record =
                    data_buffer.template load<12>(third_base);
                weight_ring[slot].third =
                    __builtin_bit_cast(uint4, third_record);
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
                    iq2r_wait_vmcnt<2 + kActivationLoadsPerThread +
                                     (StagedScales ? 1 : kMAtoms)>();
                else
                    iq2r_wait_vmcnt<0>();

                const int pair_column = load_column * 2;
                const int pair = pair_column >> 1;
                const int bit = pair_column & 1;
#pragma unroll
                for(int load_atom = 0;
                    load_atom < kActivationLoadsPerThread;
                    ++load_atom)
                {
                    const int row =
                        load_row + load_atom * kActivationLoadRows;
                    const int swizzled_column =
                        ((pair ^ (row & (kActivationColumns / 2 - 1))) << 1) +
                        bit;
                    const auto staged = __builtin_bit_cast(
                        uint4, activation_ring[slot].data[load_atom]);
                    *reinterpret_cast<uint4*>(
                        &shared.payload.input.activation_cache[slot]
                                                              [row *
                                                                   kActivationColumns +
                                                               swizzled_column]) =
                        staged;
                }
                if constexpr(StagedScales)
                {
                    if(linear_thread < kScaleElements)
                        shared.payload.input.scale_cache
                            [slot * kScaleElements + linear_thread] =
                                activation_ring[slot].staged_scale;
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
                        &shared.payload.input.activation_cache[slot]
                                                              [row *
                                                                   kActivationColumns +
                                                               swizzled0]);
                    halves[1] = *reinterpret_cast<const uint4*>(
                        &shared.payload.input.activation_cache[slot]
                                                              [row *
                                                                   kActivationColumns +
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
                    uint32_t exponent;
                    if constexpr(StagedScales)
                        exponent = shared.payload.input.scale_cache[
                            slot * kScaleElements +
                            (m_atom * kTileN + lane_row) *
                                kScaleGroupsPerTile +
                            lane_group];
                    else
                        exponent = activation_ring[slot].scales[m_atom];
                    const uint32_t scale_a = exponent * 0x01010101u;
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

            constexpr int kEpilogueElementsPerThread = 8;
            constexpr int kEpilogueFragmentColumns =
                kOutputColumns / kEpilogueElementsPerThread;
            constexpr int kEpilogueRowsPerIteration =
                kThreads / kEpilogueFragmentColumns;
            constexpr int kEpilogueWriteIterations =
                (kTileN + kEpilogueRowsPerIteration - 1) /
                kEpilogueRowsPerIteration;
            constexpr int kEpilogueStageElements = kTileN * kOutputColumns;
            auto* epilogue_cache = &shared.payload.epilogue_cache[0][0];

            auto epilogue_offset = [](int row, int column) {
                constexpr int kSwizzleMask = 7;
                const int fragment_column =
                    column / kEpilogueElementsPerThread;
                return row * kOutputColumns +
                       (fragment_column ^ (row & kSwizzleMask)) *
                           kEpilogueElementsPerThread +
                       column % kEpilogueElementsPerThread;
            };

            float bias_values[kNAtomsPerWave] = {};
#pragma unroll
            for(int atom = 0; atom < kNAtomsPerWave; ++atom)
            {
                const int output_column =
                    (n_block_base + atom) * kTileN + lane_row;
                if constexpr(UseBias)
                {
                    if(output_column < N)
                        bias_values[atom] = __bfloat162float(all_bias[
                            static_cast<int64_t>(expert_index) * N +
                            output_column]);
                }
            }

            auto push_epilogue = [&](int m_atom, int stage) {
                auto* stage_base =
                    epilogue_cache + stage * kEpilogueStageElements;
#pragma unroll
                for(int atom = 0; atom < kNAtomsPerWave; ++atom)
                {
                    const int local_column =
                        wave * kNAtomsPerWave * kTileN + atom * kTileN +
                        lane_row;
#pragma unroll
                    for(int item = 0; item < 4; ++item)
                    {
                        const int local_row = lane_group * 4 + item;
                        const float value = accumulators[m_atom][atom][item] +
                                            bias_values[atom];
                        stage_base[epilogue_offset(local_row, local_column)] =
                            __float2bfloat16(value);
                    }
                }
            };

            auto pop_epilogue = [&](int m_atom, int stage) {
                auto* stage_base =
                    epilogue_cache + stage * kEpilogueStageElements;
                const int epilogue_row =
                    linear_thread / kEpilogueFragmentColumns;
                const int local_column =
                    (linear_thread % kEpilogueFragmentColumns) *
                    kEpilogueElementsPerThread;
#pragma unroll
                for(int iteration = 0; iteration < kEpilogueWriteIterations;
                    ++iteration)
                {
                    const int local_row =
                        epilogue_row + iteration * kEpilogueRowsPerIteration;
                    const int output_row =
                        row_base + m_atom * kTileN + local_row;
                    const int output_column =
                        n_tile_index * kOutputColumns + local_column;
                    if(local_row < kTileN && output_row < sub_end &&
                       output_column + kEpilogueElementsPerThread <= N)
                    {
                        const uint4 packed = *reinterpret_cast<const uint4*>(
                            stage_base +
                            epilogue_offset(local_row, local_column));
                        *reinterpret_cast<uint4*>(
                            output + static_cast<int64_t>(output_row) * N +
                            output_column) = packed;
                    }
                }
            };

            // The activation LDS is dead after the final K tile, so reuse it
            // as a two-stage output transpose. This turns per-lane scalar
            // writes into coalesced 128-bit stores for the full output tile.
            __syncthreads();
            push_epilogue(0, 0);
#pragma unroll
            for(int m_atom = 0; m_atom < kMAtoms - 1; ++m_atom)
            {
                __syncthreads();
                pop_epilogue(m_atom, m_atom & 1);
                push_epilogue(m_atom + 1, (m_atom + 1) & 1);
            }
            __syncthreads();
            pop_epilogue(kMAtoms - 1, (kMAtoms - 1) & 1);
            __syncthreads();
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
         int LoadAux = 2>
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
    static_assert(!ActivationLookahead || OutputAtoms == 6);
    static_assert(!(StagedScales && TiledActivationScales));
    constexpr int kOutputColumns = OutputAtoms * kTileN;
    constexpr int kDirectTriplets = OutputAtoms / kAtomsPerTriplet;
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
                physical_tile(n_block_base, k_begin, k_tiles)));

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
                    iq2r_cooperative_triplet_mfma<0>(current_words,
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
                    iq2r_cooperative_triplet_mfma<3>(current_words,
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
#pragma unroll
                for(int triplet = 0; triplet < kDirectTriplets; ++triplet)
                    iq2r_issue_direct_lds_triplet<LoadAux>(data_buffer,
                                                   shared.weight_cache[wave],
                                                   lane,
                                                   triplet,
                                                   next_data_base +
                                                   triplet * kTripletBytes);
                next_data_base += kGroupBytes;
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
                            iq2r_issue_direct_lds_triplet<LoadAux>(
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
                        iq2r_cooperative_triplet_mfma<3>(
                            activation_fragment.words,
                            weight_fragments,
                            accumulators,
                            scale_a,
                            scale_b);
                    }
                    else if(iteration + 1 < iterations)
                        next_data_base += kGroupBytes;
                }
            }

            if constexpr(StagedScales)
                __syncthreads();

            const int output_row = row_base + (lane / 16) * 4 + wave;
#pragma unroll
            for(int atom = 0; atom < 3; ++atom)
                shared.reusable.partial[wave][atom][lane] = accumulators[atom];
            __syncthreads();
#pragma unroll
            for(int atom = 0; atom < 3; ++atom)
            {
                const int output_column =
                    n_tile_index * kOutputColumns + atom * 16 + lane % 16;
                if(wave < 4 && output_row < sub_end && output_column < N)
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
                        for(int source_wave = 0; source_wave < PhysicalWaves;
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

            if constexpr(OutputAtoms == 6)
            {
#pragma unroll
                for(int atom = 0; atom < 3; ++atom)
                    shared.reusable.partial[wave][atom][lane] =
                        accumulators[atom + 3];
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
    if(gpt_oss_shape)
    {
        const int routed_m = static_cast<int>(activations.size(0));
        const int expert_count = static_cast<int>(data.size(0));
        const int active_experts = min(expert_count, routed_m);
        const int cu_count = static_cast<int>(get_num_cu_func());
        const int base_grid = 2 * cu_count;
        const bool use_wide_eight_wave_down =
            logical_n == 2880 && routed_m > 4 && routed_m <= 8;
        // GPT-OSS top-k=4 maps token counts 4..8 and 16 to 16..32 and 64
        // routed rows. Captured M=5/M=8 routes plus synthetic M=6/M=7 route
        // distributions consistently favor the narrower 3x4 down-projection
        // family in this decode band while retaining the existing 5xCU grid.
        const bool use_narrow_four_wave_down =
            logical_n == 2880 &&
            (routed_m == 16 || (routed_m >= 20 && routed_m <= 32) ||
             routed_m == 64);
        // Real GPT-OSS c7/c8 route captures show a small decode crossover that
        // is hidden by the generic routed-M heuristic.  At 20 routed rows the
        // gate/up projection benefits from the narrower 3-output-atom family;
        // at 24..32 rows both projections (except the 32-row down projection)
        // benefit from the lookahead 6-output-atom family.  These choices are
        // confined to the two MoE projections and preserve the same IQ2R math.
        const bool use_narrow_four_wave_gate =
            logical_n == 5760 && routed_m == 20;
        const bool use_lookahead_four_wave_gate =
            logical_n == 5760 && routed_m >= 24 && routed_m <= 32;
        const bool use_lookahead_four_wave_down =
            logical_n == 2880 && routed_m >= 24 && routed_m <= 28;
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
            AITER_CHECK(multiplier >= 1 && multiplier <= 16,
                        "IQ2R grid multiplier override must be in [1,16]");
            launch_grid = multiplier * cu_count;
        }
        const bool use_prefetch32a_auto =
            (family_override == nullptr || family_override[0] == '\0') &&
            !tiled_activation_scales &&
            logical_k == 2880 && routed_m == 4096 &&
            (logical_n == 5760 || logical_n == 2880);
        const bool use_prefetch64_auto =
            (family_override == nullptr || family_override[0] == '\0') &&
            !tiled_activation_scales &&
            logical_k == 2880 &&
            (routed_m == 8192 ||
             (routed_m >= 16384 && routed_m <= 65536)) &&
            (logical_n == 5760 || logical_n == 2880);
        const bool use_prefetch64_serving_auto =
            use_prefetch64_auto && routed_m > 16384;
        const bool use_prefetch64_staged_auto =
            use_prefetch64_auto &&
            (logical_n == 5760 || use_prefetch64_serving_auto);
        const bool use_prefetch64_wide_auto =
            use_prefetch64_staged_auto && logical_n == 5760 &&
            routed_m >= 16384;
        if(grid_override == nullptr || grid_override[0] == '\0')
        {
            if(use_narrow_four_wave_gate)
                launch_grid = 5 * cu_count;
            else if(use_lookahead_four_wave_gate)
                launch_grid = 4 * cu_count;
            else if(logical_n == 2880 && routed_m >= 20 && routed_m <= 28)
                launch_grid = 6 * cu_count;
            else if(use_prefetch32a_auto)
                launch_grid = (logical_n == 5760 ? 11 : 10) * cu_count;
            else if(use_prefetch64_auto && routed_m == 8192)
                launch_grid = (logical_n == 5760 ? 12 : 10) * cu_count;
            else if(use_prefetch64_serving_auto)
                // Captured 16K-cap serving routes span token M=6556..16383
                // (routed M=26224..65532).  Sixteen CU waves is within 1%
                // of the per-shape optimum for both GPT-OSS projections.
                launch_grid = 16 * cu_count;
            else if(use_prefetch64_auto)
                launch_grid = 8 * cu_count;
        }

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
        if(use_prefetch32a_auto || use_prefetch64_auto ||
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
            std::strcmp(family_override, "prefetch64s") == 0 ||
            std::strcmp(family_override, "prefetch64w8s") == 0 ||
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
            if(use_prefetch32a_auto || use_prefetch64_auto ||
               std::strcmp(family_override, "prefetch32") == 0 ||
               std::strcmp(family_override, "prefetch32a") == 0 ||
               std::strcmp(family_override, "prefetch32t") == 0 ||
               std::strcmp(family_override, "prefetch64") == 0 ||
               std::strcmp(family_override, "prefetch64a") == 0 ||
               std::strcmp(family_override, "prefetch64s") == 0 ||
               std::strcmp(family_override, "prefetch64w8s") == 0)
            {
                AITER_CHECK(logical_k == 2880,
                            "IQ2R prefetch family requires K=2880");
                AITER_CHECK(!tiled_activation_scales,
                            "IQ2R prefetch family requires row-major activation scales");
                const bool wide_prefetch =
                    use_prefetch64_wide_auto ||
                    (family_override != nullptr &&
                     std::strcmp(family_override, "prefetch64w8s") == 0);
                const bool staged_prefetch =
                    use_prefetch64_staged_auto ||
                    (family_override != nullptr &&
                     (std::strcmp(family_override, "prefetch64s") == 0 ||
                      std::strcmp(family_override, "prefetch64w8s") == 0));
                AITER_CHECK(!wide_prefetch || logical_n % 384 == 0,
                            "IQ2R prefetch64w8s family requires N divisible by 384");
                const bool task_persistent =
                    !use_prefetch32a_auto && !use_prefetch64_auto &&
                    std::strcmp(family_override, "prefetch32t") == 0;
                const int task_splits =
                    grid_override != nullptr && grid_override[0] != '\0'
                        ? std::atoi(grid_override)
                        : 4;
                const int prefetch_grid = task_persistent
                                              ? static_cast<int>(tasks.size(0)) *
                                                    task_splits
                                              : launch_grid;
#define IQ2R_LAUNCH_PREFETCH(M_ATOMS,                                         \
                             PHYSICAL_WAVES,                                  \
                             STAGED_SCALES,                                   \
                             TASK_PERSISTENT,                                 \
                             PIN_TO_AGPR,                                     \
                             USE_BIAS)                                        \
    hipLaunchKernelGGL(                                                       \
        (iq2r_task_gemm_prefetch_x192_kernel<M_ATOMS,                         \
                                              PHYSICAL_WAVES,                 \
                                              STAGED_SCALES,                  \
                                              TASK_PERSISTENT,                \
                                              PIN_TO_AGPR,                    \
                                              USE_BIAS>),                     \
        dim3(static_cast<uint32_t>(prefetch_grid)),                           \
        dim3(64, PHYSICAL_WAVES),                                             \
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
                if(bias_pointer != nullptr)
                {
                    if(use_prefetch32a_auto)
                        IQ2R_LAUNCH_PREFETCH(2, 4, false, false, true, true);
                    else if(use_prefetch64_auto)
                    {
                        if(wide_prefetch)
                            IQ2R_LAUNCH_PREFETCH(4, 8, true, false, false, true);
                        else if(staged_prefetch)
                            IQ2R_LAUNCH_PREFETCH(4, 4, true, false, false, true);
                        else
                            IQ2R_LAUNCH_PREFETCH(4, 4, false, false, false, true);
                    }
                    else if(std::strcmp(family_override, "prefetch32") == 0)
                        IQ2R_LAUNCH_PREFETCH(2, 4, false, false, false, true);
                    else if(std::strcmp(family_override, "prefetch32a") == 0)
                        IQ2R_LAUNCH_PREFETCH(2, 4, false, false, true, true);
                    else if(task_persistent)
                        IQ2R_LAUNCH_PREFETCH(2, 4, false, true, false, true);
                    else if(std::strcmp(family_override, "prefetch64") == 0)
                        IQ2R_LAUNCH_PREFETCH(4, 4, false, false, false, true);
                    else if(wide_prefetch)
                        IQ2R_LAUNCH_PREFETCH(4, 8, true, false, false, true);
                    else if(staged_prefetch)
                        IQ2R_LAUNCH_PREFETCH(4, 4, true, false, false, true);
                    else
                        IQ2R_LAUNCH_PREFETCH(4, 4, false, false, true, true);
                }
                else
                {
                    if(use_prefetch32a_auto)
                        IQ2R_LAUNCH_PREFETCH(2, 4, false, false, true, false);
                    else if(use_prefetch64_auto)
                    {
                        if(wide_prefetch)
                            IQ2R_LAUNCH_PREFETCH(4, 8, true, false, false, false);
                        else if(staged_prefetch)
                            IQ2R_LAUNCH_PREFETCH(4, 4, true, false, false, false);
                        else
                            IQ2R_LAUNCH_PREFETCH(4, 4, false, false, false, false);
                    }
                    else if(std::strcmp(family_override, "prefetch32") == 0)
                        IQ2R_LAUNCH_PREFETCH(2, 4, false, false, false, false);
                    else if(std::strcmp(family_override, "prefetch32a") == 0)
                        IQ2R_LAUNCH_PREFETCH(2, 4, false, false, true, false);
                    else if(task_persistent)
                        IQ2R_LAUNCH_PREFETCH(2, 4, false, true, false, false);
                    else if(std::strcmp(family_override, "prefetch64") == 0)
                        IQ2R_LAUNCH_PREFETCH(4, 4, false, false, false, false);
                    else if(wide_prefetch)
                        IQ2R_LAUNCH_PREFETCH(4, 8, true, false, false, false);
                    else if(staged_prefetch)
                        IQ2R_LAUNCH_PREFETCH(4, 4, true, false, false, false);
                    else
                        IQ2R_LAUNCH_PREFETCH(4, 4, false, false, true, false);
                }
#undef IQ2R_LAUNCH_PREFETCH
            }
            else if(std::strcmp(family_override, "large32") == 0)
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
                                            LOAD_AUX>),                        \
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
            else if(std::strcmp(family_override, "6x4") == 0)
                IQ2R_LAUNCH_COOPERATIVE(6, 4, false, false, false, true, 2);
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
                            "3x4, 3x4scalar, 3x4t, 3x4s, 3x8, 3x8l0, 3x8t, 6x4, "
                            "6x4scalar, 6x4t, 6x4a, 6x4at, 6x4s, 6x4as, "
                            "6x8, 6x8t, 6x8a, data64, "
                            "data128, or "
                            "large32, large32n, large64, large64n, "
                            "prefetch32, prefetch32a, prefetch32t, "
                            "prefetch64, prefetch64a, prefetch64s, "
                            "prefetch64w8s, or "
                            "large192");
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
        else if(use_narrow_four_wave_gate)
        {
            if(tiled_activation_scales)
                IQ2R_LAUNCH_COOPERATIVE(3, 4, false, false, true, true, 2);
            else
                IQ2R_LAUNCH_COOPERATIVE(3, 4, false, false, false, true, 2);
        }
        else if(use_lookahead_four_wave_gate || use_lookahead_four_wave_down)
        {
            if(tiled_activation_scales)
                IQ2R_LAUNCH_COOPERATIVE(6, 4, true, false, true, true, 2);
            else
                IQ2R_LAUNCH_COOPERATIVE(6, 4, true, false, false, true, 2);
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

} // namespace aiter
