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
#include <initializer_list>

namespace aiter {
namespace {

constexpr int kCodebookBytes       = 4096;
constexpr int kTileN               = 16;
constexpr int kTileK               = 128;
constexpr int kScaleBlock          = 32;
constexpr int kLaneRecordBytes     = 8;
constexpr int kAtomsPerTriplet     = 3;
constexpr int kNBlocksPerGroup     = 6;
constexpr int kAtomTwoRecordBytes  = 12;
constexpr int kAtomTwoOffset       = 1024;
constexpr int kAtomTwoMetadata     = 8;
constexpr int kTripletBytes        = 1792;
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

template<bool PinToAgpr>
__device__ __forceinline__ void
iq2r_pin_accumulator(opus::vector_t<float, 4>& accumulator)
{
    if constexpr(PinToAgpr)
        asm volatile("" : "+a"(accumulator));
}

template<int AccumulatorBase, bool PinToAgpr = false>
__device__ __forceinline__ void iq2r_triplet_mfma(
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

// GLM-5.3 IQ2R packed MoE kernels.

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

using iq2r_scheduled_i32x3 = int __attribute__((ext_vector_type(3)));

__device__ iq2r_scheduled_i32x3 iq2r_scheduled_load_dwordx3(opus::i32x4_t, int, int, int)
    __asm("llvm.amdgcn.raw.buffer.load.v3i32");

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

// Packed-stack gate records: 9-bit packed indices, sign bit planes and a
// sign-free codebook (see aiter/iq2r_packed_stack.py).
template<int MAtoms,int Batch>
__device__ __forceinline__ void glm53_quad_decode_mfma(
    const IQ2RCompressedQuad& compressed,const uint64_t* codebook,uint32_t bases,
    const IQ2RActivationFragment* activation,const uint32_t* scale_a,
    opus::vector_t<float,4> (&accumulators)[MAtoms][4],int active_m);

template<bool Packed>
__global__ __launch_bounds__(512,1) void glm53_gate_m1_kernel(
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
            if constexpr(Packed) glm53_quad_decode_mfma<MAtoms,4>(compressed,shared.codebook,bases,a,sa,accumulators,MAtoms);
            else iq2r_quad_decode_mfma<MAtoms>(compressed,shared.codebook,bases,a,sa,accumulators);
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

template<bool Packed>
__global__ __launch_bounds__(512,1) void glm53_gate_m1_kernel_tp4(
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
            if constexpr(Packed) glm53_quad_decode_mfma<MAtoms,4>(compressed,shared.codebook,bases,a,sa,accumulators,MAtoms);
            else iq2r_quad_decode_mfma<MAtoms>(compressed,shared.codebook,bases,a,sa,accumulators);
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

__device__ __forceinline__ int glm53_remap8(int index,int total)
{
 const int per=(total+7)/8,tall=total%8==0?8:total%8;
 const int xcd=index%8,local=index/8;
 return xcd<tall?xcd*per+local:tall*per+(xcd-tall)*(per-1)+local;
}

template<int MAtoms,int Batch>
__device__ __forceinline__ void glm53_quad_decode_mfma(
    const IQ2RCompressedQuad& compressed,const uint64_t* codebook,uint32_t bases,
    const IQ2RActivationFragment* activation,const uint32_t* scale_a,
    opus::vector_t<float,4> (&accumulators)[MAtoms][4],int active_m);

// Dense gate: waves cover N, while a shared K128 activation tile is reused
// across the entire CTA. Preserve the eight original K768 partial sums.
template<int Waves,bool WeightPrefetch,int XCD,int TaskGroup,int Unroll,int Decode=0,int NT=8>
__global__ __launch_bounds__(64*Waves,1) void glm53_gate_prefill_kernel(
    const opus::fp8_t* __restrict__ activations,const uint8_t* __restrict__ scales,
    const uint8_t* __restrict__ all_data,const uint8_t* __restrict__ all_auxiliary,
    const int32_t* __restrict__ tasks,const int32_t* __restrict__ task_count,
    const int32_t* __restrict__ gather,opus::fp8_t* __restrict__ output,
    uint8_t* __restrict__ output_scales,int routes,int data_bytes,int auxiliary_bytes)
{
#if defined(__gfx950__)
    constexpr int Rows=64,MAtoms=4,Threads=Waves*64;
    struct Storage {
        alignas(16) uint64_t codebook[512];
        int row_ids[Rows];
        uint8_t scales[2*Rows*4];
        union {
            alignas(16) uint8_t input[2*Rows*128];
            alignas(16) __hip_bfloat16 gate[16][Waves*64];
        } reuse;
    };
    __shared__ Storage shared;
    const int lane=threadIdx.x,wave=threadIdx.y,linear=wave*64+lane;
    const int lane_row=lane%16,lane_group=lane/16;
    const int nt=task_count[0];
    int previous_expert=-1;
    for(int work=blockIdx.x;work<nt*(NT/Waves);work+=gridDim.x)
    {
        int index=work;
        if constexpr(XCD>0)
        {
            const int total=nt*(NT/Waves);
            const int per=(total+XCD-1)/XCD,tall=total%XCD==0?XCD:total%XCD;
            const int die=work%XCD,local=work/XCD;
            index=die<tall?die*per+local:tall*per+(die-tall)*(per-1)+local;
        }
        int task=index%nt,output_tile=index/nt;
        if constexpr(TaskGroup>0)
        {
            const int first=(index/(TaskGroup*(NT/Waves)))*TaskGroup;
            const int valid=min(TaskGroup,nt-first),local=index%(TaskGroup*(NT/Waves));
            task=first+local%valid;output_tile=local/valid;
        }
        const int n_tile=output_tile*Waves+wave;
        const int begin=tasks[task*3],count=tasks[task*3+1],expert=tasks[task*3+2];
        if(begin<0 || count<=0 || count>64 || begin+count>routes || expert<0 || expert>=257)continue;
        const int active_m=(count+15)/16;
        const uint8_t* data=all_data+static_cast<int64_t>(expert)*data_bytes;
        const uint8_t* aux=all_auxiliary+static_cast<int64_t>(expert)*auxiliary_bytes;
        if(expert!=previous_expert)
        {
            for(int i=linear;i<512;i+=Threads)shared.codebook[i]=reinterpret_cast<const uint64_t*>(aux)[i];
            previous_expert=expert;
        }
        if(linear<Rows)shared.row_ids[linear]=gather[begin+min(linear,count-1)]/9;
        __syncthreads();
        uint32_t bases=0;
#pragma unroll
        for(int atom=0;atom<4;++atom)bases|=static_cast<uint32_t>(aux[kCodebookBytes+n_tile*4+atom])<<(atom*8);
        opus::gmem<uint8_t> a_buffer(reinterpret_cast<const uint8_t*>(activations),(routes/9)*6144);
        opus::gmem<uint8_t> s_buffer(scales,(routes/9)*192);
        constexpr int Copies=Rows*128/(Threads*16);
        int source_a[Copies];
#pragma unroll
        for(int copy=0;copy<Copies;++copy)
        {
            const int offset=(linear+copy*Threads)*16;
            const int r=offset/128,col=(offset%128)^(((r>>1)&7)<<4);
            source_a[copy]=shared.row_ids[r]*6144+col;
        }
        const int source_s=shared.row_ids[lane]*192;
        auto issue_a=[&](int kt,int slot)
        {
#pragma unroll
            for(int copy=0;copy<Copies;++copy)
            {
                // The LDS base is uniform within a wave. Hardware adds
                // lane*16; apply the row swizzle in the global source column.
                const int destination=slot*Rows*128+(wave*64+copy*Threads)*16;
                a_buffer.template async_load<16>(shared.reuse.input+destination,source_a[copy]+kt*128);
            }
            if(wave==0)
                s_buffer.template async_load<4>(shared.scales+slot*Rows*4,source_s+kt*4);
            asm volatile("" ::: "memory");
        };
        IQ2RCompressedQuad pending;
        if constexpr(WeightPrefetch)pending=iq2r_scheduled_load_quad(data,data_bytes,n_tile*48*kQuadBytes,lane);
        issue_a(0,0);
        opus::vector_t<float,4> total[MAtoms][4]={};
#pragma unroll 1
        for(int partition=0;partition<8;++partition)
        {
            opus::vector_t<float,4> partial[MAtoms][4]={};
#pragma unroll Unroll
            for(int iter=0;iter<6;++iter)
            {
                const int kt=partition*6+iter;
                const int slot=kt&1;
                IQ2RCompressedQuad compressed;
                if constexpr(WeightPrefetch)compressed=pending;
                else compressed=iq2r_scheduled_load_quad(data,data_bytes,(n_tile*48+kt)*kQuadBytes,lane);
                // The previous iteration issued the current A/scales. All
                // waves must finish consuming the older slot before reuse.
                asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
                __syncthreads();
                IQ2RActivationFragment a[MAtoms];uint32_t sa[MAtoms];
#pragma unroll
                for(int m=0;m<MAtoms;++m)
                {
                    const int r=m*16+lane_row,col=lane_group*16;
                    *reinterpret_cast<uint4*>(a[m].bytes)=*reinterpret_cast<const uint4*>(shared.reuse.input+slot*Rows*128+((r*128+col)^(((r>>1)&7)<<4)));
                    *reinterpret_cast<uint4*>(a[m].bytes+16)=*reinterpret_cast<const uint4*>(shared.reuse.input+slot*Rows*128+((r*128+col+64)^(((r>>1)&7)<<4)));
                    sa[m]=shared.scales[slot*Rows*4+r*4+lane_group];
                }
                if(kt<47)
                {
                    issue_a(kt+1,1-slot);
                    if constexpr(WeightPrefetch)
                        pending=iq2r_scheduled_load_quad(data,data_bytes,(n_tile*48+kt+1)*kQuadBytes,lane);
                }
                // Decode>0 reads packed-stack weights: packed indices, sign
                // planes and a sign-free codebook (Decode = lookup batch).
                if constexpr(Decode==0)iq2r_quad_sparse_decode_mfma<MAtoms>(compressed,shared.codebook,bases,a,sa,partial,MAtoms);
                else glm53_quad_decode_mfma<MAtoms,Decode>(compressed,shared.codebook,bases,a,sa,partial,active_m);
            }
#pragma unroll
            for(int m=0;m<MAtoms;++m)
#pragma unroll
                for(int atom=0;atom<4;++atom)
                    total[m][atom]+=partial[m][atom];
        }
        __syncthreads();
#pragma unroll
        for(int m=0;m<MAtoms;++m)
        {
            if(m>=active_m)continue;
#pragma unroll
            for(int atom=0;atom<4;++atom)
#pragma unroll
                for(int component=0;component<4;++component)
                    shared.reuse.gate[lane_group*4+component][wave*64+atom*16+lane_row]=__float2bfloat16(total[m][atom][component]);
            __syncthreads();
            const int local_row=linear/(Waves*4),quad=(linear/4)%Waves,qlane=linear%4;
            const int row=begin+m*16+local_row;
            float values[8];float abs_max=1.0e-10f;
#pragma unroll
            for(int element=0;element<8;++element)
            {
                const int col=quad*64+(qlane*8+element)*2;
                const float gate=__bfloat162float(shared.reuse.gate[local_row][col]);
                const float up=__bfloat162float(shared.reuse.gate[local_row][col+1]);
                values[element]=__bfloat162float(__float2bfloat16((gate/(1.0f+__expf(-gate)))*(up+0.0f)));
                abs_max=fmaxf(abs_max,fabsf(values[element]));
            }
            abs_max=fmaxf(abs_max,__shfl_xor(abs_max,1));
            abs_max=fmaxf(abs_max,__shfl_xor(abs_max,2));
            if(row<begin+count)
            {
                const auto bs=fp_f32_to_e8m0_block_scale<kDefaultMxScaleRoundMode,MxDtype::FP8_E4M3>(abs_max);
                const float inverse=1.0f/bs.dq_scale;
                opus::vector_t<opus::fp8_t,8> q;
#pragma unroll
                for(int element=0;element<8;++element)q[element]=opus::fp32_to_fp8(values[element]*inverse);
                const int out_quad=output_tile*Waves+quad;
                *reinterpret_cast<opus::vector_t<opus::fp8_t,8>*>(output+static_cast<int64_t>(row)*(NT*32)+out_quad*32+qlane*8)=q;
                if(qlane==0)output_scales[static_cast<int64_t>(row)*NT+out_quad]=bs.byte;
            }
            __syncthreads();
        }
    }
#endif
}

// Experimental entry requires validated nonnegative codebook bytes.
// OR can combine the sign mask operation into a single v_and_or_b32.
__device__ __forceinline__ uint64_t glm53_apply_signs(uint64_t magnitude,uint32_t signs)
{
    constexpr uint32_t spread=0x10204080u,mask=0x80808080u;
    const uint32_t low=static_cast<uint32_t>(magnitude)|(((signs&15u)*spread)&mask);
    const uint32_t high=static_cast<uint32_t>(magnitude>>32)|((((signs>>4)&15u)*spread)&mask);
    return static_cast<uint64_t>(high)<<32|low;
}

template<int Atom,bool SignOr>
__device__ __forceinline__ void glm53_decode_packed_atom(
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
            codeword<3?((index_lows>>(codeword*9))&511u):((index_lows>>27)|(index_highs<<5));
        decoded.codewords[codeword] =
            (SignOr?glm53_apply_signs(codebook[codebook_index],(signs>>(codeword*8))&255u):apply_signs(codebook[codebook_index],(signs>>(codeword*8))&255u));
    }
    weight_fragment = decoded.words;
    const uint32_t base = (packed_bases >> (Atom * 8)) & 0xffu;
    const uint32_t delta = (metadata >> (Atom * 8 + 4)) & 0x0fu;
    scale_b = base + delta;
}

// TP8: waves cover independent N tiles. Cooperatively cache A once per CTA.
// Retain two separate K128 accumulators and their final addition so moving K
// into one wave preserves the existing split-K arithmetic exactly.
template<int Waves,int MAtoms,int Groups,bool Prefetch,int XCD,int TaskGroup,bool SignOr>
__global__ __launch_bounds__(64*Waves,1) void glm53_down_tp8_kernel(
    const opus::fp8_t* __restrict__ activations,
    const uint8_t* __restrict__ activation_scales,
    const uint8_t* __restrict__ all_data,
    const uint8_t* __restrict__ all_auxiliary,
    const int32_t* __restrict__ tasks,
    const int32_t* __restrict__ task_count,
    const __hip_bfloat16* __restrict__ all_bias,
    __hip_bfloat16* __restrict__ output,
    int M,int N,int K,int expert_count,int data_bytes,int auxiliary_bytes)
{
#if defined(__gfx950__)
    constexpr int Rows=16*MAtoms,Columns=48*Waves*Groups,Threads=64*Waves;
    struct Storage {
        alignas(16) uint64_t codebook[512];
        alignas(16) uint8_t input[Rows*256];
        uint8_t scales[Rows*8];
    };
    __shared__ Storage shared;
    const int lane=threadIdx.x,wave=threadIdx.y,linear=wave*64+lane;
    const int lane_row=lane%16,lane_group=lane/16;
    const int nt=task_count[0];
    opus::gmem<uint8_t> a_buffer(activations,static_cast<unsigned int>(M*256));
    // Descriptor bounds are bytes; typed offsets below are BF16 elements.
    opus::gmem<opus::bf16_t> out_buffer(output,static_cast<unsigned int>(M*6144*sizeof(__hip_bfloat16)));
    opus::gmem<uint8_t> scale_buffer(activation_scales,static_cast<unsigned int>(M*8));
    int previous_expert=-1;
    for(int work=blockIdx.x;work<nt*(6144/Columns);work+=gridDim.x)
    {
        int index=work;
        if constexpr(XCD>0)
        {
            const int total=nt*(6144/Columns);
            const int per=(total+XCD-1)/XCD,tall=total%XCD==0?XCD:total%XCD;
            const int die=work%XCD,local=work/XCD;
            index=die<tall?die*per+local:tall*per+(die-tall)*(per-1)+local;
        }
        int task=index%nt,output_tile=index/nt;
        if constexpr(TaskGroup>0)
        {
            const int first=(index/(TaskGroup*(6144/Columns)))*TaskGroup;
            const int valid=min(TaskGroup,nt-first),local=index%(TaskGroup*(6144/Columns));
            task=first+local%valid;output_tile=local/valid;
        }
        const int n_tile=output_tile;
        const int begin=tasks[task*3],count=tasks[task*3+1],expert=tasks[task*3+2];
        if(begin<0 || count<=0 || begin+count>M || expert<0 || expert>=expert_count)continue;
        const uint8_t* data=all_data+static_cast<int64_t>(expert)*data_bytes;
        const uint8_t* aux=all_auxiliary+static_cast<int64_t>(expert)*auxiliary_bytes;
        if(previous_expert!=expert)
        {
            for(int i=linear;i<512;i+=Threads)shared.codebook[i]=reinterpret_cast<const uint64_t*>(aux)[i];
            previous_expert=expert;
        }
        for(int row_base=begin;row_base<begin+count;row_base+=Rows)
        {
            const int row_count=min(Rows,begin+count-row_base);
            const int active_m=(row_count+15)/16;
#pragma unroll
            for(int copy=0;copy<(Rows*256+Threads*16-1)/(Threads*16);++copy)
            {
                const int offset=(linear+copy*Threads)*16;
                const int r=offset/256,col=(offset%256)^((r&15)<<4);
                const int src=(row_base+min(r,row_count-1))*256+col;
                const int dest=(wave*64+copy*Threads)*16;
                if(dest<Rows*256)a_buffer.template async_load<16>(shared.input+dest,src);
            }
            if(linear<Rows*2)
            {
                const int r=linear/2,col=(linear%2)*4;
                scale_buffer.template async_load<4>(shared.scales+wave*64*4,(row_base+min(r,row_count-1))*8+col);
            }
            IQ2RCompressedTriplet pending[2];
            uint32_t pending_bases=0;
            if constexpr(Prefetch)
            {
                const int nb=n_tile*(Columns/16)+wave*3;
#pragma unroll
                for(int atom=0;atom<3;++atom)pending_bases|=static_cast<uint32_t>(aux[kCodebookBytes+nb+atom])<<(atom*8);
#pragma unroll
                for(int kt=0;kt<2;++kt)pending[kt]=iq2r_load_compact_triplet(data,static_cast<int>(triplet_base(physical_tile(nb,kt,2))),lane);
            }
            asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
            __syncthreads();

#pragma unroll 1
            for(int group=0;group<Groups;++group)
            {
        const int n_block=n_tile*(Columns/16)+wave*3+group*Waves*3;
        uint32_t bases=0;
        IQ2RCompressedTriplet compressed[2];
        if constexpr(Prefetch)
        {
            bases=pending_bases;
            compressed[0]=pending[0];compressed[1]=pending[1];
            if(group+1<Groups)
            {
                const int nb=n_block+Waves*3;
                pending_bases=0;
#pragma unroll
                for(int atom=0;atom<3;++atom)pending_bases|=static_cast<uint32_t>(aux[kCodebookBytes+nb+atom])<<(atom*8);
#pragma unroll
                for(int kt=0;kt<2;++kt)pending[kt]=iq2r_load_compact_triplet(data,static_cast<int>(triplet_base(physical_tile(nb,kt,2))),lane);
                asm volatile("" ::: "memory");
            }
        }
        else
        {
#pragma unroll
            for(int atom=0;atom<3;++atom)bases|=static_cast<uint32_t>(aux[kCodebookBytes+n_block+atom])<<(atom*8);
#pragma unroll
            for(int kt=0;kt<2;++kt)compressed[kt]=iq2r_load_compact_triplet(data,static_cast<int>(triplet_base(physical_tile(n_block,kt,2))),lane);
        }
            // Finish one N16 atom at a time. Retain the exact two K128
            // partial sums, while shortening the lifetime of decoded weights
            // and accumulators enough to support a full M64 tile.
            {
                opus::vector_t<float,4> first[MAtoms]={};
                opus::i32x8_t w0;uint32_t sb0;
                glm53_decode_packed_atom<0,SignOr>(compressed[0],shared.codebook,bases,w0,sb0);
#pragma unroll
                for(int m=0;m<MAtoms;++m)
                {
                    if(m>=active_m)continue;
                    const int r=m*16+lane_row,col=lane_group*16;
                    IQ2RActivationFragment a;
                    *reinterpret_cast<uint4*>(a.bytes)=*reinterpret_cast<const uint4*>(shared.input+((r*256+col)^((r&15)<<4)));
                    *reinterpret_cast<uint4*>(a.bytes+16)=*reinterpret_cast<const uint4*>(shared.input+((r*256+col+64)^((r&15)<<4)));
                    const uint32_t sa=shared.scales[r*8+lane_group];
                    auto mma=opus::mfma<opus::fp8_t,opus::fp8_t,opus::fp32_t,16,16,128>{};
                    first[m]=mma(w0,a.words,first[m],sb0,sa);
                }
                opus::i32x8_t w1;uint32_t sb1;
                glm53_decode_packed_atom<0,SignOr>(compressed[1],shared.codebook,bases,w1,sb1);
#pragma unroll
                for(int m=0;m<MAtoms;++m)
                {
                    if(m>=active_m)continue;
                    const int r=m*16+lane_row,col=128+lane_group*16;
                    IQ2RActivationFragment a;
                    *reinterpret_cast<uint4*>(a.bytes)=*reinterpret_cast<const uint4*>(shared.input+((r*256+col)^((r&15)<<4)));
                    *reinterpret_cast<uint4*>(a.bytes+16)=*reinterpret_cast<const uint4*>(shared.input+((r*256+col+64)^((r&15)<<4)));
                    const uint32_t sa=shared.scales[r*8+4+lane_group];
                    opus::vector_t<float,4> second={};
                    auto mma=opus::mfma<opus::fp8_t,opus::fp8_t,opus::fp32_t,16,16,128>{};
                    second=mma(w1,a.words,second,sb1,sa);
                    const int out_r=m*16+lane_row;
                    const int out_col=n_block*16+0*16+lane_group*4;
                    opus::vector_t<opus::bf16_t,4> packed;
#pragma unroll
                    for(int c=0;c<4;++c)
                    {
                        float value=first[m][c]+second[c];
                        if(all_bias)value+=__bfloat162float(all_bias[static_cast<int64_t>(expert)*6144+out_col+c]);
                        packed[c]=__builtin_bit_cast(opus::bf16_t,__float2bfloat16(value));
                    }
                    if(out_r<row_count)out_buffer.template store<4>(packed,(row_base+out_r)*6144+out_col);
                }
            }
            {
                opus::vector_t<float,4> first[MAtoms]={};
                opus::i32x8_t w0;uint32_t sb0;
                glm53_decode_packed_atom<1,SignOr>(compressed[0],shared.codebook,bases,w0,sb0);
#pragma unroll
                for(int m=0;m<MAtoms;++m)
                {
                    if(m>=active_m)continue;
                    const int r=m*16+lane_row,col=lane_group*16;
                    IQ2RActivationFragment a;
                    *reinterpret_cast<uint4*>(a.bytes)=*reinterpret_cast<const uint4*>(shared.input+((r*256+col)^((r&15)<<4)));
                    *reinterpret_cast<uint4*>(a.bytes+16)=*reinterpret_cast<const uint4*>(shared.input+((r*256+col+64)^((r&15)<<4)));
                    const uint32_t sa=shared.scales[r*8+lane_group];
                    auto mma=opus::mfma<opus::fp8_t,opus::fp8_t,opus::fp32_t,16,16,128>{};
                    first[m]=mma(w0,a.words,first[m],sb0,sa);
                }
                opus::i32x8_t w1;uint32_t sb1;
                glm53_decode_packed_atom<1,SignOr>(compressed[1],shared.codebook,bases,w1,sb1);
#pragma unroll
                for(int m=0;m<MAtoms;++m)
                {
                    if(m>=active_m)continue;
                    const int r=m*16+lane_row,col=128+lane_group*16;
                    IQ2RActivationFragment a;
                    *reinterpret_cast<uint4*>(a.bytes)=*reinterpret_cast<const uint4*>(shared.input+((r*256+col)^((r&15)<<4)));
                    *reinterpret_cast<uint4*>(a.bytes+16)=*reinterpret_cast<const uint4*>(shared.input+((r*256+col+64)^((r&15)<<4)));
                    const uint32_t sa=shared.scales[r*8+4+lane_group];
                    opus::vector_t<float,4> second={};
                    auto mma=opus::mfma<opus::fp8_t,opus::fp8_t,opus::fp32_t,16,16,128>{};
                    second=mma(w1,a.words,second,sb1,sa);
                    const int out_r=m*16+lane_row;
                    const int out_col=n_block*16+1*16+lane_group*4;
                    opus::vector_t<opus::bf16_t,4> packed;
#pragma unroll
                    for(int c=0;c<4;++c)
                    {
                        float value=first[m][c]+second[c];
                        if(all_bias)value+=__bfloat162float(all_bias[static_cast<int64_t>(expert)*6144+out_col+c]);
                        packed[c]=__builtin_bit_cast(opus::bf16_t,__float2bfloat16(value));
                    }
                    if(out_r<row_count)out_buffer.template store<4>(packed,(row_base+out_r)*6144+out_col);
                }
            }
            {
                opus::vector_t<float,4> first[MAtoms]={};
                opus::i32x8_t w0;uint32_t sb0;
                glm53_decode_packed_atom<2,SignOr>(compressed[0],shared.codebook,bases,w0,sb0);
#pragma unroll
                for(int m=0;m<MAtoms;++m)
                {
                    if(m>=active_m)continue;
                    const int r=m*16+lane_row,col=lane_group*16;
                    IQ2RActivationFragment a;
                    *reinterpret_cast<uint4*>(a.bytes)=*reinterpret_cast<const uint4*>(shared.input+((r*256+col)^((r&15)<<4)));
                    *reinterpret_cast<uint4*>(a.bytes+16)=*reinterpret_cast<const uint4*>(shared.input+((r*256+col+64)^((r&15)<<4)));
                    const uint32_t sa=shared.scales[r*8+lane_group];
                    auto mma=opus::mfma<opus::fp8_t,opus::fp8_t,opus::fp32_t,16,16,128>{};
                    first[m]=mma(w0,a.words,first[m],sb0,sa);
                }
                opus::i32x8_t w1;uint32_t sb1;
                glm53_decode_packed_atom<2,SignOr>(compressed[1],shared.codebook,bases,w1,sb1);
#pragma unroll
                for(int m=0;m<MAtoms;++m)
                {
                    if(m>=active_m)continue;
                    const int r=m*16+lane_row,col=128+lane_group*16;
                    IQ2RActivationFragment a;
                    *reinterpret_cast<uint4*>(a.bytes)=*reinterpret_cast<const uint4*>(shared.input+((r*256+col)^((r&15)<<4)));
                    *reinterpret_cast<uint4*>(a.bytes+16)=*reinterpret_cast<const uint4*>(shared.input+((r*256+col+64)^((r&15)<<4)));
                    const uint32_t sa=shared.scales[r*8+4+lane_group];
                    opus::vector_t<float,4> second={};
                    auto mma=opus::mfma<opus::fp8_t,opus::fp8_t,opus::fp32_t,16,16,128>{};
                    second=mma(w1,a.words,second,sb1,sa);
                    const int out_r=m*16+lane_row;
                    const int out_col=n_block*16+2*16+lane_group*4;
                    opus::vector_t<opus::bf16_t,4> packed;
#pragma unroll
                    for(int c=0;c<4;++c)
                    {
                        float value=first[m][c]+second[c];
                        if(all_bias)value+=__bfloat162float(all_bias[static_cast<int64_t>(expert)*6144+out_col+c]);
                        packed[c]=__builtin_bit_cast(opus::bf16_t,__float2bfloat16(value));
                    }
                    if(out_r<row_count)out_buffer.template store<4>(packed,(row_base+out_r)*6144+out_col);
                }
            }
            }
            __syncthreads();
        }
    }
#endif
}

// TP4 (K=512) packed-index down: the TP8 packed-index kernel with KT K128 tiles per wave.
// Each K tile keeps its own zero-initialized accumulator and the partials add
// in wave order ((k0+k1)+k2)+k3.
template<int Atom,int KT,int MAtoms,bool Guard=true>
__device__ __forceinline__ void glm53_down_tp4_atom(
    const IQ2RCompressedTriplet (&compressed)[KT],const uint64_t* codebook,uint32_t bases,
    const uint8_t* input,const uint8_t* scales,int active_m,int lane_row,int lane_group,
    opus::vector_t<float,4> (&sum)[MAtoms])
{
    constexpr int K=KT*128;
    // Issue every K tile's MFMA before the first addition so no add waits on
    // the MFMA it follows; the ((k0+k1)+k2)+k3 order is unchanged.
    opus::vector_t<float,4> part[KT][MAtoms];
    opus::i32x8_t w[KT];uint32_t sb[KT];
#pragma unroll
    for(int kt=0;kt<KT;++kt)glm53_decode_packed_atom<Atom,true>(compressed[kt],codebook,bases,w[kt],sb[kt]);
#pragma unroll
    for(int kt=0;kt<KT;++kt)
    {
#pragma unroll
        for(int m=0;m<MAtoms;++m)
        {
            if(Guard && m>=active_m)continue;
            const int r=m*16+lane_row,col=kt*128+lane_group*16;
            IQ2RActivationFragment a;
            *reinterpret_cast<uint4*>(a.bytes)=*reinterpret_cast<const uint4*>(input+((r*K+col)^((r&15)<<4)));
            *reinterpret_cast<uint4*>(a.bytes+16)=*reinterpret_cast<const uint4*>(input+((r*K+col+64)^((r&15)<<4)));
            const uint32_t sa=scales[r*(K/32)+kt*4+lane_group];
            part[kt][m]=opus::vector_t<float,4>{};
            auto mma=opus::mfma<opus::fp8_t,opus::fp8_t,opus::fp32_t,16,16,128>{};
            part[kt][m]=mma(w[kt],a.words,part[kt][m],sb[kt],sa);
        }
    }
#pragma unroll
    for(int m=0;m<MAtoms;++m)
    {
        if(Guard && m>=active_m)continue;
        sum[m]=part[0][m];
#pragma unroll
        for(int kt=1;kt<KT;++kt)
#pragma unroll
            for(int c=0;c<4;++c)sum[m][c]+=part[kt][m][c];
    }
}

// Guard=false computes every row atom unconditionally (padded rows read clamped
// valid inputs and are never stored), keeping the MFMA stream branch-free.
template<int Waves,int MAtoms,int Groups,int XCD,int TaskGroup,int KT,int Chunks=1,bool Tiled=false,int Occupancy=1,bool Prefetch=true,bool Guard=true>
__global__ __launch_bounds__(64*Waves,Occupancy) void glm53_down_tp4_kernel(
    const opus::fp8_t* __restrict__ activations,
    const uint8_t* __restrict__ activation_scales,
    const uint8_t* __restrict__ all_data,
    const uint8_t* __restrict__ all_auxiliary,
    const int32_t* __restrict__ tasks,
    const int32_t* __restrict__ task_count,
    const __hip_bfloat16* __restrict__ all_bias,
    __hip_bfloat16* __restrict__ output,
    int M,int N,int K_,int expert_count,int data_bytes,int auxiliary_bytes,int chunk)
{
#if defined(__gfx950__)
    constexpr int K=KT*128,Rows=16*MAtoms,Columns=48*Waves*Groups,Threads=64*Waves;
    struct Storage {
        alignas(16) uint64_t codebook[512];
        alignas(16) uint8_t input[Rows*K];
        uint8_t scales[Rows*(K/32)];
    };
    __shared__ Storage shared;
    const int lane=threadIdx.x,wave=threadIdx.y,linear=wave*64+lane;
    const int lane_row=lane%16,lane_group=lane/16;
    const int nt=task_count[0];
    opus::gmem<uint8_t> a_buffer(activations,static_cast<unsigned int>(M*K));
    opus::gmem<opus::bf16_t> out_buffer(output,static_cast<unsigned int>(M*6144*sizeof(__hip_bfloat16)));
    opus::gmem<uint8_t> scale_buffer(activation_scales,static_cast<unsigned int>(M*(K/32)));
    int previous_expert=-1;
    for(int work=blockIdx.x;work<nt*(6144/Columns/Chunks);work+=gridDim.x)
    {
        int index=work;
        if constexpr(XCD>0)
        {
            const int total=nt*(6144/Columns/Chunks);
            const int per=(total+XCD-1)/XCD,tall=total%XCD==0?XCD:total%XCD;
            const int die=work%XCD,local=work/XCD;
            index=die<tall?die*per+local:tall*per+(die-tall)*(per-1)+local;
        }
        int task=index%nt,output_tile=index/nt;
        if constexpr(TaskGroup>0)
        {
            const int first=(index/(TaskGroup*(6144/Columns/Chunks)))*TaskGroup;
            const int valid=min(TaskGroup,nt-first),local=index%(TaskGroup*(6144/Columns/Chunks));
            task=first+local%valid;output_tile=local/valid;
        }
        const int n_tile=output_tile+chunk*(6144/Columns/Chunks);
        const int begin=tasks[task*3],count=tasks[task*3+1],expert=tasks[task*3+2];
        if(begin<0 || count<=0 || begin+count>M || expert<0 || expert>=expert_count)continue;
        const uint8_t* data=all_data+static_cast<int64_t>(expert)*data_bytes;
        const uint8_t* aux=all_auxiliary+static_cast<int64_t>(expert)*auxiliary_bytes;
        if(previous_expert!=expert)
        {
            for(int i=linear;i<512;i+=Threads)shared.codebook[i]=reinterpret_cast<const uint64_t*>(aux)[i];
            previous_expert=expert;
        }
        for(int row_base=begin;row_base<begin+count;row_base+=Rows)
        {
            const int row_count=min(Rows,begin+count-row_base);
            const int active_m=(row_count+15)/16;
#pragma unroll
            for(int copy=0;copy<(Rows*K+Threads*16-1)/(Threads*16);++copy)
            {
                const int offset=(linear+copy*Threads)*16;
                const int r=offset/K,col=(offset%K)^((r&15)<<4);
                const int src=(row_base+min(r,row_count-1))*K+col;
                const int dest=(wave*64+copy*Threads)*16;
                if(dest<Rows*K)a_buffer.template async_load<16>(shared.input+dest,src);
            }
            if(linear<Rows*(K/128))
            {
                const int r=linear/(K/128),col=(linear%(K/128))*4;
                scale_buffer.template async_load<4>(shared.scales+wave*64*4,(row_base+min(r,row_count-1))*(K/32)+col);
            }
            IQ2RCompressedTriplet pending[Prefetch?KT:1];
            uint32_t pending_bases=0;
            if constexpr(Prefetch)
            {
                const int nb=n_tile*(Columns/16)+wave*3;
#pragma unroll
                for(int atom=0;atom<3;++atom)pending_bases|=static_cast<uint32_t>(aux[kCodebookBytes+nb+atom])<<(atom*8);
#pragma unroll
                for(int kt=0;kt<KT;++kt)pending[kt]=iq2r_load_compact_triplet(data,static_cast<int>(triplet_base(physical_tile(nb,kt,KT))),lane);
            }
            asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
            __syncthreads();
#pragma unroll 1
            for(int group=0;group<Groups;++group)
            {
                const int n_block=n_tile*(Columns/16)+wave*3+group*Waves*3;
                uint32_t bases=pending_bases;
                IQ2RCompressedTriplet compressed[KT];
                if constexpr(!Prefetch)
                {
                    bases=0;
#pragma unroll
                    for(int atom=0;atom<3;++atom)bases|=static_cast<uint32_t>(aux[kCodebookBytes+n_block+atom])<<(atom*8);
#pragma unroll
                    for(int kt=0;kt<KT;++kt)compressed[kt]=iq2r_load_compact_triplet(data,static_cast<int>(triplet_base(physical_tile(n_block,kt,KT))),lane);
                }
                else
                {
#pragma unroll
                for(int kt=0;kt<KT;++kt)compressed[kt]=pending[kt];
                }
                if constexpr(Prefetch) if(group+1<Groups)
                {
                    const int nb=n_block+Waves*3;
                    pending_bases=0;
#pragma unroll
                    for(int atom=0;atom<3;++atom)pending_bases|=static_cast<uint32_t>(aux[kCodebookBytes+nb+atom])<<(atom*8);
#pragma unroll
                    for(int kt=0;kt<KT;++kt)pending[kt]=iq2r_load_compact_triplet(data,static_cast<int>(triplet_base(physical_tile(nb,kt,KT))),lane);
                    asm volatile("" ::: "memory");
                }
#define GLM53_DOWN_TP4_ATOM(ATOM) \
                { \
                    opus::vector_t<float,4> sum[MAtoms]; \
                    glm53_down_tp4_atom<ATOM,KT,MAtoms,Guard>(compressed,shared.codebook,bases,shared.input,shared.scales,active_m,lane_row,lane_group,sum); \
                    _Pragma("unroll") \
                    for(int m=0;m<MAtoms;++m) \
                    { \
                        if(Guard && m>=active_m)continue; \
                        const int out_r=m*16+lane_row; \
                        const int out_col=n_block*16+ATOM*16+lane_group*4; \
                        opus::vector_t<opus::bf16_t,4> packed; \
                        _Pragma("unroll") \
                        for(int c=0;c<4;++c) \
                        { \
                            float value=sum[m][c]; \
                            if(all_bias)value+=__bfloat162float(all_bias[static_cast<int64_t>(expert)*6144+out_col+c]); \
                            packed[c]=__builtin_bit_cast(opus::bf16_t,__float2bfloat16(value)); \
                        } \
                        if(out_r<row_count)out_buffer.template store<4>(packed,(Tiled?((out_col/384)*M+row_base+out_r)*384+out_col%384:(row_base+out_r)*6144+out_col)); \
                    } \
                }
                GLM53_DOWN_TP4_ATOM(0) GLM53_DOWN_TP4_ATOM(1) GLM53_DOWN_TP4_ATOM(2)
#undef GLM53_DOWN_TP4_ATOM
            }
            __syncthreads();
        }
    }
#endif
}

// Packed triplet source offset; with n_block%3==0 it is affine in the K tile,
// equal to triplet_base(physical_tile(n_block,k_tile,k_tiles)).
template<int KT>
__device__ __forceinline__ int glm53_ws_source_base(int n_block,int k_tile)
{
    const uint32_t n=static_cast<uint32_t>(n_block);
    return static_cast<int>(((n/kNBlocksPerGroup)*KT+static_cast<uint32_t>(k_tile))*kGroupBytes+
                            ((n%kNBlocksPerGroup)/kAtomsPerTriplet)*kTripletBytes);
}

// TP4 (K=512) weight-stationary down: one 64-row task tile stays in LDS while
// each wave decodes a K128 weight triplet once and applies it to every row
// atom. K tiles are outermost, so each output still accumulates
// ((k0+k1)+k2)+k3 from zero-initialized per-tile MFMAs, as in glm53_down_tp4_kernel.
template<int Waves,int Groups,int XCD,int TaskGroup,int KT,int Chunks,int Occupancy,int MAtoms=4,bool Merge=false,bool Prefetch=false,bool KLoop=false,int Ablate=0>
__global__ __launch_bounds__(64*Waves,Occupancy) void glm53_down_prefill_tp4_kernel(
    const opus::fp8_t* __restrict__ activations,
    const uint8_t* __restrict__ activation_scales,
    const uint8_t* __restrict__ all_data,
    const uint8_t* __restrict__ all_auxiliary,
    const int32_t* __restrict__ tasks,
    const int32_t* __restrict__ task_count,
    __hip_bfloat16* __restrict__ output,
    int M,int expert_count,int data_bytes,int auxiliary_bytes,int chunk)
{
#if defined(__gfx950__)
    constexpr int K=KT*128,Rows=16*MAtoms,Columns=48*Waves*Groups,Threads=64*Waves;
    struct Storage {
        alignas(16) uint64_t codebook[512];
        alignas(16) uint8_t input[Rows*K];
        uint8_t scales[Rows*(K/32)];
    };
    __shared__ Storage shared;
    const int lane=threadIdx.x,wave=threadIdx.y,linear=wave*64+lane;
    const int lane_row=lane%16,lane_group=lane/16;
    const int nt=task_count[0];
    opus::gmem<uint8_t> a_buffer(activations,static_cast<unsigned int>(M*K));
    opus::gmem<opus::bf16_t> out_buffer(output,static_cast<unsigned int>(M*6144*sizeof(__hip_bfloat16)));
    opus::gmem<uint8_t> scale_buffer(activation_scales,static_cast<unsigned int>(M*(K/32)));
    int previous_expert=-1;
    for(int work=blockIdx.x;work<nt*(6144/Columns/Chunks);work+=gridDim.x)
    {
        int index=work;
        if constexpr(XCD>0)
        {
            const int total=nt*(6144/Columns/Chunks);
            const int per=(total+XCD-1)/XCD,tall=total%XCD==0?XCD:total%XCD;
            const int die=work%XCD,local=work/XCD;
            index=die<tall?die*per+local:tall*per+(die-tall)*(per-1)+local;
        }
        int task=index%nt,output_tile=index/nt;
        if constexpr(TaskGroup>0)
        {
            const int first=(index/(TaskGroup*(6144/Columns/Chunks)))*TaskGroup;
            const int valid=min(TaskGroup,nt-first),local=index%(TaskGroup*(6144/Columns/Chunks));
            task=first+local%valid;output_tile=local/valid;
        }
        const int n_tile=output_tile+chunk*(6144/Columns/Chunks);
        const int begin=tasks[task*3],expert=tasks[task*3+2];
        int count=tasks[task*3+1];
        if(begin<0 || count<=0 || begin+count>M || expert<0 || expert>=expert_count)continue;
        if constexpr(Merge)
        {
            // Pair consecutive row-contiguous tasks of one expert; the second
            // task of each pair is covered by the first.
            int position=0;
            for(int t=task-1;t>=0 && tasks[t*3+2]==expert && tasks[t*3]+tasks[t*3+1]==tasks[(t+1)*3];--t)++position;
            if(position&1)continue;
            if(task+1<nt && tasks[(task+1)*3+2]==expert && tasks[(task+1)*3]==begin+count)count+=tasks[(task+1)*3+1];
        }
        const uint8_t* data=all_data+static_cast<int64_t>(expert)*data_bytes;
        const uint8_t* aux=all_auxiliary+static_cast<int64_t>(expert)*auxiliary_bytes;
        if(previous_expert!=expert)
        {
            for(int i=linear;i<512;i+=Threads)shared.codebook[i]=reinterpret_cast<const uint64_t*>(aux)[i];
            previous_expert=expert;
        }
        for(int row_base=begin;row_base<begin+count;row_base+=Rows)
        {
            const int row_count=min(Rows,begin+count-row_base);
            const int active_m=(row_count+15)/16;
            static_assert(Ablate>=0);
            if(Ablate!=4 || work==static_cast<int>(blockIdx.x))
#pragma unroll
            for(int copy=0;copy<(Rows*K+Threads*16-1)/(Threads*16);++copy)
            {
                const int offset=(linear+copy*Threads)*16;
                const int r=offset/K,col=(offset%K)^((r&15)<<4);
                const int src=(row_base+min(r,row_count-1))*K+col;
                const int dest=(wave*64+copy*Threads)*16;
                if(dest<Rows*K)a_buffer.template async_load<16>(shared.input+dest,src);
            }
#pragma unroll
            for(int part=0;part<(Rows*(K/128)+Threads-1)/Threads;++part)
            if(linear+part*Threads<Rows*(K/128))
            {
                const int r=(linear+part*Threads)/(K/128),col=((linear+part*Threads)%(K/128))*4;
                scale_buffer.template async_load<4>(shared.scales+(wave*64+part*Threads)*4,(row_base+min(r,row_count-1))*(K/32)+col);
            }
            IQ2RCompressedTriplet pending[Prefetch?KT:1];
            uint32_t pending_bases=0;
            if constexpr(Prefetch)
            {
                const int nb=n_tile*(Columns/16)+wave*3;
#pragma unroll
                for(int atom=0;atom<3;++atom)pending_bases|=static_cast<uint32_t>(aux[kCodebookBytes+nb+atom])<<(atom*8);
#pragma unroll
                for(int kt=0;kt<KT;++kt)pending[kt]=iq2r_load_compact_triplet(data,glm53_ws_source_base<KT>(nb,kt),lane);
            }
            if constexpr(Ablate==4)
            {
                // Timing only: stage the A tile once per CTA.
            }
            asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
            __syncthreads();
#pragma unroll 1
            for(int group=0;group<Groups;++group)
            {
                const int n_block=n_tile*(Columns/16)+wave*3+group*Waves*3;
                uint32_t bases=pending_bases;
                IQ2RCompressedTriplet compressed[KT];
                if constexpr(Prefetch)
                {
#pragma unroll
                    for(int kt=0;kt<KT;++kt)compressed[kt]=pending[kt];
                    if(group+1<Groups)
                    {
                        const int nb=n_block+Waves*3;
                        pending_bases=0;
#pragma unroll
                        for(int atom=0;atom<3;++atom)pending_bases|=static_cast<uint32_t>(aux[kCodebookBytes+nb+atom])<<(atom*8);
#pragma unroll
                        for(int kt=0;kt<KT;++kt)pending[kt]=iq2r_load_compact_triplet(data,glm53_ws_source_base<KT>(nb,kt),lane);
                        asm volatile("" ::: "memory");
                    }
                }
                else
                {
                    bases=0;
#pragma unroll
                    for(int atom=0;atom<3;++atom)bases|=static_cast<uint32_t>(aux[kCodebookBytes+n_block+atom])<<(atom*8);
                    if constexpr(!KLoop)
#pragma unroll
                    for(int kt=0;kt<KT;++kt)compressed[kt]=iq2r_load_compact_triplet(data,glm53_ws_source_base<KT>(n_block,kt),lane);
                }
                opus::vector_t<float,4> sum[MAtoms][3];
                if constexpr(KLoop)
                {
                    // Rolled K loop: one compressed K tile live plus a one-ahead
                    // load, keeping VGPRs low enough for more resident waves.
                    IQ2RCompressedTriplet current=iq2r_load_compact_triplet(data,glm53_ws_source_base<KT>(n_block,0),lane);
#pragma unroll 1
                    for(int kt=0;kt<KT;++kt)
                    {
                        IQ2RCompressedTriplet next=current;
                        if(kt+1<KT)next=iq2r_load_compact_triplet(data,glm53_ws_source_base<KT>(n_block,kt+1),lane);
                        opus::i32x8_t w[3];uint32_t sb[3];
                        if constexpr(Ablate==2)
                        {
#pragma unroll
                            for(int q=0;q<8;++q){w[0][q]=reinterpret_cast<const int*>(&current.paired)[q%4];w[1][q]=reinterpret_cast<const int*>(&current.paired)[(q+1)%4];w[2][q]=reinterpret_cast<const int*>(&current.third)[q%3];}
                            sb[0]=sb[1]=sb[2]=127u;
                        }
                        else
                        {
                        glm53_decode_packed_atom<0,true>(current,shared.codebook,bases,w[0],sb[0]);
                        glm53_decode_packed_atom<1,true>(current,shared.codebook,bases,w[1],sb[1]);
                        glm53_decode_packed_atom<2,true>(current,shared.codebook,bases,w[2],sb[2]);
                        }
#pragma unroll
                        for(int m=0;m<MAtoms;++m)
                        {
                            if(m>=active_m)continue;
                            const int r=m*16+lane_row,col=kt*128+lane_group*16;
                            IQ2RActivationFragment a;
                            uint32_t sa=127u;
                            if constexpr(Ablate==3)
                            {
#pragma unroll
                                for(int q=0;q<8;++q)a.words[q]=r*131+q+kt;
                            }
                            else
                            {
                            *reinterpret_cast<uint4*>(a.bytes)=*reinterpret_cast<const uint4*>(shared.input+((r*K+col)^((r&15)<<4)));
                            *reinterpret_cast<uint4*>(a.bytes+16)=*reinterpret_cast<const uint4*>(shared.input+((r*K+col+64)^((r&15)<<4)));
                            sa=shared.scales[r*(K/32)+kt*4+lane_group];
                            }
#pragma unroll
                            for(int atom=0;atom<3;++atom)
                            {
                                auto mma=opus::mfma<opus::fp8_t,opus::fp8_t,opus::fp32_t,16,16,128>{};
                                opus::vector_t<float,4> part{};
                                if constexpr(Ablate==1)
                                {
#pragma unroll
                                    for(int c=0;c<4;++c)part[c]=__int_as_float((w[atom][c]^w[atom][c+4]^a.words[c]^a.words[c+4])&0x3fffffff)+static_cast<float>(sb[atom]+sa);
                                }
                                else part=mma(w[atom],a.words,part,sb[atom],sa);
                                if(kt==0)sum[m][atom]=part;
                                else
#pragma unroll
                                    for(int c=0;c<4;++c)sum[m][atom][c]+=part[c];
                            }
                        }
                        current=next;
                    }
                }
                else
#pragma unroll
                for(int kt=0;kt<KT;++kt)
                {
                    opus::i32x8_t w[3];uint32_t sb[3];
                    glm53_decode_packed_atom<0,true>(compressed[kt],shared.codebook,bases,w[0],sb[0]);
                    glm53_decode_packed_atom<1,true>(compressed[kt],shared.codebook,bases,w[1],sb[1]);
                    glm53_decode_packed_atom<2,true>(compressed[kt],shared.codebook,bases,w[2],sb[2]);
#pragma unroll
                    for(int m=0;m<MAtoms;++m)
                    {
                        if(m>=active_m)continue;
                        const int r=m*16+lane_row,col=kt*128+lane_group*16;
                        IQ2RActivationFragment a;
                        *reinterpret_cast<uint4*>(a.bytes)=*reinterpret_cast<const uint4*>(shared.input+((r*K+col)^((r&15)<<4)));
                        *reinterpret_cast<uint4*>(a.bytes+16)=*reinterpret_cast<const uint4*>(shared.input+((r*K+col+64)^((r&15)<<4)));
                        const uint32_t sa=shared.scales[r*(K/32)+kt*4+lane_group];
#pragma unroll
                        for(int atom=0;atom<3;++atom)
                        {
                            auto mma=opus::mfma<opus::fp8_t,opus::fp8_t,opus::fp32_t,16,16,128>{};
                            opus::vector_t<float,4> part{};
                            part=mma(w[atom],a.words,part,sb[atom],sa);
                            if(kt==0)sum[m][atom]=part;
                            else
#pragma unroll
                                for(int c=0;c<4;++c)sum[m][atom][c]+=part[c];
                        }
                    }
                }
#pragma unroll
                for(int m=0;m<MAtoms;++m)
                {
                    if(m>=active_m)continue;
                    const int out_r=m*16+lane_row;
#pragma unroll
                    for(int atom=0;atom<3;++atom)
                    {
                        const int out_col=n_block*16+atom*16+lane_group*4;
                        opus::vector_t<opus::bf16_t,4> packed;
#pragma unroll
                        for(int c=0;c<4;++c)packed[c]=__builtin_bit_cast(opus::bf16_t,__float2bfloat16(sum[m][atom][c]));
                        if(out_r<row_count)out_buffer.template store<4>(packed,((out_col/384)*M+row_base+out_r)*384+out_col%384);
                    }
                }
            }
            __syncthreads();
        }
    }
#endif
}

// TP8: waves cover independent N tiles. Cooperatively cache A once per CTA.
// Retain two separate K128 accumulators and their final addition so moving K
// into one wave preserves the existing split-K arithmetic exactly.
template<int Waves,int MAtoms,int Groups,bool Prefetch,int XCD,int TaskGroup,int Chunks,bool Tiled>
__global__ __launch_bounds__(64*Waves,1) void glm53_down_prefill_tp8_kernel(
    const opus::fp8_t* __restrict__ activations,
    const uint8_t* __restrict__ activation_scales,
    const uint8_t* __restrict__ all_data,
    const uint8_t* __restrict__ all_auxiliary,
    const int32_t* __restrict__ tasks,
    const int32_t* __restrict__ task_count,
    const __hip_bfloat16* __restrict__ all_bias,
    __hip_bfloat16* __restrict__ output,
    int M,int N,int K,int expert_count,int data_bytes,int auxiliary_bytes,int chunk)
{
#if defined(__gfx950__)
    constexpr int Rows=16*MAtoms,Columns=48*Waves*Groups,Threads=64*Waves;
    struct Storage {
        alignas(16) uint64_t codebook[512];
        alignas(16) uint8_t input[Rows*256];
        uint8_t scales[Rows*8];
    };
    __shared__ Storage shared;
    const int lane=threadIdx.x,wave=threadIdx.y,linear=wave*64+lane;
    const int lane_row=lane%16,lane_group=lane/16;
    const int nt=task_count[0];
    opus::gmem<uint8_t> a_buffer(activations,static_cast<unsigned int>(M*256));
    // Descriptor bounds are bytes; typed offsets below are BF16 elements.
    opus::gmem<opus::bf16_t> out_buffer(output,static_cast<unsigned int>(M*6144*sizeof(__hip_bfloat16)));
    opus::gmem<uint8_t> scale_buffer(activation_scales,static_cast<unsigned int>(M*8));
    int previous_expert=-1;
    for(int work=blockIdx.x;work<nt*(6144/Columns/Chunks);work+=gridDim.x)
    {
        int index=work;
        if constexpr(XCD>0)
        {
            const int total=nt*(6144/Columns/Chunks);
            const int per=(total+XCD-1)/XCD,tall=total%XCD==0?XCD:total%XCD;
            const int die=work%XCD,local=work/XCD;
            index=die<tall?die*per+local:tall*per+(die-tall)*(per-1)+local;
        }
        int task=index%nt,output_tile=index/nt;
        if constexpr(TaskGroup>0)
        {
            const int first=(index/(TaskGroup*(6144/Columns/Chunks)))*TaskGroup;
            const int valid=min(TaskGroup,nt-first),local=index%(TaskGroup*(6144/Columns/Chunks));
            task=first+local%valid;output_tile=local/valid;
        }
        const int n_tile=output_tile+chunk*(6144/Columns/Chunks);
        const int begin=tasks[task*3],count=tasks[task*3+1],expert=tasks[task*3+2];
        if(begin<0 || count<=0 || begin+count>M || expert<0 || expert>=expert_count)continue;
        const uint8_t* data=all_data+static_cast<int64_t>(expert)*data_bytes;
        const uint8_t* aux=all_auxiliary+static_cast<int64_t>(expert)*auxiliary_bytes;
        if(previous_expert!=expert)
        {
            for(int i=linear;i<512;i+=Threads)shared.codebook[i]=reinterpret_cast<const uint64_t*>(aux)[i];
            previous_expert=expert;
        }
        for(int row_base=begin;row_base<begin+count;row_base+=Rows)
        {
            const int row_count=min(Rows,begin+count-row_base);
            const int active_m=(row_count+15)/16;
#pragma unroll
            for(int copy=0;copy<(Rows*256+Threads*16-1)/(Threads*16);++copy)
            {
                const int offset=(linear+copy*Threads)*16;
                const int r=offset/256,col=(offset%256)^((r&15)<<4);
                const int src=(row_base+min(r,row_count-1))*256+col;
                const int dest=(wave*64+copy*Threads)*16;
                if(dest<Rows*256)a_buffer.template async_load<16>(shared.input+dest,src);
            }
            if(linear<Rows*2)
            {
                const int r=linear/2,col=(linear%2)*4;
                scale_buffer.template async_load<4>(shared.scales+wave*64*4,(row_base+min(r,row_count-1))*8+col);
            }
            IQ2RCompressedTriplet pending[2];
            uint32_t pending_bases=0;
            if constexpr(Prefetch)
            {
                const int nb=n_tile*(Columns/16)+wave*3;
#pragma unroll
                for(int atom=0;atom<3;++atom)pending_bases|=static_cast<uint32_t>(aux[kCodebookBytes+nb+atom])<<(atom*8);
#pragma unroll
                for(int kt=0;kt<2;++kt)pending[kt]=iq2r_load_compact_triplet(data,static_cast<int>(triplet_base(physical_tile(nb,kt,2))),lane);
            }
            asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
            __syncthreads();

#pragma unroll 1
            for(int group=0;group<Groups;++group)
            {
        const int n_block=n_tile*(Columns/16)+wave*3+group*Waves*3;
        uint32_t bases=0;
        IQ2RCompressedTriplet compressed[2];
        if constexpr(Prefetch)
        {
            bases=pending_bases;
            compressed[0]=pending[0];compressed[1]=pending[1];
            if(group+1<Groups)
            {
                const int nb=n_block+Waves*3;
                pending_bases=0;
#pragma unroll
                for(int atom=0;atom<3;++atom)pending_bases|=static_cast<uint32_t>(aux[kCodebookBytes+nb+atom])<<(atom*8);
#pragma unroll
                for(int kt=0;kt<2;++kt)pending[kt]=iq2r_load_compact_triplet(data,static_cast<int>(triplet_base(physical_tile(nb,kt,2))),lane);
                asm volatile("" ::: "memory");
            }
        }
        else
        {
#pragma unroll
            for(int atom=0;atom<3;++atom)bases|=static_cast<uint32_t>(aux[kCodebookBytes+n_block+atom])<<(atom*8);
#pragma unroll
            for(int kt=0;kt<2;++kt)compressed[kt]=iq2r_load_compact_triplet(data,static_cast<int>(triplet_base(physical_tile(n_block,kt,2))),lane);
        }
            // Finish one N16 atom at a time. Retain the exact two K128
            // partial sums, while shortening the lifetime of decoded weights
            // and accumulators enough to support a full M64 tile.
            {
                opus::vector_t<float,4> first[MAtoms]={};
                opus::i32x8_t w0;uint32_t sb0;
                glm53_decode_packed_atom<0,true>(compressed[0],shared.codebook,bases,w0,sb0);
#pragma unroll
                for(int m=0;m<MAtoms;++m)
                {
                    if(m>=active_m)continue;
                    const int r=m*16+lane_row,col=lane_group*16;
                    IQ2RActivationFragment a;
                    *reinterpret_cast<uint4*>(a.bytes)=*reinterpret_cast<const uint4*>(shared.input+((r*256+col)^((r&15)<<4)));
                    *reinterpret_cast<uint4*>(a.bytes+16)=*reinterpret_cast<const uint4*>(shared.input+((r*256+col+64)^((r&15)<<4)));
                    const uint32_t sa=shared.scales[r*8+lane_group];
                    auto mma=opus::mfma<opus::fp8_t,opus::fp8_t,opus::fp32_t,16,16,128>{};
                    first[m]=mma(w0,a.words,first[m],sb0,sa);
                }
                opus::i32x8_t w1;uint32_t sb1;
                glm53_decode_packed_atom<0,true>(compressed[1],shared.codebook,bases,w1,sb1);
#pragma unroll
                for(int m=0;m<MAtoms;++m)
                {
                    if(m>=active_m)continue;
                    const int r=m*16+lane_row,col=128+lane_group*16;
                    IQ2RActivationFragment a;
                    *reinterpret_cast<uint4*>(a.bytes)=*reinterpret_cast<const uint4*>(shared.input+((r*256+col)^((r&15)<<4)));
                    *reinterpret_cast<uint4*>(a.bytes+16)=*reinterpret_cast<const uint4*>(shared.input+((r*256+col+64)^((r&15)<<4)));
                    const uint32_t sa=shared.scales[r*8+4+lane_group];
                    opus::vector_t<float,4> second={};
                    auto mma=opus::mfma<opus::fp8_t,opus::fp8_t,opus::fp32_t,16,16,128>{};
                    second=mma(w1,a.words,second,sb1,sa);
                    const int out_r=m*16+lane_row;
                    const int out_col=n_block*16+0*16+lane_group*4;
                    opus::vector_t<opus::bf16_t,4> packed;
#pragma unroll
                    for(int c=0;c<4;++c)
                    {
                        float value=first[m][c]+second[c];
                        if(all_bias)value+=__bfloat162float(all_bias[static_cast<int64_t>(expert)*6144+out_col+c]);
                        packed[c]=__builtin_bit_cast(opus::bf16_t,__float2bfloat16(value));
                    }
                    if(out_r<row_count)out_buffer.template store<4>(packed,(Tiled?(n_tile*M+row_base+out_r)*Columns+out_col%Columns:(row_base+out_r)*6144+out_col));
                }
            }
            {
                opus::vector_t<float,4> first[MAtoms]={};
                opus::i32x8_t w0;uint32_t sb0;
                glm53_decode_packed_atom<1,true>(compressed[0],shared.codebook,bases,w0,sb0);
#pragma unroll
                for(int m=0;m<MAtoms;++m)
                {
                    if(m>=active_m)continue;
                    const int r=m*16+lane_row,col=lane_group*16;
                    IQ2RActivationFragment a;
                    *reinterpret_cast<uint4*>(a.bytes)=*reinterpret_cast<const uint4*>(shared.input+((r*256+col)^((r&15)<<4)));
                    *reinterpret_cast<uint4*>(a.bytes+16)=*reinterpret_cast<const uint4*>(shared.input+((r*256+col+64)^((r&15)<<4)));
                    const uint32_t sa=shared.scales[r*8+lane_group];
                    auto mma=opus::mfma<opus::fp8_t,opus::fp8_t,opus::fp32_t,16,16,128>{};
                    first[m]=mma(w0,a.words,first[m],sb0,sa);
                }
                opus::i32x8_t w1;uint32_t sb1;
                glm53_decode_packed_atom<1,true>(compressed[1],shared.codebook,bases,w1,sb1);
#pragma unroll
                for(int m=0;m<MAtoms;++m)
                {
                    if(m>=active_m)continue;
                    const int r=m*16+lane_row,col=128+lane_group*16;
                    IQ2RActivationFragment a;
                    *reinterpret_cast<uint4*>(a.bytes)=*reinterpret_cast<const uint4*>(shared.input+((r*256+col)^((r&15)<<4)));
                    *reinterpret_cast<uint4*>(a.bytes+16)=*reinterpret_cast<const uint4*>(shared.input+((r*256+col+64)^((r&15)<<4)));
                    const uint32_t sa=shared.scales[r*8+4+lane_group];
                    opus::vector_t<float,4> second={};
                    auto mma=opus::mfma<opus::fp8_t,opus::fp8_t,opus::fp32_t,16,16,128>{};
                    second=mma(w1,a.words,second,sb1,sa);
                    const int out_r=m*16+lane_row;
                    const int out_col=n_block*16+1*16+lane_group*4;
                    opus::vector_t<opus::bf16_t,4> packed;
#pragma unroll
                    for(int c=0;c<4;++c)
                    {
                        float value=first[m][c]+second[c];
                        if(all_bias)value+=__bfloat162float(all_bias[static_cast<int64_t>(expert)*6144+out_col+c]);
                        packed[c]=__builtin_bit_cast(opus::bf16_t,__float2bfloat16(value));
                    }
                    if(out_r<row_count)out_buffer.template store<4>(packed,(Tiled?(n_tile*M+row_base+out_r)*Columns+out_col%Columns:(row_base+out_r)*6144+out_col));
                }
            }
            {
                opus::vector_t<float,4> first[MAtoms]={};
                opus::i32x8_t w0;uint32_t sb0;
                glm53_decode_packed_atom<2,true>(compressed[0],shared.codebook,bases,w0,sb0);
#pragma unroll
                for(int m=0;m<MAtoms;++m)
                {
                    if(m>=active_m)continue;
                    const int r=m*16+lane_row,col=lane_group*16;
                    IQ2RActivationFragment a;
                    *reinterpret_cast<uint4*>(a.bytes)=*reinterpret_cast<const uint4*>(shared.input+((r*256+col)^((r&15)<<4)));
                    *reinterpret_cast<uint4*>(a.bytes+16)=*reinterpret_cast<const uint4*>(shared.input+((r*256+col+64)^((r&15)<<4)));
                    const uint32_t sa=shared.scales[r*8+lane_group];
                    auto mma=opus::mfma<opus::fp8_t,opus::fp8_t,opus::fp32_t,16,16,128>{};
                    first[m]=mma(w0,a.words,first[m],sb0,sa);
                }
                opus::i32x8_t w1;uint32_t sb1;
                glm53_decode_packed_atom<2,true>(compressed[1],shared.codebook,bases,w1,sb1);
#pragma unroll
                for(int m=0;m<MAtoms;++m)
                {
                    if(m>=active_m)continue;
                    const int r=m*16+lane_row,col=128+lane_group*16;
                    IQ2RActivationFragment a;
                    *reinterpret_cast<uint4*>(a.bytes)=*reinterpret_cast<const uint4*>(shared.input+((r*256+col)^((r&15)<<4)));
                    *reinterpret_cast<uint4*>(a.bytes+16)=*reinterpret_cast<const uint4*>(shared.input+((r*256+col+64)^((r&15)<<4)));
                    const uint32_t sa=shared.scales[r*8+4+lane_group];
                    opus::vector_t<float,4> second={};
                    auto mma=opus::mfma<opus::fp8_t,opus::fp8_t,opus::fp32_t,16,16,128>{};
                    second=mma(w1,a.words,second,sb1,sa);
                    const int out_r=m*16+lane_row;
                    const int out_col=n_block*16+2*16+lane_group*4;
                    opus::vector_t<opus::bf16_t,4> packed;
#pragma unroll
                    for(int c=0;c<4;++c)
                    {
                        float value=first[m][c]+second[c];
                        if(all_bias)value+=__bfloat162float(all_bias[static_cast<int64_t>(expert)*6144+out_col+c]);
                        packed[c]=__builtin_bit_cast(opus::bf16_t,__float2bfloat16(value));
                    }
                    if(out_r<row_count)out_buffer.template store<4>(packed,(Tiled?(n_tile*M+row_base+out_r)*Columns+out_col%Columns:(row_base+out_r)*6144+out_col));
                }
            }
            }
            __syncthreads();
        }
    }
#endif
}

// Same nine ordered FP32 FMAs, restricted to the just-produced N range.
template<int Chunks,bool Tiled>
__global__ __launch_bounds__(128) void glm53_reduce_chunk_kernel(
    const __hip_bfloat16* __restrict__ route_output,
    const float* __restrict__ route_weights,const int32_t* __restrict__ scatter,
    __hip_bfloat16* __restrict__ output,int tokens,int chunk)
{
    constexpr int Width=6144/Chunks,Tiles=(Width+1023)/1024;
    const int token=blockIdx.x/Tiles,tile=blockIdx.x%Tiles;
    const int local=tile*1024+threadIdx.x*8;
    if(token>=tokens || local>=Width)return;
    const int col=chunk*Width+local;
    float sums[8]={};
#pragma unroll
    for(int route=0;route<9;++route)
    {
        const int sorted=__builtin_amdgcn_readfirstlane(scatter[token*9+route]);
        const float weight=__builtin_bit_cast(float,__builtin_amdgcn_readfirstlane(__builtin_bit_cast(uint32_t,route_weights[token*9+route])));
        const int64_t address=Tiled?(static_cast<int64_t>(col/384)*tokens*9+sorted)*384+col%384:static_cast<int64_t>(sorted)*6144+col;
        opus::vector_t<opus::bf16_t,8> values;
        *reinterpret_cast<uint4*>(&values)=*reinterpret_cast<const uint4*>(route_output+address);
#pragma unroll
        for(int i=0;i<8;++i)sums[i]=fmaf(static_cast<float>(values[i]),weight,sums[i]);
    }
    opus::vector_t<opus::bf16_t,8> packed;
#pragma unroll
    for(int i=0;i<8;++i)packed[i]=__builtin_bit_cast(opus::bf16_t,__float2bfloat16(sums[i]));
    *reinterpret_cast<uint4*>(output+static_cast<int64_t>(token)*6144+col)=*reinterpret_cast<const uint4*>(&packed);
}

__global__ __launch_bounds__(576,1) void glm53_down_route9_kernel(
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
            uint32_t sb0,sb1,sb2;
            glm53_decode_packed_atom<0,true>(compressed,shared.codebook[route_group],bases,b[0],sb0);
            glm53_decode_packed_atom<1,true>(compressed,shared.codebook[route_group],bases,b[1],sb1);
            glm53_decode_packed_atom<2,true>(compressed,shared.codebook[route_group],bases,b[2],sb2);
            const uint32_t sb=sb0|(sb1<<8)|(sb2<<16);
            asm volatile("" : "+v"(sa));
            iq2r_triplet_mfma<0>(a.words,b,k_sums[ktile],sa*0x01010101u,sb);
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

__global__ __launch_bounds__(576,1) void glm53_down_route9_kernel_tp4(
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
            uint32_t sb0,sb1,sb2;
            glm53_decode_packed_atom<0,true>(compressed,shared.codebook[route_group],bases,b[0],sb0);
            glm53_decode_packed_atom<1,true>(compressed,shared.codebook[route_group],bases,b[1],sb1);
            glm53_decode_packed_atom<2,true>(compressed,shared.codebook[route_group],bases,b[2],sb2);
            const uint32_t sb=sb0|(sb1<<8)|(sb2<<16);
            asm volatile("" : "+v"(sa));
            iq2r_triplet_mfma<0>(a.words,b,k_sums[ktile],sa*0x01010101u,sb);
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

template<int Word>
__device__ __forceinline__ uint64_t glm53_sign_word(uint64_t magnitude,uint32_t signs)
{
    uint32_t lo,hi;
    asm volatile(
        "v_lshlrev_b32 %0, %3, %1\n\t"
        "v_and_or_b32 %0, %0, %4, %2\n\t"
        : "=&v"(lo) : "v"(signs),"v"(static_cast<uint32_t>(magnitude)),
          "n"(7-2*Word),"s"(0x80808080u));
    if constexpr(Word==3) {
        asm volatile("v_and_or_b32 %0, %1, %3, %2"
            : "=v"(hi) : "v"(signs),"v"(static_cast<uint32_t>(magnitude>>32)),"s"(0x80808080u));
    } else {
        asm volatile(
            "v_lshlrev_b32 %0, %3, %1\n\t"
            "v_and_or_b32 %0, %0, %4, %2\n\t"
            : "=&v"(hi) : "v"(signs),"v"(static_cast<uint32_t>(magnitude>>32)),
              "n"(6-2*Word),"s"(0x80808080u));
    }
    return (static_cast<uint64_t>(hi)<<32)|lo;
}

template<int MAtoms,int Batch>
__device__ __forceinline__ void glm53_quad_decode_mfma(
    const IQ2RCompressedQuad& compressed,const uint64_t* codebook,uint32_t bases,
    const IQ2RActivationFragment* activation,const uint32_t* scale_a,
    opus::vector_t<float,4> (&accumulators)[MAtoms][4],int active_m)
{
#pragma unroll
    for(int atom=0;atom<4;++atom)
    {
        const uint32_t lows=atom==0?compressed.first.x:atom==1?compressed.first.z:atom==2?compressed.second.x:compressed.second.z;
        const uint32_t signs=atom==0?compressed.first.y:atom==1?compressed.first.w:atom==2?compressed.second.y:compressed.second.w;
        const uint32_t highs=(compressed.metadata>>(atom*8))&15u;
        union { opus::i32x8_t words;uint64_t codewords[4]; } decoded;
        uint64_t mag[4];
        if constexpr(Batch==2) {
            mag[0]=codebook[lows&511u];
            mag[1]=codebook[(lows>>9)&511u];
            asm volatile("" : "+v"(mag[0]), "+v"(mag[1]));
            decoded.codewords[0]=glm53_sign_word<0>(mag[0],signs);
            decoded.codewords[1]=glm53_sign_word<1>(mag[1],signs);
            mag[2]=codebook[(lows>>18)&511u];
            mag[3]=codebook[(lows>>27)|(highs<<5)];
            asm volatile("" : "+v"(mag[2]), "+v"(mag[3]));
            decoded.codewords[2]=glm53_sign_word<2>(mag[2],signs);
            decoded.codewords[3]=glm53_sign_word<3>(mag[3],signs);
        }
        if constexpr(Batch==4) {
            mag[0]=codebook[lows&511u];
            mag[1]=codebook[(lows>>9)&511u];
            mag[2]=codebook[(lows>>18)&511u];
            mag[3]=codebook[(lows>>27)|(highs<<5)];
            asm volatile("" : "+v"(mag[0]), "+v"(mag[1]), "+v"(mag[2]), "+v"(mag[3]));
            decoded.codewords[0]=glm53_sign_word<0>(mag[0],signs);
            decoded.codewords[1]=glm53_sign_word<1>(mag[1],signs);
            decoded.codewords[2]=glm53_sign_word<2>(mag[2],signs);
            decoded.codewords[3]=glm53_sign_word<3>(mag[3],signs);
        }
        const uint32_t sb=((bases>>(atom*8))&255u)+((compressed.metadata>>(atom*8+4))&15u);
#pragma unroll
        for(int m=0;m<MAtoms;++m)
        {
            if(m>=active_m)continue;
            auto mma=opus::mfma<opus::fp8_t,opus::fp8_t,opus::fp32_t,16,16,128>{};
            accumulators[m][atom]=mma(activation[m].words,decoded.words,accumulators[m][atom],
                                      scale_a[m],sb,opus::number<0>{},opus::number<0>{});
        }

    }
}

// TP8: waves cover independent N tiles. Cooperatively cache A once per CTA.
// M256 uses one sequential FP32 chain across both K128 tiles.
// Preserve that chain, including its rounding, in this explicit family.
template<int Waves,int MAtoms,int Groups,bool Prefetch,int XCD,int TaskGroup,bool SignOr>
__global__ __launch_bounds__(64*Waves,1) void glm53_down_tp8_ordered_kernel(
    const opus::fp8_t* __restrict__ activations,
    const uint8_t* __restrict__ activation_scales,
    const uint8_t* __restrict__ all_data,
    const uint8_t* __restrict__ all_auxiliary,
    const int32_t* __restrict__ tasks,
    const int32_t* __restrict__ task_count,
    const __hip_bfloat16* __restrict__ all_bias,
    __hip_bfloat16* __restrict__ output,
    int M,int N,int K,int expert_count,int data_bytes,int auxiliary_bytes)
{
#if defined(__gfx950__)
    constexpr int Rows=16*MAtoms,Columns=48*Waves*Groups,Threads=64*Waves;
    struct Storage {
        alignas(16) uint64_t codebook[512];
        alignas(16) uint8_t input[Rows*256];
        uint8_t scales[Rows*8];
    };
    __shared__ Storage shared;
    const int lane=threadIdx.x,wave=threadIdx.y,linear=wave*64+lane;
    const int lane_row=lane%16,lane_group=lane/16;
    const int nt=task_count[0];
    opus::gmem<uint8_t> a_buffer(activations,static_cast<unsigned int>(M*256));
    // Descriptor bounds are bytes; typed offsets below are BF16 elements.
    opus::gmem<opus::bf16_t> out_buffer(output,static_cast<unsigned int>(M*6144*sizeof(__hip_bfloat16)));
    opus::gmem<uint8_t> scale_buffer(activation_scales,static_cast<unsigned int>(M*8));
    int previous_expert=-1;
    for(int work=blockIdx.x;work<nt*(6144/Columns);work+=gridDim.x)
    {
        int index=work;
        if constexpr(XCD>0)
        {
            const int total=nt*(6144/Columns);
            const int per=(total+XCD-1)/XCD,tall=total%XCD==0?XCD:total%XCD;
            const int die=work%XCD,local=work/XCD;
            index=die<tall?die*per+local:tall*per+(die-tall)*(per-1)+local;
        }
        int task=index%nt,output_tile=index/nt;
        if constexpr(TaskGroup>0)
        {
            const int first=(index/(TaskGroup*(6144/Columns)))*TaskGroup;
            const int valid=min(TaskGroup,nt-first),local=index%(TaskGroup*(6144/Columns));
            task=first+local%valid;output_tile=local/valid;
        }
        const int n_tile=output_tile;
        const int begin=tasks[task*3],count=tasks[task*3+1],expert=tasks[task*3+2];
        if(begin<0 || count<=0 || begin+count>M || expert<0 || expert>=expert_count)continue;
        const uint8_t* data=all_data+static_cast<int64_t>(expert)*data_bytes;
        const uint8_t* aux=all_auxiliary+static_cast<int64_t>(expert)*auxiliary_bytes;
        if(previous_expert!=expert)
        {
            for(int i=linear;i<512;i+=Threads)shared.codebook[i]=reinterpret_cast<const uint64_t*>(aux)[i];
            previous_expert=expert;
        }
        for(int row_base=begin;row_base<begin+count;row_base+=Rows)
        {
            const int row_count=min(Rows,begin+count-row_base);
            const int active_m=(row_count+15)/16;
#pragma unroll
            for(int copy=0;copy<(Rows*256+Threads*16-1)/(Threads*16);++copy)
            {
                const int offset=(linear+copy*Threads)*16;
                const int r=offset/256,col=(offset%256)^((r&15)<<4);
                const int src=(row_base+min(r,row_count-1))*256+col;
                const int dest=(wave*64+copy*Threads)*16;
                if(dest<Rows*256)a_buffer.template async_load<16>(shared.input+dest,src);
            }
            if(linear<Rows*2)
            {
                const int r=linear/2,col=(linear%2)*4;
                scale_buffer.template async_load<4>(shared.scales+wave*64*4,(row_base+min(r,row_count-1))*8+col);
            }
            IQ2RCompressedTriplet pending[2];
            uint32_t pending_bases=0;
            if constexpr(Prefetch)
            {
                const int nb=n_tile*(Columns/16)+wave*3;
#pragma unroll
                for(int atom=0;atom<3;++atom)pending_bases|=static_cast<uint32_t>(aux[kCodebookBytes+nb+atom])<<(atom*8);
#pragma unroll
                for(int kt=0;kt<2;++kt)pending[kt]=iq2r_load_compact_triplet(data,static_cast<int>(triplet_base(physical_tile(nb,kt,2))),lane);
            }
            asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
            __syncthreads();

#pragma unroll 1
            for(int group=0;group<Groups;++group)
            {
        const int n_block=n_tile*(Columns/16)+wave*3+group*Waves*3;
        uint32_t bases=0;
        IQ2RCompressedTriplet compressed[2];
        if constexpr(Prefetch)
        {
            bases=pending_bases;
            compressed[0]=pending[0];compressed[1]=pending[1];
            if(group+1<Groups)
            {
                const int nb=n_block+Waves*3;
                pending_bases=0;
#pragma unroll
                for(int atom=0;atom<3;++atom)pending_bases|=static_cast<uint32_t>(aux[kCodebookBytes+nb+atom])<<(atom*8);
#pragma unroll
                for(int kt=0;kt<2;++kt)pending[kt]=iq2r_load_compact_triplet(data,static_cast<int>(triplet_base(physical_tile(nb,kt,2))),lane);
                asm volatile("" ::: "memory");
            }
        }
        else
        {
#pragma unroll
            for(int atom=0;atom<3;++atom)bases|=static_cast<uint32_t>(aux[kCodebookBytes+n_block+atom])<<(atom*8);
#pragma unroll
            for(int kt=0;kt<2;++kt)compressed[kt]=iq2r_load_compact_triplet(data,static_cast<int>(triplet_base(physical_tile(n_block,kt,2))),lane);
        }
            // Finish one N16 atom at a time, carrying K0 directly into
            // the K1 MFMA as in the original M256 kernel.
            {
                opus::vector_t<float,4> first[MAtoms]={};
                opus::i32x8_t w0;uint32_t sb0;
                glm53_decode_packed_atom<0,SignOr>(compressed[0],shared.codebook,bases,w0,sb0);
#pragma unroll
                for(int m=0;m<MAtoms;++m)
                {
                    if(m>=active_m)continue;
                    const int r=m*16+lane_row,col=lane_group*16;
                    IQ2RActivationFragment a;
                    *reinterpret_cast<uint4*>(a.bytes)=*reinterpret_cast<const uint4*>(shared.input+((r*256+col)^((r&15)<<4)));
                    *reinterpret_cast<uint4*>(a.bytes+16)=*reinterpret_cast<const uint4*>(shared.input+((r*256+col+64)^((r&15)<<4)));
                    const uint32_t sa=shared.scales[r*8+lane_group];
                    auto mma=opus::mfma<opus::fp8_t,opus::fp8_t,opus::fp32_t,16,16,128>{};
                    first[m]=mma(w0,a.words,first[m],sb0,sa);
                }
                opus::i32x8_t w1;uint32_t sb1;
                glm53_decode_packed_atom<0,SignOr>(compressed[1],shared.codebook,bases,w1,sb1);
#pragma unroll
                for(int m=0;m<MAtoms;++m)
                {
                    if(m>=active_m)continue;
                    const int r=m*16+lane_row,col=128+lane_group*16;
                    IQ2RActivationFragment a;
                    *reinterpret_cast<uint4*>(a.bytes)=*reinterpret_cast<const uint4*>(shared.input+((r*256+col)^((r&15)<<4)));
                    *reinterpret_cast<uint4*>(a.bytes+16)=*reinterpret_cast<const uint4*>(shared.input+((r*256+col+64)^((r&15)<<4)));
                    const uint32_t sa=shared.scales[r*8+4+lane_group];
                    opus::vector_t<float,4> second={};
                    auto mma=opus::mfma<opus::fp8_t,opus::fp8_t,opus::fp32_t,16,16,128>{};
                    second=mma(w1,a.words,first[m],sb1,sa);
                    const int out_r=m*16+lane_row;
                    const int out_col=n_block*16+0*16+lane_group*4;
                    opus::vector_t<opus::bf16_t,4> packed;
#pragma unroll
                    for(int c=0;c<4;++c)
                    {
                        float value=second[c];
                        if(all_bias)value+=__bfloat162float(all_bias[static_cast<int64_t>(expert)*6144+out_col+c]);
                        packed[c]=__builtin_bit_cast(opus::bf16_t,__float2bfloat16(value));
                    }
                    if(out_r<row_count)out_buffer.template store<4>(packed,(row_base+out_r)*6144+out_col);
                }
            }
            {
                opus::vector_t<float,4> first[MAtoms]={};
                opus::i32x8_t w0;uint32_t sb0;
                glm53_decode_packed_atom<1,SignOr>(compressed[0],shared.codebook,bases,w0,sb0);
#pragma unroll
                for(int m=0;m<MAtoms;++m)
                {
                    if(m>=active_m)continue;
                    const int r=m*16+lane_row,col=lane_group*16;
                    IQ2RActivationFragment a;
                    *reinterpret_cast<uint4*>(a.bytes)=*reinterpret_cast<const uint4*>(shared.input+((r*256+col)^((r&15)<<4)));
                    *reinterpret_cast<uint4*>(a.bytes+16)=*reinterpret_cast<const uint4*>(shared.input+((r*256+col+64)^((r&15)<<4)));
                    const uint32_t sa=shared.scales[r*8+lane_group];
                    auto mma=opus::mfma<opus::fp8_t,opus::fp8_t,opus::fp32_t,16,16,128>{};
                    first[m]=mma(w0,a.words,first[m],sb0,sa);
                }
                opus::i32x8_t w1;uint32_t sb1;
                glm53_decode_packed_atom<1,SignOr>(compressed[1],shared.codebook,bases,w1,sb1);
#pragma unroll
                for(int m=0;m<MAtoms;++m)
                {
                    if(m>=active_m)continue;
                    const int r=m*16+lane_row,col=128+lane_group*16;
                    IQ2RActivationFragment a;
                    *reinterpret_cast<uint4*>(a.bytes)=*reinterpret_cast<const uint4*>(shared.input+((r*256+col)^((r&15)<<4)));
                    *reinterpret_cast<uint4*>(a.bytes+16)=*reinterpret_cast<const uint4*>(shared.input+((r*256+col+64)^((r&15)<<4)));
                    const uint32_t sa=shared.scales[r*8+4+lane_group];
                    opus::vector_t<float,4> second={};
                    auto mma=opus::mfma<opus::fp8_t,opus::fp8_t,opus::fp32_t,16,16,128>{};
                    second=mma(w1,a.words,first[m],sb1,sa);
                    const int out_r=m*16+lane_row;
                    const int out_col=n_block*16+1*16+lane_group*4;
                    opus::vector_t<opus::bf16_t,4> packed;
#pragma unroll
                    for(int c=0;c<4;++c)
                    {
                        float value=second[c];
                        if(all_bias)value+=__bfloat162float(all_bias[static_cast<int64_t>(expert)*6144+out_col+c]);
                        packed[c]=__builtin_bit_cast(opus::bf16_t,__float2bfloat16(value));
                    }
                    if(out_r<row_count)out_buffer.template store<4>(packed,(row_base+out_r)*6144+out_col);
                }
            }
            {
                opus::vector_t<float,4> first[MAtoms]={};
                opus::i32x8_t w0;uint32_t sb0;
                glm53_decode_packed_atom<2,SignOr>(compressed[0],shared.codebook,bases,w0,sb0);
#pragma unroll
                for(int m=0;m<MAtoms;++m)
                {
                    if(m>=active_m)continue;
                    const int r=m*16+lane_row,col=lane_group*16;
                    IQ2RActivationFragment a;
                    *reinterpret_cast<uint4*>(a.bytes)=*reinterpret_cast<const uint4*>(shared.input+((r*256+col)^((r&15)<<4)));
                    *reinterpret_cast<uint4*>(a.bytes+16)=*reinterpret_cast<const uint4*>(shared.input+((r*256+col+64)^((r&15)<<4)));
                    const uint32_t sa=shared.scales[r*8+lane_group];
                    auto mma=opus::mfma<opus::fp8_t,opus::fp8_t,opus::fp32_t,16,16,128>{};
                    first[m]=mma(w0,a.words,first[m],sb0,sa);
                }
                opus::i32x8_t w1;uint32_t sb1;
                glm53_decode_packed_atom<2,SignOr>(compressed[1],shared.codebook,bases,w1,sb1);
#pragma unroll
                for(int m=0;m<MAtoms;++m)
                {
                    if(m>=active_m)continue;
                    const int r=m*16+lane_row,col=128+lane_group*16;
                    IQ2RActivationFragment a;
                    *reinterpret_cast<uint4*>(a.bytes)=*reinterpret_cast<const uint4*>(shared.input+((r*256+col)^((r&15)<<4)));
                    *reinterpret_cast<uint4*>(a.bytes+16)=*reinterpret_cast<const uint4*>(shared.input+((r*256+col+64)^((r&15)<<4)));
                    const uint32_t sa=shared.scales[r*8+4+lane_group];
                    opus::vector_t<float,4> second={};
                    auto mma=opus::mfma<opus::fp8_t,opus::fp8_t,opus::fp32_t,16,16,128>{};
                    second=mma(w1,a.words,first[m],sb1,sa);
                    const int out_r=m*16+lane_row;
                    const int out_col=n_block*16+2*16+lane_group*4;
                    opus::vector_t<opus::bf16_t,4> packed;
#pragma unroll
                    for(int c=0;c<4;++c)
                    {
                        float value=second[c];
                        if(all_bias)value+=__bfloat162float(all_bias[static_cast<int64_t>(expert)*6144+out_col+c]);
                        packed[c]=__builtin_bit_cast(opus::bf16_t,__float2bfloat16(value));
                    }
                    if(out_r<row_count)out_buffer.template store<4>(packed,(row_base+out_r)*6144+out_col);
                }
            }
            }
            __syncthreads();
        }
    }
#endif
}

// Each explicit ds_read_b64 contributes one outstanding LDS operation.
// A group consumes four operations. With lookahead, wait4 completes the
// current group while permitting the next four reads to remain outstanding.
__device__ __forceinline__ void glm53_issue_codebook(
    const IQ2RCompressedQuad& compressed,const uint64_t* codebook,int atom,
    uint64_t (&mag)[4])
{
    const uint32_t lows=atom==0?compressed.first.x:atom==1?compressed.first.z:atom==2?compressed.second.x:compressed.second.z;
    const uint32_t highs=(compressed.metadata>>(atom*8))&15u;
    const uint32_t base=static_cast<uint32_t>(reinterpret_cast<uintptr_t>(codebook));
    const uint32_t a0=base+(lows&511u)*8;
    const uint32_t a1=base+((lows>>9)&511u)*8;
    const uint32_t a2=base+((lows>>18)&511u)*8;
    const uint32_t a3=base+((lows>>27)|(highs<<5))*8;
    asm volatile(
        "ds_read_b64 %0, %4\n\t"
        "ds_read_b64 %1, %5\n\t"
        "ds_read_b64 %2, %6\n\t"
        "ds_read_b64 %3, %7\n\t"
        : "=&v"(mag[0]), "=&v"(mag[1]), "=&v"(mag[2]), "=&v"(mag[3])
        : "v"(a0), "v"(a1), "v"(a2), "v"(a3) : "memory");
}

template<int MAtoms,bool Lookahead>
__device__ __forceinline__ void glm53_quad_decode_mfma_cached(
    const IQ2RCompressedQuad& compressed,const uint64_t* codebook,uint32_t bases,
    const IQ2RActivationFragment* activation,const uint32_t* scale_a,
    opus::vector_t<float,4> (&accumulators)[MAtoms][4],int active_m)
{
    uint64_t pending[4];
    if constexpr(Lookahead)glm53_issue_codebook(compressed,codebook,0,pending);
#pragma unroll
    for(int atom=0;atom<4;++atom)
    {
        uint64_t mag[4];
        if constexpr(Lookahead) {
#pragma unroll
            for(int i=0;i<4;++i)mag[i]=pending[i];
            if(atom<3)glm53_issue_codebook(compressed,codebook,atom+1,pending);
        } else glm53_issue_codebook(compressed,codebook,atom,mag);
        if constexpr(Lookahead) {
            if(atom<3)asm volatile("s_waitcnt lgkmcnt(4)" ::: "memory");
            else asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
        } else asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
        const uint32_t signs=atom==0?compressed.first.y:atom==1?compressed.first.w:atom==2?compressed.second.y:compressed.second.w;
        union { opus::i32x8_t words;uint64_t codewords[4]; } decoded;
        decoded.codewords[0]=glm53_sign_word<0>(mag[0],signs);
        decoded.codewords[1]=glm53_sign_word<1>(mag[1],signs);
        decoded.codewords[2]=glm53_sign_word<2>(mag[2],signs);
        decoded.codewords[3]=glm53_sign_word<3>(mag[3],signs);
        const uint32_t sb=((bases>>(atom*8))&255u)+((compressed.metadata>>(atom*8+4))&15u);
#pragma unroll
        for(int m=0;m<MAtoms;++m) {
            if(m>=active_m)continue;
            auto mma=opus::mfma<opus::fp8_t,opus::fp8_t,opus::fp32_t,16,16,128>{};
            accumulators[m][atom]=mma(activation[m].words,decoded.words,accumulators[m][atom],
                                      scale_a[m],sb,opus::number<0>{},opus::number<0>{});
        }
    }
}

template<int MAtoms,bool XCD,int LoadMode,int Batch,int NT=8>
__global__ __launch_bounds__(512,1) void glm53_gate_decode_kernel(
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
    struct Storage {
        alignas(16) uint64_t codebook[kCodebookBytes/8];
        union {
            alignas(16) uint8_t cache[8][kQuadBytes];
            alignas(16) float partial[8][4][4][64];
        } reuse;
    };
    __shared__ Storage shared;
    const int lane=threadIdx.x,wave=threadIdx.y,linear=wave*64+lane;
    const int lane_row=lane%16,lane_group=lane/16;
    const int num_tasks=task_count[0];
    const int total=num_tasks*Subtasks*NT;
    for(int work=blockIdx.x;work<total;work+=gridDim.x)
    {
        const int index=XCD?glm53_remap8(work,total):work;
        const int task=XCD?index/(NT*Subtasks):(work/Subtasks)%num_tasks;
        const int sub=XCD?(index/NT)%Subtasks:work%Subtasks;
        const int n_tile=XCD?index%NT:work/(num_tasks*Subtasks);
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
        IQ2RCompressedQuad pending;
        if constexpr(LoadMode==1)pending=iq2r_scheduled_load_quad(data,data_bytes,base,lane);
        else iq2r_issue_quad(buffer,shared.reuse.cache[wave],lane,base);
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
            IQ2RCompressedQuad compressed;
            if constexpr(LoadMode==1) {
                compressed=pending;
                if(iteration<5) {
                    base+=kQuadBytes;
                    pending=iq2r_scheduled_load_quad(data,data_bytes,base,lane);
                }
            } else {
                iq2r_wait_vmcnt<0>();
                compressed=iq2r_read_quad(shared.reuse.cache[wave],lane);
                asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
                if(iteration<5) {
                    base+=kQuadBytes;
                    iq2r_issue_quad(buffer,shared.reuse.cache[wave],lane,base);
                }
            }
            glm53_quad_decode_mfma_cached<MAtoms,true>(compressed,shared.codebook,bases,a,sa,accumulators,active_m);
        }
        // No wave may overwrite another wave's cache before its final read.
        __syncthreads();
#pragma unroll
        for(int m=0;m<MAtoms;++m)
        {
            if(m>=active_m)continue;
#pragma unroll
            for(int atom=0;atom<4;++atom)
                for(int c=0;c<4;++c) shared.reuse.partial[wave][atom][c][lane]=accumulators[m][atom][c];
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
                        value+=shared.reuse.partial[source][atom][wave][lane];
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
                            output[static_cast<int64_t>(output_row)*(NT*32)+n_tile*32+atom*8+lane_row/2]=opus::fp32_to_fp8(activated[atom]*inverse);
                        if(lane_row==0) output_scales[static_cast<int64_t>(output_row)*NT+n_tile]=bs.byte;
                    }
                }
            }
            __syncthreads();
        }
    }
#endif
}

template<int MAtoms>
__device__ __forceinline__ void glm53_cross_k_decode_mfma(
    const IQ2RCompressedQuad& compressed,const IQ2RCompressedQuad& next,
    const uint64_t* codebook,uint32_t bases,
    const IQ2RActivationFragment* activation,const uint32_t* scale_a,
    opus::vector_t<float,4> (&accumulators)[MAtoms][4],int active_m,
    uint64_t (&pending)[4],bool first_k,bool next_k)
{
    if(first_k)glm53_issue_codebook(compressed,codebook,0,pending);
#pragma unroll
    for(int atom=0;atom<4;++atom)
    {
        uint64_t mag[4];
#pragma unroll
        for(int i=0;i<4;++i)mag[i]=pending[i];
        if(atom<3)glm53_issue_codebook(compressed,codebook,atom+1,pending);
        else if(next_k)glm53_issue_codebook(next,codebook,0,pending);
        if(atom<3 || next_k)asm volatile("s_waitcnt lgkmcnt(4)" ::: "memory");
        else asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
        const uint32_t signs=atom==0?compressed.first.y:atom==1?compressed.first.w:atom==2?compressed.second.y:compressed.second.w;
        union { opus::i32x8_t words;uint64_t codewords[4]; } decoded;
        decoded.codewords[0]=glm53_sign_word<0>(mag[0],signs);
        decoded.codewords[1]=glm53_sign_word<1>(mag[1],signs);
        decoded.codewords[2]=glm53_sign_word<2>(mag[2],signs);
        decoded.codewords[3]=glm53_sign_word<3>(mag[3],signs);
        const uint32_t sb=((bases>>(atom*8))&255u)+((compressed.metadata>>(atom*8+4))&15u);
#pragma unroll
        for(int m=0;m<MAtoms;++m) {
            if(m>=active_m)continue;
            auto mma=opus::mfma<opus::fp8_t,opus::fp8_t,opus::fp32_t,16,16,128>{};
            accumulators[m][atom]=mma(activation[m].words,decoded.words,accumulators[m][atom],
                                      scale_a[m],sb,opus::number<0>{},opus::number<0>{});
        }
    }
}

template<int MAtoms,bool XCD,int LoadMode,int Batch,int NT=8>
__global__ __launch_bounds__(512,1) void glm53_gate_decode_nobarrier_kernel(
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
    static_assert(MAtoms==1 && XCD && LoadMode==1 && Batch==4);
    constexpr int K=6144, Rows=16*MAtoms, Subtasks=1;
    struct Storage {
        alignas(16) uint64_t codebook[kCodebookBytes/8];
        union {
            alignas(16) uint8_t cache[8][kQuadBytes];
            alignas(16) float partial[8][4][4][64];
        } reuse;
    };
    __shared__ Storage shared;
    const int lane=threadIdx.x,wave=threadIdx.y,linear=wave*64+lane;
    const int lane_row=lane%16,lane_group=lane/16;
    const int num_tasks=task_count[0];
    const int total=num_tasks*Subtasks*NT;
    int previous_expert=-1;
    for(int work=blockIdx.x;work<total;work+=gridDim.x)
    {
        const int index=XCD?glm53_remap8(work,total):work;
        const int task=XCD?index/(NT*Subtasks):(work/Subtasks)%num_tasks;
        const int sub=XCD?(index/NT)%Subtasks:work%Subtasks;
        const int n_tile=XCD?index%NT:work/(num_tasks*Subtasks);
        const int row_begin=tasks[task*3]+sub*Rows;
        const int row_end=min(row_begin+Rows,tasks[task*3]+tasks[task*3+1]);
        const int expert=tasks[task*3+2];
        const int active_m=(row_end-row_begin+15)/16;
        if(row_begin>=row_end || row_begin<0 || row_end>routes || expert<0 || expert>=257) continue;
        const uint8_t* data=all_data+static_cast<int64_t>(expert)*data_bytes;
        const uint8_t* aux=all_auxiliary+static_cast<int64_t>(expert)*auxiliary_bytes;
        // Task expert and validity are uniform across this workgroup.
        // The previous item ends in a barrier. Codebook storage is disjoint
        // from the partial-output union, and is valid only in this invocation.
        // Issue the codebook, bases, gather and first weight loads together;
        // only the codebook is waited on before the LDS store and barrier.
        const bool new_expert=expert!=previous_expert;
        uint64_t book=0;
        if(new_expert) book=reinterpret_cast<const uint64_t*>(aux)[linear];
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
        IQ2RCompressedQuad pending;
        uint64_t pending_book[4]={};
        if constexpr(LoadMode==1)pending=iq2r_scheduled_load_quad(data,data_bytes,base,lane);
        if(new_expert) {
            shared.codebook[linear]=book;
            __syncthreads();
            previous_expert=expert;
        }
        if constexpr(LoadMode!=1)iq2r_issue_quad(buffer,shared.reuse.cache[wave],lane,base);
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
            IQ2RCompressedQuad compressed;
            if constexpr(LoadMode==1) {
                compressed=pending;
                if(iteration<5) {
                    base+=kQuadBytes;
                    pending=iq2r_scheduled_load_quad(data,data_bytes,base,lane);
                }
            } else {
                iq2r_wait_vmcnt<0>();
                compressed=iq2r_read_quad(shared.reuse.cache[wave],lane);
                asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
                if(iteration<5) {
                    base+=kQuadBytes;
                    iq2r_issue_quad(buffer,shared.reuse.cache[wave],lane,base);
                }
            }
            glm53_cross_k_decode_mfma<MAtoms>(compressed,pending,shared.codebook,bases,a,sa,accumulators,active_m,pending_book,iteration==0,iteration<5);
        }
        // LoadMode=1 keeps weights in registers. Codebook LDS is separate
        // from the partial-output union. The following store barrier and
        // final reader barrier still protect every shared partial value.
#pragma unroll
        for(int m=0;m<MAtoms;++m)
        {
            if(m>=active_m)continue;
#pragma unroll
            for(int atom=0;atom<4;++atom)
                for(int c=0;c<4;++c) shared.reuse.partial[wave][atom][c][lane]=accumulators[m][atom][c];
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
                        value+=shared.reuse.partial[source][atom][wave][lane];
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
                            output[static_cast<int64_t>(output_row)*(NT*32)+n_tile*32+atom*8+lane_row/2]=opus::fp32_to_fp8(activated[atom]*inverse);
                        if(lane_row==0) output_scales[static_cast<int64_t>(output_row)*NT+n_tile]=bs.byte;
                    }
                }
            }
            __syncthreads();
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

    HipDeviceGuard device_guard(device);
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
    HipDeviceGuard device_guard(data.device_id);
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

// GLM-5.3 IQ2R packed MoE launchers. Tensor-parallel width follows from the
// packed gate stack: 16 N tiles (intermediate 512) at TP4, 8 tiles at TP8.
namespace {

constexpr int kGlm53Hidden = 6144;
constexpr int kGlm53Experts = 257;
constexpr int kGlm53TopK = 9;

int glm53_tp4_from_gate(const aiter_tensor_t& data, const aiter_tensor_t& auxiliary)
{
    const bool tp4 = data.dim() == 2 && data.size(1) == 16 * 48 * kQuadBytes;
    AITER_CHECK(data.dtype() == AITER_DTYPE_u8 && data.dim() == 2 && data.size(0) == kGlm53Experts &&
                    (tp4 || data.size(1) == 8 * 48 * kQuadBytes),
                "GLM-5.3 IQ2R gate requires the packed quad stack [257, N/32*48*2304]");
    AITER_CHECK(auxiliary.dtype() == AITER_DTYPE_u8 && auxiliary.dim() == 2 &&
                    auxiliary.size(0) == kGlm53Experts &&
                    auxiliary.size(1) == expected_auxiliary_bytes(tp4 ? 1024 : 512),
                "GLM-5.3 IQ2R gate requires canonical auxiliary bytes");
    return tp4;
}

void glm53_check_device(std::initializer_list<const aiter_tensor_t*> tensors, const aiter_tensor_t& data)
{
    for(const auto* tensor : tensors)
        AITER_CHECK(tensor->is_gpu() && tensor->device_id == data.device_id && tensor->is_contiguous(),
                    "GLM-5.3 IQ2R tensors must be contiguous on the weight GPU");
}

void glm53_check_gate_io(const aiter_tensor_t& activations,
                         const aiter_tensor_t& scales,
                         const aiter_tensor_t& output,
                         const aiter_tensor_t& output_scales,
                         int rows,
                         int routes,
                         int intermediate)
{
    AITER_CHECK(activations.dtype() == AITER_DTYPE_fp8 && activations.dim() == 2 &&
                    activations.size(0) == rows && activations.size(1) == kGlm53Hidden,
                "GLM-5.3 IQ2R gate input must be FP8 [rows, 6144]");
    AITER_CHECK(scales.dtype() == AITER_DTYPE_u8 && scales.dim() == 2 && scales.size(0) == rows &&
                    scales.size(1) == kGlm53Hidden / kScaleBlock,
                "GLM-5.3 IQ2R gate input scales must be uint8 [rows, 192]");
    AITER_CHECK(output.dtype() == AITER_DTYPE_fp8 && output.dim() == 2 && output.size(0) == routes &&
                    output.size(1) == intermediate && output_scales.dtype() == AITER_DTYPE_u8 &&
                    output_scales.dim() == 2 && output_scales.size(0) == routes &&
                    output_scales.size(1) == intermediate / kScaleBlock,
                "GLM-5.3 IQ2R gate output must be FP8 [routes, I] with uint8 scales [routes, I/32]");
}

void glm53_check_tasks(const aiter_tensor_t& tasks, const aiter_tensor_t& task_count, int64_t capacity)
{
    AITER_CHECK(tasks.dtype() == AITER_DTYPE_i32 && tasks.dim() == 2 && tasks.size(1) == 3 &&
                    tasks.size(0) >= capacity,
                "GLM-5.3 IQ2R tasks must be int32 [capacity, 3]");
    AITER_CHECK(task_count.dtype() == AITER_DTYPE_i32 && task_count.dim() == 1 && task_count.size(0) == 1,
                "GLM-5.3 IQ2R task count must be int32 [1]");
}

int glm53_down_intermediate(const aiter_tensor_t& activations,
                            const aiter_tensor_t& scales,
                            const aiter_tensor_t& data,
                            const aiter_tensor_t& auxiliary)
{
    const int intermediate = static_cast<int>(activations.size(1));
    AITER_CHECK(intermediate == 256 || intermediate == 512, "GLM-5.3 IQ2R down K must be 256 or 512");
    validate_weights(data, auxiliary, kGlm53Hidden, intermediate);
    AITER_CHECK(data.size(0) == kGlm53Experts && activations.dtype() == AITER_DTYPE_fp8 &&
                    activations.dim() == 2 && activations.size(0) % kGlm53TopK == 0,
                "GLM-5.3 IQ2R down input must be FP8 [tokens*9, K]");
    AITER_CHECK(scales.dtype() == AITER_DTYPE_u8 && scales.dim() == 2 &&
                    scales.size(0) == activations.size(0) && scales.size(1) == intermediate / kScaleBlock,
                "GLM-5.3 IQ2R down scales must be uint8 [routes, K/32]");
    return intermediate;
}

} // namespace

void iq2r_glm53_gate_m1_out(const aiter_tensor_t& activations,
                            const aiter_tensor_t& scales,
                            const aiter_tensor_t& data,
                            const aiter_tensor_t& auxiliary,
                            const aiter_tensor_t& tasks,
                            const aiter_tensor_t& task_count,
                            aiter_tensor_t& output,
                            aiter_tensor_t& output_scales)
{
    const bool tp4 = glm53_tp4_from_gate(data, auxiliary);
    glm53_check_device({&activations, &scales, &auxiliary, &tasks, &task_count, &output, &output_scales}, data);
    const int routes = kGlm53TopK;
    glm53_check_gate_io(activations, scales, output, output_scales, routes, routes, tp4 ? 512 : 256);
    glm53_check_tasks(tasks, task_count, routes);
    HipDeviceGuard device_guard(data.device_id);
    hipLaunchKernelGGL((tp4 ? glm53_gate_m1_kernel_tp4<true> : glm53_gate_m1_kernel<true>),
                       dim3(2 * static_cast<int>(get_num_cu_func())),
                       dim3(64, 8),
                       0,
                       getCurrentHIPStream(),
                       static_cast<const opus::fp8_t*>(activations.data_ptr()),
                       static_cast<const uint8_t*>(scales.data_ptr()),
                       static_cast<const uint8_t*>(data.data_ptr()),
                       static_cast<const uint8_t*>(auxiliary.data_ptr()),
                       static_cast<const int32_t*>(tasks.data_ptr()),
                       static_cast<const int32_t*>(task_count.data_ptr()),
                       nullptr,
                       static_cast<opus::fp8_t*>(output.data_ptr()),
                       static_cast<uint8_t*>(output_scales.data_ptr()),
                       routes,
                       static_cast<int>(data.size(1)),
                       static_cast<int>(auxiliary.size(1)));
    HIP_CALL_LAUNCH(hipGetLastError());
}

void iq2r_glm53_gate_out(const aiter_tensor_t& activations,
                         const aiter_tensor_t& scales,
                         const aiter_tensor_t& data,
                         const aiter_tensor_t& auxiliary,
                         const aiter_tensor_t& tasks,
                         const aiter_tensor_t& task_count,
                         const aiter_tensor_t& gather,
                         aiter_tensor_t& output,
                         aiter_tensor_t& output_scales,
                         int64_t kernel,
                         int64_t grid_multiplier)
{
    const bool tp4 = glm53_tp4_from_gate(data, auxiliary);
    glm53_check_device(
        {&activations, &scales, &auxiliary, &tasks, &task_count, &gather, &output, &output_scales}, data);
    const int tokens = static_cast<int>(activations.size(0));
    const int routes = tokens * kGlm53TopK;
    glm53_check_gate_io(activations, scales, output, output_scales, tokens, routes, tp4 ? 512 : 256);
    AITER_CHECK(gather.dtype() == AITER_DTYPE_i32 && gather.dim() == 1 && gather.size(0) == routes,
                "GLM-5.3 IQ2R gate gather must be int32 [routes]");
    AITER_CHECK(grid_multiplier >= 1 && grid_multiplier <= 8, "GLM-5.3 IQ2R gate grid multiplier");
    const bool prefill = kernel == kGlm53GatePrefill;
    AITER_CHECK(kernel == kGlm53GateDecode || prefill || kernel == kGlm53GateDecodeNoBarrier,
                "GLM-5.3 IQ2R gate kernel id");
    AITER_CHECK(tokens >= 2 && tokens <= (prefill ? 4096 : 1024),
                "GLM-5.3 IQ2R gate token range: decode 2..1024, prefill 2..4096");
    const int rows = prefill ? 64 : 16;
    glm53_check_tasks(tasks, task_count, (routes + rows - 1) / rows + std::min(routes, 258));
    HipDeviceGuard device_guard(data.device_id);
    using gate_kernel = void (*)(const opus::fp8_t*, const uint8_t*, const uint8_t*, const uint8_t*,
                                 const int32_t*, const int32_t*, const int32_t*, opus::fp8_t*, uint8_t*,
                                 int, int, int);
    gate_kernel launch;
    int waves = 8;
    if(prefill)
    {
        launch = tp4 ? glm53_gate_prefill_kernel<4, true, 8, 4, 2, 4, 16>
                     : glm53_gate_prefill_kernel<4, true, 8, 4, 2, 4, 8>;
        waves  = 4;
    }
    else if(kernel == kGlm53GateDecodeNoBarrier)
        launch = tp4 ? glm53_gate_decode_nobarrier_kernel<1, true, 1, 4, 16>
                     : glm53_gate_decode_nobarrier_kernel<1, true, 1, 4, 8>;
    else
        launch = tp4 ? glm53_gate_decode_kernel<1, true, 0, 4, 16> : glm53_gate_decode_kernel<1, false, 0, 4, 8>;
    hipLaunchKernelGGL(launch,
                       dim3(static_cast<int>(grid_multiplier) * static_cast<int>(get_num_cu_func())),
                       dim3(64, waves),
                       0,
                       getCurrentHIPStream(),
                       static_cast<const opus::fp8_t*>(activations.data_ptr()),
                       static_cast<const uint8_t*>(scales.data_ptr()),
                       static_cast<const uint8_t*>(data.data_ptr()),
                       static_cast<const uint8_t*>(auxiliary.data_ptr()),
                       static_cast<const int32_t*>(tasks.data_ptr()),
                       static_cast<const int32_t*>(task_count.data_ptr()),
                       static_cast<const int32_t*>(gather.data_ptr()),
                       static_cast<opus::fp8_t*>(output.data_ptr()),
                       static_cast<uint8_t*>(output_scales.data_ptr()),
                       routes,
                       static_cast<int>(data.size(1)),
                       static_cast<int>(auxiliary.size(1)));
    HIP_CALL_LAUNCH(hipGetLastError());
}

void iq2r_glm53_down_out(const aiter_tensor_t& activations,
                         const aiter_tensor_t& scales,
                         const aiter_tensor_t& data,
                         const aiter_tensor_t& auxiliary,
                         const aiter_tensor_t& tasks,
                         const aiter_tensor_t& task_count,
                         aiter_tensor_t& output,
                         int64_t kernel,
                         int64_t grid_multiplier)
{
    const int K = glm53_down_intermediate(activations, scales, data, auxiliary);
    const bool tp4 = K == 512;
    glm53_check_device({&activations, &scales, &auxiliary, &tasks, &task_count, &output}, data);
    const int routes = static_cast<int>(activations.size(0));
    AITER_CHECK(routes >= 2 * kGlm53TopK && routes <= 1024 * kGlm53TopK, "GLM-5.3 IQ2R decode down covers M2..1024");
    glm53_check_tasks(tasks, task_count, (routes + 31) / 32 + std::min(routes, 258));
    AITER_CHECK(output.dtype() == AITER_DTYPE_bf16 && output.dim() == 2 && output.size(0) == routes &&
                    output.size(1) == kGlm53Hidden,
                "GLM-5.3 IQ2R down output must be BF16 [routes, 6144]");
    AITER_CHECK(kernel == kGlm53DownPacked || kernel == kGlm53DownSingle || (kernel == kGlm53DownOrdered && !tp4), "GLM-5.3 IQ2R down kernel id");
    AITER_CHECK(grid_multiplier >= 1 && grid_multiplier <= 16, "GLM-5.3 IQ2R down grid multiplier");
    HipDeviceGuard device_guard(data.device_id);
    const dim3 grid(static_cast<int>(grid_multiplier) * static_cast<int>(get_num_cu_func()));
#define GLM53_DOWN_ARGS                                                                     \
    static_cast<const opus::fp8_t*>(activations.data_ptr()),                                \
        static_cast<const uint8_t*>(scales.data_ptr()),                                     \
        static_cast<const uint8_t*>(data.data_ptr()),                                       \
        static_cast<const uint8_t*>(auxiliary.data_ptr()),                                  \
        static_cast<const int32_t*>(tasks.data_ptr()),                                      \
        static_cast<const int32_t*>(task_count.data_ptr()), nullptr,                        \
        static_cast<__hip_bfloat16*>(output.data_ptr()), routes, kGlm53Hidden, K, kGlm53Experts, \
        static_cast<int>(data.size(1)), static_cast<int>(auxiliary.size(1))
    // Single: one 16-row atom per pass for decode-sized tasks, three workgroups per CU.
    if(tp4 && kernel == kGlm53DownSingle)
        hipLaunchKernelGGL((glm53_down_tp4_kernel<4, 1, 2, 8, 4, 4, 1, false, 3>),
                           grid, dim3(64, 4), 0, getCurrentHIPStream(), GLM53_DOWN_ARGS, 0);
    else if(kernel == kGlm53DownSingle)
        hipLaunchKernelGGL((glm53_down_tp8_ordered_kernel<4, 1, 2, true, 8, 4, true>),
                           grid, dim3(64, 4), 0, getCurrentHIPStream(), GLM53_DOWN_ARGS);
    else if(tp4)
        hipLaunchKernelGGL((glm53_down_tp4_kernel<4, 2, 2, 8, 4, 4>),
                           grid, dim3(64, 4), 0, getCurrentHIPStream(), GLM53_DOWN_ARGS, 0);
    else if(kernel == kGlm53DownOrdered)
        hipLaunchKernelGGL((glm53_down_tp8_ordered_kernel<4, 2, 2, true, 8, 4, true>),
                           grid, dim3(64, 4), 0, getCurrentHIPStream(), GLM53_DOWN_ARGS);
    else
        hipLaunchKernelGGL((glm53_down_tp8_kernel<4, 2, 2, true, 8, 4, true>),
                           grid, dim3(64, 4), 0, getCurrentHIPStream(), GLM53_DOWN_ARGS);
#undef GLM53_DOWN_ARGS
    HIP_CALL_LAUNCH(hipGetLastError());
}

void iq2r_glm53_down_route9_out(const aiter_tensor_t& activations,
                                const aiter_tensor_t& scales,
                                const aiter_tensor_t& data,
                                const aiter_tensor_t& auxiliary,
                                const aiter_tensor_t& expert_ids,
                                const aiter_tensor_t& scatter,
                                const aiter_tensor_t& route_weights,
                                aiter_tensor_t& output)
{
    const int K = glm53_down_intermediate(activations, scales, data, auxiliary);
    glm53_check_device({&activations, &scales, &auxiliary, &expert_ids, &scatter, &route_weights, &output}, data);
    const int routes = static_cast<int>(activations.size(0));
    const int tokens = routes / kGlm53TopK;
    AITER_CHECK(tokens == 1 || tokens == 2 || tokens == 4, "GLM-5.3 IQ2R route9 down covers M1, M2 and M4");
    AITER_CHECK(expert_ids.dtype() == AITER_DTYPE_i32 && expert_ids.numel() == routes &&
                    scatter.dtype() == AITER_DTYPE_i32 && scatter.numel() == routes,
                "GLM-5.3 IQ2R route9 expert ids and scatter must be int32 [tokens, 9]");
    AITER_CHECK(route_weights.dtype() == AITER_DTYPE_fp32 && route_weights.numel() == routes,
                "GLM-5.3 IQ2R route9 weights must be float32 [tokens, 9]");
    AITER_CHECK(output.dtype() == AITER_DTYPE_bf16 && output.dim() == 2 && output.size(0) == tokens &&
                    output.size(1) == kGlm53Hidden,
                "GLM-5.3 IQ2R route9 output must be BF16 [tokens, 6144]");
    HipDeviceGuard device_guard(data.device_id);
    hipLaunchKernelGGL((K == 512 ? glm53_down_route9_kernel_tp4 : glm53_down_route9_kernel),
                       dim3(128, tokens),
                       dim3(64, 9),
                       0,
                       getCurrentHIPStream(),
                       static_cast<const opus::fp8_t*>(activations.data_ptr()),
                       static_cast<const uint8_t*>(scales.data_ptr()),
                       static_cast<const uint8_t*>(data.data_ptr()),
                       static_cast<const uint8_t*>(auxiliary.data_ptr()),
                       static_cast<const int32_t*>(expert_ids.data_ptr()),
                       static_cast<const int32_t*>(scatter.data_ptr()),
                       static_cast<const float*>(route_weights.data_ptr()),
                       static_cast<__hip_bfloat16*>(output.data_ptr()),
                       static_cast<int>(data.size(1)),
                       static_cast<int>(auxiliary.size(1)));
    HIP_CALL_LAUNCH(hipGetLastError());
}

void iq2r_glm53_down_reduce_out(const aiter_tensor_t& activations,
                                const aiter_tensor_t& scales,
                                const aiter_tensor_t& data,
                                const aiter_tensor_t& auxiliary,
                                const aiter_tensor_t& tasks,
                                const aiter_tensor_t& task_count,
                                aiter_tensor_t& route_output,
                                const aiter_tensor_t& scatter,
                                const aiter_tensor_t& route_weights,
                                aiter_tensor_t& output,
                                int64_t chunks)
{
    const int K = glm53_down_intermediate(activations, scales, data, auxiliary);
    glm53_check_device(
        {&activations, &scales, &auxiliary, &tasks, &task_count, &route_output, &scatter, &route_weights, &output},
        data);
    const int routes = static_cast<int>(activations.size(0));
    const int tokens = routes / kGlm53TopK;
    AITER_CHECK(tokens >= 2 && tokens <= 4096, "GLM-5.3 IQ2R prefill down covers M2..4096");
    glm53_check_tasks(tasks, task_count, (routes + 63) / 64 + std::min(routes, 258));
    AITER_CHECK(route_output.dtype() == AITER_DTYPE_bf16 && route_output.dim() == 2 &&
                    route_output.size(0) == routes && route_output.size(1) == kGlm53Hidden,
                "GLM-5.3 IQ2R prefill route output must be BF16 [routes, 6144]");
    AITER_CHECK(scatter.dtype() == AITER_DTYPE_i32 && scatter.numel() >= routes &&
                    route_weights.dtype() == AITER_DTYPE_fp32 && route_weights.numel() == routes,
                "GLM-5.3 IQ2R prefill scatter and weights");
    AITER_CHECK(output.dtype() == AITER_DTYPE_bf16 && output.dim() == 2 && output.size(0) == tokens &&
                    output.size(1) == kGlm53Hidden,
                "GLM-5.3 IQ2R prefill output must be BF16 [tokens, 6144]");
    AITER_CHECK(chunks == 1 || chunks == 2, "GLM-5.3 IQ2R prefill down splits N into 1 or 2 chunks");
    HipDeviceGuard device_guard(data.device_id);
    const int cus = static_cast<int>(get_num_cu_func());
    for(int chunk = 0; chunk < chunks; ++chunk)
    {
#define GLM53_PREFILL_DOWN(KERNEL, GRID, ...)                                                   \
    hipLaunchKernelGGL((KERNEL), dim3((GRID) * cus), dim3(64, 4), 0, getCurrentHIPStream(),      \
                       static_cast<const opus::fp8_t*>(activations.data_ptr()),                \
                       static_cast<const uint8_t*>(scales.data_ptr()),                         \
                       static_cast<const uint8_t*>(data.data_ptr()),                           \
                       static_cast<const uint8_t*>(auxiliary.data_ptr()),                      \
                       static_cast<const int32_t*>(tasks.data_ptr()),                          \
                       static_cast<const int32_t*>(task_count.data_ptr()), __VA_ARGS__,        \
                       static_cast<int>(data.size(1)), static_cast<int>(auxiliary.size(1)), chunk)
        // TP4: 16 persistent CTAs per CU with weight-stationary K loops; TP8:
        // N-wave tiles sharing one cached activation tile.
        if(K == 512 && chunks == 1)
            GLM53_PREFILL_DOWN((glm53_down_prefill_tp4_kernel<4, 2, 8, 4, 4, 1, 3, 4, false, false, true>), 16,
                               static_cast<__hip_bfloat16*>(route_output.data_ptr()), routes, kGlm53Experts);
        else if(K == 512)
            GLM53_PREFILL_DOWN((glm53_down_prefill_tp4_kernel<4, 2, 8, 4, 4, 2, 3, 4, false, false, true>), 16,
                               static_cast<__hip_bfloat16*>(route_output.data_ptr()), routes, kGlm53Experts);
        else if(chunks == 1)
            GLM53_PREFILL_DOWN((glm53_down_prefill_tp8_kernel<4, 2, 2, true, 8, 4, 1, true>), 8, nullptr,
                               static_cast<__hip_bfloat16*>(route_output.data_ptr()), routes, kGlm53Hidden, 256,
                               kGlm53Experts);
        else
            GLM53_PREFILL_DOWN((glm53_down_prefill_tp8_kernel<4, 2, 2, true, 8, 4, 2, true>), 8, nullptr,
                               static_cast<__hip_bfloat16*>(route_output.data_ptr()), routes, kGlm53Hidden, 256,
                               kGlm53Experts);
#undef GLM53_PREFILL_DOWN
        const int width = kGlm53Hidden / static_cast<int>(chunks);
        hipLaunchKernelGGL((chunks == 1 ? glm53_reduce_chunk_kernel<1, true> : glm53_reduce_chunk_kernel<2, true>),
                           dim3(tokens * ((width + 1023) / 1024)),
                           dim3(128),
                           0,
                           getCurrentHIPStream(),
                           static_cast<const __hip_bfloat16*>(route_output.data_ptr()),
                           static_cast<const float*>(route_weights.data_ptr()),
                           static_cast<const int32_t*>(scatter.data_ptr()),
                           static_cast<__hip_bfloat16*>(output.data_ptr()),
                           tokens,
                           chunk);
    }
    HIP_CALL_LAUNCH(hipGetLastError());
}

} // namespace aiter
