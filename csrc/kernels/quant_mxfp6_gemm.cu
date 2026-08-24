// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

#include "aiter_dispatch.h"
#include "aiter_hip_common.h"
#include "aiter_opus_plus.h"
#include "aiter_stream.h"
#include "quant.h"

#include <type_traits>

namespace aiter {
namespace {

constexpr int kTileRows            = 256;
constexpr int kKTile               = 128;
constexpr int kGroupSize           = 32;
constexpr int kGroupsPerKTile      = kKTile / kGroupSize;
constexpr int kKGuardTiles         = 2;
constexpr int kPackedTileBytes     = 24576;
constexpr int kScaleTileBytes      = 1024;
constexpr int kBlockThreads        = 256;
constexpr int kThreadsPerGroup     = 4;
constexpr int kValuesPerThread     = kGroupSize / kThreadsPerGroup;
constexpr int kLargeKThreshold     = 8192;
constexpr int kSmallKStepsPerBlock = 2;
constexpr int kLargeKStepsPerBlock = 3;
// _hadamard32_np().astype(bfloat16), represented exactly as fp32.
constexpr float kHadamard32Norm = 0.1767578125f;

using packed_u16x8_t  = opus::vector_t<uint16_t, 8>;
using packed_fp6x32_t = uint32_t __attribute__((ext_vector_type(6)));
using uint4_t         = uint32_t __attribute__((ext_vector_type(4)));
using uint2_t         = uint32_t __attribute__((ext_vector_type(2)));
using float16_t       = float __attribute__((ext_vector_type(16)));

__device__ __forceinline__ float swap_adjacent_lane(float value)
{
    return opus::mov_dpp(value, opus::number<0xb1>{});
}

__device__ __forceinline__ float swap_lane_distance_two(float value)
{
    return opus::mov_dpp(value, opus::number<0x4e>{});
}

template <typename input_t>
__device__ __forceinline__ float to_bf16_dot_operand(input_t input)
{
    const float value = static_cast<float>(input);
    if constexpr(std::is_same_v<input_t, hip_bfloat16>)
        return value;
    return __bfloat162float(__float2bfloat16(value));
}

template <typename input_t>
__device__ __forceinline__ void quant_mxfp6_group(const input_t* __restrict__ input,
                                                  uint8_t* __restrict__ packed,
                                                  uint8_t* __restrict__ packed_scale,
                                                  int64_t row,
                                                  int32_t cols,
                                                  int32_t group,
                                                  int32_t nk_pad)
{
    const int32_t lane = threadIdx.x & (kThreadsPerGroup - 1);
    const int32_t col  = group * kGroupSize + lane * kValuesPerThread;

    opus::vector_t<float, kValuesPerThread> values;
    if(col + kValuesPerThread <= cols && (cols % kValuesPerThread) == 0)
    {
        const packed_u16x8_t input_bits = *reinterpret_cast<const packed_u16x8_t*>(
            input + row * static_cast<int64_t>(cols) + col);
        const input_t* input_values = reinterpret_cast<const input_t*>(&input_bits);
#pragma unroll
        for(int i = 0; i < kValuesPerThread; ++i)
        {
            values[i] = to_bf16_dot_operand(input_values[i]);
        }
    }
    else
    {
#pragma unroll
        for(int i = 0; i < kValuesPerThread; ++i)
        {
            const int32_t k = col + i;
            values[i] =
                k < cols ? to_bf16_dot_operand(input[row * static_cast<int64_t>(cols) + k]) : 0.0f;
        }
    }

    // H8 within each lane, followed by two lane butterflies to form H32.
    opus::static_for<3>([&](auto stage) {
        constexpr int h = 1 << stage.value;
        opus::static_for<kValuesPerThread / 2>([&](auto pair) {
            constexpr int butterfly = pair.value / h;
            constexpr int offset    = pair.value % h;
            constexpr int i0        = butterfly * (2 * h) + offset;
            constexpr int i1        = i0 + h;
            const float x0          = values[i0];
            const float x1          = values[i1];
            values[i0]              = x0 + x1;
            values[i1]              = x0 - x1;
        });
    });

#pragma unroll
    for(int i = 0; i < kValuesPerThread; ++i)
    {
        const float peer = swap_adjacent_lane(values[i]);
        values[i]        = (lane & 1) == 0 ? values[i] + peer : peer - values[i];
    }
#pragma unroll
    for(int i = 0; i < kValuesPerThread; ++i)
    {
        const float peer = swap_lane_distance_two(values[i]);
        values[i]        = (lane < 2 ? values[i] + peer : peer - values[i]) * kHadamard32Norm;
    }

    float local_amax = 0.0f;
#pragma unroll
    for(int i = 0; i < kValuesPerThread; ++i)
        local_amax = fmaxf(local_amax, fabsf(values[i]));
    local_amax       = fmaxf(local_amax, swap_adjacent_lane(local_amax));
    const float amax = fmaxf(local_amax, swap_lane_distance_two(local_amax));

    int32_t scale_unbiased;
    if(amax == 0.0f)
    {
        scale_unbiased = 0;
    }
    else
    {
        const uint32_t exponent = (__builtin_bit_cast(uint32_t, amax) >> 23) & 0xFFu;
        scale_unbiased          = exponent == 0u
                                      ? -127
                                      : (exponent == 0xFFu ? 127 : static_cast<int32_t>(exponent) - 129);
        scale_unbiased          = scale_unbiased < -127 ? -127 : scale_unbiased;
        scale_unbiased          = scale_unbiased > 127 ? 127 : scale_unbiased;
    }
    const uint8_t scale_exp = static_cast<uint8_t>(scale_unbiased + 127);

    // Gather contiguous lane chunks into the even/odd vectors expected by the
    // gfx950 conversion. The instruction interleaves src0/src1 fields, yielding
    // the ordinary sequential 6-bit stream directly.
    float16_t even;
    float16_t odd;
#pragma unroll
    for(int i = 0; i < kValuesPerThread / 2; ++i)
    {
        const float v0_even = values[2 * i];
        const float v0_odd  = values[2 * i + 1];
        const float v1_even = swap_adjacent_lane(v0_even);
        const float v1_odd  = swap_adjacent_lane(v0_odd);
        const float v2_even = swap_lane_distance_two(v0_even);
        const float v2_odd  = swap_lane_distance_two(v0_odd);
        const float v3_even = swap_adjacent_lane(v2_even);
        const float v3_odd  = swap_adjacent_lane(v2_odd);
        even[i]             = v0_even;
        odd[i]              = v0_odd;
        even[4 + i]         = v1_even;
        odd[4 + i]          = v1_odd;
        even[8 + i]         = v2_even;
        odd[8 + i]          = v2_odd;
        even[12 + i]        = v3_even;
        odd[12 + i]         = v3_odd;
    }

    if(lane != 0)
        return;

    const uint32_t scale_bits =
        scale_exp == 0 ? 0x00400000u : static_cast<uint32_t>(scale_exp) << 23;
    const float mx_scale = __builtin_bit_cast(float, scale_bits);
#if defined(__gfx950__)
    const packed_fp6x32_t fp6 =
        amax == 0.0f ? packed_fp6x32_t{}
                     : __builtin_amdgcn_cvt_scalef32_2xpk16_fp6_f32(even, odd, mx_scale);
#else
    const packed_fp6x32_t fp6{};
#endif

    const int32_t tile_row  = static_cast<int32_t>(row / kTileRows);
    const int32_t rem       = static_cast<int32_t>(row % kTileRows);
    const int32_t row_block = rem / 16;
    const int32_t row16     = rem % 16;
    const int32_t step      = group / kGroupsPerKTile;
    const int32_t k_group   = group % kGroupsPerKTile;
    const int32_t block     = row_block * 64 + k_group * 16 + row16;
    const int64_t tile_base = (static_cast<int64_t>(tile_row) * nk_pad + step) * kPackedTileBytes;
    const int64_t c0_base   = tile_base + block * 16;
    const int64_t c1_base   = tile_base + 16384 + block * 8;
    *reinterpret_cast<uint4_t*>(packed + c0_base) = *reinterpret_cast<const uint4_t*>(&fp6);
    *reinterpret_cast<uint2_t*>(packed + c1_base) =
        *reinterpret_cast<const uint2_t*>(reinterpret_cast<const uint8_t*>(&fp6) + 16);

    const int32_t scale_upper = rem / 128;
    const int32_t scale_sub   = (rem % 128) / 16;
    const int64_t scale_address =
        (static_cast<int64_t>(tile_row) * nk_pad + step) * kScaleTileBytes + scale_upper * 512 +
        k_group * 128 + row16 * 8 + scale_sub;
    packed_scale[scale_address] = scale_exp;
}

template <typename input_t, int KStepsPerBlock>
__global__
__launch_bounds__(kBlockThreads) void quant_mxfp6_gemm_kernel(const input_t* __restrict__ input,
                                                              uint8_t* __restrict__ packed,
                                                              uint8_t* __restrict__ packed_scale,
                                                              int64_t rows,
                                                              int32_t cols,
                                                              int32_t num_groups,
                                                              int32_t nk_pad)
{
    const int32_t num_steps       = num_groups / kGroupsPerKTile;
    const int32_t num_work_steps  = (num_steps + KStepsPerBlock - 1) / KStepsPerBlock;
    const int64_t work_row_block  = blockIdx.x / num_work_steps;
    const int32_t work_step_block = blockIdx.x - work_row_block * num_work_steps;
    const int32_t group_local     = threadIdx.x / kThreadsPerGroup;
    const int32_t wave            = group_local / 16;
    const int32_t within_wave     = group_local % 16;
    const int64_t row             = work_row_block * 16 + wave * 4 + within_wave / 4;
    if(row >= rows)
        return;

    for(int32_t local_step = 0; local_step < KStepsPerBlock; ++local_step)
    {
        const int32_t step = work_step_block * KStepsPerBlock + local_step;
        if(step < num_steps)
        {
            const int32_t group = step * kGroupsPerKTile + within_wave % 4;
            quant_mxfp6_group(input, packed, packed_scale, row, cols, group, nk_pad);
        }
    }
}

} // namespace

void quant_mxfp6_gemm_hip(const aiter_tensor_t& input,
                          aiter_tensor_t& packed,
                          aiter_tensor_t& packed_scale)
{
    AITER_CHECK(input.is_gpu() && packed.is_gpu() && packed_scale.is_gpu(),
                __func__,
                " expected GPU tensors");
    AITER_CHECK(input.device_id == packed.device_id && input.device_id == packed_scale.device_id,
                __func__,
                " expected all tensors on the same GPU");
    HipDeviceGuard device_guard(input.device_id);
    const bool is_gfx950 = get_gpu_arch() == "gfx950";
    AITER_CHECK(is_gfx950, __func__, " requires gfx950 hardware FP6 conversion");
    AITER_CHECK(input.dim() == 2, __func__, " expected a 2D [rows, K] input");
    AITER_CHECK(input.is_contiguous(), __func__, " expected contiguous input");
    AITER_CHECK(packed.is_contiguous(), __func__, " expected contiguous packed output");
    AITER_CHECK(packed_scale.is_contiguous(), __func__, " expected contiguous packed-scale output");
    AITER_CHECK(packed.dtype() == AITER_DTYPE_u8, __func__, " expected uint8 packed output");
    AITER_CHECK(
        packed_scale.dtype() == AITER_DTYPE_u8, __func__, " expected uint8 packed-scale output");
    AITER_CHECK(input.dtype() == AITER_DTYPE_bf16 || input.dtype() == AITER_DTYPE_fp16,
                __func__,
                " expected bf16 or fp16 input");

    const int64_t rows = input.size(0);
    const int32_t cols = input.size(1);
    AITER_CHECK(rows > 0 && cols > 0, __func__, " expected non-empty input");

    const int32_t pad_cols   = (cols + kKTile - 1) / kKTile * kKTile;
    const int64_t pad_rows   = (rows + kTileRows - 1) / kTileRows * kTileRows;
    const int32_t num_groups = pad_cols / kGroupSize;
    const int32_t nk_pad     = pad_cols / kKTile + kKGuardTiles;
    const int64_t expected_packed =
        pad_rows / kTileRows * static_cast<int64_t>(nk_pad) * kPackedTileBytes;
    const int64_t expected_scale =
        pad_rows / kTileRows * static_cast<int64_t>(nk_pad) * kScaleTileBytes;
    AITER_CHECK(packed.numel() == expected_packed,
                __func__,
                " packed output has ",
                packed.numel(),
                " bytes, expected ",
                expected_packed);
    AITER_CHECK(packed_scale.numel() == expected_scale,
                __func__,
                " packed-scale output has ",
                packed_scale.numel(),
                " bytes, expected ",
                expected_scale);

    const int64_t row_blocks = (rows + 15) / 16;
    const int32_t num_steps  = num_groups / kGroupsPerKTile;
    const int32_t k_steps_per_block =
        cols >= kLargeKThreshold ? kLargeKStepsPerBlock : kSmallKStepsPerBlock;
    const int32_t num_work_steps = (num_steps + k_steps_per_block - 1) / k_steps_per_block;
    const int64_t grid_size      = row_blocks * num_work_steps;
    AITER_CHECK(grid_size <= 2147483647LL, __func__, " grid size exceeds maximum");

    const hipStream_t stream = aiter::getCurrentHIPStream();
    AITER_DISPATCH_FLOATING16_TYPES_rmTorch(input.dtype(), "quant_mxfp6_gemm_kernel", [&] {
        if(k_steps_per_block == kLargeKStepsPerBlock)
        {
            quant_mxfp6_gemm_kernel<scalar_t, kLargeKStepsPerBlock>
                <<<static_cast<int32_t>(grid_size), kBlockThreads, 0, stream>>>(
                    reinterpret_cast<const scalar_t*>(input.data_ptr()),
                    reinterpret_cast<uint8_t*>(packed.data_ptr()),
                    reinterpret_cast<uint8_t*>(packed_scale.data_ptr()),
                    rows,
                    cols,
                    num_groups,
                    nk_pad);
        }
        else
        {
            quant_mxfp6_gemm_kernel<scalar_t, kSmallKStepsPerBlock>
                <<<static_cast<int32_t>(grid_size), kBlockThreads, 0, stream>>>(
                    reinterpret_cast<const scalar_t*>(input.data_ptr()),
                    reinterpret_cast<uint8_t*>(packed.data_ptr()),
                    reinterpret_cast<uint8_t*>(packed_scale.data_ptr()),
                    rows,
                    cols,
                    num_groups,
                    nk_pad);
        }
    });
}

} // namespace aiter
