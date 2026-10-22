// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

#include "aiter_dispatch.h"
#include "aiter_hip_common.h"
#include "aiter_opus_plus.h"
#include "aiter_stream.h"
#include "mx_quant_utils.h"
#include "quant.h"

#include <cstdint>
#include <type_traits>

namespace aiter {
namespace {

constexpr int kTileRows            = 256;
constexpr int kKTile               = 128;
constexpr int kGroupSize           = 32;
constexpr int kGroupsPerKTile      = kKTile / kGroupSize;
constexpr int kKGuardTiles         = 2;
constexpr int kPackedTileBytes     = 16384;
constexpr int kScaleTileBytes      = 1024;
constexpr int64_t kMaxBufferBytes  = int64_t{1} << 31;
constexpr int kBlockThreads        = 256;
constexpr int kThreadsPerGroup     = 4;
constexpr int kValuesPerThread     = kGroupSize / kThreadsPerGroup;
constexpr int kLargeKThreshold     = 8192;
constexpr int kSmallKStepsPerBlock = 2;
constexpr int kLargeKStepsPerBlock = 3;
constexpr uintptr_t kOutputAlignment = 16;
// _hadamard32_np().astype(bfloat16), represented exactly as fp32.
constexpr float kHadamard32Norm = 0.1767578125f;
constexpr int kHadamardSafetyShift = 3;
constexpr float kHadamard32WorkNorm = kHadamard32Norm * 0.125f;

using packed_u16x8_t = opus::vector_t<uint16_t, 8>;
using packed_f32x8_t = opus::vector_t<float, 8>;

__device__ __forceinline__ float swap_adjacent_lane(float value)
{
    return opus::mov_dpp(value, opus::number<0xb1>{});
}

__device__ __forceinline__ float swap_lane_distance_two(float value)
{
    return opus::mov_dpp(value, opus::number<0x4e>{});
}

template <typename input_t>
__device__ __forceinline__ float load_as_float(input_t input)
{
    return static_cast<float>(input);
}

template <typename input_t>
__device__ __forceinline__ void load_group_values(const input_t* __restrict__ input,
                                                  opus::vector_t<float, kValuesPerThread>& values,
                                                  int64_t row,
                                                  int32_t cols,
                                                  int32_t col)
{
    const int64_t row_offset = row * static_cast<int64_t>(cols);
    if(col + kValuesPerThread <= cols && (cols % kValuesPerThread) == 0)
    {
        if constexpr(std::is_same_v<input_t, float>)
        {
            const input_t* input_ptr = input + row_offset + col;
            if(reinterpret_cast<uintptr_t>(input_ptr) % alignof(packed_f32x8_t) == 0)
            {
                const packed_f32x8_t input_values =
                    *reinterpret_cast<const packed_f32x8_t*>(input_ptr);
#pragma unroll
                for(int i = 0; i < kValuesPerThread; ++i)
                    values[i] = input_values[i];
                return;
            }
        }
        else
        {
            const input_t* input_ptr = input + row_offset + col;
            if(reinterpret_cast<uintptr_t>(input_ptr) % alignof(packed_u16x8_t) == 0)
            {
                const packed_u16x8_t input_bits =
                    *reinterpret_cast<const packed_u16x8_t*>(input_ptr);
                const input_t* input_values = reinterpret_cast<const input_t*>(&input_bits);
#pragma unroll
                for(int i = 0; i < kValuesPerThread; ++i)
                    values[i] = load_as_float(input_values[i]);
                return;
            }
        }
    }

#pragma unroll
    for(int i = 0; i < kValuesPerThread; ++i)
    {
        const int32_t k = col + i;
        values[i]       = k < cols ? load_as_float(input[row_offset + k]) : 0.0f;
    }
}

template <bool Safe>
__device__ __forceinline__ void
hadamard32(opus::vector_t<float, kValuesPerThread>& values, int32_t lane)
{
    const float norm = Safe ? kHadamard32WorkNorm : kHadamard32Norm;
#pragma unroll
    for(int i = 0; i < kValuesPerThread; ++i)
        values[i] *= norm;

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
        const float peer    = swap_lane_distance_two(values[i]);
        const float rotated = lane < 2 ? values[i] + peer : peer - values[i];
        values[i]           = __bfloat162float(__float2bfloat16(rotated));
    }
}

template <MxScaleRoundMode RoundMode>
__device__ __forceinline__ float
group_amax(const opus::vector_t<float, kValuesPerThread>& values)
{
    float local_amax = RoundMode == MxScaleRoundMode::RoundUp ? 1.0e-10f : 0.0f;
#pragma unroll
    for(int i = 0; i < kValuesPerThread; ++i)
        local_amax = fmaxf(local_amax, fabsf(values[i]));
    local_amax = fmaxf(local_amax, swap_adjacent_lane(local_amax));
    return fmaxf(local_amax, swap_lane_distance_two(local_amax));
}

template <typename input_t, MxScaleRoundMode RoundMode>
__device__ __forceinline__ void quant_mxfp4_gemm_group(const input_t* __restrict__ input,
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
    load_group_values(input, values, row, cols, col);
    hadamard32<false>(values, lane);
    float amax = group_amax<RoundMode>(values);
    int safety_shift = 0;

    const uint32_t amax_exponent =
        (__builtin_bit_cast(uint32_t, amax) >> 23) & 0xFFu;
    if(__builtin_expect(amax_exponent == 0xFFu, 0))
    {
        load_group_values(input, values, row, cols, col);
        hadamard32<true>(values, lane);
        amax         = group_amax<RoundMode>(values);
        safety_shift = kHadamardSafetyShift;
    }

    const E8m0BlockScale block_scale =
        fp_f32_to_e8m0_block_scale<RoundMode, MxDtype::FP4_E2M1>(amax);
    const uint32_t unclamped_stored_scale =
        static_cast<uint32_t>(block_scale.byte) + safety_shift;
    const uint32_t stored_scale =
        unclamped_stored_scale > 254u ? 254u : unclamped_stored_scale;
    const uint8_t stored_scale_byte = static_cast<uint8_t>(stored_scale);
    const uint8_t conversion_scale_byte =
        static_cast<uint8_t>(stored_scale - safety_shift);
    // E8M0 byte zero represents the minimum scale 2^-127. The f32 value with
    // exponent field zero is numeric zero, so use the minimum normal/subnormal
    // boundary when feeding the hardware conversion instruction.
    const uint32_t conversion_scale_bits =
        conversion_scale_byte == 0
            ? 0x00400000u
            : static_cast<uint32_t>(conversion_scale_byte) << 23;
    const float conversion_scale = __builtin_bit_cast(float, conversion_scale_bits);

    uint32_t packed_word = 0;
#if defined(__gfx950__)
    if(amax != 0.0f)
    {
        opus::static_for<kValuesPerThread / 2>([&](auto pair) {
            constexpr int i = pair.value;
            packed_word = __builtin_amdgcn_cvt_scalef32_pk_fp4_f32(
                packed_word, values[2 * i], values[2 * i + 1], conversion_scale, i);
        });
    }
#endif

    const int32_t tile_row  = static_cast<int32_t>(row / kTileRows);
    const int32_t rem       = static_cast<int32_t>(row % kTileRows);
    const int32_t row_block = rem / 16;
    const int32_t row16     = rem % 16;
    const int32_t step      = group / kGroupsPerKTile;
    const int32_t k_group   = group % kGroupsPerKTile;
    const int32_t block     = row_block * 64 + k_group * 16 + row16;
    const int64_t tile_base =
        (static_cast<int64_t>(tile_row) * nk_pad + step) * kPackedTileBytes;
    const int64_t c0_address = tile_base + block * 16 + lane * sizeof(uint32_t);
    *reinterpret_cast<uint32_t*>(packed + c0_address) = packed_word;

    if(lane == 0)
    {
        const int32_t scale_upper = rem / 128;
        const int32_t scale_sub   = (rem % 128) / 16;
        const int64_t scale_address =
            (static_cast<int64_t>(tile_row) * nk_pad + step) * kScaleTileBytes +
            scale_upper * 512 + k_group * 128 + row16 * 8 + scale_sub;
        packed_scale[scale_address] = stored_scale_byte;
    }
}

template <typename input_t, MxScaleRoundMode RoundMode, int KStepsPerBlock>
__global__ __launch_bounds__(kBlockThreads) void
quant_mxfp4_gemm_kernel(const input_t* __restrict__ input,
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
            quant_mxfp4_gemm_group<input_t, RoundMode>(
                input, packed, packed_scale, row, cols, group, nk_pad);
        }
    }
}

template <typename input_t>
void launch_quant_mxfp4_gemm(const input_t* input,
                             uint8_t* packed,
                             uint8_t* packed_scale,
                             int64_t rows,
                             int32_t cols,
                             int32_t num_groups,
                             int32_t nk_pad,
                             int32_t grid_size,
                             int32_t k_steps_per_block,
                             int round_mode,
                             hipStream_t stream)
{
#define LAUNCH_MXFP4_GEMM_MODE(MODE)                                                       \
    do                                                                                     \
    {                                                                                      \
        if(k_steps_per_block == kLargeKStepsPerBlock)                                      \
            quant_mxfp4_gemm_kernel<input_t, MODE, kLargeKStepsPerBlock>                   \
                <<<grid_size, kBlockThreads, 0, stream>>>(                                 \
                    input, packed, packed_scale, rows, cols, num_groups, nk_pad);           \
        else                                                                               \
            quant_mxfp4_gemm_kernel<input_t, MODE, kSmallKStepsPerBlock>                   \
                <<<grid_size, kBlockThreads, 0, stream>>>(                                 \
                    input, packed, packed_scale, rows, cols, num_groups, nk_pad);           \
    } while(false)

    switch(static_cast<MxScaleRoundMode>(round_mode))
    {
    case MxScaleRoundMode::RoundDown: LAUNCH_MXFP4_GEMM_MODE(MxScaleRoundMode::RoundDown); break;
    case MxScaleRoundMode::RoundUp: LAUNCH_MXFP4_GEMM_MODE(MxScaleRoundMode::RoundUp); break;
    case MxScaleRoundMode::Even: LAUNCH_MXFP4_GEMM_MODE(MxScaleRoundMode::Even); break;
    case MxScaleRoundMode::Ceil: LAUNCH_MXFP4_GEMM_MODE(MxScaleRoundMode::Ceil); break;
    }
#undef LAUNCH_MXFP4_GEMM_MODE
}

} // namespace

void quant_mxfp4_gemm_hip_out(const aiter_tensor_t& input,
                              aiter_tensor_t& packed,
                              aiter_tensor_t& packed_scale,
                              int round_mode)
{
    AITER_CHECK(input.is_gpu() && packed.is_gpu() && packed_scale.is_gpu(),
                __func__,
                " expected GPU tensors");
    AITER_CHECK(input.device_id == packed.device_id && input.device_id == packed_scale.device_id,
                __func__,
                " expected all tensors on the same GPU");
    HipDeviceGuard device_guard(input.device_id);
    AITER_CHECK(get_gpu_arch() == "gfx950", __func__, " requires gfx950 hardware FP4 conversion");
    AITER_CHECK(input.dim() == 2, __func__, " expected a 2D [rows, K] input");
    AITER_CHECK(input.is_contiguous(), __func__, " expected contiguous input");
    AITER_CHECK(packed.is_contiguous(), __func__, " expected contiguous packed output");
    AITER_CHECK(packed_scale.is_contiguous(), __func__, " expected contiguous packed-scale output");
    AITER_CHECK(packed.dtype() == AITER_DTYPE_u8, __func__, " expected uint8 packed output");
    AITER_CHECK(
        packed_scale.dtype() == AITER_DTYPE_u8, __func__, " expected uint8 packed-scale output");
    AITER_CHECK(
        reinterpret_cast<uintptr_t>(packed.data_ptr()) % kOutputAlignment == 0 &&
            reinterpret_cast<uintptr_t>(packed_scale.data_ptr()) % kOutputAlignment == 0,
        __func__,
        " expected 16-byte-aligned outputs");
    AITER_CHECK(input.dtype() == AITER_DTYPE_bf16 || input.dtype() == AITER_DTYPE_fp16 ||
                    input.dtype() == AITER_DTYPE_fp32,
                __func__,
                " expected bf16, fp16, or fp32 input");
    AITER_CHECK(round_mode >= 0 && round_mode <= 3, __func__, " expected round_mode in {0,1,2,3}");

    const int64_t rows64 = input.size(0);
    const int64_t cols64 = input.size(1);
    AITER_CHECK(rows64 > 0 && cols64 > 0, __func__, " expected non-empty input");
    AITER_CHECK(rows64 <= INT64_MAX - (kTileRows - 1), __func__, " rows exceed int64 range");
    AITER_CHECK(cols64 <= INT32_MAX - (kKTile - 1), __func__, " K exceeds int32 range");
    const int32_t cols = static_cast<int32_t>(cols64);

    const int32_t pad_cols   = (cols + kKTile - 1) / kKTile * kKTile;
    const int64_t pad_rows   = (rows64 + kTileRows - 1) / kTileRows * kTileRows;
    const int32_t num_groups = pad_cols / kGroupSize;
    const int32_t nk_pad     = pad_cols / kKTile + kKGuardTiles;
    const int64_t tile_rows  = pad_rows / kTileRows;
    AITER_CHECK(tile_rows <= kMaxBufferBytes / (static_cast<int64_t>(nk_pad) * kPackedTileBytes),
                __func__,
                " packed output exceeds the 2 GiB address range");
    const int64_t expected_packed =
        tile_rows * static_cast<int64_t>(nk_pad) * kPackedTileBytes;
    const int64_t expected_scale =
        tile_rows * static_cast<int64_t>(nk_pad) * kScaleTileBytes;
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

    const int64_t row_blocks = (rows64 + 15) / 16;
    const int32_t num_steps  = num_groups / kGroupsPerKTile;
    const int32_t k_steps_per_block =
        rows64 < 2048 || cols >= kLargeKThreshold ? kLargeKStepsPerBlock
                                                 : kSmallKStepsPerBlock;
    const int32_t num_work_steps = (num_steps + k_steps_per_block - 1) / k_steps_per_block;
    const int64_t grid_size64    = row_blocks * num_work_steps;
    AITER_CHECK(grid_size64 <= 2147483647LL, __func__, " grid size exceeds maximum");
    const int32_t grid_size = static_cast<int32_t>(grid_size64);

    const hipStream_t stream = aiter::getCurrentHIPStream();
    switch(input.dtype())
    {
    case AITER_DTYPE_bf16:
        launch_quant_mxfp4_gemm(reinterpret_cast<const hip_bfloat16*>(input.data_ptr()),
                                reinterpret_cast<uint8_t*>(packed.data_ptr()),
                                reinterpret_cast<uint8_t*>(packed_scale.data_ptr()),
                                rows64,
                                cols,
                                num_groups,
                                nk_pad,
                                grid_size,
                                k_steps_per_block,
                                round_mode,
                                stream);
        break;
    case AITER_DTYPE_fp16:
        launch_quant_mxfp4_gemm(reinterpret_cast<const __half*>(input.data_ptr()),
                                reinterpret_cast<uint8_t*>(packed.data_ptr()),
                                reinterpret_cast<uint8_t*>(packed_scale.data_ptr()),
                                rows64,
                                cols,
                                num_groups,
                                nk_pad,
                                grid_size,
                                k_steps_per_block,
                                round_mode,
                                stream);
        break;
    case AITER_DTYPE_fp32:
        launch_quant_mxfp4_gemm(reinterpret_cast<const float*>(input.data_ptr()),
                                reinterpret_cast<uint8_t*>(packed.data_ptr()),
                                reinterpret_cast<uint8_t*>(packed_scale.data_ptr()),
                                rows64,
                                cols,
                                num_groups,
                                nk_pad,
                                grid_size,
                                k_steps_per_block,
                                round_mode,
                                stream);
        break;
    default: break;
    }
}

} // namespace aiter
