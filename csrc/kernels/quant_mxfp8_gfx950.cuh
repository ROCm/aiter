// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include "aiter_hip_common.h"
#include "aiter_opus_plus.h"
#include "mx_quant_utils.h"

namespace aiter::gfx950_ep_quant {

// Match dyn_gq_tuned_arch: query runtime device properties once, not per launch.
inline bool enabled()
{
    static const bool supported = get_gpu_arch() == "gfx950";
    return supported;
}

// Device-row-count specialization used by graph-padded MXFP8 MoE inputs.
// The ordinary launcher sizes its grid from the static allocation, so a MiMo
// EP buffer with M=131072 still launches hundreds of thousands of workgroups
// when only a few thousand received rows are live. One thread owns one 1x32
// group here and advances through the device-reported live prefix with a
// persistent grid, preserving the same E8M0 quantization and output layout.
template <typename DTYPE_I, int block_size = 256>
__global__ void __launch_bounds__(block_size)
quantize_live_rows(opus::fp8_t* __restrict__ out,
                   uint8_t* __restrict__ scale,
                   DTYPE_I const* __restrict__ input,
                   int32_t ori_cols,
                   int64_t oob_size,
                   int32_t const* __restrict__ num_rows)
{
    constexpr int32_t group_size       = 32;
    constexpr int32_t thread_data_size = 32;
    using vec_i                        = opus::vector_t<DTYPE_I, thread_data_size>;

    const int64_t rows = static_cast<int64_t>(*num_rows);
    const int32_t scale_n = ori_cols / group_size;
    const int64_t total_groups = rows * scale_n;
    const int64_t group_stride = static_cast<int64_t>(gridDim.x) * blockDim.x;

    auto buffer_o = opus::make_gmem<opus::fp8_t>(out, oob_size);

    for(int64_t group_id = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        group_id < total_groups;
        group_id += group_stride)
    {
        const int64_t row = group_id / scale_n;
        const int32_t col_group = static_cast<int32_t>(group_id % scale_n);
        const int64_t input_offset = row * ori_cols + col_group * group_size;

        const vec_i thread_data = *reinterpret_cast<vec_i const*>(input + input_offset);
        float abs_max = 1e-10f;
#pragma unroll
        for(int j = 0; j < thread_data_size; ++j)
        {
            abs_max = max(abs_max, abs(static_cast<float>(thread_data[j])));
        }

        constexpr aiter::MxDtype kMxDtype = aiter::MxDtype::FP8_E4M3;
        const float row_scale =
            aiter::fp_f32_to_e8m0_scale<aiter::kDefaultMxScaleRoundMode, kMxDtype>(
                abs_max);
        scale[group_id] =
            (__builtin_bit_cast(uint32_t, row_scale) >> 23) & 0b11111111;

        const float inv_scale = 1.0f / row_scale;
        store_vector<opus::fp8_t,
                     DTYPE_I,
                     thread_data_size,
                     RT,
                     false,
                     WARP_SIZE,
                     1,
                     opus::fp8_t>(
            buffer_o, thread_data, group_id * group_size, inv_scale);
    }
}

// Called only by the dedicated EP quantization operation.
// The shared quantizer retains its ordinary dispatch for every other call.
template <typename Input>
inline bool try_launch(opus::fp8_t* out,
                       uint8_t* scales,
                       const Input* input,
                       int rows,
                       int cols,
                       int64_t oob_size,
                       const int32_t* num_rows,
                       hipStream_t stream)
{
    if(num_rows == nullptr || rows < 8192 || !enabled())
        return false;
    constexpr int block_size = 256;
    const int max_grid = get_device_cu_num() * 2;
    const int64_t static_grid =
        (static_cast<int64_t>(rows) * (cols / 32) + block_size - 1) / block_size;
    const int grid = static_grid < max_grid ? static_grid : max_grid;
    quantize_live_rows<Input, block_size><<<grid, block_size, 0, stream>>>(
        out, scales, input, cols, oob_size, num_rows);
    return true;
}

} // namespace aiter::gfx950_ep_quant
