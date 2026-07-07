// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include "../opus_moe_common.cuh"

#include <cstdint>

#ifdef __HIP_DEVICE_COMPILE__

inline __device__ uint32_t opus_moe_gfx950_cvt_pk_bf16_f32(float lo, float hi)
{
    uint32_t packed;
    asm volatile("v_cvt_pk_bf16_f32 %0, %1, %2"
                 : "=v"(packed)
                 : "v"(lo), "v"(hi));
    return packed;
}

inline __device__ hip_bfloat16 opus_moe_gfx950_bf16_from_bits(uint16_t bits)
{
#if defined(__HIPCC_RTC__) || defined(HIP_MINIMAL_HPP)
    return __builtin_bit_cast(hip_bfloat16, bits);
#else
    hip_bfloat16 value;
    value.data = bits;
    return value;
#endif
}

inline __device__ uint16_t opus_moe_gfx950_bf16_to_bits(hip_bfloat16 value)
{
#if defined(__HIPCC_RTC__) || defined(HIP_MINIMAL_HPP)
    return __builtin_bit_cast(uint16_t, value);
#else
    return static_cast<uint16_t>(value.data);
#endif
}

inline __device__ float opus_moe_gfx950_e8m0_to_float_scale(uint32_t e8m0)
{
    return __builtin_bit_cast(float, e8m0 << 23);
}

inline __device__ hip_bfloat16 opus_moe_gfx950_cvt_bf16_f32(float value)
{
    return opus_moe_gfx950_bf16_from_bits(
        static_cast<uint16_t>(opus_moe_gfx950_cvt_pk_bf16_f32(value, value)));
}

#endif
