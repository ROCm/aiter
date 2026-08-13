// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include <cstdint>
#include <hip/hip_bfloat16.h>

#ifdef __HIP_DEVICE_COMPILE__

inline __device__ uint32_t opus_moe_gfx950_cvt_pk_bf16_f32(float lo, float hi)
{
    uint32_t packed;
    asm volatile("v_cvt_pk_bf16_f32 %0, %1, %2"
                 : "=v"(packed)
                 : "v"(lo), "v"(hi));
    return packed;
}

inline __device__ hip_bfloat16 opus_moe_gfx950_cvt_bf16_f32(float value)
{
    return __builtin_bit_cast(
        hip_bfloat16,
        static_cast<uint16_t>(opus_moe_gfx950_cvt_pk_bf16_f32(value, value)));
}

#endif
