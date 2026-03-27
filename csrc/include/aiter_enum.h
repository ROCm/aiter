#pragma once
// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
// Single source of truth: aiter/ops/enum.py parses enums from this file
#include <string>


enum class ActivationType : int
{
    No     = -1,
    Silu   = 0,
    Gelu   = 1,
    Swiglu = 2,
};
enum class QuantType : int
{
    No,
    per_Tensor,
    per_Token,
    per_1x32,
    per_1x128,
    per_128x128,
    per_256x128,
    per_1024x128,
    lqq_1x64,
};
typedef enum {
    AITER_DTYPE_fp8,
    AITER_DTYPE_fp8_e8m0,
    AITER_DTYPE_fp16,
    AITER_DTYPE_bf16,
    AITER_DTYPE_fp32,
    AITER_DTYPE_i4x2,
    AITER_DTYPE_fp4x2,
    AITER_DTYPE_u32,
    AITER_DTYPE_i32,
    AITER_DTYPE_i16,
    AITER_DTYPE_i8,
    AITER_DTYPE_u8,
    AITER_DTYPE_i64,
    AITER_DTYPE_u64,
} AiterDtype;

static inline size_t AiterDtype_element_size(AiterDtype dtype)
{
    switch (dtype) {
    case AITER_DTYPE_fp8:
    case AITER_DTYPE_fp8_e8m0:
    case AITER_DTYPE_i4x2:
    case AITER_DTYPE_fp4x2:
    case AITER_DTYPE_i8:
    case AITER_DTYPE_u8:       return 1;
    case AITER_DTYPE_fp16:
    case AITER_DTYPE_bf16:
    case AITER_DTYPE_i16:      return 2;
    case AITER_DTYPE_fp32:
    case AITER_DTYPE_u32:
    case AITER_DTYPE_i32:      return 4;
    case AITER_DTYPE_i64:
    case AITER_DTYPE_u64:      return 8;
    default:                   return 0;
    }
}

static inline std::string AiterDtype_to_str(int dtype)
{
    switch (dtype) {
    case AITER_DTYPE_fp8:      return "fp8";
    case AITER_DTYPE_fp8_e8m0: return "fp8_e8m0";
    case AITER_DTYPE_fp16:     return "fp16";
    case AITER_DTYPE_bf16:     return "bf16";
    case AITER_DTYPE_fp32:     return "fp32";
    case AITER_DTYPE_i4x2:    return "i4x2";
    case AITER_DTYPE_fp4x2:   return "fp4x2";
    case AITER_DTYPE_u32:     return "u32";
    case AITER_DTYPE_i32:     return "i32";
    case AITER_DTYPE_i16:     return "i16";
    case AITER_DTYPE_i8:      return "i8";
    case AITER_DTYPE_u8:      return "u8";
    case AITER_DTYPE_i64:     return "i64";
    case AITER_DTYPE_u64:     return "u64";
    default:                   return "unknown";
    }
}
