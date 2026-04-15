// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include "aiter_enum.h"
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>
#include <cstdint>

// ============================================================================
// AITER_DISPATCH — dtype dispatch macros (torch-free, replace AT_DISPATCH_*)
//
// Usage (same pattern as PyTorch, scalar_t is auto-defined):
//
//   AITER_DISPATCH_FLOATING(tensor->dtype(), "my_kernel", [&] {
//       kernel<scalar_t><<<grid, block, 0, stream>>>(data);
//   });
// ============================================================================

// ---------------------------------------------------------------------------
// Layer 1: per-type case blocks (building blocks)
// ---------------------------------------------------------------------------
#define AITER_DISPATCH_CASE_FP16(...)                    \
    case AITER_DTYPE_fp16: {                             \
        using scalar_t = __half;                         \
        __VA_ARGS__();                                   \
        break;                                           \
    }

#define AITER_DISPATCH_CASE_BF16(...)                    \
    case AITER_DTYPE_bf16: {                             \
        using scalar_t = hip_bfloat16;                   \
        __VA_ARGS__();                                   \
        break;                                           \
    }

#define AITER_DISPATCH_CASE_FP32(...)                    \
    case AITER_DTYPE_fp32: {                             \
        using scalar_t = float;                          \
        __VA_ARGS__();                                   \
        break;                                           \
    }

#define AITER_DISPATCH_CASE_FP8(...)                     \
    case AITER_DTYPE_fp8: {                              \
        using scalar_t = uint8_t;                        \
        __VA_ARGS__();                                   \
        break;                                           \
    }

#define AITER_DISPATCH_CASE_U8(...)                      \
    case AITER_DTYPE_u8: {                               \
        using scalar_t = uint8_t;                        \
        __VA_ARGS__();                                   \
        break;                                           \
    }

#define AITER_DISPATCH_CASE_I8(...)                      \
    case AITER_DTYPE_i8: {                               \
        using scalar_t = int8_t;                         \
        __VA_ARGS__();                                   \
        break;                                           \
    }

#define AITER_DISPATCH_CASE_I16(...)                     \
    case AITER_DTYPE_i16: {                              \
        using scalar_t = int16_t;                        \
        __VA_ARGS__();                                   \
        break;                                           \
    }

#define AITER_DISPATCH_CASE_I32(...)                     \
    case AITER_DTYPE_i32: {                              \
        using scalar_t = int32_t;                        \
        __VA_ARGS__();                                   \
        break;                                           \
    }

#define AITER_DISPATCH_CASE_I64(...)                     \
    case AITER_DTYPE_i64: {                              \
        using scalar_t = int64_t;                        \
        __VA_ARGS__();                                   \
        break;                                           \
    }

// ---------------------------------------------------------------------------
// Layer 2: switch framework
// ---------------------------------------------------------------------------
#define AITER_DISPATCH_SWITCH(DTYPE, NAME, ...)          \
    [&] {                                                \
        switch (DTYPE) {                                 \
            __VA_ARGS__                                  \
            default:                                     \
                AITER_CHECK(false, NAME,                 \
                    ": unsupported dtype ",              \
                    AiterDtype_to_str(DTYPE));           \
        }                                                \
    }()

// ---------------------------------------------------------------------------
// Layer 3: pre-composed combinations (convenience macros)
// ---------------------------------------------------------------------------

// fp16, bf16
// Temporary transition name; drop the "_xxx" suffix after migration is complete.
#define AITER_DISPATCH_CASE_FLOATING16_TYPES_xxx(...)   \
    AITER_DISPATCH_CASE_FP16(__VA_ARGS__)                \
    AITER_DISPATCH_CASE_BF16(__VA_ARGS__)

#define AITER_DISPATCH_FLOATING16_TYPES_xxx(DTYPE, NAME, ...) \
    AITER_DISPATCH_SWITCH(DTYPE, NAME,                         \
        AITER_DISPATCH_CASE_FLOATING16_TYPES_xxx(__VA_ARGS__))

// fp16, bf16, fp32
#define AITER_DISPATCH_CASE_FLOATING(...)                \
    AITER_DISPATCH_CASE_FP16(__VA_ARGS__)                \
    AITER_DISPATCH_CASE_BF16(__VA_ARGS__)                \
    AITER_DISPATCH_CASE_FP32(__VA_ARGS__)

#define AITER_DISPATCH_FLOATING(DTYPE, NAME, ...)        \
    AITER_DISPATCH_SWITCH(DTYPE, NAME,                   \
        AITER_DISPATCH_CASE_FLOATING(__VA_ARGS__))

// fp16, bf16, fp32, fp8
#define AITER_DISPATCH_CASE_FLOATING_AND_FP8(...)        \
    AITER_DISPATCH_CASE_FLOATING(__VA_ARGS__)             \
    AITER_DISPATCH_CASE_FP8(__VA_ARGS__)

#define AITER_DISPATCH_FLOATING_AND_FP8(DTYPE, NAME, ...) \
    AITER_DISPATCH_SWITCH(DTYPE, NAME,                   \
        AITER_DISPATCH_CASE_FLOATING_AND_FP8(__VA_ARGS__))

// fp16, bf16, fp32, u8
#define AITER_DISPATCH_CASE_FLOATING_AND_BYTE(...)       \
    AITER_DISPATCH_CASE_FLOATING(__VA_ARGS__)             \
    AITER_DISPATCH_CASE_U8(__VA_ARGS__)

#define AITER_DISPATCH_FLOATING_AND_BYTE(DTYPE, NAME, ...) \
    AITER_DISPATCH_SWITCH(DTYPE, NAME,                   \
        AITER_DISPATCH_CASE_FLOATING_AND_BYTE(__VA_ARGS__))

// i8, i16, i32, i64
#define AITER_DISPATCH_CASE_INTEGRAL(...)                \
    AITER_DISPATCH_CASE_I8(__VA_ARGS__)                  \
    AITER_DISPATCH_CASE_I16(__VA_ARGS__)                 \
    AITER_DISPATCH_CASE_I32(__VA_ARGS__)                 \
    AITER_DISPATCH_CASE_I64(__VA_ARGS__)

#define AITER_DISPATCH_INTEGRAL(DTYPE, NAME, ...)        \
    AITER_DISPATCH_SWITCH(DTYPE, NAME,                   \
        AITER_DISPATCH_CASE_INTEGRAL(__VA_ARGS__))

// ============================================================================
// AITER_DISPATCH_CASE_VEC_SIZE — vec_size dispatch (torch-free)
// ============================================================================

#define AITER_CASE_VEC_SIZE(VC, ...)    \
    case VC: {                           \
        constexpr int32_t VEC_SIZE = VC; \
        __VA_ARGS__                      \
        break;                           \
    }

#define AITER_DISPATCH_CASE_VEC_SIZE(vec_size, ...)                                    \
    switch(vec_size)                                                                    \
    {                                                                                   \
        AITER_CASE_VEC_SIZE(32, __VA_ARGS__)                                           \
        AITER_CASE_VEC_SIZE(16, __VA_ARGS__)                                           \
        AITER_CASE_VEC_SIZE(8, __VA_ARGS__)                                            \
        AITER_CASE_VEC_SIZE(4, __VA_ARGS__)                                            \
        AITER_CASE_VEC_SIZE(2, __VA_ARGS__)                                            \
        AITER_CASE_VEC_SIZE(1, __VA_ARGS__)                                            \
    default: AITER_CHECK(false, __func__, " doesn't support vec_size=", vec_size, "."); \
    }
