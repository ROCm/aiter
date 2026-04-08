// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include "aiter_enum.h"
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>
#include <cstdint>

// ============================================================================
// AITER_DISPATCH — dtype dispatch macros (replace AT_DISPATCH_*)
//
// Usage (same pattern as PyTorch, scalar_t is auto-defined):
//
//   AITER_DISPATCH_FLOATING(tensor->dtype(), "my_kernel", [&] {
//       kernel<scalar_t><<<grid, block, 0, stream>>>(data);
//   });
//
//   AITER_DISPATCH_REDUCED_FLOATING(tensor->dtype(), "my_kernel", [&] {
//       kernel<scalar_t><<<grid, block, 0, stream>>>(data);
//   });
// ============================================================================

// fp16, bf16, fp32
#define AITER_DISPATCH_FLOATING(DTYPE, NAME, ...)        \
    [&] {                                                \
        switch (DTYPE) {                                 \
            case AITER_DTYPE_fp16: {                     \
                using scalar_t = __half;                 \
                __VA_ARGS__();                           \
                break;                                   \
            }                                            \
            case AITER_DTYPE_bf16: {                     \
                using scalar_t = hip_bfloat16;           \
                __VA_ARGS__();                           \
                break;                                   \
            }                                            \
            case AITER_DTYPE_fp32: {                     \
                using scalar_t = float;                  \
                __VA_ARGS__();                           \
                break;                                   \
            }                                            \
            default:                                     \
                AITER_CHECK(false, NAME,                 \
                    ": unsupported dtype ",              \
                    AiterDtype_to_str(DTYPE));           \
        }                                                \
    }()

// fp16, bf16
#define AITER_DISPATCH_REDUCED_FLOATING(DTYPE, NAME, ...) \
    [&] {                                                \
        switch (DTYPE) {                                 \
            case AITER_DTYPE_fp16: {                     \
                using scalar_t = __half;                 \
                __VA_ARGS__();                           \
                break;                                   \
            }                                            \
            case AITER_DTYPE_bf16: {                     \
                using scalar_t = hip_bfloat16;           \
                __VA_ARGS__();                           \
                break;                                   \
            }                                            \
            default:                                     \
                AITER_CHECK(false, NAME,                 \
                    ": unsupported dtype ",              \
                    AiterDtype_to_str(DTYPE));           \
        }                                                \
    }()

// fp16, bf16, fp32 + fp8
#define AITER_DISPATCH_FLOATING_AND_FP8(DTYPE, NAME, ...) \
    [&] {                                                \
        switch (DTYPE) {                                 \
            case AITER_DTYPE_fp8: {                      \
                using scalar_t = uint8_t;                \
                __VA_ARGS__();                           \
                break;                                   \
            }                                            \
            case AITER_DTYPE_fp16: {                     \
                using scalar_t = __half;                 \
                __VA_ARGS__();                           \
                break;                                   \
            }                                            \
            case AITER_DTYPE_bf16: {                     \
                using scalar_t = hip_bfloat16;           \
                __VA_ARGS__();                           \
                break;                                   \
            }                                            \
            case AITER_DTYPE_fp32: {                     \
                using scalar_t = float;                  \
                __VA_ARGS__();                           \
                break;                                   \
            }                                            \
            default:                                     \
                AITER_CHECK(false, NAME,                 \
                    ": unsupported dtype ",              \
                    AiterDtype_to_str(DTYPE));           \
        }                                                \
    }()

// fp16, bf16, fp32 + u8 (byte)
#define AITER_DISPATCH_FLOATING_AND_BYTE(DTYPE, NAME, ...) \
    [&] {                                                \
        switch (DTYPE) {                                 \
            case AITER_DTYPE_fp16: {                     \
                using scalar_t = __half;                 \
                __VA_ARGS__();                           \
                break;                                   \
            }                                            \
            case AITER_DTYPE_bf16: {                     \
                using scalar_t = hip_bfloat16;           \
                __VA_ARGS__();                           \
                break;                                   \
            }                                            \
            case AITER_DTYPE_fp32: {                     \
                using scalar_t = float;                  \
                __VA_ARGS__();                           \
                break;                                   \
            }                                            \
            case AITER_DTYPE_u8: {                       \
                using scalar_t = uint8_t;                \
                __VA_ARGS__();                           \
                break;                                   \
            }                                            \
            default:                                     \
                AITER_CHECK(false, NAME,                 \
                    ": unsupported dtype ",              \
                    AiterDtype_to_str(DTYPE));           \
        }                                                \
    }()

// i8, i16, i32, i64
#define AITER_DISPATCH_INTEGRAL(DTYPE, NAME, ...)        \
    [&] {                                                \
        switch (DTYPE) {                                 \
            case AITER_DTYPE_i8: {                       \
                using scalar_t = int8_t;                 \
                __VA_ARGS__();                           \
                break;                                   \
            }                                            \
            case AITER_DTYPE_i16: {                      \
                using scalar_t = int16_t;                \
                __VA_ARGS__();                           \
                break;                                   \
            }                                            \
            case AITER_DTYPE_i32: {                      \
                using scalar_t = int32_t;                \
                __VA_ARGS__();                           \
                break;                                   \
            }                                            \
            case AITER_DTYPE_i64: {                      \
                using scalar_t = int64_t;                \
                __VA_ARGS__();                           \
                break;                                   \
            }                                            \
            default:                                     \
                AITER_CHECK(false, NAME,                 \
                    ": unsupported dtype ",              \
                    AiterDtype_to_str(DTYPE));           \
        }                                                \
    }()

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
