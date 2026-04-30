// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include "aiter_enum.h"
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>
#include <cstdint>

namespace aiter::dispatch_detail {

template <AiterDtype DTYPE>
struct cpp_type_from_aiter_dtype;

template <>
struct cpp_type_from_aiter_dtype<AITER_DTYPE_fp8> {
    using type = uint8_t;
};

template <>
struct cpp_type_from_aiter_dtype<AITER_DTYPE_fp8_e8m0> {
    using type = uint8_t;
};

template <>
struct cpp_type_from_aiter_dtype<AITER_DTYPE_fp16> {
    using type = __half;
};

template <>
struct cpp_type_from_aiter_dtype<AITER_DTYPE_bf16> {
    using type = hip_bfloat16;
};

template <>
struct cpp_type_from_aiter_dtype<AITER_DTYPE_fp32> {
    using type = float;
};

template <>
struct cpp_type_from_aiter_dtype<AITER_DTYPE_i4x2> {
    using type = uint8_t;
};

template <>
struct cpp_type_from_aiter_dtype<AITER_DTYPE_fp4x2> {
    using type = uint8_t;
};

template <>
struct cpp_type_from_aiter_dtype<AITER_DTYPE_u32> {
    using type = uint32_t;
};

template <>
struct cpp_type_from_aiter_dtype<AITER_DTYPE_i32> {
    using type = int32_t;
};

template <>
struct cpp_type_from_aiter_dtype<AITER_DTYPE_i16> {
    using type = int16_t;
};

template <>
struct cpp_type_from_aiter_dtype<AITER_DTYPE_i8> {
    using type = int8_t;
};

template <>
struct cpp_type_from_aiter_dtype<AITER_DTYPE_u8> {
    using type = uint8_t;
};

template <>
struct cpp_type_from_aiter_dtype<AITER_DTYPE_i64> {
    using type = int64_t;
};

template <>
struct cpp_type_from_aiter_dtype<AITER_DTYPE_u64> {
    using type = uint64_t;
};

template <AiterDtype DTYPE>
using cpp_type_from_aiter_dtype_t = typename cpp_type_from_aiter_dtype<DTYPE>::type;

} // namespace aiter::dispatch_detail

// ============================================================================
// _rmTorch dtype dispatch macros (torch-free replacements for AT_DISPATCH_*)
//
// Usage (same pattern as PyTorch, scalar_t is auto-defined):
//
//   AITER_DISPATCH_FLOATING16_TYPES_rmTorch(dtype, "my_kernel", [&] {
//       kernel<scalar_t><<<grid, block, 0, stream>>>(data);
//   });
//
//   VLLM_DISPATCH_FLOATING_TYPES_rmTorch(dtype, "my_kernel", [&] {
//       kernel<scalar_t><<<grid, block, 0, stream>>>(data);
//   });
// ============================================================================

#define AT_DISPATCH_SWITCH_rmTorch(TYPE, NAME, ...)                                  \
    [&] {                                                                        \
        const auto& the_type = (TYPE);                                           \
        [[maybe_unused]] constexpr const char* at_dispatch_name = (NAME);        \
        const AiterDtype _st = static_cast<AiterDtype>(the_type);                \
        switch (_st) {                                                           \
            __VA_ARGS__                                                          \
            default:                                                             \
                AITER_CHECK(false,                                               \
                    at_dispatch_name,                                            \
                    " not implemented for dtype ",                               \
                    AiterDtype_to_str(_st));                                     \
        }                                                                        \
    }()

#define AT_PRIVATE_CASE_TYPE_USING_HINT_rmTorch(enum_type, HINT, ...)                     \
    case enum_type: {                                                                  \
        using HINT [[maybe_unused]] =                                                  \
            ::aiter::dispatch_detail::cpp_type_from_aiter_dtype_t<enum_type>;         \
        [[maybe_unused]] constexpr auto SCALAR_TYPE = enum_type;                       \
        return __VA_ARGS__();                                                          \
    }

#define AT_DISPATCH_CASE_rmTorch(enum_type, ...) \
    AT_PRIVATE_CASE_TYPE_USING_HINT_rmTorch(enum_type, scalar_t, __VA_ARGS__)

// fp16, bf16
#define AITER_DISPATCH_CASE_FLOATING16_TYPES_rmTorch(...) \
    AT_DISPATCH_CASE_rmTorch(AITER_DTYPE_fp16, __VA_ARGS__) \
    AT_DISPATCH_CASE_rmTorch(AITER_DTYPE_bf16, __VA_ARGS__)

#define AITER_DISPATCH_FLOATING16_TYPES_rmTorch(DTYPE, NAME, ...) \
    AT_DISPATCH_SWITCH_rmTorch(                                   \
        DTYPE,                                                \
        NAME,                                                 \
        AITER_DISPATCH_CASE_FLOATING16_TYPES_rmTorch(__VA_ARGS__))

// ============================================================================
// _rmTorch replacements for dispatch_utils.h vec_size and VLLM_DISPATCH_* macros.
// ============================================================================

// --- vec_size dispatch (_rmTorch) ---

#define AITER_CASE_VEC_SIZE_rmTorch(VC, ...)    \
    case VC: {                                   \
        constexpr int32_t VEC_SIZE = VC;         \
        __VA_ARGS__                              \
        break;                                   \
    }

#define AITER_DISPATCH_CASE_VEC_SIZE_rmTorch(vec_size, ...)                                    \
    switch(vec_size)                                                                            \
    {                                                                                           \
        AITER_CASE_VEC_SIZE_rmTorch(32, __VA_ARGS__)                                           \
        AITER_CASE_VEC_SIZE_rmTorch(16, __VA_ARGS__)                                           \
        AITER_CASE_VEC_SIZE_rmTorch(8, __VA_ARGS__)                                            \
        AITER_CASE_VEC_SIZE_rmTorch(4, __VA_ARGS__)                                            \
        AITER_CASE_VEC_SIZE_rmTorch(2, __VA_ARGS__)                                            \
        AITER_CASE_VEC_SIZE_rmTorch(1, __VA_ARGS__)                                            \
    default: AITER_CHECK(false, __func__, " doesn't support vec_size=", vec_size, ".");         \
    }

// --- VLLM_DISPATCH_* (_rmTorch) ---

// fp32, fp16, bf16
#define VLLM_DISPATCH_CASE_FLOATING_TYPES_rmTorch(...)             \
    AT_DISPATCH_CASE_rmTorch(AITER_DTYPE_fp32, __VA_ARGS__)        \
    AT_DISPATCH_CASE_rmTorch(AITER_DTYPE_fp16, __VA_ARGS__)        \
    AT_DISPATCH_CASE_rmTorch(AITER_DTYPE_bf16, __VA_ARGS__)

#define VLLM_DISPATCH_FLOATING_TYPES_rmTorch(TYPE, NAME, ...)      \
    AT_DISPATCH_SWITCH_rmTorch(                                     \
        TYPE, NAME, VLLM_DISPATCH_CASE_FLOATING_TYPES_rmTorch(__VA_ARGS__))

// fp32, fp16, bf16, u8
#define VLLM_DISPATCH_CASE_FLOATING_AND_BYTE_TYPES_rmTorch(...)    \
    AT_DISPATCH_CASE_rmTorch(AITER_DTYPE_fp32, __VA_ARGS__)        \
    AT_DISPATCH_CASE_rmTorch(AITER_DTYPE_fp16, __VA_ARGS__)        \
    AT_DISPATCH_CASE_rmTorch(AITER_DTYPE_bf16, __VA_ARGS__)        \
    AT_DISPATCH_CASE_rmTorch(AITER_DTYPE_u8, __VA_ARGS__)

#define VLLM_DISPATCH_FLOATING_AND_BYTE_TYPES_rmTorch(TYPE, NAME, ...) \
    AT_DISPATCH_SWITCH_rmTorch(                                         \
        TYPE, NAME, VLLM_DISPATCH_CASE_FLOATING_AND_BYTE_TYPES_rmTorch(__VA_ARGS__))

// u8, i8, i16, i32, i64
#define VLLM_DISPATCH_CASE_INTEGRAL_TYPES_rmTorch(...)             \
    AT_DISPATCH_CASE_rmTorch(AITER_DTYPE_u8, __VA_ARGS__)          \
    AT_DISPATCH_CASE_rmTorch(AITER_DTYPE_i8, __VA_ARGS__)          \
    AT_DISPATCH_CASE_rmTorch(AITER_DTYPE_i16, __VA_ARGS__)         \
    AT_DISPATCH_CASE_rmTorch(AITER_DTYPE_i32, __VA_ARGS__)         \
    AT_DISPATCH_CASE_rmTorch(AITER_DTYPE_i64, __VA_ARGS__)

#define VLLM_DISPATCH_INTEGRAL_TYPES_rmTorch(TYPE, NAME, ...)      \
    AT_DISPATCH_SWITCH_rmTorch(                                     \
        TYPE, NAME, VLLM_DISPATCH_CASE_INTEGRAL_TYPES_rmTorch(__VA_ARGS__))

// ============================================================================
// KV cache dtype dispatch macros (_rmTorch)
//
// These use opus::* and vllm::Fp8KVCacheDataType types which must be visible
// at the point of macro expansion (not here).
// ============================================================================

#define DISPATCH_BY_KV_CACHE_DTYPE_OPUS_rmTorch(SRC_DTYPE, KV_DTYPE, FN)               \
    if(KV_DTYPE == "auto")                                                             \
    {                                                                                  \
        if(SRC_DTYPE == AITER_DTYPE_fp32)                                              \
        {                                                                              \
            FN(float, float, vllm::Fp8KVCacheDataType::kAuto);                         \
        }                                                                              \
        else if(SRC_DTYPE == AITER_DTYPE_fp16)                                         \
        {                                                                              \
            FN(opus::fp16_t, opus::fp16_t, vllm::Fp8KVCacheDataType::kAuto);           \
        }                                                                              \
        else if(SRC_DTYPE == AITER_DTYPE_bf16)                                         \
        {                                                                              \
            FN(opus::bf16_t, opus::bf16_t, vllm::Fp8KVCacheDataType::kAuto);           \
        }                                                                              \
        else                                                                           \
        {                                                                              \
            AITER_CHECK(false, "Unsupported input type of kv cache: ", SRC_DTYPE);     \
        }                                                                              \
    }                                                                                  \
    else                                                                               \
    {                                                                                  \
        if(KV_DTYPE == "fp8" || KV_DTYPE == "fp8_e4m3")                                \
        {                                                                              \
            if(SRC_DTYPE == AITER_DTYPE_fp32)                                          \
            {                                                                          \
                FN(float, opus::fp8_t, vllm::Fp8KVCacheDataType::kFp8E4M3);            \
            }                                                                          \
            else if(SRC_DTYPE == AITER_DTYPE_fp16)                                     \
            {                                                                          \
                FN(opus::fp16_t, opus::fp8_t, vllm::Fp8KVCacheDataType::kFp8E4M3);     \
            }                                                                          \
            else if(SRC_DTYPE == AITER_DTYPE_bf16)                                     \
            {                                                                          \
                FN(opus::bf16_t, opus::fp8_t, vllm::Fp8KVCacheDataType::kFp8E4M3);     \
            }                                                                          \
            else                                                                       \
            {                                                                          \
                AITER_CHECK(false, "Unsupported input type of kv cache: ", SRC_DTYPE); \
            }                                                                          \
        }                                                                              \
        else                                                                           \
        {                                                                              \
            AITER_CHECK(false, "Unsupported data type of kv cache: ", KV_DTYPE);       \
        }                                                                              \
    }

#define DISPATCH_BY_KV_CACHE_QUERY_DTYPE_OPUS_rmTorch(SRC_DTYPE, KV_DTYPE, QUERY_DTYPE, FN)        \
    if(KV_DTYPE == "auto" && QUERY_DTYPE == "auto")                                               \
    {                                                                                             \
        if(SRC_DTYPE == AITER_DTYPE_fp32)                                                         \
        {                                                                                         \
            FN(float,                                                                             \
               float,                                                                             \
               float,                                                                             \
               vllm::Fp8KVCacheDataType::kAuto,                                                   \
               vllm::Fp8KVCacheDataType::kAuto);                                                  \
        }                                                                                         \
        else if(SRC_DTYPE == AITER_DTYPE_fp16)                                                    \
        {                                                                                         \
            FN(opus::fp16_t,                                                                      \
               opus::fp16_t,                                                                      \
               opus::fp16_t,                                                                      \
               vllm::Fp8KVCacheDataType::kAuto,                                                   \
               vllm::Fp8KVCacheDataType::kAuto);                                                  \
        }                                                                                         \
        else if(SRC_DTYPE == AITER_DTYPE_bf16)                                                    \
        {                                                                                         \
            FN(opus::bf16_t,                                                                      \
               opus::bf16_t,                                                                      \
               opus::bf16_t,                                                                      \
               vllm::Fp8KVCacheDataType::kAuto,                                                   \
               vllm::Fp8KVCacheDataType::kAuto);                                                  \
        }                                                                                         \
        else                                                                                      \
        {                                                                                         \
            AITER_CHECK(false, "Unsupported input type of kv cache: ", SRC_DTYPE);                \
        }                                                                                         \
    }                                                                                             \
    else if((KV_DTYPE == "fp8" || KV_DTYPE == "fp8_e4m3") && (QUERY_DTYPE == "auto"))             \
    {                                                                                             \
        if(SRC_DTYPE == AITER_DTYPE_fp32)                                                         \
        {                                                                                         \
            FN(float,                                                                             \
               opus::fp8_t,                                                                       \
               float,                                                                             \
               vllm::Fp8KVCacheDataType::kFp8E4M3,                                                \
               vllm::Fp8KVCacheDataType::kAuto);                                                  \
        }                                                                                         \
        else if(SRC_DTYPE == AITER_DTYPE_fp16)                                                    \
        {                                                                                         \
            FN(opus::fp16_t,                                                                      \
               opus::fp8_t,                                                                       \
               opus::fp16_t,                                                                      \
               vllm::Fp8KVCacheDataType::kFp8E4M3,                                                \
               vllm::Fp8KVCacheDataType::kAuto);                                                  \
        }                                                                                         \
        else if(SRC_DTYPE == AITER_DTYPE_bf16)                                                    \
        {                                                                                         \
            FN(opus::bf16_t,                                                                      \
               opus::fp8_t,                                                                       \
               opus::bf16_t,                                                                      \
               vllm::Fp8KVCacheDataType::kFp8E4M3,                                                \
               vllm::Fp8KVCacheDataType::kAuto);                                                  \
        }                                                                                         \
        else                                                                                      \
        {                                                                                         \
            AITER_CHECK(false, "Unsupported input type of kv cache: ", SRC_DTYPE);                \
        }                                                                                         \
    }                                                                                             \
    else if((KV_DTYPE == "fp8" || KV_DTYPE == "fp8_e4m3") &&                                      \
            (QUERY_DTYPE == "fp8" || QUERY_DTYPE == "fp8_e4m3"))                                  \
    {                                                                                             \
        if(SRC_DTYPE == AITER_DTYPE_fp32)                                                         \
        {                                                                                         \
            FN(float,                                                                             \
               opus::fp8_t,                                                                       \
               opus::fp8_t,                                                                       \
               vllm::Fp8KVCacheDataType::kFp8E4M3,                                                \
               vllm::Fp8KVCacheDataType::kFp8E4M3);                                               \
        }                                                                                         \
        else if(SRC_DTYPE == AITER_DTYPE_fp16)                                                    \
        {                                                                                         \
            FN(opus::fp16_t,                                                                      \
               opus::fp8_t,                                                                       \
               opus::fp8_t,                                                                       \
               vllm::Fp8KVCacheDataType::kFp8E4M3,                                                \
               vllm::Fp8KVCacheDataType::kFp8E4M3);                                               \
        }                                                                                         \
        else if(SRC_DTYPE == AITER_DTYPE_bf16)                                                    \
        {                                                                                         \
            FN(opus::bf16_t,                                                                      \
               opus::fp8_t,                                                                       \
               opus::fp8_t,                                                                       \
               vllm::Fp8KVCacheDataType::kFp8E4M3,                                                \
               vllm::Fp8KVCacheDataType::kFp8E4M3);                                               \
        }                                                                                         \
        else                                                                                      \
        {                                                                                         \
            AITER_CHECK(false, "Unsupported input type of kv cache: ", SRC_DTYPE);                \
        }                                                                                         \
    }                                                                                             \
    else                                                                                          \
    {                                                                                             \
        AITER_CHECK(                                                                              \
            false, "Unsupported data type of kv cache: ", KV_DTYPE, "Query type: ", QUERY_DTYPE); \
    }
