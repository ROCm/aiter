// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once
#include <torch/all.h>
#include "aiter_hip_common.h"
#if CK_TILE_USE_OCP_FP8
const auto torch_fp8 = at::ScalarType::Float8_e4m3fn;
#else
const auto torch_fp8 = at::ScalarType::Float8_e4m3fnuz;
#endif

// common utility functions
#define FOREACH_BUFFER_TORCH_TYPE_MAP(F) \
    F("fp32", torch::kFloat)             \
    F("fp16", torch::kHalf)              \
    F("bf16", torch::kBFloat16)          \
    F("int32", torch::kInt32)            \
    F("int8", torch::kInt8)              \
    F("fp8", torch_fp8)

inline std::string torchDTypeToStr(caffe2::TypeMeta dtype)
{
#define TYPE_CASE(type, torch_type) \
    case torch_type:                \
    {                               \
        return type;                \
    }

    switch (dtype.toScalarType())
    {
        FOREACH_BUFFER_TORCH_TYPE_MAP(TYPE_CASE);
    default:
        throw std::runtime_error("CKPyInterface: Unsupported data type " + std::to_string((int8_t)(dtype.toScalarType())));
    }

#undef TYPE_CASE
}