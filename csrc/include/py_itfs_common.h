/*
 * Copyright (c) 2024 Advanced Micro Devices, Inc.  All rights reserved.
 *
 * @Script: ck_py_itfs_common.cuh
 * @Author: valarLip
 * @Email: lingpeng.jin@amd.com
 * @Create At: 2024-12-02 12:53:20
 * @Last Modified By: valarLip
 * @Last Modified At: 2024-12-02 13:09:15
 * @Description: This is description.
 */

#pragma once
#include <torch/all.h>

// common utility functions
#define FOREACH_BUFFER_TORCH_TYPE_MAP(F) \
    F("fp32", torch::kFloat)             \
    F("fp16", torch::kHalf)              \
    F("bf16", torch::kBFloat16)          \
    F("int32", torch::kInt32)            \
    F("int8", torch::kInt8)         \
    F("fp8", c10::kFloat8_e4m3fnuz)

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