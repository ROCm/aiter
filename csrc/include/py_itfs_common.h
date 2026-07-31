// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

#pragma once
#include "aiter_hip_common.h"
#include "aiter_tensor.h"
#include <torch/all.h>

bool static isGPUArch(const std::vector<std::string>& archs)
{
    hipDeviceProp_t props;

    hipGetDeviceProperties(&props, 0);

    std::string device_arch = props.gcnArchName;
    for(std::string arch : archs)
    {
        size_t substring = device_arch.find(arch);
        if(substring != std::string::npos)
        {
            return true;
        }
    }
    return false;
}

#ifndef AITER_USE_OCP_FP8
#if defined(__HIP_DEVICE_COMPILE__)
#if defined(__gfx950__) || defined(__gfx12__)
#define AITER_USE_OCP_FP8 1
#else
#define AITER_USE_OCP_FP8 0
#endif
#else
#define AITER_USE_OCP_FP8 0
#endif
#endif

#ifdef __HIP_DEVICE_COMPILE__
#if AITER_USE_OCP_FP8
const constexpr auto torch_fp8 = at::ScalarType::Float8_e4m3fn;
#else
const constexpr auto torch_fp8 = at::ScalarType::Float8_e4m3fnuz;
#endif
#else
inline at::ScalarType get_torch_fp8()
{
    static const auto value =
        isGPUArch({"gfx94"}) ? at::ScalarType::Float8_e4m3fnuz : at::ScalarType::Float8_e4m3fn;
    return value;
}
#define torch_fp8 get_torch_fp8()
#endif

#ifdef TORCH_Float4_e2m1fn_x2
const constexpr auto torch_fp4x2 = torch::kFloat4_e2m1fn_x2;
#else
const constexpr auto torch_fp4x2 = torch::kUInt8;
#endif

// clang-format off
#if ENABLE_CK
template <typename T> struct t2ck;
template <> struct t2ck<float> { using type = ck_tile::fp32_t; };
template <> struct t2ck<c10::Half> { using type = ck_tile::fp16_t; };
template <> struct t2ck<c10::BFloat16> { using type = ck_tile::bf16_t; };
template <> struct t2ck<int32_t> { using type = ck_tile::index_t; };
template <> struct t2ck<int8_t> { using type = ck_tile::int8_t; };
#endif
// clang-format on

// common utility functions
#define FOREACH_BUFFER_TORCH_TYPE_MAP(F) \
    F("fp32", torch::kFloat)             \
    F("fp16", torch::kHalf)              \
    F("bf16", torch::kBFloat16)          \
    F("int32", torch::kInt32)            \
    F("int8", torch::kInt8)              \
    F("fp8", torch::kFloat8_e4m3fnuz)    \
    F("fp8", torch::kFloat8_e4m3fn)

inline std::string torchDTypeToStr(caffe2::TypeMeta dtype)
{
#define TYPE_CASE(type, torch_type) \
    case torch_type: {              \
        return type;                \
    }

    switch(dtype.toScalarType())
    {
        FOREACH_BUFFER_TORCH_TYPE_MAP(TYPE_CASE);
    default:
        throw std::runtime_error("CKPyInterface: Unsupported data type " +
                                 std::to_string((int8_t)(dtype.toScalarType())));
    }

#undef TYPE_CASE
}

// Shallow-converts a torch::Tensor to the aiter_tensor_t POD struct used by the
// torch-free kernel layer.  The caller retains ownership of the underlying
// storage; aiter_tensor_t holds a raw pointer and must not outlive the source
// tensor.
//
// Kept in a namespace rather than at file scope because csrc/kernels/
// fused_ar_mhc_post.cu also includes this header and defines its own, stricter
// make_aiter_tensor (fp32/fp16/bf16 only, throws otherwise) in an anonymous
// namespace inside `namespace aiter`.  Qualifying keeps the two unambiguous.
namespace aiter_pybind {

static inline aiter_tensor_t make_aiter_tensor(const torch::Tensor& t)
{
    TORCH_CHECK(t.is_cuda(), "aiter: tensor must be on a CUDA device");
    TORCH_CHECK(t.dim() <= 8, "aiter: tensor rank exceeds aiter_tensor_t maximum of 8");
    aiter_tensor_t at{};
    at.ptr      = t.data_ptr();
    at.numel_   = static_cast<size_t>(t.numel());
    at.ndim     = t.dim();
    for(int i = 0; i < t.dim(); ++i)
    {
        at.shape[i]   = t.size(i);
        at.strides[i] = t.stride(i);
    }
    at.device_id = static_cast<int>(t.device().index());
    switch(t.scalar_type())
    {
    case at::ScalarType::Float:    at.dtype_ = AITER_DTYPE_fp32;  break;
    case at::ScalarType::Half:     at.dtype_ = AITER_DTYPE_fp16;  break;
    case at::ScalarType::BFloat16: at.dtype_ = AITER_DTYPE_bf16;  break;
    case at::ScalarType::Byte:     at.dtype_ = AITER_DTYPE_u8;    break;
    case at::ScalarType::Char:     at.dtype_ = AITER_DTYPE_i8;    break;
    default:
        // fp4x2 / fp8 variants have no at::ScalarType; callers pass opaque
        // uint8-typed buffers whose semantic dtype is implicit in the op.
        at.dtype_ = AITER_DTYPE_fp4x2;
        break;
    }
    return at;
}

} // namespace aiter_pybind
