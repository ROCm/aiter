// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include "opus/opus_vec_io.hpp"
#include "hip_reduce.h"
// todo: remove this to use aiterTensor dtype
#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <hip/hip_bf16.h>

namespace aiter {
using namespace opus;

// todo: edit this to use aiterTensor dtype
template <typename T>
struct t2opus;
template <>
struct t2opus<float>
{
    using type = float;
};
template <>
struct t2opus<c10::Half>
{
    using type = opus::fp16_t;
};
template <>
struct t2opus<c10::BFloat16>
{
    using type = opus::bf16_t;
};
template <>
struct t2opus<int32_t>
{
    using type = int32_t;
};
template <>
struct t2opus<int8_t>
{
    using type = opus::i8_t;
};

// HIP native type -> opus type mapping
template <typename T> struct hip2opus;
template <> struct hip2opus<float>         { using type = opus::fp32_t; };
template <> struct hip2opus<__half>        { using type = opus::fp16_t; };
template <> struct hip2opus<hip_bfloat16>  { using type = opus::bf16_t; };
template <> struct hip2opus<uint8_t>       { using type = opus::fp8_t; };
template <> struct hip2opus<int8_t>        { using type = opus::i8_t; };
template <> struct hip2opus<int32_t>       { using type = int32_t; };

} // namespace aiter
