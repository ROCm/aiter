// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

#include "aiter_stream.h"
#include "aiter_tensor.h"
#include "gfx1201_norm_rope_prepare.h"
#include "rocm_ops.hpp"

void gfx1201_norm_rope_prepare_hip(const aiter_tensor_t& query,
                                   const aiter_tensor_t& key,
                                   const aiter_tensor_t& value,
                                   const aiter_tensor_t& cosine,
                                   const aiter_tensor_t& sine,
                                   aiter_tensor_t& maximum,
                                   const aiter_tensor_t& query_weight,
                                   const aiter_tensor_t& key_weight,
                                   aiter_tensor_t& query_out,
                                   aiter_tensor_t& key_out,
                                   aiter_tensor_t& value_out,
                                   aiter_tensor_t& query_scale,
                                   aiter_tensor_t& key_scale,
                                   aiter_tensor_t& value_scale,
                                   int64_t rows,
                                   int64_t padded_rows,
                                   double sm_scale,
                                   int64_t parts)
{
    launch_gfx1201_norm_rope_prepare(reinterpret_cast<int64_t>(query.ptr),
                                     reinterpret_cast<int64_t>(key.ptr),
                                     reinterpret_cast<int64_t>(value.ptr),
                                     reinterpret_cast<int64_t>(cosine.ptr),
                                     reinterpret_cast<int64_t>(sine.ptr),
                                     reinterpret_cast<int64_t>(maximum.ptr),
                                     reinterpret_cast<int64_t>(query_weight.ptr),
                                     reinterpret_cast<int64_t>(key_weight.ptr),
                                     reinterpret_cast<int64_t>(query_out.ptr),
                                     reinterpret_cast<int64_t>(key_out.ptr),
                                     reinterpret_cast<int64_t>(value_out.ptr),
                                     reinterpret_cast<int64_t>(query_scale.ptr),
                                     reinterpret_cast<int64_t>(key_scale.ptr),
                                     reinterpret_cast<int64_t>(value_scale.ptr),
                                     rows,
                                     padded_rows,
                                     sm_scale,
                                     reinterpret_cast<int64_t>(aiter::getCurrentHIPStream()),
                                     parts);
}

PYBIND11_MODULE(AITER_EXTENSION_NAME, m)
{
    AITER_SET_STREAM_PYBIND;
    m.def("gfx1201_norm_rope_prepare_hip", &gfx1201_norm_rope_prepare_hip);
}
