// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

#include "aiter_stream.h"
#include "aiter_tensor.h"
#include "gfx1201_sage_attention.h"
#include "rocm_ops.hpp"

void gfx1201_sage_attention_hip(const aiter_tensor_t& q_int8,
                                const aiter_tensor_t& k_int8,
                                const aiter_tensor_t& v_fp8,
                                const aiter_tensor_t& q_scale,
                                const aiter_tensor_t& k_scale,
                                const aiter_tensor_t& v_scale,
                                aiter_tensor_t& out,
                                int64_t batch_size,
                                int64_t padded_seq_len,
                                int64_t valid_seq_len,
                                int64_t num_heads)
{
    launch_gfx1201_sage_attention(reinterpret_cast<int64_t>(q_int8.ptr),
                                  reinterpret_cast<int64_t>(k_int8.ptr),
                                  reinterpret_cast<int64_t>(v_fp8.ptr),
                                  reinterpret_cast<int64_t>(q_scale.ptr),
                                  reinterpret_cast<int64_t>(k_scale.ptr),
                                  reinterpret_cast<int64_t>(v_scale.ptr),
                                  reinterpret_cast<int64_t>(out.ptr),
                                  batch_size,
                                  padded_seq_len,
                                  valid_seq_len,
                                  num_heads,
                                  reinterpret_cast<int64_t>(aiter::getCurrentHIPStream()));
}

PYBIND11_MODULE(AITER_EXTENSION_NAME, m)
{
    AITER_SET_STREAM_PYBIND;
    m.def("gfx1201_sage_attention_hip", &gfx1201_sage_attention_hip);
}
