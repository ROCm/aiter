// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

#include "aiter_stream.h"
#include "aiter_tensor.h"
#include "gfx1201_h3_ops.h"
#include "rocm_ops.hpp"

namespace {

int64_t address(const aiter_tensor_t& tensor) { return reinterpret_cast<int64_t>(tensor.ptr); }

int64_t current_stream() { return reinterpret_cast<int64_t>(aiter::getCurrentHIPStream()); }

} // namespace

void gfx1201_rms_modulate_hip(const aiter_tensor_t& x,
                              const aiter_tensor_t& weight,
                              const aiter_tensor_t& scale,
                              const aiter_tensor_t& shift,
                              const aiter_tensor_t& indices,
                              aiter_tensor_t& out,
                              bool indices_int64,
                              double eps)
{
    launch_gfx1201_rms_modulate(address(x), address(weight), address(scale), address(shift), address(indices),
                                indices_int64, address(out), x.size(0), scale.stride(0), shift.stride(0), eps,
                                current_stream());
}

void gfx1201_swiglu_hip(const aiter_tensor_t& in, aiter_tensor_t& out)
{
    launch_gfx1201_swiglu(address(in), address(out), in.size(0), in.size(1) / 2, current_stream());
}

void gfx1201_gated_residual_hip(const aiter_tensor_t& residual,
                                const aiter_tensor_t& projected,
                                const aiter_tensor_t& gate,
                                const aiter_tensor_t& indices,
                                aiter_tensor_t& out,
                                bool indices_int64,
                                bool vectorized)
{
    launch_gfx1201_gated_residual(address(residual), address(projected), address(gate), address(indices),
                                  indices_int64, address(out), residual.size(0), residual.size(1), gate.stride(0),
                                  vectorized, current_stream());
}

void gfx1201_vae_residual_rms_hip(aiter_tensor_t& hidden,
                                  int64_t addend,
                                  int64_t scale,
                                  int64_t weight,
                                  int64_t out,
                                  int64_t dim,
                                  double eps)
{
    launch_gfx1201_vae_residual_rms(address(hidden), addend, scale, weight, out,
                                    static_cast<int64_t>(hidden.numel()) / dim, dim, eps, current_stream());
}

void gfx1201_vae_qk_norm_rope_hip(aiter_tensor_t& query,
                                  aiter_tensor_t& key,
                                  const aiter_tensor_t& cos16,
                                  const aiter_tensor_t& sin16,
                                  int64_t heads,
                                  double eps)
{
    launch_gfx1201_vae_qk_norm_rope(address(query), address(key), address(cos16), address(sin16), query.size(0),
                                    heads, query.stride(0), key.stride(0), eps, current_stream());
}

void gfx1201_vae_swiglu_hip(const aiter_tensor_t& in, aiter_tensor_t& out)
{
    launch_gfx1201_vae_swiglu(address(in), address(out), in.size(0), in.size(1) / 2, in.stride(0),
                              current_stream());
}

PYBIND11_MODULE(AITER_EXTENSION_NAME, m)
{
    AITER_SET_STREAM_PYBIND;
    m.def("gfx1201_rms_modulate_hip", &gfx1201_rms_modulate_hip);
    m.def("gfx1201_swiglu_hip", &gfx1201_swiglu_hip);
    m.def("gfx1201_gated_residual_hip", &gfx1201_gated_residual_hip);
    m.def("gfx1201_vae_residual_rms_hip", &gfx1201_vae_residual_rms_hip);
    m.def("gfx1201_vae_qk_norm_rope_hip", &gfx1201_vae_qk_norm_rope_hip);
    m.def("gfx1201_vae_swiglu_hip", &gfx1201_vae_swiglu_hip);
}
