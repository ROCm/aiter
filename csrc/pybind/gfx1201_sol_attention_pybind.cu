// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

#include "aiter_stream.h"
#include "aiter_tensor.h"
#include "gfx1201_sol_attention.h"
#include "rocm_ops.hpp"

namespace {

int64_t address(const aiter_tensor_t& tensor) { return reinterpret_cast<int64_t>(tensor.ptr); }

int64_t current_stream() { return reinterpret_cast<int64_t>(aiter::getCurrentHIPStream()); }

} // namespace

// q/k: INT8 [1, Sp, H, 128]; v: FP8 [1, H, 128, Sp] (prepare_sage layout); the rest are SolPlan buffers.
void gfx1201_sol_route_hip(const aiter_tensor_t& q,
                           const aiter_tensor_t& q_scale,
                           const aiter_tensor_t& k,
                           const aiter_tensor_t& k_scale,
                           const aiter_tensor_t& v,
                           const aiter_tensor_t& v_scale,
                           aiter_tensor_t& kbar,
                           aiter_tensor_t& vsum,
                           aiter_tensor_t& kb,
                           aiter_tensor_t& kb_scale,
                           aiter_tensor_t& vb,
                           aiter_tensor_t& vb_scale,
                           aiter_tensor_t& mu,
                           aiter_tensor_t& var,
                           aiter_tensor_t& list,
                           aiter_tensor_t& count,
                           aiter_tensor_t& mask,
                           int64_t valid,
                           int64_t prefix,
                           double tau)
{
    launch_gfx1201_sol_route(address(q), address(q_scale), address(k), address(k_scale), address(v),
                             address(v_scale), address(kbar), address(vsum), address(kb), address(kb_scale),
                             address(vb), address(vb_scale), address(mu), address(var), address(list),
                             address(count), address(mask), q.size(1), valid, q.size(2), prefix, tau,
                             current_stream());
}

void gfx1201_sol_phase_a_hip(const aiter_tensor_t& q,
                             const aiter_tensor_t& q_scale,
                             const aiter_tensor_t& v_scale,
                             const aiter_tensor_t& kb,
                             const aiter_tensor_t& vb,
                             const aiter_tensor_t& kb_scale,
                             const aiter_tensor_t& vb_scale,
                             const aiter_tensor_t& cnt,
                             const aiter_tensor_t& mask,
                             aiter_tensor_t& state,
                             int64_t valid,
                             int64_t nkb,
                             int64_t nkbp,
                             int64_t n_wg)
{
    launch_gfx1201_sol_phase_a(address(q), address(q_scale), address(v_scale), address(kb), address(vb),
                               address(kb_scale), address(vb_scale), address(cnt), address(mask), address(state),
                               q.size(1), valid, q.size(2), nkb, nkbp, n_wg, current_stream());
}

PYBIND11_MODULE(AITER_EXTENSION_NAME, m)
{
    AITER_SET_STREAM_PYBIND;
    m.def("gfx1201_sol_route_hip", &gfx1201_sol_route_hip);
    m.def("gfx1201_sol_phase_a_hip", &gfx1201_sol_phase_a_hip);
}
