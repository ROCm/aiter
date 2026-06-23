// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

#include <torch/all.h>
#include <ATen/hip/HIPContext.h>
#include "py_itfs_common.h"
#include "mha_common.h"
#include "mha_bwd.h"
#include "torch/mha_bwd.h"

namespace aiter {
namespace torch_itfs {

std::vector<at::Tensor> fmha_v3_bwd(const at::Tensor &dout,         // [b, sq, hq, d_v]
                                    const at::Tensor &q,            // [b, sq, hq, d]
                                    const at::Tensor &k,            // [b, sk, hk, d]
                                    const at::Tensor &v,            // [b, sk, hk, d_v]
                                    const at::Tensor &out,          // [b, sq, hq, d_v]
                                    const at::Tensor &softmax_lse,  // [b, hq, sq]
                                    float p_dropout,
                                    float softmax_scale,
                                    bool is_causal,
                                    int window_size_left,
                                    int window_size_right,
                                    bool deterministic,
                                    bool is_v3_atomic_fp32,
                                    int how_v3_bf16_cvt,
                                    std::optional<at::Tensor> dq_,
                                    std::optional<at::Tensor> dk_,
                                    std::optional<at::Tensor> dv_,
                                    std::optional<const at::Tensor> alibi_slopes_, // [hq] or [b, hq]
                                    std::optional<const at::Tensor> rng_state_,
                                    std::optional<at::Generator> gen_)
{
    return detail::mha_bwd_impl(dout,
                                q,
                                k,
                                v,
                                out,
                                softmax_lse,
                                p_dropout,
                                softmax_scale,
                                is_causal,
                                window_size_left,
                                window_size_right,
                                deterministic,
                                is_v3_atomic_fp32,
                                how_v3_bf16_cvt,
                                dq_,
                                dk_,
                                dv_,
                                std::nullopt,
                                std::nullopt,
                                alibi_slopes_,
                                rng_state_,
                                gen_,
                                std::nullopt,
                                std::nullopt);
}

} // namespace torch_itfs
} // namespace aiter
