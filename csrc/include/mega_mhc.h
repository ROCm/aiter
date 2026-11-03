// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once
#include "aiter_tensor.h"

namespace aiter {
// Shared binding signature for the tuned and portable Mega-MHC modules.
// All modules use the three-operand pack in mega_mhc.py and 32 counter words
// per tile. Generic/large partials are [split_k, ceil(T/16), 16, 32]; distributed
// variants use compact [split_k, T, 32] partials in the same backing storage.
// Counters start at zero. Distributed kernels retain generation tags between
// calls; other variants reset their arrival counts to zero. Python preparation
// pairs each kernel with its matching pack, partial layout and counter protocol.
// Scratch may be reused on the same stream, never concurrently across streams.
// MXFP8 instantiations also write quant_out (E4M3) and quant_scale (uint8 E8M0
// bytes); BF16 instantiations ignore them. The rounded BF16 layer_input is kept.
// variant indexes the tuning table; mxfp8 selects the output instantiation.
void mega_mhc_bf16(aiter_tensor_t& residual_out,
                   aiter_tensor_t& post_out,
                   aiter_tensor_t& comb_out,
                   aiter_tensor_t& layer_input,
                   aiter_tensor_t& quant_out,
                   aiter_tensor_t& quant_scale,
                   aiter_tensor_t& next_pre,
                   aiter_tensor_t& residual,
                   aiter_tensor_t& fn_packed,
                   aiter_tensor_t& hc_scale,
                   aiter_tensor_t& hc_base,
                   aiter_tensor_t& pre_mix,
                   aiter_tensor_t& sublayer_out,
                   aiter_tensor_t& post_mix,
                   aiter_tensor_t& comb_mix,
                   aiter_tensor_t& norm_weight,
                   aiter_tensor_t& partials,
                   aiter_tensor_t& counters,
                   float rms_eps,
                   float hc_pre_eps,
                   float hc_sinkhorn_eps,
                   float hc_post_mult,
                   float norm_eps,
                   int sinkhorn_repeat,
                   int split_k,
                   bool has_post,
                   bool identity_pre,
                   int variant,
                   bool mxfp8);
} // namespace aiter
