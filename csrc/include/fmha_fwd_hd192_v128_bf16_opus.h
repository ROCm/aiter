// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// OPUS-based GQA/MHA flash-attention with asymmetric head dims (D_QK=192, D_V=128)
// for gfx950. Kernel-only header (no public torch API of its own — the shared opus
// forward entry point in `fmha_fwd_bf16_opus.h` dispatches to this kernel by head dim).
//
// Single-header, IMPL-guarded (mirrors fmha_fwd_hd128_bf16_opus.h):
//   * kernel args / traits + device kernel template live inside the
//     `FMHA_FWD_HD192_V128_BF16_OPUS_IMPL` guard. On the gfx950 device pass the real
//     kernel template is pulled in; otherwise an empty stub satisfies `__device_stub__`.
#pragma once

#ifdef FMHA_FWD_HD192_V128_BF16_OPUS_IMPL
// opus_gqa_d192_traits / opus_gqa_d192_kargs / ceil_div / bf16_t / OPT_* / HEADTAIL_MIN_WG.
#include "fmha_fwd_hd192_v128_bf16_opus_defs.h"

// Device kernel template — declared here, defined on the gfx950 device pass.
template <class Traits>
__global__ void gqa_d192_v128_kernel(opus_gqa_d192_kargs kargs);

#if !defined(__HIP_DEVICE_COMPILE__) || !defined(__gfx950__)
template <class Traits>
__global__ void gqa_d192_v128_kernel(opus_gqa_d192_kargs) {}
#else
// Pulls in opus + the full asymmetric head-dim kernel template (16-stage software
// pipeline, stagger specialization, causal head/tail merge, group/varlen, arbitrary
// seqlen via OOB buffer bounds + post-QK -inf masking, dwordx4 packed O store).
#include "fmha_fwd_hd192_v128_bf16_opus_kernel.hpp"
#endif

#endif // FMHA_FWD_HD192_V128_BF16_OPUS_IMPL
