// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// OPUS-based GQA flash-attention (head dim D=128) for gfx950.
// Single-header, IMPL-guarded (mirrors pa_sparse_prefill_opus.h):
//   * Public API (always visible).
//   * Kernel args / traits + device kernel template inside the
//     `FMHA_FWD_HD128_BF16_OPUS_IMPL` guard. On the gfx950 device pass the real kernel template
//     is pulled in; otherwise an empty stub satisfies `__device_stub__` symbols.
#pragma once
#include "aiter_tensor.h"

// Kernel-only header for the symmetric D=128 OPUS forward kernel. The public torch
// API now lives in the shared `fmha_fwd_bf16_opus.h` (`fmha_fwd_bf16_opus_fwd`), which
// dispatches to this kernel by head dim. This header just exposes the device kernel
// template (IMPL-guarded) so the shared host translation unit can launch it.

#ifdef FMHA_FWD_HD128_BF16_OPUS_IMPL
// Implementation section - only compiled in the .cu translation unit.

// opus_gqa_traits / opus_gqa_kargs / ceil_div / bf16_t.
#include "fmha_fwd_hd128_bf16_opus_defs.h"

// Device kernel template — declared here, defined on the gfx950 device pass.
template <class Traits>
__global__ void gqa_d128_kernel(opus_gqa_kargs kargs);

#if !defined(__HIP_DEVICE_COMPILE__) || !defined(__gfx950__)
template <class Traits>
__global__ void gqa_d128_kernel(opus_gqa_kargs) {}
#else
// Pulls in opus + the full kernel template (le2 fast path, runtime OddTail
// dispatch, arbitrary seqlen via OOB buffer bounds + post-QK -inf masking).
#include "fmha_fwd_hd128_bf16_opus_kernel.hpp"
#endif

#endif // FMHA_FWD_HD128_BF16_OPUS_IMPL
