// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// OPUS-based GQA flash-attention (head dim D=128) for gfx950, sliding-window form of the
// eight-wave 32-row Q tile kernel. Same traits, kargs and launch geometry as
// fmha_fwd_hd128_bf16_opus.h; launch_d128 takes it for the LOCAL instances.
#pragma once
#include "aiter_tensor.h"

#ifdef FMHA_FWD_HD128_BF16_OPUS_SWA_IMPL

#include "fmha_fwd_hd128_bf16_opus_defs.h"

template <class Traits>
__global__ void gqa_d128_swa_kernel(opus_gqa_kargs kargs);

#if !defined(__HIP_DEVICE_COMPILE__) || !defined(__gfx950__)
template <class Traits>
__global__ void gqa_d128_swa_kernel(opus_gqa_kargs) {}
#else
#include "fmha_fwd_hd128_bf16_opus_swa_kernel.hpp"
#endif

#endif // FMHA_FWD_HD128_BF16_OPUS_SWA_IMPL
