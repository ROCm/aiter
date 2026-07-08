// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// PA decode kernel instantiation (A16W8 / FP8 KV / GQA8 / head=128 / block=16).
// Host TU: stub declaration; device TU: opus_pa/kernels sp3 body.
#include <hip/hip_runtime.h>

#include "opus_pa/pa_decode_defs.h"

#ifndef __HIP_DEVICE_COMPILE__
template<typename Traits>
__global__ void pa_decode_kernel(pa_decode_kargs kargs)
{
}
template __global__ void pa_decode_kernel<pa_default_traits>(pa_decode_kargs);
#else
#include "opus_pa/pa_decode_kernel.hpp"
template __global__ void pa_decode_kernel<pa_default_traits>(pa_decode_kargs);
#endif
