// SPDX-License-Identifier: MIT
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include "dispatch_utils.h"
#include "hip_reduce.h"
#include "py_itfs_common.h"
#include "warp_sort.h"
#include <ATen/hip/HIPContext.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <hip/hip_runtime.h>
#include <hipcub/hipcub.hpp>
#include <hipcub/util_type.hpp>
#include <torch/all.h>

void topk_plain(torch::Tensor& output, 
                  torch::Tensor& topk_ids,  
                  int topk_num) 
{
    printf("Before launch kernel xxxxx\n");

    // LAUNCH_KERNEL()
}

