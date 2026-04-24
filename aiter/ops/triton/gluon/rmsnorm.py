# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import torch
import triton
from aiter.ops.triton.utils.device_info import get_num_sms
from aiter.ops.triton._gluon_kernels.normalization.rmsnorm import _gluon_rms_norm_kernel


def gluon_rms_norm_kernel(
    input: torch.Tensor,
    weights: torch.Tensor,
    epsilon: float,
):
    ROW, COL = input.shape
    output = torch.empty_like(input, device=input.device)
    rsigma = torch.empty((ROW,), device=input.device, dtype=input.dtype)

    MAX_FUSED_SIZE = 65536 // input.element_size()
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(COL))
    USE_BLOCK = COL > BLOCK_SIZE
    NUM_PROG = min(ROW, get_num_sms())

    grid = (NUM_PROG,)
    _gluon_rms_norm_kernel[grid](
        input,
        output,
        weights,
        rsigma,
        ROW,
        COL,
        epsilon,
        input.stride(0),
        output.stride(0),
        BLOCK_SIZE,
        USE_BLOCK,
        NUM_PROG,
    )
    return output, rsigma
