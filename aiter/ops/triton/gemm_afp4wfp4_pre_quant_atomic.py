# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

from typing import Optional
import torch
import triton
import triton.language as tl
from aiter.ops.triton.utils.logger import AiterTritonLogger
from aiter.ops.triton.gemm_a16wfp4 import (
    gemm_a16wfp4,
)

_LOGGER = AiterTritonLogger()


def gemm_afp4wfp4_pre_quant(
    x,
    w,
    w_scales,
    dtype: Optional[float] = torch.bfloat16,
    y: Optional[torch.Tensor] = None,
    config: Optional[dict] = None,
):
    _LOGGER.info(
        "gemm_afp4wfp4_pre_quant will be deprecated in future AITER release, please switch to gemm_a16wfp4"
    )
    return gemm_a16wfp4(x, w, w_scales, True, dtype, y, config)
