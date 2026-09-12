# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import torch

from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import (
    gemm_a16wfp4,
)
from aiter.ops.triton.utils.logger import AiterTritonLogger

_LOGGER = AiterTritonLogger()


def gemm_afp4wfp4_pre_quant(
    x: torch.Tensor,
    w: torch.Tensor,
    w_scales: torch.Tensor,
    dtype: float | None = torch.bfloat16,
    y: torch.Tensor | None = None,
    config: dict | None = None,
) -> torch.Tensor:
    _LOGGER.info(
        "gemm_afp4wfp4_pre_quant will be deprecated in future AITER release, please switch to gemm_a16wfp4"
    )
    return gemm_a16wfp4(x, w, w_scales, True, dtype, y, config)
