# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

from typing import Optional
import torch
from aiter.ops.triton.utils.logger import AiterTritonLogger
from aiter.ops.triton.utils.common_utils import serialize_dict
from aiter.ops.triton.batched_gemm_a16wfp4 import (
    batched_gemm_a16wfp4,
)

_LOGGER = AiterTritonLogger()

global _USE_GEMM_SPLITK_BF16
_USE_GEMM_SPLITK_BF16 = False


def set_use_gemm_splitk_bf16(value: bool):
    global _USE_GEMM_SPLITK_BF16
    _USE_GEMM_SPLITK_BF16 = value


def batched_gemm_afp4wfp4_pre_quant(
    x,
    w,
    w_scales,
    dtype: Optional[float] = torch.bfloat16,
    y: Optional[torch.Tensor] = None,
    config: Optional[dict] = None,
):
    _LOGGER.info(
        "batched_gemm_afp4wfp4_pre_quant will be deprecated in future AITER release, please switch to batched_gemm_a16wfp4"
    )

    config_hashable = serialize_dict(config) if config else None
    return batched_gemm_a16wfp4(
        x, w, w_scales, dtype, y, config_hashable, transpose_bm=False, prequant=True
    )
