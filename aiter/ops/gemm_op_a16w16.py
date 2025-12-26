# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import torch
from torch import Tensor
from typing import Optional
from aiter import logger
from ..jit.core import (
    compile_ops,
    AITER_ROOT_DIR,
)
from ..jit.utils.chip_info import get_cu_num
from ..jit.utils.chip_info import get_gfx
import functools
import pandas as pd
from ..ops.gemm_op_common import get_padded_m


def gen_gemm_a16w16_asm_fake_tensors(
    A: Tensor,
    B: Tensor,
    out: Tensor,
    semaphore: Tensor,
    bias: Optional[Tensor] = None,
    splitK: Optional[int] = None,
    kernelName: Optional[str] = None,
    bpreshuffle: bool = False,
) -> Tensor:
    return out


@compile_ops(
    "module_gemm_a16w16_asm",
    fc_name="gemm_a16w16_asm",
    gen_fake=gen_gemm_a16w16_asm_fake_tensors,
)
def gemm_a16w16_asm(
    A: Tensor,
    B: Tensor,
    out: Tensor,
    semaphore: Tensor,
    bias: Optional[Tensor] = None,
    splitK: Optional[int] = None,
    kernelName: Optional[str] = None,
    bpreshuffle: bool = False,
) -> Tensor: ...


@functools.lru_cache(maxsize=1)
def get_semaphore_workspace(device: torch.device) -> Tensor:
    return torch.zeros((16, 64), dtype=torch.uint32, device=device)


def gemm_a16w16(
    A: Tensor,
    B: Tensor,
    out: Tensor,
    bias: Optional[Tensor] = None,
    splitK: Optional[int] = None,
    kernelName: Optional[str] = None,
):
    sema = get_semaphore_workspace(out.device)
    return gemm_a16w16_asm(A, B, out, bias, sema, splitK, kernelName)
