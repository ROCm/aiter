# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import torch
from torch import Tensor
from typing import Optional
from ..jit.core import (
    compile_ops,
)
import functools


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
def _gemm_a16w16_asm(
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


def gemm_a16w16_asm(
    A: Tensor,
    B: Tensor,
    out: Tensor,
    bias: Optional[Tensor] = None,
    splitK: Optional[int] = None,
    kernelName: Optional[str] = None,
    bpreshuffle: bool = False,
):
    if splitK is None or splitK > 1:
        sema = get_semaphore_workspace(out.device)
    else:
        sema = torch.empty((0,), dtype=torch.uint32, device=out.device)

    return _gemm_a16w16_asm(A, B, out, sema, bias, splitK, kernelName, bpreshuffle)
