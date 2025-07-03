# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import torch
from typing import Optional
from ..jit.core import compile_ops


@compile_ops("module_hipbsolgemm")
def hipb_create_extension(): ...


@compile_ops("module_hipbsolgemm")
def hipb_destroy_extension(): ...


@compile_ops("module_hipbsolgemm")
def hipb_mm(
    mat1: torch.Tensor,
    mat2: torch.Tensor,
    solution_index: int,
    bias: Optional[torch.Tensor] = None,
    out_dtype: Optional[object] = None,
    scaleA: Optional[torch.Tensor] = None,
    scaleB: Optional[torch.Tensor] = None,
    scaleOut: Optional[torch.Tensor] = None,
) -> torch.Tensor: ...


@compile_ops("module_hipbsolgemm")
def hipb_findallsols(
    mat1: torch.Tensor,
    mat2: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    out_dtype: Optional[object] = None,
    scaleA: Optional[torch.Tensor] = None,
    scaleB: Optional[torch.Tensor] = None,
    scaleC: Optional[torch.Tensor] = None,
) -> list[int]: ...


@compile_ops("module_hipbsolgemm")
def getHipblasltKernelName(): ...


@compile_ops("module_rocsolgemm")
def rocb_create_extension(): ...


@compile_ops("module_rocsolgemm")
def rocb_destroy_extension(): ...


@compile_ops("module_rocsolgemm")
def rocb_mm(arg0: torch.Tensor, arg1: torch.Tensor, arg2: int) -> torch.Tensor: ...


@compile_ops("module_rocsolgemm")
def rocb_findallsols(arg0: torch.Tensor, arg1: torch.Tensor) -> list[int]: ...
