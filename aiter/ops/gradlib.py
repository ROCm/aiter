# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import torch
from ..jit.core import compile_ops


@compile_ops("module_hipbsolgemm")
def hipb_create_extension() -> None: ...


@compile_ops("module_hipbsolgemm")
def hipb_destroy_extension() -> None: ...


def gen_hipb_mm_fake_tensor(
    mat1: torch.Tensor,
    mat2: torch.Tensor,
    solution_index: int,
    bias: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
    scaleA: torch.Tensor | None = None,
    scaleB: torch.Tensor | None = None,
    scaleOut: torch.Tensor | None = None,
    bpreshuffle: bool | None = None,
    use_gelu: bool | None = None,
):
    mat1_sizes = mat1.size()
    mat2_sizes = mat2.size()
    in_dtype = mat1.dtype
    out_dtype = out_dtype if out_dtype is not None else in_dtype
    result = torch.empty(
        (mat1_sizes[0], mat2_sizes[1]), dtype=out_dtype, device=mat1.device
    )

    return result


@compile_ops("module_hipbsolgemm", gen_fake=gen_hipb_mm_fake_tensor)
def hipb_mm(
    mat1: torch.Tensor,
    mat2: torch.Tensor,
    solution_index: int,
    bias: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
    scaleA: torch.Tensor | None = None,
    scaleB: torch.Tensor | None = None,
    scaleOut: torch.Tensor | None = None,
    bpreshuffle: bool | None = None,
    use_gelu: bool | None = None,
) -> torch.Tensor: ...


@compile_ops("module_hipbsolgemm")
def hipb_findallsols(
    mat1: torch.Tensor,
    mat2: torch.Tensor,
    bias: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
    scaleA: torch.Tensor | None = None,
    scaleB: torch.Tensor | None = None,
    scaleC: torch.Tensor | None = None,
    bpreshuffle: bool = False,
    use_gelu: bool = False,
) -> list[int]: ...


@compile_ops("module_hipbsolgemm")
def getHipblasltKernelName() -> None: ...


@compile_ops("module_rocsolgemm")
def rocb_create_extension() -> None: ...


@compile_ops("module_rocsolgemm")
def rocb_destroy_extension() -> None: ...


def gen_rocb_mm_fake_tensor(
    arg0: torch.Tensor, arg1: torch.Tensor, arg2: int
) -> torch.Tensor:
    mat1_sizes = arg0.size()
    mat2_sizes = arg0.size()
    in_dtype = arg0.dtype
    result = torch.empty(
        (mat1_sizes[0], mat2_sizes[1]), dtype=in_dtype, device=arg0.device
    )

    return result


@compile_ops("module_rocsolgemm", gen_fake=gen_rocb_mm_fake_tensor)
def rocb_mm(arg0: torch.Tensor, arg1: torch.Tensor, arg2: int) -> torch.Tensor: ...


@compile_ops("module_rocsolgemm")
def rocb_findallsols(arg0: torch.Tensor, arg1: torch.Tensor) -> list[int]: ...
