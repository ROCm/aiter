# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import torch

from csrc.cpp_itfs.torch_utils import direct_register_custom_op

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


# torch-free kernel entry: writes into caller-allocated `result` (the de-torched C++
# TU can no longer torch::empty). outDtype is read from result.dtype() on the C++ side.
@compile_ops("module_hipbsolgemm", fc_name="hipb_mm", develop=True)
def _hipb_mm(
    mat1: torch.Tensor,
    mat2: torch.Tensor,
    solution_index: int,
    result: torch.Tensor,
    bias: torch.Tensor | None = None,
    scaleA: torch.Tensor | None = None,
    scaleB: torch.Tensor | None = None,
    scaleOut: torch.Tensor | None = None,
    bpreshuffle: bool | None = None,
    use_gelu: bool | None = None,
) -> None: ...


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
) -> torch.Tensor:
    out_dtype = out_dtype if out_dtype is not None else mat1.dtype
    result = torch.empty(
        (mat1.size(0), mat2.size(1)), dtype=out_dtype, device=mat1.device
    )
    _hipb_mm(
        mat1,
        mat2,
        solution_index,
        result,
        bias,
        scaleA,
        scaleB,
        scaleOut,
        bpreshuffle,
        use_gelu,
    )
    return result


direct_register_custom_op(
    "hipb_mm",
    hipb_mm,
    [],
    fake_impl=gen_hipb_mm_fake_tensor,
)


@compile_ops("module_hipbsolgemm", fc_name="hipb_findallsols", develop=True)
def _hipb_findallsols(
    mat1: torch.Tensor,
    mat2: torch.Tensor,
    result: torch.Tensor,
    bias: torch.Tensor | None = None,
    scaleA: torch.Tensor | None = None,
    scaleB: torch.Tensor | None = None,
    scaleC: torch.Tensor | None = None,
    bpreshuffle: bool = False,
    use_gelu: bool = False,
) -> list[int]: ...


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
) -> list[int]:
    out_dtype = out_dtype if out_dtype is not None else mat1.dtype
    result = torch.empty(
        (mat1.size(0), mat2.size(1)), dtype=out_dtype, device=mat1.device
    )
    return _hipb_findallsols(
        mat1, mat2, result, bias, scaleA, scaleB, scaleC, bpreshuffle, use_gelu
    )


@compile_ops("module_hipbsolgemm")
def getHipblasltKernelName() -> None: ...


@compile_ops("module_rocsolgemm")
def rocb_create_extension() -> None: ...


@compile_ops("module_rocsolgemm")
def rocb_destroy_extension() -> None: ...


def gen_rocb_mm_fake_tensor(
    arg0: torch.Tensor, arg1: torch.Tensor, arg2: int
) -> torch.Tensor:
    # gemm out = (M, N) = (arg0.size(0), arg1.size(1)).
    return torch.empty(
        (arg0.size(0), arg1.size(1)), dtype=arg0.dtype, device=arg0.device
    )


# torch-free kernel entry: writes into caller-allocated `result`.
@compile_ops("module_rocsolgemm", fc_name="rocb_mm", develop=True)
def _rocb_mm(
    arg0: torch.Tensor, arg1: torch.Tensor, result: torch.Tensor, arg2: int
) -> None: ...


def rocb_mm(arg0: torch.Tensor, arg1: torch.Tensor, arg2: int) -> torch.Tensor:
    result = torch.empty(
        (arg0.size(0), arg1.size(1)), dtype=arg0.dtype, device=arg0.device
    )
    _rocb_mm(arg0, arg1, result, arg2)
    return result


direct_register_custom_op(
    "rocb_mm",
    rocb_mm,
    [],
    fake_impl=gen_rocb_mm_fake_tensor,
)


@compile_ops("module_rocsolgemm", fc_name="rocb_findallsols", develop=True)
def _rocb_findallsols(
    arg0: torch.Tensor, arg1: torch.Tensor, result: torch.Tensor
) -> list[int]: ...


def rocb_findallsols(arg0: torch.Tensor, arg1: torch.Tensor) -> list[int]:
    result = torch.empty(
        (arg0.size(0), arg1.size(1)), dtype=arg0.dtype, device=arg0.device
    )
    return _rocb_findallsols(arg0, arg1, result)
