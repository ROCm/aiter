# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import torch
from typing import Optional
from ..jit.core import (
    compile_ops,
)

MD_NAME = "module_quick_all_reduce"


@compile_ops("module_quick_all_reduce", develop=True)
def init_custom_qr(
    rank: int, world_size: int, qr_max_size: Optional[int] = None
) -> int: ...


@compile_ops("module_quick_all_reduce", develop=True)
def qr_destroy(fa: int) -> None: ...


@compile_ops("module_quick_all_reduce", develop=True)
def qr_all_reduce(
    fa: int,
    inp: torch.Tensor,
    out: torch.Tensor,
    quant_level: int,
    cast_bf2half: bool = False,
) -> None: ...


@compile_ops("module_quick_all_reduce", fc_name="qr_get_handle", develop=True)
def _qr_get_handle(fa: int, out_ptr: int, out_nbytes: int) -> None: ...


@compile_ops("module_quick_all_reduce", fc_name="qr_open_handles", develop=True)
def _qr_open_handles(fa: int, handle_ptrs: list[int], handle_nbytes: int) -> None: ...


@compile_ops("module_quick_all_reduce", develop=True)
def qr_handle_nbytes() -> int: ...


def qr_get_handle(fa: int) -> torch.Tensor:
    out = torch.empty(qr_handle_nbytes(), dtype=torch.uint8, device="cpu")
    _qr_get_handle(fa, out.data_ptr(), out.numel())
    return out


def qr_open_handles(fa: int, handles: list[torch.Tensor]) -> None:
    handle_nbytes = qr_handle_nbytes()
    handle_ptrs = []
    for handle in handles:
        if handle.device.type != "cpu":
            raise ValueError("qr_open_handles: handles must be CPU tensors")
        if handle.dtype != torch.uint8:
            raise ValueError("qr_open_handles: handles must have dtype torch.uint8")
        if not handle.is_contiguous():
            raise ValueError("qr_open_handles: handles must be contiguous")
        if handle.numel() < handle_nbytes:
            raise ValueError("qr_open_handles: handle buffer is too small")
        handle_ptrs.append(handle.data_ptr())
    _qr_open_handles(fa, handle_ptrs, handle_nbytes)


@compile_ops("module_quick_all_reduce", develop=True)
def qr_max_size() -> int: ...
