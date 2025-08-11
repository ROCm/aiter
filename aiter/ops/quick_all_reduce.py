# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import torch
from typing import Optional
from ..jit.core import (
    compile_ops,
)

MD_NAME = "module_quick_all_reduce"


@compile_ops("module_quick_all_reduce")
def init_custom_qr(
    rank: int, world_size: int, qr_max_size: Optional[int] = None
) -> int: ...


@compile_ops("module_quick_all_reduce")
def qr_destroy(fa: int) -> None: ...


@compile_ops("module_quick_all_reduce")
def qr_all_reduce(
    fa: int,
    inp: torch.Tensor,
    out: torch.Tensor,
    quant_level: int,
    cast_bf2half: bool = False,
) -> None: ...


@compile_ops("module_quick_all_reduce")
def qr_get_handle(fa: int) -> torch.Tensor: ...


@compile_ops("module_quick_all_reduce")
def qr_open_handles(fa: int, handles: list[torch.Tensor]) -> None: ...


@compile_ops("module_quick_all_reduce")
def qr_max_size() -> int: ...
