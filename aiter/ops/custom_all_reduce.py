# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import torch
from typing import List, Optional
from ..jit.core import (
    compile_ops,
)

MD_NAME = "module_custom_all_reduce"


@compile_ops("module_custom_all_reduce")
def init_custom_ar(
    meta: torch.Tensor,
    rank_data: torch.Tensor,
    handles: list[torch.Tensor],
    offsets: list[int],
    rank: int,
    full_nvlink: bool,
) -> int: ...


@compile_ops("module_custom_all_reduce")
def all_reduce_reg(
    _fa: int, inp: torch.Tensor, out: torch.Tensor, open_fp8_quant: bool
)-> None: ...


@compile_ops("module_custom_all_reduce")
def all_reduce_unreg(
    _fa: int, inp: torch.Tensor, reg_buffer: torch.Tensor, out: torch.Tensor
)-> None: ...


@compile_ops("module_custom_all_reduce")
def all_reduce_asm_(
    inp: torch.Tensor,
    ca: int,
    reg_sig: torch.Tensor,
    reg_buffer: torch.Tensor,
    isGraph: bool,
    output: torch.Tensor
) -> None: ...


@compile_ops("module_custom_all_reduce")
def all_reduce_rmsnorm_(
    input: torch.Tensor,
    residual_in: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    epsilon: float,
    ca: int,
    reg_sig: torch.Tensor,
    reg_buffer: torch.Tensor,
    isGraph: bool,
    output: List[torch.Tensor],
) -> None: ...


@compile_ops("module_custom_all_reduce")
def all_reduce_rmsnorm_quant_(
    input: torch.Tensor,
    residual_in: torch.Tensor,
    weight: torch.Tensor,
    xscale: torch.Tensor,
    bias: torch.Tensor,
    epsilon: float,
    ca: int,
    reg_sig: torch.Tensor,
    reg_buffer: torch.Tensor,
    isGraph: bool,
    output: List[torch.Tensor],
) -> None: ...


@compile_ops("module_custom_all_reduce")
def dispose(_fa: int, out: torch.Tensor)-> None: ...


@compile_ops("module_custom_all_reduce")
def meta_size() -> int: ...


@compile_ops("module_custom_all_reduce")
def register_buffer(
    _fa: int, t: torch.Tensor, handles: list[torch.Tensor], offsets: list[int]
)-> None: ...


@compile_ops("module_custom_all_reduce")
def get_graph_buffer_ipc_meta(_fa: int, handle: torch.Tensor, offset_tensor: torch.Tensor) -> None: ...


@compile_ops("module_custom_all_reduce")
def register_graph_buffers(
    _fa: int, handles: list[torch.Tensor], offsets: list[torch.Tensor]
)-> None: ...


@compile_ops("module_custom_all_reduce")
def allocate_meta_buffer(size: int, out: torch.Tensor) -> None: ...


@compile_ops("module_custom_all_reduce")
def get_meta_buffer_ipc_handle(inp: torch.Tensor, out: torch.Tensor) -> None: ...
