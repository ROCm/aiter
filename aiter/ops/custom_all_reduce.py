# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

from typing import List, Optional

import torch

from ..jit.core import compile_ops

MD_NAME = "module_custom_all_reduce"


@compile_ops("module_custom_all_reduce")
def init_custom_ar(
    meta: torch.Tensor,
    rank_data: torch.Tensor,
    handles: List[torch.Tensor],
    offsets: List[int],
    rank: int,
    fully_connected: bool,
) -> int: ...


@compile_ops("module_custom_all_reduce")
def all_reduce(
    _fa: int,
    inp: torch.Tensor,
    out: torch.Tensor,
    open_fp8_quant: bool,
    reg_buffer: Optional[torch.Tensor] = None,
) -> None: ...


@compile_ops("module_custom_all_reduce")
def all_gather_reg(_fa: int, inp: torch.Tensor, out: torch.Tensor) -> None: ...


@compile_ops("module_custom_all_reduce")
def all_gather_unreg(
    _fa: int, inp: torch.Tensor, reg_buffer: torch.Tensor, out: torch.Tensor
) -> None: ...


@compile_ops("module_custom_all_reduce")
def fused_allreduce_rmsnorm(
    _fa: int,
    inp: torch.Tensor,
    out: torch.Tensor,
    w: torch.Tensor,
    eps: float,
    reg_buffer: Optional[torch.Tensor] = None,
) -> None: ...


def all_reduce_asm_fake_tensor(
    inp: torch.Tensor,
    ca: int,
    reg_sig: torch.Tensor,
    reg_buffer: torch.Tensor,
    isGraph: bool,
) -> torch.Tensor:

    return torch.empty_like(
        inp,
        dtype=inp.dtype,
        device=inp.device,
    )


@compile_ops("module_custom_all_reduce", gen_fake=all_reduce_asm_fake_tensor)
def all_reduce_asm_(
    inp: torch.Tensor,
    ca: int,
    reg_sig: torch.Tensor,
    reg_buffer: torch.Tensor,
    isGraph: bool,
) -> torch.Tensor: ...


def all_reduce_rmsnorm_fake_tensors(
    input: torch.Tensor,
    residual_in: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    epsilon: float,
    ca: int,
    reg_sig: torch.Tensor,
    reg_buffer: torch.Tensor,
    isGraph: bool,
) -> List[torch.Tensor]:

    output = torch.empty_like(
        input, dtype=input.dtype, device=input.device, requires_grad=input.requires_grad
    )

    residual_out = torch.empty_like(
        input, dtype=input.dtype, device=input.device, requires_grad=input.requires_grad
    )

    return [output, residual_out]


@compile_ops("module_custom_all_reduce", gen_fake=all_reduce_rmsnorm_fake_tensors)
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
) -> List[torch.Tensor]: ...


def all_reduce_rmsnorm_quant_fake_tensors(
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
) -> List[torch.Tensor]:

    N = input.size(-1)
    M = input.numel() // N

    output = torch.empty_like(
        input, dtype=input.dtype, device=input.device, requires_grad=input.requires_grad
    )

    residual_out = torch.empty_like(
        input, dtype=input.dtype, device=input.device, requires_grad=input.requires_grad
    )

    y_scale = torch.empty((M, 1), dtype=torch.float32, device=input.device)

    return [output, residual_out, y_scale]


@compile_ops("module_custom_all_reduce", gen_fake=all_reduce_rmsnorm_quant_fake_tensors)
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
) -> List[torch.Tensor]: ...


@compile_ops("module_custom_all_reduce")
def dispose(_fa: int) -> None: ...


@compile_ops("module_custom_all_reduce")
def meta_size() -> int: ...


@compile_ops("module_custom_all_reduce")
def register_buffer(
    _fa: int, t: torch.Tensor, handles: List[torch.Tensor], offsets: List[int]
) -> None: ...


# def gen_get_graph_buffer_ipc_meta_fake_tensors(_fa: int) -> List[torch.Tensor]:

#     handle_sz = 64  # sizeof(hipIpcMemHandle_t) is 64 byte
#     num_buffers = 4  # ???
#     handles = torch.empty((handle_sz * num_buffers,), dtype=torch.uint8, device="cuda")

#     offset_tensor = torch.empty((num_buffers,), dtype=torch.int64, device="cuda")

#     return [handles, offset_tensor]


@compile_ops("module_custom_all_reduce")
def get_graph_buffer_ipc_meta(_fa: int) -> tuple[torch.Tensor, torch.Tensor]: ...


@compile_ops("module_custom_all_reduce")
def register_graph_buffers(
    _fa: int, handles: List[torch.Tensor], offsets: List[torch.Tensor]
) -> None: ...


@compile_ops("module_custom_all_reduce")
def allocate_meta_buffer(size: int) -> torch.Tensor: ...


# def get_meta_buffer_ipc_handle_fake(inp: torch.Tensor) -> torch.Tensor:
#     handle_size = 64
#     if not inp.is_cuda:
#         raise RuntimeError("Input tensor must be on CUDA device")

#     return torch.empty(handle_size, dtype=torch.uint8, device=inp.device)


@compile_ops("module_custom_all_reduce")
def get_meta_buffer_ipc_handle(inp: torch.Tensor) -> torch.Tensor: ...
