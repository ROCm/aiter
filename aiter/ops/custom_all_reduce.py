# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from typing import List, Optional, Tuple

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
    use_new: bool,
    open_fp8_quant: bool,
    reg_input_buffer: Optional[torch.Tensor] = None,
    reg_output_buffer: Optional[torch.Tensor] = None,
) -> None: ...


@compile_ops("module_custom_all_reduce")
def reduce_scatter(
    _fa: int,
    inp: torch.Tensor,
    out: torch.Tensor,
    reg_buffer: Optional[torch.Tensor] = None,
) -> None: ...


@compile_ops("module_custom_all_reduce")
def all_gather_reg(
    _fa: int, inp: torch.Tensor, out: torch.Tensor, last_dim_size: int, dim: int
) -> None: ...


@compile_ops("module_custom_all_reduce")
def all_gather_unreg(
    _fa: int,
    inp: torch.Tensor,
    reg_buffer: torch.Tensor,
    out: torch.Tensor,
    last_dim_size: int,
    dim: int,
) -> None: ...


@compile_ops("module_custom_all_reduce")
def fused_allreduce_rmsnorm(
    _fa: int,
    inp: torch.Tensor,
    res_inp: torch.Tensor,
    res_out: torch.Tensor,
    out: torch.Tensor,
    w: torch.Tensor,
    eps: float,
    reg_buffer: Optional[torch.Tensor] = None,
    use_1stage: bool = False,
) -> None: ...


@compile_ops("module_custom_all_reduce")
def fused_allreduce_rmsnorm_quant(
    _fa: int,
    inp: torch.Tensor,
    res_inp: torch.Tensor,
    res_out: torch.Tensor,
    out: torch.Tensor,
    scale_out: torch.Tensor,
    w: torch.Tensor,
    eps: float,
    reg_buffer: Optional[torch.Tensor] = None,
    use_1stage: bool = False,
) -> None: ...


@compile_ops("module_asm_communication", ffi_type="ctypes")
def all_reduce_asm(
    inp: torch.Tensor,
    ca: int,
    reg_sig: torch.Tensor,
    reg_buffer: torch.Tensor,
    isGraph: bool,
) -> None: ...


def all_reduce_asm_(
    inp: torch.Tensor,
    ca: int,
    reg_sig: torch.Tensor,
    reg_buffer: torch.Tensor,
    isGraph: bool,
) -> torch.Tensor:
    all_reduce_asm(inp, ca, reg_sig, reg_buffer, isGraph)
    if isGraph:
        return inp
    else:
        return reg_buffer


@compile_ops("module_asm_communication", ffi_type="ctypes")
def all_reduce_rmsnorm(
    input: torch.Tensor,
    residual_in: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    epsilon: float,
    ca: int,
    reg_sig: torch.Tensor,
    reg_buffer: torch.Tensor,
    isGraph: bool,
) -> None: ...


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
) -> Tuple[torch.Tensor, torch.Tensor]:
    all_reduce_rmsnorm(
        input, residual_in, weight, bias, epsilon, ca, reg_sig, reg_buffer, isGraph
    )

    nbytes = input.numel() * input.element_size()
    size_pad = (nbytes + 4095) & ~4095

    storage = reg_buffer.untyped_storage()
    out = torch.empty([], dtype=input.dtype, device=input.device).set_(
        storage, size_pad, input.shape, input.stride()
    )
    residual_out = torch.empty([], dtype=input.dtype, device=input.device).set_(
        storage, size_pad * 2, input.shape, input.stride()
    )
    return out, residual_out


@compile_ops("module_asm_communication", ffi_type="ctypes")
def all_reduce_rmsnorm_quant(
    input: torch.Tensor,
    residual_in: torch.Tensor,
    xscale: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    epsilon: float,
    ca: int,
    reg_sig: torch.Tensor,
    reg_buffer: torch.Tensor,
    isGraph: bool,
) -> None: ...


def all_reduce_rmsnorm_quant_(
    input: torch.Tensor,
    residual_in: torch.Tensor,
    xscale: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    epsilon: float,
    ca: int,
    reg_sig: torch.Tensor,
    reg_buffer: torch.Tensor,
    isGraph: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    all_reduce_rmsnorm_quant(
        input,
        residual_in,
        xscale,
        weight,
        bias,
        epsilon,
        ca,
        reg_sig,
        reg_buffer,
        isGraph,
    )

    nbytes = input.numel() * input.element_size()
    size_pad = (nbytes + 4095) & ~4095

    storage = reg_buffer.untyped_storage()
    out = torch.empty([], dtype=input.dtype, device=input.device).set_(
        storage, size_pad, input.shape, input.stride()
    )
    residual_out = torch.empty([], dtype=input.dtype, device=input.device).set_(
        storage, size_pad * 2, input.shape, input.stride()
    )

    N = input.size(-1)
    M = input.numel() // N
    yscale = torch.empty([], dtype=torch.float32, device=input.device).set_(
        storage, size_pad * 3, (M, 1), (1, 1)
    )
    return out, residual_out, yscale


@compile_ops("module_custom_all_reduce")
def dispose(_fa: int) -> None: ...


@compile_ops("module_custom_all_reduce")
def meta_size() -> int: ...


@compile_ops("module_custom_all_reduce")
def register_input_buffer(
    _fa: int, t: torch.Tensor, handles: List[torch.Tensor], offsets: List[int]
) -> None: ...


@compile_ops("module_custom_all_reduce")
def register_output_buffer(
    _fa: int, t: torch.Tensor, handles: List[torch.Tensor], offsets: List[int]
) -> None: ...


# def gen_get_graph_buffer_ipc_meta_fake_tensors(_fa: int) -> List[torch.Tensor]:

#     handle_sz = 64  # sizeof(hipIpcMemHandle_t) is 64 byte
#     num_buffers = 4  # ???
#     handles = torch.empty((handle_sz * num_buffers,), dtype=torch.uint8, device="cuda")

#     offset_tensor = torch.empty((num_buffers,), dtype=torch.int64, device="cuda")

#     return [handles, offset_tensor]


@compile_ops("module_custom_all_reduce")
def get_graph_buffer_ipc_meta(_fa: int) -> Tuple[torch.Tensor, torch.Tensor]: ...


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
