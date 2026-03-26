# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from typing import List, Optional, Tuple

import torch

from ..jit.core import compile_ops

MD_NAME = "module_custom_all_reduce"


@compile_ops("module_custom_all_reduce")
def init_custom_ar(
    meta_ptr: int,
    rank_data_ptr: int,
    rank_data_sz: int,
    ipc_handle_ptrs: List[int],
    offsets: List[int],
    rank: int,
    fully_connected: bool,
) -> int: ...


@compile_ops("module_custom_all_reduce")
def all_reduce(
    _fa: int,
    inp,  # aiter_tensor_t
    out,  # aiter_tensor_t
    use_new: bool,
    open_fp8_quant: bool,
    reg_inp_ptr: int,
    reg_inp_bytes: int,
    reg_out_ptr: int,
    reg_out_bytes: int,
    stream: int,
) -> None: ...


@compile_ops("module_custom_all_reduce")
def reduce_scatter(
    _fa: int,
    inp,  # aiter_tensor_t
    out,  # aiter_tensor_t
    reg_ptr: int,
    reg_bytes: int,
    stream: int,
) -> None: ...


@compile_ops("module_custom_all_reduce")
def all_gather_reg(
    _fa: int,
    inp,  # aiter_tensor_t
    out,  # aiter_tensor_t
    dim: int,
    stream: int,
) -> None: ...


@compile_ops("module_custom_all_reduce")
def all_gather_unreg(
    _fa: int,
    inp,  # aiter_tensor_t
    reg_buffer: int,
    out,  # aiter_tensor_t
    reg_bytes: int,
    dim: int,
    stream: int,
) -> None: ...


@compile_ops("module_custom_all_reduce")
def fused_allreduce_rmsnorm(
    _fa: int,
    inp,  # aiter_tensor_t
    res_inp,  # aiter_tensor_t
    res_out,  # aiter_tensor_t
    out,  # aiter_tensor_t
    w,  # aiter_tensor_t
    eps: float,
    reg_ptr: int,
    reg_bytes: int,
    use_1stage: bool,
    stream: int,
) -> None: ...


@compile_ops("module_custom_all_reduce")
def fused_allreduce_rmsnorm_quant(
    _fa: int,
    inp,  # aiter_tensor_t
    res_inp,  # aiter_tensor_t
    res_out,  # aiter_tensor_t
    out,  # aiter_tensor_t
    scale_out,  # aiter_tensor_t
    w,  # aiter_tensor_t
    eps: float,
    reg_ptr: int,
    reg_bytes: int,
    use_1stage: bool,
    stream: int,
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
def register_input_buffer(
    _fa: int, self_ptr: int, ipc_handle_ptrs: List[int], offsets: List[int]
) -> None: ...


@compile_ops("module_custom_all_reduce")
def register_output_buffer(
    _fa: int, self_ptr: int, ipc_handle_ptrs: List[int], offsets: List[int]
) -> None: ...


@compile_ops("module_custom_all_reduce")
def get_graph_buffer_count(_fa: int) -> int: ...


@compile_ops("module_custom_all_reduce")
def get_graph_buffer_ipc_meta(_fa: int, handle_out: int, offset_out: int) -> None: ...


@compile_ops("module_custom_all_reduce")
def register_graph_buffers(
    _fa: int, handle_ptrs: List[int], offset_ptrs: List[int]
) -> None: ...


@compile_ops("module_custom_all_reduce")
def allocate_meta_buffer(size: int, stream: int) -> int: ...


@compile_ops("module_custom_all_reduce")
def free_meta_buffer(ptr: int) -> None: ...


@compile_ops("module_custom_all_reduce")
def get_meta_buffer_ipc_handle(inp_ptr: int, out_handle_ptr: int) -> None: ...
