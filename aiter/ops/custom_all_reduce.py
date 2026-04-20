# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from typing import List

import torch

from ..jit.core import compile_ops

MD_NAME = "module_custom_all_reduce"


@compile_ops("module_custom_all_reduce", develop=True)
def init_custom_ar(
    meta_ptr: int,
    rank_data_ptr: int,
    rank_data_sz: int,
    ipc_handle_ptrs: List[int],
    offsets: List[int],
    rank: int,
    fully_connected: bool,
) -> int: ...


@compile_ops("module_custom_all_reduce", develop=True)
def all_reduce(
    _fa: int,
    inp,
    out,
    use_new: bool,
    open_fp8_quant: bool,
    reg_inp_ptr: int,
    reg_inp_bytes: int,
) -> None: ...


@compile_ops("module_custom_all_reduce", develop=True)
def reduce_scatter(
    _fa: int,
    inp,
    out,
    reg_ptr: int,
    reg_bytes: int,
) -> None: ...


@compile_ops("module_custom_all_reduce", develop=True)
def all_gather_reg(
    _fa: int,
    inp,
    out,
    dim: int,
) -> None: ...


@compile_ops("module_custom_all_reduce", develop=True)
def all_gather_unreg(
    _fa: int,
    inp,
    reg_buffer: int,
    out,
    reg_bytes: int,
    dim: int,
) -> None: ...


@compile_ops("module_custom_all_reduce", develop=True)
def fused_allreduce_rmsnorm(
    _fa: int,
    inp,
    res_inp,
    res_out,
    out,
    w,
    eps: float,
    reg_ptr: int,
    reg_bytes: int,
    use_1stage: bool,
) -> None: ...


@compile_ops("module_custom_all_reduce", develop=True)
def fused_allreduce_rmsnorm_quant(
    _fa: int,
    inp,
    res_inp,
    res_out,
    out,
    scale_out,
    w,
    eps: float,
    reg_ptr: int,
    reg_bytes: int,
    use_1stage: bool,
) -> None: ...


@compile_ops("module_custom_all_reduce", develop=True)
def fused_allreduce_rmsnorm_quant_per_group(
    _fa: int,
    inp: torch.Tensor,
    res_inp: torch.Tensor,
    res_out: torch.Tensor,
    out: torch.Tensor,
    scale_out: torch.Tensor,
    w: torch.Tensor,
    eps: float,
    group_size: int,
    reg_ptr: int,
    reg_bytes: int,
    use_1stage: bool,
    bf16_out_ptr: int = 0,
) -> None: ...


@compile_ops("module_custom_all_reduce", develop=True)
def dispose(_fa: int) -> None: ...


@compile_ops("module_custom_all_reduce", develop=True)
def meta_size() -> int: ...


@compile_ops("module_custom_all_reduce", develop=True)
def register_input_buffer(
    _fa: int, self_ptr: int, ipc_handle_ptrs: List[int], offsets: List[int]
) -> None: ...


@compile_ops("module_custom_all_reduce", develop=True)
def register_output_buffer(
    _fa: int, self_ptr: int, ipc_handle_ptrs: List[int], offsets: List[int]
) -> None: ...


@compile_ops("module_custom_all_reduce", develop=True)
def get_graph_buffer_count(_fa: int) -> int: ...


@compile_ops("module_custom_all_reduce", develop=True)
def get_graph_buffer_ipc_meta(_fa: int, handle_out: int, offset_out: int) -> None: ...


@compile_ops("module_custom_all_reduce", develop=True)
def register_graph_buffers(
    _fa: int, handle_ptrs: List[int], offset_ptrs: List[int]
) -> None: ...


@compile_ops("module_custom_all_reduce", develop=True)
def allocate_meta_buffer(size: int) -> int: ...


@compile_ops("module_custom_all_reduce", develop=True)
def free_meta_buffer(ptr: int) -> None: ...


@compile_ops("module_custom_all_reduce", develop=True)
def get_meta_buffer_ipc_handle(inp_ptr: int, out_handle_ptr: int) -> None: ...
