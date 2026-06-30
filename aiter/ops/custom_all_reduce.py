# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from typing import List

import torch

from ..jit.core import compile_ops

MD_NAME = "module_custom_all_reduce"
FUSED_AR_MHC_MD_NAME = "module_fused_ar_mhc"


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
    inp: torch.Tensor,
    out: torch.Tensor,
    use_new: bool,
    open_fp8_quant: bool,
    reg_inp_ptr: int,
    reg_inp_bytes: int,
) -> None: ...


@compile_ops("module_custom_all_reduce", develop=True)
def reduce_scatter(
    _fa: int,
    inp: torch.Tensor,
    out: torch.Tensor,
    m: int,
    n: int,
    k: int,
    split_dim: int,
    reg_ptr: int,
    reg_bytes: int,
) -> None: ...


@compile_ops("module_custom_all_reduce", develop=True)
def all_gather_reg(
    _fa: int,
    inp: torch.Tensor,
    out: torch.Tensor,
    dim: int,
) -> None: ...


@compile_ops("module_custom_all_reduce", develop=True)
def all_gather_unreg(
    _fa: int,
    inp: torch.Tensor,
    reg_buffer: int,
    out: torch.Tensor,
    reg_bytes: int,
    dim: int,
) -> None: ...


@compile_ops("module_custom_all_reduce", develop=True)
def fused_allreduce_rmsnorm(
    _fa: int,
    inp: torch.Tensor,
    res_inp: torch.Tensor,
    res_out: torch.Tensor,
    out: torch.Tensor,
    w: torch.Tensor,
    eps: float,
    reg_ptr: int,
    reg_bytes: int,
    use_1stage: bool,
    gemma_norm: bool = False,
) -> None: ...


@compile_ops("module_custom_all_reduce", develop=True)
def fused_allreduce_rmsnorm_pad(
    _fa: int,
    inp: torch.Tensor,
    res_inp: torch.Tensor,
    res_out: torch.Tensor,
    out: torch.Tensor,
    w: torch.Tensor,
    eps: float,
    reg_ptr: int,
    reg_bytes: int,
    use_1stage: bool,
    gemma_norm: bool = False,
) -> None: ...


@compile_ops("module_custom_all_reduce", develop=True)
def fused_allreduce_rmsnorm_quant(
    _fa: int,
    inp: torch.Tensor,
    res_inp: torch.Tensor,
    res_out: torch.Tensor,
    out: torch.Tensor,
    scale_out: torch.Tensor,
    w: torch.Tensor,
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
    transpose_scale: bool = False,
) -> None: ...


@compile_ops("module_custom_all_reduce", develop=True)
def fused_allreduce_rmsnorm_mxfp4_quant(
    _fa: int,
    inp: torch.Tensor,
    res_inp: torch.Tensor,
    res_out: torch.Tensor,
    out: torch.Tensor,
    scale_out: torch.Tensor,
    w: torch.Tensor,
    eps: float,
    reg_ptr: int,
    reg_bytes: int,
    use_1stage: bool,
    bf16_out_ptr: int = 0,
) -> None: ...


@compile_ops("module_custom_all_reduce", develop=True)
def fused_qknorm_allreduce(
    _fa: int,
    qkv_in: torch.Tensor,
    q_w: torch.Tensor,
    k_w: torch.Tensor,
    q_out: torch.Tensor,
    k_out: torch.Tensor,
    v_out: torch.Tensor,
    eps: float,
    reg_ptr: int,
    reg_bytes: int,
) -> None: ...


@compile_ops("module_custom_all_reduce", develop=True)
def fused_qknorm_allreduce_rope(
    _fa: int,
    qkv_in: torch.Tensor,
    q_w: torch.Tensor,
    k_w: torch.Tensor,
    q_out: torch.Tensor,
    k_out: torch.Tensor,
    v_out: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    position_ids: torch.Tensor,
    head_dim: int,
    rotary_dim: int,
    eps: float,
    reg_ptr: int,
    reg_bytes: int,
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


@compile_ops(FUSED_AR_MHC_MD_NAME)
def fused_allreduce_mhc_post_only(
    _fa: int,
    inp: torch.Tensor,
    next_residual: torch.Tensor,
    residual_in: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
    use_new: bool = True,
    open_fp8_quant: bool = False,
    reg_ptr: int = 0,
    reg_bytes: int = 0,
) -> None: ...


@compile_ops(FUSED_AR_MHC_MD_NAME)
def fused_allreduce_mhc_post_one_stage(
    _fa: int,
    inp: torch.Tensor,
    next_residual: torch.Tensor,
    residual_in: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
    use_new: bool = True,
    open_fp8_quant: bool = False,
    reg_ptr: int = 0,
    reg_bytes: int = 0,
) -> None: ...


@compile_ops(FUSED_AR_MHC_MD_NAME)
def fused_allreduce_mhc_post_split(
    _fa: int,
    inp: torch.Tensor,
    next_residual: torch.Tensor,
    residual_in: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
    use_new: bool = True,
    open_fp8_quant: bool = False,
    reg_ptr: int = 0,
    reg_bytes: int = 0,
) -> None: ...


def _launch_fused_allreduce_mhc_post(
    _fa: int,
    inp: torch.Tensor,
    residual_in: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
    *,
    next_residual: torch.Tensor | None = None,
    split_path: bool = False,
    use_new: bool = True,
    open_fp8_quant: bool = False,
    reg_ptr: int = 0,
    reg_bytes: int = 0,
) -> torch.Tensor:
    if post_layer_mix.ndim == 3:
        post_layer_mix = post_layer_mix.squeeze(-1)
    if next_residual is None:
        next_residual = torch.empty_like(residual_in)
    launch_fn = (
        fused_allreduce_mhc_post_split if split_path else fused_allreduce_mhc_post_only
    )
    launch_fn(
        _fa,
        inp,
        next_residual,
        residual_in,
        post_layer_mix,
        comb_res_mix,
        use_new,
        open_fp8_quant,
        reg_ptr,
        reg_bytes,
    )
    return next_residual


def launch_fused_allreduce_mhc_post_only(
    _fa: int,
    inp: torch.Tensor,
    residual_in: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
    *,
    use_new: bool = True,
    open_fp8_quant: bool = False,
    reg_ptr: int = 0,
    reg_bytes: int = 0,
) -> torch.Tensor:
    """Launch fused custom AllReduce + MHC post (no pre / RMSNorm)."""
    return _launch_fused_allreduce_mhc_post(
        _fa,
        inp,
        residual_in,
        post_layer_mix,
        comb_res_mix,
        split_path=False,
        use_new=use_new,
        open_fp8_quant=open_fp8_quant,
        reg_ptr=reg_ptr,
        reg_bytes=reg_bytes,
    )


def launch_fused_allreduce_mhc_post_split(
    _fa: int,
    inp: torch.Tensor,
    residual_in: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
    *,
    next_residual: torch.Tensor | None = None,
    use_new: bool = True,
    open_fp8_quant: bool = False,
    reg_ptr: int = 0,
    reg_bytes: int = 0,
) -> torch.Tensor:
    """Launch 2-stage split AR + MHC post (large-M optimized path)."""
    return _launch_fused_allreduce_mhc_post(
        _fa,
        inp,
        residual_in,
        post_layer_mix,
        comb_res_mix,
        next_residual=next_residual,
        split_path=True,
        use_new=use_new,
        open_fp8_quant=open_fp8_quant,
        reg_ptr=reg_ptr,
        reg_bytes=reg_bytes,
    )
