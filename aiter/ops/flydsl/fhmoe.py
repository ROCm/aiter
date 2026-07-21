# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL runtime support for fused heterogeneous MoE (FHMoE)."""

import functools
from typing import Optional

import torch

from aiter.ops.flydsl.kernels.tensor_shim import ptr_arg

from .moe_kernels import (
    _flydsl_moe_stage1_impl,
    _flydsl_moe_stage2_impl,
)


def compile_flydsl_fhmoe_stage1(
    model_dim: int,
    inter_dim: int,
    experts: int,
    topk: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    doweight_stage1: bool,
    a_dtype: str,
    b_dtype: str,
    out_dtype: str,
    act: str = "silu",
    persist_m: int = 1,
    use_async_copy: bool = False,
    k_batch: int = 1,
    waves_per_eu: int = 3,
    b_nt: int = 2,
    gate_mode: str = "separated",
    model_dim_pad: int = 0,
    inter_dim_pad: int = 0,
    enable_bias: bool = False,
    a_scale_one: bool = False,
    k_wave: int = 1,
    shared_expert_id: int = -1,
):
    """Compile the heterogeneous stage1 kernel."""
    from .kernels.fhmoe import compile_mixed_fhmoe_gemm1
    from .moe_common import GateMode

    if b_dtype != "fp4":
        raise ValueError(f"FHMoE stage1 requires routed MXFP4 weights, got {b_dtype}")
    return compile_mixed_fhmoe_gemm1(
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=experts,
        topk=topk,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        doweight_stage1=doweight_stage1,
        a_dtype=a_dtype,
        b_dtype=b_dtype,
        out_dtype=out_dtype,
        act=act,
        persist_m=persist_m,
        use_async_copy=use_async_copy,
        k_batch=k_batch,
        waves_per_eu=waves_per_eu,
        b_nt=b_nt,
        gate_mode=GateMode(gate_mode),
        model_dim_pad=model_dim_pad,
        inter_dim_pad=inter_dim_pad,
        enable_bias=enable_bias,
        a_scale_one=a_scale_one,
        k_wave=k_wave,
        shared_expert_id=shared_expert_id,
    )


def compile_flydsl_fhmoe_stage2(
    model_dim: int,
    inter_dim: int,
    experts: int,
    topk: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    doweight_stage2: bool,
    a_dtype: str,
    b_dtype: str,
    out_dtype: str,
    accumulate: bool = True,
    persist_m: int = 1,
    sort_block_m: int = 0,
    waves_per_eu: Optional[int] = None,
    use_async_copy: bool = False,
    cu_num_mul: int = 1,
    b_nt: int = 0,
    model_dim_pad: int = 0,
    inter_dim_pad: int = 0,
    enable_bias: bool = False,
    shared_expert_id: int = -1,
):
    """Compile the heterogeneous stage2 kernel."""
    from .kernels.fhmoe import compile_mixed_fhmoe_gemm2

    if b_dtype != "fp4":
        raise ValueError(f"FHMoE stage2 requires routed MXFP4 weights, got {b_dtype}")
    return compile_mixed_fhmoe_gemm2(
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=experts,
        topk=topk,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        doweight_stage2=doweight_stage2,
        a_dtype=a_dtype,
        b_dtype=b_dtype,
        out_dtype=out_dtype,
        accumulate=accumulate,
        persist_m=persist_m,
        sort_block_m=sort_block_m,
        waves_per_eu=waves_per_eu,
        use_async_copy=use_async_copy,
        cu_num_mul=cu_num_mul,
        b_nt=b_nt,
        model_dim_pad=model_dim_pad,
        inter_dim_pad=inter_dim_pad,
        enable_bias=enable_bias,
        shared_expert_id=shared_expert_id,
    )


def _compile_fhmoe_xcd0(compiler, *, xcd_swizzle: int, **kwargs):
    if xcd_swizzle != 0:
        raise ValueError("FHMoE kernels do not support XCD swizzling")
    return compiler(**kwargs)


def _s1_args_fhmoe(
    out,
    a,
    w,
    a_scale,
    w_scale,
    sorted_ids,
    sorted_expert_ids,
    sorted_weights,
    num_valid_ids,
    out_scale_sorted,
    token_num,
    n_in,
    k_in,
    size_expert_ids_in,
    dev,
    bias=None,
    stream=None,
    swiglu_limit=float("inf"),
    *,
    shared_w,
    shared_w_scale,
):
    """Build the expanded heterogeneous stage1 launch ABI."""
    empty_f32 = torch.empty(0, device=dev, dtype=torch.float32)
    kernel_bias = bias if bias is not None else empty_f32
    if stream is None:
        stream = torch.cuda.current_stream()
    return (
        ptr_arg(out),
        ptr_arg(a),
        ptr_arg(w),
        ptr_arg(a_scale),
        ptr_arg(w_scale),
        ptr_arg(shared_w),
        ptr_arg(shared_w_scale),
        ptr_arg(sorted_ids),
        ptr_arg(sorted_expert_ids),
        ptr_arg(sorted_weights),
        ptr_arg(num_valid_ids),
        ptr_arg(kernel_bias),
        ptr_arg(out_scale_sorted),
        token_num,
        n_in,
        k_in,
        size_expert_ids_in,
        float(swiglu_limit),
        stream,
    )


def _s2_args_fhmoe(
    target,
    a,
    w,
    a_scale,
    w_scale,
    sorted_ids,
    sorted_expert_ids,
    sorted_weights,
    num_valid_ids,
    token_num,
    n_in,
    k_in,
    blocks,
    dev,
    bias=None,
    stream=None,
    *,
    shared_w,
    shared_w_scale,
):
    """Build the expanded heterogeneous stage2 launch ABI."""
    kernel_bias = (
        bias.view(-1)
        if bias is not None
        else torch.empty(0, device=dev, dtype=torch.float32)
    )
    if stream is None:
        stream = torch.cuda.current_stream()
    return (
        ptr_arg(target),
        ptr_arg(a),
        ptr_arg(w),
        ptr_arg(a_scale),
        ptr_arg(w_scale),
        ptr_arg(shared_w),
        ptr_arg(shared_w_scale),
        ptr_arg(sorted_ids),
        ptr_arg(sorted_expert_ids),
        ptr_arg(sorted_weights),
        ptr_arg(num_valid_ids),
        ptr_arg(kernel_bias),
        token_num,
        n_in,
        k_in,
        blocks,
        stream,
    )


def flydsl_fhmoe_stage1(
    a: torch.Tensor,
    w1: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    sorted_expert_ids: torch.Tensor,
    num_valid_ids: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    topk: int = 1,
    *,
    tile_m: int = 32,
    tile_n: int = 256,
    tile_k: int = 256,
    a_dtype: str = "fp8",
    b_dtype: str = "fp4",
    out_dtype: str = "bf16",
    act: str = "silu",
    w1_scale: Optional[torch.Tensor] = None,
    a1_scale: Optional[torch.Tensor] = None,
    sorted_weights: Optional[torch.Tensor] = None,
    persist_m: int = 0,
    use_async_copy: bool = False,
    k_batch: int = 1,
    waves_per_eu: int = 3,
    b_nt: int = 0,
    gate_mode: str = "separated",
    model_dim_pad: int = 0,
    inter_dim_pad: int = 0,
    bias: Optional[torch.Tensor] = None,
    topk_ids: Optional[torch.Tensor] = None,
    a_scale_one: bool = False,
    swiglu_limit: Optional[float] = None,
    k_wave: int = 1,
    shared_w1: torch.Tensor,
    shared_w1_scale: torch.Tensor,
    shared_expert_id: int,
):
    """Run stage1 with MXFP4 routed experts and one FP8 shared expert."""
    compile_kernel = functools.partial(
        _compile_fhmoe_xcd0,
        functools.partial(
            compile_flydsl_fhmoe_stage1,
            shared_expert_id=shared_expert_id,
        ),
    )
    build_mx_args = functools.partial(
        _s1_args_fhmoe,
        shared_w=shared_w1.view(-1),
        shared_w_scale=shared_w1_scale.view(-1),
    )
    return _flydsl_moe_stage1_impl(
        a=a,
        w1=w1,
        sorted_token_ids=sorted_token_ids,
        sorted_expert_ids=sorted_expert_ids,
        num_valid_ids=num_valid_ids,
        out=out,
        topk=topk,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        a_dtype=a_dtype,
        b_dtype=b_dtype,
        out_dtype=out_dtype,
        act=act,
        w1_scale=w1_scale,
        a1_scale=a1_scale,
        sorted_weights=sorted_weights,
        persist_m=persist_m,
        use_async_copy=use_async_copy,
        k_batch=k_batch,
        waves_per_eu=waves_per_eu,
        b_nt=b_nt,
        gate_mode=gate_mode,
        model_dim_pad=model_dim_pad,
        inter_dim_pad=inter_dim_pad,
        bias=bias,
        topk_ids=topk_ids,
        a_scale_one=a_scale_one,
        xcd_swizzle=0,
        swiglu_limit=swiglu_limit,
        k_wave=k_wave,
        _compile_kernel=compile_kernel,
        _build_mx_args=build_mx_args,
    )


def flydsl_fhmoe_stage2(
    inter_states: torch.Tensor,
    w2: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    sorted_expert_ids: torch.Tensor,
    num_valid_ids: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    topk: int = 1,
    *,
    tile_m: int = 32,
    tile_n: int = 128,
    tile_k: int = 256,
    a_dtype: str = "fp8",
    b_dtype: str = "fp4",
    out_dtype: str = "bf16",
    mode: str = "atomic",
    w2_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
    sorted_weights: Optional[torch.Tensor] = None,
    sort_block_m: int = 0,
    persist: Optional[bool] = None,
    waves_per_eu: Optional[int] = None,
    use_async_copy: bool = False,
    cu_num_mul: int = 1,
    b_nt: int = 0,
    model_dim_pad: int = 0,
    inter_dim_pad: int = 0,
    bias: Optional[torch.Tensor] = None,
    return_per_slot: bool = False,
    expert_mask: Optional[torch.Tensor] = None,
    topk_ids: Optional[torch.Tensor] = None,
    shared_w2: torch.Tensor,
    shared_w2_scale: torch.Tensor,
    shared_expert_id: int,
) -> torch.Tensor:
    """Run stage2 with MXFP4 routed experts and one FP8 shared expert."""
    compile_kernel = functools.partial(
        _compile_fhmoe_xcd0,
        functools.partial(
            compile_flydsl_fhmoe_stage2,
            shared_expert_id=shared_expert_id,
        ),
    )
    build_mx_args = functools.partial(
        _s2_args_fhmoe,
        shared_w=shared_w2.view(-1),
        shared_w_scale=shared_w2_scale.view(-1),
    )
    return _flydsl_moe_stage2_impl(
        inter_states=inter_states,
        w2=w2,
        sorted_token_ids=sorted_token_ids,
        sorted_expert_ids=sorted_expert_ids,
        num_valid_ids=num_valid_ids,
        out=out,
        topk=topk,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        a_dtype=a_dtype,
        b_dtype=b_dtype,
        out_dtype=out_dtype,
        mode=mode,
        w2_scale=w2_scale,
        a2_scale=a2_scale,
        sorted_weights=sorted_weights,
        sort_block_m=sort_block_m,
        persist=persist,
        waves_per_eu=waves_per_eu,
        use_async_copy=use_async_copy,
        cu_num_mul=cu_num_mul,
        b_nt=b_nt,
        model_dim_pad=model_dim_pad,
        inter_dim_pad=inter_dim_pad,
        xcd_swizzle=0,
        bias=bias,
        return_per_slot=return_per_slot,
        expert_mask=expert_mask,
        topk_ids=topk_ids,
        _compile_kernel=compile_kernel,
        _build_mx_args=build_mx_args,
    )
