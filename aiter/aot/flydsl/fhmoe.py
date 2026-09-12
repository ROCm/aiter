#!/usr/bin/env python3

# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""AOT adapters for fused heterogeneous MoE (FHMoE)."""

from __future__ import annotations

from dataclasses import dataclass

from aiter.aot.flydsl.common import cu_num_to_arch


def _shared_weight(device, n_in: int, k_in: int):
    import torch

    return torch.zeros((1, n_in, k_in), dtype=torch.uint8, device=device)


def _shared_scale(device, n_in: int, k_in: int):
    import torch

    rows = (n_in + 255) // 256 * 256
    cols = ((k_in + 255) // 256 * 256) // 32
    return torch.zeros((rows, cols), dtype=torch.uint8, device=device)


@dataclass(frozen=True)
class _FHMoEAOTBackend:
    shared_expert_id: int

    def build_stage1_args(
        self,
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
        pass_swiglu_limit: bool = True,
    ):
        from aiter.ops.flydsl.fhmoe import _s1_args_fhmoe

        shared_w = _shared_weight(dev, n_in, k_in)
        shared_w_scale = _shared_scale(dev, n_in, k_in)
        return _s1_args_fhmoe(
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
            bias=bias,
            stream=stream,
            swiglu_limit=swiglu_limit,
            pass_swiglu_limit=pass_swiglu_limit,
            shared_w=shared_w.view(-1),
            shared_w_scale=shared_w_scale.view(-1),
        )

    def build_stage2_args(
        self,
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
    ):
        from aiter.ops.flydsl.fhmoe import _s2_args_fhmoe

        shared_w = _shared_weight(dev, n_in, k_in)
        shared_w_scale = _shared_scale(dev, n_in, k_in)
        return _s2_args_fhmoe(
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
            bias=bias,
            stream=stream,
            shared_w=shared_w,
            shared_w_scale=shared_w_scale,
        )

    def compile_stage1(self, **kwargs):
        from aiter.ops.flydsl.fhmoe import compile_flydsl_fhmoe_stage1

        return compile_flydsl_fhmoe_stage1(
            **kwargs,
            shared_expert_id=self.shared_expert_id,
        )

    def compile_stage2(self, **kwargs):
        from aiter.ops.flydsl.fhmoe import compile_flydsl_fhmoe_stage2

        return compile_flydsl_fhmoe_stage2(
            **kwargs,
            shared_expert_id=self.shared_expert_id,
        )


def precompile_fhmoe_to_cache(
    *,
    experts: int,
    shared_expert_id: int,
    a_dtype: str = "fp8",
    b_dtype: str = "fp4",
    act: str = "silu",
    cu_num: int = 0,
    enable_bias: bool = False,
    **kwargs,
):
    """Precompile one heterogeneous MoE job through the shared AOT harness."""
    if shared_expert_id != experts - 1:
        raise ValueError(
            "FHMoE AOT expects the shared expert to be the final logical expert; "
            f"got {shared_expert_id=} for {experts=}"
        )
    if a_dtype != "fp8" or b_dtype != "fp4":
        raise ValueError(
            "FHMoE AOT supports routed FP8 activations and MXFP4 weights; "
            f"got {a_dtype=} and {b_dtype=}"
        )
    if enable_bias:
        raise ValueError("FHMoE AOT does not support expert bias")
    if act != "silu":
        raise ValueError(f"FHMoE AOT supports only SiLU, got {act=}")
    if cu_num_to_arch(cu_num) != "gfx950":
        raise ValueError(f"FHMoE AOT supports only gfx950, got {cu_num=}")

    from aiter.aot.flydsl.moe import _precompile_to_cache

    return _precompile_to_cache(
        experts=experts,
        a_dtype=a_dtype,
        b_dtype=b_dtype,
        act=act,
        cu_num=cu_num,
        enable_bias=enable_bias,
        _aot_backend=_FHMoEAOTBackend(shared_expert_id),
        **kwargs,
    )
