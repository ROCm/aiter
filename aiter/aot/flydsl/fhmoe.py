#!/usr/bin/env python3

# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""AOT adapters for fused heterogeneous MoE (FHMoE)."""

from __future__ import annotations

from dataclasses import dataclass

from aiter.fhmoe_contract import _is_hy4_mxfp8_fhmoe_contract


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
        x_rows,
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
            x_rows,
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

        clamp_shared = kwargs.pop("clamp_shared", True)
        return compile_flydsl_fhmoe_stage1(
            **kwargs,
            shared_expert_id=self.shared_expert_id,
            clamp_shared=clamp_shared,
        )

    def compile_stage2(self, **kwargs):
        from aiter.ops.flydsl.fhmoe import compile_flydsl_fhmoe_stage2

        return compile_flydsl_fhmoe_stage2(
            **kwargs,
            shared_expert_id=self.shared_expert_id,
        )


def precompile_fhmoe_to_cache(
    *,
    stage: int,
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
    if stage not in (1, 2):
        raise ValueError(f"FHMoE AOT stage must be 1 or 2, got {stage}")
    if shared_expert_id != experts - 1:
        raise ValueError(
            "FHMoE AOT expects the shared expert to be the final logical expert; "
            f"got {shared_expert_id=} for {experts=}"
        )
    if a_dtype != "fp8" or b_dtype not in ("fp4", "fp8"):
        raise ValueError(
            "FHMoE AOT supports routed FP8 activations with MXFP4 or MXFP8 weights; "
            f"got {a_dtype=} and {b_dtype=}"
        )
    gate_mode = kwargs.get("gate_mode", "separated")
    if stage == 1 and b_dtype == "fp8" and gate_mode != "interleave":
        raise ValueError(
            "FHMoE AOT requires interleaved gate/up layout for MXFP8 routed "
            f"weights, got {gate_mode=}"
        )
    clamp_shared = kwargs.get("clamp_shared", True)
    if (
        stage == 1
        and not clamp_shared
        and not _is_hy4_mxfp8_fhmoe_contract(
            model_dim=kwargs["model_dim"],
            inter_dim=kwargs["inter_dim"],
            experts=experts,
            topk=kwargs["topk"],
            routed_mxfp8=b_dtype == "fp8",
            hidden_pad=kwargs.get("model_dim_pad", 0),
            intermediate_pad=kwargs.get("inter_dim_pad", 0),
            gate_interleaved=gate_mode == "interleave",
            doweight_stage1=kwargs.get("doweight_stage1", False),
            shared_expert_id=shared_expert_id,
        )
    ):
        raise ValueError(
            "clamp_shared=False requires the HY4-compatible MXFP8 FHMoE "
            "shape and layout contract"
        )
    if enable_bias:
        raise ValueError("FHMoE AOT does not support expert bias")
    if act != "silu":
        raise ValueError(f"FHMoE AOT supports only SiLU, got {act=}")
    if cu_num not in (0, 256):
        raise ValueError(
            f"FHMoE AOT supports only gfx950 or the default CU sentinel, got {cu_num=}"
        )
    effective_cu_num = 256 if cu_num == 0 else cu_num

    from aiter.aot.flydsl.moe import _precompile_to_cache

    return _precompile_to_cache(
        stage=stage,
        experts=experts,
        a_dtype=a_dtype,
        b_dtype=b_dtype,
        act=act,
        cu_num=effective_cu_num,
        enable_bias=enable_bias,
        _aot_backend=_FHMoEAOTBackend(shared_expert_id),
        **kwargs,
    )
