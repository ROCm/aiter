#!/usr/bin/env python3

# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""AOT adapters for fused heterogeneous MoE (FHMoE)."""

from __future__ import annotations

from aiter.aot.flydsl.common import cu_num_to_arch


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

    from aiter.aot.flydsl.moe import compile_moe_job

    return compile_moe_job(
        {
            "experts": experts,
            "shared_expert_id": shared_expert_id,
            "a_dtype": a_dtype,
            "b_dtype": b_dtype,
            "act": act,
            "cu_num": cu_num,
            "enable_bias": enable_bias,
            **kwargs,
        }
    )
