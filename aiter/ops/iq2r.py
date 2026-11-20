# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Compiled gfx950 operations of the GLM-5.3 IQ2R-v2 MoE."""

from __future__ import annotations

from torch import Tensor

from ..jit.core import compile_ops

# GLM-5.3 packed-layout MoE stages (TP4/TP8, 256 routed experts, top-8,
# hidden 6144).  Shapes are validated natively; see
# ``aiter.iq2r_glm53`` for the orchestration and tuned dispatch.


@compile_ops("module_iq2r_moe", fc_name="iq2r_glm53_m1_route_quant_out", develop=True)
def iq2r_glm53_m1_route_quant_out(
    input: Tensor,
    topk_ids: Tensor,
    scatter_indices: Tensor,
    tasks: Tensor,
    task_count: Tensor,
    output: Tensor,
    scales: Tensor,
    shared: int,
) -> None: ...


@compile_ops("module_iq2r_moe", fc_name="iq2r_glm53_sort_quant_out", develop=True)
def iq2r_glm53_sort_quant_out(
    input: Tensor,
    topk_ids: Tensor,
    gather_indices: Tensor,
    scatter_indices: Tensor,
    tasks: Tensor,
    task_count: Tensor,
    output: Tensor,
    scales: Tensor,
    gate_tasks: Tensor,
    gate_task_count: Tensor,
    shared: int,
) -> None: ...


@compile_ops(
    "module_iq2r_moe", fc_name="iq2r_glm53_prefill_sort_quant_out", develop=True
)
def iq2r_glm53_prefill_sort_quant_out(
    input: Tensor,
    topk_ids: Tensor,
    gather_indices: Tensor,
    scatter_indices: Tensor,
    tasks: Tensor,
    task_count: Tensor,
    output: Tensor,
    scales: Tensor,
    scratch: Tensor,
) -> None: ...


@compile_ops("module_iq2r_moe", fc_name="iq2r_glm53_route_reduce_out", develop=True)
def iq2r_glm53_route_reduce_out(
    route_output: Tensor,
    route_weights: Tensor,
    scatter_indices: Tensor,
    output: Tensor,
) -> None: ...


@compile_ops("module_iq2r_moe", fc_name="iq2r_glm53_gate_m1_out", develop=True)
def iq2r_glm53_gate_m1_out(
    activations: Tensor,
    scales: Tensor,
    data: Tensor,
    auxiliary: Tensor,
    tasks: Tensor,
    task_count: Tensor,
    output: Tensor,
    output_scales: Tensor,
    shared_gate: Tensor,
    shared_gate_scales: Tensor,
    shared_act: Tensor,
) -> None: ...


@compile_ops("module_iq2r_moe", fc_name="iq2r_glm53_gate_out", develop=True)
def iq2r_glm53_gate_out(
    activations: Tensor,
    scales: Tensor,
    data: Tensor,
    auxiliary: Tensor,
    tasks: Tensor,
    task_count: Tensor,
    gather: Tensor,
    output: Tensor,
    output_scales: Tensor,
    kernel: int,
    grid_multiplier: int,
    shared_gate: Tensor,
    shared_gate_scales: Tensor,
    shared_act: Tensor,
) -> None: ...


@compile_ops("module_iq2r_moe", fc_name="iq2r_glm53_down_out", develop=True)
def iq2r_glm53_down_out(
    activations: Tensor,
    scales: Tensor,
    data: Tensor,
    auxiliary: Tensor,
    tasks: Tensor,
    task_count: Tensor,
    output: Tensor,
    kernel: int,
    grid_multiplier: int,
    scatter: Tensor,
    shared_down: Tensor,
    shared_down_scales: Tensor,
    shared_act: Tensor,
) -> None: ...


@compile_ops("module_iq2r_moe", fc_name="iq2r_glm53_down_per_token_out", develop=True)
def iq2r_glm53_down_per_token_out(
    activations: Tensor,
    scales: Tensor,
    data: Tensor,
    auxiliary: Tensor,
    expert_ids: Tensor,
    scatter: Tensor,
    route_weights: Tensor,
    output: Tensor,
    shared_down: Tensor,
    shared_down_scales: Tensor,
    shared_act: Tensor,
) -> None: ...


@compile_ops("module_iq2r_moe", fc_name="iq2r_glm53_down_reduce_out", develop=True)
def iq2r_glm53_down_reduce_out(
    activations: Tensor,
    scales: Tensor,
    data: Tensor,
    auxiliary: Tensor,
    tasks: Tensor,
    task_count: Tensor,
    route_output: Tensor,
    scatter: Tensor,
    route_weights: Tensor,
    output: Tensor,
    chunks: int,
) -> None: ...


__all__ = [
    "iq2r_glm53_down_out",
    "iq2r_glm53_down_per_token_out",
    "iq2r_glm53_down_reduce_out",
    "iq2r_glm53_gate_m1_out",
    "iq2r_glm53_gate_out",
    "iq2r_glm53_m1_route_quant_out",
    "iq2r_glm53_prefill_sort_quant_out",
    "iq2r_glm53_route_reduce_out",
    "iq2r_glm53_sort_quant_out",
]
