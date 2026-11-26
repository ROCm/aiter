# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Run a resolved plan on one call.

Statement for statement this is the part of the legacy `_fused_moe_impl` and
`fused_moe_2stages` that the plan's segments reach, in the same order: the same
validation, buffers, kernel launches and benchmark hooks.

NOTE: aiter.moev2 is under active development (MoE v2 migration, not graduated).
Please do not modify this code without coordinating with the moev2 owners (@qli88). See
aiter/moev2/README.md.
"""

import functools
from dataclasses import dataclass, replace

import torch

from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe_registry import FusedMoeRequest
from aiter.ops.flydsl.moe_common import GateMode

from .. import hooks
from ..shapes import _is_row_major_2d, get_inter_dim
from .output import (
    _return_output,
    _validate_output_buffer_metadata,
    _validate_output_buffer_no_overlap,
)


@dataclass
class Call:
    """A `fused_moe_` call after the legacy prologue (enums decoded, shapes derived)."""

    args: dict
    activation: ActivationType
    quant_type: QuantType
    gate_mode: GateMode
    block_size_M: int | None
    M: int
    topk: int
    E: int
    model_dim: int
    inter_dim: int
    isG1U1: bool
    global_E: int
    dtype: torch.dtype

    def __getitem__(self, name):
        return self.args[name]


def prologue(call, quant_remap, *, prepare=True) -> Call:
    """The validation and derived values of `_fused_moe_impl`, up to dispatch.

    ``prepare=False`` builds a side-effect-free dispatch preview. A path that accepts the
    preview calls the full prologue before execution; a declined call reaches legacy
    without materializing a strided activation or validating its output buffer twice.
    """
    activation = ActivationType(call["activation"])
    quant_type = QuantType(call["quant_type"])
    gate_mode = GateMode(call["gate_mode"])
    block_size_M = call["block_size_M"]
    if block_size_M == -1:
        block_size_M = None
    hidden_states, w1, w2 = call["hidden_states"], call["w1"], call["w2"]
    M, topk = call["topk_ids"].shape
    E, model_dim, inter_dim = get_inter_dim(w1.shape, w2.shape)
    if (
        prepare
        and not hidden_states.is_contiguous()
        and not _is_row_major_2d(hidden_states)
    ):
        hidden_states = hidden_states.contiguous()
        call = {**call, "hidden_states": hidden_states}

    assert w1.shape[1] in [
        inter_dim,
        inter_dim * 2,
    ], f"Invalid MoE weight: {w1.shape=} {w2.shape=}"
    isG1U1 = inter_dim != w1.shape[1]
    global_E = E
    if call["expert_mask"] is not None:
        global_E = call["expert_mask"].numel()
    dtype = hidden_states.dtype if call["dtype"] is None else call["dtype"]
    assert dtype in [
        dtypes.fp16,
        dtypes.bf16,
    ], f"Fused_moe unsupported out dtype: {dtype}"
    if prepare:
        output = call["output"]
        _validate_output_buffer_metadata(
            output, (M, model_dim), dtype, hidden_states.device
        )
        _validate_output_buffer_no_overlap(output, hidden_states)
    quant_type = quant_remap.get(quant_type, quant_type)
    return Call(
        call,
        activation,
        quant_type,
        gate_mode,
        block_size_M,
        M,
        topk,
        E,
        model_dim,
        inter_dim,
        isG1U1,
        global_E,
        dtype,
    )


def run_whole_graph(c: Call, metadata, block_size_M, q_dtype_a):
    request = FusedMoeRequest(
        hidden_states=c["hidden_states"],
        w1=c["w1"],
        w2=c["w2"],
        topk_weight=c["topk_weight"],
        topk_ids=c["topk_ids"],
        expert_mask=c["expert_mask"],
        activation=c.activation,
        quant_type=c.quant_type,
        doweight_stage1=c["doweight_stage1"],
        w1_scale=c["w1_scale"],
        w2_scale=c["w2_scale"],
        a1_scale=c["a1_scale"],
        a2_scale=c["a2_scale"],
        block_size_m=(int(block_size_M) if block_size_M is not None else None),
        ksplit=metadata.ksplit,
        num_local_tokens=c["num_local_tokens"],
        moe_sorting_dispatch_policy=c["moe_sorting_dispatch_policy"],
        dtype=c.dtype,
        hidden_pad=c["hidden_pad"],
        intermediate_pad=c["intermediate_pad"],
        bias1=c["bias1"],
        bias2=c["bias2"],
        swiglu_limit=c["swiglu_limit"],
        beta=c["beta"],
        linear_beta=c["linear_beta"],
        gate_mode=c.gate_mode,
        q_dtype_a=q_dtype_a,
        q_dtype_w=c["w1"].dtype,
        w1_global_scale=c["w1_global_scale"],
        w2_global_scale=c["w2_global_scale"],
    )
    full_call = functools.partial(metadata.full_impl, request)
    hooks.record_kernel("whole_graph", full_call)
    full_output = full_call()
    return _return_output(full_output, c["output"])


def _e8m0_view(w, scale):
    # Only one-byte packed scales may be reinterpreted as E8M0.
    return (
        scale.view(dtypes.fp8_e8m0)
        if w.dtype in (dtypes.fp4x2, dtypes.fp8)
        and scale is not None
        and scale.element_size() == 1
        else scale
    )


def run_two_stage(c: Call, metadata, plan, q_dtype_a, after_sort=None):
    """Sort, quantize, GEMM1, quantize, GEMM2: `_fused_moe_impl` + `fused_moe_2stages`.

    `after_sort()` runs where `fused_moe_2stages` starts, after the sort.
    """
    if not c["hidden_states"].is_contiguous():
        c = replace(
            c, args={**c.args, "hidden_states": c["hidden_states"].contiguous()}
        )
    block_size_M = metadata.block_m if c.block_size_M is None else c.block_size_M
    if metadata.full_impl is not None:
        return run_whole_graph(c, metadata, block_size_M, q_dtype_a)
    # Ensure block_size_M is int (metadata.block_m from CSV may be float)
    if block_size_M is not None:
        block_size_M = int(block_size_M)
    sort = plan.segment("sort").segment
    quant_a1 = plan.segment("quant.a1").segment
    quant_inter = plan.segment("quant.inter").segment

    hidden_states, w1, w2 = c["hidden_states"], c["w1"], c["w2"]
    topk_ids, topk_weight = c["topk_ids"], c["topk_weight"]
    num_local_tokens = c["num_local_tokens"]
    output = c["output"]
    doweight_stage1 = c["doweight_stage1"]
    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_out = sort.run(
        topk_ids,
        topk_weight,
        c.global_E,
        c.model_dim,
        c.dtype,
        block_size_M,
        c["expert_mask"],
        num_local_tokens,
        c["moe_sorting_dispatch_policy"],
        output,
    )
    if after_sort is not None:
        after_sort()

    token_num, _ = hidden_states.shape
    topk = c.topk
    inter_dim = c.inter_dim
    dtype = moe_out.dtype
    device = hidden_states.device
    if moe_out.numel() == 0:
        moe_out = (
            output
            if output is not None
            else torch.empty((token_num, c.model_dim), dtype=dtype, device=device)
        )
    a1, a1_scale = quant_a1.run(
        c.quant_type, hidden_states, c["a1_scale"], q_dtype_a, num_local_tokens
    )
    # Stage1 does not write slots with topk_ids == -1 or masked out by
    # expert_mask, but dynamic per-tensor a2 quant reads the whole buffer,
    # so stale memory in those slots can set the scale.
    _a2_alloc = (
        torch.zeros
        if c.quant_type == QuantType.per_Tensor and c["a2_scale"] is None
        else torch.empty
    )
    a2 = _a2_alloc(
        (token_num, topk, inter_dim),
        dtype=dtype,
        device=device,
    )
    _stage1_call = functools.partial(
        metadata.stage1,
        a1,
        w1,
        w2,
        sorted_ids,
        sorted_expert_ids,
        num_valid_ids,
        None if metadata.fuse_quant else a2,
        topk,
        block_m=block_size_M,
        a1_scale=a1_scale,
        w1_scale=_e8m0_view(w1, c["w1_scale"]),
        sorted_weights=sorted_weights if doweight_stage1 else None,
    )
    hooks.record_kernel("stage1", _stage1_call)
    a2 = _stage1_call()
    a2, a2_scale = quant_inter.run(
        c.quant_type,
        a2,
        c["a2_scale"],
        q_dtype_a,
        num_local_tokens,
        token_num,
        topk,
        inter_dim,
    )
    _stage2_call = functools.partial(
        metadata.stage2,
        a2,
        w1,
        w2,
        sorted_ids,
        sorted_expert_ids,
        num_valid_ids,
        moe_out,
        topk,
        w2_scale=_e8m0_view(w2, c["w2_scale"]),
        a2_scale=a2_scale,
        block_m=block_size_M,
        sorted_weights=sorted_weights if not doweight_stage1 else None,
    )
    hooks.record_kernel("stage2", _stage2_call)
    _stage2_call()
    return _return_output(moe_out, output)
