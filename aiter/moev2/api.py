# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Public entry points of aiter.moev2, with the legacy signatures.

`fused_moe` here always runs the v2 code: it ignores AITER_MOEV2 and raises when no
registered path takes the call. The legacy `aiter.fused_moe.fused_moe` is the entry
point that honours the switch.

`build_problem` turns a `fused_moe_` call into the `MoeProblem` dispatch resolves.

NOTE: aiter.moev2 is under active development (MoE v2 migration, not graduated).
Please do not modify this code without coordinating with the moev2 owners (@qli88). See
aiter/moev2/README.md.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from aiter import ActivationType, QuantType
from aiter.jit.utils.chip_info import get_cu_num, get_gfx_runtime
from aiter.presorted_moe import drop_presort_after

from . import paths
from .policy import _normalize_mxfp4_activation_params, resolve_activation_dtype
from .problem import MoeProblem
from .shapes import get_padded_M

if TYPE_CHECKING:
    from aiter.ops.flydsl.kernels.mega_moe_gfx1250.types import Stage2ScatterContext

    from .execute.executor import Call


def dispatch_key(c: Call):
    """`(args, kwargs, q_dtype_a)`: the legacy `get_2stage_cfgs` arguments for a call.

    Built exactly as `_fused_moe_impl` builds them, from the prologue's derived values.
    """
    hidden_states, w1, w2 = c["hidden_states"], c["w1"], c["w2"]
    q_dtype_w = w1.dtype
    q_dtype_a = resolve_activation_dtype(
        c.quant_type,
        q_dtype_w,
        activation=c.activation,
        gate_mode=c.gate_mode,
        M=c.M,
        hidden_dtype=hidden_states.dtype,
        has_a1_scale=c["a1_scale"] is not None,
        gfx=get_gfx_runtime(),
        gate_up_interleave=getattr(w1, "gate_up_interleave", None),
    )
    if c["quant_dtype_a"] is not None:
        q_dtype_a = c["quant_dtype_a"]
    situ_beta, situ_linear_beta, _ = _normalize_mxfp4_activation_params(
        c.activation, c["beta"], c["linear_beta"], c["swiglu_limit"]
    )
    args = (
        get_padded_M(c.M),
        c.model_dim,
        c.inter_dim,
        c.E,
        c.topk,
        c.dtype,
        q_dtype_a,
        q_dtype_w,
        c.quant_type,
        c.isG1U1,
        c.activation,
        c["doweight_stage1"],
        c["hidden_pad"],
        c["intermediate_pad"],
        getattr(w1, "is_shuffled", False) or getattr(w2, "is_shuffled", False),
        c.gate_mode,
    )
    kwargs = {
        "is_ep": c["expert_mask"] is not None,
        "has_stage1_bias": c["bias1"] is not None,
        "has_stage2_bias": c["bias2"] is not None,
        "situ_beta": situ_beta,
        "situ_linear_beta": situ_linear_beta,
        "swiglu_limit": c["swiglu_limit"],
        "opus_weights_shuffled": getattr(w1, "is_shuffled", False)
        and getattr(w2, "is_shuffled", False),
        "gate_up_interleave": getattr(w1, "gate_up_interleave", None),
        "config_file": None,
        "_disable_inline_sort": False,
        "q_dtype_a2": c["quant_dtype_a2"],
        "input_dtype": hidden_states.dtype,
        "has_stage2_scatter": c["ep_source_token_map"] is not None,
        "has_activation_scales": c["a1_scale"] is not None or c["a2_scale"] is not None,
        "has_num_local_tokens": c["num_local_tokens"] is not None,
        "_exclude_mxmoe": False,
        "_has_input_activation_scales": c["a1_scale"] is not None,
        "has_global_scale": (
            c["w1_global_scale"] is not None or c["w2_global_scale"] is not None
        ),
    }
    return args, kwargs, q_dtype_a


def problem_from_key(args, kwargs, arch=None, cu_num=None) -> MoeProblem:
    return MoeProblem.from_legacy_key(
        *args,
        **kwargs,
        arch=get_gfx_runtime() if arch is None else arch,
        cu_num=get_cu_num() if cu_num is None else cu_num,
    )


def build_problem(call, arch=None, cu_num=None) -> MoeProblem:
    """The `MoeProblem` of a `fused_moe_` call (`call` maps its parameter names)."""
    from .execute.executor import prologue
    from .policy import quant_remap

    args, kwargs, _ = dispatch_key(prologue(call, quant_remap))
    return problem_from_key(args, kwargs, arch, cu_num)


def op_call(call) -> dict | None:
    """The `fused_moe_` arguments the legacy `fused_moe` passes for `call`.

    None when the legacy `fused_moe` would not reach `fused_moe_` (shared experts go to
    FHMoE).
    """
    has_shared_expert = (
        any(
            call[name] is not None
            for name in ("shared_w1", "shared_w2", "shared_w1_scale", "shared_w2_scale")
        )
        or call["shared_expert_id"] != -1
    )
    if has_shared_expert:
        return None
    if not call["clamp_shared"]:
        raise ValueError("clamp_shared=False requires FHMoE shared-expert dispatch")
    block_size_M = call["block_size_M"] or -1
    scatter = call["stage2_scatter"]
    enable_ep_scatter = scatter is not None
    quant_type_a = call["quant_type_a"]
    copy_out = call["mxfp8_copy_out"]
    return {
        "hidden_states": call["hidden_states"],
        "w1": call["w1"],
        "w2": call["w2"],
        "topk_weight": call["topk_weight"],
        "topk_ids": call["topk_ids"],
        "expert_mask": call["expert_mask"],
        "activation": call["activation"].value,
        "quant_type": call["quant_type"].value,
        "doweight_stage1": call["doweight_stage1"],
        "w1_scale": call["w1_scale"],
        "w2_scale": call["w2_scale"],
        "a1_scale": call["a1_scale"],
        "a2_scale": call["a2_scale"],
        "block_size_M": block_size_M,
        "num_local_tokens": call["num_local_tokens"],
        "moe_sorting_dispatch_policy": call["moe_sorting_dispatch_policy"],
        "dtype": call["dtype"],
        "hidden_pad": call["hidden_pad"],
        "intermediate_pad": call["intermediate_pad"],
        "bias1": call["bias1"],
        "bias2": call["bias2"],
        "splitk": call["splitk"],
        "swiglu_limit": call["swiglu_limit"],
        "beta": call["beta"],
        "linear_beta": call["linear_beta"],
        "gate_mode": call["gate_mode"],
        "ep_arena_handle": scatter.arena_handle if enable_ep_scatter else 0,
        "ep_combine_input_offset": (
            scatter.combine_input_offset if enable_ep_scatter else 0
        ),
        "ep_slot_stride_bytes": scatter.slot_stride_bytes if enable_ep_scatter else 0,
        "ep_max_tokens_per_rank": (
            scatter.max_tokens_per_rank if enable_ep_scatter else 0
        ),
        "ep_world_size": scatter.world_size if enable_ep_scatter else 0,
        "ep_combine_quant": (
            int(scatter.combine_quant_bits) if enable_ep_scatter else 0
        ),
        "ep_source_token_map": scatter.source_token_map if enable_ep_scatter else None,
        "output": call["output"],
        "quant_type_a": None if quant_type_a is None else quant_type_a.value,
        "quant_dtype_a": call["quant_dtype_a"],
        "quant_dtype_a2": call["quant_dtype_a2"],
        "herd_topk": call["herd_topk"],
        "herd_routed_scaling_factor": call["herd_routed_scaling_factor"],
        "deterministic": call["deterministic"],
        "w1_global_scale": call["w1_global_scale"],
        "w2_global_scale": call["w2_global_scale"],
        "mxfp8_copy_payload": None if copy_out is None else copy_out[0],
        "mxfp8_copy_scale": None if copy_out is None else copy_out[1],
    }


@drop_presort_after
def fused_moe(
    hidden_states,
    w1,
    w2,
    topk_weight,
    topk_ids,
    expert_mask: torch.Tensor | None = None,
    activation=ActivationType.Silu,
    quant_type=QuantType.No,
    doweight_stage1=False,
    w1_scale: torch.Tensor | None = None,
    w2_scale: torch.Tensor | None = None,
    a1_scale: torch.Tensor | None = None,
    a2_scale: torch.Tensor | None = None,
    block_size_M=None,
    num_local_tokens: torch.Tensor | None = None,
    moe_sorting_dispatch_policy=0,
    dtype=None,
    hidden_pad=0,
    intermediate_pad=0,
    bias1=None,
    bias2=None,
    splitk=0,
    swiglu_limit=None,
    beta=None,
    linear_beta=None,
    gate_mode: str | None = "separated",
    shared_w1: torch.Tensor | None = None,
    shared_w2: torch.Tensor | None = None,
    shared_w1_scale: torch.Tensor | None = None,
    shared_w2_scale: torch.Tensor | None = None,
    shared_expert_id: int = -1,
    stage2_scatter: Stage2ScatterContext | None = None,
    output: torch.Tensor | None = None,
    quant_type_a: QuantType | None = None,
    quant_dtype_a: torch.dtype | None = None,
    quant_dtype_a2: torch.dtype | None = None,
    clamp_shared: bool = True,
    herd_topk: int = 0,
    herd_routed_scaling_factor: float = 1.0,
    deterministic: bool = False,
    w1_global_scale: torch.Tensor | None = None,
    w2_global_scale: torch.Tensor | None = None,
    mxfp8_copy_out: tuple[torch.Tensor, torch.Tensor] | None = None,
):
    """Same contract as `aiter.fused_moe.fused_moe`, served by aiter.moev2."""
    call = dict(locals())
    op = None
    for path in paths.registered().values():
        if path.entry == "op":
            op = op_call(call) if op is None else op
            path_call = op
        else:
            path_call = call
        if path_call is not None and path.claims(path_call):
            out = path.run(path_call)
            if out is not paths.DECLINED:
                return out
    raise NotImplementedError(
        "no aiter.moev2 path handles this fused_moe call yet; "
        "use aiter.fused_moe.fused_moe (the legacy code)"
    )
