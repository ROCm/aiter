# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Path `a8w8_pt`: fp8 activations and weights, per-tensor scales.

Claims `fused_moe_` calls with per-tensor fp8 weights (the arch's fp8) on gfx942/gfx950,
Silu or Gelu, separated gate/up, no bias, swiglu_limit, beta or linear_beta, padding,
scatter, caller-chosen activation dtypes, HERD routing, deterministic mode or registered
presort, outside AITER_ONLINE_TUNE. Today these resolve to CK 2-stage (default and most
tuned rows), ASM stage 1 + CK stage 2, or the FlyDSL gfx942 whole graph; any other tuned
row is declined back to the legacy code, and so is a call whose second lookup after the
sort raises or is declined. int8 per-tensor stays on the legacy code.
"""

import functools
from dataclasses import replace

from aiter import ActivationType, QuantType, dtypes, presorted_moe
from aiter.jit.utils.chip_info import get_cu_num, get_gfx, get_gfx_runtime
from aiter.ops.flydsl.moe_common import GateMode

from .. import api, paths
from ..dispatch import overrides, resolver, tuned
from ..execute import executor
from ..policy import quant_remap
from ..segments.asm import AsmStage1
from ..segments.ck import CkGemm1, CkGemm2
from ..segments.quant import QuantA1, QuantInter
from ..segments.sort import Sort
from ..segments.whole_graph import WholeGraph
from ..shapes import get_inter_dim, get_padded_M

NAME = "a8w8_pt"
ARCHS = ("gfx942", "gfx950")
SEGMENTS = (
    Sort(),
    QuantA1(),
    CkGemm1(),
    AsmStage1(),
    QuantInter(),
    CkGemm2(),
    WholeGraph(),
)
# After sorting, fused_moe_2stages looks the key up again without these, on the
# hidden_states rows. _fused_moe_impl's _metadata_transform then replaces the result
# with the first lookup's, so the second lookup only logs, or raises.
_RELOOKUP_DROPS = frozenset({"act_scales", "num_local_tokens", "scatter"})


def claims(call) -> bool:
    """Cheap checks on the `fused_moe_` arguments; `run` may still decline."""
    if overrides.online_tune():
        return False
    if call["herd_topk"] or call["deterministic"]:
        return False
    # The legacy code consumes a registered presort on every call; this path does not.
    if getattr(presorted_moe._PRESORTED_MOE, "aux", None) is not None:
        return False
    if QuantType(call["quant_type"]) != QuantType.per_Tensor:
        return False
    quant_type_a = call["quant_type_a"]
    if quant_type_a is not None and QuantType(quant_type_a) != QuantType.per_Tensor:
        return False
    if any(
        call[name] is not None
        for name in (
            "quant_dtype_a",
            "quant_dtype_a2",
            "ep_source_token_map",
            "bias1",
            "bias2",
            "swiglu_limit",
            "beta",
            "linear_beta",
        )
    ):
        return False
    if call["hidden_pad"] or call["intermediate_pad"]:
        return False
    if GateMode(call["gate_mode"]) != GateMode.SEPARATED:
        return False
    if ActivationType(call["activation"]) not in (
        ActivationType.Silu,
        ActivationType.Gelu,
    ):
        return False
    if get_gfx() not in ARCHS or get_gfx_runtime() not in ARCHS:
        return False
    w1, w2 = call["w1"], call["w2"]
    if (
        w1.dtype != dtypes.fp8
        or w2.dtype != dtypes.fp8
        or w1.dim() != 3
        or w2.dim() != 3
    ):
        return False
    _, _, inter_dim = get_inter_dim(w1.shape, w2.shape)
    if w1.shape[1] not in (inter_dim, 2 * inter_dim):
        return False
    hidden_states = call["hidden_states"]
    dtype = hidden_states.dtype if call["dtype"] is None else call["dtype"]
    return dtype in (dtypes.bf16, dtypes.fp16) and call["topk_ids"].dim() == 2


@tuned.cached(maxsize=2048)
def _peek(problem):
    """The resolution, with nothing emitted; None where resolving raises."""
    try:
        return resolver.resolve(problem, SEGMENTS, emit_on_raise=False)
    except Exception:  # noqa: BLE001 - callers decline, or resolve again to raise
        return None


@tuned.cached(maxsize=2048)
def _resolve(problem, relookup):
    resolution = _peek(problem)
    if resolution is None:
        resolution = resolver.resolve(problem, SEGMENTS)  # raises after the logs
    resolution.emit()
    return resolution


def _relookup_problem(problem, token):
    shape = replace(problem.shape, m=token)
    return replace(
        problem.with_features(remove=_RELOOKUP_DROPS), inter=None, shape=shape
    )


@tuned.cached(maxsize=2048)
def _relookup(problem, token):
    """The second lookup, keyed on the `hidden_states` rows rather than `topk_ids`."""
    return _resolve(_relookup_problem(problem, token), True)


def _relookup_decline(problem, token):
    """Why the second lookup sends the call back to legacy, or None.

    Legacy raises there after the sort; v2 cannot emit the lines of a lookup it declines.
    """
    first = _peek(problem)
    if first is None or first.plan is None or first.metadata.full_impl is not None:
        return None
    second = _peek(_relookup_problem(problem, token))
    if second is None:
        return "the lookup after the sort raises"
    if second.plan is None:
        return f"the lookup after the sort declines ({second.declined})"
    return None


@tuned.cached(maxsize=2048)
def _dispatch(args, kwargs_items, arch, cu_num, token):
    """`(resolution, problem)` for a key, or None when declined.

    `token` is the bucketed `hidden_states` row count. `problem` is None when no second
    lookup follows (whole-graph plans).
    """
    problem = api.problem_from_key(args, dict(kwargs_items), arch, cu_num)
    reason = _relookup_decline(problem, token)
    if reason is not None:
        if overrides.explain():
            line = f"declined: {reason}; the legacy code runs this call"
            resolver.Resolution(None, declined=reason, explain=(line,)).emit()
        return None
    primary = _resolve(problem, False)
    if primary.plan is None:
        return None
    if primary.metadata.full_impl is not None:
        return primary, None
    return primary, problem


def run(call):
    c = executor.prologue(call, quant_remap)
    args, kwargs, q_dtype_a = api.dispatch_key(c)
    token = get_padded_M(c["hidden_states"].shape[0])
    dispatched = _dispatch(
        args, tuple(kwargs.items()), get_gfx_runtime(), get_cu_num(), token
    )
    if dispatched is None:
        return paths.DECLINED
    resolution, problem = dispatched
    after_sort = None
    if problem is not None:
        after_sort = functools.partial(_relookup, problem, token)
    return executor.run_two_stage(
        c, resolution.metadata, resolution.plan, q_dtype_a, after_sort=after_sort
    )


def dispatch_record(args, kwargs):
    """The v2 `MOEMetadata` for a legacy key (logs emitted), or None if not taken."""
    problem = api.problem_from_key(args, kwargs)
    if problem.activation.swiglu_limit is not None or problem.inter is not None:
        return None
    if (problem.activation.situ_beta, problem.activation.situ_linear_beta) != (
        1.0,
        1.0,
    ):
        return None
    if _relookup_decline(problem, problem.shape.m) is not None:
        return None
    resolution = resolver.resolve(problem, SEGMENTS)
    resolution.emit()
    return None if resolution.plan is None else resolution.metadata


# Not graduated: runs only when enabled with AITER_MOEV2=a8w8_pt (or `on`).
PATH = paths.Path(
    NAME,
    claims,
    run,
    dispatch_record=dispatch_record,
    entry="op",
    segments=SEGMENTS,
)
paths.register(PATH)
