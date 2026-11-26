# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Resolve a `MoeProblem` to a plan.

The order and every message are the legacy `get_2stage_cfgs` ones for the kernel families
the given segments cover: tuned lookup (with its fallbacks), checks that discard a tuned
row (logged as today), then the tuned or the default parameters. A tuned row naming a
family no given segment owns is not resolved here: the result is a decline, with nothing
logged, and the caller hands the call back to the legacy code.

Log lines are collected, not emitted, so a declined resolution prints nothing; the caller
emits them (`Resolution.emit`) once per problem, where the legacy cache would have logged.

NOTE: aiter.moev2 is under active development (MoE v2 migration, not graduated).
Please do not modify this code without coordinating with the moev2 owners (@qli88). See
aiter/moev2/README.md.
"""

import logging
from dataclasses import dataclass

from aiter import logger

from ..execute.metadata import AsmMoeMode, MOEMetadata
from ..segments.asm import stage2_block_m
from ..segments.base import Params
from ..segments.ck import _require_shuffled_gfx950_fp8
from ..segments.defaults import BLOCK_SIZE_M, get_block_size_M, use_nt
from ..segments.whole_graph import FusedMoeImplResolutionError, resolve_fused_moe_impl
from . import kernel_names, overrides, profile, ranking, tuned
from .ops import chain
from .plan import MoePlan, SegmentChoice

INFO = logging.INFO
WARNING = logging.WARNING
EXPLAIN_PREFIX = "[aiter.moe] "


@dataclass(frozen=True)
class Resolution:
    plan: MoePlan | None  # None when declined
    metadata: MOEMetadata | None = (
        None  # the plan lowered to the legacy stage callables
    )
    logs: tuple = ()  # (level, message), in legacy order
    declined: str | None = None
    explain: tuple = ()

    def emit(self):
        for line in self.explain:
            logger.info(EXPLAIN_PREFIX + line)
        for level, message in self.logs:
            if level == WARNING:
                logger.warning(message)
            else:
                logger.info(message)


def feasible_plans(problem, segments):
    """`(plans, rejected)`: segment tuples covering the op chain, and `{name: reason}`."""
    ops = chain(problem)
    rejected = {}
    usable = []
    for segment in segments:
        reason = segment.reject_reason(problem)
        if reason is None:
            usable.append(segment)
        else:
            rejected[segment.name] = reason

    def walk(i):
        if i == len(ops):
            yield ()
            return
        for segment in usable:
            start, end = segment.covers
            if start != ops[i] or end not in ops:
                continue
            for rest in walk(ops.index(end) + 1):
                yield (segment,) + rest

    return list(walk(0)), rejected


def _plan_name(structure):
    return " | ".join(s.name for s in structure)


def _owned_kernel(raw):
    """The parsed kernel name of a tuned cell, or a decline reason (str)."""
    if not isinstance(raw, str):
        return f"kernel name {raw!r} is not a string"
    if raw != raw.strip():
        return f"kernel name {raw!r} has surrounding blanks"
    kernel = kernel_names.try_parse(raw)
    if raw and kernel is None:
        return f"unknown kernel name {raw!r}"
    return kernel


def match_row(plans, row):
    """The plan structure that runs tuned `row`, or a decline reason (str)."""
    if row.get("run_1stage", False):
        return "1-stage tuned row"
    if row.get("flat", AsmMoeMode.SORTED) == AsmMoeMode.POST_REDUCE:
        return "post-reduction tuned row"
    k1 = _owned_kernel(row.get("kernelName1", ""))
    if isinstance(k1, str):
        return k1
    if k1 is not None and k1.family == kernel_names.IMPL:
        for structure in plans:
            if len(structure) == 1 and structure[0].owns(k1):
                return structure
        return f"no segment runs {k1}"
    k2 = _owned_kernel(row.get("kernelName2", ""))
    if isinstance(k2, str):
        return k2
    for structure in plans:
        gemm1 = [s for s in structure if s.covers[0].name == "GEMM1"]
        gemm2 = [s for s in structure if s.covers[0].name == "GEMM2"]
        if gemm1 and gemm2 and gemm1[0].owns(k1) and gemm2[0].owns(k2):
            return structure
    return f"no segments run kernelName1={k1}, kernelName2={k2}"


def _two_stage_plan(
    problem,
    structure,
    source,
    kernelName1,
    kernelName2,
    block_m,
    ksplit,
    nt,
    run_1stage,
):
    names = [s.name for s in structure]
    is_asm = "asm.stage1" in names
    if is_asm:
        block_m = stage2_block_m(block_m)
    choices = []
    for segment in structure:
        if segment.name in ("ck.gemm1", "asm.stage1"):
            params = Params(kernel=kernelName1, ksplit=ksplit, nt=nt)
        elif segment.name == "ck.gemm2":
            params = Params(kernel=kernelName2, nt=nt)
        else:
            params = Params()
        choices.append(SegmentChoice(segment, params))
    plan = MoePlan(
        tuple(choices),
        source,
        Params(block_m=block_m, ksplit=ksplit),
        tuple(c.segment.produces(problem, c.params) for c in choices[:-1]),
    )
    stage1 = plan.segments[names.index("asm.stage1" if is_asm else "ck.gemm1")]
    stage2 = plan.segments[names.index("ck.gemm2")]
    metadata = MOEMetadata(
        stage1.segment.build(problem, stage1.params),
        stage2.segment.build(problem, stage2.params),
        block_m,
        ksplit if is_asm else int(ksplit),
        run_1stage,
    )
    return plan, metadata


def _require_shuffled_ck_fp8(problem, logs, explain, emit):
    """Raise as the legacy code does, after the lines it has logged by then."""
    try:
        _require_shuffled_gfx950_fp8(
            problem.arch, problem.w1.dtype, problem.quant_type, problem.fully_shuffled
        )
    except RuntimeError:
        if emit:
            Resolution(None, logs=tuple(logs), explain=tuple(explain or ())).emit()
        raise


def resolve(problem, segments, emit_on_raise=True) -> Resolution:
    """The plan for `problem`, with the legacy log lines to emit.

    Where legacy raises, this emits the lines legacy logged before raising (unless
    `emit_on_raise` is False) and raises the same error.
    """
    explain = [] if overrides.explain() else None

    def note(line):
        if explain is not None:
            explain.append(line)

    note(f"problem: {problem}")
    plans, rejected = feasible_plans(problem, segments)
    note(f"feasible structures ({len(plans)}):")
    for i, structure in enumerate(plans, 1):
        note(f"   S{i} {_plan_name(structure)}")
    if rejected:
        note("rejected: " + "; ".join(f"{k} ({v})" for k, v in rejected.items()))

    def declined(reason):
        note(f"declined: {reason}; the legacy code runs this call")
        return Resolution(None, declined=reason, explain=tuple(explain or ()))

    if not plans:
        return declined("no feasible plan")
    keys = problem.log_keys()
    cfg = tuned.lookup(problem)
    if cfg is None:
        note(f"tuned lookup: miss (tier {problem.shape.m})")
        matched = None
    else:
        note(
            f"tuned lookup: kernelName1={cfg.get('kernelName1')!r} "
            f"kernelName2={cfg.get('kernelName2')!r}"
        )
        matched = match_row(plans, cfg)
        if isinstance(matched, str):
            return declined(matched)
    logs = []
    bypass_tuned_config = overrides.bypass_tuned()
    kernel_name1 = (
        str(cfg.get("kernelName1", "") or "").strip() if cfg is not None else ""
    )
    try:
        full_impl = (
            None if bypass_tuned_config else resolve_fused_moe_impl(kernel_name1)
        )
    except FusedMoeImplResolutionError as error:
        logs.append((WARNING, f"[fused_moe] {error}; using default heuristics."))
        cfg = None
        full_impl = None
    if full_impl is not None:
        rejection = matched[0].check_row(problem, cfg, kernel_name1)
        if rejection is not None:
            if rejection.warning is not None:
                logs.append((WARNING, rejection.warning))
            note(f"tuned row discarded: {rejection.warning or 'expert_mask'}")
            cfg = None
            full_impl = None
    if full_impl is not None:
        block_m = int(cfg.get("block_m", BLOCK_SIZE_M))
        ksplit = int(cfg.get("ksplit", 0))
        logs.append((INFO, f"[fused_moe] using {kernel_name1!r} for {keys}"))
        whole = matched[0]
        plan = MoePlan(
            (SegmentChoice(whole, Params(kernel=kernel_name1, impl=full_impl)),),
            "tuned",
            Params(block_m=block_m, ksplit=ksplit),
        )
        note(f"selected {plan.describe()}")
        metadata = MOEMetadata(None, None, block_m, ksplit, full_impl=full_impl)
        return Resolution(plan, metadata, tuple(logs), explain=tuple(explain or ()))
    for _row in profile.exact(problem):
        pass  # no profile files are read yet

    s = problem.shape
    q_type = problem.quant_type
    if cfg is None or bypass_tuned_config:
        best = ranking.best(problem, ranking.rankable(plans))
        if best is None:
            return declined("no default ranking entry applies")
        region, structure = best
        ksplit = 0
        kernelName1 = ""
        kernelName2 = ""
        run_1stage = False
        run_1stage_xbf16 = False
        block_m = get_block_size_M(s.m, s.topk, s.experts, s.inter_dim)
        use_non_temporal_load = use_nt(s.m, s.topk, s.experts)
        logs.append(
            (
                INFO,
                f"run_1stage = {run_1stage}, xbf16 = {run_1stage_xbf16}, ksplit = {ksplit} q_type = {q_type} block_m = {block_m} use_nt = {use_non_temporal_load}, estimated_m_per_expert = {s.m * s.topk // s.experts}",
            )
        )
        source = "override" if cfg is not None else "default"
        note(f"ranking: {_plan_name(structure)} ({region})")
    else:
        structure = matched
        tuned_names = (
            str(cfg.get("kernelName1", "") or ""),
            str(cfg.get("kernelName2", "") or ""),
        )
        if not cfg.get("run_1stage", False) and any(
            "moe_ck2stages" in name for name in tuned_names
        ):
            _require_shuffled_ck_fp8(problem, logs, explain, emit_on_raise)
        use_non_temporal_load = False
        block_m = cfg["block_m"]
        nt_override = overrides.nt_override()
        if nt_override != -1:
            use_non_temporal_load = bool(nt_override)
        elif "nt" in cfg:
            try:
                use_non_temporal_load = bool(int(float(cfg["nt"])))
            except (TypeError, ValueError):
                # blank or malformed column: keep the pre-column behaviour
                use_non_temporal_load = False
        ksplit = 0 if overrides.ksplit_disabled() else cfg["ksplit"]
        kernelName1 = cfg["kernelName1"]
        kernelName2 = cfg["kernelName2"]
        run_1stage = cfg.get("run_1stage", False)
        if not problem.any_shuffled and not run_1stage:
            message = (
                f"[fused_moe] tuned config found for {keys} but is_shuffled=False. "
                "Tuned kernels are optimized for preshuffled weights (preshuffle_on). "
                "Running with preshuffle_off may produce incorrect results."
            )
            logs.append((WARNING, message))
        source = "tuned"
    tag = f"({kernelName1=}, {kernelName2=})"
    logs.append(
        (
            INFO,
            f"[fused_moe] using 2stage {'default' if cfg is None else tag} for {keys} ",
        )
    )
    names = [s.name for s in structure]
    if "ck.gemm1" in names and "ck.gemm2" in names:
        _require_shuffled_ck_fp8(problem, logs, explain, emit_on_raise)
    plan, metadata = _two_stage_plan(
        problem,
        structure,
        source,
        kernelName1,
        kernelName2,
        block_m,
        ksplit,
        use_non_temporal_load,
        run_1stage,
    )
    note(f"selected {plan.describe()}")
    return Resolution(plan, metadata, tuple(logs), explain=tuple(explain or ()))
