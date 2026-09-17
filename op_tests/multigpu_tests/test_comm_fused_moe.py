# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""TP8 correctness and performance smoke test for communication-fused MoE.

Run with:
    torchrun --standalone --nproc_per_node=8 \
        op_tests/multigpu_tests/test_comm_fused_moe.py

The production target is the DeepSeek-V4-Pro TP8 Stage2 shape:

* hidden = 7168
* local intermediate = 3072 / TP8 = 384
* routed experts = 384
* top-k = 6

The test compares both the ordinary and communication-fused paths with an
independent torch reference built from the unshuffled MXFP8/MXFP4 inputs.  It
also exercises eager execution, graph replay, the model-facing runtime,
separate-stream execution, and runtime padding from M=3 to M=4.
"""

from __future__ import annotations

import argparse
import itertools
import os
from collections.abc import Callable
from dataclasses import dataclass, field

import pandas as pd
import torch
import torch.distributed as dist
import torch.nn.functional as F

import aiter
from aiter import ActivationType, QuantType, dtypes
from aiter.dist.parallel_state import (
    ensure_model_parallel_initialized,
    get_dcp_group,
    get_dp_group,
    get_ep_group,
    get_pcp_group,
    get_pp_group,
    get_tp_group,
    init_distributed_environment,
    set_custom_all_reduce,
)
from aiter.fused_moe import (
    fused_moe,
    get_2stage_cfgs,
    get_padded_M,
    moe_sorting,
    stage2_uses_route_reduce,
    torch_moe_stage1,
    torch_moe_stage2,
)
from aiter.jit.utils.chip_info import get_cu_num, get_gfx_runtime
from aiter.ops.comm_fused_moe_runtime import CommFusedMoeRuntime
from aiter.ops.flydsl.comm_fused_moe_host import (
    ShapeKey,
    config_name,
    create_flydsl_comm_fused_runners,
    is_flydsl_comm_fused_moe_available,
    winners_for,
)
from aiter.ops.quant import (
    mxfp4_moe_sort_fwd,
    per_1x32_f4_quant,
    per_1x32_f8_scale_f8_quant,
)
from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight_a16w4
from aiter.test_common import benchmark, checkAllclose, run_perftest

SUPPORTED_GFX = ("gfx950",)
TP_SIZE = 8
MODEL_DIM = 7168
INTER_DIM = 384
EXPERTS = 384
TOPK = 6
PRODUCTION_TOKENS = (1, 2, 4, 8, 16)
ROUTES = ("uniform", "skew")
MODES = ("eager", "graph")

# Accuracy limits used by the production tuner and host dispatch.
MAX_ABS = 1.0
MAX_REL_L2 = 0.05
MAX_ERR_RATIO = 0.05

# Performance is reported for visibility only; it is not a CI pass/fail gate.
PERF_WARMUP = 20
PERF_ITERS = 10


@dataclass(slots=True)
class Stage2Weights:
    kernel: torch.Tensor
    kernel_scale: torch.Tensor
    reference: torch.Tensor
    reference_scale: torch.Tensor


@dataclass(slots=True)
class Stage2Case:
    tokens: int
    block_m: int
    inter_states: torch.Tensor
    a2_scale: torch.Tensor
    reference_a2_scale: torch.Tensor
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor
    sorted_token_ids: torch.Tensor
    sorted_expert_ids: torch.Tensor
    sorted_weights: torch.Tensor
    num_valid_ids: torch.Tensor
    partial_out: torch.Tensor


@dataclass(slots=True)
class Stage2Fixture:
    case: Stage2Case
    metadata: object
    ordinary_kernel: str
    requires_output_zero: bool
    shared_partial: torch.Tensor
    reference: torch.Tensor


@dataclass(slots=True)
class FullMoeCase:
    hidden_states: torch.Tensor
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor
    w1: torch.Tensor
    w1_scale: torch.Tensor
    reference_w1: torch.Tensor
    reference_w1_scale: torch.Tensor


@dataclass(slots=True)
class FullMoeFixture:
    case: FullMoeCase
    shared_partial: torch.Tensor
    reference: torch.Tensor
    stage2_stream: torch.cuda.Stream


@dataclass(slots=True)
class TestSession:
    rank: int
    world: int
    device: torch.device
    group: object
    gfx: str
    runners: dict
    weights: Stage2Weights
    graph_replays: int
    stage2_fixtures: dict[tuple[int, str], Stage2Fixture] = field(default_factory=dict)
    full_fixture: FullMoeFixture | None = None


# Keep distributed state out of the @benchmark signatures so the summary table
# contains only real sweep axes (tokens, route, and execution mode).
_ACTIVE_SESSION: TestSession | None = None


def _session() -> TestSession:
    if _ACTIVE_SESSION is None:
        raise RuntimeError("comm-fused MoE test session is not initialized")
    return _ACTIVE_SESSION


def _setup_distributed(expected_world: int):
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    if world != expected_world:
        raise ValueError(
            f"torchrun world size is {world}, but expected {expected_world}"
        )
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    set_custom_all_reduce(True)
    init_distributed_environment(
        world_size=world,
        rank=rank,
        local_rank=local_rank,
        distributed_init_method="env://",
        backend="nccl",
    )
    ensure_model_parallel_initialized(world, 1)
    group = get_tp_group().device_group
    dist.all_reduce(torch.zeros(1, dtype=torch.int32, device=device), group=group)
    torch.cuda.synchronize(device)
    return rank, world, device, group


def _barrier(group) -> None:
    torch.cuda.synchronize()
    dist.barrier(group=group)


def _cleanup_distributed(rank: int) -> None:
    if rank == 0:
        aiter.logger.info("releasing distributed communicators")
    for get_group in (
        get_tp_group,
        get_pcp_group,
        get_pp_group,
        get_dp_group,
        get_ep_group,
        get_dcp_group,
    ):
        group = get_group()
        communicator = group.device_communicator
        if communicator is not None:
            communicator.destroy()
            group.device_communicator = None
        group.mq_broadcaster = None
    dist.destroy_process_group()


def _make_routes(tokens: int, route: str, device):
    token = torch.arange(tokens, dtype=torch.int64, device=device)[:, None]
    slot = torch.arange(TOPK, dtype=torch.int64, device=device)[None, :]
    if route == "uniform":
        topk_ids = (token * TOPK + slot) % EXPERTS
    elif route == "skew":
        topk_ids = 1 + ((token + slot - 1) % (EXPERTS - 1))
        hot_slot = (token % 8) != 7
        cold_slot = 1 + ((token + TOPK - 1) % (EXPERTS - 1))
        topk_ids[:, :1] = torch.where(hot_slot, 0, cold_slot)
    else:
        raise ValueError(f"unsupported route pattern: {route}")
    weights = (slot + 1).expand(tokens, -1).to(torch.float32)
    return (
        topk_ids.to(torch.int32).contiguous(),
        (weights / weights.sum(dim=1, keepdim=True)).contiguous(),
    )


def _make_stage2_weights(rank: int, device) -> Stage2Weights:
    """Create one deterministic W2 shared by every token/route case."""

    generator = torch.Generator(device=device).manual_seed(20260902 + 1000 + rank)

    def quantize_expert():
        weight = torch.randn(
            (MODEL_DIM, INTER_DIM),
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        ).mul_(INTER_DIM**-0.25)
        return per_1x32_f4_quant(weight, quant_dtype=dtypes.fp4x2)

    first_quant, first_scale = quantize_expert()
    reference = torch.empty(
        (EXPERTS, *first_quant.shape), dtype=first_quant.dtype, device=device
    )
    reference_scale = torch.empty(
        (EXPERTS, *first_scale.shape), dtype=first_scale.dtype, device=device
    )
    reference[0].copy_(first_quant)
    reference_scale[0].copy_(first_scale)
    for expert in range(1, EXPERTS):
        quant, scale = quantize_expert()
        reference[expert].copy_(quant)
        reference_scale[expert].copy_(scale)

    kernel = shuffle_weight_a16w4(reference, 16, False).contiguous()
    kernel_scale = shuffle_scale_a16w4(
        reference_scale.view(-1, reference_scale.shape[-1]), EXPERTS, False
    ).contiguous()
    return Stage2Weights(kernel, kernel_scale, reference, reference_scale)


def _resolve_ordinary_stage2(tokens: int):
    metadata = get_2stage_cfgs(
        get_padded_M(tokens),
        MODEL_DIM,
        INTER_DIM,
        EXPERTS,
        TOPK,
        dtypes.bf16,
        dtypes.fp8,
        dtypes.fp4x2,
        QuantType.per_1x32,
        True,
        ActivationType.Silu,
        False,
        0,
        0,
        is_shuffled=True,
        gate_mode="separated",
        is_ep=False,
        has_stage2_bias=False,
        opus_weights_shuffled=True,
    )
    kernel_name = str(
        getattr(metadata.stage2, "keywords", {}).get("kernelName")
        or getattr(metadata.stage2, "keywords", {}).get("kernelName2")
        or ""
    )
    return metadata, kernel_name, not stage2_uses_route_reduce(metadata.stage2)


def _make_stage2_case(
    tokens: int,
    route: str,
    block_m: int,
    requires_output_zero: bool,
    rank: int,
    device,
) -> Stage2Case:
    topk_ids, topk_weights = _make_routes(tokens, route, device)
    (
        sorted_ids,
        sorted_weights,
        sorted_expert_ids,
        num_valid_ids,
        sorting_out,
    ) = moe_sorting(
        topk_ids,
        topk_weights,
        EXPERTS,
        MODEL_DIM,
        torch.bfloat16,
        block_m,
        accumulate=requires_output_zero,
    )
    sorted_ids = sorted_ids.to(torch.int32).contiguous()
    sorted_weights = sorted_weights.to(torch.float32).contiguous()
    sorted_expert_ids = sorted_expert_ids.to(torch.int32).contiguous()
    num_valid_ids = num_valid_ids.to(torch.int32).contiguous()
    row_capacity = int(sorted_expert_ids.numel()) * block_m
    if sorted_ids.numel() < row_capacity:
        sentinel = (TOPK << 24) | tokens
        sorted_ids = F.pad(
            sorted_ids, (0, row_capacity - sorted_ids.numel()), value=sentinel
        ).contiguous()
        sorted_weights = F.pad(
            sorted_weights,
            (0, row_capacity - sorted_weights.numel()),
            value=0.0,
        ).contiguous()

    seed = 20260902 + 2 * tokens + int(route == "skew") + rank
    generator = torch.Generator(device=device).manual_seed(seed)
    activations = torch.randn(
        (tokens, TOPK, INTER_DIM),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    ).mul_(INTER_DIM**-0.25)
    inter_states, reference_a2_scale = per_1x32_f8_scale_f8_quant(
        activations,
        quant_dtype=dtypes.fp8,
        scale_type=dtypes.fp8_e8m0,
    )
    a2_scale = mxfp4_moe_sort_fwd(
        reference_a2_scale.view(tokens * TOPK, -1),
        sorted_ids=sorted_ids,
        num_valid_ids=num_valid_ids,
        token_num=tokens,
        cols=INTER_DIM,
    ).contiguous()
    partial_out = (
        sorting_out
        if sorting_out.numel()
        else torch.empty((tokens, MODEL_DIM), dtype=torch.bfloat16, device=device)
    )
    return Stage2Case(
        tokens=tokens,
        block_m=block_m,
        inter_states=inter_states.view(tokens, TOPK, INTER_DIM),
        a2_scale=a2_scale,
        reference_a2_scale=reference_a2_scale,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        sorted_token_ids=sorted_ids,
        sorted_expert_ids=sorted_expert_ids,
        sorted_weights=sorted_weights,
        num_valid_ids=num_valid_ids,
        partial_out=partial_out,
    )


def _shared_partial(tokens: int, rank: int, device):
    token_term = (
        torch.arange(tokens, device=device, dtype=torch.float32)
        .remainder(7)
        .mul_(1.0 / 32.0)
        .view(-1, 1)
    )
    column_term = (
        torch.arange(MODEL_DIM, device=device, dtype=torch.float32)
        .remainder(17)
        .mul_(1.0 / 128.0)
        .view(1, -1)
    )
    return (token_term + column_term + float(rank + 1) / 16.0).to(torch.bfloat16)


@torch.no_grad()
def _torch_stage2_partial(
    inter_states: torch.Tensor,
    a2_scale: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    weights: Stage2Weights,
) -> torch.Tensor:
    # torch_moe_stage2 only reads w1.shape to infer dimensions. Use the real
    # packed-MXFP4 shape on a meta tensor, avoiding unused Stage1 storage.
    w1_shape = torch.empty((EXPERTS, INTER_DIM * 2, MODEL_DIM // 2), device="meta")
    return torch_moe_stage2(
        inter_states,
        w1_shape,
        weights.reference,
        topk_weights,
        topk_ids,
        dtype=torch.bfloat16,
        quant_type=QuantType.per_1x32,
        w2_scale=weights.reference_scale,
        a2_scale=a2_scale,
        doweight=True,
    )


@torch.no_grad()
def _torch_stage2_allreduce(
    case: Stage2Case,
    weights: Stage2Weights,
    shared_partial: torch.Tensor,
    group,
) -> torch.Tensor:
    output = _torch_stage2_partial(
        case.inter_states,
        case.reference_a2_scale,
        case.topk_ids,
        case.topk_weights,
        weights,
    )
    output.add_(shared_partial)
    dist.all_reduce(output, group=group)
    return output


def _run_ordinary_stage2(
    fixture: Stage2Fixture, weights: Stage2Weights
) -> torch.Tensor:
    case = fixture.case
    if fixture.requires_output_zero:
        case.partial_out.zero_()
    fixture.metadata.stage2(
        case.inter_states,
        None,
        weights.kernel,
        case.sorted_token_ids,
        case.sorted_expert_ids,
        case.num_valid_ids,
        case.partial_out,
        TOPK,
        w2_scale=weights.kernel_scale.view(dtypes.fp8_e8m0),
        a2_scale=case.a2_scale,
        block_m=case.block_m,
        sorted_weights=case.sorted_weights,
    )
    case.partial_out.add_(fixture.shared_partial)
    return get_tp_group().all_reduce(case.partial_out, ca_fp8_quant=False)


def _stage2_fixture(session: TestSession, tokens: int, route: str) -> Stage2Fixture:
    key = (tokens, route)
    if key in session.stage2_fixtures:
        return session.stage2_fixtures[key]

    metadata, kernel_name, requires_zero = _resolve_ordinary_stage2(tokens)
    case = _make_stage2_case(
        tokens,
        route,
        int(metadata.block_m),
        requires_zero,
        session.rank,
        session.device,
    )
    shared = _shared_partial(tokens, session.rank, session.device)
    reference = _torch_stage2_allreduce(case, session.weights, shared, session.group)
    fixture = Stage2Fixture(
        case=case,
        metadata=metadata,
        ordinary_kernel=kernel_name,
        requires_output_zero=requires_zero,
        shared_partial=shared,
        reference=reference,
    )
    session.stage2_fixtures[key] = fixture
    return fixture


def _make_full_moe_case(rank: int, device) -> FullMoeCase:
    tokens = 3
    generator = torch.Generator(device=device).manual_seed(20260903 + rank)
    hidden_states = torch.randn(
        (tokens, MODEL_DIM),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    ).mul_(MODEL_DIM**-0.25)
    topk_ids, topk_weights = _make_routes(tokens, "skew", device)

    # Only routed experts are initialized. Both the production kernel and torch
    # reference are forbidden from reading any other expert.
    routed_experts = {int(expert) for expert in topk_ids.unique().tolist()}
    if 0 not in routed_experts:
        raise AssertionError("skew route must include the hot expert")

    def quantize_expert():
        weight = torch.randn(
            (INTER_DIM * 2, MODEL_DIM),
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        ).mul_(MODEL_DIM**-0.25)
        return per_1x32_f4_quant(weight, quant_dtype=dtypes.fp4x2)

    first_quant, first_scale = quantize_expert()
    reference_w1 = torch.empty(
        (EXPERTS, *first_quant.shape), dtype=first_quant.dtype, device=device
    )
    reference_w1_scale = torch.empty(
        (EXPERTS, *first_scale.shape), dtype=first_scale.dtype, device=device
    )
    reference_w1[0].copy_(first_quant)
    reference_w1_scale[0].copy_(first_scale)
    for expert in sorted(routed_experts - {0}):
        quant, scale = quantize_expert()
        reference_w1[expert].copy_(quant)
        reference_w1_scale[expert].copy_(scale)

    w1 = shuffle_weight_a16w4(reference_w1, 16, True).contiguous()
    w1_scale = shuffle_scale_a16w4(
        reference_w1_scale.view(-1, reference_w1_scale.shape[-1]), EXPERTS, True
    ).contiguous()
    return FullMoeCase(
        hidden_states=hidden_states,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        w1=w1,
        w1_scale=w1_scale,
        reference_w1=reference_w1,
        reference_w1_scale=reference_w1_scale,
    )


@torch.no_grad()
def _torch_full_moe_allreduce(
    case: FullMoeCase,
    weights: Stage2Weights,
    shared_partial: torch.Tensor,
    group,
) -> torch.Tensor:
    stage1 = torch_moe_stage1(
        case.hidden_states,
        case.reference_w1,
        weights.reference,
        case.topk_weights,
        case.topk_ids,
        dtype=torch.bfloat16,
        activation=ActivationType.Silu,
        quant_type=QuantType.per_1x32,
        w1_scale=case.reference_w1_scale,
        doweight=False,
    )

    a2_q, a2_scale = per_1x32_f8_scale_f8_quant(
        stage1,
        quant_dtype=dtypes.fp8,
        scale_type=dtypes.fp8_e8m0,
    )
    output = _torch_stage2_partial(
        a2_q,
        a2_scale,
        case.topk_ids,
        case.topk_weights,
        weights,
    )
    output.add_(shared_partial)
    dist.all_reduce(output, group=group)
    return output


def _full_moe_fixture(session: TestSession) -> FullMoeFixture:
    if session.full_fixture is not None:
        return session.full_fixture
    case = _make_full_moe_case(session.rank, session.device)
    shared = _shared_partial(3, session.rank, session.device)
    reference = _torch_full_moe_allreduce(case, session.weights, shared, session.group)
    session.full_fixture = FullMoeFixture(
        case=case,
        shared_partial=shared,
        reference=reference,
        stage2_stream=torch.cuda.Stream(device=session.device),
    )
    return session.full_fixture


def _accuracy_metrics(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    session: TestSession,
    label: str,
) -> dict[str, float]:
    if actual.shape != expected.shape or actual.dtype != torch.bfloat16:
        raise AssertionError(
            f"{label}: expected BF16 shape {tuple(expected.shape)}, got "
            f"dtype={actual.dtype} shape={tuple(actual.shape)}"
        )

    actual_f32 = actual.to(dtypes.fp32)
    expected_f32 = expected.to(dtypes.fp32)
    diff = actual_f32 - expected_f32
    finite = torch.tensor(
        int(torch.isfinite(actual_f32).all() and torch.isfinite(expected_f32).all()),
        dtype=torch.int32,
        device=session.device,
    )
    stats = torch.stack(
        (
            diff.abs().max(),
            diff.norm() / expected_f32.norm().clamp_min(1.0e-12),
        )
    )
    dist.all_reduce(finite, op=dist.ReduceOp.MIN, group=session.group)
    dist.all_reduce(stats, op=dist.ReduceOp.MAX, group=session.group)

    local_err = checkAllclose(
        expected_f32,
        actual_f32,
        rtol=0.05,
        atol=MAX_ABS,
        tol_err_ratio=MAX_ERR_RATIO,
        msg=f"{label}: ",
        printLog=session.rank == 0,
    )
    err = torch.tensor(float(local_err), dtype=torch.float32, device=session.device)
    dist.all_reduce(err, op=dist.ReduceOp.MAX, group=session.group)

    max_abs, rel_l2, err_ratio = (
        float(stats[0].item()),
        float(stats[1].item()),
        float(err.item()),
    )
    if (
        not int(finite.item())
        or max_abs > MAX_ABS
        or rel_l2 > MAX_REL_L2
        or err_ratio > MAX_ERR_RATIO
    ):
        raise AssertionError(
            f"{label}: finite={bool(finite.item())} max_abs={max_abs:.6f} "
            f"rel_l2={rel_l2:.6f} err={err_ratio:.6f}"
        )
    return {"err": err_ratio, "max_abs": max_abs, "rel_l2": rel_l2}


def _capture_graph(run: Callable[[], torch.Tensor], group):
    _barrier(group)
    run()
    _barrier(group)
    graph = torch.cuda.CUDAGraph()
    with get_tp_group().graph_capture() as capture, torch.cuda.graph(
        graph, stream=capture.stream
    ):
        output = run()
    _barrier(group)
    return graph, output


def _rank_max(value: float, session: TestSession) -> float:
    result = torch.tensor(float(value), dtype=torch.float32, device=session.device)
    dist.all_reduce(result, op=dist.ReduceOp.MAX, group=session.group)
    return float(result.item())


def _measure_candidate(
    *,
    name: str,
    run: Callable[[], torch.Tensor],
    mode: str,
    reference: torch.Tensor,
    session: TestSession,
) -> tuple[float, dict[str, float]]:
    if mode == "graph":
        graph, output = _capture_graph(run, session.group)
        timed = graph.replay
    else:
        graph = None
        output = None
        timed = run

    _barrier(session.group)
    _, local_us = run_perftest(
        timed,
        num_iters=PERF_ITERS,
        num_warmup=PERF_WARMUP,
        use_cuda_event=True,
    )
    _barrier(session.group)
    latency_us = _rank_max(local_us, session)

    samples = []
    replays = session.graph_replays if graph is not None else 1
    for replay in range(replays):
        if graph is None:
            actual = run()
        else:
            # Never accept the output left by capture or an earlier replay. This
            # catches missing child-stream launches and partially captured graphs.
            output.fill_(float("nan"))
            torch.cuda.synchronize(session.device)
            graph.replay()
            actual = output
        _barrier(session.group)
        samples.append(
            _accuracy_metrics(
                actual,
                reference,
                session=session,
                label=f"{name} {mode} replay={replay + 1}",
            )
        )

    return latency_us, {
        metric: max(sample[metric] for sample in samples)
        for metric in ("err", "max_abs", "rel_l2")
    }


def _stage2_work(case: Stage2Case, world: int) -> tuple[int, int]:
    """Return cluster-wide GEMM FLOPs and algorithmic input/output bytes."""

    active_experts = int(torch.unique(case.topk_ids).numel())
    flops = 2 * case.tokens * TOPK * MODEL_DIM * INTER_DIM * world
    per_rank_bytes = (
        case.tokens * TOPK * (INTER_DIM + INTER_DIM // 32)
        + active_experts * MODEL_DIM * (INTER_DIM // 2 + INTER_DIM // 32)
        + case.tokens * TOPK * (4 + 4)
        + case.tokens * MODEL_DIM * (2 + 2)
    )
    return flops, per_rank_bytes * world


def _full_moe_work(case: FullMoeCase, world: int) -> tuple[int, int]:
    """Return cluster-wide Stage1+Stage2 FLOPs and algorithmic bytes."""

    tokens = case.hidden_states.shape[0]
    active_experts = int(torch.unique(case.topk_ids).numel())
    flops = 6 * tokens * TOPK * MODEL_DIM * INTER_DIM * world
    per_rank_bytes = (
        tokens * MODEL_DIM * 2
        + active_experts
        * (
            INTER_DIM * 2 * (MODEL_DIM // 2 + MODEL_DIM // 32)
            + MODEL_DIM * (INTER_DIM // 2 + INTER_DIM // 32)
        )
        + tokens * TOPK * (4 + 4)
        + tokens * MODEL_DIM * (2 + 2)
    )
    return flops, per_rank_bytes * world


def _record_candidates(
    *,
    candidates: dict[str, Callable[[], torch.Tensor]],
    mode: str,
    reference: torch.Tensor,
    flops: int,
    nbytes: int,
    session: TestSession,
) -> dict[str, float | str]:
    ret: dict[str, float | str] = {"gfx": session.gfx}
    for name, run in candidates.items():
        latency_us, accuracy = _measure_candidate(
            name=name,
            run=run,
            mode=mode,
            reference=reference,
            session=session,
        )
        ret[f"{name} us"] = latency_us
        ret[f"{name} TFLOPS"] = flops / latency_us / 1.0e6
        ret[f"{name} TB/s"] = nbytes / latency_us / 1.0e6
        ret[f"{name} err"] = accuracy["err"]
        ret[f"{name} max_abs"] = accuracy["max_abs"]
        ret[f"{name} rel_l2"] = accuracy["rel_l2"]
    return ret


def _run_stage2_case(
    session: TestSession, tokens: int, route: str, mode: str
) -> dict[str, float | str]:
    fixture = _stage2_fixture(session, tokens, route)
    runner = session.runners[tokens]
    if getattr(runner.config, "collective", None) != "direct":
        raise AssertionError(
            f"M={tokens} expected direct collective, got {runner.config!r}"
        )

    def run_ordinary():
        return _run_ordinary_stage2(fixture, session.weights)

    def run_comm_fused():
        case = fixture.case
        prepared = runner.prepare_shared_partial(fixture.shared_partial)
        return runner(
            stage2_args=(
                case.inter_states,
                None,
                session.weights.kernel,
                case.sorted_token_ids,
                case.sorted_expert_ids,
                case.num_valid_ids,
                case.partial_out,
                TOPK,
            ),
            stage2_kwargs={
                "w2_scale": session.weights.kernel_scale.view(dtypes.fp8_e8m0),
                "a2_scale": case.a2_scale,
                "block_m": case.block_m,
                "sorted_weights": case.sorted_weights,
            },
            shared_partial=prepared,
            ordinary_stage2=fixture.metadata.stage2,
        )

    flops, nbytes = _stage2_work(fixture.case, session.world)
    result = _record_candidates(
        candidates={"ordinary": run_ordinary, "comm_fused": run_comm_fused},
        mode=mode,
        reference=fixture.reference,
        flops=flops,
        nbytes=nbytes,
        session=session,
    )
    return {
        "ordinary kernel": fixture.ordinary_kernel,
        "comm_fused kernel": config_name(runner.config),
        **result,
    }


def _run_full_runtime_case(
    session: TestSession, tokens: int, route: str, mode: str
) -> dict[str, float | str]:
    if tokens != 3 or route != "skew":
        raise ValueError("the runtime-padding case is fixed to M=3, route=skew")
    fixture = _full_moe_fixture(session)
    case = fixture.case
    runtime = CommFusedMoeRuntime(runners=session.runners)
    if not runtime.supports(tokens):
        raise AssertionError("M=3 must resolve through the production M=4 runner")

    moe_args = {
        "hidden_states": case.hidden_states,
        "w1": case.w1,
        "w2": session.weights.kernel,
        "topk_weight": case.topk_weights,
        "topk_ids": case.topk_ids,
        "activation": ActivationType.Silu,
        "quant_type": QuantType.per_1x32,
        "doweight_stage1": False,
        "w1_scale": case.w1_scale,
        "w2_scale": session.weights.kernel_scale,
        "hidden_pad": 0,
        "intermediate_pad": 0,
        "gate_mode": "interleave",
    }

    def run_ordinary():
        output = fused_moe(**moe_args)
        output.add_(fixture.shared_partial)
        return get_tp_group().all_reduce(output, ca_fp8_quant=False)

    def run_comm_fused(stage2_stream=None):
        return runtime.run(
            shared_partial=None,
            before_stage2=lambda: fixture.shared_partial,
            stage2_stream=stage2_stream,
            **moe_args,
        )

    candidates = {
        "ordinary": run_ordinary,
        "comm_fused": run_comm_fused,
    }
    if mode == "eager":
        # A separate stream is a model-facing execution variant, not a graph
        # candidate; graph capture exercises the default stream choreography.
        candidates["comm_fused_separate_stream"] = lambda: run_comm_fused(
            fixture.stage2_stream
        )

    flops, nbytes = _full_moe_work(case, session.world)
    result = _record_candidates(
        candidates=candidates,
        mode=mode,
        reference=fixture.reference,
        flops=flops,
        nbytes=nbytes,
        session=session,
    )
    return {
        "ordinary kernel": "fused_moe",
        "comm_fused kernel": config_name(session.runners[4].config),
        **result,
    }


@benchmark()
def test_comm_fused_stage2(tokens: int, route: str, mode: str):
    """Benchmark production Stage2 candidates against a torch reference."""

    return _run_stage2_case(_session(), tokens, route, mode)


@benchmark()
def test_comm_fused_runtime(tokens: int, route: str, mode: str):
    """Benchmark the model-facing M=3 -> M=4 runtime-padding path."""

    return _run_full_runtime_case(_session(), tokens, route, mode)


def _parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="TP8 communication-fused MoE production validation",
    )
    parser.add_argument(
        "-m",
        "--tokens",
        type=int,
        nargs="+",
        default=list(PRODUCTION_TOKENS),
        help="Production token buckets to test (default: 1 2 4 8 16).",
    )
    parser.add_argument(
        "-r",
        "--routes",
        choices=ROUTES,
        nargs="+",
        default=list(ROUTES),
        help="Routing distributions to test (default: uniform skew).",
    )
    parser.add_argument(
        "--graph-replays",
        type=int,
        default=3,
        help="Poisoned graph replays checked per candidate (default: 3).",
    )
    args = parser.parse_args()
    if args.graph_replays <= 0:
        parser.error("--graph-replays must be positive")
    return args


def _runtime_skip_reason() -> str | None:
    gfx = get_gfx_runtime()
    if gfx not in SUPPORTED_GFX:
        return f"requires one of {SUPPORTED_GFX}, got {gfx}"
    if not is_flydsl_comm_fused_moe_available():
        return "requires ROCm >= 7.2 and a compatible mori.cco installation"
    return None


def _log_summary(name: str, rows: list[dict]) -> None:
    if rows:
        aiter.logger.info(
            "%s summary (markdown):\n%s",
            name,
            pd.DataFrame(rows).to_markdown(index=False),
        )


def main() -> None:
    global _ACTIVE_SESSION

    args = _parse_args()
    skip_reason = _runtime_skip_reason()
    if skip_reason is not None:
        if int(os.environ.get("LOCAL_RANK", "0")) == 0:
            aiter.logger.warning("comm-fused MoE test skipped: %s", skip_reason)
        return

    requested_tokens = tuple(dict.fromkeys(args.tokens))
    requested_routes = tuple(dict.fromkeys(args.routes))
    invalid_tokens = sorted(set(requested_tokens).difference(PRODUCTION_TOKENS))
    if invalid_tokens:
        raise ValueError(
            f"unsupported production token buckets: {invalid_tokens}; "
            f"expected a subset of {list(PRODUCTION_TOKENS)}"
        )

    rank, world, device, group = _setup_distributed(TP_SIZE)
    if rank != 0:
        aiter.logger.setLevel("WARNING")
    previous_fp8_bound = os.environ.get("AITER_BF16_FP8_MOE_BOUND")
    os.environ["AITER_BF16_FP8_MOE_BOUND"] = "0"
    try:
        shape = ShapeKey(
            get_gfx_runtime(),
            MODEL_DIM,
            INTER_DIM,
            EXPERTS,
            TOPK,
            TP_SIZE,
            get_cu_num(),
        )
        configs = winners_for(shape)
        missing = sorted(set(requested_tokens).difference(configs))
        if missing:
            raise AssertionError(
                f"production comm-fused rows are missing: {missing}; "
                f"available={sorted(configs)}"
            )
        if 32 in configs:
            raise AssertionError("M=32 fallback unexpectedly created a fused runner")

        session = TestSession(
            rank=rank,
            world=world,
            device=device,
            group=group,
            gfx=shape.gfx,
            runners=create_flydsl_comm_fused_runners(
                tp_group=get_tp_group(),
                model_dim=MODEL_DIM,
                inter_dim=INTER_DIM,
                experts=EXPERTS,
                topk=TOPK,
            ),
            weights=_make_stage2_weights(rank, device),
            graph_replays=args.graph_replays,
        )
        _ACTIVE_SESSION = session

        stage2_rows = []
        for tokens, route, mode in itertools.product(
            requested_tokens, requested_routes, MODES
        ):
            row = test_comm_fused_stage2(tokens, route, mode)
            if rank == 0:
                stage2_rows.append(row)

        runtime_rows = []
        for mode in MODES:
            row = test_comm_fused_runtime(3, "skew", mode)
            if rank == 0:
                runtime_rows.append(row)

        if rank == 0:
            _log_summary("comm-fused MoE Stage2", stage2_rows)
            _log_summary("comm-fused MoE runtime padding", runtime_rows)
            print(
                f"COMM_FUSED_UT_OK stage2_cases={len(stage2_rows)} "
                f"runtime_cases={len(runtime_rows)}",
                flush=True,
            )
    finally:
        _ACTIVE_SESSION = None
        if previous_fp8_bound is None:
            os.environ.pop("AITER_BF16_FP8_MOE_BOUND", None)
        else:
            os.environ["AITER_BF16_FP8_MOE_BOUND"] = previous_fp8_bound
        _cleanup_distributed(rank)


if __name__ == "__main__":
    main()
