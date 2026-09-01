# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""TP8 production correctness smoke test for communication-fused MoE.

Run with:
    torchrun --standalone --nproc_per_node=8 \
        op_tests/multigpu_tests/test_comm_fused_moe.py

The test resolves production runners from CSV and covers both the direct
Stage2 path and the complete AITer MoE runtime.  The latter intentionally
uses M=3, so it also validates the production M=3 -> M=4 padding path.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from types import SimpleNamespace

import torch
import torch.distributed as dist

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
)
from aiter.jit.utils.chip_info import get_cu_num, get_gfx_runtime
from aiter.ops.comm_fused_moe_runtime import CommFusedMoeRuntime
from aiter.ops.flydsl.comm_fused_moe_host import (
    ShapeKey,
    create_flydsl_comm_fused_runners,
    winners_for,
)
from aiter.ops.flydsl.kernels.comm_fused_moe.gfx950.a8w4 import (
    gemm2_tp_megakernel,
)
from aiter.ops.quant import (
    mxfp4_moe_sort_fwd,
    per_1x32_f4_quant,
    per_1x32_f8_scale_f8_quant,
)
from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight_a16w4


@dataclass
class Stage2Case:
    tokens: int
    hidden: int
    inter_dim: int
    experts: int
    topk: int
    block_m: int
    inter_states: torch.Tensor
    w2: torch.Tensor
    w2_scale: torch.Tensor
    a2_scale: torch.Tensor
    sorted_token_ids: torch.Tensor
    sorted_expert_ids: torch.Tensor
    sorted_weights: torch.Tensor
    num_valid_ids: torch.Tensor
    partial_out: torch.Tensor


@dataclass
class FullMoeCase:
    hidden_states: torch.Tensor
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor
    w1: torch.Tensor
    w1_scale: torch.Tensor
    w2: torch.Tensor
    w2_scale: torch.Tensor


def setup_distributed(expected_world: int):
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
    return rank, world, local_rank, device, group


def barrier(group):
    torch.cuda.synchronize()
    dist.barrier(group=group)


def cleanup_distributed(rank: int):
    if rank == 0:
        print("[STAGE] release distributed communicators", flush=True)
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
    if rank == 0:
        print("[STAGE] distributed teardown complete", flush=True)


def make_routes(tokens: int, experts: int, topk: int, device, route: str):
    token = torch.arange(tokens, dtype=torch.int64, device=device)[:, None]
    slot = torch.arange(topk, dtype=torch.int64, device=device)[None, :]
    if route == "uniform":
        topk_ids = (token * topk + slot) % experts
    elif route == "skew":
        topk_ids = 1 + ((token + slot - 1) % (experts - 1))
        hot_slot = (token % 8) != 7
        cold_slot = 1 + ((token + topk - 1) % (experts - 1))
        topk_ids[:, :1] = torch.where(hot_slot, 0, cold_slot)
    else:
        raise ValueError(f"unsupported route pattern: {route}")
    weights = (slot + 1).expand(tokens, -1).to(torch.float32)
    return (
        topk_ids.to(torch.int32).contiguous(),
        (weights / weights.sum(dim=1, keepdim=True)).contiguous(),
    )


def make_stage2_case(
    args,
    rank: int,
    device,
    *,
    shared_w2: torch.Tensor | None,
    shared_w2_scale: torch.Tensor | None,
    seed_offset: int,
    accumulate: bool,
) -> Stage2Case:
    topk_ids, topk_weights = make_routes(
        args.tokens, args.experts, args.topk, device, args.route
    )
    (
        sorted_ids,
        sorted_weights,
        sorted_expert_ids,
        num_valid_ids,
        sorting_out,
    ) = moe_sorting(
        topk_ids,
        topk_weights,
        args.experts,
        args.hidden,
        torch.bfloat16,
        args.tile_m,
        accumulate=accumulate,
    )
    sorted_ids = sorted_ids.to(torch.int32).contiguous()
    sorted_weights = sorted_weights.to(torch.float32).contiguous()
    sorted_expert_ids = sorted_expert_ids.to(torch.int32).contiguous()
    num_valid_ids = num_valid_ids.to(torch.int32).contiguous()
    row_capacity = int(sorted_expert_ids.numel()) * args.tile_m
    if sorted_ids.numel() < row_capacity:
        sentinel = (args.topk << 24) | args.tokens
        sorted_ids = torch.nn.functional.pad(
            sorted_ids, (0, row_capacity - sorted_ids.numel()), value=sentinel
        ).contiguous()
        sorted_weights = torch.nn.functional.pad(
            sorted_weights,
            (0, row_capacity - sorted_weights.numel()),
            value=0.0,
        ).contiguous()

    generator = torch.Generator(device=device).manual_seed(
        args.seed + rank + seed_offset
    )
    activations = torch.randn(
        (args.tokens, args.topk, args.inter_dim),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    ).mul_(args.inter_dim**-0.25)
    inter_states, a2_scale_unsorted = per_1x32_f8_scale_f8_quant(
        activations,
        quant_dtype=dtypes.fp8,
        scale_type=dtypes.fp8_e8m0,
    )
    del activations
    inter_states = inter_states.view(args.tokens, args.topk, args.inter_dim)
    a2_scale = mxfp4_moe_sort_fwd(
        a2_scale_unsorted.view(args.tokens * args.topk, -1),
        sorted_ids=sorted_ids,
        num_valid_ids=num_valid_ids,
        token_num=args.tokens,
        cols=args.inter_dim,
    ).contiguous()

    if (shared_w2 is None) != (shared_w2_scale is None):
        raise ValueError("shared W2 and scale must be provided together")
    if shared_w2 is None:
        w2_bf16 = torch.randn(
            (args.experts, args.hidden, args.inter_dim),
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        ).mul_(args.inter_dim**-0.25)
        first_quant, first_scale = per_1x32_f4_quant(
            w2_bf16[0], quant_dtype=dtypes.fp4x2
        )
        w2_quant = torch.empty(
            (args.experts, *first_quant.shape),
            dtype=first_quant.dtype,
            device=device,
        )
        w2_scale = torch.empty(
            (args.experts, *first_scale.shape),
            dtype=first_scale.dtype,
            device=device,
        )
        w2_quant[0].copy_(first_quant)
        w2_scale[0].copy_(first_scale)
        for expert in range(1, args.experts):
            quant, scale = per_1x32_f4_quant(w2_bf16[expert], quant_dtype=dtypes.fp4x2)
            w2_quant[expert].copy_(quant)
            w2_scale[expert].copy_(scale)
        del w2_bf16
        w2 = shuffle_weight_a16w4(w2_quant, 16, False).contiguous()
        w2_scale = shuffle_scale_a16w4(
            w2_scale.view(-1, w2_scale.shape[-1]), args.experts, False
        ).contiguous()
        del w2_quant
    else:
        w2, w2_scale = shared_w2, shared_w2_scale

    partial_out = (
        sorting_out
        if sorting_out.numel()
        else torch.empty(
            (args.tokens, args.hidden), dtype=torch.bfloat16, device=device
        )
    )
    return Stage2Case(
        args.tokens,
        args.hidden,
        args.inter_dim,
        args.experts,
        args.topk,
        args.tile_m,
        inter_states,
        w2,
        w2_scale,
        a2_scale,
        sorted_ids,
        sorted_expert_ids,
        sorted_weights,
        num_valid_ids,
        partial_out,
    )


def make_full_moe_case(
    *, rank: int, device, w2: torch.Tensor, w2_scale: torch.Tensor
) -> FullMoeCase:
    """Create one complete DSV4 A8W4 case without a full-BF16 W1 allocation."""

    tokens, hidden, inter_dim, experts, topk = 3, 7168, 384, 384, 6
    generator = torch.Generator(device=device).manual_seed(20260903 + rank)
    hidden_states = torch.randn(
        (tokens, hidden),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    ).mul_(hidden**-0.25)
    topk_ids, topk_weights = make_routes(tokens, experts, topk, device, "skew")

    # Initialize only routed W1 rows to avoid a transient 4+ GiB allocation.
    first_weight = torch.randn(
        (inter_dim * 2, hidden),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    ).mul_(hidden**-0.25)
    first_quant, first_scale = per_1x32_f4_quant(first_weight, quant_dtype=dtypes.fp4x2)
    # FP4 tensors cannot fill_ or zero_; initialize only the routed experts.
    w1_quant = torch.empty(
        (experts, *first_quant.shape), dtype=first_quant.dtype, device=device
    )
    w1_scale = torch.empty(
        (experts, *first_scale.shape), dtype=first_scale.dtype, device=device
    )
    routed_experts = {int(expert) for expert in topk_ids.unique().tolist()}
    if 0 not in routed_experts:
        raise AssertionError("skew route must include the hot expert")
    w1_quant[0].copy_(first_quant)
    w1_scale[0].copy_(first_scale)
    for expert in sorted(routed_experts - {0}):
        weight = torch.randn(
            (inter_dim * 2, hidden),
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        ).mul_(hidden**-0.25)
        quant, scale = per_1x32_f4_quant(weight, quant_dtype=dtypes.fp4x2)
        w1_quant[expert].copy_(quant)
        w1_scale[expert].copy_(scale)
    w1 = shuffle_weight_a16w4(w1_quant, 16, True).contiguous()
    w1_scale = shuffle_scale_a16w4(
        w1_scale.view(-1, w1_scale.shape[-1]), experts, True
    ).contiguous()
    del w1_quant

    return FullMoeCase(
        hidden_states=hidden_states,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        w1=w1,
        w1_scale=w1_scale,
        w2=w2,
        w2_scale=w2_scale,
    )


def resolve_ordinary_stage2(args):
    metadata = get_2stage_cfgs(
        get_padded_M(args.target_tokens),
        args.hidden,
        args.inter_dim,
        args.experts,
        args.topk,
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


def run_ordinary_stage2_allreduce(
    case, metadata, *, requires_output_zero, shared_partial
):
    if requires_output_zero:
        case.partial_out.zero_()
    metadata.stage2(
        case.inter_states,
        None,
        case.w2,
        case.sorted_token_ids,
        case.sorted_expert_ids,
        case.num_valid_ids,
        case.partial_out,
        case.topk,
        w2_scale=case.w2_scale.view(dtypes.fp8_e8m0),
        a2_scale=case.a2_scale,
        block_m=int(metadata.block_m),
        sorted_weights=case.sorted_weights,
    )
    case.partial_out.add_(shared_partial)
    return get_tp_group().all_reduce(case.partial_out, ca_fp8_quant=False)


def error_stats(actual: torch.Tensor, expected: torch.Tensor, group):
    diff = actual.float() - expected.float()
    stats = torch.stack(
        (
            diff.abs().max(),
            diff.norm() / expected.float().norm().clamp_min(1.0e-12),
        )
    )
    dist.all_reduce(stats, op=dist.ReduceOp.MAX, group=group)
    return float(stats[0].item()), float(stats[1].item())


_PRODUCTION_TOKENS = (1, 2, 4, 8, 16)
_DEFAULT_CASES = (
    (1, "uniform"),
    (2, "skew"),
    (4, "uniform"),
    (8, "skew"),
    (16, "uniform"),
)
_MAX_ABS = 1.0
_MAX_REL_L2 = 0.05


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, nargs="+")
    parser.add_argument(
        "--routes",
        choices=("uniform", "skew"),
        nargs="+",
        default=("uniform", "skew"),
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run every production fused bucket with every selected route.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--eager-only", action="store_true")
    mode.add_argument("--graph-only", action="store_true")
    parser.add_argument("--graph-replays", type=int, default=3)
    args = parser.parse_args()
    if args.graph_replays <= 0:
        parser.error("--graph-replays must be positive")
    return args


def _shape_key() -> ShapeKey:
    return ShapeKey(
        get_gfx_runtime(),
        7168,
        384,
        384,
        6,
        8,
        get_cu_num(),
    )


def _case_specs(args, configs):
    if args.tokens is not None:
        requested = tuple(args.tokens)
        missing = sorted(set(requested).difference(configs))
        if missing:
            raise AssertionError(
                "requested tokens do not have a production comm-fused runner: "
                f"{missing}; available={sorted(configs)}"
            )
        return tuple((token, route) for token in requested for route in args.routes)
    if args.full:
        return tuple(
            (token, route) for token in sorted(configs) for route in args.routes
        )
    missing = sorted(set(_PRODUCTION_TOKENS).difference(configs))
    if missing:
        raise AssertionError(f"production comm-fused rows are missing: {missing}")
    return _DEFAULT_CASES


def _shared_partial(tokens: int, hidden: int, rank: int, device):
    token_term = (
        torch.arange(tokens, device=device, dtype=torch.float32)
        .remainder(7)
        .mul_(1.0 / 32.0)
        .view(-1, 1)
    )
    column_term = (
        torch.arange(hidden, device=device, dtype=torch.float32)
        .remainder(17)
        .mul_(1.0 / 128.0)
        .view(1, -1)
    )
    return (token_term + column_term + float(rank + 1) / 16.0).to(torch.bfloat16)


def _assert_output(name, actual, expected, *, group, rank):
    if actual.shape != expected.shape or actual.dtype != torch.bfloat16:
        raise AssertionError(
            f"{name}: expected BF16 shape {tuple(expected.shape)}, got "
            f"dtype={actual.dtype} shape={tuple(actual.shape)}"
        )
    finite = torch.tensor(
        int(torch.isfinite(actual).all() and torch.isfinite(expected).all()),
        device=actual.device,
        dtype=torch.int32,
    )
    dist.all_reduce(finite, op=dist.ReduceOp.MIN, group=group)
    max_abs, rel_l2 = error_stats(actual, expected, group)
    if rank == 0:
        print(
            f"COMM_FUSED_UT {name} max_abs={max_abs:.6f} rel_l2={rel_l2:.6f}",
            flush=True,
        )
    if not int(finite.item()) or max_abs > _MAX_ABS or rel_l2 > _MAX_REL_L2:
        raise AssertionError(
            f"{name}: finite={bool(finite.item())} max_abs={max_abs} "
            f"rel_l2={rel_l2}"
        )


def _capture_graph(run, group):
    barrier(group)
    run()
    barrier(group)
    graph = torch.cuda.CUDAGraph()
    with get_tp_group().graph_capture() as capture, torch.cuda.graph(
        graph, stream=capture.stream
    ):
        run()
    barrier(group)
    return graph


def _run_full_runtime_case(
    *,
    rank,
    device,
    group,
    runners,
    w2,
    w2_scale,
    run_eager,
    run_graph,
    graph_replays,
):
    """Exercise Stage1 + padding + fused Stage2 through the public runtime."""

    case = make_full_moe_case(
        rank=rank,
        device=device,
        w2=w2,
        w2_scale=w2_scale,
    )
    runtime = CommFusedMoeRuntime(runners=runners)
    if not runtime.supports(3):
        raise AssertionError("M=3 must resolve through the production M=4 runner")
    shared = _shared_partial(3, 7168, rank, device)
    moe_args = {
        "hidden_states": case.hidden_states,
        "w1": case.w1,
        "w2": case.w2,
        "topk_weight": case.topk_weights,
        "topk_ids": case.topk_ids,
        "activation": ActivationType.Silu,
        "quant_type": QuantType.per_1x32,
        "doweight_stage1": False,
        "w1_scale": case.w1_scale,
        "w2_scale": case.w2_scale,
        "hidden_pad": 0,
        "intermediate_pad": 0,
        "gate_mode": "interleave",
    }

    def run_ordinary():
        partial = fused_moe(**moe_args)
        partial.add_(shared)
        return get_tp_group().all_reduce(partial, ca_fp8_quant=False)

    # Match ATOM's before_stage2 call path.
    def run_fused():
        return runtime.run(
            shared_partial=None,
            before_stage2=lambda: shared,
            **moe_args,
        )

    ordinary = run_ordinary().clone()
    label = "runtime_e2e M=3->4 route=skew"
    if run_eager:
        fused = run_fused()
        barrier(group)
        _assert_output(f"{label} eager", fused, ordinary, group=group, rank=rank)
    if run_graph:
        graph = _capture_graph(run_fused, group)
        for replay in range(graph_replays):
            graph.replay()
            barrier(group)
            _assert_output(
                f"{label} graph_replay={replay + 1}",
                runners[4].output[:3],
                ordinary,
                group=group,
                rank=rank,
            )
    barrier(group)


def _run_case(
    *,
    token,
    route,
    rank,
    world,
    device,
    group,
    runners,
    shared_w2,
    shared_w2_scale,
    run_eager,
    run_graph,
    graph_replays,
    seed_offset,
):
    case_args = SimpleNamespace(
        tokens=token,
        target_tokens=token,
        logical_tokens=token,
        tp_size=world,
        hidden=7168,
        inter_dim=384,
        experts=384,
        topk=6,
        tile_m=64,
        tile_n=256,
        tile_k=128,
        route=route,
        seed=20260902,
        low_memory_w2_quant=True,
    )
    metadata, ordinary_kernel, requires_zero = resolve_ordinary_stage2(case_args)
    case_args.tile_m = int(metadata.block_m)
    case = make_stage2_case(
        case_args,
        rank,
        device,
        shared_w2=shared_w2,
        shared_w2_scale=shared_w2_scale,
        seed_offset=seed_offset,
        accumulate=requires_zero,
    )
    shared = _shared_partial(token, case_args.hidden, rank, device)
    ordinary = run_ordinary_stage2_allreduce(
        case,
        metadata,
        requires_output_zero=requires_zero,
        shared_partial=shared,
    ).clone()

    runner = runners[token]
    config = runner.config
    if not isinstance(config, gemm2_tp_megakernel.Gemm2TPMegakernelConfig):
        raise TypeError(f"M={token} resolved unexpected runner {type(config)!r}")
    if config.collective != "direct":
        raise AssertionError(
            f"M={token} expected direct collective, got {config.collective}"
        )

    w2_scale = (
        case.w2_scale.view(dtypes.fp8_e8m0)
        if case.w2_scale.element_size() == 1
        else case.w2_scale
    )

    def run_fused():
        prepared = runner.prepare_shared_partial(shared)
        return runner(
            stage2_args=(
                case.inter_states,
                None,
                case.w2,
                case.sorted_token_ids,
                case.sorted_expert_ids,
                case.num_valid_ids,
                case.partial_out,
                case.topk,
            ),
            stage2_kwargs={
                "w2_scale": w2_scale,
                "a2_scale": case.a2_scale,
                "block_m": case.block_m,
                "sorted_weights": case.sorted_weights,
            },
            shared_partial=prepared,
            ordinary_stage2=metadata.stage2,
        )

    label = f"M={token} route={route} ordinary={ordinary_kernel}"
    if rank == 0:
        print(
            f"COMM_FUSED_UT_DISPATCH {label} config={type(config).__name__} "
            f"kernel={config}",
            flush=True,
        )
    if run_eager:
        fused = run_fused()
        barrier(group)
        _assert_output(f"{label} eager", fused, ordinary, group=group, rank=rank)
    if run_graph:
        fused_graph = _capture_graph(run_fused, group)
        for replay in range(graph_replays):
            fused_graph.replay()
            barrier(group)
            _assert_output(
                f"{label} graph_replay={replay + 1}",
                runner.output,
                ordinary,
                group=group,
                rank=rank,
            )
    barrier(group)
    return case.w2, case.w2_scale


def main() -> None:
    args = _parse_args()
    rank, world, _local_rank, device, group = setup_distributed(8)
    previous_fp8_bound = os.environ.get("AITER_BF16_FP8_MOE_BOUND")
    # Force MXFP8 so small-M does not select the unrelated FP4 fallback.
    os.environ["AITER_BF16_FP8_MOE_BOUND"] = "0"
    try:
        shape = _shape_key()
        configs = winners_for(shape)
        if 32 in configs:
            raise AssertionError(
                "M=32 fallback row unexpectedly created a fused runner"
            )
        specs = _case_specs(args, configs)
        runners = create_flydsl_comm_fused_runners(
            tp_group=get_tp_group(),
            model_dim=shape.model_dim,
            inter_dim=shape.inter_dim,
            experts=shape.experts,
            topk=shape.topk,
        )
        shared_w2 = None
        shared_w2_scale = None
        for index, (token, route) in enumerate(specs):
            shared_w2, shared_w2_scale = _run_case(
                token=token,
                route=route,
                rank=rank,
                world=world,
                device=device,
                group=group,
                runners=runners,
                shared_w2=shared_w2,
                shared_w2_scale=shared_w2_scale,
                run_eager=not args.graph_only,
                run_graph=not args.eager_only,
                graph_replays=args.graph_replays,
                seed_offset=index,
            )
        if shared_w2 is None or shared_w2_scale is None:
            raise AssertionError("Stage2 cases did not initialize shared W2")
        _run_full_runtime_case(
            rank=rank,
            device=device,
            group=group,
            runners=runners,
            w2=shared_w2,
            w2_scale=shared_w2_scale,
            run_eager=not args.graph_only,
            run_graph=not args.eager_only,
            graph_replays=args.graph_replays,
        )
        if rank == 0:
            print(
                f"COMM_FUSED_UT_OK stage2_cases={len(specs)} runtime_cases=1",
                flush=True,
            )
    finally:
        if previous_fp8_bound is None:
            os.environ.pop("AITER_BF16_FP8_MOE_BOUND", None)
        else:
            os.environ["AITER_BF16_FP8_MOE_BOUND"] = previous_fp8_bound
        cleanup_distributed(rank)


if __name__ == "__main__":
    main()
