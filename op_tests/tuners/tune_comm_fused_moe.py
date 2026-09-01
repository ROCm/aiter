# SPDX-License-Identifier: MIT
"""Offline tuner for communication-fused FlyDSL MoE."""

import argparse
import csv
import os
import statistics
from dataclasses import MISSING, dataclass, fields
from itertools import product
from pathlib import Path

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
    get_2stage_cfgs,
    get_padded_M,
    moe_sorting,
    stage2_uses_route_reduce,
)
from aiter.jit.utils.chip_info import get_cu_num, get_gfx_runtime
from aiter.ops.flydsl.comm_fused_moe_host import (
    PipelineConfig,
    ShapeKey,
    config_name,
    create_runner,
)
from aiter.ops.flydsl.kernels.comm_fused_moe.gfx950.a8w4 import (
    gemm2_tp_atomic_pipeline,
    gemm2_tp_megakernel,
    gemm2_tp_window_pipeline,
)
from aiter.ops.flydsl.kernels.comm_fused_moe.gfx950.a8w4.shape import (
    Gemm2TPShape,
)
from aiter.ops.quant import (
    mxfp4_moe_sort_fwd,
    per_1x32_f4_quant,
    per_1x32_f8_scale_f8_quant,
)
from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight_a16w4


@dataclass
class _Stage2Case:
    inter_states: torch.Tensor
    w2: torch.Tensor
    w2_scale: torch.Tensor
    a2_scale: torch.Tensor
    sorted_token_ids: torch.Tensor
    sorted_expert_ids: torch.Tensor
    sorted_weights: torch.Tensor
    num_valid_ids: torch.Tensor
    partial_out: torch.Tensor
    topk: int
    block_m: int


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
    return rank, world, local_rank, device, group


def _cleanup_distributed(rank: int):
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


def _make_stage2_case(args, rank: int, device, *, accumulate: bool):
    token = torch.arange(args.token, dtype=torch.int64, device=device)[:, None]
    slot = torch.arange(args.topk, dtype=torch.int64, device=device)[None, :]
    if args.route == "uniform":
        topk_ids = (token * args.topk + slot) % args.experts
    else:
        topk_ids = 1 + ((token + slot - 1) % (args.experts - 1))
        hot_slot = (token % 8) != 7
        cold_slot = 1 + ((token + args.topk - 1) % (args.experts - 1))
        topk_ids[:, :1] = torch.where(hot_slot, 0, cold_slot)
    weights = (slot + 1).expand(args.token, -1).to(torch.float32)
    topk_weights = (weights / weights.sum(dim=1, keepdim=True)).contiguous()
    (
        sorted_ids,
        sorted_weights,
        sorted_expert_ids,
        num_valid_ids,
        sorting_out,
    ) = moe_sorting(
        topk_ids.to(torch.int32).contiguous(),
        topk_weights,
        args.experts,
        args.model_dim,
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
        sentinel = (args.topk << 24) | args.token
        sorted_ids = torch.nn.functional.pad(
            sorted_ids, (0, row_capacity - sorted_ids.numel()), value=sentinel
        ).contiguous()
        sorted_weights = torch.nn.functional.pad(
            sorted_weights,
            (0, row_capacity - sorted_weights.numel()),
            value=0.0,
        ).contiguous()

    generator = torch.Generator(device=device).manual_seed(20260819 + rank)
    activations = torch.randn(
        (args.token, args.topk, args.inter_dim),
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
    inter_states = inter_states.view(args.token, args.topk, args.inter_dim)
    a2_scale = mxfp4_moe_sort_fwd(
        a2_scale_unsorted.view(args.token * args.topk, -1),
        sorted_ids=sorted_ids,
        num_valid_ids=num_valid_ids,
        token_num=args.token,
        cols=args.inter_dim,
    ).contiguous()

    w2_bf16 = torch.randn(
        (args.experts, args.model_dim, args.inter_dim),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    ).mul_(args.inter_dim**-0.25)
    first_quant, first_scale = per_1x32_f4_quant(w2_bf16[0], quant_dtype=dtypes.fp4x2)
    w2_quant = torch.empty(
        (args.experts, *first_quant.shape), dtype=first_quant.dtype, device=device
    )
    w2_scale = torch.empty(
        (args.experts, *first_scale.shape), dtype=first_scale.dtype, device=device
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
    partial_out = (
        sorting_out
        if sorting_out.numel()
        else torch.empty(
            (args.token, args.model_dim), dtype=torch.bfloat16, device=device
        )
    )
    return _Stage2Case(
        inter_states,
        w2,
        w2_scale,
        a2_scale,
        sorted_ids,
        sorted_expert_ids,
        sorted_weights,
        num_valid_ids,
        partial_out,
        args.topk,
        args.tile_m,
    )


def _resolve_ordinary_stage2(args):
    metadata = get_2stage_cfgs(
        get_padded_M(args.token),
        args.model_dim,
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
    return metadata, not stage2_uses_route_reduce(metadata.stage2)


def _run_ordinary_stage2_allreduce(
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


_WINNER_KEY_FIELDS = (
    "gfx",
    "cu_num",
    "token",
    "model_dim",
    "inter_dim",
    "expert",
    "topk",
    "tp",
    "act_type",
    "dtype",
    "q_dtype_a",
    "q_dtype_w",
    "q_type",
    "use_g1u1",
    "doweight_stage1",
)
CSV_FIELDS = (
    *_WINNER_KEY_FIELDS,
    "block_m",
    "us",
    "kernelName",
    "max_abs",
    "rel_l2",
)


@dataclass(frozen=True, slots=True)
class TuningResult:
    config: PipelineConfig
    latency_us: float
    max_abs: float
    rel_l2: float
    block_m: int | None = None


def _graph_latency_us(
    run,
    *,
    tp_group,
    process_group,
    device,
    warmup_replays,
    rounds,
    iterations,
):
    dist.barrier(group=process_group)
    graph = torch.cuda.CUDAGraph()
    with tp_group.graph_capture() as capture, torch.cuda.graph(
        graph, stream=capture.stream
    ):
        run()
    for _ in range(warmup_replays):
        graph.replay()
    torch.cuda.synchronize(device)
    dist.barrier(group=process_group)

    world = dist.get_world_size(process_group)
    samples = []
    for _ in range(rounds):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            graph.replay()
        end.record()
        end.synchronize()
        local = torch.tensor(
            [start.elapsed_time(end) * 1000.0 / iterations],
            dtype=torch.float64,
            device=device,
        )
        ranks = torch.empty(world, dtype=torch.float64, device=device)
        dist.all_gather_into_tensor(ranks, local, group=process_group)
        samples.append(float(ranks.max().item()))
    return statistics.median(samples)


def _candidates(config_type, shape, m, axes):
    config_fields = tuple(
        field for field in fields(config_type) if field.name not in ("shape", "m")
    )
    value_axes = []
    for field in config_fields:
        if field.name in axes:
            value_axes.append(axes[field.name])
        elif field.default is not MISSING:
            value_axes.append((field.default,))
        elif field.default_factory is not MISSING:
            value_axes.append((field.default_factory(),))
        else:
            raise KeyError(field.name)
    for values in product(*value_axes):
        yield config_type(
            shape=shape,
            m=m,
            **{field.name: value for field, value in zip(config_fields, values)},
        )


def gemm2_tp_atomic_pipeline_candidates(*, shape: Gemm2TPShape, m, **axes):
    return _candidates(
        gemm2_tp_atomic_pipeline.Gemm2TPAtomicPipelineConfig,
        shape,
        m,
        axes,
    )


def gemm2_tp_megakernel_candidates(*, shape: Gemm2TPShape, m, **axes):
    return _candidates(
        gemm2_tp_megakernel.Gemm2TPMegakernelConfig,
        shape,
        m,
        axes,
    )


def gemm2_tp_window_pipeline_candidates(*, shape: Gemm2TPShape, m, **axes):
    return _candidates(
        gemm2_tp_window_pipeline.Gemm2TPWindowPipelineConfig,
        shape,
        m,
        axes,
    )


def benchmark(
    *,
    tp_group,
    process_group,
    config: PipelineConfig,
    stage2_args: tuple,
    stage2_kwargs: dict,
    ordinary_stage2=None,
    shared_partial: torch.Tensor,
    reference: torch.Tensor,
    warmup_replays: int = 100,
    rounds: int = 3,
    iterations: int = 20,
) -> TuningResult:
    """Measure one complete Stage2 + shared + TP communication candidate."""

    runner = create_runner(tp_group, config)
    candidate_shared = shared_partial.clone()
    prepare_shared_partial = getattr(runner, "prepare_shared_partial", None)
    if prepare_shared_partial is not None:
        candidate_shared = prepare_shared_partial(candidate_shared)

    def run():
        return runner(
            stage2_args=stage2_args,
            stage2_kwargs=stage2_kwargs,
            shared_partial=candidate_shared,
            ordinary_stage2=ordinary_stage2,
        )

    reference_f32 = reference.float()
    diff = run().float() - reference_f32
    error = torch.stack(
        (
            diff.abs().max(),
            diff.norm() / reference_f32.norm().clamp_min(1.0e-12),
        )
    )
    dist.all_reduce(error, op=dist.ReduceOp.MAX, group=process_group)

    return TuningResult(
        config,
        _graph_latency_us(
            run,
            tp_group=tp_group,
            process_group=process_group,
            device=shared_partial.device,
            warmup_replays=warmup_replays,
            rounds=rounds,
            iterations=iterations,
        ),
        float(error[0].item()),
        float(error[1].item()),
        int(stage2_kwargs["block_m"]),
    )


_CONFIG_TYPES = {
    "atomic": gemm2_tp_atomic_pipeline.Gemm2TPAtomicPipelineConfig,
    "mega": gemm2_tp_megakernel.Gemm2TPMegakernelConfig,
    "window": gemm2_tp_window_pipeline.Gemm2TPWindowPipelineConfig,
}


def _parse_axis(field, raw_values: str):
    def parse(raw):
        if field.type is bool:
            normalized = raw.strip().lower()
            if normalized in ("1", "true", "yes", "on"):
                return True
            if normalized in ("0", "false", "no", "off"):
                return False
            raise ValueError(f"invalid boolean {raw!r} for {field.name}")
        if field.type is str:
            return raw
        return int(raw)

    values = tuple(parse(raw) for raw in raw_values.split(","))
    if not values:
        raise ValueError(f"empty axis for {field.name}")
    return values


def candidate_configs(current, family: str, axis_specs: tuple[str, ...]):
    config_type = type(current) if family == "current" else _CONFIG_TYPES[family]
    config_fields = {
        field.name: field
        for field in fields(config_type)
        if field.name not in ("shape", "m")
    }
    axes = (
        {name: (getattr(current, name),) for name in config_fields}
        if type(current) is config_type
        else {}
    )
    for spec in axis_specs:
        name, separator, raw_values = spec.partition("=")
        if not separator or name not in config_fields:
            valid = ", ".join(sorted(config_fields))
            raise ValueError(f"invalid --axis {spec!r}; valid fields: {valid}")
        axes[name] = _parse_axis(config_fields[name], raw_values)
    candidates = list(_candidates(config_type, current.shape, current.m, axes))
    return tuple(dict.fromkeys(candidates))


def select_winner(
    results,
    *,
    max_abs: float = 1.0,
    max_rel_l2: float = 0.05,
    ordinary_us: float | None = None,
) -> TuningResult | None:
    valid = tuple(
        result
        for result in results
        if result.max_abs <= max_abs and result.rel_l2 <= max_rel_l2
    )
    if not valid:
        return None
    winner = min(valid, key=lambda result: result.latency_us)
    if ordinary_us is not None and winner.latency_us >= ordinary_us:
        return None
    return winner


def _winner_key(shape: ShapeKey, token: int) -> dict:
    return {
        "gfx": shape.gfx,
        "cu_num": shape.cu_num if shape.cu_num is not None else get_cu_num(),
        "token": token,
        "model_dim": shape.model_dim,
        "inter_dim": shape.inter_dim,
        "expert": shape.experts,
        "topk": shape.topk,
        "act_type": shape.act_type,
        "dtype": shape.dtype,
        "q_dtype_a": shape.q_dtype_a,
        "q_dtype_w": shape.q_dtype_w,
        "q_type": shape.q_type,
        "use_g1u1": shape.use_g1u1,
        "doweight_stage1": shape.doweight_stage1,
        "tp": shape.tp,
    }


def winner_row(shape: ShapeKey, result: TuningResult) -> dict:
    config = result.config
    block_m = getattr(config, "sort_block_m", result.block_m)
    if block_m is None:
        raise ValueError("block_m is required for an atomic pipeline winner")
    if result.block_m is not None and int(result.block_m) != int(block_m):
        raise ValueError(
            f"tuned block_m={result.block_m} does not match "
            f"kernel sort_block_m={block_m}"
        )
    row = {
        **_winner_key(shape, config.m),
        "block_m": block_m,
        "us": result.latency_us,
        "kernelName": config_name(config),
        "max_abs": result.max_abs,
        "rel_l2": result.rel_l2,
    }
    return {field: row.get(field, "") for field in CSV_FIELDS}


def write_winner(path, row: dict) -> None:
    path = Path(path)
    rows = []
    if path.exists():
        with path.open(newline="") as file:
            rows = list(csv.DictReader(file))
    key = tuple(str(row[field]) for field in _WINNER_KEY_FIELDS)
    rows = [
        old
        for old in rows
        if tuple(str(old[field]) for field in _WINNER_KEY_FIELDS) != key
    ]
    rows.append({field: row.get(field, "") for field in CSV_FIELDS})
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def remove_winner(path, key_row: dict) -> None:
    path = Path(path)
    if not path.exists():
        return
    with path.open(newline="") as file:
        rows = list(csv.DictReader(file))
    key = tuple(str(key_row[field]) for field in _WINNER_KEY_FIELDS)
    kept = [
        row
        for row in rows
        if tuple(str(row[field]) for field in _WINNER_KEY_FIELDS) != key
    ]
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(kept)


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Tune one complete FlyDSL GEMM2 + TP communication shape."
    )
    parser.add_argument("--token", type=int, required=True)
    parser.add_argument("--model-dim", type=int, default=7168)
    parser.add_argument("--inter-dim", type=int, default=384)
    parser.add_argument("--experts", type=int, default=384)
    parser.add_argument("--topk", type=int, default=6)
    parser.add_argument("--tp", type=int, default=8)
    parser.add_argument("--route", choices=("uniform", "skew"), default="uniform")
    parser.add_argument(
        "--family",
        choices=("current", *_CONFIG_TYPES),
        default="current",
        help="Candidate family; current starts from the production winner.",
    )
    parser.add_argument(
        "--axis",
        action="append",
        default=[],
        metavar="FIELD=V1,V2",
        help="Override one config field with a comma-separated search axis.",
    )
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument(
        "--warmup",
        type=int,
        default=100,
        help="Graph replays used to warm each candidate before timing.",
    )
    parser.add_argument("--max-abs", type=float, default=1.0)
    parser.add_argument("--max-rel-l2", type=float, default=0.05)
    parser.add_argument(
        "--ordinary-only",
        action="store_true",
        help="Measure the ordinary Stage2 + TP AllReduce baseline and exit.",
    )
    parser.add_argument("--profile-output", type=Path)
    parser.add_argument("--winner-output", type=Path)
    return parser.parse_args()


def main():
    args = _parse_args()
    rank, _world, _local_rank, device, group = _setup_distributed(args.tp)
    try:
        metadata, requires_zero = _resolve_ordinary_stage2(args)
        args.tile_m = int(metadata.block_m)
        case = _make_stage2_case(args, rank, device, accumulate=requires_zero)
        shared = (
            torch.arange(args.token, device=device, dtype=torch.float32)
            .remainder(7)
            .mul_(1.0 / 32.0)
            .view(-1, 1)
            + torch.arange(args.model_dim, device=device, dtype=torch.float32)
            .remainder(17)
            .mul_(1.0 / 128.0)
            .view(1, -1)
            + float(rank + 1) / 16.0
        ).to(torch.bfloat16)
        reference = _run_ordinary_stage2_allreduce(
            case,
            metadata,
            requires_output_zero=requires_zero,
            shared_partial=shared,
        ).clone()

        def run_ordinary():
            return _run_ordinary_stage2_allreduce(
                case,
                metadata,
                requires_output_zero=requires_zero,
                shared_partial=shared,
            )

        ordinary_us = _graph_latency_us(
            run_ordinary,
            tp_group=get_tp_group(),
            process_group=group,
            device=device,
            warmup_replays=args.warmup,
            rounds=args.rounds,
            iterations=args.iterations,
        )
        if rank == 0:
            print(
                f"COMM_FUSED_TUNE_ORDINARY us={ordinary_us:.4f}",
                flush=True,
            )
        if args.ordinary_only:
            return
        shape = ShapeKey(
            get_gfx_runtime(),
            args.model_dim,
            args.inter_dim,
            args.experts,
            args.topk,
            args.tp,
            get_cu_num(),
        )
        from aiter.ops.flydsl.comm_fused_moe_host import winners_for

        current = winners_for(shape).get(args.token)
        if current is None:
            if args.family == "current":
                raise KeyError(
                    f"no production comm_fused config for M={args.token}; "
                    "select an explicit --family"
                )
            current = _CONFIG_TYPES[args.family](
                shape=shape.kernel_shape(),
                m=args.token,
            )
        candidates = candidate_configs(current, args.family, tuple(args.axis))
        if rank == 0:
            print(
                f"COMM_FUSED_TUNE_START M={args.token} route={args.route} "
                f"family={args.family} candidates={len(candidates)}",
                flush=True,
            )
        stage2_args = (
            case.inter_states,
            None,
            case.w2,
            case.sorted_token_ids,
            case.sorted_expert_ids,
            case.num_valid_ids,
            case.partial_out,
            case.topk,
        )
        stage2_kwargs = {
            "w2_scale": case.w2_scale.view(dtypes.fp8_e8m0),
            "a2_scale": case.a2_scale,
            "block_m": case.block_m,
            "sorted_weights": case.sorted_weights,
        }
        results = []
        for index, config in enumerate(candidates, 1):
            result = benchmark(
                tp_group=get_tp_group(),
                process_group=group,
                config=config,
                stage2_args=stage2_args,
                stage2_kwargs=stage2_kwargs,
                ordinary_stage2=metadata.stage2,
                shared_partial=shared,
                reference=reference,
                warmup_replays=args.warmup,
                rounds=args.rounds,
                iterations=args.iterations,
            )
            results.append(result)
            if rank == 0:
                print(
                    f"COMM_FUSED_TUNE_RESULT {index}/{len(candidates)} "
                    f"us={result.latency_us:.4f} "
                    f"max_abs={result.max_abs:.6f} "
                    f"rel_l2={result.rel_l2:.6f} "
                    f"kernel={config_name(config)}",
                    flush=True,
                )
        winner = select_winner(
            results,
            max_abs=args.max_abs,
            max_rel_l2=args.max_rel_l2,
            ordinary_us=ordinary_us,
        )
        if rank == 0:
            if args.profile_output is not None:
                args.profile_output.parent.mkdir(parents=True, exist_ok=True)
                with args.profile_output.open("w", newline="") as file:
                    writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
                    writer.writeheader()
                    writer.writerows(winner_row(shape, result) for result in results)
            if winner is None:
                if args.winner_output is not None:
                    remove_winner(
                        args.winner_output,
                        _winner_key(shape, args.token),
                    )
                print(
                    f"COMM_FUSED_TUNE_WINNER ordinary_us={ordinary_us:.4f} "
                    "kernel=ordinary",
                    flush=True,
                )
            else:
                winner_data = winner_row(shape, winner)
                if args.winner_output is not None:
                    args.winner_output.parent.mkdir(parents=True, exist_ok=True)
                    write_winner(args.winner_output, winner_data)
                print(
                    f"COMM_FUSED_TUNE_WINNER us={winner.latency_us:.4f} "
                    f"ordinary_us={ordinary_us:.4f} "
                    f"speedup={ordinary_us / winner.latency_us:.4f}x "
                    f"max_abs={winner.max_abs:.6f} "
                    f"rel_l2={winner.rel_l2:.6f} "
                    f"kernel={winner_data['kernelName']}",
                    flush=True,
                )
    finally:
        _cleanup_distributed(rank)


if __name__ == "__main__":
    main()
