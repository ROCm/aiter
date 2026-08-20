# SPDX-License-Identifier: MIT
"""Offline tuner for communication-fused FlyDSL MoE."""

import csv
import statistics
from dataclasses import asdict, dataclass, fields
from itertools import product
from pathlib import Path

import torch
import torch.distributed as dist

from aiter.ops.flydsl.comm_fused_moe_host import (
    PipelineConfig,
    ShapeKey,
    create_runner,
)
from aiter.ops.flydsl.kernels.comm_fused_moe import full_width
from aiter.ops.flydsl.kernels.comm_fused_moe import persistent_window
from aiter.ops.flydsl.kernels.comm_fused_moe import windowed


CSV_FIELDS = tuple(
    "gfx model_dim inter_dim experts topk tp m family tile_m tile_n tile_k "
    "sort_block_m window local_workers reduce_scatter_grid all_gather_grid "
    "service_grid".split()
)
_WINNER_KEY_FIELDS = (
    "gfx",
    "model_dim",
    "inter_dim",
    "experts",
    "topk",
    "tp",
    "m",
)
_CONFIG_FAMILIES = {
    full_width.Config: "full",
    windowed.Config: "window",
    persistent_window.Config: "persistent",
}


@dataclass(frozen=True, slots=True)
class TuningResult:
    config: PipelineConfig
    latency_us: float
    max_abs: float
    rel_l2: float


def _candidates(config_type, m, axes):
    names = tuple(field.name for field in fields(config_type) if field.name != "m")
    for values in product(*(axes[name] for name in names)):
        yield config_type(m, *values)


def full_width_candidates(*, m, **axes):
    return _candidates(full_width.Config, m, axes)


def windowed_candidates(*, m, **axes):
    return _candidates(windowed.Config, m, axes)


def persistent_window_candidates(*, m, **axes):
    return _candidates(persistent_window.Config, m, axes)


def benchmark(
    *,
    tp_group,
    process_group,
    config: PipelineConfig,
    stage2_args: tuple,
    stage2_kwargs: dict,
    shared_partial: torch.Tensor,
    reference: torch.Tensor,
    rounds: int = 3,
    iterations: int = 20,
) -> TuningResult:
    """Measure one complete Stage2 + shared + TP communication candidate."""

    runner = create_runner(tp_group, config)

    def run():
        return runner(
            stage2_args=stage2_args,
            stage2_kwargs=stage2_kwargs,
            shared_partial=shared_partial,
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

    dist.barrier(group=process_group)
    graph = torch.cuda.CUDAGraph()
    with tp_group.graph_capture() as capture:
        with torch.cuda.graph(graph, stream=capture.stream):
            run()
    for _ in range(3):
        graph.replay()
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
            device=shared_partial.device,
        )
        ranks = torch.empty(world, dtype=torch.float64, device=local.device)
        dist.all_gather_into_tensor(ranks, local, group=process_group)
        samples.append(float(ranks.max().item()))

    return TuningResult(
        config,
        statistics.median(samples),
        float(error[0].item()),
        float(error[1].item()),
    )


def select_winner(
    results, *, max_abs: float = 1.0, max_rel_l2: float = 0.05
) -> TuningResult:
    return min(
        (
            result
            for result in results
            if result.max_abs <= max_abs and result.rel_l2 <= max_rel_l2
        ),
        key=lambda result: result.latency_us,
    )


def winner_row(shape: ShapeKey, result: TuningResult) -> dict:
    config = result.config
    row = {
        **asdict(shape),
        **asdict(config),
        "family": _CONFIG_FAMILIES[type(config)],
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
