# SPDX-License-Identifier: MIT
"""Standalone 1x32 MXFP4 quantization benchmark for MI355.

The primary comparison deliberately calls the two public fast paths:

* ``aiter.ops.quant.per_1x32_f4_quant_hip``;
* ``mega_moe.quant.per_1x32_mx_quant(..., quant_mode="fp4")``.

The eager Torch implementation is optional and is reported as a numerical
reference, never as a fast-path baseline.  Multi-rank runs use Gloo only for
round alignment and post-timing rank-max reduction; no GPU collective is
inserted into the measured interval.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class QuantPath:
    name: str
    call: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]
    kernel_launches_per_call: int | str
    measurement_scope: str


def _dist_enabled() -> bool:
    return dist.is_available() and dist.is_initialized()


def _barrier() -> None:
    if _dist_enabled():
        dist.barrier()


def _sample_stats(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("cannot summarize an empty sample list")
    ordered = sorted(values)
    p95_index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return {
        "mean": float(sum(values) / len(values)),
        "min": float(ordered[0]),
        "p50": float(statistics.median(values)),
        "p95": float(ordered[p95_index]),
        "max": float(ordered[-1]),
    }


def _tensor_bytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel() * tensor.element_size())


def _byte_view(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.contiguous().view(torch.uint8).reshape(-1)


def _output_contract(
    q: torch.Tensor, scale: torch.Tensor
) -> dict[str, object]:
    return {
        "q_shape": list(q.shape),
        "q_dtype": str(q.dtype),
        "q_bytes": _tensor_bytes(q),
        "scale_shape": list(scale.shape),
        "scale_dtype": str(scale.dtype),
        "scale_bytes": _tensor_bytes(scale),
        "total_output_bytes": _tensor_bytes(q) + _tensor_bytes(scale),
    }


def _rank_max_samples(
    gpu_us: list[float], host_us: list[float], submit_us: list[float]
) -> tuple[list[float], list[float], list[float]]:
    samples = torch.tensor(
        list(zip(gpu_us, host_us, submit_us)), dtype=torch.float64
    )
    if _dist_enabled():
        # Timing has ended before this CPU/Gloo collective is submitted.
        dist.all_reduce(samples, op=dist.ReduceOp.MAX)
    return (
        samples[:, 0].tolist(),
        samples[:, 1].tolist(),
        samples[:, 2].tolist(),
    )


def _cross_rank_diagnostics(
    gpu_us: list[float],
    host_us: list[float],
    submit_us: list[float],
    tail_iterations: int,
) -> tuple[list[dict[str, object]] | None, list[dict[str, float]] | None]:
    rank = dist.get_rank() if _dist_enabled() else 0
    world = dist.get_world_size() if _dist_enabled() else 1
    payload = {
        "rank": rank,
        "gpu_us": gpu_us,
        "host_us": host_us,
        "submit_us": submit_us,
    }
    gathered = [None] * world if rank == 0 else None
    if _dist_enabled():
        dist.gather_object(payload, gathered, dst=0)
    else:
        gathered = [payload]
    if rank != 0:
        return None, None

    assert gathered is not None
    ordered = sorted(gathered, key=lambda item: int(item["rank"]))
    local_by_rank = [
        {
            "rank": int(item["rank"]),
            "all_gpu_us": _sample_stats(item["gpu_us"]),
            "tail_gpu_us": _sample_stats(item["gpu_us"][-tail_iterations:]),
            "all_host_us": _sample_stats(item["host_us"]),
            "tail_host_us": _sample_stats(item["host_us"][-tail_iterations:]),
            "all_submit_us": _sample_stats(item["submit_us"]),
            "tail_submit_us": _sample_stats(
                item["submit_us"][-tail_iterations:]
            ),
            "gpu_samples_us": item["gpu_us"],
            "host_samples_us": item["host_us"],
            "submit_samples_us": item["submit_us"],
        }
        for item in ordered
    ]
    per_iteration = []
    for iteration in range(len(gpu_us)):
        gpu = [float(item["gpu_us"][iteration]) for item in ordered]
        host = [float(item["host_us"][iteration]) for item in ordered]
        submit = [float(item["submit_us"][iteration]) for item in ordered]
        gpu_min, gpu_max = min(gpu), max(gpu)
        host_min, host_max = min(host), max(host)
        per_iteration.append(
            {
                "iteration": iteration,
                "gpu_min_us": gpu_min,
                "gpu_max_us": gpu_max,
                "gpu_skew_us": gpu_max - gpu_min,
                "gpu_max_over_min": (
                    gpu_max / gpu_min if gpu_min > 0.0 else float("inf")
                ),
                "host_min_us": host_min,
                "host_max_us": host_max,
                "host_skew_us": host_max - host_min,
                "host_max_over_min": (
                    host_max / host_min if host_min > 0.0 else float("inf")
                ),
                "submit_min_us": min(submit),
                "submit_max_us": max(submit),
                "submit_skew_us": max(submit) - min(submit),
            }
        )
    return local_by_rank, per_iteration


def _run_path(
    path: QuantPath,
    x: torch.Tensor,
    device: torch.device,
    *,
    warmup: int,
    iterations: int,
    tail_iterations: int,
) -> tuple[dict[str, object], tuple[torch.Tensor, torch.Tensor]]:
    # First call materializes HIP/FlyDSL modules and allocator state.  It is
    # intentionally separate from both warmup and measured iterations.
    output = path.call(x)
    torch.cuda.synchronize(device)
    _barrier()

    for _ in range(warmup):
        torch.cuda.synchronize(device)
        _barrier()
        output = path.call(x)
        torch.cuda.synchronize(device)

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    local_gpu_us: list[float] = []
    local_host_us: list[float] = []
    local_submit_us: list[float] = []

    for iteration in range(iterations):
        torch.cuda.synchronize(device)
        _barrier()
        host_start = time.perf_counter()
        starts[iteration].record()
        submit_start = time.perf_counter()
        output = path.call(x)
        local_submit_us.append((time.perf_counter() - submit_start) * 1.0e6)
        ends[iteration].record()
        ends[iteration].synchronize()
        local_host_us.append((time.perf_counter() - host_start) * 1.0e6)
        local_gpu_us.append(starts[iteration].elapsed_time(ends[iteration]) * 1.0e3)

    rank_max_gpu_us, rank_max_host_us, rank_max_submit_us = _rank_max_samples(
        local_gpu_us, local_host_us, local_submit_us
    )
    local_by_rank, per_iteration = _cross_rank_diagnostics(
        local_gpu_us, local_host_us, local_submit_us, tail_iterations
    )
    tail_gpu = rank_max_gpu_us[-tail_iterations:]
    tail_host = rank_max_host_us[-tail_iterations:]
    result = {
        "path": path.name,
        "kernel_launches_per_call": path.kernel_launches_per_call,
        "measurement_scope": path.measurement_scope,
        "output": _output_contract(*output),
        "warmup": warmup,
        "iterations": iterations,
        "tail_iterations": tail_iterations,
        "tail_rank_max_gpu_us": _sample_stats(tail_gpu),
        "tail_rank_max_host_us": _sample_stats(tail_host),
        "tail_rank_max_submit_us": _sample_stats(
            rank_max_submit_us[-tail_iterations:]
        ),
        "rank_max_gpu_us": rank_max_gpu_us,
        "rank_max_host_us": rank_max_host_us,
        "rank_max_submit_us": rank_max_submit_us,
        "local_by_rank": local_by_rank,
        "per_iteration_rank_spread": per_iteration,
    }
    return result, output


def _correctness(
    outputs: dict[str, tuple[torch.Tensor, torch.Tensor]],
) -> dict[str, dict[str, int]]:
    names = list(outputs)
    if len(names) < 2:
        return {}
    reference_name = names[0]
    reference_q, reference_scale = outputs[reference_name]
    reference_q_bytes = _byte_view(reference_q)
    reference_scale_bytes = _byte_view(reference_scale)
    comparisons: dict[str, dict[str, int]] = {}
    for name in names[1:]:
        q, scale = outputs[name]
        q_bytes = _byte_view(q)
        scale_bytes = _byte_view(scale)
        if q_bytes.numel() != reference_q_bytes.numel():
            raise AssertionError(
                f"{name} q bytes={q_bytes.numel()}, "
                f"{reference_name} q bytes={reference_q_bytes.numel()}"
            )
        if scale_bytes.numel() != reference_scale_bytes.numel():
            raise AssertionError(
                f"{name} scale bytes={scale_bytes.numel()}, "
                f"{reference_name} scale bytes={reference_scale_bytes.numel()}"
            )
        comparisons[f"{reference_name}_vs_{name}"] = {
            "q_byte_mismatch": int(
                torch.count_nonzero(q_bytes != reference_q_bytes).item()
            ),
            "scale_byte_mismatch": int(
                torch.count_nonzero(
                    scale_bytes != reference_scale_bytes
                ).item()
            ),
        }
    return comparisons


def _build_paths(
    x: torch.Tensor, include_reference: bool
) -> list[QuantPath]:
    import flydsl.expr as fx

    from aiter.ops.flydsl.kernels.megamoe_tile.kernels.quant_core import (
        _get_launcher as get_quant_core_launcher,
    )
    from aiter.ops.flydsl.kernels.megamoe_tile.kernels.quant_core import megamoe_tile_quant_core
    from aiter.ops.flydsl.kernels.mega_moe.quant import (
        _get_launcher as get_flydsl_quant_launcher,
    )
    from aiter.ops.flydsl.kernels.mega_moe.quant import per_1x32_mx_quant
    from aiter.ops.quant import (
        dynamic_per_group_scaled_quant,
        per_1x32_f4_quant_hip,
    )
    from aiter.utility import dtypes

    m, n = x.shape
    scale_n = n // 32
    grid_blocks = (m * scale_n + 63) // 64
    fx_stream = fx.Stream(torch.cuda.current_stream().cuda_stream)

    hip_q = torch.empty((m, n // 2), dtype=dtypes.fp4x2, device=x.device)
    hip_scale = torch.empty(
        (m, scale_n), dtype=torch.uint8, device=x.device
    ).view(dtypes.fp8_e8m0)

    def hip_preallocated(inp):
        dynamic_per_group_scaled_quant(
            hip_q,
            inp,
            hip_scale,
            32,
            shuffle_scale=False,
        )
        return hip_q, hip_scale

    flydsl_q_bytes = torch.empty(
        (m, n // 2), dtype=torch.uint8, device=x.device
    )
    flydsl_q = flydsl_q_bytes.view(torch.float4_e2m1fn_x2)
    flydsl_scale = torch.empty(
        (m, scale_n), dtype=torch.uint8, device=x.device
    )
    flydsl_launcher = get_flydsl_quant_launcher(n, "fp4")

    def flydsl_preallocated(inp):
        flydsl_launcher(
            inp,
            flydsl_q_bytes,
            flydsl_scale,
            int(m),
            int(grid_blocks),
            stream=fx_stream,
        )
        return flydsl_q, flydsl_scale

    core_q_bytes = torch.empty(
        (m, n // 2), dtype=torch.uint8, device=x.device
    )
    core_q = core_q_bytes.view(torch.float4_e2m1fn_x2)
    core_scale = torch.empty(
        (m, scale_n), dtype=torch.uint8, device=x.device
    )
    core_launcher = get_quant_core_launcher(n)

    def core_preallocated(inp):
        core_launcher(
            inp,
            core_q_bytes,
            core_scale,
            int(m),
            stream=fx_stream,
        )
        return core_q, core_scale

    paths = [
        QuantPath(
            "aiter_hip_per_1x32_f4_quant",
            lambda x: per_1x32_f4_quant_hip(x, shuffle=False),
            1,
            "public_wrapper_including_output_allocation_and_enqueue_gap",
        ),
        QuantPath(
            "aiter_hip_quant_kernel_preallocated",
            hip_preallocated,
            1,
            "preallocated_direct_binding_drained_stream_interval",
        ),
        QuantPath(
            "flydsl_per_1x32_mx_quant_fp4",
            lambda x: per_1x32_mx_quant(x, quant_mode="fp4"),
            1,
            "public_wrapper_including_output_allocation_and_enqueue_gap",
        ),
        QuantPath(
            "flydsl_quant_kernel_preallocated",
            flydsl_preallocated,
            1,
            "preallocated_cached_launcher_drained_stream_interval",
        ),
        QuantPath(
            "megamoe_tile_blockidx_quant_core_fp4",
            megamoe_tile_quant_core,
            1,
            "public_wrapper_including_output_allocation_and_enqueue_gap",
        ),
        QuantPath(
            "megamoe_tile_blockidx_quant_core_preallocated",
            core_preallocated,
            1,
            "preallocated_cached_launcher_drained_stream_interval",
        ),
    ]
    if include_reference:
        from aiter.ops.quant import per_1x32_f4_quant

        paths.append(
            QuantPath(
                "torch_reference_per_1x32_f4_quant",
                lambda x: per_1x32_f4_quant(x, shuffle=False),
                "multiple_runtime_dependent",
                "torch_reference_public_wrapper",
            )
        )
    return paths


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=7168)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--tail-iters", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--include-reference", action="store_true")
    parser.add_argument(
        "--path-set", choices=("all", "public", "kernel"), default="all"
    )
    args = parser.parse_args()
    if args.tokens <= 0 or args.hidden <= 0 or args.hidden % 32:
        raise ValueError("tokens/hidden must be positive and hidden divisible by 32")
    if args.warmup < 0 or args.iters < 1:
        raise ValueError("warmup must be non-negative and iters >= 1")
    if not 1 <= args.tail_iters <= args.iters:
        raise ValueError("tail-iters must be in [1, iters]")

    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world > 1:
        dist.init_process_group("gloo", rank=rank, world_size=world)
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    try:
        generator = torch.Generator(device=device).manual_seed(args.seed + rank)
        x = torch.randn(
            (args.tokens, args.hidden),
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        )
        paths = _build_paths(x, args.include_reference)
        if args.path_set == "public":
            paths = [
                path
                for path in paths
                if path.measurement_scope.startswith("public_wrapper")
                or path.measurement_scope.startswith("torch_reference")
            ]
        elif args.path_set == "kernel":
            paths = [
                path
                for path in paths
                if path.measurement_scope.startswith("preallocated_")
            ]
        results: list[dict[str, object]] = []
        outputs: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        for path in paths:
            _barrier()
            result, output = _run_path(
                path,
                x,
                device,
                warmup=args.warmup,
                iterations=args.iters,
                tail_iterations=args.tail_iters,
            )
            results.append(result)
            outputs[path.name] = output
            # Keep other ranks silent at a barrier while rank 0 emits a JSON
            # line larger than PIPE_BUF; otherwise torchrun can interleave the
            # next rank's stdout into the record.
            _barrier()
            if rank == 0:
                print(
                    "MEGAMOE_QUANT_PATH_RESULT "
                    + json.dumps(result, sort_keys=True),
                    flush=True,
                )
            _barrier()

        correctness = _correctness(outputs)
        torch.cuda.synchronize(device)
        local_mismatch = sum(
            value["q_byte_mismatch"] + value["scale_byte_mismatch"]
            for value in correctness.values()
        )
        mismatch = torch.tensor([local_mismatch], dtype=torch.int64)
        if _dist_enabled():
            dist.all_reduce(mismatch, op=dist.ReduceOp.SUM)
        _barrier()
        if rank == 0:
            print(
                "MEGAMOE_QUANT_1X32_FP4_BENCH "
                + json.dumps(
                    {
                        "shape_per_rank": [args.tokens, args.hidden],
                        "input_dtype": str(x.dtype),
                        "world_size": world,
                        "coordination_backend": (
                            "gloo" if world > 1 else "single_process"
                        ),
                        "warmup": args.warmup,
                        "iterations": args.iters,
                        "tail_iterations": args.tail_iters,
                        "path_set": args.path_set,
                        "paths": results,
                        "rank0_correctness": correctness,
                        "global_byte_mismatch_sum": int(mismatch.item()),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        _barrier()
    finally:
        if _dist_enabled():
            dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
