# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""
Benchmark: fused allreduce+rmsnorm+quant+gemm vs unfused (NCCL) baseline.

Both variants run in the same mode (eager or graph) for fair comparison.

Usage:
    torchrun --nproc_per_node=8 bench_fused_allreduce_add_rms_quant_gemm.py
    torchrun --nproc_per_node=8 bench_fused_allreduce_add_rms_quant_gemm.py \
        --mode graph --num-tokens 4 512 1024
"""

import argparse
import os
from dataclasses import dataclass
from typing import Callable, Literal, Optional

import torch
import torch.distributed as dist

from aiter.dist.parallel_state import (
    graph_capture as aiter_graph_capture,
    init_distributed_environment,
    ensure_model_parallel_initialized,
    set_custom_all_reduce,
)
from aiter.ops.triton.comms.fused_allreduce_add_rms_quant_gemm import (
    fused_allreduce_add_rms_quant_gemm,
)
from aiter.ops.triton.utils.types import get_fp8_e4m3_dtype

FP8_DTYPE = get_fp8_e4m3_dtype()


@dataclass
class Variant:
    name: str
    impl: Literal["iris", "ref"]
    is_baseline: bool = False

    def make_fn(
        self,
        num_tokens: int,
        hidden_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Callable[[], None]:
        input_tensor = torch.randn(num_tokens, hidden_dim, dtype=dtype, device=device)
        residual = torch.randn_like(input_tensor)
        rms_weight = torch.ones(hidden_dim, dtype=dtype, device=device)
        gemm_weight = (
            torch.rand(
                hidden_dim,
                hidden_dim,
                dtype=torch.float32,
                device=device,
            )
            .to(FP8_DTYPE)
            .contiguous()
            .t()
        )
        weight_scale = torch.tensor(
            1.0,
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0)
        impl = self.impl

        def run():
            fused_allreduce_add_rms_quant_gemm(
                input_tensor,
                rms_weight,
                1e-6,
                FP8_DTYPE,
                "",
                gemm_weight,
                weight_scale,
                dtype,
                residual=residual,
                impl=impl,
            )

        return run


VARIANTS: dict[str, Variant] = {
    "unfused": Variant(name="unfused", impl="ref", is_baseline=True),
    "fused": Variant(name="fused", impl="iris"),
}


def warmup(fn: Callable[[], None], iterations: int) -> None:
    for _ in range(iterations):
        fn()
        torch.cuda.synchronize()


def benchmark_eager(fn: Callable[[], None], trials: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times: list[float] = []

    for _ in range(trials):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    times.sort()
    mid = len(times) // 2
    return times[mid] if len(times) % 2 else (times[mid - 1] + times[mid]) / 2


def benchmark_graph(
    graph: torch.cuda.CUDAGraph, trials: int, stream: torch.cuda.Stream
) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times: list[float] = []

    for _ in range(trials):
        with torch.cuda.stream(stream):
            start.record()
            graph.replay()
            end.record()
        stream.synchronize()
        times.append(start.elapsed_time(end))

    times.sort()
    mid = len(times) // 2
    return times[mid] if len(times) % 2 else (times[mid - 1] + times[mid]) / 2


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark fused allreduce+rmsnorm+quant+gemm vs unfused"
    )
    parser.add_argument(
        "--num-tokens",
        type=int,
        nargs="+",
        default=[4, 1024],
    )
    parser.add_argument("--hidden-dim", type=int, default=8192)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument(
        "--mode",
        choices=["eager", "graph"],
        default="graph",
    )
    parser.add_argument(
        "--variant",
        choices=["all", "unfused", "fused"],
        default="all",
    )
    parser.add_argument(
        "--profile",
        type=str,
        default=None,
        metavar="DIR",
        help="Enable torch.profiler and write chrome traces to DIR.",
    )
    args = parser.parse_args()

    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        raise RuntimeError(
            "Must run with torchrun. Example: "
            "torchrun --nproc_per_node=8 bench_fused_allreduce_add_rms_quant_gemm.py"
        )

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    torch.set_default_device(device)

    set_custom_all_reduce(True)
    init_distributed_environment(
        world_size=world_size,
        rank=rank,
        distributed_init_method="env://",
    )
    ensure_model_parallel_initialized(world_size, 1)

    if world_size <= 1:
        raise ValueError(f"World size must be > 1. Got world_size={world_size}.")

    # Select variants
    variants: list[Variant] = (
        list(VARIANTS.values()) if args.variant == "all" else [VARIANTS[args.variant]]
    )

    # Build run functions and warmup
    run_fns: dict[int, dict[str, Callable]] = {}
    for num_tokens in args.num_tokens:
        run_fns[num_tokens] = {}
        for v in variants:
            fn = v.make_fn(num_tokens, args.hidden_dim, torch.bfloat16, device)
            run_fns[num_tokens][v.name] = fn
            warmup(fn, args.warmup)
            if rank == 0:
                print(f"Warmup complete: {v.name}, tokens={num_tokens}")

    # Capture graphs if graph mode
    graphs: dict[int, dict[str, torch.cuda.CUDAGraph]] = {}
    capture_stream: Optional[torch.cuda.Stream] = None
    if args.mode == "graph":
        with aiter_graph_capture() as gc:
            capture_stream = gc.stream
            for num_tokens in args.num_tokens:
                graphs[num_tokens] = {}
                for v in variants:
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=capture_stream):
                        run_fns[num_tokens][v.name]()
                    capture_stream.synchronize()
                    graphs[num_tokens][v.name] = graph
                    if rank == 0:
                        print(f"Captured graph: {v.name}, tokens={num_tokens}")

    # Timed runs
    all_timings: dict[int, dict[str, float]] = {}
    for num_tokens in args.num_tokens:
        timings: dict[str, float] = {}
        for v in variants:
            if args.profile is not None:
                prof = torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ],
                    record_shapes=True,
                    with_stack=True,
                )
                prof.start()

            if args.mode == "graph":
                assert capture_stream is not None
                timings[v.name] = benchmark_graph(
                    graphs[num_tokens][v.name], args.trials, capture_stream
                )
            else:
                timings[v.name] = benchmark_eager(
                    run_fns[num_tokens][v.name], args.trials
                )

            if args.profile is not None:
                prof.stop()
                if rank == 0:
                    print(f"\n--- Profile: {v.name}, tokens={num_tokens} ---")
                    print(
                        prof.key_averages().table(
                            sort_by="self_cuda_time_total",
                            row_limit=30,
                        )
                    )
                trace_dir = os.path.join(args.profile, f"{v.name}_M{num_tokens}")
                os.makedirs(trace_dir, exist_ok=True)
                prof.export_chrome_trace(
                    os.path.join(trace_dir, f"trace_rank{rank}.json")
                )

        all_timings[num_tokens] = timings

    # Print results (rank 0 only)
    if rank == 0:
        baseline_variants = [v for v in variants if v.is_baseline]
        baseline = baseline_variants[0].name if baseline_variants else None

        print(f"\n{'=' * 70}")
        print(f"Benchmark: {' vs '.join(v.name for v in variants)}  mode={args.mode}")
        print(
            f"world_size={world_size}  hidden_dim={args.hidden_dim}  "
            f"dtype=bf16  warmup={args.warmup}  trials={args.trials}"
        )
        print(f"{'=' * 70}")

        col_headers = ["Tokens"]
        for v in variants:
            col_headers.append(f"{v.name} (ms)")
            if baseline and v.name != baseline:
                col_headers.append("Speedup")
        print("  ".join(f"{h:>15s}" for h in col_headers))
        print("-" * (17 * len(col_headers)))

        for num_tokens in args.num_tokens:
            timings = all_timings[num_tokens]
            cols = [f"{num_tokens:>15d}"]
            baseline_ms = timings.get(baseline) if baseline else None
            for v in variants:
                t = timings[v.name]
                cols.append(f"{t:>15.3f}")
                if baseline and v.name != baseline:
                    if baseline_ms and t > 0:
                        cols.append(f"{baseline_ms / t:>14.2f}x")
                    else:
                        cols.append(f"{'N/A':>15s}")
            print("".join(cols))
        print()

    dist.barrier()


if __name__ == "__main__":
    main()
