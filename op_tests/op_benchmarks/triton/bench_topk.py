#!/usr/bin/env python3
from __future__ import annotations

import functools
import itertools
import math
import shutil
from collections import namedtuple
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import triton

from aiter.ops.triton.topk import topk as triton_topk
from op_tests.op_benchmarks.triton.utils.argparse import (
    get_do_bench_graph_kwargs,
    get_parser,
    get_rot_bufs_kwargs,
    override_parser_action,
)
from op_tests.op_benchmarks.triton.utils.benchmark_utils import (
    get_model_configs,
)
from op_tests.op_benchmarks.triton.utils.rotating_buffer_bench import (
    RotatingBufferBench,
)

DEVICE = "cuda"
CACHE_DIR = Path.home() / ".triton" / "cache"
BATCH_SIZES = [1, 2, 3, 4, 5, 6, 7, 8, 16, 1335]
DIM2S = (16, 128, 256, 128256)  # row length M
KS = (2, 8)  # top-k values

# MI300x ceilings (single-precision)
BW_PEAK_BYTES = 5300e9  # 5.3 TB/s
FLOPS_PEAK_FP32 = 163e12  # 163 TFLOP/s

_FlopMem = namedtuple("_FlopMem", "flops bytes")


def purge_cache() -> None:
    """Delete Triton's on-disk cache so each (M,K) recompiles once."""
    shutil.rmtree(CACHE_DIR, ignore_errors=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _generate_topk_pool_inputs(B: int, M: int) -> tuple[torch.Tensor]:
    """One rotating-buffer pool entry: a fresh (B, M) fp32 input. The
    outputs are allocated inside topk itself, so only x rotates."""
    return (torch.rand(B, M, device=DEVICE, dtype=torch.float32),)


_POOL_KEYS = ["x"]


def _bench_rot(bencher: RotatingBufferBench, B: int, M: int, fn) -> float:
    """Build a rotating pool of (B, M) inputs and time ``fn(x)`` over it.
    Returns the mean per-call latency in ms."""
    bencher.build_pool(
        [functools.partial(_generate_topk_pool_inputs, B, M)], [_POOL_KEYS]
    )
    return bencher.bench(fn)


def _triton_topk_fn(k: int):
    def fn(x):
        return triton_topk(x, k, largest=True, sorted=True)

    return fn


def _torch_topk_fn(k: int):
    def fn(x):
        return x.topk(k, dim=1, largest=True, sorted=True)

    return fn


def _flopmem_one_stage(B: int, M: int, K: int, val_b: int, idx_b: int = 8) -> _FlopMem:
    flops = 6 * B * M * K
    mem = B * M * val_b + B * K * (val_b + idx_b)
    return _FlopMem(flops, mem)


def _bitonic_flops(n: int) -> int:
    s = int(math.log2(n))
    return n * s * (s + 1) // 2


def _flopmem_two_stage(B: int, M: int, K: int, chunk: int, elem_size: int) -> _FlopMem:
    cnum = math.ceil(M / chunk)
    fl_s1 = 6 * B * cnum * K * chunk
    bytes_s1 = B * cnum * K * (elem_size + 8)
    n2 = cnum * K
    fl_s2 = B * _bitonic_flops(n2)
    mem = (
        B * M * elem_size  # read whole matrix
        + bytes_s1  # stage-1 write
        + bytes_s1  # stage-2 read
        + B * K * (elem_size + 8)  # final write
    )
    return _FlopMem(fl_s1 + fl_s2, mem)


@dataclass
class RoofDot:
    batch: int
    AI: float  # FLOPs / byte
    P: float  # FLOPs / s


def _measure_topk(
    bencher: RotatingBufferBench, batch: int, M: int, K: int
) -> tuple[float, _FlopMem]:
    elem_size = torch.tensor([], dtype=torch.float32).element_size()
    tiny_row_thresh = 1024
    if M <= tiny_row_thresh and K <= 8:
        fm = _flopmem_one_stage(batch, M, K, elem_size)
    else:
        chunk = 256 if M < 1024 else 1024
        if chunk < K:
            chunk = triton.next_power_of_2(K)
        fm = _flopmem_two_stage(batch, M, K, chunk, elem_size)

    ms = _bench_rot(bencher, batch, M, _triton_topk_fn(K))
    return ms / 1e3, fm


def _collect_roof_points(bencher: RotatingBufferBench, M: int, K: int) -> list[RoofDot]:
    pts: list[RoofDot] = []
    for B in BATCH_SIZES:
        t, fm = _measure_topk(bencher, B, M, K)
        pts.append(RoofDot(B, fm.flops / fm.bytes, fm.flops / t))
    return pts


def _plot_roofline(points: list[RoofDot], M: int, K: int, out_dir: Path) -> None:
    I_vals = [p.AI for p in points]
    P_vals = [p.P for p in points]
    labels = [str(p.batch) for p in points]

    I_ridge = FLOPS_PEAK_FP32 / BW_PEAK_BYTES
    I_min = 0.05
    I_max = max(max(I_vals) * 10, I_ridge * 5)

    I_axis = np.logspace(math.log10(I_min), math.log10(I_max), 512)
    mem_roof = BW_PEAK_BYTES * I_axis
    comp_roof = np.full_like(I_axis, FLOPS_PEAK_FP32)

    plt.figure(figsize=(7, 5))
    plt.loglog(I_axis, mem_roof, label="Memory roof (HBM)", lw=2)
    plt.loglog(I_axis, comp_roof, label="FP32 compute roof", lw=2)
    plt.scatter(I_vals, P_vals, c=range(len(points)), cmap="viridis", zorder=5)
    for x, y, t in zip(I_vals, P_vals, labels):
        plt.text(x * 1.05, y * 1.05, t, fontsize=8)

    plt.axvline(I_ridge, color="grey", ls="--", lw=1)
    plt.text(
        I_ridge * 1.05,
        FLOPS_PEAK_FP32 * 0.6,
        r"$I_{ridge}$",
        rotation=90,
        va="center",
        ha="left",
        fontsize=8,
        color="grey",
    )

    plt.xlabel("Arithmetic intensity  AI  [FLOPs / byte]")
    plt.ylabel("Throughput  P  [FLOPs/s]")
    plt.title(f"Roofline  (M={M}, K={K}) - Triton Top-K")
    plt.grid(True, which="both", ls=":", lw=0.6)
    plt.legend()

    out_dir.mkdir(parents=True, exist_ok=True)
    fname = out_dir / f"topk_roofline_M={M}_K={K}.png"
    plt.tight_layout()
    plt.savefig(fname, dpi=170)
    plt.close()
    print(f"?  Roofline saved to {fname.resolve()}")


# latency & memory benchmarks
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["batch"],
        x_vals=BATCH_SIZES,
        x_log=False,
        line_arg="provider",
        line_vals=["triton", "torch"],
        line_names=["Triton", "Torch"],
        styles=[("blue", "-"), ("green", "-")],
        ylabel="Latency (us)",
        plot_name="topk_latency",
        args={},
    )
)
def bench_latency(batch, provider, *, dim2: int, k: int, bencher: RotatingBufferBench):
    """Return mean per-call latency in micro-seconds."""
    if provider == "torch":
        fn = _torch_topk_fn(k)
    elif provider == "triton":
        fn = _triton_topk_fn(k)
    else:
        raise ValueError(provider)

    return _bench_rot(bencher, batch, dim2, fn) * 1000


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["batch"],
        x_vals=BATCH_SIZES,
        x_log=False,
        line_arg="provider",
        line_vals=["triton", "torch"],
        line_names=["Triton", "Torch"],
        styles=[("blue", "--"), ("green", "--")],
        ylabel="Peak memory (MB)",
        plot_name="topk_memory",
        args={},
    )
)
def bench_memory(batch, provider, *, dim2: int, k: int):
    """Return (peak, peak, peak) memory usage in MB."""
    x = torch.rand(batch, dim2, device=DEVICE, dtype=torch.float32)

    if provider == "torch":

        def fn():
            return x.topk(k, dim=1, largest=True, sorted=True)

    elif provider == "triton":

        def fn():
            return triton_topk(x, k, largest=True, sorted=True)

    else:
        raise ValueError(provider)

    for _ in range(10):  # warm-up
        fn()

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    fn()
    torch.cuda.synchronize()
    peak_mb = torch.cuda.max_memory_allocated() / 1024**2
    return (peak_mb,) * 3


def run_roofline(args, bencher: RotatingBufferBench):
    for M, K in itertools.product(DIM2S, KS):
        if args.purge_cache:
            purge_cache()
        print(f"\n=== Collecting roofline (M={M}, K={K}) ===")
        pts = _collect_roof_points(bencher, M, K)
        _plot_roofline(pts, M, K, args.save_dir)


def run_latency_memory(args, bencher: RotatingBufferBench):
    lat_bench = bench_latency.benchmarks
    mem_bench = bench_memory.benchmarks
    for M, K in itertools.product(DIM2S, KS):
        if args.purge_cache:
            purge_cache()
        lat_bench.plot_name = f"topk_latency_M={M}_K={K}"
        bench_latency.run(
            print_data=True,
            save_path=str(args.save_dir),
            dim2=M,
            k=K,
            bencher=bencher,
        )
        if args.purge_cache:
            purge_cache()
        mem_bench.plot_name = f"topk_memory_M={M}_K={K}"
        bench_memory.run(print_data=True, save_path=str(args.save_dir), dim2=M, k=K)
    # quick sanity for missing figs
    for M, K in itertools.product(DIM2S, KS):
        for stem in ("latency", "memory"):
            path = args.save_dir / f"topk_{stem}_M={M}_K={K}.png"
            if path.exists():
                print(f"?  {path.name} saved")
            else:
                print(f"?  {path.name} missing")


def run_benchmark(args, x_vals_list, bencher: RotatingBufferBench):
    """
    Runs benchmark given the --model argument.
    """
    x_names = ["M", "N", "topk"]

    if args.metric == "time":
        ylabel = "Time (ms)"
    elif args.metric == "memory":
        ylabel = "Memory (MB)"
    else:
        raise NotImplementedError(f"{args.metric} is not supported")
    line_names = ["Triton"]
    line_vals = line_names
    benchmark = triton.testing.Benchmark(
        x_names=x_names,
        x_vals=x_vals_list,
        line_arg="provider",
        line_vals=line_vals,
        line_names=line_names,
        styles=[("green", "-")],  # match line names to colors
        ylabel=ylabel,
        plot_name="Benchmark Routing Layer",
        args={"metric": args.metric},
    )

    @triton.testing.perf_report([benchmark])
    def bench_topk(M, N, topk, **kwargs):
        ms = _bench_rot(bencher, M, N, _triton_topk_fn(topk))
        if args.metric == "time":
            return ms
        else:
            raise ValueError("Unknown metric: " + args.metric)

    bench_topk.run(save_path=".", print_data=True)


def parse_args():
    parser = get_parser(kernel_name="Top-K")

    parser.add_argument(
        "--shape",
        type=int,
        nargs=2,
        metavar=("M", "N"),
        help="user-defined shape to benchmark (M, N), with M being the batch size and N being the vocab size.",
    )
    parser.add_argument(
        "--topk", type=int, default=1, help="Number of elements to select by top-K"
    )
    override_parser_action(parser, action="--metric", choices=["time"], default="time")
    # --model sweeps go up to (16384, 128256) fp32, an ~8 GB input per pool
    # entry; a floor of 4 entries would quadruple that. Inputs that large
    # already exceed the last-level cache on their own, so a floor of 2 is
    # enough to keep consecutive calls from reusing a cache-resident input.
    override_parser_action(parser, action="--rot-bufs-min-pool-len", default=2)
    parser.add_argument(
        "--save-dir", type=Path, default=Path("./figs"), help="Directory for PNG output"
    )
    parser.add_argument(
        "--purge-cache",
        action="store_true",
        help="Clear Triton cache before each sweep",
    )
    parser.add_argument(
        "--roofline",
        action="store_true",
        help="Generate Roofline plots instead of latency/memory",
    )
    parser.add_argument(
        "--run_latency_memory",
        action="store_true",
        help="Run latency and memory benchmarks",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    bencher = RotatingBufferBench.from_dict(
        {**get_rot_bufs_kwargs(args), **get_do_bench_graph_kwargs(args)}
    )
    assert (
        args.model is None or args.shape is None
    ), "Specify one of --model or --shape, not both."
    if args.model is not None:
        config_file = args.model_configs
        configs = get_model_configs(config_path=config_file, models=args.model)
        x_vals = []
        for config in configs.values():
            # layer takes (batch_size, vocab_size) as input and selects (batch_size, topk)
            N = config["vocab_size"]
            for M in [2**i for i in range(15)]:
                x_vals.append((M, N, args.topk))
        run_benchmark(args, x_vals, bencher)
    elif args.shape is not None:
        M, N = args.shape
        run_benchmark(args, [(M, N, args.topk)], bencher)
    else:
        # Default
        M, N, K = 1024, 128256, 16
        run_benchmark(args, [(M, N, K)], bencher)

    args.save_dir.mkdir(parents=True, exist_ok=True)
    if args.roofline:
        run_roofline(args, bencher)
    if args.run_latency_memory:
        run_latency_memory(args, bencher)


if __name__ == "__main__":
    main()
