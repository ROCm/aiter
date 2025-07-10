#!/usr/bin/env python3
from __future__ import annotations
import argparse
import itertools
import math
import shutil
from collections import namedtuple
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import torch
import triton
from triton.testing import runtime
from aiter.ops.triton.topk import topk as triton_topk

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


def _measure_topk(batch: int, M: int, K: int) -> Tuple[float, _FlopMem]:
    x = torch.rand(batch, M, device=DEVICE, dtype=torch.float32)
    tiny_row_thresh = 1024
    if M <= tiny_row_thresh and K <= 8:
        fm = _flopmem_one_stage(batch, M, K, x.element_size())
    else:
        chunk = 256 if M < 1024 else 1024
        if chunk < K:
            chunk = triton.next_power_of_2(K)
        fm = _flopmem_two_stage(batch, M, K, chunk, x.element_size())

    runtime.driver.active.get_empty_cache_for_benchmark().zero_()
    ms = triton.testing.do_bench(
        lambda: triton_topk(x, K, largest=True), warmup=25, rep=100
    )
    return ms / 1e3, fm


def _collect_roof_points(M: int, K: int) -> List[RoofDot]:
    pts: List[RoofDot] = []
    for B in BATCH_SIZES:
        t, fm = _measure_topk(B, M, K)
        pts.append(RoofDot(B, fm.flops / fm.bytes, fm.flops / t))
    return pts


def _plot_roofline(points: List[RoofDot], M: int, K: int, out_dir: Path) -> None:
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
def bench_latency(batch, provider, *, dim2: int, k: int):
    """Return (median, p20, p80) latency in micro-seconds."""
    runtime.driver.active.get_empty_cache_for_benchmark().zero_()

    x = torch.rand(batch, dim2, device=DEVICE, dtype=torch.float32)

    if provider == "torch":

        def fn():
            return x.topk(k, dim=1, largest=True, sorted=True)

    elif provider == "triton":

        def fn():
            return triton_topk(x, k, largest=True, sorted=True)

    else:
        raise ValueError(provider)

    ms, p20, p80 = triton.testing.do_bench(
        fn, warmup=25, rep=100, quantiles=[0.5, 0.2, 0.8]
    )
    return ms * 1000, p20 * 1000, p80 * 1000


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


def parse_args():
    p = argparse.ArgumentParser("Triton Top-K benchmark & Roofline")
    p.add_argument(
        "--save-dir", type=Path, default=Path("./figs"), help="Directory for PNG output"
    )
    p.add_argument(
        "--purge-cache",
        action="store_true",
        help="Clear Triton cache before each sweep",
    )
    p.add_argument(
        "--roofline",
        action="store_true",
        help="Generate Roofline plots instead of latency/memory",
    )
    return p.parse_args()


def run_roofline(args):
    for M, K in itertools.product(DIM2S, KS):
        if args.purge_cache:
            purge_cache()
        print(f"\n=== Collecting roofline (M={M}, K={K}) ===")
        pts = _collect_roof_points(M, K)
        _plot_roofline(pts, M, K, args.save_dir)


def run_latency_memory(args):
    lat_bench = bench_latency.benchmarks
    mem_bench = bench_memory.benchmarks
    for M, K in itertools.product(DIM2S, KS):
        if args.purge_cache:
            purge_cache()
        lat_bench.plot_name = f"topk_latency_M={M}_K={K}"
        bench_latency.run(print_data=True, save_path=str(args.save_dir), dim2=M, k=K)
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


def main():
    args = parse_args()
    args.save_dir.mkdir(parents=True, exist_ok=True)
    if args.roofline:
        run_roofline(args)
    else:
        run_latency_memory(args)


if __name__ == "__main__":
    main()
