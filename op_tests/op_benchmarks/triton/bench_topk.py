#!/usr/bin/env python3
from __future__ import annotations
import argparse, itertools, shutil
from pathlib import Path
import pathlib
import torch, triton
from aiter.ops.triton.topk import topk as triton_topk


# ───────────────────────── configuration ────────────────────────────────────
CACHE_DIR   = Path.home() / ".triton" / "cache"
DEVICE      = "cuda"

BATCH_SIZES = [1, 2, 3, 4, 5, 6, 7, 8, 16, 1335]
DIM2S       = (16, 128_256)        # tuple of row lengths (M)
KS          = (2, 8)               # tuple of top-k values


# ─────────────────────────── helpers ────────────────────────────────────────
def purge_cache() -> None:
    """Delete Triton’s on-disk cache so each (M,K) recompiles once."""
    shutil.rmtree(CACHE_DIR, ignore_errors=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


# ───────────────── latency benchmark ────────────────────────────────────────
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["batch"],
        x_vals=BATCH_SIZES,
        x_log=False,
        line_arg="provider",
        line_vals=["triton", "torch"],
        line_names=["Triton", "Torch"],
        styles=[("blue", "-"), ("green", "-")],
        ylabel="Latency (µs)",
        plot_name="topk_latency",
        args={},
    )
)
def benchmark_latency(batch, provider, *, dim2: int, k: int):
    """Return (median, p20, p80) latency in micro-seconds."""
    from triton.testing import runtime
    runtime.driver.active.get_empty_cache_for_benchmark().zero_()

    x = torch.rand(batch, dim2, device=DEVICE, dtype=torch.float32)

    if provider == "torch":
        fn = lambda: x.topk(k, dim=1, largest=True, sorted=True)
    elif provider == "triton":
        fn = lambda: triton_topk(x, k, largest=True, sorted=True)
    else:
        raise ValueError(provider)

    ms, p20, p80 = triton.testing.do_bench(
        fn, warmup=25, rep=100, quantiles=[0.5, 0.2, 0.8]
    )
    return ms * 1000, p20 * 1000, p80 * 1000


# ───────────────── memory benchmark ─────────────────────────────────────────
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
def benchmark_memory(batch, provider, *, dim2: int, k: int):
    """Return (peak, peak, peak) memory usage in MB."""
    x = torch.rand(batch, dim2, device=DEVICE, dtype=torch.float32)

    if provider == "torch":
        fn = lambda: x.topk(k, dim=1, largest=True, sorted=True)
    elif provider == "triton":
        fn = lambda: triton_topk(x, k, largest=True, sorted=True)
    else:
        raise ValueError(provider)

    for _ in range(10):  # warm-up
        fn()

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    fn();  torch.cuda.synchronize()
    peak_mb = torch.cuda.max_memory_allocated() / 1024**2
    return (peak_mb,) * 3


# ─────────────────────────── CLI entry-point ────────────────────────────────
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Top-K Triton benchmark (multiple M,K combos)")
    p.add_argument("--save-dir", type=pathlib.Path, default=pathlib.Path("./figs"),
                   help="Directory where PNG files will be written")
    p.add_argument("--no-show", action="store_true",
                   help="Do not open GUI windows")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    args.save_dir.mkdir(parents=True, exist_ok=True)

    lat_bench = benchmark_latency.benchmarks
    mem_bench = benchmark_memory.benchmarks

    # ─── iterate over every (M, K) pair ─────────────────────────────────────
    for dim2, k in itertools.product(DIM2S, KS):
        purge_cache()

        # dynamically set file titles
        lat_bench.plot_name = f"topk_latency_M={dim2}_K={k}"
        benchmark_latency.run(
            print_data=True,
            show_plots=not args.no_show,
            save_path=str(args.save_dir),
            dim2=dim2, k=k,
        )

        purge_cache()

        mem_bench.plot_name = f"topk_memory_M={dim2}_K={k}"
        benchmark_memory.run(
            print_data=True,
            show_plots=not args.no_show,
            save_path=str(args.save_dir),
            dim2=dim2, k=k,
        )

    # verify figures exist
    for dim2, k in itertools.product(DIM2S, KS):
        for stem in ("latency", "memory"):
            name = f"topk_{stem}_M={dim2}_K={k}.png"
            path = args.save_dir / name
            if path.exists():
                print(f"✅  {name} saved to {path.resolve()}")
            else:
                print(f"⚠️  {name} missing")

if __name__ == "__main__":
    main()
