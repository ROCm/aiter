# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for FlyDSL preshuffle split-K HGEMM kernels."""

from __future__ import annotations

import argparse

import pytest
import torch

from aiter.ops.shuffle import shuffle_weight
from aiter.ops.flydsl.utils import is_flydsl_available
from aiter.test_common import run_perftest

if not torch.cuda.is_available():
    pytest.skip("ROCm not available. Skipping GPU tests.", allow_module_level=True)
if not is_flydsl_available():
    pytest.skip(
        "flydsl is not installed. Skipping FlyDSL HGEMM tests.", allow_module_level=True
    )

try:
    from aiter.ops.flydsl.gemm_kernels import flydsl_hgemm
except ImportError as exc:
    pytest.skip(
        f"Unable to import FlyDSL HGEMM kernels: {exc}", allow_module_level=True
    )

torch.set_default_device("cuda")

DEFAULT_BENCH_ITERS = 20
DEFAULT_BENCH_WARMUP = 3


def run_torch_acc(
    a: torch.Tensor, b: torch.Tensor, dtype=torch.float32
) -> torch.Tensor:
    return torch.mm(a.to(torch.float32), b.to(torch.float32).t()).to(dtype)


def run_torch_bench(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.mm(a, b.t())


def _check_output(
    ref: torch.Tensor,
    out: torch.Tensor,
    *,
    atol: float = 1e-2,
    rtol: float = 1e-2,
    pass_pct: float = 95.0,
) -> bool:
    close_mask = torch.isclose(ref, out, atol=atol, rtol=rtol)
    pct_close = close_mask.float().mean().item() * 100.0
    max_delta = (ref - out).abs().max().item()
    print(
        f"  max_delta={max_delta:.4f}, {pct_close:.1f}% close (atol={atol}, rtol={rtol})"
    )
    return pct_close >= pass_pct


@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize(
    "b_preshuffle",
    [
        pytest.param(False, id="raw-b"),
        pytest.param(True, id="preshuffled-b"),
    ],
)
@pytest.mark.parametrize(
    "m, n, k, tile_k, tile_m, tile_n, pack_n, split_k",
    [
        (24, 384, 7168, 128, 16, 64, 1, 1),
        (32, 7168, 2048, 128, 32, 128, 1, 4),
        (4096, 4096, 4096, 64, 128, 128, 1, 1),
    ],
)
@pytest.mark.parametrize(
    "test_graph",
    [
        pytest.param(False, id="eager"),
        pytest.param(True, id="graph"),
    ],
)
def test_flydsl_splitk_hgemm(
    dtype: str,
    b_preshuffle: bool,
    m: int,
    n: int,
    k: int,
    tile_k: int,
    tile_m: int,
    tile_n: int,
    pack_n: int,
    split_k: int,
    test_graph: bool,
    bench_iters: int = DEFAULT_BENCH_ITERS,
    bench_warmup: int = DEFAULT_BENCH_WARMUP,
):
    print("=" * 80)
    print(
        "[flydsl] preshuffle split-K HGEMM "
        f"dtype={dtype} b_preshuffle={b_preshuffle} shape=({m}, {n}, {k})"
    )
    print("=" * 80)

    torch_dtype = torch.bfloat16 if dtype == "bf16" else torch.float16
    if m < tile_m or m % tile_m != 0:
        bench_warmup = min(bench_warmup, 1)
        bench_iters = min(bench_iters, 3)
    a = torch.rand((m, k), device="cuda", dtype=torch_dtype)
    b = torch.rand((n, k), device="cuda", dtype=torch_dtype)

    _, ref_us = run_perftest(
        run_torch_bench,
        a,
        b,
        num_iters=bench_iters,
        num_warmup=bench_warmup,
        testGraph=test_graph,
    )
    torch.cuda.synchronize()
    ref = run_torch_bench(a, b)

    if b_preshuffle:
        b_shuf = shuffle_weight(b, layout=(16 * pack_n, 16))
    else:
        b_shuf = b
    out = torch.empty((m, n), device="cuda", dtype=torch_dtype)
    _, us = run_perftest(
        flydsl_hgemm,
        a,
        b_shuf,
        out=out,
        num_iters=bench_iters,
        num_warmup=bench_warmup,
        testGraph=test_graph,
        tile_k=tile_k,
        tile_m=tile_m,
        tile_n=tile_n,
        pack_n=pack_n,
        split_k=split_k,
        b_preshuffle=b_preshuffle,
    )
    torch.cuda.synchronize()

    out = flydsl_hgemm(
        a,
        b_shuf,
        out=out,
        tile_k=tile_k,
        tile_m=tile_m,
        tile_n=tile_n,
        pack_n=pack_n,
        split_k=split_k,
        b_preshuffle=b_preshuffle,
    )
    assert _check_output(ref, out)

    bytes_moved = (
        (m * k * a.element_size())
        + (n * k * b.element_size())
        + (m * n * a.element_size())
    )
    flops = 2 * m * n * k
    tflops = flops / (us / 1e6) / 1e12
    tbps = bytes_moved / 1e12 / (us / 1e6)
    speedup = ref_us / us
    print(
        f"[flydsl] Throughput: {us:.1f} us, {tflops:.2f} TFLOPS, "
        f"BW: {tbps:.3f} TB/s, Torch(us): {ref_us:.1f}, Speedup: {speedup:.1f}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run FlyDSL preshuffle split-K HGEMM test"
    )
    parser.add_argument("--dtype", type=str, default="bf16", choices=["fp16", "bf16"])
    parser.add_argument("-m", type=int, default=128)
    parser.add_argument("-n", type=int, default=1024)
    parser.add_argument("-k", type=int, default=8192)
    parser.add_argument("--tile-k", type=int, default=64)
    parser.add_argument("--tile-m", type=int, default=128)
    parser.add_argument("--tile-n", type=int, default=128)
    parser.add_argument("--pack-n", type=int, default=1)
    parser.add_argument("--split-k", type=int, default=1)
    parser.add_argument("--b-preshuffle", action="store_true", default=False)
    parser.add_argument("--num-warmup", type=int, default=DEFAULT_BENCH_WARMUP)
    parser.add_argument("--num-iters", type=int, default=DEFAULT_BENCH_ITERS)
    parser.add_argument("--test-graph", action="store_true", default=False)
    args = parser.parse_args()

    test_flydsl_splitk_hgemm(
        args.dtype,
        m=args.m,
        n=args.n,
        k=args.k,
        tile_k=args.tile_k,
        tile_m=args.tile_m,
        tile_n=args.tile_n,
        pack_n=args.pack_n,
        split_k=args.split_k,
        b_preshuffle=args.b_preshuffle,
        test_graph=args.test_graph,
        bench_iters=args.num_iters,
        bench_warmup=args.num_warmup,
    )
