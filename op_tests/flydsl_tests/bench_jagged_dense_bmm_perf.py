# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

from __future__ import annotations

import argparse
import sys

import torch
import triton

import flydsl.compiler as flyc
from aiter.ops.flydsl.jagged_dense_bmm_dispatch import jagged_dense_bmm_dispatched
from aiter.ops.flydsl.kernels.jagged_dense_bmm_gen import BLOCK_M as _BLOCK_M

try:
    from generative_recommenders.ops.triton.triton_jagged import (
        triton_jagged_dense_bmm_add_fwd,
    )

    _HAS_TRITON = True
except Exception as _exc:  # pragma: no cover - environment dependent
    _HAS_TRITON = False
    _TRITON_ERR = _exc


def default_benchmark_configs():
    return [
        (120, 256, 256),
        (120, 512, 512),
        (1024, 256, 256),
        (1024, 512, 512),
    ]


def _make_seq_offsets(B, Mi, regime, seed, device):
    if regime == "uniform":
        return torch.arange(0, (B + 1) * Mi, Mi, dtype=torch.int32, device=device)
    g = torch.Generator().manual_seed(seed)
    u = torch.rand(B, generator=g)
    t = (Mi * (u**4)).floor().to(torch.int64)
    t[: max(1, B // 5)] = 0  # ~20% empty groups
    t[-1] = Mi  # one full-envelope group
    if B > 1:
        t[-2] = int(0.9 * Mi)  # one near-full group
    so = torch.zeros(B + 1, dtype=torch.int32)
    for i in range(B):
        so[i + 1] = so[i] + int(t[i])
    return so.to(device)


def _make_inputs(B, D, Kout, Mi, regime="uniform", seed=1234, device="cuda"):
    torch.manual_seed(0)
    N, K = Kout, D
    seq_offsets = _make_seq_offsets(B, Mi, regime, seed, device)
    L = int(seq_offsets[-1].item())

    jagged = torch.randn(max(L, 1), K, dtype=torch.bfloat16, device=device)
    dense = torch.randn(B, K, N, dtype=torch.bfloat16, device=device)
    bias = torch.randn(B, N, dtype=torch.bfloat16, device=device)
    return jagged, dense, bias, seq_offsets, L, N, K


def _flydsl_fn(jagged, dense, bias, seq_offsets, B, Mi, N, K, regime="uniform"):
    dense_tall = dense.transpose(1, 2).reshape(B * N, K).contiguous()
    bias_flat = bias.reshape(B * N).contiguous()
    L = jagged.shape[0]
    out = torch.zeros(L + _BLOCK_M, N, dtype=torch.bfloat16, device="cuda")
    tA = flyc.from_dlpack(jagged).mark_layout_dynamic(leading_dim=1, divisibility=8)
    tC = flyc.from_dlpack(out).mark_layout_dynamic(leading_dim=1, divisibility=8)

    uniform = regime == "uniform"

    def fn():
        jagged_dense_bmm_dispatched(
            tC,
            tA,
            dense_tall,
            bias_flat,
            seq_offsets,
            B,
            Mi,
            stream=torch.cuda.current_stream(),
            uniform_seqlen=uniform,
        )

    return fn, out, L


def _triton_fn(jagged, dense, bias, seq_offsets, Mi):
    so64 = seq_offsets.to(torch.int64)

    def fn():
        return triton_jagged_dense_bmm_add_fwd(Mi, so64, jagged, dense, bias)

    return fn


def _torch_reference(jagged, dense, bias, seq_offsets, N):
    L = jagged.shape[0]
    out = torch.zeros((L, N), dtype=torch.bfloat16, device=jagged.device)
    for b in range(dense.shape[0]):
        s = int(seq_offsets[b].item())
        e = int(seq_offsets[b + 1].item())
        if e > s:
            out[s:e] = (
                jagged[s:e].float() @ dense[b].float() + bias[b].float()[None, :]
            ).to(torch.bfloat16)
    return out


def _run_one_regime(custom, args, regime, providers, unit):
    if custom:
        x_vals = [(args.b, args.d, args.kout)]
    else:
        x_vals = default_benchmark_configs()
    x_vals = [(*v, args.mi) for v in x_vals]

    config = triton.testing.Benchmark(
        x_names=["B", "D", "KOUT", "MI"],
        x_vals=x_vals,
        line_arg="provider",
        line_vals=providers,
        line_names=providers,
        styles=[("blue", "-"), ("red", "-")][: len(providers)],
        ylabel=unit,
        plot_name=f"bench_jagged_dense_bmm_{regime}_{args.metric}",
        args={"regime": regime},
    )

    @triton.testing.perf_report([config])
    def bench_fn(B, D, KOUT, MI, provider, regime):
        jagged, dense, bias, seq_offsets, L, N, K = _make_inputs(
            B, D, KOUT, MI, regime=regime, seed=args.seed
        )

        if provider == "flydsl":
            fn, out, _ = _flydsl_fn(
                jagged, dense, bias, seq_offsets, B, MI, N, K, regime
            )
            out_view = out[:L]
        else:
            fn = _triton_fn(jagged, dense, bias, seq_offsets, MI)
            out_view = None

        if args.test:
            ref = _torch_reference(jagged, dense, bias, seq_offsets, N)
            if provider == "flydsl":
                fn()
                torch.cuda.synchronize()
                got = out_view
            else:
                got, _, _, _ = fn()
                torch.cuda.synchronize()
            cos = torch.nn.functional.cosine_similarity(
                ref.float().flatten(), got.float().flatten(), dim=0
            ).item()
            tag = f"({regime:7s} B={B}, D={D}, Kout={KOUT}, Mi={MI})"
            print(
                f"  {'PASS' if cos > 0.999 else 'FAIL'} [{provider:6s}] {tag}  cos={cos:.4f}"
            )

        ms = triton.testing.do_bench(fn, warmup=args.warmup, rep=args.rep)

        flops = 2.0 * L * D * N
        mem = (L * D + B * D * N + B * N + L * N) * 2

        if args.metric == "time":
            return ms
        elif args.metric == "throughput":
            return flops / ms * 1e-9
        elif args.metric == "bandwidth":
            return mem / ms * 1e-6
        raise ValueError(f"Unknown metric: {args.metric}")

    print(f"\n=== regime={regime}  (metric={args.metric} [{unit}]) ===")
    bench_fn.run(save_path="." if args.o else None, print_data=True)


def run_benchmark(custom, args):
    providers = []
    if not args.triton_only:
        providers.append("flydsl")
    if _HAS_TRITON and not args.flydsl_only:
        providers.append("triton")
    if not providers:
        print("No providers selected / available.")
        return

    unit = {"time": "ms", "throughput": "TFLOPS", "bandwidth": "GB/s"}[args.metric]
    regimes = ["uniform", "skew"] if args.regime == "both" else [args.regime]
    for regime in regimes:
        _run_one_regime(custom, args, regime, providers, unit)


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        prog="Benchmark jagged_dense_bmm (jdbba): FlyDSL vs Triton",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--metric", choices=["time", "throughput", "bandwidth"], default="time"
    )
    p.add_argument(
        "--regime",
        choices=["uniform", "skew", "both"],
        default="uniform",
        help="sequence-length distribution (uniform baseline / skewed deployment / both)",
    )
    p.add_argument("--seed", type=int, default=1234, help="skew RNG seed")
    p.add_argument("-b", type=int, default=0, help="B groups (custom shape)")
    p.add_argument("-d", type=int, default=0, help="D = reduction K (custom shape)")
    p.add_argument("-kout", type=int, default=0, help="Kout = output N (custom shape)")
    p.add_argument(
        "-mi",
        type=int,
        default=7680,
        help="max_seq_len (per-group rows in uniform; envelope in skew)",
    )
    p.add_argument("--warmup", type=int, default=25, help="do_bench warmup ms")
    p.add_argument("--rep", type=int, default=100, help="do_bench rep ms")
    p.add_argument(
        "-test", action="store_true", help="correctness check vs torch eager"
    )
    p.add_argument("--flydsl-only", action="store_true")
    p.add_argument("--triton-only", action="store_true")
    p.add_argument("-o", action="store_true", help="save CSV/plot")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if not _HAS_TRITON and not args.flydsl_only:
        print(f"WARNING: upstream Triton kernel unavailable: {_TRITON_ERR}")
    custom = bool(args.b or args.d or args.kout)
    if custom:
        assert args.b and args.d and args.kout, "custom shape needs -b, -d, -kout"
    run_benchmark(custom, args)


if __name__ == "__main__":
    sys.exit(main())
