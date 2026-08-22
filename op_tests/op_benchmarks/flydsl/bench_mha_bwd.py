# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Benchmark the FlyDSL varlen FMHA backward on gfx942.

Covers d_qk=192, d_v=128, causal, bf16, THD.  Times the three ways aiter can
produce this shape's gradients:

    flydsl     the FlyDSL backward, native to the unpadded d_v = 128 shape
    ck         what the router reaches without FlyDSL -- the varlen ASM v3 gate
               requires hdim_q == hdim_v, so d_qk=192 / d_v=128 falls through to
               generic CK-tile
    asm_vpad   ASM v3 with v zero-padded 128 -> 192 to open that gate.  v/out
               come pre-padded from the padded forward such a model would
               already run; padding dout and narrowing dv back to 128 are
               charged to the backward.

`speedup` is median_ms of the baseline over this row's, where the baseline is
`ck` when it is in the run (the incumbent the router reaches today) and otherwise
the first backend timed.

Correctness is NOT checked here -- that is op_tests/test_mha_flydsl_varlen_bwd.py.

Progress goes to stderr and the table to stdout, so aiter's JIT-import chatter
cannot land in the middle of the results.  Redirect stderr to drop it:
`python bench_mha_bwd.py 2>/dev/null`.

Usage:
    # Default single shape
    python bench_mha_bwd.py

    # One explicit ragged batch
    python bench_mha_bwd.py --seqlens 4096 3840 3584 3072 --nheads 8

    # Uniform batch
    python bench_mha_bwd.py --num-seqs 8 --seqlen 4096

    # Sweep, FlyDSL only, to CSV
    python bench_mha_bwd.py --sweep --backend flydsl -o bwd.csv
"""

from __future__ import annotations

import argparse
import itertools
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import triton

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl import is_flydsl_available
from aiter.ops.mha import flash_attn_varlen_func, fmha_v3_varlen_bwd, mha_varlen_bwd

DEVICE = "cuda"
SUPPORTED_GFX = ("gfx942",)

HEAD_DIM_QK = 192
HEAD_DIM_V = 128

BACKENDS = ("flydsl", "ck", "asm_vpad")

# Sweep defaults.  `num_seqs` x `seqlen` spans both of the kernel's dispatch
# regimes: small grids (few short sequences) are co-resident on one round of
# workgroups and take the split-K path, large ones do not.  See `_split()` in
# the kernel's launcher.
SEQLENS = [1024, 2048, 4096]
NUM_SEQS = [1, 4, 16]
NHEADS = [2, 8]


def _make_inputs(seqlens: list[int], nheads: int, dtype: torch.dtype):
    """Build one varlen THD batch plus the forward's `out`/`lse`.

    The real forward is used so the LSE convention under test is aiter's own.
    """
    total = sum(seqlens)
    max_seqlen = max(seqlens)
    scale = 1.0 / math.sqrt(HEAD_DIM_QK)

    torch.manual_seed(0)
    cu_seqlens = torch.tensor(
        [0] + list(itertools.accumulate(seqlens)), dtype=dtypes.i32, device=DEVICE
    )
    q = torch.randn((total, nheads, HEAD_DIM_QK), dtype=dtype, device=DEVICE)
    k = torch.randn((total, nheads, HEAD_DIM_QK), dtype=dtype, device=DEVICE)
    v = torch.randn((total, nheads, HEAD_DIM_V), dtype=dtype, device=DEVICE)

    with torch.no_grad():
        out, lse = flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens,
            cu_seqlens,
            max_seqlen,
            max_seqlen,
            softmax_scale=scale,
            causal=True,
            return_lse=True,
        )
    dout = torch.randn_like(out)

    return {
        "q": q,
        "k": k,
        "v": v,
        "out": out,
        "lse": lse,
        "dout": dout,
        "cu_seqlens": cu_seqlens,
        "max_seqlen": max_seqlen,
        "scale": scale,
        "total": total,
        "nheads": nheads,
        "seqlens": seqlens,
        # Reused across reps: autograd allocates these per backward, so reusing
        # them keeps the measurement on the kernel rather than on the allocator.
        "dq": torch.empty_like(q),
        "dk": torch.empty_like(k),
        "dv": torch.empty_like(v),
    }


def _make_fn(backend: str, x: dict):
    """Return a zero-argument callable running one backward of `backend` over `x`."""
    q, k, v, out, lse, dout = x["q"], x["k"], x["v"], x["out"], x["lse"], x["dout"]
    dq, dk, dv = x["dq"], x["dk"], x["dv"]
    cu, ms_, scale = x["cu_seqlens"], x["max_seqlen"], x["scale"]

    if backend == "flydsl":
        from aiter.ops.flydsl.fmha_kernels import flydsl_flash_attn_varlen_bwd

        def _flydsl():
            flydsl_flash_attn_varlen_bwd(
                dout, q, k, v, out, lse, dq, dk, dv, cu, ms_, ms_, scale
            )

        return _flydsl

    if backend == "ck":

        def _ck():
            mha_varlen_bwd(
                dout,
                q,
                k,
                v,
                out,
                lse,
                cu,
                cu,
                ms_,
                ms_,
                0.0,
                scale,
                False,
                True,
                -1,
                -1,
                False,
                dq,
                dk,
                dv,
            )

        return _ck

    if backend == "asm_vpad":
        pad = HEAD_DIM_QK - HEAD_DIM_V
        v_pad = F.pad(v, (0, pad))
        out_pad = F.pad(out, (0, pad))
        dv_pad = torch.empty_like(v_pad)

        def _asm_vpad():
            dout_pad = F.pad(dout, (0, pad))
            fmha_v3_varlen_bwd(
                dout_pad,
                q,
                k,
                v_pad,
                out_pad,
                lse,
                cu,
                cu,
                ms_,
                ms_,
                0.0,
                scale,
                False,
                True,
                -1,
                -1,
                False,
                True,
                1,
                dq,
                dk,
                dv_pad,
            )
            return dv_pad[..., :HEAD_DIM_V].contiguous()

        return _asm_vpad

    raise ValueError(f"unknown backend: {backend}")


def _flops_bytes(seqlens: list[int], nheads: int, esz: int) -> tuple[int, int]:
    """Causal FLOPs and minimum HBM traffic for one backward.

    Only the j <= i half of each sequence's score matrix is computed.  Five
    GEMMs contract over d: S, dQ and dK over d_qk; dP and dV over d_v.
    """
    total = sum(seqlens)
    pairs = sum(n * (n + 1) // 2 for n in seqlens) * nheads
    flops = 2 * pairs * (3 * HEAD_DIM_QK + 2 * HEAD_DIM_V)
    # q, k, dq, dk at d_qk; v, out, dout, dv at d_v; lse fp32.
    nbytes = (
        total * nheads * (4 * HEAD_DIM_QK + 4 * HEAD_DIM_V) * esz + total * nheads * 4
    )
    return flops, nbytes


def bench_one(
    backend: str,
    seqlens: list[int],
    nheads: int,
    dtype: torch.dtype,
    warmup: int,
    rep: int,
) -> dict:
    x = _make_inputs(seqlens, nheads, dtype)
    fn = _make_fn(backend, x)
    ms, p20, p80 = triton.testing.do_bench(
        fn, warmup=warmup, rep=rep, quantiles=[0.5, 0.2, 0.8]
    )
    flops, nbytes = _flops_bytes(seqlens, nheads, x["q"].element_size())
    return {
        "backend": backend,
        "num_seqs": len(seqlens),
        "max_seqlen": max(seqlens),
        "nheads": nheads,
        "total_tokens": sum(seqlens),
        "median_ms": ms,
        "p20_ms": p20,
        "p80_ms": p80,
        "tflops": flops / ms / 1e9,
        "tbs": nbytes / ms / 1e9,
    }


_COLUMNS = (
    ("backend", 10, ""),
    ("num_seqs", 10, "d"),
    ("max_seqlen", 12, "d"),
    ("nheads", 8, "d"),
    ("total_tokens", 14, "d"),
    ("median_ms", 12, ".4f"),
    ("p20_ms", 10, ".4f"),
    ("p80_ms", 10, ".4f"),
    ("tflops", 10, ".1f"),
    ("tbs", 9, ".2f"),
    ("speedup", 9, ".2f"),
)


def _shape_key(r: dict) -> tuple:
    return (r["num_seqs"], r["max_seqlen"], r["nheads"])


def _add_speedups(results: list[dict], baseline: str):
    """Fill each row's `speedup` relative to `baseline` at the same shape."""
    base_ms: dict[tuple, float] = {
        _shape_key(r): r["median_ms"] for r in results if r["backend"] == baseline
    }
    for r in results:
        ref = base_ms.get(_shape_key(r))
        r["speedup"] = ref / r["median_ms"] if ref else float("nan")


def _print_header():
    print("".join(f"{name:>{w}}" for name, w, _ in _COLUMNS))
    print("-" * sum(w for _, w, _ in _COLUMNS))


def _print_row(r: dict):
    print("".join(f"{r[name]:>{w}{fmt}}" for name, w, fmt in _COLUMNS))


def _save_results_csv(filepath: str, results: list[dict]):
    path = Path(filepath)
    keys = [name for name, _, _ in _COLUMNS]
    with open(path, "w") as f:
        f.write(",".join(keys) + "\n")
        f.writelines(",".join(str(r[key]) for key in keys) + "\n" for r in results)
    print(f"\nResults saved to {path.resolve()}")


def parse_args():
    parser = argparse.ArgumentParser(
        prog="bench_mha_bwd",
        description="Benchmark the FlyDSL varlen FMHA backward "
        "(d_qk=192, d_v=128, causal).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--backend",
        nargs="+",
        choices=[*BACKENDS, "all"],
        default=["all"],
        help="Backend(s) to time.",
    )
    parser.add_argument(
        "--seqlens",
        type=int,
        nargs="+",
        help="Explicit ragged sequence lengths for a single run "
        "(overrides --num-seqs/--seqlen).",
    )
    parser.add_argument(
        "--num-seqs",
        type=int,
        dest="num_seqs",
        default=8,
        help="Number of sequences for a single run.",
    )
    parser.add_argument(
        "--seqlen",
        type=int,
        default=4096,
        help="Per-sequence length for a single run.",
    )
    parser.add_argument(
        "--nheads",
        type=int,
        default=2,
        help="Number of attention heads for a single run.",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Sweep --seqlen-values x --num-seqs-values x --nheads-values "
        "instead of a single run.",
    )
    parser.add_argument(
        "--seqlen-values",
        type=int,
        nargs="+",
        dest="seqlen_values",
        default=SEQLENS,
        help="Per-sequence lengths for the sweep.",
    )
    parser.add_argument(
        "--num-seqs-values",
        type=int,
        nargs="+",
        dest="num_seqs_values",
        default=NUM_SEQS,
        help="Sequence counts for the sweep.",
    )
    parser.add_argument(
        "--nheads-values",
        type=int,
        nargs="+",
        dest="nheads_values",
        default=NHEADS,
        help="Head counts for the sweep.",
    )
    parser.add_argument(
        "-o",
        type=str,
        metavar="FILE",
        help="Output CSV file path for results.",
    )
    parser.add_argument("--warmup", type=int, default=25, help="Warmup iterations.")
    parser.add_argument("--rep", type=int, default=100, help="Timed repetitions.")
    return parser.parse_args()


def main():
    # Parsed before the arch gate so `--help` works everywhere.
    args = parse_args()

    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning(
            "flydsl varlen fmha backward unsupported on %s; skipping", get_gfx()
        )
        return
    if not is_flydsl_available():
        aiter.logger.warning("flydsl is not installed; skipping")
        return

    backends = list(BACKENDS) if "all" in args.backend else args.backend
    dtype = dtypes.bf16

    if args.sweep:
        shapes = [
            ([seqlen] * num_seqs, nheads)
            for num_seqs, seqlen, nheads in itertools.product(
                args.num_seqs_values, args.seqlen_values, args.nheads_values
            )
        ]
    elif args.seqlens:
        shapes = [(args.seqlens, args.nheads)]
    else:
        shapes = [([args.seqlen] * args.num_seqs, args.nheads)]

    baseline = "ck" if "ck" in backends else backends[0]
    total = len(shapes) * len(backends)

    # Progress on stderr, results on stdout: aiter logs a module import and a
    # full type-hint override the first time each backend is touched, which
    # would otherwise be interleaved between the header and the rows.
    print(
        f"timing {total} runs ({len(shapes)} shapes x {len(backends)} backends)",
        file=sys.stderr,
    )

    results = []
    for seqlens, nheads in shapes:
        for backend in backends:
            print(
                f"  [{len(results) + 1}/{total}] {backend} "
                f"num_seqs={len(seqlens)} max_seqlen={max(seqlens)} nheads={nheads}",
                file=sys.stderr,
            )
            results.append(
                bench_one(backend, seqlens, nheads, dtype, args.warmup, args.rep)
            )

    _add_speedups(results, baseline)

    print(
        f"\nFlyDSL varlen FMHA backward -- d_qk={HEAD_DIM_QK}, d_v={HEAD_DIM_V}, "
        f"causal, bf16, {get_gfx()}"
    )
    print(
        f"backends: {', '.join(backends)}   shapes: {len(shapes)}   "
        f"speedup baseline: {baseline}\n"
    )
    _print_header()
    for row in results:
        _print_row(row)

    if args.o:
        _save_results_csv(args.o, results)


if __name__ == "__main__":
    main()
