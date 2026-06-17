# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""A/B performance harness: FlyDSL vs Triton FP8 MQA logits (gfx942).

Times both kernels on identical inputs and prints a side-by-side table
(time, TFLOP/s, and FlyDSL/Triton speedup). 

Examples:
    # default DeepSeek-ish shape
    python op_tests/op_benchmarks/triton/bench_flydsl_vs_triton_fp8_mqa_logits.py

    # custom shape + parity check against the torch reference
    python .../bench_flydsl_vs_triton_fp8_mqa_logits.py \
        --seq_q_l 1024 --seq_kv_l 4096 --num_heads_q 64 --head_dim 128 --verify
"""
import argparse
import os
import sys

# Make the script runnable directly (python path/to/bench.py) by putting the
# repo root on sys.path, so `import aiter` / `import op_tests` resolve without
# requiring PYTHONPATH or an installed aiter package.
_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch
import triton

from aiter.ops.triton.attention.fp8_mqa_logits import fp8_mqa_logits as triton_logits
from aiter.ops.triton.utils.types import e4m3_dtype
from aiter.ops.flydsl import is_flydsl_available
from op_tests.triton_tests.attention.test_fp8_mqa_logits import (
    per_custom_dims_cast_to_fp8,
)


def calculate_tflops(start_inds, end_inds, num_heads_q, head_dim, time_ms):
    time_s = time_ms * 1e-3
    start_inds = start_inds.to("cpu").numpy()
    end_inds = end_inds.to("cpu").numpy()
    total_flops = 0.0
    for i in range(len(start_inds)):
        total_flops += 2.0 * num_heads_q * head_dim * (end_inds[i] - start_inds[i])
    return total_flops / (time_s * 1e12)


def _make_inputs(batch_size, seq_q_l, seq_kv_l, num_heads_q, head_dim):
    s_q = batch_size * seq_q_l
    s_k = batch_size * seq_kv_l

    q = torch.randn(s_q, num_heads_q, head_dim, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(s_k, head_dim, device="cuda", dtype=torch.bfloat16)
    # Round-trip kv through fp8 so the bf16 `kv` used by the torch reference
    # matches what the kernels actually consume (mirrors the unit test).
    kv_fp8, scales = per_custom_dims_cast_to_fp8(kv, (0,), False)
    kv = (kv_fp8.to(torch.float32) * scales.reshape(-1, 1)).to(torch.bfloat16)
    weights = torch.randn(s_q, num_heads_q, device="cuda", dtype=torch.float32)

    ks = torch.zeros(s_q, dtype=torch.int, device="cuda")
    ke = torch.zeros(s_q, dtype=torch.int, device="cuda")
    arange_q = torch.arange(seq_q_l, dtype=torch.int, device="cuda")
    for b in range(batch_size):
        qs = b * seq_q_l
        kvs = b * seq_kv_l
        ks[qs : qs + seq_q_l] = kvs
        ke[qs : qs + seq_q_l] = kvs + (seq_kv_l - seq_q_l) + arange_q + 1

    q_fp8 = q.to(e4m3_dtype)
    kv_fp8, scales = per_custom_dims_cast_to_fp8(kv, (0,), False)
    # `q`/`kv` (bf16) are returned for the torch reference; the rest feed the
    # actual kernels.
    return q, kv, q_fp8, kv_fp8, scales, weights, ks, ke


def _time_ms(func, warmup=25, rep=100):
    """Return median time in ms, or None if the implementation isn't ready."""
    try:
        func()  # one eager call to surface NotImplementedError / compile errors
    except NotImplementedError:
        return None
    torch.cuda.synchronize()
    return triton.testing.do_bench(func, warmup=warmup, rep=rep)


# Perf-oriented preset sweep (DeepSeek-style square + rectangular shapes).
# Each entry: (batch_size, seq_q_l, seq_kv_l, num_heads_q, head_dim).
# Covers both head counts the ticket calls out (H in {64, 128}) at D=128,
# square + rectangular windows, plus a non-aligned s_kv "tail" shape.
PRESET_SHAPES = [
    # H = 64
    (1, 1024, 1024, 64, 128),
    (1, 2048, 2048, 64, 128),
    (1, 4096, 4096, 64, 128),
    (1, 1024, 4096, 64, 128),
    (1, 4096, 16384, 64, 128),
    (1, 4096, 16000, 64, 128),  # non-aligned s_kv (tail)
    # H = 128
    (1, 1024, 1024, 128, 128),
    (1, 2048, 2048, 128, 128),
    (1, 4096, 4096, 128, 128),
    (1, 1024, 4096, 128, 128),
    (1, 4096, 16384, 128, 128),
    (1, 4096, 16000, 128, 128),  # non-aligned s_kv (tail)
]


def _select_impls(which):
    """Return {name: fn} for the requested impl selection ('all'/'triton'/'flydsl')."""
    available = {"triton": triton_logits}
    if is_flydsl_available():
        from aiter.ops.flydsl import flydsl_fp8_mqa_logits

        available["flydsl"] = flydsl_fp8_mqa_logits

    if which == "all":
        if "flydsl" not in available:
            print("[warn] flydsl unavailable -- benchmarking Triton only.")
        return available
    if which == "flydsl" and "flydsl" not in available:
        raise SystemExit("[error] --impl flydsl requested but flydsl is unavailable.")
    return {which: available[which]}


def run(args):
    impls = _select_impls(args.impl)
    shapes = _resolve_shapes(args)

    rows = []
    total = len(shapes)
    for n, (idx, (batch_size, seq_q_l, seq_kv_l, num_heads_q, head_dim)) in enumerate(
        shapes, start=1
    ):
        shape = argparse.Namespace(
            batch_size=batch_size,
            seq_q_l=seq_q_l,
            seq_kv_l=seq_kv_l,
            num_heads_q=num_heads_q,
            head_dim=head_dim,
            clean_logits=args.clean_logits,
        )
        # Progress to stderr so it doesn't interleave with the final table.
        print(f"[{n}/{total}] {_fmt_shape(shape)}", file=sys.stderr, flush=True)
        rows.append(_run_one(idx, impls, shape, args))

    _print_table(rows, impls, args)


def _run_one(idx, impls, shape, args):
    """Time + (optionally) verify one case; return a row dict for the table."""
    q, kv, q_fp8, kv_fp8, scales, weights, ks, ke = _make_inputs(
        shape.batch_size, shape.seq_q_l, shape.seq_kv_l, shape.num_heads_q,
        shape.head_dim,
    )

    def make_func(fn):
        return lambda: fn(q_fp8, kv_fp8, scales, weights, ks, ke, shape.clean_logits)

    times, tflops = {}, {}
    for name, fn in impls.items():
        t = _time_ms(make_func(fn), args.warmup, args.rep)
        times[name] = t
        tflops[name] = (
            calculate_tflops(ks, ke, shape.num_heads_q, shape.head_dim, t)
            if t is not None else None
        )

    verify = (
        _verify(impls, q, kv, q_fp8, kv_fp8, scales, weights, ks, ke, shape)
        if args.verify else {name: "N/A" for name in impls}
    )

    base = times.get("triton")
    fly = times.get("flydsl")
    speedup = f"{base / fly:.2f}x" if base and fly else "-"

    return {
        "idx": idx,
        "shape": shape,
        "times": times,
        "tflops": tflops,
        "verify": verify,
        "speedup": speedup,
    }


def _verify(impls, q, kv, q_fp8, kv_fp8, scales, weights, ks, ke, shape):
    """Grade every implementation against the pure-PyTorch reference.

    ``ref_fp8_mqa_logits`` (the DeepGEMM-derived torch implementation, also used
    by the unit test) is the ground truth so Triton and FlyDSL are each
    validated independently. Returns {impl_name: "PASS"|"FAIL"|"SKIP"}.
    """
    from op_tests.triton_tests.attention.test_fp8_mqa_logits import (
        calc_diff,
        ref_fp8_mqa_logits,
    )

    ref, _ = ref_fp8_mqa_logits(
        q=q, kv=kv, weights=weights, cu_seqlen_ks=ks, cu_seqlen_ke=ke
    )
    status = {}
    for name, fn in impls.items():
        try:
            out = fn(q_fp8, kv_fp8, scales, weights, ks, ke, shape.clean_logits)
        except NotImplementedError:
            status[name] = "SKIP"
            continue
        m = (ref == float("-inf")) | (out == float("-inf"))
        diff = calc_diff(out.masked_fill(m, 0), ref.masked_fill(m, 0))
        status[name] = "PASS" if diff < 1e-3 else "FAIL"
    return status


def _fmt_shape(shape):
    return (
        f"bs{shape.batch_size} {shape.seq_q_l}x{shape.seq_kv_l} "
        f"H{shape.num_heads_q} D{shape.head_dim}"
    )


def _print_table(rows, impls, args):
    """Print one row per case across all shapes in a single table."""
    has_tri = "triton" in impls
    has_fly = "flydsl" in impls

    cols = [("idx", 5), ("shape", 26)]
    if has_tri:
        cols += [("tri_ms", 12), ("tri_TFLOP/s", 13), ("tri_vrf", 9)]
    if has_fly:
        cols += [("fly_ms", 12), ("fly_TFLOP/s", 13), ("fly_vrf", 9)]
    if has_tri and has_fly:
        cols += [("vs-triton", 11)]

    header = "".join(f"{name:>{w}}" for name, w in cols)
    print()
    print(header)
    print("-" * len(header))

    def cell(val, fmt):
        return "N/A" if val is None else format(val, fmt)

    for r in rows:
        out = [f"{str(r['idx']):>5}", f"{_fmt_shape(r['shape']):>26}"]
        if has_tri:
            out += [
                f"{cell(r['times']['triton'], '.4f'):>12}",
                f"{cell(r['tflops']['triton'], '.2f'):>13}",
                f"{r['verify'].get('triton', 'N/A'):>9}",
            ]
        if has_fly:
            out += [
                f"{cell(r['times']['flydsl'], '.4f'):>12}",
                f"{cell(r['tflops']['flydsl'], '.2f'):>13}",
                f"{r['verify'].get('flydsl', 'N/A'):>9}",
            ]
        if has_tri and has_fly:
            out += [f"{r['speedup']:>11}"]
        print("".join(out))


# Names of the per-shape manual override flags (no defaults -- all or nothing).
_MANUAL_SHAPE_FLAGS = ("batch_size", "num_heads_q", "head_dim", "seq_q_l", "seq_kv_l")


def _resolve_shapes(args):
    """Resolve the shape selection into a list of (idx, (bs, s_q, s_kv, H, D)).

    ``idx`` is the 1-based case number shown in the table and accepted by
    ``--shape-index`` (converted to a 0-based offset into PRESET_SHAPES here).

    Three mutually-exclusive modes:
      * default (no shape args)      -> the full PRESET_SHAPES sweep
      * --shape-index N              -> a single preset shape by 1-based index
      * manual flags (all required)  -> a single custom shape
    """
    manual = {f: getattr(args, f) for f in _MANUAL_SHAPE_FLAGS}
    any_manual = any(v is not None for v in manual.values())

    if args.shape_index is not None and any_manual:
        raise SystemExit(
            "[error] --shape-index and manual shape flags are mutually exclusive."
        )

    if args.shape_index is not None:
        if not (1 <= args.shape_index <= len(PRESET_SHAPES)):
            raise SystemExit(
                f"[error] --shape-index must be in [1, {len(PRESET_SHAPES)}]; "
                f"got {args.shape_index}. Use --list to see the preset shapes."
            )
        zero_based = args.shape_index - 1
        return [(args.shape_index, PRESET_SHAPES[zero_based])]

    if any_manual:
        missing = [f for f, v in manual.items() if v is None]
        if missing:
            raise SystemExit(
                "[error] manual shape requires all of "
                f"{list(_MANUAL_SHAPE_FLAGS)}; missing: {missing}."
            )
        # "M" marks a manual (non-preset) shape -- not re-runnable by index.
        return [
            (
                "M",
                (
                    manual["batch_size"],
                    manual["seq_q_l"],
                    manual["seq_kv_l"],
                    manual["num_heads_q"],
                    manual["head_dim"],
                ),
            )
        ]

    return [(i + 1, s) for i, s in enumerate(PRESET_SHAPES)]


def main():
    p = argparse.ArgumentParser(
        description="FlyDSL vs Triton FP8 MQA Logits A/B benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Shape selection (3 mutually-exclusive modes; see _resolve_shapes).
    p.add_argument(
        "--shape-index",
        type=int,
        default=None,
        help="run a single shape from the preset sweep by 1-based index (see --list)",
    )
    # Manual override flags -- NO defaults; supplying any requires all of them.
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--num_heads_q", type=int, default=None)
    p.add_argument("--head_dim", type=int, default=None)
    p.add_argument("--seq_q_l", type=int, default=None)
    p.add_argument("--seq_kv_l", type=int, default=None)

    p.add_argument(
        "--impl",
        choices=["all", "triton", "flydsl"],
        default="all",
        help="which implementation(s) to run",
    )
    p.add_argument("--clean_logits", type=int, default=1)
    p.add_argument("--warmup", type=int, default=25)
    p.add_argument("--rep", type=int, default=100)
    p.add_argument(
        "--verify",
        action="store_true",
        help="check each impl against the torch reference (calc_diff < 1e-3)",
    )
    p.add_argument(
        "--list",
        action="store_true",
        help="list the preset shapes (with their --shape-index) and exit",
    )
    args = p.parse_args()

    if args.list:
        print("preset shapes (index: bs, s_q, s_kv, H, D):")
        for i, s in enumerate(PRESET_SHAPES):
            print(f"  {i + 1}: {s}")
        return

    run(args)


if __name__ == "__main__":
    main()
