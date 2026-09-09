# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Benchmark the FlyDSL varlen FMHA backward on gfx942.

Times -- and checks against the same fp32 torch reference
op_tests/test_mha_flydsl_varlen_bwd.py uses -- two groups of backends on
identical causal bf16 THD batches.  Both groups run the same tokens, sequence
pattern and head count, so only the head dims differ between them:

  ``192/128``  the shape this kernel exists for (d_qk=192, d_v=128)

      flydsl     the FlyDSL backward, native to the unpadded d_v = 128 shape
      asm_vpad   ASM v3 with v zero-padded 128 -> 192 to open its gate.  v/out
                 come pre-padded from the padded forward such a model would
                 already run; padding dout and narrowing dv back to 128 are
                 charged to the backward.  Runs the rtz dS -> bf16 conversion
                 (`how_v3_bf16_cvt=2`, see `V3_BF16_CVT_RTZ`).
      ck         what the router reaches without FlyDSL -- the varlen ASM v3
                 gate requires hdim_q == hdim_v, so d_qk=192 / d_v=128 falls
                 through to generic CK-tile.  The group's speedup baseline.

  ``128/128``  the square shape, as a reference point rather than an
               alternative: no model runs 192/128 attention at d=128.  It says
               what this hardware does on a head dim the ASM path was written
               for, which is the ceiling the 192/128 numbers are chasing.

      asm_128    ASM v3, natively eligible -- no padding anywhere.
      ck_128     CK-tile at d=128.  The group's speedup baseline.

One row per (group, backend), reporting the median of `--iters` per-iteration
GPU latencies plus the 20th/80th percentiles, the TFLOPS that median implies,
and the speedup over that row's own group baseline.  Percentiles rather than a
mean because a stray host-side stall skews a mean and leaves no trace in the
table; a p80 well above the median is the visible symptom.

Shapes here are uniform batches (`num_seqs` x `seqlen`); the ragged model
patterns and the pytest accuracy gate live in
op_tests/test_mha_flydsl_varlen_bwd.py, which this module imports its input
builder, reference and roofline helpers from.

Usage -- run from the repo root, which has to be on ``sys.path`` for the
``op_tests`` import below to resolve (``-m`` does that; a plain script path
would put this file's own directory there instead):

    # Default sweep, all backends
    python -m op_tests.op_benchmarks.flydsl.bench_mha_bwd

    # One shape, skipping the (slow) fp32 reference
    python -m op_tests.op_benchmarks.flydsl.bench_mha_bwd -n 1 -s 4096 -nh 16 --no-check

    # FlyDSL against its own group's baseline only, also written to CSV
    python -m op_tests.op_benchmarks.flydsl.bench_mha_bwd --backend flydsl ck -o bwd.csv
"""

from __future__ import annotations

import argparse
import itertools

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.fmha_kernels import flydsl_flash_attn_varlen_bwd
from aiter.ops.mha import fmha_v3_varlen_bwd, mha_varlen_bwd
from aiter.test_common import checkAllclose
from op_tests.test_mha_flydsl_varlen_bwd import (
    ATOL,
    HEAD_DIM_QK,
    HEAD_DIM_V,
    RTOL,
    SUPPORTED_GFX,
    TOL_ERR_RATIO,
    flops_bytes,
    make_varlen_inputs,
    mean_rel,
    run_torch,
)

torch.set_default_device("cuda")

# Group -> (d_qk, d_v).  Insertion order is table order.
GROUPS = {
    "192/128": (HEAD_DIM_QK, HEAD_DIM_V),
    "128/128": (HEAD_DIM_V, HEAD_DIM_V),
}

# Backend -> group, and the backend each group's speedup column divides by.
BACKEND_GROUP = {
    "flydsl": "192/128",
    "asm_vpad": "192/128",
    "ck": "192/128",
    "asm_128": "128/128",
    "ck_128": "128/128",
}
BASELINE = {"192/128": "ck", "128/128": "ck_128"}
BACKENDS = tuple(BACKEND_GROUP)

# dS -> bf16 rounding mode for the ASM v3 backward: 0 = rtne, 1 = rtna, 2 = rtz.
# Each mode is a separately compiled kernel, and hsa/gfx942/fmha_v3_bwd/
# fmha_bwd_dqdkdv.csv carries all three for every bf16 config -- the 192/128
# shape resolves to bwd_hd192_bf16_causal_a32_rtz_psskddv_group.co.  Pinned to
# rtz (truncation, the cheapest of the three) rather than left on the router's
# rtna default, so the baseline is timed on its fastest conversion.
V3_BF16_CVT_RTZ = 2

# Sweep defaults.  `num_seqs` x `seqlen` spans both of the kernel's dispatch
# regimes: small grids (few short sequences) are co-resident on one round of
# workgroups and take the split-K path, large ones do not.  See `_split()` in
# the kernel's launcher.
NUM_SEQS = [1, 4, 16]
SEQLENS = [1024, 2048, 4096]
NHEADS = [2, 8]

# The columns the summary table shows; the DataFrame carries more (shape, TB/s,
# accuracy) and those reach the CSV.
DISPLAY_COLS = [
    "group",
    "backend",
    "median_ms",
    "p20",
    "p80",
    "TFLOPS",
    "speedup vs its CK",
]


def time_latencies(fn, num_iters, num_warmup) -> tuple[object, np.ndarray]:
    """Per-iteration GPU latencies in ms, plus whatever `fn` last returned.

    Every iteration is bracketed by its own event pair and the whole run is
    synchronized once at the end, so the launches stay pipelined: a sync per
    iteration would drain the queue each time and bill the drain to a kernel
    that in practice never pays it.
    """
    for _ in range(num_warmup):
        fn()
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]
    for i in range(num_iters):
        starts[i].record()
        out = fn()
        ends[i].record()
    torch.cuda.synchronize()

    return out, np.array([s.elapsed_time(e) for s, e in zip(starts, ends)])


def candidates_192(x, dq, dk, dv, names):
    """The d_qk=192 / d_v=128 backends, each returning (dq, dk, dv)."""
    cand = {}

    if "flydsl" in names:

        def _flydsl():
            flydsl_flash_attn_varlen_bwd(
                x.dout,
                x.q,
                x.k,
                x.v,
                x.out,
                x.lse,
                dq,
                dk,
                dv,
                x.cu_seqlens,
                x.max_seqlen,
                x.max_seqlen,
                x.scale,
            )
            return dq, dk, dv

        cand["flydsl"] = _flydsl

    if "asm_vpad" in names:
        # v/out are padded once, outside the timed callable: a model on this
        # path runs a padded forward and already holds them at d_qk.  dout's
        # pad and dv's narrowing are inside, since they exist only to serve
        # this backward.
        pad = x.d_qk - x.d_v
        v_pad = F.pad(x.v, (0, pad))
        out_pad = F.pad(x.out, (0, pad))
        dv_pad = torch.empty_like(v_pad)

        def _asm_vpad():
            fmha_v3_varlen_bwd(
                F.pad(x.dout, (0, pad)),
                x.q,
                x.k,
                v_pad,
                out_pad,
                x.lse,
                x.cu_seqlens,
                x.cu_seqlens,
                x.max_seqlen,
                x.max_seqlen,
                dropout_p=0.0,
                softmax_scale=x.scale,
                zero_tensors=False,
                is_causal=True,
                window_size_left=-1,
                window_size_right=-1,
                deterministic=False,
                is_v3_atomic_fp32=True,
                how_v3_bf16_cvt=V3_BF16_CVT_RTZ,
                dq=dq,
                dk=dk,
                dv=dv_pad,
            )
            return dq, dk, dv_pad[..., : x.d_v].contiguous()

        cand["asm_vpad"] = _asm_vpad

    if "ck" in names:
        cand["ck"] = _ck_call(x, dq, dk, dv)

    return cand


def candidates_128(x, dq, dk, dv, names):
    """The square d=128 backends.  ASM v3 is natively eligible here."""
    cand = {}

    if "asm_128" in names:

        def _asm_128():
            fmha_v3_varlen_bwd(
                x.dout,
                x.q,
                x.k,
                x.v,
                x.out,
                x.lse,
                x.cu_seqlens,
                x.cu_seqlens,
                x.max_seqlen,
                x.max_seqlen,
                dropout_p=0.0,
                softmax_scale=x.scale,
                zero_tensors=False,
                is_causal=True,
                window_size_left=-1,
                window_size_right=-1,
                deterministic=False,
                is_v3_atomic_fp32=True,
                how_v3_bf16_cvt=V3_BF16_CVT_RTZ,
                dq=dq,
                dk=dk,
                dv=dv,
            )
            return dq, dk, dv

        cand["asm_128"] = _asm_128

    if "ck_128" in names:
        cand["ck_128"] = _ck_call(x, dq, dk, dv)

    return cand


def _ck_call(x, dq, dk, dv):
    """CK-tile varlen backward; the baseline of whichever group it is in."""

    def _ck():
        mha_varlen_bwd(
            x.dout,
            x.q,
            x.k,
            x.v,
            x.out,
            x.lse,
            x.cu_seqlens,
            x.cu_seqlens,
            x.max_seqlen,
            x.max_seqlen,
            dropout_p=0.0,
            softmax_scale=x.scale,
            zero_tensors=False,
            is_causal=True,
            window_size_left=-1,
            window_size_right=-1,
            deterministic=False,
            dq=dq,
            dk=dk,
            dv=dv,
        )
        return dq, dk, dv

    return _ck


def bench_group(group, seqlens, nheads, dtype, names, iters, warmup, check):
    """Time one group's backends on one shape.  Returns a list of row dicts."""
    d_qk, d_v = GROUPS[group]
    x = make_varlen_inputs(seqlens, nheads, dtype, d_qk=d_qk, d_v=d_v)

    ref = None
    if check:
        ref = run_torch(x.dout, x.q, x.k, x.v, x.out, x.lse, x.cu_seqlens, x.scale)

    # Preallocated and shared by every backend in the group: autograd allocates
    # these per backward, so reusing them across the timed repeats keeps the
    # measurement on the kernel rather than on the caching allocator.
    dq = torch.empty_like(x.q)
    dk = torch.empty_like(x.k)
    dv = torch.empty_like(x.v)

    builder = candidates_192 if group == "192/128" else candidates_128
    cand = builder(x, dq, dk, dv, names)

    flops, nbytes = flops_bytes(seqlens, nheads, x.q.element_size(), d_qk, d_v)

    rows = []
    for name, fn in cand.items():
        got, lat = time_latencies(fn, iters, warmup)
        p20, median, p80 = np.percentile(lat, [20, 50, 80])
        row = {
            "group": group,
            "backend": name,
            "median_ms": median,
            "p20": p20,
            "p80": p80,
            "TFLOPS": flops / median / 1e9,
            "TB/s": nbytes / median / 1e9,
        }
        if check:
            grads = tuple(zip("qkv", ref, got))
            row["err"] = max(
                checkAllclose(
                    r.to(dtypes.fp32),
                    g.to(dtypes.fp32),
                    rtol=RTOL,
                    atol=ATOL,
                    tol_err_ratio=TOL_ERR_RATIO,
                    msg=f"{group} {name}: d{tag}",
                )
                for tag, r, g in grads
            )
            # Worst of dq/dk/dv, so one column still catches a single bad gradient.
            row["mean_rel"] = max(mean_rel(r, g) for _, r, g in grads)
        rows.append(row)

    # The 128/128 group holds a second full set of activations; drop this one
    # before the next group builds its own.
    del x, dq, dk, dv, cand, ref
    torch.cuda.empty_cache()
    return rows


def bench_shape(num_seqs, seqlen, nheads, dtype, names, iters, warmup, check):
    """Every selected backend on one shape, ordered group by group."""
    seqlens = [seqlen] * num_seqs
    rows = []
    for group in GROUPS:
        selected = [n for n in names if BACKEND_GROUP[n] == group]
        if not selected:
            continue
        rows += bench_group(
            group, seqlens, nheads, dtype, selected, iters, warmup, check
        )

    # Speedup is against the row's own group baseline, so it is only defined
    # where that baseline was actually timed.
    base = {
        r["group"]: r["median_ms"] for r in rows if r["backend"] == BASELINE[r["group"]]
    }
    for r in rows:
        b = base.get(r["group"])
        r["speedup vs its CK"] = b / r["median_ms"] if b else float("nan")
        r["num_seqs"] = num_seqs
        r["seqlen"] = seqlen
        r["nheads"] = nheads
    return rows


def format_table(df):
    """The pictured columns, fastest first within each group."""
    out = df[DISPLAY_COLS].copy()
    order = {g: i for i, g in enumerate(GROUPS)}
    out = out.sort_values(
        ["group", "median_ms"], key=lambda c: c.map(order) if c.name == "group" else c
    )
    for col in ("median_ms", "p20", "p80"):
        out[col] = out[col].map("{:.4f}".format)
    out["TFLOPS"] = out["TFLOPS"].map("{:.1f}".format)
    out["speedup vs its CK"] = out["speedup vs its CK"].map(
        lambda v: "n/a" if np.isnan(v) else f"{v:.2f}x"
    )
    return out.to_markdown(index=False)


def parse_args():
    parser = argparse.ArgumentParser(
        prog="bench_mha_bwd",
        formatter_class=argparse.RawTextHelpFormatter,
        description="Benchmark the FlyDSL varlen FMHA backward "
        "(d_qk=192, d_v=128, causal).",
    )
    parser.add_argument(
        "-d",
        "--dtype",
        type=dtypes.str2Dtype,
        choices=[dtypes.d_dtypes["bf16"]],
        nargs="*",
        default="bf16,",
        metavar="{bf16}",
        help="data type.\ne.g.: -d bf16",
    )
    parser.add_argument(
        "-n",
        "--num-seqs",
        type=int,
        nargs="*",
        dest="num_seqs",
        default=NUM_SEQS,
        help="number of sequences in the batch.\ne.g.: -n 1 8",
    )
    parser.add_argument(
        "-s",
        "--seqlen",
        type=int,
        nargs="*",
        default=SEQLENS,
        help="per-sequence length.\ne.g.: -s 4096",
    )
    parser.add_argument(
        "-nh",
        "--nheads",
        type=int,
        nargs="*",
        default=NHEADS,
        help="number of attention heads.\ne.g.: -nh 2 8",
    )
    parser.add_argument(
        "--backend",
        type=str,
        nargs="*",
        choices=list(BACKENDS),
        default=list(BACKENDS),
        help=f"backend(s) to time, from {list(BACKENDS)}."
        "\nDrop a group's baseline and that group's speedup reads n/a."
        "\ne.g.: --backend flydsl ck",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=100,
        help="timed iterations per backend.\ne.g.: --iters 500",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="untimed warmup iterations per backend (the first FlyDSL call also"
        "\npays a JIT compile).\ne.g.: --warmup 25",
    )
    parser.add_argument(
        "--no-check",
        dest="check",
        action="store_false",
        help="skip the fp32 torch reference.  It is a per-(sequence, head)"
        "\nPython loop and dominates the wall time of a large sweep.",
    )
    parser.add_argument(
        "-o",
        type=str,
        metavar="FILE",
        help="also write the full results (incl. shape, TB/s, accuracy) to this"
        "\nCSV file.",
    )
    return parser.parse_args()


def main():
    # Parsed before the arch gate so `--help` works everywhere.
    args = parse_args()

    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning(
            "flydsl varlen fmha backward unsupported on %s; skipping", get_gfx()
        )
        return

    names = [b for b in BACKENDS if b in args.backend]
    if not names:
        return

    for dtype in args.dtype:
        frames = []
        for num_seqs, seqlen, nheads in itertools.product(
            args.num_seqs, args.seqlen, args.nheads
        ):
            df = pd.DataFrame(
                bench_shape(
                    num_seqs,
                    seqlen,
                    nheads,
                    dtype,
                    names,
                    args.iters,
                    args.warmup,
                    args.check,
                )
            )
            aiter.logger.info(
                "varlen fmha backward, %d x %d tokens, %d heads, %s (markdown):\n%s",
                num_seqs,
                seqlen,
                nheads,
                dtype,
                format_table(df),
            )
            frames.append(df)

        if args.o:
            pd.concat(frames, ignore_index=True).to_csv(args.o, index=False)
            aiter.logger.info("results saved to %s", args.o)


if __name__ == "__main__":
    main()
