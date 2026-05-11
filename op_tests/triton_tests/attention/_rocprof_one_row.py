#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
Run a single CSV row's CK + Triton calls for rocprofv3 to time.

This is the helper invoked under `rocprofv3 -- python _rocprof_one_row.py
--idx N --csv pawel-2d-3d.csv --warmup 10 --iters 50 --side {ck,triton,both}`.

Kept deliberately small so the trace contains *only* the kernels we care
about: CK unified attention or the Triton unified attention wrapper.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import torch

from aiter.ops.unified_attention import unified_attention_fwd
from aiter.ops.triton.attention import unified_attention as ua_mod


def make_inputs(row, num_blocks_cap=16384, dtype=torch.bfloat16):
    b   = int(row["num_seqs"])
    hq  = int(row["num_q_heads"])
    hk  = int(row["num_kv_heads"])
    d   = int(row["head_size"])
    blk = int(row["block_size"])
    sq  = int(row["max_seqlen_q"])
    sk  = int(row["max_seqlen_k"])
    total_q = int(row["total_q_tokens"])
    scale   = 1.0 / math.sqrt(d)

    max_blks_per_seq = (sk + blk - 1) // blk
    nb = min(num_blocks_cap, max(1024, 2 * max_blks_per_seq))

    q = torch.randn(total_q, hq, d, dtype=dtype, device="cuda")
    k = torch.randn(nb, blk, hk, d, dtype=dtype, device="cuda")
    v = torch.randn_like(k)

    if sq * b == total_q:
        cu = torch.arange(0, b + 1, dtype=torch.int32, device="cuda") * sq
    else:
        base = total_q // b
        rem  = total_q - base * b
        lens = [min(sq, base + (1 if i < rem else 0)) for i in range(b)]
        miss = total_q - sum(lens)
        if miss > 0 and lens[0] + miss <= sq:
            lens[0] += miss
        elif miss > 0:
            for i in range(b):
                if miss == 0:
                    break
                room = sq - lens[i]
                add  = min(room, miss)
                lens[i] += add
                miss   -= add
        cu = torch.tensor([0] + list(torch.tensor(lens).cumsum(0).tolist()),
                          dtype=torch.int32, device="cuda")

    seq_lens_k   = torch.full((b,), sk, dtype=torch.int32, device="cuda")
    block_tables = torch.randint(0, nb, (b, max_blks_per_seq),
                                 dtype=torch.int32, device="cuda")

    return q, k, v, cu, seq_lens_k, block_tables, scale, sq, sk, blk


def _disable_ck_short_circuits():
    """Force the Triton wrapper to actually launch a Triton kernel.

    The production wrapper has two short-circuits that can silently route
    to CK (`_try_ck_splitkv_attention`, `_try_ck_unified_attention`). For
    a fair "is CK faster than Triton" comparison we have to neuter those
    so that 'Triton' really means Triton kernels."""
    for attr in ("_try_ck_splitkv_attention", "_try_ck_unified_attention"):
        if hasattr(ua_mod, attr):
            setattr(ua_mod, attr, lambda *a, **kw: False)


def _force_triton_path(which):
    """Force the wrapper's 2D-vs-3D selector to a specific path."""
    assert which in ("2d", "3d")
    ua_mod.use_2d_kernel = (
        (lambda *a, **kw: True) if which == "2d"
        else (lambda *a, **kw: False)
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--idx", required=True)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters",  type=int, default=50)
    p.add_argument("--side", choices=["ck", "triton_2d", "triton_3d", "all"],
                   default="all",
                   help="Which kernel path to time. 'all' runs CK + 2D + 3D in "
                        "one process so a single rocprofv3 invocation covers "
                        "everything. 2D/3D both disable the CK short-circuits.")
    p.add_argument("--use-graph", action="store_true",
                   help="capture each call into a CUDA graph and replay")
    p.add_argument("--check-correctness", action="store_true",
                   help="run each enabled path once on the same inputs (with "
                        "different output buffers), allclose against CK at "
                        "atol=2e-2 rtol=1e-2; print + non-zero exit on mismatch")
    args = p.parse_args()

    with open(args.csv) as f:
        for r in csv.DictReader(f):
            if r["idx"] == args.idx:
                row = r
                break
        else:
            raise SystemExit(f"idx {args.idx} not found in {args.csv}")

    torch.manual_seed(42)
    inp = make_inputs(row)
    q, k, v, cu, seq_lens_k, bt, scale, sq, sk, blk = inp
    out_ck   = torch.empty_like(q)
    out_t2d  = torch.empty_like(q)
    out_t3d  = torch.empty_like(q)
    window = tuple(int(x) for x in row["window_size"].split(","))

    def call_ck(out=out_ck):
        unified_attention_fwd(
            out, q, k, v, bt, seq_lens_k, cu,
            mask_type=2, scale_s=scale,
            scale=1.0, scale_k=1.0, scale_v=1.0, scale_out=1.0,
        )

    def call_triton(out):
        ua_mod.unified_attention(
            q=q, k=k, v=v, out=out, cu_seqlens_q=cu,
            max_seqlen_q=sq, seqused_k=seq_lens_k, max_seqlen_k=sk,
            softmax_scale=scale, causal=True, window_size=window,
            block_table=bt, softcap=0.0,
            q_descale=None, k_descale=None, v_descale=None,
            alibi_slopes=None, output_scale=None, qq_bias=None, sinks=None,
        )

    def run_loop(fn):
        for _ in range(args.warmup):
            fn()
        torch.cuda.synchronize()
        if args.use_graph:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                fn()
            torch.cuda.synchronize()
            for _ in range(args.iters):
                graph.replay()
        else:
            for _ in range(args.iters):
                fn()
        torch.cuda.synchronize()

    # --- correctness check (once per path, fresh inputs not needed) ---
    correctness_msg = ""
    if args.check_correctness:
        # always run CK once for the reference output
        call_ck(out=out_ck)
        torch.cuda.synchronize()
        ck_ref = out_ck.float().clone()
        check_lines = []
        if args.side in ("triton_2d", "all"):
            _disable_ck_short_circuits()
            _force_triton_path("2d")
            call_triton(out=out_t2d)
            torch.cuda.synchronize()
            d = (out_t2d.float() - ck_ref).abs().max().item()
            ok = torch.allclose(out_t2d.float(), ck_ref, atol=2e-2, rtol=1e-2)
            check_lines.append(f"  ck vs triton_2d: max_abs_diff={d:.6f} match={ok}")
            if not ok:
                raise SystemExit(f"correctness FAIL ck vs 2d: max_diff={d}")
        if args.side in ("triton_3d", "all"):
            _disable_ck_short_circuits()
            _force_triton_path("3d")
            call_triton(out=out_t3d)
            torch.cuda.synchronize()
            d = (out_t3d.float() - ck_ref).abs().max().item()
            ok = torch.allclose(out_t3d.float(), ck_ref, atol=2e-2, rtol=1e-2)
            check_lines.append(f"  ck vs triton_3d: max_abs_diff={d:.6f} match={ok}")
            if not ok:
                raise SystemExit(f"correctness FAIL ck vs 3d: max_diff={d}")
        correctness_msg = "\n".join(check_lines)

    # --- timed runs ---
    if args.side in ("ck", "all"):
        # CK path: no short-circuit interference possible, direct call.
        run_loop(call_ck)

    if args.side in ("triton_2d", "all"):
        _disable_ck_short_circuits()
        _force_triton_path("2d")
        run_loop(lambda: call_triton(out=out_t2d))

    if args.side in ("triton_3d", "all"):
        _disable_ck_short_circuits()
        _force_triton_path("3d")
        run_loop(lambda: call_triton(out=out_t3d))

    print(f"done idx={args.idx} side={args.side} warmup={args.warmup} "
          f"iters={args.iters} use_graph={args.use_graph}")
    if correctness_msg:
        print(correctness_msg)


if __name__ == "__main__":
    main()
