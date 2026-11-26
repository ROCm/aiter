# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Time the KDA chunk backward: FlyDSL (``chunk_kda_bwd_flydsl_full``) vs Triton
(``chunk_delta_attn_bwd``), on the same GPU and operands.

Inputs follow Kimi-K3's setting: q/k l2norm, the ``lower_bound=-5`` gate and
beta sigmoid in kernel, safe_gate, bf16, K = V = 128, no initial state. Both
backends are called directly (no autograd) and include their forward-
intermediate recompute. Per shape the table reports the mean CUDA-event time
over ``--iters`` calls, nominal TFLOPS (the op_test's count) and the Triton /
FlyDSL ratio; ``--stages`` adds one profiled FlyDSL call per shape that prints
its per-stage times (``AITER_KDA_FLYDSL_PROFILE``). Accuracy is covered by
``op_tests/test_kda_chunk_bwd.py``; this script only times.

Shapes are ``b,t,h,varlen``: ``varlen=1`` packs ``b`` sequences of ``t``
tokens with ``cu_seqlens``. Usage, from the repo root:

    python -m op_tests.op_benchmarks.flydsl.bench_kda_chunk_bwd
    python -m op_tests.op_benchmarks.flydsl.bench_kda_chunk_bwd \\
        -s 1,32768,96,0 8,2048,96,1 --prepare triton flydsl
"""

from __future__ import annotations

import argparse
import importlib

import pandas as pd
import torch

from aiter.ops.flydsl import kda_chunk_bwd as fb
from aiter.ops.triton._triton_kernels.kimi_delta_attn.chunk_bwd import (
    chunk_delta_attn_bwd,
)

D = 128
BT = 64
DEFAULT_SHAPES = ["1,1000,96,0", "1,2048,96,0", "1,8192,96,0", "8,2048,96,1"]


def make_inputs(b, t, h, varlen, seed=0):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    rows, tokens = (1, b * t) if varlen else (b, t)

    def r(*shape, dtype=torch.bfloat16):
        x = torch.randn(*shape, device="cuda", generator=gen)
        return x.to(dtype)

    x = {
        "do": r(rows, tokens, h, D),
        "q": r(rows, tokens, h, D),
        "k": r(rows, tokens, h, D),
        "v": r(rows, tokens, h, D),
        "g": r(rows, tokens, h, D),
        "beta": r(rows, tokens, h, dtype=torch.float32),
        "A_log": r(h, dtype=torch.float32) * 0.5,
        "dt_bias": r(h * D, dtype=torch.float32) * 0.5,
    }
    cu = (
        torch.arange(0, b * t + 1, t, dtype=torch.int32, device="cuda")
        if varlen
        else None
    )
    return x, cu


def call(backend, x, cu, prepare):
    kw = {
        "cu_seqlens": cu,
        "chunk_size": BT,
        "safe_gate": True,
        "lower_bound": -5.0,
        "use_gate_in_kernel": True,
        "A_log": x["A_log"],
        "dt_bias": x["dt_bias"],
        "use_qk_l2norm_in_kernel": True,
        "use_beta_sigmoid_in_kernel": True,
    }
    args = (x["do"], None, x["q"], x["k"], x["v"], x["g"], x["beta"], D**-0.5)
    if backend == "triton":
        return chunk_delta_attn_bwd(*args, **kw)
    return fb.chunk_kda_bwd_flydsl_full(*args, prepare=prepare, **kw)


def time_us(fn, args, warmup, iters):
    for _ in range(warmup):
        fn(*args)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn(*args)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1e3 / iters


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter, description=__doc__
    )
    parser.add_argument("-s", "--shape", nargs="*", default=DEFAULT_SHAPES)
    parser.add_argument(
        "--prepare",
        nargs="*",
        default=["triton"],
        help="FlyDSL forward-prepare backends to time (registered names)",
    )
    parser.add_argument(
        "--prepare-module",
        nargs="*",
        default=[],
        help="modules to import first; they call register_fwd_prepare",
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument(
        "--no-stages", dest="stages", action="store_false", help="skip the breakdown"
    )
    parser.add_argument("-o", "--output", default=None, help="also write a CSV")
    args = parser.parse_args()
    for mod in args.prepare_module:
        importlib.import_module(mod)

    rows = []
    for spec in args.shape:
        b, t, h, vl = (int(v) for v in spec.split(","))
        x, cu = make_inputs(b, t, h, bool(vl))
        tokens = b * t
        flops = 2 * h * tokens * (4 * BT * D + 2 * BT * D + 6 * D * D)
        row = {"b": b, "t": t, "h": h, "varlen": vl}
        tri = time_us(call, ("triton", x, cu, None), args.warmup, args.iters)
        row["triton us"] = round(tri, 1)
        row["triton TFLOPS"] = round(flops / tri / 1e6, 2)
        for prep in args.prepare:
            name = "flydsl" if prep == "triton" else f"flydsl+{prep}"
            us = time_us(call, ("flydsl", x, cu, prep), args.warmup, args.iters)
            row[f"{name} us"] = round(us, 1)
            row[f"{name} TFLOPS"] = round(flops / us / 1e6, 2)
            row[f"triton/{name}"] = round(tri / us, 3)
            if args.stages:
                fb._PROFILE = True
                try:
                    call("flydsl", x, cu, prep)
                finally:
                    fb._PROFILE = False
        rows.append(row)
        del x, cu
        torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    print(df.to_markdown(index=False))
    if args.output:
        df.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
