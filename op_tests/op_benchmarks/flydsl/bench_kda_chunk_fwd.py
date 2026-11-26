# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""FlyDSL vs Triton KDA chunk forward (``chunk_kimi_delta_attn``) timing.

Kimi-K3's call: H x 128, bf16, chunk 64, in-kernel gate with lower_bound=-5,
q/k l2norm and beta sigmoid, no state. ``-s BxT`` runs B sequences of T
tokens; a trailing ``v`` packs them varlen with ``cu_seqlens``. Both backends
get the same inputs; the table also shows their max abs difference.

Per-stage FlyDSL times (intra / k5 / output), one line per call::

    AITER_KDA_FLYDSL_PROFILE=1 python3 op_tests/op_benchmarks/flydsl/bench_kda_chunk_fwd.py
"""

import argparse

import pandas as pd
import torch

from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops.triton.kimi_delta_attn import chunk_kimi_delta_attn
from aiter.test_common import run_perftest

D = 128


def inputs(b, t, h, varlen, seed=0):
    gen = torch.Generator(device="cuda").manual_seed(seed)

    def r(*shape):
        return torch.randn(*shape, generator=gen, device="cuda")

    B, T = (1, b * t) if varlen else (b, t)
    bf = torch.bfloat16
    x = {
        "q": r(B, T, h, D).to(bf),
        "k": r(B, T, h, D).to(bf),
        "v": r(B, T, h, D).to(bf),
        "g": r(B, T, h, D),
        "beta": r(B, T, h),
        "A_log": torch.log(torch.rand(h, generator=gen, device="cuda") * 15 + 1),
        "dt_bias": r(h * D) * 0.5,
    }
    if varlen:
        x["cu_seqlens"] = torch.arange(0, b * t + 1, t, device="cuda")
    return x


def bench(b, t, h, varlen, iters):
    x = inputs(b, t, h, varlen)
    kw = {
        "scale": D**-0.5,
        "use_gate_in_kernel": True,
        "lower_bound": -5.0,
        "safe_gate": True,
        "use_qk_l2norm_in_kernel": True,
        "use_beta_sigmoid_in_kernel": True,
        "chunk_size": 64,
    }
    row = {"shape": f"{b}x{t}{'v' if varlen else ''}", "H": h}
    outs = {}
    for be in ("triton", "flydsl"):
        (o, _), us = run_perftest(
            lambda be=be: chunk_kimi_delta_attn(**x, **kw, fwd_backend=be),
            num_iters=iters,
            num_warmup=3,
        )
        outs[be] = o.float()
        row[f"{be} us"] = us
        row[f"{be} TFLOPS"] = (
            2 * b * t * h * (4 * 64 * D + 4 * D * D + 64 * D) / us / 1e6
        )
    row["flydsl/triton"] = row["flydsl us"] / row["triton us"]
    row["max |diff|"] = (outs["flydsl"] - outs["triton"]).abs().max().item()
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "-s",
        "--shape",
        nargs="*",
        default=["1x1000", "1x2048", "1x8192", "8x2048v"],
        help="BxT, trailing v = varlen",
    )
    parser.add_argument("-H", "--heads", type=int, nargs="*", default=[96])
    parser.add_argument("--iters", type=int, default=50)
    args = parser.parse_args()
    rows = []
    for sh in args.shape:
        varlen = sh.endswith("v")
        b, t = (int(n) for n in sh.rstrip("v").split("x"))
        for h in args.heads:
            rows.append(bench(b, t, h, varlen, args.iters))
    print(f"gfx: {get_gfx_runtime()}")
    print(pd.DataFrame(rows).to_markdown(index=False, floatfmt=".3g"))


if __name__ == "__main__":
    main()
