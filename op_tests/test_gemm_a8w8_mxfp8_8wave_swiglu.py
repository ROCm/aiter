# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Bytes and timing of the 8-wave MXFP8 gate_up GEMM whose epilogue applies
SwiGLU-OAI and the MXFP8 quant of the down projection's input
(aiter.ops.flydsl.gemm_a8w8_mxfp8_8wave.flydsl_mxfp8_8wave_gemm_swiglu_w4).

The reference is the same 8-wave GEMM with its bf16 output, then SwiGLU-OAI and
the MXFP8 quant in torch (fp32, in the epilogue's order, one op per kernel, so
nothing is contracted), its scales laid out by aiter.ops.shuffle.shuffle_scale.
The e4m3 codes, and the e8m0 scales of rows below M, must be equal byte for
byte. ``wide`` spreads the activation scales so the gate reaches the sigmoid's
underflow (gate <= -51.3) and the up clamp; shapes are MiniMax-M3.1 TP1's
shared expert (I 3072) and dense MLP (I 12288) gate_up, K 6144.

    python3 op_tests/test_gemm_a8w8_mxfp8_8wave_swiglu.py
"""

import argparse
import sys

import pandas as pd
import torch

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops.flydsl.gemm_a8w8_mxfp8_8wave import (
    flydsl_mxfp8_8wave_gemm,
    flydsl_mxfp8_8wave_gemm_swiglu_w4,
)
from aiter.ops.shuffle import shuffle_scale, shuffle_weight
from aiter.test_common import benchmark, run_perftest

torch.set_default_device("cuda")
ALPHA, BETA, LIMIT = 1.702, 1.0, 7.0
K = 6144


def swiglu_oai(gate_up):
    """bf16 [M, 2I] -> SwiGLU-OAI in the epilogue's fp32 order, rounded to bf16."""
    inter = gate_up.shape[1] // 2
    gate, up = gate_up[:, :inter].float(), gate_up[:, inter:].float()
    g = torch.minimum(gate, torch.tensor(LIMIT))
    m = (ALPHA * -g) * 1.4426950408889634
    sig = 1.0 / (torch.exp2(m) + 1.0)
    u = up.clamp(-LIMIT, LIMIT)
    return ((BETA + u) * (g * sig)).to(torch.bfloat16)


def mxfp8(x):
    """MXFP8 1x32 of a bf16 [M, I]: e4m3 codes, row-major e8m0 scale bytes."""
    m, n = x.shape
    xf = x.float().view(m, n // 32, 32)
    amax = xf.abs().amax(-1).clamp_min(1e-10)
    bits = (amax * (1.0 / 448.0)).view(torch.int32)
    code = (bits >> 23) + ((bits & 0x7FFFFF) != 0).int()
    recip = ((254 - code).clamp_min(0) << 23).view(torch.float32)
    q = (xf * recip.unsqueeze(-1)).clamp(-448.0, 448.0).to(dtypes.fp8)
    return q.view(m, n), code.to(torch.uint8)


def make_case(m, n, seed, wide):
    g = torch.Generator(device="cuda").manual_seed(seed * 1009 + m)
    x = torch.randn(m, K, generator=g).to(dtypes.fp8)
    w = (torch.randn(n, K, generator=g) * 0.02).to(dtypes.fp8)
    lo, hi = (122, 134) if wide else (124, 129)
    xs = torch.randint(lo, hi, (m, K // 32), dtype=torch.uint8, generator=g)
    ws = torch.randint(122, 127, (n, K // 32), dtype=torch.uint8, generator=g)
    return (
        x,
        shuffle_weight(w, layout=(16, 16)),
        shuffle_scale(xs.view(dtypes.fp8_e8m0)).view(-1),
        shuffle_scale(ws.view(dtypes.fp8_e8m0)).view(-1),
    )


def scale_rows_mask(m, inter):
    """Where rows < m land in the 1-D w4 scale layout of [pad256(m), inter / 32]."""
    rows = torch.arange(m, dtype=torch.int32).view(m, 1).expand(m, inter // 32) + 2
    return shuffle_scale(rows.contiguous()).view(-1) >= 2  # the pad reads 1


@benchmark()
def test_swiglu_w4(m, n, seed=0, wide=False):
    x, w, xs, ws = make_case(m, n, seed, wide)
    inter = n // 2
    out = torch.empty((m, n), dtype=torch.bfloat16)
    act = swiglu_oai(flydsl_mxfp8_8wave_gemm(x, w, xs, ws, out))
    q_ref, s_ref = mxfp8(act)
    s_ref = shuffle_scale(s_ref.view(dtypes.fp8_e8m0)).view(-1)
    kw = {"alpha": ALPHA, "beta": BETA, "limit": LIMIT}
    y, s = flydsl_mxfp8_8wave_gemm_swiglu_w4(x, w, xs, ws, **kw)
    y2, _ = flydsl_mxfp8_8wave_gemm_swiglu_w4(x, w, xs, ws, **kw)
    rows = scale_rows_mask(m, inter)
    d = {
        "e4m3 diff": int((y.view(torch.uint8) != q_ref.view(torch.uint8)).sum()),
        "e8m0 diff": int(
            (s.view(torch.uint8)[rows] != s_ref.view(torch.uint8)[rows]).sum()
        ),
        "rerun diff": int((y2.view(torch.uint8) != y.view(torch.uint8)).sum()),
    }
    assert all(v == 0 for v in d.values()), f"m={m} n={n} seed={seed} wide={wide}: {d}"
    return {
        **d,
        "gate <= -51.3": int((act.float() == 0).sum()),
        "up clamped": int((out[:, inter:].float().abs() > LIMIT).sum()),
    }


@benchmark()
def perf_swiglu_w4(m, n):
    x, w, xs, ws = make_case(m, n, 0, False)
    out = torch.empty((m, n), dtype=torch.bfloat16)
    _, us_gemm = run_perftest(flydsl_mxfp8_8wave_gemm, x, w, xs, ws, out)
    _, us = run_perftest(
        flydsl_mxfp8_8wave_gemm_swiglu_w4,
        x,
        w,
        xs,
        ws,
        alpha=ALPHA,
        beta=BETA,
        limit=LIMIT,
    )
    return {"bf16 GEMM us": us_gemm, "GEMM + SwiGLU + MXFP8 us": us}


parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--quick", action="store_true", help="fewer cases")
args = parser.parse_args()

if get_gfx_runtime() != "gfx950":
    aiter.logger.info("the 8-wave MXFP8 GEMM targets gfx950; skipping")
    sys.exit(0)

rows = []
for n in (6144, 24576):
    for m in [1024, 8192] if args.quick else [1024, 2048, 5000, 8191, 8192, 16384]:
        rows.append(test_swiglu_w4(m, n))
    rows.append(test_swiglu_w4(8192, n, seed=1, wide=True))
    rows.append(test_swiglu_w4(3001, n, seed=2, wide=True))
aiter.logger.info(f"SwiGLU + MXFP8 epilogue vs GEMM + torch:\n{pd.DataFrame(rows)}")
perf = [perf_swiglu_w4(m, n) for n in (6144, 24576) for m in (8192, 16384)]
aiter.logger.info(f"timing:\n{pd.DataFrame(perf).to_markdown(index=False)}")
