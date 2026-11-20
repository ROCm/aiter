# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness and performance sweep of ``flydsl_router_sqrtsoftplus_topk``.

The DeepSeek-V4.1-Flash MoE router for decode rows:

    logits = x [M, 5120] @ gate [384, 5120]^T      (bf16, fp32 accumulate)
    score  = sqrt(softplus(logits))                (logits above 20 pass through)
    ids    = top-6 of score + bias                 (bias fp32 [384])
    w      = score[ids] * 1.5 / sum(score[ids])

Candidates: the FlyDSL op with the row-major gate and with the load-time
shuffled gate (``shuffle_router_gate``), and AITER's ``topk_gating`` on a
``torch.mm`` (hipBLASLt) fp32 GEMM, each checked against an fp32 torch
reference (rows whose 6th and 7th candidates tie within fp32 noise excluded).
Edge cases (ties, padding, strides, graph replay) are in
``op_tests/flydsl_tests/test_flydsl_router_sqrtsoftplus_topk.py``.

Run:
    python3 op_tests/test_flydsl_router_sqrtsoftplus_topk.py
    python3 op_tests/test_flydsl_router_sqrtsoftplus_topk.py -m 16 192 768
"""

import argparse

import pandas as pd
import torch

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.test_common import benchmark, checkAllclose, run_perftest

SUPPORTED_GFX = ["gfx950"]
SCALE = 1.5
HIDDEN, EXPERTS = 5120, 384


def run_torch_router(logits, bias):
    score = torch.where(logits > 20.0, logits, torch.log1p(torch.exp(logits))).sqrt()
    top = torch.topk(score + bias, 7, dim=-1)
    ids = top.indices[:, :6]
    w = score.gather(1, ids)
    w = w * (SCALE / w.sum(-1, keepdim=True))
    # A row whose 6th and 7th candidates tie within fp32 noise may pick either.
    gap = top.values[:, 5] - top.values[:, 6]
    return w, ids.int(), gap > 1e-5 * top.values[:, 5].abs().clamp_min(1.0)


@benchmark()
def test_router_sqrtsoftplus_topk(m):
    from aiter.ops.flydsl import flydsl_router_sqrtsoftplus_topk, shuffle_router_gate
    from aiter.ops.topk import topk_gating

    torch.manual_seed(m)
    x = torch.randn(m, HIDDEN).to(dtypes.bf16)
    w = (torch.randn(EXPERTS, HIDDEN) * 0.02).to(dtypes.bf16)
    ws = shuffle_router_gate(w)
    bias = torch.randn(EXPERTS) * 0.1
    ref_w, ref_ids, clear = run_torch_router(torch.mm(x.float(), w.float().t()), bias)

    def mm_topk_gating():
        tw = torch.empty(m, 6)
        ti = torch.empty(m, 6, dtype=torch.int32)
        logits = torch.mm(x, w.t(), out_dtype=torch.float32)
        topk_gating(tw, ti, logits, bias, True, SCALE, "sqrtsoftplus")
        return tw, ti

    candidates = {
        "flydsl": lambda: flydsl_router_sqrtsoftplus_topk(x, w, bias, SCALE),
        "flydsl_shuffled": lambda: flydsl_router_sqrtsoftplus_topk(
            x, ws, bias, SCALE, weight_shuffled=True
        ),
        "mm+topk_gating": mm_topk_gating,
    }
    flops = 2 * m * HIDDEN * EXPERTS
    nbytes = (m + EXPERTS) * HIDDEN * 2 + m * 6 * 8
    ret = {"gfx": get_gfx_runtime()}
    for name, fn in candidates.items():
        (tw, ti), us = run_perftest(fn)
        err_ids = checkAllclose(
            ref_ids[clear].float(), ti[clear].float(), rtol=0, atol=0,
            tol_err_ratio=0, msg=f"{name}: ids",
        )  # fmt: skip
        err_w = checkAllclose(
            ref_w[clear], tw[clear], rtol=1e-4, atol=1e-6, tol_err_ratio=0,
            msg=f"{name}: weights",
        )  # fmt: skip
        assert max(err_ids, err_w) == 0, f"{name}: mismatch"
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = max(err_ids, err_w)
    return ret


def main():
    if get_gfx_runtime() not in SUPPORTED_GFX:
        aiter.logger.warning(
            "FlyDSL router unsupported on %s; skipping", get_gfx_runtime()
        )
        return
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="FlyDSL MoE router: bf16 GEMM + sqrt-softplus top-6",
    )
    parser.add_argument(
        "-m",
        type=int,
        nargs="*",
        default=[1, 16, 64, 128, 192, 256, 384, 512, 768, 1024],
        help="rows (decode tokens), e.g. -m 16 192",
    )
    args = parser.parse_args()
    torch.set_default_device("cuda")
    df = pd.DataFrame([test_router_sqrtsoftplus_topk(m) for m in args.m])
    aiter.logger.info(
        "router sqrt-softplus top-6 summary (markdown):\n%s",
        df.to_markdown(index=False),
    )


if __name__ == "__main__":
    main()
