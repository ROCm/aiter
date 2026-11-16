# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Correctness + perf sweep for `aiter.ops.fp8_mqa_topk.fp8_mqa_topk`, the fused prefill indexer
logits + per-row top-K (gfx950, nh=32 / hd=128).

Candidates, each picking the top-k of every row's window [ks, ke):

  fused        aiter.ops.fp8_mqa_topk.fp8_mqa_topk
  hip+topk     the HIP fp8_mqa_logits (clean_logits=False) + aiter.top_k_per_row_prefill
  triton+topk  the Triton fp8_mqa_logits (clean_logits=False) + aiter.top_k_per_row_prefill

All are graded against the fp32 torch reference of test_fp8_mqa_logits.py: the reference logits
at the picked indices, sorted, against the reference top-k values. The fused logits are
bit-identical to the HIP fp8_mqa_logits', so the fused pick (like hip+topk) is also asserted to
be an exact top-k of that kernel's output, and its short rows to read 0..len-1 then -1.
"""

import argparse
import itertools
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import torch

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.fp8_mqa_logits import fp8_mqa_logits as hip_logits
from aiter.ops.fp8_mqa_topk import fp8_mqa_topk
from aiter.ops.triton.attention.fp8_mqa_logits import fp8_mqa_logits as triton_logits
from aiter.test_common import benchmark, checkAllclose, run_perftest
from op_tests.test_fp8_mqa_logits import _num_requests, make_windows, run_torch
from op_tests.triton_tests.attention.test_fp8_mqa_logits import (
    e4m3_type,
    per_custom_dims_cast_to_fp8,
)

torch.set_default_device("cuda")

SUPPORTED_GFX = ["gfx950"]
NUM_HEADS, HEAD_DIM = 32, 128
WINDOWS = ["kpool", "causal", "misaligned", "empty", "past_end", "multi_req"]


def make_topk_windows(s_q, s_k, window):
    """The [ks, ke) row windows: test_fp8_mqa_logits' modes plus `kpool`, the GLM-5.x
    k-pool indexer's prefill chunk -- one causal request whose keys each pool 4 tokens, so
    row t (token s_k*4 - s_q + t) sees keys [0, (token + 1) // 4)."""
    if window != "kpool":
        return make_windows(s_q, s_k, window)
    tok = torch.arange(s_k * 4 - s_q, s_k * 4)
    ks = torch.zeros(s_q, dtype=torch.int32)
    ke = ((tok + 1) // 4).clamp(0, s_k).to(torch.int32)
    return ks, ke


def make_inputs(s_q, s_k, data):
    """randn: test_fp8_mqa_logits' inputs. ties: the keys repeat 61 distinct vectors, so every
    row holds large exact ties at its k-th value. flat: one key vector, every logit of a row
    equal, so every key is a candidate (the select's multi-pass radix over global memory).
    fallback: randn inputs run with force_fallback -- every row of more than k keys takes the
    exact fallback (the first 64 recomputed by the whole grid, the rest one wave per row).
    """
    torch.manual_seed(0)
    q = torch.randn(s_q, NUM_HEADS, HEAD_DIM, dtype=dtypes.bf16)
    distinct = {"randn": s_k, "ties": 61, "flat": 1, "fallback": s_k}[data]
    kv = torch.randn(distinct, HEAD_DIM, dtype=dtypes.bf16)[
        torch.arange(s_k) % distinct
    ]
    kv_fp8, scales = per_custom_dims_cast_to_fp8(kv, (0,), False)
    weights = torch.randn(s_q, NUM_HEADS, dtype=dtypes.fp32)
    return q.to(e4m3_type), kv_fp8, scales, weights


def check_rows(idx, ks, ke, s_k, k, name):
    """Structural contract (asserted): short rows read 0..len-1 then -1; long rows hold k
    distinct indices inside the window. Indices are relative to max(ks, 0)."""
    lens = (ke.clamp(max=s_k) - ks.clamp(min=0)).clamp(min=0).long()
    short = lens <= k
    ar = torch.arange(k)
    want = torch.where(ar[None] < lens[:, None], ar[None], -1)
    bad_short = (short & (idx.long() != want).any(1)).sum()
    srt = idx.long().sort(1).values
    long_ok = (srt[:, 0] >= 0) & (srt[:, -1] < lens) & (srt.diff(dim=1) > 0).all(1)
    bad_long = (~short & ~long_ok).sum()
    assert (
        int(bad_short) == 0 and int(bad_long) == 0
    ), f"{name}: {int(bad_short)} short / {int(bad_long)} long rows break the output contract"
    return short


def picked_values(logits, idx, ks, short):
    """Each long row's logits at the picked indices, sorted descending (short rows zeroed)."""
    s_k = logits.shape[1]
    cols = (idx.long() + ks.clamp(min=0).long()[:, None]).clamp(0, s_k - 1)
    vals = logits.gather(1, cols).sort(1, descending=True).values
    return vals.masked_fill(short[:, None], 0)


def topk_values(logits, k, short):
    """Each long row's reference top-k values, descending (short rows zeroed)."""
    kk = min(k, logits.shape[1])
    vals = torch.nn.functional.pad(logits.topk(kk, dim=1).values, (0, k - kk))
    return vals.masked_fill(short[:, None], 0)


@benchmark()
def test_fp8_mqa_topk(s_q, s_k, k, window, data):
    q_fp8, kv_fp8, scales, weights = make_inputs(s_q, s_k, data)
    ks, ke = make_topk_windows(s_q, s_k, window)
    ref, cost = run_torch(q_fp8, kv_fp8, scales, weights, ks, ke)
    hip_ref = hip_logits(q_fp8, kv_fp8, scales, weights, ks, ke, clean_logits=True)

    # the output buffer the indexer passes in; the graded calls poison it first
    idx = torch.empty((s_q, k), dtype=dtypes.i32)
    nfb = torch.zeros(1, dtype=dtypes.i32)

    def _topk(lg, out):
        aiter.top_k_per_row_prefill(
            lg, ks, ke, out, None, s_q, lg.stride(0), lg.stride(1), k
        )
        return out

    # (timed call, graded call) per candidate
    force = data == "fallback"
    candidates = {
        "fused": (
            lambda: fp8_mqa_topk(
                q_fp8, kv_fp8, scales, weights, ks, ke, idx, force_fallback=force
            ),
            lambda: fp8_mqa_topk(
                q_fp8,
                kv_fp8,
                scales,
                weights,
                ks,
                ke,
                idx.fill_(-7),
                fallback_rows=nfb,
                force_fallback=force,
            ),
        ),
    }
    # top_k_per_row_prefill does not clamp a window to s_k (a past-the-end row comes back as
    # columns past the logits row), so the unfused baselines run on in-range windows only.
    if window != "past_end":
        candidates["hip+topk"] = (
            lambda: _topk(
                hip_logits(q_fp8, kv_fp8, scales, weights, ks, ke, clean_logits=False),
                idx,
            ),
            lambda: _topk(
                hip_logits(q_fp8, kv_fp8, scales, weights, ks, ke, clean_logits=False),
                idx.fill_(-7),
            ),
        )
        candidates["triton+topk"] = (
            lambda: _topk(
                triton_logits(q_fp8, kv_fp8, scales, weights, ks, ke, False), idx
            ),
            lambda: _topk(
                triton_logits(q_fp8, kv_fp8, scales, weights, ks, ke, False),
                idx.fill_(-7),
            ),
        )
    flops = cost.item() * NUM_HEADS * HEAD_DIM * 2
    nbytes = (
        (s_q * NUM_HEADS + s_k) * HEAD_DIM * q_fp8.element_size()  # q, k
        + (s_k + s_q * NUM_HEADS) * 4  # scales, weights
        + 2 * s_q * 4  # ks, ke
        + s_q * k * 4  # top-k indices
    )

    ret = {"gfx": get_gfx()}
    for name, (time_fn, grade_fn) in candidates.items():
        _, us = run_perftest(time_fn)
        out = grade_fn().clone()
        if name != "fused":  # top_k_per_row_prefill writes absolute columns
            out = torch.where(out >= 0, out - ks.clamp(min=0)[:, None], out)
        short = check_rows(out, ks, ke, s_k, k, name)
        if name == "fused" and force:
            n_long = int((~short).sum())
            assert (
                int(nfb.item()) == n_long
            ), f"forced fallback ran on {int(nfb.item())} rows, expected {n_long}"
        if name != "triton+topk":
            # an exact top-k of the HIP logits (the fused scan's are bit-identical to them)
            got = picked_values(hip_ref, out, ks, short)
            assert torch.equal(
                got, topk_values(hip_ref, k, short)
            ), f"{name}: not a top-{k} of fp8_mqa_logits' output"
        # long rows have more than k finite reference logits, so the -inf outside the
        # windows never reaches a compared top-k value
        err = checkAllclose(
            topk_values(ref, k, short),
            picked_values(ref, out, ks, short),
            rtol=1e-2,
            atol=1e-2,
            msg=f"{name}: fp8_mqa_topk",
            printLog=False,
        )
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6 if flops > 0 else float("nan")
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = err
    ret["fused fallback rows"] = int(nfb.item())
    return ret


def main():
    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning("fp8_mqa_topk unsupported on %s; skipping", get_gfx())
        return

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="config input of test",
    )
    parser.add_argument(
        "-s",
        "--shapes",
        type=dtypes.str2tuple,
        nargs="*",
        default=[
            (31, 1000),
            (100, 4096),
            (300, 4160),
            (2048, 8192),
            (4096, 32768),
            (512, 40000),
            (1024, 131072),
            # GLM-5.x prefill chunks (16384 tokens): the 2nd and the 8th of a 128K prompt
            (16384, 8192),
            (16384, 32768),
        ],
        help="(s_q, s_k) pairs",
    )
    parser.add_argument(
        "-k", type=int, nargs="*", default=[64, 512], help="top-k (<= 512)"
    )
    parser.add_argument(
        "-w", "--window", type=str, nargs="*", default=WINDOWS, choices=WINDOWS
    )
    parser.add_argument(
        "--data",
        type=str,
        nargs="*",
        default=["randn", "ties", "flat", "fallback"],
        choices=["randn", "ties", "flat", "fallback"],
    )
    args = parser.parse_args()

    df = []
    for (s_q, s_k), k, win, data in itertools.product(
        args.shapes, args.k, args.window, args.data
    ):
        # multi_req degenerates to causal below two requests
        if win == "multi_req" and _num_requests(s_q, s_k) < 2:
            continue
        # flat (every key a candidate) and fallback (every long row recomputed, one wave per
        # row past the first 64) are stress cases: chunks of at most 2^26 logits keep the
        # sweep in minutes
        if data in ("flat", "fallback") and s_q * s_k > 1 << 26:
            continue
        df.append(test_fp8_mqa_topk(s_q, s_k, k, win, data))
    df = pd.DataFrame(df)
    aiter.logger.info(
        "fp8_mqa_topk summary (markdown):\n%s", df.to_markdown(index=False)
    )


if __name__ == "__main__":
    main()
