# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Tests of ``minimax_m3_index_topk_prefill_flydsl`` (gfx950): the 16 block ids of ragged fp32 score rows against a
torch reference of the contract (order-preserving fp32 key, NaN as -inf, ties to the larger block, init / local
forced to the 1e30 / 1e29 sentinel keys), bit for bit, then a perf sweep at prefill shapes against
``topk_per_row_small_k`` (which serves the same rows but breaks score ties the other way, so only its time is
compared on tie-free rows).

    python op_tests/test_flydsl_minimax_m3_index_topk_prefill.py [-r ROWS ...] [-p PREFIX ...]
"""

import argparse
import itertools
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import torch

import aiter
from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops.flydsl.kernels.minimax_m3_index_topk_prefill import (
    minimax_m3_index_topk_prefill_flydsl,
)
from aiter.test_common import benchmark, checkAllclose, run_perftest

NEG_INF = float("-inf")


def reference(score, nvalid, init, local):
    """Top-16 ids of every (head, row): the contract, in torch (int64 keys, exact)."""
    heads, rows, width = score.shape
    s = score.clone()
    s[torch.isnan(s)] = NEG_INF
    bits = s.view(torch.int32).to(torch.int64)
    key = bits ^ ((bits >> 31) & 0x7FFFFFFF)
    blk = torch.arange(width, device=score.device)[None, None, :]
    nv = nvalid.view(heads, rows)[:, :, None].to(torch.int64)
    key = torch.where(blk < init, torch.full_like(key, 0x7149F2CA), key)
    key = torch.where(blk >= nv - local, torch.full_like(key, 0x6FA18F08), key)
    comp = (key << 32) | blk  # signed compare: key first, then the larger block
    live = blk < torch.minimum(nv, torch.tensor(width, device=score.device))
    comp = torch.where(live, comp, torch.full_like(comp, torch.iinfo(torch.int64).min))
    top = comp.topk(min(16, width), dim=-1).values
    ids = (top & 0xFFFFFFFF).to(torch.int32)
    nlive = torch.minimum(nv, torch.tensor(width, device=score.device))
    pos = torch.arange(top.shape[-1], device=score.device)[None, None, :]
    ids = torch.where(pos < nlive, ids, torch.full_like(ids, -1))
    if ids.shape[-1] < 16:
        ids = torch.cat(
            [
                ids,
                torch.full(
                    (heads, rows, 16 - ids.shape[-1]),
                    -1,
                    dtype=torch.int32,
                    device=score.device,
                ),
            ],
            -1,
        )
    return ids


def make(kind, heads, rows, width, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    if kind == "normal":
        s = torch.randn(heads, rows, width, generator=g, device="cuda") * 3
    elif kind == "ties":  # few distinct values: the tie rule decides
        s = torch.randint(
            -3, 4, (heads, rows, width), generator=g, device="cuda"
        ).float()
    elif kind == "special":
        s = torch.randn(heads, rows, width, generator=g, device="cuda")
        m = torch.rand(heads, rows, width, generator=g, device="cuda")
        s[m < 0.05] = float("nan")
        s[(m >= 0.05) & (m < 0.1)] = NEG_INF
        s[(m >= 0.1) & (m < 0.12)] = float("inf")
        s[(m >= 0.12) & (m < 0.14)] = -0.0
        s[(m >= 0.14) & (m < 0.16)] = 0.0
    elif kind == "neginf_most":
        s = torch.full((heads, rows, width), NEG_INF, device="cuda")
        s[..., :: max(1, width // 7)] = torch.randn(1, device="cuda")
    return s.contiguous()


def run_checks():
    """Every (kind, width, init, local) case below must match the reference exactly; returns the failure count."""
    fails = 0
    cases = []
    for kind in ("normal", "ties", "special", "neginf_most"):
        for width, msl in (
            (9, 1152),
            (40, 5120),
            (300, 38400),
            (1000, 128000),
            (1700, 217600),
            (3000, 384000),
        ):
            for init, local in ((0, 1), (0, 0), (2, 3)):
                cases.append((kind, 1, 37, width, msl, init, local, True))
        cases.append((kind, 2, 19, 700, 90000, 0, 1, False))
    for kind, heads, rows, width, msl, init, local, ragged in cases:
        s = make(kind, heads, rows, width, 0)
        g = torch.Generator(device="cuda").manual_seed(1)
        if ragged:
            # 0 live blocks (a pad row) up to past the readable width (clamped to it)
            nv = torch.randint(
                0,
                width + 4,
                (heads * rows,),
                generator=g,
                device="cuda",
                dtype=torch.int32,
            )
        else:
            nv = torch.full((heads * rows,), width, device="cuda", dtype=torch.int32)
        out = torch.full((heads, rows, 16), -7, dtype=torch.int32, device="cuda")
        minimax_m3_index_topk_prefill_flydsl(s, nv, out, msl, init, local)
        err = checkAllclose(
            reference(s, nv, init, local).float(),
            out.float(),
            rtol=0,
            atol=0,
            msg=f"{kind} h{heads} rows{rows} w{width} init{init} local{local} ragged{ragged}: ",
            printLog=False,
        )
        if err:
            print(
                f"  FAIL {kind} h{heads} rows{rows} w{width} init{init} local{local} ragged{ragged}"
            )
        fails += err != 0
    aiter.logger.info(f"{len(cases) - fails}/{len(cases)} reference cases PASS")
    return fails


@benchmark()
def test_prefill_topk(rows, prefix, kind="normal", init=0, local=1):
    """One request of ``rows`` new tokens after ``prefix`` cached tokens: row q sees ceil((prefix + q + 1) / 128)
    blocks, as the prefill selector does (head 1, 128-token blocks)."""
    width = (prefix + rows + 127) // 128
    s = make(kind, 1, rows, width, 3)
    nv = (
        ((prefix + torch.arange(rows, device="cuda") + 128) // 128)
        .clamp(max=width)
        .int()
        .contiguous()
    )
    msl = prefix + rows
    want = reference(s, nv, init, local)
    out = torch.empty((1, rows, 16), dtype=torch.int32, device="cuda")
    _, us = run_perftest(
        minimax_m3_index_topk_prefill_flydsl,
        s,
        nv,
        out,
        msl,
        init,
        local,
        num_iters=20,
        num_rotate_args=3,
    )
    ret = {"width": width, "us": us}
    ret["err"] = checkAllclose(
        want.float(),
        out.float(),
        rtol=0,
        atol=0,
        msg=f"rows{rows} prefix{prefix} {kind}: ",
    )
    try:
        from aiter.ops.flydsl.topk.topk_per_row_small_k import (
            topk_per_row_small_k,
            topk_per_row_small_k_supported,
        )

        flat, out2 = s.view(rows, width), torch.empty(
            (rows, 16), dtype=torch.int32, device="cuda"
        )
        if topk_per_row_small_k_supported(flat, nv, out2, 16):
            _, ret["small_k us"] = run_perftest(
                topk_per_row_small_k,
                flat,
                nv,
                out2,
                16,
                init,
                local,
                num_iters=20,
                num_rotate_args=3,
            )
    except ImportError:
        pass
    return ret


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("-r", "--rows", type=int, nargs="*", default=[2048, 8192, 16384])
    ap.add_argument(
        "-p", "--prefix", type=int, nargs="*", default=[32768, 102400, 163840]
    )
    args = ap.parse_args(argv)
    if get_gfx_runtime() != "gfx950":
        print("gfx950 only; skipped")
        return 0
    fails = run_checks()
    rows = []
    for r, p in itertools.product(args.rows, args.prefix):
        row = test_prefill_topk(r, p)
        fails += row["err"] != 0
        rows.append(row)
    aiter.logger.info(
        "minimax_m3_index_topk_prefill (us per launch):\n%s",
        pd.DataFrame(rows).to_markdown(index=False),
    )
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
