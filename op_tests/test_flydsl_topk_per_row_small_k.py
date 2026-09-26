# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Per-row TopK for a small k: correctness against torch, and the perf sweep.

Shapes come from a block-sparse indexer, which is what this selector is for: the
row is one score per KV block, so `width = ceil(context / block_size)` and `k` is
a block count, not a token count. The MiniMax-M3 sparse config (4 index heads,
`sparse_topk_blocks` 16, `sparse_block_size` 128, `sparse_local_block` 1) puts a
1M-token context at width 8192, k 16, and one row per (index head, query token).

The distributions are not decoration. `ascending` is the adversarial case for any
threshold-narrowing selector and `clustered` is what an fp8 index cache actually
produces -- both exercise the tie rule, which is load-bearing: the selection
order is the order attention accumulates in.
"""

import argparse
import itertools

import pandas as pd
import torch

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.topk.topk_per_row_small_k import (
    topk_per_row_small_k,
    topk_per_row_small_k_supported,
)
from aiter.test_common import benchmark, checkAllclose, run_perftest

torch.set_default_device("cuda")

SUPPORTED_GFX = ["gfx942", "gfx950"]
_PIN_INIT = 1e30
_PIN_LOCAL = 1e29


def make_scores(rows, width, dist, dtype=dtypes.fp32, seed=0):
    """One score matrix per distribution, in the layout the indexer emits."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    if dist == "gauss":
        return torch.randn(rows, width, dtype=dtype, generator=gen)
    if dist == "recency":
        # Indexer scores drift up with position; ~2 sigma across the row.
        ramp = torch.linspace(0, 2, width, dtype=dtype).expand(rows, width)
        return torch.randn(rows, width, dtype=dtype, generator=gen) + ramp
    if dist == "clustered":
        # A block score is a max over 128 keys of a low-mantissa cache, so exact
        # ties are common.
        x = torch.randn(rows, width, dtype=dtype, generator=gen)
        return (x * 8).round() / 8
    if dist == "ascending":
        # Worst case for the narrowing: every chunk maximum sits at its end.
        base = torch.arange(width, dtype=dtype) / width
        return base.expand(rows, width).contiguous()
    raise ValueError(f"unknown distribution {dist}")


def pinned_scores(scores, row_lens, init_blocks, local_blocks):
    """Scores with the forced blocks pinned and dead columns removed.

    Both the reference and the tie checks need this, and it is the one place the
    kernel's ordering contract is written down: local outranks init, and a column
    at or past its row's length is invisible.
    """
    rows, width = scores.shape
    col = torch.arange(width).expand(rows, width)
    live = col < row_lens[:, None]
    local_start = torch.clamp(row_lens[:, None] - local_blocks, min=0)
    out = torch.where(
        live & (col < init_blocks), scores.new_full((), _PIN_INIT), scores
    )
    out = torch.where(live & (col >= local_start), out.new_full((), _PIN_LOCAL), out)
    # NaN loses every selection it enters; -inf then keeps dead columns below it.
    out = torch.where(out.isnan(), out.new_full((), float("-inf")), out)
    return torch.where(live, out, out.new_full((), float("-inf"))), live


def run_torch(scores, row_lens, k, init_blocks, local_blocks):
    """Reference selection: column ids, descending by score, ties to the larger
    column, -1 padded. Not timed, not in the table."""
    pinned, _ = pinned_scores(scores, row_lens, init_blocks, local_blocks)
    rows, width = scores.shape
    col = torch.arange(width).expand(rows, width)
    # Visit columns descending, then stable-sort by score descending: equal
    # scores keep the visit order, which is the tie rule. Leaving the tie to
    # torch.topk instead would compare against an unspecified order.
    by_col = torch.argsort(col, dim=1, descending=True)
    scores_by_col = torch.gather(pinned, 1, by_col)
    order = torch.argsort(scores_by_col, dim=1, descending=True, stable=True)
    picks = torch.gather(by_col, 1, order)[:, :k].to(torch.int32)
    real = torch.clamp(row_lens, max=k)
    slot = torch.arange(k).expand(rows, k)
    return torch.where(slot < real[:, None], picks, torch.full_like(picks, -1))


def selected_scores(scores, row_lens, picks, init_blocks, local_blocks):
    """Scores behind a selection, for a tie-robust comparison.

    Comparing index vectors fails on ties for reasons that are not bugs; the
    scores a selection pulls out are what the consumer sees.
    """
    pinned, _ = pinned_scores(scores, row_lens, init_blocks, local_blocks)
    safe = torch.clamp(picks.long(), min=0)
    got = torch.gather(pinned, 1, safe)
    return torch.where(picks >= 0, got, got.new_full((), float("-inf")))


def torch_topk_indices(scores, row_lens, k, init_blocks, local_blocks):
    """The naive whole-row candidate: pin, mask, topk."""
    pinned, _ = pinned_scores(scores, row_lens, init_blocks, local_blocks)
    picks = torch.topk(pinned, k, dim=1).indices.to(torch.int32)
    real = torch.clamp(row_lens, max=k)
    slot = torch.arange(k).expand(picks.shape)
    return torch.where(slot < real[:, None], picks, torch.full_like(picks, -1))


@benchmark()
def test_topk_per_row_small_k(rows, width, k, dist, init_blocks, local_blocks):
    scores = make_scores(rows, width, dist)
    row_lens = torch.full((rows,), width, dtype=dtypes.i32)
    ref = run_torch(scores, row_lens, k, init_blocks, local_blocks)
    ref_scores = selected_scores(scores, row_lens, ref, init_blocks, local_blocks)

    out = torch.empty(rows, k, dtype=dtypes.i32)
    assert topk_per_row_small_k_supported(scores, row_lens, out, k)

    candidates = {
        "flydsl": lambda: (
            topk_per_row_small_k(scores, row_lens, out, k, init_blocks, local_blocks),
            out,
        )[1],
        "torch_topk": lambda: torch_topk_indices(
            scores, row_lens, k, init_blocks, local_blocks
        ),
    }

    # No floating-point math happens here, so the "flop" counted is the one
    # comparison per element the narrowing pass does; TB/s is the metric that
    # means something for a selector.
    flops = rows * width
    nbytes = rows * width * scores.element_size() + rows * k * 4

    ret = {"gfx": get_gfx()}
    for name, fn in candidates.items():
        got, us = run_perftest(fn)
        got_scores = selected_scores(scores, row_lens, got, init_blocks, local_blocks)
        err = checkAllclose(
            ref_scores.to(dtypes.fp32),
            got_scores.to(dtypes.fp32),
            rtol=0,
            atol=0,
            msg=f"{name}: selected scores",
        )
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = err
    return ret


@benchmark()
def test_topk_per_row_small_k_ragged(rows, width, k, init_blocks, local_blocks):
    """Per-row lengths, including rows shorter than k."""
    scores = make_scores(rows, width, "recency", seed=7)
    row_lens = torch.randint(0, width + 1, (rows,), dtype=dtypes.i32)
    row_lens[0] = 0
    row_lens[1 % rows] = min(k - 1, width)
    row_lens[2 % rows] = width
    ref = run_torch(scores, row_lens, k, init_blocks, local_blocks)
    ref_scores = selected_scores(scores, row_lens, ref, init_blocks, local_blocks)

    out = torch.empty(rows, k, dtype=dtypes.i32)
    got, us = run_perftest(
        lambda: (
            topk_per_row_small_k(scores, row_lens, out, k, init_blocks, local_blocks),
            out,
        )[1]
    )
    got_scores = selected_scores(scores, row_lens, got, init_blocks, local_blocks)
    err = checkAllclose(
        ref_scores.to(dtypes.fp32),
        got_scores.to(dtypes.fp32),
        rtol=0,
        atol=0,
        msg="flydsl ragged: selected scores",
    )
    # A row shorter than k must pad with -1 and nothing else.
    pad = torch.arange(k).expand(rows, k) >= torch.clamp(row_lens, max=k)[:, None]
    assert bool((got[pad] == -1).all()), "short rows must be -1 padded"
    assert bool((got[~pad] >= 0).all()), "live slots must hold a real column"

    nbytes = rows * width * scores.element_size() + rows * k * 4
    return {
        "gfx": get_gfx(),
        "flydsl us": us,
        "flydsl TFLOPS": rows * width / us / 1e6,
        "flydsl TB/s": nbytes / us / 1e6,
        "flydsl err": err,
    }


def summarize(name, rows):
    aiter.logger.info(
        "%s (markdown):\n%s", name, pd.DataFrame(rows).to_markdown(index=False)
    )


def main():
    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning(
            "topk_per_row_small_k unsupported on %s; skipping", get_gfx()
        )
        return

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="config input of test",
    )
    parser.add_argument(
        "-r",
        "--rows",
        type=int,
        nargs="*",
        default=[16, 128, 512, 2048],
        help="rows = index heads x query tokens",
    )
    parser.add_argument(
        "-w",
        "--width",
        type=int,
        nargs="*",
        default=[256, 1024, 8192],
        help="row width = ceil(context / sparse block size)",
    )
    parser.add_argument("-k", "--topk", type=int, nargs="*", default=[16])
    parser.add_argument(
        "--dist",
        type=str,
        nargs="*",
        default=["gauss", "recency", "clustered", "ascending"],
    )
    parser.add_argument("--init-blocks", type=int, nargs="*", default=[0])
    parser.add_argument("--local-blocks", type=int, nargs="*", default=[1])
    args = parser.parse_args()

    summarize(
        "topk_per_row_small_k",
        [
            test_topk_per_row_small_k(rows, width, k, dist, init_b, local_b)
            for rows, width, k, dist, init_b, local_b in itertools.product(
                args.rows,
                args.width,
                args.topk,
                args.dist,
                args.init_blocks,
                args.local_blocks,
            )
        ],
    )
    summarize(
        "topk_per_row_small_k ragged",
        [
            test_topk_per_row_small_k_ragged(rows, width, k, init_b, local_b)
            for rows, width, k, init_b, local_b in itertools.product(
                args.rows,
                args.width,
                args.topk,
                args.init_blocks,
                args.local_blocks,
            )
        ],
    )


if __name__ == "__main__":
    main()
