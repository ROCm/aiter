# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Per-row TopK for a small k: correctness against torch, and the perf sweep.

k=16 rows of up to 8192 columns on gfx950 are served by a one-wave-per-row build whose ids are checked exactly
against an int64-key torch reference of the tie rule (order-preserving fp32 key, NaN as -inf, ties to the larger
column, init / local forced); the chunked build behind every other geometry is checked on the scores it selects,
since it may keep a different tied column at the k-th place.

Shapes come from a block-sparse indexer, which is what this selector is for: the
row is one score per KV block, so `width = ceil(context / block_size)` and `k` is
a block count, not a token count. A sparse config of 4 index heads, 16 selected
blocks, 128-token blocks and 1 local block puts a 1M-token context at width 8192, k 16, and one row per (index head, query token).

The distributions are not decoration. `ascending` is the adversarial case for any
threshold-narrowing selector and `clustered` is what an fp8 index cache actually
produces -- both exercise the tie rule, which is load-bearing: the selection
order is the order attention accumulates in.
"""

import argparse
import itertools
import sys

import pandas as pd
import torch

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx, get_gfx_runtime
from aiter.ops.flydsl.kernels.topk.topk_launch import run_topk_launcher
from aiter.ops.flydsl.kernels.topk.topk_per_row_small_k import (
    build_topk_per_row_small_k_module,
)
from aiter.ops.flydsl.kernels.topk.topk_per_row_small_k_wave import (
    MAX_WIDTH as WAVE_MAX_WIDTH,
)
from aiter.ops.flydsl.kernels.topk.topk_per_row_small_k_wave import (
    build_topk_per_row_small_k_wave_module,
    wave_strips,
)
from aiter.ops.flydsl.topk.row_lens import row_len_slot
from aiter.ops.flydsl.topk.topk_per_row_small_k import (
    _row_bucket,
    _wave_tile,
    _wave_width_limit,
    topk_per_row_small_k,
    topk_per_row_small_k_supported,
)
from aiter.test_common import benchmark, checkAllclose, run_perftest

torch.set_default_device("cuda")

SUPPORTED_GFX = ["gfx942", "gfx950", "gfx1250"]
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


def exact_ids(scores, row_lens, k, init_blocks, local_blocks):
    """The contract in torch, exact: int64 keys of the order-preserving fp32 bits (NaN as -inf), the forced blocks
    at the bits of 1e30 / 1e29, then the larger column first. Row lengths past the width are clamped to it.
    """
    rows, width = scores.shape
    s = torch.where(scores.isnan(), scores.new_full((), float("-inf")), scores)
    bits = s.view(torch.int32).to(torch.int64)
    key = bits ^ ((bits >> 31) & 0x7FFFFFFF)
    col = torch.arange(width)[None, :]
    n = torch.clamp(row_lens.to(torch.int64), 0, width)[:, None]
    key = torch.where(col < init_blocks, torch.full_like(key, 0x7149F2CA), key)
    key = torch.where(col >= n - local_blocks, torch.full_like(key, 0x6FA18F08), key)
    comp = (key << 32) | col
    comp = torch.where(
        col < n, comp, torch.full_like(comp, torch.iinfo(torch.int64).min)
    )
    kk = min(k, width)
    ids = (comp.topk(kk, dim=-1).values & 0xFFFFFFFF).to(torch.int32)
    ids = torch.where(torch.arange(kk)[None, :] < n, ids, torch.full_like(ids, -1))
    return torch.cat([ids, ids.new_full((rows, k - kk), -1)], dim=1)


def make_kind(kind, rows, width, seed):
    """Score kinds of the exact-id cases: tie-free, heavy ties, NaN / inf / signed-zero mixes, almost all -inf."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    if kind == "normal":
        s = torch.randn(rows, width, generator=g) * 3
    elif kind == "ties":  # few distinct values: the tie rule decides
        s = torch.randint(-3, 4, (rows, width), generator=g).float()
    elif kind == "special":
        s = torch.randn(rows, width, generator=g)
        m = torch.rand(rows, width, generator=g)
        s[m < 0.05] = float("nan")
        s[(m >= 0.05) & (m < 0.1)] = float("-inf")
        s[(m >= 0.1) & (m < 0.12)] = float("inf")
        s[(m >= 0.12) & (m < 0.14)] = -0.0
        s[(m >= 0.14) & (m < 0.16)] = 0.0
    elif kind == "neginf_most":
        s = torch.full((rows, width), float("-inf"))
        s[:, :: max(1, width // 7)] = torch.randn(1)
    else:
        raise ValueError(kind)
    return s.contiguous()


def launch_wave(scores, row_lens, out, init_blocks, local_blocks):
    """The one-wave build directly, for rows narrower than k that the op itself declines."""
    rows, width = scores.shape
    run_topk_launcher(
        build_topk_per_row_small_k_wave_module(wave_strips(width), row_lens is None),
        scores.data_ptr(),
        scores.stride(0),
        width,
        row_len_slot(row_lens, scores.device).data_ptr(),
        init_blocks,
        local_blocks,
        out.data_ptr(),
        out.stride(0),
        rows,
        torch.cuda.current_stream(),
    )


def launch_chunked(scores, row_lens, out, k, init_blocks, local_blocks):
    """The chunked build directly, to time it against the op where the op takes the one-wave build."""
    rows, width = scores.shape
    run_topk_launcher(
        build_topk_per_row_small_k_module(
            k,
            _row_bucket(width),
            64,
            forced_blocks=bool(init_blocks or local_blocks),
            fixed_row_len=width if row_lens is None else 0,
        ),
        scores,
        row_len_slot(row_lens, scores.device),
        out,
        width,
        init_blocks,
        local_blocks,
        rows,
        torch.cuda.current_stream(),
    )


@benchmark()
def test_topk_per_row_small_k_exact(
    kind, rows, width, k, init_blocks, local_blocks, ragged
):
    """Exact ids against the contract. One case of the grid in `exact_cases`: the one-wave kernel is checked
    directly at every width it builds for, and the op through its dispatch -- exact ids where the op takes the
    one-wave build, the selected scores where it takes the chunked one (it may keep another tied column).
    """
    scores = make_kind(kind, rows, width, 0)
    g = torch.Generator(device="cuda").manual_seed(1)
    if ragged:
        # 0 live columns (a pad row) up to past the width (clamped to it)
        row_lens = torch.randint(0, width + 4, (rows,), generator=g, dtype=dtypes.i32)
    else:
        row_lens = torch.full((rows,), width, dtype=dtypes.i32)
    ref = exact_ids(scores, row_lens, k, init_blocks, local_blocks)
    live_lens = row_lens.clamp(max=width)
    out = torch.full((rows, k), -7, dtype=dtypes.i32)
    gfx950 = get_gfx_runtime() == "gfx950"
    wave = gfx950 and bool(_wave_tile(k, width, 64, _wave_width_limit(rows, False)))
    ret = {"gfx": get_gfx(), "one-wave": wave, "err": 0}

    def check(got, exact, msg):
        if exact:
            err = checkAllclose(
                ref.float(), got.float(), rtol=0, atol=0, msg=msg, printLog=False
            )
            if err:
                bad = (got != ref).any(dim=1)
                aiter.logger.warning(
                    "%s%d of %d rows differ (kind=%s width=%d k=%d init=%d local=%d)",
                    msg,
                    int(bad.sum()),
                    rows,
                    kind,
                    width,
                    k,
                    init_blocks,
                    local_blocks,
                )
            return err
        return checkAllclose(
            selected_scores(scores, live_lens, ref, init_blocks, local_blocks),
            selected_scores(scores, live_lens, got, init_blocks, local_blocks),
            rtol=0,
            atol=0,
            msg=msg,
            printLog=False,
        )

    if gfx950 and k == 16 and width <= WAVE_MAX_WIDTH:
        # the kernel itself, at widths the dispatch may leave to the chunked build (never past its 13-bit
        # column limit, which the op's own dispatch enforces). Every fourth row gets a negative length: a pad
        # row with no live column, whose ids are all -1 (the exact reference clamps lengths to 0..width).
        direct = torch.full((rows, k), -7, dtype=dtypes.i32)
        neg = torch.arange(rows) % 4 == 0
        direct_lens = torch.where(neg, -1 - torch.arange(rows), row_lens).to(dtypes.i32)
        launch_wave(scores, direct_lens, direct, init_blocks, local_blocks)
        direct_ref = exact_ids(scores, direct_lens, k, init_blocks, local_blocks)
        ret["err"] += (
            checkAllclose(
                direct_ref.float(),
                direct.float(),
                rtol=0,
                atol=0,
                msg="one-wave kernel exact ids: ",
                printLog=False,
            )
            != 0
        )
    # The chunked build ranks a NaN above +inf (what `topk_select` documents); the one-wave build treats it as -inf
    # (the Triton selector's contract), so NaN rows compare only where the op takes the one-wave build.
    nan_policy_differs = kind == "special" and not wave
    if not nan_policy_differs and topk_per_row_small_k_supported(
        scores, row_lens, out, k
    ):
        got, ret["us"] = run_perftest(
            lambda: (
                topk_per_row_small_k(
                    scores, row_lens, out, k, init_blocks, local_blocks
                ),
                out,
            )[1],
            num_iters=5,
            num_warmup=1,
        )
        ret["err"] += check(got, wave or kind == "normal", "op: ") != 0
    return ret


def test_topk_select_batch_invariant():
    """`topk_select(deterministic=True)` selects a row's columns whatever the batch it sits in: an all-zero
    2048-column row at k=16 has every column tied, and the one-wave build (taken with many rows) and the
    chunked build (taken with one) keep different ones unless `exact` pins the one-wave build for both.
    """
    from aiter.ops.topk_select import topk_select

    err = 0
    for kw in ({"deterministic": True}, {"tie": "high"}):
        alone = topk_select(torch.zeros(1, 2048), 16, sorted_index=True, **kw)[1]
        batched = topk_select(torch.zeros(4096, 2048), 16, sorted_index=True, **kw)[1]
        same = bool((batched == alone).all())
        aiter.logger.info("topk_select %s: alone vs batched rows equal: %s", kw, same)
        err += not same
    return err


def exact_cases():
    """The 76 cases of the one-wave build (4 kinds x 6 widths x 3 forced-block settings + one dense multi-row case
    each), the widest rows (NaN mixes only where the one-wave build serves), then the chunked build's geometries: k=8 and rows wider than 8192.
    """
    cases = []
    for kind in ("normal", "ties", "special", "neginf_most"):
        for width in (9, 40, 300, 1000, 1700, 3000):
            for init, local in ((0, 1), (0, 0), (2, 3)):
                cases.append((kind, 37, width, 16, init, local, True))
        cases.append((kind, 38, 700, 16, 0, 1, False))
    for kind in ("ties", "special"):
        # the widest rows the kernel builds, and the many-row batch the dispatch gives it at 4096 columns
        cases.append((kind, 37, 8192, 16, 0, 1, True))
        cases.append((kind, 4096, 4096, 16, 0, 1, True))
    for kind in ("normal", "ties"):
        cases.append((kind, 37, 300, 8, 1, 2, True))
        cases.append((kind, 16, 9000, 16, 0, 1, True))
    return cases


@benchmark()
def test_topk_per_row_small_k_prefill(
    rows, prefix, kind="normal", init_blocks=0, local_blocks=1
):
    """One request of `rows` new tokens after `prefix` cached tokens: row q sees ceil((prefix + q + 1) / 128)
    blocks, as a prefill scorer's rows do (one index head, 128-token blocks). Exact ids, and the op against the
    chunked build it replaces on these rows."""
    width = (prefix + rows + 127) // 128
    scores = make_kind(kind, rows, width, 3)
    row_lens = (
        ((prefix + torch.arange(rows) + 128) // 128).clamp(max=width).int().contiguous()
    )
    ref = exact_ids(scores, row_lens, 16, init_blocks, local_blocks)
    out = torch.empty(rows, 16, dtype=dtypes.i32)
    wave = get_gfx_runtime() == "gfx950" and bool(
        _wave_tile(16, width, 64, _wave_width_limit(rows, False))
    )
    ret = {"width": width, "one-wave": wave}
    got, ret["us"] = run_perftest(
        lambda: (
            topk_per_row_small_k(scores, row_lens, out, 16, init_blocks, local_blocks),
            out,
        )[1],
        num_iters=20,
        num_rotate_args=3,
    )
    exact = ret["one-wave"]
    ret["err"] = checkAllclose(
        (
            ref.float()
            if exact
            else selected_scores(scores, row_lens, ref, init_blocks, local_blocks)
        ),
        (
            got.float()
            if exact
            else selected_scores(scores, row_lens, got, init_blocks, local_blocks)
        ),
        rtol=0,
        atol=0,
        msg=f"rows{rows} prefix{prefix} {kind}: ",
    )
    if ret["one-wave"]:
        out2 = torch.empty(rows, 16, dtype=dtypes.i32)
        _, ret["chunked us"] = run_perftest(
            lambda: (
                launch_chunked(scores, row_lens, out2, 16, init_blocks, local_blocks),
                out2,
            )[1],
            num_iters=20,
            num_rotate_args=3,
        )
    return ret


def summarize(name, rows):
    aiter.logger.info(
        "%s (markdown):\n%s", name, pd.DataFrame(rows).to_markdown(index=False)
    )


def main():
    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning(
            "topk_per_row_small_k unsupported on %s; skipping", get_gfx()
        )
        return 0

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
    parser.add_argument(
        "--prefill-rows", type=int, nargs="*", default=[2048, 8192, 16384]
    )
    parser.add_argument(
        "--prefill-prefix", type=int, nargs="*", default=[32768, 102400, 163840]
    )
    args = parser.parse_args()

    fails = 0
    if get_gfx_runtime() == "gfx950":
        fails += test_topk_select_batch_invariant()
        exact = [test_topk_per_row_small_k_exact(*case) for case in exact_cases()]
        fails += sum(r["err"] != 0 for r in exact)
        aiter.logger.info("exact ids: %d/%d cases PASS", len(exact) - fails, len(exact))
        prefill = [
            test_topk_per_row_small_k_prefill(r, p, kind)
            for kind in ("normal", "ties")
            for r, p in itertools.product(args.prefill_rows, args.prefill_prefix)
        ]
        fails += sum(r["err"] != 0 for r in prefill)
        summarize("topk_per_row_small_k prefill", prefill)

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
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
