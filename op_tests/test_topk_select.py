# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""`aiter.topk_select`: the dispatched per-row top-k, against its four backends.

Two halves. The sweep times `topk_select` beside every backend that can serve
the shape, so the dispatch can be read off the table rather than trusted; a row
where `topk_select` is far off the best column is a threshold that has moved.

The invariants are the other half, and they are the part a value comparison
cannot reach. Three bugs in this entry were silent -- an offset applied before
the gather, a row range handed to the backend that reads it as a whole-row
selection, and a `sorted_index` that reordered the indices without the values --
and all three returned plausible tensors. Each has an assertion here.

    python op_tests/test_topk_select.py
"""

import argparse
import itertools

import pandas as pd
import torch

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.topk_select import (
    _available,
    _wave_size,
    topk_select,
    topk_select_backend,
)
from aiter.test_common import benchmark, checkAllclose, run_perftest

torch.set_default_device("cuda")

SUPPORTED_GFX = ["gfx942", "gfx950"]


def run_torch(x, row_lens, k):
    """Reference: mask each row past its length, then torch's own selector."""
    masked = x.to(dtypes.fp32).clone()
    lens = row_lens.tolist()
    for r, n in enumerate(lens):
        masked[r, n:] = float("-inf")
    return torch.sort(
        torch.topk(masked, k, dim=1).values, dim=1, descending=True
    ).values


def _sorted_values(x, idx):
    """The selected scores, padding included, in descending order."""
    got = torch.where(
        idx >= 0,
        x.gather(1, idx.clamp_min(0).long()),
        torch.tensor(float("-inf"), device=x.device),
    )
    return torch.sort(got.to(dtypes.fp32), dim=1, descending=True).values


@benchmark()
def test_topk_select(m, n, k, tie, deterministic):
    x = torch.randn(m, n, dtype=dtypes.fp32)
    row_lens = torch.full((m,), n, dtype=dtypes.i32)
    ref = run_torch(x, row_lens, k)
    serving = _available(n, k, _wave_size(), False)

    candidates = {
        "topk_select": lambda: topk_select(x, k, tie=tie, deterministic=deterministic)[
            1
        ],
    }
    # Every backend that can serve this shape, so the dispatch is visible rather
    # than asserted. Driven through the entry with the others withheld, which is
    # the same path the dispatcher takes.
    for backend in sorted(serving):
        candidates[backend] = lambda b=backend: _run_single_backend(x, row_lens, k, b)

    # One pass of the row plus the k written back.
    nbytes = (m * n + m * k) * x.element_size()
    ret = {"gfx": get_gfx(), "picked": topk_select_backend(m, n, k, serving)}
    for name, fn in candidates.items():
        out, us = run_perftest(fn)
        ret[f"{name} us"] = us
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = checkAllclose(
            ref, _sorted_values(x, out), rtol=0, atol=0, msg=f"{name}: topk_select"
        )
    return ret


def _run_single_backend(x, row_lens, k, backend):
    """Call one backend through the entry by hiding the others from it.

    Asserts that the entry really used it. It did not, once: the dispatch tail
    could name a backend outside the set it was given, so hiding the others made
    `small_k` fall through to decode and this function timed decode under the
    small-k column -- 210.7us for a selector that runs in 40.2.
    """
    import aiter.ops.topk_select as ts

    keep = ts._BACKENDS_BY_TIE
    narrowed = dict(keep)
    narrowed[None] = (backend,)
    ts._BACKENDS_BY_TIE = narrowed
    try:
        rows, width = x.shape
        served = ts._available(width, k, ts._wave_size(), False) & {backend}
        picked = ts.topk_select_backend(rows, width, k, served)
        if picked != backend:
            raise AssertionError(f"asked for {backend}, the dispatch chose {picked}")
        return topk_select(x, k)[1]
    finally:
        ts._BACKENDS_BY_TIE = keep


def test_invariants(m, n, k):
    """The properties a value comparison cannot see."""
    failures = []
    x = torch.randn(m, n, dtype=dtypes.fp32)

    def want(cond, label):
        if not cond:
            failures.append(label)

    # values[j] is the score at idx[j], under every ordering flag. The pairing
    # broke once when `sorted_index` reordered the indices on their own.
    # `return_value` is off by default, so every check that reads the values
    # asks for them; the ones that only read indices deliberately do not, which
    # is also what keeps the default path covered.
    for flags in ({}, {"sorted": True}, {"sorted_index": True}):
        v, i = topk_select(x, k, return_value=True, **flags)
        torch.cuda.synchronize()
        want(torch.equal(v, x.gather(1, i.long())), f"values pair with idx {flags}")
    want(topk_select(x, k)[0] is None, "return_value defaults off")
    want(
        torch.equal(
            topk_select(x, k, sorted=True, return_value=True)[0],
            torch.sort(
                topk_select(x, k, return_value=True)[0], dim=1, descending=True
            ).values,
        ),
        "sorted=True orders the values",
    )
    # `sorted` has to order the indices even when the values are dropped, which
    # it can only do by gathering them anyway. It silently did nothing once.
    #
    # Tested as the property rather than against the `return_value=True` call:
    # two calls return the same set but not the same slot order, and where two
    # selected scores are exactly equal -- 2 slots in 64 rows of gaussian noise,
    # often enough to matter -- an unstable `torch.sort` puts that pair either
    # way round. Descending values is what `sorted` promises; identical tensors
    # is not.
    by_value = x.gather(1, topk_select(x, k, sorted=True)[1].long())
    want(
        bool((by_value[:, :-1] >= by_value[:, 1:]).all()),
        "sorted=True orders the indices without return_value",
    )
    v, i = topk_select(x, k, sorted_index=True)
    want(bool((i[:, :-1] <= i[:, 1:]).all()), "sorted_index orders the indices")

    # A row range must bound the answer, not be read as "no range given".
    end = torch.randint(k, n, (m,), dtype=dtypes.i32)
    _, i = topk_select(x, k, end=end)
    torch.cuda.synchronize()
    want(bool((i < end.view(m, 1)).all()), "end= bounds every index")
    want(
        torch.equal(_sorted_values(x, i), run_torch(x, end, k)),
        "end= selects the right values",
    )

    # The offset renumbers the output; the values still come from this tensor.
    # Compared under `deterministic=True` and on sorted indices, because the
    # default leaves both the slot order and -- on `plain` -- the selected set
    # free to move between calls, and comparing two calls element-wise would be
    # testing that freedom rather than the offset.
    off = torch.arange(m, dtype=dtypes.i32) * n
    base_v, base_i = topk_select(
        x, k, deterministic=True, sorted_index=True, return_value=True
    )
    shift_v, shifted = topk_select(
        x,
        k,
        deterministic=True,
        sorted_index=True,
        output_idx_offset=off,
        return_value=True,
    )
    torch.cuda.synchronize()
    want(torch.equal(shifted, base_i + off.view(m, 1)), "offset shifts the indices")
    want(torch.equal(shift_v, base_v), "offset leaves the values alone")

    # `tie` and `deterministic` promise the selected *set* does not move, not the
    # order it comes back in: the streaming selector takes its output slots from
    # a shared counter, so a row with far more ties at the cut than places
    # permutes between calls while selecting the same columns every time. Sorted
    # here for that reason, and `sorted_index=True` below is the stronger promise
    # for callers who need the tensor itself to be reproducible.
    tied = x.clone()
    tied[:, : 4 * k] = 7.0
    for kw in ({"tie": "low"}, {"deterministic": True}):
        first = torch.sort(topk_select(tied, k, **kw)[1], dim=1).values
        again = torch.sort(topk_select(tied, k, **kw)[1], dim=1).values
        torch.cuda.synchronize()
        want(torch.equal(first, again), f"selects the same set under {kw}")
        pinned = topk_select(tied, k, sorted_index=True, **kw)[1]
        repeat = topk_select(tied, k, sorted_index=True, **kw)[1]
        torch.cuda.synchronize()
        want(torch.equal(pinned, repeat), f"sorted_index is reproducible under {kw}")

    for label in failures:
        aiter.logger.error("INVARIANT FAILED: %s", label)
    return failures


def main():
    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning("topk_select unsupported on %s; skipping", get_gfx())
        return

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="config input of test",
    )
    parser.add_argument(
        "-m", "--rows", type=int, nargs="*", default=[1, 64, 1024], help="rows per call"
    )
    parser.add_argument(
        "-n",
        "--width",
        type=int,
        nargs="*",
        default=[2048, 32768, 262144],
        help="columns per row",
    )
    parser.add_argument(
        "-k",
        "--topk",
        type=int,
        nargs="*",
        default=[16, 512, 2048],
        help="elements selected per row",
    )
    parser.add_argument(
        "-t",
        "--tie",
        type=str,
        nargs="*",
        default=["none"],
        choices=["none", "low", "high"],
    )
    args = parser.parse_args()

    bad = test_invariants(64, 32768, 512)
    aiter.logger.info(
        "invariants: %s", "all hold" if not bad else f"{len(bad)} FAILED: {bad}"
    )

    for tie in args.tie:
        df = []
        for m, n, k in itertools.product(args.rows, args.width, args.topk):
            if k > n or m * n * 4 >= 2**32:
                continue
            df.append(test_topk_select(m, n, k, None if tie == "none" else tie, False))
        df = pd.DataFrame(df)
        aiter.logger.info(
            "topk_select summary (tie=%s):\n%s", tie, df.to_markdown(index=False)
        )


if __name__ == "__main__":
    main()
