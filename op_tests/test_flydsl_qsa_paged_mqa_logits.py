# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL QSA paged MQA logits (prefill indexer scorer) vs a torch reference.

Per query row ``t`` of request ``r`` and compressed group ``c``::

    logits[t, c] = sum_h ReLU(<q[t, h, :], K_r[c, :]>) / divisor

where ``K_r`` is request ``r``'s compressed key cache gathered through its page
table (16 compressed rows per page). Row ``t`` sees groups
``[0, min((pos_t + 1) // 4, ctx_r // 4))``; only that window is checked, as
top-k reads nothing past it. ``visible_groups`` must match exactly and
``row_starts`` must be zeroed.

Cases cover prefill-sized inputs (the kernel serves > 256 query rows): a fresh
prefill, chunked prefill against a long cached context, unaligned multi-request
batches with padded rows, a column count that is not a tile multiple, and int64
metadata.

The column-tile sweep uses vLLM prefill geometry: chunks of up to 16384 rows,
contexts of 8K, 32K, 128K and 256K, and one, two or four requests. The page
table is padded to max_model_len 262144 (4096 pages of 64 tokens). This kernel's
compressed page is 16, not vLLM's 392-wide pool page, and its buffer addressing
stops at 2 GiB, so a 16384-row chunk at 128K or 256K is skipped.

Usage:
    python op_tests/test_flydsl_qsa_paged_mqa_logits.py
    python op_tests/test_flydsl_qsa_paged_mqa_logits.py -c chunked_8192
"""

import argparse
import itertools

import pandas as pd
import torch

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops.flydsl.kernels.mqa_logits.qsa_flydsl_layout import (
    _MAX_BYTES,
    COMPRESS_RATIO,
    HDIMS,
    PAGE_SIZE,
    Q_HEADS,
    _n_col_tiles_for,
    qsa_paged_mqa_logits_flydsl,
)
from aiter.test_common import benchmark, checkAllclose, run_perftest

torch.set_default_device("cuda")

SUPPORTED_GFX = ["gfx942", "gfx950"]

# vLLM max_model_len 262144. One compressed page covers 64 tokens.
_MAX_MODEL_LEN = 262144
_PAD_PAGES = _MAX_MODEL_LEN // (PAGE_SIZE * COMPRESS_RATIO)
_COL_TILES = (1, 2, 4, 8)

# name -> (requests as (cached_tokens, new_tokens), padded rows, extra columns,
#          int64 metadata). Padded rows carry request id -1 and must see nothing.
CASES = {
    "fresh_1040": ([(0, 1040)], 0, 0, False),
    "chunked_1040": ([(32256 - 1040, 1040)], 0, 0, False),
    "chunked_8192": ([(32256 - 8192, 8192)], 0, 0, False),
    "multi_req_unaligned": (
        [(0, 300), (517, 129), (4093, 1001), (0, 63), (12345, 700)],
        37,
        5,
        False,
    ),
    "multi_req_4x2048": ([(0, 2048)] * 4, 0, 0, False),
    "int64_metadata": ([(1000, 777), (0, 513)], 0, 0, True),
}


def make_inputs(reqs, padded_rows, extra_columns, int64_meta, seed, pad_pages=None):
    torch.manual_seed(seed)
    ctx = [c + n for c, n in reqs]
    pages_per_req = [-(-(c // COMPRESS_RATIO) // PAGE_SIZE) for c in ctx]
    # One spare logical page per request, left unmapped (-1).
    width = max(pages_per_req) + 1
    if pad_pages is not None:
        width = max(width, pad_pages)
    total_pages = sum(pages_per_req) + 3
    perm = torch.randperm(total_pages, device="cpu").to(dtypes.i32)
    page_table = torch.full((len(reqs), width), -1, dtype=dtypes.i32, device="cpu")
    o = 0
    for r, n in enumerate(pages_per_req):
        page_table[r, :n] = perm[o : o + n]
        o += n

    # Row t sees the complete groups up to its own position (pos < ctx).
    t2r, pos, visible = [], [], []
    for r, (c, n) in enumerate(reqs):
        t2r += [r] * n
        pos += list(range(c, c + n))
        visible += [(p + 1) // COMPRESS_RATIO for p in range(c, c + n)]
    t2r += [-1] * padded_rows
    pos += [0] * padded_rows
    visible += [0] * padded_rows
    rows = len(t2r)

    meta = dtypes.i64 if int64_meta else dtypes.i32
    q = torch.randn(rows, Q_HEADS, HDIMS, dtype=dtypes.bf16)
    k = torch.randn(total_pages, PAGE_SIZE, 1, HDIMS, dtype=dtypes.bf16)
    page_table = page_table.to("cuda", meta)
    token_to_request = torch.tensor(t2r, dtype=meta)
    query_positions = torch.tensor(pos, dtype=meta)
    context_lens = torch.tensor(ctx, dtype=meta)

    visible = torch.tensor(visible, dtype=dtypes.i32)
    columns = int(visible.max().item()) + extra_columns
    assert columns <= width * PAGE_SIZE
    return (
        q,
        k,
        page_table,
        token_to_request,
        query_positions,
        context_lens,
        visible,
        columns,
    )


def run_torch(q, k, page_table, token_to_request, columns, divisor):
    out = torch.zeros(q.shape[0], columns, dtype=dtypes.fp32)
    for r in range(page_table.shape[0]):
        rows = (token_to_request == r).nonzero().squeeze(1)
        if rows.numel() == 0:
            continue
        pages = page_table[r].long().clamp(min=0)
        kr = k[pages, :, 0, :].reshape(-1, HDIMS)[:columns].float()
        qr = q[rows].float()
        acc = torch.zeros(rows.numel(), columns, dtype=dtypes.fp32)
        for h in range(Q_HEADS):
            acc += (qr[:, h, :] @ kr.T).relu()
        out[rows] = acc / divisor
    return out


@benchmark()
def test_qsa_paged_mqa_logits(case, n_col_tiles=None):
    reqs, padded_rows, extra_columns, int64_meta = CASES[case]
    seed = list(CASES).index(case)
    (
        q,
        k,
        page_table,
        token_to_request,
        query_positions,
        context_lens,
        ref_visible,
        columns,
    ) = make_inputs(reqs, padded_rows, extra_columns, int64_meta, seed)
    divisor = HDIMS**0.5
    rows = q.shape[0]

    logits = torch.empty(rows, columns, dtype=dtypes.fp32)
    visible = torch.full((rows,), -7, dtype=dtypes.i32)
    row_starts = torch.full((rows,), -7, dtype=dtypes.i32)

    def launch():
        qsa_paged_mqa_logits_flydsl(
            q,
            k,
            page_table,
            token_to_request,
            query_positions,
            context_lens,
            COMPRESS_RATIO,
            divisor,
            logits,
            visible,
            row_starts,
            n_col_tiles=n_col_tiles,
        )
        return logits

    _, us = run_perftest(launch)

    assert torch.equal(visible, ref_visible), f"{case}: visible_groups mismatch"
    assert bool((row_starts == 0).all()), f"{case}: row_starts not zeroed"

    ref = run_torch(q, k, page_table, token_to_request, columns, divisor)
    mask = torch.arange(columns)[None, :] < ref_visible[:, None]
    err = checkAllclose(
        ref, logits, rtol=1e-3, atol=1e-3, mask=mask, msg=f"{case}: logits"
    )
    assert err == 0, f"{case}: logits mismatch ratio {err}"

    live = int(ref_visible.sum().item())
    return {
        "gfx": get_gfx_runtime(),
        "rows": rows,
        "columns": columns,
        "flydsl us": us,
        # Useful work only: the causal window each row actually scores.
        "TFLOPS": 2 * Q_HEADS * HDIMS * live / us / 1e6,
        "err": err,
    }


def _chunk_requests(chunk_rows, context, n_req, skew):
    """Tail chunk of ``context`` tokens, split evenly across ``n_req`` requests.

    ``skew == 0`` gives every request that context. ``skew == 1`` keeps the first
    request long and makes the others a fresh prompt of their own row count, so
    most row blocks exit before the far columns.
    """
    rows_i = chunk_rows // n_req
    if skew == 0:
        return [(context - rows_i, rows_i)] * n_req
    return [(context - rows_i, rows_i)] + [(0, rows_i)] * (n_req - 1)


def _skip_tile_shape(chunk_rows, context, n_req, skew):
    if n_req < 1 or chunk_rows % n_req != 0:
        return "rows are not divisible by the request count"
    rows_i = chunk_rows // n_req
    if context < rows_i:
        return "context is shorter than one request's rows"
    if skew == 1 and n_req == 1:
        return "skew needs more than one request"
    if skew == 1 and rows_i >= context:
        return "the short requests would not be shorter"
    columns = context // COMPRESS_RATIO
    if chunk_rows * columns * 4 > _MAX_BYTES:
        return "logits exceed the kernel's 2 GiB buffer addressing"
    return None


@benchmark()
def test_qsa_mqa_logits_col_tiles(chunk_rows, context, n_req, skew):
    """Score one prefill step at each column-tile count and at the heuristic."""
    reason = _skip_tile_shape(chunk_rows, context, n_req, skew)
    if reason is not None:
        raise ValueError(reason)
    reqs = _chunk_requests(chunk_rows, context, n_req, skew)
    (
        q,
        k,
        page_table,
        token_to_request,
        query_positions,
        context_lens,
        ref_visible,
        columns,
    ) = make_inputs(reqs, 0, 0, False, chunk_rows + context + n_req + skew, _PAD_PAGES)
    divisor = HDIMS**0.5
    rows = q.shape[0]
    ref = run_torch(q, k, page_table, token_to_request, columns, divisor)
    mask = torch.arange(columns)[None, :] < ref_visible[:, None]
    live = int(ref_visible.sum().item())
    flops = 2 * Q_HEADS * HDIMS * live
    # Q once, compressed K once, and the fp32 score buffer the kernel writes.
    nbytes = q.nbytes + k.nbytes + rows * columns * 4
    logits = torch.empty(rows, columns, dtype=dtypes.fp32)
    visible = torch.empty(rows, dtype=dtypes.i32)
    row_starts = torch.empty(rows, dtype=dtypes.i32)

    def launch(n_col_tiles):
        qsa_paged_mqa_logits_flydsl(
            q,
            k,
            page_table,
            token_to_request,
            query_positions,
            context_lens,
            COMPRESS_RATIO,
            divisor,
            logits,
            visible,
            row_starts,
            n_col_tiles=n_col_tiles,
        )
        return logits, visible, row_starts

    def bind(n_col_tiles):
        return lambda: launch(n_col_tiles)

    candidates = {f"tile{n}": bind(n) for n in _COL_TILES}
    candidates["heur"] = bind(None)

    ret = {
        "gfx": get_gfx_runtime(),
        "columns": columns,
        "heur_tiles": _n_col_tiles_for(rows, columns),
    }
    for name, fn in candidates.items():
        logits.fill_(float("nan"))
        visible.fill_(-7)
        row_starts.fill_(-7)
        (logits, visible, row_starts), us = run_perftest(fn)
        assert torch.equal(visible, ref_visible), f"{name}: visible_groups mismatch"
        assert bool((row_starts == 0).all()), f"{name}: row_starts not zeroed"
        err = checkAllclose(
            ref.to(dtypes.fp32),
            logits.to(dtypes.fp32),
            rtol=1e-3,
            atol=1e-3,
            mask=mask,
            msg=f"{name}: logits",
        )
        assert err == 0, f"{name}: {err:.2%} of elements outside rtol/atol"
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = err
    return ret


def main():
    if get_gfx_runtime() not in SUPPORTED_GFX:
        aiter.logger.warning(
            "qsa_paged_mqa_logits_flydsl unsupported on %s; skipping",
            get_gfx_runtime(),
        )
        return

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="config input of test",
    )
    parser.add_argument(
        "-c",
        "--case",
        type=str,
        nargs="*",
        choices=list(CASES),
        default=list(CASES),
        help="input cases to run",
    )
    parser.add_argument(
        "--rows",
        type=int,
        nargs="*",
        default=[4096, 8192, 16384],
        help="prefill-chunk rows",
    )
    parser.add_argument(
        "--context",
        type=int,
        nargs="*",
        default=[8192, 32768, 131072, 262144],
        help="longest request context, in tokens",
    )
    parser.add_argument(
        "--requests",
        type=int,
        nargs="*",
        default=[1, 2, 4],
        help="requests sharing the chunk",
    )
    parser.add_argument(
        "--skew",
        type=int,
        nargs="*",
        default=[0, 1],
        help="0: equal contexts; 1: one long request and the rest short",
    )
    args = parser.parse_args()

    df = pd.DataFrame([test_qsa_paged_mqa_logits(case) for case in args.case])
    aiter.logger.info(
        "qsa_paged_mqa_logits_flydsl summary (markdown):\n%s",
        df.to_markdown(index=False),
    )

    tile_rows = []
    for chunk_rows, context, n_req, skew in itertools.product(
        args.rows, args.context, args.requests, args.skew
    ):
        reason = _skip_tile_shape(chunk_rows, context, n_req, skew)
        if reason is not None:
            aiter.logger.warning(
                "skip rows=%s context=%s requests=%s skew=%s: %s",
                chunk_rows,
                context,
                n_req,
                skew,
                reason,
            )
            continue
        tile_rows.append(
            test_qsa_mqa_logits_col_tiles(chunk_rows, context, n_req, skew)
        )
    tile_df = pd.DataFrame(tile_rows)
    aiter.logger.info(
        "qsa_paged_mqa_logits column tiles (markdown):\n%s",
        tile_df.to_markdown(index=False),
    )


if __name__ == "__main__":
    main()
