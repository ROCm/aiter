#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""FlyDSL vs vLLM Triton QSA at Qwen3.8-Flash-Next TP2 shapes, run as vLLM runs them.

Not a unit test: a standalone benchmark for ROCm/aiter#5996 that compares the
FlyDSL K1/K2 against the Triton QSA path vLLM ships on ROCm, on caches laid out
the way vLLM lays them out.

- Block tables are padded to max_model_len: 168 pages of 392 compressed indexer
  rows (65856 columns) and of 1568 KV tokens.
- Cache pages have a padded page stride and are reached through an as_strided
  view, as a layer's view into vLLM's shared KV pool is. With --wide the page
  stride is stretched so the view spans more than 4 GiB (5 GiB here; 194 GiB
  in a real server), which is what every serving call sees and which selects
  AITER's wide K1/K2 body.
- Decode is timed under HIP-graph replay, as vLLM captures it. Prefill is timed
  eagerly, with the table trimmed to the longest request, as #59437 does.
- Prefill batches include multiple requests and prompts with decode rows
  attached, at the scheduler's default 16384-token budget.

Rows printed per case:

    select      vLLM's selection entry point (score + top-k + expand)
    k1 parts    scorer+selector at the full width (captured) vs at the live
                width, and each selector alone
    triton      qsa_mqa_paged scorer and top_k_per_row_decode alone
    k2          sparse GQA, FlyDSL vs Triton, on Triton's selection, with the
                max abs difference of the outputs

Times are microseconds per call.

Requirements, on gfx950 (one GPU, about 20 GiB free for --wide):

- A ROCm vLLM build from main (e.g. the vllm/vllm-openai-rocm nightly), which
  provides vllm.models.qwen4_exp.amd.ops.qsa (the Triton path).
- AITER at this PR, installed over the image's AITER.
- vllm/models/qwen4_exp/amd/ops/qsa_flydsl.py from vllm-project/vllm#59437
  (the vLLM glue around K1/K2), copied into the installed vllm package:

    SITE=$(python3 -c 'import vllm, os; print(os.path.dirname(vllm.__file__))')
    curl -fsSL -o $SITE/models/qwen4_exp/amd/ops/qsa_flydsl.py \\
      https://raw.githubusercontent.com/mjkvaak-amd/vllm-project/60d258fb67/vllm/models/qwen4_exp/amd/ops/qsa_flydsl.py

Run:

    HIP_VISIBLE_DEVICES=<gpu> python3 op_tests/bench_flydsl_qsa_vllm_layout.py          # cache < 4 GiB
    HIP_VISIBLE_DEVICES=<gpu> python3 op_tests/bench_flydsl_qsa_vllm_layout.py --wide   # cache > 4 GiB
    ... [--prefill-only] [--rows 4 16 64] [--ctx 8192 60000]
"""

from __future__ import annotations

import argparse
import os

os.environ["VLLM_ROCM_QSA_FLYDSL"] = "1"

import torch
from vllm import _custom_ops as ops
from vllm.models.qwen4_exp.amd.ops import qsa as tqsa
from vllm.models.qwen4_exp.amd.ops import qsa_flydsl as fqsa

from aiter.ops.flydsl.kernels.qsa import k1 as fly_k1
from aiter.ops.flydsl.topk.topk_per_row import flydsl_top_k_per_row_decode
from aiter.ops.topk_select import topk_select

PAGES = 168
IDX_PAGE = 392
KV_PAGE = 1568
IDX_HEADS, IDX_D = 4, 128
Q_HEADS, KV_HEADS, D = 12, 1, 256
RATIO, TOKEN_TOPK, BLOCK_TOPK = 4, 2048, 512
SCALE = IDX_D**-0.5


def graph_us(fn, iters: int = 200) -> float:
    fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        g.capture_begin()
        fn()
        g.capture_end()
    torch.cuda.current_stream().wait_stream(s)
    for _ in range(20):
        g.replay()
    start, end = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
    start.record()
    for _ in range(iters):
        g.replay()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1e3 / iters


def eager_us(fn, iters: int = 200) -> float:
    for _ in range(20):
        fn()
    torch.cuda.synchronize()
    start, end = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1e3 / iters


WIDE_SPAN_BYTES = 0


def strided_pages(n_pages: int, page: int, heads: int, d: int) -> torch.Tensor:
    """A [pages, page, heads, d] view with a padded page stride, like vLLM's layer view.

    With --wide the page stride is stretched so the view spans more than 4 GiB,
    as a layer view into vLLM's shared KV pool does.
    """
    page_elems = page * heads * d
    page_stride = page_elems + 256
    if WIDE_SPAN_BYTES:
        page_stride = max(page_stride, (WIDE_SPAN_BYTES // 2 // n_pages) // 256 * 256)
    numel = (n_pages - 1) * page_stride + page_elems
    buf = torch.empty(numel, dtype=torch.bfloat16, device="cuda")
    view = buf.as_strided((n_pages, page, heads, d), (page_stride, heads * d, d, 1))
    view.normal_()
    return view


def case(rows: int, ctx: int) -> dict[str, float]:
    dev = torch.device("cuda")
    n_req = rows
    t2r = torch.arange(rows, dtype=torch.int32, device=dev)
    slen = torch.full((n_req,), ctx, dtype=torch.int32, device=dev)
    qpos = (slen - 1).clone()

    compressed = -(-ctx // RATIO)
    idx_pages_per_req = -(-compressed // IDX_PAGE)
    kc = strided_pages(n_req * idx_pages_per_req, IDX_PAGE, 1, IDX_D)
    pt = torch.zeros((n_req, PAGES), dtype=torch.int32, device=dev)
    pt[:, :idx_pages_per_req] = torch.arange(
        n_req * idx_pages_per_req, dtype=torch.int32, device=dev
    ).view(n_req, idx_pages_per_req)
    q_idx = torch.randn(rows, IDX_HEADS, IDX_D, dtype=torch.bfloat16, device=dev)
    n_columns = PAGES * IDX_PAGE
    live = min(ctx // RATIO, n_columns)

    width = TOKEN_TOPK + RATIO - 1
    sel_t = torch.empty((rows, width), dtype=torch.int32, device=dev)
    sel_f = torch.empty_like(sel_t)
    res: dict[str, float] = {}

    def t_select():
        tqsa.qsa_select_paged_tokens(
            q_idx, kc, pt, t2r, qpos, slen, TOKEN_TOPK, RATIO, out=sel_t
        )

    def f_select():
        assert (
            fqsa.flydsl_select_paged_tokens(
                q_idx, kc, pt, t2r, qpos, slen, TOKEN_TOPK, RATIO, out=sel_f
            )
            is not None
        )

    res["select triton (graph)"] = graph_us(t_select)
    res["select flydsl (graph)"] = graph_us(f_select)
    res["select flydsl (eager, readback)"] = eager_us(f_select)
    res["select triton (eager)"] = eager_us(t_select)

    blocks = torch.empty((rows, BLOCK_TOPK), dtype=torch.int32, device=dev)

    def k1_full():
        fly_k1.qsa_k1_score_and_select(
            q_idx, kc, pt, t2r, qpos, slen, blocks, n_columns, SCALE, IDX_HEADS
        )

    def k1_live():
        fly_k1.qsa_k1_score_and_select(
            q_idx,
            kc,
            pt,
            t2r,
            qpos,
            slen,
            blocks,
            n_columns,
            SCALE,
            IDX_HEADS,
            live_columns=live,
        )

    res["k1 score+select, full width"] = graph_us(k1_full)
    res["k1 score+select, live width"] = graph_us(k1_live)

    scores = torch.randn(rows, n_columns, dtype=torch.float32, device=dev)
    row_lens = torch.full((rows,), live, dtype=torch.int32, device=dev)

    res["topk_select stream, full width"] = graph_us(
        lambda: topk_select(
            scores, BLOCK_TOPK, end=row_lens, output_idx=blocks, tie="low"
        )
    )
    narrow = scores.narrow(1, 0, live)
    res["flydsl decode radix, live width"] = graph_us(
        lambda: flydsl_top_k_per_row_decode(
            narrow,
            1,
            row_lens,
            blocks,
            rows,
            narrow.stride(0),
            narrow.stride(1),
            k=BLOCK_TOPK,
            stable=True,
        )
    )
    res["vllm top_k_per_row_decode, full width"] = graph_us(
        lambda: ops.top_k_per_row_decode(
            scores,
            1,
            row_lens,
            blocks,
            rows,
            scores.stride(0),
            scores.stride(1),
            BLOCK_TOPK,
        )
    )
    res["triton qsa_mqa_paged scorer"] = graph_us(
        lambda: tqsa.qsa_mqa_paged(q_idx, kc, pt, t2r, qpos, slen, RATIO)
    )

    t_select()
    torch.cuda.synchronize()
    kv_pages_per_req = -(-ctx // KV_PAGE)
    k_cache = strided_pages(n_req * kv_pages_per_req, KV_PAGE, KV_HEADS, D)
    v_cache = strided_pages(n_req * kv_pages_per_req, KV_PAGE, KV_HEADS, D)
    bt = torch.zeros((n_req, PAGES), dtype=torch.int32, device=dev)
    bt[:, :kv_pages_per_req] = torch.arange(
        n_req * kv_pages_per_req, dtype=torch.int32, device=dev
    ).view(n_req, kv_pages_per_req)
    q = torch.randn(rows, Q_HEADS, D, dtype=torch.bfloat16, device=dev)
    o_t = torch.empty_like(q)
    o_f = torch.empty_like(q)
    res["k2 triton sparse GQA"] = graph_us(
        lambda: tqsa.qsa_sparse_paged_attention(
            q, k_cache, v_cache, sel_t, bt, t2r, out=o_t
        )
    )
    if fqsa.flydsl_sparse_paged_attention(q, k_cache, v_cache, sel_t, bt, t2r, o_f):
        res["k2 flydsl sparse GQA"] = graph_us(
            lambda: fqsa.flydsl_sparse_paged_attention(
                q, k_cache, v_cache, sel_t, bt, t2r, o_f
            )
        )
        res["k2 max |flydsl - triton|"] = float((o_f.float() - o_t.float()).abs().max())
    return res


def prefill_case(segments: list[tuple[int, int]]) -> dict[str, float]:
    """One batch: each (rows, end) is a request scheduling positions [end - rows, end).

    Prefill batches run outside the graphs, and #59437 trims the indexer table
    to the longest live request, so both the table and the timing are eager.
    """
    dev = torch.device("cuda")
    n_req = len(segments)
    lens = [end for _, end in segments]
    slen = torch.tensor(lens, dtype=torch.int32, device=dev)
    t2r_l: list[int] = []
    qpos_l: list[int] = []
    for req, (n, end) in enumerate(segments):
        t2r_l += [req] * n
        qpos_l += list(range(end - n, end))
    t2r = torch.tensor(t2r_l, dtype=torch.int32, device=dev)
    qpos = torch.tensor(qpos_l, dtype=torch.int32, device=dev)
    rows = t2r.numel()

    compressed = -(-max(lens) // RATIO)
    width_pages = -(-compressed // IDX_PAGE)
    kc = strided_pages(n_req * width_pages, IDX_PAGE, 1, IDX_D)
    pt = torch.arange(n_req * width_pages, dtype=torch.int32, device=dev).view(
        n_req, width_pages
    )
    q_idx = torch.randn(rows, IDX_HEADS, IDX_D, dtype=torch.bfloat16, device=dev)
    width = TOKEN_TOPK + RATIO - 1
    sel_t = torch.empty((rows, width), dtype=torch.int32, device=dev)
    sel_f = torch.empty_like(sel_t)
    res: dict[str, float] = {}

    def t_select():
        tqsa.qsa_select_paged_tokens(
            q_idx, kc, pt, t2r, qpos, slen, TOKEN_TOPK, RATIO, out=sel_t
        )

    def f_select():
        assert (
            fqsa.flydsl_select_paged_tokens(
                q_idx, kc, pt, t2r, qpos, slen, TOKEN_TOPK, RATIO, out=sel_f
            )
            is not None
        )

    res["select triton"] = eager_us(t_select, 20)
    res["select flydsl"] = eager_us(f_select, 20)
    res["triton qsa_mqa_paged scorer"] = eager_us(
        lambda: tqsa.qsa_mqa_paged(q_idx, kc, pt, t2r, qpos, slen, RATIO), 20
    )
    n_columns = width_pages * IDX_PAGE
    blocks = torch.empty((rows, BLOCK_TOPK), dtype=torch.int32, device=dev)
    res["k1 score+select"] = eager_us(
        lambda: fly_k1.qsa_k1_score_and_select(
            q_idx, kc, pt, t2r, qpos, slen, blocks, n_columns, SCALE, IDX_HEADS
        ),
        20,
    )

    t_select()
    torch.cuda.synchronize()
    kv_pages = -(-max(lens) // KV_PAGE)
    k_cache = strided_pages(n_req * kv_pages, KV_PAGE, KV_HEADS, D)
    v_cache = strided_pages(n_req * kv_pages, KV_PAGE, KV_HEADS, D)
    bt = torch.arange(n_req * kv_pages, dtype=torch.int32, device=dev).view(
        n_req, kv_pages
    )
    q = torch.randn(rows, Q_HEADS, D, dtype=torch.bfloat16, device=dev)
    o_t = torch.empty_like(q)
    o_f = torch.empty_like(q)
    res["k2 triton sparse GQA"] = eager_us(
        lambda: tqsa.qsa_sparse_paged_attention(
            q, k_cache, v_cache, sel_t, bt, t2r, out=o_t
        ),
        20,
    )
    if fqsa.flydsl_sparse_paged_attention(q, k_cache, v_cache, sel_t, bt, t2r, o_f):
        res["k2 flydsl sparse GQA"] = eager_us(
            lambda: fqsa.flydsl_sparse_paged_attention(
                q, k_cache, v_cache, sel_t, bt, t2r, o_f
            ),
            20,
        )
        res["k2 max |flydsl - triton|"] = float((o_f.float() - o_t.float()).abs().max())
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, nargs="+", default=[4, 16, 64])
    ap.add_argument("--ctx", type=int, nargs="+", default=[8192, 60000])
    ap.add_argument("--prefill-only", action="store_true")
    ap.add_argument("--wide", action="store_true", help="cache views span > 4 GiB")
    args = ap.parse_args()
    global WIDE_SPAN_BYTES
    WIDE_SPAN_BYTES = (5 << 30) if args.wide else 0
    torch.manual_seed(0)
    print(
        torch.cuda.get_device_name(),
        "| table",
        PAGES,
        "pages,",
        PAGES * IDX_PAGE,
        "columns",
    )
    cases = {
        "8k: one 8192 prompt": [(8192, 8192)],
        "8k: one 8192 prompt + 3 decodes": [(8192, 8192)] + [(1, 8192)] * 3,
        "8k: two 8192 prompts (MBT 16384)": [(8192, 8192), (8192, 8192)],
        "8k: 8192 prompt + 8189 of the next + 3 decodes": [(8192, 8192), (8189, 8189)]
        + [(1, 8192)] * 3,
        "60k: one 16384 chunk ending at 60000": [(16384, 60000)],
        "60k: 16381 chunk + 3 decodes": [(16381, 60000)] + [(1, 60000)] * 3,
        "60k: 16384 chunk at 32768": [(16384, 32768)],
    }
    for name, segs in cases.items():
        res = prefill_case(segs)
        print(f"\n## prefill {name}: {sum(n for n, _ in segs)} rows (eager)")
        for k, v in res.items():
            print(f"  {k:<42} {v:10.1f}")
    if args.prefill_only:
        return
    for ctx in args.ctx:
        for rows in args.rows:
            res = case(rows, ctx)
            print(
                f"\n## decode rows={rows} ctx={ctx} (live columns {min(ctx // RATIO, PAGES * IDX_PAGE)})"
            )
            for k, v in res.items():
                print(f"  {k:<42} {v:10.1f}")


if __name__ == "__main__":
    main()
