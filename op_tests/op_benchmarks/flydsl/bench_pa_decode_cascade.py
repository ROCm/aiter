# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Isolated cascade-vs-plain decode attention benchmark.

Builds one attention layer's KV pool and block tables from a synthetic
conversation tree (sibling groups with a long shared prefix and per-row
private tails, shaped like the coding-agent traces in the design doc),
then times the plain FlyDSL ``pa_decode`` against
``pa_decode_cascade`` (grouping planner + three-pass op) on the same
tables.

Run with: python3 op_tests/op_benchmarks/flydsl/bench_pa_decode_cascade.py
"""

import argparse

import torch

PAGE = 16


def build_traffic(batch, mean_ctx, fanout, shared_frac, width_max=None):
    """Tree traffic: groups of ~fanout rows share shared_frac of the prefix.

    Returns (block_tables, context_lengths) on CUDA. Physical page ids are
    allocated densely, shared where rows agree -- the same distribution a
    prefix-cached server produces.
    """
    lens = (
        torch.clamp(
            torch.randn(batch, dtype=torch.float32) * (mean_ctx * 0.1) + mean_ctx,
            min=mean_ctx // 2,
            max=mean_ctx * 3 // 2,
        )
        .round()
        .to(torch.int32)
    )
    depths = (lens + (PAGE - 1)) // PAGE
    width = int(depths.max()) + 1
    if width_max:
        width = max(width, width_max)
    bt = torch.zeros(batch, width, dtype=torch.int32)
    cursor = 0
    r = 0
    while r < batch:
        size = min(int(torch.poisson(torch.tensor(fanout - 1.0)).item()) + 1, batch - r)
        base_depth = int(depths[r : r + size].min()) if size > 1 else int(depths[r])
        shared = max(int(base_depth * shared_frac), 1) if size > 1 else 0
        base = torch.arange(cursor, cursor + shared, dtype=torch.int32)
        cursor += shared
        for i in range(size):
            depth = int(depths[r + i])
            tail = torch.arange(cursor, cursor + depth - shared, dtype=torch.int32)
            cursor += depth - shared
            row = torch.cat([base[:shared], tail])
            bt[r + i, : row.numel()] = row
        r += size
    return bt.cuda(), lens.cuda(), cursor


def time_op(fn, iters=20, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=340)
    ap.add_argument("--mean-ctx", type=int, default=39000)
    ap.add_argument("--fanout", type=float, default=4.0)
    ap.add_argument("--shared-frac", type=float, default=0.85)
    ap.add_argument("--kv-heads", type=int, default=2)
    ap.add_argument("--group", type=int, default=4)
    ap.add_argument("--head-dim", type=int, default=512)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument(
        "--no-planner",
        action="store_true",
        help="exclude planner fill from cascade timing",
    )
    args = ap.parse_args()

    from aiter.ops.flydsl.kernels.pa_decode_plan import plan_pa_decode
    from aiter.ops.flydsl.pa_decode import pa_decode
    from aiter.ops.flydsl.pa_decode_cascade import (
        pa_decode_cascade,
        plan_pa_decode_cascade,
    )
    from aiter.ops.flydsl.pa_decode_cascade_plan import (
        fill_pa_decode_cascade_plan,
        plan_pa_decode_cascade_groups,
    )

    torch.manual_seed(0)
    device = torch.device("cuda")
    heads = args.kv_heads * args.group
    dim = args.head_dim

    bt, lens, num_pages = build_traffic(
        args.batch, args.mean_ctx, args.fanout, args.shared_frac
    )
    print(f"traffic: batch={args.batch} pages={num_pages} width={bt.shape[1]}")
    logical_blocks = int(lens.sum().item()) // PAGE
    print(
        f"logical KV bytes {logical_blocks * PAGE * dim * args.kv_heads * 2 * 2 / 2**30:.1f} GiB"
    )

    kc = (
        torch.randn(num_pages, args.kv_heads, PAGE, dim // 8, 8, device=device)
        .permute(0, 1, 3, 2, 4)
        .contiguous()
        .bfloat16()
    )
    vc = (
        torch.randn(num_pages, args.kv_heads, PAGE // 8, 8, dim, device=device)
        .permute(0, 1, 2, 4, 3)
        .contiguous()
        .bfloat16()
    )
    query = torch.randn(args.batch, heads, dim, device=device).bfloat16()
    out = torch.empty_like(query)

    scale = dim**-0.5
    plan = plan_pa_decode(lens, args.kv_heads)
    t_plain = time_op(
        lambda: pa_decode(out, query, kc, vc, lens, bt, scale, 1, work_plan=plan),
        args.iters,
    )

    ws = plan_pa_decode_cascade(
        device,
        num_q_heads=heads,
        num_kv_heads=args.kv_heads,
        head_dim=dim,
        members_per_item=4,
        max_items=args.batch // 2,
        max_singles=args.batch,
        table_width=bt.shape[1],
    )
    pl = plan_pa_decode_cascade_groups(
        device, max_rows=args.batch, table_width=bt.shape[1], max_b_rows=ws.max_b_rows
    )

    def cascade():
        if not args.no_planner:
            fill_pa_decode_cascade_plan(ws, pl, bt, lens)
        pa_decode_cascade(out, query, kc, vc, lens, bt, scale, workspace=ws)

    t_cascade = time_op(cascade, args.iters)
    planner_ms = time_op(
        lambda: fill_pa_decode_cascade_plan(ws, pl, bt, lens), args.iters
    )

    n_items = int(ws.counts[0])
    sum(
        int(ws.items[i, 1]) - 1 for i in range(n_items)
    ) * 0  # placeholder; reported via traffic stats below
    groups = ws.items[:n_items, 1]
    sblk = ws.items[:n_items, 0]
    saved = int((((groups - 1) * sblk).sum()).item())
    total = int(lens.cpu().div(PAGE, rounding_mode="floor").sum())
    print(
        f"groups: {n_items} items, {int(ws.counts[1])} members, {int(ws.counts[2])} singles"
    )
    print(f"dedup coverage: {saved}/{total} full blocks ({saved/max(total,1):.1%})")
    print(f"plain pa_decode : {t_plain:8.3f} ms")
    print(
        f"cascade op      : {t_cascade:8.3f} ms  (planner fill {planner_ms:.3f} ms included: {not args.no_planner})"
    )
    print(f"speedup         : {t_plain / t_cascade:6.2f}x")


if __name__ == "__main__":
    main()
