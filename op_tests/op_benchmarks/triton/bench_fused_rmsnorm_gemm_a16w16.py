# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""fused_rmsnorm_gemm_a16w16 against the norm kernel + gemm_a16w16 it
replaces, timed as CUDA graph replays (the gap between kernels counts).

Cases, at the Qwen3.5 MTP drafter shapes:
  fc       cat(norm_e(embed[ids]), norm_h(hidden)) @ W^T, N 2048, K 4096:
           aiter fused_qk_rmsnorm_group_quant (two inputs, gathered) + GEMM.
  fc_pre   the same with norm_e(embed) normed beforehand (a table): the
           fused kernel gathers it and norms only the hidden half.
  qkv      fused_rmsnorm_add (Gemma) + GEMM, N 2560, K 2048.

Every replay reads another copy of the weights (--copies), so W comes
from HBM as in the model, where the LM head evicts it.
"""

import argparse

import torch

from aiter import logger
from aiter.ops.fused_qk_rmsnorm_group_quant import fused_qk_rmsnorm_group_quant
from aiter.ops.triton.fusions.fused_rmsnorm_add import fused_rmsnorm_add
from aiter.ops.triton.gemm.basic.gemm_a16w16 import gemm_a16w16
from aiter.ops.triton.gemm.fused.fused_rmsnorm_gemm_a16w16 import (
    fused_rmsnorm_gemm_a16w16,
)

H = 2048
EPS = 1e-6
VOCAB = 248320


def _graph_us(fn, args_list, iters=10, rounds=20):
    """Median us of one fn call, from replays of a graph of all calls."""
    for args in args_list:
        fn(*args)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for args in args_list:
            fn(*args)
    for _ in range(3):
        graph.replay()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(rounds):
        start.record()
        for _ in range(iters):
            graph.replay()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1e3 / iters / len(args_list))
    return sorted(times)[len(times) // 2]


def _aiter_norm_rows(x, w, out):
    fused_qk_rmsnorm_group_quant(
        q=x, q_weight=w, q_epsilon=EPS, q_out_unquantized=out, gemma_norm=True
    )


def bench_fc(Ms, copies, pre_normed):
    N = 2048
    embed = torch.randn((VOCAB, H), dtype=torch.bfloat16, device="cuda")
    we = torch.randn((H,), dtype=torch.bfloat16, device="cuda") * 0.1
    wh = torch.randn((H,), dtype=torch.bfloat16, device="cuda") * 0.1
    ws = [
        torch.randn((N, 2 * H), dtype=torch.bfloat16, device="cuda") * 0.02
        for _ in range(copies)
    ]
    table = None
    if pre_normed:
        table = torch.empty_like(embed)
        for i in range(0, VOCAB, 8192):
            _aiter_norm_rows(embed[i : i + 8192], we, table[i : i + 8192])

    def old(ids, hidden, w, a):
        fused_qk_rmsnorm_group_quant(
            q=embed,
            q_weight=we,
            q_epsilon=EPS,
            q_out_unquantized=a[:, :H],
            k=hidden,
            k_weight=wh,
            k_epsilon=EPS,
            k_out=a[:, H:],
            gemma_norm=True,
            q_row_index=ids,
        )
        return gemm_a16w16(a, w)

    def new(ids, hidden, w, a):
        if pre_normed:
            return fused_rmsnorm_gemm_a16w16(
                table, w, None, None, hidden, wh, EPS, row_index=ids
            )
        return fused_rmsnorm_gemm_a16w16(
            embed, w, we, EPS, hidden, wh, EPS, row_index=ids
        )

    rows = []
    for M in Ms:
        args = [
            (
                torch.randint(0, VOCAB, (M,), dtype=torch.int32, device="cuda"),
                torch.randn((M, H), dtype=torch.bfloat16, device="cuda"),
                ws[i],
                torch.empty((M, 2 * H), dtype=torch.bfloat16, device="cuda"),
            )
            for i in range(copies)
        ]
        rows.append((M, _graph_us(old, args), _graph_us(new, args)))
    return rows


def bench_qkv(Ms, copies):
    N = 2560
    nw = torch.randn((H,), dtype=torch.bfloat16, device="cuda") * 0.1
    ws = [
        torch.randn((N, H), dtype=torch.bfloat16, device="cuda") * 0.02
        for _ in range(copies)
    ]

    def old(x, w):
        return gemm_a16w16(fused_rmsnorm_add(x, nw, EPS, gemma_norm=True), w)

    def new(x, w):
        return fused_rmsnorm_gemm_a16w16(x, w, nw, EPS, norm_mode="gluon")

    rows = []
    for M in Ms:
        args = [
            (torch.randn((M, H), dtype=torch.bfloat16, device="cuda"), ws[i])
            for i in range(copies)
        ]
        rows.append((M, _graph_us(old, args), _graph_us(new, args)))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--case", choices=["fc", "fc_pre", "qkv", "all"], default="all")
    parser.add_argument("-M", type=int, nargs="+", default=[8, 16, 32, 64, 80, 320])
    parser.add_argument("--copies", type=int, default=64)
    args = parser.parse_args()
    cases = ["fc", "fc_pre", "qkv"] if args.case == "all" else [args.case]
    for case in cases:
        if case == "qkv":
            rows = bench_qkv(args.M, args.copies)
        else:
            rows = bench_fc(args.M, args.copies, pre_normed=case == "fc_pre")
        for M, t_old, t_new in rows:
            logger.warning(
                "%-6s M=%4d  norm + gemm %6.2f us  fused %6.2f us  saved %6.2f us",
                case,
                M,
                t_old,
                t_new,
                t_old - t_new,
            )


if __name__ == "__main__":
    main()
