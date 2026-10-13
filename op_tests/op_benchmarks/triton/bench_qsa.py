# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Benchmark for QSA paged token selection and sparse paged GQA.

Shapes from Qwen3.8-Flash-Next-120B:
  index_heads=4, index_dim=128, compress_ratio=4, budget=2048,
  query_heads=24, query_dim=256, kv_heads=2, block_size=64.

Usage:
    python op_tests/op_benchmarks/triton/bench_qsa.py
    python op_tests/op_benchmarks/triton/bench_qsa.py --op select
    python op_tests/op_benchmarks/triton/bench_qsa.py --op gqa -metric bandwidth
"""

import argparse

import torch
import triton

from aiter.ops.triton.attention.qsa import (
    qsa_select_paged_tokens,
    qsa_sparse_paged_gqa,
)
from op_tests.op_benchmarks.triton.utils.benchmark_utils import get_caller_name_no_ext

torch.set_default_device("cuda")

# Qwen3.8-Flash-Next-120B model config
BLOCK_SIZE = 64
RATIO = 4
COMPRESSED_PAGE_SIZE = BLOCK_SIZE // RATIO  # 16
INDEX_HEADS = 4
INDEX_DIM = 128
QUERY_HEADS = 24
QUERY_DIM = 256
KV_HEADS = 2
BUDGET = 2048


def _make_scenario(batch, tokens_per_seq, context_len):
    """Build tensors for a multi-request paged QSA scenario.

    Args:
        batch: number of requests
        tokens_per_seq: tokens per request (1 for decode, >1 for prefill/verify)
        context_len: context length per request
    """
    rows = batch * tokens_per_seq
    pages_per_req = (context_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    total_pages = batch * pages_per_req

    positions = torch.arange(rows, dtype=torch.int32)
    for b in range(batch):
        start = b * tokens_per_seq
        positions[start : start + tokens_per_seq] = torch.arange(
            context_len - tokens_per_seq,
            context_len,
            dtype=torch.int32,
        )

    tables = torch.randint(0, total_pages, (batch, pages_per_req), dtype=torch.int32)
    requests = torch.arange(batch, dtype=torch.int32).repeat_interleave(tokens_per_seq)
    lengths = torch.full((batch,), context_len, dtype=torch.int32)

    index_q = torch.randn(rows, INDEX_HEADS, INDEX_DIM, dtype=torch.bfloat16)
    compressed = torch.randn(
        total_pages,
        COMPRESSED_PAGE_SIZE,
        1,
        INDEX_DIM,
        dtype=torch.bfloat16,
    )
    query = torch.randn(rows, QUERY_HEADS, QUERY_DIM, dtype=torch.bfloat16)
    keys = torch.randn(
        total_pages, BLOCK_SIZE, KV_HEADS, QUERY_DIM, dtype=torch.bfloat16
    )
    values = torch.randn_like(keys)
    return {
        "positions": positions,
        "tables": tables,
        "requests": requests,
        "lengths": lengths,
        "index_q": index_q,
        "compressed": compressed,
        "query": query,
        "keys": keys,
        "values": values,
        "rows": rows,
        "total_pages": total_pages,
        "pages_per_req": pages_per_req,
    }


# -- qsa_select_paged_tokens (scoring + topk + expand) --------------------

# (batch, tokens_per_seq, context_len)
_SELECT_SHAPES = [
    # decode: single request
    (1, 1, 4096),
    (1, 1, 16384),
    (1, 1, 65536),
    # decode: concurrent requests
    (16, 1, 4096),
    (32, 1, 4096),
    (64, 1, 4096),
    (128, 1, 4096),
    # MTP verification
    (16, 3, 4096),
    (32, 3, 4096),
    # prefill
    (1, 1024, 4096),
    (1, 4096, 4096),
]


def bench_select(args):
    unit = "ms" if args.metric == "time" else "GB/s"

    config = triton.testing.Benchmark(
        x_names=["batch", "tokens_per_seq", "context_len"],
        x_vals=_SELECT_SHAPES,
        line_arg="provider",
        line_vals=["select"],
        line_names=[f"qsa_select_paged_tokens ({unit})"],
        styles=[("blue", "-")],
        ylabel=unit,
        plot_name=get_caller_name_no_ext() + "_select",
        args={},
    )

    @triton.testing.perf_report([config])
    def _run(batch, tokens_per_seq, context_len, provider):
        torch.manual_seed(42)
        s = _make_scenario(batch, tokens_per_seq, context_len)
        columns = s["pages_per_req"] * COMPRESSED_PAGE_SIZE
        rows = s["rows"]
        elem = 2  # bf16

        fn = lambda: qsa_select_paged_tokens(
            s["index_q"],
            s["compressed"],
            s["tables"],
            s["requests"],
            s["positions"],
            s["lengths"],
            BUDGET,
            RATIO,
        )

        ms = triton.testing.do_bench(fn, warmup=25, rep=100)
        if args.metric == "time":
            return ms
        mem = (
            rows * INDEX_HEADS * INDEX_DIM * elem
            + columns * INDEX_DIM * elem
            + rows * columns * 4  # fp32 logits
            + rows * min(BUDGET // RATIO, columns) * 4  # int32 selected groups
            + rows * (BUDGET + RATIO - 1) * 4  # int32 expanded output
        )
        return mem * 1e-9 / (ms * 1e-3)

    _run.run(save_path="." if args.o else None, print_data=True, show_plots=False)


# -- qsa_sparse_paged_gqa ------------------------------------------------

# (batch, tokens_per_seq, context_len)
_GQA_SHAPES = [
    # decode: single request
    (1, 1, 4096),
    (1, 1, 16384),
    (1, 1, 65536),
    # decode: concurrent requests
    (16, 1, 4096),
    (32, 1, 4096),
    (64, 1, 4096),
    (128, 1, 4096),
    # MTP verification
    (16, 3, 4096),
    (32, 3, 4096),
    # prefill
    (1, 1024, 4096),
    (1, 4096, 4096),
]


def bench_gqa(args):
    unit = "ms" if args.metric == "time" else "GB/s"

    config = triton.testing.Benchmark(
        x_names=["batch", "tokens_per_seq", "context_len"],
        x_vals=_GQA_SHAPES,
        line_arg="provider",
        line_vals=["gqa"],
        line_names=[f"qsa_sparse_paged_gqa ({unit})"],
        styles=[("green", "-")],
        ylabel=unit,
        plot_name=get_caller_name_no_ext() + "_gqa",
        args={},
    )

    @triton.testing.perf_report([config])
    def _run(batch, tokens_per_seq, context_len, provider):
        torch.manual_seed(42)
        s = _make_scenario(batch, tokens_per_seq, context_len)
        rows = s["rows"]
        elem = 2  # bf16

        selected = qsa_select_paged_tokens(
            s["index_q"],
            s["compressed"],
            s["tables"],
            s["requests"],
            s["positions"],
            s["lengths"],
            BUDGET,
            RATIO,
        )
        topk = selected.shape[1]

        fn = lambda: qsa_sparse_paged_gqa(
            s["query"],
            s["keys"],
            s["values"],
            selected,
            s["tables"],
            s["requests"],
            num_decode_requests=batch if tokens_per_seq == 1 else None,
        )

        ms = triton.testing.do_bench(fn, warmup=25, rep=100)
        if args.metric == "time":
            return ms
        mem = (
            rows * QUERY_HEADS * QUERY_DIM * elem
            + rows * topk * KV_HEADS * QUERY_DIM * elem * 2  # K + V
            + rows * topk * 4  # int32 indices
            + rows * QUERY_HEADS * QUERY_DIM * elem  # output
        )
        return mem * 1e-9 / (ms * 1e-3)

    _run.run(save_path="." if args.o else None, print_data=True, show_plots=False)


# -- main -----------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(prog="Benchmark QSA", allow_abbrev=False)
    parser.add_argument(
        "--op",
        choices=["select", "gqa", "all"],
        default="all",
        help="Which op to benchmark (default: all)",
    )
    parser.add_argument(
        "-metric",
        nargs="?",
        const="time",
        choices=["time", "bandwidth"],
        default="time",
    )
    parser.add_argument("-o", action="store_true", default=False)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(0)

    if args.op in ("select", "all"):
        bench_select(args)

    if args.op in ("gqa", "all"):
        bench_gqa(args)


if __name__ == "__main__":
    main()
