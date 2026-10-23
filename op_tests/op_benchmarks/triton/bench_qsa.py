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

import functools

import torch
import triton

from aiter.ops.qsa import qsa_select_paged_tokens
from aiter.ops.triton.attention.qsa import qsa_sparse_paged_gqa
from op_tests.op_benchmarks.triton.utils.argparse import (
    get_do_bench_graph_kwargs,
    get_parser,
    get_rot_bufs_kwargs,
    override_parser_action,
)
from op_tests.op_benchmarks.triton.utils.benchmark_utils import (
    do_bench_aiter_triton,
    generate_rotating_buffers_pool,
    get_caller_name_no_ext,
)

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

    positions = torch.arange(rows, dtype=torch.int32, device="cuda")
    for b in range(batch):
        start = b * tokens_per_seq
        positions[start : start + tokens_per_seq] = torch.arange(
            context_len - tokens_per_seq,
            context_len,
            dtype=torch.int32,
            device="cuda",
        )

    tables = torch.randint(
        0, total_pages, (batch, pages_per_req), dtype=torch.int32, device="cuda"
    )
    requests = torch.arange(batch, dtype=torch.int32, device="cuda").repeat_interleave(
        tokens_per_seq
    )
    lengths = torch.full((batch,), context_len, dtype=torch.int32, device="cuda")

    index_q = torch.randn(
        rows, INDEX_HEADS, INDEX_DIM, dtype=torch.bfloat16, device="cuda"
    )
    compressed = torch.randn(
        total_pages,
        COMPRESSED_PAGE_SIZE,
        1,
        INDEX_DIM,
        dtype=torch.bfloat16,
        device="cuda",
    )
    query = torch.randn(
        rows, QUERY_HEADS, QUERY_DIM, dtype=torch.bfloat16, device="cuda"
    )
    keys = torch.randn(
        total_pages,
        BLOCK_SIZE,
        KV_HEADS,
        QUERY_DIM,
        dtype=torch.bfloat16,
        device="cuda",
    )
    values = torch.randn_like(keys)
    return (
        positions,
        tables,
        requests,
        lengths,
        index_q,
        compressed,
        query,
        keys,
        values,
        rows,
        total_pages,
        pages_per_req,
    )


_SCENARIO_POOL_FIELDS = [
    "positions",
    "tables",
    "requests",
    "lengths",
    "index_q",
    "compressed",
    "query",
    "keys",
    "values",
    "rows",
    "total_pages",
    "pages_per_req",
]


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


def _select_kernel_fn(index_q, compressed, tables, requests, positions, lengths, **_):
    qsa_select_paged_tokens(
        index_q, compressed, tables, requests, positions, lengths, BUDGET, RATIO
    )


def bench_select(args):
    unit = "ms" if args.metric == "time" else "GB/s"
    kwargs_do_bench_graph = get_do_bench_graph_kwargs(args)
    kwargs_rot_bufs = get_rot_bufs_kwargs(args)

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
        pool_gen = functools.partial(_make_scenario, batch, tokens_per_seq, context_len)
        rot_bufs = generate_rotating_buffers_pool(
            [pool_gen], [_SCENARIO_POOL_FIELDS], **kwargs_rot_bufs
        )
        columns = rot_bufs[0]["pages_per_req"] * COMPRESSED_PAGE_SIZE
        rows = rot_bufs[0]["rows"]
        elem = 2  # bf16

        ms = do_bench_aiter_triton(_select_kernel_fn, rot_bufs, **kwargs_do_bench_graph)
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


def _generate_gqa_pool_entry(batch, tokens_per_seq, context_len):
    """One scenario plus its (untimed) selected-token indices.

    qsa_sparse_paged_gqa needs qsa_select_paged_tokens' output as input, so
    each pool entry primes it the same way moe2 is primed by moe1's output in
    the MoE benchmarks: one untimed call ahead of time, per entry.
    """
    (
        positions,
        tables,
        requests,
        lengths,
        index_q,
        compressed,
        query,
        keys,
        values,
        rows,
        total_pages,
        pages_per_req,
    ) = _make_scenario(batch, tokens_per_seq, context_len)
    selected = qsa_select_paged_tokens(
        index_q, compressed, tables, requests, positions, lengths, BUDGET, RATIO
    )
    return (
        query,
        keys,
        values,
        selected,
        tables,
        requests,
        rows,
        total_pages,
        pages_per_req,
    )


_GQA_POOL_FIELDS = [
    "query",
    "keys",
    "values",
    "selected",
    "tables",
    "requests",
    "rows",
    "total_pages",
    "pages_per_req",
]


def _make_gqa_kernel_fn(batch, tokens_per_seq):
    def kernel_fn(query, keys, values, selected, tables, requests, **_):
        qsa_sparse_paged_gqa(
            query,
            keys,
            values,
            selected,
            tables,
            requests,
            num_decode_requests=batch if tokens_per_seq == 1 else None,
        )

    return kernel_fn


def bench_gqa(args):
    unit = "ms" if args.metric == "time" else "GB/s"
    kwargs_do_bench_graph = get_do_bench_graph_kwargs(args)
    kwargs_rot_bufs = get_rot_bufs_kwargs(args)

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
        pool_gen = functools.partial(
            _generate_gqa_pool_entry, batch, tokens_per_seq, context_len
        )
        rot_bufs = generate_rotating_buffers_pool(
            [pool_gen], [_GQA_POOL_FIELDS], **kwargs_rot_bufs
        )
        rows = rot_bufs[0]["rows"]
        topk = rot_bufs[0]["selected"].shape[1]
        elem = 2  # bf16

        kernel_fn = _make_gqa_kernel_fn(batch, tokens_per_seq)
        ms = do_bench_aiter_triton(kernel_fn, rot_bufs, **kwargs_do_bench_graph)
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
    parser = get_parser(kernel_name="QSA")
    override_parser_action(
        parser, action="--metric", choices=["time", "bandwidth"], default="time"
    )
    parser.add_argument(
        "--op",
        choices=["select", "gqa", "all"],
        default="all",
        help="Which op to benchmark (default: all)",
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
