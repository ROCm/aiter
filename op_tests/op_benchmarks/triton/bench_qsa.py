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
    python op_tests/op_benchmarks/triton/bench_qsa.py --op logits \
        --index-dtype fp4 --stacked-decode
    python op_tests/op_benchmarks/triton/bench_qsa.py --op gqa --kv-dtype fp8
"""

import argparse

import torch
import triton

from aiter.ops.qsa import (
    qsa_cache_row_layout,
    qsa_paged_mqa_logits,
    qsa_select_paged_tokens,
    qsa_sparse_paged_gqa,
    qsa_store_quantized_rows,
)
from op_tests.op_benchmarks.triton.utils.benchmark_utils import get_caller_name_no_ext

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


def _quantize(values, fmt, second=None):
    """Store paged `values` `[pages, page, heads, dim]` (and optionally a
    second tensor of the same shape) in `fmt`; returns (caches, scales)."""
    if fmt == "bf16":
        return (values, second), (None, None)
    pages, page, heads, dim = values.shape
    vdim, vdtype, tail, sdtype = qsa_cache_row_layout(fmt, dim)
    slots = torch.arange(pages * page, device="cuda")
    caches, scales = [], []
    for _ in range(1 if second is None else 2):
        caches.append(
            torch.empty(pages, page, heads, vdim, dtype=vdtype, device="cuda")
        )
        scales.append(
            torch.empty(pages, page, heads, *tail, dtype=sdtype, device="cuda")
        )
    if second is None:
        qsa_store_quantized_rows(
            caches[0], scales[0], slots, values.view(-1, heads, dim)
        )
        return (caches[0], None), (scales[0], None)
    qsa_store_quantized_rows(
        caches[0],
        scales[0],
        slots,
        values.view(-1, heads, dim),
        caches[1],
        scales[1],
        second.view(-1, heads, dim),
    )
    return tuple(caches), tuple(scales)


def _row_bytes(fmt, dim):
    """Bytes of one cached row including its scale."""
    vdim, vdtype, tail, sdtype = qsa_cache_row_layout(fmt, dim)
    size = vdim * vdtype.itemsize
    if tail is not None:
        n = 1
        for t in tail:
            n *= t
        size += n * sdtype.itemsize
    return size


def _make_scenario(batch, tokens_per_seq, context_len, index_dtype, kv_dtype):
    """Build tensors for a multi-request paged QSA scenario.

    Args:
        batch: number of requests
        tokens_per_seq: tokens per request (1 for decode, >1 for prefill/verify)
        context_len: context length per request
        index_dtype: compressed index-key cache format (bf16 / fp8 / fp4)
        kv_dtype: K/V cache format (bf16 / fp8)
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
    (compressed, _), (compressed_scale, _) = _quantize(compressed, index_dtype)
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
    (keys, values), (k_scale, v_scale) = _quantize(keys, kv_dtype, values)
    return {
        "positions": positions,
        "tables": tables,
        "requests": requests,
        "lengths": lengths,
        "index_q": index_q,
        "compressed": compressed,
        "compressed_scale": compressed_scale,
        "query": query,
        "keys": keys,
        "values": values,
        "k_scale": k_scale,
        "v_scale": v_scale,
        "rows": rows,
        "total_pages": total_pages,
        "pages_per_req": pages_per_req,
    }


def _scenario(args, batch, tokens_per_seq, context_len):
    return _make_scenario(
        batch, tokens_per_seq, context_len, args.index_dtype, args.kv_dtype
    )


def _label(args, name):
    parts = []
    if name in ("select", "logits"):
        parts.append(f"index {args.index_dtype}")
        if args.stacked_decode:
            parts.append("stacked")
    if name == "gqa":
        parts.append(f"kv {args.kv_dtype}")
    return f"{name}, {', '.join(parts)}"


# -- qsa_paged_mqa_logits / qsa_select_paged_tokens ------------------------

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
    (9, 4, 8192),
    # prefill
    (1, 1024, 4096),
    (1, 4096, 4096),
]


def bench_select(args, scoring_only=False):
    unit = "ms" if args.metric == "time" else "GB/s"
    name = "logits" if scoring_only else "select"
    fn_name = "qsa_paged_mqa_logits" if scoring_only else "qsa_select_paged_tokens"

    config = triton.testing.Benchmark(
        x_names=["batch", "tokens_per_seq", "context_len"],
        x_vals=_SELECT_SHAPES,
        line_arg="provider",
        line_vals=[name],
        line_names=[f"{fn_name} [{_label(args, name)}] ({unit})"],
        styles=[("blue", "-")],
        ylabel=unit,
        plot_name=get_caller_name_no_ext() + "_" + name,
        args={},
    )

    @triton.testing.perf_report([config])
    def _run(batch, tokens_per_seq, context_len, provider):
        torch.manual_seed(42)
        s = _scenario(args, batch, tokens_per_seq, context_len)
        columns = s["pages_per_req"] * COMPRESSED_PAGE_SIZE
        rows = s["rows"]
        elem = 2  # bf16
        common = (
            s["index_q"],
            s["compressed"],
            s["tables"],
            s["requests"],
            s["positions"],
            s["lengths"],
        )
        kwargs = {
            "compressed_k_scale": s["compressed_scale"],
            "stacked_decode": args.stacked_decode,
        }

        if scoring_only:
            fn = lambda: qsa_paged_mqa_logits(*common, RATIO, **kwargs)
        else:
            fn = lambda: qsa_select_paged_tokens(*common, BUDGET, RATIO, **kwargs)

        ms = triton.testing.do_bench(fn, warmup=25, rep=100)
        if args.metric == "time":
            return ms
        mem = (
            rows * INDEX_HEADS * INDEX_DIM * elem
            + columns * _row_bytes(args.index_dtype, INDEX_DIM)
            + rows * columns * 4  # fp32 logits
        )
        if not scoring_only:
            mem += (
                rows * min(BUDGET // RATIO, columns) * 4  # int32 selected groups
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
    (9, 4, 8192),
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
        line_names=[f"qsa_sparse_paged_gqa [{_label(args, 'gqa')}] ({unit})"],
        styles=[("green", "-")],
        ylabel=unit,
        plot_name=get_caller_name_no_ext() + "_gqa",
        args={},
    )

    @triton.testing.perf_report([config])
    def _run(batch, tokens_per_seq, context_len, provider):
        torch.manual_seed(42)
        s = _scenario(args, batch, tokens_per_seq, context_len)
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
            compressed_k_scale=s["compressed_scale"],
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
            k_scale=s["k_scale"],
            v_scale=s["v_scale"],
        )

        ms = triton.testing.do_bench(fn, warmup=25, rep=100)
        if args.metric == "time":
            return ms
        kv_row = _row_bytes(args.kv_dtype, QUERY_DIM)
        mem = (
            rows * QUERY_HEADS * QUERY_DIM * elem
            + rows * topk * KV_HEADS * kv_row * 2  # K + V
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
        choices=["select", "logits", "gqa", "all"],
        default="all",
        help="Which op to benchmark (default: all = select and gqa)",
    )
    parser.add_argument(
        "--index-dtype",
        choices=["bf16", "fp8", "fp4"],
        default="bf16",
        help="Compressed index-key cache format for select / logits",
    )
    parser.add_argument(
        "--kv-dtype",
        choices=["bf16", "fp8"],
        default="bf16",
        help="K/V cache format for gqa",
    )
    parser.add_argument(
        "--stacked-decode",
        action="store_true",
        default=False,
        help="Score decode-sized calls with the stacked-head MFMA scorer",
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

    if args.op == "logits":
        bench_select(args, scoring_only=True)

    if args.op in ("gqa", "all"):
        bench_gqa(args)


if __name__ == "__main__":
    main()
