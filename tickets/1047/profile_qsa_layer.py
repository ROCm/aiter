# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""One family A live-AMD QSA layer for rocprof / HIP-graph decode replay.

Primary bar: vLLM AMD MQA + top-k/expand + sparse GQA. Oracle is not run.
Intended wrap::

    HIP_VISIBLE_DEVICES=6 rocprofv3 --kernel-trace --stats -f csv \\
      -d /tmp/qsa_rocprof_short -- python3 tickets/1047/profile_qsa_layer.py \\
      -b 1 -s 512 --graph
"""

from __future__ import annotations

import argparse
import time

import torch

from aiter import dtypes
from aiter.ops.flydsl.qsa import FAMILY_A_GQA, FAMILY_A_INDEXER
from aiter.ops.triton.attention.qsa_vllm_amd import (
    qsa_select_paged_tokens,
    qsa_sparse_paged_attention,
)
from op_tests.test_flydsl_qsa import _pack_family_a, _query_positions


def _event_us(fn, warmup: int, iters: int) -> float:
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
    return start.elapsed_time(end) * 1e3 / iters


def _build_layer(m: int, seq_len: int, page_size: int, dtype):
    idx = FAMILY_A_INDEXER
    gqa = FAMILY_A_GQA
    device = torch.device("cuda")
    n_blocks = seq_len // idx.compress_ratio
    torch.manual_seed(0)
    q_indexer = torch.randn(
        m, idx.n_heads, idx.head_dim, dtype=dtype, device=device
    ).contiguous()
    k_bar = torch.randn(n_blocks, idx.head_dim, dtype=dtype, device=device)
    q_gqa = torch.randn(
        m, gqa.n_heads, gqa.head_dim, dtype=dtype, device=device
    ).contiguous()
    k = torch.randn(seq_len, gqa.kv_heads, gqa.head_dim, dtype=dtype, device=device)
    v = torch.randn(seq_len, gqa.kv_heads, gqa.head_dim, dtype=dtype, device=device)
    qpos = _query_positions(m, seq_len, device).contiguous()
    slen = torch.full((1,), seq_len, dtype=torch.int32, device=device)
    token_to_req = torch.zeros(m, dtype=torch.int32, device=device)
    index_cache, index_table, k_cache, v_cache, kv_table = _pack_family_a(
        k_bar, k, v, page_size, device
    )
    index_cache = index_cache.contiguous()
    k_cache = k_cache.contiguous()
    v_cache = v_cache.contiguous()
    index_table = index_table.contiguous()
    kv_table = kv_table.contiguous()
    indices = torch.empty((m, idx.index_width), dtype=torch.int32, device=device)
    attn_out = torch.empty_like(q_gqa)

    def select():
        return qsa_select_paged_tokens(
            q_indexer,
            index_cache,
            index_table,
            token_to_req,
            qpos,
            slen,
            idx.token_budget,
            idx.compress_ratio,
            out=indices,
        )

    def attend():
        return qsa_sparse_paged_attention(
            q_gqa, k_cache, v_cache, indices, kv_table, token_to_req, out=attn_out
        )

    def layer():
        select()
        attend()
        return attn_out

    return layer, select, attend, n_blocks


def _try_graph(layer):
    layer()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    try:
        with torch.cuda.stream(stream):
            graph.capture_begin()
            layer()
            graph.capture_end()
    except RuntimeError as exc:
        torch.cuda.current_stream().wait_stream(stream)
        return None, str(exc)
    torch.cuda.current_stream().wait_stream(stream)
    return graph, None


def main():
    parser = argparse.ArgumentParser(description="Profile one live-AMD QSA layer")
    parser.add_argument("-b", "--batch", type=int, default=1, help="decode M")
    parser.add_argument(
        "-s",
        "--seq",
        type=int,
        nargs="*",
        default=[512, 8192, 32768],
        help="context lengths (add 131072 for 128k)",
    )
    parser.add_argument("-p", "--page-size", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument(
        "--graph",
        action="store_true",
        help="capture HIP/CUDA graph of the full layer (decode)",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="seconds to sleep after warmup (lets rocprof drop compile)",
    )
    args = parser.parse_args()
    dtype = dtypes.bf16
    if not torch.cuda.is_available():
        raise SystemExit("QSA layer profile requires a GPU")

    print(
        f"gfx={torch.cuda.get_device_name(0)} m={args.batch} graph={args.graph} "
        f"warmup={args.warmup} iters={args.iters}"
    )
    print("seq_len n_blocks select_us gqa_us layer_us graph_us graph_ok")
    for seq_len in args.seq:
        if args.batch > seq_len:
            print(f"skip m={args.batch} seq_len={seq_len}")
            continue
        layer, select, attend, n_blocks = _build_layer(
            args.batch, seq_len, args.page_size, dtype
        )
        select_us = _event_us(select, args.warmup, args.iters)
        gqa_us = _event_us(attend, args.warmup, args.iters)
        layer_us = _event_us(layer, args.warmup, args.iters)
        graph_us = float("nan")
        graph_ok = "n/a"
        if args.graph:
            graph, err = _try_graph(layer)
            if graph is None:
                graph_ok = f"fail:{err.split(chr(10), 1)[0][:80]}"
            else:
                graph_ok = "ok"
                graph_us = _event_us(graph.replay, args.warmup, args.iters)
        if args.sleep:
            time.sleep(args.sleep)
        print(
            f"{seq_len} {n_blocks} {select_us:.3f} {gqa_us:.3f} {layer_us:.3f} "
            f"{graph_us:.3f} {graph_ok}"
        )


if __name__ == "__main__":
    main()
