#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Decode-step microbench for DeepSeek-V4.1-Flash indexer and CSA/SWA attention.

One GPU stands in for one TP=4 rank. Attention heads are sharded (64 / 4 = 16);
the FP8 indexer is replicated, so every rank scores the full key set. Decode
concurrency is the query batch: one new token per sequence, speculative
decoding off. Context is the bench's 100k-token decode.

Kernels are the ones --kv_cache_dtype bf16 --index-cache-dtype fp8 selects on
gfx942: sparse_attn_v4_paged_2src, and deepgemm_fp8_paged_mqa_logits plus
top_k_per_row_decode. No checkpoint.

--index-cache empty (the default) leaves the fp8 index cache as torch.empty
hands it over, which here is zeros, so every logit is 0 and the top-k sees one
512-way tie across the whole row. --index-cache random fills it with fp8 keys
and positive scales, so the logits are real relu-weighted scores.
--topk ballot swaps top_k_per_row_decode for the ballot top-k in
csrc/kernels/topk_ballot_kernels.cu, which returns the same indices; it splits
a long row over several workgroups when that is faster. Needs ATOM for the
attention kernels.

    python op_tests/op_benchmarks/hip/v41_decode_microbench.py --topk ballot --index-cache random
"""

import argparse
import ctypes
import os
import sys

import torch
from atom.model_ops.v4_kernels import sparse_attn_v4_paged_2src

from aiter import dtypes
from aiter.ops.topk import top_k_per_row_decode
from aiter.ops.triton.attention.pa_mqa_logits import deepgemm_fp8_paged_mqa_logits
from atom.model_ops.deepseek_v41.rotary import RotaryEmbedding

DEVICE = "cuda"
HEADS = 16  # 64 attention heads / TP 4
INDEX_HEADS = 32
INDEX_DIM = 128
ATTN_DIM = 512
ROPE_DIM = 64
TOPK = 512
WINDOW = 128
INDEX_BLOCK = 16  # MQA_LOGITS_PRESHUFFLE_ROWS
SINK_SCALE = ATTN_DIM**-0.5

# 100k-token decode. visible keys = (position + 1) // ratio.
# Indexer runs on 8 layers: three ratio-2 owners, the ratio-1 candidate
# producer, and four candidate consumers (2048 blocks * 8 rows).
# Attention: 38 compressed layers (CSA top-k + window) and 2 window-only.
INDEX_CASES = (
    ("indexer ratio2 50k", 50_000, 3),
    ("indexer ratio1 100k", 100_000, 1),
    ("indexer candidates 16k", 16_384, 4),
)
ATTN_LAYERS = {"csa+swa": 38, "swa": 2}
CONCURRENCY = (1, 4, 8)

ARGS = None
_BALLOT = None
_HIP_ATTN = None
_ROPE = None


def hip_attn_op():
    global _HIP_ATTN
    if _HIP_ATTN is None:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from bench_sparse_attn_v4_gfx942 import HipSparseAttnV4

        _HIP_ATTN = HipSparseAttnV4(ARGS.attn_so)
    return _HIP_ATTN


def get_rope():
    global _ROPE
    if _ROPE is None:
        _ROPE = RotaryEmbedding(ROPE_DIM, 102400, base=160000).to(DEVICE)
    return _ROPE


def ballot_lib():
    """topk_ballot.so: the split launcher, how many parts it picks, and the
    workspace those parts need."""
    global _BALLOT
    if _BALLOT is None:
        path = ARGS.ballot_so
        if path is None:
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from bench_topk_ballot import build_so

            path = build_so()
        lib = ctypes.CDLL(path)
        lib.topk_ballot_split_launch.restype = ctypes.c_int
        lib.topk_ballot_split_launch.argtypes = (
            [ctypes.c_void_p] * 4 + [ctypes.c_int] * 5 + [ctypes.c_void_p] * 2
        )
        lib.topk_ballot_parts.restype = ctypes.c_int
        lib.topk_ballot_parts.argtypes = [ctypes.c_int, ctypes.c_int]
        lib.topk_ballot_workspace_bytes.restype = ctypes.c_longlong
        lib.topk_ballot_workspace_bytes.argtypes = [ctypes.c_int, ctypes.c_int]
        _BALLOT = lib
    return _BALLOT


def _events(fn, iters):
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iters * 1000.0  # microseconds


def bench(fn, warmup=5, iters=30):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    graph_us = None
    try:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            fn()
        torch.cuda.synchronize()
        graph_us = _events(graph.replay, iters)
    # Graph capture is what the server uses; keep the eager time if it refuses.
    except Exception as exc:  # noqa: BLE001 -- any refusal, reported in the table
        graph_us = f"capture failed: {type(exc).__name__}: {exc}"
    eager_us = _events(fn, iters)
    return eager_us, graph_us


def make_indexer(batch, n_keys):
    assert n_keys % INDEX_BLOCK == 0
    # The preshuffle logits kernel prefetches past the last live block. A table
    # sized exactly to the context faults; the real block table is wider than
    # the live length. The extra columns are not part of context_lens.
    pad_blocks = 64
    n_blocks = n_keys // INDEX_BLOCK
    q = torch.randn(
        batch, 1, INDEX_HEADS, INDEX_DIM, device=DEVICE, dtype=torch.bfloat16
    ).to(dtypes.fp8)
    weights = torch.rand(batch, INDEX_HEADS, device=DEVICE, dtype=torch.float32)
    cache = torch.empty(
        n_blocks + pad_blocks,
        INDEX_BLOCK,
        1,
        INDEX_DIM + 4,
        device=DEVICE,
        dtype=torch.uint8,
    )
    if ARGS.index_cache == "random":
        # A page is INDEX_BLOCK rows of fp8 keys, then their INDEX_BLOCK fp32 scales.
        page = cache.view(n_blocks + pad_blocks, -1)
        split = INDEX_BLOCK * INDEX_DIM
        page[:, :split] = (
            torch.randn(page.shape[0], split, device=DEVICE)
            .to(dtypes.fp8)
            .view(torch.uint8)
        )
        scales = 0.5 + torch.rand(page.shape[0], INDEX_BLOCK, device=DEVICE)
        page[:, split:] = scales.view(torch.uint8)
    table = torch.zeros(batch, n_blocks + pad_blocks, device=DEVICE, dtype=torch.int32)
    table[:, :n_blocks] = torch.arange(n_blocks, device=DEVICE)
    lens = torch.full((batch,), n_keys, device=DEVICE, dtype=torch.int32)
    logits = torch.empty(batch, n_keys, device=DEVICE, dtype=torch.float32)
    selected = torch.empty(batch, TOPK, device=DEVICE, dtype=torch.int32)
    parts, workspace = 1, None
    if ARGS.topk == "ballot":
        # A split row needs a zeroed workspace, allocated outside graph capture.
        lib = ballot_lib()
        parts = lib.topk_ballot_parts(batch, logits.stride(0))
        if parts > 1:
            size = lib.topk_ballot_workspace_bytes(batch, parts)
            workspace = torch.zeros(size, device=DEVICE, dtype=torch.uint8)

    def logits_only():
        deepgemm_fp8_paged_mqa_logits(
            q,
            cache,
            weights,
            logits,
            lens,
            table,
            n_keys,
            Preshuffle=True,
            KVBlockSize=INDEX_BLOCK,
        )

    def topk_only():
        if ARGS.topk == "ballot":
            rc = ballot_lib().topk_ballot_split_launch(
                logits.data_ptr(),
                lens.data_ptr(),
                selected.data_ptr(),
                None,
                batch,
                logits.stride(0),
                1,
                TOPK,
                parts,
                None if workspace is None else workspace.data_ptr(),
                torch.cuda.current_stream().cuda_stream,
            )
            if rc:
                raise RuntimeError(f"topk_ballot_split_launch returned {rc}")
            return
        top_k_per_row_decode(
            logits,
            1,
            lens,
            selected,
            batch,
            logits.stride(0),
            logits.stride(1),
            k=TOPK,
            stable=True,
        )

    def both():
        logits_only()
        topk_only()

    return both, logits_only, topk_only


def make_attention(batch, csa, swa):
    q = torch.randn(batch, HEADS, ATTN_DIM, device=DEVICE, dtype=torch.bfloat16)
    out = torch.empty_like(q)
    sink = torch.zeros(HEADS, device=DEVICE, dtype=torch.float32)

    def source(rows_per_query):
        if rows_per_query == 0:
            pool = torch.empty(1, ATTN_DIM, device=DEVICE, dtype=torch.bfloat16)
            indices = torch.empty(0, device=DEVICE, dtype=torch.int32)
            indptr = torch.zeros(batch + 1, device=DEVICE, dtype=torch.int32)
            return pool, indices, indptr
        pool = torch.randn(
            batch * rows_per_query, ATTN_DIM, device=DEVICE, dtype=torch.bfloat16
        )
        indices = torch.arange(pool.shape[0], device=DEVICE, dtype=torch.int32)
        indptr = (
            torch.arange(batch + 1, device=DEVICE, dtype=torch.int32) * rows_per_query
        )
        return pool, indices, indptr

    csa_pool, csa_idx, csa_ptr = source(csa)
    swa_pool, swa_idx, swa_ptr = source(swa)
    positions = torch.full((batch,), 99999, device=DEVICE, dtype=torch.int64)
    rope = get_rope() if ARGS.fuse_irope else None

    def run():
        if ARGS.attn == "hip":
            hip_attn_op()(
                q,
                csa_pool,
                csa_idx,
                csa_ptr,
                swa_pool,
                swa_idx,
                swa_ptr,
                sink,
                SINK_SCALE,
                out=out,
                cos_cache=None if rope is None else rope.cos_cache,
                sin_cache=None if rope is None else rope.sin_cache,
                positions=None if rope is None else positions,
                rope_dim=ROPE_DIM if rope is not None else 0,
            )
            return
        sparse_attn_v4_paged_2src(
            q,
            csa_pool,
            csa_idx,
            csa_ptr,
            swa_pool,
            swa_idx,
            swa_ptr,
            sink,
            SINK_SCALE,
            out=out,
        )
        if rope is not None:
            rope(out.unsqueeze(0), positions, inverse=True)

    return run


def us(value):
    return value if isinstance(value, str) else f"{value:8.1f}"


def main():
    global ARGS
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--topk", choices=("aiter", "ballot"), default="aiter")
    ap.add_argument("--attn", choices=("triton", "hip"), default="triton")
    ap.add_argument(
        "--fuse-irope",
        action="store_true",
        help="include inverse_rope (unfused for triton, fused in epilogue for hip)",
    )
    ap.add_argument("--index-cache", choices=("empty", "random"), default="empty")
    ap.add_argument(
        "--ballot-so", default=None, help="a prebuilt topk_ballot.so; built if absent"
    )
    ap.add_argument(
        "--attn-so", default=None, help="a prebuilt sparse_attn_v4_gfx942.so; built if absent"
    )
    ARGS = ap.parse_args()

    props = torch.cuda.get_device_properties(0)
    arch = props.gcnArchName.split(":")[0]
    print(
        f"device {props.name} {arch}  CUs {props.multi_processor_count}  "
        f"TP4 attention heads/rank {HEADS}  indexer heads {INDEX_HEADS}"
    )
    print(
        f"top-k {ARGS.topk}  attn {ARGS.attn}  fuse-irope {ARGS.fuse_irope}  "
        f"index cache {ARGS.index_cache}"
    )
    print(
        "times are microseconds per launch, one rank. graph is the server's decode path."
    )
    print(
        f"{'case':<28} {'con':>4} {'eager':>10} {'graph':>10} "
        f"{'x layers':>8} {'forward':>10}"
    )
    for batch in CONCURRENCY:
        for name, n_keys, layers in INDEX_CASES:
            both, logits_only, topk_only = make_indexer(batch, n_keys)
            for label, fn in (
                (name, both),
                (name + " logits", logits_only),
                (name + " topk", topk_only),
            ):
                eager, graph = bench(fn)
                counted = layers if label == name else 0
                forward = ""
                if counted and isinstance(graph, float):
                    forward = f"{graph * counted:8.1f}"
                elif counted and isinstance(eager, float):
                    forward = f"{eager * counted:8.1f}"
                print(
                    f"{label:<28} {batch:4d} {us(eager):>10} {us(graph):>10} "
                    f"{counted:8d} {forward:>10}"
                )
        for kind, csa, swa in (("csa+swa", TOPK, WINDOW), ("swa", 0, WINDOW)):
            eager, graph = bench(make_attention(batch, csa, swa))
            layers = ATTN_LAYERS[kind]
            base = graph if isinstance(graph, float) else eager
            forward = f"{base * layers:8.1f}" if isinstance(base, float) else ""
            print(
                f"{'attn ' + kind:<28} {batch:4d} {us(eager):>10} {us(graph):>10} "
                f"{layers:8d} {forward:>10}"
            )


if __name__ == "__main__":
    main()
