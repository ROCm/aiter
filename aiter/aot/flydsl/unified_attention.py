#!/usr/bin/env python3

# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""AOT pre-compilation for the gfx950 FlyDSL BF16 unified attention (MiMo).

``unified_attention_flydsl`` derives its kernels from the call's shapes: the
query length, power-of-two buckets of the batch and of its query blocks, the
planner's block size, whether CTAs pair, and the reducer's partition count,
which follows ``max_seqlen_k`` in 256-token steps up to 64 partitions and in
powers of two beyond. A serving process therefore JIT-compiles a new kernel
whenever traffic reaches a new combination. This module replays the router
itself on FakeTensors inside :func:`aiter.aot.flydsl.dry_run.dry_run` (see
``replay.py``) for each geometry in ``DEFAULT_SHAPES``, over a grid of batch
sizes, query token counts and KV lengths, so the heuristics stay the router's
own and changing them needs no AOT change.

A job is one slice of that grid; its ``m`` indexes the slice's points
(``_points``). ``SCAN_ARCHS`` has every point identified without compiling
and one point per new set of kernels replayed for real.

Usage:
    python -m aiter.aot.flydsl.unified_attention                 # DEFAULT_SHAPES
    python -m aiter.aot.flydsl.unified_attention --shape 64:1:127:f32
    python -m aiter.aot.flydsl.unified_attention --check
"""

from __future__ import annotations

import argparse
import functools
import sys

from aiter.aot.flydsl import replay as _replay
from aiter.aot.flydsl.common import dedupe_jobs

LABEL = "UNIFIED_ATTENTION"
# No tuned table: the jobs come from DEFAULT_SHAPES or --shape.
DEFAULT_CSVS: list[str] = []
AOT_ARCH_DEFAULT = "gfx950"
SCAN_ARCHS = ("gfx950",)
# The optimized varlen specialization needs 256 CUs (MI350X, MI355X).
_CU_NUM = 256
_HEAD_DIM, _VALUE_DIM, _GQA = 192, 128, 16
_SINK_DTYPES = ("f32", "bf16", "f16")
# Geometries to pre-compile, per model: (page size, KV heads per rank, left
# window or 0 for full attention, sink dtype or None).
DEFAULT_SHAPES = {
    "mimo_v2": [
        (64, 1, 0, None),
        (64, 1, 127, "f32"),
        (128, 1, 0, None),
        (128, 1, 127, "f32"),
    ],
}
# The largest query token count of a varlen call (the engine's batched-token
# budget).
DEFAULT_MAX_TOKENS = 32768
# KV lengths: every partition count the reducer specializes exactly (1..64 of
# 256 tokens), its power-of-two rungs beyond, and the optimized varlen
# specialization's 4096-token floor.
_CONTEXTS = (
    *(256 * p for p in range(1, 65)),
    4095,
    16385,
    32769,
    65536,
    262144,
)


def _batches(kv_heads: int, dense: bool) -> tuple[int, ...]:
    """Batch sizes to replay. ``dense``: every batch up to 64 (the reducer's
    group count follows batch x query rows), else 1..3. Then each power of two
    and its successor up to the router's limit, and the batch from which the
    plan gives each sequence one slot."""
    from aiter.ops.flydsl.unified_attention_bf16 import _MAX_BATCH

    sizes = set(range(1, 65 if dense else 4))
    for bit in range(1, _MAX_BATCH.bit_length()):
        sizes |= {1 << bit, (1 << bit) + 1}
    full = -(-2 * _CU_NUM // kv_heads)
    sizes |= {full - 1, full}
    return tuple(sorted(s for s in sizes if s <= _MAX_BATCH))


def _varlen_tokens(num_seqs: int, max_tokens: int) -> list[int]:
    """Query token counts at which a varlen batch's query blocks (one per 16
    tokens, plus one per sequence) cross each power of two, on either side.
    A count must not make the batch uniform (all sequences 1..8 tokens)."""
    tokens = set()
    for bit in range(17):
        edge = 16 * ((1 << bit) - num_seqs)
        tokens |= {edge, edge + 1}
    tokens.add(max_tokens)
    return sorted(
        t for t in tokens if num_seqs < t <= max_tokens and (num_seqs > 1 or t > 8)
    )


def _contexts(window: int, min_context: int) -> tuple[int, ...]:
    if window:
        # A window launch is independent of the KV length.
        return (min_context,)
    return tuple(sorted({c for c in _CONTEXTS if c >= min_context} | {min_context}))


@functools.cache
def _points(
    route: str,
    window: int,
    query_length: int,
    batches: tuple,
    tokens: int | None,
    max_tokens: int,
) -> tuple[tuple[int, int, int, int], ...]:
    """(num_seqs, num_tokens, max_seqlen_q, max_seqlen_k) of each point.
    Varlen points take every ``_varlen_tokens`` count, or only ``tokens``."""
    points = []
    for s in batches:
        if route == "uniform":
            for c in _contexts(window, query_length):
                points.append((s, s * query_length, query_length, c))
            continue
        for t in _varlen_tokens(s, max_tokens):
            if tokens is not None and t != tokens:
                continue
            # One long sequence, the rest a token each.
            q = t - s + 1
            for c in _contexts(window, q):
                points.append((s, t, q, c))
    return tuple(points)


def _job_points(job: dict) -> tuple:
    return _points(
        job["route"],
        job["window"],
        job["query_length"],
        job["batches"],
        job["tokens"],
        job["max_tokens"],
    )


def parse_shape(spec: str) -> tuple[int, int, int, str | None]:
    """``"page:kv_heads[:window[:sink]]"`` -> ``(page, kv_heads, window, sink)``."""
    parts = spec.strip().split(":")
    page, kv_heads = int(parts[0]), int(parts[1])
    window = int(parts[2]) if len(parts) > 2 else 0
    sink = parts[3] if len(parts) > 3 and parts[3] not in ("", "none") else None
    if sink is not None and sink not in _SINK_DTYPES:
        raise ValueError(f"sink dtype must be one of {_SINK_DTYPES}, got {sink!r}")
    return page, kv_heads, window, sink


def jobs_for_shape(
    page: int,
    kv_heads: int,
    window: int,
    sink: str | None,
    scale: float,
    max_tokens: int,
) -> list[dict]:
    """Jobs that together cover the router's uniform (1..8 queries per
    sequence) and varlen routes for one geometry. Without a window, a uniform
    job is one query length and batch over every KV length, a varlen job one
    batch and token count over the KV lengths; a window launch ignores the KV
    length, so its jobs span batches and token counts instead. Varlen full
    attention with sinks is declined by the router and has no jobs."""
    from aiter.ops.flydsl.unified_attention_bf16 import _MAX_QUERY_LENGTH

    base = {
        "gfx": AOT_ARCH_DEFAULT,
        "cu_num": _CU_NUM,
        "page": page,
        "kv_heads": kv_heads,
        "window": window,
        "sink": sink,
        "scale": scale,
        "max_tokens": max_tokens,
        "m_low": 0,
    }
    name = (
        f"ua_bf16_p{page}_hkv{kv_heads}_"
        f"{f'sw{window}' if window else 'full'}_{sink or 'nosink'}"
    )
    slices = []
    dense = _batches(kv_heads, dense=True)
    for q in range(1, _MAX_QUERY_LENGTH + 1):
        if window:
            slices.append(("uniform", q, dense, None))
        else:
            slices += [("uniform", q, (s,), None) for s in dense]
    if window or sink is None:
        sparse = _batches(kv_heads, dense=False)
        if window:
            slices.append(("varlen", 0, sparse, None))
        else:
            slices += [
                ("varlen", 0, (s,), t)
                for s in sparse
                for t in _varlen_tokens(s, max_tokens)
            ]
    jobs = []
    for route, q, batches, tokens in slices:
        job = {
            **base,
            "kernel_name": f"{name}_{route}{q or ''}",
            "route": route,
            "query_length": q,
            "batches": batches,
            "tokens": tokens,
        }
        points = _job_points(job)
        if points:
            # The job replays its last point, then scans the others.
            jobs.append({**job, "m": len(points) - 1})
    return jobs


def shape_jobs(
    shapes: dict[str, list] = DEFAULT_SHAPES,
    scale: float = _HEAD_DIM**-0.5,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> list[dict]:
    jobs = []
    for entries in shapes.values():
        for page, kv_heads, window, sink in entries:
            jobs += jobs_for_shape(page, kv_heads, window, sink, scale, max_tokens)
    return dedupe_jobs(jobs)


def replay(job: dict) -> None:
    import torch

    from aiter.ops.flydsl.unified_attention_bf16 import unified_attention_flydsl

    num_seqs, num_tokens, max_q, max_k = _job_points(job)[job["m"]]
    page, kv_heads = job["page"], job["kv_heads"]
    heads = _GQA * kv_heads
    pages = -(-max_k // page)

    def empty(*shape, dtype=torch.bfloat16):
        return torch.empty(shape, dtype=dtype, device="cuda")

    sinks = None
    if job["sink"] is not None:
        sinks = empty(
            heads,
            dtype={
                "f32": torch.float32,
                "bf16": torch.bfloat16,
                "f16": torch.float16,
            }[job["sink"]],
        )
    window = job["window"]
    unified_attention_flydsl(
        empty(num_tokens, heads, _HEAD_DIM),
        empty(pages + 1, page, kv_heads, _HEAD_DIM),
        empty(pages + 1, page, kv_heads, _VALUE_DIM),
        empty(num_tokens, heads, _VALUE_DIM),
        empty(num_seqs + 1, dtype=torch.int32),
        max_q,
        empty(num_seqs, dtype=torch.int32),
        max_k,
        job["scale"],
        True,
        (window, 0) if window else (-1, -1),
        empty(num_seqs, pages, dtype=torch.int32),
        0,
        None,
        None,
        None,
        sinks=sinks,
    )


def describe(job: dict) -> str:
    num_seqs, num_tokens, max_q, max_k = _job_points(job)[job["m"]]
    return (
        f"{job['kernel_name']} seqs={num_seqs} tokens={num_tokens} "
        f"max_q={max_q} max_k={max_k}"
    )


def collect_jobs(_csv_paths: list[str], args: argparse.Namespace | None) -> list:
    scale = _HEAD_DIM**-0.5 if args is None else args.scale
    max_tokens = DEFAULT_MAX_TOKENS if args is None else args.max_tokens
    if args is None or not args.shape:
        return shape_jobs(scale=scale, max_tokens=max_tokens)
    try:
        shapes = {"cli": [parse_shape(spec) for spec in args.shape]}
    except ValueError as error:
        print(f"Error: {error}")
        sys.exit(1)
    return shape_jobs(shapes, scale, max_tokens)


def configure(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--shape",
        action="append",
        default=[],
        help="page:kv_heads[:window[:sink]], repeatable; replaces DEFAULT_SHAPES",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=_HEAD_DIM**-0.5,
        help="softmax scale the model passes (compiled into the kernels)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help="largest query token count of a varlen call",
    )


def main() -> None:
    _replay.main(
        "aiter.aot.flydsl.unified_attention",
        "AOT pre-compile the gfx950 FlyDSL BF16 unified attention",
    )


if __name__ == "__main__":
    main()
