#!/usr/bin/env python3

# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""AOT pre-compilation for the gfx942 FlyDSL fp8 unified-attention kernels.

Replays ``unified_attention`` on FakeTensors inside ``dry_run`` (see
``replay.py``) with each call in ``CALLS``, per layer shape in
``DEFAULT_SHAPES``, served page size and canonical K/V layout (two contiguous
caches, or vLLM's views of one): the all-decode batch reaches the decode-only
build and its combine, the lone chunk over a prefix the unified build and its
combine. Other K/V strides, such as a padded cache, compile on first use, and
under ``FLYDSL_RUNTIME_RUN_ONLY=1`` the gate sends them to Triton.

Usage:
    python -m aiter.aot.flydsl.unified_attention
    python -m aiter.aot.flydsl.unified_attention --check
"""

from __future__ import annotations

import argparse
import itertools

from aiter.aot.flydsl import replay as _replay
from aiter.aot.flydsl.common import dedupe_jobs

LABEL = "UNIFIED_ATTENTION"
# No tuned table: the jobs come from DEFAULT_SHAPES.
DEFAULT_CSVS: list[str] = []
AOT_ARCH_DEFAULT = "gfx942"
# MI300X. The split planners read the CU count; the compile keys do not.
_CU_NUM = 304
# Layer shapes per model: (num_heads, num_kv_heads, head_dim, window keys or
# None).
DEFAULT_SHAPES = {
    "gemma4_31b_tp1": [(32, 4, 512, None), (32, 16, 256, 1024)],
}
# (query lengths, key lengths) of each replayed call. Both are served, not ceded
# to Triton, and plan more than one split, so each reaches its combine as well.
CALLS = {
    "decode": ([1] * 4, [1024] * 4),
    "unified": ([64], [1024]),
}


def _fake_kv(layout, page_size, num_kv_heads, head_dim, blocks=8, device=None):
    """fp8 K and V in one of the adapter's ``KV_LAYOUTS``."""
    import torch

    fp8 = torch.float8_e4m3fnuz
    if layout == "plain":
        shape = (blocks, page_size, num_kv_heads, head_dim)
        return (
            torch.empty(shape, dtype=fp8, device=device),
            torch.empty(shape, dtype=fp8, device=device),
        )
    cache = torch.empty(
        (blocks, num_kv_heads, page_size, 2 * head_dim), dtype=fp8, device=device
    )
    kv = cache.transpose(1, 2)
    return kv[..., :head_dim], kv[..., head_dim:]


def default_jobs(shapes: dict[str, list] = DEFAULT_SHAPES) -> list[dict]:
    """A job per layer shape, served page size, canonical K/V layout and call."""
    from aiter.ops.flydsl.unified_attention_kernels import _PAGE_SIZES, KV_LAYOUTS

    jobs = []
    for entries in shapes.values():
        for num_heads, num_kv_heads, head_dim, window in entries:
            for page_size, layout, call in itertools.product(
                _PAGE_SIZES, KV_LAYOUTS, CALLS
            ):
                jobs.append(
                    {
                        "kernel_name": (
                            f"flydsl_unified_attn_gfx942_h{num_heads}"
                            f"_hkv{num_kv_heads}_d{head_dim}_w{window or 0}"
                            f"_p{page_size}_{layout}_{call}"
                        ),
                        "gfx": AOT_ARCH_DEFAULT,
                        "cu_num": _CU_NUM,
                        "num_heads": num_heads,
                        "num_kv_heads": num_kv_heads,
                        "head_dim": head_dim,
                        "window": window,
                        "page_size": page_size,
                        "layout": layout,
                        "call": call,
                    }
                )
    return dedupe_jobs(jobs)


def replay(job: dict) -> None:
    import torch

    from aiter.ops.unified_attention import unified_attention

    def empty(*shape, dtype):
        return torch.empty(shape, dtype=dtype, device="cuda")

    query_lens, kv_lens = CALLS[job["call"]]
    num_heads, head_dim, window = job["num_heads"], job["head_dim"], job["window"]
    page_size = job["page_size"]
    pages = [-(-n // page_size) for n in kv_lens]
    k, v = _fake_kv(
        job["layout"], page_size, job["num_kv_heads"], head_dim, sum(pages), "cuda"
    )
    tokens = sum(query_lens)
    # backend="flydsl" raises where the router would hand the call to Triton,
    # so a call that stops reaching the kernel fails the build.
    unified_attention(
        q=empty(tokens, num_heads, head_dim, dtype=torch.float8_e4m3fnuz),
        k=k,
        v=v,
        out=empty(tokens, num_heads, head_dim, dtype=torch.bfloat16),
        cu_seqlens_q=empty(len(query_lens) + 1, dtype=torch.int32),
        max_seqlen_q=max(query_lens),
        seqused_k=empty(len(kv_lens), dtype=torch.int32),
        max_seqlen_k=max(kv_lens),
        softmax_scale=1.0,
        causal=True,
        window_size=(-1, -1) if window is None else (window - 1, 0),
        block_table=empty(len(kv_lens), max(pages), dtype=torch.int32),
        softcap=0,
        q_descale=empty(1, dtype=torch.float32),
        k_descale=empty(1, dtype=torch.float32),
        v_descale=empty(1, dtype=torch.float32),
        backend="flydsl",
    )


def describe(job: dict) -> str:
    query_lens, kv_lens = CALLS[job["call"]]
    return (
        f"{job['kernel_name']} seqs={len(query_lens)} "
        f"q={max(query_lens)} k={max(kv_lens)}"
    )


def collect_jobs(_csv_paths: list[str], _args: argparse.Namespace | None) -> list:
    return default_jobs()


def main() -> None:
    _replay.main(
        "aiter.aot.flydsl.unified_attention",
        "AOT pre-compile the gfx942 FlyDSL fp8 unified-attention kernels",
    )


if __name__ == "__main__":
    main()
