#!/usr/bin/env python3

# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""AOT pre-compilation for the gfx950 FlyDSL fp8 unified-attention kernel.

Unified attention has no tuning CSV. Its kernel key is a cross-product of the
router's build-time choices for the Qwen3-VL shape (Hkv=4, GQA group 16/8/4,
head_dim=128, fp8 QKV, paged + varlen): causal mask, output dtype, KV layout,
rescale threshold, ``batch_chunks`` bucket and the interleaved hint. This module
replays the public ``unified_attention(..., backend="flydsl")`` on FakeTensors
inside :func:`aiter.aot.flydsl.dry_run.dry_run` (see ``replay.py``), one job per
key, so the router's own host code picks the key and a router change that
strands a job fails the build (the replay raises when the router declines).
Decode calls (``max_seqlen_q == 1``) reach the same packed keys as prefill, so
these jobs cover them.

Usage:
    python -m aiter.aot.flydsl.unified_attention
    python -m aiter.aot.flydsl.unified_attention --check
"""

from __future__ import annotations

from aiter.aot.flydsl import replay as _replay

LABEL = "UNIFIED_ATTENTION"
# No tuned table: the jobs are the router's key space.
DEFAULT_CSVS: list[str] = []
AOT_ARCH_DEFAULT = "gfx950"
_CU_NUM = 256

_HEAD_DIM = 128
_NUM_KV_HEADS = 4  # Qwen3-VL production KV heads
# GQA group sizes the router packs into BN64 (_PACKED_BN64_GROUPS); Q heads = 4 * group.
_GROUPS = (16, 8, 4)
_OUT_DTYPES = ("bf16", "f16")
_PAGE = 64
_VEC = 16  # 5D-shuffled vectorization width
# Query length of the prefill jobs. Above 2 so that a ragged batch (fewer rows than
# a uniform one) still has more rows than (B - 1) * S + 1 and passes the router's
# no-decode proof for causal batches.
_PREFILL_SEQLEN_Q = 4
_FAKE_NUM_BLOCKS = 8


def _kv_lens() -> dict[float, int]:
    """One key/value length per rescale threshold the router can pick."""
    from aiter.ops.flydsl.kernels.flash_attn_func_fp8_gfx950 import (
        _FP8_LONG_SEQ,
        _fp8_rescale_threshold,
    )

    lens: dict[float, int] = {}
    for kv_len in (_FP8_LONG_SEQ, _FP8_LONG_SEQ + _PAGE):
        lens.setdefault(_fp8_rescale_threshold(kv_len), kv_len)
    return lens


def default_jobs() -> list[dict]:
    """One job per kernel key the router can select for the Qwen3-VL shape."""
    # The router's own bucket set: every key it can select needs an artifact.
    from aiter.ops.flydsl.unified_attention_kernels import _SCAN_LANES, _buckets_for

    kv_lens = _kv_lens()
    jobs = []
    for group in _GROUPS:
        for causal in (True, False):
            for out_dtype in _OUT_DTYPES:
                for shuffled in (False, True):
                    for threshold, kv_len in kv_lens.items():
                        for chunks in _buckets_for(causal):
                            # Rank-major order is causal only; the router never sets
                            # it otherwise.
                            for uniform in (True, False) if causal else (False,):
                                jobs.append(
                                    {
                                        "kernel_name": (
                                            "flydsl_unified_attn"
                                            f"_c{int(causal)}_{out_dtype}_g{group}"
                                            f"_shuf{int(shuffled)}_thr{threshold:g}"
                                            f"_bc{chunks}_il{int(uniform)}"
                                        ),
                                        "gfx": AOT_ARCH_DEFAULT,
                                        "cu_num": _CU_NUM,
                                        "group": group,
                                        "causal": causal,
                                        "out_dtype_str": out_dtype,
                                        "shuffled_kv_cache": shuffled,
                                        "max_seqlen_k": kv_len,
                                        # Largest batch of the bucket: need =
                                        # ceil((B + 1) / 64) == chunks.
                                        "num_seqs": chunks * _SCAN_LANES - 1,
                                        "uniform": uniform,
                                    }
                                )
    return jobs


def _fake_kv(torch, shuffled_kv_cache: bool):
    """Fake paged K/V pools in the layout the router expects."""
    fp8 = torch.float8_e4m3fn
    d, hkv = _HEAD_DIM, _NUM_KV_HEADS
    if shuffled_kv_cache:
        # 5D shuffled: K [nb, Hkv, D//x, PAGE, x]; V [nb, Hkv, PAGE//x, D, x].
        shapes = (
            (_FAKE_NUM_BLOCKS, hkv, d // _VEC, _PAGE, _VEC),
            (_FAKE_NUM_BLOCKS, hkv, _PAGE // _VEC, d, _VEC),
        )
    else:
        shapes = ((_FAKE_NUM_BLOCKS, _PAGE, hkv, d),) * 2
    return tuple(torch.empty(s, dtype=fp8, device="cuda") for s in shapes)


def replay(job: dict) -> None:
    import torch

    from aiter.ops.unified_attention import unified_attention

    def empty(*shape, dtype):
        return torch.empty(shape, dtype=dtype, device="cuda")

    out_dtype = torch.bfloat16 if job["out_dtype_str"] == "bf16" else torch.float16
    heads = _NUM_KV_HEADS * job["group"]
    num_seqs, kv_len = job["num_seqs"], job["max_seqlen_k"]
    max_seqlen_q = _PREFILL_SEQLEN_Q
    # A uniform batch fills q at max_seqlen_q per sequence (the router's interleaved
    # hint); one row fewer is ragged, with the same grid.
    total_q = max_seqlen_q * num_seqs - (0 if job["uniform"] else 1)
    k, v = _fake_kv(torch, job["shuffled_kv_cache"])
    descales = [empty(1, dtype=torch.float32) for _ in range(3)]
    unified_attention(
        empty(total_q, heads, _HEAD_DIM, dtype=torch.float8_e4m3fn),
        k,
        v,
        empty(total_q, heads, _HEAD_DIM, dtype=out_dtype),
        empty(num_seqs + 1, dtype=torch.int32),
        max_seqlen_q,
        empty(num_seqs, dtype=torch.int32),
        kv_len,
        _HEAD_DIM**-0.5,
        job["causal"],
        (-1, -1),
        empty(num_seqs, -(-kv_len // _PAGE), dtype=torch.int32),
        0,
        *descales,
        shuffled_kv_cache=job["shuffled_kv_cache"],
        backend="flydsl",
    )


def describe(job: dict) -> str:
    return (
        f"{job['kernel_name']} B={job['num_seqs']} skv={job['max_seqlen_k']} "
        f"uniform={job['uniform']}"
    )


def collect_jobs(_csv_paths: list[str], _args) -> list:
    return default_jobs()


def main() -> None:
    _replay.main(
        "aiter.aot.flydsl.unified_attention",
        "AOT pre-compile the gfx950 FlyDSL fp8 unified-attention kernel",
    )


if __name__ == "__main__":
    main()
