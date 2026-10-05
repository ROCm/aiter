#!/usr/bin/env python3

# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""AOT pre-compilation for the gfx942 FlyDSL fp8 unified-attention kernels.

Unified attention has no tuning CSV. Its builder is keyed by the layer shape,
the page size, the K/V strides, and whether the batch is decode-only, so the
set a model can reach is finite: per shape in ``DEFAULT_SHAPES``, every page
size the adapter serves times two K/V layouts (separate contiguous caches, and
vLLM's views of one ``[blocks, kv_heads, page, 2 * head_dim]`` cache) times
the unified and decode-only builds, plus one combine kernel per (head dim,
query heads). Each job takes its launcher from the builders the adapter calls,
with the same arguments, and invokes it under ``FakeTensorMode`` +
``COMPILE_ONLY=1`` on fake tensors whose dtypes, ranks, and strides match the
runtime call.

Usage:
    python -m aiter.aot.flydsl.unified_attention
    python -m aiter.aot.flydsl.unified_attention --list
"""

from __future__ import annotations

import argparse
import time

from aiter.aot.flydsl.common import (
    compile_only_env,
    dedupe_jobs,
    override_env,
    run_jobs_parallel,
)

AOT_ARCH = "gfx942"
# Layer shapes to pre-compile, per model: (num_heads, num_kv_heads, head_dim,
# sliding window in keys or None), with the head counts per rank.
DEFAULT_SHAPES = {
    "gemma4_31b_tp1": [(32, 4, 512, None), (32, 16, 256, 1024)],
}
LAYOUTS = ("plain", "vllm")
# The adapter builds decode-only batches (max_seqlen_q == 1) separately.
MODES = ("unified", "decode")
# Fake geometry: sizes never reach the compile key, only dtypes, ranks, and
# strides do.
_FAKE_SEQS = 4
_FAKE_TOKENS = 16
_FAKE_BLOCKS = 8


def _attention_name(num_heads, num_kv_heads, head_dim, window, page_size, layout, mode):
    return (
        f"flydsl_unified_attn_gfx942_h{num_heads}_hkv{num_kv_heads}_d{head_dim}"
        f"_w{window or 0}_p{page_size}_{layout}_{mode}"
    )


def default_jobs(shapes: dict[str, list] = DEFAULT_SHAPES) -> list[dict]:
    """Every attention and combine variant ``DEFAULT_SHAPES`` can reach."""
    from aiter.ops.flydsl.unified_attention_kernels import _PAGE_SIZES

    jobs = []
    for entries in shapes.values():
        for num_heads, num_kv_heads, head_dim, window in entries:
            for page_size in _PAGE_SIZES:
                for layout in LAYOUTS:
                    for mode in MODES:
                        jobs.append(
                            {
                                "kernel_name": _attention_name(
                                    num_heads,
                                    num_kv_heads,
                                    head_dim,
                                    window,
                                    page_size,
                                    layout,
                                    mode,
                                ),
                                "path": "attention",
                                "num_heads": num_heads,
                                "num_kv_heads": num_kv_heads,
                                "head_dim": head_dim,
                                "window": window,
                                "page_size": page_size,
                                "layout": layout,
                                "mode": mode,
                            }
                        )
            jobs.append(
                {
                    "kernel_name": (
                        f"flydsl_unified_attn_gfx942_combine_h{num_heads}_d{head_dim}"
                    ),
                    "path": "combine",
                    "num_heads": num_heads,
                    "head_dim": head_dim,
                }
            )
    return dedupe_jobs(jobs)


def _fake_kv(layout, page_size, num_kv_heads, head_dim):
    """int8 K and V, as the adapter passes them, in one cache layout."""
    import torch

    if layout == "plain":
        shape = (_FAKE_BLOCKS, page_size, num_kv_heads, head_dim)
        return torch.empty(shape, dtype=torch.int8), torch.empty(
            shape, dtype=torch.int8
        )
    cache = torch.empty(
        (_FAKE_BLOCKS, num_kv_heads, page_size, 2 * head_dim), dtype=torch.int8
    )
    kv = cache.transpose(1, 2)
    return kv[..., :head_dim], kv[..., head_dim:]


def _compile_attention(
    num_heads, num_kv_heads, head_dim, window, page_size, layout, mode
):
    import torch

    from aiter.ops.flydsl.kernels.flash_attn_fp8_gfx942 import (
        build_flash_attn_fp8_gfx942_module,
    )

    k, v = _fake_kv(layout, page_size, num_kv_heads, head_dim)
    launch = build_flash_attn_fp8_gfx942_module(
        head_dim,
        num_heads,
        num_kv_heads,
        window,
        page_size,
        k.stride()[:3] + v.stride()[:3],
        decode_only=mode == "decode",
    )

    def t(shape, dtype):
        return torch.empty(shape, dtype=dtype)

    # Mirror the adapter's _launch call: the same argument kinds and dtypes.
    launch(
        t((_FAKE_TOKENS, num_heads, head_dim), torch.int8),
        k,
        v,
        t((_FAKE_TOKENS, num_heads, head_dim), torch.bfloat16),
        t((1024,), torch.float32),
        t((_FAKE_SEQS + 1,), torch.int32),
        t((_FAKE_SEQS,), torch.int32),
        t((_FAKE_SEQS, 4), torch.int32),
        t((1,), torch.float32),
        t((1,), torch.float32),
        t((1,), torch.float32),
        _FAKE_SEQS,
        _FAKE_TOKENS,
        1,
        3,
        4,
        1.0,
        _FAKE_TOKENS + _FAKE_SEQS,
        stream=None,
    )


def _compile_combine(num_heads, head_dim):
    import torch

    from aiter.ops.flydsl.kernels.flash_attn_fp8_gfx942 import (
        build_flash_attn_fp8_gfx942_combine_module,
    )

    combine = build_flash_attn_fp8_gfx942_combine_module(head_dim, num_heads)
    combine(
        torch.empty((1024,), dtype=torch.float32),
        torch.empty((_FAKE_TOKENS, num_heads, head_dim), dtype=torch.bfloat16),
        torch.empty((_FAKE_SEQS + 1,), dtype=torch.int32),
        _FAKE_SEQS,
        4,
        stream=None,
    )


def compile_one_config(**job) -> dict:
    result = {**job, "compile_time": None}
    params = {k: v for k, v in job.items() if k not in ("kernel_name", "path")}
    from torch._subclasses.fake_tensor import FakeTensorMode

    started = time.time()
    try:
        # COMPILE_ONLY is mandatory: without it the launcher also launches.
        with (
            override_env("FLYDSL_GPU_ARCH", AOT_ARCH),
            compile_only_env(),
            FakeTensorMode(),
        ):
            if job["path"] == "attention":
                _compile_attention(**params)
            else:
                _compile_combine(**params)
        result["compile_time"] = time.time() - started
    except Exception as error:  # noqa: BLE001
        print(f"  [FAIL] {job['kernel_name']}: {error}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--list", action="store_true", help="print the jobs and exit")
    args = parser.parse_args()
    jobs = default_jobs()
    if args.list:
        for job in jobs:
            print(job["kernel_name"])
        print(f"{len(jobs)} jobs")
        return
    results = run_jobs_parallel(compile_one_config, jobs)
    failed = sum(result["compile_time"] is None for result in results)
    print(f"Compiled: {len(results) - failed} ok, {failed} failed")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
