#!/usr/bin/env python3

# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""AOT pre-compilation for the sampled FlyDSL decode TopK tier.

This fixed counts-off specialization is the only new decode-TopK AOT artifact:
one-block and multi-CTA kernels keep their existing runtime coverage. Compact
and paged-logits variants remain JIT-only.

The replay calls the sampled wrapper used by runtime dispatch directly.
Packaging is intentionally independent of the ``AITER_TOPK_SAMPLED`` runtime
opt-out. FakeTensors let FlyDSL derive and compile the cache identity without a
GPU.

Usage:
    python -m aiter.aot.flydsl.topk_decode
    GPU_ARCHS=gfx950 python -m aiter.aot.flydsl.topk_decode
    python -m aiter.aot.flydsl.topk_decode --check
"""

from __future__ import annotations

import argparse

from aiter.aot.flydsl import replay as _replay

LABEL = "TOPK_DECODE"
DEFAULT_CSVS: list[str] = []
AOT_ARCH_DEFAULT = "gfx950"
_CU_NUM = 256
_ROWS = 48
_WIDTH = 98_304
_NEXT_N = 1
_K = 2048


def default_jobs() -> list[dict]:
    """The one fixed counts-off sampled specialization."""
    return [
        {
            "kernel_name": f"sampled_topk_gfx950_k{_K}",
            "gfx": AOT_ARCH_DEFAULT,
            "cu_num": _CU_NUM,
            "rows": _ROWS,
            "width": _WIDTH,
            "next_n": _NEXT_N,
            "k": _K,
        }
    ]


def replay(job: dict) -> None:
    import torch

    from aiter.ops.flydsl.topk.topk_per_row import flydsl_sampled_topk_decode

    logits = torch.empty(
        (job["rows"], job["width"]), dtype=torch.float32, device="cuda"
    )
    seq_lens = torch.empty(job["rows"], dtype=torch.int32, device="cuda")
    indices = torch.empty((job["rows"], job["k"]), dtype=torch.int32, device="cuda")
    flydsl_sampled_topk_decode(logits, job["next_n"], seq_lens, indices, job["k"])


def describe(job: dict) -> str:
    return (
        f"{job['kernel_name']} rows={job['rows']} width={job['width']} "
        f"next_n={job['next_n']}"
    )


def collect_jobs(_csv_paths: list[str], _args: argparse.Namespace | None) -> list:
    return default_jobs()


def main() -> None:
    _replay.main(
        "aiter.aot.flydsl.topk_decode",
        "AOT pre-compile the sampled FlyDSL decode TopK tier",
    )


if __name__ == "__main__":
    main()
