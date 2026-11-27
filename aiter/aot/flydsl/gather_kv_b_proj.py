#!/usr/bin/env python3

# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""AOT pre-compilation for the gfx1250 FlyDSL MLA gather + kv_b_proj.

``gather_kv_b_proj_flydsl`` on gfx1250 compiles one kernel per head count, output
dtype and activation-scale placement: FP8 outputs take the head-loop kernel (heads
per workgroup is a runtime argument, so every chunk size shares one kernel), BF16
outputs the tile kernel, whose A/RoPE multicast follows the row count. A serving
process would JIT each of them on its first chunked-prefill request. This module
replays the wrapper itself on FakeTensors inside
:func:`aiter.aot.flydsl.dry_run.dry_run` (see ``replay.py``) for each head count in
``DEFAULT_SHAPES`` and both output dtypes, and for BF16 over row counts bisected
where the multicast rule switches. The activation scale is replayed as a device
tensor, as ATOM passes it; a host scale is read with ``float()``, which a dry run
cannot do, and stays a JIT compile.

Usage:
    python -m aiter.aot.flydsl.gather_kv_b_proj                # DEFAULT_SHAPES
    python -m aiter.aot.flydsl.gather_kv_b_proj --heads 16     # ad hoc head count
    python -m aiter.aot.flydsl.gather_kv_b_proj --check
"""

from __future__ import annotations

import argparse
import sys

from aiter.aot.flydsl import replay as _replay
from aiter.aot.flydsl.common import dedupe_jobs

LABEL = "GATHER_KV_B_PROJ"
# No tuned table: the jobs come from DEFAULT_SHAPES or --heads.
DEFAULT_CSVS: list[str] = []
AOT_ARCH_DEFAULT = "gfx1250"
_CU_NUM = 256
# Attention heads per rank (Kimi-K3 MLA: 96 heads at TP1/2/4/8). Jobs dedupe.
DEFAULT_SHAPES = {
    "kimi_k3": [96, 48, 24, 12],
}
_KV_C, _KV_PE, _NOPE, _V = 512, 64, 128, 128
# Largest row count the BF16 tile kernel's multicast rule is bisected over.
_MAX_ROWS = 4096
_CACHE_BLOCKS = 1 << 16


def unsupported_reason(heads: int) -> str | None:
    if heads <= 0:
        return f"heads={heads} must be positive"
    return None


def jobs_for_heads(heads: int) -> list[dict]:
    """FP8 (head loop when the head count is even) and BF16 outputs."""
    jobs = []
    for out_fp8 in (True, False):
        job = {
            "kernel_name": f"gather_kv_b_proj_gfx1250_h{heads}",
            "gfx": AOT_ARCH_DEFAULT,
            "cu_num": _CU_NUM,
            "heads": heads,
            "out_fp8": out_fp8,
            "m": _MAX_ROWS,
        }
        if not out_fp8:
            job["m_low"] = 1  # the tile kernel's multicast follows the row count
        jobs.append(job)
    return jobs


def shape_jobs(shapes: dict[str, list] = DEFAULT_SHAPES) -> list[dict]:
    jobs = []
    for model, entries in shapes.items():
        for heads in entries:
            reason = unsupported_reason(heads)
            if reason:
                print(
                    f"  [WARN] {model}: {heads} heads unsupported ({reason}), skipping"
                )
                continue
            jobs.extend(jobs_for_heads(heads))
    return dedupe_jobs(jobs)


def replay(job: dict) -> None:
    import torch

    from aiter.ops.flydsl import gather_kv_b_proj as op
    from aiter.ops.flydsl import gather_kv_b_proj_gfx1250 as op1250

    # Per-process caches of the device the wrapper reads: a worker may replay
    # jobs of several archs / cu_num in turn.
    for fn in (getattr(op, "_arch_of", None), getattr(op1250, "_num_cus", None)):
        if hasattr(fn, "cache_clear"):
            fn.cache_clear()

    def empty(*shape, dtype, device="cuda"):
        return torch.empty(shape, dtype=dtype, device=device)

    heads, m = job["heads"], job["m"]
    fp8 = torch.float8_e4m3fn
    out_dtype = fp8 if job["out_fp8"] else torch.bfloat16
    weight = empty(heads * (_NOPE + _V), _KV_C, dtype=fp8)
    weight.is_shuffled = True
    kwargs = {}
    if job["out_fp8"]:
        kwargs = {
            "k_out_scale": empty(1, dtype=torch.float32),
            "v_out_scale": empty(1, dtype=torch.float32),
        }
    op.gather_kv_b_proj_flydsl(
        empty(_CACHE_BLOCKS, 1, _KV_C + _KV_PE, dtype=fp8),
        empty(1, dtype=torch.float32),
        None,
        empty(m, dtype=torch.int32),
        None,
        weight,
        empty(heads * (_NOPE + _V), 1, dtype=torch.float32),
        empty(m, heads, _NOPE + _KV_PE, dtype=out_dtype),
        empty(m, heads, _V, dtype=out_dtype),
        num_tokens=m,
        weight_preshuffle=True,
        **kwargs,
    )


def describe(job: dict) -> str:
    out = "fp8" if job["out_fp8"] else "bf16"
    return f"{job['kernel_name']} {out} out, {job['m']} rows"


def collect_jobs(_csv_paths: list[str], args: argparse.Namespace | None) -> list:
    if args is None or not args.heads:
        return shape_jobs()
    jobs = []
    for heads in args.heads:
        reason = unsupported_reason(heads)
        if reason:
            print(f"Error: {heads} heads unsupported: {reason}")
            sys.exit(1)
        jobs.extend(jobs_for_heads(heads))
    return dedupe_jobs(jobs)


def configure(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--heads",
        type=int,
        action="append",
        default=[],
        help="attention heads per rank, repeatable; replaces DEFAULT_SHAPES",
    )


def main() -> None:
    _replay.main(
        "aiter.aot.flydsl.gather_kv_b_proj",
        "AOT pre-compile the gfx1250 FlyDSL MLA gather + kv_b_proj",
    )


if __name__ == "__main__":
    main()
