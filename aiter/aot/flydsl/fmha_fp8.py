#!/usr/bin/env python3

# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""AOT pre-compilation for the gfx950 FlyDSL FP8 flash-attention forward.

``flydsl_flash_attn_fp8_func`` picks its rescale threshold, BLOCK_M, split-K
factor and batch interleave per call from the batch and sequence lengths, so
each new combination is a JIT compile on the serving path. This module
replays the wrapper itself on FakeTensors inside
:func:`aiter.aot.flydsl.dry_run.dry_run` (see ``replay.py``) for each head
shape in ``DEFAULT_SHAPES``, over a grid of batch sizes and sequence lengths:
for every batch and query length, at each grid key/value length, bisecting
between grid points where the wrapper switches kernels. The heuristics stay
the wrapper's own, so changing them needs no AOT change.

Usage:
    python -m aiter.aot.flydsl.fmha_fp8                          # DEFAULT_SHAPES
    python -m aiter.aot.flydsl.fmha_fp8 --shape 12:12:192:128    # ad hoc
    python -m aiter.aot.flydsl.fmha_fp8 --check
"""

from __future__ import annotations

import argparse
import itertools
import sys

from aiter.aot.flydsl import replay as _replay
from aiter.aot.flydsl.common import dedupe_jobs

LABEL = "FMHA_FP8"
# No tuned table: the jobs come from DEFAULT_SHAPES or --shape.
DEFAULT_CSVS: list[str] = []
AOT_ARCH_DEFAULT = "gfx950"
_CU_NUM = 256
LAYOUTS = {
    "varlen_cross": (True, True),
    "varlen": (True, False),
    "dense": (False, False),
    "dense_cross": (False, True),
}
# Head shapes to pre-compile, per model: (num_heads, num_kv_heads, head_dim,
# head_dim_v, layout) with the head counts per rank. ``varlen_cross`` is the
# chunked-prefill serving path. Jobs dedupe, so two models may share a head
# shape.
DEFAULT_SHAPES = {
    "kimi_k3_tp8": [(12, 12, 192, 128, "varlen_cross")],
}
# The sizes the wrapper is replayed at. Its heuristics compare tile and
# workgroup counts against thresholds and the CU count, so the kernels change
# only between these points, where the replay bisects. Past the longest
# sequence the flattened tensors outgrow the int32 ABI.
_BATCHES = (1, 2, 3, 4, 8, 16, 32, 64, 128)
_SEQLENS = tuple(2**i for i in range(4, 18))


def _layout(value: str | None) -> str:
    layout = str(value or "").strip() or "varlen_cross"
    if layout not in LAYOUTS:
        raise ValueError(
            f"unknown layout {layout!r}; expected one of {sorted(LAYOUTS)}"
        )
    return layout


def parse_shape(spec: str) -> tuple[int, int, int, int, str]:
    """``"H:Hkv:D:Dv[@layout]"`` -> ``(H, Hkv, D, Dv, layout)``."""
    dims, _, layout = spec.strip().partition("@")
    h, hkv, d, dv = (int(x) for x in dims.split(":"))
    return h, hkv, d, dv, _layout(layout)


def unsupported_reason(h: int, hkv: int, d: int, dv: int) -> str | None:
    """Why the kernel cannot serve this head shape, or None. The runtime gate
    uses the same check, so a rejected shape is never dispatched."""
    from aiter.ops.flydsl.kernels.flash_attn_func_fp8_gfx950 import (
        _fp8_config_reason,
    )

    return _fp8_config_reason(h, hkv, d, dv)


def jobs_for_shape(h: int, hkv: int, d: int, dv: int, layout: str) -> list[dict]:
    """A job per causal mask, LSE output, batch and query length, at each grid
    key/value length (the query length itself, without cross attention)."""
    _, cross = LAYOUTS[layout]
    jobs = []
    for causal, return_lse, batch in itertools.product(
        (True, False), (False, True), _BATCHES
    ):
        base = {
            "kernel_name": f"fmha_fp8_h{h}_hkv{hkv}_d{d}_dv{dv}_{layout}",
            "gfx": AOT_ARCH_DEFAULT,
            "cu_num": _CU_NUM,
            "num_heads": h,
            "num_kv_heads": hkv,
            "head_dim": d,
            "head_dim_v": dv,
            "layout": layout,
            "causal": causal,
            "return_lse": return_lse,
            "batch": batch,
        }
        for seqlen_q in _SEQLENS if cross else (None,):
            points = [s for s in _SEQLENS if seqlen_q is None or s >= seqlen_q]
            for low, high in zip([None, *points], points):
                jobs.append(
                    {
                        **base,
                        "seqlen_q": seqlen_q,
                        "m": high,
                        "m_low": None if low is None else low + 1,
                    }
                )
    return jobs


def shape_jobs(shapes: dict[str, list] = DEFAULT_SHAPES) -> list[dict]:
    """Every job of ``shapes``; a shape the kernel's own gate rejects is
    skipped with a warning rather than failing the whole build."""
    jobs = []
    for model, entries in shapes.items():
        for h, hkv, d, dv, layout in entries:
            reason = unsupported_reason(h, hkv, d, dv)
            if reason:
                print(
                    f"  [WARN] {model}: {h}:{hkv}:{d}:{dv} unsupported "
                    f"({reason}), skipping"
                )
                continue
            jobs.extend(jobs_for_shape(h, hkv, d, dv, _layout(layout)))
    return dedupe_jobs(jobs)


def replay(job: dict) -> None:
    import torch

    from aiter.ops.flydsl.kernels.flash_attn_func_fp8_gfx950 import (
        flydsl_flash_attn_fp8_func,
    )

    def empty(*shape, dtype=torch.float8_e4m3fn):
        return torch.empty(shape, dtype=dtype, device="cuda")

    varlen, cross = LAYOUTS[job["layout"]]
    h, hkv = job["num_heads"], job["num_kv_heads"]
    d, dv, batch = job["head_dim"], job["head_dim_v"], job["batch"]
    seqlen_kv = job["m"]
    seqlen_q = job["seqlen_q"] if cross else seqlen_kv
    kwargs = {}
    if varlen:
        q = empty(batch * seqlen_q, h, d)
        k = empty(batch * seqlen_kv, hkv, d)
        v = empty(batch * seqlen_kv, hkv, dv)
        kwargs = {
            "cu_seqlens_q": empty(batch + 1, dtype=torch.int32),
            "cu_seqlens_kv": empty(batch + 1, dtype=torch.int32),
            "max_seqlen_q": seqlen_q,
            "max_seqlen_kv": seqlen_kv,
            "cross_seqlen": cross,
        }
    else:
        q = empty(batch, seqlen_q, h, d)
        k = empty(batch, seqlen_kv, hkv, d)
        v = empty(batch, seqlen_kv, hkv, dv)
    descale = [empty(1, dtype=torch.float32) for _ in range(3)]
    flydsl_flash_attn_fp8_func(
        q,
        k,
        v,
        causal=job["causal"],
        q_descale=descale[0],
        k_descale=descale[1],
        v_descale=descale[2],
        return_lse=job["return_lse"],
        **kwargs,
    )


def describe(job: dict) -> str:
    seqlens = (
        f"sq={job['seqlen_q']} skv={job['m']}"
        if job["seqlen_q"] is not None
        else f"s={job['m']}"
    )
    return (
        f"{job['kernel_name']} causal={job['causal']} lse={job['return_lse']} "
        f"b={job['batch']} {seqlens}"
    )


def collect_jobs(_csv_paths: list[str], args: argparse.Namespace | None) -> list:
    if args is None or not args.shape:
        return shape_jobs()
    jobs = []
    for spec in args.shape:
        h, hkv, d, dv, layout = parse_shape(spec)
        reason = unsupported_reason(h, hkv, d, dv)
        if reason:
            print(f"Error: {spec} is unsupported by the FP8 FMHA kernel: {reason}")
            sys.exit(1)
        jobs.extend(jobs_for_shape(h, hkv, d, dv, layout))
    return dedupe_jobs(jobs)


def configure(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--shape",
        action="append",
        default=[],
        help="H:Hkv:D:Dv[@layout], repeatable; replaces DEFAULT_SHAPES",
    )


def main() -> None:
    _replay.main(
        "aiter.aot.flydsl.fmha_fp8",
        "AOT pre-compile the gfx950 FlyDSL FP8 flash-attention forward",
    )


if __name__ == "__main__":
    main()
