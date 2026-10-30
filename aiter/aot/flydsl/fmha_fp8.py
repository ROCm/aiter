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

Between grid points of the batch and query length the wrapper can still pick
a launcher no grid point does, so each launcher it can pick at all
(``_variant_space``) is replayed too, with its tile and split count pinned
through the wrapper's own overrides.

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
# workgroup counts against thresholds and the CU count, so along the key/value
# length the kernels change only between these points, where the replay
# bisects. Past the longest sequence the flattened tensors outgrow the int32
# ABI.
_BATCHES = (1, 2, 3, 4, 8, 16, 32, 64, 128)
_SEQLENS = tuple(2**i for i in range(4, 18))
# The query length of a pinned cross-attention replay: one causal split-K
# serves, with a workspace far below the int32 ABI at every split count.
_PINNED_SEQLEN_Q = 512
# The wrapper's overrides a ``_variant_jobs`` job pins.
_PINS = ("fp8_block_m", "num_kv_splits")


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


def _variant_space(causal: bool, cross: bool) -> list[tuple[float, int, int, int]]:
    """Every ``(rescale_threshold, block_m, num_kv_splits, batch_interleave_group)``
    the wrapper can pick, as an upper bound derived from its heuristics.

    - threshold: 6.0 while seqlen_kv <= _FP8_LONG_SEQ, else 4.0.
    - block_m: the narrow tile only while kv_tiles <= _FP8_NARROW_MAX_KV_TILES.
    - splits: a candidate s > 1 needs kv_tiles // s >= _FP8_AUTOSPLIT_MIN_TILES,
      so the largest kv_tiles a (threshold, block_m) bucket admits bounds s.
      Causal self-attention always lands in the skewed branch (kept < 0.75 once
      split-K is allowed), which only returns 1 or 2.
    - interleave group: > 1 only for causal self-attention without split-K.
    """
    from aiter.ops.flydsl.kernels import flash_attn_func_fp8_gfx950 as fa

    wide = fa.DUALWAVE_SWP_BLOCK_M
    narrow = wide // 2
    long_tiles = fa._FP8_LONG_SEQ // fa._FP8_BLOCK_N
    narrow_tiles = fa._FP8_NARROW_MAX_KV_TILES
    unbounded = None

    buckets = []  # (threshold, block_m, max kv_tiles in the bucket or None)
    buckets.append((6.0, wide, long_tiles))
    buckets.append((6.0, narrow, min(narrow_tiles, long_tiles)))
    buckets.append((4.0, wide, unbounded))
    if narrow_tiles > long_tiles:
        buckets.append((4.0, narrow, narrow_tiles))

    max_splits = 2 if causal and not cross else max(fa._FP8_AUTOSPLIT_CANDIDATES)
    out = []
    for thr, bm, max_tiles in buckets:
        for s in fa._FP8_AUTOSPLIT_CANDIDATES:
            if s > max_splits:
                continue
            if (
                s > 1
                and max_tiles is not unbounded
                and max_tiles // s < fa._FP8_AUTOSPLIT_MIN_TILES
            ):
                continue
            groups = {1}
            if causal and not cross and s == 1:
                groups.add(fa._FP8_BATCH_INTERLEAVE_GROUP)
            for g in sorted(groups):
                out.append((thr, bm, s, g))
    return out


def _variant_jobs(base: dict, causal: bool, cross: bool) -> list[dict]:
    """A job per launcher of ``_variant_space``, its tile and split count pinned:
    the longest grid key/value length of its threshold, and a batch of its
    interleave group. Causal cross attention at batch 1, query length 5675 and
    key/value length 66157, say, splits 11 ways, which no grid point does."""
    from aiter.ops.flydsl.kernels import flash_attn_func_fp8_gfx950 as fa

    jobs = []
    for threshold, block_m, splits, group in _variant_space(causal, cross):
        seqlen_kv = max(
            s for s in _SEQLENS if fa._fp8_rescale_threshold(s) == threshold
        )
        jobs.append(
            {
                **base,
                "batch": group,
                "seqlen_q": _PINNED_SEQLEN_Q if cross else None,
                "m": seqlen_kv,
                "m_low": None,
                "fp8_block_m": block_m,
                "num_kv_splits": splits,
            }
        )
    return jobs


def jobs_for_shape(h: int, hkv: int, d: int, dv: int, layout: str) -> list[dict]:
    """A job per causal mask, LSE output, batch and query length, at each grid
    key/value length (the query length itself, without cross attention), and
    one per launcher the wrapper can pick (``_variant_jobs``)."""
    _, cross = LAYOUTS[layout]
    jobs = []
    for causal, return_lse in itertools.product((True, False), (False, True)):
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
        }
        for batch, seqlen_q in itertools.product(
            _BATCHES, _SEQLENS if cross else (None,)
        ):
            points = [s for s in _SEQLENS if seqlen_q is None or s >= seqlen_q]
            for low, high in zip([None, *points], points):
                jobs.append(
                    {
                        **base,
                        "batch": batch,
                        "seqlen_q": seqlen_q,
                        "m": high,
                        "m_low": None if low is None else low + 1,
                    }
                )
        jobs += _variant_jobs(base, causal, cross)
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
    for name in _PINS:
        if job.get(name) is not None:
            kwargs[name] = job[name]
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
    pinned = [f"{name}={job[name]}" for name in _PINS if job.get(name) is not None]
    return " ".join(
        [
            f"{job['kernel_name']} causal={job['causal']} lse={job['return_lse']}",
            f"b={job['batch']} {seqlens}",
            *pinned,
        ]
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
