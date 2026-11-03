#!/usr/bin/env python3

# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""AOT pre-compilation for the gfx950 single-kernel FlyDSL GDN prefill.

``gdn_fused_prefill_fwd`` compiles one kernel per (H, Hg, BV, HAS_TRACK,
HAS_INIT_FLAG, scale): BV follows the batch's sequence count and the CU count,
and the two flags follow the optional ``track_state`` / ``has_initial_state``
arguments, so a serving process would JIT each new combination on the request
path. This module replays the wrapper itself on FakeTensors inside
:func:`aiter.aot.flydsl.dry_run.dry_run` (see ``replay.py``) for each head
shape in ``DEFAULT_SHAPES``, with and without each optional argument, over
batches of 1 to ``_MAX_SEQUENCES`` sequences, bisected where the BV rule
switches. Only the default scale is covered.

Usage:
    python -m aiter.aot.flydsl.gdn_fused_prefill               # DEFAULT_SHAPES
    python -m aiter.aot.flydsl.gdn_fused_prefill --shape 48:16  # ad hoc H:Hg
    python -m aiter.aot.flydsl.gdn_fused_prefill --check
"""

from __future__ import annotations

import argparse
import itertools
import sys

from aiter.aot.flydsl import replay as _replay
from aiter.aot.flydsl.common import dedupe_jobs

LABEL = "GDN_FUSED_PREFILL"
# No tuned table: the jobs come from DEFAULT_SHAPES or --shape.
DEFAULT_CSVS: list[str] = []
AOT_ARCH_DEFAULT = "gfx950"
_CU_NUM = 256
# Value/key head counts per rank, (H, Hg), at TP1/2/4/8. Jobs dedupe, so two
# models may share a head shape.
DEFAULT_SHAPES = {
    "qwen3_8_flash_next": [(48, 16), (24, 8), (12, 4), (6, 2)],
    "qwen3_5_397b": [(64, 16), (32, 8), (16, 4), (8, 2)],
    "qwen3_5_35b": [(32, 16), (16, 8), (8, 4), (4, 2)],
}
# Largest batch the BV rule is bisected over: past it the grid never fits in
# one pass over the CUs, so every batch picks the wide BV.
_MAX_SEQUENCES = 1024
_KD = _VD = 128
_BT = 64


def parse_shape(spec: str) -> tuple[int, int]:
    """``"H:Hg"`` -> ``(H, Hg)``."""
    h, hg = (int(x) for x in spec.strip().split(":"))
    return h, hg


def unsupported_reason(h: int, hg: int) -> str | None:
    """Why the kernel cannot serve this head shape, or None."""
    if h <= 0 or hg <= 0 or h % hg:
        return f"H={h} must be a positive multiple of Hg={hg}"
    return None


def jobs_for_shape(h: int, hg: int) -> list[dict]:
    """A job per optional argument pair, over the BV rule's range of batches."""
    return [
        {
            "kernel_name": f"gdn_fused_prefill_h{h}_hg{hg}",
            "gfx": AOT_ARCH_DEFAULT,
            "cu_num": _CU_NUM,
            "H": h,
            "Hg": hg,
            "has_track": has_track,
            "has_init": has_init,
            "m": _MAX_SEQUENCES,
            "m_low": 1,
        }
        for has_track, has_init in itertools.product((False, True), (False, True))
    ]


def shape_jobs(shapes: dict[str, list] = DEFAULT_SHAPES) -> list[dict]:
    """Every job of ``shapes``; an unsupported shape warns and is skipped."""
    jobs = []
    for model, entries in shapes.items():
        for h, hg in entries:
            reason = unsupported_reason(h, hg)
            if reason:
                print(f"  [WARN] {model}: {h}:{hg} unsupported ({reason}), skipping")
                continue
            jobs.extend(jobs_for_shape(h, hg))
    return dedupe_jobs(jobs)


def replay(job: dict) -> None:
    import torch

    from aiter.ops.flydsl import gdn_fused_prefill as op

    # Per-process caches of the device the wrapper reads: a worker may replay
    # jobs of several cu_num in turn.
    op._num_cus.cache_clear()
    op._is_gfx950.cache_clear()

    def empty(*shape, dtype):
        return torch.empty(shape, dtype=dtype, device="cuda")

    h, hg, n = job["H"], job["Hg"], job["m"]
    t = n * _BT
    # q/k/v as views of one packed projection, as serving stacks pass them.
    mixed = empty(t, 2 * hg * _KD + h * _VD, dtype=torch.bfloat16)
    q, k, v = torch.split(mixed, [hg * _KD, hg * _KD, h * _VD], dim=-1)
    kwargs = {}
    if job["has_track"]:
        kwargs["track_state"] = empty(n, h, _VD, _KD, dtype=torch.float32)
        kwargs["track_chunk_idx"] = empty(n, dtype=torch.int32)
    if job["has_init"]:
        kwargs["has_initial_state"] = empty(n, dtype=torch.bool)
    op.gdn_fused_prefill_fwd(
        q.view(1, t, hg, _KD),
        k.view(1, t, hg, _KD),
        v.view(1, t, h, _VD),
        empty(1, t, h, dtype=torch.float32),
        empty(1, t, h, dtype=torch.float32),
        ssm_states=empty(n, h, _VD, _KD, dtype=torch.bfloat16),
        cache_indices=empty(n, dtype=torch.int32),
        cu_seqlens=empty(n + 1, dtype=torch.int32),
        **kwargs,
    )


def describe(job: dict) -> str:
    flags = " ".join(name for name in ("has_track", "has_init") if job[name])
    return f"{job['kernel_name']} {job['m']} sequences {flags}".rstrip()


def collect_jobs(_csv_paths: list[str], args: argparse.Namespace | None) -> list:
    if args is None or not args.shape:
        return shape_jobs()
    jobs = []
    for spec in args.shape:
        h, hg = parse_shape(spec)
        reason = unsupported_reason(h, hg)
        if reason:
            print(f"Error: {spec} is unsupported: {reason}")
            sys.exit(1)
        jobs.extend(jobs_for_shape(h, hg))
    return dedupe_jobs(jobs)


def configure(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--shape",
        action="append",
        default=[],
        help="H:Hg, repeatable; replaces DEFAULT_SHAPES",
    )


def main() -> None:
    _replay.main(
        "aiter.aot.flydsl.gdn_fused_prefill",
        "AOT pre-compile the gfx950 single-kernel FlyDSL GDN prefill",
    )


if __name__ == "__main__":
    main()
