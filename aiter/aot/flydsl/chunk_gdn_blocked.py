#!/usr/bin/env python3

# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""AOT pre-compile the blocked GDN prefill kernels (build_map, carry, emit).

The blocked path has no tuned table of its own: the public K5 wrapper takes it
when a call passes its gate (see ``_resolve_adaptive_k5_metadata``), so each
job replays ``chunk_gated_delta_rule_fwd_h_flydsl_opt`` on one long sequence
that passes the gate and lets the runtime pick build_map -> carry -> emit.
Which ``(H, Hg)`` pairs to build comes from the K5 tuned CSVs, whatever arch
the row was tuned for, because the blocked gate does not consult the table.

Usage:
    python -m aiter.aot.flydsl.chunk_gdn_blocked
    python -m aiter.aot.flydsl.chunk_gdn_blocked --csv /path/to/tuned.csv
    python -m aiter.aot.flydsl.chunk_gdn_blocked --check
"""

from __future__ import annotations

import csv
import itertools

from aiter.aot.flydsl import chunk_gdn_h as _serial
from aiter.aot.flydsl import replay as _replay
from aiter.jit.core import AITER_CONFIGS

LABEL = "CHUNK_GDN_BLOCKED"
AOT_ARCH_DEFAULT = "gfx950"
DEFAULT_CSVS = [AITER_CONFIGS.AITER_CONFIG_GDN_K5_OPT_FILE]
_KERNEL_NAME = "chunk_gdn_blocked"
# gfx950 parts as (XCDs, CUs): MI355X 8 / 256, MI350P 4 / 128. replay forks each
# job per entry (see ``_with_xcd_variants``); the XCD count is part of the cache
# key of the blocked kernels.
XCD_VARIANTS = {"gfx950": ((8, 256), (4, 128))}
_MIN_CHUNKS = 128


def _chunks(H: int) -> int:
    """Chunks of the one sequence a job replays. With block size
    s = max(_BLOCKED_MIN_BLOCK_CHUNKS, ceil(L / _GDN_K5_TARGET_SEGMENTS)) there
    are at most _GDN_K5_TARGET_SEGMENTS blocks, so the gate
    (H * blocks <= _BLOCKED_MAX_WORK_PER_CHUNK * L, L >= _BLOCKED_MIN_CHUNKS)
    holds from L = ceil(_GDN_K5_TARGET_SEGMENTS * H / _BLOCKED_MAX_WORK_PER_CHUNK)."""
    # Lazy: the runtime module needs FlyDSL, which the AOT collector may not load.
    from aiter.ops.flydsl.linear_attention_prefill_kernels import (
        _BLOCKED_MAX_WORK_PER_CHUNK,
        _BLOCKED_MIN_CHUNKS,
        _GDN_K5_TARGET_SEGMENTS,
    )

    return max(
        _MIN_CHUNKS,
        _BLOCKED_MIN_CHUNKS,
        -(-_GDN_K5_TARGET_SEGMENTS * H // _BLOCKED_MAX_WORK_PER_CHUNK),
    )


def parse_csv(csv_path: str) -> list[dict]:
    """One job per eligible ``(H, Hg)`` pair of the table and variant of the
    kernels (initial state, state dtype); XCD variants are added by replay. Rows qualify by shape and
    dtype only: the blocked gate does not consult the table, so the arch a row
    was tuned for does not matter."""
    pairs = set()
    with open(csv_path, encoding="utf-8", newline="") as f:
        rows = csv.DictReader(line for line in f if not line.lstrip().startswith("#"))
        for row in rows:
            try:
                dtype = (row.get("dtype") or "torch.bfloat16").strip()
                K, V = int(row["K"]), int(row["V"])
                BT = int(row.get("BT") or 64)
                pair = (int(row["H"]), int(row["Hg"]))
            except (KeyError, TypeError, ValueError) as e:
                print(f"  [WARN] malformed row in {csv_path}: {e}")
                continue
            if dtype == "torch.bfloat16" and (K, V, BT) == (128, 128, 64):
                pairs.add(pair)
    jobs = []
    for (H, Hg), use_h0, state_bf16 in itertools.product(
        sorted(pairs), (False, True), (False, True)
    ):
        chunks = _chunks(H)
        jobs.append(
            {
                "kernel_name": _KERNEL_NAME,
                "dtype": "torch.bfloat16",
                "gfx": AOT_ARCH_DEFAULT,
                "K": 128,
                "V": 128,
                "BT": 64,
                "H": H,
                "Hg": Hg,
                "is_varlen": True,
                "use_h0": use_h0,
                "store_fs": True,
                "snapshot_bf16": True,
                "state_bf16": state_bf16,
                "g_head_major": True,
                "use_state_indices": False,
                "total_chunks": chunks,
                "max_seq_chunks": chunks,
            }
        )
    return jobs


def replay(job: dict) -> None:
    # The public wrapper's own gate selects build_map -> carry -> emit.
    _serial.replay(job)


def describe(job: dict) -> str:
    flags = " ".join(name for name in ("use_h0", "state_bf16") if job[name])
    return (
        f"chunk_gdn_blocked H={job['H']} Hg={job['Hg']} "
        f"chunks={job['total_chunks']} {flags} num_xcds={job.get('num_xcds')}"
    )


def config_env(csv_paths: list[str]) -> dict[str, str]:
    return _serial.config_env(csv_paths)


def main() -> None:
    _replay.main(
        "aiter.aot.flydsl.chunk_gdn_blocked",
        "AOT pre-compile FlyDSL blocked GDN prefill kernels",
    )


if __name__ == "__main__":
    main()
