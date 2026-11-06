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
_CU_NUM = 256
# XCD counts of gfx950 parts (MI350P: 4, MI355X: 8). Set on each job, since the
# blocked kernels are gfx950-only and the count is part of their cache key.
_NUM_XCDS = (4, 8)
_MIN_CHUNKS = 128


def _chunks(H: int) -> int:
    """Chunks of the one sequence a job replays. With block size
    s = max(12, ceil(L / 8)) there are at most 8 blocks, so the gate
    (H * blocks <= 6 * L, L >= 96) holds from L = ceil(8H / 6)."""
    return max(_MIN_CHUNKS, -(-8 * H // 6))


def parse_csv(csv_path: str) -> list[dict]:
    """One job per eligible ``(H, Hg)`` pair of the table and variant of the
    kernels (initial state, state dtype, XCD count). Rows qualify by shape and
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
    for (H, Hg), use_h0, state_bf16, num_xcds in itertools.product(
        sorted(pairs), (False, True), (False, True), _NUM_XCDS
    ):
        chunks = _chunks(H)
        jobs.append(
            {
                "kernel_name": _KERNEL_NAME,
                "dtype": "torch.bfloat16",
                "gfx": AOT_ARCH_DEFAULT,
                "cu_num": _CU_NUM,
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
                "num_xcds": num_xcds,
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
        f"chunks={job['total_chunks']} {flags} num_xcds={job['num_xcds']}"
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
