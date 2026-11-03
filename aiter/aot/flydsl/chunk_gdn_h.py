#!/usr/bin/env python3

# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""AOT pre-compile FlyDSL chunk_gdn_h opt (K5) kernels from tuned CSV configs.

Each tuned row is replayed through ``chunk_gated_delta_rule_fwd_h_flydsl_opt``
itself on FakeTensors inside :func:`aiter.aot.flydsl.dry_run.dry_run` (see
``replay.py``). The runtime picks ``BV`` per batch shape -- the tuned row for
its ``(total_chunks, max_seq_chunks)``, else an LDS/CU rule -- so each row is
replayed at its own batch shape, and each of its layouts once more over a
range of batch shapes the table never measured (``m`` single-chunk
sequences, ``m`` from 1 to ``_MAX_SEQUENCES``), bisected where the rule
switches ``BV``. The caller-side layouts a row does not fix fork the kernel
too: ``g`` head- or token-major, and a dense or indexed (state-pool) initial
state.

Usage:
    python -m aiter.aot.flydsl.chunk_gdn_h
    python -m aiter.aot.flydsl.chunk_gdn_h --csv /path/to/tuned.csv
    python -m aiter.aot.flydsl.chunk_gdn_h --check
"""

from __future__ import annotations

import csv
import itertools
import os

from aiter.aot.flydsl import replay as _replay
from aiter.jit.core import AITER_CONFIGS

LABEL = "CHUNK_GDN_H"
AOT_ARCH_DEFAULT = "gfx950"
# Keep the default AOT coverage aligned with runtime config resolution.
DEFAULT_CSVS = [AITER_CONFIGS.AITER_CONFIG_GDN_K5_OPT_FILE]
_KERNEL_NAME = "chunk_gdn_fwd_h_flydsl_opt"
_DTYPES = ("torch.bfloat16", "torch.float16")
# Largest batch the BV rule is bisected over: past it every rule picks the
# widest BV, as the work per CU only grows.
_MAX_SEQUENCES = 8192
_G_HEAD_MAJOR = (False, True)


def _bool(value: str | None, default: bool) -> bool:
    value = (value or "").strip()
    if not value:
        return default
    if value in ("True", "true", "1", "yes"):
        return True
    if value in ("False", "false", "0", "no"):
        return False
    raise ValueError(f"unrecognised bool literal {value!r}")


def parse_csv(csv_path: str) -> list[dict]:
    """One job per tuned row and caller layout at the row's batch shape, and
    one per distinct layout over the rule's range of batch shapes."""
    jobs, rules = [], {}
    with open(csv_path, encoding="utf-8", newline="") as f:
        rows = csv.DictReader(line for line in f if not line.lstrip().startswith("#"))
        for row in rows:
            dtype = (row.get("dtype") or "torch.bfloat16").strip()
            if dtype not in _DTYPES:
                print(f"  [WARN] Unsupported dtype {dtype!r}, skipping")
                continue
            try:
                shape = {
                    "dtype": dtype,
                    "gfx": (row.get("gfx") or "").strip(),
                    "cu_num": int(row.get("cu_num") or 0),
                    "K": int(row["K"]),
                    "V": int(row["V"]),
                    "BT": int(row.get("BT") or 64),
                    "H": int(row["H"]),
                    "Hg": int(row["Hg"]),
                    "is_varlen": _bool(row.get("is_varlen"), True),
                    "use_h0": _bool(row.get("use_h0"), True),
                    "store_fs": _bool(row.get("store_fs"), True),
                    "snapshot_bf16": _bool(row.get("snapshot_bf16"), True),
                    "state_bf16": _bool(row.get("state_bf16"), False),
                }
                chunks = (int(row["total_chunks"]), int(row["max_seq_chunks"]))
            except (KeyError, TypeError, ValueError) as e:
                print(f"  [WARN] malformed row in {csv_path}: {e}")
                continue
            indexed = (
                (False, True) if shape["use_h0"] and shape["store_fs"] else (False,)
            )
            for g_head_major, use_state_indices in itertools.product(
                _G_HEAD_MAJOR, indexed
            ):
                layout = {
                    **shape,
                    "kernel_name": _KERNEL_NAME,
                    "g_head_major": g_head_major,
                    "use_state_indices": use_state_indices,
                }
                jobs.append(
                    {**layout, "total_chunks": chunks[0], "max_seq_chunks": chunks[1]}
                )
                rule = {**layout, "m": _MAX_SEQUENCES, "m_low": 1}
                rules[tuple(sorted(rule.items()))] = rule
    return jobs + list(rules.values())


def _sequences(job: dict) -> list[int]:
    """Chunk counts of the batch a job replays: ``m`` sequences of one chunk,
    or the row's ``total_chunks`` over sequences of at most ``max_seq_chunks``."""
    if "m" in job:
        return [1] * job["m"]
    total, longest = job["total_chunks"], job["max_seq_chunks"]
    counts = []
    while total > 0:
        counts.append(min(longest, total))
        total -= counts[-1]
    return counts


def replay(job: dict) -> None:
    import torch

    from aiter.ops.flydsl import linear_attention_prefill_kernels as k5

    # Per-process caches of the device the BV rule reads: a worker replays
    # rows of several cu_num in turn.
    k5._TUNED_BV_RUNTIME = None
    k5._HIPEQ_BV_CACHE.clear()

    def empty(*shape, dtype):
        return torch.empty(shape, dtype=dtype, device="cuda")

    dtype = getattr(torch, job["dtype"].split(".")[-1])
    state_dtype = torch.bfloat16 if job["state_bf16"] else torch.float32
    H, Hg, K, V, BT = job["H"], job["Hg"], job["K"], job["V"], job["BT"]
    lengths = [count * BT for count in _sequences(job)]
    if job["is_varlen"]:
        B, T, N = 1, sum(lengths), len(lengths)
    else:
        B, T, N = len(lengths), max(lengths), len(lengths)
    initial_state = indices = None
    if job["use_h0"]:
        initial_state = empty(N, H, V, K, dtype=state_dtype)
        if job["use_state_indices"]:
            indices = empty(N, dtype=torch.int32)
    g_shape = (B, H, T) if job["g_head_major"] else (B, T, H)
    k5.chunk_gated_delta_rule_fwd_h_flydsl_opt(
        empty(B, T, Hg, K, dtype=dtype),
        empty(B, H, T, K, dtype=dtype),
        empty(B, H, T, V, dtype=dtype),
        g=empty(*g_shape, dtype=torch.float32),
        initial_state=initial_state,
        output_final_state=job["store_fs"],
        chunk_size=BT,
        cu_seqlens=empty(N + 1, dtype=torch.int32) if job["is_varlen"] else None,
        seq_lens_cpu=lengths if job["is_varlen"] else None,
        state_dtype=state_dtype,
        initial_state_indices=indices,
        g_head_major=job["g_head_major"],
        snapshot_dtype=None if job["snapshot_bf16"] else torch.float32,
    )


def describe(job: dict) -> str:
    batch = (
        f"{job['m']} sequences"
        if "m" in job
        else f"chunks={job['total_chunks']}/{job['max_seq_chunks']}"
    )
    flags = " ".join(
        name
        for name in (
            "is_varlen",
            "use_h0",
            "store_fs",
            "snapshot_bf16",
            "state_bf16",
            "g_head_major",
            "use_state_indices",
        )
        if job[name]
    )
    return (
        f"chunk_gdn_h_opt K={job['K']} V={job['V']} BT={job['BT']} H={job['H']} "
        f"Hg={job['Hg']} {job['dtype']} {batch} {flags}"
    )


def config_env(csv_paths: list[str]) -> dict[str, str]:
    return {"AITER_CONFIG_GDN_K5_OPT": os.pathsep.join(csv_paths)}


def main() -> None:
    _replay.main(
        "aiter.aot.flydsl.chunk_gdn_h",
        "AOT pre-compile FlyDSL chunk-gdn-h opt (K5) kernels from aiter CSV config",
    )


if __name__ == "__main__":
    main()
