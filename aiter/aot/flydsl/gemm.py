#!/usr/bin/env python3

# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""AOT pre-compilation for the FlyDSL kernels of tuned GEMM configs.

Each tuned row is replayed through the GEMM's own public entry --
``gemm_a8w8_bpreshuffle``, ``gemm_a16w16``, ``gemm_a4w4`` and so on, one per
table (see ``TABLES``) -- on FakeTensors inside
:func:`aiter.aot.flydsl.dry_run.dry_run`. The runtime looks the row up (at M
or a padded M), takes its libtype branch, picks and parameterizes the FlyDSL
kernel, sizes its workspaces and launches it as at inference, and the launch
compiles into the cache instead. This module only builds what a caller passes
in, so a kernel, its name parser or its launcher can change without an AOT
change.

A row tuned at M also serves the M between the previous tuned M of its shape
and M, and some compile-time parameters follow the actual M (a4w4's cluster_m
and bounded-M variants, mxfp8's split-K and cluster fallbacks), so every row
is replayed at its M and at the smaller M it serves. On gfx1250 every one of
those M is identified without compiling, and one M per set of kernels they
request is replayed (``SCAN_ARCHS``, ``replay.py``): bisecting between the two
ends would not do there, as cluster_m is the largest that divides M's tile
count, so the kernels change and change back as M grows. Rows of other archs
are replayed at the smallest M they serve and bisected between the two ends
where they request different kernels. Either way that also reaches the FlyDSL
heuristic kernels the runtime falls back to below a shape's smallest tuned M.

Rows are replayed for the arch their ``gfx`` column names (``cu_num`` for
legacy rows); with ``GPU_ARCHS`` naming targets, only those archs. The
runtime reads the tables the replay processes point it at: the ``--csv``
paths, grouped by table, or by default the tables the runtime itself merges.

Usage:
    python -m aiter.aot.flydsl.gemm
    python -m aiter.aot.flydsl.gemm --csv /path/to/a8w8_bpreshuffle_tuned_gemm.csv
    python -m aiter.aot.flydsl.gemm --check     # verify a cache, compile nothing

Environment variables:
    FLYDSL_RUNTIME_CACHE_DIR  Cache directory (default: ~/.flydsl/cache)
    GPU_ARCHS                 Archs to replay (default: every arch the tables have)
"""

from __future__ import annotations

import csv
import os
import re
from collections.abc import Callable
from dataclasses import dataclass

from aiter.aot.flydsl import replay as _replay
from aiter.jit.core import AITER_CONFIGS

LABEL = "GEMM"
AOT_ARCH_DEFAULT = "gfx950"
# Archs whose GEMMs pick kernels non-monotonically in M: cluster_m is fitted to
# the number of M tiles, and a ragged last tile is a variant of its own.
SCAN_ARCHS = ("gfx1250",)
# Tables without a cu_num column were tuned on a full part of their gfx.
_DEFAULT_CU_NUM = 256


def _empty(*shape, dtype):
    import torch

    return torch.empty(shape, dtype=dtype, device="cuda")


def _dtype(name: str):
    import torch

    return getattr(torch, str(name).strip().split(".")[-1])


def _int(value, default: int = 0) -> int:
    value = str(value or "").strip()
    return int(float(value)) if value else default


def _flag(value) -> bool:
    return str(value or "").strip().lower() in ("1", "true", "yes")


def _blocks(w_scale_block: str, default: str) -> tuple[int, int]:
    """(rows, cols) of a w_scale_block cell; a blank or 0 cell reads as
    ``default``, as the runtime's loader of that table reads it."""
    from aiter.ops.gemm_op_common import mxscale_w_scale_block_cell

    cell = mxscale_w_scale_block_cell(w_scale_block, default)
    rows, cols = (int(v) for v in cell.split("x"))
    return rows, cols


_PRESHUFFLE_OUT = re.compile(r"_(?:F8|I8)_(?:F8|I8)_(B16|F16)_")


def _out_dtype(kernel_name: str):
    """The output dtype a caller asks for: bf16, or the fp16 a preshuffle
    kernelName was tuned for."""
    from aiter.utility import dtypes

    match = _PRESHUFFLE_OUT.search(kernel_name)
    return dtypes.fp16 if match and match.group(1) == "F16" else dtypes.bf16


# --------------------------------------------------------------------------
# Callers, one per table: the inputs serving passes the public entry.
# --------------------------------------------------------------------------
def _a8w8_bpreshuffle(job: dict) -> None:
    from aiter.ops.gemm_op_a8w8 import gemm_a8w8_bpreshuffle
    from aiter.utility import dtypes

    m, n, k = job["m"], job["n"], job["k"]
    gemm_a8w8_bpreshuffle(
        _empty(m, k, dtype=dtypes.fp8),
        _empty(n, k, dtype=dtypes.fp8),
        _empty(m, 1, dtype=dtypes.fp32),
        _empty(n, 1, dtype=dtypes.fp32),
        dtype=_out_dtype(job["kernel_name"]),
    )


def _a8w8_blockscale_abpreshuffle(job: dict) -> None:
    from aiter.ops.gemm_op_a8w8 import gemm_a8w8_blockscale_abpreshuffle
    from aiter.utility import dtypes

    m, n, k = job["m"], job["n"], job["k"]
    # shuffle_mxfp8fp4_a packs row pairs, so an odd M arrives padded by one.
    gemm_a8w8_blockscale_abpreshuffle(
        _empty(m + (m & 1), k, dtype=dtypes.fp8),
        _empty(n, k, dtype=dtypes.fp8),
        _empty(m, k // 128, dtype=dtypes.fp8_e8m0),
        _empty(-(-n // 128), k // 128, dtype=dtypes.fp8_e8m0),
        dtype=dtypes.bf16,
    )


def _a8w8_mxfp8_bpreshuffle(job: dict) -> None:
    from aiter.ops.gemm_op_a8w8 import gemm_a8w8_mxfp8_bpreshuffle
    from aiter.utility import dtypes

    m, n, k = job["m"], job["n"], job["k"]
    rows, cols = _blocks(job["w_scale_block"], "1x32")
    x_block = 128 if cols == 128 else 32
    gemm_a8w8_mxfp8_bpreshuffle(
        _empty(m, k, dtype=dtypes.fp8),
        _empty(n, k, dtype=dtypes.fp8),
        _empty(m, k // x_block, dtype=dtypes.fp8_e8m0),
        _empty(n // rows, k // cols, dtype=dtypes.fp8_e8m0),
        dtype=dtypes.bf16,
    )


def _a8w8_blockscale_mxscale_bpreshuffle(job: dict) -> None:
    """E8M0 block scales on (16, 16)-preshuffled weights, with the x_scale
    block ``x_block``: 1x128 (blockscale, column-major bytes of the same shape)
    or 1x32 (group32/MX)."""
    from aiter.ops.gemm_op_a8w8 import gemm_a8w8_blockscale_bpreshuffle
    from aiter.utility import dtypes

    m, n, k = job["m"], job["n"], job["k"]
    rows, cols = _blocks(job["w_scale_block"], "128x128")
    x_block = job["x_block"]
    gemm_a8w8_blockscale_bpreshuffle(
        _empty(m, k, dtype=dtypes.fp8),
        _empty(n, k, dtype=dtypes.fp8),
        _empty(m, k // x_block, dtype=dtypes.fp8_e8m0),
        _empty(-(-n // rows), k // cols, dtype=dtypes.fp8_e8m0),
        dtype=dtypes.bf16,
    )


def _batched_a8w8_mxscale_bpreshuffle(job: dict) -> None:
    from aiter.ops.batched_gemm_op_a8w8 import batched_gemm_a8w8_mxscale_bpreshuffle
    from aiter.utility import dtypes

    b, m, n, k = job["b"], job["m"], job["n"], job["k"]
    rows, cols = _blocks(job["w_scale_block"], "128x128")
    batched_gemm_a8w8_mxscale_bpreshuffle(
        _empty(m, b, k, dtype=dtypes.fp8),
        _empty(b, n, k, dtype=dtypes.fp8),
        _empty(m, b, k // cols, dtype=dtypes.fp8_e8m0),
        _empty(b, n // rows, k // cols, dtype=dtypes.fp8_e8m0),
        dtype=dtypes.bf16,
    )


def _a16w16(job: dict) -> None:
    from aiter.tuned_gemm import gemm_a16w16

    m, n, k = job["m"], job["n"], job["k"]
    dtype, otype = _dtype(job["dtype"]), _dtype(job["outdtype"])
    weight = _empty(n, k, dtype=dtype)
    if _flag(job["bpreshuffle"]):
        weight.is_shuffled = True
    gemm_a16w16(
        _empty(m, k, dtype=dtype),
        weight,
        bias=_empty(n, dtype=dtype) if _flag(job["bias"]) else None,
        otype=otype,
    )


def _a4w4_mxscale_bpreshuffle(job: dict) -> None:
    from aiter.ops.gemm_op_a4w4 import gemm_a4w4
    from aiter.utility import dtypes

    m, n, k = job["m"], job["n"], job["k"]
    gemm_a4w4(
        _empty(m, k // 2, dtype=dtypes.fp4x2),
        _empty(n, k // 2, dtype=dtypes.fp4x2),
        _empty(-(-m // 32) * 32, k // 32, dtype=dtypes.fp8_e8m0),
        _empty(n, k // 32, dtype=dtypes.fp8_e8m0),
        dtype=dtypes.bf16,
    )


@dataclass(frozen=True)
class Table:
    """One tuned GEMM table: the runtime's name for it (``tuned_file_name``,
    which the runtime's merged and per-model files carry too), the variable
    that points the runtime at it, the columns that pick a row besides M, and
    the caller that replays one."""

    name: str
    env: str
    keys: tuple[str, ...]
    caller: Callable[[dict], None]
    # Replay every row, not only FlyDSL ones: M below a shape's smallest tuned
    # row, or a row the runtime cannot launch, falls back to a FlyDSL kernel.
    all_rows: bool = False


TABLES = (
    Table(
        "a8w8_bpreshuffle_tuned_gemm",
        "AITER_CONFIG_GEMM_A8W8_BPRESHUFFLE",
        ("gfx", "cu_num", "N", "K", "q_dtype_w"),
        _a8w8_bpreshuffle,
    ),
    Table(
        "a8w8_blockscale_abpreshuffle_tuned_gemm",
        "AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_ABPRESHUFFLE",
        ("gfx", "cu_num", "N", "K"),
        _a8w8_blockscale_abpreshuffle,
    ),
    Table(
        "a8w8_mxfp8_bpreshuffle_tuned_gemm",
        "AITER_CONFIG_GEMM_A8W8_MXFP8_BPRESHUFFLE",
        ("gfx", "cu_num", "N", "K", "w_scale_block"),
        _a8w8_mxfp8_bpreshuffle,
    ),
    Table(
        "a8w8_blockscale_mxscale_bpreshuffle_tuned_gemm",
        "AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_MXSCALE_BPRESHUFFLE",
        ("gfx", "cu_num", "N", "K", "w_scale_block"),
        _a8w8_blockscale_mxscale_bpreshuffle,
    ),
    Table(
        "batched_gemm_a8w8_blockscale_mxscale_bpreshuffle_tuned",
        "AITER_CONFIG_BATCHED_GEMM_A8W8_BLOCKSCALE_MXSCALE_BPRESHUFFLE",
        ("gfx", "b", "n", "k", "w_scale_block"),
        _batched_a8w8_mxscale_bpreshuffle,
    ),
    Table(
        "bf16_tuned_gemm",
        "AITER_CONFIG_GEMM_BF16",
        ("gfx", "cu_num", "N", "K", "bias", "dtype", "outdtype", "scaleAB")
        + ("bpreshuffle",),
        _a16w16,
    ),
    Table(
        "a4w4_mxscale_bpreshuffle_tuned_gemm",
        "AITER_CONFIG_GEMM_A4W4_MXSCALE_BPRESHUFFLE",
        ("gfx", "cu_num", "N", "K"),
        _a4w4_mxscale_bpreshuffle,
        all_rows=True,
    ),
)
_TABLES = {table.name: table for table in TABLES}


def _runtime_file(table: Table) -> str:
    """The file the runtime's lookup of ``table`` reads: the one its variable
    names, of any name, else the merged default."""
    return getattr(AITER_CONFIGS, f"{table.env}_FILE")


DEFAULT_CSVS = [_runtime_file(table) for table in TABLES]


def table_of(csv_path: str) -> Table | None:
    """The table a CSV belongs to: the one whose runtime lookup reads it, else
    the one whose name (the runtime's, which its merged and per-model files
    carry too) the file name holds, the longest that matches:
    ``bf16_tuned_gemm`` must not claim a batched table."""
    path = os.path.realpath(csv_path)
    for table in TABLES:
        if os.path.realpath(_runtime_file(table)) == path:
            return table
    base = os.path.basename(csv_path)
    names = [name for name in _TABLES if name in base]
    return _TABLES[max(names, key=len)] if names else None


def _column(row: dict, name: str) -> str:
    """A cell by name, M/N/K read case-blind (the batched table is lowercase)."""
    if name in row:
        return str(row[name] or "").strip()
    return str(row.get(name.lower(), row.get(name.upper(), "")) or "").strip()


def parse_csv(csv_path: str) -> list[dict]:
    """One job per FlyDSL row of a tuned GEMM table: its shape and key cells,
    ``m`` its tuned M and ``m_low`` the smallest M it serves (the previous
    tuned M of its shape, plus one)."""
    table = table_of(csv_path)
    if table is None:
        raise ValueError(
            f"{csv_path} is no FlyDSL GEMM table: no table's AITER_CONFIG_* "
            f"variable names it, and its file name holds none of {sorted(_TABLES)}"
        )
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(line for line in f if not line.startswith("#")))
    shapes: dict[tuple, list[int]] = {}
    for row in rows:
        key = tuple(_column(row, c) for c in table.keys)
        shapes.setdefault(key, []).append(_int(_column(row, "M")))

    jobs = []
    for row in rows:
        kernel_name = _column(row, "kernelName")
        if not table.all_rows and _column(row, "libtype") != "flydsl":
            continue
        key = tuple(_column(row, c) for c in table.keys)
        m = _int(_column(row, "M"))
        smaller = [other for other in shapes[key] if other < m]
        job = {
            "table": table.name,
            "kernel_name": kernel_name or _column(row, "libtype"),
            "gfx": _column(row, "gfx"),
            "cu_num": _int(_column(row, "cu_num"), _DEFAULT_CU_NUM),
            "m": m,
            "m_low": max(smaller, default=0) + 1,
            "n": _int(_column(row, "N")),
            "k": _int(_column(row, "K")),
        }
        for name in table.keys:
            job.setdefault(name, _column(row, name))
        if "b" in table.keys:
            job["b"] = _int(job["b"])
        jobs += [{**job, **variant} for variant in _variants(table, job)]
    return jobs


def _variants(table: Table, job: dict) -> list[dict]:
    """Caller-side choices a row leaves open: the x_scale block of an e8m0
    blockscale GEMM, which pairs with the weight's (1x128 blockscale with
    128x128 blocks, 1x32 group32 with 32x32 and 1x32 ones)."""
    if table.caller is _a8w8_blockscale_mxscale_bpreshuffle:
        _, cols = _blocks(job["w_scale_block"], "128x128")
        return [{"x_block": 128 if cols == 128 else 32}]
    return [{}]


def replay(job: dict) -> None:
    _TABLES[job["table"]].caller(job)


def describe(job: dict) -> str:
    shape = f"M={job['m']} N={job['n']} K={job['k']}"
    if "b" in job:
        shape = f"B={job['b']} {shape}"
    extra = [
        f"{name}={job[name]}"
        for name in (
            "w_scale_block",
            "x_block",
            "dtype",
            "outdtype",
            "bias",
            "bpreshuffle",
        )
        if job.get(name) not in (None, "", "False")
    ]
    return f"{job['table']} {job['kernel_name']}  {shape} {' '.join(extra)}".rstrip()


def config_env(csv_paths: list[str]) -> dict[str, str]:
    """Point each table's runtime lookup at the given CSVs of that table (it
    merges them as it does the per-model files). Tables none of them belong
    to keep the runtime's default."""
    paths: dict[str, list[str]] = {}
    for csv_path in csv_paths:
        table = table_of(csv_path)
        if table is not None:
            paths.setdefault(table.env, []).append(csv_path)
    return {env: os.pathsep.join(found) for env, found in paths.items()}


def main() -> None:
    _replay.main(
        "aiter.aot.flydsl.gemm",
        "AOT pre-compile the FlyDSL kernels of tuned GEMM configs",
    )


if __name__ == "__main__":
    main()
