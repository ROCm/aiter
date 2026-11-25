#!/usr/bin/env python3

# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""AOT pre-compilation for the gfx950 MiniMax-M3.1 TP4 QKV front (``aiter.ops.flydsl.kernels.qkv_chain``).

``qkv_chain_insert`` compiles the one-launch cooperative kernel per row tile (16 / 32 / 48 / 64 rows) for 1-64 rows,
and for 65-128 rows the split-K GEMM per row-table tile plus the reduce-insert kernel, so a serving process would
otherwise JIT each of them on its first decode batch of that size. This module replays the entry itself on FakeTensors
inside :func:`aiter.aot.flydsl.dry_run.dry_run` (see ``replay.py``) at 128 rows, bisected down to 1 row where the
runtime switches kernels. The split factor ``ks`` is the default (3); another ``ks`` is a caller's override.

Usage:
    python -m aiter.aot.flydsl.qkv_front
    python -m aiter.aot.flydsl.qkv_front --check
"""

from __future__ import annotations

import argparse

from aiter.aot.flydsl import replay as _replay

LABEL = "QKV_FRONT"
# No tuned table: one static op, jobs from the row range.
DEFAULT_CSVS: list[str] = []
AOT_ARCH_DEFAULT = "gfx950"
_CU_NUM = 256
_MIN_ROWS, _MAX_ROWS = 1, 128


def collect_jobs(_csv_paths: list[str], _args: argparse.Namespace | None) -> list:
    return [
        {
            "kernel_name": "qkv_front_n2560_k6144",
            "gfx": AOT_ARCH_DEFAULT,
            "cu_num": _CU_NUM,
            "m": _MAX_ROWS,
            "m_low": _MIN_ROWS,
        }
    ]


def replay(job: dict) -> None:
    import torch

    from aiter.ops.flydsl.kernels.qkv_chain import N, qkv_chain_insert

    def empty(*shape, dtype):
        return torch.empty(shape, dtype=dtype, device="cuda")

    m, k = job["m"], 6144
    qkv_chain_insert(
        empty(m, k, dtype=torch.float8_e4m3fn),
        empty(m, k // 32, dtype=torch.uint8),
        empty(N, k, dtype=torch.float8_e4m3fn),
        empty(N, k // 32, dtype=torch.uint8),
        m,
        *(empty(128, dtype=torch.bfloat16) for _ in range(4)),
        empty(128, 64, dtype=torch.bfloat16),
        *(empty(m, dtype=torch.int64) for _ in range(3)),
        empty(1, 2, 128, 72, dtype=torch.uint8),
        empty(1, 128, 72, dtype=torch.uint8),
        empty(1, dtype=torch.float32),
        empty(1, dtype=torch.float32),
        empty(m, 2048, dtype=torch.bfloat16),
        empty(m, 128, dtype=torch.float8_e4m3fn),
        eps=1e-6,
    )


def describe(job: dict) -> str:
    return f"{job['kernel_name']} {job['m']} rows"


def main() -> None:
    _replay.main(
        "aiter.aot.flydsl.qkv_front",
        "AOT pre-compile the gfx950 MiniMax-M3.1 TP4 QKV front",
    )


if __name__ == "__main__":
    main()
