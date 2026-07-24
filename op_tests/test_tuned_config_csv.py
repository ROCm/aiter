# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""
Validates that tuned-config CSVs contain no rows redundant under the
get_CKGEMM_config fallback chain (gl=None → gl=0 fine-grained → gl=1 coarse).

A row is redundant if removing it would still yield the same kernelName via
the fallback — it wastes space and can mask a wrong kernel being committed for
the gl=0/gl=1-aligned M value.

Redundancy mirrors get_CKGEMM_config exactly: the runtime probes keys
[gl=None (M itself), gl=0, gl=1] and returns the FIRST one that EXISTS
(kernel of the target is irrelevant to which probe wins). So a row is
redundant iff the first existing fallback target has the SAME kernel.

No GPU required; run with:
    pytest op_tests/test_tuned_config_csv.py -v
"""

import csv
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parents[1]

TUNED_CONFIG_CSVS = [
    "aiter/configs/model_configs/a8w8_blockscale_tuned_gemm_gfx1201.csv",
]


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


def _get_padded_m(M: int, N: int, K: int, gl: int) -> int:
    """Mirrors csrc/py_itfs_cu/gemm_common.cu:getPaddedM exactly."""
    if gl == 0:
        if M <= 256:
            return (M + 15) // 16 * 16
        if M <= 1024:
            return (M + 31) // 32 * 32
        if M <= 4096:
            return (M + 63) // 64 * 64
        return (M + 127) // 128 * 128
    if gl == 1:
        return 8192 if (M > 8192 and N > 4096) else _next_pow2(M)
    raise ValueError(f"unknown gl={gl}")


def _fallback_target(rows: dict, gfx, cu, M, N, K):
    """
    Mirror get_CKGEMM_config's probe order for a row's own shape: the runtime
    tries gl=0 then gl=1 and returns the FIRST key that EXISTS (kernel of the
    target is not consulted). Returns (padded_M, kernelName) of that target, or
    None if neither exists. The gl=None probe is the row itself and is excluded.
    """
    for gl in (0, 1):
        pm = _get_padded_m(M, N, K, gl)
        if pm == M:
            continue  # padded target is the row itself; runtime would re-hit it
        kernel = rows.get((gfx, cu, pm, N, K))
        if kernel is not None:
            return pm, kernel
    return None


def _find_redundant_rows(csv_path: str):
    """
    Returns list of (gfx, cu_num, M, N, K, reason) for rows whose runtime
    fallback target exists AND carries the same kernel — i.e. removing the row
    changes nothing at runtime.
    """
    rows: dict = {}
    with open(REPO_ROOT / csv_path) as f:
        for r in csv.DictReader(f):
            key = (
                r.get("gfx", ""),
                int(r["cu_num"]),
                int(r["M"]),
                int(r["N"]),
                int(r["K"]),
            )
            rows[key] = r["kernelName"]

    redundant = []
    for (gfx, cu, M, N, K), kernel in rows.items():
        target = _fallback_target(rows, gfx, cu, M, N, K)
        if target is not None and target[1] == kernel:
            pm = target[0]
            redundant.append(
                (gfx, cu, M, N, K, f"same kernel as existing padded M={pm}")
            )

    return redundant


@pytest.mark.parametrize("csv_path", TUNED_CONFIG_CSVS)
def test_no_redundant_rows(csv_path):
    redundant = _find_redundant_rows(csv_path)
    if redundant:
        lines = "\n".join(
            f"  gfx={g} cu_num={cu} M={M} N={N} K={K}  ({reason})"
            for g, cu, M, N, K, reason in redundant
        )
        pytest.fail(
            f"{csv_path}: {len(redundant)} redundant row(s) that are reachable via"
            f" fallback with same kernel:\n{lines}"
        )
