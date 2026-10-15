# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The tuning fits' view of the multi-GPU tests' bench CSVs.

``tune.sh`` runs the bench scenario of ``test_flydsl_one_shot_allreduce.py``
and ``test_flydsl_quick_allreduce.py`` with ``--candidates candidates.csv
--csv ...``. Those CSVs have one row per ``(tp, tokens, hidden)`` and one
``<candidate> <field>`` column per candidate and field. This module reads them
into the columns the fit scripts work in:

* ``TP``, ``M`` (tokens), ``K`` (hidden) and ``_nbytes``;
* the shipping engine and the stock kernels under the keys the fits report
  (``SHIPPING``), so that one key names one kernel across both tests;
* ``timing``, which is always ``graph``: the harness only replays graphs.

The pinned rows keep the names ``candidates.csv`` gave them, and
``candidates()`` maps a name back to its knobs.
"""

from __future__ import annotations

import os
import sys

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_MULTIGPU = os.path.dirname(_HERE)
if _MULTIGPU not in sys.path:
    sys.path.insert(0, _MULTIGPU)

import flydsl_allreduce_bench as fab

CANDIDATES_CSV = os.path.join(_HERE, "candidates.csv")

# ``(operation, column prefix in a test CSV) -> key``, for every column that
# is not a ``candidates.csv`` row.
SHIPPING = {
    ("ar", "flydsl"): "fly_1stage",
    ("ar", "mesh"): "fly_int4",
    ("ar", "ring"): "fly_int4_ring",
    ("ar", "hip"): "qr_int4",
    ("ar", "cdr"): "cdr",
    ("ar", "cdr_naive"): "cdr_naive",
    ("ar", "rccl"): "rccl",
    ("ar_rmsnorm", "flydsl"): "fused_fly_1stage",
    ("ar_rmsnorm", "mesh"): "fused_fly_mesh",
    ("ar_rmsnorm", "ring"): "fused_fly_ring",
    ("ar_rmsnorm", "cdr_fused_1stage"): "fused_cdr_1stage",
    ("ar_rmsnorm", "cdr_fused_2stage"): "fused_cdr_2stage",
    ("ar_rmsnorm", "cdr+norm"): "separate_cdr",
    ("ar_rmsnorm", "rccl+norm"): "separate_rccl",
    ("ar_rmsnorm", "hip_fused"): "fused_qr_int4",
    ("ar_rmsnorm", "hip+norm"): "separate_qr_int4",
}
# The key of each family's shipping engine, per operation.
AUTO_KEYS = {
    "ar": {"oneshot": "fly_1stage", "mesh": "fly_int4", "ring": "fly_int4_ring"},
    "ar_rmsnorm": {
        "oneshot": "fused_fly_1stage",
        "mesh": "fused_fly_mesh",
        "ring": "fused_fly_ring",
    },
}


def candidates(operation: str | None = None) -> list[fab.Candidate]:
    """Every row of ``candidates.csv``, or those of one *operation*."""
    ops = fab.OPERATIONS if operation is None else (operation,)
    out = []
    for op in ops:
        out += fab.load_candidates(
            CANDIDATES_CSV, operation=op, algorithms=("oneshot", "mesh", "ring")
        )
    return out


def read_bench_csv(path: str) -> pd.DataFrame:
    """One bench CSV from the multi-GPU tests, in the fits' columns."""
    df = pd.read_csv(path)
    if "operation" not in df.columns:
        raise SystemExit(
            f"{path}: no `operation` column; is this the --csv of a current "
            "test_flydsl_*_allreduce.py bench run?"
        )
    ops = set(df["operation"].dropna())
    if len(ops) != 1:
        raise SystemExit(f"{path}: expected one operation, got {sorted(ops)}")
    op = ops.pop()
    renames = {"tp": "TP", "tokens": "M", "hidden": "K"}
    for col in df.columns:
        prefix, _sep, field = col.rpartition(" ")
        if prefix.endswith(" spread"):
            prefix, field = prefix[: -len(" spread")], f"spread {field}"
        if prefix.endswith(" SQNR"):
            prefix, field = prefix[: -len(" SQNR")], f"SQNR {field}"
        key = SHIPPING.get((op, prefix))
        if key is not None:
            renames[col] = f"{key} {field}"
    # One concat rather than column inserts: a bench CSV has hundreds of columns.
    df = df.rename(columns=renames)
    extra = pd.DataFrame(
        {"_nbytes": df["M"] * df["K"] * 2, "timing": "graph"}, index=df.index
    )
    return pd.concat([df, extra], axis=1)


def auto_family(
    tp: int, nbytes: int, *, operation: str, arch: str, link: str, accuracy: str
) -> str | None:
    """The family the shipped policy routes *nbytes* to, or None when FlyDSL
    declines it (``cross_device_reduce``/RCCL, or the unfused path).

    ``accuracy`` is ``fast`` (quantization on, quick reduce at INT4) or
    ``exact``.
    """
    from aiter.ops.flydsl import allreduce_policy as P

    resolve = P.resolve_family_policy if operation == "ar" else P.resolve_fused_policy
    alg = resolve(world_size=int(tp), arch=arch, link=link).select(
        int(nbytes), quantized=accuracy == "fast"
    )
    return None if alg is None else alg.value
