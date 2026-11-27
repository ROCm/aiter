# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Config search for the gfx1250 grouped-GEMM MoE tuner (``--grouped-gemm``).

One tuned CSV row drives two FlyDSL TDM GEMM kernels (gemm1: gate/up +
activation + requant, gemm2: down projection); grouped_moe_gfx1250 reads every
knob below from it. This module holds the knob space and the launch rules of
those kernels, a bounded coordinate-descent search, and the bench that times
candidates in a worker process, so a candidate that aborts the HSA queue only
costs a worker restart.

The pure-Python part (knobs, rules, search, table reads) imports neither torch
nor aiter, so it can be unit tested without a GPU.
"""

from __future__ import annotations

import csv
import itertools
import math
import os
import queue
import re
import shutil
import statistics
import tempfile
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

# CSV columns of each GEMM, by role.
STAGE_COLUMNS = {
    1: {
        "tile_m": "tile_m",
        "tile_n": "tile_n",
        "tile_k": "tile_k",
        "m_warp": "m_warp",
        "n_warp": "n_warp",
        "buffers": "num_buffers",
        "split_k": "split_k1",
        "cluster_m": "cluster_m",
        "cluster_n": "cluster_n",
        "wpt": "waves_per_tensor_tdm",
        "persistent": "persistent_workers",
    },
    2: {
        "tile_m": "tile_m2",
        "tile_n": "tile_n2",
        "tile_k": "tile_k2",
        "m_warp": "m_warp2",
        "n_warp": "n_warp2",
        "buffers": "num_buffer_stage2",
        "split_k": "split_k2",
        "cluster_m": "cluster_m2",
        "cluster_n": "cluster_n2",
        "wpt": "waves_per_tensor_tdm2",
        "persistent": "persistent_workers2",
    },
}
# Read by both kernels.
SHARED_COLUMNS = (
    "next_stage_prefetch",
    "tdm_as_in_prologue",
    "tdm_b_th",
    "lds_soa_load_interleave",
    "rolled_drain",
    "a_preshuffle",
)
KNOB_COLUMNS = (
    tuple(STAGE_COLUMNS[1].values()) + tuple(STAGE_COLUMNS[2].values()) + SHARED_COLUMNS
)

TILE_M = (16, 32, 64, 128, 256)
TILE_N = (64, 128, 256, 512)
TILE_K = (128, 256, 512)
WARPS = (1, 2, 4, 8)
CLUSTERS = ((1, 1), (2, 1), (4, 1), (1, 2), (1, 4), (2, 2))
LDS_BYTES = 320 * 1024  # gfx1250 LDS per workgroup
# Accumulator VGPRs per lane: the largest shipped tile uses 768 of the 1024;
# past that the operands no longer fit.
ACC_DWORDS = 768

# Environment overrides that replace CSV knobs inside grouped_moe_gfx1250 or
# the TDM launcher. The bench worker drops them, so a candidate runs as written.
KNOB_ENV = (
    "AITER_TDM_TILE_M",
    "AITER_TDM_TILE_N",
    "AITER_TDM_TILE_K",
    "AITER_TDM_NUM_BUFFERS",
    "AITER_TDM_TILE_M2",
    "AITER_TDM_TILE_N2",
    "AITER_TDM_TILE_K2",
    "AITER_TDM_NUM_BUFFERS2",
    "AITER_TDM_M_WARP",
    "AITER_TDM_N_WARP",
    "AITER_TDM_M_WARP2",
    "AITER_TDM_N_WARP2",
    "AITER_GROUPED_A4W4_A_PRESHUFFLE",
    "AITER_FLYDSL_MXFP4_CLUSTER_M",
    "AITER_FLYDSL_MXFP4_CLUSTER_N",
    "AITER_TDM_B_TH",
    "AITER_TDM_NEXT_STAGE_PREFETCH",
    "AITER_GROUPED_GEMM_AS_PROLOGUE",
    "AITER_FLYDSL_NUM_WAVES_PER_TENSOR_TDM",
    "AITER_FLYDSL_PERSISTENT_WORKERS",
    "AITER_FLYDSL_PERSISTENT_WORKERS1",
    "AITER_FLYDSL_PERSISTENT_WORKERS2",
    "AITER_FLYDSL_PLANAR_LDS",
    "AITER_FLYDSL_INTERLEAVED_LDS_LOAD",
    "AITER_FLYDSL_DISABLE_GEMM1_ACT",
    "AITER_FLYDSL_DISABLE_GEMM1_REQUANT",
)

# Columns _find_grouped_config matches on (blank = any).
LOOKUP_COLUMNS = (
    "gfx",
    "cu_num",
    "token",
    "model_dim",
    "inter_dim",
    "expert",
    "topk",
    "act_type",
    "dtype",
    "q_dtype_a",
    "q_dtype_w",
    "q_type",
    "ep_fused",
)

FP4 = "torch.float4_e2m1fn_x2"
FP8 = "torch.float8_e4m3fn"


def cell_text(value) -> str:
    """A CSV cell as the lookup compares it ("256.0" -> "256", nan -> "")."""
    text = "" if value is None else str(value).strip()
    if text.lower() in ("nan", "none", "<na>"):
        return ""
    try:
        number = float(text)
        if number == int(number):
            return str(int(number))
    except (ValueError, OverflowError):
        pass
    return text


def _cell_int(value, default: int) -> int:
    text = cell_text(value)
    return default if text == "" else int(float(text))


def resolve_config(row) -> dict[str, int]:
    """Knob values grouped_gemm_gfx1250_a8w4 runs a CSV row with: blank cells
    take the same defaults and stage-1 fallbacks as the runtime."""

    def get(column, default):
        return _cell_int(row.get(column), default)

    cfg = {"tile_m": get("tile_m", 64), "n_warp": get("n_warp", 4)}
    cfg["num_buffers"] = get("num_buffers", 2)
    cfg["tile_n"] = get("tile_n", cfg["n_warp"] * 64)
    cfg["tile_k"] = get("tile_k", 128)
    cfg["m_warp"] = get("m_warp", 1)
    for col2, col1 in (
        ("tile_m2", "tile_m"),
        ("tile_n2", "tile_n"),
        ("tile_k2", "tile_k"),
        ("m_warp2", "m_warp"),
        ("n_warp2", "n_warp"),
        ("num_buffer_stage2", "num_buffers"),
    ):
        cfg[col2] = get(col2, cfg[col1])
    cfg["cluster_m"] = get("cluster_m", -1)
    cfg["cluster_n"] = get("cluster_n", -1)
    cfg["cluster_m2"] = get("cluster_m2", -1)
    cfg["cluster_n2"] = get("cluster_n2", cfg["cluster_n"])
    cfg["waves_per_tensor_tdm"] = get("waves_per_tensor_tdm", -1)
    cfg["waves_per_tensor_tdm2"] = get(
        "waves_per_tensor_tdm2", cfg["waves_per_tensor_tdm"]
    )
    # Blank or 0 split_k means no split.
    cfg["split_k1"] = max(1, get("split_k1", 1))
    cfg["split_k2"] = max(1, get("split_k2", 1))
    for column in ("persistent_workers", "persistent_workers2") + SHARED_COLUMNS:
        cfg[column] = get(column, 0)
    return {c: cfg[c] for c in KNOB_COLUMNS}


@dataclass(frozen=True)
class Shape:
    token: int
    model_dim: int
    inter_dim: int
    expert: int
    topk: int
    a_fp4: bool
    cu_num: int = 256

    def dims(self, stage: int) -> tuple[int, int]:
        """(N, K) of a GEMM."""
        if stage == 1:
            return 2 * self.inter_dim, self.model_dim
        return self.model_dim, self.inter_dim


def shape_of(row, cu_num: int | None = None) -> Shape:
    return Shape(
        token=int(float(row["token"])),
        model_dim=int(float(row["model_dim"])),
        inter_dim=int(float(row["inter_dim"])),
        expert=int(float(row["expert"])),
        topk=int(float(row["topk"])),
        a_fp4=str(row["q_dtype_a"]).strip() == FP4,
        cu_num=int(cu_num or _cell_int(row.get("cu_num"), 256)),
    )


def _stage(cfg, stage):
    return {role: cfg[col] for role, col in STAGE_COLUMNS[stage].items()}


def _ceil(x: int, a: int) -> int:
    return -(-x // a) * a


def _effective_stage(cfg, shape: Shape, stage: int) -> tuple:
    """What the launcher actually builds (auto cluster, clamped buffers...)."""
    s = _stage(cfg, stage)
    n, k = shape.dims(stage)
    split_k = max(1, s["split_k"])
    buffers = min(s["buffers"], max(1, -(-k // s["tile_k"]) // split_k))
    n_tiles = -(-n // s["tile_n"])
    cn = s["cluster_n"] if s["cluster_n"] in (2, 3, 4) else 1
    cn = cn if n_tiles % cn == 0 else 1
    cm = s["cluster_m"]
    if cm < 1:
        cm = 4 if stage == 1 and cn == 4 else 1
    wpt = s["wpt"] if s["wpt"] in (1, 2, 4) else 2
    pw = _ceil(s["persistent"], cn) if s["persistent"] > 0 else 0
    return (
        s["tile_m"],
        s["tile_n"],
        s["tile_k"],
        s["m_warp"],
        s["n_warp"],
        buffers,
        split_k,
        cm,
        cn,
        wpt,
        pw,
    )


def effective_key(cfg, shape: Shape) -> tuple:
    """Configs with the same key build the same two kernels."""
    shared = tuple(int(cfg[c]) for c in SHARED_COLUMNS)
    return _effective_stage(cfg, shape, 1) + _effective_stage(cfg, shape, 2) + shared


def lds_bytes(cfg, shape: Shape, stage: int) -> int:
    """LDS one workgroup allocates, after build_gemm_a8w4_tdm_task."""
    eff = _effective_stage(cfg, shape, stage)
    tm, tn, tk, mw, nw, buffers, split_k = eff[:7]
    _, k = shape.dims(stage)
    a_fp4 = shape.a_fp4
    rep_m = tm // mw // 16
    a_row = tk // (2 if a_fp4 else 1) + (0 if cfg["a_preshuffle"] else 16)
    stage_a = _ceil(tm * a_row, 16)
    stage_b = _ceil((tn // 16) * (tk // 2) * 16, 16)
    stage_sa = _ceil(mw * (tk // 128) * rep_m * 16 * 4, 16)
    stage_sb = _ceil((tn // 32) * (tk // 4) * 4, 16)
    if cfg["lds_soa_load_interleave"]:
        b_off = _ceil(buffers * (stage_sa + stage_a + stage_sb), 65536)
        inputs = b_off + buffers * stage_b
    else:
        inputs = buffers * _ceil(stage_a + stage_b + stage_sa + stage_sb, 512)
    if cfg["tdm_as_in_prologue"]:
        inputs += _ceil(mw * (-(-k // 128)) * rep_m * 16 * 4, 128)
    arena = max(inputs, _ceil(tm * (tn + 16) * 2, 128))
    if split_k > 1:
        wmma_n = 32 if a_fp4 else 16
        partial = rep_m * (tn // nw // wmma_n) * (wmma_n // 2) * 4 * mw * nw * 32
        arena = _ceil(max(arena, partial), 512) * split_k
    return arena


def _check_stage(cfg, shape: Shape, stage: int) -> str | None:
    s = _stage(cfg, stage)
    n, k = shape.dims(stage)
    tm, tn, tk, mw, nw = (
        s[r] for r in ("tile_m", "tile_n", "tile_k", "m_warp", "n_warp")
    )
    split_k = s["split_k"]
    if min(tm, tn, tk, mw, nw, s["buffers"], split_k) < 1:
        return "non-positive tile, wave or buffer count"
    if n % tn:
        return f"tile_n {tn} does not divide N={n}"
    if k % 64 or tk % 128:
        return f"K={k} must be a multiple of 64 and tile_k={tk} of 128"
    if k % 128 and tk != 128:
        return "a half-k128 K tail needs tile_k 128"
    if tm % (16 * mw) or tn % (16 * nw):
        return f"tile {tm}x{tn} does not split into {mw}x{nw} waves"
    waves = mw * nw
    if waves * split_k * 32 > 1024:
        return "more than 1024 threads"
    warp_n = tn // nw
    if warp_n % 32:
        return "the warp tile N must be a multiple of 32"
    if stage == 1 and (warp_n // 16) % 4:
        return "the fused requant needs a warp tile N multiple of 64"
    rep_m = tm // mw // 16
    if rep_m != 1 and rep_m % 2:
        return "the warp tile M must be 16 or a multiple of 32"
    wmma_n = 32 if shape.a_fp4 else 16
    if rep_m * (warp_n // wmma_n) * (wmma_n // 2) > ACC_DWORDS:
        return "the accumulators exceed the VGPR budget"
    if s["wpt"] not in (-1, 0, 1, 2, 4):
        return "waves_per_tensor_tdm must be 1, 2 or 4"
    if s["cluster_m"] >= 1 and s["cluster_m"] not in (1, 2, 4):
        return "cluster_m must be 1, 2 or 4"
    eff = _effective_stage(cfg, shape, stage)
    cm, cn, wpt, pw = eff[7], eff[8], eff[9], eff[10]
    if wpt > waves or (4 * wpt) % waves:
        return f"{wpt} waves per tensor do not cover {waves} waves"
    if cfg["tdm_as_in_prologue"] and (waves != 4 or split_k > 1):
        return "tdm_as_in_prologue needs four waves and no split-K"
    if cm * cn > 16:
        return "a cluster holds at most 16 workgroups"
    if s["persistent"] < 0:
        return "negative persistent_workers"
    if pw and cm != 1:
        return "persistent workers need cluster_m 1"
    if split_k > 1 and (k % tk or (k // tk) % split_k or cm > 1 or cn > 1 or pw):
        return "split-K needs equal whole K slices and a plain grid"
    if lds_bytes(cfg, shape, stage) > LDS_BYTES:
        return f"needs {lds_bytes(cfg, shape, stage)} B of LDS"
    return None


def check_config(cfg, shape: Shape) -> str | None:
    """Why the grouped MoE would reject or fail to launch ``cfg``, or None.

    Mirrors the checks of grouped_moe_gfx1250, grouped_gemm_mxfp4 and the TDM
    kernel builder, plus LDS and accumulator budgets that otherwise only show
    up as a launch failure (which aborts the HSA queue).
    """
    if cfg["tdm_b_th"] not in range(7):
        return "tdm_b_th must be 0..6"
    for column in (
        "next_stage_prefetch",
        "tdm_as_in_prologue",
        "lds_soa_load_interleave",
        "rolled_drain",
        "a_preshuffle",
    ):
        if cfg[column] not in (0, 1):
            return f"{column} must be 0 or 1"
    if cfg["a_preshuffle"] and not shape.a_fp4:
        return "a_preshuffle needs fp4 activations"
    align = max(cfg["tile_m"], cfg["tile_m2"])
    if align % cfg["tile_m"] or align % cfg["tile_m2"]:
        return "tile_m and tile_m2 must both divide their max"
    for stage in (1, 2):
        why = _check_stage(cfg, shape, stage)
        if why:
            return f"gemm{stage}: {why}"
    return None


def _with(cfg, **changes):
    out = dict(cfg)
    out.update(changes)
    return out


def _cluster_m_cell(cm: int, cn: int, stage: int) -> int:
    # -1 is the table's spelling of "no M cluster", except where it would mean
    # the automatic 4 (gemm1 with cluster_n 4).
    if cm == 1 and not (stage == 1 and cn == 4):
        return -1
    return cm


def _tile_moves(cfg, shape, stage):
    col = STAGE_COLUMNS[stage]
    s = _stage(cfg, stage)
    tm, tn, tk, mw, nw = (
        s[r] for r in ("tile_m", "tile_n", "tile_k", "m_warp", "n_warp")
    )
    out = []

    def add(label, **roles):
        out.append((label, _with(cfg, **{col[r]: v for r, v in roles.items()})))

    # Halve or double one tile edge, keeping the wave grid when it still fits.
    for v in (tm // 2, tm * 2):
        if v in TILE_M:
            m = next(
                (m for m in (mw, mw // 2, mw * 2) if m in WARPS and v % (16 * m) == 0),
                None,
            )
            if m:
                add(f"tile_m {v} w{m}x{nw}", tile_m=v, m_warp=m)
    for v in (tn // 2, tn * 2):
        if v in TILE_N:
            n = next(
                (n for n in (nw, nw // 2, nw * 2) if n in WARPS and v % (32 * n) == 0),
                None,
            )
            if n:
                add(f"tile_n {v} w{mw}x{n}", tile_n=v, n_warp=n)
    for v in (tk // 2, tk * 2):
        if v in TILE_K:
            add(f"tile_k {v}", tile_k=v)
    for m, n in itertools.product(WARPS, WARPS):
        if (m, n) != (mw, nw) and 2 <= m * n <= 8:
            add(f"waves {m}x{n}", m_warp=m, n_warp=n)
    return out


def _cluster_moves(cfg, shape, stage):
    col = STAGE_COLUMNS[stage]
    return [
        (
            f"cluster {cm}x{cn}",
            _with(
                cfg,
                **{
                    col["cluster_m"]: _cluster_m_cell(cm, cn, stage),
                    col["cluster_n"]: cn if cn > 1 else -1,
                },
            ),
        )
        for cm, cn in CLUSTERS
    ]


def _value_moves(role, values):
    def moves(cfg, shape, stage):
        column = STAGE_COLUMNS[stage][role]
        return [(f"{role} {v}", _with(cfg, **{column: v})) for v in values(shape)]

    return moves


def _persistent_moves(cfg, shape, stage):
    col = STAGE_COLUMNS[stage]
    out = []
    for workers in (0, shape.cu_num, 2 * shape.cu_num):
        changes = {col["persistent"]: workers}
        if workers:
            # Persistent workers need cluster_m 1; keep the N cluster.
            cn = cfg[col["cluster_n"]]
            changes[col["cluster_m"]] = _cluster_m_cell(1, cn, stage)
        out.append((f"persistent {workers}", _with(cfg, **changes)))
    return out


def _split_k_moves(cfg, shape, stage):
    col = STAGE_COLUMNS[stage]
    plain = {col["cluster_m"]: -1, col["cluster_n"]: -1, col["persistent"]: 0}
    return [
        (
            f"split_k {kw}",
            _with(cfg, **{col["split_k"]: kw}, **(plain if kw > 1 else {})),
        )
        for kw in (1, 2, 4)
    ]


def _pipeline_moves(cfg, shape):
    return [
        (f"{c} {1 - int(cfg[c])}", _with(cfg, **{c: 1 - int(cfg[c])}))
        for c in (
            "rolled_drain",
            "next_stage_prefetch",
            "lds_soa_load_interleave",
            "tdm_as_in_prologue",
        )
    ]


def _b_th_moves(cfg, shape):
    return [(f"tdm_b_th {v}", _with(cfg, tdm_b_th=v)) for v in range(7)]


def _staged(fn, stage):
    def moves(cfg, shape):
        return [(f"gemm{stage} {label}", c) for label, c in fn(cfg, shape, stage)]

    return moves


_buffer_moves = _value_moves("buffers", lambda shape: (2, 3, 4))
_wpt_moves = _value_moves("wpt", lambda shape: (1, 2, 4))

# Knob groups in the order one descent pass visits them: the tiles move the
# most, the TDM hint the least.
GROUPS: tuple[tuple[str, Callable], ...] = (
    ("gemm1 tile", _staged(_tile_moves, 1)),
    ("gemm2 tile", _staged(_tile_moves, 2)),
    ("gemm1 cluster", _staged(_cluster_moves, 1)),
    ("gemm2 cluster", _staged(_cluster_moves, 2)),
    ("gemm1 buffers", _staged(_buffer_moves, 1)),
    ("gemm2 buffers", _staged(_buffer_moves, 2)),
    ("pipeline", _pipeline_moves),
    ("gemm1 persistent", _staged(_persistent_moves, 1)),
    ("gemm2 persistent", _staged(_persistent_moves, 2)),
    ("gemm1 waves/tensor", _staged(_wpt_moves, 1)),
    ("gemm2 waves/tensor", _staged(_wpt_moves, 2)),
    ("gemm1 split-k", _staged(_split_k_moves, 1)),
    ("gemm2 split-k", _staged(_split_k_moves, 2)),
    ("tdm_b_th", _b_th_moves),
)


# --------------------------------------------------------------------------
# Start points from the tuned tables


def read_rows(path: str) -> list[dict]:
    """Rows of a tuned CSV as text. Read-only: never goes through the AITER
    config merge, whose duplicate check rewrites source files."""
    try:
        with open(path, newline="") as f:
            return list(csv.DictReader(f))
    except (OSError, csv.Error, UnicodeError):
        return []


def lookup_key(row) -> dict[str, str]:
    key = {c: cell_text(row.get(c)) for c in LOOKUP_COLUMNS}
    key["ep_fused"] = "1" if key["ep_fused"] in ("1", "True", "true") else "0"
    return key


def row_matches(table_row, key: dict[str, str], *, token=True, cu_num=True) -> bool:
    """_find_grouped_config's match: a blank table cell matches anything."""
    for column, want in key.items():
        if (column == "token" and not token) or (column == "cu_num" and not cu_num):
            continue
        have = cell_text(table_row.get(column))
        if column == "ep_fused":
            have = "" if have == "" else ("1" if have in ("1", "True", "true") else "0")
        if have and have != want:
            return False
    return True


def _by_us(row):
    try:
        us = float(cell_text(row.get("us")) or 0)
    except ValueError:
        us = 0.0
    return (us <= 0, us)


def table_row_in_use(rows: list[dict], key: dict[str, str]) -> dict | None:
    """The row the runtime lookup picks for ``key`` (exact bucket, then the
    largest tuned bucket below it), or None."""
    for need_cu in (True, False):
        same = [r for r in rows if row_matches(r, key, token=False, cu_num=need_cu)]
        if same:
            break
    exact = [r for r in same if cell_text(r.get("token")) in ("", key["token"])]
    if not exact:
        tokens = [
            int(t) for t in (cell_text(r.get("token")) for r in same) if t.isdigit()
        ]
        below = [t for t in tokens if t < int(key["token"])]
        if not below:
            return None
        exact = [r for r in same if cell_text(r.get("token")) == str(max(below))]
    return min(exact, key=_by_us)


def _fit_default(cfg, shape):
    # Narrow like grouped_moe_gfx1250._fit_tile_n does for its built-in default.
    for stage in (1, 2):
        col = STAGE_COLUMNS[stage]
        n, k = shape.dims(stage)
        while n % cfg[col["tile_n"]] and cfg[col["tile_n"]] > 64:
            cfg[col["tile_n"]] //= 2
            if cfg[col["tile_n"]] // cfg[col["n_warp"]] < 64 and cfg[col["n_warp"]] > 1:
                cfg[col["n_warp"]] //= 2
        if k % 128:
            cfg[col["tile_k"]] = 128
    return cfg


def make_seeds(
    row, shape: Shape, tables: Iterable[str], gfx: str
) -> list[tuple[str, dict]]:
    """Start points: the row the runtime uses for this bucket today (or the
    runtime default when there is none), the rows tuned for neighbouring
    buckets of the same shape, and a tile_m sized to the rows per expert."""
    key = lookup_key(row)
    key["gfx"] = gfx
    key["cu_num"] = str(shape.cu_num)
    key["ep_fused"] = "0"
    rows = [r for path in dict.fromkeys(tables) for r in read_rows(path)]
    seeds = []
    current = table_row_in_use(rows, key)
    if current is not None:
        seeds.append(
            (
                f"table row (token {cell_text(current.get('token'))})",
                resolve_config(current),
            )
        )
    # The runtime miss default (grouped_gemm_gfx1250_a8w4 without a row).
    default = _fit_default(
        resolve_config(
            {"tile_m": 64, "tile_n": 256, "tile_k": 128, "n_warp": 4, "num_buffers": 3}
        ),
        shape,
    )
    seeds.append(("runtime default", default))
    # Same shape, any bucket; like the lookup, other cu_num rows when none match.
    same = [r for r in rows if row_matches(r, key, token=False)] or [
        r for r in rows if row_matches(r, key, token=False, cu_num=False)
    ]
    by_distance = sorted(
        (r for r in same if cell_text(r.get("token")).isdigit()),
        key=lambda r: (
            abs(math.log2(int(cell_text(r["token"])) / max(shape.token, 1))),
            _by_us(r),
        ),
    )
    for r in by_distance[:6]:
        seeds.append((f"token {cell_text(r['token'])} row", resolve_config(r)))
    per_expert = shape.token * shape.topk / max(shape.expert, 1)
    tile_m = min(128, max(16, 1 << max(0, math.ceil(math.log2(max(per_expert, 1))))))
    seeds.append(
        (
            f"tile_m {tile_m} for {per_expert:.0f} rows/expert",
            _fit_default(
                resolve_config(
                    {"tile_m": tile_m, "tile_n": 256, "tile_k": 256, "n_warp": 4}
                ),
                shape,
            ),
        )
    )
    out, seen = [], set()
    for i, (label, cfg) in enumerate(seeds):
        k = effective_key(cfg, shape)
        # The row in use stays the start point even if a rule here flags it.
        if k in seen or (i > 0 and check_config(cfg, shape)):
            continue
        seen.add(k)
        out.append((label, cfg))
    return out


# --------------------------------------------------------------------------
# Search


@dataclass
class SearchOptions:
    budget_s: float = 1800.0  # per shape, before the final whole-op check
    rounds: int = 5  # interleaved timing rounds per batch
    min_gain: float = 0.01  # a move must cut the median ratio by this much
    min_win: float = 0.75  # ... and win this share of the rounds
    batch: int = 8  # configs per interleaved batch, incumbent included
    passes: int = 2


def _median(values):
    return statistics.median(values) if values else math.inf


def score(result, base) -> tuple[float, float]:
    """(median per-round time ratio vs base, fraction of rounds won)."""
    ratios = [t / b for t, b in zip(result["total"], base["total"]) if b > 0]
    if not ratios:
        return math.inf, 0.0
    return _median(ratios), sum(r < 1 for r in ratios) / len(ratios)


def _stage_medians(result):
    return {name: round(_median(v), 2) for name, v in result.get("stages", {}).items()}


def _delta(cfg, base):
    return {c: cfg[c] for c in KNOB_COLUMNS if cfg[c] != base[c]}


def search(
    shape: Shape,
    seeds: list[tuple[str, dict]],
    measure: Callable[[list[dict], int], list[dict]],
    opts: SearchOptions,
    log: Callable[[str], None] = print,
    clock: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    """Coordinate descent and a small refine from the best seed and from the
    start point, the better of the two, then a confirmation.

    ``measure(configs, rounds)`` times the configs in interleaved rounds and
    returns per config ``{"ok", "error", "total": [us per round], "stages":
    {kernel group: [us per round]}}``. ``seeds[0]`` is the start point; the
    result is never a config that lost to it in the final confirmation, which
    re-times the two with twice the rounds.
    """
    t0 = clock()
    records: list[dict] = []
    stage_us: dict[tuple, dict] = {}
    measured: set[tuple] = set()

    def run(phase, group, cands, rounds):
        """cands: [(label, cfg)], scored against cands[0] (or the first that ran)."""
        results = measure([c for _, c in cands], rounds)
        for c in cands:
            measured.add(effective_key(c[1], shape))
        base = next((r for r in results if r.get("ok")), results[0])
        scored = []
        for (label, cfg), res in zip(cands, results):
            sc, win = (
                score(res, base)
                if res.get("ok") and base.get("ok")
                else (math.inf, 0.0)
            )
            if res.get("ok"):
                stage_us[effective_key(cfg, shape)] = _stage_medians(res)
            records.append(
                dict(
                    phase=phase,
                    group=group,
                    label=label,
                    ok=bool(res.get("ok")),
                    error=res.get("error", ""),
                    ratio=round(sc, 4) if math.isfinite(sc) else "",
                    win=round(win, 2),
                    total_us=round(_median(res["total"]), 2) if res.get("ok") else "",
                    **{f"{k}_us": v for k, v in _stage_medians(res).items()},
                    **cfg,
                )
            )
            scored.append((sc, win, label, cfg, res))
        return scored

    def over_budget():
        return clock() - t0 > opts.budget_s

    def taken(scored):
        # Faster by min_gain in the median, and in most rounds.
        return scored[0] <= 1 - opts.min_gain and scored[1] >= opts.min_win

    def descend(tag, inc_label, inc):
        """Coordinate descent, one knob group per batch, then a refine that
        combines moves which each gained less than min_gain."""
        tried = {effective_key(c, shape) for _, c in seeds}
        promising: list[tuple[float, dict]] = []
        history = [inc_label]
        log(f"{tag}: descent from {inc_label}")
        for p in range(opts.passes):
            improved = False
            for group, gen in GROUPS:
                if over_budget():
                    break
                moves, keys = [], set()
                for label, cfg in gen(inc, shape):
                    k = effective_key(cfg, shape)
                    if k in tried or k in keys or check_config(cfg, shape):
                        continue
                    keys.add(k)
                    moves.append((label, cfg))
                for chunk_at in range(0, len(moves), opts.batch - 1):
                    if over_budget():
                        break
                    chunk = moves[chunk_at : chunk_at + opts.batch - 1]
                    tried.update(effective_key(c, shape) for _, c in chunk)
                    scored = run(
                        f"{tag}.pass{p + 1}",
                        group,
                        [("incumbent", inc)] + chunk,
                        opts.rounds,
                    )
                    if not scored[0][4].get("ok"):
                        log(f"{group}: incumbent failed ({scored[0][4].get('error')})")
                        break
                    cands = [s for s in scored[1:] if s[4].get("ok")]
                    failed = len(scored) - 1 - len(cands)
                    best = min(cands, key=lambda s: s[0]) if cands else None
                    accept = best is not None and taken(best)
                    desc = (
                        f"best {best[2]} x{best[0]:.3f} (won {best[1]:.0%})"
                        if best
                        else "none ran"
                    )
                    log(
                        f"{tag} {group}: {len(chunk)} moves ({failed} failed), {desc}"
                        + (" -> accepted" if accept else "")
                        + f" [{clock() - t0:.0f}s]"
                    )
                    for sc, _, label, cfg, _ in cands:
                        if sc < 1 and not (accept and cfg is best[3]):
                            promising.append((sc, _delta(cfg, inc)))
                    if accept:
                        inc = best[3]
                        history.append(best[2])
                        improved = True
                        break
            if not improved or over_budget():
                break
        if promising and not over_budget():
            top = []
            for sc, d in sorted(promising, key=lambda x: x[0]):
                if d and d not in top and any(inc[c] != v for c, v in d.items()):
                    top.append(d)
            top = top[:4]
            combos = [(d, _with(inc, **d)) for d in top]
            for a, b in itertools.combinations(top[:3], 2):
                if not set(a) & set(b):
                    combos.append(({**a, **b}, _with(inc, **a, **b)))
            cands, keys = [], set()
            for d, cfg in combos:
                k = effective_key(cfg, shape)
                if k not in tried and k not in keys and not check_config(cfg, shape):
                    keys.add(k)
                    label = "refine " + ",".join(f"{c}={v}" for c, v in d.items())
                    cands.append((label, cfg))
            if cands:
                cands = cands[: opts.batch - 1]
                scored = run(
                    f"{tag}.refine",
                    "refine",
                    [("incumbent", inc)] + cands,
                    2 * opts.rounds,
                )
                ok = [s for s in scored[1:] if s[4].get("ok")]
                if scored[0][4].get("ok") and ok:
                    best = min(ok, key=lambda s: s[0])
                    if taken(best):
                        log(f"{tag} refine: {best[2]} x{best[0]:.3f} -> accepted")
                        inc = best[3]
                        history.append(best[2])
        return inc, history

    # Seeds: the fastest becomes the incumbent. A start point that fails here
    # (e.g. a row this build rejects) hands over to the first seed that runs.
    scored = run("seed", "seeds", seeds[: opts.batch], opts.rounds)
    ok = [s for s in scored if s[4].get("ok")]
    if not ok:
        return {
            "ok": False,
            "error": "; ".join(str(s[4].get("error")) for s in scored)[:400],
            "records": records,
        }
    if not scored[0][4].get("ok"):
        log(
            f"start point failed ({scored[0][4].get('error')}); starting from {ok[0][2]}"
        )
    start_label, start = ok[0][2], ok[0][3]
    best = min(ok, key=lambda s: s[0])
    log("seeds: " + ", ".join(f"{s[2]} x{s[0]:.3f}" for s in ok))

    # Descend from the best seed and, when another seed won, also from the
    # start point: a row tuned for this bucket can lead to a better optimum
    # than the seed that is fastest as is.
    origins = [(start_label, start)]
    if best[0] <= 1 - opts.min_gain:
        origins.insert(0, (best[2], best[3]))
    ends = []
    for i, (label, cfg) in enumerate(origins):
        if i and over_budget():
            break
        ends.append(descend(f"d{i + 1}", label, cfg))
    inc, history = ends[0]
    if len(ends) > 1 and effective_key(ends[0][0], shape) != effective_key(
        ends[1][0], shape
    ):
        scored = run(
            "multi-start",
            "multi-start",
            [(f"from {h[0]}", c) for c, h in ends],
            2 * opts.rounds,
        )
        if scored[1][4].get("ok") and (not scored[0][4].get("ok") or scored[1][0] < 1):
            inc, history = ends[1]
        log(
            f"multi-start: from {ends[1][1][0]} x{scored[1][0]:.3f} vs from "
            f"{ends[0][1][0]}; keeping the descent from {history[0]}"
        )

    # Confirmation: the start point against the result, fresh rounds (the
    # winner of many comparisons is also the one noise favoured most).
    finals = [(start_label, start)]
    if effective_key(inc, shape) != effective_key(start, shape):
        finals.append(("searched", inc))
    results = measure([c for _, c in finals], 2 * opts.rounds)
    pick = 0
    if not results[0].get("ok") and len(finals) > 1 and results[1].get("ok"):
        pick = 1
    elif len(finals) > 1 and results[0].get("ok") and results[1].get("ok"):
        sc, win = score(results[1], results[0])
        log(f"confirm: searched vs start x{sc:.3f} (won {win:.0%})")
        if taken((sc, win)):
            pick = 1
    for (label, cfg), res in zip(finals, results):
        if res.get("ok"):
            stage_us[effective_key(cfg, shape)] = _stage_medians(res)
        records.append(
            dict(
                phase="confirm",
                group="final",
                label=label,
                ok=bool(res.get("ok")),
                error=res.get("error", ""),
                total_us=round(_median(res["total"]), 2) if res.get("ok") else "",
                **{f"{k}_us": v for k, v in _stage_medians(res).items()},
                **cfg,
            )
        )
    chosen = results[pick]
    if not chosen.get("ok"):
        return {
            "ok": False,
            "error": f"whole-op run failed: {chosen.get('error')}",
            "records": records,
        }
    label, cfg = finals[pick]
    out = {
        "ok": True,
        "config": cfg,
        "label": label,
        "history": history,
        "us": round(_median(chosen["total"]), 4),
        "start_us": (
            round(_median(results[0]["total"]), 4) if results[0].get("ok") else None
        ),
        "stage_us": stage_us.get(effective_key(cfg, shape), {}),
        "measured": len(measured),
        "seconds": round(clock() - t0, 1),
        "records": records,
    }
    return out


# --------------------------------------------------------------------------
# GPU bench: a worker process per GPU


def unsupported_reason(row) -> str | None:
    """Why the grouped bench cannot time an untuned row, or None."""
    act = str(row.get("act_type", "")).split(".")[-1]
    if act not in ("Silu", "Swiglu", "Situv2"):
        return f"activation {act} is not served by the grouped MoE"
    if str(row.get("q_type", "")).split(".")[-1] != "per_1x32":
        return "the grouped MoE serves per_1x32 MXFP4 only"
    if cell_text(row.get("use_g1u1")) not in ("1", "True", "true"):
        return "the grouped MoE needs fused gate/up weights (use_g1u1=1)"
    if str(row.get("q_dtype_w", "")).strip() != FP4:
        return "the grouped MoE needs MXFP4 weights"
    if str(row.get("q_dtype_a", "")).strip() not in (FP4, FP8):
        return "the grouped MoE needs fp4 or fp8 activations"
    if cell_text(row.get("ep_fused")) in ("1", "True", "true"):
        return (
            "ep_fused rows time GEMM2's EP scatter, which needs a multi-rank arena; "
            "use op_tests/multigpu_tests/bench_mega_moe.py"
        )
    if _cell_int(row.get("topk"), -1) < 1:
        return "EP rows (topk -1) need the EP routing of the multi-GPU bench"
    return None


# CSV column -> keyword of the captured launch, to prove the row was applied.
_LAUNCH_ARGS = {
    "gemm1": {
        "tile_m": "tile_m",
        "tile_n": "tile_n",
        "tile_k": "tile_k",
        "m_warp": "m_warp",
        "n_warp": "n_warp",
        "num_buffers": "num_buffers",
        "cluster_m": "cluster_m",
        "cluster_n": "cluster_n",
        "waves_per_tensor_tdm": "waves_per_tensor_tdm",
        "persistent_workers": "persistent_workers",
        "k_waves": "split_k1",
    },
    "gemm2": {
        "tile_m": "tile_m2",
        "tile_n": "tile_n2",
        "tile_k": "tile_k2",
        "m_warp": "m_warp2",
        "n_warp": "n_warp2",
        "num_buffers": "num_buffer_stage2",
        "cluster_m": "cluster_m2",
        "cluster_n": "cluster_n2",
        "waves_per_tensor_tdm": "waves_per_tensor_tdm2",
        "persistent_workers": "persistent_workers2",
        "k_waves": "split_k2",
    },
}
_SHARED_LAUNCH_ARGS = (
    "next_stage_prefetch",
    "tdm_as_in_prologue",
    "tdm_b_th",
    "lds_soa_load_interleave",
    "rolled_drain",
)


def kernel_group(name: str) -> str:
    """gemm1 / gemm2 / other (route, quant, reduce) from a kernel name: both
    GEMMs are mxfp4_tdm_* kernels, gemm1 carries its activation (_act<n>)."""
    if name.startswith("mxfp4_tdm"):
        return "gemm1" if re.search(r"_act\d", name) else "gemm2"
    return "other"


def launch_mismatch(calls: dict, cfg: dict) -> str | None:
    """Compare the captured GEMM launches with the row; None if they agree."""
    for stage, columns in _LAUNCH_ARGS.items():
        fn = calls.get(stage)
        if fn is None:
            return f"no {stage} launch captured (grouped path not taken)"
        kw = getattr(fn, "keywords", {})
        for arg, column in {**columns, **{c: c for c in _SHARED_LAUNCH_ARGS}}.items():
            if arg in kw and int(kw[arg]) != int(cfg[column]):
                return f"{stage} ran {arg}={kw[arg]} but the row has {column}={cfg[column]}"
    return None


class _Worker:
    """Owns one shape's tensors on one GPU and times candidate rows on one or
    more cases (token count x routing); a candidate's time is the cases' sum."""

    def __init__(self, cfg_dir):
        import torch

        from aiter.jit.utils.chip_info import get_cu_num, get_gfx

        self.torch = torch
        self.cfg_dir = cfg_dir
        self.serial = 0
        self.gfx = str(get_gfx())
        self.cu_num = int(get_cu_num())
        self.cases = []
        self.kwargs = {}
        self.refs = None
        self.ref_key = None

    def identity(self):
        return {"gfx": self.gfx, "cu_num": self.cu_num}

    # -- inputs
    def setup(self, row, routing, seed, tokens=None):
        import torch

        from aiter import ActivationType, QuantType
        from aiter.ops.flydsl.moe_common import GateMode
        from aiter.ops.shuffle import moe_shuffle_scale, moe_shuffle_weight
        from aiter.utility.dtypes import str2ActivationType

        self.cases, self.refs, self.ref_key = [], None, None
        torch.cuda.empty_cache()
        d, i = int(float(row["model_dim"])), int(float(row["inter_dim"]))
        e, k = int(float(row["expert"])), int(float(row["topk"]))
        dev = "cuda"
        g = torch.Generator(device=dev).manual_seed(seed)
        u8 = torch.uint8
        # Random MXFP4 bytes with e8m0 scales 2^-9..2^-1 keep the GEMMs finite;
        # GUGU weights and n32k4 scales are the layouts the grouped path reads.
        w1 = torch.randint(
            0, 256, (e, 2 * i, d // 2), dtype=u8, device=dev, generator=g
        )
        w2 = torch.randint(0, 256, (e, d, i // 2), dtype=u8, device=dev, generator=g)
        s1 = torch.randint(
            118, 127, (e * 2 * i, d // 32), dtype=u8, device=dev, generator=g
        )
        s2 = torch.randint(
            118, 127, (e * d, i // 32), dtype=u8, device=dev, generator=g
        )
        w1 = moe_shuffle_weight(w1, experts_cnt=e, is_guinterleave=True, gate_up=True)
        w2 = moe_shuffle_weight(w2, experts_cnt=e, is_guinterleave=True, gate_up=False)
        s1 = moe_shuffle_scale(s1, e, is_guinterleave=True, gate_up=True)
        s2 = moe_shuffle_scale(s2, e, is_guinterleave=True, gate_up=False)
        dtype = getattr(torch, str(row.get("dtype", "torch.bfloat16")).split(".")[-1])
        info = []
        for m in tokens or [int(float(row["token"]))]:
            hidden = (torch.randn(m, d, device=dev, generator=g) / 4).to(dtype)
            for kind in str(routing).split(","):
                ids, weights = make_routing(m, e, k, kind, g)
                self.cases.append((hidden, w1, w2, weights, ids))
                counts = torch.bincount(ids.reshape(-1).long(), minlength=e)
                info.append(
                    f"M={m} {kind}: max {int(counts.max())}/expert, "
                    f"{int((counts == 0).sum())} empty"
                )
        act = str2ActivationType(str(row["act_type"]).split(".")[-1])
        assert act in (
            ActivationType.Silu,
            ActivationType.Swiglu,
            ActivationType.Situv2,
        )
        # Activation parameters, when the untuned row carries them.
        extra = {
            c: float(cell_text(row[c]))
            for c in ("beta", "linear_beta", "swiglu_limit")
            if cell_text(row.get(c))
        }
        self.kwargs = dict(
            activation=act,
            quant_type=QuantType.per_1x32,
            doweight_stage1=cell_text(row.get("doweight_stage1")) in ("1", "True"),
            w1_scale=s1,
            w2_scale=s2,
            dtype=dtype,
            # The grouped kernels read GUGU weights only, so they serve
            # GateMode.INTERLEAVE; the lookup does not key on gate_mode.
            gate_mode=GateMode.INTERLEAVE.value,
            # The row's activation dtype, not the env-dependent default.
            quant_dtype_a=getattr(torch, str(row["q_dtype_a"]).split(".")[-1]),
            **extra,
        )
        torch.cuda.synchronize()
        return {"cases": info, **self.identity()}

    # -- one candidate
    def _write(self, cfg):
        # Key cells blank (match any lookup), so the candidate is always the
        # row selected; launch_mismatch() proves it.
        path = os.path.join(self.cfg_dir, f"cand{self.serial}.csv")
        self.serial += 1
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(KNOB_COLUMNS) + ["us"])
            w.writeheader()
            w.writerow({**{c: int(cfg[c]) for c in KNOB_COLUMNS}, "us": 1})
        return path

    def _select(self, path):
        from aiter.ops.flydsl import grouped_moe_gfx1250 as grouped

        # A single private file: the runtime reads it as is, no merge.
        os.environ["AITER_CONFIG_GROUPED_FMOE"] = path
        grouped._GROUPED_CONFIG_CACHE.clear()
        for fn in ("_find_grouped_config", "_widest_grouped_tile"):
            cached = getattr(grouped, fn, None)
            if hasattr(cached, "cache_clear"):
                cached.cache_clear()

    def _call(self, case=0):
        from aiter.fused_moe import fused_moe

        return fused_moe(*self.cases[case], **self.kwargs)

    def _run(self, cfg, capture):
        """Run ``cfg`` once per case; capture case 0's launches."""
        from aiter.ops.flydsl import grouped_moe_gfx1250 as grouped

        path = self._write(cfg)
        self._select(path)
        grouped.kernel_bench_callable = [] if capture else None
        try:
            outs = [self._call(0)]
            calls = dict(grouped.kernel_bench_callable or [])
        finally:
            grouped.kernel_bench_callable = None
        outs += [self._call(c) for c in range(1, len(self.cases))]
        self.torch.cuda.synchronize()
        return path, outs, calls

    def _diff(self, outs):
        """Largest output difference to the reference over the cases."""
        torch = self.torch
        worst = 0.0
        for c, out in enumerate(outs):
            if not bool(torch.isfinite(out).all()):
                return math.inf
            if self.refs is not None:
                a, b = self.refs[c].float(), out.float()
                den = float((a * a).sum() + (b * b).sum())
                if den > 0:
                    worst = max(worst, 1 - 2 * float((a * b).sum()) / den)
        return worst

    def _ensure_ref(self, cfg):
        key = tuple(sorted(cfg.items()))
        if self.refs is not None and self.ref_key == key:
            return
        self.refs = None
        # A start row that fails is reported per candidate; the first candidate
        # that runs becomes the reference instead.
        try:
            _, outs, _ = self._run(cfg, capture=False)
        except Exception:  # noqa: BLE001
            return
        if all(bool(self.torch.isfinite(o).all()) for o in outs):
            self.refs, self.ref_key = [o.clone() for o in outs], key

    # -- timing
    def _profile(self, case, iters):
        """Device time per fused_moe call, by kernel group, over ``iters``
        eager calls. Device times leave host launch gaps out, and eager calls
        reuse the allocator's blocks, so every candidate runs on the same
        buffers (separately captured buffers skew single kernels by a few %)."""
        from torch.profiler import ProfilerActivity, profile

        torch = self.torch
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(iters):
                self._call(case)
            torch.cuda.synchronize()
        groups = {"gemm1": 0.0, "gemm2": 0.0, "other": 0.0}
        for e in prof.events():
            if "CUDA" not in str(e.device_type):
                continue
            us = getattr(e, "device_time_total", None)
            if us is None:
                us = e.cuda_time_total
            groups[kernel_group(e.name)] += us
        return {k: v / iters for k, v in groups.items()}

    def measure(self, conn, configs, rounds, ref_cfg, err_ratio, warmup):
        torch = self.torch
        self._ensure_ref(ref_cfg)
        results = [{"ok": False, "error": "not run"} for _ in configs]
        live = []
        for idx, cfg in enumerate(configs):
            conn.send(("start", idx))
            try:
                path, outs, calls = self._run(cfg, capture=True)
                why = launch_mismatch(calls, cfg)
                del calls
                finite = all(bool(torch.isfinite(o).all()) for o in outs)
                if self.refs is None and why is None and finite:
                    self.refs = [o.clone() for o in outs]
                diff = self._diff(outs)
                del outs
                if why is None and not diff <= err_ratio:
                    why = f"output differs from the start row (diff {diff:.3g})"
                if why:
                    results[idx]["error"] = why
                    continue
                live.append((idx, path, diff))
            except torch.cuda.OutOfMemoryError:
                results[idx]["error"] = "out of memory"
                torch.cuda.empty_cache()
            # A per-candidate failure, e.g. a kernel the compiler rejects. A
            # launch the GPU refuses also kills the context: report and exit.
            except Exception as exc:  # noqa: BLE001
                results[idx]["error"] = f"{type(exc).__name__}: {exc}"[:300]
                if context_lost(torch, exc):
                    conn.send(("lost", results[idx]["error"]))
                    os._exit(3)
        conn.send(("timing", None))
        if live:
            self._select(live[0][1])
            iters = []
            for case in range(len(self.cases)):
                for _ in range(max(1, warmup)):
                    self._call(case)
                torch.cuda.synchronize()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                self._call(case)
                end.record()
                end.synchronize()
                # ~10 ms per sample: long enough to average out the clock
                # swings of a power-capped GPU.
                call_us = max(start.elapsed_time(end) * 1e3, 1.0)
                iters.append(int(min(100, max(3, math.ceil(10000.0 / call_us)))))
            samples = {idx: [] for idx, _, _ in live}
            # Interleaved rounds, rotating the order, so drift hits all alike.
            for r in range(rounds):
                order = live[r % len(live) :] + live[: r % len(live)]
                if r % 2:
                    order.reverse()
                for idx, path, _ in order:
                    self._select(path)
                    groups = {}
                    for case, n in enumerate(iters):
                        for _ in range(2):
                            self._call(case)
                        for kname, us in self._profile(case, n).items():
                            groups[kname] = groups.get(kname, 0.0) + us
                    samples[idx].append(groups)
            for idx, _, diff in live:
                stages = {k: [s[k] for s in samples[idx]] for k in samples[idx][0]}
                total = [sum(s.values()) for s in samples[idx]]
                results[idx] = {
                    "ok": True,
                    "stages": stages,
                    "total": total,
                    "diff": diff,
                }
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        return results


def make_routing(m, e, k, kind, gen):
    """topk ids/weights: balanced (round robin), uniform, or zipf:<s>."""
    import torch

    dev = gen.device
    if kind == "balanced":
        tok = torch.arange(m, device=dev).view(m, 1)
        ids = (tok * k + torch.arange(k, device=dev).view(1, k)) % e
    else:
        if kind == "uniform":
            logits = torch.zeros(m, e, device=dev)
        elif kind.startswith("zipf:"):
            s = float(kind.split(":", 1)[1])
            rank = torch.empty(e, device=dev)
            rank[torch.randperm(e, device=dev, generator=gen)] = torch.arange(
                1, e + 1, device=dev, dtype=torch.float32
            )
            logits = (-s * torch.log(rank)).view(1, e).expand(m, e)
        else:
            raise ValueError(f"unknown routing {kind!r}")
        # Gumbel top-k: k distinct experts per token, sampled without replacement.
        u = torch.rand(m, e, device=dev, generator=gen).clamp_(1e-9, 1 - 1e-9)
        ids = (logits - torch.log(-torch.log(u))).topk(k, dim=1).indices
    weights = torch.rand(m, k, device=dev, generator=gen) + 0.5
    weights = weights / weights.sum(1, keepdim=True)
    return ids.to(torch.int32).contiguous(), weights.float().contiguous()


def check_routing(kinds: str) -> str:
    """A comma list of balanced, uniform and zipf:<s>."""
    for kind in kinds.split(","):
        if kind.startswith("zipf:"):
            float(kind.split(":", 1)[1])
        elif kind not in ("balanced", "uniform"):
            raise ValueError(
                f"routing must be balanced, uniform or zipf:<s>, got {kind!r}"
            )
    return kinds


def bucket_tokens(token: int, fractions: Iterable[float]) -> list[int]:
    """Token counts to time a bucket at: fractions of it, in (0.5, 1] so that
    they still map to the bucket (the runtime pads M to the next power of 2)."""
    out = []
    for f in fractions:
        if not 0.5 < f <= 1.0:
            raise ValueError(f"token fractions must be in (0.5, 1], got {f}")
        m = max(1, round(f * token))
        if m not in out:
            out.append(m)
    return out


_ENV_LOCK = threading.Lock()


def context_lost(torch, exc) -> bool:
    """Did ``exc`` leave the HIP context unusable (launch out of resources,
    queue abort, memory fault)? Only a new process recovers from that."""
    if type(exc).__name__ == "AcceleratorError":
        return True
    try:
        torch.cuda.synchronize()
    except Exception:  # noqa: BLE001
        return True
    return False


def _worker_main(conn):
    dropped = [k for k in KNOB_ENV if os.environ.pop(k, None) is not None]
    cfg_dir = tempfile.mkdtemp(prefix="aiter_grouped_tune_")
    # Pinned to a private file for the worker's life: the AITER config merge
    # (and its duplicate rewrite of source tables) never runs here.
    os.environ["AITER_CONFIG_GROUPED_FMOE"] = os.path.join(cfg_dir, "none.csv")
    os.environ["AITER_USE_GROUPED_GEMM"] = "1"
    import torch

    torch.cuda.set_device(0)
    worker = _Worker(cfg_dir)
    conn.send(("ready", {"dropped_env": dropped, **worker.identity()}))
    while True:
        try:
            op, kw = conn.recv()
        except EOFError:
            op = "close"
        if op == "close":
            shutil.rmtree(cfg_dir, ignore_errors=True)
            return
        try:
            if op == "setup":
                conn.send(("ok", worker.setup(**kw)))
            elif op == "measure":
                conn.send(("ok", worker.measure(conn, **kw)))
            else:
                conn.send(("error", f"unknown op {op}"))
        except Exception as exc:  # noqa: BLE001  reported to the parent
            conn.send(("error", f"{type(exc).__name__}: {exc}"[:400]))
            if context_lost(torch, exc):
                os._exit(3)


class WorkerLost(RuntimeError):
    pass


class Bench:
    """Parent side of one GPU's worker; restarts it after a crash or hang."""

    def __init__(self, gpu, *, timeout=1800.0, err_ratio=0.05, warmup=3, log=print):
        import multiprocessing as mp

        self.ctx = mp.get_context("spawn")
        self.gpu = gpu
        self.timeout = timeout
        self.err_ratio = err_ratio
        self.warmup = warmup
        self.log = log
        self.proc = None
        self.conn = None
        self.setup_kw = None
        self.ref_cfg = None

    def _start(self):
        parent, child = self.ctx.Pipe()
        self.proc = self.ctx.Process(target=_worker_main, args=(child,), daemon=True)
        # The worker sees only its GPU (as device 0); env is read at start.
        with _ENV_LOCK:
            name = next(
                (
                    v
                    for v in ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES")
                    if os.environ.get(v)
                ),
                "HIP_VISIBLE_DEVICES",
            )
            old = os.environ.get(name)
            visible = old.split(",") if old else None
            os.environ[name] = visible[self.gpu] if visible else str(self.gpu)
            try:
                self.proc.start()
            finally:
                if old is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = old
        child.close()
        self.conn = parent
        _, info = self._recv()
        if info.get("dropped_env"):
            self.log(f"worker ignores knob overrides {info['dropped_env']}")
        self.setup_info = None
        if self.setup_kw is not None:
            self.setup_info = self._request("setup", self.setup_kw)

    def _recv(self):
        try:
            if not self.conn.poll(self.timeout):
                raise WorkerLost(f"no reply in {self.timeout:.0f}s")
            return self.conn.recv()
        except (EOFError, OSError) as exc:
            raise WorkerLost(f"worker died ({type(exc).__name__})") from exc

    def _request(self, op, kw):
        self.conn.send((op, kw))
        kind, payload = self._recv()
        if kind in ("error", "lost"):
            raise RuntimeError(payload)
        return payload

    def _kill(self):
        if self.proc is not None:
            code = self.proc.exitcode
            if self.proc.is_alive():
                self.proc.kill()
            self.proc.join(30)
            code = self.proc.exitcode if code is None else code
            self.proc = None
            return code
        return None

    def close(self):
        if self.proc is not None and self.proc.is_alive():
            try:
                self.conn.send(("close", {}))
                self.proc.join(30)
            except (OSError, BrokenPipeError):
                pass
        self._kill()

    def setup(self, row, routing, seed, ref_cfg, tokens=None):
        self.setup_kw = {
            "row": dict(row),
            "routing": routing,
            "seed": seed,
            "tokens": tokens,
        }
        self.ref_cfg = ref_cfg
        if self.proc is None or not self.proc.is_alive():
            self._start()
            return self.setup_info
        return self._request("setup", self.setup_kw)

    def measure(self, configs, rounds):
        results = [None] * len(configs)
        pending = list(range(len(configs)))
        restarts = 0
        while pending:
            if self.proc is None or not self.proc.is_alive():
                self._start()
            started, timing, lost = None, False, ""
            try:
                self.conn.send(
                    (
                        "measure",
                        {
                            "configs": [configs[i] for i in pending],
                            "rounds": rounds,
                            "ref_cfg": self.ref_cfg,
                            "err_ratio": self.err_ratio,
                            "warmup": self.warmup,
                        },
                    )
                )
                while True:
                    kind, payload = self._recv()
                    if kind == "start":
                        started = pending[payload]
                    elif kind == "timing":
                        timing = True
                    elif kind == "lost":
                        lost = payload
                    elif kind == "ok":
                        for i, res in zip(pending, payload):
                            results[i] = res
                        pending = []
                        break
                    else:
                        raise RuntimeError(payload)
            except WorkerLost as exc:
                code = self._kill()
                restarts += 1
                why = lost or f"{exc} (exit {code})"
                if started is not None and not timing:
                    results[started] = {
                        "ok": False,
                        "error": f"{why}; worker restarted",
                    }
                    pending.remove(started)
                    self.log(f"worker lost on a candidate ({why[:120]}); restarting")
                elif restarts > 2:
                    for i in pending:
                        results[i] = {"ok": False, "error": f"{exc} (exit {code})"}
                    pending = []
                if restarts > len(configs) + 2:
                    for i in pending:
                        results[i] = {"ok": False, "error": "too many worker restarts"}
                    pending = []
        return results


@dataclass
class Job:
    row: dict
    shape: Shape
    seeds: list
    routing: str = "balanced"
    seed: int = 0
    label: str = ""
    tokens: list | None = None
    result: dict = field(default_factory=dict)


def run_jobs(
    jobs: list[Job],
    gpus: list[int],
    opts: SearchOptions,
    *,
    timeout,
    err_ratio,
    warmup,
    log=print,
):
    """Search every job, one worker per GPU, shapes handed out as GPUs free up."""
    todo: queue.Queue = queue.Queue()
    for job in jobs:
        todo.put(job)

    def runner(gpu):
        def glog(msg, job=None):
            log(f"[grouped] gpu{gpu} {msg}")

        bench = Bench(
            gpu, timeout=timeout, err_ratio=err_ratio, warmup=warmup, log=glog
        )
        try:
            while True:
                try:
                    job = todo.get_nowait()
                except queue.Empty:
                    return

                def jlog(msg, job=job):
                    glog(f"{job.label}: {msg}")

                try:
                    info = bench.setup(
                        job.row, job.routing, job.seed, job.seeds[0][1], job.tokens
                    )
                    jlog(f"timing cases {info['cases']}")
                    job.result = search(
                        job.shape, job.seeds, bench.measure, opts, log=jlog
                    )
                # The shape fails, the run goes on.
                except Exception as exc:  # noqa: BLE001
                    job.result = {
                        "ok": False,
                        "error": f"{type(exc).__name__}: {exc}"[:400],
                        "records": [],
                    }
                    jlog(f"failed: {job.result['error']}")
        finally:
            bench.close()

    threads = [threading.Thread(target=runner, args=(g,), daemon=True) for g in gpus]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return jobs
