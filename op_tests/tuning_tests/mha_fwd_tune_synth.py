# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""
Synthetic data generators for the MHA forward tuning tool tests
(`csrc/cpp_itfs/mha/tools/`).

Everything here is pure standard library; no GPU, torch or aiter import is
needed. The formats mirror what the real producers emit:

* `make_dump_line`   -> one `[MHA_FWD] ...` line as written by
                        `csrc/include/mha_fwd_dump.h` / `mha_common.h`.
* `write_dump_log`   -> a log file made of such lines.
* `write_untune_csv` -> a `mha_untune_*.csv` as written by
                        `mha_count_shape.py generate_tune_range`.
* `make_tile_expr`   -> a `best_tile_expr` string as written by `mha_tune.py`.
* `write_tuned_csv`  -> a `mha_tuned_*.csv` with the same columns
                        `mha_tune.py` writes.

This file intentionally does not start with `test_` so that test runners do
not collect it.
"""

from __future__ import annotations

import csv
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

# Field order of a dump line (see mha_fwd_dump.h / mha_common.h).
_COMMON_FIELD_ORDER = (
    "mode",
    "dtype",
    "hdim_q",
    "hdim_v",
    "nhead_q",
    "nhead_k",
    "batch",
    "max_seqlen_q",
    "mask_type",
    "bias_type",
    "has_lse",
    "has_dropout",
    "window_left",
    "window_right",
    "sink_size",
    "has_sink",
    "has_logits_soft_cap",
    "qscale_type",
)
_GROUP_FIELD_ORDER = (
    *_COMMON_FIELD_ORDER,
    "min_seqlen_q",
    "total_q",
    "total_k",
    "seqlens_q",
    "seqlens_k",
)
_BATCH_FIELD_ORDER = (*_COMMON_FIELD_ORDER, "seqlen_q", "seqlen_k")

# Full group signature (same order as mha_count_shape.GROUP_COLS) and its
# default values (bf16 hdim 72/72, 16/16 heads, no mask, no traits).
SIGNATURE_DEFAULTS: dict[str, Any] = {
    "mode": "group",
    "dtype": "bf16",
    "hdim_q": 72,
    "hdim_v": 72,
    "mask_type": 0,
    "nhead_q": 16,
    "nhead_k": 16,
    "has_logits_soft_cap": 0,
    "bias_type": 0,
    "has_lse": 0,
    "has_dropout": 0,
    "skip_min_seqlen_q": 0,
    "qscale_type": 0,
    "has_sink": 0,
}

_BOOL_TOKENS = {
    "has_logits_soft_cap": ("nlogits", "logits"),
    "has_lse": ("nlse", "lse"),
    "has_dropout": ("ndropout", "dropout"),
    "skip_min_seqlen_q": ("nskip", "skip"),
    "has_sink": ("nsink", "sink"),
}
_BIAS_TOKENS = {0: "nbias", 1: "bias", 2: "alibi"}
_QSCALE_TOKENS = {0: "nqscale", 1: "pertensor", 2: "blockscale"}


def signature(**over: Any) -> dict[str, Any]:
    """SIGNATURE_DEFAULTS with overrides (unknown keys rejected)."""
    bad = set(over) - set(SIGNATURE_DEFAULTS)
    if bad:
        raise KeyError(f"unknown signature fields: {sorted(bad)}")
    return {**SIGNATURE_DEFAULTS, **over}


def signature_name(sig: Mapping[str, Any]) -> str:
    """Filename signature, e.g. group_bf16_hq72_hv72_mask0_nh16_nhk16_
    nlogits_nbias_nlse_ndropout_nskip_nqscale_nsink (independent copy of
    mha_count_shape.group_signature, so the tests cross-check it)."""
    s = {**SIGNATURE_DEFAULTS, **sig}
    b = {c: int(s[c]) != 0 for c in _BOOL_TOKENS}
    return "_".join(
        [
            str(s["mode"]),
            str(s["dtype"]),
            f"hq{s['hdim_q']}",
            f"hv{s['hdim_v']}",
            f"mask{s['mask_type']}",
            f"nh{s['nhead_q']}",
            f"nhk{s['nhead_k']}",
            _BOOL_TOKENS["has_logits_soft_cap"][b["has_logits_soft_cap"]],
            _BIAS_TOKENS[int(s["bias_type"])],
            _BOOL_TOKENS["has_lse"][b["has_lse"]],
            _BOOL_TOKENS["has_dropout"][b["has_dropout"]],
            _BOOL_TOKENS["skip_min_seqlen_q"][b["skip_min_seqlen_q"]],
            _QSCALE_TOKENS[int(s["qscale_type"])],
            _BOOL_TOKENS["has_sink"][b["has_sink"]],
        ]
    )


# Columns written by mha_tune.py for a tuned csv (keep in sync with
# mha_tune.TUNED_CSV_COLS; asserted by the tests).
TUNED_FIELDNAMES = (
    "max_seqlen",
    *SIGNATURE_DEFAULTS,
    "bench_variant",
    "best_hdim_q",
    "best_hdim_v",
    "best_time_ms",
    "best_tflops",
    "best_gbps",
    "best_kname",
    "best_tile_name",
    "best_tile_expr",
    "status",
)

# A valid 19-int FmhaFwdTileSize (hdim 80/96 flavour) used as default.
DEFAULT_TILE = (128, 64, 64, 96, 32, 128, 4, 1, 1, 4, 1, 1, 32, 32, 16, 32, 32, 16, 2)
# A second, different tile, handy for "tile switches at boundary" tests.
ALT_TILE = (64, 64, 64, 96, 32, 128, 4, 1, 1, 4, 1, 1, 16, 16, 16, 16, 16, 16, 1)


def _fmt(v: Any) -> str:
    if isinstance(v, (list, tuple)):
        return "[" + ",".join(str(int(x)) for x in v) + "]"
    return str(v)


def make_dump_line(
    seqlens_q: Sequence[int] = (1024,),
    seqlens_k: Sequence[int] | None = None,
    mode: str = "group",
    drop: Iterable[str] = (),
    **fields: Any,
) -> str:
    """Return one `[MHA_FWD]` dump line.

    Defaults describe a bf16, hdim 72/72, 16/16 heads, no-mask call
    (window -1/-1, no sink, no soft-cap, no qscale, min_seqlen_q=0).

    `mode="group"` emits the varlen fields (min_seqlen_q, total_q/k,
    seqlens_q/k). `mode="batch"` emits scalar `seqlen_q`/`seqlen_k` taken
    from `seqlens_q[0]`/`seqlens_k[0]`, with `batch=len(seqlens_q)`.
    Any other `mode` value is written verbatim with the group layout.

    Any field may be overridden via kwargs; extra (unknown) kwargs are
    appended after the known fields in the given order. Passing a field
    value of `None` (or listing its name in `drop`) omits it from the line,
    which is useful to emulate older or malformed log formats.
    """
    sq = list(seqlens_q)
    sk = list(seqlens_k) if seqlens_k is not None else list(sq)
    rec: dict[str, Any] = {
        "mode": mode,
        "dtype": "bf16",
        "hdim_q": 72,
        "hdim_v": 72,
        "nhead_q": 16,
        "nhead_k": 16,
        "batch": len(sq),
        "max_seqlen_q": max(sq) if sq else 0,
        "mask_type": 0,
        "bias_type": 0,
        "has_lse": 0,
        "has_dropout": 0,
        "window_left": -1,
        "window_right": -1,
        "sink_size": 0,
        "has_sink": 0,
        "has_logits_soft_cap": 0,
        "qscale_type": 0,
    }
    if mode == "batch":
        order = _BATCH_FIELD_ORDER
        rec["seqlen_q"] = sq[0] if sq else 0
        rec["seqlen_k"] = sk[0] if sk else 0
    else:
        order = _GROUP_FIELD_ORDER
        rec.update(
            {
                "min_seqlen_q": 0,
                "total_q": sum(sq),
                "total_k": sum(sk),
                "seqlens_q": sq,
                "seqlens_k": sk,
            }
        )
    extra: list[str] = []
    for k, v in fields.items():
        if k not in order:
            extra.append(k)
        rec[k] = v

    dropped = set(drop)
    parts = [
        f"{k}={_fmt(rec[k])}"
        for k in (*order, *extra)
        if rec.get(k) is not None and k not in dropped
    ]
    return "[MHA_FWD] " + " ".join(parts)


def write_dump_log(path: Path, lines: Iterable[str], noise: bool = True) -> Path:
    """Write dump lines to `path`. With `noise=True`, interleave unrelated
    log lines, as a real application log would contain."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out: list[str] = []
    if noise:
        out.append("INFO: unrelated application log line")
    for ln in lines:
        out.append(ln)
        if noise:
            out.append("")
    path.write_text("\n".join(out) + "\n", encoding="utf-8")
    return path


def untune_csv_name(sig: Mapping[str, Any], gid: int = 0) -> str:
    return f"mha_untune_{gid}_{signature_name(sig)}.csv"


def write_untune_csv(
    path: Path,
    max_seqlens: Iterable[int],
    meta: Mapping[str, Any] | None = None,
    columns: Sequence[str] | None = None,
    row_over: Mapping[int, Mapping[str, Any]] | None = None,
) -> Path:
    """Write a synthetic `mha_untune_*.csv` as produced by
    `mha_count_shape.py generate_tune_range` (one row per max_seqlen).

    `meta` overrides SIGNATURE_DEFAULTS. If `path` is a directory, the
    canonical filename for `meta` is generated inside it. `columns` limits
    the signature columns written (to emulate legacy / broken CSVs);
    `row_over` maps a 0-based row index to per-row overrides.
    """
    path = Path(path)
    base = {**SIGNATURE_DEFAULTS, **(meta or {})}
    if path.is_dir():
        path = path / untune_csv_name(base)
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = list(columns) if columns is not None else list(base.keys())
    fieldnames = ["max_seqlen", *cols]
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for i, m in enumerate(max_seqlens):
            writer.writerow({"max_seqlen": m, **base, **(row_over or {}).get(i, {})})
    return path


def make_tile_expr(
    tile: Sequence[int] = DEFAULT_TILE, hdim_q: int = 80, hdim_v: int = 96
) -> str:
    """Render a `best_tile_expr` like mha_tune.py does."""
    if len(tile) != 19:
        raise ValueError(f"tile must have 19 ints, got {len(tile)}")
    return (
        f"({hdim_q}, {hdim_v}) : [FmhaFwdTileSize({', '.join(str(x) for x in tile)})]"
    )


def tuned_csv_name(meta: Mapping[str, Any], gid: int = 0) -> str:
    """Filename `mha_tuned_<gid>_<sig>.csv` for the signature in `meta`."""
    sig = {k: v for k, v in meta.items() if k in SIGNATURE_DEFAULTS}
    return f"mha_tuned_{gid}_{signature_name(sig)}.csv"


def write_tuned_csv(
    path: Path,
    meta: Mapping[str, Any],
    rows: Iterable[Mapping[str, Any]],
    extra_fieldnames: Sequence[str] = (),
    drop_columns: Iterable[str] = (),
) -> Path:
    """Write a synthetic tuned csv.

    `meta` holds per-file constants (mode, dtype, hdim_q, hdim_v, mask_type,
    best_hdim_q, best_hdim_v, plus any extra columns). Each entry of `rows`
    needs at least `max_seqlen`; `tile` (19 ints) and `status` are optional.
    If `path` is a directory, a canonical filename is generated inside it.
    `drop_columns` omits columns from the header, to emulate legacy csvs.
    """
    path = Path(path)
    if path.is_dir():
        path = path / tuned_csv_name(meta)
    path.parent.mkdir(parents=True, exist_ok=True)

    base = {
        **SIGNATURE_DEFAULTS,
        "bench_variant": "nskip",
        "best_hdim_q": 80,
        "best_hdim_v": 96,
    }
    base.update(meta)
    dropped = set(drop_columns)
    fieldnames = [f for f in TUNED_FIELDNAMES if f not in dropped] + [
        f for f in extra_fieldnames if f not in TUNED_FIELDNAMES
    ]

    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            tile = r.get("tile", DEFAULT_TILE)
            row: dict[str, Any] = dict(base)
            row.update(
                {
                    "max_seqlen": r["max_seqlen"],
                    "best_time_ms": r.get("best_time_ms", 0.1),
                    "best_tflops": r.get("best_tflops", 100.0),
                    "best_gbps": r.get("best_gbps", 1000.0),
                    "best_kname": r.get("best_kname", "fmha_fwd_synth"),
                    "best_tile_name": r.get("best_tile_name", "synth_tile"),
                    "status": r.get("status", "ok"),
                }
            )
            row.update({k: v for k, v in r.items() if k != "tile"})
            row["best_tile_expr"] = make_tile_expr(
                tile, base["best_hdim_q"], base["best_hdim_v"]
            )
            writer.writerow(row)
    return path
