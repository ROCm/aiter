# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
"""
Opus a16w16 tuned-CSV lookup against the **global** aiter BF16 GEMM CSVs.

Source of truth:

  aiter/configs/bf16_tuned_gemm.csv
  aiter/configs/model_configs/*_bf16_tuned_gemm.csv

These are the same CSVs that the aiter-global `gemm_a16w16` dispatcher
reads; opus rows are stamped with `libtype == 'opus'` and coexist with
`asm` / `triton` / `skinny` / `flydsl` / `torch` / `hipblaslt` rows for
the same (cu_num, M, N, K, ...) keys. We filter by `libtype == 'opus'`
here, so the opus runtime dispatch only returns a tuned winner when one
of those CSVs has an opus row matching the shape.

Schema (matches gradlib/GemmTuner.py output):

  gfx, cu_num, M, N, K, bias, dtype, outdtype, scaleAB, bpreshuffle,
  libtype, solidx, splitK, us, kernelName, err_ratio, tflops, bw

`gfx` is optional (the legacy opus-private CSV did not have it); when
absent we tolerate it. Rows missing any of the 9 key columns
(cu_num/M/N/K/bias/dtype/outdtype/scaleAB/bpreshuffle) are skipped.

Configuration:

  AITER_OPUS_TUNED_CSV_GLOB
      Colon-separated list of glob patterns for tuned CSVs. Default
      includes both the global BF16 GEMM CSV and the per-model CSVs
      under aiter/configs/model_configs/.

  (Removed in this rewrite: AITER_OPUS_A16W16_TUNED_CSV,
   AITER_OPUS_A16W16_UNTUNED_CSV, AITER_OPUS_LOG_UNTUNED, and the
   autolog feature. Untuned-shape collection is no longer supported;
   use gradlib/gemm_tuner.py --libtype opus to tune shapes offline.)
"""

from __future__ import annotations

import functools
import glob
import os
from typing import Optional

import pandas as pd
import torch

from aiter.jit.core import AITER_ROOT_DIR

# ---- Env / default paths --------------------------------------------------

# Colon-separated list of glob patterns; each pattern is expanded with
# glob.glob() and the results concatenated. Order does not matter -- on
# duplicate keys we keep the row with the smallest `us` (best winner)
# across all files.
_DEFAULT_TUNED_CSV_GLOB = (
    f"{AITER_ROOT_DIR}/aiter/configs/bf16_tuned_gemm.csv"
    f":{AITER_ROOT_DIR}/aiter/configs/model_configs/*_bf16_tuned_gemm.csv"
)

OPUS_TUNED_CSV_GLOB = os.getenv("AITER_OPUS_TUNED_CSV_GLOB", _DEFAULT_TUNED_CSV_GLOB)


# ---- Tuned CSV lookup -----------------------------------------------------

_KEY_COLUMNS = (
    "cu_num",
    "M",
    "N",
    "K",
    "bias",
    "dtype",
    "outdtype",
    "scaleAB",
    "bpreshuffle",
)


def _resolve_csv_paths() -> list[str]:
    """Expand OPUS_TUNED_CSV_GLOB into a deduplicated list of file paths."""
    paths: list[str] = []
    seen: set[str] = set()
    for pattern in OPUS_TUNED_CSV_GLOB.split(os.pathsep):
        pattern = pattern.strip()
        if not pattern:
            continue
        for path in sorted(glob.glob(pattern)):
            if path in seen:
                continue
            seen.add(path)
            paths.append(path)
    return paths


@functools.lru_cache(maxsize=1)
def _load_tuned_dict() -> dict:
    """Load opus-flagged rows from all configured tuned CSVs into a dict.

    Returns a mapping `key -> {'solidx', 'splitK', 'kernelName'}` where key
    is the 9-tuple from `_KEY_COLUMNS`. When multiple CSVs report a winner
    for the same key, the one with the smallest `us` is retained (best
    timing wins).

    Cached for the process lifetime. Call `_load_tuned_dict.cache_clear()`
    if a fresh CSV is dropped in between invocations (rare in production).
    """
    paths = _resolve_csv_paths()
    if not paths:
        return {}

    frames: list[pd.DataFrame] = []
    for path in paths:
        try:
            df = pd.read_csv(path)
        except (pd.errors.EmptyDataError, FileNotFoundError):
            continue
        if "libtype" not in df.columns:
            # CSVs without a `libtype` column predate the multi-backend
            # schema; they cannot contain opus rows by definition. Skip
            # rather than misclassify their rows as opus.
            continue
        df = df[df["libtype"] == "opus"]
        if df.empty:
            continue
        missing = [c for c in _KEY_COLUMNS if c not in df.columns]
        if missing:
            # Malformed / partial-schema CSV. Skip rather than crash.
            continue
        frames.append(df)

    if not frames:
        return {}
    combined = pd.concat(frames, ignore_index=True).drop_duplicates()

    # Conflict resolution: same 9-tuple key from multiple files -> keep
    # the row with the smallest `us` (best timing). If `us` is missing,
    # fall back to first-write-wins.
    has_us = "us" in combined.columns
    if has_us:
        combined = combined.sort_values("us", ascending=True, kind="mergesort")
    out: dict = {}
    for _, row in combined.iterrows():
        key = tuple(row[c] for c in _KEY_COLUMNS)
        if key in out:
            continue  # already kept the better one
        try:
            out[key] = {
                "solidx": int(row["solidx"]),
                "splitK": int(row["splitK"]),
                "kernelName": str(row.get("kernelName", "")),
            }
        except (KeyError, ValueError, TypeError):
            # Missing solidx / splitK or non-int values; skip this row.
            continue
    return out


def _key_from_runtime(
    M: int,
    N: int,
    K: int,
    bias: bool,
    dtype: torch.dtype,
    outdtype: torch.dtype,
    scaleAB: bool = False,
    bpreshuffle: bool = False,
) -> tuple:
    """Build the 9-tuple lookup key using the current device's cu_num."""
    cu_num = torch.cuda.get_device_properties(
        torch.cuda.current_device()
    ).multi_processor_count
    return (
        int(cu_num),
        int(M),
        int(N),
        int(K),
        bool(bias),
        str(dtype),
        str(outdtype),
        bool(scaleAB),
        bool(bpreshuffle),
    )


# Mono-tile kid → (B_M, B_N, B_K). Must stay in lock-step with
# csrc/opus_gemm/opus_gemm_common.py:_MONO_TILE_TILES; the runtime guard
# below uses it to validate (N, K) alignment for CSV-picked mono kids,
# since tuned_gemm.get_padded_m pads the lookup key by M only and can
# surface a kid whose B_N / B_K does not divide the actual N / K.
_MONO_TILE_KID_TILES = {
    1400: (192, 256, 64),
    1401: (128, 256, 64),
    1402: (192, 128, 64),
    1403: (128, 128, 64),
    1404: (64, 128, 64),
}


def mono_kid_shape_ok(kid: int, N: int, K: int) -> bool:
    """Return True iff `kid` is a mono-tile kid whose B_N / B_K divides N / K.

    Returns True for non-mono kids (out of range) so callers can use this
    as an unconditional gate without having to special-case the kid range.
    B_M is intentionally NOT checked: the mono-tile launcher now handles
    M-tail via the bounded gmem descriptor (commit 41e2d482a), so M may
    be non-tile-aligned. N and K must still be tile-aligned -- the kernel
    has no N-tail mask (column writes would spill into the next row) and
    no K-tail mask.
    """
    bm_bn_bk = _MONO_TILE_KID_TILES.get(int(kid))
    if bm_bn_bk is None:
        return True
    _, B_N, B_K = bm_bn_bk
    return (int(N) % B_N == 0) and (int(K) % B_K == 0)


def lookup_tuned(
    M: int,
    N: int,
    K: int,
    bias: bool,
    dtype: torch.dtype,
    outdtype: torch.dtype,
    scaleAB: bool = False,
    bpreshuffle: bool = False,
) -> Optional[dict]:
    """Look up a tuned winner for this shape; returns dict or None.

    Dict contains 'solidx' (kernelId), 'splitK', 'kernelName'.
    """
    key = _key_from_runtime(M, N, K, bias, dtype, outdtype, scaleAB, bpreshuffle)
    return _load_tuned_dict().get(key)


__all__ = [
    "OPUS_TUNED_CSV_GLOB",
    "lookup_tuned",
    "mono_kid_shape_ok",
]
