# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
"""
Opus a16w16 private tuned-CSV lookup + untuned-shape autolog.

This module is the Python-side dispatch helper for `gemm_a16w16_opus`.
It intentionally lives *inside* the opus module tree rather than
piggybacking on `aiter.jit.core.AITER_CONFIGS` (which is reserved for
aiter-global configuration shared across backends such as triton / asm /
flydsl). Opus owns its own env vars and default paths:

    AITER_OPUS_A16W16_TUNED_CSV
        Default: aiter/ops/opus/configs/opus_gemm_a16w16_tuned.csv
        Read once per process via lru_cache; ~215 rows typically.

    AITER_OPUS_A16W16_UNTUNED_CSV
        Default: aiter/ops/opus/configs/opus_a16w16_untuned_gemm.csv
        Target file for AITER_OPUS_LOG_UNTUNED=1 autolog. Missing file
        is created on first write.

    AITER_OPUS_LOG_UNTUNED={0,1}
        Default: 0. When 1, every (M,N,K,bias,dtype,outdtype,scaleAB,
        bpreshuffle) tuple that misses the tuned lookup is appended
        (with de-dup) to the untuned CSV, ready to be fed to
        `csrc/opus_gemm/opus_gemm_tune.py -i` for offline tuning.

The lookup uses the 9-column composite key that matches the tuned CSV
schema from splitk_flatmm plan section 10 (cu_num, M, N, K, bias, dtype,
outdtype, scaleAB, bpreshuffle). Rows whose libtype != 'opus' are
filtered out so this file can coexist with a global
aiter/configs/bf16_tuned_gemm.csv in future integrations.
"""

from __future__ import annotations

import functools
import os
import threading
from typing import Optional

import pandas as pd
import torch

from aiter.jit.core import AITER_ROOT_DIR


# ---- Env / default paths --------------------------------------------------

OPUS_A16W16_TUNED_CSV = os.getenv(
    "AITER_OPUS_A16W16_TUNED_CSV",
    f"{AITER_ROOT_DIR}/aiter/ops/opus/configs/opus_gemm_a16w16_tuned.csv",
)

OPUS_A16W16_UNTUNED_CSV = os.getenv(
    "AITER_OPUS_A16W16_UNTUNED_CSV",
    f"{AITER_ROOT_DIR}/aiter/ops/opus/configs/opus_a16w16_untuned_gemm.csv",
)

AITER_OPUS_LOG_UNTUNED = int(os.getenv("AITER_OPUS_LOG_UNTUNED", "0"))


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


@functools.lru_cache(maxsize=1)
def _load_tuned_dict():
    """Load opus tuned CSV into (key_tuple -> dict) for O(1) lookup.

    Cached for the lifetime of the process. Callers can invalidate via
    `_load_tuned_dict.cache_clear()` if a fresh tune CSV is dropped in
    between invocations (rare in production hot path).
    """
    path = OPUS_A16W16_TUNED_CSV
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path).drop_duplicates()
    if "libtype" in df.columns:
        df = df[df["libtype"] == "opus"]
    missing = [c for c in _KEY_COLUMNS if c not in df.columns]
    if missing:
        # CSV is malformed / from a different schema. Fall back to empty
        # so callers degrade to C++ heuristic instead of raising.
        return {}
    out: dict = {}
    for _, row in df.iterrows():
        key = tuple(row[c] for c in _KEY_COLUMNS)
        out[key] = {
            "solidx": int(row["solidx"]),
            "splitK": int(row["splitK"]),
            "kernelName": str(row.get("kernelName", "")),
        }
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
    """Build the 9-tuple lookup key using the current device's cu_num.

    Dtype is serialized as `str(torch.dtype)` (e.g. 'torch.bfloat16'),
    matching what the tuner's `result_to_df` writes.
    """
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
    key = _key_from_runtime(
        M, N, K, bias, dtype, outdtype, scaleAB, bpreshuffle
    )
    return _load_tuned_dict().get(key)


# ---- Untuned shape autolog ------------------------------------------------

_UNTUNED_LOCK = threading.Lock()
_UNTUNED_COLUMNS = (
    "M",
    "N",
    "K",
    "bias",
    "dtype",
    "outdtype",
    "scaleAB",
    "bpreshuffle",
)


def maybe_log_untuned_shape(
    M: int,
    N: int,
    K: int,
    bias: bool,
    dtype: torch.dtype,
    outdtype: torch.dtype,
    scaleAB: bool = False,
    bpreshuffle: bool = False,
) -> None:
    """Append (M,N,K,bias,dtype,outdtype,scaleAB,bpreshuffle) to the
    untuned CSV for offline tuning. Safe to call unconditionally; this
    function short-circuits when AITER_OPUS_LOG_UNTUNED != 1.

    Row schema matches aiter/configs/model_configs/gptoss_bf16_untuned_gemm.csv
    so the file can feed directly into `csrc/opus_gemm/opus_gemm_tune.py -i`.
    """
    if not AITER_OPUS_LOG_UNTUNED:
        return
    path = OPUS_A16W16_UNTUNED_CSV
    row = {
        "M": int(M),
        "N": int(N),
        "K": int(K),
        "bias": bool(bias),
        "dtype": str(dtype),
        "outdtype": str(outdtype),
        "scaleAB": bool(scaleAB),
        "bpreshuffle": bool(bpreshuffle),
    }
    with _UNTUNED_LOCK:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if os.path.exists(path):
            try:
                old = pd.read_csv(path)
            except pd.errors.EmptyDataError:
                old = pd.DataFrame(columns=_UNTUNED_COLUMNS)
        else:
            old = pd.DataFrame(columns=_UNTUNED_COLUMNS)
        new = pd.concat([old, pd.DataFrame([row])], ignore_index=True)
        new = new.drop_duplicates().reset_index(drop=True)
        new.to_csv(path, index=False)


__all__ = [
    "OPUS_A16W16_TUNED_CSV",
    "OPUS_A16W16_UNTUNED_CSV",
    "AITER_OPUS_LOG_UNTUNED",
    "lookup_tuned",
    "maybe_log_untuned_shape",
]
