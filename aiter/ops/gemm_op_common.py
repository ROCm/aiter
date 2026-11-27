# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import threading
from collections import OrderedDict

from aiter import logger

from ..jit.core import AITER_LOG_TUNED_CONFIG, compile_ops


@compile_ops("module_gemm_common", fc_name="getPaddedM", ffi_type="ctypes")
def get_padded_m(M: int, N: int, K: int, gl: int) -> int: ...


# Misses already reported, as (family, *key), least recently seen first. Per
# process, like the lookup caches. Capped: a key that falls off the end may be
# logged again. The lock makes the check and the insert one step.
_MAX_REPORTED_TUNED_CONFIG_MISSES = 4096
_REPORTED_TUNED_CONFIG_MISSES: OrderedDict = OrderedDict()
_REPORTED_TUNED_CONFIG_MISSES_LOCK = threading.Lock()


def _first_report_of(report_key: tuple) -> bool:
    """True the first time ``report_key`` is seen (again after it was evicted)."""
    with _REPORTED_TUNED_CONFIG_MISSES_LOCK:
        if report_key in _REPORTED_TUNED_CONFIG_MISSES:
            _REPORTED_TUNED_CONFIG_MISSES.move_to_end(report_key)
            return False
        _REPORTED_TUNED_CONFIG_MISSES[report_key] = None
        while len(_REPORTED_TUNED_CONFIG_MISSES) > _MAX_REPORTED_TUNED_CONFIG_MISSES:
            _REPORTED_TUNED_CONFIG_MISSES.popitem(last=False)
        return True


def log_tuned_config_miss(family: str, key: tuple, message: str) -> bool:
    """Report a tuned-config lookup miss; True when a line was logged.

    Serving sees a new M on nearly every batch, so logging each (M, N, K) miss
    floods the log. Unless AITER_LOG_TUNED_CONFIG=1, a miss is logged once per
    ``(family, *key)``; ``key`` is everything the tuned row is keyed on except
    M (gfx, cu_num, N, K, dtypes, ...), plus whatever the fallback it picks
    depends on, so a different default for the same N/K is logged again. With
    AITER_LOG_TUNED_CONFIG=1 every miss is logged, as every hit is.

    The set of reported keys holds at most ``_MAX_REPORTED_TUNED_CONFIG_MISSES``
    entries, least recently seen dropped first, so a key can be logged again
    after it was evicted. The check and the insert are atomic across threads.
    Only the log line is affected: recording untuned shapes (AITER_TUNE_GEMM)
    is done elsewhere and does not call this.
    """
    if AITER_LOG_TUNED_CONFIG:
        logger.info(message)
        return True
    if not _first_report_of((family, *key)):
        return False
    logger.info(
        f"{message} (Not repeated for other M while this N/K/default key stays "
        "cached. Set AITER_LOG_TUNED_CONFIG=1 to log every miss.)"
    )
    return True


def find_padded_m_row(tuned: dict, key, m: int, n: int, k: int):
    """(row, M it was tuned at): the row for M itself, else for M padded at the
    two getPaddedM granularities, the walk the CK / asm lookups make. ``key``
    maps an M to the table key; row is None when no level hits."""
    padded_m = m
    for gl in (None, 0, 1):
        padded_m = m if gl is None else get_padded_m(m, n, k, gl)
        row = tuned.get(key(padded_m))
        if row is not None:
            return row, padded_m
    return None, padded_m


# Weight-scale block (rows x K) of an e8m0 mxscale GEMM. Tuned mxscale tables key
# on it, so kernels that read the same weight layout with different scale blocks
# never share a row.
MXSCALE_W_SCALE_BLOCKS = ("32x32", "128x128", "1x32", "1x128")


def mxscale_w_scale_block(w_scale_shape: tuple[int, ...], n: int, k: int) -> str:
    """The w_scale block of a [..., N / rows, K / cols] scale for an N x K weight."""
    n_blocks, k_blocks = w_scale_shape[-2], w_scale_shape[-1]
    block = f"{n // n_blocks}x{k // k_blocks}"
    if n % n_blocks or k % k_blocks or block not in MXSCALE_W_SCALE_BLOCKS:
        raise ValueError(
            f"w_scale {tuple(w_scale_shape)} is no {MXSCALE_W_SCALE_BLOCKS} block "
            f"of a {n} x {k} weight"
        )
    return block


def mxscale_bmm_groups_of_block(w_scale_block: str) -> tuple[int, int]:
    """(rows, cols) of a batched e8m0 GEMM's w_scale block."""
    groups = {"1x32": (1, 32), "32x32": (32, 32), "128x128": (128, 128)}
    if w_scale_block not in groups:
        raise ValueError(f"unsupported mxscale BMM w_scale block {w_scale_block!r}")
    return groups[w_scale_block]


def mxscale_w_scale_block_cell(cell, default: str | None = "128x128") -> str | None:
    """A tuned table's w_scale_block cell; a blank one, or the 0 a config merge
    fills a missing column with, reads as ``default``."""
    cell = "" if cell is None else str(cell).strip()
    return default if cell in ("", "0", "0.0", "nan") else cell


def with_mxscale_w_scale_block(df, path: str, default: str = "128x128"):
    """A tuned mxscale table with a checked w_scale_block column; a table from
    before the column holds only ``default`` rows, and so do the blank / 0 cells
    a merge of such a table with a newer one leaves."""
    if "w_scale_block" not in df.columns:
        logger.warning(
            "%s has no 'w_scale_block' column; reading its rows as %s. "
            "Re-run the tuner or migrate the CSV.",
            path,
            default,
        )
        df = df.assign(w_scale_block=default)
    else:
        cells = df["w_scale_block"].map(lambda c: mxscale_w_scale_block_cell(c, None))
        if cells.isna().any():
            logger.warning(
                "%s has rows without a w_scale_block; reading them as %s.",
                path,
                default,
            )
        df = df.assign(w_scale_block=cells.fillna(default))
    unknown = set(df["w_scale_block"]) - set(MXSCALE_W_SCALE_BLOCKS)
    if unknown:
        raise ValueError(
            f"{path}: w_scale_block {sorted(unknown)} not in {MXSCALE_W_SCALE_BLOCKS}"
        )
    return df
