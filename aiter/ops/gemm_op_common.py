# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from aiter import logger

from ..jit.core import compile_ops


@compile_ops("module_gemm_common", fc_name="getPaddedM", ffi_type="ctypes")
def get_padded_m(M: int, N: int, K: int, gl: int) -> int: ...


def padded_m_py(M: int, N: int, K: int, gl: int) -> int:
    """Mirrors getPaddedM in csrc/py_itfs_cu/gemm_common.cu for GPU-free callers
    (AOT wheel build); keep in sync, parity is checked by
    op_tests/tuning_tests/test_padded_m_parity.py. Valid for M >= 0."""

    def next_pow2(num: int) -> int:
        return 1 if num <= 1 else 1 << (num - 1).bit_length()

    if gl == 0:
        # below one 16-row step the next power of two is the finer bucket
        if 0 < M < 16:
            return next_pow2(M)
        if M <= 256:
            return (M + 15) // 16 * 16
        if M <= 1024:
            return (M + 31) // 32 * 32
        if M <= 4096:
            return (M + 63) // 64 * 64
        return (M + 127) // 128 * 128
    if gl == 1:
        return 8192 if M > 8192 and N > 4096 else next_pow2(M)
    return M


def find_padded_m_row(tuned: dict, key, m: int, n: int, k: int, pad=None):
    """(row, M it was tuned at): the row for M itself, else for M padded at the
    two getPaddedM granularities, the walk the CK / asm lookups make. ``key``
    maps an M to the table key; row is None when no level hits. ``pad`` replaces
    get_padded_m for callers without a GPU (see padded_m_py)."""
    pad = pad or get_padded_m
    padded_m = m
    for gl in (None, 0, 1):
        padded_m = m if gl is None else pad(m, n, k, gl)
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


def with_mxscale_w_scale_block(df, path: str, default: str = "128x128"):
    """A tuned mxscale table with a checked w_scale_block column; a table from
    before the column holds only ``default`` rows, and so do the blank cells a
    merge of such a table with a newer one leaves."""
    if "w_scale_block" not in df.columns:
        logger.warning(
            "%s has no 'w_scale_block' column; reading its rows as %s. "
            "Re-run the tuner or migrate the CSV.",
            path,
            default,
        )
        df = df.assign(w_scale_block=default)
    elif df["w_scale_block"].isna().any():
        logger.warning(
            "%s has rows without a w_scale_block; reading them as %s.", path, default
        )
        df = df.assign(w_scale_block=df["w_scale_block"].fillna(default))
    unknown = set(df["w_scale_block"]) - set(MXSCALE_W_SCALE_BLOCKS)
    if unknown:
        raise ValueError(
            f"{path}: w_scale_block {sorted(unknown)} not in {MXSCALE_W_SCALE_BLOCKS}"
        )
    return df
