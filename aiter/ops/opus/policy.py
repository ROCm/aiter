# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Caller-side selection policy for OPUS A16W16 and MXFP8 BMM.

This module runs before the exact-kid :func:`opus_gemm` or :func:`opus_bmm`
entry. A16W16 policy validates tuned candidates and supplies explicit
per-architecture heuristic candidates. The gfx950 MXFP8 BMM policy owns tuned
CSV discovery, legacy-id normalization and its fallback kid/split selection.
Exact launch, workspace materialization and C++ dispatch remain outside this
module; no policy path replaces a caller-supplied exact kid inside the public
launcher.
"""

from __future__ import annotations

from functools import cache, lru_cache

import pandas as pd

from aiter import logger
from csrc.opus_gemm.opus_gemm_common import (
    DEFAULT_COMPILED_KIDS_BY_ARCH,
    GFX942_BF16WS_EXACT_N,
    canonical_output_dtype,
    get_kernel_instance,
)

from ...jit.core import AITER_CONFIGS, AITER_LOG_TUNED_CONFIG
from ...jit.utils.chip_info import get_gfx_runtime as get_gfx
from ..gemm_op_common import get_padded_m
from .launch_plan import (
    A16W16LaunchPlan,
    _get_cached_a16w16_launch_plan,
)

# ---- A16W16 tuned-candidate and heuristic policy -------------------------

_A16W16_TUNED_KEY_COLUMNS = (
    "gfx",
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


@cache
def _load_a16w16_opus_tuned() -> dict:
    """Load only OPUS rows from the merged global A16W16 tuned config."""
    path = AITER_CONFIGS.AITER_CONFIG_GEMM_BF16_FILE
    try:
        df = pd.read_csv(path).drop_duplicates()
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return {}
    except (OSError, UnicodeDecodeError, pd.errors.ParserError) as exc:
        logger.warning(
            "Ignoring unreadable A16W16 tuned CSV %r; OPUS will use its "
            "heuristic fallback: %s",
            path,
            exc,
        )
        return {}

    required = set(_A16W16_TUNED_KEY_COLUMNS) | {"libtype", "solidx", "splitK"}
    missing = required.difference(df.columns)
    if missing:
        logger.warning(
            "Ignoring A16W16 tuned CSV %r; missing columns %s",
            path,
            sorted(missing),
        )
        return {}

    df = df[df["libtype"].eq("opus")]
    if df.empty:
        return {}

    integer_columns = ["solidx", "splitK"]
    numeric = df[integer_columns].apply(pd.to_numeric, errors="coerce")
    valid = (
        numeric.notna().all(axis=1)
        & numeric.ge(0).all(axis=1)
        & numeric.lt(float("inf")).all(axis=1)
        & numeric.eq(numeric.round()).all(axis=1)
    )
    invalid_rows = int((~valid).sum())
    if invalid_rows:
        logger.warning(
            "Skipping %d malformed OPUS row(s) in A16W16 tuned CSV %r: "
            "solidx and splitK must be non-negative integers",
            invalid_rows,
            path,
        )
    df = df.loc[valid].copy()
    if df.empty:
        return {}
    df[integer_columns] = numeric.loc[valid].astype("int64")

    if "us" in df.columns:
        df["_opus_sort_us"] = pd.to_numeric(df["us"], errors="coerce")
        df = df.sort_values("_opus_sort_us", kind="stable", na_position="last").drop(
            columns="_opus_sort_us"
        )
    keys = list(_A16W16_TUNED_KEY_COLUMNS)
    return df.drop_duplicates(keys).set_index(keys).to_dict("index")


@lru_cache(maxsize=4096)
def lookup_a16w16_opus_config(
    *,
    arch: str,
    cu_num: int,
    M: int,
    N: int,
    K: int,
    has_bias: bool,
    input_dtype: object,
    output_dtype: object,
) -> dict | None:
    """Return the exact-shape OPUS tuned row used by the legacy OPUS caller."""
    key = (
        str(arch).lower().split(":", 1)[0],
        int(cu_num),
        int(M),
        int(N),
        int(K),
        bool(has_bias),
        str(input_dtype),
        str(output_dtype),
        False,
        False,
    )
    config = _load_a16w16_opus_tuned().get(key)
    return None if config is None else dict(config)


def _heuristic_a16w16_kid_gfx950(
    M: int,
    N: int,
    K: int,
    batch: int = 1,
    has_bias: bool = False,
    output_dtype: object = "bf16",
) -> int:
    """Return the original gfx950 no-tuned-row fallback kid."""
    del batch, output_dtype
    M, N, K = map(int, (M, N, K))
    split_barrier_ok = N % 16 == 0 and K % 64 == 0 and (K // 64) % 2 == 0

    if M <= 4:
        if M % 64 == 0 and N % 64 == 0 and K % 128 == 0:
            return 1208
        return 208
    if M <= 64:
        if M % 64 == 0 and N % 32 == 0 and K % 128 == 0:
            return 1206
        return 206
    if M <= 128:
        if M % 64 == 0 and N % 64 == 0 and K % 64 == 0:
            return 1200
        return 200
    if split_barrier_ok and not has_bias:
        if M % 256 == 0 and N % 256 == 0 and K % 64 == 0:
            return 1300
        return 300
    if M % 64 == 0 and N % 64 == 0 and K % 64 == 0:
        return 1200
    return 200


def _heuristic_a16w16_kid_gfx1250(
    M: int,
    N: int,
    K: int,
    batch: int = 1,
    has_bias: bool = False,
    output_dtype: object = "bf16",
) -> int:
    """Return the original gfx1250 no-tuned-row fallback kid."""
    del K, batch, has_bias, output_dtype
    M, N = map(int, (M, N))
    if M % 32 == 0:
        if N % 128 == 0:
            return 20007
        if N % 64 == 0:
            return 20006
        if N % 32 == 0:
            return 20005
    if N % 128 == 0:
        return 20004
    if N % 64 == 0:
        return 20003
    return 20000


@lru_cache(maxsize=1)
def _gfx942_heuristic_symbol_to_kid() -> dict[str, int]:
    """Build the canonical gfx942 launcher-symbol mapping from the registry."""
    result: dict[str, int] = {}
    for kid in DEFAULT_COMPILED_KIDS_BY_ARCH["gfx942"]:
        instance = get_kernel_instance("gfx942", "a16w16", kid)
        if instance is None:
            raise RuntimeError(f"gfx942 heuristic kid {kid} has no a16w16 instance")
        previous = result.setdefault(instance.name, int(kid))
        if previous != kid:
            raise RuntimeError(
                f"duplicate gfx942 launcher symbol {instance.name!r}: "
                f"kids {previous} and {kid}"
            )
    return result


def _gfx942_heuristic_kid_for_symbol(symbol: str) -> int:
    try:
        return _gfx942_heuristic_symbol_to_kid()[symbol]
    except KeyError as exc:
        raise RuntimeError(
            f"gfx942 heuristic returned unknown launcher symbol {symbol!r}"
        ) from exc


def _gfx942_heuristic_split_barrier_ok(N: int, K: int) -> bool:
    loops = (K + 63) // 64
    return N % 16 == 0 and K % 64 == 0 and loops >= 2 and loops % 2 == 0


def _gfx942_heuristic_bf16ws_band(M: int, N: int, K: int) -> bool:
    return (
        K >= 4096 and K % 64 == 0 and 104 <= M <= 608 and (N == 256 or 512 <= N <= 2048)
    )


def _gfx942_heuristic_bf16_symbol(M: int, N: int, K: int) -> str:
    """Port of the original gfx942 BF16 launcher-choice ordering."""
    k64_ok = K % 64 == 0
    k32_ok = K % 32 == 0
    wkc_bk64_ok = K >= 4096 and K % 512 == 0
    p1_ok = K % 128 == 0
    sb_ok = _gfx942_heuristic_split_barrier_ok(N, K)

    if K == 4096:
        if p1_ok and (M in (48, 64) and N == 1024):
            return "opus_gemm_gfx942_splitk_p1_bk128_bf16ws_256x64x64x128_2x2_16x16x16_0x0x0"
        if p1_ok and ((M == 128 and N == 512) or (M == 256 and N == 256)):
            return "opus_gemm_gfx942_splitk_p1_bk128_bf16ws_256x64x64x128_2x2_16x16x16_0x0x0"
        if p1_ok and M == 512 and N == 256:
            return "opus_gemm_gfx942_splitk_p1_bk128_256x64x64x128_2x2_16x16x16_0x0x0"
        if M in (48, 64) and 1536 <= N <= 2048:
            return "opus_gemm_gfx942_splitk_legacy_512x64x128x64_2x4_16x16x16_0x0x0"
        if (M == 128 and N == 1024) or (M == 256 and N == 512):
            return "opus_gemm_gfx942_splitk_legacy_512x64x128x64_2x4_16x16x16_0x0x0"
        if (
            (M == 128 and 1536 <= N <= 2048)
            or (M == 256 and N == 1024)
            or (M == 512 and N == 512)
        ):
            return "opus_gemm_gfx942_splitk_legacy_512x128x128x64_2x4_16x16x16_0x0x0"

    if K >= 1024 and k32_ok and N >= 1536 and M <= 32:
        if M <= 4 and N >= 4096:
            return "opus_gemm_gfx942_wkc_512x16x16x64_1x1_16x16x16_0x0x0"
        if M <= 16:
            if wkc_bk64_ok:
                return "opus_gemm_gfx942_wkc_512x16x32x64_1x1_16x16x16_0x0x0"
            return "opus_gemm_gfx942_wkc_512x16x32x32_1x1_16x16x16_0x0x0"
        if M == 32 and K == 4096 and wkc_bk64_ok:
            return "opus_gemm_gfx942_wkc_512x16x32x64_1x1_16x16x16_0x0x0"
        return "opus_gemm_gfx942_wkc_256x32x32x64_1x1_16x16x16_0x0x0"

    if (
        K >= 512
        and k64_ok
        and (N <= 64 or (M <= 128 and N <= 1024) or (M <= 8 and N <= 1536))
    ):
        if N <= 64 and M > 128:
            return "opus_gemm_gfx942_wkc_512x32x16x64_1x1_16x16x16_0x0x0"
        if N <= 256 or M <= 8 or (M <= 16 and N <= 800):
            return "opus_gemm_gfx942_wkc_512x16x16x64_1x1_16x16x16_0x0x0"
        return "opus_gemm_gfx942_wkc_512x32x16x64_1x1_16x16x16_0x0x0"

    if _gfx942_heuristic_bf16ws_band(M, N, K):
        return "opus_gemm_gfx942_splitk_legacy_bf16ws_512x128x128x64_2x4_16x16x16_0x0x0"

    if N == 384 and K >= 4096:
        if M <= 128:
            return "opus_gemm_gfx942_wkc_512x32x16x64_1x1_16x16x16_0x0x0"
        if M <= 224:
            return "opus_gemm_gfx942_splitk_p1_256x64x64x64_2x2_16x16x16_0x0x0"
        if 392 <= M <= 512:
            return "opus_gemm_gfx942_splitk_em3en4_lds1_pgr2_256x128x96x128_2x2_16x16x16_0x0x0"
        return "opus_gemm_gfx942_splitk_legacy_512x128x128x64_2x4_16x16x16_0x0x0"

    if k64_ok and N >= 4096 and K <= 3200:
        if K <= 640 and M <= 128:
            return "opus_gemm_gfx942_p1_256x64x64x64_2x2_16x16x16_0x0x0"
        return "opus_gemm_gfx942_512x128x128x64_2x4_16x16x16_0x0x0"

    if sb_ok and M >= 128:
        return "opus_gemm_gfx942_512x128x128x64_2x4_16x16x16_0x0x0"
    if N <= 256 and p1_ok:
        return "opus_gemm_gfx942_splitk_p1_256x64x64x64_2x2_16x16x16_0x0x0"
    return "opus_gemm_gfx942_splitk_legacy_512x128x128x64_2x4_16x16x16_0x0x0"


def _heuristic_a16w16_kid_gfx942(
    M: int,
    N: int,
    K: int,
    batch: int = 1,
    has_bias: bool = False,
    output_dtype: object = "bf16",
) -> int:
    """Return the original gfx942 no-tuned-row fallback kid."""
    del batch
    M, N, K = map(int, (M, N, K))
    if canonical_output_dtype(output_dtype) == "bf16_t" and not has_bias:
        symbol = _gfx942_heuristic_bf16_symbol(M, N, K)
    elif N <= 256 and K % 128 == 0:
        symbol = "opus_gemm_gfx942_splitk_p1_256x64x64x64_2x2_16x16x16_0x0x0"
    else:
        symbol = "opus_gemm_gfx942_splitk_legacy_512x128x128x64_2x4_16x16x16_0x0x0"
    return _gfx942_heuristic_kid_for_symbol(symbol)


_A16W16_HEURISTICS = {
    "gfx942": _heuristic_a16w16_kid_gfx942,
    "gfx950": _heuristic_a16w16_kid_gfx950,
    "gfx1250": _heuristic_a16w16_kid_gfx1250,
}

_UINT32_MAX_BYTES = (1 << 32) - 1


def _check_a16w16_heuristic_4g(
    *,
    arch: str,
    M: int,
    N: int,
    K: int,
    output_dtype: object,
) -> None:
    """Mirror the legacy C++ heuristic guard for 32-bit buffer descriptors."""
    if arch not in ("gfx950", "gfx1250"):
        return

    output_itemsize = {"bf16_t": 2, "fp32_t": 4}.get(
        canonical_output_dtype(output_dtype)
    )
    if output_itemsize is None:
        return

    M, N, K = map(int, (M, N, K))
    if max(M * K * 2, N * K * 2, M * N * output_itemsize) <= _UINT32_MAX_BYTES:
        return

    reason = (
        "legacy kids require a tuned 4g_safe kid"
        if arch == "gfx950"
        else "launcher gmem descriptors are 32-bit"
    )
    raise RuntimeError(
        f"opus {arch} a16w16 heuristic refuses >4 GiB shape "
        f"(M={M} N={N} K={K}): {reason}"
    )


def select_a16w16_heuristic_kid(
    *,
    arch: str,
    M: int,
    N: int,
    K: int,
    batch: int,
    has_bias: bool,
    output_dtype: object,
) -> int:
    """Select one baseline-parity A16 kid before the exact public call."""
    arch = str(arch).lower().split(":", 1)[0]
    heuristic = _A16W16_HEURISTICS.get(arch)
    if heuristic is None:
        raise ValueError(f"no OPUS a16w16 heuristic for runtime arch {arch}")
    kid = int(heuristic(M, N, K, batch, has_bias, output_dtype))
    if kid not in DEFAULT_COMPILED_KIDS_BY_ARCH.get(arch, frozenset()):
        raise RuntimeError(
            f"{arch} a16w16 heuristic returned kid {kid}, which is not in "
            "DEFAULT_COMPILED_KIDS_BY_ARCH"
        )
    return kid


def _resolve_a16w16_candidate(
    *,
    arch: str,
    M: int,
    N: int,
    K: int,
    batch: int,
    cu_num: int,
    has_bias: bool,
    input_dtype: object,
    output_dtype: object,
    kid: int,
    split_k: int,
) -> A16W16LaunchPlan | None:
    """Apply caller-only redirects, then validate one exact candidate."""
    if arch == "gfx942" and N not in GFX942_BF16WS_EXACT_N:
        if kid == 10210:
            kid = 10200
        elif kid == 10213:
            kid = 10203
        elif kid == 10216:
            return None

    try:
        return _get_cached_a16w16_launch_plan(
            arch,
            M,
            N,
            K,
            batch,
            cu_num,
            has_bias,
            input_dtype,
            output_dtype,
            kid,
            split_k,
        )
    except (ValueError, OverflowError):
        return None


def resolve_a16w16_tuned_candidate(
    *,
    arch: str,
    M: int,
    N: int,
    K: int,
    batch: int,
    cu_num: int,
    has_bias: bool,
    input_dtype: object,
    output_dtype: object,
    requested_kid: object,
    requested_split_k: object = 0,
) -> A16W16LaunchPlan | None:
    """Validate one tuned candidate without invoking a heuristic."""
    arch = str(arch).lower().split(":", 1)[0]
    try:
        split_k = int(requested_split_k)
        kid = int(requested_kid)
    except (TypeError, ValueError):
        return None
    return _resolve_a16w16_candidate(
        arch=arch,
        M=M,
        N=N,
        K=K,
        batch=batch,
        cu_num=cu_num,
        has_bias=has_bias,
        input_dtype=input_dtype,
        output_dtype=output_dtype,
        kid=kid,
        split_k=split_k,
    )


def resolve_a16w16_heuristic_candidate(
    *,
    arch: str,
    M: int,
    N: int,
    K: int,
    batch: int,
    cu_num: int,
    has_bias: bool,
    input_dtype: object,
    output_dtype: object,
    requested_split_k: object = 0,
) -> A16W16LaunchPlan | None:
    """Select and validate one per-architecture heuristic candidate."""
    arch = str(arch).lower().split(":", 1)[0]
    try:
        split_k = int(requested_split_k)
        _check_a16w16_heuristic_4g(
            arch=arch,
            M=M,
            N=N,
            K=K,
            output_dtype=output_dtype,
        )
        preferred = select_a16w16_heuristic_kid(
            arch=arch,
            M=M,
            N=N,
            K=K,
            batch=batch,
            has_bias=has_bias,
            output_dtype=output_dtype,
        )
    except (TypeError, ValueError):
        return None
    return _resolve_a16w16_candidate(
        arch=arch,
        M=M,
        N=N,
        K=K,
        batch=batch,
        cu_num=cu_num,
        has_bias=has_bias,
        input_dtype=input_dtype,
        output_dtype=output_dtype,
        kid=preferred,
        split_k=split_k,
    )


def resolve_a16w16_caller_candidate(
    *,
    arch: str,
    M: int,
    N: int,
    K: int,
    batch: int,
    cu_num: int,
    has_bias: bool,
    input_dtype: object,
    output_dtype: object,
    requested_kid: object | None,
    requested_split_k: object = 0,
) -> A16W16LaunchPlan | None:
    """Resolve a tuned kid when supplied, otherwise use the arch heuristic."""
    if requested_kid is None:
        return resolve_a16w16_heuristic_candidate(
            arch=arch,
            M=M,
            N=N,
            K=K,
            batch=batch,
            cu_num=cu_num,
            has_bias=has_bias,
            input_dtype=input_dtype,
            output_dtype=output_dtype,
            requested_split_k=requested_split_k,
        )
    return resolve_a16w16_tuned_candidate(
        arch=arch,
        M=M,
        N=N,
        K=K,
        batch=batch,
        cu_num=cu_num,
        has_bias=has_bias,
        input_dtype=input_dtype,
        output_dtype=output_dtype,
        requested_kid=requested_kid,
        requested_split_k=requested_split_k,
    )


# ---- gfx950 MXFP8 BMM tuned-row and heuristic policy ---------------------

_MXSCALE_BMM_KID_OFFSET = 8000
_MXSCALE_BMM_LOCAL_KID_MAX = 653
_MXSCALE_BMM_GLOBAL_KID_MAX = _MXSCALE_BMM_KID_OFFSET + _MXSCALE_BMM_LOCAL_KID_MAX
_TUNED_PERF_COLUMNS = ("us", "tflops", "bw", "errRatio")


@cache
def _load_mxscale_bmm_tuned(libtype: str | None = None) -> dict:
    path = AITER_CONFIGS.AITER_CONFIG_BATCHED_GEMM_A8W8_BLOCKSCALE_MXSCALE_FILE
    try:
        df = pd.read_csv(path).drop_duplicates()
    except FileNotFoundError:
        logger.warning("MXFP8 BMM tuned CSV was not found at %s", path)
        return {}

    required = {"gfx", "b", "m", "n", "k", "kernelId", "splitK"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"MXFP8 BMM tuned CSV is missing columns {sorted(missing)}")

    # The checked-in tuned data uses the original local OPUS ids (0..653),
    # while the unified public dispatcher owns the 8000 band. Keep the source
    # CSV unchanged and translate only OPUS rows in memory.
    opus_rows = (
        df["libtype"].eq("opus")
        if "libtype" in df.columns
        else pd.Series(True, index=df.index, dtype=bool)
    )
    legacy_opus_rows = opus_rows & df["kernelId"].between(0, _MXSCALE_BMM_LOCAL_KID_MAX)
    df.loc[legacy_opus_rows, "kernelId"] += _MXSCALE_BMM_KID_OFFSET

    if libtype is not None and "libtype" in df.columns:
        df = df[df["libtype"] == libtype]

    selected_opus_rows = (
        df["libtype"].eq("opus")
        if "libtype" in df.columns
        else pd.Series(True, index=df.index, dtype=bool)
    )
    invalid_opus_kids = selected_opus_rows & ~df["kernelId"].between(
        _MXSCALE_BMM_KID_OFFSET, _MXSCALE_BMM_GLOBAL_KID_MAX
    )
    if invalid_opus_kids.any():
        raise ValueError(
            "MXFP8 BMM tuned CSV must contain global OPUS kids in the "
            f"{_MXSCALE_BMM_KID_OFFSET}-{_MXSCALE_BMM_GLOBAL_KID_MAX} range "
            "or legacy local OPUS kids in the "
            f"0-{_MXSCALE_BMM_LOCAL_KID_MAX} range"
        )

    shape_keys = ["gfx", "b", "m", "n", "k"]
    duplicate_shapes = df.duplicated(subset=shape_keys, keep=False)
    if duplicate_shapes.any():
        rows = df.loc[duplicate_shapes, shape_keys].drop_duplicates().to_dict("records")
        raise RuntimeError(f"duplicate shapes across MXFP8 BMM tuned CSV files: {rows}")
    return df.set_index(shape_keys).to_dict("index")


@lru_cache(maxsize=1024)
def lookup_mxscale_bmm_config(
    b: int,
    m: int,
    n: int,
    k: int,
    *,
    libtype: str | None = None,
):
    """Return the exact or existing padded-M tuned row for one shape."""
    gfx = get_gfx()
    tuned = _load_mxscale_bmm_tuned(libtype)
    row, padded_m = None, m
    for gl in (None, 0, 1):
        padded_m = m if gl is None else get_padded_m(m, n, k, gl)
        row = tuned.get((gfx, b, padded_m, n, k))
        if row is not None:
            break

    if row is None:
        logger.info(
            "shape B:%s M:%s N:%s K:%s has no MXFP8 BMM tuned row",
            b,
            m,
            n,
            k,
        )
        return None
    if AITER_LOG_TUNED_CONFIG:
        cfg = {
            key: value for key, value in row.items() if key not in _TUNED_PERF_COLUMNS
        }
        logger.info(
            "shape B:%s M:%s N:%s K:%s uses padded_M:%s MXFP8 config %s",
            b,
            m,
            n,
            k,
            padded_m,
            cfg,
        )
    return row


@cache
def _mxscale_bmm_kid_m_align() -> dict[int, int]:
    from csrc.opus_gemm.opus_gemm_common import a8w8_mxscale_bmm_kernel_lists

    return {
        int(kid): int(instance.m_align)
        for family in a8w8_mxscale_bmm_kernel_lists
        for kid, instance in family.items()
    }


def _mxscale_bmm_kid_runs_m(kid: int, m: int) -> bool:
    align = _mxscale_bmm_kid_m_align().get(int(kid))
    return align is not None and m % align == 0


def _heuristic_mxscale_bmm_kid(g: int, m: int, n: int, k: int) -> int:
    """Choose a final global kid only when the tuned table has no usable row."""

    def divisible(value: int, divisor: int) -> bool:
        return value % divisor == 0

    if (
        divisible(n, 256)
        and divisible(k, 128)
        and (m >= 2048 or (m >= 1024 and g >= 8))
    ):
        return 8158 if 4096 <= k <= 8192 else 8150
    if m < 64:
        return 8640 if divisible(n, 64) and divisible(k, 256) else 8653
    if m <= 256 and k <= 1024 and divisible(n, 32) and divisible(k, 256):
        return 8320
    if divisible(n, 64) and divisible(k, 128):
        return 8653
    return 8000


def resolve_a8w8_mxscale_bmm_plan(
    g: int,
    m: int,
    n: int,
    k: int,
) -> tuple[int, int]:
    """Resolve one final global kid/split pair for the high-level caller."""
    config = lookup_mxscale_bmm_config(g, m, n, k)
    libtype = config.get("libtype", "opus") if config is not None else "opus"
    if libtype != "opus":
        raise NotImplementedError(
            f"MXFP8 BMM tuned row requests unsupported backend {libtype!r}"
        )

    kid = int(config["kernelId"]) if config is not None else None
    split_k = int(config["splitK"]) if config is not None else 1
    if kid is None or not _mxscale_bmm_kid_runs_m(kid, m):
        kid = _heuristic_mxscale_bmm_kid(g, m, n, k)
        split_k = 1
    return kid, split_k


__all__ = [
    "lookup_a16w16_opus_config",
    "lookup_mxscale_bmm_config",
    "resolve_a8w8_mxscale_bmm_plan",
    "resolve_a16w16_caller_candidate",
    "resolve_a16w16_heuristic_candidate",
    "resolve_a16w16_tuned_candidate",
    "select_a16w16_heuristic_kid",
]
