# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Select tuned or heuristic kernels for OPUS A16W16 and MXFP8 BMM."""

from __future__ import annotations

from functools import cache, lru_cache

import pandas as pd

from aiter import logger
from csrc.opus_gemm.opus_gemm_common import (
    DEFAULT_COMPILED_KIDS_BY_NAMESPACE,
    GFX942_BF16WS_EXACT_N,
    MX32_KID_STRIDE,
    canonical_output_dtype,
    get_kernel_instance,
    get_registry,
    kid_for_name,
    parse_integer,
)

from ...jit.core import AITER_CONFIGS, AITER_LOG_TUNED_CONFIG
from ...jit.utils.chip_info import get_cu_num
from ...jit.utils.chip_info import get_gfx_runtime as get_gfx
from ..gemm_op_common import get_padded_m, with_mxscale_w_scale_block
from ._arch import GFX942, GFX950, GFX1250
from .launch_plan import (
    A16W16LaunchPlan,
    _get_cached_a8w8_mxscale_bmm_plan,
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
    """Load only OPUS rows from the merged A16W16 tuned config."""
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

    df = df[
        df["libtype"].eq("opus") & df["scaleAB"].eq(False) & df["bpreshuffle"].eq(False)
    ]
    rows = []
    for row in df.to_dict("records"):
        try:
            kid = parse_integer(row["solidx"], name="saved kernel id")
            cu_num = parse_integer(row["cu_num"], name="saved CU count", minimum=1)
            instance = get_kernel_instance(row["gfx"], "a16w16", kid, cu_num=cu_num)
            name = str(row.get("kernelName", ""))
            if instance is None or (
                name not in ("", "nan", "None") and name != instance.name
            ):
                continue
            row["solidx"] = kid
            row["splitK"] = parse_integer(row["splitK"], name="splitK")
        except (ValueError, KeyError, TypeError):
            continue
        rows.append(row)
    invalid_rows = len(df) - len(rows)
    if invalid_rows:
        logger.warning(
            "Skipping %d invalid OPUS row(s) in A16W16 tuned CSV %r",
            invalid_rows,
            path,
        )
    if not rows:
        return {}
    df = pd.DataFrame(rows)

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
            return 60
        return 14
    if M <= 64:
        if M % 64 == 0 and N % 32 == 0 and K % 128 == 0:
            return 58
        return 12
    if M <= 128:
        if M % 64 == 0 and N % 64 == 0 and K % 64 == 0:
            return 52
        return 6
    if split_barrier_ok and not has_bias:
        if M % 256 == 0 and N % 256 == 0 and K % 64 == 0:
            return 76
        return 30
    if M % 64 == 0 and N % 64 == 0 and K % 64 == 0:
        return 52
    return 6


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
            return 7
        if N % 64 == 0:
            return 6
        if N % 32 == 0:
            return 5
    if N % 128 == 0:
        return 4
    if N % 64 == 0:
        return 3
    return 0


@lru_cache(maxsize=8)
def _gfx942_heuristic_symbol_to_kid(cu_num=None) -> dict[str, int]:
    """Map heuristic symbols in the selected gfx942 CU table."""
    cu_num = get_cu_num() if cu_num is None else cu_num
    result: dict[str, int] = {}
    for kid in DEFAULT_COMPILED_KIDS_BY_NAMESPACE.get((GFX942, cu_num), ()):
        instance = get_kernel_instance(GFX942, "a16w16", kid, cu_num=cu_num)
        if instance is None:
            raise RuntimeError(f"gfx942 heuristic kid {kid} has no a16w16 instance")
        previous = result.setdefault(instance.name, int(kid))
        if previous != kid:
            raise RuntimeError(
                f"duplicate gfx942 launcher symbol {instance.name!r}: "
                f"kids {previous} and {kid}"
            )
    return result


def _gfx942_heuristic_kid_for_symbol(symbol: str, cu_num=None) -> int:
    try:
        return _gfx942_heuristic_symbol_to_kid(cu_num)[symbol]
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
    *,
    cu_num: int | None = None,
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
    return _gfx942_heuristic_kid_for_symbol(symbol, cu_num)


_A16W16_HEURISTICS = {
    GFX942: _heuristic_a16w16_kid_gfx942,
    GFX950: _heuristic_a16w16_kid_gfx950,
    GFX1250: _heuristic_a16w16_kid_gfx1250,
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
    if arch not in (GFX950, GFX1250):
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
        if arch == GFX950
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
    cu_num: int | None = None,
) -> int:
    """Select one baseline-parity A16 kid before the exact public call."""
    arch = str(arch).lower().split(":", 1)[0]
    heuristic = _A16W16_HEURISTICS.get(arch)
    if heuristic is None:
        raise ValueError(f"no OPUS a16w16 heuristic for runtime arch {arch}")
    cu_num = get_cu_num() if cu_num is None else cu_num
    args = (M, N, K, batch, has_bias, output_dtype)
    kid = int(heuristic(*args, cu_num=cu_num) if arch == GFX942 else heuristic(*args))
    if kid not in DEFAULT_COMPILED_KIDS_BY_NAMESPACE.get((arch, cu_num), ()):
        raise RuntimeError(
            f"{arch}/{cu_num} a16w16 heuristic returned kid {kid} outside its default compile set"
        )
    instance = get_kernel_instance(arch, "a16w16", kid, cu_num=cu_num)
    if instance is None:
        raise ValueError(f"unknown OPUS a16w16 kernel {arch}/{cu_num}/{kid}")
    if instance.max_m is not None and int(M) > instance.max_m:
        raise RuntimeError(
            f"opus {arch} a16w16 heuristic refuses M={M}: "
            f"kid {kid} requires M <= {instance.max_m}. "
            "Tune this shape for a non-split-K kernel."
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
    if arch == GFX942 and N not in GFX942_BF16WS_EXACT_N:
        instance = get_kernel_instance(arch, "a16w16", kid, cu_num=cu_num)
        if instance is not None and instance.splitk_workspace_dtype == "bf16_t":
            if "_bf16ws_" not in instance.name:
                return None
            try:
                kid = kid_for_name(
                    "a16w16", arch, cu_num, instance.name.replace("_bf16ws_", "_")
                )
            except ValueError:
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
            cu_num=cu_num,
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

_TUNED_PERF_COLUMNS = ("us", "tflops", "bw", "errRatio")
_C_INT_MAX = (1 << 31) - 1


def _parse_mxscale_bmm_tuned_split_k(value: object) -> int:
    """Require one saved split-K value that can cross the C++ int ABI."""
    split_k = parse_integer(value, name="splitK", minimum=1)
    if split_k > _C_INT_MAX:
        raise ValueError(f"splitK must be a positive integer, got {value!r}")
    return split_k


@cache
def _load_mxscale_bmm_tuned(
    libtype: str | None = None, bpreshuffle: bool = False
) -> dict:
    # Select the tuned CSV by the caller's weight layout.
    path = (
        AITER_CONFIGS.AITER_CONFIG_BATCHED_GEMM_A8W8_BLOCKSCALE_MXSCALE_BPRESHUFFLE_FILE
        if bpreshuffle
        else AITER_CONFIGS.AITER_CONFIG_BATCHED_GEMM_A8W8_BLOCKSCALE_MXSCALE_FILE
    )
    try:
        df = pd.read_csv(path).drop_duplicates()
    except FileNotFoundError:
        logger.warning("MXFP8 BMM tuned CSV was not found at %s", path)
        return {}

    required = {
        "gfx",
        "cu_num",
        "b",
        "m",
        "n",
        "k",
        "kernelId",
        "splitK",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"MXFP8 BMM tuned CSV is missing columns {sorted(missing)}")
    df = with_mxscale_w_scale_block(df, path)

    if libtype is not None and "libtype" in df.columns:
        df = df[df["libtype"] == libtype].copy()

    opus_rows = (
        df["libtype"].eq("opus")
        if "libtype" in df.columns
        else pd.Series(True, index=df.index, dtype=bool)
    )
    valid_opus_rows = pd.Series(False, index=df.index, dtype=bool)
    for index in df.index[opus_rows]:
        try:
            kid = parse_integer(df.at[index, "kernelId"], name="saved kernel id")
            cu_num = parse_integer(
                df.at[index, "cu_num"], name="saved CU count", minimum=1
            )
            arch = str(df.at[index, "gfx"]).lower().split(":", 1)[0]
            instance = get_kernel_instance(arch, "a8w8_mxscale_bmm", kid, cu_num=cu_num)
            name = str(df.at[index, "kernelName"]) if "kernelName" in df else ""
            if instance is None or (
                name not in ("", "nan", "None") and name != instance.name
            ):
                raise ValueError("saved kernel symbol disagrees with the catalog")
            if (
                f"{instance.GROUP_N}x{instance.GROUP_K}"
                != df.at[index, "w_scale_block"]
            ):
                raise ValueError("kid and weight-scale block disagree")
            split_k = _parse_mxscale_bmm_tuned_split_k(df.at[index, "splitK"])
            batch, m, n, k = (
                int(df.at[index, column]) for column in ("b", "m", "n", "k")
            )
            _get_cached_a8w8_mxscale_bmm_plan(
                arch,
                kid,
                "bf16",
                m,
                batch,
                n,
                k,
                split_k,
                cu_num,
            )
        except (TypeError, ValueError, OverflowError):
            continue
        else:
            df.at[index, "gfx"] = arch
            df.at[index, "cu_num"] = cu_num
            df.at[index, "kernelId"] = kid
            df.at[index, "splitK"] = split_k
            valid_opus_rows.at[index] = True

    invalid_opus_rows = opus_rows & ~valid_opus_rows
    if invalid_opus_rows.any():
        logger.warning(
            "Skipping %d invalid OPUS row(s) in MXFP8 BMM tuned CSV %r: "
            "kernelId, splitK and shape must form a compatible registered "
            "MXFP8 BMM launch",
            int(invalid_opus_rows.sum()),
            path,
        )
        df = df.loc[~invalid_opus_rows].copy()

    keys = ["gfx", "cu_num", "b", "m", "n", "k", "w_scale_block"]
    duplicate_shapes = df.duplicated(subset=keys, keep=False)
    if duplicate_shapes.any():
        rows = df.loc[duplicate_shapes, keys].drop_duplicates().to_dict("records")
        raise RuntimeError(f"duplicate shapes across MXFP8 BMM tuned CSV files: {rows}")
    return df.set_index(keys).to_dict("index")


@lru_cache(maxsize=1024)
def lookup_mxscale_bmm_config(
    b: int,
    m: int,
    n: int,
    k: int,
    *,
    w_scale_block: str = "128x128",
    libtype: str | None = None,
    bpreshuffle: bool = False,
    gfx: str | None = None,
    cu_num: int | None = None,
):
    """Find a tuned row by device, shape, weight layout and scale block."""
    if gfx is None:
        gfx = get_gfx()
    if cu_num is None:
        cu_num = get_cu_num()
    tuned = _load_mxscale_bmm_tuned(libtype, bpreshuffle)
    row, padded_m = None, m
    for gl in (None, 0, 1):
        padded_m = m if gl is None else get_padded_m(m, n, k, gl)
        row = tuned.get((gfx, cu_num, b, padded_m, n, k, w_scale_block))
        if row is not None:
            break

    if row is None:
        logger.info(
            "shape B:%s M:%s N:%s K:%s w_scale %s has no MXFP8 BMM tuned row",
            b,
            m,
            n,
            k,
            w_scale_block,
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


def _validate_mxscale_bmm_group_size(group_size: int) -> None:
    if group_size not in (32, 128):
        raise ValueError(f"group_size must be 32 or 128, got {group_size!r}")


def mxscale_bmm_group_of_block(w_scale_block: str) -> int:
    """Return K grouping; use groups_of_block for weight-scale addressing."""
    return mxscale_bmm_groups_of_block(w_scale_block)[1]


def mxscale_bmm_groups_of_block(w_scale_block: str) -> tuple[int, int]:
    groups = {"1x32": (1, 32), "32x32": (32, 32), "128x128": (128, 128)}
    if w_scale_block not in groups:
        raise ValueError(f"unsupported OPUS MXFP8 BMM scale block {w_scale_block!r}")
    return groups[w_scale_block]


def _heuristic_mxscale_bmm_kid(
    g: int,
    m: int,
    n: int,
    k: int,
    *,
    group_size: int = 128,
    gfx: str | None = None,
    cu_num: int | None = None,
) -> int:
    """Select a row-major fallback in the current gfx/CU namespace."""
    _validate_mxscale_bmm_group_size(group_size)

    def pick(kid):
        return kid + MX32_KID_STRIDE if group_size == 32 else kid

    if (
        group_size == 128
        and n % 256 == 0
        and k % 128 == 0
        and (m >= 2048 or (m >= 1024 and g >= 8))
    ):
        seed = 16 if 4096 <= k <= 8192 else 15
    elif m < 64:
        seed = pick(140) if n % 64 == 0 and k % 256 == 0 else pick(144)
    elif m <= 256 and k <= 1024 and n % 32 == 0 and k % 256 == 0:
        seed = pick(71)
    elif n % 64 == 0 and k % 128 == 0:
        seed = pick(144)
    else:
        seed = pick(0)
    gfx = gfx or get_gfx()
    cu_num = get_cu_num() if cu_num is None else cu_num
    instance = get_kernel_instance(gfx, "a8w8_mxscale_bmm", seed, cu_num=cu_num)
    if instance is None:
        raise ValueError(f"no OPUS MXFP8 BMM heuristic for {gfx}")
    return seed


# Preshuffled fallback seeds by M band and scale group.
_BPRESHUFFLE_FALLBACK_BANDS = {
    128: (
        (64, (28, 88, 24, 80, 138, 139)),
        (512, (24, 27, 80, 88, 138, 139)),
        (None, (34, 83, 27, 80, 88, 139, 138)),
    ),
    32: (
        (64, (1028, 1088, 1024, 1080)),
        (512, (1024, 1027, 1080, 1088)),
        (None, (1034, 1083, 1027, 1080, 1088)),
    ),
}


def _heuristic_mxscale_bmm_bpreshuffle_kid(
    g: int,
    m: int,
    n: int,
    k: int,
    *,
    group_size: int = 128,
    group_n: int | None = None,
    output_dtype="bf16",
    gfx: str | None = None,
    cu_num: int | None = None,
) -> int | None:
    """A preshuffled-B kid that runs this shape, for a shape the table lacks.

    Per-row K32 scales prefer larger tiles for sufficiently large output grids.
    Every candidate is checked for admissibility; None when no candidate runs.
    """
    gfx = gfx or get_gfx()
    cu_num = get_cu_num() if cu_num is None else cu_num
    scope = {"gfx": gfx, "cu_num": cu_num}
    _validate_mxscale_bmm_group_size(group_size)
    group_n = group_size if group_n is None else group_n
    block = f"{group_n}x{group_size}"
    mxscale_bmm_groups_of_block(block)

    if block == "1x32":
        # Prefer wider tiles when the 1x32 output grid is large enough.
        wide_tiles = g * ((m + 63) // 64) * (n // 64)
        preferred = (242,) if m >= 64 and wide_tiles >= 256 else ()
        if m >= 256:
            preferred += (234,)
        candidates = (*preferred, 231, 235, 239, 241)
    else:
        candidates = next(
            c
            for bound, c in _BPRESHUFFLE_FALLBACK_BANDS[group_size]
            if bound is None or m <= bound
        )
    for kid in candidates:
        if not (
            mxscale_bmm_kid_block(kid, **scope) == block
            and mxscale_bmm_kid_takes_b_layout(kid, True, **scope)
            and mxscale_bmm_kid_takes_plain_scales(kid, **scope)
        ):
            continue
        try:
            _get_cached_a8w8_mxscale_bmm_plan(
                gfx, kid, output_dtype, m, g, n, k, 1, cu_num=cu_num
            )
        except (TypeError, ValueError, OverflowError):
            continue
        return kid
    return None


# Read kernel layout, scale grouping and M alignment from the registry.


def _mxscale_bmm_entry(kid: int, gfx: str | None, cu_num: int | None):
    gfx = gfx or get_gfx()
    cu_num = get_cu_num() if cu_num is None else cu_num
    return get_registry("a8w8_mxscale_bmm", gfx, cu_num).get(int(kid))


def mxscale_bmm_kid_runs_m(kid: int, m: int, *, gfx=None, cu_num=None) -> bool:
    """True iff kid's launcher accepts this M. Unknown kid -> False."""
    entry = _mxscale_bmm_entry(kid, gfx, cu_num)
    return entry is not None and m % entry.m_align == 0


def mxscale_bmm_kid_takes_b_layout(
    kid: int, b_preshuffled: bool, *, gfx=None, cu_num=None
) -> bool:
    """True iff kid wants B in the layout the caller says it has."""
    entry = _mxscale_bmm_entry(kid, gfx, cu_num)
    return entry is not None and entry.needs_preshuffled_b == bool(b_preshuffled)


def mxscale_bmm_kid_takes_plain_scales(kid: int, *, gfx=None, cu_num=None) -> bool:
    """True iff kid reads x_scale / w_scale as the public entry passes them."""
    entry = _mxscale_bmm_entry(kid, gfx, cu_num)
    return entry is not None and not (
        entry.needs_mpacked_sfa or entry.needs_shuffle_scale
    )


def mxscale_bmm_kid_group(kid: int, *, gfx=None, cu_num=None) -> int | None:
    """The kid's quantisation block (GROUP_K), or None for an unknown kid."""
    entry = _mxscale_bmm_entry(kid, gfx, cu_num)
    return None if entry is None else entry.GROUP_K


def mxscale_bmm_kid_block(kid: int, *, gfx=None, cu_num=None) -> str | None:
    """Complete weight-scale block, including independent N grouping."""
    entry = _mxscale_bmm_entry(kid, gfx, cu_num)
    return None if entry is None else f"{entry.GROUP_N}x{entry.GROUP_K}"


def resolve_a8w8_mxscale_bmm_plan(
    g: int,
    m: int,
    n: int,
    k: int,
    *,
    w_scale_block: str = "128x128",
    gfx: str | None = None,
    cu_num: int | None = None,
) -> tuple[int, int]:
    """Resolve one final local kid/split pair for the caller's scale blocks."""
    gfx = gfx or get_gfx()
    cu_num = get_cu_num() if cu_num is None else cu_num
    scope = {"gfx": gfx, "cu_num": cu_num}
    group_size = mxscale_bmm_group_of_block(w_scale_block)
    if w_scale_block == "1x32":
        raise ValueError("1x32 OPUS BMM requires the preshuffled-weight entry")
    config = lookup_mxscale_bmm_config(
        g, m, n, k, w_scale_block=w_scale_block, gfx=gfx, cu_num=cu_num
    )
    libtype = config.get("libtype", "opus") if config is not None else "opus"
    if libtype != "opus":
        raise NotImplementedError(
            f"MXFP8 BMM tuned row requests unsupported backend {libtype!r}"
        )

    if config is not None:
        try:
            kid = int(config["kernelId"])
            entry = _mxscale_bmm_entry(kid, gfx, cu_num)
            if (
                entry is None
                or entry.needs_preshuffled_b
                or entry.needs_mpacked_sfa
                or entry.needs_shuffle_scale
            ):
                raise ValueError(
                    f"kid {kid} wants a preshuffled B or host-rearranged scales"
                )
            split_k = int(config["splitK"])
            plan = _get_cached_a8w8_mxscale_bmm_plan(
                gfx, kid, "bf16", m, g, n, k, split_k, cu_num=cu_num
            )
        except (TypeError, ValueError, OverflowError) as exc:
            logger.warning(
                "Ignoring invalid OPUS MXFP8 BMM tuned row for "
                "B:%s M:%s N:%s K:%s (kernelId=%r, splitK=%r): %s; "
                "using heuristic fallback",
                g,
                m,
                n,
                k,
                config.get("kernelId"),
                config.get("splitK"),
                exc,
            )
        else:
            return plan.resolved_kid, plan.abi_split_k

    kid = _heuristic_mxscale_bmm_kid(g, m, n, k, group_size=group_size, **scope)
    return kid, 1


__all__ = [
    "lookup_a16w16_opus_config",
    "lookup_mxscale_bmm_config",
    "mxscale_bmm_group_of_block",
    "mxscale_bmm_groups_of_block",
    "mxscale_bmm_kid_block",
    "mxscale_bmm_kid_group",
    "mxscale_bmm_kid_runs_m",
    "mxscale_bmm_kid_takes_b_layout",
    "mxscale_bmm_kid_takes_plain_scales",
    "resolve_a8w8_mxscale_bmm_plan",
    "resolve_a16w16_caller_candidate",
    "resolve_a16w16_heuristic_candidate",
    "resolve_a16w16_tuned_candidate",
    "select_a16w16_heuristic_kid",
]
