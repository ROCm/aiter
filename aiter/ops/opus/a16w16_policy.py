# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Caller-side A16W16 tuned-candidate and heuristic policy.

This module runs before the unified exact-kid :func:`opus_gemm` entry.  It may
validate a tuned candidate, select a per-architecture heuristic candidate, or
reject either so the upper caller can continue its fallback policy.  The
public launcher and C++ dispatcher never call this module and therefore never
replace a caller-supplied exact kid.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

from csrc.opus_gemm.opus_gemm_common import (
    DEFAULT_COMPILED_KIDS_BY_ARCH,
    GFX942_BF16WS_EXACT_N,
    get_kernel_instance,
)

if TYPE_CHECKING:
    from .gemm_op_a16w16 import LaunchConfig


def _output_dtype_name(dtype: object) -> str:
    value = str(dtype).lower()
    if value in {"bf16", "bfloat16", "bf16_t", "torch.bfloat16"}:
        return "bf16"
    if value in {"fp32", "float", "float32", "fp32_t", "torch.float32"}:
        return "fp32"
    return value


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
            raise RuntimeError(
                f"gfx942 heuristic kid {kid} has no a16w16 instance"
            )
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
        K >= 4096
        and K % 64 == 0
        and 104 <= M <= 608
        and (N == 256 or 512 <= N <= 2048)
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

    if K >= 512 and k64_ok and (
        N <= 64 or (M <= 128 and N <= 1024) or (M <= 8 and N <= 1536)
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
    if _output_dtype_name(output_dtype) == "bf16" and not has_bias:
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
) -> LaunchConfig | None:
    """Apply caller-only redirects, then validate one exact candidate."""
    if arch == "gfx942" and N not in GFX942_BF16WS_EXACT_N:
        if kid == 10210:
            kid = 10200
        elif kid == 10213:
            kid = 10203
        elif kid == 10216:
            return None

    # Lazy import keeps the execution module free to re-export compatibility
    # aliases without creating an import cycle.  The exact validator itself is
    # shared with the launch path and performs no allocation or launch.
    from .gemm_op_a16w16 import _resolve_exact_a16w16_config

    try:
        return _resolve_exact_a16w16_config(
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
    except ValueError:
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
) -> LaunchConfig | None:
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
) -> LaunchConfig | None:
    """Select and validate one per-architecture heuristic candidate."""
    arch = str(arch).lower().split(":", 1)[0]
    try:
        split_k = int(requested_split_k)
        kid = select_a16w16_heuristic_kid(
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
        kid=kid,
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
) -> LaunchConfig | None:
    """Compatibility wrapper for the former combined caller resolver."""
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


__all__ = [
    "resolve_a16w16_caller_candidate",
    "resolve_a16w16_heuristic_candidate",
    "resolve_a16w16_tuned_candidate",
    "select_a16w16_heuristic_kid",
]
