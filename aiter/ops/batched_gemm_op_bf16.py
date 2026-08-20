# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import functools
from typing import Any

import pandas as pd
import torch
from torch import Tensor

from aiter import logger

from ..jit.core import (
    AITER_CONFIGS,
    AITER_LOG_TUNED_CONFIG,
    compile_ops,
)
from ..jit.utils.chip_info import get_cu_num
from ..jit.utils.chip_info import get_gfx_runtime as get_gfx
from ..utility import dtypes


def gen_batched_gemm_bf16_tune_fake_tensor(
    XQ: Tensor, WQ: Tensor, out: Tensor, kernelId: int, splitK: int = 0
) -> Tensor:
    return out


@compile_ops(
    "module_batched_gemm_bf16",
    fc_name="batched_gemm_bf16",
    gen_fake=gen_batched_gemm_bf16_tune_fake_tensor,
)
def batched_gemm_bf16(
    XQ: Tensor, WQ: Tensor, out: Tensor, bias: Tensor | None = None, splitK: int = 0
) -> Tensor: ...


@functools.lru_cache(maxsize=1024)
def compute_batched_gemm_SplitK(
    M: int, N: int, K: int, tile_m: int, tile_n: int, tile_k: int
):

    cu_num = get_cu_num()
    tile_num = ((M + tile_m - 1) // tile_m) * ((N + tile_n - 1) // tile_n)
    cusPerTile = cu_num / tile_num
    splitK = 0
    while cusPerTile >= pow(2, splitK + 1) and (pow(2, splitK + 1) * tile_k) < 2 * K:
        splitK += 1
    return splitK


def _canonical_batched_gemm_bf16_libtype(value: object) -> str:
    """Return the canonical backend name for old and mixed tuned CSVs.

    ``bf16_tuned_batched_gemm.csv`` predates multi-backend tuning, so an
    absent/empty ``libtype`` cell means CK.  ``jit.core`` may also fill a
    missing column with integer zero while merging legacy model configs.
    """
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return "ck"
    token = str(value).strip().lower()
    if token in ("", "0", "nan", "none", "null"):
        return "ck"
    return token


@functools.lru_cache(maxsize=1)
def _load_batched_gemm_bf16_configs() -> tuple[dict[tuple[Any, ...], dict], bool]:
    config_file = AITER_CONFIGS.AITER_CONFIG_BF16_BATCHED_GEMM_FILE
    rows = pd.read_csv(config_file).drop_duplicates()
    if "libtype" not in rows.columns:
        rows["libtype"] = "ck"
    else:
        rows["libtype"] = rows["libtype"].map(
            _canonical_batched_gemm_bf16_libtype
        )

    # Use (gfx, cu_num, B, M, N, K) when possible.  The legacy fallback is
    # retained for user-provided CSVs created before gfx became part of the
    # tuned key.
    has_gfx = "gfx" in rows.columns
    if has_gfx:
        index = ["gfx", "cu_num", "B", "M", "N", "K"]
    else:
        logger.warning(
            f"{config_file} has no 'gfx' column — falling back to "
            "cu_num-only key. Re-run the tuner or migrate the CSV."
        )
        index = ["cu_num", "B", "M", "N", "K"]
    return rows.set_index(index).to_dict("index"), has_gfx


def _parse_ck_batched_gemm_config(config: dict) -> dict:
    """Add the legacy CK tile fields without mutating the cached CSV row."""
    result = dict(config)
    kernel_name = str(result.get("kernelName", ""))
    try:
        mnk = kernel_name.split("_")[2].split("x")[1:]
        result["tile_m"] = int(mnk[0])
        result["tile_n"] = int(mnk[1])
        result["tile_k"] = int(mnk[2])
    except (IndexError, TypeError, ValueError):
        # Runtime only needs splitK.  Keep accepting hand-written/legacy CK
        # rows whose descriptive kernelName does not use the standard format.
        pass
    return result


@functools.lru_cache(maxsize=4096)
def _get_batched_gemm_bf16_config_for_device(
    gfx: str,
    cu_num: int,
    B: int,
    M: int,
    N: int,
    K: int,
) -> dict | None:
    rows, has_gfx = _load_batched_gemm_bf16_configs()
    key = (
        (str(gfx), int(cu_num), int(B), int(M), int(N), int(K))
        if has_gfx
        else (int(cu_num), int(B), int(M), int(N), int(K))
    )
    config = rows.get(key)
    if config is None:
        return None
    result = dict(config)
    result["libtype"] = _canonical_batched_gemm_bf16_libtype(
        result.get("libtype")
    )
    if result["libtype"] == "ck":
        result = _parse_ck_batched_gemm_config(result)
    return result


def _clear_batched_gemm_bf16_config_caches() -> None:
    """Clear every cache that can retain a tuned BF16 BMM CSV row."""
    _load_batched_gemm_bf16_configs.cache_clear()
    _get_batched_gemm_bf16_config_for_device.cache_clear()
    get_BatchedGEMM_config.cache_clear()
    get_CKBatchedGEMM_config.cache_clear()


@functools.lru_cache(maxsize=1024)
def get_BatchedGEMM_config(
    B: int,
    M: int,
    N: int,
    K: int,
) -> dict | None:
    """Return the tuned CK/OPUS winner for one BF16 BMM shape."""
    return _get_batched_gemm_bf16_config_for_device(
        get_gfx(), get_cu_num(), B, M, N, K
    )


@functools.lru_cache(maxsize=1024)
def get_CKBatchedGEMM_config(
    B: int,
    M: int,
    N: int,
    K: int,
):
    config = get_BatchedGEMM_config(B, M, N, K)
    if config is not None and config["libtype"] != "ck":
        return None
    if config is not None:
        if AITER_LOG_TUNED_CONFIG:
            logger.info(
                f"shape is B:{B}, M:{M}, N:{N}, K:{K} dtype is bf16, "
                f"is tuned for CK on cu_num = {get_cu_num()} in "
                f"{AITER_CONFIGS.AITER_CONFIG_BF16_BATCHED_GEMM_FILE}, "
                f"kernel name is {config['kernelName']}, splitK is "
                f"{config['splitK']}!"
            )
    else:
        logger.info(
            f"shape is B:{B}, M:{M}, N:{N}, K:{K} dtype is bf16, "
            "not found tuned CK config, will use the CK default config!"
        )
    return config


def _bmm_shape(XQ: Tensor, WQ: Tensor) -> tuple[int, int, int, int]:
    if XQ.dim() != 3 or WQ.dim() != 3:
        raise ValueError(
            "batched_gemm_bf16_OPUS expects XQ [B,M,K] and WQ [B,N,K]"
        )
    B, M, K = map(int, XQ.shape)
    N = int(WQ.shape[1])
    return B, M, N, K


def _config_opus_kid(config: dict) -> int | None:
    for key in ("kernelId", "solidx"):
        value = config.get(key)
        if value is not None and pd.notna(value):
            try:
                return int(value)
            except (TypeError, ValueError):
                return None
    return None


def _config_split_k(config: dict, default: int = 0) -> int:
    value = config.get("splitK", default)
    if value is None or not pd.notna(value):
        return int(default)
    return int(value)


def _resolve_opus_bf16_bmm_candidate(
    *,
    arch: str,
    cu_num: int,
    B: int,
    M: int,
    N: int,
    K: int,
    has_bias: bool,
    input_dtype: torch.dtype,
    output_dtype: torch.dtype,
    kid: int | None,
    split_k: int,
):
    from .opus.policy import (
        resolve_a16w16_heuristic_candidate,
        resolve_a16w16_tuned_candidate,
    )

    common = dict(
        arch=arch,
        M=M,
        N=N,
        K=K,
        batch=B,
        cu_num=cu_num,
        has_bias=has_bias,
        input_dtype=input_dtype,
        output_dtype=output_dtype,
        requested_split_k=split_k,
    )
    if kid is None:
        return resolve_a16w16_heuristic_candidate(**common)
    return resolve_a16w16_tuned_candidate(
        **common,
        requested_kid=kid,
    )


def _launch_batched_gemm_bf16_opus(
    XQ: Tensor,
    WQ: Tensor,
    bias: Tensor | None,
    dtype: torch.dtype,
    *,
    kid: int,
    split_k: int,
) -> Tensor:
    from .opus import opus_bmm

    B, M, N, _K = _bmm_shape(XQ, WQ)
    Y = torch.empty((B, M, N), dtype=dtype, device=XQ.device)
    return opus_bmm(
        XQ,
        WQ,
        Y,
        kid=int(kid),
        bias=bias,
        split_k=int(split_k),
    )


def batched_gemm_bf16_CK(
    XQ: Tensor,
    WQ: Tensor,
    bias: Tensor | None = None,
    dtype=dtypes.bf16,
    splitK: int | None = None,
):
    assert dtype in [
        dtypes.bf16,
        dtypes.fp16,
    ], f"Output {dtype=} is currently not supported in batched_gemm_bf16"

    b = XQ.shape[0]
    m = XQ.shape[1]
    n = WQ.shape[1]
    k = XQ.shape[2]
    ck_config = get_CKBatchedGEMM_config(b, m, n, k)
    if splitK is None:
        if ck_config is not None:
            splitK = ck_config["splitK"]
        else:
            splitK = 0
    Y = torch.empty(b, m, n, dtype=dtype, device=XQ.device)
    return batched_gemm_bf16(XQ, WQ, Y, bias, splitK)


def batched_gemm_bf16_OPUS(
    XQ: Tensor,
    WQ: Tensor,
    bias: Tensor | None = None,
    dtype: torch.dtype = dtypes.bf16,
    *,
    kid: int | None = None,
    splitK: int | None = None,
) -> Tensor:
    """Run BF16 batch-first BMM through the OPUS A16W16 backend.

    An explicit ``kid`` is treated as an exact candidate and validated by the
    shared A16W16 policy.  Without one, an OPUS row from the mixed BF16 BMM
    tuned CSV is preferred; if no valid OPUS row exists, the normal A16W16
    heuristic supplies the exact kid.  Workspace planning/allocation remains
    entirely inside :func:`opus_bmm`.
    """
    from .opus._arch import _device_arch_and_cu

    B, M, N, K = _bmm_shape(XQ, WQ)
    arch, cu_num = _device_arch_and_cu(XQ.device)
    requested_kid = None if kid is None else int(kid)
    requested_split_k = 0 if splitK is None else int(splitK)

    if requested_kid is None:
        config = _get_batched_gemm_bf16_config_for_device(
            arch, cu_num, B, M, N, K
        )
        if config is not None and config["libtype"] == "opus":
            requested_kid = _config_opus_kid(config)
            if splitK is None:
                requested_split_k = _config_split_k(config)

    plan = _resolve_opus_bf16_bmm_candidate(
        arch=arch,
        cu_num=cu_num,
        B=B,
        M=M,
        N=N,
        K=K,
        has_bias=bias is not None,
        input_dtype=XQ.dtype,
        output_dtype=dtype,
        kid=requested_kid,
        split_k=requested_split_k,
    )
    if plan is None and kid is None and requested_kid is not None:
        # A stale OPUS tuned row must not make the explicit OPUS API unusable.
        # Discard the entire tuned pair and ask the caller-side heuristic for a
        # fresh exact candidate.  The mixed automatic dispatcher below instead
        # falls back to CK, preserving the measured backend choice contract.
        logger.warning(
            "discarding invalid tuned OPUS BF16 BMM candidate "
            f"kid={requested_kid}, splitK={requested_split_k} for "
            f"shape ({B}, {M}, {N}, {K}); using the OPUS heuristic"
        )
        requested_kid = None
        requested_split_k = 0 if splitK is None else int(splitK)
        plan = _resolve_opus_bf16_bmm_candidate(
            arch=arch,
            cu_num=cu_num,
            B=B,
            M=M,
            N=N,
            K=K,
            has_bias=bias is not None,
            input_dtype=XQ.dtype,
            output_dtype=dtype,
            kid=None,
            split_k=requested_split_k,
        )
    if plan is None:
        source = f"kid={kid}" if kid is not None else "the OPUS heuristic"
        raise ValueError(
            f"no valid OPUS A16W16 BMM candidate from {source} for "
            f"shape (B={B}, M={M}, N={N}, K={K}), "
            f"input={XQ.dtype}, output={dtype}, bias={bias is not None}"
        )

    if AITER_LOG_TUNED_CONFIG and requested_kid is not None:
        logger.info(
            f"shape is B:{B}, M:{M}, N:{N}, K:{K} dtype is bf16, "
            f"using OPUS kid {plan.resolved_kid}, splitK "
            f"{requested_split_k}"
        )
    return _launch_batched_gemm_bf16_opus(
        XQ,
        WQ,
        bias,
        dtype,
        kid=plan.resolved_kid,
        split_k=requested_split_k,
    )


def batched_gemm_bf16_tuned(
    XQ: Tensor,
    WQ: Tensor,
    bias: Tensor | None = None,
    dtype: torch.dtype = dtypes.bf16,
    splitK: int | None = None,
) -> Tensor:
    """Run the tuned CK/OPUS BF16 BMM winner, defaulting safely to CK.

    Legacy CSV rows have no ``libtype`` and therefore remain CK rows.  OPUS is
    selected only by an explicit ``libtype=opus`` winner produced by the
    multi-backend tuner.  If that row is stale for the requested dtype/bias or
    current registry, the operation uses the existing CK fallback instead of
    silently running an unmeasured OPUS heuristic.
    """
    from .opus._arch import _device_arch_and_cu

    B, M, N, K = _bmm_shape(XQ, WQ)
    arch, cu_num = _device_arch_and_cu(XQ.device)
    config = _get_batched_gemm_bf16_config_for_device(
        arch, cu_num, B, M, N, K
    )
    if config is not None and config["libtype"] == "opus":
        requested_kid = _config_opus_kid(config)
        requested_split_k = (
            _config_split_k(config) if splitK is None else int(splitK)
        )
        if requested_kid is not None:
            plan = _resolve_opus_bf16_bmm_candidate(
                arch=arch,
                cu_num=cu_num,
                B=B,
                M=M,
                N=N,
                K=K,
                has_bias=bias is not None,
                input_dtype=XQ.dtype,
                output_dtype=dtype,
                kid=requested_kid,
                split_k=requested_split_k,
            )
            if plan is not None:
                return _launch_batched_gemm_bf16_opus(
                    XQ,
                    WQ,
                    bias,
                    dtype,
                    kid=plan.resolved_kid,
                    split_k=requested_split_k,
                )
        logger.warning(
            "discarding invalid tuned OPUS BF16 BMM row for "
            f"shape ({B}, {M}, {N}, {K}); falling back to CK"
        )
    elif config is not None and config["libtype"] != "ck":
        logger.warning(
            f"unsupported BF16 BMM tuned backend {config['libtype']!r}; "
            "falling back to CK"
        )
    return batched_gemm_bf16_CK(
        XQ,
        WQ,
        bias=bias,
        dtype=dtype,
        splitK=splitK,
    )


@compile_ops(
    "module_batched_gemm_bf16_tune",
    fc_name="batched_gemm_bf16_tune",
    gen_fake=gen_batched_gemm_bf16_tune_fake_tensor,
)
def batched_gemm_bf16_tune(
    XQ: Tensor, WQ: Tensor, out: Tensor, kernelId: int, splitK: int = 0
) -> Tensor: ...
