# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import functools
import os
from pathlib import Path

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
from ..jit.utils.torch_guard import torch_compile_guard
from ..utility import dtypes
from .gemm_op_common import get_padded_m


def gen_batched_gemm_a8w8_fake_tensors(
    XQ: Tensor,
    WQ: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    out: Tensor,
    bias: Tensor | None = None,
    splitK: int = 0,
) -> Tensor:
    return out


@compile_ops(
    "module_batched_gemm_a8w8",
    fc_name="batched_gemm_a8w8",
    gen_fake=gen_batched_gemm_a8w8_fake_tensors,
)
def batched_gemm_a8w8(
    XQ: Tensor,
    WQ: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    out: Tensor,
    bias: Tensor | None = None,
    splitK: int = 0,
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


@functools.lru_cache(maxsize=1024)
def get_CKBatchedGEMM_config(
    B: int,
    M: int,
    N: int,
    K: int,
):
    if not hasattr(get_CKBatchedGEMM_config, "ck_batched_gemm_dict"):
        print(
            "Loading CKBatchedGEMM config from:",
            AITER_CONFIGS.AITER_CONFIG_A8W8_BATCHED_GEMM_FILE,
        )
        ck_batched_gemm_dict = pd.read_csv(
            AITER_CONFIGS.AITER_CONFIG_A8W8_BATCHED_GEMM_FILE
        ).drop_duplicates()
        # Use (gfx, cu_num, B, M, N, K) key when the CSV has a gfx column (new schema).
        # Fall back to (cu_num, B, M, N, K) for old CSVs that pre-date the gfx column.
        if "gfx" in ck_batched_gemm_dict.columns:
            get_CKBatchedGEMM_config.ck_batched_gemm_dict = (
                ck_batched_gemm_dict.set_index(
                    ["gfx", "cu_num", "B", "M", "N", "K"]
                ).to_dict("index")
            )
            get_CKBatchedGEMM_config.has_gfx = True
        else:
            logger.warning(
                f"{AITER_CONFIGS.AITER_CONFIG_A8W8_BATCHED_GEMM_FILE} has no 'gfx' column — "
                "falling back to cu_num-only key. Re-run the tuner or migrate the CSV."
            )
            get_CKBatchedGEMM_config.ck_batched_gemm_dict = (
                ck_batched_gemm_dict.set_index(["cu_num", "B", "M", "N", "K"]).to_dict(
                    "index"
                )
            )
            get_CKBatchedGEMM_config.has_gfx = False
    gfx = get_gfx()
    cu_num = get_cu_num()
    key = (
        (gfx, cu_num, B, M, N, K)
        if get_CKBatchedGEMM_config.has_gfx
        else (cu_num, B, M, N, K)
    )
    config = get_CKBatchedGEMM_config.ck_batched_gemm_dict.get(key, None)
    if config is not None:
        if AITER_LOG_TUNED_CONFIG:
            logger.info(
                f"shape is B:{B}, M:{M}, N:{N}, K:{K}, is tuned on cu_num = {cu_num} in {AITER_CONFIGS.AITER_CONFIG_A8W8_BATCHED_GEMM_FILE}, kernel name is {config['kernelName']}, splitK is {config['splitK']}!"
            )
        mnk = config["kernelName"].split("_")[3].split("x")[1:]
        config["tile_m"] = int(mnk[0])
        config["tile_n"] = int(mnk[1])
        config["tile_k"] = int(mnk[2])
    else:
        logger.info(
            f"shape is B:{B}, M:{M}, N:{N}, K:{K}, not found tuned config in CKGEMM, will use default config!"
        )
    return config


def batched_gemm_a8w8_CK(
    XQ: Tensor,
    WQ: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    bias: Tensor | None = None,
    dtype=dtypes.bf16,
    splitK: int | None = None,
):
    assert dtype in [
        dtypes.bf16,
        dtypes.fp16,
    ], f"Output {dtype=} is currently not supported in batched_gemm_a8w8"

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
    return batched_gemm_a8w8(XQ, WQ, x_scale, w_scale, Y, bias, splitK)


# ---------------------------------------------------------------------------
# gfx950 MXFP8 BMM tuned caller. The final global kid is passed verbatim to the
# one OPUS public entry; the private family launcher performs no selection.

_MXSCALE_BMM_CONFIG_ENV = "AITER_CONFIG_BATCHED_GEMM_A8W8_BLOCKSCALE_MXSCALE"
_MXSCALE_BMM_CONFIG_STEM = "batched_gemm_a8w8_blockscale_mxscale_tuned"
_MXSCALE_BMM_KID_OFFSET = 8000
_TUNED_PERF_COLUMNS = ("us", "tflops", "bw", "errRatio")


def _mxscale_bmm_config_paths() -> tuple[Path, ...]:
    configured = os.getenv(_MXSCALE_BMM_CONFIG_ENV)
    if configured:
        return tuple(
            Path(token).expanduser()
            for token in configured.split(os.pathsep)
            if token.strip()
        )

    config_dir = Path(__file__).resolve().parents[1] / "configs"
    candidates = [config_dir / f"{_MXSCALE_BMM_CONFIG_STEM}.csv"]
    candidates.extend(
        sorted(
            (config_dir / "model_configs").glob(
                f"*_{_MXSCALE_BMM_CONFIG_STEM}.csv"
            )
        )
    )
    return tuple(path for path in candidates if path.is_file())


@functools.cache
def _load_mxscale_bmm_tuned(libtype: str | None = None) -> dict:
    paths = _mxscale_bmm_config_paths()
    if not paths:
        logger.warning("no MXFP8 BMM tuned CSV was found")
        return {}

    frames = [pd.read_csv(path) for path in paths]
    df = pd.concat(frames, ignore_index=True).drop_duplicates()
    required = {"gfx", "b", "m", "n", "k", "kernelId", "splitK"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            f"MXFP8 BMM tuned CSV is missing columns {sorted(missing)}"
        )
    if libtype is not None and "libtype" in df.columns:
        df = df[df["libtype"] == libtype]
    if not df.empty and int(df["kernelId"].min()) < _MXSCALE_BMM_KID_OFFSET:
        raise ValueError(
            "MXFP8 BMM tuned CSV must contain global OPUS kids in the "
            "8000-8653 range"
        )
    shape_keys = ["gfx", "b", "m", "n", "k"]
    duplicate_shapes = df.duplicated(subset=shape_keys, keep=False)
    if duplicate_shapes.any():
        rows = df.loc[duplicate_shapes, shape_keys].drop_duplicates().to_dict("records")
        raise RuntimeError(
            f"duplicate shapes across MXFP8 BMM tuned CSV files: {rows}"
        )
    return df.set_index(["gfx", "b", "m", "n", "k"]).to_dict("index")


@functools.lru_cache(maxsize=1024)
def lookup_mxscale_bmm_config(
    b: int, m: int, n: int, k: int, *, libtype: str | None = None
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
        cfg = {key: value for key, value in row.items() if key not in _TUNED_PERF_COLUMNS}
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


@functools.cache
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

    if divisible(n, 256) and divisible(k, 128) and (
        m >= 2048 or (m >= 1024 and g >= 8)
    ):
        return 8158 if 4096 <= k <= 8192 else 8150
    if m < 64:
        return 8640 if divisible(n, 64) and divisible(k, 256) else 8653
    if m <= 256 and k <= 1024 and divisible(n, 32) and divisible(k, 256):
        return 8320
    if divisible(n, 64) and divisible(k, 128):
        return 8653
    return 8000


def _batched_gemm_a8w8_mxscale_impl(
    x: Tensor,
    wo_a: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    dtype: torch.dtype = dtypes.bf16,
) -> Tensor:
    from .opus import opus_gemm

    m, g, k = map(int, x.shape)
    n = int(wo_a.shape[1])
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

    Y = torch.empty((m, g, n), dtype=dtype, device=x.device)
    return opus_gemm(
        x,
        wo_a,
        Y,
        kid=kid,
        layout="mxscale_bmm",
        x_scale=x_scale,
        w_scale=w_scale,
        split_k=split_k,
    )


def _batched_gemm_a8w8_mxscale_fake(
    x: Tensor,
    wo_a: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    dtype: torch.dtype = dtypes.bf16,
) -> Tensor:
    return torch.empty(
        (x.shape[0], x.shape[1], wo_a.shape[1]),
        dtype=dtype,
        device=x.device,
    )


@torch_compile_guard(mutates_args=[], gen_fake=_batched_gemm_a8w8_mxscale_fake)
def batched_gemm_a8w8_mxscale(
    x: Tensor,
    wo_a: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    dtype: torch.dtype = dtypes.bf16,
) -> Tensor:
    """Run gfx950 E8M0 MXFP8 BMM and return token-major ``[M,G,N]``."""
    return _batched_gemm_a8w8_mxscale_impl(
        x, wo_a, x_scale, w_scale, dtype=dtype
    )


def gen_batched_gemm_a8w8_tune_fake_tensors(
    XQ: Tensor,
    WQ: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    out: Tensor,
    kernelId: int,
    splitK: int = 0,
) -> Tensor:
    return out


@compile_ops(
    "module_batched_gemm_a8w8_tune",
    fc_name="batched_gemm_a8w8_tune",
    gen_fake=gen_batched_gemm_a8w8_tune_fake_tensors,
)
def batched_gemm_a8w8_tune(
    XQ: Tensor,
    WQ: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    out: Tensor,
    kernelId: int,
    splitK: int = 0,
) -> Tensor: ...
