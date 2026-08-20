# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import functools

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
from .opus.policy import (
    lookup_mxscale_bmm_config,
    resolve_a8w8_mxscale_bmm_plan as _resolve_a8w8_mxscale_bmm_plan,
)


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
# gfx950 MXFP8 BMM high-level caller. Tuned-row and heuristic selection live
# in ``opus.policy``; this module owns only the hot launch cache,
# output allocation and split-one/workspace execution choice.


@functools.cache
def _get_mxscale_bmm_launchers():
    """Resolve the checked split-1 launcher and workspace planner once."""
    from .opus import opus_bmm
    from .opus.gemm_op_a8w8 import _opus_gemm_a8w8_mxscale_bmm_launch_raw

    return _opus_gemm_a8w8_mxscale_bmm_launch_raw, opus_bmm

_MXSCALE_BMM_LAUNCH_PLANS: dict[tuple[int, int, int, int], tuple[object, object, int, int]] = {}


def _batched_gemm_a8w8_mxscale_impl(
    x: Tensor,
    wo_a: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    dtype: torch.dtype = dtypes.bf16,
) -> Tensor:
    # This body executes behind the public custom-op boundary, so real eager
    # tensors carry concrete integer dimensions here.  Avoid four redundant
    # Python int() conversions on every short BMM launch.
    m, g, k = x.shape
    n = wo_a.shape[1]
    plan_key = (g, m, n, k)
    try:
        raw_launch, opus_bmm, kid, split_k = _MXSCALE_BMM_LAUNCH_PLANS[plan_key]
    except KeyError:
        raw_launch, opus_bmm = _get_mxscale_bmm_launchers()
        kid, split_k = _resolve_a8w8_mxscale_bmm_plan(g, m, n, k)
        _MXSCALE_BMM_LAUNCH_PLANS[plan_key] = (
            raw_launch,
            opus_bmm,
            kid,
            split_k,
        )

    Y = torch.empty((m, g, n), dtype=dtype, device=x.device)
    if split_k <= 1:
        # The shape resolver already returns a final canonical global kid.
        # Enter the checked C++ launcher directly for the common no-workspace
        # path instead of repeating the unified public routing contract.  The
        # C++ boundary still validates dtype, shape, device, stride, arch and
        # exact kid.  Workspace cases retain the unified Python planner below.
        raw_launch(
            x,
            wo_a,
            Y,
            x_scale,
            w_scale,
            None,
            kid,
            max(1, split_k),
        )
        return Y
    opus_bmm(
        x.transpose(0, 1),
        wo_a,
        Y.transpose(0, 1),
        kid=kid,
        layout="mxscale_bmm",
        x_scale=x_scale.transpose(0, 1),
        w_scale=w_scale,
        split_k=split_k,
    )
    return Y


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
