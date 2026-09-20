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
from .gemm_op_common import get_padded_m
from .opus.policy import (
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
                f"{AITER_CONFIGS.AITER_CONFIG_A8W8_BATCHED_GEMM_FILE} has no 'gfx' column; "
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
_TUNED_PERF_COLUMNS = ("us", "tflops", "bw", "errRatio")




@functools.cache
def _get_mxscale_bmm_launchers():
    """Resolve the checked split-1 launcher and workspace planner once."""
    from .opus import opus_bmm
    from .opus.gemm_op_a8w8 import _opus_gemm_a8w8_mxscale_bmm_launch_raw

    return _opus_gemm_a8w8_mxscale_bmm_launch_raw, opus_bmm






@functools.lru_cache(maxsize=1024)
def _get_mxscale_bmm_launch_plan(
    g: int,
    m: int,
    n: int,
    k: int,
) -> tuple[int, int]:
    return _resolve_a8w8_mxscale_bmm_plan(g, m, n, k)


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
    raw_launch, opus_bmm = _get_mxscale_bmm_launchers()
    kid, split_k = _get_mxscale_bmm_launch_plan(g, m, n, k)

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
    return _batched_gemm_a8w8_mxscale_impl(x, wo_a, x_scale, w_scale, dtype=dtype)


# Same family, preshuffled weight.
def _batched_gemm_a8w8_mxscale_bpreshuffle_impl(
    x: Tensor,
    wo_a: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    dtype: torch.dtype = dtypes.bf16,
) -> Tensor:
    """Eager tuned-CSV lookup + libtype dispatch; returns token-major [M, G, N].

    Two backends carry a preshuffled B and they want different activation scale
    layouts, so the tuned row's libtype decides both which kernel runs and which
    ``x_scale`` the caller must have produced -- see the public entry.
    """
    # One implementation, in policy.py: this module used to carry a second copy
    # that differed only by the bpreshuffle switch, and two lookups reading two
    # tables is exactly the drift that picks a kid for the wrong B layout.
    from .opus.policy import lookup_mxscale_bmm_config

    m, g, k = int(x.shape[0]), int(x.shape[1]), int(x.shape[2])
    n = int(wo_a.shape[1])

    cfg = lookup_mxscale_bmm_config(g, m, n, k, bpreshuffle=True)
    if cfg is not None:
        libtype = cfg["libtype"]
    else:
        # Untuned shapes: opus carries a shape heuristic, the gfx1250 flydsl path
        # does not, so the arch picks the fallback.
        libtype = "flydsl" if get_gfx() == "gfx1250" else _MXSCALE_BMM_DEFAULT_LIBTYPE

    if libtype == "opus":
        from .opus.gemm_op_a8w8 import bmm_a8w8_mxscale_opus

        # Whether the tuned kernel can run this M, whether it reads B in the
        # declared layout, and what to do when it cannot, is the backend's job.
        return bmm_a8w8_mxscale_opus(
            x,
            wo_a,
            x_scale,
            w_scale,
            None,
            dtype=dtype,
            kernelId=int(cfg["kernelId"]) if cfg is not None else None,
            splitK=int(cfg["splitK"]) if cfg is not None else None,
            b_preshuffled=True,
        )

    if libtype == "flydsl":
        from .flydsl.batched_gemm_a8w8_gfx1250 import run_bmm_a8w8_mxfp8_128_gfx1250

        return run_bmm_a8w8_mxfp8_128_gfx1250(
            x,
            wo_a,
            x_scale,
            w_scale,
            torch.empty((m, g, n), dtype=dtype, device=x.device),
            kernel_name=str(cfg["kernelName"]) if cfg is not None else None,
        )

    raise NotImplementedError(
        f"tuned row for B:{g}, M:{m}, N:{n}, K:{k} wants libtype {libtype!r}, "
        "which has no preshuffled-B batched GEMM; row-major rows are served by "
        "batched_gemm_a8w8_mxscale"
    )


@torch_compile_guard(mutates_args=[], gen_fake=_batched_gemm_a8w8_mxscale_fake)
def batched_gemm_a8w8_mxscale_bpreshuffle(
    x: Tensor,
    wo_a: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    dtype: torch.dtype = dtypes.bf16,
) -> Tensor:
    """fp8 e8m0 mxscale batched GEMM whose ``wo_a`` is already preshuffled.

    Same math as ``batched_gemm_a8w8_mxscale``, but ``wo_a`` was baked into the
    (16, 16) MFMA-fragment layout (``aiter.ops.shuffle.shuffle_weight``) at load
    time, which a serving stack does offline and cannot express any other way.
    That is why it is a separate entry rather than a flag: the shuffled weight
    has the same shape, dtype and strides as the row-major one, so a mismatched
    kernel reads the right bytes in the wrong order and returns a plausible wrong
    answer instead of failing. The entry also picks the tuned CSV, so the two
    layouts cannot pull each other's rows.

    * ``x``       : [M, G, K] fp8 activation, token-major and contiguous.
    * ``wo_a``    : [G, N, K] fp8 weight, preshuffled as above.
    * ``w_scale`` : [G, N/128, K/128] uint8 e8m0.
    * ``x_scale`` : [M, G, K/128] uint8 e8m0, **in the layout the resolved
                    backend wants** -- and the two differ:

      - ``flydsl`` (gfx1250): row-major, i.e. ``inverse_rope_group_quant(...,
        scale_layout="row")``.
      - ``opus`` (gfx950): the MFMA-tile layout, i.e. ``scale_layout="mfma_tile"``
        (byte-identical to ``shuffle_scale_a(xs, k, sub=16)``). Its kernels fold
        the row offset into the tile index, so a per-token scale slab is passed
        with ``stride(0) == 0`` via ``.expand()``.

      Getting this wrong is the same silent-wrong-answer failure as the weight
      layout, so the caller has to know which backend its shape resolves to.

    Returns a fresh token-major [M, G, N]. A caller that must write into its own
    buffer calls the backend directly (both keep ``out=``):
    ``run_bmm_a8w8_mxfp8_128_gfx1250`` or
    ``aiter.ops.opus.gemm_op_a8w8.bmm_a8w8_mxscale_opus``.
    """
    return _batched_gemm_a8w8_mxscale_bpreshuffle_impl(
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
