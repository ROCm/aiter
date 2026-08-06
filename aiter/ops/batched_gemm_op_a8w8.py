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
                f"{AITER_CONFIGS.AITER_CONFIG_A8W8_BATCHED_GEMM_FILE} has no 'gfx' column -- "
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
# Shared tuned-CSV lookup for the mxscale batched GEMM.
#
# Shaped like tuned_gemm.py's multi-backend lookup: this layer locates the row
# and never interprets the kernel identifier, since that differs per backend
# (opus names kernels with an integer kernelId, flydsl with a kernelName). The
# row comes back whole, libtype included, so a caller can dispatch on it;
# libtype also filters up front for CSVs that carry one row per (shape, backend)
# rather than a single cross-backend winner per shape.

# Tuner bookkeeping rather than selection inputs, so the lookup log drops them
# and stays readable.
_TUNED_PERF_COLUMNS = ("us", "tflops", "bw", "errRatio")


@functools.cache
def _load_mxscale_bmm_tuned(libtype: str | None = None) -> dict:
    """{(gfx,b,m,n,k): row} from the mxscale BMM tuned CSV; {} if it is missing."""
    path = AITER_CONFIGS.AITER_CONFIG_BATCHED_GEMM_A8W8_BLOCKSCALE_MXSCALE_FILE
    try:
        df = pd.read_csv(path).drop_duplicates()
    except FileNotFoundError:
        logger.warning("mxscale BMM tuned CSV not found at %s", path)
        return {}
    if libtype is not None and "libtype" in df.columns:
        df = df[df["libtype"] == libtype]
    return df.set_index(["gfx", "b", "m", "n", "k"]).to_dict("index")


@functools.lru_cache(maxsize=1024)
def lookup_mxscale_bmm_config(
    b: int, m: int, n: int, k: int, *, libtype: str | None = None
):
    """Exact tuned row for this shape, else one at a padded M.

    Same exact-then-two-granularities walk over the shared C++ getPaddedM that
    the CK / asm / a16w16 lookups use. A bucket table built from the CSV's own M
    values was the alternative and bought nothing: over every M up to the
    largest tuned one, both cover the same shapes and reach the same kernel on
    131070 of 131072 M, so this keeps the one rounding rule the repo already has.

    Cached per shape like get_CKGEMM_config, and for the same reason: getPaddedM
    is a ctypes hop into C++ at ~10us, and the padded levels run on every call
    whose M is not itself a tuned row. DPA+MTP decode is exactly that case (M is
    the ragged token count a rank happened to get), and paying it once per layer
    per step cost ~1% end-to-end before this. The row is shared, so callers must
    treat it as read-only.

    Returns the row, or None when no level hits. The log prints the row whole
    instead of named fields, so a backend gets its own kernel identifier
    reported without this layer knowing which column holds it.
    """
    gfx = get_gfx()
    path = AITER_CONFIGS.AITER_CONFIG_BATCHED_GEMM_A8W8_BLOCKSCALE_MXSCALE_FILE
    tuned = _load_mxscale_bmm_tuned(libtype)

    row, padded_m = None, m
    for gl in (None, 0, 1):
        padded_m = m if gl is None else get_padded_m(m, n, k, gl)
        row = tuned.get((gfx, b, padded_m, n, k))
        if row is not None:
            break

    if row is None:
        logger.info(
            f"shape is B:{b}, M:{m}, N:{n}, K:{k}, not found tuned/padded config "
            f"in {path}, the caller will fall back!"
        )
        return None

    if AITER_LOG_TUNED_CONFIG:
        cfg = {c: v for c, v in row.items() if c not in _TUNED_PERF_COLUMNS}
        if padded_m == m:
            logger.info(
                f"shape is B:{b}, M:{m}, N:{n}, K:{k}, is tuned on gfx = {gfx} "
                f"in {path}, config is {cfg}!"
            )
        else:
            logger.info(
                f"shape is B:{b}, M:{m}, N:{n}, K:{k}, exact miss on gfx = {gfx}; "
                f"using padded_M: {padded_m} config {cfg} from {path}!"
            )
    return row


# ---------------------------------------------------------------------------
# fp8 e8m0 mxscale (block-scale) batched GEMM -- public entry for the family.
#
# This file is the per-family (a8w8 batched) public surface, not a CK-only
# file: like aiter/ops/gemm_op_a8w8.py hosts gemm_a8w8 (CK rowwise) +
# gemm_a8w8_blockscale (ck/cktile/triton/asm) side by side and lazy-imports
# backend impls, we host the mxscale batched entry here too. The concrete
# kernels stay in their backend dirs (opus -> aiter.ops.opus.bmm_op).
#
# Dispatch follows tuned_gemm.mm: look the shape up once here, then let the
# winning row's libtype pick the backend, which is why the lookup runs
# unfiltered -- the tuner writes one winning row per shape and its libtype says
# who won. A second backend then only has to add rows and a branch below; it
# does not repeat the lookup.

# Untuned shapes go to opus: it is the backend carrying a shape heuristic for
# rows the CSV does not have.
_MXSCALE_BMM_DEFAULT_LIBTYPE = "opus"


def batched_gemm_a8w8_mxscale(
    x: Tensor,
    wo_a: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    out: Tensor | None = None,
    dtype: torch.dtype = dtypes.bf16,
) -> Tensor:
    """fp8 e8m0 mxscale (128x128 block-scale) batched GEMM.

    mmajor DSV4 wo_a layout (matches the opus kernels + op test):

    * ``x``       : [M, G, K] fp8 activation (per-token e8m0; transposed view
                    of batch-major [G, M, K]).
    * ``wo_a``    : [G, N, K] fp8 weight (batch-major).
    * ``x_scale`` : [M, G, K/128] uint8 e8m0 activation scale.
    * ``w_scale`` : [G, N/128, K/128] uint8 e8m0 weight scale.
    * ``out``     : optional preallocated [M, G, N] output (fp32 or bf16).

    Note this is *microscaling* (e8m0) block scale -- distinct from
    ``gemm_a8w8_blockscale`` which uses fp32 block scale. Scale type is baked
    into the name so a future fp32-block batched variant stays separate.

    The shape is looked up in the tuned CSV here and the winning row's libtype
    picks the backend. No kernel override lives on this entry: how a kernel is
    named is backend-specific, so pin one at the backend (aiter.ops.opus.bmm_op).
    """
    from .opus.bmm_op import bmm_a8w8_mxscale_opus

    m, g, k = int(x.shape[0]), int(x.shape[1]), int(x.shape[2])
    n = int(wo_a.shape[1])

    cfg = lookup_mxscale_bmm_config(g, m, n, k)
    libtype = cfg["libtype"] if cfg is not None else _MXSCALE_BMM_DEFAULT_LIBTYPE
    if libtype != "opus":
        raise NotImplementedError(
            f"tuned row for B:{g}, M:{m}, N:{n}, K:{k} wants libtype "
            f"{libtype!r}, which has no batched mxscale backend here yet"
        )

    # Reading opus columns is this branch's job; whether that kernel can run
    # this M, and what to do when it cannot, is the backend's.
    return bmm_a8w8_mxscale_opus(
        x,
        wo_a,
        x_scale,
        w_scale,
        out,
        dtype=dtype,
        kernelId=int(cfg["kernelId"]) if cfg is not None else None,
        splitK=int(cfg["splitK"]) if cfg is not None else None,
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
