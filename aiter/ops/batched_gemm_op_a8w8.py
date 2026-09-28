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
from .gemm_op_common import mxscale_w_scale_block
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


def _mxscale_bmm_tuned_path(bpreshuffle: bool) -> str:
    """Tuned table for one weight layout; preshuffled rows live in their own CSV."""
    return (
        AITER_CONFIGS.AITER_CONFIG_BATCHED_GEMM_A8W8_BLOCKSCALE_MXSCALE_BPRESHUFFLE_FILE
        if bpreshuffle
        else AITER_CONFIGS.AITER_CONFIG_BATCHED_GEMM_A8W8_BLOCKSCALE_MXSCALE_FILE
    )


@functools.cache
def _get_mxscale_bmm_launchers():
    """Resolve the checked split-1 launcher and workspace planner once."""
    from .opus import opus_bmm
    from .opus.gemm_op_a8w8 import _opus_gemm_a8w8_mxscale_bmm_launch_raw

    return _opus_gemm_a8w8_mxscale_bmm_launch_raw, opus_bmm


# One implementation, in opus.policy: two lookups reading one table is exactly
# the drift that picks a kid for the wrong B layout. These names stay for the
# callers of this module.
def _load_mxscale_bmm_tuned(
    libtype: str | None = None, bpreshuffle: bool = False
) -> dict:
    """{(gfx,b,m,n,k,w_scale_block): row} from the mxscale BMM tuned CSV."""
    from .opus.policy import _load_mxscale_bmm_tuned as load

    return load(libtype, bpreshuffle)


def lookup_mxscale_bmm_config(
    b: int,
    m: int,
    n: int,
    k: int,
    *,
    w_scale_block: str = "128x128",
    libtype: str | None = None,
    bpreshuffle: bool = False,
):
    """Exact tuned row for this shape and weight-scale block, else one at a
    padded M; None when no level hits. The row is shared: treat it as read-only."""
    from .opus.policy import lookup_mxscale_bmm_config as lookup

    return lookup(
        b, m, n, k, w_scale_block=w_scale_block, libtype=libtype, bpreshuffle=bpreshuffle
    )


@functools.lru_cache(maxsize=64)
def _w_scale_block_of(w_scale_shape: tuple, n: int, k: int) -> str:
    return mxscale_w_scale_block(w_scale_shape, n, k)


@functools.lru_cache(maxsize=1024)
def _get_mxscale_bmm_launch_plan(
    g: int,
    m: int,
    n: int,
    k: int,
    w_scale_block: str = "128x128",
) -> tuple[int, int]:
    return _resolve_a8w8_mxscale_bmm_plan(g, m, n, k, w_scale_block=w_scale_block)


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
    w_scale_block = _w_scale_block_of(tuple(w_scale.shape), n, k)
    kid, split_k = _get_mxscale_bmm_launch_plan(g, m, n, k, w_scale_block)

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
    """Run gfx950 E8M0 MXFP8 BMM and return token-major ``[M,G,N]``.

    The w_scale block (``[G, N/block, K/block]``, 128x128 or 32x32) picks the
    tuned row and the kernel, since a kernel built for the other block reads the
    scales at the wrong stride.
    """
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

    The preshuffle table holds both backends' kernels -- each row names the
    faster of opus and flydsl for its shape and w_scale block -- so the row's
    libtype picks the backend. A shape without a row runs flydsl's heuristic.
    """
    m, g, k = int(x.shape[0]), int(x.shape[1]), int(x.shape[2])
    n = int(wo_a.shape[1])
    w_scale_block = _w_scale_block_of(tuple(w_scale.shape), n, k)

    cfg = lookup_mxscale_bmm_config(
        g, m, n, k, w_scale_block=w_scale_block, bpreshuffle=True
    )
    libtype = cfg["libtype"] if cfg is not None else "flydsl"

    if libtype == "opus":
        from .opus.gemm_op_a8w8 import bmm_a8w8_mxscale_opus
        from .opus.policy import mxscale_bmm_group_of_block

        # Whether the tuned kernel can run this M, whether it reads B in the
        # declared layout, and what to do when it cannot, is the backend's job.
        return bmm_a8w8_mxscale_opus(
            x,
            wo_a,
            x_scale,
            w_scale,
            None,
            dtype=dtype,
            kernelId=int(cfg["kernelId"]),
            splitK=int(cfg["splitK"]),
            b_preshuffled=True,
            group_size=mxscale_bmm_group_of_block(w_scale_block),
        )

    if libtype == "flydsl":
        from .flydsl.batched_gemm_a8w8 import (
            bmm_a8w8_mxfp8_supported,
            run_bmm_a8w8_mxfp8,
        )

        if not bmm_a8w8_mxfp8_supported(w_scale_block):
            raise NotImplementedError(
                f"no preshuffled mxscale BMM kernel reads a {w_scale_block} w_scale "
                f"on {get_gfx()}"
            )
        return run_bmm_a8w8_mxfp8(
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
    """fp8 e8m0 mxscale batched GEMM with a preshuffled weight.

    Same math as ``batched_gemm_a8w8_mxscale``, but ``wo_a`` was baked into the
    (16, 16) MFMA-fragment layout at load time. That is why it is a separate
    entry rather than a flag: the shuffled weight has the same shape, dtype and
    strides as the row-major one, so a mismatched kernel reads the right bytes in
    the wrong order and returns a plausible wrong answer instead of failing. The
    entry also picks the tuned CSV, so the two layouts cannot pull each other's
    rows.

    * ``x``       : [M, G, K] fp8 activation, token-major and contiguous.
    * ``wo_a``    : [G, N, K] fp8 weight, ``shuffle_weight(w, layout=(16, 16))``.
    * ``x_scale`` : [M, G, K/block] uint8 e8m0, row-major, i.e.
                    ``inverse_rope_group_quant(..., scale_layout="row")``; every
                    backend reads it that way.
    * ``w_scale`` : [G, N/block, K/block] uint8 e8m0.

    The arch and the w_scale block select the kernels:

    * gfx950, 32x32 or 128x128: the tuned row's libtype, opus or flydsl,
      whichever was faster at that shape; flydsl's heuristic off the table.
      DeepSeek-V4.1's original wo_a weight (32x32) and V4's (128x128) both run
      as is.
    * gfx1250, 128x128: flydsl.

    Returns a fresh token-major [M, G, N]. A caller that must write into its own
    buffer calls a backend directly (both keep ``out=``): the arch's
    ``run_bmm_a8w8_mxfp8_*`` or ``aiter.ops.opus.gemm_op_a8w8.bmm_a8w8_mxscale_opus``.
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
