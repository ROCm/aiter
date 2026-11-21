# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
"""OPUS gfx950 MXFP4 x MXFP4 GEMM (family ``a4w4``), ``layout="bpreshuffle"`` kids.

``layout`` declares the operand *content* (as in every OPUS family):
``"bpreshuffle"`` = the inputs of aiter ``gemm_a4w4(..., bpreshuffle=True)``:

* ``XQ``: [M, K/2] (or [batch, M, K/2]) fp4x2, row-major.
* ``WQ``: [N, K/2] (or [batch, N, K/2]) fp4x2 in ``shuffle_weight(w, layout=(16, 16))``.
* ``x_scale`` / ``w_scale``: 1x32 e8m0 in the ``e8m0_shuffle`` layout, 2D
  [batch * rows, cols] or 3D [batch, rows, cols], rows >= M / N padded to 32
  (a multiple of 32), cols = K/32 padded to 8.

Any M; N % 16 == 0; K % 256 == 0. Output bf16 [M, N] (or [batch, M, N]).

Local kids (``a4w4/gfx950/<CU>``, bpreshuffle): 0-5 persistent, 10-15 one tile
per block, 21-25 split-K, 30 persistent without M/N boundary handling (the
pre-compiled 2x2 pipeline, ``gen_co/gfx950/a4w4_bpreshuffle``), 410-416 the
JIT-built 1x4 small-tile pipeline (split-K through ``split_k`` and a Torch
workspace). Launched through ``opus_gemm`` / ``opus_bmm`` (``family="a4w4"``,
``layout="bpreshuffle"``); ``aiter.gemm_a4w4`` dispatches tuned-CSV rows whose
kernelName is the OPUS instance name.
"""

from functools import lru_cache

import torch
from torch import Tensor

from csrc.opus_gemm.opus_gemm_common import (
    A4W4_BPRESHUFFLE_CO_MODE,
    get_kernel_instance,
)

from ...jit.core import compile_ops
from ...utility import dtypes
from ._arch import _device_arch_and_cu

A4W4_FAMILY = "a4w4"
# Split-K workspace of the bpreshuffle .co kids (opus_mxfp4_bpreshuffle_sk_ws_bytes in
# opus_gemm_traits_a4w4_mxfp4_nt_bpreshuffle_gfx950.cuh): [4 KB tile counters][tiles][S][B_M * B_N] fp32.
_BPRESHUFFLE_SK_MAX_TILES = 1024
_BPRESHUFFLE_SK_COUNTER_NUMEL = _BPRESHUFFLE_SK_MAX_TILES
# Small-tile split-K workspace: [256 B handle][S][batch][padded_M][padded_N] fp32.
_BPRESHUFFLE_SMALL_WS_HEADER_NUMEL = 64


# ---- Low-level backend -------------------------------------------------------


def _gen_opus_gemm_a4w4_bpreshuffle_launch_fake_tensors(
    XQ: Tensor,
    WQ: Tensor,
    Y: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    workspace: Tensor | None,
    kid: int,
    split_k: int,
) -> Tensor:
    return Y


@compile_ops(
    "module_deepgemm_opus",
    fc_name="opus_gemm_a4w4_bpreshuffle_launch",
    gen_fake=_gen_opus_gemm_a4w4_bpreshuffle_launch_fake_tensors,
    develop=True,
)
def _opus_gemm_a4w4_bpreshuffle_launch_raw(
    XQ: Tensor,
    WQ: Tensor,
    Y: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    workspace: Tensor | None,
    kid: int,
    split_k: int,
) -> Tensor: ...


def _cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


@lru_cache(maxsize=4096)
def _a4w4_bpreshuffle_plan(
    arch: str, cu_num: int, kid: int, M: int, N: int, K: int, batch: int, split_k: int
) -> tuple[int, int]:
    """Return (ABI split_k, fp32 workspace elements; 0 = none) for one launch.

    Split-K .co kids with split_k = 0 pick S: the largest of {8, 4, 2} with
    tiles * S <= CUs and K a multiple of S * 256 (whole K tiles per segment).
    """
    inst = get_kernel_instance(arch, A4W4_FAMILY, kid, torch.bfloat16, cu_num=cu_num)
    if inst is None:
        raise ValueError(
            f"no registered OPUS kernel for (arch={arch!r}, "
            f"family={A4W4_FAMILY!r}, kid={kid}, cu_num={cu_num})"
        )
    if split_k < 0:
        raise ValueError(f"OPUS a4w4 split_k must be >= 0, got {split_k}")
    tiles = _cdiv(M, inst.B_M) * _cdiv(N, inst.B_N) * batch
    if inst.kernel_tag == "a4w4_mxfp4_nt_bpreshuffle_small":
        S = max(1, split_k)
        if K // inst.B_K < S:
            raise ValueError(
                f"OPUS a4w4 kid {kid}: K / {inst.B_K} must be >= split_k={S}"
            )
        if S == 1:
            return 1, 0
        padded = _cdiv(M, inst.B_M) * inst.B_M * _cdiv(N, inst.B_N) * inst.B_N
        return S, _BPRESHUFFLE_SMALL_WS_HEADER_NUMEL + S * batch * padded
    if inst.co_variant == "nob" and (M % inst.B_M or N % inst.B_N):
        raise ValueError(
            f"OPUS a4w4 kid {kid} needs M % {inst.B_M} == 0 and N % {inst.B_N} == 0"
        )
    if A4W4_BPRESHUFFLE_CO_MODE[inst.co_variant] != 2:
        if split_k > 1:
            raise ValueError(
                f"OPUS a4w4 kid {kid} does not split K (split_k={split_k})"
            )
        return 1, 0
    S = split_k
    if S == 0:
        S = next(
            (
                s
                for s in (8, 4, 2)
                if tiles <= _BPRESHUFFLE_SK_MAX_TILES
                and tiles * s <= cu_num
                and K % (s * inst.B_K) == 0
            ),
            1,
        )
    if K % (S * inst.B_K):
        raise ValueError(
            f"OPUS a4w4 kid {kid}: K={K} is not a multiple of "
            f"split_k * {inst.B_K} (split_k={S})"
        )
    if S == 1:
        return 1, 0
    if tiles > _BPRESHUFFLE_SK_MAX_TILES:
        raise ValueError(
            f"OPUS a4w4 kid {kid}: split-K supports at most "
            f"{_BPRESHUFFLE_SK_MAX_TILES} tiles"
        )
    return S, _BPRESHUFFLE_SK_COUNTER_NUMEL + tiles * S * inst.B_M * inst.B_N


# ---- Execution adapters (opus_gemm / opus_bmm) ---------------------------------


def _launch_a4w4_bpreshuffle_bmm(
    XQ: Tensor,
    WQ: Tensor,
    Y: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    *,
    kid: int,
    split_k: int = 0,
    workspace: Tensor | None = None,
    route_arch: str | None = None,
) -> Tensor:
    """Launch batch-first ``[B,M,K/2] x [B,N,K/2] -> bf16 [B,M,N]``.

    The checked C++ entry owns the tensor contract (dtypes, shapes, scale rows,
    contiguity); Python only plans split-K and sizes the fp32 workspace.
    """
    batch, M, K2 = map(int, XQ.shape)
    device_arch, cu_num = _device_arch_and_cu(XQ.device)
    abi_split_k, ws_numel = _a4w4_bpreshuffle_plan(
        route_arch or device_arch,
        cu_num,
        int(kid),
        M,
        int(WQ.shape[1]),
        2 * K2,
        batch,
        int(split_k),
    )
    if ws_numel == 0:
        workspace = None
    elif workspace is None:
        workspace = torch.empty(ws_numel, dtype=torch.float32, device=XQ.device)
    elif (
        workspace.dtype != torch.float32
        or not workspace.is_contiguous()
        or workspace.numel() < ws_numel
    ):
        raise ValueError(
            f"OPUS a4w4 kid {kid} needs a contiguous fp32 workspace of "
            f">= {ws_numel} elements, got {workspace.dtype} {tuple(workspace.shape)}"
        )
    with torch.cuda.device(XQ.device):
        _opus_gemm_a4w4_bpreshuffle_launch_raw(
            XQ, WQ, Y, x_scale, w_scale, workspace, int(kid), abi_split_k
        )
    return Y


def _launch_a4w4_bpreshuffle_gemm(
    XQ: Tensor,
    WQ: Tensor,
    Y: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    **kwargs,
) -> Tensor:
    """Launch logical 2D ``[M,K/2] x [N,K/2] -> bf16 [M,N]``."""
    _launch_a4w4_bpreshuffle_bmm(
        XQ.unsqueeze(0), WQ.unsqueeze(0), Y.unsqueeze(0), x_scale, w_scale, **kwargs
    )
    return Y


def _validate_a4w4_bpreshuffle_public_contract(
    *,
    kid: int,
    input_dtype: torch.dtype,
    weight_dtype: torch.dtype,
    output_dtype: torch.dtype,
    layout: str,
    has_x_scale: bool,
    has_w_scale: bool,
    has_bias: bool,
) -> None:
    """Immutable per-kid contract checked (and cached) by ``dispatch._resolve_contract``."""
    if input_dtype != dtypes.fp4x2 or weight_dtype != dtypes.fp4x2:
        raise ValueError(
            f"OPUS a4w4 kid {kid} requires fp4x2 XQ/WQ; got "
            f"{input_dtype}/{weight_dtype}"
        )
    if output_dtype != torch.bfloat16:
        raise ValueError(f"OPUS a4w4 kid {kid} requires bf16 Y; got {output_dtype}")
    if layout != "bpreshuffle":
        raise ValueError(
            f"OPUS kid {kid} is an a4w4 bpreshuffle kid and requires "
            f"layout='bpreshuffle'; got {layout!r}"
        )
    if not (has_x_scale and has_w_scale):
        raise ValueError("OPUS a4w4 requires x_scale and w_scale")
    if has_bias:
        raise ValueError("OPUS family a4w4 does not support bias")
