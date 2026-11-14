# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
"""OPUS gfx950 MXFP4 x MXFP4 NT GEMM (family ``a4w4``, 1x32 e8m0 scales).

Inputs are those of aiter ``gemm_a4w4(..., bpreshuffle=True)``:

* ``XQ``: [M, K/2] (or [batch, M, K/2]) fp4x2, row-major.
* ``WQ``: [N, K/2] (or [batch, N, K/2]) fp4x2 in ``shuffle_weight(w, layout=(16, 16))``.
* ``x_scale`` / ``w_scale``: e8m0 in the ``e8m0_shuffle`` layout, 2D
  [batch * rows, cols] or 3D [batch, rows, cols], rows >= M / N padded to 32
  (a multiple of 32), cols = K/32 padded to 8.

Any M; N % 16 == 0; K % 256 == 0. Output bf16 [M, N] (or [batch, M, N]).

Local kids (``a4w4/gfx950/<CU>``): 0-5 persistent, 10-15 one tile per block,
21-25 split-K, 30 persistent without M/N boundary handling (the pre-compiled
2x2 pipeline, ``gen_co/gfx950``), 410-416 the JIT-built 1x4 small-tile
pipeline. Launched through ``opus_gemm`` / ``opus_bmm`` (``family="a4w4"``,
``layout="bpreshuffle"``); ``aiter.gemm_a4w4`` dispatches tuned-CSV rows whose
kernelName is the OPUS instance name (kernelId = local kid).
"""

from functools import lru_cache

import torch
from torch import Tensor

from csrc.opus_gemm.opus_gemm_common import A4W4_CO_MODE, get_kernel_instance

from ...jit.core import compile_ops
from ...utility import dtypes

A4W4_FAMILY = "a4w4"
# Mirrors OPUS_MXFP4_SK_MAX_TILES / opus_mxfp4_sk_ws_bytes in
# opus_gemm_traits_a4w4_mxfp4_nt_gfx950.cuh: [4 KB tile counters][tiles][S][B_M * B_N] fp32.
_SK_MAX_TILES = 1024
_SK_COUNTER_BYTES = _SK_MAX_TILES * 4
# Small-tile split-K workspace: [256 B handle][S][batch][padded_M][padded_N] fp32.
_SMALL_WS_HEADER_NUMEL = 64


def _gen_opus_gemm_a4w4_launch_fake_tensors(
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
    fc_name="opus_gemm_a4w4_launch",
    gen_fake=_gen_opus_gemm_a4w4_launch_fake_tensors,
    develop=True,
)
def _opus_gemm_a4w4_launch_raw(
    XQ: Tensor,
    WQ: Tensor,
    Y: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    workspace: Tensor | None,
    kid: int,
    split_k: int,
) -> Tensor: ...


# ---- launch planning --------------------------------------------------------


def a4w4_pick_split(tiles: int, K: int, cu_num: int) -> int:
    """Split-K factor of the ``sk`` kids: the largest S in {8, 4, 2} with
    tiles * S <= CUs and K a multiple of S * 256 (whole K tiles per segment)."""
    if tiles > _SK_MAX_TILES:
        return 1
    for s in (8, 4, 2):
        if tiles * s <= cu_num and K % (s * 256) == 0:
            return s
    return 1


def _cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


@lru_cache(maxsize=4096)
def _a4w4_plan(
    arch: str, cu_num: int, kid: int, M: int, N: int, K: int, batch: int, split_k: int
) -> tuple[int, int]:
    """Return (ABI split_k, fp32 workspace elements; 0 = none) for one launch."""
    inst = get_kernel_instance(arch, A4W4_FAMILY, kid, torch.bfloat16, cu_num=cu_num)
    if inst is None:
        raise ValueError(
            f"no registered OPUS kernel for (arch={arch!r}, family={A4W4_FAMILY!r}, "
            f"kid={kid}, cu_num={cu_num})"
        )
    if split_k < 0:
        raise ValueError(f"OPUS a4w4 split_k must be >= 0, got {split_k}")
    tiles = _cdiv(M, inst.B_M) * _cdiv(N, inst.B_N) * batch
    if inst.kernel_tag == "a4w4_mxfp4_nt_small":
        S = max(1, split_k)
        if K // inst.B_K < S:
            raise ValueError(
                f"OPUS a4w4 kid {kid}: K / {inst.B_K} must be >= split_k={S}"
            )
        if S == 1:
            return 1, 0
        padded = _cdiv(M, inst.B_M) * inst.B_M * _cdiv(N, inst.B_N) * inst.B_N
        return S, _SMALL_WS_HEADER_NUMEL + S * batch * padded
    if inst.co_variant == "nob" and (M % inst.B_M or N % inst.B_N):
        raise ValueError(
            f"OPUS a4w4 kid {kid} needs M % {inst.B_M} == 0 and N % {inst.B_N} == 0"
        )
    if A4W4_CO_MODE[inst.co_variant] != 2:
        if split_k > 1:
            raise ValueError(
                f"OPUS a4w4 kid {kid} does not split K (split_k={split_k})"
            )
        return 1, 0
    S = split_k if split_k > 0 else a4w4_pick_split(tiles, K, cu_num)
    if K % (S * inst.B_K):
        raise ValueError(
            f"OPUS a4w4 kid {kid}: K={K} is not a multiple of split_k * {inst.B_K} (split_k={S})"
        )
    if S == 1:
        return 1, 0
    if tiles > _SK_MAX_TILES:
        raise ValueError(
            f"OPUS a4w4 kid {kid}: split-K supports at most {_SK_MAX_TILES} tiles"
        )
    return S, (_SK_COUNTER_BYTES + tiles * S * inst.B_M * inst.B_N * 4) // 4


def _launch_a4w4(
    XQ: Tensor,
    WQ: Tensor,
    Y: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    *,
    kid: int,
    split_k: int,
    workspace: Tensor | None,
    arch: str,
    cu_num: int,
) -> None:
    """Launch batch-first 3D tensors (scales 2D or 3D) with a caller- or Torch-owned workspace."""
    batch, M, K2 = map(int, XQ.shape)
    N = int(WQ.shape[1])
    abi_split_k, ws_numel = _a4w4_plan(
        arch, cu_num, int(kid), M, N, 2 * K2, batch, int(split_k)
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
            f"OPUS a4w4 kid {kid} needs a contiguous fp32 workspace of >= {ws_numel} "
            f"elements, got {workspace.dtype} {tuple(workspace.shape)}"
        )
    with torch.cuda.device(XQ.device):
        _opus_gemm_a4w4_launch_raw(
            XQ, WQ, Y, x_scale, w_scale, workspace, int(kid), abi_split_k
        )


def _check_a4w4_inputs(
    XQ: Tensor, WQ: Tensor, x_scale: Tensor, w_scale: Tensor
) -> None:
    batch, M, K = XQ.shape[0], XQ.shape[1], XQ.shape[2] * 2
    N = WQ.shape[1]
    if WQ.shape[0] != batch or WQ.shape[2] * 2 != K:
        raise ValueError(
            f"opus a4w4: WQ shape {tuple(WQ.shape)} does not match XQ {tuple(XQ.shape)}"
        )
    if N % 16:
        raise ValueError(f"opus a4w4: N must be a multiple of 16 (got N={N})")
    if K % 256 or K <= 0:
        raise ValueError(f"opus a4w4: K must be a multiple of 256 (got K={K})")
    cols = (K // 32 + 7) // 8 * 8
    for s, rows, name in ((x_scale, M, "x_scale"), (w_scale, N, "w_scale")):
        if (
            s.dim() not in (2, 3)
            or (s.dim() == 3 and s.shape[0] != batch)
            or (s.dim() == 2 and s.shape[0] % batch)
        ):
            raise ValueError(
                f"opus a4w4: {name} must be 2D [batch*rows, cols] or [batch, rows, cols], "
                f"got {tuple(s.shape)}"
            )
        r = s.shape[1] if s.dim() == 3 else s.shape[0] // batch
        need = (rows + 31) // 32 * 32
        if s.shape[-1] != cols or r < need or r % 32:
            raise ValueError(
                f"opus a4w4: {name} must be e8m0_shuffle'd: >= {need} rows (multiple of 32) "
                f"x {cols} cols per batch, got {tuple(s.shape)}"
            )


# ---- unified opus_gemm / opus_bmm route -----------------------------------


def _validate_a4w4_public_contract(
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
    if input_dtype != dtypes.fp4x2 or weight_dtype != dtypes.fp4x2:
        raise ValueError(
            f"OPUS a4w4 kid {kid} requires fp4x2 XQ/WQ; got {input_dtype}/{weight_dtype}"
        )
    if output_dtype != torch.bfloat16:
        raise ValueError(f"OPUS a4w4 kid {kid} requires bf16 Y; got {output_dtype}")
    if layout != "bpreshuffle":
        raise ValueError(
            f"OPUS kid {kid} belongs to family a4w4 and requires layout='bpreshuffle'; got {layout!r}"
        )
    if not (has_x_scale and has_w_scale):
        raise ValueError("OPUS a4w4 requires x_scale and w_scale")
    if has_bias:
        raise ValueError("OPUS family a4w4 does not support bias")


def _launch_a4w4_public(
    operation: str,
    XQ: Tensor,
    WQ: Tensor,
    Y: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    *,
    kid: int,
    split_k: int,
    workspace: Tensor | None,
    arch: str,
    cu_num: int,
) -> Tensor:
    """``opus_gemm`` (2D) / ``opus_bmm`` (batch-first 3D) entry of the a4w4 family."""
    if operation == "opus_gemm":
        x3, w3, y3 = XQ.unsqueeze(0), WQ.unsqueeze(0), Y.unsqueeze(0)
    else:
        x3, w3, y3 = XQ, WQ, Y
    _check_a4w4_inputs(x3, w3, x_scale, w_scale)
    _launch_a4w4(
        x3,
        w3,
        y3,
        x_scale,
        w_scale,
        kid=kid,
        split_k=split_k,
        workspace=workspace,
        arch=arch,
        cu_num=cu_num,
    )
    return Y


__all__ = ["a4w4_pick_split"]
