# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
"""OPUS gfx950 MXFP4 x MXFP4 GEMM (family ``a4w4``), ``layout="plain"`` kids.

``layout`` declares the operand *content* (as in every OPUS family):
``"plain"`` = row-major operands and scales, the layout of Triton
``gemm_afp4wfp4``. NT / NN / TT / TN are not a ``layout`` value; like
``a8w8_mxfp8`` they would come from the XQ / WQ strides (only NT exists today).

``layout="plain"`` (NT) operands:

* ``XQ``: [M, K/2] (or [batch, M, K/2]) fp4x2, row-major.
* ``WQ``: [N, K/2] (or [batch, N, K/2]) fp4x2, row-major (no
  ``shuffle_weight``).
  ``opus_gemm`` / ``opus_bmm`` reject any other XQ / WQ dtype; only
  :func:`gemm_afp4wfp4_opus` also takes uint8 (viewed as fp4x2).
* ``x_scale`` / ``w_scale``: 1x32 e8m0 (or uint8), row-major (no
  ``e8m0_shuffle``), 2D [batch * rows, K/32] or 3D [batch, rows, K/32] with
  rows >= M / N (no padding needed).

Any M; N % 8 == 0; K % 256 == 0. Output bf16 [M, N] (or [batch, M, N]).

Local kids (``a4w4/gfx950/<CU>``, plain): 100-105 persistent, 110-115 one
tile per block, 130 persistent without M/N boundary handling (the pre-compiled
2x2 pipeline, ``gen_co/gfx950/a4w4``), 500-504 the JIT-built 1x4 small-tile pipeline
(split-K through ``split_k`` and a Torch workspace). Launched through
``opus_gemm`` / ``opus_bmm`` (``family="a4w4"``, ``layout="plain"``) or
:func:`gemm_afp4wfp4_opus`, which takes the arguments of Triton ``gemm_afp4wfp4``.
"""

from functools import lru_cache

import torch
from torch import Tensor

from csrc.opus_gemm.opus_gemm_common import get_kernel_instance

from ...jit.core import compile_ops
from ...utility import dtypes
from ._arch import _device_arch_and_cu

A4W4_FAMILY = "a4w4"
# Small-tile split-K workspace: [256 B handle][S][batch][padded_M][padded_N] fp32.
_SMALL_WS_HEADER_NUMEL = 64


# ---- Low-level backend -------------------------------------------------------


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
            f"no registered OPUS kernel for (arch={arch!r}, "
            f"family={A4W4_FAMILY!r}, kid={kid}, cu_num={cu_num})"
        )
    if split_k < 0:
        raise ValueError(f"OPUS a4w4 split_k must be >= 0, got {split_k}")
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
            f"OPUS a4w4 kid {kid} needs M % {inst.B_M} == 0 and " f"N % {inst.B_N} == 0"
        )
    if split_k > 1:
        raise ValueError(f"OPUS a4w4 kid {kid} does not split K (split_k={split_k})")
    return 1, 0


# ---- Execution adapters (opus_gemm / opus_bmm, gemm_afp4wfp4_opus) -----------


def _launch_a4w4_bmm(
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
    abi_split_k, ws_numel = _a4w4_plan(
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
        _opus_gemm_a4w4_launch_raw(
            XQ, WQ, Y, x_scale, w_scale, workspace, int(kid), abi_split_k
        )
    return Y


def _launch_a4w4_gemm(
    XQ: Tensor,
    WQ: Tensor,
    Y: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    **kwargs,
) -> Tensor:
    """Launch logical 2D ``[M,K/2] x [N,K/2] -> bf16 [M,N]``."""
    _launch_a4w4_bmm(
        XQ.unsqueeze(0), WQ.unsqueeze(0), Y.unsqueeze(0), x_scale, w_scale, **kwargs
    )
    return Y


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
    """Immutable per-kid contract checked (and cached) by ``dispatch._resolve_contract``."""
    if input_dtype != dtypes.fp4x2 or weight_dtype != dtypes.fp4x2:
        raise ValueError(
            f"OPUS a4w4 kid {kid} requires fp4x2 XQ/WQ; got "
            f"{input_dtype}/{weight_dtype}"
        )
    if output_dtype != torch.bfloat16:
        raise ValueError(f"OPUS a4w4 kid {kid} requires bf16 Y; got {output_dtype}")
    if layout != "plain":
        raise ValueError(
            f"OPUS kid {kid} is an a4w4 row-major kid and requires "
            f"layout='plain'; got {layout!r}"
        )
    if not (has_x_scale and has_w_scale):
        raise ValueError("OPUS a4w4 requires x_scale and w_scale")
    if has_bias:
        raise ValueError("OPUS family a4w4 does not support bias")


def gemm_afp4wfp4_opus(
    x: Tensor,
    w: Tensor,
    x_scales: Tensor,
    w_scales: Tensor,
    dtype: torch.dtype | None = torch.bfloat16,
    y: Tensor | None = None,
    kernelId: int | None = None,
    splitK: int | None = None,
) -> Tensor:
    """OPUS counterpart of Triton ``gemm_afp4wfp4``: ``y = x @ w^T`` (gfx950).

    Same arguments as ``gemm_afp4wfp4`` (row-major ``x`` [M, K/2] / ``w``
    [N, K/2] fp4x2 or uint8, row-major 1x32 e8m0 ``x_scales`` / ``w_scales``),
    bf16 output only. ``kernelId=None`` picks
    :func:`aiter.ops.opus.policy.a4w4_heuristic_kid`; ``splitK``
    (small-tile kids 500-504 only): None / 0 / 1 = no split, S >= 2 splits K S
    ways. 3D (batched) ``x`` / ``w`` / ``y`` are accepted too. Returns ``y``,
    allocated when omitted.
    """
    if dtype not in (None, torch.bfloat16):
        raise ValueError(f"OPUS a4w4 GEMM supports bf16 output, got {dtype}")
    x3 = (x if x.dim() == 3 else x.unsqueeze(0)).view(dtypes.fp4x2)
    w3 = (w if w.dim() == 3 else w.unsqueeze(0)).view(dtypes.fp4x2)
    batch, M, K2 = map(int, x3.shape)
    N = int(w3.shape[1])
    if y is None:
        shape = (M, N) if x.dim() == 2 else (batch, M, N)
        y = torch.empty(shape, device=x.device, dtype=torch.bfloat16)
    if kernelId is None:
        from .policy import a4w4_heuristic_kid

        gfx, cu_num = _device_arch_and_cu(x.device)
        kernelId = a4w4_heuristic_kid(M, N, 2 * K2, batch, gfx=gfx, cu_num=cu_num)
    _launch_a4w4_bmm(
        x3,
        w3,
        y if y.dim() == 3 else y.unsqueeze(0),
        x_scales,
        w_scales,
        kid=int(kernelId),
        split_k=int(splitK or 0),
    )
    return y


__all__ = ["gemm_afp4wfp4_opus"]
