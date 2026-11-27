# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Private OPUS A8W8 exact-kid launch APIs."""

# Keep annotations eager in this binding module.  ``torch_compile_guard`` uses
# the first parameter's concrete ``torch.Tensor`` identity to decide whether a
# custom op already has a Tensor dispatch key.  Postponed string annotations
# make it add a dummy CUDA Tensor to every raw launch, which is observable host
# overhead on short BMM kernels.

import math
import re

import torch
from torch import Tensor

from csrc.opus_gemm.opus_gemm_common import (
    a8w8_mxscale_bpreshuffle_supports_shape,
    bpreshuffle_workspace_bytes,
    get_registry,
    mxfp8_direction,
)

from ...jit.core import compile_ops
from ._arch import _device_arch_and_cu
from .launch_plan import (
    _A8W8_BLOCKSCALE_FAMILY,
    _A8W8_BPRESHUFFLE_FAMILY,
    _A8W8_FAMILY,
    _A8W8_MXFP8_1X32_BPRESHUFFLE_FAMILY,
    _A8W8_MXFP8_1X32_FAMILY,
    _A8W8_MXFP8_FAMILY,
    _A8W8_MXSCALE_BMM_FAMILY,
    _FP8_DTYPES,
    _get_cached_a8w8_mxscale_bmm_plan,
    _require_registered_kid,
)

_E8M0_DTYPES = frozenset(
    dtype
    for dtype in (
        torch.uint8,
        getattr(torch, "float8_e8m0fnu", None),
    )
    if dtype is not None
)


# ---- Low-level A8W8 backend ----------------------------------------------


def _gen_opus_gemm_a8w8_launch_fake_tensors(
    XQ: Tensor,
    WQ: Tensor,
    Y: Tensor,
    kid: int,
) -> Tensor:
    return Y


@compile_ops(
    "module_deepgemm_opus",
    fc_name="opus_gemm_a8w8_launch",
    gen_fake=_gen_opus_gemm_a8w8_launch_fake_tensors,
    develop=True,
)
def _opus_gemm_a8w8_launch_raw(
    XQ: Tensor,
    WQ: Tensor,
    Y: Tensor,
    kid: int,
) -> Tensor: ...


def _gen_opus_gemm_a8w8_blockscale_launch_fake_tensors(
    XQ: Tensor,
    WQ: Tensor,
    Y: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    kid: int,
) -> Tensor:
    return Y


@compile_ops(
    "module_deepgemm_opus",
    fc_name="opus_gemm_a8w8_blockscale_launch",
    gen_fake=_gen_opus_gemm_a8w8_blockscale_launch_fake_tensors,
    develop=True,
)
def _opus_gemm_a8w8_blockscale_launch_raw(
    XQ: Tensor,
    WQ: Tensor,
    Y: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    kid: int,
) -> Tensor: ...


def _gen_opus_gemm_a8w8_blockscale_bpreshuffle_launch_fake_tensors(
    XQ: Tensor,
    WQ: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    Y: Tensor,
    kid: int,
    workspace: Tensor | None = None,
) -> Tensor:
    return Y


def _gen_opus_gemm_a8w8_blockscale_mxfp8_launch_fake_tensors(
    XQ: Tensor,
    WQ: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    Y: Tensor,
    kid: int,
) -> Tensor:
    return Y


@compile_ops(
    "module_deepgemm_opus",
    fc_name="opus_gemm_a8w8_blockscale_mxfp8_launch",
    gen_fake=_gen_opus_gemm_a8w8_blockscale_mxfp8_launch_fake_tensors,
    develop=True,
)
def _opus_gemm_a8w8_blockscale_mxfp8_launch_raw(
    XQ: Tensor,
    WQ: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    Y: Tensor,
    kid: int,
) -> Tensor: ...


@compile_ops(
    "module_deepgemm_opus",
    fc_name="opus_gemm_a8w8_blockscale_bpreshuffle_launch",
    gen_fake=_gen_opus_gemm_a8w8_blockscale_bpreshuffle_launch_fake_tensors,
    develop=True,
)
def _opus_gemm_a8w8_blockscale_bpreshuffle_launch_raw(
    XQ: Tensor,
    WQ: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    Y: Tensor,
    kid: int,
    workspace: Tensor | None = None,
) -> Tensor: ...


def _gen_opus_gemm_a8w8_mxfp8_launch_fake_tensors(
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
    fc_name="opus_gemm_a8w8_mxfp8_launch",
    gen_fake=_gen_opus_gemm_a8w8_mxfp8_launch_fake_tensors,
    develop=True,
)
def _opus_gemm_a8w8_mxfp8_launch_raw(
    XQ: Tensor,
    WQ: Tensor,
    Y: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    workspace: Tensor | None,
    kid: int,
    split_k: int,
) -> Tensor: ...


@compile_ops(
    "module_deepgemm_opus",
    fc_name="opus_gemm_a8w8_mxfp8_bpreshuffle_launch",
    gen_fake=_gen_opus_gemm_a8w8_mxfp8_launch_fake_tensors,
    develop=True,
)
def _opus_gemm_a8w8_mxfp8_bpreshuffle_launch_raw(
    XQ: Tensor,
    WQ: Tensor,
    Y: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    workspace: Tensor | None,
    kid: int,
    split_k: int,
) -> Tensor: ...


def opus_gemm_a8w8_blockscale_bpreshuffle_tune(
    XQ: Tensor,
    WQ: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    Y: Tensor | None = None,
    kernelId: int = 0,
) -> Tensor:
    """Compatibility entry for the existing A8W8 blockscale tuner."""
    if Y is None:
        Y = torch.empty(
            (XQ.shape[-2], WQ.shape[-2]),
            device=XQ.device,
            dtype=torch.bfloat16,
        )

    return _launch_a8w8_blockscale_bpreshuffle_gemm(
        XQ,
        WQ,
        x_scale,
        w_scale,
        Y,
        kid=int(kernelId),
    )


def _gen_opus_gemm_a8w8_mxscale_bmm_launch_fake_tensors(
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
    fc_name="opus_gemm_a8w8_mxscale_bmm_launch",
    gen_fake=_gen_opus_gemm_a8w8_mxscale_bmm_launch_fake_tensors,
    develop=True,
)
def _opus_gemm_a8w8_mxscale_bmm_launch_raw(
    XQ: Tensor,
    WQ: Tensor,
    Y: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    workspace: Tensor | None,
    kid: int,
    split_k: int,
) -> Tensor: ...


def _launch_a8w8_backend(
    XQ: Tensor,
    WQ: Tensor,
    Y: Tensor,
    x_scale: Tensor | None,
    w_scale: Tensor | None,
    workspace: Tensor | None,
    family: str,
    kid: int,
    split_k: int,
) -> None:
    if family == _A8W8_FAMILY:
        if x_scale is not None or w_scale is not None:
            raise RuntimeError("A8W8 no-scale backend received scale tensors")
        if workspace is not None or split_k != 0:
            raise RuntimeError("A8W8 no-scale backend received split-K state")
        _opus_gemm_a8w8_launch_raw(XQ, WQ, Y, kid)
        return

    if x_scale is None or w_scale is None:
        raise RuntimeError(f"A8W8 backend family {family!r} requires both scales")

    if family == _A8W8_BLOCKSCALE_FAMILY:
        if workspace is not None or split_k != 0:
            raise RuntimeError("A8W8 blockscale backend received split-K state")
        _opus_gemm_a8w8_blockscale_launch_raw(
            XQ,
            WQ,
            Y,
            x_scale,
            w_scale,
            kid,
        )
        return

    if family == _A8W8_BPRESHUFFLE_FAMILY:
        if split_k != 0:
            raise RuntimeError(
                "A8W8 blockscale-bpreshuffle backend received split-K state"
            )
        _opus_gemm_a8w8_blockscale_bpreshuffle_launch_raw(
            XQ,
            WQ,
            x_scale,
            w_scale,
            Y,
            kid,
            *((workspace,) if workspace is not None else ()),
        )
        return

    if family == _A8W8_MXFP8_FAMILY:
        if workspace is not None or split_k != 0:
            raise RuntimeError("A8W8 blockscale-mxfp8 backend received split-K state")
        _opus_gemm_a8w8_blockscale_mxfp8_launch_raw(XQ, WQ, x_scale, w_scale, Y, kid)
        return

    if family == _A8W8_MXSCALE_BMM_FAMILY:
        _opus_gemm_a8w8_mxscale_bmm_launch_raw(
            XQ,
            WQ,
            Y,
            x_scale,
            w_scale,
            workspace,
            kid,
            split_k,
        )
        return

    raise RuntimeError(f"unsupported A8W8 backend family {family!r}")


# ---- A8W8 execution and logical adapters ---------------------------------


def _launch_a8w8_gemm(
    XQ: Tensor,
    WQ: Tensor,
    Y: Tensor,
    *,
    kid: int,
    route_arch: str | None = None,
    instance: object | None = None,
) -> Tensor:
    """Launch logical 2D no-scale FP8 ``[M,K] x [N,K] -> FP32 [M,N]``."""
    resolved_kid = kid
    if instance is None:
        if XQ.dim() != 2 or WQ.dim() != 2 or Y.dim() != 2:
            raise ValueError(
                "opus_gemm A8W8 expects logical 2D XQ/WQ/Y; " "this family is GEMM-only"
            )
        device_arch, cu_num = _device_arch_and_cu(XQ.device)
        arch = route_arch or device_arch
        resolved_kid = _require_registered_kid(
            arch=arch,
            family=_A8W8_FAMILY,
            kid=kid,
            output_dtype=Y.dtype,
            cu_num=cu_num,
        )
    _launch_a8w8_backend(
        XQ.unsqueeze(0),
        WQ.unsqueeze(0),
        Y.unsqueeze(0),
        None,
        None,
        None,
        _A8W8_FAMILY,
        resolved_kid,
        0,
    )
    return Y


def _launch_a8w8_blockscale_gemm(
    XQ: Tensor,
    WQ: Tensor,
    Y: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    *,
    kid: int,
    route_arch: str | None = None,
    instance: object | None = None,
) -> Tensor:
    """Launch logical 2D blockscale A8W8 GEMM with 2D scales."""
    resolved_kid = kid
    if instance is None:
        if any(tensor.dim() != 2 for tensor in (XQ, WQ, Y, x_scale, w_scale)):
            raise ValueError(
                "opus_gemm A8W8 blockscale expects logical 2D "
                "XQ/WQ/Y/x_scale/w_scale; this family is GEMM-only"
            )
        device_arch, cu_num = _device_arch_and_cu(XQ.device)
        arch = route_arch or device_arch
        resolved_kid = _require_registered_kid(
            arch=arch,
            family=_A8W8_BLOCKSCALE_FAMILY,
            kid=kid,
            output_dtype=Y.dtype,
            cu_num=cu_num,
        )
    _launch_a8w8_backend(
        XQ.unsqueeze(0),
        WQ.unsqueeze(0),
        Y.unsqueeze(0),
        x_scale,
        w_scale,
        None,
        _A8W8_BLOCKSCALE_FAMILY,
        resolved_kid,
        0,
    )
    return Y


def _launch_a8w8_blockscale_bpreshuffle_gemm(
    XQ: Tensor,
    WQ: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    Y: Tensor,
    *,
    kid: int,
    route_arch: str | None = None,
    instance: object | None = None,
    workspace: Tensor | None = None,
) -> Tensor:
    """Launch logical 2D bpreshuffled blockscale A8W8 GEMM.

    ``WQ`` pre-shuffle is a content/layout semantic. It
    cannot be proven from Tensor shape or strides. Build it with
    ``shuffle_weight(WQ, layout=(16, 16))``. The generated launcher checks
    output dtype, scale layout, batch and tile alignment.
    """
    resolved_kid = kid
    if instance is None:
        if any(tensor.dim() != 2 for tensor in (XQ, WQ, Y, x_scale, w_scale)):
            raise ValueError(
                "opus_gemm A8W8 blockscale bpreshuffle expects logical 2D "
                "XQ/WQ/Y/x_scale/w_scale; this family is GEMM-only"
            )
        device_arch, cu_num = _device_arch_and_cu(XQ.device)
        arch = route_arch or device_arch
        resolved_kid = _require_registered_kid(
            arch=arch,
            family=_A8W8_BPRESHUFFLE_FAMILY,
            kid=kid,
            output_dtype=Y.dtype,
            cu_num=cu_num,
        )
        instance = get_registry(_A8W8_BPRESHUFFLE_FAMILY, arch, cu_num)[resolved_kid]
    launch_workspace = workspace
    if instance.splitk_workspace_dtype is not None:
        if launch_workspace is None:
            M, K = map(int, XQ.shape)
            N = int(WQ.shape[0])
            if not a8w8_mxscale_bpreshuffle_supports_shape(instance, M, N, K):
                raise ValueError(
                    f"OPUS bpreshuffle kid {resolved_kid}: "
                    "shape exceeds this pipeline configuration"
                )
            required_bytes = bpreshuffle_workspace_bytes(instance, M, N, K)
            launch_workspace = torch.empty(
                (required_bytes // 4,), dtype=torch.float32, device=XQ.device
            )
    elif launch_workspace is not None:
        raise ValueError(f"OPUS bpreshuffle kid {resolved_kid} does not use workspace")
    _launch_a8w8_backend(
        XQ.unsqueeze(0),
        WQ.unsqueeze(0),
        Y.unsqueeze(0),
        x_scale,
        w_scale,
        launch_workspace,
        _A8W8_BPRESHUFFLE_FAMILY,
        resolved_kid,
        0,
    )
    if workspace is not None:
        workspace.record_stream(torch.cuda.current_stream(XQ.device))
    return Y


def _launch_a8w8_blockscale_mxfp8_gemm(
    XQ: Tensor,
    WQ: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    Y: Tensor,
    *,
    kid: int,
    route_arch: str | None = None,
    instance: object | None = None,
) -> Tensor:
    """Launch logical 2D MXFP8 blockscale GEMM with plain A / B.

    ``XQ`` [M, K] / ``WQ`` [N, K] are logical views; the kid's layout (NT / NN /
    TT / TN) must match their strides, which the generated launcher checks.
    """
    resolved_kid = kid
    if instance is None:
        if any(tensor.dim() != 2 for tensor in (XQ, WQ, Y, x_scale, w_scale)):
            raise ValueError(
                "opus_gemm A8W8 blockscale mxfp8 expects logical 2D "
                "XQ/WQ/Y/x_scale/w_scale; this family is GEMM-only"
            )
        device_arch, cu_num = _device_arch_and_cu(XQ.device)
        resolved_kid = _require_registered_kid(
            arch=route_arch or device_arch,
            family=_A8W8_MXFP8_FAMILY,
            kid=kid,
            output_dtype=Y.dtype,
            cu_num=cu_num,
        )
    _launch_a8w8_backend(
        XQ.unsqueeze(0),
        WQ.unsqueeze(0),
        Y.unsqueeze(0),
        x_scale,
        w_scale,
        None,
        _A8W8_MXFP8_FAMILY,
        resolved_kid,
        0,
    )
    return Y


# ---- MXFP8 NT / NN / TT / TN, plain B (gfx950 CO kids) ------------------------


def _mxfp8_direction(XQ: Tensor, WQ: Tensor) -> str | None:
    """Storage of logical XQ [M, K] / WQ [N, K] as the kernel names spell it.

    First letter A, second B. A "N" is stored [M, K] (strides (K, 1)), A "T"
    [K, M] (strides (1, M)); B "T" is stored [N, K] (strides (K, 1)), B "N"
    [K, N] (strides (1, N)). Returns "NN", "TT", "TN", "NT" or None.
    """
    (M, K), N = XQ.shape, WQ.shape[0]
    if XQ.stride() == (K, 1):
        a = "N"
    elif XQ.stride() == (1, M):
        a = "T"
    else:
        return None
    if WQ.stride() == (K, 1):
        b = "T"
    elif WQ.stride() == (1, N):
        b = "N"
    else:
        return None
    return a + b


def _select_mxfp8_direction_kid(
    direction: str, M: int, N: int, K: int, arch: str, cu_num: int
) -> int | None:
    """Pick a supported NT / NN / TT / TN kid for the shape.

    256x128 tiles for N not a multiple of 256 or up to 2048 x 2048 outputs
    (more tiles to fill the CUs), else 256x256; GroupM 16 from M = 16384;
    PadM only when M is not a multiple of 256.
    """
    want_bn = 256 if N % 256 == 0 and M * N > 2048 * 2048 else 128
    want_group_m = 16 if M >= 16384 else 0
    best = None
    for kid, instance in get_registry(_A8W8_MXFP8_FAMILY, arch, cu_num).items():
        if mxfp8_direction(instance.bpreshuffle_co) != direction:
            continue
        if not a8w8_mxscale_bpreshuffle_supports_shape(instance, M, N, K):
            continue
        group_m = re.search(r"_gm(\d+)", instance.name)
        rank = (
            instance.B_N != want_bn,
            (int(group_m[1]) if group_m else 0) != want_group_m,
            instance.m_align != instance.B_M,
            kid,
        )
        if best is None or rank < best[0]:
            best = (rank, kid)
    return None if best is None else best[1]


def gemm_a8w8_blockscale_opus(
    XQ: Tensor,
    WQ: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    dtype: torch.dtype = torch.bfloat16,
    *,
    kernelId: int | None = None,
    out: Tensor | None = None,
) -> Tensor:
    """gfx950 MXFP8 1x128 blockscale GEMM for NT / NN / TT / TN operand storage.

    Computes Y[M, N] = XQ[M, K] @ WQ[N, K]^T on logical shapes; the storage is
    read from the strides (see ``_mxfp8_direction``), so pass the transposed
    view of a [K, M] A or of a [K, N] B, never a copy. A and B are plain FP8
    (no ``shuffle_weight``), as DeepGEMM's fp8_gemm_{nt,nn,tn,tt}. x_scale is
    the E8M0 [M, K/128] scale with column-major bytes; w_scale the E8M0
    [N/128, K/128] scale stored like WQ ([N/128, K/128] for B "T",
    [K/128, N/128] for B "N"). Output is BF16.

    ``kernelId`` runs that a8w8_blockscale_mxfp8 kid (0-39: NT 0-9, NN 10-19,
    TT 20-29, TN 30-39); otherwise one is picked by shape.
    """
    if XQ.dim() != 2 or WQ.dim() != 2:
        raise ValueError("gemm_a8w8_blockscale_opus expects 2D XQ / WQ")
    if dtype != torch.bfloat16:
        raise ValueError("gemm_a8w8_blockscale_opus writes BF16 only")
    direction = _mxfp8_direction(XQ, WQ)
    if direction is None:
        raise ValueError(
            "gemm_a8w8_blockscale_opus serves NT / NN / TT / TN storage; got "
            f"XQ strides {tuple(XQ.stride())}, WQ strides {tuple(WQ.stride())}"
        )
    M, K = map(int, XQ.shape)
    N = int(WQ.shape[0])
    if out is None:
        out = torch.empty((M, N), dtype=dtype, device=XQ.device)
    elif tuple(out.shape) != (M, N) or out.dtype != dtype:
        raise ValueError(
            f"out must be {dtype} [{M}, {N}], got {out.dtype} {tuple(out.shape)}"
        )
    if kernelId is None:
        arch, cu_num = _device_arch_and_cu(XQ.device)
        kernelId = _select_mxfp8_direction_kid(direction, M, N, K, arch, cu_num)
        if kernelId is None:
            raise ValueError(
                f"no {direction} MXFP8 kernel supports M={M}, N={N}, K={K} on {arch} "
                "(M % 16, N % 128 and K % 128 must be 0)"
            )
    return _launch_a8w8_blockscale_mxfp8_gemm(
        XQ, WQ, x_scale, w_scale, out, kid=int(kernelId)
    )


# a8w8_mxfp8 (plain): XQ [M, K] / WQ [N, K] are logical views of the stored
# operands; the kid's launcher picks the layout image from their strides.
# A stored [K, M] needs M % 16 and B stored [K, N] N % 16.
#   NT: XQ (K,1)  WQ (K,1)      NN: XQ (K,1)  WQ (1,N)
#   TT: XQ (1,M)  WQ (K,1)      TN: XQ (1,M)  WQ (1,N)
# a8w8_mxfp8_bpreshuffle: NT, WQ = shuffle_weight(w, layout=(16, 16)) of B [N, K].
# x_scale [M, K/32] and w_scale [N, K/32] are row-major e8m0 and Y [M, N] bf16
# in every layout. Both families have the same kids (tiles / variants).
_MXFP8_K_ALIGN = 256  # two K tiles of 128: the kernels run K tiles in pairs


def a8w8_mxfp8_kid_runs(
    instance, M: int, N: int, K: int, split_k: int = 0, layout: str = "NT"
) -> bool:
    """Whether a8w8_mxfp8(_bpreshuffle) kid ``instance`` runs (M, N, K) in
    ``layout`` (NT / NN / TT / TN) with ``split_k``. A / B are based per tile,
    so they may exceed 4 GiB: the 32-bit buffer ranges bound one tile of an
    operand (B_M / B_N rows of K, or B_K rows of a stored [K, M] / [K, N]) and
    its scales, one C tile to 2 GiB (C itself may exceed 4 GiB) and the
    split-K workspace to 4 GiB. K is a multiple of 32; the boundary kids
    ("np", persistent) run one that is not a multiple of 256 on scales whose
    1x32 groups past K are zeroed (see :func:`_a8w8_mxfp8_scale_groups`)."""
    a_t, b_t = layout.upper()[0] == "T", layout.upper()[1] == "N"
    # A stored [K, M] needs M % 16 (B stored [K, N] N % 16, as every N below).
    if a_t and M % 16:
        return False
    if K <= 0 or K % 32:
        return False
    if K % _MXFP8_K_ALIGN and instance.co_variant not in ("np", ""):
        return False
    limit = (1 << 31) - 1
    for transposed, rows, tile_rows in ((a_t, M, instance.B_M), (b_t, N, instance.B_N)):
        tile_bytes = (instance.B_K + 1) * rows if transposed else tile_rows * K
        if tile_bytes > limit or rows * (K // 32) > limit:
            return False
    # C: a whole tile whatever M is (see the launcher's check).
    if 2 * ((instance.B_M - 1) * N + instance.B_N) > limit:
        return False
    if instance.co_variant == "nob":
        if M % instance.B_M or N % instance.B_N:
            return False
    elif N % 16:
        return False
    if instance.co_variant == "sk":
        return split_k <= 1 or (
            K % (split_k * _MXFP8_K_ALIGN) == 0
            and _a8w8_mxfp8_workspace_bytes(instance, M, N, split_k) <= (1 << 32) - 1
        )
    return split_k <= 1


def _a8w8_mxfp8_scale_groups(K: int) -> int:
    """Scale columns a launch of K reads: K / 32, or for a K that is not a
    multiple of 256 every 1x32 group of the K tile pairs the kernel runs
    (ceil(K / 256) * 8). The groups past K / 32 must be 0x00 (2^-127): they
    multiply whatever follows the operands' K bytes, and both scales being
    zero the products underflow to exactly 0."""
    return K // 32 if K % _MXFP8_K_ALIGN == 0 else -(-K // _MXFP8_K_ALIGN) * 8


def _a8w8_mxfp8_scale(scale: Tensor, K: int) -> Tensor:
    """``scale`` [rows, K / 32] with the zeroed groups of a ragged K appended
    (a copy of K / 32 bytes per row); returned as is when K is a multiple of
    256 or the caller already passes every group."""
    groups = _a8w8_mxfp8_scale_groups(K)
    if scale.shape[-1] >= groups:
        return scale
    padded = scale.new_zeros((*scale.shape[:-1], groups), dtype=torch.uint8)
    padded[..., : scale.shape[-1]] = scale.view(torch.uint8)
    return padded.view(scale.dtype)


def _a8w8_mxfp8_workspace_bytes(instance, M: int, N: int, split_k: int) -> int:
    """fp32 partials [S][padded_M][padded_N] of a split-K launch, in bytes;
    splitk_reduce_kernel addresses them through one 32-bit buffer range."""
    padded_m = -(-M // instance.B_M) * instance.B_M
    padded_n = -(-N // instance.B_N) * instance.B_N
    return split_k * padded_m * padded_n * 4


def _a8w8_mxfp8_plan(instance, M: int, N: int, K: int, split_k: int, cu_num: int):
    """(ABI split_k, fp32 workspace elements; 0 = none) of one launch. Split-K
    kids with split_k = 0 take the largest S in {8, 4, 2} with tiles * S <= CUs,
    K % (S * 256) == 0 and a workspace under 4 GiB. S > 1 needs fp32 partials
    [S][padded_M][padded_N], summed into Y by splitk_reduce_kernel; the launcher
    checks split_k / K / the workspace size."""
    if instance.co_variant != "sk":
        return split_k, 0
    padded_m = -(-M // instance.B_M) * instance.B_M
    padded_n = -(-N // instance.B_N) * instance.B_N
    tiles = (padded_m // instance.B_M) * (padded_n // instance.B_N)
    S = split_k or next(
        (
            s
            for s in (8, 4, 2)
            if tiles * s <= cu_num
            and K % (s * _MXFP8_K_ALIGN) == 0
            and _a8w8_mxfp8_workspace_bytes(instance, M, N, s) <= (1 << 32) - 1
        ),
        1,
    )
    return S, S * padded_m * padded_n if S > 1 else 0


def _launch_a8w8_mxfp8_gemm(
    XQ: Tensor,
    WQ: Tensor,
    Y: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    *,
    family: str = _A8W8_MXFP8_1X32_FAMILY,
    kid: int,
    split_k: int = 0,
    workspace: Tensor | None = None,
    route_arch: str | None = None,
    instance: object | None = None,
) -> Tensor:
    """Launch logical 2D MXFP8 x MXFP8 GEMM (1x32 e8m0 scales, gfx950) on a kid
    of ``family`` (a8w8_mxfp8 or a8w8_mxfp8_bpreshuffle). The kid's launcher
    checks the shape / stride / workspace contract and picks the layout image
    from the XQ / WQ strides.
    ``split_k``: split-K kids only, 0 = auto (see :func:`_a8w8_mxfp8_plan`)."""
    if any(tensor.dim() != 2 for tensor in (XQ, WQ, Y, x_scale, w_scale)):
        raise ValueError(
            f"opus_gemm {family} expects logical 2D XQ/WQ/Y/x_scale/w_scale; "
            "this family is GEMM-only"
        )
    if x_scale.dtype not in _E8M0_DTYPES or w_scale.dtype not in _E8M0_DTYPES:
        raise ValueError(
            f"OPUS {family} requires e8m0 scales; got {x_scale.dtype}/{w_scale.dtype}"
        )
    resolved_kid = kid
    device_arch, cu_num = _device_arch_and_cu(XQ.device)
    if instance is None:
        arch = route_arch or device_arch
        resolved_kid = _require_registered_kid(
            arch=arch,
            family=family,
            kid=kid,
            output_dtype=Y.dtype,
            cu_num=cu_num,
        )
        instance = get_registry(family, arch, cu_num)[resolved_kid]
    m, k = XQ.shape
    n = WQ.shape[0]
    abi_split_k, ws_numel = _a8w8_mxfp8_plan(instance, m, n, k, int(split_k), cu_num)
    if ws_numel and workspace is None:
        workspace = torch.empty(ws_numel, dtype=torch.float32, device=XQ.device)
    launch = (
        _opus_gemm_a8w8_mxfp8_bpreshuffle_launch_raw
        if family == _A8W8_MXFP8_1X32_BPRESHUFFLE_FAMILY
        else _opus_gemm_a8w8_mxfp8_launch_raw
    )
    with torch.cuda.device(XQ.device):
        launch(
            XQ.unsqueeze(0),
            WQ.unsqueeze(0),
            Y.unsqueeze(0),
            _a8w8_mxfp8_scale(x_scale, k),
            _a8w8_mxfp8_scale(w_scale, k),
            workspace,
            resolved_kid,
            abi_split_k,
        )
    return Y


# Measured us per 1000 K of a (B_M, B_N) tile on the "np" kids (MI355X, 256 CUs),
# per layout: (a launch of up to half the CUs' tiles, one round of cu_num tiles
# of a launch of several rounds). A tile's time is not its area's share (the K
# loop has a fixed part, larger for an operand stored [K, M] / [K, N]) and tiles
# sharing a CU slow each other down, hence the two columns.
_MXFP8_TILE_US = {
    "NT": {
        (256, 256): (9.0, 15.6),
        (256, 192): (7.8, 14.0),
        (192, 256): (7.7, 12.8),
        (256, 128): (5.2, 9.4),
        (128, 256): (5.3, 9.5),
        (128, 128): (4.0, 6.2),
    },
    "NN": {
        (256, 256): (9.6, 16.6),
        (256, 192): (8.9, 16.1),
        (192, 256): (8.1, 13.6),
        (256, 128): (7.3, 11.6),
        (128, 256): (5.6, 10.0),
        (128, 128): (5.2, 7.4),
    },
    "TT": {
        (256, 256): (9.4, 15.9),
        (256, 192): (8.4, 13.8),
        (192, 256): (9.1, 14.1),
        (256, 128): (6.0, 10.3),
        (128, 256): (7.5, 11.0),
        (128, 128): (4.7, 7.2),
    },
    "TN": {
        (256, 256): (9.9, 15.9),
        (256, 192): (9.5, 15.5),
        (192, 256): (9.5, 14.0),
        (256, 128): (7.9, 11.5),
        (128, 256): (7.8, 11.0),
        (128, 128): (7.2, 8.8),
    },
}
_MXFP8_SPLITK_US = 14.0  # a split-K launch: the workspace and splitk_reduce_kernel
_MXFP8_SPLITK_MIN_US = 45.0  # below this a launch is not worth splitting


def _a8w8_mxfp8_kid_us(
    direction: str, instance, M: int, N: int, K: int, cu_num: int
) -> float | None:
    """Estimated us of one launch of ``instance`` (split-K kids at their auto
    split, see :func:`_a8w8_mxfp8_plan`), None for a kid the estimate does not
    cover. Only the order of the kids of one shape is meant to be right.

    Up to cu_num tiles each has a CU: the tile's time, up to 25 % more as the
    launch goes from half the CUs to all of them. Past that the tiles run in
    rounds of cu_num, a started round costing half a full one. Split-K divides
    K among ``split`` blocks per tile and adds a fixed cost."""
    tile_us = _MXFP8_TILE_US[direction].get((instance.B_M, instance.B_N))
    if tile_us is None or instance.co_variant not in ("nob", "np", "sk"):
        return None
    k = -(-K // _MXFP8_K_ALIGN) * _MXFP8_K_ALIGN / 1000
    tiles = -(-M // instance.B_M) * -(-N // instance.B_N)
    if instance.co_variant == "sk":
        split, _ = _a8w8_mxfp8_plan(instance, M, N, K, 0, cu_num)
        if split <= 1:
            return None
        k /= split
        tiles *= split
    rounds = tiles / cu_num
    if rounds <= 1:
        us = k * tile_us[0] * (1 + 0.5 * max(0.0, rounds - 0.5))
    else:
        us = k * tile_us[1] * (rounds + 0.5 * (math.ceil(rounds) - rounds))
    if instance.co_variant == "sk":
        return us + _MXFP8_SPLITK_US
    # No boundary code: about 3 % faster than the "np" kid of the same tile.
    return us * 0.97 if instance.co_variant == "nob" else us


def _select_a8w8_mxfp8_kid(
    direction: str, M: int, N: int, K: int, arch: str, cu_num: int
) -> int | None:
    """Pick an a8w8_mxfp8 (plain) kid for the shape: the "nob" / "np" kid with
    the lowest estimate (:func:`_a8w8_mxfp8_kid_us`), or a split-K kid when that
    launch is long enough to split and the split-K estimate is lower. Ties go
    to the larger tile. The persistent kids time as the "np" ones and are only
    run by ``kernelId``."""
    best = best_sk = None
    for kid, instance in get_registry(_A8W8_MXFP8_1X32_FAMILY, arch, cu_num).items():
        us = _a8w8_mxfp8_kid_us(direction, instance, M, N, K, cu_num)
        if us is None:
            continue
        split_k = 0
        if instance.co_variant == "sk":
            split_k, _ = _a8w8_mxfp8_plan(instance, M, N, K, 0, cu_num)
        if not a8w8_mxfp8_kid_runs(instance, M, N, K, split_k, direction):
            continue
        rank = (us, -instance.B_M * instance.B_N, -instance.B_M, kid)
        if instance.co_variant == "sk":
            if best_sk is None or rank < best_sk:
                best_sk = rank
        elif best is None or rank < best:
            best = rank
    if best is None:
        return None
    if best_sk is not None and best[0] > _MXFP8_SPLITK_MIN_US and best_sk[0] < best[0]:
        return best_sk[-1]
    return best[-1]


def _a8w8_mxfp8_pad_k(operand: Tensor, transposed: bool, K: int) -> Tensor:
    """Logical [rows, K'] ``operand`` zero-padded to [rows, K] in the storage
    it has ([rows, K'], or [K', rows] when ``transposed``): a copy."""
    rows, k = operand.shape
    shape = (K, rows) if transposed else (rows, K)
    padded = torch.zeros(shape, dtype=torch.uint8, device=operand.device)
    if transposed:
        padded[:k] = operand.T.view(torch.uint8)
        return padded.view(operand.dtype).T
    padded[:, :k] = operand.view(torch.uint8)
    return padded.view(operand.dtype)


def gemm_a8w8_mxfp8_opus(
    XQ: Tensor,
    WQ: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    dtype: torch.dtype = torch.bfloat16,
    *,
    kernelId: int | None = None,
    out: Tensor | None = None,
) -> Tensor:
    """gfx950 MXFP8 1x32 GEMM for plain NT / NN / TT / TN operand storage.

    Computes Y[M, N] = XQ[M, K] @ WQ[N, K]^T on logical shapes; the storage is
    read from the strides (see ``_mxfp8_direction``), so pass the transposed
    view of a [K, M] A or of a [K, N] B, never a copy. WQ is not shuffled
    (the shuffle_weight NT input belongs to ``gemm_a8w8_mxfp8_bpreshuffle``).
    x_scale [M, K/32] and w_scale [N, K/32] are row-major E8M0 in every layout.
    Output is BF16.

    Any M (A stored [K, M]: M % 16), N % 16 and any K. A K that is a multiple
    of 32 runs on the operands as they are. Any other K has ceil(K/32) scale
    groups, the last one over its K % 32 elements; XQ and WQ are then copied
    into zero-padded operands of the next multiple of 32, which costs their
    size per call -- pad the weight once and pass that K to avoid it.

    ``kernelId`` runs that kid (split-K kids pick their split); otherwise one
    is picked by shape.
    """
    if XQ.dim() != 2 or WQ.dim() != 2:
        raise ValueError("gemm_a8w8_mxfp8_opus expects 2D XQ / WQ")
    if dtype != torch.bfloat16:
        raise ValueError("gemm_a8w8_mxfp8_opus writes BF16 only")
    direction = _mxfp8_direction(XQ, WQ)
    if direction is None:
        raise ValueError(
            f"gemm_a8w8_mxfp8_opus needs dense [M, K] / [K, M] XQ and [N, K] / [K, N] "
            f"WQ storage; got XQ strides {tuple(XQ.stride())}, WQ strides {tuple(WQ.stride())}"
        )
    M, K = map(int, XQ.shape)
    N = int(WQ.shape[0])
    if int(WQ.shape[1]) != K:
        raise ValueError(f"XQ is [{M}, {K}] but WQ is {tuple(WQ.shape)}")
    groups = -(-K // 32)
    if tuple(x_scale.shape) != (M, groups) or tuple(w_scale.shape) != (N, groups):
        raise ValueError(
            f"x_scale / w_scale must be [{M}, {groups}] / [{N}, {groups}] for K={K}, "
            f"got {tuple(x_scale.shape)} / {tuple(w_scale.shape)}"
        )
    if out is None:
        out = torch.empty((M, N), dtype=dtype, device=XQ.device)
    elif tuple(out.shape) != (M, N) or out.dtype != dtype:
        raise ValueError(
            f"out must be {dtype} [{M}, {N}], got {out.dtype} {tuple(out.shape)}"
        )
    if K % 32:
        # The kernels take whole 1x32 groups: zero the K % 32 tail of the last one.
        K = groups * 32
        XQ = _a8w8_mxfp8_pad_k(XQ, direction[0] == "T", K)
        WQ = _a8w8_mxfp8_pad_k(WQ, direction[1] == "N", K)
    if kernelId is None:
        arch, cu_num = _device_arch_and_cu(XQ.device)
        kernelId = _select_a8w8_mxfp8_kid(direction, M, N, K, arch, cu_num)
        if kernelId is None:
            raise ValueError(
                f"no {direction} MXFP8 1x32 kernel supports M={M}, N={N}, K={K} on "
                f"{arch} (N % 16 must be 0; A stored [K, M] needs M % 16)"
            )
    return _launch_a8w8_mxfp8_gemm(XQ, WQ, out, x_scale, w_scale, kid=int(kernelId))


def _validate_a8w8_mxscale_bmm_tensors(
    XQ: Tensor,
    WQ: Tensor,
    Y: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    kid: int,
) -> None:
    entry = "opus_gemm_a8w8_mxscale_bmm_launch"
    tensors = (XQ, WQ, Y, x_scale, w_scale)
    if any(tensor.dim() != 3 for tensor in tensors):
        raise ValueError(f"{entry}: all inputs and Y must be 3D")
    # This path validates inputs before Python allocates split-K workspace;
    # ordinary launches leave the same-device contract to the checked C++ ABI.
    device = XQ.device
    if any(tensor.device != device for tensor in tensors[1:]):
        devices = {tensor.device for tensor in tensors}
        raise ValueError(
            f"{entry}: all tensors must be on one device; got "
            f"{sorted(map(str, devices))}"
        )
    if XQ.dtype not in _FP8_DTYPES or WQ.dtype != XQ.dtype:
        raise ValueError(f"{entry}: XQ and WQ must have the same FP8 dtype")
    if Y.dtype not in (torch.bfloat16, torch.float32):
        raise ValueError(f"{entry}: Y must be BF16 or FP32")
    if x_scale.dtype not in _E8M0_DTYPES or w_scale.dtype not in _E8M0_DTYPES:
        raise ValueError(
            f"{entry}: x_scale and w_scale must contain one-byte E8M0 values"
        )

    M, batch, K = map(int, XQ.shape)
    w_batch, N, w_K = map(int, WQ.shape)
    if min(M, batch, N, K) <= 0:
        raise ValueError(f"{entry}: M, batch, N and K must be positive")
    # The kid's own quantisation block, read off the same instance table the
    # generated kid->group table in C++ is built from. It used to be a literal
    # 128 here and in opus_bmm_a8w8_common_checks, and fixing only the C++ copy
    # left this one rejecting every correct GROUP_K=32 launch that reached it --
    # the same rule living in two places and drifting.
    arch, cu_num = _device_arch_and_cu(XQ.device)
    inst = get_registry(_A8W8_MXSCALE_BMM_FAMILY, arch, cu_num)[int(kid)]
    group_n, group = inst.GROUP_N, inst.GROUP_K
    if N % group_n or K % group:
        raise ValueError(
            f"{entry}: N and K must be multiples of {group_n} and {group}; got N={N}, K={K}"
        )
    if (w_batch, w_K) != (batch, K):
        raise ValueError(
            f"{entry}: WQ must have shape [{batch},N,{K}], got {tuple(WQ.shape)}"
        )
    if tuple(Y.shape) != (M, batch, N):
        raise ValueError(
            f"{entry}: Y must have shape {(M, batch, N)}, got {tuple(Y.shape)}"
        )
    expected_x_scale = (M, batch, K // group)
    expected_w_scale = (batch, N // group_n, K // group)
    if tuple(x_scale.shape) != expected_x_scale:
        raise ValueError(
            f"{entry}: x_scale must have shape {expected_x_scale}, "
            f"got {tuple(x_scale.shape)}"
        )
    if tuple(w_scale.shape) != expected_w_scale:
        raise ValueError(
            f"{entry}: w_scale must have shape {expected_w_scale}, "
            f"got {tuple(w_scale.shape)}"
        )
    if any(tensor.stride(-1) != 1 for tensor in (XQ, WQ, Y, x_scale, w_scale)):
        raise ValueError(f"{entry}: every tensor must be contiguous in its last axis")


def _launch_a8w8_mxscale_bmm(
    XQ: Tensor,
    WQ: Tensor,
    Y: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    *,
    kid: int,
    split_k: int,
    workspace: Tensor | None,
    route_arch: str | None = None,
    instance: object | None = None,
) -> Tensor:
    """Launch batch-first MXFP8 BMM through the shared physical launcher.

    Public tensors use ``[B,M,K]``, ``[B,N,K]`` and ``[B,M,N]``.  The raw
    kernels retain their established M-major ``[M,B,*]`` activation/output
    ABI; transpose views bridge the two contracts without copying storage.
    """
    if instance is None and any(
        tensor.dim() != 3 for tensor in (XQ, WQ, Y, x_scale, w_scale)
    ):
        raise ValueError(
            "opus_bmm A8W8 mxscale expects batch-first 3D " "XQ/WQ/Y/x_scale/w_scale"
        )
    launch_x = XQ.transpose(0, 1)
    launch_y = Y.transpose(0, 1)
    launch_x_scale = x_scale.transpose(0, 1)

    if instance is not None and workspace is None and split_k <= 1:
        # The checked C++ entry owns the dynamic tensor contract.  Public
        # routing already validated the immutable family/kid contract, so the
        # common split-one path need not repeat the Python registry/planner.
        _launch_a8w8_backend(
            launch_x,
            WQ,
            launch_y,
            launch_x_scale,
            w_scale,
            None,
            _A8W8_MXSCALE_BMM_FAMILY,
            kid,
            max(1, split_k),
        )
        return Y

    M, batch, K = map(int, launch_x.shape)
    N = int(WQ.shape[1])
    device_arch, cu_num = _device_arch_and_cu(XQ.device)
    arch = route_arch or device_arch
    plan = _get_cached_a8w8_mxscale_bmm_plan(
        arch,
        int(kid),
        Y.dtype,
        M,
        batch,
        N,
        K,
        int(split_k),
        cu_num=cu_num,
    )

    launch_workspace = workspace
    workspace_spec = plan.workspace_spec
    if workspace_spec is not None:
        # Workspace sizing reads Tensor dimensions before entering C++. Keep
        # the full Python contract here so malformed metadata cannot drive an
        # allocation. The split-one hot path above remains C++-checked only.
        _validate_a8w8_mxscale_bmm_tensors(
            launch_x,
            WQ,
            launch_y,
            launch_x_scale,
            w_scale,
            int(kid),
        )
        required_numel = workspace_spec.shape[0]
        if launch_workspace is None:
            launch_workspace = torch.empty(
                workspace_spec.shape,
                dtype=workspace_spec.dtype,
                device=XQ.device,
            )
        else:
            if not isinstance(launch_workspace, Tensor):
                raise TypeError(
                    "opus_gemm_a8w8_mxscale_bmm_launch: workspace must be a Tensor"
                )
            if launch_workspace.device != XQ.device:
                raise ValueError(
                    "opus_gemm_a8w8_mxscale_bmm_launch: workspace must be on "
                    f"{XQ.device}, got {launch_workspace.device}"
                )
            if launch_workspace.dtype != workspace_spec.dtype:
                raise ValueError(
                    "opus_gemm_a8w8_mxscale_bmm_launch: workspace must be FP32"
                )
            if not launch_workspace.is_contiguous():
                raise ValueError(
                    "opus_gemm_a8w8_mxscale_bmm_launch: workspace must be contiguous"
                )
            if launch_workspace.numel() < required_numel:
                raise ValueError(
                    "opus_gemm_a8w8_mxscale_bmm_launch: workspace capacity is "
                    f"{launch_workspace.numel()}, but {required_numel} elements "
                    "are required"
                )
    elif launch_workspace is not None:
        raise ValueError(
            f"OPUS BMM kid {plan.resolved_kid} with split_k={split_k} "
            "does not use workspace"
        )

    _launch_a8w8_backend(
        launch_x,
        WQ,
        launch_y,
        launch_x_scale,
        w_scale,
        launch_workspace,
        _A8W8_MXSCALE_BMM_FAMILY,
        plan.resolved_kid,
        plan.abi_split_k,
    )
    return Y


def bmm_a8w8_mxscale_opus(
    x: Tensor,
    wo_a: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    out: Tensor | None = None,
    dtype: torch.dtype = torch.bfloat16,
    kernelId: int | None = None,
    splitK: int | None = None,
    *,
    group_size: int = 128,
) -> Tensor:
    """Run OPUS MXFP8 BMM with a row-major weight.

    ``x`` is [M, G, K], ``wo_a`` is [G, N, K], ``x_scale`` is
    [M, G, K/group_size], and ``w_scale`` is
    [G, N/group_size, K/group_size]. Scales are ordinary E8M0 blocks;
    ``group_size`` is 32 or 128. Returns ``out`` [M, G, N], allocating it
    with ``dtype`` when omitted.

    ``kernelId=None`` selects the shape heuristic. An incompatible tuned id
    falls back to a compatible kernel and drops its splitK. ``splitK=None``,
    0 and 1 all mean no split. Use :func:`bmm_a8w8_mxscale_opus_bpreshuffle`
    for a weight already in the (16, 16) MFMA-fragment layout.
    """
    from .policy import (
        _heuristic_mxscale_bmm_kid,
        _validate_mxscale_bmm_group_size,
        mxscale_bmm_kid_block,
        mxscale_bmm_kid_runs_m,
        mxscale_bmm_kid_takes_b_layout,
        mxscale_bmm_kid_takes_plain_scales,
    )

    _validate_mxscale_bmm_group_size(group_size)
    block = f"{group_size}x{group_size}"
    gfx, cu_num = _device_arch_and_cu(x.device)
    scope = {"gfx": gfx, "cu_num": cu_num}
    m, g, k = map(int, x.shape)
    n = int(wo_a.shape[1])
    Y = out if out is not None else torch.empty((m, g, n), dtype=dtype, device=x.device)

    # A stale tuned row may name a different layout, scale block or M alignment.
    # Discard its splitK along with the id before choosing a row-major fallback.
    if kernelId is not None and not (
        mxscale_bmm_kid_takes_b_layout(int(kernelId), False, **scope)
        and mxscale_bmm_kid_takes_plain_scales(int(kernelId), **scope)
        and mxscale_bmm_kid_block(int(kernelId), **scope) == block
        and mxscale_bmm_kid_runs_m(int(kernelId), m, **scope)
    ):
        kernelId = splitK = None
    if kernelId is None:
        kernelId = _heuristic_mxscale_bmm_kid(
            g, m, n, k, group_size=group_size, **scope
        )

    return _launch_bmm_a8w8_mxscale_opus(
        x, wo_a, x_scale, w_scale, Y, kid=int(kernelId), split_k=splitK
    )


def bmm_a8w8_mxscale_opus_bpreshuffle(
    x: Tensor,
    wo_a: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    out: Tensor | None = None,
    dtype: torch.dtype = torch.bfloat16,
    kernelId: int | None = None,
    splitK: int | None = None,
    *,
    group_size: int = 128,
    group_n: int | None = None,
) -> Tensor:
    """Run OPUS MXFP8 BMM with an already preshuffled weight.

    ``wo_a`` [G, N, K] must come from ``shuffle_weight(w, layout=(16, 16))``;
    this entry does not shuffle it. ``x`` is [M, G, K], ``x_scale`` is
    [M, G, K/group_size], and ``w_scale`` is
    [G, N/group_n, K/group_size]. Scales keep their ordinary E8M0 layout.
    ``group_n`` defaults to ``group_size``; supported weight-scale blocks are
    32x32, 128x128 and 1x32 (``group_n=1, group_size=32``).

    Returns ``out`` [M, G, N], allocating it with ``dtype`` when omitted.
    ``kernelId=None`` selects a compatible preshuffled kernel. An incompatible
    tuned id falls back to one and drops its splitK. ``splitK=None``, 0 and 1
    all mean no split.
    """
    from .policy import (
        _heuristic_mxscale_bmm_bpreshuffle_kid,
        mxscale_bmm_groups_of_block,
        mxscale_bmm_kid_block,
        mxscale_bmm_kid_runs_m,
        mxscale_bmm_kid_takes_b_layout,
        mxscale_bmm_kid_takes_plain_scales,
    )

    group_n = group_size if group_n is None else group_n
    block = f"{group_n}x{group_size}"
    mxscale_bmm_groups_of_block(block)
    gfx, cu_num = _device_arch_and_cu(x.device)
    scope = {"gfx": gfx, "cu_num": cu_num}
    m, g, k = map(int, x.shape)
    n = int(wo_a.shape[1])
    Y = out if out is not None else torch.empty((m, g, n), dtype=dtype, device=x.device)

    # The preshuffled entry accepts only matching B and ordinary scale layouts.
    # A padded-M tuned row can also require an alignment the real M lacks.
    if kernelId is not None and not (
        mxscale_bmm_kid_takes_b_layout(int(kernelId), True, **scope)
        and mxscale_bmm_kid_takes_plain_scales(int(kernelId), **scope)
        and mxscale_bmm_kid_block(int(kernelId), **scope) == block
        and mxscale_bmm_kid_runs_m(int(kernelId), m, **scope)
    ):
        kernelId = splitK = None
    if kernelId is None:
        kernelId = _heuristic_mxscale_bmm_bpreshuffle_kid(
            g,
            m,
            n,
            k,
            group_size=group_size,
            group_n=group_n,
            output_dtype=Y.dtype,
            **scope,
        )
        if kernelId is None:
            raise ValueError(
                f"no preshuffled-B kid runs (g={g}, m={m}, n={n}, k={k}, "
                f"group_size={group_size}): the tuned row is absent or "
                f"unusable and no fallback tile divides this N and K "
                f"for weight scale block {block}."
            )

    return _launch_bmm_a8w8_mxscale_opus(
        x, wo_a, x_scale, w_scale, Y, kid=int(kernelId), split_k=splitK
    )


def _launch_bmm_a8w8_mxscale_opus(
    x: Tensor,
    wo_a: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    Y: Tensor,
    *,
    kid: int,
    split_k: int | None,
) -> Tensor:
    """Allocate split-K workspace and launch the entry's already selected kid."""
    split_k = int(split_k if split_k is not None else 1)
    if split_k == 0:  # Public no-split convention; launchers require one.
        split_k = 1
    _opus_gemm_a8w8_mxscale_bmm_launch_raw(
        x,
        wo_a,
        Y,
        x_scale,
        w_scale,
        workspace=_mxscale_bmm_workspace(x, wo_a, Y, kid, split_k),
        kid=kid,
        split_k=split_k,
    )
    return Y


def _mxscale_bmm_workspace(
    x: Tensor, wo_a: Tensor, Y: Tensor, kid: int, split_k: int
) -> Tensor | None:
    """The FP32 split-K workspace kid needs on mmajor ``x`` [M, G, K], or None."""
    if split_k <= 1:
        return None
    m, g, k = map(int, x.shape)
    arch, cu_num = _device_arch_and_cu(x.device)
    spec = _get_cached_a8w8_mxscale_bmm_plan(
        arch, kid, Y.dtype, m, g, int(wo_a.shape[1]), k, split_k, cu_num
    ).workspace_spec
    if spec is None:
        return None
    return torch.empty(spec.shape, dtype=spec.dtype, device=x.device)


__all__ = [
    "bmm_a8w8_mxscale_opus",
    "bmm_a8w8_mxscale_opus_bpreshuffle",
    "gemm_a8w8_blockscale_opus",
    "opus_gemm_a8w8_blockscale_bpreshuffle_tune",
]
