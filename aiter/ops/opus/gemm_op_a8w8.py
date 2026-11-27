# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Private OPUS A8W8 exact-kid launch APIs."""

# Keep annotations eager in this binding module.  ``torch_compile_guard`` uses
# the first parameter's concrete ``torch.Tensor`` identity to decide whether a
# custom op already has a Tensor dispatch key.  Postponed string annotations
# make it add a dummy CUDA Tensor to every raw launch, which is observable host
# overhead on short BMM kernels.

import re
from functools import lru_cache

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

    First letter A, second B. A "N" is stored [M, K] (strides (>= K, 1)), A "T"
    [K, M] (strides (1, M)); B "T" is stored [N, K] (strides (>= K, 1)), B "N"
    [K, N] (strides (1, N)). Returns "NN", "TT", "TN", "NT" or None. The rows of
    a K-contiguous operand may be more than K apart: a [M, K] view of a wider buffer.
    """
    (M, K), N = XQ.shape, WQ.shape[0]

    def k_contiguous(operand):
        return operand.stride(1) == 1 and operand.stride(0) >= K

    if k_contiguous(XQ):
        a = "N"
    elif XQ.stride() == (1, M):
        a = "T"
    else:
        return None
    if k_contiguous(WQ):
        b = "T"
    elif WQ.stride() == (1, N):
        b = "N"
    else:
        return None
    return a + b


# Scalar-only key, as _get_cached_a16w16_launch_plan: the pick filters the registry
# (one CO image stat per kid), which costs more than a small GEMM if redone per call.
@lru_cache(maxsize=256)
def _select_mxfp8_direction_kid(
    direction: str, M: int, N: int, K: int, arch: str, cu_num: int
) -> int | None:
    """Pick a supported NT / NN / TT / TN kid for the shape.

    The rule is fitted to a sweep of every kid over 264 shapes per direction on a
    256-CU gfx950 (M 16 to 32768, N 128 to 8192, K 512 to 7168): there the pick is
    0.5% slower than the fastest kid on average and within 2% of it on 95% of the
    shapes. On 60 shapes per direction outside that grid (NN, TT) it is 1.6% slower
    on average, within 2% on 78% and at most 17% slower.

    * 256x128 tiles while they do not outnumber the CUs (or N % 256 != 0), else
      256x256: more, smaller tiles only pay while there are CUs left to fill.
    * GroupM for a tall, narrow grid (at least 4 M tiles of at most 8 N tiles):
      8 on 256x128, 4 on 256x256 (8 from 128 M tiles of at most 4 N tiles).
    * PadM only when M is not a multiple of 256 (it has no GroupM variant).
    """
    m_tiles = -(-M // 256)
    want_bn = 256 if N % 256 == 0 and m_tiles * (N // 128) > cu_num else 128
    n_tiles = N // want_bn
    want_group_m = 0
    if m_tiles >= 4 and n_tiles <= 8:
        wide_tall = want_bn == 256 and not (m_tiles >= 128 and n_tiles <= 4)
        want_group_m = 4 if wide_tall else 8
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


def _mxfp8_run_shape(direction: str, M: int, N: int, K: int) -> tuple[int, int, int]:
    """The (M, N, K) the kernel runs for a logical one.

    The kernels take M % 16 == 0 (PadM; x_scale and a [K, M] A are read 16 rows at
    a time), N % 128 == 0 (one w_scale group; C is written in whole 128-column
    quadrants) and any K, from 16 with a K-contiguous operand (its partial K tile
    is rewritten from the last 16 bytes of each row).
    """
    run_k = 16 if K < 16 and direction != "TN" else K
    return -(-M // 16) * 16, -(-N // 128) * 128, run_k


def _mxfp8_padded_operand(
    operand: Tensor, k_contiguous: bool, rows: int, K: int
) -> Tensor:
    """Logical [R, K0] operand zero-padded to [rows, K], in the same storage.

    Zero rows and zero K only add zero products. A K-contiguous copy gets rows a
    multiple of 128 bytes apart, the layout the kernel reads fastest.
    """
    R, K0 = operand.shape
    if k_contiguous:
        padded = operand.new_zeros((rows, -(-K // 128) * 128))
        padded[:R, :K0] = operand
        return padded[:, :K]
    padded = operand.new_zeros((K, rows))
    padded[:K0, :R] = operand.t()
    return padded.t()


def _mxfp8_padded_x_scale(x_scale: Tensor, M: int, rows: int) -> Tensor:
    """x_scale [M, groups] (column-major bytes) padded to [rows, groups], as a strides
    (1, rows) view. The scales of the padded rows multiply zeros."""
    if x_scale.dim() != 2 or x_scale.shape[0] != M:
        raise ValueError(
            f"x_scale must be [{M}, ceil(K / 128)], got {tuple(x_scale.shape)}"
        )
    groups = x_scale.shape[1]
    if x_scale.stride() == (1, M):
        columns = x_scale.t()
    elif x_scale.is_contiguous():
        columns = x_scale.reshape(groups, M)
    else:
        raise ValueError(
            "x_scale must hold column-major bytes: contiguous or a strides (1, M) "
            f"view, got strides {tuple(x_scale.stride())}"
        )
    padded = x_scale.new_zeros((groups, rows))
    padded[:, :M] = columns
    return padded.t()


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
    [K/128, N/128] for B "N"). Output is BF16. XQ, WQ and the output may each be
    4 GiB or larger.

    M, N and K are any positive sizes. The scales have ceil(K / 128) K groups, the
    last one over the K % 128 elements left, and w_scale ceil(N / 128) N groups.
    The kernels take M % 16 == 0, N % 128 == 0 and, with a [M, K] / [N, K]
    operand, K >= 16: otherwise XQ and x_scale (M), WQ (N) or both operands (K) are
    first copied into zero-padded ones and the [M, N] result is cut from the padded
    output (a copy when N is padded or ``out`` is given). Any other K is read in
    place.
    A [M, K] / [N, K] operand may be a view of a wider buffer (rows >= K apart;
    what follows a row's K bytes is not read into the result). The kernel copies
    128 bytes of a row per K tile and is fastest when every row starts on a
    128-byte boundary (rows a multiple of 128 apart); the rows of a K % 128 != 0
    operand packed [M, K] do not, which costs up to 2x on a small GEMM.

    ``kernelId`` runs that a8w8_blockscale_mxfp8 kid (0-39: NT 0-9, NN 10-19,
    TT 20-29, TN 30-39); otherwise one is picked by shape.
    """
    # 1. Check the request; read the storage (NT / NN / TT / TN) off the strides.
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
    if min(M, N, K) <= 0:
        raise ValueError(
            f"gemm_a8w8_blockscale_opus needs M, N, K > 0, got {M}, {N}, {K}"
        )
    if out is not None and (tuple(out.shape) != (M, N) or out.dtype != dtype):
        raise ValueError(
            f"out must be {dtype} [{M}, {N}], got {out.dtype} {tuple(out.shape)}"
        )

    # 2. Pad the inputs to the shape the kernel runs (a no-op for a shape it takes).
    run_m, run_n, run_k = _mxfp8_run_shape(direction, M, N, K)
    if (run_m, run_k) != (M, K):
        XQ = _mxfp8_padded_operand(XQ, direction[0] == "N", run_m, run_k)
    if run_m != M:
        x_scale = _mxfp8_padded_x_scale(x_scale, M, run_m)
    if (run_n, run_k) != (N, K):
        WQ = _mxfp8_padded_operand(WQ, direction[1] == "T", run_n, run_k)
    padded_out = (run_m, run_n) != (M, N)
    if padded_out or out is None:
        Y = torch.empty((run_m, run_n), dtype=dtype, device=XQ.device)
    else:
        Y = out

    # 3. Pick the kid for the run shape, unless the caller named one.
    if kernelId is None:
        arch, cu_num = _device_arch_and_cu(XQ.device)
        kernelId = _select_mxfp8_direction_kid(
            direction, run_m, run_n, run_k, arch, cu_num
        )
        if kernelId is None:
            raise ValueError(
                f"no {direction} MXFP8 kernel supports M={M}, N={N}, K={K} on {arch}: "
                "one tile's view of A / B (256 rows of an operand stored [M, K] / "
                "[N, K], 128 rows of one stored [K, M] / [K, N]) and x_scale must "
                "each stay below 2**31 bytes"
            )

    # 4. Launch; the generated launcher checks the kid against the tensors.
    _launch_a8w8_blockscale_mxfp8_gemm(XQ, WQ, x_scale, w_scale, Y, kid=int(kernelId))

    # 5. Cut [M, N] from a padded output.
    if not padded_out:
        return Y
    if out is not None:
        out.copy_(Y[:M, :N])
        return out
    return Y[:M, :N].contiguous()


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
