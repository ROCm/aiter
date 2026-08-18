# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Private OPUS A8W8 exact-kid launch APIs."""

# Keep annotations eager in this binding module.  ``torch_compile_guard`` uses
# the first parameter's concrete ``torch.Tensor`` identity to decide whether a
# custom op already has a Tensor dispatch key.  Postponed string annotations
# make it add a dummy CUDA Tensor to every raw launch, which is observable host
# overhead on short BMM kernels.

import functools

import torch
from torch import Tensor

from csrc.opus_gemm.opus_gemm_common import get_kernel_instance

from ...jit.core import compile_ops
from ._arch import _device_arch

_A8W8_FAMILY = "a8w8"
_A8W8_BLOCKSCALE_FAMILY = "a8w8_blockscale"
_A8W8_BPRESHUFFLE_FAMILY = "a8w8_blockscale_bpreshuffle"
_A8W8_MXSCALE_BMM_FAMILY = "a8w8_mxscale_bmm"
_A8W8_MXSCALE_BMM_TAGS = frozenset(
    {
        "a8w8_mxscale_bmm_flatmm_splitk",
        "a8w8_mxscale_bmm_fused",
        "a8w8_mxscale_bmm_minterleave",
        "a8w8_mxscale_bmm_mouter",
        "a8w8_mxscale_bmm_mouter_tunable",
        "a8w8_mxscale_bmm_pipeline",
        "a8w8_mxscale_bmm_wave8n2",
        "a8w8_mxscale_bmm_wave4m2_selfload",
    }
)
_A8W8_MXSCALE_BMM_WORKSPACE_TAGS = frozenset(
    {
        "a8w8_mxscale_bmm_flatmm_splitk",
        "a8w8_mxscale_bmm_fused",
    }
)
_A8W8_FAMILY_BY_TAG = {
    "a8w8": _A8W8_FAMILY,
    "a8w8_scale": _A8W8_BLOCKSCALE_FAMILY,
    "a8w8_blockscale_bpreshuffle_singlebuf": _A8W8_BPRESHUFFLE_FAMILY,
    **{tag: _A8W8_MXSCALE_BMM_FAMILY for tag in _A8W8_MXSCALE_BMM_TAGS},
}
_A8W8_FAMILY_LAYOUT = {
    _A8W8_FAMILY: "plain",
    _A8W8_BLOCKSCALE_FAMILY: "plain",
    _A8W8_BPRESHUFFLE_FAMILY: "bpreshuffle",
    _A8W8_MXSCALE_BMM_FAMILY: "mxscale_bmm",
}
_FP8_DTYPES = frozenset(
    dtype
    for dtype in (
        getattr(torch, "float8_e4m3fnuz", None),
        getattr(torch, "float8_e4m3fn", None),
    )
    if dtype is not None
)
_MISSING_TENSOR = object()
_E8M0_DTYPES = frozenset(
    dtype
    for dtype in (
        torch.uint8,
        getattr(torch, "float8_e8m0fnu", None),
    )
    if dtype is not None
)


def _validate_a8w8_public_contract(
    *,
    kernel_tag: str,
    kid: int,
    input_dtype: torch.dtype,
    weight_dtype: torch.dtype,
    output_dtype: torch.dtype,
    layout: str,
    has_x_scale: bool,
    has_w_scale: bool,
    has_bias: bool,
    has_workspace: bool,
    split_k: int,
) -> str:
    """Validate A8W8-only options shared by both public operation routers."""
    try:
        family = _A8W8_FAMILY_BY_TAG[kernel_tag]
    except KeyError as exc:
        raise ValueError(
            f"OPUS kid {kid} has unsupported registry tag {kernel_tag!r}"
        ) from exc

    if input_dtype != weight_dtype:
        raise ValueError(
            f"OPUS requires matching XQ/WQ dtypes; got "
            f"{input_dtype}/{weight_dtype}"
        )
    if input_dtype not in _FP8_DTYPES:
        raise ValueError(
            f"OPUS kid {kid} requires FP8 XQ/WQ; got {input_dtype}"
        )

    if family == _A8W8_MXSCALE_BMM_FAMILY:
        if output_dtype not in (torch.bfloat16, torch.float32):
            raise ValueError(
                f"OPUS kid {kid} requires BF16 or FP32 Y; got {output_dtype}"
            )
    else:
        expected_output_dtype = (
            torch.bfloat16
            if family == _A8W8_BPRESHUFFLE_FAMILY
            else torch.float32
        )
        if output_dtype != expected_output_dtype:
            raise ValueError(
                f"OPUS kid {kid} does not support Y.dtype={output_dtype}; "
                f"expected {expected_output_dtype}"
            )

    expected_layout = _A8W8_FAMILY_LAYOUT[family]
    if layout != expected_layout:
        raise ValueError(
            f"OPUS kid {kid} belongs to family {family} and requires "
            f"layout={expected_layout!r}; got {layout!r}"
        )
    if has_x_scale != has_w_scale:
        raise ValueError("OPUS requires x_scale and w_scale together")
    if has_bias:
        raise ValueError(f"OPUS family {family} does not support bias")
    if family == _A8W8_MXSCALE_BMM_FAMILY:
        if not has_x_scale:
            raise ValueError("OPUS a8w8_mxscale_bmm requires x_scale and w_scale")
        if split_k == 0 and has_workspace:
            raise ValueError(
                "OPUS a8w8_mxscale_bmm split_k=0 does not use workspace"
            )
        return family
    if has_workspace:
        raise ValueError(f"OPUS family {family} does not use workspace")
    if split_k != 0:
        raise ValueError(f"OPUS family {family} does not accept split_k")
    if family == _A8W8_FAMILY:
        if has_x_scale:
            raise ValueError("OPUS a8w8 kid does not accept scales")
    elif not has_x_scale:
        raise ValueError(f"OPUS family {family} requires x_scale and w_scale")
    return family


def _check_same_device(
    entry: str,
    XQ: Tensor,
    WQ: Tensor,
    Y: Tensor,
    x_scale: object = _MISSING_TENSOR,
    w_scale: object = _MISSING_TENSOR,
) -> None:
    """Require Tensor inputs on one device."""
    has_x_scale = x_scale is not _MISSING_TENSOR
    has_w_scale = w_scale is not _MISSING_TENSOR
    if not (
        isinstance(XQ, Tensor)
        and isinstance(WQ, Tensor)
        and isinstance(Y, Tensor)
        and (not has_x_scale or isinstance(x_scale, Tensor))
        and (not has_w_scale or isinstance(w_scale, Tensor))
    ):
        named_tensors = [("XQ", XQ), ("WQ", WQ), ("Y", Y)]
        if has_x_scale:
            named_tensors.append(("x_scale", x_scale))
        if has_w_scale:
            named_tensors.append(("w_scale", w_scale))
        invalid = [
            name for name, tensor in named_tensors if not isinstance(tensor, Tensor)
        ]
        raise TypeError(
            f"{entry}: {', '.join(invalid)} must be Tensor objects"
        )

    device = XQ.device
    same_device = WQ.device == device and Y.device == device
    if has_x_scale:
        same_device = same_device and x_scale.device == device
    if has_w_scale:
        same_device = same_device and w_scale.device == device
    if not same_device:
        tensors = [XQ, WQ, Y]
        if has_x_scale:
            tensors.append(x_scale)
        if has_w_scale:
            tensors.append(w_scale)
        devices = {tensor.device for tensor in tensors}
        raise ValueError(
            f"{entry}: all tensors must be on one device; got "
            f"{sorted(map(str, devices))}"
        )


@functools.lru_cache(maxsize=None)
def _require_registered_kid_cached(
    arch: str, family: str, resolved: int, output_dtype: torch.dtype
) -> int:
    """Validate one kernel registration and cache successful lookups."""
    instance = get_kernel_instance(arch, family, resolved, output_dtype)
    if instance is None:
        raise ValueError(
            "no registered OPUS kernel for "
            f"(arch={arch!r}, family={family!r}, kid={resolved}, "
            f"Y.dtype={output_dtype})"
        )
    return resolved


def _require_registered_kid(
    *, arch: str, family: str, kid: object, output_dtype: torch.dtype
) -> int:
    """Normalize a kid and require a matching kernel registration."""
    try:
        resolved = int(kid)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"OPUS {family} kid must be an integer, got {kid!r}") from exc
    return _require_registered_kid_cached(arch, family, resolved, output_dtype)


# ---- Private exact-kid pybind bindings -----------------------------------


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
) -> Tensor:
    return Y


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
) -> Tensor: ...


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


# ---- Exact launchers and logical GEMM/BMM adapters -----------------------


def _require_logical_rank(
    entry: str,
    rank: int,
    **tensors: Tensor,
) -> None:
    invalid = [name for name, tensor in tensors.items() if tensor.dim() != rank]
    if invalid:
        operation = "opus_gemm" if rank == 2 else "opus_bmm"
        other = "opus_bmm" if rank == 2 else "opus_gemm"
        raise ValueError(
            f"{entry}: {operation} expects {rank}D "
            f"{', '.join(invalid)}; use {other} for "
            f"{'batch-first 3D' if rank == 2 else 'logical 2D'} tensors"
        )


def _launch_a8w8_exact(
    XQ: Tensor,
    WQ: Tensor,
    Y: Tensor,
    *,
    kid: int,
    route_arch: str | None = None,
    instance: object | None = None,
) -> Tensor:
    """Launch the shared physical 3D no-scale A8W8 ABI."""
    resolved = kid
    if instance is None:
        entry = "opus_gemm_a8w8_launch"
        _check_same_device(entry, XQ, WQ, Y)
        arch = route_arch or _device_arch(XQ.device)
        resolved = _require_registered_kid(
            arch=arch, family=_A8W8_FAMILY, kid=kid, output_dtype=Y.dtype
        )
    _opus_gemm_a8w8_launch_raw(XQ, WQ, Y, resolved)
    return Y


def _launch_a8w8_gemm(
    XQ: Tensor,
    WQ: Tensor,
    Y: Tensor,
    *,
    kid: int,
    route_arch: str | None = None,
    instance: object | None = None,
) -> Tensor:
    """Launch logical 2D ``[M,K] x [N,K] -> [M,N]`` A8W8 GEMM."""
    if instance is None:
        _require_logical_rank("a8w8", 2, XQ=XQ, WQ=WQ, Y=Y)
    _launch_a8w8_exact(
        XQ.unsqueeze(0),
        WQ.unsqueeze(0),
        Y.unsqueeze(0),
        kid=kid,
        route_arch=route_arch,
        instance=instance,
    )
    return Y


def _launch_a8w8_bmm(
    XQ: Tensor,
    WQ: Tensor,
    Y: Tensor,
    *,
    kid: int,
    route_arch: str | None = None,
    instance: object | None = None,
) -> Tensor:
    """Launch batch-first ``[B,M,K] x [B,N,K] -> [B,M,N]`` A8W8 BMM.

    Inputs are contiguous FP8 ``[B,M,K]`` and ``[B,N,K]``; output is contiguous
    FP32 ``[B,M,N]``. The generated launcher checks its K-loop limits.
    """
    if instance is None:
        _require_logical_rank("a8w8", 3, XQ=XQ, WQ=WQ, Y=Y)
    return _launch_a8w8_exact(
        XQ,
        WQ,
        Y,
        kid=kid,
        route_arch=route_arch,
        instance=instance,
    )


def _launch_a8w8_blockscale_exact(
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
    """Launch the shared physical 3D blockscale A8W8 ABI."""
    resolved = kid
    if instance is None:
        entry = "opus_gemm_a8w8_blockscale_launch"
        _check_same_device(entry, XQ, WQ, Y, x_scale, w_scale)
        arch = route_arch or _device_arch(XQ.device)
        resolved = _require_registered_kid(
            arch=arch,
            family=_A8W8_BLOCKSCALE_FAMILY,
            kid=kid,
            output_dtype=Y.dtype,
        )
    _opus_gemm_a8w8_blockscale_launch_raw(
        XQ, WQ, Y, x_scale, w_scale, resolved
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
    if instance is None:
        _require_logical_rank(
            "a8w8_blockscale",
            2,
            XQ=XQ,
            WQ=WQ,
            Y=Y,
            x_scale=x_scale,
            w_scale=w_scale,
        )
    _launch_a8w8_blockscale_exact(
        XQ.unsqueeze(0),
        WQ.unsqueeze(0),
        Y.unsqueeze(0),
        x_scale,
        w_scale,
        kid=kid,
        route_arch=route_arch,
        instance=instance,
    )
    return Y


def _launch_a8w8_blockscale_bmm(
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
    """Launch batch-first 3D blockscale A8W8 BMM.

    Both contiguous FP32 scales are required. Their shapes are
    ``[B,M,K/128]`` and ``[B,N/128,K/128]``.
    """
    if instance is None:
        _require_logical_rank(
            "a8w8_blockscale",
            3,
            XQ=XQ,
            WQ=WQ,
            Y=Y,
            x_scale=x_scale,
            w_scale=w_scale,
        )
    return _launch_a8w8_blockscale_exact(
        XQ,
        WQ,
        Y,
        x_scale,
        w_scale,
        kid=kid,
        route_arch=route_arch,
        instance=instance,
    )


def _launch_a8w8_blockscale_bpreshuffle_exact(
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
    """Launch the shared physical bpreshuffle A8W8 ABI."""
    resolved = kid
    if instance is None:
        entry = "opus_gemm_a8w8_blockscale_bpreshuffle_launch"
        _check_same_device(entry, XQ, WQ, Y, x_scale, w_scale)
        arch = route_arch or _device_arch(XQ.device)
        resolved = _require_registered_kid(
            arch=arch,
            family=_A8W8_BPRESHUFFLE_FAMILY,
            kid=kid,
            output_dtype=Y.dtype,
        )
    _opus_gemm_a8w8_blockscale_bpreshuffle_launch_raw(
        XQ, WQ, x_scale, w_scale, Y, resolved
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
) -> Tensor:
    """Launch logical 2D bpreshuffled blockscale A8W8 GEMM.

    ``WQ`` pre-shuffle is a content/layout semantic. It
    cannot be proven from Tensor shape or strides. Build it with
    ``shuffle_weight(WQ, layout=(16, 16))``. The generated launcher checks
    output dtype, scale layout, batch and tile alignment.
    """
    if instance is None:
        _require_logical_rank(
            "a8w8_blockscale_bpreshuffle",
            2,
            XQ=XQ,
            WQ=WQ,
            Y=Y,
            x_scale=x_scale,
            w_scale=w_scale,
        )
    _launch_a8w8_blockscale_bpreshuffle_exact(
        XQ.unsqueeze(0),
        WQ.unsqueeze(0),
        x_scale,
        w_scale,
        Y.unsqueeze(0),
        kid=kid,
        route_arch=route_arch,
        instance=instance,
    )
    return Y


def _launch_a8w8_blockscale_bpreshuffle_bmm(
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
    """Launch batch-first 3D bpreshuffled blockscale A8W8 BMM."""
    if instance is None:
        _require_logical_rank(
            "a8w8_blockscale_bpreshuffle",
            3,
            XQ=XQ,
            WQ=WQ,
            Y=Y,
            x_scale=x_scale,
            w_scale=w_scale,
        )
    # The only currently registered bpreshuffle BMM-capable ABI is gfx942,
    # whose exact kid is batch-one and physically stores its scales as 2D.
    # Preserve a uniform batch-first public contract with no-copy views.
    resolved_arch = route_arch or _device_arch(XQ.device)
    launch_x_scale = x_scale
    launch_w_scale = w_scale
    if resolved_arch == "gfx942" and XQ.shape[0] == 1:
        launch_x_scale = x_scale.squeeze(0)
        launch_w_scale = w_scale.squeeze(0)
    return _launch_a8w8_blockscale_bpreshuffle_exact(
        XQ,
        WQ,
        launch_x_scale,
        launch_w_scale,
        Y,
        kid=kid,
        route_arch=resolved_arch,
        instance=instance,
    )


def _validate_a8w8_mxscale_bmm_tensors(
    XQ: Tensor,
    WQ: Tensor,
    Y: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
) -> tuple[int, int, int, int]:
    entry = "opus_gemm_a8w8_mxscale_bmm_launch"
    _check_same_device(entry, XQ, WQ, Y, x_scale, w_scale)
    if any(tensor.dim() != 3 for tensor in (XQ, WQ, Y, x_scale, w_scale)):
        raise ValueError(f"{entry}: all inputs and Y must be 3D")
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
    if N % 128 or K % 128:
        raise ValueError(
            f"{entry}: N and K must be multiples of 128; got N={N}, K={K}"
        )
    if (w_batch, w_K) != (batch, K):
        raise ValueError(
            f"{entry}: WQ must have shape [{batch},N,{K}], got {tuple(WQ.shape)}"
        )
    if tuple(Y.shape) != (M, batch, N):
        raise ValueError(
            f"{entry}: Y must have shape {(M, batch, N)}, got {tuple(Y.shape)}"
        )
    expected_x_scale = (M, batch, K // 128)
    expected_w_scale = (batch, N // 128, K // 128)
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
    return M, batch, N, K


def _mxscale_bmm_shape_for_plan(
    XQ: Tensor,
    WQ: Tensor,
) -> tuple[int, int, int, int]:
    """Read only the four dimensions needed by the immutable launch plan.

    The checked C++ BMM entry owns the dynamic Tensor contract (dtype, shape,
    device and stride).  Python still needs these dimensions before launch to
    resolve the exact-kid workspace plan.  Keep malformed-rank handling here,
    but do not repeat the full C++ contract on the valid eager hot path.
    """
    try:
        M, batch, K = map(int, XQ.shape)
        _w_batch, N, _w_K = map(int, WQ.shape)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(
            "opus_gemm_a8w8_mxscale_bmm_launch: XQ and WQ must be 3D"
        ) from exc
    return M, batch, N, K


@functools.lru_cache(maxsize=4096)
def _cached_a8w8_mxscale_bmm_plan(
    arch: str,
    kid: int,
    output_dtype: torch.dtype,
    M: int,
    batch: int,
    N: int,
    K: int,
    split_k: int,
) -> tuple[int, int]:
    """Return effective split and required FP32 workspace elements."""
    instance = get_kernel_instance(arch, _A8W8_MXSCALE_BMM_FAMILY, kid, output_dtype)
    if instance is None:
        raise ValueError(
            "no registered OPUS kernel for "
            f"(arch={arch!r}, family={_A8W8_MXSCALE_BMM_FAMILY!r}, "
            f"kid={kid}, Y.dtype={output_dtype})"
        )
    tag = instance.kernel_tag
    if tag not in _A8W8_MXSCALE_BMM_TAGS:
        raise ValueError(f"OPUS kid {kid} is not an MXFP8 BMM kernel")

    effective_split = max(1, split_k)
    m_align = max(1, int(instance.m_align))
    if M % m_align:
        raise ValueError(
            f"OPUS BMM kid {kid} requires M % {m_align} == 0; got M={M}"
        )
    if instance.direct_only and effective_split != 1:
        raise ValueError(f"OPUS BMM kid {kid} requires split_k <= 1")

    workspace_numel = 0
    if tag in _A8W8_MXSCALE_BMM_WORKSPACE_TAGS:
        if effective_split > 1:
            tiles_m = (M + instance.B_M - 1) // instance.B_M
            tiles_n = (N + instance.B_N - 1) // instance.B_N
            padded_m = tiles_m * instance.B_M
            padded_n = tiles_n * instance.B_N
            partial_numel = effective_split * batch * padded_m * padded_n
            if tag == "a8w8_mxscale_bmm_fused":
                counter_offset = (partial_numel * 4 + 255) & ~255
                counter_bytes = batch * tiles_m * tiles_n * 4
                workspace_numel = (counter_offset + counter_bytes + 3) // 4
            else:
                workspace_numel = partial_numel
    elif tag == "a8w8_mxscale_bmm_mouter_tunable":
        pass
    elif effective_split != 1:
        raise ValueError(f"OPUS BMM kid {kid} requires split_k <= 1")

    return effective_split, workspace_numel


def _launch_a8w8_mxscale_bmm_exact(
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
    """Launch the physical MXFP8 ABI: X/Y/x_scale are M-major 3D views."""
    M, batch, N, K = _mxscale_bmm_shape_for_plan(XQ, WQ)
    arch = route_arch or _device_arch(XQ.device)
    resolved = kid
    if instance is None:
        resolved = _require_registered_kid(
            arch=arch,
            family=_A8W8_MXSCALE_BMM_FAMILY,
            kid=kid,
            output_dtype=Y.dtype,
        )
    effective_split, required_numel = _cached_a8w8_mxscale_bmm_plan(
        arch, resolved, Y.dtype, M, batch, N, K, split_k
    )

    launch_workspace = workspace
    if required_numel:
        # Automatic allocation must not use untrusted shape metadata.  The
        # common split_k=1/no-workspace path skips this duplicate Python check;
        # workspace launches retain it until sizing moves behind the checked
        # C++ boundary as well.
        _validate_a8w8_mxscale_bmm_tensors(XQ, WQ, Y, x_scale, w_scale)
        if launch_workspace is None:
            launch_workspace = torch.empty(
                required_numel, dtype=torch.float32, device=XQ.device
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
            if launch_workspace.dtype != torch.float32:
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
            f"OPUS BMM kid {resolved} with split_k={split_k} does not use workspace"
        )

    _opus_gemm_a8w8_mxscale_bmm_launch_raw(
        XQ,
        WQ,
        Y,
        x_scale,
        w_scale,
        launch_workspace,
        resolved,
        effective_split,
    )
    return Y


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
    if instance is None:
        _require_logical_rank(
            "a8w8_mxscale_bmm",
            3,
            XQ=XQ,
            WQ=WQ,
            Y=Y,
            x_scale=x_scale,
            w_scale=w_scale,
        )
    launch_x = XQ.transpose(0, 1)
    launch_y = Y.transpose(0, 1)
    launch_x_scale = x_scale.transpose(0, 1)

    if instance is not None and workspace is None and split_k <= 1:
        # The checked C++ entry owns the dynamic tensor contract.  Public
        # routing already validated the immutable family/kid contract, so the
        # common split-one path need not repeat the Python registry/planner.
        _opus_gemm_a8w8_mxscale_bmm_launch_raw(
            launch_x,
            WQ,
            launch_y,
            launch_x_scale,
            w_scale,
            None,
            kid,
            max(1, split_k),
        )
        return Y

    _launch_a8w8_mxscale_bmm_exact(
        launch_x,
        WQ,
        launch_y,
        launch_x_scale,
        w_scale,
        kid=kid,
        split_k=split_k,
        workspace=workspace,
        route_arch=route_arch,
        instance=instance,
    )
    return Y


__all__: list[str] = []
