# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Private OPUS A8W8 exact-kid launch APIs."""

# Keep annotations eager in this binding module.  ``torch_compile_guard`` uses
# the first parameter's concrete ``torch.Tensor`` identity to decide whether a
# custom op already has a Tensor dispatch key.  Postponed string annotations
# make it add a dummy CUDA Tensor to every raw launch, which is observable host
# overhead on short BMM kernels.

import torch
from torch import Tensor

from ...jit.core import compile_ops
from ._arch import _device_arch
from .launch_plan import (
    _A8W8_BLOCKSCALE_FAMILY,
    _A8W8_BPRESHUFFLE_FAMILY,
    _A8W8_FAMILY,
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
    """Launch one resolved physical A8W8 ABI through its pybind binding.

    A8W8 currently has no stable C ABI, so unlike the A16W16 backend this
    facade does not prime pybind and switch to ctypes.  It is nevertheless the
    single low-level entry used by all A8 adapters, keeping raw binding choice
    out of shape, registry and workspace policy.
    """
    if family == _A8W8_FAMILY:
        if x_scale is not None or w_scale is not None:
            raise RuntimeError("A8W8 no-scale backend received scale tensors")
        if workspace is not None or split_k != 0:
            raise RuntimeError("A8W8 no-scale backend received split-K state")
        _opus_gemm_a8w8_launch_raw(XQ, WQ, Y, kid)
        return None

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
        return None

    if family == _A8W8_BPRESHUFFLE_FAMILY:
        if workspace is not None or split_k != 0:
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
        )
        return None

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
        return None

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
                "opus_gemm A8W8 expects logical 2D XQ/WQ/Y; "
                "this family is GEMM-only"
            )
        arch = route_arch or _device_arch(XQ.device)
        resolved_kid = _require_registered_kid(
            arch=arch,
            family=_A8W8_FAMILY,
            kid=kid,
            output_dtype=Y.dtype,
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
        if any(
            tensor.dim() != 2 for tensor in (XQ, WQ, Y, x_scale, w_scale)
        ):
            raise ValueError(
                "opus_gemm A8W8 blockscale expects logical 2D "
                "XQ/WQ/Y/x_scale/w_scale; this family is GEMM-only"
            )
        arch = route_arch or _device_arch(XQ.device)
        resolved_kid = _require_registered_kid(
            arch=arch,
            family=_A8W8_BLOCKSCALE_FAMILY,
            kid=kid,
            output_dtype=Y.dtype,
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
) -> Tensor:
    """Launch logical 2D bpreshuffled blockscale A8W8 GEMM.

    ``WQ`` pre-shuffle is a content/layout semantic. It
    cannot be proven from Tensor shape or strides. Build it with
    ``shuffle_weight(WQ, layout=(16, 16))``. The generated launcher checks
    output dtype, scale layout, batch and tile alignment.
    """
    resolved_kid = kid
    if instance is None:
        if any(
            tensor.dim() != 2 for tensor in (XQ, WQ, Y, x_scale, w_scale)
        ):
            raise ValueError(
                "opus_gemm A8W8 blockscale bpreshuffle expects logical 2D "
                "XQ/WQ/Y/x_scale/w_scale; this family is GEMM-only"
            )
        arch = route_arch or _device_arch(XQ.device)
        resolved_kid = _require_registered_kid(
            arch=arch,
            family=_A8W8_BPRESHUFFLE_FAMILY,
            kid=kid,
            output_dtype=Y.dtype,
        )
    _launch_a8w8_backend(
        XQ.unsqueeze(0),
        WQ.unsqueeze(0),
        Y.unsqueeze(0),
        x_scale,
        w_scale,
        None,
        _A8W8_BPRESHUFFLE_FAMILY,
        resolved_kid,
        0,
    )
    return Y


def _validate_a8w8_mxscale_bmm_tensors(
    XQ: Tensor,
    WQ: Tensor,
    Y: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
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
        if any(
            tensor.dim() != 3 for tensor in (XQ, WQ, Y, x_scale, w_scale)
        ):
            raise ValueError(
                "opus_bmm A8W8 mxscale expects batch-first 3D "
                "XQ/WQ/Y/x_scale/w_scale"
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
    arch = route_arch or _device_arch(XQ.device)
    plan = _get_cached_a8w8_mxscale_bmm_plan(
        arch,
        int(kid),
        Y.dtype,
        M,
        batch,
        N,
        K,
        int(split_k),
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


__all__: list[str] = []
