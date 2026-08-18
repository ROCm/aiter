# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Public OPUS GEMM/BMM interfaces backed by shared exact-kid launchers."""

from __future__ import annotations

from functools import lru_cache
from operator import index

from torch import Tensor
import torch

from csrc.opus_gemm.opus_gemm_common import kernels_list

_LAYOUT_ALIASES = {
    "plain": "plain",
    "normal": "plain",
    "row_major": "plain",
    "bpreshuffle": "bpreshuffle",
    "preshuffle": "bpreshuffle",
    "bpreshuffled": "bpreshuffle",
    "mxscale_bmm": "mxscale_bmm",
    "mxfp8_bmm": "mxscale_bmm",
    "bmm_mxscale": "mxscale_bmm",
}


def _normalize_kid(kid: object) -> int:
    if type(kid) is int:
        return kid
    if isinstance(kid, bool):
        raise ValueError(f"OPUS kid must be an integer id, got {kid!r}")
    try:
        return int(index(kid))
    except TypeError as exc:
        raise ValueError(f"OPUS kid must be an integer id, got {kid!r}") from exc


def _normalize_split_k(split_k: object) -> int:
    if type(split_k) is int:
        resolved = split_k
    elif isinstance(split_k, bool):
        raise ValueError(f"OPUS split_k must be an integer, got {split_k!r}")
    else:
        try:
            resolved = int(index(split_k))
        except TypeError as exc:
            raise ValueError(
                f"OPUS split_k must be an integer, got {split_k!r}"
            ) from exc
    if resolved < 0:
        raise ValueError(f"OPUS split_k must be non-negative, got {resolved}")
    return resolved


def _normalize_layout(layout: object) -> str:
    if layout in ("plain", "bpreshuffle", "mxscale_bmm"):
        return layout
    token = str(layout).strip().lower()
    try:
        return _LAYOUT_ALIASES[token]
    except KeyError as exc:
        raise ValueError(
            f"unsupported OPUS weight layout {layout!r}; expected "
            "'plain', 'bpreshuffle' or 'mxscale_bmm'"
        ) from exc


def _require_tensor(operation: str, name: str, value: object) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError(
            f"{operation}: {name} must be a Tensor, got {type(value)!r}"
        )
    return value


def _require_gpu_tensor(tensor: Tensor) -> None:
    """Reject an all-CPU call before entering a GPU-only raw launcher."""
    if tensor.device.type != "cuda":
        raise RuntimeError(f"OPUS requires a GPU tensor; got device {tensor.device}")


@lru_cache(maxsize=4096)
def _cached_public_contract(
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
) -> tuple[str, str, object, object]:
    """Validate/cache immutable routing and its lazily imported family module."""
    instance = kernels_list.get(kid)
    if instance is None:
        raise ValueError(f"unknown OPUS kid {kid}")

    route_arch = (instance.arch_prefix or "gfx950").lower()
    if instance.kernel_tag.startswith("a16w16"):
        from . import gemm_op_a16w16 as family_module

        family_module._validate_a16w16_public_contract(
            kid=kid,
            instance=instance,
            input_dtype=input_dtype,
            weight_dtype=weight_dtype,
            output_dtype=output_dtype,
            layout=layout,
            has_x_scale=has_x_scale,
            has_w_scale=has_w_scale,
        )
        return route_arch, "a16w16", instance, family_module

    from . import gemm_op_a8w8 as family_module

    family = family_module._validate_a8w8_public_contract(
        kernel_tag=instance.kernel_tag,
        kid=kid,
        input_dtype=input_dtype,
        weight_dtype=weight_dtype,
        output_dtype=output_dtype,
        layout=layout,
        has_x_scale=has_x_scale,
        has_w_scale=has_w_scale,
        has_bias=has_bias,
        has_workspace=has_workspace,
        split_k=split_k,
    )
    return route_arch, family, instance, family_module


def _require_operation_rank(
    operation: str,
    rank: int,
    XQ: Tensor,
    WQ: Tensor,
    Y: Tensor,
    x_scale: Tensor | None,
    w_scale: Tensor | None,
) -> None:
    tensors = {"XQ": XQ, "WQ": WQ, "Y": Y}
    if x_scale is not None:
        tensors["x_scale"] = _require_tensor(operation, "x_scale", x_scale)
    if w_scale is not None:
        tensors["w_scale"] = _require_tensor(operation, "w_scale", w_scale)
    invalid = [name for name, tensor in tensors.items() if tensor.dim() != rank]
    if invalid:
        other = "opus_bmm" if operation == "opus_gemm" else "opus_gemm"
        expected = "logical 2D" if rank == 2 else "batch-first 3D"
        raise ValueError(
            f"{operation} expects {expected} {', '.join(invalid)}; use "
            f"{other} for {'batch-first 3D' if rank == 2 else 'logical 2D'} "
            "tensors"
        )


def _opus_dispatch(
    operation: str,
    rank: int,
    XQ: Tensor,
    WQ: Tensor,
    Y: Tensor,
    *,
    kid: int,
    layout: str = "plain",
    x_scale: Tensor | None = None,
    w_scale: Tensor | None = None,
    bias: Tensor | None = None,
    split_k: int = 0,
    workspace: Tensor | None = None,
) -> Tensor:
    if not (
        isinstance(XQ, Tensor)
        and isinstance(WQ, Tensor)
        and isinstance(Y, Tensor)
    ):
        XQ = _require_tensor(operation, "XQ", XQ)
        WQ = _require_tensor(operation, "WQ", WQ)
        Y = _require_tensor(operation, "Y", Y)
    _require_operation_rank(
        operation, rank, XQ, WQ, Y, x_scale, w_scale
    )

    resolved_kid = kid if type(kid) is int else _normalize_kid(kid)
    resolved_layout = (
        layout
        if layout in ("plain", "bpreshuffle", "mxscale_bmm")
        else _normalize_layout(layout)
    )
    if operation == "opus_gemm" and resolved_layout == "mxscale_bmm":
        raise ValueError("layout='mxscale_bmm' is only supported by opus_bmm")
    resolved_split_k = (
        split_k
        if type(split_k) is int and split_k >= 0
        else _normalize_split_k(split_k)
    )

    has_x_scale = x_scale is not None
    has_w_scale = w_scale is not None
    route_arch, family, instance, family_module = _cached_public_contract(
        resolved_kid,
        XQ.dtype,
        WQ.dtype,
        Y.dtype,
        resolved_layout,
        has_x_scale,
        has_w_scale,
        bias is not None,
        workspace is not None,
        resolved_split_k,
    )

    if family == "a16w16":
        launch = (
            family_module._launch_a16w16_gemm
            if operation == "opus_gemm"
            else family_module._launch_a16w16_bmm
        )
        return launch(
            XQ,
            WQ,
            Y,
            bias,
            kid=resolved_kid,
            split_k=resolved_split_k,
            workspace=workspace,
            route_arch=route_arch,
            instance=instance,
        )

    # The A8 family adapters receive the cached registry instance so their
    # common paths enter the checked raw C++ ABI directly.  The adapters own
    # only logical GEMM/BMM layout conversion; device, dtype, stride, shape and
    # exact-kid checks remain at the unchanged raw boundary.
    _require_gpu_tensor(XQ)

    if family == "a8w8":
        launch = (
            family_module._launch_a8w8_gemm
            if operation == "opus_gemm"
            else family_module._launch_a8w8_bmm
        )
        return launch(
            XQ,
            WQ,
            Y,
            kid=resolved_kid,
            route_arch=route_arch,
            instance=instance,
        )
    assert x_scale is not None and w_scale is not None
    if family == "a8w8_blockscale":
        launch = (
            family_module._launch_a8w8_blockscale_gemm
            if operation == "opus_gemm"
            else family_module._launch_a8w8_blockscale_bmm
        )
        return launch(
            XQ,
            WQ,
            Y,
            x_scale,
            w_scale,
            kid=resolved_kid,
            route_arch=route_arch,
            instance=instance,
        )
    if family == "a8w8_blockscale_bpreshuffle":
        launch = (
            family_module._launch_a8w8_blockscale_bpreshuffle_gemm
            if operation == "opus_gemm"
            else family_module._launch_a8w8_blockscale_bpreshuffle_bmm
        )
        return launch(
            XQ,
            WQ,
            x_scale,
            w_scale,
            Y,
            kid=resolved_kid,
            route_arch=route_arch,
            instance=instance,
        )
    if family == "a8w8_mxscale_bmm":
        return family_module._launch_a8w8_mxscale_bmm(
            XQ,
            WQ,
            Y,
            x_scale,
            w_scale,
            kid=resolved_kid,
            split_k=resolved_split_k,
            workspace=workspace,
            route_arch=route_arch,
            instance=instance,
        )
    raise RuntimeError(f"unsupported canonical OPUS family {family!r}")


def opus_gemm(
    XQ: Tensor,
    WQ: Tensor,
    Y: Tensor,
    *,
    kid: int,
    layout: str = "plain",
    x_scale: Tensor | None = None,
    w_scale: Tensor | None = None,
    bias: Tensor | None = None,
    split_k: int = 0,
    workspace: Tensor | None = None,
) -> Tensor:
    """Launch logical 2D ``[M,K] x [N,K] -> [M,N]`` by exact ``kid``.

    ``Y`` is caller-owned and returned. ``layout='bpreshuffle'`` declares a
    transformed WQ content layout that Tensor metadata cannot prove.
    """
    return _opus_dispatch(
        "opus_gemm",
        2,
        XQ,
        WQ,
        Y,
        kid=kid,
        layout=layout,
        x_scale=x_scale,
        w_scale=w_scale,
        bias=bias,
        split_k=split_k,
        workspace=workspace,
    )


def opus_bmm(
    XQ: Tensor,
    WQ: Tensor,
    Y: Tensor,
    *,
    kid: int,
    layout: str = "plain",
    x_scale: Tensor | None = None,
    w_scale: Tensor | None = None,
    bias: Tensor | None = None,
    split_k: int = 0,
    workspace: Tensor | None = None,
) -> Tensor:
    """Launch batch-first ``[B,M,K] x [B,N,K] -> [B,M,N]`` by exact kid."""
    return _opus_dispatch(
        "opus_bmm",
        3,
        XQ,
        WQ,
        Y,
        kid=kid,
        layout=layout,
        x_scale=x_scale,
        w_scale=w_scale,
        bias=bias,
        split_k=split_k,
        workspace=workspace,
    )


__all__ = ["opus_gemm", "opus_bmm"]
