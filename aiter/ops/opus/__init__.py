# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
"""One public OPUS GEMM interface backed by exact-kid family launchers."""

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


def _require_tensor(name: str, value: object) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError(f"opus_gemm: {name} must be a Tensor, got {type(value)!r}")
    return value


def _require_gpu_tensor(tensor: Tensor) -> None:
    """Reject an all-CPU call before entering a GPU-only raw launcher."""
    if tensor.device.type != "cuda":
        raise RuntimeError(
            f"OPUS GEMM requires a GPU tensor; got device {tensor.device}"
        )


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
    """Launch one registered OPUS GEMM kernel selected by exact ``kid``.

    ``kid`` is mandatory and uniquely determines architecture and logical
    family from the canonical registry. Tensor dtypes, ``layout`` and scale
    presence are validated against that route before dispatching to the
    family-specific raw launch API. ``Y`` is caller-owned and returned.

    ``layout='bpreshuffle'`` declares that ``WQ`` has already been transformed
    for a bpreshuffle kernel; Tensor metadata cannot prove this content layout.
    """
    if not (
        isinstance(XQ, Tensor)
        and isinstance(WQ, Tensor)
        and isinstance(Y, Tensor)
    ):
        XQ = _require_tensor("XQ", XQ)
        WQ = _require_tensor("WQ", WQ)
        Y = _require_tensor("Y", Y)

    resolved_kid = kid if type(kid) is int else _normalize_kid(kid)
    resolved_layout = (
        layout
        if layout in ("plain", "bpreshuffle", "mxscale_bmm")
        else _normalize_layout(layout)
    )
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
        if (
            route_arch == "gfx950"
            and workspace is not None
            and resolved_split_k > 0
            and instance.splitk_workspace_dtype is not None
        ):
            # A caller-owned gfx950 workspace already gives this exact-kid
            # path everything it needs to launch.  Preserve the dynamic
            # shape/stride contract and the cached exact-plan validation, but
            # avoid re-reading device metadata and re-entering the generic
            # workspace planner on every short public call.  The existing C
            # ABI remains the final checked boundary (device/stream guard,
            # tensor/workspace checks and exact-kid dispatch).
            _require_gpu_tensor(XQ)
            family_module._check_a16w16_launch_layout(XQ, WQ, Y)
            batch, M, K = XQ.shape
            N = Y.shape[2]
            actual_kid, launch_split_k, workspace_plan = (
                family_module._cached_explicit_a16w16_plan(
                    route_arch,
                    M,
                    N,
                    K,
                    batch,
                    1,  # CU count is not consulted by explicit gfx950 plans.
                    bias is not None,
                    XQ.dtype,
                    Y.dtype,
                    resolved_kid,
                    resolved_split_k,
                )
            )
            if workspace_plan is None:
                raise RuntimeError(
                    f"OPUS gfx950 kid {actual_kid} unexpectedly has no "
                    "caller-workspace plan"
                )
            family_module._opus_gemm_a16w16_launch_ctypes_raw(
                XQ,
                WQ,
                Y,
                bias,
                workspace,
                actual_kid,
                launch_split_k,
            )
            return Y
        return family_module._launch_a16w16(
            XQ,
            WQ,
            Y,
            bias,
            kid=resolved_kid,
            split_k=resolved_split_k,
            workspace=workspace,
        )

    # The A8 raw entries are checked C++ launchers: they validate device,
    # dtype, shape/stride, architecture and exact kid before dispatch.  The
    # cached public contract above already owns immutable Python routing and
    # option-presence checks, so repeating the private family registry/device
    # walk here only adds host latency.  Keep the private adapters unchanged
    # for direct callers and for BMM workspace planning.
    _require_gpu_tensor(XQ)

    if family == "a8w8":
        family_module._opus_gemm_a8w8_launch_raw(XQ, WQ, Y, resolved_kid)
        return Y
    assert x_scale is not None and w_scale is not None
    if family == "a8w8_blockscale":
        family_module._opus_gemm_a8w8_blockscale_launch_raw(
            XQ, WQ, Y, x_scale, w_scale, resolved_kid
        )
        return Y
    if family == "a8w8_blockscale_bpreshuffle":
        family_module._opus_gemm_a8w8_blockscale_bpreshuffle_launch_raw(
            XQ, WQ, x_scale, w_scale, Y, resolved_kid
        )
        return Y
    if family == "a8w8_mxscale_bmm":
        if workspace is None and resolved_split_k <= 1:
            family_module._opus_gemm_a8w8_mxscale_bmm_launch_raw(
                XQ,
                WQ,
                Y,
                x_scale,
                w_scale,
                None,
                resolved_kid,
                max(1, resolved_split_k),
            )
            return Y
        return family_module._launch_a8w8_mxscale_bmm(
            XQ,
            WQ,
            Y,
            x_scale,
            w_scale,
            kid=resolved_kid,
            split_k=resolved_split_k,
            workspace=workspace,
        )
    raise RuntimeError(f"unsupported canonical OPUS family {family!r}")


__all__ = ["opus_gemm"]
