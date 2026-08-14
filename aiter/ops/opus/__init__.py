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
) -> tuple[str, str, object]:
    """Validate/cache immutable routing and option-presence scalars only."""
    instance = kernels_list.get(kid)
    if instance is None:
        raise ValueError(f"unknown OPUS kid {kid}")

    route_arch = (instance.arch_prefix or "gfx950").lower()
    if instance.kernel_tag.startswith("a16w16"):
        from .gemm_op_a16w16 import _validate_a16w16_public_contract

        _validate_a16w16_public_contract(
            kid=kid,
            instance=instance,
            input_dtype=input_dtype,
            weight_dtype=weight_dtype,
            output_dtype=output_dtype,
            layout=layout,
            has_x_scale=has_x_scale,
            has_w_scale=has_w_scale,
        )
        return route_arch, "a16w16", instance

    from .gemm_op_a8w8 import _validate_a8w8_public_contract

    family = _validate_a8w8_public_contract(
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
    return route_arch, family, instance


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
    _route_arch, family, _instance = _cached_public_contract(
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
        from .gemm_op_a16w16 import _launch_a16w16

        return _launch_a16w16(
            XQ,
            WQ,
            Y,
            bias,
            kid=resolved_kid,
            split_k=resolved_split_k,
            workspace=workspace,
        )

    from .gemm_op_a8w8 import (
        _launch_a8w8,
        _launch_a8w8_blockscale,
        _launch_a8w8_blockscale_bpreshuffle,
        _launch_a8w8_mxscale_bmm,
    )

    if family == "a8w8":
        return _launch_a8w8(XQ, WQ, Y, kid=resolved_kid)
    assert x_scale is not None and w_scale is not None
    if family == "a8w8_blockscale":
        return _launch_a8w8_blockscale(
            XQ,
            WQ,
            Y,
            x_scale,
            w_scale,
            kid=resolved_kid,
        )
    if family == "a8w8_blockscale_bpreshuffle":
        return _launch_a8w8_blockscale_bpreshuffle(
            XQ,
            WQ,
            x_scale,
            w_scale,
            Y,
            kid=resolved_kid,
        )
    if family == "a8w8_mxscale_bmm":
        return _launch_a8w8_mxscale_bmm(
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
