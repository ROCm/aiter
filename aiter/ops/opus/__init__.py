# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Public OPUS GEMM/BMM interfaces backed by shared exact-kid launchers."""

from __future__ import annotations

from functools import lru_cache

import torch
from torch import Tensor

from csrc.opus_gemm.opus_gemm_common import (
    OpusGemmInstance,
    get_kernel_instance,
    kernels_list,
)


def _validate_a16w16_public_contract(
    *,
    kid: int,
    instance: OpusGemmInstance,
    input_dtype: torch.dtype,
    weight_dtype: torch.dtype,
    output_dtype: torch.dtype,
    layout: str,
    has_x_scale: bool,
    has_w_scale: bool,
) -> None:
    """Validate A16W16-only options shared by both public routers."""
    if input_dtype != weight_dtype:
        raise ValueError(
            f"OPUS requires matching XQ/WQ dtypes; got "
            f"{input_dtype}/{weight_dtype}"
        )
    if input_dtype != torch.bfloat16:
        raise ValueError(
            f"OPUS kid {kid} requires bf16 XQ/WQ; got {input_dtype}"
        )
    if layout != "plain":
        raise ValueError(
            f"OPUS kid {kid} belongs to family a16w16 and requires "
            f"layout='plain'; got {layout!r}"
        )
    if has_x_scale or has_w_scale:
        raise ValueError("OPUS a16w16 does not accept x_scale/w_scale")
    arch = (instance.arch_prefix or "gfx950").lower()
    if get_kernel_instance(arch, "a16w16", kid, output_dtype) is None:
        raise ValueError(
            f"OPUS kid {kid} does not support Y.dtype={output_dtype}"
        )


@lru_cache(maxsize=4096)
def _resolve_contract(
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
) -> tuple[str, object, object]:
    """Validate/cache the public contract and its lazily imported family module."""
    instance = kernels_list.get(kid)
    if instance is None:
        raise ValueError(f"unknown OPUS kid {kid}")

    if instance.kernel_tag.startswith("a16w16"):
        from . import gemm_op_a16w16 as family_module

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
        return "a16w16", instance, family_module

    from . import launch_plan

    family = launch_plan._validate_a8w8_public_contract(
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
    from . import gemm_op_a8w8 as family_module

    return family, instance, family_module


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
    bad_name = None
    bad_value = None
    if not isinstance(XQ, Tensor):
        bad_name, bad_value = "XQ", XQ
    elif not isinstance(WQ, Tensor):
        bad_name, bad_value = "WQ", WQ
    elif not isinstance(Y, Tensor):
        bad_name, bad_value = "Y", Y
    elif x_scale is not None and not isinstance(x_scale, Tensor):
        bad_name, bad_value = "x_scale", x_scale
    elif w_scale is not None and not isinstance(w_scale, Tensor):
        bad_name, bad_value = "w_scale", w_scale
    if bad_name is not None:
        raise TypeError(
            f"{operation}: {bad_name} must be a Tensor, got {type(bad_value)!r}"
        )

    bad_rank = None
    if XQ.dim() != rank:
        bad_rank = "XQ"
    elif WQ.dim() != rank:
        bad_rank = "WQ"
    elif Y.dim() != rank:
        bad_rank = "Y"
    elif x_scale is not None and x_scale.dim() != rank:
        bad_rank = "x_scale"
    elif w_scale is not None and w_scale.dim() != rank:
        bad_rank = "w_scale"
    if bad_rank is not None:
        expected = "logical 2D" if rank == 2 else "batch-first 3D"
        raise ValueError(
            f"{operation} expects {expected} {bad_rank}; the selected kid "
            "family must also support that operation"
        )

    if type(kid) is not int:
        raise ValueError(f"OPUS kid must be an integer id, got {kid!r}")
    if type(split_k) is not int:
        raise ValueError(f"OPUS split_k must be an integer, got {split_k!r}")
    if split_k < 0:
        raise ValueError(f"OPUS split_k must be non-negative, got {split_k}")
    if layout not in (
        "plain",
        "bpreshuffle",
        "mxscale_bmm",
    ):
        raise ValueError(
            f"unsupported OPUS weight layout {layout!r}; expected "
            "'plain', 'bpreshuffle' or 'mxscale_bmm'"
        )
    if operation == "opus_gemm" and layout == "mxscale_bmm":
        raise ValueError("layout='mxscale_bmm' is only supported by opus_bmm")

    has_x_scale = x_scale is not None
    has_w_scale = w_scale is not None
    family, instance, family_module = _resolve_contract(
        kid,
        XQ.dtype,
        WQ.dtype,
        Y.dtype,
        layout,
        has_x_scale,
        has_w_scale,
        bias is not None,
        workspace is not None,
        split_k,
    )
    route_arch = (instance.arch_prefix or "gfx950").lower()

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
            kid=kid,
            split_k=split_k,
            workspace=workspace,
            route_arch=route_arch,
            instance=instance,
        )

    if family == "a8w8_mxscale_bmm":
        if operation != "opus_bmm":
            raise ValueError("OPUS a8w8_mxscale_bmm supports opus_bmm only")
        assert x_scale is not None and w_scale is not None
        return family_module._launch_a8w8_mxscale_bmm(
            XQ,
            WQ,
            Y,
            x_scale,
            w_scale,
            kid=kid,
            split_k=split_k,
            workspace=workspace,
            route_arch=route_arch,
            instance=instance,
        )

    if operation != "opus_gemm":
        raise ValueError(f"OPUS family {family} is GEMM-only; use opus_gemm")

    if family == "a8w8":
        return family_module._launch_a8w8_gemm(
            XQ,
            WQ,
            Y,
            kid=kid,
            route_arch=route_arch,
            instance=instance,
        )
    assert x_scale is not None and w_scale is not None
    if family == "a8w8_blockscale":
        return family_module._launch_a8w8_blockscale_gemm(
            XQ,
            WQ,
            Y,
            x_scale,
            w_scale,
            kid=kid,
            route_arch=route_arch,
            instance=instance,
        )
    if family == "a8w8_blockscale_bpreshuffle":
        return family_module._launch_a8w8_blockscale_bpreshuffle_gemm(
            XQ,
            WQ,
            x_scale,
            w_scale,
            Y,
            kid=kid,
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
