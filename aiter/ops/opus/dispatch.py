# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Contract validation and exact-kid dispatch for public OPUS operations."""

from __future__ import annotations

from functools import lru_cache

import torch
from torch import Tensor

from csrc.opus_gemm.opus_gemm_common import (
    KernelKey,
    OpusGemmInstance,
    get_kernel_instance,
)

from . import gemm_op_a8w8 as _a8w8_family
from . import gemm_op_a16w16 as _a16w16_family
from . import launch_plan
from ._arch import _device_arch_and_cu


def _validate_a16w16_public_contract(
    *,
    kid: int,
    input_dtype: torch.dtype,
    weight_dtype: torch.dtype,
    layout: str,
    has_x_scale: bool,
    has_w_scale: bool,
) -> None:
    """Validate A16W16-only options shared by both public routers."""
    if input_dtype != weight_dtype:
        raise ValueError(
            f"OPUS requires matching XQ/WQ dtypes; got " f"{input_dtype}/{weight_dtype}"
        )
    if input_dtype != torch.bfloat16:
        raise ValueError(f"OPUS kid {kid} requires bf16 XQ/WQ; got {input_dtype}")
    if layout != "plain":
        raise ValueError(
            f"OPUS kid {kid} belongs to family a16w16 and requires "
            f"layout='plain'; got {layout!r}"
        )
    if has_x_scale or has_w_scale:
        raise ValueError("OPUS a16w16 does not accept x_scale/w_scale")


@lru_cache(maxsize=4096)
def _resolve_contract(
    key: KernelKey,
    operation: str,
    input_dtype: torch.dtype,
    weight_dtype: torch.dtype,
    output_dtype: torch.dtype,
    layout: str,
    has_x_scale: bool,
    has_w_scale: bool,
    has_bias: bool,
    has_workspace: bool,
    split_k: int,
) -> OpusGemmInstance:
    """Validate/cache the public contract and selected kernel instance."""
    instance = get_kernel_instance(key.gfx, key.family, key.kid, cu_num=key.cu_num)
    if instance is None:
        raise ValueError(f"unknown OPUS kernel {key.encode()}")
    kid = key.kid

    if key.family == "a16w16":
        _validate_a16w16_public_contract(
            kid=kid,
            input_dtype=input_dtype,
            weight_dtype=weight_dtype,
            layout=layout,
            has_x_scale=has_x_scale,
            has_w_scale=has_w_scale,
        )
        return instance

    launch_plan._validate_a8w8_public_contract(
        family=key.family,
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
    if key.family == "a8w8_mxscale_bmm":
        if operation != "opus_bmm":
            raise ValueError("OPUS a8w8_mxscale_bmm supports opus_bmm only")
    elif operation != "opus_gemm":
        raise ValueError(f"OPUS family {key.family} is GEMM-only; use opus_gemm")
    return instance


def _opus_dispatch(
    operation: str,
    rank: int,
    XQ: Tensor,
    WQ: Tensor,
    Y: Tensor,
    *,
    kid: int,
    family: str,
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

    if type(split_k) is not int:
        raise ValueError(f"OPUS split_k must be an integer, got {split_k!r}")
    if split_k < 0:
        raise ValueError(f"OPUS split_k must be non-negative, got {split_k}")

    has_x_scale = x_scale is not None
    has_w_scale = w_scale is not None
    arch, cu_num = _device_arch_and_cu(XQ.device)
    key = KernelKey(family, arch, cu_num, kid)
    instance = _resolve_contract(
        key,
        operation,
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
    family = key.family
    if family == "a16w16":
        launch = (
            _a16w16_family._launch_a16w16_gemm
            if operation == "opus_gemm"
            else _a16w16_family._launch_a16w16_bmm
        )
        return launch(
            XQ,
            WQ,
            Y,
            bias,
            kid=kid,
            split_k=split_k,
            workspace=workspace,
            route_arch=key.gfx,
            route_cu_num=key.cu_num,
            instance=instance,
        )

    if family == "a8w8_mxscale_bmm":
        return _a8w8_family._launch_a8w8_mxscale_bmm(
            XQ,
            WQ,
            Y,
            x_scale,
            w_scale,
            kid=kid,
            split_k=split_k,
            workspace=workspace,
            instance=instance,
        )

    if family == "a8w8":
        return _a8w8_family._launch_a8w8_gemm(
            XQ,
            WQ,
            Y,
            kid=kid,
            instance=instance,
        )
    if family == "a8w8_blockscale":
        return _a8w8_family._launch_a8w8_blockscale_gemm(
            XQ,
            WQ,
            Y,
            x_scale,
            w_scale,
            kid=kid,
            instance=instance,
        )
    if family == "a8w8_blockscale_bpreshuffle":
        return _a8w8_family._launch_a8w8_blockscale_bpreshuffle_gemm(
            XQ,
            WQ,
            x_scale,
            w_scale,
            Y,
            kid=kid,
            instance=instance,
        )
    raise RuntimeError(f"unsupported canonical OPUS family {family!r}")
