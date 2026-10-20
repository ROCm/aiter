# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import math

import torch
import triton

from aiter.ops.triton._triton_kernels.common.activation import _get_activation_from_str
from aiter.ops.triton._triton_kernels.normalization.group_norm import (
    _group_norm_apply_kernel,
    _group_norm_stats_kernel,
)
from aiter.ops.triton.utils.device_info import get_num_sms
from aiter.ops.triton.utils.logger import AiterTritonLogger

_LOGGER = AiterTritonLogger()

# Elements of one group that one program normalizes. Larger groups are split
# over several programs, so that a few large groups still fill the GPU.
_ELEMENTS_PER_PROGRAM = 32768
_BLOCK = 1024


def group_norm(
    x: torch.Tensor,
    num_groups: int,
    weight: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    eps: float = 1e-5,
    temb: torch.Tensor | None = None,
    activation: str | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Group normalization with an optional input add and output activation, in one pass
    over the output:

        out = act(group_norm(x + temb[:, :, None, ...], num_groups, weight, bias, eps))

    A UNet or VAE ResNet block uses this pattern: GroupNorm then SiLU, and in the
    second norm of a UNet block, the time embedding is added first.

    Key parameters:
    - x: (N, C, *) input, contiguous (NCHW, NCDHW or (N, C, L)). C must be a
      multiple of num_groups.
    - num_groups: number of groups.
    - weight, bias: optional (C,) affine parameters.
    - eps: added to the variance.
    - temb: optional (N, C) tensor, added to x per channel before the norm.
    - activation: optional activation after the affine ("silu", "gelu",
      "gelu_tanh", "relu", ...).
    - out: optional output with the shape and dtype of x.

    The statistics are computed in fp32. The output is rounded to the dtype of x
    once, after the activation.

    Returns:
    - out: same shape and dtype as x.
    """
    _LOGGER.info(
        "GROUP_NORM: x=%s groups=%s temb=%s act=%s",
        tuple(x.shape),
        num_groups,
        None if temb is None else tuple(temb.shape),
        activation,
    )

    assert x.ndim >= 2, f"x must be (N, C, *), got {tuple(x.shape)}"
    assert x.is_contiguous(), "x must be contiguous"
    N, C = x.shape[0], x.shape[1]
    assert C % num_groups == 0, f"C={C} must be a multiple of num_groups={num_groups}"
    for name, p in (("weight", weight), ("bias", bias)):
        assert p is None or p.shape == (
            C,
        ), f"{name} must be ({C},), got {tuple(p.shape)}"
    if temb is not None:
        assert temb.shape == (N, C), f"temb must be ({N}, {C}), got {tuple(temb.shape)}"
        assert temb.is_contiguous(), "temb must be contiguous"

    if out is None:
        out = torch.empty_like(x)
    else:
        assert (
            out.shape == x.shape and out.dtype == x.dtype and out.is_contiguous()
        ), "out must be contiguous, with the shape and dtype of x"

    HW = math.prod(x.shape[2:])
    CPG = C // num_groups
    GS = CPG * HW
    n_groups = N * num_groups

    # Split a group only when the groups alone do not fill the GPU.
    split = triton.cdiv(GS, _ELEMENTS_PER_PROGRAM)
    split = min(split, max(1, triton.cdiv(4 * get_num_sms(), n_groups)))
    chunk = triton.cdiv(triton.cdiv(GS, split), _BLOCK) * _BLOCK
    split = triton.cdiv(GS, chunk)

    t = temb if temb is not None else x
    part = x
    if split > 1:
        part = torch.empty(
            (n_groups * split * 3,), dtype=torch.float32, device=x.device
        )
        _group_norm_stats_kernel[(n_groups, split)](
            x,
            t,
            part,
            C,
            HW,
            CPG,
            GS,
            chunk,
            G=num_groups,
            HAS_TEMB=temb is not None,
            BLOCK=_BLOCK,
            num_warps=8,
        )

    _group_norm_apply_kernel[(n_groups, split)](
        x,
        t,
        weight,
        bias,
        part,
        out,
        C,
        HW,
        CPG,
        GS,
        chunk,
        eps,
        G=num_groups,
        SPLIT_P2=triton.next_power_of_2(split),
        INLINE_STATS=split == 1,
        HAS_TEMB=temb is not None,
        HAS_WEIGHT=weight is not None,
        HAS_BIAS=bias is not None,
        activation=_get_activation_from_str(activation) if activation else "",
        use_activation=activation is not None,
        BLOCK=_BLOCK,
        num_warps=8,
    )
    return out
