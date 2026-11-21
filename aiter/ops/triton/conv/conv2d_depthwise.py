# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import torch
import triton

from aiter.ops.triton._triton_kernels.common.activation import _get_activation_from_str
from aiter.ops.triton._triton_kernels.conv.conv2d_depthwise import (
    _conv2d_depthwise_kernel,
)
from aiter.ops.triton.utils.conv_config_utils import format_shape_key, get_conv_config
from aiter.ops.triton.utils.logger import AiterTritonLogger

_LOGGER = AiterTritonLogger()

__all__ = ["conv2d_depthwise"]


def _pair(v) -> tuple[int, int]:
    return (v, v) if isinstance(v, int) else (int(v[0]), int(v[1]))


def conv2d_depthwise(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    stride=1,
    padding=0,
    dilation=1,
    activation: str | None = None,
    glu_activation: str | None = None,
) -> torch.Tensor:
    """
    Depthwise 2-D convolution (``groups = C``) with a fused bias, activation or
    GLU, in one kernel.

    Computes ``F.conv2d(x, weight, bias, stride, padding, dilation, groups=C)``,
    then either ``act(y)`` or, with ``glu_activation``, the GLU of the two
    channel halves: ``y[:, :C/2] * glu_act(y[:, C/2:])``. The GLU form is the
    ``conv_depth`` + ``chunk`` + ``x * silu(gate)`` of the SANA ``GLUMBConv``.

    Args:
        x (torch.Tensor): Input (N, C, H, W), NCHW with a contiguous last dim.
            fp32, fp16 or bf16.
        weight (torch.Tensor): (C, 1, R, S), same dtype as x.
        bias (Optional[torch.Tensor]): C elements, same dtype as x.
        stride, padding, dilation (int or 2-tuple): As in ``F.conv2d``.
        activation (Optional[str]): Activation on the output, for example
            "silu" or "gelu". See ``_get_activation_from_str``.
        glu_activation (Optional[str]): Activation of the gate half. When set,
            C must be even and the output has C / 2 channels.

    Returns:
        torch.Tensor: Contiguous output (N, C or C / 2, OH, OW), dtype of x.

    Limits: ``padding_mode="zeros"`` only. The conv is in fp32 and rounds to the
    input dtype; the activation and the GLU product round at the same points as
    eager PyTorch.
    """
    assert x.dim() == 4, f"x must be (N, C, H, W), got {tuple(x.shape)}"
    assert x.stride(-1) == 1, "x must have a contiguous last dim (NCHW)"
    assert x.dtype in (
        torch.float32,
        torch.float16,
        torch.bfloat16,
    ), f"unsupported dtype: {x.dtype}"
    N, C, H, W = x.shape
    assert (
        weight.dim() == 4 and weight.shape[0] == C and weight.shape[1] == 1
    ), f"weight must be ({C}, 1, R, S) for a depthwise conv, got {tuple(weight.shape)}"
    assert weight.dtype == x.dtype, f"dtype mismatch: {weight.dtype} vs {x.dtype}"
    assert weight.device == x.device, "weight must be on the device of x"
    if bias is not None:
        assert bias.numel() == C, f"bias must have {C} elements, got {bias.numel()}"
        assert bias.dtype == x.dtype, f"dtype mismatch: {bias.dtype} vs {x.dtype}"
        assert bias.device == x.device, "bias must be on the device of x"
        bias = bias.contiguous()
    assert not (
        activation and glu_activation
    ), "set activation or glu_activation, not both"
    if glu_activation:
        assert C % 2 == 0, f"GLU needs an even channel count, got {C}"
    weight = weight.contiguous()

    R, S = weight.shape[2], weight.shape[3]
    sh, sw = _pair(stride)
    ph, pw = _pair(padding)
    dh, dw = _pair(dilation)
    OH = (H + 2 * ph - dh * (R - 1) - 1) // sh + 1
    OW = (W + 2 * pw - dw * (S - 1) - 1) // sw + 1
    assert OH > 0 and OW > 0, f"empty output for input {H}x{W} and kernel {R}x{S}"
    C_out = C // 2 if glu_activation else C

    _LOGGER.info(
        "CONV2D_DEPTHWISE: x=%s kernel=%dx%d stride=%s padding=%s act=%s glu=%s",
        tuple(x.shape),
        R,
        S,
        (sh, sw),
        (ph, pw),
        activation,
        glu_activation,
    )

    y = torch.empty((N, C_out, OH, OW), dtype=x.dtype, device=x.device)
    if y.numel() == 0:
        return y

    shape_key = format_shape_key(N, C, H, W, C, R, S, sh, sw, ph, pw, dh, dw)
    config = get_conv_config("CONV2D-DEPTHWISE", shape_key=shape_key, M=OH * OW)
    BLOCK_H = config.pop("BLOCK_H")
    BLOCK_W = config.pop("BLOCK_W")
    act_name = glu_activation or activation
    grid = (triton.cdiv(OH, BLOCK_H) * triton.cdiv(OW, BLOCK_W), C_out, N)
    _conv2d_depthwise_kernel[grid](
        x,
        weight,
        bias,
        y,
        C_out,
        H,
        W,
        OH,
        OW,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        y.stride(0),
        y.stride(1),
        y.stride(2),
        sh,
        sw,
        ph,
        pw,
        dh,
        dw,
        activation=_get_activation_from_str(act_name) if act_name else "",
        R=R,
        S=S,
        BLOCK_H=BLOCK_H,
        BLOCK_W=BLOCK_W,
        HAS_BIAS=bias is not None,
        USE_ACTIVATION=activation is not None,
        USE_GLU=glu_activation is not None,
        **config,
    )
    return y
