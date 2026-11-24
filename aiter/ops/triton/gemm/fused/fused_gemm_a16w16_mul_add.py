# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import torch
import triton

from aiter.ops.triton._triton_kernels.gemm.fused.fused_gemm_a16w16_mul_add import (
    _fused_gemm_a16w16_mul_add_kernel,
    _get_config,
)
from aiter.ops.triton.utils.device_info import get_num_xcds
from aiter.ops.triton.utils.gemm_config_utils import compute_splitk_params
from aiter.ops.triton.utils.logger import AiterTritonLogger

_LOGGER = AiterTritonLogger()


def _fits_int32(t: torch.Tensor) -> bool:
    """True if the largest element offset of t fits in int32."""
    return sum((n - 1) * s for n, s in zip(t.shape, t.stride())) < 2**31


def _mul_add_operand(v, B: int, S: int, N: int, device):
    """Return (value, is_scalar, strides) for one epilogue operand."""
    if isinstance(v, (float, int)):
        return float(v), True, (0, 0, 0)
    assert isinstance(
        v, torch.Tensor
    ), f"a and b must be scalars or tensors, got {type(v)}"
    assert v.device == device, f"a and b must be on {device}, got {v.device}"
    v = v.broadcast_to((B, S, N))
    return v, False, v.stride()


def fused_gemm_a16w16_mul_add(
    x: torch.Tensor,
    w: torch.Tensor,
    a: torch.Tensor | float,
    b: torch.Tensor | float,
    bias: torch.Tensor | None = None,
    dtype: torch.dtype | None = torch.bfloat16,
    y: torch.Tensor | None = None,
    config: dict | None = None,
):
    """
    Computes out = a * (X @ W^T + bias) + b in one kernel, with 16 bit inputs.

    A diffusion transformer block uses this for the gated residual after the
    attention or MLP output projection: out = residual + gate * Linear(x).

    Args:
        x (torch.Tensor): Input with shape (M, K) or (B, S, K). The rows can
            have a stride, for example a slice of a longer sequence.
        w (torch.Tensor): Weight matrix with shape (N, K), internally transposed.
        a (torch.Tensor | float): Multiplier. A scalar or a tensor that
            broadcasts to (B, S, N), for example a gate with shape (B, 1, N).
            For a 2D x, B is 1 and S is M.
        b (torch.Tensor | float): Addend. Same rules as a, for example a
            residual with shape (B, S, N).
        bias (Optional[torch.Tensor]): Bias vector with shape (N,).
        dtype (Optional[torch.dtype]): Output datatype (BF16 or FP16).
        y (Optional[torch.Tensor]): Pre-allocated output with M * N elements
            that can be viewed as (M, N).
        config (Optional[dict]): Kernel tuning parameters. The default comes
            from FUSED-GEMM-A16W16-MUL_ADD on gfx1151 and from GEMM-A16W16
            on other arches.

    Returns:
        torch.Tensor: Output with shape x.shape[:-1] + (N,).

    Limits: the epilogue runs in fp32 and rounds once. Split-K is not
    supported, so NUM_KSPLIT is always 1. All element offsets must fit in
    int32.
    """
    assert x.dim() in (2, 3), f"x must be (M, K) or (B, S, K), got {tuple(x.shape)}"
    assert (
        x.shape[-1] == w.shape[1]
    ), f"x has K={x.shape[-1]} and w is (N, K)={tuple(w.shape)}, K must match"
    assert x.dtype in (
        torch.float16,
        torch.bfloat16,
    ), f"Activations (x) must be fp16 or bf16, got {x.dtype}"
    assert w.dtype in (
        torch.float16,
        torch.bfloat16,
    ), f"Weights (w) must be fp16 or bf16, got {w.dtype}"

    _LOGGER.info(
        "FUSED_GEMM_A16W16_MUL_ADD: x=%s w=%s",
        tuple(x.shape),
        tuple(w.shape),
    )

    N, K = w.shape
    if x.dim() == 2:
        B, S = 1, x.shape[0]
        stride_ab = 0
        stride_as, stride_ak = x.stride()
    else:
        B, S = x.shape[0], x.shape[1]
        stride_ab, stride_as, stride_ak = x.stride()
    M = B * S

    a, is_a_scalar, stride_a = _mul_add_operand(a, B, S, N, x.device)
    b, is_b_scalar, stride_b = _mul_add_operand(b, B, S, N, x.device)

    # The kernel reads the bias with unit stride, as every other Triton GEMM
    # in aiter does. An (N,) view with a stride needs a copy first.
    if bias is not None:
        bias = bias.contiguous()

    w = w.T

    if y is None:
        y = torch.empty((*x.shape[:-1], N), dtype=dtype, device=x.device)
    y2d = y.view(M, N)
    for t in (x, w, y2d) + tuple(v for v in (a, b) if isinstance(v, torch.Tensor)):
        assert _fits_int32(t), "The kernel uses int32 offsets, tensors are too large"

    if config is None:
        config, _ = _get_config(M, N, K)
    else:
        config = dict(config)
    config["NUM_KSPLIT"] = 1
    config = compute_splitk_params(config, K)
    config.pop("NUM_KSPLIT")
    config.pop("SPLITK_BLOCK_SIZE")

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    _fused_gemm_a16w16_mul_add_kernel[grid](
        x,
        w,
        bias,
        y2d,
        a,
        b,
        M,
        N,
        K,
        S,
        stride_ab,
        stride_as,
        stride_ak,
        w.stride(0),
        w.stride(1),
        y2d.stride(0),
        y2d.stride(1),
        *stride_a,
        *stride_b,
        ADD_BIAS=(bias is not None),
        IS_A_SCALAR=is_a_scalar,
        IS_B_SCALAR=is_b_scalar,
        NUM_XCDS=get_num_xcds(),
        **config,
    )

    return y
