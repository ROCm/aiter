# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import torch
import triton

from aiter.ops.triton._triton_kernels.fusions.silu_and_mul_backward import (
    _silu_and_mul_backward_kernel,
)
from aiter.ops.triton.utils.config_utils import (
    load_config_json,
    resolve_config_dir,
    select_leq_config,
)
from aiter.ops.triton.utils.logger import AiterTritonLogger

_LOGGER = AiterTritonLogger()

__all__ = ["silu_and_mul_backward"]


def _get_config(n_cols: int) -> dict:
    config_dir = resolve_config_dir("fusions", "SILU-AND-MUL-BACKWARD")
    configs = load_config_json(f"{config_dir}/DEFAULT.json", required=True)
    return select_leq_config(configs, n_cols)


def silu_and_mul_backward(
    grad_output: torch.Tensor,
    x: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute the input gradient of ``silu(gate) * up``.

    Args:
        grad_output: Gradient tensor shaped like ``x`` with its last dimension
            halved, on the same device and with the same dtype. Strided tensors
            are supported.
        x: Contiguous FP16, BF16, or FP32 tensor whose last dimension contains
            concatenated ``[gate, up]`` values.
        out: Optional contiguous destination with the same shape, dtype, and
            device as ``x``.

    Returns:
        The gradient with respect to ``x``. SiLU and both gradient products are
        evaluated in FP32 before conversion to ``x.dtype``.

    Width thresholds select ``BLOCK_M``, ``BLOCK_N``, ``num_warps``, and
    ``num_stages`` from the current architecture's ``DEFAULT.json``.
    """
    assert x.is_contiguous(), "x must be contiguous"
    assert x.dtype in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ), f"unsupported dtype: {x.dtype}"
    last = x.size(-1)
    assert last > 0, "x last dimension must be non-zero"
    assert last % 2 == 0, "x last dimension must be even"
    n_cols = last // 2
    expected_shape = (*x.shape[:-1], n_cols)
    assert (
        grad_output.shape == expected_shape
    ), f"grad_output shape must be {expected_shape}, got {tuple(grad_output.shape)}"
    assert grad_output.dtype == x.dtype, "grad_output dtype must match x dtype"
    assert grad_output.device == x.device, "grad_output device must match x device"

    if out is None:
        out = torch.empty_like(x)
    else:
        assert out.shape == x.shape, "out shape must match x shape"
        assert out.dtype == x.dtype, "out dtype must match x dtype"
        assert out.device == x.device, "out device must match x device"
        assert out.is_contiguous(), "out must be contiguous"

    flat_x = x.reshape(-1, last)
    flat_grad_output = grad_output.reshape(-1, n_cols)
    flat_out = out.reshape(-1, last)
    n_rows = flat_x.size(0)
    if n_rows == 0:
        return out

    _LOGGER.info(
        "SILU_AND_MUL_BACKWARD: x=%s grad=%s",
        tuple(x.shape),
        tuple(grad_output.shape),
    )
    config = _get_config(n_cols)
    block_m = min(config.pop("BLOCK_M"), triton.next_power_of_2(n_rows))
    block_n = config.pop("BLOCK_N")
    grid = (triton.cdiv(n_rows, block_m), triton.cdiv(n_cols, block_n))
    _silu_and_mul_backward_kernel[grid](
        flat_grad_output,
        flat_x,
        flat_out,
        n_rows,
        n_cols,
        flat_grad_output.stride(0),
        flat_grad_output.stride(1),
        flat_x.stride(0),
        flat_x.stride(1),
        flat_out.stride(0),
        flat_out.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        **config,
    )
    return out
