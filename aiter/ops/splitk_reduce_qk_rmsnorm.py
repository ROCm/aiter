# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import torch
from torch import Tensor

from ..jit.core import compile_ops

SPLITK_PLANES = 6
Q_LORA_RANK = 2048
KV_LORA_RANK = 512
ROPE_DIM = 64
OUT_DIM = Q_LORA_RANK + KV_LORA_RANK + ROPE_DIM

__all__ = ["splitk_reduce_qk_rmsnorm"]


@compile_ops(
    "module_splitk_reduce_qk_rmsnorm",
    fc_name="splitk_reduce_qk_rmsnorm",
    develop=True,
)
def _splitk_reduce_qk_rmsnorm(
    partial: Tensor,
    out: Tensor,
    q_out: Tensor,
    k_out: Tensor,
    q_weight: Tensor,
    k_weight: Tensor,
    q_eps: float,
    k_eps: float,
) -> None:
    """Internal binding; valid calls require the operand device to be current.

    The public wrapper sets that context before compile_ops binds the HIP stream.
    """


def _check(
    tensor: Tensor, name: str, shape: tuple, dtype: torch.dtype, alignment: int = 16
) -> None:
    if tuple(tensor.shape) != shape or tensor.dtype != dtype:
        raise ValueError(
            f"{name} must be {dtype} {list(shape)}, got {tensor.dtype} {list(tensor.shape)}"
        )
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    if not tensor.is_cuda:
        raise ValueError(f"{name} must be a GPU tensor")
    if tensor.data_ptr() % alignment:
        raise ValueError(f"{name} must be {alignment}-byte aligned")


def splitk_reduce_qk_rmsnorm(
    partial: Tensor,
    q_weight: Tensor,
    q_eps: float,
    k_weight: Tensor,
    k_eps: float,
    out: Tensor | None = None,
    q_out: Tensor | None = None,
    k_out: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Reduce a six-way split-K MLA input projection and apply its q/kv RMSNorm.

    ``partial`` is ``[6, M, 2624]`` fp32: plane ``p`` is the GEMM's partial sum
    over K range ``p``. The planes are added in plane order, starting from
    ``p0 + 0``, and rounded to bf16 once, so the result is deterministic. The
    2624 columns are q_lora (2048), kv_lora (512) and the rotary key (64).

    Returns ``(out, q_out, k_out)``, all bf16: ``out`` ``[M, 2624]`` is the sum,
    ``q_out`` ``[M, 2048]`` is ``RMSNorm(out[:, :2048]) * q_weight`` and ``k_out``
    ``[M, 512]`` is ``RMSNorm(out[:, 2048:2560]) * k_weight``, computed as
    ``fused_qk_rmsnorm`` computes them. Output tensors are allocated when not
    given. All tensors must be on the same GPU and contiguous. The partial
    buffer must be 32-byte aligned; BF16 operands and outputs must be 16-byte
    aligned. Outputs must not overlap each other or any input.
    """
    if partial.dim() != 3:
        raise ValueError(f"partial must be [6, M, 2624], got {list(partial.shape)}")
    m = partial.shape[1]
    _check(partial, "partial", (SPLITK_PLANES, m, OUT_DIM), torch.float32, 32)
    _check(q_weight, "q_weight", (Q_LORA_RANK,), torch.bfloat16)
    _check(k_weight, "k_weight", (KV_LORA_RANK,), torch.bfloat16)
    if out is None:
        out = partial.new_empty((m, OUT_DIM), dtype=torch.bfloat16)
    if q_out is None:
        q_out = partial.new_empty((m, Q_LORA_RANK), dtype=torch.bfloat16)
    if k_out is None:
        k_out = partial.new_empty((m, KV_LORA_RANK), dtype=torch.bfloat16)
    _check(out, "out", (m, OUT_DIM), torch.bfloat16)
    _check(q_out, "q_out", (m, Q_LORA_RANK), torch.bfloat16)
    _check(k_out, "k_out", (m, KV_LORA_RANK), torch.bfloat16)
    tensors = (partial, q_weight, k_weight, out, q_out, k_out)
    if any(t.device != partial.device for t in tensors):
        raise ValueError("all tensors must be on the same device")

    for index, output in enumerate((out, q_out, k_out)):
        if output.numel() == 0:
            continue
        start = output.data_ptr()
        end = start + output.numel() * output.element_size()
        for other in (partial, q_weight, k_weight, *(out, q_out, k_out)[:index]):
            other_start = other.data_ptr()
            other_end = other_start + other.numel() * other.element_size()
            if start < other_end and other_start < end:
                raise ValueError("outputs must not overlap inputs or other outputs")

    # develop=True captures the current device's stream before entering C++.
    # Select the operand device here so the native guard and stream agree.
    with torch.cuda.device(partial.device):
        _splitk_reduce_qk_rmsnorm(
            partial, out, q_out, k_out, q_weight, k_weight, float(q_eps), float(k_eps)
        )
    return out, q_out, k_out
