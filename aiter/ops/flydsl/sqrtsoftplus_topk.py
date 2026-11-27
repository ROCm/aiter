# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""``topk_gating(score_func="sqrtsoftplus")`` on gfx950 for the DeepSeek-V4 / V4.1
router shapes (384 experts top-6, 128 experts top-3), through the FlyDSL kernel
in ``kernels/topk/sqrtsoftplus_topk.py``. Outputs are bit for bit those of the
HIP ``topk_gating_kernel_opt``. One wave a row: on MI355X in a CUDA graph a call
takes 2.4 instead of 2.9 us (384 / 6) and 2.0 instead of 2.5 us (128 / 3) at
8..256 rows, and is also faster up to 16384 rows."""

import torch

from ...jit.utils.chip_info import get_gfx_runtime
from .kernels.tensor_shim import _run_compiled, current_stream_arg
from .kernels.topk.sqrtsoftplus_topk import (
    SQRTSOFTPLUS_TOPK_SHAPES,
    build_sqrtsoftplus_topk_module,
)
from .kernels.topk.topk_launch import _launcher_on


def sqrtsoftplus_topk_supported(
    topk_weights: torch.Tensor,
    topk_indices: torch.Tensor,
    gating_output: torch.Tensor,
    correction_bias: torch.Tensor | None,
    need_renorm: bool,
    routed_scaling_factor: float,
    score_func: str,
    num_shared_experts: int,
    num_valid_rows: torch.Tensor | None,
) -> bool:
    """Whether this ``topk_gating`` call is the kernel's. Metadata only, nothing is read."""
    if score_func != "sqrtsoftplus" or num_shared_experts != 0:
        return False
    if type(need_renorm) is not bool or type(routed_scaling_factor) not in (int, float):
        return False
    tensors = (topk_weights, topk_indices, gating_output, correction_bias)
    if any(
        not isinstance(t, torch.Tensor)
        or t.layout != torch.strided
        or t.is_neg()
        or t.is_conj()
        for t in tensors
    ):
        return False
    if not gating_output.is_cuda or any(
        t.device != gating_output.device for t in tensors
    ):
        return False
    if gating_output.dim() != 2 or topk_indices.dim() != 2 or topk_weights.dim() != 2:
        return False
    rows, experts = gating_output.shape
    topk = topk_indices.shape[1]
    if (experts, topk) not in SQRTSOFTPLUS_TOPK_SHAPES:
        return False
    if (
        rows < 1
        or gating_output.dtype != torch.bfloat16
        or gating_output.stride(1) != 1
        or (rows > 1 and gating_output.stride(0) < experts)
        or correction_bias.shape != (experts,)
        or correction_bias.dtype != torch.float32
        or correction_bias.stride(0) != 1
        or topk_indices.shape[0] != rows
        or topk_weights.shape[0] != rows
        or topk_weights.shape[1] < topk
        or topk_weights.dtype != torch.float32
        or topk_indices.dtype != torch.int32
        or topk_weights.stride(1) != 1
        or topk_indices.stride(1) != 1
        or (rows > 1 and topk_weights.stride(0) != topk_indices.stride(0))
        or (rows > 1 and topk_indices.stride(0) < topk)
        # buffer offsets are 32-bit
        or rows * max(gating_output.stride(0), experts) >= 2**30
    ):
        return False
    if num_valid_rows is not None and (
        not isinstance(num_valid_rows, torch.Tensor)
        or num_valid_rows.dtype != torch.int32
        or num_valid_rows.numel() < 1
        or num_valid_rows.device != gating_output.device
        or not num_valid_rows.is_contiguous()
    ):
        return False
    return get_gfx_runtime() == "gfx950"


def sqrtsoftplus_topk(
    topk_weights: torch.Tensor,
    topk_indices: torch.Tensor,
    gating_output: torch.Tensor,
    correction_bias: torch.Tensor,
    need_renorm: bool = True,
    routed_scaling_factor: float = 1.0,
    num_valid_rows: torch.Tensor | None = None,
) -> None:
    """Writes ``topk_weights`` / ``topk_indices`` in place, for a call
    ``sqrtsoftplus_topk_supported`` accepts."""
    rows, experts = gating_output.shape
    topk = topk_indices.shape[1]
    launcher = build_sqrtsoftplus_topk_module(
        experts,
        topk,
        num_valid_rows is not None,
        float(routed_scaling_factor),
        bool(need_renorm),
    )
    # Unread without a count; any device tensor stands in for the argument.
    nvr = num_valid_rows if num_valid_rows is not None else correction_bias
    # One launcher a GPU: a FlyDSL build is loaded on the GPU that first ran it.
    exe = _launcher_on(launcher, gating_output.device.index)
    _run_compiled(
        exe,
        gating_output.data_ptr(),
        correction_bias.data_ptr(),
        topk_weights.data_ptr(),
        topk_indices.data_ptr(),
        nvr.data_ptr(),
        gating_output.stride(0) if rows > 1 else experts,
        topk_indices.stride(0) if rows > 1 else topk,
        rows,
        current_stream_arg(gating_output.device),
    )
