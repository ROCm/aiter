# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""JIT wrapper for the gfx1201 SageAttention HIP core."""

import torch
from torch import Tensor

from aiter.jit.core import compile_ops


@compile_ops(
    "module_gfx1201_sage_attention",
    fc_name="gfx1201_sage_attention_hip",
    develop=True,
)
def gfx1201_sage_attention_hip(
    q_int8: Tensor,
    k_int8: Tensor,
    v_fp8: Tensor,
    q_scale: Tensor,
    k_scale: Tensor,
    v_scale: Tensor,
    out: Tensor,
    batch_size: int,
    padded_seq_len: int,
    valid_seq_len: int,
    num_heads: int,
) -> None:
    """Launch the gfx1201 SageAttention HIP core into a preallocated BF16 output."""


def launch_hip_sage_core(
    q_int8: torch.Tensor,
    k_int8: torch.Tensor,
    v_fp8: torch.Tensor,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    out: torch.Tensor,
    batch_size: int,
    padded_seq_len: int,
    valid_seq_len: int,
    num_heads: int,
) -> None:
    """Launch the HIP core into ``out`` (bf16 ``[B, S_pad, H, D]``)."""
    if out.dtype != torch.bfloat16:
        raise TypeError("gfx1201_sage_attention currently writes bf16 only")
    gfx1201_sage_attention_hip(
        q_int8,
        k_int8,
        v_fp8,
        q_scale,
        k_scale,
        v_scale,
        out,
        int(batch_size),
        int(padded_seq_len),
        int(valid_seq_len),
        int(num_heads),
    )
