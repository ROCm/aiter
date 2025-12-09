# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

from torch import Tensor
from ..jit.core import compile_ops
from typing import List


@compile_ops("module_fused_mrope_rms")
def fused_mrope_3d_rms(
    qkv: Tensor,
    qw: Tensor,
    kw: Tensor,
    cos_sin: Tensor,
    positions: Tensor,
    num_tokens: int,
    num_heads_q: int,
    num_heads_k: int,
    num_heads_v: int,
    head_size: int,
    is_neox_style: bool,
    mrope_section_: List[int],
    is_interleaved: bool,
    eps: float,
) -> None: ...


@compile_ops("module_fused_mrope_rms")
def fused_rope_rms(
    qkv: Tensor,
    qw: Tensor,
    kw: Tensor,
    cos_sin: Tensor,
    positions: Tensor,
    num_tokens: int,
    num_heads_q: int,
    num_heads_k: int,
    num_heads_v: int,
    head_size: int,
    is_neox_style: bool,
    eps: float,
) -> None: ...
