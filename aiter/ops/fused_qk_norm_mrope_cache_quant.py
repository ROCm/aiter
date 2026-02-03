# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from torch import Tensor
from ..jit.core import compile_ops
from typing import List, Optional


@compile_ops("module_fused_qk_norm_mrope_cache_quant_shuffle")
def fused_qk_norm_mrope_3d_cache_pts_quant_shuffle(
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
    q: Tensor,
    k_cache: Tensor,
    v_cache: Tensor,
    kv_loc: Tensor,
    k_scale: Tensor,
    v_scale: Tensor,
    k_out: Optional[Tensor],
    v_out: Optional[Tensor],
    return_kv: bool,
    use_shuffle_layout: bool,
    block_size: int,
    x: int,
) -> None: ...
