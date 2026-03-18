# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from typing import Optional

from torch import Tensor

from ..jit.core import compile_ops

MD_NAME = "module_fused_split_gdr_update"


@compile_ops("module_fused_split_gdr_update")
def fused_split_gdr_update(
    mixed_qkv: Tensor,
    A_log: Tensor,
    a: Tensor,
    dt_bias: Tensor,
    b_gate: Tensor,
    initial_state_source: Tensor,
    initial_state_indices: Tensor,
    key_dim: int,
    value_dim: int,
    num_heads_qk: int,
    num_heads_v: int,
    head_dim: int,
    softplus_beta: float = 1.0,
    softplus_threshold: float = 20.0,
    scale: float = -1.0,
    use_qk_l2norm_in_kernel: bool = True,
    output: Optional[Tensor] = None,
) -> Tensor:
    """
    HIP fused split GDR decode update (ksplit4_db backend).

    Args:
        mixed_qkv: [B, 2*key_dim+value_dim, T], bfloat16.
        A_log: [HV], float32.
        a: [B*T, HV], bfloat16.
        dt_bias: [HV], bfloat16.
        b_gate: [B*T, HV], bfloat16.
        initial_state_source: [N, HV, K/4, V, 4], float32 swizzled state, updated in-place.
        initial_state_indices: [B], int32 indices into initial_state_source.
    """
    ...
