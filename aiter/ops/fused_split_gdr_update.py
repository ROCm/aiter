# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from typing import Optional

import torch
from torch import Tensor

from ..jit.core import compile_ops

MD_NAME = "module_fused_split_gdr_update"


@compile_ops(
    "module_fused_split_gdr_update", fc_name="fused_split_gdr_update", develop=True
)
def _fused_split_gdr_update(
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
    softplus_beta: float,
    softplus_threshold: float,
    scale: float,
    use_qk_l2norm_in_kernel: bool,
    output: Tensor,
) -> None: ...


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
    # The C side no longer allocates memory (Python owns all I/O). Pre-allocate
    # the output buffer and default index tensor here, then forward to the
    # de-torched binding which writes into `output` in-place.
    B = mixed_qkv.shape[0]
    T = mixed_qkv.shape[2]
    HV = num_heads_v
    V = head_dim

    if output is None:
        output = torch.empty(
            (B, T, HV, V), dtype=mixed_qkv.dtype, device=mixed_qkv.device
        )

    if initial_state_indices is None or initial_state_indices.numel() == 0:
        initial_state_indices = torch.zeros(
            (B,), dtype=torch.int32, device=mixed_qkv.device
        )

    if initial_state_source is None:
        initial_state_source = torch.empty(
            0, dtype=torch.float32, device=mixed_qkv.device
        )

    _fused_split_gdr_update(
        mixed_qkv,
        A_log,
        a,
        dt_bias,
        b_gate,
        initial_state_source,
        initial_state_indices,
        key_dim,
        value_dim,
        num_heads_qk,
        num_heads_v,
        head_dim,
        softplus_beta,
        softplus_threshold,
        scale,
        use_qk_l2norm_in_kernel,
        output,
    )
    return output
