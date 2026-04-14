# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""High-level FlyDSL Linear Attention APIs."""

from __future__ import annotations


import torch


from .kernels.gdr_decode import create_shuffle_gdr_decode_kernel
from .kernels.tensor_shim import get_dtype_str

__all__ = [
    "flydsl_gdr_decode",
]


def get_default_kwargs(batch_size, seq_length):
    d = {}
    b_to_vs = {
        1: 4,
        2: 4,
        3: 4,
        4: 2,
        5: 2,
        6: 2,
        7: 2,
        8: 2,
        9: 2,
        10: 2,
        11: 1,
    }
    if b_to_vs.get(batch_size, None) is not None:
        d["NUM_BLOCKS_PER_V_DIM"] = b_to_vs[batch_size]
    return d


def flydsl_gdr_decode(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    A_log: torch.Tensor,
    indices: torch.Tensor,
    state: torch.Tensor,
    out: torch.Tensor,
    use_qk_l2norm: bool,
    need_shuffle_state: bool,
    stream: torch.cuda.Stream = torch.cuda.current_stream(),
):
    if need_shuffle_state:
        state_ = state.permute(0, 1, 3, 2).contiguous()
    else:
        state_ = state
    seq_length, batch_size, num_k_heads, head_k_dim = query.shape
    num_v_heads = value.shape[-2]
    head_v_dim = value.shape[-1]
    kwargs = get_default_kwargs(batch_size, seq_length)
    state_stride_0, state_stride_1, state_stride_2, state_stride_3 = state.stride()
    exe = create_shuffle_gdr_decode_kernel(
        state_stride_0,
        state_stride_1,
        state_stride_2,
        state_stride_3,
        get_dtype_str(query.dtype),
        get_dtype_str(A_log.dtype),
        seq_length,
        num_k_heads,
        num_v_heads,
        head_k_dim,
        head_v_dim,
        use_qk_l2norm,
        **kwargs,
    )
    # exe_compiled = exe.compile(
    #     query,
    #     key,
    #     value,
    #     a,
    #     b,
    #     dt_bias,
    #     A_log,
    #     indices,
    #     state_,
    #     out,
    #     batch_size,
    #     stream,
    # )
    exe(
        query,
        key,
        value,
        a,
        b,
        dt_bias,
        A_log,
        indices,
        state_,
        out,
        batch_size,
        stream,
    )
    if need_shuffle_state:
        state_ = state_.permute(0, 1, 3, 2).contiguous()
        state.copy_(state_)
