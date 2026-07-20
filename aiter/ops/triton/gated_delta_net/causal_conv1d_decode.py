# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Triton decode causal_conv1d_update (fused split q/k/v) — public host entry.

Decode-stage counterpart of the prefill ``causal_conv1d_prefill`` module: a
single-/few-token autoregressive conv update that directly outputs split
q/k/v. The ``@triton.jit`` / Gluon kernels live in
``aiter.ops.triton._triton_kernels.gated_delta_rule.decode.causal_conv1d_update_split_qkv``.
"""

import torch
import triton

from aiter.ops.triton._triton_kernels.gated_delta_rule.decode.causal_conv1d_update_split_qkv import (
    PAD_SLOT_ID,
    _causal_conv1d_update_split_qkv_kernel,
    gluon_causal_conv1d_update_split_qkv_kernel,
    gluon_causal_conv1d_update_split_qkv_kernel_notuple,
)

__all__ = ["causal_conv1d_update_split_qkv", "PAD_SLOT_ID"]


def causal_conv1d_update_split_qkv(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    key_dim: int,
    value_dim: int,
    bias: torch.Tensor | None = None,
    activation: bool | str | None = "silu",
    conv_state_indices: torch.Tensor | None = None,
    pad_slot_id: int = PAD_SLOT_ID,
    use_gluon: bool = True,
    use_gluon_notuple: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Optimized causal_conv1d_update that directly outputs split q, k, v.

    Args:
        x: Input tensor (batch, dim, seqlen) where dim = 2*key_dim + value_dim
        conv_state: Convolution state (num_cache_lines, dim, state_len)
        weight: Convolution weights (dim, width)
        key_dim: Dimension of query and key
        value_dim: Dimension of value
        bias: Optional bias (dim,)
        activation: Activation function ("silu", "swish", or None)
        conv_state_indices: Optional batch indices for continuous batching
        pad_slot_id: ID for padded slots
        use_gluon: Whether to use Gluon kernel (default: True)
        use_gluon_notuple: Use the Gluon kernel variant that avoids tuple
                           operations (default: False)

    Returns:
        Tuple of (query, key, value) tensors
    """
    if isinstance(activation, bool):
        activation = "silu" if activation is True else None
    elif activation is not None:
        assert activation in ["silu", "swish"]

    unsqueeze = x.dim() == 2
    if unsqueeze:
        x = x.unsqueeze(-1)

    batch, dim, seqlen = x.shape
    assert dim == 2 * key_dim + value_dim, f"dim {dim} != 2*{key_dim} + {value_dim}"

    _, width = weight.shape
    num_cache_lines = conv_state.size(0)

    query = torch.empty(
        (batch, key_dim, seqlen),
        dtype=x.dtype,
        device=x.device,
    )
    key = torch.empty(
        (batch, key_dim, seqlen),
        dtype=x.dtype,
        device=x.device,
    )
    value = torch.empty(
        (batch, value_dim, seqlen),
        dtype=x.dtype,
        device=x.device,
    )

    stride_state_indices = (
        conv_state_indices.stride(0) if conv_state_indices is not None else 0
    )
    state_len = width - 1
    np2_statelen = triton.next_power_of_2(state_len)

    BLOCK_N = 256
    grid = (batch, triton.cdiv(dim, BLOCK_N))

    if use_gluon:
        kernel_fn = (
            gluon_causal_conv1d_update_split_qkv_kernel_notuple
            if use_gluon_notuple
            else gluon_causal_conv1d_update_split_qkv_kernel
        )
    else:
        kernel_fn = _causal_conv1d_update_split_qkv_kernel

    kernel_fn[grid](
        x_ptr=x,
        w_ptr=weight,
        bias_ptr=bias,
        conv_state_ptr=conv_state,
        conv_state_indices_ptr=conv_state_indices,
        q_ptr=query,
        k_ptr=key,
        v_ptr=value,
        key_dim=key_dim,
        value_dim=value_dim,
        batch=batch,
        dim=dim,
        seqlen=seqlen,
        state_len=state_len,
        num_cache_lines=num_cache_lines,
        stride_x_seq=x.stride(0),
        stride_x_dim=x.stride(1),
        stride_x_token=x.stride(2),
        stride_w_dim=weight.stride(0),
        stride_w_width=weight.stride(1),
        stride_conv_state_seq=conv_state.stride(0),
        stride_conv_state_dim=conv_state.stride(1),
        stride_conv_state_tok=conv_state.stride(2),
        stride_state_indices=stride_state_indices,
        stride_q_seq=query.stride(0),
        stride_q_dim=query.stride(1),
        stride_q_token=query.stride(2),
        stride_k_seq=key.stride(0),
        stride_k_dim=key.stride(1),
        stride_k_token=key.stride(2),
        stride_v_seq=value.stride(0),
        stride_v_dim=value.stride(1),
        stride_v_token=value.stride(2),
        pad_slot_id=pad_slot_id,
        HAS_BIAS=bias is not None,
        KERNEL_WIDTH=width,
        SILU_ACTIVATION=activation in ["silu", "swish"],
        IS_CONTINUOUS_BATCHING=conv_state_indices is not None,
        NP2_STATELEN=np2_statelen,
        USE_PAD_SLOT=pad_slot_id is not None,
        BLOCK_N=BLOCK_N,
        num_warps=2 if use_gluon else 4,
    )

    if unsqueeze:
        query = query.squeeze(-1)
        key = key.squeeze(-1)
        value = value.squeeze(-1)

    return query, key, value
