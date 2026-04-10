# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2024, Tri Dao.
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Launchers for **causal conv1d update** fast paths.

``causal_conv1d_update_fast`` updates ``conv_state`` **in place** inside the Triton kernel
(the "update" in the name), then writes the convolution output into ``x``/``out`` as in vLLM.
"""

from __future__ import annotations

import torch
import triton

from aiter.ops.triton._triton_kernels.causal_conv1d import PAD_SLOT_ID
from aiter.ops.triton._triton_kernels.causal_conv1d_update_fast import (
    _causal_conv1d_update_fast_kernel,
    _reshape_causal_conv1d_update_fast_kernel,
)


def _default_conv_state_indices(batch: int, device: torch.device) -> torch.Tensor:
    return torch.arange(batch, device=device, dtype=torch.int32)


def causal_conv1d_update_fast(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation: bool | str | None = None,
    conv_state_indices: torch.Tensor | None = None,
    num_accepted_tokens: torch.Tensor | None = None,
    query_start_loc: torch.Tensor | None = None,
    max_query_len: int = -1,
    pad_slot_id: int = PAD_SLOT_ID,
    block_idx_last_scheduled_token: torch.Tensor | None = None,
    initial_state_idx: torch.Tensor | None = None,
    validate_data: bool = False,
) -> torch.Tensor:
    assert (
        num_accepted_tokens is None
    ), f"num_accepted_tokens must be None, got {num_accepted_tokens}"
    assert (
        query_start_loc is None
    ), f"query_start_loc must be None, got {query_start_loc}"
    if validate_data:
        assert pad_slot_id is not None
        assert x.stride(1) == 1
    if isinstance(activation, bool):
        activation = "silu" if activation is True else None
    elif activation is not None:
        assert activation in ["silu", "swish"]

    original_x_dtype = x.dtype
    x = x.to(conv_state.dtype)
    unsqueeze = query_start_loc is None and x.dim() == 2
    if unsqueeze:
        x = x.unsqueeze(-1)
    if query_start_loc is None:
        batch, dim, seqlen = x.shape
    else:
        assert conv_state_indices is not None
        batch = conv_state_indices.size(0)
        dim = x.size(1)
        seqlen = max_query_len
    _, width = weight.shape
    num_cache_lines, _, state_len = conv_state.size()

    if conv_state_indices is None:
        conv_state_indices = _default_conv_state_indices(batch, x.device)

    if validate_data:
        assert dim == weight.size(0)
        assert conv_state.stride(-2) == 1, (
            f"ERROR: expect contiguous along feat-dim of conv_state "
            f"(currently stride={conv_state.stride()})"
        )
        assert state_len >= width - 1
        assert dim == conv_state.size(1)
        assert (batch,) == conv_state_indices.shape
        assert num_cache_lines >= batch
        assert weight.stride(1) == 1

    out = x
    stride_w_dim, stride_w_width = weight.stride()

    if query_start_loc is None:
        stride_x_seq, stride_x_dim, stride_x_token = x.stride()
        stride_o_seq, stride_o_dim, stride_o_token = out.stride()
    else:
        stride_x_token, stride_x_dim = x.stride()
        stride_x_seq = 0
        stride_o_token, stride_o_dim = out.stride()
        stride_o_seq = 0

    stride_istate_seq, stride_istate_dim, stride_istate_token = conv_state.stride()
    stride_state_indices = conv_state_indices.stride(0)
    if num_accepted_tokens is not None:
        state_len = width - 1 + (seqlen - 1)
    else:
        state_len = width - 1
    np2_statelen = triton.next_power_of_2(state_len)

    def grid(META):
        return (batch, triton.cdiv(dim, META["BLOCK_N"]))

    _causal_conv1d_update_fast_kernel[grid](
        x,
        weight,
        bias,
        conv_state,
        conv_state_indices,
        block_idx_last_scheduled_token,
        initial_state_idx,
        out,
        batch,
        dim,
        seqlen,
        state_len,
        num_cache_lines,
        stride_x_seq,
        stride_x_dim,
        stride_x_token,
        stride_w_dim,
        stride_w_width,
        stride_istate_seq,
        stride_istate_dim,
        stride_istate_token,
        stride_state_indices,
        stride_o_seq,
        stride_o_dim,
        stride_o_token,
        pad_slot_id,
        HAS_BIAS=bias is not None,
        KERNEL_WIDTH=width,
        SILU_ACTIVATION=activation in ["silu", "swish"],
        IS_APC_ENABLED=block_idx_last_scheduled_token is not None,
        NP2_STATELEN=np2_statelen,
        USE_PAD_SLOT=pad_slot_id is not None,
        BLOCK_N=256,
    )
    if unsqueeze:
        out = out.squeeze(-1)
    return out.to(original_x_dtype)


def fused_reshape_causal_conv1d_update_fast(
    x: torch.Tensor,
    num_actual_tokens: int,
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    ba: torch.Tensor,
    z_out: torch.Tensor,
    core_attn_out: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation: bool | str | None = None,
    conv_state_indices: torch.Tensor | None = None,
    num_accepted_tokens: torch.Tensor | None = None,
    query_start_loc: torch.Tensor | None = None,
    max_query_len: int = -1,
    pad_slot_id: int = PAD_SLOT_ID,
    block_idx_last_scheduled_token: torch.Tensor | None = None,
    initial_state_idx: torch.Tensor | None = None,
    validate_data: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert (
        num_accepted_tokens is None
    ), f"num_accepted_tokens must be None, got {num_accepted_tokens}"
    assert (
        query_start_loc is None
    ), f"query_start_loc must be None, got {query_start_loc}"
    assert z_out.is_contiguous(), "z_out should be contiguous"
    assert core_attn_out.is_contiguous(), "core_attn_out should be contiguous"
    x = x.view(x.shape[0], -1)
    ba = ba.view(ba.shape[0], -1)
    assert z_out.size() == core_attn_out.size()
    original_z_shape = z_out.shape
    num_tokens = z_out.shape[0]
    z_out = z_out.view(original_z_shape[0], -1)
    core_attn_out = core_attn_out.view(original_z_shape[0], -1)
    if validate_data:
        assert pad_slot_id is not None
        assert x.stride(1) == 1
    if isinstance(activation, bool):
        activation = "silu" if activation is True else None
    elif activation is not None:
        assert activation in ["silu", "swish"]

    original_x_dtype = x.dtype
    x = x.to(conv_state.dtype)
    unsqueeze = query_start_loc is None and x.dim() == 2
    if unsqueeze:
        x = x.unsqueeze(-1)
    _, qkvz_dim, seqlen = x.shape
    batch = num_actual_tokens
    _, width = weight.shape
    head_dim = head_k_dim + head_k_dim + head_v_dim * num_v_heads // num_k_heads
    head_qkvz_dim = head_dim + head_v_dim * num_v_heads // num_k_heads
    dim = num_k_heads * head_dim
    expected_qkvz_dim = num_k_heads * head_qkvz_dim
    assert (
        qkvz_dim == expected_qkvz_dim
    ), f"ERROR: expect qkvz_dim to be {expected_qkvz_dim}, got {qkvz_dim}"
    num_cache_lines, _, state_len = conv_state.size()

    if conv_state_indices is None:
        conv_state_indices = _default_conv_state_indices(batch, x.device)

    if validate_data:
        assert dim == weight.size(0)
        assert conv_state.stride(-2) == 1, (
            f"ERROR: expect contiguous along feat-dim of conv_state "
            f"(currently stride={conv_state.stride()})"
        )
        assert state_len >= width - 1
        assert dim == conv_state.size(1)
        assert (batch,) == conv_state_indices.shape
        assert num_cache_lines >= batch
        assert weight.stride(1) == 1

    out = torch.empty((num_actual_tokens, dim, seqlen), dtype=x.dtype, device=x.device)
    b_out = torch.empty(
        (num_actual_tokens, num_v_heads), dtype=ba.dtype, device=ba.device
    )
    a_out = torch.empty(
        (num_actual_tokens, num_v_heads), dtype=ba.dtype, device=ba.device
    )
    stride_w_dim, stride_w_width = weight.stride()

    if query_start_loc is None:
        stride_x_seq, stride_x_dim, stride_x_token = x.stride()
        stride_o_seq, stride_o_dim, stride_o_token = out.stride()
    else:
        stride_x_token, stride_x_dim = x.stride()
        stride_x_seq = 0
        stride_o_token, stride_o_dim = out.stride()
        stride_o_seq = 0

    stride_z_seq = z_out.stride(0)
    stride_ba_seq, stride_ba_token = ba.stride()
    stride_b_seq = b_out.stride(0)

    stride_istate_seq, stride_istate_dim, stride_istate_token = conv_state.stride()
    stride_state_indices = conv_state_indices.stride(0)
    if num_accepted_tokens is not None:
        state_len = width - 1 + (seqlen - 1)
    else:
        state_len = width - 1
    np2_statelen = triton.next_power_of_2(state_len)
    HV = triton.next_power_of_2(num_v_heads)
    BLOCK_Z = 512
    num_program_write_z = triton.cdiv(num_v_heads * head_v_dim, BLOCK_Z)

    def grid(META):
        return (
            batch,
            1 + num_program_write_z + triton.cdiv(dim, META["BLOCK_N"]),
        )

    _reshape_causal_conv1d_update_fast_kernel[grid](
        x,
        ba,
        z_out,
        core_attn_out,
        b_out,
        a_out,
        weight,
        bias,
        conv_state,
        conv_state_indices,
        block_idx_last_scheduled_token,
        initial_state_idx,
        out,
        batch,
        num_tokens,
        num_k_heads,
        num_v_heads,
        head_k_dim,
        head_v_dim,
        dim,
        head_qkvz_dim,
        seqlen,
        state_len,
        num_cache_lines,
        stride_x_seq,
        stride_x_dim,
        stride_x_token,
        stride_w_dim,
        stride_w_width,
        stride_istate_seq,
        stride_istate_dim,
        stride_istate_token,
        stride_state_indices,
        stride_o_seq,
        stride_o_dim,
        stride_o_token,
        stride_z_seq,
        stride_ba_seq,
        stride_ba_token,
        stride_b_seq,
        pad_slot_id,
        num_program_write_z,
        BLOCK_Z,
        HV=HV,
        HAS_BIAS=bias is not None,
        KERNEL_WIDTH=width,
        SILU_ACTIVATION=activation in ["silu", "swish"],
        IS_APC_ENABLED=block_idx_last_scheduled_token is not None,
        NP2_STATELEN=np2_statelen,
        USE_PAD_SLOT=pad_slot_id is not None,
        BLOCK_N=256,
    )
    if unsqueeze:
        out = out.squeeze(-1)
    z_out = z_out.view(original_z_shape)
    core_attn_out = core_attn_out.view(original_z_shape)
    return out.to(original_x_dtype), b_out, a_out
