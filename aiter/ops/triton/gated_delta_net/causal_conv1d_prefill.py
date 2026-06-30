# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Triton prefill causal-conv1d (fused split q/k/v) — public host entry points.

This is the public Triton namespace for the Gated-Delta-Rule (GDR / Mamba)
prefill front-end depthwise conv. It exposes two interchangeable Triton kernels;
the upper-layer framework selects the backend explicitly:

* ``causal_conv1d_split_qkv_triton_fn``      -- 1D per-token kernel (portable; any width).
* ``causal_conv1d_split_qkv_triton_tile_fn`` -- 2D-tiled kernel (vectorized over both
  feature and token axes; requires conv width in {2, 3, 4}).

The other backends live in their own namespaces:

* HIP    : ``aiter.ops.causal_conv1d_fwd_split_qkv.causal_conv1d_split_qkv_hip_fn``
* FlyDSL : ``aiter.ops.flydsl.causal_conv1d_flydsl.causal_conv1d_split_qkv_flydsl_fn``

The ``@triton.jit`` kernels themselves live in
``aiter.ops.triton._triton_kernels.gated_delta_rule.prefill.causal_conv1d_fwd_split_qkv``.

``x`` has shape ``[dim, cu_seqlen]`` with the channel axis packed as the
concatenation ``[Q | K | V]`` (``dim == 2*k_dim + v_dim``).
"""

from typing import List, Optional, Tuple

import torch
import triton

from aiter.ops.triton._triton_kernels.gated_delta_rule.prefill.causal_conv1d_fwd_split_qkv import (
    PAD_SLOT_ID,
    _causal_conv1d_fwd_split_qkv_tile_kernel,
    _causal_conv1d_fwd_split_qkv_kernel,
)

__all__ = [
    "causal_conv1d_split_qkv_triton_fn",
    "causal_conv1d_split_qkv_triton_tile_fn",
    "PAD_SLOT_ID",
]


def causal_conv1d_split_qkv_triton_fn(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    conv_states: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens_cpu: List[int],
    k_dim: int,
    v_dim: int,
    cache_indices: Optional[torch.Tensor] = None,
    has_initial_state: Optional[torch.Tensor] = None,
    activation: str = "silu",
    pad_slot_id: int = PAD_SLOT_ID,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """1D per-token Triton causal conv1d with fused split output for prefill.

    Returns contiguous (q, k, v) of shapes [cu_seqlen, k_dim] / [cu_seqlen, v_dim].
    """
    dim, cu_seqlen = x.shape
    _, width = weight.shape
    state_len = width - 1
    np2_statelen = triton.next_power_of_2(state_len)

    q_out = torch.empty(cu_seqlen, k_dim, device=x.device, dtype=x.dtype)
    k_out = torch.empty(cu_seqlen, k_dim, device=x.device, dtype=x.dtype)
    v_out = torch.empty(cu_seqlen, v_dim, device=x.device, dtype=x.dtype)

    stride_x_dim = x.stride(0)
    stride_x_token = x.stride(1)
    stride_w_dim = weight.stride(0)
    stride_w_width = weight.stride(1)

    num_cache_lines = 0
    stride_istate_seq = 0
    stride_istate_dim = 0
    stride_istate_token = 0
    if conv_states is not None:
        num_cache_lines = conv_states.size(0)
        stride_istate_seq = conv_states.stride(0)
        stride_istate_dim = conv_states.stride(1)
        stride_istate_token = conv_states.stride(2)

    def grid(META):
        max_seq_len = max(seq_lens_cpu)
        return (
            len(seq_lens_cpu),
            (max_seq_len + META["BLOCK_M"] - 1) // META["BLOCK_M"],
            triton.cdiv(dim, META["BLOCK_N"]),
        )

    _causal_conv1d_fwd_split_qkv_kernel[grid](
        x,
        weight,
        bias,
        conv_states,
        cache_indices,
        has_initial_state,
        query_start_loc,
        q_out,
        k_out,
        v_out,
        k_dim,
        v_dim,
        dim,
        cu_seqlen,
        num_cache_lines,
        stride_x_dim,
        stride_x_token,
        stride_w_dim,
        stride_w_width,
        stride_istate_seq,
        stride_istate_dim,
        stride_istate_token,
        q_out.stride(0),
        q_out.stride(1),
        k_out.stride(0),
        k_out.stride(1),
        v_out.stride(0),
        v_out.stride(1),
        pad_slot_id,
        HAS_BIAS=bias is not None,
        KERNEL_WIDTH=width,
        SILU_ACTIVATION=activation in ["silu", "swish"],
        HAS_INITIAL_STATES=has_initial_state is not None,
        HAS_CACHE=conv_states is not None,
        IS_CONTINUOUS_BATCHING=cache_indices is not None,
        USE_PAD_SLOT=pad_slot_id is not None,
        NP2_STATELEN=np2_statelen,
        BLOCK_M=8,
        BLOCK_N=256,
        num_stages=2,
    )

    return q_out, k_out, v_out


def _build_chunk_schedule(
    query_start_loc: torch.Tensor, block_m: int
) -> Tuple[int, torch.Tensor, torch.Tensor]:
    """Vectorized (sequence, chunk) schedule, exact-sized (no padding).

    ``batch_ptr[pid]`` is the sequence owning program ``pid`` and
    ``token_chunk_offset_ptr[pid]`` its within-sequence chunk index, identical
    to the HIP kernel's schedule. The within-sequence offset is
    ``arange(tot) - repeat_interleave(exclusive_prefix_sum(nums), nums)``.
    """
    device = query_start_loc.device
    seqlens = query_start_loc.diff().to("cpu")
    nums = (-(-seqlens // block_m)).to(torch.int64)  # ceil-div per sequence
    n_seqs = nums.numel()
    tot = int(nums.sum().item())
    if tot == 0:
        z = torch.zeros(0, dtype=torch.int32, device=device)
        return 0, z, z
    seq_ids = torch.arange(n_seqs, dtype=torch.int32)
    batch_ptr = torch.repeat_interleave(seq_ids, nums)
    starts = nums.cumsum(0) - nums  # exclusive prefix sum
    base = torch.repeat_interleave(starts, nums)
    tco = (torch.arange(tot, dtype=torch.int64) - base).to(torch.int32)
    return tot, batch_ptr.to(device), tco.to(device)


def causal_conv1d_split_qkv_triton_tile_fn(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    conv_states: torch.Tensor,
    query_start_loc: torch.Tensor,
    k_dim: int,
    v_dim: int,
    cache_indices: Optional[torch.Tensor] = None,
    has_initial_state: Optional[torch.Tensor] = None,
    activation: Optional[str] = "silu",
    pad_slot_id: int = PAD_SLOT_ID,
    block_m: int = 64,
    block_n: int = 32,
    num_warps: int = 4,
    metadata=None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """2D-tiled prefill causal conv1d with fused split q/k/v output.

    Drop-in alternative to the 1D ``causal_conv1d_split_qkv_triton_fn``.

    x: [dim, cu_seqlen] bf16, channels packed ``[Q | K | V]``.
    weight: [dim, width] (width in {2,3,4}); conv_states: [lines, dim, width-1].
    Returns contiguous (q, k, v) of shapes [cu_seqlen, k_dim] / [cu_seqlen, v_dim].
    """
    x = x.to(conv_states.dtype)
    dim, cu_seqlen = x.shape
    _, width = weight.shape
    if width not in (2, 3, 4):
        raise ValueError(f"2D-tile kernel requires width in (2,3,4), got {width}.")
    state_len = width - 1
    np2_statelen = triton.next_power_of_2(state_len)

    BLOCK_M = int(block_m)
    BLOCK_N = int(block_n)
    n_seqs = int(query_start_loc.numel() - 1)

    if cache_indices is None:
        cache_indices = torch.arange(n_seqs, dtype=torch.int32, device=x.device)
    # The kernel loads ``has_initial_states_ptr`` unconditionally, so it must
    # always be a valid tensor; triton reads it as ``.to(tl.int1)``.
    if has_initial_state is None:
        has_initial_state = torch.zeros(n_seqs, dtype=torch.bool, device=x.device)

    # (sequence, chunk) schedule for the chosen BLOCK_M, optionally memoized
    # in ``metadata.nums_dict`` and shared with the HIP backend.
    nums_dict = getattr(metadata, "nums_dict", None) if metadata is not None else None
    if nums_dict is not None and BLOCK_M in nums_dict:
        entry = nums_dict[BLOCK_M]
        tot = int(entry["tot"])
        batch_ptr = entry["batch_ptr"]
        token_chunk_offset_ptr = entry["token_chunk_offset_ptr"]
        if batch_ptr.device != x.device:
            batch_ptr = batch_ptr.to(x.device)
            token_chunk_offset_ptr = token_chunk_offset_ptr.to(x.device)
    else:
        tot, batch_ptr, token_chunk_offset_ptr = _build_chunk_schedule(
            query_start_loc, BLOCK_M
        )
        if nums_dict is not None:
            nums_dict[BLOCK_M] = {
                "tot": tot,
                "batch_ptr": batch_ptr,
                "token_chunk_offset_ptr": token_chunk_offset_ptr,
            }

    num_cache_lines = conv_states.size(0)
    stride_istate_seq = conv_states.stride(0)
    stride_istate_dim = conv_states.stride(1)
    stride_istate_token = conv_states.stride(2)
    stride_cache_indices = cache_indices.stride(0)

    query = torch.empty([cu_seqlen, k_dim], dtype=x.dtype, device=x.device)
    key = torch.empty([cu_seqlen, k_dim], dtype=x.dtype, device=x.device)
    value = torch.empty([cu_seqlen, v_dim], dtype=x.dtype, device=x.device)

    if tot == 0:
        return query, key, value

    grid = (tot, triton.cdiv(dim, BLOCK_N))
    _causal_conv1d_fwd_split_qkv_tile_kernel[grid](
        x,
        weight,
        bias,
        conv_states,
        cache_indices,
        has_initial_state,
        query_start_loc,
        batch_ptr,
        token_chunk_offset_ptr,
        None,  # block_idx_first_scheduled_token (APC disabled)
        None,  # block_idx_last_scheduled_token  (APC disabled)
        None,  # initial_state_idx               (APC disabled)
        None,  # num_computed_tokens             (APC disabled)
        query,
        key,
        value,
        dim,
        k_dim,
        v_dim,
        cu_seqlen,
        num_cache_lines,
        x.stride(0),
        x.stride(1),
        weight.stride(0),
        weight.stride(1),
        stride_istate_seq,
        stride_istate_dim,
        stride_istate_token,
        stride_cache_indices,
        query.stride(1),
        query.stride(0),
        key.stride(1),
        key.stride(0),
        value.stride(1),
        value.stride(0),
        0,  # stride_block_m (APC only)
        pad_slot_id,
        HAS_BIAS=bias is not None,
        KERNEL_WIDTH=width,
        SILU_ACTIVATION=activation in ["silu", "swish"],
        IS_APC_ENABLED=False,
        USE_PAD_SLOT=pad_slot_id is not None,
        NP2_STATELEN=np2_statelen,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=1,
    )
    return query, key, value
