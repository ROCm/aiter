# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Triton prefill causal-conv1d (fused split q/k/v) — public host entry points.

This is the public Triton namespace for the Gated-Delta-Rule (GDR / Mamba)
prefill front-end depthwise conv. It exposes two interchangeable Triton kernels;
the upper-layer framework selects the backend explicitly:

* ``causal_conv1d_split_qkv_triton_fn``      -- 1D per-token kernel (portable; any width).
* ``causal_conv1d_split_qkv_triton_tile_fn`` -- 2D-tiled kernel (vectorized over both
  feature and token axes; requires conv width in {2, 3, 4}). For x with unit channel
  stride and conv width 4 it runs a sliding-window kernel instead.

The other backends live in their own namespaces:

* HIP    : ``aiter.ops.causal_conv1d_fwd_split_qkv.causal_conv1d_split_qkv_hip_fn``
* FlyDSL : ``aiter.ops.flydsl.causal_conv1d_flydsl.causal_conv1d_split_qkv_flydsl_fn``

The ``@triton.jit`` kernels themselves live in
``aiter.ops.triton._triton_kernels.gated_delta_net.prefill.causal_conv1d_fwd_split_qkv``.

``x`` has shape ``[dim, cu_seqlen]`` with the channel axis packed as the
concatenation ``[Q | K | V]`` (``dim == 2*k_dim + v_dim``).
"""

import functools
import os

import torch
import triton

from aiter.ops.prefill_batch_metadata import CausalConvPrefillMetadata
from aiter.ops.triton._triton_kernels.gated_delta_net.prefill.causal_conv1d_fwd_split_qkv import (
    PAD_SLOT_ID,
    _causal_conv1d_fwd_split_qkv_kernel,
    _causal_conv1d_fwd_split_qkv_tile_kernel,
    _causal_conv1d_fwd_split_qkv_window_kernel,
)
from aiter.ops.triton.utils.conv_config_utils import get_conv_config, has_conv_config
from aiter.ops.triton.utils.device_info import get_num_xcds

__all__ = [
    "PAD_SLOT_ID",
    "causal_conv1d_split_qkv_triton_fn",
    "causal_conv1d_split_qkv_triton_tile_fn",
]


def causal_conv1d_split_qkv_triton_fn(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    conv_states: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens_cpu: list[int],
    k_dim: int,
    v_dim: int,
    cache_indices: torch.Tensor | None = None,
    has_initial_state: torch.Tensor | None = None,
    activation: str = "silu",
    pad_slot_id: int = PAD_SLOT_ID,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
) -> tuple[int, torch.Tensor, torch.Tensor]:
    """Vectorized (sequence, chunk) schedule, exact-sized (no padding).

    ``batch_ptr[pid]`` is the sequence owning program ``pid`` and
    ``token_chunk_offset_ptr[pid]`` its within-sequence chunk index, identical
    to the HIP kernel's schedule. The within-sequence offset is
    ``arange(tot) - repeat_interleave(exclusive_prefix_sum(nums), nums)``.
    """
    device = query_start_loc.device
    seqlens = query_start_loc.diff()
    nums = (-(-seqlens // block_m)).to(torch.int64)  # ceil-div per sequence
    n_seqs = nums.numel()
    tot = int(nums.sum().item())
    if tot == 0:
        z = torch.zeros(0, dtype=torch.int32, device=device)
        return 0, z, z
    seq_ids = torch.arange(n_seqs, dtype=torch.int32, device=device)
    batch_ptr = torch.repeat_interleave(seq_ids, nums, output_size=tot)
    starts = nums.cumsum(0) - nums  # exclusive prefix sum
    base = torch.repeat_interleave(starts, nums, output_size=tot)
    tco = (torch.arange(tot, dtype=torch.int64, device=device) - base).to(torch.int32)
    return tot, batch_ptr, tco


def causal_conv1d_split_qkv_triton_tile_fn(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    conv_states: torch.Tensor,
    query_start_loc: torch.Tensor,
    k_dim: int,
    v_dim: int,
    cache_indices: torch.Tensor | None = None,
    has_initial_state: torch.Tensor | None = None,
    activation: str | None = "silu",
    pad_slot_id: int = PAD_SLOT_ID,
    block_m: int = 64,
    block_n: int | None = None,
    num_warps: int = 4,
    metadata=None,
    preserve_input_dtype: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """2D-tiled prefill causal conv1d with fused split q/k/v output.

    Drop-in alternative to the 1D ``causal_conv1d_split_qkv_triton_fn``.

    x: [dim, cu_seqlen] bf16, channels packed ``[Q | K | V]``.
    weight: [dim, width] (width in {2,3,4}); conv_states: [lines, dim, width-1].
    Returns contiguous (q, k, v) of shapes [cu_seqlen, k_dim] / [cu_seqlen, v_dim].

    preserve_input_dtype: return q/k/v in x's dtype to match vLLM
    block_n: None picks 16 B per lane for channel-contiguous x, else 32.

    Channel-contiguous x with conv width 4 runs the sliding-window kernel (tuned config; block_m, block_n,
    num_warps and metadata unused). AITER_CONV1D_SPLIT_QKV_WINDOW=0 turns it off.
    """
    window_config = _window_config(x, weight, conv_states, k_dim, v_dim)
    if window_config is not None:
        return _causal_conv1d_split_qkv_window(
            window_config,
            x,
            weight,
            bias,
            conv_states,
            query_start_loc,
            k_dim,
            v_dim,
            cache_indices,
            has_initial_state,
            activation,
            pad_slot_id,
            x.dtype if preserve_input_dtype else conv_states.dtype,
        )
    # Cast back only when asked; a no-op when cache and activations share a dtype
    out_dtype = x.dtype if preserve_input_dtype else conv_states.dtype
    x = x.to(conv_states.dtype)
    dim, cu_seqlen = x.shape
    _, width = weight.shape
    if width not in (2, 3, 4):
        raise ValueError(f"2D-tile kernel requires width in (2,3,4), got {width}.")
    state_len = width - 1
    np2_statelen = triton.next_power_of_2(state_len)

    BLOCK_M = int(block_m)
    if block_n is None:
        # 16 B per lane for channel-contiguous x
        block_n = (
            min(512, max(32, 8 * num_warps * _warp_size() // BLOCK_M))
            if x.stride(0) == 1
            else 32
        )
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
    if isinstance(metadata, CausalConvPrefillMetadata):
        metadata.validate(
            query_start_loc,
            total_tokens=cu_seqlen,
            num_sequences=n_seqs,
        )
        grid = metadata.get_chunk_grid(BLOCK_M)
        tot = grid.total_chunks
        batch_ptr = grid.sequence_ids
        token_chunk_offset_ptr = grid.chunk_ids
        nums_dict = None
    else:
        nums_dict = (
            getattr(metadata, "nums_dict", None) if metadata is not None else None
        )
    if not isinstance(metadata, CausalConvPrefillMetadata) and (
        nums_dict is not None and BLOCK_M in nums_dict
    ):
        entry = nums_dict[BLOCK_M]
        tot = int(entry["tot"])
        batch_ptr = entry["batch_ptr"]
        token_chunk_offset_ptr = entry["token_chunk_offset_ptr"]
        if batch_ptr.device != x.device:
            batch_ptr = batch_ptr.to(x.device)
            token_chunk_offset_ptr = token_chunk_offset_ptr.to(x.device)
    elif not isinstance(metadata, CausalConvPrefillMetadata):
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
        return (
            query.to(out_dtype),
            key.to(out_dtype),
            value.to(out_dtype),
        )

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
        QKV_ALIGNED=k_dim % BLOCK_N == 0 and v_dim % BLOCK_N == 0,
        num_warps=num_warps,
        num_stages=1,
    )
    return (
        query.to(out_dtype),
        key.to(out_dtype),
        value.to(out_dtype),
    )


_WINDOW_CONFIG_NAME = "CAUSAL_CONV1D_SPLIT_QKV"


@functools.cache
def _warp_size() -> int:
    return triton.runtime.driver.active.get_current_target().warp_size


def _window_config(x, weight, conv_states, k_dim, v_dim) -> dict | None:
    """Window kernel config, or None if it does not apply"""
    dim, cu_seqlen = x.shape
    if (
        os.environ.get("AITER_CONV1D_SPLIT_QKV_WINDOW", "1") == "0"
        or not has_conv_config(_WINDOW_CONFIG_NAME)
        or weight.dim() != 2
        or weight.shape[1] != 4
        or x.stride(0) != 1
        # State must hold x exactly
        or conv_states.dtype not in (x.dtype, torch.float32)
        or dim != 2 * k_dim + v_dim
        # 32-bit row offsets
        or (cu_seqlen + 3) * x.stride(1) + dim >= 2**31
    ):
        return None
    config = get_conv_config(_WINDOW_CONFIG_NAME, M=cu_seqlen)
    # Shrink BD until it divides q, k and v; keep R * BD
    while (k_dim % config["BD"] or v_dim % config["BD"]) and config["BD"] > 64:
        config["BD"] //= 2
        config["R"] *= 2
    if k_dim % config["BD"] or v_dim % config["BD"]:
        return None
    return config


def _causal_conv1d_split_qkv_window(
    config,
    x,
    weight,
    bias,
    conv_states,
    query_start_loc,
    k_dim,
    v_dim,
    cache_indices,
    has_initial_state,
    activation,
    pad_slot_id,
    out_dtype,
):
    """Window path of ``causal_conv1d_split_qkv_triton_tile_fn``"""
    dim, cu_seqlen = x.shape
    n_seq = query_start_loc.numel() - 1
    out = torch.empty(cu_seqlen * dim, dtype=out_dtype, device=x.device)
    q = out[: cu_seqlen * k_dim].view(cu_seqlen, k_dim)
    k = out[cu_seqlen * k_dim : 2 * cu_seqlen * k_dim].view(cu_seqlen, k_dim)
    v = out[2 * cu_seqlen * k_dim :].view(cu_seqlen, v_dim)
    if cu_seqlen == 0 or n_seq == 0:
        return q, k, v
    if cache_indices is None:
        cache_indices = torch.arange(n_seq, dtype=torch.int32, device=x.device)
    if has_initial_state is None:
        has_initial_state = torch.zeros(n_seq, dtype=torch.bool, device=x.device)
    if not weight.is_contiguous():
        weight = weight.contiguous()

    # Smallest sequence-start vector that fits, up to SEQ_VEC
    config["SEQ_VEC"] = min(config["SEQ_VEC"], max(16, triton.next_power_of_2(n_seq)))
    block_r, block_d = config["R"], config["BD"]
    # 16 B per lane unless the config sets num_warps
    num_warps = config.pop("num_warps", max(1, block_r * block_d // (8 * _warp_size())))
    num_xcds = get_num_xcds()
    n_blocks = dim // block_d
    n_tiles = triton.cdiv(cu_seqlen, block_r * config["BC"])
    # Head programs, padded to a multiple of NUM_XCDS so tiles keep their XCD
    n_head_programs = (
        triton.cdiv(triton.cdiv(n_seq, block_r) * n_blocks, num_xcds) * num_xcds
    )
    _causal_conv1d_fwd_split_qkv_window_kernel[(n_head_programs + n_tiles * n_blocks,)](
        x,
        weight,
        bias,
        conv_states,
        cache_indices,
        has_initial_state,
        query_start_loc,
        out,
        n_seq,
        cu_seqlen,
        x.stride(1),
        conv_states.stride(0),
        conv_states.stride(1),
        conv_states.stride(2),
        pad_slot_id,
        n_tiles,
        n_head_programs,
        KD=k_dim,
        VD=v_dim,
        HAS_BIAS=bias is not None,
        SILU=activation in ("silu", "swish"),
        NUM_XCDS=num_xcds,
        num_warps=num_warps,
        **config,
    )
    return q, k, v
