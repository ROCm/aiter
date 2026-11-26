# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Triton backward of the varlen causal conv1d with split q/k/v output.

This is the training counterpart of
``aiter.ops.triton.gated_delta_net.causal_conv1d_prefill.causal_conv1d_split_qkv_triton_tile_fn``.
It uses the same layout: ``x`` is ``[dim, cu_seqlen]`` with the channels packed
as ``[Q | K | V]`` (``dim == 2 * k_dim + v_dim``), and the output gradients are
``dq, dk: [cu_seqlen, k_dim]`` and ``dv: [cu_seqlen, v_dim]``.

The ``@triton.jit`` kernel lives in
``aiter.ops.triton._triton_kernels.gated_delta_net.prefill.causal_conv1d_bwd_split_qkv``.
"""

import torch
import triton

from aiter.ops.prefill_batch_metadata import CausalConvPrefillMetadata
from aiter.ops.triton._triton_kernels.gated_delta_net.prefill.causal_conv1d_bwd_split_qkv import (
    _causal_conv1d_bwd_split_qkv_kernel,
)
from aiter.ops.triton._triton_kernels.gated_delta_net.prefill.causal_conv1d_fwd_split_qkv import (
    PAD_SLOT_ID,
)
from aiter.ops.triton.gated_delta_net.causal_conv1d_prefill import (
    _build_chunk_schedule,
)
from aiter.ops.triton.utils.conv_config_utils import get_conv_config, has_conv_config

__all__ = ["causal_conv1d_split_qkv_bwd_triton_fn"]

_BWD_CONFIG_NAME = "CAUSAL_CONV1D_SPLIT_QKV_BWD"
_DEFAULT_BWD_CONFIG = {"BLOCK_M": 64, "BLOCK_N": 128, "num_warps": 1}


def _bwd_config(cu_seqlen: int) -> dict:
    """Launch config: the arch table when it exists (gfx950), else the default."""
    if has_conv_config(_BWD_CONFIG_NAME):
        return get_conv_config(_BWD_CONFIG_NAME, M=cu_seqlen)
    return dict(_DEFAULT_BWD_CONFIG)


def _schedule(query_start_loc, block_m, metadata, cu_seqlen, n_seqs):
    """(n_chunks, sequence ids, chunk ids).

    A plain ``metadata.nums_dict`` schedule needs no host sync, so it works in
    graph capture.
    """
    if isinstance(metadata, CausalConvPrefillMetadata):
        metadata.validate(query_start_loc, total_tokens=cu_seqlen, num_sequences=n_seqs)
        grid = metadata.get_chunk_grid(block_m)
        return grid.total_chunks, grid.sequence_ids, grid.chunk_ids
    nums_dict = getattr(metadata, "nums_dict", None) if metadata is not None else None
    if nums_dict is not None and block_m in nums_dict:
        entry = nums_dict[block_m]
        return int(entry["tot"]), entry["batch_ptr"], entry["token_chunk_offset_ptr"]
    tot, batch_ptr, chunk_ptr = _build_chunk_schedule(query_start_loc, block_m)
    if nums_dict is not None:
        nums_dict[block_m] = {
            "tot": tot,
            "batch_ptr": batch_ptr,
            "token_chunk_offset_ptr": chunk_ptr,
        }
    return tot, batch_ptr, chunk_ptr


def causal_conv1d_split_qkv_bwd_triton_fn(
    dq: torch.Tensor,
    dk: torch.Tensor,
    dv: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    query_start_loc: torch.Tensor,
    k_dim: int,
    v_dim: int,
    initial_states: torch.Tensor | None = None,
    cache_indices: torch.Tensor | None = None,
    has_initial_state: torch.Tensor | None = None,
    activation: str | None = "silu",
    pad_slot_id: int | None = PAD_SLOT_ID,
    metadata=None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Gradients of the varlen causal conv1d with split q/k/v output.

    The forward is ``out = act(bias + depthwise_causal_conv(x, weight))`` per
    sequence, split into q, k and v along the channels. The kernel recomputes
    the pre-activation from ``x``, so the forward saves nothing.

    Args:
        dq, dk, dv: gradients of q, k, v. ``[cu_seqlen, k_dim]``,
            ``[cu_seqlen, k_dim]``, ``[cu_seqlen, v_dim]``, one dtype. Any strides.
        x: forward input ``[dim, cu_seqlen]``. Any strides.
        weight: ``[dim, width]``, width in {2, 3, 4}.
        bias: ``[dim]`` or None.
        query_start_loc: ``[n_seqs + 1]`` int32 cumulative sequence starts.
        k_dim, v_dim: channel split, ``dim == 2 * k_dim + v_dim``.
        initial_states: the conv states as they were before the forward,
            ``[lines, dim, >= width - 1]``. Only the first ``width - 1`` columns
            are read. Required when ``has_initial_state`` is given. The forward
            updates the states in place, so save a copy before it runs.
        cache_indices: ``[n_seqs]`` state line of each sequence. Default: the
            sequence index.
        has_initial_state: ``[n_seqs]`` bool. Default: no initial states.
        activation: None, "silu" or "swish".
        pad_slot_id: a ``cache_indices`` entry equal to this value marks a padded
            sequence. Its gradients are zero. None disables the check.
        metadata: optional chunk schedule, as for the forward: a
            ``CausalConvPrefillMetadata`` built with the launch ``BLOCK_M``, or an
            object with a ``nums_dict`` that this call fills and later calls
            reuse. A prefilled ``nums_dict`` needs no host sync, so the call
            can be captured in a graph.

    Returns:
        ``(dx, dweight, dbias, dinitial_states)``.
        ``dx`` is ``[dim, cu_seqlen]`` in ``x.dtype`` with the strides of ``x``
        when ``x`` has unit channel stride, else contiguous. Tokens after the
        last sequence get zero.
        ``dweight`` is ``[dim, width]`` in ``weight.dtype``. ``dbias`` is
        ``[dim]`` in ``bias.dtype`` or None. Both are summed in fp32 in a fixed
        order, so the result is deterministic.
        ``dinitial_states`` is ``[n_seqs, dim, width - 1]`` in
        ``initial_states.dtype``, zero for sequences without an initial state,
        or None when ``has_initial_state`` is None.
    """
    if activation not in (None, "silu", "swish"):
        raise ValueError(f"activation must be None, silu or swish, got {activation}")
    dim, cu_seqlen = x.shape
    if weight.dim() != 2 or weight.shape[0] != dim:
        raise ValueError(f"weight must be [{dim}, width], got {tuple(weight.shape)}")
    width = weight.shape[1]
    if width not in (2, 3, 4):
        raise ValueError(f"width must be 2, 3 or 4, got {width}")
    if dim != 2 * k_dim + v_dim:
        raise ValueError(f"dim {dim} != 2 * k_dim + v_dim ({k_dim}, {v_dim})")
    if dq.shape != (cu_seqlen, k_dim) or dk.shape != (cu_seqlen, k_dim):
        raise ValueError("dq and dk must be [cu_seqlen, k_dim]")
    if dv.shape != (cu_seqlen, v_dim):
        raise ValueError("dv must be [cu_seqlen, v_dim]")
    if not dq.dtype == dk.dtype == dv.dtype:
        raise ValueError("dq, dk and dv must have the same dtype")
    has_init = has_initial_state is not None
    if has_init and initial_states is None:
        raise ValueError("has_initial_state needs initial_states")

    n_seqs = query_start_loc.numel() - 1
    if x.stride(0) == 1:
        dx = torch.empty((cu_seqlen, dim), dtype=x.dtype, device=x.device).t()
    else:
        dx = torch.empty((dim, cu_seqlen), dtype=x.dtype, device=x.device)
    dinit = (
        torch.zeros(
            (n_seqs, dim, width - 1), dtype=initial_states.dtype, device=x.device
        )
        if has_init
        else None
    )

    config = _bwd_config(cu_seqlen)
    block_m, block_n = config["BLOCK_M"], config["BLOCK_N"]
    tot, batch_ptr, chunk_ptr = _schedule(
        query_start_loc, block_m, metadata, cu_seqlen, n_seqs
    )
    if tot == 0:
        # No sequence has a token, so every token is past the last sequence.
        dx.zero_()
        dweight = torch.zeros_like(weight)
        dbias = torch.zeros_like(bias) if bias is not None else None
        return dx, dweight, dbias, dinit

    has_cache_indices = cache_indices is not None
    if not has_cache_indices:
        cache_indices = query_start_loc
    if has_init:
        states = initial_states
        st_seq, st_dim, st_tok = initial_states.stride()
        di_seq, di_dim, di_tok = dinit.stride()
    else:
        states, has_initial_state = x, query_start_loc
        st_seq = st_dim = st_tok = di_seq = di_dim = di_tok = 0

    # [chunk, channel, tap]: summed over chunks it is already the dweight layout.
    n_rows = width + (1 if bias is not None else 0)
    partial = torch.empty((tot, dim, n_rows), dtype=torch.float32, device=x.device)

    grid = (tot, triton.cdiv(dim, block_n))
    _causal_conv1d_bwd_split_qkv_kernel[grid](
        x,
        weight,
        bias,
        states,
        cache_indices,
        has_initial_state,
        query_start_loc,
        batch_ptr,
        chunk_ptr,
        dq,
        dk,
        dv,
        dx,
        partial,
        dinit if has_init else partial,
        dim,
        k_dim,
        v_dim,
        n_seqs,
        cu_seqlen,
        n_rows,
        x.stride(0),
        x.stride(1),
        weight.stride(0),
        weight.stride(1),
        st_seq,
        st_dim,
        st_tok,
        cache_indices.stride(0),
        dq.stride(0),
        dq.stride(1),
        dk.stride(0),
        dk.stride(1),
        dv.stride(0),
        dv.stride(1),
        dx.stride(0),
        dx.stride(1),
        di_seq,
        di_dim,
        di_tok,
        PAD_SLOT_ID if pad_slot_id is None else pad_slot_id,
        HAS_BIAS=bias is not None,
        KERNEL_WIDTH=width,
        SILU_ACTIVATION=activation in ("silu", "swish"),
        HAS_INITIAL_STATES=has_init,
        HAS_CACHE_INDICES=has_cache_indices,
        USE_PAD_SLOT=pad_slot_id is not None,
        X_UNIT_DIM=x.stride(0) == 1 and dx.stride(0) == 1,
        G_UNIT_DIM=dq.stride(1) == 1 and dk.stride(1) == 1 and dv.stride(1) == 1,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=config["num_warps"],
    )
    if tot >= 256 and tot % 16 == 0:
        # A single reduction over many chunks is slow on gfx950 (52 vs 28 us at 512 x 4608 x 4).
        sums = partial.view(tot // 16, 16, dim, n_rows).sum(dim=1).sum(dim=0)
    else:
        sums = partial.sum(dim=0)
    if bias is None:
        dweight, dbias = sums.to(weight.dtype), None
    else:
        dweight = sums[:, :width].contiguous().to(weight.dtype)
        dbias = sums[:, width].to(bias.dtype)
    return dx, dweight, dbias, dinit
