# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from typing import Optional, Tuple

import torch
from torch import Tensor

from ..jit.core import compile_ops

MD_NAME = "module_causal_conv1d_fwd_split_qkv"

PAD_SLOT_ID = -1


# No C++ torch dependency: the kernel uses the ``aiter_tensor_t`` ABI (like
# ``chunk_gated_delta_rule_fwd_h``). ``develop=True`` makes ``compile_ops``
# convert each ``torch.Tensor`` to a pybind ``aiter_tensor_t`` before the call.
# Outputs ``q``/``k``/``v`` are pre-allocated by the caller and written in place,
# so the op returns ``None``.
@compile_ops(MD_NAME, develop=True)
def causal_conv1d_fwd_split_qkv_hip(
    x: Tensor,
    weight: Tensor,
    bias: Tensor,
    conv_states: Tensor,
    cache_indices: Tensor,
    has_initial_state: Tensor,
    query_start_loc: Tensor,
    batch_ptr: Tensor,
    token_chunk_offset_ptr: Tensor,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    k_dim: int,
    v_dim: int,
    n_programs: int,
    block_m: int,
    has_bias: bool,
    silu: bool,
    pad_slot_id: int,
) -> None: ...


# Token-tile auto-dispatch thresholds. Each pair is
# ``(avg_seqlen_upper_bound, TM)``; the first match wins, else TM=64. The tile
# is picked so that ``TM ~= avg seqlen``, minimizing both the masked-out waste
# of an oversized tile and the per-chunk fixed cost of an undersized one.
# Crossovers: <=12 ->8, (12,24] ->16, (24,48] ->32, >48 ->64.
_TILE_SEGMENTS = ((12, 8), (24, 16), (48, 32))


def _pick_tile(avg_seqlen: float) -> int:
    for thr, tm in _TILE_SEGMENTS:
        if avg_seqlen <= thr:
            return tm
    return 64


def _build_chunk_metadata(
    query_start_loc: Tensor, block_m: int
) -> Tuple[int, Tensor, Tensor]:
    """Build (num_programs, batch_ptr, token_chunk_offset_ptr).

    The HIP kernel decodes ``grid.x`` (a flattened ``(sequence, chunk)``
    list) through these two index tensors, exactly like the Triton tile kernel.
    ``batch_ptr[pid]`` is the sequence index owning chunk ``pid`` and
    ``token_chunk_offset_ptr[pid]`` is the chunk offset within that sequence.

    Fully vectorized (no Python loop): the within-sequence chunk offset is
    ``arange(tot) - repeat_interleave(exclusive_prefix_sum(nums), nums)``. The
    index tensors are sized to the exact program count (no padding).
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


def causal_conv1d_split_qkv_hip_fn(
    x: Tensor,
    weight: Tensor,
    bias: Optional[Tensor],
    conv_states: Tensor,
    query_start_loc: Tensor,
    k_dim: int,
    v_dim: int,
    cache_indices: Optional[Tensor] = None,
    has_initial_state: Optional[Tensor] = None,
    activation: Optional[str] = "silu",
    pad_slot_id: int = PAD_SLOT_ID,
    block_m: Optional[int] = None,
    metadata=None,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Prefill causal-conv1d (HIP) with fused split q/k/v output.

    Drop-in replacement for the Triton ``causal_conv1d_split_qkv_triton_fn``.

    x: [dim, cu_seqlen] bf16, channels are concatenated ``[Q | K | V]``.
    weight: [dim, width]; conv_states: [num_cache_lines, dim, width-1] (in/out).
    Returns contiguous (q, k, v) of shapes [cu_seqlen, k_dim] / [cu_seqlen, v_dim].
    """
    if x.dtype != torch.bfloat16:
        raise TypeError("HIP causal_conv1d kernel requires bfloat16 `x`.")
    if conv_states.dtype != torch.bfloat16:
        raise TypeError("HIP causal_conv1d kernel requires bfloat16 `conv_states`.")

    dim, _cu_seqlen = x.shape
    _, width = weight.shape
    if width != 4:
        raise ValueError(f"HIP causal_conv1d kernel requires width=4, got {width}.")

    n_seqs = query_start_loc.numel() - 1

    # Token-tile dispatch. Caller may force 8/16/32/64; otherwise auto-pick from
    # the host-known average sequence length (no device sync; graph-capture safe)
    # so that ``TM ~= avg seqlen`` (see ``_TILE_SEGMENTS``).
    if block_m in (8, 16, 32, 64):
        bm = int(block_m)
    else:
        bm = _pick_tile(_cu_seqlen / max(n_seqs, 1))

    if cache_indices is None:
        cache_indices = torch.arange(n_seqs, dtype=torch.int32, device=x.device)
    # ``torch.bool`` has no ``aiter_tensor_t`` dtype; the kernel reads this flag
    # as ``unsigned char`` anyway, so pass it as uint8.
    if has_initial_state is None:
        has_initial_state = torch.zeros(n_seqs, dtype=torch.uint8, device=x.device)
    else:
        has_initial_state = has_initial_state.to(torch.uint8)

    nums_dict = getattr(metadata, "nums_dict", None) if metadata is not None else None
    if nums_dict is not None and bm in nums_dict:
        # Precomputed schedule for the chosen tile: reuse it (zero extra cost).
        entry = nums_dict[bm]
        tot = int(entry["tot"])
        batch_ptr = entry["batch_ptr"]
        token_chunk_offset_ptr = entry["token_chunk_offset_ptr"]
        if batch_ptr.device != x.device:
            batch_ptr = batch_ptr.to(x.device)
            token_chunk_offset_ptr = token_chunk_offset_ptr.to(x.device)
    else:
        # Build the chosen tile's schedule once. When structured metadata is
        # present, memoize it back into the shared ``nums_dict`` so sibling
        # layers in this step reuse it instead of rebuilding (an auto-picked
        # TM=16/32 typically is not in the caller's precomputed list).
        tot, batch_ptr, token_chunk_offset_ptr = _build_chunk_metadata(
            query_start_loc, bm
        )
        if nums_dict is not None:
            nums_dict[bm] = {
                "tot": tot,
                "batch_ptr": batch_ptr,
                "token_chunk_offset_ptr": token_chunk_offset_ptr,
            }

    has_bias = bias is not None
    bias_arg = (
        bias if bias is not None else torch.empty(0, dtype=x.dtype, device=x.device)
    )
    silu = activation in ("silu", "swish")

    # Outputs are allocated here and written in place by the kernel (the op uses
    # the ``aiter_tensor_t`` ABI and returns ``None``).
    q = torch.empty((_cu_seqlen, k_dim), dtype=x.dtype, device=x.device)
    k = torch.empty((_cu_seqlen, k_dim), dtype=x.dtype, device=x.device)
    v = torch.empty((_cu_seqlen, v_dim), dtype=x.dtype, device=x.device)

    causal_conv1d_fwd_split_qkv_hip(
        x,
        weight,
        bias_arg,
        conv_states,
        cache_indices.to(torch.int32),
        has_initial_state,
        query_start_loc.to(torch.int32),
        batch_ptr.to(torch.int32),
        token_chunk_offset_ptr.to(torch.int32),
        q,
        k,
        v,
        int(k_dim),
        int(v_dim),
        int(tot),
        int(bm),
        bool(has_bias),
        bool(silu),
        int(pad_slot_id),
    )
    return q, k, v
