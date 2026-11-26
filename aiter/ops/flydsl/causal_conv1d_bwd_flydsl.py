# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL backward of the varlen causal conv1d with split q/k/v output.

Same contract as the Triton
``aiter.ops.triton.gated_delta_net.causal_conv1d_bwd.causal_conv1d_split_qkv_bwd_triton_fn``:
``x`` is ``[dim, cu_seqlen]`` with channels ``[Q | K | V]`` and the gradients are
``dq, dk: [cu_seqlen, k_dim]``, ``dv: [cu_seqlen, v_dim]``. The kernel is in
``aiter.ops.flydsl.kernels.causal_conv1d_bwd``.
"""

import flydsl.expr as fx
import torch

from aiter.ops.flydsl.kernels.causal_conv1d_bwd import create_causal_conv1d_bwd_kernel
from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled
from aiter.ops.prefill_batch_metadata import CausalConvPrefillMetadata

__all__ = ["causal_conv1d_split_qkv_bwd_flydsl_fn"]

PAD_SLOT_ID = -1


def _build_schedule(query_start_loc: torch.Tensor, block_m: int):
    """(n_chunks, sequence id, chunk id) of every ``block_m``-token chunk; one host sync."""
    device = query_start_loc.device
    cpu = torch.device("cpu")
    nums = (query_start_loc.diff().to(cpu, torch.int64) + block_m - 1) // block_m
    tot = int(nums.sum())
    if tot == 0:
        empty = torch.zeros(0, dtype=torch.int32, device=device)
        return 0, empty, empty
    seq_ids = torch.arange(nums.numel(), dtype=torch.int32, device=cpu)
    batch_ptr = torch.repeat_interleave(seq_ids, nums)
    first = torch.repeat_interleave(nums.cumsum(0) - nums, nums)
    chunk_ptr = (torch.arange(tot, dtype=torch.int64, device=cpu) - first).to(
        torch.int32
    )
    return tot, batch_ptr.to(device), chunk_ptr.to(device)


def _schedule(query_start_loc, block_m, metadata, cu_seqlen, n_seqs):
    """(n_chunks, sequence ids, chunk ids); a prefilled ``nums_dict`` needs no host sync."""
    if isinstance(metadata, CausalConvPrefillMetadata):
        metadata.validate(query_start_loc, total_tokens=cu_seqlen, num_sequences=n_seqs)
        grid = metadata.get_chunk_grid(block_m)
        return grid.total_chunks, grid.sequence_ids, grid.chunk_ids
    nums_dict = getattr(metadata, "nums_dict", None) if metadata is not None else None
    if nums_dict is not None and block_m in nums_dict:
        entry = nums_dict[block_m]
        return int(entry["tot"]), entry["batch_ptr"], entry["token_chunk_offset_ptr"]
    tot, batch_ptr, chunk_ptr = _build_schedule(query_start_loc, block_m)
    if nums_dict is not None:
        nums_dict[block_m] = {
            "tot": tot,
            "batch_ptr": batch_ptr,
            "token_chunk_offset_ptr": chunk_ptr,
        }
    return tot, batch_ptr, chunk_ptr


def _unit_axis(t: torch.Tensor) -> torch.Tensor:
    """``t`` itself when one axis has unit stride, else a contiguous copy."""
    return t if 1 in t.stride() else t.contiguous()


def _span(t: torch.Tensor) -> int:
    return 1 + sum((n - 1) * s for n, s in zip(t.shape, t.stride()))


def causal_conv1d_split_qkv_bwd_flydsl_fn(
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
    *,
    block_m: int = 64,
    block: int = 64,
    prefetch: int = 4,
    channels_per_thread: int | None = None,
    runtime_loop: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Gradients of the varlen causal conv1d with split q/k/v output.

    Arguments and returns are those of ``causal_conv1d_split_qkv_bwd_triton_fn``.
    ``block_m`` is the chunk length of the schedule (a ``metadata`` schedule must
    be built with it), ``block`` the threads per program, ``prefetch`` the token
    rows loaded ahead, and ``channels_per_thread`` 1 or 2 (default: 2 when every
    channel count is even). The defaults are the best of a gfx950 sweep on the
    Kimi-K3 shapes.
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
    if not dq.dtype == dk.dtype == dv.dtype == x.dtype:
        raise ValueError("x, dq, dk and dv must have the same dtype")
    if x.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError(f"unsupported dtype {x.dtype}")
    if block not in (64, 128, 256) or block_m < width - 1 or block_m > 256:
        raise ValueError("block must be 64/128/256 and width - 1 <= block_m <= 256")
    if prefetch not in (1, 2, 4, 8, 16):
        raise ValueError("prefetch must be 1/2/4/8/16")
    has_init = has_initial_state is not None
    if has_init and initial_states is None:
        raise ValueError("has_initial_state needs initial_states")
    lanes = channels_per_thread
    if lanes is None:
        lanes = 2 if (k_dim % 2 == 0 and v_dim % 2 == 0) else 1
    if lanes not in (1, 2) or k_dim % lanes or v_dim % lanes:
        raise ValueError("channels_per_thread must be 1 or 2 and divide k_dim, v_dim")

    x = _unit_axis(x)
    dq, dk, dv = (_unit_axis(g) for g in (dq, dk, dv))
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

    tot, batch_ptr, chunk_ptr = _schedule(
        query_start_loc, block_m, metadata, cu_seqlen, n_seqs
    )
    if tot == 0:
        # No sequence has a token, so every token is past the last sequence.
        dx.zero_()
        dweight = torch.zeros_like(weight)
        dbias = torch.zeros_like(bias) if bias is not None else None
        return dx, dweight, dbias, dinit

    # [chunk, channel, tap]: summed over chunks it is already the dweight layout.
    n_rows = width + (1 if bias is not None else 0)
    partial = torch.empty((tot, dim, n_rows), dtype=torch.float32, device=x.device)
    state = initial_states if has_init else x
    ss = initial_states.stride() if has_init else (0, 0, 0)

    # The kernel names the unit-stride axis 0 for channels and 1 for tokens. x is
    # [channel, token]; dq, dk and dv are [token, channel].
    x_unit = x.stride().index(1)
    g_axes = tuple(g.stride().index(1) for g in (dq, dk, dv))
    g_units = tuple(1 - a for a in g_axes)
    launch = create_causal_conv1d_bwd_kernel(
        dim,
        k_dim,
        width,
        x_unit,
        g_units,
        tuple(weight.stride()),
        tuple(ss),
        bias is not None,
        activation in ("silu", "swish"),
        has_init,
        cache_indices is not None,
        pad_slot_id,
        block,
        block_m,
        prefetch,
        lanes,
        x.dtype,
        state.dtype,
        runtime_loop,
    )
    # Absent optional pointers are never dereferenced by the specialized kernel.
    args = (
        x,
        weight,
        bias if bias is not None else weight,
        state,
        query_start_loc,
        cache_indices if cache_indices is not None else query_start_loc,
        has_initial_state if has_init else query_start_loc,
        batch_ptr,
        chunk_ptr,
        dq,
        dk,
        dv,
        dx,
        partial,
        dinit if has_init else partial,
    )
    runtime = (
        *(_span(t) for t in (x, weight, state, dq, dk, dv, dx, partial)),
        _span(dinit) if has_init else _span(partial),
        x.stride(1 - x_unit),
        dx.stride(1 - x_unit),
        *(g.stride(1 - a) for g, a in zip((dq, dk, dv), g_axes)),
        n_seqs,
        cu_seqlen,
        tot,
    )
    with torch.cuda.device(x.device):
        stream = torch.cuda.current_stream(x.device)
        # FlyDSL specializes tensors on dtype, rank and the first unit-stride
        # axis; sizes and the other strides stay dynamic.
        units = (1 in t.stride() and t.stride().index(1) for t in args)
        key = (x.device.index, tuple(zip((t.dtype for t in args), units)))
        _run_compiled(
            launch, *args, *runtime, fx.Stream(stream), specialization_key=key
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
