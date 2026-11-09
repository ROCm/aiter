# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL routes behind the public unified-attention entry point.

Validated route: BF16 paged decode/verification on gfx950 through ``pa_decode``
with the linear ``[blocks, page, kv_heads, dim]`` KV cache, the MiMo V2.6 global
attention geometry (Q/K 192, V/output 128, 16 query heads per KV head), page
sizes 64 and 128, causal full attention and a uniform query length of 1 to 8
tokens per sequence. ``unified_attention_flydsl`` returns ``None`` for anything
else; ``flydsl_decline_reason`` names the first unsupported property.

Graph capture: the first call of a geometry on a stream must run uncaptured to
allocate its workspace. Each call, captured or not, records a GPU plan refresh
from ``seqused_k``, so lengths may change between replays up to the captured
``max_seqlen_k``. Calls and replays sharing a workspace must not overlap.
"""

from __future__ import annotations

import os

import torch

from aiter import logger

__all__ = ["flydsl_decline_reason", "unified_attention_flydsl"]

_HEAD_DIM = 192
_VALUE_DIM = 128
_GQA = 16
_PAGE_SIZES = (64, 128)
_MAX_QUERY_LENGTH = 8
_MAX_BATCH = 4096


def _env_enabled() -> bool:
    return os.environ.get("AITER_UNIFIED_ATTN_FLYDSL", "1") != "0"


def _arch() -> str | None:
    try:
        from aiter.jit.utils.chip_info import get_gfx_runtime
    except ModuleNotFoundError:
        return None
    return get_gfx_runtime()


def flydsl_decline_reason(
    q,
    k,
    v,
    out,
    cu_seqlens_q,
    max_seqlen_q,
    seqused_k,
    max_seqlen_k,
    softmax_scale,
    causal,
    window_size,
    block_table,
    softcap,
    q_descale,
    k_descale,
    v_descale,
    q_scales=None,
    alibi_slopes=None,
    output_scale=None,
    qq_bias=None,
    sinks=None,
    shuffled_kv_cache: bool = False,
    skip_reduce: bool = False,
) -> str | None:
    """Return why the FlyDSL route declines this call, or None if it runs it.

    Host-side checks only: tensor values (lengths, offsets, page ids) are not read.
    """
    if not _env_enabled():
        return "disabled by AITER_UNIFIED_ATTN_FLYDSL=0"
    if not q.is_cuda:
        return "query is not on a GPU"
    arch = _arch()
    if arch != "gfx950":
        return f"requires gfx950, got {arch}"
    if shuffled_kv_cache or k.dim() != 4 or v.dim() != 4:
        return "requires the linear [blocks, page, kv_heads, dim] KV cache"
    tensors = (("q", q), ("k", k), ("v", v), ("out", out))
    for name, t in tensors:
        if t.dtype != torch.bfloat16:
            return f"requires bfloat16 q/k/v/out, got {name} {t.dtype}"
    if any(x is not None for x in (q_descale, k_descale, v_descale, q_scales)):
        return "BF16 route takes no q/k/v descales or q_scales"
    if output_scale is not None:
        return "output_scale is not supported"
    if alibi_slopes is not None or qq_bias is not None:
        return "alibi_slopes and qq_bias are not supported"
    if softcap not in (None, 0, 0.0):
        return f"softcap is not supported, got {softcap}"
    if skip_reduce:
        return "skip_reduce is not supported"
    if not causal:
        return "requires causal attention"
    if window_size is not None and tuple(window_size) != (-1, -1):
        return f"requires full attention, got window_size={tuple(window_size)}"
    if q.dim() != 3 or out.dim() != 3:
        return "q and out must be [tokens, heads, dim]"
    num_tokens, num_q_heads, head_dim = q.shape
    num_blocks, block_size, num_kv_heads, key_dim = k.shape
    if (head_dim, key_dim) != (_HEAD_DIM, _HEAD_DIM):
        return f"requires Q/K head dim {_HEAD_DIM}, got {head_dim}/{key_dim}"
    if tuple(v.shape) != (num_blocks, block_size, num_kv_heads, _VALUE_DIM):
        return (
            f"requires V [{num_blocks}, {block_size}, {num_kv_heads}, {_VALUE_DIM}], "
            f"got {tuple(v.shape)}"
        )
    if tuple(out.shape) != (num_tokens, num_q_heads, _VALUE_DIM):
        return f"out must be [{num_tokens}, {num_q_heads}, {_VALUE_DIM}]"
    if block_size not in _PAGE_SIZES:
        return f"requires page size in {_PAGE_SIZES}, got {block_size}"
    if num_kv_heads < 1 or num_q_heads != _GQA * num_kv_heads:
        return (
            f"requires {_GQA} query heads per KV head, got {num_q_heads}/{num_kv_heads}"
        )
    if not (k.is_contiguous() and v.is_contiguous() and out.is_contiguous()):
        return "k, v and out must be contiguous"
    if q.stride(2) != 1:
        return "q must have a contiguous head_dim axis"
    if sinks is not None and (
        sinks.shape != (num_q_heads,)
        or sinks.dtype not in (torch.bfloat16, torch.float16, torch.float32)
        or not sinks.is_contiguous()
    ):
        return "sinks must be a contiguous [num_query_heads] float tensor"
    for name, t in (("seqused_k", seqused_k), ("block_table", block_table)):
        if t.dtype != torch.int32 or not t.is_contiguous() or t.device != q.device:
            return f"{name} must be a contiguous int32 tensor on the query device"
    num_seqs = seqused_k.numel()
    if not 1 <= num_seqs <= _MAX_BATCH:
        return f"requires 1..{_MAX_BATCH} sequences, got {num_seqs}"
    if block_table.dim() != 2 or block_table.shape[0] != num_seqs:
        return "block_table must be [num_seqs, max_blocks_per_seq]"
    query_length = int(max_seqlen_q)
    if not 1 <= query_length <= _MAX_QUERY_LENGTH:
        return f"requires max_seqlen_q in 1..{_MAX_QUERY_LENGTH}, got {query_length}"
    if num_tokens != num_seqs * query_length:
        return (
            "requires a uniform query length: q has "
            f"{num_tokens} tokens for {num_seqs} sequences of {query_length}"
        )
    if not 0 <= int(max_seqlen_k) <= block_table.shape[1] * block_size:
        return "max_seqlen_k exceeds the block-table capacity"
    return None


# One grow-only backing allocation per device, stream and geometry. Calls take
# exact-sized views. Growth during capture is impossible, so a captured call
# needs an uncaptured warmup of at least its batch. Replaced allocations that a
# graph captured stay referenced so replays keep valid pointers.
_WORKSPACES: dict = {}
_COLD_CAPTURE_WARNED: set = set()


def _plan_capacity(num_seqs, num_kv_heads, num_cu):
    # plan_pa_decode's default: a 2-per-CU task budget, at least one per sequence.
    return min(num_seqs * num_cu, max(num_seqs, -(-2 * num_cu // num_kv_heads)))


def _workspace(q, num_seqs, num_kv_heads, query_length, capacity, explicit):
    stream = torch.cuda.current_stream(q.device)
    key = (q.device.index, stream.cuda_stream, num_kv_heads, q.shape[1])
    state = _WORKSPACES.get(key)
    current = None if state is None else state["current"]
    rows = query_length * (q.shape[1] // num_kv_heads)
    need = {
        "work_info": capacity * 4,
        "reduce_info": num_seqs * 2,
        "exp_sums": num_kv_heads * capacity * rows,
    }
    if current is not None and all(current[n].numel() >= c for n, c in need.items()):
        return current
    if torch.cuda.is_current_stream_capturing():
        message = (
            "FlyDSL unified_attention needs an uncaptured warmup call with at least "
            f"this batch ({num_seqs} sequences, query length {query_length}) on the "
            "capturing stream before graph capture"
        )
        if explicit:
            raise RuntimeError(message)
        if key not in _COLD_CAPTURE_WARNED:
            _COLD_CAPTURE_WARNED.add(key)
            logger.warning(f"[FlyDSL] {message}; falling back to Triton/Gluon")
        return None
    # Round up so a growing batch reallocates a logarithmic number of times.
    sizes = {n: 1 << max(c - 1, 1).bit_length() for n, c in need.items()}
    if current is not None:
        sizes = {n: max(s, current[n].numel()) for n, s in sizes.items()}
    dev = q.device
    replacement = {
        "captured": False,
        "work_info": torch.empty(sizes["work_info"], dtype=torch.int32, device=dev),
        "reduce_info": torch.empty(sizes["reduce_info"], dtype=torch.int32, device=dev),
        "exp_sums": torch.empty(sizes["exp_sums"], dtype=torch.float32, device=dev),
        "max_logits": torch.empty(sizes["exp_sums"], dtype=torch.float32, device=dev),
        "temporary_output": torch.empty(
            sizes["exp_sums"] * _VALUE_DIM, dtype=torch.bfloat16, device=dev
        ),
    }
    if state is None:
        _WORKSPACES[key] = {"current": replacement, "pinned": []}
    else:
        if current["captured"]:
            state["pinned"].append(current)
        state["current"] = replacement
    return replacement


def unified_attention_flydsl(
    q,
    k,
    v,
    out,
    cu_seqlens_q,
    max_seqlen_q,
    seqused_k,
    max_seqlen_k,
    softmax_scale,
    causal,
    window_size,
    block_table,
    softcap,
    q_descale,
    k_descale,
    v_descale,
    q_scales=None,
    alibi_slopes=None,
    output_scale=None,
    qq_bias=None,
    sinks=None,
    shuffled_kv_cache: bool = False,
    skip_reduce: bool = False,
    explicit: bool = False,
) -> torch.Tensor | None:
    """Run a validated configuration and return ``out``, else return None.

    ``cu_seqlens_q`` must equal ``arange(num_seqs + 1) * max_seqlen_q``: query
    rows are taken per sequence from the tensor shapes, never read from the GPU.
    ``seqused_k`` holds KV lengths including the query tokens, as for Triton.
    With ``explicit`` an unprepared graph capture raises instead of declining.
    """
    reason = flydsl_decline_reason(
        q,
        k,
        v,
        out,
        cu_seqlens_q,
        max_seqlen_q,
        seqused_k,
        max_seqlen_k,
        softmax_scale,
        causal,
        window_size,
        block_table,
        softcap,
        q_descale,
        k_descale,
        v_descale,
        q_scales=q_scales,
        alibi_slopes=alibi_slopes,
        output_scale=output_scale,
        qq_bias=qq_bias,
        sinks=sinks,
        shuffled_kv_cache=shuffled_kv_cache,
        skip_reduce=skip_reduce,
    )
    if reason is not None:
        return None
    from .pa_decode import PADecodePlan, pa_decode, plan_pa_decode

    num_seqs = seqused_k.numel()
    num_kv_heads = k.shape[2]
    query_length = int(max_seqlen_q)
    group = q.shape[1] // num_kv_heads
    with torch.cuda.device(q.device):
        num_cu = torch.cuda.get_device_properties(q.device).multi_processor_count
        capacity = _plan_capacity(num_seqs, num_kv_heads, num_cu)
        workspace = _workspace(
            q, num_seqs, num_kv_heads, query_length, capacity, explicit
        )
        if workspace is None:
            return None
        if torch.cuda.is_current_stream_capturing():
            workspace["captured"] = True
        shape = (num_kv_heads, capacity, query_length * group)
        scalars = shape[0] * shape[1] * shape[2]
        plan = PADecodePlan(
            work_info=workspace["work_info"][: capacity * 4].view(capacity, 4),
            reduce_info=workspace["reduce_info"][: num_seqs * 2].view(num_seqs, 2),
            num_kv_heads=num_kv_heads,
            max_partitions=num_cu,
            query_length=query_length,
        )
        # Every call records the refresh, so captured graphs follow new lengths.
        plan_pa_decode(seqused_k, num_kv_heads, query_length=query_length, plan=plan)
        pa_decode(
            out,
            q,
            k,
            v,
            seqused_k,
            block_table,
            float(softmax_scale),
            query_length,
            sinks=sinks,
            work_plan=plan,
            max_context_length=int(max_seqlen_k),
            exp_sums=workspace["exp_sums"][:scalars].view(shape),
            max_logits=workspace["max_logits"][:scalars].view(shape),
            temporary_output=workspace["temporary_output"][: scalars * _VALUE_DIM].view(
                *shape, _VALUE_DIM
            ),
            kv_layout="linear",
        )
    return out
