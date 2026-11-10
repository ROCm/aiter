# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL BF16 routes behind the public unified-attention entry point.

Validated route: BF16 paged attention on gfx950 with the linear
``[blocks, page, kv_heads, dim]`` KV cache, the MiMo V2.6 global attention
geometry (Q/K 192, V/output 128, 16 query heads per KV head), page sizes 64 and
128 and causal full attention. A uniform query length of 1 to 8 tokens per
sequence runs through ``pa_decode``. Other query lengths (prefill and mixed
batches, without sinks) split each sequence's queries into blocks of 16 that
run as pseudo-sequences of the same kernels. Causal sliding windows, with or
without sinks, take any query lengths: one CTA per query block and KV head
covers its window and writes ``out`` directly. ``unified_attention_flydsl``
returns ``None`` for anything else; ``flydsl_decline_reason`` names the first
unsupported property.

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
# Queries per varlen pseudo-sequence.
_QUERY_BLOCK = int(os.environ.get("PP_VL_QB", "16"))
# Queries per windowed varlen block: the kernel's two-CTAs-per-CU layout.
_WINDOW_QUERY_BLOCK = int(os.environ.get("PP_WIN_QB", "8"))
# Varlen pseudo-sequences the planner's LDS holds on gfx950 (160 KiB).
_MAX_PSEUDO_SEQS = 40000


def _sliding_window(window_size) -> int | None:
    """Keys a query sees including its own (0: all), as Triton's 1 + left; None if unsupported."""
    if window_size is None:
        return 0
    left, right = (int(x) for x in window_size)
    if left < -1 or right not in (-1, 0):
        return None
    return left + 1


def _env_enabled() -> bool:
    return os.environ.get("AITER_UNIFIED_ATTN_FLYDSL", "1") != "0"


def _swa_enabled() -> bool:
    return os.environ.get("AITER_UNIFIED_ATTN_FLYDSL_SWA", "1") != "0"


def _varlen_enabled() -> bool:
    return os.environ.get("AITER_UNIFIED_ATTN_FLYDSL_VARLEN", "1") != "0"


def _pseudo_capacity(num_tokens, num_seqs, block=_QUERY_BLOCK):
    # Sequence i has ceil(q_i / b) <= q_i / b + (b - 1) / b query blocks.
    return -(-num_tokens // block) + num_seqs


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
    return _route(
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
        q_scales,
        alibi_slopes,
        output_scale,
        qq_bias,
        sinks,
        shuffled_kv_cache,
        skip_reduce,
    )[1]


def _route(
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
    q_scales,
    alibi_slopes,
    output_scale,
    qq_bias,
    sinks,
    shuffled_kv_cache,
    skip_reduce,
) -> tuple[str | None, str | None]:
    """(route, None) or (None, reason); routes "window" and "window_varlen"
    are the sliding-window variants of "uniform" and "varlen"."""
    reason = _common_decline(
        q,
        k,
        v,
        out,
        seqused_k,
        max_seqlen_k,
        causal,
        window_size,
        block_table,
        softcap,
        q_descale,
        k_descale,
        v_descale,
        q_scales,
        alibi_slopes,
        output_scale,
        qq_bias,
        sinks,
        shuffled_kv_cache,
        skip_reduce,
    )
    if reason is not None:
        return None, reason
    num_tokens = q.shape[0]
    num_seqs = seqused_k.numel()
    query_length = int(max_seqlen_q)
    uniform = (
        1 <= query_length <= _MAX_QUERY_LENGTH and num_tokens == num_seqs * query_length
    )
    windowed = bool(_sliding_window(window_size))
    if uniform:
        return ("window" if windowed else "uniform"), None
    if not _varlen_enabled():
        return None, (
            "non-uniform or > 8 query lengths are disabled by "
            "AITER_UNIFIED_ATTN_FLYDSL_VARLEN=0"
        )
    if sinks is not None and not windowed:
        return (
            None,
            "sinks without a sliding window require a uniform query length of 1..8",
        )
    if (
        cu_seqlens_q is None
        or cu_seqlens_q.dtype != torch.int32
        or not cu_seqlens_q.is_contiguous()
        or cu_seqlens_q.device != q.device
        or cu_seqlens_q.shape != (num_seqs + 1,)
    ):
        return None, "cu_seqlens_q must be a contiguous int32 [num_seqs + 1] tensor"
    if query_length < 1:
        return None, f"requires max_seqlen_q >= 1, got {query_length}"
    if not windowed and _pseudo_capacity(num_tokens, num_seqs) > _MAX_PSEUDO_SEQS:
        return None, f"{num_tokens} query tokens exceed the varlen planner"
    return ("window_varlen" if windowed else "varlen"), None


def _common_decline(
    q,
    k,
    v,
    out,
    seqused_k,
    max_seqlen_k,
    causal,
    window_size,
    block_table,
    softcap,
    q_descale,
    k_descale,
    v_descale,
    q_scales,
    alibi_slopes,
    output_scale,
    qq_bias,
    sinks,
    shuffled_kv_cache,
    skip_reduce,
) -> str | None:
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
    window = _sliding_window(window_size)
    if window is None:
        return f"requires a causal left window, got window_size={tuple(window_size)}"
    if window and not _swa_enabled():
        return "sliding windows are disabled by AITER_UNIFIED_ATTN_FLYDSL_SWA=0"
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


def _workspace(q, route, need, describe, explicit):
    """Grow-only buffers ``need`` = {name: (numel, dtype)}, or None if capturing."""
    stream = torch.cuda.current_stream(q.device)
    key = (q.device.index, stream.cuda_stream, route, q.shape[1])
    state = _WORKSPACES.get(key)
    current = None if state is None else state["current"]
    if current is not None and all(
        current[n].numel() >= c for n, (c, _) in need.items()
    ):
        return current
    if torch.cuda.is_current_stream_capturing():
        message = (
            "FlyDSL unified_attention needs an uncaptured warmup call with at least "
            f"this batch ({describe}) on the capturing stream before graph capture"
        )
        if explicit:
            raise RuntimeError(message)
        if key not in _COLD_CAPTURE_WARNED:
            _COLD_CAPTURE_WARNED.add(key)
            logger.warning(f"[FlyDSL] {message}; falling back to Triton/Gluon")
        return None
    # Round up so a growing batch reallocates a logarithmic number of times.
    sizes = {n: 1 << max(c - 1, 1).bit_length() for n, (c, _) in need.items()}
    if current is not None:
        sizes = {n: max(s, current[n].numel()) for n, s in sizes.items()}
    replacement = {"captured": False}
    for name, (_, dtype) in need.items():
        replacement[name] = torch.empty(sizes[name], dtype=dtype, device=q.device)
    if state is None:
        _WORKSPACES[key] = {"current": replacement, "pinned": []}
    else:
        if current["captured"]:
            state["pinned"].append(current)
        state["current"] = replacement
    return replacement


def _partial_need(num_kv_heads, capacity, rows):
    scalars = num_kv_heads * capacity * rows
    return {
        "exp_sums": (scalars, torch.float32),
        "max_logits": (scalars, torch.float32),
        "temporary_output": (scalars * _VALUE_DIM, torch.bfloat16),
    }


def _build_query_map(cu_seqlens_q, seqused_k, num_tokens, query_map, context_lengths):
    """Fill pseudo-sequence rows (first query row, sequence, valid queries, 0).

    Each sequence's queries split into blocks of 8; block b of a sequence with
    q queries and k KV tokens sees k - q + 8 b + valid tokens. Rows past the
    last block get no queries and no context. Rows are sorted by context,
    longest first, so the longest tasks start first. Queries at or past
    ``num_tokens`` are dropped, so offsets beyond q never address memory.
    """
    capacity = query_map.shape[0]
    dev = cu_seqlens_q.device
    query_lengths = (cu_seqlens_q[1:] - cu_seqlens_q[:-1]).long()
    blocks = (query_lengths + (_QUERY_BLOCK - 1)) // _QUERY_BLOCK
    block_ends = torch.cumsum(blocks, 0)
    pseudo = torch.arange(capacity, device=dev)
    seq = torch.searchsorted(block_ends, pseudo, right=True)
    live = seq < blocks.numel()
    seq = seq.clamp_(max=blocks.numel() - 1)
    first = (pseudo - (block_ends[seq] - blocks[seq])) * _QUERY_BLOCK
    row = cu_seqlens_q[seq].long() + first
    valid = torch.minimum(query_lengths[seq] - first, num_tokens - row)
    valid = torch.where(live, valid.clamp(0, _QUERY_BLOCK), 0)
    context = torch.where(
        live, seqused_k[seq].long() - query_lengths[seq] + first + valid, 0
    ).clamp_(min=0)
    order = torch.argsort(context, descending=True, stable=True)
    rows = torch.stack(
        (row.clamp(0, max(num_tokens - 1, 0)), seq, valid, torch.zeros_like(seq)), 1
    )
    query_map.copy_(rows[order])
    context_lengths.copy_(context[order])


def _unified_attention_varlen(
    q,
    k,
    v,
    out,
    cu_seqlens_q,
    seqused_k,
    max_seqlen_k,
    softmax_scale,
    block_table,
    explicit,
):
    import flydsl.compiler as flyc
    import flydsl.expr as fx

    from .kernels.pa_decode_pingpong import compile_pa_decode_pingpong
    from .kernels.tensor_shim import _run_compiled, ptr_arg
    from .pa_decode import PADecodePlan, launch_pa_decode_ps_reduce, plan_pa_decode

    num_tokens = q.shape[0]
    num_seqs = seqused_k.numel()
    num_kv_heads = k.shape[2]
    group = q.shape[1] // num_kv_heads
    rows = _QUERY_BLOCK * group
    pseudo = _pseudo_capacity(num_tokens, num_seqs)
    dev = q.device
    with torch.cuda.device(dev):
        num_cu = torch.cuda.get_device_properties(dev).multi_processor_count
        capacity = _plan_capacity(pseudo, num_kv_heads, num_cu)
        capacity += int(os.environ.get("PP_VL_EXTRA", "0"))
        need = {
            "query_map": (pseudo * 4, torch.int32),
            "context_lengths": (pseudo, torch.int32),
            "work_info": (capacity * 4, torch.int32),
            "reduce_info": (pseudo * 2, torch.int32),
            **_partial_need(num_kv_heads, capacity, rows),
        }
        workspace = _workspace(
            q,
            "varlen",
            need,
            f"{num_seqs} sequences, {num_tokens} query tokens",
            explicit,
        )
        if workspace is None:
            return None
        if torch.cuda.is_current_stream_capturing():
            workspace["captured"] = True
        query_map = workspace["query_map"][: pseudo * 4].view(pseudo, 4)
        context_lengths = workspace["context_lengths"][:pseudo]
        _build_query_map(
            cu_seqlens_q, seqused_k, num_tokens, query_map, context_lengths
        )
        plan = PADecodePlan(
            work_info=workspace["work_info"][: capacity * 4].view(capacity, 4),
            reduce_info=workspace["reduce_info"][: pseudo * 2].view(pseudo, 2),
            num_kv_heads=num_kv_heads,
            max_partitions=num_cu,
            query_length=_QUERY_BLOCK,
            max_batch_size=pseudo,
        )
        plan_pa_decode(
            context_lengths, num_kv_heads, query_length=_QUERY_BLOCK, plan=plan
        )
        shape = (num_kv_heads, capacity, rows)
        scalars = shape[0] * shape[1] * shape[2]
        pmax = workspace["max_logits"][:scalars].view(shape)
        psum = workspace["exp_sums"][:scalars].view(shape)
        pout = workspace["temporary_output"][: scalars * _VALUE_DIM].view(
            *shape, _VALUE_DIM
        )
        compiled = compile_pa_decode_pingpong(
            head_dim=_HEAD_DIM,
            value_dim=_VALUE_DIM,
            query_group_size=group,
            query_length=_QUERY_BLOCK,
            block_size=k.shape[1],
            softmax_scale=float(softmax_scale),
            # Longest-first tasks balance best one per CTA: pairing gives the
            # first CTAs the two longest (15-60% slower on prefill steps).
            pair_ctas=0,
            varlen=True,
        )
        stream = torch.cuda.current_stream(dev)
        no_scale = flyc.from_c_void_p(fx.Float32, 0)
        _run_compiled(
            compiled["launch"],
            ptr_arg(pmax, fx.Float32),
            ptr_arg(psum, fx.Float32),
            ptr_arg(pout, fx.BFloat16),
            ptr_arg(q, fx.BFloat16),
            ptr_arg(k, fx.BFloat16),
            ptr_arg(v, fx.BFloat16),
            ptr_arg(block_table, fx.Int32),
            no_scale,
            no_scale,
            int(block_table.shape[1]),
            pseudo,
            num_kv_heads,
            0,
            0,
            int(q.stride(0)),
            int(q.stride(1)),
            ptr_arg(plan.work_info, fx.Int32),
            capacity,
            ptr_arg(query_map, fx.Int32),
            stream,
        )
        # Pseudo-sequence p writes rows query_map[p, 0] + [0, valid) of out; the
        # view supplies the batch and head geometry, and fits short outputs.
        output = out.as_strided(
            (pseudo, 1, num_kv_heads, group, _VALUE_DIM),
            (0, out.stride(0), group * out.stride(1), out.stride(1), 1),
        )
        launch_pa_decode_ps_reduce(
            output,
            psum,
            pmax,
            pout,
            None,
            0,
            output.stride(1),
            output.stride(2),
            output.stride(3),
            pmax.stride(0),
            pmax.stride(1),
            pout.stride(0),
            pout.stride(1),
            pout.stride(2),
            query_seq_len=_QUERY_BLOCK,
            query_group_size=group,
            head_size=_VALUE_DIM,
            # Every sequence owns at least one of the capacity's task slots.
            max_partitions=max(
                1, min(num_cu, capacity - pseudo + 1, -(-int(max_seqlen_k) // 256))
            ),
            stream=stream,
            reduce_info=plan.reduce_info,
            query_map=query_map,
        )
    return out


_SINK_DTYPES = {torch.float32: "f32", torch.bfloat16: "bf16", torch.float16: "f16"}


def _unified_attention_window(
    q,
    k,
    v,
    out,
    cu_seqlens_q,
    seqused_k,
    query_length,
    softmax_scale,
    block_table,
    window,
    sinks,
    explicit,
):
    """One CTA per (sequence, KV head) over its window; writes out directly.

    ``query_length`` 1..8 takes uniform rows; None splits ``cu_seqlens_q`` into
    pseudo-sequences.
    """
    import flydsl.expr as fx

    from .kernels.pa_decode_pingpong import compile_pa_decode_pingpong
    from .kernels.pa_decode_plan import build_query_block_map
    from .kernels.tensor_shim import _run_compiled, ptr_arg

    num_tokens = q.shape[0]
    num_seqs = seqused_k.numel()
    num_kv_heads = k.shape[2]
    group = q.shape[1] // num_kv_heads
    dev = q.device
    with torch.cuda.device(dev):
        num_cu = torch.cuda.get_device_properties(dev).multi_processor_count
        cta_queries = query_length
        if query_length is None:
            pseudo = _pseudo_capacity(num_tokens, num_seqs, _WINDOW_QUERY_BLOCK)
            need = {
                "query_map": (pseudo * 4, torch.int32),
                "context_lengths": (pseudo, torch.int32),
            }
            workspace = _workspace(
                q,
                "window_varlen",
                need,
                f"{num_seqs} sequences, {num_tokens} query tokens",
                explicit,
            )
            if workspace is None:
                return None
            if torch.cuda.is_current_stream_capturing():
                workspace["captured"] = True
            query_map = workspace["query_map"][: pseudo * 4].view(pseudo, 4)
            contexts = workspace["context_lengths"][:pseudo]
            build_query_block_map(
                cu_seqlens_q,
                seqused_k,
                num_tokens,
                _WINDOW_QUERY_BLOCK,
                query_map,
                contexts,
            )
            grid_seqs = pseudo
        else:
            # A CTA's fixed latency dwarfs a window's few tiles: spread each
            # sequence's queries over CTAs while they fit the CUs.
            cta_queries = int(os.environ.get("PP_WIN_Q", "0")) or next(
                d
                for d in range(1, query_length + 1)
                if query_length % d == 0
                and (
                    num_seqs * num_kv_heads * (query_length // d) <= num_cu
                    or d == query_length
                )
            )
            query_map = seqused_k
            contexts = seqused_k
            grid_seqs = num_seqs * (query_length // cta_queries)
        compiled = compile_pa_decode_pingpong(
            head_dim=_HEAD_DIM,
            value_dim=_VALUE_DIM,
            query_group_size=group,
            query_length=_WINDOW_QUERY_BLOCK if query_length is None else cta_queries,
            block_size=k.shape[1],
            softmax_scale=float(softmax_scale),
            varlen=query_length is None,
            window=window,
            sink_dtype=None if sinks is None else _SINK_DTYPES[sinks.dtype],
            seq_query_length=query_length or 0,
        )
        sink_arg = ptr_arg(contexts, fx.Int32)
        if sinks is not None:
            sink_arg = ptr_arg(
                sinks,
                {"f32": fx.Float32, "bf16": fx.BFloat16, "f16": fx.Float16}[
                    _SINK_DTYPES[sinks.dtype]
                ],
            )
        _run_compiled(
            compiled["launch"],
            ptr_arg(out, fx.BFloat16),
            ptr_arg(q, fx.BFloat16),
            ptr_arg(k, fx.BFloat16),
            ptr_arg(v, fx.BFloat16),
            ptr_arg(block_table, fx.Int32),
            int(block_table.shape[1]),
            int(q.stride(0)),
            int(q.stride(1)),
            ptr_arg(contexts, fx.Int32),
            sink_arg,
            ptr_arg(query_map, fx.Int32),
            grid_seqs,
            num_kv_heads,
            torch.cuda.current_stream(dev),
        )
    return out


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

    When q has ``num_seqs * max_seqlen_q`` tokens and ``max_seqlen_q`` <= 8,
    ``cu_seqlens_q`` must equal ``arange(num_seqs + 1) * max_seqlen_q``: query
    rows are then taken from the tensor shapes. Otherwise ``cu_seqlens_q`` is
    read on the GPU and must cover q. ``seqused_k`` holds KV lengths including
    the query tokens, as for Triton; each must be at least its query count.
    With ``explicit``, a declined configuration or an unprepared graph capture
    raises RuntimeError naming the cause instead of returning None.
    """
    route, reason = _route(
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
        q_scales,
        alibi_slopes,
        output_scale,
        qq_bias,
        sinks,
        shuffled_kv_cache,
        skip_reduce,
    )
    if route is None:
        if explicit:
            raise RuntimeError(
                "FlyDSL unified_attention does not support this configuration: "
                f"{reason}"
            )
        return None
    if route in ("window", "window_varlen"):
        return _unified_attention_window(
            q,
            k,
            v,
            out,
            cu_seqlens_q,
            seqused_k,
            int(max_seqlen_q) if route == "window" else None,
            softmax_scale,
            block_table,
            min(_sliding_window(window_size), 1 << 30),
            sinks,
            explicit,
        )
    if route == "varlen":
        return _unified_attention_varlen(
            q,
            k,
            v,
            out,
            cu_seqlens_q,
            seqused_k,
            max_seqlen_k,
            softmax_scale,
            block_table,
            explicit,
        )
    from .pa_decode import PADecodePlan, pa_decode, plan_pa_decode

    num_seqs = seqused_k.numel()
    num_kv_heads = k.shape[2]
    query_length = int(max_seqlen_q)
    group = q.shape[1] // num_kv_heads
    sliding_window = _sliding_window(window_size)
    with torch.cuda.device(q.device):
        num_cu = torch.cuda.get_device_properties(q.device).multi_processor_count
        capacity = _plan_capacity(num_seqs, num_kv_heads, num_cu)
        need = {
            "work_info": (capacity * 4, torch.int32),
            "reduce_info": (num_seqs * 2, torch.int32),
            **_partial_need(num_kv_heads, capacity, query_length * group),
        }
        workspace = _workspace(
            q,
            "uniform",
            need,
            f"{num_seqs} sequences, query length {query_length}",
            explicit,
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
            sliding_window=sliding_window,
            query_length=query_length,
        )
        # Every call records the refresh, so captured graphs follow new lengths.
        plan_pa_decode(
            seqused_k,
            num_kv_heads,
            sliding_window=sliding_window,
            query_length=query_length,
            plan=plan,
        )
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
            sliding_window=sliding_window,
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
