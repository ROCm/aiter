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

Hardware: every route accepts any gfx950 device; the optimized varlen
specialization also needs 256 CUs, so MI350X and MI355X take it and MI350P
does not. Correctness depends on the ISA, not the SKU: plans read the device's
CU count. Performance was measured on MI355X only; MI350X is the same chip
at a lower power cap and clock, and ``AITER_UNIFIED_ATTN_FLYDSL_VARLEN_OPTIMIZED=0``
or ``AITER_UNIFIED_ATTN_FLYDSL=0`` opts out if it regresses there.

Graph capture: the first call of a geometry on a stream must run uncaptured to
allocate its workspace. Each call, captured or not, records a GPU plan refresh
from ``seqused_k``, so lengths may change between replays up to the captured
``max_seqlen_k``. Calls and replays sharing a workspace must not overlap.
Workspaces and compiled kernels persist until ``release_flydsl_resources``.

The optimized varlen specialization is additionally limited to the global page-128,
HKV=1, direct-output plan with ``max_seqlen_k >= 4096``. Set
``AITER_UNIFIED_ATTN_FLYDSL_VARLEN_OPTIMIZED=0`` to retain the prior varlen
kernel while leaving other FlyDSL routes enabled.
"""

from __future__ import annotations

import functools
import os

import torch

from aiter import logger

__all__ = [
    "flydsl_decline_reason",
    "release_flydsl_resources",
    "unified_attention_flydsl",
]

_HEAD_DIM = 192
_VALUE_DIM = 128
_GQA = 16
_PAGE_SIZES = (64, 128)
_MAX_QUERY_LENGTH = 8
_MAX_BATCH = 4096
# Queries per varlen pseudo-sequence.
_QUERY_BLOCK = 16
# Queries per windowed varlen block: the kernel's two-CTAs-per-CU layout.
_WINDOW_QUERY_BLOCK = 8
_GFX950_LDS_BYTES = 160 * 1024
# Varlen pseudo-sequences the planner's LDS holds on gfx950.
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


def _varlen_optimized_enabled() -> bool:
    return os.environ.get("AITER_UNIFIED_ATTN_FLYDSL_VARLEN_OPTIMIZED", "1") != "0"


def _pseudo_capacity(num_tokens, num_seqs, block=_QUERY_BLOCK):
    # Sequence i has ceil(q_i / b) <= q_i / b + (b - 1) / b query blocks.
    return -(-num_tokens // block) + num_seqs


@functools.cache
def _num_cu(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


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
    if not uniform:
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
    # Every route but the uniform window one reads query rows from cu_seqlens_q.
    if not (uniform and windowed) and (
        cu_seqlens_q is None
        or cu_seqlens_q.dtype != torch.int32
        or not cu_seqlens_q.is_contiguous()
        or cu_seqlens_q.device != q.device
        or cu_seqlens_q.shape != (num_seqs + 1,)
    ):
        return None, "cu_seqlens_q must be a contiguous int32 [num_seqs + 1] tensor"
    if uniform:
        return ("window" if windowed else "uniform"), None
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
# graph captured stay referenced so replays keep valid pointers. Nothing here
# knows when a graph or stream dies, so only release_flydsl_resources frees them.
_WORKSPACES: dict = {}
_COLD_CAPTURE_WARNED: set = set()


def release_flydsl_resources() -> None:
    """Free the workspaces and compiled kernels FlyDSL unified attention keeps.

    Workspaces (including allocations pinned for captured graphs) and compiled
    kernels otherwise live for the whole process. Call only once no CUDA graph
    that captured a FlyDSL unified-attention call will replay again: replays read
    the freed workspaces, and dropping a compiled kernel unloads its HIP module.
    The next call allocates and compiles again, so capture once more needs an
    uncaptured warmup.
    """
    import sys

    _WORKSPACES.clear()
    _COLD_CAPTURE_WARNED.clear()
    for name, factories in (
        (f"{__package__}.kernels.pa_decode_pingpong", ("compile_pa_decode_pingpong",)),
        (
            f"{__package__}.unified_attention_varlen",
            ("compile_tile_kernel", "compile_plan", "compile_tail_reduce"),
        ),
    ):
        # A module not yet imported has compiled nothing.
        module = sys.modules.get(name)
        if module is not None:
            for factory in factories:
                getattr(module, factory).cache_clear()


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
        n in current and current[n].numel() >= c for n, (c, _) in need.items()
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
    # Buffers that fit are kept, others grow, rounded up so a growing batch
    # reallocates a logarithmic number of times. Buffers this call does not
    # need (another plan of the same route) are kept for later calls.
    replacement = {} if current is None else dict(current)
    replacement["captured"] = False
    for name, (count, dtype) in need.items():
        old = replacement.get(name)
        if old is None or old.numel() < count:
            size = 1 << max(count - 1, 1).bit_length()
            if old is not None:
                size = max(size, old.numel())
            replacement[name] = torch.empty(size, dtype=dtype, device=q.device)
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


def _varlen_optimized_eligible(q, k, out, max_seqlen_k):
    """Whether the optimized varlen specialization (``unified_attention_varlen``)
    covers this call: one KV head, page 128, a context of at least 4096 tokens
    and a contiguous ``out``. Every miss runs the generic varlen kernel.
    """
    if not _varlen_optimized_enabled():
        return False
    if k.shape[1] != 128 or k.shape[2] != 1 or int(max_seqlen_k) < 4096:
        return False
    return out.stride() == (q.shape[1] * _VALUE_DIM, _VALUE_DIM, 1)


def _use_varlen_optimized(q, k, out, num_seqs, max_seqlen_k, num_cu):
    if not _varlen_enabled() or not _varlen_optimized_eligible(q, k, out, max_seqlen_k):
        return False
    # The specialization's schedule is tuned on the 256-CU MI355X. MI350X has the
    # same 256 CUs and takes it unmeasured; parts with fewer CUs keep the generic
    # kernel.
    if num_cu != 256:
        return False
    pseudo = _pseudo_capacity(q.shape[0], num_seqs)
    return _plan_capacity(pseudo, 1, num_cu) == pseudo


def _unified_attention_varlen_optimized(
    q,
    k,
    v,
    out,
    cu_seqlens_q,
    seqused_k,
    softmax_scale,
    block_table,
    num_cu,
    explicit,
):
    from . import unified_attention_varlen

    spec = unified_attention_varlen.workspace_requirements(
        q.shape[0],
        seqused_k.numel(),
        num_cu,
    )
    if spec is None:
        return None
    need, _, _, _ = spec
    workspace = _workspace(
        q,
        "varlen_optimized",
        need,
        f"{seqused_k.numel()} sequences, {q.shape[0]} query tokens",
        explicit,
    )
    if workspace is None:
        return None
    if torch.cuda.is_current_stream_capturing():
        workspace["captured"] = True
    return unified_attention_varlen.run(
        q,
        k,
        v,
        out,
        cu_seqlens_q,
        seqused_k,
        block_table,
        softmax_scale,
        workspace,
        num_cu,
    )


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
    query_block=_QUERY_BLOCK,
):
    import flydsl.compiler as flyc
    import flydsl.expr as fx

    from .kernels.pa_decode_pingpong import compile_pa_decode_pingpong
    from .kernels.pa_decode_plan import (
        PADecodePlan,
        _lds_max_batch,
        build_query_block_map,
        refresh_plan,
    )
    from .kernels.tensor_shim import _run_compiled, ptr_arg
    from .pa_decode import launch_pa_decode_ps_reduce

    num_tokens = q.shape[0]
    num_seqs = seqused_k.numel()
    num_kv_heads = k.shape[2]
    group = q.shape[1] // num_kv_heads
    rows = query_block * group
    pseudo = _pseudo_capacity(num_tokens, num_seqs, query_block)
    dev = q.device
    with torch.cuda.device(dev):
        num_cu = _num_cu(dev.index)
        capacity = _plan_capacity(pseudo, num_kv_heads, num_cu)
        # With a slot per pseudo-sequence the batch already fills the CUs:
        # one task each, written straight to out, and no reducer.
        direct = capacity == pseudo and out.stride() == (
            q.shape[1] * _VALUE_DIM,
            _VALUE_DIM,
            1,
        )
        # Otherwise spare slots go to the longest rows: a row has at most one
        # task plus the slots beyond the live rows, of which there are at least
        # ceil(tokens / b).
        max_partitions = (
            1 if direct else min(num_cu, capacity - -(-num_tokens // query_block) + 1)
        )
        need = {
            "query_map": (pseudo * 4, torch.int32),
            "context_lengths": (pseudo, torch.int32),
            "work_info": (capacity * 4, torch.int32),
            "reduce_info": (pseudo * 2, torch.int32),
            **({} if direct else _partial_need(num_kv_heads, capacity, rows)),
        }
        workspace = _workspace(
            q,
            (
                ("varlen_direct" if direct else "varlen")
                if query_block == _QUERY_BLOCK
                else f"uniform_varlen_{query_block}"
            ),
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
        build_query_block_map(
            cu_seqlens_q,
            seqused_k,
            num_tokens,
            query_block,
            query_map,
            context_lengths,
            longest_first=True,
        )
        plan = PADecodePlan(
            work_info=workspace["work_info"][: capacity * 4].view(capacity, 4),
            reduce_info=workspace["reduce_info"][: pseudo * 2].view(pseudo, 2),
            num_kv_heads=num_kv_heads,
            max_partitions=max_partitions,
            query_length=query_block,
            max_batch_size=pseudo,
        )
        stream = torch.cuda.current_stream(dev)
        # The plan is valid by construction, so skip plan_pa_decode's checks.
        refresh_plan(
            plan, context_lengths, _lds_max_batch(_GFX950_LDS_BYTES), True, stream
        )
        no_scale = flyc.from_c_void_p(fx.Float32, 0)
        if direct:
            partials = (no_scale, no_scale, flyc.from_c_void_p(fx.BFloat16, 0))
        else:
            shape = (num_kv_heads, capacity, rows)
            scalars = shape[0] * shape[1] * shape[2]
            pmax = workspace["max_logits"][:scalars].view(shape)
            psum = workspace["exp_sums"][:scalars].view(shape)
            pout = workspace["temporary_output"][: scalars * _VALUE_DIM].view(
                *shape, _VALUE_DIM
            )
            partials = (
                ptr_arg(pmax, fx.Float32),
                ptr_arg(psum, fx.Float32),
                ptr_arg(pout, fx.BFloat16),
            )
        compiled = compile_pa_decode_pingpong(
            head_dim=_HEAD_DIM,
            value_dim=_VALUE_DIM,
            query_group_size=group,
            query_length=query_block,
            block_size=k.shape[1],
            softmax_scale=float(softmax_scale),
            varlen=True,
            direct_output=direct,
        )
        _run_compiled(
            compiled,
            *partials,
            ptr_arg(q, fx.BFloat16),
            ptr_arg(k, fx.BFloat16),
            ptr_arg(v, fx.BFloat16),
            ptr_arg(block_table, fx.Int32),
            ptr_arg(out, fx.BFloat16) if direct else no_scale,
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
        if direct:
            return out
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
            query_seq_len=query_block,
            query_group_size=group,
            head_size=_VALUE_DIM,
            max_partitions=max(1, min(max_partitions, -(-int(max_seqlen_k) // 256))),
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
        num_cu = _num_cu(dev.index)
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
            cta_queries = next(
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
            compiled,
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

    ``cu_seqlens_q`` is read on the GPU and must cover q. When q has
    ``num_seqs * max_seqlen_q`` tokens and ``max_seqlen_q`` <= 8 it must equal
    ``arange(num_seqs + 1) * max_seqlen_q``; with a sliding window those rows
    are then taken from the tensor shapes and ``cu_seqlens_q`` is not read.
    ``seqused_k`` holds KV lengths including
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
        num_cu = _num_cu(q.device.index)
        if _use_varlen_optimized(
            q,
            k,
            out,
            seqused_k.numel(),
            max_seqlen_k,
            num_cu,
        ):
            return _unified_attention_varlen_optimized(
                q,
                k,
                v,
                out,
                cu_seqlens_q,
                seqused_k,
                softmax_scale,
                block_table,
                num_cu,
                explicit,
            )
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
    # Main's shared PA decode uses its native vectorized K/V layout. Reuse the
    # query-mapped ping-pong path for this route's linear cache; uniform batches
    # are a strict subset of its varlen plan.
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
        query_block=int(max_seqlen_q),
    )
