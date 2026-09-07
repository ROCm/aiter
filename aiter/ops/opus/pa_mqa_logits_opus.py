# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""MXFP4 paged MQA logits for DeepSeek-style sparse attention on gfx950 (OPUS kernel).

Per query row ``r`` over a window ``[s, e)``:
``out[r, s:e] = sum_H( relu(Q[r] . K^T) * weight[r] ) * weight_scale``

Quantization and layout are the CALLER's responsibility; this module never touches the
data. Of the five inputs only the two E8M0 scale arrays need a layout specific to this
kernel -- the rest are natural or already-standard:

===============  ===========================================  ==================
tensor           shape                                        layout
===============  ===========================================  ==================
``q``            ``[total_q, H, D/2]`` uint8                   natural
``weights``      ``[total_q, H]`` bfloat16                     natural
``kv_cache``     ``[num_blocks, 4, PAGE, 16]`` uint8           standard paged fp4
``q_scale``      ``[total_q, 2, 32, 4]`` uint8                 kernel-specific
``kv_scale``     ``[num_blocks, 2, 32, 4]`` uint8              kernel-specific
===============  ===========================================  ==================

``kv_cache[blk, b, o, :]`` holds the 16 packed bytes of
``K[token o of page blk][32*b : 32*b+32]``. ``q`` is a plain per-head packed row; the low
nibble is the even element.

THE TWO SCALE LAYOUTS
---------------------
Both are E8M0 bytes indexed ``[.., g, m, byte]``, where a lane's four bytes must land in
ONE aligned dword so the MFMA's ``op_sel`` can select among them. With ``H = 64``,
``D = 128``, ``PAGE = 64``, and ``e8[i, b]`` the natural per-row E8M0 for 32-K block ``b``
of row ``i`` (``b`` in ``[0, 4)``)::

    kt, g = divmod(b, 2)                       # k-tile, 32-K chunk within it

    # q_scale: head h contributes to (mi, m) = divmod(h, 32)
    q_scale[t, g, m, kt * 2 + mi] = e8_q[t * H + h, b]

    # kv_scale: page token o contributes to (nt, m) = divmod(o, 32)
    kv_scale[blk, g, m, kt * 2 + nt] = e8_kv[token of (blk, o), b]

Equivalently, as a permutation of the natural arrays::

    q_scale  = e8_q.view(T, 2, 32, 2, 2).permute(0, 4, 2, 3, 1).reshape(T, 2, 32, 4)
    kv_scale = e8_kv.view(-1, 2, 32, 2, 2).permute(0, 4, 2, 3, 1).reshape(-1, 2, 32, 4)

**Getting these wrong is silent.** Every fp4 scale layout has the same byte count, so only
the permutation differs and a wrong one yields plausible-looking wrong logits. The C++ side
checks sizes, which catches passing the wrong array but not a wrong permutation -- validate
against a dequantized reference once, on RANDOM data (a uniform-data check passes under any
permutation of K).

Both entry points are schedule-free (the per-CTA assignment is derived in-kernel from
``blockIdx``) and cudagraph-safe. ``block_k`` picks between two compiled variants that
produce identical results: 256 -> 4 waves/CTA, 64 -> 1 wave/CTA. It is a pure performance
knob; see :func:`pa_mqa_logits_mxfp4_prefill` for how to choose.
"""

import torch

from ...jit.core import compile_ops
from ...jit.utils.chip_info import get_gfx_runtime

MD_NAME_MXFP4 = "module_pa_mqa_logits_mxfp4_opus"

DEFAULT_HEADS = 64
DEFAULT_HEAD_DIM = 128

# The two compiled variants. 64 (1 wave/CTA) is the better single default on every
# path measured so far; 256 (4 waves/CTA) only pays off once one CTA's window is long
# enough to amortize the wider tiles.
BLOCK_K_1WAVE = 64
BLOCK_K_4WAVE = 256


# ── JIT stubs: signatures must match PA_MQA_LOGITS_MXFP4_PYBIND exactly ───────
# Importing this module builds nothing; the JIT module is compiled on first call.
@compile_ops(MD_NAME_MXFP4, develop=True)
def pa_mqa_logits_mxfp4_fwd_prefill(
    q: torch.Tensor,
    q_scale: torch.Tensor,
    kv_cache: torch.Tensor,
    kv_scale: torch.Tensor,
    block_tables: torch.Tensor,
    weights: torch.Tensor,
    row_to_batch: torch.Tensor,
    local_starts: torch.Tensor,
    local_ends: torch.Tensor,
    out: torch.Tensor,
    num_rows: int,
    weight_scale: float,
    block_k: int,
    kv_block_size: int,
    max_seq_len: int,
) -> None: ...


@compile_ops(MD_NAME_MXFP4, develop=True)
def pa_mqa_logits_mxfp4_fwd_decode(
    q: torch.Tensor,
    q_scale: torch.Tensor,
    kv_cache: torch.Tensor,
    kv_scale: torch.Tensor,
    block_tables: torch.Tensor,
    weights: torch.Tensor,
    cu_seq_q: torch.Tensor,
    context_lens: torch.Tensor,
    out: torch.Tensor,
    batch: int,
    next_n_max: int,
    split_kv: int,
    weight_scale: float,
    block_k: int,
    kv_block_size: int,
    max_seq_len: int,
) -> None: ...


@compile_ops(MD_NAME_MXFP4, develop=True)
def pa_mqa_logits_mxfp4_prefill_windows(
    cu_seq_q: torch.Tensor,
    context_lens: torch.Tensor,
    row_to_batch: torch.Tensor,
    local_starts: torch.Tensor,
    local_ends: torch.Tensor,
    total_q: int,
) -> None: ...


def compute_prefill_windows(
    cu_seq_q: torch.Tensor,
    context_lens: torch.Tensor,
    total_q: int,
    out: tuple | None = None,
):
    """Build the per-row ``[local_start, local_end)`` window arrays the prefill launch
    consumes, from ``cu_seq_q`` + ``context_lens``. Device-side, cudagraph-safe.

    The rule is MTP tail-causal -- batch ``b``'s ``n``-th row sees
    ``[0, context_lens[b] - (qlen - 1 - n))``, which reduces to plain causal when
    ``qlen == ctx``. That is the only rule this builder can express, and it is NOT the
    rule a compressed KV cache follows: with CSA compression at ratio ``R`` a row sees
    ``floor((pos + 1) / R)``, and ``floor((x - d) / R) != floor(x / R) - d``. Such a
    caller must build ``local_ends`` itself and pass it to the prefill entry point
    directly; the arrays are plain int32 and carry no other constraint than
    ``0 <= local_start <= local_end``.
    """
    dev = cu_seq_q.device
    cu = cu_seq_q.to(torch.int32).contiguous()
    ctx = context_lens.to(torch.int32).contiguous()
    if out is None:
        row_to_batch = torch.empty(total_q, dtype=torch.int32, device=dev)
        local_starts = torch.empty(total_q, dtype=torch.int32, device=dev)
        local_ends = torch.empty(total_q, dtype=torch.int32, device=dev)
    else:
        row_to_batch, local_starts, local_ends = out
    pa_mqa_logits_mxfp4_prefill_windows(
        cu, ctx, row_to_batch, local_starts, local_ends, int(total_q)
    )
    return row_to_batch, local_starts, local_ends


def _require_gfx950(name):
    gfx = get_gfx_runtime()
    if gfx != "gfx950":
        raise RuntimeError(f"{name} requires gfx950, got {gfx}")


def pa_mqa_logits_mxfp4_prefill(
    q_fp4: torch.Tensor,
    q_scale: torch.Tensor,
    kv_cache: torch.Tensor,
    kv_scale: torch.Tensor,
    block_tables: torch.Tensor,
    weights: torch.Tensor,
    row_to_batch: torch.Tensor,
    local_starts: torch.Tensor,
    local_ends: torch.Tensor,
    max_seq_len: int,
    *,
    weight_scale: float = 1.0,
    block_k: int = BLOCK_K_1WAVE,
    kv_block_size: int = 64,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Ragged-prefill paged MQA logits (gfx950), schedule-free: one CTA per query row
    (1D grid), each covering its whole ``[local_start, local_end)`` window.

    See the module docstring for the input layouts, in particular that ``q_scale`` and
    ``kv_scale`` need a kernel-specific layout and that getting it wrong is silent.
    ``block_tables`` must be sized for the ``block_k`` in use. A reused ``out`` must be
    pre-filled with -inf, since the kernel only writes in-window cells.

    Choosing ``block_k``: use 64. What decides it is the longest **window**
    ``max(local_end - local_start)``, not the query count, and 64 wins on every shape
    measured except a single-batch 16k-token window. Do NOT key this off ``q_len``: that
    tracks the window length only when ``kv_len == q_len``, and with a compressed or
    pooled KV cache the two differ by the compression ratio -- exactly the case where a
    ``q_len``-based rule picks 256 and loses badly.
    """

    _require_gfx950("pa_mqa_logits_mxfp4")
    total_tokens = int(q_fp4.shape[0])
    if out is None:
        out = torch.full(
            (total_tokens, max_seq_len),
            float("-inf"),
            dtype=torch.float32,
            device=q_fp4.device,
        )
    pa_mqa_logits_mxfp4_fwd_prefill(
        q_fp4,
        q_scale,
        kv_cache,
        kv_scale,
        block_tables,
        weights,
        row_to_batch.to(torch.int32).contiguous(),
        local_starts.to(torch.int32).contiguous(),
        local_ends.to(torch.int32).contiguous(),
        out,
        total_tokens,
        float(weight_scale),
        int(block_k),
        int(kv_block_size),
        int(max_seq_len),
    )
    return out


def pa_mqa_logits_mxfp4_decode(
    q_fp4: torch.Tensor,
    q_scale: torch.Tensor,
    kv_cache: torch.Tensor,
    kv_scale: torch.Tensor,
    block_tables: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    max_seq_len: int,
    next_n_max: int,
    *,
    split_ctx_len: int,
    cu_seq_q: torch.Tensor | None = None,
    weight_scale: float = 1.0,
    block_k: int = BLOCK_K_1WAVE,
    kv_block_size: int = 64,
    cta_target: int = 1024,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Decode MQA logits (MTP), schedule-free and cudagraph-safe. One path for fixed-MTP
    (``cu_seq_q=None``) and varqlen (``cu_seq_q`` given). ``q`` / ``weights`` / ``out`` are
    PACKED ``[total_q, ...]``. 3D grid (batch, next_n_max, split_kv); the MTP tail-causal
    window is derived inline from ``cu_seq_q`` + ``context_lens``, so decode needs no window
    arrays and no window-build kernel.

    ``next_n_max`` (REQUIRED): MTP width = grid y-dim. Fixed MTP -> every batch has exactly
    ``next_n_max`` tokens (uniform ``cu_seq_q`` built here). Varqlen -> padded upper bound
    (>= max per-batch qlen; rows with ``n >= qlen`` idle). Always caller-supplied, so no
    host sync.

    ``split_ctx_len`` (REQUIRED): max KV length any row actually attends to this launch,
    bounding ``split_kv = ceil(split_ctx_len / block_k)``. Distinct from ``max_seq_len``
    (the output width): when a row processes only a slice of a longer sequence, sizing the
    split off ``max_seq_len`` would launch idle CTAs. Pass ``split_ctx_len == max_seq_len``
    when each row spans the full sequence.

    ``block_k``: 64 is the default and wins on all but the largest shapes.
    """

    _require_gfx950("pa_mqa_logits_mxfp4")
    block_k = int(block_k)
    total_q = int(q_fp4.shape[0])
    batch = int(context_lens.shape[0])
    next_n_max = int(next_n_max)
    if cu_seq_q is None:  # fixed-MTP: uniform per-batch qlen == next_n_max
        cu_seq_q = torch.arange(
            0,
            (batch + 1) * next_n_max,
            next_n_max,
            dtype=torch.int32,
            device=q_fp4.device,
        )
    else:
        cu_seq_q = cu_seq_q.to(torch.int32).contiguous()

    # Split the context across CTAs only when query rows alone under-fill the GPU.
    max_chunks = max(1, (int(split_ctx_len) + block_k - 1) // block_k)
    if total_q >= cta_target:
        split_kv = 1
    else:
        split_kv = min(max_chunks, (cta_target + total_q - 1) // total_q)

    if out is None:
        out = torch.full(
            (total_q, max_seq_len),
            float("-inf"),
            dtype=torch.float32,
            device=q_fp4.device,
        )
    pa_mqa_logits_mxfp4_fwd_decode(
        q_fp4,
        q_scale,
        kv_cache,
        kv_scale,
        block_tables,
        weights,
        cu_seq_q,
        context_lens.to(torch.int32).contiguous(),
        out,
        int(batch),
        int(next_n_max),
        int(split_kv),
        float(weight_scale),
        block_k,
        int(kv_block_size),
        int(max_seq_len),
    )
    return out


__all__ = [
    "compute_prefill_windows",
    "pa_mqa_logits_mxfp4_decode",
    "pa_mqa_logits_mxfp4_fwd_decode",
    "pa_mqa_logits_mxfp4_fwd_prefill",
    "pa_mqa_logits_mxfp4_prefill",
    "pa_mqa_logits_mxfp4_prefill_windows",
]
