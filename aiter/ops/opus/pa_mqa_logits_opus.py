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
    local_ends: torch.Tensor,
    out: torch.Tensor,
    batch: int,
    next_n_max: int,
    split_kv: int,
    weight_scale: float,
    block_k: int,
    kv_block_size: int,
    max_seq_len: int,
) -> None: ...


# Underscored and kept out of `aiter.ops.opus`, unlike the `_fwd_*` entries below: those are what
# ATOM imports from the top-level namespace, this one has no reason to be called directly. It
# takes the two nullable arrays as EMPTY tensors rather than None, and getting `row_to_batch`
# wrong that way is silent, so the wrapper owns it.
@compile_ops(MD_NAME_MXFP4, fc_name="pa_mqa_logits_mxfp4_build_sched", develop=True)
def _pa_mqa_logits_mxfp4_build_sched_raw(
    local_starts: torch.Tensor,
    local_ends: torch.Tensor,
    row_to_batch: torch.Tensor,
    cta_info: torch.Tensor,
    num_rows: int,
    num_ctas: int,
    block_k: int,
    cta_target: int,
) -> None: ...


@compile_ops(MD_NAME_MXFP4, develop=True)
def pa_mqa_logits_mxfp4_fwd_sched(
    q: torch.Tensor,
    q_scale: torch.Tensor,
    kv_cache: torch.Tensor,
    kv_scale: torch.Tensor,
    block_tables: torch.Tensor,
    weights: torch.Tensor,
    cta_info: torch.Tensor,
    out: torch.Tensor,
    num_ctas: int,
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
    """Build the per-row ``[local_start, local_end)`` window arrays BOTH launches consume,
    from ``cu_seq_q`` + ``context_lens``. Device-side, cudagraph-safe. Decode reads only
    ``local_ends``; call this ONCE per forward, not once per layer.

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


# ── the schedule ──────────────────────────────────────────────────────────────
# Mirrors of the C++ definitions. Only SCHED_CTA_TARGET is public, as the builder's default; the
# rest are geometry that `..._sched_slots` / `..._sched_buffer_ints` exist to own, and a caller
# who open-codes `(num_ctas + 96) * 8` is one header change away from under-allocating.
SCHED_CTA_TARGET = 1024
SCHED_CTA_CAP = SCHED_CTA_TARGET
SCHED_RECORD_INTS = 8
SCHED_SCRATCH_RECORDS = 96


def pa_mqa_logits_mxfp4_sched_slots(num_rows: int, cta_cap: int = SCHED_CTA_CAP) -> int:
    """CTA slots to launch, i.e. the ``num_ctas`` GRID for :func:`pa_mqa_logits_mxfp4_sched`.

    The floor is ``num_rows``: below it a row could get no CTA at all, and since every other row
    would still be right, the miss is silent.

    The cap of 1024 is wrong for an unusually heavy bucket -- many rows AND long windows, e.g.
    1024 rows of 51 KV tiles want 8192 slots where 1024 rows of 8 want exactly 1024. That is a
    function of total KV tiles, which the host cannot see, so such a caller should pass
    ``num_ctas`` to :func:`pa_mqa_logits_mxfp4_build_sched` explicitly.

    To size the BUFFER use :func:`pa_mqa_logits_mxfp4_sched_buffer_ints`, not this.
    """
    return max(int(num_rows), int(cta_cap))


def pa_mqa_logits_mxfp4_sched_buffer_ints(num_ctas: int) -> int:
    """int32 elements a ``cta_info`` buffer needs for ``num_ctas`` slots.

    Slots plus the builder's own scratch, which sits past them in the same buffer. Sizing it at
    ``num_ctas * SCHED_RECORD_INTS`` instead is under-allocation; the launcher raises on it.
    """
    return (int(num_ctas) + SCHED_SCRATCH_RECORDS) * SCHED_RECORD_INTS


def pa_mqa_logits_mxfp4_build_sched(
    local_ends: torch.Tensor,
    num_rows: int,
    *,
    local_starts: torch.Tensor | None = None,
    row_to_batch: torch.Tensor | None = None,
    block_k: int = BLOCK_K_1WAVE,
    num_ctas: int | None = None,
    cta_target: int = SCHED_CTA_TARGET,
    cta_info: torch.Tensor | None = None,
) -> tuple[torch.Tensor, int]:
    """Build the per-row schedule, for either entry point. Device-side, cudagraph-safe, no sync.

    Call this ONCE PER FORWARD, not once per layer: it depends only on ``local_ends``, which is a
    per-forward quantity, while the kernel runs per CSA layer. A caller whose scorer sits inside
    a CUDAGraph capture builds it in its metadata builder and hands the same buffer to the
    capture.

    ``local_starts`` is the per-row window start. Leave it ``None`` for decode, whose rows always
    start at 0; prefill's usually do not, and it is the only field that distinguishes the two.

    ``row_to_batch`` is the ``block_tables`` row of each query row. Leave it ``None`` when a query
    row IS its own batch item and ``block_tables`` is per-token -- the ``next_n=1`` convention.
    Passing a per-BATCH map while the launch gets per-TOKEN ``block_tables`` reads the wrong
    pages and produces plausible wrong numbers.

    Returns ``(cta_info, num_ctas)``. Pass both to the launch. Reuse the buffer across forwards:
    the builder writes every slot, including the surplus ones, so nothing leaks between them.
    A caller supplying its own ``cta_info`` must size it with :func:`pa_mqa_logits_mxfp4_sched_buffer_ints`.
    """
    n = int(num_rows)
    slots = pa_mqa_logits_mxfp4_sched_slots(n) if num_ctas is None else int(num_ctas)
    if cta_info is None:
        cta_info = torch.empty(
            (slots + SCHED_SCRATCH_RECORDS, SCHED_RECORD_INTS),
            dtype=torch.int32,
            device=local_ends.device,
        )
    empty = torch.empty(0, dtype=torch.int32, device=local_ends.device)
    _pa_mqa_logits_mxfp4_build_sched_raw(
        local_starts if local_starts is not None else empty,
        local_ends.to(torch.int32).contiguous(),
        row_to_batch if row_to_batch is not None else empty,
        cta_info,
        n,
        slots,
        int(block_k),
        int(cta_target),
    )
    return cta_info, slots


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
    local_ends: torch.Tensor,
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
    PACKED ``[total_q, ...]``. 3D grid (batch, next_n_max, split_kv).

    ``local_ends`` is ``[total_q]`` int32 indexed by the PACKED row ``cu_seq_q[b] + n``,
    which is scored over ``[0, local_ends[row])``. It is read, not derived, so a compressed
    KV cache -- row ``n`` sees ``floor((pos + 1) / ratio)``, which is not
    ``ctx - (next_n - 1 - n)`` -- can express its window.
    :func:`compute_prefill_windows` builds the tail-causal case. Build it, and
    ``cu_seq_q``, ONCE per forward: both are per-forward quantities and the kernel runs
    per layer. Leaving ``cu_seq_q`` as ``None`` costs a ``torch.arange`` on every call.

    Rows with ``local_ends <= 0`` cost one workgroup that exits immediately, so padding
    the row count to a static shape is cheap.

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
    next_n_max = int(next_n_max)
    if next_n_max <= 0:
        raise ValueError(
            f"next_n_max is the grid y-dim and must be >= 1, got {next_n_max}"
        )
    if cu_seq_q is None:  # fixed-MTP: uniform per-batch qlen == next_n_max
        if total_q % next_n_max:
            raise ValueError(
                f"fixed-MTP decode wants total_q ({total_q}) divisible by next_n_max "
                f"({next_n_max}); pass cu_seq_q for a ragged batch."
            )
        batch = total_q // next_n_max
        cu_seq_q = torch.arange(
            0,
            (batch + 1) * next_n_max,
            next_n_max,
            dtype=torch.int32,
            device=q_fp4.device,
        )
    else:
        cu_seq_q = cu_seq_q.to(torch.int32).contiguous()
        batch = int(cu_seq_q.shape[0]) - 1

    # Split the context across CTAs only when query rows alone under-fill the GPU.
    # An empty batch takes the no-split branch: there is nothing to launch and the C++ side
    # returns before the grid is built, so the row count must not reach the division.
    max_chunks = max(1, (int(split_ctx_len) + block_k - 1) // block_k)
    if total_q <= 0 or total_q >= cta_target:
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
        local_ends.to(torch.int32).contiguous(),
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


def pa_mqa_logits_mxfp4_sched(
    q_fp4: torch.Tensor,
    q_scale: torch.Tensor,
    kv_cache: torch.Tensor,
    kv_scale: torch.Tensor,
    block_tables: torch.Tensor,
    weights: torch.Tensor,
    cta_info: torch.Tensor,
    num_ctas: int,
    max_seq_len: int,
    *,
    weight_scale: float = 1.0,
    block_k: int = BLOCK_K_1WAVE,
    kv_block_size: int = 64,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """MQA logits over the schedule :func:`pa_mqa_logits_mxfp4_build_sched` produced -- prefill or decode, the
    table says which. **Prefer this to :func:`pa_mqa_logits_mxfp4_decode` and to
    :func:`pa_mqa_logits_mxfp4_prefill` on any batch whose rows differ in window length**; the
    arithmetic is identical, what differs is which CTA covers which KV tiles.

    ``cta_info`` / ``num_ctas`` come from :func:`pa_mqa_logits_mxfp4_build_sched`, once per forward. There is no
    ``local_ends`` argument and that is deliberate: each record carries its own row's window, so
    the array the schedule was built from cannot disagree with the table. A reused ``out`` must
    be pre-filled with -inf, since the kernel only writes in-window cells.
    """
    _require_gfx950("pa_mqa_logits_mxfp4_sched")
    total_q = int(q_fp4.shape[0])
    if out is None:
        out = torch.full(
            (total_q, max_seq_len),
            float("-inf"),
            dtype=torch.float32,
            device=q_fp4.device,
        )
    pa_mqa_logits_mxfp4_fwd_sched(
        q_fp4,
        q_scale,
        kv_cache,
        kv_scale,
        block_tables,
        weights,
        cta_info,
        out,
        int(num_ctas),
        float(weight_scale),
        int(block_k),
        int(kv_block_size),
        int(max_seq_len),
    )
    return out


__all__ = [
    "SCHED_CTA_TARGET",
    "compute_prefill_windows",
    "pa_mqa_logits_mxfp4_build_sched",
    "pa_mqa_logits_mxfp4_decode",
    "pa_mqa_logits_mxfp4_fwd_decode",
    "pa_mqa_logits_mxfp4_fwd_prefill",
    "pa_mqa_logits_mxfp4_fwd_sched",
    "pa_mqa_logits_mxfp4_prefill",
    "pa_mqa_logits_mxfp4_prefill_windows",
    "pa_mqa_logits_mxfp4_sched",
    "pa_mqa_logits_mxfp4_sched_buffer_ints",
    "pa_mqa_logits_mxfp4_sched_slots",
]
