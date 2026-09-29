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

Prefill and decode run through ONE launch, over a per-row schedule table built on device:
:func:`pa_mqa_logits_mxfp4_build_sched` once per forward, :func:`pa_mqa_logits_mxfp4_sched`
per layer. The table says which entry point it is, by whether its rows carry a non-zero
window start. Both are cudagraph-safe.

The per-row ``[local_start, local_end)`` windows are the CALLER's, like the quantization.
Plain ``[total_q]`` int32, no constraint but ``0 <= local_start <= local_end``.

``block_k`` picks between two compiled variants that produce identical results: 256 -> 4
waves/CTA, 64 -> 1 wave/CTA, a pure performance knob. It must be the SAME value in the build
and the launch -- the table's chunk indices are in ``block_k`` units and nothing cross-checks
them.
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


# The raw entry. Prefer the wrapper: this is worth +0.54 us of host dispatch out of 16, and a
# caller host-bound at these shapes wants a CUDA graph, not a shorter python path.
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


# Mirrors of the C++ definitions. Only SCHED_CTA_TARGET is public; the rest are geometry that
# `..._sched_slots` / `..._sched_buffer_ints` own, and open-coding `(num_ctas + 96) * 8` is one
# header change away from under-allocating.
SCHED_CTA_TARGET = 1024
SCHED_CTA_CAP = SCHED_CTA_TARGET
SCHED_RECORD_INTS = 8
SCHED_SCRATCH_RECORDS = 96


def _require_gfx950(name):
    gfx = get_gfx_runtime()
    if gfx != "gfx950":
        raise RuntimeError(f"{name} requires gfx950, got {gfx}")


def pa_mqa_logits_mxfp4_sched_slots(num_rows: int, cta_cap: int = SCHED_CTA_CAP) -> int:
    """CTA slots to launch, i.e. the ``num_ctas`` GRID. For the BUFFER use
    :func:`pa_mqa_logits_mxfp4_sched_buffer_ints`.

    The floor is ``num_rows``: below it a row could get no CTA at all, and since every other row
    would still be right, the miss is silent. The cap of 1024 is low for a bucket with many rows
    AND long windows -- that depends on total KV tiles, which the host cannot see, so such a
    caller passes ``num_ctas`` to the builder explicitly.
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

    Call this ONCE PER FORWARD, not once per layer: it depends only on ``local_ends``, a
    per-forward quantity, while the kernel runs per CSA layer. A caller inside a CUDAGraph
    capture builds it in its metadata builder and hands the same buffer to the capture.

    ``local_starts`` is the per-row window start. Leave it ``None`` for decode, whose rows always
    start at 0; prefill's usually do not, and it is the only field that distinguishes the two.

    ``row_to_batch`` is the ``block_tables`` row of each query row. Leave it ``None`` when a query
    row IS its own batch item and ``block_tables`` is per-token -- the ``next_n=1`` convention.
    Passing a per-BATCH map while the launch gets per-TOKEN ``block_tables`` reads the wrong
    pages and produces plausible wrong numbers.

    Returns ``(cta_info, num_ctas)``; pass both to the launch. Safe to reuse the buffer across
    forwards -- every slot is written, surplus ones included. A caller supplying its own must
    size it with :func:`pa_mqa_logits_mxfp4_sched_buffer_ints`.
    """
    _require_gfx950("pa_mqa_logits_mxfp4_build_sched")
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
    """MQA logits over the schedule :func:`pa_mqa_logits_mxfp4_build_sched` produced -- prefill
    or decode, the table says which. The only launch entry point.

    ``cta_info`` / ``num_ctas`` come from the builder, once per forward. There is no
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
    "pa_mqa_logits_mxfp4_build_sched",
    "pa_mqa_logits_mxfp4_fwd_sched",
    "pa_mqa_logits_mxfp4_sched",
    "pa_mqa_logits_mxfp4_sched_buffer_ints",
    "pa_mqa_logits_mxfp4_sched_slots",
]
