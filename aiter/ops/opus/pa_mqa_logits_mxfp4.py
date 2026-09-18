# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""MXFP4 paged MQA logits for DeepSeek-style sparse attention (OPUS kernels).

Per query row ``r`` over a window ``[s, e)``:
``out[r, s:e] = sum_H( relu(Q[r] . K^T) * weight[r] ) * weight_scale``

Prefill and decode, through ONE launch over a schedule table built on device, on gfx950 and
gfx1250 alike. The plan depends only on per-FORWARD data while the kernel runs per CSA LAYER,
and DeepSeek-V4 has 61 of them, so a caller builds once and launches 61 times::

    plan = pa_mqa_logits_mxfp4_plan(cu_seq_q, local_ends)         # once per FORWARD
    out  = pa_mqa_logits_mxfp4(q, q_scale, kv_cache, kv_scale,
                               block_tables, weights, plan, max_seq_len)  # per LAYER

Both are device-side and read nothing back, so the path is cudagraph-safe.

Quantization and layout are the CALLER's; this module never touches the data. THREE OF THE
FIVE INPUTS HAVE A DIFFERENT LAYOUT PER TARGET, which the dispatch cannot hide because the
arrays are written by whoever quantizes:

=============  =======================================  =================================
tensor         gfx950                                   gfx1250
=============  =======================================  =================================
``q``          ``[total_q, H, D/2]`` uint8              same
``weights``    ``[total_q, H]`` bf16                    same
``q_scale``    ``[total_q, 2, 32, 4]`` MFMA-permuted    ``[total_q, H, 4]`` natural
``kv_scale``   ``[num_blocks, 2, 32, 4]`` permuted      ``[num_blocks, PAGE, 4]`` natural
``kv_cache``   ``[num_blocks, 4, PAGE, 16]``            ``[num_blocks, PAGE, D/2]`` natural
=============  =======================================  =================================

On gfx1250 every scale is the plain E8M0 byte for 32-element K block ``b`` of its row and
every packed row is 64 contiguous bytes, low nibble first. ``block_tables`` differs too: both
targets round a window up to a whole KV tile before indexing it, and that width is 64 or 256 on
gfx950 against a fixed 128 here. The C++ header states the resulting bound.

**WHY THIS MODULE VALIDATES THE LAYOUT.** Every fp4 scale layout has the same BYTE COUNT --
``q_scale`` is ``total_q * 256`` either way -- so both C++ launchers check ``numel`` and accept
the other target's arrays, returning plausible wrong logits. The SHAPES differ (permuted 4-D,
natural 3-D) and this is the only place one is seen before the pointer is taken. It cannot
catch an array reshaped to the right ndim, and nothing can: only a dequantized reference on
RANDOM data sees a wrong permutation, since uniform data agrees under any permutation of K.

TWO CONDITIONS the gfx1250 kernel cannot check and does not survive -- a CTA whose waves
disagree about the trip count DEADLOCKS on the phase barrier rather than returning a wrong
answer:

1. the window rule is NON-DECREASING in the row index within a tile, which every causal and
   CSA-compressed rule is, and which is what makes the tile's union the FIRST row's start and
   the LAST row's end -- two loads the builder takes on faith instead of a reduction;
2. the store is bounded by the WINDOW, so a ``local_ends`` entry past ``out.shape[1]`` writes
   past the row.

"A tile is contiguous rows of one batch" used to be a third and is now guaranteed by the tile
cut :func:`pa_mqa_logits_mxfp4_plan` runs.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ...jit.core import compile_ops
from ._arch import GFX950, GFX1250, _device_arch

DEFAULT_KV_BLOCK_SIZE = 64

# Query rows per CTA == waves per CTA on gfx1250; the group size the tile cut and the kernel
# agree on. Internal: the plan builder does the cut and `_gfx1250_max_tiles_for` the sizing, so
# a caller never needs it. It mirrors the traits header, and nothing ties the two together.
Q_PER_BLOCK = 4

# The gfx1250 KV tile in tokens, and internal: gfx950 takes it as `block_k` instead, so a
# caller reading this constant would be wrong there.
BLOCK_K = 128

# 4-D is the gfx950 MFMA permutation of the scales and its 4-chunk kv_cache; 3-D is gfx1250's
# all-natural form. The two differ in no other observable way -- see the module docstring.
_PERMUTED_NDIM = 4
_NATURAL_NDIM = 3


# ══ the gfx1250 implementation ═══════════════════════════════════════════════════════════════
_MD_NAME_GFX1250 = "module_pa_mqa_logits_mxfp4_gfx1250_opus"

# Mirrors of the C++ constants. `_gfx1250_compute_schedule` is the ONLY place the `cta_info`
# size is computed, and it stays that way on purpose: the builder's own scratch sits past the
# slots in the same buffer, so open-coding `num_ctas * 8` anywhere is under-allocation, which
# the launcher raises on. `_SCHED_CTA_RESIDENT` is the part's resident CTA count (256 CUs x
# occupancy 1). Not a knob: the split aims at it, and the A/B that turns the split off lives in
# the opus-ops harness rather than on this op's surface.
_SCHED_CTA_RESIDENT = 256
_SCHED_RECORD_INTS = 8
_SCHED_SCRATCH_RECORDS = 96


# ── JIT stubs: signatures must match PA_MQA_LOGITS_MXFP4_GFX1250_PYBIND exactly ──────────────
# `fc_name` keeps the pybind name while the python name says these are private. They take the
# raw ABI (empty-tensor sentinels, not None) and can carry NO arch check: `compile_ops` replaces
# the body with its own wrapper and never calls it, so a guard here would be dead code.
@compile_ops(
    _MD_NAME_GFX1250,
    fc_name="pa_mqa_logits_mxfp4_gfx1250_fwd_sched",
    develop=True,
)
def _gfx1250_fwd_sched_raw(
    q: torch.Tensor,
    q_scale: torch.Tensor,
    kv_cache: torch.Tensor,
    kv_scale: torch.Tensor,
    block_tables: torch.Tensor,
    weights: torch.Tensor,
    local_starts: torch.Tensor,
    local_ends: torch.Tensor,
    cta_info: torch.Tensor,
    out: torch.Tensor,
    num_rows: int,
    num_ctas: int,
    weight_scale: float,
    kv_block_size: int,
    max_seq_len: int,
) -> None: ...


@compile_ops(
    _MD_NAME_GFX1250,
    fc_name="pa_mqa_logits_mxfp4_gfx1250_build_tiles",
    develop=True,
)
def _gfx1250_build_tiles_raw(
    cu_seq_q: torch.Tensor,
    cu_tiles: torch.Tensor,
    total_q: int,
    max_tiles: int,
) -> None: ...


@compile_ops(
    _MD_NAME_GFX1250,
    fc_name="pa_mqa_logits_mxfp4_gfx1250_build_sched",
    develop=True,
)
def _gfx1250_build_sched_raw(
    cu_tiles: torch.Tensor,
    local_starts: torch.Tensor,
    local_ends: torch.Tensor,
    row_to_batch: torch.Tensor,
    cta_info: torch.Tensor,
    num_tiles: int,
    num_ctas: int,
    cta_resident: int,
) -> None: ...


def _gfx1250_max_tiles_for(total_q: int, batch: int) -> int:
    """Tiles the cut can produce, from the static shapes alone -- which is what keeps the launch
    cudagraph-safe.

    The real count is ``sum_b ceil(qlen_b / Q_PER_BLOCK)`` and depends on device data. This sums
    the per-batch roundings before the divide, so it is never short and is exact whenever they
    tile. The slack is at most ``batch - 1`` tiles, each of which gets an empty record.
    """
    return (int(total_q) + int(batch) * (Q_PER_BLOCK - 1)) // Q_PER_BLOCK


def _gfx1250_sched_slots(num_tiles: int) -> int:
    """The ``num_ctas`` GRID, NOT the buffer's record count -- that is this plus the builder's
    scratch and lives in the allocation below.

    The floor of one resident round lets a handful of long tiles spread over the GPU; the
    rounding leaves the split room above it, since a split needs MORE slots than tiles. Tight
    rather than generous: a surplus slot's CTA still reads a 32-byte record nobody else touches.
    """
    n = max(int(num_tiles), _SCHED_CTA_RESIDENT)
    return -(-n // _SCHED_CTA_RESIDENT) * _SCHED_CTA_RESIDENT


def _gfx1250_cta_info(device, num_ctas):
    # Slots PLUS the builder's own scratch, which sits past them in the same buffer.
    return torch.empty(
        (int(num_ctas) + _SCHED_SCRATCH_RECORDS, _SCHED_RECORD_INTS),
        dtype=torch.int32,
        device=device,
    )


def _gfx1250_cu_tiles(device, num_tiles):
    # Tile `t` covers rows [cu_tiles[t], cu_tiles[t + 1]), so the terminator is not optional.
    return torch.empty(int(num_tiles) + 1, dtype=torch.int32, device=device)


def _gfx1250_compute_tiles(cu_seq_q, total_q, out=None):
    """Cut the query rows into tiles of at most ``Q_PER_BLOCK``, device-side.

    ``num_tiles`` is an UPPER BOUND, so it stays a host int and no device read is needed to
    launch; tiles past the real count are written empty. This cut is what guarantees "a tile is
    contiguous rows of one batch", which the kernel cannot check and does not survive.
    """
    cu = cu_seq_q.to(torch.int32).contiguous()
    batch = int(cu.shape[0]) - 1
    num_tiles = _gfx1250_max_tiles_for(total_q, batch)
    cu_tiles = _gfx1250_cu_tiles(cu.device, num_tiles) if out is None else out
    _gfx1250_build_tiles_raw(cu, cu_tiles, int(total_q), int(num_tiles))
    return cu_tiles, num_tiles


def _gfx1250_compute_schedule(
    cu_tiles,
    local_ends,
    num_tiles,
    *,
    local_starts=None,
    row_to_batch=None,
    num_ctas=None,
    cta_info=None,
):
    """Build the per-tile schedule. Device-side, no sync.

    Safe to reuse the buffer across forwards -- every slot is written, surplus ones included.
    """
    n = int(num_tiles)
    slots = _gfx1250_sched_slots(n) if num_ctas is None else int(num_ctas)
    if cta_info is None:
        cta_info = _gfx1250_cta_info(local_ends.device, slots)
    empty = torch.empty(0, dtype=torch.int32, device=local_ends.device)
    _gfx1250_build_sched_raw(
        cu_tiles,
        local_starts if local_starts is not None else empty,
        local_ends.to(torch.int32).contiguous(),
        row_to_batch if row_to_batch is not None else empty,
        cta_info,
        n,
        slots,
        _SCHED_CTA_RESIDENT,
    )
    return cta_info, slots


def _gfx1250_launch(
    q_fp4,
    q_scale,
    kv_cache,
    kv_scale,
    block_tables,
    weights,
    local_ends,
    cta_info,
    num_ctas,
    max_seq_len,
    *,
    local_starts=None,
    weight_scale=1.0,
    kv_block_size=DEFAULT_KV_BLOCK_SIZE,
    out=None,
):
    """``local_ends`` is a launch argument even though the schedule was built from it, because
    the kernel reads it PER ROW for the store mask while the table carries only each tile's
    union. It must be the SAME array; a shorter one is rejected, a differently-valued one is
    not."""
    total_rows = int(q_fp4.shape[0])
    if out is None:
        out = torch.full(
            (total_rows, max_seq_len),
            float("-inf"),
            dtype=torch.float32,
            device=q_fp4.device,
        )
    empty = torch.empty(0, dtype=torch.int32, device=q_fp4.device)
    _gfx1250_fwd_sched_raw(
        q_fp4,
        q_scale,
        kv_cache,
        kv_scale,
        block_tables,
        weights,
        local_starts if local_starts is not None else empty,
        local_ends.to(torch.int32).contiguous(),
        cta_info,
        out,
        total_rows,
        int(num_ctas),
        float(weight_scale),
        int(kv_block_size),
        int(max_seq_len),
    )
    return out


# ══ the gfx950 implementation ════════════════════════════════════════════════════════════════
def _gfx950_mod():
    # Lazy and separately reported: the gfx950 op lands with ROCm/aiter#5332, which merges after
    # this one, so on this tree neither its JIT module config nor its C++ sources exist and the
    # import is what fails. The arms below are written against that module's API; when it lands,
    # its implementation folds in above and `pa_mqa_logits_opus.py` goes away.
    try:
        from . import pa_mqa_logits_opus as mod
    except ImportError as e:
        raise RuntimeError(
            "the gfx950 MXFP4 MQA-logits op is not in this aiter tree (it lands with "
            "ROCm/aiter#5332); this dispatcher's gfx950 arm needs "
            "aiter.ops.opus.pa_mqa_logits_opus"
        ) from e
    return mod


# ══ the dispatcher ═══════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True, eq=False)
class MqaLogitsPlan:
    """One forward's schedule, plus whatever else that target's launch needs to go with it.

    Reused across the forward's layers; rebuilding it per layer turns a per-forward cost into a
    per-layer one, 61x on DeepSeek-V4. A caller that reuses buffers -- a CUDAGraph capture does
    -- hands ``cta_info`` and ``cu_tiles`` back to the plan builder.

    ``eq=False`` because the dataclass holds tensors: a generated ``__eq__`` compares them
    elementwise and returns a tensor, so ``plan == other`` would raise only sometimes.
    """

    arch: str
    cta_info: torch.Tensor
    # The launch GRID. Read it at capture and hand it back to the builder every forward; a
    # graph replays the grid it captured and nothing re-checks it against a later table.
    num_ctas: int
    # gfx1250 only. The windows travel WITH the plan because the kernel reads them per row for
    # the store mask while the table carries only each tile's union, and because they must be
    # the same arrays the schedule was built from. gfx950's records carry each row's window, so
    # its launch never sees these.
    #
    # `num_tiles` is the cut's UPPER BOUND, not its exact count, and it is kept because
    # `num_ctas > num_tiles` is how a reader tells that the schedule split a tile across CTAs.
    # None on gfx950, where a CTA is one query row and there is no cut.
    cu_tiles: torch.Tensor | None = None
    num_tiles: int | None = None
    local_starts: torch.Tensor | None = None
    local_ends: torch.Tensor | None = None
    # gfx950 only: the table's chunk indices are in `block_k` units and nothing cross-checks
    # them, so the launch must be given the value the builder used.
    block_k: int | None = None


def _arch_of(t: torch.Tensor) -> str:
    """The arch of the tensor's OWN device rather than the process-wide runtime, so a caller
    working on a non-current GPU is dispatched by what it actually holds."""
    arch = _device_arch(t.device)
    if arch not in (GFX950, GFX1250):
        raise RuntimeError(
            f"the MXFP4 MQA-logits op supports {GFX950} and {GFX1250}, got {arch}"
        )
    return arch


def _check_layout(arch: str, q_scale, kv_scale, kv_cache) -> None:
    """Reject the other target's scale and cache layouts, which the C++ size checks accept."""
    if arch == GFX950:
        want, other_ndim, other = _PERMUTED_NDIM, _NATURAL_NDIM, GFX1250
    else:
        want, other_ndim, other = _NATURAL_NDIM, _PERMUTED_NDIM, GFX950
    for name, t in (
        ("q_scale", q_scale),
        ("kv_scale", kv_scale),
        ("kv_cache", kv_cache),
    ):
        if t.dim() == want:
            continue
        why = (
            f"; {t.dim()}-D is the {other} layout, which has the SAME byte count -- the C++ "
            "launcher checks numel and would accept it, then return wrong logits"
            if t.dim() == other_ndim
            else ""
        )
        raise ValueError(f"{arch}: {name} must be {want}-D, got {tuple(t.shape)}{why}")


def pa_mqa_logits_mxfp4_plan_buffers(
    device: torch.device | str | int,
    total_q: int,
    batch: int,
    *,
    num_ctas: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, int]:
    """Allocate a plan's buffers and settle its grid from the STATIC shapes alone.

    Returns ``(cta_info, cu_tiles, num_ctas)`` -- ``cu_tiles`` is ``None`` on gfx950, which has
    no tile cut -- to be handed straight back to :func:`pa_mqa_logits_mxfp4_plan` every
    forward. Nothing here needs a formula on the caller's side, which is the point: the
    ``cta_info`` shape carries the builder's own scratch past the slots and the grid is a
    two-step rounding, so open-coding either is one header change away from under-allocating.

    For a caller that must allocate BEFORE it has any windows -- a fixed buffer pool set up at
    startup, which is what a CUDAGraph metadata builder holds. A caller that can allocate
    lazily does not need this: build one plan and reuse its three outputs.

    Size them at the LARGEST shape they will be pinned to. A later forward with fewer rows
    writes fewer tiles into the same buffers and keeps the pinned grid, which is consistent;
    the reverse is what the launcher raises on.
    """
    arch = _device_arch(device)
    if arch == GFX1250:
        num_tiles = _gfx1250_max_tiles_for(total_q, batch)
        slots = _gfx1250_sched_slots(num_tiles) if num_ctas is None else int(num_ctas)
        return (
            _gfx1250_cta_info(device, slots),
            _gfx1250_cu_tiles(device, num_tiles),
            slots,
        )
    if arch == GFX950:
        mod = _gfx950_mod()
        slots = (
            mod.pa_mqa_logits_mxfp4_sched_slots(int(total_q))
            if num_ctas is None
            else int(num_ctas)
        )
        ints = mod.pa_mqa_logits_mxfp4_sched_buffer_ints(slots)
        return (
            torch.empty(
                (ints // mod.SCHED_RECORD_INTS, mod.SCHED_RECORD_INTS),
                dtype=torch.int32,
                device=device,
            ),
            None,
            slots,
        )
    raise RuntimeError(
        f"the MXFP4 MQA-logits op supports {GFX950} and {GFX1250}, got {arch}"
    )


def pa_mqa_logits_mxfp4_plan(
    cu_seq_q: torch.Tensor,
    local_ends: torch.Tensor,
    *,
    total_q: int | None = None,
    local_starts: torch.Tensor | None = None,
    row_to_batch: torch.Tensor | None = None,
    num_ctas: int | None = None,
    block_k: int | None = None,
    cta_info: torch.Tensor | None = None,
    cu_tiles: torch.Tensor | None = None,
) -> MqaLogitsPlan:
    """Build one forward's schedule on device. Call ONCE PER FORWARD, not once per layer.

    ``cu_seq_q`` is the batch's ``[batch + 1]`` query-row prefix sum. **gfx950 ignores it** and
    it is still required, because gfx1250 cuts its tiles from it and that cut is what
    guarantees "a tile is contiguous rows of one batch" -- see the module docstring for why
    breaking it deadlocks rather than answering wrong.

    ``local_ends`` is the per-row window end, ``[total_q]`` int32; ``total_q`` defaults to its
    element count. The windows are the CALLER's, so any rule works.

    ``local_starts`` may be ``None`` when every row starts at 0, which is what both ATOM paths
    do. ``row_to_batch`` may be ``None`` when ``block_tables`` is indexed by query ROW rather
    than by sequence; passing a per-sequence map against a per-token table, or the reverse,
    reads the wrong pages and produces plausible wrong numbers.

    ``block_k`` is gfx950-only and raises on gfx1250 rather than being ignored, because a
    silently-dropped tile width is a knob that looks like it took effect.

    **UNDER A CUDAGRAPH, `cta_info`, `cu_tiles` AND `num_ctas` ARE ALL REQUIRED**, and both
    failures are silent. A replay reads the POINTER and the GRID it captured while this builder
    runs outside it, so: omit the buffers and each forward writes a fresh allocation the graph
    never reads -- it reads whatever the caching allocator has since put at the captured
    address, and a garbage record's row index is not bounded anywhere; leave ``num_ctas``
    derived and a forward whose shape rounds to a different count builds a table the grid does
    not match. Read ``plan.num_ctas`` at capture and pass it back every forward. That also
    turns the remaining risk into a host-side raise, since the ``num_ctas >= num_tiles`` check
    runs eagerly here while the graph's own grid is past checking.

    Every slot is written, surplus ones included, so buffer reuse needs no clearing.
    """
    arch = _arch_of(local_ends)
    n = int(local_ends.numel()) if total_q is None else int(total_q)

    if arch == GFX1250:
        if block_k is not None:
            raise ValueError(
                "gfx1250 compiles one KV tile width; block_k is gfx950-only"
            )
        tiles, num_tiles = _gfx1250_compute_tiles(cu_seq_q, n, out=cu_tiles)
        info, slots = _gfx1250_compute_schedule(
            tiles,
            local_ends,
            num_tiles,
            local_starts=local_starts,
            row_to_batch=row_to_batch,
            num_ctas=num_ctas,
            cta_info=cta_info,
        )
        return MqaLogitsPlan(
            arch=arch,
            cta_info=info,
            num_ctas=slots,
            cu_tiles=tiles,
            num_tiles=num_tiles,
            local_starts=local_starts,
            local_ends=local_ends,
        )

    mod = _gfx950_mod()
    bk = int(mod.BLOCK_K_1WAVE if block_k is None else block_k)
    info, slots = mod.pa_mqa_logits_mxfp4_build_sched(
        local_ends,
        n,
        local_starts=local_starts,
        row_to_batch=row_to_batch,
        block_k=bk,
        num_ctas=num_ctas,
        cta_target=mod.SCHED_CTA_TARGET,
        cta_info=cta_info,
    )
    return MqaLogitsPlan(arch=arch, cta_info=info, num_ctas=slots, block_k=bk)


def pa_mqa_logits_mxfp4(
    q_fp4: torch.Tensor,
    q_scale: torch.Tensor,
    kv_cache: torch.Tensor,
    kv_scale: torch.Tensor,
    block_tables: torch.Tensor,
    weights: torch.Tensor,
    plan: MqaLogitsPlan,
    max_seq_len: int,
    *,
    weight_scale: float = 1.0,
    kv_block_size: int = DEFAULT_KV_BLOCK_SIZE,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Paged MQA logits over ``plan`` -- prefill or decode, the schedule says which.

    Returns ``out``, allocating a ``[total_q, max_seq_len]`` fp32 tensor of ``-inf`` when not
    given one; a REUSED ``out`` must be pre-filled with ``-inf``, since only in-window cells
    are written. ``block_tables`` must be sized for the KV TILE, not for ``kv_block_size``; the
    C++ header states the bound.
    """
    arch = _arch_of(q_fp4)
    if arch != plan.arch:
        raise ValueError(
            f"the plan was built for {plan.arch} and this launch is on {arch}; a plan is per "
            "forward AND per device"
        )
    _check_layout(arch, q_scale, kv_scale, kv_cache)

    if arch == GFX1250:
        return _gfx1250_launch(
            q_fp4,
            q_scale,
            kv_cache,
            kv_scale,
            block_tables,
            weights,
            plan.local_ends,
            plan.cta_info,
            plan.num_ctas,
            int(max_seq_len),
            local_starts=plan.local_starts,
            weight_scale=weight_scale,
            kv_block_size=kv_block_size,
            out=out,
        )

    return _gfx950_mod().pa_mqa_logits_mxfp4_sched(
        q_fp4,
        q_scale,
        kv_cache,
        kv_scale,
        block_tables,
        weights,
        plan.cta_info,
        plan.num_ctas,
        int(max_seq_len),
        weight_scale=weight_scale,
        block_k=plan.block_k,
        kv_block_size=kv_block_size,
        out=out,
    )


__all__ = [
    "MqaLogitsPlan",
    "pa_mqa_logits_mxfp4",
    "pa_mqa_logits_mxfp4_plan",
    "pa_mqa_logits_mxfp4_plan_buffers",
]
