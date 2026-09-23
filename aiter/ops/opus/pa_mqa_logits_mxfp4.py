# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""MXFP4 paged MQA logits for DeepSeek-style sparse attention (OPUS kernels).

Per query row ``r`` over a window ``[s, e)``:
``out[r, s:e] = sum_H( relu(Q[r] . K^T) * weight[r] ) * weight_scale``

Prefill and decode go through ONE launch over a schedule table built on device. **gfx1250 is
the only target built in this tree** -- the entry points below dispatch on arch, and the gfx950
arm raises until ROCm/aiter#5332 lands the module it calls.

USAGE. The plan depends only on per-FORWARD data while the kernel runs per CSA LAYER, and
DeepSeek-V4 has 61 of them, so a caller builds once and launches 61 times::

    # once at startup: how wide the page table has to be
    width = pa_mqa_logits_mxfp4_block_table_width(max_model_len)

    # once per buffer pool; `variant=` names the kernel instance, default is the general one
    buf  = pa_mqa_logits_mxfp4_plan_buffers(dev, total_q, batch, variant="qlen1_kv64")

    plan = pa_mqa_logits_mxfp4_plan(cu_seq_q, local_ends, buffers=buf)    # once per FORWARD

    out  = pa_mqa_logits_mxfp4(q, q_scale, kv_cache, kv_scale,            # once per LAYER
                               block_tables, weights, plan, max_seq_len, out=out)

Every step is device-side and reads nothing back, so the path is cudagraph-safe -- but under a
graph the buffers must be caller-held and reused, which :func:`pa_mqa_logits_mxfp4_plan`
spells out.

KERNEL INSTANCES. :func:`pa_mqa_logits_mxfp4_variants` lists what this build compiled. They
differ only in how many query rows one CTA covers, they take identical inputs, and one is
chosen per BUFFER POOL rather than per launch. **Nothing infers the choice from the shape**:
the default suits prefill and MTP > 1, while MTP = 1 decode should ask for the one-row
instance, which is worth up to 2.5x there and more as the batch grows.

LAYOUTS. Quantization and layout are the CALLER's; this module never touches the data. Three
of the five inputs have a different layout per target, which the dispatch cannot hide because
the arrays are written by whoever quantizes:

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
gfx950 against a fixed 64 here. The C++ header states the resulting bound.

**Handing one target's arrays to the other is caught HERE and nowhere else.** Every fp4 scale
layout has the same BYTE COUNT -- ``q_scale`` is ``total_q * 256`` either way -- so the C++
launchers check ``numel`` and would accept them, then return plausible wrong logits. The
ndim differs (permuted 4-D, natural 3-D) and this is the only place a shape is seen before the
pointer is taken. It cannot catch an array reshaped to the right ndim, and nothing can.

TWO CONDITIONS the gfx1250 kernel cannot check and does not survive -- a CTA whose waves
disagree about the trip count DEADLOCKS on the phase barrier rather than returning a wrong
answer:

1. the window rule is NON-DECREASING in the row index within a tile, which every causal and
   CSA-compressed rule is, and which is what makes the tile's union the FIRST row's start and
   the LAST row's end -- two loads the builder takes on faith instead of a reduction;
2. the store is bounded by the WINDOW, so a ``local_ends`` entry past ``out.shape[1]`` writes
   past the row -- and that applies to every row ``local_ends`` declares, padding included.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch

from ...jit.core import compile_ops
from ._arch import GFX950, GFX1250, _device_arch

DEFAULT_KV_BLOCK_SIZE = 64


@dataclass(frozen=True)
class MqaLogitsVariant:
    """One compiled kernel instance.

    Pick one by ``name`` from :func:`pa_mqa_logits_mxfp4_variants` and pass it to
    :func:`pa_mqa_logits_mxfp4_plan_buffers`. A caller needs nothing else from it: the three
    numbers are what the plan sizes its buffers and grid by.

    ``q_per_block``   query rows one CTA covers -- the axis the instances differ on.
    ``block_k``       the KV tile in tokens, which is what ``block_tables`` is sized in.
    ``cta_resident``  CTAs the part holds at once, which sizes the grid and aims the
                      schedule's split.

    **Every instance takes the same five input tensors and the same ``block_tables`` width**, so
    switching costs a caller nothing but the plan's own buffers. That is true of the instances
    this build has and is not a promise about a future one -- size ``block_tables`` with
    :func:`pa_mqa_logits_mxfp4_block_table_width` rather than from ``block_k`` by hand.
    """

    name: str
    q_per_block: int
    block_k: int
    cta_resident: int

    def __str__(self) -> str:
        return self.name


# `eq=False` on this and on MqaLogitsPlan: both hold tensors, so a generated `__eq__` would
# compare them elementwise and return a tensor instead of a bool.
@dataclass(frozen=True, eq=False)
class MqaLogitsBuffers:
    """One plan's caller-held buffers, the grid they were settled for, and **the kernel
    instance they were sized FOR**.

    Allocate it with :func:`pa_mqa_logits_mxfp4_plan_buffers` and hand the whole object back to
    :func:`pa_mqa_logits_mxfp4_plan` every forward. The instance travels WITH the memory because
    the memory is sized for it: ``cta_info``'s length follows ``cta_resident`` and ``cu_tiles``'
    follows ``q_per_block``, so a set allocated for one instance cannot build a plan for
    another.
    """

    cta_info: torch.Tensor
    # The launch GRID, pinned for a graph's lifetime once captured.
    num_ctas: int
    # Both None on gfx950, which has no tile cut and no named instances.
    cu_tiles: torch.Tensor | None = None
    variant: MqaLogitsVariant | None = None


# 4-D is the gfx950 MFMA permutation of the scales and its 4-chunk kv_cache; 3-D is gfx1250's
# all-natural form. They differ in no other observable way -- see the module docstring.
_PERMUTED_NDIM = 4
_NATURAL_NDIM = 3


# ══ the gfx1250 implementation ═══════════════════════════════════════════════════════════════
_MD_NAME_GFX1250 = "module_pa_mqa_logits_mxfp4_gfx1250_opus"

# THE KERNEL INSTANCES, most general first. **This list and the dispatch in
# `pa_mqa_logits_mxfp4_gfx1250_fwd_sched` are two halves of one fact and are edited together**:
# adding an instance is an arm there and a row here. A row naming a pair the module did not
# compile is refused by that dispatch, loudly, at the first launch.
#
# **`cta_resident` is TUNED and nothing checks it, on either side.** It is the CTAs the part
# holds at once; it follows the kernel's occupancy, so it has to be re-measured -- per instance,
# by sweeping, never by deriving -- whenever the traits' KV tile moves. A stale one leaves most
# of the part idle and nothing reports it.
_VARIANTS_GFX1250 = (
    # Four query rows per CTA, each wave one row sharing the CTA's KV tile: prefill and MTP > 1.
    MqaLogitsVariant("qlen4_kv64", q_per_block=4, block_k=64, cta_resident=768),
    # ONE row per CTA, a single wave32: MTP = 1 decode, where the above masks three waves off.
    MqaLogitsVariant("qlen1_kv64", q_per_block=1, block_k=64, cta_resident=3072),
)

# What a caller that passes no `variant` gets. NAMED and not positional, so adding an instance
# to the list above cannot move it, and `_as_variant` raises if the name ever stops existing.
#
# **A fixed default, deliberately, and not a rule over the shape.** Any such rule reads a mean
# over the batch, so a mixed forward misroutes part of it, and it gets harder to state as
# instances are added. Choosing is the CALLER's -- it is the one that knows its regime.
_DEFAULT_VARIANT_GFX1250 = "qlen4_kv64"

# Mirrors of the C++ constants. `_gfx1250_cta_info` is the ONLY place the buffer size is
# computed: the builder's own scratch sits past the slots in the same buffer, so open-coding
# `num_ctas * 8` anywhere is under-allocation.
_SCHED_RECORD_INTS = 8
_SCHED_SCRATCH_RECORDS = 96


# ── JIT stubs: signatures must match PA_MQA_LOGITS_MXFP4_GFX1250_PYBIND exactly ──────────────
# `fc_name` keeps the pybind name while the python name says these are private. They take the
# raw ABI (empty-tensor sentinels, not None) and can carry NO arch check: `compile_ops` replaces
# the body and never calls it, so a guard here would be dead code.
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
    q_per_block: int,
    block_k: int,
) -> None: ...


def _variants_gfx1250() -> tuple[MqaLogitsVariant, ...]:
    """The kernel instances, from :data:`_VARIANTS_GFX1250` -- see that constant."""
    return _VARIANTS_GFX1250


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
    q_per_block: int,
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
    block_k: int,
) -> None: ...


def _as_i32(t):
    """The int32, contiguous form the C++ launchers require, or None.

    Called ONCE, in the plan builder, and the result is what the plan carries -- not per launch.
    Both are no-ops on a tensor that already conforms, so the common caller pays nothing; a
    caller holding int64 windows pays one copy per forward instead of one per CSA layer, which
    is 61 of them on DeepSeek-V4.
    """
    return None if t is None else t.to(torch.int32).contiguous()


def _gfx1250_max_tiles_for(total_q: int, batch: int, variant: MqaLogitsVariant) -> int:
    """Tiles the cut can produce, from the static shapes alone -- which is what keeps the launch
    cudagraph-safe.

    The real count is ``sum_b ceil(qlen_b / Q_PER_BLOCK)`` and depends on device data. This sums
    the per-batch roundings before the divide, so it is never short and is exact whenever they
    tile. The slack is at most ``batch - 1`` tiles, each of which gets an empty record.
    """
    qpb = variant.q_per_block
    return (int(total_q) + int(batch) * (qpb - 1)) // qpb


def _gfx1250_sched_slots(num_tiles: int, variant: MqaLogitsVariant) -> int:
    """The ``num_ctas`` GRID, NOT the buffer's record count -- that is this plus the builder's
    scratch and lives in the allocation below.

    The floor of one resident round lets a handful of long tiles spread over the GPU; the
    rounding leaves the split room above it, since a split needs MORE slots than tiles. Tight
    rather than generous: a surplus slot's CTA still reads a 32-byte record nobody else touches.
    """
    r = variant.cta_resident
    n = max(int(num_tiles), r)
    return -(-n // r) * r


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


def _gfx1250_compute_tiles(cu_seq_q, total_q, variant, out=None):
    """Cut the query rows into tiles of at most ``Q_PER_BLOCK``, device-side.

    ``num_tiles`` is an UPPER BOUND, so it stays a host int and no device read is needed to
    launch; tiles past the real count are written empty. This cut is what guarantees "a tile is
    contiguous rows of one batch", which the kernel cannot check and does not survive.
    """
    cu = cu_seq_q.to(torch.int32).contiguous()
    batch = int(cu.shape[0]) - 1
    num_tiles = _gfx1250_max_tiles_for(total_q, batch, variant)
    cu_tiles = _gfx1250_cu_tiles(cu.device, num_tiles) if out is None else out
    _gfx1250_build_tiles_raw(
        cu, cu_tiles, int(total_q), int(num_tiles), variant.q_per_block
    )
    return cu_tiles, num_tiles


def _gfx1250_compute_schedule(
    cu_tiles,
    local_ends,
    num_tiles,
    variant,
    *,
    local_starts=None,
    row_to_batch=None,
    num_ctas=None,
    cta_info=None,
):
    """Build the per-tile schedule. Device-side, no sync.

    The three window arrays arrive already `_as_i32`, from the plan builder; this does not
    convert them, so a caller reaching past the plan gets the C++ dtype check rather than a
    silent copy.

    Safe to reuse the buffer across forwards -- every slot is written, surplus ones included.
    """
    n = int(num_tiles)
    slots = _gfx1250_sched_slots(n, variant) if num_ctas is None else int(num_ctas)
    if cta_info is None:
        cta_info = _gfx1250_cta_info(local_ends.device, slots)
    empty = torch.empty(0, dtype=torch.int32, device=local_ends.device)
    _gfx1250_build_sched_raw(
        cu_tiles,
        local_starts if local_starts is not None else empty,
        local_ends,
        row_to_batch if row_to_batch is not None else empty,
        cta_info,
        n,
        slots,
        variant.cta_resident,
        variant.block_k,
    )
    return cta_info, slots


def _gfx1250_launch(
    variant,
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
    union. It must be the SAME array -- the plan's, already `_as_i32` and not re-converted here,
    since this runs once per CSA layer. A shorter one is rejected, a differently-valued one is
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
        local_ends,
        cta_info,
        out,
        total_rows,
        int(num_ctas),
        float(weight_scale),
        int(kv_block_size),
        int(max_seq_len),
        variant.q_per_block,
        variant.block_k,
    )
    return out


# ══ the gfx950 implementation ════════════════════════════════════════════════════════════════
def _gfx950_mod():
    # Lazy and separately reported: the gfx950 op lands with ROCm/aiter#5332, so on this tree
    # neither its JIT config nor its C++ sources exist and the IMPORT is what fails. The arms
    # below are written against that module's API.
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

    Built by :func:`pa_mqa_logits_mxfp4_plan` and passed to every one of the forward's layer
    launches; rebuilding it per layer turns a per-forward cost into a per-layer one, 61x on
    DeepSeek-V4. Read ``num_ctas`` at a CUDAGraph capture and compare it against ``num_tiles``
    to see whether the schedule split a tile across CTAs (``num_ctas > num_tiles``).
    """

    arch: str
    cta_info: torch.Tensor
    # The launch GRID. A graph replays the grid it captured and nothing re-checks it against a
    # later table, so it has to come from buffers the caller holds.
    num_ctas: int
    # gfx1250 only, int32 + contiguous, normalized once here so the per-layer launch does not.
    # The windows travel WITH the plan because the kernel reads them PER ROW for the store mask
    # while the table carries only each tile's union, and they must be the same arrays the
    # schedule was built from. gfx950's records carry each row's window, so its launch never
    # sees these. `num_tiles` is the cut's UPPER BOUND, not its exact count.
    cu_tiles: torch.Tensor | None = None
    num_tiles: int | None = None
    local_starts: torch.Tensor | None = None
    local_ends: torch.Tensor | None = None
    # gfx950 only: the table's chunk indices are in `block_k` units and nothing cross-checks
    # them, so the launch must be given the value the builder used.
    block_k: int | None = None
    # The kernel instance the buffers and grid are sized for, and the one the launch dispatches
    # to. It comes from the buffers and is never a launch argument, because every one of the 61
    # per-forward launches would be a chance to disagree with the table the plan built.
    variant: MqaLogitsVariant | None = None


def pa_mqa_logits_mxfp4_variants(
    device: torch.device | str | int | None = None,
) -> tuple[MqaLogitsVariant, ...]:
    """The kernel instances this build compiled, most general first.

    Pass one -- or its ``name`` -- to :func:`pa_mqa_logits_mxfp4_plan_buffers`; leave that
    argument ``None`` and the buffers take the general instance. **Nothing infers the choice
    from the shape**, so a caller that knows its regime asks for what it wants: MTP = 1 decode
    wants the one-row instance, everything else the default.

    They all take the same five input tensors and the same ``block_tables`` width, so switching
    is free for the caller's own allocations; only the plan's buffers and grid move.

    ``device`` defaults to the current one. Past that arch probe this does no GPU work and
    never triggers the JIT build, so it is safe to call while sizing buffers at startup.

    Returns ``()`` on gfx950, which selects its kernel with ``block_k`` instead.
    """
    arch = _device_arch(torch.cuda.current_device() if device is None else device)
    if arch == GFX1250:
        return _variants_gfx1250()
    if arch == GFX950:
        # gfx950 selects its kernel with `block_k` rather than a named instance; see
        # `pa_mqa_logits_mxfp4_plan`'s `block_k` argument.
        return ()
    raise RuntimeError(
        f"the MXFP4 MQA-logits op supports {GFX950} and {GFX1250}, got {arch}"
    )


def _as_variant(variant, arch: str) -> MqaLogitsVariant:
    """Accept a name or a MqaLogitsVariant; reject anything this build did not compile."""
    if arch != GFX1250:
        raise ValueError(f"named variants are gfx1250-only; this device is {arch}")
    compiled = _variants_gfx1250()
    if isinstance(variant, MqaLogitsVariant):
        if variant not in compiled:
            raise ValueError(
                f"variant {variant.name!r} is not one this build compiled; available: "
                f"{[v.name for v in compiled]}"
            )
        return variant
    for v in compiled:
        if v.name == variant:
            return v
    raise ValueError(
        f"unknown variant {variant!r}; available: {[v.name for v in compiled]}"
    )


def pa_mqa_logits_mxfp4_block_table_width(
    max_seq_len: int,
    variant: MqaLogitsVariant | str | Iterable[MqaLogitsVariant | str] | None = None,
    *,
    kv_block_size: int = DEFAULT_KV_BLOCK_SIZE,
) -> int:
    """Minimum ``block_tables.shape[1]``, i.e. page-table entries per sequence.

    **A CTA rounds its window up to a whole KV TILE and reads the table at every page of the
    last one**, including where the window stops inside it -- so the bound is the tile rounding,
    not the page rounding, and the C++ launcher raises below it. Use this rather than computing
    it from ``kv_block_size``: whether a tile happens to be one page is a property of the build.

    ``variant`` may be one, several, or ``None`` for all of them. A caller holding ONE
    ``block_tables`` while switching instances must size it for the widest, and passing several
    is how to say so.

    Pure arithmetic -- no GPU, no JIT build -- so it is safe to call at startup.
    """
    if variant is None:
        chosen = _variants_gfx1250()
    elif isinstance(variant, (MqaLogitsVariant, str)):
        chosen = (_as_variant(variant, GFX1250),)
    else:
        chosen = tuple(_as_variant(v, GFX1250) for v in variant)
    n = int(max_seq_len)
    ksz = int(kv_block_size)
    # A tile must be a whole number of pages or the rounding below is not one: floor division
    # would return a width of 0 at `kv_block_size > block_k`, leaving the launcher to raise
    # about something else.
    bad = [v for v in chosen if ksz < 1 or v.block_k % ksz]
    if bad:
        raise ValueError(
            f"kv_block_size={ksz} must divide the KV tile; it does not for "
            f"{[(v.name, v.block_k) for v in bad]}"
        )
    return max(-(-n // v.block_k) * (v.block_k // ksz) for v in chosen)


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
    variant: MqaLogitsVariant | str | None = None,
    num_ctas: int | None = None,
) -> MqaLogitsBuffers:
    """Allocate a plan's buffers and settle its grid from the STATIC shapes alone.

    Returns a :class:`MqaLogitsBuffers` to hand back to :func:`pa_mqa_logits_mxfp4_plan` every
    forward, as ``buffers=``. **A CUDAGraph caller must go through this**, because a replay
    reads the pointers and the grid it captured; see that function.

    ``variant`` names the kernel instance, by :class:`MqaLogitsVariant` or by name, and
    defaults to the general one. The buffers are sized FOR it and carry it, so the plan takes
    the choice from the memory and a set allocated for one instance cannot build a plan for
    another.

    Do not open-code what this computes: ``cta_info`` carries the builder's own scratch past
    the slots, and the grid is a two-step rounding, so either is one header change away from
    under-allocating.

    Size the buffers at the LARGEST shape they will be pinned to. A later forward with fewer
    rows writes fewer tiles into the same buffers and keeps the pinned grid, which is
    consistent; the reverse is what the launcher raises on.
    """
    arch = _device_arch(device)
    if arch == GFX1250:
        v = _as_variant(_DEFAULT_VARIANT_GFX1250 if variant is None else variant, arch)
        num_tiles = _gfx1250_max_tiles_for(total_q, batch, v)
        slots = (
            _gfx1250_sched_slots(num_tiles, v) if num_ctas is None else int(num_ctas)
        )
        return MqaLogitsBuffers(
            cta_info=_gfx1250_cta_info(device, slots),
            num_ctas=slots,
            cu_tiles=_gfx1250_cu_tiles(device, num_tiles),
            variant=v,
        )
    if arch == GFX950:
        mod = _gfx950_mod()
        slots = (
            mod.pa_mqa_logits_mxfp4_sched_slots(int(total_q))
            if num_ctas is None
            else int(num_ctas)
        )
        ints = mod.pa_mqa_logits_mxfp4_sched_buffer_ints(slots)
        return MqaLogitsBuffers(
            cta_info=torch.empty(
                (ints // mod.SCHED_RECORD_INTS, mod.SCHED_RECORD_INTS),
                dtype=torch.int32,
                device=device,
            ),
            num_ctas=slots,
        )
    raise RuntimeError(
        f"the MXFP4 MQA-logits op supports {GFX950} and {GFX1250}, got {arch}"
    )


def pa_mqa_logits_mxfp4_plan(
    cu_seq_q: torch.Tensor,
    local_ends: torch.Tensor,
    *,
    buffers: MqaLogitsBuffers | None = None,
    total_q: int | None = None,
    local_starts: torch.Tensor | None = None,
    row_to_batch: torch.Tensor | None = None,
    block_k: int | None = None,
) -> MqaLogitsPlan:
    """Build one forward's schedule on device. Call ONCE PER FORWARD, not once per layer.

    ``cu_seq_q`` is the batch's ``[batch + 1]`` query-row prefix sum. **gfx950 ignores it** and
    it is still required, because gfx1250 cuts its tiles from it, and that cut is what keeps a
    tile to contiguous rows of one batch -- a condition the kernel cannot check and DEADLOCKS
    on rather than answering wrong.

    ``local_ends`` is the per-row window end, ``[total_q]`` int32; ``total_q`` defaults to its
    element count. The windows are the CALLER's, so any rule works that satisfies the two
    conditions in the module docstring.

    ``local_starts`` may be ``None`` when every row starts at 0, which is what both ATOM paths
    do. ``row_to_batch`` may be ``None`` when ``block_tables`` is indexed by query ROW rather
    than by sequence; passing a per-sequence map against a per-token table, or the reverse,
    reads the wrong pages and produces plausible wrong numbers.

    ``block_k`` is gfx950-only and raises on gfx1250 rather than being ignored, because a
    silently-dropped tile width is a knob that looks like it took effect.

    ``buffers`` is a :func:`pa_mqa_logits_mxfp4_plan_buffers` result. Pass one and the plan uses
    ITS memory, ITS grid and ITS kernel instance; omit it and the plan allocates, settles the
    grid and takes the default instance. **There is no ``variant=`` here** -- a caller wanting
    another instance allocates its buffers with one and hands them back.

    **UNDER A CUDAGRAPH, ``buffers`` IS REQUIRED**, and omitting it fails silently. A replay
    reads the POINTERS and the GRID it captured, while this builder runs outside the graph: a
    fresh allocation per forward is one the graph never reads, so it reads whatever the caching
    allocator has since left at the captured address. Allocate once, capture, and hand the same
    object back every forward; the grid cannot drift either, since it comes from that object.

    Every slot is written, surplus ones included, so reusing buffers needs no clearing.
    """
    arch = _arch_of(local_ends)
    n = int(local_ends.numel()) if total_q is None else int(total_q)

    if arch == GFX1250:
        if block_k is not None:
            raise ValueError(
                "gfx1250 compiles one KV tile width; block_k is gfx950-only"
            )
        # All three ONCE, here, and the plan carries the result -- the launch reads the windows
        # per CSA LAYER, so converting there would be 61 copies for a caller holding int64.
        if buffers is None:
            v = _as_variant(_DEFAULT_VARIANT_GFX1250, arch)
            cta_info, cu_tiles, num_ctas = None, None, None
        else:
            if buffers.variant is None:
                raise ValueError(
                    "these buffers carry no kernel instance, so they were allocated for "
                    "gfx950; allocate them on the gfx1250 device"
                )
            v = buffers.variant
            cta_info, cu_tiles, num_ctas = (
                buffers.cta_info,
                buffers.cu_tiles,
                buffers.num_ctas,
            )
        ends, starts = _as_i32(local_ends), _as_i32(local_starts)
        tiles, num_tiles = _gfx1250_compute_tiles(cu_seq_q, n, v, out=cu_tiles)
        info, slots = _gfx1250_compute_schedule(
            tiles,
            ends,
            num_tiles,
            v,
            local_starts=starts,
            row_to_batch=_as_i32(row_to_batch),
            num_ctas=num_ctas,
            cta_info=cta_info,
        )
        return MqaLogitsPlan(
            arch=arch,
            cta_info=info,
            num_ctas=slots,
            cu_tiles=tiles,
            num_tiles=num_tiles,
            local_starts=starts,
            local_ends=ends,
            variant=v,
        )

    if buffers is not None and buffers.variant is not None:
        raise ValueError(
            "these buffers carry a gfx1250 kernel instance; allocate them on the gfx950 device"
        )
    mod = _gfx950_mod()
    bk = int(mod.BLOCK_K_1WAVE if block_k is None else block_k)
    info, slots = mod.pa_mqa_logits_mxfp4_build_sched(
        local_ends,
        n,
        local_starts=local_starts,
        row_to_batch=row_to_batch,
        block_k=bk,
        num_ctas=None if buffers is None else buffers.num_ctas,
        cta_target=mod.SCHED_CTA_TARGET,
        cta_info=None if buffers is None else buffers.cta_info,
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

    Call it once per CSA LAYER with the forward's one plan; which kernel instance runs comes
    from the plan and is not an argument here.

    Returns ``out``, allocating a ``[total_q, max_seq_len]`` fp32 tensor of ``-inf`` when not
    given one -- **pass an ``out`` you hold**, or a 61-layer forward allocates that tensor 61
    times. A reused ``out`` must be pre-filled with ``-inf``, since only in-window cells are
    written.

    ``block_tables`` must be sized by :func:`pa_mqa_logits_mxfp4_block_table_width`, which
    rounds to the KV TILE and not to ``kv_block_size``.
    """
    arch = _arch_of(q_fp4)
    if arch != plan.arch:
        raise ValueError(
            f"the plan was built for {plan.arch} and this launch is on {arch}; a plan is per "
            "forward AND per device"
        )
    _check_layout(arch, q_scale, kv_scale, kv_cache)

    if arch == GFX1250:
        if plan.variant is None:
            # Unreachable from `pa_mqa_logits_mxfp4_plan`, which always settles one. A default
            # here would dispatch to an instance the buffers were not sized for.
            raise ValueError(
                "this plan carries no kernel instance; build it with "
                "pa_mqa_logits_mxfp4_plan on a gfx1250 device"
            )
        return _gfx1250_launch(
            plan.variant,
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
    "MqaLogitsBuffers",
    "MqaLogitsPlan",
    "MqaLogitsVariant",
    "pa_mqa_logits_mxfp4",
    "pa_mqa_logits_mxfp4_block_table_width",
    "pa_mqa_logits_mxfp4_plan",
    "pa_mqa_logits_mxfp4_plan_buffers",
    "pa_mqa_logits_mxfp4_variants",
]
