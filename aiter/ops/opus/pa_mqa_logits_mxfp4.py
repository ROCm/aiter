# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""MXFP4 paged MQA logits for DeepSeek-style sparse attention (OPUS kernels), gfx950 and gfx1250.

Per query row ``r`` over a window ``[s, e)``:
``out[r, s:e] = sum_H( relu(Q[r] . K^T) * weight[r] ) * weight_scale``

Prefill and decode share one launch over a per-tile schedule built on device. The plan depends
only on per-forward data, so build it once per forward and reuse it for every CSA layer::

    width = pa_mqa_logits_mxfp4_block_table_width(max_model_len)          # once at startup
    buf  = pa_mqa_logits_mxfp4_plan_buffers(dev, total_q, batch,         # once per pool
                                            variant="qlen1_kv64")
    plan = pa_mqa_logits_mxfp4_plan(cu_seq_q, local_ends, buffers=buf)    # once per forward
    out  = pa_mqa_logits_mxfp4(q, q_scale, kv_cache, kv_scale,            # once per layer
                               block_tables, weights, plan, max_seq_len, out=out)

Every step is device-side with no readback, so the path is CUDAGraph-safe provided the buffers
are caller-held and reused (see :func:`pa_mqa_logits_mxfp4_plan`).

KERNEL INSTANCES (:func:`pa_mqa_logits_mxfp4_variants`) take identical inputs and are chosen
per buffer pool, never inferred from the shape. On gfx1250 the default suits prefill and
MTP > 1; MTP = 1 decode should ask for ``qlen1_kv64``.

LAYOUTS are the caller's; this module never touches the data:

=============  =======================================  =================================
tensor         gfx950                                   gfx1250
=============  =======================================  =================================
``q``          ``[total_q, H, D/2]`` uint8              same
``weights``    ``[total_q, H]`` bf16                    same
``q_scale``    ``[total_q, 2, 32, 4]`` MFMA-permuted    ``[total_q, H, 4]`` natural
``kv_scale``   ``[num_blocks, 2, 32, 4]`` permuted      ``[num_blocks, PAGE, 4]`` natural
``kv_cache``   ``[num_blocks, 4, PAGE, 16]``            ``[num_blocks, PAGE, D/2]`` natural
=============  =======================================  =================================

On gfx1250 each scale is the plain E8M0 byte of one 32-element K block, and each packed row is
64 contiguous bytes, low nibble first. Both layouts have the same byte count, so the C++
launchers (which check ``numel``) would accept the wrong one and return plausible wrong logits;
the ndim check here is the only guard, and it cannot catch an array reshaped to the right ndim.

THREE CONDITIONS the kernel cannot check:

1. the window rule is non-decreasing in the row index within a tile (true of every causal and
   CSA-compressed rule), so a tile's union is its first row's start and last row's end. If
   broken, a CTA's waves disagree on the trip count and DEADLOCK;
2. stores are bounded only by the window, so a ``local_ends`` entry past ``out.shape[1]``
   writes past the row -- padding rows included;
3. a row that belongs to no live sequence (a CUDAGraph pad row) carries an EMPTY window
   (``local_ends <= local_starts``), and only such a row may have a negative ``row_to_batch``.
   At ``q_per_block == 1`` ``cu_seq_q`` is not consulted, so this is what keeps a pad row from
   reading ``q`` or ``block_tables``.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace

import torch

from ...jit.core import compile_ops
from ._arch import GFX950, GFX1250, _device_arch

DEFAULT_KV_BLOCK_SIZE = 64


@dataclass(frozen=True)
class MqaLogitsVariant:
    """One compiled kernel instance; pick one from :func:`pa_mqa_logits_mxfp4_variants`.

    ``q_per_block``   query rows one CTA covers -- 4 or 1 on gfx1250, always 1 on gfx950.
    ``block_k``       the KV tile in tokens.
    ``cta_resident``  CTAs resident at once; sizes the grid and the schedule's split.
    ``arch``          the target it was compiled for; refused on any other. Both arches have
                      a ``qlen1_kv64``. ``None`` (hand-built) matches by value on the device.

    All instances of one arch take the same inputs; size ``block_tables`` with
    :func:`pa_mqa_logits_mxfp4_block_table_width`.
    """

    name: str
    q_per_block: int
    block_k: int
    cta_resident: int
    arch: str | None = None

    def __str__(self) -> str:
        return self.name


# eq=False here and on MqaLogitsPlan: a generated __eq__ would compare tensors elementwise.
@dataclass(frozen=True, eq=False)
class MqaLogitsBuffers:
    """A plan's caller-held buffers, their grid, and the kernel instance they are sized for.

    Allocate with :func:`pa_mqa_logits_mxfp4_plan_buffers` and pass back to
    :func:`pa_mqa_logits_mxfp4_plan` every forward. ``cu_tiles`` is empty at
    ``q_per_block == 1`` (a tile is a row). Buffers for one instance or arch cannot plan another.
    """

    cta_info: torch.Tensor
    cu_tiles: torch.Tensor
    # The launch grid; pinned once a graph captures it.
    num_ctas: int
    variant: MqaLogitsVariant


# gfx950 scales/kv_cache are 4-D (permuted); gfx1250's are 3-D (natural).
_PERMUTED_NDIM = 4
_NATURAL_NDIM = 3


# == the kernel instances ==========================================================================
_MD_NAME = "module_pa_mqa_logits_mxfp4_opus"

# Kernel instances per arch, most general first. Each row must match an arm of the C++
# `pa_mqa_logits_mxfp4_fwd_sched` dispatch, which refuses an uncompiled pair at launch.
#
# `cta_resident` is the CTAs resident at once and follows the kernel's occupancy; retune it
# when the KV tile changes. A stale value only under-fills the GPU, never gives wrong results.
# Both gfx950 instances use 1024.
_VARIANTS = {
    GFX1250: (
        # Four rows per CTA, one per wave, sharing a KV tile: prefill, MTP > 1.
        MqaLogitsVariant(
            "qlen4_kv64", q_per_block=4, block_k=64, cta_resident=768, arch=GFX1250
        ),
        # One row per CTA, a single wave32: MTP = 1 decode.
        MqaLogitsVariant(
            "qlen1_kv64", q_per_block=1, block_k=64, cta_resident=3072, arch=GFX1250
        ),
    ),
    GFX950: (
        # One row per CTA, one wave over a 64-token KV tile: the default.
        MqaLogitsVariant(
            "qlen1_kv64", q_per_block=1, block_k=64, cta_resident=1024, arch=GFX950
        ),
        # One row per CTA, four waves over a 256-token KV tile: for long windows.
        MqaLogitsVariant(
            "qlen1_kv256", q_per_block=1, block_k=256, cta_resident=1024, arch=GFX950
        ),
    ),
}

# Default instance per arch, by name. Fixed rather than shape-derived: a rule over the shape
# would average over the batch and misroute part of a mixed forward.
_DEFAULT_VARIANT = {GFX1250: "qlen4_kv64", GFX950: "qlen1_kv64"}


def _check_variant_tables():
    """Import-time check: each instance is filed under its own arch and has positive numbers."""
    for arch, variants in _VARIANTS.items():
        for v in variants:
            if v.arch != arch:
                raise AssertionError(
                    f"{v} is filed under {arch} but names arch={v.arch}"
                )
            if min(v.q_per_block, v.block_k, v.cta_resident) < 1:
                raise AssertionError(
                    f"{arch} {v}: q_per_block / block_k / cta_resident < 1"
                )


_check_variant_tables()

# Mirrors of the C++ constants. Size cta_info only via `_cta_info` (it includes the scratch).
_SCHED_RECORD_INTS = 8
_SCHED_SCRATCH_RECORDS = 96


# == JIT stubs: signatures must match PA_MQA_LOGITS_MXFP4_PYBIND exactly ===========================
# Raw ABI (empty-tensor sentinels, not None). `compile_ops` replaces the body, so no checks can
# live here; the C++ side dispatches on the runtime arch.
@compile_ops(
    _MD_NAME,
    fc_name="pa_mqa_logits_mxfp4_fwd_sched",
    develop=True,
)
def _fwd_sched_raw(
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


@compile_ops(
    _MD_NAME,
    fc_name="pa_mqa_logits_mxfp4_build_tiles",
    develop=True,
)
def _build_tiles_raw(
    cu_seq_q: torch.Tensor,
    cu_tiles: torch.Tensor,
    total_q: int,
    max_tiles: int,
    q_per_block: int,
) -> None: ...


@compile_ops(
    _MD_NAME,
    fc_name="pa_mqa_logits_mxfp4_build_sched",
    develop=True,
)
def _build_sched_raw(
    cu_tiles: torch.Tensor,
    local_starts: torch.Tensor,
    local_ends: torch.Tensor,
    row_to_batch: torch.Tensor,
    cta_info: torch.Tensor,
    num_tiles: int,
    num_rows: int,
    num_ctas: int,
    cta_resident: int,
    block_k: int,
    q_per_block: int,
) -> None: ...


def _variants_for(arch: str) -> tuple[MqaLogitsVariant, ...]:
    """The kernel instances this build compiled for ``arch``, most general first."""
    try:
        return _VARIANTS[arch]
    except KeyError:
        raise RuntimeError(
            f"the MXFP4 MQA-logits op supports {GFX950} and {GFX1250}, got {arch}"
        ) from None


def _default_variant(arch: str) -> str:
    """The name a caller that passes no ``variant`` gets, for ``arch``."""
    try:
        return _DEFAULT_VARIANT[arch]
    except KeyError:
        raise RuntimeError(
            f"the MXFP4 MQA-logits op supports {GFX950} and {GFX1250}, got {arch}"
        ) from None


def _as_i32(t):
    """int32 contiguous form required by C++, or None. Done once per plan, not per launch."""
    return None if t is None else t.to(torch.int32).contiguous()


def _max_tiles_for(total_q: int, batch: int, variant: MqaLogitsVariant) -> int:
    """Upper bound on ``sum_b ceil(qlen_b / q_per_block)`` from static shapes only.

    Slack is at most ``batch - 1`` (empty) tiles; exact at ``q_per_block == 1``.
    """
    qpb = variant.q_per_block
    return (int(total_q) + int(batch) * (qpb - 1)) // qpb


def _sched_slots(num_tiles: int, variant: MqaLogitsVariant) -> int:
    """The ``num_ctas`` grid: ``num_tiles`` rounded up to whole resident rounds, at least one
    round so a few long tiles can be split across the GPU. Excludes the builder's scratch.
    """
    r = variant.cta_resident
    n = max(int(num_tiles), r)
    return -(-n // r) * r


def _cta_info(device, num_ctas):
    # Slots plus the builder's scratch, which follows them.
    return torch.empty(
        (int(num_ctas) + _SCHED_SCRATCH_RECORDS, _SCHED_RECORD_INTS),
        dtype=torch.int32,
        device=device,
    )


def _cu_tiles(device, num_tiles):
    # Tile t covers rows [cu_tiles[t], cu_tiles[t + 1]); hence the terminator.
    return torch.empty(int(num_tiles) + 1, dtype=torch.int32, device=device)


def _compute_tiles(cu_seq_q, total_q, variant, out=None):
    """Cut rows into tiles of at most ``q_per_block`` rows of one batch, on device.

    Only used at ``q_per_block > 1``. ``num_tiles`` is a host-side upper bound; extra tiles are
    written empty.
    """
    cu = cu_seq_q.to(torch.int32).contiguous()
    batch = int(cu.shape[0]) - 1
    num_tiles = _max_tiles_for(total_q, batch, variant)
    cu_tiles = _cu_tiles(cu.device, num_tiles) if out is None else out
    _build_tiles_raw(cu, cu_tiles, int(total_q), int(num_tiles), variant.q_per_block)
    return cu_tiles, num_tiles


def _compute_schedule(
    cu_tiles,
    local_ends,
    num_tiles,
    num_rows,
    variant,
    *,
    local_starts=None,
    row_to_batch=None,
    num_ctas=None,
    cta_info=None,
):
    """Build the per-tile schedule on device, no sync. Window arrays must already be int32.

    Every slot is written, so ``cta_info`` can be reused across forwards without clearing.
    """
    n = int(num_tiles)
    slots = _sched_slots(n, variant) if num_ctas is None else int(num_ctas)
    if cta_info is None:
        cta_info = _cta_info(local_ends.device, slots)
    empty = torch.empty(0, dtype=torch.int32, device=local_ends.device)
    _build_sched_raw(
        cu_tiles if cu_tiles is not None else empty,
        local_starts if local_starts is not None else empty,
        local_ends,
        row_to_batch if row_to_batch is not None else empty,
        cta_info,
        n,
        int(num_rows),
        slots,
        variant.cta_resident,
        variant.block_k,
        variant.q_per_block,
    )
    return cta_info, slots


def _launch(
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
    num_rows,
    max_seq_len,
    *,
    local_starts=None,
    weight_scale=1.0,
    kv_block_size=DEFAULT_KV_BLOCK_SIZE,
    out=None,
):
    """Launch the kernel. ``local_ends``/``local_starts`` must be the plan's own arrays (the
    kernel masks stores per row); ``num_rows`` is the plan's row count, not ``q``'s."""
    if out is None:
        out = torch.full(
            (int(q_fp4.shape[0]), max_seq_len),
            float("-inf"),
            dtype=torch.float32,
            device=q_fp4.device,
        )
    empty = torch.empty(0, dtype=torch.int32, device=q_fp4.device)
    _fwd_sched_raw(
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
        int(num_rows),
        int(num_ctas),
        float(weight_scale),
        int(kv_block_size),
        int(max_seq_len),
        variant.q_per_block,
        variant.block_k,
    )
    return out


# == the dispatcher ================================================================================
@dataclass(frozen=True, eq=False)
class MqaLogitsPlan:
    """One forward's schedule plus the windows and instance its layer launches need.

    Built by :func:`pa_mqa_logits_mxfp4_plan`; reuse it for every layer of the forward.
    ``num_ctas > num_tiles`` means the schedule split tiles across CTAs.
    """

    arch: str
    cta_info: torch.Tensor
    # The launch grid; under a graph it must come from caller-held buffers.
    num_ctas: int
    # Windows are int32 contiguous; the kernel reads them per row for the store mask.
    # `num_tiles` is an upper bound. `cu_tiles` is empty at `q_per_block == 1` (tile t = row t).
    cu_tiles: torch.Tensor
    num_tiles: int
    local_ends: torch.Tensor
    local_starts: torch.Tensor | None = None
    # The instance the buffers are sized for; the launch dispatches on it.
    variant: MqaLogitsVariant | None = None
    # Rows the schedule covers (`total_q`); the launch raises if `q` holds fewer.
    num_rows: int | None = None


def pa_mqa_logits_mxfp4_variants(
    device: torch.device | str | int | None = None,
) -> tuple[MqaLogitsVariant, ...]:
    """The kernel instances compiled for the device's arch, most general first.

    Pass one (or its ``name``) to :func:`pa_mqa_logits_mxfp4_plan_buffers`; the choice is never
    inferred from the shape. ``device`` defaults to the current one. Only probes the arch; no
    GPU work or JIT build, so it is safe at startup.
    """
    arch = _device_arch(torch.cuda.current_device() if device is None else device)
    return _variants_for(arch)


def _as_variant(variant, arch: str) -> MqaLogitsVariant:
    """Resolve a name or MqaLogitsVariant to this arch's compiled instance, or raise.
    ``arch=None`` objects are matched by value against this arch's instances only."""
    compiled = _variants_for(arch)
    if isinstance(variant, MqaLogitsVariant):
        if variant.arch is not None and variant.arch != arch:
            raise ValueError(
                f"variant {variant.name!r} was compiled for {variant.arch}, not {arch}; take "
                f"this device's from pa_mqa_logits_mxfp4_variants()"
            )
        match = [v for v in compiled if v == replace(variant, arch=v.arch)]
        if not match:
            raise ValueError(
                f"variant {variant.name!r} is not one {arch} compiled; available: "
                f"{[v.name for v in compiled]}"
            )
        return match[0]
    for v in compiled:
        if v.name == variant:
            return v
    raise ValueError(
        f"unknown variant {variant!r} for {arch}; available: {[v.name for v in compiled]}"
    )


def pa_mqa_logits_mxfp4_block_table_width(
    max_seq_len: int,
    variant: MqaLogitsVariant | str | Iterable[MqaLogitsVariant | str] | None = None,
    *,
    device: torch.device | str | int | None = None,
    kv_block_size: int = DEFAULT_KV_BLOCK_SIZE,
) -> int:
    """Minimum ``block_tables.shape[1]`` (page-table entries per sequence).

    A CTA rounds its window up to a whole KV tile (``block_k``) and reads the table for every
    page of it, so the width is ``max_seq_len`` rounded up to the tile, in pages; the C++
    launcher raises below it. Use this rather than deriving it from ``kv_block_size``.

    ``variant`` may be one, several, or ``None`` for all of the arch's; the widest wins, so a
    table shared across instances should pass several or ``None`` (on gfx950 that sizes for
    ``block_k`` 256). ``device`` picks the arch for names or ``None`` (default: current device);
    with only :class:`MqaLogitsVariant` objects no GPU is touched. Never triggers the JIT build.
    """
    wanted = (
        None
        if variant is None
        else (
            (variant,)
            if isinstance(variant, (MqaLogitsVariant, str))
            else tuple(variant)
        )
    )
    if (
        device is None
        and wanted is not None
        and all(isinstance(v, MqaLogitsVariant) and v.arch is not None for v in wanted)
    ):
        # Each object names its arch, so no device probe; mixing arches is a caller error.
        arches = {v.arch for v in wanted}
        if len(arches) > 1:
            raise ValueError(f"variants from more than one arch: {sorted(arches)}")
        chosen = tuple(_as_variant(v, v.arch) for v in wanted)
    else:
        arch = _device_arch(torch.cuda.current_device() if device is None else device)
        chosen = (
            _variants_for(arch)
            if wanted is None
            else tuple(_as_variant(v, arch) for v in wanted)
        )
    n = int(max_seq_len)
    ksz = int(kv_block_size)
    # A tile must be a whole number of pages, or the width below would be wrong (0 when
    # kv_block_size > block_k).
    bad = [v for v in chosen if ksz < 1 or v.block_k % ksz]
    if bad:
        raise ValueError(
            f"kv_block_size={ksz} must divide the KV tile; it does not for "
            f"{[(v.name, v.block_k) for v in bad]}"
        )
    return max(-(-n // v.block_k) * (v.block_k // ksz) for v in chosen)


def _arch_of(t: torch.Tensor) -> str:
    """The arch of the tensor's own device (not the current one); raises if unsupported."""
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
    """Allocate a plan's buffers and fix its grid from static shapes only.

    Pass the result to :func:`pa_mqa_logits_mxfp4_plan` as ``buffers=`` every forward.
    **Required under a CUDAGraph**, since a replay reuses the captured pointers and grid.

    ``variant`` (object or name; default: the arch's general instance) is carried by the
    buffers, which cannot plan for another instance or arch. Size for the LARGEST
    ``total_q``/``batch`` the buffers will serve; smaller forwards reuse them, larger ones raise.
    """
    arch = _device_arch(device)
    v = _as_variant(_default_variant(arch) if variant is None else variant, arch)
    num_tiles = _max_tiles_for(total_q, batch, v)
    slots = _sched_slots(num_tiles, v) if num_ctas is None else int(num_ctas)
    return MqaLogitsBuffers(
        cta_info=_cta_info(device, slots),
        # One row per tile needs no cu_tiles.
        cu_tiles=(
            torch.empty(0, dtype=torch.int32, device=device)
            if v.q_per_block == 1
            else _cu_tiles(device, num_tiles)
        ),
        num_ctas=slots,
        variant=v,
    )


def pa_mqa_logits_mxfp4_plan(
    cu_seq_q: torch.Tensor,
    local_ends: torch.Tensor,
    *,
    buffers: MqaLogitsBuffers | None = None,
    total_q: int | None = None,
    local_starts: torch.Tensor | None = None,
    row_to_batch: torch.Tensor | None = None,
) -> MqaLogitsPlan:
    """Build one forward's schedule on device. Call once per forward, not per layer.

    ``cu_seq_q`` is the ``[batch + 1]`` query-row prefix sum. At ``q_per_block > 1`` tiles are
    cut from it so each holds rows of one batch only (the kernel would deadlock otherwise). At
    ``q_per_block == 1`` it is not consulted and the row domain is ``total_q``, not
    ``cu_seq_q[batch]``.

    ``local_ends`` is the per-row window end, ``[total_q]``; ``total_q`` defaults to its
    ``numel`` (pass it when ``local_ends`` is longer than ``q``). ``local_starts=None`` means
    every row starts at 0. Windows must meet the module docstring's conditions.

    ``row_to_batch`` maps each row to its ``block_tables`` row; ``None`` means ``block_tables``
    is indexed per query row. Mixing per-sequence and per-row conventions silently reads wrong
    pages. Negative ids are allowed only on empty-window rows.

    ``buffers`` (from :func:`pa_mqa_logits_mxfp4_plan_buffers`) supplies the memory, grid and
    kernel instance; without it the plan allocates fresh and uses the arch's default instance.

    **Under a CUDAGraph, ``buffers`` is REQUIRED**: a replay reads the captured pointers and
    grid, so fresh allocations here would silently go unread. Reuse the same object every
    forward; no clearing is needed.
    """
    arch = _arch_of(local_ends)
    n = int(local_ends.numel()) if total_q is None else int(total_q)

    if buffers is None:
        v = _as_variant(_default_variant(arch), arch)
        cta_info, cu_tiles, num_ctas = None, None, None
    else:
        if buffers.variant.arch not in (None, arch):
            raise ValueError(
                f"these buffers were sized for {buffers.variant.arch}'s "
                f"{buffers.variant.name!r} instance and this plan is for {arch}; allocate them "
                "on the device the launch will run on"
            )
        v = _as_variant(buffers.variant, arch)
        cta_info, cu_tiles, num_ctas = (
            buffers.cta_info,
            buffers.cu_tiles,
            buffers.num_ctas,
        )
    # Convert once per forward; the plan carries the result to every layer.
    ends, starts = _as_i32(local_ends), _as_i32(local_starts)
    if v.q_per_block == 1:
        # One tile per row: no cut kernel, and `num_tiles` is exact (== rows).
        tiles = (
            cu_tiles
            if cu_tiles is not None
            else torch.empty(0, dtype=torch.int32, device=ends.device)
        )
        num_tiles = _max_tiles_for(n, int(cu_seq_q.shape[0]) - 1, v)
    else:
        tiles, num_tiles = _compute_tiles(cu_seq_q, n, v, out=cu_tiles)
    info, slots = _compute_schedule(
        tiles,
        ends,
        num_tiles,
        n,
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
        num_rows=n,
    )


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
    """Paged MQA logits over ``plan`` (prefill or decode). Call once per layer.

    The kernel instance comes from ``plan``. Returns ``out``; if not given, allocates a
    ``[total_q, max_seq_len]`` fp32 tensor of ``-inf`` (pass one to avoid a per-layer
    allocation). Only in-window cells ``[local_start, local_end)`` are written, so a
    caller-held ``out`` needs ``-inf`` pre-fill only if its reader looks outside the windows.

    Size ``block_tables`` with :func:`pa_mqa_logits_mxfp4_block_table_width`.
    """
    arch = _arch_of(q_fp4)
    if arch != plan.arch:
        raise ValueError(
            f"the plan was built for {plan.arch} and this launch is on {arch}; a plan is per "
            "forward AND per device"
        )
    _check_layout(arch, q_scale, kv_scale, kv_cache)
    if plan.variant is None:
        # Never guess: the buffers are sized for a specific instance.
        raise ValueError(
            "this plan carries no kernel instance; build it with pa_mqa_logits_mxfp4_plan"
        )
    return _launch(
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
        # A hand-built plan without `num_rows` covers every row of q.
        int(q_fp4.shape[0]) if plan.num_rows is None else plan.num_rows,
        int(max_seq_len),
        local_starts=plan.local_starts,
        weight_scale=weight_scale,
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
