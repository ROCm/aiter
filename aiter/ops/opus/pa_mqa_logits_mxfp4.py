# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""MXFP4 paged MQA logits (OPUS kernels): one entry for DeepSeek-V4 (64 index heads) and V4.1
(32 index heads, full and candidate layers).

Per query row ``r`` of sequence ``b`` over its window ``[0, row_ends[r])``::

    out[r, c] = sum_h relu(q[r, h] . k_b[c]) * weights[r, h] * weight_scale

Sequence ``b`` holds rows ``query_start_loc[b] .. query_start_loc[b + 1] - 1`` and reads its keys
through ``block_tables[b]``, an entry naming ``pages_per_block`` consecutive ``page_size``-row
pages (its value is the first page's id / ``pages_per_block``)::

    cfg   = mqa_logits_mxfp4_config(dev, heads=64, page_size=64, max_rows=..., max_seqs=...,
                                    max_qlen=4, max_seq_len=...)   # once per shape bucket
    width = cfg.block_table_width                                   # min block_tables.shape[1]
    buf   = cfg.schedule_buffers()                                  # optional, caller-held
    sched = mqa_logits_mxfp4_schedule(cfg, query_start_loc, row_ends, buffers=buf)  # per forward
    out   = mqa_logits_mxfp4(q, q_scale, kv_cache, kv_scale, weights, block_tables,
                             query_start_loc, row_ends, cfg=cfg, schedule=sched, out=out)

The caller never names a kernel instance: the config picks one from the static bounds. A
schedule is optional. Given, the launch runs the table-driven mode; absent, it runs the
instance's schedule-free mode, or (an instance without one) builds a schedule internally into
buffers the config owns -- correct and graph-safe, one more launch a call, not reentrant across
concurrent streams. Nothing reads device data on the host, so every path can be captured.

A slice of rows ``[start, start + count)`` (a band) is scored by slicing the per-row tensors and
passing ``query_start_loc.clamp(start, start + count) - start``.

LAYOUTS (the caller's; nothing here touches the data). Every arch::

    q          [rows, H, D/2]  uint8 / fp4x2     weights          [rows, H]       bf16
    row_ends   [rows]          int32             query_start_loc  [seqs + 1]      int32
    block_tables [seqs, width] int32             out [rows, >= max_seq_len] fp32, or flat
                                                     with out_offsets [rows] int64

and per arch (the KV cache writer's; both have the same byte counts, so only the ndim tells
them apart -- checked here, as nothing on device can)::

               gfx1250 (natural)                 gfx950 (MFMA-permuted, page_size 64)
    q_scale    [rows, H, D/32]   uint8 (E8M0)    [rows, 2, 32, 4]
    kv_cache   [pages, page_size, D/2]           [pages, 4, page_size, 16]
    kv_scale   [pages, page_size, D/32]          [pages, 2, 32, 4]

Only ``out[r, c]`` for ``c < row_ends[r]`` is written; rows past ``query_start_loc[-1]`` (CUDAGraph
padding) are untouched. The caller owes what the kernel cannot check: ``row_ends`` does not
decrease along a sequence, ``row_ends[r] <= max_seq_len`` and within the sequence's table, and a
page's offset fits 32 bits -- on gfx950 its byte offset (page id x 4 KiB: a ``kv_cache`` of at most
4 GiB), on gfx1250 its row (page id x ``page_size``).

NaN: relu is IEEE ``maximum``. A logit is NaN exactly where a key's or the row's E8M0 scales hold
0xFF, or a weight is NaN or Inf (the kernels compute relu as max_num and put the NaN back from the
scales; the gfx1250 TU is built with -fno-honor-nans).
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

import torch

from ...jit.core import compile_ops
from ._arch import GFX950, GFX1250, _device_arch

_MD_NAME = "module_pa_mqa_logits_mxfp4_opus"

# Schedule modes (opus_logits::mqa_logits_mode).
TABLE, FREE, PREFILL = 0, 1, 2
_MODE_NAMES = {"table": TABLE, "free": FREE, "prefill": PREFILL}

_TILE = 64  # keys a KV tile, every instance's (gfx1250's rule tells V4.1 layers by it)
# Mirrors of the C++ builder's constants (pa_mqa_logits_mxfp4.cu).
_SCHED_RECORD_INTS = 8
_SCHED_SCRATCH_RECORDS = 96
_GROUPS_BUILD_MAX_SEQS = 2048
# max_qlen above this is a prefill bucket.
_DECODE_MAX_QLEN = 8
# Decode tiers: the smallest tier >= max_qlen picks the instance. 64 heads (V4): rows a CTA, one a
# wave; a tier above the CTA's rows runs two or three groups a sequence (a CTA that fills a CU
# stalls every SIMD at its per-phase barrier at once).
_GFX1250_H64_DECODE_TIERS = {1: 1, 2: 2, 4: 4, 6: 6, 8: 4}
_GFX950_H64_DECODE_TIERS = {1: 1, 2: 2, 4: 4, 6: 2, 8: 4}
# 32 heads (V4.1 full layers): (rows_per_wave, waves, scale_tdm), a sequence's rows a CTA.
_GFX1250_H32_DECODE_TIERS = {
    1: (1, 1, 0),
    2: (1, 2, 1),
    3: (3, 1, 1),
    4: (2, 2, 1),
    6: (3, 2, 1),
    8: (4, 2, 1),
}
# V4.1 candidate layers switch to the shallower ring at prefill size (ATOM's 16384-token batch).
_CANDIDATE_PREFILL_ROWS = 16384
# The Prefill split, from static shapes only. Full layers: PREFILL_ROUNDS rounds of resident CTAs
# so the dispatcher balances the tail, no piece under PREFILL_MIN_WALK tiles. Candidate rows: the
# fewest rounds that keep every CTA's walk within CAND_MAX_WALK tiles.
_PREFILL_ROUNDS = 8
_PREFILL_MIN_WALK = 16
_CAND_MAX_WALK = 32
_MAX_PREFILL_PARTS = 65536  # the kernels' 32-bit split (part_point) needs parts <= 2^16

# Per arch: the ndim of q_scale / kv_scale / kv_cache, and kv_cache's page-rows dim. gfx950's are
# MFMA-permuted (4-D, kv_cache [pages, 4, page_size, 16]); gfx1250's natural (3-D).
_LAYOUT = {GFX950: (4, 2), GFX1250: (3, 1)}
# Arches whose KV layout is defined for one page size only: gfx950 permutes a 64-row page at a time.
_PAGE_SIZE = {GFX950: 64}
# Requests an arch serves through another op, not this one.
_ELSEWHERE = {
    (GFX950, 32): "V4.1 (heads=32) on gfx950 runs FlyDSL's row-group kernel "
    "(aiter.ops.flydsl.flydsl_pa_mqa_logits_fp4), not this op",
}


# == JIT stubs: signatures must match MQA_LOGITS_MXFP4_PYBIND exactly ============================
@compile_ops(_MD_NAME, fc_name="mqa_logits_mxfp4_instances", develop=True)
def _instances_raw(device_id: int, out: torch.Tensor) -> int: ...


@compile_ops(_MD_NAME, fc_name="mqa_logits_mxfp4_build_schedule", develop=True)
def _build_schedule_raw(
    query_start_loc: torch.Tensor,
    row_ends: torch.Tensor,
    cu_groups: torch.Tensor,
    cta_info: torch.Tensor,
    instance: int,
    num_ctas: int,
    cta_resident: int,
) -> None: ...


# The tensors lead: compile_ops gives an op whose first argument is not a tensor a dummy one,
# allocated every call.
@compile_ops(_MD_NAME, fc_name="mqa_logits_mxfp4_launch", develop=True)
def _launch_raw(
    q: torch.Tensor,
    q_scale: torch.Tensor,
    kv_cache: torch.Tensor,
    kv_scale: torch.Tensor,
    weights: torch.Tensor,
    block_tables: torch.Tensor,
    query_start_loc: torch.Tensor,
    row_ends: torch.Tensor,
    cta_info: torch.Tensor,
    out: torch.Tensor,
    out_offsets: torch.Tensor,
    instance: int,
    mode: int,
    weight_scale: float,
    tpb_shift: int,
    cta_target: int,
    prefill_parts: int,
    grid: int,
) -> None: ...


# == the registry ===================================================================================
@dataclass(frozen=True)
class _Instance:
    """One compiled kernel instance, as its registry entry describes it."""

    id: int
    heads: int
    rows_per_wave: int
    waves: int
    stages: int
    scale_tdm: int  # 1: the key scales are copied with the tile, 0: loaded per lane
    gather: int  # 1: a 64-key tile gathers 8-row table entries (V4.1 candidate layers)
    entry_rows_min: int  # rows a table entry may name: page_size * pages_per_block
    entry_rows_max: int
    tile_keys: int
    modes: int  # bit m: schedule mode m is compiled
    flat_out: int  # reads out_offsets
    resident: tuple[int, int, int]  # CTAs resident at once, a mode (0: not compiled)

    @property
    def rows_per_cta(self) -> int:
        return self.rows_per_wave * self.waves

    @property
    def name(self) -> str:
        g = "g" if self.gather else ""
        s = "T" if self.scale_tdm else "L"
        shape = f"r{self.rows_per_wave}w{self.waves}s{self.stages}"
        return f"h{self.heads}{g}_{shape}{s}"

    def has(self, mode: int) -> bool:
        return bool(self.modes >> mode & 1)

    def reads(self, entry_rows: int) -> bool:
        """Whether it reads a table whose entries name `entry_rows` plane rows."""
        if not self.entry_rows_min <= entry_rows <= self.entry_rows_max:
            return False
        if self.gather:
            return True
        # contiguous tiles: an entry is a power-of-two run of whole tiles, or a tile a
        # power-of-two run of whole entries (the registry's entry_rows bounds say which)
        big, small = max(entry_rows, self.tile_keys), min(entry_rows, self.tile_keys)
        runs = big // small
        return big % small == 0 and runs & (runs - 1) == 0


@functools.cache
def _instances(device: torch.device) -> tuple[_Instance, ...]:
    """The instances compiled for `device`'s arch, with their resident CTAs a mode."""
    buf = torch.empty(1024, dtype=torch.int32, device="cpu")
    need = _instances_raw(device.index, buf)
    if need > buf.numel():
        buf = torch.empty(need, dtype=torch.int32, device="cpu")
        _instances_raw(device.index, buf)
    flat = buf[:need].tolist()
    n, rows = flat[0], flat[1:]
    out = []
    for i in range(0, len(rows), n):
        f = rows[i : i + n]
        out.append(_Instance(*f[:12], resident=(f[12], f[13], f[14])))
    return tuple(out)


def _describe(insts) -> str:
    return (
        ", ".join(
            f"{i.name} (heads {i.heads}, entry rows {i.entry_rows_min}"
            + (f"..{i.entry_rows_max}" if i.entry_rows_max != i.entry_rows_min else "")
            + ")"
            for i in insts
        )
        or "none"
    )


def _tier(tiers: dict, max_qlen: int):
    return tiers[min(t for t in tiers if t >= max_qlen)]


def _rule_gfx1250(heads: int, entry_rows: int, max_rows: int, max_qlen: int):
    """V4 decode by tier, prefill four rows a CTA; V4.1 full layers decode by tier, prefill
    eight rows a CTA; V4.1 candidate layers one row a CTA, a deeper ring below prefill size.
    """
    decode = max_qlen <= _DECODE_MAX_QLEN
    if heads == 64:
        rpc = _tier(_GFX1250_H64_DECODE_TIERS, max_qlen) if decode else 4
        want = {"rows_per_cta": rpc, "stages": 2, "scale_tdm": 1, "gather": 0}
        return want, FREE if decode else PREFILL
    if heads == 32 and entry_rows >= _TILE:
        rpw, waves, tdm = (
            _tier(_GFX1250_H32_DECODE_TIERS, max_qlen) if decode else (4, 2, 1)
        )
        want = {
            "rows_per_wave": rpw,
            "waves": waves,
            "stages": 2,
            "scale_tdm": tdm,
            "gather": 0,
        }
        return want, FREE if decode else PREFILL
    if heads == 32:
        stages = 3 if max_rows >= _CANDIDATE_PREFILL_ROWS else 4
        want = {"rows_per_cta": 1, "stages": stages, "scale_tdm": 0, "gather": 1}
        return want, PREFILL
    return None


def _rule_gfx950(heads: int, entry_rows: int, max_rows: int, max_qlen: int):
    """V4 decode by tier (qshare above one row a sequence), prefill one row a CTA. V4.1 runs
    FlyDSL's kernel (_ELSEWHERE)."""
    if heads != 64:
        return None
    decode = max_qlen <= _DECODE_MAX_QLEN
    rpc = _tier(_GFX950_H64_DECODE_TIERS, max_qlen) if decode else 1
    want = {"rows_per_cta": rpc, "tile_keys": _TILE}
    return want, FREE if decode else PREFILL


# Each arch's selection: (heads, entry_rows, max_rows, max_qlen) -> (wanted attributes, the
# schedule-free mode), or None when no rule covers the request. Mixed batches run the prefill
# choice.
_RULES = {GFX1250: _rule_gfx1250, GFX950: _rule_gfx950}


def _matches(inst: _Instance, want: dict) -> bool:
    return all(getattr(inst, k) == v for k, v in want.items())


# == the config =====================================================================================
@dataclass(frozen=True, eq=False)
class MqaLogitsScheduleBuffers:
    """One live schedule's device memory, from :meth:`MqaLogitsConfig.schedule_buffers`."""

    cfg: MqaLogitsConfig
    cta_info: torch.Tensor  # [num_ctas + scratch, 8] int32 records
    cu_groups: (
        torch.Tensor
    )  # the group cut, then each group's sequence; empty at one row a CTA
    num_ctas: int  # the Table grid; pinned once a graph captures it


@dataclass(frozen=True, eq=False)
class MqaLogitsSchedule:
    """One forward's Table schedule, bound to its cfg and to the very `query_start_loc` and
    `row_ends` tensors it was built from (the launch checks identity, not contents)."""

    cfg: MqaLogitsConfig
    query_start_loc: torch.Tensor
    row_ends: torch.Tensor
    cta_info: torch.Tensor
    num_ctas: int


@dataclass(frozen=True, eq=False)
class MqaLogitsConfig:
    """An opaque launch configuration for one shape bucket (:func:`mqa_logits_mxfp4_config`).

    Public, read-only: ``block_table_width`` (the minimum ``block_tables.shape[1]``), ``arch``,
    ``heads``, ``page_size``, ``pages_per_block`` and the four bounds.
    """

    device: torch.device
    arch: str
    heads: int
    page_size: int
    pages_per_block: int
    max_rows: int
    max_seqs: int
    max_qlen: int
    max_seq_len: int
    block_table_width: int
    _instance: _Instance
    # The schedule-free mode, or None: no schedule means an internal one.
    _mode: int | None
    _tpb_shift: int  # log2(tiles a table entry); 0 where a tile spans entries
    _table_slots: int  # the Table grid at the bounds
    _empty_i32: torch.Tensor
    _empty_i64: torch.Tensor
    _internal: list  # [MqaLogitsScheduleBuffers], built on first need

    def schedule_buffers(self) -> MqaLogitsScheduleBuffers:
        """Buffers for one live schedule, sized from the bounds. Hold one set per micro-batch or
        captured graph and pass it to every forward's :func:`mqa_logits_mxfp4_schedule`.
        """
        rpc = self._instance.rows_per_cta
        groups = _max_groups(self.max_rows, self.max_seqs, rpc)
        return MqaLogitsScheduleBuffers(
            cfg=self,
            cta_info=torch.empty(
                (self._table_slots + _SCHED_SCRATCH_RECORDS, _SCHED_RECORD_INTS),
                dtype=torch.int32,
                device=self.device,
            ),
            cu_groups=(
                self._empty_i32
                if rpc == 1
                else torch.empty(
                    2 * (groups + 1), dtype=torch.int32, device=self.device
                )
            ),
            num_ctas=self._table_slots,
        )

    @property
    def instance_name(self) -> str:
        """The chosen instance, for logs only: nothing takes it back."""
        return self._instance.name


def _max_groups(rows: int, seqs: int, rows_per_cta: int) -> int:
    """Groups of at most `rows_per_cta` rows of one sequence that `rows` rows in `seqs`
    sequences cut into, at most (each sequence's last group may be short)."""
    if rows_per_cta == 1:
        return rows
    return (rows + seqs * (rows_per_cta - 1)) // rows_per_cta


def _resolve_instance(insts, override) -> _Instance:
    for i in insts:
        if override in (i.id, i.name):
            return i
    raise ValueError(
        f"no MXFP4 MQA-logits instance {override!r} on this device; it has {_describe(insts)}"
    )


def mqa_logits_mxfp4_config(
    device: torch.device | str | int,
    *,
    heads: int,
    page_size: int,
    pages_per_block: int = 1,
    max_rows: int,
    max_seqs: int,
    max_qlen: int,
    max_seq_len: int,
    _instance: int | str | None = None,
    _mode: str | None = None,
) -> MqaLogitsConfig:
    """The launch configuration for a shape bucket: static bounds only, no device data read.

    ``heads``            index heads: 64 (V4) or 32 (V4.1).
    ``page_size``        rows of one KV page (``kv_cache.shape[1]``): 64 (V4) or 8 (V4.1).
    ``pages_per_block``  consecutive pages a table entry names (V4.1 full layers: 16 at ratio 2,
                         32 at ratio 1; candidate layers and V4: 1).
    ``max_rows``         query rows of a call, at most (CUDAGraph padding included).
    ``max_seqs``         ``len(query_start_loc) - 1``, at most.
    ``max_qlen``         rows of one sequence, at most; a decode bucket passes its MTP depth + 1.
    ``max_seq_len``      the logits width: the most keys a row sees.

    Cached: equal arguments give the same object. Building one asks HIP for each instance's
    occupancy once a device. Raises if no instance compiled for the device's arch supports the
    request. ``_instance`` (an id or name) and ``_mode`` ("table" / "free" / "prefill") override
    the choice, for tuning and tests.
    """
    dev = (
        torch.device(device)
        if not isinstance(device, int)
        else torch.device("cuda", device)
    )
    if dev.index is None:
        dev = torch.device(dev.type, torch.cuda.current_device())
    return _config(
        dev,
        int(heads),
        int(page_size),
        int(pages_per_block),
        int(max_rows),
        int(max_seqs),
        int(max_qlen),
        int(max_seq_len),
        _instance,
        _mode,
    )


@functools.cache
def _config(
    dev,
    heads,
    page_size,
    pages_per_block,
    max_rows,
    max_seqs,
    max_qlen,
    max_seq_len,
    override,
    mode_name,
):
    bounds = (
        heads,
        page_size,
        pages_per_block,
        max_rows,
        max_seqs,
        max_qlen,
        max_seq_len,
    )
    if min(bounds) < 1:
        raise ValueError("every bound must be >= 1")
    arch = _device_arch(dev)
    if (arch, heads) in _ELSEWHERE:
        raise ValueError(_ELSEWHERE[arch, heads])
    if _PAGE_SIZE.get(arch, page_size) != page_size:
        raise ValueError(
            f"{arch}'s KV layout is permuted a {_PAGE_SIZE[arch]}-row page at a time; "
            f"page_size={page_size}"
        )
    insts = _instances(dev)
    entry_rows = page_size * pages_per_block
    supported = [i for i in insts if i.heads == heads and i.reads(entry_rows)]
    rule_of = _RULES.get(arch)
    rule = rule_of(heads, entry_rows, max_rows, max_qlen) if rule_of else None
    if override is not None:
        inst = _resolve_instance(insts, override)
        if inst not in supported:
            raise ValueError(
                f"instance {inst.name} does not take heads={heads} with {entry_rows}-row table "
                f"entries; {arch} has {_describe(insts)}"
            )
        free_mode = rule[1] if rule is not None else None
    else:
        chosen = [i for i in supported if rule is not None and _matches(i, rule[0])]
        if not chosen:
            raise ValueError(
                f"no MXFP4 MQA-logits instance on {arch} for heads={heads}, "
                f"page_size={page_size} x pages_per_block={pages_per_block}, "
                f"max_qlen={max_qlen}; {arch} has {_describe(insts)}"
            )
        inst, free_mode = chosen[0], rule[1]
    if mode_name is not None:
        mode = _MODE_NAMES[mode_name]
        if not inst.has(mode):
            raise ValueError(f"instance {inst.name} has no {mode_name} mode")
        free_mode = None if mode == TABLE else mode
    elif free_mode is not None and not inst.has(free_mode):
        free_mode = None
    if (
        inst.rows_per_cta > 1
        and max_seqs > _GROUPS_BUILD_MAX_SEQS
        and free_mode is None
    ):
        raise ValueError(
            f"the Table schedule cuts at most {_GROUPS_BUILD_MAX_SEQS} sequences into "
            f"{inst.rows_per_cta}-row tiles; max_seqs={max_seqs}"
        )

    tiles = -(-max_seq_len // inst.tile_keys)
    width = -(-tiles * inst.tile_keys // entry_rows)
    tpb = max(1, entry_rows // inst.tile_keys)
    resident = inst.resident[TABLE]
    slots = max(_max_groups(max_rows, max_seqs, inst.rows_per_cta), resident)
    slots = -(-slots // resident) * resident
    return MqaLogitsConfig(
        device=dev,
        arch=arch,
        heads=heads,
        page_size=page_size,
        pages_per_block=pages_per_block,
        max_rows=max_rows,
        max_seqs=max_seqs,
        max_qlen=max_qlen,
        max_seq_len=max_seq_len,
        block_table_width=width,
        _instance=inst,
        _mode=free_mode,
        _tpb_shift=0 if inst.gather else tpb.bit_length() - 1,
        _table_slots=slots,
        _empty_i32=torch.empty(0, dtype=torch.int32, device=dev),
        _empty_i64=torch.empty(0, dtype=torch.int64, device=dev),
        _internal=[],
    )


# == the schedule ===================================================================================
def _check_rows(cfg: MqaLogitsConfig, query_start_loc, row_ends) -> tuple[int, int]:
    for name, t in (("query_start_loc", query_start_loc), ("row_ends", row_ends)):
        if t.dtype != torch.int32 or t.dim() != 1 or not t.is_contiguous():
            raise ValueError(f"{name} must be a contiguous 1-D int32 tensor")
        if t.device != cfg.device:
            raise ValueError(f"{name} is on {t.device}; the config is for {cfg.device}")
    seqs, rows = query_start_loc.numel() - 1, row_ends.numel()
    if not 1 <= seqs <= cfg.max_seqs or rows > cfg.max_rows:
        raise ValueError(
            f"{rows} rows in {seqs} sequences exceed the config's bounds "
            f"(max_rows={cfg.max_rows}, max_seqs={cfg.max_seqs})"
        )
    return rows, seqs


def _build(cfg, buf, query_start_loc, row_ends) -> None:
    inst = cfg._instance
    _build_schedule_raw(
        query_start_loc,
        row_ends,
        buf.cu_groups,
        buf.cta_info,
        inst.id,
        buf.num_ctas,
        inst.resident[TABLE],
    )


def mqa_logits_mxfp4_schedule(
    cfg: MqaLogitsConfig,
    query_start_loc: torch.Tensor,
    row_ends: torch.Tensor,
    *,
    buffers: MqaLogitsScheduleBuffers | None = None,
) -> MqaLogitsSchedule:
    """Build one forward's Table schedule on device (no host readback; capturable).

    Optional: only a caller reusing it across many calls (V4's many layers) gains. ``buffers``
    from ``cfg.schedule_buffers()`` hold it; ``None`` allocates a fresh set (eager use only: a
    replayed graph reads the captured pointers). One buffer set holds one live schedule.
    """
    _check_rows(cfg, query_start_loc, row_ends)
    if buffers is None:
        buffers = cfg.schedule_buffers()
    elif buffers.cfg is not cfg:
        raise ValueError("these buffers were sized for another config")
    _build(cfg, buffers, query_start_loc, row_ends)
    return MqaLogitsSchedule(
        cfg=cfg,
        query_start_loc=query_start_loc,
        row_ends=row_ends,
        cta_info=buffers.cta_info,
        num_ctas=buffers.num_ctas,
    )


# == the launch =====================================================================================
def _prefill_parts(inst: _Instance, groups: int, width_tiles: int) -> int:
    """Pieces a group is cut into in the Prefill schedule, from static shapes only (the logits
    width bounds every window), so a captured graph keeps its grid. The grid is a whole number of
    rounds of resident CTAs: a part-filled last round costs a whole one."""
    resident = inst.resident[PREFILL]
    groups = max(1, groups)
    if inst.gather:
        rounds = max(1, -(-groups * width_tiles // (_CAND_MAX_WALK * resident)))
        return max(1, min(rounds * resident // groups, _MAX_PREFILL_PARTS))
    return max(
        1,
        min(
            _PREFILL_ROUNDS * resident // groups,
            width_tiles // _PREFILL_MIN_WALK,
            _MAX_PREFILL_PARTS,
        ),
    )


def mqa_logits_mxfp4(
    q: torch.Tensor,
    q_scale: torch.Tensor,
    kv_cache: torch.Tensor,
    kv_scale: torch.Tensor,
    weights: torch.Tensor,
    block_tables: torch.Tensor,
    query_start_loc: torch.Tensor,
    row_ends: torch.Tensor,
    *,
    cfg: MqaLogitsConfig,
    schedule: MqaLogitsSchedule | None = None,
    weight_scale: float = 1.0,
    out: torch.Tensor | None = None,
    out_offsets: torch.Tensor | None = None,
) -> torch.Tensor:
    """Paged MQA logits of ``q``'s rows (see the module docstring); returns ``out``.

    ``out=None`` allocates ``[rows, max_seq_len]`` fp32 filled with ``-inf`` (pass one to avoid a
    per-call allocation; pre-fill it only if something reads past the windows). With
    ``out_offsets`` [rows] int64, ``out`` is flat and row r's logits start at ``out_offsets[r]``.
    ``schedule`` must come from :func:`mqa_logits_mxfp4_schedule` with this ``cfg`` and these
    very ``query_start_loc`` / ``row_ends`` tensors.
    """
    inst = cfg._instance
    if q.device != cfg.device:
        raise ValueError(f"q is on {q.device}; the config is for {cfg.device}")
    want, page_dim = _LAYOUT[cfg.arch]
    for name, t in (
        ("q_scale", q_scale),
        ("kv_scale", kv_scale),
        ("kv_cache", kv_cache),
    ):
        if t.dim() != want:
            other = [a for a, (n, _) in _LAYOUT.items() if n == t.dim()]
            why = (
                f"; {t.dim()}-D is {other[0]}'s layout, which has the SAME byte count -- the "
                "launch would accept it and return wrong logits"
                if other
                else ""
            )
            raise ValueError(
                f"{cfg.arch}: {name} must be {want}-D, got {tuple(t.shape)}{why}"
            )
    if q.dim() != 3 or q.shape[1] != cfg.heads:
        raise ValueError(f"q must be [rows, {cfg.heads}, D/2], got {tuple(q.shape)}")
    if kv_cache.shape[page_dim] != cfg.page_size:
        raise ValueError(
            f"kv_cache pages hold {kv_cache.shape[page_dim]} rows; the config is for "
            f"{cfg.page_size}"
        )
    if block_tables.dim() != 2 or block_tables.shape[1] < cfg.block_table_width:
        raise ValueError(
            f"block_tables must be [seqs, >= {cfg.block_table_width}] (cfg.block_table_width), "
            f"got {tuple(block_tables.shape)}"
        )
    if weights.dtype != torch.bfloat16:
        raise ValueError(f"weights must be bf16, got {weights.dtype}")
    rows, seqs = _check_rows(cfg, query_start_loc, row_ends)
    if q.shape[0] != rows:
        raise ValueError(f"q has {q.shape[0]} rows and row_ends {rows}")

    if out_offsets is not None:
        if out is None:
            raise ValueError("a flat out (out_offsets) must be passed in")
        if out_offsets.dtype != torch.int64:
            raise ValueError(f"out_offsets must be int64, got {out_offsets.dtype}")
        if not inst.flat_out:
            raise ValueError(
                f"instance {inst.name} does not write a flat out (out_offsets)"
            )
    elif out is None:
        out = torch.full(
            (rows, cfg.max_seq_len), float("-inf"), dtype=torch.float32, device=q.device
        )
    elif out.dim() != 2 or out.shape[0] < rows or out.shape[1] < cfg.max_seq_len:
        # The kernels bound a row's stores by its row_ends only.
        raise ValueError(
            f"out must be [>= {rows}, >= {cfg.max_seq_len}] (rows, cfg.max_seq_len), "
            f"got {tuple(out.shape)}"
        )
    if out.dtype != torch.float32:
        raise ValueError(f"out must be fp32, got {out.dtype}")

    cta_target, parts = 1, 1
    if schedule is not None:
        if (
            schedule.cfg is not cfg
            or schedule.query_start_loc is not query_start_loc
            or schedule.row_ends is not row_ends
        ):
            raise ValueError(
                "the schedule was built for another config or other query_start_loc / row_ends "
                "tensors; a schedule is bound to the very tensors it was built from"
            )
        mode, cta_info, grid = TABLE, schedule.cta_info, schedule.num_ctas
    elif cfg._mode is not None:
        mode, cta_info = cfg._mode, cfg._empty_i32
        groups = _max_groups(rows, seqs, inst.rows_per_cta)
        if mode == FREE:
            cta_target = inst.resident[FREE]
            grid = groups + cta_target
        else:
            parts = _prefill_parts(inst, groups, -(-cfg.max_seq_len // inst.tile_keys))
            grid = groups * parts
    else:
        if not cfg._internal:
            cfg._internal.append(cfg.schedule_buffers())
        buf = cfg._internal[0]
        _build(cfg, buf, query_start_loc, row_ends)
        mode, cta_info, grid = TABLE, buf.cta_info, buf.num_ctas

    _launch_raw(
        q,
        q_scale,
        kv_cache,
        kv_scale,
        weights,
        block_tables,
        query_start_loc,
        row_ends,
        cta_info,
        out,
        cfg._empty_i64 if out_offsets is None else out_offsets,
        inst.id,
        mode,
        float(weight_scale),
        cfg._tpb_shift,
        cta_target,
        parts,
        int(grid),
    )
    return out


__all__ = [
    "MqaLogitsConfig",
    "MqaLogitsSchedule",
    "MqaLogitsScheduleBuffers",
    "mqa_logits_mxfp4",
    "mqa_logits_mxfp4_config",
    "mqa_logits_mxfp4_schedule",
]
