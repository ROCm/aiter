# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Flag protocol and fanout helpers that the flag-based collectives share.

The mesh and ring kernels stage a packet in LDS, store it to one or more peer
inboxes, colour a flag behind it, and spin on the flags their peers colour in
their own inbox. This module holds those steps: the fanout from LDS, the
publish, the wait and the colour sequence. The primitives under them, and the
cache policy that decides their fences, are in ``transport``.

The helpers that branch or spin on traced values are ``@flyc.jit``: a kernel
body gets the AST rewrite, but the functions it calls do not. Inside a trace,
``@flyc.jit`` runs the rewritten function in place.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl

from .transport import (
    FLAG_I32_PER_LANE,
    FLAG_LANES,
    QUAD_LANES,
    _acquire_inbox,
    _i32_to_bytes,
    _load_flag,
    _release_inbox,
    _store_flag_peer,
    _store_v4i32_peer,
)


def stripes(n_sectors: int, n_dest: int, block: int) -> list[tuple[int, int]]:
    """``(first sector, width)`` stripes of a striped multi-destination fanout.

    One quad per ``(destination, sector)`` of a stripe, in a single pass, so a
    stripe is at most ``quads / n_dest`` sectors wide, and never wider than 8.
    """
    width = min(8, (int(block) // QUAD_LANES) // int(n_dest))
    return [(b, min(width, n_sectors - b)) for b in range(0, n_sectors, width)]


def fanout_quads(block: int, n_dest: int, n_sectors: int) -> tuple[int, int]:
    """``(quads the block has, quads a full-width striped fanout needs)``.

    A block with fewer quads than that would silently drop the sectors past
    the end.
    """
    full = [(b, min(8, n_sectors - b)) for b in range(0, n_sectors, 8)]
    return int(block) // QUAD_LANES, max(int(n_dest) * w for _, w in full)


def select_peer_base(peers, dests, j):
    """Inbox base of ``dests[j]`` for a lane-varying *j*."""
    base = peers[dests[0]]
    for i in range_constexpr(1, len(dests)):
        base = (j == fx.Int32(i)).select(peers[dests[i]], base)
    return base


@flyc.jit
def next_color(color, *, parity_safe: bool = False):
    """The colour after *color*, skipping 0, the unset sentinel.

    A schedule that double-buffers its inbox on ``color & 1`` must resume at 2,
    not 1: the colour before the wrap is -1, which is odd, so resuming at 1
    would put two consecutive tiles in the same slot, and a rank one tile ahead
    would overwrite data a peer is still reading.
    """
    color = color + fx.Int32(1)
    if color == fx.Int32(0):
        color = fx.Int32(2 if parity_safe else 1)
    return color


@flyc.jit
def lds_write_packet(codec, pack, row, words, scale_word, is_leader, tid, scale_slot):
    """Stage one codec packet into LDS row *row* of *pack*.

    Each payload word is stored by the thread that owns it (``plane_slots``);
    the group-16 scale word by the group leader only.
    """
    for (off, pred), word in zip(codec.plane_slots(tid), words):
        if pred:
            fx.memref_store(word, pack, (row, off))
    if const_expr(codec.has_scale):  # noqa: SIM102
        if is_leader:
            fx.memref_store(
                scale_word, pack, (row, fx.Int32(codec.scale_i32_off) + scale_slot)
            )


@flyc.jit
def fanout_sectors(
    *,
    codec,
    n_dest,
    n_atoms,
    stripes,
    sector_fastest,
    quad_id,
    lane_in_quad,
    smem_ptr,
    pack_layout,
    slot_i32,
    dest_base,
    payload_policy,
):
    """Store *n_atoms* staged rank-tiles per destination from LDS to *n_dest*
    peers, one 64 B sector per quad.

    Pack row ``j * n_atoms + k`` holds rank-tile *k* for destination *j*; it
    lands at ``slot_i32() + k * rank_tile_i32`` in ``dest_base(j)``. *slot_i32*
    is a callable rather than a value so that the offset is computed at its use,
    inside the store's guard. Lockstep stripes of up to 8 sectors
    (:func:`stripes`) cover a rank-tile: at the
    default block INT4 is 8+8+2 (16 nibble sectors then the 2-sector E4M3
    tail), fp16 is eight full stripes.

    One quad per (destination, sector) of a stripe; leftover quads sit idle.
    Which axis runs fastest across consecutive quads is a fabric question:

    - "sector": consecutive quads target consecutive peers of one sector, so a
      single store instruction hits every GPU -- ideal on xGMI, whose native
      packet is exactly the 64 B a quad writes.
    - "peer": consecutive quads walk the sectors of one peer, giving each
      destination a ``64 * width`` B contiguous run -- ideal on PCIe.
    """
    # 64 B NT sectors of one rank-tile: (sector, lane-in-quad) -> i32 start of
    # the dwordx4.
    nt_own_layout = fx.make_layout((codec.n_sectors, QUAD_LANES), (16, 4))
    for k in range_constexpr(n_atoms):
        for sector_base, width in stripes:
            n_quads = fx.Int32(n_dest * width)
            safe = (quad_id < n_quads).select(quad_id, fx.Int32(0))
            if const_expr(sector_fastest):
                j = safe % fx.Int32(n_dest)
                sector_in_stripe = safe // fx.Int32(n_dest)
            else:
                j = safe // fx.Int32(width)
                sector_in_stripe = safe % fx.Int32(width)
            sector = fx.Int32(sector_base) + sector_in_stripe
            if quad_id < n_quads:
                vec_idx = fx.get_scalar(
                    fx.crd2idx((sector, lane_in_quad), nt_own_layout)
                )
                pack_row = j
                wire_idx = vec_idx
                if const_expr(n_atoms != 1):
                    pack_row = j * fx.Int32(n_atoms) + fx.Int32(k)
                    wire_idx = vec_idx + fx.Int32(k * codec.rank_tile_i32)
                # 4xi32 NT vector cannot go through the i32 pack view.
                v4 = fx.ptr_load(
                    smem_ptr
                    + fx.get_scalar(fx.crd2idx((pack_row, vec_idx), pack_layout)),
                    result_type=fx.Vector.make_type(4, fx.Int32),
                )
                byte_off = _i32_to_bytes(slot_i32() + wire_idx)
                _store_v4i32_peer(dest_base(j) + byte_off, v4, payload_policy)


@flyc.jit
def fanout_contiguous(
    *,
    codec,
    n_atoms,
    quad_id,
    lane_in_quad,
    quads_per_block,
    smem_ptr,
    slot_i32,
    dest_base,
    payload_policy,
):
    """Store *n_atoms* staged rank-tiles from LDS into one peer, in address
    order, so the whole ``n_atoms * rank_tile B`` lands as one contiguous run.

    Quads past the sector count do not store; ``safe`` keeps their address
    arithmetic in range. Flat sector id -> (rank-tile, sector) -> i32 offset by
    arithmetic rather than a layout: at TP8 *n_atoms* is 1, and a unit mode in
    a layout does not survive coalescing intact. The same offset addresses LDS
    and the wire, because the LDS rows and the wire rank-tiles share a row
    stride. *slot_i32* is called at each store, as in :func:`fanout_sectors`.
    """
    n_sectors = codec.n_sectors
    n_total = n_atoms * n_sectors
    for rnd in range_constexpr(-(-n_total // quads_per_block)):
        s = quad_id + fx.Int32(rnd * quads_per_block)
        in_range = s < fx.Int32(n_total)
        safe = in_range.select(s, fx.Int32(0))
        j = safe // fx.Int32(n_sectors)
        sector = safe % fx.Int32(n_sectors)
        if s < fx.Int32(n_total):
            flat = (
                j * fx.Int32(codec.rank_tile_i32)
                + sector * fx.Int32(16)
                + lane_in_quad * fx.Int32(4)
            )
            v4 = fx.ptr_load(
                smem_ptr + flat,
                result_type=fx.Vector.make_type(4, fx.Int32),
            )
            byte_off = _i32_to_bytes(slot_i32() + flat)
            _store_v4i32_peer(dest_base + byte_off, v4, payload_policy)


def _drain_and_release(release_scope):
    """Retire this wave's payload stores, join the workgroup, and release."""
    rocdl.s_waitcnt(vmcnt=0)
    gpu.barrier()
    if const_expr(release_scope is not None):
        _release_inbox(release_scope)


@flyc.jit
def publish_flags(
    *, tid, n_dest, dest_base, flag_i32, color, flag_policy, release_scope
):
    """Drain the payload stores, then write *color* into *n_dest* peers' flags.

    ``vmcnt(0)`` retires this wave's payload stores; the barrier joins the
    other waves, whose ``vmcnt`` is separate, so a wave-0 flag cannot race
    stores still in flight. On a cacheable inbox retiring is not enough -- the
    lines can sit in this XCD's L2 -- so the release fence writes them back
    before the flag goes out. Every workgroup issues its own: L2 is per-XCD.

    ``FLAG_LANES`` lanes per destination, 8 B each, so the 64 B sector at
    ``flag_i32()`` (the same offset in every destination's inbox) is one store
    instruction from wave 0. ``dest_base(j)`` maps a lane-varying destination
    index to an inbox base, e.g. :func:`select_peer_base`. *flag_i32* is
    called inside the guard, as in :func:`fanout_sectors`.
    """
    _drain_and_release(release_scope)
    limit = fx.Int32(n_dest)
    dest = tid // fx.Int32(FLAG_LANES)
    safe = (dest < limit).select(dest, fx.Int32(0))
    if dest < limit:
        elem = flag_i32() + (tid % fx.Int32(FLAG_LANES)) * fx.Int32(FLAG_I32_PER_LANE)
        _store_flag_peer(dest_base(safe) + _i32_to_bytes(elem), color, flag_policy)


@flyc.jit
def publish_flag(*, tid, dest_base, flag_i32, color, flag_policy, release_scope):
    """:func:`publish_flags` for a schedule with one destination at
    *dest_base*: the first ``FLAG_LANES`` lanes write the flag, and no lane
    needs a destination index.
    """
    _drain_and_release(release_scope)
    if tid < fx.Int32(FLAG_LANES):
        elem = flag_i32() + tid * fx.Int32(FLAG_I32_PER_LANE)
        _store_flag_peer(dest_base + _i32_to_bytes(elem), color, flag_policy)


@flyc.jit
def wait_flags(
    *, tid, n_src, skip_rank, flag_addr, color, acquire_scope, writeback_scope=None
):
    """Spin until *n_src* sources have coloured their flags with *color*.

    Lane ``t`` watches source ``t``; with *skip_rank* set, the lanes step over
    that rank, whose flag is never published. ``flag_addr(src)`` is the flag's
    address in our own inbox. The poll is ``sc0 sc1`` (``_load_flag``), so no
    retry is served from a stale line and the loop body needs no fence.

    The fences are after the join and unconditional: only the spinning lanes
    ran the loop, so an acquire scoped to it would leave the other waves
    reading the payload unordered after it, and would be skipped entirely when
    the flag is already set on the first read. *acquire_scope* is the inbox
    policy's. Pass *writeback_scope* when this block may hold dirty output
    lines and the acquire is a system-scope invalidate: the stores are drained
    and written back first, or the invalidate silently discards them.
    """
    spin_src = tid
    if const_expr(skip_rank is not None):  # noqa: SIM102
        if tid >= fx.Int32(skip_rank):
            spin_src = tid + fx.Int32(1)
    if tid < fx.Int32(n_src):
        flag = flag_addr(spin_src)
        current = _load_flag(flag)
        while current != color:
            current = _load_flag(flag)
    _join_and_acquire(acquire_scope, writeback_scope)


@flyc.jit
def wait_flag(*, tid, flag_addr, color, acquire_scope, writeback_scope=None):
    """:func:`wait_flags` for a schedule with one source: lane 0 spins on
    ``flag_addr()``, and the fences are the same.
    """
    if tid == fx.Int32(0):
        flag = flag_addr()
        current = _load_flag(flag)
        while current != color:
            current = _load_flag(flag)
    _join_and_acquire(acquire_scope, writeback_scope)


def _join_and_acquire(acquire_scope, writeback_scope):
    """Join the workgroup after a spin, then write back (if asked) and acquire."""
    gpu.barrier()
    if const_expr(writeback_scope is not None):
        rocdl.s_waitcnt(vmcnt=0)
        _release_inbox(writeback_scope)
    _acquire_inbox(acquire_scope)
