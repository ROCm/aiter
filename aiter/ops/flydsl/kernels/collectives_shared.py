# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Transport assets shared by the IPC collectives: the one-shot, mesh and ring
all-reduces and the mesh and ring all-to-alls.

Tile geometry, the inbox cache-policy table, the peer store/load primitives and
the LDS staging factory.

Note: Editing the shared modules doesn't invalidate the FlyDSL compiler cache.
Hence, one may end up running stale kernels unless one sets
export FLYDSL_EXTRA_SOURCE_DIRS=$PWD/aiter/ops/flydsl/kernels at the repo root.
"""

import logging

import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T, as_ir_value

from .communication_ops_utils import traced

logger = logging.getLogger("aiter")

WORLD = 8  # Default value for world size
SUPPORTED_WORLDS = (2, 4, 8)
BLOCK = 256
ATOMS = 8
TILE_BYTES = BLOCK * ATOMS * 16
DEFAULT_GRID_CAP = 304 * 4
WAVE = 64
WAVES = BLOCK // WAVE
# 4 lanes × 16 B = one 64 B NT sector. Not world_size.
QUAD_LANES = 4
QUADS_PER_WAVE = WAVE // QUAD_LANES
# Wire/inbox addresses are byte pointers; tile math is in i32 slots.
I32_BYTES = 4
# The 64 B handshake sector is written as 8 lanes × 8 B: one colour-bearing
# store per destination. 8 B per lane because it is the widest store the
# backend performs atomically.
FLAG_LANES = 8
FLAG_I32_PER_LANE = 2

# Buffer aux bits on gfx942/gfx950. This is LLVM's CPol encoding, which the
# backend renames for CDNA: bit 0 (GLC) prints as `sc0`, bit 1 (SLC) as `nt`,
# bit 4 (SCC) as `sc1`. Bit 2 exists in the encoding but CDNA has no use for
# it, so it is dropped and emits nothing.
_CM_SC0 = 1
_CM_NT = 2
_CM_SC1 = 16

# Cache policy for the peer stores in _fanout_nt / _publish, per inbox memory
# type.
#
# On an uncached inbox the memory type does all the work: a store cannot sit in
# any cache, so every peer sees the payload as soon as `vmcnt(0)` retires and
# the release needs nothing beyond that.
#
#
# ``payload`` and ``flag`` are keyword arguments for ``fx.generic_store``, as
# ``(name, value)`` tuples rather than dicts: the kernel factories capture them,
# and FlyDSL folds a captured tuple into its compile-cache key but silently
# leaves a dict out. On gfx942/gfx950 they lower to:
#
#   _ST_NT     global_store_* ... nt
#   _ST_SYSTEM  global_store_* ... sc0 sc1   (a relaxed system-scope atomic)
#
# ``release`` is the sync scope of the release fence that precedes the flag, or
# None for none. System scope: the peers are other GPUs, and
# only a system-scope release orders the payload before the flag for them.
#
# ``acquire`` is the sync scope of the acquire fence between the flag spin and
# the payload loads. At system scope it lowers to ``buffer_inv sc0 sc1``, a full
# L1+L2 invalidate, which only a cached inbox needs. Every kernel takes it from
# here; the ring additionally writes back before a system-scope acquire.
#
# ``acquire`` and ``recv`` are one decision and must change together. A
# workgroup-scope acquire invalidates nothing, so it is only safe when ``recv``
# makes every payload load bypass L1 and L2 (``sc0 sc1``) -- ``nt`` is a reuse
# hint, not a bypass, and could be answered from a line cached by an earlier
# call into the same slot. A system-scope acquire is what makes ``nt`` safe.
#
# ``fanout`` picks which axis of the (peer, sector) fanout runs fastest across
# consecutive quads; see the layouts in the kernel body.
#
# ``recv`` is the cache modifier the *reader* uses on its payload loads. It is
# part of the same policy because it is the other half of the same decision: the
# more the writer is allowed to cache, the harder the reader has to work to
# avoid a stale line.
_ST_NT = (("nontemporal", True),)
_ST_SYSTEM = (("memory_order", fx.AtomicOrdering.Monotonic),)
_SYSTEM_SYNC_SCOPE = rocdl.SyncScope.OneAs
# The acquire for an un-cached inbox no cache: a compiler barrier, so
# the payload loads cannot be hoisted above the flag spin. Do not replace it
# with no fence at all, which drops that barrier.
_WORKGROUP_SYNC_SCOPE = rocdl.SyncScope.WorkgroupOneAs

_INBOX_POLICY = {
    "uncached": {
        "payload": _ST_NT,
        "flag": _ST_NT,
        "release": None,
        "acquire": _WORKGROUP_SYNC_SCOPE,
        "fanout": "sector",
        "recv": _CM_SC0 | _CM_SC1,
    },
    "finegrained": {
        "payload": _ST_NT,
        "flag": _ST_SYSTEM,
        "release": _SYSTEM_SYNC_SCOPE,
        "acquire": _SYSTEM_SYNC_SCOPE,
        "fanout": "peer",
        "recv": _CM_NT,
    },
}


def has_release_fence(inbox_memory: str) -> bool:
    """Whether this inbox type needs an L2 writeback at every publish."""
    return _INBOX_POLICY[inbox_memory]["release"] is not None


def _i32_to_bytes(i32_off):
    return fx.Int64(i32_off) * fx.Int64(I32_BYTES)


def _to_sgpr_i64(addr):
    """Copy a wave-uniform i64 from vector to scalar registers."""
    return fx.Int64(rocdl.readfirstlane(T.i64, as_ir_value(addr)))


def _global_ptr(addr_i64, elem_ty, alignment):
    """A global-address-space pointer at a raw byte address."""
    ptr_ty = fx.PointerType.get(
        elem_ty, address_space=fx.AddressSpace.Global, alignment=alignment
    )
    return fx.inttoptr(ptr_ty, fx.Int64(addr_i64))


def _store_v4i32_peer(addr_i64, data, policy):
    """Store 16 B to a peer through a per-lane global address.

    *policy* is the inbox's ``payload`` entry in ``_INBOX_POLICY``. Every wire
    offset is a multiple of 16 B, hence the alignment.
    """
    fx.generic_store(_global_ptr(addr_i64, T.i32, 16), data, **dict(policy))


def _flag_word(color):
    """One flag lane's 8 B: *color* twice, so the sector's first i32 is it."""
    return fx.Vector.from_elements([color, color], fx.Int32).bitcast(fx.Int64)[0]


def _store_flag_peer(addr_i64, color, policy):
    """Write one lane's share of a peer's 64 B handshake sector.

    *policy* is the inbox's ``flag`` entry in ``_INBOX_POLICY``. On a cacheable
    inbox it is a relaxed system-scope atomic store, which the backend emits
    ``sc0 sc1`` (write-through) without the ``vmcnt(0)`` a volatile store would
    add after every flag. Atomic stores stop at 64 bits, which is why a sector
    takes ``FLAG_LANES`` lanes of 8 B rather than four of 16 B.
    """
    fx.generic_store(_global_ptr(addr_i64, T.i64, 8), _flag_word(color), **dict(policy))


def _release_inbox(scope):
    """Release fence over global memory: publish everything stored before it."""
    fx.memory_fence(ordering=fx.AtomicOrdering.Release, syncscope=scope)


def _load_flag(addr_i64):
    """Poll one handshake flag: a relaxed system-scope atomic i32 load.

    The backend emits ``global_load_dword ... sc0 sc1``, fetched past L1 and L2
    every time, so a spin loop needs no fence in its body and no retry can see
    a stale line.

    A global load takes a per-lane address, so lanes spinning on different
    sources issue together.
    """
    return fx.generic_load(
        _global_ptr(addr_i64, T.i32, 4),
        dtype=fx.Int32,
        memory_order=fx.AtomicOrdering.Monotonic,
    )


def _poll_v4i32(addr_i64):
    """Re-read 16 B of an inbox that a peer writes without a flag.

    Two relaxed system-scope atomic i64 loads, ``global_load_dwordx2 ... sc0
    sc1``, so each retry is fetched past L1 and L2. The atomic is what keeps a
    spin on it alive: a side-effect-free loop over a plain or buffer load may be
    assumed to terminate, and the compiler is then free to delete it. A 128-bit
    atomic would be lowered to a library call, and ``volatile`` would add a
    ``vmcnt(0)`` after every load.
    """
    halves = [
        fx.generic_load(
            _global_ptr(addr_i64 + fx.Int64(8 * h), T.i64, 8),
            dtype=fx.Int64,
            memory_order=fx.AtomicOrdering.Monotonic,
        )
        for h in range(2)
    ]
    return fx.Vector.from_elements(halves, fx.Int64).bitcast(fx.Int32)


def _buffer_ptr(addr_i64, elem_ty, alignment, num_records_bytes=None):
    """A buffer-descriptor pointer at a raw global byte address.

    Without *num_records_bytes* the descriptor spans 4 GiB unchecked; with
    it, a load past the end returns 0 and a store is dropped. *addr_i64* must
    sit in scalar registers, see :func:`_to_sgpr_i64`.
    """
    return rocdl.make_buffer_ptr(
        _global_ptr(addr_i64, elem_ty, alignment), num_records_bytes=num_records_bytes
    )


def _buffer_load(ptr, elem_off, n, dtype, cache_modifier=0):
    """*n* consecutive elements of a buffer pointer, at an element offset.

    *cache_modifier* is ``_CM_*`` bits. An inbox read passes the policy's
    ``recv`` entry. ``nt`` is a reuse *hint* and does not stop the load being
    answered from the reader's own L1/L2, so it is only used where the
    policy's ``acquire`` invalidates both first (fine-grained). The uncached
    inbox reads ``_CM_SC0 | _CM_SC1``, which does bypass.
    """
    atom = fx.make_copy_atom(rocdl.BufferCopy(n * dtype.width, cache_modifier), dtype)
    reg = fx.make_rmem_tensor(n, dtype)
    fx.copy(atom, fx.make_view(ptr + elem_off, fx.make_layout(n, 1)), reg)
    return fx.Vector(fx.memref_load_vec(reg))


def _buffer_store(ptr, elem_off, vec, cache_modifier=0):
    """Store the vector *vec* to a buffer pointer, at an element offset.

    *cache_modifier* is ``_CM_*`` bits; ``_CM_SC0 | _CM_SC1`` writes through
    to memory.
    """
    atom = fx.make_copy_atom(
        rocdl.BufferCopy(vec.numel * vec.dtype.width, cache_modifier), vec.dtype
    )
    reg = fx.make_rmem_tensor(vec.numel, vec.dtype)
    fx.memref_store_vec(vec, reg)
    fx.copy(atom, reg, fx.make_view(ptr + elem_off, fx.make_layout(vec.numel, 1)))


def _load_peers(peer_ptrs, world_size):
    """The ``world_size`` inbox base addresses from the device-side peer table."""
    table = _buffer_ptr(peer_ptrs, T.i64, 8)
    return [_buffer_load(table, i, 1, fx.Int64)[0] for i in range(world_size)]


def _color_io(colors_ptr, bid):
    """``(load, store)`` for this workgroup's colour, one i32 per block."""
    colors = _buffer_ptr(colors_ptr, T.i32, 4)

    def load():
        return _buffer_load(colors, bid, 1, fx.Int32)[0]

    def store(color):
        _buffer_store(colors, bid, fx.Vector.from_elements([color], fx.Int32))

    return load, store


def _payload_io(
    inp_ptr, out_ptr, nbytes, num_tiles, atoms, block, tid, decode=None, encode=None
):
    """``(load, store)`` for this thread's 16 B of one payload atom.

    The payload is ``(num_tiles, atoms, block * 4)`` i32 and thread *tid*
    owns i32 ``[4 * tid, 4 * tid + 4)`` of every atom row. ``load(tile,
    atom)`` reads it from *inp_ptr* as i32x4, passed through *decode* if
    given; ``store(tile, atom, vec)`` writes ``encode(vec)`` to *out_ptr*.
    Both tensors are bounded by *nbytes*, the live payload, so a partial last
    tile reads 0 and its stores are dropped rather than faulting.
    """
    row_i32 = block * 4
    layout = fx.make_layout((num_tiles, atoms, row_i32), (atoms * row_i32, row_i32, 1))
    row_layout = fx.make_layout((1, row_i32), (row_i32, 1))
    copy_atom = fx.make_copy_atom(rocdl.BufferCopy128b(), fx.Int32)
    thr_copy = fx.make_tiled_copy_tv(
        copy_atom,
        fx.make_layout((1, block), (1, 1)),
        fx.make_layout((1, 4), (1, 1)),
    ).get_slice(tid)

    def _tensor(addr):
        view = fx.make_view(_global_ptr(addr, T.i32, 16), layout)
        return rocdl.make_buffer_tensor(view, max_size=False, num_records_bytes=nbytes)

    inp, out = _tensor(inp_ptr), _tensor(out_ptr)

    def _row(buf, tile, atom):
        return fx.make_view(fx.get_iter(fx.slice(buf, (tile, atom, None))), row_layout)

    def load(tile, atom):
        src = thr_copy.partition_S(_row(inp, tile, atom))
        frag = fx.make_fragment_like(src)
        fx.copy(copy_atom, src, frag)
        vec = fx.Vector(frag.load())
        return vec if decode is None else decode(vec)

    def store(tile, atom, vec):
        dst = thr_copy.partition_D(_row(out, tile, atom))
        frag = fx.make_fragment_like(dst)
        frag.store(vec if encode is None else encode(vec))
        fx.copy(copy_atom, frag, dst)

    return load, store


def _acquire_inbox(scope=_SYSTEM_SYNC_SCOPE):
    """Acquire fence over global memory at *scope*, system scope by default."""
    fx.memory_fence(ordering=fx.AtomicOrdering.Acquire, syncscope=scope)


def atom_bf16_to_f32(atom_i32):
    """16 B of bf16 (8 values) -> 8 f32.

    bf16 is the high half of f32, so this is a widening move, not a conversion
    -- exact, no rounding.
    """
    return fx.Vector(atom_i32).bitcast(fx.BFloat16).to(fx.Float32)


def atom_f32_to_bf16(acc_f32):
    """8 f32 -> 16 B of bf16.

    One rounding, at the end of whatever computed *acc_f32*, which is what makes
    the exact schedules bit-comparable with ``cross_device_reduce``'s fp32
    accumulate + single downcast.
    """
    return acc_f32.to(fx.BFloat16).bitcast(fx.Int32)


def make_pack_storage(n_i32: int):
    """LDS staging for *n_i32* packed words, 16 B aligned."""

    @fx.struct
    class PackStorage:
        pack: fx.Array[fx.Int32, n_i32, 16]

    return PackStorage


def make_hbm_operand(*, padded, nbytes, hbm_i32_ptr, hbm_layout):
    """``operand(ptr, records=None)``: the handle the atom load/store helpers
    address an HBM operand through, given its raw ``Int64`` base pointer.

    A padded build hands the pointer back untouched: its pad lanes need a
    *per-lane* bound, which a whole-tensor descriptor cannot express, so
    ``make_rowbuf_atom_row`` builds a bounded descriptor per row from it. An
    unpadded build hands back a tiled-copy tensor (layout + descriptor) whose
    ``num_records_bytes`` is *records*, the live payload by default, so a
    partial last tile reads 0 and its stores are dropped rather than faulting.

    ``hbm_i32_ptr``/``hbm_layout`` are only read on the unpadded branch, so a
    padded build may pass ``None`` for ``hbm_layout`` (it is never built there).
    The one-shot, mesh and ring kernels all share this verbatim.
    """

    def _operand(ptr, records=None):
        if padded:
            return ptr
        n = nbytes if records is None else records
        view = fx.make_view(fx.inttoptr(hbm_i32_ptr, ptr), hbm_layout)
        return rocdl.make_buffer_tensor(view, max_size=False, num_records_bytes=n)

    return _operand


# ---------------------------------------------------------------------------
# Persistent-grid sizing (host side)
# ---------------------------------------------------------------------------

# (world_size, super_tile) → VGPR-limited workgroups per CU, measured on the
# mesh all-reduce. Super-tile widens the live atom list, so residency falls as
# it grows; world size narrows each rank's share of a tile, so it rises with N.
RESIDENT_WGS_PER_CU = {
    (2, 1): 3,
    (2, 8): 4,
    (4, 1): 4,
    (4, 8): 5,
    (8, 1): 4,
    (8, 8): 6,
}


def clamp_grid_cap(
    requested: int,
    *,
    arch: str,
    world_size: int,
    super_tile: int,
    cu_count: int,
    block: int = BLOCK,
) -> int:
    """Clamp a requested persistent grid to what actually fits on the device.

    A persistent kernel deadlocks if it launches more workgroups than can be
    co-resident, so the cap has to respect VGPR-limited occupancy.

    An unmeasured *super_tile* -- the rings run ST=16 and ST=32, which this
    table does not cover -- falls back to the smallest measurement for that
    world size rather than raising. Under-launching a persistent kernel is
    always safe (each block simply loops over more tiles); over-launching is
    the failure mode, so the fallback has to err small. An unknown *arch* or
    *world_size* still raises, because there is nothing to be conservative
    with.

    *block* wider than the measured 256 scales the answer down: the table is
    VGPR- and wave-slot-limited workgroups per CU at four waves, and a 14-wave
    workgroup takes 3.5x the slots. That scaling is a **model**, not a
    measurement, and over-estimating residency here is a hang rather than a
    slowdown -- so a fused build must not rely on it alone. The host bounds
    those launches at one workgroup per CU as well; see
    ``FlyQuickAllReduceRMSNorm``.
    """
    if requested < 1 or cu_count < 1:
        raise ValueError("grid_cap and cu_count must be positive")
    if block < 1:
        raise ValueError("block must be positive")
    if arch not in ("gfx942", "gfx950"):
        raise ValueError(f"no residency measurement for {arch!r}")
    key = (int(world_size), int(super_tile))
    resident = RESIDENT_WGS_PER_CU.get(key)
    if resident is None:
        for_world = [v for (w, _st), v in RESIDENT_WGS_PER_CU.items() if w == key[0]]
        if not for_world:
            raise ValueError(f"no residency measurement for world_size={world_size}")
        resident = min(for_world)
    if block > BLOCK:
        resident = max(1, (resident * BLOCK) // int(block))
    return min(int(requested), resident * int(cu_count))


def stripes_for(n_sectors: int, n_dest: int, block: int) -> list[tuple[int, int]]:
    """``(first sector, width)`` stripes of a striped multi-destination fanout.

    One quad per ``(destination, sector)`` of a stripe, in a single pass, so a
    stripe is at most ``quads / n_dest`` sectors wide, and never wider than 8.
    """
    width = min(8, (int(block) // QUAD_LANES) // int(n_dest))
    return [(b, min(width, n_sectors - b)) for b in range(0, n_sectors, width)]


def fanout_quads(block: int, n_dest: int, n_sectors: int) -> tuple[int, int]:
    """``(quads the block has, quads a full-width striped fanout needs)``.

    A block with fewer quads than that would silently drop the sectors past
    the end; see :func:`fanout_sectors`.
    """
    stripes = [(b, min(8, n_sectors - b)) for b in range(0, n_sectors, 8)]
    return int(block) // QUAD_LANES, max(int(n_dest) * w for _, w in stripes)


# ---------------------------------------------------------------------------
# Wire protocol (device side)
#
# The kernels drive these from their bodies. The ones that branch or spin on
# traced values are run through ``traced`` -- only a ``@flyc.kernel`` body gets
# the AST rewrite for free -- and only ever *do* things: none returns a value,
# because the rewrite has no notion of one leaving a traced branch. The ones
# that compute something are straight-line and use ``select``.
# ---------------------------------------------------------------------------

# Cache policy for a payload read that must actually go to memory, on every
# inbox type. ``nt`` is only a reuse hint; ``sc0 sc1`` bypasses L1 and L2.
RECV_BYPASS = _CM_SC0 | _CM_SC1


def load_i32_drained(ptr, elem_off, cache_modifier):
    """One i32 from the buffer pointer *ptr* at an element offset, drained
    before it is read."""
    val = _buffer_load(ptr, elem_off, 1, fx.Int32, cache_modifier)[0]
    rocdl.s_waitcnt(vmcnt=0)
    return val


def select_peer_base(peers, dests, j):
    """Inbox base of ``dests[j]`` for a lane-varying *j*.

    A ``select`` chain over the Python list of loaded peer pointers rather than
    a dynamic extract from an ``fx.Vector``, which lowers to a scratch round
    trip (an earlier flag-based one-shot publish corrupted its payload that way).
    """
    base = peers[dests[0]]
    for i in range_constexpr(1, len(dests)):
        base = (j == fx.Int32(i)).select(peers[dests[i]], base)
    return base


def next_color(color, *, parity_safe: bool):
    """The colour after *color*, skipping 0, the unset sentinel.

    A schedule that double-buffers its inbox on ``color & 1`` must resume at 2,
    not 1: the colour before the wrap is -1, which is odd, so resuming at 1
    would put two consecutive tiles in the same slot, and a rank one tile ahead
    would overwrite data a peer is still reading.
    """
    color = color + fx.Int32(1)
    restart = fx.Int32(2 if parity_safe else 1)
    return (color == fx.Int32(0)).select(restart, color)


def parity_slot_i32(parity, bid, src, *, grid: int, n_src: int, wire_tile_i32: int):
    """i32 offset of the double-buffered wire slot ``[parity][bid][src]``.

    Plain arithmetic rather than ``crd2idx`` on a 3-D layout: at ``grid == 1``
    the middle mode is unit and gets coalesced away, after which a
    three-coordinate lookup silently returns a wrong (negative) index.
    """
    return (
        parity * fx.Int32(grid * n_src * wire_tile_i32)
        + bid * fx.Int32(n_src * wire_tile_i32)
        + fx.Int32(src) * fx.Int32(wire_tile_i32)
    )


def chunk_payload_io(
    inp_ptr, out_ptr, chunk_bytes, chunk, num_tiles, atoms, block, tid, **codec
):
    """:func:`_payload_io` over chunk *chunk* of an equal-split buffer.

    Both tensors are rebased to ``chunk * chunk_bytes`` and bounded by
    *chunk_bytes*, so a partial last tile reads 0 and drops its stores rather
    than spilling into the neighbouring chunk.
    """
    off = fx.Int64(chunk) * chunk_bytes
    return _payload_io(
        inp_ptr + off, out_ptr + off, chunk_bytes, num_tiles, atoms, block, tid, **codec
    )


@traced
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


@traced
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
    lands at ``slot_i32 + k * rank_tile_i32`` in ``dest_base(j)``. Lockstep
    stripes of up to 8 sectors cover a rank-tile (INT4 at the default block is
    8+8+2: 16 nibble sectors then the 2-sector E4M3 tail). One quad per
    (destination, sector) of a stripe; leftover quads sit idle. Which axis runs
    fastest across consecutive quads is a fabric question:

    - "sector": consecutive quads target consecutive peers of one sector, so a
      single store instruction hits every GPU -- ideal on xGMI, whose native
      packet is exactly the 64 B a quad writes.
    - "peer": consecutive quads walk the sectors of one peer, giving each
      destination a ``64 * width`` B contiguous run -- ideal on PCIe.
    """
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
                byte_off = _i32_to_bytes(slot_i32 + wire_idx)
                _store_v4i32_peer(dest_base(j) + byte_off, v4, payload_policy)


@traced
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

    Quads past the sector count sit idle rather than branching -- ``safe``
    keeps their address arithmetic in range. Flat sector id -> (rank-tile,
    sector) -> i32 offset by arithmetic rather than a layout: at TP8 *n_atoms*
    is 1, and a unit mode does not survive coalescing intact. The same offset
    addresses LDS and the wire, because the LDS rows and the wire rank-tiles
    share a row stride.
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
            byte_off = _i32_to_bytes(slot_i32 + flat)
            _store_v4i32_peer(dest_base + byte_off, v4, payload_policy)


def fanout_registers(stores, payload_policy):
    """Register-direct fanout: ``stores`` is ``[(byte address, i32x4)]``, issued
    in list order.

    No LDS: when a thread's 16 B lands at the same offset in every destination,
    it can go straight from registers. The caller's list order is the fabric
    knob (one destination at a time, or interleaved).
    """
    for addr, vec in stores:
        _store_v4i32_peer(addr, vec, payload_policy)


@traced
def publish_flags(
    *, tid, n_dest, dest_base, flag_i32, color, flag_policy, release_scope
):
    """Drain the payload stores, then write *color* into *n_dest* peers' flags.

    ``vmcnt(0)`` retires this wave's payload stores; the barrier joins the
    other waves, whose ``vmcnt`` is separate. On a cacheable inbox retiring is
    not enough -- the lines can sit in this XCD's L2 -- so the release fence
    writes them back before the flag goes out. Every workgroup issues its own:
    L2 is per-XCD.

    ``FLAG_LANES`` lanes per destination, 8 B each, so the 64 B sector at
    *flag_i32* (the same offset in every destination's inbox) is one store
    instruction from wave 0. ``dest_base(j)`` maps a lane-varying destination
    index to an inbox base, e.g. :func:`select_peer_base`.
    """
    rocdl.s_waitcnt(vmcnt=0)
    gpu.barrier()
    if const_expr(release_scope is not None):
        _release_inbox(release_scope)
    limit = fx.Int32(n_dest)
    dest = tid // fx.Int32(FLAG_LANES)
    safe = (dest < limit).select(dest, fx.Int32(0))
    if dest < limit:
        elem = flag_i32 + (tid % fx.Int32(FLAG_LANES)) * fx.Int32(FLAG_I32_PER_LANE)
        _store_flag_peer(dest_base(safe) + _i32_to_bytes(elem), color, flag_policy)


@traced
def wait_flags(
    *,
    tid,
    n_src,
    skip_rank,
    flag_addr,
    color,
    acquire_scope,
    writeback_scope=None,
    drain=False,
    spin_sleep=0,
):
    """Spin until *n_src* sources have coloured their flags with *color*.

    Lane ``t`` watches source ``t``; with *skip_rank* set, the lanes step over
    that rank. ``flag_addr(src)`` is the flag's address in our own inbox. The
    poll is ``sc0 sc1`` (:func:`_load_flag`), so no retry is served from a
    stale line and the loop body needs no fence.

    The fences are after the join and unconditional: only the spinning lanes
    ran the loop, and a fence inside it would be skipped when the flag is
    already set. *acquire_scope* is the inbox policy's. Pass *writeback_scope*
    when this block may hold dirty output lines and the acquire is a
    system-scope invalidate: the payload stores are drained and written back
    first, or the invalidate silently discards them. *drain* retires this
    wave's stores (``vmcnt(0)``) before the fences even without a writeback.
    A nonzero *spin_sleep* backs off ``s_sleep(spin_sleep)`` between polls:
    each poll bypasses both caches, and under arrival skew that runs for the
    whole skew window against the very line the peer is trying to write.
    """
    spin_src = tid
    if const_expr(skip_rank is not None):  # noqa: SIM102
        if tid >= fx.Int32(skip_rank):
            spin_src = tid + fx.Int32(1)
    if tid < fx.Int32(n_src):
        flag = flag_addr(spin_src)
        current = _load_flag(flag)
        while current != color:
            if const_expr(spin_sleep):
                rocdl.s_sleep(spin_sleep)
            current = _load_flag(flag)
    gpu.barrier()
    if const_expr(drain or writeback_scope is not None):
        rocdl.s_waitcnt(vmcnt=0)
    if const_expr(writeback_scope is not None):
        _release_inbox(writeback_scope)
    _acquire_inbox(acquire_scope)
