# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""gfx942/gfx950 TP∈{2,4,8} INT4 **mesh** all-reduce.

Topology of each lap: every rank pushes directly to all ``N-1`` peers, twice.

INT4 nibble: [-8,+7], −1/8, 4 B/thread, 1152 B rank-tile. Scale is
group-16 signed E5M2 (bias 12) in the 128 B region. Super-tile ST∈{1,8}; host
uses ST=1 when ``num_tiles ≤`` the occupancy-clamped persistent grid.
Payload HBM is bf16; in-kernel math is packed fp16. Each rank owns
``ATOMS / world_size`` atoms of a tile (8 GPUs → 1, 4 → 2, 2 → 4); LDS
stays ``ATOMS * rank_tile_bytes``.

The geometry and the cache policy are in ``transport``.
The wire codec is in ``quick_allreduce_codec``.
The Lamport receive and re-arm helpers are in ``quick_allreduce_lamport``.

One tuning knob besides the super-tile: ``block``, threads per workgroup. It
sets the tile (``block * ATOMS * 16 B``), hence how many blocks a payload gets
and how many flags it costs.

The Lamport build also takes ``knobs`` (``MeshKnobs``): default-off levers on
the fanout stripe, the poll, the receive order, a next-tile prefetch, the LDS
staging and the wave priority. The defaults are the kernel as it was, and every
other value is bit-identical to it; each lever's argument is at its code.

A rank never round-trips through its own inbox: its reduce-scatter share is
added from registers, and its reduced chunk is decoded from the same packet it
sends. That needs the rank at trace time, which costs one binary per rank.

``divisor`` defaults to ``DEFAULT_DIVISOR``. Loaded fp16 is divided by it
before the first pack; the gathered sum is multiplied by it once, through
the all-gather dequant scale, before the bf16 store. The second pack is
left unchanged. Only a codec with a scale plane (INT4) uses it, and the
fused all-reduce + RMSNorm build does not take one.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T

from ...allreduce_policy import (
    DEFAULT_MESH_KNOBS,
    MESH_CODECS,
    SUPPORTED_ARCHS,
    SUPPORTED_WORLDS,
    MeshKnobs,
    mesh_super_tiles,
)
from .quick_allreduce_codec import (
    SCALE_CODEC_TAG,
    _clamp_fp16_overflow,
    _codec_dequant,
    _codec_load,
    _codec_quant,
    _f16x2,
    _i32,
    _scale_from_word,
    _splat_f16x2,
    codecs_for_block,
    lamport_fix,
    scale_slot_of,
    thread_lane,
)
from .quick_allreduce_epilogue import QuickReducePlainEpilogue
from .quick_allreduce_fusions import check_fusion_args
from .quick_allreduce_lamport import (
    check_handshake,
    lamport_addrs,
    lamport_defers_rearm,
    lamport_first_reads,
    lamport_offsets,
    lamport_packets,
    lamport_poll,
    lamport_pushed,
    lamport_rearm_dwords,
    lamport_rearm_modifier,
    lamport_rearm_span,
    lamport_spin,
)
from .quick_allreduce_rmsnorm import MeshRMSNormEpilogue
from .transport import (
    _INBOX_POLICY,
    ATOMS,
    BLOCK,
    FLAG_I32_PER_LANE,
    FLAG_LANES,
    LAMPORT_FILL_BYTE,
    QUAD_LANES,
    QUADS_PER_WAVE,
    TILE_BYTES,
    WORLD,
    _acquire_inbox,
    _buffer_load,
    _buffer_ptr,
    _color_io,
    _i32_to_bytes,
    _lds_free,
    _load_flag,
    _load_peers,
    _release_inbox,
    _store_flag_peer,
    _store_v4i32_peer,
    _to_sgpr_i64,
    make_pack_storage,
)

# 1 keeps the undivided INT4 result; larger divisors are opt-in.
DEFAULT_DIVISOR = 1.0

# Re-exported for the host, which imports its tile math from this module.
__all__ = [
    "DEFAULT_DIVISOR",
    "TILE_BYTES",
    "WORLD",
    "clamp_grid_cap",
    "make_quick_allreduce_mesh_kernel",
    "mesh_lds_bytes",
]

PHASES = 2
PHASE_REDUCE_SCATTER = 0
PHASE_ALL_GATHER = 1

# Inbox buffers of the Lamport handshake. A tile (or super-tile group) is two
# rounds, its reduce-scatter and its all-gather, and round ``r`` uses buffer
# ``r % 3`` for both laps. A peer is at most one round ahead: its all-gather of
# group ``g`` needs our reduce-scatter of ``g``, and its reduce-scatter of
# ``g + 1`` needs our all-gather of ``g``. So while we read round ``r``, a peer
# writes ``r`` or ``r + 1``, and it writes ``r + 3`` -- our buffer of ``r``
# again -- only after our push of ``r + 2``. Our re-arm of ``r`` is drained
# before that push (``_lamport_recv``), so it is never on the critical path.
LAMPORT_BUFFERS = 3

# Inbox buffers of the Lamport handshake on a cacheable (fine-grained) inbox,
# where the re-arm is deferred (``lamport_defers_rearm``). After our push of
# round ``q`` we re-arm every slot of round ``q - 2`` (``_rearm_round``): by
# then we hold every peer's packet of round ``q - 1``, which it pushed after
# its release of round ``q - 2``. A peer writes the buffer of ``q - 2`` again
# in round ``q + 2``, only after our push of ``q + 1``; the poll of round
# ``q`` comes before that push, and its ``vmcnt(0)`` drains the re-arm, so it
# stays off the critical path.
LAMPORT_BUFFERS_DEFERRED = 4

# (world_size, super_tile) → VGPR-limited workgroups per CU, measured on the
# mesh kernel. Super-tile widens the live atom list, so residency falls as it
# grows; world size narrows each rank's share of a tile, so it rises with N.
_RESIDENT_WGS_PER_CU = {
    (2, 1): 3,
    (2, 8): 4,
    (4, 1): 4,
    (4, 8): 5,
    (8, 1): 4,
    (8, 8): 6,
}
_LAMPORT_RESIDENT_WGS_PER_CU = {"int5": 3, "int6": 3, "fp16": 2}


class _LazyPolled:
    """``polled`` of ``recv_order="per_source"``: ``{(src, k): (words, scale)}``
    whose source is spun on at its first lookup, and re-armed once by ``finish``.

    Trace-time only. It holds no loop of its own: the spin is *fetch*, a kernel
    closure the FlyDSL rewriter sees.
    """

    def __init__(self, fetch, finish):
        self._fetch, self._finish, self._got = fetch, finish, {}

    def __getitem__(self, key):
        if key not in self._got:
            self._got.update(self._fetch(key[0]))
        return self._got[key]

    def finish(self):
        self._finish()


def clamp_grid_cap(
    requested: int,
    *,
    arch: str,
    world_size: int,
    super_tile: int,
    cu_count: int,
    block: int = BLOCK,
    handshake: str = "flag",
    knobs: MeshKnobs = DEFAULT_MESH_KNOBS,
    codec: str = "int4",
) -> int:
    """Clamp a requested persistent grid to what actually fits on the device.

    A persistent kernel deadlocks if it launches more workgroups than can be
    co-resident, so the cap has to respect VGPR-limited occupancy.

    An unmeasured *super_tile* -- the ring runs ST=16 and ST=32, which this
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

    The table is measured on the flag handshake. The Lamport build keeps a
    sub-tile's polled packets live across its reduce, ~120 VGPRs at every
    super-tile on gfx950 TP8 (flag ST=8: 88), so a super-tile does not buy it
    the residency the flag ST=8 row measures: it takes the smallest
    measurement for the world size.

    ``prefetch=1`` (Lamport tuning knob) carries a whole tile's atoms across the
    poll: 145/139/133 VGPRs at TP8/4/2 against 121/116/109, which is 3 resident
    workgroups per CU where the table says 4 (TP8, TP4), so it takes one off.
    The other knobs stay within the table's VGPR counts (<= 128 at TP8/TP4);
    ``lds_bufs=2`` is bounded through the host's LDS clamp instead.
    """
    if requested < 1 or cu_count < 1:
        raise ValueError("grid_cap and cu_count must be positive")
    if block < 1:
        raise ValueError("block must be positive")
    if arch not in SUPPORTED_ARCHS:
        raise ValueError(
            f"quick_allreduce_mesh has no residency measurement for {arch!r}"
        )
    key = (int(world_size), int(super_tile))
    resident = None if handshake == "lamport" else _RESIDENT_WGS_PER_CU.get(key)
    if resident is None:
        for_world = [v for (w, _st), v in _RESIDENT_WGS_PER_CU.items() if w == key[0]]
        if not for_world:
            raise ValueError(
                "quick_allreduce_mesh has no residency measurement for "
                f"world_size={world_size}"
            )
        resident = min(for_world)
    if handshake == "lamport":
        resident = min(resident, _LAMPORT_RESIDENT_WGS_PER_CU.get(codec, resident))
    if handshake == "lamport" and knobs.prefetch:
        resident = max(1, resident - 1)
    if block > BLOCK:
        resident = max(1, (resident * BLOCK) // int(block))
    return min(int(requested), resident * int(cu_count))


def mesh_fanout_quads(block: int, world_size: int, codec: str) -> tuple[int, int]:
    """``(quads the block has, quads the mesh fanout needs)`` at this geometry.

    A rank-tile's sectors go out in stripes of up to 8, one quad per
    ``(peer, sector)`` of a stripe, in a single pass. A block with fewer quads
    than that would silently drop the sectors past the end.
    """
    c = codecs_for_block(int(block))[codec]
    stripes = [(b, min(8, c.n_sectors - b)) for b in range(0, c.n_sectors, 8)]
    return int(block) // QUAD_LANES, max(int(world_size) * w for _, w in stripes)


def mesh_lds_bytes(
    block: int,
    world_size: int,
    codec: str,
    partial_floats: int = 0,
    lds_bufs: int = 1,
) -> int:
    """LDS of one mesh workgroup: a packed rank-tile per (destination,
    rank-atom), ``lds_bufs`` times over, then the epilogue's per-wave
    partials, if any.

    The wider codecs pack bigger rank-tiles, so from INT6 on LDS, not VGPRs,
    can bound how many workgroups a CU holds (gfx942 TP8 ST=8).
    """
    c = codecs_for_block(int(block))[codec]
    pack_rows = (int(world_size) - 1) * (ATOMS // int(world_size))
    return int(lds_bufs) * pack_rows * c.rank_tile_bytes + int(partial_floats) * 4


def mesh_fanout_fits(block: int, world_size: int, codec: str) -> bool:
    """Whether the mesh fanout fits in *block*'s quads. For host-side gates."""
    have, need = mesh_fanout_quads(block, world_size, codec)
    return have >= need


def make_quick_allreduce_mesh_kernel(
    *,
    world_size: int,
    super_tile: int = 1,
    grid: int,
    inbox_memory: str,
    codec: str,
    fusion: str = "none",
    hidden: int | None = None,
    block: int | None = None,
    h_pad: int | None = None,
    mxfp4: bool = False,
    low_latency: bool = False,
    rank: int,
    handshake: str = "flag",
    knobs: MeshKnobs = DEFAULT_MESH_KNOBS,
):
    check_fusion_args(fusion, hidden, h_pad, mxfp4=mxfp4)
    lamport = check_handshake(handshake)
    knobs = MeshKnobs.parse(knobs)
    knobs.check(handshake=handshake, super_tile=super_tile, where="knobs")
    if knobs.prefetch and fusion != "none":
        # The fused epilogue's ``prefetch`` result would have to ride the loop
        # carry too; only the plain build is wired for it.
        raise ValueError("prefetch=1 needs the plain epilogue (fusion='none')")
    if low_latency and (fusion == "none" or super_tile != 1):
        raise ValueError("low_latency needs a fused build with super_tile=1")
    if world_size not in SUPPORTED_WORLDS:
        raise ValueError(
            f"world_size must be one of {SUPPORTED_WORLDS}, got {world_size}"
        )
    if inbox_memory not in _INBOX_POLICY:
        raise ValueError(
            f"inbox_memory must be one of {tuple(_INBOX_POLICY)}, got {inbox_memory!r}"
        )
    if codec not in MESH_CODECS:
        raise ValueError(f"codec must be one of {MESH_CODECS}, got {codec!r}")
    if not 0 <= int(rank) < world_size:
        raise ValueError(f"rank must be in [0, {world_size}), got {rank}")

    if fusion == "none":
        epilogue = QuickReducePlainEpilogue(block=block, world_size=world_size)
    else:
        epilogue = MeshRMSNormEpilogue(
            hidden=hidden,
            h_pad=h_pad,
            block=block,
            world_size=world_size,
            mxfp4=mxfp4,
            low_latency=low_latency,
        )
    block = epilogue.block
    tile_i32 = block * ATOMS * 4
    tile_bytes = tile_i32 * 4

    c = codecs_for_block(block)[codec]
    quads_per_block = block // QUAD_LANES
    policy = _INBOX_POLICY[inbox_memory]
    payload_policy = policy["payload"]
    flag_policy = policy["flag"]
    release_scope = policy["release"]
    acquire_scope = policy["acquire"]
    recv_policy = policy["recv"]
    if ATOMS % world_size != 0:
        raise ValueError(f"ATOMS={ATOMS} is not divisible by world_size={world_size}")
    legal_super_tiles = mesh_super_tiles(handshake)
    if super_tile not in legal_super_tiles:
        raise ValueError(
            f"super_tile must be one of {legal_super_tiles} on handshake="
            f"{handshake!r}, got {super_tile!r}"
        )
    if grid < 1:
        raise ValueError(f"grid must be positive, got {grid}")
    # Each rank owns this many 16-byte atoms of a 32 KiB tile
    # (8 GPUs → 1, 4 → 2, 2 → 4). LDS still holds all ATOMS atoms.
    rank_atoms = ATOMS // world_size
    # Last-sector pad is ST * rank_atoms * rank_tile_i32 after the ST tiles.
    rank_payload_i32 = rank_atoms * c.rank_tile_i32
    release_i32_off = super_tile * rank_payload_i32
    # A Lamport slot has no handshake sector.
    wire_tile_i32 = release_i32_off + (0 if lamport else 16)
    wire_tile_bytes = wire_tile_i32 * 4
    rearm_cm = lamport_rearm_modifier(inbox_memory)
    # Re-arm a round late, whole slots, rather than each polled dword at once.
    defer_rearm = lamport and lamport_defers_rearm(inbox_memory)

    # This rank's own index as a trace-time constant: its own slice of every
    # tile never leaves registers, and "all peers but me" is only expressible
    # over trace-time peer indices.
    self_rank = int(rank)
    # Destinations this rank pushes packets and flags to, in rank order. Every
    # per-destination structure below -- LDS pack rows, fanout quads, flag
    # lanes -- is indexed by position in this list, not by rank.
    push_peers = [p for p in range(world_size) if p != self_rank]
    n_push = len(push_peers)

    # One quad per (destination, sector) of a stripe, in a single pass -- unlike
    # the ring, which loops rounds. A block narrower than the shipped 256 would
    # run out of quads and silently drop the sectors past the end, so a fused
    # build that lands there is rejected rather than built. (The plain path can
    # still narrow ``stripe_width`` below; only fused derives its own block.)
    if fusion != "none":
        _, need_quads = mesh_fanout_quads(block, world_size, codec)
        if quads_per_block < need_quads:
            raise ValueError(
                f"fused hidden={hidden} gives BLOCK={block} ({quads_per_block} "
                f"quads), under the {need_quads} the {codec} fanout needs at "
                f"world_size={world_size}; use a wider hidden or the ring schedule"
            )

    # A rank-tile's sectors, in stripes of up to 8, one quad per (destination,
    # sector) of a stripe, in a single pass.
    # ``stripe=full`` (Lamport tuning knob) lifts the cap of 8: any stripe
    # width up to ``quads_per_block // n_push`` still gives every (destination,
    # sector) of the stripe its own quad, so no sector is dropped or doubled;
    # it only changes how many store instructions cover the rank-tile.
    stripe_width = quads_per_block // n_push
    if knobs.stripe == "cap8":
        stripe_width = min(8, stripe_width)
    stripes = [
        (b, min(stripe_width, c.n_sectors - b))
        for b in range(0, c.n_sectors, stripe_width)
    ]
    # ``fanout`` knob: either quad order is a bijection of the stripe's
    # (destination, sector) pairs onto quads (see ``_fanout_nt``), so forcing
    # one moves stores between lanes but not between addresses.
    fanout_axis = policy["fanout"] if knobs.fanout == "policy" else knobs.fanout
    sector_fastest = fanout_axis != "peer"
    lds_bufs = knobs.lds_bufs

    # One pack row per (destination, rank-atom).
    pack_rows = n_push * rank_atoms
    # ``lds_bufs`` copies of the pack staging, one after another.
    pack_i32 = lds_bufs * pack_rows * c.rank_tile_i32
    # The epilogue's per-wave LDS partials, if any, follow the pack staging.
    lds_bytes = mesh_lds_bytes(
        block, world_size, codec, epilogue.partial_floats, lds_bufs
    )
    PackStorage = make_pack_storage(pack_i32)

    # flags_i32 is also the i32 offset of the wire area, so the flag prefix has
    # to be a whole number of 64 B sectors (16 i32s). At a smaller multiple every
    # rank-tile and release sector straddles two hardware sectors, so the 64 B
    # fanout stores and the last-sector release stop being one sector wide.
    # A Lamport inbox has no flag prefix: its wire area starts at 0.
    grid_multiple = 1 if lamport else 16 // (PHASES * world_size)
    if grid % grid_multiple != 0:
        raise ValueError(
            f"grid must be a multiple of {grid_multiple} at "
            f"world_size={world_size} to keep the wire area 64 B aligned, got "
            f"{grid}"
        )
    flags_i32 = 0 if lamport else PHASES * grid * world_size
    if not lamport:
        wire_buffers = PHASES
    elif defer_rearm:
        wire_buffers = LAMPORT_BUFFERS_DEFERRED
    else:
        wire_buffers = LAMPORT_BUFFERS

    # One signature for both modes; the plain launcher below passes zeros for
    # the fused operands, which a plain epilogue never reads. See the ring
    # kernel for why the split is at the launcher and not here.
    @flyc.kernel(known_block_size=[block, 1, 1])
    def quick_allreduce_mesh(
        rank: fx.Int32,
        nbytes: fx.Int64,
        num_tiles: fx.Int32,
        inp_ptr: fx.Int64,
        out_ptr: fx.Int64,
        peer_ptrs: fx.Int64,
        colors_ptr: fx.Int64,
        n_blocks: fx.Int32,
        res_in_ptr: fx.Int64,
        res_out_ptr: fx.Int64,
        w_ptr: fx.Int64,
        eps: fx.Float32,
        w_bias: fx.Float32,
        q_ptr: fx.Int64,
        q_scale_ptr: fx.Int64,
        divisor: fx.Float32 = fx.Float32(DEFAULT_DIVISOR),  # noqa: B008
    ):
        _clamp_fp16_overflow()
        tid = fx.Int32(gpu.thread_id("x"))
        bid = fx.Int32(gpu.block_id("x"))

        wave, lane = thread_lane(tid, block)
        quad_layout = fx.make_layout((QUADS_PER_WAVE, QUAD_LANES), (QUAD_LANES, 1))
        quad, lane_in_quad = fx.idx2crd(lane, quad_layout).unpack()
        quad_id = wave * fx.Int32(QUADS_PER_WAVE) + quad

        pack_layout = fx.make_layout(
            (lds_bufs * pack_rows, c.rank_tile_i32), (c.rank_tile_i32, 1)
        )
        # 64 B NT sectors of one rank-tile: (sector, lane-in-quad) -> i32
        # start of the dwordx4. Isolated NT store stays explicit.
        nt_own_layout = fx.make_layout((c.n_sectors, QUAD_LANES), (16, 4))
        # Four group-16 scale bytes share the i32 slot eight threads already own.
        scale_slot, pair_in_slot = scale_slot_of(tid, block)
        # The first mode is the lap on the flag handshake, the buffer on the
        # Lamport one.
        wire_slot_layout = fx.make_layout(
            (wire_buffers, grid, world_size, super_tile),
            (
                grid * world_size * wire_tile_i32,
                world_size * wire_tile_i32,
                wire_tile_i32,
                rank_payload_i32,
            ),
        )

        allocator = fx.SharedAllocator()
        lds = allocator.allocate(PackStorage).peek()
        pack = lds.pack.view(pack_layout)
        smem_ptr = lds.pack.ptr
        peers = _load_peers(peer_ptrs, world_size)
        peer_vec = fx.Vector.from_elements(peers, dtype=fx.Int64)
        inbox = _buffer_ptr(_to_sgpr_i64(peer_vec[rank]), T.i32, 4)
        inbox_base = peers[self_rank]

        def _push_base(j):
            """Inbox base of destination *j*, a lane-varying ``push_peers`` index."""

            base = peers[push_peers[0]]
            for i in range_constexpr(1, n_push):
                base = (j == fx.Int32(i)).select(peers[push_peers[i]], base)
            return base

        # Everything that touches HBM goes through the epilogue: the plain one
        # moves whole tiles through ``_payload_io``, a fused one reaches the
        # same rows as raw bf16 atoms. Its LDS, if any, comes from this kernel's
        # one allocator, after the pack staging.
        ep = epilogue.bind(
            tid=tid,
            nbytes=nbytes,
            num_tiles=num_tiles,
            inp_ptr=inp_ptr,
            out_ptr=out_ptr,
            res_in_ptr=res_in_ptr,
            res_out_ptr=res_out_ptr,
            w_ptr=w_ptr,
            eps=eps,
            w_bias=w_bias,
            q_ptr=q_ptr,
            q_scale_ptr=q_scale_ptr,
            allocator=allocator,
        )
        # Bare names, not ``ep.<hook>(...)``: the rewriter takes a method call
        # inside a runtime loop or branch as a possible mutation of its base
        # and would try to carry ``ep`` through the loop as an SSA value.
        _load_atom_raw = ep.load_atom
        _stash_own = ep.stash_own
        _unstash_own = ep.unstash_own
        _prefetch = ep.prefetch
        _finish_tile = ep.finish_tile
        _load_color, _store_color = _color_io(colors_ptr, bid)

        # One shared divisor. Every read of the input is scaled, so the group
        # max and E4M3 byte of the first pack see the scaled values, and so
        # does our own share, which stays in registers or is reloaded. The
        # second pack's input is already a scaled sum, so it must not scale.
        # A packed fp16 divide expands per element, so multiply by a
        # reciprocal taken once.
        inv_div_h = _splat_f16x2((fx.Float32(1.0) / divisor).to(fx.Float16))
        div_h = _splat_f16x2(divisor.to(fx.Float16))

        # The fused build keeps the divisor at 1, so it skips the scaling.
        scaled = c.has_scale and not epilogue.takes_operands

        def _load_atom(tile, atom):
            raw = _load_atom_raw(tile, atom)
            if const_expr(not scaled):
                return raw
            return fx.Vector.from_elements(
                [_i32(_f16x2(raw[j]) * inv_div_h) for j in range_constexpr(4)],
                fx.Int32,
            )

        def _ag_scale(scale):
            """The all-gather dequant scale, with the divisor multiply folded in:
            one packed multiply per packet instead of one per fp16 pair."""
            return scale * div_h if const_expr(scaled) else scale

        def _pack_off(peer, i32_idx):
            return fx.get_scalar(fx.crd2idx((peer, i32_idx), pack_layout))

        def _sub_tile_i32(phase, src, sub):
            slot = fx.get_scalar(
                fx.crd2idx((fx.Int32(phase), bid, src, sub), wire_slot_layout)
            )
            return fx.Int32(flags_i32) + slot

        def _load_tile_atoms(tile):
            return [_load_atom(tile, atom) for atom in range_constexpr(ATOMS)]

        def _own_atoms(atoms):
            """This rank's reduce-scatter share, out of a whole tile's atoms."""
            return atoms[self_rank * rank_atoms : (self_rank + 1) * rank_atoms]

        def _add_f16(a, b):
            """Packed fp16 ``a + b`` of two atoms -- the codec's own FP16 add."""
            return fx.Vector.from_elements(
                [_i32(_f16x2(a[i]) + _f16x2(b[i])) for i in range_constexpr(4)],
                fx.Int32,
            )

        def _row_in(buf_row0, row):
            """*row* of the pack staging, in the buffer at row *buf_row0*;
            ``None`` is the first (and, at ``lds_bufs=1``, only) buffer."""
            if const_expr(buf_row0 is None):
                return row
            return row + buf_row0

        def _lds_write_packet(slot, words, scale, is_leader):
            for (off, pred), word in zip(c.plane_slots(tid), words):
                if pred:
                    fx.memref_store(word, pack, (slot, off))
            if const_expr(c.has_scale):  # noqa: SIM102
                if is_leader:
                    fx.memref_store(
                        scale, pack, (slot, fx.Int32(c.scale_i32_off) + scale_slot)
                    )

        def _pack_reduce_scatter(atoms, row0=None):
            """Quantize each destination's slice of this tile into LDS.

            *atoms* is this thread's share of the whole tile, as loaded by
            ``_load_tile_atoms``: always ``ATOMS`` (8) 16 B atoms. Destination *d*
            owns ``atoms[d * rank_atoms : (d+1) * rank_atoms]``. Those packets
            are later NT-stored into *d*'s reduce-scatter inbox. Our own
            slice is never packed: it stays in registers.
            """
            for j, dest in enumerate(push_peers):
                for k in range_constexpr(rank_atoms):
                    words, scale, is_leader = _codec_quant(
                        c, atoms[dest * rank_atoms + k], lane, tid
                    )
                    if const_expr(lamport):
                        words = lamport_fix(c, words, tid)
                    _lds_write_packet(
                        _row_in(row0, fx.Int32(j * rank_atoms + k)),
                        words,
                        scale,
                        is_leader,
                    )

        def _pack_all_gather(accs, row0=None):
            """Quantize the reduced slice and replicate it for every peer.

            After reduce-scatter this rank holds ``rank_atoms`` reduced
            atoms. Copy the same packets into every destination slot so the
            NT fanout can push them into every peer's all-gather inbox, and
            return our own chunk decoded from those same packets.
            """
            own = []
            for k in range_constexpr(rank_atoms):
                words, scale, is_leader = _codec_quant(c, accs[k], lane, tid)
                if const_expr(lamport):
                    # Before our own decode below, so it decodes what the
                    # peers do.
                    words = lamport_fix(c, words, tid)
                for j in range_constexpr(n_push):
                    _lds_write_packet(
                        _row_in(row0, fx.Int32(j * rank_atoms + k)),
                        words,
                        scale,
                        is_leader,
                    )
                # The same scale arithmetic as a peer's decode of this packet,
                # so every rank writes identical bits.
                own.append(
                    _codec_dequant(
                        c,
                        words,
                        _ag_scale(_scale_from_word(c, scale, pair_in_slot)),
                        tid,
                    )
                )
            return own

        def _fanout_nt(phase, inbox_src, sub, row0=None):
            """NT-store one rank-tile from LDS to every destination's inbox.

            Lockstep stripes of up to 8 sectors cover the rank-tile: at the
            default block INT4 is 8+8+2 (16 nibble sectors then the 2-sector
            scale tail), fp16 is eight full stripes. ``sector_base`` is the first
            sector of each stripe.

            One quad per (destination, sector) of a stripe; leftover quads sit
            idle. Which axis runs fastest across consecutive quads is a fabric
            question.
                - "sector": consecutive quads target consecutive peers of
                  one sector, so a single store instruction hits every GPU
                  -- ideal on xGMI, whose native packet is exactly the 64 B
                  a quad writes.
                - "peer": consecutive quads walk the sectors of one peer,
                  giving each destination a ``64 * width`` B contiguous run
                  -- ideal on PCIe.
            """
            for k in range_constexpr(rank_atoms):
                for sector_base, width in stripes:
                    n_quads = fx.Int32(n_push * width)
                    safe = (quad_id < n_quads).select(quad_id, fx.Int32(0))
                    if const_expr(sector_fastest):
                        # sector fastest
                        j = safe % fx.Int32(n_push)
                        sector_in_stripe = safe // fx.Int32(n_push)
                    else:
                        # peer fastest
                        j = safe // fx.Int32(width)
                        sector_in_stripe = safe % fx.Int32(width)
                    sector = fx.Int32(sector_base) + sector_in_stripe
                    if quad_id < n_quads:
                        vec_idx = fx.get_scalar(
                            fx.crd2idx((sector, lane_in_quad), nt_own_layout)
                        )
                        pack_row = j
                        wire_idx = vec_idx
                        if const_expr(rank_atoms != 1):
                            pack_row = j * fx.Int32(rank_atoms) + fx.Int32(k)
                            wire_idx = vec_idx + fx.Int32(k * c.rank_tile_i32)
                        pack_row = _row_in(row0, pack_row)
                        # 4xi32 NT vector cannot go through the i32 pack view.
                        v4 = fx.ptr_load(
                            smem_ptr + _pack_off(pack_row, vec_idx),
                            result_type=fx.Vector.make_type(4, fx.Int32),
                        )
                        byte_off = _i32_to_bytes(
                            _sub_tile_i32(phase, inbox_src, sub) + wire_idx
                        )
                        _store_v4i32_peer(_push_base(j) + byte_off, v4, payload_policy)

        def _publish(phase, inbox_src, color):
            """Drain payload NT stores, then write *color* into every peer inbox.

            Last 64 B of this rank's slot (after the ST rank-tiles) is the
            handshake: 16 i32s all equal to *color*. Peers spin on that
            sector in their copy of our slot; seeing *color* means our
            payload is visible.

            ``vmcnt(0)``: this 64-lane wave's NT payload stores are done.
            The workgroup barrier: the other three 64-lane waves issued
            payload too; ``vmcnt`` is per-wave, so without the join a
            wave-0 handshake could race stores still in flight. Neither
            can move after the color store, and neither can be dropped.

            On a cacheable inbox retiring the stores is not enough -- they
            can be sitting in this XCD's L2. The release fence after the join
            writes them back (``buffer_wbl2``) and waits for that to land
            before the flag goes out. Every workgroup issues its own: L2 is
            per-XCD.

            ``FLAG_LANES`` lanes per destination, 8 B each, so at most 64
            lanes: the whole handshake is one store instruction from wave 0.
            """
            rocdl.s_waitcnt(vmcnt=0)
            gpu.barrier()
            if const_expr(release_scope is not None):
                _release_inbox(release_scope)
            limit = fx.Int32(n_push)
            dest = tid // fx.Int32(FLAG_LANES)
            safe = (dest < limit).select(dest, fx.Int32(0))
            if dest < limit:
                elem = (
                    _sub_tile_i32(phase, inbox_src, fx.Int32(0))
                    + fx.Int32(release_i32_off)
                    + (tid % fx.Int32(FLAG_LANES)) * fx.Int32(FLAG_I32_PER_LANE)
                )
                _store_flag_peer(
                    _push_base(safe) + _i32_to_bytes(elem), color, flag_policy
                )

        def _wait_flag(flag, color):
            # No fence in the loop body: _load_flag carries `sc0 sc1`, so a
            # retry cannot be served from a stale line. The acquire that
            # orders the payload reads is in _wait_release, once, after the
            # join.
            current = _load_flag(flag)
            while current != color:
                current = _load_flag(flag)

        def _wait_release(phase, color):
            # Lane ``t`` watches one source. Our own flag is never published,
            # so the N-1 lanes step over our own index.
            spin_src = tid
            if tid >= fx.Int32(self_rank):
                spin_src = tid + fx.Int32(1)
            if tid < n_push:
                elem = _sub_tile_i32(phase, spin_src, fx.Int32(0)) + fx.Int32(
                    release_i32_off
                )
                _wait_flag(peer_vec[rank] + _i32_to_bytes(elem), color)
            gpu.barrier()
            # Unconditional and after the join. Only `tid < n_push` spun,
            # so scoping the acquire to the spin would leave the other waves
            # of this workgroup reading the payload unordered after it -- and
            # would also skip it entirely in the common case where the flag is
            # already set on the first read.
            #
            # The scope is the inbox policy's. On a fine-grained inbox it is
            # system scope, an L1+L2 invalidate, which the `nt` payload loads
            # rely on. On an uncached inbox it is workgroup scope, which
            # invalidates nothing and is only a compiler barrier; that is safe
            # because the policy pairs it with `sc0 sc1` payload loads
            # (`recv_policy`), which bypass both caches.
            _acquire_inbox(acquire_scope)

        def _recv_quantized(phase, src, sub, k=0):
            base = _sub_tile_i32(phase, src, sub)
            if const_expr(k):
                base = base + fx.Int32(k * c.rank_tile_i32)

            def _get(off):
                return _buffer_load(inbox, base + off, 1, fx.Int32, recv_policy)[0]

            words, word = _codec_load(c, _get, tid, scale_slot)
            return words, _scale_from_word(c, word, pair_in_slot)

        def _reduce_scattered(sub, own, polled=None):
            """Dequant-accumulate every peer's reduce-scatter packet for *sub*,
            and *own*, our share from registers, in rank order.

            *polled* is ``_lamport_recv``'s result on the Lamport handshake;
            on the flag handshake the packets are read here."""
            accs = [None] * rank_atoms
            for src in range_constexpr(world_size):
                for k in range_constexpr(rank_atoms):
                    if const_expr(src == self_rank):
                        if const_expr(accs[k] is None):
                            accs[k] = own[k]
                        else:
                            accs[k] = _add_f16(own[k], accs[k])
                    else:
                        if const_expr(polled is None):
                            words, scale = _recv_quantized(
                                PHASE_REDUCE_SCATTER, fx.Int32(src), sub, k
                            )
                        else:
                            words, scale = polled[(src, k)]
                        if const_expr(accs[k] is None):
                            accs[k] = _codec_dequant(c, words, scale, tid)
                        else:
                            accs[k] = _codec_dequant(c, words, scale, tid, accs[k])
            if const_expr(isinstance(polled, _LazyPolled)):
                polled.finish()
            return accs

        def _recv_all_gather(sub, own, polled=None):
            """Dequantize every peer's all-gather packet back into full-tile atoms.

            Our own chunk is *own*, with ``None`` for atoms stored already.
            Both are already multiplied back by the divisor, which is folded
            into each packet's group scale.
            """
            gathered = []
            for src in range_constexpr(world_size):
                for k in range_constexpr(rank_atoms):
                    if const_expr(src == self_rank):
                        gathered.append(own[k])
                    else:
                        if const_expr(polled is None):
                            words, scale = _recv_quantized(
                                PHASE_ALL_GATHER, fx.Int32(src), sub, k
                            )
                        else:
                            words, scale = polled[(src, k)]
                        gathered.append(_codec_dequant(c, words, _ag_scale(scale), tid))
            if const_expr(isinstance(polled, _LazyPolled)):
                polled.finish()
            return gathered

        def _prio(level):
            """Raise (or reset, at 0) this wave's issue priority.

            ``prio`` knob. ``s_setprio`` only reorders which ready wave the
            scheduler picks; it orders no memory operation and no barrier, so
            it cannot change a result or open a deadlock. FlyDSL has no typed
            wave-priority API, hence the raw call; it is emitted only when the
            knob is non-zero, and every raise is paired with a reset to 0 once
            the poll it covers has finished.
            """
            if const_expr(knobs.prio != 0):
                rocdl.s_setprio(level)

        def _lamport_fanout(buf, sub, row0=None, keep_prio=False):
            """Push one sub-tile into every peer's slot for this rank in *buf*.

            No drain and no flag: the packet words are their own flag. On a
            cacheable inbox each wave writes its stores back out of L2.

            The ``prio`` knob raises the wave for the fanout; a caller that
            polls next passes *keep_prio*, and the poll resets it.
            """
            _prio(knobs.prio)
            _fanout_nt(buf, rank, sub, row0)
            lamport_pushed(release_scope)
            if const_expr(not keep_prio):
                _prio(0)

        def _lamport_rearm(rearm):
            """``lamport_rearm_dwords``, then drop the ``prio`` boost that the
            fanout and the poll shared. With the re-arm deferred
            (``_rearm_round``) only ``vmcnt(0)``: it drains the round re-arm
            issued before this poll (``LAMPORT_BUFFERS_DEFERRED``)."""
            if const_expr(defer_rearm):
                rocdl.s_waitcnt(vmcnt=0)
            else:
                lamport_rearm_dwords(inbox, rearm, rearm_cm)
            _prio(0)

        def _lamport_recv(buf, sub):
            """Every peer's packet of *sub* in *buf*, ``{(src, k): (words,
            scale)}``, once none of the dwords this thread reads is the
            sentinel (``lamport_poll``, ``poll_sleep`` knob); then put the
            sentinel back in them (``lamport_rearm_dwords``, ``LAMPORT_BUFFERS``),
            or, deferred, leave that to ``_rearm_round``.

            ``recv_order="per_source"`` keeps the first reads (all issued up
            front, so the round trips still overlap) but tests and re-reads one
            source's dwords at a time, and returns a ``_LazyPolled``: a
            source's spin runs when the consumer first asks for it, so the
            dequant of source ``s`` runs while the later sources' first reads
            are still in flight. The consumer asks in rank order, which is the
            accumulation order of the batch path, so the sum is bit-identical.
            The re-arm and its ``vmcnt(0)`` run once, after the consumer's
            last source (``_LazyPolled.finish``), so no re-arm store is issued
            before a dword of this round has been decoded and every lane of a
            wave has left every spin, which the shared-dword re-arm needs.
            """
            keys = [(src, k) for src in push_peers for k in range_constexpr(rank_atoms)]
            offs, rearm = lamport_offsets(
                c,
                tid,
                scale_slot,
                (
                    _sub_tile_i32(buf, fx.Int32(src), sub)
                    + fx.Int32(k * c.rank_tile_i32)
                    for src, k in keys
                ),
            )
            if const_expr(knobs.recv_order == "per_source"):
                first = lamport_first_reads(inbox, offs)
                addrs = lamport_addrs(inbox_base, offs)
                span = len(offs) // n_push

                def _fetch(src):
                    i = push_peers.index(src)
                    lo, hi = i * span, (i + 1) * span
                    got = lamport_spin(
                        fx.Vector.from_elements(first[lo:hi], fx.Int32),
                        addrs[lo:hi],
                        knobs.poll_sleep,
                    )
                    return {
                        (src, k): (words, _scale_from_word(c, word, pair_in_slot))
                        for k, (words, word) in enumerate(
                            lamport_packets(c, got, rank_atoms)
                        )
                    }

                return _LazyPolled(_fetch, lambda: _lamport_rearm(rearm))
            got = lamport_poll(inbox, inbox_base, offs, knobs.poll_sleep)
            _lamport_rearm(rearm)
            return {
                key: (words, _scale_from_word(c, word, pair_in_slot))
                for key, (words, word) in zip(keys, lamport_packets(c, got, len(keys)))
            }

        def _next_buf(buf):
            nxt = buf + fx.Int32(1)
            return (nxt >= fx.Int32(wire_buffers)).select(fx.Int32(0), nxt)

        def _rearm_round(pushed_buf):
            """Deferred re-arm, after our push of the round in *pushed_buf*:
            every slot of this block in the buffer of two rounds before
            (``LAMPORT_BUFFERS_DEFERRED``)."""
            if const_expr(defer_rearm):
                back = pushed_buf - fx.Int32(2)
                back = (back < fx.Int32(0)).select(back + fx.Int32(wire_buffers), back)
                lamport_rearm_span(
                    inbox,
                    _sub_tile_i32(back, fx.Int32(0), fx.Int32(0)),
                    world_size * wire_tile_i32,
                    tid,
                    block,
                    rearm_cm,
                )

        # ``lds_bufs`` knob: the packs of a group alternate between two LDS
        # buffers by their running index p (ST=1: RS is 0, AG is 1; ST>1: RS
        # sub-tile s is s, AG sub-tile s is n_this + s, and every group starts
        # again at 0 -- its last pack is odd, so the first of the next group
        # lands in the other buffer). Pack p+2 reuses the buffer of pack p. The
        # reads of pack p's buffer are the fanout after its post-pack barrier,
        # all issued before pack p+1's post-pack barrier (p+1's pack, then its
        # barrier, come after p's fanout in program order). With ``lgkmcnt(0)``
        # right before that barrier every wave's reads have landed, and no wave
        # writes pack p+2 before it passes the barrier. So the join that
        # ``_lds_free`` puts before each pack is redundant: the post-pack
        # barrier does it, one pack later. The same holds across the group
        # boundary and the poll, which touch no LDS.
        def _stage_row0(p):
            """First pack row of the buffer pack number *p* writes."""
            if const_expr(lds_bufs == 1):
                return None
            return (p % fx.Int32(2)) * fx.Int32(pack_rows)

        def _pack_free():
            """Before a pack: with one buffer, wait until no wave still reads
            the rows it is about to overwrite."""
            if const_expr(lds_bufs == 1):
                _lds_free()

        def _pack_join():
            """After a pack: every wave's pack is in LDS before any fanout
            reads it. With two buffers it also lands this wave's LDS reads of
            the previous pack's buffer (see above)."""
            if const_expr(lds_bufs == 2):
                rocdl.s_waitcnt(lgkmcnt=0)
            gpu.barrier()

        # ST>1 carries nothing in registers between the three super-tile
        # loops; these move our own share and reduced chunk instead.
        def _load_own_share(tile, k):
            """Our reduce-scatter share of *tile*, reloaded from the input."""
            return _load_atom(tile, self_rank * rank_atoms + k)

        def _unstash_own_chunk(tile):
            """The own slot for ``_recv_all_gather`` in the third loop: what
            ``_stash_own`` parked, or ``None`` per atom where that store was
            already final."""
            return [
                _unstash_own(tile, self_rank * rank_atoms + k)
                for k in range_constexpr(rank_atoms)
            ]

        # Stride by the *launched* grid, not the compile-time cap. The host
        # launches fewer blocks than `grid` whenever it wants each block to own
        # several tiles (see FlyQuickAllReduce._grid_x); striding by the cap
        # instead would silently leave every tile above n_blocks unprocessed.
        # `grid` still sizes the wire slots and colour array, so
        # n_blocks <= grid always.
        n_block_tiles = (num_tiles - bid + n_blocks - fx.Int32(1)) // n_blocks
        # On the flag handshake the colour is the flag value; on the Lamport
        # one it is the buffer of the next reduce-scatter round.
        color = _load_color()

        def _lamport_tile_st1(tile, atoms, pre, color, next_tile=None):
            """One ST=1 Lamport tile: both rounds. Returns the next colour, and
            the atoms of *next_tile* if one is given.

            ``prefetch`` knob: the next tile's input loads are issued after
            this tile's reduce-scatter fanout and before its poll, so they
            ride the poll's wait. They read the input, which nothing in the
            collective writes, and ``_payload_io`` bounds every load by the
            live payload, so a *next_tile* past the last one reads zeros that
            the loop never uses. The next tile's ``pre`` is not taken: only the
            plain epilogue, whose ``prefetch`` is ``None``, can build this.
            """
            ag_buf = _next_buf(color)
            rs_row0 = None  # pack 0 of the tile: the first buffer
            ag_row0 = fx.Int32(pack_rows) if const_expr(lds_bufs == 2) else None
            _pack_free()
            _pack_reduce_scatter(atoms, rs_row0)
            _pack_join()
            _lamport_fanout(color, fx.Int32(0), rs_row0, keep_prio=True)
            _rearm_round(color)
            nxt = None
            if const_expr(next_tile is not None):
                nxt = _load_tile_atoms(next_tile)
            polled = _lamport_recv(color, fx.Int32(0))
            acc = _reduce_scattered(fx.Int32(0), _own_atoms(atoms), polled)

            _pack_free()
            own_ag = _pack_all_gather(acc, ag_row0)
            _pack_join()
            _lamport_fanout(ag_buf, fx.Int32(0), ag_row0, keep_prio=True)
            _rearm_round(ag_buf)
            polled = _lamport_recv(ag_buf, fx.Int32(0))
            gathered = _recv_all_gather(fx.Int32(0), own_ag, polled)
            _finish_tile(tile, gathered, pre)
            return _next_buf(ag_buf), nxt

        if const_expr(lamport and super_tile == 1 and knobs.prefetch):
            # Loop-carried: our colour and the atoms of the tile in flight.
            state0 = [color, *_load_tile_atoms(bid)]
            for i, state in range(fx.Int32(0), n_block_tiles, fx.Int32(1), init=state0):
                # ``init=`` loops index with ``index``, not ``i32``.
                tile = bid + fx.Int32(i) * n_blocks
                next_color, nxt = _lamport_tile_st1(
                    tile,
                    list(state[1:]),
                    None,
                    state[0],
                    next_tile=tile + n_blocks,
                )
                results = yield [next_color, *nxt]
            color = results[0]
        elif const_expr(lamport and super_tile == 1):
            for i in range(fx.Int32(0), n_block_tiles, fx.Int32(1)):
                tile = bid + i * n_blocks
                atoms = _load_tile_atoms(tile)
                pre = _prefetch(tile)
                color, _unused = _lamport_tile_st1(tile, atoms, pre, color)
        elif const_expr(lamport):
            st_i = fx.Int32(super_tile)
            for i in range(fx.Int32(0), n_block_tiles, st_i):
                remain = n_block_tiles - i
                n_this = (remain < st_i).select(remain, st_i)
                ag_buf = _next_buf(color)

                for s in range(fx.Int32(0), n_this, fx.Int32(1)):
                    tile = bid + (i + s) * n_blocks
                    atoms = _load_tile_atoms(tile)
                    rs_row0 = _stage_row0(s)
                    _pack_free()
                    _pack_reduce_scatter(atoms, rs_row0)
                    _pack_join()
                    _lamport_fanout(color, s, rs_row0)
                _rearm_round(color)

                # Each sub-tile is reduced as soon as its own packets land,
                # not once the whole group has.
                for s in range(fx.Int32(0), n_this, fx.Int32(1)):
                    tile = bid + (i + s) * n_blocks
                    own_rs = [
                        _load_own_share(tile, k) for k in range_constexpr(rank_atoms)
                    ]
                    ag_row0 = _stage_row0(n_this + s)
                    _prio(knobs.prio)
                    polled = _lamport_recv(color, s)
                    acc = _reduce_scattered(s, own_rs, polled)
                    _pack_free()
                    own_ag = _pack_all_gather(acc, ag_row0)
                    for k in range_constexpr(rank_atoms):
                        _stash_own(tile, self_rank * rank_atoms + k, own_ag[k])
                    _pack_join()
                    _lamport_fanout(ag_buf, s, ag_row0)
                _rearm_round(ag_buf)

                for s in range(fx.Int32(0), n_this, fx.Int32(1)):
                    tile = bid + (i + s) * n_blocks
                    _prio(knobs.prio)
                    polled = _lamport_recv(ag_buf, s)
                    gathered = _recv_all_gather(s, _unstash_own_chunk(tile), polled)
                    _finish_tile(tile, gathered, None)

                color = _next_buf(ag_buf)
        elif const_expr(super_tile == 1):
            for i in range(fx.Int32(0), n_block_tiles, fx.Int32(1)):
                tile = bid + i * n_blocks
                atoms = _load_tile_atoms(tile)
                pre = _prefetch(tile)
                _pack_reduce_scatter(atoms)
                gpu.barrier()
                _fanout_nt(PHASE_REDUCE_SCATTER, rank, fx.Int32(0))
                _publish(PHASE_REDUCE_SCATTER, rank, color)

                _wait_release(PHASE_REDUCE_SCATTER, color)
                # ST=1 keeps the whole tile in registers across the wait, so
                # our share is simply read back out of it.
                acc = _reduce_scattered(fx.Int32(0), _own_atoms(atoms))

                own_ag = _pack_all_gather(acc)
                gpu.barrier()
                _fanout_nt(PHASE_ALL_GATHER, rank, fx.Int32(0))
                _publish(PHASE_ALL_GATHER, rank, color)

                _wait_release(PHASE_ALL_GATHER, color)
                gathered = _recv_all_gather(fx.Int32(0), own_ag)
                _finish_tile(tile, gathered, pre)

                color = color + fx.Int32(1)
                if color == fx.Int32(0):  # 0 is unset sentinel
                    color = fx.Int32(1)
        else:
            st_i = fx.Int32(super_tile)
            for i in range(fx.Int32(0), n_block_tiles, st_i):
                remain = n_block_tiles - i
                n_this = (remain < st_i).select(remain, st_i)

                for s in range(fx.Int32(0), n_this, fx.Int32(1)):
                    tile = bid + (i + s) * n_blocks
                    atoms = _load_tile_atoms(tile)
                    _pack_reduce_scatter(atoms)
                    gpu.barrier()
                    _fanout_nt(PHASE_REDUCE_SCATTER, rank, s)
                    if (s + fx.Int32(1)) < n_this:
                        # Drain this wave's LDS loads, then join the WG.
                        # world_size<8 leaves waves idle in fanout; without the
                        # barrier they pack the next sub-tile into LDS while
                        # a busy wave still ptr_loads it. lgkmcnt only: NT
                        # payload stays in flight until _publish.
                        rocdl.s_waitcnt(lgkmcnt=0)
                        gpu.barrier()

                _publish(PHASE_REDUCE_SCATTER, rank, color)
                _wait_release(PHASE_REDUCE_SCATTER, color)

                for s in range(fx.Int32(0), n_this, fx.Int32(1)):
                    # Nothing is carried from the first loop -- that would be ST
                    # tiles of registers across the wait -- so our share is
                    # reloaded from the input, a local cached read against the
                    # uncached inbox read it replaces. Likewise our reduced
                    # chunk is stored now rather than carried to the third loop;
                    # the all-gather publish below releases it.
                    tile = bid + (i + s) * n_blocks
                    own_rs = [
                        _load_own_share(tile, k) for k in range_constexpr(rank_atoms)
                    ]
                    acc = _reduce_scattered(s, own_rs)
                    own_ag = _pack_all_gather(acc)
                    for k in range_constexpr(rank_atoms):
                        _stash_own(tile, self_rank * rank_atoms + k, own_ag[k])
                    gpu.barrier()
                    _fanout_nt(PHASE_ALL_GATHER, rank, s)
                    if (s + fx.Int32(1)) < n_this:
                        rocdl.s_waitcnt(lgkmcnt=0)
                        gpu.barrier()

                _publish(PHASE_ALL_GATHER, rank, color)
                _wait_release(PHASE_ALL_GATHER, color)

                for s in range(fx.Int32(0), n_this, fx.Int32(1)):
                    tile = bid + (i + s) * n_blocks
                    gathered = _recv_all_gather(s, _unstash_own_chunk(tile))
                    _finish_tile(tile, gathered, None)

                color = color + fx.Int32(1)
                if color == fx.Int32(0):  # 0 is unset sentinel
                    color = fx.Int32(1)
        if tid == 0:
            _store_color(color)
        gpu.barrier()

    flat_wg = f"{block},{block}"

    # Two launchers over one kernel: the plain one keeps the eight-argument
    # signature the host builds today and passes zeros for the fused operands.
    @flyc.jit
    def launch_quick_allreduce_mesh(
        rank: fx.Int32,
        nbytes: fx.Int64,
        num_tiles: fx.Int32,
        inp_ptr: fx.Int64,
        out_ptr: fx.Int64,
        peer_ptrs: fx.Int64,
        colors_ptr: fx.Int64,
        grid_x: fx.Int32,
        divisor: fx.Float32 = fx.Float32(DEFAULT_DIVISOR),  # noqa: B008
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        quick_allreduce_mesh(
            rank,
            nbytes,
            num_tiles,
            inp_ptr,
            out_ptr,
            peer_ptrs,
            colors_ptr,
            grid_x,
            fx.Int64(0),
            fx.Int64(0),
            fx.Int64(0),
            fx.Float32(0.0),
            fx.Float32(0.0),
            fx.Int64(0),
            fx.Int64(0),
            divisor,
            value_attrs={"rocdl.flat_work_group_size": flat_wg},
        ).launch(
            grid=(grid_x, 1, 1),
            block=(block, 1, 1),
            stream=stream,
        )

    @flyc.jit
    def launch_quick_allreduce_mesh_fused(
        rank: fx.Int32,
        nbytes: fx.Int64,
        num_tiles: fx.Int32,
        inp_ptr: fx.Int64,
        out_ptr: fx.Int64,
        peer_ptrs: fx.Int64,
        colors_ptr: fx.Int64,
        grid_x: fx.Int32,
        res_in_ptr: fx.Int64,
        res_out_ptr: fx.Int64,
        w_ptr: fx.Int64,
        eps: fx.Float32,
        w_bias: fx.Float32,
        q_ptr: fx.Int64,
        q_scale_ptr: fx.Int64,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        quick_allreduce_mesh(
            rank,
            nbytes,
            num_tiles,
            inp_ptr,
            out_ptr,
            peer_ptrs,
            colors_ptr,
            grid_x,
            res_in_ptr,
            res_out_ptr,
            w_ptr,
            eps,
            w_bias,
            q_ptr,
            q_scale_ptr,
            fx.Float32(DEFAULT_DIVISOR),
            value_attrs={"rocdl.flat_work_group_size": flat_wg},
        ).launch(
            grid=(grid_x, 1, 1),
            block=(block, 1, 1),
            stream=stream,
        )

    # The inbox memory type and wire codec both change the emitted code, so
    # both have to be part of the symbol name -- two variants that differ only
    # in cache bits or wire format must not collide in the JIT cache.
    tag = f"{SCALE_CODEC_TAG}_ws{world_size}_r{self_rank}_st{super_tile}_g{grid}"
    tag += f"_{inbox_memory}_{codec}_b{block}" + epilogue.tag
    if lamport:
        tag += "_lamport"
    tag += knobs.tag()
    launcher = (
        launch_quick_allreduce_mesh_fused
        if epilogue.takes_operands
        else launch_quick_allreduce_mesh
    )
    launcher.func.__name__ = f"launch_quick_allreduce_mesh_{tag}"
    try:
        quick_allreduce_mesh.func.__name__ = f"quick_allreduce_mesh_{tag}"
    except AttributeError:
        pass
    return {
        "launch": launcher,
        "flags_bytes": flags_i32 * 4,
        "data_bytes": wire_buffers * grid * world_size * wire_tile_bytes,
        "lds_bytes": lds_bytes,
        "tile_bytes": tile_bytes,
        # HBM bytes one tile spans, which is what a tile *count* has to come
        # from. Equal to tile_bytes unless the build is padded, where the wire
        # tile is rows_per_tile rows of h_pad but the footprint is
        # rows_per_tile rows of hidden.
        "hbm_tile_bytes": epilogue.hbm_tile_bytes,
        "tile_fp16": tile_bytes // 2,
        "rank_tile_bytes": c.rank_tile_bytes,
        "wire_tile_bytes": wire_tile_bytes,
        "super_tile": super_tile,
        "world_size": world_size,
        "inbox_memory": inbox_memory,
        "codec": codec,
        "payload_policy": payload_policy,
        "flag_policy": flag_policy,
        "release_scope": release_scope,
        "rank_atoms": rank_atoms,
        "grid": grid,
        "block": block,
        "rank": self_rank,
        "handshake": handshake,
        "knobs": knobs,
        # Byte every inbox byte starts as: the Lamport sentinel, or 0.
        "inbox_fill": LAMPORT_FILL_BYTE if lamport else 0,
        **epilogue.spec(),
    }
