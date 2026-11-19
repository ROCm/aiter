# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""gfx942/gfx950 quantized **hierarchical** all-reduce, for hosts whose GPUs
form uniform groups (the CPU sockets of a two-socket PCIe host).

The ranks split into two groups of ``G``. Ranks at the same local index in the
two groups are *partners*. One super-tile group of tiles takes three laps:

1. **Local reduce-scatter.** A tile's ``ATOMS`` atoms split into ``G`` chunks
   of ``ATOMS / G`` atoms; chunk ``l`` belongs to local index ``l``. Each rank
   ends the lap with its group's partial sum of its own chunk.
2. **Cross exchange.** Each rank quantizes its group's partial sum and sends
   it to its partner, a one-shot between two ranks, and also into its own
   inbox. Both partners then evaluate ``dq(P_group0) + dq(P_group1)`` from the
   same two packets in the same order, so they hold bit-identical sums.
3. **Local all-gather.** Each rank quantizes the sum once, decodes its own
   chunk from that packet, and the group passes the packets around. All ``N``
   ranks decode the same packets, so they end with the same bits.

Each local lap runs one of two schedules inside the group (``local_rs``,
``local_ag``):

* ``"mesh"``: one hop. Each rank pushes every chunk it does not own straight
  to its owner (reduce-scatter), or its own chunk to every other rank of the
  group (all-gather). Every value is quantized once on the lap.
* ``"ring"``: ``G - 1`` hops to the next rank of the group, each a single
  contiguous run into one inbox, as in the ring kernel. The reduce-scatter
  requantizes the running partial at every hop; the all-gather forwards the
  bytes it received untouched, so it adds one quantization, as the mesh does.

Only lap 2 crosses between groups: ``1/G`` of a tile per rank, on ``G``
concurrent pair streams, instead of the mesh's fanout to every remote peer or
the ring's single crossing stream. Laps 1 and 3 move the bytes of a TP``G``
mesh or ring inside each group. Each lap has its own codec.

Handshakes are per hop and per super-tile group, in a 64 B sector at the tail
of each inbox slot, as in the ring. The mesh's single-colour argument -- a
peer cannot write group ``i + 1`` into a slot before we have read group ``i``
-- holds for the local laps of either schedule: every local slot's next write
needs a send of ours that comes after our read (a group's all-gather needs
every rank's chunk, and its reduce-scatter every rank's contribution). It
does not hold for the cross slot: once our partner has our group ``i`` packet
it can finish group ``i`` and run group ``i + 1``'s local reduce-scatter
inside its own group without us, and send its next cross packet. The cross
slots are therefore double-buffered by the parity of the colour; group
``i + 2`` cannot arrive before we have sent ``i + 1``, which is after we read
``i``.

Lamport handshake (``handshake="lamport"``)
-------------------------------------------
No flags and no handshake sectors, on all three laps. An unwritten inbox dword
holds the sentinel (the inbox starts filled with ``LAMPORT_FILL_BYTE``), every
packet goes through ``lamport_fix`` before it is staged, and a reader polls
the dwords it reads, one sub-tile at a time, until none is the sentinel. On a
cacheable inbox each wave writes its pushes back out of L2 with a release
after every lap (every hop on the ring); there is no drain barrier and no flag
round trip.

A slot is **re-armed one super-tile group late**. On a fine-grained inbox a
push can reach the reader by eviction before the writer's release writes the
line back, and a sentinel put back at once can be overwritten by the same
old bytes (seen with the cross-exchange mover prototype). A reader therefore
re-arms a writer's slots only once it holds a later packet from that writer,
which left after the writer's release:

* Local slots of group ``i - 1``, after group ``i``'s reduce-scatter has
  landed: every local rank's contribution to it, directly on the mesh or
  through the ring, left after that rank had finished ``i - 1``.
* The partner's cross slot of ``i - 1``, after its packet of ``i`` has landed.

Whole slots are re-armed with 16 B write-through stores (``_rearm``).

Buffers:

* Local slots, two by group parity. A local rank writes group ``i + 1`` while
  we may still read ``i``. It writes ``i + 2``, the buffer of ``i`` again,
  only after it has our all-gather of ``i + 1``, which we push after the cross
  poll of ``i + 1``; that poll's ``vmcnt(0)`` drains our re-arm of ``i``,
  issued after the reduce-scatter of ``i + 1`` landed.
* Cross slots, three. The partner runs up to a group ahead (above), and our
  re-arm of ``i`` waits for its packet of ``i + 1``. It writes ``i + 3`` only
  after it has our ``i + 2``, which we push after the reduce-scatter polls of
  ``i + 2`` have drained the re-arm.

The colour cycles through ``0 .. 5``: its parity is the local buffer, its
value mod 3 the cross buffer. Our own cross packet goes into our own inbox as
on the flag handshake; lap 3 reads it after the partner's has landed, by which
time every wave's poll has drained its stores.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T

from ...allreduce_policy import (
    HANDSHAKES,
    HIER_CODECS,
    HIER_LOCAL_SCHEDULES,
    HIER_SUPER_TILES,
)
from .quick_allreduce_codec import (
    GROUP,
    SCALE_CODEC_TAG,
    _clamp_fp16_overflow,
    _codec_dequant,
    _codec_load,
    _codec_quant,
    _f16x2,
    _i32,
    _scale_from_word,
    codecs_for_block,
    lamport_fix,
    lamport_slots,
    scale_slot_of,
    split_lamport_packet,
    thread_lane,
)
from .quick_allreduce_epilogue import QuickReducePlainEpilogue
from .transport import (
    _CM_SC1,
    _INBOX_POLICY,
    _RECV_POLICY,
    ATOMS,
    FLAG_I32_PER_LANE,
    FLAG_LANES,
    LAMPORT_FILL_BYTE,
    LAMPORT_SENTINEL,
    QUAD_LANES,
    QUADS_PER_WAVE,
    _acquire_inbox,
    _any_sentinel,
    _buffer_load,
    _buffer_ptr,
    _buffer_store,
    _color_io,
    _i32_to_bytes,
    _load_flag,
    _load_peers,
    _release_inbox,
    _store_flag_peer,
    _store_v4i32_peer,
    _to_sgpr_i64,
    lamport_rearm_modifier,
    make_pack_storage,
)

# The one 64 B handshake sector at the tail of every slot, in i32.
_FLAG_SECTOR_I32 = 16

# Inbox buffers and colour cycle of the Lamport handshake; see the module
# docstring.
_LAMPORT_LOCAL_BUFFERS = 2
_LAMPORT_X_BUFFERS = 3
_LAMPORT_COLORS = 6


def check_groups(groups, world_size: int) -> tuple[tuple[int, ...], ...]:
    """*groups* as a tuple of sorted tuples, or raise.

    Two groups of equal size ``G >= 2`` that together hold every rank once,
    with ``ATOMS`` divisible by ``G``.
    """
    groups = tuple(tuple(sorted(int(r) for r in g)) for g in groups)
    if len(groups) != 2:
        raise ValueError(f"hier needs exactly 2 groups, got {groups}")
    size = len(groups[0])
    if size < 2 or any(len(g) != size for g in groups):
        raise ValueError(f"hier needs 2 groups of equal size >= 2, got {groups}")
    if sorted(r for g in groups for r in g) != list(range(world_size)):
        raise ValueError(
            f"groups {groups} must hold every rank of world_size={world_size} once"
        )
    if ATOMS % size:
        raise ValueError(f"ATOMS={ATOMS} is not divisible by the group size {size}")
    return groups


def check_local_schedule(label: str, schedule: str) -> str:
    """*schedule* if a local lap can run it, or raise."""
    if schedule not in HIER_LOCAL_SCHEDULES:
        raise ValueError(
            f"{label} must be one of {HIER_LOCAL_SCHEDULES}, got {schedule!r}"
        )
    return schedule


def _local_rows(schedule: str, group_size: int) -> int:
    """Pack rows of a local lap, in units of ``ATOMS / G`` rank-tiles: one
    destination per hop on the ring, ``G - 1`` at once on the mesh."""
    return int(group_size) - 1 if schedule == "mesh" else 1


def hier_fanout_quads(block: int, group_size: int, codecs) -> tuple[int, int]:
    """``(quads the block has, quads the local fanout needs)``, for the
    *codecs* of the laps that run the mesh. A block with fewer would drop the
    sectors past the end. The ring's one destination stores in rounds, so it
    fits any block."""
    have = int(block) // QUAD_LANES
    need = 0
    for name in codecs:
        c = codecs_for_block(int(block))[name]
        need = max(need, (group_size - 1) * min(8, c.n_sectors))
    return have, need


def hier_lds_bytes(
    block: int,
    group_size: int,
    rs_codec,
    x_codec,
    ag_codec,
    local_rs: str = "mesh",
    local_ag: str = "mesh",
) -> int:
    """LDS of one workgroup: the largest of the laps' staging, ``ATOMS / G``
    rank-tiles per destination. A mesh lap stages ``G - 1`` destinations, a
    ring hop and the cross lap one."""
    cs = codecs_for_block(int(block))
    rank_atoms = ATOMS // int(group_size)
    widest = max(
        _local_rows(local_rs, group_size) * cs[rs_codec].rank_tile_i32,
        cs[x_codec].rank_tile_i32,
        _local_rows(local_ag, group_size) * cs[ag_codec].rank_tile_i32,
    )
    return 4 * rank_atoms * widest


def make_quick_allreduce_hier_kernel(
    *,
    world_size: int,
    rank: int,
    groups,
    super_tile: int = 1,
    grid: int,
    inbox_memory: str,
    rs_codec: str,
    x_codec: str,
    ag_codec: str,
    block: int,
    local_rs: str = "mesh",
    local_ag: str = "mesh",
    handshake: str = "flag",
):
    """Build the hierarchical kernel for one *rank*.

    *groups* is two tuples of ranks; partners sit at the same position of the
    sorted groups. The rank, its group, its local peers and its partner are
    all trace-time constants, so this is one binary per rank. *local_rs* and
    *local_ag* pick the schedule of the two local laps, ``"mesh"`` or
    ``"ring"``. *handshake* is one of ``HANDSHAKES``, for all three laps.
    """
    if handshake not in HANDSHAKES:
        raise ValueError(f"handshake must be one of {HANDSHAKES}, got {handshake!r}")
    lamport = handshake == "lamport"
    groups = check_groups(groups, world_size)
    if not 0 <= int(rank) < world_size:
        raise ValueError(f"rank must be in [0, {world_size}), got {rank}")
    if inbox_memory not in _INBOX_POLICY:
        raise ValueError(
            f"inbox_memory must be one of {tuple(_INBOX_POLICY)}, got {inbox_memory!r}"
        )
    for label, name in (("rs", rs_codec), ("x", x_codec), ("ag", ag_codec)):
        if name not in HIER_CODECS:
            raise ValueError(
                f"{label}_codec must be one of {HIER_CODECS}, got {name!r}"
            )
    if super_tile not in HIER_SUPER_TILES:
        raise ValueError(
            f"super_tile must be one of {HIER_SUPER_TILES}, got {super_tile!r}"
        )
    if grid < 1:
        raise ValueError(f"grid must be positive, got {grid}")
    check_local_schedule("local_rs", local_rs)
    check_local_schedule("local_ag", local_ag)
    rs_ring = local_rs == "ring"
    ag_ring = local_ag == "ring"

    epilogue = QuickReducePlainEpilogue(block=block, world_size=len(groups[0]))
    block = epilogue.block
    tile_bytes = block * ATOMS * 16
    quads_per_block = block // QUAD_LANES

    self_rank = int(rank)
    my_group = 0 if self_rank in groups[0] else 1
    local = groups[my_group]
    G = len(local)
    li = local.index(self_rank)
    partner = groups[1 - my_group][li]
    # The ring successor inside the group, for the local laps that run the ring.
    nxt = local[(li + 1) % G]
    # Local peers in local-index order; every per-destination structure of a
    # local lap -- pack rows, fanout quads, flag lanes -- is indexed by
    # position in this list.
    push_locals = [m for m in range(G) if m != li]
    push_ranks = [local[m] for m in push_locals]
    n_push = len(push_ranks)
    rank_atoms = ATOMS // G

    mesh_codecs = [
        c for c, ring in ((rs_codec, rs_ring), (ag_codec, ag_ring)) if not ring
    ]
    have, need = hier_fanout_quads(block, G, mesh_codecs)
    if have < need:
        raise ValueError(
            f"block={block} has {have} quads, under the {need} the local fanout "
            f"needs at group size {G}"
        )

    policy = _INBOX_POLICY[inbox_memory]
    payload_policy = policy["payload"]
    flag_policy = policy["flag"]
    release_scope = policy["release"]
    acquire_scope = policy["acquire"]
    recv_policy = policy["recv"]
    sector_fastest = policy["fanout"] != "peer"

    cs = codecs_for_block(block)
    rs, xc, ag = cs[rs_codec], cs[x_codec], cs[ag_codec]

    # Inbox geometry, in i32. Each slot is ST sub-tiles of rank_atoms
    # rank-tiles, then, on the flag handshake, the 64 B handshake sector.
    # Every rank-tile is a whole number of sectors, so every slot and region
    # stays 64 B aligned.
    flag_sector = 0 if lamport else _FLAG_SECTOR_I32

    def _slot(c):
        payload = rank_atoms * c.rank_tile_i32
        return payload, super_tile * payload, super_tile * payload + flag_sector

    rs_payload, rs_release, rs_slot = _slot(rs)
    x_payload, x_release, x_slot = _slot(xc)
    ag_payload, ag_release, ag_slot = _slot(ag)
    # A local lap's slots per block: on the mesh one per source local index,
    # on the ring one per hop, each written only by the ring predecessor.
    rs_n = G - 1 if rs_ring else G
    ag_n = G - 1 if ag_ring else G
    local_buffers = _LAMPORT_LOCAL_BUFFERS if lamport else 1
    x_buffers = _LAMPORT_X_BUFFERS if lamport else 2
    # LOCAL_RS [local_buffers][grid][rs_n], CROSS [x_buffers][grid][2 groups],
    # LOCAL_AG [local_buffers][grid][ag_n].
    rs_base = 0
    x_base = rs_base + local_buffers * grid * rs_n * rs_slot
    ag_base = x_base + x_buffers * grid * 2 * x_slot
    inbox_i32 = ag_base + local_buffers * grid * ag_n * ag_slot
    rearm_cm = lamport_rearm_modifier(inbox_memory)

    lds_bytes = hier_lds_bytes(
        block, G, rs_codec, x_codec, ag_codec, local_rs, local_ag
    )
    PackStorage = make_pack_storage(lds_bytes // 4)

    @flyc.kernel(known_block_size=[block, 1, 1])
    def quick_allreduce_hier(
        rank_unused: fx.Int32,
        nbytes: fx.Int64,
        num_tiles: fx.Int32,
        inp_ptr: fx.Int64,
        out_ptr: fx.Int64,
        peer_ptrs: fx.Int64,
        colors_ptr: fx.Int64,
        n_blocks: fx.Int32,
    ):
        _clamp_fp16_overflow()
        tid = fx.Int32(gpu.thread_id("x"))
        bid = fx.Int32(gpu.block_id("x"))

        wave, lane = thread_lane(tid, block)
        quad_layout = fx.make_layout((QUADS_PER_WAVE, QUAD_LANES), (QUAD_LANES, 1))
        quad, lane_in_quad = fx.idx2crd(lane, quad_layout).unpack()
        quad_id = wave * fx.Int32(QUADS_PER_WAVE) + quad
        scale_slot, pair_in_slot = scale_slot_of(tid, block)

        # One allocation, one row view per codec.
        allocator = fx.SharedAllocator()
        lds = allocator.allocate(PackStorage).peek()
        smem_ptr = lds.pack.ptr
        pack_rows = {c.name: lds_bytes // 4 // c.rank_tile_i32 for c in (rs, xc, ag)}
        packs = {
            c.name: lds.pack.view(
                fx.make_layout(
                    (pack_rows[c.name], c.rank_tile_i32), (c.rank_tile_i32, 1)
                )
            )
            for c in (rs, xc, ag)
        }

        peers = _load_peers(peer_ptrs, world_size)
        self_base = peers[self_rank]
        partner_base = peers[partner]
        next_base = peers[nxt]
        inbox = _buffer_ptr(_to_sgpr_i64(self_base), T.i32, 4)

        ep = epilogue.bind(
            tid=tid,
            nbytes=nbytes,
            num_tiles=num_tiles,
            inp_ptr=inp_ptr,
            out_ptr=out_ptr,
        )
        # Bare names, not ``ep.<hook>(...)``; see the mesh kernel.
        _load_atom = ep.load_atom
        _stash_own = ep.stash_own
        _complete = ep.complete
        _unstash_own = ep.unstash_own
        _finish_tile = ep.finish_tile
        _load_color, _store_color = _color_io(colors_ptr, bid)

        def _push_base(j):
            """Inbox base of local destination *j*, a lane-varying index."""
            base = peers[push_ranks[0]]
            for i in range_constexpr(1, n_push):
                base = (j == fx.Int32(i)).select(peers[push_ranks[i]], base)
            return base

        # Slot offsets, by arithmetic rather than a layout: at G = 4 or 8 a
        # unit mode would not survive coalescing (see the ring kernel).
        def _local_slot(base, slot, payload, n, idx, sub, buf):
            row = bid if buf is None else buf * fx.Int32(grid) + bid
            return (
                fx.Int32(base)
                + (row * fx.Int32(n) + idx) * fx.Int32(slot)
                + sub * fx.Int32(payload)
            )

        def _rs_slot(idx, sub, buf=None):
            """Mesh: *idx* is the source local index. Ring: the hop. *buf* is
            the Lamport buffer (None on the flag handshake, which has one)."""
            return _local_slot(rs_base, rs_slot, rs_payload, rs_n, idx, sub, buf)

        def _ag_slot(idx, sub, buf=None):
            return _local_slot(ag_base, ag_slot, ag_payload, ag_n, idx, sub, buf)

        def _x_slot(parity, grp, sub):
            return (
                fx.Int32(x_base)
                + ((parity * fx.Int32(grid) + bid) * fx.Int32(2) + fx.Int32(grp))
                * fx.Int32(x_slot)
                + sub * fx.Int32(x_payload)
            )

        def _lds_write_packet(c, row, words, scale, is_leader):
            pack = packs[c.name]
            for (off, pred), word in zip(c.plane_slots(tid), words):
                if pred:
                    fx.memref_store(word, pack, (row, off))
            if const_expr(c.has_scale):  # noqa: SIM102
                if is_leader:
                    fx.memref_store(
                        scale, pack, (row, fx.Int32(c.scale_i32_off) + scale_slot)
                    )

        def _recv_raw(c, base):
            """One packet of codec *c* from our inbox at i32 offset *base*,
            undecoded, as the ring's all-gather forwards it."""

            def _get(off):
                return _buffer_load(inbox, base + off, 1, fx.Int32, recv_policy)[0]

            return _codec_load(c, _get, tid, scale_slot)

        def _recv(c, base):
            """One packet of codec *c* from our inbox at i32 offset *base*."""
            words, word = _recv_raw(c, base)
            return words, _scale_from_word(c, word, pair_in_slot)

        def _fanout_local(c, slot_i32):
            """NT-store the staged rank-tiles from LDS into every local
            destination's inbox, at the same *slot_i32* in each.

            Pack row ``j * rank_atoms + k`` goes to ``push_ranks[j]``, as
            rank-tile ``k`` of the slot. One quad per (destination, sector) of
            a stripe, as in the mesh.
            """
            width_max = min(8, quads_per_block // n_push)
            stripes = [
                (b, min(width_max, c.n_sectors - b))
                for b in range(0, c.n_sectors, width_max)
            ]
            for k in range_constexpr(rank_atoms):
                for sector_base, width in stripes:
                    n_quads = fx.Int32(n_push * width)
                    safe = (quad_id < n_quads).select(quad_id, fx.Int32(0))
                    if const_expr(sector_fastest):
                        j = safe % fx.Int32(n_push)
                        sector_in_stripe = safe // fx.Int32(n_push)
                    else:
                        j = safe // fx.Int32(width)
                        sector_in_stripe = safe % fx.Int32(width)
                    sector = fx.Int32(sector_base) + sector_in_stripe
                    if quad_id < n_quads:
                        vec_idx = sector * fx.Int32(16) + lane_in_quad * fx.Int32(4)
                        row = j * fx.Int32(rank_atoms) + fx.Int32(k)
                        v4 = fx.ptr_load(
                            smem_ptr + row * fx.Int32(c.rank_tile_i32) + vec_idx,
                            result_type=fx.Vector.make_type(4, fx.Int32),
                        )
                        byte_off = _i32_to_bytes(
                            slot_i32 + fx.Int32(k * c.rank_tile_i32) + vec_idx
                        )
                        _store_v4i32_peer(_push_base(j) + byte_off, v4, payload_policy)

        def _store_rows(c, dest_base, slot_i32):
            """Store pack rows ``0..rank_atoms-1`` of codec *c* to one
            destination as a single contiguous run, sectors in address order."""
            n_total = rank_atoms * c.n_sectors
            for rnd in range_constexpr(-(-n_total // quads_per_block)):
                s = quad_id + fx.Int32(rnd * quads_per_block)
                if s < fx.Int32(n_total):
                    flat = s * fx.Int32(16) + lane_in_quad * fx.Int32(4)
                    v4 = fx.ptr_load(
                        smem_ptr + flat,
                        result_type=fx.Vector.make_type(4, fx.Int32),
                    )
                    byte_off = _i32_to_bytes(slot_i32 + flat)
                    _store_v4i32_peer(dest_base + byte_off, v4, payload_policy)

        def _drain():
            """Every wave's stores retired and, on a cacheable inbox, written
            back from L2, ahead of a flag. See the mesh's ``_publish``."""
            rocdl.s_waitcnt(vmcnt=0)
            gpu.barrier()
            if const_expr(release_scope is not None):
                _release_inbox(release_scope)

        def _next_subtile(s, n_this):
            """Between sub-tiles: this wave's LDS reads land before the next
            sub-tile overwrites the rows; the stores stay in flight."""
            if (s + fx.Int32(1)) < n_this:
                rocdl.s_waitcnt(lgkmcnt=0)
                gpu.barrier()

        def _publish_local(flag_i32, color):
            """Colour the flag at *flag_i32* in every local destination."""
            _drain()
            dest = tid // fx.Int32(FLAG_LANES)
            safe = (dest < fx.Int32(n_push)).select(dest, fx.Int32(0))
            if dest < fx.Int32(n_push):
                elem = flag_i32 + (tid % fx.Int32(FLAG_LANES)) * fx.Int32(
                    FLAG_I32_PER_LANE
                )
                _store_flag_peer(
                    _push_base(safe) + _i32_to_bytes(elem), color, flag_policy
                )

        def _publish_one(dest_base, flag_i32, color):
            """Colour the flag at *flag_i32* in one destination: the partner,
            or the ring successor."""
            _drain()
            if tid < fx.Int32(FLAG_LANES):
                elem = flag_i32 + tid * fx.Int32(FLAG_I32_PER_LANE)
                _store_flag_peer(dest_base + _i32_to_bytes(elem), color, flag_policy)

        def _spin(flag_i32, color):
            flag = self_base + _i32_to_bytes(flag_i32)
            current = _load_flag(flag)
            while current != color:
                current = _load_flag(flag)

        def _acquire():
            # After the join, for every wave; see the mesh's _wait_release.
            gpu.barrier()
            _acquire_inbox(acquire_scope)

        def _wait_one(flag_i32, color):
            """One source: thread 0 spins on the flag at *flag_i32*."""
            if tid == fx.Int32(0):
                _spin(flag_i32, color)
            _acquire()

        def _wait_local(slot_fn, release_i32, color):
            """Lane ``t`` spins on the ``t``-th local source's flag."""
            src = tid
            if tid >= fx.Int32(li):
                src = tid + fx.Int32(1)
            if tid < fx.Int32(n_push):
                _spin(slot_fn(src, fx.Int32(0)) + fx.Int32(release_i32), color)
            _acquire()

        def _own_share(tile, k):
            return _load_atom(tile, li * rank_atoms + k)

        def _add_f16(a, b):
            """Packed fp16 ``a + b`` of two atoms."""
            return fx.Vector.from_elements(
                [_i32(_f16x2(a[i]) + _f16x2(b[i])) for i in range_constexpr(4)],
                fx.Int32,
            )

        def _rs_mesh(i, n_this, color):
            """Lap 1 on the mesh: quantize each local peer's chunk and push
            it; our own chunk is reloaded in lap 2."""
            for s in range(fx.Int32(0), n_this, fx.Int32(1)):
                tile = bid + (i + s) * n_blocks
                for j in range_constexpr(n_push):
                    for k in range_constexpr(rank_atoms):
                        atom = _load_atom(tile, push_locals[j] * rank_atoms + k)
                        words, scale, is_leader = _codec_quant(rs, atom, lane, tid)
                        _lds_write_packet(
                            rs, fx.Int32(j * rank_atoms + k), words, scale, is_leader
                        )
                gpu.barrier()
                _fanout_local(rs, _rs_slot(fx.Int32(li), s))
                _next_subtile(s, n_this)
            _publish_local(
                _rs_slot(fx.Int32(li), fx.Int32(0)) + fx.Int32(rs_release), color
            )
            _wait_local(_rs_slot, rs_release, color)

        def _rs_ring(i, n_this, color):
            """Lap 1 on the ring: hops ``1 .. G-1``. Hop ``h`` adds our share
            of chunk ``li - h`` to the partial that came in on hop ``h - 1``
            (hop 1 starts from our share alone) and sends it into slot
            ``h - 1`` of the successor. The last hop into us carries the
            partial of our own chunk, which lap 2 completes."""
            for _h in range_constexpr(G - 1):
                h = _h + 1
                chunk = (li - h) % G
                if const_expr(h >= 2):
                    _wait_one(
                        _rs_slot(fx.Int32(h - 2), fx.Int32(0)) + fx.Int32(rs_release),
                        color,
                    )
                else:
                    pass  # hop 1 is a pure send
                for s in range(fx.Int32(0), n_this, fx.Int32(1)):
                    tile = bid + (i + s) * n_blocks
                    for k in range_constexpr(rank_atoms):
                        acc = _load_atom(tile, chunk * rank_atoms + k)
                        if const_expr(h >= 2):
                            words, scale = _recv(
                                rs,
                                _rs_slot(fx.Int32(h - 2), s)
                                + fx.Int32(k * rs.rank_tile_i32),
                            )
                            acc = _codec_dequant(rs, words, scale, tid, acc)
                        else:
                            pass
                        words, scale, is_leader = _codec_quant(rs, acc, lane, tid)
                        _lds_write_packet(rs, fx.Int32(k), words, scale, is_leader)
                    gpu.barrier()
                    _store_rows(rs, next_base, _rs_slot(fx.Int32(h - 1), s))
                    _next_subtile(s, n_this)
                _publish_one(
                    next_base,
                    _rs_slot(fx.Int32(h - 1), fx.Int32(0)) + fx.Int32(rs_release),
                    color,
                )
            _wait_one(
                _rs_slot(fx.Int32(G - 2), fx.Int32(0)) + fx.Int32(rs_release), color
            )

        def _group_partial(tile, s, k, polled=None):
            """Our group's partial sum of atom *k* of our own chunk.

            *polled* is ``_poll_rs``'s result on the Lamport handshake; on the
            flag handshake the packets are read here."""

            def _packet(idx):
                if const_expr(polled is None):
                    return _recv(
                        rs, _rs_slot(fx.Int32(idx), s) + fx.Int32(k * rs.rank_tile_i32)
                    )
                words, word = polled[(idx, k)]
                return words, _scale_from_word(rs, word, pair_in_slot)

            if const_expr(rs_ring):
                # The ring's last hop: the other G - 1 contributions, plus ours.
                words, scale = _packet(G - 2)
                acc = _codec_dequant(rs, words, scale, tid, _own_share(tile, k))
            else:
                # In local-index order.
                acc = None
                for m in range_constexpr(G):
                    if const_expr(m == li):
                        own = _own_share(tile, k)
                        if const_expr(acc is None):
                            acc = own
                        else:
                            acc = _add_f16(own, acc)
                    else:
                        words, scale = _packet(m)
                        acc = _codec_dequant(rs, words, scale, tid, acc)
            return acc

        def _ag_mesh(i, n_this, color):
            """Lap 3's handshake and the gather on the mesh: our packet is
            already in every local peer's inbox."""
            _publish_local(
                _ag_slot(fx.Int32(li), fx.Int32(0)) + fx.Int32(ag_release), color
            )
            _wait_local(_ag_slot, ag_release, color)
            for s in range(fx.Int32(0), n_this, fx.Int32(1)):
                tile = bid + (i + s) * n_blocks
                gathered = []
                for m in range_constexpr(G):
                    for k in range_constexpr(rank_atoms):
                        if const_expr(m == li):
                            gathered.append(_unstash_own(tile, li * rank_atoms + k))
                        else:
                            words, scale = _recv(
                                ag,
                                _ag_slot(fx.Int32(m), s)
                                + fx.Int32(k * ag.rank_tile_i32),
                            )
                            gathered.append(_codec_dequant(ag, words, scale, tid))
                _finish_tile(tile, gathered, None)

        def _ag_ring(i, n_this, color):
            """Lap 3's handshake and hops ``2 .. G`` on the ring: our packet
            went into slot 0 of the successor as hop 1. Hop ``h`` brings
            chunk ``li - h + 1`` into slot ``h - 2``: decode it into the
            output and forward the bytes untouched into slot ``h - 1`` of
            the successor, except on the last hop, whose chunk the successor
            owns."""
            _publish_one(
                next_base,
                _ag_slot(fx.Int32(0), fx.Int32(0)) + fx.Int32(ag_release),
                color,
            )
            is_leader = (tid % fx.Int32(GROUP)) == fx.Int32(0)
            for _h in range_constexpr(G - 1):
                h = _h + 2
                chunk = (li - h + 1) % G
                _wait_one(
                    _ag_slot(fx.Int32(h - 2), fx.Int32(0)) + fx.Int32(ag_release),
                    color,
                )
                for s in range(fx.Int32(0), n_this, fx.Int32(1)):
                    tile = bid + (i + s) * n_blocks
                    for k in range_constexpr(rank_atoms):
                        words, word = _recv_raw(
                            ag,
                            _ag_slot(fx.Int32(h - 2), s)
                            + fx.Int32(k * ag.rank_tile_i32),
                        )
                        _complete(
                            tile,
                            chunk * rank_atoms + k,
                            _codec_dequant(
                                ag, words, _scale_from_word(ag, word, pair_in_slot), tid
                            ),
                            None,
                        )
                        if const_expr(h < G):
                            _lds_write_packet(ag, fx.Int32(k), words, word, is_leader)
                        else:
                            pass
                    if const_expr(h < G):
                        gpu.barrier()
                        _store_rows(ag, next_base, _ag_slot(fx.Int32(h - 1), s))
                        _next_subtile(s, n_this)
                    else:
                        pass  # the last hop receives only
                if const_expr(h < G):
                    _publish_one(
                        next_base,
                        _ag_slot(fx.Int32(h - 1), fx.Int32(0)) + fx.Int32(ag_release),
                        color,
                    )
                else:
                    pass

        # ---- Lamport handshake; see the module docstring. ----

        rearm_vec = fx.Vector.from_elements([fx.Int32(LAMPORT_SENTINEL)] * 4, fx.Int32)

        def _poll(c, bases):
            """``[(words, scale_word)]`` of the packets of codec *c* at the i32
            offsets *bases* of our inbox, raw as ``_recv_raw`` returns them,
            once none of the dwords this thread reads is the sentinel.

            As the mesh's Lamport receive: every dword is read before any is
            tested, past L1 and L2, then all are re-read together with relaxed
            system-scope atomics until every one has landed. There is no
            re-arm here (``_rearm``); ``vmcnt(0)`` drains every store issued
            before, the re-arms among them, which the buffer argument needs.
            """
            slots = lamport_slots(c, tid, scale_slot)
            offs = [b + off for b in bases for off, _pred in slots]
            got = fx.Vector.from_elements(
                [_buffer_load(inbox, o, 1, fx.Int32, _RECV_POLICY)[0] for o in offs],
                fx.Int32,
            )
            addrs = [self_base + _i32_to_bytes(o) for o in offs]
            while _any_sentinel(got):
                got = fx.Vector.from_elements([_load_flag(a) for a in addrs], fx.Int32)
            rocdl.s_waitcnt(vmcnt=0)
            per = len(slots)
            return [
                split_lamport_packet(
                    c, [got[i * per + j] for j in range_constexpr(per)]
                )
                for i in range_constexpr(len(bases))
            ]

        def _rearm(base_i32, n_i32):
            """Put the sentinel back in *n_i32* dwords of our inbox from
            *base_i32*, 16 B per lane (``lamport_rearm_modifier``)."""
            n_vec = fx.Int32(n_i32 // 4)
            for v0 in range(fx.Int32(0), n_vec, fx.Int32(block)):
                v = v0 + tid
                if v < n_vec:
                    _buffer_store(
                        inbox, base_i32 + v * fx.Int32(4), rearm_vec, rearm_cm
                    )

        def _rearm_local(buf):
            """Every local slot of this block in buffer *buf*."""
            _rearm(_rs_slot(fx.Int32(0), fx.Int32(0), buf), rs_n * rs_slot)
            _rearm(_ag_slot(fx.Int32(0), fx.Int32(0), buf), ag_n * ag_slot)

        def _rearm_cross(buf):
            """The partner's cross slot of this block in buffer *buf*."""
            _rearm(_x_slot(buf, 1 - my_group, fx.Int32(0)), x_slot)

        def _pushed():
            """After a Lamport lap's (or ring hop's) pushes: on a cacheable
            inbox this wave writes them back out of L2. No barrier, no flag."""
            if const_expr(release_scope is not None):
                _release_inbox(release_scope)

        def _lds_free():
            """Join the workgroup once this wave's LDS reads have landed, so
            the next pack cannot overwrite rows a push still reads. The flag
            handshake gets this join from its publish and wait."""
            rocdl.s_waitcnt(lgkmcnt=0)
            gpu.barrier()

        def _rank_tiles(slot_fn, idxs, s, c, buf):
            """``(keys, bases)``: ``(idx, k)`` and the i32 offset of rank-tile
            *k* of sub-tile *s* in slot *idx*, for every *idx* of *idxs*."""
            keys = [(m, k) for m in idxs for k in range_constexpr(rank_atoms)]
            bases = [
                slot_fn(fx.Int32(m), s, buf) + fx.Int32(k * c.rank_tile_i32)
                for m, k in keys
            ]
            return keys, bases

        def _poll_rs(s, buf):
            """The reduce-scatter packets lap 2 reads for sub-tile *s*,
            ``{(slot, k): (words, scale_word)}``: the ring's last hop, or every
            other local rank's on the mesh."""
            idxs = [G - 2] if rs_ring else push_locals
            keys, bases = _rank_tiles(_rs_slot, idxs, s, rs, buf)
            return dict(zip(keys, _poll(rs, bases)))

        def _recv_own(c, base):
            """Our own cross packet, which this block stored into our own
            inbox in lap 2: from L2 (``sc1``), where its ``nt`` stores may
            still sit."""

            def _get(off):
                return _buffer_load(inbox, base + off, 1, fx.Int32, _CM_SC1)[0]

            return _codec_load(c, _get, tid, scale_slot)

        def _rs_mesh_lamport(i, n_this, buf):
            """Lap 1 on the mesh: as ``_rs_mesh``, without the handshake."""
            for s in range(fx.Int32(0), n_this, fx.Int32(1)):
                tile = bid + (i + s) * n_blocks
                _lds_free()
                for j in range_constexpr(n_push):
                    for k in range_constexpr(rank_atoms):
                        atom = _load_atom(tile, push_locals[j] * rank_atoms + k)
                        words, scale, is_leader = _codec_quant(rs, atom, lane, tid)
                        words = lamport_fix(rs, words, tid)
                        _lds_write_packet(
                            rs, fx.Int32(j * rank_atoms + k), words, scale, is_leader
                        )
                gpu.barrier()
                _fanout_local(rs, _rs_slot(fx.Int32(li), s, buf))
            _pushed()

        def _rs_ring_lamport(i, n_this, buf):
            """Lap 1 on the ring: as ``_rs_ring``, each sub-tile of hop ``h``
            starting as soon as the same sub-tile of hop ``h - 1`` lands."""
            for _h in range_constexpr(G - 1):
                h = _h + 1
                chunk = (li - h) % G
                for s in range(fx.Int32(0), n_this, fx.Int32(1)):
                    tile = bid + (i + s) * n_blocks
                    if const_expr(h >= 2):
                        _keys, bases = _rank_tiles(_rs_slot, [h - 2], s, rs, buf)
                        polled = _poll(rs, bases)
                    else:
                        pass  # hop 1 is a pure send
                    _lds_free()
                    for k in range_constexpr(rank_atoms):
                        acc = _load_atom(tile, chunk * rank_atoms + k)
                        if const_expr(h >= 2):
                            words, word = polled[k]
                            acc = _codec_dequant(
                                rs,
                                words,
                                _scale_from_word(rs, word, pair_in_slot),
                                tid,
                                acc,
                            )
                        else:
                            pass
                        words, scale, is_leader = _codec_quant(rs, acc, lane, tid)
                        words = lamport_fix(rs, words, tid)
                        _lds_write_packet(rs, fx.Int32(k), words, scale, is_leader)
                    gpu.barrier()
                    _store_rows(rs, next_base, _rs_slot(fx.Int32(h - 1), s, buf))
                _pushed()

        def _ag_mesh_lamport(i, n_this, buf):
            """The gather on the mesh: every local peer's packet of each
            sub-tile, as soon as it lands."""
            for s in range(fx.Int32(0), n_this, fx.Int32(1)):
                tile = bid + (i + s) * n_blocks
                keys, bases = _rank_tiles(_ag_slot, push_locals, s, ag, buf)
                polled = dict(zip(keys, _poll(ag, bases)))
                gathered = []
                for m in range_constexpr(G):
                    for k in range_constexpr(rank_atoms):
                        if const_expr(m == li):
                            gathered.append(_unstash_own(tile, li * rank_atoms + k))
                        else:
                            words, word = polled[(m, k)]
                            gathered.append(
                                _codec_dequant(
                                    ag,
                                    words,
                                    _scale_from_word(ag, word, pair_in_slot),
                                    tid,
                                )
                            )
                _finish_tile(tile, gathered, None)

        def _ag_ring_lamport(i, n_this, buf):
            """Hops ``2 .. G`` on the ring: as ``_ag_ring``, sub-tile by
            sub-tile. The forwarded bytes are already off the sentinel."""
            is_leader = (tid % fx.Int32(GROUP)) == fx.Int32(0)
            for _h in range_constexpr(G - 1):
                h = _h + 2
                chunk = (li - h + 1) % G
                for s in range(fx.Int32(0), n_this, fx.Int32(1)):
                    tile = bid + (i + s) * n_blocks
                    _keys, bases = _rank_tiles(_ag_slot, [h - 2], s, ag, buf)
                    polled = _poll(ag, bases)
                    if const_expr(h < G):
                        _lds_free()
                    else:
                        pass
                    for k in range_constexpr(rank_atoms):
                        words, word = polled[k]
                        _complete(
                            tile,
                            chunk * rank_atoms + k,
                            _codec_dequant(
                                ag, words, _scale_from_word(ag, word, pair_in_slot), tid
                            ),
                            None,
                        )
                        if const_expr(h < G):
                            _lds_write_packet(ag, fx.Int32(k), words, word, is_leader)
                        else:
                            pass
                    if const_expr(h < G):
                        gpu.barrier()
                        _store_rows(ag, next_base, _ag_slot(fx.Int32(h - 1), s, buf))
                    else:
                        pass  # the last hop receives only
                if const_expr(h < G):
                    _pushed()
                else:
                    pass

        def _main_lamport():
            n_block_tiles = (num_tiles - bid + n_blocks - fx.Int32(1)) // n_blocks
            color = _load_color()
            st_i = fx.Int32(super_tile)
            for i in range(fx.Int32(0), n_block_tiles, st_i):
                remain = n_block_tiles - i
                n_this = (remain < st_i).select(remain, st_i)
                lbuf = color & fx.Int32(1)
                xbuf = color % fx.Int32(_LAMPORT_X_BUFFERS)

                # Lap 1: local reduce-scatter.
                if const_expr(rs_ring):
                    _rs_ring_lamport(i, n_this, lbuf)
                else:
                    _rs_mesh_lamport(i, n_this, lbuf)

                # Lap 2: each sub-tile's group partial as soon as its
                # reduce-scatter packets land, to the partner and to our own
                # cross slot.
                for s in range(fx.Int32(0), n_this, fx.Int32(1)):
                    tile = bid + (i + s) * n_blocks
                    polled = _poll_rs(s, lbuf)
                    _lds_free()
                    for k in range_constexpr(rank_atoms):
                        acc = _group_partial(tile, s, k, polled)
                        words, scale, is_leader = _codec_quant(xc, acc, lane, tid)
                        words = lamport_fix(xc, words, tid)
                        _lds_write_packet(xc, fx.Int32(k), words, scale, is_leader)
                    gpu.barrier()
                    _store_rows(xc, partner_base, _x_slot(xbuf, my_group, s))
                    _store_rows(xc, self_base, _x_slot(xbuf, my_group, s))
                _pushed()
                # This group's reduce-scatter has landed from every local
                # rank: the previous group's local slots can be re-armed.
                _rearm_local(fx.Int32(1) - lbuf)

                # Lap 3: the final sum, group 0's packet first on both
                # partners, as on the flag handshake. The poll's barrier
                # (``_lds_free``) follows every wave's ``vmcnt(0)``, so our own
                # cross packet is in L2 by the time any wave reads it.
                for s in range(fx.Int32(0), n_this, fx.Int32(1)):
                    tile = bid + (i + s) * n_blocks
                    theirs = _poll(
                        xc,
                        [
                            _x_slot(xbuf, 1 - my_group, s)
                            + fx.Int32(k * xc.rank_tile_i32)
                            for k in range_constexpr(rank_atoms)
                        ],
                    )
                    _lds_free()
                    for k in range_constexpr(rank_atoms):
                        acc = None
                        for g in range_constexpr(2):
                            if const_expr(g == my_group):
                                words, word = _recv_own(
                                    xc,
                                    _x_slot(xbuf, g, s)
                                    + fx.Int32(k * xc.rank_tile_i32),
                                )
                            else:
                                words, word = theirs[k]
                            acc = _codec_dequant(
                                xc,
                                words,
                                _scale_from_word(xc, word, pair_in_slot),
                                tid,
                                acc,
                            )
                        words, scale, is_leader = _codec_quant(ag, acc, lane, tid)
                        # Before our own decode below, so it decodes what the
                        # peers do.
                        words = lamport_fix(ag, words, tid)
                        for j in range_constexpr(_local_rows(local_ag, G)):
                            _lds_write_packet(
                                ag,
                                fx.Int32(j * rank_atoms + k),
                                words,
                                scale,
                                is_leader,
                            )
                        _stash_own(
                            tile,
                            li * rank_atoms + k,
                            _codec_dequant(
                                ag,
                                words,
                                _scale_from_word(ag, scale, pair_in_slot),
                                tid,
                            ),
                        )
                    gpu.barrier()
                    if const_expr(ag_ring):
                        _store_rows(ag, next_base, _ag_slot(fx.Int32(0), s, lbuf))
                    else:
                        _fanout_local(ag, _ag_slot(fx.Int32(li), s, lbuf))
                _pushed()
                # The partner's packet of this group has landed: its previous
                # cross slot can be re-armed.
                _rearm_cross(
                    (xbuf + fx.Int32(_LAMPORT_X_BUFFERS - 1))
                    % fx.Int32(_LAMPORT_X_BUFFERS)
                )

                if const_expr(ag_ring):
                    _ag_ring_lamport(i, n_this, lbuf)
                else:
                    _ag_mesh_lamport(i, n_this, lbuf)

                color = color + fx.Int32(1)
                if color == fx.Int32(_LAMPORT_COLORS):
                    color = fx.Int32(0)
            if tid == 0:
                _store_color(color)
            gpu.barrier()

        if const_expr(lamport):
            _main_lamport()
        else:
            n_block_tiles = (num_tiles - bid + n_blocks - fx.Int32(1)) // n_blocks
            color = _load_color()
            st_i = fx.Int32(super_tile)
            for i in range(fx.Int32(0), n_block_tiles, st_i):
                remain = n_block_tiles - i
                n_this = (remain < st_i).select(remain, st_i)
                parity = color & fx.Int32(1)

                # Lap 1: local reduce-scatter.
                if const_expr(rs_ring):
                    _rs_ring(i, n_this, color)
                else:
                    _rs_mesh(i, n_this, color)

                # Lap 2: our group's partial of our own chunk, sent to the partner
                # and to our own cross slot.
                for s in range(fx.Int32(0), n_this, fx.Int32(1)):
                    tile = bid + (i + s) * n_blocks
                    for k in range_constexpr(rank_atoms):
                        acc = _group_partial(tile, s, k)
                        words, scale, is_leader = _codec_quant(xc, acc, lane, tid)
                        _lds_write_packet(xc, fx.Int32(k), words, scale, is_leader)
                    gpu.barrier()
                    _store_rows(xc, partner_base, _x_slot(parity, my_group, s))
                    _store_rows(xc, self_base, _x_slot(parity, my_group, s))
                    _next_subtile(s, n_this)
                _publish_one(
                    partner_base,
                    _x_slot(parity, my_group, fx.Int32(0)) + fx.Int32(x_release),
                    color,
                )
                _wait_one(
                    _x_slot(parity, 1 - my_group, fx.Int32(0)) + fx.Int32(x_release),
                    color,
                )

                # Lap 3: the final sum, group 0's packet first on both partners,
                # quantized once. Our own chunk is decoded from the same packet
                # and stored now. The packet goes to every local peer (mesh) or to
                # the successor as the all-gather's first hop (ring).
                for s in range(fx.Int32(0), n_this, fx.Int32(1)):
                    tile = bid + (i + s) * n_blocks
                    for k in range_constexpr(rank_atoms):
                        acc = None
                        for g in range_constexpr(2):
                            words, scale = _recv(
                                xc,
                                _x_slot(parity, g, s) + fx.Int32(k * xc.rank_tile_i32),
                            )
                            acc = _codec_dequant(xc, words, scale, tid, acc)
                        words, scale, is_leader = _codec_quant(ag, acc, lane, tid)
                        for j in range_constexpr(_local_rows(local_ag, G)):
                            _lds_write_packet(
                                ag,
                                fx.Int32(j * rank_atoms + k),
                                words,
                                scale,
                                is_leader,
                            )
                        _stash_own(
                            tile,
                            li * rank_atoms + k,
                            _codec_dequant(
                                ag,
                                words,
                                _scale_from_word(ag, scale, pair_in_slot),
                                tid,
                            ),
                        )
                    gpu.barrier()
                    if const_expr(ag_ring):
                        _store_rows(ag, next_base, _ag_slot(fx.Int32(0), s))
                    else:
                        _fanout_local(ag, _ag_slot(fx.Int32(li), s))
                    _next_subtile(s, n_this)
                if const_expr(ag_ring):
                    _ag_ring(i, n_this, color)
                else:
                    _ag_mesh(i, n_this, color)

                color = color + fx.Int32(1)
                # 0 is the unset sentinel. Wrap to 2, not 1, so that the parity
                # keeps alternating and the cross slots stay double-buffered.
                if color == fx.Int32(0):
                    color = fx.Int32(2)
            if tid == 0:
                _store_color(color)
            gpu.barrier()

    flat_wg = f"{block},{block}"

    @flyc.jit
    def launch_quick_allreduce_hier(
        rank_arg: fx.Int32,
        nbytes: fx.Int64,
        num_tiles: fx.Int32,
        inp_ptr: fx.Int64,
        out_ptr: fx.Int64,
        peer_ptrs: fx.Int64,
        colors_ptr: fx.Int64,
        grid_x: fx.Int32,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        quick_allreduce_hier(
            rank_arg,
            nbytes,
            num_tiles,
            inp_ptr,
            out_ptr,
            peer_ptrs,
            colors_ptr,
            grid_x,
            value_attrs={"rocdl.flat_work_group_size": flat_wg},
        ).launch(
            grid=(grid_x, 1, 1),
            block=(block, 1, 1),
            stream=stream,
        )

    # The groups, rank and codecs are all baked into the binary.
    gtag = "_".join("".join(str(r) for r in g) for g in groups)
    tag = (
        f"{SCALE_CODEC_TAG}_ws{world_size}_g{gtag}_r{self_rank}_st{super_tile}"
        f"_g{grid}_{inbox_memory}_{rs_codec}_{x_codec}_{ag_codec}_b{block}"
        f"_{local_rs}_{local_ag}"
    )
    if lamport:
        tag += "_lamport"
    launch_quick_allreduce_hier.func.__name__ = f"launch_quick_allreduce_hier_{tag}"
    try:
        quick_allreduce_hier.func.__name__ = f"quick_allreduce_hier_{tag}"
    except AttributeError:
        pass
    return {
        "launch": launch_quick_allreduce_hier,
        # The flag handshakes ride in each slot's 64 B tail; Lamport has none.
        "flags_bytes": 0,
        "data_bytes": inbox_i32 * 4,
        "lds_bytes": lds_bytes,
        "tile_bytes": tile_bytes,
        "hbm_tile_bytes": epilogue.hbm_tile_bytes,
        "tile_fp16": tile_bytes // 2,
        "rank_tile_bytes": rs.rank_tile_bytes,
        "wire_tile_bytes": rs_slot * 4,
        "super_tile": super_tile,
        "world_size": world_size,
        "groups": groups,
        "group_size": G,
        "partner": partner,
        "inbox_memory": inbox_memory,
        "rs_codec": rs_codec,
        "x_codec": x_codec,
        "ag_codec": ag_codec,
        "local_rs": local_rs,
        "local_ag": local_ag,
        "payload_policy": payload_policy,
        "flag_policy": flag_policy,
        "release_scope": release_scope,
        "rank_atoms": rank_atoms,
        "grid": grid,
        "block": block,
        "rank": self_rank,
        "handshake": handshake,
        # Byte every inbox byte starts as: the Lamport sentinel, or 0.
        "inbox_fill": LAMPORT_FILL_BYTE if lamport else 0,
        **epilogue.spec(),
    }
