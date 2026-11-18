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
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T

from ...allreduce_policy import HIER_CODECS, HIER_LOCAL_SCHEDULES, HIER_SUPER_TILES
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
    scale_slot_of,
    thread_lane,
)
from .quick_allreduce_epilogue import QuickReducePlainEpilogue
from .transport import (
    _CM_SC0,
    _CM_SC1,
    _INBOX_POLICY,
    _RECV_POLICY,
    ATOMS,
    FLAG_I32_PER_LANE,
    FLAG_LANES,
    QUAD_LANES,
    QUADS_PER_WAVE,
    WAVE,
    _acquire_inbox,
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
    make_pack_storage,
)

# The one 64 B handshake sector at the tail of every slot, in i32.
_FLAG_SECTOR_I32 = 16

# Cross-slot buffers, rotated by the colour. The flag exchange needs two (the
# module docstring); the mover's pipelined schedule lets the partner run up to
# three groups ahead of our read, so it needs four.
_X_BUFFERS_FLAG = 2
_X_BUFFERS_MOVER = 4

# The mover's Lamport sentinel: every dword of an unwritten cross slot. The
# inbox is filled with it (``inbox_fill``) and a reader re-arms what it read.
X_SENTINEL_FILL_BYTE = 0xFF
_X_SENTINEL = -1

# Mover copy: 16 B per lane, this many loads in flight per lane.
_MOVER_UNROLL = 8


def _lamport_fix(codec, words, tid):
    """*words* of one cross packet, with no dword equal to the sentinel.

    The reader takes a dword as landed once it is not ``0xFFFFFFFF``, so the
    producer must never write that value. Per plane:

    * Scale: four signed E5M2 bytes of fp16 extrema, exponent field at most
      28, so no byte is ``0xFF``. Never the sentinel.
    * INT6 extra plane: the dword is one scale group (a lane pair). The group's
      extremum maps to ``q <= 0``, code ``<= 32``, extra field ``<= 2``; all
      sixteen fields 3 cannot happen.
    * INT5 extra plane: the dword is two scale groups (a lane quartet), one bit
      per value. All ones means both groups quantize to ``q = 0`` (every code
      16), so the four threads' nibble words are 0. The quartet's leader moves
      its value 0 from code 16 to 15 (``q = -1``): one step of a scale that is
      0, or the INT5 floor ``2**-11`` when the extremum is below ``2**-16``.
    * Nibble plane: all eight low nibbles 15 can happen. Flip bit 0, one code
      step on value 0 of this thread.
    * FP16 passthrough: two NaN halves ``0xFFFF``; becomes ``0x7FFF`` NaN.

    Both partners decode the same fixed bytes, so the sums stay identical.
    """
    sentinel = fx.Int32(_X_SENTINEL)
    if not codec.has_scale:
        return tuple(
            (w == sentinel).select(fx.Int32(0x7FFF7FFF), w) for w in words
        )
    nib = words[0]
    rest = tuple(words[1:])
    if codec.hi2_i32_off is not None and codec.hi_bits == 1:
        hi = words[1]
        hit = hi == sentinel
        leader = (tid & fx.Int32(codec.hi_share - 1)) == fx.Int32(0)
        nib = hit.select(leader.select(nib | fx.Int32(0xF), nib), nib)
        rest = (hit.select(hi & fx.Int32(-2), hi),)
    nib = (nib == sentinel).select(nib ^ fx.Int32(1), nib)
    return (nib, *rest)


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
    x_mover: bool = False,
):
    """Build the hierarchical kernel for one *rank*.

    *groups* is two tuples of ranks; partners sit at the same position of the
    sorted groups. The rank, its group, its local peers and its partner are
    all trace-time constants, so this is one binary per rank. *local_rs* and
    *local_ag* pick the schedule of the two local laps, ``"mesh"`` or
    ``"ring"``.

    *x_mover* moves the cross exchange off the compute workgroups (PROTOTYPE).
    The launch adds ``ceil(n_blocks / waves)`` mover workgroups after the
    ``n_blocks`` compute ones; mover wave ``w`` serves compute block ``w``.

    * The schedule is pipelined: a block runs lap 1 and 2 of group ``g``, then
      lap 3 of group ``g - 1``. Lap 2 writes only our own cross slot, drains,
      and colours a ready flag in it.
    * The mover wave waits for that flag and copies the slot, verbatim, into
      the partner's inbox. Its stores are outside the compute waves'
      ``vmcnt`` and fences, so the exchange stays in flight under the next
      group's laps. They are written through (``sc0 sc1``). ``nt`` stores and
      a release per slot are not safe here: a line can reach the partner by
      eviction before the release, the partner reads and re-arms it, and the
      release's writeback of the same line puts the old bytes back over the
      sentinel, read as landed four groups later (measured: rare cross-rank
      differences under run-ahead; write-through passes).
    * No flag goes with the data (a Lamport exchange): an unwritten slot holds
      the sentinel in every dword (``_lamport_fix`` keeps the producer off
      it), lap 3 polls the partner's packet until no dword is the sentinel,
      and re-arms what it read, write-through.
    * Four cross buffers (``_X_BUFFERS_MOVER``), and the colour wraps to 4.
      Mover wave ``w`` keeps its own copy of block ``w``'s colour at
      ``colors[grid + w]`` (``color_slots``).
    """
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
    # rank-tiles, then the 64 B handshake sector. Every rank-tile is a whole
    # number of sectors, so every slot and region stays 64 B aligned.
    def _slot(c):
        payload = rank_atoms * c.rank_tile_i32
        return payload, super_tile * payload, super_tile * payload + _FLAG_SECTOR_I32

    rs_payload, rs_release, rs_slot = _slot(rs)
    x_payload, x_release, x_slot = _slot(xc)
    ag_payload, ag_release, ag_slot = _slot(ag)
    # A local lap's slots per block: on the mesh one per source local index,
    # on the ring one per hop, each written only by the ring predecessor.
    rs_n = G - 1 if rs_ring else G
    ag_n = G - 1 if ag_ring else G
    # LOCAL_RS [grid][rs_n], CROSS [x_buffers][grid][2 groups],
    # LOCAL_AG [grid][ag_n].
    x_buffers = _X_BUFFERS_MOVER if x_mover else _X_BUFFERS_FLAG
    # The colour skips 0 (unset); wrapping to the buffer count keeps
    # ``color % x_buffers`` rotating through 0xFFFFFFFF.
    color_wrap = x_buffers
    waves = block // WAVE
    # The flag schedule's single-colour argument for the local slots needs an
    # all-gather between two reduce-scatters and a reduce-scatter between two
    # all-gathers. The mover's pipelined order breaks both: a block's first
    # iteration runs RS(g + 1) right after RS(g), and its last runs AG(L)
    # right after AG(L - 1). So both local laps are double-buffered by colour
    # parity there: a peer's lap ``j + 2`` needs its lap ``j + 1``, which needs
    # our send of lap ``j + 1``, which comes after our reads of lap ``j``.
    rs_bufs = ag_bufs = 2 if x_mover else 1
    rs_region = grid * rs_n * rs_slot
    ag_region = grid * ag_n * ag_slot
    rs_base = 0
    x_base = rs_base + rs_bufs * rs_region
    ag_base = x_base + x_buffers * grid * 2 * x_slot
    inbox_i32 = ag_base + ag_bufs * ag_region

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
        def _local_slot(base, slot, payload, n, idx, sub):
            return (
                fx.Int32(base)
                + (bid * fx.Int32(n) + idx) * fx.Int32(slot)
                + sub * fx.Int32(payload)
            )

        # The reduce-scatter buffer of the group being reduced (mover only):
        # set from its colour before each lap 1 and lap 2.
        rs_cur = [None]

        def _use_rs_buffer(color):
            # A setter, not an assignment in the loop body: FlyDSL's loop
            # rewriter would try to carry ``rs_cur`` through the scf.for.
            rs_cur[0] = color & fx.Int32(1)

        def _rs_slot(idx, sub):
            """Mesh: *idx* is the source local index. Ring: the hop."""
            off = _local_slot(rs_base, rs_slot, rs_payload, rs_n, idx, sub)
            if const_expr(rs_bufs > 1):
                off = off + rs_cur[0] * fx.Int32(rs_region)
            return off

        ag_cur = [None]

        def _use_ag_buffer(color):
            ag_cur[0] = color & fx.Int32(1)

        def _ag_slot(idx, sub):
            off = _local_slot(ag_base, ag_slot, ag_payload, ag_n, idx, sub)
            if const_expr(ag_bufs > 1):
                off = off + ag_cur[0] * fx.Int32(ag_region)
            return off

        def _x_slot(parity, grp, sub, blk=None):
            """*parity* is the buffer: ``color % x_buffers``. *blk* is the
            compute block that owns the slot (the mover serves another one)."""
            blk = bid if blk is None else blk
            return (
                fx.Int32(x_base)
                + ((parity * fx.Int32(grid) + blk) * fx.Int32(2) + fx.Int32(grp))
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

        def _group_partial(tile, s, k):
            """Our group's partial sum of atom *k* of our own chunk."""
            if const_expr(rs_ring):
                # The ring's last hop: the other G - 1 contributions, plus ours.
                words, scale = _recv(
                    rs, _rs_slot(fx.Int32(G - 2), s) + fx.Int32(k * rs.rank_tile_i32)
                )
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
                        words, scale = _recv(
                            rs,
                            _rs_slot(fx.Int32(m), s) + fx.Int32(k * rs.rank_tile_i32),
                        )
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

        # ---- The mover schedule (x_mover); see the builder's docstring. ----

        def _x_buf(color):
            return color & fx.Int32(x_buffers - 1)

        def _next_color(color):
            color = color + fx.Int32(1)
            return (color == fx.Int32(0)).select(fx.Int32(color_wrap), color)

        def _x_lap2_self(i, n_this, color):
            """Lap 2 on the mover schedule: our group's partial of our own
            chunk into our own cross slot only, then a drain and the ready
            flag (in that slot's tail) that our mover wave waits for."""
            buf = _x_buf(color)
            for s in range(fx.Int32(0), n_this, fx.Int32(1)):
                tile = bid + (i + s) * n_blocks
                for k in range_constexpr(rank_atoms):
                    acc = _group_partial(tile, s, k)
                    words, scale, is_leader = _codec_quant(xc, acc, lane, tid)
                    words = _lamport_fix(xc, words, tid)
                    _lds_write_packet(xc, fx.Int32(k), words, scale, is_leader)
                gpu.barrier()
                _store_rows(xc, self_base, _x_slot(buf, my_group, s))
                _next_subtile(s, n_this)
            _publish_one(
                self_base,
                _x_slot(buf, my_group, fx.Int32(0)) + fx.Int32(x_release),
                color,
            )

        def _any_sentinel(vec):
            hit = fx.Int32(0)
            for e in range_constexpr(vec.numel):
                hit = hit | (vec[e] == fx.Int32(_X_SENTINEL)).select(
                    fx.Int32(1), fx.Int32(0)
                )
            return hit != fx.Int32(0)

        def _x_poll(buf, s):
            """The partner's packets of sub-tile *s*, ``[(words, scale_word)]``
            per rank-tile, once no dword this thread reads is the sentinel.

            Every dword is re-read on each pass, together, as in the one-shot's
            poll; the relaxed system-scope loads (``_load_flag``) go past L1
            and L2 and keep the spin alive. Then the dwords are re-armed,
            write-through (an ``nt`` store could sit dirty in our L2 and
            overwrite the partner's next packet on eviction). A shared dword
            (extra plane, scale) is read by lanes of one wave, which all leave
            the spin before any of them re-arms; its owner re-arms it.
            """
            scale_owner = (tid % fx.Int32(GROUP)) == fx.Int32(0)
            offs = []
            for k in range_constexpr(rank_atoms):
                base = _x_slot(buf, 1 - my_group, s) + fx.Int32(k * xc.rank_tile_i32)
                for off, pred in xc.plane_slots(tid):
                    offs.append((base + off, pred))
                if const_expr(xc.has_scale):
                    offs.append(
                        (base + fx.Int32(xc.scale_i32_off) + scale_slot, scale_owner)
                    )
            addrs = [self_base + _i32_to_bytes(o) for o, _pred in offs]

            def _load_all():
                return fx.Vector.from_elements(
                    [_load_flag(a) for a in addrs], fx.Int32
                )

            got = _load_all()
            while _any_sentinel(got):
                got = _load_all()
            armed = fx.Vector.from_elements([fx.Int32(_X_SENTINEL)], fx.Int32)
            for o, pred in offs:
                if pred:
                    _buffer_store(inbox, o, armed, _CM_SC0 | _CM_SC1)
            per = xc.n_words_per_thread + (1 if xc.has_scale else 0)
            out = []
            for k in range_constexpr(rank_atoms):
                words = tuple(
                    got[k * per + w] for w in range_constexpr(xc.n_words_per_thread)
                )
                word = got[k * per + per - 1] if xc.has_scale else None
                out.append((words, word))
            return out

        def _recv_own(c, base):
            """A packet of our own, written by this block: read past L1/L2."""

            def _get(off):
                return _buffer_load(inbox, base + off, 1, fx.Int32, _RECV_POLICY)[0]

            return _codec_load(c, _get, tid, scale_slot)

        def _x_lap3(i, n_this, color):
            """Lap 3 and the all-gather on the mover schedule: as the flag
            schedule's, with the partner's packet polled (``_x_poll``)."""
            buf = _x_buf(color)
            _use_ag_buffer(color)
            for s in range(fx.Int32(0), n_this, fx.Int32(1)):
                tile = bid + (i + s) * n_blocks
                theirs = _x_poll(buf, s)
                for k in range_constexpr(rank_atoms):
                    acc = None
                    for g in range_constexpr(2):
                        if const_expr(g == my_group):
                            words, word = _recv_own(
                                xc,
                                _x_slot(buf, g, s) + fx.Int32(k * xc.rank_tile_i32),
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
                    for j in range_constexpr(_local_rows(local_ag, G)):
                        _lds_write_packet(
                            ag, fx.Int32(j * rank_atoms + k), words, scale, is_leader
                        )
                    _stash_own(
                        tile,
                        li * rank_atoms + k,
                        _codec_dequant(
                            ag, words, _scale_from_word(ag, scale, pair_in_slot), tid
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

        def _compute_pipelined():
            """A compute block: laps 1-2 of group ``g``, then lap 3 of
            ``g - 1``; the last group's lap 3 after the loop."""
            n_block_tiles = (num_tiles - bid + n_blocks - fx.Int32(1)) // n_blocks
            st_i = fx.Int32(super_tile)
            color = _load_color()
            pcolor = color
            for i in range(fx.Int32(0), n_block_tiles, st_i):
                remain = n_block_tiles - i
                n_this = (remain < st_i).select(remain, st_i)
                _use_rs_buffer(color)
                if const_expr(rs_ring):
                    _rs_ring(i, n_this, color)
                else:
                    _rs_mesh(i, n_this, color)
                _x_lap2_self(i, n_this, color)
                if i > fx.Int32(0):
                    # Every group before the last is a whole super-tile.
                    _x_lap3(i - st_i, st_i, pcolor)
                pcolor = color
                color = _next_color(color)
            last_i = ((n_block_tiles - fx.Int32(1)) // st_i) * st_i
            _x_lap3(last_i, n_block_tiles - last_i, pcolor)
            if tid == 0:
                _store_color(color)
            gpu.barrier()

        def _mover():
            """A mover block: wave ``w`` copies compute block ``w``'s cross
            slot of every group to the partner, as each becomes ready."""
            served = (bid - n_blocks) * fx.Int32(waves) + wave
            if served < n_blocks:
                n_block_tiles = (
                    num_tiles - served + n_blocks - fx.Int32(1)
                ) // n_blocks
                st_i = fx.Int32(super_tile)
                load_mc, store_mc = _color_io(colors_ptr, fx.Int32(grid) + served)
                mcolor = load_mc()
                partner_buf = _buffer_ptr(_to_sgpr_i64(partner_base), T.i32, 4)
                step = WAVE * _MOVER_UNROLL
                for i in range(fx.Int32(0), n_block_tiles, st_i):
                    remain = n_block_tiles - i
                    n_this = (remain < st_i).select(remain, st_i)
                    src = _x_slot(_x_buf(mcolor), my_group, fx.Int32(0), served)
                    if lane == fx.Int32(0):
                        _spin(src + fx.Int32(x_release), mcolor)
                    # System scope (buffer_inv sc0 sc1): the mover usually runs
                    # on another XCD than the block it serves, and a workgroup-
                    # scope acquire here gave stale cross packets (2026-11-17).
                    _acquire_inbox(acquire_scope)
                    n_vec = n_this * fx.Int32(x_payload // 4)
                    last = n_vec - fx.Int32(1)
                    # Where a lane past the end stores: this slot's tail sector
                    # in the partner's inbox, which nobody reads on the mover
                    # schedule (ready flags live in each rank's own half). It
                    # must not store a payload vector twice: the receiver
                    # re-arms what it read, and a late duplicate would then put
                    # stale bytes back, to be read as landed a lap later.
                    spare = src + fx.Int32(x_release)
                    for v0 in range(fx.Int32(0), n_vec, fx.Int32(step)):
                        # No branch, so the compiler cannot sink a load into
                        # one: an idle lane loads the last vector and stores it
                        # to the spare sector.
                        lds_off, st_off = [], []
                        for u in range_constexpr(_MOVER_UNROLL):
                            x = v0 + lane + fx.Int32(u * WAVE)
                            live = x < n_vec
                            lds_off.append(src + live.select(x, last) * fx.Int32(4))
                            st_off.append(live.select(src + x * fx.Int32(4), spare))
                        data = [
                            _buffer_load(inbox, o, 4, fx.Int32, _CM_SC0 | _CM_SC1)
                            for o in lds_off
                        ]
                        for u in range_constexpr(_MOVER_UNROLL):
                            _buffer_store(
                                partner_buf, st_off[u], data[u], _CM_SC0 | _CM_SC1
                            )
                    mcolor = _next_color(mcolor)
                if lane == fx.Int32(0):
                    store_mc(mcolor)

        if const_expr(x_mover):
            if bid < n_blocks:
                _compute_pipelined()
            else:
                _mover()
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
                                xc, _x_slot(parity, g, s) + fx.Int32(k * xc.rank_tile_i32)
                            )
                            acc = _codec_dequant(xc, words, scale, tid, acc)
                        words, scale, is_leader = _codec_quant(ag, acc, lane, tid)
                        for j in range_constexpr(_local_rows(local_ag, G)):
                            _lds_write_packet(
                                ag, fx.Int32(j * rank_atoms + k), words, scale, is_leader
                            )
                        _stash_own(
                            tile,
                            li * rank_atoms + k,
                            _codec_dequant(
                                ag, words, _scale_from_word(ag, scale, pair_in_slot), tid
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
            # The mover blocks follow the compute blocks, one wave per block.
            grid=(
                (grid_x + (grid_x + fx.Int32(waves - 1)) // fx.Int32(waves))
                if x_mover
                else grid_x,
                1,
                1,
            ),
            block=(block, 1, 1),
            stream=stream,
        )

    # The groups, rank and codecs are all baked into the binary.
    gtag = "_".join("".join(str(r) for r in g) for g in groups)
    tag = (
        f"{SCALE_CODEC_TAG}_ws{world_size}_g{gtag}_r{self_rank}_st{super_tile}"
        f"_g{grid}_{inbox_memory}_{rs_codec}_{x_codec}_{ag_codec}_b{block}"
        f"_{local_rs}_{local_ag}"
        + ("_xm" if x_mover else "")
    )
    launch_quick_allreduce_hier.func.__name__ = f"launch_quick_allreduce_hier_{tag}"
    try:
        quick_allreduce_hier.func.__name__ = f"quick_allreduce_hier_{tag}"
    except AttributeError:
        pass
    return {
        "launch": launch_quick_allreduce_hier,
        "flags_bytes": 0,  # the handshakes ride in each slot's 64 B tail
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
        "x_mover": bool(x_mover),
        # Mover blocks launched per compute block (see the launch).
        "mover_waves": waves if x_mover else 0,
        # colors[grid + w]: mover wave w's copy of block w's colour.
        "color_slots": 2 if x_mover else 1,
        **({"inbox_fill": X_SENTINEL_FILL_BYTE} if x_mover else {}),
        **epilogue.spec(),
    }
