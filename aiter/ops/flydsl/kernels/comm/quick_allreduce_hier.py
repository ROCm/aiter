# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""gfx942/gfx950 quantized **hierarchical** all-reduce, for hosts whose GPUs
form uniform groups (the CPU sockets of a two-socket PCIe host).

The ranks split into two groups of ``G``. Ranks at the same local index in the
two groups are *partners*. One super-tile group of tiles takes three laps:

1. **Local reduce-scatter.** A tile's ``ATOMS`` atoms split into ``G`` chunks
   of ``ATOMS / G`` atoms; chunk ``l`` belongs to local index ``l``. Each rank
   pushes every other chunk to its owner in the group, mesh style, and adds
   the ``G - 1`` it receives to its own chunk, which stays in registers.
2. **Cross exchange.** Each rank quantizes its group's partial sum and sends
   it to its partner, a one-shot between two ranks, and also into its own
   inbox. Both partners then evaluate ``dq(P_group0) + dq(P_group1)`` from the
   same two packets in the same order, so they hold bit-identical sums.
3. **Local all-gather.** Each rank quantizes the sum once and pushes it to
   the ``G - 1`` other ranks of its group, mesh style, and decodes its own
   chunk from the same packet. All ``N`` ranks end with the same bits.

Only lap 2 crosses between groups: ``1/G`` of a tile per rank, on ``G``
concurrent pair streams, instead of the mesh's fanout to every remote peer or
the ring's single crossing stream. Laps 1 and 3 move the bytes of a TP``G``
mesh inside each group. Every value is quantized three times, once per lap,
each lap with its own codec.

Handshakes are per lap and per super-tile group, in a 64 B sector at the tail
of each inbox slot, as in the ring. The mesh's single-colour argument -- a
peer cannot write group ``i + 1`` into a slot before we have read group ``i``
-- holds for the two local laps, whose next write needs a send of ours that
comes after our read. It does not hold for the cross slot: once our partner
has our group ``i`` packet it can finish group ``i`` and run group ``i + 1``'s
local reduce-scatter inside its own group without us, and send its next cross
packet. The cross slots are therefore double-buffered by the parity of the
colour; group ``i + 2`` cannot arrive before we have sent ``i + 1``, which is
after we read ``i``.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T

from ...allreduce_policy import HIER_CODECS, HIER_SUPER_TILES
from .quick_allreduce_codec import (
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
    _INBOX_POLICY,
    ATOMS,
    FLAG_I32_PER_LANE,
    FLAG_LANES,
    QUAD_LANES,
    QUADS_PER_WAVE,
    _acquire_inbox,
    _buffer_load,
    _buffer_ptr,
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


def hier_fanout_quads(block: int, group_size: int, codecs) -> tuple[int, int]:
    """``(quads the block has, quads the local fanout needs)``. A block with
    fewer would drop the sectors past the end."""
    have = int(block) // QUAD_LANES
    need = 0
    for name in codecs:
        c = codecs_for_block(int(block))[name]
        need = max(need, (group_size - 1) * min(8, c.n_sectors))
    return have, need


def hier_lds_bytes(block: int, group_size: int, rs_codec, x_codec, ag_codec) -> int:
    """LDS of one workgroup: the larger of the local laps' ``(G - 1)``
    destinations and the cross lap's one, ``ATOMS / G`` rank-tiles each."""
    cs = codecs_for_block(int(block))
    rank_atoms = ATOMS // int(group_size)
    local = (int(group_size) - 1) * rank_atoms
    return 4 * max(
        local * max(cs[rs_codec].rank_tile_i32, cs[ag_codec].rank_tile_i32),
        rank_atoms * cs[x_codec].rank_tile_i32,
    )


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
):
    """Build the hierarchical kernel for one *rank*.

    *groups* is two tuples of ranks; partners sit at the same position of the
    sorted groups. The rank, its group, its local peers and its partner are
    all trace-time constants, so this is one binary per rank.
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
    # Local peers in local-index order; every per-destination structure of a
    # local lap -- pack rows, fanout quads, flag lanes -- is indexed by
    # position in this list.
    push_locals = [m for m in range(G) if m != li]
    push_ranks = [local[m] for m in push_locals]
    n_push = len(push_ranks)
    rank_atoms = ATOMS // G

    have, need = hier_fanout_quads(block, G, (rs_codec, ag_codec))
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
    # LOCAL_RS [grid][G src], CROSS [2 parity][grid][2 groups],
    # LOCAL_AG [grid][G src].
    rs_base = 0
    x_base = rs_base + grid * G * rs_slot
    ag_base = x_base + 2 * grid * 2 * x_slot
    inbox_i32 = ag_base + grid * G * ag_slot

    lds_bytes = hier_lds_bytes(block, G, rs_codec, x_codec, ag_codec)
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
        def _local_slot(base, slot, payload, src, sub):
            return (
                fx.Int32(base)
                + (bid * fx.Int32(G) + src) * fx.Int32(slot)
                + sub * fx.Int32(payload)
            )

        def _rs_slot(src, sub):
            return _local_slot(rs_base, rs_slot, rs_payload, src, sub)

        def _ag_slot(src, sub):
            return _local_slot(ag_base, ag_slot, ag_payload, src, sub)

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

        def _recv(c, base):
            """One packet of codec *c* from our inbox at i32 offset *base*."""

            def _get(off):
                return _buffer_load(inbox, base + off, 1, fx.Int32, recv_policy)[0]

            words, word = _codec_load(c, _get, tid, scale_slot)
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

        def _publish_partner(flag_i32, color):
            _drain()
            if tid < fx.Int32(FLAG_LANES):
                elem = flag_i32 + tid * fx.Int32(FLAG_I32_PER_LANE)
                _store_flag_peer(partner_base + _i32_to_bytes(elem), color, flag_policy)

        def _spin(flag_i32, color):
            flag = self_base + _i32_to_bytes(flag_i32)
            current = _load_flag(flag)
            while current != color:
                current = _load_flag(flag)

        def _acquire():
            # After the join, for every wave; see the mesh's _wait_release.
            gpu.barrier()
            _acquire_inbox(acquire_scope)

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

        n_block_tiles = (num_tiles - bid + n_blocks - fx.Int32(1)) // n_blocks
        color = _load_color()
        st_i = fx.Int32(super_tile)
        for i in range(fx.Int32(0), n_block_tiles, st_i):
            remain = n_block_tiles - i
            n_this = (remain < st_i).select(remain, st_i)
            parity = color & fx.Int32(1)

            # Lap 1: local reduce-scatter. Quantize each local peer's chunk
            # and push it; our own chunk is reloaded in lap 2.
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

            # Lap 2: sum the group's chunk in local-index order, then send the
            # partial to the partner and to our own cross slot.
            for s in range(fx.Int32(0), n_this, fx.Int32(1)):
                tile = bid + (i + s) * n_blocks
                for k in range_constexpr(rank_atoms):
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
                                _rs_slot(fx.Int32(m), s)
                                + fx.Int32(k * rs.rank_tile_i32),
                            )
                            acc = _codec_dequant(rs, words, scale, tid, acc)
                    words, scale, is_leader = _codec_quant(xc, acc, lane, tid)
                    _lds_write_packet(xc, fx.Int32(k), words, scale, is_leader)
                gpu.barrier()
                _store_rows(xc, partner_base, _x_slot(parity, my_group, s))
                _store_rows(xc, self_base, _x_slot(parity, my_group, s))
                _next_subtile(s, n_this)
            _publish_partner(
                _x_slot(parity, my_group, fx.Int32(0)) + fx.Int32(x_release), color
            )
            if tid == fx.Int32(0):
                _spin(
                    _x_slot(parity, 1 - my_group, fx.Int32(0)) + fx.Int32(x_release),
                    color,
                )
            _acquire()

            # Lap 3: the final sum, group 0's packet first on both partners,
            # quantized once and pushed to every local peer. Our own chunk is
            # decoded from the same packet and stored now.
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
                    for j in range_constexpr(n_push):
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
                _fanout_local(ag, _ag_slot(fx.Int32(li), s))
                _next_subtile(s, n_this)
            _publish_local(
                _ag_slot(fx.Int32(li), fx.Int32(0)) + fx.Int32(ag_release), color
            )
            _wait_local(_ag_slot, ag_release, color)

            # Gather the other chunks of the group.
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
        "payload_policy": payload_policy,
        "flag_policy": flag_policy,
        "release_scope": release_scope,
        "rank_atoms": rank_atoms,
        "grid": grid,
        "block": block,
        "rank": self_rank,
        **epilogue.spec(),
    }
