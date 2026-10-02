# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""gfx942/gfx950 TP∈{2,4,8} equal-split **mesh** all-to-all.

The input is ``N`` equal chunks of ``C`` bytes: chunk ``j`` goes to rank ``j``,
and output chunk ``i`` comes from rank ``i``. Every tile, a rank pushes each
peer its share straight into that peer's inbox, publishes one flag per peer,
copies its own chunk locally, then waits for the ``N-1`` flags addressed to it
and drains its inbox into the output. One hop, ``(N-1)/N`` of the payload on
the wire.

Geometry is the all-reduce's: a tile is ``ATOMS`` atoms of ``block x 16 B``, of
which each destination gets ``ATOMS / N`` atoms *of its own chunk*. So the LDS
staging, the codec rank-tiles and the residency table carry over unchanged.

Wire formats:

* ``"none"`` -- register-direct. A thread's 16 B lands at the same offset in
  every destination, so it is stored straight from registers: no LDS, no
  codec, byte-exact for any dtype.
* ``"fp16"`` -- the same bytes through the LDS-staged sector fanout. Lossless;
  isolates the transport from the codec in tests.
* ``"int4"`` / ``"int6"`` -- group-16 E4M3-scaled codecs, bf16 payload.

There is one phase, so unlike the two-shot all-reduce nothing stops a rank
running a tile ahead and overwriting a slot its peer has not read yet. The
inbox is double-buffered on ``color & 1``, as in the one-shot: rank A only
reaches tile ``i+2`` after receiving B's tile ``i+1``, which B sends only after
it has finished reading A's tile ``i``.

The rank is a trace-time constant (one binary per rank): the destination list
"every peer but me" and the self copy are unrolled over it.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import Int32, Int64, Stream, T

from .collectives_shared import (
    _INBOX_POLICY,
    _SYSTEM_SYNC_SCOPE,
    ATOMS,
    BLOCK,
    QUAD_LANES,
    QUADS_PER_WAVE,
    SUPPORTED_WORLDS,
    _buffer_load,
    _buffer_ptr,
    _color_io,
    _i32_to_bytes,
    _load_peers,
    _to_sgpr_i64,
    chunk_payload_io,
    fanout_registers,
    fanout_sectors,
    lds_write_packet,
    make_pack_storage,
    next_color,
    parity_slot_i32,
    publish_flags,
    select_peer_base,
    stripes_for,
    wait_flags,
)
from .quick_allreduce_codec import (
    SUPPORTED_BLOCKS,
    _atom_bf16_to_f16,
    _atom_f16_to_bf16,
    _clamp_fp16_overflow,
    _codec_dequant,
    _codec_quant,
    codec_recv,
    codecs_for_block,
    scale_slot_of,
    thread_lane,
)

__all__ = [
    "A2A_CODECS",
    "A2A_DIRECT_ORDERS",
    "A2A_MESH_ST_LADDER",
    "A2A_MESH_SUPER_TILES",
    "a2a_mesh_st_ladder",
    "make_quick_alltoall_mesh_kernel",
]

# Wire formats. "none" is register-direct; the rest go through LDS staging.
A2A_CODECS = ("none", "fp16", "int4", "int6")
# Codecs that quantize, i.e. need bf16 in and the fp16 math path.
A2A_QUANT_CODECS = ("int4", "int6")
# Store order of the register-direct path: one destination's atoms back to
# back ("peer"), or each atom to every destination in turn ("atom").
A2A_DIRECT_ORDERS = ("peer", "atom")
A2A_MESH_SUPER_TILES = (1, 8)
PARITIES = 2
# One 64 B handshake sector at the tail of every wire slot.
FLAG_I32 = 16

# ``(min_bytes, super_tile, grid_cap, block, skip_self)`` rungs, ascending, keyed
# on ``(link, world_size)``; *min_bytes* is the whole input. ``skip_self`` is
# unused -- the all-to-all never routes its own chunk through the inbox -- and
# kept so the host's ladder machinery is shared with the all-reduce. Seeded
# from the all-reduce mesh; to be refitted with ``bench_comm.py --operation
# a2a``.
A2A_MESH_ST_LADDER = {
    (link, ws): ((0, 1, 128, 256, False), (4 << 20, 8, 128, 512, False))
    for link in ("xgmi", "pcie")
    for ws in SUPPORTED_WORLDS
}


def a2a_mesh_st_ladder(world_size: int, link: str = "pcie"):
    """Rungs for *(link, world_size)*, or ``()`` when there is no ladder."""
    return A2A_MESH_ST_LADDER.get((str(link), int(world_size)), ())


def a2a_rank_tile_i32(codec: str, block: int) -> int:
    """i32 one atom occupies on the wire."""
    if codec == "none":
        return block * 4
    return codecs_for_block(block)[codec].rank_tile_i32


def make_quick_alltoall_mesh_kernel(
    *,
    world_size: int,
    rank: int,
    super_tile: int = 1,
    grid: int,
    inbox_memory: str = "uncached",
    codec: str = "none",
    block: int = BLOCK,
    order: str = "peer",
):
    if world_size not in SUPPORTED_WORLDS:
        raise ValueError(
            f"world_size must be one of {SUPPORTED_WORLDS}, got {world_size}"
        )
    if not 0 <= int(rank) < world_size:
        raise ValueError(f"rank must be in [0, {world_size}), got {rank}")
    if inbox_memory not in _INBOX_POLICY:
        raise ValueError(
            f"inbox_memory must be one of {tuple(_INBOX_POLICY)}, got {inbox_memory!r}"
        )
    if codec not in A2A_CODECS:
        raise ValueError(f"codec must be one of {A2A_CODECS}, got {codec!r}")
    if block not in SUPPORTED_BLOCKS:
        raise ValueError(f"block must be one of {SUPPORTED_BLOCKS}, got {block!r}")
    if super_tile not in A2A_MESH_SUPER_TILES:
        raise ValueError(
            f"super_tile must be one of {A2A_MESH_SUPER_TILES}, got {super_tile!r}"
        )
    if order not in A2A_DIRECT_ORDERS:
        raise ValueError(f"order must be one of {A2A_DIRECT_ORDERS}, got {order!r}")
    if grid < 1:
        raise ValueError(f"grid must be positive, got {grid}")
    if ATOMS % world_size != 0:
        raise ValueError(f"ATOMS={ATOMS} is not divisible by world_size={world_size}")

    rank = int(rank)
    policy = _INBOX_POLICY[inbox_memory]
    payload_policy = policy["payload"]
    flag_policy = policy["flag"]
    release_scope = policy["release"]
    acquire_scope = policy["acquire"]
    recv_policy = policy["recv"]
    sector_fastest = policy["fanout"] != "peer"
    # The self copy and every receive store output lines this block may still
    # hold dirty when it next waits; a system-scope acquire is an L2 invalidate,
    # so those must be written back first.
    writeback_scope = (
        _SYSTEM_SYNC_SCOPE if acquire_scope == _SYSTEM_SYNC_SCOPE else None
    )

    direct = codec == "none"
    quantized = codec in A2A_QUANT_CODECS
    c = None if direct else codecs_for_block(block)[codec]
    rank_atoms = ATOMS // world_size
    rank_tile_i32 = a2a_rank_tile_i32(codec, block)
    # One sub-tile from one source, then ST of them and the handshake sector.
    payload_i32 = rank_atoms * rank_tile_i32
    release_i32_off = super_tile * payload_i32
    wire_tile_i32 = release_i32_off + FLAG_I32
    data_bytes = PARITIES * grid * world_size * wire_tile_i32 * 4
    # HBM bytes one tile spans in each chunk.
    tile_bytes = rank_atoms * block * 16

    push_peers = [p for p in range(world_size) if p != rank]
    n_push = len(push_peers)
    if direct:
        fanout_pairs = (
            [(p, k) for p in push_peers for k in range(rank_atoms)]
            if order == "peer"
            else [(p, k) for k in range(rank_atoms) for p in push_peers]
        )
        pack_rows = 0
        lds_bytes = 0
        stripes = None
        PackStorage = None
    else:
        fanout_pairs = None
        pack_rows = n_push * rank_atoms
        lds_bytes = pack_rows * c.rank_tile_bytes
        stripes = stripes_for(c.n_sectors, n_push, block)
        PackStorage = make_pack_storage(pack_rows * c.rank_tile_i32)
    codec_io = (
        {"decode": _atom_bf16_to_f16, "encode": _atom_f16_to_bf16} if quantized else {}
    )

    @flyc.kernel(known_block_size=[block, 1, 1])
    def quick_alltoall_mesh(
        rank_unused: Int32,
        nbytes: Int64,
        num_tiles: Int32,
        inp_ptr: Int64,
        out_ptr: Int64,
        peer_ptrs: Int64,
        colors_ptr: Int64,
        n_blocks: Int32,
    ):
        if const_expr(quantized):
            _clamp_fp16_overflow()
        tid = fx.Int32(gpu.thread_id("x"))
        bid = fx.Int32(gpu.block_id("x"))

        wave, lane = thread_lane(tid, block)
        quad_layout = fx.make_layout((QUADS_PER_WAVE, QUAD_LANES), (QUAD_LANES, 1))
        quad, lane_in_quad = fx.idx2crd(lane, quad_layout).unpack()
        quad_id = wave * fx.Int32(QUADS_PER_WAVE) + quad
        scale_slot, pair_in_slot = scale_slot_of(tid, block)

        pack = None
        pack_layout = None
        smem_ptr = None
        if const_expr(not direct):
            pack_layout = fx.make_layout(
                (pack_rows, c.rank_tile_i32), (c.rank_tile_i32, 1)
            )
            lds = fx.SharedAllocator().allocate(PackStorage).peek()
            pack = lds.pack.view(pack_layout)
            smem_ptr = lds.pack.ptr

        peers = _load_peers(peer_ptrs, world_size)
        self_base = peers[rank]
        inbox = _buffer_ptr(_to_sgpr_i64(self_base), T.i32, 16, data_bytes)

        # Chunk p's (load, store): load reads the input chunk we send to p, store
        # writes the output chunk we receive from p. Both bounded by the chunk.
        io = {
            p: chunk_payload_io(
                inp_ptr,
                out_ptr,
                nbytes,
                p,
                num_tiles,
                rank_atoms,
                block,
                tid,
                **codec_io,
            )
            for p in push_peers
        }
        self_load, self_store = chunk_payload_io(
            inp_ptr, out_ptr, nbytes, rank, num_tiles, rank_atoms, block, tid
        )
        _load_color, _store_color = _color_io(colors_ptr, bid)

        def _slot(parity, src):
            return parity_slot_i32(
                parity,
                bid,
                src,
                grid=grid,
                n_src=world_size,
                wire_tile_i32=wire_tile_i32,
            )

        def _push_base(j):
            return select_peer_base(peers, push_peers, j)

        def _send(tile, parity, sub):
            """Push this tile's share of every destination chunk."""
            base = _slot(parity, rank) + sub * fx.Int32(payload_i32)
            if const_expr(direct):
                atoms = {(p, k): io[p][0](tile, k) for p, k in fanout_pairs}
                fanout_registers(
                    [
                        (
                            peers[p]
                            + _i32_to_bytes(
                                base + fx.Int32(k * rank_tile_i32) + tid * fx.Int32(4)
                            ),
                            atoms[(p, k)],
                        )
                        for p, k in fanout_pairs
                    ],
                    payload_policy,
                )
            else:
                for j, p in enumerate(push_peers):
                    for k in range_constexpr(rank_atoms):
                        words, scale, leader = _codec_quant(
                            c, io[p][0](tile, k), lane, tid
                        )
                        lds_write_packet(
                            c,
                            pack,
                            fx.Int32(j * rank_atoms + k),
                            words,
                            scale,
                            leader,
                            tid,
                            scale_slot,
                        )
                gpu.barrier()
                fanout_sectors(
                    codec=c,
                    n_dest=n_push,
                    n_atoms=rank_atoms,
                    stripes=stripes,
                    sector_fastest=sector_fastest,
                    quad_id=quad_id,
                    lane_in_quad=lane_in_quad,
                    smem_ptr=smem_ptr,
                    pack_layout=pack_layout,
                    slot_i32=base,
                    dest_base=_push_base,
                    payload_policy=payload_policy,
                )

        def _recv(tile, parity, sub):
            """Drain every source's packet for this tile into its output chunk."""
            for src in push_peers:
                for k in range_constexpr(rank_atoms):
                    base = (
                        _slot(parity, src)
                        + sub * fx.Int32(payload_i32)
                        + fx.Int32(k * rank_tile_i32)
                    )
                    if const_expr(direct):
                        vec = _buffer_load(
                            inbox, base + tid * fx.Int32(4), 4, fx.Int32, recv_policy
                        )
                        io[src][1](tile, k, vec)
                    else:

                        def _get(off, base=base):
                            return _buffer_load(
                                inbox, base + off, 1, fx.Int32, recv_policy
                            )[0]

                        words, scale = codec_recv(
                            c, _get, tid, scale_slot, pair_in_slot
                        )
                        io[src][1](tile, k, _codec_dequant(c, words, scale, tid))

        def _flag_addr(parity):
            def _addr(src):
                return self_base + _i32_to_bytes(
                    _slot(parity, src) + fx.Int32(release_i32_off)
                )

            return _addr

        # Stride by the *launched* grid, not the compile-time cap: the host may
        # launch fewer blocks than ``grid``. ``grid`` still sizes the wire slots
        # and the colour array, so n_blocks <= grid always.
        n_block_tiles = (num_tiles - bid + n_blocks - fx.Int32(1)) // n_blocks
        color = _load_color()
        st_i = fx.Int32(super_tile)
        for i in range(fx.Int32(0), n_block_tiles, st_i):
            remain = n_block_tiles - i
            n_this = (remain < st_i).select(remain, st_i)
            parity = color & fx.Int32(1)

            for s in range(fx.Int32(0), n_this, fx.Int32(1)):
                _send(bid + (i + s) * n_blocks, parity, s)
                if const_expr(not direct):  # noqa: SIM102
                    if (s + fx.Int32(1)) < n_this:
                        # This wave's LDS reads must land before the next
                        # sub-tile overwrites the staging rows. lgkmcnt only:
                        # the payload stores stay in flight until the publish.
                        rocdl.s_waitcnt(lgkmcnt=0)
                        gpu.barrier()
            publish_flags(
                tid=tid,
                n_dest=n_push,
                dest_base=_push_base,
                flag_i32=_slot(parity, rank) + fx.Int32(release_i32_off),
                color=color,
                flag_policy=flag_policy,
                release_scope=release_scope,
            )

            # Our own chunk never touches the wire; copying it here hides it
            # behind the flags' round trip.
            for s in range(fx.Int32(0), n_this, fx.Int32(1)):
                tile = bid + (i + s) * n_blocks
                for k in range_constexpr(rank_atoms):
                    self_store(tile, k, self_load(tile, k))

            wait_flags(
                tid=tid,
                n_src=n_push,
                skip_rank=rank,
                flag_addr=_flag_addr(parity),
                color=color,
                acquire_scope=acquire_scope,
                writeback_scope=writeback_scope,
            )
            for s in range(fx.Int32(0), n_this, fx.Int32(1)):
                _recv(bid + (i + s) * n_blocks, parity, s)

            color = next_color(color, parity_safe=True)
        if tid == 0:
            _store_color(color)
        gpu.barrier()

    flat_wg = f"{block},{block}"

    @flyc.jit
    def launch_quick_alltoall_mesh(
        rank_arg: Int32,
        nbytes: Int64,
        num_tiles: Int32,
        inp_ptr: Int64,
        out_ptr: Int64,
        peer_ptrs: Int64,
        colors_ptr: Int64,
        grid_x: Int32,
        stream: Stream = Stream(None),  # noqa: B008
    ):
        quick_alltoall_mesh(
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

    # Everything baked in at trace time has to reach the symbol name, or two
    # variants collide in the JIT cache.
    tag = f"ws{world_size}_r{rank}_st{super_tile}_g{grid}_{inbox_memory}_{codec}"
    tag += f"_b{block}"
    if direct:
        tag += f"_{order}"
    launch_quick_alltoall_mesh.func.__name__ = f"launch_quick_alltoall_mesh_{tag}"
    try:
        quick_alltoall_mesh.func.__name__ = f"quick_alltoall_mesh_{tag}"
    except AttributeError:
        pass
    return {
        "launch": launch_quick_alltoall_mesh,
        "flags_bytes": 0,  # the handshake rides in each slot's 64 B tail
        "data_bytes": data_bytes,
        "lds_bytes": lds_bytes,
        "tile_bytes": tile_bytes,
        "tile_fp16": tile_bytes // 2,
        "rank_tile_bytes": rank_tile_i32 * 4,
        "wire_tile_bytes": wire_tile_i32 * 4,
        "super_tile": super_tile,
        "world_size": world_size,
        "rank": rank,
        "inbox_memory": inbox_memory,
        "codec": codec,
        "order": order,
        "payload_policy": payload_policy,
        "flag_policy": flag_policy,
        "release_scope": release_scope,
        "rank_atoms": rank_atoms,
        "grid": grid,
        "block": block,
        "skip_self": False,
    }
