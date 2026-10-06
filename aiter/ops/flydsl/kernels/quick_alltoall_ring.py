# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""gfx942/gfx950 TP∈{2,4,8} equal-split **shifted-pairwise** all-to-all.

Same contract, wire formats, geometry and double-buffered inbox as the mesh
all-to-all (``quick_alltoall_mesh``); what differs is the order the bytes go
out in. ``N-1`` steps per super-tile group: at step ``s`` rank ``r`` writes
only to ``(r + s) % N`` and reads only from ``(r - s) % N``, so

* every store goes to one destination, as one contiguous run of
  ``ST * rank_atoms * rank_tile`` bytes -- the property the all-reduce ring
  exploits on a PCIe host, where a GPU has a single shared x16 uplink and
  destination-interleaved writes collapse;
* when ranks run in step, each GPU receives from exactly one peer at a time;
* unlike a store-and-forward ring, nothing is forwarded, so the wire volume is
  still the minimal ``(N-1)/N`` of the payload. A forwarding ring would move
  ``N/2`` times that.

The receive for step ``s - 1`` runs after the publish for step ``s``, so each
wait overlaps the next send.
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
    fanout_contiguous,
    fanout_registers,
    lds_write_packet,
    make_pack_storage,
    next_color,
    parity_slot_i32,
    publish_flags,
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
from .quick_alltoall_mesh import (
    A2A_CODECS,
    A2A_QUANT_CODECS,
    FLAG_I32,
    PARITIES,
    a2a_rank_tile_i32,
)

__all__ = [
    "A2A_RING_ST_LADDER",
    "A2A_RING_SUPER_TILES",
    "a2a_ring_st_ladder",
    "make_quick_alltoall_ring_kernel",
]

# One publish per step per group, ``N-1`` per group: a bigger super-tile is the
# knob that amortizes them, as in the all-reduce ring.
A2A_RING_SUPER_TILES = (1, 8, 16, 32)

# ``(min_bytes, super_tile, grid_cap, block)``, as for the mesh. Seeded from
# the all-reduce ring; to be refitted with ``bench_comm.py --operation a2a``.
A2A_RING_ST_LADDER = {
    (link, ws): ((0, 8, 128, 256), (16 << 20, 8, 128, 512))
    for link in ("xgmi", "pcie")
    for ws in SUPPORTED_WORLDS
}


def a2a_ring_st_ladder(world_size: int, link: str = "pcie"):
    """Rungs for *(link, world_size)*, or ``()`` when there is no ladder."""
    return A2A_RING_ST_LADDER.get((str(link), int(world_size)), ())


def make_quick_alltoall_ring_kernel(
    *,
    world_size: int,
    rank: int,
    super_tile: int = 8,
    grid: int,
    inbox_memory: str = "finegrained",
    codec: str = "none",
    block: int = BLOCK,
    order: str = "peer",
):
    """Build the shifted-pairwise all-to-all for one *rank*.

    *order* is accepted for signature parity with the mesh and ignored: every
    step has a single destination, so there is no store order to choose.
    """
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
    if super_tile not in A2A_RING_SUPER_TILES:
        raise ValueError(
            f"super_tile must be one of {A2A_RING_SUPER_TILES}, got {super_tile!r}"
        )
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
    writeback_scope = (
        _SYSTEM_SYNC_SCOPE if acquire_scope == _SYSTEM_SYNC_SCOPE else None
    )

    direct = codec == "none"
    quantized = codec in A2A_QUANT_CODECS
    c = None if direct else codecs_for_block(block)[codec]
    rank_atoms = ATOMS // world_size
    rank_tile_i32 = a2a_rank_tile_i32(codec, block)
    payload_i32 = rank_atoms * rank_tile_i32
    release_i32_off = super_tile * payload_i32
    wire_tile_i32 = release_i32_off + FLAG_I32
    data_bytes = PARITIES * grid * world_size * wire_tile_i32 * 4
    tile_bytes = rank_atoms * block * 16
    quads_per_block = block // QUAD_LANES

    # Step s sends to dests[s - 1] and, one step later, receives from srcs[s - 1].
    dests = [(rank + s) % world_size for s in range(1, world_size)]
    srcs = [(rank - s) % world_size for s in range(1, world_size)]
    # One destination is staged at a time: rank_atoms rows.
    lds_bytes = 0 if direct else rank_atoms * c.rank_tile_bytes
    PackStorage = None if direct else make_pack_storage(rank_atoms * c.rank_tile_i32)
    codec_io = (
        {"decode": _atom_bf16_to_f16, "encode": _atom_f16_to_bf16} if quantized else {}
    )

    @flyc.kernel(known_block_size=[block, 1, 1])
    def quick_alltoall_ring(
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
        smem_ptr = None
        if const_expr(not direct):
            lds = fx.SharedAllocator().allocate(PackStorage).peek()
            pack = lds.pack.view(
                fx.make_layout((rank_atoms, c.rank_tile_i32), (c.rank_tile_i32, 1))
            )
            smem_ptr = lds.pack.ptr

        peers = _load_peers(peer_ptrs, world_size)
        self_base = peers[rank]
        inbox = _buffer_ptr(_to_sgpr_i64(self_base), T.i32, 16, data_bytes)

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
            for p in dests
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

        def _send(dest, tile, parity, sub):
            """Push this tile's share of chunk *dest* into *dest*'s inbox."""
            base = _slot(parity, rank) + sub * fx.Int32(payload_i32)
            if const_expr(direct):
                fanout_registers(
                    [
                        (
                            peers[dest]
                            + _i32_to_bytes(
                                base + fx.Int32(k * rank_tile_i32) + tid * fx.Int32(4)
                            ),
                            io[dest][0](tile, k),
                        )
                        for k in range(rank_atoms)
                    ],
                    payload_policy,
                )
            else:
                for k in range_constexpr(rank_atoms):
                    words, scale, leader = _codec_quant(
                        c, io[dest][0](tile, k), lane, tid
                    )
                    lds_write_packet(
                        c, pack, fx.Int32(k), words, scale, leader, tid, scale_slot
                    )
                gpu.barrier()
                fanout_contiguous(
                    codec=c,
                    n_atoms=rank_atoms,
                    quad_id=quad_id,
                    lane_in_quad=lane_in_quad,
                    quads_per_block=quads_per_block,
                    smem_ptr=smem_ptr,
                    slot_i32=base,
                    dest_base=peers[dest],
                    payload_policy=payload_policy,
                )

        def _recv(src, tile, parity, sub):
            """Drain *src*'s packet for this tile into output chunk *src*.

            ``io`` is keyed on ``dests``, which is every peer -- the same set
            as ``srcs`` -- so chunk *src*'s store is there.
            """
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

                    words, scale = codec_recv(c, _get, tid, scale_slot, pair_in_slot)
                    io[src][1](tile, k, _codec_dequant(c, words, scale, tid))

        def _publish(dest, parity, color):
            publish_flags(
                tid=tid,
                n_dest=1,
                dest_base=lambda _j: peers[dest],
                flag_i32=_slot(parity, rank) + fx.Int32(release_i32_off),
                color=color,
                flag_policy=flag_policy,
                release_scope=release_scope,
            )

        def _wait(src, parity, color):
            wait_flags(
                tid=tid,
                n_src=1,
                skip_rank=None,
                flag_addr=lambda _t: self_base
                + _i32_to_bytes(_slot(parity, src) + fx.Int32(release_i32_off)),
                color=color,
                acquire_scope=acquire_scope,
                writeback_scope=writeback_scope,
            )

        def _send_step(step, i, n_this, parity, color):
            dest = dests[step]
            for s in range(fx.Int32(0), n_this, fx.Int32(1)):
                _send(dest, bid + (i + s) * n_blocks, parity, s)
                if const_expr(not direct):  # noqa: SIM102
                    if (s + fx.Int32(1)) < n_this:
                        # This wave's LDS reads must land before the next
                        # sub-tile overwrites the staging rows.
                        rocdl.s_waitcnt(lgkmcnt=0)
                        gpu.barrier()
            _publish(dest, parity, color)

        def _recv_step(step, i, n_this, parity, color):
            src = srcs[step]
            _wait(src, parity, color)
            for s in range(fx.Int32(0), n_this, fx.Int32(1)):
                _recv(src, bid + (i + s) * n_blocks, parity, s)

        n_block_tiles = (num_tiles - bid + n_blocks - fx.Int32(1)) // n_blocks
        color = _load_color()
        st_i = fx.Int32(super_tile)
        for i in range(fx.Int32(0), n_block_tiles, st_i):
            remain = n_block_tiles - i
            n_this = (remain < st_i).select(remain, st_i)
            parity = color & fx.Int32(1)
            for step in range_constexpr(world_size - 1):
                _send_step(step, i, n_this, parity, color)
                if const_expr(step == 0):
                    # Our own chunk, behind the first flag's round trip.
                    for s in range(fx.Int32(0), n_this, fx.Int32(1)):
                        tile = bid + (i + s) * n_blocks
                        for k in range_constexpr(rank_atoms):
                            self_store(tile, k, self_load(tile, k))
                else:
                    _recv_step(step - 1, i, n_this, parity, color)
            _recv_step(world_size - 2, i, n_this, parity, color)
            color = next_color(color, parity_safe=True)
        if tid == 0:
            _store_color(color)
        gpu.barrier()

    flat_wg = f"{block},{block}"

    @flyc.jit
    def launch_quick_alltoall_ring(
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
        quick_alltoall_ring(
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

    tag = f"ws{world_size}_r{rank}_st{super_tile}_g{grid}_{inbox_memory}_{codec}"
    tag += f"_b{block}"
    launch_quick_alltoall_ring.func.__name__ = f"launch_quick_alltoall_ring_{tag}"
    try:
        quick_alltoall_ring.func.__name__ = f"quick_alltoall_ring_{tag}"
    except AttributeError:
        pass
    return {
        "launch": launch_quick_alltoall_ring,
        "flags_bytes": 0,
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
    }
