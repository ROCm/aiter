# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""gfx942/gfx950 TP∈{2,4,8} INT4 **mesh** all-reduce.

Topology of each lap: every rank pushes directly to all ``N-1`` peers, twice.

INT4 nibble: [-8,+7], −1/8, 4 B/thread, 1152 B rank-tile. Scale is
group-16 signed E4M3 in the 128 B region. Super-tile ST∈{1,8}; host
uses ST=1 when ``num_tiles ≤`` the occupancy-clamped persistent grid.
Payload HBM is bf16; in-kernel math is packed fp16. Each rank owns
``ATOMS / world_size`` atoms of a tile (8 GPUs → 1, 4 → 2, 2 → 4); LDS
stays ``ATOMS * rank_tile_bytes``.

Geometry, cache policy and the wire codec live in
``quick_allreduce_shared`` and ``quick_allreduce_codec``, which the ring
and one-shot schedules share byte for byte.

Two tuning knobs besides the super-tile:

* ``block`` -- threads per workgroup. It sets the tile (``block * ATOMS * 16 B``), 
  hence how many blocks a payload gets and how many flags it costs.
* ``skip_self`` -- drop this rank's round trip through its own inbox: 
  its reduce-scatter share is added from registers, and its reduced chunk
  is decoded from the same packet it sends. It needs the rank at trace time, ´
  which costs one binary per rank.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import Int32, Int64, Stream, T

from . import buffer_ops
from .quick_allreduce_codec import (
    SUPER_TILES,
    SUPPORTED_BLOCKS,
    _atom_bf16_to_f16,
    _atom_f16_to_bf16,
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
from .quick_allreduce_shared import (
    _INBOX_POLICY,
    ATOMS,
    BLOCK,
    DEFAULT_GRID_CAP,
    FLAG_I32_PER_LANE,
    FLAG_LANES,
    QUAD_LANES,
    QUADS_PER_WAVE,
    SUPPORTED_WORLDS,
    TILE_BYTES,
    WORLD,
    _acquire_inbox,
    _i32_to_bytes,
    _load_flag,
    _load_i32_nt,
    _release_inbox,
    _store_flag_peer,
    _store_v4i32_peer,
    _to_sgpr_i64,
    make_pack_storage,
)

# Re-exported for the host, which imports its tile math from this module.
__all__ = [
    "DEFAULT_GRID_CAP",
    "MESH_CODECS",
    "MESH_ST_LADDER",
    "SUPER_TILES",
    "SUPPORTED_BLOCKS",
    "SUPPORTED_WORLDS",
    "TILE_BYTES",
    "WORLD",
    "clamp_grid_cap",
    "make_quick_allreduce_int4_kernel",
    "mesh_st_ladder",
]

PHASES = 2
PHASE_REDUCE_SCATTER = 0
PHASE_ALL_GATHER = 1

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

    Under-launching a persistent kernel is always safe (each block simply loops over more tiles)
    whereas over-launching is the failure mode.
    """
    if requested < 1 or cu_count < 1:
        raise ValueError("grid_cap and cu_count must be positive")
    if block < 1:
        raise ValueError("block must be positive")
    if arch not in ("gfx942", "gfx950"):
        raise ValueError(
            f"quick_allreduce_int4 has no residency measurement for {arch!r}"
        )
    key = (int(world_size), int(super_tile))
    resident = _RESIDENT_WGS_PER_CU.get(key)
    if resident is None:
        for_world = [v for (w, _st), v in _RESIDENT_WGS_PER_CU.items() if w == key[0]]
        if not for_world:
            raise ValueError(
                "quick_allreduce_int4 has no residency measurement for "
                f"world_size={world_size}"
            )
        resident = min(for_world)
    if block > BLOCK:
        resident = max(1, (resident * BLOCK) // int(block))
    return min(int(requested), resident * int(cu_count))


# Per-``(link, world_size)`` tuning ladder: ``(min_bytes, super_tile, grid_cap,
# block, skip_self)`` rungs, ascending.
#
#   TP2  ST=1 everywhere.
#   TP4  ST=8 everywhere.
#   TP8  ST=1 up to 768 KiB then ST=8.
#
_MESH_DEFAULT = {
    2: ((0, 1, 128, BLOCK, False),),
    4: ((0, 8, 128, BLOCK, False),),
    8: ((0, 1, 128, BLOCK, False), (768 << 10, 8, 128, BLOCK, False)),
}
# TODO: Measure the TP=8 case.
#  ``(min_bytes, super_tile, grid_cap, block, skip_self)``
MESH_ST_LADDER = {
    **{("xgmi", ws): rungs for ws, rungs in _MESH_DEFAULT.items()},
    ("pcie", 2): ((0, 1, 128, 128, True), (1536 << 10, 8, 128, 256, True)),
    ("pcie", 4): ((0, 1, 128, 256, True), (384 << 10, 8, 128, 512, True)),
    ("pcie", 8): _MESH_DEFAULT[8],
}


def mesh_st_ladder(world_size: int, link: str = "pcie"):
    """Rungs for *(link, world_size)*, or ``()`` when there is no ladder."""
    return MESH_ST_LADDER.get((str(link), int(world_size)), ())
# Wire formats the mesh can build.
MESH_CODECS = ("int4", "fp16")


def make_quick_allreduce_int4_kernel(
    *,
    world_size: int = WORLD,
    super_tile: int = 1,
    grid: int,
    inbox_memory: str = "uncached",
    codec: str = "int4",
    block: int = BLOCK,
    skip_self: bool = False,
    rank: int | None = None,
):
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
    if block not in SUPPORTED_BLOCKS:
        raise ValueError(f"block must be one of {SUPPORTED_BLOCKS}, got {block!r}")
    if skip_self and not 0 <= (rank if rank is not None else -1) < world_size:
        raise ValueError(
            f"skip_self needs the rank at trace time, got rank={rank!r} for "
            f"world_size={world_size}"
        )
    c = codecs_for_block(block)[codec]
    tile_bytes = block * ATOMS * 16
    tile_i32 = tile_bytes // 4
    quads_per_block = block // QUAD_LANES
    policy = _INBOX_POLICY[inbox_memory]
    payload_policy = policy["payload"]
    flag_policy = policy["flag"]
    release_scope = policy["release"]
    recv_policy = policy["recv"]
    if ATOMS % world_size != 0:
        raise ValueError(f"ATOMS={ATOMS} is not divisible by world_size={world_size}")
    if super_tile not in SUPER_TILES:
        raise ValueError(f"super_tile must be one of {SUPER_TILES}, got {super_tile!r}")
    if grid < 1:
        raise ValueError(f"grid must be positive, got {grid}")
    # Each rank owns this many 16-byte atoms of a 32 KiB tile
    # (8 GPUs → 1, 4 → 2, 2 → 4). LDS still holds all ATOMS atoms.
    rank_atoms = ATOMS // world_size
    # Last-sector pad is ST * rank_atoms * rank_tile_i32 after the ST tiles.
    rank_payload_i32 = rank_atoms * c.rank_tile_i32
    release_i32_off = super_tile * rank_payload_i32
    wire_tile_i32 = release_i32_off + 16
    wire_tile_bytes = wire_tile_i32 * 4

    # This rank's own index as a trace-time constant, or None when the self
    # slot is being used.
    self_rank = int(rank) if skip_self else None
    # Destinations this rank pushes packets and flags to, in rank order. Every
    # per-destination structure below -- LDS pack rows, fanout quads, flag
    # lanes -- is indexed by position in this list, not by rank.
    push_peers = [p for p in range(world_size) if p != self_rank]
    n_push = len(push_peers)

    # A rank-tile's sectors, in stripes of up to 8, one quad per (destination,
    # sector) of a stripe, in a single pass.
    stripe_width = min(8, quads_per_block // n_push)
    stripes = [
        (b, min(stripe_width, c.n_sectors - b))
        for b in range(0, c.n_sectors, stripe_width)
    ]
    sector_fastest = policy["fanout"] != "peer"

    # One pack row per (destination, rank-atom).
    pack_rows = n_push * rank_atoms
    pack_i32 = pack_rows * c.rank_tile_i32
    lds_bytes = pack_rows * c.rank_tile_bytes
    PackStorage = make_pack_storage(pack_i32)

    # flags_i32 is also the i32 offset of the wire area, so the flag prefix has
    # to be a whole number of 64 B sectors (16 i32s). At a smaller multiple every
    # rank-tile and release sector straddles two hardware sectors, so the 64 B
    # fanout stores and the last-sector release stop being one sector wide.
    grid_multiple = 16 // (PHASES * world_size)
    if grid % grid_multiple != 0:
        raise ValueError(
            f"grid must be a multiple of {grid_multiple} at "
            f"world_size={world_size} to keep the wire area 64 B aligned, got "
            f"{grid}"
        )
    flags_i32 = PHASES * grid * world_size

    @flyc.kernel(known_block_size=[block, 1, 1])
    def quick_allreduce_int4(
        rank: Int32,
        nbytes: Int64,
        num_tiles: Int32,
        inp_ptr: Int64,
        out_ptr: Int64,
        peer_ptrs: Int64,
        colors_ptr: Int64,
        n_blocks: Int32,
    ):
        _clamp_fp16_overflow()
        tid = fx.Int32(gpu.thread_id("x"))
        bid = fx.Int32(gpu.block_id("x"))

        wave, lane = thread_lane(tid, block)
        quad_layout = fx.make_layout((QUADS_PER_WAVE, QUAD_LANES), (QUAD_LANES, 1))
        quad, lane_in_quad = fx.idx2crd(lane, quad_layout).unpack()
        quad_id = wave * fx.Int32(QUADS_PER_WAVE) + quad

        pack_layout = fx.make_layout(
            (pack_rows, c.rank_tile_i32), (c.rank_tile_i32, 1)
        )
        # 64 B NT sectors of one rank-tile: (sector, lane-in-quad) -> i32
        # start of the dwordx4. Isolated NT store stays explicit.
        nt_own_layout = fx.make_layout((c.n_sectors, QUAD_LANES), (16, 4))
        # Remote NT fanout stays explicit global_store_dwordx4 nt.
        hbm_layout = fx.make_layout(
            (num_tiles, ATOMS, block * 4),
            (tile_i32, block * 4, 1),
        )
        hbm_row_layout = fx.make_layout((1, block * 4), (block * 4, 1))
        hbm_copy_atom = fx.make_copy_atom(rocdl.BufferCopy128b(), fx.Int32)
        hbm_copy = fx.make_tiled_copy_tv(
            hbm_copy_atom,
            fx.make_layout((1, block), (1, 1)),
            fx.make_layout((1, 4), (1, 1)),
        ).get_slice(tid)
        # Four group-16 E4M3 bytes share the i32 slot eight threads already own.
        scale_slot, pair_in_slot = scale_slot_of(tid, block)
        color_layout = fx.make_layout((grid,), (1,))
        wire_slot_layout = fx.make_layout(
            (PHASES, grid, world_size, super_tile),
            (
                grid * world_size * wire_tile_i32,
                world_size * wire_tile_i32,
                wire_tile_i32,
                rank_payload_i32,
            ),
        )

        lds = fx.SharedAllocator().allocate(PackStorage).peek()
        pack = lds.pack.view(pack_layout)
        smem_ptr = lds.pack.ptr

        peer_rsrc = buffer_ops.create_buffer_resource_from_addr(peer_ptrs)
        peers = [
            buffer_ops.buffer_load(peer_rsrc, i, vec_width=1, dtype=T.i64)
            for i in range(world_size)
        ]
        peer_vec = fx.Vector.from_elements(peers, dtype=fx.Int64)
        self_rsrc = buffer_ops.create_buffer_resource_from_addr(
            _to_sgpr_i64(peer_vec[rank])
        )
        def _push_base(j):
            """Inbox base of destination *j*, a lane-varying ``push_peers`` index."""

            base = fx.Int64(peers[push_peers[0]])
            for i in range_constexpr(1, n_push):
                base = (j == fx.Int32(i)).select(fx.Int64(peers[push_peers[i]]), base)
            return base
        # inp/out are a 3-D i32 tensor consumed by TiledCopy (BufferCopy128b).
        # That API needs a FlyDSL buffer-backed tensor (layout + descriptor),
        # not a raw descriptor. create_buffer_resource_from_addr is the
        # scalar-offset buffer_load/store path used for the peer-pointer
        # table, IPC inbox, and color flags. num_records_bytes is the live
        # tensor size so a partial last tile is out-of-range safe.
        hbm_i32_ptr = fx.PointerType.get(
            T.i32, address_space=fx.AddressSpace.Global, alignment=16
        )

        def _payload_tensor(ptr):
            view = fx.make_view(fx.inttoptr(hbm_i32_ptr, ptr), hbm_layout)
            return rocdl.make_buffer_tensor(
                view, max_size=False, num_records_bytes=nbytes
            )

        in_buf = _payload_tensor(inp_ptr)
        out_buf = _payload_tensor(out_ptr)
        color_rsrc = buffer_ops.create_buffer_resource_from_addr(colors_ptr)

        def _pack_off(peer, i32_idx):
            return fx.get_scalar(fx.crd2idx((peer, i32_idx), pack_layout))

        def _sub_tile_i32(phase, src, sub):
            slot = fx.get_scalar(
                fx.crd2idx((fx.Int32(phase), bid, src, sub), wire_slot_layout)
            )
            return fx.Int32(flags_i32) + slot

        def _hbm_atom_row(buf, tile, atom):
            return fx.make_view(
                fx.get_iter(fx.slice(buf, (tile, atom, None))),
                hbm_row_layout,
            )

        def _load_color():
            off = fx.get_scalar(fx.crd2idx((bid,), color_layout))
            return fx.Int32(
                buffer_ops.buffer_load(color_rsrc, off, vec_width=1, dtype=T.i32)
            )

        def _store_color(color):
            off = fx.get_scalar(fx.crd2idx((bid,), color_layout))
            buffer_ops.buffer_store(color, color_rsrc, off)

        def _load_tile_atoms(tile):
            return [_load_atom(tile, atom) for atom in range_constexpr(ATOMS)]

        def _load_atom(tile, atom):
            src = hbm_copy.partition_S(_hbm_atom_row(in_buf, tile, atom))
            frag = fx.make_fragment_like(src)
            fx.copy(hbm_copy_atom, src, frag)
            return _atom_bf16_to_f16(fx.Vector(frag.load()))

        def _store_atom(tile, atom, value):
            dst = hbm_copy.partition_D(_hbm_atom_row(out_buf, tile, atom))
            frag = fx.make_fragment_like(dst)
            frag.store(_atom_f16_to_bf16(value))
            fx.copy(hbm_copy_atom, frag, dst)

        def _store_tile_atoms(tile, atoms):
            """Store a gathered tile; ``None`` marks atoms already stored."""
            for atom in range_constexpr(ATOMS):
                if const_expr(atoms[atom] is not None):
                    _store_atom(tile, atom, atoms[atom])

        def _own_atoms(atoms):
            """This rank's reduce-scatter share, out of a whole tile's atoms."""
            return atoms[self_rank * rank_atoms : (self_rank + 1) * rank_atoms]

        def _add_f16(a, b):
            """Packed fp16 ``a + b`` of two atoms -- the codec's own FP16 add."""
            return fx.Vector.from_elements(
                [_i32(_f16x2(a[i]) + _f16x2(b[i])) for i in range_constexpr(4)],
                fx.Int32,
            )

        def _lds_write_packet(slot, words, scale, is_leader):
            for (off, pred), word in zip(c.plane_slots(tid), words):
                if pred:
                    fx.memref_store(word, pack, (slot, off))
            if const_expr(c.has_scale):  # noqa: SIM102
                if is_leader:
                    fx.memref_store(
                        scale, pack, (slot, fx.Int32(c.scale_i32_off) + scale_slot)
                    )

        def _pack_reduce_scatter(atoms):
            """Quantize each destination's slice of this tile into LDS.

            *atoms* is this thread's share of the whole tile, as loaded by
            ``_load_tile_atoms``: always ``ATOMS`` (8) 16 B atoms. Destination *d*
            owns ``atoms[d * rank_atoms : (d+1) * rank_atoms]``. Those packets
            are later NT-stored into *d*'s reduce-scatter inbox. Under
            skip_self our own slice is never packed: it stays in registers.
            """
            for j, dest in enumerate(push_peers):
                for k in range_constexpr(rank_atoms):
                    words, scale, is_leader = _codec_quant(
                        c, atoms[dest * rank_atoms + k], lane, tid
                    )
                    _lds_write_packet(
                        fx.Int32(j * rank_atoms + k), words, scale, is_leader
                    )

        def _pack_all_gather(accs):
            """Quantize the reduced slice and replicate it for every peer.

            After reduce-scatter this rank holds ``rank_atoms`` reduced
            atoms. Copy the same packets into every destination slot so the
            NT fanout can push them into every peer's all-gather inbox.
            """
            own = [] if const_expr(self_rank is not None) else None
            for k in range_constexpr(rank_atoms):
                words, scale, is_leader = _codec_quant(c, accs[k], lane, tid)
                for j in range_constexpr(n_push):
                    _lds_write_packet(
                        fx.Int32(j * rank_atoms + k), words, scale, is_leader
                    )
                if const_expr(own is not None):
                    own.append(
                        _codec_dequant(
                            c, words, _scale_from_word(c, scale, pair_in_slot), tid
                        )
                    )
            return own

        def _fanout_nt(phase, inbox_src, sub):
            """NT-store one rank-tile from LDS to every destination's inbox.

            Lockstep stripes of up to 8 sectors cover the rank-tile: at the
            default block INT4 is 8+8+2 (16 nibble sectors then the 2-sector
            E4M3 tail), fp16 is eight full stripes. ``sector_base`` is the first
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
            # retry cannot be served from a stale line. The acquire the
            # payload reads need is in _wait_release, once, after the join.
            current = _load_flag(flag)
            while current != color:
                current = _load_flag(flag)

        def _wait_release(phase, color):
            # Lane ``t`` watches one source. Without skip_self that is source
            # ``t``; with it our own flag is never published, so the N-1 lanes
            # step over our own index.
            spin_src = tid
            if const_expr(self_rank is not None):  # noqa: SIM102
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
            # of this workgroup reading the payload with nothing invalidated
            # on their behalf -- and would also skip it entirely in the common
            # case where the flag is already set on the first read.
            _acquire_inbox()

        def _recv_quantized(phase, src, sub, k=0):
            base = _sub_tile_i32(phase, src, sub)
            if const_expr(k):
                base = base + fx.Int32(k * c.rank_tile_i32)

            def _get(off):
                return _load_i32_nt(self_rsrc, base + off, recv_policy)

            words, word = _codec_load(c, _get, tid, scale_slot)
            return words, _scale_from_word(c, word, pair_in_slot)

        def _reduce_scattered(sub, own=None):
            """Dequant-accumulate every peer's reduce-scatter packet for *sub*."""
            accs = [None] * rank_atoms
            for src in range_constexpr(world_size):
                for k in range_constexpr(rank_atoms):
                    # self_rank is None if self-skip is disabled.
                    if const_expr(src == self_rank):
                        if const_expr(accs[k] is None):
                            accs[k] = own[k]
                        else:
                            accs[k] = _add_f16(own[k], accs[k])
                    else:
                        words, scale = _recv_quantized(
                            PHASE_REDUCE_SCATTER, fx.Int32(src), sub, k
                        )
                        if const_expr(accs[k] is None):
                            accs[k] = _codec_dequant(c, words, scale, tid)
                        else:
                            accs[k] = _codec_dequant(c, words, scale, tid, accs[k])
            return accs

        def _recv_all_gather(sub, own=None):
            """Dequantize every peer's all-gather packet back into full-tile atoms."""
            gathered = []
            for src in range_constexpr(world_size):
                for k in range_constexpr(rank_atoms):
                    # self_rank is None if self-skip is disabled.
                    if const_expr(src == self_rank):
                        gathered.append(None if own is None else own[k])
                    else:
                        words, scale = _recv_quantized(
                            PHASE_ALL_GATHER, fx.Int32(src), sub, k
                        )
                        gathered.append(_codec_dequant(c, words, scale, tid))
            return gathered

        # Stride by the *launched* grid, not the compile-time cap. The host
        # launches fewer blocks than `grid` whenever it wants each block to own
        # several tiles (see QuickAllReduceInt4._grid_x); striding by the cap instead would
        # silently leave every tile above n_blocks unprocessed. `grid` still
        # sizes the wire slots and colour array, so n_blocks <= grid always.
        n_block_tiles = (num_tiles - bid + n_blocks - fx.Int32(1)) // n_blocks
        color = _load_color()
        if const_expr(super_tile == 1):
            for i in range(fx.Int32(0), n_block_tiles, fx.Int32(1)):
                tile = bid + i * n_blocks
                atoms = _load_tile_atoms(tile)
                _pack_reduce_scatter(atoms)
                gpu.barrier()
                _fanout_nt(PHASE_REDUCE_SCATTER, rank, fx.Int32(0))
                _publish(PHASE_REDUCE_SCATTER, rank, color)

                _wait_release(PHASE_REDUCE_SCATTER, color)
                # ST=1 keeps the whole tile in registers across the wait, so
                # under skip_self our share is simply read back out of it.
                own_rs = None
                if const_expr(self_rank is not None):
                    own_rs = _own_atoms(atoms)
                acc = _reduce_scattered(fx.Int32(0), own_rs)

                own_ag = _pack_all_gather(acc)
                gpu.barrier()
                _fanout_nt(PHASE_ALL_GATHER, rank, fx.Int32(0))
                _publish(PHASE_ALL_GATHER, rank, color)

                _wait_release(PHASE_ALL_GATHER, color)
                gathered = _recv_all_gather(fx.Int32(0), own_ag)
                _store_tile_atoms(tile, gathered)

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
                    # Under skip_self nothing is carried from the first loop --
                    # that would be ST tiles of registers across the wait -- so
                    # our share is reloaded from the input, a local cached read
                    # against the uncached inbox read it replaces. Likewise our
                    # reduced chunk is stored now rather than carried to the
                    # third loop; the all-gather publish below releases it.
                    tile = bid + (i + s) * n_blocks
                    own_rs = None
                    if const_expr(self_rank is not None):
                        own_rs = [
                            _load_atom(tile, self_rank * rank_atoms + k)
                            for k in range_constexpr(rank_atoms)
                        ]
                    acc = _reduce_scattered(s, own_rs)
                    own_ag = _pack_all_gather(acc)
                    if const_expr(own_ag is not None):
                        for k in range_constexpr(rank_atoms):
                            _store_atom(tile, self_rank * rank_atoms + k, own_ag[k])
                    gpu.barrier()
                    _fanout_nt(PHASE_ALL_GATHER, rank, s)
                    if (s + fx.Int32(1)) < n_this:
                        rocdl.s_waitcnt(lgkmcnt=0)
                        gpu.barrier()

                _publish(PHASE_ALL_GATHER, rank, color)
                _wait_release(PHASE_ALL_GATHER, color)

                for s in range(fx.Int32(0), n_this, fx.Int32(1)):
                    gathered = _recv_all_gather(s)
                    tile = bid + (i + s) * n_blocks
                    _store_tile_atoms(tile, gathered)

                color = color + fx.Int32(1)
                if color == fx.Int32(0):  # 0 is unset sentinel
                    color = fx.Int32(1)
        if tid == 0:
            _store_color(color)
        gpu.barrier()

    flat_wg = f"{block},{block}"

    @flyc.jit
    def launch_quick_allreduce_int4(
        rank: Int32,
        nbytes: Int64,
        num_tiles: Int32,
        inp_ptr: Int64,
        out_ptr: Int64,
        peer_ptrs: Int64,
        colors_ptr: Int64,
        grid_x: Int32,
        stream: Stream = Stream(None),  # noqa: B008
    ):
        quick_allreduce_int4(
            rank,
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

    tag = f"ws{world_size}_st{super_tile}_g{grid}_{inbox_memory}_{codec}"
    tag += f"_b{block}"
    if skip_self:
        # ``_r<n>_`` is the rank field the bench's variant comparison already
        # collapses before checking that the ranks agree.
        tag += f"_r{self_rank}_ss"
    launch_quick_allreduce_int4.func.__name__ = f"launch_quick_allreduce_int4_{tag}"
    try:
        quick_allreduce_int4.func.__name__ = f"quick_allreduce_int4_{tag}"
    except AttributeError:
        pass
    return {
        "launch": launch_quick_allreduce_int4,
        "flags_bytes": flags_i32 * 4,
        "data_bytes": PHASES * grid * world_size * wire_tile_bytes,
        "lds_bytes": lds_bytes,
        "tile_bytes": tile_bytes,
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
        "skip_self": skip_self,
    }
