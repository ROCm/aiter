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
  is decoded from the same packet it sends. It needs the rank at trace time,
  which costs one binary per rank.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T

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
from .quick_allreduce_fusions import (
    ATOM_ELEMS,
    FUSIONS,
    make_rowbuf_atom_row,
    make_wave_partials,
    pack_bf16,
    quick_reduce_row_block_at,
    residual_add,
    rms_rstd,
    scale_by_weight,
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
    WAVE,
    WORLD,
    _acquire_inbox,
    _buffer_load,
    _buffer_ptr,
    _color_io,
    _i32_to_bytes,
    _load_flag,
    _load_peers,
    _payload_io,
    _release_inbox,
    _store_flag_peer,
    _store_v4i32_peer,
    _to_sgpr_i64,
    make_hbm_operand,
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
    "make_quick_allreduce_mesh_kernel",
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
    """
    if requested < 1 or cu_count < 1:
        raise ValueError("grid_cap and cu_count must be positive")
    if block < 1:
        raise ValueError("block must be positive")
    if arch not in ("gfx942", "gfx950"):
        raise ValueError(
            f"quick_allreduce_mesh has no residency measurement for {arch!r}"
        )
    key = (int(world_size), int(super_tile))
    resident = _RESIDENT_WGS_PER_CU.get(key)
    if resident is None:
        for_world = [v for (w, _st), v in _RESIDENT_WGS_PER_CU.items() if w == key[0]]
        if not for_world:
            raise ValueError(
                "quick_allreduce_mesh has no residency measurement for "
                f"world_size={world_size}"
            )
        resident = min(for_world)
    if block > BLOCK:
        resident = max(1, (resident * BLOCK) // int(block))
    return min(int(requested), resident * int(cu_count))


# Per-``(link, world_size)`` tuning ladder: ``(min_bytes, super_tile, grid_cap,
# block, skip_self)`` rungs, ascending.
MESH_ST_LADDER = {
    ("xgmi", 2): ((0, 1, 128, 256, True), (4 << 20, 1, 128, 512, True)),
    ("xgmi", 4): ((0, 8, 128, 256, True), (4 << 20, 1, 128, 512, True)),
    ("xgmi", 8): (
        (0, 1, 128, 256, True),
        (4 << 20, 1, 128, 512, True),
        (6 << 20, 8, 128, 512, True),
    ),
    ("pcie", 2): ((0, 1, 128, 256, True), (4 << 20, 8, 128, 512, True)),
    ("pcie", 4): ((0, 1, 128, 256, True), (768 << 10, 8, 128, 512, True)),
    ("pcie", 8): ((0, 1, 128, 256, True), (96 << 10, 8, 128, 512, True)),
}


def mesh_st_ladder(world_size: int, link: str = "pcie"):
    """Rungs for *(link, world_size)*, or ``()`` when there is no ladder."""
    return MESH_ST_LADDER.get((str(link), int(world_size)), ())


# Wire formats the mesh can build.
MESH_CODECS = ("int4", "fp16")


def mesh_fanout_quads(block: int, world_size: int, codec: str) -> tuple[int, int]:
    """``(quads the block has, quads the mesh fanout needs)`` at this geometry.

    A rank-tile's sectors go out in stripes of up to 8, one quad per
    ``(peer, sector)`` of a stripe, in a single pass. A block with fewer quads
    than that would silently drop the sectors past the end.
    """
    c = codecs_for_block(int(block))[codec]
    stripes = [(b, min(8, c.n_sectors - b)) for b in range(0, c.n_sectors, 8)]
    return int(block) // QUAD_LANES, max(int(world_size) * w for _, w in stripes)


def mesh_fanout_fits(block: int, world_size: int, codec: str) -> bool:
    """Whether the mesh fanout fits in *block*'s quads. For host-side gates."""
    have, need = mesh_fanout_quads(block, world_size, codec)
    return have >= need


def make_quick_allreduce_mesh_kernel(
    *,
    world_size: int = WORLD,
    super_tile: int = 1,
    grid: int,
    inbox_memory: str = "uncached",
    codec: str = "int4",
    fusion: str = "none",
    hidden: int | None = None,
    block: int | None = None,
    h_pad: int | None = None,
    skip_self: bool = False,
    rank: int | None = None,
):
    if fusion not in FUSIONS:
        raise ValueError(f"fusion must be one of {FUSIONS}, got {fusion!r}")
    if fusion != "none" and hidden is None:
        raise ValueError(f"fusion={fusion!r} requires hidden")
    if fusion == "none" and hidden is not None:
        raise ValueError("hidden is only meaningful for a fused build")
    if fusion == "none" and h_pad is not None:
        raise ValueError("h_pad is only meaningful for a fused build")
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
    if skip_self and not 0 <= (rank if rank is not None else -1) < world_size:
        raise ValueError(
            f"skip_self needs the rank at trace time, got rank={rank!r} for "
            f"world_size={world_size}"
        )

    fused = fusion == "rmsnorm"
    if fused:
        # ``h_pad`` is the width the *workgroup* covers, ``hidden`` the width the
        # *tensor* has. The block, the tile, the codec and the wire all follow
        # h_pad; the HBM row stride and the RMS denominator follow hidden.
        h_pad = hidden if h_pad is None else int(h_pad)
        if h_pad < hidden:
            raise ValueError(f"h_pad={h_pad} is narrower than hidden={hidden}")
        if h_pad != hidden and hidden % ATOM_ELEMS:
            raise ValueError(
                f"a padded fused build needs hidden to be a whole number of "
                f"{ATOM_ELEMS}-element atoms so the pad boundary lands on an "
                f"atom granule, got hidden={hidden}"
            )
        block, atoms_per_row = quick_reduce_row_block_at(h_pad, world_size, block)
    else:
        # The plain kernel takes a caller-chosen block; the fused build derives
        # its own from the row geometry just above.
        block = BLOCK if block is None else block
        if block not in SUPPORTED_BLOCKS:
            raise ValueError(f"block must be one of {SUPPORTED_BLOCKS}, got {block!r}")
        atoms_per_row = 1
    padded = fused and h_pad != hidden
    rows_per_tile = ATOMS // atoms_per_row
    n_waves = block // WAVE
    tile_i32 = block * ATOMS * 4
    tile_bytes = tile_i32 * 4
    # i32 between one token row and the next *in HBM*, and the HBM bytes a whole
    # tile spans. These part company from ``tile_i32``/``tile_bytes`` exactly
    # when a build is padded: the wire tile is rows_per_tile rows of h_pad, the
    # HBM footprint is rows_per_tile rows of hidden. The host derives num_tiles
    # from ``hbm_tile_bytes``, never from ``tile_bytes``.
    row_stride_i32 = (hidden // 2) if fused else tile_i32
    hbm_tile_i32 = rows_per_tile * row_stride_i32 if fused else tile_i32
    hbm_tile_bytes = hbm_tile_i32 * 4

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

    # One quad per (destination, sector) of a stripe, in a single pass -- unlike
    # the ring, which loops rounds. A block narrower than the shipped 256 would
    # run out of quads and silently drop the sectors past the end, so a fused
    # build that lands there is rejected rather than built. (The plain path can
    # still narrow ``stripe_width`` below; only fused derives its own block.)
    if const_expr(fused):
        _, need_quads = mesh_fanout_quads(block, world_size, codec)
        if quads_per_block < need_quads:
            raise ValueError(
                f"fused hidden={hidden} gives BLOCK={block} ({quads_per_block} "
                f"quads), under the {need_quads} the {codec} fanout needs at "
                f"world_size={world_size}; use a wider hidden or the ring schedule"
            )

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

    # Per-wave partials for the fused sum of squares, one slot per row of a
    # tile. Allocated from the same allocator as the pack staging (FlyDSL
    # permits one per kernel) but as its own region, so it cannot alias a
    # packet the fanout is still reading.
    n_partials = rows_per_tile * n_waves if (fused and n_waves > 1) else 0
    WavePartials = make_wave_partials(n_partials) if n_partials else None
    lds_bytes += n_partials * 4
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

    # One signature for both modes; the plain launcher below passes zeros for
    # the fused operands, which ``const_expr(fused)`` elides every use of. See
    # the ring kernel for why the split is at the launcher and not here.
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
    ):
        _clamp_fp16_overflow()
        tid = fx.Int32(gpu.thread_id("x"))
        bid = fx.Int32(gpu.block_id("x"))

        wave, lane = thread_lane(tid, block)
        quad_layout = fx.make_layout((QUADS_PER_WAVE, QUAD_LANES), (QUAD_LANES, 1))
        quad, lane_in_quad = fx.idx2crd(lane, quad_layout).unpack()
        quad_id = wave * fx.Int32(QUADS_PER_WAVE) + quad

        pack_layout = fx.make_layout((pack_rows, c.rank_tile_i32), (c.rank_tile_i32, 1))
        # 64 B NT sectors of one rank-tile: (sector, lane-in-quad) -> i32
        # start of the dwordx4. Isolated NT store stays explicit.
        nt_own_layout = fx.make_layout((c.n_sectors, QUAD_LANES), (16, 4))
        # The fused epilogue reaches its HBM operands (residual, weight, final
        # output) as raw bf16 atoms through a tiled copy; the plain reduce path
        # uses the shared ``_payload_io`` helpers below and never builds these.
        # ``hbm_row_layout`` (one atom-wide row) and ``hbm_copy`` serve both fused
        # sub-modes; a padded build addresses each row through a per-row bounded
        # descriptor (``_rowbuf_atom_row``), an unpadded one slices the
        # whole-tensor buffer tensor (``_hbm_atom_row``). ``hbm_layout`` is the
        # 3-D whole-tensor layout consumed only by the unpadded
        # ``make_hbm_operand``; a padded build leaves it None.
        hbm_i32_ptr = None
        hbm_layout = None
        hbm_row_layout = None
        hbm_copy_atom = None
        hbm_copy = None
        if const_expr(fused):
            hbm_i32_ptr = fx.PointerType.get(
                T.i32, address_space=fx.AddressSpace.Global, alignment=16
            )
            if const_expr(not padded):
                hbm_layout = fx.make_layout(
                    (num_tiles, ATOMS, block * 4),
                    (hbm_tile_i32, block * 4, 1),
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
        wire_slot_layout = fx.make_layout(
            (PHASES, grid, world_size, super_tile),
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
        # A second region from the same allocator -- FlyDSL permits one per
        # kernel -- hoisted here because it is static: an allocation reached
        # from inside the tile loop would emit one LDS symbol per visit.
        sq_lds = None
        if const_expr(n_partials > 0):
            sq_lds = allocator.allocate(WavePartials).peek().wave.ptr

        peers = _load_peers(peer_ptrs, world_size)
        peer_vec = fx.Vector.from_elements(peers, dtype=fx.Int64)
        inbox = _buffer_ptr(_to_sgpr_i64(peer_vec[rank]), T.i32, 4)

        def _push_base(j):
            """Inbox base of destination *j*, a lane-varying ``push_peers`` index."""

            base = peers[push_peers[0]]
            for i in range_constexpr(1, n_push):
                base = (j == fx.Int32(i)).select(peers[push_peers[i]], base)
            return base

        # The plain reduce path moves whole tiles through the shared payload
        # helpers; the fused epilogue below reaches the same rows as raw bf16
        # atoms through the tiled copy instead.
        _load_atom, _store_atom = _payload_io(
            inp_ptr,
            out_ptr,
            nbytes,
            num_tiles,
            ATOMS,
            block,
            tid,
            decode=_atom_bf16_to_f16,
            encode=_atom_f16_to_bf16,
        )
        _load_color, _store_color = _color_io(colors_ptr, bid)

        if const_expr(fused):
            # A padded build keeps the HBM operands as raw ``Int64`` base
            # pointers so ``_rowbuf_atom_row`` can bound a fresh descriptor per
            # row; an unpadded build wraps them in the whole-tensor buffer tensor.
            _operand = make_hbm_operand(
                padded=padded,
                nbytes=nbytes,
                hbm_i32_ptr=hbm_i32_ptr,
                hbm_layout=hbm_layout,
            )
            # A padded build addresses each row through a per-row buffer
            # descriptor bounded to the true width, and to nothing at all for a
            # row past M in a partial last tile; ``_rowbuf_atom_row(ptr, tile,
            # atom)`` builds it from the operand's raw base pointer. Unpadded is
            # None (unused).
            _rowbuf_atom_row = (
                make_rowbuf_atom_row(
                    atoms_per_row=atoms_per_row,
                    rows_per_tile=rows_per_tile,
                    row_stride_i32=row_stride_i32,
                    block=block,
                    hidden=hidden,
                    hbm_i32_ptr=hbm_i32_ptr,
                    hbm_row_layout=hbm_row_layout,
                    nbytes=nbytes,
                )
                if padded
                else None
            )

            # residual in/out share the payload's (M, hidden) bf16 shape, so they
            # ride the same addressing, as does the final output row. The gain is
            # one (hidden,) row -- ``atoms_per_row`` atoms at tile 0; a padded
            # build reads it through the per-row descriptor at row 0, an unpadded
            # one bounds a one-row buffer tensor at the true width (the whole mask
            # for this operand, since nothing lies past it).
            out_buf = _operand(out_ptr)
            res_in_buf = _operand(res_in_ptr)
            res_out_buf = _operand(res_out_ptr)
            w_buf = _operand(w_ptr, records=fx.Int64(hidden * 2))

        def _pack_off(peer, i32_idx):
            return fx.get_scalar(fx.crd2idx((peer, i32_idx), pack_layout))

        def _sub_tile_i32(phase, src, sub):
            slot = fx.get_scalar(
                fx.crd2idx((fx.Int32(phase), bid, src, sub), wire_slot_layout)
            )
            return fx.Int32(flags_i32) + slot

        if const_expr(fused):

            def _hbm_atom_row(buf, tile, atom):
                return fx.make_view(
                    fx.get_iter(fx.slice(buf, (tile, atom, None))),
                    hbm_row_layout,
                )

            # The row-view function is chosen once at trace time: a padded build
            # addresses each row through a per-row bounded descriptor, an
            # unpadded one slices the whole-tensor buffer tensor. Both return a
            # one-atom-wide row view that ``partition_S/D`` consume, so the copy
            # bodies are uniform.
            _atom_row = _rowbuf_atom_row if padded else _hbm_atom_row

        def _load_tile_atoms(tile):
            return [_load_atom(tile, atom) for atom in range_constexpr(ATOMS)]

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

        if const_expr(fused):

            def _load_raw_atom(buf, tile, atom):
                """One 16 B atom, unconverted -- the fused epilogue's bf16
                operands are not codec values and never become fp16."""
                src = hbm_copy.partition_S(_atom_row(buf, tile, atom))
                frag = fx.make_fragment_like(src)
                fx.copy(hbm_copy_atom, src, frag)
                return fx.Vector(frag.load())

            def _store_raw_atom(buf, tile, atom, value):
                dst = hbm_copy.partition_D(_atom_row(buf, tile, atom))
                frag = fx.make_fragment_like(dst)
                frag.store(value)
                fx.copy(hbm_copy_atom, frag, dst)

            # The reduce-scatter input read must reach ``inp`` through the same
            # per-row bounded descriptor the epilogue uses, not the plain path's
            # whole-payload ``_payload_io`` tensor. At h_pad != hidden the wire
            # tile is wider than the HBM row, so ``_payload_io``'s single bound
            # lets a pad lane read the next row instead of zero -- which then
            # rides the wire into residual_out. Re-home ``_load_tile_atoms`` onto
            # the raw-atom route (bf16 -> f16 for the codec), shadowing the plain
            # definition above for the fused build only.
            in_buf = _operand(inp_ptr)

            def _load_tile_atoms(tile):
                atoms = []
                for atom in range_constexpr(ATOMS):
                    atoms.append(_atom_bf16_to_f16(_load_raw_atom(in_buf, tile, atom)))
                return atoms

            def _atom_f16_to_f32(atom):
                """Packed fp16 -> 8 f32. A widening move; exact, no rounding."""
                return fx.Vector(atom).bitcast(fx.Float16).to(fx.Float32)

            def _epilogue(tile, gathered, w_atoms):
                """Residual add + RMSNorm over the whole tile.

                The mesh ends a tile holding every atom of it, and the fused
                geometry makes ``atoms_per_row`` of them one token row, so the
                tile is ``rows_per_tile`` independent rows and each reduces inside
                this block. All of them go through one ``rms_rstd`` call, which is
                one barrier rather than one per row.
                """
                rows = []
                for r in range_constexpr(rows_per_tile):
                    idx = [
                        r * atoms_per_row + a for a in range_constexpr(atoms_per_row)
                    ]
                    xs = residual_add(
                        [_atom_f16_to_f32(gathered[j]) for j in idx],
                        [_load_raw_atom(res_in_buf, tile, j) for j in idx],
                    )
                    rows.append(xs)
                    # residual_out is final here, so it is issued ahead of the
                    # reduction rather than behind its barrier.
                    packed = pack_bf16(xs)
                    for a in range_constexpr(atoms_per_row):
                        _store_raw_atom(res_out_buf, tile, idx[a], packed[a])

                rstds = rms_rstd(rows, eps, hidden, tid=tid, block=block, lds=sq_lds)
                for r in range_constexpr(rows_per_tile):
                    outs = scale_by_weight(rows[r], rstds[r], w_atoms)
                    for a in range_constexpr(atoms_per_row):
                        _store_raw_atom(out_buf, tile, r * atoms_per_row + a, outs[a])
                if const_expr(n_partials > 0):
                    # The next tile reuses these slots, and the super-tile store
                    # loop runs back-to-back with no barrier of its own between
                    # iterations -- this is what separates its writes from reads.
                    gpu.barrier()

        def _finish_tile(tile, gathered, w_atoms):
            if const_expr(fused):
                _epilogue(tile, gathered, w_atoms)
            else:
                _store_tile_atoms(tile, gathered)

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
            # retry cannot be served from a stale line. The acquire that
            # orders the payload reads is in _wait_release, once, after the
            # join.
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

        # Self-skip at ST>1 carries nothing in registers between the three
        # super-tile loops; these move our own share and reduced chunk instead.
        def _load_own_share(tile, k):
            """Our reduce-scatter share of *tile*, reloaded from the input. A
            fused build reads it through the same per-row route as
            ``_load_tile_atoms``, so a padded row's pad lanes come back zero."""
            atom = self_rank * rank_atoms + k
            if const_expr(fused):
                return _atom_bf16_to_f16(_load_raw_atom(in_buf, tile, atom))
            return _load_atom(tile, atom)

        def _stash_own(tile, k, value):
            """Park our reduced chunk in ``out`` until the tile is finished.

            A plain build's store here is already its final one. A fused build
            parks the raw fp16 bits in the same slot for ``_unstash_own`` --
            exact, where a bf16 store would round -- and the epilogue then
            overwrites them. The all-gather publish between the two drains
            ``vmcnt``, so the reload sees the store.
            """
            atom = self_rank * rank_atoms + k
            if const_expr(fused):
                _store_raw_atom(out_buf, tile, atom, value)
            else:
                _store_atom(tile, atom, value)

        def _unstash_own(tile):
            """The own slot for ``_recv_all_gather`` in the third loop: what
            ``_stash_own`` parked on a fused self-skip build, else ``None`` (no
            self-skip, or a plain build whose store was final)."""
            if const_expr(fused and self_rank is not None):
                return [
                    _load_raw_atom(out_buf, tile, self_rank * rank_atoms + k)
                    for k in range_constexpr(rank_atoms)
                ]
            return None

        # One row shared by every token, so the gain is read once here rather
        # than once per tile.
        w_atoms = None
        if const_expr(fused):
            w_atoms = [
                _load_raw_atom(w_buf, fx.Int32(0), a)
                for a in range_constexpr(atoms_per_row)
            ]

        # Stride by the *launched* grid, not the compile-time cap. The host
        # launches fewer blocks than `grid` whenever it wants each block to own
        # several tiles (see FlyQuickAllReduce._grid_x); striding by the cap
        # instead would silently leave every tile above n_blocks unprocessed.
        # `grid` still sizes the wire slots and colour array, so
        # n_blocks <= grid always.
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
                _finish_tile(tile, gathered, w_atoms)

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
                            _load_own_share(tile, k)
                            for k in range_constexpr(rank_atoms)
                        ]
                    acc = _reduce_scattered(s, own_rs)
                    own_ag = _pack_all_gather(acc)
                    if const_expr(own_ag is not None):
                        for k in range_constexpr(rank_atoms):
                            _stash_own(tile, k, own_ag[k])
                    gpu.barrier()
                    _fanout_nt(PHASE_ALL_GATHER, rank, s)
                    if (s + fx.Int32(1)) < n_this:
                        rocdl.s_waitcnt(lgkmcnt=0)
                        gpu.barrier()

                _publish(PHASE_ALL_GATHER, rank, color)
                _wait_release(PHASE_ALL_GATHER, color)

                for s in range(fx.Int32(0), n_this, fx.Int32(1)):
                    tile = bid + (i + s) * n_blocks
                    gathered = _recv_all_gather(s, _unstash_own(tile))
                    _finish_tile(tile, gathered, w_atoms)

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
            value_attrs={"rocdl.flat_work_group_size": flat_wg},
        ).launch(
            grid=(grid_x, 1, 1),
            block=(block, 1, 1),
            stream=stream,
        )

    # The inbox memory type and wire codec both change the emitted code, so
    # both have to be part of the symbol name -- two variants that differ only
    # in cache bits or wire format must not collide in the JIT cache.
    tag = f"ws{world_size}_st{super_tile}_g{grid}_{inbox_memory}_{codec}"
    tag += f"_b{block}"
    if skip_self:
        # ``_r<n>_`` is the rank field the bench's variant comparison already
        # collapses before checking that the ranks agree.
        tag += f"_r{self_rank}_ss"
    if fused:
        # h_pad changes both the geometry and which loads carry a mask, so a
        # padded build must not share a JIT key with the plain one.
        tag += f"_rms_h{hidden}"
        if padded:
            tag += f"_p{h_pad}"
    launcher = (
        launch_quick_allreduce_mesh_fused if fused else launch_quick_allreduce_mesh
    )
    launcher.func.__name__ = f"launch_quick_allreduce_mesh_{tag}"
    try:
        quick_allreduce_mesh.func.__name__ = f"quick_allreduce_mesh_{tag}"
    except AttributeError:
        pass
    return {
        "launch": launcher,
        "flags_bytes": flags_i32 * 4,
        "data_bytes": PHASES * grid * world_size * wire_tile_bytes,
        "lds_bytes": lds_bytes,
        "tile_bytes": tile_bytes,
        # HBM bytes one tile spans, which is what a tile *count* has to come
        # from. Equal to tile_bytes unless the build is padded, where the wire
        # tile is rows_per_tile rows of h_pad but the footprint is
        # rows_per_tile rows of hidden.
        "hbm_tile_bytes": hbm_tile_bytes,
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
        "fusion": fusion,
        "hidden": hidden,
        "h_pad": h_pad,
        "padded": padded,
        "atoms_per_row": atoms_per_row,
        "rows_per_tile": rows_per_tile,
        "skip_self": skip_self,
    }
