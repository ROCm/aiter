# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""gfx942/gfx950 TP∈{2,4,8} INT4/INT6 **ring** all-reduce.

Topology of the lap: the ranks form a cycle and every step is a single
contiguous run into exactly one peer's inbox:

* ``2(N-1)`` hops and ``2(N-1)`` handshakes.
* ``2(N-1)/N`` of the payload on the wire. The ring is bandwidth
  optimal (Patarasuk & Yuan, JPDC 69(2), 2009).
* One destination per store. On a PCIe host, a GPU has a single shared x16 uplink,
  and a ring step writes ``rank_atoms * rank_tile B`` to one peer, in address
  order, with no destination switch.

The ring pays for that schedule is accuracy: its reduce-scatter lap
requantizes the *running partial sum* ``N-1`` times where the mesh requantizes
once, and the partial's extremum grows with the number of contributions folded
into it. Hence the two codec knobs. Widening the reduce-scatter lap to INT6 improves
accuracy with the cost of using slightly more bandwidth. The all-gather lap forwards
the bytes it received untouched, so it contributes exactly one quantization and stays
INT4 unless the caller pins it via the ``ag_codec`` argument.

A third wire format, ``"fp16"``, is a lossless passthrough. Mainly for testing.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T

from .collectives_shared import (
    _INBOX_POLICY,
    _SYSTEM_SYNC_SCOPE,
    ATOMS,
    BLOCK,
    DEFAULT_GRID_CAP,  # noqa: F401  -- re-exported for host symmetry
    QUAD_LANES,
    QUADS_PER_WAVE,
    RECV_BYPASS,
    SUPPORTED_WORLDS,
    WAVE,
    _buffer_ptr,
    _color_io,
    _i32_to_bytes,
    _load_peers,
    _payload_io,
    _to_sgpr_i64,
    fanout_contiguous,
    lds_write_packet,
    load_i32_drained,
    make_hbm_operand,
    make_pack_storage,
    next_color,
    publish_flags,
    wait_flags,
)
from .quick_allreduce_codec import (
    GROUP,
    SUPPORTED_BLOCKS,
    _atom_bf16_to_f16,
    _atom_f16_to_bf16,
    _clamp_fp16_overflow,
    _codec_dequant,
    _codec_load,
    _codec_quant,
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

# Super-tile values the ring accepts.
#
# The ring publishes 2(N-1) times per super-tile
# On a cacheable inbox every publish is a full L2 writeback. Publishes per rank
# are ``num_tiles / ST * 2(N-1)`` -- independent of the block count, so ST is
# the only tunable knob that reduces them, and it pays for them in parallelism:
# ``_grid_x`` derives blocks from ``num_tiles / ST``.
#
# The optimum is payload-dependent, so it is not a single value -- see
# ``RING_ST_LADDER`` below, which is what the host actually selects from. These
# are the values a caller may pin with ``super_tile=``.
#
# The buffer size is
# ``(N-1) * grid * (ST * rank_atoms * (rs_tile + ag_tile) + 128)`` bytes, so
# ST=16 at the default cap of 1216 is ~269 MB per rank all-INT4, ~329 MB with
# an INT6 reduce-scatter lap, against 28/35 MB at cap 128.
# Pin ``grid_cap`` alongside ``super_tile``.
RING_SUPER_TILES = (1, 8, 16, 32)

# Payload-size ladder: ``(min_bytes, super_tile, grid_cap, block)``, ascending,
# per ``(link, world_size)``.
_RING_DEFAULT = {
    2: ((0, 8, 128, BLOCK), (24 << 20, 16, 128, BLOCK)),
    4: (
        (0, 8, 128, BLOCK),
        (24 << 20, 16, 128, BLOCK),
        (48 << 20, 32, 128, BLOCK),
    ),
    8: ((0, 16, 128, BLOCK), (48 << 20, 32, 128, BLOCK)),
}

# ``(min_bytes, super_tile, grid_cap, block)``
RING_ST_LADDER = {
    **{("xgmi", ws): rungs for ws, rungs in _RING_DEFAULT.items()},
    ("pcie", 2): ((0, 16, 128, 512),),
    ("pcie", 4): ((0, 32, 128, 512),),
    ("pcie", 8): ((0, 32, 128, 512),),
}


def ring_st_ladder(world_size: int, link: str = "pcie"):
    """Rungs for *(link, world_size)*, or the PCIe TP4 shape for an unlisted one."""
    return RING_ST_LADDER.get((str(link), int(world_size)), RING_ST_LADDER[("pcie", 4)])


# Wire formats accepted per lap.
#
# Both laps take the same set, but they are separate arguments because they are
# separate decisions: the reduce-scatter lap requantizes ``N-1`` times and is
# what INT6 is for, while the all-gather lap forwards the bytes it received
# without touching them (see ``_op_substep``) and so contributes exactly one
# quantization -- the one at op ``N``, where this rank's chunk is final.
RS_CODECS = ("int4", "int6", "fp16")
AG_CODECS = ("int4", "int6", "fp16")

# Cache policy for reading a rank-tile out of our own inbox, on every inbox
# type. The mesh takes the policy's ``recv`` instead, which is `nt` on a
# fine-grained inbox -- only a non-temporal hint, not a bypass. For the ring
# algorithm a hint is never enough: `sc0 sc1` makes the load actually go to
# memory.
_RECV_POLICY = RECV_BYPASS


def ring_steps(world_size: int) -> int:
    """Wire slots, which is also hops on the critical path: ``2(N-1)``.

    ``N-1`` to reduce-scatter and ``N-1`` to all-gather, with the last reduce
    and the first gather-send fused into one op -- which is what keeps this at
    ``2(N-1)`` rather than ``2N``.
    """
    return 2 * (world_size - 1)


def make_quick_allreduce_ring_kernel(
    *,
    world_size: int,
    rank: int,
    super_tile: int = 1,
    grid: int,
    inbox_memory: str = "finegrained",
    rs_codec: str = "int4",
    ag_codec: str = "int4",
    fusion: str = "none",
    hidden: int | None = None,
    block: int | None = None,
    h_pad: int | None = None,
):
    """Build the ring kernel for one *rank*.

    ``rank`` is a **compile-time** parameter here, unlike the mesh kernel
    where it is a runtime argument. At op ``k`` the chunk a block works on is
    ``(rank - k) % N``, and that index selects from a Python list of
    register-resident atom fragments -- it has to be a Python constant. Baking
    it in costs nothing: a rank only ever compiles its own binary, so the
    variant count per process is unchanged. It does have to reach the JIT symbol
    name, which ``tag`` below handles.

    The kernel *signature* keeps its ``rank`` argument, unused, so the host's
    ``_launch_eng`` is identical for both schedules.

    ``block`` is threads per workgroup.

    ``fusion="rmsnorm"`` appends a residual-add + RMSNorm epilogue and requires
    ``hidden``. It changes the geometry rather than just the tail: the block is
    sized so one 16 B atom is one token row (see
    ``quick_allreduce_fusions.quick_reduce_row_block``), which is what lets the
    epilogue run inside the op that receives a chunk. Without it a row would
    span several ops, and since this schedule carries nothing across a publish,
    the only way to reassemble one would be to pin ``super_tile`` at 1 -- which
    is where the ring's throughput goes to die.
    """
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
    if not 0 <= int(rank) < world_size:
        raise ValueError(f"rank must be in [0, {world_size}), got {rank}")
    if inbox_memory not in _INBOX_POLICY:
        raise ValueError(
            f"inbox_memory must be one of {tuple(_INBOX_POLICY)}, got {inbox_memory!r}"
        )
    if rs_codec not in RS_CODECS:
        raise ValueError(f"rs_codec must be one of {RS_CODECS}, got {rs_codec!r}")
    if ag_codec not in AG_CODECS:
        raise ValueError(f"ag_codec must be one of {AG_CODECS}, got {ag_codec!r}")
    if super_tile not in RING_SUPER_TILES:
        raise ValueError(
            f"super_tile must be one of {RING_SUPER_TILES}, got {super_tile!r}"
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
    # policy["fanout"] is not consulted: it picks which axis of a (peer, sector)
    # fanout runs fastest across quads, and a ring has no peer axis. Sectors run
    # fastest by construction, which is the "peer" (PCIe-favourable) answer.

    rank_atoms = ATOMS // world_size
    steps = ring_steps(world_size)
    n_ops = 2 * world_size - 1  # ops are 1-based; op k reads slot k-2, writes k-1
    nxt = (rank + 1) % world_size

    fused = fusion == "rmsnorm"
    # An explicit block picks among the widths this hidden dim admits; None keeps
    # the widest.
    if fused:
        # ``h_pad`` is the width the *workgroup* covers, ``hidden`` the width the
        # *tensor* has. The block, the chunk, the codec and the wire follow
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
    rows_per_chunk = rank_atoms // atoms_per_row
    rows_per_tile = ATOMS // atoms_per_row
    quads_per_block = block // QUAD_LANES
    n_waves = block // WAVE
    tile_i32 = block * ATOMS * 4
    tile_bytes = tile_i32 * 4
    # i32 between one token row and the next *in HBM*, and the HBM bytes a whole
    # tile spans. These part company from ``tile_i32``/``tile_bytes`` exactly
    # when a build is padded; the host derives num_tiles from hbm_tile_bytes.
    row_stride_i32 = (hidden // 2) if fused else tile_i32
    hbm_tile_i32 = rows_per_tile * row_stride_i32 if fused else tile_i32
    hbm_tile_bytes = hbm_tile_i32 * 4

    codecs = codecs_for_block(block)
    rs = codecs[rs_codec]
    ag = codecs[ag_codec]
    # Which codec each wire slot carries. Ops 1..N-1 fill the reduce-scatter
    # slots and ops N..2N-1 the all-gather ones, so the split is exactly at
    # N-1 and is a compile-time property of the step index.
    step_codec = [rs] * (world_size - 1) + [ag] * (world_size - 1)

    # Per-step geometry. All Python ints, so every use below folds at trace
    # time -- the mixed-codec inbox costs no address arithmetic over the
    # single-codec one it replaces.
    payload_i32 = [rank_atoms * c.rank_tile_i32 for c in step_codec]
    release_i32_off = [super_tile * p for p in payload_i32]
    wire_tile_i32 = [r + 16 for r in release_i32_off]  # + one 64 B handshake sector
    step_base_i32, _acc = [], 0
    for w in wire_tile_i32:
        step_base_i32.append(_acc)
        _acc += grid * w
    inbox_bytes = _acc * 4

    # Every slot has exactly one writer -- the ring predecessor -- so the mesh's
    # per-sender axis collapses away entirely. That is what keeps the
    # atomics-free design sound here (there are no peer atomics over PCIe), and
    # it makes the buffer (N-1)/N of the mesh's rather than larger.

    # One staging buffer, sized for whichever codec needs more. Only
    # ``rank_atoms`` rows: a ring stages one destination's packet, not every
    # destination's, so this is 1664 B at TP8 against the mesh's 9216 B -- an
    # INT6 ring costs less LDS than an INT4 mesh.
    pack_row_i32 = max(rs.rank_tile_i32, ag.rank_tile_i32)
    pack_i32 = rank_atoms * pack_row_i32
    PackStorage = make_pack_storage(pack_i32)

    # Per-wave partials for the fused sum of squares, one slot per row of a
    # chunk. Allocated separately from the pack staging.
    n_partials = rows_per_chunk * n_waves if (fused and n_waves > 1) else 0
    WavePartials = make_wave_partials(n_partials) if n_partials else None
    lds_bytes = pack_i32 * 4 + n_partials * 4

    def _chunk_of(k: int) -> int:
        """Which chunk op *k* carries, following the standard ring schedule.

        Ops ``1..N`` are the reduce-scatter lap, walking chunks ``r-1`` down to
        ``r``; ops ``N+1..2N-1`` are the all-gather lap, walking ``r-1`` down to
        ``r+1``. Op ``N`` completes this rank's own chunk and is also the
        all-gather's first send.
        """
        j = k if k <= world_size else k - world_size
        return (rank - j) % world_size

    @flyc.kernel(known_block_size=[block, 1, 1])
    def quick_allreduce_ring(
        rank_unused: fx.Int32,
        nbytes: fx.Int64,
        num_tiles: fx.Int32,
        inp_ptr: fx.Int64,
        out_ptr: fx.Int64,
        peer_ptrs: fx.Int64,
        colors_ptr: fx.Int64,
        n_blocks: fx.Int32,
        # These args are discarded for non-fused kernel
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

        # The fused epilogue reaches its HBM operands (residual, weight, final
        # output) as raw bf16 atoms through a tiled copy; the plain reduce path
        # uses the shared ``_payload_io`` helpers and never builds these.
        # ``hbm_row_layout`` (one atom-wide row) and ``hbm_copy`` serve both fused
        # sub-modes; a padded build addresses each row through a per-row bounded
        # descriptor (``_rowbuf_atom_row``), an unpadded one slices the
        # whole-tensor buffer tensor (``_hbm_atom_row``). ``hbm_layout`` is the
        # 3-D whole-tensor layout consumed only by the unpadded
        # ``make_hbm_operand``; a padded build leaves it None.
        hbm_layout = None
        hbm_row_layout = None
        hbm_copy_atom = None
        hbm_copy = None
        if const_expr(fused):
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
        scale_slot, pair_in_slot = scale_slot_of(tid, block)

        # One allocation, one view per codec.
        allocator = fx.SharedAllocator()
        lds = allocator.allocate(PackStorage).peek()
        smem_ptr = lds.pack.ptr
        pack_views = {
            c.name: lds.pack.view(
                fx.make_layout((rank_atoms, c.rank_tile_i32), (c.rank_tile_i32, 1))
            )
            for c in ({rs.name: rs, ag.name: ag}).values()
        }
        # A second allocation from the same allocator -- FlyDSL allows only
        # one per kernel -- so it cannot alias the staged packet the fanout is
        # still reading. Hoisted here because the allocator is static: one
        # reached from inside the op loop would emit an LDS symbol per visit.
        sq_lds = None
        if const_expr(n_partials > 0):
            sq_lds = allocator.allocate(WavePartials).peek().wave.ptr

        peers = _load_peers(peer_ptrs, world_size)
        # A ring only ever names two of the peers, and ``rank`` is compile-time,
        # so both are plain Python indices into the loaded pointers. The
        # mesh packs these into an fx.Vector because its fanout selects a
        # peer with a *runtime* lane-dependent index; doing that here would put
        # a dynamic extract in front of a constant and get the wrong element.
        self_base = peers[rank]
        next_base = peers[nxt]
        # Bounded on purpose. Every ring access is in range by construction,
        # so an out-of-range one is a bug -- and with num_records set the
        # hardware returns zero instead of faulting, which turns a
        # process-killing page fault into a wrong SQNR you can bisect.
        inbox = _buffer_ptr(_to_sgpr_i64(self_base), T.i32, 4, inbox_bytes)

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
            hbm_i32_ptr = fx.PointerType.get(
                T.i32, address_space=fx.AddressSpace.Global, alignment=16
            )
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

            # residual in/out are (M, hidden) bf16 exactly like the payload, so
            # they ride the same addressing, as does the final output row. The
            # gain is a single (hidden,) row shared by every token --
            # ``atoms_per_row`` atoms at tile 0; a padded build reads it through
            # the per-row descriptor at row 0, an unpadded one bounds a one-row
            # buffer tensor at the true width.
            out_buf = _operand(out_ptr)
            res_in_buf = _operand(res_in_ptr)
            res_out_buf = _operand(res_out_ptr)
            w_buf = _operand(w_ptr, records=fx.Int64(hidden * 2))

        def _slot_i32(step, sub):
            """i32 offset of (*step*, this block, *sub*) inside the inbox.

            Plain arithmetic rather than ``crd2idx`` over a layout. The natural
            layout here is ``(steps, grid, super_tile)``, whose trailing mode has
            extent **1** at ST=1; a unit mode gets coalesced away, after which a
            three-coordinate lookup no longer lines up with the modes and
            silently returns a *negative* index. ``step`` is a Python int, so the
            first term folds to a constant at trace time and this is no more work
            than the layout version.

            With two codecs the steps are no longer uniformly sized, so the base
            is a prefix sum rather than a product.
            """
            return (
                fx.Int32(step_base_i32[step])
                + bid * fx.Int32(wire_tile_i32[step])
                + sub * fx.Int32(payload_i32[step])
            )

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

            def _load_raw_atom(buf, tile, atom):
                """One 16 B atom, unconverted. For the bf16 operands of the fused
                epilogue, which are not codec values and never become fp16."""
                src = hbm_copy.partition_S(_atom_row(buf, tile, atom))
                frag = fx.make_fragment_like(src)
                fx.copy(hbm_copy_atom, src, frag)
                return fx.Vector(frag.load())

            def _store_raw_atom(buf, tile, atom, value):
                dst = hbm_copy.partition_D(_atom_row(buf, tile, atom))
                frag = fx.make_fragment_like(dst)
                frag.store(value)
                fx.copy(hbm_copy_atom, frag, dst)

        def _load_chunk_atoms(tile, chunk):
            """This rank's own bf16 data for *chunk*, as fp16 register atoms.

            Only ``rank_atoms`` of the tile, not all 8: a ring touches one chunk
            per op, and across the reduce-scatter lap it visits every chunk
            exactly once -- so total HBM traffic matches the mesh's single
            bulk load, with far fewer values live across the spin-waits.
            """
            return [
                _load_atom(tile, chunk * rank_atoms + j)
                for j in range_constexpr(rank_atoms)
            ]

        if const_expr(fused):
            # See the mesh kernel: a padded fused build must read ``inp`` through
            # the per-row bounded descriptor, not ``_payload_io``'s whole-payload
            # tensor. At h_pad != hidden the wire tile is wider than the HBM row,
            # so ``_payload_io``'s single bound lets a pad lane read the next row
            # instead of zero -- which rides the wire into residual_out. Shadow
            # the plain ``_load_chunk_atoms`` above for the fused build only.
            in_buf = _operand(inp_ptr)

            def _load_chunk_atoms(tile, chunk):
                return [
                    _atom_bf16_to_f16(
                        _load_raw_atom(in_buf, tile, chunk * rank_atoms + j)
                    )
                    for j in range_constexpr(rank_atoms)
                ]

        def _store_chunk_atom(tile, chunk, j, value):
            _store_atom(tile, chunk * rank_atoms + j, value)

        def _lds_write_packet(codec, j, words, scale_word, is_leader):
            """Stage one packet of *codec* into the row this hop will send."""
            lds_write_packet(
                codec,
                pack_views[codec.name],
                fx.Int32(j),
                words,
                scale_word,
                is_leader,
                tid,
                scale_slot,
            )

        def _recv_raw(codec, step, sub, j):
            """This rank's inbox slot for *step*, exactly as the predecessor wrote it.

            Returned undecoded on purpose -- the 2-bit plane stays compacted.
            The all-gather lap restages these same values into LDS and forwards
            them untouched, which is what keeps that lap free of any additional
            quantization, and that only works if nothing here reinterprets them.
            """
            base = _slot_i32(step, sub) + fx.Int32(j * codec.rank_tile_i32)

            def _get(off):
                return load_i32_drained(inbox, base + off, _RECV_POLICY)

            return _codec_load(codec, _get, tid, scale_slot)

        def _scale_of(codec, word):
            return _scale_from_word(codec, word, pair_in_slot)

        def _fanout_to_next(step, sub):
            """Push the staged rank-tiles from LDS into the successor's inbox,
            as one contiguous run.

            INT6 is 26 sectors to INT4's 18, so a TP8 hop drives 26 of the 64
            quads rather than 18: the wider codec uses the fanout better.
            """
            fanout_contiguous(
                codec=step_codec[step],
                n_atoms=rank_atoms,
                quad_id=quad_id,
                lane_in_quad=lane_in_quad,
                quads_per_block=quads_per_block,
                smem_ptr=smem_ptr,
                slot_i32=_slot_i32(step, sub),
                dest_base=next_base,
                payload_policy=payload_policy,
            )

        def _publish(step, color):
            """Drain the payload, make it visible, then colour the slot tail.

            The mesh's publish with one destination: one quad, where the mesh
            needs one per peer.
            """
            publish_flags(
                tid=tid,
                n_dest=1,
                dest_base=lambda _j: next_base,
                flag_i32=_slot_i32(step, fx.Int32(0)) + fx.Int32(release_i32_off[step]),
                color=color,
                flag_policy=flag_policy,
                release_scope=release_scope,
            )

        def _wait(step, color):
            """Spin until the predecessor has coloured *step*'s slot in our inbox.

            One source, so one thread spins where the mesh needs one per peer.

            One colour covers all ``2(N-1)`` slots of a super-tile group. That is
            safe without extra sequencing because the ring's own dependency
            chain bounds how far a rank can run ahead: for the predecessor to be
            writing slot ``k-1`` of the *next* group while we are reading slot
            ``k-2`` of this one, it would have had to complete this group, which
            transitively requires us to have completed op ``N`` -- i.e. to be
            past the read we are blocked on.

            ``_RECV_POLICY`` is ``sc0 sc1``, so the payload loads bypass both
            caches on every inbox; an uncached inbox therefore gets the policy's
            workgroup-scope acquire (no invalidate), every other inbox a
            system-scope one. That one must be preceded by a writeback: the ring
            stores one chunk per all-gather op and then waits again, so an
            invalidate here sits directly on top of dirty output lines, and
            discarding them silently loses whole chunks.
            """
            wait_flags(
                tid=tid,
                n_src=1,
                skip_rank=None,
                flag_addr=lambda _src: self_base
                + _i32_to_bytes(
                    _slot_i32(step, fx.Int32(0)) + fx.Int32(release_i32_off[step])
                ),
                color=color,
                acquire_scope=acquire_scope,
                writeback_scope=(
                    _SYSTEM_SYNC_SCOPE if acquire_scope == _SYSTEM_SYNC_SCOPE else None
                ),
            )

        def _atom_f16_to_f32(atom):
            """Packed fp16 -> 8 f32. A widening move; exact, no rounding."""
            return fx.Vector(atom).bitcast(fx.Float16).to(fx.Float32)

        def _complete(tile, chunk, j, value, recv):
            """Dispose of an atom whose all-reduce is finished.

            Plain: straight to ``out`` in bf16, as before. Fused: held for the
            epilogue, which needs the whole row -- and has it, because the block
            is sized so a chunk is ``rows_per_chunk`` complete rows. ``recv[0]``
            is the prefetched residual, so the atoms follow it.
            """
            if const_expr(fused):
                recv.append(value)
            else:
                _store_chunk_atom(tile, chunk, j, value)

        def _load_residual(tile, chunk):
            """This chunk's residual atoms, issued before the payload arrives.

            Plain HBM with no dependence on any peer, so the load retires while
            the receive and the dequantize are still running rather than
            stalling the epilogue on a cold read after them. Same reasoning as
            the one-shot's residual prefetch.
            """
            base = chunk * rank_atoms
            return [
                _load_raw_atom(res_in_buf, tile, base + j)
                for j in range_constexpr(rank_atoms)
            ]

        def _epilogue(tile, chunk, chunk_atoms, res_atoms, w_atoms):
            """Residual add + RMSNorm over the rows this chunk just completed.

            Runs *after* the chunk has been forwarded, so no norm arithmetic
            sits between a receive and the send that unblocks the successor.

            ``chunk`` is a Python int (``_chunk_of`` folds at trace time), so
            the atom indices below are compile-time and the loads are the same
            static offsets the plain path uses.
            """
            base = chunk * rank_atoms
            rows = []
            for r in range_constexpr(rows_per_chunk):
                idx = [r * atoms_per_row + a for a in range_constexpr(atoms_per_row)]
                xs = residual_add(
                    [_atom_f16_to_f32(chunk_atoms[j]) for j in idx],
                    [res_atoms[j] for j in idx],
                )
                rows.append(xs)
                # residual_out is final already, so it goes out ahead of the
                # reduction rather than behind its barrier.
                packed = pack_bf16(xs)
                for a in range_constexpr(atoms_per_row):
                    _store_raw_atom(res_out_buf, tile, base + idx[a], packed[a])

            # One reduction for every row of the chunk: one barrier, not
            # ``rows_per_chunk`` of them.
            rstds = rms_rstd(rows, eps, hidden, tid=tid, block=block, lds=sq_lds)
            for r in range_constexpr(rows_per_chunk):
                outs = scale_by_weight(rows[r], rstds[r], w_atoms)
                for a in range_constexpr(atoms_per_row):
                    _store_raw_atom(
                        out_buf, tile, base + r * atoms_per_row + a, outs[a]
                    )
            if const_expr(n_partials > 0):
                # The next chunk reuses these slots. The last op has no fanout,
                # hence no barrier of its own between our reads above and those
                # writes -- this is it.
                gpu.barrier()

        def _op_substep(k, tile, sub, recv):
            """One op of the ring, for one sub-tile. Stages LDS; does not send.

            Written as one ``if/elif/else`` over the compile-time op number, with
            **no early returns**. That is not a style choice: inside a
            ``@flyc.kernel`` body the AST transform rewrites ``return`` away, so
            ``if cond: ...; return`` falls through and emits the *following*
            branch as well -- every op would run every body. ``if``/``elif``/
            ``else`` on a Python-level condition is evaluated at trace time and
            selects exactly one branch, so all compile-time branching here uses
            that form.

            For the same reason the completed atoms leave through *recv*, a
            caller-owned list this appends to, rather than through a return
            value. A fused build hands them to the epilogue once the fanout has
            gone out; a plain build passes a list it ignores, and the atoms go
            straight to ``out`` here as before.

            The codecs are compile-time too: an op reads the codec of the step
            it receives from (``k-2``) and writes the codec of the step it sends
            into (``k-1``). Those coincide everywhere except op ``N``, which is
            the seam between the two laps -- it reads a reduce-scatter slot and
            writes an all-gather one.
            """
            chunk = _chunk_of(k)
            is_leader = (tid % fx.Int32(GROUP)) == fx.Int32(0)
            if const_expr(fused and k >= world_size):
                # Ahead of the receive, so the HBM latency hides behind it.
                recv.append(_load_residual(tile, chunk))
            c_in = step_codec[k - 2] if k >= 2 else None
            c_out = step_codec[k - 1] if k <= steps else None

            if const_expr(k == 1):
                # Pipeline fill: nothing to receive, push our own contribution.
                atoms = _load_chunk_atoms(tile, chunk)
                for j in range_constexpr(rank_atoms):
                    words, word, leader = _codec_quant(c_out, atoms[j], lane, tid)
                    _lds_write_packet(c_out, j, words, word, leader)
            elif const_expr(k < world_size):
                # Reduce-scatter: add our contribution to the running partial.
                atoms = _load_chunk_atoms(tile, chunk)
                for j in range_constexpr(rank_atoms):
                    words_in, word_in = _recv_raw(c_in, k - 2, sub, j)
                    acc = _codec_dequant(
                        c_in, words_in, _scale_of(c_in, word_in), tid, atoms[j]
                    )
                    words, word, leader = _codec_quant(c_out, acc, lane, tid)
                    _lds_write_packet(c_out, j, words, word, leader)
            elif const_expr(k == world_size):
                # Last reduce, and the seam: c_in is the reduce-scatter codec,
                # c_out the all-gather one. Every rank has now contributed, so
                # this is the final sum for our own chunk: store it from the
                # unquantized accumulator, since it never makes another wire
                # hop. The same value is quantized once for the all-gather's
                # first send, and that single quantization is the whole error
                # the gather lap contributes.
                atoms = _load_chunk_atoms(tile, chunk)
                for j in range_constexpr(rank_atoms):
                    words_in, word_in = _recv_raw(c_in, k - 2, sub, j)
                    acc = _codec_dequant(
                        c_in, words_in, _scale_of(c_in, word_in), tid, atoms[j]
                    )
                    _complete(tile, chunk, j, acc, recv)
                    words, word, leader = _codec_quant(c_out, acc, lane, tid)
                    _lds_write_packet(c_out, j, words, word, leader)
            elif const_expr(k < n_ops):
                # All-gather: the chunk is already final, so decode it for our
                # own output and forward the bytes we received unmodified. Not
                # dequantizing-and-requantizing is what keeps this lap free of
                # additional error -- and it is why c_in is c_out here.
                for j in range_constexpr(rank_atoms):
                    words_in, word_in = _recv_raw(c_in, k - 2, sub, j)
                    _complete(
                        tile,
                        chunk,
                        j,
                        _codec_dequant(c_in, words_in, _scale_of(c_in, word_in), tid),
                        recv,
                    )
                    _lds_write_packet(c_out, j, words_in, word_in, is_leader)
            else:
                # Final op: the last chunk arrives and stops here.
                for j in range_constexpr(rank_atoms):
                    words_in, word_in = _recv_raw(c_in, k - 2, sub, j)
                    _complete(
                        tile,
                        chunk,
                        j,
                        _codec_dequant(c_in, words_in, _scale_of(c_in, word_in), tid),
                        recv,
                    )

        def _ring_group(i, n_this, color, w_atoms):
            """One super-tile group: ``2N-1`` ops, each a wait / work / publish.

            The wait is hoisted above the sub-tile loop and the publish sits
            below it, so a group of ST tiles costs ``2(N-1)`` handshakes in
            total rather than per tile. Nothing is carried across the publish --
            each op re-reads its inbox slot -- which is what keeps register
            pressure flat as ST grows.

            Op 1 never waits and op ``2N-1`` never sends; both are expressed as
            ``if/else`` on the Python op number, never as a bare ``if`` guarding
            a compile-time-dead block. See ``_op_substep`` for why.

            """
            for _ki in range_constexpr(n_ops):
                k = _ki + 1
                if const_expr(k >= 2):
                    _wait(k - 2, color)
                else:
                    pass  # op 1 is a pure send: there is nothing to wait for
                for s in range(fx.Int32(0), n_this, fx.Int32(1)):
                    tile = bid + (i + s) * n_blocks
                    # A caller-owned list rather than a return value; see
                    # ``_op_substep``. Empty for every op that completes nothing,
                    # and for every op at all in a plain build.
                    recv = []
                    _op_substep(k, tile, s, recv)
                    if const_expr(k <= steps):
                        gpu.barrier()
                        _fanout_to_next(k - 1, s)
                        if (s + fx.Int32(1)) < n_this:
                            # This wave's LDS reads must land before the next
                            # sub-tile overwrites the staging rows. lgkmcnt only:
                            # the payload stores stay in flight until _publish.
                            rocdl.s_waitcnt(lgkmcnt=0)
                            gpu.barrier()
                    else:
                        pass  # the last op receives only
                    if const_expr(fused and k >= world_size):
                        # After the fanout: the successor is already unblocked,
                        # so the epilogue runs off the ring's critical path.
                        _epilogue(tile, _chunk_of(k), recv[1:], recv[0], w_atoms)
                    else:
                        pass
                if const_expr(k <= steps):
                    _publish(k - 1, color)
                else:
                    pass

        # One row shared by every token, so the gain is read once here rather
        # than once per chunk.
        w_atoms = None
        if const_expr(fused):
            w_atoms = [
                _load_raw_atom(w_buf, fx.Int32(0), a)
                for a in range_constexpr(atoms_per_row)
            ]

        # Stride by the launched grid, not the compile-time cap: the host
        # launches fewer blocks than `grid` when it wants each block to own a
        # whole super-tile, and striding by the cap would silently leave every
        # tile above n_blocks unprocessed.
        n_block_tiles = (num_tiles - bid + n_blocks - fx.Int32(1)) // n_blocks
        color = _load_color()
        st_i = fx.Int32(super_tile)
        for i in range(fx.Int32(0), n_block_tiles, st_i):
            remain = n_block_tiles - i
            n_this = (remain < st_i).select(remain, st_i)
            _ring_group(i, n_this, color, w_atoms)
            color = next_color(color, parity_safe=False)
        if tid == 0:
            _store_color(color)
        gpu.barrier()

    flat_wg = f"{block},{block}"

    # Two launchers over one kernel. The plain one keeps the eight-argument
    # signature the host's ``_launch_args`` builds and passes zeros for the
    # fused operands, which a plain build never reads; the fused one takes them
    # for real. Splitting here rather than at the kernel keeps the body single-
    # sourced -- it has to stay lexically inside one ``@flyc.kernel``, because
    # only that function's AST is rewritten.
    @flyc.jit
    def launch_quick_allreduce_ring(
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
        quick_allreduce_ring(
            rank_arg,
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
    def launch_quick_allreduce_ring_fused(
        rank_arg: fx.Int32,
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
        quick_allreduce_ring(
            rank_arg,
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

    # rank is baked into the schedule, and the inbox memory type into the store
    # policy, so both have to reach the symbol name -- variants that differ only
    # in a compile-time constant must not collide in the JIT cache.
    tag = (
        f"ws{world_size}_r{rank}_st{super_tile}_g{grid}_{inbox_memory}"
        f"_{rs_codec}_{ag_codec}"
    )
    tag += f"_b{block}"
    if fused:
        tag += f"_rms_h{hidden}"
        if padded:
            tag += f"_p{h_pad}"
    launcher = (
        launch_quick_allreduce_ring_fused if fused else launch_quick_allreduce_ring
    )
    launcher.func.__name__ = f"launch_quick_allreduce_ring_{tag}"
    try:
        quick_allreduce_ring.func.__name__ = f"quick_allreduce_ring_{tag}"
    except AttributeError:
        pass
    return {
        "launch": launcher,
        "flags_bytes": 0,  # the handshake rides in each slot's 64 B tail
        "data_bytes": inbox_bytes,
        "lds_bytes": lds_bytes,
        "tile_bytes": tile_bytes,
        # HBM bytes one tile spans -- what a tile *count* has to come from.
        # Equal to tile_bytes unless the build is padded.
        "hbm_tile_bytes": hbm_tile_bytes,
        "tile_fp16": tile_bytes // 2,
        "rank_tile_bytes": rs.rank_tile_bytes,
        "wire_tile_bytes": wire_tile_i32[0] * 4,
        "ag_rank_tile_bytes": ag.rank_tile_bytes,
        "ag_wire_tile_bytes": wire_tile_i32[-1] * 4,
        "super_tile": super_tile,
        "world_size": world_size,
        "rank": rank,
        "inbox_memory": inbox_memory,
        "rs_codec": rs_codec,
        "ag_codec": ag_codec,
        "payload_policy": payload_policy,
        "flag_policy": flag_policy,
        "release_scope": release_scope,
        "rank_atoms": rank_atoms,
        "steps": steps,
        "grid": grid,
        "block": block,
        "fusion": fusion,
        "hidden": hidden,
        "h_pad": h_pad,
        "padded": padded,
        "atoms_per_row": atoms_per_row,
        "rows_per_tile": rows_per_tile,
        "rows_per_chunk": rows_per_chunk,
    }
