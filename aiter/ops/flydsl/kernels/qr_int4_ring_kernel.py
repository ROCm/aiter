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
INT4 unless asked otherwise by env variable AITER_ALL_REDUCE_CODEC.

A third wire format, ``"fp16"``, is a lossless passthrough. Mainly for testing.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import Int32, Int64, Stream, T

from . import buffer_ops
from .qr_int_codec import (
    CODECS,
    GROUP,
    _atom_bf16_to_f16,
    _atom_f16_to_bf16,
    _clamp_fp16_overflow,
    _codec_dequant,
    _codec_load,
    _codec_quant,
    _scale_from_word,
    scale_slot_of,
    thread_lane,
)
from .qr_int_shared import (
    _CM_SC0,
    _CM_SC1,
    _INBOX_POLICY,
    ATOMS,
    BLOCK,
    DEFAULT_GRID_CAP,  # noqa: F401  -- re-exported for host symmetry
    QUAD_LANES,
    QUADS_PER_WAVE,
    SUPPORTED_WORLDS,
    TILE_BYTES,
    TILE_FP16,
    TILE_I32,
    _acquire_inbox,
    _i32_to_bytes,
    _store_v4i32_peer,
    _to_sgpr_i64,
    make_pack_storage,
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

# Payload-size ladder: ``(min_bytes, super_tile, grid_cap)``, ascending.
RING_ST_LADDER = (
    (0, 8, 128),
    (18 << 20, 16, 128),
    (64 << 20, 32, 128),
)

# Wire formats accepted per lap.
#
# Both laps take the same set, but they are separate arguments because they are
# separate decisions: the reduce-scatter lap requantizes ``N-1`` times and is
# what INT6 is for, while the all-gather lap forwards the bytes it received
# without touching them (see ``_op_substep``) and so contributes exactly one
# quantization -- the one at op ``N``, where this rank's chunk is final.
RS_CODECS = ("int4", "int6", "fp16")
AG_CODECS = ("int4", "int6", "fp16")

# 64 quads of 4 lanes; one quad writes one 64 B fabric sector.
QUADS_PER_BLOCK = BLOCK // QUAD_LANES

# Cache policy for reading a rank-tile out of our own inbox.
#
# The mesh reads its inbox `nt`, which is only a non-temporal hint -- it
# does not bypass. For the ring algorithm, a hint is not enough:
# `sc0 sc1` makes the load actually go to memory.
_RECV_POLICY = _CM_SC0 | _CM_SC1


def _load_i32_at(rsrc, elem_off, cache_modifier):
    """One i32 from *rsrc* at an element offset, drained before it is read.

    ``qr_int_shared._load_i32_uncached`` does the same but hardcodes offset 0,
    which would force a fresh per-call descriptor here; the ring always has the
    inbox descriptor in hand and only the offset varies.
    """
    val = buffer_ops.buffer_load(
        rsrc, elem_off, vec_width=1, dtype=T.i32, cache_modifier=cache_modifier
    )
    rocdl.s_waitcnt(vmcnt=0)
    return fx.Int32(val)


def ring_steps(world_size: int) -> int:
    """Wire slots, which is also hops on the critical path: ``2(N-1)``.

    ``N-1`` to reduce-scatter and ``N-1`` to all-gather, with the last reduce
    and the first gather-send fused into one op -- which is what keeps this at
    ``2(N-1)`` rather than ``2N``.
    """
    return 2 * (world_size - 1)


def make_qr_int4_ring_kernel(
    *,
    world_size: int,
    rank: int,
    super_tile: int = 1,
    grid: int,
    inbox_memory: str = "finegrained",
    rs_codec: str = "int4",
    ag_codec: str = "int4",
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
    release_writeback = policy["writeback"]
    # policy["fanout"] is not consulted: it picks which axis of a (peer, sector)
    # fanout runs fastest across quads, and a ring has no peer axis. Sectors run
    # fastest by construction, which is the "peer" (PCIe-favourable) answer.

    rank_atoms = ATOMS // world_size
    steps = ring_steps(world_size)
    n_ops = 2 * world_size - 1  # ops are 1-based; op k reads slot k-2, writes k-1
    nxt = (rank + 1) % world_size

    rs = CODECS[rs_codec]
    ag = CODECS[ag_codec]
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
    total_sectors = [rank_atoms * c.n_sectors for c in step_codec]
    fanout_rounds = [-(-t // QUADS_PER_BLOCK) for t in total_sectors]

    # One staging buffer, sized for whichever codec needs more. Only
    # ``rank_atoms`` rows: a ring stages one destination's packet, not every
    # destination's, so this is 1664 B at TP8 against the mesh's 9216 B -- an
    # INT6 ring costs less LDS than an INT4 mesh.
    pack_row_i32 = max(rs.rank_tile_i32, ag.rank_tile_i32)
    pack_i32 = rank_atoms * pack_row_i32
    lds_bytes = pack_i32 * 4
    PackStorage = make_pack_storage(pack_i32)

    def _chunk_of(k: int) -> int:
        """Which chunk op *k* carries, following the standard ring schedule.

        Ops ``1..N`` are the reduce-scatter lap, walking chunks ``r-1`` down to
        ``r``; ops ``N+1..2N-1`` are the all-gather lap, walking ``r-1`` down to
        ``r+1``. Op ``N`` completes this rank's own chunk and is also the
        all-gather's first send.
        """
        j = k if k <= world_size else k - world_size
        return (rank - j) % world_size

    @flyc.kernel(known_block_size=[BLOCK, 1, 1])
    def qr_int4_ring(
        rank_unused: Int32,
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

        wave, lane = thread_lane(tid)
        quad_layout = fx.make_layout((QUADS_PER_WAVE, QUAD_LANES), (QUAD_LANES, 1))
        quad, lane_in_quad = fx.idx2crd(lane, quad_layout).unpack()
        quad_id = wave * fx.Int32(QUADS_PER_WAVE) + quad

        hbm_layout = fx.make_layout(
            (num_tiles, ATOMS, BLOCK * 4),
            (TILE_I32, BLOCK * 4, 1),
        )
        hbm_row_layout = fx.make_layout((1, BLOCK * 4), (BLOCK * 4, 1))
        hbm_copy_atom = fx.make_copy_atom(rocdl.BufferCopy128b(), fx.Int32)
        hbm_copy = fx.make_tiled_copy_tv(
            hbm_copy_atom,
            fx.make_layout((1, BLOCK), (1, 1)),
            fx.make_layout((1, 4), (1, 1)),
        ).get_slice(tid)
        scale_slot, pair_in_slot = scale_slot_of(tid)
        color_layout = fx.make_layout((grid,), (1,))

        # One allocation, one view per codec.
        lds = fx.SharedAllocator().allocate(PackStorage).peek()
        smem_ptr = lds.pack.ptr
        pack_views = {
            c.name: lds.pack.view(
                fx.make_layout((rank_atoms, c.rank_tile_i32), (c.rank_tile_i32, 1))
            )
            for c in ({rs.name: rs, ag.name: ag}).values()
        }

        peer_rsrc = buffer_ops.create_buffer_resource_from_addr(peer_ptrs)
        peers = [
            buffer_ops.buffer_load(peer_rsrc, i, vec_width=1, dtype=T.i64)
            for i in range(world_size)
        ]
        # A ring only ever names two of the peers, and ``rank`` is compile-time,
        # so both are plain Python indices into the loaded pointers. The
        # mesh packs these into an fx.Vector because its fanout selects a
        # peer with a *runtime* lane-dependent index; doing that here would put
        # a dynamic extract in front of a constant and get the wrong element.
        self_base = fx.Int64(peers[rank])
        next_base = fx.Int64(peers[nxt])
        # Bounded on purpose. Every ring access is in range by construction,
        # so an out-of-range one is a bug -- and with num_records set the
        # hardware returns zero instead of faulting, which turns a
        # process-killing page fault into a wrong SQNR you can bisect.
        self_rsrc = buffer_ops.create_buffer_resource_from_addr(
            _to_sgpr_i64(self_base), num_records_bytes=inbox_bytes
        )

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

        def _load_chunk_atoms(tile, chunk):
            """This rank's own bf16 data for *chunk*, as fp16 register atoms.

            Only ``rank_atoms`` of the tile, not all 8: a ring touches one chunk
            per op, and across the reduce-scatter lap it visits every chunk
            exactly once -- so total HBM traffic matches the mesh's single
            bulk load, with far fewer values live across the spin-waits.
            """
            out = []
            for j in range_constexpr(rank_atoms):
                src = hbm_copy.partition_S(
                    _hbm_atom_row(in_buf, tile, chunk * rank_atoms + j)
                )
                frag = fx.make_fragment_like(src)
                fx.copy(hbm_copy_atom, src, frag)
                out.append(_atom_bf16_to_f16(fx.Vector(frag.load())))
            return out

        def _store_chunk_atom(tile, chunk, j, value):
            dst = hbm_copy.partition_D(
                _hbm_atom_row(out_buf, tile, chunk * rank_atoms + j)
            )
            frag = fx.make_fragment_like(dst)
            frag.store(_atom_f16_to_bf16(value))
            fx.copy(hbm_copy_atom, frag, dst)

        def _lds_write_packet(codec, j, words, scale_word, is_leader):
            """Stage one packet of *codec* into the row this hop will send."""
            pack = pack_views[codec.name]
            row = fx.Int32(j)
            for (off, pred), word in zip(codec.plane_slots(tid), words):
                if pred:
                    fx.memref_store(word, pack, (row, off))
            if const_expr(codec.has_scale):  # noqa: SIM102
                if is_leader:
                    fx.memref_store(
                        scale_word,
                        pack,
                        (row, fx.Int32(codec.scale_i32_off) + scale_slot),
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
                return _load_i32_at(self_rsrc, base + off, _RECV_POLICY)

            return _codec_load(codec, _get, tid, scale_slot)

        def _scale_of(codec, word):
            return _scale_from_word(codec, word, pair_in_slot)

        def _fanout_to_next(step, sub):
            """Push the staged rank-tiles from LDS into the successor's inbox.

            One destination, sectors in address order, so the whole
            ``rank_atoms * rank_tile B`` lands as a single contiguous run. Quads
            past the sector count sit idle rather than branching -- ``safe``
            keeps their address arithmetic in range, mirroring the mesh.

            INT6 is 26 sectors to INT4's 18, so a TP8 hop drives 26 of the 64
            quads rather than 18: the wider codec uses the fanout better.
            """
            codec = step_codec[step]
            n_sectors = codec.n_sectors
            n_total = total_sectors[step]
            for rnd in range_constexpr(fanout_rounds[step]):
                s = quad_id + fx.Int32(rnd * QUADS_PER_BLOCK)
                in_range = s < fx.Int32(n_total)
                safe = in_range.select(s, fx.Int32(0))
                # Flat sector id -> (rank-atom, sector), then -> i32 offset. Both
                # by arithmetic, for the same reason as _slot_i32: at TP8
                # rank_atoms is 1, and a unit mode in a layout does not survive
                # coalescing intact. The same offset addresses LDS and the
                # wire, because this codec's LDS view and its wire rank-tiles
                # share a row stride.
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
                    byte_off = _i32_to_bytes(_slot_i32(step, sub) + flat)
                    _store_v4i32_peer(next_base + byte_off, v4, payload_policy)

        def _publish(step, color):
            """Drain the payload, make it visible, then colour the slot tail.

            Identical in shape to the mesh's publish and for the same
            reasons -- ``vmcnt`` is per-wave so the workgroup has to join before
            the flag goes out, and on a cacheable inbox a retired store is not
            yet a visible one, so the release needs an explicit L2 writeback.
            What differs is the width: one quad, one destination, where the mesh
            needs one quad per peer.
            """
            rocdl.s_waitcnt(vmcnt=0)
            gpu.barrier()
            if const_expr(release_writeback is not None):
                llvm.InlineAsmOp(None, [], release_writeback, "", has_side_effects=True)
                rocdl.s_waitcnt(vmcnt=0)
            if quad_id == fx.Int32(0):
                vec_idx = fx.Int32(release_i32_off[step]) + lane_in_quad * fx.Int32(4)
                v4 = fx.Vector.from_elements([color, color, color, color], fx.Int32)
                byte_off = _i32_to_bytes(_slot_i32(step, fx.Int32(0)) + vec_idx)
                _store_v4i32_peer(next_base + byte_off, v4, flag_policy)

        def _wait(step, color):
            """Spin until the predecessor has coloured *step*'s slot in our inbox.

            One source, so one thread spins where the mesh needs one per
            peer. ``buffer_inv sc1`` between attempts is not optional: without
            it the load can be answered forever from a stale line, which is a
            hang rather than a slowdown.

            One colour covers all ``2(N-1)`` slots of a super-tile group. That is
            safe without extra sequencing because the ring's own dependency
            chain bounds how far a rank can run ahead: for the predecessor to be
            writing slot ``k-1`` of the *next* group while we are reading slot
            ``k-2`` of this one, it would have had to complete this group, which
            transitively requires us to have completed op ``N`` -- i.e. to be
            past the read we are blocked on.

            Spins on the one shared ``self_rsrc`` descriptor at an element
            offset, rather than building a fresh descriptor from
            ``self_base + elem*4``. The mesh can afford the latter because
            its ``elem`` depends on ``tid`` (one spinner per source rank) and the
            resulting waterfall is genuine. Here there is exactly one source, so
            a per-call descriptor is a *uniform* value that LLVM cannot prove
            uniform: it sources all four descriptor dwords from VGPRs and
            serializes the wave around them, num_records included.
            """
            if tid == fx.Int32(0):
                elem = _slot_i32(step, fx.Int32(0)) + fx.Int32(release_i32_off[step])
                # `sc0 sc1`, so each retry is fetched past L1 and L2 and no
                # fence is needed in the loop; the acquire below covers the
                # payload reads, once, after the join.
                current = _load_i32_at(self_rsrc, elem, _RECV_POLICY)
                while current != color:
                    current = _load_i32_at(self_rsrc, elem, _RECV_POLICY)
            gpu.barrier()
            # Unconditional, *after* the join, and not just inside the spin.
            # Only `tid == 0` spins, so an acquire inside the loop would cover
            # one lane of one wave and leave the rest of the workgroup reading
            # the payload with nothing invalidated on its behalf -- and would be
            # skipped entirely in the common case where the flag is already set
            # on the first read.
            #
            # This is now defensive rather than load-bearing: `_RECV_POLICY` is
            # `sc0 sc1`, so the payload loads below already bypass both caches
            # and cannot be served a stale line on their own. It is kept because
            # it costs one fence per step and the failure it guards against is a
            # silent wrong result. (An earlier version of this comment said the
            # payload was read `nt`; that describes the mesh kernel, not
            # this one -- see the note on _RECV_POLICY above.)
            #
            # Write back *before* invalidating. The ring stores
            # one chunk per all-gather op and then waits again -- so an acquire
            # here sits directly on top of dirty output lines, and discarding
            # them silently loses whole chunks.
            rocdl.s_waitcnt(vmcnt=0)
            llvm.InlineAsmOp(None, [], "buffer_wbl2 sc1", "", has_side_effects=True)
            rocdl.s_waitcnt(vmcnt=0)
            _acquire_inbox()

        def _op_substep(k, tile, sub):
            """One op of the ring, for one sub-tile. Stages LDS; does not send.

            Written as one ``if/elif/else`` over the compile-time op number, with
            **no early returns**. That is not a style choice: inside a
            ``@flyc.kernel`` body the AST transform rewrites ``return`` away, so
            ``if cond: ...; return`` falls through and emits the *following*
            branch as well -- every op would run every body. ``if``/``elif``/
            ``else`` on a Python-level condition is evaluated at trace time and
            selects exactly one branch, so all compile-time branching here uses
            that form.

            The codecs are compile-time too: an op reads the codec of the step
            it receives from (``k-2``) and writes the codec of the step it sends
            into (``k-1``). Those coincide everywhere except op ``N``, which is
            the seam between the two laps -- it reads a reduce-scatter slot and
            writes an all-gather one.
            """
            chunk = _chunk_of(k)
            is_leader = (tid % fx.Int32(GROUP)) == fx.Int32(0)
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
                    _store_chunk_atom(tile, chunk, j, acc)
                    words, word, leader = _codec_quant(c_out, acc, lane, tid)
                    _lds_write_packet(c_out, j, words, word, leader)
            elif const_expr(k < n_ops):
                # All-gather: the chunk is already final, so decode it for our
                # own output and forward the bytes we received unmodified. Not
                # dequantizing-and-requantizing is what keeps this lap free of
                # additional error -- and it is why c_in is c_out here.
                for j in range_constexpr(rank_atoms):
                    words_in, word_in = _recv_raw(c_in, k - 2, sub, j)
                    _store_chunk_atom(
                        tile,
                        chunk,
                        j,
                        _codec_dequant(
                            c_in, words_in, _scale_of(c_in, word_in), tid
                        ),
                    )
                    _lds_write_packet(c_out, j, words_in, word_in, is_leader)
            else:
                # Final op: the last chunk arrives and stops here.
                for j in range_constexpr(rank_atoms):
                    words_in, word_in = _recv_raw(c_in, k - 2, sub, j)
                    _store_chunk_atom(
                        tile,
                        chunk,
                        j,
                        _codec_dequant(
                            c_in, words_in, _scale_of(c_in, word_in), tid
                        ),
                    )

        def _ring_group(i, n_this, color):
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
                    _op_substep(k, tile, s)
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
                if const_expr(k <= steps):
                    _publish(k - 1, color)
                else:
                    pass

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
            _ring_group(i, n_this, color)
            color = color + fx.Int32(1)
            if color == fx.Int32(0):  # 0 is the unset sentinel
                color = fx.Int32(1)
        if tid == 0:
            _store_color(color)
        gpu.barrier()

    flat_wg = f"{BLOCK},{BLOCK}"

    @flyc.jit
    def launch_qr_int4_ring(
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
        qr_int4_ring(
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
            block=(BLOCK, 1, 1),
            stream=stream,
        )

    # rank is baked into the schedule, and the inbox memory type into the store
    # policy, so both have to reach the symbol name -- variants that differ only
    # in a compile-time constant must not collide in the JIT cache.
    tag = (
        f"ws{world_size}_r{rank}_st{super_tile}_{inbox_memory}"
        f"_{rs_codec}_{ag_codec}"
    )
    launch_qr_int4_ring.func.__name__ = f"launch_qr_int4_ring_{tag}"
    try:
        qr_int4_ring.func.__name__ = f"qr_int4_ring_{tag}"
    except AttributeError:
        pass
    return {
        "launch": launch_qr_int4_ring,
        "flags_bytes": 0,  # the handshake rides in each slot's 64 B tail
        "data_bytes": inbox_bytes,
        "lds_bytes": lds_bytes,
        "tile_bytes": TILE_BYTES,
        "tile_fp16": TILE_FP16,
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
        "release_writeback": release_writeback,
        "rank_atoms": rank_atoms,
        "steps": steps,
        "grid": grid,
        "block": BLOCK,
    }
