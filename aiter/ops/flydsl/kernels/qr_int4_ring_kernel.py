# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""gfx942/gfx950 TP∈{2,4,8} INT4 **ring** all-reduce.

Same codec, tiling, IPC inbox and colour handshake as the two-shot kernel in
``qr_int4_kernel`` -- only the schedule differs. Instead of each rank pushing to
all ``N-1`` peers twice, the ranks form a cycle and every step is a single
contiguous run into exactly *one* peer's inbox:

* ``2(N-1)`` hops instead of 2, so ``2(N-1)`` handshakes instead of 2.
* The same ``2(N-1)/N`` of the payload on the wire -- a ring is bandwidth
  optimal (Patarasuk & Yuan, JPDC 69(2), 2009), so nothing is paid for the extra
  hops in volume.
* **One destination per store.** That is the point. On a PCIe host a GPU has a
  single shared x16 uplink, and the two-shot fanout hands it 64 B per
  destination round-robin; coarsening to >=256 B runs against one destination
  measured 54.03 GB/s against 36.25 (MI350P, cached memory, 3 peers). A ring
  step writes ``rank_atoms * 1152 B`` to one peer, in address order, with no
  destination switch at all.

Everything the codec does is imported from ``qr_int4_kernel`` rather than
restated: there must be exactly one INT4 + group-16 E4M3 implementation in the
tree, or the two kernels can drift into disagreeing about the wire format.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm
from flydsl.expr import gpu, range_constexpr, rocdl
from flydsl.expr.typing import Int32, Int64, Stream, T

from . import buffer_ops

# The codec, the peer-store/load primitives and the geometry are shared with the
# two-shot kernel verbatim. They are private there by convention, not by intent:
# these two kernels are the only consumers and they must agree byte for byte.
from .qr_int4_kernel import (
    _CM_SC0,
    _CM_SC1,
    _INBOX_POLICY,
    ATOMS,
    BLOCK,
    DEFAULT_GRID_CAP,  # noqa: F401  -- re-exported for host symmetry
    GROUP,
    LDS_BYTES,
    N_SECTORS,
    PAIR,
    QUAD_LANES,
    QUADS_PER_WAVE,
    RANK_TILE_BYTES,
    RANK_TILE_I32,
    SCALE_I32_OFF,
    SUPPORTED_WORLDS,
    TILE_BYTES,
    TILE_FP16,
    TILE_I32,
    WAVE,
    WAVES,
    PackStorage,
    _atom_bf16_to_f16,
    _atom_f16_to_bf16,
    _clamp_fp16_overflow,
    _codec_dequant,
    _codec_quant,
    _e4m3_decoding_scale,
    _i32_to_bytes,
    _invalidate_l1,
    _store_v4i32_peer,
    _to_sgpr_i64,
)

# Super-tile values the ring accepts. Deliberately the same as the two-shot's.
#
# The ring publishes 2(N-1) times per super-tile against the two-shot's 2, and
# on a cacheable inbox every publish is a full L2 writeback -- so the obvious
# reflex is to raise ST until they amortize. Resist it: the wire buffer is
# ``2(N-1) * grid * (ST * rank_atoms * 1152 + 64)`` bytes, already ~150 MB at
# TP4/ST=8/grid=1216, and ST=32 would be gigabytes. Shrink ``grid`` instead --
# ``QRInt4._grid_x`` already hands each block a whole super-tile.
RING_SUPER_TILES = (1, 8)

# Wire formats accepted for the reduce-scatter lap.
#
# The all-gather lap is always INT4 and is not listed here: it forwards the
# bytes it received without touching them (see ``_ring_body``), so it has no
# format of its own to choose. Widening the RS lap to INT6 is the planned
# response if measured SQNR at TP8 lands below the two-shot's ~18 dB gate --
# see the plan; not implemented yet, which is why "int6" is absent rather than
# accepted-and-broken.
RS_CODECS = ("int4",)

# 64 quads of 4 lanes; one quad writes one 64 B fabric sector.
QUADS_PER_BLOCK = BLOCK // QUAD_LANES

# Cache policy for reading a rank-tile out of our own inbox.
#
# The two-shot reads its inbox `nt`, which is only a non-temporal *hint* -- it
# does not bypass. It gets away with that because a payload it reads was
# published two hops and a full grid of unrelated traffic ago. The ring reads a
# slot the predecessor wrote microseconds earlier, on the same line, with the
# flag arriving write-through right behind it, so a hint is not enough:
# `sc0 sc1` makes the load actually go to memory.
_RECV_POLICY = _CM_SC0 | _CM_SC1


def _load_i32_at(rsrc, elem_off, cache_modifier):
    """One i32 from *rsrc* at an element offset, drained before it is read.

    ``qr_int4_kernel._load_i32_uncached`` does the same but hardcodes offset 0,
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
):
    """Build the ring kernel for one *rank*.

    ``rank`` is a **compile-time** parameter here, unlike the two-shot kernel
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

    payload_i32 = rank_atoms * RANK_TILE_I32
    release_i32_off = super_tile * payload_i32
    wire_tile_i32 = release_i32_off + 16  # + one 64 B handshake sector
    wire_tile_bytes = wire_tile_i32 * 4
    inbox_bytes = steps * grid * wire_tile_bytes

    # Every slot has exactly one writer -- the ring predecessor -- so the
    # two-shot's per-sender axis collapses away entirely. That is what keeps the
    # atomics-free design sound here (there are no peer atomics over PCIe), and
    # it makes the buffer (N-1)/N of the two-shot's rather than larger.
    total_sectors = rank_atoms * N_SECTORS
    fanout_rounds = -(-total_sectors // QUADS_PER_BLOCK)

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

        thread_layout = fx.make_layout((WAVES, WAVE), (WAVE, 1))
        wave, lane = fx.idx2crd(tid, thread_layout).unpack()
        quad_layout = fx.make_layout((QUADS_PER_WAVE, QUAD_LANES), (QUAD_LANES, 1))
        quad, lane_in_quad = fx.idx2crd(lane, quad_layout).unpack()
        quad_id = wave * fx.Int32(QUADS_PER_WAVE) + quad

        pack_layout = fx.make_layout((ATOMS, RANK_TILE_I32), (RANK_TILE_I32, 1))

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
        scale_own_layout = fx.make_layout(
            (BLOCK // GROUP, GROUP // PAIR, PAIR), (GROUP, PAIR, 1)
        )
        scale_slot, pair_in_slot, _lane_in_pair = fx.idx2crd(
            tid, scale_own_layout
        ).unpack()
        color_layout = fx.make_layout((grid,), (1,))

        # Only rank_atoms of the ATOMS rows are used -- a ring stages one
        # destination's packet, not every destination's. Reusing the two-shot's
        # struct wastes at most 8 KiB of a 160 KiB budget and keeps one
        # definition of the staging layout.
        lds = fx.SharedAllocator().allocate(PackStorage).peek()
        pack = lds.pack.view(pack_layout)
        smem_ptr = lds.pack.ptr

        peer_rsrc = buffer_ops.create_buffer_resource_from_addr(peer_ptrs)
        peers = [
            buffer_ops.buffer_load(peer_rsrc, i, vec_width=1, dtype=T.i64)
            for i in range(world_size)
        ]
        # A ring only ever names two of the peers, and ``rank`` is compile-time,
        # so both are plain Python indices into the loaded pointers. The
        # two-shot packs these into an fx.Vector because its fanout selects a
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
            """
            return (
                fx.Int32(step * grid * wire_tile_i32)
                + bid * fx.Int32(wire_tile_i32)
                + sub * fx.Int32(payload_i32)
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
            exactly once -- so total HBM traffic matches the two-shot's single
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

        def _lds_write_packet(j, packed, scale_word, is_leader):
            fx.memref_store(packed, pack, (fx.Int32(j), tid))
            if is_leader:
                fx.memref_store(
                    scale_word,
                    pack,
                    (fx.Int32(j), fx.Int32(SCALE_I32_OFF) + scale_slot),
                )

        def _recv_raw(step, sub, j):
            """This rank's inbox slot for *step*: the packed nibbles and the raw
            E4M3 word, both exactly as the predecessor wrote them.

            Returned undecoded on purpose. The all-gather lap re-stages these
            same two values into LDS and forwards them untouched, which is what
            keeps that lap free of any additional quantization.
            """
            base = _slot_i32(step, sub) + fx.Int32(j * RANK_TILE_I32)
            packed = _load_i32_at(self_rsrc, base + tid, _RECV_POLICY)
            word = _load_i32_at(
                self_rsrc, base + fx.Int32(SCALE_I32_OFF) + scale_slot, _RECV_POLICY
            )
            return packed, word

        def _scale_of(word):
            e = word.shrui(pair_in_slot * fx.Int32(8)) & fx.Int32(0xFF)
            return _e4m3_decoding_scale(e)

        def _fanout_to_next(step, sub):
            """Push the staged rank-tiles from LDS into the successor's inbox.

            One destination, sectors in address order, so the whole
            ``rank_atoms * 1152 B`` lands as a single contiguous run. Quads past
            the sector count sit idle rather than branching -- ``safe`` keeps
            their address arithmetic in range, mirroring the two-shot.
            """
            for rnd in range_constexpr(fanout_rounds):
                s = quad_id + fx.Int32(rnd * QUADS_PER_BLOCK)
                in_range = s < fx.Int32(total_sectors)
                safe = in_range.select(s, fx.Int32(0))
                # Flat sector id -> (rank-atom, sector), then -> i32 offset. Both
                # by arithmetic, for the same reason as _slot_i32: at TP8
                # rank_atoms is 1, and a unit mode in a layout does not survive
                # coalescing intact. The same offset addresses LDS and the wire,
                # because the LDS pack rows and the wire rank-tiles share the
                # RANK_TILE_I32 stride.
                j = safe // fx.Int32(N_SECTORS)
                sector = safe % fx.Int32(N_SECTORS)
                if s < fx.Int32(total_sectors):
                    flat = (
                        j * fx.Int32(RANK_TILE_I32)
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

            Identical in shape to the two-shot's publish and for the same
            reasons -- ``vmcnt`` is per-wave so the workgroup has to join before
            the flag goes out, and on a cacheable inbox a retired store is not
            yet a visible one, so the release needs an explicit L2 writeback.
            What differs is the width: one quad, one destination, where the
            two-shot needs one quad per peer.
            """
            rocdl.s_waitcnt(vmcnt=0)
            gpu.barrier()
            if release_writeback is not None:
                llvm.InlineAsmOp(None, [], release_writeback, "", has_side_effects=True)
                rocdl.s_waitcnt(vmcnt=0)
            if quad_id == fx.Int32(0):
                vec_idx = fx.Int32(release_i32_off) + lane_in_quad * fx.Int32(4)
                v4 = fx.Vector.from_elements([color, color, color, color], fx.Int32)
                byte_off = _i32_to_bytes(_slot_i32(step, fx.Int32(0)) + vec_idx)
                _store_v4i32_peer(next_base + byte_off, v4, flag_policy)

        def _wait(step, color):
            """Spin until the predecessor has coloured *step*'s slot in our inbox.

            One source, so one thread spins where the two-shot needs one per
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
            ``self_base + elem*4``. The two-shot can afford the latter because
            its ``elem`` depends on ``tid`` (one spinner per source rank) and the
            resulting waterfall is genuine. Here there is exactly one source, so
            a per-call descriptor is a *uniform* value that LLVM cannot prove
            uniform: it sources all four descriptor dwords from VGPRs and
            serializes the wave around them, num_records included.
            """
            if tid == fx.Int32(0):
                elem = _slot_i32(step, fx.Int32(0)) + fx.Int32(release_i32_off)
                current = _load_i32_at(self_rsrc, elem, _CM_SC1)
                while current != color:
                    current = _load_i32_at(self_rsrc, elem, _CM_SC1)
                    _invalidate_l1()
            gpu.barrier()
            # Unconditional, *after* the join, and not just inside the spin.
            # The flag is read `sc1` so it is never stale, but the payload is
            # read `nt` -- a hint, not a bypass -- so it can be answered from a
            # line this CU cached before the predecessor wrote it. Invalidating
            # only on a failed spin leaves the common case uncovered: when the
            # flag is already present on the first read the loop body never
            # runs, and the payload loads below can be served stale.
            #
            # This is why the bug scaled with block count. At high occupancy
            # unrelated traffic evicts the stale lines and the reads happen to
            # be correct; at one block nothing evicts them and whole chunks come
            # back as a predecessor's earlier state, or as zero.
            #
            # Write back *before* invalidating. Unlike the two-shot, which does
            # all of a tile's output stores after its last wait, the ring stores
            # one chunk per all-gather op and then waits again -- so an acquire
            # here sits directly on top of dirty output lines, and discarding
            # them silently loses whole chunks.
            rocdl.s_waitcnt(vmcnt=0)
            llvm.InlineAsmOp(None, [], "buffer_wbl2 sc1", "", has_side_effects=True)
            rocdl.s_waitcnt(vmcnt=0)
            _invalidate_l1()

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
            """
            chunk = _chunk_of(k)
            is_leader = (tid % fx.Int32(GROUP)) == fx.Int32(0)

            if k == 1:
                # Pipeline fill: nothing to receive, push our own contribution.
                atoms = _load_chunk_atoms(tile, chunk)
                for j in range_constexpr(rank_atoms):
                    packed, word, leader = _codec_quant(atoms[j], lane, tid)
                    _lds_write_packet(j, packed, word, leader)
            elif k < world_size:
                # Reduce-scatter: add our contribution to the running partial.
                atoms = _load_chunk_atoms(tile, chunk)
                for j in range_constexpr(rank_atoms):
                    packed_in, word_in = _recv_raw(k - 2, sub, j)
                    acc = _codec_dequant(packed_in, _scale_of(word_in), atoms[j])
                    packed, word, leader = _codec_quant(acc, lane, tid)
                    _lds_write_packet(j, packed, word, leader)
            elif k == world_size:
                # Last reduce. Every rank has now contributed, so this is the
                # final sum for our own chunk: store it from the *unquantized*
                # accumulator, since it never makes another wire hop. The same
                # value is quantized once for the all-gather's first send.
                atoms = _load_chunk_atoms(tile, chunk)
                for j in range_constexpr(rank_atoms):
                    packed_in, word_in = _recv_raw(k - 2, sub, j)
                    acc = _codec_dequant(packed_in, _scale_of(word_in), atoms[j])
                    _store_chunk_atom(tile, chunk, j, acc)
                    packed, word, leader = _codec_quant(acc, lane, tid)
                    _lds_write_packet(j, packed, word, leader)
            elif k < n_ops:
                # All-gather: the chunk is already final, so decode it for our
                # own output and forward the bytes we received *unmodified*. Not
                # dequantizing-and-requantizing is what keeps this lap free of
                # additional error -- the ring's one accuracy advantage over the
                # two-shot, which requantizes its reduced slice to broadcast it.
                for j in range_constexpr(rank_atoms):
                    packed_in, word_in = _recv_raw(k - 2, sub, j)
                    _store_chunk_atom(
                        tile, chunk, j, _codec_dequant(packed_in, _scale_of(word_in))
                    )
                    _lds_write_packet(j, packed_in, word_in, is_leader)
            else:
                # Final op: the last chunk arrives and stops here.
                for j in range_constexpr(rank_atoms):
                    packed_in, word_in = _recv_raw(k - 2, sub, j)
                    _store_chunk_atom(
                        tile, chunk, j, _codec_dequant(packed_in, _scale_of(word_in))
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

            The op loop is ``range_constexpr``, **not** ``range``. Inside a
            ``@flyc.kernel`` body a plain ``range`` lowers to a *device* loop
            with a runtime induction variable, which collapses the ``2N-1`` op
            bodies into one and quietly demotes every compile-time decision
            keyed on ``k`` -- which chunk, which branch, which wire slot. It
            does not fail loudly: the kernel still runs and most tiles still
            come out right. The tell is the instruction count (2 peer stores
            emitted at TP4 where 12 are needed).
            """
            for _ki in range_constexpr(n_ops):
                k = _ki + 1
                if k >= 2:
                    _wait(k - 2, color)
                else:
                    pass  # op 1 is a pure send: there is nothing to wait for
                for s in range(fx.Int32(0), n_this, fx.Int32(1)):
                    tile = bid + (i + s) * n_blocks
                    _op_substep(k, tile, s)
                    if k <= steps:
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
                if k <= steps:
                    _publish(k - 1, color)
                else:
                    pass

        # Stride by the *launched* grid, not the compile-time cap: the host
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
    tag = f"ws{world_size}_r{rank}_st{super_tile}_{inbox_memory}_{rs_codec}"
    launch_qr_int4_ring.func.__name__ = f"launch_qr_int4_ring_{tag}"
    try:
        qr_int4_ring.func.__name__ = f"qr_int4_ring_{tag}"
    except AttributeError:
        pass
    return {
        "launch": launch_qr_int4_ring,
        "flags_bytes": 0,  # the handshake rides in each slot's 64 B tail
        "data_bytes": inbox_bytes,
        "lds_bytes": LDS_BYTES,
        "tile_bytes": TILE_BYTES,
        "tile_fp16": TILE_FP16,
        "rank_tile_bytes": RANK_TILE_BYTES,
        "wire_tile_bytes": wire_tile_bytes,
        "super_tile": super_tile,
        "world_size": world_size,
        "rank": rank,
        "inbox_memory": inbox_memory,
        "rs_codec": rs_codec,
        "payload_policy": payload_policy,
        "flag_policy": flag_policy,
        "release_writeback": release_writeback,
        "rank_atoms": rank_atoms,
        "steps": steps,
        "grid": grid,
        "block": BLOCK,
    }
