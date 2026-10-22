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

The ring schedule has a cost in accuracy. The reduce-scatter lap quantizes the
*running partial sum* again at each of its ``N-1`` hops. The maximum absolute
value of the partial sum increases with each contribution that the lap adds.
For this reason, the kernel has two codec arguments, one for each lap. An INT6
reduce-scatter lap gives better accuracy and uses slightly more bandwidth.

The all-gather lap sends the bytes that it receives without a change. For this
reason, it adds only one quantization. It stays INT4 unless the caller sets
``ag_codec``.

The third wire format, ``"fp16"``, sends the values without quantization.
This is used in testing.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T

from ...allreduce_policy import (
    AG_CODECS,
    RING_SUPER_TILES,
    RS_CODECS,
    SUPPORTED_WORLDS,
)
from .quick_allreduce_codec import (
    GROUP,
    _clamp_fp16_overflow,
    _codec_dequant,
    _codec_load,
    _codec_quant,
    _scale_from_word,
    codecs_for_block,
    scale_slot_of,
    thread_lane,
)
from .quick_allreduce_epilogue import QuickReducePlainEpilogue
from .quick_allreduce_fusions import check_fusion_args
from .quick_allreduce_rmsnorm import RingRMSNormEpilogue
from .transport import (
    _INBOX_POLICY,
    _RECV_POLICY,
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


def _load_i32_at(ptr, elem_off, cache_modifier):
    """One i32 from the buffer pointer *ptr* at an element offset.

    The ring always has the inbox pointer in hand and only the offset
    varies, so its payload reads go through the one shared descriptor.
    """
    return _buffer_load(ptr, elem_off, 1, fx.Int32, cache_modifier)[0]


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
    inbox_memory: str,
    rs_codec: str,
    ag_codec: str,
    fusion: str = "none",
    hidden: int | None = None,
    block: int | None = None,
    h_pad: int | None = None,
    mxfp4: bool = False,
):
    """Build the ring kernel for one *rank*.

    ``rank`` is a **compile-time** parameter. At op ``k``, a block operates on
    chunk ``(rank - k) % N``. This index selects an item from a Python list of
    atom fragments in registers, so the index must be a Python constant.

    ``block`` is threads per workgroup.

    ``fusion="rmsnorm"`` appends a residual-add + RMSNorm epilogue
    (``quick_allreduce_rmsnorm``) and requires ``hidden``. It changes the
    geometry rather than just the tail: the block is sized so a chunk is a
    whole number of token rows (see
    ``quick_allreduce_fusions.quick_reduce_row_block``), which is what lets the
    epilogue run inside the op that receives a chunk. Without it a row would
    span several ops, and since this schedule carries nothing across a publish,
    the only way to reassemble one would be to pin ``super_tile`` at 1 -- which
    is where the ring's throughput goes to die.
    """
    check_fusion_args(fusion, hidden, h_pad, mxfp4=mxfp4)
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

    if fusion == "none":
        epilogue = QuickReducePlainEpilogue(block=block, world_size=world_size)
    else:
        epilogue = RingRMSNormEpilogue(
            hidden=hidden,
            h_pad=h_pad,
            block=block,
            world_size=world_size,
            mxfp4=mxfp4,
        )
    block = epilogue.block
    quads_per_block = block // QUAD_LANES
    tile_i32 = block * ATOMS * 4
    tile_bytes = tile_i32 * 4

    codecs = codecs_for_block(block)
    rs = codecs[rs_codec]
    ag = codecs[ag_codec]
    # Which codec each wire slot carries. Ops 1..N-1 fill the reduce-scatter
    # slots and ops N..2N-1 the all-gather ones, so the split is exactly at
    # N-1 and is a compile-time property of the step index.
    step_codec = [rs] * (world_size - 1) + [ag] * (world_size - 1)

    # Per-step geometry.
    payload_i32 = [rank_atoms * c.rank_tile_i32 for c in step_codec]
    release_i32_off = [super_tile * p for p in payload_i32]
    wire_tile_i32 = [r + 16 for r in release_i32_off]  # + one 64 B handshake sector
    step_base_i32, _acc = [], 0
    for w in wire_tile_i32:
        step_base_i32.append(_acc)
        _acc += grid * w
    inbox_bytes = _acc * 4

    # Each slot has only one writer, the ring predecessor. For this reason, the
    # inbox has no sender axis and the design does not need atomics. PCIe has
    # no peer atomics.
    total_sectors = [rank_atoms * c.n_sectors for c in step_codec]
    fanout_rounds = [-(-t // quads_per_block) for t in total_sectors]

    # One staging buffer, with the size for the larger of the two codecs. It
    # has only ``rank_atoms`` rows, because a ring step stages the packet for
    # one destination only. At TP8 with INT6, this is 1664 B.
    pack_row_i32 = max(rs.rank_tile_i32, ag.rank_tile_i32)
    pack_i32 = rank_atoms * pack_row_i32
    PackStorage = make_pack_storage(pack_i32)

    # The epilogue's per-wave LDS partials, if any, follow the pack staging.
    lds_bytes = pack_i32 * 4 + epilogue.partial_floats * 4

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
        # Read only by a fused epilogue; the plain launcher passes zeros.
        res_in_ptr: fx.Int64,
        res_out_ptr: fx.Int64,
        w_ptr: fx.Int64,
        eps: fx.Float32,
        w_bias: fx.Float32,
        q_ptr: fx.Int64,
        q_scale_ptr: fx.Int64,
    ):
        _clamp_fp16_overflow()
        tid = fx.Int32(gpu.thread_id("x"))
        bid = fx.Int32(gpu.block_id("x"))

        wave, lane = thread_lane(tid, block)
        quad_layout = fx.make_layout((QUADS_PER_WAVE, QUAD_LANES), (QUAD_LANES, 1))
        quad, lane_in_quad = fx.idx2crd(lane, quad_layout).unpack()
        quad_id = wave * fx.Int32(QUADS_PER_WAVE) + quad

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
        peers = _load_peers(peer_ptrs, world_size)
        # The ring uses only two peers: this rank and its successor.
        self_base = peers[rank]
        next_base = peers[nxt]
        inbox = _buffer_ptr(_to_sgpr_i64(self_base), T.i32, 4, inbox_bytes)

        # Everything that touches HBM goes through the epilogue: the plain one
        # moves atoms through ``_payload_io``, a fused one reaches the same rows
        # as raw bf16 atoms. Its LDS, if any, comes from this kernel's one
        # allocator, after the pack staging.
        ep = epilogue.bind(
            tid=tid,
            nbytes=nbytes,
            num_tiles=num_tiles,
            inp_ptr=inp_ptr,
            out_ptr=out_ptr,
            res_in_ptr=res_in_ptr,
            res_out_ptr=res_out_ptr,
            w_ptr=w_ptr,
            eps=eps,
            w_bias=w_bias,
            q_ptr=q_ptr,
            q_scale_ptr=q_scale_ptr,
            allocator=allocator,
        )
        # Bare names, not ``ep.<hook>(...)``: the rewriter takes a method call
        # inside a runtime loop or branch as a possible mutation of its base
        # and would try to carry ``ep`` through the loop as an SSA value.
        _load_atom = ep.load_atom
        _begin_chunk = ep.begin_chunk
        _complete_atom = ep.complete
        _end_chunk = ep.end_chunk
        _load_color, _store_color = _color_io(colors_ptr, bid)

        def _slot_i32(step, sub):
            """i32 offset of (*step*, this block, *sub*) inside the inbox.

            With two codecs the steps are no longer uniformly sized, so the base
            is a prefix sum rather than a product.
            """
            return (
                fx.Int32(step_base_i32[step])
                + bid * fx.Int32(wire_tile_i32[step])
                + sub * fx.Int32(payload_i32[step])
            )

        def _load_chunk_atoms(tile, chunk):
            """This rank's own bf16 data for *chunk*, as fp16 register atoms."""
            return [
                _load_atom(tile, chunk * rank_atoms + j)
                for j in range_constexpr(rank_atoms)
            ]

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
                return _load_i32_at(inbox, base + off, _RECV_POLICY)

            return _codec_load(codec, _get, tid, scale_slot)

        def _scale_of(codec, word):
            return _scale_from_word(codec, word, pair_in_slot)

        def _fanout_to_next(step, sub):
            """Push the staged rank-tiles from LDS into the successor's inbox.

            One destination, sectors in address order, so the whole
            ``rank_atoms * rank_tile B`` lands as a single contiguous run. Quads
            past the sector count do not store. ``safe`` keeps their address
            arithmetic in range.

            INT6 is 26 sectors to INT4's 18, so a TP8 hop drives 26 of the 64
            quads rather than 18: the wider codec uses the fanout better.
            """
            codec = step_codec[step]
            n_sectors = codec.n_sectors
            n_total = total_sectors[step]
            for rnd in range_constexpr(fanout_rounds[step]):
                s = quad_id + fx.Int32(rnd * quads_per_block)
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

            ``vmcnt`` counts the stores of one wave only. For this reason, all
            waves of the workgroup must join at a barrier before the flag
            store. On a cacheable inbox, a retired store can stay in L2, where
            the peer cannot see it. For this reason, the release must write
            back L2. There is only
            one destination, so the first ``FLAG_LANES`` lanes write the flag.
            """
            rocdl.s_waitcnt(vmcnt=0)
            gpu.barrier()
            if const_expr(release_scope is not None):
                _release_inbox(release_scope)
            if tid < fx.Int32(FLAG_LANES):
                elem = (
                    _slot_i32(step, fx.Int32(0))
                    + fx.Int32(release_i32_off[step])
                    + tid * fx.Int32(FLAG_I32_PER_LANE)
                )
                _store_flag_peer(next_base + _i32_to_bytes(elem), color, flag_policy)

        def _wait(step, color):
            """Spin until the predecessor has coloured *step*'s slot in our inbox.

            There is only one source, so only one thread spins. The poll
            must bypass the caches (``_load_flag`` is ``sc0
            sc1``): a load answered from a stale line would spin forever, which
            is a hang rather than a slowdown.

            One colour covers all ``2(N-1)`` slots of a super-tile group. That is
            safe without extra sequencing because the ring's own dependency
            chain bounds how far a rank can run ahead: for the predecessor to be
            writing slot ``k-1`` of the *next* group while we are reading slot
            ``k-2`` of this one, it would have had to complete this group, which
            transitively requires us to have completed op ``N`` -- i.e. to be
            past the read we are blocked on.

            Polls through a global address rather than a buffer descriptor:
            a descriptor built from ``self_base + elem*4`` would be a uniform
            value LLVM cannot prove uniform, sourcing all four descriptor
            dwords from VGPRs and serializing the wave around them.
            """
            if tid == fx.Int32(0):
                flag = self_base + _i32_to_bytes(
                    _slot_i32(step, fx.Int32(0)) + fx.Int32(release_i32_off[step])
                )
                # `sc0 sc1`, so each retry is fetched past L1 and L2 and no
                # fence is needed in the loop; the acquire below orders the
                # payload reads after it, once, after the join.
                current = _load_flag(flag)
                while current != color:
                    current = _load_flag(flag)
            gpu.barrier()
            # Unconditional, *after* the join, and not just inside the spin.
            # Only `tid == 0` spins, so an acquire inside the loop would cover
            # one lane of one wave and leave the rest of the workgroup reading
            # the payload unordered after it -- and would be skipped entirely
            # in the common case where the flag is already set on the first
            # read.
            #
            # The scope is the inbox policy's. `_RECV_POLICY` is `sc0 sc1`, so
            # the payload loads below bypass both caches on every inbox and
            # cannot be served a stale line on their own. On an uncached inbox
            # the policy therefore gives a workgroup-scope acquire: no
            # invalidate, only a compiler barrier that keeps those loads below
            # the spin. Every other inbox keeps the system-scope acquire, an
            # L1+L2 invalidate, as a defence against a silent wrong result.
            #
            # A system-scope acquire must be preceded by a writeback. The ring
            # stores one chunk per all-gather op and then waits again -- so an
            # invalidate here sits directly on top of dirty output lines, and
            # discarding them silently loses whole chunks. A workgroup-scope
            # acquire invalidates nothing, so it needs neither the writeback
            # nor the drain before it.
            if const_expr(acquire_scope == rocdl.SyncScope.OneAs):
                rocdl.s_waitcnt(vmcnt=0)
                _release_inbox(rocdl.SyncScope.OneAs)
            _acquire_inbox(acquire_scope)

        def _complete(tile, chunk, j, value, recv):
            """Hand an atom whose all-reduce is finished to the epilogue."""
            _complete_atom(tile, chunk * rank_atoms + j, value, recv)

        def _op_substep(k, tile, sub, recv):
            """One op of the ring, for one sub-tile. Stages LDS; does not send.

            Completed atoms go to the epilogue's ``complete``, which may hold
            them in *recv*, a caller-owned list, until ``end_chunk`` runs after
            the fanout -- a list rather than a return value: inside a
            ``@flyc.kernel`` body the AST transform rewrites ``return`` away.

            The codecs are compile-time too: an op reads the codec of the step
            it receives from (``k-2``) and writes the codec of the step it sends
            into (``k-1``). Those coincide everywhere except op ``N``, which is
            the seam between the two laps -- it reads a reduce-scatter slot and
            writes an all-gather one.
            """
            chunk = _chunk_of(k)
            is_leader = (tid % fx.Int32(GROUP)) == fx.Int32(0)
            if const_expr(k >= world_size):
                # Ahead of the receive, so the HBM latency hides behind it.
                _begin_chunk(tile, chunk, recv)
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
                # this is the final sum for our own chunk. Quantize it for the
                # all-gather's first send, and write our own output from the
                # packet that we just built. The other ranks decode the same packet,
                # so every rank ends with the same bits.
                atoms = _load_chunk_atoms(tile, chunk)
                for j in range_constexpr(rank_atoms):
                    words_in, word_in = _recv_raw(c_in, k - 2, sub, j)
                    acc = _codec_dequant(
                        c_in, words_in, _scale_of(c_in, word_in), tid, atoms[j]
                    )
                    words, word, leader = _codec_quant(c_out, acc, lane, tid)
                    _lds_write_packet(c_out, j, words, word, leader)
                    _complete(
                        tile,
                        chunk,
                        j,
                        _codec_dequant(c_out, words, _scale_of(c_out, word), tid),
                        recv,
                    )
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
                    # A caller-owned list rather than a return value; see
                    # ``_op_substep``. Only the epilogue reads it.
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
                    if const_expr(k >= world_size):
                        # After the fanout, but before this group's _publish: the
                        # successor's flag waits for the epilogue's stores too,
                        # since _publish drains every store of this wave.
                        _end_chunk(tile, _chunk_of(k), recv)
                    else:
                        pass
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
            fx.Float32(0.0),
            fx.Int64(0),
            fx.Int64(0),
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
        w_bias: fx.Float32,
        q_ptr: fx.Int64,
        q_scale_ptr: fx.Int64,
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
            w_bias,
            q_ptr,
            q_scale_ptr,
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
    tag += f"_b{block}" + epilogue.tag
    launcher = (
        launch_quick_allreduce_ring_fused
        if epilogue.takes_operands
        else launch_quick_allreduce_ring
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
        "hbm_tile_bytes": epilogue.hbm_tile_bytes,
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
        "rows_per_chunk": epilogue.rows_per_chunk,
        **epilogue.spec(),
    }
