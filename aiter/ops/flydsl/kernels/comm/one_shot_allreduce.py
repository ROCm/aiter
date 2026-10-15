# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""gfx942/gfx950 TP∈{2,4,8} exact one-shot (1-stage) all-reduce.

Decode-regime kernel: bf16 in, fp32 accumulate, bf16 out, no codec. One
communication round, no grid-wide barrier and no flags -- each rank pushes its
whole tile into every peer's inbox, and the payload is its own flag (a Lamport
scheme). Inbox slots hold a sentinel value until a peer's data lands. The receiver
polls its 16 B per source until no half-word is the sentinel, reduces the N
copies, and re-arms the slot two rounds ahead.

This kernel trades wire volume for round trips:
(N-1)*S pushed rather than (N-1)*S read, but one serialized fabric traversal.

No LDS is needed: thread ``t``'s 16 B lands at the same offset in every destination,
so it can be pushed straight from registers.

On a cacheable (fine-grained) inbox each wave writes its push back out of L2,
and the re-arm is written through.

Padded builds (a hidden dim with no native row geometry) mask their pad columns
with a per-row buffer descriptor bounded to ``hidden*2`` bytes: pad columns are
out of bounds for that descriptor, so a plain tiled copy gets hardware OOB
handling for free (load->0, store dropped), with no exec-mask split. See
``quick_allreduce_fusions.make_rowbuf_atom_row``.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T

# The peer-store/load primitives, the cache-policy table and the inbox-memory
# taxonomy are shared with the quantized kernels verbatim.
from ...allreduce_policy import (
    FANOUT_ORDERS,
    SUPPORTED_ATOMS,
    SUPPORTED_BLOCKS,
    SUPPORTED_SPLITS,
    SUPPORTED_WORLDS,
)
from .one_shot_allreduce_rmsnorm import OneShotRMSNormEpilogue
from .quick_allreduce_fusions import check_fusion_args, make_row_io
from .transport import (
    _CM_SC0,
    _CM_SC1,
    _INBOX_POLICY,
    _RECV_POLICY,
    _buffer_load,
    _buffer_ptr,
    _buffer_store,
    _color_io,
    _i32_to_bytes,
    _load_peers,
    _poll_v4i32,
    _release_inbox,
    _store_v4i32_peer,
    _to_sgpr_i64,
    atom_bf16_to_f32,
    atom_f32_to_bf16,
)

# ``block`` is threads per block, which sets the tile width:
# ``tile = block * atoms * 16 B``. It sets the parallelism floor at a given
# payload. The trade is per-block fixed cost: every block polls and re-arms its
# own slots, so that work rises by the same factor the block count does.
#
# 16 B per thread per atom -- one ``global_store_dwordx4``.
ATOM_BYTES = 16
ATOM_I32 = ATOM_BYTES // 4

# Three inbox buffers rotate by colour. Round ``c`` reads buffer ``c``, and
# re-arms buffer ``c + 2`` (mod 3) with the sentinel. No peer can be writing that
# one: a peer is at most one round ahead, and that round uses ``c + 1``.
INBOX_BUFFERS = 3
# The sentinel is bf16 (and fp16) NaN ``0xFFFF`` in every half-word, so an unused
# inbox is just a ``0xFF`` memset. A sender rewrites a sentinel half-word in its
# own payload to ``0x7FFF``, which is still NaN, so the result is unchanged.
SENTINEL_FILL_BYTE = 0xFF
# Bit 15 of both half-words of an i32, as a signed i32.
_HALF_SIGN_BITS = -0x7FFF8000  # 0x80008000


def _sentinel_mask(atom_i32):
    """Bit 15 of every half-word of *atom_i32* that is the sentinel ``0xFFFF``.

    Branch-free zero test per 16-bit lane on the complement: a half-word of
    ``~x`` is non-zero iff adding ``0x7FFF`` to its low 15 bits, or its own bit
    15, sets bit 15. The low 15 bits are masked first, so no carry crosses into
    the upper half-word.
    """
    x = atom_i32 ^ -1
    nonzero = ((x & 0x7FFF7FFF) + 0x7FFF7FFF) | x
    return (nonzero & _HALF_SIGN_BITS) ^ _HALF_SIGN_BITS


def _canonicalize(atom_i32):
    """*atom_i32* with every sentinel half-word ``0xFFFF`` turned into ``0x7FFF``."""
    return atom_i32 ^ _sentinel_mask(atom_i32)


def _has_sentinel(vec_i32):
    """Whether any half-word of the i32 vector *vec_i32* is still the sentinel."""
    mask = _sentinel_mask(vec_i32)
    acc = mask[0]
    for i in range(1, vec_i32.numel):
        acc = acc | mask[i]
    return acc != 0


class _PlainEpilogue:
    """The plain all-reduce's end of a tile: round the fp32 sum once to bf16
    and store it.

    Also the reference for the hooks the schedule drives an epilogue through
    (``one_shot_allreduce_rmsnorm.OneShotRMSNormEpilogue`` is the other one).
    At build time an epilogue fixes ``block`` and the HBM row geometry; in the
    kernel body ``bind`` returns an object with

    * ``prefetch(tile)`` -- loads issued before the push, so they retire
      during the poll; handed back to ``finish``.
    * ``init_state(color)`` / ``finish(tile, color, accs, prefetched, state)``
      / ``persist(color, state)`` -- *state* is carried from tile to tile (a
      pytree of values, ``()`` when there is none); ``finish`` stores the
      reduced fp32 *accs* and returns the next state; ``persist`` runs on one
      thread after the last tile.

    Hooks run inline in the kernel trace, so they hold no runtime ``if`` or
    loop of their own -- those live in the kernel body or a ``@flyc.jit``
    helper, the only code the AST rewriter sees.
    """

    fusion = "none"
    padded = False
    hidden = None
    xchg_bytes = 0
    lds_bytes = 0
    tag = ""

    def __init__(self, *, block: int | None, atoms: int):
        if block is None:
            raise ValueError("a plain build requires block")
        block = int(block)
        if block not in SUPPORTED_BLOCKS:
            raise ValueError(f"block must be one of {SUPPORTED_BLOCKS}, got {block!r}")
        self.block = block
        self.row_stride_i32 = block * atoms * ATOM_I32

    def spec(self) -> dict:
        return {
            "fusion": self.fusion,
            "hidden": None,
            "h_pad": None,
            "padded": False,
            "split": 1,
            "xchg_bytes": 0,
        }

    def bind(self, *, io, out_buf, **_unused):
        self.io = io
        self.out_buf = out_buf
        return self

    def prefetch(self, tile):
        return None

    def init_state(self, color):
        return ()

    def finish(self, tile, color, accs, prefetched, state):
        self.io.store_tile(self.out_buf, tile, [atom_f32_to_bf16(a) for a in accs])
        return state

    def persist(self, color, state):
        pass


def make_one_shot_allreduce_kernel(
    *,
    world_size: int,
    atoms: int,
    grid: int,
    inbox_memory: str,
    fanout: str,
    rank: int,
    block: int | None = None,
    fusion: str = "none",
    hidden: int | None = None,
    h_pad: int | None = None,
    split: int = 1,
):
    """Build the one-shot all-reduce, plain or fused with residual-add + RMSNorm.

    ``split`` (fused only) spreads each token row over that many workgroups;
    see ``one_shot_allreduce_rmsnorm``.
    """
    check_fusion_args(fusion, hidden, h_pad)
    split = int(split)
    if split not in SUPPORTED_SPLITS:
        raise ValueError(f"split must be one of {SUPPORTED_SPLITS}, got {split!r}")
    if split > 1 and fusion == "none":
        raise ValueError("split is only meaningful for a fused build")
    if world_size not in SUPPORTED_WORLDS:
        raise ValueError(
            f"world_size must be one of {SUPPORTED_WORLDS}, got {world_size}"
        )
    if not 0 <= int(rank) < world_size:
        raise ValueError(f"rank must be in [0, {world_size}), got {rank}")
    if atoms not in SUPPORTED_ATOMS:
        raise ValueError(f"atoms must be one of {SUPPORTED_ATOMS}, got {atoms!r}")
    if inbox_memory not in _INBOX_POLICY:
        raise ValueError(
            f"inbox_memory must be one of {tuple(_INBOX_POLICY)}, got {inbox_memory!r}"
        )
    if fanout not in FANOUT_ORDERS:
        raise ValueError(f"fanout must be one of {FANOUT_ORDERS}, got {fanout!r}")
    if grid < 1:
        raise ValueError(f"grid must be positive, got {grid}")

    policy = _INBOX_POLICY[inbox_memory]
    payload_policy = policy["payload"]
    release_scope = policy["release"]

    if fusion == "none":
        epi = _PlainEpilogue(block=block, atoms=atoms)
    else:
        epi = OneShotRMSNormEpilogue(
            hidden=hidden,
            h_pad=h_pad,
            split=split,
            atoms=atoms,
            block=block,
            grid=grid,
            n_colors=INBOX_BUFFERS,
        )
    block = epi.block
    padded = epi.padded

    tile_bytes = block * atoms * ATOM_BYTES
    tile_i32 = tile_bytes // 4
    data_bytes = INBOX_BUFFERS * grid * world_size * tile_bytes

    # This rank's own index as a trace-time constant. It has to be
    # compile-time: the peer fanout, the poll and the reduce are all unrolled
    # over trace-time peer indices, and "all peers but me" is only expressible
    # there. The cost is one kernel binary per rank -- but a process is one
    # rank, so it compiles exactly one.
    self_rank = int(rank)
    # Peers this rank pushes payload to. Our own contribution never round-trips
    # through our own inbox: it stays in registers. Our own inbox slot is simply
    # never touched -- the wire format is unchanged, the slot is still
    # allocated, and no peer can observe the difference.
    push_peers = [p for p in range(world_size) if p != self_rank]

    # (peer, atom) iteration order for the fanout, unrolled at trace time.
    if fanout == "peer":
        fanout_pairs = [(p, a) for p in push_peers for a in range(atoms)]
    else:
        fanout_pairs = [(p, a) for a in range(atoms) for p in push_peers]

    # One signature for both modes. The fused-only arguments are present (and
    # passed as zeros) in a plain build rather than being appended to a second
    # kernel; a plain epilogue never reads them, so only five unused kernargs
    # remain in a plain build. ``xchg_ptr`` is the split build's exchange
    # state, and 0 (never read) in every other build.
    @flyc.kernel(known_block_size=[block, 1, 1])
    def one_shot_allreduce(
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
        xchg_ptr: fx.Int64,
    ):
        tid = fx.Int32(gpu.thread_id("x"))
        bid = fx.Int32(gpu.block_id("x"))

        # Every HBM operand moves through one tiled copy, one atom-wide row at
        # a time. A padded build addresses each row through a per-row buffer
        # descriptor bounded to the true width; an unpadded one slices a
        # whole-tensor buffer tensor. The one-shot's tile is a single row, so
        # it never outruns M and the rows need no bound of their own.
        io = make_row_io(
            tid=tid,
            nbytes=nbytes,
            num_tiles=num_tiles,
            block=block,
            atoms_per_row=atoms,
            rows_per_tile=1,
            row_stride_i32=epi.row_stride_i32,
            padded=padded,
            hidden=epi.hidden,
            bound_rows=False,
        )

        peers = _load_peers(peer_ptrs, world_size)
        peer_vec = fx.Vector.from_elements(peers, dtype=fx.Int64)
        inbox = _buffer_ptr(_to_sgpr_i64(peer_vec[rank]), T.i32, 16)

        in_buf = io.operand(inp_ptr)
        out_buf = io.operand(out_ptr)
        _load_color, _store_color = _color_io(colors_ptr, bid)

        def _wrap(color):
            """*color* reduced into ``[0, INBOX_BUFFERS)``, for a *color* below
            twice that."""
            if color >= fx.Int32(INBOX_BUFFERS):
                color = color - fx.Int32(INBOX_BUFFERS)
            return color

        ep = epi.bind(
            tid=tid,
            bid=bid,
            io=io,
            out_buf=out_buf,
            res_in_ptr=res_in_ptr,
            res_out_ptr=res_out_ptr,
            w_ptr=w_ptr,
            eps=eps,
            xchg_ptr=xchg_ptr,
            wrap=_wrap,
        )
        # Bare names, not ``ep.<hook>(...)``: the rewriter takes a method call
        # inside a runtime loop or branch as a possible mutation of its base
        # and would try to carry ``ep`` through the loop as an SSA value.
        _prefetch = ep.prefetch
        _finish = ep.finish
        _persist = ep.persist

        def _slot_i32(buf, src):
            """i32 offset of the wire slot ``[buf][bid][src]``.

            Plain arithmetic rather than ``crd2idx`` on a 3-D layout: at
            ``grid == 1`` the middle mode is unit and gets coalesced away,
            after which a three-coordinate lookup silently returns a wrong
            (negative) index. The ring kernel hit exactly this.
            """
            return (
                buf * fx.Int32(grid * world_size * tile_i32)
                + bid * fx.Int32(world_size * tile_i32)
                + src * fx.Int32(tile_i32)
            )

        def _load_tile(tile):
            return io.load_tile(in_buf, tile)

        def _fanout(buf, my_atoms):
            """Push this thread's atoms into every peer's slot for this rank.

            Thread ``t``'s data lands at the same offset in every destination,
            so it goes straight from registers -- no LDS staging.

            Our own inbox is not among the destinations: our contribution stays
            in registers, which saves 1/N of the stores, the polls and the
            re-arms.

            On a cacheable inbox an ``nt`` push can sit in this XCD's L2 until
            something evicts it, so each wave writes it back with a release
            fence.
            """
            for peer, atom in fanout_pairs:
                _store_v4i32_peer(
                    peer_vec[peer]
                    + _i32_to_bytes(
                        _slot_i32(buf, rank)
                        + fx.Int32(atom * block * ATOM_I32)
                        + tid * fx.Int32(ATOM_I32)
                    ),
                    my_atoms[atom],
                    payload_policy,
                )
            if const_expr(release_scope is not None):
                _release_inbox(release_scope)

        def _recv_i32(buf, src, atom):
            """i32 offset of this thread's 16 B of *atom* in the slot of *src*."""
            return (
                _slot_i32(buf, fx.Int32(src))
                + fx.Int32(atom * block * ATOM_I32)
                + tid * fx.Int32(ATOM_I32)
            )

        def _load_recv(buf, src, atom):
            return _buffer_load(
                inbox, _recv_i32(buf, src, atom), ATOM_I32, fx.Int32, _RECV_POLICY
            )

        def _poll(buf):
            """Receive: this thread's 16 B of every atom of every source,
            keyed ``(src, atom)``, once none of them is the sentinel.

            The first reads all go out before any is tested. Until every slot
            has landed, every slot is then re-read on each pass, together,
            rather than one source at a time: a serial spin would leave the later
            sources' reads stale, and pay one more round trip per source after
            the last one lands. Re-reading a slot that has already landed is
            harmless, since nobody writes it again this round. The re-reads go
            through ``_poll_v4i32``, so the spin cannot be deleted.
            """
            keys = [
                (src, atom) for src in push_peers for atom in range_constexpr(atoms)
            ]
            addrs = [
                peer_vec[rank] + _i32_to_bytes(_recv_i32(buf, src, atom))
                for src, atom in keys
            ]

            def _pack(vecs):
                return fx.Vector.from_elements(
                    [v[e] for v in vecs for e in range_constexpr(ATOM_I32)], fx.Int32
                )

            got = _pack([_load_recv(buf, src, atom) for src, atom in keys])
            while _has_sentinel(got):
                got = _pack([_poll_v4i32(addr) for addr in addrs])
            return {
                key: fx.Vector.from_elements(
                    [got[i * ATOM_I32 + e] for e in range_constexpr(ATOM_I32)],
                    fx.Int32,
                )
                for i, key in enumerate(keys)
            }

        def _rearm(buf):
            """Put the sentinel back in every slot this thread reads, in *buf*.

            On a cacheable inbox the sentinel is written through (``sc0 sc1``).
            An ``nt`` store could leave it dirty in our own L2, where it would
            hide a peer's later push from the poll, and its eventual eviction
            would overwrite that push in memory. The stores are local, so
            writing through costs no fabric traffic.
            """
            sentinel = fx.Vector.filled((ATOM_I32,), -1, fx.Int32)
            for src in push_peers:
                for atom in range_constexpr(atoms):
                    if const_expr(release_scope is not None):
                        _buffer_store(
                            inbox,
                            _recv_i32(buf, src, atom),
                            sentinel,
                            _CM_SC0 | _CM_SC1,
                        )
                    else:
                        _store_v4i32_peer(
                            peer_vec[rank] + _i32_to_bytes(_recv_i32(buf, src, atom)),
                            sentinel,
                            payload_policy,
                        )

        def _reduce_f32(my_atoms, polled):
            """Sum this thread's atom across all N contributions, in rank order.

            Rank order, not a rotated order: every rank must accumulate in the
            same sequence or the results differ in the last bit across ranks.
            ``cross_device_reduce`` makes the same promise for the same reason.
            Our own contribution comes out of the registers; the peers' come
            from *polled*, ``_poll``'s result.

            Left unrounded: the epilogue rounds -- the plain one once, at the
            end of the sum, and the fused one needs the fp32 accumulator for
            the norm.
            """
            outs = []
            for atom in range_constexpr(atoms):
                acc = None
                for src in range_constexpr(world_size):
                    if const_expr(src == self_rank):
                        raw = my_atoms[atom]
                    else:
                        raw = polled[(src, atom)]
                    v = atom_bf16_to_f32(raw)
                    acc = v if acc is None else acc + v
                outs.append(acc)
            return outs

        def _round(tile, color, raw_atoms, prefetched, state):
            """One tile, from its input atoms and the epilogue's prefetch.

            Returns ``(next colour, next epilogue state)``. The colour is the
            buffer index itself, kept in [0, 3), so it never wraps out of step
            with the rotation.
            """
            my_atoms = [_canonicalize(v) for v in raw_atoms]
            _fanout(color, my_atoms)
            polled = _poll(color)
            state = _finish(
                tile, color, _reduce_f32(my_atoms, polled), prefetched, state
            )
            _rearm(_wrap(color + fx.Int32(2)))
            return _wrap(color + fx.Int32(1)), state

        # Stride by the *launched* grid, not the compile-time cap: the host may
        # launch fewer blocks than ``grid``, and striding by the cap would leave
        # every tile above n_blocks unprocessed.
        n_block_tiles = (num_tiles - bid + n_blocks - fx.Int32(1)) // n_blocks
        # The host never launches more blocks than tiles, so every block has a
        # first tile, and its input is read before the colour: the two loads are
        # then in flight together rather than one after the other.
        first_atoms = _load_tile(bid)
        first_x = _prefetch(bid)
        color = _load_color()
        state = ep.init_state(color)
        first = _round(bid, color, first_atoms, first_x, state)
        color = first[0]
        state = first[1]
        for i in range(fx.Int32(1), n_block_tiles, fx.Int32(1)):
            # The last round's re-arm must land before this round's push: a peer
            # that sees the push can be one round further on, writing into the
            # slot just re-armed. Between calls the kernel boundary does this.
            rocdl.s_waitcnt(vmcnt=0)
            tile = bid + i * n_blocks
            step = _round(tile, color, _load_tile(tile), _prefetch(tile), state)
            color = step[0]
            state = step[1]
        if tid == 0:
            _store_color(color)
            _persist(color, state)
        gpu.barrier()

    flat_wg = f"{block},{block}"

    @flyc.jit
    def launch_one_shot_allreduce(
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
        xchg_ptr: fx.Int64,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        one_shot_allreduce(
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
            xchg_ptr,
            value_attrs={"rocdl.flat_work_group_size": flat_wg},
        ).launch(grid=(grid_x, 1, 1), block=(block, 1, 1), stream=stream)

    tag = f"ws{world_size}_r{self_rank}_a{atoms}_g{grid}_{inbox_memory}_b{block}"
    if atoms > 1:
        tag += f"_{fanout}"
    tag += epi.tag
    launch_one_shot_allreduce.func.__name__ = f"launch_one_shot_allreduce_{tag}"
    try:
        one_shot_allreduce.func.__name__ = f"one_shot_allreduce_{tag}"
    except AttributeError:
        pass
    return {
        "launch": launch_one_shot_allreduce,
        "flags_bytes": 0,
        "data_bytes": data_bytes,
        "lds_bytes": epi.lds_bytes,
        "tile_bytes": tile_bytes,
        # No handshake sector: a wire slot is just the payload.
        "wire_tile_bytes": tile_bytes,
        # Shims for ``quick_allreduce._StEngine``, which is reused verbatim for the IPC
        # inbox and peer table. This schedule has no super-tile (one round per
        # tile, nothing to batch) and no per-rank tile split (every rank sends
        # the whole tile), so the two are 1 and the full tile respectively.
        "super_tile": 1,
        "rank_tile_bytes": tile_bytes,
        "tile_fp16": tile_bytes // 2,
        "atoms": atoms,
        "world_size": world_size,
        "inbox_memory": inbox_memory,
        "fanout": fanout,
        "rank": self_rank,
        # Byte every inbox byte starts as.
        "inbox_fill": SENTINEL_FILL_BYTE,
        "grid": grid,
        "block": block,
        **epi.spec(),
    }
