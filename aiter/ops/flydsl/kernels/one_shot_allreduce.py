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
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import Int32, Int64, Stream, T

# The peer-store/load primitives, the cache-policy table and the inbox-memory
# taxonomy are shared with the quantized kernels verbatim.
from .quick_allreduce_shared import (
    _CM_SC0,
    _CM_SC1,
    _INBOX_POLICY,
    SUPPORTED_WORLDS,
    _buffer_load,
    _buffer_ptr,
    _buffer_store,
    _color_io,
    _i32_to_bytes,
    _load_peers,
    _payload_io,
    _poll_v4i32,
    _release_inbox,
    _store_v4i32_peer,
    _to_sgpr_i64,
)

DEFAULT_BLOCK = 256
# Threads per block, which sets the tile width: ``tile = block * atoms * 16 B``.
# It sets the parallelism floor at a given payload.
#
# The trade is per-block fixed cost: every block polls and re-arms its own
# slots, so that work rises by the same factor the block count does.
SUPPORTED_BLOCKS = (64, 128, 256, 512)
# 16 B per thread per atom -- one ``global_store_dwordx4``.
ATOM_BYTES = 16
ATOM_I32 = ATOM_BYTES // 4
DEFAULT_ATOMS = 1
# Atoms per thread per tile. More atoms means a bigger tile, hence fewer blocks
# for a given payload, at the cost of coarser load balance on the last partial
# tile.
SUPPORTED_ATOMS = (1, 2, 4)
DEFAULT_GRID_CAP = 64

# Per-``(link, world_size)`` tuning ladder: ``(min_bytes, atoms, grid_cap,
# fanout, block, skip_self)`` rungs. The host builds one engine per rung and
# selects by payload size at launch. Created from a tuning sweep.
#
# ``atoms`` and ``block`` both scale the tile, and their product is what
# matters to the block count; they are separate knobs because only ``block``
# also changes the workgroup size, and only ``atoms`` also changes how many
# stores one thread has in flight.
#
#   PCIe
#
#     TP2  b256 a1 cap128 -- the 4 KiB tile to 384 KiB, then b512 a4 cap128,
#                            the 32 KiB tile, over the rest of the exact-mode
#                            window, which runs to 64 MiB.
#     TP4  b128 a1 cap128 -- the 2 KiB tile to 96 KiB, then b512 a2 cap64, the
#                            16 KiB tile, to the 1 MiB exact-mode ceiling.
#     TP8  b64 a1 cap64   -- the 1 KiB tile, except b256 a1 cap64, the 4 KiB
#                            tile, from 48 to 192 KiB.
#
#   xGMI
#
#     TP2  atoms=1 cap128 b128/b256 -- the 2 KiB tile to 256 KiB, the most it
#                               covers in one round, then the 4 KiB tile.
#     TP4  atoms=1 cap128 b256 -- one rung over the whole window
#     TP8  atoms=1 cap128 b128 -- one rung over the whole window
ONESHOT_LADDER = {
    ("pcie", 2): (
        (0, 1, 128, "peer", 256, True),
        (384 << 10, 4, 128, "peer", 512, True),
    ),
    ("pcie", 4): (
        (0, 1, 128, "peer", 128, True),
        (96 << 10, 2, 64, "peer", 512, True),
    ),
    ("pcie", 8): (
        (0, 1, 64, "peer", 64, True),
        (48 << 10, 1, 64, "peer", 256, True),
        (192 << 10, 1, 64, "peer", 64, True),
    ),
    ("xgmi", 2): (
        (0, 1, 128, "peer", 128, True),
        (256 << 10, 1, 128, "peer", 256, True),
    ),
    ("xgmi", 4): ((0, 1, 128, "peer", 256, True),),
    ("xgmi", 8): ((0, 1, 128, "peer", 128, True),),
}


def oneshot_ladder(world_size: int, link: str = "pcie"):
    """Rungs for *(link, world_size)*, or a single default rung if unlisted.

    *link* defaults to ``"pcie"`` so a caller that has not resolved the fabric
    gets the conservative table: its fatter tiles cost throughput on xGMI but
    are never wrong in the sense of failing.
    """
    return ONESHOT_LADDER.get(
        (str(link), int(world_size)),
        (
            (
                0,
                DEFAULT_ATOMS,
                DEFAULT_GRID_CAP,
                DEFAULT_FANOUT,
                DEFAULT_BLOCK,
                DEFAULT_SKIP_SELF,
            ),
        ),
    )


# Read our own inbox with the caches bypassed: a peer wrote these lines
# microseconds ago and an L1 hit here is a stale hit. Same reasoning as the
# ring kernel's ``_RECV_POLICY``.
_RECV_POLICY = _CM_SC0 | _CM_SC1

# Which axis of the (peer, atom) fanout runs fastest across consecutive stores.
#
# "peer": a thread pushes all its atoms to one destination before moving to the
# next, so a wave hands each destination a contiguous ``block * 16`` B run.
#
# "atom": consecutive stores walk the peers of one atom. On xGMI the native
# packet is 64 B and there is no per-destination run-length benefit to collect,
# so spreading across links sooner can start more of them in parallel.
FANOUT_ORDERS = ("peer", "atom")
DEFAULT_FANOUT = "peer"

# Whether a rank pushes its own contribution through its own inbox. Keeping it
# costs a store, a poll and a re-arm per tile in memory the rank already holds
# in registers, which is 1/N of each; dropping it specialises the binary per
# rank.
DEFAULT_SKIP_SELF = False

# Three inbox buffers rotate by colour. Round ``c`` reads buffer ``c``, and
# re-arms buffer ``c + 2`` (mod 3) with the sentinel. No peer can be writing that
# one: a peer is at most one round ahead, and that round uses ``c + 1``.
INBOX_BUFFERS = 3
# Inbox memory types this kernel has been validated on.
SUPPORTED_INBOX_MEMORY = ("uncached", "finegrained")
# The sentinel is bf16 (and fp16) NaN ``0xFFFF`` in every half-word, so an unused
# inbox is just a ``0xFF`` memset. A sender rewrites a sentinel half-word in its
# own payload to ``0x7FFF``, which is still NaN, so the result is unchanged.
SENTINEL_FILL_BYTE = 0xFF
# Bit 15 of both half-words of an i32, as a signed i32.
_HALF_SIGN_BITS = -0x7FFF8000  # 0x80008000


def _atom_bf16_to_f32(atom_i32):
    """16 B of bf16 (8 values) -> 8 f32. bf16 is the high half of f32, so this
    is a widening move, not a conversion -- exact, no rounding."""
    return fx.Vector(atom_i32).bitcast(fx.BFloat16).to(fx.Float32)


def _atom_f32_to_bf16(acc_f32):
    """8 f32 -> 16 B of bf16. One rounding, at the end of the reduction, which
    is what makes this bit-comparable with ``cross_device_reduce``'s fp32
    accumulate + single ``downcast``."""
    return acc_f32.to(fx.BFloat16).bitcast(fx.Int32)


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


def make_one_shot_allreduce_kernel(
    *,
    world_size: int,
    atoms: int = DEFAULT_ATOMS,
    grid: int,
    inbox_memory: str = "uncached",
    fanout: str = DEFAULT_FANOUT,
    skip_self: bool = False,
    rank: int | None = None,
    block: int = DEFAULT_BLOCK,
):
    if block not in SUPPORTED_BLOCKS:
        raise ValueError(f"block must be one of {SUPPORTED_BLOCKS}, got {block!r}")
    if world_size not in SUPPORTED_WORLDS:
        raise ValueError(
            f"world_size must be one of {SUPPORTED_WORLDS}, got {world_size}"
        )
    if skip_self and not 0 <= (rank if rank is not None else -1) < world_size:
        raise ValueError(
            f"skip_self needs the rank at trace time, got rank={rank!r} for "
            f"world_size={world_size}"
        )
    if atoms not in SUPPORTED_ATOMS:
        raise ValueError(f"atoms must be one of {SUPPORTED_ATOMS}, got {atoms!r}")
    if inbox_memory not in SUPPORTED_INBOX_MEMORY:
        raise ValueError(
            f"inbox_memory must be one of {SUPPORTED_INBOX_MEMORY}, got "
            f"{inbox_memory!r}"
        )
    if fanout not in FANOUT_ORDERS:
        raise ValueError(f"fanout must be one of {FANOUT_ORDERS}, got {fanout!r}")
    if grid < 1:
        raise ValueError(f"grid must be positive, got {grid}")

    policy = _INBOX_POLICY[inbox_memory]
    payload_policy = policy["payload"]
    release_scope = policy["release"]

    tile_bytes = block * atoms * ATOM_BYTES
    tile_i32 = tile_bytes // 4
    data_bytes = INBOX_BUFFERS * grid * world_size * tile_bytes

    # This rank's own index as a trace-time constant, or None when the self
    # slot is being used. It has to be compile-time: the peer fanout, the poll
    # and the reduce are all unrolled over trace-time peer indices, and
    # "all peers but me" is only expressible there. The cost is one kernel
    # binary per rank -- but a process is one rank, so it compiles exactly one.
    self_rank = int(rank) if skip_self else None
    # Peers this rank pushes payload to. With ``skip_self`` our own
    # inbox slot is simply never touched: the wire format is unchanged, the slot
    # is still allocated, and no peer can observe the difference.
    push_peers = [p for p in range(world_size) if p != self_rank]

    # (peer, atom) iteration order for the fanout, unrolled at trace time.
    if fanout == "peer":
        fanout_pairs = [(p, a) for p in push_peers for a in range(atoms)]
    else:
        fanout_pairs = [(p, a) for a in range(atoms) for p in push_peers]

    @flyc.kernel(known_block_size=[block, 1, 1])
    def one_shot_allreduce(
        rank: Int32,
        nbytes: Int64,
        num_tiles: Int32,
        inp_ptr: Int64,
        out_ptr: Int64,
        peer_ptrs: Int64,
        colors_ptr: Int64,
        n_blocks: Int32,
    ):
        tid = fx.Int32(gpu.thread_id("x"))
        bid = fx.Int32(gpu.block_id("x"))

        peers = _load_peers(peer_ptrs, world_size)
        peer_vec = fx.Vector.from_elements(peers, dtype=fx.Int64)
        inbox = _buffer_ptr(_to_sgpr_i64(peer_vec[rank]), T.i32, 16)

        _load_payload, _store_payload = _payload_io(
            inp_ptr, out_ptr, nbytes, num_tiles, atoms, block, tid
        )
        _load_color, _store_color = _color_io(colors_ptr, bid)

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
            """This thread's 16 B of each atom of *tile*, as raw i32x4."""
            return [_load_payload(tile, atom) for atom in range_constexpr(atoms)]

        def _store_tile(tile, vals):
            for atom in range_constexpr(atoms):
                _store_payload(tile, atom, vals[atom])

        def _fanout(buf, my_atoms):
            """Push this thread's atoms into every peer's slot for this rank.

            Thread ``t``'s data lands at the same offset in every destination,
            so it goes straight from registers -- no LDS staging.

            ``skip_self`` decides whether the fanout includes our own inbox.
            Keeping it makes the receive loop uniform over ``world_size``;
            dropping it removes 1/N of the stores, the polls and the re-arms, at
            the cost of one kernel binary per rank.

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
                            peer_vec[rank]
                            + _i32_to_bytes(_recv_i32(buf, src, atom)),
                            sentinel,
                            payload_policy,
                        )

        def _reduce(my_atoms, polled):
            """Sum this thread's atom across all N contributions, in rank order.

            Rank order, not a rotated order: every rank must accumulate in the
            same sequence or the results differ in the last bit across ranks.
            ``cross_device_reduce`` makes the same promise for the same reason.
            Under ``skip_self`` our own contribution comes out of the registers.
            The peers' come from *polled*, ``_poll``'s result.
            """
            outs = []
            for atom in range_constexpr(atoms):
                acc = None
                for src in range_constexpr(world_size):
                    if const_expr(src == self_rank):
                        raw = my_atoms[atom]
                    else:
                        raw = polled[(src, atom)]
                    v = _atom_bf16_to_f32(raw)
                    acc = v if acc is None else acc + v
                outs.append(_atom_f32_to_bf16(acc))
            return outs

        def _round(tile, color, raw_atoms):
            """One tile, from its input atoms; returns the next colour. The
            colour is the buffer index itself, kept in [0, 3), so it never wraps
            out of step with the rotation."""
            my_atoms = [_canonicalize(v) for v in raw_atoms]
            _fanout(color, my_atoms)
            _store_tile(tile, _reduce(my_atoms, _poll(color)))
            rearm = color + fx.Int32(2)
            if rearm >= fx.Int32(INBOX_BUFFERS):
                rearm = rearm - fx.Int32(INBOX_BUFFERS)
            _rearm(rearm)
            color = color + fx.Int32(1)
            if color == fx.Int32(INBOX_BUFFERS):
                color = fx.Int32(0)
            return color

        # Stride by the *launched* grid, not the compile-time cap: the host may
        # launch fewer blocks than ``grid``, and striding by the cap would leave
        # every tile above n_blocks unprocessed.
        n_block_tiles = (num_tiles - bid + n_blocks - fx.Int32(1)) // n_blocks
        # The host never launches more blocks than tiles, so every block has a
        # first tile, and its input is read before the colour: the two loads are
        # then in flight together rather than one after the other.
        first_atoms = _load_tile(bid)
        color = _load_color()
        color = _round(bid, color, first_atoms)
        for i in range(fx.Int32(1), n_block_tiles, fx.Int32(1)):
            # The last round's re-arm must land before this round's push: a peer
            # that sees the push can be one round further on, writing into the
            # slot just re-armed. Between calls the kernel boundary does this.
            rocdl.s_waitcnt(vmcnt=0)
            tile = bid + i * n_blocks
            color = _round(tile, color, _load_tile(tile))
        if tid == 0:
            _store_color(color)
        gpu.barrier()

    flat_wg = f"{block},{block}"

    @flyc.jit
    def launch_one_shot_allreduce(
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
        one_shot_allreduce(
            rank,
            nbytes,
            num_tiles,
            inp_ptr,
            out_ptr,
            peer_ptrs,
            colors_ptr,
            grid_x,
            value_attrs={"rocdl.flat_work_group_size": flat_wg},
        ).launch(grid=(grid_x, 1, 1), block=(block, 1, 1), stream=stream)

    tag = f"ws{world_size}_a{atoms}_g{grid}_{inbox_memory}_b{block}"
    if atoms > 1:
        tag += f"_{fanout}"
    if skip_self:
        # ``_r<n>_`` is the rank field the bench's variant comparison already
        # knows to collapse before checking that the ranks agree; a build
        # specialised per rank legitimately reports a different string on each.
        tag += f"_r{self_rank}_ss"
    launch_one_shot_allreduce.func.__name__ = f"launch_one_shot_allreduce_{tag}"
    try:
        one_shot_allreduce.func.__name__ = f"one_shot_allreduce_{tag}"
    except AttributeError:
        pass
    return {
        "launch": launch_one_shot_allreduce,
        "flags_bytes": 0,
        "data_bytes": data_bytes,
        "lds_bytes": 0,
        "tile_bytes": tile_bytes,
        # No handshake sector: a wire slot is just the payload.
        "wire_tile_bytes": tile_bytes,
        # Shims for ``quick_allreduce_int4._StEngine``, which is reused verbatim for the IPC
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
        "skip_self": skip_self,
        # Byte every inbox byte starts as.
        "inbox_fill": SENTINEL_FILL_BYTE,
        "grid": grid,
        "block": block,
    }
