# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""gfx942/gfx950 TP∈{2,4,8} exact one-shot (1-stage) all-reduce.

Decode-regime kernel: bf16 in, fp32 accumulate, bf16 out, no codec. One
communication round and no grid-wide barrier -- each rank pushes its whole
tile into every peer's inbox, publishes a colour flag, waits for the N flags,
then reduces N copies out of its own inbox.

This kernel trades wire volume for round trips:
(N-1)*S pushed rather than (N-1)*S read, but ~2 serialized fabric traversals
rather than ~6.

No LDS is needed: thread ``t``'s 16 B lands at the same offset in every destination,
so it can be pushed straight from registers.

``lamport=True`` builds the same push without the flags: the payload is its own
flag. Inbox slots hold a sentinel until a peer's data lands, the receiver polls
its 16 B per source until no half-word is the sentinel, and then re-arms a slot
two rounds ahead. That removes the ``vmcnt(0)`` drain, both barriers and the
flag round trip from every tile.
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
    FLAG_I32_PER_LANE,
    FLAG_LANES,
    SUPPORTED_WORLDS,
    _acquire_inbox,
    _buffer_load,
    _buffer_ptr,
    _color_io,
    _i32_to_bytes,
    _load_flag,
    _load_peers,
    _payload_io,
    _poll_v4i32,
    _release_inbox,
    _store_flag_peer,
    _store_v4i32_peer,
    _to_sgpr_i64,
)

DEFAULT_BLOCK = 256
# Threads per block, which sets the tile width: ``tile = block * atoms * 16 B``.
# It sets the parallelism floor at a given payload.
#
# The trade is flags and per-block fixed cost: the flag count (``blocks * (N-1)``)
# rises by the same factor the block count does.
SUPPORTED_BLOCKS = (64, 128, 256, 512)
# 16 B per thread per atom -- one ``global_store_dwordx4``.
ATOM_BYTES = 16
ATOM_I32 = ATOM_BYTES // 4
DEFAULT_ATOMS = 1
# Atoms per thread per tile. More atoms means a bigger tile, hence fewer blocks
# and fewer flags for a given payload, at the cost of coarser load balance on
# the last partial tile.
SUPPORTED_ATOMS = (1, 2, 4)
DEFAULT_GRID_CAP = 64

# Per-``(link, world_size)`` tuning ladder: ``(min_bytes, atoms, grid_cap,
# fanout, block, skip_self, lamport)`` rungs. The host builds one engine per
# rung and selects by payload size at launch. Created from a tuning sweep.
#
# ``atoms`` and ``block`` both scale the tile, and their product is what
# matters to the block count; they are separate knobs because only ``block``
# also changes the workgroup size, and only ``atoms`` also changes how many
# stores one thread has in flight.
#
#   PCIe
#
#     TP2  block 512, atoms=1/4 -- the 8 KiB tile to 384 KiB (the whole fast-mode
#                               window, where the mesh takes over), then the
#                               32 KiB tile with cap 128 over the rest of the
#                               exact-mode window, which runs to 64 MiB.
#     TP4  block 256, atoms=1/2/4 -- Three-phases: the 4 KiB tile wins to 42 KiB,
#                               the 8 KiB tile to ~98 KiB, the 16 KiB tile above.
#     TP8  block 512, atoms=1/2 -- the 8 KiB tile to 16 KiB, then the 16 KiB
#                               tile over the rest of the 80 KiB window. The
#                               wide workgroup wins at every size.
#
#   xGMI
#
#     TP2  atoms=1 cap128 b256 -- one rung over the whole window
#     TP4  atoms=1 cap128 b256 -- one rung over the whole window
#     TP8  atoms=1 cap128 b256 -- one rung over the whole window
ONESHOT_LADDER = {
    ("pcie", 2): (
        (0, 1, 64, "peer", 512, True, False),
        (384 << 10, 4, 128, "peer", 512, True, False),
    ),
    ("pcie", 4): (
        (0, 1, 64, "peer", 256, True, False),
        (48 << 10, 2, 64, "atom", 256, True, False),
        (96 << 10, 4, 128, "peer", 256, True, False),
    ),
    ("pcie", 8): (
        (0, 1, 64, "peer", 512, True, False),
        (16 << 10, 2, 64, "peer", 512, True, False),
    ),
    ("xgmi", 2): ((0, 1, 128, "peer", 256, False, False),),
    ("xgmi", 4): ((0, 1, 128, "peer", 256, False, False),),
    ("xgmi", 8): ((0, 1, 128, "peer", 256, False, False),),
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
                DEFAULT_LAMPORT,
            ),
        ),
    )


# Inbox slots are indexed by ``colour & 1``. Two buffers is exactly enough to
# let one rank run a whole call ahead of another without overwriting a slot the
# straggler has not read.
PARITIES = 2
# 64 B handshake sector at the tail of each wire slot, as 16 i32 copies of the
# colour -- one 8 B store from each of ``FLAG_LANES`` lanes.
FLAG_I32 = 16
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
# costs a store, a load and a flag per tile in memory the rank already holds in
# registers, which is 1/N of each; dropping it specialises the binary per rank.
DEFAULT_SKIP_SELF = False

# Lamport mode: the payload is its own flag.
#
# Three inbox buffers rotate by colour. Round ``c`` reads buffer ``c``, and
# re-arms buffer ``c + 2`` (mod 3) with the sentinel. No peer can be writing that
# one: a peer is at most one round ahead, and that round uses ``c + 1``.
LAMPORT_BUFFERS = 3
# The sentinel is bf16 (and fp16) NaN ``0xFFFF`` in every half-word, so an unused
# inbox is just a ``0xFF`` memset. A sender rewrites a sentinel half-word in its
# own payload to ``0x7FFF``, which is still NaN, so the result is unchanged.
LAMPORT_FILL_BYTE = 0xFF
DEFAULT_LAMPORT = False
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
    lamport: bool = DEFAULT_LAMPORT,
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
    if inbox_memory not in _INBOX_POLICY:
        raise ValueError(
            f"inbox_memory must be one of {tuple(_INBOX_POLICY)}, got {inbox_memory!r}"
        )
    if fanout not in FANOUT_ORDERS:
        raise ValueError(f"fanout must be one of {FANOUT_ORDERS}, got {fanout!r}")
    if grid < 1:
        raise ValueError(f"grid must be positive, got {grid}")
    if lamport and inbox_memory != "uncached":
        # The receiver has no flag to order its loads after, so a payload store
        # must not be able to sit in the writer's L2: only the uncached inbox
        # guarantees that for an ``nt`` store.
        raise ValueError(
            f"lamport needs the uncached inbox, got inbox_memory={inbox_memory!r}"
        )

    policy = _INBOX_POLICY[inbox_memory]
    payload_policy = policy["payload"]
    flag_policy = policy["flag"]
    release_scope = policy["release"]
    acquire_scope = policy["acquire"]

    tile_bytes = block * atoms * ATOM_BYTES
    tile_i32 = tile_bytes // 4
    # Payload then the 64 B handshake sector, or the payload alone in Lamport
    # mode.
    wire_tile_i32 = tile_i32 + (0 if lamport else FLAG_I32)
    wire_tile_bytes = wire_tile_i32 * 4
    buffers = LAMPORT_BUFFERS if lamport else PARITIES
    data_bytes = buffers * grid * world_size * wire_tile_bytes

    # This rank's own index as a trace-time constant, or None when the self
    # slot is being used. It has to be compile-time: the peer fanout, the flag
    # publish and the reduce are all unrolled over trace-time peer indices, and
    # "all peers but me" is only expressible there. The cost is one kernel
    # binary per rank -- but a process is one rank, so it compiles exactly one.
    self_rank = int(rank) if skip_self else None
    # Peers this rank pushes payload and flags to. With ``skip_self`` our own
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

        def _slot_i32(parity, src):
            """i32 offset of the wire slot ``[parity][bid][src]``.

            Plain arithmetic rather than ``crd2idx`` on a 3-D layout: at
            ``grid == 1`` the middle mode is unit and gets coalesced away,
            after which a three-coordinate lookup silently returns a wrong
            (negative) index. The ring kernel hit exactly this.
            """
            return (
                parity * fx.Int32(grid * world_size * wire_tile_i32)
                + bid * fx.Int32(world_size * wire_tile_i32)
                + src * fx.Int32(wire_tile_i32)
            )

        def _load_tile(tile):
            """This thread's 16 B of each atom of *tile*, as raw i32x4."""
            return [_load_payload(tile, atom) for atom in range_constexpr(atoms)]

        def _store_tile(tile, vals):
            for atom in range_constexpr(atoms):
                _store_payload(tile, atom, vals[atom])

        def _fanout(parity, my_atoms):
            """Push this thread's atoms into every peer's slot for this rank.

            Thread ``t``'s data lands at the same offset in every destination,
            so it goes straight from registers -- no LDS staging.

            ``skip_self`` decides whether the fanout includes our own inbox.
            Keeping it makes the receive loop uniform over ``world_size``;
            dropping it removes 1/N of the stores, 1/N of the reduce's loads and
            1/N of the flags, at the cost of one kernel binary per rank.
            """
            for peer, atom in fanout_pairs:
                _store_v4i32_peer(
                    peer_vec[peer]
                    + _i32_to_bytes(
                        _slot_i32(parity, rank)
                        + fx.Int32(atom * block * ATOM_I32)
                        + tid * fx.Int32(ATOM_I32)
                    ),
                    my_atoms[atom],
                    payload_policy,
                )

        def _publish(parity, color):
            """Drain the payload stores, then write *color* into every peer.

            ``vmcnt(0)`` retires this wave's stores; the barrier joins the other
            waves, whose ``vmcnt`` is separate. On a cacheable inbox retiring is
            not enough -- the lines can sit in this XCD's L2 -- so the release
            fence writes them back and waits for that before the flag goes out.
            Every workgroup issues its own writeback: L2 is per-XCD.
            """
            rocdl.s_waitcnt(vmcnt=0)
            gpu.barrier()
            if const_expr(release_scope is not None):
                _release_inbox(release_scope)
            # FLAG_LANES lanes, 8 B each -> the 64 B sector, unrolled over the
            # destinations. The peer index must be a trace-time constant: an
            # earlier version keyed it off the lane (``peer = tid // 4``, 4 lanes
            # per destination), which made ``peer_vec[peer]`` a *lane-varying*
            # extract from a 4xi64 vector. That lowers to a scratch round-trip,
            # and at ``atoms>1`` the register pressure made it land in the
            # payload: 8 B of peer pointer at 16 B stride over a 64 B span, once
            # per 256 B, in atom 0 of the highest-numbered rank's inbox. See the
            # ``SUPPORTED_ATOMS`` note. ``_fanout`` always unrolled; this is now
            # consistent with it.
            if tid < fx.Int32(FLAG_LANES):
                elem = (
                    _slot_i32(parity, rank)
                    + fx.Int32(tile_i32)
                    + tid * fx.Int32(FLAG_I32_PER_LANE)
                )
                for peer in push_peers:
                    _store_flag_peer(
                        peer_vec[peer] + _i32_to_bytes(elem), color, flag_policy
                    )

        def _wait(parity, color):
            """Spin until every rank's flag in our own inbox shows *color*.

            One spinner per source. The fences after the join are placed there,
            not inside the spin, on purpose: if the flag is already present the
            loop body never runs, and a fence placed only inside it would be
            skipped in the common case.

            The fences depend on the inbox type (``_INBOX_POLICY``). A cacheable
            inbox gets a writeback then a system-scope acquire, i.e., an L1+L2
            invalidate, so the payload loads cannot hit a stale line. Write back
            *before* invalidating, or the output lines this block already wrote
            are discarded. The uncached inbox has no stale line to invalidate:
            the memory is never cached and the loads bypass L1 and L2
            (``_RECV_POLICY``). It gets a workgroup-scope acquire, which only
            keeps the payload loads below the spin.
            """
            # Lane ``t`` watches one source. Without ``skip_self`` that is
            # source ``t``; with it our own flag is never published, so the
            # N-1 lanes step over our own index and the last lane sits out.
            # Computed before the guard rather than nested inside it, so the
            # remap is a flat ``scf.if`` yielding one value.
            spin_src = tid
            if const_expr(skip_self):  # noqa: SIM102
                if tid >= fx.Int32(self_rank):
                    spin_src = tid + fx.Int32(1)
            if tid < fx.Int32(len(push_peers)):
                flag = peer_vec[rank] + _i32_to_bytes(
                    _slot_i32(parity, spin_src) + fx.Int32(tile_i32)
                )
                # `sc0 sc1`, so each retry is fetched past L1 and L2 and no
                # fence is needed in the loop; the fences below order the
                # payload reads after it, once, after the join.
                current = _load_flag(flag)
                while current != color:
                    current = _load_flag(flag)
            gpu.barrier()
            rocdl.s_waitcnt(vmcnt=0)
            if const_expr(release_scope is not None):
                _release_inbox(release_scope)
            _acquire_inbox(acquire_scope)

        def _recv_i32(parity, src, atom):
            """i32 offset of this thread's 16 B of *atom* in the slot of *src*."""
            return (
                _slot_i32(parity, fx.Int32(src))
                + fx.Int32(atom * block * ATOM_I32)
                + tid * fx.Int32(ATOM_I32)
            )

        def _load_recv(parity, src, atom):
            return _buffer_load(
                inbox, _recv_i32(parity, src, atom), ATOM_I32, fx.Int32, _RECV_POLICY
            )

        def _poll(parity):
            """Lamport receive: this thread's 16 B of every atom of every source,
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
                peer_vec[rank] + _i32_to_bytes(_recv_i32(parity, src, atom))
                for src, atom in keys
            ]

            def _pack(vecs):
                return fx.Vector.from_elements(
                    [v[e] for v in vecs for e in range_constexpr(ATOM_I32)], fx.Int32
                )

            got = _pack([_load_recv(parity, src, atom) for src, atom in keys])
            while _has_sentinel(got):
                got = _pack([_poll_v4i32(addr) for addr in addrs])
            return {
                key: fx.Vector.from_elements(
                    [got[i * ATOM_I32 + e] for e in range_constexpr(ATOM_I32)],
                    fx.Int32,
                )
                for i, key in enumerate(keys)
            }

        def _rearm(parity):
            """Put the sentinel back in every slot this thread reads, in *parity*."""
            sentinel = fx.Vector.filled((ATOM_I32,), -1, fx.Int32)
            for src in push_peers:
                for atom in range_constexpr(atoms):
                    _store_v4i32_peer(
                        peer_vec[rank] + _i32_to_bytes(_recv_i32(parity, src, atom)),
                        sentinel,
                        payload_policy,
                    )

        def _reduce(parity, my_atoms, polled=None):
            """Sum this thread's atom across all N contributions, in rank order.

            Rank order, not a rotated order: every rank must accumulate in the
            same sequence or the results differ in the last bit across ranks.
            ``cross_device_reduce`` makes the same promise for the same reason.
            Under ``skip_self`` our own contribution comes out of the registers
            rather than out of the inbox. *polled* is ``_poll``'s result in
            Lamport mode, where the inbox has already been read.
            """
            outs = []
            for atom in range_constexpr(atoms):
                acc = None
                for src in range_constexpr(world_size):
                    if const_expr(src == self_rank):
                        raw = my_atoms[atom]
                    elif const_expr(polled is not None):
                        raw = polled[(src, atom)]
                    else:
                        raw = _load_recv(parity, src, atom)
                    v = _atom_bf16_to_f32(raw)
                    acc = v if acc is None else acc + v
                outs.append(_atom_f32_to_bf16(acc))
            return outs

        def _lamport_round(tile, color, raw_atoms):
            """One tile in Lamport mode, from its input atoms; returns the next
            colour. The colour is the buffer index itself, kept in [0, 3), so
            it never wraps out of step with the rotation."""
            my_atoms = [_canonicalize(v) for v in raw_atoms]
            _fanout(color, my_atoms)
            _store_tile(tile, _reduce(color, my_atoms, _poll(color)))
            rearm = color + fx.Int32(2)
            if rearm >= fx.Int32(LAMPORT_BUFFERS):
                rearm = rearm - fx.Int32(LAMPORT_BUFFERS)
            _rearm(rearm)
            color = color + fx.Int32(1)
            if color == fx.Int32(LAMPORT_BUFFERS):
                color = fx.Int32(0)
            return color

        # Stride by the *launched* grid, not the compile-time cap: the host may
        # launch fewer blocks than ``grid``, and striding by the cap would leave
        # every tile above n_blocks unprocessed.
        n_block_tiles = (num_tiles - bid + n_blocks - fx.Int32(1)) // n_blocks
        if const_expr(lamport):
            # The host never launches more blocks than tiles, so every block has
            # a first tile, and its input is read before the colour: the two
            # loads are then in flight together rather than one after the other.
            first_atoms = _load_tile(bid)
            color = _load_color()
            color = _lamport_round(bid, color, first_atoms)
            for i in range(fx.Int32(1), n_block_tiles, fx.Int32(1)):
                # The last round's re-arm must land before this round's push: a
                # peer that sees the push can be one round further on, writing
                # into the slot just re-armed. Between calls the kernel boundary
                # does this.
                rocdl.s_waitcnt(vmcnt=0)
                tile = bid + i * n_blocks
                color = _lamport_round(tile, color, _load_tile(tile))
        else:
            color = _load_color()
            for i in range(fx.Int32(0), n_block_tiles, fx.Int32(1)):
                tile = bid + i * n_blocks
                parity = color & fx.Int32(1)
                my_atoms = _load_tile(tile)
                _fanout(parity, my_atoms)
                _publish(parity, color)
                _wait(parity, color)
                _store_tile(tile, _reduce(parity, my_atoms))
                color = color + fx.Int32(1)
                # 0 is the unset sentinel. The
                # inbox slot is `color & 1`, and the colour before the wrap is -1,
                # which is odd: resuming at 1 would put two consecutive tiles in
                # the same slot, and a rank one tile ahead would overwrite data a
                # peer is still reading.
                if color == fx.Int32(0):
                    color = fx.Int32(2)
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
    if lamport:
        tag += "_lp"
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
        "wire_tile_bytes": wire_tile_bytes,
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
        "lamport": lamport,
        # Byte every inbox byte starts as.
        "inbox_fill": LAMPORT_FILL_BYTE if lamport else 0,
        "grid": grid,
        "block": block,
    }
