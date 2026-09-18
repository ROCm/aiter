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
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import (
    Float32,
    Int32,
    Int64,
    Stream,
    T,
    as_ir_value,
)

from . import buffer_ops

# The fused epilogue and the row-to-workgroup geometry are shared with the
# quantized schedules; see quick_allreduce_fusions.
from .quick_allreduce_fusions import (
    ATOM_ELEMS,
    FUSIONS,
    make_wave_partials,
    pack_bf16,
    residual_add,
    rms_rstd,
    row_block,
    row_block_options,
    row_block_supported,
    scale_by_weight,
)

# The peer-store/load primitives, the cache-policy table and the inbox-memory
# taxonomy are shared with the quantized kernels verbatim.
from .quick_allreduce_shared import (
    _CM_SC0,
    _CM_SC1,
    _INBOX_POLICY,
    SUPPORTED_WORLDS,
    WAVE,
    _acquire_inbox,
    _i32_to_bytes,
    _to_sgpr_i64,
    atom_bf16_to_f32,
    atom_f32_to_bf16,
)


def _store_v4i32_peer_multi(pairs, policy):
    """Emit a whole fanout of 16 B peer stores as ONE inline-asm block.

    Same instruction as ``quick_allreduce_shared._store_v4i32_peer``, but every store in
    the group lives inside a single ``InlineAsmOp``. It lives here rather than
    beside it because this kernel is its only consumer; the two must keep
    agreeing on the ``global_store_dwordx4 ... {policy}`` encoding.

    Why that matters: a VMEM store samples its address and data VGPRs
    asynchronously *after* issue, so those registers must stay live until
    ``vmcnt`` retires the store. LLVM guarantees that for real store
    instructions -- ``SIInsertWaitcnts`` tracks the operands -- but it cannot
    see inside inline asm. It therefore believes the data is dead the instant
    the asm "executes" and is free to recycle those VGPRs for the next
    address computation:

        global_store_dwordx4 v[44:45], v[14:17], off nt   ; reads v[14:17]
        v_lshl_add_u64       v[14:15], v[46:47], 0, v[8:9] ; clobbers them

    which sends the next peer's *pointer* down the wire in place of the first
    8 B of payload. Only shows up under register pressure -- one atom per
    thread has slack, four does not.

    Grouping the stores fixes it because LLVM allocates every operand of one
    asm block to a distinct register and emits nothing between them, so
    nothing can clobber a pending store's sources. The caller must still
    ``s_waitcnt vmcnt(0)`` before reusing the values, which is what
    ``_publish`` already does on the next line.

    *pairs* is a sequence of ``(addr_i64, data_v4i32)``.
    """
    ptr_ty = ir.Type.parse("!llvm.ptr<1>")
    operands, slots = [], []
    # Reference a repeated payload once: listing the same value N times would
    # have LLVM allocate N copies of it. Keyed on the *caller's* Python object,
    # not on ``==`` over the lowered ir.Value -- MLIR compares those
    # structurally, which silently folds four distinct atoms into one operand
    # and stores atom 0's data for every atom.
    data_slot: dict[int, int] = {}

    for addr_i64, data in pairs:
        ptr = llvm.IntToPtrOp(ptr_ty, as_ir_value(addr_i64)).result
        operands.append(ptr)
        a = len(operands) - 1
        key = id(data)
        if key not in data_slot:
            operands.append(as_ir_value(data))
            data_slot[key] = len(operands) - 1
        slots.append((a, data_slot[key]))

    asm = "\n\t".join(
        f"global_store_dwordx4 ${a}, ${d}, off {policy}" for a, d in slots
    )
    llvm.InlineAsmOp(
        None,
        operands,
        asm,
        ",".join("v" * len(operands)),
        has_side_effects=True,
    )


DEFAULT_BLOCK = 256
# Threads per block, which sets the tile width: ``tile = block * atoms * 16 B``. 
# It sets the parallelism floor at a given payload.
#
# The trade is flags and per-block fixed cost: the flag count (``blocks * (N-1)``)
# rises by the same factor the block count does.
SUPPORTED_BLOCKS = (64, 128, 256)
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
#     TP2  block 128, atoms=2, cap 64 -- a 4 KiB tile reached with a half-size
#                               workgroup. Below 96 KiB the schedule is
#                               flag-bound, and 128 threads is the cheapest way
#                               to hold the tile narrow.
#          block 256, atoms=4, cap 128 -- above 96 KiB the trade reverses and
#                               the wider cap matters, because this window runs
#                               to 1.5 MiB: at a 16 KiB tile that is 96 tiles,
#                               so a cap of 64 would leave half the blocks
#                               running two serialized handshake rounds.
#     TP4  block 256, atoms=1/2/4 -- Three-phases: the 4 KiB tile wins to 42 KiB, 
#                               the 8 KiB tile to ~98 KiB, the 16 KiB tile above.
#     TP8  block 256, atoms=4, cap 64 -- the fattest tile, one rung over the
#                               whole 80 KiB window: the fanout is to 7 peers
#                               and cutting the flag count matters more than
#                               the handful of blocks lost.
#
#   TODO: xGMI needs to re-measured to see if self-skip is beneficial also here.
#   xGMI 
#     TP2  atoms=2, cap 64   -- one rung over the whole 4 MiB window.
#     TP4  atoms=1, cap 128  -- the narrow tile wins throughout, and the extra
#                               blocks matter more than tile width because peer
#                               bandwidth is not the constraint.
#     TP8  atoms=1, cap 64   -- one rung over the whole 256 KiB window.
ONESHOT_LADDER = {
    ("pcie", 2): (
        (0, 2, 64, "peer", 128, True),
        (96 << 10, 4, 128, "peer", 256, True),
    ),
    ("pcie", 4): (
        (0, 1, 64, "peer", 256, True),
        (48 << 10, 2, 64, "atom", 256, True),
        (96 << 10, 4, 128, "peer", 256, True),
    ),
    ("pcie", 8): ((0, 4, 64, "peer", 256, True),),
    ("xgmi", 2): ((0, 2, 64, "peer", 256, False),),
    ("xgmi", 4): ((0, 1, 128, "peer", 256, False),),
    ("xgmi", 8): ((0, 1, 64, "peer", 256, False),),
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


# In the plain schedule ``atoms`` sets the *tile width*: tile = BLOCK*atoms*16 B,
# so a bigger atom count means fewer, fatter tiles and fewer flags. TP2 and TP8
# pick atoms=4 for exactly that reason.
#
# In the fused schedule the tile is pinned to one token row, because RMSNorm
# reduces over the row and every element has to be reachable from one workgroup.
# ``atoms`` therefore sets the *block width* instead -- BLOCK = hidden/(8*atoms)
# -- and the tile, the flag count and the block count are all independent of it.
FUSED_ONESHOT_LADDER = {
    2: ((0, 1, 128, "peer", False),),
    4: ((0, 1, 32, "peer", False),),
    8: ((0, 1, 64, "peer", False),),
}


def fused_oneshot_ladder(world_size: int):
    """Rungs for *world_size* under ``fusion="rmsnorm"``. See FUSED_ONESHOT_LADDER."""
    return FUSED_ONESHOT_LADDER.get(
        int(world_size),
        ((0, 1, DEFAULT_GRID_CAP, DEFAULT_FANOUT, DEFAULT_SKIP_SELF),),
    )


def fused_block(hidden: int, atoms: int) -> int:
    """Threads per block for a fused build, or raise saying why *hidden* is out.

    One block covers one whole token row -- ``BLOCK * atoms * 8 == hidden`` --
    because the RMSNorm reduction spans the row and a row split across two
    blocks could only be joined with a grid-wide barrier.

    This schedule carries no codec, so the only alignment a block owes is whole
    waves for the reduction's shuffles. The quantized schedules need a stricter
    one; see ``quick_allreduce_fusions.quick_reduce_row_block``.
    """
    return row_block(hidden, per_thread=ATOM_ELEMS * int(atoms), align=WAVE)


def fused_hidden_supported(hidden: int, atoms: int = 1) -> bool:
    """Whether a fused build exists for this (hidden, atoms). For host-side gates."""
    return row_block_supported(hidden, per_thread=ATOM_ELEMS * int(atoms), align=WAVE)


def fused_block_options(hidden: int) -> tuple[tuple[int, int], ...]:
    """Every ``(block, atoms)`` a fused build can use at hidden dim, widest first."""
    return row_block_options(hidden, atoms_choices=SUPPORTED_ATOMS, align=WAVE)


def fused_atoms_for_block(hidden: int, block: int) -> int:
    """``atoms`` giving *block* threads at hidden dim, or raise naming the legal set."""
    block = int(block)
    opts = fused_block_options(hidden)
    for b, a in opts:
        if b == block:
            return a
    legal = ", ".join(str(b) for b, _ in opts) if opts else "none"
    raise ValueError(
        f"fused hidden={hidden} has no build at block={block}: one workgroup "
        f"covers one token row, so block is pinned to hidden/(8*atoms) for "
        f"atoms in {SUPPORTED_ATOMS} -- the legal blocks here are {legal}"
    )


# Inbox slots are indexed by ``colour & 1``. Two buffers is exactly enough to
# let one rank run a whole call ahead of another without overwriting a slot the
# straggler has not read.
PARITIES = 2
# 64 B handshake sector at the tail of each wire slot, as 16 i32 copies of the
# colour -- one ``dwordx4`` from each of 4 lanes.
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

# Measurement-only build variants. "sync" keeps the colour loop, the flag
# publish and the flag wait but touches no payload, so it times the floor this
# schedule cannot go below: launch path + one flag round trip + rank-arrival
# skew. It computes a wrong answer by construction and must never be dispatched
# to; ``OneShotAllReduce`` refuses to run a probe build through ``allreduce``.
PROBE_MODES = ("full", "sync")

# ``s_sleep`` interval for the flag spin, 0 to spin flat out.
DEFAULT_SPIN_SLEEP = 0

# Whether a rank pushes its own contribution through its own inbox. Keeping it
# costs a store, a load and a flag per tile in memory the rank already holds in
# registers, which is 1/N of each; dropping it specialises the binary per rank.
DEFAULT_SKIP_SELF = False


def _load_v4i32_at(rsrc, elem_off, policy):
    """One 16 B atom from a buffer descriptor, at an i32 element offset."""
    return fx.Vector(
        buffer_ops.buffer_load(
            rsrc, elem_off, vec_width=4, dtype=T.i32, cache_modifier=policy
        )
    )


def _load_i32_at(rsrc, elem_off, policy):
    val = buffer_ops.buffer_load(
        rsrc, elem_off, vec_width=1, dtype=T.i32, cache_modifier=policy
    )
    rocdl.s_waitcnt(vmcnt=0)
    return fx.Int32(val)


def make_one_shot_allreduce_kernel(
    *,
    world_size: int,
    atoms: int = DEFAULT_ATOMS,
    grid: int,
    inbox_memory: str = "uncached",
    fanout: str = DEFAULT_FANOUT,
    probe: str = "full",
    spin_sleep: int = DEFAULT_SPIN_SLEEP,
    skip_self: bool = False,
    rank: int | None = None,
    block: int | None = None,
    fusion: str = "none",
    hidden: int | None = None,
):
    if fusion not in FUSIONS:
        raise ValueError(f"fusion must be one of {FUSIONS}, got {fusion!r}")
    if fusion != "none" and hidden is None:
        raise ValueError(f"fusion={fusion!r} requires hidden")
    if fusion == "none" and hidden is not None:
        raise ValueError("hidden is only meaningful for a fused build")
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
    if probe not in PROBE_MODES:
        raise ValueError(f"probe must be one of {PROBE_MODES}, got {probe!r}")
    if not 0 <= int(spin_sleep) <= 0xFFFF:
        raise ValueError(f"spin_sleep must fit s_sleep's imm16, got {spin_sleep!r}")
    if grid < 1:
        raise ValueError(f"grid must be positive, got {grid}")

    policy = _INBOX_POLICY[inbox_memory]
    payload_policy = policy["payload"]
    flag_policy = policy["flag"]
    release_writeback = policy["writeback"]

    fused = fusion == "rmsnorm"
    if fused:
        row_width = fused_block(hidden, atoms)
        if block is not None and int(block) != row_width:
            raise ValueError(
                f"fused hidden={hidden} at atoms={atoms} needs block={row_width}, "
                f"got block={block}; the legal (block, atoms) pairs for this "
                f"width are {fused_block_options(hidden)}"
            )
        block = row_width
    else:
        block = DEFAULT_BLOCK if block is None else int(block)
        if block not in SUPPORTED_BLOCKS:
            raise ValueError(f"block must be one of {SUPPORTED_BLOCKS}, got {block!r}")
    # Per-wave partials for the block-wide sum of squares. The plain build has
    # no LDS at all.
    n_waves = block // WAVE
    lds_bytes = n_waves * 4 if fused else 0

    tile_bytes = block * atoms * ATOM_BYTES
    tile_i32 = tile_bytes // 4
    # Payload then the 64 B handshake sector.
    wire_tile_i32 = tile_i32 + FLAG_I32
    wire_tile_bytes = wire_tile_i32 * 4
    data_bytes = PARITIES * grid * world_size * wire_tile_bytes

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

    # LDS for the fused sum-of-squares. Carries the per-wave partial sums
    # only: the 1/hidden, the +eps and the rsqrt all happen afterwards in
    # registers, per thread, so no scale is ever broadcast through LDS.
    _RmsShared = make_wave_partials(n_waves) if fused else None

    # One signature for both modes. The fused-only arguments are present (and
    # passed as zeros) in a plain build rather than being appended to a second
    # kernel. The fused-only code is still elided by ``const_expr(fused)``,
    # only four unused kernargs remain in a plain build.
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
        res_in_ptr: Int64,
        res_out_ptr: Int64,
        w_ptr: Int64,
        eps: Float32,
    ):
        tid = fx.Int32(gpu.thread_id("x"))
        bid = fx.Int32(gpu.block_id("x"))
        lane_in_quad = tid % fx.Int32(4)

        hbm_layout = fx.make_layout(
            (num_tiles, atoms, block * ATOM_I32),
            (tile_i32, block * ATOM_I32, 1),
        )
        hbm_row_layout = fx.make_layout((1, block * ATOM_I32), (block * ATOM_I32, 1))
        hbm_copy_atom = fx.make_copy_atom(rocdl.BufferCopy128b(), fx.Int32)
        hbm_copy = fx.make_tiled_copy_tv(
            hbm_copy_atom,
            fx.make_layout((1, block), (1, 1)),
            fx.make_layout((1, ATOM_I32), (1, 1)),
        ).get_slice(tid)
        color_layout = fx.make_layout((grid,), (1,))

        peer_rsrc = buffer_ops.create_buffer_resource_from_addr(peer_ptrs)
        peers = [
            buffer_ops.buffer_load(peer_rsrc, i, vec_width=1, dtype=T.i64)
            for i in range(world_size)
        ]
        peer_vec = fx.Vector.from_elements(peers, dtype=fx.Int64)
        self_rsrc = buffer_ops.create_buffer_resource_from_addr(
            _to_sgpr_i64(peer_vec[rank])
        )

        hbm_i32_ptr = fx.PointerType.get(
            T.i32, address_space=fx.AddressSpace.Global, alignment=16
        )

        def _payload_tensor(ptr, records=None):
            # num_records_bytes is the live payload, so a partial last tile
            # reads 0 and stores are dropped rather than faulting.
            view = fx.make_view(fx.inttoptr(hbm_i32_ptr, ptr), hbm_layout)
            return rocdl.make_buffer_tensor(
                view,
                max_size=False,
                num_records_bytes=nbytes if records is None else records,
            )

        in_buf = _payload_tensor(inp_ptr)
        out_buf = _payload_tensor(out_ptr)
        color_rsrc = buffer_ops.create_buffer_resource_from_addr(colors_ptr)

        if const_expr(fused):
            # residual in/out are (M, hidden) bf16 exactly like the payload, so
            # they ride the payload layout and the same tiled copy.
            res_in_buf = _payload_tensor(res_in_ptr)
            res_out_buf = _payload_tensor(res_out_ptr)
            # The gain is a single (hidden,) row shared by every token: same
            # shape as one tile, so it reuses the layout with tile index 0 and
            # its own (one-tile) record bound.
            w_buf = _payload_tensor(w_ptr, records=fx.Int64(tile_bytes))

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

        def _hbm_atom_row(buf, tile, atom):
            return fx.make_view(
                fx.get_iter(fx.slice(buf, (tile, atom, None))), hbm_row_layout
            )

        def _load_color():
            off = fx.get_scalar(fx.crd2idx((bid,), color_layout))
            return fx.Int32(
                buffer_ops.buffer_load(color_rsrc, off, vec_width=1, dtype=T.i32)
            )

        def _store_color(color):
            off = fx.get_scalar(fx.crd2idx((bid,), color_layout))
            buffer_ops.buffer_store(color, color_rsrc, off)

        def _load_rows(buf, tile):
            """This thread's 16 B of each atom of *tile* of *buf*, as raw i32x4."""
            out = []
            for atom in range_constexpr(atoms):
                src = hbm_copy.partition_S(_hbm_atom_row(buf, tile, atom))
                frag = fx.make_fragment_like(src)
                fx.copy(hbm_copy_atom, src, frag)
                out.append(fx.Vector(frag.load()))
            return out

        def _load_tile(tile):
            return _load_rows(in_buf, tile)

        def _store_rows(buf, tile, vals):
            for atom in range_constexpr(atoms):
                dst = hbm_copy.partition_D(_hbm_atom_row(buf, tile, atom))
                frag = fx.make_fragment_like(dst)
                frag.store(vals[atom])
                fx.copy(hbm_copy_atom, frag, dst)

        def _store_tile(tile, vals):
            _store_rows(out_buf, tile, vals)

        def _fanout(parity, my_atoms):
            """Push this thread's atoms into every peer's slot for this rank.

            Thread ``t``'s data lands at the same offset in every destination,
            so it goes straight from registers -- no LDS staging.

            ``skip_self`` decides whether the fanout includes our own inbox.
            Keeping it makes the receive loop uniform over ``world_size``;
            dropping it removes 1/N of the stores, 1/N of the reduce's loads and
            1/N of the flags, at the cost of one kernel binary per rank.
            """
            # One asm block for the whole fanout: the stores must not have
            # their data VGPRs recycled before ``vmcnt`` retires them, and
            # LLVM cannot see that through inline asm. See
            # ``_store_v4i32_peer_multi``.
            _store_v4i32_peer_multi(
                [
                    (
                        peer_vec[peer]
                        + _i32_to_bytes(
                            _slot_i32(parity, rank)
                            + fx.Int32(atom * block * ATOM_I32)
                            + tid * fx.Int32(ATOM_I32)
                        ),
                        my_atoms[atom],
                    )
                    for peer, atom in fanout_pairs
                ],
                payload_policy,
            )

        def _publish(parity, color):
            """Drain the payload stores, then write *color* into every peer.

            ``vmcnt(0)`` retires this wave's stores; the barrier joins the other
            waves, whose ``vmcnt`` is separate. On a cacheable inbox retiring is
            not enough -- the lines can sit in this XCD's L2 -- so write back and
            wait for that before the flag goes out. Every workgroup issues its
            own writeback: L2 is per-XCD.
            """
            rocdl.s_waitcnt(vmcnt=0)
            gpu.barrier()
            if const_expr(release_writeback is not None):
                llvm.InlineAsmOp(None, [], release_writeback, "", has_side_effects=True)
                rocdl.s_waitcnt(vmcnt=0)
            # 4 lanes, one dwordx4 each -> the 64 B sector, unrolled over the
            # destinations. The peer index must be a trace-time constant: an
            # earlier version keyed it off the lane (``peer = tid // 4``, 4 lanes
            # per destination), which made ``peer_vec[peer]`` a *lane-varying*
            # extract from a 4xi64 vector. That lowers to a scratch round-trip,
            # and at ``atoms>1`` the register pressure made it land in the
            # payload: 8 B of peer pointer at 16 B stride over a 64 B span, once
            # per 256 B, in atom 0 of the highest-numbered rank's inbox. See the
            # ``SUPPORTED_ATOMS`` note. ``_fanout`` always unrolled; this is now
            # consistent with it.
            if tid < fx.Int32(4):
                elem = (
                    _slot_i32(parity, rank)
                    + fx.Int32(tile_i32)
                    + lane_in_quad * fx.Int32(4)
                )
                v4 = fx.Vector.from_elements([color, color, color, color], fx.Int32)
                _store_v4i32_peer_multi(
                    [
                        (peer_vec[peer] + _i32_to_bytes(elem), v4)
                        for peer in push_peers
                    ],
                    flag_policy,
                )

        def _wait(parity, color):
            """Spin until every rank's flag in our own inbox shows *color*.

            One spinner per source. The writeback-then-invalidate after the join
            is unconditional on purpose: if the flag is already present the loop
            body never runs, and an invalidate placed only inside it would leave
            the common case reading stale payload. Write back *before*
            invalidating or the output lines this block already wrote are
            discarded.
            """
            # Lane ``t`` watches one source. Without ``skip_self`` that is
            # source ``t``; with it our own flag is never published, so the
            # N-1 lanes step over our own index and the last lane sits out.
            # Computed before the guard rather than nested inside it, so the
            # remap is a flat ``scf.if`` yielding one value.
            spin_src = tid
            if const_expr(skip_self):
                if tid >= fx.Int32(self_rank):
                    spin_src = tid + fx.Int32(1)
            if tid < fx.Int32(len(push_peers)):
                elem = _slot_i32(parity, spin_src) + fx.Int32(tile_i32)
                flag_rsrc = buffer_ops.create_buffer_resource_from_addr(
                    peer_vec[rank] + _i32_to_bytes(elem)
                )
                # `sc0 sc1`, so each retry is fetched past L1 and L2 and no
                # fence is needed in the loop; the acquire below covers the
                # payload reads, once, after the join.
                current = _load_i32_at(flag_rsrc, fx.Int32(0), _RECV_POLICY)
                while current != color:
                    if const_expr(spin_sleep):
                        # Back off between polls. Each iteration is a load that
                        # bypasses both caches, and under arrival skew that runs
                        # for the whole skew window against the same line the
                        # peer is trying to write.
                        llvm.InlineAsmOp(
                            None, [], f"s_sleep {spin_sleep}", "", has_side_effects=True
                        )
                    current = _load_i32_at(flag_rsrc, fx.Int32(0), _RECV_POLICY)
            gpu.barrier()
            rocdl.s_waitcnt(vmcnt=0)
            if const_expr(release_writeback is not None):
                llvm.InlineAsmOp(None, [], release_writeback, "", has_side_effects=True)
                rocdl.s_waitcnt(vmcnt=0)
            _acquire_inbox()

        def _reduce_f32(parity, my_atoms):
            """Sum this thread's atom across all N contributions, in rank order.

            Rank order, not a rotated order: every rank must accumulate in the
            same sequence or the results differ in the last bit across ranks.
            ``cross_device_reduce`` makes the same promise for the same reason.
            Under ``skip_self`` our own contribution comes out of the registers
            rather than out of the inbox.

            Left unrounded: the plain path rounds once in ``_reduce``, and the
            fused path needs the fp32 accumulator for the norm.
            """
            outs = []
            for atom in range_constexpr(atoms):
                acc = None
                for src in range_constexpr(world_size):
                    if const_expr(src == self_rank):
                        v = atom_bf16_to_f32(my_atoms[atom])
                    else:
                        elem = (
                            _slot_i32(parity, fx.Int32(src))
                            + fx.Int32(atom * block * ATOM_I32)
                            + tid * fx.Int32(ATOM_I32)
                        )
                        v = atom_bf16_to_f32(
                            _load_v4i32_at(self_rsrc, elem, _RECV_POLICY)
                        )
                    acc = v if acc is None else acc + v
                outs.append(acc)
            return outs

        def _reduce(parity, my_atoms):
            """The plain path's result: one rounding, at the end of the sum."""
            return [atom_f32_to_bf16(a) for a in _reduce_f32(parity, my_atoms)]

        def _epilogue(tile, x_atoms, w_atoms, parity, sq_lds, my_atoms):
            """bf16 round-trip, residual add, RMSNorm.

            The arithmetic lives in ``quick_allreduce_fusions``, which the mesh
            and ring epilogues share; what stays here is the store placement.
            ``residual_out`` is written *before* the reduction, so it is in
            flight across the barrier rather than issued behind it -- it has no
            dependence on the norm.
            """
            accs = residual_add(_reduce_f32(parity, my_atoms), x_atoms)
            _store_rows(res_out_buf, tile, pack_bf16(accs))
            # One block covers one row, so there is a single row to reduce.
            rstd = rms_rstd([accs], eps, hidden, tid=tid, block=block, lds=sq_lds)[0]
            _store_tile(tile, scale_by_weight(accs, rstd, w_atoms))

        sq_lds = None
        if const_expr(fused):
            # The gain is one row shared by every token, so it is read once here
            # rather than once per token.
            w_atoms = _load_rows(w_buf, fx.Int32(0))
            if const_expr(n_waves > 1):
                # Allocated once, at kernel scope: SharedAllocator is static, so
                # an allocation reached from inside the tile loop would emit a
                # fresh LDS symbol per trace-time visit.
                sq_lds = fx.SharedAllocator().allocate(_RmsShared).peek().wave.ptr

        # Stride by the *launched* grid, not the compile-time cap: the host may
        # launch fewer blocks than ``grid``, and striding by the cap would leave
        # every tile above n_blocks unprocessed.
        n_block_tiles = (num_tiles - bid + n_blocks - fx.Int32(1)) // n_blocks
        color = _load_color()
        for i in range(fx.Int32(0), n_block_tiles, fx.Int32(1)):
            tile = bid + i * n_blocks
            parity = color & fx.Int32(1)
            # ``probe`` is a trace-time constant, so only one arm is emitted.
            # "sync" is measurement-only: the handshake with no payload
            # touched, so the wall time is the launch path plus the flag round
            # trip plus rank-arrival skew. Nothing this schedule does to the
            # data movement can go below it. Output is garbage.
            if const_expr(probe == "full"):
                my_atoms = _load_tile(tile)
                # Issued before the handshake on purpose: the residual is plain
                # HBM with no dependence on any peer, so this load retires
                # during the flag spin instead of after it.
                if const_expr(fused):
                    x_atoms = _load_rows(res_in_buf, tile)
                _fanout(parity, my_atoms)
            _publish(parity, color)
            _wait(parity, color)
            if const_expr(probe == "full"):
                if const_expr(fused):
                    _epilogue(tile, x_atoms, w_atoms, parity, sq_lds, my_atoms)
                else:
                    _store_tile(tile, _reduce(parity, my_atoms))
            color = color + fx.Int32(1)
            if color == fx.Int32(0):  # 0 is the unset sentinel
                color = fx.Int32(1)
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
        res_in_ptr: Int64,
        res_out_ptr: Int64,
        w_ptr: Int64,
        eps: Float32,
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
            res_in_ptr,
            res_out_ptr,
            w_ptr,
            eps,
            value_attrs={"rocdl.flat_work_group_size": flat_wg},
        ).launch(grid=(grid_x, 1, 1), block=(block, 1, 1), stream=stream)

    # Every compile-time knob that changes the emitted code has to be in the
    # symbol name, or two variants collide in the JIT cache. At ``atoms == 1``, 
    # the (peer, atom) product has one atom per peer, so both fanout orders 
    # unroll to the same store sequence.
    tag = f"ws{world_size}_a{atoms}_{inbox_memory}"
    if block != DEFAULT_BLOCK:
        tag += f"_b{block}"
    if atoms > 1:
        tag += f"_{fanout}"
    if fused:
        # hidden sets BLOCK and the tile, so it belongs in the key just as much
        # as atoms does.
        tag += f"_rms_h{hidden}"
    if probe != "full":
        tag += f"_{probe}"
    if spin_sleep:
        tag += f"_sl{spin_sleep}"
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
        "lds_bytes": lds_bytes,
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
        "probe": probe,
        "spin_sleep": spin_sleep,
        "skip_self": skip_self,
        "grid": grid,
        "block": block,
        "fusion": fusion,
        "hidden": hidden,
    }
