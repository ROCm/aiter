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

import math

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.typing import (
    Float32,
    Int32,
    Int64,
    ReductionOp,
    Stream,
    T,
    as_ir_value,
)

from . import buffer_ops

# The peer-store/load primitives, the cache-policy table and the inbox-memory
# taxonomy are shared with the quantized kernels verbatim.
from .quick_allreduce_shared import (
    _CM_SC0,
    _CM_SC1,
    _INBOX_POLICY,
    SUPPORTED_WORLDS,
    _acquire_inbox,
    _i32_to_bytes,
    _to_sgpr_i64,
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


BLOCK = 256
# 16 B per thread per atom -- one ``global_store_dwordx4``.
ATOM_BYTES = 16
ATOM_I32 = ATOM_BYTES // 4
DEFAULT_ATOMS = 1
# Atoms per thread per tile. More atoms means a bigger tile, hence fewer blocks
# and fewer flags for a given payload, at the cost of coarser load balance on
# the last partial tile. 1 is the decode default: at TP8/M=1 (14 KiB) it gives
# 4 blocks, which is already more parallelism than the payload needs.
#
# ``atoms>1`` used to be gated off here as incorrect. The diagnosis blamed the
# per-atom stride in ``_fanout`` / ``_reduce`` / ``hbm_layout``; that was wrong,
# those three always agreed. The real fault was a VMEM store hazard the fanout
# only exposes under the register pressure of several atoms -- see
# ``_store_v4i32_peer_multi`` below, which now carries the
# whole fanout in one asm block. Passes at TP2 and TP4 for 1, 2 and 4 including
# the run-ahead loop.
SUPPORTED_ATOMS = (1, 2, 4)
DEFAULT_GRID_CAP = 64

# Per-world-size tuning ladder: ``(min_bytes, atoms, grid_cap, fanout)`` rungs.
# The host builds one engine per rung and selects by payload size at launch.
# Created from a tuning sweep.
#
#   TP2  atoms=4  -- fattest tile, because at N=2 there is wire volume to spare
#                    and the payload runs far enough up to want fewer flags.
#   TP4  atoms=1  -- narrowest tile and a *smaller* grid cap; this window tops
#                    out at 96 KiB, where the schedule is flag-bound and blocks
#                    are worth more than tile width.
#   TP8  atoms=4  -- the fattest tile again, for the opposite reason: the window
#                    is only 48 KiB wide, the fanout is to 7 peers, and cutting
#                    the flag count matters more than the handful of blocks lost.
ONESHOT_LADDER = {
    2: ((0, 4, 128, "peer"),),
    4: ((0, 1, 32, "peer"),),
    8: ((0, 4, 64, "peer"),),
}


def oneshot_ladder(world_size: int):
    """Rungs for *world_size*, or a single default rung for an unlisted one."""
    return ONESHOT_LADDER.get(
        int(world_size), ((0, DEFAULT_ATOMS, DEFAULT_GRID_CAP, "peer"),)
    )


# Fused epilogues this factory can append to the reduce. 
FUSIONS = ("none", "rmsnorm")

# In the plain schedule ``atoms`` sets the *tile width*: tile = BLOCK*atoms*16 B,
# so a bigger atom count means fewer, fatter tiles and fewer flags. TP2 and TP8
# pick atoms=4 for exactly that reason.
#
# In the fused schedule the tile is pinned to one token row, because RMSNorm
# reduces over the row and every element has to be reachable from one workgroup.
# ``atoms`` therefore sets the *block width* instead -- BLOCK = hidden/(8*atoms)
# -- and the tile, the flag count and the block count are all independent of it.
FUSED_ONESHOT_LADDER = {
    2: ((0, 1, 128, "peer"),),
    4: ((0, 1, 32, "peer"),),
    8: ((0, 1, 64, "peer"),),
}


def fused_oneshot_ladder(world_size: int):
    """Rungs for *world_size* under ``fusion="rmsnorm"``. See FUSED_ONESHOT_LADDER."""
    return FUSED_ONESHOT_LADDER.get(
        int(world_size), ((0, 1, DEFAULT_GRID_CAP, "peer"),)
    )


# Wave width on gfx942/gfx950. The sum-of-squares butterfly is a full-wave shuffle_xor.
WAVE = 64


def fused_block(hidden: int, atoms: int) -> int:
    """Threads per block for a fused build, or raise saying why *hidden* is out.

    One block covers one whole token row -- ``BLOCK * atoms * 8 == hidden`` --
    because the RMSNorm reduction spans the row and a row split across two
    blocks could only be joined with a grid-wide barrier.
    """
    hidden = int(hidden)
    per_thread = 8 * int(atoms)  # 8 bf16 per 16 B atom
    if hidden <= 0 or hidden % per_thread != 0:
        raise ValueError(
            f"fused hidden must be a multiple of {per_thread} (8*atoms), got {hidden}"
        )
    block = hidden // per_thread
    if block % WAVE != 0:
        raise ValueError(
            f"fused hidden={hidden} with atoms={atoms} gives BLOCK={block}, "
            f"not a multiple of the {WAVE}-lane wave; needs hidden % {WAVE * per_thread} == 0"
        )
    if block > 1024:
        raise ValueError(
            f"fused hidden={hidden} with atoms={atoms} needs BLOCK={block} threads, "
            f"over the 1024 limit; raise atoms or split the row"
        )
    return block


def fused_hidden_supported(hidden: int, atoms: int = 1) -> bool:
    """Whether a fused build exists for this (hidden, atoms). For host-side gates."""
    try:
        fused_block(hidden, atoms)
    except ValueError:
        return False
    return True


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
# next, so a wave hands each destination a contiguous ``BLOCK * 16`` B run.
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


def _atom_bf16_to_f32(atom_i32):
    """16 B of bf16 (8 values) -> 8 f32. bf16 is the high half of f32, so this
    is a widening move, not a conversion -- exact, no rounding."""
    return fx.Vector(atom_i32).bitcast(fx.BFloat16).to(fx.Float32)


def _atom_f32_to_bf16(acc_f32):
    """8 f32 -> 16 B of bf16. One rounding, at the end of the reduction, which
    is what makes this bit-comparable with ``cross_device_reduce``'s fp32
    accumulate + single ``downcast``."""
    return acc_f32.to(fx.BFloat16).bitcast(fx.Int32)


def make_one_shot_allreduce_kernel(
    *,
    world_size: int,
    atoms: int = DEFAULT_ATOMS,
    grid: int,
    inbox_memory: str = "uncached",
    fanout: str = DEFAULT_FANOUT,
    probe: str = "full",
    spin_sleep: int = DEFAULT_SPIN_SLEEP,
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
    # The plain schedule keeps the shipped 256-thread block, so its emitted code
    # is untouched by this factory growing a fused mode. A fused build sizes the
    # block to the row instead; see ``fused_block``.
    block = fused_block(hidden, atoms) if fused else BLOCK
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

    # (peer, atom) iteration order for the fanout, unrolled at trace time.
    if fanout == "peer":
        fanout_pairs = [(p, a) for p in range(world_size) for a in range(atoms)]
    else:
        fanout_pairs = [(p, a) for a in range(atoms) for p in range(world_size)]

    # LDS for the fused sum-of-squares. Carries the per-wave partial sums
    # only: the 1/hidden, the +eps and the rsqrt all happen afterwards in
    # registers, per thread, so no scale is ever broadcast through LDS.
    if fused:
        @fx.struct
        class _RmsShared:
            wave_sq: fx.Array[fx.Float32, max(1, n_waves), 16]

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
            so it goes straight from registers -- no LDS staging. Includes the
            self-store: it is a local write into our own inbox and keeps the
            receive loop uniform over ``world_size``. Dropping it is a tuning
            lever, not a correctness one.
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
                        for peer in range(world_size)
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
            if tid < fx.Int32(world_size):
                elem = _slot_i32(parity, tid) + fx.Int32(tile_i32)
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

        def _reduce_f32(parity):
            """Sum this thread's atom across all N inbox copies, in rank order.

            Rank order, not a rotated order: every rank must accumulate in the
            same sequence or the results differ in the last bit across ranks.
            ``cross_device_reduce`` makes the same promise for the same reason.

            Returns fp32 vectors; the plain path rounds them once in
            ``_reduce``, the fused path needs them unrounded for the norm.
            """
            outs = []
            for atom in range_constexpr(atoms):
                acc = None
                for src in range_constexpr(world_size):
                    elem = (
                        _slot_i32(parity, fx.Int32(src))
                        + fx.Int32(atom * block * ATOM_I32)
                        + tid * fx.Int32(ATOM_I32)
                    )
                    v = _atom_bf16_to_f32(_load_v4i32_at(self_rsrc, elem, _RECV_POLICY))
                    acc = v if acc is None else acc + v
                outs.append(acc)
            return outs

        def _reduce(parity):
            return [_atom_f32_to_bf16(a) for a in _reduce_f32(parity)]

        def _block_sum_sq(accs):
            """Block-wide ``sum(acc^2)`` over the whole row, one value per thread.

            Three levels: per-thread over its ``8*atoms`` channels, a full-wave
            ``shuffle_xor`` butterfly, then the cross-wave combine through LDS.

            What lands in LDS is a partial sum -- every thread
            reads the ``n_waves`` partials back and adds them itself, and
            computes its own ``rsqrt`` afterwards. That redundancy is bought
            deliberately: it removes the broadcast, so this costs only one barrier.
            """
            local = None
            for atom in range_constexpr(atoms):
                sq = accs[atom] * accs[atom]
                part = fx.Float32(sq.reduce(ReductionOp.ADD))
                local = part if local is None else local + part

            for sh in range_constexpr(int(math.log2(WAVE))):
                local = local + local.shuffle_xor(WAVE // (2 << sh), WAVE)

            if const_expr(n_waves == 1):
                # One wave covers the row: the butterfly already finished it and
                # LDS would only add a barrier.
                return local

            sq_lds = fx.SharedAllocator().allocate(_RmsShared).peek().wave_sq.ptr
            wid = tid // fx.Int32(WAVE)
            if tid % fx.Int32(WAVE) == fx.Int32(0):
                sq_lds[wid] = local
            gpu.barrier()
            total = None
            for w in range_constexpr(n_waves):
                v = fx.Float32(sq_lds[fx.Int32(w)])
                total = v if total is None else total + v
            return total

        def _epilogue(tile, x_atoms, w_atoms, parity):
            """bf16 round-trip, residual add, RMSNorm.
            """
            accs = _reduce_f32(parity)
            res_out = []
            for atom in range_constexpr(atoms):
                # Round the all-reduce result to bf16 and back before adding the
                # residual. 
                # TODO: Keeping the extra f32 mantissa bits would be more accurate,
                # diverging 1 ULP per layer from the kernel this replaces.
                a = accs[atom].to(fx.BFloat16).to(fx.Float32)
                a = a + fx.Vector(x_atoms[atom]).bitcast(fx.BFloat16).to(fx.Float32)
                accs[atom] = a
                res_out.append(_atom_f32_to_bf16(a))
            _store_rows(res_out_buf, tile, res_out)

            # rsqrt on the fp32 accumulator, not on the bf16 just stored.
            rstd = fmath.rsqrt(_block_sum_sq(accs) * (1.0 / hidden) + eps)
            outs = []
            for atom in range_constexpr(atoms):
                w = fx.Vector(w_atoms[atom]).bitcast(fx.BFloat16).to(fx.Float32)
                outs.append(_atom_f32_to_bf16(accs[atom] * rstd * w))
            _store_tile(tile, outs)

        # The gain is one row shared by every token, so it is read once here
        # rather than once per token.
        if const_expr(fused):
            w_atoms = _load_rows(w_buf, fx.Int32(0))

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
                    _epilogue(tile, x_atoms, w_atoms, parity)
                else:
                    _store_tile(tile, _reduce(parity))
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
    # symbol name, or two variants collide in the JIT cache.
    tag = f"ws{world_size}_a{atoms}_{inbox_memory}_{fanout}"
    if fused:
        # hidden sets BLOCK and the tile, so it belongs in the key just as much
        # as atoms does.
        tag += f"_rms_h{hidden}"
    if probe != "full":
        tag += f"_{probe}"
    if spin_sleep:
        tag += f"_sl{spin_sleep}"
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
        "grid": grid,
        "block": block,
        "fusion": fusion,
        "hidden": hidden,
    }
