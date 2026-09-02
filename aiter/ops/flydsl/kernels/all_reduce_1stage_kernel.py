# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""gfx942/gfx950 TP∈{2,4,8} exact one-shot (1-stage) all-reduce.

Decode-regime kernel: bf16 in, fp32 accumulate, bf16 out, no codec. One
communication round and **no grid-wide barrier** -- each rank pushes its whole
tile into every peer's inbox, publishes a colour flag, waits for the N flags,
then reduces N copies out of its own inbox.

Against ``cross_device_reduce_1stage``, which barriers, reads every peer's
input, reduces, and barriers again, this trades wire volume for round trips:
(N-1)*S pushed rather than (N-1)*S read, but ~2 serialized fabric traversals
rather than ~6. That is the right trade only while the payload is small, hence
``MAX_PAYLOAD_BYTES`` on the host -- above it the two-shot's 2(N-1)/N wire
volume wins and this kernel should not be selected.

Why there is no LDS here and 9 KiB of it in ``qr_int4_kernel``: quantization
redistributes data across threads, so a thread's packed output belongs in a
different peer's packet than the one it loaded from. Without a codec, thread
``t``'s 16 B lands at the same offset in every destination, so it can be pushed
straight from registers.

See docs/all_reduce_1stage.md.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm
from flydsl.expr import gpu, range_constexpr, rocdl
from flydsl.expr.typing import Int32, Int64, Stream, T

from . import buffer_ops

# The peer-store/load primitives, the cache-policy table and the inbox-memory
# taxonomy are shared with the quantized kernels verbatim. Private there by
# convention, not by intent -- see the same import block in
# ``qr_int4_ring_kernel``.
from .qr_int4_kernel import (
    _CM_SC0,
    _CM_SC1,
    _INBOX_POLICY,
    SUPPORTED_WORLDS,
    _i32_to_bytes,
    _invalidate_l1,
    _store_v4i32_peer,
    _to_sgpr_i64,
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
# ONLY 1 IS CORRECT TODAY. atoms=4 was measured wrong at TP4 on every shape
# tested (m=1..16): the last rank's output diverges from the others and SQNR is
# -inf, i.e. the reduction is reading something that is not the payload. atoms=1
# passes the same test at TP2 and TP4 including the run-ahead loop, so the fault
# is in the multi-atom addressing -- the per-atom stride is used in three places
# (``_fanout``, ``_reduce`` and the ``hbm_layout`` slice) and they are not yet
# proven to agree. Listed as (1,) rather than left open so the broken values are
# rejected at construction instead of silently producing garbage.
SUPPORTED_ATOMS = (1,)
DEFAULT_GRID_CAP = 64

# Inbox slots are indexed by ``colour & 1``. Two buffers is exactly enough to
# let one rank run a whole call ahead of another without overwriting a slot the
# straggler has not read; see the run-ahead argument in
# docs/all_reduce_1stage.md §6.2. It is *not* enough to be sloppy about the
# wait: the proof leans on every rank's publish for call k+1 happening after its
# own read of call k, which is only true because a kernel launch is ordered
# against the previous one on the same stream.
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
# next, so a wave hands each destination a contiguous ``BLOCK * 16`` B run. This
# is what PCIe wants -- interleaving destinations costs ~1.5x against >=256 B
# runs (36.25 vs 54.03 GB/s measured on MI350P, see docs/qr_int4_mi350p.md).
#
# "atom": consecutive stores walk the peers of one atom. On xGMI the native
# packet is 64 B and there is no per-destination run-length benefit to collect,
# so spreading across links sooner can start more of them in parallel.
#
# Resolved on the host side of the factory because a binding made inside an
# ``if`` does not survive FlyDSL's trace.
FANOUT_ORDERS = ("peer", "atom")
DEFAULT_FANOUT = "peer"


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


def make_all_reduce_1stage_kernel(
    *,
    world_size: int,
    atoms: int = DEFAULT_ATOMS,
    grid: int,
    inbox_memory: str = "uncached",
    fanout: str = DEFAULT_FANOUT,
):
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
    if grid < 1:
        raise ValueError(f"grid must be positive, got {grid}")

    policy = _INBOX_POLICY[inbox_memory]
    payload_policy = policy["payload"]
    flag_policy = policy["flag"]
    release_writeback = policy["writeback"]

    tile_bytes = BLOCK * atoms * ATOM_BYTES
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

    @flyc.kernel(known_block_size=[BLOCK, 1, 1])
    def all_reduce_1stage(
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
        lane_in_quad = tid % fx.Int32(4)

        hbm_layout = fx.make_layout(
            (num_tiles, atoms, BLOCK * ATOM_I32),
            (tile_i32, BLOCK * ATOM_I32, 1),
        )
        hbm_row_layout = fx.make_layout((1, BLOCK * ATOM_I32), (BLOCK * ATOM_I32, 1))
        hbm_copy_atom = fx.make_copy_atom(rocdl.BufferCopy128b(), fx.Int32)
        hbm_copy = fx.make_tiled_copy_tv(
            hbm_copy_atom,
            fx.make_layout((1, BLOCK), (1, 1)),
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

        def _payload_tensor(ptr):
            # num_records_bytes is the live payload, so a partial last tile
            # reads 0 and stores are dropped rather than faulting.
            view = fx.make_view(fx.inttoptr(hbm_i32_ptr, ptr), hbm_layout)
            return rocdl.make_buffer_tensor(
                view, max_size=False, num_records_bytes=nbytes
            )

        in_buf = _payload_tensor(inp_ptr)
        out_buf = _payload_tensor(out_ptr)
        color_rsrc = buffer_ops.create_buffer_resource_from_addr(colors_ptr)

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

        def _load_tile(tile):
            """This thread's 16 B of each atom of *tile*, as raw i32x4."""
            out = []
            for atom in range_constexpr(atoms):
                src = hbm_copy.partition_S(_hbm_atom_row(in_buf, tile, atom))
                frag = fx.make_fragment_like(src)
                fx.copy(hbm_copy_atom, src, frag)
                out.append(fx.Vector(frag.load()))
            return out

        def _store_tile(tile, vals):
            for atom in range_constexpr(atoms):
                dst = hbm_copy.partition_D(_hbm_atom_row(out_buf, tile, atom))
                frag = fx.make_fragment_like(dst)
                frag.store(vals[atom])
                fx.copy(hbm_copy_atom, frag, dst)

        def _fanout(parity, my_atoms):
            """Push this thread's atoms into every peer's slot for this rank.

            Thread ``t``'s data lands at the same offset in every destination,
            so it goes straight from registers -- no LDS staging. Includes the
            self-store: it is a local write into our own inbox and keeps the
            receive loop uniform over ``world_size``. Dropping it is a tuning
            lever, not a correctness one.
            """
            for peer, atom in fanout_pairs:
                elem = (
                    _slot_i32(parity, rank)
                    + fx.Int32(atom * BLOCK * ATOM_I32)
                    + tid * fx.Int32(ATOM_I32)
                )
                _store_v4i32_peer(
                    peer_vec[peer] + _i32_to_bytes(elem),
                    my_atoms[atom],
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
            if release_writeback is not None:
                llvm.InlineAsmOp(None, [], release_writeback, "", has_side_effects=True)
                rocdl.s_waitcnt(vmcnt=0)
            limit = fx.Int32(world_size * 4)
            safe = (tid < limit).select(tid, fx.Int32(0))
            if tid < limit:
                # 4 lanes per destination, one dwordx4 each -> the 64 B sector.
                peer = safe // fx.Int32(4)
                elem = (
                    _slot_i32(parity, rank)
                    + fx.Int32(tile_i32)
                    + lane_in_quad * fx.Int32(4)
                )
                v4 = fx.Vector.from_elements([color, color, color, color], fx.Int32)
                _store_v4i32_peer(
                    peer_vec[peer] + _i32_to_bytes(elem), v4, flag_policy
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
                current = _load_i32_at(flag_rsrc, fx.Int32(0), _CM_SC1)
                while current != color:
                    current = _load_i32_at(flag_rsrc, fx.Int32(0), _CM_SC1)
                    _invalidate_l1()
            gpu.barrier()
            rocdl.s_waitcnt(vmcnt=0)
            if release_writeback is not None:
                llvm.InlineAsmOp(None, [], release_writeback, "", has_side_effects=True)
                rocdl.s_waitcnt(vmcnt=0)
            _invalidate_l1()

        def _reduce(parity):
            """Sum this thread's atom across all N inbox copies, in rank order.

            Rank order, not a rotated order: every rank must accumulate in the
            same sequence or the results differ in the last bit across ranks.
            ``cross_device_reduce`` makes the same promise for the same reason.
            """
            outs = []
            for atom in range_constexpr(atoms):
                acc = None
                for src in range_constexpr(world_size):
                    elem = (
                        _slot_i32(parity, fx.Int32(src))
                        + fx.Int32(atom * BLOCK * ATOM_I32)
                        + tid * fx.Int32(ATOM_I32)
                    )
                    v = _atom_bf16_to_f32(
                        _load_v4i32_at(self_rsrc, elem, _RECV_POLICY)
                    )
                    acc = v if acc is None else acc + v
                outs.append(_atom_f32_to_bf16(acc))
            return outs

        # Stride by the *launched* grid, not the compile-time cap: the host may
        # launch fewer blocks than ``grid``, and striding by the cap would leave
        # every tile above n_blocks unprocessed.
        n_block_tiles = (num_tiles - bid + n_blocks - fx.Int32(1)) // n_blocks
        color = _load_color()
        for i in range(fx.Int32(0), n_block_tiles, fx.Int32(1)):
            tile = bid + i * n_blocks
            parity = color & fx.Int32(1)
            my_atoms = _load_tile(tile)
            _fanout(parity, my_atoms)
            _publish(parity, color)
            _wait(parity, color)
            _store_tile(tile, _reduce(parity))
            color = color + fx.Int32(1)
            if color == fx.Int32(0):  # 0 is the unset sentinel
                color = fx.Int32(1)
        if tid == 0:
            _store_color(color)
        gpu.barrier()

    flat_wg = f"{BLOCK},{BLOCK}"

    @flyc.jit
    def launch_all_reduce_1stage(
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
        all_reduce_1stage(
            rank,
            nbytes,
            num_tiles,
            inp_ptr,
            out_ptr,
            peer_ptrs,
            colors_ptr,
            grid_x,
            value_attrs={"rocdl.flat_work_group_size": flat_wg},
        ).launch(grid=(grid_x, 1, 1), block=(BLOCK, 1, 1), stream=stream)

    # Every compile-time knob that changes the emitted code has to be in the
    # symbol name, or two variants collide in the JIT cache.
    tag = f"ws{world_size}_a{atoms}_{inbox_memory}_{fanout}"
    launch_all_reduce_1stage.func.__name__ = f"launch_all_reduce_1stage_{tag}"
    try:
        all_reduce_1stage.func.__name__ = f"all_reduce_1stage_{tag}"
    except AttributeError:
        pass
    return {
        "launch": launch_all_reduce_1stage,
        "flags_bytes": 0,
        "data_bytes": data_bytes,
        "lds_bytes": 0,
        "tile_bytes": tile_bytes,
        "wire_tile_bytes": wire_tile_bytes,
        # Shims for ``qr_int4._StEngine``, which is reused verbatim for the IPC
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
        "grid": grid,
        "block": BLOCK,
    }
