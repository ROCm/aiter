# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL port of ``op_tests/multigpu_tests/peer_write_bw.cpp``, as a parity check.

WHY
===
The MI350P investigation
(``op_tests/dump_data/docs/mi350p_peer_write_primitive_2026-09-07.md``) is stuck
on one number: a standalone **HIP** kernel implementing the two-shot wire
protocol runs at 28.6 GB/s per rank on this host, while the **FlyDSL**
``qr_int4`` kernel's communication path -- codec and tiles ablated away, so it
is doing nothing but peer writes and a handshake -- runs at 14.2 GB/s at matched
cadence, block count and LDS sourcing. Everything measurable outside the kernel
is healthy: the fabric, the allocation type, the write granularity, the latency,
the publish drain, the receive path, the clocks, the grid policy, and
TransferBench's own roofline.

The two implementations differ in exactly one thing that has not been isolated:
one is compiled by hipcc, the other by FlyDSL. This file removes that variable.
It writes the *same* microbenchmark in FlyDSL -- same access pattern, same
instruction, same allocation flags, same geometry, same timing method -- and
compares against the C++ numbers measured on 2026-09-07.

    FlyDSL matches C++   -> FlyDSL codegen is exonerated for this pattern, and
                            the 2x lives in something structural about the
                            qr_int4 kernel rather than in how FlyDSL compiles a
                            peer-write loop.
    FlyDSL is ~2x slower -> the 2x is FlyDSL codegen, and this file is a small,
                            fast reproducer to hand to whoever fixes it.

Either answer closes a question that has stayed open through three rounds of
measurement, which is why it is worth a test rather than a script.

WHAT IS HELD IDENTICAL TO THE C++
=================================
* **The store.** ``_store_v4i32_peer`` is imported from ``qr_int4_kernel`` rather
  than reimplemented, so this exercises the production code path verbatim: the
  same ``global_store_dwordx4 $0, $1, off nt`` inline asm through the same
  ``IntToPtrOp``. If that helper changes, this test changes with it, which is
  the point.
* **The pattern.** 16 B per lane, four lanes per 64 B sector, whole sectors
  round-robined across peers. ``chunk`` is the contiguous run handed to one
  destination before switching -- 64 B is ``_fanout_nt``'s "sector" order, 512 B
  its "peer" order.
* **The allocation.** ``UncachedIpcHeap.alloc`` with the same
  ``hipExtMallocWithFlags`` modes ``QRInt4`` uses (0x0 coarse, 0x1 fine-grained,
  0x3 uncached).
* **The geometry.** 32 MiB total per measured iteration, 512 blocks x 256
  threads, blocks owning a contiguous partition of the store index space.
* **The peer-pointer table.** Loaded once into a ``fx.Vector`` and indexed by a
  runtime lane value, exactly as ``_fanout_nt`` does. This is deliberately *not*
  the C++ formulation: the C++ needed a hand-pinned select chain because clang
  sank the indexed load into the loop, and whether FlyDSL has the same problem
  is one of the things under test.

WHAT IS NOT PORTED
==================
The cross-process pieces: ``--ipc`` (the C++ found IPC and same-process peer
mapping identical to 0.2%) and ``--handshake`` (the peer spin-wait, which the
C++ found worth only 10-15% on top of the publish). Both need a rank per GPU;
this runs in one process against peer-mapped buffers, which is the C++ ``p2p``
path.

RUNNING
=======
    pytest -q op_tests/flydsl_tests/test_flydsl_peer_write_bw.py
    python3 op_tests/flydsl_tests/test_flydsl_peer_write_bw.py   # markdown sweep

Needs >= 2 GPUs; the 3-peer cases need 4. Pytest skips what it cannot run.
"""

# NB: no `from __future__ import annotations`. `@fx.struct` resolves its field
# annotations at class-creation time, and stringised annotations break it --
# `fx.Array[fx.Int32, lds_i32]` becomes an unresolvable string and the struct
# layout fails with "does not implement the Storable protocol".

import ctypes
import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import pytest
import torch

pytest.importorskip("flydsl")

import flydsl.compiler as flyc  # noqa: E402
import flydsl.expr as fx  # noqa: E402
from flydsl._mlir.dialects import llvm  # noqa: E402
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl  # noqa: E402
from flydsl.expr.typing import Int32, Int64, Stream, T  # noqa: E402

from aiter.jit.utils.chip_info import get_gfx_runtime  # noqa: E402
from aiter.ops.flydsl.kernels import buffer_ops  # noqa: E402
from aiter.ops.flydsl.kernels.qr_int4_ipc import UncachedIpcHeap  # noqa: E402
from aiter.ops.flydsl.kernels.qr_int4_kernel import _store_v4i32_peer  # noqa: E402

_ARCHS = ("gfx942", "gfx950")

BLOCK = 256
QUAD_LANES = 4
V4_BYTES = 16
# 32 MiB moved per measured iteration -- the total the 2026-08-31 table used and
# the C++ default, so the GB/s numbers are directly comparable.
TOTAL_BYTES = 32 << 20
# Room past the payload for one 64 B publish flag per block per peer.
FLAG_TAIL_BYTES = 8 * 1024 * 64
DEFAULT_BLOCKS = 512
ITERS = 30
WARMUP = 5

ALLOC_FLAGS = {
    "coarse": UncachedIpcHeap._HIP_DEVICE_MALLOC_DEFAULT,
    "fine": UncachedIpcHeap._HIP_DEVICE_MALLOC_FINEGRAINED,
    "uncached": UncachedIpcHeap._HIP_DEVICE_MALLOC_UNCACHED,
}

# Measured by op_tests/multigpu_tests/peer_write_bw.cpp on 2026-09-07, MI350P,
# HIP_VISIBLE_DEVICES=0,1,2,3, one writer, `nt` payload, 512 blocks, 32 MiB.
# Raw data: op_tests/dump_data/mi350p_peer_write_bw_2026-09-07.csv.
# Keyed (alloc, peers, chunk_B, publish_B) -> GB/s.
CPP_REFERENCE = {
    # allocation type x peer count, 64 B destination interleave
    ("coarse", 1, 64, 0): 51.73,
    ("coarse", 2, 64, 0): 52.33,
    ("coarse", 3, 64, 0): 35.77,
    ("fine", 1, 64, 0): 51.66,
    ("fine", 2, 64, 0): 52.34,
    ("fine", 3, 64, 0): 35.35,
    ("uncached", 1, 64, 0): 55.09,
    ("uncached", 2, 64, 0): 4.17,
    ("uncached", 3, 64, 0): 1.44,
    # per-destination run length at 3 peers
    ("fine", 3, 128, 0): 51.45,
    ("fine", 3, 512, 0): 52.90,
    ("fine", 3, 4096, 0): 52.18,
    ("uncached", 3, 128, 0): 1.87,
    ("uncached", 3, 512, 0): 2.30,
    ("uncached", 3, 4096, 0): 2.22,
    # publish cadence (drain + buffer_wbl2 + flag), fine-grained, 512 B runs
    ("fine", 3, 512, 4096): 9.21,
    ("fine", 3, 512, 6912): 13.09,
    ("fine", 3, 512, 16384): 23.38,
    ("fine", 3, 512, 55296): 32.33,
}

# How far FlyDSL may sit from the C++ number before the test fails.
#
# Two gates, because the two regimes have very different stability.
#
# The C++ is reproducible to +-0.1% *within a process* (five repeats of
# uncached/3-peer/512 B: 1.95 1.95 1.95 1.95 1.95). Across processes the
# cacheable modes hold to ~1%, but the uncached multi-peer cells move by up to
# 15% -- 1.95 here against the 2.30 recorded in the table below. Those cells sit
# at ~2 GB/s in the per-destination serialization regime, where throughput is
# set by where the pages physically landed, so a fresh allocation is a different
# measurement. That is a property of the regime, not of either implementation.
#
# So: tight against the cacheable modes, loose against uncached multi-peer. The
# loose gate applies even when a live C++ run is available (``_cpp_binary``),
# because that still runs in its own process against its own allocation -- a
# live reference removes reference *rot*, not placement sensitivity. Either way
# a 2x miss, the thing this file exists to detect, fails by a wide margin.
TOLERANCE = 0.15
TOLERANCE_UNSTABLE = 0.35


def _unstable(alloc: str, npeers: int) -> bool:
    """Cells whose absolute value depends on physical page placement."""
    return alloc == "uncached" and npeers >= 2


# ---------------------------------------------------------------------------
# Live C++ comparison, when the binary is around
# ---------------------------------------------------------------------------
#
# Preferred over the recorded table: it puts both implementations in the same
# session against freshly allocated buffers, which removes the placement
# sensitivity above and keeps the test from rotting as the reference ages.


def _cpp_binary():
    """Path to the built ``peer_write_bw`` binary, or None."""
    for cand in (
        os.environ.get("PEER_WRITE_BW_BIN"),
        "/tmp/peer_write_bw",
        os.path.join(_REPO_ROOT, "op_tests", "multigpu_tests", "peer_write_bw"),
    ):
        if cand and os.access(cand, os.X_OK):
            return cand
    return None


def _cpp_measure(binary, alloc, npeers, chunk, publish) -> float | None:
    """Run one C++ configuration and return GB/s, or None if it failed."""
    import subprocess

    cmd = [
        binary, "--alloc", alloc, "--peers", str(npeers), "--chunk", str(chunk),
        "--publish", str(publish), "--blocks", str(DEFAULT_BLOCKS),
        "--total", str(TOTAL_BYTES), "--iters", str(ITERS),
        "--warmup", str(WARMUP), "--no-header",
    ]
    try:
        out = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300
        ).stdout.strip().splitlines()
    except (OSError, subprocess.SubprocessError):
        return None
    for line in out:
        f = line.split(",")
        # mode,map,alloc,policy,dst,writers,peers,chunk,publish,wbl2,total,
        # blocks,threads,us,gbps_per_writer,...
        if len(f) > 14 and f[0] == "write":
            return float(f[14])
    return None


# ---------------------------------------------------------------------------
# HIP peer access
# ---------------------------------------------------------------------------


def _hip():
    return UncachedIpcHeap._load_hip()


def _enable_peer_access(src_dev: int, peer_devs) -> None:
    """Map every peer's memory into *src_dev*'s address space.

    Same-process equivalent of the IPC import ``QRInt4`` performs. The C++
    measured the two mappings as identical to 0.2%, so this is the cheaper
    formulation of the same thing.
    """
    hip = _hip()
    torch.cuda.set_device(src_dev)
    for d in peer_devs:
        if d == src_dev:
            continue
        err = hip.hipDeviceEnablePeerAccess(ctypes.c_int(d), ctypes.c_uint(0))
        # 704 == hipErrorPeerAccessAlreadyEnabled, which is not a failure.
        if err not in (0, 704):
            raise RuntimeError(f"hipDeviceEnablePeerAccess({d}) failed: {err}")


# ---------------------------------------------------------------------------
# The kernel
# ---------------------------------------------------------------------------


def make_peer_write_kernel(
    *,
    npeers: int,
    publish_bytes: int,
    wbl2: bool = True,
    policy: str = "nt",
    from_lds: bool = False,
    active_threads: int = BLOCK,
    lds_i32: int = 0,
    subtiles: int = 1,
    slot_pitch: int = 0,
    runtime_subtiles: bool = False,
):
    """Build the fanout kernel for a fixed peer count and publish cadence.

    ``npeers`` and ``publish_bytes`` are baked in rather than passed at runtime,
    mirroring the C++ where the peer count is a template parameter: it lets the
    round-robin ``% npeers`` strength-reduce instead of expanding to a runtime
    reciprocal, and it keeps the publish out of the store loop as a branch.
    """
    pub16 = publish_bytes // V4_BYTES
    if from_lds and not lds_i32:
        raise ValueError("from_lds needs lds_i32 > 0")
    # Sub-tiles per publish group, each followed by a workgroup barrier -- the
    # ST loop in qr_int4_kernel.py, which fans out `super_tile` tiles with an
    # `s_waitcnt lgkmcnt(0)` + `gpu.barrier()` between each and one publish at
    # the end. ATT says that kernel spends 45.9% of its stall in `s_barrier`
    # against this primitive's 11.6%, so the barriers are the axis to test.
    if subtiles > 1 and pub16 % subtiles:
        raise ValueError(f"publish_bytes must divide by subtiles={subtiles}")
    sub16 = pub16 // subtiles if subtiles > 1 else 0
    # Destination pitch per block, in 16 B units. 0 keeps the dense layout, where
    # a block's bytes sit immediately after the previous block's. qr_int4 instead
    # gives every block a slot in a PHASES x grid x world_size x wire_tile inbox
    # -- 172 MiB at M=4096 -- and fills 18,432 B of each 73,984 B slot, so its
    # peer writes are 25% dense in the address space where this primitive's are
    # 100%. This is the last structural difference between the two that has not
    # been measured.
    pitch16 = slot_pitch // V4_BYTES
    # Power-of-two slot count so the LDS index is a mask, not a modulo. The
    # allocation may be larger (9216 B to match qr_int4's group segment); only
    # the first LDS_SLOTS * 16 B are addressed.
    lds_slots = 1 << ((lds_i32 // 4).bit_length() - 1) if lds_i32 else 0

    if lds_i32:

        @fx.struct
        class PayloadStorage:
            buf: fx.Array[fx.Int32, lds_i32]


    @flyc.kernel(known_block_size=[BLOCK, 1, 1])
    def peer_write(
        peer_ptrs: Int64,
        total16: Int32,
        run_shift: Int32,
        flag_off16: Int32,
        n_blocks: Int32,
    ):
        tid = fx.Int32(gpu.thread_id("x"))
        bid = fx.Int32(gpu.block_id("x"))

        if const_expr(lds_i32 > 0):
            lds = fx.SharedAllocator().allocate(PayloadStorage).peek()
            smem_ptr = lds.buf.ptr

        # Peer pointer table, loaded once into registers and indexed by a
        # runtime lane value. Identical to _fanout_nt's `peer_vec[peer]`.
        peer_rsrc = buffer_ops.create_buffer_resource_from_addr(peer_ptrs)
        peers = [
            buffer_ops.buffer_load(peer_rsrc, i, vec_width=1, dtype=T.i64)
            for i in range(npeers)
        ]
        peer_vec = fx.Vector.from_elements(peers, dtype=fx.Int64)

        run_mask = (fx.Int32(1) << run_shift) - fx.Int32(1)

        # Blocks own a contiguous partition rather than grid-striding, because a
        # publish is per-block and has to sit between two runs of that block's
        # own payload -- and because that is how qr_int4 is structured, one
        # block per super-tile.
        spb = (total16 + n_blocks - fx.Int32(1)) // n_blocks
        begin = bid * spb
        cap = begin + spb
        end = (cap < total16).select(cap, total16)

        # `const_expr` because the AST rewriter turns a bare `if` in a kernel
        # body into `scf.if`, and a value bound inside one does not escape it
        # (NameError at trace time). With const_expr only the taken branch is
        # traced, so the binding is an ordinary Python one.
        if const_expr(pub16 != 0):
            group = fx.Int32(pub16)
        else:
            group = spb
        ngroups = (spb + group - fx.Int32(1)) // group

        def _one_store(s):
            if const_expr(pitch16 > 0):
                # Block-local index, then place the block at its own slot.
                loc = s - begin
                run = loc >> run_shift
                peer = run % fx.Int32(npeers)
                off16 = (
                    bid * fx.Int32(pitch16)
                    + ((run // fx.Int32(npeers)) << run_shift)
                    + (loc & run_mask)
                )
            else:
                run = s >> run_shift
                peer = run % fx.Int32(npeers)
                off16 = ((run // fx.Int32(npeers)) << run_shift) + (s & run_mask)
            if const_expr(from_lds):
                # One ds_read_b128 feeding each store, as _fanout_nt does. The
                # C++ found this free; whether FlyDSL hoists it the same way is
                # the point of the rung.
                slot = (s & fx.Int32(lds_slots - 1)) * fx.Int32(4)
                v4 = fx.ptr_load(
                    smem_ptr + slot,
                    result_type=fx.Vector.make_type(4, fx.Int32),
                )
            else:
                # Synthesised, so the measurement is the write path alone.
                v4 = fx.Vector.from_elements(
                    [
                        s,
                        s ^ fx.Int32(0x5A5A5A5A),
                        s + fx.Int32(7),
                        s ^ fx.Int32(-1),
                    ],
                    dtype=fx.Int32,
                )
            dest = peer_vec[peer]
            _store_v4i32_peer(dest + fx.Int64(off16) * fx.Int64(V4_BYTES), v4, policy)

        def _store_range(g0, g1):
            # `active_threads < BLOCK` reproduces the idle lanes in the real
            # fanout: `n_quads = world_size * n_sectors` is 32 of a block's 64
            # quads at TP4 for the two INT4 stripes and 8 for the scale tail, so
            # half the block issues no payload store at all. Same bytes, fewer
            # lanes carrying them.
            if const_expr(active_threads >= BLOCK):
                for s in range(g0 + tid, g1, fx.Int32(BLOCK)):
                    _one_store(s)
            else:
                if tid < fx.Int32(active_threads):
                    for s in range(g0 + tid, g1, fx.Int32(active_threads)):
                        _one_store(s)

        def _publish():
            """qr_int4's ``_publish``, minus the colour bookkeeping.

            ``s_waitcnt vmcnt(0)`` retires this wave's peer stores, the barrier
            joins the other waves (vmcnt is per-wave), ``buffer_wbl2`` pushes
            anything still in this XCD's L2, and one 64 B sector per peer goes
            out write-through. The drain is the expensive part -- the C++ found
            ``buffer_wbl2`` free against a fine-grained destination, because
            such pages are not cacheable and there is nothing to write back.
            """
            rocdl.s_waitcnt(vmcnt=0)
            gpu.barrier()
            if const_expr(wbl2):
                llvm.InlineAsmOp(
                    None, [], "buffer_wbl2 sc1", "", has_side_effects=True
                )
                rocdl.s_waitcnt(vmcnt=0)
            quad = tid // fx.Int32(QUAD_LANES)
            lane_in_quad = tid % fx.Int32(QUAD_LANES)
            limit = fx.Int32(npeers)
            safe = (quad < limit).select(quad, fx.Int32(0))
            if quad < limit:
                color = fx.Int32(1)
                v4 = fx.Vector.from_elements(
                    [color, color, color, color], dtype=fx.Int32
                )
                slot = flag_off16 + (bid * fx.Int32(QUAD_LANES)) + lane_in_quad
                dest = peer_vec[safe]
                _store_v4i32_peer(
                    dest + fx.Int64(slot) * fx.Int64(V4_BYTES), v4, "sc0 sc1 nt"
                )

        for g in range(fx.Int32(0), ngroups, fx.Int32(1)):
            g0 = begin + g * group
            cap_g = g0 + group
            g1 = (cap_g < end).select(cap_g, end)
            if const_expr(subtiles > 1 and runtime_subtiles):
                # A *runtime* trip count, as qr_int4 has: its ST loop is
                # `for s in range(fx.Int32(0), n_this, fx.Int32(1))` where
                # `n_this = min(n_block_tiles - i, super_tile)` is only known at
                # run time. That is an scf.for the compiler cannot unroll, so it
                # cannot software-pipeline one sub-tile's stores against the
                # next -- unlike the range_constexpr form below, where all eight
                # sub-tiles are laid out flat and free to overlap.
                n_sub = fx.Int32(subtiles)
                for st in range(fx.Int32(0), n_sub, fx.Int32(1)):
                    s0 = g0 + st * fx.Int32(sub16)
                    cap_s = s0 + fx.Int32(sub16)
                    s1 = (cap_s < g1).select(cap_s, g1)
                    _store_range(s0, s1)
                    if (st + fx.Int32(1)) < n_sub:
                        rocdl.s_waitcnt(lgkmcnt=0)
                        gpu.barrier()
            elif const_expr(subtiles > 1):
                for st in range_constexpr(subtiles):
                    s0 = g0 + fx.Int32(st * sub16)
                    cap_s = s0 + fx.Int32(sub16)
                    s1 = (cap_s < g1).select(cap_s, g1)
                    _store_range(s0, s1)
                    if const_expr(st + 1 < subtiles):
                        rocdl.s_waitcnt(lgkmcnt=0)
                        gpu.barrier()
            else:
                _store_range(g0, g1)
            if const_expr(pub16 != 0):
                _publish()

    flat_wg = f"{BLOCK},{BLOCK}"
    tag = (
        f"p{npeers}_pub{publish_bytes}_{'wbl2' if wbl2 else 'nowbl2'}"
        f"_{'lds' if from_lds else 'reg'}_t{active_threads}_l{lds_i32}"
        f"_sub{subtiles}{'rt' if runtime_subtiles else ''}_pitch{slot_pitch}"
    )

    @flyc.jit
    def launch_peer_write(
        peer_ptrs: Int64,
        total16: Int32,
        run_shift: Int32,
        flag_off16: Int32,
        n_blocks: Int32,
        grid_x: Int32,
        stream: Stream = Stream(None),  # noqa: B008
    ):
        peer_write(
            peer_ptrs,
            total16,
            run_shift,
            flag_off16,
            n_blocks,
            value_attrs={"rocdl.flat_work_group_size": flat_wg},
        ).launch(grid=(grid_x, 1, 1), block=(BLOCK, 1, 1), stream=stream)

    launch_peer_write.func.__name__ = f"launch_peer_write_{tag}"
    try:
        peer_write.func.__name__ = f"peer_write_{tag}"
    except AttributeError:
        pass
    _KERNEL_CACHE[
        (npeers, publish_bytes, wbl2, policy, from_lds, active_threads, lds_i32,
         subtiles, slot_pitch, runtime_subtiles)
    ] = launch_peer_write
    return launch_peer_write


_KERNEL_CACHE: dict = {}


def _get_kernel(
    npeers, publish_bytes, wbl2=True, policy="nt",
    from_lds=False, active_threads=BLOCK, lds_i32=0, subtiles=1,
    slot_pitch=0, runtime_subtiles=False,
):
    key = (npeers, publish_bytes, wbl2, policy, from_lds, active_threads,
           lds_i32, subtiles, slot_pitch, runtime_subtiles)
    if key not in _KERNEL_CACHE:
        make_peer_write_kernel(
            npeers=npeers, publish_bytes=publish_bytes, wbl2=wbl2, policy=policy,
            from_lds=from_lds, active_threads=active_threads, lds_i32=lds_i32,
            subtiles=subtiles, slot_pitch=slot_pitch,
            runtime_subtiles=runtime_subtiles,
        )
    return _KERNEL_CACHE[key]


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------


def _shift_of(chunk_bytes: int) -> int:
    if chunk_bytes < V4_BYTES or chunk_bytes & (chunk_bytes - 1):
        raise ValueError(f"chunk must be a power of two >= 16, got {chunk_bytes}")
    return (chunk_bytes // V4_BYTES).bit_length() - 1


class _Fabric:
    """Destination buffers on every device, peer-mapped into the writer's space.

    One instance per allocation mode: the mode is a property of the buffers, so
    reusing an allocation across modes would silently measure the first one.
    """

    def __init__(self, devs, alloc: str, nbytes: int):
        self.devs = list(devs)
        self.alloc = alloc
        self.bufs = []
        flags = ALLOC_FLAGS[alloc]
        for d in self.devs:
            torch.cuda.set_device(d)
            self.bufs.append(UncachedIpcHeap.alloc(nbytes, flags))
        _enable_peer_access(self.devs[0], self.devs)
        torch.cuda.set_device(self.devs[0])
        # Peer-pointer table on the writer, uncached like qr_int4's meta buffer:
        # the kernel reads it, no peer ever writes it.
        self._table = None

    def table(self, npeers: int) -> int:
        """Device array of the first *npeers* destination pointers."""
        if self._table is not None:
            UncachedIpcHeap.free_device_mem(self._table)
        torch.cuda.set_device(self.devs[0])
        self._table = UncachedIpcHeap.alloc_uncached(npeers * 8)
        ptrs = [self.bufs[(1 + i) % len(self.devs)] for i in range(npeers)]
        UncachedIpcHeap.copy_host_to_device(
            self._table, (ctypes.c_int64 * npeers)(*ptrs), npeers * 8
        )
        return self._table

    def close(self):
        if self._table is not None:
            UncachedIpcHeap.free_device_mem(self._table)
            self._table = None
        for d, b in zip(self.devs, self.bufs):
            torch.cuda.set_device(d)
            UncachedIpcHeap.free_device_mem(b)
        self.bufs = []


def measure(
    fabric: _Fabric,
    *,
    npeers: int,
    chunk_bytes: int,
    publish_bytes: int = 0,
    wbl2: bool = True,
    total_bytes: int = TOTAL_BYTES,
    blocks: int = DEFAULT_BLOCKS,
    iters: int = ITERS,
    warmup: int = WARMUP,
    from_lds: bool = False,
    active_threads: int = BLOCK,
    lds_i32: int = 0,
    subtiles: int = 1,
    slot_pitch: int = 0,
    runtime_subtiles: bool = False,
) -> float:
    """GB/s of peer writes for one configuration."""
    launch = _get_kernel(
        npeers, publish_bytes, wbl2,
        from_lds=from_lds, active_threads=active_threads, lds_i32=lds_i32,
        subtiles=subtiles, slot_pitch=slot_pitch,
        runtime_subtiles=runtime_subtiles,
    )
    torch.cuda.set_device(fabric.devs[0])

    args = (
        Int64(int(fabric.table(npeers))),
        Int32(total_bytes // V4_BYTES),
        Int32(_shift_of(chunk_bytes)),
        Int32(total_bytes // V4_BYTES),  # flags live just past the payload
        Int32(blocks),
        Int32(blocks),
        Stream(None),
    )
    compiled = flyc.compile(launch, *args)  # compiles AND launches once
    for _ in range(warmup):
        compiled(*args)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        compiled(*args)
    stop.record()
    stop.synchronize()

    us = start.elapsed_time(stop) * 1e3 / iters
    return total_bytes / (us * 1e3)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def _skip_reasons(npeers: int):
    if not torch.cuda.is_available():
        return "no GPU"
    if get_gfx_runtime() not in _ARCHS:
        return f"arch {get_gfx_runtime()} not in {_ARCHS}"
    if torch.cuda.device_count() < npeers + 1:
        return f"needs {npeers + 1} GPUs, {torch.cuda.device_count()} visible"
    return None


_CASES = sorted(CPP_REFERENCE)


@pytest.fixture(scope="module")
def fabrics():
    """One allocation per mode, shared across cases; freed at teardown."""
    made: dict = {}
    yield made
    for f in made.values():
        f.close()


def _fabric_for(made, alloc, ndev):
    if alloc not in made:
        made[alloc] = _Fabric(
            range(ndev), alloc, TOTAL_BYTES + FLAG_TAIL_BYTES
        )
    return made[alloc]


@pytest.mark.parametrize("alloc,npeers,chunk,publish", _CASES)
def test_matches_cpp(fabrics, alloc, npeers, chunk, publish):
    """The FlyDSL primitive reproduces the C++ primitive.

    A failure here is not a flaky benchmark -- it means the same access pattern,
    the same instruction and the same memory behave differently depending on
    which compiler emitted the loop, which is exactly the open question in
    section 9.3 of the 09-07 report.
    """
    reason = _skip_reasons(npeers)
    if reason:
        pytest.skip(reason)

    ndev = min(torch.cuda.device_count(), 4)
    fabric = _fabric_for(fabrics, alloc, ndev)
    got = measure(
        fabric, npeers=npeers, chunk_bytes=chunk, publish_bytes=publish
    )

    binary = _cpp_binary()
    want = _cpp_measure(binary, alloc, npeers, chunk, publish) if binary else None
    if want is not None:
        src = f"live {os.path.basename(binary)}"
    else:
        want = CPP_REFERENCE[(alloc, npeers, chunk, publish)]
        src = "recorded 2026-09-07"
    tol = TOLERANCE_UNSTABLE if _unstable(alloc, npeers) else TOLERANCE

    ratio = got / want
    assert abs(ratio - 1.0) <= tol, (
        f"FlyDSL {got:.2f} GB/s vs C++ {want:.2f} GB/s ({ratio:.2f}x, {src}) for "
        f"alloc={alloc} peers={npeers} chunk={chunk}B publish={publish}B. "
        "Same pattern, same instruction, same memory -- a gap here is a codegen "
        "difference between hipcc and FlyDSL."
    )


def test_uncached_collapses_at_multiple_peers(fabrics):
    """The 2026-08-31 finding the fine-grained inbox was built on.

    Independent of the reference table above: a self-contained statement that
    uncached peer writes serialize per destination while fine-grained ones do
    not. If this stops holding, the ``inbox_memory`` default in ``QRInt4`` is
    wrong for this host and the docstrings that cite it are stale.
    """
    reason = _skip_reasons(3)
    if reason:
        pytest.skip(reason)

    ndev = min(torch.cuda.device_count(), 4)
    fine = measure(
        _fabric_for(fabrics, "fine", ndev), npeers=3, chunk_bytes=64
    )
    unc = measure(
        _fabric_for(fabrics, "uncached", ndev), npeers=3, chunk_bytes=64
    )
    assert fine / unc > 10.0, (
        f"fine-grained {fine:.2f} GB/s vs uncached {unc:.2f} GB/s to 3 peers "
        f"({fine / unc:.1f}x); 08-31 measured ~23x and the C++ primitive "
        "measured 24.5x on 2026-09-07"
    )


def test_contiguous_runs_beat_sector_interleave(fabrics):
    """>=128 B per destination beats the 64 B interleave, the ``fanout`` knob.

    ``_INBOX_POLICY['finegrained']['fanout'] == 'peer'`` exists because of this.
    """
    reason = _skip_reasons(3)
    if reason:
        pytest.skip(reason)

    ndev = min(torch.cuda.device_count(), 4)
    fabric = _fabric_for(fabrics, "fine", ndev)
    sector = measure(fabric, npeers=3, chunk_bytes=64)
    peer = measure(fabric, npeers=3, chunk_bytes=512)
    assert peer / sector > 1.25, (
        f"512 B runs {peer:.2f} GB/s vs 64 B interleave {sector:.2f} GB/s "
        f"({peer / sector:.2f}x); the C++ primitive measured 1.50x"
    )


# ---------------------------------------------------------------------------
# Standalone sweep
# ---------------------------------------------------------------------------


def main_one(args) -> int:
    """Run a single configuration repeatedly.

    Exists so a WaveScope capture profile has one dispatch shape to point at:
    the sweep in ``main`` launches a different kernel per rung, which would give
    the profiler several distinct binaries under one `kernelRegex`.
    """
    ndev = min(torch.cuda.device_count(), 4)
    if ndev < args.peers + 1:
        print(f"needs {args.peers + 1} GPUs, {ndev} visible")
        return 0
    fabric = _Fabric(range(ndev), args.alloc, TOTAL_BYTES + FLAG_TAIL_BYTES)
    try:
        gbps = measure(
            fabric, npeers=args.peers, chunk_bytes=args.chunk,
            publish_bytes=args.publish, blocks=args.blocks,
            iters=args.iters, warmup=args.warmup,
            from_lds=args.src == "lds",
            lds_i32=2304 if args.src == "lds" else 0,
        )
    finally:
        fabric.close()
    print(f"{gbps:.2f} GB/s  alloc={args.alloc} peers={args.peers} "
          f"chunk={args.chunk}B publish={args.publish}B blocks={args.blocks}")
    return 0


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--one", action="store_true",
                    help="run a single configuration instead of the sweep")
    ap.add_argument("--alloc", default="fine", choices=sorted(ALLOC_FLAGS))
    ap.add_argument("--peers", type=int, default=3)
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--publish", type=int, default=55296)
    ap.add_argument("--blocks", type=int, default=DEFAULT_BLOCKS)
    ap.add_argument("--src", default="reg", choices=("reg", "lds"))
    ap.add_argument("--iters", type=int, default=ITERS)
    ap.add_argument("--warmup", type=int, default=WARMUP)
    args = ap.parse_args()

    if not torch.cuda.is_available() or get_gfx_runtime() not in _ARCHS:
        print("no supported GPU; nothing to do")
        return 0
    if args.one:
        return main_one(args)

    ndev = min(torch.cuda.device_count(), 4)
    if ndev < 2:
        print(f"needs >= 2 GPUs, {ndev} visible")
        return 0

    binary = _cpp_binary()
    made: dict = {}
    rows = []
    try:
        for alloc, npeers, chunk, publish in _CASES:
            if npeers + 1 > ndev:
                continue
            fabric = _fabric_for(made, alloc, ndev)
            got = measure(
                fabric, npeers=npeers, chunk_bytes=chunk, publish_bytes=publish
            )
            want = (
                _cpp_measure(binary, alloc, npeers, chunk, publish)
                if binary
                else None
            )
            if want is None:
                want = CPP_REFERENCE[(alloc, npeers, chunk, publish)]
            rows.append((alloc, npeers, chunk, publish, got, want, got / want))
    finally:
        for f in made.values():
            f.close()

    print()
    src = _cpp_binary() or "recorded 2026-09-07"
    print(f"peer-write bandwidth, FlyDSL vs C++ (MI350P, {ndev} GPUs, "
          f"{TOTAL_BYTES >> 20} MiB, {DEFAULT_BLOCKS} blocks)")
    print(f"C++ reference: {src}")
    print()
    print("| alloc | peers | chunk B | publish B | FlyDSL GB/s | C++ GB/s | "
          "FlyDSL/C++ |")
    print("|:---|---:|---:|---:|---:|---:|---:|")
    for alloc, npeers, chunk, publish, got, want, ratio in rows:
        print(f"| {alloc} | {npeers} | {chunk} | {publish} | {got:.2f} | "
              f"{want:.2f} | {ratio:.2f}x |")
    print()
    worst = max((abs(r[6] - 1.0) for r in rows), default=0.0)
    print(f"worst deviation from the C++ primitive: {worst * 100:.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
