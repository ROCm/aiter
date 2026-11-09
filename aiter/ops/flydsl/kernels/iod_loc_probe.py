# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""IOD phase probe for ``iod_locality`` (MI455X SPX + NPS1, gfx1250).

home(addr) = a14 ^ a17 ^ a20 ^ phase[addr >> 21]; workgroup b of a grid whose
workgroups all run is on domain ((b + 3) % 8) >= 4. One launch of 8 probers,
one per workgroup class, all running: prober b walks every page and times one
cold load in each of its first 16 granules, whose in-page home bits
a14 ^ a17 alternate, so the faster parity names domain(b) ^ phase(page) without
an absolute latency threshold. Eight concurrent single loads do not contend.

A launch whose other workgroups exit at once does not keep that mapping (two of
the eight classes land on the other domain), so every prober has to run.

Where workgroup 0 starts is dispatcher state, not a constant: a launch sees
domain(b) = ((b + c) & 7) >= 4 for a rotation c that follows the hardware pipe
it lands on (HW_ID2 PIPE_ID 1, 2, 3 -> c = 3, 2, 1; pipe 0 -> 0 or 4). A process
that launches from many streams (ATOM, MORI) gets pipe 0 at load time, so the
probe saw c = 0 there on every rank. The classes still split cleanly into a
rotation, so each pass first reads its rotation from the relative pattern of
the eight classes and only passes on the c = 3 rotation the GEMM schedules
assume are kept (c = 7, its complement, was never observed); the others are
relaunched on another stream. Each class must agree with the domain formula on
(nearly) every page of the kept passes; if one does not, or too few passes land
on c = 3, the probe reports it instead of returning a phase.
"""

import numpy as np
import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr import arith
from flydsl.expr.typing import T

from .buffer_ops import buffer_store, create_buffer_resource_from_addr
from .tensor_shim import _run_compiled

PROBERS = 8
NGRAN = 16
#: A class agreeing with the domain formula on fewer pages than this fails the probe.
MIN_CLASS_AGREE = 0.9
#: Launches in all, warm-up included: each reads its own cold 128 B line of a granule.
MAX_LAUNCHES = 128 // PROBERS
_PAR = np.array([(g ^ (g >> 3)) & 1 for g in range(NGRAN)])
_ROT = np.array(
    [[((b + c) & 7) >= 4 for b in range(PROBERS)] for c in range(8)], dtype=np.int64
)
#: hwreg(HW_REG_HW_ID2, offset 0, size 6): QUEUE_ID [3:0], PIPE_ID [5:4].
_HW_ID2_QUEUE_PIPE = 24 | (0 << 6) | ((6 - 1) << 11)


def launch_rotation():
    """Rotation c of the running launch: workgroup b is on IOD domain ((b + c) & 7) >= 4.

    The dispatch of a launch starts on the XCC its hardware queue slot selects:
    c = 4 - PIPE_ID - 4 QUEUE_ID mod 8, up to one constant common to all slots,
    which the probe's phase absorbs. Read from HW_ID2, it is the same for every
    workgroup of a launch.
    """
    imm = arith.unwrap(fx.Int32(_HW_ID2_QUEUE_PIPE))
    v = fx.Int32(_llvm.call_intrinsic(T.i32, "llvm.amdgcn.s.getreg", [imm], [], []))
    return (fx.Int32(4) - (v >> fx.Int32(4)) - (v << fx.Int32(2))) & fx.Int32(7)


def _timed_load(addr):
    """Realtime ticks around one global load at ``addr`` (i64), waited in place."""
    res = ir.Type.parse("!llvm.struct<(i64, i64, i32)>")
    r = _llvm.inline_asm(
        res,
        [arith.unwrap(fx.Int64(addr))],
        "s_sendmsg_rtn_b64 $0, sendmsg(MSG_RTN_GET_REALTIME)\n"
        "s_wait_kmcnt 0\n"
        "global_load_b32 $2, $3, off\n"
        "s_wait_loadcnt 0\n"
        "s_sendmsg_rtn_b64 $1, sendmsg(MSG_RTN_GET_REALTIME)\n"
        "s_wait_kmcnt 0",
        "=&s,=&s,=&v,v",
        has_side_effects=True,
    )
    t0 = fx.Int64(_llvm.extractvalue(T.i64, r, [0]))
    t1 = fx.Int64(_llvm.extractvalue(T.i64, r, [1]))
    return fx.Int32(arith.trunci(T.i32, arith.unwrap(t1 - t0)))


@flyc.jit
def _launch_probe(
    base: fx.Int64,
    lat: fx.Int64,
    npg: fx.Int32,
    line: fx.Int32,
    stream: fx.Stream,
):
    # The cache key follows only this body's globals, not the kernel's below.
    _ = launch_rotation

    @flyc.kernel(name="iod_loc_probe", known_block_size=[32, 1, 1])
    def kernel(
        base: fx.Int64,
        lat: fx.Int64,
        npg: fx.Int32,
        line: fx.Int32,
    ):
        bid = fx.Int32(fx.block_idx.x)
        tid = fx.Int32(fx.thread_idx.x)
        if tid == fx.Int32(0):
            rsrc = create_buffer_resource_from_addr(arith.unwrap(lat))
            # A 128 B line per (pass, prober): cold, and never shared.
            lo = fx.Int64((line * fx.Int32(PROBERS) + bid) & fx.Int32(127)) * fx.Int64(128)
            for p in range(fx.Int32(0), npg, fx.Int32(1)):
                for g in range(fx.Int32(0), fx.Int32(NGRAN), fx.Int32(1)):
                    off = (fx.Int64(p) << fx.Int64(21)) + fx.Int64(g) * fx.Int64(16384) + lo
                    t = _timed_load(base + off)
                    buffer_store(
                        arith.unwrap(t),
                        rsrc,
                        arith.unwrap((bid * npg + p) * fx.Int32(NGRAN) + g),
                    )
            buffer_store(
                arith.unwrap(launch_rotation()),
                rsrc,
                arith.unwrap(fx.Int32(PROBERS * NGRAN) * npg + bid),
            )

    kernel(base, lat, npg, line).launch(
        grid=(PROBERS, 1, 1), block=(32, 1, 1), stream=stream
    )


def _rotation(odd_fast: np.ndarray):
    """Rotation c in 0..3 of one launch (c and c + 4 read the same), or None.

    Class b's odd-fast pages are domain(b) ^ phase; relative to class 0 the
    phase cancels and leaves domain(b) ^ domain(0), which names the rotation.
    """
    rel = (odd_fast != odd_fast[:1]).mean(1)
    if np.minimum(rel, 1 - rel).max() > 1 - MIN_CLASS_AGREE:
        return None
    rel = (rel > 0.5).astype(np.int64)
    for c in range(4):
        if np.array_equal(_ROT[c] ^ _ROT[c][0], rel):
            return c
    return None


def probe_phase(buf: torch.Tensor, npass: int = 4):
    """Phase of every 2 MiB page of ``buf`` (2 MiB aligned).

    Returns (phase u8[npg], weak pages, per-class agreement, rotation of every
    launch); phase is None when a workgroup class does not follow the domain
    formula or fewer than ``npass`` launches after the warm-up run at the
    rotation ``launch_rotation`` reads from their hardware queue.
    """
    npg = buf.numel() * buf.element_size() // (2 << 20)
    assert buf.data_ptr() % (2 << 20) == 0 and npg > 0
    assert npass + 1 <= MAX_LAUNCHES
    dev = buf.device
    lat = torch.zeros(PROBERS * npg * NGRAN + PROBERS, dtype=torch.int32, device=dev)
    current = torch.cuda.current_stream(dev)
    stream = current
    est = []
    rotations = []
    # Launch 0 is a discarded warm-up: the first loads after an idle GPU read
    # slow on either domain.
    for line in range(MAX_LAUNCHES):
        stream.wait_stream(current)
        _run_compiled(
            _launch_probe, buf.data_ptr(), lat.data_ptr(), npg, line, fx.Stream(stream)
        )
        current.wait_stream(stream)
        h = lat.cpu().numpy()
        t = h[: PROBERS * npg * NGRAN].reshape(PROBERS, npg, NGRAN)
        odd_fast = np.median(t[:, :, _PAR == 1], 2) < np.median(t[:, :, _PAR == 0], 2)
        c = int(h[-PROBERS])
        rot = _rotation(odd_fast)
        rotations.append(rot)
        if rot != c & 3 or (h[-PROBERS:] != c).any():
            # Another stream is the only handle on which pipe the next launch takes.
            stream = torch.cuda.Stream(dev)
            continue
        if line == 0:
            continue
        # Odd granules are homed on 1 ^ phase: fast iff that is the prober's domain.
        est.append(_ROT[c][:, None] ^ odd_fast.astype(np.int64))
        if len(est) == npass:
            break
    if len(est) < npass:
        return None, npg, np.full(PROBERS, len(est) / npass), rotations
    est = np.stack(est)
    votes = est.sum((0, 1))
    n = npass * PROBERS
    phase = (2 * votes > n).astype(np.uint8)
    weak = int((np.abs(2 * votes - n) * 2 < n).sum())
    agree = (est == phase[None, None, :]).mean((0, 2))
    if agree.min() < MIN_CLASS_AGREE:
        return None, weak, agree, rotations
    return phase, weak, agree, rotations
