# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""IOD phase probe for ``iod_locality`` (MI455X SPX + NPS1, gfx1250).

home(addr) = a14 ^ a17 ^ a20 ^ phase[addr >> 21]; workgroup b of a 256-CTA
grid runs on domain ((b + 3) % 8) >= 4. Prober j = 8 kline + cls times one
cold load per page it owns; a load homed on its own domain is ~34 ticks of the
100 MHz realtime clock, a remote one ~44.5. Each launch lets only prober j run
(the other CTAs exit), so loads never contend and no grid barrier is needed.
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

PROBERS = 256
KLINES = PROBERS // 8
FAST_TICKS = 39
_HASH = 2654435761


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
    prober: fx.Int32,
    pass_idx: fx.Int32,
    stream: fx.Stream,
):
    @flyc.kernel(name="iod_loc_probe", known_block_size=[32, 1, 1])
    def kernel(
        base: fx.Int64,
        lat: fx.Int64,
        npg: fx.Int32,
        prober: fx.Int32,
        pass_idx: fx.Int32,
    ):
        bid = fx.Int32(fx.block_idx.x)
        tid = fx.Int32(fx.thread_idx.x)
        if (bid == prober) & (tid == fx.Int32(0)):
            kline = prober >> fx.Int32(3)
            cls = prober & fx.Int32(7)
            rsrc = create_buffer_resource_from_addr(arith.unwrap(lat))
            for b in range(fx.Int32(0), npg, fx.Int32(1)):
                c0 = fx.Int32(
                    arith.trunci(
                        T.i32,
                        arith.unwrap(
                            ((fx.Int64(b) * fx.Int64(_HASH)) & fx.Int64(0xFFFFFFFF))
                            >> fx.Int64(29)
                        ),
                    )
                )
                if ((c0 + kline + pass_idx) & fx.Int32(7)) == cls:
                    off = (
                        (fx.Int64(b) << fx.Int64(21))
                        + fx.Int64(kline) * fx.Int64(65536)
                        + fx.Int64(pass_idx) * fx.Int64(4096)
                        + fx.Int64(cls) * fx.Int64(128)
                    )
                    t = _timed_load(base + off)
                    buffer_store(arith.unwrap(t), rsrc, arith.unwrap(b * fx.Int32(KLINES) + kline))

    kernel(base, lat, npg, prober, pass_idx).launch(
        grid=(PROBERS, 1, 1), block=(32, 1, 1), stream=stream
    )


def _inpage(off):
    return ((off >> 14) ^ (off >> 17) ^ (off >> 20)) & 1


def probe_phase(buf: torch.Tensor, npass: int = 4):
    """Phase of every 2 MiB page of ``buf`` (2 MiB aligned); returns (phase u8[npg], weak pages)."""
    npg = buf.numel() * buf.element_size() // (2 << 20)
    assert buf.data_ptr() % (2 << 20) == 0 and npg > 0
    dev = buf.device
    lat = torch.zeros(npg * KLINES, dtype=torch.int32, device=dev)
    flush = torch.empty(1 << 30, dtype=torch.uint8, device=dev)
    stream = torch.cuda.current_stream(dev)
    b = np.arange(npg, dtype=np.uint64)
    c0 = ((b * _HASH) & 0xFFFFFFFF) >> 29
    votes = np.zeros(npg, dtype=np.int64)
    n = np.zeros(npg, dtype=np.int64)
    # Pass -1 is a discarded warm-up: the first loads after an idle GPU
    # read slow on either domain.
    for p in range(-1, npass):
        flush.fill_(p & 0xFF)
        for j in range(PROBERS):
            _run_compiled(
                _launch_probe,
                buf.data_ptr(),
                lat.data_ptr(),
                npg,
                j,
                max(p, 0),
                fx.Stream(stream),
            )
        if p < 0:
            continue
        h = lat.view(npg, KLINES).cpu().numpy()
        for kk in range(KLINES):
            cls = (c0 + kk + p) % 8
            j = 8 * kk + cls
            dom = ((j + 3) & 7) >= 4
            off = (b << np.uint64(21)) + np.uint64(kk * 65536 + p * 4096) + 128 * cls
            x = dom.astype(np.int64) ^ _inpage(off).astype(np.int64)
            fast = h[:, kk] < FAST_TICKS
            votes += np.where(fast, x, 1 - x)
            n += 1
    del flush
    phase = (2 * votes > n).astype(np.uint8)
    weak = int((np.abs(2 * votes - n) * 2 < n).sum())
    return phase, weak
