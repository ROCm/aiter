# SPDX-License-Identifier: MIT
"""M0 unit kernels: the mailbox, the epoch and the lane primitives."""

from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from deepseek_v4_gluon.common.epoch import epoch_begin, epoch_end, load_epoch
from deepseek_v4_gluon.common.ops import lane_ids, sum_desc6
from deepseek_v4_gluon.common.sync import (
    T512,
    deadline,
    fail,
    poll1,
    poll4,
    publish,
    put,
)


@gluon.jit
def k_ring(scratch, health, epoch, out, rank, SYSTEM: gl.constexpr):
    """CTA b publishes pair b (value b * 1000 + 7) with the launch's tag, then
    polls CTA (b + 1) % 256's pair and four of CTA (b + 2)'s: every CTA
    depends on another, a ring through every XCD. Out[b][t] = the polled value
    butterfly-summed over the wave (64 x value)."""
    tid = gl.arange(0, 512, layout=T512)
    bid = gl.program_id(0)
    ep = load_epoch(epoch)
    epoch_begin(epoch, ep, bid, tid)
    tag = (ep << 1) + 2 + gl.zeros_like(tid)
    lane = lane_ids(tid)
    zero = gl.zeros_like(tid)
    base = scratch.to(gl.uint64)
    publish(base, zero + bid, zero + bid * 1000 + 7, tag, tid == 0, SYSTEM)
    dl = deadline(tid)
    src = (bid + 1) % 256
    v, f = poll1(base + (zero + src).to(gl.uint64) * 8, tag, tid < 64, dl, health,
                 SYSTEM)  # fmt: skip
    fail(health, f, rank, 1, tid < 64)
    s2 = (bid + 2) % 256
    a = base + (zero + s2).to(gl.uint64) * 8
    w0, w1, w2, w3, f4 = poll4(a, a, a, a, tag, tid < 64, dl, health, SYSTEM)
    fail(health, f4, rank, 2, tid < 64)
    tot = sum_desc6(v.to(gl.float32) + (w0 + w1 + w2 + w3).to(gl.float32), lane)
    gl.store(out + bid * 512 + tid, tot, mask=tid < 64)
    epoch_end(epoch, ep, bid, tid, health, rank)
