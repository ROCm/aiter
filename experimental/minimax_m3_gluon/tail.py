# SPDX-License-Identifier: MIT
"""Gluon TP4 tail for MiniMax M3's prepared two-stage MoE output.

The peer message is self-contained: two BF16 values and a 32-bit epoch in one
aligned uint64. Rank-ordered FP32 addition follows ATOM's small-batch path.
"""

from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl


@g.jit
def tail(
    MoEOutput,
    PeerAddresses,
    Local,
    Out,
    Epochs,
    Errors,
    B: gl.constexpr,
    RANK: gl.constexpr,
):
    gl.static_assert(B >= 1 and B <= 16)
    gl.static_assert(RANK >= 0 and RANK < 4)
    pid = gl.program_id(0)
    epoch = gl.load(Epochs + pid) + 1
    slot = epoch & 1
    pairs: gl.constexpr = B * 6144 // 2
    layout: gl.constexpr = gl.BlockedLayout([1], [64], [8], [0])
    tid = gl.arange(0, 512, layout=layout)
    wave = tid // 64
    lane = tid % 64
    dest = gl.load(PeerAddresses + wave, wave < 4, other=0).to(
        gl.pointer_type(gl.uint64)
    )
    own = Local + slot * 4 * pairs
    task = pid
    while task < gl.cdiv(pairs, 64):
        pair = task * 64 + lane
        active = pair < pairs
        low = gl.load(MoEOutput + pair * 2, mask=active, other=0, cache_modifier=".cv")
        high = gl.load(
            MoEOutput + pair * 2 + 1, mask=active, other=0, cache_modifier=".cv"
        )
        bits = low.to(gl.uint16, bitcast=True).to(gl.uint32)
        bits |= high.to(gl.uint16, bitcast=True).to(gl.uint32) << 16
        message = (epoch.to(gl.uint64) << 32) | bits.to(gl.uint64)
        gl.atomic_store(
            dest + (slot * 4 + RANK) * pairs + pair,
            message,
            mask=(wave < 4) & active,
            sem="relaxed",
            scope="sys",
        )
        gl.barrier()
        local_pair = task * 64 + lane
        use = (tid < 64) & (local_pair < pairs)
        p0 = own + 0 * pairs + local_pair
        p1 = own + 1 * pairs + local_pair
        p2 = own + 2 * pairs + local_pair
        p3 = own + 3 * pairs + local_pair
        missing = gl.full((512,), True, gl.int1, layout)
        w0 = gl.full((512,), 0, gl.uint64, layout)
        w1 = gl.full((512,), 0, gl.uint64, layout)
        w2 = gl.full((512,), 0, gl.uint64, layout)
        w3 = gl.full((512,), 0, gl.uint64, layout)
        for attempt in range(100000):
            w0 = gl.atomic_load(p0, mask=use, sem="acquire", scope="sys")
            w1 = gl.atomic_load(p1, mask=use, sem="acquire", scope="sys")
            w2 = gl.atomic_load(p2, mask=use, sem="acquire", scope="sys")
            w3 = gl.atomic_load(p3, mask=use, sem="acquire", scope="sys")
            missing = use & (
                ((w0 >> 32) != epoch)
                | ((w1 >> 32) != epoch)
                | ((w2 >> 32) != epoch)
                | ((w3 >> 32) != epoch)
            )
            if gl.sum(missing.to(gl.int32), 0) == 0:
                break
        if gl.sum(missing.to(gl.int32), 0) != 0:
            gl.store(Errors + pid * 64 + lane, 1, mask=tid < 64)
        else:
            # The reduction order is rank 0, 1, 2, 3, with one rounding at
            # the final BF16 store, as in ATOM's small-batch FFN reduction.
            lo = (w0.to(gl.uint32) << 16).to(gl.float32, bitcast=True)
            lo += (w1.to(gl.uint32) << 16).to(gl.float32, bitcast=True)
            lo += (w2.to(gl.uint32) << 16).to(gl.float32, bitcast=True)
            lo += (w3.to(gl.uint32) << 16).to(gl.float32, bitcast=True)
            hi = (w0.to(gl.uint32) & 4294901760).to(gl.float32, bitcast=True)
            hi += (w1.to(gl.uint32) & 4294901760).to(gl.float32, bitcast=True)
            hi += (w2.to(gl.uint32) & 4294901760).to(gl.float32, bitcast=True)
            hi += (w3.to(gl.uint32) & 4294901760).to(gl.float32, bitcast=True)
            gl.store(Out + local_pair * 2, lo, mask=use)
            gl.store(Out + local_pair * 2 + 1, hi, mask=use)
        task += 256
    gl.store(Epochs + pid, epoch)
