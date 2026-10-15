# SPDX-License-Identifier: MIT
"""Gluon TP4 tail for MiniMax M3's prepared two-stage MoE output.

The peer message is self-contained: two BF16 values and a 32-bit epoch in one
aligned uint64. Rank-ordered FP32 addition follows ATOM's small-batch path.
"""

from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl

from .sync import poll_peers4


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
        gl.atomic_xchg(
            dest + (slot * 4 + RANK) * pairs + pair,
            message,
            mask=(wave < 4) & active,
            sem="relaxed",
            scope="sys",
        )
        gl.barrier()
        local_pair = task * 64 + lane
        use = (tid < 64) & (local_pair < pairs)
        w0, w1, w2, w3, failed = poll_peers4(own, local_pair, epoch, use, pairs)
        gl.store(Errors + pid * 64 + lane, 1, mask=use & (failed != 0))
        # Rank 0, 1, 2, 3 in FP32; round only at the final BF16 store.
        lo = (w0.to(gl.uint32) << 16).to(gl.float32, bitcast=True)
        lo += (w1.to(gl.uint32) << 16).to(gl.float32, bitcast=True)
        lo += (w2.to(gl.uint32) << 16).to(gl.float32, bitcast=True)
        lo += (w3.to(gl.uint32) << 16).to(gl.float32, bitcast=True)
        hi = (w0.to(gl.uint32) & 4294901760).to(gl.float32, bitcast=True)
        hi += (w1.to(gl.uint32) & 4294901760).to(gl.float32, bitcast=True)
        hi += (w2.to(gl.uint32) & 4294901760).to(gl.float32, bitcast=True)
        hi += (w3.to(gl.uint32) & 4294901760).to(gl.float32, bitcast=True)
        gl.store(Out + local_pair * 2, lo, mask=use & (failed == 0))
        gl.store(Out + local_pair * 2 + 1, hi, mask=use & (failed == 0))
        task += 256
    gl.store(Epochs + pid, epoch)
