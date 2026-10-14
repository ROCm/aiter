"""TP4/TP8 Kimi peer reduction: one destination wave and batched source polling.

Ported schedule: FlyDSL 21a3d1e kernels/monokernel/k3/kernel.py:1481-1548.
The address form is global rather than buffer; the gfx950 transfer width and
coherence controls are the same. Generated ISA is verified by the harness.
"""

from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl

from .primitives import bf16_sum_peers, pack_bf16, poll_peers, publish_sys


@g.jit
def allreduce(
    X,
    PeerAddresses,
    Local,
    Out,
    Epochs,
    Errors,
    N: gl.constexpr,
    GRID: gl.constexpr,
    RANK: gl.constexpr,
    DELAY_RANK: gl.constexpr = -1,
    DELAY: gl.constexpr = 0,
    WORLD: gl.constexpr = 8,
):
    """Standalone adapter for the compute-tile peer reduction.

    X: two changing versions, int32 views of BF16 pairs. Counter and error
    storage belong exclusively to this stream. TP4/TP8; N divisible by 128.
    One epoch per CTA advances internally for graph replay. Two slots protect
    a previous epoch when a peer is one launch ahead. Epoch overflow requires
    a coordinated host reset. Errors has GRID*512 int32 entries, initially zero.
    """
    pid = gl.program_id(0)
    epoch = gl.load(Epochs + pid) + 1
    slot = epoch & 1
    pairs: gl.constexpr = N // 2
    layout: gl.constexpr = gl.BlockedLayout([1], [64], [8], [0])
    tid = gl.arange(0, 512, layout=layout)
    wave = tid // 64
    lane = tid % 64
    peer = gl.load(PeerAddresses + wave, wave < WORLD, 0).to(gl.pointer_type(gl.uint64))
    local_slot = Local + slot * WORLD * pairs
    task = pid
    while task < pairs // 64:
        pair = task * 64 + lane
        bits = gl.load(X + slot * pairs + pair)
        if RANK == DELAY_RANK:
            for _ in range(DELAY):
                gl.inline_asm_elementwise(
                    "s_sleep 1\n s_mov_b32 $0, 0",
                    "=s",
                    [],
                    dtype=gl.int32,
                    is_pure=False,
                    pack=1,
                )
        publish_sys(
            peer + (slot * WORLD + RANK) * pairs + pair, bits, epoch, wave < WORLD
        )
        gl.barrier()
        words, failed = poll_peers(local_slot, pair, epoch, tid < 64, pairs, WORLD)
        lo, hi = bf16_sum_peers(words)
        gl.store(Out + pair, pack_bf16(lo, hi), tid < 64)
        gl.store(Errors + pid * 512 + tid, 1, (tid < 64) & (failed != 0))
        task += GRID
    gl.store(Epochs + pid, epoch)
