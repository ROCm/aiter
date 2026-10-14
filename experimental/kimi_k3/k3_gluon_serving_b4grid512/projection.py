# Frozen JIT snapshot: retain original imports and tuning overrides.
# ruff: noqa: E402
from .coherent_memory import coherent_load

"""Kimi KDA stage 3: BF16 12288/TP -> 7168 projection and optional TP4/TP8 sum.

Reference: FlyDSL 21a3d1e k3/kernel.py bf16_mfma, publish_mfma_pairs,
and stage 3. The batch dimension of the MFMA below represents two K splits,
not request batching; requests are the 16 columns of the MFMA instruction.
"""
import triton
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from .primitives import bf16_sum_peers, pack_bf16, poll_peers, publish_local

STAGE_INPUTS = gl.constexpr(True)
PROJECTION_UNROLL = gl.constexpr(4)
PROJECTION_BUFFER = gl.constexpr(True)


def pack_weights(w):
    """FlyDSL pack_bf16 layout, returned as BF16 for Gluon dot operands."""
    assert (
        w.shape[0] == 7168
        and w.shape[1] in (1536, 3072)
        and (str(w.dtype) == "torch.bfloat16")
    )
    return (
        w.reshape(448, 16, w.shape[1] // 64, 2, 4, 8)
        .permute(0, 2, 3, 4, 1, 5)
        .contiguous()
        .view(-1)
    )


@g.jit
def project_tile(
    X,
    W,
    slot,
    sample_base,
    row_base,
    B: gl.constexpr,
    K: gl.constexpr,
    ROWS: gl.constexpr = 64,
    Gain=None,
    Inverse=None,
):
    SPLITS: gl.constexpr = 128 // ROWS
    STEPS: gl.constexpr = K // (SPLITS * 32)
    gl.static_assert(ROWS == 16 or ROWS == 32 or ROWS == 64)
    gl.static_assert(K % (SPLITS * 32) == 0)
    mma: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=[16, 16, 32],
        transposed=False,
        warps_per_cta=[SPLITS, ROWS // 16, 1],
    )
    al: gl.constexpr = gl.BlockedLayout(
        [1, 1, 8], [1, 16, 4], [SPLITS, ROWS // 16, 1], [1, 2, 0]
    )
    bl: gl.constexpr = gl.BlockedLayout(
        [1, 8, 1], [1, 4, 16], [SPLITS, 1, ROWS // 16], [2, 1, 0]
    )
    sa = gl.arange(0, SPLITS, layout=gl.SliceLayout(1, gl.SliceLayout(2, al)))
    ra = gl.arange(0, ROWS, layout=gl.SliceLayout(0, gl.SliceLayout(2, al)))
    ka = gl.arange(0, 32, layout=gl.SliceLayout(0, gl.SliceLayout(0, al)))
    sb = gl.arange(0, SPLITS, layout=gl.SliceLayout(1, gl.SliceLayout(2, bl)))
    kb = gl.arange(0, 32, layout=gl.SliceLayout(0, gl.SliceLayout(2, bl)))
    nb = gl.arange(0, 16, layout=gl.SliceLayout(0, gl.SliceLayout(0, bl)))
    row = row_base + ra
    sample = gl.minimum(sample_base + nb, gl.minimum(sample_base + 3, B - 1))
    if STAGE_INPUTS:
        extent: gl.constexpr = triton.next_power_of_2(min(B, 4) * K)
        ii = gl.arange(0, extent, layout=gl.BlockedLayout([8], [64], [8], [0]))
        staged_input = coherent_load(
            X + (slot * B + sample_base) * K + ii,
            (ii < min(B, 4) * K) & (sample_base * K + ii < B * K),
            0,
        )
        if Gain is not None:
            gl.static_assert(B == 1)
            gain = gl.load(Gain + ii, ii < K, 0).to(gl.float32)
            staged_input = (staged_input.to(gl.float32) * Inverse * gain).to(
                gl.bfloat16
            )
        cache = gl.allocate_shared_memory(
            gl.bfloat16, [extent], gl.SwizzledSharedLayout(1, 1, 1, [0]), staged_input
        )
    else:
        gl.static_assert(Gain is None)
    acc = gl.full((SPLITS, ROWS, 16), 0, gl.float32, mma)
    unroll: gl.constexpr = (
        PROJECTION_UNROLL
        if STEPS % PROJECTION_UNROLL == 0
        else (
            PROJECTION_UNROLL // 2
            if PROJECTION_UNROLL > 1 and STEPS % (PROJECTION_UNROLL // 2) == 0
            else 1
        )
    )
    gl.static_assert(STEPS % unroll == 0)
    for outer in range(STEPS // unroll):
        for inner in gl.static_range(unroll):
            step = outer * unroll + inner
            ak = (
                gl.expand_dims(gl.expand_dims(sa, 1), 2) * (K // SPLITS)
                + step * 32
                + gl.expand_dims(gl.expand_dims(ka, 0), 0)
            )
            rows = gl.expand_dims(gl.expand_dims(row, 0), 2)
            offset = (
                (
                    ((rows // 16 * (K // 64) + ak // 64) * 2 + ak // 32 % 2) * 4
                    + ak // 8 % 4
                )
                * 16
                + rows % 16
            ) * 8 + ak % 8
            if PROJECTION_BUFFER:
                a = gl.amd.cdna3.buffer_load(W, offset)
            else:
                a = gl.load(W + offset)
            bk = (
                gl.expand_dims(gl.expand_dims(sb, 1), 2) * (K // SPLITS)
                + step * 32
                + gl.expand_dims(gl.expand_dims(kb, 0), 2)
            )
            samples = gl.expand_dims(gl.expand_dims(sample, 0), 0)
            if STAGE_INPUTS:
                index = (samples - sample_base) * K + bk
                b = gl.reshape(
                    cache.gather(gl.reshape(index, (SPLITS * 32 * 16,)), 0),
                    (SPLITS, 32, 16),
                )
            else:
                b = coherent_load(X + (slot * B + samples) * K + bk)
            a = gl.convert_layout(a, gl.DotOperandLayout(0, mma, 8))
            b = gl.convert_layout(b, gl.DotOperandLayout(1, mma, 8))
            acc = gl.amd.cdna3.mfma(a, b, acc)
    if STAGE_INPUTS:
        cache._keep_alive()
    return gl.sum(acc, 0)


@g.jit
def output_projection(
    X,
    W,
    PeerAddresses,
    Local,
    Out,
    Epochs,
    Errors,
    B: gl.constexpr,
    RANK: gl.constexpr,
    FUSED: gl.constexpr = True,
    DELAY_RANK: gl.constexpr = -1,
    DELAY: gl.constexpr = 0,
    Attention=None,
    DELAY_CTA: gl.constexpr = -1,
    Epoch=None,
    DebugIPC=None,
    WORLD: gl.constexpr = 8,
    INPUT_DOUBLE_BUFFERED: gl.constexpr = True,
    WATCHDOG_TICKS: gl.constexpr = 0,
):
    """One launch, one CTA per 64 rows and group of up to four requests.

    X contains two input versions. FUSED=False writes both-version-shaped
    local BF16 partials to Out[epoch & 1], for the two-launch baseline.
    FUSED=True writes the reduced [B,7168] result. Errors has grid*512 ints.
    All ranks must use identical ordered invocations and exclusive buffers.
    """
    pid = gl.program_id(0)
    if Epoch is None:
        epoch = gl.load(Epochs + pid) + 1
        gl.barrier()
    else:
        epoch = Epoch
    slot = epoch & 1
    row_base = pid % 112 * 64
    sample_base = pid // 112 * 4
    c = project_tile(
        X,
        W,
        slot if INPUT_DOUBLE_BUFFERED else 0,
        sample_base,
        row_base,
        B,
        12288 // WORLD,
    )
    out_layout: gl.constexpr = gl.BlockedLayout([1, 4], [1, 64], [8, 1], [1, 0])
    values = gl.convert_layout(gl.permute(c, (1, 0)), out_layout)
    if not FUSED:
        ns = gl.arange(0, 16, layout=gl.SliceLayout(1, out_layout)) + sample_base
        rs = gl.arange(0, 64, layout=gl.SliceLayout(0, out_layout)) + row_base
        gl.atomic_store(
            Out + (slot * B + ns[:, None]) * 7168 + rs[None, :],
            values,
            (ns[:, None] < B) & (ns[:, None] < sample_base + 4),
            sem="relaxed",
            scope="gpu",
        )
    else:
        shared = gl.allocate_shared_memory(
            gl.float32,
            [1024],
            gl.SwizzledSharedLayout(1, 1, 1, [0]),
            gl.reshape(values, [1024]),
        )
        layout: gl.constexpr = gl.BlockedLayout([1], [64], [8], [0])
        tid = gl.arange(0, 512, layout=layout)
        lane = tid % 64
        peer = gl.load(PeerAddresses + tid // 64, tid // 64 < WORLD, 0).to(
            gl.pointer_type(gl.uint64)
        )
        pairs: gl.constexpr = B * 3584
        if (RANK == DELAY_RANK) & ((DELAY_CTA < 0) | (pid == DELAY_CTA)):
            for _ in range(DELAY):
                gl.inline_asm_elementwise(
                    "s_sleep 1\n s_mov_b32 $0, 0",
                    "=s",
                    [],
                    dtype=gl.int32,
                    is_pure=False,
                    pack=1,
                )
        for round in gl.static_range(1 if B == 1 else 2):
            local_pair = lane + round * 64
            sample = sample_base + local_pair // 32
            row = row_base + local_pair % 32 * 2
            pair = sample * 3584 + row // 2
            valid = (sample < B) & (sample < sample_base + 4)
            low = shared.gather(local_pair * 2, 0)
            high = shared.gather(local_pair * 2 + 1, 0)
            message = pack_bf16(low, high).to(gl.uint64) | epoch.to(gl.uint64) << 32
            gl.atomic_store(
                peer + (slot * WORLD + RANK) * pairs + pair,
                message,
                mask=valid & (tid // 64 < WORLD),
                sem="relaxed",
                scope="sys",
            )
        gl.barrier()
        sample = sample_base + tid // 32
        pair = sample * 3584 + row_base // 2 + tid % 32
        active = (tid < 128) & (sample < B) & (sample < sample_base + 4)
        words, failed = poll_peers(
            Local + slot * WORLD * pairs,
            pair,
            epoch,
            active,
            pairs,
            WORLD,
            WATCHDOG_TICKS,
        )
        if DebugIPC is not None:
            dbg = DebugIPC + (pid * 512 + tid) * 9
            first = gl.load(
                dbg, active & (failed != 0), -1, cache_modifier=".cv", volatile=True
            )
            gl.atomic_store(
                dbg,
                epoch,
                active & (failed != 0) & (first == 0),
                sem="relaxed",
                scope="gpu",
            )
            for index in gl.static_range(WORLD):
                gl.atomic_store(
                    dbg + 1 + index,
                    (words[index] >> 32).to(gl.int32),
                    active & (failed != 0) & (first == 0),
                    sem="relaxed",
                    scope="gpu",
                )
        lo, hi = bf16_sum_peers(words)
        gl.atomic_store(
            Out.to(gl.pointer_type(gl.uint32)) + pair,
            pack_bf16(lo, hi),
            active,
            sem="relaxed",
            scope="gpu",
        )
        if Attention is not None:
            publish_local(Attention + pair, pack_bf16(lo, hi), epoch, active)
        gl.atomic_store(
            Errors + pid * 512 + tid,
            1,
            active & (failed != 0),
            sem="relaxed",
            scope="gpu",
        )
        shared._keep_alive()
    if Epoch is None:
        gl.atomic_store(Epochs + pid, epoch, sem="relaxed", scope="gpu")
