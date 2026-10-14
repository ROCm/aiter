# Frozen JIT snapshot: retain original imports and tuning overrides.
# ruff: noqa: E402
from .coherent_memory import coherent_load

"""Batch-1 front: 32 rows and four K splits per CTA, versus 64 rows/two splits.

This doubles producer CTAs to 188 at TP8 and shortens their serial MFMA loops.
The router retains FP32 logits; projections retain BF16 weights and outputs.
"""
import triton
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from .projection import STAGE_INPUTS, PROJECTION_UNROLL, PROJECTION_BUFFER
from .moe_complete import dense_input_ready, wait_ready, publish
from .moe_front import select_routes, finish_stores


@g.jit
def project_tile32(X, W, slot, sample_base, row_base, B: gl.constexpr, K: gl.constexpr):
    mma: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 32], transposed=False, warps_per_cta=[4, 2, 1]
    )
    al: gl.constexpr = gl.BlockedLayout([1, 1, 8], [1, 16, 4], [4, 2, 1], [1, 2, 0])
    bl: gl.constexpr = gl.BlockedLayout([1, 8, 1], [1, 4, 16], [4, 1, 2], [2, 1, 0])
    sa = gl.arange(0, 4, layout=gl.SliceLayout(1, gl.SliceLayout(2, al)))
    ra = gl.arange(0, 32, layout=gl.SliceLayout(0, gl.SliceLayout(2, al)))
    ka = gl.arange(0, 32, layout=gl.SliceLayout(0, gl.SliceLayout(0, al)))
    sb = gl.arange(0, 4, layout=gl.SliceLayout(1, gl.SliceLayout(2, bl)))
    kb = gl.arange(0, 32, layout=gl.SliceLayout(0, gl.SliceLayout(2, bl)))
    nb = gl.arange(0, 16, layout=gl.SliceLayout(0, gl.SliceLayout(0, bl)))
    row = row_base + ra
    sample = gl.minimum(sample_base + nb, gl.minimum(sample_base + 3, B - 1))
    if STAGE_INPUTS:
        extent: gl.constexpr = triton.next_power_of_2(min(B, 4) * K)
        ii = gl.arange(0, extent, layout=gl.BlockedLayout([8], [64], [8], [0]))
        cache = gl.allocate_shared_memory(
            gl.bfloat16,
            [extent],
            gl.SwizzledSharedLayout(1, 1, 1, [0]),
            coherent_load(
                X + (slot * B + sample_base) * K + ii,
                (ii < min(B, 4) * K) & (sample_base * K + ii < B * K),
                0,
            ),
        )
    acc = gl.full((4, 32, 16), 0, gl.float32, mma)
    unroll: gl.constexpr = (
        PROJECTION_UNROLL
        if K // 128 % PROJECTION_UNROLL == 0
        else PROJECTION_UNROLL // 2
    )
    gl.static_assert(K // 128 % unroll == 0)
    for outer in range(K // 128 // unroll):
        for inner in gl.static_range(unroll):
            step = outer * unroll + inner
            ak = (
                gl.expand_dims(gl.expand_dims(sa, 1), 2) * (K // 4)
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
                gl.expand_dims(gl.expand_dims(sb, 1), 2) * (K // 4)
                + step * 32
                + gl.expand_dims(gl.expand_dims(kb, 0), 2)
            )
            samples = gl.expand_dims(gl.expand_dims(sample, 0), 0)
            if STAGE_INPUTS:
                index = (samples - sample_base) * K + bk
                b = gl.reshape(
                    cache.gather(gl.reshape(index, (4 * 32 * 16,)), 0), (4, 32, 16)
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
def dense_front32(
    X,
    Router,
    Bias,
    Down,
    Shared,
    Scores,
    Latent,
    GU,
    IDs,
    Weights,
    SharedMid,
    RouterReady,
    LatentReady,
    SharedReady,
    RouteReady,
    SharedMidReady,
    Errors,
    epoch,
    pid,
    B: gl.constexpr,
    WORLD: gl.constexpr,
    InputReady=None,
    WATCHDOG_TICKS: gl.constexpr = 0,
):
    gl.static_assert(B == 1 and WORLD == 8)
    groups: gl.constexpr = (B + 3) // 4
    shared_tiles: gl.constexpr = 384 // WORLD
    layout: gl.constexpr = gl.BlockedLayout([1, 4], [1, 64], [8, 1], [1, 0])
    rr = gl.arange(0, 32, layout=gl.SliceLayout(0, layout))
    ss = gl.arange(0, 16, layout=gl.SliceLayout(1, layout))
    if pid < groups * (28 + 112 + shared_tiles):
        if pid < groups * 28:
            base = pid // 28 * 4
            tile = pid % 28
            dense_input_ready(InputReady, base, epoch, Errors, pid, B, WATCHDOG_TICKS)
            value = project_tile32(X, Router, 0, base, tile * 32, B, 7168)
            value = gl.convert_layout(gl.permute(value, (1, 0)), layout)
            score = 1.0 / (1.0 + gl.exp(-value))
            gl.atomic_store(
                Scores + (base + ss[:, None]) * 896 + tile * 32 + rr[None, :],
                score,
                (base + ss[:, None] < B) & (ss[:, None] < 4),
                sem="relaxed",
                scope="gpu",
            )
            finish_stores()
            fl: gl.constexpr = gl.BlockedLayout([1], [64], [8], [0])
            idx = gl.arange(0, 8, layout=fl)
            router_sample = base + idx // 2
            gl.atomic_store(
                RouterReady + router_sample * 56 + tile * 2 + idx % 2,
                epoch.to(gl.uint64) << 32 | 1,
                mask=router_sample < B,
                sem="relaxed",
                scope="gpu",
            )
        elif pid < groups * 140:
            task = pid - groups * 28
            base = task // 112 * 4
            tile = task % 112
            dense_input_ready(InputReady, base, epoch, Errors, pid, B, WATCHDOG_TICKS)
            value = project_tile32(X, Down, 0, base, tile * 32, B, 7168)
            value = gl.convert_layout(gl.permute(value, (1, 0)), layout)
            gl.atomic_store(
                Latent + (base + ss[:, None]) * 3584 + tile * 32 + rr[None, :],
                value.to(gl.bfloat16),
                (base + ss[:, None] < B) & (ss[:, None] < 4),
                sem="relaxed",
                scope="gpu",
            )
            finish_stores()
            fl: gl.constexpr = gl.BlockedLayout([1], [64], [8], [0])
            sample = base + gl.arange(0, 4, layout=fl)
            gl.atomic_store(
                LatentReady + sample * 112 + tile,
                epoch.to(gl.uint64) << 32 | 1,
                mask=sample < B,
                sem="relaxed",
                scope="gpu",
            )
        else:
            task = pid - groups * 140
            base = task // shared_tiles * 4
            tile = task % shared_tiles
            dense_input_ready(InputReady, base, epoch, Errors, pid, B, WATCHDOG_TICKS)
            value = project_tile32(X, Shared, 0, base, tile * 32, B, 7168)
            value = gl.convert_layout(gl.permute(value, (1, 0)), layout)
            gl.atomic_store(
                GU + (base + ss[:, None]) * (12288 // WORLD) + tile * 32 + rr[None, :],
                value.to(gl.bfloat16),
                (base + ss[:, None] < B) & (ss[:, None] < 4),
                sem="relaxed",
                scope="gpu",
            )
            finish_stores()
            fl: gl.constexpr = gl.BlockedLayout([1], [64], [8], [0])
            sample = base + gl.arange(0, 4, layout=fl)
            gl.atomic_store(
                SharedReady + sample * shared_tiles + tile,
                epoch.to(gl.uint64) << 32 | 1,
                mask=sample < B,
                sem="relaxed",
                scope="gpu",
            )
    if pid == 0:
        select_routes(
            Scores,
            Bias,
            RouterReady,
            IDs,
            Weights,
            RouteReady,
            Errors,
            epoch,
            B,
            WATCHDOG_TICKS,
        )
    if pid < B:
        fl: gl.constexpr = gl.BlockedLayout([1], [64], [8], [0])
        idx = gl.arange(0, 64, layout=fl)
        bad = wait_ready(
            SharedReady,
            pid * shared_tiles + idx,
            idx < shared_tiles,
            epoch,
            Errors,
            pid,
            WATCHDOG_TICKS,
        )
        if not bad:
            row = gl.arange(0, 2048, layout=fl)
            gate = coherent_load(
                GU + pid * (12288 // WORLD) + row, row < 6144 // WORLD, 0
            ).to(gl.float32)
            up = coherent_load(
                GU + pid * (12288 // WORLD) + 6144 // WORLD + row,
                row < 6144 // WORLD,
                0,
            ).to(gl.float32)
            val = 4.0 * (2.0 / (1.0 + gl.exp(-0.5 * gate)) - 1.0)
            val *= (
                1.0
                / (1.0 + gl.exp(-gate))
                * 25.0
                * (2.0 / (1.0 + gl.exp(-0.08 * up)) - 1.0)
            )
            gl.atomic_store(
                SharedMid + pid * (6144 // WORLD) + row,
                val.to(gl.bfloat16),
                row < 6144 // WORLD,
                sem="relaxed",
                scope="gpu",
            )
            publish(SharedMidReady, pid, epoch)
