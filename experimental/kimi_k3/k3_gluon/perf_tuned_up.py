"""Experimental K-fragment tiling for the first performance regression.

Derived from expert_up.expert_tile; production defaults remain unchanged.
FlyDSL loads K128 weight fragments and reuses them across four K32 MFMAs.
"""

from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from .moe_front import finish_stores, wait_acquire


@g.jit
def expert_tile_tuned(
    Latent,
    W,
    Scale,
    Out,
    Ready,
    DebugGU,
    expert,
    sample,
    route,
    tile,
    epoch,
    STORE_GU: gl.constexpr,
    INTER: gl.constexpr,
    AITER_LAYOUT: gl.constexpr = False,
    CHUNK: gl.constexpr = 32,
    ALIGNED: gl.constexpr = False,
    UNROLL: gl.constexpr = 1,
    LOAD_VEC: gl.constexpr = 4,
):
    mma: gl.constexpr = gl.amd.AMDMFMALayout(4, [16, 16, 32], False, [8, 1, 1])
    packed: gl.constexpr = gl.BlockedLayout(
        [1, 1, LOAD_VEC], [1, 16, 4], [8, 1, 1], [2, 1, 0]
    )
    unpacked: gl.constexpr = gl.BlockedLayout(
        [1, 1, LOAD_VEC * 2], [1, 16, 4], [8, 1, 1], [2, 1, 0]
    )
    bl: gl.constexpr = gl.BlockedLayout([1, 8, 1], [1, 4, 16], [8, 1, 1], [2, 1, 0])
    wave = gl.arange(0, 8, layout=gl.SliceLayout(1, gl.SliceLayout(2, packed)))
    r = gl.arange(0, 16, layout=gl.SliceLayout(0, gl.SliceLayout(2, packed)))
    kb = gl.arange(0, CHUNK // 2, layout=gl.SliceLayout(0, gl.SliceLayout(0, packed)))
    sw = gl.arange(0, 8, layout=gl.SliceLayout(1, gl.SliceLayout(2, unpacked)))
    sr = gl.arange(0, 16, layout=gl.SliceLayout(0, gl.SliceLayout(2, unpacked)))
    sk = gl.arange(0, CHUNK, layout=gl.SliceLayout(0, gl.SliceLayout(0, unpacked)))
    bw = gl.arange(0, 8, layout=gl.SliceLayout(1, gl.SliceLayout(2, bl)))
    bk = gl.arange(0, CHUNK, layout=gl.SliceLayout(0, gl.SliceLayout(2, bl)))
    bn = gl.arange(0, 16, layout=gl.SliceLayout(0, gl.SliceLayout(0, bl)))
    rows = wave[:, None, None] // 4 * INTER + tile * 16 + r[None, :, None]
    scale_rows = sw[:, None, None] // 4 * INTER + tile * 16 + sr[None, :, None]
    acc = gl.full((8, 16, 16), 0.0, gl.float32, mma)
    gl.static_assert((896 // CHUNK) % UNROLL == 0)
    for outer in range(896 // CHUNK // UNROLL):
        for inner in gl.static_range(UNROLL):
            step = outer * UNROLL + inner
            k = (
                wave[:, None, None] % 4 * 896
                + step * CHUNK
                + gl.expand_dims(gl.expand_dims(kb, 0), 0) * 2
            )
            offset = (
                (((rows // 16 * 28 + k // 128) * 4 + (k % 32) // 8) * 16 + rows % 16)
                * 4
                + (k % 128) // 32
            ) * 4 + (k % 8) // 2
            if AITER_LAYOUT:
                offset = (
                    ((rows // 16 * 28 + k // 128) * 4 + (k % 128) // 32) * 16
                    + rows % 16
                ) * 16 + (k % 32) // 2
            if ALIGNED:
                offset = gl.max_contiguous(
                    gl.multiple_of(offset, (1, 1, LOAD_VEC)), (1, 1, LOAD_VEC)
                )
            raw = gl.load(W + expert.to(gl.int64) * ((2 * INTER) * 1792) + offset)
            scale_k = (
                sw[:, None, None] % 4 * 896
                + step * CHUNK
                + gl.expand_dims(gl.expand_dims(sk, 0), 0)
            )
            if AITER_LAYOUT:
                group = scale_k // 32
                soff = ((scale_rows // 32 * 14 + group // 8) * 4 + group % 4) * 64
                soff += (
                    (scale_rows % 16) * 4
                    + ((group // 4) % 2) * 2
                    + (scale_rows // 16) % 2
                )
            else:
                soff = scale_rows * 112 + scale_k // 32
            scales = gl.load(Scale + expert.to(gl.int64) * ((2 * INTER) * 112) + soff)
            a = gl.amd.cdna4.scaled_upcast(raw, scales, gl.bfloat16, axis=2)
            # Source constructs scale FP32 by exponent-byte << 23: byte zero is 0.
            a = gl.where(scales == 0, 0.0, a).to(gl.bfloat16)
            a = gl.convert_layout(a, gl.DotOperandLayout(0, mma, 8))
            input_k = (
                bw[:, None, None] % 4 * 896
                + step * CHUNK
                + bk[None, :, None]
                + gl.expand_dims(gl.expand_dims(bn, 0), 0) * 0
            )
            b = gl.convert_layout(
                gl.load(Latent + sample * 3584 + input_k),
                gl.DotOperandLayout(1, mma, 8),
            )
            partial = gl.amd.cdna3.mfma(
                a, b, gl.full((8, 16, 16), 0.0, gl.float32, mma)
            )
            acc += partial
    plain: gl.constexpr = gl.BlockedLayout([1, 1, 1], [1, 16, 4], [8, 1, 1], [1, 2, 0])
    shared = gl.allocate_shared_memory(
        gl.float32,
        [2048],
        gl.SwizzledSharedLayout(1, 1, 1, [0]),
        gl.reshape(gl.convert_layout(acc, plain), (2048,)),
    )
    layout: gl.constexpr = gl.BlockedLayout([1], [64], [8], [0])
    row = gl.arange(0, 16, layout=layout)
    gate = gl.full((16,), 0.0, gl.float32, layout)
    up = gl.full((16,), 0.0, gl.float32, layout)
    for split in gl.static_range(4):
        gate += shared.gather(split * 256 + row * 16, 0)
        up += shared.gather((split + 4) * 256 + row * 16, 0)
    gate = gate.to(gl.bfloat16).to(gl.float32)
    up = up.to(gl.bfloat16).to(gl.float32)
    if STORE_GU:
        gl.store(DebugGU + route * (2 * INTER) + tile * 16 + row, gate.to(gl.bfloat16))
        gl.store(
            DebugGU + route * (2 * INTER) + INTER + tile * 16 + row, up.to(gl.bfloat16)
        )
    tg = 2.0 / (1.0 + gl.exp(-0.5 * gate)) - 1.0
    tu = 2.0 / (1.0 + gl.exp(-0.08 * up)) - 1.0
    mid = 4.0 * tg * (1.0 / (1.0 + gl.exp(-gate))) * 25.0 * tu
    gl.store(Out + route * INTER + tile * 16 + row, mid.to(gl.bfloat16))
    shared._keep_alive()
    finish_stores()
    gl.atomic_store(
        Ready + route * (INTER // 16) + tile,
        (epoch.to(gl.uint64) << 32) | 1,
        sem="release",
        scope="gpu",
    )


@g.jit
def run_experts_tuned(
    Latent,
    IDs,
    LatentReady,
    RouteReady,
    W,
    Scale,
    Out,
    Ready,
    DebugGU,
    Errors,
    epoch,
    pid,
    B: gl.constexpr,
    STORE_GU: gl.constexpr = False,
    DELAY_EXPERT: gl.constexpr = 0,
    WORLD: gl.constexpr = 8,
    LATENT_TILES: gl.constexpr = 28,
    WATCHDOG_TICKS: gl.constexpr = 0,
    AITER_LAYOUT: gl.constexpr = False,
):
    layout: gl.constexpr = gl.BlockedLayout([1], [64], [8], [0])
    idx = gl.arange(0, 32 if LATENT_TILES == 28 else 64, layout=layout)
    for task in range(pid, B * (3072 // WORLD), 256):
        sample = task // (3072 // WORLD)
        route = task // (192 // WORLD)
        tile = task % (192 // WORLD)
        bad_route = wait_acquire(
            RouteReady + sample + idx * 0,
            idx == 0,
            epoch,
            Errors,
            pid * 512 + idx,
            WATCHDOG_TICKS,
        )
        bad_latent = wait_acquire(
            LatentReady + sample * LATENT_TILES + idx,
            idx < LATENT_TILES,
            epoch,
            Errors,
            pid * 512 + idx,
            WATCHDOG_TICKS,
        )
        # A failed dependency invalidates this task. Do not read unpublished
        # routing data or use an uninitialized expert ID as a weight address.
        if gl.sum((bad_route | bad_latent).to(gl.int32), 0) == 0:
            expert = gl.load(IDs + route)
            if (expert >= 0) & (expert < 896):
                if DELAY_EXPERT > 0:
                    if task == B * (3072 // WORLD) - 1:
                        for _ in range(DELAY_EXPERT):
                            gl.inline_asm_elementwise(
                                "s_sleep 1\n s_mov_b32 $0, 0",
                                "=s",
                                [],
                                gl.int32,
                                False,
                                1,
                            )
                expert_tile_tuned(
                    Latent,
                    W,
                    Scale,
                    Out,
                    Ready,
                    DebugGU,
                    expert,
                    sample,
                    route,
                    tile,
                    epoch,
                    STORE_GU,
                    3072 // WORLD,
                    AITER_LAYOUT,
                    128,
                    True,
                    7,
                    4,
                )
            else:
                gl.store(Errors + pid * 512, 1)
