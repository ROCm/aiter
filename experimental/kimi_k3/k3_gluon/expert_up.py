"""Kimi FlyDSL stage 6: direct routed MXFP4/BF16 expert gate/up and SiTU.

Source: 21a3d1ee6ba2d6c0f33e3feb56e23cbf9e79c085. One 16-row task per
route; four K splits for gate and four for up. Each K32 MFMA starts from
zero, then adds its partial to the split accumulator, as in mxfp4_apply.
"""

from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from .moe_front import wait_acquire, finish_stores, projection_moe_front
from .diagnostics import cta_timestamp


def pack_expert_up(q):
    """Source native MXFP4 layout; adjacent K values occupy low/high nibbles."""
    assert q.ndim == 3 and q.shape[1] in (768, 1536) and q.shape[2] == 1792
    return (
        q.reshape(q.shape[0], q.shape[1] // 16, 16, 28, 4, 4, 4)
        .permute(0, 1, 3, 5, 2, 4, 6)
        .contiguous()
        .view(-1)
    )


@g.jit
def expert_tile(
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
):
    mma: gl.constexpr = gl.amd.AMDMFMALayout(4, [16, 16, 32], False, [8, 1, 1])
    packed: gl.constexpr = gl.BlockedLayout([1, 1, 4], [1, 16, 4], [8, 1, 1], [2, 1, 0])
    unpacked: gl.constexpr = gl.BlockedLayout(
        [1, 1, 8], [1, 16, 4], [8, 1, 1], [2, 1, 0]
    )
    bl: gl.constexpr = gl.BlockedLayout([1, 8, 1], [1, 4, 16], [8, 1, 1], [2, 1, 0])
    wave = gl.arange(0, 8, layout=gl.SliceLayout(1, gl.SliceLayout(2, packed)))
    r = gl.arange(0, 16, layout=gl.SliceLayout(0, gl.SliceLayout(2, packed)))
    kb = gl.arange(0, 16, layout=gl.SliceLayout(0, gl.SliceLayout(0, packed)))
    sw = gl.arange(0, 8, layout=gl.SliceLayout(1, gl.SliceLayout(2, unpacked)))
    sr = gl.arange(0, 16, layout=gl.SliceLayout(0, gl.SliceLayout(2, unpacked)))
    sk = gl.arange(0, 32, layout=gl.SliceLayout(0, gl.SliceLayout(0, unpacked)))
    bw = gl.arange(0, 8, layout=gl.SliceLayout(1, gl.SliceLayout(2, bl)))
    bk = gl.arange(0, 32, layout=gl.SliceLayout(0, gl.SliceLayout(2, bl)))
    bn = gl.arange(0, 16, layout=gl.SliceLayout(0, gl.SliceLayout(0, bl)))
    rows = wave[:, None, None] // 4 * INTER + tile * 16 + r[None, :, None]
    scale_rows = sw[:, None, None] // 4 * INTER + tile * 16 + sr[None, :, None]
    acc = gl.full((8, 16, 16), 0.0, gl.float32, mma)
    for step in range(28):
        k = (
            wave[:, None, None] % 4 * 896
            + step * 32
            + gl.expand_dims(gl.expand_dims(kb, 0), 0) * 2
        )
        offset = (
            (((rows // 16 * 28 + k // 128) * 4 + (k % 32) // 8) * 16 + rows % 16) * 4
            + (k % 128) // 32
        ) * 4 + (k % 8) // 2
        if AITER_LAYOUT:
            offset = (
                ((rows // 16 * 28 + k // 128) * 4 + (k % 128) // 32) * 16 + rows % 16
            ) * 16 + (k % 32) // 2
        raw = gl.load(W + expert.to(gl.int64) * ((2 * INTER) * 1792) + offset)
        scale_k = (
            sw[:, None, None] % 4 * 896
            + step * 32
            + gl.expand_dims(gl.expand_dims(sk, 0), 0)
        )
        if AITER_LAYOUT:
            group = scale_k // 32
            soff = ((scale_rows // 32 * 14 + group // 8) * 4 + group % 4) * 64
            soff += (
                (scale_rows % 16) * 4 + ((group // 4) % 2) * 2 + (scale_rows // 16) % 2
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
            + step * 32
            + bk[None, :, None]
            + gl.expand_dims(gl.expand_dims(bn, 0), 0) * 0
        )
        b = gl.convert_layout(
            gl.load(Latent + sample * 3584 + input_k), gl.DotOperandLayout(1, mma, 8)
        )
        partial = gl.amd.cdna3.mfma(a, b, gl.full((8, 16, 16), 0.0, gl.float32, mma))
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
def run_experts(
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
                expert_tile(
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
                )
            else:
                gl.store(Errors + pid * 512, 1)


@g.jit
def expert_up_only(
    Latent,
    IDs,
    LatentReady,
    RouteReady,
    W,
    Scale,
    Out,
    Ready,
    DebugGU,
    Epochs,
    Errors,
    B: gl.constexpr,
    STORE_GU: gl.constexpr = False,
    DELAY_EXPERT: gl.constexpr = 0,
    WORLD: gl.constexpr = 8,
):
    pid = gl.program_id(0)
    epoch = gl.load(Epochs + pid) + 1
    gl.barrier()
    run_experts(
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
        B,
        STORE_GU,
        DELAY_EXPERT,
        WORLD,
    )
    gl.store(Epochs + pid, epoch)


@g.jit
def projection_expert_up(
    X,
    W,
    Peers,
    Local,
    Delta,
    Attention,
    Prefix,
    Blocks,
    Norm,
    Query,
    OutNorm,
    Updated,
    Moe,
    Quant,
    Scales,
    Stats,
    Ready,
    Epochs,
    Errors,
    RouterW,
    Bias,
    LatentW,
    LatentScale,
    SharedW,
    SharedScale,
    Scores,
    Latent,
    GU,
    IDs,
    Weights,
    Mid,
    RouterReady,
    LatentReady,
    SharedReady,
    RouteReady,
    MidReady,
    ExpertW,
    ExpertScale,
    ExpertMid,
    ExpertReady,
    DebugGU,
    B: gl.constexpr,
    RANK: gl.constexpr,
    N_BLOCKS: gl.constexpr,
    BOUNDARY: gl.constexpr = False,
    DELAY: gl.constexpr = 0,
    DELAY_EXPERT: gl.constexpr = 0,
    Trace=None,
    TimeTrace=None,
    DebugIPC=None,
    WORLD: gl.constexpr = 8,
    Progress=None,
    PROGRESS_EPOCH: gl.constexpr = 0,
):
    pid = gl.program_id(0)
    epoch = gl.load(Epochs + pid) + 1
    # All waves must snapshot the counter before any nested stage can run.
    # Nested stages consume this epoch and never advance it themselves.
    gl.barrier()
    if Progress is not None:
        cta_timestamp(Progress, epoch, pid, 0, PROGRESS_EPOCH)
    if TimeTrace is not None:
        trace_clock(TimeTrace, epoch, pid, 0)
    projection_moe_front(
        X,
        W,
        Peers,
        Local,
        Delta,
        Attention,
        Prefix,
        Blocks,
        Norm,
        Query,
        OutNorm,
        Updated,
        Moe,
        Quant,
        Scales,
        Stats,
        Ready,
        Epochs,
        Errors,
        RouterW,
        Bias,
        LatentW,
        LatentScale,
        SharedW,
        SharedScale,
        Scores,
        Latent,
        GU,
        IDs,
        Weights,
        Mid,
        RouterReady,
        LatentReady,
        SharedReady,
        RouteReady,
        MidReady,
        B,
        RANK,
        N_BLOCKS,
        BOUNDARY,
        DELAY,
        Epoch=epoch,
        DebugIPC=DebugIPC,
        WORLD=WORLD,
    )
    if Progress is not None:
        cta_timestamp(Progress, epoch, pid, 1, PROGRESS_EPOCH)
    if Trace is not None:
        trace_errors(Errors, Trace, epoch, pid)
    if TimeTrace is not None:
        trace_clock(TimeTrace, epoch, pid, 1)
    run_experts(
        Latent,
        IDs,
        LatentReady,
        RouteReady,
        ExpertW,
        ExpertScale,
        ExpertMid,
        ExpertReady,
        DebugGU,
        Errors,
        epoch,
        pid,
        B,
        False,
        DELAY_EXPERT,
        WORLD,
    )
    if Progress is not None:
        cta_timestamp(Progress, epoch, pid, 2, PROGRESS_EPOCH)
    if Trace is not None:
        trace_errors(Errors, Trace + 256, epoch, pid)
    if TimeTrace is not None:
        trace_clock(TimeTrace, epoch, pid, 2)
    gl.store(Epochs + pid, epoch)


@g.jit
def trace_errors(Errors, Trace, epoch, pid):
    finish_stores()
    idx = gl.arange(0, 512, layout=gl.BlockedLayout([1], [64], [8], [0]))
    failed = gl.max(
        gl.load(Errors + pid * 512 + idx, cache_modifier=".cv", volatile=True), 0
    )
    if failed != 0:
        gl.atomic_cas(Trace + pid, 0, epoch, sem="relaxed", scope="gpu")


@g.jit
def trace_clock(Trace, epoch, pid, stage: gl.constexpr):
    if pid == 0:
        ticks = gl.inline_asm_elementwise(
            "s_memrealtime $0\n s_waitcnt lgkmcnt(0)",
            "=s,~{memory}",
            [],
            gl.uint64,
            False,
            1,
        )
        gl.store(Trace + (epoch % 32768) * 3 + stage, ticks)
