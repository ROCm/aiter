# Frozen JIT snapshot: retain original imports and tuning overrides.
# ruff: noqa: E402
from .coherent_memory import coherent_load

"""Kimi stage 5: BF16 router, native MXFP8 dense projections, top-16, SiTU.

Reference: FlyDSL 21a3d1e K3 kernel stage 5 and monokernel/packing.py.
Dense tiles use all eight waves (128 rows) instead of 3/4/6 active waves.
Ready flags publish separate payloads: poll coherently, then acquire once.
"""
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from .primitives import poll_local
from .attnres import projection_attnres

PAIR_TOPK = gl.constexpr(False)
LOAD_SELECTED_BIAS = gl.constexpr(False)


@g.jit
def max_score_id(score_a, id_a, score_b, id_b):
    take_b = (score_b > score_a) | (score_b == score_a) & (id_b < id_a)
    return (gl.where(take_b, score_b, score_a), gl.where(take_b, id_b, id_a))


def pack_router(w):
    assert w.shape == (896, 7168)
    return (
        w.reshape(56, 16, 112, 2, 4, 8).permute(0, 2, 3, 4, 1, 5).contiguous().view(-1)
    )


def pack_mx(q, scales):
    n, k = q.shape
    assert n in (3584, 1536, 3072) and k == 7168 and (scales.shape == (n, 224))
    weight = (
        q.reshape(n // 16, 16, 112, 4, 16).permute(0, 2, 3, 1, 4).contiguous().view(-1)
    )
    scale = (
        scales.reshape(n // 32, 2, 16, 28, 2, 4)
        .permute(0, 3, 5, 2, 4, 1)
        .contiguous()
        .view(-1)
    )
    return (weight, scale)


@g.jit
def wait_acquire(
    Ptrs, active, epoch, Errors, error_indices, WATCHDOG_TICKS: gl.constexpr = 0
):
    _, failed = poll_local(Ptrs, epoch, active, WATCHDOG_TICKS)
    observed = gl.atomic_load(Ptrs, mask=active, sem="relaxed", scope="gpu")
    bad = active & ((failed != 0) | (observed >> 32 != epoch.to(gl.uint64)))
    gl.atomic_store(Errors + error_indices, 1, bad, sem="relaxed", scope="gpu")
    gl.barrier()
    return bad


@g.jit
def wait_inputs(Ready, sample_base, epoch, Errors, pid, B: gl.constexpr):
    layout: gl.constexpr = gl.BlockedLayout([1], [64], [8], [0])
    idx = gl.arange(0, 16, layout=layout)
    sample = sample_base + idx // 4
    wait_acquire(
        Ready + sample * 4 + idx % 4, sample < B, epoch, Errors, pid * 512 + idx
    )


@g.jit
def finish_stores():
    gl.inline_asm_elementwise(
        "s_waitcnt vmcnt(0)\n s_waitcnt lgkmcnt(0)\n s_mov_b32 $0, 0",
        "=s,~{memory}",
        [],
        dtype=gl.int32,
        is_pure=False,
        pack=1,
    )
    gl.barrier()


@g.jit
def publish_rows(Ready, sample_base, tile, epoch, B: gl.constexpr, TILES: gl.constexpr):
    finish_stores()
    layout: gl.constexpr = gl.BlockedLayout([1], [64], [8], [0])
    sample = sample_base + gl.arange(0, 4, layout=layout)
    gl.atomic_store(
        Ready + sample * TILES + tile,
        epoch.to(gl.uint64) << 32 | 1,
        mask=sample < B,
        sem="relaxed",
        scope="gpu",
    )


@g.jit
def router_tile(Moe, W, Scores, Ready, sample_base, tile, epoch, B: gl.constexpr):
    mma: gl.constexpr = gl.amd.AMDMFMALayout(4, [16, 16, 32], False, [8, 1, 1])
    al: gl.constexpr = gl.BlockedLayout([1, 1, 8], [1, 16, 4], [8, 1, 1], [1, 2, 0])
    bl: gl.constexpr = gl.BlockedLayout([1, 8, 1], [1, 4, 16], [8, 1, 1], [2, 1, 0])
    sa = gl.arange(0, 8, layout=gl.SliceLayout(1, gl.SliceLayout(2, al)))
    ra = gl.arange(0, 16, layout=gl.SliceLayout(0, gl.SliceLayout(2, al)))
    ka = gl.arange(0, 32, layout=gl.SliceLayout(0, gl.SliceLayout(0, al)))
    sb = gl.arange(0, 8, layout=gl.SliceLayout(1, gl.SliceLayout(2, bl)))
    kb = gl.arange(0, 32, layout=gl.SliceLayout(0, gl.SliceLayout(2, bl)))
    nb = gl.arange(0, 16, layout=gl.SliceLayout(0, gl.SliceLayout(0, bl)))
    sample = gl.minimum(sample_base + nb, gl.minimum(sample_base + 3, B - 1))
    rows = gl.expand_dims(gl.expand_dims(ra + tile * 16, 0), 2)
    acc = gl.full((8, 16, 16), 0.0, gl.float32, mma)
    for step in range(28):
        ak = (
            gl.expand_dims(gl.expand_dims(sa, 1), 2) * 896
            + step * 32
            + gl.expand_dims(gl.expand_dims(ka, 0), 0)
        )
        offset = (
            (((rows // 16 * 112 + ak // 64) * 2 + ak // 32 % 2) * 4 + ak // 8 % 4) * 16
            + rows % 16
        ) * 8 + ak % 8
        a = gl.convert_layout(gl.load(W + offset), gl.DotOperandLayout(0, mma, 8))
        bk = (
            gl.expand_dims(gl.expand_dims(sb, 1), 2) * 896
            + step * 32
            + gl.expand_dims(gl.expand_dims(kb, 0), 2)
        )
        samples = gl.expand_dims(gl.expand_dims(sample, 0), 0)
        b = gl.convert_layout(
            coherent_load(Moe + samples * 7168 + bk), gl.DotOperandLayout(1, mma, 8)
        )
        acc = gl.amd.cdna3.mfma(a, b, acc)
    plain: gl.constexpr = gl.BlockedLayout([1, 1, 1], [1, 16, 4], [8, 1, 1], [1, 2, 0])
    flat_acc = gl.reshape(gl.convert_layout(acc, plain), (2048,))
    shared = gl.allocate_shared_memory(
        gl.float32, [2048], gl.SwizzledSharedLayout(1, 1, 1, [0]), flat_acc
    )
    out_layout: gl.constexpr = gl.BlockedLayout([1, 1], [16, 4], [1, 8], [0, 1])
    flat_layout: gl.constexpr = gl.BlockedLayout([1], [64], [8], [0])
    indices = gl.arange(0, 256, layout=flat_layout)
    value = gl.full((256,), 0.0, gl.float32, flat_layout)
    for split in gl.static_range(8):
        value += shared.gather(split * 256 + indices, 0)
    value = gl.convert_layout(gl.reshape(value, (16, 16)), out_layout)
    value = value.to(gl.bfloat16).to(gl.float32)
    score = 1.0 / (1.0 + gl.exp(-value))
    row = tile * 16 + gl.arange(0, 16, layout=gl.SliceLayout(1, out_layout))
    sample = sample_base + gl.arange(0, 16, layout=gl.SliceLayout(0, out_layout))
    gl.atomic_store(
        Scores + sample[None, :] * 896 + row[:, None],
        score,
        (sample[None, :] < B) & (sample[None, :] < sample_base + 4),
        sem="relaxed",
        scope="gpu",
    )
    shared._keep_alive()
    publish_rows(Ready, sample_base, tile, epoch, B, 56)


@g.jit
def dense_tile(
    Quant,
    XScale,
    W,
    WScale,
    Out,
    Ready,
    sample_base,
    tile,
    epoch,
    B: gl.constexpr,
    N: gl.constexpr,
):
    mma: gl.constexpr = gl.amd.AMDMFMALayout(4, [16, 16, 128], False, [8, 1])
    ad: gl.constexpr = gl.DotOperandLayout(0, mma, 16)
    bd: gl.constexpr = gl.DotOperandLayout(1, mma, 16)
    al: gl.constexpr = gl.BlockedLayout([1, 16], [16, 4], [8, 1], [0, 1])
    bl: gl.constexpr = gl.BlockedLayout([16, 1], [4, 16], [1, 8], [1, 0])
    asl: gl.constexpr = gl.amd.cdna4.get_mfma_scale_layout(ad, [128, 4])
    bsl: gl.constexpr = gl.amd.cdna4.get_mfma_scale_layout(bd, [16, 4])
    row = tile * 128 + gl.arange(0, 128, layout=gl.SliceLayout(1, al))
    ak = gl.arange(0, 128, layout=gl.SliceLayout(0, al))
    bk = gl.arange(0, 128, layout=gl.SliceLayout(1, bl))
    sample = sample_base + gl.arange(0, 16, layout=gl.SliceLayout(0, bl))
    scale_row = tile * 128 + gl.arange(0, 128, layout=gl.SliceLayout(1, asl))
    ag = gl.arange(0, 4, layout=gl.SliceLayout(0, asl))
    scale_sample = sample_base + gl.arange(0, 16, layout=gl.SliceLayout(1, bsl))
    bg = gl.arange(0, 4, layout=gl.SliceLayout(0, bsl))
    acc = gl.full((128, 16), 0.0, gl.float32, mma)
    for step in range(56):
        k = step * 128 + ak
        offset = (
            ((row[:, None] // 16 * 112 + k[None, :] // 64) * 4 + k[None, :] // 16 % 4)
            * 16
            + row[:, None] % 16
        ) * 16 + k[None, :] % 16
        a = gl.convert_layout(gl.load(W + offset), ad)
        b = coherent_load(
            Quant + sample[None, :] * 7168 + step * 128 + bk[:, None],
            (sample[None, :] < B) & (sample[None, :] < sample_base + 4),
            0,
        )
        b = gl.convert_layout(b, bd)
        group_a = step * 4 + ag
        aoff = (
            (scale_row[:, None] // 32 * 28 + group_a[None, :] // 8) * 4
            + group_a[None, :] % 4
        ) * 64 + scale_row[:, None] % 16 * 4
        aoff += group_a[None, :] // 4 % 2 * 2 + scale_row[:, None] // 16 % 2
        ascale = gl.load(WScale + aoff)
        group_b = step * 4 + bg
        boff = group_b[None, :] // 8 * 256 + group_b[None, :] % 4 * 64
        boff += scale_sample[:, None] * 4 + group_b[None, :] // 4 % 2 * 2
        bscale = coherent_load(
            XScale + boff,
            (scale_sample[:, None] < B) & (scale_sample[:, None] < sample_base + 4),
            127,
        )
        acc = gl.amd.cdna4.mfma_scaled(a, ascale, "e4m3", b, bscale, "e4m3", acc)
    layout: gl.constexpr = gl.BlockedLayout([1, 1], [16, 4], [8, 1], [0, 1])
    values = gl.convert_layout(acc, layout).to(gl.bfloat16)
    rows = tile * 128 + gl.arange(0, 128, layout=gl.SliceLayout(1, layout))
    samples = sample_base + gl.arange(0, 16, layout=gl.SliceLayout(0, layout))
    gl.atomic_store(
        Out + samples[None, :] * N + rows[:, None],
        values,
        (samples[None, :] < B) & (samples[None, :] < sample_base + 4),
        sem="relaxed",
        scope="gpu",
    )
    publish_rows(Ready, sample_base, tile, epoch, B, N // 128)


@g.jit
def select_routes(
    Scores,
    Bias,
    RouterReady,
    IDs,
    Weights,
    RouteReady,
    Errors,
    epoch,
    B: gl.constexpr,
    WATCHDOG_TICKS: gl.constexpr = 0,
):
    layout: gl.constexpr = gl.BlockedLayout([1, 1], [1, 64], [8, 1], [1, 0])
    sample = gl.arange(0, 8, layout=gl.SliceLayout(1, layout))
    tile = gl.arange(0, 64, layout=gl.SliceLayout(0, layout))
    active = (sample[:, None] < B) & (tile[None, :] < 56)
    if WATCHDOG_TICKS:
        fl: gl.constexpr = gl.BlockedLayout([1], [64], [8], [0])
        flat = gl.arange(0, 512, layout=fl)
        wait_acquire(
            RouterReady + flat // 64 * 56 + flat % 64,
            (flat // 64 < B) & (flat % 64 < 56),
            epoch,
            Errors,
            flat,
            WATCHDOG_TICKS,
        )
    else:
        wait_acquire(
            RouterReady + sample[:, None] * 56 + tile[None, :],
            active,
            epoch,
            Errors,
            sample[:, None] * 64 + tile[None, :],
        )
    expert = gl.arange(0, 1024, layout=gl.SliceLayout(0, layout))
    bias = gl.load(Bias + expert, expert < 896, 0).to(gl.float32)
    score = coherent_load(
        Scores + sample[:, None] * 896 + expert[None, :],
        (sample[:, None] < B) & (expert[None, :] < 896),
        0,
    )
    corrected = gl.where(expert[None, :] < 896, score + bias[None, :], float("-inf"))
    selected = ()
    total = gl.full((8,), 0.0, gl.float32, gl.SliceLayout(1, layout))
    for index in gl.static_range(16):
        if PAIR_TOPK:
            best, best_id = gl.reduce(
                (corrected, expert[None, :] + sample[:, None] * 0), 1, max_score_id
            )
        else:
            best = gl.max(corrected, 1)
            best_id = gl.min(
                gl.where(corrected == best[:, None], expert[None, :], 2147483647), 1
            )
        if PAIR_TOPK or LOAD_SELECTED_BIAS:
            best_bias = gl.load(Bias + best_id).to(gl.float32)
        else:
            best_bias = gl.sum(
                gl.where(expert[None, :] == best_id[:, None], bias[None, :], 0.0), 1
            )
        raw = best - best_bias
        selected += (raw,)
        total += raw
        gl.atomic_store(
            IDs + sample * 16 + index, best_id, sample < B, sem="relaxed", scope="gpu"
        )
        corrected = gl.where(
            expert[None, :] == best_id[:, None], float("-inf"), corrected
        )
    for index in gl.static_range(16):
        gl.atomic_store(
            Weights + sample * 16 + index,
            selected[index] / total,
            sample < B,
            sem="relaxed",
            scope="gpu",
        )
    finish_stores()
    gl.atomic_store(
        RouteReady + sample,
        epoch.to(gl.uint64) << 32 | 1,
        mask=sample < B,
        sem="relaxed",
        scope="gpu",
    )


@g.jit
def shared_situ(GU, Ready, Mid, MidReady, Errors, epoch, pid, SI: gl.constexpr):
    layout: gl.constexpr = gl.BlockedLayout([1], [64], [8], [0])
    idx = gl.arange(0, 16 if SI == 768 else 32, layout=layout)
    wait_acquire(
        Ready + pid * (SI // 64) + idx, idx < SI // 64, epoch, Errors, pid * 512 + idx
    )
    row = gl.arange(0, 1024 if SI == 768 else 2048, layout=layout)
    gate = coherent_load(GU + pid * (2 * SI) + row, row < SI, 0).to(gl.float32)
    up = coherent_load(GU + pid * (2 * SI) + SI + row, row < SI, 0).to(gl.float32)
    tg = 2.0 / (1.0 + gl.exp(-0.5 * gate)) - 1.0
    tu = 2.0 / (1.0 + gl.exp(-0.08 * up)) - 1.0
    value = 4.0 * tg * (1.0 / (1.0 + gl.exp(-gate))) * 25.0 * tu
    gl.atomic_store(
        Mid + pid * SI + row,
        value.to(gl.bfloat16),
        row < SI,
        sem="relaxed",
        scope="gpu",
    )
    finish_stores()
    gl.atomic_store(
        MidReady + pid, epoch.to(gl.uint64) << 32 | 1, sem="relaxed", scope="gpu"
    )


@g.jit
def run_front(
    Moe,
    Quant,
    XScale,
    InputReady,
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
    Errors,
    epoch,
    pid,
    B: gl.constexpr,
    WORLD: gl.constexpr,
):
    groups: gl.constexpr = (B + 3) // 4
    if pid < groups * (84 + 96 // WORLD):
        if pid < groups * 56:
            base = pid // 56 * 4
            wait_inputs(InputReady, base, epoch, Errors, pid, B)
            router_tile(Moe, RouterW, Scores, RouterReady, base, pid % 56, epoch, B)
        elif pid < groups * 84:
            task = pid - groups * 56
            base = task // 28 * 4
            wait_inputs(InputReady, base, epoch, Errors, pid, B)
            dense_tile(
                Quant,
                XScale,
                LatentW,
                LatentScale,
                Latent,
                LatentReady,
                base,
                task % 28,
                epoch,
                B,
                3584,
            )
        else:
            task = pid - groups * 84
            base = task // (96 // WORLD) * 4
            wait_inputs(InputReady, base, epoch, Errors, pid, B)
            dense_tile(
                Quant,
                XScale,
                SharedW,
                SharedScale,
                GU,
                SharedReady,
                base,
                task % (96 // WORLD),
                epoch,
                B,
                12288 // WORLD,
            )
    if pid == 0:
        select_routes(
            Scores, Bias, RouterReady, IDs, Weights, RouteReady, Errors, epoch, B
        )
    if pid < B:
        shared_situ(GU, SharedReady, Mid, MidReady, Errors, epoch, pid, 6144 // WORLD)


@g.jit
def front_only(
    Moe,
    Quant,
    XScale,
    InputReady,
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
    Epochs,
    Errors,
    B: gl.constexpr,
    WORLD: gl.constexpr = 8,
):
    pid = gl.program_id(0)
    epoch = gl.load(Epochs + pid) + 1
    run_front(
        Moe,
        Quant,
        XScale,
        InputReady,
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
        Errors,
        epoch,
        pid,
        B,
        WORLD,
    )
    gl.atomic_store(Epochs + pid, epoch, sem="relaxed", scope="gpu")


@g.jit
def projection_moe_front(
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
    B: gl.constexpr,
    RANK: gl.constexpr,
    N_BLOCKS: gl.constexpr,
    BOUNDARY: gl.constexpr = False,
    DELAY: gl.constexpr = 0,
    Epoch=None,
    DebugIPC=None,
    WORLD: gl.constexpr = 8,
):
    pid = gl.program_id(0)
    if Epoch is None:
        epoch = gl.load(Epochs + pid) + 1
        gl.barrier()
    else:
        epoch = Epoch
    projection_attnres(
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
        B,
        RANK,
        N_BLOCKS,
        BOUNDARY,
        DELAY,
        Epoch=epoch,
        DebugIPC=DebugIPC,
        WORLD=WORLD,
    )
    run_front(
        Moe,
        Quant,
        Scales,
        Ready,
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
        Errors,
        epoch,
        pid,
        B,
        WORLD,
    )
    if Epoch is None:
        gl.atomic_store(Epochs + pid, epoch, sem="relaxed", scope="gpu")
