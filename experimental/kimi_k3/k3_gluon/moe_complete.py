"""Complete K3 latent MoE: FlyDSL stages 5--8 in one resident launch.

The schedule follows FlyDSL PR1204 (21a3d1e). Serving keeps BF16 dense
weights/activations and FP32 router logits, as vLLM's A16W4 path does.
Expert weights remain MXFP4; no expert sorting or weight decompression launch
is needed. Every consumer acquires its producer's tagged publication.
"""

from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl

from .projection import project_tile
from .expert_up import run_experts
from .moe_front import select_routes, finish_stores
from .primitives import poll_local, poll_peers, pack_bf16, bf16_sum_peers
from .serving import projection_attnres_live


def pack_dense(w):
    n, k = w.shape
    assert n % 64 == 0 and k % 64 == 0
    return (
        w.reshape(n // 16, 16, k // 64, 2, 4, 8)
        .permute(0, 2, 3, 4, 1, 5)
        .contiguous()
        .view(-1)
    )


def pack_down(q):
    e, n, halfk = q.shape
    assert n == 3584 and halfk in (192, 384)
    return (
        q.reshape(e, n // 16, 16, halfk // 64, 4, 4, 4)
        .permute(0, 1, 3, 5, 2, 4, 6)
        .contiguous()
        .view(-1)
    )


@g.jit
def wait_ready(
    Ready, indices, active, epoch, Errors, pid, WATCHDOG_TICKS: gl.constexpr
):
    _, failed = poll_local(Ready + indices, epoch, active, WATCHDOG_TICKS)
    seen = gl.atomic_load(Ready + indices, mask=active, sem="acquire", scope="gpu")
    bad = active & ((failed != 0) | ((seen >> 32) != epoch.to(gl.uint64)))
    any_bad = gl.sum(bad.to(gl.int32), 0) != 0
    if any_bad:
        gl.store(Errors + pid * 512, 1)
    gl.barrier()
    return any_bad


@g.jit
def publish(Ready, index, epoch):
    finish_stores()
    gl.atomic_store(
        Ready + index, (epoch.to(gl.uint64) << 32) | 1, sem="release", scope="gpu"
    )


@g.jit
def dense_input_ready(
    InputReady, base, epoch, Errors, pid, B: gl.constexpr, WATCHDOG_TICKS: gl.constexpr
):
    if InputReady is not None:
        idx = gl.arange(0, 16, layout=gl.BlockedLayout([1], [64], [8], [0]))
        sample = base + idx // 4
        wait_ready(
            InputReady,
            sample * 4 + idx % 4,
            sample < B,
            epoch,
            Errors,
            pid,
            WATCHDOG_TICKS,
        )


@g.jit
def dense_front(
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
    # Router publishes16-row flags; latent/shared publish64-row flags.
    groups: gl.constexpr = (B + 3) // 4
    shared_tiles: gl.constexpr = 192 // WORLD
    layout: gl.constexpr = gl.BlockedLayout([1, 4], [1, 64], [8, 1], [1, 0])
    rr = gl.arange(0, 64, layout=gl.SliceLayout(0, layout))
    ss = gl.arange(0, 16, layout=gl.SliceLayout(1, layout))
    if pid < groups * (14 + 56 + shared_tiles):
        if pid < groups * 14:
            base = pid // 14 * 4
            tile = pid % 14
            dense_input_ready(InputReady, base, epoch, Errors, pid, B, WATCHDOG_TICKS)
            value = project_tile(X, Router, 0, base, tile * 64, B, 7168)
            value = gl.convert_layout(gl.permute(value, (1, 0)), layout)
            score = 1.0 / (1.0 + gl.exp(-value))
            gl.store(
                Scores + (base + ss[:, None]) * 896 + tile * 64 + rr[None, :],
                score,
                (base + ss[:, None] < B) & (ss[:, None] < 4),
            )
            finish_stores()
            fl: gl.constexpr = gl.BlockedLayout([1], [64], [8], [0])
            idx = gl.arange(0, 16, layout=fl)
            router_sample = base + idx // 4
            gl.atomic_store(
                RouterReady + router_sample * 56 + tile * 4 + idx % 4,
                (epoch.to(gl.uint64) << 32) | 1,
                mask=router_sample < B,
                sem="release",
                scope="gpu",
            )
        elif pid < groups * 70:
            task = pid - groups * 14
            base = task // 56 * 4
            tile = task % 56
            dense_input_ready(InputReady, base, epoch, Errors, pid, B, WATCHDOG_TICKS)
            value = project_tile(X, Down, 0, base, tile * 64, B, 7168)
            value = gl.convert_layout(gl.permute(value, (1, 0)), layout)
            gl.store(
                Latent + (base + ss[:, None]) * 3584 + tile * 64 + rr[None, :],
                value.to(gl.bfloat16),
                (base + ss[:, None] < B) & (ss[:, None] < 4),
            )
            # Each flag covers64 rows here; expert consumer has a matching mode.
            finish_stores()
            fl: gl.constexpr = gl.BlockedLayout([1], [64], [8], [0])
            sample = base + gl.arange(0, 4, layout=fl)
            gl.atomic_store(
                LatentReady + sample * 56 + tile,
                (epoch.to(gl.uint64) << 32) | 1,
                mask=sample < B,
                sem="release",
                scope="gpu",
            )
        else:
            task = pid - groups * 70
            base = task // shared_tiles * 4
            tile = task % shared_tiles
            dense_input_ready(InputReady, base, epoch, Errors, pid, B, WATCHDOG_TICKS)
            value = project_tile(X, Shared, 0, base, tile * 64, B, 7168)
            value = gl.convert_layout(gl.permute(value, (1, 0)), layout)
            gl.store(
                GU + (base + ss[:, None]) * (12288 // WORLD) + tile * 64 + rr[None, :],
                value.to(gl.bfloat16),
                (base + ss[:, None] < B) & (ss[:, None] < 4),
            )
            finish_stores()
            fl: gl.constexpr = gl.BlockedLayout([1], [64], [8], [0])
            sample = base + gl.arange(0, 4, layout=fl)
            gl.atomic_store(
                SharedReady + sample * shared_tiles + tile,
                (epoch.to(gl.uint64) << 32) | 1,
                mask=sample < B,
                sem="release",
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
            gate = gl.load(
                GU + pid * (12288 // WORLD) + row, row < 6144 // WORLD, 0
            ).to(gl.float32)
            up = gl.load(
                GU + pid * (12288 // WORLD) + 6144 // WORLD + row,
                row < 6144 // WORLD,
                0,
            ).to(gl.float32)
            val = 4.0 * (2.0 / (1.0 + gl.exp(-0.5 * gate)) - 1.0)
            val *= (
                (1.0 / (1.0 + gl.exp(-gate)))
                * 25.0
                * (2.0 / (1.0 + gl.exp(-0.08 * up)) - 1.0)
            )
            gl.store(
                SharedMid + pid * (6144 // WORLD) + row,
                val.to(gl.bfloat16),
                row < 6144 // WORLD,
            )
            publish(SharedMidReady, pid, epoch)


@g.jit
def down_tile(
    Mid,
    IDs,
    Weights,
    W,
    Scale,
    sample,
    tile,
    INTER: gl.constexpr,
    AITER_LAYOUT: gl.constexpr = False,
):
    # One wave per route, two rounds for top16, following FlyDSL stage7.
    mma: gl.constexpr = gl.amd.AMDMFMALayout(4, [16, 16, 32], False, [8, 1, 1])
    al: gl.constexpr = gl.BlockedLayout([1, 1, 4], [1, 16, 4], [8, 1, 1], [2, 1, 0])
    ul: gl.constexpr = gl.BlockedLayout([1, 1, 8], [1, 16, 4], [8, 1, 1], [2, 1, 0])
    bl: gl.constexpr = gl.BlockedLayout([1, 8, 1], [1, 4, 16], [8, 1, 1], [2, 1, 0])
    aw = gl.arange(0, 8, layout=gl.SliceLayout(1, gl.SliceLayout(2, al)))
    ar = gl.arange(0, 16, layout=gl.SliceLayout(0, gl.SliceLayout(2, al)))
    ak = gl.arange(0, 16, layout=gl.SliceLayout(0, gl.SliceLayout(0, al)))
    uw = gl.arange(0, 8, layout=gl.SliceLayout(1, gl.SliceLayout(2, ul)))
    ur = gl.arange(0, 16, layout=gl.SliceLayout(0, gl.SliceLayout(2, ul)))
    uk = gl.arange(0, 32, layout=gl.SliceLayout(0, gl.SliceLayout(0, ul)))
    bw = gl.arange(0, 8, layout=gl.SliceLayout(1, gl.SliceLayout(2, bl)))
    bk = gl.arange(0, 32, layout=gl.SliceLayout(0, gl.SliceLayout(2, bl)))
    bn = gl.arange(0, 16, layout=gl.SliceLayout(0, gl.SliceLayout(0, bl)))
    weight_layout: gl.constexpr = gl.SliceLayout(1, gl.SliceLayout(2, mma))
    mw = gl.arange(0, 8, layout=weight_layout)
    total = gl.full((8, 16, 16), 0.0, gl.float32, mma)
    for round in gl.static_range(2):
        ae = gl.load(IDs + sample * 16 + round * 8 + aw).to(gl.int64)
        ue = gl.load(IDs + sample * 16 + round * 8 + uw).to(gl.int64)
        route_weight = gl.load(Weights + sample * 16 + round * 8 + mw)
        acc = gl.full((8, 16, 16), 0.0, gl.float32, mma)
        for step in range(INTER // 32):
            rows = tile * 16 + ar[None, :, None]
            k = step * 32 + gl.expand_dims(gl.expand_dims(ak, 0), 0) * 2
            off = (
                (
                    ((rows // 16 * (INTER // 128) + k // 128) * 4 + (k % 32) // 8) * 16
                    + rows % 16
                )
                * 4
                + (k % 128) // 32
            ) * 4 + (k % 8) // 2
            if AITER_LAYOUT:
                off = (
                    ((rows // 16 * (INTER // 128) + k // 128) * 4 + (k % 128) // 32)
                    * 16
                    + rows % 16
                ) * 16 + (k % 32) // 2
            raw = gl.load(W + ae[:, None, None] * (3584 * INTER // 2) + off)
            scale_row = tile * 16 + ur[None, :, None]
            group = (step * 32 + gl.expand_dims(gl.expand_dims(uk, 0), 0)) // 32
            if AITER_LAYOUT:
                stride: gl.constexpr = ((INTER // 32 + 7) // 8) * 8
                soff = (
                    (scale_row // 32 * (stride // 8) + group // 8) * 4 + group % 4
                ) * 64
                soff += (
                    (scale_row % 16) * 4
                    + ((group // 4) % 2) * 2
                    + (scale_row // 16) % 2
                )
            else:
                stride: gl.constexpr = INTER // 32
                soff = scale_row * stride + group
            scales = gl.load(Scale + ue[:, None, None] * (3584 * stride) + soff)
            a = gl.amd.cdna4.scaled_upcast(raw, scales, gl.bfloat16, axis=2)
            a = gl.where(scales == 0, 0.0, a).to(gl.bfloat16)
            a = gl.convert_layout(a, gl.DotOperandLayout(0, mma, 8))
            inp = gl.load(
                Mid
                + ((sample * 16 + round * 8 + bw[:, None, None]) * INTER)
                + step * 32
                + bk[None, :, None]
                + gl.expand_dims(gl.expand_dims(bn, 0), 0) * 0
            )
            inp = gl.convert_layout(inp, gl.DotOperandLayout(1, mma, 8))
            acc += gl.amd.cdna3.mfma(a, inp, gl.full((8, 16, 16), 0.0, gl.float32, mma))
        total += acc * route_weight[:, None, None]
    plain: gl.constexpr = gl.BlockedLayout([1, 1, 1], [1, 16, 4], [8, 1, 1], [1, 2, 0])
    shared = gl.allocate_shared_memory(
        gl.float32,
        [2048],
        gl.SwizzledSharedLayout(1, 1, 1, [0]),
        gl.reshape(gl.convert_layout(total, plain), (2048,)),
    )
    fl: gl.constexpr = gl.BlockedLayout([1], [64], [8], [0])
    row = gl.arange(0, 16, layout=fl)
    value = gl.full((16,), 0.0, gl.float32, fl)
    for wave in gl.static_range(8):
        value += shared.gather(wave * 256 + row * 16, 0)
    shared._keep_alive()
    return value


@g.jit
def reduce_tile(
    values,
    Peers,
    Local,
    Out,
    sample,
    row_base,
    epoch,
    pid,
    Errors,
    B: gl.constexpr,
    N: gl.constexpr,
    TILE: gl.constexpr,
    RANK: gl.constexpr,
    WORLD: gl.constexpr,
    WATCHDOG_TICKS: gl.constexpr,
    Residual=None,
    FinalPrefix=None,
):
    # Single-stage all-to-all publication of BF16 pairs; rank-order FP32 sum.
    sl: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [0])
    shared = gl.allocate_shared_memory(gl.float32, [TILE], sl, values)
    fl: gl.constexpr = gl.BlockedLayout([1], [64], [8], [0])
    tid = gl.arange(0, 512, layout=fl)
    pair_in_tile = tid % (TILE // 2)
    low = shared.gather(pair_in_tile * 2, 0)
    high = shared.gather(pair_in_tile * 2 + 1, 0)
    pair = sample * (N // 2) + row_base // 2 + pair_in_tile
    peer_index = tid // (TILE // 2)
    peer = gl.load(Peers + peer_index, peer_index < WORLD, 0).to(
        gl.pointer_type(gl.uint64)
    )
    slot = epoch & 1
    word = pack_bf16(low, high).to(gl.uint64) | (epoch.to(gl.uint64) << 32)
    gl.atomic_store(
        peer + (slot * WORLD + RANK) * (B * N // 2) + pair,
        word,
        mask=peer_index < WORLD,
        sem="relaxed",
        scope="sys",
    )
    gl.barrier()
    active = tid < TILE // 2
    words, bad = poll_peers(
        Local + slot * WORLD * (B * N // 2),
        pair,
        epoch,
        active,
        B * N // 2,
        WORLD,
        WATCHDOG_TICKS,
    )
    low, high = bf16_sum_peers(words)
    gl.store(Out.to(gl.pointer_type(gl.uint32)) + pair, pack_bf16(low, high), active)
    if Residual is not None:
        # vLLM first materializes the BF16 MoE result, then adds the residual.
        residual = gl.load(Residual.to(gl.pointer_type(gl.uint32)) + pair, active, 0)
        rlo = (residual << 16).to(gl.float32, bitcast=True)
        rhi = (residual & 0xFFFF0000).to(gl.float32, bitcast=True)
        final = pack_bf16(
            low.to(gl.bfloat16).to(gl.float32) + rlo,
            high.to(gl.bfloat16).to(gl.float32) + rhi,
        )
        gl.store(FinalPrefix.to(gl.pointer_type(gl.uint32)) + pair, final, active)
    gl.store(Errors + pid * 512 + tid, 1, active & (bad != 0))
    shared._keep_alive()


@g.jit
def run_tail(
    Mid,
    MidReady,
    IDs,
    Weights,
    WDown,
    SDown,
    Routed,
    RoutedReady,
    NormWeight,
    Normed,
    NormReady,
    SharedMid,
    SharedMidReady,
    SharedDown,
    LatentUp,
    PeersDown,
    LocalDown,
    PeersFinal,
    LocalFinal,
    Out,
    Errors,
    epoch,
    pid,
    B: gl.constexpr,
    RANK: gl.constexpr,
    WORLD: gl.constexpr,
    WATCHDOG_TICKS: gl.constexpr,
    AITER_LAYOUT: gl.constexpr = False,
    Residual=None,
    FinalPrefix=None,
):
    fl: gl.constexpr = gl.BlockedLayout([1], [64], [8], [0])
    flags = gl.arange(0, 1024, layout=fl)
    for task in range(pid, B * 224, 256):
        sample = task // 224
        tile = task % 224
        bad = wait_ready(
            MidReady,
            sample * (3072 // WORLD) + flags,
            flags < 3072 // WORLD,
            epoch,
            Errors,
            pid,
            WATCHDOG_TICKS,
        )
        if not bad:
            down_values = down_tile(
                Mid,
                IDs,
                Weights,
                WDown,
                SDown,
                sample,
                tile,
                3072 // WORLD,
                AITER_LAYOUT,
            )
            reduce_tile(
                down_values,
                PeersDown,
                LocalDown,
                Routed,
                sample,
                tile * 16,
                epoch,
                pid,
                Errors,
                B,
                3584,
                16,
                RANK,
                WORLD,
                WATCHDOG_TICKS,
            )
            publish(RoutedReady, task, epoch)
    if pid < B:
        bad = wait_ready(
            RoutedReady,
            pid * 224 + flags,
            flags < 224,
            epoch,
            Errors,
            pid,
            WATCHDOG_TICKS,
        )
        if not bad:
            norm_row = gl.arange(0, 4096, layout=fl)
            x = gl.load(Routed + pid * 3584 + norm_row, norm_row < 3584, 0).to(
                gl.float32
            )
            inv = gl.rsqrt(gl.sum(x * x, 0) / 3584.0 + 1.0e-5)
            gain = gl.load(NormWeight + norm_row, norm_row < 3584, 0).to(gl.float32)
            gl.store(
                Normed + pid * 3584 + norm_row,
                (x * inv * gain).to(gl.bfloat16),
                norm_row < 3584,
            )
            publish(NormReady, pid, epoch)
    # The final projection uses64 rows; each request has112 tasks.
    for task in range(pid, B * 112, 256):
        sample = task // 112
        tile = task % 112
        one = gl.arange(0, 1, layout=fl)
        bad = wait_ready(
            NormReady, sample + one, one == 0, epoch, Errors, pid, WATCHDOG_TICKS
        )
        bad_shared = wait_ready(
            SharedMidReady, sample + one, one == 0, epoch, Errors, pid, WATCHDOG_TICKS
        )
        if not (bad | bad_shared):
            shared = project_tile(
                SharedMid, SharedDown, 0, sample, tile * 64, B, 6144 // WORLD
            )
            local_tile = tile - RANK * (112 // WORLD)
            if (local_tile >= 0) & (local_tile < 112 // WORLD):
                latent = project_tile(
                    Normed, LatentUp, 0, sample, local_tile * 64, B, 3584
                )
                # vLLM addmm_ rounds after adding to the BF16 shared partial.
                combined = shared.to(gl.bfloat16).to(gl.float32) + latent
            else:
                combined = shared.to(gl.bfloat16).to(gl.float32)
            plain: gl.constexpr = gl.BlockedLayout([1, 1], [16, 4], [8, 1], [0, 1])
            lds = gl.allocate_shared_memory(
                gl.float32,
                [1024],
                gl.SwizzledSharedLayout(1, 1, 1, [0]),
                gl.reshape(gl.convert_layout(combined, plain), (1024,)),
            )
            row = gl.arange(0, 64, layout=fl)
            values = lds.gather(row * 16, 0)
            reduce_tile(
                values,
                PeersFinal,
                LocalFinal,
                Out,
                sample,
                tile * 64,
                epoch,
                pid,
                Errors,
                B,
                7168,
                64,
                RANK,
                WORLD,
                WATCHDOG_TICKS,
                Residual,
                FinalPrefix,
            )
            lds._keep_alive()


@g.jit
def complete_moe(
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
    WUp,
    SUp,
    Mid,
    MidReady,
    WDown,
    SDown,
    Routed,
    RoutedReady,
    NormWeight,
    Normed,
    NormReady,
    SharedDown,
    LatentUp,
    PeersDown,
    LocalDown,
    PeersFinal,
    LocalFinal,
    Out,
    Epochs,
    Errors,
    B: gl.constexpr,
    RANK: gl.constexpr,
    WORLD: gl.constexpr = 8,
    WATCHDOG_TICKS: gl.constexpr = 0,
    AITER_LAYOUT: gl.constexpr = False,
    ProjectionX=None,
    ProjectionW=None,
    PeersProjection=None,
    LocalProjection=None,
    Delta=None,
    Attention=None,
    Prefix=None,
    Blocks=None,
    Norm=None,
    Query=None,
    OutNorm=None,
    Updated=None,
    Quant=None,
    Scales=None,
    Stats=None,
    InputReady=None,
    N_BLOCKS: gl.constexpr = 0,
    BOUNDARY: gl.constexpr = False,
    BLOCK_STRIDE_M: gl.constexpr = 0,
    Residual=None,
    FinalPrefix=None,
):
    pid = gl.program_id(0)
    epoch = gl.load(Epochs + pid) + 1
    gl.barrier()
    if ProjectionX is not None:
        projection_attnres_live(
            ProjectionX,
            ProjectionW,
            PeersProjection,
            LocalProjection,
            Delta,
            Attention,
            Prefix,
            Blocks,
            Norm,
            Query,
            OutNorm,
            Updated,
            X,
            Quant,
            Scales,
            Stats,
            InputReady,
            Epochs,
            Errors,
            B,
            RANK,
            N_BLOCKS,
            BOUNDARY,
            BLOCK_STRIDE_M,
            WORLD=WORLD,
            WATCHDOG_TICKS=WATCHDOG_TICKS,
            Epoch=epoch,
        )
    dense_front(
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
        B,
        WORLD,
        InputReady,
        WATCHDOG_TICKS,
    )
    run_experts(
        Latent,
        IDs,
        LatentReady,
        RouteReady,
        WUp,
        SUp,
        Mid,
        MidReady,
        Mid,
        Errors,
        epoch,
        pid,
        B,
        WORLD=WORLD,
        LATENT_TILES=56,
        WATCHDOG_TICKS=WATCHDOG_TICKS,
        AITER_LAYOUT=AITER_LAYOUT,
    )
    run_tail(
        Mid,
        MidReady,
        IDs,
        Weights,
        WDown,
        SDown,
        Routed,
        RoutedReady,
        NormWeight,
        Normed,
        NormReady,
        SharedMid,
        SharedMidReady,
        SharedDown,
        LatentUp,
        PeersDown,
        LocalDown,
        PeersFinal,
        LocalFinal,
        Out,
        Errors,
        epoch,
        pid,
        B,
        RANK,
        WORLD,
        WATCHDOG_TICKS,
        AITER_LAYOUT,
        Residual,
        FinalPrefix,
    )
    gl.store(Epochs + pid, epoch)
