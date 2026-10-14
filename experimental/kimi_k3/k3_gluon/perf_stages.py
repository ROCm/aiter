"""Stage-isolation kernels for performance diagnosis; serving uses moe_complete.

Expert scheduling follows FlyDSL PR1204 stages 6/7. Unchanged production math
is called here so split-versus-fused measurements isolate launch/synchronization
and compilation effects rather than introducing a new arithmetic baseline.
"""

from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from .expert_up import expert_tile
from .perf_tuned_up import expert_tile_tuned
from .moe_complete import dense_front, down_tile, wait_ready


@g.jit
def front(
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
    Epochs,
    Errors,
    B: gl.constexpr,
    WORLD: gl.constexpr,
):
    pid = gl.program_id(0)
    epoch = gl.load(Epochs + pid) + 1
    gl.barrier()
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
        None,
        500000000,
    )
    gl.store(Epochs + pid, epoch)


@g.jit
def experts(
    Latent,
    IDs,
    Weights,
    WUp,
    SUp,
    WDown,
    SDown,
    Mid,
    MidReady,
    Out,
    Epochs,
    Errors,
    B: gl.constexpr,
    WORLD: gl.constexpr,
    UP: gl.constexpr,
    DOWN: gl.constexpr,
    WATCHDOG_TICKS: gl.constexpr,
    TUNED: gl.constexpr = False,
    CHUNK: gl.constexpr = 32,
    ALIGNED: gl.constexpr = False,
    UNROLL: gl.constexpr = 1,
    LOAD_VEC: gl.constexpr = 4,
):
    pid = gl.program_id(0)
    epoch = gl.load(Epochs + pid) + (1 if UP else 0)
    gl.barrier()
    if UP:
        for task in range(pid, B * (3072 // WORLD), gl.num_programs(0)):
            sample = task // (3072 // WORLD)
            route = task // (192 // WORLD)
            tile = task % (192 // WORLD)
            expert = gl.load(IDs + route)
            if TUNED:
                expert_tile_tuned(
                    Latent,
                    WUp,
                    SUp,
                    Mid,
                    MidReady,
                    Mid,
                    expert,
                    sample,
                    route,
                    tile,
                    epoch,
                    False,
                    3072 // WORLD,
                    True,
                    CHUNK,
                    ALIGNED,
                    UNROLL,
                    LOAD_VEC,
                )
            else:
                expert_tile(
                    Latent,
                    WUp,
                    SUp,
                    Mid,
                    MidReady,
                    Mid,
                    expert,
                    sample,
                    route,
                    tile,
                    epoch,
                    False,
                    3072 // WORLD,
                    True,
                )
    if DOWN:
        fl: gl.constexpr = gl.BlockedLayout([1], [64], [8], [0])
        flags = gl.arange(0, 1024, layout=fl)
        row = gl.arange(0, 16, layout=fl)
        for task in range(pid, B * 224, gl.num_programs(0)):
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
                values = down_tile(
                    Mid, IDs, Weights, WDown, SDown, sample, tile, 3072 // WORLD, True
                )
                gl.store(Out + sample * 3584 + tile * 16 + row, values.to(gl.bfloat16))
    if UP:
        gl.store(Epochs + pid, epoch)
