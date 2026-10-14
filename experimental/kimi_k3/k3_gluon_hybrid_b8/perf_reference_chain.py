# Frozen JIT snapshot: retain original imports and tuning overrides.
# ruff: noqa: E402, F401
from .coherent_memory import coherent_load

"""Incremental fusion ablations of the validated serving kernel.

Stages: 0 output projection/TP; 1 AttnRes; 2 MoE front; 3 expert gate/up;
4 expert down/TP/norm/shared-down/latent-up/TP/residual. Identical arithmetic
and workspaces are retained across split and fused measurements.
"""
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from .moe_complete import dense_front, publish, wait_ready
from .perf_native_front import run_front as native_front, quantize_input
from .perf_reference_tail import run_tail
from .expert_up import run_experts
from .perf_reference_up import run_experts_reference
from .projection import output_projection
from .attnres import attnres_chunk
from .perf_timeline import stamp
from .perf_front32 import dense_front32

FRONT32 = gl.constexpr(True)
SKIP_UNUSED_QUANT = gl.constexpr(False)


@g.jit
def chain(
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
    START: gl.constexpr = 0,
    END: gl.constexpr = 4,
    BEGIN: gl.constexpr = True,
    TUNED: gl.constexpr = False,
    PhaseReady=None,
    PHASE_BARRIER: gl.constexpr = False,
    BUFFER: gl.constexpr = False,
    LDS: gl.constexpr = False,
    ZERO_GUARD: gl.constexpr = True,
    DOWN_CHUNK: gl.constexpr = 32,
    DOWN_UNROLL: gl.constexpr = 1,
    DOWN_LDS: gl.constexpr = False,
    WAVE_PUBLISH: gl.constexpr = False,
    SINGLE_SAMPLE: gl.constexpr = False,
    BATCH_TAIL: gl.constexpr = False,
    Timeline=None,
    DIRECT_ACC: gl.constexpr = False,
    NATIVE_DENSE: gl.constexpr = False,
    DownScale=None,
    SharedScale=None,
    TAIL_ROWS: gl.constexpr = 64,
    MFMA_LANES: gl.constexpr = False,
    NATIVE_LDS: gl.constexpr = False,
    NormStats=None,
    ExternalRouted=None,
):
    pid = gl.program_id(0)
    epoch = gl.load(Epochs + pid) + (1 if BEGIN else 0)
    gl.barrier()
    stamp(Timeline, 0, pid)
    if START <= 0 and END >= 0:
        if pid < (B + 3) // 4 * 112:
            output_projection(
                ProjectionX,
                ProjectionW,
                PeersProjection,
                LocalProjection,
                Delta,
                Epochs,
                Errors,
                B,
                RANK,
                Attention=Attention,
                Epoch=epoch,
                WORLD=WORLD,
                INPUT_DOUBLE_BUFFERED=False,
                WATCHDOG_TICKS=WATCHDOG_TICKS,
            )
    stamp(Timeline, 1, pid)
    if START <= 1 and END >= 1:
        if pid < B * 4:
            attnres_chunk(
                Attention,
                Delta,
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
                Errors,
                epoch,
                pid,
                B,
                N_BLOCKS,
                BOUNDARY,
                START == 0,
                INPUT_DOUBLE_BUFFERED=False,
                BLOCK_STRIDE_M=BLOCK_STRIDE_M,
                WATCHDOG_TICKS=WATCHDOG_TICKS,
                QUANTIZE=NATIVE_DENSE or not SKIP_UNUSED_QUANT,
            )
    stamp(Timeline, 2, pid)
    if START <= 2 and END >= 2:
        if NATIVE_DENSE:
            if START == 2 and pid < B * 4:
                quantize_input(X, Quant, Scales, InputReady, pid, epoch)
            native_front(
                X,
                Quant,
                Scales,
                InputReady,
                Router,
                Bias,
                Down,
                DownScale,
                Shared,
                SharedScale,
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
                WATCHDOG_TICKS,
                NATIVE_LDS,
            )
        elif FRONT32:
            dense_front32(
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
                InputReady if START <= 1 else None,
                WATCHDOG_TICKS,
            )
        else:
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
                InputReady if START <= 1 else None,
                WATCHDOG_TICKS,
            )
    stamp(Timeline, 3, pid)
    if START <= 3 and END >= 3 and ExternalRouted is None:
        if TUNED:
            run_experts_reference(
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
                LATENT_TILES=28 if NATIVE_DENSE else 112 if FRONT32 else 56,
                WATCHDOG_TICKS=WATCHDOG_TICKS,
                AITER_LAYOUT=AITER_LAYOUT,
                BUFFER=BUFFER,
                LDS=LDS,
                ZERO_GUARD=ZERO_GUARD,
                DIRECT_ACC=DIRECT_ACC,
                MFMA_LANES=MFMA_LANES,
            )
        else:
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
                LATENT_TILES=28 if NATIVE_DENSE else 112 if FRONT32 else 56,
                WATCHDOG_TICKS=WATCHDOG_TICKS,
                AITER_LAYOUT=AITER_LAYOUT,
            )
    stamp(Timeline, 4, pid)
    if PHASE_BARRIER and START <= 3 and (END >= 4):
        publish(PhaseReady, pid, epoch)
        flags = gl.arange(0, 256, layout=gl.BlockedLayout([1], [64], [8], [0]))
        wait_ready(PhaseReady, flags, flags < 256, epoch, Errors, pid, WATCHDOG_TICKS)
    if START <= 4 and END >= 4:
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
            BUFFER,
            DOWN_CHUNK,
            DOWN_UNROLL,
            DOWN_LDS,
            ZERO_GUARD,
            WAVE_PUBLISH,
            SINGLE_SAMPLE,
            BATCH_TAIL,
            Timeline,
            DIRECT_ACC,
            TAIL_ROWS,
            MFMA_LANES,
            NormStats,
            ExternalRouted,
        )
    gl.atomic_store(Epochs + pid, epoch, sem="relaxed", scope="gpu")
