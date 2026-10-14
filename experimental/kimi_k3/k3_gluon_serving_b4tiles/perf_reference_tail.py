# Frozen JIT snapshot: retain original imports and tuning overrides.
# ruff: noqa: E402, F401
from .coherent_memory import coherent_load

"""Isolated expert-down and peer-publication ablations for the MoE tail."""
import triton
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from .projection import project_tile
from .perf_reference_projection import project_tile_single, project_tile_rows
from .moe_complete import wait_ready, publish
from .primitives import poll_peers, poll_local, publish_local, pack_bf16, bf16_sum_peers
from .moe_front import finish_stores
from .perf_reference_down import down_tile_reference
from .perf_timeline import stamp

COLLAPSE_COLUMNS = gl.constexpr(False)


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
    WAVE_PUBLISH: gl.constexpr = False,
    NormStats=None,
):
    sl: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [0])
    shared = gl.allocate_shared_memory(gl.float32, [TILE], sl, values)
    fl: gl.constexpr = gl.BlockedLayout([1], [64], [8], [0])
    tid = gl.arange(0, 512, layout=fl)
    pair_in_tile = tid % (TILE // 2)
    low = shared.gather(pair_in_tile * 2, 0)
    high = shared.gather(pair_in_tile * 2 + 1, 0)
    pair = sample * (N // 2) + row_base // 2 + pair_in_tile
    peer_index = tid // 64 if WAVE_PUBLISH else tid // (TILE // 2)
    sending = (peer_index < WORLD) & (tid % 64 < TILE // 2 if WAVE_PUBLISH else True)
    peer = gl.load(Peers + peer_index, peer_index < WORLD, 0).to(
        gl.pointer_type(gl.uint64)
    )
    slot = epoch & 1
    word = pack_bf16(low, high).to(gl.uint64) | epoch.to(gl.uint64) << 32
    gl.atomic_store(
        peer + (slot * WORLD + RANK) * (B * N // 2) + pair,
        word,
        mask=sending,
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
    gl.atomic_store(
        Out.to(gl.pointer_type(gl.uint32)) + pair,
        pack_bf16(low, high),
        active,
        sem="relaxed",
        scope="gpu",
    )
    if Residual is not None:
        residual = coherent_load(
            Residual.to(gl.pointer_type(gl.uint32)) + pair, active, 0
        )
        rlo = (residual << 16).to(gl.float32, bitcast=True)
        rhi = (residual & 4294901760).to(gl.float32, bitcast=True)
        final = pack_bf16(
            low.to(gl.bfloat16).to(gl.float32) + rlo,
            high.to(gl.bfloat16).to(gl.float32) + rhi,
        )
        gl.atomic_store(
            FinalPrefix.to(gl.pointer_type(gl.uint32)) + pair,
            final,
            active,
            sem="relaxed",
            scope="gpu",
        )
    gl.atomic_store(
        Errors + pid * 512 + tid, 1, active & (bad != 0), sem="relaxed", scope="gpu"
    )
    if NormStats is not None:
        rounded_low = low.to(gl.bfloat16).to(gl.float32)
        rounded_high = high.to(gl.bfloat16).to(gl.float32)
        square_sum = gl.sum(
            gl.where(
                active, rounded_low * rounded_low + rounded_high * rounded_high, 0.0
            ),
            0,
        )
        finish_stores()
        publish_local(
            NormStats + sample * (N // TILE) + row_base // TILE,
            square_sum.to(gl.uint32, bitcast=True),
            epoch,
        )
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
    BUFFER: gl.constexpr = False,
    DOWN_CHUNK: gl.constexpr = 32,
    DOWN_UNROLL: gl.constexpr = 1,
    DOWN_LDS: gl.constexpr = False,
    ZERO_GUARD: gl.constexpr = True,
    WAVE_PUBLISH: gl.constexpr = False,
    SINGLE_SAMPLE: gl.constexpr = False,
    BATCH_TAIL: gl.constexpr = False,
    Timeline=None,
    DIRECT_ACC: gl.constexpr = False,
    TAIL_ROWS: gl.constexpr = 64,
    MFMA_LANES: gl.constexpr = False,
    NormStats=None,
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
            down_values = down_tile_reference(
                Mid,
                IDs,
                Weights,
                WDown,
                SDown,
                sample,
                tile,
                3072 // WORLD,
                AITER_LAYOUT,
                BUFFER,
                DOWN_CHUNK,
                DOWN_UNROLL,
                DOWN_LDS,
                ZERO_GUARD,
                DIRECT_ACC,
                MFMA_LANES,
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
                WAVE_PUBLISH=WAVE_PUBLISH,
                NormStats=NormStats,
            )
            if NormStats is None:
                publish(RoutedReady, task, epoch)
    stamp(Timeline, 5, pid)
    if NormStats is not None:
        gl.static_assert(B == 1 and TAIL_ROWS != 64 and (not SINGLE_SAMPLE))
        if pid < B:
            parts = gl.arange(0, 256, layout=fl)
            bits, failed = poll_local(
                NormStats + pid * 224 + parts, epoch, parts < 224, WATCHDOG_TICKS
            )
            gl.atomic_store(
                Errors + pid * 512 + parts,
                1,
                (parts < 224) & (failed != 0),
                sem="relaxed",
                scope="gpu",
            )
            square = gl.sum(
                gl.where(parts < 224, bits.to(gl.float32, bitcast=True), 0.0), 0
            )
            inv = gl.rsqrt(square / 3584.0 + 1e-05)
            publish_local(NormReady + pid, inv.to(gl.uint32, bitcast=True), epoch)
    elif pid < B:
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
            x = coherent_load(Routed + pid * 3584 + norm_row, norm_row < 3584, 0).to(
                gl.float32
            )
            inv = gl.rsqrt(gl.sum(x * x, 0) / 3584.0 + 1e-05)
            gain = gl.load(NormWeight + norm_row, norm_row < 3584, 0).to(gl.float32)
            gl.atomic_store(
                Normed + pid * 3584 + norm_row,
                (x * inv * gain).to(gl.bfloat16),
                norm_row < 3584,
                sem="relaxed",
                scope="gpu",
            )
            publish(NormReady, pid, epoch)
    stamp(Timeline, 6, pid)
    group_size: gl.constexpr = min(B, 4) if BATCH_TAIL else 1
    gl.static_assert(not (BATCH_TAIL and SINGLE_SAMPLE))
    for task in range(pid, triton.cdiv(B, group_size) * (7168 // TAIL_ROWS), 256):
        sample = task // (7168 // TAIL_ROWS) * group_size
        tile = task % (7168 // TAIL_ROWS)
        one = gl.arange(0, triton.next_power_of_2(group_size), layout=fl)
        live = (one < group_size) & (sample + one < B)
        bad = wait_ready(
            NormReady, sample + one, live, epoch, Errors, pid, WATCHDOG_TICKS
        )
        bad_shared = wait_ready(
            SharedMidReady, sample + one, live, epoch, Errors, pid, WATCHDOG_TICKS
        )
        if not bad | bad_shared:
            if NormStats is not None:
                inverse = (
                    gl.atomic_load(NormReady + sample, sem="relaxed", scope="gpu")
                    .to(gl.uint32)
                    .to(gl.float32, bitcast=True)
                )
            if TAIL_ROWS != 64:
                shared = project_tile(
                    SharedMid,
                    SharedDown,
                    0,
                    sample,
                    tile * TAIL_ROWS,
                    B,
                    6144 // WORLD,
                    TAIL_ROWS,
                )
            elif SINGLE_SAMPLE:
                shared = project_tile_single(
                    SharedMid, SharedDown, 0, sample, tile * 64, B, 6144 // WORLD
                )
            else:
                shared = project_tile(
                    SharedMid, SharedDown, 0, sample, tile * 64, B, 6144 // WORLD
                )
            local_tile = tile - RANK * (7168 // TAIL_ROWS // WORLD)
            if (local_tile >= 0) & (local_tile < 7168 // TAIL_ROWS // WORLD):
                if NormStats is not None:
                    latent = project_tile(
                        Routed,
                        LatentUp,
                        0,
                        sample,
                        local_tile * TAIL_ROWS,
                        B,
                        3584,
                        TAIL_ROWS,
                        NormWeight,
                        inverse,
                    )
                elif TAIL_ROWS != 64:
                    latent = project_tile(
                        Normed,
                        LatentUp,
                        0,
                        sample,
                        local_tile * TAIL_ROWS,
                        B,
                        3584,
                        TAIL_ROWS,
                    )
                elif SINGLE_SAMPLE:
                    latent = project_tile_single(
                        Normed, LatentUp, 0, sample, local_tile * 64, B, 3584
                    )
                else:
                    latent = project_tile(
                        Normed, LatentUp, 0, sample, local_tile * 64, B, 3584
                    )
                combined = shared.to(gl.bfloat16).to(gl.float32) + latent
            else:
                combined = shared.to(gl.bfloat16).to(gl.float32)
            if COLLAPSE_COLUMNS:
                gl.static_assert(B == 1 and (not SINGLE_SAMPLE))
                values = gl.max(combined, 1)
                reduce_tile(
                    values,
                    PeersFinal,
                    LocalFinal,
                    Out,
                    sample,
                    tile * TAIL_ROWS,
                    epoch,
                    pid,
                    Errors,
                    B,
                    7168,
                    TAIL_ROWS,
                    RANK,
                    WORLD,
                    WATCHDOG_TICKS,
                    Residual,
                    FinalPrefix,
                    WAVE_PUBLISH,
                )
            else:
                plain: gl.constexpr = gl.BlockedLayout([1, 1], [16, 4], [8, 1], [0, 1])
                lds = gl.allocate_shared_memory(
                    gl.float32,
                    [TAIL_ROWS * 16],
                    gl.SwizzledSharedLayout(1, 1, 1, [0]),
                    gl.reshape(gl.convert_layout(combined, plain), (TAIL_ROWS * 16,)),
                )
                row = gl.arange(0, TAIL_ROWS, layout=fl)
                for column in gl.static_range(group_size):
                    if sample + column < B:
                        values = lds.gather(row * 16 + column, 0)
                        reduce_tile(
                            values,
                            PeersFinal,
                            LocalFinal,
                            Out,
                            sample + column,
                            tile * TAIL_ROWS,
                            epoch,
                            pid,
                            Errors,
                            B,
                            7168,
                            TAIL_ROWS,
                            RANK,
                            WORLD,
                            WATCHDOG_TICKS,
                            Residual,
                            FinalPrefix,
                            WAVE_PUBLISH,
                        )
                lds._keep_alive()
    stamp(Timeline, 7, pid)
