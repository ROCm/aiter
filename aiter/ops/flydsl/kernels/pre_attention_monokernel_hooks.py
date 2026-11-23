# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Trace-time (const_expr) and runtime branches nest here and do not merge.
# ruff: noqa: SIM102

"""Hooks of the pre-attention monokernel (``gemm_hooks`` contract) and its memory plan.

The fused Q/K/V roles GEMM runs these hooks:
  - after each role's tiles: the role's RMS factors (Q/K, norm across heads) and, for
    per-tensor scales, its amax per destination; the coordinator (last workgroup)
    exchanges amax with every peer;
  - after each workgroup's first block of the next role, when every workgroup has a tile
    of every role: the previous role's quantize into the cached staging and its puts;
  - after every V tile: V's amax per destination (per-tensor V);
  - after V's tiles: V's quantize and puts, the drain, and the arrival flags.
Each role's receive buffers are the A2A op's packed buffers for the profile, and each codec
(``pre_attention_monokernel_quant``) reproduces its transport bytes. Per-tensor and
per-channel scales come from cross-rank exchanges; MX scales are local to each 32-value
block.

Control words that other workgroups or peers read live in the uncached CCO window; bulk
data stays in cached memory, with one L2 writeback per XCD before the copy engines read it.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
import mori.cco.device.flydsl as cco
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr import math as fmath

from .communication_ops_utils import spin_until_ge_i32_agent, spin_until_ge_i64
from .pre_attention_monokernel_quant import THREADS, make_row_quantize
from .tensor_shim import buf_copy_load, buf_copy_store, ptr_buf_tensor

# Window control words, bytes from the control region, for up to 8 ranks: amax
# [3][src 8][dst 8] FP32, amax generation [3][src 8], arrival generation [src 8], four
# counters per role, scale [3][dst 8] FP32, the coordinator's per-wave amax [4][8], V's
# amax per destination from its tiles [8], per-channel exchange generation [src 8] and
# counter, then per-workgroup amax [3][workgroups][8] from PEAKS and every source's
# per-channel V maxima [src][world * channels] FP32 from Plan.pc_val.
AMAX_VAL, AMAX_GEN, RECV_GEN, COUNTERS, SCALE, WAVE_AMAX, V_AMAX = (
    0,
    1024,
    1280,
    1536,
    1600,
    1792,
    2048,
)
PC_GEN, PC_COUNT, PEAKS = 2112, 2176, 4096
MAX_WORLD = 8
# Counters per role: workgroups' amax in, workgroups' rows quantized, XCDs written back,
# scales ready.
AMAXED, DONE, FLUSHED, READY = 0, 4, 8, 12
# aux (ordinary memory): the GEMM's tile counters per role at 0, RMS factors [2][M] FP32
# from FACTORS, a scratch word, then per-channel V maxima per 128 rows
# [M / 128][heads * 128] and the per-channel V scales [heads * 128] (Plan.vpc,
# Plan.scale_pc).
FACTORS = 64
HEAD_DIM = 128


def _align(value, to=256):
    return (value + to - 1) // to * to


class Plan:
    """Byte offsets of the window (receive buffers, control words), the staging and aux.

    The receive buffers are double-buffered by call parity: a peer's puts of call n + 1
    land while this rank's MHA may still read call n's. A peer cannot reach call n + 2
    before every rank has passed call n + 1's arrival handshake, after its MHA of call n.
    """

    def __init__(self, layouts, world, rows, heads, workgroups):
        if world > MAX_WORLD:
            raise ValueError(
                f"the monokernel supports up to {MAX_WORLD} ranks, got {world}"
            )
        self.recv_payload, self.recv_scale, offset = [], [], 0
        for layout in layouts:
            self.recv_payload.append(offset)
            offset = _align(offset + layout.payload_bytes)
            self.recv_scale.append(offset)
            offset = _align(offset + layout.scale_bytes)
        self.parity_bytes = (
            offset  # receive buffers of one parity; parity 1 follows parity 0
        )
        self.ctl = 2 * offset
        self.channels = heads * HEAD_DIM  # all V channels, every destination's
        self.pc_val = max(32768, _align(PEAKS + 3 * workgroups * MAX_WORLD * 4))
        self.ctl_bytes = _align(self.pc_val + world * self.channels * 4)
        self.window_bytes = self.ctl + self.ctl_bytes
        self.stage, offset = [], 0
        for layout in layouts:
            self.stage.append(offset)
            offset += world * layout.stage_bytes
        self.scratch = offset
        self.staging_bytes = offset + 64
        self.half_blocks = (rows + 127) // 128
        # Per-channel V: the GEMM epilogue's maxima per 128 rows, then the scales of every
        # destination's channels, both in cached memory.
        self.vpc = _align(FACTORS + (2 * rows + 1) * 4)
        self.scale_pc = _align(self.vpc + self.half_blocks * self.channels * 4)
        self.aux_bytes = self.scale_pc + self.channels * 4


def make_hooks(
    rank,
    world,
    rows,
    role_cols,
    layouts,
    plan,
    tiles_per_role,
    workgroups,
    xcds,
    across_heads=True,
    head_dim=128,
    eps=1e-6,
):
    """The role, first-block and tile hooks for ``layouts`` (``role_layouts``).

    ``tiles_per_role`` is the host GEMM's tile count per role, over ``workgroups``
    persistent workgroups on ``xcds`` XCDs. ``across_heads`` computes Q's and K's RMS
    factors (norm across heads) and folds them into the quantization. Returns the three
    hooks; the first-block hook is None when quantize and puts run in the role hook.
    """
    heads_role = role_cols // head_dim
    heads_all = 3 * heads_role
    heads_local = heads_role // world
    channels = heads_local * head_dim  # one destination's V channels
    ctl = plan.ctl
    pc_val = plan.pc_val
    coord = workgroups - 1
    factor_scratch = FACTORS // 4 + 2 * rows
    if heads_role % 4:
        raise ValueError(f"heads per role must be a multiple of 4, got {heads_role}")
    if workgroups % xcds or workgroups < xcds:
        raise ValueError(f"{workgroups} workgroups do not cover {xcds} XCDs evenly")
    # Lane l of a workgroup takes rows bid + (l + 64 * chunk) * workgroups.
    row_chunks = ((rows + workgroups - 1) // workgroups + 63) // 64
    # The deferred quantize runs in the first-block hook of the next role, which needs a
    # tile of every role in every workgroup.
    defer = tiles_per_role >= workgroups
    kinds = [layout.scale for layout in layouts]
    v_tile_amax = kinds[2] == "tensor"
    quantizers = [
        make_row_quantize(
            layout, rank, world, rows, heads_local, workgroups, across_heads
        )
        for layout in layouts
    ]
    pc_channels = world * channels  # every destination's V channels
    per_wg = (pc_channels + workgroups - 1) // workgroups
    peak_loads = (workgroups + THREADS - 1) // THREADS
    if pc_channels % THREADS:
        raise ValueError(f"V channels {pc_channels} must be a multiple of {THREADS}")

    def gptr(address, dtype):
        return fx.inttoptr(
            fx.PointerType.get(dtype.ir_type, fx.AddressSpace.Global, 8),
            fx.Int64(address),
        )

    def load_f32(address):
        return fx.Float32(fx.generic_load(gptr(address, fx.Float32)))

    def release_store(address, value, dtype):
        fx.generic_store(
            gptr(address, dtype),
            value,
            memory_order=fx.AtomicOrdering.Release,
            syncscope=fx.rocdl.SyncScope.OneAs,
        )

    def count(address):
        fx.atomic_add(
            gptr(address, fx.Int32), fx.Int32(1), syncscope=fx.rocdl.SyncScope.Agent
        )

    def wave_max(value):
        for shift in (32, 16, 8, 4, 2, 1):
            value = value.maximumf(value.shuffle_xor(shift, 64))
        return value

    def workgroup_peak(own, role, tid, d):
        """Maximum over every workgroup's amax for destination d; thread t reads
        workgroups t, t + THREADS, ..."""
        if workgroups == THREADS:
            return wave_max(
                load_f32(
                    own
                    + (ctl + PEAKS)
                    + ((role * workgroups + fx.Int64(tid)) * 8 + d) * 4
                )
            )
        value = fx.Float32(0.0)
        for j in range(peak_loads):
            wg = tid + j * THREADS
            live = wg < workgroups
            wg = live.select(wg, fx.Int32(0))
            peak = load_f32(
                own + (ctl + PEAKS) + ((role * workgroups + fx.Int64(wg)) * 8 + d) * 4
            )
            value = value.maximumf(live.select(peak, fx.Float32(0.0)))
        return wave_max(value)

    def parity(generation):
        """Byte offset of this call's receive buffers in every rank's window."""
        return (fx.Int64(generation) & 1) * plan.parity_bytes

    def unpack(role, args):
        c, partials, staging, dc, w, generation = args
        window = cco.Window(w)
        own = window.lsa_ptr(rank, 0)
        return (
            c,
            partials,
            staging,
            dc,
            w,
            generation,
            window,
            own,
            fx.Int32(fx.thread_idx.x),
            fx.Int32(fx.block_idx.x),
            fx.Int32(generation),
            own + (ctl + COUNTERS + role * 16),
        )

    @flyc.jit
    def amax_step(role, aux, args):
        """1. This rank's tiles of the role; then wave 0 of each workgroup takes its rows'
        factors (Q/K) and, for per-tensor scales, their amax per destination from the
        epilogue's tables."""
        (
            _c,
            partials,
            _staging,
            _dc,
            _w,
            _generation,
            _window,
            own,
            tid,
            bid,
            gen,
            counters,
        ) = unpack(role, args)
        if tid == 0:
            spin_until_ge_i32_agent(aux + role * 4, gen * tiles_per_role)
        fx.barrier()
        fx.memory_fence(
            ordering=fx.AtomicOrdering.Acquire, syncscope=fx.rocdl.SyncScope.Agent
        )
        tables = ptr_buf_tensor(partials, fx.Float32, unit_elems=4, unit_stride=1)
        lane_peaks = [None] * world
        for chunk in range_constexpr(row_chunks):
            if const_expr(chunk == 0):
                row = bid + (tid % 64) * workgroups
            else:
                row = bid + (tid % 64 + 64 * chunk) * workgroups
            valid = row < rows
            row = valid.select(row, fx.Int32(0))
            factor = fx.Float32(1.0)
            if const_expr(role < 2 and across_heads):
                sq = fx.Float32(0.0)
                for q in range_constexpr(heads_role // 4):
                    v = fx.Vector(
                        buf_copy_load(
                            tables,
                            row * heads_all + role * heads_role + 4 * q,
                            fx.Float32,
                            4,
                        )
                    )
                    for k in range_constexpr(4):
                        sq = sq + v[k]
                factor = fmath.rsqrt(sq * (1.0 / role_cols) + eps)
                index = valid.select(
                    fx.Int32(FACTORS // 4 + role * rows) + row, fx.Int32(factor_scratch)
                )
                buf_copy_store(
                    ptr_buf_tensor(aux, fx.Float32), index, factor, fx.Float32
                )
            if const_expr(kinds[role] == "tensor"):
                maxima = []
                for q in range_constexpr(heads_role // 4):
                    v = fx.Vector(
                        buf_copy_load(
                            tables,
                            (rows + row) * heads_all + role * heads_role + 4 * q,
                            fx.Float32,
                            4,
                        )
                    )
                    maxima += [v[k] for k in range(4)]
                for d in range_constexpr(world):
                    peak = maxima[d * heads_local]
                    for h in range_constexpr(1, heads_local):
                        peak = peak.maximumf(maxima[d * heads_local + h])
                    peak = valid.select(peak * factor, fx.Float32(0.0))
                    if const_expr(chunk == 0):
                        lane_peaks[d] = peak
                    else:
                        lane_peaks[d] = lane_peaks[d].maximumf(peak)
        if const_expr(kinds[role] == "tensor"):
            peaks = [wave_max(lane_peaks[d]) for d in range(world)]
            if tid == 0:
                for d in range_constexpr(world):
                    fx.generic_store(
                        gptr(
                            own
                            + (ctl + PEAKS)
                            + ((role * workgroups + fx.Int64(bid)) * 8 + d) * 4,
                            fx.Float32,
                        ),
                        peaks[d],
                    )
        fx.rocdl.s_waitcnt(vmcnt=0)
        fx.barrier()
        if const_expr(kinds[role] == "tensor"):
            if tid == 0:
                count(counters + AMAXED)

    @flyc.jit
    def exchange_step(role, aux, args):
        """2. The coordinator: this rank's per-tensor amax per destination, exchanged with
        every peer; the scales, and this rank's own into its receive scale buffer."""
        (
            _c,
            _partials,
            _staging,
            _dc,
            _w,
            generation,
            window,
            own,
            tid,
            bid,
            gen,
            counters,
        ) = unpack(role, args)
        tile_amax = v_tile_amax and role == 2
        if bid == coord:
            if const_expr(tile_amax):
                # The tiles' atomics land before each tile is counted.
                if tid == 0:
                    spin_until_ge_i32_agent(aux + role * 4, gen * tiles_per_role)
                fx.barrier()
                fx.memory_fence(
                    ordering=fx.AtomicOrdering.Acquire,
                    syncscope=fx.rocdl.SyncScope.Agent,
                )
            else:
                if tid == 0:
                    spin_until_ge_i32_agent(counters + AMAXED, gen * workgroups)
                fx.barrier()
                fx.memory_fence(
                    ordering=fx.AtomicOrdering.Acquire,
                    syncscope=fx.rocdl.SyncScope.Agent,
                )
                # A runtime `if` may not call methods of a list, hence the comprehension.
                mine = [workgroup_peak(own, role, tid, d) for d in range(world)]
                if tid % 64 == 0:
                    for d in range_constexpr(world):
                        fx.generic_store(
                            gptr(
                                own
                                + (ctl + WAVE_AMAX)
                                + (fx.Int64(tid // 64) * 8 + d) * 4,
                                fx.Float32,
                            ),
                            mine[d],
                        )
                fx.rocdl.s_waitcnt(vmcnt=0)
                fx.barrier()
            if tid < world:
                peer = tid
                for d in range_constexpr(world):
                    if const_expr(tile_amax):
                        local = load_f32(own + (ctl + V_AMAX) + d * 4)
                    else:
                        local = load_f32(own + (ctl + WAVE_AMAX) + d * 4)
                        for wave in range_constexpr(1, THREADS // 64):
                            local = local.maximumf(
                                load_f32(own + (ctl + WAVE_AMAX) + (wave * 8 + d) * 4)
                            )
                    fx.generic_store(
                        gptr(
                            window.lsa_ptr(
                                peer, ctl + AMAX_VAL + ((role * 8 + rank) * 8 + d) * 4
                            ),
                            fx.Float32,
                        ),
                        local,
                    )
                release_store(
                    window.lsa_ptr(peer, ctl + AMAX_GEN + (role * 8 + rank) * 8),
                    generation,
                    fx.Int64,
                )
                spin_until_ge_i64(
                    own + ctl + AMAX_GEN + (role * 8 + fx.Int64(peer)) * 8, generation
                )
            fx.barrier()
            if const_expr(tile_amax):
                if tid < world:
                    # Every lane has read it: clear V's amax for the next call's tiles.
                    fx.generic_store(
                        gptr(own + (ctl + V_AMAX) + fx.Int64(tid) * 4, fx.Float32),
                        fx.Float32(0.0),
                    )
            if tid < world:
                dst = tid
                fx.memory_fence(
                    ordering=fx.AtomicOrdering.Acquire,
                    syncscope=fx.rocdl.SyncScope.OneAs,
                )
                maximum = fx.Float32(0.0)
                for src in range_constexpr(world):
                    maximum = maximum.maximumf(
                        load_f32(
                            own
                            + ctl
                            + AMAX_VAL
                            + ((role * 8 + src) * 8 + fx.Int64(dst)) * 4
                        )
                    )
                # As the op's per-tensor push: E4M3 Q/K amax * (1 / 448), E4M3 V amax / 448,
                # INT8 amax / 127, and 1 for an all-zero tensor.
                if const_expr(layouts[role].codec == "int8"):
                    scale = maximum / fx.Float32(127.0)
                elif const_expr(role < 2):
                    scale = maximum * fx.Float32(1.0 / 448.0)
                else:
                    scale = maximum / fx.Float32(448.0)
                scale = (scale > 0.0).select(scale, fx.Float32(1.0))
                fx.generic_store(
                    gptr(
                        own + ctl + SCALE + (role * 8 + fx.Int64(dst)) * 4, fx.Float32
                    ),
                    scale,
                )
                if dst == rank:
                    fx.generic_store(
                        gptr(
                            own + parity(generation) + plan.recv_scale[role], fx.Float32
                        ),
                        scale,
                    )
            fx.rocdl.s_waitcnt(vmcnt=0)
            fx.barrier()
            if tid == 0:
                fx.generic_store(gptr(counters + READY, fx.Int32), gen)

    @flyc.jit
    def channel_step(role, aux, args):
        """1-2 for per-channel V: every workgroup reduces its channels' maxima over the GEMM
        epilogue's 128-row blocks and gives them to every peer; the coordinator meets every
        peer, then publishes each destination's per-channel scales and this rank's
        descales."""
        (
            _c,
            _partials,
            _staging,
            _dc,
            _w,
            generation,
            window,
            own,
            tid,
            bid,
            gen,
            counters,
        ) = unpack(role, args)
        if tid == 0:
            spin_until_ge_i32_agent(aux + role * 4, gen * tiles_per_role)
        fx.barrier()
        fx.memory_fence(
            ordering=fx.AtomicOrdering.Acquire, syncscope=fx.rocdl.SyncScope.Agent
        )
        ch = bid * per_wg + tid
        owned = tid < per_wg
        if const_expr(pc_channels % workgroups):
            owned = owned & (ch < pc_channels)
        if owned:
            table = ptr_buf_tensor(aux + plan.vpc, fx.Float32)
            maximum = fx.Float32(0.0)
            for half in range_constexpr(plan.half_blocks):
                maximum = maximum.maximumf(
                    fx.Float32(
                        buf_copy_load(table, half * pc_channels + ch, fx.Float32)
                    )
                )
            for peer in range_constexpr(world):
                fx.generic_store(
                    gptr(
                        window.lsa_ptr(peer, ctl + pc_val)
                        + (fx.Int64(rank) * pc_channels + fx.Int64(ch)) * 4,
                        fx.Float32,
                    ),
                    maximum,
                )
        # Remote stores to the uncached window are acknowledged once they landed.
        fx.rocdl.s_waitcnt(vmcnt=0)
        fx.barrier()
        if tid == 0:
            count(own + ctl + PC_COUNT)
        if bid == coord:
            if tid == 0:
                spin_until_ge_i32_agent(own + ctl + PC_COUNT, gen * workgroups)
            fx.barrier()
            if tid < world:
                peer = tid
                fx.memory_fence(
                    ordering=fx.AtomicOrdering.Release,
                    syncscope=fx.rocdl.SyncScope.OneAs,
                )
                release_store(
                    window.lsa_ptr(peer, ctl + PC_GEN + rank * 8), generation, fx.Int64
                )
                spin_until_ge_i64(own + ctl + PC_GEN + fx.Int64(peer) * 8, generation)
            fx.barrier()
            fx.memory_fence(
                ordering=fx.AtomicOrdering.Acquire, syncscope=fx.rocdl.SyncScope.OneAs
            )
            for k in range_constexpr(pc_channels // THREADS):
                ch = tid + k * THREADS
                maximum = fx.Float32(0.0)
                for src in range_constexpr(world):
                    maximum = maximum.maximumf(
                        load_f32(
                            own
                            + (ctl + pc_val)
                            + (fx.Int64(src) * pc_channels + fx.Int64(ch)) * 4
                        )
                    )
                # As the op's per-channel push: zero channels clamp to 1e-12, then * (1 / 448).
                scale = maximum.maximumf(fx.Float32(1.0e-12)) * fx.Float32(1.0 / 448.0)
                fx.generic_store(
                    gptr(aux + plan.scale_pc + fx.Int64(ch) * 4, fx.Float32), scale
                )
                if ch // channels == rank:
                    fx.generic_store(
                        gptr(
                            own
                            + parity(generation)
                            + plan.recv_scale[role]
                            + fx.Int64(ch - rank * channels) * 4,
                            fx.Float32,
                        ),
                        scale,
                    )
            fx.rocdl.s_waitcnt(vmcnt=0)
            fx.barrier()
            if tid == 0:
                # Every quantizer reads the scales through its own XCD's L2: write this one back.
                fx.memory_fence(
                    ordering=fx.AtomicOrdering.Release,
                    syncscope=fx.rocdl.SyncScope.Agent,
                )
                fx.generic_store(gptr(counters + READY, fx.Int32), gen)

    @flyc.jit
    def quantize_step(role, aux, args):
        """3. Every workgroup quantizes its rows of the role into the staging buffer; one
        per XCD then writes its L2 back."""
        (
            c,
            _partials,
            staging,
            _dc,
            _w,
            _generation,
            _window,
            own,
            tid,
            bid,
            gen,
            counters,
        ) = unpack(role, args)
        kind = kinds[role]
        if const_expr(kind != "block"):
            if tid == 0:
                spin_until_ge_i32_agent(counters + READY, gen)
            fx.barrier()
            fx.memory_fence(
                ordering=fx.AtomicOrdering.Acquire, syncscope=fx.rocdl.SyncScope.Agent
            )
        scales = 0
        if const_expr(kind == "tensor"):
            scales = [
                load_f32(own + ctl + SCALE + (role * 8 + d) * 4) for d in range(world)
            ]
        if const_expr(kind == "channel"):
            scales = aux + plan.scale_pc
        quantizers[role](
            c + role * rows * role_cols * 2,
            aux + FACTORS + role * rows * 4,
            staging + plan.stage[role],
            plan.scratch - plan.stage[role],
            scales,
        )
        fx.rocdl.s_waitcnt(vmcnt=0)
        fx.barrier()
        if tid == 0:
            count(counters + DONE)
        # Workgroups go to the XCDs round robin, so the last xcds cover every L2.
        if bid >= workgroups - xcds:
            if tid == 0:
                spin_until_ge_i32_agent(counters + DONE, gen * workgroups)
                # Every row is in some L2: write this XCD's back for the copy engines.
                fx.memory_fence(
                    ordering=fx.AtomicOrdering.Release,
                    syncscope=fx.rocdl.SyncScope.OneAs,
                )
                count(counters + FLUSHED)

    @flyc.jit
    def put_step(role, aux, args):
        """4. The coordinator posts the role's puts, one per range and destination; after V
        it drains them and meets every peer."""
        (
            _c,
            _partials,
            staging,
            dc,
            w,
            generation,
            window,
            own,
            tid,
            bid,
            gen,
            counters,
        ) = unpack(role, args)
        layout = layouts[role]
        sdma = cco.DevComm(dc).sdma()
        if bid == coord:
            if tid == 0:
                spin_until_ge_i32_agent(counters + FLUSHED, gen * xcds)
            fx.barrier()
            if tid < world:
                peer = tid
                for part in layout.ranges:
                    region = (
                        plan.recv_payload[role]
                        if part.region == "payload"
                        else plan.recv_scale[role]
                    )
                    # The put's source offset is relative to the window; size_t arithmetic wraps.
                    sdma.put(
                        peer,
                        w,
                        parity(generation) + (region + part.recv),
                        w,
                        staging
                        + (plan.stage[role] + part.stage)
                        + fx.Int64(peer) * layout.stage_bytes
                        - own,
                        part.length,
                        0,
                    )
            if const_expr(role == 2):
                if tid < world:
                    peer = tid
                    sdma.quiet_queue(peer, 0)
                    fx.memory_fence(
                        ordering=fx.AtomicOrdering.Release,
                        syncscope=fx.rocdl.SyncScope.OneAs,
                    )
                    release_store(
                        window.lsa_ptr(peer, ctl + RECV_GEN + rank * 8),
                        generation,
                        fx.Int64,
                    )
                    spin_until_ge_i64(
                        own + ctl + RECV_GEN + fx.Int64(peer) * 8, generation
                    )
                fx.barrier()
                fx.memory_fence(
                    ordering=fx.AtomicOrdering.Acquire,
                    syncscope=fx.rocdl.SyncScope.OneAs,
                )

    @flyc.jit
    def role_hook(role, aux, args, c_m):
        if const_expr(role == 2 and kinds[2] == "channel"):
            channel_step(role, aux, args)
        else:
            if const_expr(not (v_tile_amax and role == 2)):
                amax_step(role, aux, args)
            if const_expr(kinds[role] == "tensor"):
                exchange_step(role, aux, args)
        if const_expr(not defer or role == 2):
            quantize_step(role, aux, args)
            put_step(role, aux, args)

    @flyc.jit
    def first_block_hook(role, aux, args, c_m):
        # The previous role's quantize and puts: by now its scales are out, so no
        # workgroup waits for the exchange.
        quantize_step(role - 1, aux, args)
        put_step(role - 1, aux, args)

    @flyc.jit
    def tile_hook(role, aux, args, c_m, row_base, col_base):
        # Per-tensor V's amax per destination, from the tile this wave just stored: 128
        # rows of one head. (Per-channel V takes its maxima from the GEMM epilogue,
        # col_amax_offset.)
        if const_expr(role == 2):
            (
                _c,
                partials,
                _staging,
                _dc,
                _w,
                _gen64,
                _window,
                own,
                tid,
                _bid,
                _gen,
                _counters,
            ) = unpack(role, args)
            lane = tid % 64
            head = fx.Int32(col_base) // head_dim
            tables = ptr_buf_tensor(partials, fx.Float32)
            peak = fx.Float32(0.0)
            for r in range_constexpr(128 // 64):
                row = fx.Int32(row_base) + lane + r * 64
                valid = row < rows
                row = valid.select(row, fx.Int32(0))
                peak = peak.maximumf(
                    valid.select(
                        fx.Float32(
                            buf_copy_load(
                                tables, (rows + row) * heads_all + head, fx.Float32
                            )
                        ),
                        fx.Float32(0.0),
                    )
                )
            peak = wave_max(peak)
            if lane == 0:
                dest = (head - 2 * heads_role) // heads_local
                fx.atomic_fmax(
                    gptr(own + (ctl + V_AMAX) + fx.Int64(dest) * 4, fx.Float32),
                    peak,
                    is_positive=True,
                    syncscope=fx.rocdl.SyncScope.Agent,
                )
            # Thread 0 counts the tile right after this hook: every wave's atomics land first.
            fx.rocdl.s_waitcnt(vmcnt=0)
            fx.barrier()

    return (
        role_hook,
        first_block_hook if defer else None,
        tile_hook if v_tile_amax else None,
    )
