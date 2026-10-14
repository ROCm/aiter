# Frozen JIT snapshot: retain original imports and tuning overrides.
# ruff: noqa: E402
from .coherent_memory import coherent_load

"""Kimi post-attention AttnRes and MXFP8, FlyDSL KDA stage 4.

Four CTAs/request publish tagged FP32 statistics, then consume each other's
statistics. EPS=1e-5 and the E8M0 scale rounding follow FlyDSL 21a3d1e.
N_BLOCKS counts completed blocks on entry, including an already-appended
pre-attention block at a boundary. BOUNDARY selects updated=attention rather
than updated=BF16(prefix+attention). This stage never appends blocks itself.
"""
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from .primitives import pack_bf16, poll_local, publish_local
from .projection import output_projection


@g.jit
def unpack_pair(word):
    word = word.to(gl.uint32)
    return (
        (word << 16).to(gl.float32, bitcast=True),
        (word & 4294901760).to(gl.float32, bitcast=True),
    )


@g.jit
def add_statistics(square_a, weighted_a, square_b, weighted_b):
    return (square_a + square_b, weighted_a + weighted_b)


@g.jit
def stat_total(
    Stats, sample, index, epoch, Errors, pid, WATCHDOG_TICKS: gl.constexpr = 0
):
    layout: gl.constexpr = gl.BlockedLayout([1], [64], [8], [0])
    chunks = gl.arange(0, 4, layout=layout)
    bits, failed = poll_local(
        Stats + (sample * 4 + chunks) * 19 + index,
        epoch,
        gl.full((4,), True, gl.int1, layout),
        WATCHDOG_TICKS,
    )
    gl.atomic_store(
        Errors + pid * 512 + chunks, 1, failed != 0, sem="relaxed", scope="gpu"
    )
    return gl.sum(bits.to(gl.float32, bitcast=True), 0)


@g.jit
def attnres_chunk(
    Attention,
    Delta,
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
    Errors,
    epoch,
    pid,
    B: gl.constexpr,
    N_BLOCKS: gl.constexpr,
    BOUNDARY: gl.constexpr,
    TAGGED: gl.constexpr,
    INPUT_DOUBLE_BUFFERED: gl.constexpr = True,
    BLOCK_STRIDE_M: gl.constexpr = 0,
    WATCHDOG_TICKS: gl.constexpr = 0,
    QUANTIZE: gl.constexpr = True,
):
    gl.static_assert(B >= 1 and B <= 8 and (N_BLOCKS >= 0) and (N_BLOCKS <= 8))
    sample = pid // 4
    chunk = pid % 4
    slot = epoch & 1 if INPUT_DOUBLE_BUFFERED else 0
    block_stride: gl.constexpr = BLOCK_STRIDE_M if BLOCK_STRIDE_M else N_BLOCKS * 7168
    layout: gl.constexpr = gl.BlockedLayout([1], [64], [8], [0])
    p = gl.arange(0, 1024, layout=layout)
    valid = p < 896
    pair_in_row = chunk * 896 + p
    pair = sample * 3584 + pair_in_row
    if TAGGED:
        word, failed = poll_local(Attention + pair, epoch, valid, WATCHDOG_TICKS)
        gl.atomic_store(
            Errors + pid * 512 + p % 512,
            1,
            valid & (failed != 0),
            sem="relaxed",
            scope="gpu",
        )
    else:
        word = coherent_load(Delta.to(gl.pointer_type(gl.uint32)) + pair, valid, 0)
    lo, hi = unpack_pair(word)
    if not BOUNDARY:
        prefix_word = coherent_load(
            Prefix.to(gl.pointer_type(gl.uint32)) + slot * B * 3584 + pair, valid, 0
        )
        pl, ph = unpack_pair(prefix_word)
        lo = (pl + lo).to(gl.bfloat16).to(gl.float32)
        hi = (ph + hi).to(gl.bfloat16).to(gl.float32)
    gl.atomic_store(
        Updated.to(gl.pointer_type(gl.uint32)) + pair,
        pack_bf16(lo, hi),
        valid,
        sem="relaxed",
        scope="gpu",
    )
    nw = gl.load(Norm.to(gl.pointer_type(gl.uint32)) + pair_in_row, valid, 0)
    qw = gl.load(Query.to(gl.pointer_type(gl.uint32)) + pair_in_row, valid, 0)
    nl, nh = unpack_pair(nw)
    ql, qh = unpack_pair(qw)
    for source in gl.static_range(N_BLOCKS + 1):
        if source < N_BLOCKS:
            address = (
                (slot * B + sample) * (block_stride // 2) + source * 3584 + pair_in_row
            )
            bw = coherent_load(
                Blocks.to(gl.pointer_type(gl.uint32)) + address, valid, 0
            )
            vl, vh = unpack_pair(bw)
        else:
            vl, vh = (lo, hi)
        square, weighted = gl.reduce(
            (
                gl.where(valid, vl * vl + vh * vh, 0.0),
                gl.where(valid, vl * nl * ql + vh * nh * qh, 0.0),
            ),
            0,
            add_statistics,
        )
        publish_local(
            Stats + pid * 19 + source * 2, square.to(gl.uint32, bitcast=True), epoch
        )
        publish_local(
            Stats + pid * 19 + source * 2 + 1,
            weighted.to(gl.uint32, bitcast=True),
            epoch,
        )
    logits = ()
    maximum = float("-inf")
    for source in gl.static_range(N_BLOCKS + 1):
        sq = stat_total(Stats, sample, source * 2, epoch, Errors, pid, WATCHDOG_TICKS)
        weighted = stat_total(
            Stats, sample, source * 2 + 1, epoch, Errors, pid, WATCHDOG_TICKS
        )
        logit = weighted * gl.rsqrt(sq * (1.0 / 7168) + 1e-05)
        logits += (logit,)
        maximum = gl.maximum(maximum, logit)
    weights = ()
    denominator = 0.0
    for source in gl.static_range(N_BLOCKS + 1):
        prob = gl.exp(logits[source] - maximum)
        weights += (prob,)
        denominator += prob
    mixed_lo = gl.full((1024,), 0.0, gl.float32, layout)
    mixed_hi = gl.full((1024,), 0.0, gl.float32, layout)
    for source in gl.static_range(N_BLOCKS + 1):
        if source < N_BLOCKS:
            address = (
                (slot * B + sample) * (block_stride // 2) + source * 3584 + pair_in_row
            )
            bw = coherent_load(
                Blocks.to(gl.pointer_type(gl.uint32)) + address, valid, 0
            )
            vl, vh = unpack_pair(bw)
        else:
            vl, vh = (lo, hi)
        probability = weights[source] / denominator
        mixed_lo = mixed_lo + probability * vl
        mixed_hi = mixed_hi + probability * vh
    mixed_sq = gl.sum(
        gl.where(valid, mixed_lo * mixed_lo + mixed_hi * mixed_hi, 0.0), 0
    )
    publish_local(
        Stats + pid * 19 + 2 * (N_BLOCKS + 1),
        mixed_sq.to(gl.uint32, bitcast=True),
        epoch,
    )
    full_sq = stat_total(
        Stats, sample, 2 * (N_BLOCKS + 1), epoch, Errors, pid, WATCHDOG_TICKS
    )
    inv = gl.rsqrt(full_sq * (1.0 / 7168) + 1e-05)
    ow = gl.load(OutNorm.to(gl.pointer_type(gl.uint32)) + pair_in_row, valid, 0)
    ol, oh = unpack_pair(ow)
    result_lo = (mixed_lo * inv * ol).to(gl.bfloat16).to(gl.float32)
    result_hi = (mixed_hi * inv * oh).to(gl.bfloat16).to(gl.float32)
    gl.atomic_store(
        Moe.to(gl.pointer_type(gl.uint32)) + pair,
        pack_bf16(result_lo, result_hi),
        valid,
        sem="relaxed",
        scope="gpu",
    )
    if QUANTIZE:
        amax = gl.maximum(gl.abs(result_lo), gl.abs(result_hi))
        grouped = gl.reshape(amax, (64, 16))
        block_max = gl.max(grouped, 1)
        bits = (block_max * (1.0 / 448.0)).to(gl.uint32, bitcast=True)
        exponent = bits >> 23 & 255
        up = (bits & 4194304 != 0) & (
            (bits & 2097152 != 0) | (bits & 2097151 != 0) | (exponent > 0)
        )
        exponent += up.to(gl.uint32)
        scale = gl.where(
            block_max > 0, (exponent << 23).to(gl.float32, bitcast=True), 1.0
        )
        inverse = gl.reshape(
            (1.0 / scale)[:, None]
            + gl.full((64, 16), 0.0, gl.float32, grouped.type.layout),
            (1024,),
        )
        qlo = gl.minimum(gl.maximum(result_lo * inverse, -448.0), 448.0).to(
            gl.float8e4nv
        )
        qhi = gl.minimum(gl.maximum(result_hi * inverse, -448.0), 448.0).to(
            gl.float8e4nv
        )
        gl.atomic_store(
            Quant + pair * 2,
            qlo.to(gl.uint8, bitcast=True),
            valid,
            sem="relaxed",
            scope="gpu",
        )
        gl.atomic_store(
            Quant + pair * 2 + 1,
            qhi.to(gl.uint8, bitcast=True),
            valid,
            sem="relaxed",
            scope="gpu",
        )
        scale_layout: gl.constexpr = gl.SliceLayout(1, grouped.type.layout)
        group = chunk * 56 + gl.arange(0, 64, layout=scale_layout)
        offset = group // 8 * 256 + group % 4 * 64 + sample * 4 + group // 4 % 2 * 2
        gl.atomic_store(
            Scales + offset,
            exponent.to(gl.uint8),
            group < (chunk + 1) * 56,
            sem="relaxed",
            scope="gpu",
        )
    gl.inline_asm_elementwise(
        "s_waitcnt vmcnt(0)\n s_waitcnt lgkmcnt(0)\n s_mov_b32 $0, 0",
        "=s,~{memory}",
        [],
        dtype=gl.int32,
        is_pure=False,
        pack=1,
    )
    gl.barrier()
    gl.atomic_store(
        Ready + pid, epoch.to(gl.uint64) << 32 | 1, sem="relaxed", scope="gpu"
    )


@g.jit
def projection_attnres(
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
    B: gl.constexpr,
    RANK: gl.constexpr,
    N_BLOCKS: gl.constexpr,
    BOUNDARY: gl.constexpr = False,
    DELAY: gl.constexpr = 0,
    Epoch=None,
    DebugIPC=None,
    WORLD: gl.constexpr = 8,
):
    """Launch 256 resident CTAs, eight waves each, on unpartitioned gfx950."""
    pid = gl.program_id(0)
    if Epoch is None:
        epoch = gl.load(Epochs + pid) + 1
        gl.barrier()
    else:
        epoch = Epoch
    if pid < (B + 3) // 4 * 112:
        output_projection(
            X,
            W,
            Peers,
            Local,
            Delta,
            Epochs,
            Errors,
            B,
            RANK,
            DELAY_RANK=0,
            DELAY=DELAY,
            Attention=Attention,
            DELAY_CTA=(B + 3) // 4 * 112 - 1,
            Epoch=epoch,
            DebugIPC=DebugIPC,
            WORLD=WORLD,
        )
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
            Moe,
            Quant,
            Scales,
            Stats,
            Ready,
            Errors,
            epoch,
            pid,
            B,
            N_BLOCKS,
            BOUNDARY,
            True,
        )
    if Epoch is None:
        gl.atomic_store(Epochs + pid, epoch, sem="relaxed", scope="gpu")


@g.jit
def attnres_only(
    Delta,
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
    B: gl.constexpr,
    N_BLOCKS: gl.constexpr,
    BOUNDARY: gl.constexpr = False,
):
    pid = gl.program_id(0)
    epoch = gl.load(Epochs + pid) + 1
    attnres_chunk(
        Delta,
        Delta,
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
        Errors,
        epoch,
        pid,
        B,
        N_BLOCKS,
        BOUNDARY,
        False,
    )
    gl.atomic_store(Epochs + pid, epoch, sem="relaxed", scope="gpu")
