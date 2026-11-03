# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Fused GDN decode step for gfx1250: qkvz/ba split, causal conv1d, delta-rule
update and gated RMSNorm, with an optional FP8 epilogue.

This is the gfx950 ``_decode_group_ratio`` kernel for the wave32 layouts and buffer
instructions of gfx1250. It imports the helpers of that kernel that do not depend
on the arch. One program updates the value heads of one (token, key-head group),
one head at a time. Thus it keeps only one ``[V, K]`` state tile in registers, and
two 16-wave programs fit on one CU. The kernel also accepts a bf16 state (the
update is in fp32), token-strided projections and no conv bias."""

from triton.experimental import gluon
from triton.experimental.gluon import language as gl

# Arch-independent helpers of the gfx950 kernel.
from aiter.ops.triton._gluon_kernels.gfx950.gated_delta_net.fused_gdn_decode_qkvz_ratio import (
    _dynamics,
    _epilogue,
    _update,
)
from aiter.ops.triton._triton_kernels.common.activation import _silu
from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr

_ratio_repr = make_kernel_repr(
    "_decode_group_ratio", ["RATIO", "NW", "KL", "KS", "HAS_FP8", "FLAT"]
)


@gluon.jit
def _conv_inputs(
    Projection, Weight, Bias, projection_offset, channel, HAS_BIAS: gl.constexpr
):
    """Conv operands that do not need the slot: the new input, the four taps and
    the bias of one channel per lane."""
    x = gl.load(Projection + projection_offset)
    w0 = gl.load(Weight + channel * 4)
    w1 = gl.load(Weight + channel * 4 + 1)
    w2 = gl.load(Weight + channel * 4 + 2)
    w3 = gl.load(Weight + channel * 4 + 3)
    if HAS_BIAS:
        bias = gl.load(Bias + channel)
    else:
        bias = gl.zeros_like(w0)
    return x, w0, w1, w2, w3, bias


@gluon.jit
def _conv_history(
    State, channel, slot, conv_slot_stride, conv_channel_stride, conv_width_stride
):
    """Three-entry conv window of one channel per lane. The loads are not masked:
    padding lanes use a valid, clamped channel."""
    history_ptr = State + slot * conv_slot_stride + channel * conv_channel_stride
    h0 = gl.load(history_ptr)
    h1 = gl.load(history_ptr + conv_width_stride)
    h2 = gl.load(history_ptr + 2 * conv_width_stride)
    return history_ptr, h0, h1, h2


@gluon.jit
def _conv_step(
    history_ptr, h0, h1, h2, x, w0, w1, w2, w3, bias, live, conv_width_stride
):
    """Width-four causal conv step for one channel per lane, then the window shift.
    Only the shift is masked, so a padding lane does not race the channel owner."""
    acc = gl.fma(h0.to(gl.float32), w0.to(gl.float32), bias.to(gl.float32))
    acc = gl.fma(h1.to(gl.float32), w1.to(gl.float32), acc)
    acc = gl.fma(h2.to(gl.float32), w2.to(gl.float32), acc)
    acc = gl.fma(x.to(gl.float32), w3.to(gl.float32), acc)
    activated = _silu(acc)
    gl.store(history_ptr, h1, mask=live)
    gl.store(history_ptr + conv_width_stride, h2, mask=live)
    gl.store(history_ptr + 2 * conv_width_stride, x.to(h2.dtype), mask=live)
    return activated.to(gl.bfloat16).to(gl.float32)  # bf16 boundary as in the reference


@gluon.jit
def _gating(
    BA,
    ALog,
    DTBias,
    ba_row,
    head,
    VH: gl.constexpr,
    RATIO: gl.constexpr,
    FLAT: gl.constexpr,
):
    """Gating inputs of one value head (loads only). Unlike the gfx950 helper, it
    takes the row offset of a token-strided ``BA`` and accepts a bf16 ``A_log``."""
    if FLAT:
        b = gl.load(BA + ba_row + head).to(gl.float32)
        a = gl.load(BA + ba_row + VH + head).to(gl.float32)
    else:
        offset = ba_row + (head // RATIO) * (2 * RATIO) + head % RATIO
        b = gl.load(BA + offset).to(gl.float32)
        a = gl.load(BA + offset + RATIO).to(gl.float32)
    a_log = gl.load(ALog + head).to(gl.float32)
    bias = gl.load(DTBias + head).to(gl.float32)
    return b, a, a_log, bias


@gluon.jit(repr=_ratio_repr)
def _decode_group_ratio(
    Projection,
    BA,
    ConvState,
    State,
    Indices,
    QueryStartLoc,
    ConvWeight,
    ConvBias,
    ALog,
    DtBias,
    NormWeight,
    Output,
    Quantized,
    Scales,
    scale,
    eps,
    fp8_max,
    proj_token_stride,
    ba_token_stride,
    conv_slot_stride,
    conv_channel_stride,
    conv_width_stride,
    ssm_slot_stride,
    KH: gl.constexpr,
    VH: gl.constexpr,
    K: gl.constexpr,
    V: gl.constexpr,
    RATIO: gl.constexpr,
    CONV_P2: gl.constexpr,
    NW: gl.constexpr,
    KL: gl.constexpr,
    KS: gl.constexpr,
    PAD_SLOT_ID: gl.constexpr,
    HAS_FP8: gl.constexpr,
    FLAT: gl.constexpr,
    HAS_QSL: gl.constexpr,
    HAS_BIAS: gl.constexpr,
):
    """One program per (token, key-head group). The kernel skips padded rows: rows
    with an empty ``QueryStartLoc`` segment (``HAS_QSL``) and rows whose slot is
    ``PAD_SLOT_ID``. The ``*_stride`` arguments are element strides."""
    gl.static_assert(VH == RATIO * KH, "VH must be RATIO * KH")
    gl.static_assert(RATIO >= 1, "RATIO must be positive")
    gl.static_assert(K % (KL * KS) == 0, "KL lanes x KS elements must tile K")
    token_group = gl.program_id(0)
    token = token_group // KH
    group = token_group % KH
    group_width: gl.constexpr = 2 * K + 2 * RATIO * V
    conv_width: gl.constexpr = 2 * K + RATIO * V
    gl.static_assert(
        CONV_P2 >= conv_width, "CONV_P2 must cover the group's conv channels"
    )
    # Phase 1: issue the loads that do not need the slot (projections, gating
    # inputs, conv taps, norm weight). Their latency then overlaps the slot load.
    tile_layout: gl.constexpr = gl.BlockedLayout(
        [1, KS], [32 // KL, KL], [NW, 1], [1, 0]
    )
    k = gl.arange(0, K, layout=gl.SliceLayout(0, tile_layout))
    vrow = gl.arange(0, V, layout=gl.SliceLayout(1, tile_layout))
    state_offset = vrow[:, None] * K + k[None, :]
    head0 = group * RATIO
    ba_row = token * ba_token_stride
    bs = ()
    as_ = ()
    logs = ()
    biases = ()
    for h in gl.static_range(RATIO):
        b, a, a_log, bias = _gating(
            BA, ALog, DtBias, ba_row, head0 + h, VH, RATIO, FLAT
        )
        bs = bs + (b,)
        as_ = as_ + (a,)
        logs = logs + (a_log,)
        biases = biases + (bias,)
    proj_row = token * proj_token_stride
    if FLAT:
        gate_ptr = Projection + proj_row + 2 * KH * K + VH * V + head0 * V + vrow
    else:
        base = proj_row + group * group_width
        gate_ptr = Projection + base + 2 * K + RATIO * V + vrow
    gates = ()
    for h in gl.static_range(RATIO):
        gates = gates + (gl.load(gate_ptr + h * V).to(gl.float32),)
    weight = gl.load(NormWeight + vrow).to(gl.float32)
    # One lane per conv channel. Padding lanes use the last channel of the group:
    # their loads stay in bounds and their history stores are masked.
    conv_layout: gl.constexpr = gl.BlockedLayout([1], [32], [NW], [0])
    c = gl.arange(0, CONV_P2, layout=conv_layout)
    cc = gl.minimum(c, conv_width - 1)
    channel = gl.where(
        cc < K,
        group * K + cc,
        gl.where(
            cc < 2 * K,
            KH * K + group * K + cc - K,
            2 * KH * K + group * RATIO * V + cc - 2 * K,
        ),
    )
    if FLAT:
        proj_offset = proj_row + channel
    else:
        proj_offset = base + cc
    x, w0, w1, w2, w3, conv_bias = _conv_inputs(
        Projection, ConvWeight, ConvBias, proj_offset, channel, HAS_BIAS
    )

    # Nested, because HAS_QSL is constexpr: without it, QueryStartLoc is only a
    # placeholder pointer and the kernel must not load it.
    if HAS_QSL:  # noqa: SIM102
        if gl.load(QueryStartLoc + token + 1) - gl.load(QueryStartLoc + token) == 0:
            return
    slot = gl.load(Indices + token).to(gl.int64)
    if slot == PAD_SLOT_ID:
        return
    # The conv window needs the slot. Each state tile loads later, just before the
    # update of its head.
    history_ptr, h0, h1, h2 = _conv_history(
        ConvState,
        channel,
        slot,
        conv_slot_stride,
        conv_channel_stride,
        conv_width_stride,
    )
    state_base = State + slot * ssm_slot_stride + head0 * V * K
    activated = _conv_step(
        history_ptr,
        h0,
        h1,
        h2,
        x,
        w0,
        w1,
        w2,
        w3,
        conv_bias,
        c < conv_width,
        conv_width_stride,
    )
    # Phase 2: conv through LDS, q/k norms, per-head update and store.
    shared_layout: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [0])
    conv_shared = gl.allocate_shared_memory(
        gl.float32, [CONV_P2], shared_layout, activated
    )
    q = conv_shared.slice(0, K).load(gl.SliceLayout(0, tile_layout))
    key = conv_shared.slice(K, K).load(gl.SliceLayout(0, tile_layout))
    q = q * (gl.rsqrt(gl.sum(q * q, 0) + 1.0e-6) * scale)
    key = key * gl.rsqrt(gl.sum(key * key, 0) + 1.0e-6)
    outs = ()
    for h in gl.static_range(RATIO):
        decay, beta = _dynamics(bs[h], as_[h], logs[h], biases[h])
        value = conv_shared.slice(2 * K + h * V, V).load(gl.SliceLayout(1, tile_layout))
        state = gl.amd.gfx1250.buffer_load(
            state_base + h * V * K, state_offset, cache=".cg"
        ).to(gl.float32)
        updated, out = _update(state, q, key, value, decay, beta)
        gl.amd.gfx1250.buffer_store(
            updated.to(State.dtype.element_ty),
            state_base + h * V * K,
            state_offset,
            cache=".wt",
        )
        outs = outs + (out,)
    # Phase 3: per-head gated RMSNorm (+ FP8) from registers.
    row = token * VH + head0
    for h in gl.static_range(RATIO):
        _epilogue(
            outs[h],
            gates[h],
            weight,
            row + h,
            vrow,
            Output,
            Quantized,
            Scales,
            eps,
            fp8_max,
            V,
            HAS_FP8,
        )
