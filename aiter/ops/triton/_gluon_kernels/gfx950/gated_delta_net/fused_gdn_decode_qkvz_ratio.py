# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Fused GDN decode step (conv1d + delta rule + gated RMSNorm) for any value/key
head ratio, gfx950. One program per (token, key-head group); the group's RATIO
``[V, K]`` fp32 state tiles stay in registers between the two matrix-vector
products. Supports the interleaved (Qwen3-Next) and flat (Qwen3.5/3.8) qkvz
packings and slot-strided state pools; generalises the ratio-2 kernels in
``fused_gdn_decode_qkvz.py``."""

from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from aiter.ops.triton._gluon_kernels.common.utils import softplus
from aiter.ops.triton._triton_kernels.common.activation import _silu
from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr

from .fused_gdn_decode_qkvz import _exp, _quantization_scale, _sigmoid

_ratio_repr = make_kernel_repr(
    "_decode_group_ratio", ["RATIO", "NW", "KL", "KS", "HAS_FP8", "FLAT"]
)


@gluon.jit
def _conv_channels_clamped(
    Projection,
    State,
    Weight,
    Bias,
    projection_offset,
    channel,
    slot,
    live,
    conv_slot_stride,
    conv_channel_stride,
    conv_width_stride,
):
    """Width-four causal conv step, one channel per lane. Loads are unmasked
    (padding lanes carry a clamped, valid channel); only the history shift is
    masked so a padding lane never races the channel's owner."""
    x = gl.load(Projection + projection_offset).to(gl.float32)
    history_ptr = State + slot * conv_slot_stride + channel * conv_channel_stride
    h0 = gl.load(history_ptr).to(gl.float32)
    h1 = gl.load(history_ptr + conv_width_stride).to(gl.float32)
    h2 = gl.load(history_ptr + 2 * conv_width_stride).to(gl.float32)
    w0 = gl.load(Weight + channel * 4).to(gl.float32)
    w1 = gl.load(Weight + channel * 4 + 1).to(gl.float32)
    w2 = gl.load(Weight + channel * 4 + 2).to(gl.float32)
    w3 = gl.load(Weight + channel * 4 + 3).to(gl.float32)
    acc = gl.load(Bias + channel).to(gl.float32)
    acc = gl.fma(h0, w0, acc)
    acc = gl.fma(h1, w1, acc)
    acc = gl.fma(h2, w2, acc)
    acc = gl.fma(x, w3, acc)
    activated = _silu(acc)
    gl.store(history_ptr, h1.to(gl.bfloat16), mask=live)
    gl.store(history_ptr + conv_width_stride, h2.to(gl.bfloat16), mask=live)
    gl.store(history_ptr + 2 * conv_width_stride, x.to(gl.bfloat16), mask=live)
    return activated.to(gl.bfloat16).to(gl.float32)  # bf16 boundary as in the reference


@gluon.jit
def _gating(
    BA,
    ALog,
    DTBias,
    token,
    head,
    VH: gl.constexpr,
    RATIO: gl.constexpr,
    FLAT: gl.constexpr,
):
    """Gating inputs of one value head (loads only, so they can all be in flight)."""
    if FLAT:
        b = gl.load(BA + token * 2 * VH + head).to(gl.float32)
        a = gl.load(BA + token * 2 * VH + VH + head).to(gl.float32)
    else:
        offset = token * 2 * VH + (head // RATIO) * (2 * RATIO) + head % RATIO
        b = gl.load(BA + offset).to(gl.float32)
        a = gl.load(BA + offset + RATIO).to(gl.float32)
    a_log = gl.load(ALog + head)
    bias = gl.load(DTBias + head).to(gl.float32)
    return b, a, a_log, bias


@gluon.jit
def _dynamics(b, a, a_log, bias):
    decay = _exp(-_exp(a_log) * softplus(a + bias))
    beta = _sigmoid(b).to(gl.bfloat16).to(gl.float32)
    return decay, beta


@gluon.jit
def _update(state, q, key, value, decay, beta):
    """Delta-rule step on one [V, K] tile: returns (updated state, readout)."""
    decayed = state * decay
    predicted = gl.sum(decayed * key[None, :], 1)
    residual = (value - predicted) * beta
    updated = decayed + residual[:, None] * key[None, :]
    out = gl.sum(updated * q[None, :], 1)
    return updated, out


@gluon.jit
def _epilogue(
    out,
    gate,
    weight,
    row,
    vrow,
    Output,
    Quantized,
    Scales,
    eps,
    fp8_max,
    V: gl.constexpr,
    HAS_FP8: gl.constexpr,
):
    """Gated RMSNorm of one head's bf16-rounded readout, plus optional FP8."""
    core = out.to(gl.bfloat16).to(gl.float32)
    rms = gl.rsqrt(gl.sum(core * core, 0) / V + eps)
    normalized = (core * rms * weight * _silu(gate)).to(gl.bfloat16)
    gl.store(Output + row * V + vrow, normalized)
    if HAS_FP8:
        # FP8 scales consume the rounded BF16 output, not the FP32 precursor.
        values = normalized.to(gl.float32)
        maximum = gl.maximum(gl.max(gl.abs(values), 0), 1.0e-10)
        quant_scale = _quantization_scale(maximum, fp8_max, False)
        gl.store(Scales + row, quant_scale)
        gl.store(
            Quantized + row * V + vrow,
            gl.clamp(values / quant_scale, -fp8_max, fp8_max),
        )


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
):
    """One program per (token, key-head group). Rows with an empty ``QueryStartLoc``
    segment (``HAS_QSL``) or ``PAD_SLOT_ID`` are padded and skipped; ``conv_*_stride``
    / ``ssm_slot_stride`` are the element strides of the (possibly paged) pools."""
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
    # Kept nested: HAS_QSL is constexpr, so the loads exist only in variants that
    # take a query_start_loc (the pointer is a placeholder otherwise).
    if HAS_QSL:  # noqa: SIM102
        if gl.load(QueryStartLoc + token + 1) - gl.load(QueryStartLoc + token) == 0:
            return
    slot = gl.load(Indices + token).to(gl.int64)
    if slot == PAD_SLOT_ID:
        return

    # Phase 1: issue every independent load before the first wait.
    tile_layout: gl.constexpr = gl.BlockedLayout(
        [1, KS], [64 // KL, KL], [NW, 1], [1, 0]
    )
    k = gl.arange(0, K, layout=gl.SliceLayout(0, tile_layout))
    vrow = gl.arange(0, V, layout=gl.SliceLayout(1, tile_layout))
    state_offset = vrow[:, None] * K + k[None, :]
    head0 = group * RATIO
    state_base = State + slot * ssm_slot_stride + head0 * V * K
    states = ()
    for h in gl.static_range(RATIO):
        states = states + (
            gl.amd.cdna3.buffer_load(state_base + h * V * K, state_offset, cache=".cg"),
        )
    bs = ()
    as_ = ()
    logs = ()
    biases = ()
    for h in gl.static_range(RATIO):
        b, a, a_log, bias = _gating(BA, ALog, DtBias, token, head0 + h, VH, RATIO, FLAT)
        bs = bs + (b,)
        as_ = as_ + (a,)
        logs = logs + (a_log,)
        biases = biases + (bias,)
    if FLAT:
        proj_row = token * (KH * group_width)
        gate_ptr = Projection + proj_row + 2 * KH * K + VH * V + head0 * V + vrow
    else:
        base = token_group * group_width
        gate_ptr = Projection + base + 2 * K + RATIO * V + vrow
    gates = ()
    for h in gl.static_range(RATIO):
        gates = gates + (gl.load(gate_ptr + h * V).to(gl.float32),)
    weight = gl.load(NormWeight + vrow).to(gl.float32)
    # One lane per conv channel on a power-of-two range; padding lanes alias the
    # group's last channel (loads stay in bounds, their history stores are masked).
    conv_layout: gl.constexpr = gl.BlockedLayout([1], [64], [NW], [0])
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
    activated = _conv_channels_clamped(
        Projection,
        ConvState,
        ConvWeight,
        ConvBias,
        proj_offset,
        channel,
        slot,
        c < conv_width,
        conv_slot_stride,
        conv_channel_stride,
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
        updated, out = _update(states[h], q, key, value, decay, beta)
        gl.amd.cdna3.buffer_store(
            updated, state_base + h * V * K, state_offset, cache=".wt"
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
