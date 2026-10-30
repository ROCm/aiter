# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Fused GDN decode step (conv1d + delta rule + gated RMSNorm) for any value/key
head ratio, gfx950. One program per (token, key-head group); the group's RATIO
``[V, K]`` state tiles pass through registers in row chunks, with ``NUM_BUFFERS``
chunk loads in flight. Supports the interleaved (Qwen3-Next) and flat (Qwen3.5/3.8) qkvz
packings and slot-strided state pools; generalises the ratio-2 kernels in
``fused_gdn_decode_qkvz.py``."""

from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from aiter.ops.triton._gluon_kernels.common.utils import softplus
from aiter.ops.triton._triton_kernels.common.activation import _silu
from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr

from .fused_gdn_decode_qkvz import _exp, _quantization_scale, _sigmoid

_ratio_repr = make_kernel_repr(
    "_decode_group_ratio",
    ["RATIO", "NW", "KL", "KS", "HAS_FP8", "FLAT", "NUM_BUFFERS", "BA_KSPLIT"],
)


@gluon.jit
def _conv_loads(
    Projection,
    State,
    Weight,
    Bias,
    projection_offset,
    channel,
    slot,
    conv_slot_stride,
    conv_channel_stride,
    conv_width_stride,
):
    """Inputs of the width-four causal conv step, one channel per lane. Loads are
    unmasked (padding lanes carry a clamped, valid channel)."""
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
    return x, history_ptr, h0, h1, h2, w0, w1, w2, w3, acc


@gluon.jit
def _conv_finish(
    x, history_ptr, h0, h1, h2, w0, w1, w2, w3, acc, live, conv_width_stride
):
    """Conv step on the inputs from _conv_loads. Only the history shift is masked,
    so a padding lane never races the channel's owner; also returns the shifted-in
    values, for a caller that stores the masked-off lanes later."""
    acc = gl.fma(h0, w0, acc)
    acc = gl.fma(h1, w1, acc)
    acc = gl.fma(h2, w2, acc)
    acc = gl.fma(x, w3, acc)
    activated = _silu(acc)
    gl.store(history_ptr, h1.to(gl.bfloat16), mask=live)
    gl.store(history_ptr + conv_width_stride, h2.to(gl.bfloat16), mask=live)
    gl.store(history_ptr + 2 * conv_width_stride, x.to(gl.bfloat16), mask=live)
    # bf16 boundary as in the reference
    return activated.to(gl.bfloat16).to(gl.float32), history_ptr, h1, h2, x


@gluon.jit
def _gating(
    BA,
    ALog,
    DTBias,
    token,
    head,
    ba_row_stride,
    ba_split_stride,
    VH: gl.constexpr,
    RATIO: gl.constexpr,
    FLAT: gl.constexpr,
    BA_KSPLIT: gl.constexpr,
):
    """Gating inputs of one value head (loads only, so they can all be in flight).
    With BA_KSPLIT > 1, BA holds that many fp32 split-K partials of ba; their sum is
    rounded to bf16 like the GEMM output it replaces."""
    if FLAT:
        b_offset = token * ba_row_stride + head
        a_offset = b_offset + VH
    else:
        b_offset = token * ba_row_stride + (head // RATIO) * (2 * RATIO) + head % RATIO
        a_offset = b_offset + RATIO
    if BA_KSPLIT == 1:
        b = gl.load(BA + b_offset).to(gl.float32)
        a = gl.load(BA + a_offset).to(gl.float32)
    else:
        b = gl.load(BA + b_offset)
        a = gl.load(BA + a_offset)
        for s in gl.static_range(1, BA_KSPLIT):
            b += gl.load(BA + s * ba_split_stride + b_offset)
            a += gl.load(BA + s * ba_split_stride + a_offset)
        b = b.to(gl.bfloat16).to(gl.float32)
        a = a.to(gl.bfloat16).to(gl.float32)
    a_log = gl.load(ALog + head)
    bias = gl.load(DTBias + head).to(gl.float32)
    return b, a, a_log, bias


@gluon.jit
def _dynamics(b, a, a_log, bias):
    decay = _exp(-_exp(a_log) * softplus(a + bias))
    beta = _sigmoid(b).to(gl.bfloat16).to(gl.float32)
    return decay, beta


@gluon.jit
def _gate_loads(
    gate_ptr,
    NormWeight,
    rows,
    V: gl.constexpr,
    ROWS: gl.constexpr,
    HEADS: gl.constexpr,
    BLOCKS: gl.constexpr,
):
    """Output gates (HEADS x BLOCKS row blocks) and the norm weight (BLOCKS blocks)."""
    gates = ()
    for h in gl.static_range(HEADS):
        for part in gl.static_range(BLOCKS):
            gates = gates + (gl.load(gate_ptr + h * V + part * ROWS).to(gl.float32),)
    weights = ()
    for part in gl.static_range(BLOCKS):
        weights = weights + (gl.load(NormWeight + part * ROWS + rows).to(gl.float32),)
    return gates, weights


@gluon.jit
def _epilogue(
    outs,
    gates,
    weights,
    row,
    rows,
    Output,
    Quantized,
    Scales,
    eps,
    fp8_max,
    V: gl.constexpr,
    ROWS: gl.constexpr,
    BLOCKS: gl.constexpr,
    HAS_FP8: gl.constexpr,
):
    """Gated RMSNorm of one head's bf16-rounded readout, given as BLOCKS row blocks,
    plus optional FP8."""
    cores = ()
    for part in gl.static_range(BLOCKS):
        cores = cores + (outs[part].to(gl.bfloat16).to(gl.float32),)
    squares = cores[0] * cores[0]
    for part in gl.static_range(1, BLOCKS):
        squares = squares + cores[part] * cores[part]
    rms = gl.rsqrt(gl.sum(squares, 0) / V + eps)
    normed = ()
    for part in gl.static_range(BLOCKS):
        value = (cores[part] * rms * weights[part] * _silu(gates[part])).to(gl.bfloat16)
        normed = normed + (value,)
        gl.store(Output + row * V + part * ROWS + rows, value)
    if HAS_FP8:
        # FP8 scales consume the rounded BF16 output, not the FP32 precursor.
        peak = gl.abs(normed[0].to(gl.float32))
        for part in gl.static_range(1, BLOCKS):
            peak = gl.maximum(peak, gl.abs(normed[part].to(gl.float32)))
        maximum = gl.maximum(gl.max(peak, 0), 1.0e-10)
        quant_scale = _quantization_scale(maximum, fp8_max, False)
        gl.store(Scales + row, quant_scale)
        for part in gl.static_range(BLOCKS):
            gl.store(
                Quantized + row * V + part * ROWS + rows,
                gl.clamp(normed[part].to(gl.float32) / quant_scale, -fp8_max, fp8_max),
            )


@gluon.constexpr_function
def _lds_epilogue_layout(V, HP2, NW):
    """[HP2, V] layout of the shared-memory epilogue: up to V / 64 warps per head row, any
    other warps on the head axis (replicated when that exceeds HP2)."""
    wpr = max(min(NW // HP2, V // 64), 1)
    return gl.BlockedLayout([1, V // (64 * wpr)], [1, 64], [NW // wpr, wpr], [1, 0])


@gluon.jit
def _lds_gate_loads(
    gate_base,
    NormWeight,
    V: gl.constexpr,
    HEADS: gl.constexpr,
    HP2: gl.constexpr,
    NW: gl.constexpr,
):
    """Output gates and norm weight in the shared-memory epilogue's layout."""
    out_layout: gl.constexpr = _lds_epilogue_layout(V, HP2, NW)
    head = gl.arange(0, HP2, layout=gl.SliceLayout(1, out_layout))
    col = gl.arange(0, V, layout=gl.SliceLayout(0, out_layout))
    live = (head[:, None] < HEADS) & (col[None, :] < V)
    gate = gl.load(gate_base + head[:, None] * V + col[None, :], mask=live, other=0.0)
    return gate, gl.load(NormWeight + col)


@gluon.jit
def _lds_epilogue(
    readouts,
    gate,
    weight,
    Output,
    row,
    eps,
    V: gl.constexpr,
    HEADS: gl.constexpr,
    HP2: gl.constexpr,
    NW: gl.constexpr,
):
    """Gated RMSNorm of the HEADS readouts staged in ``readouts`` ([HP2 * V] LDS),
    one head per warp row, so the sum of squares stays within a warp."""
    out_layout: gl.constexpr = _lds_epilogue_layout(V, HP2, NW)
    head = gl.arange(0, HP2, layout=gl.SliceLayout(1, out_layout))
    col = gl.arange(0, V, layout=gl.SliceLayout(0, out_layout))
    live = (head[:, None] < HEADS) & (col[None, :] < V)
    core = gl.where(
        live,
        readouts.reshape([HP2, V]).load(out_layout).to(gl.bfloat16).to(gl.float32),
        0.0,
    )
    rms = gl.rsqrt(gl.sum(core * core, 1) / V + eps)
    value = (
        core
        * rms[:, None]
        * weight.to(gl.float32)[None, :]
        * _silu(gate.to(gl.float32))
    )
    gl.store(
        Output + (row + head)[:, None] * V + col[None, :],
        value.to(gl.bfloat16),
        mask=live,
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
    proj_row_stride,
    ba_row_stride,
    ba_split_stride,
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
    REG_EPILOGUE: gl.constexpr = False,
    NUM_BUFFERS: gl.constexpr = 4,
    BA_KSPLIT: gl.constexpr = 1,
):
    """One program per (token, key-head group). Rows with an empty ``QueryStartLoc`` segment (``HAS_QSL``) or ``PAD_SLOT_ID`` are
    padded and skipped; ``conv_*_stride``
    / ``ssm_slot_stride`` are the element strides of the (possibly paged) pools, and
    ``proj_row_stride`` / ``ba_row_stride`` the token-row strides of the projections.
    ``State`` is fp32 or bf16; the update runs in fp32 and is stored in its dtype."""
    gl.static_assert(VH == RATIO * KH, "VH must be RATIO * KH")
    gl.static_assert(RATIO >= 1, "RATIO must be positive")
    gl.static_assert(K % (KL * KS) == 0, "KL lanes x KS elements must tile K")
    gl.static_assert(not HAS_FP8 or REG_EPILOGUE, "FP8 output needs REG_EPILOGUE")
    pid = gl.program_id(0)
    token = pid // KH
    group = pid % KH
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
    head0 = group * RATIO
    state_base = State + slot * ssm_slot_stride + head0 * V * K
    states = ()
    # RC-row chunks with NUM_BUFFERS loads in flight, issued after the conv and gate loads.
    RC: gl.constexpr = (64 // KL) * NW
    gl.static_assert(V % RC == 0, "a chunk must not straddle two heads")
    NCH: gl.constexpr = RATIO * V // RC
    NPRE: gl.constexpr = NUM_BUFFERS if NUM_BUFFERS < NCH else NCH  # noqa: FURB136
    HP2: gl.constexpr = 1 if RATIO == 1 else (2 if RATIO == 2 else 4)
    crow = gl.arange(0, RC, layout=gl.SliceLayout(1, tile_layout))
    chunk_offset = crow[:, None] * K + k[None, :]
    bs = ()
    as_ = ()
    logs = ()
    biases = ()
    for h in gl.static_range(RATIO):
        b, a, a_log, bias = _gating(
            BA,
            ALog,
            DtBias,
            token,
            head0 + h,
            ba_row_stride,
            ba_split_stride,
            VH,
            RATIO,
            FLAT,
            BA_KSPLIT,
        )
        bs = bs + (b,)
        as_ = as_ + (a,)
        logs = logs + (a_log,)
        biases = biases + (bias,)
    if FLAT:
        proj_row = token * proj_row_stride
        gate_base = Projection + proj_row + 2 * KH * K + VH * V + head0 * V
    else:
        base = token * proj_row_stride + group * group_width
        gate_base = Projection + base + 2 * K + RATIO * V
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
    live = c < conv_width
    x, history_ptr, h0, h1, h2, w0, w1, w2, w3, acc = _conv_loads(
        Projection,
        ConvState,
        ConvWeight,
        ConvBias,
        proj_offset,
        channel,
        slot,
        conv_slot_stride,
        conv_channel_stride,
        conv_width_stride,
    )
    if REG_EPILOGUE:
        gates, weights = _gate_loads(
            gate_base + crow, NormWeight, crow, V, RC, RATIO, V // RC
        )
    else:
        gate, weight = _lds_gate_loads(gate_base, NormWeight, V, RATIO, HP2, NW)
    for ci in gl.static_range(NPRE):
        states = states + (
            gl.amd.cdna3.buffer_load(
                state_base + ci * RC * K, chunk_offset, cache=".cg"
            ),
        )
    activated, history_ptr, h1, h2, x = _conv_finish(
        x, history_ptr, h0, h1, h2, w0, w1, w2, w3, acc, live, conv_width_stride
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
    decays = ()
    betas = ()
    for h in gl.static_range(RATIO):
        decay, beta = _dynamics(bs[h], as_[h], logs[h], biases[h])
        decays = decays + (decay,)
        betas = betas + (beta,)
    # Readout from the decayed state before the write:
    # (S + r k^T) q = S q + r (k . q), so both row sums are independent.
    qk = gl.sum(q * key, 0)
    if not REG_EPILOGUE:
        readouts = gl.allocate_shared_memory(gl.float32, [HP2 * V], shared_layout)
    for c in gl.static_range(NCH):
        chunk = states[0].to(gl.float32)
        if c + NUM_BUFFERS < NCH:
            states = states[1:] + (
                gl.amd.cdna3.buffer_load(
                    state_base + (c + NUM_BUFFERS) * RC * K,
                    chunk_offset,
                    cache=".cg",
                ),
            )
        else:
            states = states[1:]
        value = conv_shared.slice(2 * K + c * RC, RC).load(
            gl.SliceLayout(1, tile_layout)
        )
        decayed = chunk * decays[c // (V // RC)]
        predicted = gl.sum(decayed * key[None, :], 1)
        readout = gl.sum(decayed * q[None, :], 1)
        residual = (value - predicted) * betas[c // (V // RC)]
        gl.amd.cdna3.buffer_store(
            (decayed + residual[:, None] * key[None, :]).to(State.dtype.element_ty),
            state_base + c * RC * K,
            chunk_offset,
            cache=".wt",
        )
        if REG_EPILOGUE:
            outs = outs + (readout + residual * qk,)
        else:
            readouts.slice(c * RC, RC).store(readout + residual * qk)
    if REG_EPILOGUE:
        for h in gl.static_range(RATIO):
            outs_h = ()
            gates_h = ()
            for part in gl.static_range(V // RC):
                outs_h = outs_h + (outs[(V // RC) * h + part],)
                gates_h = gates_h + (gates[(V // RC) * h + part],)
            _epilogue(
                outs_h,
                gates_h,
                weights,
                token * VH + head0 + h,
                crow,
                Output,
                Quantized,
                Scales,
                eps,
                fp8_max,
                V,
                RC,
                V // RC,
                HAS_FP8,
            )
    else:
        _lds_epilogue(
            readouts,
            gate,
            weight,
            Output,
            token * VH + head0,
            eps,
            V,
            RATIO,
            HP2,
            NW,
        )
