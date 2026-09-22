# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Gluon kernels for the fused Qwen3-Next GDN decode step (gfx950/CDNA4).

One eight-wave workgroup owns a token/key-head group's convolution history and
both FP32 value-head states, fusing: the packed qkvz/ba split, a depthwise
causal conv1d with bias and SiLU, the delta-rule gating, the FP32 recurrent
state update, a gated RMSNorm with a SiLU gate, and an optional per-head
group-128 FP8 quantization epilogue.

Partial-column prefetch overlaps state reads with QKV preparation. Pool-size
bounds select safe 32-bit or full-width addressing. Wave-local quantization
preserves BF16 rounding and maximum-number semantics.

Rows whose state index equals ``PAD_SLOT_ID`` are skipped without reading or
writing state; their outputs are undefined.

Generated from Artemis kernel pack source, then modified to add the pad-slot
guard and the optional FP8 epilogue. Do not hand-edit; see build_aiter_kernel.py.

  source: gdn_decode_m16.py
  sha256: 16ec8eaf1236ed1f62b495fcc4aaa9294a35e2856affd5d0672d48d5cfae6682
  source: gdn_decode_m128_m256.py   (large-batch band)
  sha256: 339400d8b376329c3a727e6c1258d5c43bf54d3147f5707d257ecd1db0e3c9ad
"""

from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

# Config-aware names so a trace row maps to the exact specialization
# (aiter/ops/triton/README.md, "Config-aware kernel names in traces").
# PAD_SLOT_ID is deliberately excluded: it is a protocol sentinel rather than a
# tuning key, and make_kernel_repr does not sanitize the sign of a negative
# value, so including it emits an invalid function identifier.
_small_repr = make_kernel_repr("_fused_decode", ["BATCH", "INDEX64", "HAS_FP8"])
_tiled_repr = make_kernel_repr("_fused_decode_tiled", ["B32", "INDEX64", "HAS_FP8"])
_group_repr = make_kernel_repr("_decode_group", ["NW", "KL", "KS", "HAS_FP8"])


@gluon.jit
def _exp(value):
    return gl.exp2(value * 1.4426950408889634)


@gluon.jit
def _sigmoid(value):
    denominator = 1.0 + _exp(-value)
    return gl.inline_asm_elementwise(
        "v_rcp_f32 $0, $1",
        "=v,v",
        (denominator,),
        dtype=gl.float32,
        is_pure=True,
        pack=1,
    )


@gluon.jit
def _quantization_scale(magnitude, quant_max, FAST_QUANT: gl.constexpr):
    # Both format bounds and floored BF16 magnitudes give normal quotients.
    # Quotient residual correction suffices for their rounding; the optional
    # reciprocal refinement is retained where its schedule is faster.
    # Division fixup preserves infinity/NaN behavior without range rescaling.
    # The wave maximum has already applied the 1e-10 floor.
    numerator = magnitude
    inverse = gl.inline_asm_elementwise(
        "v_rcp_f32 $0, $1",
        "=v,v",
        (quant_max,),
        dtype=gl.float32,
        is_pure=True,
        pack=1,
    )
    if not FAST_QUANT:
        inverse = gl.fma(gl.fma(-quant_max, inverse, 1.0), inverse, inverse)
    quotient = numerator * inverse
    residual = gl.fma(-quotient, quant_max, numerator)
    quotient = gl.fma(residual, inverse, quotient)
    return gl.inline_asm_elementwise(
        "v_div_fixup_f32 $0, $1, $2, $3",
        "=v,v,v,v",
        (quotient, quant_max, numerator),
        dtype=gl.float32,
        is_pure=True,
        pack=1,
    )


@gluon.jit
def _convolve(X, History, Weight, Bias, x_offset, history_offset, channel):
    x = gl.load(X + x_offset).to(gl.float32)
    h0 = gl.load(History + history_offset).to(gl.float32)
    h1 = gl.load(History + history_offset + 1).to(gl.float32)
    h2 = gl.load(History + history_offset + 2).to(gl.float32)
    w0 = gl.load(Weight + channel * 4).to(gl.float32)
    w1 = gl.load(Weight + channel * 4 + 1).to(gl.float32)
    w2 = gl.load(Weight + channel * 4 + 2).to(gl.float32)
    w3 = gl.load(Weight + channel * 4 + 3).to(gl.float32)
    bias = gl.load(Bias + channel).to(gl.float32)
    acc = gl.fma(h0, w0, bias)
    acc = gl.fma(h1, w1, acc)
    acc = gl.fma(h2, w2, acc)
    acc = gl.fma(x, w3, acc)
    gl.store(History + history_offset, h1)
    gl.store(History + history_offset + 1, h2)
    gl.store(History + history_offset + 2, x)
    return (acc * _sigmoid(acc)).to(gl.bfloat16).to(gl.float32)


@gluon.jit
def _head_dynamics(
    BA, ALog, DTBias, token, head, VH: gl.constexpr, NATIVE_LOG: gl.constexpr
):
    offset = token * 2 * VH + (head // 2) * 4 + head % 2
    b = gl.load(BA + offset).to(gl.float32)
    a = gl.load(BA + offset + 2).to(gl.float32)
    a_log = gl.load(ALog + head)
    bias = gl.load(DTBias + head).to(gl.float32)
    arg = a + bias
    if NATIVE_LOG:
        # The logarithm's argument is >= 1, so no subnormal rescaling is needed.
        log2 = gl.inline_asm_elementwise(
            "v_log_f32 $0, $1",
            "=v,v",
            (1.0 + _exp(arg),),
            dtype=gl.float32,
            is_pure=True,
            pack=1,
        )
        softplus = gl.where(arg <= 20.0, log2 * 0.6931471805599453, arg)
    else:
        softplus = gl.where(arg <= 20.0, gl.log(1.0 + _exp(arg)), arg)
    decay = _exp(-_exp(a_log) * softplus)
    beta = _sigmoid(b).to(gl.bfloat16).to(gl.float32)
    return decay, beta


@gluon.jit
def _load_epilogue_inputs(
    X, NormWeight, token, k_head, KH: gl.constexpr, EARLY_SIGMOID: gl.constexpr
):
    layout: gl.constexpr = gl.BlockedLayout([1, 2], [1, 64], [8, 1], [1, 0])
    head = gl.arange(0, 2, gl.SliceLayout(1, layout))
    col = gl.arange(0, 128, gl.SliceLayout(0, layout))
    z_offset = (token * KH + k_head) * 768 + 512 + head[:, None] * 128 + col[None, :]
    z = gl.load(X + z_offset).to(gl.float32)
    weight = gl.load(NormWeight + col).to(gl.float32)
    sigmoid = _sigmoid(z) if EARLY_SIGMOID else z
    return z, weight, sigmoid


@gluon.jit
def _wave_maximum_pair(a, b):
    # All 64 lanes own adjacent pairs from the same head. Max-number ignores
    # mixed NaNs; BF16 rounding has quieted any signaling inputs. The floor
    # also supplies a finite all-NaN identity. Explicit waits cover DPP and
    # readlane operand hazards, including the scalar result's consumers.
    maximum, _scratch = gl.inline_asm_elementwise(
        """
        v_max_f32 $1, |$2|, |$3|
        v_max_f32 $1, 0x2edbe6ff, $1
        s_nop 1
        v_max_f32 $1, $1, $1 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
        s_nop 1
        v_max_f32 $1, $1, $1 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
        s_nop 1
        v_max_f32 $1, $1, $1 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
        s_nop 1
        v_max_f32 $1, $1, $1 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
        s_nop 1
        v_max_f32 $1, $1, $1 row_bcast:15 row_mask:0xa bank_mask:0xf bound_ctrl:1
        s_nop 1
        v_max_f32 $1, $1, $1 row_bcast:31 row_mask:0xc bank_mask:0xf bound_ctrl:1
        s_nop 0
        v_readlane_b32 $0, $1, 63
        s_nop 1
        """,
        "=&s,=&v,v,v",
        (a, b),
        dtype=(gl.float32, gl.float32),
        is_pure=True,
        pack=1,
    )
    return maximum


@gluon.jit
def _wave_sum_pair(a, b):
    total, _scratch = gl.inline_asm_elementwise(
        """
        v_add_f32 $1, $2, $3
        s_nop 1
        v_add_f32 $1, $1, $1 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
        s_nop 1
        v_add_f32 $1, $1, $1 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
        s_nop 1
        v_add_f32 $1, $1, $1 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
        s_nop 1
        v_add_f32 $1, $1, $1 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
        s_nop 1
        v_add_f32 $1, $1, $1 row_bcast:15 row_mask:0xa bank_mask:0xf bound_ctrl:1
        s_nop 1
        v_add_f32 $1, $1, $1 row_bcast:31 row_mask:0xc bank_mask:0xf bound_ctrl:1
        s_nop 0
        v_readlane_b32 $0, $1, 63
        s_nop 1
        """,
        "=&s,=&v,v,v",
        (a, b),
        dtype=(gl.float32, gl.float32),
        is_pure=True,
        pack=1,
    )
    return total


@gluon.jit
def _finish_pair(
    core,
    Output,
    Quantized,
    Scales,
    token,
    k_head,
    eps,
    quant_max,
    VH: gl.constexpr,
    z,
    weight,
    cached_sigmoid,
    EARLY_SIGMOID: gl.constexpr,
    LATE_STORES: gl.constexpr,
    FAST_QUANT: gl.constexpr,
    DPP_RMS: gl.constexpr,
    HAS_FP8: gl.constexpr,
):
    # Each complete head is reduced within one wave.
    layout: gl.constexpr = gl.BlockedLayout([1, 2], [1, 64], [8, 1], [1, 0])
    head = gl.arange(0, 2, gl.SliceLayout(1, layout))
    col = gl.arange(0, 128, gl.SliceLayout(0, layout))
    values = gl.convert_layout(core, layout).to(gl.float32)
    if DPP_RMS:
        square0, square1 = gl.split((values * values).reshape((2, 64, 2)))
        square_sum = _wave_sum_pair(square0, square1)
        inv_rms_pair = gl.rsqrt(square_sum / 128.0 + eps)
        inv_rms = gl.join(inv_rms_pair, inv_rms_pair).reshape((2, 128))
    else:
        inv_rms = gl.rsqrt(gl.sum(values * values, 1) / 128.0 + eps)[:, None]
    sigmoid = cached_sigmoid if EARLY_SIGMOID else _sigmoid(z)
    normalized = (values * inv_rms * weight[None, :] * z * sigmoid).to(gl.bfloat16)
    offset = (token * VH + k_head * 2 + head[:, None]) * 128 + col[None, :]
    if not LATE_STORES:
        gl.store(Output + offset, normalized)

    if not HAS_FP8:
        # bf16-only caller: skip the group quantization entirely.
        if LATE_STORES:
            gl.store(Output + offset, normalized)
        return

    # Quantize the externally visible BF16 output, preserving its rounding point.
    values = normalized.to(gl.float32)
    value0, value1 = gl.split(values.reshape((2, 64, 2)))
    group_max = _wave_maximum_pair(value0, value1)
    scale_pair = _quantization_scale(group_max, quant_max, FAST_QUANT)
    group_scale = gl.join(scale_pair, scale_pair).reshape((2, 128))
    if not LATE_STORES:
        gl.store(
            Scales
            + token * VH
            + k_head * 2
            + head[:, None]
            + gl.zeros((1, 128), gl.int32, layout),
            group_scale,
            col[None, :] == 0,
        )

    # The scale floor and BF16 range keep finite scales and their reciprocals
    # normal in both supported formats. Share the reciprocal across the head,
    # retaining the quotient residual correction at FP8 rounding boundaries.
    # Division fixup preserves signed zeros and exceptional-value behavior.
    inv_scale = gl.inline_asm_elementwise(
        "v_rcp_f32 $0, $1",
        "=v,v",
        (group_scale,),
        dtype=gl.float32,
        is_pure=True,
        pack=1,
    )
    if not FAST_QUANT:
        inv_scale = gl.fma(gl.fma(-group_scale, inv_scale, 1.0), inv_scale, inv_scale)
    quotient = values * inv_scale
    error = gl.fma(-quotient, group_scale, values)
    quotient = gl.fma(error, inv_scale, quotient)
    quotient = gl.inline_asm_elementwise(
        "v_div_fixup_f32 $0, $1, $2, $3",
        "=v,v,v,v",
        (quotient, group_scale, values),
        dtype=gl.float32,
        is_pure=True,
        pack=1,
    )
    quantized = gl.clamp(quotient, -quant_max, quant_max)
    if LATE_STORES:
        gl.store(Output + offset, normalized)
    gl.store(Quantized + offset, quantized)
    if LATE_STORES:
        gl.store(
            Scales
            + token * VH
            + k_head * 2
            + head[:, None]
            + gl.zeros((1, 128), gl.int32, layout),
            group_scale,
            col[None, :] == 0,
        )


@gluon.jit
def _load_state(base, offset, BUFFER_STATE: gl.constexpr):
    if BUFFER_STATE:
        return gl.amd.cdna3.buffer_load(base, offset, cache=".cg")
    return gl.load(base + offset, cache_modifier=".cg")


@gluon.jit
def _store_state(base, offset, value, STREAM: gl.constexpr, BUFFER_STATE: gl.constexpr):
    if BUFFER_STATE:
        gl.amd.cdna3.buffer_store(value, base, offset, cache=".cs")
    else:
        gl.store(base + offset, value, cache_modifier=".cs" if STREAM else "")


@gluon.jit
def _normalize_qk(shared, col, scale):
    q = shared.gather(col, 0)
    k = shared.gather(128 + col, 0)
    q *= gl.rsqrt(gl.sum(q * q, 0) + 1.0e-6) * scale
    k *= gl.rsqrt(gl.sum(k * k, 0) + 1.0e-6)
    return q, k


@gluon.jit
def _state_dot(state, vector, MODE: gl.constexpr):
    # Fold only register-owned K strips before the cross-lane reduction.
    # Prediction and readout use independent static schedules.
    if MODE == "plain":
        return gl.sum(state * vector[None, None, :], 2)
    elif MODE == "tree2" or MODE == "tree4":
        parts: gl.constexpr = 2 if MODE == "tree2" else 4
        rows: gl.constexpr = state.type.shape[1]
        width: gl.constexpr = 128 // parts
        grouped = state.reshape((2, rows, parts, width))
        vector_layout: gl.constexpr = gl.SliceLayout(
            0, gl.SliceLayout(1, grouped.type.layout)
        )
        grouped_vector = gl.convert_layout(
            vector.reshape((parts, width)), vector_layout
        )
        partial = gl.sum(grouped * grouped_vector[None, None, :, :], 2)
        reduced = gl.sum(partial, 2)
        return gl.convert_layout(reduced, gl.SliceLayout(2, state.type.layout))
    else:
        gl.static_assert(MODE == "chain4")
        rows: gl.constexpr = state.type.shape[1]
        s0, s1 = gl.split(state.reshape((2, rows, 2, 64)).permute(0, 1, 3, 2))
        v0, v1 = gl.split(vector.reshape((2, 64)).permute(1, 0))
        s00, s01 = gl.split(s0.reshape((2, rows, 2, 32)).permute(0, 1, 3, 2))
        s10, s11 = gl.split(s1.reshape((2, rows, 2, 32)).permute(0, 1, 3, 2))
        v00, v01 = gl.split(v0.reshape((2, 32)).permute(1, 0))
        v10, v11 = gl.split(v1.reshape((2, 32)).permute(1, 0))
        vector_layout: gl.constexpr = gl.SliceLayout(
            0, gl.SliceLayout(1, s00.type.layout)
        )
        v00 = gl.convert_layout(v00, vector_layout)
        v01 = gl.convert_layout(v01, vector_layout)
        v10 = gl.convert_layout(v10, vector_layout)
        v11 = gl.convert_layout(v11, vector_layout)
        product = gl.fma(s01, v01[None, None, :], s00 * v00[None, None, :])
        product = gl.fma(s10, v10[None, None, :], product)
        product = gl.fma(s11, v11[None, None, :], product)
        reduced = gl.sum(product, 2)
        return gl.convert_layout(reduced, gl.SliceLayout(2, state.type.layout))


@gluon.jit
def _quadrants(value):
    lower, upper = gl.split(value.reshape((2, 2, 64, 128)).permute(0, 2, 3, 1))
    q00, q01 = gl.split(lower.reshape((2, 64, 2, 64)).permute(0, 1, 3, 2))
    q10, q11 = gl.split(upper.reshape((2, 64, 2, 64)).permute(0, 1, 3, 2))
    return q00, q01, q10, q11


@gluon.jit(repr=_small_repr)
def _fused_decode(
    X,
    BA,
    History,
    State,
    Indices,
    Weight,
    Bias,
    ALog,
    DTBias,
    NormWeight,
    Output,
    Quantized,
    Scales,
    scale,
    eps,
    quant_max,
    KH: gl.constexpr,
    VH: gl.constexpr,
    BATCH: gl.constexpr,
    INDEX64: gl.constexpr,
    PAD_SLOT_ID: gl.constexpr,
    HAS_FP8: gl.constexpr,
):
    NW: gl.constexpr = 8
    K_LANES: gl.constexpr = 4 if BATCH <= 8 else 8
    PREP_LANES: gl.constexpr = 16
    V_PACK: gl.constexpr = 2
    STREAM: gl.constexpr = BATCH >= 16
    EARLY_DYNAMICS: gl.constexpr = BATCH == 16
    EPILOGUE_PREFETCH: gl.constexpr = 1 if BATCH <= 16 else 2
    EARLY_SIGMOID: gl.constexpr = BATCH <= 8
    STATE_PREFIX: gl.constexpr = 32 if BATCH <= 8 else 64 if BATCH == 64 else 0
    PRED_DOT: gl.constexpr = "tree2" if BATCH <= 8 else "plain"
    CORE_DOT: gl.constexpr = (
        "tree2" if BATCH <= 8 else "tree4" if BATCH == 16 else "plain"
    )
    FAST_QUANT: gl.constexpr = BATCH <= 32
    DPP_RMS: gl.constexpr = BATCH <= 8
    PHASED_STORE: gl.constexpr = BATCH == 16 or BATCH == 64

    if BATCH == 64:
        token = gl.program_id(1)
        k_head = gl.program_id(0)
    else:
        token = gl.program_id(0)
        k_head = gl.program_id(1)
    slot = gl.load(Indices + token)
    if slot == PAD_SLOT_ID:
        # Padded CUDA-graph row: read no state and write none. Outputs for this
        # row are left undefined, matching aiter's other indexed state kernels.
        return
    if INDEX64:
        slot = slot.to(gl.int64)

    channels: gl.constexpr = (2 * KH + VH) * 128
    gl.static_assert(NW <= 8)
    gl.static_assert(
        V_PACK * (64 // K_LANES) * (NW // 2) <= 128,
        "Recurrent-state ownership must not replicate across waves",
    )
    layout: gl.constexpr = gl.BlockedLayout(
        [1, V_PACK, 4],
        [1, 64 // K_LANES, K_LANES],
        [2, NW // 2, 1],
        [2, 1, 0],
    )
    head = gl.arange(0, 2, gl.SliceLayout(1, gl.SliceLayout(2, layout)))
    row = gl.arange(0, 128, gl.SliceLayout(0, gl.SliceLayout(2, layout)))
    col = gl.arange(0, 128, gl.SliceLayout(0, gl.SliceLayout(1, layout)))
    state_base = State + (slot * VH + k_head * 2) * 16384
    state_offset = (
        head[:, None, None] * 16384 + row[None, :, None] * 128 + col[None, None, :]
    )
    if EPILOGUE_PREFETCH == 1:
        cached_z, cached_weight, cached_sigmoid = _load_epilogue_inputs(
            X,
            NormWeight,
            token,
            k_head,
            KH,
            EARLY_SIGMOID,
        )
    if STATE_PREFIX:
        # Only a register-owned column prefix stays live across convolution.
        # The complementary load below reads every remaining state element.
        state_prefix = gl.load(
            state_base + state_offset,
            col[None, None, :] < STATE_PREFIX,
            other=0,
            cache_modifier=".cg" if STREAM else "",
        )
    else:
        state = _load_state(state_base, state_offset, False)
    if EARLY_DYNAMICS:
        decay, beta = _head_dynamics(
            BA,
            ALog,
            DTBias,
            token,
            k_head * 2 + head,
            VH,
            STREAM,
        )

    # This layout gives each mutable history channel exactly one physical owner.
    prep_layout: gl.constexpr = gl.BlockedLayout(
        [1, 1],
        [64 // PREP_LANES, PREP_LANES],
        [PREP_LANES // 16, NW * 16 // PREP_LANES],
        [1, 0],
    )
    part = gl.arange(0, 4, gl.SliceLayout(1, prep_layout))
    prep_col = gl.arange(0, 128, gl.SliceLayout(0, prep_layout))
    channel_base = gl.where(
        part < 2,
        (part * KH + k_head) * 128,
        2 * KH * 128 + (k_head * 2 + part - 2) * 128,
    )
    channel = channel_base[:, None] + prep_col[None, :]
    values = _convolve(
        X,
        History,
        Weight,
        Bias,
        (token * KH + k_head) * 768 + part[:, None] * 128 + prep_col[None, :],
        (slot * channels + channel) * 3,
        channel,
    )
    shared = gl.allocate_shared_memory(
        gl.float32,
        (512,),
        gl.SwizzledSharedLayout(1, 1, 1, [0]),
        gl.reshape(values, (512,)),
    )
    q, k = _normalize_qk(shared, col, scale)
    v_indices = gl.reshape(256 + head[:, None] * 128 + row[None, :], (256,))
    v = gl.convert_layout(
        gl.reshape(shared.gather(v_indices, 0), (2, 128)),
        gl.SliceLayout(2, layout),
    )
    if not EARLY_DYNAMICS:
        decay, beta = _head_dynamics(
            BA,
            ALog,
            DTBias,
            token,
            k_head * 2 + head,
            VH,
            STREAM,
        )
    if STATE_PREFIX:
        state_tail = gl.load(
            state_base + state_offset,
            col[None, None, :] >= STATE_PREFIX,
            other=0,
            cache_modifier=".cg" if STREAM else "",
        )
        state = gl.where(col[None, None, :] < STATE_PREFIX, state_prefix, state_tail)
    if EPILOGUE_PREFETCH == 2:
        cached_z, cached_weight, cached_sigmoid = _load_epilogue_inputs(
            X,
            NormWeight,
            token,
            k_head,
            KH,
            EARLY_SIGMOID,
        )

    decayed_state = state * decay[:, None, None]
    prediction = _state_dot(decayed_state, k, PRED_DOT)
    residual = (v - prediction) * beta[:, None]
    next_state = gl.fma(residual[:, :, None], k[None, None, :], decayed_state)
    core = _state_dot(next_state, q, CORE_DOT).to(gl.bfloat16)
    if PHASED_STORE:
        s00, s01, s10, s11 = _quadrants(next_state)
        o00, o01, o10, o11 = _quadrants(state_offset)
        _store_state(state_base, o00, s00, STREAM, False)
        _store_state(state_base, o01, s01, STREAM, False)
        _store_state(state_base, o10, s10, STREAM, False)
        epilogue_layout: gl.constexpr = gl.BlockedLayout(
            [1, 2],
            [1, 64],
            [NW, 1],
            [1, 0],
        )
        core = gl.convert_layout(core, epilogue_layout)
        _store_state(state_base, o11, s11, STREAM, False)
    else:
        _store_state(state_base, state_offset, next_state, STREAM, False)
    _finish_pair(
        core,
        Output,
        Quantized,
        Scales,
        token,
        k_head,
        eps,
        quant_max,
        VH,
        cached_z,
        cached_weight,
        cached_sigmoid,
        EARLY_SIGMOID,
        True,
        FAST_QUANT,
        DPP_RMS,
        HAS_FP8,
    )
    # Separate convolution storage from the epilogue's conversion scratch to
    # avoid a shared-memory reuse barrier.
    shared._keep_alive()


@gluon.jit
def _recurrent_tile(
    state,
    state_base,
    state_offset,
    q,
    k,
    v,
    decay,
    beta,
    BUFFER_STATE: gl.constexpr,
    DOT_MODE: gl.constexpr,
):
    decayed_state = state * decay[:, None, None]
    prediction = _state_dot(decayed_state, k, DOT_MODE)
    residual = (v - prediction) * beta[:, None]
    next_state = gl.fma(residual[:, :, None], k[None, None, :], decayed_state)
    core = _state_dot(next_state, q, DOT_MODE).to(gl.bfloat16)
    _store_state(state_base, state_offset, next_state, True, BUFFER_STATE)
    return core


@gluon.jit(repr=_tiled_repr)
def _fused_decode_tiled(
    X,
    BA,
    History,
    State,
    Indices,
    Weight,
    Bias,
    ALog,
    DTBias,
    NormWeight,
    Output,
    Quantized,
    Scales,
    scale,
    eps,
    quant_max,
    KH: gl.constexpr,
    VH: gl.constexpr,
    INDEX64: gl.constexpr,
    B32: gl.constexpr,
    PAD_SLOT_ID: gl.constexpr,
    HAS_FP8: gl.constexpr,
):
    NW: gl.constexpr = 8
    K_LANES: gl.constexpr = 8
    PREP_LANES: gl.constexpr = 64
    V_PACK: gl.constexpr = 1 if B32 else 2
    DOT_MODE: gl.constexpr = "chain4" if B32 else "plain"

    if B32:
        program = gl.program_id(0)
        token = program // (KH * 8) * 8 + program % 8
        k_head = program // 8 % KH
    else:
        token = gl.program_id(0)
        k_head = gl.program_id(1)
    slot = gl.load(Indices + token)
    if slot == PAD_SLOT_ID:
        # Padded CUDA-graph row: read no state and write none. Outputs for this
        # row are left undefined, matching aiter's other indexed state kernels.
        return
    if INDEX64:
        slot = slot.to(gl.int64)

    channels: gl.constexpr = (2 * KH + VH) * 128
    gl.static_assert(NW <= 8)
    gl.static_assert(
        V_PACK * (64 // K_LANES) * (NW // 2) <= 64,
        "Recurrent-state ownership must not replicate across waves",
    )
    layout: gl.constexpr = gl.BlockedLayout(
        [1, V_PACK, 4],
        [1, 64 // K_LANES, K_LANES],
        [2, NW // 2, 1],
        [2, 1, 0],
    )
    head = gl.arange(0, 2, gl.SliceLayout(1, gl.SliceLayout(2, layout)))
    row = gl.arange(0, 64, gl.SliceLayout(0, gl.SliceLayout(2, layout)))
    col = gl.arange(0, 128, gl.SliceLayout(0, gl.SliceLayout(1, layout)))
    state_base = State + (slot * VH + k_head * 2) * 16384
    state_offset = (
        head[:, None, None] * 16384 + row[None, :, None] * 128 + col[None, None, :]
    )
    state = _load_state(state_base, state_offset, B32)
    if B32:
        state1 = _load_state(state_base, state_offset + 64 * 128, B32)

    # This layout gives each mutable history channel exactly one physical owner.
    prep_layout: gl.constexpr = gl.BlockedLayout(
        [1, 1],
        [64 // PREP_LANES, PREP_LANES],
        [PREP_LANES // 16, NW * 16 // PREP_LANES],
        [1, 0],
    )
    part = gl.arange(0, 4, gl.SliceLayout(1, prep_layout))
    prep_col = gl.arange(0, 128, gl.SliceLayout(0, prep_layout))
    channel_base = gl.where(
        part < 2,
        (part * KH + k_head) * 128,
        2 * KH * 128 + (k_head * 2 + part - 2) * 128,
    )
    channel = channel_base[:, None] + prep_col[None, :]
    values = _convolve(
        X,
        History,
        Weight,
        Bias,
        (token * KH + k_head) * 768 + part[:, None] * 128 + prep_col[None, :],
        (slot * channels + channel) * 3,
        channel,
    )
    shared = gl.allocate_shared_memory(
        gl.float32,
        (512,),
        gl.SwizzledSharedLayout(1, 1, 1, [0]),
        gl.reshape(values, (512,)),
    )
    q, k = _normalize_qk(shared, col, scale)
    v_indices = gl.reshape(256 + head[:, None] * 128 + row[None, :], (128,))
    v = gl.convert_layout(
        gl.reshape(shared.gather(v_indices, 0), (2, 64)),
        gl.SliceLayout(2, layout),
    )
    decay, beta = _head_dynamics(
        BA,
        ALog,
        DTBias,
        token,
        k_head * 2 + head,
        VH,
        True,
    )
    cached_z, cached_weight, cached_sigmoid = _load_epilogue_inputs(
        X,
        NormWeight,
        token,
        k_head,
        KH,
        False,
    )
    if not B32:
        state1 = _load_state(state_base, state_offset + 64 * 128, B32)
    v1_indices = gl.reshape(256 + head[:, None] * 128 + 64 + row[None, :], (128,))
    v1 = gl.convert_layout(
        gl.reshape(shared.gather(v1_indices, 0), (2, 64)),
        gl.SliceLayout(2, layout),
    )
    # Completing each independent row tile exposes its stores before the next
    # tile's readout, without intermediate global storage or an extra barrier.
    core0 = _recurrent_tile(
        state, state_base, state_offset, q, k, v, decay, beta, B32, DOT_MODE
    )
    if B32:
        # Retain tile 1 until the readout has crossed into its wave-local
        # epilogue layout, then overlap the wide write with normalization.
        decayed_state1 = state1 * decay[:, None, None]
        prediction1 = _state_dot(decayed_state1, k, DOT_MODE)
        residual1 = (v1 - prediction1) * beta[:, None]
        next_state1 = gl.fma(residual1[:, :, None], k[None, None, :], decayed_state1)
        core1 = _state_dot(next_state1, q, DOT_MODE).to(gl.bfloat16)
        core = gl.join(core0, core1).permute(0, 2, 1).reshape((2, 128))
        epilogue_layout: gl.constexpr = gl.BlockedLayout(
            [1, 2],
            [1, 64],
            [NW, 1],
            [1, 0],
        )
        core = gl.convert_layout(core, epilogue_layout)
        _store_state(state_base, state_offset + 64 * 128, next_state1, True, B32)
    else:
        core1 = _recurrent_tile(
            state1,
            state_base,
            state_offset + 64 * 128,
            q,
            k,
            v1,
            decay,
            beta,
            B32,
            DOT_MODE,
        )
        core = gl.join(core0, core1).permute(0, 2, 1).reshape((2, 128))
    _finish_pair(
        core,
        Output,
        Quantized,
        Scales,
        token,
        k_head,
        eps,
        quant_max,
        VH,
        cached_z,
        cached_weight,
        cached_sigmoid,
        False,
        not B32,
        B32,
        False,
        HAS_FP8,
    )
    # Separate convolution storage from the epilogue's conversion scratch to
    # avoid a shared-memory reuse barrier.
    shared._keep_alive()


@gluon.jit
def _conv_channels(
    Projection,
    State,
    Weight,
    Bias,
    projection_offset,
    channel,
    slot,
    CHANNELS: gl.constexpr,
):
    """Advance width-four convolution; callers assign each channel one owner."""
    x = gl.load(Projection + projection_offset).to(gl.float32)
    history_ptr = State + (slot * CHANNELS + channel) * 3
    h0 = gl.load(history_ptr).to(gl.float32)
    h1 = gl.load(history_ptr + 1).to(gl.float32)
    h2 = gl.load(history_ptr + 2).to(gl.float32)
    w0 = gl.load(Weight + channel * 4).to(gl.float32)
    w1 = gl.load(Weight + channel * 4 + 1).to(gl.float32)
    w2 = gl.load(Weight + channel * 4 + 2).to(gl.float32)
    w3 = gl.load(Weight + channel * 4 + 3).to(gl.float32)
    acc = gl.load(Bias + channel).to(gl.float32)
    acc = gl.fma(h0, w0, acc)
    acc = gl.fma(h1, w1, acc)
    acc = gl.fma(h2, w2, acc)
    acc = gl.fma(x, w3, acc)
    activated = acc / (1.0 + gl.exp(-acc))
    gl.store(history_ptr, h1)
    gl.store(history_ptr + 1, h2)
    gl.store(history_ptr + 2, x)
    return activated.to(gl.bfloat16).to(gl.float32)


@gluon.jit(repr=_group_repr)
def _decode_group(
    Projection,
    BA,
    ConvState,
    State,
    Indices,
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
    KH: gl.constexpr,
    VH: gl.constexpr,
    K: gl.constexpr,
    V: gl.constexpr,
    NW: gl.constexpr,
    KL: gl.constexpr,
    KS: gl.constexpr,
    PAD_SLOT_ID: gl.constexpr,
    HAS_FP8: gl.constexpr,
):
    """One program owns all caches and outputs of a complete Q/K group."""
    token_group = gl.program_id(0)
    token = token_group // KH
    group = token_group % KH
    ratio: gl.constexpr = VH // KH
    channels: gl.constexpr = 2 * KH * K + VH * V
    group_width: gl.constexpr = 2 * K + 2 * ratio * V
    conv_width: gl.constexpr = 2 * K + ratio * V
    gl.static_assert(NW * 64 <= conv_width)
    base = token_group * group_width
    slot = gl.load(Indices + token).to(gl.int64)
    if slot == PAD_SLOT_ID:
        # Padded CUDA-graph row: read no state and write none.
        return
    layout: gl.constexpr = gl.BlockedLayout([1, KS], [64 // KL, KL], [NW, 1], [1, 0])
    k = gl.arange(0, K, layout=gl.SliceLayout(0, layout))
    v = gl.arange(0, ratio * V, layout=gl.SliceLayout(1, layout))
    conv_layout: gl.constexpr = gl.BlockedLayout([1], [64], [NW], [0])
    # This packed tile has no replicated lanes/waves. Every history read and
    # write has one owner, even though Q/K are subsequently shared by two heads.
    c = gl.arange(0, conv_width, layout=conv_layout)
    channel = gl.where(
        c < K,
        group * K + c,
        gl.where(
            c < 2 * K,
            KH * K + group * K + c - K,
            2 * KH * K + group * ratio * V + c - 2 * K,
        ),
    )
    activated = _conv_channels(
        Projection,
        ConvState,
        ConvWeight,
        ConvBias,
        base + c,
        channel,
        slot,
        channels,
    )
    shared_layout: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [0])
    conv_shared = gl.allocate_shared_memory(
        gl.float32, [conv_width], shared_layout, activated
    )
    q = conv_shared.slice(0, K).load(gl.SliceLayout(0, layout))
    key = conv_shared.slice(K, K).load(gl.SliceLayout(0, layout))
    value = conv_shared.slice(2 * K, ratio * V).load(gl.SliceLayout(1, layout))
    q = q * (gl.rsqrt(gl.sum(q * q, 0) + 1.0e-6) * scale)
    key = key * gl.rsqrt(gl.sum(key * key, 0) + 1.0e-6)
    head = group * ratio + v // V
    ba_offset = token * 2 * VH + group * 2 * ratio + v // V
    a = gl.load(BA + ba_offset + ratio).to(gl.float32)
    b = gl.load(BA + ba_offset).to(gl.float32)
    arg = a + gl.load(DtBias + head).to(gl.float32)
    softplus = gl.where(arg <= 20.0, gl.log(1.0 + gl.exp(arg)), arg)
    decay = gl.exp(-gl.exp(gl.load(ALog + head)) * softplus)
    beta = (1.0 / (1.0 + gl.exp(-b))).to(gl.bfloat16).to(gl.float32)
    state_base = State + (slot * VH + group * ratio) * V * K
    state_offset = v[:, None] * K + k[None, :]
    # The full state tile stays in registers through both matrix-vector products.
    state = gl.amd.cdna3.buffer_load(state_base, state_offset, cache=".cg")
    decayed = state * decay[:, None]
    predicted = gl.sum(decayed * key[None, :], 1)
    residual = (value - predicted) * beta
    updated = gl.fma(state, decay[:, None], residual[:, None] * key[None, :])
    out = gl.sum(updated * q[None, :], 1)
    gl.amd.cdna3.buffer_store(updated, state_base, state_offset, cache=".wt")

    # Preserve the BF16 core boundary before separate per-head RMS reductions.
    out_layout: gl.constexpr = gl.BlockedLayout(
        [1, 1], [1, 64], [ratio, NW // ratio], [1, 0]
    )
    core = gl.convert_layout(
        gl.reshape(out.to(gl.bfloat16).to(gl.float32), (ratio, V)), out_layout
    )
    h = gl.arange(0, ratio, layout=gl.SliceLayout(1, out_layout))
    o = gl.arange(0, V, layout=gl.SliceLayout(0, out_layout))
    gate = gl.load(
        Projection + base + 2 * K + ratio * V + h[:, None] * V + o[None, :]
    ).to(gl.float32)
    weight = gl.load(NormWeight + o).to(gl.float32)
    rms = gl.rsqrt(gl.sum(core * core, 1) / V + eps)
    sigmoid = 1.0 / (1.0 + gl.exp(-gate))
    normalized = (core * rms[:, None] * weight[None, :] * gate * sigmoid).to(
        gl.bfloat16
    )
    out_offset = token_group * ratio * V + h[:, None] * V + o[None, :]
    gl.store(Output + out_offset, normalized)
    if HAS_FP8:
        # FP8 scales must consume the rounded BF16 output, not the FP32 precursor.
        values = normalized.to(gl.float32)
        quant_scale = gl.maximum(gl.max(gl.abs(values), 1), 1.0e-10) / fp8_max
        gl.store(Scales + token_group * ratio + h, quant_scale)
        gl.store(
            Quantized + out_offset,
            gl.clamp(values / quant_scale[:, None], -fp8_max, fp8_max),
        )
