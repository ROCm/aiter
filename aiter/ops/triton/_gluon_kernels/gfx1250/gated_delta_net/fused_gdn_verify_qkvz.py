# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Gluon kernel of fused_gdn_verify_qkvz for gfx1250: the Gated DeltaNet
speculative-verify step on ReplaySSM state. The wrapper docstring describes the
inputs and the ReplaySSM protocol."""

import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from aiter.ops.triton._gluon_kernels.common.utils import exp_scaled, l2norm, softplus
from aiter.ops.triton._triton_kernels.common.activation import _sigmoid, _silu
from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr

_verify_repr = make_kernel_repr(
    "_fused_gdn_verify", ["RATIO", "T_MAX", "CAP", "NW", "KL", "KS"]
)


@gluon.constexpr_function
def _row_layout(rows, num_warps):
    """A ``[rows, 128]`` layout with each row in one wave."""
    waves = min(rows, num_warps)
    return gl.BlockedLayout([1, 4], [1, 32], [waves, num_warps // waves], [1, 0])


@gluon.jit
def _gating_inputs(BA, ALog, DtBias, ba_row, head, VH: gl.constexpr):
    """The gating inputs of one value head of one token. ``BA`` rows are ``b|a``."""
    b = gl.load(BA + ba_row + head).to(gl.float32)
    a = gl.load(BA + ba_row + VH + head).to(gl.float32)
    a_log = gl.load(ALog + head).to(gl.float32)
    dt_bias = gl.load(DtBias + head).to(gl.float32)
    return b, a, a_log, dt_bias


@gluon.jit(repr=_verify_repr)
def _fused_gdn_verify(
    # Only the first 30 argument dwords arrive in SGPRs, so the pointers of the
    # first two round trips come first.
    QueryStartLoc,
    SlotIdx,
    ConvIndices,
    NumAccepted,
    WritePos,
    Checkpoint,
    ConvState,
    Projection,
    ConvWeight,
    BA,
    ALog,
    DtBias,
    RecordG,
    RecordK,
    RecordU,
    NormWeight,
    Output,
    ConvBias,
    scale,
    eps,
    proj_token_stride: gl.constexpr,
    ba_token_stride: gl.constexpr,
    conv_index_stride: gl.constexpr,
    conv_slot_stride: gl.constexpr,
    conv_channel_stride: gl.constexpr,
    conv_width_stride: gl.constexpr,
    ckpt_slot_stride: gl.constexpr,
    rk_slot_stride: gl.constexpr,
    rk_head_stride: gl.constexpr,
    rk_pos_stride: gl.constexpr,
    ru_slot_stride: gl.constexpr,
    ru_head_stride: gl.constexpr,
    ru_pos_stride: gl.constexpr,
    rg_slot_stride: gl.constexpr,
    rg_head_stride: gl.constexpr,
    rg_pos_stride: gl.constexpr,
    KH: gl.constexpr,
    VH: gl.constexpr,
    K: gl.constexpr,
    V: gl.constexpr,
    RATIO: gl.constexpr,
    T_MAX: gl.constexpr,
    CAP: gl.constexpr,
    NW: gl.constexpr,
    KL: gl.constexpr,
    KS: gl.constexpr,
    PAD_SLOT_ID: gl.constexpr,
    HAS_BIAS: gl.constexpr,
):
    """One program per (sequence, key-head group). ``Projection`` holds the
    ``q|k|v|z`` columns and ``BA`` the ``b|a`` columns of each token. The kernel
    stores nothing for a sequence without tokens or with a ``PAD_SLOT_ID`` slot. The
    strides are element strides. They are fixed for a deployment, so they compile in."""
    gl.static_assert(VH == RATIO * KH, "VH must be RATIO * KH")
    gl.static_assert(K == V, "the record tiles share one column index")
    gl.static_assert(K % (KL * KS) == 0, "KL lanes x KS elements must tile K")
    gl.static_assert(CAP >= 2 * T_MAX, "the records must hold two windows")
    seq_group = gl.program_id(0)
    seq = seq_group // KH
    group = seq_group % KH
    conv_width: gl.constexpr = 2 * K + RATIO * V
    CONV_P2: gl.constexpr = triton.next_power_of_2(conv_width)
    T_P2: gl.constexpr = triton.next_power_of_2(T_MAX)
    CAP_P2: gl.constexpr = triton.next_power_of_2(CAP)
    RATIO_P2: gl.constexpr = triton.next_power_of_2(RATIO)

    # The indices of the sequence, in one round trip. A padded sequence runs on slot
    # 0 and row 0 and stores nothing, so no load waits behind an early exit.
    bos = gl.load(QueryStartLoc + seq)
    tokens = gl.load(QueryStartLoc + seq + 1) - bos
    slot = gl.load(SlotIdx + seq)
    conv_slot = gl.load(ConvIndices + seq * conv_index_stride)
    accepted = gl.load(NumAccepted + seq)
    valid = (tokens > 0) & (slot != PAD_SLOT_ID) & (conv_slot != PAD_SLOT_ID)
    tokens = gl.where(valid, tokens, 0)
    bos = gl.where(valid, bos, 0)
    slot = gl.where(valid, slot, 0).to(gl.int64)
    conv_slot = gl.where(valid, conv_slot, 0).to(gl.int64)
    accepted = gl.where(valid, accepted, 1)
    # A token t >= tokens reads a valid row. Its records and outputs are not stored,
    # and no stored value depends on it.
    last = gl.maximum(tokens, 1) - 1
    # The flush rule must match the commit, which derives it again from the cursor.
    committed = gl.maximum(gl.load(WritePos + slot), 0)
    flush = committed + 2 * T_MAX > CAP
    first = gl.where(flush, 0, committed)

    # The [V, K] state tile keeps K in a wave, so the reductions of the token loop
    # stay in the wave. The K-major checkpoint moves with 16 B of V on each lane.
    tile_layout: gl.constexpr = gl.BlockedLayout(
        [1, KS], [32 // KL, KL], [NW, 1], [1, 0]
    )
    krow = gl.arange(0, K, layout=gl.SliceLayout(1, tile_layout))
    vcol = gl.arange(0, V, layout=gl.SliceLayout(0, tile_layout))
    ckpt_offset = krow[:, None] * V + vcol[None, :]
    # The records of a head move as [CAP, K] tiles. The rebuild reads them back from
    # LDS one row at a time.
    rec_layout: gl.constexpr = gl.BlockedLayout([1, 4], [4, 8], [NW // 4, 4], [1, 0])
    rec_shared: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [0])
    rec_shared_2d: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [1, 0])
    jrec = gl.arange(0, CAP_P2, layout=gl.SliceLayout(1, rec_layout))
    crec = gl.arange(0, K, layout=gl.SliceLayout(0, rec_layout))
    # Each thread sums the gate pairs of a head, so the record weights need no other
    # thread. Above 64 gates for the group, a scan takes its place to save registers.
    PAIRS: gl.constexpr = CAP_P2 * RATIO <= 64
    pair_layout: gl.constexpr = gl.BlockedLayout(
        [1, CAP_P2], [4, 8], [NW // 4, 4], [1, 0]
    )
    later_idx = gl.arange(0, CAP_P2, layout=gl.SliceLayout(0, pair_layout))
    rec_idx = gl.arange(0, CAP_P2, layout=gl.SliceLayout(1, pair_layout))
    # The scan keeps all gates of a head in each thread.
    scan_layout: gl.constexpr = gl.BlockedLayout([CAP_P2], [32], [NW], [0])
    jall = gl.arange(0, CAP_P2, layout=scan_layout)

    # Phase 1 loads: the conv operands, one lane per conv channel of the group. The
    # previous step accepted NumAccepted tokens, so the three inputs before the
    # window start at NumAccepted - 1. The channel and row are clamped, so the loads
    # need no mask.
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
    live = c < conv_width
    w0 = gl.load(ConvWeight + channel * 4).to(gl.float32)
    w1 = gl.load(ConvWeight + channel * 4 + 1).to(gl.float32)
    w2 = gl.load(ConvWeight + channel * 4 + 2).to(gl.float32)
    w3 = gl.load(ConvWeight + channel * 4 + 3).to(gl.float32)
    if HAS_BIAS:
        conv_bias = gl.load(ConvBias + channel).to(gl.float32)
    else:
        conv_bias = gl.zeros_like(w0)
    window = ConvState + conv_slot * conv_slot_stride + channel * conv_channel_stride
    history = window + (accepted - 1) * conv_width_stride
    h0 = gl.load(history)
    h1 = gl.load(history + conv_width_stride)
    h2 = gl.load(history + 2 * conv_width_stride)
    inputs = (h0, h1, h2)
    for t in gl.static_range(T_MAX):
        row = bos + gl.minimum(t, last)
        inputs = inputs + (gl.load(Projection + row * proj_token_stride + channel),)

    # The outputs leave as [WIN, V] tiles with one row per (value head, token), so
    # the stores are vectors and the RMSNorm of a row stays in its wave.
    WIN: gl.constexpr = RATIO_P2 * T_P2
    win_layout: gl.constexpr = _row_layout(WIN, NW)
    wrow = gl.arange(0, WIN, layout=gl.SliceLayout(1, win_layout))
    wcol = gl.arange(0, V, layout=gl.SliceLayout(0, win_layout))
    whead = group * RATIO + wrow // T_P2
    wtoken = wrow % T_P2
    wlive = (wtoken < tokens)[:, None] & (wcol < V)[None, :]
    if RATIO_P2 != RATIO:
        wlive = wlive & (wrow // T_P2 < RATIO)[:, None]

    # Phase 2 loads: all that the value heads read from global memory. They are
    # issued before any of it is used, so that the round trips overlap.
    weight = gl.load(NormWeight + wcol).to(gl.float32)
    z_gate = gl.load(
        Projection
        + (bos + wtoken)[:, None] * proj_token_stride
        + 2 * KH * K
        + VH * V
        + whead[:, None] * V
        + wcol[None, :],
        mask=wlive,
        other=0.0,
    )
    rec_live = (jrec < committed)[:, None] & (crec < K)[None, :]
    heads = ()
    for i in gl.static_range(RATIO):
        head = group * RATIO + i
        record_g = RecordG + slot * rg_slot_stride + head * rg_head_stride
        if not PAIRS:
            g_rec = gl.load(record_g + jrec * rg_pos_stride, mask=jrec < CAP, other=0.0)
        elif CAP_P2 != CAP:
            g_rec = gl.load(
                record_g + later_idx * rg_pos_stride, mask=later_idx < CAP, other=0.0
            )
        else:
            g_rec = gl.load(record_g + later_idx * rg_pos_stride)
        k_rec = gl.load(
            RecordK
            + slot * rk_slot_stride
            + head * rk_head_stride
            + jrec[:, None] * rk_pos_stride
            + crec[None, :],
            mask=rec_live,
            other=0.0,
        )
        u_rec = gl.load(
            RecordU
            + slot * ru_slot_stride
            + head * ru_head_stride
            + jrec[:, None] * ru_pos_stride
            + crec[None, :],
            mask=rec_live,
            other=0.0,
        )
        ckpt_raw = gl.amd.gfx1250.buffer_load(
            Checkpoint + slot * ckpt_slot_stride + head * K * V,
            ckpt_offset,
            cache=".cg",
        )
        gates = ()
        decays = ()
        betas = ()
        for t in gl.static_range(T_MAX):
            row = bos + gl.minimum(t, last)
            b, a, a_log, dt_bias = _gating_inputs(
                BA, ALog, DtBias, row * ba_token_stride, head, VH
            )
            g = -exp_scaled(1.0, a_log) * softplus(a + dt_bias)
            gates = gates + (g,)
            decays = decays + (exp_scaled(1.0, g),)
            betas = betas + (_sigmoid(b).to(gl.bfloat16).to(gl.float32),)
        heads = heads + ((g_rec, k_rec, u_rec, ckpt_raw, gates, decays, betas),)

    # Phase 1: the conv of the window, one row of conv_out per token.
    conv_out = gl.allocate_shared_memory(gl.float32, [T_P2, CONV_P2], rec_shared)
    for t in gl.static_range(T_MAX):
        acc = gl.fma(inputs[t].to(gl.float32), w0, conv_bias)
        acc = gl.fma(inputs[t + 1].to(gl.float32), w1, acc)
        acc = gl.fma(inputs[t + 2].to(gl.float32), w2, acc)
        acc = gl.fma(inputs[t + 3].to(gl.float32), w3, acc)
        # q, k and v round to bf16, as the unfused chain stores them.
        conv_out.index(t).store(_silu(acc).to(gl.bfloat16).to(gl.float32))
    # The new window is [h1, h2, x_0 .. x_{tokens-1}]. Each lane writes only the
    # channel that it read, and one program owns all the channels of its group.
    gl.store(window, h1, mask=live & valid)
    gl.store(window + conv_width_stride, h2, mask=live & valid)
    for t in gl.static_range(T_MAX):
        gl.store(
            window + (2 + t) * conv_width_stride,
            inputs[3 + t].to(h2.dtype),
            mask=live & (t < tokens),
        )

    # q and k of each token, L2-normalized once for all value heads.
    norm_layout: gl.constexpr = _row_layout(T_P2, NW)
    conv_rows = conv_out.reinterpret(gl.float32, [T_P2, CONV_P2], rec_shared_2d)
    q_rows = conv_rows.slice(0, K, dim=1).load(norm_layout)
    k_rows = conv_rows.slice(K, K, dim=1).load(norm_layout)
    q_rows = q_rows * (gl.rsqrt(gl.sum(q_rows * q_rows, 1) + 1.0e-6) * scale)[:, None]
    k_rows = l2norm(k_rows)
    q_norm = gl.allocate_shared_memory(gl.float32, [T_P2, K], rec_shared)
    q_norm.reinterpret(gl.float32, [T_P2, K], rec_shared_2d).store(q_rows)
    k_norm = gl.allocate_shared_memory(gl.float32, [T_P2, K], rec_shared)
    k_norm.reinterpret(gl.float32, [T_P2, K], rec_shared_2d).store(k_rows)

    # The window rows: the residual of each token (its u record) and the readout.
    # The token loops write them, and phase 4 stores them.
    u_win = gl.allocate_shared_memory(gl.float32, [WIN, V], rec_shared)
    o_win = gl.allocate_shared_memory(gl.float32, [WIN, V], rec_shared)

    # Phase 3: the value heads of the group. All heads stage first, so the LDS round
    # trips of a head do not wait behind the token loop of the head before it.
    staged = ()
    for i in gl.static_range(RATIO):
        g_rec, k_rec, u_rec, ckpt_raw, gates, decays, betas = heads[i]
        # S = exp(sum g) S_0 + sum_j exp(sum of the gates after j) k_j u_j^T.
        if PAIRS:
            g_live = gl.where(later_idx < committed, g_rec, 0.0)
            after = gl.sum(
                gl.where(later_idx[None, :] > rec_idx[:, None], g_live[None, :], 0.0), 1
            )
            w_rec = gl.convert_layout(
                gl.where(rec_idx < committed, exp_scaled(1.0, after), 0.0),
                gl.SliceLayout(1, rec_layout),
            )
        else:
            g_live = gl.convert_layout(
                gl.where(jrec < committed, g_rec, 0.0), scan_layout
            )
            after = gl.sum(g_live, 0) - gl.cumsum(g_live, 0)
            w_rec = gl.convert_layout(
                gl.where(jall < committed, exp_scaled(1.0, after), 0.0),
                gl.SliceLayout(1, rec_layout),
            )
        # Rows past the cursor are staged but not read.
        k_shared = gl.allocate_shared_memory(gl.float32, [CAP_P2, K], rec_shared)
        k_shared.reinterpret(gl.float32, [CAP_P2, K], rec_shared_2d).store(
            k_rec.to(gl.float32)
        )
        u_shared = gl.allocate_shared_memory(gl.float32, [CAP_P2, V], rec_shared)
        u_shared.reinterpret(gl.float32, [CAP_P2, V], rec_shared_2d).store(
            u_rec.to(gl.float32) * w_rec[:, None]
        )
        start = gl.convert_layout(gl.permute(ckpt_raw, [1, 0]), tile_layout)
        staged = staged + (
            (k_shared, u_shared, start, exp_scaled(1.0, gl.sum(g_live, 0))),
        )

    for i in gl.static_range(RATIO):
        head = group * RATIO + i
        gates, decays, betas = heads[i][4], heads[i][5], heads[i][6]
        k_shared, u_shared, start, start_decay = staged[i]
        ckpt = Checkpoint + slot * ckpt_slot_stride + head * K * V
        record_g = RecordG + slot * rg_slot_stride + head * rg_head_stride
        state = start_decay * start.to(gl.float32)
        for j in range(committed):
            k_j = k_shared.index(j).load(gl.SliceLayout(0, tile_layout))
            u_j = u_shared.index(j).load(gl.SliceLayout(1, tile_layout))
            state = gl.fma(u_j[:, None], k_j[None, :], state)
        if flush & valid:
            # All waves have read the checkpoint and the records before any wave
            # writes the new checkpoint and writes records from position 0.
            gl.barrier()
            flushed = gl.convert_layout(
                gl.permute(state.to(Checkpoint.dtype.element_ty), [1, 0]), tile_layout
            )
            gl.amd.gfx1250.buffer_store(flushed, ckpt, ckpt_offset, cache=".wt")

        # The tokens of the window, in order. No step of the chain crosses waves or
        # stores to global memory.
        for t in gl.static_range(T_MAX):
            q = q_norm.index(t).load(gl.SliceLayout(0, tile_layout))
            key = k_norm.index(t).load(gl.SliceLayout(0, tile_layout))
            value = (
                conv_out.index(t)
                .slice(2 * K + i * V, V)
                .load(gl.SliceLayout(1, tile_layout))
            )
            decayed = state * decays[t]
            predicted = gl.sum(decayed * key[None, :], 1)
            residual = (value - predicted) * betas[t]
            state = gl.fma(residual[:, None], key[None, :], decayed)
            u_win.index(i * T_P2 + t).store(residual)
            o_win.index(i * T_P2 + t).store(gl.sum(state * q[None, :], 1))
            gl.store(record_g + (first + t) * rg_pos_stride, gates[t], mask=t < tokens)

    # Phase 4: the records of the window and the gated RMSNorm of the readout. The
    # value heads of a group share the keys of the window.
    nrow = gl.arange(0, T_P2, layout=gl.SliceLayout(1, norm_layout))
    ncol = gl.arange(0, K, layout=gl.SliceLayout(0, norm_layout))
    keys = k_rows.to(RecordK.dtype.element_ty)
    key_ptr = (
        RecordK
        + slot * rk_slot_stride
        + (first + nrow)[:, None] * rk_pos_stride
        + ncol[None, :]
    )
    key_live = (nrow < tokens)[:, None] & (ncol < K)[None, :]
    for i in gl.static_range(RATIO):
        gl.store(key_ptr + (group * RATIO + i) * rk_head_stride, keys, mask=key_live)
    position = (first + wtoken)[:, None]
    residuals = u_win.reinterpret(gl.float32, [WIN, V], rec_shared_2d).load(win_layout)
    gl.store(
        RecordU
        + slot * ru_slot_stride
        + whead[:, None] * ru_head_stride
        + position * ru_pos_stride
        + wcol[None, :],
        residuals.to(RecordU.dtype.element_ty),
        mask=wlive,
    )
    # The readout rounds to bf16 before the norm, as the unfused chain stores it.
    core = o_win.reinterpret(gl.float32, [WIN, V], rec_shared_2d).load(win_layout)
    core = core.to(gl.bfloat16).to(gl.float32)
    rms = gl.rsqrt(gl.sum(core * core, 1) / V + eps)
    normalized = core * rms[:, None] * weight[None, :] * _silu(z_gate.to(gl.float32))
    gl.store(
        Output + ((bos + wtoken) * VH + whead)[:, None] * V + wcol[None, :],
        normalized.to(gl.bfloat16),
        mask=wlive,
    )
