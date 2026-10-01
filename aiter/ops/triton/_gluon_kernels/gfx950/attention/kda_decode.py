# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.


import math

from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from aiter.ops.triton._gluon_kernels.common.utils import (
    exp_scaled,
    sigmoid,
    softplus,
)
from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr
from aiter.ops.triton.utils.config_utils import load_config_json, resolve_config_dir


@gluon.jit
def _conv_load(hb, wb, ho, wo, m, s_pos, LP: gl.constexpr):
    """The three history taps and four conv weights of this head's q/k/v rows."""
    h0 = gl.amd.cdna4.buffer_load(hb, ho, mask=m, cache=".cg")
    h1 = gl.amd.cdna4.buffer_load(hb, ho + s_pos, mask=m, cache=".cg")
    h2 = gl.amd.cdna4.buffer_load(hb, ho + 2 * s_pos, mask=m, cache=".cg")
    w0 = gl.amd.cdna4.buffer_load(wb, wo, mask=m).to(gl.float32)
    w1 = gl.amd.cdna4.buffer_load(wb, wo + LP, mask=m).to(gl.float32)
    w2 = gl.amd.cdna4.buffer_load(wb, wo + 2 * LP, mask=m).to(gl.float32)
    w3 = gl.amd.cdna4.buffer_load(wb, wo + 3 * LP, mask=m).to(gl.float32)
    return h0, h1, h2, w0, w1, w2, w3


@gluon.jit
def _token(
    xp,
    gp,
    hb,
    wb,
    ho,
    wo,
    m,
    s_pos,
    a_exp,
    dt,
    gate_c,
    LP: gl.constexpr,
    USE_CONV: gl.constexpr,
    USE_GATE_IN_KERNEL: gl.constexpr,
    USE_LOWER_BOUND: gl.constexpr,
):
    x = gl.load(xp)
    if gp is not None:
        # g has its own dtype: load it apart and splice it into row 2
        x = gl.where(m, x.to(gl.float32), gl.load(gp, mask=~m).to(gl.float32))
    if USE_CONV:
        h0, h1, h2, w0, w1, w2, w3 = _conv_load(hb, wb, ho, wo, m, s_pos, LP)
        acc = h0.to(gl.float32) * w0 + h1.to(gl.float32) * w1
        acc += h2.to(gl.float32) * w2 + x.to(gl.float32) * w3
        gl.amd.cdna4.buffer_store(h1, hb, ho, mask=m, cache=".cs")
        gl.amd.cdna4.buffer_store(h2, hb, ho + s_pos, mask=m, cache=".cs")
        gl.amd.cdna4.buffer_store(
            x.to(hb.dtype.element_ty), hb, ho + 2 * s_pos, mask=m, cache=".cs"
        )
    else:
        acc = x.to(gl.float32)
    return _finish(
        x, acc, m, a_exp, dt, gate_c, USE_CONV, USE_GATE_IN_KERNEL, USE_LOWER_BOUND
    )


@gluon.jit
def _finish(
    x,
    acc,
    m,
    a_exp,
    dt,
    gate_c,
    USE_CONV: gl.constexpr,
    USE_GATE_IN_KERNEL: gl.constexpr,
    USE_LOWER_BOUND: gl.constexpr,
):
    """Gate chain on row 2 and SiLU on the convolved q/k/v rows."""
    x = x.to(gl.float32)
    if USE_GATE_IN_KERNEL and USE_LOWER_BOUND:
        sg = sigmoid(gl.where(m, acc, a_exp * (x + dt)))
        a = gl.exp2(gate_c * sg)
    elif USE_GATE_IN_KERNEL:
        sg = sigmoid(acc)
        a = exp_scaled(-a_exp, softplus(x + dt))
    else:
        sg = sigmoid(acc)
        a = gl.exp(x)
    if USE_CONV:
        acc = acc * sg  # SiLU
    return gl.where(m, acc, a)


_fused_recurrent_kda_packed_decode_repr = make_kernel_repr(
    "fused_recurrent_kda_packed_decode_kernel",
    [
        "BV",
        "SK",
        "NUM_WARPS",
        "PAD_SLOT_GUARD",
        "USE_CONV",
        "USE_RMS_GATE",
        "RESIDENT_STATE",
    ],
)


@gluon.jit(repr=_fused_recurrent_kda_packed_decode_repr)
def fused_recurrent_kda_packed_decode_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    g_ptr,
    beta_ptr,
    A_log_ptr,
    dt_bias_ptr,
    o_ptr,
    state_ptr,
    state_out_ptr,
    cu_seqlens_ptr,
    state_indices_ptr,
    num_accepted_ptr,
    conv_state_ptr,
    conv_weight_ptr,
    out_gate_ptr,
    norm_weight_ptr,
    lower_bound,
    norm_eps,
    T,
    stride_indices_seq,
    stride_q_token: gl.constexpr,
    stride_k_token: gl.constexpr,
    stride_v_token: gl.constexpr,
    stride_g_token: gl.constexpr,
    stride_beta_token: gl.constexpr,
    stride_o_token: gl.constexpr,
    stride_og_token: gl.constexpr,
    stride_state_slot_rows,
    stride_state_out_slot_rows,
    state_rows,
    state_out_rows,
    stride_cs_slot,
    stride_cs_dim,
    stride_cs_pos,
    scale: gl.constexpr,
    H: gl.constexpr,
    HV: gl.constexpr,
    K: gl.constexpr,
    V: gl.constexpr,
    BV: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    SK: gl.constexpr,
    NUM_BUFFERS: gl.constexpr,
    IS_VARLEN: gl.constexpr,
    IS_CONTINUOUS_BATCHING: gl.constexpr,
    IS_SPEC_DECODING: gl.constexpr,
    IS_BETA_HEADWISE: gl.constexpr,
    USE_INITIAL_STATE: gl.constexpr,
    STORE_FINAL_STATE: gl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: gl.constexpr,
    USE_GATE_IN_KERNEL: gl.constexpr,
    HAS_DT_BIAS: gl.constexpr,
    USE_LOWER_BOUND: gl.constexpr,
    APPLY_BETA_SIGMOID: gl.constexpr,
    ALLOW_NEG_EIGVAL: gl.constexpr,
    INPLACE_FINAL_STATE: gl.constexpr,
    STATE_V_FIRST: gl.constexpr,
    USE_TDM_STORE: gl.constexpr = False,
    USE_TDM_LOAD: gl.constexpr = False,
    CACHE_STATE_UPDATES: gl.constexpr = False,
    USE_TDM_FUSED_LOAD: gl.constexpr = False,
    PAD_SLOT_GUARD: gl.constexpr = False,
    W: gl.constexpr = 4,
    USE_CONV: gl.constexpr = False,
    USE_RMS_GATE: gl.constexpr = False,
    TDM_STORE_BUFS: gl.constexpr = 2,
    RESIDENT_STATE: gl.constexpr = False,  # multi-token: state held in registers
):
    gl.static_assert(STATE_V_FIRST, "gfx950 kda decode keeps the state [V, K]")
    gl.static_assert(BV == V and K == V, "gfx950 kda decode needs BV == V == K")
    VEC: gl.constexpr = 4
    gl.static_assert(
        SK == 16 and K % (16 * VEC) == 0, "rows reduce over one 16-lane row"
    )
    gl.static_assert(V % (64 * NUM_WARPS // SK) == 0, "row chunks must tile V")
    gl.static_assert(
        NUM_WARPS % 4 == 0 and (4 * K) % (64 * NUM_WARPS) == 0,
        "the q/k/g/v tile needs a multiple of 4 warps",
    )
    gl.static_assert(STORE_FINAL_STATE, "the state streams through its destination")
    gl.static_assert(
        1 <= NUM_BUFFERS and NUM_BUFFERS * 64 * NUM_WARPS <= V * SK,
        "NUM_BUFFERS: state row chunks in flight",
    )
    gl.static_assert((not USE_CONV) or W == 4, "USE_CONV needs W == 4")
    gl.static_assert(
        not (
            CACHE_STATE_UPDATES or USE_TDM_STORE or USE_TDM_LOAD or USE_TDM_FUSED_LOAD
        ),
        "not supported on gfx950",
    )
    gl.static_assert(
        (not IS_SPEC_DECODING)
        or (RESIDENT_STATE and IS_CONTINUOUS_BATCHING and INPLACE_FINAL_STATE),
        "spec decoding runs the paged, in-place, register-resident path",
    )
    gl.static_assert(
        (not PAD_SLOT_GUARD) or IS_CONTINUOUS_BATCHING,
        "PAD_SLOT_GUARD is a paged-mode contract",
    )

    RC: gl.constexpr = (64 * NUM_WARPS) // SK
    NC: gl.constexpr = V // RC
    CHUNK: gl.constexpr = gl.BlockedLayout(
        [1, VEC], [64 // SK, SK], [NUM_WARPS, 1], [1, 0]
    )
    K_LAYOUT: gl.constexpr = gl.SliceLayout(0, CHUNK)
    R_LAYOUT: gl.constexpr = gl.SliceLayout(1, CHUNK)
    RED: gl.constexpr = gl.BlockedLayout([RC], [64], [NUM_WARPS], [0])
    OUT: gl.constexpr = gl.BlockedLayout([1], [64], [NUM_WARPS], [0])
    FLAT: gl.constexpr = gl.BlockedLayout(
        [1, (4 * K) // (64 * NUM_WARPS)], [1, 64], [4, NUM_WARPS // 4], [1, 0]
    )
    SMEM1: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [0])
    LP: gl.constexpr = H * K

    pid = gl.program_id(0)
    if USE_CONV:
        npid = gl.num_programs(0)
        xcd = pid % 8
        tall = (npid - 1) % 8 + 1
        pid = pid // 8 + xcd * ((npid + 7) // 8) - gl.maximum(xcd - tall, 0)
    i_n = pid // HV
    i_hv = pid % HV
    i_h = i_hv // (HV // H)

    if IS_VARLEN:
        bos = gl.load(cu_seqlens_ptr + i_n).to(gl.int32)
        n_tok = gl.load(cu_seqlens_ptr + i_n + 1).to(gl.int32) - bos
    else:
        bos = i_n * T
        n_tok = T
    # Kept nested: RESIDENT_STATE is constexpr, so only resident variants get the
    # runtime early return (and the control-flow rewrite it brings).
    if RESIDENT_STATE:  # noqa: SIM102
        if n_tok == 0:
            return

    tok0 = bos.to(gl.int64)
    o_p = o_ptr + tok0 * stride_o_token + i_hv * V
    off_o = gl.arange(0, V, layout=OUT)

    rows = gl.arange(0, RC, layout=R_LAYOUT) * NC
    o_idx = (off_o % NC) * RC + off_o // NC
    off_s = rows[:, None] * K + gl.arange(0, K, layout=K_LAYOUT)[None, :]
    r = gl.arange(0, 4, layout=gl.SliceLayout(1, FLAT))[:, None]
    c = gl.arange(0, K, layout=gl.SliceLayout(0, FLAT))[None, :]
    m = r != 2

    if USE_GATE_IN_KERNEL:
        a_exp = gl.exp(gl.load(A_log_ptr + i_hv).to(gl.float32))
        if HAS_DT_BIAS:
            dt = gl.amd.cdna4.buffer_load(
                dt_bias_ptr + i_hv * K, c + r * 0, mask=r == 2
            ).to(gl.float32)
        else:
            dt = 0.0
        gate_c = lower_bound * 1.4426950408889634
    else:
        a_exp = 0.0
        dt = 0.0
        gate_c = 0.0

    q_p = q_ptr + tok0 * stride_q_token + i_h * K
    k_p = k_ptr + tok0 * stride_k_token + i_h * K
    g_p = g_ptr + tok0 * stride_g_token + i_hv * K
    v_p = v_ptr + tok0 * stride_v_token + i_hv * V
    if g_ptr.dtype == q_ptr.dtype:
        xp = gl.where(r == 0, q_p, gl.where(r == 1, k_p, gl.where(m, v_p, g_p))) + c
        gp = None
    else:
        xp = gl.where(r == 0, q_p, gl.where(r == 1, k_p, v_p)) + c
        gp = g_p + c + r * 0
    xs = gl.where(
        r == 0,
        stride_q_token,
        gl.where(r == 1, stride_k_token, gl.where(m, stride_v_token, stride_g_token)),
    )
    grp = gl.where(r == 3, 2, r)
    cc = gl.where(r == 3, i_hv * V, i_h * K) + c
    ho = (grp * LP + cc) * stride_cs_dim
    wo = grp * (W * LP) + cc
    if IS_BETA_HEADWISE:
        b_p = beta_ptr + tok0 * stride_beta_token + i_hv * V
    else:
        b_p = beta_ptr + tok0 * stride_beta_token + i_hv
    if USE_RMS_GATE:
        og_p = out_gate_ptr + tok0 * stride_og_token + i_hv * V
        nw = gl.amd.cdna4.buffer_load(norm_weight_ptr, off_o).to(gl.float32)
    if IS_CONTINUOUS_BATCHING:
        if IS_SPEC_DECODING:
            seed = gl.maximum(gl.load(num_accepted_ptr + i_n).to(gl.int32) - 1, 0)
        else:
            seed = 0
        slot = gl.load(state_indices_ptr + i_n * stride_indices_seq + seed).to(gl.int32)
    else:
        seed = 0
        slot = i_n
    if PAD_SLOT_GUARD:  # noqa: SIM102 (constexpr guard, runtime test)
        if slot <= 0:
            gl.store(
                o_p + off_o,
                gl.full([V], 0.0, o_ptr.dtype.element_ty, OUT),
                mask=off_o < n_tok * V,
            )
            return

    if USE_INITIAL_STATE:
        s_src = state_ptr + (slot * stride_state_slot_rows + i_hv * V).to(gl.int64) * K
        s_mask = None
        s_other = None
    else:
        s_src = state_out_ptr
    if USE_CONV:
        if IS_SPEC_DECODING:
            cs_slot = gl.load(state_indices_ptr + i_n * stride_indices_seq).to(gl.int32)
        else:
            cs_slot = slot
        hb = conv_state_ptr + cs_slot.to(gl.int64) * stride_cs_slot
    else:
        hb = conv_state_ptr
    qkav_smem = gl.allocate_shared_memory(gl.float32, [4 * K], SMEM1)
    obuf = gl.allocate_shared_memory(gl.float32, [V], SMEM1)
    if USE_RMS_GATE:
        sbuf = gl.allocate_shared_memory(gl.float32, [RC], SMEM1)
    if RESIDENT_STATE:
        st = ()
        for i in gl.static_range(NC):
            if USE_INITIAL_STATE:
                st = st + (
                    gl.amd.cdna4.buffer_load(s_src + i * K, off_s, cache=".cg").to(
                        gl.float32
                    ),
                )
            else:
                st = st + (gl.zeros([RC, K], gl.float32, CHUNK),)
        if USE_CONV:
            h0, h1, h2, w0, w1, w2, w3 = _conv_load(
                hb, conv_weight_ptr, ho + seed * stride_cs_pos, wo, m, stride_cs_pos, LP
            )
            if IS_SPEC_DECODING:
                keep1 = h1
                keep2 = h2
        x_nxt = gl.load(xp)
        if gp is not None:
            g_nxt = gl.load(gp, mask=~m)
        if not IS_BETA_HEADWISE:
            b_nxt = gl.load(b_p)
        if USE_RMS_GATE:
            og_nxt = gl.amd.cdna4.buffer_load(og_p, off_o)
        if INPLACE_FINAL_STATE and IS_CONTINUOUS_BATCHING:
            slot_nxt = gl.load(state_indices_ptr + i_n * stride_indices_seq)

    for t in range(n_tok):
        if RESIDENT_STATE:
            more = (t + 1 < n_tok).to(gl.int32)
        if RESIDENT_STATE and INPLACE_FINAL_STATE and IS_CONTINUOUS_BATCHING:
            nxt = slot_nxt.to(gl.int32)
            slot_nxt = gl.load(state_indices_ptr + i_n * stride_indices_seq + t + more)
            if PAD_SLOT_GUARD and not IS_SPEC_DECODING:
                nxt = gl.where(nxt > 0, nxt, slot)
            slot = nxt
            row_out = slot * stride_state_out_slot_rows + i_hv * V
            s_dst = state_out_ptr + row_out.to(gl.int64) * K
        elif INPLACE_FINAL_STATE and IS_CONTINUOUS_BATCHING and n_tok > 1:
            nxt = gl.load(state_indices_ptr + i_n * stride_indices_seq + t).to(gl.int32)
            if PAD_SLOT_GUARD:
                nxt = gl.where(nxt > 0, nxt, slot)
            slot = nxt
            row_out = slot * stride_state_out_slot_rows + i_hv * V
            s_dst = state_out_ptr + row_out.to(gl.int64) * K
        elif INPLACE_FINAL_STATE:
            s_dst = s_src
        else:
            n_out = bos + t if IS_CONTINUOUS_BATCHING else i_n
            row_out = n_out * stride_state_out_slot_rows + i_hv * V
            s_dst = state_out_ptr + row_out.to(gl.int64) * K
        if not USE_INITIAL_STATE:
            s_mask = (off_s >= 0) & (t > 0)
            s_other = 0.0
        if RESIDENT_STATE:
            x = x_nxt
            x_nxt = gl.load(xp + xs * more)
            if gp is not None:
                # g has its own dtype: loaded apart and spliced into row 2
                x = gl.where(m, x.to(gl.float32), g_nxt.to(gl.float32))
                g_nxt = gl.load(gp + stride_g_token * more, mask=~m)
            if not IS_BETA_HEADWISE:
                b = b_nxt.to(gl.float32)
                b_nxt = gl.load(b_p + stride_beta_token * more)
            if USE_RMS_GATE:
                og = og_nxt
                og_nxt = gl.amd.cdna4.buffer_load(og_p + stride_og_token * more, off_o)
        else:
            if not IS_BETA_HEADWISE:
                b = gl.load(b_p).to(gl.float32)
            if USE_RMS_GATE:
                og = gl.amd.cdna4.buffer_load(og_p, off_o)
        if RESIDENT_STATE:
            if USE_CONV:
                acc = h0.to(gl.float32) * w0 + h1.to(gl.float32) * w1
                acc += h2.to(gl.float32) * w2 + x.to(gl.float32) * w3
                h0 = h1
                h1 = h2
                h2 = x.to(hb.dtype.element_ty)
                if IS_SPEC_DECODING:  # this token's input lands at W-2+t
                    gl.amd.cdna4.buffer_store(
                        h2, hb, ho + (W - 2 + t) * stride_cs_pos, mask=m, cache=".cs"
                    )
            else:
                acc = x.to(gl.float32)
            y = _finish(
                x,
                acc,
                m,
                a_exp,
                dt,
                gate_c,
                USE_CONV,
                USE_GATE_IN_KERNEL,
                USE_LOWER_BOUND,
            )
        else:
            bufs = ()
            for i in gl.static_range(NUM_BUFFERS):
                bufs = bufs + (
                    gl.amd.cdna4.buffer_load(
                        s_src + i * K, off_s, mask=s_mask, other=s_other, cache=".cg"
                    ),
                )
            y = _token(
                xp,
                gp,
                hb,
                conv_weight_ptr,
                ho,
                wo,
                m,
                stride_cs_pos,
                a_exp,
                dt,
                gate_c,
                LP,
                USE_CONV,
                USE_GATE_IN_KERNEL,
                USE_LOWER_BOUND,
            )
        if not IS_BETA_HEADWISE and APPLY_BETA_SIGMOID:
            b = sigmoid(b)
            if ALLOW_NEG_EIGVAL:
                b = b * 2.0
        qkav_smem.store(gl.reshape(y, [4 * K]))
        qv = qkav_smem.slice(0, K).load(K_LAYOUT)
        kv = qkav_smem.slice(K, K).load(K_LAYOUT)
        a = qkav_smem.slice(2 * K, K).load(K_LAYOUT)
        qk = gl.sum(qv * kv, axis=0)
        if USE_QK_L2NORM_IN_KERNEL:
            rq = gl.rsqrt(gl.sum(qv * qv, axis=0) + 1e-6) * scale
            rk = gl.rsqrt(gl.sum(kv * kv, axis=0) + 1e-6)
            qv = qv * rq
            kv = kv * rk
            qk = qk * (rq * rk)
        else:
            qv = qv * scale
            qk = qk * scale

        if RESIDENT_STATE and not IS_CONTINUOUS_BATCHING:
            s_keep = (off_s >= 0) & (t == n_tok - 1)
        elif IS_SPEC_DECODING and PAD_SLOT_GUARD:
            s_keep = (off_s >= 0) & (slot > 0)
        else:
            s_keep = None
        ss = gl.zeros([RC], gl.float32, R_LAYOUT)
        if RESIDENT_STATE:
            upd = ()
        vvs = ()
        for i in gl.static_range(NC):
            vvs = vvs + (qkav_smem.slice(3 * K, K).gather(rows + i, 0),)
        for i in gl.static_range(NC):
            if RESIDENT_STATE:
                S = st[i]
            else:
                S = bufs[0]
                if i + NUM_BUFFERS < NC:
                    nxt = gl.amd.cdna4.buffer_load(
                        s_src + (i + NUM_BUFFERS) * K,
                        off_s,
                        mask=s_mask,
                        other=s_other,
                        cache=".cg",
                    )
                    bufs = bufs[1:] + (nxt,)
                else:
                    bufs = bufs[1:]
            vv = vvs[i]
            if IS_BETA_HEADWISE:
                b = gl.load(b_p + rows + i).to(gl.float32)
                if APPLY_BETA_SIGMOID:
                    b = sigmoid(b)
                    if ALLOW_NEG_EIGVAL:
                        b = b * 2.0
            S = S.to(gl.float32) * a[None, :]  # decay:  Diag(alpha) S
            sk = gl.sum(S * kv[None, :], axis=1)  # read:   S^T k
            sq = gl.sum(S * qv[None, :], axis=1)  # read:   S^T q, pre-write
            err = (vv - sk) * b  # error:  beta * (v - S^T k)
            o = sq + err * qk  # output: (S + err (x) k)^T q
            S = S + err[:, None] * kv[None, :]  # write:  S += err (x) k

            gl.amd.cdna4.buffer_store(
                S.to(s_dst.dtype.element_ty),
                s_dst + i * K,
                off_s,
                mask=s_keep,
                cache=".wt",
            )
            if RESIDENT_STATE:
                upd = upd + (S,)
            obuf.slice(i * RC, RC).store(o)
            ss += o * o
        if RESIDENT_STATE:
            st = upd
        if USE_RMS_GATE:
            sbuf.store(ss)
        o = obuf.gather(o_idx, 0)
        if USE_RMS_GATE:
            rstd = gl.rsqrt(gl.sum(sbuf.load(RED), axis=0) / V + norm_eps)
            o = o * rstd * nw * sigmoid(og.to(gl.float32))
            og_p += stride_og_token
        gl.store(o_p + off_o, o.to(o_ptr.dtype.element_ty))
        s_src = s_dst
        xp += xs
        if gp is not None:
            gp += stride_g_token
        b_p += stride_beta_token
        o_p += stride_o_token

    if RESIDENT_STATE and USE_CONV:
        if IS_SPEC_DECODING:
            gl.amd.cdna4.buffer_store(keep1, hb, ho, mask=m, cache=".cs")
            gl.amd.cdna4.buffer_store(
                keep2, hb, ho + stride_cs_pos, mask=m, cache=".cs"
            )
        else:
            gl.amd.cdna4.buffer_store(h0, hb, ho, mask=m, cache=".cs")
            gl.amd.cdna4.buffer_store(h1, hb, ho + stride_cs_pos, mask=m, cache=".cs")
            gl.amd.cdna4.buffer_store(
                h2, hb, ho + 2 * stride_cs_pos, mask=m, cache=".cs"
            )


def get_kda_config(
    avg_T: int,
    num_seqs: int,
    HV: int,
    K: int,
    V: int,
    overrides: dict | None = None,
    fused: bool = False,
) -> dict:
    cfg_dir = resolve_config_dir("attention", "KDA_DECODE", backend="gluon")
    tuned = load_config_json(f"{cfg_dir}/DEFAULT.json")
    num_seq_heads = num_seqs * HV
    aligned = K % 32 == 0 and V % 32 == 0
    if fused:
        multi = avg_T > 1 and num_seq_heads < 3072 and "fused_t_gt1" in tuned
        bucket = "fused_t_gt1" if multi else "fused"
    elif avg_T > 1:
        if aligned and num_seq_heads >= 3072 and V % 128 == 0:
            bucket = "t_gt1_seq_heads_geq_3072"
        elif aligned and num_seq_heads >= 384:
            bucket = "t_gt1_seq_heads_geq_384"
        elif K % 16 == 0:
            bucket = "t_gt1_default"
        else:
            bucket = "default"
    elif aligned and 256 <= num_seq_heads <= 512:
        bucket = "t1_seq_heads_256_to_512"
    else:
        bucket = "default"
    config = dict(tuned[bucket])
    overrides = overrides or {}
    config.update(overrides)
    if "SK" not in overrides:
        config["SK"] = min(config["SK"], math.gcd(32, K))
        if "num_warps" not in overrides:
            config["num_warps"] = max(
                1, min(config["num_warps"], config["BV"] * config["SK"] // 32)
            )
    return config
