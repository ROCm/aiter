# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.


import math

from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.language.extra.hip import libdevice

_TDM = gl.amd.gfx1250.tdm


@gluon.constexpr_function
def _state_layout(v_first, ROWS, K, SK, NUM_WARPS):
    if v_first:
        return gl.BlockedLayout([ROWS, K // SK], [32 // SK, SK], [NUM_WARPS, 1], [1, 0])
    return gl.BlockedLayout([K // SK, ROWS], [SK, 32 // SK], [1, NUM_WARPS], [0, 1])


@gluon.constexpr_function
def _smem_layout(v_first, K, BV):
    if v_first:
        return gl.PaddedSharedLayout.with_identity_for([[K, 4]], [BV, K], [1, 0])
    return gl.PaddedSharedLayout.with_identity_for([[BV, 4]], [K, BV], [1, 0])


@gluon.jit
def _exp_scaled(scale, x):
    return gl.exp2((scale * math.log2(math.e)) * x)


@gluon.jit
def _softplus(x):
    return gl.where(x < 20.0, gl.log(1.0 + gl.exp(x)), x)


@gluon.jit
def _sigmoid(x):
    return 0.5 + 0.5 * libdevice.tanh(0.5 * x)


@gluon.jit
def _state_row(
    slot,
    stride_slot_rows,
    i_hv,
    i_v,
    K: gl.constexpr,
    V: gl.constexpr,
    BV: gl.constexpr,
    STATE_V_FIRST: gl.constexpr,
):
    if STATE_V_FIRST:
        return slot.to(gl.int32) * stride_slot_rows + i_hv * V + i_v * BV
    return slot.to(gl.int32) * stride_slot_rows + i_hv * K


@gluon.jit
def _fetch_token(
    q_p,
    k_p,
    v_p,
    g_p,
    b_p,
    off_k,
    off_v,
    IS_BETA_HEADWISE: gl.constexpr,
    NT_STREAM: gl.constexpr,
):
    NT: gl.constexpr = ".cs" if NT_STREAM else None
    qr = gl.amd.gfx1250.buffer_load(q_p, off_k, cache=NT)
    kr = gl.amd.gfx1250.buffer_load(k_p, off_k, cache=NT)
    vr = gl.amd.gfx1250.buffer_load(v_p, off_v, cache=NT)
    gr = gl.amd.gfx1250.buffer_load(g_p, off_k, cache=NT)
    if IS_BETA_HEADWISE:
        br = gl.amd.gfx1250.buffer_load(b_p, off_v, cache=NT)
    else:
        br = gl.load(b_p)
    return qr, kr, vr, gr, br


@gluon.jit
def _process_token(
    raw,
    a_exp,
    b_bias,
    lower_bound,
    scale: gl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: gl.constexpr,
    USE_GATE_IN_KERNEL: gl.constexpr,
    HAS_DT_BIAS: gl.constexpr,
    USE_LOWER_BOUND: gl.constexpr,
    APPLY_BETA_SIGMOID: gl.constexpr,
    ALLOW_NEG_EIGVAL: gl.constexpr,
):
    qr, kr, vr, gr, br = raw
    qv = qr.to(gl.float32)
    kv = kr.to(gl.float32)
    vv = vr.to(gl.float32)
    if USE_QK_L2NORM_IN_KERNEL:
        qv = qv * gl.rsqrt(gl.sum(qv * qv, axis=0) + 1e-6)
        kv = kv * gl.rsqrt(gl.sum(kv * kv, axis=0) + 1e-6)
    qv = qv * scale

    g = gr.to(gl.float32)
    if USE_GATE_IN_KERNEL:
        if HAS_DT_BIAS:
            g = g + b_bias
        if USE_LOWER_BOUND:
            a = _exp_scaled(lower_bound, _sigmoid(a_exp * g))
        else:
            a = _exp_scaled(-a_exp, _softplus(g))
    else:
        a = gl.exp(g)

    b = br.to(gl.float32)
    if APPLY_BETA_SIGMOID:
        b = _sigmoid(b)
        if ALLOW_NEG_EIGVAL:
            b = b * 2.0
    return a, kv, qv, vv, b


@gluon.jit(do_not_specialize=["T"])
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
    lower_bound,
    T,
    stride_indices_seq,
    stride_state_slot_rows,
    stride_state_out_slot_rows,
    state_out_rows,
    scale: gl.constexpr,
    H: gl.constexpr,
    HV: gl.constexpr,
    K: gl.constexpr,
    V: gl.constexpr,
    BV: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    SK: gl.constexpr,
    NUM_BUFFERS: gl.constexpr,
    NT_STREAM: gl.constexpr,
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
):
    gl.static_assert(V % BV == 0, "BV must divide V")
    gl.static_assert(32 % SK == 0, "SK must divide the wave")
    gl.static_assert(K % SK == 0, "SK must divide K")
    gl.static_assert(
        (BV * SK) % (32 * NUM_WARPS) == 0, "BV*SK must cover 32*NUM_WARPS lanes"
    )
    gl.static_assert(
        NUM_BUFFERS == 1 or NUM_BUFFERS == 2 or NUM_BUFFERS == 3,
    )

    ROWS: gl.constexpr = (BV * SK) // (32 * NUM_WARPS)
    STATE_LAYOUT: gl.constexpr = _state_layout(STATE_V_FIRST, ROWS, K, SK, NUM_WARPS)
    K_LAYOUT: gl.constexpr = gl.SliceLayout(0 if STATE_V_FIRST else 1, STATE_LAYOUT)
    V_LAYOUT: gl.constexpr = gl.SliceLayout(1 if STATE_V_FIRST else 0, STATE_LAYOUT)
    ROWLEN: gl.constexpr = K if STATE_V_FIRST else V

    NV: gl.constexpr = V // BV
    pid = gl.program_id(0)
    i_v = pid % NV
    i_nh = pid // NV
    i_n = i_nh // HV
    i_hv = i_nh % HV
    i_h = i_hv // (HV // H)

    if IS_VARLEN:
        bos = gl.load(cu_seqlens_ptr + i_n).to(gl.int32)
        n_tok = gl.load(cu_seqlens_ptr + i_n + 1).to(gl.int32) - bos
    else:
        bos = i_n * T
        n_tok = T
    if n_tok == 0:
        return

    off_k = gl.arange(0, K, layout=K_LAYOUT)
    off_v = gl.arange(0, BV, layout=V_LAYOUT)
    col = 0 if STATE_V_FIRST else i_v * BV
    if STATE_V_FIRST:
        off_s = off_v[:, None] * K + off_k[None, :]
    else:
        off_s = off_k[:, None] * V + off_v[None, :]

    if USE_GATE_IN_KERNEL:
        a_exp = gl.exp(gl.load(A_log_ptr + i_hv).to(gl.float32))
        if HAS_DT_BIAS:
            b_bias = gl.amd.gfx1250.buffer_load(dt_bias_ptr + i_hv * K, off_k).to(
                gl.float32
            )
        else:
            b_bias = 0.0
    else:
        a_exp = 0.0
        b_bias = 0.0

    tok0 = bos.to(gl.int64)
    q_p = q_ptr + (tok0 * H + i_h) * K
    k_p = k_ptr + (tok0 * H + i_h) * K
    g_p = g_ptr + (tok0 * HV + i_hv) * K
    v_p = v_ptr + (tok0 * HV + i_hv) * V + i_v * BV
    o_p = o_ptr + (tok0 * HV + i_hv) * V + i_v * BV
    if IS_BETA_HEADWISE:
        b_p = beta_ptr + (tok0 * HV + i_hv) * V + i_v * BV
        B_STEP: gl.constexpr = HV * V
    else:
        b_p = beta_ptr + tok0 * HV + i_hv
        B_STEP: gl.constexpr = HV

    if NUM_BUFFERS == 3:
        pf_end = (bos + n_tok).to(gl.int32)
        desc_q = _TDM.make_tensor_descriptor(
            base=q_ptr + i_h * K,
            shape=(pf_end, K),
            strides=(H * K, 1),
            block_shape=(8, K),
            layout=gl.SwizzledSharedLayout(1, 1, 1, order=[1, 0]),
        )
        _TDM.prefetch(desc_q, [bos, 0])
        desc_k = _TDM.make_tensor_descriptor(
            base=k_ptr + i_h * K,
            shape=(pf_end, K),
            strides=(H * K, 1),
            block_shape=(8, K),
            layout=gl.SwizzledSharedLayout(1, 1, 1, order=[1, 0]),
        )
        _TDM.prefetch(desc_k, [bos, 0])
        desc_g = _TDM.make_tensor_descriptor(
            base=g_ptr + i_hv * K,
            shape=(pf_end, K),
            strides=(HV * K, 1),
            block_shape=(8, K),
            layout=gl.SwizzledSharedLayout(1, 1, 1, order=[1, 0]),
        )
        _TDM.prefetch(desc_g, [bos, 0])
        desc_v = _TDM.make_tensor_descriptor(
            base=v_ptr + i_hv * V + i_v * BV,
            shape=(pf_end, BV),
            strides=(HV * V, 1),
            block_shape=(8, BV),
            layout=gl.SwizzledSharedLayout(1, 1, 1, order=[1, 0]),
        )
        _TDM.prefetch(desc_v, [bos, 0])
        if IS_BETA_HEADWISE:
            desc_b = _TDM.make_tensor_descriptor(
                base=beta_ptr + i_hv * V + i_v * BV,
                shape=(pf_end, BV),
                strides=(HV * V, 1),
                block_shape=(8, BV),
                layout=gl.SwizzledSharedLayout(1, 1, 1, order=[1, 0]),
            )
        else:
            desc_b = _TDM.make_tensor_descriptor(
                base=beta_ptr + i_hv,
                shape=(pf_end, 1),
                strides=(HV, 1),
                block_shape=(8, 1),
                layout=gl.SwizzledSharedLayout(1, 1, 1, order=[1, 0]),
            )
        _TDM.prefetch(desc_b, [bos, 0])

    if NUM_BUFFERS >= 2:
        nxt = _fetch_token(
            q_p, k_p, v_p, g_p, b_p, off_k, off_v, IS_BETA_HEADWISE, NT_STREAM
        )

    if USE_INITIAL_STATE:
        if IS_CONTINUOUS_BATCHING:
            if IS_SPEC_DECODING:
                seed = gl.load(num_accepted_ptr + i_n).to(gl.int32) - 1
            else:
                seed = 0
            slot = gl.load(state_indices_ptr + i_n * stride_indices_seq + seed).to(
                gl.int32
            )
        else:
            slot = i_n
        row_in = _state_row(
            slot, stride_state_slot_rows, i_hv, i_v, K, V, BV, STATE_V_FIRST
        )
        S = gl.amd.gfx1250.buffer_load(
            state_ptr + row_in.to(gl.int64) * ROWLEN + col, off_s
        ).to(gl.float32)
    elif STATE_V_FIRST:
        S = gl.full([BV, K], 0.0, gl.float32, STATE_LAYOUT)
    else:
        S = gl.full([K, BV], 0.0, gl.float32, STATE_LAYOUT)

    if USE_TDM_STORE and (IS_CONTINUOUS_BATCHING or STORE_FINAL_STATE):
        SMEM: gl.constexpr = _smem_layout(STATE_V_FIRST, K, BV)
        if STATE_V_FIRST:
            smem = gl.allocate_shared_memory(gl.float32, [2, BV, K], SMEM)
            desc_out = _TDM.make_tensor_descriptor(
                base=state_out_ptr,
                shape=(state_out_rows, K),
                strides=(K, 1),
                block_shape=(BV, K),
                layout=SMEM,
            )
        else:
            smem = gl.allocate_shared_memory(gl.float32, [2, K, BV], SMEM)
            desc_out = _TDM.make_tensor_descriptor(
                base=state_out_ptr,
                shape=(state_out_rows, V),
                strides=(V, 1),
                block_shape=(K, BV),
                layout=SMEM,
            )
        buf: gl.int32 = 0

    for t in range(n_tok):
        if NUM_BUFFERS >= 2:
            a, kv, qv, vv, b = _process_token(
                nxt,
                a_exp,
                b_bias,
                lower_bound,
                scale,
                USE_QK_L2NORM_IN_KERNEL,
                USE_GATE_IN_KERNEL,
                HAS_DT_BIAS,
                USE_LOWER_BOUND,
                APPLY_BETA_SIGMOID,
                ALLOW_NEG_EIGVAL,
            )
            adv = (t + 1 < n_tok).to(gl.int32)
            q_p += adv * H * K
            k_p += adv * H * K
            v_p += adv * HV * V
            g_p += adv * HV * K
            b_p += adv * B_STEP
            nxt = _fetch_token(
                q_p, k_p, v_p, g_p, b_p, off_k, off_v, IS_BETA_HEADWISE, NT_STREAM
            )
        else:
            raw = _fetch_token(
                q_p, k_p, v_p, g_p, b_p, off_k, off_v, IS_BETA_HEADWISE, NT_STREAM
            )
            q_p += H * K
            k_p += H * K
            v_p += HV * V
            g_p += HV * K
            b_p += B_STEP
            a, kv, qv, vv, b = _process_token(
                raw,
                a_exp,
                b_bias,
                lower_bound,
                scale,
                USE_QK_L2NORM_IN_KERNEL,
                USE_GATE_IN_KERNEL,
                HAS_DT_BIAS,
                USE_LOWER_BOUND,
                APPLY_BETA_SIGMOID,
                ALLOW_NEG_EIGVAL,
            )

        kq = gl.sum(kv * qv, axis=0)
        if STATE_V_FIRST:
            S = S * a[None, :]  # decay:  Diag(alpha) S
            stored = gl.sum(S * kv[None, :], axis=1)  # read:   S^T k
            odec = gl.sum(S * qv[None, :], axis=1)  # output: S^T q, pre-write
            err = (vv - stored) * b  # error:  beta * (v - S^T k)
            S = S + err[:, None] * kv[None, :]  # write:  S += beta * k (x) err
        else:
            S = S * a[:, None]  # decay:  Diag(alpha) S
            stored = gl.sum(S * kv[:, None], axis=0)  # read:   S^T k
            odec = gl.sum(S * qv[:, None], axis=0)  # output: S^T q, pre-write
            err = (vv - stored) * b  # error:  beta * (v - S^T k)
            S = S + kv[:, None] * err[None, :]  # write:  S += beta * k (x) err
        o = odec + err * kq
        gl.amd.gfx1250.buffer_store(
            o.to(o_ptr.dtype.element_ty), o_p, off_v, cache=".cs" if NT_STREAM else None
        )

        if IS_CONTINUOUS_BATCHING:
            if INPLACE_FINAL_STATE:
                out_slot = gl.load(state_indices_ptr + i_n * stride_indices_seq + t).to(
                    gl.int32
                )
            else:
                out_slot = bos + t
            row_out = _state_row(
                out_slot, stride_state_out_slot_rows, i_hv, i_v, K, V, BV, STATE_V_FIRST
            )
            if USE_TDM_STORE:
                _TDM.async_wait(1)
                smem.index(buf).store(S)
                gl.barrier()
                _TDM.async_store(desc_out, [row_out, col], smem.index(buf))
                buf = (buf + 1) % 2
            else:
                gl.amd.gfx1250.buffer_store(
                    S.to(state_out_ptr.dtype.element_ty),
                    state_out_ptr + row_out.to(gl.int64) * ROWLEN + col,
                    off_s,
                )

        o_p += HV * V

    if USE_TDM_STORE and IS_CONTINUOUS_BATCHING:
        _TDM.async_wait(0)

    if STORE_FINAL_STATE and not IS_CONTINUOUS_BATCHING:
        row_out = _state_row(
            i_n, stride_state_out_slot_rows, i_hv, i_v, K, V, BV, STATE_V_FIRST
        )
        if USE_TDM_STORE:
            smem.index(0).store(S)
            gl.barrier()
            _TDM.async_store(desc_out, [row_out, col], smem.index(0))
            _TDM.async_wait(0)
        else:
            gl.amd.gfx1250.buffer_store(
                S.to(state_out_ptr.dtype.element_ty),
                state_out_ptr + row_out.to(gl.int64) * ROWLEN + col,
                off_s,
            )
