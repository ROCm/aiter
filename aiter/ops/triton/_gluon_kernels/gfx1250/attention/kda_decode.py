# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.


import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

_TDM = gl.amd.gfx1250.tdm


@gluon.constexpr_function
def _state_layout(v_first, ROWS, K, NUM_WARPS):
    if v_first:
        return gl.BlockedLayout([ROWS, K], [32, 1], [NUM_WARPS, 1], [1, 0])
    return gl.BlockedLayout([K, ROWS], [1, 32], [1, NUM_WARPS], [0, 1])


@gluon.constexpr_function
def _smem_layout(v_first, K, BV, LDS_PAD):
    if v_first:
        return gl.PaddedSharedLayout.with_identity_for([[K, LDS_PAD]], [BV, K], [1, 0])
    return gl.PaddedSharedLayout.with_identity_for([[BV, LDS_PAD]], [K, BV], [1, 0])


@gluon.jit
def _softplus(x):
    return gl.where(x < 20.0, gl.log(1.0 + gl.exp(x)), x)


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
def _load_token(
    q_p,
    k_p,
    v_p,
    g_p,
    b_p,
    off_k,
    off_v,
    a_exp,
    b_bias,
    lower_bound,
    scale: gl.constexpr,
    IS_BETA_HEADWISE: gl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: gl.constexpr,
    USE_GATE_IN_KERNEL: gl.constexpr,
    HAS_DT_BIAS: gl.constexpr,
    USE_LOWER_BOUND: gl.constexpr,
    APPLY_BETA_SIGMOID: gl.constexpr,
    ALLOW_NEG_EIGVAL: gl.constexpr,
):
    qv = gl.amd.gfx1250.buffer_load(q_p, off_k).to(gl.float32)
    kv = gl.amd.gfx1250.buffer_load(k_p, off_k).to(gl.float32)
    vv = gl.amd.gfx1250.buffer_load(v_p, off_v).to(gl.float32)
    if USE_QK_L2NORM_IN_KERNEL:
        qv = qv / gl.sqrt(gl.sum(qv * qv, axis=0) + 1e-6)
        kv = kv / gl.sqrt(gl.sum(kv * kv, axis=0) + 1e-6)
    qv = qv * scale

    g = gl.amd.gfx1250.buffer_load(g_p, off_k).to(gl.float32)
    if USE_GATE_IN_KERNEL:
        if HAS_DT_BIAS:
            g = g + b_bias
        if USE_LOWER_BOUND:
            gk = lower_bound * tl.sigmoid(a_exp * g)
        else:
            gk = -a_exp * _softplus(g)
    else:
        gk = g
    a = gl.exp(gk)

    if IS_BETA_HEADWISE:
        b = gl.amd.gfx1250.buffer_load(b_p, off_v).to(gl.float32)
    else:
        b = gl.load(b_p).to(gl.float32)
    if APPLY_BETA_SIGMOID:
        b = tl.sigmoid(b)
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
    state_rows,
    state_out_rows,
    scale: gl.constexpr,
    H: gl.constexpr,
    HV: gl.constexpr,
    K: gl.constexpr,
    V: gl.constexpr,
    BV: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    NUM_BUFFERS: gl.constexpr,
    LDS_PAD: gl.constexpr,
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
):
    gl.static_assert(V % BV == 0, "BV must divide V")
    gl.static_assert(BV % (32 * NUM_WARPS) == 0, "BV must cover 32*NUM_WARPS lanes")

    ROWS: gl.constexpr = BV // (32 * NUM_WARPS)
    STATE_LAYOUT: gl.constexpr = _state_layout(STATE_V_FIRST, ROWS, K, NUM_WARPS)
    SMEM_LAYOUT: gl.constexpr = _smem_layout(STATE_V_FIRST, K, BV, LDS_PAD)
    K_LAYOUT: gl.constexpr = gl.SliceLayout(0 if STATE_V_FIRST else 1, STATE_LAYOUT)
    V_LAYOUT: gl.constexpr = gl.SliceLayout(1 if STATE_V_FIRST else 0, STATE_LAYOUT)

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

    if USE_INITIAL_STATE:
        if STATE_V_FIRST:
            desc_in = _TDM.make_tensor_descriptor(
                base=state_ptr,
                shape=(state_rows, K),
                strides=(K, 1),
                block_shape=(BV, K),
                layout=SMEM_LAYOUT,
            )
        else:
            desc_in = _TDM.make_tensor_descriptor(
                base=state_ptr,
                shape=(state_rows, V),
                strides=(V, 1),
                block_shape=(K, BV),
                layout=SMEM_LAYOUT,
            )
    if IS_CONTINUOUS_BATCHING or STORE_FINAL_STATE:
        if STATE_V_FIRST:
            desc_out = _TDM.make_tensor_descriptor(
                base=state_out_ptr,
                shape=(state_out_rows, K),
                strides=(K, 1),
                block_shape=(BV, K),
                layout=SMEM_LAYOUT,
            )
        else:
            desc_out = _TDM.make_tensor_descriptor(
                base=state_out_ptr,
                shape=(state_out_rows, V),
                strides=(V, 1),
                block_shape=(K, BV),
                layout=SMEM_LAYOUT,
            )
    if STATE_V_FIRST:
        smem = gl.allocate_shared_memory(gl.float32, [NUM_BUFFERS, BV, K], SMEM_LAYOUT)
    else:
        smem = gl.allocate_shared_memory(gl.float32, [NUM_BUFFERS, K, BV], SMEM_LAYOUT)

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
        _TDM.async_load(
            desc_in,
            [
                _state_row(
                    slot, stride_state_slot_rows, i_hv, i_v, K, V, BV, STATE_V_FIRST
                ),
                col,
            ],
            smem.index(0),
        )
        _TDM.async_wait(0)
        S = smem.index(0).load(STATE_LAYOUT)
    elif STATE_V_FIRST:
        S = gl.full([BV, K], 0.0, gl.float32, STATE_LAYOUT)
    else:
        S = gl.full([K, BV], 0.0, gl.float32, STATE_LAYOUT)
    buf: gl.int32 = 0

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

    for t in range(n_tok):
        a, kv, qv, vv, b = _load_token(
            q_p,
            k_p,
            v_p,
            g_p,
            b_p,
            off_k,
            off_v,
            a_exp,
            b_bias,
            lower_bound,
            scale,
            IS_BETA_HEADWISE,
            USE_QK_L2NORM_IN_KERNEL,
            USE_GATE_IN_KERNEL,
            HAS_DT_BIAS,
            USE_LOWER_BOUND,
            APPLY_BETA_SIGMOID,
            ALLOW_NEG_EIGVAL,
        )

        if STATE_V_FIRST:
            S = S * a[None, :]  # decay:  Diag(alpha) S
            stored = gl.sum(S * kv[None, :], axis=1)  # read:   S^T k
            err = (vv - stored) * b  # error:  beta * (v - S^T k)
            S = S + err[:, None] * kv[None, :]  # write:  S += beta * k (x) err
            o = gl.sum(S * qv[None, :], axis=1)  # output: S^T q
        else:
            S = S * a[:, None]  # decay:  Diag(alpha) S
            stored = gl.sum(S * kv[:, None], axis=0)  # read:   S^T k
            err = (vv - stored) * b  # error:  beta * (v - S^T k)
            S = S + kv[:, None] * err[None, :]  # write:  S += beta * k (x) err
            o = gl.sum(S * qv[:, None], axis=0)  # output: S^T q
        gl.amd.gfx1250.buffer_store(o.to(o_ptr.dtype.element_ty), o_p, off_v)

        if IS_CONTINUOUS_BATCHING:
            if INPLACE_FINAL_STATE:
                out_slot = gl.load(state_indices_ptr + i_n * stride_indices_seq + t).to(
                    gl.int32
                )
            else:
                out_slot = bos + t
            _TDM.async_wait(NUM_BUFFERS - 1)  # wait only for what we refill
            smem.index(buf).store(S)
            gl.barrier()
            _TDM.async_store(
                desc_out,
                [
                    _state_row(
                        out_slot,
                        stride_state_out_slot_rows,
                        i_hv,
                        i_v,
                        K,
                        V,
                        BV,
                        STATE_V_FIRST,
                    ),
                    col,
                ],
                smem.index(buf),
            )
            buf = (buf + 1) % NUM_BUFFERS

        q_p += H * K
        k_p += H * K
        o_p += HV * V
        v_p += HV * V
        g_p += HV * K
        b_p += B_STEP

    if IS_CONTINUOUS_BATCHING:
        _TDM.async_wait(0)
    elif STORE_FINAL_STATE:
        smem.index(0).store(S)
        gl.barrier()
        _TDM.async_store(
            desc_out,
            [
                _state_row(
                    i_n, stride_state_out_slot_rows, i_hv, i_v, K, V, BV, STATE_V_FIRST
                ),
                col,
            ],
            smem.index(0),
        )
        _TDM.async_wait(0)
