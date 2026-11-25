# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""gfx1250 Gluon kernel of ``kda_gating_decode``: decode, fork and spec verify.

One program keeps ``BV`` V-rows of one (sequence, value head) state tile in
registers for all tokens of the sequence. Each lane holds runs of ``VEC``
consecutive K columns, so the state loads and stores are wide. The program
requests the whole tile before the math, so it waits for memory one time.

The read slot is ``read[seq, accepted - 1]`` in spec verify, else
``read[seq]``. A negative read slot, an empty sequence or an accepted count
below 1 leaves the output and all slots unchanged. A negative write slot still
writes the output.
"""

from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from aiter.ops.triton._gluon_kernels.common.utils import fast_sigmoid
from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr
from aiter.ops.triton.utils.config_utils import (
    load_config_json,
    resolve_config_dir,
    select_leq_config,
)

_LOG2E = gl.constexpr(1.4426950408889634)


@gluon.constexpr_function
def _tile_layout(BV, K, TK, VEC, NUM_WARPS):
    tr = 32 // TK
    return gl.BlockedLayout(
        [BV // (tr * NUM_WARPS), VEC], [tr, TK], [NUM_WARPS, 1], [1, 0]
    )


@gluon.constexpr_function
def _gate_layout(L, NUM_WARPS, GATE_DIST):
    """Layout of the per-column gate inputs.

    With ``GATE_DIST > 0``, each thread holds one K column. Then each decay is
    computed one time in the program and moved to the tile layout.
    """
    if GATE_DIST > 0:
        return gl.BlockedLayout([1], [32], [NUM_WARPS], [0])
    return gl.SliceLayout(0, L)


@gluon.jit
def _token_inputs(q_p, k_p, a_p, v_p, b_p, cols, acols, rows):
    q = gl.amd.gfx1250.buffer_load(q_p, cols)
    k = gl.amd.gfx1250.buffer_load(k_p, cols)
    a = gl.amd.gfx1250.buffer_load(a_p, acols)
    v = gl.amd.gfx1250.buffer_load(v_p, rows)
    b = gl.load(b_p)
    return q, k, a, v, b


@gluon.jit
def _step(S, q, k, a, v, b, exp_alog, bias, lower_bound, scale, FAST_DIV: gl.constexpr):
    """One recurrence step on the register tile. Returns (S, o)."""
    q = q.to(gl.float32)
    k = k.to(gl.float32)
    q = q * (gl.rsqrt(gl.sum(q * q, axis=0) + 1e-6) * scale)
    k = k * gl.rsqrt(gl.sum(k * k, axis=0) + 1e-6)
    z = exp_alog * (a.to(gl.float32) + bias)
    if FAST_DIV:
        decay = gl.exp2((lower_bound * _LOG2E) * fast_sigmoid(z))
        beta = fast_sigmoid(b)
    else:
        # The tuned in-place configs at up to 4596 sequence-heads are faster
        # with the IEEE divide than with fast_sigmoid.
        decay = gl.exp2((lower_bound * _LOG2E) / (1.0 + gl.exp2(-z * _LOG2E)))
        beta = 1.0 / (1.0 + gl.exp2(-b.to(gl.float32) * _LOG2E))
    decay = gl.convert_layout(decay, k.type.layout)
    S = S * decay[None, :]
    err = (v.to(gl.float32) - gl.sum(S * k[None, :], axis=1)) * beta
    S = S + err[:, None] * k[None, :]
    o = gl.sum(S * q[None, :], axis=1)
    return S, o


@gluon.jit
def _one_token(
    q_ptr,
    k_ptr,
    v_ptr,
    a_ptr,
    b_ptr,
    o_ptr,
    state_ptr,
    tok,
    slot_in,
    slot_out,
    i_h,
    i_hv,
    i_v,
    exp_alog,
    dt_bias_ptr,
    stride_q_tok,
    stride_k_tok,
    stride_v_tok,
    stride_a_tok,
    stride_b_tok,
    stride_o_tok,
    stride_slot,
    lower_bound,
    scale,
    K: gl.constexpr,
    V: gl.constexpr,
    BV: gl.constexpr,
    TK: gl.constexpr,
    VEC: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    LOAD_CACHE: gl.constexpr,
    STORE_CACHE: gl.constexpr,
    GATE_DIST: gl.constexpr,
    STORE_BURST: gl.constexpr = False,
    FAST_DIV: gl.constexpr = True,
):
    L: gl.constexpr = _tile_layout(BV, K, TK, VEC, NUM_WARPS)
    rows = gl.arange(0, BV, layout=gl.SliceLayout(1, L))
    cols = gl.arange(0, K, layout=gl.SliceLayout(0, L))
    acols = gl.arange(0, K, layout=_gate_layout(L, NUM_WARPS, GATE_DIST))
    off_s = rows[:, None] * K + cols[None, :]
    tile = (i_hv * V + i_v * BV) * K
    if GATE_DIST < 2:
        S = gl.amd.gfx1250.buffer_load(
            state_ptr + (slot_in.to(gl.int64) * stride_slot + tile),
            off_s,
            cache=LOAD_CACHE,
        )
    q, k, a, v, b = _token_inputs(
        q_ptr + tok * stride_q_tok + i_h * K,
        k_ptr + tok * stride_k_tok + i_h * K,
        a_ptr + tok * stride_a_tok + i_hv * K,
        v_ptr + tok * stride_v_tok + i_hv * V + i_v * BV,
        b_ptr + tok * stride_b_tok + i_hv,
        cols,
        acols,
        rows,
    )
    bias = gl.amd.gfx1250.buffer_load(dt_bias_ptr + i_hv * K, acols).to(gl.float32)
    if GATE_DIST >= 2:
        # Loads complete in order: the gate math runs while the state loads.
        gl.amd.hint.sched_barrier()
        S = gl.amd.gfx1250.buffer_load(
            state_ptr + (slot_in.to(gl.int64) * stride_slot + tile),
            off_s,
            cache=LOAD_CACHE,
        )
    if GATE_DIST > 0:
        # Keep the state load above the gate math.
        gl.amd.hint.sched_barrier()
    S, o = _step(S, q, k, a, v, b, exp_alog, bias, lower_bound, scale, FAST_DIV)
    if STORE_BURST:
        # Issue the state stores together, after all the math.
        gl.amd.hint.sched_barrier()
    if slot_out >= 0:
        gl.amd.gfx1250.buffer_store(
            S,
            state_ptr + (slot_out.to(gl.int64) * stride_slot + tile),
            off_s,
            cache=STORE_CACHE,
        )
    gl.amd.gfx1250.buffer_store(
        o.to(o_ptr.dtype.element_ty),
        o_ptr + tok * stride_o_tok + i_hv * V + i_v * BV,
        rows,
    )


@gluon.jit
def _multi_token(
    q_ptr,
    k_ptr,
    v_ptr,
    a_ptr,
    b_ptr,
    o_ptr,
    state_ptr,
    write_ptr,
    idx_row,
    bos,
    n,
    slot_in,
    i_h,
    i_hv,
    i_v,
    exp_alog,
    dt_bias_ptr,
    stride_q_tok,
    stride_k_tok,
    stride_v_tok,
    stride_a_tok,
    stride_b_tok,
    stride_o_tok,
    stride_slot,
    lower_bound,
    scale,
    K: gl.constexpr,
    V: gl.constexpr,
    BV: gl.constexpr,
    TK: gl.constexpr,
    VEC: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    LOAD_CACHE: gl.constexpr,
    STORE_CACHE: gl.constexpr,
    GATE_DIST: gl.constexpr,
):
    L: gl.constexpr = _tile_layout(BV, K, TK, VEC, NUM_WARPS)
    rows = gl.arange(0, BV, layout=gl.SliceLayout(1, L))
    cols = gl.arange(0, K, layout=gl.SliceLayout(0, L))
    acols = gl.arange(0, K, layout=_gate_layout(L, NUM_WARPS, GATE_DIST))
    off_s = rows[:, None] * K + cols[None, :]
    tile = (i_hv * V + i_v * BV) * K
    S = gl.amd.gfx1250.buffer_load(
        state_ptr + (slot_in.to(gl.int64) * stride_slot + tile), off_s, cache=LOAD_CACHE
    )
    tok = bos.to(gl.int64)
    q_p = q_ptr + tok * stride_q_tok + i_h * K
    k_p = k_ptr + tok * stride_k_tok + i_h * K
    a_p = a_ptr + tok * stride_a_tok + i_hv * K
    v_p = v_ptr + tok * stride_v_tok + i_hv * V + i_v * BV
    b_p = b_ptr + tok * stride_b_tok + i_hv
    o_p = o_ptr + tok * stride_o_tok + i_hv * V + i_v * BV
    q, k, a, v, b = _token_inputs(q_p, k_p, a_p, v_p, b_p, cols, acols, rows)
    bias = gl.amd.gfx1250.buffer_load(dt_bias_ptr + i_hv * K, acols).to(gl.float32)
    slot_out = gl.load(write_ptr + idx_row)
    for t in range(n):
        # Prefetch the inputs of the next token. The last token loads itself again.
        nx = gl.minimum(t + 1, n - 1).to(gl.int64)
        nq, nk, na, nv, nb = _token_inputs(
            q_p + nx * stride_q_tok,
            k_p + nx * stride_k_tok,
            a_p + nx * stride_a_tok,
            v_p + nx * stride_v_tok,
            b_p + nx * stride_b_tok,
            cols,
            acols,
            rows,
        )
        nslot = gl.load(write_ptr + idx_row + nx)
        # Keep the prefetch above the math.
        gl.amd.hint.sched_barrier()
        S, o = _step(S, q, k, a, v, b, exp_alog, bias, lower_bound, scale, True)
        gl.amd.gfx1250.buffer_store(
            o.to(o_ptr.dtype.element_ty), o_p + t * stride_o_tok, rows
        )
        if slot_out >= 0:
            gl.amd.gfx1250.buffer_store(
                S,
                state_ptr + (slot_out.to(gl.int64) * stride_slot + tile),
                off_s,
                cache=STORE_CACHE,
            )
        q, k, a, v, b = nq, nk, na, nv, nb
        slot_out = nslot


@gluon.jit
def _multi_token_pre(
    q_ptr,
    k_ptr,
    v_ptr,
    a_ptr,
    b_ptr,
    o_ptr,
    state_ptr,
    write_ptr,
    idx_row,
    bos,
    n,
    slot_in,
    i_h,
    i_hv,
    i_v,
    exp_alog,
    dt_bias_ptr,
    stride_q_tok,
    stride_k_tok,
    stride_v_tok,
    stride_a_tok,
    stride_b_tok,
    stride_o_tok,
    stride_slot,
    lower_bound,
    scale,
    K: gl.constexpr,
    V: gl.constexpr,
    BV: gl.constexpr,
    TK: gl.constexpr,
    VEC: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    LOAD_CACHE: gl.constexpr,
    STORE_CACHE: gl.constexpr,
    CH: gl.constexpr,
):
    """Multi-token recurrence that precomputes the per-token column vectors.

    All threads compute decay, normalized q, decay * k and beta * k for ``CH``
    tokens at a time into LDS. The token loop reads them from LDS, so each row
    group does not compute them again.
    """
    L: gl.constexpr = _tile_layout(BV, K, TK, VEC, NUM_WARPS)
    COL: gl.constexpr = gl.SliceLayout(0, L)
    rows = gl.arange(0, BV, layout=gl.SliceLayout(1, L))
    cols = gl.arange(0, K, layout=COL)
    off_s = rows[:, None] * K + cols[None, :]
    tile = (i_hv * V + i_v * BV) * K
    S = gl.amd.gfx1250.buffer_load(
        state_ptr + (slot_in.to(gl.int64) * stride_slot + tile), off_s, cache=LOAD_CACHE
    )
    PL: gl.constexpr = gl.BlockedLayout(
        [1, 4], [32 // (K // 4), K // 4], [NUM_WARPS, 1], [1, 0]
    )
    SH: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [1, 0])
    SH1: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [0])
    pt = gl.arange(0, CH, layout=gl.SliceLayout(1, PL))
    pc = gl.arange(0, K, layout=gl.SliceLayout(0, PL))
    bias = gl.load(dt_bias_ptr + i_hv * K + pc).to(gl.float32)
    s_d = gl.allocate_shared_memory(gl.float32, [CH, K], SH)
    s_q = gl.allocate_shared_memory(gl.float32, [CH, K], SH)
    s_dk = gl.allocate_shared_memory(gl.float32, [CH, K], SH)
    s_bk = gl.allocate_shared_memory(gl.float32, [CH, K], SH)
    tok0 = bos.to(gl.int64)
    v_p = v_ptr + tok0 * stride_v_tok + i_hv * V + i_v * BV
    o_p = o_ptr + tok0 * stride_o_tok + i_hv * V + i_v * BV
    v = gl.amd.gfx1250.buffer_load(v_p, rows)
    slot_out = gl.load(write_ptr + idx_row)
    for c0 in range(0, n, CH):
        tt = (tok0 + c0 + pt).to(gl.int64)
        m = (c0 + pt) < n
        m2 = m[:, None]
        qc = gl.load(
            q_ptr + tt[:, None] * stride_q_tok + i_h * K + pc[None, :],
            mask=m2,
            other=0.0,
        ).to(gl.float32)
        kc = gl.load(
            k_ptr + tt[:, None] * stride_k_tok + i_h * K + pc[None, :],
            mask=m2,
            other=0.0,
        ).to(gl.float32)
        ac = gl.load(
            a_ptr + tt[:, None] * stride_a_tok + i_hv * K + pc[None, :],
            mask=m2,
            other=0.0,
        ).to(gl.float32)
        bc = gl.load(b_ptr + tt * stride_b_tok + i_hv, mask=m, other=0.0).to(gl.float32)
        qc = qc * (gl.rsqrt(gl.sum(qc * qc, axis=1) + 1e-6) * scale)[:, None]
        kc = kc * gl.rsqrt(gl.sum(kc * kc, axis=1) + 1e-6)[:, None]
        z = exp_alog * (ac + bias[None, :])
        dc = gl.exp2((lower_bound * _LOG2E) * fast_sigmoid(z))
        beta = fast_sigmoid(bc)
        s_d.store(dc)
        s_q.store(qc)
        s_dk.store(dc * kc)
        s_bk.store(kc * beta[:, None])
        for t in range(c0, gl.minimum(c0 + CH, n)):
            nx = gl.minimum(t + 1, n - 1).to(gl.int64)
            nv = gl.amd.gfx1250.buffer_load(v_p + nx * stride_v_tok, rows)
            nslot = gl.load(write_ptr + idx_row + nx)
            j = t - c0
            d = s_d.reinterpret(gl.float32, [CH, K], SH1).index(j).load(COL)
            dk = s_dk.reinterpret(gl.float32, [CH, K], SH1).index(j).load(COL)
            bk = s_bk.reinterpret(gl.float32, [CH, K], SH1).index(j).load(COL)
            q = s_q.reinterpret(gl.float32, [CH, K], SH1).index(j).load(COL)
            gl.amd.hint.sched_barrier()
            err = v.to(gl.float32) - gl.sum(S * dk[None, :], axis=1)
            S = S * d[None, :] + err[:, None] * bk[None, :]
            o = gl.sum(S * q[None, :], axis=1)
            gl.amd.gfx1250.buffer_store(
                o.to(o_ptr.dtype.element_ty), o_p + t * stride_o_tok, rows
            )
            if slot_out >= 0:
                gl.amd.gfx1250.buffer_store(
                    S,
                    state_ptr + (slot_out.to(gl.int64) * stride_slot + tile),
                    off_s,
                    cache=STORE_CACHE,
                )
            v = nv
            slot_out = nslot


_kda_gating_repr = make_kernel_repr(
    "kda_gating_kernel",
    [
        "HV",
        "BV",
        "TK",
        "VEC",
        "NUM_WARPS",
        "LOAD_CACHE",
        "STORE_CACHE",
        "GATE_DIST",
        "INPLACE_TK",
        "INPLACE_VEC",
        "INPLACE_LOAD_CACHE",
        "INPLACE_STORE_CACHE",
        "INPLACE_GATE_DIST",
        "INPLACE_BURST",
        "INPLACE_FAST_DIV",
        "MULTI_TOKEN",
        "IS_SPEC",
        "MULTI_PRE",
    ],
)


@gluon.jit(repr=_kda_gating_repr)
def kda_gating_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    a_ptr,
    b_ptr,
    A_log_ptr,
    dt_bias_ptr,
    o_ptr,
    state_ptr,
    cu_ptr,
    write_ptr,
    read_ptr,
    accepted_ptr,
    stride_idx_seq,
    stride_q_tok,
    stride_k_tok,
    stride_v_tok,
    stride_a_tok,
    stride_b_tok,
    stride_o_tok,
    stride_slot,
    lower_bound,
    scale,
    H: gl.constexpr,
    HV: gl.constexpr,
    K: gl.constexpr,
    V: gl.constexpr,
    BV: gl.constexpr,
    TK: gl.constexpr,
    VEC: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    LOAD_CACHE: gl.constexpr,
    STORE_CACHE: gl.constexpr,
    INPLACE_TK: gl.constexpr,
    INPLACE_VEC: gl.constexpr,
    INPLACE_LOAD_CACHE: gl.constexpr,
    INPLACE_STORE_CACHE: gl.constexpr,
    MULTI_TOKEN: gl.constexpr,
    IS_SPEC: gl.constexpr,
    GATE_DIST: gl.constexpr,
    INPLACE_GATE_DIST: gl.constexpr,
    INPLACE_BURST: gl.constexpr,
    INPLACE_FAST_DIV: gl.constexpr,
    MULTI_PRE: gl.constexpr,
):
    """One program computes ``BV`` V-rows of one (sequence, value head) tile.

    With ``INPLACE_VEC > 0``, a one-token program whose write slot is its read
    slot uses the ``INPLACE_*`` layout and cache policy. An in-place store
    goes to the lines that its load brought in, and a fork store goes to lines
    that nothing reads. The other one-token programs, and all of them when
    ``INPLACE_VEC`` is 0, compute the sigmoids with ``fast_sigmoid``
    (``fast_dividef``). The ``INPLACE_*`` config uses the IEEE divide when
    ``INPLACE_FAST_DIV`` is False, so the two configs can give results that
    are different in the last bits.
    """
    NV: gl.constexpr = V // BV
    pid = gl.program_id(0)
    i_v = pid % NV
    i_nh = pid // NV
    i_hv = i_nh % HV
    seq = i_nh // HV
    i_h = i_hv // (HV // H)

    bos = gl.load(cu_ptr + seq)
    eos = gl.load(cu_ptr + seq + 1)
    idx_row = seq.to(gl.int64) * stride_idx_seq
    if IS_SPEC:
        col = gl.load(accepted_ptr + seq) - 1
        # A count below 1, which an empty sequence can have, names no read slot.
        slot_in = gl.load(
            read_ptr + idx_row + col, mask=(eos > bos) & (col >= 0), other=-1
        )
    else:
        slot_in = gl.load(read_ptr + idx_row)
        # Load the write slot with the read slot: the in-place test needs both.
        slot_out = gl.load(write_ptr + idx_row)

    if (eos > bos) & (slot_in >= 0):
        exp_alog = gl.exp(gl.load(A_log_ptr + i_hv).to(gl.float32))
        if MULTI_TOKEN and (MULTI_PRE > 0):
            _multi_token_pre(
                q_ptr,
                k_ptr,
                v_ptr,
                a_ptr,
                b_ptr,
                o_ptr,
                state_ptr,
                write_ptr,
                idx_row,
                bos,
                eos - bos,
                slot_in,
                i_h,
                i_hv,
                i_v,
                exp_alog,
                dt_bias_ptr,
                stride_q_tok,
                stride_k_tok,
                stride_v_tok,
                stride_a_tok,
                stride_b_tok,
                stride_o_tok,
                stride_slot,
                lower_bound,
                scale,
                K,
                V,
                BV,
                TK,
                VEC,
                NUM_WARPS,
                LOAD_CACHE,
                STORE_CACHE,
                MULTI_PRE,
            )
        elif MULTI_TOKEN:
            _multi_token(
                q_ptr,
                k_ptr,
                v_ptr,
                a_ptr,
                b_ptr,
                o_ptr,
                state_ptr,
                write_ptr,
                idx_row,
                bos,
                eos - bos,
                slot_in,
                i_h,
                i_hv,
                i_v,
                exp_alog,
                dt_bias_ptr,
                stride_q_tok,
                stride_k_tok,
                stride_v_tok,
                stride_a_tok,
                stride_b_tok,
                stride_o_tok,
                stride_slot,
                lower_bound,
                scale,
                K,
                V,
                BV,
                TK,
                VEC,
                NUM_WARPS,
                LOAD_CACHE,
                STORE_CACHE,
                GATE_DIST,
            )
        else:
            tok = bos.to(gl.int64)
            # The loop is unrolled at compile time, so each config gets its own
            # code. USE_FORK = 0 is the INPLACE_* config. It is first because
            # the compiler puts an LDS barrier before the second copy: it does
            # not know that only one copy runs. Keep the test in the if: in a
            # local variable, the constexpr False of INPLACE_VEC = 0 becomes a
            # tensor, and the INPLACE_* copy is then compiled.
            for USE_FORK in gl.static_range(2):
                if ((INPLACE_VEC > 0) and (slot_out == slot_in)) != (USE_FORK == 1):
                    _one_token(
                        q_ptr,
                        k_ptr,
                        v_ptr,
                        a_ptr,
                        b_ptr,
                        o_ptr,
                        state_ptr,
                        tok,
                        slot_in,
                        slot_out,
                        i_h,
                        i_hv,
                        i_v,
                        exp_alog,
                        dt_bias_ptr,
                        stride_q_tok,
                        stride_k_tok,
                        stride_v_tok,
                        stride_a_tok,
                        stride_b_tok,
                        stride_o_tok,
                        stride_slot,
                        lower_bound,
                        scale,
                        K,
                        V,
                        BV,
                        TK if USE_FORK else INPLACE_TK,
                        VEC if USE_FORK else INPLACE_VEC,
                        NUM_WARPS,
                        LOAD_CACHE if USE_FORK else INPLACE_LOAD_CACHE,
                        STORE_CACHE if USE_FORK else INPLACE_STORE_CACHE,
                        GATE_DIST if USE_FORK else INPLACE_GATE_DIST,
                        False if USE_FORK else INPLACE_BURST,
                        True if USE_FORK else INPLACE_FAST_DIV,
                    )


def get_kda_gating_config(num_seq_heads: int, multi_token: bool) -> dict:
    """Return the ``kda_gating_kernel`` config for ``num_seq_heads`` programs.

    ``num_seq_heads`` is the number of sequences times the number of value
    heads. A spec-verify or other 2D slot table uses the ``multi_token`` table.
    """
    cfg_dir = resolve_config_dir("attention", "KDA_GATING", backend="gluon")
    tuned = load_config_json(f"{cfg_dir}/DEFAULT.json")
    table = tuned["multi_token" if multi_token else "one_token"]
    return select_leq_config(table, num_seq_heads, prefix="SEQ_HEADS_LEQ_")
