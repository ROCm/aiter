# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Context-parallel GDN K5 kernels using affine segment summaries.

The ordinary K5 launch assigns one program to each ``(sequence, head, V tile)``
and walks every 64-token chunk serially.  A long single sequence therefore
launches only 128 programs on the Qwen3.8 TP8 shape.  This path cuts each
sequence into independent segments and uses three recurrence passes:

* run each segment from zero and identity to obtain ``b_seg`` and ``A_seg``;
* scan those short affine summaries to recover each segment's incoming state;
* rerun all segments from the correct state, writing K5's snapshots and v_new.

The chunk map acts on the right: ``h' = h @ A + b`` for ``h[V,K]``.  Starting
the same recurrence at zero gives ``b``; starting it at identity with ``u=0``
gives ``A`` without materialising a dense operator for every chunk.
"""

import triton
import triton.language as tl

from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr

_gdn_segment_kernel_repr = make_kernel_repr(
    "_gdn_segment_kernel",
    [
        "BV",
        "DUAL_SUMMARY",
        "HAS_H_IN",
        "WRITE_OUTPUTS",
        "STORE_H_OUT",
        "STORE_FINAL",
        "USE_STATE_INDICES",
        "STATE_BF16",
    ],
)

_gdn_segment_scan_kernel_repr = make_kernel_repr(
    "_gdn_segment_scan_kernel",
    ["BV", "HAS_H0", "USE_STATE_INDICES"],
)


@triton.jit(repr=_gdn_segment_kernel_repr)
def _gdn_segment_kernel(
    k,
    u,
    w,
    g,
    h_in,
    h_out,
    a_out,
    h_snapshots,
    v_new,
    final_state,
    state_indices,
    seg_chunk_base,
    seg_nchunks,
    seg_tok_base,
    seg_tok_end,
    seg_seq,
    seg_is_last,
    T_FLAT,
    H: tl.constexpr,
    HG: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BV: tl.constexpr,
    DUAL_SUMMARY: tl.constexpr,
    HAS_H_IN: tl.constexpr,
    WRITE_OUTPUTS: tl.constexpr,
    STORE_H_OUT: tl.constexpr,
    STORE_FINAL: tl.constexpr,
    USE_STATE_INDICES: tl.constexpr,
    STATE_BF16: tl.constexpr,
):
    i_v = tl.program_id(0)
    i_sh = tl.program_id(1)
    i_seg, i_h = i_sh // H, i_sh % H

    chunk_base = tl.load(seg_chunk_base + i_seg).to(tl.int64)
    n_chunks = tl.load(seg_nchunks + i_seg)
    tok_base = tl.load(seg_tok_base + i_seg).to(tl.int64)
    tok_end = tl.load(seg_tok_end + i_seg).to(tl.int64)
    i_n = tl.load(seg_seq + i_seg).to(tl.int64)
    i_hg = i_h // (H // HG)

    o_v = i_v * BV + tl.arange(0, BV)
    m_v = o_v < V
    o_k1 = tl.arange(0, 64)
    o_k2 = 64 + tl.arange(0, 64)

    if DUAL_SUMMARY:
        b_h1 = tl.zeros([BV, 64], tl.float32)
        b_h2 = tl.zeros([BV, 64], tl.float32)
        a_h1 = tl.where(o_v[:, None] == o_k1[None, :], 1.0, 0.0)
        a_h2 = tl.where(o_v[:, None] == o_k2[None, :], 1.0, 0.0)
    elif HAS_H_IN:
        # h_in is one row per segment, already gathered by the wrapper when the
        # caller passes an indexed state pool.
        hb = (i_seg * H + i_h) * V * K + o_v[:, None] * K
        b_h1 = tl.load(h_in + hb + o_k1[None, :], mask=m_v[:, None], other=0.0).to(
            tl.float32
        )
        b_h2 = tl.load(h_in + hb + o_k2[None, :], mask=m_v[:, None], other=0.0).to(
            tl.float32
        )
    else:
        b_h1 = tl.zeros([BV, 64], tl.float32)
        b_h2 = tl.zeros([BV, 64], tl.float32)

    o_t = tl.arange(0, 64)
    for j in range(n_chunks):
        chunk_idx = chunk_base + j
        token = tok_base + j * 64 + o_t
        m_t = token < tok_end
        m_tv = m_t[:, None] & m_v[None, :]

        if WRITE_OUTPUTS:
            hs = (chunk_idx * H + i_h) * V * K + o_v[:, None] * K
            tl.store(
                h_snapshots + hs + o_k1[None, :],
                b_h1.to(h_snapshots.dtype.element_ty),
                mask=m_v[:, None],
            )
            tl.store(
                h_snapshots + hs + o_k2[None, :],
                b_h2.to(h_snapshots.dtype.element_ty),
                mask=m_v[:, None],
            )

        # v_new = u - w @ h.T
        wh = (i_h * T_FLAT + token)[:, None] * K
        b_w1 = tl.load(w + wh + o_k1[None, :], mask=m_t[:, None], other=0.0)
        b_w2 = tl.load(w + wh + o_k2[None, :], mask=m_t[:, None], other=0.0)
        b_v = tl.dot(b_w1, tl.trans(b_h1).to(b_w1.dtype))
        b_v += tl.dot(b_w2, tl.trans(b_h2).to(b_w2.dtype))
        if DUAL_SUMMARY:
            a_v = tl.dot(b_w1, tl.trans(a_h1).to(b_w1.dtype))
            a_v += tl.dot(b_w2, tl.trans(a_h2).to(b_w2.dtype))
            a_v = -a_v
        uv = (i_h * T_FLAT + token)[:, None] * V + o_v[None, :]
        b_v = tl.load(u + uv, mask=m_tv, other=0.0) - b_v

        if WRITE_OUTPUTS:
            vn = (i_h * T_FLAT + token)[:, None] * V + o_v[None, :]
            tl.store(v_new + vn, b_v.to(v_new.dtype.element_ty), mask=m_tv)

        # g is head-major and already log2(e)-scaled.
        last = tl.minimum(tok_base + (j + 1) * 64, tok_end) - 1
        g_base = i_h * T_FLAT
        g_last = tl.load(g + g_base + last)
        g_row = tl.load(g + g_base + token, mask=m_t, other=g_last)
        gate = tl.where(m_t, tl.math.exp2(g_last - g_row), 0.0)
        b_h1 *= tl.math.exp2(g_last)
        b_h2 *= tl.math.exp2(g_last)
        b_v = (b_v * gate[:, None]).to(k.dtype.element_ty)
        if DUAL_SUMMARY:
            a_h1 *= tl.math.exp2(g_last)
            a_h2 *= tl.math.exp2(g_last)
            a_v = (a_v * gate[:, None]).to(k.dtype.element_ty)

        # h += gated_v.T @ k
        kh = (token * HG + i_hg)[:, None] * K
        b_k1 = tl.load(k + kh + o_k1[None, :], mask=m_t[:, None], other=0.0)
        b_k2 = tl.load(k + kh + o_k2[None, :], mask=m_t[:, None], other=0.0)
        b_h1 += tl.dot(tl.trans(b_v), b_k1)
        b_h2 += tl.dot(tl.trans(b_v), b_k2)
        if DUAL_SUMMARY:
            a_h1 += tl.dot(tl.trans(a_v), b_k1)
            a_h2 += tl.dot(tl.trans(a_v), b_k2)

    if STORE_H_OUT:
        hb = (i_seg * H + i_h) * V * K + o_v[:, None] * K
        tl.store(
            h_out + hb + o_k1[None, :],
            b_h1.to(h_out.dtype.element_ty),
            mask=m_v[:, None],
        )
        tl.store(
            h_out + hb + o_k2[None, :],
            b_h2.to(h_out.dtype.element_ty),
            mask=m_v[:, None],
        )
        if DUAL_SUMMARY:
            tl.store(
                a_out + hb + o_k1[None, :],
                a_h1.to(a_out.dtype.element_ty),
                mask=m_v[:, None],
            )
            tl.store(
                a_out + hb + o_k2[None, :],
                a_h2.to(a_out.dtype.element_ty),
                mask=m_v[:, None],
            )

    if STORE_FINAL:  # noqa: SIM102
        if tl.load(seg_is_last + i_seg) == 1:
            state_n = tl.load(state_indices + i_n) if USE_STATE_INDICES else i_n
            fb = (state_n * H + i_h).to(tl.int64) * V * K + o_v[:, None] * K
            out1 = b_h1.to(tl.bfloat16) if STATE_BF16 else b_h1
            out2 = b_h2.to(tl.bfloat16) if STATE_BF16 else b_h2
            tl.store(final_state + fb + o_k1[None, :], out1, mask=m_v[:, None])
            tl.store(final_state + fb + o_k2[None, :], out2, mask=m_v[:, None])


@triton.jit(repr=_gdn_segment_scan_kernel_repr)
def _gdn_segment_scan_kernel(
    a_seg,
    b_seg,
    h_in,
    h0,
    state_indices,
    seq_seg_offsets,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BV: tl.constexpr,
    HAS_H0: tl.constexpr,
    USE_STATE_INDICES: tl.constexpr,
):
    i_v = tl.program_id(0)
    i_nh = tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H
    s0 = tl.load(seq_seg_offsets + i_n)
    s1 = tl.load(seq_seg_offsets + i_n + 1)

    o_v = i_v * BV + tl.arange(0, BV)
    m_v = o_v < V
    o_k1 = tl.arange(0, 64)
    o_k2 = 64 + tl.arange(0, 64)

    if HAS_H0:
        state_n = tl.load(state_indices + i_n) if USE_STATE_INDICES else i_n
        h0b = (state_n * H + i_h).to(tl.int64) * V * K + o_v[:, None] * K
        b_h1 = tl.load(h0 + h0b + o_k1[None, :], mask=m_v[:, None], other=0.0)
        b_h2 = tl.load(h0 + h0b + o_k2[None, :], mask=m_v[:, None], other=0.0)
        b_h1 = b_h1.to(tl.float32)
        b_h2 = b_h2.to(tl.float32)
    else:
        b_h1 = tl.zeros([BV, 64], tl.float32)
        b_h2 = tl.zeros([BV, 64], tl.float32)

    for s in range(s0, s1):
        sb = (s * H + i_h).to(tl.int64) * V * K + o_v[:, None] * K
        tl.store(h_in + sb + o_k1[None, :], b_h1, mask=m_v[:, None])
        tl.store(h_in + sb + o_k2[None, :], b_h2, mask=m_v[:, None])

        ab = (s * H + i_h) * K * K
        a11 = tl.load(a_seg + ab + o_k1[:, None] * K + o_k1[None, :])
        a12 = tl.load(a_seg + ab + o_k1[:, None] * K + o_k2[None, :])
        a21 = tl.load(a_seg + ab + o_k2[:, None] * K + o_k1[None, :])
        a22 = tl.load(a_seg + ab + o_k2[:, None] * K + o_k2[None, :])

        h1 = b_h1.to(tl.bfloat16)
        h2 = b_h2.to(tl.bfloat16)
        n1 = tl.dot(h1, a11) + tl.dot(h2, a21)
        n2 = tl.dot(h1, a12) + tl.dot(h2, a22)
        b_h1 = n1 + tl.load(b_seg + sb + o_k1[None, :], mask=m_v[:, None], other=0.0)
        b_h2 = n2 + tl.load(b_seg + sb + o_k2[None, :], mask=m_v[:, None], other=0.0)


__all__ = ["_gdn_segment_kernel", "_gdn_segment_scan_kernel"]
