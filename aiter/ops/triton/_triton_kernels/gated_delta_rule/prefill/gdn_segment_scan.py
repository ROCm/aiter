# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Context-parallel GDN K5 using affine segment summaries.

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

from __future__ import annotations

import functools
import os

import torch
import triton
import triton.language as tl

_K = 128
_V = 128
_BT = 64
_DEFAULT_CHUNKS_PER_SEGMENT = 16


@triton.jit
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
    TOTAL_CHUNKS: tl.constexpr,
    T_FLAT: tl.constexpr,
    H: tl.constexpr,
    HG: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BV: tl.constexpr,
    INIT_IDENTITY: tl.constexpr,
    DUAL_SUMMARY: tl.constexpr,
    HAS_H_IN: tl.constexpr,
    HAS_U: tl.constexpr,
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
    elif INIT_IDENTITY:
        b_h1 = tl.where(o_v[:, None] == o_k1[None, :], 1.0, 0.0)
        b_h2 = tl.where(o_v[:, None] == o_k2[None, :], 1.0, 0.0)
    elif HAS_H_IN:
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
        if HAS_U:
            uv = (i_h * T_FLAT + token)[:, None] * V + o_v[None, :]
            b_v = tl.load(u + uv, mask=m_tv, other=0.0) - b_v
        else:
            b_v = -b_v

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


@triton.jit
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


@functools.lru_cache(maxsize=64)
def _build_segments(
    seq_lens: tuple[int, ...],
    chunks_per_segment: int,
    device: torch.device,
):
    chunk_base: list[int] = []
    nchunks: list[int] = []
    tok_base: list[int] = []
    tok_end: list[int] = []
    seq_id: list[int] = []
    is_last: list[int] = []
    seq_seg_offsets = [0]
    global_chunk = 0
    global_token = 0
    for i, length in enumerate(seq_lens):
        n_chunks = triton.cdiv(length, _BT)
        n_segments = max(1, triton.cdiv(n_chunks, chunks_per_segment))
        for s in range(n_segments):
            c0 = s * chunks_per_segment
            count = min(chunks_per_segment, n_chunks - c0)
            chunk_base.append(global_chunk + c0)
            nchunks.append(count)
            tok_base.append(global_token + c0 * _BT)
            tok_end.append(
                min(global_token + (c0 + count) * _BT, global_token + length)
            )
            seq_id.append(i)
            is_last.append(1 if s == n_segments - 1 else 0)
        seq_seg_offsets.append(len(chunk_base))
        global_chunk += n_chunks
        global_token += length
    desc = torch.tensor(
        [chunk_base, nchunks, tok_base, tok_end, seq_id, is_last],
        dtype=torch.int32,
        device=device,
    )
    offsets = torch.tensor(seq_seg_offsets, dtype=torch.int32, device=device)
    return (
        desc,
        offsets,
        len(chunk_base),
        max(
            seq_seg_offsets[i + 1] - seq_seg_offsets[i]
            for i in range(len(seq_seg_offsets) - 1)
        ),
    )


def gdn_segment_scan_fwd(
    *,
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor,
    initial_state: torch.Tensor | None,
    output_final_state: bool,
    seq_lens: tuple[int, ...],
    state_indices: torch.Tensor | None = None,
    inplace_final_state: bool = False,
    snapshot_dtype: torch.dtype = torch.bfloat16,
    state_dtype: torch.dtype = torch.float32,
    chunks_per_segment: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Run segmented K5 for the narrow production GDN shape."""
    B, _T, HG, K = k.shape
    H, T_flat, V = w.shape[1], w.shape[2], u.shape[-1]
    if (B, K, V) != (1, _K, _V):
        raise ValueError(
            f"GDN segment scan requires B=1,K=V=128; got B={B},K={K},V={V}."
        )
    if g.shape != (B, H, T_flat):
        raise ValueError(f"GDN segment scan needs head-major g; got {tuple(g.shape)}.")
    if sum(seq_lens) != T_flat:
        raise ValueError(f"seq_lens sum to {sum(seq_lens)}, expected {T_flat}.")

    max_chunks = max(triton.cdiv(length, _BT) for length in seq_lens)
    if chunks_per_segment is None:
        override = os.getenv("AITER_GDN_K5_SEGMENT_CHUNKS", "").strip()
        chunks_per_segment = int(override) if override else _DEFAULT_CHUNKS_PER_SEGMENT
    chunks_per_segment = min(max_chunks, max(1, chunks_per_segment))
    tile_v = int(os.getenv("AITER_GDN_K5_SEGMENT_BV", "64"))
    if tile_v not in (16, 32, 64):
        raise ValueError(
            f"AITER_GDN_K5_SEGMENT_BV must be 16, 32, or 64, got {tile_v}."
        )

    desc, seq_seg_offsets, num_segments, max_segments = _build_segments(
        seq_lens, chunks_per_segment, k.device
    )
    seg_chunk_base, seg_nchunks, seg_tok_base, seg_tok_end, seg_seq, seg_is_last = desc
    total_chunks = sum(triton.cdiv(length, _BT) for length in seq_lens)

    # The regular K5 output contract.
    snapshots = torch.empty(
        B, total_chunks, H, V, K, dtype=snapshot_dtype, device=k.device
    )
    v_new = torch.empty(B, H, T_flat, V, dtype=u.dtype, device=k.device)
    if output_final_state:
        final_state = (
            initial_state
            if inplace_final_state
            else torch.empty(len(seq_lens), H, V, K, dtype=state_dtype, device=k.device)
        )
    else:
        final_state = None

    # Segmentation has no benefit when the sequence is already short.
    if max_segments == 1:
        h_in = initial_state
    else:
        b_seg = torch.empty(num_segments, H, V, K, dtype=torch.float32, device=k.device)
        a_seg = torch.empty(
            num_segments, H, V, K, dtype=torch.bfloat16, device=k.device
        )
        common = {
            "k": k,
            "u": u,
            "w": w,
            "g": g,
            "h_in": None,
            "a_out": None,
            "h_snapshots": None,
            "v_new": None,
            "final_state": None,
            "state_indices": state_indices,
            "seg_chunk_base": seg_chunk_base,
            "seg_nchunks": seg_nchunks,
            "seg_tok_base": seg_tok_base,
            "seg_tok_end": seg_tok_end,
            "seg_seq": seg_seq,
            "seg_is_last": seg_is_last,
            "TOTAL_CHUNKS": total_chunks,
            "T_FLAT": T_flat,
            "H": H,
            "HG": HG,
            "K": K,
            "V": V,
            "BV": tile_v,
            "HAS_H_IN": False,
            "WRITE_OUTPUTS": False,
            "STORE_H_OUT": True,
            "STORE_FINAL": False,
            "USE_STATE_INDICES": state_indices is not None,
            "STATE_BF16": state_dtype is torch.bfloat16,
            "num_warps": 4,
            "num_stages": 1,
        }
        grid = (triton.cdiv(V, tile_v), num_segments * H)
        fuse_summaries = os.getenv("AITER_GDN_K5_FUSE_SUMMARIES", "1") == "1"
        if fuse_summaries:
            _gdn_segment_kernel[grid](
                h_out=b_seg,
                a_out=a_seg,
                INIT_IDENTITY=False,
                DUAL_SUMMARY=True,
                HAS_U=True,
                **{key: value for key, value in common.items() if key != "a_out"},
            )
        else:
            _gdn_segment_kernel[grid](
                h_out=b_seg,
                INIT_IDENTITY=False,
                DUAL_SUMMARY=False,
                HAS_U=True,
                **common,
            )
            _gdn_segment_kernel[grid](
                h_out=a_seg,
                INIT_IDENTITY=True,
                DUAL_SUMMARY=False,
                HAS_U=False,
                **common,
            )
        h_in = torch.empty(num_segments, H, V, K, dtype=torch.float32, device=k.device)
        scan_warps = int(os.getenv("AITER_GDN_K5_SCAN_WARPS", "4"))
        scan_bv = int(os.getenv("AITER_GDN_K5_SCAN_BV", "16"))
        _gdn_segment_scan_kernel[(triton.cdiv(V, scan_bv), len(seq_lens) * H)](
            a_seg=a_seg,
            b_seg=b_seg,
            h_in=h_in,
            h0=initial_state,
            state_indices=state_indices,
            seq_seg_offsets=seq_seg_offsets,
            H=H,
            K=K,
            V=V,
            BV=scan_bv,
            HAS_H0=initial_state is not None,
            USE_STATE_INDICES=state_indices is not None,
            num_warps=scan_warps,
            num_stages=1,
        )

    _gdn_segment_kernel[(triton.cdiv(V, tile_v), num_segments * H)](
        k=k,
        u=u,
        w=w,
        g=g,
        h_in=h_in,
        h_out=None,
        a_out=None,
        h_snapshots=snapshots,
        v_new=v_new,
        final_state=final_state,
        state_indices=state_indices,
        seg_chunk_base=seg_chunk_base,
        seg_nchunks=seg_nchunks,
        seg_tok_base=seg_tok_base,
        seg_tok_end=seg_tok_end,
        seg_seq=seg_seq,
        seg_is_last=seg_is_last,
        TOTAL_CHUNKS=total_chunks,
        T_FLAT=T_flat,
        H=H,
        HG=HG,
        K=K,
        V=V,
        BV=tile_v,
        INIT_IDENTITY=False,
        DUAL_SUMMARY=False,
        HAS_H_IN=h_in is not None,
        HAS_U=True,
        WRITE_OUTPUTS=True,
        STORE_H_OUT=False,
        STORE_FINAL=output_final_state,
        USE_STATE_INDICES=state_indices is not None,
        STATE_BF16=state_dtype is torch.bfloat16,
        num_warps=4,
        num_stages=1,
    )
    return snapshots, v_new, final_state


__all__ = ["gdn_segment_scan_fwd"]
