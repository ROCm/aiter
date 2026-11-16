# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Gluon kernels of the batch-invariant sparse MLA forward (gfx950).

One (row, head tile) walks its key list in 64-key tiles. Split s owns tiles s,
s + NUM_SPLITS, ... and keeps an FP32 online softmax. The splits are merged in
the order 0 .. NUM_SPLITS - 1. The split kernel + combine kernel and the fused
kernel run the same helpers, so they give the same bits.
"""

from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.runtime.jit import constexpr_function

from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr

_LOG2E = gl.constexpr(1.4426950408889634)
_LN2 = gl.constexpr(0.6931471805599453)


@constexpr_function
def _mfma_layout(num_warps):
    # bf16 16x16x32; warps tile the key (QK) and latent (PV) columns.
    return gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=[16, 16, 32],
        transposed=True,
        warps_per_cta=[1, num_warps],
    )


@constexpr_function
def _gather_layout(num_warps, vec, cols):
    # vec elements (16 B) per lane along a row, the rest of the wave on rows.
    tw = min(64, cols // vec)
    return gl.BlockedLayout(
        size_per_thread=[1, vec],
        threads_per_warp=[64 // tw, tw],
        warps_per_cta=[num_warps, 1],
        order=[1, 0],
    )


@constexpr_function
def _kv_shared(rows, cols):
    # 16 B pad per row spreads the transposed K reads over the banks.
    return gl.PaddedSharedLayout.with_identity_for([[cols, 8]], [rows, cols], [1, 0])


@gluon.jit
def _load_q(q_ptr, q_base, q_s1, h0, num_heads, BLOCK_H, D, R, NUM_WARPS):
    """q tile [BLOCK_H, D] and its rope [BLOCK_H, R] in the QK operand layout."""
    mfma: gl.constexpr = _mfma_layout(NUM_WARPS)
    dot_a: gl.constexpr = gl.DotOperandLayout(0, mfma, 8)
    lq: gl.constexpr = _gather_layout(NUM_WARPS, 8, D)
    lr: gl.constexpr = _gather_layout(NUM_WARPS, 8, R)
    h = h0 + gl.arange(0, BLOCK_H, layout=gl.SliceLayout(1, lq))
    d = gl.arange(0, D, layout=gl.SliceLayout(0, lq))
    q = gl.load(
        q_ptr + q_base + h.to(gl.int64)[:, None] * q_s1 + d[None, :],
        mask=(h < num_heads)[:, None],
        other=0.0,
    )
    hr = h0 + gl.arange(0, BLOCK_H, layout=gl.SliceLayout(1, lr))
    dr = gl.arange(0, R, layout=gl.SliceLayout(0, lr))
    qr = gl.load(
        q_ptr + q_base + hr.to(gl.int64)[:, None] * q_s1 + D + dr[None, :],
        mask=(hr < num_heads)[:, None],
        other=0.0,
    )
    return gl.convert_layout(q, dot_a), gl.convert_layout(qr, dot_a)


@gluon.jit
def _stage_tile(
    kv_ptr,
    kv_bf16_ptr,
    kv_f32_ptr,
    idx_ptr,
    beg,
    k0,
    n,
    block_size,
    page_stride,
    kv_smem,
    rope_smem,
    FP8: gl.constexpr,
    HAS_INVALID: gl.constexpr,
    KEY_TILE: gl.constexpr,
    D: gl.constexpr,
    R: gl.constexpr,
    NUM_WARPS: gl.constexpr,
):
    """Gather key tile [k0, k0 + KEY_TILE) of the row into LDS as bf16 (latent and
    rope). Keys past n (and -1 slots with HAS_INVALID) load zeros."""
    G: gl.constexpr = 128  # latent columns per chunk (one fp8_dsv32_mla scale group)
    VEC: gl.constexpr = 16 if FP8 else 8
    lg: gl.constexpr = _gather_layout(NUM_WARPS, VEC, G)
    lr: gl.constexpr = _gather_layout(NUM_WARPS, 8, R)
    k = k0 + gl.arange(0, KEY_TILE, layout=gl.SliceLayout(1, lg))
    valid = k < n
    slot = gl.load(idx_ptr + beg + k, mask=valid, other=0)
    if HAS_INVALID:
        valid = valid & (slot >= 0)
        slot = gl.where(valid, slot, 0)
    # Row start: bytes (fp8_dsv32_mla) or bf16 elements (bf16), 64-bit.
    ROW: gl.constexpr = (D + 4 * (D // 128) + 2 * R) if FP8 else (D + R)
    row = (slot // block_size).to(gl.int64) * page_stride + (slot % block_size).to(
        gl.int64
    ) * ROW
    cols = gl.arange(0, G, layout=gl.SliceLayout(0, lg))
    m2 = valid[:, None]
    for c in gl.static_range(D // G):
        if FP8:
            x = gl.load(kv_ptr + row[:, None] + (c * G + cols)[None, :], mask=m2, other=0)
            sc = gl.load(kv_f32_ptr + (row // 4 + D // 4 + c), mask=valid, other=0.0)
            val = (x.to(gl.float8e4nv, bitcast=True).to(gl.float32) * sc[:, None]).to(
                gl.bfloat16
            )
        else:
            val = gl.load(
                kv_bf16_ptr + row[:, None] + (c * G + cols)[None, :], mask=m2, other=0.0
            )
        kv_smem.slice(c * G, G, dim=1).store(val)
    rowr = gl.convert_layout(row, gl.SliceLayout(1, lr))
    validr = gl.convert_layout(valid, gl.SliceLayout(1, lr))
    if FP8:
        rope0 = rowr // 2 + (D + 4 * (D // 128)) // 2
    else:
        rope0 = rowr + D
    dr = gl.arange(0, R, layout=gl.SliceLayout(0, lr))
    rope = gl.load(
        kv_bf16_ptr + rope0[:, None] + dr[None, :], mask=validr[:, None], other=0.0
    )
    rope_smem.store(rope)
    return valid


@gluon.jit
def _split_state(
    q_dot,
    qr_dot,
    kv_ptr,
    kv_bf16_ptr,
    kv_f32_ptr,
    idx_ptr,
    beg,
    n,
    s,
    scale_log2,
    block_size,
    page_stride,
    kv_smem,
    rope_smem,
    FP8: gl.constexpr,
    HAS_INVALID: gl.constexpr,
    BLOCK_H: gl.constexpr,
    KEY_TILE: gl.constexpr,
    D: gl.constexpr,
    R: gl.constexpr,
    NUM_SPLITS: gl.constexpr,
    NUM_WARPS: gl.constexpr,
):
    """(m, l, acc) of split s: FP32 online softmax over tiles s, s + NUM_SPLITS, ...
    in increasing order. m is in the log2 domain (scores * scale * log2 e)."""
    mfma: gl.constexpr = _mfma_layout(NUM_WARPS)
    dot_a: gl.constexpr = gl.DotOperandLayout(0, mfma, 8)
    dot_b: gl.constexpr = gl.DotOperandLayout(1, mfma, 8)
    row_l: gl.constexpr = gl.SliceLayout(1, mfma)
    col_l: gl.constexpr = gl.SliceLayout(0, mfma)
    m = gl.full([BLOCK_H], float("-inf"), gl.float32, layout=row_l)
    l = gl.zeros([BLOCK_H], gl.float32, layout=row_l)
    acc = gl.zeros([BLOCK_H, D], gl.float32, layout=mfma)
    n_tiles = (n + KEY_TILE - 1) // KEY_TILE
    for t in range(s, n_tiles, NUM_SPLITS):
        k0 = t * KEY_TILE
        valid = _stage_tile(
            kv_ptr,
            kv_bf16_ptr,
            kv_f32_ptr,
            idx_ptr,
            beg,
            k0,
            n,
            block_size,
            page_stride,
            kv_smem,
            rope_smem,
            FP8,
            HAS_INVALID,
            KEY_TILE,
            D,
            R,
            NUM_WARPS,
        )
        sc = gl.zeros([BLOCK_H, KEY_TILE], gl.float32, layout=mfma)
        sc = gl.amd.cdna4.mfma(q_dot, kv_smem.permute([1, 0]).load(dot_b), sc)
        sc = gl.amd.cdna4.mfma(qr_dot, rope_smem.permute([1, 0]).load(dot_b), sc)
        if HAS_INVALID:
            keep = gl.convert_layout(valid, col_l)
        else:
            keep = k0 + gl.arange(0, KEY_TILE, layout=col_l) < n
        sc = gl.where(keep[None, :], sc, float("-inf"))
        # max commutes with the positive scale, so scale the row max only.
        m_new = gl.maximum(m, gl.max(sc, axis=1) * scale_log2)
        m_use = gl.where(m_new > float("-inf"), m_new, 0.0)
        p = gl.exp2(sc * scale_log2 - m_use[:, None])
        alpha = gl.exp2(m - m_use)
        l = l * alpha + gl.sum(p, axis=1)
        v = kv_smem.load(dot_b)
        p_dot = gl.convert_layout(p.to(gl.bfloat16), dot_a)
        acc = gl.amd.cdna4.mfma(p_dot, v, acc * alpha[:, None])
        m = m_new
    return m, l, acc


@gluon.jit
def _merge(M, L, O, m, l, acc):
    """Fold one split state into the running (M, L, O). Empty splits (l == 0) are
    skipped; merging into (-inf, 0, 0) returns the split state exactly."""
    take = l > 0.0
    M_new = gl.where(take, gl.maximum(M, m), M)
    a = gl.where(take, gl.exp2(M - M_new), 1.0)
    b = gl.where(take, gl.exp2(m - M_new), 0.0)
    L = L * a + l * b
    O = gl.where(take[:, None], O * a[:, None] + acc * b[:, None], O)
    return M_new, L, O


@gluon.jit
def _finish(
    M,
    L,
    O,
    sink_ptr,
    out_ptr,
    out_base,
    out_s1,
    lse_ptr,
    lse_base,
    h0,
    num_heads,
    HAS_SINK: gl.constexpr,
    HAS_LSE: gl.constexpr,
    BLOCK_H: gl.constexpr,
    D: gl.constexpr,
    NUM_WARPS: gl.constexpr,
):
    """Sink as a last virtual split, then out = O / L (0 for an empty row)."""
    mfma: gl.constexpr = _mfma_layout(NUM_WARPS)
    row_l: gl.constexpr = gl.SliceLayout(1, mfma)
    h = h0 + gl.arange(0, BLOCK_H, layout=row_l)
    hm = h < num_heads
    if HAS_SINK:
        sk = gl.load(sink_ptr + h, mask=hm, other=0.0) * _LOG2E
        M, L, O = _merge(
            M,
            L,
            O,
            sk,
            gl.full([BLOCK_H], 1.0, gl.float32, layout=row_l),
            gl.zeros([BLOCK_H, D], gl.float32, layout=mfma),
        )
    nz = L > 0.0
    o = gl.where(nz[:, None], O / gl.where(nz, L, 1.0)[:, None], 0.0)
    d = gl.arange(0, D, layout=gl.SliceLayout(0, mfma))
    gl.store(
        out_ptr + out_base + h.to(gl.int64)[:, None] * out_s1 + d[None, :],
        o.to(gl.bfloat16),
        mask=hm[:, None],
    )
    if HAS_LSE:
        lse = gl.where(nz, (M + gl.log2(gl.where(nz, L, 1.0))) * _LN2, float("-inf"))
        gl.store(lse_ptr + lse_base + h, lse, mask=hm)


_split_repr = make_kernel_repr(
    "_sparse_mla_bi_split_kernel",
    ["FP8", "HAS_INVALID", "BLOCK_H", "KEY_TILE", "NUM_SPLITS"],
)


@gluon.jit(repr=_split_repr)
def _sparse_mla_bi_split_kernel(
    q_ptr,
    kv_ptr,
    kv_bf16_ptr,
    kv_f32_ptr,
    idx_ptr,
    indptr_ptr,
    part_acc_ptr,
    part_m_ptr,
    part_l_ptr,
    scale_log2,
    q_s0,
    q_s1,
    block_size,
    page_stride,
    num_heads,
    FP8: gl.constexpr,
    HAS_INVALID: gl.constexpr,
    BLOCK_H: gl.constexpr,
    KEY_TILE: gl.constexpr,
    D: gl.constexpr,
    R: gl.constexpr,
    NUM_SPLITS: gl.constexpr,
):
    """Grid (rows, NUM_SPLITS, head tiles): one split state -> FP32 partials
    [rows, NUM_SPLITS, H, D] and m, l [rows, NUM_SPLITS, H]."""
    NUM_WARPS: gl.constexpr = gl.num_warps()
    r = gl.program_id(0)
    s = gl.program_id(1)
    h0 = gl.program_id(2) * BLOCK_H
    beg = gl.load(indptr_ptr + r)
    n = gl.load(indptr_ptr + r + 1) - beg
    q_dot, qr_dot = _load_q(
        q_ptr, r.to(gl.int64) * q_s0, q_s1, h0, num_heads, BLOCK_H, D, R, NUM_WARPS
    )
    kv_smem = gl.allocate_shared_memory(
        gl.bfloat16, [KEY_TILE, D], _kv_shared(KEY_TILE, D)
    )
    rope_smem = gl.allocate_shared_memory(
        gl.bfloat16, [KEY_TILE, R], _kv_shared(KEY_TILE, R)
    )
    m, l, acc = _split_state(
        q_dot,
        qr_dot,
        kv_ptr,
        kv_bf16_ptr,
        kv_f32_ptr,
        idx_ptr,
        beg,
        n,
        s,
        scale_log2,
        block_size,
        page_stride,
        kv_smem,
        rope_smem,
        FP8,
        HAS_INVALID,
        BLOCK_H,
        KEY_TILE,
        D,
        R,
        NUM_SPLITS,
        NUM_WARPS,
    )
    mfma: gl.constexpr = _mfma_layout(NUM_WARPS)
    h = h0 + gl.arange(0, BLOCK_H, layout=gl.SliceLayout(1, mfma))
    hm = h < num_heads
    base = (r.to(gl.int64) * NUM_SPLITS + s) * num_heads
    gl.store(part_m_ptr + base + h, m, mask=hm)
    gl.store(part_l_ptr + base + h, l, mask=hm)
    d = gl.arange(0, D, layout=gl.SliceLayout(0, mfma))
    gl.store(
        part_acc_ptr + (base + h.to(gl.int64))[:, None] * D + d[None, :],
        acc,
        mask=hm[:, None],
    )


_combine_repr = make_kernel_repr(
    "_sparse_mla_bi_combine_kernel", ["HAS_SINK", "HAS_LSE", "BLOCK_H", "NUM_SPLITS"]
)


@gluon.jit(repr=_combine_repr)
def _sparse_mla_bi_combine_kernel(
    part_acc_ptr,
    part_m_ptr,
    part_l_ptr,
    sink_ptr,
    out_ptr,
    lse_ptr,
    out_s0,
    out_s1,
    num_heads,
    HAS_SINK: gl.constexpr,
    HAS_LSE: gl.constexpr,
    BLOCK_H: gl.constexpr,
    D: gl.constexpr,
    NUM_SPLITS: gl.constexpr,
):
    """Grid (rows, head tiles): merge the split partials in order 0 .. NUM_SPLITS - 1."""
    NUM_WARPS: gl.constexpr = gl.num_warps()
    mfma: gl.constexpr = _mfma_layout(NUM_WARPS)
    row_l: gl.constexpr = gl.SliceLayout(1, mfma)
    r = gl.program_id(0)
    h0 = gl.program_id(1) * BLOCK_H
    h = h0 + gl.arange(0, BLOCK_H, layout=row_l)
    hm = h < num_heads
    d = gl.arange(0, D, layout=gl.SliceLayout(0, mfma))
    M = gl.full([BLOCK_H], float("-inf"), gl.float32, layout=row_l)
    L = gl.zeros([BLOCK_H], gl.float32, layout=row_l)
    O = gl.zeros([BLOCK_H, D], gl.float32, layout=mfma)
    for s in range(NUM_SPLITS):
        base = (r.to(gl.int64) * NUM_SPLITS + s) * num_heads
        m = gl.load(part_m_ptr + base + h, mask=hm, other=float("-inf"))
        l = gl.load(part_l_ptr + base + h, mask=hm, other=0.0)
        acc = gl.load(
            part_acc_ptr + (base + h.to(gl.int64))[:, None] * D + d[None, :],
            mask=hm[:, None],
            other=0.0,
        )
        M, L, O = _merge(M, L, O, m, l, acc)
    _finish(
        M,
        L,
        O,
        sink_ptr,
        out_ptr,
        r.to(gl.int64) * out_s0,
        out_s1,
        lse_ptr,
        r.to(gl.int64) * num_heads,
        h0,
        num_heads,
        HAS_SINK,
        HAS_LSE,
        BLOCK_H,
        D,
        NUM_WARPS,
    )


_fused_repr = make_kernel_repr(
    "_sparse_mla_bi_fused_kernel",
    ["FP8", "HAS_INVALID", "HAS_SINK", "HAS_LSE", "BLOCK_H", "KEY_TILE", "NUM_SPLITS"],
)


@gluon.jit(repr=_fused_repr)
def _sparse_mla_bi_fused_kernel(
    q_ptr,
    kv_ptr,
    kv_bf16_ptr,
    kv_f32_ptr,
    idx_ptr,
    indptr_ptr,
    sink_ptr,
    out_ptr,
    lse_ptr,
    scale_log2,
    q_s0,
    q_s1,
    out_s0,
    out_s1,
    block_size,
    page_stride,
    num_heads,
    FP8: gl.constexpr,
    HAS_INVALID: gl.constexpr,
    HAS_SINK: gl.constexpr,
    HAS_LSE: gl.constexpr,
    BLOCK_H: gl.constexpr,
    KEY_TILE: gl.constexpr,
    D: gl.constexpr,
    R: gl.constexpr,
    NUM_SPLITS: gl.constexpr,
):
    """Grid (rows, head tiles): the splits one after the other, merged in order."""
    NUM_WARPS: gl.constexpr = gl.num_warps()
    mfma: gl.constexpr = _mfma_layout(NUM_WARPS)
    row_l: gl.constexpr = gl.SliceLayout(1, mfma)
    r = gl.program_id(0)
    h0 = gl.program_id(1) * BLOCK_H
    beg = gl.load(indptr_ptr + r)
    n = gl.load(indptr_ptr + r + 1) - beg
    q_dot, qr_dot = _load_q(
        q_ptr, r.to(gl.int64) * q_s0, q_s1, h0, num_heads, BLOCK_H, D, R, NUM_WARPS
    )
    kv_smem = gl.allocate_shared_memory(
        gl.bfloat16, [KEY_TILE, D], _kv_shared(KEY_TILE, D)
    )
    rope_smem = gl.allocate_shared_memory(
        gl.bfloat16, [KEY_TILE, R], _kv_shared(KEY_TILE, R)
    )
    M = gl.full([BLOCK_H], float("-inf"), gl.float32, layout=row_l)
    L = gl.zeros([BLOCK_H], gl.float32, layout=row_l)
    O = gl.zeros([BLOCK_H, D], gl.float32, layout=mfma)
    for s in range(NUM_SPLITS):
        m, l, acc = _split_state(
            q_dot,
            qr_dot,
            kv_ptr,
            kv_bf16_ptr,
            kv_f32_ptr,
            idx_ptr,
            beg,
            n,
            s,
            scale_log2,
            block_size,
            page_stride,
            kv_smem,
            rope_smem,
            FP8,
            HAS_INVALID,
            BLOCK_H,
            KEY_TILE,
            D,
            R,
            NUM_SPLITS,
            NUM_WARPS,
        )
        M, L, O = _merge(M, L, O, m, l, acc)
    _finish(
        M,
        L,
        O,
        sink_ptr,
        out_ptr,
        r.to(gl.int64) * out_s0,
        out_s1,
        lse_ptr,
        r.to(gl.int64) * num_heads,
        h0,
        num_heads,
        HAS_SINK,
        HAS_LSE,
        BLOCK_H,
        D,
        NUM_WARPS,
    )
