# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Gluon (gfx950 / CDNA4) single-pass sparse prefill over one KV pool.

The attention is absorbed MLA without RoPE: K = V = the 512-wide row of the
pool. For query row t, its slots ``idx[indptr[t]:indptr[t + 1]]`` and head h:

    O_th = sum_j p_tj kv[slot_j] / l_th,  p_tj = 2^(c s_tj - m_th),  s_tj = q_th . kv[slot_j]

with c = fp32(scale) * fp32(log2(e)). The softmax is online over BLOCK_K-key
tiles, m starts at -FLT_MAX, p is rounded to bf16 for the PV product and l sums
the fp32 p. The optional sink is one more key with value zero, added after the last
tile. A row with no slots writes zeros.

One program computes one query row for H heads, the last rows first; with
HEAD_GROUPS > 1, HEAD_GROUPS programs share a row, H heads each. Each
tile is copied from the pool straight into one LDS buffer (128 bits per lane,
64-bit offsets), then feeds the QK and PV MFMAs.

The launcher is in ``aiter.ops.triton.attention.pa_prefill_sparse``. This module
does not import torch.
"""

from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr


@gluon.jit
def _load_slots(row, cols, last):
    # Columns past the last slot of the row read that slot again; their scores
    # are masked. A mask here compiles to one branch per index.
    return gl.load(row + gl.minimum(cols, last))


@gluon.jit
def _load_q(q_ptr, r, stride_qt, stride_qh, H: gl.constexpr, dot_a: gl.constexpr):
    DC: gl.constexpr = 128
    h = gl.arange(0, H, layout=gl.SliceLayout(1, dot_a))
    d = gl.arange(0, DC, layout=gl.SliceLayout(0, dot_a))
    p = (
        q_ptr
        + r.to(gl.int64) * stride_qt
        + h[:, None].to(gl.int64) * stride_qh
        + d[None, :]
    )
    return gl.load(p), gl.load(p + DC), gl.load(p + 2 * DC), gl.load(p + 3 * DC)


@gluon.jit
def _k_chunk(tile, c: gl.constexpr, dot_b: gl.constexpr):
    # Relaxed read: the wait_group and the barrier before it order the copy.
    return gl.amd.cdna4.async_copy.load_shared_relaxed(
        tile.slice(c * 128, 128, dim=1).permute([1, 0]), dot_b
    )


@gluon.jit
def _v_chunk(tile, c: gl.constexpr, dot_b: gl.constexpr):
    return gl.amd.cdna4.async_copy.load_shared_relaxed(
        tile.slice(c * 128, 128, dim=1), dot_b
    )


@gluon.jit
def _row_and_group(HEAD_GROUPS: gl.constexpr, NUM_XCDS: gl.constexpr):
    # A row's head groups run at program ids L, L + NUM_XCDS, ..: on one XCD,
    # so the later groups hit the first one's gathers in L2. The last
    # T % NUM_XCDS rows run their groups at adjacent ids. Rows run last first.
    L = gl.program_id(0)
    T = gl.num_programs(0) // HEAD_GROUPS
    full = T // NUM_XCDS * NUM_XCDS * HEAD_GROUPS
    if L < full:
        row = L // (NUM_XCDS * HEAD_GROUPS) * NUM_XCDS + L % NUM_XCDS
        hg = L // NUM_XCDS % HEAD_GROUPS
    else:
        row = full // HEAD_GROUPS + (L - full) // HEAD_GROUPS
        hg = (L - full) % HEAD_GROUPS
    return T - 1 - row, hg


_pa_prefill_sparse_repr = make_kernel_repr(
    "_pa_prefill_sparse",
    ["H", "HEAD_GROUPS", "BLOCK_K", "HAS_SINK"],
)


@gluon.jit(repr=_pa_prefill_sparse_repr)
def _pa_prefill_sparse(
    q_ptr,  # [T, HEAD_GROUPS * H, 512] bf16
    kv_ptr,  # [S, 512] bf16
    idx_ptr,  # [nnz] int32, slots into kv
    indptr_ptr,  # [T + 1] int32
    sink_ptr,  # [HEAD_GROUPS * H] fp32, read only if HAS_SINK
    o_ptr,  # [T, HEAD_GROUPS * H, 512] bf16
    stride_qt,
    stride_qh,
    stride_kv,
    stride_ot,
    stride_oh,
    qk_scale,  # fp32(scale) * fp32(log2(e))
    H: gl.constexpr,
    HEAD_GROUPS: gl.constexpr,
    NUM_XCDS: gl.constexpr,
    BLOCK_K: gl.constexpr,
    HAS_SINK: gl.constexpr,
):
    D: gl.constexpr = 512
    DC: gl.constexpr = 128
    NUM_WARPS: gl.constexpr = gl.num_warps()
    # Warps split the heads in 16-row slices and the keys with the rest.
    M_WARPS: gl.constexpr = min(H // 16, NUM_WARPS)
    N_WARPS: gl.constexpr = NUM_WARPS // M_WARPS
    gl.static_assert(
        BLOCK_K % (16 * N_WARPS) == 0, "BLOCK_K must split in 16-key slices per warp"
    )
    mfma_l: gl.constexpr = gl.amd.cdna4.AMDMFMALayout(
        version=4,
        instr_shape=[16, 16, 32],
        transposed=True,
        warps_per_cta=[M_WARPS, N_WARPS],
    )
    dot_a: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mfma_l, k_width=8)
    dot_b: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=mfma_l, k_width=8)
    # One 1 KB KV row per warp instruction: 64 lanes x 8 bf16.
    gather_l: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 8],
        threads_per_warp=[1, 64],
        warps_per_cta=[NUM_WARPS, 1],
        order=[1, 0],
    )
    # 16 B of padding after each row: the 16 keys of a K-operand read start
    # 4 banks apart.
    kv_sl: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[D, 8]], [BLOCK_K, D], [1, 0]
    )

    tile = gl.allocate_shared_memory(gl.bfloat16, [BLOCK_K, D], kv_sl)
    if HEAD_GROUPS == 1:
        r = gl.num_programs(0) - 1 - gl.program_id(0)
    else:
        r, hg = _row_and_group(HEAD_GROUPS, NUM_XCDS)
        q_ptr += hg * H * stride_qh
        o_ptr += hg * H * stride_oh
        if HAS_SINK:
            sink_ptr += hg * H
    beg = gl.load(indptr_ptr + r)
    n = gl.load(indptr_ptr + r + 1) - beg
    last = gl.maximum(n - 1, 0)
    # An empty row runs no tile. Its one prologue read is from indptr, which is
    # always valid, not from idx.
    row = gl.where(n > 0, idx_ptr + beg, indptr_ptr)
    n_tiles = (n + BLOCK_K - 1) // BLOCK_K
    cols_g = gl.arange(0, BLOCK_K, layout=gl.SliceLayout(1, gather_l))
    d_g = gl.arange(0, D, layout=gl.SliceLayout(0, gather_l))
    q0, q1, q2, q3 = _load_q(q_ptr, r, stride_qt, stride_qh, H, dot_a)
    a0 = gl.zeros([H, DC], gl.float32, layout=mfma_l)
    a1 = a0
    a2 = a0
    a3 = a0
    m = gl.full(
        [H], -3.4028234663852886e38, gl.float32, layout=gl.SliceLayout(1, mfma_l)
    )
    l = gl.zeros([H], gl.float32, layout=gl.SliceLayout(1, mfma_l))
    slots = _load_slots(row, cols_g, last)
    for j in range(n_tiles):
        s_off = slots.to(gl.int64) * stride_kv
        gl.amd.cdna4.async_copy.global_load_to_shared(
            tile, kv_ptr + s_off[:, None] + d_g[None, :]
        )
        gl.amd.cdna4.async_copy.commit_group()
        gl.amd.cdna4.async_copy.wait_group(0)
        gl.barrier()
        # The slots of the next tile, read now and used in the next iteration.
        slots = _load_slots(row, (j + 1) * BLOCK_K + cols_g, last)
        s = gl.zeros([H, BLOCK_K], gl.float32, layout=mfma_l)
        s = gl.amd.cdna4.mfma(q0, _k_chunk(tile, 0, dot_b), s)
        s = gl.amd.cdna4.mfma(q1, _k_chunk(tile, 1, dot_b), s)
        s = gl.amd.cdna4.mfma(q2, _k_chunk(tile, 2, dot_b), s)
        s = gl.amd.cdna4.mfma(q3, _k_chunk(tile, 3, dot_b), s)
        # Built per tile, not hoisted out of the loop: hoisted, the kernel ran
        # up to 5 % slower with Triton 3.8 on some index lists (no difference
        # with Triton 3.7).
        cols_c = gl.arange(0, BLOCK_K, layout=gl.SliceLayout(0, mfma_l))
        s = gl.where((j * BLOCK_K + cols_c < n)[None, :], s, float("-inf"))
        m_new = gl.maximum(m, gl.max(s, axis=1) * qk_scale)
        p = gl.exp2(s * qk_scale - m_new[:, None])
        alpha = gl.exp2(m - m_new)
        l = l * alpha + gl.sum(p, axis=1)
        m = m_new
        p_dot = gl.convert_layout(p.to(gl.bfloat16), dot_a)
        al = alpha[:, None]
        a0 = gl.amd.cdna4.mfma(p_dot, _v_chunk(tile, 0, dot_b), a0 * al)
        a1 = gl.amd.cdna4.mfma(p_dot, _v_chunk(tile, 1, dot_b), a1 * al)
        a2 = gl.amd.cdna4.mfma(p_dot, _v_chunk(tile, 2, dot_b), a2 * al)
        a3 = gl.amd.cdna4.mfma(p_dot, _v_chunk(tile, 3, dot_b), a3 * al)
        gl.barrier()  # all reads of the tile end before the next copy
    h = gl.arange(0, H, layout=gl.SliceLayout(1, mfma_l))
    if HAS_SINK:
        sink_l2 = gl.load(sink_ptr + h) * 1.44269504089
        m_f = gl.maximum(m, sink_l2)
        alpha_f = gl.exp2(m - m_f)
        l_f = l * alpha_f + gl.exp2(sink_l2 - m_f)
        o_scale = gl.where(l_f > 0.0, alpha_f / l_f, 0.0)
    else:
        o_scale = gl.where(l > 0.0, 1.0 / l, 0.0)
    z = o_scale[:, None]
    d = gl.arange(0, DC, layout=gl.SliceLayout(0, mfma_l))
    p_o = (
        o_ptr
        + r.to(gl.int64) * stride_ot
        + h[:, None].to(gl.int64) * stride_oh
        + d[None, :]
    )
    gl.store(p_o, (a0 * z).to(gl.bfloat16))
    gl.store(p_o + DC, (a1 * z).to(gl.bfloat16))
    gl.store(p_o + 2 * DC, (a2 * z).to(gl.bfloat16))
    gl.store(p_o + 3 * DC, (a3 * z).to(gl.bfloat16))
