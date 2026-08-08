##############################################################################
# MIT License
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
##############################################################################

import json

from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from aiter.ops.triton.utils._triton import arch_info
from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr
from aiter.ops.triton.utils._triton.mha_kernel_utils import (
    _compute_fp8_scaling_factors,
)
from aiter.ops.triton.utils._triton.pid_preprocessing import remap_xcd
from aiter.ops.triton.utils.core import AITER_TRITON_CONFIGS_PATH


@gluon.constexpr_function
def _make_kv_shared_layouts(
    head_dim_pow2, elem_bytes, k_width=8, non_k_dim=16, banks=64
):
    """Swizzled LDS layouts for the K/V staging tiles."""
    bank_line_bytes = banks * 4
    bank_line_elems = bank_line_bytes // elem_bytes
    read_vec_bytes = min(k_width * elem_bytes, 16)
    num_threads_same_cycle = bank_line_bytes // read_vec_bytes
    per_phase = (bank_line_elems + head_dim_pow2 - 1) // head_dim_pow2
    swizzle_vec = min(k_width * max(1, per_phase // 2), read_vec_bytes // elem_bytes)
    max_phase = min(
        min(non_k_dim, num_threads_same_cycle) // per_phase,
        bank_line_elems // swizzle_vec,
    )
    k_shared = gl.SwizzledSharedLayout(swizzle_vec, per_phase, max_phase, order=[0, 1])
    v_shared = gl.SwizzledSharedLayout(swizzle_vec, per_phase, max_phase, order=[1, 0])
    return k_shared, v_shared


@gluon.constexpr_function
def _make_load_layout(block_dmodel, load_vec, num_warps, transposed, lanes=64):
    """Blocked layout for one tile load."""
    warp_elems = lanes * load_vec
    if transposed:
        return gl.BlockedLayout(
            [load_vec, 1],
            [block_dmodel // load_vec, warp_elems // block_dmodel],
            [1, num_warps],
            [0, 1],
        )
    return gl.BlockedLayout(
        [1, load_vec],
        [warp_elems // block_dmodel, block_dmodel // load_vec],
        [num_warps, 1],
        [1, 0],
    )


@gluon.jit
def _buffer_load_2d(
    base, offsets, offset_first, offset_second, boundary_first, boundary_second
):
    """buffer_load of one tile into registers; masked lanes read 0."""
    if offset_first is not None and offset_second is not None:
        mask = (offset_first[:, None] < boundary_first) & (
            offset_second[None, :] < boundary_second
        )
        tile = gl.amd.cdna4.buffer_load(ptr=base, offsets=offsets, mask=mask, other=0.0)
    elif offset_first is not None:
        mask = offset_first[:, None] < boundary_first
        tile = gl.amd.cdna4.buffer_load(ptr=base, offsets=offsets, mask=mask, other=0.0)
    elif offset_second is not None:
        mask = offset_second[None, :] < boundary_second
        tile = gl.amd.cdna4.buffer_load(ptr=base, offsets=offsets, mask=mask, other=0.0)
    else:
        tile = gl.amd.cdna4.buffer_load(ptr=base, offsets=offsets)
    return tile


@gluon.jit
def _load_k(
    k_base,
    k_offsets,
    k_pe_offsets,
    load_start_n,
    seqlen_k,
    kLoadLayout: gl.constexpr,
    kPeLoadLayout: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_DMODEL: gl.constexpr,
    BLOCK_DMODEL_POW2: gl.constexpr,
    BLOCK_DMODEL_PE: gl.constexpr,
    MASK_STEPS: gl.constexpr,
    PADDED_HEAD: gl.constexpr,
    HAS_PE: gl.constexpr,
):
    """buffer_load one K block ([BLOCK_DMODEL_POW2, BLOCK_N]), and its PE ([BLOCK_DMODEL_PE, BLOCK_N]) when PE is on (else None)."""
    k_offs_n = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, kLoadLayout))
    k_offs_d = gl.arange(0, BLOCK_DMODEL_POW2, layout=gl.SliceLayout(1, kLoadLayout))
    if MASK_STEPS:
        k_n = load_start_n + k_offs_n
    else:
        k_n = None
    if PADDED_HEAD:
        k_d = k_offs_d
    else:
        k_d = None
    k_tile = _buffer_load_2d(k_base, k_offsets, k_d, k_n, BLOCK_DMODEL, seqlen_k)

    if HAS_PE:
        if MASK_STEPS:
            k_pe_n = load_start_n + gl.arange(
                0, BLOCK_N, layout=gl.SliceLayout(0, kPeLoadLayout)
            )
        else:
            k_pe_n = None
        return k_tile, _buffer_load_2d(
            k_base, k_pe_offsets, None, k_pe_n, BLOCK_DMODEL_PE, seqlen_k
        )
    else:
        return k_tile, None


@gluon.jit
def _load_v(
    v_base,
    v_offsets,
    load_start_n,
    seqlen_k,
    vLoadLayout: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_DMODEL: gl.constexpr,
    BLOCK_DMODEL_POW2: gl.constexpr,
    MASK_STEPS: gl.constexpr,
    PADDED_HEAD: gl.constexpr,
):
    """buffer_load one V block ([BLOCK_N, BLOCK_DMODEL_POW2]) into registers."""
    v_offs_n = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(1, vLoadLayout))
    v_offs_d = gl.arange(0, BLOCK_DMODEL_POW2, layout=gl.SliceLayout(0, vLoadLayout))
    if MASK_STEPS:
        v_n = load_start_n + v_offs_n
    else:
        v_n = None
    if PADDED_HEAD:
        v_d = v_offs_d
    else:
        v_d = None
    return _buffer_load_2d(v_base, v_offsets, v_n, v_d, seqlen_k, BLOCK_DMODEL)


@gluon.jit
def _store_k_smem(smemK, smemKpe, buf, k_tile, k_pe_tile, HAS_PE: gl.constexpr):
    smemK.index(buf).store(k_tile)
    if HAS_PE:
        smemKpe.index(buf).store(k_pe_tile)


@gluon.jit
def _load_k_smem(smemK, smemKpe, buf, dotK: gl.constexpr, HAS_PE: gl.constexpr):
    k = smemK.index(buf).load(dotK)
    if HAS_PE:
        return k, smemKpe.index(buf).load(dotK)
    else:
        return k, None


@gluon.jit
def _attn_qk(
    q,
    k,
    q_pe,
    k_pe,
    start_n,
    offs_n,
    window_min,
    qk_scale,
    mfmaLayout: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    HAS_PE: gl.constexpr,
    IS_FP8: gl.constexpr,
    SLIDING_WINDOW: gl.constexpr,
    seqlen_q=None,
    seqlen_k=None,
    block_max=None,
    n_extra_tokens=None,
    offs_m=None,
    IS_CAUSAL: gl.constexpr = False,
    MASK_STEPS: gl.constexpr = False,
):
    """QK^T + scale + mask for one already-staged key block. ``k`` is already in
    its MFMA dot-operand layout; returns float32 scores in ``mfmaLayout``. For
    FP8 the QK^T uses the CDNA4 scaled MFMA (32x32x64).
    """
    qk = gl.zeros([BLOCK_M, BLOCK_N], dtype=gl.float32, layout=mfmaLayout)
    if IS_FP8:
        if HAS_PE:
            qk = gl.amd.cdna4.mfma_scaled(q_pe, None, "e4m3", k_pe, None, "e4m3", qk)
        qk = gl.amd.cdna4.mfma_scaled(q, None, "e4m3", k, None, "e4m3", qk)
    else:
        if HAS_PE:
            qk = gl.amd.cdna4.mfma(q_pe, k_pe, qk)
        qk = gl.amd.cdna4.mfma(q, k, qk)
    qk = qk * qk_scale

    if MASK_STEPS or IS_CAUSAL or SLIDING_WINDOW > 0:
        key_pos = start_n + offs_n
        mask = gl.full([BLOCK_M, BLOCK_N], True, dtype=gl.int1, layout=mfmaLayout)
        if MASK_STEPS:
            # Only the last visible block can be partial (seqlen_k not a multiple
            # of BLOCK_N).
            bound_cond = (start_n + BLOCK_N == block_max) and (n_extra_tokens != 0)
            mask_partial = key_pos[None, :] < seqlen_k
            mask = gl.where(bound_cond, mask_partial, mask)
        if IS_CAUSAL:
            causal_boundary = key_pos + (seqlen_q - seqlen_k)
            mask = mask & (offs_m[:, None] >= causal_boundary[None, :])
        if SLIDING_WINDOW > 0:
            mask = mask & (window_min[:, None] <= key_pos[None, :])
        qk = gl.where(mask, qk, float("-inf"))

    return qk


@gluon.jit
def _attn_softmax_pv(
    acc,
    l_i,
    m_i,
    qk,
    v,
    descale_v,
    dotP: gl.constexpr,
    mfmaLayout: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_DMODEL_POW2: gl.constexpr,
    IS_FP8: gl.constexpr,
    FP8_MAX: gl.constexpr,
):
    """Online-softmax rescale + P@V accumulation for one key block.
    Returns updated (acc, l_i, m_i)."""

    m_ij = gl.maximum(m_i, gl.max(qk, 1))
    p = gl.exp2(qk - m_ij[:, None])
    alpha = gl.exp2(m_i - m_ij)
    l_ij = gl.sum(p, 1)

    acc = acc * alpha[:, None]

    if IS_FP8:
        scale_p, descale_p = _compute_fp8_scaling_factors(p, FP8_MAX)
        p = gl.convert_layout(
            (p * scale_p).to(v.dtype), layout=dotP, assert_trivial=True
        )
        pv = gl.zeros([BLOCK_M, BLOCK_DMODEL_POW2], dtype=gl.float32, layout=mfmaLayout)
        pv = gl.amd.cdna4.mfma_scaled(p, None, "e4m3", v, None, "e4m3", pv)
        acc = acc + pv * (descale_p * descale_v)
    else:
        p = gl.convert_layout(p.to(v.dtype), layout=dotP, assert_trivial=True)
        acc = gl.amd.cdna4.mfma(p, v, acc)

    l_i = l_i * alpha + l_ij
    m_i = m_ij

    return acc, l_i, m_i


@gluon.jit
def _attn_fwd_inner(
    acc,
    l_i,
    m_i,
    q,
    q_pe,
    k_base,
    k_offsets,
    k_pe_offsets,
    v_base,
    v_offsets,
    smemK,
    smemKpe,
    smemV,
    stride_kn,
    stride_vn,
    seqlen_k,
    block_min,
    block_max,
    window_min,
    qk_scale,
    descale_v,
    mfmaLayout: gl.constexpr,
    dotK: gl.constexpr,
    dotP: gl.constexpr,
    dotV: gl.constexpr,
    kLoadLayout: gl.constexpr,
    kPeLoadLayout: gl.constexpr,
    vLoadLayout: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_DMODEL: gl.constexpr,
    BLOCK_DMODEL_POW2: gl.constexpr,
    BLOCK_DMODEL_PE: gl.constexpr,
    HAS_PE: gl.constexpr,
    NUM_KV_BUFFERS: gl.constexpr,
    IS_FP8: gl.constexpr,
    FP8_MAX: gl.constexpr,
    SLIDING_WINDOW: gl.constexpr,
):
    """Software-pipelined online-softmax loop over the blocks that need no
    boundary or causal mask (the sliding-window mask, if any, still applies).
    """
    PADDED_HEAD: gl.constexpr = BLOCK_DMODEL != BLOCK_DMODEL_POW2

    if SLIDING_WINDOW > 0:
        offs_n = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, mfmaLayout))
    else:
        offs_n = None

    n_iter = (block_max - block_min) // BLOCK_N

    # Prologue
    k_tile, k_pe_tile = _load_k(
        k_base,
        k_offsets,
        k_pe_offsets,
        block_min,
        seqlen_k,
        kLoadLayout,
        kPeLoadLayout,
        BLOCK_N,
        BLOCK_DMODEL,
        BLOCK_DMODEL_POW2,
        BLOCK_DMODEL_PE,
        False,
        PADDED_HEAD,
        HAS_PE,
    )
    v_tile = _load_v(
        v_base,
        v_offsets,
        block_min,
        seqlen_k,
        vLoadLayout,
        BLOCK_N,
        BLOCK_DMODEL,
        BLOCK_DMODEL_POW2,
        False,
        PADDED_HEAD,
    )
    _store_k_smem(smemK, smemKpe, 0, k_tile, k_pe_tile, HAS_PE)
    smemV.index(0).store(v_tile)

    if n_iter > 1:
        k_tile, k_pe_tile = _load_k(
            k_base + BLOCK_N * stride_kn,
            k_offsets,
            k_pe_offsets,
            block_min + BLOCK_N,
            seqlen_k,
            kLoadLayout,
            kPeLoadLayout,
            BLOCK_N,
            BLOCK_DMODEL,
            BLOCK_DMODEL_POW2,
            BLOCK_DMODEL_PE,
            False,
            PADDED_HEAD,
            HAS_PE,
        )
        v_tile = _load_v(
            v_base + BLOCK_N * stride_vn,
            v_offsets,
            block_min + BLOCK_N,
            seqlen_k,
            vLoadLayout,
            BLOCK_N,
            BLOCK_DMODEL,
            BLOCK_DMODEL_POW2,
            False,
            PADDED_HEAD,
        )
        _store_k_smem(smemK, smemKpe, 1, k_tile, k_pe_tile, HAS_PE)
        smemV.index(1).store(v_tile)

    for i in range(n_iter - NUM_KV_BUFFERS):
        buf = i % NUM_KV_BUFFERS

        # Read block i
        k, k_pe = _load_k_smem(smemK, smemKpe, buf, dotK, HAS_PE)
        v = smemV.index(buf).load(dotV)

        # Issue block i+2
        k_pf, k_pe_pf = _load_k(
            k_base + (i + 2) * BLOCK_N * stride_kn,
            k_offsets,
            k_pe_offsets,
            block_min + (i + 2) * BLOCK_N,
            seqlen_k,
            kLoadLayout,
            kPeLoadLayout,
            BLOCK_N,
            BLOCK_DMODEL,
            BLOCK_DMODEL_POW2,
            BLOCK_DMODEL_PE,
            False,
            PADDED_HEAD,
            HAS_PE,
        )
        v_pf = _load_v(
            v_base + (i + 2) * BLOCK_N * stride_vn,
            v_offsets,
            block_min + (i + 2) * BLOCK_N,
            seqlen_k,
            vLoadLayout,
            BLOCK_N,
            BLOCK_DMODEL,
            BLOCK_DMODEL_POW2,
            False,
            PADDED_HEAD,
        )

        qk = _attn_qk(
            q,
            k,
            q_pe,
            k_pe,
            block_min + i * BLOCK_N,
            offs_n,
            window_min,
            qk_scale,
            mfmaLayout=mfmaLayout,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            HAS_PE=HAS_PE,
            IS_FP8=IS_FP8,
            SLIDING_WINDOW=SLIDING_WINDOW,
        )

        acc, l_i, m_i = _attn_softmax_pv(
            acc,
            l_i,
            m_i,
            qk,
            v,
            descale_v,
            dotP,
            mfmaLayout,
            BLOCK_M,
            BLOCK_DMODEL_POW2,
            IS_FP8,
            FP8_MAX,
        )

        # Store block i+2
        _store_k_smem(smemK, smemKpe, buf, k_pf, k_pe_pf, HAS_PE)
        smemV.index(buf).store(v_pf)

    # Epilogue
    if n_iter > 1:
        buf = (n_iter - 2) % NUM_KV_BUFFERS
        k, k_pe = _load_k_smem(smemK, smemKpe, buf, dotK, HAS_PE)
        v = smemV.index(buf).load(dotV)
        qk = _attn_qk(
            q,
            k,
            q_pe,
            k_pe,
            block_min + (n_iter - 2) * BLOCK_N,
            offs_n,
            window_min,
            qk_scale,
            mfmaLayout=mfmaLayout,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            HAS_PE=HAS_PE,
            IS_FP8=IS_FP8,
            SLIDING_WINDOW=SLIDING_WINDOW,
        )
        acc, l_i, m_i = _attn_softmax_pv(
            acc,
            l_i,
            m_i,
            qk,
            v,
            descale_v,
            dotP,
            mfmaLayout,
            BLOCK_M,
            BLOCK_DMODEL_POW2,
            IS_FP8,
            FP8_MAX,
        )

    buf = (n_iter - 1) % NUM_KV_BUFFERS
    k, k_pe = _load_k_smem(smemK, smemKpe, buf, dotK, HAS_PE)
    v = smemV.index(buf).load(dotV)
    qk = _attn_qk(
        q,
        k,
        q_pe,
        k_pe,
        block_min + (n_iter - 1) * BLOCK_N,
        offs_n,
        window_min,
        qk_scale,
        mfmaLayout=mfmaLayout,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        HAS_PE=HAS_PE,
        IS_FP8=IS_FP8,
        SLIDING_WINDOW=SLIDING_WINDOW,
    )
    acc, l_i, m_i = _attn_softmax_pv(
        acc,
        l_i,
        m_i,
        qk,
        v,
        descale_v,
        dotP,
        mfmaLayout,
        BLOCK_M,
        BLOCK_DMODEL_POW2,
        IS_FP8,
        FP8_MAX,
    )

    return acc, l_i, m_i


@gluon.jit
def _attn_fwd_inner_masked(
    acc,
    l_i,
    m_i,
    q,
    q_pe,
    k_base,
    k_offsets,
    k_pe_offsets,
    v_base,
    v_offsets,
    smemK,
    smemKpe,
    smemV,
    stride_kn,
    stride_vn,
    seqlen_q,
    seqlen_k,
    block_min,
    block_max,
    n_extra_tokens,
    offs_m,
    window_min,
    qk_scale,
    descale_v,
    mfmaLayout: gl.constexpr,
    dotK: gl.constexpr,
    dotP: gl.constexpr,
    dotV: gl.constexpr,
    kLoadLayout: gl.constexpr,
    kPeLoadLayout: gl.constexpr,
    vLoadLayout: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_DMODEL: gl.constexpr,
    BLOCK_DMODEL_POW2: gl.constexpr,
    BLOCK_DMODEL_PE: gl.constexpr,
    HAS_PE: gl.constexpr,
    IS_CAUSAL: gl.constexpr,
    MASK_STEPS: gl.constexpr,
    IS_FP8: gl.constexpr,
    FP8_MAX: gl.constexpr,
    SLIDING_WINDOW: gl.constexpr,
):
    """Non-pipelined online-softmax loop over the boundary / causal (and, if
    enabled, sliding-window) masked blocks.
    """
    PADDED_HEAD: gl.constexpr = BLOCK_DMODEL != BLOCK_DMODEL_POW2

    offs_n = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, mfmaLayout))

    n_iter = (block_max - block_min) // BLOCK_N

    for i in range(n_iter):
        start_n = block_min + i * BLOCK_N

        k_tile, k_pe_tile = _load_k(
            k_base,
            k_offsets,
            k_pe_offsets,
            start_n,
            seqlen_k,
            kLoadLayout,
            kPeLoadLayout,
            BLOCK_N,
            BLOCK_DMODEL,
            BLOCK_DMODEL_POW2,
            BLOCK_DMODEL_PE,
            MASK_STEPS,
            PADDED_HEAD,
            HAS_PE,
        )
        v_tile = _load_v(
            v_base,
            v_offsets,
            start_n,
            seqlen_k,
            vLoadLayout,
            BLOCK_N,
            BLOCK_DMODEL,
            BLOCK_DMODEL_POW2,
            MASK_STEPS,
            PADDED_HEAD,
        )
        _store_k_smem(smemK, smemKpe, 0, k_tile, k_pe_tile, HAS_PE)
        smemV.index(0).store(v_tile)
        k_base += BLOCK_N * stride_kn
        v_base += BLOCK_N * stride_vn

        k, k_pe = _load_k_smem(smemK, smemKpe, 0, dotK, HAS_PE)
        v = smemV.index(0).load(dotV)

        qk = _attn_qk(
            q,
            k,
            q_pe,
            k_pe,
            start_n,
            offs_n,
            window_min,
            qk_scale,
            mfmaLayout=mfmaLayout,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            HAS_PE=HAS_PE,
            IS_FP8=IS_FP8,
            SLIDING_WINDOW=SLIDING_WINDOW,
            seqlen_q=seqlen_q,
            seqlen_k=seqlen_k,
            block_max=block_max,
            n_extra_tokens=n_extra_tokens,
            offs_m=offs_m,
            IS_CAUSAL=IS_CAUSAL,
            MASK_STEPS=MASK_STEPS,
        )
        acc, l_i, m_i = _attn_softmax_pv(
            acc,
            l_i,
            m_i,
            qk,
            v,
            descale_v,
            dotP,
            mfmaLayout,
            BLOCK_M,
            BLOCK_DMODEL_POW2,
            IS_FP8,
            FP8_MAX,
        )

    return acc, l_i, m_i


_attn_fwd_repr = make_kernel_repr(
    "_attn_fwd",
    [
        "IS_CAUSAL",
        "NUM_Q_HEADS",
        "NUM_K_HEADS",
        "BLOCK_M",
        "BLOCK_N",
        "BLOCK_DMODEL",
        "IS_FP8",
        "VARLEN",
        "NUM_XCD",
        "USE_INT64_STRIDES",
        "ENABLE_SINK",
        "SLIDING_WINDOW",
    ],
)


@gluon.jit(repr=_attn_fwd_repr)
def _attn_fwd(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    descale_q_ptr,
    descale_k_ptr,
    descale_v_ptr,
    sink_ptr,
    sm_scale,
    cu_seqlens_q,
    cu_seqlens_k,
    SEQLEN_Q,
    SEQLEN_K,
    stride_qz_in,
    stride_qh_in,
    stride_qm_in,
    stride_qk_in,
    stride_kz_in,
    stride_kh_in,
    stride_kn_in,
    stride_kk_in,
    stride_vz_in,
    stride_vh_in,
    stride_vn_in,
    stride_vk_in,
    stride_oz_in,
    stride_oh_in,
    stride_om_in,
    stride_on_in,
    stride_descale_q_z_in,
    stride_descale_k_z_in,
    stride_descale_v_z_in,
    NUM_Q_HEADS: gl.constexpr,
    NUM_K_HEADS: gl.constexpr,
    IS_CAUSAL: gl.constexpr,
    VARLEN: gl.constexpr,
    BATCH,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_DMODEL: gl.constexpr,
    BLOCK_DMODEL_POW2: gl.constexpr,
    BLOCK_DMODEL_PE: gl.constexpr,  # zero, or a power of 2 >= 16
    NUM_XCD: gl.constexpr,
    USE_INT64_STRIDES: gl.constexpr,
    IS_FP8: gl.constexpr,
    FP8_MAX: gl.constexpr,
    ENABLE_SINK: gl.constexpr,
    SLIDING_WINDOW: gl.constexpr,
    HEAD_STRIDE_ALIGNED_8: gl.constexpr = False,
    num_warps: gl.constexpr = 4,
):
    RCP_LN2: gl.constexpr = 1.4426950408889634
    PADDED_HEAD: gl.constexpr = BLOCK_DMODEL != BLOCK_DMODEL_POW2
    HAS_PE: gl.constexpr = BLOCK_DMODEL_PE > 0

    # NOTE:
    # Base-pointer and seqlen-loop offset arithmetic is performed using the
    # stride's integer width. With 32-bit strides, these products can overflow
    # and cause segfaults on very large tensors. Upcasting the strides to int64
    # ensures that this arithmetic uses 64-bit precision. The per-tile offset
    # tensors are still downcast to int32 for buffer_load, which is safe, as a
    # single tile's offsets are small.
    if USE_INT64_STRIDES:
        stride_qz = gl.cast(stride_qz_in, gl.int64)
        stride_qh = gl.cast(stride_qh_in, gl.int64)
        stride_qm = gl.cast(stride_qm_in, gl.int64)
        stride_qk = gl.cast(stride_qk_in, gl.int64)
        stride_kz = gl.cast(stride_kz_in, gl.int64)
        stride_kh = gl.cast(stride_kh_in, gl.int64)
        stride_kn = gl.cast(stride_kn_in, gl.int64)
        stride_kk = gl.cast(stride_kk_in, gl.int64)
        stride_vz = gl.cast(stride_vz_in, gl.int64)
        stride_vh = gl.cast(stride_vh_in, gl.int64)
        stride_vn = gl.cast(stride_vn_in, gl.int64)
        stride_vk = gl.cast(stride_vk_in, gl.int64)
        if IS_FP8:
            stride_descale_q_z = gl.cast(stride_descale_q_z_in, gl.int64)
            stride_descale_k_z = gl.cast(stride_descale_k_z_in, gl.int64)
            stride_descale_v_z = gl.cast(stride_descale_v_z_in, gl.int64)
        stride_oz = gl.cast(stride_oz_in, gl.int64)
        stride_oh = gl.cast(stride_oh_in, gl.int64)
        stride_om = gl.cast(stride_om_in, gl.int64)
        stride_on = gl.cast(stride_on_in, gl.int64)
    else:
        stride_qz = stride_qz_in
        stride_qh = stride_qh_in
        stride_qm = stride_qm_in
        stride_qk = stride_qk_in
        stride_kz = stride_kz_in
        stride_kh = stride_kh_in
        stride_kn = stride_kn_in
        stride_kk = stride_kk_in
        stride_vz = stride_vz_in
        stride_vh = stride_vh_in
        stride_vn = stride_vn_in
        stride_vk = stride_vk_in
        stride_descale_q_z = stride_descale_q_z_in
        stride_descale_k_z = stride_descale_k_z_in
        stride_descale_v_z = stride_descale_v_z_in
        stride_oz = stride_oz_in
        stride_oh = stride_oh_in
        stride_om = stride_om_in
        stride_on = stride_on_in

    # program -> (batch, q_head, query block). SEQLEN_Q is the max query length,
    # so NUM_BLOCKS_M matches the launch grid in both fixed and varlen mode.
    NUM_BLOCKS_M = gl.cdiv(SEQLEN_Q, BLOCK_M)
    pid = gl.program_id(axis=0)
    off_q_head = pid % NUM_Q_HEADS
    # Remap the q-head index across XCDs for better cache locality.
    off_q_head = remap_xcd(off_q_head, NUM_Q_HEADS, NUM_XCD)
    start_m = (pid // NUM_Q_HEADS) % NUM_BLOCKS_M
    off_z = pid // (NUM_Q_HEADS * NUM_BLOCKS_M) % BATCH

    # In varlen mode the lengths come from cu_seqlens and the batch axis is
    # collapsed (stride_*z == 0); in fixed mode use the SEQLEN_Q/SEQLEN_K args.
    if VARLEN:
        cu_seqlens_q_start = gl.load(cu_seqlens_q + off_z)
        seqlen_q = gl.load(cu_seqlens_q + off_z + 1) - cu_seqlens_q_start
        # This query block is entirely past the end of this batch's sequence.
        if start_m * BLOCK_M >= seqlen_q:
            return
        cu_seqlens_k_start = gl.load(cu_seqlens_k + off_z)
        seqlen_k = gl.load(cu_seqlens_k + off_z + 1) - cu_seqlens_k_start
    else:
        cu_seqlens_q_start = 0
        cu_seqlens_k_start = 0
        seqlen_q = SEQLEN_Q
        seqlen_k = SEQLEN_K

    grp_sz: gl.constexpr = NUM_Q_HEADS // NUM_K_HEADS
    off_k_head = off_q_head // grp_sz

    if IS_FP8:
        descale_q = gl.load(descale_q_ptr + off_z * stride_descale_q_z + off_q_head)
        descale_k = gl.load(descale_k_ptr + off_z * stride_descale_k_z + off_k_head)
        descale_v = gl.load(descale_v_ptr + off_z * stride_descale_v_z + off_k_head)
    else:
        descale_q = 1.0
        descale_k = 1.0
        descale_v = 1.0

    MFMA_INSTR: gl.constexpr = [32, 32, 64] if IS_FP8 else [32, 32, 16]
    mfmaLayout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=MFMA_INSTR,
        transposed=True,
        warps_per_cta=[num_warps, 1],
    )

    K_WIDTH: gl.constexpr = 16 if IS_FP8 else 8
    PV_K_WIDTH: gl.constexpr = 4
    dotQ: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=mfmaLayout, k_width=K_WIDTH
    )
    dotK: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=mfmaLayout, k_width=K_WIDTH
    )
    dotP: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=mfmaLayout, k_width=PV_K_WIDTH
    )
    dotV: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=mfmaLayout, k_width=PV_K_WIDTH
    )

    LOAD_VEC: gl.constexpr = 16 if IS_FP8 else 8
    qLoadLayout: gl.constexpr = _make_load_layout(
        BLOCK_DMODEL_POW2, LOAD_VEC, num_warps, transposed=False
    )
    kLoadLayout: gl.constexpr = _make_load_layout(
        BLOCK_DMODEL_POW2, LOAD_VEC, num_warps, transposed=True
    )
    vLoadLayout: gl.constexpr = _make_load_layout(
        BLOCK_DMODEL_POW2, LOAD_VEC, num_warps, transposed=False
    )

    # Swizzled shared layouts for the reg->LDS staging of K and V (conflict-free
    # ds_write / ds_read).
    ELEM_BYTES: gl.constexpr = k_ptr.dtype.element_ty.primitive_bitwidth // 8
    _KV_SHARED: gl.constexpr = _make_kv_shared_layouts(
        BLOCK_DMODEL_POW2, ELEM_BYTES, k_width=K_WIDTH
    )
    kSharedLayout: gl.constexpr = _KV_SHARED[0]
    vSharedLayout: gl.constexpr = _KV_SHARED[1]

    if HAS_PE:
        qPeLoadLayout: gl.constexpr = _make_load_layout(
            BLOCK_DMODEL_PE, LOAD_VEC, num_warps, transposed=False
        )
        kPeLoadLayout: gl.constexpr = _make_load_layout(
            BLOCK_DMODEL_PE, LOAD_VEC, num_warps, transposed=True
        )
        _KPE_SHARED: gl.constexpr = _make_kv_shared_layouts(
            BLOCK_DMODEL_PE, ELEM_BYTES, k_width=K_WIDTH
        )
        kPeSharedLayout: gl.constexpr = _KPE_SHARED[0]
    else:
        qPeLoadLayout: gl.constexpr = None
        kPeLoadLayout: gl.constexpr = None
        kPeSharedLayout: gl.constexpr = None

    # When the caller guarantees Q/K/V head strides are multiples of 8 elements,
    # the head-axis offset is 16-byte aligned; hinting the multiple lets AxisInfo
    # widen the global loads.
    qh_off = off_q_head * stride_qh
    kh_off = off_k_head * stride_kh
    vh_off = off_k_head * stride_vh
    if HEAD_STRIDE_ALIGNED_8:
        qh_off = gl.multiple_of(qh_off, 8)
        kh_off = gl.multiple_of(kh_off, 8)
        vh_off = gl.multiple_of(vh_off, 8)

    # Load Q (stays resident for the whole key loop).
    offs_qm = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, qLoadLayout))
    offs_qd = gl.arange(0, BLOCK_DMODEL_POW2, layout=gl.SliceLayout(0, qLoadLayout))
    q_base = (
        q_ptr
        + off_z * stride_qz
        + qh_off
        + cu_seqlens_q_start * stride_qm
        + start_m * BLOCK_M * stride_qm
    )
    q_offsets = (offs_qm[:, None] * stride_qm + offs_qd[None, :] * stride_qk).to(
        gl.int32
    )
    q_mask = (start_m * BLOCK_M + offs_qm)[:, None] < seqlen_q
    if PADDED_HEAD:
        q_mask = q_mask & (offs_qd[None, :] < BLOCK_DMODEL)
    # Cache Q at .cg when a single Q block spans at least one full head.
    if BLOCK_M >= NUM_Q_HEADS:
        q_cache_mod: gl.constexpr = ".cg"
    else:
        q_cache_mod: gl.constexpr = ""
    q = gl.amd.cdna4.buffer_load(
        ptr=q_base, offsets=q_offsets, mask=q_mask, other=0.0, cache=q_cache_mod
    )
    q = gl.convert_layout(q, layout=dotQ)

    # The PE slice sits immediately after the NOPE slice along the head dim of
    # Q and K, so it shares their base pointer and only shifts the head-dim
    # offsets by BLOCK_DMODEL. V and the output only span the NOPE slice.
    if HAS_PE:
        offs_qpm = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, qPeLoadLayout))
        offs_qpd = BLOCK_DMODEL + gl.arange(
            0, BLOCK_DMODEL_PE, layout=gl.SliceLayout(0, qPeLoadLayout)
        )
        q_pe_offsets = (
            offs_qpm[:, None] * stride_qm + offs_qpd[None, :] * stride_qk
        ).to(gl.int32)
        q_pe_mask = (start_m * BLOCK_M + offs_qpm)[:, None] < seqlen_q
        q_pe = gl.amd.cdna4.buffer_load(
            ptr=q_base,
            offsets=q_pe_offsets,
            mask=q_pe_mask,
            other=0.0,
            cache=q_cache_mod,
        )
        q_pe = gl.convert_layout(q_pe, layout=dotQ)
    else:
        q_pe = None

    offs_kd = gl.arange(0, BLOCK_DMODEL_POW2, layout=gl.SliceLayout(1, kLoadLayout))
    offs_kn = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, kLoadLayout))
    k_base = k_ptr + off_z * stride_kz + kh_off + cu_seqlens_k_start * stride_kn
    k_offsets = (offs_kd[:, None] * stride_kk + offs_kn[None, :] * stride_kn).to(
        gl.int32
    )

    if HAS_PE:
        offs_kpd = BLOCK_DMODEL + gl.arange(
            0, BLOCK_DMODEL_PE, layout=gl.SliceLayout(1, kPeLoadLayout)
        )
        offs_kpn = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, kPeLoadLayout))
        k_pe_offsets = (
            offs_kpd[:, None] * stride_kk + offs_kpn[None, :] * stride_kn
        ).to(gl.int32)
    else:
        k_pe_offsets = None

    offs_vn = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(1, vLoadLayout))
    offs_vd = gl.arange(0, BLOCK_DMODEL_POW2, layout=gl.SliceLayout(0, vLoadLayout))
    v_base = v_ptr + off_z * stride_vz + vh_off + cu_seqlens_k_start * stride_vn
    v_offsets = (offs_vn[:, None] * stride_vn + offs_vd[None, :] * stride_vk).to(
        gl.int32
    )

    # Shared-memory tiles for the K/V staging.
    NUM_KV_BUFFERS: gl.constexpr = 2
    smemK = gl.allocate_shared_memory(
        k_ptr.dtype.element_ty,
        [NUM_KV_BUFFERS, BLOCK_DMODEL_POW2, BLOCK_N],
        kSharedLayout,
    )
    smemV = gl.allocate_shared_memory(
        v_ptr.dtype.element_ty,
        [NUM_KV_BUFFERS, BLOCK_N, BLOCK_DMODEL_POW2],
        vSharedLayout,
    )
    if HAS_PE:
        smemKpe = gl.allocate_shared_memory(
            k_ptr.dtype.element_ty,
            [NUM_KV_BUFFERS, BLOCK_DMODEL_PE, BLOCK_N],
            kPeSharedLayout,
        )
    else:
        smemKpe = None

    # online-softmax state.
    if ENABLE_SINK:
        m_i_init = gl.load(sink_ptr + off_q_head).to(gl.float32) * RCP_LN2
    elif SLIDING_WINDOW > 0:
        # A sliding-window block can be fully masked for some rows, and -inf as the
        # running max would then make exp2(-inf - m_i) NaN. A finite floor keeps the
        # probabilities at 0 and the rescale factor at exactly 1.0.
        m_i_init = -1.0e30
    else:
        m_i_init = float("-inf")

    m_i = gl.full(
        [BLOCK_M], m_i_init, dtype=gl.float32, layout=gl.SliceLayout(1, mfmaLayout)
    )
    l_i = gl.full(
        [BLOCK_M], 1.0, dtype=gl.float32, layout=gl.SliceLayout(1, mfmaLayout)
    )
    acc = gl.zeros([BLOCK_M, BLOCK_DMODEL_POW2], dtype=gl.float32, layout=mfmaLayout)

    qk_scale = sm_scale * RCP_LN2
    if IS_FP8:
        qk_scale = qk_scale * descale_q * descale_k

    # Query positions used for the causal mask, in the MFMA result layout.
    offs_m = start_m * BLOCK_M + gl.arange(
        0, BLOCK_M, layout=gl.SliceLayout(1, mfmaLayout)
    )

    # Lowest key index each query row may attend. Like the causal mask, the
    # window is aligned to the bottom right corner.
    if SLIDING_WINDOW > 0:
        window_min = offs_m + (seqlen_k - seqlen_q - SLIDING_WINDOW)
    else:
        window_min = None

    # Classify key blocks: full (no boundary/causal mask) vs masked.
    n_blocks = gl.cdiv(seqlen_k, BLOCK_N)
    if IS_CAUSAL:
        n_blocks_causal = gl.cdiv(
            (start_m + 1) * BLOCK_M + seqlen_k - seqlen_q, BLOCK_N
        )
        n_blocks = min(n_blocks, n_blocks_causal)

        if n_blocks <= 0:
            storeLayout: gl.constexpr = qLoadLayout
            offs_od = gl.arange(
                0, BLOCK_DMODEL_POW2, layout=gl.SliceLayout(0, storeLayout)
            )
            offs_rm = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, storeLayout))
            offs_om = start_m * BLOCK_M + offs_rm
            o_base = (
                o_ptr
                + off_z * stride_oz
                + off_q_head * stride_oh
                + cu_seqlens_q_start * stride_om
                + start_m * BLOCK_M * stride_om
            )
            o_offsets = (
                offs_rm[:, None] * stride_om + offs_od[None, :] * stride_on
            ).to(gl.int32)
            zeros = gl.zeros(
                [BLOCK_M, BLOCK_DMODEL_POW2],
                dtype=o_ptr.dtype.element_ty,
                layout=storeLayout,
            )
            o_mask = offs_om[:, None] < seqlen_q
            if PADDED_HEAD:
                o_mask = o_mask & (offs_od[None, :] < BLOCK_DMODEL)
            gl.amd.cdna4.buffer_store(zeros, ptr=o_base, offsets=o_offsets, mask=o_mask)
            return

    n_extra_tokens = 0
    if seqlen_k < BLOCK_N:
        n_extra_tokens = BLOCK_N - seqlen_k
    elif seqlen_k % BLOCK_N:
        n_extra_tokens = seqlen_k % BLOCK_N
    padded_block_k = n_extra_tokens != 0
    is_modulo_mn = (not padded_block_k) and (seqlen_q % BLOCK_M == 0)

    # Skip K blocks that are fully left of the earliest key position
    # reachable by this Q block. The first retained block can still be
    # partially outside the window, so we keep the per-element mask below.
    skipped_blocks = 0
    if SLIDING_WINDOW > 0:
        window_start_n = start_m * BLOCK_M + seqlen_k - seqlen_q - SLIDING_WINDOW
        skipped_blocks = min(max(window_start_n, 0) // BLOCK_N, n_blocks)

    if IS_CAUSAL:
        # There are always at least BLOCK_M // BLOCK_N masked blocks.
        # Additionally there might be one more due to dissimilar seqlens.
        masked_blocks = BLOCK_M // BLOCK_N + (not is_modulo_mn)
    else:
        masked_blocks = padded_block_k

    # if IS_CAUSAL, not is_modulo_mn does not always result in an additional block.
    # In this case we might exceed n_blocks so pick the min.
    visible_blocks = n_blocks - skipped_blocks
    masked_blocks = min(masked_blocks, visible_blocks)
    n_full_blocks = visible_blocks - masked_blocks
    block_min = skipped_blocks * BLOCK_N
    block_max = n_blocks * BLOCK_N

    if SLIDING_WINDOW > 0:
        # k_base also anchors the PE slice, which shares K's base pointer.
        k_base += skipped_blocks * BLOCK_N * stride_kn
        v_base += skipped_blocks * BLOCK_N * stride_vn

    # Full blocks: no boundary mask, no causal mask.
    if n_full_blocks > 0:
        block_max = block_min + n_full_blocks * BLOCK_N
        acc, l_i, m_i = _attn_fwd_inner(
            acc,
            l_i,
            m_i,
            q,
            q_pe,
            k_base,
            k_offsets,
            k_pe_offsets,
            v_base,
            v_offsets,
            smemK,
            smemKpe,
            smemV,
            stride_kn,
            stride_vn,
            seqlen_k,
            block_min,
            block_max,
            window_min,
            qk_scale,
            descale_v,
            mfmaLayout=mfmaLayout,
            dotK=dotK,
            dotP=dotP,
            dotV=dotV,
            kLoadLayout=kLoadLayout,
            kPeLoadLayout=kPeLoadLayout,
            vLoadLayout=vLoadLayout,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_DMODEL=BLOCK_DMODEL,
            BLOCK_DMODEL_POW2=BLOCK_DMODEL_POW2,
            BLOCK_DMODEL_PE=BLOCK_DMODEL_PE,
            HAS_PE=HAS_PE,
            NUM_KV_BUFFERS=NUM_KV_BUFFERS,
            IS_FP8=IS_FP8,
            FP8_MAX=FP8_MAX,
            SLIDING_WINDOW=SLIDING_WINDOW,
        )
        block_min = block_max
        block_max = n_blocks * BLOCK_N

    # Remaining blocks carry the boundary / causal masking (non-pipelined path).
    if masked_blocks > 0:
        k_base += n_full_blocks * BLOCK_N * stride_kn
        v_base += n_full_blocks * BLOCK_N * stride_vn
        acc, l_i, m_i = _attn_fwd_inner_masked(
            acc,
            l_i,
            m_i,
            q,
            q_pe,
            k_base,
            k_offsets,
            k_pe_offsets,
            v_base,
            v_offsets,
            smemK,
            smemKpe,
            smemV,
            stride_kn,
            stride_vn,
            seqlen_q,
            seqlen_k,
            block_min,
            block_max,
            n_extra_tokens,
            offs_m,
            window_min,
            qk_scale,
            descale_v,
            mfmaLayout=mfmaLayout,
            dotK=dotK,
            dotP=dotP,
            dotV=dotV,
            kLoadLayout=kLoadLayout,
            kPeLoadLayout=kPeLoadLayout,
            vLoadLayout=vLoadLayout,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_DMODEL=BLOCK_DMODEL,
            BLOCK_DMODEL_POW2=BLOCK_DMODEL_POW2,
            BLOCK_DMODEL_PE=BLOCK_DMODEL_PE,
            HAS_PE=HAS_PE,
            IS_CAUSAL=IS_CAUSAL,
            MASK_STEPS=True,
            IS_FP8=IS_FP8,
            FP8_MAX=FP8_MAX,
            SLIDING_WINDOW=SLIDING_WINDOW,
        )

    # epilogue: normalize and write
    acc = acc / l_i[:, None]

    # If seqlen_q > seqlen_k but the delta is not a multiple of BLOCK_M,
    # then we have one block with a row of all NaNs which come from computing
    # softmax over a row of all -infs (-inf - inf = NaN). We check for that here
    # and store 0s where there are NaNs as these rows should've been zeroed out.
    end_m_idx = (start_m + 1) * BLOCK_M
    start_m_idx = start_m * BLOCK_M
    causal_start_idx = seqlen_q - seqlen_k
    if IS_CAUSAL:  # noqa: SIM102
        if (causal_start_idx > start_m_idx) and (causal_start_idx < end_m_idx):
            out_mask_boundary = gl.full(
                [BLOCK_DMODEL_POW2],
                causal_start_idx,
                dtype=gl.int32,
                layout=gl.SliceLayout(0, mfmaLayout),
            )
            mask_m_offsets = start_m_idx + gl.arange(
                0, BLOCK_M, layout=gl.SliceLayout(1, mfmaLayout)
            )
            out_ptrs_mask = mask_m_offsets[:, None] >= out_mask_boundary[None, :]
            acc = gl.where(out_ptrs_mask, acc, 0.0)

    out = acc.to(o_ptr.dtype.element_ty)

    storeLayout: gl.constexpr = qLoadLayout
    out = gl.convert_layout(out, layout=storeLayout)

    offs_rm = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, storeLayout))
    offs_om = start_m * BLOCK_M + offs_rm
    offs_od = gl.arange(0, BLOCK_DMODEL_POW2, layout=gl.SliceLayout(0, storeLayout))

    o_base = (
        o_ptr
        + off_z * stride_oz
        + off_q_head * stride_oh
        + cu_seqlens_q_start * stride_om
        + start_m * BLOCK_M * stride_om
    )
    o_offsets = (offs_rm[:, None] * stride_om + offs_od[None, :] * stride_on).to(
        gl.int32
    )

    overflow_size = end_m_idx - seqlen_q
    out_mask = gl.full([BLOCK_M, 1], True, dtype=gl.int1, layout=storeLayout)
    if overflow_size > 0:
        out_mask = out_mask & (offs_om[:, None] < seqlen_q)
    if PADDED_HEAD:
        out_mask = out_mask & (offs_od[None, :] < BLOCK_DMODEL)
    gl.amd.cdna4.buffer_store(out, ptr=o_base, offsets=o_offsets, mask=out_mask)


def _get_config(is_fp8: bool, has_pe: bool = False):
    if not hasattr(_get_config, "_config_dict"):
        dev = arch_info.get_arch()
        fpath = f"{AITER_TRITON_CONFIGS_PATH}/{dev}-MHA-GLUON.json"
        with open(fpath, "r") as file:
            _get_config._config_dict = json.load(file)
    fwd_cfg = _get_config._config_dict["fwd"]
    # TODO: configs are not tuned
    if is_fp8:
        return fwd_cfg["fp8"]
    elif has_pe:
        return fwd_cfg["pe"]
    else:
        return fwd_cfg["default"]
