# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.amd.cdna4 import async_copy as cdna4_async
from triton.language.extra.libdevice import fast_dividef

from aiter.ops.triton._gluon_kernels.gfx950.attention.mha import (
    _kv_async_copy_layouts,
    _make_kv_shared_layouts,
    _stride,
)
from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr

_mha_varlen_fwd_fp8_repr = make_kernel_repr(
    "_mha_varlen_fwd_fp8",
    [
        "CAUSAL",
        "NUM_Q_HEADS",
        "NUM_K_HEADS",
        "BLOCK_M",
        "BLOCK_N",
        "HEAD_DIM_NOPE",
        "HEAD_DIM_PE",
        "HEAD_DIM_V",
        "RETURN_LSE",
        "INT64_OFFSETS",
    ],
)
_mha_varlen_fp8_combine_repr = make_kernel_repr(
    "_mha_varlen_fp8_combine",
    ["NUM_SPLITS", "HEAD_DIM_V", "RETURN_LSE", "INT64_OFFSETS"],
)

# P is lifted by log2(448) so its row max lands on the e4m3 max. exp2 can round a hair
# above 448; that is safe because the gfx950 fp32 -> e4m3fn conversion saturates.
LOG2_FP8_MAX = gl.constexpr(8.807354922057604)


@gluon.jit
def fast_maximum(a, b):
    return gl.maximum(a, b, propagate_nan=tl.PropagateNan.ALL)


@gluon.constexpr_function
def blocked_load_layout(head_dim, load_vec, num_warps, transposed, lanes=64):
    warp_elems = lanes * load_vec
    if transposed:
        return gl.BlockedLayout(
            [load_vec, 1],
            [head_dim // load_vec, warp_elems // head_dim],
            [1, num_warps],
            [0, 1],
        )
    return gl.BlockedLayout(
        [1, load_vec],
        [warp_elems // head_dim, head_dim // load_vec],
        [num_warps, 1],
        [1, 0],
    )


@gluon.jit
def async_load_k(
    k_buffer,
    k_pe_buffer,
    k_ptr,
    k_offs,
    k_pe_offs,
    k_offs_n,
    k_pe_offs_n,
    write_idx,
    start_n,
    seqlen_k,
    stride_k_n,
    MASK: gl.constexpr,
):
    k_ptr += start_n * stride_k_n
    if MASK:
        cdna4_async.buffer_load_to_shared(
            k_buffer.index(write_idx % 2),
            k_ptr,
            k_offs,
            mask=(k_offs_n < seqlen_k - start_n)[None, :],
        )
        cdna4_async.buffer_load_to_shared(
            k_pe_buffer.index(write_idx % 2),
            k_ptr,
            k_pe_offs,
            mask=(k_pe_offs_n < seqlen_k - start_n)[None, :],
        )
    else:
        cdna4_async.buffer_load_to_shared(k_buffer.index(write_idx % 2), k_ptr, k_offs)
        cdna4_async.buffer_load_to_shared(
            k_pe_buffer.index(write_idx % 2), k_ptr, k_pe_offs
        )
    cdna4_async.commit_group()


@gluon.jit
def async_load_v(
    v_buffer,
    v_ptr,
    v_offs,
    v_offs_n,
    write_idx,
    start_n,
    seqlen_k,
    stride_v_n,
    MASK: gl.constexpr,
):
    v_ptr += start_n * stride_v_n
    if MASK:
        cdna4_async.buffer_load_to_shared(
            v_buffer.index(write_idx % 2),
            v_ptr,
            v_offs,
            mask=(v_offs_n < seqlen_k - start_n)[:, None],
        )
    else:
        cdna4_async.buffer_load_to_shared(v_buffer.index(write_idx % 2), v_ptr, v_offs)
    cdna4_async.commit_group()


@gluon.jit
def local_load_k(k_buffer, k_pe_buffer, read_idx, layout):
    k = cdna4_async.load_shared_relaxed(k_buffer.index(read_idx % 2), layout)
    k_pe = cdna4_async.load_shared_relaxed(k_pe_buffer.index(read_idx % 2), layout)
    return k, k_pe


@gluon.jit
def local_load_v(v_buffer, read_idx, layout):
    return cdna4_async.load_shared_relaxed(v_buffer.index(read_idx % 2), layout)


@gluon.jit
def apply_mask(
    qk,
    start_n,
    offs_m,
    seqlen_k,
    causal_offset,
    BLOCK_N,
    LAYOUT,
    CAUSAL,
):
    offs_n = start_n + gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, LAYOUT))
    mask = offs_n[None, :] < seqlen_k
    if CAUSAL:
        mask = mask & ((offs_m + causal_offset)[:, None] >= offs_n[None, :])
    qk = tl.where(mask, qk, float("-inf"))
    return qk


@gluon.jit
def attn_fwd_tile(
    acc,
    l_i,
    m_i,
    q,
    q_pe,
    k,
    k_pe,
    v,
    start_n,
    offs_m,
    seqlen_k,
    causal_offset,
    qk_scale,
    MFMA_LAYOUT: gl.constexpr,
    DOT_LAYOUT_P: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    CAUSAL: gl.constexpr,
    MASK: gl.constexpr,
):
    qk = gl.zeros([BLOCK_M, BLOCK_N], dtype=gl.float32, layout=MFMA_LAYOUT)
    qk = gl.amd.cdna4.mfma_scaled(q_pe, None, "e4m3", k_pe, None, "e4m3", qk)
    qk = gl.amd.cdna4.mfma_scaled(q, None, "e4m3", k, None, "e4m3", qk)
    if MASK:
        qk = apply_mask(
            qk, start_n, offs_m, seqlen_k, causal_offset, BLOCK_N, MFMA_LAYOUT, CAUSAL
        )
    m_ij = fast_maximum(m_i, gl.reduce(qk, 1, fast_maximum) * qk_scale)
    # A row with no visible key yet keeps m_ij = -inf; shift it by 0 so p and alpha are 0, not NaN.
    m_shift = gl.where(m_ij == float("-inf"), 0.0, m_ij)
    qk_shifted = gl.fma(qk, qk_scale, -(m_shift - LOG2_FP8_MAX)[:, None])
    p = gl.exp2(qk_shifted)
    alpha = gl.exp2(m_i - m_shift)
    l_ij = gl.sum(p, 1)
    l_i = gl.fma(l_i, alpha, l_ij)
    acc = acc * alpha[:, None]
    p_quant = p.to(v.dtype)
    p_quant = gl.convert_layout(p_quant, layout=DOT_LAYOUT_P, assert_trivial=True)
    acc = gl.amd.cdna4.mfma_scaled(p_quant, None, "e4m3", v, None, "e4m3", acc)
    return acc, l_i, m_ij


@gluon.jit
def attn_fwd_inner(
    acc,
    l_i,
    m_i,
    q,
    q_pe,
    k_ptr,
    v_ptr,
    k_offs,
    k_pe_offs,
    v_offs,
    k_offs_n,
    k_pe_offs_n,
    v_offs_n,
    k_buffer,
    k_pe_buffer,
    v_buffer,
    stride_k_n,
    stride_v_n,
    seqlen_k,
    causal_offset,
    tile_min,
    tile_max,
    offs_m,
    qk_scale,
    MFMA_LAYOUT: gl.constexpr,
    DOT_LAYOUT_K: gl.constexpr,
    DOT_LAYOUT_P: gl.constexpr,
    DOT_LAYOUT_V: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    CAUSAL: gl.constexpr,
    MASK: gl.constexpr,
):
    num_n_iter = (tile_max - tile_min) // BLOCK_N
    async_load_k(
        k_buffer,
        k_pe_buffer,
        k_ptr,
        k_offs,
        k_pe_offs,
        k_offs_n,
        k_pe_offs_n,
        0,
        tile_min,
        seqlen_k,
        stride_k_n,
        MASK,
    )
    async_load_v(
        v_buffer, v_ptr, v_offs, v_offs_n, 0, tile_min, seqlen_k, stride_v_n, MASK
    )
    for i in range(num_n_iter):
        start_n = tile_min + i * BLOCK_N
        next_n = gl.minimum(start_n + BLOCK_N, tile_max - BLOCK_N)
        async_load_k(
            k_buffer,
            k_pe_buffer,
            k_ptr,
            k_offs,
            k_pe_offs,
            k_offs_n,
            k_pe_offs_n,
            i + 1,
            next_n,
            seqlen_k,
            stride_k_n,
            MASK,
        )
        async_load_v(
            v_buffer, v_ptr, v_offs, v_offs_n, i + 1, next_n, seqlen_k, stride_v_n, MASK
        )
        cdna4_async.wait_group(2)
        k, k_pe = local_load_k(k_buffer, k_pe_buffer, i, DOT_LAYOUT_K)
        v = local_load_v(v_buffer, i, DOT_LAYOUT_V)
        acc, l_i, m_i = attn_fwd_tile(
            acc,
            l_i,
            m_i,
            q,
            q_pe,
            k,
            k_pe,
            v,
            start_n,
            offs_m,
            seqlen_k,
            causal_offset,
            qk_scale,
            MFMA_LAYOUT,
            DOT_LAYOUT_P,
            BLOCK_M,
            BLOCK_N,
            CAUSAL,
            MASK,
        )
        gl.barrier()
    cdna4_async.wait_group(0)
    return acc, l_i, m_i


@gluon.jit(
    repr=_mha_varlen_fwd_fp8_repr, do_not_specialize=["MAX_SEQLEN_Q", "num_splits"]
)
def _mha_varlen_fwd_fp8(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    lse_ptr,
    q_scale_ptr,
    k_scale_ptr,
    v_scale_ptr,
    cu_seqlens_q,
    cu_seqlens_k,
    stride_q_m,
    stride_q_h,
    stride_k_n,
    stride_k_h,
    stride_v_n,
    stride_v_h,
    stride_o_m,
    stride_o_h,
    stride_lse_h,
    stride_o_s,
    stride_lse_s,
    SM_SCALE: gl.constexpr,
    CAUSAL: gl.constexpr,
    NUM_Q_HEADS: gl.constexpr,
    NUM_K_HEADS: gl.constexpr,
    MAX_SEQLEN_Q,
    num_splits,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    HEAD_DIM_NOPE: gl.constexpr,
    HEAD_DIM_PE: gl.constexpr,
    HEAD_DIM_V: gl.constexpr,
    RETURN_LSE: gl.constexpr,
    INT64_OFFSETS: gl.constexpr,
    num_warps: gl.constexpr,
):
    """FP8 MHA forward over a varlen batch.

    The batch is varlen, not batched heads: the tokens of all sequences are packed
    along one dimension, and cu_seqlens_q / cu_seqlens_k give the start of each
    sequence. q and k are [total_tokens, H, HEAD_DIM_NOPE + HEAD_DIM_PE], v is
    [total_tokens, H, HEAD_DIM_V], and o is [total_tokens, H, HEAD_DIM_V]. There is no
    [B, H, S, D] layout and no padding to a common sequence length.

    One program computes one BLOCK_M block of queries of one sequence, for one head
    and one key split. The causal mask is bottom-right aligned. With num_splits > 1,
    each split writes a partial output and its LSE, and the combine kernel merges
    them. The LSE is natural log, at lse_ptr[h * stride_lse_h + token].
    """

    MFMA_LAYOUT: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=[32, 32, 64],
        transposed=True,
        warps_per_cta=[num_warps, 1],
    )

    DOT_LAYOUT_Q: gl.constexpr = gl.DotOperandLayout(0, MFMA_LAYOUT, k_width=16)
    DOT_LAYOUT_K: gl.constexpr = gl.DotOperandLayout(1, MFMA_LAYOUT, k_width=16)
    DOT_LAYOUT_P: gl.constexpr = gl.DotOperandLayout(0, MFMA_LAYOUT, k_width=4)
    DOT_LAYOUT_V: gl.constexpr = gl.DotOperandLayout(1, MFMA_LAYOUT, k_width=4)

    LOAD_LAYOUT_Q: gl.constexpr = blocked_load_layout(
        HEAD_DIM_NOPE, 16, num_warps, False
    )
    LOAD_LAYOUT_Q_PE: gl.constexpr = blocked_load_layout(
        HEAD_DIM_PE, 16, num_warps, False
    )
    ASYNC_LAYOUT_K: gl.constexpr = _kv_async_copy_layouts(
        HEAD_DIM_NOPE, BLOCK_N, num_warps, 8, 16
    )[0]
    ASYNC_LAYOUT_K_PE: gl.constexpr = _kv_async_copy_layouts(
        HEAD_DIM_PE, BLOCK_N, num_warps, 8, 16
    )[0]
    ASYNC_LAYOUT_V: gl.constexpr = _kv_async_copy_layouts(
        HEAD_DIM_V, BLOCK_N, num_warps, 8, 16
    )[1]
    STORE_LAYOUT_O: gl.constexpr = blocked_load_layout(HEAD_DIM_V, 8, num_warps, False)

    SHARED_LAYOUT_K: gl.constexpr = _make_kv_shared_layouts(HEAD_DIM_NOPE, BLOCK_N, 8)[
        0
    ]
    SHARED_LAYOUT_K_PE: gl.constexpr = _make_kv_shared_layouts(HEAD_DIM_PE, BLOCK_N, 8)[
        0
    ]
    SHARED_LAYOUT_V: gl.constexpr = _make_kv_shared_layouts(HEAD_DIM_V, BLOCK_N, 8)[1]

    num_blocks_m = gl.cdiv(MAX_SEQLEN_Q, BLOCK_M)
    pid = gl.program_id(0)
    split = pid % num_splits
    pid = pid // num_splits
    h_q = pid % NUM_Q_HEADS
    pid_m = (pid // NUM_Q_HEADS) % num_blocks_m
    if CAUSAL:
        pid_m = num_blocks_m - 1 - pid_m
    z = pid // (NUM_Q_HEADS * num_blocks_m)
    GQA_SIZE: gl.constexpr = NUM_Q_HEADS // NUM_K_HEADS
    h_k = h_q // GQA_SIZE

    q_start = gl.load(cu_seqlens_q + z)
    seqlen_q = gl.load(cu_seqlens_q + z + 1) - q_start
    if pid_m * BLOCK_M >= seqlen_q:
        return
    k_start = gl.load(cu_seqlens_k + z)
    seqlen_k = gl.load(cu_seqlens_k + z + 1) - k_start
    causal_offset = seqlen_k - seqlen_q

    qk_scale_const: gl.constexpr = SM_SCALE * 1.44269504

    q_scale = gl.load(q_scale_ptr)
    k_scale = gl.load(k_scale_ptr)
    qk_scale = q_scale * k_scale * qk_scale_const

    # Base offsets (token * stride) use int64 strides when the wrapper finds that a tensor
    # is too large for int32. The per-tile offsets below stay int32.
    base_stride_q_m = _stride(stride_q_m, INT64_OFFSETS)
    base_stride_k_n = _stride(stride_k_n, INT64_OFFSETS)
    base_stride_v_n = _stride(stride_v_n, INT64_OFFSETS)
    base_stride_o_m = _stride(stride_o_m, INT64_OFFSETS)
    base_stride_o_s = _stride(stride_o_s, INT64_OFFSETS)
    base_stride_lse_s = _stride(stride_lse_s, INT64_OFFSETS)
    q_row = q_start + pid_m * BLOCK_M
    h_q_off = _stride(h_q, INT64_OFFSETS)
    h_k_off = _stride(h_k, INT64_OFFSETS)
    q_ptr += h_q_off * stride_q_h + q_row * base_stride_q_m
    k_ptr += h_k_off * stride_k_h + k_start * base_stride_k_n
    v_ptr += h_k_off * stride_v_h + k_start * base_stride_v_n
    o_ptr += split * base_stride_o_s + h_q_off * stride_o_h + q_row * base_stride_o_m

    offs_q_m = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, LOAD_LAYOUT_Q))
    offs_q_d = gl.arange(0, HEAD_DIM_NOPE, layout=gl.SliceLayout(0, LOAD_LAYOUT_Q))
    q = gl.amd.cdna4.buffer_load(
        ptr=q_ptr,
        offsets=offs_q_m[:, None] * stride_q_m + offs_q_d[None, :],
        mask=((pid_m * BLOCK_M + offs_q_m) < seqlen_q)[:, None],
        other=0.0,
    )
    q = gl.convert_layout(q, layout=DOT_LAYOUT_Q)
    offs_q_pe_m = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, LOAD_LAYOUT_Q_PE))
    offs_q_pe_d = HEAD_DIM_NOPE + gl.arange(
        0, HEAD_DIM_PE, layout=gl.SliceLayout(0, LOAD_LAYOUT_Q_PE)
    )
    q_pe = gl.amd.cdna4.buffer_load(
        ptr=q_ptr,
        offsets=offs_q_pe_m[:, None] * stride_q_m + offs_q_pe_d[None, :],
        mask=((pid_m * BLOCK_M + offs_q_pe_m) < seqlen_q)[:, None],
        other=0.0,
    )
    q_pe = gl.convert_layout(q_pe, layout=DOT_LAYOUT_Q)

    k_offs_d = gl.arange(0, HEAD_DIM_NOPE, layout=gl.SliceLayout(1, ASYNC_LAYOUT_K))
    k_offs_n = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, ASYNC_LAYOUT_K))
    k_offs = k_offs_d[:, None] + k_offs_n[None, :] * stride_k_n
    k_pe_offs_d = HEAD_DIM_NOPE + gl.arange(
        0, HEAD_DIM_PE, layout=gl.SliceLayout(1, ASYNC_LAYOUT_K_PE)
    )
    k_pe_offs_n = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, ASYNC_LAYOUT_K_PE))
    k_pe_offs = k_pe_offs_d[:, None] + k_pe_offs_n[None, :] * stride_k_n
    v_offs_n = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(1, ASYNC_LAYOUT_V))
    v_offs_d = gl.arange(0, HEAD_DIM_V, layout=gl.SliceLayout(0, ASYNC_LAYOUT_V))
    v_offs = v_offs_n[:, None] * stride_v_n + v_offs_d[None, :]

    k_buffer = gl.allocate_shared_memory(
        k_ptr.dtype.element_ty, [2, HEAD_DIM_NOPE, BLOCK_N], SHARED_LAYOUT_K
    )
    k_pe_buffer = gl.allocate_shared_memory(
        k_ptr.dtype.element_ty, [2, HEAD_DIM_PE, BLOCK_N], SHARED_LAYOUT_K_PE
    )
    v_buffer = gl.allocate_shared_memory(
        v_ptr.dtype.element_ty, [2, BLOCK_N, HEAD_DIM_V], SHARED_LAYOUT_V
    )

    acc = gl.zeros((BLOCK_M, HEAD_DIM_V), dtype=gl.float32, layout=MFMA_LAYOUT)
    m_i = gl.full(
        [BLOCK_M],
        float("-inf"),
        dtype=gl.float32,
        layout=gl.SliceLayout(1, MFMA_LAYOUT),
    )
    l_i = gl.zeros([BLOCK_M], dtype=gl.float32, layout=gl.SliceLayout(1, MFMA_LAYOUT))
    offs_m = pid_m * BLOCK_M + gl.arange(
        0, BLOCK_M, layout=gl.SliceLayout(1, MFMA_LAYOUT)
    )

    n_tiles = gl.cdiv(seqlen_k, BLOCK_N)
    tile_lo = n_tiles * 0
    if CAUSAL:
        n_tiles = gl.minimum(
            n_tiles, gl.cdiv((pid_m + 1) * BLOCK_M + causal_offset, BLOCK_N)
        )
        n_full = gl.maximum(
            gl.minimum((pid_m * BLOCK_M + causal_offset) // BLOCK_N, n_tiles), 0
        )
    else:
        n_full = seqlen_k // BLOCK_N
        if num_splits > 1:
            per_split = gl.cdiv(n_tiles, num_splits)
            tile_lo = gl.minimum(split * per_split, n_tiles)
            n_tiles = gl.minimum(tile_lo + per_split, n_tiles)
            n_full = gl.minimum(n_full, n_tiles)
    if n_full > tile_lo:
        acc, l_i, m_i = attn_fwd_inner(
            acc,
            l_i,
            m_i,
            q,
            q_pe,
            k_ptr,
            v_ptr,
            k_offs,
            k_pe_offs,
            v_offs,
            k_offs_n,
            k_pe_offs_n,
            v_offs_n,
            k_buffer,
            k_pe_buffer,
            v_buffer,
            base_stride_k_n,
            base_stride_v_n,
            seqlen_k,
            causal_offset,
            tile_lo * BLOCK_N,
            n_full * BLOCK_N,
            offs_m,
            qk_scale,
            MFMA_LAYOUT,
            DOT_LAYOUT_K,
            DOT_LAYOUT_P,
            DOT_LAYOUT_V,
            BLOCK_M,
            BLOCK_N,
            CAUSAL,
            False,
        )
    masked_lo = gl.maximum(n_full, tile_lo)
    if n_tiles > masked_lo:
        acc, l_i, m_i = attn_fwd_inner(
            acc,
            l_i,
            m_i,
            q,
            q_pe,
            k_ptr,
            v_ptr,
            k_offs,
            k_pe_offs,
            v_offs,
            k_offs_n,
            k_pe_offs_n,
            v_offs_n,
            k_buffer,
            k_pe_buffer,
            v_buffer,
            base_stride_k_n,
            base_stride_v_n,
            seqlen_k,
            causal_offset,
            masked_lo * BLOCK_N,
            n_tiles * BLOCK_N,
            offs_m,
            qk_scale,
            MFMA_LAYOUT,
            DOT_LAYOUT_K,
            DOT_LAYOUT_P,
            DOT_LAYOUT_V,
            BLOCK_M,
            BLOCK_N,
            CAUSAL,
            True,
        )

    l_safe = gl.where(l_i > 0.0, l_i, 1.0)
    v_scale = gl.load(v_scale_ptr)
    ep_norm = fast_dividef(v_scale, l_safe)
    acc = acc * ep_norm[:, None]
    if RETURN_LSE:
        LN2: gl.constexpr = 0.6931471824645996
        lse = gl.where(
            l_i > 0.0, (m_i + gl.log2(l_safe) - LOG2_FP8_MAX) * LN2, float("-inf")
        )
        lse_base = (
            lse_ptr
            + split * base_stride_lse_s
            + h_q_off * stride_lse_h
            + _stride(q_start, INT64_OFFSETS)
        )
        gl.amd.cdna4.buffer_store(
            lse,
            ptr=lse_base,
            offsets=offs_m,
            mask=offs_m < seqlen_q,
        )
    acc = gl.convert_layout(acc.to(o_ptr.dtype.element_ty), layout=STORE_LAYOUT_O)
    offs_o_m = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, STORE_LAYOUT_O))
    offs_o_d = gl.arange(0, HEAD_DIM_V, layout=gl.SliceLayout(0, STORE_LAYOUT_O))
    gl.amd.cdna4.buffer_store(
        acc,
        ptr=o_ptr,
        offsets=offs_o_m[:, None] * stride_o_m + offs_o_d[None, :],
        mask=((pid_m * BLOCK_M + offs_o_m) < seqlen_q)[:, None],
    )


@gluon.jit(repr=_mha_varlen_fp8_combine_repr)
def _mha_varlen_fp8_combine(
    o_part_ptr,
    lse_part_ptr,
    o_ptr,
    lse_ptr,
    cu_seqlens_q,
    batch,
    stride_o_part_s,
    stride_o_part_m,
    stride_o_part_h,
    stride_lse_part_s,
    stride_lse_part_h,
    stride_o_m,
    stride_o_h,
    stride_lse_h,
    NUM_Q_HEADS: gl.constexpr,
    NUM_SPLITS: gl.constexpr,
    HEAD_DIM_V: gl.constexpr,
    RETURN_LSE: gl.constexpr,
    INT64_OFFSETS: gl.constexpr,
):
    LAYOUT: gl.constexpr = gl.BlockedLayout(
        [1, HEAD_DIM_V // 64], [1, 64], [1, 1], [1, 0]
    )
    pid = gl.program_id(0)
    m = pid // NUM_Q_HEADS
    h_q = pid % NUM_Q_HEADS
    # The forward kernel writes no partials for rows outside the packed sequences.
    if (m < gl.load(cu_seqlens_q)) | (m >= gl.load(cu_seqlens_q + batch)):
        return
    m = _stride(m, INT64_OFFSETS)
    h_q = _stride(h_q, INT64_OFFSETS)
    offs_s = gl.arange(0, NUM_SPLITS, layout=gl.SliceLayout(1, LAYOUT))
    offs_d = gl.arange(0, HEAD_DIM_V, layout=gl.SliceLayout(0, LAYOUT))
    lse_s = gl.load(
        lse_part_ptr + offs_s * stride_lse_part_s + h_q * stride_lse_part_h + m
    )
    lse_max = gl.max(lse_s, 0)
    lse_max = gl.where(lse_max == float("-inf"), 0.0, lse_max)
    w = gl.exp(lse_s - lse_max)
    w_sum = gl.sum(w, 0)
    w = w / gl.where(w_sum > 0.0, w_sum, 1.0)
    o_s = gl.load(
        o_part_ptr
        + _stride(offs_s, INT64_OFFSETS)[:, None] * stride_o_part_s
        + m * stride_o_part_m
        + h_q * stride_o_part_h
        + offs_d[None, :]
    )
    acc = gl.sum(o_s * w[:, None], 0)
    gl.store(
        o_ptr + m * stride_o_m + h_q * stride_o_h + offs_d,
        acc.to(o_ptr.dtype.element_ty),
    )
    if RETURN_LSE:
        lse = gl.where(w_sum > 0.0, lse_max + gl.log(w_sum), float("-inf"))
        gl.store(lse_ptr + h_q * stride_lse_h + m, lse)
