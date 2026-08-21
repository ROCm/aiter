# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Kernels for the DeepSeek-V4 sparse-MLA training BACKWARD (gfx950 / CDNA4).

All operate on the official V4 form (``K == V == kv``, one dense 512-wide tensor, RoPE already
applied in place caller-side, scale ``1/sqrt(512)``, ``attn_sink`` in the softmax denominator
only, ``topk == -1`` masked). The two MFMA phases are Gluon; the two memory-bound phases are
plain Triton and live here rather than under ``_triton_kernels/`` because there is no Triton
implementation of this backward to fall back to -- they are parts of this kernel, not an
alternative to it.

``_dq_v4_kernel``
    Per (query token, head block): ``S = Q@kv^T``, ``P = exp(S - lse)``, ``dP = dO@kv^T``,
    ``dS = P*(dP - delta)*scale``, ``dQ += dS@kv``. Three MFMAs per tile; the gathered KV tile
    is read from LDS once and feeds both the ``S`` and ``dP`` MFMAs. Also emits the ``dS`` / ``P``
    chunks the dKV-interm kernel consumes.

``_dkv_interm_v4_kernel``
    ``interm[t, slot, d] = sum_h ( dS[t,h,slot]*Q[t,h,d] + P[t,h,slot]*dO[t,h,d] )``, contracting
    over ALL heads inside one MFMA pair so nothing accumulates across a loop over heads. Q and dO
    are transposed once into registers and D is split across ``grid.y``, which is what keeps them
    read once instead of ``topk/TILE_K`` times.

``_delta_v4_kernel``
    ``delta = rowsum(O * dO)`` -- the standard flash-attention "o_dot_do" preamble. Streams the
    bf16 inputs and accumulates in fp32, so it moves exactly the working set.

``_bwd_dkv_gather_acc_v4`` + ``build_inverted_topk``
    Reduce ``interm[t, slot, :]`` into ``dkv[kv_row, :]`` over the top-k mapping. The scatter is
    inverted into a CSR gather (each output KV row collects its own contributors), so no atomics
    are needed. ``BLOCK_E`` entries are carried per loop iteration, which both widens the load
    and cuts the trip count on the long runs a realistic top-k produces.

Public entry: ``aiter.ops.triton.attention.sparse_attention_dsv4_bwd.sparse_mla_bwd_dsv4``.
"""

import torch
import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr

_dq_v4_kernel_repr = make_kernel_repr(
    "_dq_v4_kernel",
    [
        "R_CHUNK",
        "BLOCK_H",
        "TILE_K",
        "D",
        "IS_FIRST_CHUNK",
    ],
)


@gluon.jit(repr=_dq_v4_kernel_repr)
def _dq_v4_kernel(
    Q_ptr,  # [T, H, D] bf16
    KV_ptr,  # [T, D]    bf16   (K == V)
    dO_ptr,  # [T, H, D] bf16
    TopK_ptr,  # [T, TOPK_padded] int32
    LSE_ptr,  # [T, H] fp32 (sink-inclusive)
    Delta_ptr,  # [T, H] fp32
    dQ_ptr,  # [T, H, D] bf16 (RMW across chunks)
    dS_ptr,  # [T, H, R_CHUNK] bf16
    P_ptr,  # [T, H, R_CHUNK] bf16
    stride_q_t: tl.int64,
    stride_q_h: tl.int64,
    stride_kv_t: tl.int64,
    stride_do_t: tl.int64,
    stride_do_h: tl.int64,
    stride_dq_t: tl.int64,
    stride_dq_h: tl.int64,
    stride_topk_t: tl.int64,
    stride_ds_t: tl.int64,
    stride_ds_h: tl.int64,
    scale: tl.float32,
    num_heads: tl.int32,
    R_START: tl.int32,
    R_CHUNK: gl.constexpr,
    BLOCK_H: gl.constexpr,
    TILE_K: gl.constexpr,
    D: gl.constexpr,
    IS_FIRST_CHUNK: gl.constexpr,
):
    gl.static_assert(TILE_K % 32 == 0, "16x16x32 needs TILE_K multiple of 32")
    gl.static_assert(D % 32 == 0, "16x16x32 needs D multiple of 32")

    # ---- single 16x16x32 MFMA parent (score contracts D; accumulate contracts TILE_K) ----
    mma: gl.constexpr = gl.amd.cdna4.AMDMFMALayout(
        version=4,
        instr_shape=[16, 16, 32],
        transposed=True,
        warps_per_cta=[4, 1],
    )
    qa: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mma, k_width=8)
    qb: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=mma, k_width=8)

    # ---- blocked layouts for global loads ----
    _q_tpw_k: gl.constexpr = min(64, D // 8)
    _q_tpw_m: gl.constexpr = 64 // _q_tpw_k
    blk_q: gl.constexpr = gl.BlockedLayout(  # [BLOCK_H, D]  (Q, dO, dQ)
        size_per_thread=[1, 8],
        threads_per_warp=[_q_tpw_m, _q_tpw_k],
        warps_per_cta=[4, 1],
        order=[1, 0],
    )
    _kv_tpw_m: gl.constexpr = min(64, D // 8)
    _kv_tpw_n: gl.constexpr = 64 // _kv_tpw_m
    blk_kv: gl.constexpr = gl.BlockedLayout(  # [D, TILE_K]
        size_per_thread=[8, 1],
        threads_per_warp=[_kv_tpw_m, _kv_tpw_n],
        warps_per_cta=[1, 4],
        order=[0, 1],
    )
    sh_kv: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[512, 16]], [D, TILE_K], [0, 1]
    )

    # ---- program ids ----
    token_idx = gl.program_id(axis=0)
    hg_idx = gl.program_id(axis=1)
    hg_offset = hg_idx * BLOCK_H
    NUM_TILES: gl.constexpr = R_CHUNK // TILE_K

    # ---- Q / dO offsets + load (register, convert to dot operand) ----
    offs_h_q = hg_offset + gl.arange(0, BLOCK_H, layout=gl.SliceLayout(1, blk_q))
    offs_d_q = gl.arange(0, D, layout=gl.SliceLayout(0, blk_q))
    mask_h_q = offs_h_q < num_heads
    q_base = token_idx.to(tl.int64) * stride_q_t
    q_offs = (
        q_base
        + offs_h_q[:, None].to(tl.int64) * stride_q_h
        + offs_d_q[None, :].to(tl.int64)
    )
    do_base = token_idx.to(tl.int64) * stride_do_t
    do_offs = (
        do_base
        + offs_h_q[:, None].to(tl.int64) * stride_do_h
        + offs_d_q[None, :].to(tl.int64)
    )

    q_blk = gl.amd.cdna4.buffer_load(
        ptr=Q_ptr, offsets=q_offs.to(tl.int32), mask=mask_h_q[:, None], other=0.0
    )
    do_blk = gl.amd.cdna4.buffer_load(
        ptr=dO_ptr, offsets=do_offs.to(tl.int32), mask=mask_h_q[:, None], other=0.0
    )
    Q_dot = gl.convert_layout(q_blk, qa)
    dO_dot = gl.convert_layout(do_blk, qa)

    # ---- topk / KV offsets ----
    topk_base = token_idx.to(tl.int64) * stride_topk_t + R_START
    stride_kv_t_i32: tl.int32 = stride_kv_t.to(tl.int32)
    offs_tile_kv = gl.arange(0, TILE_K, layout=gl.SliceLayout(0, blk_kv))
    offs_tile_mma = gl.arange(0, TILE_K, layout=gl.SliceLayout(0, mma))
    offs_d_kv = gl.arange(0, D, layout=gl.SliceLayout(1, blk_kv))

    smem_kv = gl.allocate_shared_memory(
        KV_ptr.dtype.element_ty, [2, D, TILE_K], layout=sh_kv
    )

    dQ_acc = gl.zeros([BLOCK_H, D], dtype=gl.float32, layout=mma)

    offs_h_s = hg_offset + gl.arange(0, BLOCK_H, layout=gl.SliceLayout(1, mma))
    mask_h_s = offs_h_s < num_heads
    lse = gl.amd.cdna4.buffer_load(
        ptr=LSE_ptr,
        offsets=(token_idx * num_heads + offs_h_s).to(tl.int32),
        mask=mask_h_s,
        other=0.0,
    )
    delta = gl.amd.cdna4.buffer_load(
        ptr=Delta_ptr,
        offsets=(token_idx * num_heads + offs_h_s).to(tl.int32),
        mask=mask_h_s,
        other=0.0,
    )

    # ---- prologue: gather kv tile 0 ----
    topk_pos_kv = gl.amd.cdna4.buffer_load(
        ptr=TopK_ptr,
        offsets=(topk_base + offs_tile_kv).to(tl.int32),
        mask=offs_tile_kv < R_CHUNK,
        other=-1,
    )
    topk_pos_mma = gl.amd.cdna4.buffer_load(
        ptr=TopK_ptr,
        offsets=(topk_base + offs_tile_mma).to(tl.int32),
        mask=offs_tile_mma < R_CHUNK,
        other=-1,
    )
    valid_kv = topk_pos_kv != -1
    valid_mma = topk_pos_mma != -1
    safe_kv = gl.where(valid_kv, topk_pos_kv, 0)
    kv_offs = safe_kv[None, :] * stride_kv_t_i32 + offs_d_kv[:, None]
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        dest=smem_kv.index(0), ptr=KV_ptr, offsets=kv_offs, mask=valid_kv[None, :]
    )
    gl.amd.cdna4.async_copy.commit_group()

    ds_base = (
        token_idx.to(tl.int64) * stride_ds_t
        + hg_idx.to(tl.int64) * BLOCK_H * stride_ds_h
    )
    offs_h_dsp = gl.arange(0, BLOCK_H, layout=gl.SliceLayout(1, mma))
    offs_tile_dsp = gl.arange(0, TILE_K, layout=gl.SliceLayout(0, mma))
    mask_h_dsp = (hg_offset + offs_h_dsp) < num_heads

    cur_buf = 0
    for t in range(NUM_TILES - 1):
        next_offs_kv = (t + 1) * TILE_K + offs_tile_kv
        next_offs_mma = (t + 1) * TILE_K + offs_tile_mma
        topk_pos_kv_next = gl.amd.cdna4.buffer_load(
            ptr=TopK_ptr,
            offsets=(topk_base + next_offs_kv).to(tl.int32),
            mask=next_offs_kv < R_CHUNK,
            other=-1,
        )
        topk_pos_mma_next = gl.amd.cdna4.buffer_load(
            ptr=TopK_ptr,
            offsets=(topk_base + next_offs_mma).to(tl.int32),
            mask=next_offs_mma < R_CHUNK,
            other=-1,
        )
        valid_kv_next = (next_offs_kv < R_CHUNK) & (topk_pos_kv_next != -1)
        valid_mma_next = (next_offs_mma < R_CHUNK) & (topk_pos_mma_next != -1)
        safe_kv_next = gl.where(valid_kv_next, topk_pos_kv_next, 0)

        next_buf = 1 - cur_buf
        kv_offs_next = safe_kv_next[None, :] * stride_kv_t_i32 + offs_d_kv[:, None]
        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            dest=smem_kv.index(next_buf),
            ptr=KV_ptr,
            offsets=kv_offs_next,
            mask=valid_kv_next[None, :],
        )
        gl.amd.cdna4.async_copy.commit_group()

        gl.amd.cdna4.async_copy.wait_group(1)

        kv_smem_cur = smem_kv.index(cur_buf)
        # score K (direct); V (permuted) read LATE, before the accumulate
        K_T_dot = gl.amd.cdna4.async_copy.load_shared_relaxed(kv_smem_cur, qb)

        S = gl.amd.cdna4.mfma(
            Q_dot, K_T_dot, gl.zeros([BLOCK_H, TILE_K], dtype=gl.float32, layout=mma)
        )
        S = S * scale
        offs_h_mma = hg_offset + gl.arange(0, BLOCK_H, layout=gl.SliceLayout(1, mma))
        valid_mask = valid_mma[None, :] & (offs_h_mma < num_heads)[:, None]
        S = gl.where(valid_mask, S, float("-inf"))

        P = gl.exp(S - lse[:, None])
        P = gl.where(valid_mask, P, 0.0)
        dP = gl.amd.cdna4.mfma(
            dO_dot, K_T_dot, gl.zeros([BLOCK_H, TILE_K], dtype=gl.float32, layout=mma)
        )
        dS = P * (dP - delta[:, None]) * scale
        dS = gl.where(valid_mask, dS, 0.0)

        dS_bf = dS.to(KV_ptr.dtype.element_ty)
        dS_dot = gl.convert_layout(dS_bf, qa)
        K_v_dot = gl.amd.cdna4.async_copy.load_shared_relaxed(
            kv_smem_cur.permute([1, 0]), qb
        )  # load V LATE
        dQ_acc = gl.amd.cdna4.mfma(dS_dot, K_v_dot, dQ_acc)

        col = t * TILE_K + offs_tile_dsp
        dsp_offs = (
            ds_base
            + offs_h_dsp[:, None].to(tl.int64) * stride_ds_h
            + col[None, :].to(tl.int64)
        )
        gl.amd.cdna4.buffer_store(
            stored_value=dS_bf,
            ptr=dS_ptr,
            offsets=dsp_offs.to(tl.int32),
            mask=mask_h_dsp[:, None],
        )
        gl.amd.cdna4.buffer_store(
            stored_value=P.to(KV_ptr.dtype.element_ty),
            ptr=P_ptr,
            offsets=dsp_offs.to(tl.int32),
            mask=mask_h_dsp[:, None],
        )

        cur_buf = next_buf
        valid_mma = valid_mma_next

    # ---- epilogue: last tile ----
    gl.amd.cdna4.async_copy.wait_group(0)
    t = NUM_TILES - 1
    kv_smem_cur = smem_kv.index(cur_buf)
    K_T_dot = gl.amd.cdna4.async_copy.load_shared_relaxed(kv_smem_cur, qb)

    S = gl.amd.cdna4.mfma(
        Q_dot, K_T_dot, gl.zeros([BLOCK_H, TILE_K], dtype=gl.float32, layout=mma)
    )
    S = S * scale
    offs_h_mma = hg_offset + gl.arange(0, BLOCK_H, layout=gl.SliceLayout(1, mma))
    valid_mask = valid_mma[None, :] & (offs_h_mma < num_heads)[:, None]
    S = gl.where(valid_mask, S, float("-inf"))

    P = gl.exp(S - lse[:, None])
    P = gl.where(valid_mask, P, 0.0)
    dP = gl.amd.cdna4.mfma(
        dO_dot, K_T_dot, gl.zeros([BLOCK_H, TILE_K], dtype=gl.float32, layout=mma)
    )
    dS = P * (dP - delta[:, None]) * scale
    dS = gl.where(valid_mask, dS, 0.0)

    dS_bf = dS.to(KV_ptr.dtype.element_ty)
    dS_dot = gl.convert_layout(dS_bf, qa)
    K_v_dot = gl.amd.cdna4.async_copy.load_shared_relaxed(
        kv_smem_cur.permute([1, 0]), qb
    )
    dQ_acc = gl.amd.cdna4.mfma(dS_dot, K_v_dot, dQ_acc)

    col = t * TILE_K + offs_tile_dsp
    dsp_offs = (
        ds_base
        + offs_h_dsp[:, None].to(tl.int64) * stride_ds_h
        + col[None, :].to(tl.int64)
    )
    gl.amd.cdna4.buffer_store(
        stored_value=dS_bf,
        ptr=dS_ptr,
        offsets=dsp_offs.to(tl.int32),
        mask=mask_h_dsp[:, None],
    )
    gl.amd.cdna4.buffer_store(
        stored_value=P.to(KV_ptr.dtype.element_ty),
        ptr=P_ptr,
        offsets=dsp_offs.to(tl.int32),
        mask=mask_h_dsp[:, None],
    )

    # ---- store dQ (RMW across chunks) ----
    dq_base = token_idx.to(tl.int64) * stride_dq_t
    offs_h_o = hg_offset + gl.arange(0, BLOCK_H, layout=gl.SliceLayout(1, blk_q))
    offs_d_o = gl.arange(0, D, layout=gl.SliceLayout(0, blk_q))
    mask_h_o = offs_h_o < num_heads
    dq_offs = (
        dq_base
        + offs_h_o[:, None].to(tl.int64) * stride_dq_h
        + offs_d_o[None, :].to(tl.int64)
    )
    dq_blk = gl.convert_layout(dQ_acc.to(dQ_ptr.dtype.element_ty), blk_q)
    if not IS_FIRST_CHUNK:
        prev = gl.amd.cdna4.buffer_load(
            ptr=dQ_ptr, offsets=dq_offs.to(tl.int32), mask=mask_h_o[:, None], other=0.0
        )
        dq_blk = (dq_blk.to(gl.float32) + prev.to(gl.float32)).to(
            dQ_ptr.dtype.element_ty
        )
    gl.amd.cdna4.buffer_store(
        stored_value=dq_blk,
        ptr=dQ_ptr,
        offsets=dq_offs.to(tl.int32),
        mask=mask_h_o[:, None],
    )


def sparse_mla_bwd_dq(
    q,
    kv,
    do,
    topk,
    lse,
    delta,
    dq,
    chunk_dS,
    chunk_P,
    scale,
    r_start,
    R_CHUNK,
    BLOCK_H=64,
    TILE_K=32,
    is_first_chunk=True,
):
    """Launch the dQ kernel for one rank chunk. Writes ``dq`` (RMW when not the first chunk)
    plus this chunk's ``chunk_dS`` / ``chunk_P``."""
    T, H, D = q.shape
    _dq_v4_kernel[(T, triton.cdiv(H, BLOCK_H))](
        q,
        kv,
        do,
        topk,
        lse,
        delta,
        dq,
        chunk_dS,
        chunk_P,
        q.stride(0),
        q.stride(1),
        kv.stride(0),
        do.stride(0),
        do.stride(1),
        dq.stride(0),
        dq.stride(1),
        topk.stride(0),
        chunk_dS.stride(0),
        chunk_dS.stride(1),
        scale,
        H,
        r_start,
        R_CHUNK=R_CHUNK,
        BLOCK_H=BLOCK_H,
        TILE_K=TILE_K,
        D=D,
        IS_FIRST_CHUNK=is_first_chunk,
        num_warps=4,
        waves_per_eu=1,
    )


_dkv_interm_v4_kernel_repr = make_kernel_repr(
    "_dkv_interm_v4_kernel",
    [
        "R_CHUNK",
        "TILE_K",
        "NH",
        "BD",
        "D",
        "MFMA_K",
        "DUAL_STAGE",
    ],
)


@gluon.jit(repr=_dkv_interm_v4_kernel_repr)
def _dkv_interm_v4_kernel(
    Q_ptr,  # [T, H, D] bf16
    dO_ptr,  # [T, H, D] bf16
    dS_ptr,  # [T, H, R_CHUNK] bf16
    P_ptr,  # [T, H, R_CHUNK] bf16
    Interm_ptr,  # [T, R_CHUNK, D] bf16
    stride_q_t: tl.int64,
    stride_q_h: tl.int64,
    stride_do_t: tl.int64,
    stride_do_h: tl.int64,
    stride_ds_t: tl.int64,
    stride_ds_h: tl.int64,
    stride_interm_t: tl.int64,
    stride_interm_r: tl.int64,
    num_heads: tl.int32,
    R_CHUNK: gl.constexpr,
    TILE_K: gl.constexpr,
    NH: gl.constexpr,
    BD: gl.constexpr,
    D: gl.constexpr,
    MFMA_K: gl.constexpr,
    DUAL_STAGE: gl.constexpr,
):
    """Grid (T, D//BD). NH is the padded head count and the mfma contraction dim."""
    # instr_shape[2]=32: on gfx950 v_mfma_f32_16x16x32_bf16 does 2x the FLOPs of the 16-deep
    # form in the same 16 cycles. The old kernel used 16 and it did not matter there because it
    # was bandwidth-saturated at 7.3 TB/s; once the traffic is halved the matrix rate binds.
    mfma: gl.constexpr = gl.amd.cdna4.AMDMFMALayout(
        version=4,
        instr_shape=[16, 16, MFMA_K],
        transposed=True,
        warps_per_cta=[4, 1],
    )
    _q_tpw_k: gl.constexpr = min(64, BD // 8)
    _q_tpw_m: gl.constexpr = 64 // _q_tpw_k
    blk_q: gl.constexpr = gl.BlockedLayout(  # [H, BD] global load
        size_per_thread=[1, 8],
        threads_per_warp=[_q_tpw_m, _q_tpw_k],
        warps_per_cta=[4, 1],
        order=[1, 0],
    )
    blk_ds: gl.constexpr = gl.BlockedLayout(  # [H, TILE_K] global load
        size_per_thread=[1, 4],
        threads_per_warp=[16, 4],
        warps_per_cta=[4, 1],
        order=[1, 0],
    )
    sh_q: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[512, 16]], [NH, BD], [1, 0]
    )

    dot_a: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mfma, k_width=8)
    dot_b: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=mfma, k_width=8)

    token_idx = gl.program_id(axis=0)
    dblk = gl.program_id(axis=1)
    d_off = dblk * BD
    NUM_TILES: gl.constexpr = R_CHUNK // TILE_K

    q_base = token_idx.to(tl.int64) * stride_q_t
    do_base = token_idx.to(tl.int64) * stride_do_t
    ds_base = token_idx.to(tl.int64) * stride_ds_t
    interm_base = token_idx.to(tl.int64) * stride_interm_t

    # ---- prologue: stage Q, transpose to [BD, H] registers; reuse the buffer for dO ----
    offs_h_q = gl.arange(0, NH, layout=gl.SliceLayout(1, blk_q))
    offs_d_q = d_off + gl.arange(0, BD, layout=gl.SliceLayout(0, blk_q))
    mask_h_q = offs_h_q < num_heads
    q_offs = (
        q_base
        + offs_h_q[:, None].to(tl.int64) * stride_q_h
        + offs_d_q[None, :].to(tl.int64)
    )
    do_offs = (
        do_base
        + offs_h_q[:, None].to(tl.int64) * stride_do_h
        + offs_d_q[None, :].to(tl.int64)
    )

    if DUAL_STAGE:
        # two buffers -> BOTH HBM->LDS copies in flight behind ONE drain. Costs 2x LDS
        # (128 KB at BD=256, so occ-1) but halves the exposed prologue latency.
        smem_q = gl.allocate_shared_memory(
            Q_ptr.dtype.element_ty, [NH, BD], layout=sh_q
        )
        smem_do = gl.allocate_shared_memory(
            dO_ptr.dtype.element_ty, [NH, BD], layout=sh_q
        )
        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            dest=smem_q, ptr=Q_ptr, offsets=q_offs.to(tl.int32), mask=mask_h_q[:, None]
        )
        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            dest=smem_do,
            ptr=dO_ptr,
            offsets=do_offs.to(tl.int32),
            mask=mask_h_q[:, None],
        )
        gl.amd.cdna4.async_copy.commit_group()
        gl.amd.cdna4.async_copy.wait_group(0)
        Q_T = smem_q.permute([1, 0]).load(dot_a)  # [BD, H], register-resident
        dO_T = smem_do.permute([1, 0]).load(dot_a)
    else:
        # one buffer, re-used after a barrier: half the LDS (occ-2 at BD=256) but the two
        # HBM round trips serialize.
        smem_stage = gl.allocate_shared_memory(
            Q_ptr.dtype.element_ty, [NH, BD], layout=sh_q
        )
        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            dest=smem_stage,
            ptr=Q_ptr,
            offsets=q_offs.to(tl.int32),
            mask=mask_h_q[:, None],
        )
        gl.amd.cdna4.async_copy.commit_group()
        gl.amd.cdna4.async_copy.wait_group(0)
        Q_T = smem_stage.permute([1, 0]).load(dot_a)
        gl.barrier()  # all warps done reading before reuse
        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            dest=smem_stage,
            ptr=dO_ptr,
            offsets=do_offs.to(tl.int32),
            mask=mask_h_q[:, None],
        )
        gl.amd.cdna4.async_copy.commit_group()
        gl.amd.cdna4.async_copy.wait_group(0)
        dO_T = smem_stage.permute([1, 0]).load(dot_a)

    # ---- main loop: rank tile inner, head contraction folded into the mfma ----
    offs_h_ds = gl.arange(0, NH, layout=gl.SliceLayout(1, blk_ds))
    offs_k_ds = gl.arange(0, TILE_K, layout=gl.SliceLayout(0, blk_ds))
    mask_h_ds = offs_h_ds < num_heads
    offs_d_st = d_off + gl.arange(0, BD, layout=gl.SliceLayout(1, mfma))
    offs_col_st = gl.arange(0, TILE_K, layout=gl.SliceLayout(0, mfma))

    for t in range(NUM_TILES):
        col_c = t * TILE_K + offs_k_ds
        offs_c = (
            ds_base
            + offs_h_ds[:, None].to(tl.int64) * stride_ds_h
            + col_c[None, :].to(tl.int64)
        )
        dS_blk = gl.amd.cdna4.buffer_load(
            ptr=dS_ptr, offsets=offs_c.to(tl.int32), mask=mask_h_ds[:, None], other=0.0
        )
        P_blk = gl.amd.cdna4.buffer_load(
            ptr=P_ptr, offsets=offs_c.to(tl.int32), mask=mask_h_ds[:, None], other=0.0
        )

        dS_dot = gl.convert_layout(dS_blk, dot_b)
        P_dot = gl.convert_layout(P_blk, dot_b)

        dKV = gl.zeros([BD, TILE_K], dtype=gl.float32, layout=mfma)
        dKV = gl.amd.cdna4.mfma(Q_T, dS_dot, dKV)
        dKV = gl.amd.cdna4.mfma(dO_T, P_dot, dKV)

        col_st = t * TILE_K + offs_col_st
        interm_offs = (
            interm_base
            + col_st[None, :].to(tl.int64) * stride_interm_r
            + offs_d_st[:, None].to(tl.int64)
        )
        gl.amd.cdna4.buffer_store(
            stored_value=dKV.to(Interm_ptr.dtype.element_ty),
            ptr=Interm_ptr,
            offsets=interm_offs.to(tl.int32),
        )


def sparse_mla_bwd_dkv_interm_v4(
    q,
    do,
    chunk_dS,
    chunk_P,
    R_CHUNK,
    BD=256,
    TILE_K=128,
    MFMA_K=32,
    DUAL_STAGE=1,
    H_POW2=None,
    num_warps=4,
    interm=None,
):
    """V4 dKV-interm, Q/dO read once. Returns interm [T, R_CHUNK, D] bf16.

    ``BD`` splits D across ``grid.y``; dS/P are re-read once per D block, so a larger BD moves
    less of them. ``MFMA_K=32`` is the CDNA4 16x16x32 depth.
    """
    T, H, D = q.shape
    assert R_CHUNK % TILE_K == 0
    assert D % BD == 0
    h_pow2 = H_POW2 or triton.next_power_of_2(H)
    if interm is None:
        interm = torch.empty(T, R_CHUNK, D, dtype=torch.bfloat16, device=q.device)
    _dkv_interm_v4_kernel[(T, D // BD)](
        q,
        do,
        chunk_dS,
        chunk_P,
        interm,
        q.stride(0),
        q.stride(1),
        do.stride(0),
        do.stride(1),
        chunk_dS.stride(0),
        chunk_dS.stride(1),
        interm.stride(0),
        interm.stride(1),
        H,
        R_CHUNK=R_CHUNK,
        TILE_K=TILE_K,
        NH=h_pow2,
        BD=BD,
        D=D,
        MFMA_K=MFMA_K,
        DUAL_STAGE=DUAL_STAGE,
        num_warps=num_warps,
    )
    return interm


_delta_v4_kernel_repr = make_kernel_repr(
    "_delta_v4_kernel",
    [
        "D",
        "BLOCK_R",
    ],
)


@triton.jit(repr=_delta_v4_kernel_repr)
def _delta_v4_kernel(
    O_ptr,  # [n_rows, D] bf16   (rows = T*H, contiguous)
    dO_ptr,  # [n_rows, D] bf16
    Delta_ptr,  # [n_rows]    fp32
    n_rows,
    D: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    """Grid (cdiv(n_rows, BLOCK_R),) — each program reduces BLOCK_R rows of width D."""
    pid = tl.program_id(0)
    rows = pid * BLOCK_R + tl.arange(0, BLOCK_R)
    mask = rows < n_rows
    offs = rows.to(tl.int64)[:, None] * D + tl.arange(0, D)[None, :]
    o = tl.load(O_ptr + offs, mask=mask[:, None], other=0.0).to(tl.float32)
    d = tl.load(dO_ptr + offs, mask=mask[:, None], other=0.0).to(tl.float32)
    tl.store(Delta_ptr + rows, tl.sum(o * d, axis=1), mask=mask)


def delta_v4(o, do, out=None, BLOCK_R=8, num_warps=8):
    # BLOCK_R=8 keeps each lane loading >= 8 bf16, i.e. a dwordx4; narrower blocks drop to a
    # dword and the kernel loses most of its bandwidth.
    """o[T,H,D] bf16, do[T,H,D] bf16 -> delta[T,H] fp32 = sum_d o*do.

    ``do`` must already be the D-wide (lora) slice, contiguous — same contract as the dQ kernel.
    """
    assert o.shape == do.shape and o.is_contiguous() and do.is_contiguous()
    T, H, D = o.shape
    n_rows = T * H
    if out is None:
        out = torch.empty(T, H, dtype=torch.float32, device=o.device)
    _delta_v4_kernel[(triton.cdiv(n_rows, BLOCK_R),)](
        o,
        do,
        out,
        n_rows,
        D=D,
        BLOCK_R=BLOCK_R,
        num_warps=num_warps,
    )
    return out


_bwd_dkv_gather_acc_v4_repr = make_kernel_repr(
    "_bwd_dkv_gather_acc_v4",
    [
        "D",
        "BLOCK_E",
        "ACCUMULATE",
    ],
)


@triton.jit(repr=_bwd_dkv_gather_acc_v4_repr)
def _bwd_dkv_gather_acc_v4(
    Interm_ptr,  # [T, R_CHUNK, D] bf16, flat [T*R_CHUNK, D]
    InvPtr_ptr,  # [num_kv+1] int32 — CSR row pointers
    InvData_ptr,  # [valid] int32 — encoded q*R_CHUNK+local_r, sorted by KV token
    dKV_acc_ptr,  # [num_kv, D] fp32 — accumulator
    stride_interm_r: tl.int64,
    stride_acc_t: tl.int64,
    D: tl.constexpr,
    BLOCK_E: tl.constexpr,
    ACCUMULATE: tl.constexpr,
):
    """Grid (num_kv,) — one CTA per KV token, BLOCK_E CSR entries in flight.

    ``BLOCK_E`` entries are carried per iteration, which the gather needs for two reasons:

      * **load width.** A bare ``tl.arange(0, D)`` block over 256 threads is 2 bf16 = 4 B per
        lane -- a dword. The [BLOCK_E, D] block gives ``BLOCK_E*D/threads`` elements per lane,
        so the loads become dwordx4. The gather is issue-bound, so this dominates.
      * **trip count.** ``tl.sum`` folds the entry axis, so the run is consumed BLOCK_E at a
        time. A realistic top-k gives run lengths up to ~3000 on the pool rows.

    ``ACCUMULATE=False`` writes the destination instead of reading it back first. The caller
    uses it for the first chunk, where the accumulator is still zero.
    """
    k = tl.program_id(0)
    offs_d = tl.arange(0, D)
    offs_e = tl.arange(0, BLOCK_E)
    start = tl.load(InvPtr_ptr + k)
    end = tl.load(InvPtr_ptr + k + 1)
    acc_base = k.to(tl.int64) * stride_acc_t

    if ACCUMULATE:
        acc = tl.load(dKV_acc_ptr + acc_base + offs_d).to(tl.float32)
    else:
        acc = tl.zeros([D], dtype=tl.float32)

    for i0 in range(start, end, BLOCK_E):
        idx = i0 + offs_e
        m = idx < end
        entry = tl.load(InvData_ptr + idx, mask=m, other=0).to(tl.int64)
        vals = tl.load(
            Interm_ptr + entry[:, None] * stride_interm_r + offs_d[None, :],
            mask=m[:, None],
            other=0.0,
        )
        acc += tl.sum(vals.to(tl.float32), axis=0)

    tl.store(dKV_acc_ptr + acc_base + offs_d, acc)


def build_inverted_topk(topk_indices_slice, num_kv):
    """CSR inverted index over ``num_kv`` KV rows.

    One stable sort yields both the permutation (``inv_data``) and the sorted keys;
    ``inv_ptr[k] = searchsorted(sorted, k, 'left')`` = the number of entries with value < k.
    Invalid (-1) entries sort to the front, so ``inv_ptr[0]`` starts past them and they are
    never visited.

    The sort key is narrowed to int16 when ``num_kv`` fits, which is what keeps the radix sort
    to two byte-passes.

    Returns ``inv_ptr[num_kv+1]`` int32, ``inv_data[T*R]`` int32.
    """
    # row_ids is the searchsorted query: [0 .. num_kv], one per KV row plus the end sentinel.
    # Its dtype must match `keys` -- searchsorted is built per branch for that reason, not by
    # accident.
    flat_kv = topk_indices_slice.reshape(-1)  # [T*R] int32; -1 = invalid
    if num_kv < 32767:  # int16 range, -1 included
        keys = flat_kv.to(torch.int16)
        row_ids = torch.arange(num_kv + 1, device=flat_kv.device, dtype=torch.int16)
    else:
        keys = flat_kv.to(torch.int32)
        row_ids = torch.arange(num_kv + 1, device=flat_kv.device, dtype=torch.int32)
    sorted_vals, inv_data = torch.sort(keys, stable=True)
    inv_ptr = torch.searchsorted(sorted_vals, row_ids).to(torch.int32)
    return inv_ptr, inv_data.to(torch.int32)


def dkv_gather_acc(
    interm, inv_ptr, inv_data, dkv_acc, BLOCK_E=64, num_warps=8, accumulate=True
):
    """interm[T,R,D] bf16 -> dkv_acc[num_kv,D] fp32 via the entry-blocked CSR gather.

    Grid is ``num_kv`` (from ``dkv_acc``), not ``T``, so a compressed-pool KV works.
    """
    _, _, D = interm.shape
    num_kv = dkv_acc.shape[0]
    _bwd_dkv_gather_acc_v4[(num_kv,)](
        interm,
        inv_ptr,
        inv_data,
        dkv_acc,
        interm.stride(1),
        dkv_acc.stride(0),
        D=D,
        BLOCK_E=BLOCK_E,
        ACCUMULATE=accumulate,
        num_warps=num_warps,
    )
