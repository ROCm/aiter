# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Gluon kernels for the DeepSeek-V4 sparse-MLA training BACKWARD (gfx950 / CDNA4).

The two MFMA phases. Both operate on the official V4 form: ``K == V == kv`` as one dense
512-wide tensor, RoPE already applied in place caller-side, scale ``1/sqrt(512)``, ``attn_sink``
in the softmax denominator only, ``topk == -1`` masked.

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

Launchers live in ``aiter.ops.triton.attention.sparse_attention_dsv4_bwd``; this module stays
free of torch so the kernels can be called without it.
"""

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
