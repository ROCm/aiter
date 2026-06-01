# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Gluon (CDNA4 / gfx950) port of the DeepSeek V4 sparse-MLA attention kernels.

Hand-translated from the Triton kernels in
``aiter/ops/triton/_triton_kernels/attention/sparse_attention_dsv4.py``.
Layouts (MFMA / blocked / shared / dot-operand) are taken verbatim from the
ttgir the Triton compiler produces for the gfx950 launch config
(``BLOCK_H=16, BLOCK_K=16, BLOCK_D=512, num_warps=8, matrix_instr_nonkdim=16``):

* QK dot  -> ``#mma``  = amd_mfma v4, warpsPerCTA=[8,1], instrShape=[16,16,32],
             transposed, dot-operand kWidth=8.
* PV dot  -> ``#mma1`` = amd_mfma v4, warpsPerCTA=[1,8], instrShape=[16,16,16],
             transposed, dot-operand kWidth=4.

Both kernels reduce over the sparse KV axis with one online-softmax statistic.
The scores live in the QK layout (``#mma``); the value accumulator lives in the
PV layout (``#mma1``); the per-row softmax rescale factor ``alpha`` is converted
between the two slice layouts each iteration (mirroring the ttgir).
"""

import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

# Every layout derives its warpsPerCTA from gl.num_warps(), so num_warps is a
# safely autotunable parameter.
_WARP_SIZE = gl.constexpr(64)

# This module is imported only on gfx950 (see the arch guard in the wrapper),
# whose per-CU LDS is 160 KiB.
_LDS_LIMIT = 163840


# ---------------------------------------------------------------------------
# Autotune config space (mirrors the Triton kernels' search, minus num_stages —
# the Gluon kernels carry an explicit software pipeline — and minus
# matrix_instr_nonkdim, which is fixed by the explicit AMDMFMALayout).
# ---------------------------------------------------------------------------


def _prefill_prune_configs(configs, named_args, **kwargs):
    """Drop prefill configs whose padded KV LDS tile exceeds per-CU LDS.

    The kernel's only shared allocation is one [BLOCK_K, BLOCK_D] bf16 tile with
    16-element row padding (the compiler-matching PaddedSharedLayout).
    """
    BLOCK_D = kwargs.get("BLOCK_D", named_args.get("BLOCK_D"))
    pruned = [
        c for c in configs if (BLOCK_D + 16) * c.kwargs["BLOCK_K"] * 2 <= _LDS_LIMIT
    ]
    return pruned or [configs[0]]


def _get_prefill_autotune_configs():
    configs = []
    for BLOCK_H in [16, 32, 64]:
        for BLOCK_K in [16, 32, 64, 128]:
            for WPE in [0, 2]:
                for nw in [4, 8]:
                    # (BLOCK_K=64, num_warps=8, BLOCK_H<64) miscompiles the
                    # MFMA/blocked layout on gfx950 (verified numerically wrong),
                    # so it is excluded from the search space.
                    if BLOCK_K == 64 and nw == 8 and BLOCK_H < 64:
                        continue
                    configs.append(
                        triton.Config(
                            {"BLOCK_H": BLOCK_H, "BLOCK_K": BLOCK_K, "waves_per_eu": WPE},
                            num_warps=nw,
                            num_stages=1,  # Gluon kernel owns its pipeline.
                        )
                    )
    return configs


def _decode_prune_configs(configs, named_args, **kwargs):
    """Drop decode configs whose KV LDS tiles exceed per-CU LDS.

    Per tile the kernel stages four bf16 buffers (NoPE + RoPE, each in the direct
    [BLOCK_K, dim] and transposed [dim, BLOCK_K] orientation):
    ``2 * (NOPE_BLOCK + ROPE_DIM) * BLOCK_K`` elements.
    """
    NOPE_BLOCK = kwargs.get("NOPE_BLOCK", named_args.get("NOPE_BLOCK"))
    ROPE_DIM = kwargs.get("ROPE_DIM", named_args.get("ROPE_DIM"))
    pruned = [
        c
        for c in configs
        if 2 * (NOPE_BLOCK + ROPE_DIM) * c.kwargs["BLOCK_K"] * 2 <= _LDS_LIMIT
    ]
    return pruned or [configs[0]]


def _get_decode_autotune_configs():
    return [
        triton.Config(
            {"BLOCK_H": BLOCK_H, "BLOCK_K": BLOCK_K, "waves_per_eu": WPE},
            num_warps=nw,
            num_stages=1,  # Gluon kernel owns its pipeline.
        )
        for BLOCK_H in [16, 32, 64]
        for BLOCK_K in [16, 32, 64, 128]
        for WPE in [0, 1, 2]
        for nw in [4, 8]
    ]


# ---------------------------------------------------------------------------
# Prefill kernel
# ---------------------------------------------------------------------------


@triton.autotune(
    configs=_get_prefill_autotune_configs(),
    key=["num_heads", "head_dim", "HAS_ATTN_SINK"],
    prune_configs_by={"early_config_prune": _prefill_prune_configs},
)
@gluon.jit
def _sparse_attn_prefill_kernel(
    q_ptr,
    kv_ptr,
    kv_indices_ptr,
    kv_indptr_ptr,
    attn_sink_ptr,
    out_ptr,
    q_stride_t,
    q_stride_h,
    q_stride_d,
    kv_stride_n,
    kv_stride_d,
    out_stride_t,
    out_stride_h,
    out_stride_d,
    num_heads,
    head_dim,
    num_kv,
    scale,
    HAS_ATTN_SINK: gl.constexpr,
    BLOCK_H: gl.constexpr,
    BLOCK_D: gl.constexpr,
    BLOCK_K: gl.constexpr,
):
    query_idx = gl.program_id(axis=0)
    pid_h = gl.program_id(axis=1)

    # ---- layouts -------------------------------------------------------
    nw: gl.constexpr = gl.num_warps()
    mma_qk: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 32], transposed=True,
        warps_per_cta=[nw, 1],
    )
    mma_pv: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 16], transposed=True,
        warps_per_cta=[1, nw],
    )
    qk_a: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mma_qk, k_width=8)
    qk_b: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=mma_qk, k_width=8)
    pv_a: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mma_pv, k_width=4)
    pv_b: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=mma_pv, k_width=4)

    # [BLOCK_K, BLOCK_D] load layout (also used for q [BLOCK_H, BLOCK_D]).
    blk: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 8], threads_per_warp=[1, _WARP_SIZE],
        warps_per_cta=[nw, 1], order=[1, 0],
    )
    # One LDS buffer holds the KV tile [BLOCK_K, BLOCK_D] (dim contiguous). It is
    # consumed twice: as the QK B-operand transposed to [BLOCK_D, BLOCK_K] (via a
    # permute *view*, no data movement) and as the PV B-operand directly. The
    # compiler-matching padded layout (gfx950 + async_copy + bf16 + mfmaNonKDim=16
    # + inner dim == paddingInterval 512) is bank-conflict-free for both reads, so
    # a single buffer suffices instead of one per orientation.
    sh_kv: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[512, 16]], [BLOCK_K, BLOCK_D], [1, 0]
    )

    # slice layouts for the per-row / per-col index vectors
    sl_h_blk: gl.constexpr = gl.SliceLayout(1, blk)      # heads / kpos (axis0 of blk)
    sl_d_blk: gl.constexpr = gl.SliceLayout(0, blk)      # dim (axis1 of blk)
    sl_h_qk: gl.constexpr = gl.SliceLayout(1, mma_qk)    # heads, scores rows
    sl_k_qk: gl.constexpr = gl.SliceLayout(0, mma_qk)    # kpos, scores cols
    sl_h_pv: gl.constexpr = gl.SliceLayout(1, mma_pv)    # heads, acc rows
    sl_d_pv: gl.constexpr = gl.SliceLayout(0, mma_pv)    # dim, acc cols

    # ---- load q --------------------------------------------------------
    head_off = pid_h * BLOCK_H + gl.arange(0, BLOCK_H, layout=sl_h_blk)
    dim_off = gl.arange(0, BLOCK_D, layout=sl_d_blk)
    head_mask = head_off < num_heads
    q_off = head_off[:, None] * q_stride_h + dim_off[None, :] * q_stride_d
    q = gl.amd.cdna4.buffer_load(
        ptr=q_ptr + query_idx * q_stride_t,
        offsets=q_off,
        mask=head_mask[:, None],
        other=0.0,
    )
    q_dot = gl.convert_layout(q, qk_a)

    # ---- running softmax state ----------------------------------------
    m_i = gl.full([BLOCK_H], float("-inf"), dtype=gl.float32, layout=sl_h_qk)
    l_i = gl.zeros([BLOCK_H], dtype=gl.float32, layout=sl_h_qk)
    acc = gl.zeros([BLOCK_H, BLOCK_D], dtype=gl.float32, layout=mma_pv)

    kv_start = gl.load(kv_indptr_ptr + query_idx)
    kv_end = gl.load(kv_indptr_ptr + query_idx + 1)
    kv_len = kv_end - kv_start

    k_off_blk = gl.arange(0, BLOCK_K, layout=sl_h_blk)   # kpos (rows of kv tile)

    smem_kv = gl.allocate_shared_memory(
        kv_ptr.dtype.element_ty, [BLOCK_K, BLOCK_D], sh_kv
    )

    for k_start in tl.range(0, kv_len, BLOCK_K):
        kpos = k_start + k_off_blk
        slot = gl.amd.cdna4.buffer_load(
            ptr=kv_indices_ptr + kv_start,
            offsets=kpos,
            mask=kpos < kv_len,
            other=-1,
        )
        valid_blk = (kpos < kv_len) & (slot >= 0) & (slot < num_kv)
        safe_slot = gl.where(valid_blk, slot, 0)

        # DMA the gathered KV tile [BLOCK_K, BLOCK_D] global->LDS directly (no
        # register staging). safe_slot is clamped in-bounds so every offset is a
        # valid read; the softmax score mask below zeroes out-of-range kpos, so no
        # load mask is needed (an unmasked DMA also sidesteps the CDNA4
        # broadcast-mask lowering restriction).
        kv_off = safe_slot[:, None] * kv_stride_n + dim_off[None, :] * kv_stride_d
        gl.amd.cdna4.async_copy.buffer_load_to_shared(smem_kv, kv_ptr, kv_off)
        gl.amd.cdna4.async_copy.commit_group()
        gl.amd.cdna4.async_copy.wait_group(0)

        # scores-axis (kpos) validity, converted into the #mma column slice
        valid_qk = gl.convert_layout(valid_blk, sl_k_qk)

        # ---- QK (read the LDS tile transposed via a permute view) ----
        acc_qk = gl.zeros([BLOCK_H, BLOCK_K], dtype=gl.float32, layout=mma_qk)
        kvT_dot = smem_kv.permute([1, 0]).load(qk_b)
        scores = gl.amd.cdna4.mfma(q_dot, kvT_dot, acc_qk)
        scores = scores * scale
        scores = gl.where(valid_qk[None, :], scores, float("-inf"))

        # ---- online softmax ----
        m_block = gl.max(scores, axis=1)
        m_new = gl.maximum(m_i, m_block)
        alpha = gl.exp(m_i - m_new)
        p = gl.exp(scores - m_new[:, None])
        p = gl.where(valid_qk[None, :], p, 0.0)
        l_i = l_i * alpha + gl.sum(p, axis=1)
        m_i = m_new

        # ---- PV ----
        p_dot = gl.convert_layout(p.to(kv_ptr.dtype.element_ty), pv_a)
        kv_dot = smem_kv.load(layout=pv_b)
        alpha_pv = gl.convert_layout(alpha, sl_h_pv)
        acc = acc * alpha_pv[:, None]
        acc = gl.amd.cdna4.mfma(p_dot, kv_dot, acc)

    if HAS_ATTN_SINK:
        sink = gl.load(
            attn_sink_ptr + head_off, mask=head_mask, other=float("-inf")
        ).to(gl.float32)
        sink = gl.convert_layout(sink, sl_h_qk)
        m_final = gl.maximum(m_i, sink)
        alpha = gl.exp(m_i - m_final)
        l_final = l_i * alpha + gl.exp(sink - m_final)
        denom = gl.maximum(l_final, 1.0e-30)
        scale_row = alpha / denom
        guard = l_final > 0.0
    else:
        denom = gl.maximum(l_i, 1.0e-30)
        scale_row = 1.0 / denom
        guard = l_i > 0.0

    # Select on the OUTPUT (mirrors Triton's tl.where(l>0, acc/denom, 0)): a per-lane
    # select yields clean 0 for empty / all-invalid rows even when acc is NaN (a
    # leading fully-invalid kv tile makes alpha=exp(-inf-(-inf))=NaN -> acc=NaN).
    scale_pv = gl.convert_layout(scale_row, sl_h_pv)
    guard_pv = gl.convert_layout(guard, sl_h_pv)
    out = gl.where(guard_pv[:, None], acc * scale_pv[:, None], 0.0)

    out_head = pid_h * BLOCK_H + gl.arange(0, BLOCK_H, layout=sl_h_pv)
    out_dim = gl.arange(0, BLOCK_D, layout=sl_d_pv)
    out_off = out_head[:, None] * out_stride_h + out_dim[None, :] * out_stride_d
    gl.amd.cdna4.buffer_store(
        out.to(out_ptr.dtype.element_ty),
        ptr=out_ptr + query_idx * out_stride_t,
        offsets=out_off,
        mask=(out_head < num_heads)[:, None],
    )


# ---------------------------------------------------------------------------
# Decode kernel (fp8_ds_mla paged cache, dual ragged passes)
# ---------------------------------------------------------------------------


@gluon.jit
def _decode_core_attn(
    cache_ptr,
    indices_ptr,
    seg_start,
    seg_len,
    cache_stride0,
    block_size,
    num_rows,
    q_nope_dot,
    q_rope_dot,
    scale,
    m_i,
    l_i,
    acc_nope,
    acc_rope,
    smem_knT,
    smem_krT,
    smem_kn,
    smem_kr,
    NOPE_DIM: gl.constexpr,
    NOPE_BLOCK: gl.constexpr,
    ROPE_DIM: gl.constexpr,
    BLOCK_H: gl.constexpr,
    BLOCK_K: gl.constexpr,
    IS_FNUZ: gl.constexpr,
):
    """One ragged sparse pass; folds tiles into the shared softmax state."""
    nw: gl.constexpr = gl.num_warps()
    mma_qk: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 32], transposed=True,
        warps_per_cta=[nw, 1],
    )
    mma_pv: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 16], transposed=True,
        warps_per_cta=[1, nw],
    )
    qk_b: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=mma_qk, k_width=8)
    pv_a: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mma_pv, k_width=4)
    pv_b: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=mma_pv, k_width=4)

    blk_n: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 16], threads_per_warp=[2, 32],
        warps_per_cta=[nw, 1], order=[1, 0],
    )
    blk_r: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 2], threads_per_warp=[2, 32],
        warps_per_cta=[nw, 1], order=[1, 0],
    )
    sl_k_n: gl.constexpr = gl.SliceLayout(1, blk_n)   # kpos rows (nope tile)
    sl_d_n: gl.constexpr = gl.SliceLayout(0, blk_n)   # nope dim cols
    sl_k_r: gl.constexpr = gl.SliceLayout(1, blk_r)   # kpos rows (rope tile)
    sl_d_r: gl.constexpr = gl.SliceLayout(0, blk_r)   # rope dim cols
    sl_k_qk: gl.constexpr = gl.SliceLayout(0, mma_qk)
    sl_h_pv: gl.constexpr = gl.SliceLayout(1, mma_pv)

    nope_off = gl.arange(0, NOPE_BLOCK, layout=sl_d_n)
    nope_mask = nope_off < NOPE_DIM
    nope_grp = nope_off // 64
    rope_off = gl.arange(0, ROPE_DIM, layout=sl_d_r)
    k_off_n = gl.arange(0, BLOCK_K, layout=sl_k_n)
    k_off_r = gl.arange(0, BLOCK_K, layout=sl_k_r)

    for k_start in tl.range(0, seg_len, BLOCK_K):
        # --- nope tile (fp8 dequant) ---
        kpos = k_start + k_off_n
        slot = gl.load(indices_ptr + seg_start + kpos, mask=kpos < seg_len, other=-1)
        valid = (kpos < seg_len) & (slot >= 0) & (slot < num_rows)
        safe = gl.where(valid, slot, 0)
        block_off = (safe // block_size).to(gl.int64) * cache_stride0
        pos_in_block = safe % block_size
        token_ptr = cache_ptr + block_off + pos_in_block * 576
        scale_ptr = cache_ptr + block_off + block_size * 576 + pos_in_block * 8

        x_u8 = gl.load(
            token_ptr[:, None] + nope_off[None, :],
            mask=valid[:, None] & nope_mask[None, :],
            other=0,
        )
        if IS_FNUZ:
            x_fp8 = x_u8.to(gl.float8e4b15, bitcast=True)
        else:
            x_fp8 = x_u8.to(gl.float8e4nv, bitcast=True)
        enc = gl.load(
            scale_ptr[:, None] + nope_grp[None, :],
            mask=valid[:, None] & nope_mask[None, :],
            other=127,
        )
        scales = gl.exp2(enc.to(gl.float32) - 127.0)
        k_nope = x_fp8.to(gl.bfloat16) * scales.to(gl.bfloat16)
        zero_n = gl.zeros([BLOCK_K, NOPE_BLOCK], dtype=gl.bfloat16, layout=blk_n)
        k_nope = gl.where(valid[:, None] & nope_mask[None, :], k_nope, zero_n)
        k_nope = gl.where(k_nope == k_nope, k_nope, zero_n)

        # --- rope tile (bf16) ---
        kpos_r = k_start + k_off_r
        slot_r = gl.load(
            indices_ptr + seg_start + kpos_r, mask=kpos_r < seg_len, other=-1
        )
        valid_r = (kpos_r < seg_len) & (slot_r >= 0) & (slot_r < num_rows)
        safe_r = gl.where(valid_r, slot_r, 0)
        block_off_r = (safe_r // block_size).to(gl.int64) * cache_stride0
        token_off_r = block_off_r + (safe_r % block_size) * 576 + NOPE_DIM
        rope_base = (cache_ptr + token_off_r).to(gl.pointer_type(gl.bfloat16))
        k_rope = gl.load(
            rope_base[:, None] + rope_off[None, :],
            mask=valid_r[:, None],
            other=0.0,
        )
        zero_r = gl.zeros([BLOCK_K, ROPE_DIM], dtype=gl.bfloat16, layout=blk_r)
        k_rope = gl.where(valid_r[:, None], k_rope, zero_r)
        k_rope = gl.where(k_rope == k_rope, k_rope, zero_r)

        # --- stage K/V tiles through shared memory ---
        smem_kn.store(k_nope)
        smem_kr.store(k_rope)
        smem_knT.store(gl.permute(k_nope, [1, 0]))
        smem_krT.store(gl.permute(k_rope, [1, 0]))

        valid_qk = gl.convert_layout(valid, sl_k_qk)

        # --- QK: scores = q_nope @ k_nope^T + q_rope @ k_rope^T ---
        scores = gl.zeros([BLOCK_H, BLOCK_K], dtype=gl.float32, layout=mma_qk)
        scores = gl.amd.cdna4.mfma(q_nope_dot, smem_knT.load(layout=qk_b), scores)
        scores = gl.amd.cdna4.mfma(q_rope_dot, smem_krT.load(layout=qk_b), scores)
        scores = scores * scale
        scores = gl.where(valid_qk[None, :], scores, float("-inf"))

        # --- online softmax ---
        m_block = gl.max(scores, axis=1)
        m_new = gl.maximum(m_i, m_block)
        alpha = gl.exp(m_i - m_new)
        p = gl.exp(scores - m_new[:, None])
        p = gl.where(valid_qk[None, :], p, 0.0)
        l_i = l_i * alpha + gl.sum(p, axis=1)
        m_i = m_new

        # --- PV ---
        p_dot = gl.convert_layout(p.to(gl.bfloat16), pv_a)
        alpha_pv = gl.convert_layout(alpha, sl_h_pv)
        acc_nope = acc_nope * alpha_pv[:, None]
        acc_nope = gl.amd.cdna4.mfma(p_dot, smem_kn.load(layout=pv_b), acc_nope)
        acc_rope = acc_rope * alpha_pv[:, None]
        acc_rope = gl.amd.cdna4.mfma(p_dot, smem_kr.load(layout=pv_b), acc_rope)

    return m_i, l_i, acc_nope, acc_rope


@triton.autotune(
    configs=_get_decode_autotune_configs(),
    key=["num_heads", "NOPE_DIM", "ROPE_DIM", "HAS_ATTN_SINK", "HAS_EXTRA"],
    prune_configs_by={"early_config_prune": _decode_prune_configs},
)
@gluon.jit
def _sparse_attn_decode_kernel(
    q_ptr,
    main_cache_ptr,
    main_indices_ptr,
    main_indptr_ptr,
    extra_cache_ptr,
    extra_indices_ptr,
    extra_indptr_ptr,
    attn_sink_ptr,
    out_ptr,
    q_stride0,
    q_stride1,
    out_stride0,
    out_stride1,
    main_cache_stride0,
    extra_cache_stride0,
    main_num_rows,
    extra_num_rows,
    main_block_size,
    extra_block_size,
    scale,
    num_heads,
    HAS_ATTN_SINK: gl.constexpr,
    HAS_EXTRA: gl.constexpr,
    NOPE_DIM: gl.constexpr,
    NOPE_BLOCK: gl.constexpr,
    ROPE_DIM: gl.constexpr,
    IS_FNUZ: gl.constexpr,
    BLOCK_H: gl.constexpr,
    BLOCK_K: gl.constexpr,
):
    query_idx = gl.program_id(axis=0)
    pid_h = gl.program_id(axis=1)

    nw: gl.constexpr = gl.num_warps()
    mma_qk: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 32], transposed=True,
        warps_per_cta=[nw, 1],
    )
    mma_pv: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 16], transposed=True,
        warps_per_cta=[1, nw],
    )
    qk_a: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mma_qk, k_width=8)

    blk_n: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 16], threads_per_warp=[2, 32],
        warps_per_cta=[nw, 1], order=[1, 0],
    )
    blk_r: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 2], threads_per_warp=[2, 32],
        warps_per_cta=[nw, 1], order=[1, 0],
    )
    sh_knT: gl.constexpr = gl.SwizzledSharedLayout(
        vec=8, per_phase=1, max_phase=16, order=[0, 1]
    )
    sh_krT: gl.constexpr = gl.SwizzledSharedLayout(
        vec=8, per_phase=2, max_phase=8, order=[0, 1]
    )
    sh_kn: gl.constexpr = gl.SwizzledSharedLayout(
        vec=4, per_phase=1, max_phase=16, order=[1, 0]
    )
    sh_kr: gl.constexpr = gl.SwizzledSharedLayout(
        vec=4, per_phase=2, max_phase=8, order=[1, 0]
    )

    sl_h_n: gl.constexpr = gl.SliceLayout(1, blk_n)
    sl_d_n: gl.constexpr = gl.SliceLayout(0, blk_n)
    sl_h_r: gl.constexpr = gl.SliceLayout(1, blk_r)
    sl_d_r: gl.constexpr = gl.SliceLayout(0, blk_r)
    sl_h_qk: gl.constexpr = gl.SliceLayout(1, mma_qk)
    sl_h_pv: gl.constexpr = gl.SliceLayout(1, mma_pv)
    sl_d_pv: gl.constexpr = gl.SliceLayout(0, mma_pv)

    # --- load q (nope + rope) ---
    head_n = pid_h * BLOCK_H + gl.arange(0, BLOCK_H, layout=sl_h_n)
    head_r = pid_h * BLOCK_H + gl.arange(0, BLOCK_H, layout=sl_h_r)
    nope_off = gl.arange(0, NOPE_BLOCK, layout=sl_d_n)
    nope_mask = nope_off < NOPE_DIM
    rope_off = gl.arange(0, ROPE_DIM, layout=sl_d_r)

    q_base = q_ptr + query_idx * q_stride0
    q_nope = gl.amd.cdna4.buffer_load(
        ptr=q_base,
        offsets=head_n[:, None] * q_stride1 + nope_off[None, :],
        mask=(head_n < num_heads)[:, None] & nope_mask[None, :],
        other=0.0,
    )
    q_rope = gl.amd.cdna4.buffer_load(
        ptr=q_base,
        offsets=head_r[:, None] * q_stride1 + NOPE_DIM + rope_off[None, :],
        mask=(head_r < num_heads)[:, None],
        other=0.0,
    )
    q_nope_dot = gl.convert_layout(q_nope, qk_a)
    q_rope_dot = gl.convert_layout(q_rope, qk_a)

    # --- running softmax state ---
    m_i = gl.full([BLOCK_H], float("-inf"), dtype=gl.float32, layout=sl_h_qk)
    l_i = gl.zeros([BLOCK_H], dtype=gl.float32, layout=sl_h_qk)
    acc_nope = gl.zeros([BLOCK_H, NOPE_BLOCK], dtype=gl.float32, layout=mma_pv)
    acc_rope = gl.zeros([BLOCK_H, ROPE_DIM], dtype=gl.float32, layout=mma_pv)

    # --- shared staging buffers (reused across both passes) ---
    smem_knT = gl.allocate_shared_memory(gl.bfloat16, [NOPE_BLOCK, BLOCK_K], sh_knT)
    smem_krT = gl.allocate_shared_memory(gl.bfloat16, [ROPE_DIM, BLOCK_K], sh_krT)
    smem_kn = gl.allocate_shared_memory(gl.bfloat16, [BLOCK_K, NOPE_BLOCK], sh_kn)
    smem_kr = gl.allocate_shared_memory(gl.bfloat16, [BLOCK_K, ROPE_DIM], sh_kr)

    main_start = gl.load(main_indptr_ptr + query_idx)
    main_end = gl.load(main_indptr_ptr + query_idx + 1)
    m_i, l_i, acc_nope, acc_rope = _decode_core_attn(
        main_cache_ptr, main_indices_ptr, main_start, main_end - main_start,
        main_cache_stride0, main_block_size, main_num_rows,
        q_nope_dot, q_rope_dot, scale,
        m_i, l_i, acc_nope, acc_rope,
        smem_knT, smem_krT, smem_kn, smem_kr,
        NOPE_DIM, NOPE_BLOCK, ROPE_DIM, BLOCK_H, BLOCK_K, IS_FNUZ,
    )

    if HAS_EXTRA:
        extra_start = gl.load(extra_indptr_ptr + query_idx)
        extra_end = gl.load(extra_indptr_ptr + query_idx + 1)
        m_i, l_i, acc_nope, acc_rope = _decode_core_attn(
            extra_cache_ptr, extra_indices_ptr, extra_start, extra_end - extra_start,
            extra_cache_stride0, extra_block_size, extra_num_rows,
            q_nope_dot, q_rope_dot, scale,
            m_i, l_i, acc_nope, acc_rope,
            smem_knT, smem_krT, smem_kn, smem_kr,
            NOPE_DIM, NOPE_BLOCK, ROPE_DIM, BLOCK_H, BLOCK_K, IS_FNUZ,
        )

    # --- finalize ---
    head_q = pid_h * BLOCK_H + gl.arange(0, BLOCK_H, layout=sl_h_qk)
    if HAS_ATTN_SINK:
        sink = gl.load(
            attn_sink_ptr + head_q, mask=head_q < num_heads, other=float("-inf")
        ).to(gl.float32)
        m_final = gl.maximum(m_i, sink)
        alpha = gl.exp(m_i - m_final)
        l_final = l_i * alpha + gl.exp(sink - m_final)
        denom = gl.maximum(l_final, 1.0e-30)
        scale_row = alpha / denom
        guard = l_final > 0.0
    else:
        denom = gl.maximum(l_i, 1.0e-30)
        scale_row = 1.0 / denom
        guard = l_i > 0.0

    # Select on the OUTPUT (mirrors Triton): clean 0 for empty / all-invalid rows
    # even when acc is NaN (leading fully-invalid kv tile -> alpha NaN -> acc NaN).
    scale_pv = gl.convert_layout(scale_row, sl_h_pv)
    guard_pv = gl.convert_layout(guard, sl_h_pv)
    out_nope = gl.where(guard_pv[:, None], acc_nope * scale_pv[:, None], 0.0)
    out_rope = gl.where(guard_pv[:, None], acc_rope * scale_pv[:, None], 0.0)

    out_head = pid_h * BLOCK_H + gl.arange(0, BLOCK_H, layout=sl_h_pv)
    nope_off_pv = gl.arange(0, NOPE_BLOCK, layout=sl_d_pv)
    rope_off_pv = gl.arange(0, ROPE_DIM, layout=sl_d_pv)
    out_base = out_ptr + query_idx * out_stride0
    gl.amd.cdna4.buffer_store(
        out_nope.to(out_ptr.dtype.element_ty),
        ptr=out_base,
        offsets=out_head[:, None] * out_stride1 + nope_off_pv[None, :],
        mask=(out_head < num_heads)[:, None] & (nope_off_pv < NOPE_DIM)[None, :],
    )
    gl.amd.cdna4.buffer_store(
        out_rope.to(out_ptr.dtype.element_ty),
        ptr=out_base,
        offsets=out_head[:, None] * out_stride1 + NOPE_DIM + rope_off_pv[None, :],
        mask=(out_head < num_heads)[:, None],
    )
