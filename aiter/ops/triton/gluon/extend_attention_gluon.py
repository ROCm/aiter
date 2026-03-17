# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Gluon extend-attention kernel for gfx950 (MI350X / CDNA4).

Three dispatch paths -- basic request-centric, mask-split, and persistent-CTA
(work-centric scheduling) -- are selected at launch time based on workload shape.
Supports D=64/128, causal/non-causal, logit cap, sliding window, custom mask,
FP8 KV dequant (k_scale/v_scale), and XAI temperature.
"""

import math

import torch
import triton
import triton.language as tl

from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.amd import AMDMFMALayout, warp_pipeline_stage
from triton.experimental.gluon.language.amd.cdna4 import async_copy as cdna4_async
from triton.experimental.gluon.language.amd.cdna4 import (
    buffer_load as cdna4_buffer_load,
)
from triton.experimental.gluon.language.amd.cdna4 import mfma as mfma_cdna4
from triton.experimental.gluon.language._layouts import (
    DistributedLinearLayout,
    DotOperandLayout,
    PaddedSharedLayout,
)

LOG2E = tl.constexpr(1.4426950408889634)


# ===-----------------------------------------------------------------------===#
# Primitives (inlined from f16_extend_attention_gfx950)
# ===-----------------------------------------------------------------------===#


@gluon.jit
def _nan_propagating_max(a, b):
    return gl.maximum(a, b, propagate_nan=tl.PropagateNan.ALL)


@gluon.jit
def nan_propagating_max(x, axis):
    return gl.reduce(x, axis, _nan_propagating_max)


@gluon.jit
def do_mma(a, b, c):
    return mfma_cdna4(a, b, c)


@gluon.jit
def issue_async_load_k_prefix(
    kt_smem,
    k_prefix_base,  #
    kv_indices,
    kv_start,
    start_n,
    seq_len_prefix,  #
    stride_buf_kbs,  #
    BLOCK_N: gl.constexpr,
    BLOCK_DMODEL: gl.constexpr,
    ACTUAL_BLOCK_DMODEL: gl.constexpr,  #
    kt_async_layout: gl.constexpr,  #
):
    kt_offs_d_layout: gl.constexpr = gl.SliceLayout(dim=1, parent=kt_async_layout)
    kt_offs_n_layout: gl.constexpr = gl.SliceLayout(dim=0, parent=kt_async_layout)
    kt_offs_d = gl.arange(0, BLOCK_DMODEL, layout=kt_offs_d_layout)
    kt_offs_n = gl.arange(0, BLOCK_N, layout=kt_offs_n_layout)

    n_idx = start_n + kt_offs_n
    mask_n = n_idx < seq_len_prefix
    kv_locs = gl.load(kv_indices + kv_start + n_idx, mask=mask_n, other=0).to(tl.int32)

    kt_offsets = (kt_offs_d[:, None] + kv_locs[None, :] * stride_buf_kbs).to(tl.int32)

    kt_mask = mask_n[None, :]
    if ACTUAL_BLOCK_DMODEL != BLOCK_DMODEL:
        kt_mask = kt_mask & (kt_offs_d[:, None] < ACTUAL_BLOCK_DMODEL)
    cdna4_async.buffer_load_to_shared(
        kt_smem, k_prefix_base, kt_offsets, mask=kt_mask, other=0.0
    )
    cdna4_async.commit_group()


@gluon.jit
def issue_async_load_v_prefix(
    v_smem,
    v_prefix_base,  #
    kv_indices,
    kv_start,
    start_n,
    seq_len_prefix,  #
    stride_buf_vbs,  #
    BLOCK_N: gl.constexpr,
    BLOCK_DMODEL: gl.constexpr,
    ACTUAL_BLOCK_DMODEL: gl.constexpr,  #
    v_async_layout: gl.constexpr,  #
):
    v_offs_n_layout: gl.constexpr = gl.SliceLayout(dim=1, parent=v_async_layout)
    v_offs_d_layout: gl.constexpr = gl.SliceLayout(dim=0, parent=v_async_layout)
    v_offs_n = gl.arange(0, BLOCK_N, layout=v_offs_n_layout)
    v_offs_d = gl.arange(0, BLOCK_DMODEL, layout=v_offs_d_layout)

    n_idx = start_n + v_offs_n
    mask_n = n_idx < seq_len_prefix
    kv_locs = gl.load(kv_indices + kv_start + n_idx, mask=mask_n, other=0).to(tl.int32)

    v_offsets = (kv_locs[:, None] * stride_buf_vbs + v_offs_d[None, :]).to(tl.int32)

    v_mask = mask_n[:, None]
    if ACTUAL_BLOCK_DMODEL != BLOCK_DMODEL:
        v_mask = v_mask & (v_offs_d[None, :] < ACTUAL_BLOCK_DMODEL)
    cdna4_async.buffer_load_to_shared(
        v_smem, v_prefix_base, v_offsets, mask=v_mask, other=0.0
    )
    cdna4_async.commit_group()


@gluon.jit
def issue_dma_k_prefix_from_locs(
    kt_smem,
    k_prefix_base,  #
    kv_locs,
    mask_n,
    stride_buf_kbs,  #
    BLOCK_N: gl.constexpr,
    BLOCK_DMODEL: gl.constexpr,
    ACTUAL_BLOCK_DMODEL: gl.constexpr,  #
    kt_async_layout: gl.constexpr,  #
):
    kt_offs_d_layout: gl.constexpr = gl.SliceLayout(dim=1, parent=kt_async_layout)
    kt_offs_d = gl.arange(0, BLOCK_DMODEL, layout=kt_offs_d_layout)
    kt_offsets = (kt_offs_d[:, None] + kv_locs[None, :] * stride_buf_kbs).to(tl.int32)
    kt_mask = mask_n[None, :]
    if ACTUAL_BLOCK_DMODEL != BLOCK_DMODEL:
        kt_mask = kt_mask & (kt_offs_d[:, None] < ACTUAL_BLOCK_DMODEL)
    cdna4_async.buffer_load_to_shared(
        kt_smem, k_prefix_base, kt_offsets, mask=kt_mask, other=0.0
    )
    cdna4_async.commit_group()


@gluon.jit
def issue_dma_v_prefix_from_locs(
    v_smem,
    v_prefix_base,  #
    kv_locs,
    mask_n,
    stride_buf_vbs,  #
    BLOCK_N: gl.constexpr,
    BLOCK_DMODEL: gl.constexpr,
    ACTUAL_BLOCK_DMODEL: gl.constexpr,  #
    v_async_layout: gl.constexpr,  #
):
    v_offs_d_layout: gl.constexpr = gl.SliceLayout(dim=0, parent=v_async_layout)
    v_offs_d = gl.arange(0, BLOCK_DMODEL, layout=v_offs_d_layout)
    v_offsets = (kv_locs[:, None] * stride_buf_vbs + v_offs_d[None, :]).to(tl.int32)
    v_mask = mask_n[:, None]
    if ACTUAL_BLOCK_DMODEL != BLOCK_DMODEL:
        v_mask = v_mask & (v_offs_d[None, :] < ACTUAL_BLOCK_DMODEL)
    cdna4_async.buffer_load_to_shared(
        v_smem, v_prefix_base, v_offsets, mask=v_mask, other=0.0
    )
    cdna4_async.commit_group()


@gluon.jit
def issue_dma_k_prefix_from_locs_hot(
    kt_smem,
    k_prefix_base,  #
    kv_locs,
    scalar_mask,
    stride_buf_kbs,  #
    BLOCK_N: gl.constexpr,
    BLOCK_DMODEL: gl.constexpr,
    ACTUAL_BLOCK_DMODEL: gl.constexpr,  #
    kt_async_layout: gl.constexpr,  #
):
    kt_offs_d_layout: gl.constexpr = gl.SliceLayout(dim=1, parent=kt_async_layout)
    kt_offs_d = gl.arange(0, BLOCK_DMODEL, layout=kt_offs_d_layout)
    kt_offsets = (kt_offs_d[:, None] + kv_locs[None, :] * stride_buf_kbs).to(tl.int32)
    if ACTUAL_BLOCK_DMODEL != BLOCK_DMODEL:
        kt_mask = kt_offs_d[:, None] < ACTUAL_BLOCK_DMODEL
        cdna4_async.buffer_load_to_shared(
            kt_smem, k_prefix_base, kt_offsets, mask=kt_mask, other=0.0
        )
    else:
        cdna4_async.buffer_load_to_shared(
            kt_smem, k_prefix_base, kt_offsets, mask=scalar_mask
        )
    cdna4_async.commit_group()


@gluon.jit
def issue_dma_v_prefix_from_locs_hot(
    v_smem,
    v_prefix_base,  #
    kv_locs,
    scalar_mask,
    stride_buf_vbs,  #
    BLOCK_N: gl.constexpr,
    BLOCK_DMODEL: gl.constexpr,
    ACTUAL_BLOCK_DMODEL: gl.constexpr,  #
    v_async_layout: gl.constexpr,  #
):
    v_offs_d_layout: gl.constexpr = gl.SliceLayout(dim=0, parent=v_async_layout)
    v_offs_d = gl.arange(0, BLOCK_DMODEL, layout=v_offs_d_layout)
    v_offsets = (kv_locs[:, None] * stride_buf_vbs + v_offs_d[None, :]).to(tl.int32)
    if ACTUAL_BLOCK_DMODEL != BLOCK_DMODEL:
        v_mask = v_offs_d[None, :] < ACTUAL_BLOCK_DMODEL
        cdna4_async.buffer_load_to_shared(
            v_smem, v_prefix_base, v_offsets, mask=v_mask, other=0.0
        )
    else:
        cdna4_async.buffer_load_to_shared(
            v_smem, v_prefix_base, v_offsets, mask=scalar_mask
        )
    cdna4_async.commit_group()


@gluon.jit
def issue_async_load_k_extend(
    kt_smem,
    k_base,
    start_n,
    seq_len_extend,  #
    stride_kbs,  #
    BLOCK_N: gl.constexpr,
    BLOCK_DMODEL: gl.constexpr,
    ACTUAL_BLOCK_DMODEL: gl.constexpr,  #
    kt_async_layout: gl.constexpr,  #
):
    kt_offs_d_layout: gl.constexpr = gl.SliceLayout(dim=1, parent=kt_async_layout)
    kt_offs_n_layout: gl.constexpr = gl.SliceLayout(dim=0, parent=kt_async_layout)
    kt_offs_d = gl.arange(0, BLOCK_DMODEL, layout=kt_offs_d_layout)
    kt_offs_n = gl.arange(0, BLOCK_N, layout=kt_offs_n_layout)
    kt_offsets = (kt_offs_d[:, None] + (start_n + kt_offs_n[None, :]) * stride_kbs).to(
        tl.int32
    )

    kt_mask = (start_n + kt_offs_n[None, :]) < seq_len_extend
    if ACTUAL_BLOCK_DMODEL != BLOCK_DMODEL:
        kt_mask = kt_mask & (kt_offs_d[:, None] < ACTUAL_BLOCK_DMODEL)
    cdna4_async.buffer_load_to_shared(
        kt_smem, k_base, kt_offsets, mask=kt_mask, other=0.0
    )
    cdna4_async.commit_group()


@gluon.jit
def issue_async_load_v_extend(
    v_smem,
    v_base,
    start_n,
    seq_len_extend,  #
    stride_vbs,  #
    BLOCK_N: gl.constexpr,
    BLOCK_DMODEL: gl.constexpr,
    ACTUAL_BLOCK_DMODEL: gl.constexpr,  #
    v_async_layout: gl.constexpr,  #
):
    v_offs_n_layout: gl.constexpr = gl.SliceLayout(dim=1, parent=v_async_layout)
    v_offs_d_layout: gl.constexpr = gl.SliceLayout(dim=0, parent=v_async_layout)
    v_offs_n = gl.arange(0, BLOCK_N, layout=v_offs_n_layout)
    v_offs_d = gl.arange(0, BLOCK_DMODEL, layout=v_offs_d_layout)
    v_offsets = ((start_n + v_offs_n[:, None]) * stride_vbs + v_offs_d[None, :]).to(
        tl.int32
    )

    v_mask = (start_n + v_offs_n[:, None]) < seq_len_extend
    if ACTUAL_BLOCK_DMODEL != BLOCK_DMODEL:
        v_mask = v_mask & (v_offs_d[None, :] < ACTUAL_BLOCK_DMODEL)
    cdna4_async.buffer_load_to_shared(v_smem, v_base, v_offsets, mask=v_mask, other=0.0)
    cdna4_async.commit_group()


@gluon.jit
def compute_softmax_prefix(
    acc,
    l_i,
    m_i,
    qk,
    start_n,
    seq_len_prefix,  #
    qk_scale: gl.constexpr,
    LOGIT_CAP: gl.constexpr,  #
    xai_temperature_reg,
    XAI_TEMPERATURE_LEN: gl.constexpr,  #
    q_abs_pos,
    SLIDING_WINDOW_SIZE: gl.constexpr,  #
    Mask,
    mask_base_idx,
    mask_row_stride,
    q_extend_offs,  #
    USE_CUSTOM_MASK: gl.constexpr,
    SKIP_PREFIX_CUSTOM_MASK: gl.constexpr,  #
    ENABLE_PREFIX_UNMASKED: gl.constexpr,  #
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,  #
    mma_layout: gl.constexpr,
    mma_offs_n_col: gl.constexpr,  #
):
    qk_scaled = qk * qk_scale
    if LOGIT_CAP > 0:
        log2_cap: gl.constexpr = LOGIT_CAP * LOG2E
        inv_cap: gl.constexpr = 2.0 / LOGIT_CAP
        e_neg = tl.math.exp2(-qk_scaled * inv_cap)
        sig = 1.0 / (1.0 + e_neg)
        qk_scaled = log2_cap * (2.0 * sig - 1.0)
    if XAI_TEMPERATURE_LEN > 0:
        qk_scaled = qk_scaled * xai_temperature_reg[:, None]
    bound_offs = start_n + gl.arange(0, BLOCK_N, layout=mma_offs_n_col)
    use_custom_prefix_mask = USE_CUSTOM_MASK and (not SKIP_PREFIX_CUSTOM_MASK)
    is_partial_tail = (start_n + BLOCK_N) > seq_len_prefix
    use_unmasked_path = (
        ENABLE_PREFIX_UNMASKED
        and (SLIDING_WINDOW_SIZE <= 0)
        and (not use_custom_prefix_mask)
        and (not is_partial_tail)
    )

    if use_unmasked_path:
        m_ij = nan_propagating_max(qk_scaled, axis=1)
        m_new = gl.maximum(m_i, m_ij, propagate_nan=tl.PropagateNan.ALL)
        p = gl.exp2(qk_scaled - m_new[:, None])
    else:
        bound_mask = (q_abs_pos[:, None] >= 0) & (bound_offs[None, :] < seq_len_prefix)
        if SLIDING_WINDOW_SIZE > 0:
            bound_mask = bound_mask & (
                q_abs_pos[:, None] <= bound_offs[None, :] + SLIDING_WINDOW_SIZE
            )
        if use_custom_prefix_mask:
            mask_ptrs = (
                Mask
                + mask_base_idx
                + q_extend_offs[:, None] * mask_row_stride
                + start_n
                + gl.arange(0, BLOCK_N, layout=mma_offs_n_col)[None, :]
            )
            mask_vals = gl.load(mask_ptrs, mask=bound_mask, other=0)
            bound_mask = bound_mask & (mask_vals != 0)
        qk_scaled = gl.where(
            bound_mask,
            qk_scaled,
            gl.full(
                [BLOCK_M, BLOCK_N], float("-inf"), dtype=gl.float32, layout=mma_layout
            ),
        )

        m_ij = nan_propagating_max(qk_scaled, axis=1)
        m_new = gl.maximum(m_i, m_ij, propagate_nan=tl.PropagateNan.ALL)
        if SLIDING_WINDOW_SIZE > 0 or use_custom_prefix_mask:
            m_new = gl.maximum(
                m_new,
                gl.full(
                    [BLOCK_M],
                    -1e20,
                    dtype=gl.float32,
                    layout=gl.SliceLayout(dim=1, parent=mma_layout),
                ),
            )
        p = gl.exp2(qk_scaled - m_new[:, None])
    l_ij = gl.sum(p, axis=1)
    alpha = gl.exp2(m_i - m_new)
    l_i = l_i * alpha + l_ij
    acc = acc * alpha[:, None]
    m_i = m_new
    return acc, l_i, m_i, p


@gluon.jit
def compute_softmax_extend(
    acc,
    l_i,
    m_i,
    qk,
    start_n,
    cur_block_m,
    seq_len_extend,  #
    qk_scale: gl.constexpr,
    LOGIT_CAP: gl.constexpr,  #
    xai_temperature_reg,
    XAI_TEMPERATURE_LEN: gl.constexpr,  #
    SLIDING_WINDOW_SIZE: gl.constexpr,  #
    IS_CAUSAL: gl.constexpr,  #
    Mask,
    mask_base_idx,
    mask_row_stride,
    mask_kv_col_offset,  #
    USE_CUSTOM_MASK: gl.constexpr,  #
    MASK_STEPS: gl.constexpr,  #
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,  #
    mma_layout: gl.constexpr,
    mma_offs_n_col: gl.constexpr,
    mma_offs_m_row: gl.constexpr,  #
):
    qk_scaled = qk * qk_scale
    if LOGIT_CAP > 0:
        log2_cap: gl.constexpr = LOGIT_CAP * LOG2E
        inv_cap: gl.constexpr = 2.0 / LOGIT_CAP
        e_neg = tl.math.exp2(-qk_scaled * inv_cap)
        sig = 1.0 / (1.0 + e_neg)
        qk_scaled = log2_cap * (2.0 * sig - 1.0)
    if XAI_TEMPERATURE_LEN > 0:
        qk_scaled = qk_scaled * xai_temperature_reg[:, None]
    if MASK_STEPS:
        bound_offs = start_n + gl.arange(0, BLOCK_N, layout=mma_offs_n_col)
        q_offs = cur_block_m * BLOCK_M + gl.arange(0, BLOCK_M, layout=mma_offs_m_row)
        valid_mask = q_offs[:, None] < seq_len_extend
        valid_mask = valid_mask & (bound_offs[None, :] < seq_len_extend)
        if USE_CUSTOM_MASK:
            mask_ptrs = (
                Mask
                + mask_base_idx
                + q_offs[:, None] * mask_row_stride
                + mask_kv_col_offset
                + start_n
                + gl.arange(0, BLOCK_N, layout=mma_offs_n_col)[None, :]
            )
            mask_vals = gl.load(mask_ptrs, mask=valid_mask, other=0)
            valid_mask = valid_mask & (mask_vals != 0)
        elif IS_CAUSAL:
            valid_mask = valid_mask & (q_offs[:, None] >= bound_offs[None, :])
        if SLIDING_WINDOW_SIZE > 0:
            valid_mask = valid_mask & (
                q_offs[:, None] <= bound_offs[None, :] + SLIDING_WINDOW_SIZE
            )
        qk_scaled = gl.where(
            valid_mask,
            qk_scaled,
            gl.full(
                [BLOCK_M, BLOCK_N], float("-inf"), dtype=gl.float32, layout=mma_layout
            ),
        )

        m_ij = nan_propagating_max(qk_scaled, axis=1)
        m_new = gl.maximum(m_i, m_ij, propagate_nan=tl.PropagateNan.ALL)
        if SLIDING_WINDOW_SIZE > 0 or USE_CUSTOM_MASK:
            m_new = gl.maximum(
                m_new,
                gl.full(
                    [BLOCK_M],
                    -1e20,
                    dtype=gl.float32,
                    layout=gl.SliceLayout(dim=1, parent=mma_layout),
                ),
            )
        p = gl.exp2(qk_scaled - m_new[:, None])
    else:
        m_ij = nan_propagating_max(qk_scaled, axis=1)
        m_new = gl.maximum(m_i, m_ij, propagate_nan=tl.PropagateNan.ALL)
        p = gl.exp2(qk_scaled - m_new[:, None])
    l_ij = gl.sum(p, axis=1)
    alpha = gl.exp2(m_i - m_new)
    l_i = l_i * alpha + l_ij
    acc = acc * alpha[:, None]
    m_i = m_new
    return acc, l_i, m_i, p


# ===-----------------------------------------------------------------------===#
# Prefix inner loops
# ===-----------------------------------------------------------------------===#


@gluon.jit
def attn_fwd_inner_prefix_pipelined(
    acc,
    l_i,
    m_i,
    q_dot,  #
    K_Buffer,
    V_Buffer,
    kv_indices,  #
    kv_start,
    cur_kv_head,
    seq_len_prefix,  #
    stride_buf_kbs,
    stride_buf_kh,  #
    stride_buf_vbs,
    stride_buf_vh,  #
    kt_smem,
    v_smem,  #
    qk_scale: gl.constexpr,
    LOGIT_CAP: gl.constexpr,  #
    xai_temperature_reg,
    XAI_TEMPERATURE_LEN: gl.constexpr,  #
    q_abs_pos,
    SLIDING_WINDOW_SIZE: gl.constexpr,  #
    Mask,
    mask_base_idx,
    mask_row_stride,
    q_extend_offs,  #
    USE_CUSTOM_MASK: gl.constexpr,
    SKIP_PREFIX_CUSTOM_MASK: gl.constexpr,  #
    ENABLE_PREFIX_UNMASKED: gl.constexpr,  #
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,  #
    BLOCK_DMODEL: gl.constexpr,
    ACTUAL_BLOCK_DMODEL: gl.constexpr,  #
    NUM_STAGES: gl.constexpr,  #
    kt_async_layout: gl.constexpr,
    v_async_layout: gl.constexpr,  #
    kt_dot_layout: gl.constexpr,
    p_dot_layout: gl.constexpr,
    v_dot_layout: gl.constexpr,  #
    mma_layout: gl.constexpr,
    mma_offs_n_col: gl.constexpr,  #
):
    n_prefix_blocks = (seq_len_prefix + BLOCK_N - 1) // BLOCK_N
    k_prefix_base = K_Buffer + cur_kv_head * stride_buf_kh
    v_prefix_base = V_Buffer + cur_kv_head * stride_buf_vh

    kt_offs_n_layout: gl.constexpr = gl.SliceLayout(dim=0, parent=kt_async_layout)
    v_offs_n_layout: gl.constexpr = gl.SliceLayout(dim=1, parent=v_async_layout)
    kt_offs_n_pf = gl.arange(0, BLOCK_N, layout=kt_offs_n_layout)
    v_offs_n_pf = gl.arange(0, BLOCK_N, layout=v_offs_n_layout)

    for stage in gl.static_range(NUM_STAGES):
        start_n = stage * BLOCK_N
        issue_async_load_k_prefix(
            kt_smem.index(stage),
            k_prefix_base,
            kv_indices,
            kv_start,
            start_n,
            seq_len_prefix,
            stride_buf_kbs,
            BLOCK_N,
            BLOCK_DMODEL,
            ACTUAL_BLOCK_DMODEL,
            kt_async_layout,
        )
        issue_async_load_v_prefix(
            v_smem.index(stage),
            v_prefix_base,
            kv_indices,
            kv_start,
            start_n,
            seq_len_prefix,
            stride_buf_vbs,
            BLOCK_N,
            BLOCK_DMODEL,
            ACTUAL_BLOCK_DMODEL,
            v_async_layout,
        )

    pf_start_n = NUM_STAGES * BLOCK_N
    n_idx_kt_pf = pf_start_n + kt_offs_n_pf
    mask_n_kt_pf = n_idx_kt_pf < seq_len_prefix
    kv_locs_kt_pf = gl.load(
        kv_indices + kv_start + n_idx_kt_pf, mask=mask_n_kt_pf, other=0
    ).to(tl.int32)
    n_idx_v_pf = pf_start_n + v_offs_n_pf
    mask_n_v_pf = n_idx_v_pf < seq_len_prefix
    kv_locs_v_pf = gl.load(
        kv_indices + kv_start + n_idx_v_pf, mask=mask_n_v_pf, other=0
    ).to(tl.int32)

    WAIT_K: gl.constexpr = 2 * NUM_STAGES - 1
    WAIT_V: gl.constexpr = 2 * NUM_STAGES - 2

    cdna4_async.wait_group(WAIT_K)
    kt_dot = cdna4_async.load_shared_relaxed(kt_smem.index(0), kt_dot_layout)

    main_loop_end = n_prefix_blocks - NUM_STAGES
    for block_n in tl.range(0, main_loop_end, loop_unroll_factor=2):

        with warp_pipeline_stage("compute0", priority=0):
            stage_idx = (block_n % NUM_STAGES).to(tl.int32)
            start_n = (block_n * BLOCK_N).to(tl.int32)
            qk = gl.zeros([BLOCK_M, BLOCK_N], dtype=gl.float32, layout=mma_layout)
            qk = do_mma(q_dot, kt_dot, qk)

        cdna4_async.wait_group(WAIT_V)

        with warp_pipeline_stage("memory0", priority=1):
            v_dot = cdna4_async.load_shared_relaxed(
                v_smem.index(stage_idx), v_dot_layout
            )
            issue_dma_k_prefix_from_locs(
                kt_smem.index(stage_idx),
                k_prefix_base,
                kv_locs_kt_pf,
                mask_n_kt_pf,
                stride_buf_kbs,
                BLOCK_N,
                BLOCK_DMODEL,
                ACTUAL_BLOCK_DMODEL,
                kt_async_layout,
            )

        with warp_pipeline_stage("compute1", priority=0):
            acc, l_i, m_i, p = compute_softmax_prefix(
                acc,
                l_i,
                m_i,
                qk,
                start_n,
                seq_len_prefix,
                qk_scale,
                LOGIT_CAP,
                xai_temperature_reg,
                XAI_TEMPERATURE_LEN,
                q_abs_pos,
                SLIDING_WINDOW_SIZE,
                Mask,
                mask_base_idx,
                mask_row_stride,
                q_extend_offs,
                USE_CUSTOM_MASK,
                SKIP_PREFIX_CUSTOM_MASK,
                ENABLE_PREFIX_UNMASKED,
                BLOCK_M,
                BLOCK_N,
                mma_layout,
                mma_offs_n_col,
            )

        with warp_pipeline_stage("memory1", priority=1):
            issue_dma_v_prefix_from_locs(
                v_smem.index(stage_idx),
                v_prefix_base,
                kv_locs_v_pf,
                mask_n_v_pf,
                stride_buf_vbs,
                BLOCK_N,
                BLOCK_DMODEL,
                ACTUAL_BLOCK_DMODEL,
                v_async_layout,
            )
            nf_start_n = ((block_n + NUM_STAGES + 1) * BLOCK_N).to(tl.int32)
            n_idx_kt_nf = nf_start_n + kt_offs_n_pf
            mask_n_kt_pf = n_idx_kt_nf < seq_len_prefix
            kv_locs_kt_pf = gl.load(
                kv_indices + kv_start + n_idx_kt_nf, mask=mask_n_kt_pf, other=0
            ).to(tl.int32)

        with warp_pipeline_stage("compute2", priority=0):
            p_cast = p.to(v_dot.dtype)
            p_dot_reg = gl.convert_layout(p_cast, p_dot_layout)
            acc = do_mma(p_dot_reg, v_dot, acc)

        cdna4_async.wait_group(WAIT_K)

        with warp_pipeline_stage("memory2", priority=1):
            next_stage_idx = ((block_n + 1) % NUM_STAGES).to(tl.int32)
            kt_dot = cdna4_async.load_shared_relaxed(
                kt_smem.index(next_stage_idx), kt_dot_layout
            )
            n_idx_v_nf = nf_start_n + v_offs_n_pf
            mask_n_v_pf = n_idx_v_nf < seq_len_prefix
            kv_locs_v_pf = gl.load(
                kv_indices + kv_start + n_idx_v_nf, mask=mask_n_v_pf, other=0
            ).to(tl.int32)

    for tail_i in gl.static_range(NUM_STAGES):
        cdna4_async.wait_group(2 * (NUM_STAGES - tail_i) - 1)
        stage_idx = ((main_loop_end + tail_i) % NUM_STAGES).to(tl.int32)
        start_n = (main_loop_end + tail_i) * BLOCK_N

        kt_dot_tail = cdna4_async.load_shared_relaxed(
            kt_smem.index(stage_idx), kt_dot_layout
        )
        qk = gl.zeros([BLOCK_M, BLOCK_N], dtype=gl.float32, layout=mma_layout)
        qk = do_mma(q_dot, kt_dot_tail, qk)

        acc, l_i, m_i, p = compute_softmax_prefix(
            acc,
            l_i,
            m_i,
            qk,
            start_n,
            seq_len_prefix,
            qk_scale,
            LOGIT_CAP,
            xai_temperature_reg,
            XAI_TEMPERATURE_LEN,
            q_abs_pos,
            SLIDING_WINDOW_SIZE,
            Mask,
            mask_base_idx,
            mask_row_stride,
            q_extend_offs,
            USE_CUSTOM_MASK,
            SKIP_PREFIX_CUSTOM_MASK,
            ENABLE_PREFIX_UNMASKED,
            BLOCK_M,
            BLOCK_N,
            mma_layout,
            mma_offs_n_col,
        )

        cdna4_async.wait_group(2 * (NUM_STAGES - tail_i) - 2)
        v_dot_tail = cdna4_async.load_shared_relaxed(
            v_smem.index(stage_idx), v_dot_layout
        )
        p_cast = p.to(v_dot_tail.dtype)
        p_dot_tail = gl.convert_layout(p_cast, p_dot_layout)
        acc = do_mma(p_dot_tail, v_dot_tail, acc)

    return acc, l_i, m_i


@gluon.jit
def attn_fwd_inner_prefix_pipelined_scalar_mask(
    acc,
    l_i,
    m_i,
    q_dot,  #
    K_Buffer,
    V_Buffer,
    kv_indices,  #
    kv_start,
    cur_kv_head,
    seq_len_prefix,  #
    stride_buf_kbs,
    stride_buf_kh,  #
    stride_buf_vbs,
    stride_buf_vh,  #
    kt_smem,
    v_smem,  #
    qk_scale: gl.constexpr,
    LOGIT_CAP: gl.constexpr,  #
    xai_temperature_reg,
    XAI_TEMPERATURE_LEN: gl.constexpr,  #
    q_abs_pos,
    SLIDING_WINDOW_SIZE: gl.constexpr,  #
    Mask,
    mask_base_idx,
    mask_row_stride,
    q_extend_offs,  #
    USE_CUSTOM_MASK: gl.constexpr,
    SKIP_PREFIX_CUSTOM_MASK: gl.constexpr,  #
    ENABLE_PREFIX_UNMASKED: gl.constexpr,  #
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,  #
    BLOCK_DMODEL: gl.constexpr,
    ACTUAL_BLOCK_DMODEL: gl.constexpr,  #
    NUM_STAGES: gl.constexpr,  #
    kt_async_layout: gl.constexpr,
    v_async_layout: gl.constexpr,  #
    kt_dot_layout: gl.constexpr,
    p_dot_layout: gl.constexpr,
    v_dot_layout: gl.constexpr,  #
    mma_layout: gl.constexpr,
    mma_offs_n_col: gl.constexpr,  #
):
    n_prefix_blocks = (seq_len_prefix + BLOCK_N - 1) // BLOCK_N
    k_prefix_base = K_Buffer + cur_kv_head * stride_buf_kh
    v_prefix_base = V_Buffer + cur_kv_head * stride_buf_vh

    kt_offs_n_layout: gl.constexpr = gl.SliceLayout(dim=0, parent=kt_async_layout)
    v_offs_n_layout: gl.constexpr = gl.SliceLayout(dim=1, parent=v_async_layout)
    kt_offs_n_pf = gl.arange(0, BLOCK_N, layout=kt_offs_n_layout)
    v_offs_n_pf = gl.arange(0, BLOCK_N, layout=v_offs_n_layout)

    for stage in gl.static_range(NUM_STAGES):
        start_n = stage * BLOCK_N
        issue_async_load_k_prefix(
            kt_smem.index(stage),
            k_prefix_base,
            kv_indices,
            kv_start,
            start_n,
            seq_len_prefix,
            stride_buf_kbs,
            BLOCK_N,
            BLOCK_DMODEL,
            ACTUAL_BLOCK_DMODEL,
            kt_async_layout,
        )
        issue_async_load_v_prefix(
            v_smem.index(stage),
            v_prefix_base,
            kv_indices,
            kv_start,
            start_n,
            seq_len_prefix,
            stride_buf_vbs,
            BLOCK_N,
            BLOCK_DMODEL,
            ACTUAL_BLOCK_DMODEL,
            v_async_layout,
        )

    pf_start_n = NUM_STAGES * BLOCK_N
    n_idx_kt_pf = pf_start_n + kt_offs_n_pf
    mask_n_kt_pf = n_idx_kt_pf < seq_len_prefix
    kv_locs_kt_pf = gl.load(
        kv_indices + kv_start + n_idx_kt_pf, mask=mask_n_kt_pf, other=0
    ).to(tl.int32)
    n_idx_v_pf = pf_start_n + v_offs_n_pf
    mask_n_v_pf = n_idx_v_pf < seq_len_prefix
    kv_locs_v_pf = gl.load(
        kv_indices + kv_start + n_idx_v_pf, mask=mask_n_v_pf, other=0
    ).to(tl.int32)

    WAIT_K: gl.constexpr = 2 * NUM_STAGES - 1
    WAIT_V: gl.constexpr = 2 * NUM_STAGES - 2

    dma_mask = seq_len_prefix > 0

    cdna4_async.wait_group(WAIT_K)
    kt_dot = cdna4_async.load_shared_relaxed(kt_smem.index(0), kt_dot_layout)

    main_loop_end = n_prefix_blocks - NUM_STAGES
    for block_n in tl.range(0, main_loop_end, loop_unroll_factor=2):

        with warp_pipeline_stage("compute0", priority=0):
            stage_idx = (block_n % NUM_STAGES).to(tl.int32)
            start_n = (block_n * BLOCK_N).to(tl.int32)
            qk = gl.zeros([BLOCK_M, BLOCK_N], dtype=gl.float32, layout=mma_layout)
            qk = do_mma(q_dot, kt_dot, qk)

        cdna4_async.wait_group(WAIT_V)

        with warp_pipeline_stage("memory0", priority=1):
            v_dot = cdna4_async.load_shared_relaxed(
                v_smem.index(stage_idx), v_dot_layout
            )
            issue_dma_k_prefix_from_locs_hot(
                kt_smem.index(stage_idx),
                k_prefix_base,
                kv_locs_kt_pf,
                dma_mask,
                stride_buf_kbs,
                BLOCK_N,
                BLOCK_DMODEL,
                ACTUAL_BLOCK_DMODEL,
                kt_async_layout,
            )

        with warp_pipeline_stage("compute1", priority=0):
            acc, l_i, m_i, p = compute_softmax_prefix(
                acc,
                l_i,
                m_i,
                qk,
                start_n,
                seq_len_prefix,
                qk_scale,
                LOGIT_CAP,
                xai_temperature_reg,
                XAI_TEMPERATURE_LEN,
                q_abs_pos,
                SLIDING_WINDOW_SIZE,
                Mask,
                mask_base_idx,
                mask_row_stride,
                q_extend_offs,
                USE_CUSTOM_MASK,
                SKIP_PREFIX_CUSTOM_MASK,
                ENABLE_PREFIX_UNMASKED,
                BLOCK_M,
                BLOCK_N,
                mma_layout,
                mma_offs_n_col,
            )

        with warp_pipeline_stage("memory1", priority=1):
            issue_dma_v_prefix_from_locs_hot(
                v_smem.index(stage_idx),
                v_prefix_base,
                kv_locs_v_pf,
                dma_mask,
                stride_buf_vbs,
                BLOCK_N,
                BLOCK_DMODEL,
                ACTUAL_BLOCK_DMODEL,
                v_async_layout,
            )
            nf_start_n = ((block_n + NUM_STAGES + 1) * BLOCK_N).to(tl.int32)
            n_idx_kt_nf = nf_start_n + kt_offs_n_pf
            mask_n_kt_pf = n_idx_kt_nf < seq_len_prefix
            kv_locs_kt_pf = gl.load(
                kv_indices + kv_start + n_idx_kt_nf, mask=mask_n_kt_pf, other=0
            ).to(tl.int32)

        with warp_pipeline_stage("compute2", priority=0):
            p_cast = p.to(v_dot.dtype)
            p_dot_reg = gl.convert_layout(p_cast, p_dot_layout)
            acc = do_mma(p_dot_reg, v_dot, acc)

        cdna4_async.wait_group(WAIT_K)

        with warp_pipeline_stage("memory2", priority=1):
            next_stage_idx = ((block_n + 1) % NUM_STAGES).to(tl.int32)
            kt_dot = cdna4_async.load_shared_relaxed(
                kt_smem.index(next_stage_idx), kt_dot_layout
            )
            n_idx_v_nf = nf_start_n + v_offs_n_pf
            mask_n_v_pf = n_idx_v_nf < seq_len_prefix
            kv_locs_v_pf = gl.load(
                kv_indices + kv_start + n_idx_v_nf, mask=mask_n_v_pf, other=0
            ).to(tl.int32)

    for tail_i in gl.static_range(NUM_STAGES):
        cdna4_async.wait_group(2 * (NUM_STAGES - tail_i) - 1)
        stage_idx = ((main_loop_end + tail_i) % NUM_STAGES).to(tl.int32)
        start_n = (main_loop_end + tail_i) * BLOCK_N

        kt_dot_tail = cdna4_async.load_shared_relaxed(
            kt_smem.index(stage_idx), kt_dot_layout
        )
        qk = gl.zeros([BLOCK_M, BLOCK_N], dtype=gl.float32, layout=mma_layout)
        qk = do_mma(q_dot, kt_dot_tail, qk)

        acc, l_i, m_i, p = compute_softmax_prefix(
            acc,
            l_i,
            m_i,
            qk,
            start_n,
            seq_len_prefix,
            qk_scale,
            LOGIT_CAP,
            xai_temperature_reg,
            XAI_TEMPERATURE_LEN,
            q_abs_pos,
            SLIDING_WINDOW_SIZE,
            Mask,
            mask_base_idx,
            mask_row_stride,
            q_extend_offs,
            USE_CUSTOM_MASK,
            SKIP_PREFIX_CUSTOM_MASK,
            ENABLE_PREFIX_UNMASKED,
            BLOCK_M,
            BLOCK_N,
            mma_layout,
            mma_offs_n_col,
        )

        cdna4_async.wait_group(2 * (NUM_STAGES - tail_i) - 2)
        v_dot_tail = cdna4_async.load_shared_relaxed(
            v_smem.index(stage_idx), v_dot_layout
        )
        p_cast = p.to(v_dot_tail.dtype)
        p_dot_tail = gl.convert_layout(p_cast, p_dot_layout)
        acc = do_mma(p_dot_tail, v_dot_tail, acc)

    return acc, l_i, m_i


@gluon.jit
def attn_fwd_inner_prefix_short(
    acc,
    l_i,
    m_i,
    q_dot,  #
    K_Buffer,
    V_Buffer,
    kv_indices,  #
    kv_start,
    cur_kv_head,
    seq_len_prefix,  #
    stride_buf_kbs,
    stride_buf_kh,  #
    stride_buf_vbs,
    stride_buf_vh,  #
    kt_smem,
    v_smem,  #
    qk_scale: gl.constexpr,
    LOGIT_CAP: gl.constexpr,  #
    xai_temperature_reg,
    XAI_TEMPERATURE_LEN: gl.constexpr,  #
    q_abs_pos,
    SLIDING_WINDOW_SIZE: gl.constexpr,  #
    Mask,
    mask_base_idx,
    mask_row_stride,
    q_extend_offs,  #
    USE_CUSTOM_MASK: gl.constexpr,
    SKIP_PREFIX_CUSTOM_MASK: gl.constexpr,  #
    ENABLE_PREFIX_UNMASKED: gl.constexpr,  #
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,  #
    BLOCK_DMODEL: gl.constexpr,
    ACTUAL_BLOCK_DMODEL: gl.constexpr,  #
    kt_async_layout: gl.constexpr,
    v_async_layout: gl.constexpr,  #
    kt_dot_layout: gl.constexpr,
    p_dot_layout: gl.constexpr,
    v_dot_layout: gl.constexpr,  #
    mma_layout: gl.constexpr,
    mma_offs_n_col: gl.constexpr,  #
):
    n_prefix_blocks = (seq_len_prefix + BLOCK_N - 1) // BLOCK_N
    k_prefix_base = K_Buffer + cur_kv_head * stride_buf_kh
    v_prefix_base = V_Buffer + cur_kv_head * stride_buf_vh

    for block_n in tl.range(0, n_prefix_blocks):
        start_n = block_n * BLOCK_N

        issue_async_load_k_prefix(
            kt_smem.index(0),
            k_prefix_base,
            kv_indices,
            kv_start,
            start_n,
            seq_len_prefix,
            stride_buf_kbs,
            BLOCK_N,
            BLOCK_DMODEL,
            ACTUAL_BLOCK_DMODEL,
            kt_async_layout,
        )
        issue_async_load_v_prefix(
            v_smem.index(0),
            v_prefix_base,
            kv_indices,
            kv_start,
            start_n,
            seq_len_prefix,
            stride_buf_vbs,
            BLOCK_N,
            BLOCK_DMODEL,
            ACTUAL_BLOCK_DMODEL,
            v_async_layout,
        )

        cdna4_async.wait_group(1)
        kt_dot = cdna4_async.load_shared_relaxed(kt_smem.index(0), kt_dot_layout)
        qk = gl.zeros([BLOCK_M, BLOCK_N], dtype=gl.float32, layout=mma_layout)
        qk = do_mma(q_dot, kt_dot, qk)

        acc, l_i, m_i, p = compute_softmax_prefix(
            acc,
            l_i,
            m_i,
            qk,
            start_n,
            seq_len_prefix,
            qk_scale,
            LOGIT_CAP,
            xai_temperature_reg,
            XAI_TEMPERATURE_LEN,
            q_abs_pos,
            SLIDING_WINDOW_SIZE,
            Mask,
            mask_base_idx,
            mask_row_stride,
            q_extend_offs,
            USE_CUSTOM_MASK,
            SKIP_PREFIX_CUSTOM_MASK,
            ENABLE_PREFIX_UNMASKED,
            BLOCK_M,
            BLOCK_N,
            mma_layout,
            mma_offs_n_col,
        )

        cdna4_async.wait_group(0)
        v_dot = cdna4_async.load_shared_relaxed(v_smem.index(0), v_dot_layout)
        p_cast = p.to(v_dot.dtype)
        p_dot = gl.convert_layout(p_cast, p_dot_layout)
        acc = do_mma(p_dot, v_dot, acc)

    return acc, l_i, m_i


@gluon.jit
def attn_fwd_inner_prefix_dma_simple(
    acc,
    l_i,
    m_i,
    q_dot,  #
    K_Buffer,
    V_Buffer,
    kv_indices,  #
    kv_start,
    cur_kv_head,
    seq_len_prefix,  #
    stride_buf_kbs,
    stride_buf_kh,  #
    stride_buf_vbs,
    stride_buf_vh,  #
    kt_smem,
    v_smem,  #
    qk_scale: gl.constexpr,
    LOGIT_CAP: gl.constexpr,  #
    xai_temperature_reg,
    XAI_TEMPERATURE_LEN: gl.constexpr,  #
    q_abs_pos,
    SLIDING_WINDOW_SIZE: gl.constexpr,  #
    Mask,
    mask_base_idx,
    mask_row_stride,
    q_extend_offs,  #
    USE_CUSTOM_MASK: gl.constexpr,
    SKIP_PREFIX_CUSTOM_MASK: gl.constexpr,  #
    ENABLE_PREFIX_UNMASKED: gl.constexpr,  #
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,  #
    BLOCK_DMODEL: gl.constexpr,
    ACTUAL_BLOCK_DMODEL: gl.constexpr,  #
    NUM_STAGES: gl.constexpr,  #
    kt_async_layout: gl.constexpr,
    v_async_layout: gl.constexpr,  #
    kt_dot_layout: gl.constexpr,
    p_dot_layout: gl.constexpr,
    v_dot_layout: gl.constexpr,  #
    mma_layout: gl.constexpr,
    mma_offs_n_col: gl.constexpr,  #
):
    n_prefix_blocks = (seq_len_prefix + BLOCK_N - 1) // BLOCK_N
    k_prefix_base = K_Buffer + cur_kv_head * stride_buf_kh
    v_prefix_base = V_Buffer + cur_kv_head * stride_buf_vh
    kv_indices_base = kv_indices + kv_start

    kt_offs_n_layout: gl.constexpr = gl.SliceLayout(dim=0, parent=kt_async_layout)
    v_offs_n_layout: gl.constexpr = gl.SliceLayout(dim=1, parent=v_async_layout)
    kt_offs_n = gl.arange(0, BLOCK_N, layout=kt_offs_n_layout)
    v_offs_n = gl.arange(0, BLOCK_N, layout=v_offs_n_layout)

    WAIT_K: gl.constexpr = 2 * NUM_STAGES - 1
    WAIT_V: gl.constexpr = 2 * NUM_STAGES - 2

    for stage in gl.static_range(NUM_STAGES):
        cdna4_async.wait_group(0)
        start_n = stage * BLOCK_N
        n_idx_k = start_n + kt_offs_n
        mask_n_k = n_idx_k < seq_len_prefix
        kv_locs_k = cdna4_buffer_load(
            kv_indices_base, n_idx_k.to(tl.int32), mask=mask_n_k, other=0
        ).to(tl.int32)
        n_idx_v = start_n + v_offs_n
        mask_n_v = n_idx_v < seq_len_prefix
        kv_locs_v = cdna4_buffer_load(
            kv_indices_base, n_idx_v.to(tl.int32), mask=mask_n_v, other=0
        ).to(tl.int32)
        issue_dma_k_prefix_from_locs(
            kt_smem.index(stage),
            k_prefix_base,
            kv_locs_k,
            mask_n_k,
            stride_buf_kbs,
            BLOCK_N,
            BLOCK_DMODEL,
            ACTUAL_BLOCK_DMODEL,
            kt_async_layout,
        )
        issue_dma_v_prefix_from_locs(
            v_smem.index(stage),
            v_prefix_base,
            kv_locs_v,
            mask_n_v,
            stride_buf_vbs,
            BLOCK_N,
            BLOCK_DMODEL,
            ACTUAL_BLOCK_DMODEL,
            v_async_layout,
        )

    cdna4_async.wait_group(0)
    pf_start_n = NUM_STAGES * BLOCK_N
    n_idx_k_pf = pf_start_n + kt_offs_n
    mask_k_pf = n_idx_k_pf < seq_len_prefix
    kv_locs_k_pf = cdna4_buffer_load(
        kv_indices_base, n_idx_k_pf.to(tl.int32), mask=mask_k_pf, other=0
    ).to(tl.int32)
    n_idx_v_pf = pf_start_n + v_offs_n
    mask_v_pf = n_idx_v_pf < seq_len_prefix
    kv_locs_v_pf = cdna4_buffer_load(
        kv_indices_base, n_idx_v_pf.to(tl.int32), mask=mask_v_pf, other=0
    ).to(tl.int32)

    cdna4_async.wait_group(WAIT_K)
    kt_dot = cdna4_async.load_shared_relaxed(kt_smem.index(0), kt_dot_layout)

    main_loop_end = n_prefix_blocks - NUM_STAGES
    for block_n in tl.range(0, main_loop_end):
        stage_idx = (block_n % NUM_STAGES).to(tl.int32)
        start_n = (block_n * BLOCK_N).to(tl.int32)

        nf_start_n = ((block_n + NUM_STAGES + 1) * BLOCK_N).to(tl.int32)
        n_idx_k_nf = nf_start_n + kt_offs_n
        mask_k_nf = n_idx_k_nf < seq_len_prefix
        kv_locs_k_nf = cdna4_buffer_load(
            kv_indices_base, n_idx_k_nf.to(tl.int32), mask=mask_k_nf, other=0
        ).to(tl.int32)
        n_idx_v_nf = nf_start_n + v_offs_n
        mask_v_nf = n_idx_v_nf < seq_len_prefix
        kv_locs_v_nf = cdna4_buffer_load(
            kv_indices_base, n_idx_v_nf.to(tl.int32), mask=mask_v_nf, other=0
        ).to(tl.int32)
        issue_dma_k_prefix_from_locs(
            kt_smem.index(stage_idx),
            k_prefix_base,
            kv_locs_k_pf,
            mask_k_pf,
            stride_buf_kbs,
            BLOCK_N,
            BLOCK_DMODEL,
            ACTUAL_BLOCK_DMODEL,
            kt_async_layout,
        )

        qk = gl.zeros([BLOCK_M, BLOCK_N], dtype=gl.float32, layout=mma_layout)
        qk = do_mma(q_dot, kt_dot, qk)

        cdna4_async.wait_group(WAIT_V)
        v_dot = cdna4_async.load_shared_relaxed(v_smem.index(stage_idx), v_dot_layout)

        acc, l_i, m_i, p = compute_softmax_prefix(
            acc,
            l_i,
            m_i,
            qk,
            start_n,
            seq_len_prefix,
            qk_scale,
            LOGIT_CAP,
            xai_temperature_reg,
            XAI_TEMPERATURE_LEN,
            q_abs_pos,
            SLIDING_WINDOW_SIZE,
            Mask,
            mask_base_idx,
            mask_row_stride,
            q_extend_offs,
            USE_CUSTOM_MASK,
            SKIP_PREFIX_CUSTOM_MASK,
            ENABLE_PREFIX_UNMASKED,
            BLOCK_M,
            BLOCK_N,
            mma_layout,
            mma_offs_n_col,
        )

        issue_dma_v_prefix_from_locs(
            v_smem.index(stage_idx),
            v_prefix_base,
            kv_locs_v_pf,
            mask_v_pf,
            stride_buf_vbs,
            BLOCK_N,
            BLOCK_DMODEL,
            ACTUAL_BLOCK_DMODEL,
            v_async_layout,
        )

        p_cast = p.to(v_dot.dtype)
        p_dot_reg = gl.convert_layout(p_cast, p_dot_layout)
        acc = do_mma(p_dot_reg, v_dot, acc)

        next_stage_idx = ((block_n + 1) % NUM_STAGES).to(tl.int32)
        cdna4_async.wait_group(WAIT_K)
        kt_dot = cdna4_async.load_shared_relaxed(
            kt_smem.index(next_stage_idx), kt_dot_layout
        )

        kv_locs_k_pf = kv_locs_k_nf
        kv_locs_v_pf = kv_locs_v_nf
        mask_k_pf = mask_k_nf
        mask_v_pf = mask_v_nf

    for tail_i in gl.static_range(NUM_STAGES):
        cdna4_async.wait_group(2 * (NUM_STAGES - tail_i) - 1)
        stage_idx = ((main_loop_end + tail_i) % NUM_STAGES).to(tl.int32)
        start_n = (main_loop_end + tail_i) * BLOCK_N

        kt_dot_tail = cdna4_async.load_shared_relaxed(
            kt_smem.index(stage_idx), kt_dot_layout
        )
        qk = gl.zeros([BLOCK_M, BLOCK_N], dtype=gl.float32, layout=mma_layout)
        qk = do_mma(q_dot, kt_dot_tail, qk)

        acc, l_i, m_i, p = compute_softmax_prefix(
            acc,
            l_i,
            m_i,
            qk,
            start_n,
            seq_len_prefix,
            qk_scale,
            LOGIT_CAP,
            xai_temperature_reg,
            XAI_TEMPERATURE_LEN,
            q_abs_pos,
            SLIDING_WINDOW_SIZE,
            Mask,
            mask_base_idx,
            mask_row_stride,
            q_extend_offs,
            USE_CUSTOM_MASK,
            SKIP_PREFIX_CUSTOM_MASK,
            ENABLE_PREFIX_UNMASKED,
            BLOCK_M,
            BLOCK_N,
            mma_layout,
            mma_offs_n_col,
        )

        cdna4_async.wait_group(2 * (NUM_STAGES - tail_i) - 2)
        v_dot_tail = cdna4_async.load_shared_relaxed(
            v_smem.index(stage_idx), v_dot_layout
        )
        p_cast = p.to(v_dot_tail.dtype)
        p_dot_tail = gl.convert_layout(p_cast, p_dot_layout)
        acc = do_mma(p_dot_tail, v_dot_tail, acc)

    return acc, l_i, m_i


# ===-----------------------------------------------------------------------===#
# Extend inner loops
# ===-----------------------------------------------------------------------===#


@gluon.jit
def attn_fwd_inner_extend_dma(
    acc,
    l_i,
    m_i,
    q_dot,  #
    k_base,
    v_base,  #
    cur_block_m,
    seq_len_extend,  #
    stride_kbs,
    stride_vbs,  #
    block_start,
    block_end,  #
    kt_smem,
    v_smem,  #
    qk_scale: gl.constexpr,
    LOGIT_CAP: gl.constexpr,  #
    xai_temperature_reg,
    XAI_TEMPERATURE_LEN: gl.constexpr,  #
    SLIDING_WINDOW_SIZE: gl.constexpr,  #
    IS_CAUSAL: gl.constexpr,  #
    Mask,
    mask_base_idx,
    mask_row_stride,
    mask_kv_col_offset,  #
    USE_CUSTOM_MASK: gl.constexpr,  #
    MASK_STEPS: gl.constexpr,  #
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,  #
    BLOCK_DMODEL: gl.constexpr,
    ACTUAL_BLOCK_DMODEL: gl.constexpr,  #
    NUM_STAGES: gl.constexpr,  #
    kt_async_layout: gl.constexpr,
    v_async_layout: gl.constexpr,  #
    kt_dot_layout: gl.constexpr,
    p_dot_layout: gl.constexpr,
    v_dot_layout: gl.constexpr,  #
    mma_layout: gl.constexpr,
    mma_offs_n_col: gl.constexpr,
    mma_offs_m_row: gl.constexpr,  #
):
    cdna4_async.wait_group(0)

    for stage in gl.static_range(NUM_STAGES):
        start_n = (block_start + stage) * BLOCK_N
        issue_async_load_k_extend(
            kt_smem.index(stage),
            k_base,
            start_n,
            seq_len_extend,
            stride_kbs,
            BLOCK_N,
            BLOCK_DMODEL,
            ACTUAL_BLOCK_DMODEL,
            kt_async_layout,
        )
        issue_async_load_v_extend(
            v_smem.index(stage),
            v_base,
            start_n,
            seq_len_extend,
            stride_vbs,
            BLOCK_N,
            BLOCK_DMODEL,
            ACTUAL_BLOCK_DMODEL,
            v_async_layout,
        )

    WAIT_K: gl.constexpr = 2 * NUM_STAGES - 1
    WAIT_V: gl.constexpr = 2 * NUM_STAGES - 2
    cdna4_async.wait_group(WAIT_K)
    kt_dot = cdna4_async.load_shared_relaxed(kt_smem.index(0), kt_dot_layout)

    main_loop_end = block_end - NUM_STAGES
    for block_n in tl.range(block_start, main_loop_end, loop_unroll_factor=2):
        stage_idx = ((block_n - block_start) % NUM_STAGES).to(tl.int32)
        start_n = (block_n * BLOCK_N).to(tl.int32)
        future_start_n = ((block_n + NUM_STAGES) * BLOCK_N).to(tl.int32)

        qk = gl.zeros([BLOCK_M, BLOCK_N], dtype=gl.float32, layout=mma_layout)
        qk = do_mma(q_dot, kt_dot, qk)

        cdna4_async.wait_group(WAIT_V)
        v_dot = cdna4_async.load_shared_relaxed(v_smem.index(stage_idx), v_dot_layout)
        issue_async_load_k_extend(
            kt_smem.index(stage_idx),
            k_base,
            future_start_n,
            seq_len_extend,
            stride_kbs,
            BLOCK_N,
            BLOCK_DMODEL,
            ACTUAL_BLOCK_DMODEL,
            kt_async_layout,
        )

        acc, l_i, m_i, p = compute_softmax_extend(
            acc,
            l_i,
            m_i,
            qk,
            start_n,
            cur_block_m,
            seq_len_extend,
            qk_scale,
            LOGIT_CAP,
            xai_temperature_reg,
            XAI_TEMPERATURE_LEN,
            SLIDING_WINDOW_SIZE,
            IS_CAUSAL,
            Mask,
            mask_base_idx,
            mask_row_stride,
            mask_kv_col_offset,
            USE_CUSTOM_MASK,
            MASK_STEPS,
            BLOCK_M,
            BLOCK_N,
            mma_layout,
            mma_offs_n_col,
            mma_offs_m_row,
        )
        p_cast = p.to(v_dot.dtype)
        p_dot_reg = gl.convert_layout(p_cast, p_dot_layout)
        acc = do_mma(p_dot_reg, v_dot, acc)

        cdna4_async.wait_group(WAIT_V)
        next_stage_idx = ((block_n + 1 - block_start) % NUM_STAGES).to(tl.int32)
        kt_dot = cdna4_async.load_shared_relaxed(
            kt_smem.index(next_stage_idx), kt_dot_layout
        )
        issue_async_load_v_extend(
            v_smem.index(stage_idx),
            v_base,
            future_start_n,
            seq_len_extend,
            stride_vbs,
            BLOCK_N,
            BLOCK_DMODEL,
            ACTUAL_BLOCK_DMODEL,
            v_async_layout,
        )

    for tail_i in gl.static_range(NUM_STAGES):
        cdna4_async.wait_group(2 * (NUM_STAGES - tail_i) - 1)
        stage_idx = ((main_loop_end + tail_i - block_start) % NUM_STAGES).to(tl.int32)
        start_n = (main_loop_end + tail_i) * BLOCK_N

        kt_dot_tail = cdna4_async.load_shared_relaxed(
            kt_smem.index(stage_idx), kt_dot_layout
        )
        qk = gl.zeros([BLOCK_M, BLOCK_N], dtype=gl.float32, layout=mma_layout)
        qk = do_mma(q_dot, kt_dot_tail, qk)

        acc, l_i, m_i, p = compute_softmax_extend(
            acc,
            l_i,
            m_i,
            qk,
            start_n,
            cur_block_m,
            seq_len_extend,
            qk_scale,
            LOGIT_CAP,
            xai_temperature_reg,
            XAI_TEMPERATURE_LEN,
            SLIDING_WINDOW_SIZE,
            IS_CAUSAL,
            Mask,
            mask_base_idx,
            mask_row_stride,
            mask_kv_col_offset,
            USE_CUSTOM_MASK,
            MASK_STEPS,
            BLOCK_M,
            BLOCK_N,
            mma_layout,
            mma_offs_n_col,
            mma_offs_m_row,
        )

        cdna4_async.wait_group(2 * (NUM_STAGES - tail_i) - 2)
        v_dot_tail = cdna4_async.load_shared_relaxed(
            v_smem.index(stage_idx), v_dot_layout
        )
        p_cast = p.to(v_dot_tail.dtype)
        p_dot_tail = gl.convert_layout(p_cast, p_dot_layout)
        acc = do_mma(p_dot_tail, v_dot_tail, acc)

    return acc, l_i, m_i


@gluon.jit
def attn_fwd_inner_extend_pipelined(
    acc,
    l_i,
    m_i,
    q_dot,  #
    k_base,
    v_base,  #
    cur_block_m,
    seq_len_extend,  #
    stride_kbs,
    stride_vbs,  #
    block_start,
    block_end,  #
    kt_smem,
    v_smem,  #
    qk_scale: gl.constexpr,
    LOGIT_CAP: gl.constexpr,  #
    xai_temperature_reg,
    XAI_TEMPERATURE_LEN: gl.constexpr,  #
    SLIDING_WINDOW_SIZE: gl.constexpr,  #
    IS_CAUSAL: gl.constexpr,  #
    Mask,
    mask_base_idx,
    mask_row_stride,
    mask_kv_col_offset,  #
    USE_CUSTOM_MASK: gl.constexpr,  #
    MASK_STEPS: gl.constexpr,  #
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,  #
    BLOCK_DMODEL: gl.constexpr,
    ACTUAL_BLOCK_DMODEL: gl.constexpr,  #
    NUM_STAGES: gl.constexpr,  #
    kt_async_layout: gl.constexpr,
    v_async_layout: gl.constexpr,  #
    kt_dot_layout: gl.constexpr,
    p_dot_layout: gl.constexpr,
    v_dot_layout: gl.constexpr,  #
    mma_layout: gl.constexpr,
    mma_offs_n_col: gl.constexpr,
    mma_offs_m_row: gl.constexpr,  #
):
    cdna4_async.wait_group(0)

    for stage in gl.static_range(NUM_STAGES):
        start_n = (block_start + stage) * BLOCK_N
        issue_async_load_k_extend(
            kt_smem.index(stage),
            k_base,
            start_n,
            seq_len_extend,
            stride_kbs,
            BLOCK_N,
            BLOCK_DMODEL,
            ACTUAL_BLOCK_DMODEL,
            kt_async_layout,
        )
        issue_async_load_v_extend(
            v_smem.index(stage),
            v_base,
            start_n,
            seq_len_extend,
            stride_vbs,
            BLOCK_N,
            BLOCK_DMODEL,
            ACTUAL_BLOCK_DMODEL,
            v_async_layout,
        )

    WAIT_INIT: gl.constexpr = 2 * NUM_STAGES - 1
    WAIT_LOOP: gl.constexpr = 2 * NUM_STAGES - 2
    cdna4_async.wait_group(WAIT_INIT)
    kt_dot = cdna4_async.load_shared_relaxed(kt_smem.index(0), kt_dot_layout)

    main_loop_end = block_end - NUM_STAGES
    for block_n in tl.range(block_start, main_loop_end, loop_unroll_factor=2):

        with warp_pipeline_stage("dot1", priority=0):
            stage_idx = ((block_n - block_start) % NUM_STAGES).to(tl.int32)
            start_n = (block_n * BLOCK_N).to(tl.int32)
            future_start_n = ((block_n + NUM_STAGES) * BLOCK_N).to(tl.int32)
            qk = gl.zeros([BLOCK_M, BLOCK_N], dtype=gl.float32, layout=mma_layout)
            qk = do_mma(q_dot, kt_dot, qk)

        cdna4_async.wait_group(WAIT_LOOP)

        with warp_pipeline_stage("mem1", priority=1):
            v_dot = cdna4_async.load_shared_relaxed(
                v_smem.index(stage_idx), v_dot_layout
            )
            issue_async_load_k_extend(
                kt_smem.index(stage_idx),
                k_base,
                future_start_n,
                seq_len_extend,
                stride_kbs,
                BLOCK_N,
                BLOCK_DMODEL,
                ACTUAL_BLOCK_DMODEL,
                kt_async_layout,
            )

        with warp_pipeline_stage("dot2a", priority=0):
            acc, l_i, m_i, p = compute_softmax_extend(
                acc,
                l_i,
                m_i,
                qk,
                start_n,
                cur_block_m,
                seq_len_extend,
                qk_scale,
                LOGIT_CAP,
                xai_temperature_reg,
                XAI_TEMPERATURE_LEN,
                SLIDING_WINDOW_SIZE,
                IS_CAUSAL,
                Mask,
                mask_base_idx,
                mask_row_stride,
                mask_kv_col_offset,
                USE_CUSTOM_MASK,
                MASK_STEPS,
                BLOCK_M,
                BLOCK_N,
                mma_layout,
                mma_offs_n_col,
                mma_offs_m_row,
            )

        with warp_pipeline_stage("dot2b", priority=0):
            p_cast = p.to(v_dot.dtype)
            p_dot_reg = gl.convert_layout(p_cast, p_dot_layout)
            acc = do_mma(p_dot_reg, v_dot, acc)

        cdna4_async.wait_group(WAIT_LOOP)

        with warp_pipeline_stage("mem2", priority=1):
            next_stage_idx = ((block_n + 1 - block_start) % NUM_STAGES).to(tl.int32)
            kt_dot = cdna4_async.load_shared_relaxed(
                kt_smem.index(next_stage_idx), kt_dot_layout
            )
            issue_async_load_v_extend(
                v_smem.index(stage_idx),
                v_base,
                future_start_n,
                seq_len_extend,
                stride_vbs,
                BLOCK_N,
                BLOCK_DMODEL,
                ACTUAL_BLOCK_DMODEL,
                v_async_layout,
            )

    for tail_i in gl.static_range(NUM_STAGES):
        cdna4_async.wait_group(2 * (NUM_STAGES - tail_i) - 1)
        stage_idx = ((main_loop_end + tail_i - block_start) % NUM_STAGES).to(tl.int32)
        start_n = (main_loop_end + tail_i) * BLOCK_N

        kt_dot_tail = cdna4_async.load_shared_relaxed(
            kt_smem.index(stage_idx), kt_dot_layout
        )
        qk = gl.zeros([BLOCK_M, BLOCK_N], dtype=gl.float32, layout=mma_layout)
        qk = do_mma(q_dot, kt_dot_tail, qk)

        acc, l_i, m_i, p = compute_softmax_extend(
            acc,
            l_i,
            m_i,
            qk,
            start_n,
            cur_block_m,
            seq_len_extend,
            qk_scale,
            LOGIT_CAP,
            xai_temperature_reg,
            XAI_TEMPERATURE_LEN,
            SLIDING_WINDOW_SIZE,
            IS_CAUSAL,
            Mask,
            mask_base_idx,
            mask_row_stride,
            mask_kv_col_offset,
            USE_CUSTOM_MASK,
            MASK_STEPS,
            BLOCK_M,
            BLOCK_N,
            mma_layout,
            mma_offs_n_col,
            mma_offs_m_row,
        )

        cdna4_async.wait_group(2 * (NUM_STAGES - tail_i) - 2)
        v_dot_tail = cdna4_async.load_shared_relaxed(
            v_smem.index(stage_idx), v_dot_layout
        )
        p_cast = p.to(v_dot_tail.dtype)
        p_dot_tail = gl.convert_layout(p_cast, p_dot_layout)
        acc = do_mma(p_dot_tail, v_dot_tail, acc)

    return acc, l_i, m_i


# ===-----------------------------------------------------------------------===#
# Serial inner loops (4-warp path)
# ===-----------------------------------------------------------------------===#


@gluon.jit
def attn_fwd_inner_prefix_serial(
    acc,
    l_i,
    m_i,
    q_dot,  #
    K_Buffer,
    V_Buffer,
    kv_indices,  #
    kv_start,
    kv_head,
    seq_len_prefix,  #
    stride_buf_kbs,
    stride_buf_kh,  #
    stride_buf_vbs,
    stride_buf_vh,  #
    kt_serial_smem,
    v_serial_smem,  #
    qk_scale: gl.constexpr,
    LOGIT_CAP: gl.constexpr,  #
    xai_temperature_reg,
    XAI_TEMPERATURE_LEN: gl.constexpr,  #
    q_abs_pos,
    SLIDING_WINDOW_SIZE: gl.constexpr,  #
    Mask,
    mask_base_idx,
    mask_row_stride,
    q_extend_offs,  #
    USE_CUSTOM_MASK: gl.constexpr,
    SKIP_PREFIX_CUSTOM_MASK: gl.constexpr,  #
    ENABLE_PREFIX_UNMASKED: gl.constexpr,  #
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,  #
    BLOCK_DMODEL: gl.constexpr,
    ACTUAL_BLOCK_DMODEL: gl.constexpr,  #
    kt_blocked_layout: gl.constexpr,
    blocked_layout: gl.constexpr,  #
    kt_dot_layout: gl.constexpr,
    p_dot_layout: gl.constexpr,
    v_dot_layout: gl.constexpr,  #
    mma_layout: gl.constexpr,
    mma_offs_n_col: gl.constexpr,  #
):
    kt_offs_d = gl.arange(
        0, BLOCK_DMODEL, layout=gl.SliceLayout(dim=1, parent=kt_blocked_layout)
    )
    kt_offs_n = gl.arange(
        0, BLOCK_N, layout=gl.SliceLayout(dim=0, parent=kt_blocked_layout)
    )
    v_offs_n = gl.arange(
        0, BLOCK_N, layout=gl.SliceLayout(dim=1, parent=blocked_layout)
    )
    v_offs_d = gl.arange(
        0, BLOCK_DMODEL, layout=gl.SliceLayout(dim=0, parent=blocked_layout)
    )

    k_prefix_base = K_Buffer + kv_head * stride_buf_kh
    v_prefix_base = V_Buffer + kv_head * stride_buf_vh
    n_prefix_blocks = (seq_len_prefix + BLOCK_N - 1) // BLOCK_N

    for block_n in tl.range(0, n_prefix_blocks):
        start_n = block_n * BLOCK_N

        n_idx_k = start_n + kt_offs_n
        mask_n_k = n_idx_k < seq_len_prefix
        kv_locs_k = gl.load(kv_indices + kv_start + n_idx_k, mask=mask_n_k, other=0).to(
            tl.int32
        )

        kt_ptrs = (
            k_prefix_base + kt_offs_d[:, None] + kv_locs_k[None, :] * stride_buf_kbs
        )
        kt_mask = mask_n_k[None, :]
        if ACTUAL_BLOCK_DMODEL != BLOCK_DMODEL:
            kt_mask = kt_mask & (kt_offs_d[:, None] < ACTUAL_BLOCK_DMODEL)
        kt_global = gl.load(kt_ptrs, mask=kt_mask, other=0.0)
        kt_serial_smem.store(kt_global)
        kt_dot = kt_serial_smem.load(kt_dot_layout)

        qk = gl.zeros([BLOCK_M, BLOCK_N], dtype=gl.float32, layout=mma_layout)
        qk = do_mma(q_dot, kt_dot, qk)

        acc, l_i, m_i, p = compute_softmax_prefix(
            acc,
            l_i,
            m_i,
            qk,
            start_n,
            seq_len_prefix,
            qk_scale,
            LOGIT_CAP,
            xai_temperature_reg,
            XAI_TEMPERATURE_LEN,
            q_abs_pos,
            SLIDING_WINDOW_SIZE,
            Mask,
            mask_base_idx,
            mask_row_stride,
            q_extend_offs,
            USE_CUSTOM_MASK,
            SKIP_PREFIX_CUSTOM_MASK,
            ENABLE_PREFIX_UNMASKED,
            BLOCK_M,
            BLOCK_N,
            mma_layout,
            mma_offs_n_col,
        )

        n_idx_v = start_n + v_offs_n
        mask_n_v = n_idx_v < seq_len_prefix
        kv_locs_v = gl.load(kv_indices + kv_start + n_idx_v, mask=mask_n_v, other=0).to(
            tl.int32
        )

        v_ptrs = v_prefix_base + kv_locs_v[:, None] * stride_buf_vbs + v_offs_d[None, :]
        v_mask = mask_n_v[:, None]
        if ACTUAL_BLOCK_DMODEL != BLOCK_DMODEL:
            v_mask = v_mask & (v_offs_d[None, :] < ACTUAL_BLOCK_DMODEL)
        v_global = gl.load(v_ptrs, mask=v_mask, other=0.0)
        v_serial_smem.store(v_global)
        v_dot = v_serial_smem.load(v_dot_layout)

        p_cast = p.to(v_dot.dtype)
        p_dot = gl.convert_layout(p_cast, p_dot_layout)
        acc = do_mma(p_dot, v_dot, acc)

    return acc, l_i, m_i


@gluon.jit
def attn_fwd_inner_extend_serial(
    acc,
    l_i,
    m_i,
    q_dot,  #
    k_extend_base,
    v_extend_base,  #
    cur_block_m,
    seq_len_extend,  #
    stride_kbs,
    stride_vbs,  #
    block_start,
    block_end,  #
    kt_serial_smem,
    v_serial_smem,  #
    qk_scale: gl.constexpr,
    LOGIT_CAP: gl.constexpr,  #
    xai_temperature_reg,
    XAI_TEMPERATURE_LEN: gl.constexpr,  #
    SLIDING_WINDOW_SIZE: gl.constexpr,  #
    IS_CAUSAL: gl.constexpr,  #
    Mask,
    mask_base_idx,
    mask_row_stride,
    mask_kv_col_offset,  #
    USE_CUSTOM_MASK: gl.constexpr,  #
    MASK_STEPS: gl.constexpr,  #
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,  #
    BLOCK_DMODEL: gl.constexpr,
    ACTUAL_BLOCK_DMODEL: gl.constexpr,  #
    kt_blocked_layout: gl.constexpr,
    blocked_layout: gl.constexpr,  #
    kt_dot_layout: gl.constexpr,
    p_dot_layout: gl.constexpr,
    v_dot_layout: gl.constexpr,  #
    mma_layout: gl.constexpr,
    mma_offs_n_col: gl.constexpr,
    mma_offs_m_row: gl.constexpr,  #
):
    kt_offs_d = gl.arange(
        0, BLOCK_DMODEL, layout=gl.SliceLayout(dim=1, parent=kt_blocked_layout)
    )
    kt_offs_n = gl.arange(
        0, BLOCK_N, layout=gl.SliceLayout(dim=0, parent=kt_blocked_layout)
    )
    v_offs_n = gl.arange(
        0, BLOCK_N, layout=gl.SliceLayout(dim=1, parent=blocked_layout)
    )
    v_offs_d = gl.arange(
        0, BLOCK_DMODEL, layout=gl.SliceLayout(dim=0, parent=blocked_layout)
    )

    for block_n in tl.range(block_start, block_end):
        start_n = block_n * BLOCK_N

        kt_ptrs = (
            k_extend_base
            + kt_offs_d[:, None]
            + (start_n + kt_offs_n[None, :]) * stride_kbs
        )
        kt_mask = (start_n + kt_offs_n[None, :]) < seq_len_extend
        if ACTUAL_BLOCK_DMODEL != BLOCK_DMODEL:
            kt_mask = kt_mask & (kt_offs_d[:, None] < ACTUAL_BLOCK_DMODEL)
        kt_global = gl.load(kt_ptrs, mask=kt_mask, other=0.0)
        kt_serial_smem.store(kt_global)
        kt_dot = kt_serial_smem.load(kt_dot_layout)

        qk = gl.zeros([BLOCK_M, BLOCK_N], dtype=gl.float32, layout=mma_layout)
        qk = do_mma(q_dot, kt_dot, qk)

        acc, l_i, m_i, p = compute_softmax_extend(
            acc,
            l_i,
            m_i,
            qk,
            start_n,
            cur_block_m,
            seq_len_extend,
            qk_scale,
            LOGIT_CAP,
            xai_temperature_reg,
            XAI_TEMPERATURE_LEN,
            SLIDING_WINDOW_SIZE,
            IS_CAUSAL,
            Mask,
            mask_base_idx,
            mask_row_stride,
            mask_kv_col_offset,
            USE_CUSTOM_MASK,
            MASK_STEPS,
            BLOCK_M,
            BLOCK_N,
            mma_layout,
            mma_offs_n_col,
            mma_offs_m_row,
        )

        v_ptrs = (
            v_extend_base
            + (start_n + v_offs_n[:, None]) * stride_vbs
            + v_offs_d[None, :]
        )
        v_mask = (start_n + v_offs_n[:, None]) < seq_len_extend
        if ACTUAL_BLOCK_DMODEL != BLOCK_DMODEL:
            v_mask = v_mask & (v_offs_d[None, :] < ACTUAL_BLOCK_DMODEL)
        v_global = gl.load(v_ptrs, mask=v_mask, other=0.0)
        v_serial_smem.store(v_global)
        v_dot = v_serial_smem.load(v_dot_layout)

        p_cast = p.to(v_dot.dtype)
        p_dot = gl.convert_layout(p_cast, p_dot_layout)
        acc = do_mma(p_dot, v_dot, acc)

    return acc, l_i, m_i


# ===-----------------------------------------------------------------------===#
# Extend short path (preload-all, defined in this file)
# ===-----------------------------------------------------------------------===#


@gluon.jit
def attn_fwd_inner_extend_short(
    acc,
    l_i,
    m_i,
    q_dot,  #
    k_base,
    v_base,  #
    cur_block_m,
    seq_len_extend,  #
    stride_kbs,
    stride_vbs,  #
    block_start,
    block_end,  #
    kt_smem,
    v_smem,  #
    qk_scale: gl.constexpr,
    LOGIT_CAP: gl.constexpr,  #
    xai_temperature_reg,
    XAI_TEMPERATURE_LEN: gl.constexpr,  #
    SLIDING_WINDOW_SIZE: gl.constexpr,  #
    IS_CAUSAL: gl.constexpr,  #
    Mask,
    mask_base_idx,
    mask_row_stride,
    mask_kv_col_offset,  #
    USE_CUSTOM_MASK: gl.constexpr,  #
    MASK_STEPS: gl.constexpr,  #
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,  #
    BLOCK_DMODEL: gl.constexpr,
    ACTUAL_BLOCK_DMODEL: gl.constexpr,  #
    kt_async_layout: gl.constexpr,
    v_async_layout: gl.constexpr,  #
    kt_dot_layout: gl.constexpr,
    p_dot_layout: gl.constexpr,
    v_dot_layout: gl.constexpr,  #
    mma_layout: gl.constexpr,
    mma_offs_n_col: gl.constexpr,
    mma_offs_m_row: gl.constexpr,  #
):
    """Preload all K/V blocks into distinct SMEM slots, then compute.

    Requires (block_end - block_start) <= NUM_STAGES so each block gets its own buffer.
    Phase 1: issue K[i]->smem[i] and V[i]->smem[i] for all i.
    Phase 2: wait_group(0) -- everything fully drained.
    Phase 3: compute QK.softmax.PV block-by-block from resident SMEM.
    """
    cdna4_async.wait_group(0)

    # Phase 1: bulk-issue all DMAs
    n_local_blocks = block_end - block_start
    for local_idx in tl.range(0, n_local_blocks):
        start_n = (block_start + local_idx) * BLOCK_N
        issue_async_load_k_extend(
            kt_smem.index(local_idx),
            k_base,
            start_n,
            seq_len_extend,
            stride_kbs,
            BLOCK_N,
            BLOCK_DMODEL,
            ACTUAL_BLOCK_DMODEL,
            kt_async_layout,
        )
        issue_async_load_v_extend(
            v_smem.index(local_idx),
            v_base,
            start_n,
            seq_len_extend,
            stride_vbs,
            BLOCK_N,
            BLOCK_DMODEL,
            ACTUAL_BLOCK_DMODEL,
            v_async_layout,
        )

    # Phase 2: drain every outstanding DMA
    cdna4_async.wait_group(0)

    # Phase 3: compute from fully-resident SMEM (no DMA during this loop)
    for local_idx in tl.range(0, n_local_blocks):
        start_n = (block_start + local_idx) * BLOCK_N

        kt_dot = cdna4_async.load_shared_relaxed(
            kt_smem.index(local_idx), kt_dot_layout
        )
        qk = gl.zeros([BLOCK_M, BLOCK_N], dtype=gl.float32, layout=mma_layout)
        qk = do_mma(q_dot, kt_dot, qk)

        acc, l_i, m_i, p = compute_softmax_extend(
            acc,
            l_i,
            m_i,
            qk,
            start_n,
            cur_block_m,
            seq_len_extend,
            qk_scale,
            LOGIT_CAP,
            xai_temperature_reg,
            XAI_TEMPERATURE_LEN,
            SLIDING_WINDOW_SIZE,
            IS_CAUSAL,
            Mask,
            mask_base_idx,
            mask_row_stride,
            mask_kv_col_offset,
            USE_CUSTOM_MASK,
            MASK_STEPS,
            BLOCK_M,
            BLOCK_N,
            mma_layout,
            mma_offs_n_col,
            mma_offs_m_row,
        )

        v_dot = cdna4_async.load_shared_relaxed(v_smem.index(local_idx), v_dot_layout)
        p_cast = p.to(v_dot.dtype)
        p_dot = gl.convert_layout(p_cast, p_dot_layout)
        acc = do_mma(p_dot, v_dot, acc)

    return acc, l_i, m_i


# ===-----------------------------------------------------------------------===#
# Main Kernel
# ===-----------------------------------------------------------------------===#


@gluon.jit
def gluon_extend_attn_fwd(
    Q_Extend,
    K_Extend,
    V_Extend,
    O_Extend,  #
    K_Buffer,
    V_Buffer,  #
    qo_indptr,
    kv_indptr,
    kv_indices,  #
    Mask,
    MaskIndptr,
    WindowKvOffsets,  #
    SM_SCALE: gl.constexpr,
    kv_group_num,  #
    stride_qbs,
    stride_qh,  #
    stride_kbs,
    stride_kh,  #
    stride_vbs,
    stride_vh,  #
    stride_obs,
    stride_oh,  #
    stride_buf_kbs,
    stride_buf_kh,  #
    stride_buf_vbs,
    stride_buf_vh,  #
    IS_CAUSAL: gl.constexpr,  #
    USE_CUSTOM_MASK: gl.constexpr,
    SKIP_PREFIX_CUSTOM_MASK: gl.constexpr,  #
    ENABLE_PREFIX_UNMASKED: gl.constexpr,
    ENABLE_MASK_SPLIT: gl.constexpr,  #
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,  #
    BLOCK_DMODEL: gl.constexpr,
    ACTUAL_BLOCK_DMODEL: gl.constexpr,  #
    NUM_STAGES: gl.constexpr,  #
    MMA_INSTR_M: gl.constexpr,
    MMA_INSTR_N: gl.constexpr,
    MMA_INSTR_K: gl.constexpr,  #
    QK_K_WIDTH: gl.constexpr,
    PV_K_WIDTH: gl.constexpr,  #
    ASYNC_PAD_K: gl.constexpr,
    ASYNC_PAD_V: gl.constexpr,  #
    Sinks,
    HAS_SINK: gl.constexpr,  #
    LOGIT_CAP: gl.constexpr,  #
    XAI_TEMPERATURE_LEN: gl.constexpr,  #
    SLIDING_WINDOW_SIZE: gl.constexpr,  #
    V_SCALE: gl.constexpr,  #
):
    num_warps: gl.constexpr = gl.num_warps()

    cur_seq = gl.program_id(0)
    cur_head = gl.program_id(1)
    cur_block_m = gl.program_id(2)
    cur_kv_head = cur_head // kv_group_num

    cur_seq_q_start_idx = gl.load(qo_indptr + cur_seq)
    seq_len_extend = gl.load(qo_indptr + cur_seq + 1) - cur_seq_q_start_idx
    cur_seq_kv_start_idx = gl.load(kv_indptr + cur_seq)
    seq_len_prefix = gl.load(kv_indptr + cur_seq + 1) - cur_seq_kv_start_idx

    if cur_block_m * BLOCK_M >= seq_len_extend:
        return

    if USE_CUSTOM_MASK:
        mask_base_idx = gl.load(MaskIndptr + cur_seq).to(tl.int64)
        window_kv_offset = 0
        if SLIDING_WINDOW_SIZE > 0:
            window_kv_offset = gl.load(WindowKvOffsets + cur_seq)
        cur_seq_len = seq_len_prefix + seq_len_extend
        mask_row_stride = (cur_seq_len + window_kv_offset).to(tl.int64)
        mask_base_idx = mask_base_idx + window_kv_offset.to(tl.int64)
        mask_kv_col_offset = (seq_len_prefix).to(tl.int64)
    else:
        mask_base_idx = tl.cast(0, tl.int64)
        mask_row_stride = tl.cast(0, tl.int64)
        mask_kv_col_offset = tl.cast(0, tl.int64)

    # layouts (same as v1)
    mma_layout: gl.constexpr = AMDMFMALayout(
        version=4,
        instr_shape=[MMA_INSTR_M, MMA_INSTR_N, MMA_INSTR_K],
        transposed=True,
        warps_per_cta=[num_warps, 1],
    )
    k_width: gl.constexpr = QK_K_WIDTH
    threads_per_warp: gl.constexpr = 64
    pv_k_width: gl.constexpr = PV_K_WIDTH

    q_dot_layout: gl.constexpr = DotOperandLayout(
        operand_index=0, parent=mma_layout, k_width=k_width
    )
    kt_dot_layout: gl.constexpr = DotOperandLayout(
        operand_index=1, parent=mma_layout, k_width=k_width
    )
    p_dot_layout: gl.constexpr = DotOperandLayout(
        operand_index=0, parent=mma_layout, k_width=pv_k_width
    )
    v_dot_layout: gl.constexpr = DotOperandLayout(
        operand_index=1, parent=mma_layout, k_width=pv_k_width
    )

    blocked_layout: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 8],
        threads_per_warp=[threads_per_warp // 4, 4],
        warps_per_cta=[num_warps, 1],
        order=[1, 0],
    )

    offs_m_layout: gl.constexpr = gl.SliceLayout(dim=1, parent=blocked_layout)
    offs_d_layout: gl.constexpr = gl.SliceLayout(dim=0, parent=blocked_layout)
    mma_offs_n_col: gl.constexpr = gl.SliceLayout(dim=0, parent=mma_layout)
    mma_offs_m_row: gl.constexpr = gl.SliceLayout(dim=1, parent=mma_layout)
    mma_m_layout: gl.constexpr = gl.SliceLayout(dim=1, parent=mma_layout)

    offs_m = gl.arange(0, BLOCK_M, layout=offs_m_layout)
    offs_d = gl.arange(0, BLOCK_DMODEL, layout=offs_d_layout)

    USE_SERIAL: gl.constexpr = num_warps < 8

    # Q load
    q_ptrs = (
        Q_Extend
        + (cur_seq_q_start_idx + cur_block_m * BLOCK_M + offs_m[:, None]) * stride_qbs
        + cur_head * stride_qh
        + offs_d[None, :]
    )
    q_mask = (cur_block_m * BLOCK_M + offs_m[:, None]) < seq_len_extend
    if ACTUAL_BLOCK_DMODEL != BLOCK_DMODEL:
        q_mask = q_mask & (offs_d[None, :] < ACTUAL_BLOCK_DMODEL)
    q = gl.load(q_ptrs, mask=q_mask, other=0.0)

    # softmax state
    m_i = gl.full([BLOCK_M], float("-inf"), dtype=gl.float32, layout=mma_m_layout)
    l_i = gl.full([BLOCK_M], 1.0, dtype=gl.float32, layout=mma_m_layout)
    acc = gl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=gl.float32, layout=mma_layout)
    qk_scale: gl.constexpr = SM_SCALE * LOG2E

    q_abs_pos = (
        seq_len_prefix
        + cur_block_m * BLOCK_M
        + gl.arange(0, BLOCK_M, layout=mma_offs_m_row)
    )
    q_extend_raw = cur_block_m * BLOCK_M + gl.arange(0, BLOCK_M, layout=mma_offs_m_row)
    if USE_CUSTOM_MASK:
        q_extend_offs = tl.minimum(q_extend_raw, tl.maximum(seq_len_extend - 1, 0))
    else:
        q_extend_offs = q_extend_raw

    if XAI_TEMPERATURE_LEN > 0:
        inv_log2_len = 1.0 / tl.log2(float(XAI_TEMPERATURE_LEN))
        xai_temperature_reg = gl.where(
            q_abs_pos > XAI_TEMPERATURE_LEN,
            tl.log2(q_abs_pos.to(gl.float32)) * inv_log2_len,
            gl.full([BLOCK_M], 1.0, dtype=gl.float32, layout=mma_offs_m_row),
        )
    else:
        xai_temperature_reg = gl.full(
            [BLOCK_M], 1.0, dtype=gl.float32, layout=mma_offs_m_row
        )

    # SWA prefix skip: jump past prefix tiles entirely outside the window.
    # For the M-tile, min q_abs_pos = seq_len_prefix + cur_block_m * BLOCK_M.
    # Any prefix block whose last key position < (min_q - SWS) is fully masked.
    pfx_kv_start = cur_seq_kv_start_idx
    pfx_seq_len = seq_len_prefix
    pfx_q_abs_pos = q_abs_pos
    pfx_mask_base = mask_base_idx
    if SLIDING_WINDOW_SIZE > 0:
        q_min_abs = seq_len_prefix + cur_block_m * BLOCK_M
        first_useful_pos = tl.maximum(q_min_abs - SLIDING_WINDOW_SIZE, 0)
        prefix_skip_n = (first_useful_pos // BLOCK_N) * BLOCK_N
        pfx_kv_start = cur_seq_kv_start_idx + prefix_skip_n
        pfx_seq_len = seq_len_prefix - prefix_skip_n
        pfx_q_abs_pos = q_abs_pos - prefix_skip_n
        if USE_CUSTOM_MASK:
            pfx_mask_base = mask_base_idx + prefix_skip_n.to(tl.int64)

    if USE_SERIAL:
        if NUM_STAGES >= 2 and BLOCK_DMODEL >= 128:
            # 4-warp DMA path
            kt_offset_bases: gl.constexpr = [
                [1, 0],
                [2, 0],
                [4, 0],
                [8, 0],
                [16, 0],
                [32, 0],
                [64, 0],
                [0, 16],
                [0, 32],
                [0, 1],
                [0, 2],
                [0, 4],
                [0, 8],
            ]
            v_offset_bases: gl.constexpr = [
                [0, 1],
                [0, 2],
                [0, 4],
                [0, 8],
                [0, 16],
                [0, 32],
                [0, 64],
                [16, 0],
                [32, 0],
                [1, 0],
                [2, 0],
                [4, 0],
                [8, 0],
            ]
            kt_async_layout: gl.constexpr = DistributedLinearLayout(
                reg_bases=[[1, 0], [2, 0], [4, 0], [0, 4], [0, 8]],
                lane_bases=[[8, 0], [16, 0], [32, 0], [64, 0], [0, 16], [0, 32]],
                warp_bases=[[0, 1], [0, 2]],
                block_bases=[],
                shape=[BLOCK_DMODEL, BLOCK_N],
            )
            v_async_layout: gl.constexpr = DistributedLinearLayout(
                reg_bases=[[0, 1], [0, 2], [0, 4], [4, 0], [8, 0]],
                lane_bases=[[0, 8], [0, 16], [0, 32], [0, 64], [16, 0], [32, 0]],
                warp_bases=[[1, 0], [2, 0]],
                block_bases=[],
                shape=[BLOCK_N, BLOCK_DMODEL],
            )

            kt_smem_layout: gl.constexpr = PaddedSharedLayout(
                interval_padding_pairs=[[512, ASYNC_PAD_K]],
                offset_bases=kt_offset_bases,
                cga_layout=[],
                shape=[BLOCK_DMODEL, BLOCK_N],
            )
            v_smem_layout: gl.constexpr = PaddedSharedLayout(
                interval_padding_pairs=[[512, ASYNC_PAD_V]],
                offset_bases=v_offset_bases,
                cga_layout=[],
                shape=[BLOCK_N, BLOCK_DMODEL],
            )

            kt_smem = gl.allocate_shared_memory(
                Q_Extend.dtype.element_ty,
                [NUM_STAGES, BLOCK_DMODEL, BLOCK_N],
                layout=kt_smem_layout,
            )
            v_smem = gl.allocate_shared_memory(
                Q_Extend.dtype.element_ty,
                [NUM_STAGES, BLOCK_N, BLOCK_DMODEL],
                layout=v_smem_layout,
            )

            for _s in gl.static_range(NUM_STAGES):
                v_zero = gl.zeros(
                    [BLOCK_N, BLOCK_DMODEL],
                    dtype=Q_Extend.dtype.element_ty,
                    layout=v_async_layout,
                )
                v_smem.index(_s).store(v_zero)
            gl.barrier()

            q_dot = gl.convert_layout(q, q_dot_layout)

            # prefix (same as v1)
            if pfx_seq_len > 0:
                n_prefix_blocks = (pfx_seq_len + BLOCK_N - 1) // BLOCK_N
                n_extend_est = (seq_len_extend + BLOCK_N - 1) // BLOCK_N
                use_pipe_prefix = n_prefix_blocks >= NUM_STAGES
                if LOGIT_CAP > 0:
                    use_pipe_prefix = use_pipe_prefix and (n_extend_est < NUM_STAGES)
                if use_pipe_prefix:
                    acc, l_i, m_i = attn_fwd_inner_prefix_dma_simple(
                        acc,
                        l_i,
                        m_i,
                        q_dot,
                        K_Buffer,
                        V_Buffer,
                        kv_indices,
                        pfx_kv_start,
                        cur_kv_head,
                        pfx_seq_len,
                        stride_buf_kbs,
                        stride_buf_kh,
                        stride_buf_vbs,
                        stride_buf_vh,
                        kt_smem,
                        v_smem,
                        qk_scale,
                        LOGIT_CAP,
                        xai_temperature_reg,
                        XAI_TEMPERATURE_LEN,
                        pfx_q_abs_pos,
                        SLIDING_WINDOW_SIZE,
                        Mask,
                        pfx_mask_base,
                        mask_row_stride,
                        q_extend_offs,
                        USE_CUSTOM_MASK,
                        SKIP_PREFIX_CUSTOM_MASK,
                        ENABLE_PREFIX_UNMASKED,
                        BLOCK_M,
                        BLOCK_N,
                        BLOCK_DMODEL,
                        ACTUAL_BLOCK_DMODEL,
                        NUM_STAGES,
                        kt_async_layout,
                        v_async_layout,
                        kt_dot_layout,
                        p_dot_layout,
                        v_dot_layout,
                        mma_layout,
                        mma_offs_n_col,
                    )
                else:
                    acc, l_i, m_i = attn_fwd_inner_prefix_short(
                        acc,
                        l_i,
                        m_i,
                        q_dot,
                        K_Buffer,
                        V_Buffer,
                        kv_indices,
                        pfx_kv_start,
                        cur_kv_head,
                        pfx_seq_len,
                        stride_buf_kbs,
                        stride_buf_kh,
                        stride_buf_vbs,
                        stride_buf_vh,
                        kt_smem,
                        v_smem,
                        qk_scale,
                        LOGIT_CAP,
                        xai_temperature_reg,
                        XAI_TEMPERATURE_LEN,
                        pfx_q_abs_pos,
                        SLIDING_WINDOW_SIZE,
                        Mask,
                        pfx_mask_base,
                        mask_row_stride,
                        q_extend_offs,
                        USE_CUSTOM_MASK,
                        SKIP_PREFIX_CUSTOM_MASK,
                        ENABLE_PREFIX_UNMASKED,
                        BLOCK_M,
                        BLOCK_N,
                        BLOCK_DMODEL,
                        ACTUAL_BLOCK_DMODEL,
                        kt_async_layout,
                        v_async_layout,
                        kt_dot_layout,
                        p_dot_layout,
                        v_dot_layout,
                        mma_layout,
                        mma_offs_n_col,
                    )

            cdna4_async.wait_group(0)

            # EXTEND: per-CTA dispatch (v2 change)
            if IS_CAUSAL:
                causal_kv_end = (cur_block_m + 1) * BLOCK_M
                effective_end = tl.minimum(seq_len_extend, causal_kv_end)
            else:
                effective_end = seq_len_extend
            n_extend_blocks = (effective_end + BLOCK_N - 1) // BLOCK_N
            if (not ENABLE_MASK_SPLIT) or USE_CUSTOM_MASK or SLIDING_WINDOW_SIZE > 0:
                n_full_blocks = 0
            else:
                partial_block = ((effective_end % BLOCK_N) != 0).to(tl.int32)
                if IS_CAUSAL:
                    masked_blocks = ((BLOCK_M + BLOCK_N - 1) // BLOCK_N) + partial_block
                else:
                    masked_blocks = partial_block
                masked_blocks = tl.minimum(masked_blocks, n_extend_blocks)
                n_full_blocks = n_extend_blocks - masked_blocks

            k_extend_base = (
                K_Extend + cur_seq_q_start_idx * stride_kbs + cur_kv_head * stride_kh
            )
            v_extend_base = (
                V_Extend + cur_seq_q_start_idx * stride_vbs + cur_kv_head * stride_vh
            )

            if n_full_blocks >= NUM_STAGES:
                acc, l_i, m_i = attn_fwd_inner_extend_dma(
                    acc,
                    l_i,
                    m_i,
                    q_dot,
                    k_extend_base,
                    v_extend_base,
                    cur_block_m,
                    seq_len_extend,
                    stride_kbs,
                    stride_vbs,
                    0,
                    n_full_blocks,
                    kt_smem,
                    v_smem,
                    qk_scale,
                    LOGIT_CAP,
                    xai_temperature_reg,
                    XAI_TEMPERATURE_LEN,
                    SLIDING_WINDOW_SIZE,
                    IS_CAUSAL,
                    Mask,
                    mask_base_idx,
                    mask_row_stride,
                    mask_kv_col_offset,
                    USE_CUSTOM_MASK,
                    False,
                    BLOCK_M,
                    BLOCK_N,
                    BLOCK_DMODEL,
                    ACTUAL_BLOCK_DMODEL,
                    NUM_STAGES,
                    kt_async_layout,
                    v_async_layout,
                    kt_dot_layout,
                    p_dot_layout,
                    v_dot_layout,
                    mma_layout,
                    mma_offs_n_col,
                    mma_offs_m_row,
                )
            elif n_full_blocks > 0:
                acc, l_i, m_i = attn_fwd_inner_extend_short(
                    acc,
                    l_i,
                    m_i,
                    q_dot,
                    k_extend_base,
                    v_extend_base,
                    cur_block_m,
                    seq_len_extend,
                    stride_kbs,
                    stride_vbs,
                    0,
                    n_full_blocks,
                    kt_smem,
                    v_smem,
                    qk_scale,
                    LOGIT_CAP,
                    xai_temperature_reg,
                    XAI_TEMPERATURE_LEN,
                    SLIDING_WINDOW_SIZE,
                    IS_CAUSAL,
                    Mask,
                    mask_base_idx,
                    mask_row_stride,
                    mask_kv_col_offset,
                    USE_CUSTOM_MASK,
                    False,
                    BLOCK_M,
                    BLOCK_N,
                    BLOCK_DMODEL,
                    ACTUAL_BLOCK_DMODEL,
                    kt_async_layout,
                    v_async_layout,
                    kt_dot_layout,
                    p_dot_layout,
                    v_dot_layout,
                    mma_layout,
                    mma_offs_n_col,
                    mma_offs_m_row,
                )
            masked_start = n_full_blocks
            remaining_blocks = n_extend_blocks - masked_start
            if remaining_blocks >= NUM_STAGES:
                acc, l_i, m_i = attn_fwd_inner_extend_dma(
                    acc,
                    l_i,
                    m_i,
                    q_dot,
                    k_extend_base,
                    v_extend_base,
                    cur_block_m,
                    seq_len_extend,
                    stride_kbs,
                    stride_vbs,
                    masked_start,
                    n_extend_blocks,
                    kt_smem,
                    v_smem,
                    qk_scale,
                    LOGIT_CAP,
                    xai_temperature_reg,
                    XAI_TEMPERATURE_LEN,
                    SLIDING_WINDOW_SIZE,
                    IS_CAUSAL,
                    Mask,
                    mask_base_idx,
                    mask_row_stride,
                    mask_kv_col_offset,
                    USE_CUSTOM_MASK,
                    True,
                    BLOCK_M,
                    BLOCK_N,
                    BLOCK_DMODEL,
                    ACTUAL_BLOCK_DMODEL,
                    NUM_STAGES,
                    kt_async_layout,
                    v_async_layout,
                    kt_dot_layout,
                    p_dot_layout,
                    v_dot_layout,
                    mma_layout,
                    mma_offs_n_col,
                    mma_offs_m_row,
                )
            elif remaining_blocks > 0:
                acc, l_i, m_i = attn_fwd_inner_extend_short(
                    acc,
                    l_i,
                    m_i,
                    q_dot,
                    k_extend_base,
                    v_extend_base,
                    cur_block_m,
                    seq_len_extend,
                    stride_kbs,
                    stride_vbs,
                    masked_start,
                    n_extend_blocks,
                    kt_smem,
                    v_smem,
                    qk_scale,
                    LOGIT_CAP,
                    xai_temperature_reg,
                    XAI_TEMPERATURE_LEN,
                    SLIDING_WINDOW_SIZE,
                    IS_CAUSAL,
                    Mask,
                    mask_base_idx,
                    mask_row_stride,
                    mask_kv_col_offset,
                    USE_CUSTOM_MASK,
                    True,
                    BLOCK_M,
                    BLOCK_N,
                    BLOCK_DMODEL,
                    ACTUAL_BLOCK_DMODEL,
                    kt_async_layout,
                    v_async_layout,
                    kt_dot_layout,
                    p_dot_layout,
                    v_dot_layout,
                    mma_layout,
                    mma_offs_n_col,
                    mma_offs_m_row,
                )

        else:
            # 4-warp serial path (unchanged -- already correct for any n_extend_blocks)
            kt_blocked_layout: gl.constexpr = gl.BlockedLayout(
                size_per_thread=[1, 8],
                threads_per_warp=[threads_per_warp // 4, 4],
                warps_per_cta=[1, num_warps],
                order=[0, 1],
            )
            kt_serial_smem_layout: gl.constexpr = gl.SwizzledSharedLayout(
                vec=8,
                per_phase=1,
                max_phase=16,
                order=[0, 1],
            )
            v_serial_smem_layout: gl.constexpr = gl.SwizzledSharedLayout(
                vec=8,
                per_phase=1,
                max_phase=16,
                order=[1, 0],
            )
            q_smem_layout: gl.constexpr = gl.SwizzledSharedLayout(
                vec=8,
                per_phase=1,
                max_phase=16,
                order=[1, 0],
            )

            kt_serial_smem = gl.allocate_shared_memory(
                Q_Extend.dtype.element_ty,
                [BLOCK_DMODEL, BLOCK_N],
                layout=kt_serial_smem_layout,
            )
            v_serial_smem = gl.allocate_shared_memory(
                Q_Extend.dtype.element_ty,
                [BLOCK_N, BLOCK_DMODEL],
                layout=v_serial_smem_layout,
            )
            q_smem = gl.allocate_shared_memory(
                Q_Extend.dtype.element_ty,
                [BLOCK_M, BLOCK_DMODEL],
                layout=q_smem_layout,
            )

            q_smem.store(q)
            q_dot = q_smem.load(q_dot_layout)

            if pfx_seq_len > 0:
                acc, l_i, m_i = attn_fwd_inner_prefix_serial(
                    acc,
                    l_i,
                    m_i,
                    q_dot,
                    K_Buffer,
                    V_Buffer,
                    kv_indices,
                    pfx_kv_start,
                    cur_kv_head,
                    pfx_seq_len,
                    stride_buf_kbs,
                    stride_buf_kh,
                    stride_buf_vbs,
                    stride_buf_vh,
                    kt_serial_smem,
                    v_serial_smem,
                    qk_scale,
                    LOGIT_CAP,
                    xai_temperature_reg,
                    XAI_TEMPERATURE_LEN,
                    pfx_q_abs_pos,
                    SLIDING_WINDOW_SIZE,
                    Mask,
                    pfx_mask_base,
                    mask_row_stride,
                    q_extend_offs,
                    USE_CUSTOM_MASK,
                    SKIP_PREFIX_CUSTOM_MASK,
                    ENABLE_PREFIX_UNMASKED,
                    BLOCK_M,
                    BLOCK_N,
                    BLOCK_DMODEL,
                    ACTUAL_BLOCK_DMODEL,
                    kt_blocked_layout,
                    blocked_layout,
                    kt_dot_layout,
                    p_dot_layout,
                    v_dot_layout,
                    mma_layout,
                    mma_offs_n_col,
                )

            if IS_CAUSAL:
                causal_kv_end = (cur_block_m + 1) * BLOCK_M
                effective_end = tl.minimum(seq_len_extend, causal_kv_end)
            else:
                effective_end = seq_len_extend
            n_extend_blocks = (effective_end + BLOCK_N - 1) // BLOCK_N
            if (not ENABLE_MASK_SPLIT) or USE_CUSTOM_MASK or SLIDING_WINDOW_SIZE > 0:
                n_full_blocks = 0
            else:
                partial_block = ((effective_end % BLOCK_N) != 0).to(tl.int32)
                if IS_CAUSAL:
                    masked_blocks = ((BLOCK_M + BLOCK_N - 1) // BLOCK_N) + partial_block
                else:
                    masked_blocks = partial_block
                masked_blocks = tl.minimum(masked_blocks, n_extend_blocks)
                n_full_blocks = n_extend_blocks - masked_blocks

            k_extend_base = (
                K_Extend + cur_seq_q_start_idx * stride_kbs + cur_kv_head * stride_kh
            )
            v_extend_base = (
                V_Extend + cur_seq_q_start_idx * stride_vbs + cur_kv_head * stride_vh
            )

            if n_full_blocks > 0:
                acc, l_i, m_i = attn_fwd_inner_extend_serial(
                    acc,
                    l_i,
                    m_i,
                    q_dot,
                    k_extend_base,
                    v_extend_base,
                    cur_block_m,
                    seq_len_extend,
                    stride_kbs,
                    stride_vbs,
                    0,
                    n_full_blocks,
                    kt_serial_smem,
                    v_serial_smem,
                    qk_scale,
                    LOGIT_CAP,
                    xai_temperature_reg,
                    XAI_TEMPERATURE_LEN,
                    SLIDING_WINDOW_SIZE,
                    IS_CAUSAL,
                    Mask,
                    mask_base_idx,
                    mask_row_stride,
                    mask_kv_col_offset,
                    USE_CUSTOM_MASK,
                    False,
                    BLOCK_M,
                    BLOCK_N,
                    BLOCK_DMODEL,
                    ACTUAL_BLOCK_DMODEL,
                    kt_blocked_layout,
                    blocked_layout,
                    kt_dot_layout,
                    p_dot_layout,
                    v_dot_layout,
                    mma_layout,
                    mma_offs_n_col,
                    mma_offs_m_row,
                )
            masked_start = n_full_blocks
            if n_extend_blocks > masked_start:
                acc, l_i, m_i = attn_fwd_inner_extend_serial(
                    acc,
                    l_i,
                    m_i,
                    q_dot,
                    k_extend_base,
                    v_extend_base,
                    cur_block_m,
                    seq_len_extend,
                    stride_kbs,
                    stride_vbs,
                    masked_start,
                    n_extend_blocks,
                    kt_serial_smem,
                    v_serial_smem,
                    qk_scale,
                    LOGIT_CAP,
                    xai_temperature_reg,
                    XAI_TEMPERATURE_LEN,
                    SLIDING_WINDOW_SIZE,
                    IS_CAUSAL,
                    Mask,
                    mask_base_idx,
                    mask_row_stride,
                    mask_kv_col_offset,
                    USE_CUSTOM_MASK,
                    True,
                    BLOCK_M,
                    BLOCK_N,
                    BLOCK_DMODEL,
                    ACTUAL_BLOCK_DMODEL,
                    kt_blocked_layout,
                    blocked_layout,
                    kt_dot_layout,
                    p_dot_layout,
                    v_dot_layout,
                    mma_layout,
                    mma_offs_n_col,
                    mma_offs_m_row,
                )

    else:
        # 8-warp DMA path

        if BLOCK_DMODEL >= 128:
            kt_offset_bases: gl.constexpr = [
                [1, 0],
                [2, 0],
                [4, 0],
                [8, 0],
                [16, 0],
                [32, 0],
                [64, 0],
                [0, 16],
                [0, 32],
                [0, 1],
                [0, 2],
                [0, 4],
                [0, 8],
            ]
            v_offset_bases: gl.constexpr = [
                [0, 1],
                [0, 2],
                [0, 4],
                [0, 8],
                [0, 16],
                [0, 32],
                [0, 64],
                [16, 0],
                [32, 0],
                [1, 0],
                [2, 0],
                [4, 0],
                [8, 0],
            ]
            kt_async_layout: gl.constexpr = DistributedLinearLayout(
                reg_bases=[[1, 0], [2, 0], [4, 0], [0, 8]],
                lane_bases=[[8, 0], [16, 0], [32, 0], [64, 0], [0, 16], [0, 32]],
                warp_bases=[[0, 1], [0, 2], [0, 4]],
                block_bases=[],
                shape=[BLOCK_DMODEL, BLOCK_N],
            )
            v_async_layout: gl.constexpr = DistributedLinearLayout(
                reg_bases=[[0, 1], [0, 2], [0, 4], [8, 0]],
                lane_bases=[[0, 8], [0, 16], [0, 32], [0, 64], [16, 0], [32, 0]],
                warp_bases=[[1, 0], [2, 0], [4, 0]],
                block_bases=[],
                shape=[BLOCK_N, BLOCK_DMODEL],
            )

            kt_async_smem_layout: gl.constexpr = PaddedSharedLayout(
                interval_padding_pairs=[[512, ASYNC_PAD_K]],
                offset_bases=kt_offset_bases,
                cga_layout=[],
                shape=[BLOCK_DMODEL, BLOCK_N],
            )
            v_async_smem_layout: gl.constexpr = PaddedSharedLayout(
                interval_padding_pairs=[[512, ASYNC_PAD_V]],
                offset_bases=v_offset_bases,
                cga_layout=[],
                shape=[BLOCK_N, BLOCK_DMODEL],
            )

            kt_smem = gl.allocate_shared_memory(
                Q_Extend.dtype.element_ty,
                [NUM_STAGES, BLOCK_DMODEL, BLOCK_N],
                layout=kt_async_smem_layout,
            )
            v_smem = gl.allocate_shared_memory(
                Q_Extend.dtype.element_ty,
                [NUM_STAGES, BLOCK_N, BLOCK_DMODEL],
                layout=v_async_smem_layout,
            )

            for _s in gl.static_range(NUM_STAGES):
                v_zero = gl.zeros(
                    [BLOCK_N, BLOCK_DMODEL],
                    dtype=Q_Extend.dtype.element_ty,
                    layout=v_async_layout,
                )
                v_smem.index(_s).store(v_zero)
            gl.barrier()

            q_dot = gl.convert_layout(q, q_dot_layout)

            # prefix dispatch (same as v1)
            n_prefix_blocks = (pfx_seq_len + BLOCK_N - 1) // BLOCK_N
            if n_prefix_blocks >= NUM_STAGES:
                if NUM_STAGES >= 3:
                    acc, l_i, m_i = attn_fwd_inner_prefix_pipelined_scalar_mask(
                        acc,
                        l_i,
                        m_i,
                        q_dot,
                        K_Buffer,
                        V_Buffer,
                        kv_indices,
                        pfx_kv_start,
                        cur_kv_head,
                        pfx_seq_len,
                        stride_buf_kbs,
                        stride_buf_kh,
                        stride_buf_vbs,
                        stride_buf_vh,
                        kt_smem,
                        v_smem,
                        qk_scale,
                        LOGIT_CAP,
                        xai_temperature_reg,
                        XAI_TEMPERATURE_LEN,
                        pfx_q_abs_pos,
                        SLIDING_WINDOW_SIZE,
                        Mask,
                        pfx_mask_base,
                        mask_row_stride,
                        q_extend_offs,
                        USE_CUSTOM_MASK,
                        SKIP_PREFIX_CUSTOM_MASK,
                        ENABLE_PREFIX_UNMASKED,
                        BLOCK_M,
                        BLOCK_N,
                        BLOCK_DMODEL,
                        ACTUAL_BLOCK_DMODEL,
                        NUM_STAGES,
                        kt_async_layout,
                        v_async_layout,
                        kt_dot_layout,
                        p_dot_layout,
                        v_dot_layout,
                        mma_layout,
                        mma_offs_n_col,
                    )
                else:
                    acc, l_i, m_i = attn_fwd_inner_prefix_pipelined(
                        acc,
                        l_i,
                        m_i,
                        q_dot,
                        K_Buffer,
                        V_Buffer,
                        kv_indices,
                        pfx_kv_start,
                        cur_kv_head,
                        pfx_seq_len,
                        stride_buf_kbs,
                        stride_buf_kh,
                        stride_buf_vbs,
                        stride_buf_vh,
                        kt_smem,
                        v_smem,
                        qk_scale,
                        LOGIT_CAP,
                        xai_temperature_reg,
                        XAI_TEMPERATURE_LEN,
                        pfx_q_abs_pos,
                        SLIDING_WINDOW_SIZE,
                        Mask,
                        pfx_mask_base,
                        mask_row_stride,
                        q_extend_offs,
                        USE_CUSTOM_MASK,
                        SKIP_PREFIX_CUSTOM_MASK,
                        ENABLE_PREFIX_UNMASKED,
                        BLOCK_M,
                        BLOCK_N,
                        BLOCK_DMODEL,
                        ACTUAL_BLOCK_DMODEL,
                        NUM_STAGES,
                        kt_async_layout,
                        v_async_layout,
                        kt_dot_layout,
                        p_dot_layout,
                        v_dot_layout,
                        mma_layout,
                        mma_offs_n_col,
                    )
            elif pfx_seq_len > 0:
                acc, l_i, m_i = attn_fwd_inner_prefix_short(
                    acc,
                    l_i,
                    m_i,
                    q_dot,
                    K_Buffer,
                    V_Buffer,
                    kv_indices,
                    pfx_kv_start,
                    cur_kv_head,
                    pfx_seq_len,
                    stride_buf_kbs,
                    stride_buf_kh,
                    stride_buf_vbs,
                    stride_buf_vh,
                    kt_smem,
                    v_smem,
                    qk_scale,
                    LOGIT_CAP,
                    xai_temperature_reg,
                    XAI_TEMPERATURE_LEN,
                    pfx_q_abs_pos,
                    SLIDING_WINDOW_SIZE,
                    Mask,
                    pfx_mask_base,
                    mask_row_stride,
                    q_extend_offs,
                    USE_CUSTOM_MASK,
                    SKIP_PREFIX_CUSTOM_MASK,
                    ENABLE_PREFIX_UNMASKED,
                    BLOCK_M,
                    BLOCK_N,
                    BLOCK_DMODEL,
                    ACTUAL_BLOCK_DMODEL,
                    kt_async_layout,
                    v_async_layout,
                    kt_dot_layout,
                    p_dot_layout,
                    v_dot_layout,
                    mma_layout,
                    mma_offs_n_col,
                )

            # EXTEND: per-CTA dispatch (v2 core change)
            if IS_CAUSAL:
                causal_kv_end = (cur_block_m + 1) * BLOCK_M
                effective_end = tl.minimum(seq_len_extend, causal_kv_end)
            else:
                effective_end = seq_len_extend
            n_extend_blocks = (effective_end + BLOCK_N - 1) // BLOCK_N
            if (not ENABLE_MASK_SPLIT) or USE_CUSTOM_MASK or SLIDING_WINDOW_SIZE > 0:
                n_full_blocks = 0
            else:
                partial_block = ((effective_end % BLOCK_N) != 0).to(tl.int32)
                if IS_CAUSAL:
                    masked_blocks = ((BLOCK_M + BLOCK_N - 1) // BLOCK_N) + partial_block
                else:
                    masked_blocks = partial_block
                masked_blocks = tl.minimum(masked_blocks, n_extend_blocks)
                n_full_blocks = n_extend_blocks - masked_blocks

            k_extend_base = (
                K_Extend + cur_seq_q_start_idx * stride_kbs + cur_kv_head * stride_kh
            )
            v_extend_base = (
                V_Extend + cur_seq_q_start_idx * stride_vbs + cur_kv_head * stride_vh
            )

            if n_full_blocks >= NUM_STAGES:
                acc, l_i, m_i = attn_fwd_inner_extend_pipelined(
                    acc,
                    l_i,
                    m_i,
                    q_dot,
                    k_extend_base,
                    v_extend_base,
                    cur_block_m,
                    seq_len_extend,
                    stride_kbs,
                    stride_vbs,
                    0,
                    n_full_blocks,
                    kt_smem,
                    v_smem,
                    qk_scale,
                    LOGIT_CAP,
                    xai_temperature_reg,
                    XAI_TEMPERATURE_LEN,
                    SLIDING_WINDOW_SIZE,
                    IS_CAUSAL,
                    Mask,
                    mask_base_idx,
                    mask_row_stride,
                    mask_kv_col_offset,
                    USE_CUSTOM_MASK,
                    False,
                    BLOCK_M,
                    BLOCK_N,
                    BLOCK_DMODEL,
                    ACTUAL_BLOCK_DMODEL,
                    NUM_STAGES,
                    kt_async_layout,
                    v_async_layout,
                    kt_dot_layout,
                    p_dot_layout,
                    v_dot_layout,
                    mma_layout,
                    mma_offs_n_col,
                    mma_offs_m_row,
                )
            elif n_full_blocks > 0:
                acc, l_i, m_i = attn_fwd_inner_extend_short(
                    acc,
                    l_i,
                    m_i,
                    q_dot,
                    k_extend_base,
                    v_extend_base,
                    cur_block_m,
                    seq_len_extend,
                    stride_kbs,
                    stride_vbs,
                    0,
                    n_full_blocks,
                    kt_smem,
                    v_smem,
                    qk_scale,
                    LOGIT_CAP,
                    xai_temperature_reg,
                    XAI_TEMPERATURE_LEN,
                    SLIDING_WINDOW_SIZE,
                    IS_CAUSAL,
                    Mask,
                    mask_base_idx,
                    mask_row_stride,
                    mask_kv_col_offset,
                    USE_CUSTOM_MASK,
                    False,
                    BLOCK_M,
                    BLOCK_N,
                    BLOCK_DMODEL,
                    ACTUAL_BLOCK_DMODEL,
                    kt_async_layout,
                    v_async_layout,
                    kt_dot_layout,
                    p_dot_layout,
                    v_dot_layout,
                    mma_layout,
                    mma_offs_n_col,
                    mma_offs_m_row,
                )
            masked_start = n_full_blocks
            remaining_blocks = n_extend_blocks - masked_start
            if remaining_blocks >= NUM_STAGES:
                acc, l_i, m_i = attn_fwd_inner_extend_pipelined(
                    acc,
                    l_i,
                    m_i,
                    q_dot,
                    k_extend_base,
                    v_extend_base,
                    cur_block_m,
                    seq_len_extend,
                    stride_kbs,
                    stride_vbs,
                    masked_start,
                    n_extend_blocks,
                    kt_smem,
                    v_smem,
                    qk_scale,
                    LOGIT_CAP,
                    xai_temperature_reg,
                    XAI_TEMPERATURE_LEN,
                    SLIDING_WINDOW_SIZE,
                    IS_CAUSAL,
                    Mask,
                    mask_base_idx,
                    mask_row_stride,
                    mask_kv_col_offset,
                    USE_CUSTOM_MASK,
                    True,
                    BLOCK_M,
                    BLOCK_N,
                    BLOCK_DMODEL,
                    ACTUAL_BLOCK_DMODEL,
                    NUM_STAGES,
                    kt_async_layout,
                    v_async_layout,
                    kt_dot_layout,
                    p_dot_layout,
                    v_dot_layout,
                    mma_layout,
                    mma_offs_n_col,
                    mma_offs_m_row,
                )
            elif remaining_blocks > 0:
                acc, l_i, m_i = attn_fwd_inner_extend_short(
                    acc,
                    l_i,
                    m_i,
                    q_dot,
                    k_extend_base,
                    v_extend_base,
                    cur_block_m,
                    seq_len_extend,
                    stride_kbs,
                    stride_vbs,
                    masked_start,
                    n_extend_blocks,
                    kt_smem,
                    v_smem,
                    qk_scale,
                    LOGIT_CAP,
                    xai_temperature_reg,
                    XAI_TEMPERATURE_LEN,
                    SLIDING_WINDOW_SIZE,
                    IS_CAUSAL,
                    Mask,
                    mask_base_idx,
                    mask_row_stride,
                    mask_kv_col_offset,
                    USE_CUSTOM_MASK,
                    True,
                    BLOCK_M,
                    BLOCK_N,
                    BLOCK_DMODEL,
                    ACTUAL_BLOCK_DMODEL,
                    kt_async_layout,
                    v_async_layout,
                    kt_dot_layout,
                    p_dot_layout,
                    v_dot_layout,
                    mma_layout,
                    mma_offs_n_col,
                    mma_offs_m_row,
                )

        else:
            # 8-warp BLOCK_DMODEL < 128
            kt_offset_bases: gl.constexpr = [
                [1, 0],
                [2, 0],
                [4, 0],
                [8, 0],
                [16, 0],
                [32, 0],
                [0, 16],
                [0, 32],
                [0, 1],
                [0, 2],
                [0, 4],
                [0, 8],
            ]
            v_offset_bases: gl.constexpr = [
                [0, 1],
                [0, 2],
                [0, 4],
                [0, 8],
                [0, 16],
                [0, 32],
                [16, 0],
                [32, 0],
                [1, 0],
                [2, 0],
                [4, 0],
                [8, 0],
            ]
            kt_async_layout: gl.constexpr = DistributedLinearLayout(
                reg_bases=[[1, 0], [2, 0], [4, 0]],
                lane_bases=[[8, 0], [16, 0], [32, 0], [0, 16], [0, 32], [0, 1]],
                warp_bases=[[0, 2], [0, 4], [0, 8]],
                block_bases=[],
                shape=[BLOCK_DMODEL, BLOCK_N],
            )
            v_async_layout: gl.constexpr = DistributedLinearLayout(
                reg_bases=[[0, 1], [0, 2], [0, 4]],
                lane_bases=[[0, 8], [0, 16], [0, 32], [16, 0], [32, 0], [1, 0]],
                warp_bases=[[2, 0], [4, 0], [8, 0]],
                block_bases=[],
                shape=[BLOCK_N, BLOCK_DMODEL],
            )

            kt_async_smem_layout: gl.constexpr = PaddedSharedLayout(
                interval_padding_pairs=[[512, ASYNC_PAD_K]],
                offset_bases=kt_offset_bases,
                cga_layout=[],
                shape=[BLOCK_DMODEL, BLOCK_N],
            )
            v_async_smem_layout: gl.constexpr = PaddedSharedLayout(
                interval_padding_pairs=[[512, ASYNC_PAD_V]],
                offset_bases=v_offset_bases,
                cga_layout=[],
                shape=[BLOCK_N, BLOCK_DMODEL],
            )

            kt_smem = gl.allocate_shared_memory(
                Q_Extend.dtype.element_ty,
                [NUM_STAGES, BLOCK_DMODEL, BLOCK_N],
                layout=kt_async_smem_layout,
            )
            v_smem = gl.allocate_shared_memory(
                Q_Extend.dtype.element_ty,
                [NUM_STAGES, BLOCK_N, BLOCK_DMODEL],
                layout=v_async_smem_layout,
            )

            for _s in gl.static_range(NUM_STAGES):
                v_zero = gl.zeros(
                    [BLOCK_N, BLOCK_DMODEL],
                    dtype=Q_Extend.dtype.element_ty,
                    layout=v_async_layout,
                )
                v_smem.index(_s).store(v_zero)
            gl.barrier()

            q_dot = gl.convert_layout(q, q_dot_layout)

            # prefix
            n_prefix_blocks = (pfx_seq_len + BLOCK_N - 1) // BLOCK_N
            if n_prefix_blocks >= NUM_STAGES:
                acc, l_i, m_i = attn_fwd_inner_prefix_pipelined(
                    acc,
                    l_i,
                    m_i,
                    q_dot,
                    K_Buffer,
                    V_Buffer,
                    kv_indices,
                    pfx_kv_start,
                    cur_kv_head,
                    pfx_seq_len,
                    stride_buf_kbs,
                    stride_buf_kh,
                    stride_buf_vbs,
                    stride_buf_vh,
                    kt_smem,
                    v_smem,
                    qk_scale,
                    LOGIT_CAP,
                    xai_temperature_reg,
                    XAI_TEMPERATURE_LEN,
                    pfx_q_abs_pos,
                    SLIDING_WINDOW_SIZE,
                    Mask,
                    pfx_mask_base,
                    mask_row_stride,
                    q_extend_offs,
                    USE_CUSTOM_MASK,
                    SKIP_PREFIX_CUSTOM_MASK,
                    ENABLE_PREFIX_UNMASKED,
                    BLOCK_M,
                    BLOCK_N,
                    BLOCK_DMODEL,
                    ACTUAL_BLOCK_DMODEL,
                    NUM_STAGES,
                    kt_async_layout,
                    v_async_layout,
                    kt_dot_layout,
                    p_dot_layout,
                    v_dot_layout,
                    mma_layout,
                    mma_offs_n_col,
                )
            elif pfx_seq_len > 0:
                acc, l_i, m_i = attn_fwd_inner_prefix_short(
                    acc,
                    l_i,
                    m_i,
                    q_dot,
                    K_Buffer,
                    V_Buffer,
                    kv_indices,
                    pfx_kv_start,
                    cur_kv_head,
                    pfx_seq_len,
                    stride_buf_kbs,
                    stride_buf_kh,
                    stride_buf_vbs,
                    stride_buf_vh,
                    kt_smem,
                    v_smem,
                    qk_scale,
                    LOGIT_CAP,
                    xai_temperature_reg,
                    XAI_TEMPERATURE_LEN,
                    pfx_q_abs_pos,
                    SLIDING_WINDOW_SIZE,
                    Mask,
                    pfx_mask_base,
                    mask_row_stride,
                    q_extend_offs,
                    USE_CUSTOM_MASK,
                    SKIP_PREFIX_CUSTOM_MASK,
                    ENABLE_PREFIX_UNMASKED,
                    BLOCK_M,
                    BLOCK_N,
                    BLOCK_DMODEL,
                    ACTUAL_BLOCK_DMODEL,
                    kt_async_layout,
                    v_async_layout,
                    kt_dot_layout,
                    p_dot_layout,
                    v_dot_layout,
                    mma_layout,
                    mma_offs_n_col,
                )

            # EXTEND: per-CTA dispatch (v2 core change)
            if IS_CAUSAL:
                causal_kv_end = (cur_block_m + 1) * BLOCK_M
                effective_end = tl.minimum(seq_len_extend, causal_kv_end)
            else:
                effective_end = seq_len_extend
            n_extend_blocks = (effective_end + BLOCK_N - 1) // BLOCK_N
            if (not ENABLE_MASK_SPLIT) or USE_CUSTOM_MASK or SLIDING_WINDOW_SIZE > 0:
                n_full_blocks = 0
            else:
                partial_block = ((effective_end % BLOCK_N) != 0).to(tl.int32)
                if IS_CAUSAL:
                    masked_blocks = ((BLOCK_M + BLOCK_N - 1) // BLOCK_N) + partial_block
                else:
                    masked_blocks = partial_block
                masked_blocks = tl.minimum(masked_blocks, n_extend_blocks)
                n_full_blocks = n_extend_blocks - masked_blocks

            k_extend_base = (
                K_Extend + cur_seq_q_start_idx * stride_kbs + cur_kv_head * stride_kh
            )
            v_extend_base = (
                V_Extend + cur_seq_q_start_idx * stride_vbs + cur_kv_head * stride_vh
            )

            if n_full_blocks >= NUM_STAGES:
                acc, l_i, m_i = attn_fwd_inner_extend_pipelined(
                    acc,
                    l_i,
                    m_i,
                    q_dot,
                    k_extend_base,
                    v_extend_base,
                    cur_block_m,
                    seq_len_extend,
                    stride_kbs,
                    stride_vbs,
                    0,
                    n_full_blocks,
                    kt_smem,
                    v_smem,
                    qk_scale,
                    LOGIT_CAP,
                    xai_temperature_reg,
                    XAI_TEMPERATURE_LEN,
                    SLIDING_WINDOW_SIZE,
                    IS_CAUSAL,
                    Mask,
                    mask_base_idx,
                    mask_row_stride,
                    mask_kv_col_offset,
                    USE_CUSTOM_MASK,
                    False,
                    BLOCK_M,
                    BLOCK_N,
                    BLOCK_DMODEL,
                    ACTUAL_BLOCK_DMODEL,
                    NUM_STAGES,
                    kt_async_layout,
                    v_async_layout,
                    kt_dot_layout,
                    p_dot_layout,
                    v_dot_layout,
                    mma_layout,
                    mma_offs_n_col,
                    mma_offs_m_row,
                )
            elif n_full_blocks > 0:
                acc, l_i, m_i = attn_fwd_inner_extend_short(
                    acc,
                    l_i,
                    m_i,
                    q_dot,
                    k_extend_base,
                    v_extend_base,
                    cur_block_m,
                    seq_len_extend,
                    stride_kbs,
                    stride_vbs,
                    0,
                    n_full_blocks,
                    kt_smem,
                    v_smem,
                    qk_scale,
                    LOGIT_CAP,
                    xai_temperature_reg,
                    XAI_TEMPERATURE_LEN,
                    SLIDING_WINDOW_SIZE,
                    IS_CAUSAL,
                    Mask,
                    mask_base_idx,
                    mask_row_stride,
                    mask_kv_col_offset,
                    USE_CUSTOM_MASK,
                    False,
                    BLOCK_M,
                    BLOCK_N,
                    BLOCK_DMODEL,
                    ACTUAL_BLOCK_DMODEL,
                    kt_async_layout,
                    v_async_layout,
                    kt_dot_layout,
                    p_dot_layout,
                    v_dot_layout,
                    mma_layout,
                    mma_offs_n_col,
                    mma_offs_m_row,
                )
            masked_start = n_full_blocks
            remaining_blocks = n_extend_blocks - masked_start
            if remaining_blocks >= NUM_STAGES:
                acc, l_i, m_i = attn_fwd_inner_extend_pipelined(
                    acc,
                    l_i,
                    m_i,
                    q_dot,
                    k_extend_base,
                    v_extend_base,
                    cur_block_m,
                    seq_len_extend,
                    stride_kbs,
                    stride_vbs,
                    masked_start,
                    n_extend_blocks,
                    kt_smem,
                    v_smem,
                    qk_scale,
                    LOGIT_CAP,
                    xai_temperature_reg,
                    XAI_TEMPERATURE_LEN,
                    SLIDING_WINDOW_SIZE,
                    IS_CAUSAL,
                    Mask,
                    mask_base_idx,
                    mask_row_stride,
                    mask_kv_col_offset,
                    USE_CUSTOM_MASK,
                    True,
                    BLOCK_M,
                    BLOCK_N,
                    BLOCK_DMODEL,
                    ACTUAL_BLOCK_DMODEL,
                    NUM_STAGES,
                    kt_async_layout,
                    v_async_layout,
                    kt_dot_layout,
                    p_dot_layout,
                    v_dot_layout,
                    mma_layout,
                    mma_offs_n_col,
                    mma_offs_m_row,
                )
            elif remaining_blocks > 0:
                acc, l_i, m_i = attn_fwd_inner_extend_short(
                    acc,
                    l_i,
                    m_i,
                    q_dot,
                    k_extend_base,
                    v_extend_base,
                    cur_block_m,
                    seq_len_extend,
                    stride_kbs,
                    stride_vbs,
                    masked_start,
                    n_extend_blocks,
                    kt_smem,
                    v_smem,
                    qk_scale,
                    LOGIT_CAP,
                    xai_temperature_reg,
                    XAI_TEMPERATURE_LEN,
                    SLIDING_WINDOW_SIZE,
                    IS_CAUSAL,
                    Mask,
                    mask_base_idx,
                    mask_row_stride,
                    mask_kv_col_offset,
                    USE_CUSTOM_MASK,
                    True,
                    BLOCK_M,
                    BLOCK_N,
                    BLOCK_DMODEL,
                    ACTUAL_BLOCK_DMODEL,
                    kt_async_layout,
                    v_async_layout,
                    kt_dot_layout,
                    p_dot_layout,
                    v_dot_layout,
                    mma_layout,
                    mma_offs_n_col,
                    mma_offs_m_row,
                )

    # sinks
    if HAS_SINK:
        cur_sink = gl.load(Sinks + cur_head)
        l_i = l_i + gl.exp2(cur_sink * LOG2E - m_i)

    # normalize and store
    l_recip = 1.0 / l_i
    acc = acc * l_recip[:, None]
    if V_SCALE != 1.0:
        acc = acc * V_SCALE

    o_ptrs = (
        O_Extend
        + (cur_seq_q_start_idx + cur_block_m * BLOCK_M + offs_m[:, None]) * stride_obs
        + cur_head * stride_oh
        + offs_d[None, :]
    )
    o_mask = (cur_block_m * BLOCK_M + offs_m[:, None]) < seq_len_extend
    if ACTUAL_BLOCK_DMODEL != BLOCK_DMODEL:
        o_mask = o_mask & (offs_d[None, :] < ACTUAL_BLOCK_DMODEL)
    out = gl.convert_layout(acc, blocked_layout).to(O_Extend.dtype.element_ty)
    gl.store(o_ptrs, out, mask=o_mask)


# ===-----------------------------------------------------------------------===#
# Persistent-CTA Kernel (work-centric scheduling)
# ===-----------------------------------------------------------------------===#


@gluon.jit
def gluon_extend_attn_fwd_persistent(
    Q_Extend,
    K_Extend,
    V_Extend,
    O_Extend,  #
    K_Buffer,
    V_Buffer,  #
    qo_indptr,
    kv_indptr,
    kv_indices,  #
    Mask,
    MaskIndptr,
    WindowKvOffsets,  #
    SM_SCALE: gl.constexpr,
    kv_group_num,  #
    stride_qbs,
    stride_qh,  #
    stride_kbs,
    stride_kh,  #
    stride_vbs,
    stride_vh,  #
    stride_obs,
    stride_oh,  #
    stride_buf_kbs,
    stride_buf_kh,  #
    stride_buf_vbs,
    stride_buf_vh,  #
    tile_seq_info,  #  [total_valid_tiles] int32 -- batch index per tile
    tile_head_info,  # [total_valid_tiles] int32 -- head index per tile
    tile_m_info,  #    [total_valid_tiles] int32 -- m_tile index per tile
    total_valid_tiles,  # int32 scalar
    total_programs,  #    int32 scalar (= grid dim 0)
    IS_CAUSAL: gl.constexpr,  #
    USE_CUSTOM_MASK: gl.constexpr,
    SKIP_PREFIX_CUSTOM_MASK: gl.constexpr,  #
    ENABLE_PREFIX_UNMASKED: gl.constexpr,
    ENABLE_MASK_SPLIT: gl.constexpr,  #
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,  #
    BLOCK_DMODEL: gl.constexpr,
    ACTUAL_BLOCK_DMODEL: gl.constexpr,  #
    NUM_STAGES: gl.constexpr,  #
    MMA_INSTR_M: gl.constexpr,
    MMA_INSTR_N: gl.constexpr,
    MMA_INSTR_K: gl.constexpr,  #
    QK_K_WIDTH: gl.constexpr,
    PV_K_WIDTH: gl.constexpr,  #
    ASYNC_PAD_K: gl.constexpr,
    ASYNC_PAD_V: gl.constexpr,  #
    Sinks,
    HAS_SINK: gl.constexpr,  #
    LOGIT_CAP: gl.constexpr,  #
    XAI_TEMPERATURE_LEN: gl.constexpr,  #
    SLIDING_WINDOW_SIZE: gl.constexpr,  #
    V_SCALE: gl.constexpr,  #
):
    num_warps: gl.constexpr = gl.num_warps()
    cta_id = gl.program_id(0)

    mma_layout: gl.constexpr = AMDMFMALayout(
        version=4,
        instr_shape=[MMA_INSTR_M, MMA_INSTR_N, MMA_INSTR_K],
        transposed=True,
        warps_per_cta=[num_warps, 1],
    )
    k_width: gl.constexpr = QK_K_WIDTH
    threads_per_warp: gl.constexpr = 64
    pv_k_width: gl.constexpr = PV_K_WIDTH

    q_dot_layout: gl.constexpr = DotOperandLayout(
        operand_index=0, parent=mma_layout, k_width=k_width
    )
    kt_dot_layout: gl.constexpr = DotOperandLayout(
        operand_index=1, parent=mma_layout, k_width=k_width
    )
    p_dot_layout: gl.constexpr = DotOperandLayout(
        operand_index=0, parent=mma_layout, k_width=pv_k_width
    )
    v_dot_layout: gl.constexpr = DotOperandLayout(
        operand_index=1, parent=mma_layout, k_width=pv_k_width
    )

    blocked_layout: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 8],
        threads_per_warp=[threads_per_warp // 4, 4],
        warps_per_cta=[num_warps, 1],
        order=[1, 0],
    )

    offs_m_layout: gl.constexpr = gl.SliceLayout(dim=1, parent=blocked_layout)
    offs_d_layout: gl.constexpr = gl.SliceLayout(dim=0, parent=blocked_layout)
    mma_offs_n_col: gl.constexpr = gl.SliceLayout(dim=0, parent=mma_layout)
    mma_offs_m_row: gl.constexpr = gl.SliceLayout(dim=1, parent=mma_layout)
    mma_m_layout: gl.constexpr = gl.SliceLayout(dim=1, parent=mma_layout)

    offs_m = gl.arange(0, BLOCK_M, layout=offs_m_layout)
    offs_d = gl.arange(0, BLOCK_DMODEL, layout=offs_d_layout)

    USE_SERIAL: gl.constexpr = num_warps < 8
    qk_scale: gl.constexpr = SM_SCALE * LOG2E

    # === Persistent tile loop ===
    tile_idx = cta_id
    while tile_idx < total_valid_tiles:
        cur_seq = gl.load(tile_seq_info + tile_idx).to(tl.int32)
        cur_head = gl.load(tile_head_info + tile_idx).to(tl.int32)
        cur_block_m = gl.load(tile_m_info + tile_idx).to(tl.int32)
        cur_kv_head = cur_head // kv_group_num

        cur_seq_q_start_idx = gl.load(qo_indptr + cur_seq)
        seq_len_extend = gl.load(qo_indptr + cur_seq + 1) - cur_seq_q_start_idx
        cur_seq_kv_start_idx = gl.load(kv_indptr + cur_seq)
        seq_len_prefix = gl.load(kv_indptr + cur_seq + 1) - cur_seq_kv_start_idx

        if USE_CUSTOM_MASK:
            mask_base_idx = gl.load(MaskIndptr + cur_seq).to(tl.int64)
            window_kv_offset = 0
            if SLIDING_WINDOW_SIZE > 0:
                window_kv_offset = gl.load(WindowKvOffsets + cur_seq)
            cur_seq_len = seq_len_prefix + seq_len_extend
            mask_row_stride = (cur_seq_len + window_kv_offset).to(tl.int64)
            mask_base_idx = mask_base_idx + window_kv_offset.to(tl.int64)
            mask_kv_col_offset = (seq_len_prefix).to(tl.int64)
        else:
            mask_base_idx = tl.cast(0, tl.int64)
            mask_row_stride = tl.cast(0, tl.int64)
            mask_kv_col_offset = tl.cast(0, tl.int64)

        q_ptrs = (
            Q_Extend
            + (cur_seq_q_start_idx + cur_block_m * BLOCK_M + offs_m[:, None])
            * stride_qbs
            + cur_head * stride_qh
            + offs_d[None, :]
        )
        q_mask = (cur_block_m * BLOCK_M + offs_m[:, None]) < seq_len_extend
        if ACTUAL_BLOCK_DMODEL != BLOCK_DMODEL:
            q_mask = q_mask & (offs_d[None, :] < ACTUAL_BLOCK_DMODEL)
        q = gl.load(q_ptrs, mask=q_mask, other=0.0)

        m_i = gl.full([BLOCK_M], float("-inf"), dtype=gl.float32, layout=mma_m_layout)
        l_i = gl.full([BLOCK_M], 1.0, dtype=gl.float32, layout=mma_m_layout)
        acc = gl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=gl.float32, layout=mma_layout)

        q_abs_pos = (
            seq_len_prefix
            + cur_block_m * BLOCK_M
            + gl.arange(0, BLOCK_M, layout=mma_offs_m_row)
        )
        q_extend_raw = cur_block_m * BLOCK_M + gl.arange(
            0, BLOCK_M, layout=mma_offs_m_row
        )
        if USE_CUSTOM_MASK:
            q_extend_offs = tl.minimum(q_extend_raw, tl.maximum(seq_len_extend - 1, 0))
        else:
            q_extend_offs = q_extend_raw

        if XAI_TEMPERATURE_LEN > 0:
            inv_log2_len = 1.0 / tl.log2(float(XAI_TEMPERATURE_LEN))
            xai_temperature_reg = gl.where(
                q_abs_pos > XAI_TEMPERATURE_LEN,
                tl.log2(q_abs_pos.to(gl.float32)) * inv_log2_len,
                gl.full([BLOCK_M], 1.0, dtype=gl.float32, layout=mma_offs_m_row),
            )
        else:
            xai_temperature_reg = gl.full(
                [BLOCK_M], 1.0, dtype=gl.float32, layout=mma_offs_m_row
            )

        pfx_kv_start = cur_seq_kv_start_idx
        pfx_seq_len = seq_len_prefix
        pfx_q_abs_pos = q_abs_pos
        pfx_mask_base = mask_base_idx
        if SLIDING_WINDOW_SIZE > 0:
            q_min_abs = seq_len_prefix + cur_block_m * BLOCK_M
            first_useful_pos = tl.maximum(q_min_abs - SLIDING_WINDOW_SIZE, 0)
            prefix_skip_n = (first_useful_pos // BLOCK_N) * BLOCK_N
            pfx_kv_start = cur_seq_kv_start_idx + prefix_skip_n
            pfx_seq_len = seq_len_prefix - prefix_skip_n
            pfx_q_abs_pos = q_abs_pos - prefix_skip_n
            if USE_CUSTOM_MASK:
                pfx_mask_base = mask_base_idx + prefix_skip_n.to(tl.int64)

        if USE_SERIAL:
            if NUM_STAGES >= 2 and BLOCK_DMODEL >= 128:
                kt_offset_bases: gl.constexpr = [
                    [1, 0],
                    [2, 0],
                    [4, 0],
                    [8, 0],
                    [16, 0],
                    [32, 0],
                    [64, 0],
                    [0, 16],
                    [0, 32],
                    [0, 1],
                    [0, 2],
                    [0, 4],
                    [0, 8],
                ]
                v_offset_bases: gl.constexpr = [
                    [0, 1],
                    [0, 2],
                    [0, 4],
                    [0, 8],
                    [0, 16],
                    [0, 32],
                    [0, 64],
                    [16, 0],
                    [32, 0],
                    [1, 0],
                    [2, 0],
                    [4, 0],
                    [8, 0],
                ]
                kt_async_layout: gl.constexpr = DistributedLinearLayout(
                    reg_bases=[[1, 0], [2, 0], [4, 0], [0, 4], [0, 8]],
                    lane_bases=[[8, 0], [16, 0], [32, 0], [64, 0], [0, 16], [0, 32]],
                    warp_bases=[[0, 1], [0, 2]],
                    block_bases=[],
                    shape=[BLOCK_DMODEL, BLOCK_N],
                )
                v_async_layout: gl.constexpr = DistributedLinearLayout(
                    reg_bases=[[0, 1], [0, 2], [0, 4], [4, 0], [8, 0]],
                    lane_bases=[[0, 8], [0, 16], [0, 32], [0, 64], [16, 0], [32, 0]],
                    warp_bases=[[1, 0], [2, 0]],
                    block_bases=[],
                    shape=[BLOCK_N, BLOCK_DMODEL],
                )

                kt_smem_layout: gl.constexpr = PaddedSharedLayout(
                    interval_padding_pairs=[[512, ASYNC_PAD_K]],
                    offset_bases=kt_offset_bases,
                    cga_layout=[],
                    shape=[BLOCK_DMODEL, BLOCK_N],
                )
                v_smem_layout: gl.constexpr = PaddedSharedLayout(
                    interval_padding_pairs=[[512, ASYNC_PAD_V]],
                    offset_bases=v_offset_bases,
                    cga_layout=[],
                    shape=[BLOCK_N, BLOCK_DMODEL],
                )

                kt_smem = gl.allocate_shared_memory(
                    Q_Extend.dtype.element_ty,
                    [NUM_STAGES, BLOCK_DMODEL, BLOCK_N],
                    layout=kt_smem_layout,
                )
                v_smem = gl.allocate_shared_memory(
                    Q_Extend.dtype.element_ty,
                    [NUM_STAGES, BLOCK_N, BLOCK_DMODEL],
                    layout=v_smem_layout,
                )

                for _s in gl.static_range(NUM_STAGES):
                    v_zero = gl.zeros(
                        [BLOCK_N, BLOCK_DMODEL],
                        dtype=Q_Extend.dtype.element_ty,
                        layout=v_async_layout,
                    )
                    v_smem.index(_s).store(v_zero)
                gl.barrier()

                q_dot = gl.convert_layout(q, q_dot_layout)

                if pfx_seq_len > 0:
                    n_prefix_blocks = (pfx_seq_len + BLOCK_N - 1) // BLOCK_N
                    n_extend_est = (seq_len_extend + BLOCK_N - 1) // BLOCK_N
                    use_pipe_prefix = n_prefix_blocks >= NUM_STAGES
                    if LOGIT_CAP > 0:
                        use_pipe_prefix = use_pipe_prefix and (
                            n_extend_est < NUM_STAGES
                        )
                    if use_pipe_prefix:
                        acc, l_i, m_i = attn_fwd_inner_prefix_dma_simple(
                            acc,
                            l_i,
                            m_i,
                            q_dot,
                            K_Buffer,
                            V_Buffer,
                            kv_indices,
                            pfx_kv_start,
                            cur_kv_head,
                            pfx_seq_len,
                            stride_buf_kbs,
                            stride_buf_kh,
                            stride_buf_vbs,
                            stride_buf_vh,
                            kt_smem,
                            v_smem,
                            qk_scale,
                            LOGIT_CAP,
                            xai_temperature_reg,
                            XAI_TEMPERATURE_LEN,
                            pfx_q_abs_pos,
                            SLIDING_WINDOW_SIZE,
                            Mask,
                            pfx_mask_base,
                            mask_row_stride,
                            q_extend_offs,
                            USE_CUSTOM_MASK,
                            SKIP_PREFIX_CUSTOM_MASK,
                            ENABLE_PREFIX_UNMASKED,
                            BLOCK_M,
                            BLOCK_N,
                            BLOCK_DMODEL,
                            ACTUAL_BLOCK_DMODEL,
                            NUM_STAGES,
                            kt_async_layout,
                            v_async_layout,
                            kt_dot_layout,
                            p_dot_layout,
                            v_dot_layout,
                            mma_layout,
                            mma_offs_n_col,
                        )
                    else:
                        acc, l_i, m_i = attn_fwd_inner_prefix_short(
                            acc,
                            l_i,
                            m_i,
                            q_dot,
                            K_Buffer,
                            V_Buffer,
                            kv_indices,
                            pfx_kv_start,
                            cur_kv_head,
                            pfx_seq_len,
                            stride_buf_kbs,
                            stride_buf_kh,
                            stride_buf_vbs,
                            stride_buf_vh,
                            kt_smem,
                            v_smem,
                            qk_scale,
                            LOGIT_CAP,
                            xai_temperature_reg,
                            XAI_TEMPERATURE_LEN,
                            pfx_q_abs_pos,
                            SLIDING_WINDOW_SIZE,
                            Mask,
                            pfx_mask_base,
                            mask_row_stride,
                            q_extend_offs,
                            USE_CUSTOM_MASK,
                            SKIP_PREFIX_CUSTOM_MASK,
                            ENABLE_PREFIX_UNMASKED,
                            BLOCK_M,
                            BLOCK_N,
                            BLOCK_DMODEL,
                            ACTUAL_BLOCK_DMODEL,
                            kt_async_layout,
                            v_async_layout,
                            kt_dot_layout,
                            p_dot_layout,
                            v_dot_layout,
                            mma_layout,
                            mma_offs_n_col,
                        )

                cdna4_async.wait_group(0)

                if IS_CAUSAL:
                    causal_kv_end = (cur_block_m + 1) * BLOCK_M
                    effective_end = tl.minimum(seq_len_extend, causal_kv_end)
                else:
                    effective_end = seq_len_extend
                n_extend_blocks = (effective_end + BLOCK_N - 1) // BLOCK_N
                if (
                    (not ENABLE_MASK_SPLIT)
                    or USE_CUSTOM_MASK
                    or SLIDING_WINDOW_SIZE > 0
                ):
                    n_full_blocks = 0
                else:
                    partial_block = ((effective_end % BLOCK_N) != 0).to(tl.int32)
                    if IS_CAUSAL:
                        masked_blocks = (
                            (BLOCK_M + BLOCK_N - 1) // BLOCK_N
                        ) + partial_block
                    else:
                        masked_blocks = partial_block
                    masked_blocks = tl.minimum(masked_blocks, n_extend_blocks)
                    n_full_blocks = n_extend_blocks - masked_blocks

                k_extend_base = (
                    K_Extend
                    + cur_seq_q_start_idx * stride_kbs
                    + cur_kv_head * stride_kh
                )
                v_extend_base = (
                    V_Extend
                    + cur_seq_q_start_idx * stride_vbs
                    + cur_kv_head * stride_vh
                )

                if n_full_blocks >= NUM_STAGES:
                    acc, l_i, m_i = attn_fwd_inner_extend_dma(
                        acc,
                        l_i,
                        m_i,
                        q_dot,
                        k_extend_base,
                        v_extend_base,
                        cur_block_m,
                        seq_len_extend,
                        stride_kbs,
                        stride_vbs,
                        0,
                        n_full_blocks,
                        kt_smem,
                        v_smem,
                        qk_scale,
                        LOGIT_CAP,
                        xai_temperature_reg,
                        XAI_TEMPERATURE_LEN,
                        SLIDING_WINDOW_SIZE,
                        IS_CAUSAL,
                        Mask,
                        mask_base_idx,
                        mask_row_stride,
                        mask_kv_col_offset,
                        USE_CUSTOM_MASK,
                        False,
                        BLOCK_M,
                        BLOCK_N,
                        BLOCK_DMODEL,
                        ACTUAL_BLOCK_DMODEL,
                        NUM_STAGES,
                        kt_async_layout,
                        v_async_layout,
                        kt_dot_layout,
                        p_dot_layout,
                        v_dot_layout,
                        mma_layout,
                        mma_offs_n_col,
                        mma_offs_m_row,
                    )
                elif n_full_blocks > 0:
                    acc, l_i, m_i = attn_fwd_inner_extend_short(
                        acc,
                        l_i,
                        m_i,
                        q_dot,
                        k_extend_base,
                        v_extend_base,
                        cur_block_m,
                        seq_len_extend,
                        stride_kbs,
                        stride_vbs,
                        0,
                        n_full_blocks,
                        kt_smem,
                        v_smem,
                        qk_scale,
                        LOGIT_CAP,
                        xai_temperature_reg,
                        XAI_TEMPERATURE_LEN,
                        SLIDING_WINDOW_SIZE,
                        IS_CAUSAL,
                        Mask,
                        mask_base_idx,
                        mask_row_stride,
                        mask_kv_col_offset,
                        USE_CUSTOM_MASK,
                        False,
                        BLOCK_M,
                        BLOCK_N,
                        BLOCK_DMODEL,
                        ACTUAL_BLOCK_DMODEL,
                        kt_async_layout,
                        v_async_layout,
                        kt_dot_layout,
                        p_dot_layout,
                        v_dot_layout,
                        mma_layout,
                        mma_offs_n_col,
                        mma_offs_m_row,
                    )
                masked_start = n_full_blocks
                remaining_blocks = n_extend_blocks - masked_start
                if remaining_blocks >= NUM_STAGES:
                    acc, l_i, m_i = attn_fwd_inner_extend_dma(
                        acc,
                        l_i,
                        m_i,
                        q_dot,
                        k_extend_base,
                        v_extend_base,
                        cur_block_m,
                        seq_len_extend,
                        stride_kbs,
                        stride_vbs,
                        masked_start,
                        n_extend_blocks,
                        kt_smem,
                        v_smem,
                        qk_scale,
                        LOGIT_CAP,
                        xai_temperature_reg,
                        XAI_TEMPERATURE_LEN,
                        SLIDING_WINDOW_SIZE,
                        IS_CAUSAL,
                        Mask,
                        mask_base_idx,
                        mask_row_stride,
                        mask_kv_col_offset,
                        USE_CUSTOM_MASK,
                        True,
                        BLOCK_M,
                        BLOCK_N,
                        BLOCK_DMODEL,
                        ACTUAL_BLOCK_DMODEL,
                        NUM_STAGES,
                        kt_async_layout,
                        v_async_layout,
                        kt_dot_layout,
                        p_dot_layout,
                        v_dot_layout,
                        mma_layout,
                        mma_offs_n_col,
                        mma_offs_m_row,
                    )
                elif remaining_blocks > 0:
                    acc, l_i, m_i = attn_fwd_inner_extend_short(
                        acc,
                        l_i,
                        m_i,
                        q_dot,
                        k_extend_base,
                        v_extend_base,
                        cur_block_m,
                        seq_len_extend,
                        stride_kbs,
                        stride_vbs,
                        masked_start,
                        n_extend_blocks,
                        kt_smem,
                        v_smem,
                        qk_scale,
                        LOGIT_CAP,
                        xai_temperature_reg,
                        XAI_TEMPERATURE_LEN,
                        SLIDING_WINDOW_SIZE,
                        IS_CAUSAL,
                        Mask,
                        mask_base_idx,
                        mask_row_stride,
                        mask_kv_col_offset,
                        USE_CUSTOM_MASK,
                        True,
                        BLOCK_M,
                        BLOCK_N,
                        BLOCK_DMODEL,
                        ACTUAL_BLOCK_DMODEL,
                        kt_async_layout,
                        v_async_layout,
                        kt_dot_layout,
                        p_dot_layout,
                        v_dot_layout,
                        mma_layout,
                        mma_offs_n_col,
                        mma_offs_m_row,
                    )

            else:
                kt_blocked_layout: gl.constexpr = gl.BlockedLayout(
                    size_per_thread=[1, 8],
                    threads_per_warp=[threads_per_warp // 4, 4],
                    warps_per_cta=[1, num_warps],
                    order=[0, 1],
                )
                kt_serial_smem_layout: gl.constexpr = gl.SwizzledSharedLayout(
                    vec=8,
                    per_phase=1,
                    max_phase=16,
                    order=[0, 1],
                )
                v_serial_smem_layout: gl.constexpr = gl.SwizzledSharedLayout(
                    vec=8,
                    per_phase=1,
                    max_phase=16,
                    order=[1, 0],
                )
                q_smem_layout: gl.constexpr = gl.SwizzledSharedLayout(
                    vec=8,
                    per_phase=1,
                    max_phase=16,
                    order=[1, 0],
                )

                kt_serial_smem = gl.allocate_shared_memory(
                    Q_Extend.dtype.element_ty,
                    [BLOCK_DMODEL, BLOCK_N],
                    layout=kt_serial_smem_layout,
                )
                v_serial_smem = gl.allocate_shared_memory(
                    Q_Extend.dtype.element_ty,
                    [BLOCK_N, BLOCK_DMODEL],
                    layout=v_serial_smem_layout,
                )
                q_smem = gl.allocate_shared_memory(
                    Q_Extend.dtype.element_ty,
                    [BLOCK_M, BLOCK_DMODEL],
                    layout=q_smem_layout,
                )

                q_smem.store(q)
                q_dot = q_smem.load(q_dot_layout)

                if pfx_seq_len > 0:
                    acc, l_i, m_i = attn_fwd_inner_prefix_serial(
                        acc,
                        l_i,
                        m_i,
                        q_dot,
                        K_Buffer,
                        V_Buffer,
                        kv_indices,
                        pfx_kv_start,
                        cur_kv_head,
                        pfx_seq_len,
                        stride_buf_kbs,
                        stride_buf_kh,
                        stride_buf_vbs,
                        stride_buf_vh,
                        kt_serial_smem,
                        v_serial_smem,
                        qk_scale,
                        LOGIT_CAP,
                        xai_temperature_reg,
                        XAI_TEMPERATURE_LEN,
                        pfx_q_abs_pos,
                        SLIDING_WINDOW_SIZE,
                        Mask,
                        pfx_mask_base,
                        mask_row_stride,
                        q_extend_offs,
                        USE_CUSTOM_MASK,
                        SKIP_PREFIX_CUSTOM_MASK,
                        ENABLE_PREFIX_UNMASKED,
                        BLOCK_M,
                        BLOCK_N,
                        BLOCK_DMODEL,
                        ACTUAL_BLOCK_DMODEL,
                        kt_blocked_layout,
                        blocked_layout,
                        kt_dot_layout,
                        p_dot_layout,
                        v_dot_layout,
                        mma_layout,
                        mma_offs_n_col,
                    )

                if IS_CAUSAL:
                    causal_kv_end = (cur_block_m + 1) * BLOCK_M
                    effective_end = tl.minimum(seq_len_extend, causal_kv_end)
                else:
                    effective_end = seq_len_extend
                n_extend_blocks = (effective_end + BLOCK_N - 1) // BLOCK_N
                if (
                    (not ENABLE_MASK_SPLIT)
                    or USE_CUSTOM_MASK
                    or SLIDING_WINDOW_SIZE > 0
                ):
                    n_full_blocks = 0
                else:
                    partial_block = ((effective_end % BLOCK_N) != 0).to(tl.int32)
                    if IS_CAUSAL:
                        masked_blocks = (
                            (BLOCK_M + BLOCK_N - 1) // BLOCK_N
                        ) + partial_block
                    else:
                        masked_blocks = partial_block
                    masked_blocks = tl.minimum(masked_blocks, n_extend_blocks)
                    n_full_blocks = n_extend_blocks - masked_blocks

                k_extend_base = (
                    K_Extend
                    + cur_seq_q_start_idx * stride_kbs
                    + cur_kv_head * stride_kh
                )
                v_extend_base = (
                    V_Extend
                    + cur_seq_q_start_idx * stride_vbs
                    + cur_kv_head * stride_vh
                )

                if n_full_blocks > 0:
                    acc, l_i, m_i = attn_fwd_inner_extend_serial(
                        acc,
                        l_i,
                        m_i,
                        q_dot,
                        k_extend_base,
                        v_extend_base,
                        cur_block_m,
                        seq_len_extend,
                        stride_kbs,
                        stride_vbs,
                        0,
                        n_full_blocks,
                        kt_serial_smem,
                        v_serial_smem,
                        qk_scale,
                        LOGIT_CAP,
                        xai_temperature_reg,
                        XAI_TEMPERATURE_LEN,
                        SLIDING_WINDOW_SIZE,
                        IS_CAUSAL,
                        Mask,
                        mask_base_idx,
                        mask_row_stride,
                        mask_kv_col_offset,
                        USE_CUSTOM_MASK,
                        False,
                        BLOCK_M,
                        BLOCK_N,
                        BLOCK_DMODEL,
                        ACTUAL_BLOCK_DMODEL,
                        kt_blocked_layout,
                        blocked_layout,
                        kt_dot_layout,
                        p_dot_layout,
                        v_dot_layout,
                        mma_layout,
                        mma_offs_n_col,
                        mma_offs_m_row,
                    )
                masked_start = n_full_blocks
                remaining_blocks = n_extend_blocks - masked_start
                if remaining_blocks > 0:
                    acc, l_i, m_i = attn_fwd_inner_extend_serial(
                        acc,
                        l_i,
                        m_i,
                        q_dot,
                        k_extend_base,
                        v_extend_base,
                        cur_block_m,
                        seq_len_extend,
                        stride_kbs,
                        stride_vbs,
                        masked_start,
                        n_extend_blocks,
                        kt_serial_smem,
                        v_serial_smem,
                        qk_scale,
                        LOGIT_CAP,
                        xai_temperature_reg,
                        XAI_TEMPERATURE_LEN,
                        SLIDING_WINDOW_SIZE,
                        IS_CAUSAL,
                        Mask,
                        mask_base_idx,
                        mask_row_stride,
                        mask_kv_col_offset,
                        USE_CUSTOM_MASK,
                        True,
                        BLOCK_M,
                        BLOCK_N,
                        BLOCK_DMODEL,
                        ACTUAL_BLOCK_DMODEL,
                        kt_blocked_layout,
                        blocked_layout,
                        kt_dot_layout,
                        p_dot_layout,
                        v_dot_layout,
                        mma_layout,
                        mma_offs_n_col,
                        mma_offs_m_row,
                    )

        else:
            # 8-warp path -- identical inner dispatch to basic kernel but in persistent loop
            if BLOCK_DMODEL >= 128:
                kt_offset_bases: gl.constexpr = [
                    [1, 0],
                    [2, 0],
                    [4, 0],
                    [8, 0],
                    [16, 0],
                    [32, 0],
                    [64, 0],
                    [0, 16],
                    [0, 32],
                    [0, 1],
                    [0, 2],
                    [0, 4],
                    [0, 8],
                ]
                v_offset_bases: gl.constexpr = [
                    [0, 1],
                    [0, 2],
                    [0, 4],
                    [0, 8],
                    [0, 16],
                    [0, 32],
                    [0, 64],
                    [16, 0],
                    [32, 0],
                    [1, 0],
                    [2, 0],
                    [4, 0],
                    [8, 0],
                ]
                kt_async_layout: gl.constexpr = DistributedLinearLayout(
                    reg_bases=[[1, 0], [2, 0], [4, 0], [0, 8]],
                    lane_bases=[[8, 0], [16, 0], [32, 0], [64, 0], [0, 16], [0, 32]],
                    warp_bases=[[0, 1], [0, 2], [0, 4]],
                    block_bases=[],
                    shape=[BLOCK_DMODEL, BLOCK_N],
                )
                v_async_layout: gl.constexpr = DistributedLinearLayout(
                    reg_bases=[[0, 1], [0, 2], [0, 4], [8, 0]],
                    lane_bases=[[0, 8], [0, 16], [0, 32], [0, 64], [16, 0], [32, 0]],
                    warp_bases=[[1, 0], [2, 0], [4, 0]],
                    block_bases=[],
                    shape=[BLOCK_N, BLOCK_DMODEL],
                )

                kt_async_smem_layout: gl.constexpr = PaddedSharedLayout(
                    interval_padding_pairs=[[512, ASYNC_PAD_K]],
                    offset_bases=kt_offset_bases,
                    cga_layout=[],
                    shape=[BLOCK_DMODEL, BLOCK_N],
                )
                v_async_smem_layout: gl.constexpr = PaddedSharedLayout(
                    interval_padding_pairs=[[512, ASYNC_PAD_V]],
                    offset_bases=v_offset_bases,
                    cga_layout=[],
                    shape=[BLOCK_N, BLOCK_DMODEL],
                )

                kt_smem = gl.allocate_shared_memory(
                    Q_Extend.dtype.element_ty,
                    [NUM_STAGES, BLOCK_DMODEL, BLOCK_N],
                    layout=kt_async_smem_layout,
                )
                v_smem = gl.allocate_shared_memory(
                    Q_Extend.dtype.element_ty,
                    [NUM_STAGES, BLOCK_N, BLOCK_DMODEL],
                    layout=v_async_smem_layout,
                )

                for _s in gl.static_range(NUM_STAGES):
                    v_zero = gl.zeros(
                        [BLOCK_N, BLOCK_DMODEL],
                        dtype=Q_Extend.dtype.element_ty,
                        layout=v_async_layout,
                    )
                    v_smem.index(_s).store(v_zero)
                gl.barrier()

                q_dot = gl.convert_layout(q, q_dot_layout)

                n_prefix_blocks = (pfx_seq_len + BLOCK_N - 1) // BLOCK_N
                if n_prefix_blocks >= NUM_STAGES:
                    if NUM_STAGES >= 3:
                        acc, l_i, m_i = attn_fwd_inner_prefix_pipelined_scalar_mask(
                            acc,
                            l_i,
                            m_i,
                            q_dot,
                            K_Buffer,
                            V_Buffer,
                            kv_indices,
                            pfx_kv_start,
                            cur_kv_head,
                            pfx_seq_len,
                            stride_buf_kbs,
                            stride_buf_kh,
                            stride_buf_vbs,
                            stride_buf_vh,
                            kt_smem,
                            v_smem,
                            qk_scale,
                            LOGIT_CAP,
                            xai_temperature_reg,
                            XAI_TEMPERATURE_LEN,
                            pfx_q_abs_pos,
                            SLIDING_WINDOW_SIZE,
                            Mask,
                            pfx_mask_base,
                            mask_row_stride,
                            q_extend_offs,
                            USE_CUSTOM_MASK,
                            SKIP_PREFIX_CUSTOM_MASK,
                            ENABLE_PREFIX_UNMASKED,
                            BLOCK_M,
                            BLOCK_N,
                            BLOCK_DMODEL,
                            ACTUAL_BLOCK_DMODEL,
                            NUM_STAGES,
                            kt_async_layout,
                            v_async_layout,
                            kt_dot_layout,
                            p_dot_layout,
                            v_dot_layout,
                            mma_layout,
                            mma_offs_n_col,
                        )
                    else:
                        acc, l_i, m_i = attn_fwd_inner_prefix_pipelined(
                            acc,
                            l_i,
                            m_i,
                            q_dot,
                            K_Buffer,
                            V_Buffer,
                            kv_indices,
                            pfx_kv_start,
                            cur_kv_head,
                            pfx_seq_len,
                            stride_buf_kbs,
                            stride_buf_kh,
                            stride_buf_vbs,
                            stride_buf_vh,
                            kt_smem,
                            v_smem,
                            qk_scale,
                            LOGIT_CAP,
                            xai_temperature_reg,
                            XAI_TEMPERATURE_LEN,
                            pfx_q_abs_pos,
                            SLIDING_WINDOW_SIZE,
                            Mask,
                            pfx_mask_base,
                            mask_row_stride,
                            q_extend_offs,
                            USE_CUSTOM_MASK,
                            SKIP_PREFIX_CUSTOM_MASK,
                            ENABLE_PREFIX_UNMASKED,
                            BLOCK_M,
                            BLOCK_N,
                            BLOCK_DMODEL,
                            ACTUAL_BLOCK_DMODEL,
                            NUM_STAGES,
                            kt_async_layout,
                            v_async_layout,
                            kt_dot_layout,
                            p_dot_layout,
                            v_dot_layout,
                            mma_layout,
                            mma_offs_n_col,
                        )
                elif pfx_seq_len > 0:
                    acc, l_i, m_i = attn_fwd_inner_prefix_short(
                        acc,
                        l_i,
                        m_i,
                        q_dot,
                        K_Buffer,
                        V_Buffer,
                        kv_indices,
                        pfx_kv_start,
                        cur_kv_head,
                        pfx_seq_len,
                        stride_buf_kbs,
                        stride_buf_kh,
                        stride_buf_vbs,
                        stride_buf_vh,
                        kt_smem,
                        v_smem,
                        qk_scale,
                        LOGIT_CAP,
                        xai_temperature_reg,
                        XAI_TEMPERATURE_LEN,
                        pfx_q_abs_pos,
                        SLIDING_WINDOW_SIZE,
                        Mask,
                        pfx_mask_base,
                        mask_row_stride,
                        q_extend_offs,
                        USE_CUSTOM_MASK,
                        SKIP_PREFIX_CUSTOM_MASK,
                        ENABLE_PREFIX_UNMASKED,
                        BLOCK_M,
                        BLOCK_N,
                        BLOCK_DMODEL,
                        ACTUAL_BLOCK_DMODEL,
                        kt_async_layout,
                        v_async_layout,
                        kt_dot_layout,
                        p_dot_layout,
                        v_dot_layout,
                        mma_layout,
                        mma_offs_n_col,
                    )

                if IS_CAUSAL:
                    causal_kv_end = (cur_block_m + 1) * BLOCK_M
                    effective_end = tl.minimum(seq_len_extend, causal_kv_end)
                else:
                    effective_end = seq_len_extend
                n_extend_blocks = (effective_end + BLOCK_N - 1) // BLOCK_N
                if (
                    (not ENABLE_MASK_SPLIT)
                    or USE_CUSTOM_MASK
                    or SLIDING_WINDOW_SIZE > 0
                ):
                    n_full_blocks = 0
                else:
                    partial_block = ((effective_end % BLOCK_N) != 0).to(tl.int32)
                    if IS_CAUSAL:
                        masked_blocks = (
                            (BLOCK_M + BLOCK_N - 1) // BLOCK_N
                        ) + partial_block
                    else:
                        masked_blocks = partial_block
                    masked_blocks = tl.minimum(masked_blocks, n_extend_blocks)
                    n_full_blocks = n_extend_blocks - masked_blocks

                k_extend_base = (
                    K_Extend
                    + cur_seq_q_start_idx * stride_kbs
                    + cur_kv_head * stride_kh
                )
                v_extend_base = (
                    V_Extend
                    + cur_seq_q_start_idx * stride_vbs
                    + cur_kv_head * stride_vh
                )

                if n_full_blocks >= NUM_STAGES:
                    acc, l_i, m_i = attn_fwd_inner_extend_pipelined(
                        acc,
                        l_i,
                        m_i,
                        q_dot,
                        k_extend_base,
                        v_extend_base,
                        cur_block_m,
                        seq_len_extend,
                        stride_kbs,
                        stride_vbs,
                        0,
                        n_full_blocks,
                        kt_smem,
                        v_smem,
                        qk_scale,
                        LOGIT_CAP,
                        xai_temperature_reg,
                        XAI_TEMPERATURE_LEN,
                        SLIDING_WINDOW_SIZE,
                        IS_CAUSAL,
                        Mask,
                        mask_base_idx,
                        mask_row_stride,
                        mask_kv_col_offset,
                        USE_CUSTOM_MASK,
                        False,
                        BLOCK_M,
                        BLOCK_N,
                        BLOCK_DMODEL,
                        ACTUAL_BLOCK_DMODEL,
                        NUM_STAGES,
                        kt_async_layout,
                        v_async_layout,
                        kt_dot_layout,
                        p_dot_layout,
                        v_dot_layout,
                        mma_layout,
                        mma_offs_n_col,
                        mma_offs_m_row,
                    )
                elif n_full_blocks > 0:
                    acc, l_i, m_i = attn_fwd_inner_extend_short(
                        acc,
                        l_i,
                        m_i,
                        q_dot,
                        k_extend_base,
                        v_extend_base,
                        cur_block_m,
                        seq_len_extend,
                        stride_kbs,
                        stride_vbs,
                        0,
                        n_full_blocks,
                        kt_smem,
                        v_smem,
                        qk_scale,
                        LOGIT_CAP,
                        xai_temperature_reg,
                        XAI_TEMPERATURE_LEN,
                        SLIDING_WINDOW_SIZE,
                        IS_CAUSAL,
                        Mask,
                        mask_base_idx,
                        mask_row_stride,
                        mask_kv_col_offset,
                        USE_CUSTOM_MASK,
                        False,
                        BLOCK_M,
                        BLOCK_N,
                        BLOCK_DMODEL,
                        ACTUAL_BLOCK_DMODEL,
                        kt_async_layout,
                        v_async_layout,
                        kt_dot_layout,
                        p_dot_layout,
                        v_dot_layout,
                        mma_layout,
                        mma_offs_n_col,
                        mma_offs_m_row,
                    )
                masked_start = n_full_blocks
                remaining_blocks = n_extend_blocks - masked_start
                if remaining_blocks >= NUM_STAGES:
                    acc, l_i, m_i = attn_fwd_inner_extend_pipelined(
                        acc,
                        l_i,
                        m_i,
                        q_dot,
                        k_extend_base,
                        v_extend_base,
                        cur_block_m,
                        seq_len_extend,
                        stride_kbs,
                        stride_vbs,
                        masked_start,
                        n_extend_blocks,
                        kt_smem,
                        v_smem,
                        qk_scale,
                        LOGIT_CAP,
                        xai_temperature_reg,
                        XAI_TEMPERATURE_LEN,
                        SLIDING_WINDOW_SIZE,
                        IS_CAUSAL,
                        Mask,
                        mask_base_idx,
                        mask_row_stride,
                        mask_kv_col_offset,
                        USE_CUSTOM_MASK,
                        True,
                        BLOCK_M,
                        BLOCK_N,
                        BLOCK_DMODEL,
                        ACTUAL_BLOCK_DMODEL,
                        NUM_STAGES,
                        kt_async_layout,
                        v_async_layout,
                        kt_dot_layout,
                        p_dot_layout,
                        v_dot_layout,
                        mma_layout,
                        mma_offs_n_col,
                        mma_offs_m_row,
                    )
                elif remaining_blocks > 0:
                    acc, l_i, m_i = attn_fwd_inner_extend_short(
                        acc,
                        l_i,
                        m_i,
                        q_dot,
                        k_extend_base,
                        v_extend_base,
                        cur_block_m,
                        seq_len_extend,
                        stride_kbs,
                        stride_vbs,
                        masked_start,
                        n_extend_blocks,
                        kt_smem,
                        v_smem,
                        qk_scale,
                        LOGIT_CAP,
                        xai_temperature_reg,
                        XAI_TEMPERATURE_LEN,
                        SLIDING_WINDOW_SIZE,
                        IS_CAUSAL,
                        Mask,
                        mask_base_idx,
                        mask_row_stride,
                        mask_kv_col_offset,
                        USE_CUSTOM_MASK,
                        True,
                        BLOCK_M,
                        BLOCK_N,
                        BLOCK_DMODEL,
                        ACTUAL_BLOCK_DMODEL,
                        kt_async_layout,
                        v_async_layout,
                        kt_dot_layout,
                        p_dot_layout,
                        v_dot_layout,
                        mma_layout,
                        mma_offs_n_col,
                        mma_offs_m_row,
                    )

            else:
                # 8-warp BLOCK_DMODEL < 128
                kt_offset_bases: gl.constexpr = [
                    [1, 0],
                    [2, 0],
                    [4, 0],
                    [8, 0],
                    [16, 0],
                    [32, 0],
                    [0, 16],
                    [0, 32],
                    [0, 1],
                    [0, 2],
                    [0, 4],
                    [0, 8],
                ]
                v_offset_bases: gl.constexpr = [
                    [0, 1],
                    [0, 2],
                    [0, 4],
                    [0, 8],
                    [0, 16],
                    [0, 32],
                    [16, 0],
                    [32, 0],
                    [1, 0],
                    [2, 0],
                    [4, 0],
                    [8, 0],
                ]
                kt_async_layout: gl.constexpr = DistributedLinearLayout(
                    reg_bases=[[1, 0], [2, 0], [4, 0]],
                    lane_bases=[[8, 0], [16, 0], [32, 0], [0, 16], [0, 32], [0, 1]],
                    warp_bases=[[0, 2], [0, 4], [0, 8]],
                    block_bases=[],
                    shape=[BLOCK_DMODEL, BLOCK_N],
                )
                v_async_layout: gl.constexpr = DistributedLinearLayout(
                    reg_bases=[[0, 1], [0, 2], [0, 4]],
                    lane_bases=[[0, 8], [0, 16], [0, 32], [16, 0], [32, 0], [1, 0]],
                    warp_bases=[[2, 0], [4, 0], [8, 0]],
                    block_bases=[],
                    shape=[BLOCK_N, BLOCK_DMODEL],
                )

                kt_async_smem_layout: gl.constexpr = PaddedSharedLayout(
                    interval_padding_pairs=[[512, ASYNC_PAD_K]],
                    offset_bases=kt_offset_bases,
                    cga_layout=[],
                    shape=[BLOCK_DMODEL, BLOCK_N],
                )
                v_async_smem_layout: gl.constexpr = PaddedSharedLayout(
                    interval_padding_pairs=[[512, ASYNC_PAD_V]],
                    offset_bases=v_offset_bases,
                    cga_layout=[],
                    shape=[BLOCK_N, BLOCK_DMODEL],
                )

                kt_smem = gl.allocate_shared_memory(
                    Q_Extend.dtype.element_ty,
                    [NUM_STAGES, BLOCK_DMODEL, BLOCK_N],
                    layout=kt_async_smem_layout,
                )
                v_smem = gl.allocate_shared_memory(
                    Q_Extend.dtype.element_ty,
                    [NUM_STAGES, BLOCK_N, BLOCK_DMODEL],
                    layout=v_async_smem_layout,
                )

                for _s in gl.static_range(NUM_STAGES):
                    v_zero = gl.zeros(
                        [BLOCK_N, BLOCK_DMODEL],
                        dtype=Q_Extend.dtype.element_ty,
                        layout=v_async_layout,
                    )
                    v_smem.index(_s).store(v_zero)
                gl.barrier()

                q_dot = gl.convert_layout(q, q_dot_layout)

                n_prefix_blocks = (pfx_seq_len + BLOCK_N - 1) // BLOCK_N
                if n_prefix_blocks >= NUM_STAGES:
                    acc, l_i, m_i = attn_fwd_inner_prefix_pipelined(
                        acc,
                        l_i,
                        m_i,
                        q_dot,
                        K_Buffer,
                        V_Buffer,
                        kv_indices,
                        pfx_kv_start,
                        cur_kv_head,
                        pfx_seq_len,
                        stride_buf_kbs,
                        stride_buf_kh,
                        stride_buf_vbs,
                        stride_buf_vh,
                        kt_smem,
                        v_smem,
                        qk_scale,
                        LOGIT_CAP,
                        xai_temperature_reg,
                        XAI_TEMPERATURE_LEN,
                        pfx_q_abs_pos,
                        SLIDING_WINDOW_SIZE,
                        Mask,
                        pfx_mask_base,
                        mask_row_stride,
                        q_extend_offs,
                        USE_CUSTOM_MASK,
                        SKIP_PREFIX_CUSTOM_MASK,
                        ENABLE_PREFIX_UNMASKED,
                        BLOCK_M,
                        BLOCK_N,
                        BLOCK_DMODEL,
                        ACTUAL_BLOCK_DMODEL,
                        NUM_STAGES,
                        kt_async_layout,
                        v_async_layout,
                        kt_dot_layout,
                        p_dot_layout,
                        v_dot_layout,
                        mma_layout,
                        mma_offs_n_col,
                    )
                elif pfx_seq_len > 0:
                    acc, l_i, m_i = attn_fwd_inner_prefix_short(
                        acc,
                        l_i,
                        m_i,
                        q_dot,
                        K_Buffer,
                        V_Buffer,
                        kv_indices,
                        pfx_kv_start,
                        cur_kv_head,
                        pfx_seq_len,
                        stride_buf_kbs,
                        stride_buf_kh,
                        stride_buf_vbs,
                        stride_buf_vh,
                        kt_smem,
                        v_smem,
                        qk_scale,
                        LOGIT_CAP,
                        xai_temperature_reg,
                        XAI_TEMPERATURE_LEN,
                        pfx_q_abs_pos,
                        SLIDING_WINDOW_SIZE,
                        Mask,
                        pfx_mask_base,
                        mask_row_stride,
                        q_extend_offs,
                        USE_CUSTOM_MASK,
                        SKIP_PREFIX_CUSTOM_MASK,
                        ENABLE_PREFIX_UNMASKED,
                        BLOCK_M,
                        BLOCK_N,
                        BLOCK_DMODEL,
                        ACTUAL_BLOCK_DMODEL,
                        kt_async_layout,
                        v_async_layout,
                        kt_dot_layout,
                        p_dot_layout,
                        v_dot_layout,
                        mma_layout,
                        mma_offs_n_col,
                    )

                if IS_CAUSAL:
                    causal_kv_end = (cur_block_m + 1) * BLOCK_M
                    effective_end = tl.minimum(seq_len_extend, causal_kv_end)
                else:
                    effective_end = seq_len_extend
                n_extend_blocks = (effective_end + BLOCK_N - 1) // BLOCK_N
                if (
                    (not ENABLE_MASK_SPLIT)
                    or USE_CUSTOM_MASK
                    or SLIDING_WINDOW_SIZE > 0
                ):
                    n_full_blocks = 0
                else:
                    partial_block = ((effective_end % BLOCK_N) != 0).to(tl.int32)
                    if IS_CAUSAL:
                        masked_blocks = (
                            (BLOCK_M + BLOCK_N - 1) // BLOCK_N
                        ) + partial_block
                    else:
                        masked_blocks = partial_block
                    masked_blocks = tl.minimum(masked_blocks, n_extend_blocks)
                    n_full_blocks = n_extend_blocks - masked_blocks

                k_extend_base = (
                    K_Extend
                    + cur_seq_q_start_idx * stride_kbs
                    + cur_kv_head * stride_kh
                )
                v_extend_base = (
                    V_Extend
                    + cur_seq_q_start_idx * stride_vbs
                    + cur_kv_head * stride_vh
                )

                if n_full_blocks >= NUM_STAGES:
                    acc, l_i, m_i = attn_fwd_inner_extend_pipelined(
                        acc,
                        l_i,
                        m_i,
                        q_dot,
                        k_extend_base,
                        v_extend_base,
                        cur_block_m,
                        seq_len_extend,
                        stride_kbs,
                        stride_vbs,
                        0,
                        n_full_blocks,
                        kt_smem,
                        v_smem,
                        qk_scale,
                        LOGIT_CAP,
                        xai_temperature_reg,
                        XAI_TEMPERATURE_LEN,
                        SLIDING_WINDOW_SIZE,
                        IS_CAUSAL,
                        Mask,
                        mask_base_idx,
                        mask_row_stride,
                        mask_kv_col_offset,
                        USE_CUSTOM_MASK,
                        False,
                        BLOCK_M,
                        BLOCK_N,
                        BLOCK_DMODEL,
                        ACTUAL_BLOCK_DMODEL,
                        NUM_STAGES,
                        kt_async_layout,
                        v_async_layout,
                        kt_dot_layout,
                        p_dot_layout,
                        v_dot_layout,
                        mma_layout,
                        mma_offs_n_col,
                        mma_offs_m_row,
                    )
                elif n_full_blocks > 0:
                    acc, l_i, m_i = attn_fwd_inner_extend_short(
                        acc,
                        l_i,
                        m_i,
                        q_dot,
                        k_extend_base,
                        v_extend_base,
                        cur_block_m,
                        seq_len_extend,
                        stride_kbs,
                        stride_vbs,
                        0,
                        n_full_blocks,
                        kt_smem,
                        v_smem,
                        qk_scale,
                        LOGIT_CAP,
                        xai_temperature_reg,
                        XAI_TEMPERATURE_LEN,
                        SLIDING_WINDOW_SIZE,
                        IS_CAUSAL,
                        Mask,
                        mask_base_idx,
                        mask_row_stride,
                        mask_kv_col_offset,
                        USE_CUSTOM_MASK,
                        False,
                        BLOCK_M,
                        BLOCK_N,
                        BLOCK_DMODEL,
                        ACTUAL_BLOCK_DMODEL,
                        kt_async_layout,
                        v_async_layout,
                        kt_dot_layout,
                        p_dot_layout,
                        v_dot_layout,
                        mma_layout,
                        mma_offs_n_col,
                        mma_offs_m_row,
                    )
                masked_start = n_full_blocks
                remaining_blocks = n_extend_blocks - masked_start
                if remaining_blocks >= NUM_STAGES:
                    acc, l_i, m_i = attn_fwd_inner_extend_pipelined(
                        acc,
                        l_i,
                        m_i,
                        q_dot,
                        k_extend_base,
                        v_extend_base,
                        cur_block_m,
                        seq_len_extend,
                        stride_kbs,
                        stride_vbs,
                        masked_start,
                        n_extend_blocks,
                        kt_smem,
                        v_smem,
                        qk_scale,
                        LOGIT_CAP,
                        xai_temperature_reg,
                        XAI_TEMPERATURE_LEN,
                        SLIDING_WINDOW_SIZE,
                        IS_CAUSAL,
                        Mask,
                        mask_base_idx,
                        mask_row_stride,
                        mask_kv_col_offset,
                        USE_CUSTOM_MASK,
                        True,
                        BLOCK_M,
                        BLOCK_N,
                        BLOCK_DMODEL,
                        ACTUAL_BLOCK_DMODEL,
                        NUM_STAGES,
                        kt_async_layout,
                        v_async_layout,
                        kt_dot_layout,
                        p_dot_layout,
                        v_dot_layout,
                        mma_layout,
                        mma_offs_n_col,
                        mma_offs_m_row,
                    )
                elif remaining_blocks > 0:
                    acc, l_i, m_i = attn_fwd_inner_extend_short(
                        acc,
                        l_i,
                        m_i,
                        q_dot,
                        k_extend_base,
                        v_extend_base,
                        cur_block_m,
                        seq_len_extend,
                        stride_kbs,
                        stride_vbs,
                        masked_start,
                        n_extend_blocks,
                        kt_smem,
                        v_smem,
                        qk_scale,
                        LOGIT_CAP,
                        xai_temperature_reg,
                        XAI_TEMPERATURE_LEN,
                        SLIDING_WINDOW_SIZE,
                        IS_CAUSAL,
                        Mask,
                        mask_base_idx,
                        mask_row_stride,
                        mask_kv_col_offset,
                        USE_CUSTOM_MASK,
                        True,
                        BLOCK_M,
                        BLOCK_N,
                        BLOCK_DMODEL,
                        ACTUAL_BLOCK_DMODEL,
                        kt_async_layout,
                        v_async_layout,
                        kt_dot_layout,
                        p_dot_layout,
                        v_dot_layout,
                        mma_layout,
                        mma_offs_n_col,
                        mma_offs_m_row,
                    )

        if HAS_SINK:
            cur_sink = gl.load(Sinks + cur_head)
            l_i = l_i + gl.exp2(cur_sink * LOG2E - m_i)

        l_recip = 1.0 / l_i
        acc = acc * l_recip[:, None]
        if V_SCALE != 1.0:
            acc = acc * V_SCALE

        o_ptrs = (
            O_Extend
            + (cur_seq_q_start_idx + cur_block_m * BLOCK_M + offs_m[:, None])
            * stride_obs
            + cur_head * stride_oh
            + offs_d[None, :]
        )
        o_mask = (cur_block_m * BLOCK_M + offs_m[:, None]) < seq_len_extend
        if ACTUAL_BLOCK_DMODEL != BLOCK_DMODEL:
            o_mask = o_mask & (offs_d[None, :] < ACTUAL_BLOCK_DMODEL)
        out = gl.convert_layout(acc, blocked_layout).to(O_Extend.dtype.element_ty)
        gl.store(o_ptrs, out, mask=o_mask)

        tile_idx += total_programs


# ===-----------------------------------------------------------------------===#
# Persistent-CTA Tile Scheduling
# ===-----------------------------------------------------------------------===#


def _build_tile_schedule(qo_indptr, head_num, BLOCK_M, device):
    extend_lens_cpu = (qo_indptr[1:] - qo_indptr[:-1]).cpu().tolist()
    batch_size = len(extend_lens_cpu)
    n_m_tiles_per_batch = [(el + BLOCK_M - 1) // BLOCK_M for el in extend_lens_cpu]
    total_m_tiles = sum(n_m_tiles_per_batch)
    total_valid_tiles = total_m_tiles * head_num

    if total_valid_tiles == 0:
        empty = torch.empty(0, dtype=torch.int32, device=device)
        return empty, empty, empty, 0

    tile_b = torch.empty(total_valid_tiles, dtype=torch.int32)
    tile_h = torch.empty(total_valid_tiles, dtype=torch.int32)
    tile_m = torch.empty(total_valid_tiles, dtype=torch.int32)

    idx = 0
    for b in range(batch_size):
        n_m = n_m_tiles_per_batch[b]
        seg = n_m * head_num
        m_range = torch.arange(n_m, dtype=torch.int32)
        h_range = torch.arange(head_num, dtype=torch.int32)
        tile_b[idx : idx + seg] = b
        tile_h[idx : idx + seg] = h_range.repeat_interleave(n_m)
        tile_m[idx : idx + seg] = m_range.repeat(head_num)
        idx += seg

    return (tile_b.to(device), tile_h.to(device), tile_m.to(device), total_valid_tiles)


# ===-----------------------------------------------------------------------===#
# Python Wrappers
# ===-----------------------------------------------------------------------===#


def _launch_persistent(
    q_extend,
    k_extend,
    v_extend,
    o_extend,
    k_buffer,
    v_buffer,
    qo_indptr,
    kv_indptr,
    kv_indices,
    custom_mask,
    is_causal,
    mask_indptr,
    max_len_extend,
    k_scale=1.0,
    v_scale=1.0,
    sm_scale=None,
    logit_cap=0.0,
    skip_prefix_custom_mask=True,
    sliding_window_size=-1,
    sinks=None,
    window_kv_offsets=None,
    xai_temperature_len=-1,
    enable_mask_split=True,
    enable_prefix_unmasked=True,
    _force_block_m=None,
    _force_num_warps=None,
    _force_num_stages=None,
    _force_mma_shape=None,
    _force_waves_per_eu=None,
    _force_async_pad_k=None,
    _force_async_pad_v=None,
    min_len_extend=None,
):
    Lq = q_extend.shape[-1]
    Lv = v_extend.shape[-1]
    assert Lq == Lv

    USE_CUSTOM_MASK = custom_mask is not None
    SKIP_PREFIX_CUSTOM_MASK = skip_prefix_custom_mask
    if not USE_CUSTOM_MASK:
        custom_mask = torch.empty(0, dtype=torch.uint8, device=q_extend.device)
        mask_indptr = torch.zeros(
            q_extend.shape[0] + 1, dtype=torch.int64, device=q_extend.device
        )
    if window_kv_offsets is None:
        window_kv_offsets = torch.zeros(
            qo_indptr.shape[0] - 1, dtype=torch.int32, device=q_extend.device
        )
    assert q_extend.shape[1] % k_extend.shape[1] == 0

    BLOCK_DMODEL = max(triton.next_power_of_2(Lq), 16)
    BLOCK_N = 64
    batch_size = qo_indptr.shape[0] - 1

    if min_len_extend is None:
        extend_lens = qo_indptr[1:] - qo_indptr[:-1]
        min_len_extend = int(extend_lens.min().item())
    head_num = q_extend.shape[1]

    if _force_block_m is not None and _force_num_warps is not None:
        BLOCK_M = _force_block_m
        num_warps = _force_num_warps
    elif max_len_extend <= 128:
        BLOCK_M = 128
        num_warps = 8
    elif batch_size <= 4:
        BLOCK_M = 128
        num_warps = 8
    elif BLOCK_DMODEL >= 128 and min_len_extend >= 64 and max_len_extend >= 256:
        BLOCK_M = 256
        num_warps = 8
    else:
        BLOCK_M = 128
        num_warps = 8

    if _force_num_stages is not None:
        NUM_STAGES = _force_num_stages
    elif BLOCK_M == 64:
        NUM_STAGES = 1
    else:
        NUM_STAGES = 4

    if (
        BLOCK_M == 128
        and num_warps == 8
        and NUM_STAGES == 2
        and _force_num_stages is None
    ):
        NUM_STAGES = 3

    if _force_mma_shape == "32x32x16":
        MMA_INSTR_M, MMA_INSTR_N, MMA_INSTR_K = 32, 32, 16
        QK_K_WIDTH, PV_K_WIDTH = 32, 4
    else:
        MMA_INSTR_M, MMA_INSTR_N, MMA_INSTR_K = 16, 16, 32
        QK_K_WIDTH, PV_K_WIDTH = 8, 4

    ASYNC_PAD_K = _force_async_pad_k if _force_async_pad_k is not None else 16
    ASYNC_PAD_V = _force_async_pad_v if _force_async_pad_v is not None else 16

    sm_scale = sm_scale or 1.0 / math.sqrt(Lq)
    sm_scale = sm_scale * k_scale
    kv_group_num = q_extend.shape[1] // k_extend.shape[1]

    device = q_extend.device
    tile_seq_info, tile_head_info, tile_m_info, total_valid_tiles = (
        _build_tile_schedule(qo_indptr, head_num, BLOCK_M, device)
    )

    if total_valid_tiles == 0:
        return

    num_CUs = torch.cuda.get_device_properties(device).multi_processor_count
    total_programs = min(total_valid_tiles, 2 * num_CUs)
    grid = (total_programs,)

    gluon_extend_attn_fwd_persistent[grid](
        q_extend,
        k_extend,
        v_extend,
        o_extend,
        k_buffer,
        v_buffer,
        qo_indptr,
        kv_indptr,
        kv_indices,
        custom_mask,
        mask_indptr,
        window_kv_offsets,
        sm_scale,
        kv_group_num,
        q_extend.stride(0),
        q_extend.stride(1),
        k_extend.stride(0),
        k_extend.stride(1),
        v_extend.stride(0),
        v_extend.stride(1),
        o_extend.stride(0),
        o_extend.stride(1),
        k_buffer.stride(0),
        k_buffer.stride(1),
        v_buffer.stride(0),
        v_buffer.stride(1),
        tile_seq_info,
        tile_head_info,
        tile_m_info,
        total_valid_tiles,
        total_programs,
        IS_CAUSAL=is_causal,
        USE_CUSTOM_MASK=USE_CUSTOM_MASK,
        SKIP_PREFIX_CUSTOM_MASK=SKIP_PREFIX_CUSTOM_MASK,
        ENABLE_PREFIX_UNMASKED=enable_prefix_unmasked,
        ENABLE_MASK_SPLIT=enable_mask_split,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_DMODEL=BLOCK_DMODEL,
        ACTUAL_BLOCK_DMODEL=Lq,
        NUM_STAGES=NUM_STAGES,
        MMA_INSTR_M=MMA_INSTR_M,
        MMA_INSTR_N=MMA_INSTR_N,
        MMA_INSTR_K=MMA_INSTR_K,
        QK_K_WIDTH=QK_K_WIDTH,
        PV_K_WIDTH=PV_K_WIDTH,
        ASYNC_PAD_K=ASYNC_PAD_K,
        ASYNC_PAD_V=ASYNC_PAD_V,
        Sinks=sinks,
        HAS_SINK=sinks is not None,
        LOGIT_CAP=logit_cap,
        XAI_TEMPERATURE_LEN=xai_temperature_len,
        SLIDING_WINDOW_SIZE=sliding_window_size,
        V_SCALE=v_scale,
        num_warps=num_warps,
        num_stages=1,
        waves_per_eu=_force_waves_per_eu if _force_waves_per_eu is not None else 2,
        matrix_instr_nonkdim=32,
    )


_dummy_cm = None
_dummy_mi = None
_dummy_mi_size = 0
_dummy_wkvo = None
_dummy_wkvo_size = 0


def _ensure_dummies(device, mi_size, wkvo_size):
    """Lazy-init module-level singleton dummy tensors on first use."""
    global _dummy_cm, _dummy_mi, _dummy_mi_size, _dummy_wkvo, _dummy_wkvo_size
    if _dummy_cm is None:
        _dummy_cm = torch.empty(0, dtype=torch.uint8, device=device)
    if _dummy_mi is None or _dummy_mi_size < mi_size:
        _dummy_mi = torch.zeros(mi_size, dtype=torch.int64, device=device)
        _dummy_mi_size = mi_size
    if _dummy_wkvo is None or _dummy_wkvo_size < wkvo_size:
        _dummy_wkvo = torch.zeros(wkvo_size, dtype=torch.int32, device=device)
        _dummy_wkvo_size = wkvo_size


def gluon_extend_attention_fwd(
    q_extend,
    k_extend,
    v_extend,
    o_extend,
    k_buffer,
    v_buffer,
    qo_indptr,
    kv_indptr,
    kv_indices,
    custom_mask,
    is_causal,
    mask_indptr,
    max_len_extend,
    k_scale=1.0,
    v_scale=1.0,
    sm_scale=None,
    logit_cap=0.0,
    skip_prefix_custom_mask=True,
    sliding_window_size=-1,
    sinks=None,
    window_kv_offsets=None,
    xai_temperature_len=-1,
    _force_block_m=None,
    _force_num_warps=None,
    _force_num_stages=None,
    _force_mma_shape=None,
    _force_waves_per_eu=None,
    _force_async_pad_k=None,
    _force_async_pad_v=None,
    _force_use_persistent=None,
    _mask_split_ext_threshold=1024,
    min_len_extend=None,
    total_prefix_len=None,
    total_extend_len=None,
):
    Lq = q_extend.shape[-1]
    batch_size = qo_indptr.shape[0] - 1
    head_num = q_extend.shape[1]

    # -- Fast path: no custom mask, no test overrides, uniform batch or B=1 --
    # Skips all heuristic computation and launches with hardcoded constants.
    # 4w-s4 dispatch: BM=64/NW=4/NS=4 when total KV per request is moderate.
    if (
        _force_block_m is None
        and _force_use_persistent is None
        and custom_mask is None
        and (batch_size <= 1 or min_len_extend == max_len_extend)
    ):
        _sm = (sm_scale if sm_scale is not None else Lq**-0.5) * k_scale
        _kv_gn = head_num // k_extend.shape[1]
        _BLOCK_DMODEL = (
            Lq
            if (Lq & (Lq - 1) == 0 and Lq >= 16)
            else max(triton.next_power_of_2(Lq), 16)
        )
        _wkvo = window_kv_offsets
        if _wkvo is None or _dummy_cm is None:
            _ensure_dummies(q_extend.device, q_extend.shape[0] + 1, batch_size)
        if _wkvo is None:
            _wkvo = _dummy_wkvo[:batch_size]

        _BM, _NW, _NS = 128, 8, 4
        if max_len_extend <= 512:
            _total_pfx = kv_indices.shape[0]
            _avg_pfx = _total_pfx // max(1, batch_size)
            if batch_size == 1 and _avg_pfx >= 256:
                _avg_total = _avg_pfx + max_len_extend
                if 512 <= _avg_total <= 1024:
                    _BM, _NW, _NS = 64, 4, 4
            elif batch_size <= 8 and max_len_extend <= 64 and 768 <= _avg_pfx <= 1280:
                _BM, _NW, _NS = 64, 4, 4
            elif batch_size > 8 and max_len_extend <= 64 and 384 <= _avg_pfx <= 768:
                _BM, _NW, _NS = 64, 4, 4

        gluon_extend_attn_fwd.run(
            q_extend,
            k_extend,
            v_extend,
            o_extend,
            k_buffer,
            v_buffer,
            qo_indptr,
            kv_indptr,
            kv_indices,
            _dummy_cm,
            _dummy_mi[: q_extend.shape[0] + 1],
            _wkvo,
            _sm,
            _kv_gn,
            q_extend.stride(0),
            q_extend.stride(1),
            k_extend.stride(0),
            k_extend.stride(1),
            v_extend.stride(0),
            v_extend.stride(1),
            o_extend.stride(0),
            o_extend.stride(1),
            k_buffer.stride(0),
            k_buffer.stride(1),
            v_buffer.stride(0),
            v_buffer.stride(1),
            IS_CAUSAL=is_causal,
            USE_CUSTOM_MASK=False,
            SKIP_PREFIX_CUSTOM_MASK=skip_prefix_custom_mask,
            ENABLE_PREFIX_UNMASKED=False,
            ENABLE_MASK_SPLIT=False,
            BLOCK_M=_BM,
            BLOCK_N=64,
            BLOCK_DMODEL=_BLOCK_DMODEL,
            ACTUAL_BLOCK_DMODEL=Lq,
            NUM_STAGES=_NS,
            MMA_INSTR_M=16,
            MMA_INSTR_N=16,
            MMA_INSTR_K=32,
            QK_K_WIDTH=8,
            PV_K_WIDTH=4,
            ASYNC_PAD_K=16,
            ASYNC_PAD_V=16,
            Sinks=sinks,
            HAS_SINK=sinks is not None,
            LOGIT_CAP=logit_cap,
            XAI_TEMPERATURE_LEN=xai_temperature_len,
            SLIDING_WINDOW_SIZE=sliding_window_size,
            V_SCALE=v_scale,
            num_warps=_NW,
            num_stages=1,
            waves_per_eu=2,
            matrix_instr_nonkdim=32,
            grid=(batch_size, head_num, (max_len_extend + _BM - 1) // _BM),
            warmup=False,
        )
        return

    # -- Full dispatch path (heterogeneous batches, custom mask, test overrides) --
    Lv = v_extend.shape[-1]
    assert Lq == Lv, "BLOCK_DV != BLOCK_DMODEL not yet supported"

    if min_len_extend is None:
        min_len_extend = int((qo_indptr[1:] - qo_indptr[:-1]).min().item())

    _ensure_dummies(q_extend.device, q_extend.shape[0] + 1, batch_size)

    # Persistent CTA dispatch.  Uses CPU-side totals when available.
    needs_detailed_check = False
    if _force_use_persistent is not None:
        use_persistent = bool(_force_use_persistent)
    else:
        ext_ratio = max_len_extend / max(1, min_len_extend)
        use_persistent = False

        needs_detailed_check = (ext_ratio > 20.0 and max_len_extend >= 1024) or (
            max_len_extend >= 2048 and batch_size >= 2
        )
        if needs_detailed_check:
            if total_prefix_len is not None and total_extend_len is not None:
                total_prefix = total_prefix_len
                total_extend = total_extend_len
            else:
                total_prefix = int((kv_indptr[-1] - kv_indptr[0]).item())
                total_extend = int((qo_indptr[-1] - qo_indptr[0]).item())
            ppct = total_prefix / max(1, total_prefix + total_extend)

            work_ratio = 1.0
            if batch_size >= 2 and max_len_extend >= 2048 and ext_ratio <= 4.0:
                ext_lens = (qo_indptr[1:] - qo_indptr[:-1]).tolist()
                pfx_lens = (kv_indptr[1:] - kv_indptr[:-1]).tolist()
                per_req = [e * (p + e) for e, p in zip(ext_lens, pfx_lens)]
                work_ratio = max(per_req) / max(1, min(per_req))

            use_persistent = (
                (ext_ratio > 20.0 and max_len_extend >= 1024 and ppct < 0.95)
                or (ext_ratio > 4.0 and max_len_extend >= 2048 and batch_size >= 2)
                or (work_ratio > 1.5 and max_len_extend >= 2048 and batch_size >= 2)
            )

    if use_persistent:
        _launch_persistent(
            q_extend,
            k_extend,
            v_extend,
            o_extend,
            k_buffer,
            v_buffer,
            qo_indptr,
            kv_indptr,
            kv_indices,
            custom_mask,
            is_causal,
            mask_indptr,
            max_len_extend,
            k_scale=k_scale,
            v_scale=v_scale,
            sm_scale=sm_scale,
            logit_cap=logit_cap,
            skip_prefix_custom_mask=skip_prefix_custom_mask,
            sliding_window_size=sliding_window_size,
            sinks=sinks,
            window_kv_offsets=window_kv_offsets,
            xai_temperature_len=xai_temperature_len,
            _force_block_m=_force_block_m,
            _force_num_warps=_force_num_warps,
            _force_num_stages=_force_num_stages,
            _force_mma_shape=_force_mma_shape,
            _force_waves_per_eu=_force_waves_per_eu,
            _force_async_pad_k=_force_async_pad_k,
            _force_async_pad_v=_force_async_pad_v,
            min_len_extend=min_len_extend,
        )
        return

    # Mask-split dispatch.
    if max_len_extend >= 2048 and batch_size >= 2:
        if not needs_detailed_check:
            if total_prefix_len is not None and total_extend_len is not None:
                total_prefix = total_prefix_len
                total_extend = total_extend_len
            else:
                total_prefix = int((kv_indptr[-1] - kv_indptr[0]).item())
                total_extend = int((qo_indptr[-1] - qo_indptr[0]).item())
            ppct = total_prefix / max(1, total_prefix + total_extend)
        enable_mask_split = ppct < 0.60
    else:
        enable_mask_split = False
    enable_prefix_unmasked = enable_mask_split

    USE_CUSTOM_MASK = custom_mask is not None
    if not USE_CUSTOM_MASK:
        custom_mask = _dummy_cm
        mask_indptr = _dummy_mi[: q_extend.shape[0] + 1]
    if window_kv_offsets is None:
        window_kv_offsets = _dummy_wkvo[:batch_size]

    BLOCK_DMODEL = max(triton.next_power_of_2(Lq), 16)
    BLOCK_N = 64

    # 4w-s4 dispatch for moderate-prefix short-extend (full dispatch path).
    use_4w_dispatch = False
    if _force_block_m is None and _force_num_warps is None and max_len_extend <= 512:
        total_prefix = kv_indices.shape[0]
        avg_pfx = total_prefix // max(1, batch_size)
        if batch_size == 1 and avg_pfx >= 256:
            avg_total = avg_pfx + max_len_extend
            use_4w_dispatch = 512 <= avg_total <= 1024
        elif batch_size <= 8 and max_len_extend <= 64:
            use_4w_dispatch = 768 <= avg_pfx <= 1280
        elif batch_size > 8 and max_len_extend <= 64:
            use_4w_dispatch = 384 <= avg_pfx <= 768

    if _force_block_m is not None and _force_num_warps is not None:
        BLOCK_M = _force_block_m
        num_warps = _force_num_warps
    elif use_4w_dispatch:
        BLOCK_M = 64
        num_warps = 4
    elif max_len_extend <= 128 or batch_size <= 4:
        BLOCK_M = 128
        num_warps = 8
    elif BLOCK_DMODEL >= 128 and min_len_extend >= 64 and max_len_extend >= 256:
        BLOCK_M = 256
        num_warps = 8
    else:
        BLOCK_M = 128
        num_warps = 8

    if _force_num_stages is not None:
        NUM_STAGES = _force_num_stages
    elif use_4w_dispatch:
        NUM_STAGES = 4
    elif BLOCK_M == 64:
        NUM_STAGES = 1
    else:
        NUM_STAGES = 4

    if (
        BLOCK_M == 128
        and num_warps == 8
        and NUM_STAGES == 2
        and _force_num_stages is None
    ):
        NUM_STAGES = 3

    if _force_mma_shape == "32x32x16":
        MMA_INSTR_M, MMA_INSTR_N, MMA_INSTR_K = 32, 32, 16
        QK_K_WIDTH, PV_K_WIDTH = 32, 4
    else:
        MMA_INSTR_M, MMA_INSTR_N, MMA_INSTR_K = 16, 16, 32
        QK_K_WIDTH, PV_K_WIDTH = 8, 4

    ASYNC_PAD_K = _force_async_pad_k if _force_async_pad_k is not None else 16
    ASYNC_PAD_V = _force_async_pad_v if _force_async_pad_v is not None else 16

    sm_scale = sm_scale or 1.0 / math.sqrt(Lq)
    sm_scale = sm_scale * k_scale
    kv_group_num = q_extend.shape[1] // k_extend.shape[1]

    gluon_extend_attn_fwd.run(
        q_extend,
        k_extend,
        v_extend,
        o_extend,
        k_buffer,
        v_buffer,
        qo_indptr,
        kv_indptr,
        kv_indices,
        custom_mask,
        mask_indptr,
        window_kv_offsets,
        sm_scale,
        kv_group_num,
        q_extend.stride(0),
        q_extend.stride(1),
        k_extend.stride(0),
        k_extend.stride(1),
        v_extend.stride(0),
        v_extend.stride(1),
        o_extend.stride(0),
        o_extend.stride(1),
        k_buffer.stride(0),
        k_buffer.stride(1),
        v_buffer.stride(0),
        v_buffer.stride(1),
        IS_CAUSAL=is_causal,
        USE_CUSTOM_MASK=USE_CUSTOM_MASK,
        SKIP_PREFIX_CUSTOM_MASK=skip_prefix_custom_mask,
        ENABLE_PREFIX_UNMASKED=enable_prefix_unmasked,
        ENABLE_MASK_SPLIT=enable_mask_split,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_DMODEL=BLOCK_DMODEL,
        ACTUAL_BLOCK_DMODEL=Lq,
        NUM_STAGES=NUM_STAGES,
        MMA_INSTR_M=MMA_INSTR_M,
        MMA_INSTR_N=MMA_INSTR_N,
        MMA_INSTR_K=MMA_INSTR_K,
        QK_K_WIDTH=QK_K_WIDTH,
        PV_K_WIDTH=PV_K_WIDTH,
        ASYNC_PAD_K=ASYNC_PAD_K,
        ASYNC_PAD_V=ASYNC_PAD_V,
        Sinks=sinks,
        HAS_SINK=sinks is not None,
        LOGIT_CAP=logit_cap,
        XAI_TEMPERATURE_LEN=xai_temperature_len,
        SLIDING_WINDOW_SIZE=sliding_window_size,
        V_SCALE=v_scale,
        num_warps=num_warps,
        num_stages=1,
        waves_per_eu=_force_waves_per_eu if _force_waves_per_eu is not None else 2,
        matrix_instr_nonkdim=32,
        grid=(batch_size, head_num, triton.cdiv(max_len_extend, BLOCK_M)),
        warmup=False,
    )


# ===-----------------------------------------------------------------------===#
# Test / Benchmark Helpers
# ===-----------------------------------------------------------------------===#


def _run_gluon(q, k, v, kb, vb, qo, kv, ki, o, elens, causal, **kw):
    gluon_extend_attention_fwd(
        q,
        k,
        v,
        o,
        kb,
        vb,
        qo,
        kv,
        ki,
        custom_mask=None,
        is_causal=causal,
        mask_indptr=None,
        max_len_extend=max(elens),
        sm_scale=1.0 / math.sqrt(q.shape[-1]),
        min_len_extend=min(elens),
        **kw,
    )


def _run_gluon_persistent(q, k, v, kb, vb, qo, kv, ki, o, elens, causal, **kw):
    _launch_persistent(
        q,
        k,
        v,
        o,
        kb,
        vb,
        qo,
        kv,
        ki,
        custom_mask=None,
        is_causal=causal,
        mask_indptr=None,
        max_len_extend=max(elens),
        sm_scale=1.0 / math.sqrt(q.shape[-1]),
        min_len_extend=min(elens),
        **kw,
    )
