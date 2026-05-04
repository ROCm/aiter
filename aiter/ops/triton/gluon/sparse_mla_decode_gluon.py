# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

# Sparse MLA decode kernel (DeepSeek-V4) derived from `mla_decode_gluon`.
#
# Triton requirement: this kernel uses the same gluon-language layouts as
# `mla_decode_gluon` (PaddedSharedLayout etc.) and therefore requires
# AMD's `amd-triton` wheel (>= 3.7+amd) — community Triton 3.6's gluon
# API does not match. JIT-compilation is validated against amd-triton.
# Tests skip when the version is below 3.7. For deployment under
# community Triton (e.g., to keep vllm's mxfp4 MoE matmul_ogs kernels
# working, which currently hit gfx950's 160KB LDS limit on amd-triton
# 3.7), an AOT precompile path modelled after csrc/cpp_itfs/pa_gluon_aot
# is the recommended follow-up.
#
# Differences from the dense V3.2-style mla_decode_gluon:
#   1. K = V (single 512-d latent). RoPE is applied externally before the
#      kernel, so there is no separate Q_pe / K_pe / KV_PE_OFFSET path —
#      the QK and PV use one MFMA chain over the full 512 dims.
#   2. KV iteration is sparse top-k (gather), not dense block_table. The
#      `Topk_indices [batch, TOPK_PADDED]` tensor holds absolute slot ids
#      in the flat KV pool, with -1 padding for invalid positions.
#   3. Optional learnable per-head `attn_sink` is folded into the softmax
#      denominator (V4 attention sink). Only used when NUM_KV_SPLITS == 1.
#   4. Optional sink-FREE per-(token, head) `lse` output for external
#      two-pool merge (vllm decode helper). Only valid when NUM_KV_SPLITS == 1.
#   5. Lonely-row guard: when all topk in a row are -1 (HCA short-context
#      corner case), L=0 → would 0/0 → NaN; we emit 0 output and lse=-inf.
#   6. Drops the `batch_size in {64,128,256}` assertion. Cudagraph batch
#      sizes (1, 2, 4, ..., 512) are all accepted.
#   7. The wrapper transparently pads `topk_count < 256` up to 256 with -1
#      so the in-kernel `gl.assume(num_iter > 3)` (PIPELINE_STAGES=3) holds
#      for the V4 SWA layer's window_size=128 case.
#
# The 3-stage software pipeline, MFMA layouts, async-copy schedule, and
# stage-2 reduce kernel (`_mla_softmax_reducev_kernel`) are reused from
# `mla_decode_gluon` unchanged. We import the reduce directly.

import torch
import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

import aiter.ops.triton.utils._triton.arch_info as arch_info
from aiter.ops.triton.utils.device_info import get_num_xcds
from aiter.ops.triton.gluon.mla_decode_gluon import _mla_softmax_reducev_kernel


# Module-level cache for the all-(-1) pad buffer used by the wrapper to
# extend topk_count up to 256 for SWA layers. Keyed by (device, dtype, n_pad).
_AITER_GLUON_TOPK_PAD_CACHE: dict = {}


# fmt: off
@gluon.jit
def _sparse_mla_decode_gluon(
    Q,                      # [batch, nhead, HEAD_DIM_CKV] BF16 (RoPE pre-applied)
    Kv_c_cache,             # [N_total, HEAD_DIM_CKV] BF16 (or [N, 1, HEAD_DIM_CKV]; stride taken on row dim)
    Topk_indices,           # [batch, TOPK_PADDED] int32 (-1 padded for invalid positions)
    Attn_sink,              # [nhead] FP32 or null (only read when HAS_ATTN_SINK)
    O,                      # [batch, nhead, NUM_KV_SPLITS, HEAD_DIM_CKV(+1 if NUM_KV_SPLITS>1)] BF16
    Lse,                    # [batch, nhead] FP32 or null (only stored when RETURN_LSE)
    sm_scale,
    stride_q_bs,            # q.stride(0)
    stride_q_h,             # q.stride(1)
    stride_kv_bs,           # kv.stride(-2): bytes from one slot to the next
    stride_topk_bs,         # topk.stride(0): elems from one row to the next
    stride_o_b,
    stride_o_h,
    stride_o_s,
    stride_lse_b,
    stride_lse_h,
    BLOCK_H: gl.constexpr,
    BLOCK_N: gl.constexpr,
    NUM_KV_SPLITS: gl.constexpr,
    HEAD_DIM_CKV: gl.constexpr,
    TOPK_PADDED: gl.constexpr,
    HAS_ATTN_SINK: gl.constexpr,
    RETURN_LSE: gl.constexpr,
    WITHIN_2GB: gl.constexpr,
    NUM_XCDS: gl.constexpr,
):
    cur_batch = gl.program_id(0) + (gl.program_id(2) // NUM_KV_SPLITS) * NUM_XCDS
    cur_head_id = gl.program_id(1)
    split_kv_id = gl.program_id(2) % NUM_KV_SPLITS

    # cur_batch_seq_len is constexpr (TOPK_PADDED); same per-batch shape.
    batch_topk_start = stride_topk_bs * cur_batch

    # layout for Q (matches dense kernel's blocked_q_nope / shared_q_nope)
    blocked_q: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 8],
        threads_per_warp=[1, 64],
        warps_per_cta=[4, 1],
        order=[1, 0],
    )
    shared_q: gl.constexpr = gl.PaddedSharedLayout(
        interval_padding_pairs=[[512, 16]],
        offset_bases=[[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64], [0, 128], [0, 256], [1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0]],
        cga_layout=[],
        shape=[64, 512]
    )
    dtype = Q.type.element_ty
    buf_q = gl.allocate_shared_memory(dtype, shape=[BLOCK_H, HEAD_DIM_CKV], layout=shared_q)

    # layout for K/V (single 512-d latent; reused for both K and V)
    blocked_kv: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=((1, 0), (2, 0), (4, 0), (0, 8), (0, 4), (0, 16), (0, 32)),
        lane_bases=((8, 0), (16, 0), (32, 0), (64, 0), (128, 0), (256, 0)),
        warp_bases=((0, 1), (0, 2)),
        block_bases=[],
        shape=[512, 64],
    )
    shared_kv: gl.constexpr = gl.PaddedSharedLayout(
        interval_padding_pairs=[[512, 16]],
        offset_bases=[[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0], [64, 0], [128, 0], [256, 0], [0, 1], [0, 2], [0, 8], [0, 4], [0, 16], [0, 32]],
        cga_layout=[],
        shape=[512, 64]
    )
    linear_v: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=((0, 1), (0, 2), (0, 4), (0, 32), (16, 0), (32, 0), (64, 0), (128, 0), (256, 0)),
        lane_bases=((1, 0), (2, 0), (4, 0), (8, 0), (0, 8), (0, 16)),
        warp_bases=((0, 0), (0, 0)),
        block_bases=[],
        shape=[512, 64],
    )

    # MFMA layouts
    mfma_layout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=[16, 16, 32],
        transposed=True,
        warps_per_cta=[4, 1],
    )
    mfma_layout_a: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mfma_layout, k_width=8)
    mfma_layout_b: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=mfma_layout, k_width=8)

    # load Q (single 512-d latent; no Q_pe in V4)
    offs_d_ckv = gl.arange(0, HEAD_DIM_CKV, layout=gl.SliceLayout(0, blocked_q))
    cur_head = cur_head_id * BLOCK_H + gl.arange(0, BLOCK_H, layout=gl.SliceLayout(1, blocked_q))
    offs_q = cur_batch * stride_q_bs + cur_head[:, None] * stride_q_h + offs_d_ckv[None, :]
    gl.amd.cdna4.async_copy.buffer_load_to_shared(buf_q, Q, offs_q)
    gl.amd.cdna4.async_copy.commit_group()

    e_max = gl.zeros([BLOCK_H], dtype=gl.float32, layout=gl.SliceLayout(1, mfma_layout)) - float("inf")
    e_sum = gl.zeros([BLOCK_H], dtype=gl.float32, layout=gl.SliceLayout(1, mfma_layout))
    acc = gl.zeros([BLOCK_H, HEAD_DIM_CKV], dtype=gl.float32, layout=mfma_layout)

    # split-KV: each program covers [split_kv_start, split_kv_end) along the
    # topk axis. TOPK_PADDED is constexpr and the wrapper picks NUM_KV_SPLITS
    # so kv_len_per_split is BLOCK_N-aligned and >= 4*BLOCK_N for the pipeline.
    kv_len_per_split: gl.constexpr = TOPK_PADDED // NUM_KV_SPLITS
    split_kv_start = kv_len_per_split * split_kv_id
    split_kv_end = split_kv_start + kv_len_per_split
    num_iter = kv_len_per_split // BLOCK_N
    start_n = split_kv_start

    # bufs of topk indices (was page numbers in dense)
    blocked_page: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=((0,),),
        lane_bases=((1,), (2,), (4,), (8,), (16,), (32,)),
        warp_bases=((0,), (0,)),
        block_bases=[],
        shape=[64],
    )
    shared_page: gl.constexpr = gl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[0])
    bufs_page = gl.allocate_shared_memory(gl.int32, shape=[2, BLOCK_N], layout=shared_page)

    offs_page_raw = gl.arange(0, BLOCK_N, layout=blocked_page)

    ################ prologue
    #### global load topk indices [0]
    offs_n_page = start_n + offs_page_raw
    offs_topk = batch_topk_start + offs_n_page
    gl.amd.cdna4.async_copy.buffer_load_to_shared(bufs_page.index(0), Topk_indices, offs_topk, offs_n_page < split_kv_end)
    gl.amd.cdna4.async_copy.commit_group()

    start_n += BLOCK_N
    #### global load topk indices [1]
    offs_n_page = start_n + offs_page_raw
    offs_topk = batch_topk_start + offs_n_page
    gl.amd.cdna4.async_copy.buffer_load_to_shared(bufs_page.index(1), Topk_indices, offs_topk, offs_n_page < split_kv_end)
    gl.amd.cdna4.async_copy.commit_group()

    #### local load Q
    gl.amd.cdna4.async_copy.wait_group(2)
    q = gl.amd.cdna4.async_copy.load_shared_relaxed(buf_q, mfma_layout_a)

    #################### move here to work around allocate_shared_memory bug
    bufs_kv = gl.allocate_shared_memory(dtype, shape=[2, HEAD_DIM_CKV, BLOCK_N], layout=shared_kv)

    blocked_kv_slice: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=((1, 0), (2, 0), (4, 0), (0, 8), (0, 4), (0, 16)),
        lane_bases=((8, 0), (16, 0), (32, 0), (64, 0), (128, 0), (256, 0)),
        warp_bases=((0, 1), (0, 2)),
        block_bases=[],
        shape=[512, 32],
    )

    #### global load K slice 0 from page[0]
    gl.amd.cdna4.async_copy.wait_group(1)
    bufs_page_0 = bufs_page.index(0).slice(0, 32, 0)
    topk_pos_0 = gl.amd.cdna4.async_copy.load_shared_relaxed(bufs_page_0, gl.SliceLayout(0, blocked_kv_slice))
    valid_0 = topk_pos_0 != -1
    # Clamp -1 -> 0 so the buffer_load address arithmetic is in-bounds (the
    # buffer descriptor masks out-of-bounds reads to 0 anyway under WITHIN_2GB,
    # but clamping avoids issuing wrap-around addresses).
    kv_loc0 = gl.where(valid_0, topk_pos_0, 0)

    offs_d_ckv_10 = gl.arange(0, HEAD_DIM_CKV, layout=gl.SliceLayout(1, blocked_kv_slice))
    offs_k_c0 = kv_loc0[None, :] * stride_kv_bs + offs_d_ckv_10[:, None]
    bufs_kv0 = bufs_kv.index(0).slice(0, 32, 1)
    if WITHIN_2GB:
        gl.amd.cdna4.async_copy.buffer_load_to_shared(bufs_kv0, Kv_c_cache, offs_k_c0, mask=valid_0[None, :])
    else:
        gl.amd.cdna4.async_copy.global_load_to_shared(bufs_kv0, Kv_c_cache + offs_k_c0, mask=valid_0[None, :], other=0)
    gl.amd.cdna4.async_copy.commit_group()

    #### global load K slice 1 from page[0]
    bufs_page_1 = bufs_page.index(0).slice(32, 32, 0)
    topk_pos_1 = gl.amd.cdna4.async_copy.load_shared_relaxed(bufs_page_1, gl.SliceLayout(0, blocked_kv_slice))
    valid_1 = topk_pos_1 != -1
    kv_loc1 = gl.where(valid_1, topk_pos_1, 0)

    bufs_kv1 = bufs_kv.index(0).slice(32, 32, 1)
    offs_k_c1 = kv_loc1[None, :] * stride_kv_bs + offs_d_ckv_10[:, None]
    if WITHIN_2GB:
        gl.amd.cdna4.async_copy.buffer_load_to_shared(bufs_kv1, Kv_c_cache, offs_k_c1, mask=valid_1[None, :])
    else:
        gl.amd.cdna4.async_copy.global_load_to_shared(bufs_kv1, Kv_c_cache + offs_k_c1, mask=valid_1[None, :], other=0)
    gl.amd.cdna4.async_copy.commit_group()

    gl.assume(num_iter > 3)
    buf_idx = 0
    ################ loop
    for i in range(num_iter - 2):
        async_idx = (buf_idx + 1) % 2

        gl.amd.cdna4.async_copy.wait_group(0)

        # IMPORTANT: capture the valid mask for the K we are about to process
        # (which lives in bufs_kv.index(buf_idx) and was loaded from
        # bufs_page.index(buf_idx)) BEFORE the next-future-iter page-load
        # below overwrites bufs_page.index(buf_idx).
        topk_pos_qk = gl.amd.cdna4.async_copy.load_shared_relaxed(
            bufs_page.index(buf_idx), gl.SliceLayout(0, mfma_layout)
        )
        valid_qk = topk_pos_qk != -1

        #### global load topk indices for next iteration
        offs_n_page = start_n + BLOCK_N + offs_page_raw
        offs_topk = batch_topk_start + offs_n_page
        gl.amd.cdna4.async_copy.buffer_load_to_shared(bufs_page.index(buf_idx), Topk_indices, offs_topk, offs_n_page < split_kv_end)
        gl.amd.cdna4.async_copy.commit_group()

        #### global load K slice 0 from page[async_idx]
        bufs_kv0 = bufs_kv.index(async_idx).slice(0, 32, 1)
        bufs_kv1 = bufs_kv.index(async_idx).slice(32, 32, 1)
        bufs_page_0 = bufs_page.index(async_idx).slice(0, 32, 0)
        topk_pos_0 = gl.amd.cdna4.async_copy.load_shared_relaxed(bufs_page_0, gl.SliceLayout(0, blocked_kv_slice))
        valid_0 = topk_pos_0 != -1
        kv_loc0 = gl.where(valid_0, topk_pos_0, 0)
        offs_n_nope0 = start_n + gl.arange(0, 32, layout=gl.SliceLayout(0, blocked_kv_slice))
        offs_k_c0 = kv_loc0[None, :] * stride_kv_bs + offs_d_ckv_10[:, None]
        if WITHIN_2GB:
            gl.amd.cdna4.async_copy.buffer_load_to_shared(bufs_kv0, Kv_c_cache, offs_k_c0, mask=valid_0[None, :])
        else:
            gl.amd.cdna4.async_copy.global_load_to_shared(bufs_kv0, Kv_c_cache + offs_k_c0, mask=valid_0[None, :], other=0)
        gl.amd.cdna4.async_copy.commit_group()

        #### dot, softmax, dot (part0)
        k_c = gl.amd.cdna4.async_copy.load_shared_relaxed(bufs_kv.index(buf_idx), mfma_layout_b)
        zeros = gl.zeros([BLOCK_H, BLOCK_N], dtype=gl.float32, layout=mfma_layout)
        qk = gl.amd.cdna4.mfma(q, k_c.to(q.dtype), zeros)

        # global load K slice 1 from page[async_idx]
        bufs_page_1 = bufs_page.index(async_idx).slice(32, 32, 0)
        topk_pos_1 = gl.amd.cdna4.async_copy.load_shared_relaxed(bufs_page_1, gl.SliceLayout(0, blocked_kv_slice))
        valid_1 = topk_pos_1 != -1
        kv_loc1 = gl.where(valid_1, topk_pos_1, 0)
        offs_n1 = offs_n_nope0 + 32
        offs_k_c1 = kv_loc1[None, :] * stride_kv_bs + offs_d_ckv_10[:, None]
        if WITHIN_2GB:
            gl.amd.cdna4.async_copy.buffer_load_to_shared(bufs_kv1, Kv_c_cache, offs_k_c1, mask=valid_1[None, :])
        else:
            gl.amd.cdna4.async_copy.global_load_to_shared(bufs_kv1, Kv_c_cache + offs_k_c1, mask=valid_1[None, :], other=0)
        gl.amd.cdna4.async_copy.commit_group()

        #### dot, softmax, dot (part1)
        # `valid_qk` was captured at the top of this iteration before
        # the new-page async_copy clobbered bufs_page.index(buf_idx).
        qk *= sm_scale
        qk = gl.where(valid_qk[None, :], qk, float("-inf"))
        n_e_max = gl.maximum(gl.max(qk, 1), e_max)
        # M-clamp: if entire tile is invalid, n_e_max stays -inf; clamp to 0
        # so subsequent exp(S - m_j) doesn't produce NaN. Mirrors
        # unified_attention_sparse_mla.py's `where(m_j > -inf, m_j, 0.0)`.
        n_e_max = gl.where(n_e_max > float("-inf"), n_e_max, 0.0)
        LOG2E: gl.constexpr = 1.4426950408889634
        re_scale = gl.exp2((e_max - n_e_max) * LOG2E)
        p = gl.exp2((qk - n_e_max[:, None]) * LOG2E)
        e_sum = e_sum * re_scale + gl.sum(p, 1)
        e_max = n_e_max
        p = p.to(dtype)
        p = gl.convert_layout(p, mfma_layout_a)
        acc *= re_scale[:, None]
        v_c = gl.amd.cdna4.async_copy.load_shared_relaxed(bufs_kv.index(buf_idx), linear_v)
        v_c = gl.permute(v_c, [1, 0])
        v_c = gl.convert_layout(v_c, mfma_layout_b)
        acc = gl.amd.cdna4.mfma(p, v_c, acc)

        start_n += BLOCK_N
        buf_idx = (buf_idx + 1) % 2

    ################ epilogue
    async_idx = (buf_idx + 1) % 2

    #### global load K from full page (final iteration -1)
    gl.amd.cdna4.async_copy.wait_group(3)
    topk_pos = gl.amd.cdna4.async_copy.load_shared_relaxed(bufs_page.index(async_idx), gl.SliceLayout(0, blocked_kv))
    valid_full = topk_pos != -1
    kv_loc = gl.where(valid_full, topk_pos, 0)
    offs_n_full = start_n + gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, blocked_kv))
    offs_d_ckv_1 = gl.arange(0, HEAD_DIM_CKV, layout=gl.SliceLayout(1, blocked_kv))
    offs_k_c = kv_loc[None, :] * stride_kv_bs + offs_d_ckv_1[:, None]
    if WITHIN_2GB:
        gl.amd.cdna4.async_copy.buffer_load_to_shared(bufs_kv.index(async_idx), Kv_c_cache, offs_k_c, mask=valid_full[None, :])
    else:
        gl.amd.cdna4.async_copy.global_load_to_shared(bufs_kv.index(async_idx), Kv_c_cache + offs_k_c, mask=valid_full[None, :], other=0)
    gl.amd.cdna4.async_copy.commit_group()

    # dot, softmax, dot (epilogue tile 1)
    gl.amd.cdna4.async_copy.wait_group(2)
    k_c = bufs_kv.index(buf_idx).load(layout=mfma_layout_b)
    zeros = gl.zeros([BLOCK_H, BLOCK_N], dtype=gl.float32, layout=mfma_layout)
    qk = gl.amd.cdna4.mfma(q, k_c.to(q.dtype), zeros)

    topk_pos_qk = gl.amd.cdna4.async_copy.load_shared_relaxed(
        bufs_page.index(buf_idx), gl.SliceLayout(0, mfma_layout)
    )
    valid_qk = topk_pos_qk != -1
    qk *= sm_scale
    qk = gl.where(valid_qk[None, :], qk, float("-inf"))
    n_e_max = gl.maximum(gl.max(qk, 1), e_max)
    n_e_max = gl.where(n_e_max > float("-inf"), n_e_max, 0.0)
    LOG2E: gl.constexpr = 1.4426950408889634
    re_scale = gl.exp2((e_max - n_e_max) * LOG2E)
    p = gl.exp2((qk - n_e_max[:, None]) * LOG2E)
    e_sum = e_sum * re_scale + gl.sum(p, 1)
    e_max = n_e_max
    p = p.to(dtype)
    p = gl.convert_layout(p, mfma_layout_a)
    acc *= re_scale[:, None]
    v_c = bufs_kv.index(buf_idx).load(layout=linear_v)
    v_c = gl.permute(v_c, [1, 0])
    v_c = gl.convert_layout(v_c, mfma_layout_b)
    acc = gl.amd.cdna4.mfma(p, v_c, acc)

    start_n += BLOCK_N
    buf_idx = (buf_idx + 1) % 2

    #### dot, softmax, dot (epilogue tile 2)
    gl.amd.cdna4.async_copy.wait_group(0)
    k_c = bufs_kv.index(buf_idx).load(layout=mfma_layout_b)
    zeros = gl.zeros([BLOCK_H, BLOCK_N], dtype=gl.float32, layout=mfma_layout)
    qk = gl.amd.cdna4.mfma(q, k_c.to(q.dtype), zeros)

    topk_pos_qk = gl.amd.cdna4.async_copy.load_shared_relaxed(
        bufs_page.index(buf_idx), gl.SliceLayout(0, mfma_layout)
    )
    valid_qk = topk_pos_qk != -1
    qk *= sm_scale
    qk = gl.where(valid_qk[None, :], qk, float("-inf"))
    n_e_max = gl.maximum(gl.max(qk, 1), e_max)
    n_e_max = gl.where(n_e_max > float("-inf"), n_e_max, 0.0)
    re_scale = gl.exp2((e_max - n_e_max) * LOG2E)
    p = gl.exp2((qk - n_e_max[:, None]) * LOG2E)
    e_sum = e_sum * re_scale + gl.sum(p, 1)
    e_max = n_e_max
    p = p.to(dtype)
    p = gl.convert_layout(p, mfma_layout_a)
    acc *= re_scale[:, None]
    v_c = bufs_kv.index(buf_idx).load(layout=linear_v)
    v_c = gl.permute(v_c, [1, 0])
    v_c = gl.convert_layout(v_c, mfma_layout_b)
    acc = gl.amd.cdna4.mfma(p, v_c, acc)

    cur_head_o = cur_head_id * BLOCK_H + gl.arange(0, BLOCK_H, layout=gl.SliceLayout(1, mfma_layout))
    offs_d_ckv_o = gl.arange(0, HEAD_DIM_CKV, layout=gl.SliceLayout(0, mfma_layout))
    offs_o = cur_batch * stride_o_b + cur_head_o[:, None] * stride_o_h + split_kv_id * stride_o_s + offs_d_ckv_o[None, :]

    # Capture "was lonely" state BEFORE sink fold modifies e_sum. When all
    # topk for a row are -1 (HCA short-context corner case), e_sum=0 here,
    # and we want both:
    #   - the (sink-FREE) lse to be -inf
    #   - the final output to be 0 (the only thing in softmax is the sink
    #     virtual token, whose value=0)
    was_lonely = e_sum == 0.0

    # Sink-FREE lse: emit BEFORE sink folds into e_sum.
    if RETURN_LSE:
        safe_e_sum_sink_free = gl.where(was_lonely, 1.0, e_sum)
        lse_val = gl.where(
            was_lonely,
            float("-inf"),
            e_max + gl.log(safe_e_sum_sink_free),
        )
        offs_lse = cur_batch * stride_lse_b + cur_head_o * stride_lse_h
        gl.amd.cdna4.buffer_store(lse_val, ptr=Lse, offsets=offs_lse)

    # Optional in-kernel sink fold. Wrapper forces NUM_KV_SPLITS=1 whenever
    # HAS_ATTN_SINK is on, since the stage-2 reduce kernel doesn't know
    # about sink.
    if HAS_ATTN_SINK:
        # Sink contributes exp(sink_h - e_max) to the denominator only.
        # `cur_head_o` has SliceLayout(1, mfma_layout); pointer arithmetic
        # preserves its layout so `sink_h` matches `e_max`.
        sink_h = gl.load(Attn_sink + cur_head_o)
        sink_contrib = gl.exp2((sink_h - e_max) * LOG2E)
        # Skip sink contribution on lonely rows: their final output is
        # forced to 0 below (sink token has value=0, so it contributes
        # nothing to the numerator).
        sink_contrib = gl.where(was_lonely, 0.0, sink_contrib)
        e_sum = e_sum + sink_contrib

    # Lonely-row guard: produce 0 output for rows that started lonely
    # (sink alone, with value=0, produces 0 anyway; this also keeps the
    # NUM_KV_SPLITS>1 split-output path NaN-free).
    safe_e_sum = gl.where(e_sum == 0.0, 1.0, e_sum)
    rcp = 1.0 / safe_e_sum
    raw = (acc * rcp[:, None]).to(dtype)
    zero_out = gl.zeros([BLOCK_H, HEAD_DIM_CKV], dtype=dtype, layout=mfma_layout)
    stored_value = gl.where(was_lonely[:, None], zero_out, raw)
    gl.amd.cdna4.buffer_store(stored_value, ptr=O, offsets=offs_o)

    if NUM_KV_SPLITS > 1:
        # Per-split lse for stage-2 reduce (stored at the trailing slot).
        # For lonely rows we store a very-negative finite lse instead of
        # -inf, so the reduce's `exp(lse - n_e_max)` produces 0 (not NaN
        # via -inf - -inf).
        offs_o_lse = cur_batch * stride_o_b + cur_head_o * stride_o_h + split_kv_id * stride_o_s + HEAD_DIM_CKV
        lse_for_reduce = gl.where(
            was_lonely,
            -1.0e30,  # very negative, finite, so reduce arithmetic stays defined
            e_max + gl.log(safe_e_sum),
        )
        gl.amd.cdna4.buffer_store(lse_for_reduce.to(dtype), ptr=O, offsets=offs_o_lse)
# fmt: on


def _get_topk_pad_buffer(batch: int, n_pad: int, device: torch.device) -> torch.Tensor:
    """Cached all-(-1) buffer used to pad SWA-layer topk_count up to 256.
    Cudagraph-friendly: same address across replays per (device, batch, n_pad)."""
    key = (device.type, device.index, batch, n_pad)
    cached = _AITER_GLUON_TOPK_PAD_CACHE.get(key)
    if cached is None:
        cached = torch.full(
            (batch, n_pad), -1, dtype=torch.int32, device=device
        )
        _AITER_GLUON_TOPK_PAD_CACHE[key] = cached
    return cached


def sparse_mla_decode_gluon(
    q: torch.Tensor,                # [batch, nhead, head_dim_ckv] BF16
    kv: torch.Tensor,               # [N_total, 1, head_dim_ckv] OR [N_total, head_dim_ckv] BF16
    topk_indices: torch.Tensor,     # [batch, topk_count] int32 (-1 padded)
    o: torch.Tensor,                # [batch, nhead, head_dim_ckv] BF16
    sm_scale: float,
    attn_sink: torch.Tensor | None = None,  # [nhead] FP32 or None
    return_lse: bool = False,
    lse_out: torch.Tensor | None = None,    # [batch, nhead] FP32 or None
):
    """Sparse MLA decode for DeepSeek-V4 on MI355X (gfx950).

    K = V (single 512-d latent; RoPE pre-applied externally).
    Sparse top-k gather over `topk_indices` (-1 entries are masked).
    Optional per-head `attn_sink` and sink-FREE per-(token, head) `lse`
    output for the vllm two-pool merge path.

    Returns: (attn_logits, lse_out)
      attn_logits: when NUM_KV_SPLITS == 1, the same tensor as `o`. When
                   NUM_KV_SPLITS > 1, a temp buffer used by stage-2 reduce.
      lse_out: the lse tensor (only populated when return_lse=True).
    """
    assert (
        arch_info.get_arch() == "gfx950"
    ), f"sparse_mla_decode_gluon requires gfx950 (CDNA4), got {arch_info.get_arch()}"

    batch_size, nhead, head_dim_ckv = q.shape
    assert nhead in (
        64,
        128,
    ), f"sparse_mla_decode_gluon requires nhead in (64, 128), got {nhead}"
    assert (
        head_dim_ckv == 512
    ), f"sparse_mla_decode_gluon requires head_dim_ckv=512, got {head_dim_ckv}"
    assert q.dtype == torch.bfloat16, f"q must be bf16, got {q.dtype}"
    assert kv.dtype == torch.bfloat16, f"kv must be bf16, got {kv.dtype}"
    assert topk_indices.dtype == torch.int32, (
        f"topk_indices must be int32, got {topk_indices.dtype}"
    )
    assert topk_indices.shape[0] == batch_size, (
        f"topk_indices batch {topk_indices.shape[0]} != q batch {batch_size}"
    )
    if attn_sink is not None:
        assert attn_sink.dtype == torch.float32, (
            f"attn_sink must be fp32, got {attn_sink.dtype}"
        )
        assert attn_sink.shape == (nhead,), (
            f"attn_sink must be [nhead]={nhead}, got {tuple(attn_sink.shape)}"
        )

    PAGE_SIZE = 1
    BLOCK_H = 64
    BLOCK_N = 64
    NUM_XCDS = get_num_xcds()
    PIPELINE_STAGES = 3
    MIN_KV_PER_SPLIT = (PIPELINE_STAGES + 1) * BLOCK_N  # require num_iter > 3

    topk_count = topk_indices.shape[1]
    # Transparent pad: V4 SWA layers have topk_count=128, but the in-kernel
    # `gl.assume(num_iter > 3)` (PIPELINE_STAGES=3) requires
    # NUM_KV_SPLITS == 1 AND topk_count >= 4 * BLOCK_N == 256. For the SWA
    # case we cat a cached all-(-1) buffer to extend to 256.
    if topk_count < MIN_KV_PER_SPLIT:
        n_pad = MIN_KV_PER_SPLIT - topk_count
        pad = _get_topk_pad_buffer(batch_size, n_pad, topk_indices.device)
        topk_indices = torch.cat([topk_indices, pad], dim=1)
        topk_count = topk_indices.shape[1]
    assert topk_count % BLOCK_N == 0, (
        f"topk_count={topk_count} must be a multiple of BLOCK_N={BLOCK_N}; "
        f"if your caller emits unaligned topk, round up by padding with -1"
    )

    # NUM_KV_SPLITS picker.
    #
    # `gl.assume(num_iter > 3)` requires each split to have > PIPELINE_STAGES
    # iterations, but the 3-stage software pipeline only earns back its
    # startup+drain overhead in two regimes (calibrated by sweep on V4
    # shapes batch ∈ {1..32}, topk ∈ {256, 512, 1024}, MI355X):
    #
    #   regime A: num_iter_total <  16  (covers SWA topk=128→256 pad, HCA topk=512)
    #             → splits=1.  Each split would be too short, the launch +
    #               Q-load + stage-2 reduce overhead dominates.  Empirically
    #               splits=2 was ~20% slower for topk=512.
    #   regime B: num_iter_total >= 16  (CSA topk=1024)
    #             → splits up to min(target, num_iter_total//4, 4).
    #               4 iters per split is enough to amortize, and we cap at 4
    #               so the stage-2 reduce kernel doesn't dominate.
    #
    # When attn_sink is in-kernel or return_lse is requested, force splits=1
    # (the stage-2 reduce kernel doesn't fold sink and doesn't emit lse).
    SPLIT_BENEFIT_TOTAL_ITER_FLOOR = 16
    SPLIT_MIN_ITER_PER_SPLIT = 4
    num_iter_total = topk_count // BLOCK_N
    if attn_sink is not None or return_lse or num_iter_total < SPLIT_BENEFIT_TOTAL_ITER_FLOOR:
        NUM_KV_SPLITS = 1
    else:
        # Aim for ~256 workgroups (one wave on MI350) but cap so each split
        # still has >= SPLIT_MIN_ITER_PER_SPLIT iterations.
        TARGET_GRID = 256
        max_xcds = max(1, NUM_XCDS)
        base_grid = max_xcds * triton.cdiv(nhead, BLOCK_H) * max(1, batch_size // max_xcds)
        target = max(1, triton.next_power_of_2(triton.cdiv(TARGET_GRID, base_grid)))
        max_by_iter = max(1, num_iter_total // SPLIT_MIN_ITER_PER_SPLIT)
        # Round down to power of 2 <= max_by_iter
        max_by_iter_p2 = 1
        while max_by_iter_p2 * 2 <= max_by_iter:
            max_by_iter_p2 *= 2
        NUM_KV_SPLITS = max(1, min(target, max_by_iter_p2, 4))

    if NUM_KV_SPLITS == 1:
        # Fast path: stage-1 writes directly to `o`.
        attn_logits = o.view(batch_size, nhead, NUM_KV_SPLITS, head_dim_ckv)
    else:
        attn_logits = torch.empty(
            (batch_size, nhead, NUM_KV_SPLITS, head_dim_ckv + 1),
            dtype=o.dtype,
            device=o.device,
        )

    if return_lse and lse_out is None:
        lse_out = torch.empty(
            (batch_size, nhead), dtype=torch.float32, device=q.device
        )
    if lse_out is not None:
        stride_lse_b = lse_out.stride(0)
        stride_lse_h = lse_out.stride(1)
    else:
        stride_lse_b = 0
        stride_lse_h = 0

    # Grid: handle batch < NUM_XCDS by collapsing the XCDS dim.
    if batch_size >= NUM_XCDS:
        eff_xcds = NUM_XCDS
    else:
        eff_xcds = 1
    grid = (
        eff_xcds,
        triton.cdiv(nhead, BLOCK_H),
        max(1, batch_size // eff_xcds) * NUM_KV_SPLITS,
    )

    # Buffer-load 32-bit offset addressability check.
    max_kv_bytes = kv.shape[0] * kv.stride(-2) * kv.element_size()
    within_2gb = max_kv_bytes <= 0x80000000

    # `kv` may arrive as [N, 1, d] or [N, d]; we always pass the row stride
    # via stride(-2) so both work.
    _sparse_mla_decode_gluon[grid](
        q,
        kv,
        topk_indices,
        attn_sink if attn_sink is not None else q,  # ptr unused when HAS_ATTN_SINK is False
        attn_logits,
        lse_out if lse_out is not None else q,      # ptr unused when RETURN_LSE is False
        sm_scale,
        q.stride(0),
        q.stride(1),
        kv.stride(-2),
        topk_indices.stride(0),
        attn_logits.stride(0),
        attn_logits.stride(1),
        attn_logits.stride(2),
        stride_lse_b,
        stride_lse_h,
        BLOCK_H=BLOCK_H,
        BLOCK_N=BLOCK_N,
        NUM_KV_SPLITS=NUM_KV_SPLITS,
        HEAD_DIM_CKV=head_dim_ckv,
        TOPK_PADDED=topk_count,
        HAS_ATTN_SINK=attn_sink is not None,
        RETURN_LSE=lse_out is not None,
        WITHIN_2GB=within_2gb,
        NUM_XCDS=eff_xcds,
    )

    if NUM_KV_SPLITS > 1:
        # Stage-2 reduce — only valid when no in-kernel sink and no
        # external-merge lse (wrapper guarantees both are off here).
        # `B_seq_len` is unused under ALL_SPLITS_NONEMPTY=True; pass a
        # 1-elem dummy.
        dummy_seq = torch.empty(1, dtype=torch.int32, device=q.device)
        grid_reduce = (batch_size, nhead)
        _mla_softmax_reducev_kernel[grid_reduce](
            attn_logits,
            dummy_seq,
            o,
            attn_logits.stride(0),
            attn_logits.stride(1),
            attn_logits.stride(2),
            o.stride(0),
            o.stride(1),
            NUM_KV_SPLITS=NUM_KV_SPLITS,
            HEAD_DIM_CKV=head_dim_ckv,
            USE_2D_VIEW=True,
            ALL_SPLITS_NONEMPTY=True,
            num_warps=8,
        )

    return attn_logits, lse_out
