# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

# Gluon MLA decode kernel originated from FlashMLA triton kernel(https://github.com/deepseek-ai/FlashMLA/blob/main/benchmark/bench_flash_mla.py).
# Stage-1 split-KV MLA attention using explicit Gluon layouts. When
# NUM_KV_SPLITS=1 the kernel writes the final attention output to O directly
# and no reduce is launched; when NUM_KV_SPLITS>1 each program writes per-split
# (acc, lse) to a temp buffer and stage-2 (_mla_softmax_reducev_kernel) combines
# them. NUM_KV_SPLITS is auto-picked in the wrapper so the launch grid fills
# ~256 workgroups (one wave on MI350).
# Constraints: CDNA4 (gfx950), bf16, PAGE_SIZE=1, BLOCK_H=64, BLOCK_N=64,
#   head_dim_ckv=512, head_dim_kpe=64,
#   nhead in {64, 128}, batch_size in {64, 128, 256},
#   min_kv_seq_len > NUM_KV_SPLITS * (PIPELINE_STAGES*BLOCK_N + NUM_KV_SPLITS)
#   where PIPELINE_STAGES=3 (sufficient to keep per-split num_iter > 3;
#   see wrapper assert).
#
# 3-stage software pipeline (double-buffered, BLOCK_N=64 with 2x32 KV slices):
#   AC = async_copy (global->LDS), LL = load (LDS->reg), P = page, K = K-cache, V = V-cache
#
#                      iter i          iter i+1        iter i+2
#   ACP(page):        [i+2]           [i+3]           [i+4]
#   LLP+ACK(K):       [i+1]           [i+2]           [i+3]
#   LLK+MFMA+LLV:     [i]             [i+1]           [i+2]
#
#   Within each loop iteration (operating on buf_idx=current, async_idx=next):
#     ACP                                -- async_copy page numbers [i+2]
#     LLP, ACK                           -- local_load pages [i+1], async_copy K/KPE [i+1]
#     LLK, MFMA0, softmax, LLV, MFMA1   -- compute on [i]: QK dot, softmax, PV dot

import torch
import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

import aiter.ops.triton.utils._triton.arch_info as arch_info
from aiter.ops.triton.utils.device_info import get_num_xcds

# fmt: off
@gluon.jit
def _mla_decode_gluon(
    Q_nope,
    Q_pe,
    Kv_c_cache,
    K_pe_cache,
    Req_to_tokens,
    B_seq_len,
    O,  # noqa: E741
    sm_scale,
    stride_q_nope_bs,
    stride_q_nope_h,
    stride_q_pe_bs,
    stride_q_pe_h,
    stride_kv_c_bs,
    stride_k_pe_bs,
    stride_req_to_tokens_bs,
    stride_o_b,
    stride_o_h,
    stride_o_s,
    BLOCK_H: gl.constexpr,
    BLOCK_N: gl.constexpr,
    NUM_KV_SPLITS: gl.constexpr,
    PAGE_SIZE: gl.constexpr,
    HEAD_DIM_CKV: gl.constexpr,
    HEAD_DIM_KPE: gl.constexpr,
    KV_PE_OFFSET: gl.constexpr,
    USE_2D_VIEW: gl.constexpr,
    WITHIN_2GB: gl.constexpr,
    NUM_XCDS: gl.constexpr,
):
    cur_batch = gl.program_id(0) + (gl.program_id(2) // NUM_KV_SPLITS) * NUM_XCDS
    cur_head_id = gl.program_id(1)
    split_kv_id = gl.program_id(2) % NUM_KV_SPLITS

    # USE_2D_VIEW=True: fixed len or max padded VarLen
    # Req_to_tokens = block_table[batch, max_seqlen], B_seq_len = cache_seqlens[batch]
    # USE_2D_VIEW=False: flattened VarLen
    # Req_to_tokens = kv_indices[total_kv],           B_seq_len = kv_indptr[batch+1]
    if USE_2D_VIEW:
        batch_page_start = stride_req_to_tokens_bs * cur_batch
        cur_batch_seq_len = gl.load(B_seq_len + cur_batch)
    else:
        batch_page_start = gl.load(B_seq_len + cur_batch)
        cur_batch_seq_len = gl.load(B_seq_len + cur_batch + 1) - batch_page_start

    # layout for Q
    # 64x512
    blocked_q_nope: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 8],
        threads_per_warp=[1, 64],
        warps_per_cta=[4, 1],
        order=[1, 0],
    )
    shared_q_nope: gl.constexpr = gl.PaddedSharedLayout(
        interval_padding_pairs=[[512, 16]],
        offset_bases=[[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64], [0, 128], [0, 256], [1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0]],
        cga_layout=[],
        shape=[64, 512]
    )
    # 64x64
    blocked_q_pe: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=((0, 1), (0, 2), (0, 4), (32, 0)),
        lane_bases=((0, 8), (0, 16), (0, 32), (4, 0), (8, 0), (16, 0)),
        warp_bases=((1, 0), (2, 0)),
        block_bases=[],
        shape=[64, 64],
    )
    shared_q_pe: gl.constexpr = gl.PaddedSharedLayout(
        interval_padding_pairs=[[512, 16]],
        offset_bases=[[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [4, 0], [8, 0], [16, 0], [1, 0], [2, 0], [32, 0]],
        cga_layout=[],
        shape=[64, 64]
    )
    dtype = Q_nope.type.element_ty
    buf_q_nope = gl.allocate_shared_memory(dtype, shape=[BLOCK_H, HEAD_DIM_CKV], layout=shared_q_nope)
    buf_q_pe = gl.allocate_shared_memory(dtype, shape=[BLOCK_H, HEAD_DIM_KPE], layout=shared_q_pe)

    # layout for KV
    # 512x64
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

    # 64x64
    blocked_kpe: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=((1, 0), (2, 0), (4, 0), (0, 32)),
        lane_bases=((8, 0), (16, 0), (32, 0), (0, 4), (0, 8), (0, 16)),
        warp_bases=((0, 1), (0, 2)),
        block_bases=[],
        shape=[64, 64],
    )
    shared_kpe: gl.constexpr = gl.PaddedSharedLayout(
        interval_padding_pairs=[[512, 16]],
        offset_bases=[[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0], [0, 4], [0, 8], [0, 16], [0, 1], [0, 2], [0, 32]],
        cga_layout=[],
        shape=[64, 64]
    )
    linear_v: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=((0, 1), (0, 2), (0, 4), (0, 32), (16, 0), (32, 0), (64, 0), (128, 0), (256, 0)),
        lane_bases=((1, 0), (2, 0), (4, 0), (8, 0), (0, 8), (0, 16)),
        warp_bases=((0, 0), (0, 0)),
        block_bases=[],
        shape=[512, 64],
    )

    # layout for mfma
    mfma_layout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=[16, 16, 32],
        transposed=True,
        warps_per_cta=[4, 1],
    )
    mfma_layout_a: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mfma_layout, k_width=8)
    mfma_layout_b: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=mfma_layout, k_width=8)

    # load q_nope
    offs_d_ckv = gl.arange(0, HEAD_DIM_CKV, layout=gl.SliceLayout(0, blocked_q_nope))
    cur_head = cur_head_id * BLOCK_H + gl.arange(0, BLOCK_H, layout=gl.SliceLayout(1, blocked_q_nope))
    offs_q_nope = cur_batch * stride_q_nope_bs + cur_head[:, None] * stride_q_nope_h + offs_d_ckv[None, :]
    gl.amd.cdna4.async_copy.buffer_load_to_shared(buf_q_nope, Q_nope, offs_q_nope)
    gl.amd.cdna4.async_copy.commit_group()

    # load q_pe
    offs_d_kpe = gl.arange(0, HEAD_DIM_KPE, layout=gl.SliceLayout(0, blocked_q_pe))
    cur_head_qpe = cur_head_id * BLOCK_H + gl.arange(0, BLOCK_H, layout=gl.SliceLayout(1, blocked_q_pe))
    offs_q_pe = cur_batch * stride_q_pe_bs + cur_head_qpe[:, None] * stride_q_pe_h + offs_d_kpe[None, :]
    gl.amd.cdna4.async_copy.buffer_load_to_shared(buf_q_pe, Q_pe, offs_q_pe)
    gl.amd.cdna4.async_copy.commit_group()

    e_max = gl.zeros([BLOCK_H], dtype=gl.float32, layout=gl.SliceLayout(1, mfma_layout)) - float("inf")
    e_sum = gl.zeros([BLOCK_H], dtype=gl.float32, layout=gl.SliceLayout(1, mfma_layout))
    acc = gl.zeros([BLOCK_H, HEAD_DIM_CKV], dtype=gl.float32, layout=mfma_layout)

    # split-KV: each program covers [split_kv_start, split_kv_end).
    kv_len_per_split = gl.cdiv(cur_batch_seq_len, NUM_KV_SPLITS)
    split_kv_start = kv_len_per_split * split_kv_id
    split_kv_end = gl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)
    num_iter = gl.cdiv(split_kv_end - split_kv_start, BLOCK_N)
    start_n = split_kv_start

    ### bufs of page_number
    blocked_page: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=((0,),),
        lane_bases=((1,), (2,), (4,), (8,), (16,), (32,)),
        warp_bases=((0,), (0,)),
        block_bases=[],
        shape=[64],
    )
    shared_page: gl.constexpr = gl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[0])
    bufs_page = gl.allocate_shared_memory(gl.int32, shape=[2, BLOCK_N], layout=shared_page)
    gl.static_assert(PAGE_SIZE == 1)

    offs_page_raw = gl.arange(0, BLOCK_N, layout=blocked_page)

    ################ prologue
    #### global load page number
    offs_n_page = start_n + offs_page_raw
    offs_page = batch_page_start + offs_n_page // PAGE_SIZE
    gl.amd.cdna4.async_copy.buffer_load_to_shared(bufs_page.index(0), Req_to_tokens, offs_page, offs_n_page < split_kv_end)
    gl.amd.cdna4.async_copy.commit_group()

    start_n += BLOCK_N
    #### global load page number
    offs_n_page = start_n + offs_page_raw
    offs_page = batch_page_start + offs_n_page // PAGE_SIZE
    gl.amd.cdna4.async_copy.buffer_load_to_shared(bufs_page.index(1), Req_to_tokens, offs_page, offs_n_page < split_kv_end)
    gl.amd.cdna4.async_copy.commit_group()

    #### local load Q
    gl.amd.cdna4.async_copy.wait_group(2)
    q_nope = gl.amd.cdna4.async_copy.load_shared_relaxed(buf_q_nope, mfma_layout_a)
    q_pe = gl.amd.cdna4.async_copy.load_shared_relaxed(buf_q_pe, mfma_layout_a)

    #################### move here to work around allocate_shared_memory bug
    bufs_kv = gl.allocate_shared_memory(dtype, shape=[2, HEAD_DIM_CKV, BLOCK_N], layout=shared_kv)
    bufs_kpe = gl.allocate_shared_memory(dtype, shape=[2, HEAD_DIM_KPE, BLOCK_N], layout=shared_kpe)

    blocked_kv_slice: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=((1, 0), (2, 0), (4, 0), (0, 8), (0, 4), (0, 16)),
        lane_bases=((8, 0), (16, 0), (32, 0), (64, 0), (128, 0), (256, 0)),
        warp_bases=((0, 1), (0, 2)),
        block_bases=[],
        shape=[512, 32],
    )

    #### global load K
    # local load page number
    gl.amd.cdna4.async_copy.wait_group(1)
    kv_page_number_pe = gl.amd.cdna4.async_copy.load_shared_relaxed(bufs_page.index(0), gl.SliceLayout(0, blocked_kpe))
    # simplify for page_size 1
    kv_loc_pe = kv_page_number_pe

    # local load page number for slice 0
    bufs_page_0 = bufs_page.index(0).slice(0, 32, 0)
    kv_page_number_0 = gl.amd.cdna4.async_copy.load_shared_relaxed(bufs_page_0, gl.SliceLayout(0, blocked_kv_slice))
    kv_loc0 = kv_page_number_0

    # global load K_nope slice 0
    offs_d_ckv_10 = gl.arange(0, HEAD_DIM_CKV, layout=gl.SliceLayout(1, blocked_kv_slice))
    offs_k_c0 = kv_loc0[None, :] * stride_kv_c_bs + offs_d_ckv_10[:, None]
    bufs_kv0 = bufs_kv.index(0).slice(0, 32, 1)
    if WITHIN_2GB:
        gl.amd.cdna4.async_copy.buffer_load_to_shared(bufs_kv0, Kv_c_cache, offs_k_c0)
    else:
        gl.amd.cdna4.async_copy.global_load_to_shared(bufs_kv0, Kv_c_cache + offs_k_c0)
    gl.amd.cdna4.async_copy.commit_group()

    # global load K_pe
    offs_d_kpe_1 = gl.arange(0, HEAD_DIM_KPE, layout=gl.SliceLayout(1, blocked_kpe))
    offs_k_pe = kv_loc_pe[None, :] * stride_k_pe_bs + offs_d_kpe_1[:, None] + KV_PE_OFFSET
    if WITHIN_2GB:
        gl.amd.cdna4.async_copy.buffer_load_to_shared(bufs_kpe.index(0), K_pe_cache, offs_k_pe)
    else:
        gl.amd.cdna4.async_copy.global_load_to_shared(bufs_kpe.index(0), K_pe_cache + offs_k_pe)
    gl.amd.cdna4.async_copy.commit_group()

    # local load page number for slice 1
    bufs_page_1 = bufs_page.index(0).slice(32, 32, 0)
    kv_page_number_1 = gl.amd.cdna4.async_copy.load_shared_relaxed(bufs_page_1, gl.SliceLayout(0, blocked_kv_slice))
    kv_loc1 = kv_page_number_1

    # global load K_nope slice 1
    bufs_kv1 = bufs_kv.index(0).slice(32, 32, 1)
    offs_k_c1 = kv_loc1[None, :] * stride_kv_c_bs + offs_d_ckv_10[:, None]
    if WITHIN_2GB:
        gl.amd.cdna4.async_copy.buffer_load_to_shared(bufs_kv1, Kv_c_cache, offs_k_c1)
    else:
        gl.amd.cdna4.async_copy.global_load_to_shared(bufs_kv1, Kv_c_cache + offs_k_c1)
    gl.amd.cdna4.async_copy.commit_group()

    gl.assume(num_iter > 3)
    buf_idx = 0
    ################ loop
    for i in range(num_iter - 2):
        async_idx = (buf_idx + 1) % 2

        gl.amd.cdna4.async_copy.wait_group(0)
        #### global load page number
        offs_n_page = start_n + BLOCK_N + offs_page_raw
        offs_page = batch_page_start + offs_n_page // PAGE_SIZE
        gl.amd.cdna4.async_copy.buffer_load_to_shared(bufs_page.index(buf_idx), Req_to_tokens, offs_page, offs_n_page < split_kv_end)
        gl.amd.cdna4.async_copy.commit_group()

        #### global load K
        bufs_kv0 = bufs_kv.index(async_idx).slice(0, 32, 1)
        bufs_kv1 = bufs_kv.index(async_idx).slice(32, 32, 1)
        # local load page number for slice 0
        bufs_page_0 = bufs_page.index(async_idx).slice(0, 32, 0)
        kv_page_number_0 = gl.amd.cdna4.async_copy.load_shared_relaxed(bufs_page_0, gl.SliceLayout(0, blocked_kv_slice))
        kv_loc0 = kv_page_number_0
        # global load K_nope slice 0
        offs_n_nope0 = start_n + gl.arange(0, 32, layout=gl.SliceLayout(0, blocked_kv_slice))
        offs_d_ckv_10 = gl.arange(0, HEAD_DIM_CKV, layout=gl.SliceLayout(1, blocked_kv_slice))
        offs_k_c0 = kv_loc0[None, :] * stride_kv_c_bs + offs_d_ckv_10[:, None]
        if WITHIN_2GB:
            gl.amd.cdna4.async_copy.buffer_load_to_shared(bufs_kv0, Kv_c_cache, offs_k_c0, mask=offs_n_nope0[None, :] < split_kv_end)
        else:
            # No mask needed on global_load path in the loop body: all
            # iterations are guaranteed in-bounds by num_iter arithmetic.
            # Only the epilogue uses mask + other=0 for the last
            # potentially-partial block.
            gl.amd.cdna4.async_copy.global_load_to_shared(bufs_kv0, Kv_c_cache + offs_k_c0)
        gl.amd.cdna4.async_copy.commit_group()

        # local load page_number_pe
        kv_page_number_pe = gl.amd.cdna4.async_copy.load_shared_relaxed(bufs_page.index(async_idx), gl.SliceLayout(0, blocked_kpe))
        kv_loc_pe = kv_page_number_pe
        # global load K_pe
        offs_n_pe = start_n + gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, blocked_kpe))
        offs_d_kpe_1 = gl.arange(0, HEAD_DIM_KPE, layout=gl.SliceLayout(1, blocked_kpe))
        offs_k_pe = kv_loc_pe[None, :] * stride_k_pe_bs + offs_d_kpe_1[:, None] + KV_PE_OFFSET
        if WITHIN_2GB:
            gl.amd.cdna4.async_copy.buffer_load_to_shared(bufs_kpe.index(async_idx), K_pe_cache, offs_k_pe, mask=offs_n_pe[None, :] < split_kv_end)
        else:
            # No mask needed: loop iterations are in-bounds (see KV slice 0 comment).
            gl.amd.cdna4.async_copy.global_load_to_shared(bufs_kpe.index(async_idx), K_pe_cache + offs_k_pe)
        gl.amd.cdna4.async_copy.commit_group()

        #### dot, softmax, dot (part0)
        k_c = gl.amd.cdna4.async_copy.load_shared_relaxed(bufs_kv.index(buf_idx), mfma_layout_b)
        zeros = gl.zeros([BLOCK_H, BLOCK_N], dtype=gl.float32, layout=mfma_layout)
        qk = gl.amd.cdna4.mfma(q_nope, k_c.to(q_nope.dtype), zeros)
        k_pe = gl.amd.cdna4.async_copy.load_shared_relaxed(bufs_kpe.index(buf_idx), mfma_layout_b)
        qk = gl.amd.cdna4.mfma(q_pe, k_pe.to(q_pe.dtype), qk)

        # local load page number for slice 1
        bufs_page_1 = bufs_page.index(async_idx).slice(32, 32, 0)
        kv_page_number_1 = gl.amd.cdna4.async_copy.load_shared_relaxed(bufs_page_1, gl.SliceLayout(0, blocked_kv_slice))
        kv_loc1 = kv_page_number_1
        # global load K_nope slice 1
        offs_n1 = offs_n_nope0 + 32
        offs_k_c1 = kv_loc1[None, :] * stride_kv_c_bs + offs_d_ckv_10[:, None]
        if WITHIN_2GB:
            gl.amd.cdna4.async_copy.buffer_load_to_shared(bufs_kv1, Kv_c_cache, offs_k_c1, mask=offs_n1[None, :] < split_kv_end)
        else:
            # No mask needed: loop iterations are in-bounds (see KV slice 0 comment).
            gl.amd.cdna4.async_copy.global_load_to_shared(bufs_kv1, Kv_c_cache + offs_k_c1)
        gl.amd.cdna4.async_copy.commit_group()

        #### dot, softmax, dot (part1)
        qk *= sm_scale
        offs_n_qk = split_kv_start + i * BLOCK_N + gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, mfma_layout))
        qk = gl.where(offs_n_qk[None, :] < split_kv_end, qk, float("-inf"))
        n_e_max = gl.maximum(gl.max(qk, 1), e_max)
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

    #### global load K
    # local load page number
    gl.amd.cdna4.async_copy.wait_group(3)
    kv_page_number = gl.amd.cdna4.async_copy.load_shared_relaxed(bufs_page.index(async_idx), gl.SliceLayout(0, blocked_kv))
    kv_page_number_pe = gl.amd.cdna4.async_copy.load_shared_relaxed(bufs_page.index(async_idx), gl.SliceLayout(0, blocked_kpe))
    kv_loc = kv_page_number
    kv_loc_pe = kv_page_number_pe
    # global load K_nope
    offs_n_nope = start_n + gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, blocked_kv))
    offs_d_ckv_1 = gl.arange(0, HEAD_DIM_CKV, layout=gl.SliceLayout(1, blocked_kv))
    offs_k_c = kv_loc[None, :] * stride_kv_c_bs + offs_d_ckv_1[:, None]
    if WITHIN_2GB:
        gl.amd.cdna4.async_copy.buffer_load_to_shared(bufs_kv.index(async_idx), Kv_c_cache, offs_k_c, mask=offs_n_nope[None, :] < split_kv_end)
    else:
        gl.amd.cdna4.async_copy.global_load_to_shared(bufs_kv.index(async_idx), Kv_c_cache + offs_k_c, mask=offs_n_nope[None, :] < split_kv_end, other=0)
    gl.amd.cdna4.async_copy.commit_group()
    # global load K_pe
    offs_n_pe = start_n + gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, blocked_kpe))
    offs_d_kpe_1 = gl.arange(0, HEAD_DIM_KPE, layout=gl.SliceLayout(1, blocked_kpe))
    offs_k_pe = kv_loc_pe[None, :] * stride_k_pe_bs + offs_d_kpe_1[:, None] + KV_PE_OFFSET
    if WITHIN_2GB:
        gl.amd.cdna4.async_copy.buffer_load_to_shared(bufs_kpe.index(async_idx), K_pe_cache, offs_k_pe, mask=offs_n_pe[None, :] < split_kv_end)
    else:
        gl.amd.cdna4.async_copy.global_load_to_shared(bufs_kpe.index(async_idx), K_pe_cache + offs_k_pe, mask=offs_n_pe[None, :] < split_kv_end, other=0)
    gl.amd.cdna4.async_copy.commit_group()

    # dot, softmax, dot
    gl.amd.cdna4.async_copy.wait_group(2)
    k_c = bufs_kv.index(buf_idx).load(layout=mfma_layout_b)
    zeros = gl.zeros([BLOCK_H, BLOCK_N], dtype=gl.float32, layout=mfma_layout)
    qk = gl.amd.cdna4.mfma(q_nope, k_c.to(q_nope.dtype), zeros)

    k_pe = bufs_kpe.index(buf_idx).load(layout=mfma_layout_b)
    qk = gl.amd.cdna4.mfma(q_pe, k_pe.to(q_pe.dtype), qk)
    qk *= sm_scale
    offs_n_qk = split_kv_start + (num_iter - 2) * BLOCK_N + gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, mfma_layout))
    qk = gl.where(offs_n_qk[None, :] < split_kv_end, qk, float("-inf"))
    n_e_max = gl.maximum(gl.max(qk, 1), e_max)
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

    #### dot, softmax, dot
    gl.amd.cdna4.async_copy.wait_group(0)
    k_c = bufs_kv.index(buf_idx).load(layout=mfma_layout_b)
    zeros = gl.zeros([BLOCK_H, BLOCK_N], dtype=gl.float32, layout=mfma_layout)
    qk = gl.amd.cdna4.mfma(q_nope, k_c.to(q_nope.dtype), zeros)

    k_pe = bufs_kpe.index(buf_idx).load(layout=mfma_layout_b)
    qk = gl.amd.cdna4.mfma(q_pe, k_pe.to(q_pe.dtype), qk)
    qk *= sm_scale
    offs_n_qk = split_kv_start + (num_iter - 1) * BLOCK_N + gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, mfma_layout))
    qk = gl.where(offs_n_qk[None, :] < split_kv_end, qk, float("-inf"))
    n_e_max = gl.maximum(gl.max(qk, 1), e_max)
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
    rcp = 1.0 / e_sum
    stored_value = (acc * rcp[:, None]).to(dtype)
    gl.amd.cdna4.buffer_store(stored_value, ptr=O, offsets=offs_o)
    if NUM_KV_SPLITS > 1:
        # Per-split lse for stage-2 reduce (stored at the trailing slot).
        offs_o_lse = cur_batch * stride_o_b + cur_head_o * stride_o_h + split_kv_id * stride_o_s + HEAD_DIM_CKV
        lse = e_max + gl.log(e_sum)
        gl.amd.cdna4.buffer_store(lse.to(dtype), ptr=O, offsets=offs_o_lse)
# fmt: on


# fmt: off
@triton.jit
def _mla_softmax_reducev_kernel(
    Logits,
    B_seq_len,
    O,  # noqa: E741
    stride_l_b,
    stride_l_h,
    stride_l_s,
    stride_o_b,
    stride_o_h,
    NUM_KV_SPLITS: tl.constexpr,
    HEAD_DIM_CKV: tl.constexpr,
    USE_2D_VIEW: tl.constexpr,
    ALL_SPLITS_NONEMPTY: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)

    offs_d_ckv = tl.arange(0, HEAD_DIM_CKV)
    offs_l = cur_batch * stride_l_b + cur_head * stride_l_h + offs_d_ckv
    offs_l_1 = cur_batch * stride_l_b + cur_head * stride_l_h + HEAD_DIM_CKV

    e_sum = 0.0
    e_max = -float("inf")
    acc = tl.zeros([HEAD_DIM_CKV], dtype=tl.float32)

    if not ALL_SPLITS_NONEMPTY:
        if USE_2D_VIEW:
            cur_batch_seq_len = tl.load(B_seq_len + cur_batch)
        else:
            cur_batch_seq_len = tl.load(B_seq_len + cur_batch + 1) - tl.load(
                B_seq_len + cur_batch
            )

    for split_kv_id in range(0, NUM_KV_SPLITS):
        if not ALL_SPLITS_NONEMPTY:
            kv_len_per_split = tl.cdiv(cur_batch_seq_len, NUM_KV_SPLITS)
            split_kv_start = kv_len_per_split * split_kv_id
            split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)
            if split_kv_end <= split_kv_start:
                continue

        logits = tl.load(Logits + offs_l + split_kv_id * stride_l_s)
        logits_1 = tl.load(Logits + offs_l_1 + split_kv_id * stride_l_s)

        n_e_max = tl.maximum(logits_1, e_max)
        old_scale = tl.exp(e_max - n_e_max)
        acc *= old_scale
        exp_logic = tl.exp(logits_1 - n_e_max)
        acc += exp_logic * logits

        e_sum = e_sum * old_scale + exp_logic
        e_max = n_e_max

    tl.store(
        O + cur_batch * stride_o_b + cur_head * stride_o_h + offs_d_ckv,
        acc / e_sum,
    )
# fmt: on


def mla_decode_gluon(
    q_nope,  # [batch, nhead, kv_lora_rank]
    q_pe,  # [batch, nhead, qk_rope_head_dim]
    # Shared: kv_c=[N, kv_lora_rank+qk_rope_head_dim], k_pe=None,              kv_pe_offset=kv_lora_rank
    # Split:  kv_c=[N, kv_lora_rank],                k_pe=[N,qk_rope_head_dim], kv_pe_offset=0
    kv_c,
    o,  # [batch, nhead, kv_lora_rank] output buffer
    page_table,  # 2D: block_table [batch, max_seqlen] | 1D: kv_indices [total_kv]
    seq_info,  # 2D: cache_seqlens [batch]           | 1D: kv_indptr [batch+1]
    sm_scale,
    k_pe=None,
    kv_pe_offset=512,
    use_2d_view=True,
    min_kv_seq_len=1,
):
    if k_pe is None:
        k_pe = kv_c

    batch_size, nhead, head_dim_ckv = q_nope.shape
    head_dim_kpe = q_pe.shape[-1]

    PAGE_SIZE = 1
    BLOCK_H = 64
    BLOCK_N = 64
    NUM_XCDS = get_num_xcds()

    # Auto-pick NUM_KV_SPLITS so the launch fills ~256 workgroups (one wave on
    # MI350). For the supported (batch, nhead) matrix the result is in {1, 2, 4}.
    TARGET_GRID = 256
    base_grid = NUM_XCDS * triton.cdiv(nhead, BLOCK_H) * (batch_size // NUM_XCDS)
    NUM_KV_SPLITS = max(1, triton.next_power_of_2(triton.cdiv(TARGET_GRID, base_grid)))

    assert (
        arch_info.get_arch() == "gfx950"
    ), f"mla_decode_gluon requires gfx950 (CDNA4), got {arch_info.get_arch()}"
    assert batch_size in (
        64,
        128,
        256,
    ), f"mla_decode_gluon requires batch_size in (64, 128, 256), got {batch_size}"
    assert nhead in (
        64,
        128,
    ), f"mla_decode_gluon requires nhead in (64, 128), got {nhead}"
    assert (
        head_dim_ckv == 512
    ), f"mla_decode_gluon requires head_dim_ckv=512, got {head_dim_ckv}"
    assert (
        head_dim_kpe == 64
    ), f"mla_decode_gluon requires head_dim_kpe=64, got {head_dim_kpe}"
    # gl.assume(num_iter > PIPELINE_STAGES) inside the kernel requires every
    # split to have > PIPELINE_STAGES * BLOCK_N tokens. The smallest split
    # (last) for a batch of length s is s - (k-1)*ceil(s/k); a sufficient
    # bound that keeps this > min_per_split_tokens for all s >= min_kv_seq_len
    # is min_kv_seq_len > k * (min_per_split_tokens + k).
    PIPELINE_STAGES = 3  # matches gl.assume(num_iter > 3) inside the kernel
    min_per_split_tokens = PIPELINE_STAGES * BLOCK_N  # 192
    min_kv_seq_len_required = NUM_KV_SPLITS * (min_per_split_tokens + NUM_KV_SPLITS)
    assert (
        min_kv_seq_len > min_kv_seq_len_required
    ), f"mla_decode_gluon requires min_kv_seq_len > {min_kv_seq_len_required} (NUM_KV_SPLITS={NUM_KV_SPLITS}), got {min_kv_seq_len}"
    assert q_nope.dtype == torch.bfloat16, f"q_nope must be bf16, got {q_nope.dtype}"
    assert q_pe.dtype == torch.bfloat16, f"q_pe must be bf16, got {q_pe.dtype}"
    assert kv_c.dtype == torch.bfloat16, f"kv_c must be bf16, got {kv_c.dtype}"
    assert k_pe.dtype == torch.bfloat16, f"k_pe must be bf16, got {k_pe.dtype}"

    # buffer_load uses scalar base + 32-bit offsets, limiting addressable range.
    # For KV caches > 2 GB the kernel falls back to global_load (64-bit pointers).
    max_kv_bytes = kv_c.shape[0] * kv_c.stride(0) * kv_c.element_size()
    within_2gb = max_kv_bytes <= 0x80000000  # 2 GB

    if NUM_KV_SPLITS == 1:
        # Fast path: stage-1 writes the final attention output directly to o,
        # no temp buffer, no reduce.
        attn_logits = o.view(batch_size, nhead, NUM_KV_SPLITS, head_dim_ckv)
    else:
        # Stage-1 writes per-split (acc, lse) to a temp buffer; the trailing
        # slot at index head_dim_ckv holds the per-split lse.
        attn_logits = torch.empty(
            (batch_size, nhead, NUM_KV_SPLITS, head_dim_ckv + 1),
            dtype=o.dtype,
            device=o.device,
        )

    grid = (
        NUM_XCDS,
        triton.cdiv(nhead, BLOCK_H),
        (batch_size // NUM_XCDS) * NUM_KV_SPLITS,
    )
    stride_page_bs = page_table.stride(0) if use_2d_view else 0

    _mla_decode_gluon[grid](
        q_nope,
        q_pe,
        kv_c,
        k_pe,
        page_table,
        seq_info,
        attn_logits,
        sm_scale,
        q_nope.stride(0),
        q_nope.stride(1),
        q_pe.stride(0),
        q_pe.stride(1),
        kv_c.stride(-2),
        k_pe.stride(-2),
        stride_page_bs,
        attn_logits.stride(0),
        attn_logits.stride(1),
        attn_logits.stride(2),
        BLOCK_H=BLOCK_H,
        BLOCK_N=BLOCK_N,
        NUM_KV_SPLITS=NUM_KV_SPLITS,
        PAGE_SIZE=PAGE_SIZE,
        HEAD_DIM_CKV=head_dim_ckv,
        HEAD_DIM_KPE=head_dim_kpe,
        KV_PE_OFFSET=kv_pe_offset,
        USE_2D_VIEW=use_2d_view,
        WITHIN_2GB=within_2gb,
        NUM_XCDS=NUM_XCDS,
    )

    if NUM_KV_SPLITS > 1:
        # Stage-2: combine per-split (acc, lse) into the final attention in o.
        # ALL_SPLITS_NONEMPTY lets the reduce skip the per-iter range check;
        # for NUM_KV_SPLITS in {2,4} the threshold is trivially satisfied.
        all_splits_nonempty = min_kv_seq_len > (NUM_KV_SPLITS - 1) ** 2
        grid_reduce = (batch_size, nhead)
        _mla_softmax_reducev_kernel[grid_reduce](
            attn_logits,
            seq_info,
            o,
            attn_logits.stride(0),
            attn_logits.stride(1),
            attn_logits.stride(2),
            o.stride(0),
            o.stride(1),
            NUM_KV_SPLITS=NUM_KV_SPLITS,
            HEAD_DIM_CKV=head_dim_ckv,
            USE_2D_VIEW=use_2d_view,
            ALL_SPLITS_NONEMPTY=all_splits_nonempty,
            num_warps=8,
        )

    return attn_logits, None
