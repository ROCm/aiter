# SPDX-License-Identifier: MIT
# Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

import torch
import triton
import triton.language as tl

from typing import Optional
from aiter.ops.triton.utils.pid_preprocessing import remap_xcd
from aiter.ops.triton.utils.mha_kernel_utils import (
    compute_fp8_scaling_factors,
    is_fp8,
)


# This function computes delta given output Out and gradient DO
# Here is the I/O shape:
# Out: (batch, nhead_q, max_seqlens_q, headDim)
# DO: (batch, nhead_q, max_seqlens_q, headDim)
# Delta: (batch, nheads_q, max_seqlens_q), same as softmax_lse defined at
@triton.jit
def _bwd_preprocess(
    o_ptr,
    do_ptr,  # noqa: E741
    delta_ptr,
    stride_o_b,
    stride_o_h,
    stride_o_m,
    stride_o_k,
    stride_delta_b,
    stride_delta_h,
    stride_delta_m,
    stride_descale_do_z,
    cu_seqlens_q,
    max_seqlen_q,
    descale_do_ptr,
    BLOCK_M: tl.constexpr,
    BLOCK_D_MODEL: tl.constexpr,
    BLOCK_D_MODEL_POW2: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    IS_FP8: tl.constexpr,
):
    pid_m = tl.program_id(0)  # seqlen
    bid = tl.program_id(1)  # batch
    hid = tl.program_id(2)  # head

    # Handle varlen
    q_start = 0
    seqlen_q = max_seqlen_q
    if IS_VARLEN:
        q_start = tl.load(cu_seqlens_q + bid)
        q_end = tl.load(cu_seqlens_q + bid + 1)
        seqlen_q = q_end - q_start
    else:
        q_start = 0
        seqlen_q = max_seqlen_q

    # Compute offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_D_MODEL_POW2)

    # Offset O/DO by batch, head and q_start
    offs = (
        bid * stride_o_b
        + hid * stride_o_h
        + q_start * stride_o_m
        + offs_m[:, None] * stride_o_m
        + offs_k[None, :] * stride_o_k
    )

    # create masks
    mask_m = offs_m < seqlen_q
    mask = mask_m[:, None]
    PADDED_HEAD: tl.constexpr = BLOCK_D_MODEL != BLOCK_D_MODEL_POW2
    if PADDED_HEAD:
        mask &= offs_k[None, :] < BLOCK_D_MODEL

    # load [BLOCK_M, BLOCK_D_MODEL_POW2]
    o = tl.load(o_ptr + offs, mask=mask, other=0.0)
    do = tl.load(do_ptr + offs, mask=mask, other=0.0)

    # compute and write-back to delta
    if IS_FP8:
        descale_do = tl.load(descale_do_ptr + bid * stride_descale_do_z + hid)

        # NOTE: do is in the fp8 range and o is not in fp8
        delta = tl.sum(o.to(tl.float32) * (do.to(tl.float32) * descale_do), axis=1)
    else:
        delta = tl.sum(o.to(tl.float32) * do.to(tl.float32), axis=1)

    offs_delta = (
        bid * stride_delta_b
        + hid * stride_delta_h
        + q_start * stride_delta_m
        + offs_m * stride_delta_m
    )
    tl.store(delta_ptr + offs_delta, delta, mask=mask_m)


@triton.jit
def _bwd_dkdvdq_inner(
    dk,
    dv,
    Q,
    k,
    v,
    DO,
    DQ,
    M,
    D,
    sm_scale,
    stride_q_m,
    stride_q_k,
    stride_dq_m,
    stride_dq_k,
    stride_do_m,
    stride_do_k,
    stride_dropout_m,
    stride_dropout_n,
    stride_deltam,
    dropout_p,
    philox_seed,
    batch_philox_offset,
    dropout_offset,
    seqlen_q,
    seqlen_k,
    start_n,
    start_m,
    num_steps,
    descale_q,
    descale_k,
    descale_v,
    descale_do,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D_MODEL: tl.constexpr,
    BLOCK_D_MODEL_POW2: tl.constexpr,
    MASK: tl.constexpr,
    ENABLE_DROPOUT: tl.constexpr,
    IS_FP8: tl.constexpr,
    FP8_MAX: tl.constexpr,
    workgroup_id,
):
    tl.assume(stride_q_m >= 0)
    tl.assume(stride_q_k >= 0)
    tl.assume(stride_dq_m >= 0)
    tl.assume(stride_dq_k >= 0)
    tl.assume(stride_do_m >= 0)
    tl.assume(stride_do_k >= 0)
    tl.assume(stride_deltam >= 0)

    PADDED_HEAD: tl.constexpr = BLOCK_D_MODEL != BLOCK_D_MODEL_POW2
    delta_qk = seqlen_q - seqlen_k
    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_n = start_n + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_D_MODEL_POW2)

    # mask to make sure not OOB of seqlen_q
    mask_n = offs_n < seqlen_k

    qT_ptrs_start = (
        Q + offs_m[None, :] * stride_q_m + offs_k[:, None] * stride_q_k
    )  # [BLOCK_D_MODEL_POW2, BLOCK_M]
    dq_ptrs_start = (
        DQ + offs_m[:, None] * stride_dq_m + offs_k[None, :] * stride_dq_k
    )  # [BLOCK_M, BLOCK_D_MODEL_POW2]

    do_ptrs_start = DO + offs_m[:, None] * stride_do_m + offs_k[None, :] * stride_do_k
    curr_m = start_m
    step_m = BLOCK_M
    curr_philox_offset = batch_philox_offset

    # Iterate over blocks(BLOCK_M size) of Q while calculating
    # a fixed block(BLOCK_N) of dk and dv. Note, during backward
    # pass P has to be recomputed. However, this kernel computes
    # dV and dK, so we compute we need P^T and S^T. See backward pass
    # equations
    #
    # From Flash Attention Paper:
    # ForwardPass: S = QkT, P=softmax(S), O=PV
    #
    # BackwardPass equations
    # dV = P^TdO
    # dP = dOV^T
    # dS = dsoftmax(dP)
    # dQ = dSK
    # dK = QdS^T

    for iter in range(num_steps):
        # Permute the iteration order to reduce the probability that concurrent workgroups (that share the same q head idx and batch idx) are at the same iteration
        blk_idx = (iter + workgroup_id) % num_steps

        curr_m = start_m + blk_idx * step_m
        qT_ptrs = qT_ptrs_start + blk_idx * step_m * stride_q_m
        dq_ptrs = dq_ptrs_start + blk_idx * step_m * stride_dq_m
        do_ptrs = do_ptrs_start + blk_idx * step_m * stride_do_m

        offs_m = curr_m + tl.arange(0, BLOCK_M)
        mask_m = offs_m < seqlen_q
        mask_qT = mask_m[None, :]
        mask_do = mask_m[:, None]
        mask_nm = mask_n[:, None] & (offs_m[None, :] < seqlen_q)

        if PADDED_HEAD:
            mask_qT &= offs_k[:, None] < BLOCK_D_MODEL
            mask_do &= offs_k[None, :] < BLOCK_D_MODEL

        # load qT
        qT = tl.load(qT_ptrs, mask=mask_qT, other=0.0)

        # dropout
        if ENABLE_DROPOUT:
            # NOTE: dropout is transposed because it is used to mask pT
            philox_offs = (
                curr_philox_offset
                + offs_m[None, :] * stride_dropout_m
                + offs_n[:, None] * stride_dropout_n
            )
            rand_vals = tl.rand(philox_seed, philox_offs)
            dropout_mask = rand_vals > dropout_p
            dropout_scale = 1.0 / (1 - dropout_p)

        # Load M
        m = tl.load(M + offs_m * stride_deltam, mask=mask_m, other=0.0)

        # Compute qkT
        if IS_FP8:
            qkT = tl.dot(k, qT) * descale_q * descale_k
        else:
            qkT = tl.dot(k, qT)

        # Compute pT(use m and also apply sm_scale)
        pT = tl.math.exp(qkT * sm_scale - m[None, :])

        if MASK:
            causal_mask = (offs_m[None, :] - delta_qk) >= (offs_n[:, None])
            mask = causal_mask & mask_nm
            pT = tl.where(mask, pT, 0.0)

        # load DO
        do = tl.load(do_ptrs, mask=mask_do, other=0.0)

        # dV
        if ENABLE_DROPOUT:
            pT_dropout = tl.where(dropout_mask, pT, 0.0) * dropout_scale
            if IS_FP8:
                scale_p_dropout, descale_p_dropout = compute_fp8_scaling_factors(
                    pT_dropout, FP8_MAX
                )
                dv += (
                    tl.dot((pT_dropout * scale_p_dropout).to(do.type.element_ty), do)
                    * descale_p_dropout
                    * descale_do
                )
            else:
                dv += tl.dot(pT_dropout.to(do.type.element_ty), do)
        else:
            if IS_FP8:
                scale_pT, descale_pT = compute_fp8_scaling_factors(pT, FP8_MAX)
                dv += (
                    tl.dot((pT * scale_pT).to(do.type.element_ty), do)
                    * descale_pT
                    * descale_do
                )
            else:
                dv += tl.dot(pT.to(do.type.element_ty), do)

        # Load delta
        Di = tl.load(D + offs_m * stride_deltam, mask=mask_m)

        # Compute dP and dS
        if IS_FP8:
            dpT = tl.dot(v, tl.trans(do)) * descale_v * descale_do
        else:
            dpT = tl.dot(v, tl.trans(do))

        if ENABLE_DROPOUT:
            dpT = tl.where(dropout_mask, dpT, 0.0) * dropout_scale

        delta_i = Di[None, :]
        dsT = pT * (dpT - delta_i)

        # compute dk
        if IS_FP8:
            scale_dsT, descale_dsT = compute_fp8_scaling_factors(dsT, FP8_MAX)
            dk += (
                tl.dot((dsT * scale_dsT).to(qT.type.element_ty), tl.trans(qT))
                * descale_dsT
                * descale_q
            )
        else:
            dk += tl.dot(dsT.to(qT.type.element_ty), tl.trans(qT))

        # We can compute the dq_partial here and do a atomic add to the correct memory location
        # NOTE: Possible problems with the atomic add: contention, is inside a loop which has achieved bad perf before
        # (BLOCK_M, BLOCK_N) x (BLOCK_N, D)
        if IS_FP8:
            dq_partial = (
                tl.dot((dsT * scale_dsT).to(k.dtype).T, k) * descale_dsT * descale_k
            )
        else:
            dq_partial = tl.dot(dsT.to(k.dtype).T, k)
        tl.atomic_add(
            dq_ptrs,
            dq_partial * sm_scale,
            mask=mask_m[:, None] & (offs_k[None, :] < BLOCK_D_MODEL),
            sem="relaxed",
        )

    return dk, dv


@triton.jit
def _bwd_kernel_dkdvdq_causal(
    q_ptr,
    k_ptr,
    v_ptr,
    sm_scale,
    do_ptr,
    dk_ptr,
    dv_ptr,
    dq_ptr,
    m_ptr,
    delta_ptr,
    stride_q_b_in,
    stride_q_h_in,
    stride_q_m_in,
    stride_q_k_in,
    stride_k_b_in,
    stride_k_h_in,
    stride_k_n_in,
    stride_k_k_in,
    stride_v_b_in,
    stride_v_h_in,
    stride_v_n_in,
    stride_v_k_in,
    stride_dk_b_in,
    stride_dk_h_in,
    stride_dk_n_in,
    stride_dk_k_in,
    stride_dq_b_in,
    stride_dq_h_in,
    stride_dq_m_in,
    stride_dq_k_in,
    stride_delta_b_in,
    stride_delta_h_in,
    stride_delta_m_in,
    stride_do_b_in,
    stride_do_h_in,
    stride_do_m_in,
    stride_do_k_in,
    stride_dropout_b_in,
    stride_dropout_h_in,
    stride_dropout_m_in,
    stride_dropout_n_in,
    stride_descale_q_z_in,
    stride_descale_k_z_in,
    stride_descale_v_z_in,
    stride_descale_do_z_in,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    dropout_mask,
    dropout_p,
    philox_seed,
    philox_offset_base_in,
    descale_q_ptr,
    descale_k_ptr,
    descale_v_ptr,
    descale_do_ptr,
    NUM_Q_HEADS: tl.constexpr,
    NUM_K_HEADS: tl.constexpr,
    BATCH,
    NUM_K_PIDS,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLK_SLICE_FACTOR: tl.constexpr,
    BLOCK_D_MODEL: tl.constexpr,
    BLOCK_D_MODEL_POW2: tl.constexpr,
    ENABLE_DROPOUT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    IS_FP8: tl.constexpr,
    FP8_MAX: tl.constexpr,
    NUM_SMS: tl.constexpr,
    USE_INT64_STRIDES: tl.constexpr,
):
    if USE_INT64_STRIDES:
        stride_q_b = tl.cast(stride_q_b_in, tl.int64)
        stride_q_h = tl.cast(stride_q_h_in, tl.int64)
        stride_q_m = tl.cast(stride_q_m_in, tl.int64)
        stride_q_k = tl.cast(stride_q_k_in, tl.int64)
        stride_k_b = tl.cast(stride_k_b_in, tl.int64)
        stride_k_h = tl.cast(stride_k_h_in, tl.int64)
        stride_k_n = tl.cast(stride_k_n_in, tl.int64)
        stride_k_k = tl.cast(stride_k_k_in, tl.int64)
        stride_v_b = tl.cast(stride_v_b_in, tl.int64)
        stride_v_h = tl.cast(stride_v_h_in, tl.int64)
        stride_v_n = tl.cast(stride_v_n_in, tl.int64)
        stride_v_k = tl.cast(stride_v_k_in, tl.int64)
        stride_dk_b = tl.cast(stride_dk_b_in, tl.int64)
        stride_dk_h = tl.cast(stride_dk_h_in, tl.int64)
        stride_dk_n = tl.cast(stride_dk_n_in, tl.int64)
        stride_dk_k = tl.cast(stride_dk_k_in, tl.int64)
        stride_dq_b = tl.cast(stride_dq_b_in, tl.int64)
        stride_dq_h = tl.cast(stride_dq_h_in, tl.int64)
        stride_dq_m = tl.cast(stride_dq_m_in, tl.int64)
        stride_dq_k = tl.cast(stride_dq_k_in, tl.int64)
        stride_delta_b = tl.cast(stride_delta_b_in, tl.int64)
        stride_delta_h = tl.cast(stride_delta_h_in, tl.int64)
        stride_delta_m = tl.cast(stride_delta_m_in, tl.int64)
        stride_do_b = tl.cast(stride_do_b_in, tl.int64)
        stride_do_h = tl.cast(stride_do_h_in, tl.int64)
        stride_do_m = tl.cast(stride_do_m_in, tl.int64)
        stride_do_k = tl.cast(stride_do_k_in, tl.int64)
        stride_dropout_b = tl.cast(stride_dropout_b_in, tl.int64)
        stride_dropout_h = tl.cast(stride_dropout_h_in, tl.int64)
        stride_dropout_m = tl.cast(stride_dropout_m_in, tl.int64)
        stride_dropout_n = tl.cast(stride_dropout_n_in, tl.int64)
        philox_offset_base = tl.cast(philox_offset_base_in, tl.int64)
        if IS_FP8:
            stride_descale_q_z = tl.cast(stride_descale_q_z_in, tl.int64)
            stride_descale_k_z = tl.cast(stride_descale_k_z_in, tl.int64)
            stride_descale_v_z = tl.cast(stride_descale_v_z_in, tl.int64)
            stride_descale_do_z = tl.cast(stride_descale_do_z_in, tl.int64)
    else:
        stride_q_b = stride_q_b_in
        stride_q_h = stride_q_h_in
        stride_q_m = stride_q_m_in
        stride_q_k = stride_q_k_in
        stride_k_b = stride_k_b_in
        stride_k_h = stride_k_h_in
        stride_k_n = stride_k_n_in
        stride_k_k = stride_k_k_in
        stride_v_b = stride_v_b_in
        stride_v_h = stride_v_h_in
        stride_v_n = stride_v_n_in
        stride_v_k = stride_v_k_in
        stride_dk_b = stride_dk_b_in
        stride_dk_h = stride_dk_h_in
        stride_dk_n = stride_dk_n_in
        stride_dk_k = stride_dk_k_in
        stride_dq_b = stride_dq_b_in
        stride_dq_h = stride_dq_h_in
        stride_dq_m = stride_dq_m_in
        stride_dq_k = stride_dq_k_in
        stride_delta_b = stride_delta_b_in
        stride_delta_h = stride_delta_h_in
        stride_delta_m = stride_delta_m_in
        stride_do_b = stride_do_b_in
        stride_do_h = stride_do_h_in
        stride_do_m = stride_do_m_in
        stride_do_k = stride_do_k_in
        stride_dropout_b = stride_dropout_b_in
        stride_dropout_h = stride_dropout_h_in
        stride_dropout_m = stride_dropout_m_in
        stride_dropout_n = stride_dropout_n_in
        philox_offset_base = philox_offset_base_in
        stride_descale_q_z = stride_descale_q_z_in
        stride_descale_k_z = stride_descale_k_z_in
        stride_descale_v_z = stride_descale_v_z_in
        stride_descale_do_z = stride_descale_do_z_in

    GROUP_SIZE = NUM_Q_HEADS // NUM_K_HEADS
    wid = tl.program_id(0)  # workgoup id: 0, ..., NUM_Q_PIDS * BATCH * NUM_K_HEADS - 1

    NUM_XCD: tl.constexpr = 8
    head_q_idx = wid % NUM_Q_HEADS
    head_q_idx = remap_xcd(head_q_idx, NUM_Q_HEADS, NUM_XCD)
    seq_k_blk_idx = (wid // NUM_Q_HEADS) % NUM_K_PIDS
    batch_idx = (wid // (NUM_K_PIDS * NUM_Q_HEADS)) % BATCH

    # In the backward we dont want concurrent workgroups to handle consecutive heads or blocks, so remap them to be far apart.
    head_q_idx = (head_q_idx * 29) % NUM_Q_HEADS
    # seq_k_blk_idx = (seq_k_blk_idx * 29) % NUM_K_PIDS

    head_k_idx = head_q_idx // GROUP_SIZE

    # Determine q and k start along with seqlen_q and seqlen_k
    q_start = 0
    k_start = 0
    seqlen_q = max_seqlen_q
    seqlen_k = max_seqlen_k
    if IS_VARLEN:
        q_start = tl.load(cu_seqlens_q + batch_idx)
        q_end = tl.load(cu_seqlens_q + batch_idx + 1)
        k_start = tl.load(cu_seqlens_k + batch_idx)
        k_end = tl.load(cu_seqlens_k + batch_idx + 1)
        seqlen_q = q_end - q_start
        seqlen_k = k_end - k_start

    dk = tl.zeros([BLOCK_N, BLOCK_D_MODEL_POW2], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, BLOCK_D_MODEL_POW2], dtype=tl.float32)

    # Figure out causal starting block since we have seqlen_q >=< seqlen_k.
    # Unlike forward pass where we tile on M dim and iterate on N dim, so that
    # we can skip some M blocks, in backward pass, we tile on the N dim for kv
    # and iterate over the M. In this way, we cannot skip N blocks, but only to
    # determine the starting M blocks to skip some initial blocks masked by
    # causal.
    delta_qk = seqlen_q - seqlen_k

    # q < k: some blocks will have no Masked block, other needs to re-calc
    # starting position
    # delta_qk is negative so flip it, only multiple of BLOCK_N can skip the
    # masked op
    num_blocks_skip = -delta_qk // BLOCK_N
    delta_aligned = (num_blocks_skip + 1) * BLOCK_N + delta_qk
    start_delta_q_lt_k = delta_aligned // BLOCK_M * BLOCK_M
    if delta_qk >= 0:
        start_delta = delta_qk
    else:
        start_delta = start_delta_q_lt_k

    start_n = seq_k_blk_idx * BLOCK_N

    offs_k = tl.arange(0, BLOCK_D_MODEL_POW2)
    offs_n = start_n + tl.arange(0, BLOCK_N)
    # Mask for loading K and V
    mask_kv = offs_n[:, None] < seqlen_k
    PADDED_HEAD: tl.constexpr = BLOCK_D_MODEL != BLOCK_D_MODEL_POW2
    if PADDED_HEAD:
        mask_k = offs_k < BLOCK_D_MODEL
        mask_kv &= mask_k[None, :]

    GROUP_SIZE = NUM_Q_HEADS // NUM_K_HEADS
    adj_k = (
        batch_idx * stride_k_b
        + head_k_idx * stride_k_h
        + k_start * stride_k_n
        + offs_n[:, None] * stride_k_n
        + offs_k[None, :] * stride_k_k
    )
    adj_v = (
        batch_idx * stride_v_b
        + head_k_idx * stride_v_h
        + k_start * stride_v_n
        + offs_n[:, None] * stride_v_n
        + offs_k[None, :] * stride_v_k
    )
    # load K and V: they stay in SRAM throughout the inner loop.
    k = tl.load(k_ptr + adj_k, mask=mask_kv, other=0.0)
    v = tl.load(v_ptr + adj_v, mask=mask_kv, other=0.0)

    # If MQA / GQA, set the K and V head offsets appropriately.
    # for head_q_idx in range(head_k_idx * GROUP_SIZE, head_k_idx * GROUP_SIZE + GROUP_SIZE):
    if delta_qk >= 0:
        start_m = start_n + start_delta
        len_m = BLOCK_N
    else:
        start_m = max(start_n + delta_qk, 0)
        start_m = (start_m // BLOCK_M) * BLOCK_M
        # because we might shift the masked blocks up, we are deeper into
        # the masked out region, so we would potentially increase the total
        # steps with masked operation to get out of it
        residue_m = max(start_n + delta_qk - start_m, 0)
        len_m = BLOCK_N + residue_m

    # offset input and output tensor by batch and Q/K heads
    adj_q = batch_idx * stride_q_b + head_q_idx * stride_q_h + q_start * stride_q_m
    adj_dq = batch_idx * stride_dq_b + head_q_idx * stride_dq_h + q_start * stride_dq_m

    q_ptr_adj = q_ptr + adj_q
    dq_ptr_adj = dq_ptr + adj_dq

    adj_do = batch_idx * stride_do_b + head_q_idx * stride_do_h + q_start * stride_do_m
    do_ptr_adj = do_ptr + adj_do
    adj_delta = (
        batch_idx * stride_delta_b
        + head_q_idx * stride_delta_h
        + q_start * stride_delta_m
    )
    m_ptr_adj = m_ptr + adj_delta
    delta_ptr_adj = delta_ptr + adj_delta

    # batch_philox_offset is the ACTUALLY dropout offset
    # dropout_offset is for debug purpose and will be removed later
    batch_philox_offset = 0
    dropout_offset = 0
    if ENABLE_DROPOUT:
        batch_philox_offset = (
            philox_offset_base
            + batch_idx * stride_dropout_b
            + head_q_idx * stride_dropout_h
        )
        dropout_offset = (
            dropout_mask + batch_idx * stride_dropout_b + head_q_idx * stride_dropout_h
        )

    MASK_BLOCK_M: tl.constexpr = BLOCK_M // BLK_SLICE_FACTOR
    # bound the masked operation to q len so it does not have to wast cycles
    len_m = min(len_m, seqlen_q)
    num_steps = tl.cdiv(len_m, MASK_BLOCK_M)

    # when q < k, we may skip the initial masked op
    if seq_k_blk_idx < num_blocks_skip:
        num_steps = 0

    if IS_FP8:
        descale_q = tl.load(descale_q_ptr + batch_idx * stride_descale_q_z + head_q_idx)
        descale_k = tl.load(descale_k_ptr + batch_idx * stride_descale_k_z + head_k_idx)
        descale_v = tl.load(descale_v_ptr + batch_idx * stride_descale_v_z + head_k_idx)
        descale_do = tl.load(
            descale_do_ptr + batch_idx * stride_descale_do_z + head_q_idx
        )
    else:
        descale_q, descale_k, descale_v, descale_do = 1.0, 1.0, 1.0, 1.0

    # if unaligned start_m is negative, the current N-tile has no block on the
    #   diagonal of causal mask, so everything have no causal mask
    dk, dv = _bwd_dkdvdq_inner(
        dk,
        dv,  # output tensors
        q_ptr_adj,
        k,
        v,
        do_ptr_adj,
        dq_ptr_adj,
        m_ptr_adj,
        delta_ptr_adj,
        sm_scale,  # input tensors
        stride_q_m,
        stride_q_k,  # strides for q
        stride_dq_m,
        stride_dq_k,  # strides for q
        stride_do_m,
        stride_do_k,  # strides for o
        stride_dropout_m,
        stride_dropout_n,  # strides for dropout
        stride_delta_m,
        dropout_p,
        philox_seed,
        batch_philox_offset,
        dropout_offset,  #
        seqlen_q,
        seqlen_k,  # max sequence length for q and k
        start_n,
        start_m,
        num_steps,  # iteration numbers
        descale_q,
        descale_k,
        descale_v,
        descale_do,  # fp8 descale factors from user
        MASK_BLOCK_M,
        BLOCK_N,  # block dim
        BLOCK_D_MODEL,
        BLOCK_D_MODEL_POW2,  # head dim
        MASK=True,  # causal masking
        ENABLE_DROPOUT=ENABLE_DROPOUT,  # activate dropout
        IS_FP8=IS_FP8,
        FP8_MAX=FP8_MAX,
        workgroup_id=seq_k_blk_idx,
    )

    start_m += num_steps * MASK_BLOCK_M
    num_steps = tl.cdiv(seqlen_q - start_m, BLOCK_M)

    dk, dv = _bwd_dkdvdq_inner(
        dk,
        dv,  # output tensors
        q_ptr_adj,
        k,
        v,
        do_ptr_adj,
        dq_ptr_adj,
        m_ptr_adj,
        delta_ptr_adj,
        sm_scale,  # input tensors
        stride_q_m,
        stride_q_k,  # strides for q
        stride_dq_m,
        stride_dq_k,  # strides for dq
        stride_do_m,
        stride_do_k,  # strides for o
        stride_dropout_m,
        stride_dropout_n,  # strides for dropout
        stride_delta_m,
        dropout_p,
        philox_seed,
        batch_philox_offset,
        dropout_offset,  #
        seqlen_q,
        seqlen_k,  # max sequence length for q and k
        start_n,
        start_m,
        num_steps,  # iteration numbers
        descale_q,
        descale_k,
        descale_v,
        descale_do,  # fp8 descale factors from user
        BLOCK_M,
        BLOCK_N,  # block dim
        BLOCK_D_MODEL,
        BLOCK_D_MODEL_POW2,  # head dim
        MASK=False,  # causal masking
        ENABLE_DROPOUT=ENABLE_DROPOUT,  # activate dropout
        IS_FP8=IS_FP8,
        FP8_MAX=FP8_MAX,
        workgroup_id=seq_k_blk_idx,
    )

    # Write back dV and dK.
    offs_dkdv = (
        batch_idx * stride_dk_b
        + head_k_idx * stride_dk_h
        + k_start * stride_dk_n
        + offs_n[:, None] * stride_dk_n
        + offs_k[None, :] * stride_dk_k
    )
    tl.atomic_add(dv_ptr + offs_dkdv, dv, mask=mask_kv, sem="relaxed")
    dk *= sm_scale
    tl.atomic_add(dk_ptr + offs_dkdv, dk, mask=mask_kv, sem="relaxed")


@triton.jit
def _bwd_kernel_dkdvdq_noncausal(
    Q,
    K,
    V,
    sm_scale,
    DO,
    DK,
    DV,
    DQ,
    M,
    Delta,
    stride_qb_in,
    stride_qh_in,
    stride_qm_in,
    stride_qk_in,
    stride_kb_in,
    stride_kh_in,
    stride_kn_in,
    stride_kk_in,
    stride_vb_in,
    stride_vh_in,
    stride_vn_in,
    stride_vk_in,
    stride_dkb_in,
    stride_dkh_in,
    stride_dkn_in,
    stride_dkk_in,
    stride_dqb_in,
    stride_dqh_in,
    stride_dqm_in,
    stride_dqk_in,
    stride_deltab_in,
    stride_deltah_in,
    stride_deltam_in,
    stride_dob_in,
    stride_doh_in,
    stride_dom_in,
    stride_dok_in,
    stride_dropoutb_in,
    stride_dropouth_in,
    stride_dropoutm_in,
    stride_dropoutn_in,
    stride_descale_q_z_in,
    stride_descale_k_z_in,
    stride_descale_v_z_in,
    stride_descale_do_z_in,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    dropout_mask,
    dropout_p,
    philox_seed,
    philox_offset,
    descale_q_ptr,
    descale_k_ptr,
    descale_v_ptr,
    descale_do_ptr,
    NUM_Q_HEADS: tl.constexpr,
    NUM_K_HEADS: tl.constexpr,
    BATCH,
    NUM_K_PIDS,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLK_SLICE_FACTOR: tl.constexpr,
    BLOCK_D_MODEL: tl.constexpr,
    BLOCK_D_MODEL_POW2: tl.constexpr,
    ENABLE_DROPOUT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    IS_FP8: tl.constexpr,
    FP8_MAX: tl.constexpr,
    NUM_SMS: tl.constexpr,
    USE_INT64_STRIDES: tl.constexpr,
):
    if USE_INT64_STRIDES:
        stride_qb = tl.cast(stride_qb_in, tl.int64)
        stride_qh = tl.cast(stride_qh_in, tl.int64)
        stride_qm = tl.cast(stride_qm_in, tl.int64)
        stride_qk = tl.cast(stride_qk_in, tl.int64)
        stride_kb = tl.cast(stride_kb_in, tl.int64)
        stride_kh = tl.cast(stride_kh_in, tl.int64)
        stride_kn = tl.cast(stride_kn_in, tl.int64)
        stride_kk = tl.cast(stride_kk_in, tl.int64)
        stride_vb = tl.cast(stride_vb_in, tl.int64)
        stride_vh = tl.cast(stride_vh_in, tl.int64)
        stride_vn = tl.cast(stride_vn_in, tl.int64)
        stride_vk = tl.cast(stride_vk_in, tl.int64)
        stride_dkb = tl.cast(stride_dkb_in, tl.int64)
        stride_dkh = tl.cast(stride_dkh_in, tl.int64)
        stride_dkn = tl.cast(stride_dkn_in, tl.int64)
        stride_dkk = tl.cast(stride_dkk_in, tl.int64)
        stride_dqb = tl.cast(stride_dqb_in, tl.int64)
        stride_dqh = tl.cast(stride_dqh_in, tl.int64)
        stride_dqm = tl.cast(stride_dqm_in, tl.int64)
        stride_dqk = tl.cast(stride_dqk_in, tl.int64)
        stride_deltab = tl.cast(stride_deltab_in, tl.int64)
        stride_deltah = tl.cast(stride_deltah_in, tl.int64)
        stride_deltam = tl.cast(stride_deltam_in, tl.int64)
        stride_dob = tl.cast(stride_dob_in, tl.int64)
        stride_doh = tl.cast(stride_doh_in, tl.int64)
        stride_dom = tl.cast(stride_dom_in, tl.int64)
        stride_dok = tl.cast(stride_dok_in, tl.int64)
        stride_dropoutb = tl.cast(stride_dropoutb_in, tl.int64)
        stride_dropouth = tl.cast(stride_dropouth_in, tl.int64)
        stride_dropoutm = tl.cast(stride_dropoutm_in, tl.int64)
        stride_dropoutn = tl.cast(stride_dropoutn_in, tl.int64)
        stride_descale_q_z = tl.cast(stride_descale_q_z_in, tl.int64)
        stride_descale_k_z = tl.cast(stride_descale_k_z_in, tl.int64)
        stride_descale_v_z = tl.cast(stride_descale_v_z_in, tl.int64)
        stride_descale_do_z = tl.cast(stride_descale_do_z_in, tl.int64)
    else:
        stride_qb = stride_qb_in
        stride_qh = stride_qh_in
        stride_qm = stride_qm_in
        stride_qk = stride_qk_in
        stride_kb = stride_kb_in
        stride_kh = stride_kh_in
        stride_kn = stride_kn_in
        stride_kk = stride_kk_in
        stride_vb = stride_vb_in
        stride_vh = stride_vh_in
        stride_vn = stride_vn_in
        stride_vk = stride_vk_in
        stride_dkb = stride_dkb_in
        stride_dkh = stride_dkh_in
        stride_dkn = stride_dkn_in
        stride_dkk = stride_dkk_in
        stride_dqb = stride_dqb_in
        stride_dqh = stride_dqh_in
        stride_dqm = stride_dqm_in
        stride_dqk = stride_dqk_in
        stride_deltab = stride_deltab_in
        stride_deltah = stride_deltah_in
        stride_deltam = stride_deltam_in
        stride_dob = stride_dob_in
        stride_doh = stride_doh_in
        stride_dom = stride_dom_in
        stride_dok = stride_dok_in
        stride_dropoutb = stride_dropoutb_in
        stride_dropouth = stride_dropouth_in
        stride_dropoutm = stride_dropoutm_in
        stride_dropoutn = stride_dropoutn_in
        stride_descale_q_z = stride_descale_q_z_in
        stride_descale_k_z = stride_descale_k_z_in
        stride_descale_v_z = stride_descale_v_z_in
        stride_descale_do_z = stride_descale_do_z_in

    # workgroup id
    wid = tl.program_id(0)  # 0, ..., NUM_K_PIDS * BATCH * NUM_K_HEADS - 1

    # Workgroups get launched first along batch dim, then in head_k dim, and then in seq k block dim
    # This is in order to avoid contention for the tl.atomic_add (inside _bwd_dkdvdq_inner) that happens between workgroups that share the same batch and head_k.
    bid = wid % BATCH
    hkid = wid // BATCH % NUM_K_HEADS
    pid = wid // (BATCH * NUM_K_HEADS) % NUM_K_PIDS

    q_start = 0
    k_start = 0
    seqlen_q = max_seqlen_q
    seqlen_k = max_seqlen_k

    if IS_VARLEN:
        q_start = tl.load(cu_seqlens_q + bid)
        q_end = tl.load(cu_seqlens_q + bid + 1)
        k_start = tl.load(cu_seqlens_k + bid)
        k_end = tl.load(cu_seqlens_k + bid + 1)
        seqlen_q = q_end - q_start
        seqlen_k = k_end - k_start

    dk = tl.zeros([BLOCK_N, BLOCK_D_MODEL_POW2], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, BLOCK_D_MODEL_POW2], dtype=tl.float32)

    start_n = pid * BLOCK_N

    offs_k = tl.arange(0, BLOCK_D_MODEL_POW2)
    offs_n = start_n + tl.arange(0, BLOCK_N)
    mask_kv = offs_n[:, None] < seqlen_k
    PADDED_HEAD: tl.constexpr = BLOCK_D_MODEL != BLOCK_D_MODEL_POW2
    if PADDED_HEAD:
        mask_kv &= offs_k < BLOCK_D_MODEL

    GROUP_SIZE = NUM_Q_HEADS // NUM_K_HEADS
    adj_k = (
        bid * stride_kb
        + hkid * stride_kh
        + k_start * stride_kn
        + offs_n[:, None] * stride_kn
        + offs_k[None, :] * stride_kk
    )
    adj_v = (
        bid * stride_vb
        + hkid * stride_vh
        + k_start * stride_vn
        + offs_n[:, None] * stride_vn
        + offs_k[None, :] * stride_vk
    )

    k = tl.load(K + adj_k, mask=mask_kv, other=0.0)
    v = tl.load(V + adj_v, mask=mask_kv, other=0.0)

    for hqid in range(hkid * GROUP_SIZE, hkid * GROUP_SIZE + GROUP_SIZE):
        adj_q = bid * stride_qb + hqid * stride_qh + q_start * stride_qm
        adj_dq = bid * stride_dqb + hqid * stride_dqh + q_start * stride_dqm

        Q_ptr = Q + adj_q
        DQ_ptr = DQ + adj_dq

        adj_do = bid * stride_dob + hqid * stride_doh + q_start * stride_dom
        DO_ptr = DO + adj_do
        adj_delta = bid * stride_deltab + hqid * stride_deltah + q_start * stride_deltam
        M_ptr = M + adj_delta
        Delta_ptr = Delta + adj_delta

        # dropout
        batch_philox_offset = 0
        dropout_offset = 0
        if ENABLE_DROPOUT:
            batch_philox_offset = (
                philox_offset + bid * stride_dropoutb + hqid * stride_dropouth
            )
            dropout_offset = (
                dropout_mask + bid * stride_dropoutb + hqid * stride_dropouth
            )

        if IS_FP8:
            descale_q = tl.load(descale_q_ptr + bid * stride_descale_q_z + hqid)
            descale_k = tl.load(descale_k_ptr + bid * stride_descale_k_z + hkid)
            descale_v = tl.load(descale_v_ptr + bid * stride_descale_v_z + hkid)
            descale_do = tl.load(descale_do_ptr + bid * stride_descale_do_z + hqid)
        else:
            descale_q, descale_k, descale_v, descale_do = 1.0, 1.0, 1.0, 1.0

        start_m = 0
        num_steps = tl.cdiv(seqlen_q, BLOCK_M)

        dk, dv = _bwd_dkdvdq_inner(
            dk,
            dv,
            Q_ptr,
            k,
            v,
            DO_ptr,
            DQ_ptr,
            M_ptr,
            Delta_ptr,
            sm_scale,
            stride_qm,
            stride_qk,
            stride_dqm,
            stride_dqk,
            stride_dom,
            stride_dok,
            stride_dropoutm,
            stride_dropoutn,
            stride_deltam,
            dropout_p,
            philox_seed,
            batch_philox_offset,
            dropout_offset,
            seqlen_q,
            seqlen_k,
            start_n,
            start_m,
            num_steps,
            descale_q,
            descale_k,
            descale_v,
            descale_do,
            BLOCK_M,
            BLOCK_N,
            BLOCK_D_MODEL,
            BLOCK_D_MODEL_POW2,
            MASK=False,
            ENABLE_DROPOUT=ENABLE_DROPOUT,
            IS_FP8=IS_FP8,
            FP8_MAX=FP8_MAX,
            workgroup_id=wid,
        )

    adj_dkdv = (
        bid * stride_dkb
        + hkid * stride_dkh
        + k_start * stride_dkn
        + offs_n[:, None] * stride_dkn
        + offs_k[None, :] * stride_dkk
    )
    tl.store(DV + adj_dkdv, dv, mask=mask_kv)
    dk *= sm_scale
    tl.store(DK + adj_dkdv, dk, mask=mask_kv)


def flash_attn_fused_backward(
    do: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    softmax_lse: torch.Tensor,
    dq: torch.Tensor,
    dk: torch.Tensor,
    dv: torch.Tensor,
    dbias: torch.Tensor,
    sm_scale: float,
    alibi_slopes: Optional[torch.Tensor],
    causal: bool,
    cu_seqlens_q: Optional[torch.Tensor],
    cu_seqlens_k: Optional[torch.Tensor],
    max_seqlen_q: int,
    max_seqlen_k: int,
    dropout_p: float,
    philox_seed: Optional[int] = 0,
    philox_offset: Optional[int] = 0,
    descale_q: Optional[torch.Tensor] = None,
    descale_k: Optional[torch.Tensor] = None,
    descale_v: Optional[torch.Tensor] = None,
    descale_do: Optional[torch.Tensor] = None,
    USE_INT64_STRIDES: Optional[bool] = False,
):
    if dbias is not None:
        raise ValueError("Bias is not supported yet in the Triton Backend")

    IS_FP8 = is_fp8(q)
    if IS_FP8:
        FP8_MAX = torch.finfo(q.dtype).max
        descale_strides = (
            descale_q.stride(0),
            descale_k.stride(0),
            descale_v.stride(0),
            descale_do.stride(0),
        )
    else:
        FP8_MAX = None
        stride_descale_q_z = stride_descale_k_z = stride_descale_v_z = (
            stride_descale_do_z
        ) = None
        descale_strides = (
            stride_descale_q_z,
            stride_descale_k_z,
            stride_descale_v_z,
            stride_descale_do_z,
        )

    IS_VARLEN = True if cu_seqlens_q is not None else False

    # get strides and shape
    if IS_VARLEN:
        # Layout for q,k,v is thd ie [total tokens, num_head, head_dim]
        batch, seqlen_q, num_q_heads, head_sz = (
            len(cu_seqlens_q) - 1,
            max_seqlen_q,
            q.shape[1],
            q.shape[2],
        )
        _, num_k_heads = max_seqlen_k, k.shape[1]
        q_strides = (0, q.stride(1), q.stride(0), q.stride(2))
        q_strides = (0, q.stride(1), q.stride(0), q.stride(2))
        k_strides = (0, k.stride(1), k.stride(0), k.stride(2))
        v_strides = (0, v.stride(1), v.stride(0), v.stride(2))
        o_strides = (0, o.stride(1), o.stride(0), o.stride(2))
        dq_strides = (0, dq.stride(1), dq.stride(0), dq.stride(2))
        dk_strides = (0, dk.stride(1), dk.stride(0), dk.stride(2))
        do_strides = (0, do.stride(1), do.stride(0), do.stride(2))
    else:
        # Layout for q,k,v is bshd ie [batch, seq_len, num_head, head_dim]
        batch, seqlen_q, num_q_heads, head_sz = q.shape
        _, num_k_heads = k.shape[1], k.shape[2]
        q_strides = (q.stride(0), q.stride(2), q.stride(1), q.stride(3))
        k_strides = (k.stride(0), k.stride(2), k.stride(1), k.stride(3))
        v_strides = (v.stride(0), v.stride(2), v.stride(1), v.stride(3))
        o_strides = (o.stride(0), o.stride(2), o.stride(1), o.stride(3))
        dq_strides = (dq.stride(0), dq.stride(2), dq.stride(1), dq.stride(3))
        dk_strides = (dk.stride(0), dk.stride(2), dk.stride(1), dk.stride(3))
        do_strides = (do.stride(0), do.stride(2), do.stride(1), do.stride(3))

    # BLOCK_D_MODEL, BLOCK_D_MODEL_POW2
    # padding for head_dim. Power of 2 or 16
    BLOCK_D_MODEL_POW2 = triton.next_power_of_2(head_sz)
    BLOCK_D_MODEL_POW2 = max(BLOCK_D_MODEL_POW2, 16)

    # init delta
    delta = torch.zeros_like(softmax_lse)
    if IS_VARLEN:
        # [total_tokens, num_q_heads, seqlen_q]
        delta_strides = (0, delta.stride(1), delta.stride(0))
    else:
        # [batch, num_q_heads, seqlen_q]
        delta_strides = delta.stride()

    # preprocess
    # compute D(delta) = rowsum(dO*O). Note, multiplication is element-wise.
    PRE_BLOCK = 128
    pre_grid = (triton.cdiv(max_seqlen_q, PRE_BLOCK), batch, num_q_heads)

    _bwd_preprocess[pre_grid](
        o,
        do,
        delta,
        *o_strides,
        *delta_strides,
        descale_strides[3],
        cu_seqlens_q,
        max_seqlen_q,
        descale_do,
        BLOCK_M=PRE_BLOCK,
        BLOCK_D_MODEL=head_sz,
        BLOCK_D_MODEL_POW2=BLOCK_D_MODEL_POW2,
        IS_VARLEN=IS_VARLEN,
        IS_FP8=IS_FP8,
    )
    # dropout_mask
    use_dropout = dropout_p > 0.0
    if use_dropout:
        dropout_mask = torch.zeros(
            (batch, num_q_heads, max_seqlen_q, max_seqlen_k),
            device=q.device,
            dtype=torch.float32,
        )
        dropout_strides = dropout_mask.stride()
    else:
        dropout_mask = None
        dropout_strides = (0, 0, 0, 0)

    # Fuses dk,dv and dq computations into one kernel using atomics
    BLOCK_N = (
        64 if ((BLOCK_D_MODEL_POW2 > 160) or (q.dtype == torch.float32)) else 128
    )  # larger head sizes lead to oom
    config = {
        "BLOCK_M": 16,
        "BLOCK_N": BLOCK_N,
        "num_warps": 8,
        "num_stages": 1,
        "waves_per_eu": 2,
        "BLK_SLICE_FACTOR": 1,
        "matrix_instr_nonkdim": 16,
    }

    num_k_pids = (max_seqlen_k + BLOCK_N - 1) // BLOCK_N
    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
    if causal:
        grid_dkdvdq = (batch * num_q_heads * num_k_pids,)

        _bwd_kernel_dkdvdq_causal[grid_dkdvdq](
            q,
            k,
            v,
            sm_scale,
            do,
            dk,
            dv,
            dq,
            softmax_lse,
            delta,
            *q_strides,
            *k_strides,
            *v_strides,
            *dk_strides,
            *dq_strides,
            *delta_strides,
            *do_strides,
            *dropout_strides,
            *descale_strides,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            dropout_mask,
            dropout_p,
            philox_seed,
            philox_offset,
            descale_q,
            descale_k,
            descale_v,
            descale_do,
            NUM_Q_HEADS=num_q_heads,
            NUM_K_HEADS=num_k_heads,
            BATCH=batch,
            NUM_K_PIDS=num_k_pids,
            BLOCK_D_MODEL=head_sz,
            BLOCK_D_MODEL_POW2=BLOCK_D_MODEL_POW2,
            ENABLE_DROPOUT=use_dropout,
            IS_VARLEN=IS_VARLEN,
            IS_FP8=IS_FP8,
            FP8_MAX=FP8_MAX,
            NUM_SMS=NUM_SMS,
            USE_INT64_STRIDES=USE_INT64_STRIDES,
            **config,
        )
    else:
        # in non causal inner loop over grouped q heads
        grid_dkdvdq = (batch * num_k_heads * num_k_pids,)
        _bwd_kernel_dkdvdq_noncausal[grid_dkdvdq](
            q,
            k,
            v,
            sm_scale,
            do,
            dk,
            dv,
            dq,
            softmax_lse,
            delta,
            *q_strides,
            *k_strides,
            *v_strides,
            *dk_strides,
            *dq_strides,
            *delta_strides,
            *do_strides,
            *dropout_strides,
            *descale_strides,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            dropout_mask,
            dropout_p,
            philox_seed,
            philox_offset,
            descale_q,
            descale_k,
            descale_v,
            descale_do,
            NUM_Q_HEADS=num_q_heads,
            NUM_K_HEADS=num_k_heads,
            BATCH=batch,
            NUM_K_PIDS=num_k_pids,
            BLOCK_D_MODEL=head_sz,
            BLOCK_D_MODEL_POW2=BLOCK_D_MODEL_POW2,
            ENABLE_DROPOUT=use_dropout,
            IS_VARLEN=IS_VARLEN,
            IS_FP8=IS_FP8,
            FP8_MAX=FP8_MAX,
            NUM_SMS=NUM_SMS,
            USE_INT64_STRIDES=USE_INT64_STRIDES,
            **config,
        )

    return delta
