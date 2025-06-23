# SPDX-License-Identifier: MIT
# Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

import torch
import triton  # type: ignore
import triton.language as tl  # type: ignore
from typing import Optional
from aiter.ops.triton.utils.mha_kernel_utils import (
    compute_fp8_scaling_factors,
    is_fp8,
)


# NOTE: triton fails to import tl.constexprs so create them here for the file
DROPOUT_USE_PYTORCH = False
DROPOUT_DUMP = False

tl_DROPOUT_USE_PYTORCH: tl.constexpr = triton.language.constexpr(DROPOUT_USE_PYTORCH)
tl_DROPOUT_DUMP: tl.constexpr = triton.language.constexpr(DROPOUT_DUMP)


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


# The main inner-loop logic for computing dK and dV.
@triton.jit
def _bwd_dkdv_inner(
    dk,
    dv,  # output
    Q,
    k,
    v,
    DO,
    M,
    D,
    sm_scale,  # input tensor
    stride_qm,
    stride_qk,
    stride_dom,
    stride_dok,
    stride_dropoutm,
    stride_dropoutn,
    stride_deltam,
    BLOCK_M: tl.constexpr,  # 16
    BLOCK_N: tl.constexpr,  # 128
    HEAD_DIM: tl.constexpr,  #
    ACTUAL_HEAD_DIM: tl.constexpr,  #
    dropout_p,
    philox_seed,
    batch_philox_offset,
    dropout_offset,
    alibi_slope,
    seqlen_q,
    seqlen_k,  # max sequence length for q and k
    # Filled in by the wrapper.
    start_n,
    start_m,
    num_steps,  # iteration numbers
    descale_q,
    descale_k,
    descale_v,
    descale_do,  # fp8 descale factors from user
    MASK: tl.constexpr,  # causal masking, only apply to tiles on mask diagonal
    ENABLE_DROPOUT: tl.constexpr,  # activate dropout
    USE_ALIBI: tl.constexpr,
    USE_EXP2: tl.constexpr,  # activate exp2
    IS_FP8: tl.constexpr,
    FP8_MAX: tl.constexpr,
    DEBUG_TRITON: tl.constexpr,
    DEBUG_TRITON_DETAIL: tl.constexpr,
):
    # if HEAD_DIM is padded
    PADDED_HEAD: tl.constexpr = ACTUAL_HEAD_DIM != HEAD_DIM
    delta_qk = seqlen_q - seqlen_k
    offs_m = start_m + tl.arange(0, BLOCK_M)  # start_m + (0, 15)
    offs_n = start_n + tl.arange(0, BLOCK_N)  # start_m + (0, 127)
    offs_k = tl.arange(0, HEAD_DIM)
    # mask to make sure not OOB of seqlen_q
    mask_n = offs_n < seqlen_k
    # Q and DO are (seqlen_q, head_dim)
    # qT_ptrs = (1, BLOCK_M) + (HEAD_DIM, 1), transpose of q
    qT_ptrs = Q + offs_m[None, :] * stride_qm + offs_k[:, None] * stride_qk
    # do_ptrs = (BLOCK_M, 1) + (1, HEAD_DIM), NOT transposed
    do_ptrs = DO + offs_m[:, None] * stride_dom + offs_k[None, :] * stride_dok
    # BLOCK_N must be a multiple of BLOCK_M, otherwise the code wouldn't work.
    tl.static_assert(BLOCK_N % BLOCK_M == 0)
    curr_m = start_m
    step_m = BLOCK_M
    curr_philox_offset = batch_philox_offset
    curr_dropout_offset = dropout_offset
    RCP_LN2: tl.constexpr = 1.4426950408889634  # = 1.0 / ln(2)

    for blk_idx in range(num_steps):
        if DEBUG_TRITON:
            print(f"iter {blk_idx}: curr_m = {curr_m}")  # noqa: E701
        offs_m = curr_m + tl.arange(0, BLOCK_M)
        # update the mask because offs_m advanced
        mask_m = offs_m < seqlen_q
        mask_qT = mask_m[None, :]
        mask_do = mask_m[:, None]
        mask_nm = mask_n[:, None] & (offs_m[None, :] < seqlen_q)
        if PADDED_HEAD:
            mask_qT &= offs_k[:, None] < ACTUAL_HEAD_DIM
            mask_do &= offs_k[None, :] < ACTUAL_HEAD_DIM
        qT = tl.load(qT_ptrs, mask=mask_qT, other=0.0)
        # generate dropout mask
        if ENABLE_DROPOUT:
            # NOTE: dropout is transposed because it is used to mask pT
            philox_offs = (
                curr_philox_offset
                + offs_m[None, :] * stride_dropoutm
                + offs_n[:, None] * stride_dropoutn
            )
            if tl_DROPOUT_USE_PYTORCH:
                dropout_offs = (
                    offs_m[None, :] * stride_dropoutm
                    + offs_n[:, None] * stride_dropoutn
                )
                dropout_mask = tl.load(curr_dropout_offset + dropout_offs, mask=mask_nm)
            else:
                rand_vals = tl.rand(philox_seed, philox_offs)
                dropout_mask = rand_vals > dropout_p
            dropout_scale = 1.0 / (1 - dropout_p)
        # Load m before computing qk to reduce pipeline stall.
        m = tl.load(M + offs_m * stride_deltam, mask=mask_m, other=0.0)
        if IS_FP8:
            qkT = tl.dot(k, qT) * descale_q * descale_k
        else:
            qkT = tl.dot(k, qT)
        qkT_scaled = qkT * sm_scale

        if USE_ALIBI:
            relative_pos_block = offs_n[:, None] + seqlen_q - seqlen_k - offs_m[None, :]
            alibi_block = -1 * alibi_slope * tl.abs(relative_pos_block)
            qkT_scaled += alibi_block

        if DEBUG_TRITON_DETAIL:
            if start_n == 256:
                print(f"qT: {qT.shape}\n", qT)
                print(f"k: {k.shape}\n", k)
                print(f"qkT scaled: {qkT.shape}\n", qkT_scaled)
        # TODO: remove the scaling of m later when we removed re-scaling in fwd
        if USE_EXP2:
            pT = tl.math.exp2(qkT_scaled * RCP_LN2 - m[None, :] * RCP_LN2)
        else:
            pT = tl.math.exp(qkT_scaled - m[None, :])

        # Autoregressive masking.
        if MASK:
            # offset offs_m with delta_qk since the causal mask starts at
            # bottom right of the (seqlen_q, seqlen_k) matrix
            causal_mask = (offs_m[None, :] - delta_qk) >= offs_n[:, None]
            mask = causal_mask & mask_nm
            if DEBUG_TRITON_DETAIL:
                if start_n == 256:
                    print(f"causal_mask: {causal_mask.shape}\n", causal_mask)
                    print(
                        f"qkT after causal: {qkT.shape}\n",
                        tl.where(causal_mask, qkT * sm_scale, 0.0),
                    )
            pT = tl.where(mask, pT, 0.0)
        do = tl.load(do_ptrs, mask=mask_do, other=0.0)
        # Compute dV.
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

        if DEBUG_TRITON_DETAIL:
            if start_n == 256:
                print(f"pT: {pT.shape}\n", pT)
        # D (= delta) is pre-divided by ds_scale.
        Di = tl.load(D + offs_m * stride_deltam, mask=mask_m)
        # Compute dP and dS.
        if IS_FP8:
            dpT = tl.dot(v, tl.trans(do)) * descale_v * descale_do
        else:
            dpT = tl.dot(v, tl.trans(do))
        if ENABLE_DROPOUT:
            dpT = tl.where(dropout_mask, dpT, 0.0) * dropout_scale
        delta_i = Di[None, :]
        dsT = pT * (dpT - delta_i)
        if IS_FP8:
            scale_dsT, descale_dsT = compute_fp8_scaling_factors(dsT, FP8_MAX)
            dk += (
                tl.dot((dsT * scale_dsT).to(qT.type.element_ty), tl.trans(qT))
                * descale_dsT
                * descale_q
            )
        else:
            dk += tl.dot(dsT.to(qT.type.element_ty), tl.trans(qT))
        # Increment pointers.
        curr_m += step_m
        qT_ptrs += step_m * stride_qm
        do_ptrs += step_m * stride_dom
    return dk, dv


# the main inner-loop logic for computing dQ
@triton.jit
def _bwd_dq_inner(
    dq,  # output
    q,
    K,
    V,
    do,
    m,
    Delta,
    sm_scale,  # input
    # shared by Q/K/V.
    stride_qm,
    stride_qk,
    stride_kn,
    stride_kk,
    stride_vn,
    stride_vk,
    stride_dropoutm,
    stride_dropoutn,  # stride for dropout
    stride_deltam,
    seqlen_q,
    seqlen_k,  #
    BLOCK_M2: tl.constexpr,  #
    BLOCK_N2: tl.constexpr,  #
    HEAD_DIM: tl.constexpr,
    ACTUAL_HEAD_DIM: tl.constexpr,  #
    dropout_p,
    philox_seed,
    batch_philox_offset,
    dropout_offset,
    alibi_slope,
    # Filled in by the wrapper.
    start_m,
    start_n,
    end_n,
    num_steps,  #
    descale_q,
    descale_k,
    descale_v,
    descale_do,  # fp8 descale factors from user
    MASK: tl.constexpr,
    ENABLE_DROPOUT: tl.constexpr,
    USE_ALIBI: tl.constexpr,
    USE_EXP2: tl.constexpr,
    IS_FP8: tl.constexpr,
    FP8_MAX: tl.constexpr,
    DEBUG_TRITON: tl.constexpr,
    DEBUG_TRITON_DETAIL: tl.constexpr,
):
    # if HEAD_DIM is padded
    PADDED_HEAD: tl.constexpr = ACTUAL_HEAD_DIM != HEAD_DIM
    delta_qk = seqlen_q - seqlen_k
    offs_m = start_m + tl.arange(0, BLOCK_M2)
    offs_n = start_n + tl.arange(0, BLOCK_N2)
    offs_k = tl.arange(0, HEAD_DIM)

    # mask to make sure not OOB of seqlen_q
    mask_m = offs_m < seqlen_q

    kT_ptrs = K + offs_n[None, :] * stride_kn + offs_k[:, None] * stride_kk
    vT_ptrs = V + offs_n[None, :] * stride_vn + offs_k[:, None] * stride_vk
    # D (= delta) is pre-divided by ds_scale.
    Di = tl.load(Delta + offs_m * stride_deltam, mask=mask_m, other=0.0)
    # BLOCK_M2 must be a multiple of BLOCK_N2, otherwise the code wouldn't work.
    tl.static_assert(BLOCK_M2 % BLOCK_N2 == 0)
    curr_n = start_n
    step_n = BLOCK_N2
    curr_philox_offset = batch_philox_offset
    curr_dropout_offset = dropout_offset
    RCP_LN2: tl.constexpr = 1.4426950408889634  # = 1.0 / ln(2)
    for blk_idx in range(num_steps):
        if DEBUG_TRITON:
            print(f"iter {blk_idx}: curr_n = {curr_n}")  # noqa: E701
        offs_n = curr_n + tl.arange(0, BLOCK_N2)
        # end_n is needed because the end of causal True might not be perfectly
        # aligned with the end of the block
        mask_n = offs_n < end_n
        if DEBUG_TRITON_DETAIL:
            print(
                f"start_n = {start_n}, end_n = {end_n}, offs_n: {offs_n.shape}\n{offs_n}"
            )  # noqa: E701
        if DEBUG_TRITON_DETAIL:
            print(f"mask_n: {mask_n.shape}\n{mask_n}")  # noqa: E701
        mask_kT = mask_n[None, :]
        mask_mn = mask_m[:, None] & (offs_n[None, :] < end_n)
        if PADDED_HEAD:
            mask_kT &= offs_k[:, None] < ACTUAL_HEAD_DIM

        kT = tl.load(kT_ptrs, mask=mask_kT, other=0.0)
        vT = tl.load(vT_ptrs, mask=mask_kT, other=0.0)

        if ENABLE_DROPOUT:
            # NOTE: dropout is transposed because it is used to mask pT
            philox_offs = (
                curr_philox_offset
                + offs_m[:, None] * stride_dropoutm
                + offs_n[None, :] * stride_dropoutn
            )
            if tl_DROPOUT_USE_PYTORCH:
                dropout_offs = (
                    offs_m[:, None] * stride_dropoutm
                    + offs_n[None, :] * stride_dropoutn
                )
                dropout_mask = tl.load(curr_dropout_offset + dropout_offs, mask=mask_mn)
            else:
                rand_vals = tl.rand(philox_seed, philox_offs)
                dropout_mask = rand_vals > dropout_p
            dropout_scale = 1 / (1 - dropout_p)

        if IS_FP8:
            qk = tl.dot(q, kT) * descale_q * descale_k
        else:
            qk = tl.dot(q, kT)
        qk_scaled = qk * sm_scale

        if USE_ALIBI:
            relative_pos_block = offs_m[:, None] + seqlen_k - seqlen_q - offs_n[None, :]
            alibi_block = -1 * alibi_slope * tl.abs(relative_pos_block)
            qk_scaled += alibi_block

        if DEBUG_TRITON_DETAIL:
            print(f"qk scaled: {qk.shape}\n", qk_scaled)  # noqa: E701
        if USE_EXP2:
            p = tl.math.exp2(qk_scaled * RCP_LN2 - m * RCP_LN2)
        else:
            p = tl.math.exp(qk_scaled - m)

        # Autoregressive masking.
        if MASK:
            causal_mask = (offs_m[:, None] - delta_qk) >= offs_n[None, :]
            mask = causal_mask & mask_mn
            p = tl.where(mask, p, 0.0)
        # Compute dP and dS.
        if IS_FP8:
            dp = tl.dot(do, vT) * descale_do * descale_v
        else:
            dp = tl.dot(do, vT)
        if ENABLE_DROPOUT:
            dp = tl.where(dropout_mask, dp, 0.0) * dropout_scale
        delta_i = Di[:, None]
        ds = p * (dp - delta_i)
        # Compute dQ.
        # NOTE: We need to de-scale dq in the end, because kT was pre-scaled.
        if IS_FP8:
            scale_ds, descale_ds = compute_fp8_scaling_factors(ds, FP8_MAX)
            dq += (
                tl.dot((ds * scale_ds).to(kT.type.element_ty), tl.trans(kT))
                * descale_ds
                * descale_k
            )
        else:
            dq += tl.dot(ds.to(kT.type.element_ty), tl.trans(kT))
        # Increment pointers.
        curr_n += step_n
        kT_ptrs += step_n * stride_kn
        vT_ptrs += step_n * stride_vn
    return dq


# @triton.autotune(
#     configs=causal_autotune_configs,
#     key=causal_autotune_keys,
#     use_cuda_graph=True,
# )
@triton.jit
def bwd_kernel_causal(  # grid = (tl.cdiv(max_seqlen_q // BLOCK_M2), batch, nheads_q)
    Q,
    K,
    V,
    sm_scale,
    DO,
    DQ,
    DK,
    DV,
    M,
    Delta,
    stride_qb_in,
    stride_qh_in,
    stride_qm_in,
    stride_qd_in,
    stride_kb_in,
    stride_kh_in,
    stride_kn_in,
    stride_kd_in,
    stride_vb_in,
    stride_vh_in,
    stride_vn_in,
    stride_vd_in,
    stride_dqb_in,
    stride_dqh_in,
    stride_dqm_in,
    stride_dqd_in,
    stride_dkb_in,
    stride_dkh_in,
    stride_dkn_in,
    stride_dkd_in,
    stride_dvb_in,
    stride_dvh_in,
    stride_dvn_in,
    stride_dvd_in,
    stride_deltab_in,
    stride_deltah_in,
    stride_deltam_in,
    stride_dob_in,
    stride_doh_in,
    stride_dom_in,
    stride_dod_in,
    stride_dropoutb_in,
    stride_dropouth_in,
    stride_dropoutm_in,
    stride_dropoutn_in,
    stride_descale_q_z_in,
    stride_descale_k_z_in,
    stride_descale_v_z_in,
    stride_descale_do_z_in,
    stride_az_in,
    stride_ah_in,
    HQ,
    HK,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    Dropout_mask,
    dropout_p,
    philox_seed,
    philox_offset_base_in,
    Alibi_slopes,
    Descale_q,
    Descale_k,
    Descale_v,
    Descale_do,
    BLOCK_M1: tl.constexpr,
    BLOCK_N1: tl.constexpr,
    BLOCK_M2: tl.constexpr,
    BLOCK_N2: tl.constexpr,
    BLK_SLICE_FACTOR: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    ACTUAL_HEAD_DIM: tl.constexpr,
    ENABLE_DROPOUT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_ALIBI: tl.constexpr,
    USE_EXP2: tl.constexpr,
    IS_FP8: tl.constexpr,
    FP8_MAX: tl.constexpr,
    FP8_OUTPUT: tl.constexpr,
    DEBUG_TRITON: tl.constexpr,
    DEBUG_TRITON_DETAIL: tl.constexpr,
    USE_INT64_STRIDES: tl.constexpr,
):
    if USE_INT64_STRIDES:
        stride_qb = tl.cast(stride_qb_in, tl.int64)
        stride_qh = tl.cast(stride_qh_in, tl.int64)
        stride_qm = tl.cast(stride_qm_in, tl.int64)
        stride_qd = tl.cast(stride_qd_in, tl.int64)
        stride_kb = tl.cast(stride_kb_in, tl.int64)
        stride_kh = tl.cast(stride_kh_in, tl.int64)
        stride_kn = tl.cast(stride_kn_in, tl.int64)
        stride_kd = tl.cast(stride_kd_in, tl.int64)
        stride_vb = tl.cast(stride_vb_in, tl.int64)
        stride_vh = tl.cast(stride_vh_in, tl.int64)
        stride_vn = tl.cast(stride_vn_in, tl.int64)
        stride_vd = tl.cast(stride_vd_in, tl.int64)
        stride_dqb = tl.cast(stride_dqb_in, tl.int64)
        stride_dqh = tl.cast(stride_dqh_in, tl.int64)
        stride_dqm = tl.cast(stride_dqm_in, tl.int64)
        stride_dqd = tl.cast(stride_dqd_in, tl.int64)
        stride_dkb = tl.cast(stride_dkb_in, tl.int64)
        stride_dkh = tl.cast(stride_dkh_in, tl.int64)
        stride_dkn = tl.cast(stride_dkn_in, tl.int64)
        stride_dkd = tl.cast(stride_dkd_in, tl.int64)
        stride_dvb = tl.cast(stride_dvb_in, tl.int64)
        stride_dvh = tl.cast(stride_dvh_in, tl.int64)
        stride_dvn = tl.cast(stride_dvn_in, tl.int64)
        stride_dvd = tl.cast(stride_dvd_in, tl.int64)
        stride_deltab = tl.cast(stride_deltab_in, tl.int64)
        stride_deltah = tl.cast(stride_deltah_in, tl.int64)
        stride_deltam = tl.cast(stride_deltam_in, tl.int64)
        stride_dob = tl.cast(stride_dob_in, tl.int64)
        stride_doh = tl.cast(stride_doh_in, tl.int64)
        stride_dom = tl.cast(stride_dom_in, tl.int64)
        stride_dod = tl.cast(stride_dod_in, tl.int64)
        philox_offset_base = tl.cast(philox_offset_base_in, tl.int64)
        stride_dropoutb = tl.cast(stride_dropoutb_in, tl.int64)
        stride_dropouth = tl.cast(stride_dropouth_in, tl.int64)
        stride_dropoutm = tl.cast(stride_dropoutm_in, tl.int64)
        stride_dropoutn = tl.cast(stride_dropoutn_in, tl.int64)
        if IS_FP8:
            stride_descale_q_z = tl.cast(stride_descale_q_z_in, tl.int64)
            stride_descale_k_z = tl.cast(stride_descale_k_z_in, tl.int64)
            stride_descale_v_z = tl.cast(stride_descale_v_z_in, tl.int64)
            stride_descale_do_z = tl.cast(stride_descale_do_z_in, tl.int64)
        stride_az = tl.cast(stride_az_in, tl.int64)
        stride_ah = tl.cast(stride_ah_in, tl.int64)
    else:
        stride_qb = stride_qb_in
        stride_qh = stride_qh_in
        stride_qm = stride_qm_in
        stride_qd = stride_qd_in
        stride_kb = stride_kb_in
        stride_kh = stride_kh_in
        stride_kn = stride_kn_in
        stride_kd = stride_kd_in
        stride_vb = stride_vb_in
        stride_vh = stride_vh_in
        stride_vn = stride_vn_in
        stride_vd = stride_vd_in
        stride_dqb = stride_dqb_in
        stride_dqh = stride_dqh_in
        stride_dqm = stride_dqm_in
        stride_dqd = stride_dqd_in
        stride_dkb = stride_dkb_in
        stride_dkh = stride_dkh_in
        stride_dkn = stride_dkn_in
        stride_dkd = stride_dkd_in
        stride_dvb = stride_dvb_in
        stride_dvh = stride_dvh_in
        stride_dvn = stride_dvn_in
        stride_dvd = stride_dvd_in
        stride_deltab = stride_deltab_in
        stride_deltah = stride_deltah_in
        stride_deltam = stride_deltam_in
        stride_dob = stride_dob_in
        stride_doh = stride_doh_in
        stride_dom = stride_dom_in
        stride_dod = stride_dod_in
        philox_offset_base = philox_offset_base_in
        stride_dropoutb = stride_dropoutb_in
        stride_dropouth = stride_dropouth_in
        stride_dropoutm = stride_dropoutm_in
        stride_dropoutn = stride_dropoutn_in
        stride_descale_q_z = stride_descale_q_z_in
        stride_descale_k_z = stride_descale_k_z_in
        stride_descale_v_z = stride_descale_v_z_in
        stride_descale_do_z = stride_descale_do_z_in
        stride_az = stride_az_in
        stride_ah = stride_ah_in

    # program ids
    hkid = tl.program_id(0)
    pid = tl.program_id(1)
    bid = tl.program_id(2)
    if DEBUG_TRITON:
        print(f"\npid: {pid}, bid: {bid}, hkid: {hkid}")  # noqa: E701
    # figure out varlen start and end
    q_start = 0
    k_start = 0
    seqlen_q = max_seqlen_q
    seqlen_k = max_seqlen_k
    if IS_VARLEN:
        # Compute actual sequence lengths
        q_start = tl.load(cu_seqlens_q + bid)
        q_end = tl.load(cu_seqlens_q + bid + 1)
        k_start = tl.load(cu_seqlens_k + bid)
        k_end = tl.load(cu_seqlens_k + bid + 1)
        seqlen_q = q_end - q_start
        seqlen_k = k_end - k_start

    delta_qk = seqlen_q - seqlen_k
    if DEBUG_TRITON:
        print(f"delta_qk = {delta_qk}")  # noqa: E701
    PADDED_HEAD: tl.constexpr = ACTUAL_HEAD_DIM != HEAD_DIM
    offs_d = tl.arange(0, HEAD_DIM)
    GROUP_SIZE: tl.constexpr = HQ // HK

    # align the delta_qk
    start_n = pid * BLOCK_N1
    if start_n < seqlen_k:
        # This section does dk and dv
        dk = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)
        dv = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)

        # q > k: diretcly skip all the way until the start of causal block
        start_delta_q_gt_k = delta_qk
        # q < k: some blocks will have no Masked block, other needs to re-calc
        # starting position
        # delta_qk is negative so flip it, only multiple of BLOCK_N can skip the
        # masked op
        num_blocks_skip = -delta_qk // BLOCK_N1
        delta_aligned = (num_blocks_skip + 1) * BLOCK_N1 + delta_qk
        start_delta_q_lt_k = delta_aligned // BLOCK_M1 * BLOCK_M1
        if delta_qk >= 0:
            start_delta = delta_qk
            if DEBUG_TRITON:
                print(
                    f"q >= k: start_delta = delta_qk aligned to BLOCK_M = {start_delta_q_gt_k}"
                )  # noqa: E701
        else:
            start_delta = start_delta_q_lt_k
            if DEBUG_TRITON:
                print(
                    f"q < k: start_delta = residue btw multiple BLOCK_N and delta_qk = {delta_aligned} = aligned to BLOCK_M = {start_delta_q_lt_k}"
                )  # noqa: E701

        offs_n = start_n + tl.arange(0, BLOCK_N1)
        # Mask for loading K and V
        mask_kv = offs_n[:, None] < seqlen_k
        if PADDED_HEAD:
            mask_d = offs_d < ACTUAL_HEAD_DIM
            mask_kv &= mask_d[None, :]

        # K/V tensors not changed for the group
        adj_k = (
            bid * stride_kb
            + hkid * stride_kh
            + k_start * stride_kn
            + offs_n[:, None] * stride_kn
            + offs_d[None, :] * stride_kd
        )
        adj_v = (
            bid * stride_vb
            + hkid * stride_vh
            + k_start * stride_vn
            + offs_n[:, None] * stride_vn
            + offs_d[None, :] * stride_vd
        )
        # load K and V: they stay in SRAM throughout the inner loop.
        k = tl.load(K + adj_k, mask=mask_kv, other=0.0)
        v = tl.load(V + adj_v, mask=mask_kv, other=0.0)
        # If MQA / GQA, set the K and V head offsets appropriately.
        # hqid = hkid
        for hqid in range(hkid * GROUP_SIZE, hkid * GROUP_SIZE + GROUP_SIZE):
            if delta_qk >= 0:
                start_m = start_n + start_delta
                len_m = BLOCK_N1
            else:
                start_m = max(start_n + delta_qk, 0)
                start_m = start_m // BLOCK_M1 * BLOCK_M1
                # because we might shift the masked blocks up, we are deeper into
                # the masked out region, so we would potentially increase the total
                # steps with masked operation to get out of it
                residue_m = max(start_n + delta_qk - start_m, 0)
                len_m = BLOCK_N1 + residue_m
                if DEBUG_TRITON:
                    print(f"residue_m = {residue_m}")  # noqa: E701

            # offset input and output tensor by batch and Q/K heads
            adj_q = bid * stride_qb + hqid * stride_qh + q_start * stride_qm
            Q_ptr = Q + adj_q
            adj_do = bid * stride_dob + hqid * stride_doh + q_start * stride_dom
            DO_ptr = DO + adj_do
            adj_delta = (
                bid * stride_deltab + hqid * stride_deltah + q_start * stride_deltam
            )
            M_ptr = M + adj_delta
            Delta_ptr = Delta + adj_delta

            if USE_ALIBI:
                alibi_offset = bid * stride_az + hqid * stride_ah
                alibi_slope = tl.load(Alibi_slopes + alibi_offset)
            else:
                alibi_slope = None

            # batch_philox_offset is the ACTUALLY dropout offset
            # dropout_offset is for debug purpose and will be removed later
            batch_philox_offset = 0
            dropout_offset = 0
            if ENABLE_DROPOUT:
                batch_philox_offset = (
                    philox_offset_base + bid * stride_dropoutb + hqid * stride_dropouth
                )
                dropout_offset = (
                    Dropout_mask + bid * stride_dropoutb + hqid * stride_dropouth
                )

            if IS_FP8:
                descale_q = tl.load(Descale_q + bid * stride_descale_q_z + hqid)
                descale_k = tl.load(Descale_k + bid * stride_descale_k_z + hkid)
                descale_v = tl.load(Descale_v + bid * stride_descale_v_z + hkid)
                descale_do = tl.load(Descale_do + bid * stride_descale_do_z + hqid)
            else:
                descale_q, descale_k, descale_v, descale_do = 1.0, 1.0, 1.0, 1.0

            MASK_BLOCK_M1: tl.constexpr = BLOCK_M1 // BLK_SLICE_FACTOR
            # bound the masked operation to q len so it does not have to wast cycles
            len_m = min(len_m, seqlen_q)
            num_steps = tl.cdiv(len_m, MASK_BLOCK_M1)
            # when q < k, we may skip the initial masked op
            if pid < num_blocks_skip:
                num_steps = 0

            # if start_m is negative, the current N-tile has no block on the
            #   diagonal of causal mask, so everything have no causal mask
            if DEBUG_TRITON:
                print(
                    f"Masked: start_n: {start_n}; start_m: {start_m}, num_steps: {num_steps}"
                )  # noqa: E701
            dk, dv = _bwd_dkdv_inner(
                dk,
                dv,  # output tensors
                Q_ptr,
                k,
                v,
                DO_ptr,
                M_ptr,
                Delta_ptr,
                sm_scale,  # input tensors
                stride_qm,
                stride_qd,  # strides for q
                stride_dom,
                stride_dod,  # strides for o
                stride_dropoutm,
                stride_dropoutn,  # strides for dropout
                stride_deltam,
                MASK_BLOCK_M1,
                BLOCK_N1,  # block dim
                HEAD_DIM,
                ACTUAL_HEAD_DIM,  # head dim
                dropout_p,
                philox_seed,
                batch_philox_offset,
                dropout_offset,
                alibi_slope,
                seqlen_q,
                seqlen_k,  # max sequence length for q and k
                start_n,
                start_m,
                num_steps,  # iteration numbers
                descale_q,
                descale_k,
                descale_v,
                descale_do,
                MASK=True,  # causal masking
                ENABLE_DROPOUT=ENABLE_DROPOUT,  # activate dropout
                USE_ALIBI=USE_ALIBI,
                USE_EXP2=USE_EXP2,
                IS_FP8=IS_FP8,
                FP8_MAX=FP8_MAX,
                DEBUG_TRITON=DEBUG_TRITON,
                DEBUG_TRITON_DETAIL=DEBUG_TRITON_DETAIL,
            )
            start_m += num_steps * MASK_BLOCK_M1
            num_steps = tl.cdiv(seqlen_q - start_m, BLOCK_M1)
            end_m = start_m + num_steps * BLOCK_M1

            if DEBUG_TRITON:
                print(
                    f"start_m after Masked step: {start_m}; num_steps: {num_steps}"
                )  # noqa: E701
            if DEBUG_TRITON:
                print(
                    f"unMasked: start_n: {start_n}, start_m: {start_m}, end_m: {end_m}, num_steps: {num_steps}"
                )  # noqa: E701
            if DEBUG_TRITON:
                print("unMasked")  # noqa: E701
            dk, dv = _bwd_dkdv_inner(
                dk,
                dv,  # output tensors
                Q_ptr,
                k,
                v,
                DO_ptr,
                M_ptr,
                Delta_ptr,
                sm_scale,  # input tensors
                stride_qm,
                stride_qd,  # strides for q
                stride_dom,
                stride_dod,  # strides for o
                stride_dropoutm,
                stride_dropoutn,  # strides for dropout
                stride_deltam,
                BLOCK_M1,
                BLOCK_N1,  # block dim
                HEAD_DIM,
                ACTUAL_HEAD_DIM,  # head dim
                dropout_p,
                philox_seed,
                batch_philox_offset,
                dropout_offset,
                alibi_slope,
                seqlen_q,
                seqlen_k,  # max sequence length for q and k
                start_n,
                start_m,
                num_steps,  # iteration numbers
                descale_q,
                descale_k,
                descale_v,
                descale_do,
                MASK=False,  # causal masking
                ENABLE_DROPOUT=ENABLE_DROPOUT,  # activate dropout
                USE_ALIBI=USE_ALIBI,
                USE_EXP2=USE_EXP2,
                IS_FP8=IS_FP8,
                FP8_MAX=FP8_MAX,
                DEBUG_TRITON=DEBUG_TRITON,
                DEBUG_TRITON_DETAIL=DEBUG_TRITON_DETAIL,
            )
        # end of GQA/MQA of dkdv
        # Write back dV
        adj_dv = bid * stride_dvb + hkid * stride_dvh + k_start * stride_dvn
        offs_dv = offs_n[:, None] * stride_dvn + offs_d[None, :] * stride_dvd
        tl.store(DV + adj_dv + offs_dv, dv, mask=mask_kv)
        # write back dk
        adj_dk = bid * stride_dkb + hkid * stride_dkh + k_start * stride_dkn
        offs_dk = offs_n[:, None] * stride_dkn + offs_d[None, :] * stride_dkd
        dk *= sm_scale
        tl.store(DK + adj_dk + offs_dk, dk, mask=mask_kv)

    # This part does dq
    start_m = pid * BLOCK_M2
    if start_m < seqlen_q:
        # seqlen_q > seqlen_k, no need to process these tile for dq
        if DEBUG_TRITON:
            print(
                f"end_n = start_m + BLOCK_M = {start_m} + {BLOCK_M2} = {start_m + BLOCK_M2}"
            )  # noqa: E701
        if start_m + BLOCK_M2 < delta_qk:
            if DEBUG_TRITON:
                print(
                    f"start_m + BLOCK_M2 = {start_m} + {BLOCK_M2} = {start_m + BLOCK_M2} < delta_qk of {delta_qk}"
                )  # noqa: E701
            return

        offs_m = start_m + tl.arange(0, BLOCK_M2)
        # Mask for loading K and V
        mask_q = offs_m[:, None] < seqlen_q
        if PADDED_HEAD:
            mask_d = offs_d < ACTUAL_HEAD_DIM
            mask_q &= mask_d[None, :]
        offs_q = offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
        offs_do = offs_m[:, None] * stride_dom + offs_d[None, :] * stride_dod
        # NOTE: don't assume that the strides for k and v are the same!
        K += bid * stride_kb + hkid * stride_kh + k_start * stride_kn
        V += bid * stride_vb + hkid * stride_vh + k_start * stride_vn

        # If MQA / GQA, set the K and V head offsets appropriately.
        for hqid in range(hkid * GROUP_SIZE, hkid * GROUP_SIZE + GROUP_SIZE):
            # seqlen_q < seqlen_k: delta_qk more kv tokens are added at the front
            #   for every M-tile
            end_n = start_m + BLOCK_M2 - delta_qk
            # clamp end_n at [0, seqlen_k]
            end_n = max(min(end_n, seqlen_k), 0)
            if DEBUG_TRITON:
                print(f"delta_qk: {delta_qk}; end_n: {end_n}")  # noqa: E701
            # offset input and output tensor by batch and Q/K heads
            adj_q = bid * stride_qb + hqid * stride_qh + q_start * stride_qm
            adj_do = bid * stride_dob + hqid * stride_doh + q_start * stride_dom
            adj_delta = (
                bid * stride_deltab + hqid * stride_deltah + q_start * stride_deltam
            )
            Delta_ptr = Delta + adj_delta

            if USE_ALIBI:
                alibi_offset = bid * stride_az + hqid * stride_ah
                alibi_slope = tl.load(Alibi_slopes + alibi_offset)
            else:
                alibi_slope = None

            # batch_philox_offset is the ACTUALLY dropout offset
            # dropout_offset is for debug purpose and will be removed later
            batch_philox_offset = 0
            dropout_offset = 0
            if ENABLE_DROPOUT:
                batch_philox_offset = (
                    philox_offset_base + bid * stride_dropoutb + hqid * stride_dropouth
                )
                dropout_offset = (
                    Dropout_mask + bid * stride_dropoutb + hqid * stride_dropouth
                )
            q = tl.load(Q + adj_q + offs_q, mask=mask_q, other=0.0)
            do = tl.load(DO + adj_do + offs_do, mask=mask_q, other=0.0)
            m = tl.load(M + adj_delta + offs_m * stride_deltam, mask=offs_m < seqlen_q)
            m = m[:, None]

            MASK_BLOCK_N2: tl.constexpr = BLOCK_N2 // BLK_SLICE_FACTOR
            # start can only be 0 at minimum
            start_n = max(end_n - BLOCK_M2, 0)
            num_steps = tl.cdiv(end_n - start_n, MASK_BLOCK_N2)

            if IS_FP8:
                descale_q = tl.load(Descale_q + bid * stride_descale_q_z + hqid)
                descale_k = tl.load(Descale_k + bid * stride_descale_k_z + hkid)
                descale_v = tl.load(Descale_v + bid * stride_descale_v_z + hkid)
                descale_do = tl.load(Descale_do + bid * stride_descale_do_z + hqid)
            else:
                descale_q, descale_k, descale_v, descale_do = 1.0, 1.0, 1.0, 1.0

            dq = tl.zeros([BLOCK_M2, HEAD_DIM], dtype=tl.float32)
            dq = _bwd_dq_inner(
                dq,
                q,
                K,
                V,
                do,
                m,
                Delta_ptr,
                sm_scale,
                stride_qm,
                stride_qd,
                stride_kn,
                stride_kd,
                stride_vn,
                stride_vd,
                stride_dropoutm,
                stride_dropoutn,
                stride_deltam,
                seqlen_q,
                seqlen_k,
                BLOCK_M2,
                MASK_BLOCK_N2,
                HEAD_DIM,
                ACTUAL_HEAD_DIM,
                dropout_p,
                philox_seed,
                batch_philox_offset,
                dropout_offset,
                alibi_slope,
                start_m,
                start_n,
                end_n,
                num_steps,
                descale_q,
                descale_k,
                descale_v,
                descale_do,
                MASK=True,  #
                ENABLE_DROPOUT=ENABLE_DROPOUT,
                USE_ALIBI=USE_ALIBI,
                USE_EXP2=USE_EXP2,
                IS_FP8=IS_FP8,
                FP8_MAX=FP8_MAX,
                DEBUG_TRITON=DEBUG_TRITON,
                DEBUG_TRITON_DETAIL=DEBUG_TRITON_DETAIL,
            )
            end_n -= num_steps * MASK_BLOCK_N2
            num_steps = tl.cdiv(end_n, BLOCK_N2)
            start_n = max(end_n - num_steps * BLOCK_N2, 0)
            if DEBUG_TRITON:
                print(
                    f"unMasked: start_m: {start_m}, start_n: {start_n}, end_n: {end_n}, num_steps: {num_steps}"
                )  # noqa: E701
            dq = _bwd_dq_inner(
                dq,
                q,
                K,
                V,
                do,
                m,
                Delta_ptr,
                sm_scale,
                stride_qm,
                stride_qd,
                stride_kn,
                stride_kd,
                stride_vn,
                stride_vd,
                stride_dropoutm,
                stride_dropoutn,
                stride_deltam,
                seqlen_q,
                seqlen_k,
                BLOCK_M2,
                BLOCK_N2,
                HEAD_DIM,
                ACTUAL_HEAD_DIM,
                dropout_p,
                philox_seed,
                batch_philox_offset,
                dropout_offset,
                alibi_slope,
                start_m,
                start_n,
                end_n,
                num_steps,
                descale_q,
                descale_k,
                descale_v,
                descale_do,
                MASK=False,
                ENABLE_DROPOUT=ENABLE_DROPOUT,
                USE_ALIBI=USE_ALIBI,
                USE_EXP2=USE_EXP2,
                IS_FP8=IS_FP8,
                FP8_MAX=FP8_MAX,
                DEBUG_TRITON=DEBUG_TRITON,
                DEBUG_TRITON_DETAIL=DEBUG_TRITON_DETAIL,
            )
            # Write back dQ.
            adj_dq = bid * stride_dqb + hqid * stride_dqh + q_start * stride_dqm
            offs_dq = offs_m[:, None] * stride_dqm + offs_d[None, :] * stride_dqd
            dq *= sm_scale
            tl.store(DQ + adj_dq + offs_dq, dq, mask=mask_q)
            # end of GQA/MQA of dq


# @triton.autotune(
#     configs=noncausal_autotune_configs,
#     key=noncausal_autotune_keys,
#     use_cuda_graph=True,
# )
@triton.jit
def bwd_kernel_noncausal(
    Q,
    K,
    V,
    sm_scale,
    DO,
    DQ,
    DK,
    DV,
    M,
    Delta,
    stride_qb_in,
    stride_qh_in,
    stride_qm_in,
    stride_qd_in,
    stride_kb_in,
    stride_kh_in,
    stride_kn_in,
    stride_kd_in,
    stride_vb_in,
    stride_vh_in,
    stride_vn_in,
    stride_vd_in,
    stride_dqb_in,
    stride_dqh_in,
    stride_dqm_in,
    stride_dqd_in,
    stride_dkb_in,
    stride_dkh_in,
    stride_dkn_in,
    stride_dkd_in,
    stride_dvb_in,
    stride_dvh_in,
    stride_dvn_in,
    stride_dvd_in,
    stride_deltab_in,
    stride_deltah_in,
    stride_deltam_in,
    stride_dob_in,
    stride_doh_in,
    stride_dom_in,
    stride_dod_in,
    stride_dropoutb_in,
    stride_dropouth_in,
    stride_dropoutm_in,
    stride_dropoutn_in,
    stride_descale_q_z_in,
    stride_descale_k_z_in,
    stride_descale_v_z_in,
    stride_descale_do_z_in,
    stride_az_in,
    stride_ah_in,
    HQ,
    HK,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    Dropout_mask,
    dropout_p,
    philox_seed,
    philox_offset_base_in,
    Alibi_slopes,
    Descale_q,
    Descale_k,
    Descale_v,
    Descale_do,
    BLOCK_M1: tl.constexpr,  # 32
    BLOCK_N1: tl.constexpr,  # 128
    BLOCK_M2: tl.constexpr,  # 128
    BLOCK_N2: tl.constexpr,  # 32
    BLK_SLICE_FACTOR: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    ACTUAL_HEAD_DIM: tl.constexpr,
    ENABLE_DROPOUT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_ALIBI: tl.constexpr,
    USE_EXP2: tl.constexpr,
    IS_FP8: tl.constexpr,
    FP8_MAX: tl.constexpr,
    FP8_OUTPUT: tl.constexpr,
    DEBUG_TRITON: tl.constexpr,
    DEBUG_TRITON_DETAIL: tl.constexpr,
    USE_INT64_STRIDES: tl.constexpr,
):
    if USE_INT64_STRIDES:
        stride_qb = tl.cast(stride_qb_in, tl.int64)
        stride_qh = tl.cast(stride_qh_in, tl.int64)
        stride_qm = tl.cast(stride_qm_in, tl.int64)
        stride_qd = tl.cast(stride_qd_in, tl.int64)
        stride_kb = tl.cast(stride_kb_in, tl.int64)
        stride_kh = tl.cast(stride_kh_in, tl.int64)
        stride_kn = tl.cast(stride_kn_in, tl.int64)
        stride_kd = tl.cast(stride_kd_in, tl.int64)
        stride_vb = tl.cast(stride_vb_in, tl.int64)
        stride_vh = tl.cast(stride_vh_in, tl.int64)
        stride_vn = tl.cast(stride_vn_in, tl.int64)
        stride_vd = tl.cast(stride_vd_in, tl.int64)
        stride_dqb = tl.cast(stride_dqb_in, tl.int64)
        stride_dqh = tl.cast(stride_dqh_in, tl.int64)
        stride_dqm = tl.cast(stride_dqm_in, tl.int64)
        stride_dqd = tl.cast(stride_dqd_in, tl.int64)
        stride_dkb = tl.cast(stride_dkb_in, tl.int64)
        stride_dkh = tl.cast(stride_dkh_in, tl.int64)
        stride_dkn = tl.cast(stride_dkn_in, tl.int64)
        stride_dkd = tl.cast(stride_dkd_in, tl.int64)
        stride_dvb = tl.cast(stride_dvb_in, tl.int64)
        stride_dvh = tl.cast(stride_dvh_in, tl.int64)
        stride_dvn = tl.cast(stride_dvn_in, tl.int64)
        stride_dvd = tl.cast(stride_dvd_in, tl.int64)
        stride_deltab = tl.cast(stride_deltab_in, tl.int64)
        stride_deltah = tl.cast(stride_deltah_in, tl.int64)
        stride_deltam = tl.cast(stride_deltam_in, tl.int64)
        stride_dob = tl.cast(stride_dob_in, tl.int64)
        stride_doh = tl.cast(stride_doh_in, tl.int64)
        stride_dom = tl.cast(stride_dom_in, tl.int64)
        stride_dod = tl.cast(stride_dod_in, tl.int64)
        philox_offset_base = tl.cast(philox_offset_base_in, tl.int64)
        stride_dropoutb = tl.cast(stride_dropoutb_in, tl.int64)
        stride_dropouth = tl.cast(stride_dropouth_in, tl.int64)
        stride_dropoutm = tl.cast(stride_dropoutm_in, tl.int64)
        stride_dropoutn = tl.cast(stride_dropoutn_in, tl.int64)
        if IS_FP8:
            stride_descale_q_z = tl.cast(stride_descale_q_z_in, tl.int64)
            stride_descale_k_z = tl.cast(stride_descale_k_z_in, tl.int64)
            stride_descale_v_z = tl.cast(stride_descale_v_z_in, tl.int64)
            stride_descale_do_z = tl.cast(stride_descale_do_z_in, tl.int64)
        stride_az = tl.cast(stride_az_in, tl.int64)
        stride_ah = tl.cast(stride_ah_in, tl.int64)
    else:
        stride_qb = stride_qb_in
        stride_qh = stride_qh_in
        stride_qm = stride_qm_in
        stride_qd = stride_qd_in
        stride_kb = stride_kb_in
        stride_kh = stride_kh_in
        stride_kn = stride_kn_in
        stride_kd = stride_kd_in
        stride_vb = stride_vb_in
        stride_vh = stride_vh_in
        stride_vn = stride_vn_in
        stride_vd = stride_vd_in
        stride_dqb = stride_dqb_in
        stride_dqh = stride_dqh_in
        stride_dqm = stride_dqm_in
        stride_dqd = stride_dqd_in
        stride_dkb = stride_dkb_in
        stride_dkh = stride_dkh_in
        stride_dkn = stride_dkn_in
        stride_dkd = stride_dkd_in
        stride_dvb = stride_dvb_in
        stride_dvh = stride_dvh_in
        stride_dvn = stride_dvn_in
        stride_dvd = stride_dvd_in
        stride_deltab = stride_deltab_in
        stride_deltah = stride_deltah_in
        stride_deltam = stride_deltam_in
        stride_dob = stride_dob_in
        stride_doh = stride_doh_in
        stride_dom = stride_dom_in
        stride_dod = stride_dod_in
        philox_offset_base = philox_offset_base_in
        stride_dropoutb = stride_dropoutb_in
        stride_dropouth = stride_dropouth_in
        stride_dropoutm = stride_dropoutm_in
        stride_dropoutn = stride_dropoutn_in
        stride_descale_q_z = stride_descale_q_z_in
        stride_descale_k_z = stride_descale_k_z_in
        stride_descale_v_z = stride_descale_v_z_in
        stride_descale_do_z = stride_descale_do_z_in
        stride_az = stride_az_in
        stride_ah = stride_ah_in

    # program ids
    hkid = tl.program_id(0)
    pid = tl.program_id(1)
    bid = tl.program_id(2)
    if DEBUG_TRITON:
        print(f"\npid: {pid}, bid: {bid}, hkid: {hkid}")  # noqa: E701
    # figure out varlen start and end
    q_start = 0
    k_start = 0
    seqlen_q = max_seqlen_q
    seqlen_k = max_seqlen_k
    if IS_VARLEN:
        # Compute actual sequence lengths
        q_start = tl.load(cu_seqlens_q + bid)
        q_end = tl.load(cu_seqlens_q + bid + 1)
        k_start = tl.load(cu_seqlens_k + bid)
        k_end = tl.load(cu_seqlens_k + bid + 1)
        seqlen_q = q_end - q_start
        seqlen_k = k_end - k_start

    PADDED_HEAD: tl.constexpr = ACTUAL_HEAD_DIM != HEAD_DIM
    offs_d = tl.arange(0, HEAD_DIM)
    GROUP_SIZE: tl.constexpr = HQ // HK

    start_n = pid * BLOCK_N1
    if start_n < seqlen_k:
        dk = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)
        dv = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)

        offs_n = start_n + tl.arange(0, BLOCK_N1)
        # Mask for loading K and V
        mask_kv = offs_n[:, None] < seqlen_k
        if PADDED_HEAD:
            mask_d = offs_d < ACTUAL_HEAD_DIM
            mask_kv &= mask_d[None, :]
        # NOTE: don't assume that the strides for k and v are the same!
        # K/V tensors not changed for the group
        adj_k = (
            bid * stride_kb
            + hkid * stride_kh
            + k_start * stride_kn
            + offs_n[:, None] * stride_kn
            + offs_d[None, :] * stride_kd
        )
        adj_v = (
            bid * stride_vb
            + hkid * stride_vh
            + k_start * stride_vn
            + offs_n[:, None] * stride_vn
            + offs_d[None, :] * stride_vd
        )
        # load K and V: they stay in SRAM throughout the inner loop.
        k = tl.load(K + adj_k, mask=mask_kv, other=0.0)
        v = tl.load(V + adj_v, mask=mask_kv, other=0.0)
        # If MQA / GQA, set the K and V head offsets appropriately.
        for hqid in range(hkid * GROUP_SIZE, hkid * GROUP_SIZE + GROUP_SIZE):
            # offset input and output tensor by batch and Q/K heads
            adj_q = bid * stride_qb + hqid * stride_qh + q_start * stride_qm
            Q_ptr = Q + adj_q
            adj_do = bid * stride_dob + hqid * stride_doh + q_start * stride_dom
            DO_ptr = DO + adj_do
            adj_delta = (
                bid * stride_deltab + hqid * stride_deltah + q_start * stride_deltam
            )
            M_ptr = M + adj_delta
            Delta_ptr = Delta + adj_delta

            if USE_ALIBI:
                alibi_offset = bid * stride_az + hqid * stride_ah
                alibi_slope = tl.load(Alibi_slopes + alibi_offset)
            else:
                alibi_slope = None

            # batch_philox_offset is the ACTUALLY dropout offset
            # dropout_offset is for debug purpose and will be removed later
            batch_philox_offset = 0
            dropout_offset = 0
            if ENABLE_DROPOUT:
                batch_philox_offset = (
                    philox_offset_base + bid * stride_dropoutb + hqid * stride_dropouth
                )
                dropout_offset = (
                    Dropout_mask + bid * stride_dropoutb + hqid * stride_dropouth
                )

            if IS_FP8:
                descale_q = tl.load(Descale_q + bid * stride_descale_q_z + hqid)
                descale_k = tl.load(Descale_k + bid * stride_descale_k_z + hkid)
                descale_v = tl.load(Descale_v + bid * stride_descale_v_z + hkid)
                descale_do = tl.load(Descale_do + bid * stride_descale_do_z + hqid)
            else:
                descale_q, descale_k, descale_v, descale_do = 1.0, 1.0, 1.0, 1.0

            # because there is no causal, we always start from the beginning
            start_m = 0
            num_steps = tl.cdiv(seqlen_q, BLOCK_M1)
            dk, dv = _bwd_dkdv_inner(
                dk,
                dv,  # output tensors
                Q_ptr,
                k,
                v,
                DO_ptr,
                M_ptr,
                Delta_ptr,
                sm_scale,  # input tensors
                stride_qm,
                stride_qd,  # strides for q
                stride_dom,
                stride_dod,  # strides for o
                stride_dropoutm,
                stride_dropoutn,  # strides for dropout
                stride_deltam,
                BLOCK_M1,
                BLOCK_N1,  # block dim
                HEAD_DIM,
                ACTUAL_HEAD_DIM,  # head dim
                dropout_p,
                philox_seed,
                batch_philox_offset,
                dropout_offset,  #
                alibi_slope,
                seqlen_q,
                seqlen_k,  # max sequence length for q and k
                start_n,
                start_m,
                num_steps,  # iteration numbers
                descale_q,
                descale_k,
                descale_v,
                descale_do,  # fp8 descale factors from user
                MASK=False,  # causal masking
                ENABLE_DROPOUT=ENABLE_DROPOUT,  # activate dropout
                USE_ALIBI=USE_ALIBI,
                USE_EXP2=USE_EXP2,
                IS_FP8=IS_FP8,
                FP8_MAX=FP8_MAX,
                DEBUG_TRITON=DEBUG_TRITON,
                DEBUG_TRITON_DETAIL=DEBUG_TRITON_DETAIL,
            )

        # Write back dV
        adj_dv = bid * stride_dvb + hkid * stride_dvh + k_start * stride_dvn
        offs_dv = offs_n[:, None] * stride_dvn + offs_d[None, :] * stride_dvd
        tl.store(DV + adj_dv + offs_dv, dv, mask=mask_kv)
        # write back dk
        adj_dk = bid * stride_dkb + hkid * stride_dkh + k_start * stride_dkn
        offs_dk = offs_n[:, None] * stride_dkn + offs_d[None, :] * stride_dkd
        dk *= sm_scale
        tl.store(DK + adj_dk + offs_dk, dk, mask=mask_kv)

    # THIS PART DOES DQ
    start_m = pid * BLOCK_M2
    if start_m < seqlen_q:
        offs_m = start_m + tl.arange(0, BLOCK_M2)
        # Mask for loading K and V
        mask_q = offs_m[:, None] < seqlen_q
        if PADDED_HEAD:
            mask_d = offs_d < ACTUAL_HEAD_DIM
            mask_q &= mask_d[None, :]
        offs_q = offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
        offs_do = offs_m[:, None] * stride_dom + offs_d[None, :] * stride_dod
        K += bid * stride_kb + hkid * stride_kh + k_start * stride_kn
        V += bid * stride_vb + hkid * stride_vh + k_start * stride_vn
        # If MQA / GQA, set the K and V head offsets appropriately.
        for hqid in range(hkid * GROUP_SIZE, hkid * GROUP_SIZE + GROUP_SIZE):
            # offset input and output tensor by batch and Q/K heads
            adj_q = bid * stride_qb + hqid * stride_qh + q_start * stride_qm
            adj_do = bid * stride_dob + hqid * stride_doh + q_start * stride_dom
            adj_delta = (
                bid * stride_deltab + hqid * stride_deltah + q_start * stride_deltam
            )
            Delta_ptr = Delta + adj_delta

            if USE_ALIBI:
                alibi_offset = bid * stride_az + hqid * stride_ah
                alibi_slope = tl.load(Alibi_slopes + alibi_offset)
            else:
                alibi_slope = None

            # batch_philox_offset is the ACTUALLY dropout offset
            # dropout_offset is for debug purpose and will be removed later
            batch_philox_offset = 0
            dropout_offset = 0
            if ENABLE_DROPOUT:
                batch_philox_offset = (
                    philox_offset_base + bid * stride_dropoutb + hqid * stride_dropouth
                )
                dropout_offset = (
                    Dropout_mask + bid * stride_dropoutb + hqid * stride_dropouth
                )

            q = tl.load(Q + adj_q + offs_q, mask=mask_q, other=0.0)
            do = tl.load(DO + adj_do + offs_do, mask=mask_q, other=0.0)
            m = tl.load(M + adj_delta + offs_m * stride_deltam, mask=offs_m < seqlen_q)
            m = m[:, None]

            if IS_FP8:
                descale_q = tl.load(Descale_q + bid * stride_descale_q_z + hqid)
                descale_k = tl.load(Descale_k + bid * stride_descale_k_z + hkid)
                descale_v = tl.load(Descale_v + bid * stride_descale_v_z + hkid)
                descale_do = tl.load(Descale_do + bid * stride_descale_do_z + hqid)
            else:
                descale_q, descale_k, descale_v, descale_do = 1.0, 1.0, 1.0, 1.0

            # start can only be 0 at minimum
            start_n = 0
            end_n = seqlen_k
            num_steps = tl.cdiv(seqlen_k, BLOCK_N2)

            dq = tl.zeros([BLOCK_M2, HEAD_DIM], dtype=tl.float32)
            dq = _bwd_dq_inner(
                dq,
                q,
                K,
                V,
                do,
                m,
                Delta_ptr,
                sm_scale,
                stride_qm,
                stride_qd,
                stride_kn,
                stride_kd,
                stride_vn,
                stride_vd,
                stride_dropoutm,
                stride_dropoutn,
                stride_deltam,
                seqlen_q,
                seqlen_k,
                BLOCK_M2,
                BLOCK_N2,
                HEAD_DIM,
                ACTUAL_HEAD_DIM,
                dropout_p,
                philox_seed,
                batch_philox_offset,
                dropout_offset,
                alibi_slope,
                start_m,
                start_n,
                end_n,
                num_steps,
                descale_q,
                descale_k,
                descale_v,
                descale_do,
                MASK=False,
                ENABLE_DROPOUT=ENABLE_DROPOUT,
                USE_ALIBI=USE_ALIBI,
                USE_EXP2=USE_EXP2,
                IS_FP8=IS_FP8,
                FP8_MAX=FP8_MAX,
                DEBUG_TRITON=DEBUG_TRITON,
                DEBUG_TRITON_DETAIL=DEBUG_TRITON_DETAIL,
            )
            # Write back dQ.
            adj_dq = bid * stride_dqb + hqid * stride_dqh + q_start * stride_dqm
            offs_dq = offs_m[:, None] * stride_dqm + offs_d[None, :] * stride_dqd
            dq *= sm_scale
            tl.store(DQ + adj_dq + offs_dq, dq, mask=mask_q)


def is_contiguous(x, name):
    if x.is_contiguous():
        return x
    else:
        print(f"{name} is not contiguous")
        return x.contiguous()


def flash_attn_onekernel_backward(
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
    # IS_VARLEN = True if cu_seqlens_q is not None else False

    # IS_FP8 = is_fp8(q)
    # if IS_FP8:
    #     FP8_MAX = torch.finfo(q.dtype).max
    #     # assert that the main inputs are fp8

    #     stride_descale_q_z = descale_q.stride(0) if descale_q is not None else None
    #     stride_descale_k_z = descale_k.stride(0) if descale_k is not None else None
    #     stride_descale_v_z = descale_v.stride(0) if descale_v is not None else None
    #     stride_descale_do_z = descale_do.stride(0) if descale_do is not None else None
    # else:
    #     FP8_MAX = None
    #     stride_descale_q_z = stride_descale_k_z = stride_descale_v_z = (
    #         stride_descale_o_z
    #     ) = stride_descale_do_z = None
    # if IS_VARLEN:
    #     layout = "thd"
    # elif q.shape[2] == max_seqlen_q:
    #     layout = "bhsd"
    # elif q.shape[1] == max_seqlen_q:
    #     layout = "bshd"
    # else:
    #     raise ValueError("invalid layout")

    # # get strides and shape
    # batch, nheads_q, nheads_k, head_size, max_seqlen_q_final, max_seqlen_k_final = (
    #     get_shapes_from_layout(
    #         q, k, layout, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k
    #     )
    # )
    # q_strides, k_strides, v_strides, o_strides = get_strides_from_layout(
    #     q, k, v, o, layout
    # )
    # stride_qb, stride_qh, stride_qm, stride_qd = q_strides
    # stride_kb, stride_kh, stride_kn, stride_kd = k_strides
    # stride_vb, stride_vh, stride_vn, stride_vd = v_strides
    # stride_ob, stride_oh, stride_om, stride_od = o_strides
    # dq_strides, dk_strides, dv_strides, do_strides = get_strides_from_layout(
    #     dq, dk, dv, do, layout
    # )
    # stride_dqb, stride_dqh, stride_dqm, stride_dqd = dq_strides
    # stride_dkb, stride_dkh, stride_dkn, stride_dkd = dk_strides
    # stride_dvb, stride_dvh, stride_dvn, stride_dvd = dv_strides
    # stride_dob, stride_doh, stride_dom, stride_dod = do_strides
    # use_dropout = dropout_p > 0.0
    use_alibi, (stride_az, stride_ah) = (
        (True, alibi_slopes.stride()) if alibi_slopes is not None else (False, (0, 0))
    )

    # # get closest power of 2 over or equal to 32.
    # padded_d_model = 1 << (head_size - 1).bit_length()
    # padded_d_model = max(padded_d_model, 32)
    # HEAD_DIM = padded_d_model
    # ACTUAL_HEAD_DIM = head_size

    # # init delta
    # delta = torch.zeros_like(softmax_lse)
    # if IS_VARLEN:
    #     stride_deltab = 0
    #     stride_deltam, stride_deltah = delta.stride()
    # else:
    #     stride_deltab, stride_deltah, stride_deltam = delta.stride()
    # PRE_BLOCK = 128
    # pre_grid = (triton.cdiv(max_seqlen_q, PRE_BLOCK), batch, nheads_q)

    # _bwd_preprocess[pre_grid](
    #     o,
    #     do,
    #     delta,
    #     stride_ob,
    #     stride_oh,
    #     stride_om,
    #     stride_od,
    #     stride_deltab,
    #     stride_deltah,
    #     stride_deltam,
    #     stride_descale_do_z,
    #     cu_seqlens_q,
    #     max_seqlen_q,
    #     descale_do,
    #     BLOCK_M=PRE_BLOCK,
    #     BLOCK_D_MODEL=HEAD_DIM,
    #     BLOCK_D_MODEL_POW2=ACTUAL_HEAD_DIM,
    #     IS_VARLEN=IS_VARLEN,
    #     IS_FP8=IS_FP8,
    # )

    # # dropout mask tensor for debugging. We dump the dropout mask created in
    # #   the kernel for testing
    # dropout_mask = None
    # stride_dropoutb, stride_dropouth, stride_dropoutm, stride_dropoutn = (0, 0, 0, 0)
    # if use_dropout:
    #     dropout_mask = torch.zeros(
    #         (batch, nheads_q, max_seqlen_q_final, max_seqlen_k_final),
    #         device=q.device,
    #         dtype=torch.float32,
    #     )

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
        dv_strides = (0, dv.stride(1), dv.stride(0), dv.stride(2))
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
        dv_strides = (dv.stride(0), dv.stride(2), dv.stride(1), dv.stride(3))
        do_strides = (do.stride(0), do.stride(2), do.stride(1), do.stride(3))

    # BLOCK_D_MODEL, BLOCK_D_MODEL_POW2
    # padding for head_dim. Power of 2 or 16
    BLOCK_D_MODEL_POW2 = triton.next_power_of_2(head_sz)
    BLOCK_D_MODEL_POW2 = max(BLOCK_D_MODEL_POW2, 16)

    # Configs
    # PRE_BLOCK, BLOCK_M1, BLOCK_N1, BLOCK_M2, BLOCK_N2
    # BLK_SLICE_FACTOR
    # NUM_WARPS, NUM_STAGES = 4, 1
    # WAVES_PER_EU = 1
    # PRE_BLOCK = 128
    # # BLOCK_M1, BLOCK_N1, BLOCK_M2, BLOCK_N2 = 32, 128, 128, 32
    # BLOCK_M1, BLOCK_N1, BLOCK_M2, BLOCK_N2 = 16, 64, 64, 16
    # BLK_SLICE_FACTOR = 2

    NUM_WARPS, NUM_STAGES = 4, 1
    WAVES_PER_EU = 1
    PRE_BLOCK = 128
    BLOCK_M1, BLOCK_N1, BLOCK_M2, BLOCK_N2 = 32, 128, 128, 32
    BLK_SLICE_FACTOR = 2
    matrix_instr_nonkdim = 16

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

    seqlen = max(max_seqlen_q, max_seqlen_k)
    grid = (
        num_k_heads,
        triton.cdiv(seqlen, BLOCK_N1),
        batch,
    )
    NUM_WARPS, NUM_STAGES = 4, 1
    WAVES_PER_EU = 1
    BLOCK_M1, BLOCK_N1, BLOCK_M2, BLOCK_N2 = 32, 128, 128, 32
    BLK_SLICE_FACTOR = 2
    matrix_instr_nonkdim = 16

    onekernel_config = {
        "BLOCK_M1": BLOCK_M1,
        "BLOCK_N1": BLOCK_N1,
        "BLOCK_M2": BLOCK_M2,
        "BLOCK_N2": BLOCK_N2,
        "BLK_SLICE_FACTOR": BLK_SLICE_FACTOR,
        "waves_per_eu": WAVES_PER_EU,
        "matrix_instr_nonkdim": matrix_instr_nonkdim,
        "num_warps": NUM_WARPS,
        "num_ctas": 1,
        "num_stages": NUM_STAGES,
    }
    if causal:
        bwd_kernel_causal[grid](
            q,
            k,
            v,
            sm_scale,
            do,
            dq,
            dk,
            dv,
            softmax_lse,
            delta,
            *q_strides,
            *k_strides,
            *v_strides,
            *dq_strides,
            *dk_strides,
            *dv_strides,
            *delta_strides,
            *do_strides,
            *dropout_strides,
            *descale_strides,
            stride_az,
            stride_ah,
            num_q_heads,
            num_k_heads,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            dropout_mask,
            dropout_p,
            philox_seed,
            philox_offset,
            alibi_slopes,
            descale_q,
            descale_k,
            descale_v,
            descale_do,
            HEAD_DIM=head_sz,
            ACTUAL_HEAD_DIM=BLOCK_D_MODEL_POW2,
            ENABLE_DROPOUT=use_dropout,
            IS_VARLEN=IS_VARLEN,
            USE_ALIBI=use_alibi,
            USE_EXP2=True,
            IS_FP8=IS_FP8,
            FP8_MAX=FP8_MAX,
            FP8_OUTPUT=False,
            DEBUG_TRITON=False,
            DEBUG_TRITON_DETAIL=False,
            USE_INT64_STRIDES=USE_INT64_STRIDES,
            **onekernel_config,
        )
    else:
        bwd_kernel_noncausal[grid](
            q,
            k,
            v,
            sm_scale,
            do,
            dq,
            dk,
            dv,
            softmax_lse,
            delta,
            *q_strides,
            *k_strides,
            *v_strides,
            *dq_strides,
            *dk_strides,
            *dv_strides,
            *delta_strides,
            *do_strides,
            *dropout_strides,
            *descale_strides,
            stride_az,
            stride_ah,
            num_q_heads,
            num_k_heads,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            dropout_mask,
            dropout_p,
            philox_seed,
            philox_offset,
            alibi_slopes,
            descale_q,
            descale_k,
            descale_v,
            descale_do,
            HEAD_DIM=head_sz,
            ACTUAL_HEAD_DIM=BLOCK_D_MODEL_POW2,
            ENABLE_DROPOUT=use_dropout,
            IS_VARLEN=IS_VARLEN,
            USE_ALIBI=use_alibi,
            USE_EXP2=True,
            IS_FP8=IS_FP8,
            FP8_MAX=FP8_MAX,
            FP8_OUTPUT=False,
            DEBUG_TRITON=False,
            DEBUG_TRITON_DETAIL=False,
            USE_INT64_STRIDES=USE_INT64_STRIDES,
            **onekernel_config,
        )

    return delta
