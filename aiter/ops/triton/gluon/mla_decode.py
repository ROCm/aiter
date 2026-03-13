# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
MLA Decode Attention (Stage1 + Stage2).

Stage1: Paged grouped decode attention with 16×16×16 WMMA.
  - 4 warps, each handles 128 K-dims contiguously (512 total)
  - Separate PE handling (64 dims)
  - Online softmax per split
  - K→V reuse via LDS

Stage2: Cross-split log-sum-exp reduction (standard Triton kernel).

Public API:
    mla_decode(q, k_buffer, v_buffer, o, req_to_token, b_seq_len,
               attn_logits, num_kv_splits, sm_scale, page_size, logit_cap=0.0)
"""

import torch
import triton
import triton.language as tl

try:
    from triton.experimental import gluon
    from triton.experimental.gluon import language as gl
except ImportError:
    gluon = triton
    gl = tl

is_hip_ = hasattr(triton.runtime.driver.active, "get_current_target") and \
    triton.runtime.driver.active.get_current_target().backend == "hip"


# ============================================================
# Stage1 kernel: grouped decode attention
# ============================================================
@gluon.jit
def _fwd_grouped_kernel_stage1(
    Q, K_Buffer, V_Buffer, sm_scale,
    Req_to_tokens, B_Seqlen, Att_Out,
    stride_req_to_tokens_b,
    stride_qbs, stride_qh,
    stride_buf_kbs, stride_buf_kh,
    stride_buf_vbs, stride_buf_vh,
    stride_mid_ob, stride_mid_oh, stride_mid_os,
    kv_group_num: gl.constexpr,
    q_head_num: gl.constexpr,
    BLOCK_DMODEL: gl.constexpr,
    BLOCK_DPE: gl.constexpr,
    BLOCK_DV: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_H: gl.constexpr,
    NUM_KV_SPLITS: gl.constexpr,
    PAGE_SIZE: gl.constexpr,
    logit_cap: gl.constexpr,
    Lk: gl.constexpr,
    Lv: gl.constexpr,
):
    gl.static_assert(BLOCK_N == PAGE_SIZE)
    NUM_WARPS: gl.constexpr = 4

    # ====== Layout definitions (all constexpr, evaluated at compile time) ======
    load_layout_512: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=[[0, 1], [0, 2], [0, 4], [0, 16], [0, 32], [0, 64]],
        lane_bases=[[1, 0], [2, 0], [4, 0], [8, 0], [0, 8]],
        warp_bases=[[0, 128], [0, 256]],
        block_bases=[], shape=[BLOCK_H, 512],
    )
    load_layout_dpe: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=[[0, 1], [0, 2], [0, 4]],
        lane_bases=[[1, 0], [2, 0], [4, 0], [8, 0], [0, 8]],
        warp_bases=[[0, 16], [0, 32]],
        block_bases=[], shape=[BLOCK_H, 64],
    )
    wmma_layout_3d: gl.constexpr = gl.amd.AMDWMMALayout(
        version=2, transposed=True,
        warp_bases=[[1, 0, 0], [2, 0, 0]],
        instr_shape=[16, 16, 16], rank=3,
    )
    wmma_layout_2d: gl.constexpr = gl.amd.AMDWMMALayout(
        version=2, transposed=True,
        warp_bases=[[0, 1], [0, 2]],
        instr_shape=[16, 16, 16], rank=2,
    )
    k_smem_layout: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[BLOCK_N, 8]], [BLOCK_DV, BLOCK_N], [1, 0],
    )
    reduce_smem_layout: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[16384, 1]], [4, BLOCK_N, BLOCK_H], [2, 1, 0],
    )
    single_warp_reduce_load_layout: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=[
            [1, 0, 0], [2, 0, 0],
            [0, 0, 1], [0, 0, 2], [0, 0, 4],
        ],
        lane_bases=[
            [0, 0, 8],
            [0, 1, 0], [0, 2, 0], [0, 4, 0], [0, 8, 0],
        ],
        warp_bases=[[0, 0, 0], [0, 0, 0]],
        block_bases=[], shape=[4, BLOCK_H, BLOCK_N],
    )
    qk_result_2d_layout: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=[[0, 1], [0, 2], [0, 4]],
        lane_bases=[[0, 8], [1, 0], [2, 0], [4, 0], [8, 0]],
        warp_bases=[[0, 0], [0, 0]],
        block_bases=[], shape=[BLOCK_H, BLOCK_N],
    )

    k_layout_3d: gl.constexpr = gl.DotOperandLayout(1, wmma_layout_3d, 8)
    q_layout_3d: gl.constexpr = gl.DotOperandLayout(0, wmma_layout_3d, 8)
    p_dot_layout_2d: gl.constexpr = gl.DotOperandLayout(0, wmma_layout_2d, 8)
    v_dot_layout_2d: gl.constexpr = gl.DotOperandLayout(1, wmma_layout_2d, 8)
    e_max_layout: gl.constexpr = gl.SliceLayout(1, qk_result_2d_layout)

    # ====== Program IDs ======
    cur_batch = gl.program_id(0)
    cur_head_id = gl.program_id(1)
    cur_kv_head = cur_head_id // gl.cdiv(kv_group_num, BLOCK_H)
    split_kv_id = gl.program_id(2)

    if kv_group_num > BLOCK_H:
        VALID_BLOCK_H: gl.constexpr = BLOCK_H
    else:
        VALID_BLOCK_H: gl.constexpr = kv_group_num

    cur_batch_seq_len = gl.load(B_Seqlen + cur_batch)

    kv_len_per_split = gl.cdiv(cur_batch_seq_len, NUM_KV_SPLITS)
    split_kv_start = kv_len_per_split * split_kv_id
    split_kv_end = gl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)

    TOTAL_K: gl.constexpr = 512

    # ====== Shared memory ======
    k_smem = gl.allocate_shared_memory(
        K_Buffer.dtype.element_ty, shape=[BLOCK_DV, BLOCK_N], layout=k_smem_layout,
    )
    qk_reduce_smem = gl.allocate_shared_memory(
        gl.float32, shape=[NUM_WARPS, BLOCK_H, BLOCK_N], layout=reduce_smem_layout,
    )

    # ====== Offsets ======
    offs_h = cur_head_id * VALID_BLOCK_H + gl.arange(0, BLOCK_H, layout=gl.SliceLayout(1, load_layout_512))
    offs_k = gl.arange(0, TOTAL_K, layout=gl.SliceLayout(0, load_layout_512))
    offs_n = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(1, load_layout_512))

    # ====== Preload Q [16, 512] → [4, 16, 128] for 3D WMMA ======
    q_offs = cur_batch * stride_qbs + offs_h[:, None] * stride_qh + offs_k[None, :]
    q_chunk = gl.load(Q + q_offs)
    q_temp = gl.reshape(q_chunk, [BLOCK_N, NUM_WARPS, BLOCK_DMODEL // NUM_WARPS])
    q_3d = gl.permute(q_temp, [1, 0, 2])
    q_mma_input = gl.convert_layout(q_3d, q_layout_3d)

    # ====== PE preload ======
    if BLOCK_DPE > 0:
        offs_dpe_k = BLOCK_DMODEL + gl.arange(0, BLOCK_DPE, layout=gl.SliceLayout(0, load_layout_dpe))
        offs_dpe_h = cur_head_id * VALID_BLOCK_H + gl.arange(0, BLOCK_H, layout=gl.SliceLayout(1, load_layout_dpe))
        offs_dpe_n = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(1, load_layout_dpe))
        off_qpe = cur_batch * stride_qbs + offs_dpe_h[:, None] * stride_qh + offs_dpe_k[None, :]
        qpe = tl.load(Q + off_qpe)
        qpe_temp = gl.reshape(qpe, [BLOCK_N, NUM_WARPS, BLOCK_DPE // NUM_WARPS])
        qpe_3d = gl.permute(qpe_temp, [1, 0, 2])
        qpe_mma_input = gl.convert_layout(qpe_3d, q_layout_3d)

    req_to_tokens_base = Req_to_tokens + stride_req_to_tokens_b * cur_batch

    # ====== Accumulators ======
    acc = gl.zeros([BLOCK_H, BLOCK_DV], dtype=gl.float32, layout=wmma_layout_2d)
    e_max = gl.full([BLOCK_H], float("-inf"), dtype=gl.float32, layout=e_max_layout)
    e_sum = gl.zeros([BLOCK_H], dtype=gl.float32, layout=e_max_layout)

    # ====== Main loop over token blocks ======
    if split_kv_end > split_kv_start:
        for start_n in range(split_kv_start, split_kv_end, BLOCK_N):
            qk_3d_acc = gl.zeros([NUM_WARPS, BLOCK_H, BLOCK_N], dtype=gl.float32, layout=wmma_layout_3d)

            # Page table lookup
            page_idx = start_n // PAGE_SIZE
            kv_page_id = tl.load(req_to_tokens_base + page_idx)

            # Load K [16, 512]
            k_offs = (kv_page_id * PAGE_SIZE + offs_n[:, None]) * stride_buf_kbs + \
                      cur_kv_head * stride_buf_kh + offs_k[None, :]
            k_2d = gl.load(K_Buffer + k_offs)

            # Store K to LDS for V reuse
            k_2d_transposed = gl.permute(k_2d, [1, 0])
            k_smem.store(k_2d_transposed)

            # Reshape K for 3D WMMA
            k_temp = gl.reshape(k_2d, [BLOCK_N, NUM_WARPS, BLOCK_DMODEL // NUM_WARPS])
            k_3d = gl.permute(k_temp, [1, 0, 2])
            k_t = gl.permute(k_3d, [0, 2, 1])
            k_op1 = gl.convert_layout(k_t, k_layout_3d)

            # PE handling
            if BLOCK_DPE > 0:
                k_offs_pe = (kv_page_id * PAGE_SIZE + offs_dpe_n[:, None]) * stride_buf_kbs + \
                             cur_kv_head * stride_buf_kh + offs_dpe_k[None, :]
                k_2d_pe = gl.load(K_Buffer + k_offs_pe)
                k_temp_pe = gl.reshape(k_2d_pe, [BLOCK_N, NUM_WARPS, BLOCK_DPE // NUM_WARPS])
                k_3d_pe = gl.permute(k_temp_pe, [1, 0, 2])
                k_t_pe = gl.permute(k_3d_pe, [0, 2, 1])
                k_op1_pe = gl.convert_layout(k_t_pe, k_layout_3d)

            # QK WMMA
            qk_3d_acc = gl.amd.rdna4.wmma(q_mma_input, k_op1, qk_3d_acc)
            if BLOCK_DPE > 0:
                qk_3d_acc = gl.amd.rdna4.wmma(qpe_mma_input, k_op1_pe, qk_3d_acc)

            # Cross-warp reduction [4,16,16] → [16,16]
            qk_reduce_smem.store(qk_3d_acc)
            gl.barrier()
            qk_all = qk_reduce_smem.load(layout=single_warp_reduce_load_layout)
            qk_2d = gl.sum(qk_all, 0)
            qk_result = gl.convert_layout(qk_2d, qk_result_2d_layout)

            # Online softmax
            qk_scaled = qk_result * sm_scale
            offs_n_qk = start_n + gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, qk_result_2d_layout))
            mask_n_qk = offs_n_qk < split_kv_end
            qk_scaled = gl.where(mask_n_qk[None, :], qk_scaled, float("-inf"))

            qk_max_cur = gl.max(qk_scaled, 1)
            qk_max_new = gl.maximum(e_max, qk_max_cur)
            rescale = gl.exp(e_max - qk_max_new)
            p = gl.exp(qk_scaled - qk_max_new[:, None])
            exp_sum_cur = gl.sum(p, 1)
            e_sum = e_sum * rescale + exp_sum_cur
            e_max = qk_max_new

            # Rescale acc + PV WMMA
            rescale_for_acc = gl.convert_layout(rescale, gl.SliceLayout(1, wmma_layout_2d))
            acc = acc * rescale_for_acc[:, None]

            p_bf16 = p.to(K_Buffer.dtype.element_ty)
            p_op0 = gl.convert_layout(p_bf16, p_dot_layout_2d)
            v_smem = k_smem.permute([1, 0])
            v_op1 = v_smem.load(layout=v_dot_layout_2d)
            acc = gl.amd.rdna4.wmma(p_op0, v_op1, acc)

        # ====== Store normalized acc + LSE ======
        offs_h_store = cur_head_id * VALID_BLOCK_H + gl.arange(0, BLOCK_H, layout=gl.SliceLayout(1, wmma_layout_2d))
        offs_dv_store = gl.arange(0, BLOCK_DV, layout=gl.SliceLayout(0, wmma_layout_2d))
        offs_mid_o = (cur_batch * stride_mid_ob +
                      offs_h_store[:, None] * stride_mid_oh +
                      split_kv_id * stride_mid_os + offs_dv_store[None, :])

        mask_h = offs_h_store < q_head_num
        mask_dv = offs_dv_store < Lv
        e_sum_for_acc = gl.convert_layout(e_sum, gl.SliceLayout(1, wmma_layout_2d))
        acc_normalized = acc / e_sum_for_acc[:, None]
        gl.store(Att_Out + offs_mid_o, acc_normalized, mask=mask_h[:, None] & mask_dv[None, :])

        offs_h_emax = cur_head_id * VALID_BLOCK_H + gl.arange(0, BLOCK_H, layout=e_max_layout)
        mask_h_emax = offs_h_emax < q_head_num
        offs_mid_o_1 = (cur_batch * stride_mid_ob + offs_h_emax * stride_mid_oh +
                        split_kv_id * stride_mid_os + Lv)
        gl.store(Att_Out + offs_mid_o_1, e_max + gl.log(e_sum), mask=mask_h_emax)


# ============================================================
# Stage2 kernel: cross-split reduction
# ============================================================
@triton.jit
def _fwd_kernel_stage2(
    Mid_O, o, B_Seqlen,
    stride_mid_ob, stride_mid_oh, stride_mid_os,
    stride_obs, stride_oh,
    NUM_KV_SPLITS: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    Lv: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)

    offs_d = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < Lv

    e_sum = 0.0
    e_max = -float("inf")
    acc = tl.zeros([BLOCK_DV], dtype=tl.float32)

    offs_v = cur_batch * stride_mid_ob + cur_head * stride_mid_oh + offs_d
    offs_logic = cur_batch * stride_mid_ob + cur_head * stride_mid_oh + Lv

    for split_kv_id in range(0, NUM_KV_SPLITS):
        kv_len_per_split = tl.cdiv(cur_batch_seq_len, NUM_KV_SPLITS)
        split_kv_start = kv_len_per_split * split_kv_id
        split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)

        if split_kv_end > split_kv_start:
            tv = tl.load(Mid_O + offs_v + split_kv_id * stride_mid_os, mask=mask_d, other=0.0)
            tlogic = tl.load(Mid_O + offs_logic + split_kv_id * stride_mid_os)
            n_e_max = tl.maximum(tlogic, e_max)

            old_scale = tl.exp(e_max - n_e_max)
            acc *= old_scale
            exp_logic = tl.exp(tlogic - n_e_max)
            acc += exp_logic * tv

            e_sum = e_sum * old_scale + exp_logic
            e_max = n_e_max

    tl.store(o + cur_batch * stride_obs + cur_head * stride_oh + offs_d,
             acc / e_sum, mask=mask_d)


# ============================================================
# Stage1 launcher
# ============================================================
def _decode_grouped_att_m_fwd(
    q, k_buffer, v_buffer, att_out,
    Req_to_tokens, B_Seqlen,
    num_kv_splits, sm_scale, page_size, logit_cap,
):
    BLOCK = 16
    Lk = k_buffer.shape[-1]
    Lv = v_buffer.shape[-1]

    if Lk == 576:
        BLOCK_DMODEL = 512
        BLOCK_DPE = 64
    elif Lk == 288:
        BLOCK_DMODEL = 256
        BLOCK_DPE = 32
    else:
        BLOCK_DMODEL = triton.next_power_of_2(Lk)
        BLOCK_DPE = 0
    BLOCK_DV = triton.next_power_of_2(Lv)

    batch, head_num = q.shape[0], q.shape[1]
    kv_group_num = q.shape[1] // k_buffer.shape[-2]

    BLOCK_H = 16
    grid = (batch, triton.cdiv(head_num, min(BLOCK_H, kv_group_num)), num_kv_splits)

    _fwd_grouped_kernel_stage1[grid](
        q, k_buffer, v_buffer, sm_scale,
        Req_to_tokens, B_Seqlen, att_out,
        Req_to_tokens.stride(0),
        q.stride(0), q.stride(1),
        k_buffer.stride(-3), k_buffer.stride(-2),
        v_buffer.stride(-3), v_buffer.stride(-2),
        att_out.stride(0), att_out.stride(1), att_out.stride(2),
        kv_group_num=kv_group_num,
        q_head_num=head_num,
        BLOCK_DMODEL=BLOCK_DMODEL,
        BLOCK_DPE=BLOCK_DPE,
        BLOCK_DV=BLOCK_DV,
        BLOCK_N=BLOCK,
        BLOCK_H=BLOCK_H,
        NUM_KV_SPLITS=num_kv_splits,
        PAGE_SIZE=page_size,
        logit_cap=logit_cap,
        Lk=Lk, Lv=Lv,
        num_warps=4, num_stages=1, waves_per_eu=1,
    )


# ============================================================
# Stage2 launcher
# ============================================================
def _decode_softmax_reducev_fwd(logits, q, o, v_buffer, b_seq_len, num_kv_splits):
    batch, head_num = q.shape[0], q.shape[1]
    Lv = v_buffer.shape[-1]
    BLOCK_DV = triton.next_power_of_2(Lv)

    extra_kargs = {}
    if is_hip_:
        extra_kargs = {"waves_per_eu": 4, "matrix_instr_nonkdim": 16, "kpack": 2}

    grid = (batch, head_num)
    _fwd_kernel_stage2[grid](
        logits, o, b_seq_len,
        logits.stride(0), logits.stride(1), logits.stride(2),
        o.stride(0), o.stride(1),
        NUM_KV_SPLITS=num_kv_splits,
        BLOCK_DV=BLOCK_DV, Lv=Lv,
        num_warps=4, num_stages=2,
        **extra_kargs,
    )


# ============================================================
# Public API
# ============================================================
def mla_decode(
    q, k_buffer, v_buffer, o, req_to_token, b_seq_len,
    attn_logits, num_kv_splits, sm_scale, page_size, logit_cap=0.0,
):
    """MLA decode attention (Stage1 + Stage2)."""
    _decode_grouped_att_m_fwd(
        q, k_buffer, v_buffer, attn_logits,
        req_to_token, b_seq_len,
        num_kv_splits, sm_scale, page_size, logit_cap,
    )
    _decode_softmax_reducev_fwd(
        attn_logits, q, o, v_buffer, b_seq_len, num_kv_splits,
    )


def mla_decode_stage1(
    q, k_buffer, v_buffer, attn_logits, req_to_token, b_seq_len,
    num_kv_splits, sm_scale, page_size, logit_cap=0.0,
):
    """Stage1 only (for testing intermediate results)."""
    _decode_grouped_att_m_fwd(
        q, k_buffer, v_buffer, attn_logits,
        req_to_token, b_seq_len,
        num_kv_splits, sm_scale, page_size, logit_cap,
    )


def mla_decode_stage2(
    attn_logits, q, o, v_buffer, b_seq_len, num_kv_splits,
):
    """Stage2 only (for testing cross-split reduction)."""
    _decode_softmax_reducev_fwd(
        attn_logits, q, o, v_buffer, b_seq_len, num_kv_splits,
    )
