# SPDX-License-Identifier: MIT
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.


import triton.language as tl

# from aiter.ops.triton.activation import _tanh
# import aiter.ops.triton.utils.arch_info as arch_info
# from aiter.ops.triton.utils.core import AITER_TRITON_CONFIGS_PATH
# from aiter.ops.triton.utils.pid_preprocessing import remap_xcd
from aiter.jit.utils.chip_info import get_cu_num
from aiter import dtypes

from triton.experimental import gluon
from triton.experimental.gluon import language as gl


@gluon.jit
def mla_core(
    kv_col_start,
    kv_col_end,
    q_nope_0,
    q_nope_1,
    q_rope,
    p_k_buffer,
    p_kv_indices,
    stride_k_bs,
    kv_ld_nope_local_col_idx,
    kv_ld_rope_local_col_idx,
    kv_ld_nope_row_idx,
    kv_ld_rope_row_idx,
    lds_kv_nope_0,
    lds_kv_nope_1,
    lds_k_rope,
    lds_p,
    mfma_layout_qk,
    mfma_layout_kv,
    sm_scale,
    acc_0,
    acc_1,
    row_max,
    row_sum_e,
    FIRST_ITER: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    TRANSFORM_P_VIA_LDS: gl.constexpr,
    kv_lora_rank: gl.constexpr,
    lds_layout_k: gl.constexpr,
    lds_layout_v: gl.constexpr,
):
    log2e: gl.constexpr = 1.4426950408889634

    zero_qk = gl.zeros([BLOCK_M, BLOCK_N], dtype=gl.float32, layout=mfma_layout_qk)

    dot_k_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=mfma_layout_qk, k_width=4
    )
    dot_p_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=mfma_layout_kv, k_width=4
    )
    dot_v_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=mfma_layout_kv, k_width=4
    )

    # Read KV
    kv_ld_nope_col_idx = kv_col_start + kv_ld_nope_local_col_idx
    kv_ld_rope_col_idx = kv_col_start + kv_ld_rope_local_col_idx
    kv_indices_nope = gl.load(p_kv_indices + kv_ld_nope_col_idx)
    kv_indices_rope = gl.load(p_kv_indices + kv_ld_rope_col_idx)
    kv_nope_offsets = (
        kv_indices_nope[:, None] * stride_k_bs + kv_ld_nope_row_idx[None, :]
    )
    kv_rope_offsets = (
        kv_indices_rope[:, None] * stride_k_bs + kv_ld_rope_row_idx[None, :]
    )
    # TODO: We don't need this mask unless this is last iter
    kv_nope_mask = kv_ld_nope_col_idx < kv_col_end
    kv_rope_mask = kv_ld_rope_col_idx < kv_col_end

    kv_nope_data_0 = gl.amd.cdna3.buffer_load(
        ptr=p_k_buffer,
        offsets=kv_nope_offsets,
        mask=kv_nope_mask[:, None],
    )
    kv_nope_data_1 = gl.amd.cdna3.buffer_load(
        ptr=p_k_buffer,
        offsets=kv_nope_offsets + kv_lora_rank // 2,
        mask=kv_nope_mask[:, None],
    )
    kv_rope_data = gl.amd.cdna3.buffer_load(
        ptr=p_k_buffer,
        offsets=kv_rope_offsets,
        mask=kv_rope_mask[:, None],
    )

    lds_kv_nope_0.store(kv_nope_data_0.T)
    lds_kv_nope_1.store(kv_nope_data_1.T)
    lds_k_rope.store(kv_rope_data.T)
    kv_nope_0 = lds_kv_nope_0.load(layout=dot_k_layout)
    kv_nope_1 = lds_kv_nope_1.load(layout=dot_k_layout)
    k_rope = lds_k_rope.load(layout=dot_k_layout)

    # QK GEMM
    qk = gl.amd.cdna3.mfma(q_nope_0, kv_nope_0, zero_qk)
    qk = gl.amd.cdna3.mfma(q_nope_1, kv_nope_1, qk)
    qk = gl.amd.cdna3.mfma(q_rope, k_rope, qk)

    # Store V in LDS
    lds_kv_nope_0 = lds_kv_nope_0._reinterpret(
        p_k_buffer.type.element_ty, [BLOCK_N, kv_lora_rank // 2], layout=lds_layout_v
    )
    lds_kv_nope_1 = lds_kv_nope_1._reinterpret(
        p_k_buffer.type.element_ty, [BLOCK_N, kv_lora_rank // 2], layout=lds_layout_v
    )
    lds_kv_nope_0.store(tl.trans(kv_nope_data_0).T)
    lds_kv_nope_1.store(tl.trans(kv_nope_data_1).T)

    # Softmax
    qk *= sm_scale
    row_local_max = gl.convert_layout(gl.max(qk, 1), gl.SliceLayout(1, mfma_layout_qk))
    if FIRST_ITER:
        new_row_max = row_local_max
    else:
        new_row_max = gl.maximum(row_local_max, row_max)
        rescale = tl.math.exp2((row_max - new_row_max) * log2e)
    p = tl.math.exp2((qk - new_row_max[:, None]) * log2e)
    # TODO: Only required by last block of BLOCK_N
    p_col_idx = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, mfma_layout_qk))
    p_mask = (p_col_idx + kv_col_start) < kv_col_end
    p = gl.where(p_mask[None, :], p, 0)
    row_max = new_row_max
    if FIRST_ITER:
        row_sum_e = gl.sum(p, 1)
    else:
        row_sum_e = row_sum_e * rescale + gl.sum(p, 1)

    # Transform P
    if TRANSFORM_P_VIA_LDS:
        lds_p.store(p.cast(lds_p.type.element_ty))
        p = lds_p.load(layout=dot_p_layout)
    else:
        p = gl.convert_layout(p.cast(lds_kv_nope_0.type.element_ty), dot_p_layout)

    # Load V from LDS
    v_0 = lds_kv_nope_0.load(layout=dot_v_layout)
    v_1 = lds_kv_nope_1.load(layout=dot_v_layout)
    lds_kv_nope_0 = lds_kv_nope_0._reinterpret(
        p_k_buffer.type.element_ty, [kv_lora_rank // 2, BLOCK_N], layout=lds_layout_k
    )
    lds_kv_nope_1 = lds_kv_nope_1._reinterpret(
        p_k_buffer.type.element_ty, [kv_lora_rank // 2, BLOCK_N], layout=lds_layout_k
    )

    # KV GEMM
    if not FIRST_ITER:
        acc_0 *= rescale[:, None]
        acc_1 *= rescale[:, None]
    acc_0 = gl.amd.cdna3.mfma(p, v_0, acc_0)
    acc_1 = gl.amd.cdna3.mfma(p, v_1, acc_1)

    return acc_0, acc_1, row_max, row_sum_e


@gluon.jit
def kn_mla_fwd_fp8_m128_ps(
    p_q,
    p_k_buffer,
    p_v_buffer,
    p_kv_indptr,
    p_kv_indices,
    p_final_out,
    p_temp_out,
    p_temp_lse,
    p_work_indptr,
    p_work_info_set,
    sm_scale,
    max_seqlen_q,
    stride_q_bs,
    stride_q_h,
    stride_k_bs,
    stride_v_bs,
    stride_final_out_bs,
    stride_final_out_h,
    stride_temp_out_bs,
    stride_temp_out_h,
    stride_temp_lse_bs,
    stride_temp_lse_h,
    kv_lora_rank: gl.constexpr,
    qk_rope_head_dim: gl.constexpr,
    num_head_q: gl.constexpr,
    same_kv: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
):
    # Not expected to be supported
    assert same_kv

    # Private configs
    Q_LOAD_LAYOUT_DOT: gl.constexpr = False
    TRANSFORM_Q_VIA_LDS: gl.constexpr = True and not Q_LOAD_LAYOUT_DOT
    TRANSFORM_P_VIA_LDS: gl.constexpr = False

    """
    Const states
    """
    # 128 * 256, row * col
    layout_q_ld_nope: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[4, 16],
        threads_per_warp=[4, 16],
        warps_per_cta=[8, 1],
        order=[1, 0],
    )
    # 128 * 64, row * col
    layout_q_ld_rope: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[2, 8],
        threads_per_warp=[8, 8],
        warps_per_cta=[8, 1],
        order=[1, 0],
    )
    # 32 * 256, col * row
    layout_k_ld_nope: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 16],
        threads_per_warp=[4, 16],
        warps_per_cta=[8, 1],
        order=[1, 0],
    )
    # 32 * 64, col * row
    layout_k_ld_rope: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 4],
        threads_per_warp=[4, 16],
        warps_per_cta=[8, 1],
        order=[1, 0],
    )

    lds_layout_q: gl.constexpr = gl.SwizzledSharedLayout(
        vec=8, per_phase=1, max_phase=16, order=[1, 0]
    )
    lds_layout_k: gl.constexpr = gl.SwizzledSharedLayout(
        vec=8, per_phase=4, max_phase=16, order=[0, 1]
    )
    lds_layout_v: gl.constexpr = gl.SwizzledSharedLayout(
        vec=8, per_phase=1, max_phase=16, order=[0, 1]
    )
    lds_layout_p: gl.constexpr = gl.SwizzledSharedLayout(
        vec=8, per_phase=1, max_phase=16, order=[1, 0]
    )

    mfma_layout_qk: gl.constexpr = gl.amd.AMDMFMALayout(
        version=3, instr_shape=[16, 16], transposed=False, warps_per_cta=[1, 8]
    )
    mfma_layout_kv: gl.constexpr = gl.amd.AMDMFMALayout(
        version=3, instr_shape=[16, 16], transposed=False, warps_per_cta=[1, 8]
    )

    dot_q_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=mfma_layout_qk, k_width=4
    )

    """
    Runtime static values
    """
    if Q_LOAD_LAYOUT_DOT:
        q_ld_nope_row_idx = gl.arange(
            0, BLOCK_M, layout=gl.SliceLayout(1, dot_q_layout)
        )
        q_ld_nope_col_idx = gl.arange(
            0, kv_lora_rank // 2, layout=gl.SliceLayout(0, dot_q_layout)
        )
        q_ld_rope_row_idx = gl.arange(
            0, BLOCK_M, layout=gl.SliceLayout(1, dot_q_layout)
        )
        q_ld_rope_col_idx = gl.arange(
            kv_lora_rank,
            kv_lora_rank + qk_rope_head_dim,
            layout=gl.SliceLayout(0, dot_q_layout),
        )
    else:
        q_ld_nope_row_idx = gl.arange(
            0, BLOCK_M, layout=gl.SliceLayout(1, layout_q_ld_nope)
        )
        q_ld_nope_col_idx = gl.arange(
            0, kv_lora_rank // 2, layout=gl.SliceLayout(0, layout_q_ld_nope)
        )
        q_ld_rope_row_idx = gl.arange(
            0, BLOCK_M, layout=gl.SliceLayout(1, layout_q_ld_rope)
        )
        q_ld_rope_col_idx = gl.arange(
            kv_lora_rank,
            kv_lora_rank + qk_rope_head_dim,
            layout=gl.SliceLayout(0, layout_q_ld_rope),
        )

    q_ld_local_offsets_nope = (
        q_ld_nope_row_idx[:, None] * stride_q_h + q_ld_nope_col_idx[None, :]
    )
    q_ld_local_offsets_rope = (
        q_ld_rope_row_idx[:, None] * stride_q_h + q_ld_rope_col_idx[None, :]
    )

    kv_ld_nope_local_col_idx = gl.arange(
        0, BLOCK_N, layout=gl.SliceLayout(1, layout_k_ld_nope)
    )
    kv_ld_rope_local_col_idx = gl.arange(
        0, BLOCK_N, layout=gl.SliceLayout(1, layout_k_ld_rope)
    )
    kv_ld_nope_row_idx = gl.arange(
        0, kv_lora_rank // 2, layout=gl.SliceLayout(0, layout_k_ld_nope)
    )
    kv_ld_rope_row_idx = gl.arange(
        kv_lora_rank,
        kv_lora_rank + qk_rope_head_dim,
        layout=gl.SliceLayout(0, layout_k_ld_rope),
    )

    """
    Main Loop
    """
    cu_idx = gl.program_id(0)
    work_start_idx = gl.load(p_work_indptr + cu_idx)
    work_end_idx = gl.load(p_work_indptr + cu_idx + 1)
    for work_idx in range(work_start_idx, work_end_idx):
        """
        Each work info contains 8 DWs
        """
        # batch_idx = gl.load(p_work_info_set + work_idx * 8)
        partial_qo_loc = gl.load(p_work_info_set + work_idx * 8 + 1)
        qo_start = gl.load(p_work_info_set + work_idx * 8 + 2)
        qo_end = gl.load(p_work_info_set + work_idx * 8 + 3)
        kv_start = gl.load(p_work_info_set + work_idx * 8 + 4)
        kv_end = gl.load(p_work_info_set + work_idx * 8 + 5)
        # kv_offset = gl.load(p_work_info_set + work_idx * 8 + 6)

        # Allocate ACCs and softmax states
        acc_0 = gl.zeros(
            [BLOCK_M, kv_lora_rank // 2], dtype=gl.float32, layout=mfma_layout_kv
        )
        acc_1 = gl.zeros(
            [BLOCK_M, kv_lora_rank // 2], dtype=gl.float32, layout=mfma_layout_kv
        )
        row_sum_e = gl.zeros(
            [BLOCK_M], dtype=gl.float32, layout=gl.SliceLayout(1, mfma_layout_qk)
        )
        row_max = gl.full(
            [BLOCK_M],
            -float("inf"),
            dtype=gl.float32,
            layout=gl.SliceLayout(1, mfma_layout_qk),
        )

        # Allocate LDS for Q
        if TRANSFORM_Q_VIA_LDS:
            lds_q_nope = gl.allocate_shared_memory(
                p_q.type.element_ty, [BLOCK_M, kv_lora_rank // 2], layout=lds_layout_q
            )
            lds_q_rope = gl.allocate_shared_memory(
                p_q.type.element_ty, [BLOCK_M, qk_rope_head_dim], layout=lds_layout_q
            )

        # Read Q
        q_nope_offsets = qo_start * stride_q_bs + q_ld_local_offsets_nope
        q_rope_offsets = qo_start * stride_q_bs + q_ld_local_offsets_rope
        q_nope_mask = (qo_start * num_head_q + q_ld_nope_row_idx) < (
            qo_end * num_head_q
        )
        q_rope_mask = (qo_start * num_head_q + q_ld_rope_row_idx) < (
            qo_end * num_head_q
        )

        q_nope_data_0 = gl.amd.cdna3.buffer_load(
            ptr=p_q,
            offsets=q_nope_offsets,
            mask=q_nope_mask[:, None],
        )
        q_nope_data_1 = gl.amd.cdna3.buffer_load(
            ptr=p_q,
            offsets=q_nope_offsets + kv_lora_rank // 2,
            mask=q_nope_mask[:, None],
        )
        q_rope_data = gl.amd.cdna3.buffer_load(
            ptr=p_q,
            offsets=q_rope_offsets,
            mask=q_rope_mask[:, None],
        )

        if TRANSFORM_Q_VIA_LDS:
            lds_q_nope.store(q_nope_data_0)
            lds_q_rope.store(q_rope_data)
            q_nope_0 = lds_q_nope.load(layout=dot_q_layout)
            lds_q_nope.store(q_nope_data_1)
            q_rope = lds_q_rope.load(layout=dot_q_layout)
            q_nope_1 = lds_q_nope.load(layout=dot_q_layout)
            lds_q_rope._keep_alive()
            lds_q_nope._keep_alive()
        else:
            q_nope_0 = gl.convert_layout(
                q_nope_data_0, dot_q_layout, assert_trivial=Q_LOAD_LAYOUT_DOT
            )
            q_nope_1 = gl.convert_layout(
                q_nope_data_1, dot_q_layout, assert_trivial=Q_LOAD_LAYOUT_DOT
            )
            q_rope = gl.convert_layout(
                q_rope_data, dot_q_layout, assert_trivial=Q_LOAD_LAYOUT_DOT
            )

        # Allocate LDS for K
        lds_kv_nope_0 = gl.allocate_shared_memory(
            p_k_buffer.type.element_ty,
            [kv_lora_rank // 2, BLOCK_N],
            layout=lds_layout_k,
        )
        lds_kv_nope_1 = gl.allocate_shared_memory(
            p_k_buffer.type.element_ty,
            [kv_lora_rank // 2, BLOCK_N],
            layout=lds_layout_k,
        )
        lds_k_rope = gl.allocate_shared_memory(
            p_k_buffer.type.element_ty, [qk_rope_head_dim, BLOCK_N], layout=lds_layout_k
        )
        if TRANSFORM_P_VIA_LDS:
            lds_p = gl.allocate_shared_memory(
                p_q.type.element_ty, [BLOCK_M, BLOCK_N], layout=lds_layout_p
            )
        else:
            lds_p = None

        # Main Loop
        acc_0, acc_1, row_max, row_sum_e = mla_core(
            kv_start,
            min(kv_start + BLOCK_N, kv_end),
            q_nope_0,
            q_nope_1,
            q_rope,
            p_k_buffer,
            p_kv_indices,
            stride_k_bs,
            kv_ld_nope_local_col_idx,
            kv_ld_rope_local_col_idx,
            kv_ld_nope_row_idx,
            kv_ld_rope_row_idx,
            lds_kv_nope_0,
            lds_kv_nope_1,
            lds_k_rope,
            lds_p,
            mfma_layout_qk,
            mfma_layout_qk,
            sm_scale,
            acc_0,
            acc_1,
            row_max,
            row_sum_e,
            True,
            BLOCK_M,
            BLOCK_N,
            TRANSFORM_P_VIA_LDS,
            kv_lora_rank,
            lds_layout_k,
            lds_layout_v,
        )
        for kv_col_start in range(kv_start + BLOCK_N, kv_end, BLOCK_N):
            acc_0, acc_1, row_max, row_sum_e = mla_core(
                kv_col_start,
                min(kv_col_start + BLOCK_N, kv_end),
                q_nope_0,
                q_nope_1,
                q_rope,
                p_k_buffer,
                p_kv_indices,
                stride_k_bs,
                kv_ld_nope_local_col_idx,
                kv_ld_rope_local_col_idx,
                kv_ld_nope_row_idx,
                kv_ld_rope_row_idx,
                lds_kv_nope_0,
                lds_kv_nope_1,
                lds_k_rope,
                lds_p,
                mfma_layout_qk,
                mfma_layout_qk,
                sm_scale,
                acc_0,
                acc_1,
                row_max,
                row_sum_e,
                False,
                BLOCK_M,
                BLOCK_N,
                TRANSFORM_P_VIA_LDS,
                kv_lora_rank,
                lds_layout_k,
                lds_layout_v,
            )

        # Release LDS for KV
        lds_kv_nope_0._keep_alive()
        lds_kv_nope_1._keep_alive()
        lds_k_rope._keep_alive()
        if TRANSFORM_P_VIA_LDS:
            lds_p._keep_alive()

        # Output results
        acc = gl.join(acc_0, acc_1)
        acc = gl.permute(acc, (0, 2, 1))
        acc = gl.reshape(acc, (BLOCK_M, kv_lora_rank))
        acc = gl.convert_layout(acc, mfma_layout_kv)
        out = acc * (1 / row_sum_e)[:, None]

        o_st_row_idx = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, mfma_layout_kv))
        o_st_col_idx = gl.arange(
            0, kv_lora_rank, layout=gl.SliceLayout(0, mfma_layout_kv)
        )
        out_mask = o_st_col_idx < kv_lora_rank

        if partial_qo_loc == -1:
            out_offsets = (
                qo_start * stride_final_out_bs
                + o_st_row_idx[:, None] * stride_final_out_h
                + o_st_col_idx[None, :]
            )
            gl.amd.cdna3.buffer_store(
                stored_value=out.cast(p_final_out.type.element_ty),
                ptr=p_final_out,
                offsets=out_offsets,
                mask=out_mask[None, :],
            )
        else:
            out_offsets = (
                partial_qo_loc * stride_temp_out_bs
                + o_st_row_idx[:, None] * stride_temp_out_h
                + o_st_col_idx[None, :]
            )
            gl.amd.cdna3.buffer_store(
                stored_value=out,
                ptr=p_temp_out,
                offsets=out_offsets,
                mask=out_mask[None, :],
            )
            lse = row_max + gl.log(row_sum_e)
            lse_offsets = (
                partial_qo_loc * stride_temp_lse_bs + o_st_row_idx * stride_temp_lse_h
            )
            gl.amd.cdna3.buffer_store(
                stored_value=lse,
                ptr=p_temp_lse,
                offsets=lse_offsets,
            )


def mla_fwd_m128(
    q,
    qo_indptr,
    kv_buffer,
    kv_indptr,
    kv_indices,
    final_out,
    tmp_out,
    tmp_lse,
    work_indptr,
    work_info_set,
    sm_scale,
    max_seqlen_q,
    kv_lora_rank,
):
    num_head_q = q.shape[1]
    qk_rope_head_dim = q.shape[-1] - kv_lora_rank

    assert (max_seqlen_q * num_head_q) % 128 == 0

    grid = (get_cu_num(),)

    config = {}
    # ROCM configs
    config["matrix_instr_nonkdim"] = 16
    # Kernel configs
    config["num_head_q"] = num_head_q
    config["same_kv"] = True
    config["kv_lora_rank"] = kv_lora_rank
    config["qk_rope_head_dim"] = qk_rope_head_dim

    if q.dtype == dtypes.fp8:
        # Gluon configs
        config["num_stages"] = 1
        config["num_warps"] = 8
        # ROCM configs
        config["waves_per_eu"] = 1
        config["BLOCK_M"] = 128
        config["BLOCK_N"] = 32
        config["BLOCK_K"] = 16

        kn_mla_fwd_fp8_m128_ps[grid](
            q,
            kv_buffer,
            kv_buffer,
            kv_indptr,
            kv_indices,
            final_out,
            tmp_out,
            tmp_lse,
            work_indptr,
            work_info_set,
            sm_scale,
            max_seqlen_q,
            q.stride(0),
            q.stride(1),
            kv_buffer.stride(0),
            kv_buffer.stride(0),
            final_out.stride(0),
            final_out.stride(1),
            tmp_out.stride(-3),
            tmp_out.stride(-2),
            tmp_lse.stride(-3),
            tmp_lse.stride(-2),
            **config,
        )
