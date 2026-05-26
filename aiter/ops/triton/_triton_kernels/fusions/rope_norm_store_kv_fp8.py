# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import triton
import triton.language as tl

from aiter.ops.triton._triton_kernels.rope.rope import _get_neox_rotated_x
from aiter.ops.triton._triton_kernels.fusions.rope_norm_store_kv import (
    _standard_rms_norm,
)


@triton.jit
def _rope_norm_store_kv_fp8_compute_pos_slot_kernel(
    q_index_ptr,             # [num_req+1] int32
    num_seqlen_per_req_ptr,  # [num_req]   int32
    kvcache_indices_ptr,     # [num_req, max_blocks] int32
    positions_ptr,           # [num_rows]  int32 OUT
    slot_indices_ptr,        # [num_rows]  int64 OUT (-1 = invalid)
    req_ids_ptr,             # [num_rows]  int32 OUT
    local_idx_ptr,           # [num_rows]  int32 OUT
    stride_kvi_r,
    stride_kvi_b,
    BLOCK_R: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Same as the BF16 helper, but also writes req_id and per-request local
    row index for each output row (needed by the FP8 kernel to address the
    [num_req, num_q_heads, max_seqlens_pad128] q_scale_out tensor in prefill).

    Skips writes for padding requests (seq_len == 0); callers must pre-fill
    slot_indices with -1 to mark untouched rows as invalid.
    """
    req = tl.program_id(0)
    start = tl.load(q_index_ptr + req).to(tl.int32)
    end = tl.load(q_index_ptr + req + 1).to(tl.int32)
    seq_len = tl.load(num_seqlen_per_req_ptr + req).to(tl.int32)
    num_rows_req = end - start

    if (seq_len > 0) & (num_rows_req > 0):
        pos_offset = seq_len - end
        num_chunks = tl.cdiv(num_rows_req, BLOCK_R)
        for chunk in tl.range(0, num_chunks):
            row_local = chunk * BLOCK_R + tl.arange(0, BLOCK_R)
            mask = row_local < num_rows_req
            row = start + row_local
            token_pos = row + pos_offset
            block_idx = token_pos // BLOCK_SIZE
            block_row = token_pos % BLOCK_SIZE
            phys_block = tl.load(
                kvcache_indices_ptr + req * stride_kvi_r + block_idx * stride_kvi_b,
                mask=mask, other=0,
            ).to(tl.int64)
            slot = phys_block * BLOCK_SIZE + block_row.to(tl.int64)
            tl.store(positions_ptr + row, token_pos, mask=mask)
            tl.store(slot_indices_ptr + row, slot, mask=mask)
            req_vec = tl.full([BLOCK_R], req, tl.int32)
            tl.store(req_ids_ptr + row, req_vec, mask=mask)
            tl.store(local_idx_ptr + row, row_local, mask=mask)


@triton.jit
def _rope_norm_store_kv_fp8_kernel(
    qkv_ptr,                 # [num_rows, hidden] bf16
    cos_sin_ptr,             # [max_seq_len, qk_head_dim] f32
    positions_ptr,           # [num_rows] int32
    slot_indices_ptr,        # [num_rows] int64 (-1 = invalid)
    req_ids_ptr,             # [num_rows] int32  (prefill dynamic-Q only)
    local_idx_ptr,           # [num_rows] int32  (prefill dynamic-Q only)
    q_norm_weight_ptr,
    k_norm_weight_ptr,
    hadamard_ptr,            # [qk_head_dim, qk_head_dim] bf16 (pre-normalized by 1/sqrt(d))
    q_scale_inv_ptr,         # [1] f32 (static Q only)
    k_scale_ptr,             # paged [num_blocks, R, num_kv_heads, L] (dynamic) or [1] (static)
    v_scale_ptr,             # [num_kv_heads] (dynamic V per-head) or [1] (static)
    q_scale_out_ptr,         # [num_req, num_q_heads, pad128] (prefill) or [num_rows, num_q_heads] (decode); dynamic Q
    out_q_ptr,               # FP8 [num_rows, num_q_heads, qk_head_dim]
    out_k_ptr,               # FP8 (optional)
    out_v_ptr,               # FP8 (optional)
    key_cache_ptr,           # FP8 [num_blocks, block_size, num_kv_heads, qk_head_dim]
    value_cache_ptr,         # FP8 [num_blocks, block_size, num_kv_heads, v_head_dim]
    eps,
    num_rows,
    total_num_kv_cache_tokens: tl.int64,
    fp8_max,
    # qkv strides
    stride_qkv_t, stride_qkv_d,
    # cos_sin strides
    stride_cos_t, stride_cos_d,
    # out_q strides
    stride_out_q_t, stride_out_q_h, stride_out_q_d,
    # out_k strides (only used if WRITE_K_TO_CACHE=False)
    stride_out_k_t, stride_out_k_h, stride_out_k_d,
    # out_v strides (only used if WRITE_V_TO_CACHE=False)
    stride_out_v_t, stride_out_v_h, stride_out_v_d,
    # key_cache strides
    stride_kc_b, stride_kc_t, stride_kc_h, stride_kc_d,
    # value_cache strides
    stride_vc_b, stride_vc_t, stride_vc_h, stride_vc_d,
    # k_scale strides (dynamic path only)
    stride_ks_b, stride_ks_r, stride_ks_h, stride_ks_l,
    # q_scale_out strides (dynamic-Q path)
    stride_qs_0, stride_qs_1, stride_qs_2,
    # constexprs
    NUM_Q_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    QK_HEAD_DIM: tl.constexpr,
    QK_HEAD_DIM_HALF: tl.constexpr,
    V_HEAD_DIM: tl.constexpr,
    V_HEAD_DIM_PAD: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_T: tl.constexpr,
    QK_NORM_POLICY: tl.constexpr,      # 0/1/2
    APPLY_Q_NORM: tl.constexpr,
    APPLY_K_NORM: tl.constexpr,
    Q_QUANT_DYNAMIC: tl.constexpr,     # True for policy 0/1/3, False for 2
    K_QUANT_DYNAMIC: tl.constexpr,     # True for policy 0/3, False for 1/2
    V_QUANT_PERHEAD: tl.constexpr,     # True for policy 0/3, False for 1/2
    APPLY_HADAMARD: tl.constexpr,      # True for policy 3
    IS_PREFILL: tl.constexpr,
    WRITE_K_TO_CACHE: tl.constexpr,
    WRITE_V_TO_CACHE: tl.constexpr,
    K_SCALE_L: tl.constexpr,           # qk_head_dim // 4 (sizeof(fp8)/sizeof(float))
):
    tl.assume(stride_qkv_t > 0)
    tl.assume(stride_qkv_d > 0)
    tl.assume(stride_cos_t > 0)
    tl.assume(stride_cos_d > 0)

    pid_t = tl.program_id(0)
    hq = tl.program_id(1)
    tl.assume(pid_t >= 0)
    tl.assume(hq >= 0)

    t_offs = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    t_mask = t_offs < num_rows

    # ===== Positions / slots / req / local_idx =====
    positions = tl.load(positions_ptr + t_offs, mask=t_mask, other=0).to(tl.int32)
    slots = tl.load(slot_indices_ptr + t_offs, mask=t_mask, other=-1)
    valid_row = (
        t_mask
        & (slots >= 0)
        & (slots < total_num_kv_cache_tokens)
    )
    safe_slots = tl.where(valid_row, slots, 0)
    phys_block = safe_slots // BLOCK_SIZE
    block_row = (safe_slots % BLOCK_SIZE).to(tl.int32)

    # ===== cos/sin (NeoX: cos/sin[d] = table[d % half]) =====
    d = tl.arange(0, QK_HEAD_DIM)
    d_mod = d % QK_HEAD_DIM_HALF
    cos_offs = positions[:, None] * stride_cos_t + d_mod[None, :] * stride_cos_d
    sin_offs = cos_offs + QK_HEAD_DIM_HALF * stride_cos_d
    cos_full = tl.load(cos_sin_ptr + cos_offs, mask=t_mask[:, None], other=1.0)
    sin_full = tl.load(cos_sin_ptr + sin_offs, mask=t_mask[:, None], other=0.0)
    qk_rotated_mask = (d < QK_HEAD_DIM_HALF)[None, :]

    # Load Hadamard once if needed
    if APPLY_HADAMARD:
        had_offs = (
            tl.arange(0, QK_HEAD_DIM)[:, None] * QK_HEAD_DIM
            + tl.arange(0, QK_HEAD_DIM)[None, :]
        )
        H = tl.load(hadamard_ptr + had_offs)

    # ===== Q =====
    q_off_base = hq * QK_HEAD_DIM
    q_in_offs = t_offs[:, None] * stride_qkv_t + (q_off_base + d)[None, :] * stride_qkv_d
    q_load_mask = t_mask[:, None]
    q = tl.load(qkv_ptr + q_in_offs, mask=q_load_mask).to(tl.float32)

    if QK_NORM_POLICY == 2:
        if APPLY_Q_NORM:
            q = _standard_rms_norm(q, q_norm_weight_ptr, QK_HEAD_DIM, eps)

    q_rot_in = _get_neox_rotated_x(
        q, qk_rotated_mask, BLOCK_T, QK_HEAD_DIM, QK_HEAD_DIM_HALF
    )
    q = q * cos_full + q_rot_in * sin_full

    if QK_NORM_POLICY == 1:
        if APPLY_Q_NORM:
            q = _standard_rms_norm(q, q_norm_weight_ptr, QK_HEAD_DIM, eps)

    if APPLY_HADAMARD:
        # Per-head Walsh-Hadamard with 1/sqrt(d) baked into H.
        q = tl.dot(q.to(H.dtype), H).to(tl.float32)

    # ===== Q quant + write =====
    if Q_QUANT_DYNAMIC:
        q_amax = tl.max(tl.abs(q), axis=-1)
        q_scale = q_amax / fp8_max
        q_scale = tl.maximum(q_scale, 1e-12)
        q_scale_recip = 1.0 / q_scale
        q_quant = q * q_scale_recip[:, None]
        if IS_PREFILL:
            req_ids = tl.load(req_ids_ptr + t_offs, mask=t_mask, other=0).to(tl.int32)
            local_idx = tl.load(local_idx_ptr + t_offs, mask=t_mask, other=0).to(tl.int32)
            qs_offs = (
                req_ids * stride_qs_0
                + hq * stride_qs_1
                + local_idx * stride_qs_2
            )
            tl.store(q_scale_out_ptr + qs_offs, q_scale, mask=valid_row)
        else:
            qs_offs = t_offs * stride_qs_0 + hq * stride_qs_1
            tl.store(q_scale_out_ptr + qs_offs, q_scale, mask=t_mask)
    else:
        q_scale_inv = tl.load(q_scale_inv_ptr)
        q_quant = q * q_scale_inv

    q_quant = tl.clamp(q_quant, -fp8_max, fp8_max)
    q_out_offs = (
        t_offs[:, None] * stride_out_q_t
        + hq * stride_out_q_h
        + d[None, :] * stride_out_q_d
    )
    tl.store(
        out_q_ptr + q_out_offs,
        q_quant.to(out_q_ptr.dtype.element_ty),
        mask=q_load_mask,
    )

    # ===== KV path =====
    if hq < NUM_KV_HEADS:
        q_dim = NUM_Q_HEADS * QK_HEAD_DIM
        k_dim = NUM_KV_HEADS * QK_HEAD_DIM

        k_off_base = q_dim + hq * QK_HEAD_DIM
        k_in_offs = t_offs[:, None] * stride_qkv_t + (k_off_base + d)[None, :] * stride_qkv_d
        k = tl.load(qkv_ptr + k_in_offs, mask=q_load_mask).to(tl.float32)

        if QK_NORM_POLICY == 2:
            if APPLY_K_NORM:
                k = _standard_rms_norm(k, k_norm_weight_ptr, QK_HEAD_DIM, eps)

        k_rot_in = _get_neox_rotated_x(
            k, qk_rotated_mask, BLOCK_T, QK_HEAD_DIM, QK_HEAD_DIM_HALF
        )
        k = k * cos_full + k_rot_in * sin_full

        if QK_NORM_POLICY == 1:
            if APPLY_K_NORM:
                k = _standard_rms_norm(k, k_norm_weight_ptr, QK_HEAD_DIM, eps)

        if APPLY_HADAMARD:
            k = tl.dot(k.to(H.dtype), H).to(tl.float32)

        # ===== K quant =====
        if K_QUANT_DYNAMIC:
            k_amax = tl.max(tl.abs(k), axis=-1)
            k_scale_dyn = k_amax / fp8_max
            k_scale_dyn = tl.maximum(k_scale_dyn, 1e-12)
            k_scale_dyn_recip = 1.0 / k_scale_dyn
            k_quant = k * k_scale_dyn_recip[:, None]
            # Write per-(slot, kv_head) scale to paged layout
            # idx = cb * stride_b + (pb // L) * stride_r + h * stride_h + (pb % L) * stride_l
            r_idx = block_row // K_SCALE_L
            l_idx = block_row % K_SCALE_L
            ks_offs = (
                phys_block * stride_ks_b
                + r_idx.to(tl.int64) * stride_ks_r
                + hq * stride_ks_h
                + l_idx.to(tl.int64) * stride_ks_l
            )
            tl.store(k_scale_ptr + ks_offs, k_scale_dyn, mask=valid_row)
        else:
            k_scale_static = tl.load(k_scale_ptr)
            k_quant = k / k_scale_static
        k_quant = tl.clamp(k_quant, -fp8_max, fp8_max)

        # ===== V quant =====
        d_v = tl.arange(0, V_HEAD_DIM_PAD)
        v_mask_d = d_v < V_HEAD_DIM
        v_off_base = q_dim + k_dim + hq * V_HEAD_DIM
        v_in_offs = t_offs[:, None] * stride_qkv_t + (v_off_base + d_v)[None, :] * stride_qkv_d
        v_load_mask = t_mask[:, None] & v_mask_d[None, :]
        v = tl.load(qkv_ptr + v_in_offs, mask=v_load_mask).to(tl.float32)

        if V_QUANT_PERHEAD:
            v_scale_h = tl.load(v_scale_ptr + hq)
        else:
            v_scale_h = tl.load(v_scale_ptr)
        v_quant = v / v_scale_h
        v_quant = tl.clamp(v_quant, -fp8_max, fp8_max)

        # ===== Store K =====
        if WRITE_K_TO_CACHE:
            k_cache_offs = (
                phys_block[:, None] * stride_kc_b
                + block_row[:, None].to(tl.int64) * stride_kc_t
                + hq * stride_kc_h
                + d[None, :] * stride_kc_d
            )
            tl.store(
                key_cache_ptr + k_cache_offs,
                k_quant.to(key_cache_ptr.dtype.element_ty),
                mask=valid_row[:, None],
            )
        else:
            k_out_offs = (
                t_offs[:, None] * stride_out_k_t
                + hq * stride_out_k_h
                + d[None, :] * stride_out_k_d
            )
            tl.store(
                out_k_ptr + k_out_offs,
                k_quant.to(out_k_ptr.dtype.element_ty),
                mask=q_load_mask,
            )

        # ===== Store V =====
        if WRITE_V_TO_CACHE:
            v_cache_offs = (
                phys_block[:, None] * stride_vc_b
                + block_row[:, None].to(tl.int64) * stride_vc_t
                + hq * stride_vc_h
                + d_v[None, :] * stride_vc_d
            )
            v_cache_mask = valid_row[:, None] & v_mask_d[None, :]
            tl.store(
                value_cache_ptr + v_cache_offs,
                v_quant.to(value_cache_ptr.dtype.element_ty),
                mask=v_cache_mask,
            )
        else:
            v_out_offs = (
                t_offs[:, None] * stride_out_v_t
                + hq * stride_out_v_h
                + d_v[None, :] * stride_out_v_d
            )
            tl.store(
                out_v_ptr + v_out_offs,
                v_quant.to(out_v_ptr.dtype.element_ty),
                mask=v_load_mask,
            )


@triton.jit
def _rope_norm_store_kv_fp8_zero_trailing_kernel(
    num_seqlen_per_req_ptr,
    kvcache_indices_ptr,
    key_cache_ptr,
    value_cache_ptr,
    stride_kvi_r,
    stride_kvi_b,
    stride_kc_b, stride_kc_t, stride_kc_h, stride_kc_d,
    stride_vc_b, stride_vc_t, stride_vc_h, stride_vc_d,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_SIZE_PAD: tl.constexpr,
    QK_HEAD_DIM: tl.constexpr,
    QK_HEAD_DIM_PAD: tl.constexpr,
    V_HEAD_DIM: tl.constexpr,
    V_HEAD_DIM_PAD: tl.constexpr,
):
    req = tl.program_id(0)
    h = tl.program_id(1)
    seq_len = tl.load(num_seqlen_per_req_ptr + req).to(tl.int32)
    if seq_len > 0:
        last_pos = seq_len - 1
        last_block_idx = last_pos // BLOCK_SIZE
        last_block_row = last_pos % BLOCK_SIZE
        phys_block = tl.load(
            kvcache_indices_ptr + req * stride_kvi_r + last_block_idx * stride_kvi_b
        ).to(tl.int64)

        slot_offs = tl.arange(0, BLOCK_SIZE_PAD)
        slot_mask = (slot_offs > last_block_row) & (slot_offs < BLOCK_SIZE)

        d_qk = tl.arange(0, QK_HEAD_DIM_PAD)
        k_offs = (
            phys_block * stride_kc_b
            + slot_offs[:, None] * stride_kc_t
            + h * stride_kc_h
            + d_qk[None, :] * stride_kc_d
        )
        k_mask = slot_mask[:, None] & (d_qk < QK_HEAD_DIM)[None, :]
        tl.store(
            key_cache_ptr + k_offs,
            tl.zeros(
                [BLOCK_SIZE_PAD, QK_HEAD_DIM_PAD],
                dtype=key_cache_ptr.dtype.element_ty,
            ),
            mask=k_mask,
        )

        d_v = tl.arange(0, V_HEAD_DIM_PAD)
        v_offs = (
            phys_block * stride_vc_b
            + slot_offs[:, None] * stride_vc_t
            + h * stride_vc_h
            + d_v[None, :] * stride_vc_d
        )
        v_mask = slot_mask[:, None] & (d_v < V_HEAD_DIM)[None, :]
        tl.store(
            value_cache_ptr + v_offs,
            tl.zeros(
                [BLOCK_SIZE_PAD, V_HEAD_DIM_PAD],
                dtype=value_cache_ptr.dtype.element_ty,
            ),
            mask=v_mask,
        )
