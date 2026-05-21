# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import triton
import triton.language as tl

from aiter.ops.triton._triton_kernels.rope.rope import _get_neox_rotated_x


@triton.jit
def _standard_rms_norm(tensor_f32, weight_ptr, BLOCK_D: tl.constexpr, eps):
    """Standard RMSNorm: y = (x / sqrt(mean(x^2) + eps)) * weight.

    Reference uses sqrt(mean+eps), not rsqrt(mean+eps), so we match exactly.
    """
    tensor_sq = tensor_f32 * tensor_f32
    mean_sq = tl.sum(tensor_sq, axis=1) / BLOCK_D
    rms = tl.sqrt(mean_sq + eps)[:, None]
    w = tl.load(weight_ptr + tl.arange(0, BLOCK_D)).to(tl.float32)
    return tensor_f32 / rms * w[None, :]


@triton.jit
def _rope_norm_store_kv_compute_pos_slot_kernel(
    q_index_ptr,             # [num_req+1] int32
    num_seqlen_per_req_ptr,  # [num_req]   int32
    kvcache_indices_ptr,     # [num_req, max_blocks] int32
    positions_ptr,           # [num_rows]  int32 OUT
    slot_indices_ptr,        # [num_rows]  int64 OUT
    stride_kvi_r,
    stride_kvi_b,
    BLOCK_R: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    req = tl.program_id(0)

    start = tl.load(q_index_ptr + req).to(tl.int32)
    end = tl.load(q_index_ptr + req + 1).to(tl.int32)
    seq_len = tl.load(num_seqlen_per_req_ptr + req).to(tl.int32)

    num_rows_req = end - start
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
            mask=mask,
            other=0,
        ).to(tl.int64)
        slot = phys_block * BLOCK_SIZE + block_row.to(tl.int64)
        tl.store(positions_ptr + row, token_pos, mask=mask)
        tl.store(slot_indices_ptr + row, slot, mask=mask)


@triton.jit
def _rope_norm_store_kv_kernel(
    qkv_ptr,                 # [num_rows, hidden]                  bf16
    cos_sin_ptr,             # [max_seq_len, qk_head_dim]          f32  (cos | sin halves)
    positions_ptr,           # [num_rows]                          int32
    slot_indices_ptr,        # [num_rows]                          int64 (slot per row)
    q_norm_weight_ptr,       # [qk_head_dim] f32 (unused if APPLY_Q_NORM=False)
    k_norm_weight_ptr,       # [qk_head_dim] f32 (unused if APPLY_K_NORM=False)
    out_q_ptr,               # [num_rows, num_q_heads,  qk_head_dim] bf16
    out_k_ptr,               # null or [num_rows, num_kv_heads, qk_head_dim] bf16
    out_v_ptr,               # null or [num_rows, num_kv_heads,  v_head_dim] bf16
    key_cache_ptr,           # [num_blocks, block_size, num_kv_heads, qk_head_dim] bf16
    value_cache_ptr,         # [num_blocks, block_size, num_kv_heads,  v_head_dim] bf16
    eps,
    num_rows,
    total_num_kv_cache_tokens: tl.int64,
    # strides
    stride_qkv_t, stride_qkv_d,
    stride_cos_t, stride_cos_d,
    stride_out_q_t, stride_out_q_h, stride_out_q_d,
    stride_out_k_t, stride_out_k_h, stride_out_k_d,
    stride_out_v_t, stride_out_v_h, stride_out_v_d,
    stride_kc_b, stride_kc_t, stride_kc_h, stride_kc_d,
    stride_vc_b, stride_vc_t, stride_vc_h, stride_vc_d,
    # constexprs
    NUM_Q_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    QK_HEAD_DIM: tl.constexpr,
    QK_HEAD_DIM_HALF: tl.constexpr,
    V_HEAD_DIM: tl.constexpr,
    V_HEAD_DIM_PAD: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_T: tl.constexpr,
    QK_NORM_POLICY: tl.constexpr,      # 0: no norm; 1: RoPE -> Norm; 2: Norm -> RoPE
    APPLY_Q_NORM: tl.constexpr,
    APPLY_K_NORM: tl.constexpr,
    WRITE_K_TO_CACHE: tl.constexpr,
    WRITE_V_TO_CACHE: tl.constexpr,
):
    tl.assume(stride_qkv_t > 0)
    tl.assume(stride_qkv_d > 0)
    tl.assume(stride_cos_t > 0)
    tl.assume(stride_cos_d > 0)
    tl.assume(stride_out_q_t > 0)
    tl.assume(stride_out_q_h > 0)
    tl.assume(stride_out_q_d > 0)

    pid_t = tl.program_id(0)
    hq = tl.program_id(1)
    tl.assume(pid_t >= 0)
    tl.assume(hq >= 0)

    t_offs = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    t_mask = t_offs < num_rows

    # ===== Position + cos/sin gather =====
    positions = tl.load(positions_ptr + t_offs, mask=t_mask, other=0).to(tl.int32)

    d = tl.arange(0, QK_HEAD_DIM)
    d_mod = d % QK_HEAD_DIM_HALF  # NeoX: cos/sin[d] = table[d % half]

    cos_offs = positions[:, None] * stride_cos_t + d_mod[None, :] * stride_cos_d
    sin_offs = cos_offs + QK_HEAD_DIM_HALF * stride_cos_d
    cos_full = tl.load(cos_sin_ptr + cos_offs, mask=t_mask[:, None], other=1.0)
    sin_full = tl.load(cos_sin_ptr + sin_offs, mask=t_mask[:, None], other=0.0)

    qk_rotated_mask = (d < QK_HEAD_DIM_HALF)[None, :]

    # ===== Q-head load + norm + RoPE =====
    q_off_base = hq * QK_HEAD_DIM
    q_in_offs = t_offs[:, None] * stride_qkv_t + (q_off_base + d)[None, :] * stride_qkv_d
    q_mask = t_mask[:, None]
    q = tl.load(qkv_ptr + q_in_offs, mask=q_mask).to(tl.float32)

    if QK_NORM_POLICY == 2:
        if APPLY_Q_NORM:
            q = _standard_rms_norm(q, q_norm_weight_ptr, QK_HEAD_DIM, eps)

    q_rot = _get_neox_rotated_x(
        q, qk_rotated_mask, BLOCK_T, QK_HEAD_DIM, QK_HEAD_DIM_HALF
    )
    q = q * cos_full + q_rot * sin_full

    if QK_NORM_POLICY == 1:
        if APPLY_Q_NORM:
            q = _standard_rms_norm(q, q_norm_weight_ptr, QK_HEAD_DIM, eps)

    q_out_offs = (
        t_offs[:, None] * stride_out_q_t
        + hq * stride_out_q_h
        + d[None, :] * stride_out_q_d
    )
    tl.store(out_q_ptr + q_out_offs, q.to(out_q_ptr.dtype.element_ty), mask=q_mask)

    # ===== KV path (hq < NUM_KV_HEADS) =====
    if hq < NUM_KV_HEADS:
        q_dim = NUM_Q_HEADS * QK_HEAD_DIM
        k_dim = NUM_KV_HEADS * QK_HEAD_DIM

        k_off_base = q_dim + hq * QK_HEAD_DIM
        k_in_offs = t_offs[:, None] * stride_qkv_t + (k_off_base + d)[None, :] * stride_qkv_d
        k = tl.load(qkv_ptr + k_in_offs, mask=q_mask).to(tl.float32)

        if QK_NORM_POLICY == 2:
            if APPLY_K_NORM:
                k = _standard_rms_norm(k, k_norm_weight_ptr, QK_HEAD_DIM, eps)

        k_rot = _get_neox_rotated_x(
            k, qk_rotated_mask, BLOCK_T, QK_HEAD_DIM, QK_HEAD_DIM_HALF
        )
        k = k * cos_full + k_rot * sin_full

        if QK_NORM_POLICY == 1:
            if APPLY_K_NORM:
                k = _standard_rms_norm(k, k_norm_weight_ptr, QK_HEAD_DIM, eps)

        # Load V (different head_dim, possibly non-pow2)
        d_v = tl.arange(0, V_HEAD_DIM_PAD)
        v_mask_d = d_v < V_HEAD_DIM
        v_off_base = q_dim + k_dim + hq * V_HEAD_DIM
        v_in_offs = t_offs[:, None] * stride_qkv_t + (v_off_base + d_v)[None, :] * stride_qkv_d
        v_load_mask = t_mask[:, None] & v_mask_d[None, :]
        v = tl.load(qkv_ptr + v_in_offs, mask=v_load_mask)

        # Slot lookup is needed if K or V is going to cache
        if WRITE_K_TO_CACHE or WRITE_V_TO_CACHE:
            slots = tl.load(slot_indices_ptr + t_offs, mask=t_mask, other=-1)
            # Defensive bounds check (matches existing fused_kv_cache pattern)
            valid_slot = (
                t_mask
                & (slots >= 0)
                & (slots < total_num_kv_cache_tokens)
            )
            safe_slots = tl.where(valid_slot, slots, 0)
            phys_block = safe_slots // BLOCK_SIZE
            block_row = safe_slots % BLOCK_SIZE

        # ===== Store K =====
        if WRITE_K_TO_CACHE:
            k_cache_offs = (
                phys_block[:, None] * stride_kc_b
                + block_row[:, None] * stride_kc_t
                + hq * stride_kc_h
                + d[None, :] * stride_kc_d
            )
            tl.store(
                key_cache_ptr + k_cache_offs,
                k.to(key_cache_ptr.dtype.element_ty),
                mask=valid_slot[:, None],
            )
        else:
            k_out_offs = (
                t_offs[:, None] * stride_out_k_t
                + hq * stride_out_k_h
                + d[None, :] * stride_out_k_d
            )
            tl.store(
                out_k_ptr + k_out_offs,
                k.to(out_k_ptr.dtype.element_ty),
                mask=q_mask,
            )

        # ===== Store V =====
        if WRITE_V_TO_CACHE:
            v_cache_offs = (
                phys_block[:, None] * stride_vc_b
                + block_row[:, None] * stride_vc_t
                + hq * stride_vc_h
                + d_v[None, :] * stride_vc_d
            )
            v_cache_mask = valid_slot[:, None] & v_mask_d[None, :]
            tl.store(
                value_cache_ptr + v_cache_offs,
                v.to(value_cache_ptr.dtype.element_ty),
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
                v.to(out_v_ptr.dtype.element_ty),
                mask=v_load_mask,
            )


@triton.jit
def _rope_norm_store_kv_zero_trailing_kernel(
    num_seqlen_per_req_ptr,  # [num_req]
    kvcache_indices_ptr,     # [num_req, max_blocks]
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

        # Zero K trailing slots for this head
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

        # Zero V trailing slots for this head
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
