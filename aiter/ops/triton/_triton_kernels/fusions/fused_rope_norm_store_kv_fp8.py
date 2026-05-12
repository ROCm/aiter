# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Fused RoPE + (optional) RMSNorm + dynamic per-token-per-head FP8 quant on
Q,K, static per-head FP8 quant on V, paged KV cache write, and K-scale slab
write. PyTorch reference: ``rope_norm_store_fp8.py``.

Grid: 1D, ``num_rows * (QH + 2 * KH)`` programs. Layout:
- ``pid in [0, T*QH)``           — Q phase (RoPE/Norm/quant, write out_q + q_scale).
- ``pid in [T*QH, T*QH+T*KH)``   — K phase (RoPE/Norm/quant, write key_cache + k_scale slab).
- ``pid in [T*QH+T*KH, end)``    — V phase (static quant, write value_cache).

One program == one (token, head). Each program issues a single
``BLOCK_D`` (head_dim) row load, does the per-row reduce (RMSNorm var, FP8
amax) in fp32 registers, and writes the final fp8 row in one store, so a
typical (head_dim=128) head fits well in a single warp.

Optimizations reused from the rest of ``_triton_kernels/fusions``:
- precomputed (token → req, token → kv_pos) maps from the wrapper so the
  kernel doesn't binary-search ``q_index`` per program (cf. ``token_to_req``
  flow in ``fused_reduce_qk_norm_rope_swa_write``);
- shared cos/sin index calc for both Q and K phases via REUSE_FREQS_FRONT_PART;
- ``_get_neox_rotated_x_1D`` / ``_get_gptj_rotated_x_1D`` helpers shared with
  ``fused_kv_cache``;
- ``num_warps=1`` per program (each program is one head row).
"""

import triton
import triton.language as tl

from aiter.ops.triton._triton_kernels.rope.rope import (
    _get_neox_rotated_x_1D,
    _get_gptj_rotated_x_1D,
)


@triton.jit
def _rope_1d(
    x,
    cos,
    sin,
    d_offs,
    IS_NEOX: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_D_HALF: tl.constexpr,
):
    if IS_NEOX:
        rot_mask = d_offs < BLOCK_D_HALF
        x_rot = _get_neox_rotated_x_1D(x, rot_mask, BLOCK_D, BLOCK_D_HALF)
    else:
        rot_mask = d_offs % 2 == 0
        x_rot = _get_gptj_rotated_x_1D(x, rot_mask, BLOCK_D, BLOCK_D_HALF)
    return x * cos + x_rot * sin


@triton.jit
def _rms_norm_1d(
    x,
    w,
    HAS_W: tl.constexpr,
    BLOCK_D: tl.constexpr,
    EPS: tl.constexpr,
):
    """x: [BLOCK_D] fp32, w: [BLOCK_D] fp32 (only used if HAS_W)."""
    var = tl.sum(x * x, axis=-1) / BLOCK_D
    inv = tl.math.rsqrt(var + EPS)
    y = x * inv
    if HAS_W:
        y = y * w
    return y


@triton.jit
def _dyn_per_token_fp8_quant(
    x,
    DTYPE_MAX: tl.constexpr,
):
    """Dynamic per-token (== per-row) FP8 quantization. Returns (qx_clamped_fp32, scale_fp32)."""
    amax = tl.max(tl.abs(x), axis=-1)
    amax = tl.where(amax < 1e-12, tl.full((), 1e-12, tl.float32), amax)
    scale = amax / DTYPE_MAX
    qx = x / scale
    qx = tl.clamp(qx, -DTYPE_MAX, DTYPE_MAX)
    return qx, scale


@triton.jit
def _fused_rope_norm_store_kv_fp8_kernel(
    # Inputs / outputs
    qkv_ptr,                   # [T, hidden] bf16, hidden = QH*D + 2*KH*D
    out_q_ptr,                 # [T, QH, D]  fp8 (write)
    q_scale_flat_ptr,          # [T, QH]     fp32 (write, flat layout)
    key_cache_ptr,             # [num_blocks, BLOCK_SIZE, KH, D] fp8 (write)
    value_cache_ptr,           # [num_blocks, BLOCK_SIZE, KH, D] fp8 (write)
    k_scale_ptr,               # [num_blocks, SCALE_ROWS, KH, SCALE_COLS] fp32 (write)
    v_scale_ptr,               # [KH] fp32 (read)
    q_norm_weight_ptr,         # [D] fp32 (read, optional)
    k_norm_weight_ptr,         # [D] fp32 (read, optional)
    cos_sin_ptr,               # [max_seq_len, D] fp32 ([:D/2]=cos, [D/2:]=sin)
    token_to_req_ptr,          # [T] int32
    token_kv_pos_ptr,          # [T] int32
    kvcache_indices_ptr,       # [num_req, max_blocks] int32
    # Strides
    qkv_stride_t,
    out_q_stride_t, out_q_stride_h, out_q_stride_d,
    q_scale_stride_t, q_scale_stride_h,
    kc_stride_blk, kc_stride_pos, kc_stride_h, kc_stride_d,
    vc_stride_blk, vc_stride_pos, vc_stride_h, vc_stride_d,
    ks_stride_blk, ks_stride_row, ks_stride_h, ks_stride_col,
    cs_stride_t, cs_stride_d,
    kvi_stride_req,
    # Constants
    QH: tl.constexpr,
    KH: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_D_HALF: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    SCALE_COLS: tl.constexpr,
    DTYPE_MAX: tl.constexpr,
    IS_NEOX: tl.constexpr,
    QK_NORM_POLICY: tl.constexpr,      # 0=none, 1=rope→norm, 2=norm→rope
    HAS_Q_NORM_WEIGHT: tl.constexpr,
    HAS_K_NORM_WEIGHT: tl.constexpr,
    RMS_EPS: tl.constexpr,
    NUM_Q_PIDS: tl.constexpr,          # T*QH
    NUM_K_PIDS: tl.constexpr,          # T*KH
    Q_HIDDEN_OFFSET: tl.constexpr,     # 0
    K_HIDDEN_OFFSET: tl.constexpr,     # QH*D
    V_HIDDEN_OFFSET: tl.constexpr,     # (QH+KH)*D
):
    tl.assume(qkv_stride_t >= 0)
    tl.assume(out_q_stride_t >= 0)
    tl.assume(kc_stride_blk >= 0)
    tl.assume(vc_stride_blk >= 0)
    tl.assume(ks_stride_blk >= 0)
    tl.assume(cs_stride_t >= 0)
    tl.assume(kvi_stride_req >= 0)

    pid = tl.program_id(0)
    tl.assume(pid >= 0)

    d_offs = tl.arange(0, BLOCK_D).to(tl.int64)

    # REUSE_FREQS_FRONT_PART: cos/sin tables are size BLOCK_D_HALF but apply over BLOCK_D.
    # For NEOX, d_freq_idx = (d if d<H else d-H). For GPT-J, d_freq_idx = d//2.
    if IS_NEOX:
        d_cos_idx = tl.where(
            d_offs >= BLOCK_D_HALF, d_offs - BLOCK_D_HALF, d_offs
        ).to(d_offs.dtype)
    else:
        d_cos_idx = d_offs // 2

    if pid < NUM_Q_PIDS:
        # ─────────────────────────── Q phase ───────────────────────────
        t = (pid // QH).to(tl.int64)
        h = (pid % QH).to(tl.int64)

        pos = tl.load(token_kv_pos_ptr + t).to(tl.int64)

        cs_row = cos_sin_ptr + pos * cs_stride_t
        cos = tl.load(cs_row + d_cos_idx * cs_stride_d).to(tl.float32)
        sin = tl.load(
            cs_row + (d_cos_idx + BLOCK_D_HALF) * cs_stride_d
        ).to(tl.float32)

        q_base = qkv_ptr + t * qkv_stride_t + Q_HIDDEN_OFFSET + h * BLOCK_D
        x = tl.load(q_base + d_offs).to(tl.float32)

        if HAS_Q_NORM_WEIGHT:
            wq = tl.load(q_norm_weight_ptr + d_offs).to(tl.float32)
        else:
            wq = tl.zeros((BLOCK_D,), dtype=tl.float32)  # unused

        if QK_NORM_POLICY == 2:
            if HAS_Q_NORM_WEIGHT:
                x = _rms_norm_1d(x, wq, True, BLOCK_D, RMS_EPS)
            x = _rope_1d(x, cos, sin, d_offs, IS_NEOX, BLOCK_D, BLOCK_D_HALF)
        elif QK_NORM_POLICY == 1:
            x = _rope_1d(x, cos, sin, d_offs, IS_NEOX, BLOCK_D, BLOCK_D_HALF)
            if HAS_Q_NORM_WEIGHT:
                x = _rms_norm_1d(x, wq, True, BLOCK_D, RMS_EPS)
        else:
            x = _rope_1d(x, cos, sin, d_offs, IS_NEOX, BLOCK_D, BLOCK_D_HALF)

        qx, scale = _dyn_per_token_fp8_quant(x, DTYPE_MAX)

        out_ptrs = (
            out_q_ptr
            + t * out_q_stride_t
            + h * out_q_stride_h
            + d_offs * out_q_stride_d
        )
        tl.store(out_ptrs, qx.to(out_q_ptr.dtype.element_ty))
        tl.store(
            q_scale_flat_ptr + t * q_scale_stride_t + h * q_scale_stride_h, scale
        )

    elif pid < NUM_Q_PIDS + NUM_K_PIDS:
        # ─────────────────────────── K phase ───────────────────────────
        kpid = pid - NUM_Q_PIDS
        t = (kpid // KH).to(tl.int64)
        h = (kpid % KH).to(tl.int64)

        pos = tl.load(token_kv_pos_ptr + t).to(tl.int64)
        req = tl.load(token_to_req_ptr + t).to(tl.int64)

        cs_row = cos_sin_ptr + pos * cs_stride_t
        cos = tl.load(cs_row + d_cos_idx * cs_stride_d).to(tl.float32)
        sin = tl.load(
            cs_row + (d_cos_idx + BLOCK_D_HALF) * cs_stride_d
        ).to(tl.float32)

        k_base = qkv_ptr + t * qkv_stride_t + K_HIDDEN_OFFSET + h * BLOCK_D
        x = tl.load(k_base + d_offs).to(tl.float32)

        if HAS_K_NORM_WEIGHT:
            wk = tl.load(k_norm_weight_ptr + d_offs).to(tl.float32)
        else:
            wk = tl.zeros((BLOCK_D,), dtype=tl.float32)

        if QK_NORM_POLICY == 2:
            if HAS_K_NORM_WEIGHT:
                x = _rms_norm_1d(x, wk, True, BLOCK_D, RMS_EPS)
            x = _rope_1d(x, cos, sin, d_offs, IS_NEOX, BLOCK_D, BLOCK_D_HALF)
        elif QK_NORM_POLICY == 1:
            x = _rope_1d(x, cos, sin, d_offs, IS_NEOX, BLOCK_D, BLOCK_D_HALF)
            if HAS_K_NORM_WEIGHT:
                x = _rms_norm_1d(x, wk, True, BLOCK_D, RMS_EPS)
        else:
            x = _rope_1d(x, cos, sin, d_offs, IS_NEOX, BLOCK_D, BLOCK_D_HALF)

        qx, scale = _dyn_per_token_fp8_quant(x, DTYPE_MAX)

        block_logical = pos // BLOCK_SIZE
        block_off = pos % BLOCK_SIZE
        block_phys = tl.load(
            kvcache_indices_ptr + req * kvi_stride_req + block_logical
        ).to(tl.int64)

        kc_ptrs = (
            key_cache_ptr
            + block_phys * kc_stride_blk
            + block_off * kc_stride_pos
            + h * kc_stride_h
            + d_offs * kc_stride_d
        )
        tl.store(kc_ptrs, qx.to(key_cache_ptr.dtype.element_ty))

        scale_row = block_off // SCALE_COLS
        scale_col = block_off % SCALE_COLS
        ks_offs = (
            block_phys * ks_stride_blk
            + scale_row * ks_stride_row
            + h * ks_stride_h
            + scale_col * ks_stride_col
        )
        tl.store(k_scale_ptr + ks_offs, scale)

    else:
        # ─────────────────────────── V phase ───────────────────────────
        vpid = pid - NUM_Q_PIDS - NUM_K_PIDS
        t = (vpid // KH).to(tl.int64)
        h = (vpid % KH).to(tl.int64)

        pos = tl.load(token_kv_pos_ptr + t).to(tl.int64)
        req = tl.load(token_to_req_ptr + t).to(tl.int64)

        v_base = qkv_ptr + t * qkv_stride_t + V_HIDDEN_OFFSET + h * BLOCK_D
        v = tl.load(v_base + d_offs).to(tl.float32)
        vs = tl.load(v_scale_ptr + h).to(tl.float32)
        vq = v / vs
        vq = tl.clamp(vq, -DTYPE_MAX, DTYPE_MAX)

        block_logical = pos // BLOCK_SIZE
        block_off = pos % BLOCK_SIZE
        block_phys = tl.load(
            kvcache_indices_ptr + req * kvi_stride_req + block_logical
        ).to(tl.int64)

        vc_ptrs = (
            value_cache_ptr
            + block_phys * vc_stride_blk
            + block_off * vc_stride_pos
            + h * vc_stride_h
            + d_offs * vc_stride_d
        )
        tl.store(vc_ptrs, vq.to(value_cache_ptr.dtype.element_ty))
