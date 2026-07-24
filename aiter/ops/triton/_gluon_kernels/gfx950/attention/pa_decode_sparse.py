# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Gluon (gfx950) DeepSeek-V4 sparse-MLA decode. Adapted from the vLLM DSv4 sparse
attention kernels.

K == V, so each tile is gathered once into one [BLOCK_K, HEAD] bf16 LDS buffer and
read permuted for QK, direct for PV. fp8 dequants to bf16 on the way in. One kernel
serves three KV formats via the UNIFORM constexpr on the fp8 gather:

    UNIFORM=False   packed fp8_ds_mla: NoPE fp8 + embedded UE8M0 + RoPE bf16
    UNIFORM=True    uniform pool: whole head fp8 + a separate fp32 kv_scales
    IS_FP8=False    bf16 pool

Two-loop (SWA + top-k) or a single merged segment; 2D and 3D (split-K + reduce)
share one kernel. Launchers: aiter/ops/triton/attention/pa_decode_sparse.py.
"""

from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr


@gluon.jit
def _cache_load(ptr, off, USE_BUFFER_LOAD: gl.constexpr, mask=None, other=None):
    # gfx950 buffer_load carries a 32-bit offset (2 GB cap); a cache past that gathers
    # via 64-bit gl.load instead. Valid slots always land in-bounds (full tiles are
    # peeled and slot-checked), so no extra masking is needed.
    if USE_BUFFER_LOAD:
        return gl.amd.cdna4.buffer_load(
            ptr=ptr, offsets=off.to(gl.int32), mask=mask, other=other
        )
    return gl.load(ptr + off.to(gl.int64), mask=mask, other=other)


@gluon.jit
def _fp8_to_f32(x_u8, FP8_FNUZ: gl.constexpr):
    # gfx950's native e4m3 cvt is OCP (float8e4nv). fnuz (float8e4b8) -> f32 has no
    # native cvt and lowers to a ~5x software unpack that spills; but fnuz -> bf16 is
    # cheap and fp8 -> bf16 is exact (3 mantissa bits), so route fnuz through bf16.
    if FP8_FNUZ:
        return x_u8.to(gl.float8e4b8, bitcast=True).to(gl.bfloat16).to(gl.float32)
    return x_u8.to(gl.float8e4nv, bitcast=True).to(gl.float32)


@gluon.jit
def _decode_tile(
    q_dot,
    cache_ptr,
    cache_bf16_ptr,
    indices_ptr,
    seg_start,
    k_start,
    hi,
    cs0,
    num_rows,
    m_i,
    l_i,
    acc,
    head_mask,
    qk_scale,
    kv_smem,
    offs_full,
    offs_rope,
    k_rng_slot,
    qk_layout: gl.constexpr,
    pv_layout: gl.constexpr,
    k_layout: gl.constexpr,
    v_layout: gl.constexpr,
    p_layout: gl.constexpr,
    gather_l: gl.constexpr,
    gather_rope_l: gl.constexpr,
    IS_FP8: gl.constexpr,
    BLOCK_SIZE: gl.constexpr,
    NOPE_DIM: gl.constexpr,
    ROPE_DIM: gl.constexpr,
    HEAD_SIZE: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_K: gl.constexpr,
    HEAD_ALIGNED: gl.constexpr,
    MASKED: gl.constexpr,
    UNIFORM: gl.constexpr,
    USE_BUFFER_LOAD: gl.constexpr,
    HAS_INVALID: gl.constexpr,
    FP8_FNUZ: gl.constexpr,
):
    """One KV tile -> online-softmax update. MASKED=False (peeled full tiles) drops
    the in-range / gather / score masking; MASKED=True (the tail) keeps it. When
    HAS_INVALID, full tiles also clamp -1 sentinels in-bounds for the gather and
    mask their scores to -inf (matching the tail's slot-validity handling)."""
    neg_inf = float("-inf")
    if not USE_BUFFER_LOAD:
        cs0 = cs0.to(gl.int64)  # >2 GB cache: 64-bit gather offsets (see _cache_load)
    k_pos = k_start + k_rng_slot
    if MASKED:
        in_range = k_pos < hi
        slot = gl.load(indices_ptr + seg_start + k_pos, mask=in_range, other=-1)
        valid1d = in_range & (slot >= 0) & (slot < num_rows)
        safe_slot = gl.where(valid1d, slot, 0)
    elif HAS_INVALID:
        slot = gl.load(indices_ptr + seg_start + k_pos)
        valid1d = slot >= 0  # -1 sentinels: clamp in-bounds, mask score below
        safe_slot = gl.where(valid1d, slot, 0)
    else:
        slot = gl.load(indices_ptr + seg_start + k_pos)
        safe_slot = slot
    block_idx = (safe_slot // BLOCK_SIZE).to(gl.int32)
    pos = (safe_slot % BLOCK_SIZE).to(gl.int32)
    block_idx_g = gl.convert_layout(block_idx, gl.SliceLayout(1, gather_l))
    pos_g = gl.convert_layout(pos, gl.SliceLayout(1, gather_l))
    if MASKED:
        valid_g = gl.convert_layout(valid1d, gl.SliceLayout(1, gather_l))

    if IS_FP8 and UNIFORM:
        # uniform pool: one fp8 gather over the whole head + separate fp32 scales.
        NGRP: gl.constexpr = HEAD_SIZE // 64
        kv_off = (block_idx_g * cs0 + pos_g * HEAD_SIZE)[:, None] + offs_full[None, :]
        scl_off = (block_idx_g * NGRP)[:, None] + (offs_full[None, :] // 64)
        if MASKED:
            x_u8 = _cache_load(
                cache_ptr, kv_off, USE_BUFFER_LOAD, mask=valid_g[:, None], other=0
            )
            sc = _cache_load(
                cache_bf16_ptr,
                scl_off,
                USE_BUFFER_LOAD,
                mask=valid_g[:, None],
                other=0.0,
            )
        else:
            x_u8 = _cache_load(cache_ptr, kv_off, USE_BUFFER_LOAD)
            sc = _cache_load(cache_bf16_ptr, scl_off, USE_BUFFER_LOAD)
        kv_smem.store((_fp8_to_f32(x_u8, FP8_FNUZ) * sc).to(gl.bfloat16))
    elif IS_FP8:
        # DSv4 packed fp8_ds_mla: NoPE fp8 + embedded UE8M0 + separate RoPE-bf16.
        nope_off = (block_idx_g * cs0 + pos_g * 576)[:, None] + offs_full[None, :]
        scl_off = (block_idx_g * cs0 + BLOCK_SIZE * 576 + pos_g * 8)[:, None] + (
            offs_full[None, :] // 64
        )
        if MASKED:
            x_u8 = _cache_load(
                cache_ptr, nope_off, USE_BUFFER_LOAD, mask=valid_g[:, None], other=0
            )
            exps = _cache_load(
                cache_ptr, scl_off, USE_BUFFER_LOAD, mask=valid_g[:, None], other=127
            )
        else:
            x_u8 = _cache_load(cache_ptr, nope_off, USE_BUFFER_LOAD)
            exps = _cache_load(cache_ptr, scl_off, USE_BUFFER_LOAD)
        x_fp8 = x_u8.to(gl.float8e4nv, bitcast=True)
        scales = gl.exp2(exps.to(gl.float32) - 127.0)
        k_nope = (x_fp8.to(gl.float32) * scales).to(gl.bfloat16)
        kv_smem.store(k_nope)
        block_idx_gr = gl.convert_layout(block_idx, gl.SliceLayout(1, gather_rope_l))
        pos_gr = gl.convert_layout(pos, gl.SliceLayout(1, gather_rope_l))
        rope_off = (block_idx_gr * (cs0 // 2) + pos_gr * 288 + 224)[
            :, None
        ] + offs_rope[None, :]
        if MASKED:
            valid_gr = gl.convert_layout(valid1d, gl.SliceLayout(1, gather_rope_l))
            k_rope = _cache_load(
                cache_bf16_ptr,
                rope_off,
                USE_BUFFER_LOAD,
                mask=valid_gr[:, None],
                other=0.0,
            )
        else:
            k_rope = _cache_load(cache_bf16_ptr, rope_off, USE_BUFFER_LOAD)
        kv_smem.slice(NOPE_DIM, ROPE_DIM, dim=1).store(k_rope)
    else:
        off = (block_idx_g * cs0 + pos_g * HEAD_SIZE)[:, None] + offs_full[None, :]
        if MASKED:
            kv = _cache_load(
                cache_bf16_ptr, off, USE_BUFFER_LOAD, mask=valid_g[:, None], other=0.0
            )
        else:
            kv = _cache_load(cache_bf16_ptr, off, USE_BUFFER_LOAD)
        kv_smem.store(kv)

    k = kv_smem.permute([1, 0]).load(k_layout)  # [HEAD_SIZE, BLOCK_K]
    S = gl.amd.cdna4.mfma(
        q_dot, k, gl.zeros([BLOCK_M, BLOCK_K], gl.float32, layout=qk_layout)
    )
    # exp2 softmax: qk_scale folds in log2(e) so we hit the HW exp2 directly.
    # Running max stays in raw-score space; masked cols (-inf) give exp2=0.
    COL_VALID: gl.constexpr = MASKED or HAS_INVALID  # valid1d defined in both cases
    NEED_MASK: gl.constexpr = COL_VALID or (not HEAD_ALIGNED)
    if NEED_MASK:
        if COL_VALID:
            col_mask = gl.convert_layout(valid1d, gl.SliceLayout(0, qk_layout))[None, :]
            if not HEAD_ALIGNED:
                col_mask = (
                    gl.convert_layout(head_mask, gl.SliceLayout(1, qk_layout))[:, None]
                    & col_mask
                )
        else:  # not HEAD_ALIGNED and no col invalidity -> head mask only
            col_mask = gl.convert_layout(head_mask, gl.SliceLayout(1, qk_layout))[
                :, None
            ]
        S = gl.where(col_mask, S, neg_inf)

    m_block = gl.max(S, axis=1)
    m_new = gl.maximum(m_i, m_block)
    m_new = gl.where(m_new > neg_inf, m_new, 0.0)  # guard all-masked rows
    m_new_s = m_new * qk_scale
    p = gl.exp2(S * qk_scale - m_new_s[:, None])
    alpha = gl.exp2(m_i * qk_scale - m_new_s)
    l_new = l_i * alpha + gl.sum(p, axis=1)

    v = kv_smem.load(v_layout)  # [BLOCK_K, HEAD_SIZE]
    p_dot = gl.convert_layout(p.to(gl.bfloat16), p_layout)
    alpha_pv = gl.convert_layout(alpha, gl.SliceLayout(1, pv_layout))
    acc = acc * alpha_pv[:, None]
    acc = gl.amd.cdna4.mfma(p_dot, v, acc)
    return m_new, l_new, acc


@gluon.jit
def _gd_fp8(
    cache_ptr,
    cache_bf16_ptr,
    indices_ptr,
    seg_start,
    k_start,
    cs0,
    offs_full,
    offs_rope,
    k_rng_slot,
    gather_l: gl.constexpr,
    gather_rope_l: gl.constexpr,
    BLOCK_SIZE: gl.constexpr,
    HEAD_SIZE: gl.constexpr,
    UNIFORM: gl.constexpr,
    USE_BUFFER_LOAD: gl.constexpr,
    HAS_INVALID: gl.constexpr,
    FP8_FNUZ: gl.constexpr,
):
    """Gather + dequant one full fp8 tile to bf16 regs (k_nope, k_rope). Split from
    the LDS-write/MFMA so the pipeline runs this tile's gather+dequant while the
    previous tile's MFMAs are on the matrix core -> the dequant hides behind them.
    UNIFORM: one fp8 gather + separate fp32 scales, no RoPE (k_rope is a dead
    alias). Else: packed NoPE fp8 (over-read) + embedded UE8M0 + RoPE bf16.
    Returns (k_nope, k_rope, valid); valid is the per-slot >=0 mask (all-True and
    DCE'd when HAS_INVALID is False), consumed by _qkpv_fp8 for the score mask."""
    slot = gl.load(indices_ptr + seg_start + k_start + k_rng_slot)
    if not USE_BUFFER_LOAD:
        cs0 = cs0.to(gl.int64)  # >2 GB cache: 64-bit gather offsets (see _cache_load)
    valid = slot >= 0
    if HAS_INVALID:
        slot = gl.where(valid, slot, 0)  # clamp -1 sentinels in-bounds for the gather
    block_idx = (slot // BLOCK_SIZE).to(gl.int32)
    pos = (slot % BLOCK_SIZE).to(gl.int32)
    bg = gl.convert_layout(block_idx, gl.SliceLayout(1, gather_l))
    pg = gl.convert_layout(pos, gl.SliceLayout(1, gather_l))
    if UNIFORM:
        NGRP: gl.constexpr = HEAD_SIZE // 64
        kv_off = (bg * cs0 + pg * HEAD_SIZE)[:, None] + offs_full[None, :]
        scl_off = (bg * NGRP)[:, None] + (offs_full[None, :] // 64)
        x_u8 = _cache_load(cache_ptr, kv_off, USE_BUFFER_LOAD)
        sc = _cache_load(cache_bf16_ptr, scl_off, USE_BUFFER_LOAD)
        k_nope = (_fp8_to_f32(x_u8, FP8_FNUZ) * sc).to(gl.bfloat16)
        k_rope = k_nope  # unused for UNIFORM (rope slice-store skipped) -> DCE'd
    else:
        nope_off = (bg * cs0 + pg * 576)[:, None] + offs_full[None, :]
        scl_off = (bg * cs0 + BLOCK_SIZE * 576 + pg * 8)[:, None] + (
            offs_full[None, :] // 64
        )
        x_u8 = _cache_load(cache_ptr, nope_off, USE_BUFFER_LOAD)
        exps = _cache_load(cache_ptr, scl_off, USE_BUFFER_LOAD)
        k_nope = (
            x_u8.to(gl.float8e4nv, bitcast=True).to(gl.float32)
            * gl.exp2(exps.to(gl.float32) - 127.0)
        ).to(gl.bfloat16)
        bgr = gl.convert_layout(block_idx, gl.SliceLayout(1, gather_rope_l))
        pgr = gl.convert_layout(pos, gl.SliceLayout(1, gather_rope_l))
        rope_off = (bgr * (cs0 // 2) + pgr * 288 + 224)[:, None] + offs_rope[None, :]
        k_rope = _cache_load(cache_bf16_ptr, rope_off, USE_BUFFER_LOAD)
    return k_nope, k_rope, valid


@gluon.jit
def _qkpv_fp8(
    k_nope,
    k_rope,
    valid,
    q_dot,
    m_i,
    l_i,
    acc,
    head_mask,
    qk_scale,
    kv_smem,
    qk_layout: gl.constexpr,
    pv_layout: gl.constexpr,
    k_layout: gl.constexpr,
    v_layout: gl.constexpr,
    p_layout: gl.constexpr,
    NOPE_DIM: gl.constexpr,
    ROPE_DIM: gl.constexpr,
    HEAD_SIZE: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_K: gl.constexpr,
    HEAD_ALIGNED: gl.constexpr,
    UNIFORM: gl.constexpr,
    HAS_INVALID: gl.constexpr,
):
    """Write a prefetched bf16 tile to LDS, then QK -> softmax -> PV. UNIFORM skips
    the RoPE slice-store (the whole head is one fp8 tile). When HAS_INVALID, mask the
    columns of -1-sentinel slots (``valid`` from _gd_fp8) to -inf."""
    neg_inf = float("-inf")
    kv_smem.store(k_nope)
    if not UNIFORM:
        kv_smem.slice(NOPE_DIM, ROPE_DIM, dim=1).store(k_rope)
    k = kv_smem.permute([1, 0]).load(k_layout)
    S = gl.amd.cdna4.mfma(
        q_dot, k, gl.zeros([BLOCK_M, BLOCK_K], gl.float32, layout=qk_layout)
    )
    NEED_MASK: gl.constexpr = HAS_INVALID or (not HEAD_ALIGNED)
    if NEED_MASK:
        if HAS_INVALID:
            col_mask = gl.convert_layout(valid, gl.SliceLayout(0, qk_layout))[None, :]
            if not HEAD_ALIGNED:
                col_mask = (
                    gl.convert_layout(head_mask, gl.SliceLayout(1, qk_layout))[:, None]
                    & col_mask
                )
        else:
            col_mask = gl.convert_layout(head_mask, gl.SliceLayout(1, qk_layout))[
                :, None
            ]
        S = gl.where(col_mask, S, neg_inf)
    m_block = gl.max(S, axis=1)
    m_new = gl.maximum(m_i, m_block)
    m_new = gl.where(m_new > neg_inf, m_new, 0.0)
    m_new_s = m_new * qk_scale
    p = gl.exp2(S * qk_scale - m_new_s[:, None])
    alpha = gl.exp2(m_i * qk_scale - m_new_s)
    l_new = l_i * alpha + gl.sum(p, axis=1)
    v = kv_smem.load(v_layout)
    p_dot = gl.convert_layout(p.to(gl.bfloat16), p_layout)
    alpha_pv = gl.convert_layout(alpha, gl.SliceLayout(1, pv_layout))
    acc = acc * alpha_pv[:, None]
    acc = gl.amd.cdna4.mfma(p_dot, v, acc)
    return m_new, l_new, acc


@gluon.jit
def _process_segment(
    q_dot,
    cache_ptr,
    cache_bf16_ptr,
    indices_ptr,
    seg_start,
    lo,
    hi,
    cs0,
    num_rows,
    m_i,
    l_i,
    acc,
    head_mask,
    qk_scale,
    kv_smem,
    qk_layout: gl.constexpr,
    pv_layout: gl.constexpr,
    k_layout: gl.constexpr,
    v_layout: gl.constexpr,
    p_layout: gl.constexpr,
    gather_l: gl.constexpr,
    gather_rope_l: gl.constexpr,
    slot_l: gl.constexpr,
    IS_FP8: gl.constexpr,
    BLOCK_SIZE: gl.constexpr,
    NOPE_DIM: gl.constexpr,
    ROPE_DIM: gl.constexpr,
    HEAD_SIZE: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_K: gl.constexpr,
    HEAD_ALIGNED: gl.constexpr,
    UNIFORM: gl.constexpr,
    USE_BUFFER_LOAD: gl.constexpr,
    HAS_INVALID: gl.constexpr,
    FP8_FNUZ: gl.constexpr,
):
    offs_full = gl.arange(0, HEAD_SIZE, layout=gl.SliceLayout(0, gather_l))
    offs_rope = gl.arange(0, ROPE_DIM, layout=gl.SliceLayout(0, gather_rope_l))
    k_rng_slot = gl.arange(0, BLOCK_K, layout=slot_l)

    # Peel the (possibly partial) last tile: [lo, hi_full) are full BLOCK_K tiles
    # whose slots are all valid -> mask-free. Only the peeled tail carries masking.
    hi_full = lo + ((hi - lo) // BLOCK_K) * BLOCK_K

    if IS_FP8:
        # Prologue/epilogue software pipeline: gather + DEQUANT tile i into bf16
        # registers while the previous tile's MFMAs run (VALU+matrix-core co-issue
        # hides the quant cost). One LDS buffer, so the LDS write of tile i still
        # waits on tile i-1's reads, but the gather+dequant is off the critical path.
        n_full = (hi_full - lo) // BLOCK_K
        if n_full > 0:
            kn, kr, vld = _gd_fp8(
                cache_ptr,
                cache_bf16_ptr,
                indices_ptr,
                seg_start,
                lo,
                cs0,
                offs_full,
                offs_rope,
                k_rng_slot,
                gather_l,
                gather_rope_l,
                BLOCK_SIZE,
                HEAD_SIZE,
                UNIFORM,
                USE_BUFFER_LOAD,
                HAS_INVALID,
                FP8_FNUZ,
            )
            for i in range(1, n_full):
                kn2, kr2, vld2 = _gd_fp8(
                    cache_ptr,
                    cache_bf16_ptr,
                    indices_ptr,
                    seg_start,
                    lo + i * BLOCK_K,
                    cs0,
                    offs_full,
                    offs_rope,
                    k_rng_slot,
                    gather_l,
                    gather_rope_l,
                    BLOCK_SIZE,
                    HEAD_SIZE,
                    UNIFORM,
                    USE_BUFFER_LOAD,
                    HAS_INVALID,
                    FP8_FNUZ,
                )
                m_i, l_i, acc = _qkpv_fp8(
                    kn,
                    kr,
                    vld,
                    q_dot,
                    m_i,
                    l_i,
                    acc,
                    head_mask,
                    qk_scale,
                    kv_smem,
                    qk_layout,
                    pv_layout,
                    k_layout,
                    v_layout,
                    p_layout,
                    NOPE_DIM,
                    ROPE_DIM,
                    HEAD_SIZE,
                    BLOCK_M,
                    BLOCK_K,
                    HEAD_ALIGNED,
                    UNIFORM,
                    HAS_INVALID,
                )
                kn, kr, vld = kn2, kr2, vld2
            m_i, l_i, acc = _qkpv_fp8(
                kn,
                kr,
                vld,
                q_dot,
                m_i,
                l_i,
                acc,
                head_mask,
                qk_scale,
                kv_smem,
                qk_layout,
                pv_layout,
                k_layout,
                v_layout,
                p_layout,
                NOPE_DIM,
                ROPE_DIM,
                HEAD_SIZE,
                BLOCK_M,
                BLOCK_K,
                HEAD_ALIGNED,
                UNIFORM,
                HAS_INVALID,
            )
    else:
        for k_start in range(lo, hi_full, BLOCK_K):
            m_i, l_i, acc = _decode_tile(
                q_dot,
                cache_ptr,
                cache_bf16_ptr,
                indices_ptr,
                seg_start,
                k_start,
                hi,
                cs0,
                num_rows,
                m_i,
                l_i,
                acc,
                head_mask,
                qk_scale,
                kv_smem,
                offs_full,
                offs_rope,
                k_rng_slot,
                qk_layout,
                pv_layout,
                k_layout,
                v_layout,
                p_layout,
                gather_l,
                gather_rope_l,
                IS_FP8,
                BLOCK_SIZE,
                NOPE_DIM,
                ROPE_DIM,
                HEAD_SIZE,
                BLOCK_M,
                BLOCK_K,
                HEAD_ALIGNED,
                False,
                UNIFORM,
                USE_BUFFER_LOAD,
                HAS_INVALID,
                FP8_FNUZ,
            )

    if hi_full < hi:
        m_i, l_i, acc = _decode_tile(
            q_dot,
            cache_ptr,
            cache_bf16_ptr,
            indices_ptr,
            seg_start,
            hi_full,
            hi,
            cs0,
            num_rows,
            m_i,
            l_i,
            acc,
            head_mask,
            qk_scale,
            kv_smem,
            offs_full,
            offs_rope,
            k_rng_slot,
            qk_layout,
            pv_layout,
            k_layout,
            v_layout,
            p_layout,
            gather_l,
            gather_rope_l,
            IS_FP8,
            BLOCK_SIZE,
            NOPE_DIM,
            ROPE_DIM,
            HEAD_SIZE,
            BLOCK_M,
            BLOCK_K,
            HEAD_ALIGNED,
            True,
            UNIFORM,
            USE_BUFFER_LOAD,
            HAS_INVALID,
            FP8_FNUZ,
        )
    return m_i, l_i, acc


_pa_decode_sparse_repr = make_kernel_repr(
    "_pa_decode_sparse",
    ["BLOCK_M", "BLOCK_K", "HEAD_SIZE", "NUM_SPLITS", "UNIFORM", "MAIN_IS_FP8"],
)


@gluon.jit(repr=_pa_decode_sparse_repr)
def _pa_decode_sparse(
    q_ptr,
    main_cache_ptr,
    main_cache_bf16_ptr,
    main_indices_ptr,
    main_indptr_ptr,
    extra_cache_ptr,
    extra_cache_bf16_ptr,
    extra_indices_ptr,
    extra_indptr_ptr,
    attn_sink_ptr,
    out_ptr,
    part_m_ptr,
    part_l_ptr,
    part_acc_ptr,
    scale: gl.constexpr,
    q_stride0: gl.constexpr,
    q_stride1: gl.constexpr,
    out_stride0: gl.constexpr,
    out_stride1: gl.constexpr,
    main_cs0,
    extra_cs0,
    main_num_rows,
    extra_num_rows,
    pm_stride0: gl.constexpr,
    pm_stride_s: gl.constexpr,
    pa_stride0: gl.constexpr,
    pa_stride_s: gl.constexpr,
    pa_stride_h: gl.constexpr,
    num_heads: gl.constexpr,
    HAS_EXTRA: gl.constexpr,
    HAS_SINK: gl.constexpr,
    MAIN_IS_FP8: gl.constexpr,
    EXTRA_IS_FP8: gl.constexpr,
    MAIN_BLOCK_SIZE: gl.constexpr,
    EXTRA_BLOCK_SIZE: gl.constexpr,
    NOPE_DIM: gl.constexpr,
    ROPE_DIM: gl.constexpr,
    HEAD_SIZE: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_K: gl.constexpr,
    NUM_SPLITS: gl.constexpr,
    HEAD_ALIGNED: gl.constexpr,
    MFMA_K: gl.constexpr,
    UNIFORM: gl.constexpr,
    USE_BUFFER_LOAD: gl.constexpr,
    HAS_INVALID: gl.constexpr,
    FP8_FNUZ: gl.constexpr,
):
    """One program = (query t, split, head-block). Two-loop: main (SWA) then
    extra (top-k). NUM_SPLITS==1 writes the output directly; NUM_SPLITS>1 stores
    un-normalized partials for the reduce kernel. HAS_INVALID gates -1-sentinel
    handling (clamp + score mask) on the full-tile fast paths."""
    NUM_WARPS: gl.constexpr = gl.num_warps()
    query_idx = gl.program_id(0)
    split_id = gl.program_id(1)
    pid_h = gl.program_id(2)

    # ---- layouts ----
    qk_layout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=[16, 16, MFMA_K],
        transposed=True,
        warps_per_cta=[1, NUM_WARPS],
    )
    pv_layout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=[16, 16, MFMA_K],
        transposed=True,
        warps_per_cta=[1, NUM_WARPS],
    )
    KW: gl.constexpr = MFMA_K // 2
    q_layout: gl.constexpr = gl.DotOperandLayout(0, qk_layout, KW)
    k_layout: gl.constexpr = gl.DotOperandLayout(1, qk_layout, KW)
    p_layout: gl.constexpr = gl.DotOperandLayout(0, pv_layout, KW)
    v_layout: gl.constexpr = gl.DotOperandLayout(1, pv_layout, KW)

    # 16 uint8 = 128-bit fp8 gather loads
    GSPT: gl.constexpr = 16
    gather_l: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, GSPT],
        threads_per_warp=[8, 8],
        warps_per_cta=[1, NUM_WARPS],
        order=[1, 0],
    )
    gather_rope_l: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 8],
        threads_per_warp=[8, 8],
        warps_per_cta=[1, NUM_WARPS],
        order=[1, 0],
    )
    slot_l: gl.constexpr = gl.SliceLayout(1, gather_l)
    blocked_q: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 8],
        threads_per_warp=[8, 8],
        warps_per_cta=[1, NUM_WARPS],
        order=[1, 0],
    )
    kv_shared: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[HEAD_SIZE, 8]], [BLOCK_K, HEAD_SIZE], [1, 0]
    )

    h_off = pid_h * BLOCK_M

    # ---- load Q [BLOCK_M, HEAD_SIZE] ----
    offs_m_q = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, blocked_q))
    offs_d_q = gl.arange(0, HEAD_SIZE, layout=gl.SliceLayout(0, blocked_q))
    h_q = h_off + offs_m_q
    h_mask_q = h_q < num_heads
    q_off = (query_idx * q_stride0 + h_q[:, None] * q_stride1 + offs_d_q[None, :]).to(
        gl.int32
    )
    q = gl.amd.cdna4.buffer_load(
        ptr=q_ptr, offsets=q_off, mask=h_mask_q[:, None], other=0.0
    )
    q_dot = gl.convert_layout(q, q_layout)

    # head mask in pv-slice layout (for output / partial masking)
    offs_m_pv = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, pv_layout))
    h_pv = h_off + offs_m_pv
    head_mask_pv = h_pv < num_heads

    # ---- online-softmax state ----
    m_i = gl.full(
        [BLOCK_M], float("-inf"), gl.float32, layout=gl.SliceLayout(1, qk_layout)
    )
    l_i = gl.zeros([BLOCK_M], gl.float32, layout=gl.SliceLayout(1, qk_layout))
    acc = gl.zeros([BLOCK_M, HEAD_SIZE], gl.float32, layout=pv_layout)

    kv_smem = gl.allocate_shared_memory(gl.bfloat16, [BLOCK_K, HEAD_SIZE], kv_shared)

    # exp2 softmax: fold scale*log2(e) into the loop exponent; keep raw `scale`
    # for the sink/normalization (sink is a scaled-score-space logit).
    RCP_LN2: gl.constexpr = 1.4426950408889634
    qk_scale = scale * RCP_LN2

    # ---- main (SWA) segment ----
    main_start = gl.load(main_indptr_ptr + query_idx)
    main_end = gl.load(main_indptr_ptr + query_idx + 1)
    main_len = main_end - main_start
    main_chunk = (main_len + NUM_SPLITS - 1) // NUM_SPLITS
    main_lo = split_id * main_chunk
    main_hi = gl.minimum(main_lo + main_chunk, main_len)
    m_i, l_i, acc = _process_segment(
        q_dot,
        main_cache_ptr,
        main_cache_bf16_ptr,
        main_indices_ptr,
        main_start,
        main_lo,
        main_hi,
        main_cs0,
        main_num_rows,
        m_i,
        l_i,
        acc,
        head_mask_pv,
        qk_scale,
        kv_smem,
        qk_layout,
        pv_layout,
        k_layout,
        v_layout,
        p_layout,
        gather_l,
        gather_rope_l,
        slot_l,
        MAIN_IS_FP8,
        MAIN_BLOCK_SIZE,
        NOPE_DIM,
        ROPE_DIM,
        HEAD_SIZE,
        BLOCK_M,
        BLOCK_K,
        HEAD_ALIGNED,
        UNIFORM,
        USE_BUFFER_LOAD,
        HAS_INVALID,
        FP8_FNUZ,
    )

    if HAS_EXTRA:
        extra_start = gl.load(extra_indptr_ptr + query_idx)
        extra_end = gl.load(extra_indptr_ptr + query_idx + 1)
        extra_len = extra_end - extra_start
        extra_chunk = (extra_len + NUM_SPLITS - 1) // NUM_SPLITS
        extra_lo = split_id * extra_chunk
        extra_hi = gl.minimum(extra_lo + extra_chunk, extra_len)
        m_i, l_i, acc = _process_segment(
            q_dot,
            extra_cache_ptr,
            extra_cache_bf16_ptr,
            extra_indices_ptr,
            extra_start,
            extra_lo,
            extra_hi,
            extra_cs0,
            extra_num_rows,
            m_i,
            l_i,
            acc,
            head_mask_pv,
            qk_scale,
            kv_smem,
            qk_layout,
            pv_layout,
            k_layout,
            v_layout,
            p_layout,
            gather_l,
            gather_rope_l,
            slot_l,
            EXTRA_IS_FP8,
            EXTRA_BLOCK_SIZE,
            NOPE_DIM,
            ROPE_DIM,
            HEAD_SIZE,
            BLOCK_M,
            BLOCK_K,
            HEAD_ALIGNED,
            UNIFORM,
            USE_BUFFER_LOAD,
            HAS_INVALID,
            FP8_FNUZ,
        )

    # m_i/l_i are in SliceLayout(1, qk_layout); acc in pv_layout. Move the row
    # reductions into pv-slice space for output/partials.
    m_pv = gl.convert_layout(m_i, gl.SliceLayout(1, pv_layout))
    l_pv = gl.convert_layout(l_i, gl.SliceLayout(1, pv_layout))

    if NUM_SPLITS == 1:
        if HAS_SINK:
            # m_pv is the RAW row-max; the sink is a scaled-score logit. Combine
            # in scaled space (Ms = scale*m) using exp2.
            Ms = m_pv * scale
            sink = gl.amd.cdna4.buffer_load(
                ptr=attn_sink_ptr, offsets=h_pv, mask=head_mask_pv, other=float("-inf")
            ).to(gl.float32)
            m_final = gl.maximum(Ms, sink)
            alpha = gl.exp2((Ms - m_final) * RCP_LN2)
            l_final = l_pv * alpha + gl.exp2((sink - m_final) * RCP_LN2)
            acc = acc * alpha[:, None]
        else:
            l_final = l_pv
        out = acc / l_final[:, None]
        # store output
        offs_d_o = gl.arange(0, HEAD_SIZE, layout=gl.SliceLayout(0, pv_layout))
        o_off = (
            query_idx * out_stride0 + h_pv[:, None] * out_stride1 + offs_d_o[None, :]
        ).to(gl.int32)
        gl.amd.cdna4.buffer_store(
            out.to(out_ptr.dtype.element_ty),
            ptr=out_ptr,
            offsets=o_off,
            mask=head_mask_pv[:, None],
        )
    else:
        # store un-normalized partials for the reduce kernel. m is stored in the
        # base-2 exponent domain (row-max * softmax_scale * log2e) so the reduce/
        # skip_reduce partials match the triton convention.
        pm_base = query_idx * pm_stride0 + split_id * pm_stride_s
        gl.amd.cdna4.buffer_store(
            m_pv * (scale * RCP_LN2),
            ptr=part_m_ptr + pm_base,
            offsets=h_pv.to(gl.int32),
            mask=head_mask_pv,
        )
        gl.amd.cdna4.buffer_store(
            l_pv, ptr=part_l_ptr + pm_base, offsets=h_pv.to(gl.int32), mask=head_mask_pv
        )
        offs_d_a = gl.arange(0, HEAD_SIZE, layout=gl.SliceLayout(0, pv_layout))
        a_base = query_idx * pa_stride0 + split_id * pa_stride_s
        a_off = (a_base + h_pv[:, None] * pa_stride_h + offs_d_a[None, :]).to(gl.int32)
        gl.amd.cdna4.buffer_store(
            acc, ptr=part_acc_ptr, offsets=a_off, mask=head_mask_pv[:, None]
        )


_pa_decode_sparse_reduce_repr = make_kernel_repr(
    "_pa_decode_sparse_reduce",
    ["BLOCK_M", "HEAD_SIZE", "NUM_SPLITS"],
)


@gluon.jit(repr=_pa_decode_sparse_reduce_repr)
def _pa_decode_sparse_reduce(
    part_m_ptr,
    part_l_ptr,
    part_acc_ptr,
    attn_sink_ptr,
    out_ptr,
    out_stride0: gl.constexpr,
    out_stride1: gl.constexpr,
    pm_stride0: gl.constexpr,
    pm_stride_s: gl.constexpr,
    pa_stride0: gl.constexpr,
    pa_stride_s: gl.constexpr,
    pa_stride_h: gl.constexpr,
    num_heads: gl.constexpr,
    HAS_SINK: gl.constexpr,
    HEAD_SIZE: gl.constexpr,
    BLOCK_M: gl.constexpr,
    NUM_SPLITS: gl.constexpr,
    HEAD_ALIGNED: gl.constexpr,
):
    """Split-KV combine: merge the per-split partials, fold the attn sink, and write
    the final output. Partials store m in the base-2 exponent domain (row-max *
    softmax_scale * log2e), matching the triton reduce. Grid: (num_queries, heads_blocks).
    """
    NUM_WARPS: gl.constexpr = gl.num_warps()
    RCP_LN2: gl.constexpr = 1.4426950408889634
    query_idx = gl.program_id(0)
    pid_h = gl.program_id(1)

    BLK: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 8],
        threads_per_warp=[8, 8],
        warps_per_cta=[1, NUM_WARPS],
        order=[1, 0],
    )
    row_l: gl.constexpr = gl.SliceLayout(1, BLK)  # [BLOCK_M]

    h_off = pid_h * BLOCK_M
    offs_m = gl.arange(0, BLOCK_M, layout=row_l)
    h = h_off + offs_m
    head_mask = h < num_heads
    offs_d = gl.arange(0, HEAD_SIZE, layout=gl.SliceLayout(0, BLK))

    neg_inf = float("-inf")
    m_final = gl.full([BLOCK_M], neg_inf, gl.float32, layout=row_l)
    # pass 1: global max over splits
    for s in range(0, NUM_SPLITS):
        base = query_idx * pm_stride0 + s * pm_stride_s
        m_s = gl.amd.cdna4.buffer_load(
            ptr=part_m_ptr + base, offsets=h, mask=head_mask, other=neg_inf
        )
        m_final = gl.maximum(m_final, m_s)  # m_s already in base-2 exponent domain
    if HAS_SINK:
        sink = gl.amd.cdna4.buffer_load(
            ptr=attn_sink_ptr, offsets=h, mask=head_mask, other=neg_inf
        ).to(gl.float32)
        m_final = gl.maximum(m_final, sink * RCP_LN2)  # lift sink to base-2

    # pass 2: weighted sums
    l_final = gl.zeros([BLOCK_M], gl.float32, layout=row_l)
    acc = gl.zeros([BLOCK_M, HEAD_SIZE], gl.float32, layout=BLK)
    for s in range(0, NUM_SPLITS):
        base = query_idx * pm_stride0 + s * pm_stride_s
        m_s = gl.amd.cdna4.buffer_load(
            ptr=part_m_ptr + base, offsets=h, mask=head_mask, other=neg_inf
        )
        l_s = gl.amd.cdna4.buffer_load(
            ptr=part_l_ptr + base, offsets=h, mask=head_mask, other=0.0
        )
        w = gl.exp2(m_s - m_final)
        l_final = l_final + w * l_s
        a_base = query_idx * pa_stride0 + s * pa_stride_s
        a_off = (a_base + h[:, None] * pa_stride_h + offs_d[None, :]).to(gl.int32)
        acc_s = gl.amd.cdna4.buffer_load(
            ptr=part_acc_ptr, offsets=a_off, mask=head_mask[:, None], other=0.0
        )
        acc = acc + w[:, None] * acc_s

    if HAS_SINK:
        sink = gl.amd.cdna4.buffer_load(
            ptr=attn_sink_ptr, offsets=h, mask=head_mask, other=neg_inf
        ).to(gl.float32)
        l_final = l_final + gl.exp2(sink * RCP_LN2 - m_final)

    out = acc / l_final[:, None]
    o_off = (query_idx * out_stride0 + h[:, None] * out_stride1 + offs_d[None, :]).to(
        gl.int32
    )
    gl.amd.cdna4.buffer_store(
        out.to(out_ptr.dtype.element_ty),
        ptr=out_ptr,
        offsets=o_off,
        mask=head_mask[:, None],
    )
