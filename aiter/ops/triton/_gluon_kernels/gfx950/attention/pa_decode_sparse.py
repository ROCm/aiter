# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Gluon (gfx950) DeepSeek-V4 sparse-MLA decode. Adapted from the vLLM DSv4 sparse
attention kernels:
https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/ops/rocm_aiter_mla_sparse.py

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
from triton.language.core import PropagateNan

from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr

# Triton's default max ignores NaN, which on AMD costs a v_max_f32 x, x, x
# canonicalize per operand before the real compare -- 60 of the 96 v_max in this
# kernel were those no-ops. Nothing here produces NaN (masked lanes are -inf and
# the all-masked row is guarded explicitly), so propagate instead.
_MAX_PROP_NAN: gl.constexpr = gl.constexpr(PropagateNan.ALL)


@gluon.jit
def _max2(a, b):
    return gl.maximum(a, b, propagate_nan=_MAX_PROP_NAN)


@gluon.jit
def _rmax(x, axis):
    return gl.reduce(x, axis, _max2)



@gluon.jit
def _cache_load(ptr, row, col, USE_BUFFER_LOAD: gl.constexpr, mask=None, other=None):
    """Gather rows[i] + col[j] out of a cache.

    ``row`` is the per-token byte offset ([BLOCK_K]); ``col`` is the in-row offset
    ([W]), which is always a small compile-time arange. Keeping them apart is what
    makes the >2 GB path affordable: buffer_load carries a 32-bit offset (2 GB cap),
    so a bigger cache has to gather through 64-bit addresses -- and adding the
    pointer to a fully materialized [BLOCK_K, W] offset tensor makes that a 64-bit
    add *per element*. Resolving one pointer per token instead costs BLOCK_K of
    them, and the leftover column offset is a constant the load can fold into its
    immediate field. (Same shape as the in-tree triton kernel's token_data_ptr.)
    """
    if USE_BUFFER_LOAD:
        return gl.amd.cdna4.buffer_load(
            ptr=ptr,
            offsets=row.to(gl.int32)[:, None] + col.to(gl.int32)[None, :],
            mask=mask,
            other=other,
            cache=".cg",
        )
    row_ptr = ptr + row.to(gl.int64)
    return gl.load(
        row_ptr[:, None] + col[None, :],
        mask=mask,
        other=other,
        cache_modifier=".cg",
    )


@gluon.jit
def _fp8_to_f32(x_u8, FP8_FNUZ: gl.constexpr):
    # gfx950's native e4m3 cvt is OCP (float8e4nv). fnuz (float8e4b8) -> f32 has no
    # native cvt and lowers to a ~5x software unpack that spills; but fnuz -> bf16 is
    # cheap and fp8 -> bf16 is exact (3 mantissa bits), so route fnuz through bf16.
    if FP8_FNUZ:
        return x_u8.to(gl.float8e4b8, bitcast=True).to(gl.bfloat16).to(gl.float32)
    return x_u8.to(gl.float8e4nv, bitcast=True).to(gl.float32)


@gluon.jit
def _scale_load(
    ptr,
    row,
    valid,
    USE_BUFFER_LOAD: gl.constexpr,
    gather_l: gl.constexpr,
    scl_l: gl.constexpr,
    NG: gl.constexpr,
    HEAD_SIZE: gl.constexpr,
    MASKED: gl.constexpr,
):
    """Gather the NG per-64-group scale bytes of each token and broadcast them out
    to HEAD_SIZE in registers.

    The obvious spelling -- index the full row with ``offs_full // 64`` -- builds a
    [BLOCK_K, HEAD_SIZE] pointer tensor for a value that has only NG distinct
    entries per row. On the buffer_load path the identical 32-bit offsets CSE away
    and it costs 48 byte loads; on the 64-bit path they do not, and it costs 288.
    (The in-tree triton kernel loads tensor<32x2x!tt.ptr<i8>> here, we were loading
    tensor<64x512x!tt.ptr<i8>>.) Gather NG wide instead and broadcast: ``scl_l`` is
    the dim-2 slice of a 3-D layout picked so that reshaping [BLOCK_K, NG, 64] back
    to [BLOCK_K, HEAD_SIZE] lands exactly on ``gather_l`` -- column c = g*64 + j
    maps to thread 4g + j//16 -- so the broadcast is a register rename.
    """
    cols = gl.arange(0, NG, layout=gl.SliceLayout(0, scl_l))
    rows = gl.convert_layout(row, gl.SliceLayout(1, scl_l))
    if MASKED:
        m = gl.convert_layout(valid, gl.SliceLayout(1, scl_l))[:, None]
        if USE_BUFFER_LOAD:
            sc = gl.amd.cdna4.buffer_load(
                ptr=ptr,
                offsets=rows.to(gl.int32)[:, None] + cols[None, :],
                mask=m,
                other=127,
                cache=".cg",
            )
        else:
            sc = gl.load(
                (ptr + rows.to(gl.int64))[:, None] + cols[None, :],
                mask=m,
                other=127,
                cache_modifier=".cg",
            )
    else:
        if USE_BUFFER_LOAD:
            sc = gl.amd.cdna4.buffer_load(
                ptr=ptr,
                offsets=rows.to(gl.int32)[:, None] + cols[None, :],
                cache=".cg",
            )
        else:
            sc = gl.load(
                (ptr + rows.to(gl.int64))[:, None] + cols[None, :],
                cache_modifier=".cg",
            )
    wide = gl.expand_dims(sc, 2).broadcast_to([sc.shape[0], NG, HEAD_SIZE // NG])
    return gl.convert_layout(
        wide.reshape([sc.shape[0], HEAD_SIZE]), gather_l, assert_trivial=True
    )


@gluon.jit
def _split2(x):
    """Contiguous register split along dim 1: [A, B] -> two [A, B//2].

    x0 takes columns [0, B//2), x1 takes [B//2, B). Both halves come back in the
    input's own layout, so with warps tiling dim 0 (see ``gather_l``) the column
    direction is a pure per-lane register repeat and the split is a rename -- no
    cross-lane traffic. ``assert_trivial`` makes a non-free split a compile error
    rather than a silent LDS round-trip.
    """
    layout: gl.constexpr = x.type.layout
    x_r = x.reshape(x.shape[0], 2, x.shape[1] // 2).permute(0, 2, 1)
    x0, x1 = gl.split(x_r)
    x0 = gl.convert_layout(x0, layout, assert_trivial=True)
    x1 = gl.convert_layout(x1, layout, assert_trivial=True)
    return x0, x1


@gluon.jit
def _split2_dim0(x):
    """Contiguous register split along dim 0: [A, B] -> two [A//2, B].

    The dim-0 counterpart of _split2, for the row-wide gather layout where dim 1
    is spent entirely on threads (one instruction = whole token rows) and the
    per-lane register repeats live on dim 0 instead.
    """
    layout: gl.constexpr = x.type.layout
    x_r = x.reshape(2, x.shape[0] // 2, x.shape[1]).permute(1, 2, 0)
    x0, x1 = gl.split(x_r)
    x0 = gl.convert_layout(x0, layout, assert_trivial=True)
    x1 = gl.convert_layout(x1, layout, assert_trivial=True)
    return x0, x1


@gluon.jit
def _split_ax(x, AXIS: gl.constexpr):
    if AXIS == 1:
        a, b = _split2(x)
    else:
        a, b = _split2_dim0(x)
    return a, b


@gluon.jit
def _deq_store(
    x_u8, sc, kv_smem, off,
    AXIS: gl.constexpr, UNIFORM: gl.constexpr, FP8_FNUZ: gl.constexpr,
):
    """Dequant one fp8 slab and write it straight to kv_smem[:, off:off+W].

    Keep the dequant in f32: gfx950 has no bf16 multiply, so a bf16 path lowers to
    an emulated mul and doubles the loop. ``sc`` is the raw per-64-group exponent
    byte (packed fp8_ds_mla, UE8M0) or an f32 scale (uniform pool).
    """
    if UNIFORM:
        scale = sc
    else:
        scale = gl.exp2(sc.to(gl.float32) - 127.0)
    val = (_fp8_to_f32(x_u8, FP8_FNUZ) * scale).to(gl.bfloat16)
    if AXIS == 1:
        kv_smem.slice(off, x_u8.shape[1], dim=1).store(val)
    else:
        kv_smem.slice(off, x_u8.shape[0], dim=0).store(val)


@gluon.jit
def _deq_store_tile(
    x_u8,
    sc,
    kv_smem,
    NOPE_CHUNK: gl.constexpr,
    AXIS: gl.constexpr,
    UNIFORM: gl.constexpr,
    FP8_FNUZ: gl.constexpr,
):
    """Dequant a whole gathered fp8 tile into kv_smem in NOPE_CHUNK-sized pieces
    along CHUNK_AXIS (0 = rows, 1 = columns).

    The f32 expansion is 4x the fp8 tile (a [64, 512] tile is 128 f32 VGPRs/lane),
    so materializing it in one shot is what pins the kernel at 1 wave/SIMD.
    Splitting first makes the dependence chain explicit -- piece c's converts feed
    piece c's ds_writes and die -- so only a 1/pieces fraction of it is ever
    live. The splits themselves are register renames (see ``_split2``).
    """
    W: gl.constexpr = x_u8.shape[1] if AXIS == 1 else x_u8.shape[0]
    if NOPE_CHUNK >= W:
        _deq_store(x_u8, sc, kv_smem, 0, AXIS, UNIFORM, FP8_FNUZ)
    else:
        x0, x1 = _split_ax(x_u8, AXIS)
        s0, s1 = _split_ax(sc, AXIS)
        W2: gl.constexpr = W // 2
        if NOPE_CHUNK >= W2:
            _deq_store(x0, s0, kv_smem, 0, AXIS, UNIFORM, FP8_FNUZ)
            _deq_store(x1, s1, kv_smem, W2, AXIS, UNIFORM, FP8_FNUZ)
        else:
            x00, x01 = _split_ax(x0, AXIS)
            s00, s01 = _split_ax(s0, AXIS)
            x10, x11 = _split_ax(x1, AXIS)
            s10, s11 = _split_ax(s1, AXIS)
            W4: gl.constexpr = W // 4
            if NOPE_CHUNK >= W4:
                _deq_store(x00, s00, kv_smem, 0, AXIS, UNIFORM, FP8_FNUZ)
                _deq_store(x01, s01, kv_smem, W4, AXIS, UNIFORM, FP8_FNUZ)
                _deq_store(x10, s10, kv_smem, 2 * W4, AXIS, UNIFORM, FP8_FNUZ)
                _deq_store(x11, s11, kv_smem, 3 * W4, AXIS, UNIFORM, FP8_FNUZ)
            else:
                W8: gl.constexpr = W // 8
                y0, y1 = _split_ax(x00, AXIS)
                t0, t1 = _split_ax(s00, AXIS)
                _deq_store(y0, t0, kv_smem, 0, AXIS, UNIFORM, FP8_FNUZ)
                _deq_store(y1, t1, kv_smem, W8, AXIS, UNIFORM, FP8_FNUZ)
                y0, y1 = _split_ax(x01, AXIS)
                t0, t1 = _split_ax(s01, AXIS)
                _deq_store(y0, t0, kv_smem, 2 * W8, AXIS, UNIFORM, FP8_FNUZ)
                _deq_store(y1, t1, kv_smem, 3 * W8, AXIS, UNIFORM, FP8_FNUZ)
                y0, y1 = _split_ax(x10, AXIS)
                t0, t1 = _split_ax(s10, AXIS)
                _deq_store(y0, t0, kv_smem, 4 * W8, AXIS, UNIFORM, FP8_FNUZ)
                _deq_store(y1, t1, kv_smem, 5 * W8, AXIS, UNIFORM, FP8_FNUZ)
                y0, y1 = _split_ax(x11, AXIS)
                t0, t1 = _split_ax(s11, AXIS)
                _deq_store(y0, t0, kv_smem, 6 * W8, AXIS, UNIFORM, FP8_FNUZ)
                _deq_store(y1, t1, kv_smem, 7 * W8, AXIS, UNIFORM, FP8_FNUZ)


@gluon.jit
def _slots(
    indices_ptr,
    seg_start,
    k_pos,
    hi,
    num_rows,
    BLOCK_SIZE: gl.constexpr,
    MASKED: gl.constexpr,
    HAS_INVALID: gl.constexpr,
):
    # Returns in whatever layout k_pos carries. Called once per gather layout: NoPE
    # and RoPE tile their warps differently, and re-loading this tiny broadcast
    # vector is cheaper than a cross-lane convert between the two.
    if MASKED:
        in_range = k_pos < hi
        slot = gl.load(indices_ptr + seg_start + k_pos, mask=in_range, other=-1)
        valid = in_range & (slot >= 0) & (slot < num_rows)
        slot = gl.where(valid, slot, 0)
    else:
        slot = gl.load(indices_ptr + seg_start + k_pos)
        valid = slot >= 0  # -1 sentinels: clamp in-bounds, mask score below
        if HAS_INVALID:
            slot = gl.where(valid, slot, 0)
    return (slot // BLOCK_SIZE).to(gl.int32), (slot % BLOCK_SIZE).to(gl.int32), valid


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
    k_rng_rope,
    qk_layout: gl.constexpr,
    pv_layout: gl.constexpr,
    k_layout: gl.constexpr,
    v_layout: gl.constexpr,
    p_layout: gl.constexpr,
    gather_l: gl.constexpr,
    gather_rope_l: gl.constexpr,
    scl_l: gl.constexpr,
    NARROW_SCALE: gl.constexpr,
    IS_FP8: gl.constexpr,
    BLOCK_SIZE: gl.constexpr,
    NOPE_DIM: gl.constexpr,
    ROPE_DIM: gl.constexpr,
    HEAD_SIZE: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_K: gl.constexpr,
    NOPE_CHUNK: gl.constexpr,
    CHUNK_AXIS: gl.constexpr,
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
    block_idx, pos, valid1d = _slots(
        indices_ptr,
        seg_start,
        k_start + k_rng_slot,
        hi,
        num_rows,
        BLOCK_SIZE,
        MASKED,
        HAS_INVALID,
    )
    block_idx_g = gl.convert_layout(block_idx, gl.SliceLayout(1, gather_l))
    pos_g = gl.convert_layout(pos, gl.SliceLayout(1, gather_l))
    if MASKED:
        valid_g = gl.convert_layout(valid1d, gl.SliceLayout(1, gather_l))

    if IS_FP8 and UNIFORM:
        # uniform pool: one fp8 gather over the whole head + separate fp32 scales.
        NGRP: gl.constexpr = HEAD_SIZE // 64
        kv_row = block_idx_g * cs0 + pos_g * HEAD_SIZE
        scl_row = block_idx_g * NGRP
        scl_col = offs_full // 64
        if MASKED:
            x_u8 = _cache_load(
                cache_ptr, kv_row, offs_full, USE_BUFFER_LOAD,
                mask=valid_g[:, None], other=0,
            )
            sc = _cache_load(
                cache_bf16_ptr,
                scl_row,
                scl_col,
                USE_BUFFER_LOAD,
                mask=valid_g[:, None],
                other=0.0,
            )  # uniform pool: fp32 scales, left wide (see _scale_load)
        else:
            x_u8 = _cache_load(cache_ptr, kv_row, offs_full, USE_BUFFER_LOAD)
            sc = _cache_load(cache_bf16_ptr, scl_row, scl_col, USE_BUFFER_LOAD)
        _deq_store_tile(x_u8, sc, kv_smem, NOPE_CHUNK, CHUNK_AXIS, True, FP8_FNUZ)
    elif IS_FP8:
        # DSv4 packed fp8_ds_mla: NoPE fp8 + embedded UE8M0 + separate RoPE-bf16.
        nope_row = block_idx_g * cs0 + pos_g * 576
        scl_row = block_idx_g * cs0 + BLOCK_SIZE * 576 + pos_g * 8
        scl_col = offs_full // 64
        if MASKED:  # scales first: see _gd_fp8
            if NARROW_SCALE and not USE_BUFFER_LOAD:
                exps = _scale_load(
                    cache_ptr, scl_row, valid1d, USE_BUFFER_LOAD, gather_l, scl_l,
                    HEAD_SIZE // 64, HEAD_SIZE, True,
                )
            else:
                exps = _cache_load(
                    cache_ptr, scl_row, scl_col, USE_BUFFER_LOAD,
                    mask=valid_g[:, None], other=127,
                )
            x_u8 = _cache_load(
                cache_ptr, nope_row, offs_full, USE_BUFFER_LOAD,
                mask=valid_g[:, None], other=0,
            )
        else:
            if NARROW_SCALE and not USE_BUFFER_LOAD:
                exps = _scale_load(
                    cache_ptr, scl_row, scl_row, USE_BUFFER_LOAD, gather_l, scl_l,
                    HEAD_SIZE // 64, HEAD_SIZE, False,
                )
            else:
                exps = _cache_load(cache_ptr, scl_row, scl_col, USE_BUFFER_LOAD)
            x_u8 = _cache_load(cache_ptr, nope_row, offs_full, USE_BUFFER_LOAD)
        _deq_store_tile(x_u8, exps, kv_smem, NOPE_CHUNK, CHUNK_AXIS, False, FP8_FNUZ)
        block_idx_gr, pos_gr, valid_gr = _slots(
            indices_ptr,
            seg_start,
            k_start + k_rng_rope,
            hi,
            num_rows,
            BLOCK_SIZE,
            MASKED,
            HAS_INVALID,
        )
        rope_row = block_idx_gr * (cs0 // 2) + pos_gr * 288 + 224
        if MASKED:
            k_rope = _cache_load(
                cache_bf16_ptr,
                rope_row,
                offs_rope,
                USE_BUFFER_LOAD,
                mask=valid_gr[:, None],
                other=0.0,
            )
        else:
            k_rope = _cache_load(cache_bf16_ptr, rope_row, offs_rope, USE_BUFFER_LOAD)
        kv_smem.slice(NOPE_DIM, ROPE_DIM, dim=1).store(k_rope)
    else:
        kv_row2 = block_idx_g * cs0 + pos_g * HEAD_SIZE
        if MASKED:
            kv = _cache_load(
                cache_bf16_ptr, kv_row2, offs_full, USE_BUFFER_LOAD,
                mask=valid_g[:, None], other=0.0,
            )
        else:
            kv = _cache_load(cache_bf16_ptr, kv_row2, offs_full, USE_BUFFER_LOAD)
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

    # Online softmax in the base-2 exponent domain: fold qk_scale*log2(e) into S
    # once, right out of the MFMA, and carry m_i already scaled. The alternative
    # (raw running max, scale at use) needs qk_scale on both the S tile and the two
    # m vectors every iteration; here the per-element work in the loop is one
    # subtract feeding exp2, and the whole epilogue -- sink combine and partial
    # store, both of which want base-2 m -- stops converting.
    S = S * qk_scale
    m_block = _rmax(S, 1)
    m_new = _max2(m_i, m_block)
    m_new = gl.where(m_new > neg_inf, m_new, 0.0)  # guard all-masked rows
    p = gl.exp2(S - m_new[:, None])
    alpha = gl.exp2(m_i - m_new)
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
    k_rng_rope,
    gather_l: gl.constexpr,
    gather_rope_l: gl.constexpr,
    scl_l: gl.constexpr,
    NARROW_SCALE: gl.constexpr,
    BLOCK_SIZE: gl.constexpr,
    HEAD_SIZE: gl.constexpr,
    UNIFORM: gl.constexpr,
    USE_BUFFER_LOAD: gl.constexpr,
    HAS_INVALID: gl.constexpr,
    FP8_FNUZ: gl.constexpr,
):
    """Gather one full fp8 tile. Split from the LDS-write/MFMA so the gather issues
    an iteration early.

    The prefetch is carried across the MFMA in **raw fp8**, not dequantized bf16:
    the loop keeps two tiles in flight, and a [BLOCK_K, HEAD_SIZE] tile is 32
    VGPRs/lane as u8 but 64 as bf16, so dequantizing here would burn an extra 64
    VGPRs on loop-carried state alone. The consumer dequants (chunked) instead.
    """
    if not USE_BUFFER_LOAD:
        cs0 = cs0.to(gl.int64)  # >2 GB cache: 64-bit gather offsets (see _cache_load)
    bg, pg, valid = _slots(
        indices_ptr,
        seg_start,
        k_start + k_rng_slot,
        0,  # hi/num_rows: unused at MASKED=False
        0,
        BLOCK_SIZE,
        False,
        HAS_INVALID,
    )
    if UNIFORM:
        NGRP: gl.constexpr = HEAD_SIZE // 64
        x_u8 = _cache_load(
            cache_ptr, bg * cs0 + pg * HEAD_SIZE, offs_full, USE_BUFFER_LOAD
        )
        sc = _cache_load(cache_bf16_ptr, bg * NGRP, offs_full // 64, USE_BUFFER_LOAD)
        k_rope = x_u8  # unused for UNIFORM (rope slice-store skipped) -> DCE'd
    else:
        nope_row = bg * cs0 + pg * 576
        scl_row = bg * cs0 + BLOCK_SIZE * 576 + pg * 8
        # UE8M0 scales first, bulk fp8 after. vmcnt is one in-order FIFO, so a
        # wait can only name "at most N outstanding", never a specific load:
        # issuing the scales last would make the first dequant piece wait behind
        # every data load as well. Issued first, the scales are covered by the
        # wait that the first piece's own data needs anyway.
        if NARROW_SCALE and not USE_BUFFER_LOAD:
            sc = _scale_load(
                cache_ptr, scl_row, scl_row, USE_BUFFER_LOAD, gather_l, scl_l,
                HEAD_SIZE // 64, HEAD_SIZE, False,
            )
        else:
            sc = _cache_load(cache_ptr, scl_row, offs_full // 64, USE_BUFFER_LOAD)
        x_u8 = _cache_load(cache_ptr, nope_row, offs_full, USE_BUFFER_LOAD)
        bgr, pgr, _ = _slots(
            indices_ptr,
            seg_start,
            k_start + k_rng_rope,
            0,
            0,
            BLOCK_SIZE,
            False,
            HAS_INVALID,
        )
        k_rope = _cache_load(
            cache_bf16_ptr, bgr * (cs0 // 2) + pgr * 288 + 224, offs_rope, USE_BUFFER_LOAD
        )
    return x_u8, sc, k_rope, valid


@gluon.jit
def _qkpv_fp8(
    x_u8,
    sc,
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
    NOPE_CHUNK: gl.constexpr,
    CHUNK_AXIS: gl.constexpr,
    HEAD_ALIGNED: gl.constexpr,
    UNIFORM: gl.constexpr,
    FP8_FNUZ: gl.constexpr,
    HAS_INVALID: gl.constexpr,
):
    """Dequant a prefetched fp8 tile into LDS, then QK -> softmax -> PV. UNIFORM
    skips the RoPE slice-store (the whole head is one fp8 tile). When HAS_INVALID,
    mask the columns of -1-sentinel slots (``valid`` from _gd_fp8) to -inf."""
    neg_inf = float("-inf")
    _deq_store_tile(x_u8, sc, kv_smem, NOPE_CHUNK, CHUNK_AXIS, UNIFORM, FP8_FNUZ)
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
    S = S * qk_scale  # see _decode_tile: m_i is carried in base-2 exponent space
    m_block = _rmax(S, 1)
    m_new = _max2(m_i, m_block)
    m_new = gl.where(m_new > neg_inf, m_new, 0.0)
    p = gl.exp2(S - m_new[:, None])
    alpha = gl.exp2(m_i - m_new)
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
    scl_l: gl.constexpr,
    NARROW_SCALE: gl.constexpr,
    slot_l: gl.constexpr,
    IS_FP8: gl.constexpr,
    BLOCK_SIZE: gl.constexpr,
    NOPE_DIM: gl.constexpr,
    ROPE_DIM: gl.constexpr,
    HEAD_SIZE: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_K: gl.constexpr,
    NOPE_CHUNK: gl.constexpr,
    CHUNK_AXIS: gl.constexpr,
    HEAD_ALIGNED: gl.constexpr,
    UNIFORM: gl.constexpr,
    USE_BUFFER_LOAD: gl.constexpr,
    HAS_INVALID: gl.constexpr,
    FP8_FNUZ: gl.constexpr,
):
    offs_full = gl.arange(0, HEAD_SIZE, layout=gl.SliceLayout(0, gather_l))
    offs_rope = gl.arange(0, ROPE_DIM, layout=gl.SliceLayout(0, gather_rope_l))
    k_rng_slot = gl.arange(0, BLOCK_K, layout=slot_l)
    k_rng_rope = gl.arange(0, BLOCK_K, layout=gl.SliceLayout(1, gather_rope_l))

    # Peel the (possibly partial) last tile: [lo, hi_full) are full BLOCK_K tiles
    # whose slots are all valid -> mask-free. Only the peeled tail carries masking.
    hi_full = lo + ((hi - lo) // BLOCK_K) * BLOCK_K

    if IS_FP8:
        n_full = (hi_full - lo) // BLOCK_K
        if n_full > 0:
            kn, ks, kr, vld = _gd_fp8(
                cache_ptr,
                cache_bf16_ptr,
                indices_ptr,
                seg_start,
                lo,
                cs0,
                offs_full,
                offs_rope,
                k_rng_slot,
                k_rng_rope,
                gather_l,
                gather_rope_l,
                scl_l,
                NARROW_SCALE,
                BLOCK_SIZE,
                HEAD_SIZE,
                UNIFORM,
                USE_BUFFER_LOAD,
                HAS_INVALID,
                FP8_FNUZ,
            )
            for i in range(1, n_full):
                kn2, ks2, kr2, vld2 = _gd_fp8(
                    cache_ptr,
                    cache_bf16_ptr,
                    indices_ptr,
                    seg_start,
                    lo + i * BLOCK_K,
                    cs0,
                    offs_full,
                    offs_rope,
                    k_rng_slot,
                    k_rng_rope,
                    gather_l,
                    gather_rope_l,
                    scl_l,
                    NARROW_SCALE,
                    BLOCK_SIZE,
                    HEAD_SIZE,
                    UNIFORM,
                    USE_BUFFER_LOAD,
                    HAS_INVALID,
                    FP8_FNUZ,
                )
                m_i, l_i, acc = _qkpv_fp8(
                    kn,
                    ks,
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
                    NOPE_CHUNK,
                    CHUNK_AXIS,
                    HEAD_ALIGNED,
                    UNIFORM,
                    FP8_FNUZ,
                    HAS_INVALID,
                )
                kn, ks, kr, vld = kn2, ks2, kr2, vld2
            m_i, l_i, acc = _qkpv_fp8(
                kn,
                ks,
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
                NOPE_CHUNK,
                CHUNK_AXIS,
                HEAD_ALIGNED,
                UNIFORM,
                FP8_FNUZ,
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
                k_rng_rope,
                qk_layout,
                pv_layout,
                k_layout,
                v_layout,
                p_layout,
                gather_l,
                gather_rope_l,
                scl_l,
                NARROW_SCALE,
                IS_FP8,
                BLOCK_SIZE,
                NOPE_DIM,
                ROPE_DIM,
                HEAD_SIZE,
                BLOCK_M,
                BLOCK_K,
                NOPE_CHUNK,
                CHUNK_AXIS,
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
            k_rng_rope,
            qk_layout,
            pv_layout,
            k_layout,
            v_layout,
            p_layout,
            gather_l,
            gather_rope_l,
            scl_l,
            NARROW_SCALE,
            IS_FP8,
            BLOCK_SIZE,
            NOPE_DIM,
            ROPE_DIM,
            HEAD_SIZE,
            BLOCK_M,
            BLOCK_K,
            NOPE_CHUNK,
            CHUNK_AXIS,
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
    CS0_ALIGN: gl.constexpr,
    NOPE_DIM: gl.constexpr,
    ROPE_DIM: gl.constexpr,
    HEAD_SIZE: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_K: gl.constexpr,
    NUM_SPLITS: gl.constexpr,
    HEAD_ALIGNED: gl.constexpr,
    MFMA_K: gl.constexpr,
    UNIFORM: gl.constexpr,
    GATHER_TW1: gl.constexpr,
    LDS_PAD: gl.constexpr,
    # NOPE_CHUNK: extent of one dequant piece along CHUNK_AXIS (0 = rows, 1 =
    # columns); >= the tile's extent on that axis means one shot.
    NOPE_CHUNK: gl.constexpr,
    CHUNK_AXIS: gl.constexpr,
    PART_STORE_CACHE: gl.constexpr,
    # How many of the NUM_SPLITS programs share the main (SWA) segment. The two
    # segments have very different shapes -- main is a contiguous window whose
    # length is fixed by the sliding window, extra is the top-k list -- so one
    # split count has to be wrong for one of them. Splitting main past
    # main_len/BLOCK_K turns every full tile into a half-empty masked one, while
    # extra still wants the CTAs. MAIN_SPLITS <= NUM_SPLITS lets main stop at whole
    # tiles and extra keep going; programs with split_id >= MAIN_SPLITS get an
    # empty main range and contribute an extra-only partial.
    MAIN_SPLITS: gl.constexpr,
    # ADAPTIVE_SPLITS: re-decide the useful split count per query at runtime.
    ADAPTIVE_SPLITS: gl.constexpr,
    # Per-cache buffer/global gate. buffer_load carries a 32-bit offset, so a cache
    # whose span exceeds that must gather via 64-bit gl.load -- but the two caches
    # are sized independently (SWA window vs full compressed history), so gating
    # them together would drop the fast path on a small cache just because its
    # partner is large. At tiny top-k the main/SWA gather is ~94% of the tokens.
    MAIN_USE_BUFFER_LOAD: gl.constexpr,
    EXTRA_USE_BUFFER_LOAD: gl.constexpr,
    HAS_INVALID: gl.constexpr,
    FP8_FNUZ: gl.constexpr,
):
    """One program = (query t, split, head-block). Two-loop: main (SWA) then
    extra (top-k). NUM_SPLITS==1 writes the output directly; NUM_SPLITS>1 stores
    un-normalized partials for the reduce kernel. HAS_INVALID gates -1-sentinel
    handling (clamp + score mask) on the full-tile fast paths."""
    NUM_WARPS: gl.constexpr = gl.num_warps()
    # Page stride alignment hint. The row base of every cache gather is
    # ``block_idx*cs0 + pos*576`` (packed) / ``... + pos*HEAD_SIZE`` (uniform) with
    # block_idx/pos gathered at runtime, so the divisibility analysis has to assume
    # 1-byte alignment and a contiguous 512-byte row gather lowers to hundreds of
    # global_load_ubyte instead of global_load_dwordx4. 576 and HEAD_SIZE are
    # literals the compiler can already reason about; cs0 is the one opaque term.
    # The driver sets CS0_ALIGN>1 only after checking the strides on the host.
    if CS0_ALIGN > 1:
        main_cs0 = gl.multiple_of(main_cs0, CS0_ALIGN)
        extra_cs0 = gl.multiple_of(extra_cs0, CS0_ALIGN)
    query_idx = gl.program_id(0)
    split_id = gl.program_id(1)
    pid_h = gl.program_id(2)

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
    # Warps on dim 1 (columns) vs dim 0 (rows). Coalescing is identical either way
    # -- a row's 128 contiguous bytes always come from 8 threads -- but the row
    # index vector lives in SliceLayout(1, gather_l), so warps-on-dim-1 replicates
    # it across all NUM_WARPS and every lane carries BLOCK_K/8 slots, while
    # warps-on-dim-0 carries BLOCK_K/(8*NUM_WARPS). Dim 0 also makes the column
    # direction a pure per-lane register repeat, so a chunked dequant can split it
    # with a trivial (register-renaming) layout convert.
    # GATHER_TW1 = threads spent on the head dim. Each thread loads 16 B, so
    # TW1=8 requests 128 B of a token row per instruction and TW1=32 requests the
    # whole 512 B row -- for a scattered gather that is one request instead of
    # four quarters. The cost is that the row-index vector lives in
    # SliceLayout(1, gather_l), so a wider TW1 leaves fewer dim-0 thread slots and
    # every lane carries more slots (BLOCK_K*TW1/64 of them). Whichever dim keeps
    # per-lane register repeats is the one the chunked dequant can split
    # (CHUNK_AXIS).
    gather_l: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, GSPT],
        threads_per_warp=[64 // GATHER_TW1, GATHER_TW1],
        warps_per_cta=[NUM_WARPS, 1],
        order=[1, 0],
    )
    # Warps tile dim 0 here, one warp already covers all 64 RoPE columns, so putting
    # them on dim 1 (as the NoPE gather does) overshoots 4x and every warp re-gathers
    # the same tile. Worth ~10% on packed fp8.
    gather_rope_l: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 8],
        threads_per_warp=[8, 8],
        warps_per_cta=[NUM_WARPS, 1],
        order=[1, 0],
    )
    # 3-D companion of gather_l used by _scale_load: dim 1 carries the NG scale
    # groups (one thread each) and dim 2 the 64 columns inside a group, so that
    # reshaping [BLOCK_K, NG, 64] back to 2-D reproduces gather_l exactly. Legal
    # only when the group's columns fill whole threads, i.e. GSPT * (TW1 // NG) is
    # the group width; otherwise fall back to the wide (redundant) gather.
    NGRP_S: gl.constexpr = HEAD_SIZE // 64
    NARROW_SCALE: gl.constexpr = (
        GATHER_TW1 % NGRP_S == 0 and GSPT * (GATHER_TW1 // NGRP_S) == 64
    )
    # ...and only worth it on the 64-bit path. With buffer_load the wide gather's
    # identical 32-bit offsets CSE to the same 48 byte loads, so the narrow form
    # only adds a layout convert per tile (+3.4% at extra=1024). Without it they do
    # not CSE and the wide form costs 288 loads.
    scl_l3: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 1, GSPT],
        threads_per_warp=[
            64 // GATHER_TW1,
            NGRP_S if NARROW_SCALE else 1,
            (GATHER_TW1 // NGRP_S) if NARROW_SCALE else GATHER_TW1,
        ],
        warps_per_cta=[NUM_WARPS, 1, 1],
        order=[2, 1, 0],
    )
    scl_l: gl.constexpr = gl.SliceLayout(2, scl_l3)
    slot_l: gl.constexpr = gl.SliceLayout(1, gather_l)
    blocked_q: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 8],
        threads_per_warp=[8, 8],
        warps_per_cta=[1, NUM_WARPS],
        order=[1, 0],
    )
    # LDS pad after every row. Row pitch (512 + LDS_PAD) bf16 sets which banks a
    # transposed K read (ds_read_b64_tr_b16 walks down a column) lands on:
    # bank = (row * pitch_dwords) mod 32, so pitch_dwords mod 32 == 4 at PAD=8 means
    # 32 lanes share 8 banks. Kept as a knob because the conflict share is high
    # (46%) but a high sub-metric ratio is not proof of a bottleneck.
    kv_shared: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[HEAD_SIZE, LDS_PAD]], [BLOCK_K, HEAD_SIZE], [1, 0]
    )

    h_off = pid_h * BLOCK_M

    # ---- segment lengths, issued first ----
    # These gate the whole KV chain: indptr -> segment range -> indices gather ->
    # cache addresses -> cache gather, three dependent memory round trips deep.
    # Q is independent of all of it, so issue the indptr loads before Q rather
    # than after: the scalar readfirstlane they feed needs s_waitcnt vmcnt(0), and
    # behind Q's load plus its through-LDS layout conversion (two barriers) that
    # wait lands at the end of a long serial prologue instead of overlapping it.
    main_start = gl.load(main_indptr_ptr + query_idx)
    main_end = gl.load(main_indptr_ptr + query_idx + 1)
    if HAS_EXTRA:
        extra_start = gl.load(extra_indptr_ptr + query_idx)
        extra_end = gl.load(extra_indptr_ptr + query_idx + 1)
        extra_len = extra_end - extra_start
    else:
        extra_start = 0
        extra_len = 0
    main_len = main_end - main_start

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

    # ---- how many of the launched splits this query actually wants ----
    # NUM_SPLITS / MAIN_SPLITS come from the host, which only knows the batch's
    # AVERAGE segment lengths. In a ragged batch (the SWA window saturates at
    # sliding_window while top-k keeps growing with context) the per-query lengths
    # differ a lot, and a split count sized for the average over-splits the short
    # queries -- each surplus program still gathers a mostly-masked tile and writes
    # a full [BLOCK_M, HEAD_SIZE] f32 partial. Recompute the useful count from this
    # query's own lengths and let the surplus programs write a neutral partial and
    # leave. The reduce skips a split whose m is -inf, so their part_acc is never
    # read and does not have to be written.
    if ADAPTIVE_SPLITS:
        m_tiles = (main_len + BLOCK_K - 1) // BLOCK_K
        e_tiles = (extra_len + BLOCK_K - 1) // BLOCK_K
        work_splits = gl.minimum(gl.maximum(gl.maximum(m_tiles, e_tiles), 1), NUM_SPLITS)
        main_splits = gl.minimum(gl.maximum(m_tiles, 1), work_splits)
        if split_id >= work_splits:
            pm_base = query_idx * pm_stride0 + split_id * pm_stride_s
            gl.amd.cdna4.buffer_store(
                gl.full(
                    [BLOCK_M],
                    float("-inf"),
                    gl.float32,
                    layout=gl.SliceLayout(1, pv_layout),
                ),
                ptr=part_m_ptr + pm_base,
                offsets=h_pv.to(gl.int32),
                mask=head_mask_pv,
            )
            gl.amd.cdna4.buffer_store(
                gl.zeros([BLOCK_M], gl.float32, layout=gl.SliceLayout(1, pv_layout)),
                ptr=part_l_ptr + pm_base,
                offsets=h_pv.to(gl.int32),
                mask=head_mask_pv,
            )
            return
    else:
        work_splits = NUM_SPLITS
        main_splits = MAIN_SPLITS

    # ---- main (SWA) segment ----
    main_chunk = (main_len + main_splits - 1) // main_splits
    main_lo = gl.minimum(split_id * main_chunk, main_len)
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
        scl_l,
        NARROW_SCALE,
        slot_l,
        MAIN_IS_FP8,
        MAIN_BLOCK_SIZE,
        NOPE_DIM,
        ROPE_DIM,
        HEAD_SIZE,
        BLOCK_M,
        BLOCK_K,
        NOPE_CHUNK,
        CHUNK_AXIS,
        HEAD_ALIGNED,
        UNIFORM,
        MAIN_USE_BUFFER_LOAD,
        HAS_INVALID,
        FP8_FNUZ,
    )

    if HAS_EXTRA:
        extra_chunk = (extra_len + work_splits - 1) // work_splits
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
            scl_l,
            NARROW_SCALE,
            slot_l,
            EXTRA_IS_FP8,
            EXTRA_BLOCK_SIZE,
            NOPE_DIM,
            ROPE_DIM,
            HEAD_SIZE,
            BLOCK_M,
            BLOCK_K,
            NOPE_CHUNK,
            CHUNK_AXIS,
            HEAD_ALIGNED,
            UNIFORM,
            EXTRA_USE_BUFFER_LOAD,
            HAS_INVALID,
            FP8_FNUZ,
        )

    # m_i/l_i are in SliceLayout(1, qk_layout); acc in pv_layout. Move the row
    # reductions into pv-slice space for output/partials.
    m_pv = gl.convert_layout(m_i, gl.SliceLayout(1, pv_layout))
    l_pv = gl.convert_layout(l_i, gl.SliceLayout(1, pv_layout))

    if NUM_SPLITS == 1:
        if HAS_SINK:
            # m_pv is already the row-max in the base-2 exponent domain
            # (row_max * softmax_scale * log2e); lift the sink -- a scaled-score
            # logit -- into the same domain and combine there.
            sink = gl.amd.cdna4.buffer_load(
                ptr=attn_sink_ptr, offsets=h_pv, mask=head_mask_pv, other=float("-inf")
            ).to(gl.float32) * RCP_LN2
            m_final = _max2(m_pv, sink)
            alpha = gl.exp2(m_pv - m_final)
            l_final = l_pv * alpha + gl.exp2(sink - m_final)
            acc = acc * alpha[:, None]
        else:
            l_final = l_pv
        out = acc / l_final[:, None]
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
        # store un-normalized partials for the reduce kernel. m is already in the
        # base-2 exponent domain (row-max * softmax_scale * log2e), which is the
        # reduce/skip_reduce convention the triton kernel also uses.
        pm_base = query_idx * pm_stride0 + split_id * pm_stride_s
        # The reduce kernel reads these back immediately, so they want to stay
        # resident rather than stream to memory. ATT puts 8.5% of all stall cycles
        # on the 68 buffer_store_dwordx4 of the accumulator alone (~154 cycles
        # each), and the partials are ~31% of the kernel's HBM traffic.
        gl.amd.cdna4.buffer_store(
            m_pv,
            ptr=part_m_ptr + pm_base,
            offsets=h_pv.to(gl.int32),
            mask=head_mask_pv,
            cache=PART_STORE_CACHE,
        )
        gl.amd.cdna4.buffer_store(
            l_pv,
            ptr=part_l_ptr + pm_base,
            offsets=h_pv.to(gl.int32),
            mask=head_mask_pv,
            cache=PART_STORE_CACHE,
        )
        offs_d_a = gl.arange(0, HEAD_SIZE, layout=gl.SliceLayout(0, pv_layout))
        a_base = query_idx * pa_stride0 + split_id * pa_stride_s
        a_off = (a_base + h_pv[:, None] * pa_stride_h + offs_d_a[None, :]).to(gl.int32)
        # Follow part_acc's own dtype: at bf16 this halves both the partial traffic
        # (~31% of the kernel's HBM bytes) and the store instruction count.
        gl.amd.cdna4.buffer_store(
            acc.to(part_acc_ptr.dtype.element_ty),
            ptr=part_acc_ptr,
            offsets=a_off,
            mask=head_mask_pv[:, None],
            cache=PART_STORE_CACHE,
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
    ADAPTIVE_SPLITS: gl.constexpr,
    PART_LOAD_CACHE: gl.constexpr,
):
    """Split-KV combine: merge the per-split partials, fold the attn sink, and write
    the final output. Partials store m in the base-2 exponent domain (row-max *
    softmax_scale * log2e), matching the triton reduce. Grid: (num_queries, heads_blocks).

    BLOCK_M is the reduce's own head tile and is deliberately decoupled from the
    attention kernel's: the combine is pure bandwidth over
    [num_queries, num_heads, HEAD_SIZE] f32, so the only thing that matters is
    having enough workgroups to cover the machine. At BLOCK_M = num_heads there is
    one workgroup per query -- 64 of them on a 256-CU part, i.e. 3/4 of the GPU
    idle and one wave to hide every partial-load latency behind.
    """
    NUM_WARPS: gl.constexpr = gl.num_warps()
    RCP_LN2: gl.constexpr = 1.4426950408889634
    query_idx = gl.program_id(0)
    pid_h = gl.program_id(1)

    # One warp covers [BLOCK_M, 64//BLOCK_M * 8] -- lay the 64 lanes out so a
    # small BLOCK_M spends them on the head dim instead of idling them on rows.
    TPW0: gl.constexpr = BLOCK_M if BLOCK_M < 8 else 8
    TPW1: gl.constexpr = 64 // TPW0
    BLK: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 8],
        threads_per_warp=[TPW0, TPW1],
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
    # NOT made adaptive. Bounding these loops by the per-query split count instead
    # of NUM_SPLITS turns them into dynamic loops, and losing the static unroll
    # costs 5-7% on a uniform batch -- more than the ~0.9 us it saves when the
    # launch over-splits. The acc load is masked on m > -inf instead, which is
    # what actually has to be right (a split that bowed out wrote no accumulator).
    m_final = gl.full([BLOCK_M], neg_inf, gl.float32, layout=row_l)
    # pass 1: global max over splits
    for s in range(NUM_SPLITS):
        base = query_idx * pm_stride0 + s * pm_stride_s
        m_s = gl.amd.cdna4.buffer_load(
            ptr=part_m_ptr + base, offsets=h, mask=head_mask, other=neg_inf,
            cache=PART_LOAD_CACHE,
        )
        m_final = _max2(m_final, m_s)  # m_s already in base-2 exponent domain
    if HAS_SINK:
        sink = gl.amd.cdna4.buffer_load(
            ptr=attn_sink_ptr, offsets=h, mask=head_mask, other=neg_inf
        ).to(gl.float32)
        m_final = _max2(m_final, sink * RCP_LN2)  # lift sink to base-2

    # pass 2: weighted sums
    l_final = gl.zeros([BLOCK_M], gl.float32, layout=row_l)
    acc = gl.zeros([BLOCK_M, HEAD_SIZE], gl.float32, layout=BLK)
    for s in range(NUM_SPLITS):
        base = query_idx * pm_stride0 + s * pm_stride_s
        m_s = gl.amd.cdna4.buffer_load(
            ptr=part_m_ptr + base, offsets=h, mask=head_mask, other=neg_inf,
            cache=PART_LOAD_CACHE,
        )
        l_s = gl.amd.cdna4.buffer_load(
            ptr=part_l_ptr + base, offsets=h, mask=head_mask, other=0.0,
            cache=PART_LOAD_CACHE,
        )
        w = gl.exp2(m_s - m_final)
        l_final = l_final + w * l_s
        a_base = query_idx * pa_stride0 + s * pa_stride_s
        a_off = (a_base + h[:, None] * pa_stride_h + offs_d[None, :]).to(gl.int32)
        # An adaptive-split program that bowed out wrote m = -inf and no
        # accumulator, so its part_acc slot is uninitialized -- mask the load
        # rather than relying on w == 0 (0 * NaN is NaN).
        if ADAPTIVE_SPLITS:
            acc_mask = head_mask[:, None] & (m_s > neg_inf)[:, None]
        else:
            acc_mask = head_mask[:, None]
        acc_s = gl.amd.cdna4.buffer_load(
            ptr=part_acc_ptr, offsets=a_off, mask=acc_mask, other=0.0,
            cache=PART_LOAD_CACHE,
        )
        acc = acc + w[:, None] * acc_s.to(gl.float32)

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
