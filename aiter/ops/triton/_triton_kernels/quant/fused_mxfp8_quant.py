# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import triton
import triton.language as tl

from aiter.ops.triton._triton_kernels.quant.quant import _mxfp8_quant_op
from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr

# Fused RMSNorm + MXFP8 (1x32 e8m0) quant. Replaces the separate
# rmsnorm_quant(fp8 fnuz + fp32 1x128) + transcode-to-MXFP8 sequence used
# upstream of MXFP8-aware GEMMs (e.g. V4 q_norm -> wq_b).
#
# One program per row. Holds the full row in registers, so K is constrained
# by the BLOCK_SIZE_K constexpr (must be a power of two >= K).
#
# In:  x (M, K) bf16 or fp16
#      g (K,)  bf16 or fp16 weight
# Out: y (M, K) fp8 e4m3fn
#      scale (M, K // 32) uint8 e8m0


_fused_rms_mxfp8_repr = make_kernel_repr(
    "_fused_rms_mxfp8_kernel",
    [
        "BLOCK_SIZE_K",
        "QUANT_BLOCK_SIZE",
    ],
)


@triton.jit(repr=_fused_rms_mxfp8_repr)
def _fused_rms_mxfp8_kernel(
    x_ptr,
    g_ptr,
    y_ptr,
    s_ptr,
    M,
    K,
    stride_xm,
    stride_xk,
    stride_ym,
    stride_yk,
    stride_sm,
    stride_sn,
    epsilon,
    BLOCK_SIZE_K: tl.constexpr,  # power-of-2 covering full K
    QUANT_BLOCK_SIZE: tl.constexpr,  # =32
    NUM_PRGMS,  # row-loop stride, = the grid. Runtime: it is M, the token
    # count, and a constexpr there builds one kernel per prefill chunk length.
):
    """One program processes one row: rmsnorm then MXFP8 quant in registers."""
    row_start = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE_K)
    mask = col_offsets < K

    for row_idx in tl.range(row_start, M, NUM_PRGMS, num_stages=2):
        # Load full row, cast to fp32
        x = tl.load(
            x_ptr + row_idx * stride_xm + col_offsets * stride_xk,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        g = tl.load(g_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)

        # RMS norm
        ss = tl.sum(x * x, axis=-1)
        norm_factor = tl.math.rsqrt((ss / K) + epsilon)
        y_fp32 = x * norm_factor * g  # (BLOCK_SIZE_K,)

        # Reshape into (K // QUANT_BLOCK_SIZE, QUANT_BLOCK_SIZE) groups for amax.
        # BLOCK_SIZE_K is the power-of-2 padded size; we keep OOB lanes masked to 0
        # via the load above, so amax over them is 0 (won't affect the in-bounds max).
        y_2d = tl.reshape(y_fp32, (BLOCK_SIZE_K // QUANT_BLOCK_SIZE, QUANT_BLOCK_SIZE))
        scale_e8m0, quant_scale = _mxfp8_quant_op(y_2d, QUANT_AXIS=1)

        # Quantize: y_quant = y_fp32 * quant_scale (broadcast along inner 32).
        qx_2d = y_2d * quant_scale
        qx = tl.reshape(qx_2d, (BLOCK_SIZE_K,))
        y_fp8 = qx.to(y_ptr.type.element_ty)

        # Store y (mask OOB).
        tl.store(
            y_ptr + row_idx * stride_ym + col_offsets * stride_yk,
            y_fp8,
            mask=mask,
        )

        # Store scales: G entries for this row.
        n_groups: tl.constexpr = BLOCK_SIZE_K // QUANT_BLOCK_SIZE
        group_offsets = tl.arange(0, n_groups)
        group_mask = group_offsets < (K // QUANT_BLOCK_SIZE)
        scale_flat = tl.reshape(scale_e8m0, (n_groups,))
        tl.store(
            s_ptr + row_idx * stride_sm + group_offsets * stride_sn,
            scale_flat,
            mask=group_mask,
        )


# Dual fused RMSNorm: Q-side (MXFP8 quant + e8m0 scale emit) + K-side (bf16 out).
# Replaces the CK `fused_qk_rmsnorm_group_quant` semantics in one Triton launch
# for the MXFP8 GEMM path (Task #77). The two halves are independent (different
# weight, different K dim) so they're packed into one program per row to amortize
# launch overhead: same kernel launch loads both rows, normalizes both, stores Q
# fp8 + scale, stores K bf16. Each row's Q and K are independently RMSNorm'd
# (separate weights, separate eps, separate K dim) -- this kernel does NOT fuse
# their normalization arithmetic, only their launch.
#
# In:  q     (M, KQ) bf16 or fp16
#      kv    (M, KK) bf16 or fp16
#      gq    (KQ,)   bf16 or fp16 Q-RMSNorm weight
#      gk    (KK,)   bf16 or fp16 K-RMSNorm weight
# Out: yq    (M, KQ) fp8 e4m3fn
#      sq    (M, KQ // 32) uint8 e8m0
#      yk    (M, KK) bf16


_fused_dual_rmsnorm_mxfp8_quant_repr = make_kernel_repr(
    "_fused_dual_rmsnorm_mxfp8_quant_kernel",
    [
        "BLOCK_SIZE_KQ",
        "BLOCK_SIZE_KK",
        "QUANT_BLOCK_SIZE",
    ],
)


@triton.jit(repr=_fused_dual_rmsnorm_mxfp8_quant_repr)
def _fused_dual_rmsnorm_mxfp8_quant_kernel(
    q_ptr,
    k_ptr,
    gq_ptr,
    gk_ptr,
    yq_ptr,
    sq_ptr,
    yk_ptr,
    M,
    KQ,
    KK,
    stride_qm,
    stride_qn,
    stride_km,
    stride_kn,
    stride_yqm,
    stride_yqn,
    stride_sqm,
    stride_sqn,
    stride_ykm,
    stride_ykn,
    eps_q,
    eps_k,
    BLOCK_SIZE_KQ: tl.constexpr,  # power-of-2 covering full KQ
    BLOCK_SIZE_KK: tl.constexpr,  # power-of-2 covering full KK
    QUANT_BLOCK_SIZE: tl.constexpr,  # =32 (MXFP8 group size)
    NUM_PRGMS,  # row-loop stride; runtime, see `_fused_rms_mxfp8_kernel`
):
    """One program per row: do Q-side RMSNorm+MXFP8 quant AND K-side RMSNorm
    (bf16 out) in one launch. Mirrors the CK `fused_qk_rmsnorm_group_quant`
    fusion topology but emits MXFP8 1x32 (e8m0) scales for Q directly."""
    row_start = tl.program_id(0)

    q_col_offsets = tl.arange(0, BLOCK_SIZE_KQ)
    q_mask = q_col_offsets < KQ
    k_col_offsets = tl.arange(0, BLOCK_SIZE_KK)
    k_mask = k_col_offsets < KK

    n_q_groups: tl.constexpr = BLOCK_SIZE_KQ // QUANT_BLOCK_SIZE

    for row_idx in tl.range(row_start, M, NUM_PRGMS, num_stages=2):
        # ===== Q side: RMSNorm + MXFP8 quant =====
        x_q = tl.load(
            q_ptr + row_idx * stride_qm + q_col_offsets * stride_qn,
            mask=q_mask,
            other=0.0,
        ).to(tl.float32)
        g_q = tl.load(gq_ptr + q_col_offsets, mask=q_mask, other=0.0).to(tl.float32)

        ss_q = tl.sum(x_q * x_q, axis=-1)
        norm_q = tl.math.rsqrt((ss_q / KQ) + eps_q)
        y_q_fp32 = x_q * norm_q * g_q

        y_q_2d = tl.reshape(y_q_fp32, (n_q_groups, QUANT_BLOCK_SIZE))
        scale_q_e8m0, quant_scale_q = _mxfp8_quant_op(y_q_2d, QUANT_AXIS=1)

        qx_q_2d = y_q_2d * quant_scale_q
        qx_q = tl.reshape(qx_q_2d, (BLOCK_SIZE_KQ,))
        y_q_fp8 = qx_q.to(yq_ptr.type.element_ty)

        tl.store(
            yq_ptr + row_idx * stride_yqm + q_col_offsets * stride_yqn,
            y_q_fp8,
            mask=q_mask,
        )

        q_group_offsets = tl.arange(0, n_q_groups)
        q_group_mask = q_group_offsets < (KQ // QUANT_BLOCK_SIZE)
        scale_q_flat = tl.reshape(scale_q_e8m0, (n_q_groups,))
        tl.store(
            sq_ptr + row_idx * stride_sqm + q_group_offsets * stride_sqn,
            scale_q_flat,
            mask=q_group_mask,
        )

        # ===== K side: RMSNorm only, bf16 out =====
        x_k = tl.load(
            k_ptr + row_idx * stride_km + k_col_offsets * stride_kn,
            mask=k_mask,
            other=0.0,
        ).to(tl.float32)
        g_k = tl.load(gk_ptr + k_col_offsets, mask=k_mask, other=0.0).to(tl.float32)

        ss_k = tl.sum(x_k * x_k, axis=-1)
        norm_k = tl.math.rsqrt((ss_k / KK) + eps_k)
        y_k_fp32 = x_k * norm_k * g_k
        y_k_out = y_k_fp32.to(yk_ptr.type.element_ty)

        tl.store(
            yk_ptr + row_idx * stride_ykm + k_col_offsets * stride_ykn,
            y_k_out,
            mask=k_mask,
        )


# Flatten-then-MXFP8 quant. Takes (M, N1, N2) input, flattens the trailing two
# dims into N = N1 * N2, and emits per-1x32 MXFP8 (FP8 e4m3fn values + uint8
# e8m0 scales) along the flattened axis. One program per (m, n1); each program
# handles a row of N2 elements that contributes BLOCK_SIZE_N2 // 32 groups to
# the M-th row of the (M, N) flattened output.


_fused_flatten_mxfp8_quant_repr = make_kernel_repr(
    "_fused_flatten_mxfp8_quant_kernel",
    [
        "BLOCK_SIZE_N2",
        "QUANT_BLOCK_SIZE",
    ],
)


@triton.jit(repr=_fused_flatten_mxfp8_quant_repr)
def _fused_flatten_mxfp8_quant_kernel(
    x_ptr,
    out_ptr,
    out_scales_ptr,
    x_stride_m,
    x_stride_n1,
    x_stride_n2,
    out_stride_m,
    out_stride_n,
    out_scales_stride_m,
    out_scales_stride_n,
    N2,
    BLOCK_SIZE_N2: tl.constexpr,
    QUANT_BLOCK_SIZE: tl.constexpr,
):
    m = tl.program_id(0)
    n1 = tl.program_id(1)

    NUM_QUANT_BLOCKS: tl.constexpr = BLOCK_SIZE_N2 // QUANT_BLOCK_SIZE
    # In the flattened (M, N1 * N2) output, each n1 segment is exactly N2 wide
    # (not BLOCK_SIZE_N2), so stride between n1 segments must use N2 — otherwise
    # non-power-of-2 N2 (e.g. 7168) would gap-write the output.
    n2_groups = N2 // QUANT_BLOCK_SIZE

    n2_offs = tl.arange(0, BLOCK_SIZE_N2)
    x_mask = n2_offs < N2
    x_offs = m * x_stride_m + n1 * x_stride_n1 + n2_offs * x_stride_n2
    x = tl.load(x_ptr + x_offs, mask=x_mask, other=0.0).to(tl.float32)

    x_2d = tl.reshape(x, (NUM_QUANT_BLOCKS, QUANT_BLOCK_SIZE))
    scale_e8m0, quant_scale = _mxfp8_quant_op(x_2d, QUANT_AXIS=1)

    qx_2d = x_2d * quant_scale
    qx = tl.reshape(qx_2d, (BLOCK_SIZE_N2,))
    tl.store(
        out_ptr + m * out_stride_m + (n1 * N2 + n2_offs) * out_stride_n,
        qx.to(out_ptr.type.element_ty),
        mask=x_mask,
    )

    block_scale_offs = tl.arange(0, NUM_QUANT_BLOCKS)
    scale_flat = tl.reshape(scale_e8m0, (NUM_QUANT_BLOCKS,))
    tl.store(
        out_scales_ptr
        + m * out_scales_stride_m
        + (n1 * n2_groups + block_scale_offs) * out_scales_stride_n,
        scale_flat,
        mask=block_scale_offs < n2_groups,
    )


@triton.jit
def _fused_deepseek_v4_mxfp8_quant_q_pack_kernel(
    q_ptr, packed_ptr, rope_ptr, fp8_max,
    NOPE: tl.constexpr, ROPE: tl.constexpr, QK: tl.constexpr,
    GROUP: tl.constexpr, NUM_TILES: tl.constexpr,
):
    """One program per (token, head) row of Q.

    Writes the packed record the gfx1250 kernel reads:
      [0, 448)   NoPE e4m3, one UE8M0 scale per 64-element group
      [448, 462) the 7 scale bytes, EACH WRITTEN TWICE
      [462, 512) zero
    plus the RoPE plane, which is never quantized.
    """
    row = tl.program_id(0)
    src = q_ptr + row * QK
    dst = packed_ptr + row * QK

    for g in tl.static_range(NUM_TILES):
        off = g * GROUP + tl.arange(0, GROUP)
        x = tl.load(src + off).to(tl.float32)
        # clamp the RATIO, matching the reference packing the kernel's own
        # tests use -- flooring amax instead would shift the stored exponent
        # for all-zero groups
        amax = tl.max(tl.abs(x), axis=0)
        ratio = tl.maximum(amax / fp8_max, 1e-4)
        exponent = tl.ceil(tl.log2(ratio))
        scale = tl.exp2(exponent)
        f8 = (x / scale).to(tl.float8e4nv)
        tl.store(dst + off, f8.to(tl.uint8, bitcast=True))
        # the scaled-MMA blocks are 32 elements wide while the quant group is
        # 64, so the kernel reads each group's scale twice
        enc = tl.minimum(tl.maximum(exponent + 127.0, 0.0), 254.0).to(tl.uint8)
        tl.store(dst + NOPE + 2 * g, enc)
        tl.store(dst + NOPE + 2 * g + 1, enc)

    # [462, 512): zero, so no stale byte reaches the MMA
    tail = tl.arange(0, QK)
    tl.store(dst + tail, tl.zeros((QK,), dtype=tl.uint8),
             mask=tail >= NOPE + 2 * NUM_TILES)

    r = tl.arange(0, ROPE)
    tl.store(rope_ptr + row * ROPE + r, tl.load(src + NOPE + r))


# ---------------------------------------------------------------------------
# DSv4 decode producer, fused
# ---------------------------------------------------------------------------
# Everything the model does to Q and KV between the projections and the sparse
# decode, in one launch:
#
#   Q   RMSNorm (no weight) -> GPT-J RoPE -> bf16 out, and the UE8M0 fp8 pack
#       taken off the SAME registers rather than from a second pass over Q
#   KV  GPT-J RoPE -> UE8M0 quant -> insert into the paged ALIGNED cache
#
# The two are different work on different tensors, so they are dispatched by
# head slot rather than fused arithmetically: grid dim 1 runs
# [0, padded_heads) for Q and one extra slot for KV. That is the same shape of
# dispatch the .cu producer uses, except the .cu gives each (token, slot) a
# WARP and 16 elements per lane where this gives it a whole program -- a
# difference worth measuring, not assuming.
#
# ALIGNED cache record, 640 B per token:
#     [  0, 448)  NoPE fp8 e4m3
#     [448, 462)  the 7 UE8M0 scales, EACH WRITTEN TWICE
#     [462, 512)  pad
#     [512, 640)  RoPE bf16
#
# Bytes [0, 512) are byte-identical to an ATOM 2buff row, which is why the
# packed Q rows above and these KV rows share one packing.


@triton.jit
def _v4_rope_pair(x_even, x_odd, cos, sin):
    return x_even * cos - x_odd * sin, x_even * sin + x_odd * cos


@triton.jit
def _fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert_aligned_kernel(
    q_in_ptr,  # [T, num_heads_q, HEAD] bf16
    q_out_ptr,  # [T, padded_heads, HEAD] bf16
    q_packed_ptr,  # [T, padded_heads, HEAD] uint8 (the 2buff row)
    q_rope_ptr,  # [T, padded_heads, ROPE] bf16
    kv_in_ptr,  # [T, HEAD] bf16
    kv_cache_ptr,  # [nb, block, REC] uint8
    kv_slot_ptr,  # [T] int, -1 to skip
    pos_ptr,  # [T]
    cs_ptr,  # [max_pos, ROPE] fp32, cos || sin
    kv_block_stride,
    kv_cache_block_size,
    eps,
    fp8_max,
    num_heads_q: tl.constexpr,
    padded_heads: tl.constexpr,
    HEAD: tl.constexpr,
    NOPE: tl.constexpr,
    ROPE: tl.constexpr,
    GROUP: tl.constexpr,
    NUM_TILES: tl.constexpr,
    REC: tl.constexpr,
    SC_OFF: tl.constexpr,
    ROPE_OFF: tl.constexpr,
    APPLY_NORM: tl.constexpr,
    PACK_Q: tl.constexpr,
    WRITE_Q: tl.constexpr,
    USE_FNUZ: tl.constexpr,
):
    """One program per (token, slot); slot == padded_heads is the KV row.

    Padding slots zero-fill every Q output. That is a correctness requirement,
    not tidiness: an unwritten scale byte of 0xFF is E8M0 NaN, and the decode
    kernel multiplies a masked score by it, so 0 * NaN would poison the row.
    """
    tok = tl.program_id(0)
    h = tl.program_id(1)
    d = tl.arange(0, HEAD)
    half: tl.constexpr = ROPE // 2
    p = tl.arange(0, ROPE // 2)

    pos = tl.load(pos_ptr + tok)
    cos = tl.load(cs_ptr + pos * ROPE + p)
    sin = tl.load(cs_ptr + pos * ROPE + half + p)

    # ---- KV: RoPE, quantize, write one aligned record ---------------------
    if h == padded_heads:
        slot = tl.load(kv_slot_ptr + tok)
        if slot == -1:
            return
        blk = slot // kv_cache_block_size
        pos_in_blk = slot % kv_cache_block_size
        # int64: blk * block_stride exceeds 2^31 on a production-sized pool
        rec = kv_cache_ptr + blk.to(tl.int64) * kv_block_stride + pos_in_blk * REC
        row = kv_in_ptr + tok * HEAD

        for g in tl.static_range(NUM_TILES):
            off = g * GROUP + tl.arange(0, GROUP)
            x = tl.load(row + off).to(tl.float32)
            amax = tl.maximum(tl.max(tl.abs(x), axis=0), 1e-4)
            exponent = tl.ceil(tl.log2(amax / fp8_max))
            scale = tl.exp2(exponent)
            xs = tl.clamp(x / scale, -fp8_max, fp8_max)
            if USE_FNUZ:
                f8 = xs.to(tl.float8e4b8)
            else:
                f8 = xs.to(tl.float8e4nv)
            tl.store(rec + off, f8.to(tl.uint8, bitcast=True))
            enc = tl.maximum(tl.minimum(exponent + 127.0, 255.0), 0.0).to(tl.uint8)
            # both copies: one scale per 32 columns, one quant group per 64
            tl.store(rec + SC_OFF + 2 * g, enc)
            tl.store(rec + SC_OFF + 2 * g + 1, enc)

        xe = tl.load(row + NOPE + 2 * p).to(tl.float32)
        xo = tl.load(row + NOPE + 2 * p + 1).to(tl.float32)
        ye, yo = _v4_rope_pair(xe, xo, cos, sin)
        rope_out = (rec + ROPE_OFF).to(tl.pointer_type(tl.bfloat16))
        tl.store(rope_out + 2 * p, ye.to(tl.bfloat16))
        tl.store(rope_out + 2 * p + 1, yo.to(tl.bfloat16))
        return

    # ---- Q: padding slot --------------------------------------------------
    dst = q_out_ptr + (tok * padded_heads + h) * HEAD
    if h >= num_heads_q:
        if WRITE_Q:
            tl.store(dst + d, tl.zeros((HEAD,), dtype=tl.bfloat16))
        if PACK_Q:
            pdst = q_packed_ptr + (tok * padded_heads + h) * HEAD
            tl.store(pdst + d, tl.zeros((HEAD,), dtype=tl.uint8))
            r0 = tl.arange(0, ROPE)
            tl.store(
                q_rope_ptr + (tok * padded_heads + h) * ROPE + r0,
                tl.zeros((ROPE,), dtype=tl.bfloat16),
            )
        return

    # ---- Q: live head -----------------------------------------------------
    base = q_in_ptr + (tok * num_heads_q + h) * HEAD
    # RMSNorm over the whole head, no weight -- one scale for both halves
    inv = 1.0
    if APPLY_NORM:
        xf = tl.load(base + d).to(tl.float32)
        inv = tl.rsqrt(tl.sum(xf * xf, axis=0) / HEAD + eps)

    xe = tl.load(base + NOPE + 2 * p).to(tl.float32) * inv
    xo = tl.load(base + NOPE + 2 * p + 1).to(tl.float32) * inv
    ye, yo = _v4_rope_pair(xe, xo, cos, sin)

    # bf16 Q out. arange must be a power of two, so the 512-wide range is
    # masked down to the NoPE half rather than sized to it.
    nope_m = d < NOPE
    if WRITE_Q:
        tl.store(
            dst + d,
            (tl.load(base + d, mask=nope_m, other=0.0).to(tl.float32) * inv).to(
                tl.bfloat16
            ),
            mask=nope_m,
        )
        tl.store(dst + NOPE + 2 * p, ye.to(tl.bfloat16))
        tl.store(dst + NOPE + 2 * p + 1, yo.to(tl.bfloat16))

    if PACK_Q:
        pdst = q_packed_ptr + (tok * padded_heads + h) * HEAD
        for g in tl.static_range(NUM_TILES):
            off = g * GROUP + tl.arange(0, GROUP)
            # the NoPE half is normed but NOT rotated, so re-read and scale
            x = tl.load(base + off).to(tl.float32) * inv
            amax = tl.max(tl.abs(x), axis=0)
            # clamp the RATIO, which is what the reference packing does
            exponent = tl.ceil(tl.log2(tl.maximum(amax / fp8_max, 1e-4)))
            xs = x / tl.exp2(exponent)
            if USE_FNUZ:
                f8 = xs.to(tl.float8e4b8)
            else:
                f8 = xs.to(tl.float8e4nv)
            tl.store(pdst + off, f8.to(tl.uint8, bitcast=True))
            enc = tl.minimum(tl.maximum(exponent + 127.0, 0.0), 254.0).to(tl.uint8)
            tl.store(pdst + NOPE + 2 * g, enc)
            tl.store(pdst + NOPE + 2 * g + 1, enc)
        tl.store(
            pdst + d,
            tl.zeros((HEAD,), dtype=tl.uint8),
            mask=d >= NOPE + 2 * NUM_TILES,
        )
        # the RoPE plane is never quantized
        rdst = q_rope_ptr + (tok * padded_heads + h) * ROPE
        tl.store(rdst + 2 * p, ye.to(tl.bfloat16))
        tl.store(rdst + 2 * p + 1, yo.to(tl.bfloat16))


# ---------------------------------------------------------------------------
# DSv4 K-cache insert (ported from vLLM's cache_utils.py)
# ---------------------------------------------------------------------------
# Writes either record layout, chosen by the geometry the caller passes:
#   packed  584 accounting / 576 stride: fp8 [0,448) | bf16 [448,576), with the
#           block's scales in a region after all of its token data
#   aligned 640: fp8 [0,448) | scales [448,462) | pad | bf16 [512,640)
@triton.jit
def quantize_and_insert_k_kernel(
    # Input tensors
    k_ptr,  # [num_tokens, 512] bf16
    slot_mapping_ptr,  # [num_tokens] int64
    # Output tensor
    k_cache_ptr,  # [num_blocks, block_bytes] as uint8 (flattened view)
    # Dimensions
    num_tokens,
    input_dim: tl.constexpr,  # 512
    fp8_dim: tl.constexpr,  # 448
    bf16_dim: tl.constexpr,  # 64
    scale_dim: tl.constexpr,  # 8
    quant_block: tl.constexpr,  # 64 (quantization block size)
    cache_block_size: tl.constexpr,  # 64 (paged cache block size)
    token_data_size: tl.constexpr,  # 576 bytes per token data
    rec_bytes: tl.constexpr,  # 584 packed, 640 aligned
    sc_in_rec: tl.constexpr,  # 0 packed, 448 aligned
    rope_in_rec: tl.constexpr,  # 448 packed, 512 aligned
    sc_step: tl.constexpr,  # 1 packed, 2 aligned (each scale written twice)
    block_stride: tl.constexpr,  # total bytes per block (padded)
    fp8_max: tl.constexpr,
    n_quant_blocks: tl.constexpr,  # 8 (7 real + 1 padding)
    use_fnuz: tl.constexpr = False,
):
    """
    Quantize K tensor and insert into paged K cache.

    K Cache block layout (block_size=64 tokens):
    - [0, 64*576): Token data, each token has 448 fp8 + 128 bf16
    - [64*576, 64*576 + 64*8): Scales, each token has 8 uint8 scales
    - [64*576 + 64*8, block_stride): Padding

    One program per token.

    ``use_fnuz=True`` selects FNUZ (``tl.float8e4b8``); default OCP
    (``tl.float8e4nv``) matches every production caller.
    """
    pid = tl.program_id(0)

    if pid >= num_tokens:
        return

    # Get slot mapping
    slot_idx = tl.load(slot_mapping_ptr + pid)
    if slot_idx == -1:
        return

    block_idx = slot_idx // cache_block_size
    pos_in_block = slot_idx % cache_block_size

    # Input pointer for this token
    input_row_ptr = k_ptr + pid * input_dim

    # int64: block_idx * block_stride can exceed 2^31 with many KV-cache blocks
    # (e.g. >= 57K at block_stride ~37K). Matches gather path below.
    cache_block_ptr = k_cache_ptr + block_idx.to(tl.int64) * block_stride

    # Token data pointer: token data is stored contiguously at start of block
    # Each token's data is at offset pos_in_block * token_data_size
    # rec_bytes == token_data_size selects the PACKED layout (scales grouped in a
    # per-block region); rec_bytes > token_data_size selects the interleaved
    # layout, where each token's scales sit at +sc_in_rec inside its own record.
    token_data_ptr = cache_block_ptr + pos_in_block * rec_bytes
    if sc_in_rec > 0:
        token_scale_ptr = token_data_ptr + sc_in_rec
    else:
        token_scale_ptr = (
            cache_block_ptr
            + cache_block_size * token_data_size
            + pos_in_block * scale_dim
        )

    # fp8 always starts the record; the bf16 half follows it on the packed
    # record and follows the scales and pad on the aligned one.
    token_fp8_ptr = token_data_ptr
    token_bf16_ptr = token_data_ptr + rope_in_rec

    # ========== Quantize and store FP8 portion (first 448 elements) ==========
    # Using UE8M0 quantization strategy (scale is power of 2, stored as uint8 exponent)
    for qblock_idx in tl.static_range(n_quant_blocks):
        qblock_start = qblock_idx * quant_block

        if qblock_start < fp8_dim:
            offsets = qblock_start + tl.arange(0, quant_block)
            mask = offsets < fp8_dim

            # Load bf16 input
            x = tl.load(input_row_ptr + offsets, mask=mask, other=0.0)

            # Compute absmax scale (same as CUDA kernel)
            abs_x = tl.abs(x)
            block_max = tl.max(abs_x, axis=0)
            block_max = tl.maximum(block_max, 1e-4)  # Match CUDA: fmaxf(amax, 1e-4)

            # UE8M0: Round scale UP to next power of 2
            # scale = 2^ceil(log2(block_max / fp8_max))
            raw_scale = block_max / fp8_max
            log_scale = tl.log2(raw_scale)
            exponent = tl.ceil(log_scale)  # Round UP to next integer exponent
            scale = tl.exp2(exponent)  # scale = 2^exponent (power of 2)

            # Quantize to fp8: fp8_value = bf16_value / scale
            x_scaled = x / scale
            x_clamped = tl.clamp(x_scaled, -fp8_max, fp8_max)

            # Convert to fp8 (FNUZ on gfx942, OCP elsewhere), then bitcast to uint8.
            if use_fnuz:
                x_fp8 = x_clamped.to(tl.float8e4b8)
            else:
                x_fp8 = x_clamped.to(tl.float8e4nv)
            x_uint8 = x_fp8.to(tl.uint8, bitcast=True)

            # Store as uint8 (1 byte each)
            tl.store(token_fp8_ptr + offsets, x_uint8, mask=mask)

            # UE8M0 scale encoding: stored_value = exponent + 127 (bias)
            # During dequant: scale = 2^(stored_value - 127)
            encoded_scale = exponent + 127.0
            encoded_scale = tl.maximum(tl.minimum(encoded_scale, 255.0), 0.0)
            tl.store(
                token_scale_ptr + qblock_idx * sc_step, encoded_scale.to(tl.uint8)
            )
            if sc_step == 2:
                tl.store(
                    token_scale_ptr + qblock_idx * sc_step + 1,
                    encoded_scale.to(tl.uint8),
                )

    if sc_step == 1:
        # The packed region is 8 bytes per token for 7 groups. An unwritten pad
        # byte of 0xFF is E8M0 NaN and 0 * NaN poisons a whole score row.
        # The aligned record has no such byte: its 14 scales fill their span and
        # the decode descriptor zero-fills the two MX blocks past them.
        tl.store(token_scale_ptr + 7, tl.zeros((), dtype=tl.uint8))

    # ========== Store BF16 portion (last 64 elements, no quantization) ==========
    bf16_input_offset = fp8_dim

    # Process bf16 in chunks of 16
    bf16_out_ptr = token_bf16_ptr.to(tl.pointer_type(tl.bfloat16))
    for i in tl.static_range(bf16_dim // 16):
        chunk_offsets = i * 16 + tl.arange(0, 16)
        bf16_vals = tl.load(input_row_ptr + bf16_input_offset + chunk_offsets)
        tl.store(bf16_out_ptr + chunk_offsets, bf16_vals)


# ---------------------------------------------------------------------------
# DSv4 KV compressor kernels (ported from vLLM's fused_compress_quant_cache.py)
# ---------------------------------------------------------------------------
# Both record layouts, chosen by the geometry the launcher passes. DSv4-Pro on
# ROCm uses the single-pass compressor and the two-stage one.

@triton.jit
def _fused_kv_compress_norm_rope_insert_sparse_attn(
    # ── state cache (compressor internal state) ──
    state_cache_ptr,
    state_cache_stride0,
    state_cache_stride1,
    # ── metadata ──
    token_to_req_indices_ptr,
    positions_ptr,
    slot_mapping_ptr,
    block_table_ptr,
    block_table_stride,
    block_size,
    # ── RMSNorm ──
    rms_norm_weight_ptr,
    rms_norm_eps,
    # ── RoPE ──
    cos_sin_cache_ptr,
    cos_sin_stride,
    # ── KV cache output ──
    k_cache_ptr,
    kv_slot_mapping_ptr,
    kv_cache_block_size,
    # ── constexprs ──
    HEAD_SIZE: tl.constexpr,
    TRITON_BLOCK_SIZE: tl.constexpr,
    STATE_WIDTH: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    OVERLAP: tl.constexpr,
    ROPE_HEAD_DIM: tl.constexpr,
    FP8_MAX: tl.constexpr,  # 448.0
    QUANT_BLOCK: tl.constexpr,  # 64 for DeepseekV4
    TOKEN_STRIDE: tl.constexpr,  # 576 for DeepseekV4
    SC_IN_REC: tl.constexpr,  # 0 packed, 448 aligned
    ROPE_IN_REC: tl.constexpr,  # 448 packed, 512 aligned
    SC_STEP: tl.constexpr,  # 1 packed, 2 aligned (each scale written twice)
    SCALE_DIM: tl.constexpr,  # 8 for DeepseekV4 (7 real + 1 pad)
    KV_BLOCK_STRIDE: tl.constexpr,
    SANITIZE_CACHE_NANS: tl.constexpr,
):
    """Fused compress → RMSNorm → FP8 quant (nope) → RoPE → bf16 store (rope).

    One program per token; early-exits for non-boundary positions.

    Cache block layout (``block_size`` tokens):
      [0, bs*576):       token data (448 fp8 + 128 bf16 each)
      [bs*576, +bs*8):   uint8 UE8M0 scales (7 real + 1 pad each)
    """
    token_idx = tl.program_id(0)

    slot_id = tl.load(slot_mapping_ptr + token_idx)
    if slot_id < 0:
        return

    position = tl.load(positions_ptr + token_idx)
    if (position + 1) % COMPRESS_RATIO != 0:
        return

    req_idx = tl.load(token_to_req_indices_ptr + token_idx)

    # ── Gather state cache entries ────────────────────────────────────
    start = position - (1 + OVERLAP) * COMPRESS_RATIO + 1
    tokens = tl.arange(0, (1 + OVERLAP) * COMPRESS_RATIO)
    pos = start + tokens
    mask_pos = pos >= 0

    block_indices = pos // block_size
    block_numbers = tl.load(
        block_table_ptr + req_idx * block_table_stride + block_indices,
        mask=mask_pos,
        other=0,
    )
    block_offsets = pos % block_size
    head_offset = (tokens >= COMPRESS_RATIO).to(tl.int32) * HEAD_SIZE

    block = tl.arange(0, TRITON_BLOCK_SIZE)
    mask = block < HEAD_SIZE
    block_numbers_i64 = block_numbers.to(tl.int64)

    # Precomputed row base shared by score and kv loads
    row_base = (
        state_cache_ptr
        + block_numbers_i64 * state_cache_stride0
        + block_offsets * state_cache_stride1
        + head_offset
    )

    combined_mask = mask_pos[:, None] & mask[None, :]

    # ── Softmax + weighted sum ───────────────────────────────────────
    score = tl.load(
        row_base[:, None] + STATE_WIDTH + block[None, :],
        mask=combined_mask,
        other=float("-inf"),
    )
    score = tl.softmax(score, dim=0)

    kv = tl.load(
        row_base[:, None] + block[None, :],
        mask=combined_mask,
        other=0.0,
    )

    compressed_kv = tl.sum(kv * score, axis=0)  # [TRITON_BLOCK_SIZE] fp32

    # ── RMSNorm (fp32 throughout) ──────────────────────────────────────
    rms_w = tl.load(rms_norm_weight_ptr + block, mask=mask, other=0.0)
    variance = tl.sum(compressed_kv * compressed_kv, axis=0) / HEAD_SIZE
    rrms = tl.rsqrt(variance + rms_norm_eps)
    normed = compressed_kv * rrms * rms_w

    # ── KV cache pointers ────────────────────────────────────────────
    kv_slot_idx = tl.load(kv_slot_mapping_ptr + token_idx)
    if kv_slot_idx < 0:
        return
    kv_block_idx = kv_slot_idx // kv_cache_block_size
    kv_pos_in_block = kv_slot_idx % kv_cache_block_size

    cache_block_ptr = k_cache_ptr + kv_block_idx.to(tl.int64) * KV_BLOCK_STRIDE
    fp8_ptr = cache_block_ptr + kv_pos_in_block * TOKEN_STRIDE
    # SC_IN_REC > 0: interleaved -- each token's scales live inside its own
    # record. 0: packed -- the scales are grouped after the block's token data.
    if SC_IN_REC > 0:
        scale_ptr = (
            cache_block_ptr + kv_pos_in_block * TOKEN_STRIDE + SC_IN_REC
        )
    else:
        scale_ptr = (
            cache_block_ptr
            + kv_cache_block_size * TOKEN_STRIDE
            + kv_pos_in_block * SCALE_DIM
        )

    NOPE_HEAD_DIM: tl.constexpr = HEAD_SIZE - ROPE_HEAD_DIM  # 448
    HALF_ROPE: tl.constexpr = ROPE_HEAD_DIM // 2  # 32

    # FP8 UE8M0 quant: cast fp32 → bf16 → fp32 before quant to match reference.
    N_QUANT_BLOCKS: tl.constexpr = TRITON_BLOCK_SIZE // QUANT_BLOCK
    N_NOPE_BLOCKS: tl.constexpr = NOPE_HEAD_DIM // QUANT_BLOCK  # 7
    INV_FP8_MAX: tl.constexpr = 1.0 / FP8_MAX

    quant_input = normed.to(tl.bfloat16).to(tl.float32)
    quant_2d = tl.reshape(quant_input, (N_QUANT_BLOCKS, QUANT_BLOCK))
    abs_2d = tl.abs(quant_2d)
    block_absmax = tl.max(abs_2d, axis=1)  # [N_QUANT_BLOCKS] fp32
    block_absmax = tl.maximum(block_absmax, 1e-4)

    raw_scales = block_absmax * INV_FP8_MAX
    exponents = tl.ceil(tl.log2(raw_scales))
    inv_scales = tl.exp2(-exponents)
    inv_scales_col = tl.reshape(inv_scales, (N_QUANT_BLOCKS, 1))
    x_scaled = quant_2d * inv_scales_col
    x_clamped = tl.clamp(x_scaled, -FP8_MAX, FP8_MAX)
    x_fp8 = x_clamped.to(tl.float8e4nv)
    x_uint8 = x_fp8.to(tl.uint8, bitcast=True)
    x_uint8_flat = tl.reshape(x_uint8, (TRITON_BLOCK_SIZE,))

    nope_mask = block < NOPE_HEAD_DIM
    tl.store(fp8_ptr + block, x_uint8_flat, mask=nope_mask)

    scale_idx = tl.arange(0, N_QUANT_BLOCKS)
    encoded = exponents + 127.0
    max_encoded: tl.constexpr = 254.0 if SANITIZE_CACHE_NANS else 255.0
    encoded = tl.maximum(tl.minimum(encoded, max_encoded), 0.0)
    tl.store(
        scale_ptr + scale_idx * SC_STEP,
        encoded.to(tl.uint8),
        mask=scale_idx < N_NOPE_BLOCKS,
    )
    if SC_STEP == 2:
        # one byte per 32 columns: what the decode kernel's scaled MMA reads,
        # against a 64-element quant group
        tl.store(
            scale_ptr + scale_idx * SC_STEP + 1,
            encoded.to(tl.uint8),
            mask=scale_idx < N_NOPE_BLOCKS,
        )
    else:
        # the packed region is 8 bytes for 7 groups; an unwritten 0xFF is E8M0
        # NaN and 0 * NaN poisons a whole score row
        tl.store(scale_ptr + N_NOPE_BLOCKS, tl.zeros((), dtype=tl.uint8))

    # GPT-J RoPE, full width. Splitting the pair into even and odd halves and
    # rebuilding a full-width value from them costs registers. The identity
    # used by _triton_kernels/rope/rope.py needs neither: for a pair (x0, x1)
    # the rotation is
    #
    #     out = x * cos + rot * sin,      rot = (-x1, x0)
    #
    # and rot is just a sign flip on the even lane followed by a flip of the
    # minor axis, so every value stays in its own lane. cos/sin are loaded per
    # LANE (index lane//2) rather than per pair, which reads each entry twice
    # from cache but removes the broadcast.
    NUM_PAIRS: tl.constexpr = TRITON_BLOCK_SIZE // 2
    NOPE_PAIRS: tl.constexpr = NOPE_HEAD_DIM // 2

    rope_pair_local = (block // 2) - NOPE_PAIRS
    is_rope_lane = rope_pair_local >= 0
    cs_idx = tl.maximum(rope_pair_local, 0)

    compressed_pos = (position // COMPRESS_RATIO) * COMPRESS_RATIO
    cache_base = cos_sin_cache_ptr + compressed_pos * cos_sin_stride
    cos_v = tl.load(cache_base + cs_idx, mask=is_rope_lane, other=1.0)
    sin_v = tl.load(cache_base + HALF_ROPE + cs_idx, mask=is_rope_lane, other=0.0)

    rot = tl.where(block % 2 == 0, normed, -normed)
    rot = tl.reshape(rot, (NUM_PAIRS, 2))
    rot = tl.flip(rot, 1)
    rot = tl.reshape(rot, (TRITON_BLOCK_SIZE,))
    result = normed * cos_v + rot * sin_v
    if SANITIZE_CACHE_NANS:
        result = tl.where(result == result, result, 0.0)

    # Store rotated rope portion as bf16 into the cache's bf16 area.
    bf16_ptr = (fp8_ptr + ROPE_IN_REC).to(tl.pointer_type(tl.bfloat16))
    rope_local = block - NOPE_HEAD_DIM
    is_rope = (block >= NOPE_HEAD_DIM) & mask
    tl.store(bf16_ptr + rope_local, result.to(tl.bfloat16), mask=is_rope)


# =============================================================================
# Split kernels variant of the head=512 compressor (deep cr=128 gather).
#  - compress gather: instead of launching one program per token, split along
#    the head dimension to maximize CU occupancy. The head dimension split
#    does not require cross-group reduction
#  - finalize norm rope quant store: same as the single pass kernel due to its
#    per-token nature
# Mirrors the CUDA cutedsl split kernel where num_splits is occupancy-targeted.
# Currently only tested and validated on ROCm gfx950
# =============================================================================

@triton.jit
def _compress_gather_split_sparse_attn(
    state_cache_ptr,
    state_cache_stride0,
    state_cache_stride1,
    positions_ptr,
    slot_mapping_ptr,
    token_to_req_indices_ptr,
    block_table_ptr,
    block_table_stride,
    block_size,
    scratch_ptr,
    scratch_stride,
    HEAD_SIZE: tl.constexpr,
    STATE_WIDTH: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    HEAD_TILE: tl.constexpr,  # HEAD_SIZE // NUM_SPLITS
):
    """Stage 1: per-(token, head-split) compress gather, write to fp32 scratch

    No-overlap gather (cr>=128) on rows [0, COMPRESS_RATIO)
    """
    pid = tl.program_id(0)
    token_idx = pid // NUM_SPLITS
    split_idx = pid % NUM_SPLITS

    slot_id = tl.load(slot_mapping_ptr + token_idx)
    if slot_id < 0:
        return
    position = tl.load(positions_ptr + token_idx)
    if (position + 1) % COMPRESS_RATIO != 0:
        return
    req_idx = tl.load(token_to_req_indices_ptr + token_idx)

    start = position - COMPRESS_RATIO + 1
    rows = tl.arange(0, COMPRESS_RATIO)
    pos = start + rows
    mask_pos = pos >= 0
    block_numbers = tl.load(
        block_table_ptr + req_idx * block_table_stride + pos // block_size,
        mask=mask_pos,
        other=0,
    ).to(tl.int64)
    block_offsets = pos % block_size

    col = split_idx * HEAD_TILE + tl.arange(0, HEAD_TILE)
    row_base = (
        state_cache_ptr
        + block_numbers * state_cache_stride0
        + block_offsets * state_cache_stride1
    )
    cmask = mask_pos[:, None]

    score = tl.load(
        row_base[:, None] + STATE_WIDTH + col[None, :],
        mask=cmask,
        other=float("-inf"),
    )
    score = tl.softmax(score, dim=0)
    kv = tl.load(row_base[:, None] + col[None, :], mask=cmask, other=0.0)
    compressed = tl.sum(kv * score, axis=0)  # [HEAD_TILE] fp32
    tl.store(scratch_ptr + token_idx * scratch_stride + col, compressed)

@triton.jit
def _finalize_norm_rope_quant_store_sparse_attn(
    scratch_ptr,
    scratch_stride,
    positions_ptr,
    slot_mapping_ptr,
    rms_norm_weight_ptr,
    rms_norm_eps,
    cos_sin_cache_ptr,
    cos_sin_stride,
    k_cache_ptr,
    kv_slot_mapping_ptr,
    kv_cache_block_size,
    HEAD_SIZE: tl.constexpr,
    TRITON_BLOCK_SIZE: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    ROPE_HEAD_DIM: tl.constexpr,
    FP8_MAX: tl.constexpr,
    QUANT_BLOCK: tl.constexpr,
    TOKEN_STRIDE: tl.constexpr,
    SC_IN_REC: tl.constexpr,  # 0 packed, 448 aligned
    ROPE_IN_REC: tl.constexpr,  # 448 packed, 512 aligned
    SC_STEP: tl.constexpr,  # 1 packed, 2 aligned (each scale written twice)
    SCALE_DIM: tl.constexpr,
    KV_BLOCK_STRIDE: tl.constexpr,
    SANITIZE_CACHE_NANS: tl.constexpr,
):
    """Stage 2: read compressed_kv[512] from scratch buffer, then
    RMSNorm + FP8 quant (nope) + RoPE + bf16 store
    """
    token_idx = tl.program_id(0)
    slot_id = tl.load(slot_mapping_ptr + token_idx)
    if slot_id < 0:
        return
    position = tl.load(positions_ptr + token_idx)
    if (position + 1) % COMPRESS_RATIO != 0:
        return

    block = tl.arange(0, TRITON_BLOCK_SIZE)
    mask = block < HEAD_SIZE
    compressed_kv = tl.load(
        scratch_ptr + token_idx * scratch_stride + block, mask=mask, other=0.0
    )

    rms_w = tl.load(rms_norm_weight_ptr + block, mask=mask, other=0.0)
    variance = tl.sum(compressed_kv * compressed_kv, axis=0) / HEAD_SIZE
    rrms = tl.rsqrt(variance + rms_norm_eps)
    normed = compressed_kv * rrms * rms_w

    kv_slot_idx = tl.load(kv_slot_mapping_ptr + token_idx)
    if kv_slot_idx < 0:
        return
    kv_block_idx = kv_slot_idx // kv_cache_block_size
    kv_pos_in_block = kv_slot_idx % kv_cache_block_size
    cache_block_ptr = k_cache_ptr + kv_block_idx.to(tl.int64) * KV_BLOCK_STRIDE
    fp8_ptr = cache_block_ptr + kv_pos_in_block * TOKEN_STRIDE
    # SC_IN_REC > 0: interleaved -- each token's scales live inside its own
    # record. 0: packed -- the scales are grouped after the block's token data.
    if SC_IN_REC > 0:
        scale_ptr = (
            cache_block_ptr + kv_pos_in_block * TOKEN_STRIDE + SC_IN_REC
        )
    else:
        scale_ptr = (
            cache_block_ptr
            + kv_cache_block_size * TOKEN_STRIDE
            + kv_pos_in_block * SCALE_DIM
        )

    NOPE_HEAD_DIM: tl.constexpr = HEAD_SIZE - ROPE_HEAD_DIM
    HALF_ROPE: tl.constexpr = ROPE_HEAD_DIM // 2
    N_QUANT_BLOCKS: tl.constexpr = TRITON_BLOCK_SIZE // QUANT_BLOCK
    N_NOPE_BLOCKS: tl.constexpr = NOPE_HEAD_DIM // QUANT_BLOCK
    INV_FP8_MAX: tl.constexpr = 1.0 / FP8_MAX

    quant_input = normed.to(tl.bfloat16).to(tl.float32)
    quant_2d = tl.reshape(quant_input, (N_QUANT_BLOCKS, QUANT_BLOCK))
    block_absmax = tl.maximum(tl.max(tl.abs(quant_2d), axis=1), 1e-4)
    raw_scales = block_absmax * INV_FP8_MAX
    exponents = tl.ceil(tl.log2(raw_scales))
    inv_scales = tl.exp2(-exponents)
    x_scaled = quant_2d * tl.reshape(inv_scales, (N_QUANT_BLOCKS, 1))
    x_clamped = tl.clamp(x_scaled, -FP8_MAX, FP8_MAX)
    x_uint8 = tl.reshape(
        x_clamped.to(tl.float8e4nv).to(tl.uint8, bitcast=True),
        (TRITON_BLOCK_SIZE,),
    )
    tl.store(fp8_ptr + block, x_uint8, mask=block < NOPE_HEAD_DIM)

    scale_idx = tl.arange(0, N_QUANT_BLOCKS)
    max_encoded: tl.constexpr = 254.0 if SANITIZE_CACHE_NANS else 255.0
    encoded = tl.maximum(tl.minimum(exponents + 127.0, max_encoded), 0.0)
    tl.store(
        scale_ptr + scale_idx * SC_STEP,
        encoded.to(tl.uint8),
        mask=scale_idx < N_NOPE_BLOCKS,
    )
    if SC_STEP == 2:
        tl.store(
            scale_ptr + scale_idx * SC_STEP + 1,
            encoded.to(tl.uint8),
            mask=scale_idx < N_NOPE_BLOCKS,
        )
    else:
        tl.store(scale_ptr + N_NOPE_BLOCKS, tl.zeros((), dtype=tl.uint8))

    # GPT-J RoPE, full width. Splitting the pair into even and odd halves and
    # rebuilding a full-width value from them costs registers. The identity
    # used by _triton_kernels/rope/rope.py needs neither: for a pair (x0, x1)
    # the rotation is
    #
    #     out = x * cos + rot * sin,      rot = (-x1, x0)
    #
    # and rot is just a sign flip on the even lane followed by a flip of the
    # minor axis, so every value stays in its own lane. cos/sin are loaded per
    # LANE (index lane//2) rather than per pair, which reads each entry twice
    # from cache but removes the broadcast.
    NUM_PAIRS: tl.constexpr = TRITON_BLOCK_SIZE // 2
    NOPE_PAIRS: tl.constexpr = NOPE_HEAD_DIM // 2

    rope_pair_local = (block // 2) - NOPE_PAIRS
    is_rope_lane = rope_pair_local >= 0
    cs_idx = tl.maximum(rope_pair_local, 0)

    compressed_pos = (position // COMPRESS_RATIO) * COMPRESS_RATIO
    cache_base = cos_sin_cache_ptr + compressed_pos * cos_sin_stride
    cos_v = tl.load(cache_base + cs_idx, mask=is_rope_lane, other=1.0)
    sin_v = tl.load(cache_base + HALF_ROPE + cs_idx, mask=is_rope_lane, other=0.0)

    rot = tl.where(block % 2 == 0, normed, -normed)
    rot = tl.reshape(rot, (NUM_PAIRS, 2))
    rot = tl.flip(rot, 1)
    rot = tl.reshape(rot, (TRITON_BLOCK_SIZE,))
    result = normed * cos_v + rot * sin_v
    if SANITIZE_CACHE_NANS:
        result = tl.where(result == result, result, 0.0)
    bf16_ptr = (fp8_ptr + ROPE_IN_REC).to(tl.pointer_type(tl.bfloat16))
    rope_local = block - NOPE_HEAD_DIM
    is_rope = (block >= NOPE_HEAD_DIM) & mask
    tl.store(bf16_ptr + rope_local, result.to(tl.bfloat16), mask=is_rope)
