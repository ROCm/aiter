# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

# Q/K Gemma RMSNorm and interleaved MRoPE, and the K/V write into the paged KV cache
# (bf16, or fp8 with one per-tensor scale and Q emitted in fp8: row kernel only; or A8K4V8, NVFP4 K + that fp8
# V and Q: the row kernel, and fused_qk_norm_mrope_k4v8_gfx1250 for large prefills)

import functools

from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr

HEAD_DIM = gl.constexpr(256)
ROPE_HALF = gl.constexpr(32)  # rotary_dim 64: two Neox halves


@functools.cache
def get_fused_qk_norm_mrope_layouts(num_warps):
    # Rows over the waves: 8 values per lane for the norm, 1 value per lane for the RoPE halves and V.
    return {
        "row_layout": gl.BlockedLayout(
            [1, HEAD_DIM // 32], [1, 32], [num_warps, 1], [1, 0]
        ),
        "rope_layout": gl.BlockedLayout([1, 1], [1, 32], [num_warps, 1], [1, 0]),
    }


_fused_qk_norm_mrope_repr = make_kernel_repr(
    "fused_qk_norm_mrope_gfx1250", ["Q_HEADS", "K_HEADS", "BLOCK_M", "CACHE", "FP8", "K_NVFP4", "KV_LOC"]
)
_fused_qk_norm_mrope_tdm_repr = make_kernel_repr(
    "fused_qk_norm_mrope_tdm_gfx1250", ["Q_HEADS", "K_HEADS", "BLOCK_M", "CACHE"]
)


NVFP4_GROUP = gl.constexpr(16)  # values per e4m3 scale


@gluon.jit
def _nvfp4_quant(x, N: gl.constexpr, BLOCK_M: gl.constexpr):
    """x [BLOCK_M, N] fp32 -> (e2m1 pairs [BLOCK_M, N // 2] uint8, low nibble first; fp32 scales
    [BLOCK_M, N // 16]). The arithmetic of the Triton _nvfp4_quant_op (scale = amax / 6, x * (1 / scale),
    round-to-nearest-even into e2m1 with saturation at 6), in a layout that keeps each 16-value group
    in one thread."""
    G: gl.constexpr = N // NVFP4_GROUP
    L2: gl.constexpr = gl.BlockedLayout([1, 16], [1, 32], [gl.num_warps(), 1], [1, 0])
    x = gl.convert_layout(x, L2)
    x = gl.reshape(x, [BLOCK_M, G, NVFP4_GROUP])
    amax = gl.max(gl.abs(x), axis=2)
    scale = amax / 6.0
    qx = x * (1.0 / scale)[:, :, None]
    qx = qx.to(gl.uint32, bitcast=True)
    s = qx & 0x80000000
    qx = qx ^ s
    qf = qx.to(gl.float32, bitcast=True)
    denormal = (qf < 1.0) & (qf < 6.0)
    normal = (qf >= 1.0) & (qf < 6.0)
    # denormals: add 2^22 (exponent 149) and take the low bits back off
    den = (qf + 4194304.0).to(gl.uint32, bitcast=True) - 1249902592  # 149 << 23
    den = den.to(gl.uint8)
    # normals: rebias the exponent, round half to even, keep the top 3 bits of exponent + mantissa
    mant_odd = (qx >> 22) & 1
    nrm = ((qx - 1054867457 + mant_odd) >> 22).to(gl.uint8)  # (126 << 23) - ((1 << 21) - 1)
    e2m1 = gl.where(normal, nrm, gl.where(denormal, den, 7).to(gl.uint8))
    e2m1 = e2m1 | (s >> 28).to(gl.uint8)
    e2m1 = gl.reshape(e2m1, [BLOCK_M, G, NVFP4_GROUP // 2, 2])
    evens, odds = gl.split(e2m1)
    packed = evens | (odds << 4)
    return gl.reshape(packed, [BLOCK_M, N // 2]), scale


@gluon.jit
def _store_k_nvfp4(
    k_cache_ptr,
    x,
    page,
    scale_page,
    offset,
    write,
    COL0: gl.constexpr,
    N: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_SIZE: gl.constexpr,
    SCALE_PRESHUFFLE: gl.constexpr,
    store_layout: gl.constexpr,
):
    """NVFP4-quantize columns [COL0, COL0 + N) of BLOCK_M K rows (x: [BLOCK_M, N] fp32, COL0 and N
    multiples of 16, so whole scale groups) and store them into the shuffled NVFP4 K page of each row, as
    _store_nvfp4_shuffled_row lays it out: the HEAD_DIM // 2 e2m1 bytes of a token in (16, 16)-byte tiles,
    then the HEAD_DIM // 16 e4m3 scales preshuffled in SCALE_PRESHUFFLE-row groups. page / scale_page: the
    byte offsets of the row's (block, head) data and scales (page + BLOCK_SIZE * HEAD_DIM // 2 in a stock
    page); offset: the row's token in the page; write: rows to store. Rows are
    in SliceLayout(1, store_layout)."""
    D_BYTES: gl.constexpr = HEAD_DIM // 2
    D_SCALES: gl.constexpr = HEAD_DIM // NVFP4_GROUP
    P: gl.constexpr = SCALE_PRESHUFFLE
    P_QUARTER: gl.constexpr = SCALE_PRESHUFFLE // 4
    data, scales = _nvfp4_quant(x, N, BLOCK_M)
    data = gl.convert_layout(data, store_layout)
    scales = gl.convert_layout(scales, store_layout)
    j = COL0 // 2 + gl.arange(0, N // 2, layout=gl.SliceLayout(0, store_layout))
    data_dst = (
        page
        + (offset // 16) * (D_BYTES * 16)
        + (offset % 16) * 16
    )[:, None] + ((j // 16) * 256 + j % 16)[None, :]
    gl.store(k_cache_ptr + data_dst, data, mask=write[:, None])
    g = COL0 // NVFP4_GROUP + gl.arange(
        0, N // NVFP4_GROUP, layout=gl.SliceLayout(0, store_layout)
    )
    sub = offset % P
    scale_dst = (
        scale_page
        + (offset // P) * (D_SCALES * P)
        + (sub % P_QUARTER) * (4 * D_SCALES)
        + (sub // P_QUARTER) * D_SCALES
    )[:, None] + g[None, :]
    gl.store(
        k_cache_ptr + scale_dst,
        scales.to(gl.float8e4nv).to(gl.uint8, bitcast=True),
        mask=write[:, None],
    )


@gluon.jit
def _store_k_nvfp4_tail(
    k_cache_ptr,
    y,
    page,
    scale_page,
    offset,
    write,
    BLOCK_M: gl.constexpr,
    BLOCK_SIZE: gl.constexpr,
    SCALE_PRESHUFFLE: gl.constexpr,
    store_layout: gl.constexpr,
):
    """Columns [64, 256) of the K rows into their NVFP4 pages: the full row is quantized (its first
    four groups zero) and only bytes [32, 128) / scales [4, 16) are stored."""
    D_BYTES: gl.constexpr = HEAD_DIM // 2
    D_SCALES: gl.constexpr = HEAD_DIM // NVFP4_GROUP
    P: gl.constexpr = SCALE_PRESHUFFLE
    P_QUARTER: gl.constexpr = SCALE_PRESHUFFLE // 4
    data, scales = _nvfp4_quant(y, HEAD_DIM, BLOCK_M)
    data = gl.convert_layout(data, store_layout)
    scales = gl.convert_layout(scales, store_layout)
    j = gl.arange(0, D_BYTES, layout=gl.SliceLayout(0, store_layout))
    data_dst = (
        page + (offset // 16) * (D_BYTES * 16) + (offset % 16) * 16
    )[:, None] + ((j // 16) * 256 + j % 16)[None, :]
    gl.store(
        k_cache_ptr + data_dst, data, mask=write[:, None] & (j >= ROPE_HALF)[None, :]
    )
    g = gl.arange(0, D_SCALES, layout=gl.SliceLayout(0, store_layout))
    sub = offset % P
    scale_dst = (
        scale_page
        + (offset // P) * (D_SCALES * P)
        + (sub % P_QUARTER) * (4 * D_SCALES)
        + (sub // P_QUARTER) * D_SCALES
    )[:, None] + g[None, :]
    gl.store(
        k_cache_ptr + scale_dst,
        scales.to(gl.float8e4nv).to(gl.uint8, bitcast=True),
        mask=write[:, None] & (g >= 2 * ROPE_HALF // NVFP4_GROUP)[None, :],
    )


@gluon.jit
def _kv_loc_gran(page, x, u):
    # Byte offset of the u-th granule of a 2 MiB page homed on domain x (aiter.ops.kv_iod.gran).
    return ((page << 7) + (u << 1) + (x ^ ((u >> 2) & 1) ^ ((u >> 5) & 1))) * 16384


@gluon.jit
def _kv_loc_k(xk, block, h):
    """Placed K of (code block, head h), x bit xk (aiter.ops.kv_iod): the byte offsets of its 16 KiB data granule
    and of its 2 KiB scale row."""
    n = block >> 1
    pk = n // 28
    uk = 2 * (n % 28) + h
    page = _kv_loc_gran(pk, xk, uk)
    scale_page = _kv_loc_gran(pk, xk, 56 + uk // 8) + (uk % 8) * 2048
    return page, scale_page


@gluon.jit
def _kv_loc_v(xv, block, h, offset, X: gl.constexpr):
    """Placed V of the token at offset (of 128) in code block, head h, x bit xv: the byte offset of its value 0.
    V half offset // 64 of the page is its own granule."""
    n = block >> 1
    pv = n // 16
    uv = 4 * (n % 16) + 2 * h + offset // 64
    return _kv_loc_gran(pv, xv, uv) + ((offset // X) % 4) * (HEAD_DIM * X) + offset % X


@gluon.jit
def _tdm_tile(ptr, num_tokens, stride, BLOCK_M: gl.constexpr, COLS: gl.constexpr):
    # A [BLOCK_M, COLS] TDM tile of a [num_tokens, COLS] tensor, and its LDS buffer.
    layout: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, order=[1, 0])
    desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        ptr, [num_tokens, COLS], [stride, 1], [BLOCK_M, COLS], layout
    )
    smem = gl.allocate_shared_memory(ptr.dtype.element_ty, [BLOCK_M, COLS], layout)
    return desc, smem


@gluon.jit(repr=_fused_qk_norm_mrope_repr)
def fused_qk_norm_mrope_gfx1250(
    q_ptr,
    k_ptr,
    q_weight_ptr,
    k_weight_ptr,
    positions_ptr,
    cos_ptr,
    sin_ptr,
    q_out_ptr,
    k_out_ptr,
    num_rows,
    stride_q,
    stride_k,
    stride_positions,
    stride_cos,
    # The first 120 bytes of arguments arrive in registers. Later ones load at their
    # first use, so the arguments of the cache write come last.
    v_ptr,
    slot_mapping_ptr,
    stride_v,
    num_slots,
    k_cache_ptr,
    v_cache_ptr,
    kv_inv_scale,
    x_table_ptr,
    EPS: gl.constexpr,
    Q_HEADS: gl.constexpr,
    K_HEADS: gl.constexpr,
    SECTION_H: gl.constexpr,
    SECTION_W: gl.constexpr,
    BLOCK_M: gl.constexpr,
    CACHE: gl.constexpr,
    BLOCK_SIZE: gl.constexpr,
    FP8: gl.constexpr,
    FP8_MAX: gl.constexpr,
    row_layout: gl.constexpr,
    rope_layout: gl.constexpr,
    K_NVFP4: gl.constexpr = False,
    SCALE_PRESHUFFLE: gl.constexpr = 128,
    KV_LOC: gl.constexpr = False,
):
    # 16 for an fp8 cache and for A8K4V8's uint8 NVFP4 K (whose fp8 V has that layout)
    X: gl.constexpr = 128 // k_cache_ptr.dtype.element_ty.primitive_bitwidth
    # Row m of the T * HEADS rows is head m % HEADS of token m // HEADS, Q heads first.
    HEADS: gl.constexpr = Q_HEADS + K_HEADS
    pid_m = gl.program_id(0)
    offs_m = pid_m * BLOCK_M + gl.arange(
        0, BLOCK_M, layout=gl.SliceLayout(1, row_layout)
    )
    offs_n = gl.arange(0, HEAD_DIM, layout=gl.SliceLayout(0, row_layout))
    token = offs_m // HEADS
    head = offs_m % HEADS
    is_q = head < Q_HEADS
    mask_m = offs_m < num_rows
    offs_x = gl.where(
        is_q,
        token * stride_q + head * HEAD_DIM,
        token * stride_k + (head - Q_HEADS) * HEAD_DIM,
    )
    offs_out = gl.where(
        is_q,
        token * (Q_HEADS * HEAD_DIM) + head * HEAD_DIM,
        token * (K_HEADS * HEAD_DIM) + (head - Q_HEADS) * HEAD_DIM,
    )

    # The positions load first: the cos/sin loads wait for them. Then the row, its weight and the RoPE
    # halves. The values with an _r suffix hold the same rows in the RoPE layout.
    x_row = gl.where(is_q, q_ptr, k_ptr) + offs_x
    offs_r = gl.arange(0, ROPE_HALF, layout=gl.SliceLayout(0, rope_layout))
    token_r = gl.convert_layout(token, gl.SliceLayout(1, rope_layout))
    is_q_r = gl.convert_layout(is_q, gl.SliceLayout(1, rope_layout))
    mask_r = gl.convert_layout(mask_m, gl.SliceLayout(1, rope_layout))
    x_row_r = gl.convert_layout(x_row, gl.SliceLayout(1, rope_layout))
    pos_t = gl.load(positions_ptr + token_r, mask=mask_r, other=0)
    pos_h = gl.load(positions_ptr + stride_positions + token_r, mask=mask_r, other=0)
    pos_w = gl.load(
        positions_ptr + 2 * stride_positions + token_r, mask=mask_r, other=0
    )
    x = gl.load(x_row[:, None] + offs_n[None, :], mask=mask_m[:, None], other=0.0)
    w = gl.where(
        is_q[:, None],
        gl.load(q_weight_ptr + offs_n)[None, :],
        gl.load(k_weight_ptr + offs_n)[None, :],
    )

    # The RoPE halves: columns [0, 32) and [32, 64).
    x_lo = gl.load(x_row_r[:, None] + offs_r[None, :], mask=mask_r[:, None], other=0.0)
    x_hi = gl.load(
        x_row_r[:, None] + ROPE_HALF + offs_r[None, :], mask=mask_r[:, None], other=0.0
    )
    head_r = gl.convert_layout(head, gl.SliceLayout(1, rope_layout))
    if CACHE:
        # The slot and the V row of each K row. Tokens past the slot table are padding.
        # Lane l holds V values l, l + 32, ...: the cache keeps the V values of a token 16 bytes apart.
        is_k_r = mask_r & (head_r >= Q_HEADS) & (token_r < num_slots)
        slot = gl.load(slot_mapping_ptr + token_r, mask=is_k_r, other=-1).to(gl.int64)
        offs_v = gl.arange(0, HEAD_DIM, layout=gl.SliceLayout(0, rope_layout))
        v_row = v_ptr + token_r * stride_v + (head_r - Q_HEADS) * HEAD_DIM
        v = gl.load(v_row[:, None] + offs_v[None, :], mask=is_k_r[:, None], other=0.0)
    w_lo = gl.where(
        is_q_r[:, None],
        gl.load(q_weight_ptr + offs_r)[None, :],
        gl.load(k_weight_ptr + offs_r)[None, :],
    )
    w_hi = gl.where(
        is_q_r[:, None],
        gl.load(q_weight_ptr + ROPE_HALF + offs_r)[None, :],
        gl.load(k_weight_ptr + ROPE_HALF + offs_r)[None, :],
    )

    # Frequency i uses the H position, the W position or the T position (interleaved sections).
    use_h = (offs_r % 3 == 1) & (offs_r < 3 * SECTION_H)
    use_w = (offs_r % 3 == 2) & (offs_r < 3 * SECTION_W)
    pos = gl.where(
        use_h[None, :],
        pos_h[:, None],
        gl.where(use_w[None, :], pos_w[:, None], pos_t[:, None]),
    )
    # The mask keeps these loads before the norm: the compiler moves an unmasked load next to its use.
    cos = gl.load(
        cos_ptr + pos * stride_cos + offs_r[None, :], mask=mask_r[:, None], other=0.0
    )
    sin = gl.load(
        sin_ptr + pos * stride_cos + offs_r[None, :], mask=mask_r[:, None], other=0.0
    )

    # Gemma RMSNorm: x / rms(x) * (1 + w). Columns [64, 256) are done. K keeps the input
    # dtype; so does Q unless it goes out in fp8.
    x = x.to(gl.float32)
    rstd = gl.rsqrt(gl.sum(x * x, axis=1) / HEAD_DIM + EPS)
    y = (x * rstd[:, None] * (w.to(gl.float32) + 1.0)).to(k_out_ptr.dtype.element_ty)
    tail = (offs_n >= 2 * ROPE_HALF)[None, :]

    # Neox RoPE of columns [0, 64), from the same normalized values.
    rstd_r = gl.convert_layout(rstd, gl.SliceLayout(1, rope_layout))[:, None]
    y_lo = x_lo.to(gl.float32) * rstd_r * (w_lo.to(gl.float32) + 1.0)
    y_hi = x_hi.to(gl.float32) * rstd_r * (w_hi.to(gl.float32) + 1.0)
    rope_lo = (y_lo * cos - y_hi * sin).to(k_out_ptr.dtype.element_ty)
    rope_hi = (y_hi * cos + y_lo * sin).to(k_out_ptr.dtype.element_ty)

    if FP8:
        # Q goes out in fp8 and K stays bf16, so they need separate stores. Q is the bf16
        # value clamped and cast, as on the unfused path: the fp8 cast does not saturate.
        fp8_ty = q_out_ptr.dtype.element_ty
        q_rows = mask_m & is_q
        k_rows = mask_m & (head >= Q_HEADS)
        q_rows_r = mask_r & is_q_r
        k_rows_r = mask_r & (head_r >= Q_HEADS)
        out_n = offs_out[:, None] + offs_n[None, :]
        q_n = gl.minimum(gl.maximum(y.to(gl.float32), -FP8_MAX), FP8_MAX)
        gl.store(q_out_ptr + out_n, q_n.to(fp8_ty), mask=q_rows[:, None] & tail)
        gl.store(k_out_ptr + out_n, y, mask=k_rows[:, None] & tail)
        offs_out_r = gl.convert_layout(offs_out, gl.SliceLayout(1, rope_layout))
        out_r = offs_out_r[:, None] + offs_r[None, :]
        q_lo = gl.minimum(gl.maximum(rope_lo.to(gl.float32), -FP8_MAX), FP8_MAX)
        q_hi = gl.minimum(gl.maximum(rope_hi.to(gl.float32), -FP8_MAX), FP8_MAX)
        gl.store(q_out_ptr + out_r, q_lo.to(fp8_ty), mask=q_rows_r[:, None])
        gl.store(q_out_ptr + ROPE_HALF + out_r, q_hi.to(fp8_ty), mask=q_rows_r[:, None])
        gl.store(k_out_ptr + out_r, rope_lo, mask=k_rows_r[:, None])
        gl.store(k_out_ptr + ROPE_HALF + out_r, rope_hi, mask=k_rows_r[:, None])
    else:
        out_row = gl.where(is_q, q_out_ptr, k_out_ptr) + offs_out
        gl.store(out_row[:, None] + offs_n[None, :], y, mask=mask_m[:, None] & tail)
        out_row_r = gl.convert_layout(out_row, gl.SliceLayout(1, rope_layout))
        gl.store(out_row_r[:, None] + offs_r[None, :], rope_lo, mask=mask_r[:, None])
        gl.store(
            out_row_r[:, None] + ROPE_HALF + offs_r[None, :],
            rope_hi,
            mask=mask_r[:, None],
        )

    if CACHE:
        # K[block, head, d // X, offset, d % X] and V[block, head, offset // X, d, offset % X].
        # A masked slot load gives -1, a pad slot.
        write = (slot >= 0)[:, None]
        block = slot // BLOCK_SIZE
        offset = slot % BLOCK_SIZE
        cache_row = (block * K_HEADS + head_r - Q_HEADS) * (HEAD_DIM * BLOCK_SIZE)
        v_dst = cache_row + (offset // X) * (HEAD_DIM * X) + offset % X
        if K_NVFP4:
            # A8K4V8: V as an fp8 cache (one per-tensor scale), K NVFP4-quantized from the bf16 roped / normed
            # values (no per-tensor scale: k_descale = 1); columns [0, 32), [32, 64) from the RoPE layout and
            # [64, 256) from the norm layout, each a whole set of 16-value scale groups. A CTA holds BLOCK_M rows,
            # mostly Q heads: only one with a K row to write quantizes and stores.
            write_k = slot >= 0
            if gl.max(write_k.to(gl.int32), axis=0) > 0:
                v_c = gl.minimum(gl.maximum(v.to(gl.float32) * kv_inv_scale, -FP8_MAX), FP8_MAX)
                v_c = v_c.to(v_cache_ptr.dtype.element_ty)
                K_ROW: gl.constexpr = HEAD_DIM // 2 + HEAD_DIM // NVFP4_GROUP
                if KV_LOC:
                    # Placed granules (aiter.ops.kv_iod): block id = code c, x bits from the layer's table.
                    # Block 128 and 2 heads; V half offset // 64 of the page is its own granule.
                    xb = gl.load(x_table_ptr + block, mask=write_k, other=0).to(gl.int64)
                    h = (head_r - Q_HEADS).to(gl.int64)
                    page, scale_page = _kv_loc_k(xb & 1, block, h)
                    v_dst = _kv_loc_v((xb >> 1) & 1, block, h, offset, X)
                else:
                    page = (block * K_HEADS + head_r - Q_HEADS) * (BLOCK_SIZE * K_ROW)
                    scale_page = page + BLOCK_SIZE * (HEAD_DIM // 2)
                gl.store(v_cache_ptr + v_dst[:, None] + offs_v[None, :] * X, v_c, mask=write)
                _store_k_nvfp4(
                    k_cache_ptr, rope_lo.to(gl.float32), page, scale_page, offset, write_k, 0, ROPE_HALF,
                    BLOCK_M, BLOCK_SIZE, SCALE_PRESHUFFLE, rope_layout,
                )
                _store_k_nvfp4(
                    k_cache_ptr, rope_hi.to(gl.float32), page, scale_page, offset, write_k, ROPE_HALF, ROPE_HALF,
                    BLOCK_M, BLOCK_SIZE, SCALE_PRESHUFFLE, rope_layout,
                )
                _store_k_nvfp4_tail(
                    k_cache_ptr, gl.where(tail, y.to(gl.float32), 0.0), page, scale_page, offset, write_k,
                    BLOCK_M, BLOCK_SIZE, SCALE_PRESHUFFLE, rope_layout,
                )
        else:
            if FP8:
                # One per-tensor scale, as reshape_and_cache(kv_cache_dtype="fp8") applies it.
                cache_ty = k_cache_ptr.dtype.element_ty
                v_c = gl.minimum(gl.maximum(v.to(gl.float32) * kv_inv_scale, -FP8_MAX), FP8_MAX)
                kc_lo = gl.minimum(gl.maximum(rope_lo.to(gl.float32) * kv_inv_scale, -FP8_MAX), FP8_MAX)
                kc_hi = gl.minimum(gl.maximum(rope_hi.to(gl.float32) * kv_inv_scale, -FP8_MAX), FP8_MAX)
                kc_n = gl.minimum(gl.maximum(y.to(gl.float32) * kv_inv_scale, -FP8_MAX), FP8_MAX)
                v_c, kc_lo, kc_hi, kc_n = (
                    v_c.to(cache_ty),
                    kc_lo.to(cache_ty),
                    kc_hi.to(cache_ty),
                    kc_n.to(cache_ty),
                )
            else:
                v_c, kc_lo, kc_hi, kc_n = v, rope_lo, rope_hi, y
            gl.store(v_cache_ptr + v_dst[:, None] + offs_v[None, :] * X, v_c, mask=write)
            # ROPE_HALF is a multiple of X: the high RoPE half is a constant step after the low one.
            k_dst = cache_row + offset * X
            k_lo = k_dst[:, None] + (offs_r // X * BLOCK_SIZE * X + offs_r % X)[None, :]
            gl.store(k_cache_ptr + k_lo, kc_lo, mask=write)
            gl.store(k_cache_ptr + k_lo + ROPE_HALF * BLOCK_SIZE, kc_hi, mask=write)
            # Columns [64, 256) from the norm layout, 16 bytes per lane.
            k_dst_n = gl.convert_layout(k_dst, gl.SliceLayout(1, row_layout))
            write_n = gl.convert_layout(slot >= 0, gl.SliceLayout(1, row_layout))
            k_n = k_dst_n[:, None] + (offs_n // X * BLOCK_SIZE * X + offs_n % X)[None, :]
            mask_n = write_n[:, None] & tail
            gl.store(k_cache_ptr + k_n, kc_n, mask=mask_n)


@gluon.jit
def _norm_rope_tile(
    x_s,
    out_s,
    w_ptr,
    t0,
    num_tokens,
    positions_ptr,
    stride_positions,
    cos_ptr,
    sin_ptr,
    stride_cos,
    EPS: gl.constexpr,
    HEADS: gl.constexpr,
    SECTION_H: gl.constexpr,
    SECTION_W: gl.constexpr,
    WAIT: gl.constexpr,
    row_layout: gl.constexpr,
    rope_layout: gl.constexpr,
):
    # Row m of the tile is head m % HEADS of token t0 + m // HEADS.
    offs_m = gl.arange(0, x_s.shape[0], layout=gl.SliceLayout(1, row_layout))
    offs_n = gl.arange(0, HEAD_DIM, layout=gl.SliceLayout(0, row_layout))
    mask_m = t0 + offs_m // HEADS < num_tokens

    # The positions, cos/sin and weights load while the tile is in flight.
    offs_r = gl.arange(0, ROPE_HALF, layout=gl.SliceLayout(0, rope_layout))
    token_r = gl.convert_layout(t0 + offs_m // HEADS, gl.SliceLayout(1, rope_layout))
    mask_r = gl.convert_layout(mask_m, gl.SliceLayout(1, rope_layout))
    pos_t = gl.load(positions_ptr + token_r, mask=mask_r, other=0)
    pos_h = gl.load(positions_ptr + stride_positions + token_r, mask=mask_r, other=0)
    pos_w = gl.load(
        positions_ptr + 2 * stride_positions + token_r, mask=mask_r, other=0
    )
    use_h = (offs_r % 3 == 1) & (offs_r < 3 * SECTION_H)
    use_w = (offs_r % 3 == 2) & (offs_r < 3 * SECTION_W)
    pos = gl.where(
        use_h[None, :],
        pos_h[:, None],
        gl.where(use_w[None, :], pos_w[:, None], pos_t[:, None]),
    )
    cos = gl.load(
        cos_ptr + pos * stride_cos + offs_r[None, :], mask=mask_r[:, None], other=0.0
    )
    sin = gl.load(
        sin_ptr + pos * stride_cos + offs_r[None, :], mask=mask_r[:, None], other=0.0
    )
    w = gl.load(w_ptr + offs_n)[None, :].to(gl.float32)
    w_lo = gl.load(w_ptr + offs_r)[None, :].to(gl.float32)
    w_hi = gl.load(w_ptr + ROPE_HALF + offs_r)[None, :].to(gl.float32)

    # Gemma RMSNorm: x / rms(x) * (1 + w). All columns go to LDS, then the RoPE values
    # replace columns [0, 64) after the barrier.
    gl.amd.gfx1250.tdm.async_wait(WAIT)
    x = x_s.load(row_layout).to(gl.float32)
    rstd = gl.rsqrt(gl.sum(x * x, axis=1) / HEAD_DIM + EPS)
    out_s.store((x * rstd[:, None] * (w + 1.0)).to(out_s.dtype))
    gl.barrier()

    # Neox RoPE of columns [0, 64), from the same normalized values.
    x_lo = x_s.slice(0, ROPE_HALF, dim=1).load(rope_layout).to(gl.float32)
    x_hi = x_s.slice(ROPE_HALF, ROPE_HALF, dim=1).load(rope_layout).to(gl.float32)
    rstd_r = gl.convert_layout(rstd, gl.SliceLayout(1, rope_layout))[:, None]
    y_lo = x_lo * rstd_r * (w_lo + 1.0)
    y_hi = x_hi * rstd_r * (w_hi + 1.0)
    out_s.slice(0, ROPE_HALF, dim=1).store((y_lo * cos - y_hi * sin).to(out_s.dtype))
    out_s.slice(ROPE_HALF, ROPE_HALF, dim=1).store(
        (y_hi * cos + y_lo * sin).to(out_s.dtype)
    )


@gluon.jit(repr=_fused_qk_norm_mrope_tdm_repr)
def fused_qk_norm_mrope_tdm_gfx1250(
    q_ptr,
    k_ptr,
    q_weight_ptr,
    k_weight_ptr,
    positions_ptr,
    cos_ptr,
    sin_ptr,
    q_out_ptr,
    k_out_ptr,
    num_tokens,
    stride_q,
    stride_k,
    stride_positions,
    stride_cos,
    # The arguments of the cache write come last, as in the row kernel.
    v_ptr,
    slot_mapping_ptr,
    stride_v,
    num_slots,
    k_cache_ptr,
    v_cache_ptr,
    EPS: gl.constexpr,
    Q_HEADS: gl.constexpr,
    K_HEADS: gl.constexpr,
    SECTION_H: gl.constexpr,
    SECTION_W: gl.constexpr,
    BLOCK_M: gl.constexpr,
    CACHE: gl.constexpr,
    BLOCK_SIZE: gl.constexpr,
    row_layout: gl.constexpr,
    rope_layout: gl.constexpr,
):
    # A program does BLOCK_M tokens. Their Q heads and their K heads move between memory and
    # LDS as one TDM tile each.
    Q_COLS: gl.constexpr = Q_HEADS * HEAD_DIM
    K_COLS: gl.constexpr = K_HEADS * HEAD_DIM
    X: gl.constexpr = 128 // k_cache_ptr.dtype.element_ty.primitive_bitwidth
    t0 = gl.program_id(0) * BLOCK_M
    q_desc, q_s = _tdm_tile(q_ptr, num_tokens, stride_q, BLOCK_M, Q_COLS)
    k_desc, k_s = _tdm_tile(k_ptr, num_tokens, stride_k, BLOCK_M, K_COLS)
    q_out_desc, q_out_s = _tdm_tile(q_out_ptr, num_tokens, Q_COLS, BLOCK_M, Q_COLS)
    k_out_desc, k_out_s = _tdm_tile(k_out_ptr, num_tokens, K_COLS, BLOCK_M, K_COLS)
    gl.amd.gfx1250.tdm.async_load(q_desc, [t0, 0], q_s)
    gl.amd.gfx1250.tdm.async_load(k_desc, [t0, 0], k_s)
    if CACHE:
        # The V tile, and the slot of each K row. Tokens past the slot table are padding.
        v_desc, v_s = _tdm_tile(v_ptr, num_tokens, stride_v, BLOCK_M, K_COLS)
        gl.amd.gfx1250.tdm.async_load(v_desc, [t0, 0], v_s)
        offs_k = gl.arange(0, BLOCK_M * K_HEADS, layout=gl.SliceLayout(1, row_layout))
        token_k = t0 + offs_k // K_HEADS
        in_table = (token_k < num_tokens) & (token_k < num_slots)
        slot = gl.load(slot_mapping_ptr + token_k, mask=in_table, other=-1).to(gl.int64)
    # Each tile waits for its own TDM load only: the V tile can still be in flight.
    _norm_rope_tile(
        q_s.reshape([BLOCK_M * Q_HEADS, HEAD_DIM]),
        q_out_s.reshape([BLOCK_M * Q_HEADS, HEAD_DIM]),
        q_weight_ptr,
        t0,
        num_tokens,
        positions_ptr,
        stride_positions,
        cos_ptr,
        sin_ptr,
        stride_cos,
        EPS,
        Q_HEADS,
        SECTION_H,
        SECTION_W,
        1 + CACHE,
        row_layout,
        rope_layout,
    )
    _norm_rope_tile(
        k_s.reshape([BLOCK_M * K_HEADS, HEAD_DIM]),
        k_out_s.reshape([BLOCK_M * K_HEADS, HEAD_DIM]),
        k_weight_ptr,
        t0,
        num_tokens,
        positions_ptr,
        stride_positions,
        cos_ptr,
        sin_ptr,
        stride_cos,
        EPS,
        K_HEADS,
        SECTION_H,
        SECTION_W,
        CACHE * 1,
        row_layout,
        rope_layout,
    )
    gl.barrier()
    gl.amd.gfx1250.tdm.async_store(q_out_desc, [t0, 0], q_out_s)
    gl.amd.gfx1250.tdm.async_store(k_out_desc, [t0, 0], k_out_s)
    if CACHE:
        # K from the K tile in LDS, 16 bytes a lane, and V one value a lane, after the V tile is in.
        gl.amd.gfx1250.tdm.async_wait(2)
        offs_n = gl.arange(0, HEAD_DIM, layout=gl.SliceLayout(0, row_layout))
        offs_v = gl.arange(0, HEAD_DIM, layout=gl.SliceLayout(0, rope_layout))
        block = slot // BLOCK_SIZE
        offset = slot % BLOCK_SIZE
        cache_row = (block * K_HEADS + offs_k % K_HEADS) * (HEAD_DIM * BLOCK_SIZE)
        k_dst = cache_row + offset * X
        k_n = k_dst[:, None] + (offs_n // X * BLOCK_SIZE * X + offs_n % X)[None, :]
        k_rows = k_out_s.reshape([BLOCK_M * K_HEADS, HEAD_DIM]).load(row_layout)
        gl.store(k_cache_ptr + k_n, k_rows, mask=(slot >= 0)[:, None])
        v_dst = cache_row + (offset // X) * (HEAD_DIM * X) + offset % X
        v_dst = gl.convert_layout(v_dst, gl.SliceLayout(1, rope_layout))
        write_v = gl.convert_layout(slot >= 0, gl.SliceLayout(1, rope_layout))[:, None]
        v_rows = v_s.reshape([BLOCK_M * K_HEADS, HEAD_DIM]).load(rope_layout)
        gl.store(
            v_cache_ptr + v_dst[:, None] + offs_v[None, :] * X, v_rows, mask=write_v
        )
    gl.amd.gfx1250.tdm.async_wait(0)


# ----------------------------------------------------------------------------------------------------------------
# A8K4V8 prefill: Q, K and V in separate CTAs (fused_qk_norm_mrope_k4v8_gfx1250)
# ----------------------------------------------------------------------------------------------------------------

SUB = gl.constexpr(16)  # tokens a block: one 16-row tile of the shuffled NVFP4 data / of the fp8 V page

_k4v8_repr = make_kernel_repr("fused_qk_norm_mrope_k4v8_gfx1250", ["Q_HEADS", "K_HEADS", "BLOCK_SIZE", "KV_LOC"])


@gluon.jit
def _issue_tables(r0, positions_ptr, stride_positions, cos_desc, sin_desc, tab_smem, n_tok, R: gl.constexpr,
                  idx_layout: gl.constexpr):
    """TDM-gather the cos / sin rows of the T, H and W positions of the R tokens from r0 into tab_smem
    [6, R, 32] (cos T/H/W, sin T/H/W), or only the T rows when every token's three positions agree (text).
    Returns whether they agreed."""
    t = r0 + gl.arange(0, R, layout=idx_layout)
    m = t < n_tok
    pos_t = gl.load(positions_ptr + t, mask=m, other=0).to(gl.int32)
    pos_h = gl.load(positions_ptr + stride_positions + t, mask=m, other=0).to(gl.int32)
    pos_w = gl.load(positions_ptr + 2 * stride_positions + t, mask=m, other=0).to(gl.int32)
    same = gl.max(((pos_h != pos_t) | (pos_w != pos_t)).to(gl.int32), axis=0) == 0
    gl.amd.gfx1250.tdm.async_gather(cos_desc, pos_t, tab_smem.index(0))
    gl.amd.gfx1250.tdm.async_gather(sin_desc, pos_t, tab_smem.index(3))
    if not same:
        gl.amd.gfx1250.tdm.async_gather(cos_desc, pos_h, tab_smem.index(1))
        gl.amd.gfx1250.tdm.async_gather(sin_desc, pos_h, tab_smem.index(4))
        gl.amd.gfx1250.tdm.async_gather(cos_desc, pos_w, tab_smem.index(2))
        gl.amd.gfx1250.tdm.async_gather(sin_desc, pos_w, tab_smem.index(5))
    return same


@gluon.jit
def _rope_from_smem(tab_smem, same, SECTION_H: gl.constexpr, SECTION_W: gl.constexpr, rope_layout: gl.constexpr):
    """cos / sin [R, 32] of the interleaved MRoPE: column i takes the H position's row when i % 3 == 1 and
    i < 3 SECTION_H, the W position's when i % 3 == 2 and i < 3 SECTION_W, else the T position's (all the T rows
    when the positions agree)."""
    offs_r = gl.arange(0, ROPE_HALF, layout=gl.SliceLayout(0, rope_layout))
    use_h = ((offs_r % 3 == 1) & (offs_r < 3 * SECTION_H))[None, :]
    use_w = ((offs_r % 3 == 2) & (offs_r < 3 * SECTION_W))[None, :]
    cos = tab_smem.index(0).load(rope_layout)
    sin = tab_smem.index(3).load(rope_layout)
    if not same:
        cos = gl.where(use_h, tab_smem.index(1).load(rope_layout), gl.where(use_w, tab_smem.index(2).load(rope_layout), cos))
        sin = gl.where(use_h, tab_smem.index(4).load(rope_layout), gl.where(use_w, tab_smem.index(5).load(rope_layout), sin))
    return cos, sin


@gluon.jit
def _norm_mrope_smem(x_smem, w_ptr, cos, sin, EPS, row_layout: gl.constexpr, rope_layout: gl.constexpr):
    """The row kernel's Gemma RMSNorm + Neox RoPE of [0, 64), same fp32 arithmetic and reduction layout, on the
    bf16 rows [16, 256] in x_smem. Returns y [16, 256] bf16 (columns [64, 256) final) and rope_lo / rope_hi
    [16, 32] bf16."""
    offs_n = gl.arange(0, HEAD_DIM, layout=gl.SliceLayout(0, row_layout))
    offs_r = gl.arange(0, ROPE_HALF, layout=gl.SliceLayout(0, rope_layout))
    x = x_smem.load(row_layout).to(gl.float32)
    x_lo = x_smem.slice(0, ROPE_HALF, dim=1).load(rope_layout).to(gl.float32)
    x_hi = x_smem.slice(ROPE_HALF, ROPE_HALF, dim=1).load(rope_layout).to(gl.float32)
    w = gl.load(w_ptr + offs_n)[None, :]
    w_lo = gl.load(w_ptr + offs_r)[None, :]
    w_hi = gl.load(w_ptr + ROPE_HALF + offs_r)[None, :]
    rstd = gl.rsqrt(gl.sum(x * x, axis=1) / HEAD_DIM + EPS)
    y = (x * rstd[:, None] * (w.to(gl.float32) + 1.0)).to(gl.bfloat16)
    rstd_r = gl.convert_layout(rstd, gl.SliceLayout(1, rope_layout))[:, None]
    y_lo = x_lo * rstd_r * (w_lo.to(gl.float32) + 1.0)
    y_hi = x_hi * rstd_r * (w_hi.to(gl.float32) + 1.0)
    rope_lo = (y_lo * cos - y_hi * sin).to(gl.bfloat16)
    rope_hi = (y_hi * cos + y_lo * sin).to(gl.bfloat16)
    return y, rope_lo, rope_hi


@gluon.jit
def _store_rows(out_ptr, tok, tok_r, mask, mask_r, row_stride, y, lo, hi,
                row_layout: gl.constexpr, rope_layout: gl.constexpr):
    """Rows of an output [T, heads * 256] at out_ptr (head offset applied): [64, 256) from y, [0, 64) from lo / hi."""
    offs_n = gl.arange(0, HEAD_DIM, layout=gl.SliceLayout(0, row_layout))
    offs_r = gl.arange(0, ROPE_HALF, layout=gl.SliceLayout(0, rope_layout))
    tail = (offs_n >= 2 * ROPE_HALF)[None, :]
    row = out_ptr + tok.to(gl.int64) * row_stride
    row_r = out_ptr + tok_r.to(gl.int64) * row_stride
    gl.store(row[:, None] + offs_n[None, :], y, mask=mask[:, None] & tail)
    gl.store(row_r[:, None] + offs_r[None, :], lo, mask=mask_r[:, None])
    gl.store(row_r[:, None] + ROPE_HALF + offs_r[None, :], hi, mask=mask_r[:, None])


@gluon.jit
def _tile_seg_start(slot_mapping_ptr, t, n_tok, n_valid):
    """Whether token t starts a tile segment (a maximal run of tokens in consecutive slots of one 16-row tile of a
    page), and its slot. Pad tokens (slot < 0) and tokens past the slot table are segments of their own."""
    cur = gl.load(slot_mapping_ptr + t, mask=(t < n_valid) & (t >= 0), other=-1).to(gl.int64)
    prev = gl.load(slot_mapping_ptr + t - 1, mask=(t - 1 < n_valid) & (t >= 1), other=-1).to(gl.int64)
    cont = (t >= 1) & (cur >= 0) & (prev >= 0) & (cur == prev + 1) & (cur % SUB != 0)
    return (t < n_tok) & ~cont, cur


@gluon.jit(repr=_k4v8_repr)
def fused_qk_norm_mrope_k4v8_gfx1250(
    q_ptr,
    k_ptr,
    v_ptr,
    q_weight_ptr,
    k_weight_ptr,
    positions_ptr,
    cos_ptr,
    sin_ptr,
    q_out_ptr,
    k_out_ptr,
    slot_mapping_ptr,
    k_cache_ptr,
    v_cache_ptr,
    num_tokens,
    num_slots,
    num_pos,
    stride_q,
    stride_k,
    stride_v,
    stride_positions,
    stride_cos,
    kv_inv_scale,
    x_table_ptr,
    EPS: gl.constexpr,
    Q_HEADS: gl.constexpr,
    K_HEADS: gl.constexpr,
    SECTION_H: gl.constexpr,
    SECTION_W: gl.constexpr,
    BLOCK_SIZE: gl.constexpr,
    SCALE_PRESHUFFLE: gl.constexpr,
    FP8_MAX: gl.constexpr,
    QH_PER: gl.constexpr,
    BLOCK_M_Q: gl.constexpr,
    BLOCK_M_V: gl.constexpr,
    KV_LOC: gl.constexpr = False,
):
    """A8K4V8 Q/K norm + MRoPE and KV-cache write of a prefill batch, byte-identical to
    fused_qk_norm_mrope_gfx1250(CACHE, FP8, K_NVFP4). Grid = K + Q + V blocks, in that program-id order, 16 tokens
    a block, one row a wave.
    - K: cdiv(T, 16) CTAs, each for every K head. CTA w owns the tile segments starting in token window [16 w,
      16 w + 16) (segments as in _tile_seg_start, at most 16 tokens): rows [a, b), a / b = the first start at or
      after 16 w / 16 w + 16. Its last start f is a full tile (16 tokens in slots 16 i .. 16 i + 15) when
      slot[f] % 16 == 0 and b - f == 16: those rows are TDM-loaded, normed, roped, and their 2 KB NVFP4 data block
      and [16, 16] scale box built in LDS and TDM-stored. The other rows (at most 31: a sequence's mid-tile head
      and tail, short sequences, pads) the same, stored row by row. k_out for every row the CTA owns.
    - Q: (Q_HEADS / QH_PER) * cdiv(T, BLOCK_M_Q) CTAs, BLOCK_M_Q tokens of QH_PER Q heads, Q out in fp8.
    - V: K_HEADS * cdiv(T, BLOCK_M_V) CTAs, V / kv_scale in fp8 into the shuffled V cache.
    Rows come in with one TDM box each, the MRoPE cos / sin rows with TDM gathers into LDS, once a CTA.
    KV_LOC: k_cache_ptr / v_cache_ptr are one layer's placed K / V regions (aiter.ops.kv_iod), the block id is a
    code and x_table_ptr its x bytes; the same bytes go to the placed addresses of each (block, head) page."""
    NW: gl.constexpr = gl.num_warps()
    P: gl.constexpr = BLOCK_SIZE
    D_BYTES: gl.constexpr = HEAD_DIM // 2
    D_SCALES: gl.constexpr = HEAD_DIM // NVFP4_GROUP
    K_ROW: gl.constexpr = D_BYTES + D_SCALES
    PQ: gl.constexpr = SCALE_PRESHUFFLE // 4
    row_layout: gl.constexpr = gl.BlockedLayout([1, HEAD_DIM // 32], [1, 32], [NW, 1], [1, 0])
    rope_layout: gl.constexpr = gl.BlockedLayout([1, 1], [1, 32], [NW, 1], [1, 0])
    idx_layout: gl.constexpr = gl.BlockedLayout([SUB], [32], [NW], [0])
    SH: gl.constexpr = gl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[1, 0])
    SH3: gl.constexpr = gl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[2, 1, 0])
    n_valid = gl.minimum(num_tokens, num_slots)
    n_blk = (num_tokens + SUB - 1) // SUB
    num_k_blocks = n_blk
    num_q_blocks = (Q_HEADS // QH_PER) * ((num_tokens + BLOCK_M_Q - 1) // BLOCK_M_Q)
    pid = gl.program_id(0)
    cos_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=cos_ptr, shape=(num_pos, ROPE_HALF), strides=(stride_cos, 1), block_shape=(SUB, ROPE_HALF), layout=SH,
    )
    sin_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=sin_ptr, shape=(num_pos, ROPE_HALF), strides=(stride_cos, 1), block_shape=(SUB, ROPE_HALF), layout=SH,
    )
    sub_n = gl.arange(0, SUB, layout=gl.SliceLayout(1, row_layout))
    sub_r = gl.arange(0, SUB, layout=gl.SliceLayout(1, rope_layout))

    if pid < num_k_blocks:
        # ---- K: the tile segments starting in token window [16 w, 16 w + 16), every K head
        w = pid
        win = w * SUB + sub_r
        st0, slot0 = _tile_seg_start(slot_mapping_ptr, win, num_tokens, n_valid)
        st1, _ = _tile_seg_start(slot_mapping_ptr, win + SUB, num_tokens, n_valid)
        a = gl.min(gl.where(st0, win, num_tokens), axis=0)
        b = gl.min(gl.where(st1, win + SUB, num_tokens), axis=0)
        f = gl.max(gl.where(st0, win, -1), axis=0)
        slot_f = gl.max(gl.where(win == f, slot0, -1), axis=0)
        full = (f >= 0) & (slot_f >= 0) & (slot_f % SUB == 0) & (b - f == SUB) & (f + SUB <= n_valid)
        end = gl.where(full, f, b)
        k_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
            base=k_ptr, shape=(num_tokens, K_HEADS * HEAD_DIM), strides=(stride_k, 1),
            block_shape=(SUB, HEAD_DIM), layout=SH,
        )
        x_smem = gl.allocate_shared_memory(k_ptr.dtype.element_ty, [K_HEADS, SUB, HEAD_DIM], SH)
        tab_smem = gl.allocate_shared_memory(cos_ptr.dtype.element_ty, [6, SUB, ROPE_HALF], SH)
        d_smem = gl.allocate_shared_memory(gl.uint8, [D_BYTES // 16, SUB, 16], SH3)
        s_smem = gl.allocate_shared_memory(gl.uint8, [SUB, D_SCALES], SH)
        offs_n = gl.arange(0, HEAD_DIM, layout=gl.SliceLayout(0, row_layout))
        tail = (offs_n >= 2 * ROPE_HALF)[None, :]
        # the rows [a, end) that are not a full tile: row by row, in at most two blocks
        for c in gl.static_range(2):
            r0 = a + c * SUB
            if r0 < end:
                gl.barrier()  # every wave is done with the buffers this block overwrites
                for hh in gl.static_range(K_HEADS):
                    gl.amd.gfx1250.tdm.async_load(k_desc, [r0, hh * HEAD_DIM], x_smem.index(hh))
                same = _issue_tables(r0, positions_ptr, stride_positions, cos_desc, sin_desc, tab_smem, num_tokens,
                                     SUB, idx_layout)
                tok = r0 + sub_n
                tok_r = r0 + sub_r
                own = tok < end
                own_r = tok_r < end
                slot = gl.load(slot_mapping_ptr + tok_r, mask=own_r & (tok_r < n_valid), other=-1).to(gl.int64)
                gl.amd.gfx1250.tdm.async_wait(0)
                gl.barrier()
                cos, sin = _rope_from_smem(tab_smem, same, SECTION_H, SECTION_W, rope_layout)
                write = own_r & (slot >= 0)
                if KV_LOC:
                    xk_r = gl.load(x_table_ptr + slot // P, mask=write, other=0).to(gl.int64) & 1
                for h in gl.static_range(K_HEADS):
                    y, rope_lo, rope_hi = _norm_mrope_smem(
                        x_smem.index(h), k_weight_ptr, cos, sin, EPS, row_layout, rope_layout
                    )
                    _store_rows(k_out_ptr + h * HEAD_DIM, tok, tok_r, own, own_r, K_HEADS * HEAD_DIM, y, rope_lo,
                                rope_hi, row_layout, rope_layout)
                    if gl.max(write.to(gl.int32), axis=0) > 0:
                        if KV_LOC:
                            page, s_page = _kv_loc_k(xk_r, slot // P, h)
                        else:
                            page = ((slot // P) * K_HEADS + h) * (P * K_ROW)
                            s_page = page + P * (HEAD_DIM // 2)
                        offset = slot % P
                        _store_k_nvfp4(k_cache_ptr, rope_lo.to(gl.float32), page, s_page, offset, write, 0,
                                       ROPE_HALF, SUB, P, SCALE_PRESHUFFLE, rope_layout)
                        _store_k_nvfp4(k_cache_ptr, rope_hi.to(gl.float32), page, s_page, offset, write,
                                       ROPE_HALF, ROPE_HALF, SUB, P, SCALE_PRESHUFFLE, rope_layout)
                        _store_k_nvfp4_tail(k_cache_ptr, gl.where(tail, y.to(gl.float32), 0.0), page, s_page,
                                            offset, write, SUB, P, SCALE_PRESHUFFLE, rope_layout)
        if full:
            # one full tile: rows f .. f + 15 are the tile's rows 0 .. 15
            gl.barrier()
            for hh in gl.static_range(K_HEADS):
                gl.amd.gfx1250.tdm.async_load(k_desc, [f, hh * HEAD_DIM], x_smem.index(hh))
            same = _issue_tables(f, positions_ptr, stride_positions, cos_desc, sin_desc, tab_smem, num_tokens,
                                 SUB, idx_layout)
            tok = f + sub_n
            tok_r = f + sub_r
            all_n = tok >= 0
            all_r = tok_r >= 0
            o0 = slot_f % P  # the tile's first row in the page
            if KV_LOC:
                xk_f = gl.load(x_table_ptr + slot_f // P).to(gl.int64) & 1
            gl.amd.gfx1250.tdm.async_wait(0)
            gl.barrier()
            cos, sin = _rope_from_smem(tab_smem, same, SECTION_H, SECTION_W, rope_layout)
            for h in gl.static_range(K_HEADS):
              if KV_LOC:
                  page_f, s_page_f = _kv_loc_k(xk_f, slot_f // P, h)
              else:
                  page_f = ((slot_f // P) * K_HEADS + h) * (P * K_ROW)
                  s_page_f = page_f + P * D_BYTES
              y, rope_lo, rope_hi = _norm_mrope_smem(
                  x_smem.index(h), k_weight_ptr, cos, sin, EPS, row_layout, rope_layout
              )
              _store_rows(k_out_ptr + h * HEAD_DIM, tok, tok_r, all_n, all_r, K_HEADS * HEAD_DIM, y, rope_lo,
                          rope_hi, row_layout, rope_layout)
              d_tail, s_tail = _nvfp4_quant(gl.where(tail, y.to(gl.float32), 0.0), HEAD_DIM, SUB)
              d_lo, s_lo = _nvfp4_quant(rope_lo.to(gl.float32), ROPE_HALF, SUB)
              d_hi, s_hi = _nvfp4_quant(rope_hi.to(gl.float32), ROPE_HALF, SUB)
              # LDS images in global order: data [byte // 16, row, byte % 16] (2 KB), scales [row, group]; the RoPE
              # columns' bytes / scales overwrite the tail quantization's first ones (its zero-input groups)
              d_smem.permute([1, 0, 2]).store(gl.reshape(d_tail, [SUB, D_BYTES // 16, 16]))
              s_smem.store(s_tail.to(gl.float8e4nv).to(gl.uint8, bitcast=True))
              gl.barrier()
              d_smem.slice(0, 1, dim=0).store(gl.reshape(d_lo, [1, SUB, 16]))
              d_smem.slice(1, 1, dim=0).store(gl.reshape(d_hi, [1, SUB, 16]))
              s_smem.slice(0, 2, dim=1).store(s_lo.to(gl.float8e4nv).to(gl.uint8, bitcast=True))
              s_smem.slice(2, 2, dim=1).store(s_hi.to(gl.float8e4nv).to(gl.uint8, bitcast=True))
              gl.barrier()
              d_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
                  base=k_cache_ptr + page_f + (o0 // SUB) * (D_BYTES * SUB),
                  shape=(D_BYTES // 16, SUB, 16), strides=(256, 16, 1),
                  block_shape=(D_BYTES // 16, SUB, 16), layout=SH3,
              )
              # scale rows o0 .. o0 + 15 sit at (o % PQ) * 64 + (o // PQ) * 16: one [16, 16] box, row stride 64
              s_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
                  base=k_cache_ptr + s_page_f + (o0 % PQ) * (4 * D_SCALES) + (o0 // PQ) * D_SCALES,
                  shape=(SUB, D_SCALES), strides=(4 * D_SCALES, 1),
                  block_shape=(SUB, D_SCALES), layout=SH,
              )
              gl.amd.gfx1250.tdm.async_store(d_desc, [0, 0, 0], d_smem)
              gl.amd.gfx1250.tdm.async_store(s_desc, [0, 0], s_smem)
              gl.amd.gfx1250.tdm.async_wait(0)
              gl.barrier()  # the LDS images are free for the next head
    elif pid < num_k_blocks + num_q_blocks:
        # ---- Q: 16 tokens of Q head h
        qb = pid - num_k_blocks
        h0 = (qb % (Q_HEADS // QH_PER)) * QH_PER
        t0 = (qb // (Q_HEADS // QH_PER)) * BLOCK_M_Q
        q_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
            base=q_ptr, shape=(num_tokens, Q_HEADS * HEAD_DIM), strides=(stride_q, 1),
            block_shape=(BLOCK_M_Q, HEAD_DIM), layout=SH,
        )
        cos_desc_q = gl.amd.gfx1250.tdm.make_tensor_descriptor(
            base=cos_ptr, shape=(num_pos, ROPE_HALF), strides=(stride_cos, 1), block_shape=(BLOCK_M_Q, ROPE_HALF),
            layout=SH,
        )
        sin_desc_q = gl.amd.gfx1250.tdm.make_tensor_descriptor(
            base=sin_ptr, shape=(num_pos, ROPE_HALF), strides=(stride_cos, 1), block_shape=(BLOCK_M_Q, ROPE_HALF),
            layout=SH,
        )
        idx_q: gl.constexpr = gl.BlockedLayout([BLOCK_M_Q], [32], [NW], [0])
        x_smem = gl.allocate_shared_memory(q_ptr.dtype.element_ty, [QH_PER, BLOCK_M_Q, HEAD_DIM], SH)
        tab_smem = gl.allocate_shared_memory(cos_ptr.dtype.element_ty, [6, BLOCK_M_Q, ROPE_HALF], SH)
        for hh in gl.static_range(QH_PER):
            gl.amd.gfx1250.tdm.async_load(q_desc, [t0, (h0 + hh) * HEAD_DIM], x_smem.index(hh))
        same = _issue_tables(t0, positions_ptr, stride_positions, cos_desc_q, sin_desc_q, tab_smem, num_tokens,
                             BLOCK_M_Q, idx_q)
        tok = t0 + gl.arange(0, BLOCK_M_Q, layout=gl.SliceLayout(1, row_layout))
        tok_r = t0 + gl.arange(0, BLOCK_M_Q, layout=gl.SliceLayout(1, rope_layout))
        mask = tok < num_tokens
        mask_r = tok_r < num_tokens
        gl.amd.gfx1250.tdm.async_wait(0)
        gl.barrier()
        cos, sin = _rope_from_smem(tab_smem, same, SECTION_H, SECTION_W, rope_layout)
        fp8_ty = q_out_ptr.dtype.element_ty
        for hh in gl.static_range(QH_PER):
            y, rope_lo, rope_hi = _norm_mrope_smem(
                x_smem.index(hh), q_weight_ptr, cos, sin, EPS, row_layout, rope_layout
            )
            _store_rows(
                q_out_ptr + (h0 + hh) * HEAD_DIM, tok, tok_r, mask, mask_r, Q_HEADS * HEAD_DIM,
                gl.minimum(gl.maximum(y.to(gl.float32), -FP8_MAX), FP8_MAX).to(fp8_ty),
                gl.minimum(gl.maximum(rope_lo.to(gl.float32), -FP8_MAX), FP8_MAX).to(fp8_ty),
                gl.minimum(gl.maximum(rope_hi.to(gl.float32), -FP8_MAX), FP8_MAX).to(fp8_ty),
                row_layout, rope_layout,
            )
    else:
        # ---- V: 16 tokens of V head h into the shuffled fp8 V cache (16 tokens of a page are 16 adjacent bytes)
        vb = pid - num_k_blocks - num_q_blocks
        h = vb % K_HEADS
        t0 = (vb // K_HEADS) * BLOCK_M_V
        v_smem = gl.allocate_shared_memory(v_ptr.dtype.element_ty, [BLOCK_M_V, HEAD_DIM], SH)
        v_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
            base=v_ptr + h * HEAD_DIM, shape=(num_tokens, HEAD_DIM), strides=(stride_v, 1),
            block_shape=(BLOCK_M_V, HEAD_DIM), layout=SH,
        )
        gl.amd.gfx1250.tdm.async_load(v_desc, [t0, 0], v_smem)
        v_layout: gl.constexpr = gl.BlockedLayout([1, 4], [16, 2], [1, NW], [0, 1])
        v_tok = t0 + gl.arange(0, BLOCK_M_V, layout=gl.SliceLayout(1, v_layout))
        offs_d = gl.arange(0, HEAD_DIM, layout=gl.SliceLayout(0, v_layout))
        v_slot = gl.load(slot_mapping_ptr + v_tok, mask=v_tok < n_valid, other=-1).to(gl.int64)
        v_write = (v_tok < n_valid) & (v_slot >= 0)
        v_off = v_slot % P
        if KV_LOC:
            xv = (gl.load(x_table_ptr + v_slot // P, mask=v_write, other=0).to(gl.int64) >> 1) & 1
            v_dst = _kv_loc_v(xv, v_slot // P, h, v_off, 16)
        else:
            v_dst = ((v_slot // P) * K_HEADS + h) * (HEAD_DIM * P) + (v_off // 16) * (HEAD_DIM * 16) + v_off % 16
        gl.amd.gfx1250.tdm.async_wait(0)
        gl.barrier()
        v_rows = v_smem.load(v_layout)
        v_c = gl.minimum(gl.maximum(v_rows.to(gl.float32) * kv_inv_scale, -FP8_MAX), FP8_MAX).to(
            v_cache_ptr.dtype.element_ty
        )
        gl.store(v_cache_ptr + v_dst[:, None] + offs_d[None, :] * 16, v_c, mask=v_write[:, None])
