# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Triton kernels for the Sol-Attn Jensen cap and pooled value tilt (PoC).

Each runs one program per (KV block, head, batch) or (Q tile, head, batch), so the cap and the
factors cost one pass over Q's bytes, one over K and V, and no per-block host work.

Float arguments go through tl.cast to fp32 first: an eager launch passes them as fp32, but
torch.compile can pass fp64 scalars, and Python floats while it checks which tensors a kernel writes.
"""

import triton
import triton.language as tl
from triton.language.extra.hip import libdevice


@triton.jit
def _sol_q_tile_absmax_kernel(
    q_ptr,
    scale_ptr,
    rot_ptr,
    out_ptr,
    seqlen,
    rot_gain,
    stride_qb,
    stride_qs,
    stride_qh,
    stride_sb,
    stride_ss,
    stride_sh,
    stride_ob,
    stride_ot,
    stride_oh,
    HAS_SCALE: tl.constexpr,
    ROTATE: tl.constexpr,
    Q_TILE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    """max |q| per (tile, head, dim), stored in out_ptr's dtype.

    HAS_SCALE dequantizes by an E8M0 byte per 32 dims first; ROTATE takes |q R| for R = rot_gain
    times the +-1 matrix at rot_ptr instead of |q|.
    """
    rot_gain = tl.cast(rot_gain, tl.float32)
    t = tl.program_id(0)
    h = tl.program_id(1)
    b = tl.program_id(2)
    rows = t * Q_TILE + tl.arange(0, Q_TILE)
    cols = tl.arange(0, HEAD_DIM)
    live = rows[:, None] < seqlen
    x = tl.load(
        q_ptr + b * stride_qb + h * stride_qh + rows[:, None] * stride_qs + cols[None, :],
        mask=live,
    )
    if ROTATE:
        signs = tl.load(rot_ptr + cols[:, None] * HEAD_DIM + cols[None, :])
        x = tl.dot(tl.where(live, x, 0.0).to(signs.dtype), signs) * rot_gain
    else:
        x = x.to(tl.float32)
    if HAS_SCALE:
        byte = tl.load(
            scale_ptr + b * stride_sb + h * stride_sh + rows[:, None] * stride_ss + (cols // 32)[None, :],
            mask=live,
        )
        x = x * tl.exp2(byte.to(tl.float32) - 127.0)
    m = tl.max(tl.where(live, tl.abs(x), 0.0), 0)
    tl.store(out_ptr + b * stride_ob + t * stride_ot + h * stride_oh + cols, m.to(out_ptr.dtype.element_ty))


@triton.jit
def _sol_q_head_absmax_kernel(
    tile_ptr,
    out_ptr,
    tiles,
    stride_tb,
    stride_tt,
    stride_th,
    stride_ob,
    stride_oh,
    TILE_CHUNK: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    """The per-tile maxima maxed over every tile of the head."""
    h = tl.program_id(0)
    b = tl.program_id(1)
    cols = tl.arange(0, HEAD_DIM)
    m = tl.zeros([HEAD_DIM], dtype=tl.float32)
    for t0 in range(0, tiles, TILE_CHUNK):
        t = t0 + tl.arange(0, TILE_CHUNK)
        tile = tl.load(
            tile_ptr + b * stride_tb + t[:, None] * stride_tt + h * stride_th + cols[None, :],
            mask=(t < tiles)[:, None],
            other=0.0,
        )
        m = tl.maximum(m, tl.max(tile.to(tl.float32), 0))
    tl.store(out_ptr + b * stride_ob + h * stride_oh + cols, m.to(out_ptr.dtype.element_ty))


@triton.jit
def _sol_jensen_cap_kernel(
    qmax_ptr,
    var_ptr,
    mask_ptr,
    out_var_ptr,
    gain_ptr,
    q_descale_ptr,
    k_descale_ptr,
    var_scale_ptr,
    out_scale_ptr,
    per_nat,
    log_kv_tile,
    tiles,
    stride_qb,
    stride_qt,
    stride_qh,
    stride_vb,
    stride_vn,
    stride_vh,
    stride_mb,
    stride_mh,
    stride_mt,
    stride_mn,
    stride_gb,
    stride_gn,
    stride_gh,
    stride_sb,
    stride_sn,
    stride_sh,
    HAS_MASK: tl.constexpr,
    HAS_DESCALE: tl.constexpr,
    HAS_VAR_SCALE: tl.constexpr,
    EXPONENT_GAIN: tl.constexpr,
    TILE_CHUNK: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    """Per block: J_max = per_nat (qd kd)^2 max over pooling tiles of sum_d qmax_d^2 var_d; gain = min(1, L / J_max).

    HAS_VAR_SCALE decodes var by the E8M0 bytes at var_scale_ptr, one per 32 dims. EXPONENT_GAIN
    applies 2^floor(log2 gain) to those bytes instead of scaling var, which stays as it is, and
    stores that power of two as the gain.
    """
    per_nat = tl.cast(per_nat, tl.float32)
    log_kv_tile = tl.cast(log_kv_tile, tl.float32)
    n = tl.program_id(0)
    h = tl.program_id(1)
    b = tl.program_id(2)
    cols = tl.arange(0, HEAD_DIM)
    var_off = b * stride_vb + n * stride_vn + h * stride_vh + cols
    var = tl.load(var_ptr + var_off).to(tl.float32)
    groups = tl.arange(0, HEAD_DIM // 32)
    scale_off = b * stride_sb + n * stride_sn + h * stride_sh + groups
    if HAS_VAR_SCALE or EXPONENT_GAIN:
        scale = tl.load(var_scale_ptr + scale_off).to(tl.int32)
    if HAS_VAR_SCALE:
        var = tl.reshape(var, [HEAD_DIM // 32, 32]) * tl.exp2(scale.to(tl.float32) - 127.0)[:, None]
        var = tl.reshape(var, [HEAD_DIM])
    if HAS_DESCALE:
        qd = tl.load(q_descale_ptr).to(tl.float32)
        kd = tl.load(k_descale_ptr).to(tl.float32)
    else:
        qd = 1.0
        kd = 1.0
    j_max = tl.zeros([TILE_CHUNK], dtype=tl.float32)
    for t0 in range(0, tiles, TILE_CHUNK):
        t = t0 + tl.arange(0, TILE_CHUNK)
        live = t < tiles
        qm = tl.load(
            qmax_ptr + b * stride_qb + t[:, None] * stride_qt + h * stride_qh + cols[None, :],
            mask=live[:, None],
            other=0.0,
        ).to(tl.float32)
        j = tl.sum(qm * qm * var[None, :], 1)
        if HAS_MASK:
            exact = tl.load(
                mask_ptr + b * stride_mb + h * stride_mh + t * stride_mt + n * stride_mn,
                mask=live,
                other=1,
            )
            j = tl.where(exact.to(tl.int32) != 0, 0.0, j)
        j_max = tl.maximum(j_max, j)
    j_top = tl.max(j_max, 0) * per_nat * (qd * kd) * (qd * kd)
    gain = tl.minimum(log_kv_tile / tl.maximum(j_top, 1e-30), 1.0)
    if EXPONENT_GAIN:
        shift = tl.floor(tl.log2(gain))
        gain = tl.exp2(shift)
        tl.store(out_scale_ptr + scale_off, tl.minimum(tl.maximum(scale + shift.to(tl.int32), 0), 254).to(tl.uint8))
    else:
        tl.store(out_var_ptr + var_off, (var * gain).to(out_var_ptr.dtype.element_ty))
    tl.store(gain_ptr + b * stride_gb + n * stride_gn + h * stride_gh, gain)


@triton.jit
def _orthonormalize(x, RANK: tl.constexpr, WIDTH: tl.constexpr, PASSES: tl.constexpr):
    """x's first RANK columns, Gram-Schmidt in order, as PASSES rounds of Cholesky-QR.

    Each round does its per-column work on the WIDTH x WIDTH x^T x. One round leaves an
    orthogonality error of eps cond(x)^2, which a second removes. A column left with under 1e-5
    of its squared norm lies in the earlier ones' span to rounding and becomes zero: its residual
    there is noise, and scaling it up would overflow the next round.
    """
    rows = tl.arange(0, WIDTH)
    for _ in tl.static_range(PASSES):
        g = tl.dot(tl.trans(x), x, input_precision="ieee")
        t = tl.zeros([WIDTH, WIDTH], dtype=tl.float32)
        for i in tl.static_range(RANK):
            col = (rows == i).to(tl.float32)
            g_col = tl.sum(g * col[None, :], 1)
            g_ii = tl.sum(col * g_col, 0)
            if i > 0:
                proj = tl.sum(t * g_col[:, None], 0)
                col = col - tl.sum(t * proj[None, :], 1)
                g_col = tl.sum(g * col[None, :], 1)
            norm2 = tl.sum(col * g_col, 0)
            col = tl.where(norm2 > 1e-5 * g_ii, col / tl.sqrt(tl.maximum(norm2, 1e-30)), 0.0)
            t = tl.where(rows[None, :] == i, col[:, None], t)
        x = tl.dot(x, t, input_precision="ieee")
    return x


@triton.jit
def _sol_tilt_factor_kernel(
    k_ptr,
    v_ptr,
    qmax_ptr,
    gain_ptr,
    a_ptr,
    b_ptr,
    peak_ptr,
    stride_kb,
    stride_ks,
    stride_kh,
    stride_vb,
    stride_vs,
    stride_vh,
    stride_qb,
    stride_qh,
    stride_gb,
    stride_gn,
    stride_gh,
    stride_fr,
    stride_fb,
    stride_fn,
    stride_fh,
    HAS_GAIN: tl.constexpr,
    TO_BF16: tl.constexpr,
    RANK: tl.constexpr,
    WIDTH: tl.constexpr,
    ITERS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    """One block's rank-RANK tilt: a = x / bound, b = (C x) bound, x the leading right singular vectors of C = V_c^T K_c.

    TO_BF16 takes K and V of any 8- or 16-bit dtypes the MFMA does not pair, exactly, to bf16.

    x comes from subspace iteration started at C's largest rows, orthonormalized in column order
    after each step, all in registers. bound = sum_d max|q_d| |x_d| over the head's queries.
    Writes a and b in fp32 to (RANK, batch, blocks, heads, d) planes and the block's max |a| and
    max |b| to peak_ptr's (batch, heads, blocks, 2).
    """
    n = tl.program_id(0)
    h = tl.program_id(1)
    b = tl.program_id(2)
    rows = n * BLOCK_N + tl.arange(0, BLOCK_N)
    dims = tl.arange(0, HEAD_DIM)
    slots = tl.arange(0, WIDTH)

    kt = tl.load(k_ptr + b * stride_kb + h * stride_kh + rows[:, None] * stride_ks + dims[None, :])
    vt = tl.load(v_ptr + b * stride_vb + h * stride_vh + rows[:, None] * stride_vs + dims[None, :])
    if TO_BF16:
        kt = kt.to(tl.bfloat16)
        vt = vt.to(tl.bfloat16)
    # Centered as V^T K - s_v s_k^T / N: 8-bit products and their sums are exact in fp32, bf16
    # products are exact and their sums round once each.
    ones = tl.full([16, BLOCK_N], 1.0, dtype=tl.float32).to(kt.dtype)
    sk = tl.dot(ones, kt)
    sv = tl.dot(ones, vt)
    c = tl.dot(tl.trans(vt), kt) - tl.dot(tl.trans(sv), sk, input_precision="ieee") / (16 * BLOCK_N)
    norms = tl.sum(c * c, 1)
    c = c.to(tl.bfloat16)
    c_t = tl.trans(c)

    pick = tl.zeros([HEAD_DIM, WIDTH], dtype=tl.float32)
    for i in tl.static_range(RANK):
        top = tl.argmax(norms, 0)
        norms = tl.where(dims == top, -1.0, norms)
        pick = tl.where((dims[:, None] == top) & (slots[None, :] == i), 1.0, pick)
    # Conditioning the start only needs one round; the last basis gets two.
    x = _orthonormalize(tl.dot(c_t, pick.to(tl.bfloat16)), RANK, WIDTH, 1 if ITERS > 0 else 2)
    for it in tl.static_range(ITERS):
        y = tl.dot(c, x.to(tl.bfloat16))
        x = _orthonormalize(tl.dot(c_t, y.to(tl.bfloat16)), RANK, WIDTH, 2 if it == ITERS - 1 else 1)

    left = tl.dot(c, x.to(tl.bfloat16))
    qm = tl.load(qmax_ptr + b * stride_qb + h * stride_qh + dims).to(tl.float32)
    bound = tl.maximum(tl.sum(tl.abs(x) * qm[:, None], 0), 1e-30)
    fa = x / bound[None, :]
    fb = left * bound[None, :]
    if HAS_GAIN:
        fb = fb * tl.load(gain_ptr + b * stride_gb + n * stride_gn + h * stride_gh)

    live = slots[None, :] < RANK
    off = slots[None, :] * stride_fr + b * stride_fb + n * stride_fn + h * stride_fh + dims[:, None]
    tl.store(a_ptr + off, fa, mask=live)
    tl.store(b_ptr + off, fb, mask=live)
    peak = peak_ptr + 2 * ((b * tl.num_programs(1) + h) * tl.num_programs(0) + n)
    tl.store(peak, tl.max(tl.max(tl.where(live, tl.abs(fa), 0.0), 1), 0))
    tl.store(peak + 1, tl.max(tl.max(tl.where(live, tl.abs(fb), 0.0), 1), 0))


@triton.jit
def _sol_tilt_shift_kernel(
    q_descale_ptr,
    k_descale_ptr,
    v_descale_ptr,
    peak_ptr,
    peaks,
    shift_ptr,
    unit_ptr,
    score_unit,
    b_per_scale,
    w_max_limit,
    fmax,
    b_headroom,
    BLOCK: tl.constexpr,
):
    """The tilt's two E8M0 shifts and its planes' multipliers, from the factor kernel's peaks.

    peak_ptr holds `peaks` (max |a|, max |b|) pairs. With base = score_unit qd kd / a_max, the
    largest score, the score shift s is floor(log2(w_max_limit / base)), so w_max = base 2^s, and
    the value shift ceil(log2(b_peak / (vd w_max fmax b_headroom))) for b_peak = b_max * b_scale,
    b_scale = qd kd vd b_per_scale. shift_ptr gets both as signed bytes (s low), clamped to
    [-128, 127]; unit_ptr gets fmax / a_max and b_scale / b_unit, b_unit = vd w_max 2^vshift.
    """
    score_unit = tl.cast(score_unit, tl.float32)
    b_per_scale = tl.cast(b_per_scale, tl.float32)
    w_max_limit = tl.cast(w_max_limit, tl.float32)
    fmax = tl.cast(fmax, tl.float32)
    b_headroom = tl.cast(b_headroom, tl.float32)
    pairs = tl.arange(0, BLOCK)
    a_max = tl.zeros([BLOCK], dtype=tl.float32)
    b_max = tl.zeros([BLOCK], dtype=tl.float32)
    for p0 in range(0, peaks, BLOCK):
        live = p0 + pairs < peaks
        a_max = tl.maximum(a_max, tl.load(peak_ptr + 2 * (p0 + pairs), mask=live, other=0.0))
        b_max = tl.maximum(b_max, tl.load(peak_ptr + 2 * (p0 + pairs) + 1, mask=live, other=0.0))
    a_top = tl.maximum(tl.max(a_max, 0), 1e-30)
    b_top = tl.max(b_max, 0)
    qd = tl.load(q_descale_ptr).to(tl.float32)
    kd = tl.load(k_descale_ptr).to(tl.float32)
    vd = tl.load(v_descale_ptr).to(tl.float32)
    base = score_unit * kd * qd / a_top
    shift = tl.minimum(tl.maximum(tl.floor(tl.log2(w_max_limit / base)), -128.0), 127.0)
    w_max = base * tl.exp2(shift)
    b_scale = vd * kd * qd * b_per_scale
    b_peak = tl.maximum(b_top * b_scale, 1e-30)
    vshift = tl.ceil(tl.log2(b_peak / (vd * w_max * fmax * b_headroom)))
    vshift = tl.minimum(tl.maximum(vshift, -128.0), 127.0)
    b_unit = vd * w_max * tl.exp2(vshift)
    tl.store(shift_ptr, (shift.to(tl.int32) & 0xFF) | ((vshift.to(tl.int32) & 0xFF) << 8))
    tl.store(unit_ptr, fmax / a_top)
    tl.store(unit_ptr + 1, b_scale / b_unit)


@triton.jit
def _sol_tilt_unit_kernel(
    q_descale_ptr,
    k_descale_ptr,
    v_descale_ptr,
    peak_ptr,
    peaks,
    unit_ptr,
    q_multiplier,
    b_per_scale,
    a_fmax,
    b_fmax,
    A_CODES: tl.constexpr,
    B_CODES: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Any format's tilt multipliers, from the factor kernel's peaks.

    A plane holds a * u0 and a B plane b * u1, which a per-tensor code format sets to fill its range
    (a_fmax / a_max, b_fmax / b_max) and any other format to 1 and b_scale. With A and B decoded
    the way mean_k and mean_v are, score = u2 (q . A) for Q as the kernel decodes it and the value
    update is P score B u3: u2 = 1 / (q_multiplier qd kd u0), u3 = b_scale / (vd u1), b_scale =
    qd kd vd b_per_scale. Writes (u0, u1, u2, u3) to unit_ptr.
    """
    q_multiplier = tl.cast(q_multiplier, tl.float32)
    b_per_scale = tl.cast(b_per_scale, tl.float32)
    a_fmax = tl.cast(a_fmax, tl.float32)
    b_fmax = tl.cast(b_fmax, tl.float32)
    pairs = tl.arange(0, BLOCK)
    a_max = tl.zeros([BLOCK], dtype=tl.float32)
    b_max = tl.zeros([BLOCK], dtype=tl.float32)
    for p0 in range(0, peaks, BLOCK):
        live = p0 + pairs < peaks
        a_max = tl.maximum(a_max, tl.load(peak_ptr + 2 * (p0 + pairs), mask=live, other=0.0))
        b_max = tl.maximum(b_max, tl.load(peak_ptr + 2 * (p0 + pairs) + 1, mask=live, other=0.0))
    qd = tl.load(q_descale_ptr).to(tl.float32)
    kd = tl.load(k_descale_ptr).to(tl.float32)
    vd = tl.load(v_descale_ptr).to(tl.float32)
    b_scale = qd * kd * vd * b_per_scale
    if A_CODES:
        a_unit = a_fmax / tl.maximum(tl.max(a_max, 0), 1e-30)
    else:
        a_unit = 1.0
    if B_CODES:
        b_unit = b_fmax / tl.maximum(tl.max(b_max, 0), 1e-30)
    else:
        b_unit = b_scale
    tl.store(unit_ptr, a_unit)
    tl.store(unit_ptr + 1, b_unit)
    tl.store(unit_ptr + 2, 1.0 / (q_multiplier * qd * kd * a_unit))
    tl.store(unit_ptr + 3, b_scale / (vd * b_unit))


@triton.jit
def _sol_tilt_valu_kernel(
    unit_ptr, q_descale_ptr, k_descale_ptr, out_ptr, w_max, stride0, stride1, stride2,
    STRIDES: tl.constexpr,
):
    """A code-score kernel's tilt parameters, from _sol_tilt_unit_kernel's units.

    The kernel takes the raw q . A code product times sigma to the weight score, and the value
    GEMM's E8M0 operand carries 2^vshift, so sigma 2^vshift = u2 qd kd u3 with sigma = u2 qd kd m,
    m = u3 2^-vshift in (w_max / 2, w_max]. out_ptr gets vshift as a signed byte in bits 8..15 of
    dword 0, sigma's bits in dword 1, then the first STRIDES of stride0, stride1 and stride2.
    """
    w_max = tl.cast(w_max, tl.float32)
    if STRIDES > 0:
        tl.store(out_ptr + 2, stride0)
    if STRIDES > 1:
        tl.store(out_ptr + 3, stride1)
    if STRIDES > 2:
        tl.store(out_ptr + 4, stride2)
    u2 = tl.load(unit_ptr + 2)
    u3 = tl.maximum(tl.load(unit_ptr + 3), 1e-30)
    qd = tl.load(q_descale_ptr).to(tl.float32)
    kd = tl.load(k_descale_ptr).to(tl.float32)
    vshift = tl.minimum(tl.maximum(tl.ceil(tl.log2(u3 / w_max)), -127.0), 127.0)
    sigma = u2 * qd * kd * (u3 * tl.exp2(-vshift))
    tl.store(out_ptr, (vshift.to(tl.int32) & 0xFF) << 8)
    tl.store(out_ptr + 1, sigma.to(tl.int32, bitcast=True))


@triton.jit
def _sol_tilt_scaled_kernel(unit_ptr, scale_stride, out_ptr, w_max):
    """A scaled-score kernel's tilt parameters, from _sol_tilt_unit_kernel's units.

    The score MFMA already decodes Q and the block-scaled A plane, so sigma must not carry qd or
    kd again: sigma 2^vshift = u2 u3, with sigma = u2 m and m = u3 2^-vshift in (w_max / 2,
    w_max]. out_ptr gets vshift as a signed byte in bits 8..15 of dword 0, sigma's bits in dword
    1, and the K-scale plane stride in dword 2. The score side has no shift of its own.
    """
    w_max = tl.cast(w_max, tl.float32)
    u2 = tl.load(unit_ptr + 2)
    u3 = tl.maximum(tl.load(unit_ptr + 3), 1e-30)
    vshift = tl.minimum(tl.maximum(tl.ceil(tl.log2(u3 / w_max)), -127.0), 127.0)
    sigma = u2 * (u3 * tl.exp2(-vshift))
    tl.store(out_ptr, (vshift.to(tl.int32) & 0xFF) << 8)
    tl.store(out_ptr + 1, sigma.to(tl.int32, bitcast=True))
    tl.store(out_ptr + 2, scale_stride)


@triton.jit
def _sol_tilt_pack_kernel(
    a_ptr,
    b_ptr,
    mean_k_ptr,
    mean_v_ptr,
    ext_k_ptr,
    ext_v_ptr,
    unit_ptr,
    rank,
    full,
    stride_fr,
    stride_fb,
    stride_fn,
    stride_fh,
    stride_mkb,
    stride_mkn,
    stride_mkh,
    stride_mvb,
    stride_mvn,
    stride_mvh,
    stride_er,
    stride_eb,
    stride_en,
    stride_eh,
    stride_vr,
    stride_veb,
    stride_ven,
    stride_veh,
    ROUND_K: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    """Plane 0 = the pooled mean, planes 1..rank = the factors in K's and V's dtypes; zero for the short last block.

    unit_ptr holds the A and B planes' multipliers. ROUND_K rounds the A codes half to even, as
    sol_prepare rounds an integer mean_k, which an integer K needs because its cast truncates.
    """
    n = tl.program_id(0)
    h = tl.program_id(1)
    b = tl.program_id(2)
    a_unit = tl.load(unit_ptr)
    b_unit = tl.load(unit_ptr + 1)
    dims = tl.arange(0, HEAD_DIM)
    ext = b * stride_eb + n * stride_en + h * stride_eh + dims
    ext_v = b * stride_veb + n * stride_ven + h * stride_veh + dims
    mk = tl.load(mean_k_ptr + b * stride_mkb + n * stride_mkn + h * stride_mkh + dims)
    mv = tl.load(mean_v_ptr + b * stride_mvb + n * stride_mvn + h * stride_mvh + dims)
    tl.store(ext_k_ptr + ext, mk.to(ext_k_ptr.dtype.element_ty))
    tl.store(ext_v_ptr + ext_v, mv.to(ext_v_ptr.dtype.element_ty))
    src = b * stride_fb + n * stride_fn + h * stride_fh + dims
    live = (dims < HEAD_DIM) & (n < full)
    for r in range(rank):
        fa = tl.load(a_ptr + r * stride_fr + src, mask=live, other=0.0) * a_unit
        fb = tl.load(b_ptr + r * stride_fr + src, mask=live, other=0.0)
        if ROUND_K:
            fa = libdevice.nearbyint(fa)
        tl.store(ext_k_ptr + (r + 1) * stride_er + ext, fa.to(ext_k_ptr.dtype.element_ty))
        tl.store(ext_v_ptr + (r + 1) * stride_vr + ext_v, (fb * b_unit).to(ext_v_ptr.dtype.element_ty))


@triton.jit
def _sol_tilt_stack_kernel(plane0_ptr, planes_ptr, out_ptr, size, width, BLOCK: tl.constexpr):
    """Row p of out_ptr's (1 + rank, width) gets plane0 for p = 0, else plane p - 1 of planes.

    Each plane holds `size` elements of out_ptr's dtype; the rest of the row is left as it is.
    """
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    p = tl.program_id(1)
    live = i < size
    if p == 0:
        x = tl.load(plane0_ptr + i, mask=live)
    else:
        x = tl.load(planes_ptr + (p - 1).to(tl.int64) * size + i, mask=live)
    tl.store(out_ptr + p.to(tl.int64) * width + i, x, mask=live)
