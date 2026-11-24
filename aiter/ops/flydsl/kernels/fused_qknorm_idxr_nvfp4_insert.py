# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# Ported from amd-kernels (attention/norm_rope_nvfp4_insert_d128_rot64_p128/kernel.py,
# Apache-2.0, Copyright contributors to the amd-kernels project).

"""Gemma RMS norm + partial NeoX RoPE of q / k / index-q / index-k and the NVFP4 K / V and index-K page inserts in
one launch (gfx950), bit for bit the HIP norm / RoPE kernel followed by the Triton NVFP4 insert.

A norm / RoPE wave is one head of ``2 * tp`` tokens, 32 lanes of 4 dims per token (the HIP kernel's lane map, so its
arithmetic carries over): per lane ((x0^2 + x1^2) + x2^2) + x3^2, a xor-16/8/4/2/1 butterfly, rsqrt of
fma(sum, 1/128, eps), (x * r) * (1 + w), RoPE fma(x, cos, -/+ RN(partner * sin)). K and index-K are rounded to bf16
and quantized as the Triton insert does: scale e4m3(min(RN(amax * RN(1/6)) * (1 / gs), 448)), codes
e2m1(x * (1 / (gs * scale))), both divisions correctly rounded. A V wave (one KV head of two tokens) reads V straight
from qkv: the first token of a run of tokens in one octet writes the run and keeps the octet's other nibbles and scales.

Waves of 2 * tp tokens: [q heads | k heads | index-k | index-q heads | v heads x tp].
"""

import functools
import struct

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl._mlir import ir as _ir
from flydsl._mlir.dialects import arith as _arith
from flydsl._mlir.dialects import llvm as _llvm
from flydsl._mlir.dialects import rocdl as _rocdl_ops
from flydsl.expr import range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from aiter.ops.flydsl.kernels import buffer_ops
from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled

NW = 4  # slots (waves) per workgroup
D = 128
BLK = 128  # tokens per NVFP4 page
# kernel arguments preloaded into SGPRs (the first 56 bytes) instead of s_load'ed
PRELOAD_HINTS = {
    "llvm_options": {
        "amdgpu-kernarg-preload": True,
        "amdgpu-kernarg-preload-count": 16,
    }
}
RCP6 = 0x3E2AAAAB  # RN(1 / 6), the Triton insert's constant
F448 = 0x43E00000  # 448.0
F448N = 0xC3E00000 - (1 << 32)  # -448.0


def _f32(bits):
    return fx.Int32(bits).bitcast(fx.Float32)


def _bits(x):
    return fx.Float32(x).bitcast(fx.Int32)


def _bf16x4(v):
    """4 fp32 of a dword pair of bf16 (low half first)."""
    w0, w1 = fx.Int32(v[0]), fx.Int32(v[1])
    return [
        (w0 << 16).bitcast(fx.Float32),
        (w0 & -65536).bitcast(fx.Float32),
        (w1 << 16).bitcast(fx.Float32),
        (w1 & -65536).bitcast(fx.Float32),
    ]


def _pk_bf16(a, b):
    """Two fp32 -> two bf16 in a dword (v_cvt_pk_bf16_f32, RNE), ``a`` low."""
    return fx.Int32(
        fx.rocdl.cvt_pk_bf16_f32(
            fx.as_ir_value(fx.Float32(a)), fx.as_ir_value(fx.Float32(b))
        )
    )


def _norm_rope_out4(xws_t, css_t, w, eps, hi16, first, in_rope, half4):
    """``_norm_rope4`` of a bf16 dword pair."""
    return _norm_rope4(_bf16x4(xws_t), css_t, w, eps, hi16, first, in_rope, half4)


def _norm_rope4(e, css_t, w, eps, hi16, first, in_rope, half4):
    """Norm + RoPE of this lane's 4 fp32 dims ``e`` of one token -> 4 fp32."""
    cos, sin = css_t
    ss = e[0] * e[0] + e[1] * e[1]
    ss = ss + e[2] * e[2]
    ss = ss + e[3] * e[3]
    for m in (16, 8, 4, 2, 1):
        ss = _xsum(ss, m)
    r = _rsq(fx.math.fma(ss, fx.Float32(0.0078125), eps))
    nrm = [(e[i] * r) * (fx.Float32(1.0) + w[i]) for i in range_constexpr(4)]
    out = []
    for i in range_constexpr(4):
        pr = _xor_lane(_bits(nrm[i]), half4, hi16).bitcast(fx.Float32)
        ps = pr * sin[i]
        roped = fx.math.fma(nrm[i], cos[i], first.select(-ps, ps))
        out.append(in_rope.select(roped, nrm[i]))
    return out


def _maxnum(a, b):
    return fx.Float32(_arith.MaxNumFOp(fx.as_ir_value(a), fx.as_ir_value(b)).result)


def _minnum(a, b):
    return fx.Float32(_arith.MinNumFOp(fx.as_ir_value(a), fx.as_ir_value(b)).result)


def _dpp(src, ctrl, old=None, bank_mask=0xF):
    """v_mov_b32_dpp of an i32 (lanes a bank mask leaves out keep ``old``; an undef
    ``old`` lets the move fold into its consumer)."""
    o = _llvm.mlir_undef(T.i32) if old is None else fx.as_ir_value(old)
    full = old is None and bank_mask == 0xF
    return fx.Int32(
        _rocdl_ops.update_dpp(T.i32, o, fx.as_ir_value(src), ctrl, 0xF, bank_mask, full)
    )


def _swap16(v):
    """v_permlane16_swap of ``v`` with itself: (row 0 / 2 value, row 1 / 3 value)
    of each 32-lane half, i.e. (own, lane ^ 16) in rows 0 / 2 and swapped in 1 / 3."""
    pair_ty = _ir.Type.parse("!llvm.struct<(i32, i32)>")
    res = _rocdl_ops.permlane16_swap(
        pair_ty, fx.as_ir_value(v), fx.as_ir_value(v), False, False
    )
    return fx.Int32(_llvm.extractvalue(T.i32, res, [0])), fx.Int32(
        _llvm.extractvalue(T.i32, res, [1])
    )


def _xor_lane(v, m, hi16):
    """Lane ``l ^ m`` (m = 1, 2, 4, 8, 16) of the 32-lane half, as an i32. ``hi16``:
    this lane's bit 4."""
    if m == 1:
        return _dpp(v, 0xB1)  # quad_perm [1, 0, 3, 2]
    if m == 2:
        return _dpp(v, 0x4E)  # quad_perm [2, 3, 0, 1]
    if m == 4:  # row_shr:4 into banks 1, 3, then row_shl:4 into banks 0, 2
        return _dpp(v, 0x104, old=_dpp(v, 0x114, bank_mask=0xA), bank_mask=0x5)
    if m == 8:
        return _dpp(v, 0x128)  # row_ror:8
    assert m == 16
    a, b = _swap16(v)
    return hi16.select(a, b)


def _xsum(x, m):
    """Adds lane l ^ m to ``x`` (a commutative fold: the xor-16 pair is unordered)."""
    if m == 16:
        a, b = _swap16(_bits(x))
        return a.bitcast(fx.Float32) + b.bitcast(fx.Float32)
    return x + _xor_lane(_bits(x), m, None).bitcast(fx.Float32)


def _xmax(x, m):
    return _maxnum(x, _xor_lane(_bits(x), m, None).bitcast(fx.Float32))


def _rsq(x):
    """The rsqrt of ocml (v_rsq_f32): its input scaling below 2^-126 never applies,
    as x = fma(sum of squares, 1/128, eps) >= eps >= 2^-126 (the host checks eps)."""
    return fx.Float32(_rocdl_ops.rsq(T.f32, fx.as_ir_value(x)))


def _e4m3_byte(x):
    """RNE e4m3 of ``x`` (v_cvt_scalef32_pk_fp8_f32, scale 1), as an i32 byte."""
    pk = _rocdl_ops.cvt_scalef32_pk_fp8_f32(
        T.vec(2, T.i16),
        fx.as_ir_value(Vec.filled(2, 0, fx.Int16)),
        fx.as_ir_value(x),
        fx.as_ir_value(x),
        fx.as_ir_value(fx.Float32(1.0)),
        False,
    )
    return Vec(pk).bitcast(fx.Int32)[0] & 0xFF


def _e4m3_value(byte):
    pk = _rocdl_ops.cvt_scalef32_pk_f32_fp8(
        T.vec(2, T.f32),
        fx.as_ir_value(fx.Int32(byte)),
        fx.as_ir_value(fx.Float32(1.0)),
        False,
    )
    return fx.Float32(Vec(pk)[0])


def _iq_fp8(a, b, old, hi):
    """Two fp32 -> two e4m3 bytes (v_cvt_pk_fp8_f32) in the low (``hi``: high) word
    of ``old``, ``a`` the lower byte."""
    return fx.Int32(
        _rocdl_ops.cvt_pk_fp8_f32(
            T.i32,
            fx.as_ir_value(fx.Float32(a)),
            fx.as_ir_value(fx.Float32(b)),
            fx.as_ir_value(fx.Int32(old)),
            hi,
        )
    )


def _e2m1x2(a, b, old, sel):
    """E2M1 codes of ``a``, ``b`` (v_cvt_scalef32_pk_fp4_f32, scale 1) into byte ``sel``
    of ``old``, ``a`` the low nibble: bit for bit nvfp4_kv._f32_to_e2m1 on every fp32
    bit pattern."""
    return fx.Int32(
        _rocdl_ops.cvt_scalef32_pk_fp4_f32(
            T.i32,
            fx.as_ir_value(fx.Int32(old)),
            fx.as_ir_value(a),
            fx.as_ir_value(b),
            fx.as_ir_value(fx.Float32(1.0)),
            sel,
        )
    )


def _quant4(xs, gs, inv_gs):
    """E2M1 codes of this lane's 4 values (16 bits, xs[0] lowest) and the e4m3 byte
    of their 16-value group (lanes l ^ 0..3): nvfp4_kv._quantize_nvfp4."""
    a = [fx.math.absf(x) for x in xs]
    m = _maxnum(_maxnum(a[0], a[1]), _maxnum(a[2], a[3]))
    m = _xmax(_xmax(m, 1), 2)
    sf = _minnum((m * _f32(RCP6)) * inv_gs, _f32(F448))
    byte = _e4m3_byte(sf)
    sv = _e4m3_value(byte)
    quant = (sv != 0.0).select(fx.Float32(1.0) / (gs * sv), fx.Float32(0.0))
    y = [x * quant for x in xs]
    return _e2m1x2(y[2], y[3], _e2m1x2(y[0], y[1], fx.Int32(0), 0), 1), byte


def _v_row(state, row, r, n_run, p0, vs, inv_vs):
    """``_v_row4`` of a bf16 dword pair."""
    return _v_row4(state, _bf16x4(row), r, n_run, p0, vs, inv_vs)


def _v_row4(state, xs, r, n_run, p0, vs, inv_vs):
    """Adds row ``r`` (this lane's 4 fp32 dims ``xs``) of a run of ``n_run`` tokens at octet
    positions p0 .. to ``state``: the 4 V dwords' new nibbles, their mask, this lane
    group's 8 octet scale bytes (two dwords) and their byte mask."""
    words, written = state[0:4], state[4]
    sc, sc_mask = state[5:7], state[7:9]
    nib, sf = _quant4(xs, vs, inv_vs)
    in_run = n_run > r
    p = p0 + r
    sh = p * 4
    words = [
        wd | in_run.select(((nib >> (4 * i)) & 15) << sh, fx.Int32(0))
        for i, wd in enumerate(words)
    ]
    written = written | in_run.select(fx.Int32(15) << sh, fx.Int32(0))
    bsh = (p % 4) * 8
    for k in range_constexpr(2):
        here = in_run & ((p // 4) == k)
        sc[k] = sc[k] | here.select(sf << bsh, fx.Int32(0))
        sc_mask[k] = sc_mask[k] | here.select(fx.Int32(255) << bsh, fx.Int32(0))
    return [*words, written, *sc, *sc_mask]


def _gptr(addr, ty, align):
    return fx.inttoptr(
        fx.PointerType.get(ty, address_space=fx.AddressSpace.Global, alignment=align),
        fx.Int64(addr),
    )


def _k_scale_off(tok, grp):
    """Byte of (token, group) in a head's main-K scale region (NVFP4_LAYOUT_MAIN_K)."""
    r = tok % 32
    n = (r // 8) * 4 + r % 4
    return ((((tok // 32) * 4 + grp // 2) * 16 + n) * 2 + (r % 8) // 4) * 2 + grp % 2


def compile_norm_rope_insert(nq: int, nkv: int, niq: int, rotary_dim: int, tp: int):
    """The main kernel's launcher.

    Args:
        tp: token pairs per norm / RoPE wave (a V wave takes one pair).
    """
    assert rotary_dim in (8, 16, 32, 64, 128)
    NR = nq + nkv + 1 + niq  # slots that norm + rope
    S = NR + nkv * tp  # waves per 2 * tp tokens
    ROW = (nq + 2 * nkv + niq + 1) * D  # qkv row (elements)
    HALF = rotary_dim // 2
    SIDE = nkv * BLK * 72  # one side of a main page (bytes)
    TPW = 2 * tp  # tokens per wave
    # nt q / index-q stores keep the QKV GEMM's output in L2: QKV chain 1024 / 2048 tokens -1.2 / -1.0%
    Q_STORE_CM = 2 if (nq, nkv, niq, rotary_dim, tp) == (16, 1, 1, 64, 4) else 0
    name = (
        f"norm_rope_nvfp4_insert_d128_p128_q{nq}_kv{nkv}_iq{niq}"
        f"_rot{rotary_dim}_pairs{tp}"
    )
    if Q_STORE_CM:
        name += "_qnt"

    # the first loads' arguments lead: the first 56 bytes arrive preloaded in SGPRs
    @flyc.kernel(name=name, known_block_size=[64 * NW, 1, 1])
    def kernel(
        i32_n: fx.Int32,
        i32_cs_rows: fx.Int32,
        arg_qkv: fx.Int64,  # [n, ROW] bf16
        arg_pos: fx.Int64,  # positions [n] i64
        arg_slot: fx.Int64,  # main slots [n] i64, -1 skips
        arg_islot: fx.Int64,  # index slots [n] i64, -1 skips
        arg_cs: fx.Int64,  # cos_sin_cache [i32_cs_rows, rotary_dim] bf16
        arg_qw: fx.Int64,  # norm weights [128] bf16: q, k, index-q, index-k
        arg_kw: fx.Int64,
        arg_iqw: fx.Int64,
        arg_ikw: fx.Int64,
        arg_q: fx.Int64,  # q_out [n, nq * 128] bf16
        arg_iq: fx.Int64,  # index_q_out rows of niq * 128 e4m3, row stride i32_iq_st
        arg_kv: fx.Int64,  # main pages [blocks, 2 * nkv, 128, 72] u8, stride i64_kv_st
        arg_idx: fx.Int64,  # index pages [blocks, 128, 72] u8, stride i64_idx_st
        arg_ks: fx.Int64,  # k / v global scales, fp32 scalars
        arg_vs: fx.Int64,
        i64_kv_st: fx.Int64,  # bytes
        i64_idx_st: fx.Int64,  # bytes
        i32_iq_st: fx.Int32,  # bytes
        i32_eps: fx.Int32,  # fp32 bits
    ):
        tx, bx = fx.thread_idx.x, fx.block_idx.x
        lane = tx % 64
        wave = fx.Int32(fx.rocdl.readfirstlane(T.i32, tx // 64))
        wid = bx * NW + wave
        s = wid % S
        hl = lane % 32
        hi16 = (hl & 16) != 0
        dim0 = hl * 4
        toks = [(wid // S) * TPW + t * 2 + lane // 32 for t in range_constexpr(tp)]
        lives = [tk < i32_n for tk in toks]

        qkv = buffer_ops.create_buffer_resource_from_addr(
            arg_qkv, num_records_bytes=fx.Int64(i32_n) * (ROW * 2)
        )
        sl_r = buffer_ops.create_buffer_resource_from_addr(
            arg_slot, num_records_bytes=fx.Int64(i32_n) * 8
        )

        if s < NR:
            is_q = s < nq
            is_k = (s >= nq) & (s < nq + nkv)
            is_ik = s == nq + nkv
            is_kq = is_k | is_ik
            # the head's column (in heads) of the qkv row
            col = (s <= nq + nkv).select(
                is_ik.select(fx.Int32(nq + 2 * nkv + niq), s),
                s - (nq + nkv + 1) + (nq + 2 * nkv),
            )
            w_addr = is_q.select(
                arg_qw, is_k.select(arg_kw, is_ik.select(arg_ikw, arg_iqw))
            )
            wr = buffer_ops.create_buffer_resource_from_addr(
                w_addr, num_records_bytes=D * 2
            )
            pos_r = buffer_ops.create_buffer_resource_from_addr(
                arg_pos, num_records_bytes=fx.Int64(i32_n) * 8
            )
            isl_r = buffer_ops.create_buffer_resource_from_addr(
                arg_islot, num_records_bytes=fx.Int64(i32_n) * 8
            )
            cs_r = buffer_ops.create_buffer_resource_from_addr(
                arg_cs, num_records_bytes=fx.Int64(i32_cs_rows) * (rotary_dim * 2)
            )
            xws, poss, slots = [], [], []
            for t in range_constexpr(tp):
                xws.append(
                    Vec(
                        buffer_ops.buffer_load(
                            qkv,
                            toks[t] * (ROW // 2) + col * (D // 2) + hl * 2,
                            vec_width=2,
                            dtype=fx.Int32,
                            mask=lives[t],
                        )
                    )
                )
                poss.append(
                    fx.Int32(
                        buffer_ops.buffer_load(
                            pos_r,
                            toks[t] * 2,
                            vec_width=1,
                            dtype=fx.Int32,
                            mask=lives[t],
                        )
                    )
                )
                kslot = fx.Int32(
                    buffer_ops.buffer_load(
                        sl_r,
                        toks[t] * 2,
                        vec_width=1,
                        dtype=fx.Int32,
                        mask=lives[t] & is_k,
                    )
                )
                islot = fx.Int32(
                    buffer_ops.buffer_load(
                        isl_r,
                        toks[t] * 2,
                        vec_width=1,
                        dtype=fx.Int32,
                        mask=lives[t] & is_ik,
                    )
                )
                slots.append(is_k.select(kslot, islot))
            w = _bf16x4(
                Vec(buffer_ops.buffer_load(wr, hl * 2, vec_width=2, dtype=fx.Int32))
            )
            first = dim0 < HALF
            in_rope = dim0 < rotary_dim
            ib = first.select(dim0, dim0 - HALF)
            css = []
            for t in range_constexpr(tp):
                cs_off = (poss[t] * rotary_dim + ib) // 2
                css.append(
                    (
                        _bf16x4(
                            Vec(
                                buffer_ops.buffer_load(
                                    cs_r,
                                    cs_off,
                                    vec_width=2,
                                    dtype=fx.Int32,
                                    mask=lives[t] & in_rope,
                                )
                            )
                        ),
                        _bf16x4(
                            Vec(
                                buffer_ops.buffer_load(
                                    cs_r,
                                    cs_off + HALF // 2,
                                    vec_width=2,
                                    dtype=fx.Int32,
                                    mask=lives[t] & in_rope,
                                )
                            )
                        ),
                    )
                )
            eps = i32_eps.bitcast(fx.Float32)
            qvs = []  # every pair first, so the scheduler can interleave them
            for t in range_constexpr(tp):
                out = _norm_rope_out4(
                    xws[t], css[t], w, eps, hi16, first, in_rope, HALF // 4
                )
                lo, hi = _pk_bf16(out[0], out[1]), _pk_bf16(out[2], out[3])
                qvs.append(Vec.from_elements([lo, hi], fx.Int32))
            if is_q:
                q_r = buffer_ops.create_buffer_resource_from_addr(
                    arg_q, num_records_bytes=fx.Int64(i32_n) * (nq * D * 2)
                )
                for t in range_constexpr(tp):
                    buffer_ops.buffer_store(
                        qvs[t],
                        q_r,
                        toks[t] * (nq * D // 2) + s * (D // 2) + hl * 2,
                        mask=lives[t],
                        cache_modifier=Q_STORE_CM,
                    )
            if s > nq + nkv:
                iq_r = buffer_ops.create_buffer_resource_from_addr(
                    arg_iq, num_records_bytes=fx.Int64(i32_n) * fx.Int64(i32_iq_st)
                )
                iq_col = (s - (nq + nkv + 1)) * D
                for t in range_constexpr(tp):
                    f = [
                        _minnum(_maxnum(x, _f32(F448N)), _f32(F448))
                        for x in _bf16x4(qvs[t])
                    ]
                    word = _iq_fp8(
                        f[2], f[3], _iq_fp8(f[0], f[1], fx.Int32(0), False), True
                    )
                    buffer_ops.buffer_store(
                        word,
                        iq_r,
                        toks[t] * i32_iq_st + iq_col + dim0,
                        mask=lives[t],
                        offset_is_bytes=True,
                        cache_modifier=Q_STORE_CM,
                    )
            if is_kq:
                gs = is_k.select(
                    fx.Float32(_gptr(arg_ks, T.f32, 4)[0]), fx.Float32(1.0)
                )
                inv_gs = fx.Float32(1.0) / gs
                quants = [
                    _quant4(_bf16x4(qvs[t]), gs, inv_gs) for t in range_constexpr(tp)
                ]
                h = s - nq
                grp = hl // 4
                for t in range_constexpr(tp):
                    slot = slots[t]
                    tp_ = slot & (BLK - 1)
                    page = is_k.select(
                        arg_kv + fx.Int64(slot >> 7) * i64_kv_st,
                        arg_idx + fx.Int64(slot >> 7) * i64_idx_st,
                    )
                    d_off = is_k.select(h * (BLK * 64), fx.Int32(0)) + tp_ * 64 + hl * 2
                    s_off = is_k.select(
                        nkv * BLK * 64 + h * (BLK * 8) + _k_scale_off(tp_, grp),
                        BLK * 64 + tp_ * 8 + grp,
                    )
                    d_ptr = _gptr(page + fx.Int64(d_off), T.i16, 2)
                    s_ptr = _gptr(page + fx.Int64(s_off), T.i8, 1)
                    packed = quants[t][0].to(fx.Int16)
                    sf8 = quants[t][1].to(fx.Int8)
                    if lives[t] & (slot >= 0):
                        d_ptr[0] = packed
                        if hl % 4 == 0:
                            s_ptr[0] = sf8
        else:
            vh = (s - NR) // tp
            tok = (wid // S) * TPW + ((s - NR) % tp) * 2 + lane // 32
            live = tok < i32_n
            vcol = (nq + nkv + vh) * (D // 2)  # V head column (dwords)
            vs = fx.Float32(_gptr(arg_vs, T.f32, 4)[0])
            near_t = tok - 7 + hl
            has_near = (hl < 16) & (near_t >= 0) & (near_t < i32_n)
            near = has_near.select(
                fx.Int32(
                    buffer_ops.buffer_load(
                        sl_r, near_t * 2, vec_width=1, dtype=fx.Int32, mask=has_near
                    )
                ),
                fx.Int32(-1),
            )
            slot = fx.Int32(
                buffer_ops.buffer_load(
                    sl_r, tok * 2, vec_width=1, dtype=fx.Int32, mask=live
                )
            )
            has_prev = live & (tok > 0)
            prev = has_prev.select(
                fx.Int32(
                    buffer_ops.buffer_load(
                        sl_r, tok * 2 - 2, vec_width=1, dtype=fx.Int32, mask=has_prev
                    )
                ),
                fx.Int32(-1),
            )
            # the rows a run starting here can cover, before its slot is known
            rows = [
                Vec(
                    buffer_ops.buffer_load(
                        qkv,
                        (tok + r) * (ROW // 2) + vcol + hl * 2,
                        vec_width=2,
                        dtype=fx.Int32,
                        mask=tok + r < i32_n,
                    )
                )
                for r in range_constexpr(8)
            ]
            base = slot - slot % 8
            in_oct = (near >= base) & (near < base + 8)
            votes = fx.Int64(fx.rocdl.ballot(T.i64, in_oct))
            mine = (votes >> fx.Int64((lane // 32) * 32)).to(fx.Int32) & 0xFFFF
            n_run = fx.Int32(fx.math.ctpop(mine))
            run_first = (prev < 0) | ((prev >> 3) != (slot >> 3))
            tp_ = slot & (BLK - 1)
            v_side = arg_kv + fx.Int64(slot >> 7) * i64_kv_st + SIDE
            v_data = v_side + fx.Int64(vh * (BLK * 64) + (tp_ // 8) * (D * 4))
            v_sf = v_side + fx.Int64(nkv * BLK * 64 + vh * (BLK * 8) + tp_ - tp_ % 8)
            dp = _gptr(v_data + fx.Int64(dim0 * 4), T.i32, 4)
            sp = _gptr(v_sf + fx.Int64((hl // 4) * BLK), T.i32, 4)
            go = live & (slot >= 0) & run_first
            inv_vs = fx.Float32(1.0) / vs
            # runs of <= 4 tokens (decode) take the short tile
            for run_rows, take in (
                (rows[:4], go & (n_run <= 4)),
                (rows, go & (n_run > 4)),
            ):
                if take:
                    old = [fx.Int32(dp[i]) for i in range_constexpr(4)]
                    old_sc = [fx.Int32(sp[i]) for i in range_constexpr(2)]
                    st = [fx.Int32(0)] * 9
                    for r in range_constexpr(len(run_rows)):
                        st = _v_row(st, run_rows[r], r, n_run, tp_ % 8, vs, inv_vs)
                    keep = ~st[4]
                    for i in range_constexpr(4):
                        dp[i] = st[i] | (old[i] & keep)
                    sc0 = st[5] | (old_sc[0] & ~st[7])
                    sc1 = st[6] | (old_sc[1] & ~st[8])
                    if hl % 4 == 0:
                        sp[0] = sc0
                        sp[1] = sc1

    @flyc.jit
    def launch(
        arg_qkv: fx.Int64,
        arg_q: fx.Int64,
        arg_iq: fx.Int64,
        arg_qw: fx.Int64,
        arg_kw: fx.Int64,
        arg_iqw: fx.Int64,
        arg_ikw: fx.Int64,
        arg_cs: fx.Int64,
        arg_pos: fx.Int64,
        arg_slot: fx.Int64,
        arg_islot: fx.Int64,
        arg_kv: fx.Int64,
        arg_idx: fx.Int64,
        arg_ks: fx.Int64,
        arg_vs: fx.Int64,
        i32_n: fx.Int32,
        i32_iq_st: fx.Int32,
        i64_kv_st: fx.Int64,
        i64_idx_st: fx.Int64,
        i32_cs_rows: fx.Int32,
        i32_eps: fx.Int32,
        stream: fx.Stream,
    ):
        waves = (i32_n + (TPW - 1)) // TPW * S
        kernel(
            i32_n,
            i32_cs_rows,
            arg_qkv,
            arg_pos,
            arg_slot,
            arg_islot,
            arg_cs,
            arg_qw,
            arg_kw,
            arg_iqw,
            arg_ikw,
            arg_q,
            arg_iq,
            arg_kv,
            arg_idx,
            arg_ks,
            arg_vs,
            i64_kv_st,
            i64_idx_st,
            i32_iq_st,
            i32_eps,
        ).launch(
            grid=(fx.Int64((waves + (NW - 1)) // NW), 1, 1),
            block=(64 * NW, 1, 1),
            stream=stream,
        )

    launch.kernel_name = name
    launch.compile_hints = PRELOAD_HINTS
    return launch


# ---------------------------------------------------------------------------
# host
# ---------------------------------------------------------------------------

# (q heads, KV heads, index heads, rotary dim) of a MiniMax-M3 layer at TP4: the head set the kernel is tuned and
# verified for. Other head sets stay on the HIP kernel.
HEADS = (16, 1, 1, 64)
PAGE_B = 9216  # bytes of one head's K (or V) side of a 128-token page: data + scales


@functools.cache
def _get_kernel(nq: int, nkv: int, niq: int, rotary_dim: int, tp: int):
    return compile_norm_rope_insert(nq, nkv, niq, rotary_dim, tp)


# the prefill variant (fewer vector memory instructions per wave) from this many tokens, eight pairs per wave in the
# range below (measured, MI355X)
PREFILL_TOKENS = 1537
PREFILL_EIGHT_PAIRS = (24576, 32768)


@functools.cache
def _get_prefill_kernel(nq: int, nkv: int, niq: int, rotary_dim: int, tp: int):
    from aiter.ops.flydsl.kernels import fused_qknorm_idxr_nvfp4_insert_prefill as pf

    return pf.compile_norm_rope_insert(nq, nkv, niq, rotary_dim, tp)


def _token_pairs(n: int) -> int:
    """Token pairs per norm / RoPE wave: four from 489 tokens, one below (measured, MI355X)."""
    return 4 if n >= 489 else 1


def nvfp4_insert_blocker(
    qkv,
    q_norm_weight,
    k_norm_weight,
    cos_sin_cache,
    positions,
    num_heads,
    num_kv_heads,
    rotary_dim,
    eps,
    index_q_norm_weight,
    index_k_norm_weight,
    num_index_heads,
    slot_mapping,
    kv_cache_k,
    kv_cache_v,
    index_cache,
    block_size,
    q_out,
    index_q_out,
    index_slot_mapping,
    k_scale,
    v_scale,
):
    """Why this call is not one the FlyDSL insert takes (None: it takes it)."""
    if (num_heads, num_kv_heads, num_index_heads, rotary_dim) != HEADS:
        return "head set"
    if block_size != 128 or eps < 2.0**-126:
        return "block_size / eps"
    if qkv.dtype != torch.bfloat16 or not qkv.is_contiguous():
        return "qkv"
    if qkv.shape[1] != (num_heads + 2 * num_kv_heads + num_index_heads + 1) * D:
        return "qkv width"
    if (
        cos_sin_cache.dtype != torch.bfloat16
        or not cos_sin_cache.is_contiguous()
        or cos_sin_cache.dim() != 2
        or cos_sin_cache.shape[1] != rotary_dim
    ):
        return "cos_sin_cache"
    for w in (
        q_norm_weight,
        k_norm_weight,
        index_q_norm_weight,
        index_k_norm_weight,
    ):
        if w is None or w.dtype != torch.bfloat16 or not w.is_contiguous():
            return "norm weights"
    if positions.dtype != torch.int64 or not positions.is_contiguous():
        return "positions"
    for s in (slot_mapping, index_slot_mapping):
        if s is None or s.dtype != torch.int64 or not s.is_contiguous():
            return "slot_mapping"
    if q_out is None or q_out.dtype != torch.bfloat16 or not q_out.is_contiguous():
        return "q_out"
    if (
        index_q_out is None
        or index_q_out.dtype != torch.float8_e4m3fn
        or index_q_out.stride(1) != 1
        or index_q_out.stride(0) % 4
    ):
        return "index_q_out"
    # one page buffer [K side | V side] for K and V alike, whole contiguous pages
    if (
        kv_cache_k is None
        or kv_cache_v is None
        or kv_cache_k.data_ptr() != kv_cache_v.data_ptr()
        or kv_cache_k.dtype != torch.uint8
        or kv_cache_k.stride(-1) != 1
        or kv_cache_k.stride(0) != 2 * num_kv_heads * PAGE_B
        or kv_cache_k.stride(0) % 4
    ):
        return "kv cache"
    if (
        index_cache is None
        or index_cache.dtype != torch.uint8
        or index_cache.stride(-1) != 1
        or index_cache.stride(0) != PAGE_B
    ):
        return "index cache"
    for s in (k_scale, v_scale):
        if s is None or s.dtype != torch.float32 or s.numel() != 1:
            return "k_scale / v_scale"
    return None


def fused_qknorm_idxr_nvfp4_insert(
    qkv,
    q_norm_weight,
    k_norm_weight,
    cos_sin_cache,
    positions,
    eps,
    index_q_norm_weight,
    index_k_norm_weight,
    slot_mapping,
    kv_cache,
    index_cache,
    q_out,
    index_q_out,
    index_slot_mapping,
    k_scale,
    v_scale,
):
    """``fused_qknorm_idxrqknorm`` for NVFP4 K / V / index-K pages on gfx950 (head set ``HEADS``), one launch.

    The caller checked ``nvfp4_insert_blocker``; ``kv_cache`` is the shared ``[pages, 2 * 9216]`` uint8 page buffer.
    Writes the same bytes as the HIP kernel (page bytes, ``q_out``, ``index_q_out``); ``qkv`` is not written.
    """
    n = qkv.shape[0]
    if n == 0:
        return
    nq, nkv, niq, rotary_dim = HEADS
    if n >= PREFILL_TOKENS:
        pairs = 8 if PREFILL_EIGHT_PAIRS[0] <= n <= PREFILL_EIGHT_PAIRS[1] else 4
        exe = _get_prefill_kernel(nq, nkv, niq, rotary_dim, pairs)
    else:
        exe = _get_kernel(nq, nkv, niq, rotary_dim, _token_pairs(n))
    _run_compiled(
        exe,
        qkv.data_ptr(),
        q_out.data_ptr(),
        index_q_out.data_ptr(),
        q_norm_weight.data_ptr(),
        k_norm_weight.data_ptr(),
        index_q_norm_weight.data_ptr(),
        index_k_norm_weight.data_ptr(),
        cos_sin_cache.data_ptr(),
        positions.data_ptr(),
        slot_mapping.data_ptr(),
        index_slot_mapping.data_ptr(),
        kv_cache.data_ptr(),
        index_cache.data_ptr(),
        k_scale.data_ptr(),
        v_scale.data_ptr(),
        n,
        index_q_out.stride(0),
        kv_cache.stride(0),
        index_cache.stride(0),
        cos_sin_cache.shape[0],
        struct.unpack("<i", struct.pack("<f", eps))[0],
        torch.cuda.current_stream(qkv.device),
    )
