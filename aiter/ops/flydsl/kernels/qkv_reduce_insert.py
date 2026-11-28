# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Technique source: amd-kernels attention/mxfp8_qkv_norm_rope_nvfp4_insert_n2560_k6144_rot64_p128/reduce_insert.py
# (split-3, one token pair per wave) and packing.py.
"""MiniMax-M3.1 QKV split-3 reduce + norm / RoPE + NVFP4 insert, fused (gfx950, TP4 shapes).

One launch in place of a split-K reduce and the standalone ``fused_qknorm_idxrqknorm`` insert: the QKV GEMM's three fp32
K-third slabs ``[3, n, 2560]`` (``mxfp8_gemm_decode_splitk``) are summed in order (0 + s0 + s1 + s2), rounded to bf16 as a
reduce store would, and fed to the norm / RoPE / NVFP4-insert arithmetic of the unfused ``fused_qknorm_idxrqknorm`` (HIP norm / RoPE + NVFP4 insert) unchanged;
the bf16 QKV intermediate is never written.

A norm / RoPE wave covers one slot (q / k / index-k / index-q head) of two tokens, 32 lanes of 4 dims per token (the
lane map of the HIP norm / RoPE kernel). V has workgroups of its own, 8 tokens each, ahead of the norm workgroups:
the rows of the 8 tokens and of the next 8 (halo, loaded only when a run can reach it) are summed, rounded and quantized
once into LDS, and the first token of every octet run packs the run's codes and scales from LDS after one CTA barrier.
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr import range_constexpr
from flydsl.expr.typing import BFloat16, Float32, T
from flydsl.expr.typing import Vector as Vec

from aiter.ops.flydsl.kernels import buffer_ops
from aiter.ops.flydsl.kernels.qkv_chain_cooperative import _lds_ptr
from aiter.ops.flydsl.kernels.qkv_front_common import (
    BLK,
    F448,
    F448N,
    PRELOAD_HINTS,
    D,
    _bf16x4,
    _bits,
    _f32,
    _gptr,
    _iq_fp8,
    _k_scale_off,
    _maxnum,
    _minnum,
    _pk_bf16,
    _quant4,
    _rsq,
    _xor_lane,
    _xsum,
)

NW = 4  # slots (waves) per workgroup
SK = 3  # default number of K-part slabs (``ks`` of compile_qkv_reduce_insert)
NQ, NKV, NIQ, ROT = 16, 1, 1, 64
NR = NQ + NKV + 1 + NIQ  # slots that norm + rope (19)
SN = 20  # wave slots per token pair: whole four-head workgroups, the last slot is padding (``s < NR`` guards it)
ROW = (NQ + 2 * NKV + NIQ + 1) * D  # qkv row (2560 elements)
HALF = ROT // 2
SIDE = NKV * BLK * 72  # one side of a main page (bytes)
TPW = 2  # tokens per wave


# ---- V page words from per-row nibbles (amd-kernels packing.py) ----
def _v_pair(a, b):
    """Two rows -> four bytes: dimension order 0, 2, 1, 3, two token nibbles per byte."""
    value = a | (b << 16)
    exchange = (value ^ (value >> 12)) & 0xF0F0
    return value ^ exchange ^ (exchange << 12)


def _v_perm(hi, lo, select):
    """Select four bytes from (hi:lo); the native byte-permute instruction implements the exact integer mapping."""
    return fx.Int32(fx.rocdl.perm_b32(fx.Int32(hi), fx.Int32(lo), fx.Int32(select)))


def _v_words(nibbles):
    """Four words, each containing four / eight tokens of one dimension (high bits zero for four tokens)."""
    pairs = [_v_pair(nibbles[i], nibbles[i + 1]) for i in range(0, len(nibbles), 2)]
    x = _v_perm(pairs[1], pairs[0], 0x06020400)
    y = _v_perm(pairs[1], pairs[0], 0x07030501)
    if len(nibbles) == 4:
        return [x & 0xFFFF, (x >> 16) & 0xFFFF, y & 0xFFFF, (y >> 16) & 0xFFFF]
    u = _v_perm(pairs[3], pairs[2], 0x06020400)
    v = _v_perm(pairs[3], pairs[2], 0x07030501)
    return [
        _v_perm(u, x, 0x05040100),
        _v_perm(u, x, 0x07060302),
        _v_perm(v, y, 0x05040100),
        _v_perm(v, y, 0x07060302),
    ]


@functools.cache
def compile_qkv_reduce_insert(ks: int = SK):
    SK = ks  # K-part slabs summed in order 0 + s0 + s1 + ...
    name = (
        f"qkv_reduce_insert_d128_p128_q{NQ}_kv{NKV}_iq{NIQ}_rot{ROT}_s{SK}"
        "_vlds_vfirst_vhalo_cta_heads20"
    )

    @fx.struct
    class Shared:
        vrow: fx.Array[
            fx.Int32, 16 * 32 * 2
        ]  # a V workgroup's 8 rows + 8 halo rows: (codes, scale byte) per lane

    # the first loads' arguments lead: the first 56 bytes arrive preloaded in SGPRs
    @flyc.kernel(name=name, known_block_size=[64 * NW, 1, 1])
    def kernel(
        i32_n: fx.Int32,
        i32_cs_rows: fx.Int32,
        arg_slab: fx.Int64,  # [3, n, ROW] fp32 partial sums of the QKV GEMM
        arg_pos: fx.Int64,  # positions [n] i64
        arg_slot: fx.Int64,  # main slots [n] i64, -1 skips
        arg_islot: fx.Int64,  # index slots [n] i64, -1 skips
        arg_cs: fx.Int64,  # cos_sin_cache [i32_cs_rows, ROT] bf16
        arg_qw: fx.Int64,  # norm weights [128] bf16: q, k, index-q, index-k
        arg_kw: fx.Int64,
        arg_iqw: fx.Int64,
        arg_ikw: fx.Int64,
        arg_q: fx.Int64,  # q_out [n, NQ * 128] bf16
        arg_iq: fx.Int64,  # index_q_out rows of 128 e4m3, row stride i32_iq_st
        arg_kv: fx.Int64,  # main pages [blocks, 2 * NKV, 128, 72] u8, stride i64_kv_st
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
        # V workgroups first (their run chain is the longest), the norm / RoPE workgroups after them
        n_v_wg = (i32_n + 7) // 8
        wid = (fx.Int32(bx) - n_v_wg) * NW + wave
        s = wid % SN
        hl = lane % 32
        hi16 = (hl & 16) != 0
        dim0 = hl * 4
        tok0 = (wid // SN) * TPW + lane // 32
        live0 = (tok0 < i32_n) & (s < NR)
        smem = fx.SharedAllocator().allocate(Shared).peek()

        slab = buffer_ops.create_buffer_resource_from_addr(
            arg_slab, num_records_bytes=fx.Int64(i32_n) * (SK * ROW * 4)
        )
        sl_r = buffer_ops.create_buffer_resource_from_addr(
            arg_slot, num_records_bytes=fx.Int64(i32_n) * 8
        )

        def load4(tok, col_elem, live):
            """This lane's 4 fp32 values of every slab at (token, head-column element base)."""
            return [
                Vec(
                    buffer_ops.buffer_load(
                        slab,
                        (k * i32_n + tok) * ROW + col_elem + dim0,
                        vec_width=4,
                        dtype=Float32,
                        mask=live,
                    )
                )
                for k in range_constexpr(SK)
            ]

        def sum4(frags):
            """The slabs summed in order (0 + s0 + s1 + s2), RNE -> bf16 as a reduce store rounds, widened back."""
            acc = Vec.filled(4, 0.0, Float32)
            for k in range_constexpr(SK):
                acc = acc + frags[k]
            e = acc.to(BFloat16).to(Float32)
            return [e[i] for i in range_constexpr(4)]

        def row4(tok, col_elem, live):
            return sum4(load4(tok, col_elem, live))

        if fx.Int32(bx) >= n_v_wg:
            is_q = s < NQ
            is_k = (s >= NQ) & (s < NQ + NKV)
            is_ik = s == NQ + NKV
            is_kq = is_k | is_ik
            # the head's column (in heads) of the qkv row
            col = (s <= NQ + NKV).select(
                is_ik.select(fx.Int32(NQ + 2 * NKV + NIQ), s),
                s - (NQ + NKV + 1) + (NQ + 2 * NKV),
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
                arg_cs, num_records_bytes=fx.Int64(i32_cs_rows) * (ROT * 2)
            )
            e = row4(tok0, col * D, live0)
            pos = fx.Int32(
                buffer_ops.buffer_load(
                    pos_r, tok0 * 2, vec_width=1, dtype=fx.Int32, mask=live0
                )
            )
            kslot = fx.Int32(
                buffer_ops.buffer_load(
                    sl_r, tok0 * 2, vec_width=1, dtype=fx.Int32, mask=live0 & is_k
                )
            )
            islot = fx.Int32(
                buffer_ops.buffer_load(
                    isl_r, tok0 * 2, vec_width=1, dtype=fx.Int32, mask=live0 & is_ik
                )
            )
            slot = is_k.select(kslot, islot)
            w = _bf16x4(
                Vec(buffer_ops.buffer_load(wr, hl * 2, vec_width=2, dtype=fx.Int32))
            )
            first = dim0 < HALF
            in_rope = dim0 < ROT
            ib = first.select(dim0, dim0 - HALF)
            cs_off = (pos * ROT + ib) // 2
            cos = _bf16x4(
                Vec(
                    buffer_ops.buffer_load(
                        cs_r,
                        cs_off,
                        vec_width=2,
                        dtype=fx.Int32,
                        mask=live0 & in_rope,
                    )
                )
            )
            sin = _bf16x4(
                Vec(
                    buffer_ops.buffer_load(
                        cs_r,
                        cs_off + HALF // 2,
                        vec_width=2,
                        dtype=fx.Int32,
                        mask=live0 & in_rope,
                    )
                )
            )
            eps = i32_eps.bitcast(fx.Float32)
            ss = e[0] * e[0] + e[1] * e[1]
            ss = ss + e[2] * e[2]
            ss = ss + e[3] * e[3]
            for m in (16, 8, 4, 2, 1):
                ss = _xsum(ss, m)
            r = _rsq(fx.math.fma(ss, fx.Float32(0.0078125), eps))
            nrm = [(e[i] * r) * (fx.Float32(1.0) + w[i]) for i in range_constexpr(4)]
            out = []
            for i in range_constexpr(4):
                pr = _xor_lane(_bits(nrm[i]), HALF // 4, hi16).bitcast(fx.Float32)
                ps = pr * sin[i]
                roped = fx.math.fma(nrm[i], cos[i], first.select(-ps, ps))
                out.append(in_rope.select(roped, nrm[i]))
            lo, hi = _pk_bf16(out[0], out[1]), _pk_bf16(out[2], out[3])
            qv = Vec.from_elements([lo, hi], fx.Int32)
            if is_q:
                q_r = buffer_ops.create_buffer_resource_from_addr(
                    arg_q, num_records_bytes=fx.Int64(i32_n) * (NQ * D * 2)
                )
                buffer_ops.buffer_store(
                    qv,
                    q_r,
                    tok0 * (NQ * D // 2) + s * (D // 2) + hl * 2,
                    mask=live0,
                )
            if s > NQ + NKV:
                iq_r = buffer_ops.create_buffer_resource_from_addr(
                    arg_iq, num_records_bytes=fx.Int64(i32_n) * fx.Int64(i32_iq_st)
                )
                iq_col = (s - (NQ + NKV + 1)) * D
                f = [_minnum(_maxnum(x, _f32(F448N)), _f32(F448)) for x in _bf16x4(qv)]
                word = _iq_fp8(
                    f[2], f[3], _iq_fp8(f[0], f[1], fx.Int32(0), False), True
                )
                buffer_ops.buffer_store(
                    word,
                    iq_r,
                    tok0 * i32_iq_st + iq_col + dim0,
                    mask=live0,
                    offset_is_bytes=True,
                )
            if is_kq:
                gs = is_k.select(
                    fx.Float32(_gptr(arg_ks, T.f32, 4)[0]), fx.Float32(1.0)
                )
                inv_gs = fx.Float32(1.0) / gs
                quant = _quant4(_bf16x4(qv), gs, inv_gs)
                h = s - NQ
                grp = hl // 4
                tp_ = slot & (BLK - 1)
                page = is_k.select(
                    arg_kv + fx.Int64(slot >> 7) * i64_kv_st,
                    arg_idx + fx.Int64(slot >> 7) * i64_idx_st,
                )
                d_off = is_k.select(h * (BLK * 64), fx.Int32(0)) + tp_ * 64 + hl * 2
                s_off = is_k.select(
                    NKV * BLK * 64 + h * (BLK * 8) + _k_scale_off(tp_, grp),
                    BLK * 64 + tp_ * 8 + grp,
                )
                d_ptr = _gptr(page + fx.Int64(d_off), T.i16, 2)
                s_ptr = _gptr(page + fx.Int64(s_off), T.i8, 1)
                packed = quant[0].to(fx.Int16)
                sf8 = quant[1].to(fx.Int8)
                if live0 & (slot >= 0):
                    d_ptr[0] = packed
                    if hl % 4 == 0:
                        s_ptr[0] = sf8
        else:
            # Only V workgroups exchange rows through LDS; norm workgroups need no CTA join.
            vbase = fx.Int32(bx) * 8
            tok = vbase + wave * 2 + lane // 32
            live = tok < i32_n
            vcol = (NQ + NKV) * D  # the V head (TP4: one)
            vs = fx.Float32(_gptr(arg_vs, T.f32, 4)[0])
            inv_vs = fx.Float32(1.0) / vs
            fr_own = load4(tok, vcol, live)
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
            # Every wave's neighbour vector includes the same two owner-boundary slots. Different octets prove that
            # no owned contiguous run reaches the halo. Keep the owned slab prefetch unconditional.
            boundary_lane = fx.Int32(14) - wave * 2
            boundary0 = fx.Int32(fx.rocdl.readlane(T.i32, near, boundary_lane))
            boundary1 = fx.Int32(fx.rocdl.readlane(T.i32, near, boundary_lane + 1))
            use_halo = (
                (boundary0 >= 0)
                & (boundary1 >= 0)
                & ((boundary0 >> 3) == (boundary1 >> 3))
            )
            fr_halo = [Vec.filled(4, 0.0, Float32) for _ in range_constexpr(SK)]
            if use_halo:
                fr_halo = load4(tok + 8, vcol, tok + 8 < i32_n)
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
            base = slot - slot % 8
            in_oct = (near >= base) & (near < base + 8)
            votes = fx.Int64(fx.rocdl.ballot(T.i64, in_oct))
            mine = (votes >> fx.Int64((lane // 32) * 32)).to(fx.Int32) & 0xFFFF
            n_run = fx.Int32(fx.math.ctpop(mine))
            run_first = (prev < 0) | ((prev >> 3) != (slot >> 3))
            go = live & (slot >= 0) & run_first
            # a lane that writes nothing reads page 0 (only V workgroups write these dwords)
            wslot = go.select(slot, fx.Int32(0))
            tp_ = wslot & (BLK - 1)
            v_side = arg_kv + fx.Int64(wslot >> 7) * i64_kv_st + SIDE
            v_data = v_side + fx.Int64((tp_ // 8) * (D * 4))
            v_sf = v_side + fx.Int64(NKV * BLK * 64 + tp_ - tp_ % 8)
            dp = _gptr(v_data + fx.Int64(dim0 * 4), T.i32, 4)
            sp = _gptr(v_sf + fx.Int64((hl // 4) * BLK), T.i32, 4)
            vold = [fx.Int32(dp[i]) for i in range_constexpr(4)]
            vold_sc = [fx.Int32(sp[i]) for i in range_constexpr(2)]
            vrow = fx.Int32(fx.ptrtoint(smem.vrow.ptr))
            for p in range_constexpr(2):
                tk = tok + p * 8
                fr = fr_own if p == 0 else fr_halo
                need = (p == 0) or use_halo
                if need:
                    nib_o, sf_o = _quant4(sum4(fr), vs, inv_vs)
                    _llvm.StoreOp(
                        fx.as_ir_value(
                            Vec.from_elements(
                                [fx.Int32(nib_o), fx.Int32(sf_o)], fx.Int32
                            )
                        ),
                        _lds_ptr(vrow + ((tk - vbase) * 32 + hl) * 8),
                        alignment=8,
                    )
            # All four waves of a V workgroup participate, so a CTA join publishes every owned / halo row.
            fx.gpu.barrier()
            for run_rows, take in ((4, go & (n_run <= 4)), (8, go & (n_run > 4))):
                if take:
                    nibbles = []
                    scales = fx.Uint64(0)
                    for r in range_constexpr(run_rows):
                        got = Vec(
                            _llvm.load(
                                T.vec(2, T.i32),
                                _lds_ptr(vrow + ((tok + r - vbase) * 32 + hl) * 8),
                                alignment=8,
                            )
                        )
                        ok = (tok + r < i32_n) & (r < n_run)
                        nib = ok.select(got[0], fx.Int32(0))
                        sf = ok.select(got[1], fx.Int32(0))
                        nibbles.append(nib)
                        scales = scales | (sf.to(fx.Uint64) << (8 * r))
                    words = _v_words(nibbles)
                    p0 = tp_ % 8
                    mask = (fx.Uint32(0xFFFFFFFF) >> ((8 - n_run) * 4)) << (p0 * 4)
                    for i in range_constexpr(4):
                        dp[i] = (words[i] << (p0 * 4)) | (vold[i] & ~mask.to(fx.Int32))
                    smask = (fx.Uint64(0xFFFFFFFFFFFFFFFF) >> ((8 - n_run) * 8)) << (
                        p0 * 8
                    )
                    scales = scales << (p0 * 8)
                    sc0 = scales.to(fx.Int32) | (vold_sc[0] & ~smask.to(fx.Int32))
                    sc1 = (scales >> 32).to(fx.Int32) | (
                        vold_sc[1] & ~(smask >> 32).to(fx.Int32)
                    )
                    if hl % 4 == 0:
                        sp[0] = sc0
                        sp[1] = sc1

    @flyc.jit
    def launch(
        arg_slab: fx.Int64,
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
        waves = (i32_n + (TPW - 1)) // TPW * SN
        kernel(
            i32_n,
            i32_cs_rows,
            arg_slab,
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
            grid=(fx.Int64((waves + (NW - 1)) // NW + (i32_n + 7) // 8), 1, 1),
            block=(64 * NW, 1, 1),
            stream=stream,
        )

    launch.kernel_name = name
    launch.compile_hints = PRELOAD_HINTS
    return launch
