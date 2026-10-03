# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Gfx942 FP8 paged unified attention, one varlen launch per call.

Covers Gemma-4's layer shapes: head 512 with GQA 8:1 and no window, and head
256 with GQA 2:1 and a sliding window. Q/K/V are FP8 E4M3FNUZ with per-tensor
FP32 descales; the output is BF16. K and V are paged with page 32 or 64 and
may be strided views (vLLM hands K and V as views of one cache tensor).

Grid y has two regions. Tile slots follow Triton's q-block mapping (sequence
s owns slots from cu_q[s] // tokens + s, found by binary search, heaviest
first); each runs the prefill body over 64 // GQA query tokens of a
multi-token sequence with four waves, so prefix-cached chunks are prefill
tiles with q_len < k_len. Decode slots give each wave one split of a
one-token sequence, running a 16-key decode body in that wave's own slice of
LDS; decode waves have different trip counts, so they fence LDS with
s_waitcnt instead of workgroup barriers. A combine launch merges the splits.

Both GEMMs use mfma_f32_16x16x32_fp8_fp8 in transposed form (S^T = K Q^T,
O^T += V^T P^T), so P needs no cross-lane moves. P is scaled by 240 before
FNUZ packing and the epilogue undoes it. LDS reads and the V-transpose
shuffles are issued in groups of eight behind sched_barriers: without them
the compiler waits out each one before its consumer.
"""

from functools import lru_cache

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm
from flydsl._mlir.dialects import rocdl as mlir_rocdl
from flydsl.expr import gpu, range_constexpr, rocdl
from flydsl.expr.typing import T

from aiter.ops.flydsl.kernels import buffer_ops

NK = 16  # decode key tile
NO_WINDOW = 1 << 30
# s_waitcnt lgkmcnt(0) on gfx942, leaving vmcnt and expcnt at their maxima.
_LGKMCNT_0_ONLY = 0xC07F


def _load(ptr, offset, dtype, alignment):
    p = buffer_ops.get_element_ptr(fx.to_llvm_ptr(ptr), byte_offset=offset)
    return llvm.LoadOp(dtype, p, alignment=alignment).result


def _store(ptr, offset, value, alignment):
    p = buffer_ops.get_element_ptr(fx.to_llvm_ptr(ptr), byte_offset=offset)
    llvm.StoreOp(
        value.ir_value() if hasattr(value, "ir_value") else value,
        p,
        alignment=alignment,
    )


def _mfma16(a, b, c):
    return fx.Vector(
        rocdl.mfma_f32_16x16x32_fp8_fp8(
            fx.Vector.make_type(4, fx.Float32),
            [a, b, c.ir_value(), 0, 0, 0],
        )
    )


def _exp2(x):
    return fx.Float32(rocdl.exp2(T.f32, x.ir_value()))


def _pack4(values):
    lo = rocdl.cvt_pk_fp8_f32(
        T.i32, values[0].ir_value(), values[1].ir_value(), fx.Int32(0).ir_value(), 0
    )
    return fx.Int32(
        rocdl.cvt_pk_fp8_f32(T.i32, values[2].ir_value(), values[3].ir_value(), lo, 1)
    )


def _pack_bf16(a, b):
    # gfx942 has no packed BF16 conversion.
    pair = fx.Vector.from_elements(
        [fx.Float32(a).to(fx.BFloat16), fx.Float32(b).to(fx.BFloat16)], fx.BFloat16
    )
    return pair.bitcast(fx.Int32)[0]


def _k_offset(key, d, dim):
    # Row-major K; XOR 16-byte chunks by key so 16 rows read in one pass.
    return key * dim + (d ^ (key % 8 * 16))


def _vt_offset(depth, key4, keys):
    # V^T in 32-depth x `keys`-key chunks, one dword per (depth, 4 keys). The
    # XOR terms keep the transpose stores and the PV reads at 2-way banks.
    bank = (depth % 32) ^ ((depth // 32) % 8 * 4) ^ (key4 % 4 * 8)
    return (depth // 32) * (32 * keys) + key4 * 128 + bank * 4


def _wave_lds_fence():
    mlir_rocdl.s_waitcnt(_LGKMCNT_0_ONLY)
    rocdl.sched_barrier(0)


def plan_num_kv_splits(num_seqs, max_seqlen_k, num_kv_heads, window):
    """Decode split count, fitted to MI325X sweeps: at least ~256 workgroups,
    about 16 tiles per split for long contexts, at most 8192 workgroups."""
    keys = max_seqlen_k if window is None else min(max_seqlen_k, window)
    tiles = max(1, (keys + NK - 1) // NK)
    units = num_seqs * num_kv_heads
    return max(1, min(tiles // 2, max(256 // units, tiles // 16), 8192 // units))


def prefill_block_q(num_q_heads, num_kv_heads):
    """Query tokens per prefill workgroup: four waves of 16 // GQA tokens."""
    return 4 * (16 // (num_q_heads // num_kv_heads))


@lru_cache(maxsize=32)
def build_flash_attn_fp8_gfx942_module(
    dim, num_q_heads, num_kv_heads, window, page_size, kv_strides, prefill_keys=None
):
    """Return the attention launcher for one layer shape and KV layout.

    dim is 256 or 512, the GQA ratio must divide 16, and window is None or the
    inclusive key count. kv_strides holds K's then V's (page, token, head)
    element strides. They are compiled in: runtime strides made head-512
    decode up to 45% slower, and a cache layout fixes them, so each layout
    compiles once. Prefill tiles are 64 keys at head 256 and 32 at head 512 (LDS holds
    2 * keys * dim, out of 128 * dim per workgroup).
    """
    if dim not in (256, 512) or page_size not in (32, 64):
        raise ValueError("dim must be 256 or 512 and page size 32 or 64")
    group = num_q_heads // num_kv_heads
    if group * num_kv_heads != num_q_heads or 16 % group:
        raise ValueError("GQA ratio must divide 16")
    tpw = 16 // group
    tokens = 4 * tpw
    ksteps = dim // 32
    dblocks = dim // 16
    lpr = dim // 256
    slice_bytes = 2 * NK * dim
    lds_bytes = 128 * dim
    stride = dim + 4
    wleft = NO_WINDOW if window is None else window - 1
    pn = prefill_keys or (64 if dim == 256 else 32)
    if pn not in (32, 64) or 2 * pn * dim > lds_bytes:
        raise ValueError("prefill_keys must be 32 or 64 and fit in LDS")
    kb_n = pn // 16
    pv_h = pn // 32
    pvt = pn * dim
    k_loads = pn * dim // 4096
    v_loads = pn // 16 * lpr
    k_strides, v_strides = tuple(kv_strides[:3]), tuple(kv_strides[3:])

    @fx.struct
    class SharedStorage:
        buf: fx.Array[fx.Int8, lds_bytes, 16]

    @flyc.kernel(known_block_size=(256, 1, 1))
    def attend(
        Q: fx.Tensor,
        K: fx.Tensor,
        V: fx.Tensor,
        O: fx.Tensor,
        Part: fx.Tensor,
        CuQ: fx.Tensor,
        UsedK: fx.Tensor,
        BT: fx.Tensor,
        QD: fx.Tensor,
        KD: fx.Tensor,
        VD: fx.Tensor,
        num_seqs: fx.Int32,
        tile_slots: fx.Int32,
        groups: fx.Int32,
        search_iters: fx.Int32,
        bt_stride: fx.Int32,
        scale: fx.Float32,
    ):
        tid = fx.Int32(gpu.thread_id("x"))
        lane = tid % 64
        wave = tid // 64
        c = lane % 16
        g = lane // 16
        kvh = fx.Int32(gpu.block_id("x"))
        y = fx.Int32(gpu.block_id("y"))
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        buf = lds.buf.ptr
        log_scale = (
            fx.Float32(fx.memref_load(QD, 0))
            * fx.Float32(fx.memref_load(KD, 0))
            * scale
            * 1.4426950408889634
        )
        vscale = fx.Float32(fx.memref_load(VD, 0)) / 240.0
        qp, kp, vp, op = fx.get_iter(Q), fx.get_iter(K), fx.get_iter(V), fx.get_iter(O)
        pp, btp = fx.get_iter(Part), fx.get_iter(BT)
        splits = groups * 4
        sel_pair = (g % 2 == 0).select(fx.Int32(0x06020400), fx.Int32(0x03070105))
        sel_quad = (g < 2).select(fx.Int32(0x05040100), fx.Int32(0x03020706))

        def kv_src(seq, klen, base_tok, key, d, strides):
            s_page, s_tok, s_head = strides
            tok = base_tok + key
            # Past klen, reload a valid key: stale FNUZ bytes can be NaN.
            safe_tok = (tok < klen).select(tok, base_tok)
            # Per-load page lookup: a 64-key tile spans two pages at page 32.
            page = fx.Int32(
                _load(
                    btp,
                    (
                        fx.Int64(seq) * fx.Int64(bt_stride)
                        + fx.Int64(safe_tok // page_size)
                    )
                    * 4,
                    T.i32,
                    4,
                )
            )
            return (
                fx.Int64(page) * s_page
                + fx.Int64(safe_tok % page_size) * s_tok
                + fx.Int64(kvh) * s_head
                + fx.Int64(d)
            )

        def transpose_store(words, base, key4s, depths, keys):
            # Issue each shuffle round for all words before consuming it.
            peers = [w.shuffle_xor(fx.Int32(16), fx.Int32(64)) for w in words]
            rocdl.sched_barrier(0)
            pairs = [
                fx.Int32(rocdl.perm_b32(p, w, sel_pair)) for p, w in zip(peers, words)
            ]
            peers = [p.shuffle_xor(fx.Int32(32), fx.Int32(64)) for p in pairs]
            rocdl.sched_barrier(0)
            for n in range_constexpr(len(words)):
                packed = rocdl.perm_b32(peers[n], pairs[n], sel_quad)
                _store(buf, base + _vt_offset(depths[n], key4s[n], keys), packed, 4)

        if y < tile_slots:
            # ---- prefill role ----
            yy = tile_slots - 1 - y
            for _it, bs in range(
                fx.Int32(0), search_iters, fx.Int32(1), init=[fx.Int32(0), num_seqs]
            ):
                lo, hi = fx.Int32(bs[0]), fx.Int32(bs[1])
                mid = (lo + hi) // 2
                ok = fx.Int32(fx.memref_load(CuQ, mid)) // tokens + mid <= yy
                found = yield [ok.select(mid, lo), ok.select(hi, mid)]
            pseq = fx.Int32(found[0])
            pq0 = fx.Int32(fx.memref_load(CuQ, pseq))
            pqlen = fx.Int32(fx.memref_load(CuQ, pseq + 1)) - pq0
            pklen = fx.Int32(fx.memref_load(UsedK, pseq))
            ptile = yy - (pq0 // tokens + pseq)
            if (pqlen > 1) & (ptile * tokens < pqlen):
                pqbase = pklen - pqlen
                plast = (ptile * tokens + tokens < pqlen).select(
                    ptile * tokens + tokens, pqlen
                )
                pfirst_pos = pqbase + ptile * tokens
                plast_pos = pqbase + plast - 1
                plow = pfirst_pos - wleft
                pstart = (plow > 0).select(plow, fx.Int32(0)) // pn
                pend = (plast_pos + pn) // pn
                pqtok = ptile * tokens + wave * tpw + c // group
                psafe = (pqtok < pqlen).select(pqtok, fx.Int32(0))
                pqpos = pqbase + pqtok
                phead = kvh * group + c % group
                pq_row = (fx.Int64(pq0) + fx.Int64(psafe)) * num_q_heads + fx.Int64(
                    phead
                )
                pq_frag = [
                    _load(qp, pq_row * dim + fx.Int64(ks * 32 + g * 8), T.i64, 8)
                    for ks in range(ksteps)
                ]

                def p_fetch(block):
                    base_tok = block * pn
                    kc = [
                        fx.Vector(
                            _load(
                                kp,
                                kv_src(
                                    pseq,
                                    pklen,
                                    base_tok,
                                    (tid * 16 + i * 4096) // dim,
                                    (tid * 16 + i * 4096) % dim,
                                    k_strides,
                                ),
                                fx.Vector.make_type(4, fx.Int32),
                                16,
                            )
                        )
                        for i in range(k_loads)
                    ]
                    vc = [
                        fx.Vector(
                            _load(
                                vp,
                                kv_src(
                                    pseq,
                                    pklen,
                                    base_tok,
                                    (wave * (pn // 16) + idx // lpr) * 4 + g,
                                    c * (dim // 16) + idx % lpr * 16,
                                    v_strides,
                                ),
                                fx.Vector.make_type(4, fx.Int32),
                                16,
                            )
                        )
                        for idx in range(v_loads)
                    ]
                    return kc + vc

                @flyc.jit
                def p_fetch_guarded(block):
                    p_chunks = [
                        fx.Vector.filled(4, 0, fx.Int32)
                        for _ in range(k_loads + v_loads)
                    ]
                    if block < pend:
                        p_chunks = p_fetch(block)
                    return p_chunks

                def p_commit(chunks):
                    for i in range_constexpr(k_loads):
                        flat = tid * 16 + i * 4096
                        _store(
                            buf, _k_offset(flat // dim, flat % dim, dim), chunks[i], 16
                        )
                    words, key4s, depths = [], [], []
                    for idx in range_constexpr(v_loads):
                        vval = fx.Vector(chunks[k_loads + idx])
                        for wi in range_constexpr(4):
                            words.append(vval[wi])
                            key4s.append(wave * (pn // 16) + idx // lpr)
                            depths.append(c * (dim // 16) + idx % lpr * 16 + wi * 4 + g)
                    transpose_store(words, pvt, key4s, depths, pn)

                def p_read_k(kb, ks):
                    return _load(
                        buf, _k_offset(kb * 16 + c, ks * 32 + g * 8, dim), T.i64, 8
                    )

                def p_read_v(db, h):
                    # PV MFMA h covers keys 32h..32h+31: key4 8h+g, then 8h+4+g.
                    depth = db * 16 + c
                    w0 = fx.Int32(
                        _load(buf, pvt + _vt_offset(depth, 8 * h + g, pn), T.i32, 4)
                    )
                    w1 = fx.Int32(
                        _load(buf, pvt + _vt_offset(depth, 8 * h + 4 + g, pn), T.i32, 4)
                    )
                    return fx.Vector.from_elements([w0, w1], fx.Int32).bitcast(
                        fx.Int64
                    )[0]

                qk_order = [(ks, kb) for ks in range(ksteps) for kb in range(kb_n)]
                # Within each group of eight, the two halves of a depth block sit four apart.
                pv_order = [
                    (db, h)
                    for d4 in range(0, dblocks, 4)
                    for h in range(pv_h)
                    for db in range(d4, d4 + 4)
                ]
                nsc = kb_n * 4

                def chain(ks, kb):
                    return kb * 2 + ks % 2 if kb_n == 2 else kb

                p_init = (
                    [fx.Float32(-1.0e30), fx.Float32(0.0)]
                    + [fx.Vector.filled(4, 0.0, fx.Float32) for _ in range(dblocks)]
                    + p_fetch_guarded(pstart)
                )
                for block, state in range(pstart, pend, fx.Int32(1), init=p_init):
                    block = fx.Int32(block)
                    rocdl.sched_barrier(0)
                    p_commit(state[2 + dblocks :])
                    gpu.barrier()
                    # The next tile's loads stay in flight through this tile's MFMAs.
                    p_ahead = p_fetch_guarded(block + 1)
                    rocdl.sched_barrier(0)
                    p_acc = [fx.Vector.filled(4, 0.0, fx.Float32) for _ in range(4)]
                    p_group = [p_read_k(kb, ks) for ks, kb in qk_order[0:8]]
                    for grp in range_constexpr(len(qk_order) // 8):
                        p_nxt = (
                            [
                                p_read_k(kb, ks)
                                for ks, kb in qk_order[grp * 8 + 8 : grp * 8 + 16]
                            ]
                            if grp < len(qk_order) // 8 - 1
                            else []
                        )
                        rocdl.sched_barrier(0)
                        for i in range_constexpr(8):
                            ks, kb = qk_order[grp * 8 + i]
                            j = chain(ks, kb)
                            p_acc[j] = _mfma16(p_group[i], pq_frag[ks], p_acc[j])
                        p_group = p_nxt
                    p_parts = (
                        [p_acc[2 * kb] + p_acc[2 * kb + 1] for kb in range(kb_n)]
                        if kb_n == 2
                        else p_acc
                    )
                    p_vgroup = [p_read_v(db, h) for db, h in pv_order[0:8]]
                    rocdl.sched_barrier(0)
                    p_sv = fx.Vector.from_elements(
                        [p_parts[r // 4][r % 4] * log_scale for r in range(nsc)],
                        fx.Float32,
                    )
                    # Only the window edge and the causal diagonal need per-element masks.
                    if (block * pn + pn - 1 > pfirst_pos) | (
                        block * pn < plast_pos - wleft
                    ):
                        p_masked = []
                        for r in range_constexpr(nsc):
                            col = block * pn + (r // 4) * 16 + g * 4 + r % 4
                            valid = (col <= pqpos) & (col >= pqpos - wleft)
                            p_masked.append(
                                valid.select(p_sv[r], fx.Float32(float("-inf")))
                            )
                        p_sv = fx.Vector.from_elements(p_masked, fx.Float32)
                    p_m_old = fx.Float32(state[0])
                    p_m = p_m_old
                    for r in range_constexpr(nsc):
                        p_m = fx.maxnumf(p_m, p_sv[r])
                    for shift in (16, 32):
                        p_m = fx.maxnumf(
                            p_m, p_m.shuffle_xor(fx.Int32(shift), fx.Int32(64))
                        )
                    p_corr = _exp2(p_m_old - p_m)
                    p_rescale = fx.Int64(
                        rocdl.ballot(T.i64, (p_m != p_m_old).ir_value())
                    ) != fx.Int64(0)
                    p_probs = []
                    p_psum = fx.Float32(0.0)
                    for r in range_constexpr(nsc):
                        p = _exp2(p_sv[r] - p_m)
                        p_psum = p_psum + p
                        p_probs.append(p * 240.0)
                    for shift in (16, 32):
                        p_psum = p_psum + p_psum.shuffle_xor(
                            fx.Int32(shift), fx.Int32(64)
                        )
                    p_denom = fx.Float32(state[1]) * p_corr + p_psum
                    p_pfrags = [
                        fx.Vector.from_elements(
                            [
                                _pack4(p_probs[8 * h : 8 * h + 4]),
                                _pack4(p_probs[8 * h + 4 : 8 * h + 8]),
                            ],
                            fx.Int32,
                        ).bitcast(fx.Int64)[0]
                        for h in range(pv_h)
                    ]
                    p_accum = [fx.Vector(state[db + 2]) for db in range(dblocks)]
                    if p_rescale:
                        p_accum = [
                            o * fx.Vector.filled(4, p_corr, fx.Float32) for o in p_accum
                        ]
                    p_next = list(p_accum)
                    for grp in range_constexpr(len(pv_order) // 8):
                        p_nxt = (
                            [
                                p_read_v(db, h)
                                for db, h in pv_order[grp * 8 + 8 : grp * 8 + 16]
                            ]
                            if grp < len(pv_order) // 8 - 1
                            else []
                        )
                        rocdl.sched_barrier(0)
                        for i in range_constexpr(8):
                            db, h = pv_order[grp * 8 + i]
                            p_next[db] = _mfma16(
                                p_vgroup[i].ir_value(),
                                p_pfrags[h].ir_value(),
                                p_next[db],
                            )
                        p_vgroup = p_nxt
                    gpu.barrier()
                    presult = yield [p_m, p_denom] + p_next + p_ahead

                if pqtok < pqlen:
                    p_norm = vscale / fx.Float32(presult[1])
                    p_orow = (fx.Int64(pq0) + fx.Int64(pqtok)) * num_q_heads + fx.Int64(
                        phead
                    )
                    for db in range_constexpr(dblocks):
                        vals = fx.Vector(presult[db + 2])
                        dest = (p_orow * dim + fx.Int64(db * 16 + g * 4)) * 2
                        _store(
                            op, dest, _pack_bf16(vals[0] * p_norm, vals[1] * p_norm), 4
                        )
                        _store(
                            op,
                            dest + 4,
                            _pack_bf16(vals[2] * p_norm, vals[3] * p_norm),
                            4,
                        )
        else:
            # ---- decode role: one split per wave, 16-key tiles, wave-private LDS ----
            u = y - tile_slots
            dseq = u // groups
            dsplit = (u % groups) * 4 + wave
            dq0 = fx.Int32(fx.memref_load(CuQ, dseq))
            dqlen = fx.Int32(fx.memref_load(CuQ, dseq + 1)) - dq0
            dklen = fx.Int32(fx.memref_load(UsedK, dseq))
            if dqlen == 1:
                dbase = wave * slice_bytes
                dvt = dbase + NK * dim
                dlow = dklen - 1 - wleft
                dfirst = (dlow > 0).select(dlow, fx.Int32(0)) // NK
                dtotal = (dklen + NK - 1) // NK
                dper = (dtotal - dfirst + splits - 1) // splits
                dstart = dfirst + dsplit * dper
                dend = (dstart + dper < dtotal).select(dstart + dper, dtotal)
                # MFMA rows past the GQA group repeat its heads; only c < group store.
                dhead = kvh * group + c % group
                dq_frag = [
                    _load(
                        qp,
                        (fx.Int64(dq0) * num_q_heads + fx.Int64(dhead)) * dim
                        + fx.Int64(ks * 32 + g * 8),
                        T.i64,
                        8,
                    )
                    for ks in range(ksteps)
                ]

                def d_fetch(block):
                    base_tok = block * NK
                    kc = [
                        fx.Vector(
                            _load(
                                kp,
                                kv_src(
                                    dseq,
                                    dklen,
                                    base_tok,
                                    (lane * 16 + i * 1024) // dim,
                                    (lane * 16 + i * 1024) % dim,
                                    k_strides,
                                ),
                                fx.Vector.make_type(4, fx.Int32),
                                16,
                            )
                        )
                        for i in range(dim // 64)
                    ]
                    vc = [
                        fx.Vector(
                            _load(
                                vp,
                                kv_src(
                                    dseq,
                                    dklen,
                                    base_tok,
                                    idx // lpr * 4 + g,
                                    c * (dim // 16) + idx % lpr * 16,
                                    v_strides,
                                ),
                                fx.Vector.make_type(4, fx.Int32),
                                16,
                            )
                        )
                        for idx in range(4 * lpr)
                    ]
                    return kc + vc

                @flyc.jit
                def d_fetch_guarded(block):
                    d_chunks = [
                        fx.Vector.filled(4, 0, fx.Int32)
                        for _ in range(dim // 64 + 4 * lpr)
                    ]
                    if block < dend:
                        d_chunks = d_fetch(block)
                    return d_chunks

                def d_commit(chunks):
                    for i in range_constexpr(dim // 64):
                        flat = lane * 16 + i * 1024
                        _store(
                            buf,
                            dbase + _k_offset(flat // dim, flat % dim, dim),
                            chunks[i],
                            16,
                        )
                    for half in range_constexpr(lpr):
                        words, key4s, depths = [], [], []
                        for n4 in range_constexpr(4):
                            idx = half * 4 + n4
                            vval = fx.Vector(chunks[dim // 64 + idx])
                            for wi in range_constexpr(4):
                                words.append(vval[wi])
                                key4s.append(idx // lpr)
                                depths.append(
                                    c * (dim // 16) + idx % lpr * 16 + wi * 4 + g
                                )
                        transpose_store(words, dvt, key4s, depths, NK)

                def d_read_k(ks):
                    return _load(
                        buf, dbase + _k_offset(c, ks * 32 + g * 8, dim), T.i64, 8
                    )

                def d_read_v(db):
                    # Keys 16-31 of the MFMA's depth are zero in both P and V^T.
                    w0 = fx.Int32(
                        _load(buf, dvt + _vt_offset(db * 16 + c, g, NK), T.i32, 4)
                    )
                    return fx.Vector.from_elements([w0, fx.Int32(0)], fx.Int32).bitcast(
                        fx.Int64
                    )[0]

                d_init = (
                    [fx.Float32(-1.0e30), fx.Float32(0.0)]
                    + [fx.Vector.filled(4, 0.0, fx.Float32) for _ in range(dblocks)]
                    + d_fetch_guarded(dstart)
                )
                for block, state in range(dstart, dend, fx.Int32(1), init=d_init):
                    block = fx.Int32(block)
                    rocdl.sched_barrier(0)
                    d_commit(state[2 + dblocks :])
                    _wave_lds_fence()
                    d_ahead = d_fetch_guarded(block + 1)
                    rocdl.sched_barrier(0)
                    d_acc = [fx.Vector.filled(4, 0.0, fx.Float32) for _ in range(2)]
                    d_group = [d_read_k(ks) for ks in range(8)]
                    for grp in range_constexpr(ksteps // 8):
                        d_nxt = (
                            [d_read_k(grp * 8 + 8 + ks) for ks in range(8)]
                            if grp < ksteps // 8 - 1
                            else []
                        )
                        rocdl.sched_barrier(0)
                        for i in range_constexpr(8):
                            ks = grp * 8 + i
                            d_acc[ks % 2] = _mfma16(
                                d_group[i], dq_frag[ks], d_acc[ks % 2]
                            )
                        d_group = d_nxt
                    d_score = d_acc[0] + d_acc[1]
                    d_vgroup = [d_read_v(i) for i in range(8)]
                    rocdl.sched_barrier(0)
                    d_sv = fx.Vector.from_elements(
                        [d_score[r] * log_scale for r in range(4)], fx.Float32
                    )
                    if (block * NK + NK > dklen) | (block * NK < dlow):
                        d_masked = []
                        for r in range_constexpr(4):
                            col = block * NK + g * 4 + r
                            # The query sits at dklen - 1 and sees keys [dlow, dklen).
                            valid = (col < dklen) & (col >= dlow)
                            d_masked.append(
                                valid.select(d_sv[r], fx.Float32(float("-inf")))
                            )
                        d_sv = fx.Vector.from_elements(d_masked, fx.Float32)
                    d_m_old = fx.Float32(state[0])
                    d_m = d_m_old
                    for r in range_constexpr(4):
                        d_m = fx.maxnumf(d_m, d_sv[r])
                    for shift in (16, 32):
                        d_m = fx.maxnumf(
                            d_m, d_m.shuffle_xor(fx.Int32(shift), fx.Int32(64))
                        )
                    d_corr = _exp2(d_m_old - d_m)
                    d_probs = []
                    d_psum = fx.Float32(0.0)
                    for r in range_constexpr(4):
                        p = _exp2(d_sv[r] - d_m)
                        d_psum = d_psum + p
                        d_probs.append(p * 240.0)
                    for shift in (16, 32):
                        d_psum = d_psum + d_psum.shuffle_xor(
                            fx.Int32(shift), fx.Int32(64)
                        )
                    d_denom = fx.Float32(state[1]) * d_corr + d_psum
                    d_pfrag = fx.Vector.from_elements(
                        [_pack4(d_probs), fx.Int32(0)], fx.Int32
                    ).bitcast(fx.Int64)[0]
                    d_accum = [
                        fx.Vector(state[db + 2])
                        * fx.Vector.filled(4, d_corr, fx.Float32)
                        for db in range(dblocks)
                    ]
                    d_next = [None] * dblocks
                    for grp in range_constexpr(dblocks // 8):
                        d_nxt = (
                            [d_read_v(grp * 8 + 8 + i) for i in range(8)]
                            if grp < dblocks // 8 - 1
                            else []
                        )
                        rocdl.sched_barrier(0)
                        for i in range_constexpr(8):
                            db = grp * 8 + i
                            d_next[db] = _mfma16(
                                d_vgroup[i].ir_value(), d_pfrag.ir_value(), d_accum[db]
                            )
                        d_vgroup = d_nxt
                    _wave_lds_fence()
                    dresult = yield [d_m, d_denom] + d_next + d_ahead

                if c < group:
                    pbase = (
                        (fx.Int64(dseq) * num_q_heads + fx.Int64(dhead))
                        * fx.Int64(splits)
                        + fx.Int64(dsplit)
                    ) * stride
                    for db in range_constexpr(dblocks):
                        vals = fx.Vector(dresult[db + 2]) * fx.Vector.filled(
                            4, vscale, fx.Float32
                        )
                        _store(pp, (pbase + fx.Int64(db * 16 + g * 4)) * 4, vals, 16)
                    if g == 0:
                        _store(pp, (pbase + dim) * 4, fx.Float32(dresult[0]), 4)
                        _store(pp, (pbase + dim + 1) * 4, fx.Float32(dresult[1]), 4)

    @flyc.jit
    def launch(
        Q: fx.Tensor,
        K: fx.Tensor,
        V: fx.Tensor,
        O: fx.Tensor,
        Part: fx.Tensor,
        CuQ: fx.Tensor,
        UsedK: fx.Tensor,
        BT: fx.Tensor,
        QD: fx.Tensor,
        KD: fx.Tensor,
        VD: fx.Tensor,
        num_seqs: fx.Int32,
        tile_slots: fx.Int32,
        groups: fx.Int32,
        search_iters: fx.Int32,
        bt_stride: fx.Int32,
        scale: fx.Float32,
        grid_y: fx.Int32,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        attend(
            Q,
            K,
            V,
            O,
            Part,
            CuQ,
            UsedK,
            BT,
            QD,
            KD,
            VD,
            num_seqs,
            tile_slots,
            groups,
            search_iters,
            bt_stride,
            scale,
        ).launch(grid=(num_kv_heads, grid_y, 1), block=(256, 1, 1), stream=stream)

    return launch


@lru_cache(maxsize=4)
def build_flash_attn_fp8_gfx942_combine_module(dim, num_q_heads):
    """Return the launcher that merges split partials for one-token sequences.

    Part holds dim floats (scaled by V's descale), then m and l, per (seq,
    head, split), padded to dim + 4 floats. Multi-token sequences are skipped.
    """
    stride = dim + 4
    vecs = dim // 256

    @flyc.kernel(known_block_size=(64, 1, 1))
    def combine(Part: fx.Tensor, O: fx.Tensor, CuQ: fx.Tensor, num_splits: fx.Int32):
        lane = fx.Int32(gpu.thread_id("x"))
        head = fx.Int32(gpu.block_id("x"))
        seq = fx.Int32(gpu.block_id("y"))
        q0 = fx.Int64(fx.memref_load(CuQ, seq))
        is_decode = (
            fx.Int32(fx.memref_load(CuQ, seq + 1)) - fx.Int32(fx.memref_load(CuQ, seq))
            == 1
        )
        trips = is_decode.select(num_splits, fx.Int32(0))
        pp, op = fx.get_iter(Part), fx.get_iter(O)
        row = (fx.Int64(seq) * num_q_heads + fx.Int64(head)) * fx.Int64(num_splits)
        init = [fx.Float32(-1.0e30), fx.Float32(0.0)] + [
            fx.Vector.filled(4, 0.0, fx.Float32) for _ in range(vecs)
        ]
        for s, state in range(fx.Int32(0), trips, fx.Int32(1), init=init):
            base = (row + fx.Int64(fx.Int32(s))) * stride
            m_s = fx.Float32(_load(pp, (base + dim) * 4, T.f32, 4))
            l_s = fx.Float32(_load(pp, (base + dim + 1) * 4, T.f32, 4))
            m_old = fx.Float32(state[0])
            m_new = fx.maxnumf(m_old, m_s)
            a = _exp2(m_old - m_new)
            b = _exp2(m_s - m_new)
            va = fx.Vector.filled(4, a, fx.Float32)
            vb = fx.Vector.filled(4, b, fx.Float32)
            accs = []
            for i in range_constexpr(vecs):
                o_s = fx.Vector(
                    _load(
                        pp,
                        (base + fx.Int64(lane * (4 * vecs) + 4 * i)) * 4,
                        fx.Vector.make_type(4, fx.Float32),
                        16,
                    )
                )
                accs.append(fx.Vector(state[2 + i]) * va + o_s * vb)
            result = yield [m_new, fx.Float32(state[1]) * a + l_s * b] + accs
        if is_decode:
            inv = 1.0 / fx.Float32(result[1])
            words = []
            for i in range_constexpr(vecs):
                acc = fx.Vector(result[2 + i])
                words += [
                    _pack_bf16(acc[0] * inv, acc[1] * inv),
                    _pack_bf16(acc[2] * inv, acc[3] * inv),
                ]
            dest = (
                (q0 * num_q_heads + fx.Int64(head)) * dim + fx.Int64(lane * (4 * vecs))
            ) * 2
            _store(op, dest, fx.Vector.from_elements(words, fx.Int32), 8 * vecs)

    @flyc.jit
    def launch_combine(
        Part: fx.Tensor,
        O: fx.Tensor,
        CuQ: fx.Tensor,
        batch: fx.Int32,
        num_splits: fx.Int32,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        combine(Part, O, CuQ, num_splits).launch(
            grid=(num_q_heads, batch), block=(64, 1, 1), stream=stream
        )

    return launch_combine
