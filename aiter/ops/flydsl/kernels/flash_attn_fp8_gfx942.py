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
one-token sequence, running a 32-key decode body entirely in registers: K
loads straight into MFMA operands, and V is byte-transposed in place, so
decode waves need no LDS and no barriers. A combine launch merges the splits.

Both GEMMs use mfma_f32_16x16x32_fp8_fp8 in transposed form (S^T = K Q^T,
O^T += V^T P^T), so P needs no cross-lane moves. P is scaled by 240 before
FNUZ packing and the epilogue undoes it. Prefill LDS reads and V-transpose
shuffles are issued in groups of eight behind sched_barriers: without them
the compiler waits out each one before its consumer.
"""

from functools import lru_cache

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T

from aiter.ops.flydsl.kernels import buffer_ops

NK = 32  # decode key tile
NO_WINDOW = 1 << 30


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


def _ir(x):
    return x.ir_value() if hasattr(x, "ir_value") else x


def _mfma16(a, b, c):
    return fx.Vector(
        rocdl.mfma_f32_16x16x32_fp8_fp8(
            fx.Vector.make_type(4, fx.Float32),
            [_ir(a), _ir(b), c.ir_value(), 0, 0, 0],
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


def _transpose4(words):
    """4x4 byte transpose in registers: byte t of word s is byte s of words[t]."""
    lo = [rocdl.perm_b32(words[2 * i + 1], words[2 * i], 0x05010400) for i in range(2)]
    hi = [rocdl.perm_b32(words[2 * i + 1], words[2 * i], 0x07030602) for i in range(2)]
    return [
        fx.Int32(rocdl.perm_b32(pair[1], pair[0], sel))
        for pair in (lo, hi)
        for sel in (0x05040100, 0x07060302)
    ]


def plan_num_kv_splits(num_seqs, max_seqlen_k, num_kv_heads, window, head_dim):
    """Decode split count, fitted to MI325X sweeps (batch 1-256, context
    0.5K-32K): about 128K / head_dim split waves in all, and at least
    head_dim / 128 tiles per split, since each split's partial costs a
    query group's worth of head_dim floats. Below 128 waves the GPU is nearly
    idle, and half that many tiles per split is faster."""
    keys = max_seqlen_k if window is None else min(max_seqlen_k, window)
    tiles = max(1, (keys + NK - 1) // NK)
    units = num_seqs * num_kv_heads
    want = (1 << 17) // head_dim // units
    splits = min(want, tiles * 128 // head_dim)
    if splits * units < 128:
        splits = min(want, tiles * 256 // head_dim)
    return max(1, splits)


def prefill_block_q(num_q_heads, num_kv_heads):
    """Query tokens per prefill workgroup: four waves of 16 // GQA tokens."""
    return 4 * (16 // (num_q_heads // num_kv_heads))


@lru_cache(maxsize=32)
def build_flash_attn_fp8_gfx942_module(
    dim,
    num_q_heads,
    num_kv_heads,
    window,
    page_size,
    kv_strides,
    prefill_keys=None,
    decode_only=False,
):
    """Return the attention launcher for one layer shape and KV layout.

    dim is 256 or 512, the GQA ratio must divide 16, and window is None or the
    inclusive key count. kv_strides holds K's then V's (page, token, head)
    element strides. They are compiled in: runtime strides made head-512
    decode up to 45% slower, and a cache layout fixes them, so each layout
    compiles once. Prefill tiles are 64 keys at head 256 and 32 at head 512; LDS
    holds one tile of K and V^T, 2 * keys * dim bytes. Decode uses no LDS.

    decode_only builds the decode role alone, one wave per workgroup, for
    batches whose sequences all have one query token: a small batch then
    spreads over four times as many CUs. Such a launch has no tile slots.
    """
    if dim not in (256, 512) or page_size not in (32, 64, 128):
        raise ValueError("dim must be 256 or 512 and page size 32, 64 or 128")
    group = num_q_heads // num_kv_heads
    if group * num_kv_heads != num_q_heads or 16 % group:
        raise ValueError("GQA ratio must divide 16")
    tpw = 16 // group
    tokens = 4 * tpw
    ksteps = dim // 32
    dblocks = dim // 16
    lpr = dim // 256
    # 16-byte chunks a decode lane loads per key: K at 64-byte, V at 256-byte steps.
    kchunks = dim // 64
    vchunks = dim // 256
    stride = dim + 4
    wleft = NO_WINDOW if window is None else window - 1
    pn = prefill_keys or (64 if dim == 256 else 32)
    lds_bytes = 2 * pn * dim
    if pn not in (32, 64) or lds_bytes > 65536:
        raise ValueError("prefill_keys must be 32 or 64 and fit in LDS")
    kb_n = pn // 16
    pv_h = pn // 32
    pvt = pn * dim
    k_loads = pn * dim // 4096
    v_loads = pn // 16 * lpr
    k_strides, v_strides = tuple(kv_strides[:3]), tuple(kv_strides[3:])
    # Waves per workgroup; each decode wave runs one split.
    waves = 1 if decode_only else 4

    @fx.struct
    class SharedStorage:
        buf: fx.Array[fx.Int8, lds_bytes, 16]

    @flyc.kernel(known_block_size=(64 * waves, 1, 1))
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
        if const_expr(not decode_only):
            buf = fx.SharedAllocator().allocate(SharedStorage).peek().buf.ptr
        log_scale = (
            fx.Float32(fx.memref_load(QD, 0))
            * fx.Float32(fx.memref_load(KD, 0))
            * scale
            * 1.4426950408889634
        )
        vscale = fx.Float32(fx.memref_load(VD, 0)) / 240.0
        qp, kp, vp, op = fx.get_iter(Q), fx.get_iter(K), fx.get_iter(V), fx.get_iter(O)
        pp, btp = fx.get_iter(Part), fx.get_iter(BT)
        splits = groups * waves
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

        if const_expr(not decode_only):  # noqa: SIM102 - constexpr guard
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
                                buf,
                                _k_offset(flat // dim, flat % dim, dim),
                                chunks[i],
                                16,
                            )
                        words, key4s, depths = [], [], []
                        for idx in range_constexpr(v_loads):
                            vval = fx.Vector(chunks[k_loads + idx])
                            for wi in range_constexpr(4):
                                words.append(vval[wi])
                                key4s.append(wave * (pn // 16) + idx // lpr)
                                depths.append(
                                    c * (dim // 16) + idx % lpr * 16 + wi * 4 + g
                                )
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
                            _load(
                                buf,
                                pvt + _vt_offset(depth, 8 * h + 4 + g, pn),
                                T.i32,
                                4,
                            )
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
                                o * fx.Vector.filled(4, p_corr, fx.Float32)
                                for o in p_accum
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
                        p_orow = (
                            fx.Int64(pq0) + fx.Int64(pqtok)
                        ) * num_q_heads + fx.Int64(phead)
                        for db in range_constexpr(dblocks):
                            vals = fx.Vector(presult[db + 2])
                            dest = (p_orow * dim + fx.Int64(db * 16 + g * 4)) * 2
                            _store(
                                op,
                                dest,
                                _pack_bf16(vals[0] * p_norm, vals[1] * p_norm),
                                4,
                            )
                            _store(
                                op,
                                dest + 4,
                                _pack_bf16(vals[2] * p_norm, vals[3] * p_norm),
                                4,
                            )
        if y >= tile_slots:
            # ---- decode role: one split per wave, 32-key tiles in registers ----
            u = y - tile_slots
            dseq = u // groups
            dsplit = (u % groups) * waves + fx.Int32(rocdl.readfirstlane(T.i32, wave))
            dq0 = fx.Int32(fx.memref_load(CuQ, dseq))
            dqlen = fx.Int32(fx.memref_load(CuQ, dseq + 1)) - dq0
            dklen = fx.Int32(fx.memref_load(UsedK, dseq))
            if dqlen == 1:
                dlow = dklen - 1 - wleft
                dfirst = (dlow > 0).select(dlow, fx.Int32(0)) // NK
                dtotal = (dklen + NK - 1) // NK
                # Splits interleave tiles, so the waves of one sequence sweep
                # neighbouring tiles together.
                dstart = dfirst + dsplit
                dend = dtotal
                dlast = (dstart < dtotal).select(
                    dstart + (dtotal - 1 - dstart) // splits * splits, dtotal - 1
                )
                # MFMA rows past the GQA group repeat its heads; only c < group store.
                dhead = kvh * group + c % group
                # K and Q share a depth order: k-step 2j + h covers depths
                # 64j + 16g + 8h .. +8, so each lane loads 16 contiguous bytes.
                dq_row = (fx.Int64(dq0) * num_q_heads + fx.Int64(dhead)) * dim
                dq_frag = []
                for j in range_constexpr(kchunks):
                    qv = fx.Vector(
                        _load(
                            qp,
                            dq_row + fx.Int64(j * 64 + g * 16),
                            fx.Vector.make_type(4, fx.Int32),
                            16,
                        )
                    ).bitcast(fx.Int64)
                    dq_frag += [qv[0], qv[1]]

                def d_clamp(block):
                    # A lookahead past the split reloads its last tile, an L2 hit.
                    return (block < dend).select(block, dlast)

                def d_page(block):
                    # NK divides the page, so one wave-uniform lookup serves a
                    # tile. A scalar load: a vector one would join the in-order
                    # vmcnt queue behind the K/V loads.
                    return fx.Int32(fx.memref_load(BT, (dseq, block * NK // page_size)))

                def d_load_k(block, page):
                    # Lane (c, g) holds keys c and 16 + c.
                    base = fx.Int64(page) * k_strides[0] + fx.Int64(kvh) * k_strides[2]
                    tok = (block * NK) & (page_size - 1)
                    return [
                        fx.Vector(
                            _load(
                                kp,
                                base
                                + fx.Int64(
                                    (tok + half * 16 + c) * k_strides[1]
                                    + j * 64
                                    + g * 16
                                ),
                                fx.Vector.make_type(4, fx.Int32),
                                16,
                            )
                        )
                        for half in range(2)
                        for j in range(kchunks)
                    ]

                def d_load_v(block, page):
                    # Lane (c, g) holds keys 4g + t and 16 + 4g + t at depths
                    # 256m + 16c .. +16.
                    base = fx.Int64(page) * v_strides[0] + fx.Int64(kvh) * v_strides[2]
                    chunks = []
                    for half in range_constexpr(2):
                        for t in range_constexpr(4):
                            key = block * NK + half * 16 + g * 4 + t
                            # Past dklen, reload a valid key: stale FNUZ bytes can be
                            # NaN, and a zero probability does not cancel a NaN.
                            key = (key < dklen).select(key, block * NK)
                            for m in range_constexpr(vchunks):
                                chunks.append(
                                    fx.Vector(
                                        _load(
                                            vp,
                                            base
                                            + fx.Int64(
                                                (key & (page_size - 1)) * v_strides[1]
                                                + m * 256
                                                + c * 16
                                            ),
                                            fx.Vector.make_type(4, fx.Int32),
                                            16,
                                        )
                                    )
                                )
                    return chunks

                nkc = 2 * kchunks

                def d_scores(block, kc):
                    """QK for one tile, masked to keys [dlow, dklen): the query
                    sits at dklen - 1. Score r is key 16 * (r // 4) + 4g + r % 4
                    of the tile."""
                    d_acc = [fx.Vector.filled(4, 0.0, fx.Float32) for _ in range(4)]
                    for j in range_constexpr(kchunks):
                        for half in range_constexpr(2):
                            kq = fx.Vector(kc[half * kchunks + j]).bitcast(fx.Int64)
                            for h in range_constexpr(2):
                                d_acc[half * 2 + h] = _mfma16(
                                    kq[h], dq_frag[2 * j + h], d_acc[half * 2 + h]
                                )
                    d_s = [d_acc[0] + d_acc[1], d_acc[2] + d_acc[3]]
                    d_sv = []
                    for r in range_constexpr(8):
                        col = block * NK + (r // 4) * 16 + g * 4 + r % 4
                        valid = (col < dklen) & (col >= dlow)
                        d_sv.append(
                            valid.select(
                                d_s[r // 4][r % 4] * log_scale,
                                fx.Float32(float("-inf")),
                            )
                        )
                    return d_sv

                def d_fold(d_sv, vc, m_old, l_old, o_old):
                    """Online softmax and PV for one tile's scores and V."""
                    d_m = m_old
                    for r in range_constexpr(8):
                        d_m = fx.maxnumf(d_m, d_sv[r])
                    for shift in (16, 32):
                        d_m = fx.maxnumf(
                            d_m, d_m.shuffle_xor(fx.Int32(shift), fx.Int32(64))
                        )
                    d_corr = _exp2(m_old - d_m)
                    d_probs = []
                    d_psum = fx.Float32(0.0)
                    for r in range_constexpr(8):
                        p = _exp2(d_sv[r] - d_m)
                        d_psum = d_psum + p
                        d_probs.append(p * 240.0)
                    for shift in (16, 32):
                        d_psum = d_psum + d_psum.shuffle_xor(
                            fx.Int32(shift), fx.Int32(64)
                        )
                    # PV k-group g is keys 4g..4g+3, then 16 + 4g..16 + 4g + 3.
                    d_pfrag = fx.Vector.from_elements(
                        [_pack4(d_probs[0:4]), _pack4(d_probs[4:8])], fx.Int32
                    ).bitcast(fx.Int64)[0]
                    o_new = [
                        fx.Vector(o) * fx.Vector.filled(4, d_corr, fx.Float32)
                        for o in o_old
                    ]
                    # V^T row c of depth block db is depth
                    # 256 * (db // 16) + 16c + db % 16.
                    for m in range_constexpr(vchunks):
                        for jj in range_constexpr(4):
                            vt = [
                                _transpose4(
                                    [
                                        fx.Vector(vc[(half * 4 + t) * vchunks + m])[jj]
                                        for t in range(4)
                                    ]
                                )
                                for half in range(2)
                            ]
                            for s in range_constexpr(4):
                                db = m * 16 + jj * 4 + s
                                o_new[db] = _mfma16(
                                    fx.Vector.from_elements(
                                        [vt[0][s], vt[1][s]], fx.Int32
                                    ).bitcast(fx.Int64)[0],
                                    d_pfrag,
                                    o_new[db],
                                )
                    return d_m, l_old * d_corr + d_psum, o_new

                # Each buffer is refilled for the next tile as soon as this tile
                # is done with it: K after QK, V after PV. So the loop carries no
                # register copies, which would wait out the in-flight loads, and
                # at most one tile's loads are outstanding (vmcnt holds 63). The
                # page lookup runs a tile ahead of the loads.
                d_b0 = d_clamp(dstart)
                d_p0 = d_page(d_b0)
                d_p1 = d_page(d_clamp(dstart + splits))
                # Issue Q, K, V in the loop's order: the wait at the loop head
                # merges this path with the back edge.
                rocdl.sched_barrier(0)
                d_k0 = d_load_k(d_b0, d_p0)
                rocdl.sched_barrier(0)
                d_v0 = d_load_v(d_b0, d_p0)
                rocdl.sched_barrier(0)
                d_init = (
                    [fx.Float32(-1.0e30), fx.Float32(0.0)]
                    + [fx.Vector.filled(4, 0.0, fx.Float32) for _ in range(dblocks)]
                    + [d_p1]
                    + d_k0
                    + d_v0
                )
                for block, state in range(dstart, dend, splits, init=d_init):
                    block = fx.Int32(block)
                    d_nb = d_clamp(block + splits)
                    d_np = fx.Int32(state[2 + dblocks])
                    d_sv = d_scores(block, state[3 + dblocks : 3 + dblocks + nkc])
                    rocdl.sched_barrier(0)
                    d_k_ahead = d_load_k(d_nb, d_np)
                    rocdl.sched_barrier(0)
                    d_m, d_l, d_o = d_fold(
                        d_sv,
                        state[3 + dblocks + nkc :],
                        fx.Float32(state[0]),
                        fx.Float32(state[1]),
                        list(state[2 : 2 + dblocks]),
                    )
                    rocdl.sched_barrier(0)
                    d_v_ahead = d_load_v(d_nb, d_np)
                    d_page_ahead = d_page(d_clamp(block + 2 * splits))
                    rocdl.sched_barrier(0)
                    dresult = yield (
                        [d_m, d_l] + d_o + [d_page_ahead] + d_k_ahead + d_v_ahead
                    )

                if c < group:
                    pbase = (
                        (fx.Int64(dseq) * num_q_heads + fx.Int64(dhead))
                        * fx.Int64(splits)
                        + fx.Int64(dsplit)
                    ) * stride
                    # Row 4g + i of depth block 16m + e is depth 256m + 16(4g + i) + e.
                    for m in range_constexpr(vchunks):
                        for i in range_constexpr(4):
                            for e4 in range_constexpr(4):
                                vals = fx.Vector.from_elements(
                                    [
                                        fx.Vector(dresult[2 + m * 16 + e4 * 4 + k])[i]
                                        * vscale
                                        for k in range(4)
                                    ],
                                    fx.Float32,
                                )
                                depth = m * 256 + (g * 4 + i) * 16 + e4 * 4
                                _store(pp, (pbase + fx.Int64(depth)) * 4, vals, 16)
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
        ).launch(
            grid=(num_kv_heads, grid_y, 1), block=(64 * waves, 1, 1), stream=stream
        )

    if dim == 256:
        # Two head-256 workgroups fit a CU only within 256 registers, which
        # the allocator overshoots by a few without the hint.
        launch.compile_hints = {"waves_per_eu": 2}
    return launch


@lru_cache(maxsize=4)
def build_flash_attn_fp8_gfx942_combine_module(dim, num_q_heads):
    """Return the launcher that merges split partials for one-token sequences.

    Part holds dim floats (scaled by V's descale), then m and l, per (seq,
    head, split), padded to dim + 4 floats. Multi-token sequences are skipped.
    Splits merge eight at a time, all eight loaded before any is merged: the
    merge is cheap and the load latency is not.
    """
    stride = dim + 4
    vecs = dim // 256
    unroll = 8

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
        for sg, state in range(
            fx.Int32(0), (trips + unroll - 1) // unroll, fx.Int32(1), init=init
        ):
            s0 = fx.Int32(sg) * unroll
            ml, os_ = [], []
            rocdl.sched_barrier(0)
            for u in range_constexpr(unroll):
                # Past the last split, reload it; its max is masked to -inf below.
                s = s0 + u
                base = (row + fx.Int64((s < trips).select(s, trips - 1))) * stride
                ml.append(
                    fx.Vector(
                        _load(
                            pp, (base + dim) * 4, fx.Vector.make_type(2, fx.Float32), 8
                        )
                    )
                )
                os_.append(
                    [
                        fx.Vector(
                            _load(
                                pp,
                                (base + fx.Int64(lane * (4 * vecs) + 4 * i)) * 4,
                                fx.Vector.make_type(4, fx.Float32),
                                16,
                            )
                        )
                        for i in range(vecs)
                    ]
                )
            # Without the barrier the scheduler interleaves the merge into the
            # loads and reuses their registers, waiting out each load in turn.
            rocdl.sched_barrier(0)
            m_old = fx.Float32(state[0])
            m_s = [
                (s0 + u < trips).select(ml[u][0], fx.Float32(float("-inf")))
                for u in range(unroll)
            ]
            m_new = m_old
            for u in range_constexpr(unroll):
                m_new = fx.maxnumf(m_new, m_s[u])
            a = _exp2(m_old - m_new)
            w = [_exp2(m_s[u] - m_new) for u in range(unroll)]
            l_new = fx.Float32(state[1]) * a
            accs = [
                fx.Vector(state[2 + i]) * fx.Vector.filled(4, a, fx.Float32)
                for i in range(vecs)
            ]
            for u in range_constexpr(unroll):
                l_new = l_new + ml[u][1] * w[u]
                vw = fx.Vector.filled(4, w[u], fx.Float32)
                accs = [accs[i] + os_[u][i] * vw for i in range(vecs)]
            result = yield [m_new, l_new] + accs
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
