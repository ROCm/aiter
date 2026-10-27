# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Gfx942 FP8 paged unified attention, one varlen launch per call.

Shapes: Gemma-4's head 512 (GQA 8:1, full) and head 256 (GQA 2:1, sliding
window). FP8 E4M3FNUZ Q/K/V with per-tensor FP32 descales, BF16 output.
Pages of 32 or 64 tokens; K/V may be strided views of one cache (vLLM).

Grid y has two regions:
- Tile slots use Triton's q-block map (sequence s owns slots from
  cu_q[s] // tokens + s; binary search, heaviest first). Four waves run the
  prefill body on 64 // GQA query tokens; a prefix chunk is a tile with
  q_len < k_len. Pages are looked up a tile ahead. A lone multi-token
  sequence may split each tile's KV range over psplits adjacent slots that
  write fp32 partials for a prefill combine.
- Decode slots: each wave runs one split of a one-token sequence, 32 keys per
  step, in registers only (K loads into MFMA operands, V byte-transposes in
  place): no LDS, no barriers. A combine launch merges the splits; a
  one-split decode-only launch writes the output itself.

Both GEMMs use mfma_f32_16x16x32_fp8_fp8 transposed (S^T = K Q^T,
O^T += V^T P^T), so P needs no cross-lane moves. P is scaled by 240 before
FNUZ packing; the epilogue undoes it. Prefill LDS reads and V-transpose
shuffles issue in groups of eight behind sched_barriers, else the compiler
waits out each one before its consumer.
"""

from functools import lru_cache

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T

from aiter.ops.flydsl.kernels.kernels_common import LOG2E

NK = 32  # decode key tile
NO_WINDOW = 1 << 30
_BUFFER_COPY = {
    32: fx.rocdl.BufferCopy32b,
    64: fx.rocdl.BufferCopy64b,
    128: fx.rocdl.BufferCopy128b,
}


def _buffer_ptr(tensor, base):
    """Buffer descriptor of `tensor` from element `base` on.

    `base` must be wave-uniform; it is 64-bit, so a descriptor may start past
    4 GiB of a cache, while offsets from it are 32-bit.
    """
    return fx.rocdl.make_buffer_ptr(fx.add_offset(fx.get_iter(tensor), base))


def _rows(tensor, base):
    """`_buffer_ptr(tensor, base)` as a view indexed by element; its extent is
    nominal, as the descriptor spans 4 GiB."""
    return fx.logical_divide(
        fx.make_view(_buffer_ptr(tensor, base), fx.make_layout(1 << 30, 1)),
        fx.make_layout(1, 1),
    )


def _buffer_load(rows, index, dtype, count):
    """`count` values of `dtype` at element `index` of a `_rows` view."""
    atom = fx.make_copy_atom(_BUFFER_COPY[count * dtype.width](), dtype)
    reg = fx.make_rmem_tensor(count, dtype)
    fx.copy(atom, fx.slice(rows, (None, fx.Int32(index))), reg)
    return fx.Vector(reg.load())


def _buffer_store(rows, index, values):
    atom = fx.make_copy_atom(
        _BUFFER_COPY[values.numel * values.dtype.width](), values.dtype
    )
    reg = fx.make_rmem_tensor(values.numel, values.dtype)
    reg.store(values)
    fx.copy(atom, reg, fx.slice(rows, (None, fx.Int32(index))))


def _at(ptr, offset, alignment):
    """`ptr` moved by `offset` bytes, promising `alignment` bytes to LLVM.

    For LDS at XOR-swizzled offsets, and for decode K/V, whose loads must stay
    non-temporal in decode-only launches.
    """
    byte = fx.add_offset(fx.recast_iter(fx.Int8, ptr), offset)
    return fx.recast_iter(
        fx.PointerType.get(fx.Int8.ir_type, byte.memspace, alignment), byte
    )


def _load(ptr, offset, dtype, alignment, count=None, nontemporal=False):
    return fx.generic_load(
        _at(ptr, offset, alignment), dtype=dtype, count=count, nontemporal=nontemporal
    )


def _store(ptr, offset, value, alignment):
    fx.generic_store(_at(ptr, offset, alignment), value)


def _frag_i64(raw):
    """One lane's i64 A or B operand as the register tensor ``fx.gemm`` takes."""
    frag = fx.make_rmem_tensor(1, fx.Int64)
    frag.store(fx.Vector.from_elements([fx.Int64(raw)]))
    return frag


def _mfma16(mma, a, b, c):
    """c + A B for one 16x16x32 FNUZ atom, on one lane's i64 fragments."""
    acc = fx.make_rmem_tensor(4, fx.Float32)
    acc.store(fx.Vector(c))
    fx.gemm(mma, acc, _frag_i64(a), _frag_i64(b), acc)
    return fx.Vector(acc.load())


def _exp2(x):
    # Only afn: masked scores are -inf and must still give exp2 == 0.
    return fx.Float32(fx.exp2(x, fastmath="afn"))


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
    # V^T in 32-depth x `keys`-key chunks, one dword per (depth, 4 keys); the
    # XOR keeps transpose stores and PV reads at 2-way bank conflicts.
    bank = (depth % 32) ^ ((depth // 32) % 8 * 4) ^ (key4 % 4 * 8)
    return (depth // 32) * (32 * keys) + key4 * 128 + bank * 4


def _sload(buf, index, count=1):
    """`count` dwords (1 or 4) at a wave-uniform dword index, as an s_buffer_load."""
    return fx.rocdl.s_buffer_load(buf, index, count=count)


def _transpose4(words):
    """4x4 byte transpose in registers: byte t of word s is byte s of words[t]."""
    lo = [rocdl.perm_b32(words[2 * i + 1], words[2 * i], 0x05010400) for i in range(2)]
    hi = [rocdl.perm_b32(words[2 * i + 1], words[2 * i], 0x07030602) for i in range(2)]
    return [
        fx.Int32(rocdl.perm_b32(pair[1], pair[0], sel))
        for pair in (lo, hi)
        for sel in (0x05040100, 0x07060302)
    ]


def plan_num_kv_splits(
    num_seqs,
    max_seqlen_k,
    num_kv_heads,
    window,
    head_dim,
    decode_only=False,
    num_cus=304,
):
    """Decode split count, fitted on MI325X (batch 1-256, 0.5K-32K keys).

    Aims for about 128K / head_dim split waves in total and at least
    head_dim / 128 tiles per split (each partial costs head_dim floats per
    query head); under 128 waves, half as many tiles per split.
    """
    keys = max_seqlen_k if window is None else min(max_seqlen_k, window)
    tiles = max(1, (keys + NK - 1) // NK)
    units = num_seqs * num_kv_heads
    want = (1 << 17) // head_dim // units
    splits = min(want, tiles * 128 // head_dim)
    if splits * units < 128:
        splits = min(want, tiles * 256 // head_dim)
    # Head-512 decode slows when its waves just overflow a whole number of CU
    # rounds. For 64-511 sequence-heads at 4K+ keys, pick the count (1-4 waves
    # per CU) that fills its rounds best, fewer splits on ties. Mixed batches
    # keep the base model.
    if (
        decode_only
        and window is None
        and head_dim == 512
        and max_seqlen_k >= 4096
        and 64 <= units < 512
    ):

        def fill(s):
            waves = units * s
            return waves / (num_cus * -(-waves // num_cus))

        fits = [s for s in (1, 2, 3, 4, 5, 6, 8) if num_cus <= units * s <= 4 * num_cus]
        if fits:
            splits = max(fits, key=lambda s: (round(fill(s), 2), -s))
        splits = min(splits, tiles)
    return max(1, splits)


def prefill_block_q(num_q_heads, num_kv_heads):
    """Query tokens per prefill workgroup: four waves of 16 // GQA tokens."""
    return 4 * (16 // (num_q_heads // num_kv_heads))


def prefill_block_k(head_dim):
    """Keys per prefill tile."""
    return 64 if head_dim == 256 else 32


# Each prefill split keeps at least this many prefix tiles; all splits' fp32
# partials stay under this many bytes.
PREFILL_SPLIT_MIN_TILES = 2
PREFILL_SPLIT_MAX_BYTES = 64 << 20


def plan_prefill_splits(
    max_seqlen_q,
    max_seqlen_k,
    num_tokens,
    num_q_heads,
    num_kv_heads,
    window,
    head_dim,
    num_cus,
):
    """KV splits for a lone multi-token sequence (prefill or prefix chunk), so
    max_seqlen_q/k are its own lengths.

    A short chunk over a long prefix has too few query tiles to fill the GPU.
    Fitted on MI325X (page 64, 16-1024 queries over 1K-16K keys): about one
    workgroup per CU, at least two prefix tiles per split. A fresh prefill
    never gains: its causal ramp already spreads keys over its tiles.
    """
    tiles = -(-max_seqlen_q // prefill_block_q(num_q_heads, num_kv_heads))
    prefix = max_seqlen_k - max_seqlen_q
    if window is not None:
        prefix = min(prefix, window - 1)
    splits = min(
        num_cus // (tiles * num_kv_heads),
        prefix // prefill_block_k(head_dim) // PREFILL_SPLIT_MIN_TILES,
    )
    row_bytes = num_tokens * num_q_heads * (head_dim + 4) * 4
    return max(1, min(splits, PREFILL_SPLIT_MAX_BYTES // row_bytes))


@lru_cache(maxsize=32)
def build_unified_attention_fp8_gfx942_module(
    dim,
    num_q_heads,
    num_kv_heads,
    window,
    page_size,
    kv_strides,
    decode_only=False,
):
    """Attention launcher for one layer shape and KV layout.

    dim is 256 or 512; the GQA ratio must divide 16; window is None or the
    inclusive key count. kv_strides are K's then V's (page, token, head)
    element strides, compiled in: runtime strides made head-512 decode up to
    45% slower, and a cache layout fixes them. Prefill tiles hold 64 keys at
    head 256 and 32 at head 512; LDS holds one tile of K and V^T
    (2 * keys * dim bytes). Decode uses no LDS.

    decode_only builds only the decode role, one wave per workgroup and no
    tile slots, for all-decode batches: a small batch then spreads over four
    times as many CUs.
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
    # 16-byte chunks a decode lane loads per key: K at 64-byte, V at 256-byte steps.
    kchunks = dim // 64
    vchunks = dim // 256
    stride = dim + 4
    wleft = NO_WINDOW if window is None else window - 1
    pn = prefill_block_k(dim)
    lds_bytes = 2 * pn * dim
    kb_n = pn // 16
    pv_h = pn // 32
    pvt = pn * dim
    k_loads = pn * dim // 4096
    v_loads = pn // 16 * lpr
    # Pages a prefill tile spans. K load i covers 4096 // dim keys from key
    # i * 4096 // dim, and wave w's V loads cover pn // 4 keys from w * pn // 4,
    # so each load sits in one page.
    nslots = max(1, pn // page_size)
    kslot = [i * 4096 // dim // page_size if nslots > 1 else 0 for i in range(k_loads)]
    nchunks = k_loads + v_loads
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
        psplits: fx.Int32,
        groups: fx.Int32,
        search_iters: fx.Int32,
        bt_stride: fx.Int32,
        scale: fx.Float32,
    ):
        mma = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 32, fx.Float8E4M3FNUZ))
        tid = fx.Int32(gpu.thread_id("x"))
        lane = tid % 64
        wave = tid // 64
        c = lane % 16
        g = lane // 16
        kvh = fx.Int32(gpu.block_id("x"))
        y = fx.Int32(gpu.block_id("y"))
        if const_expr(not decode_only):
            buf = fx.SharedAllocator().allocate(SharedStorage).peek().buf.ptr

        def descales():
            """(log2-domain QK scale, V descale over P's 240), loaded by each
            role next to its sequence lengths."""
            log_scale = (
                fx.Float32(fx.memref_load(QD, 0))
                * fx.Float32(fx.memref_load(KD, 0))
                * scale
                * LOG2E
            )
            return log_scale, fx.Float32(fx.memref_load(VD, 0)) * (1.0 / 240.0)

        kp, vp = fx.get_iter(K), fx.get_iter(V)
        splits = groups * waves
        sel_pair = (g % 2 == 0).select(fx.Int32(0x06020400), fx.Int32(0x03070105))
        sel_quad = (g < 2).select(fx.Int32(0x05040100), fx.Int32(0x03020706))

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
            if y < tile_slots * psplits:
                # ---- prefill role ----
                # Scalar loads: they are on every tile's critical path, and a
                # vector load would wait out the vmcnt queue.
                cuq = fx.rocdl.make_buffer_ptr(fx.get_iter(CuQ))
                usedk = fx.rocdl.make_buffer_ptr(fx.get_iter(UsedK))
                # Heaviest tiles first; the KV splits of a tile are adjacent.
                yy = tile_slots - 1 - y // psplits
                psplit = y % psplits
                for _it, bs in range(
                    fx.Int32(0), search_iters, fx.Int32(1), init=[fx.Int32(0), num_seqs]
                ):
                    lo, hi = fx.Int32(bs[0]), fx.Int32(bs[1])
                    mid = (lo + hi) // 2
                    ok = _sload(cuq, mid) // tokens + mid <= yy
                    found = yield [ok.select(mid, lo), ok.select(hi, mid)]
                pseq = fx.Int32(found[0])
                pq0 = _sload(cuq, pseq)
                pqlen = _sload(cuq, pseq + 1) - pq0
                pklen = _sload(usedk, pseq)
                log_scale, vscale = descales()
                ptile = yy - (pq0 // tokens + pseq)
                # One-token sequences normally use decode slots. A lone
                # sequence with an overestimated host max can reach this
                # unified build instead, including with padded Q storage.
                single_lone = (num_seqs == 1) & (pqlen == 1)
                if ((pqlen > 1) | single_lone) & (ptile * tokens < pqlen):
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
                    pqpos = pqbase + pqtok
                    # Q and O rows from the tile's first token at head
                    # kvh * group; lane (c, g) adds its token and head.
                    prow0 = (fx.Int64(pq0) + fx.Int64(ptile * tokens)) * num_q_heads
                    prow0 = prow0 + fx.Int64(kvh * group)
                    prow = (wave * tpw + c // group) * num_q_heads + c % group
                    # A token past the sequence loads the tile's first token.
                    psafe = (pqtok < pqlen).select(prow, c % group)
                    pq_rows = _rows(Q, prow0 * dim)
                    pq_frag = [
                        _buffer_load(
                            pq_rows, psafe * dim + ks * 32 + g * 8, fx.Int32, 2
                        ).bitcast(fx.Int64)[0]
                        for ks in range(ksteps)
                    ]

                    # A tile whose queries see no key has pend <= 0, and a
                    # sequence without keys has pklen == 0. Clamp both to 0 so
                    # their page lookups, which fetch nothing, stay in the row.
                    plast_page = fx.max((pklen - 1) // page_size, fx.Int32(0))
                    plast_blk = fx.max(pend - 1, fx.Int32(0))
                    vslot = fx.Int32(rocdl.readfirstlane(T.i32, wave)) * (pn // 4)
                    vslot = vslot // page_size

                    def p_clamp(block):
                        return fx.min(block, plast_blk)

                    bt_rows = _rows(BT, fx.Int64(pseq) * fx.Int64(bt_stride))

                    def p_pages(block):
                        """Block-table entries of a tile's pages, clamped to the
                        last page. Uniform vector loads queue behind the K/V
                        loads in flight, so waiting on those never waits on
                        these."""
                        first = block * pn // page_size
                        return [
                            _buffer_load(
                                bt_rows,
                                (first + h < plast_page).select(first + h, plast_page),
                                fx.Int32,
                                1,
                            )[0]
                            for h in range(nslots)
                        ]

                    def p_fetch(block, pages):
                        base_tok = block * pn
                        pages = [fx.Int32(rocdl.readfirstlane(T.i32, p)) for p in pages]

                        def bases(strides):
                            return [
                                fx.Int64(p) * strides[0] + fx.Int64(kvh) * strides[2]
                                for p in pages
                            ]

                        def src(key, d, s_tok):
                            # Past klen, reload the page's row at the tile start, a
                            # valid key (pages clamp to the last one): stale FNUZ
                            # bytes can be NaN.
                            tok = base_tok + key
                            off = (tok < pklen).select(tok, base_tok) & (page_size - 1)
                            return off * s_tok + d

                        kb, vb = bases(k_strides), bases(v_strides)
                        k_rows = [_rows(K, base) for base in kb]
                        v_rows = _rows(
                            V,
                            vb[0] if nslots == 1 else (vslot == 0).select(vb[0], vb[1]),
                        )
                        kc = [
                            _buffer_load(
                                k_rows[kslot[i]],
                                src(
                                    (tid * 16 + i * 4096) // dim,
                                    (tid * 16 + i * 4096) % dim,
                                    k_strides[1],
                                ),
                                fx.Int32,
                                4,
                            )
                            for i in range(k_loads)
                        ]
                        vc = [
                            _buffer_load(
                                v_rows,
                                src(
                                    (wave * (pn // 16) + idx // lpr) * 4 + g,
                                    c * (dim // 16) + idx % lpr * 16,
                                    v_strides[1],
                                ),
                                fx.Int32,
                                4,
                            )
                            for idx in range(v_loads)
                        ]
                        return kc + vc

                    @flyc.jit
                    def p_fetch_guarded(block, pages):
                        p_chunks = [
                            fx.Vector.filled(4, 0, fx.Int32) for _ in range(nchunks)
                        ]
                        if block < pend:
                            p_chunks = p_fetch(block, pages)
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
                            buf,
                            _k_offset(kb * 16 + c, ks * 32 + g * 8, dim),
                            fx.Int64,
                            8,
                        )

                    def p_read_v(db, h):
                        # PV MFMA h covers keys 32h..32h+31: key4 8h+g, then 8h+4+g.
                        depth = db * 16 + c
                        w0 = _load(
                            buf, pvt + _vt_offset(depth, 8 * h + g, pn), fx.Int32, 4
                        )
                        w1 = _load(
                            buf, pvt + _vt_offset(depth, 8 * h + 4 + g, pn), fx.Int32, 4
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

                    # This split's tiles: pstart + psplit, then every psplits-th.
                    pfirst_blk = pstart + psplit
                    p_init = (
                        [fx.Float32(-1.0e30), fx.Float32(0.0)]
                        + [fx.Vector.filled(4, 0.0, fx.Float32) for _ in range(dblocks)]
                        + p_fetch_guarded(pfirst_blk, p_pages(p_clamp(pfirst_blk)))
                        + p_pages(p_clamp(pfirst_blk + psplits))
                    )
                    for block, state in range(pfirst_blk, pend, psplits, init=p_init):
                        block = fx.Int32(block)
                        rocdl.sched_barrier(0)
                        p_commit(state[2 + dblocks : 2 + dblocks + nchunks])
                        gpu.barrier()
                        # The next tile's loads stay in flight through this tile's
                        # MFMAs; its pages were looked up a tile earlier, so they
                        # issue without waiting on the block table.
                        p_ahead = p_fetch_guarded(
                            block + psplits, list(state[2 + dblocks + nchunks :])
                        )
                        p_pages_ahead = p_pages(p_clamp(block + 2 * psplits))
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
                                p_acc[j] = _mfma16(
                                    mma, p_group[i], pq_frag[ks], p_acc[j]
                                )
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
                                    mma,
                                    p_vgroup[i].ir_value(),
                                    p_pfrags[h].ir_value(),
                                    p_next[db],
                                )
                            p_vgroup = p_nxt
                        gpu.barrier()
                        presult = yield (
                            [p_m, p_denom] + p_next + p_ahead + p_pages_ahead
                        )

                    if pqtok < pqlen:
                        if psplits == 1:
                            # A row that saw no key has l == 0 and writes zeros.
                            p_l = fx.Float32(presult[1])
                            p_norm = (p_l == 0.0).select(fx.Float32(0.0), vscale / p_l)
                            po_rows = _rows(O, prow0 * dim)
                            for db in range_constexpr(dblocks):
                                vals = fx.Vector(presult[db + 2])
                                words = [
                                    _pack_bf16(vals[0] * p_norm, vals[1] * p_norm),
                                    _pack_bf16(vals[2] * p_norm, vals[3] * p_norm),
                                ]
                                _buffer_store(
                                    po_rows,
                                    prow * dim + db * 16 + g * 4,
                                    fx.Vector.from_elements(words, fx.Int32),
                                )
                        else:
                            # Partials in the decode layout, one row per token
                            # and head; an empty split leaves m at -1e30.
                            pp_rows = _rows(
                                Part,
                                (prow0 * fx.Int64(psplits) + fx.Int64(psplit)) * stride,
                            )
                            plane = prow * psplits * stride
                            vs4 = fx.Vector.filled(4, vscale, fx.Float32)
                            for db in range_constexpr(dblocks):
                                _buffer_store(
                                    pp_rows,
                                    plane + db * 16 + g * 4,
                                    fx.Vector(presult[db + 2]) * vs4,
                                )
                            if g == 0:
                                ml = [fx.Float32(presult[0]), fx.Float32(presult[1])]
                                _buffer_store(
                                    pp_rows,
                                    plane + dim,
                                    fx.Vector.from_elements(ml, fx.Float32),
                                )
        if y >= tile_slots * psplits:
            # ---- decode role: one split per wave, 32-key tiles in registers ----
            u = y - tile_slots * psplits
            dseq = u // groups
            dsplit = (u % groups) * waves + fx.Int32(rocdl.readfirstlane(T.i32, wave))
            dq0 = fx.Int32(fx.memref_load(CuQ, dseq))
            dqlen = fx.Int32(fx.memref_load(CuQ, dseq + 1)) - dq0
            dklen = fx.Int32(fx.memref_load(UsedK, dseq))
            log_scale, vscale = descales()
            if dqlen == 1:
                dlow = dklen - 1 - wleft
                dfirst = (dlow > 0).select(dlow, fx.Int32(0)) // NK
                dtotal = (dklen + NK - 1) // NK
                # Splits interleave tiles, so the waves of one sequence sweep
                # neighbouring tiles together.
                dstart = dfirst + dsplit
                dend = dtotal
                # A sequence without keys (dtotal == 0) clamps to tile 0, not -1.
                dlast = (dstart < dtotal).select(
                    dstart + (dtotal - 1 - dstart) // splits * splits,
                    fx.max(dtotal - 1, fx.Int32(0)),
                )
                # Q and O rows from head kvh * group; lane c holds head
                # c % group. MFMA rows past the group repeat its heads; only
                # c < group store.
                drow0 = (fx.Int64(dq0) * num_q_heads + fx.Int64(kvh * group)) * dim
                drow = c % group * dim
                # K and Q share a depth order: k-step 2j + h covers depths
                # 64j + 16g + 8h .. +8, so each lane loads 16 contiguous bytes.
                dq_rows = _rows(Q, drow0)
                dq_frag = []
                for j in range_constexpr(kchunks):
                    qv = _buffer_load(
                        dq_rows, drow + j * 64 + g * 16, fx.Int32, 4
                    ).bitcast(fx.Int64)
                    dq_frag += [qv[0], qv[1]]

                def d_clamp(block):
                    # A lookahead past the split reloads its last tile, an L2 hit.
                    return (block < dend).select(block, dlast)

                def d_page(block):
                    # NK divides the page, so one uniform lookup serves a tile.
                    # Scalar: a vector load would queue behind the K/V loads.
                    return fx.Int32(fx.memref_load(BT, (dseq, block * NK // page_size)))

                # Decode-only launches load K and V non-temporal (each key is
                # read once); in a unified launch the hint measured slower
                # beside the prefill tiles.
                def d_load_k(block, page):
                    # Lane (c, g) holds keys c and 16 + c.
                    # Widen before multiplying: a token stride that fits in i32
                    # still overflows the product (token 16 * 2**27 == 2**31).
                    base = fx.Int64(page) * k_strides[0] + fx.Int64(kvh) * k_strides[2]
                    tok = (block * NK) & (page_size - 1)
                    return [
                        _load(
                            kp,
                            base
                            + fx.Int64(tok + half * 16 + c) * fx.Int64(k_strides[1])
                            + fx.Int64(j * 64 + g * 16),
                            fx.Int32,
                            16,
                            count=4,
                            nontemporal=decode_only,
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
                                    _load(
                                        vp,
                                        base
                                        + fx.Int64(key & (page_size - 1))
                                        * fx.Int64(v_strides[1])
                                        + fx.Int64(m * 256 + c * 16),
                                        fx.Int32,
                                        16,
                                        count=4,
                                        nontemporal=decode_only,
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
                                    mma,
                                    kq[h],
                                    dq_frag[2 * j + h],
                                    d_acc[half * 2 + h],
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
                                    mma,
                                    fx.Vector.from_elements(
                                        [vt[0][s], vt[1][s]], fx.Int32
                                    ).bitcast(fx.Int64)[0],
                                    d_pfrag,
                                    o_new[db],
                                )
                    return d_m, l_old * d_corr + d_psum, o_new

                # Each buffer is refilled for the next tile once this tile is
                # done with it (K after QK, V after PV): the loop carries no
                # register copies, which would wait out in-flight loads, and at
                # most one tile's loads are outstanding (vmcnt holds 63). Pages
                # are looked up a tile ahead.
                d_b0 = d_clamp(dstart)
                # Without keys, the prologue loads read physical page 0: the
                # sequence's block-table entries need not name a valid page.
                d_p0 = (dtotal > 0).select(d_page(d_b0), fx.Int32(0))
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

                def d_rows(norm):
                    """(depth, four scaled values) per output chunk: row 4g + i
                    of depth block 16m + e is depth 256m + 16(4g + i) + e."""
                    return [
                        (
                            m * 256 + (g * 4 + i) * 16 + e4 * 4,
                            [
                                fx.Vector(dresult[2 + m * 16 + e4 * 4 + k])[i] * norm
                                for k in range(4)
                            ],
                        )
                        for m in range(vchunks)
                        for i in range(4)
                        for e4 in range(4)
                    ]

                def d_write_partials():
                    dp_rows = _rows(
                        Part,
                        (
                            (fx.Int64(dseq) * num_q_heads + fx.Int64(kvh * group))
                            * fx.Int64(splits)
                            + fx.Int64(dsplit)
                        )
                        * stride,
                    )
                    dlane = c % group * splits * stride
                    for depth, vals in d_rows(vscale):
                        _buffer_store(
                            dp_rows,
                            dlane + depth,
                            fx.Vector.from_elements(vals, fx.Float32),
                        )
                    if g == 0:
                        ml = [fx.Float32(dresult[0]), fx.Float32(dresult[1])]
                        _buffer_store(
                            dp_rows,
                            dlane + dim,
                            fx.Vector.from_elements(ml, fx.Float32),
                        )

                def d_write_out():
                    do_rows = _rows(O, drow0)
                    # A sequence without keys has l == 0 and writes zeros.
                    d_l = fx.Float32(dresult[1])
                    norm = (d_l == 0.0).select(fx.Float32(0.0), vscale / d_l)
                    for depth, vals in d_rows(norm):
                        words = [
                            _pack_bf16(vals[0], vals[1]),
                            _pack_bf16(vals[2], vals[3]),
                        ]
                        _buffer_store(
                            do_rows,
                            drow + depth,
                            fx.Vector.from_elements(words, fx.Int32),
                        )

                if c < group:
                    if const_expr(decode_only):
                        if splits == 1:
                            d_write_out()
                        else:
                            d_write_partials()
                    else:
                        d_write_partials()

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
        psplits: fx.Int32,
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
            psplits,
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


@lru_cache(maxsize=8)
def build_unified_attention_fp8_gfx942_combine_module(dim, num_q_heads, prefill=False):
    """Launcher that merges split partials.

    Part holds, per (row, head, split), dim floats scaled by V's descale, then
    m and l, padded to dim + 4. By default grid y is a sequence; only
    one-token sequences are merged. With prefill, grid y is a query token of a
    lone multi-token sequence (tokens past it are skipped). Splits merge eight
    at a time, all eight loaded first: load latency, not the merge, dominates.
    """
    stride = dim + 4
    vecs = dim // 256
    unroll = 8

    @flyc.kernel(known_block_size=(64, 1, 1))
    def combine(Part: fx.Tensor, O: fx.Tensor, CuQ: fx.Tensor, num_splits: fx.Int32):
        lane = fx.Int32(gpu.thread_id("x"))
        head = fx.Int32(gpu.block_id("x"))
        y = fx.Int32(gpu.block_id("y"))
        if const_expr(prefill):
            q0 = fx.Int64(y)
            valid = y < fx.Int32(fx.memref_load(CuQ, 1))
        else:
            q0 = fx.Int64(fx.memref_load(CuQ, y))
            valid = (
                fx.Int32(fx.memref_load(CuQ, y + 1)) - fx.Int32(fx.memref_load(CuQ, y))
                == 1
            )
        trips = valid.select(num_splits, fx.Int32(0))
        part_base = (
            (fx.Int64(y) * num_q_heads + fx.Int64(head)) * fx.Int64(num_splits) * stride
        )
        part_rows = _rows(Part, part_base)
        # m and l are wave-uniform: scalar loads keep them off the vector queue.
        part_ml = _buffer_ptr(Part, part_base)
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
                base = (s < trips).select(s, trips - 1) * stride
                # m, l and the row's two pad floats.
                ml.append(fx.Vector(_sload(part_ml, base + dim, 4)).bitcast(fx.Float32))
                os_.append(
                    [
                        _buffer_load(
                            part_rows, base + lane * (4 * vecs) + 4 * i, fx.Float32, 4
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
        if valid:
            # A row that saw no key in any split has l == 0 and writes zeros.
            l_sum = fx.Float32(result[1])
            inv = (l_sum == 0.0).select(fx.Float32(0.0), 1.0 / l_sum)
            words = []
            for i in range_constexpr(vecs):
                acc = fx.Vector(result[2 + i])
                words += [
                    _pack_bf16(acc[0] * inv, acc[1] * inv),
                    _pack_bf16(acc[2] * inv, acc[3] * inv),
                ]
            o_rows = _rows(O, (q0 * num_q_heads + fx.Int64(head)) * dim)
            _buffer_store(
                o_rows, lane * (4 * vecs), fx.Vector.from_elements(words, fx.Int32)
            )

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
