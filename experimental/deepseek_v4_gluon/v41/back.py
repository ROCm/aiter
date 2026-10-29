# SPDX-License-Identifier: MIT
"""The DS V4.1 attention back (K2's first half) in Gluon: ``attention/back.py``
(vLLM 8321d11), one persistent launch, 256 CTAs x 512 threads, S rows:

    split   (S x chunks of CK keys): flash-decoding over one chunk of a token's
            keys (K1's key table: top-k compressed rows, then window rows) for
            every head of the rank -> the chunk's bf16 partial and its (max,
            sum) per head, published (SRDY)
    combine (S x heads / 8): a (token, head)'s chunks merged with the sink, bf16,
            inverse GPT-J RoPE of dims 448.., vLLM's MXFP8 of each 32 -> XO
    wo_a    (groups x 1024 / 32): XO rows x wo_a -> bf16 -> MXFP8 of the 32 -> X8B
    wo_b    (320 x 16 rows): X8B x wo_b -> bf16 pairs to every rank's ATTN region

Split, as upstream lays it out: wave (head group hg, key tile kt) -- 16 keys of
the chunk -- loads its keys' fp8_ds_mla records (448 e4m3 + 64 bf16 rope + 7
E8M0 codes) once, dequantizes them into its QK A operands (16x16x32 bf16
MFMA; the 32-dim chunk 2 c + h of the nope dims is {64 c + 16 g + 8 h ..},
lane group g, exactly upstream's K positions: the same sums bit for bit), q in
LDS the B operand. Scores -> max (any order: exact), p = exp2(s - m), the
16-key sums in upstream's tree, the tiles' sums in tile order (LDS), bf16 p ->
LDS. PV (16x16x16 bf16): wave w the dims 64 w + 4 r + dt (dt: four MFMAs), the
raw records (kept in LDS) dequantized again, the rope wave (w = 7) the bf16
dims. Partials bf16 [t][chunk][head][512], (max, sum) f32.

Every published store is write-through (``.wt``: sc0 sc1): a plain store stays in
its XCD's L2, where a consumer CTA on another XCD never sees it (upstream:
``CM_DEV``). Consumers read plain: the lines are new to their L2."""

from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from ..common.mx import clamp_fp8, fp8_pack4, vllm_mx_code
from ..common.ops import bf16_pair, bf_hi, bf_lo, hw_exp2, lane_gather, xshfl
from ..common.sync import SMEM_FLAT, T512, fail, load_dev32, poll1, publish, put
from .seam import attn_pair
from .moe_shared import SH1, gather_nd

NOPE = gl.constexpr(448)
ROPE = gl.constexpr(64)
HEAD_DIM = gl.constexpr(512)
KEYS = gl.constexpr(640)  # top-k 512 + window 128
DATA = gl.constexpr(576)
NEG = gl.constexpr(-3.4028234663852886e38)
LOG2E = gl.constexpr(1.4426950408889634)

# PV: C [128 = 16 w + r, 16 heads] a dt, wave w rows 16 w ..
MMA_PV = gl.constexpr(gl.amd.AMDMFMALayout(4, [16, 16, 16], False, [8, 1]))
A_PV = gl.constexpr(gl.DotOperandLayout(0, MMA_PV, 4))
B_PV = gl.constexpr(gl.DotOperandLayout(1, MMA_PV, 4))
# the query in LDS: [2 c + h][head tile][16 heads][32]: c < 7 nope chunk (c, h)
# = dims 64 c + 16 (k / 8) + 8 h + k % 8; c = 7 rope chunk h = dims 448 + 32 h
# + k (built as [c][h][..] in registers, Q5). LDS views are indexed through a
# rank-lower layout (``_reinterpret``), as Gluon requires.
Q5 = gl.constexpr(gl.BlockedLayout([1, 1, 1, 1, 8], [1, 1, 1, 16, 4], [8, 1, 1, 1, 1],
                                   [4, 3, 2, 1, 0]))  # fmt: skip
SH5 = gl.constexpr(gl.SwizzledSharedLayout(1, 1, 1, [4, 3, 2, 1, 0]))
SH4 = gl.constexpr(gl.SwizzledSharedLayout(1, 1, 1, [3, 2, 1, 0]))
SH3 = gl.constexpr(gl.SwizzledSharedLayout(1, 1, 1, [2, 1, 0]))
SH2 = gl.constexpr(gl.SwizzledSharedLayout(1, 1, 1, [1, 0]))


# ---------------------------------------------------------------- plan
@gluon.constexpr_function
def p2(n):
    return 1 << (n - 1).bit_length()


@gluon.constexpr_function
def head_groups(heads, s):
    """Upstream ``head_groups``: two wave groups (a head tile each, 64-key
    chunks) while the units fit the grid, else one."""
    return 2 if heads // 16 == 2 and s * (-(-640 // 64)) <= 256 else 1


@gluon.constexpr_function
def chunk_keys(heads, s):
    return 16 * (8 // head_groups(heads, s))


@gluon.constexpr_function
def n_chunks(heads, s):
    return -(-640 // chunk_keys(heads, s))


@gluon.constexpr_function
def qk_mma(ng):
    return gl.amd.AMDMFMALayout(4, [16, 16, 32], False, [ng, 8 // ng, 1])


@gluon.constexpr_function
def qk_a(ng):
    return gl.DotOperandLayout(0, qk_mma(ng), 8)


@gluon.constexpr_function
def qk_b(ng):
    return gl.DotOperandLayout(1, qk_mma(ng), 8)


@gluon.constexpr_function
def key_ld(ng):
    """A key tile's bytes [group][keys][32] in the QK A operand's lane order
    (lane = key + 16 kq, 8 bytes a lane): the conversion is free."""
    return gl.BlockedLayout([1, 1, 8], [1, 16, 4], [ng, 8 // ng, 1], [1, 2, 0])


# ---------------------------------------------------------------- keys
@gluon.jit
def _ar3(n0: gl.constexpr, n1: gl.constexpr, n2: gl.constexpr, L: gl.constexpr):
    a = gl.arange(0, n0, layout=gl.SliceLayout(1, gl.SliceLayout(2, L)))
    b = gl.arange(0, n1, layout=gl.SliceLayout(0, gl.SliceLayout(2, L)))
    c = gl.arange(0, n2, layout=gl.SliceLayout(0, gl.SliceLayout(1, L)))
    return a[:, None, None], b[None, :, None], c[None, None, :]


@gluon.jit
def key_rec(kt, t, k, swa, swa_stride, swa_block, comp, comp_stride, comp_block, zrec,
            RATIO: gl.constexpr, DEV: gl.constexpr):  # fmt: skip
    """(data pointer, scale pointer, valid) of token t's key k (any shape) from
    K1's key table: slot | 1 << 31 a compressed row, a window slot otherwise,
    -1 none (a zero record: masked scores, zero values). ``DEV``: a table this
    launch wrote, read past the CU's cache."""
    at = t * KEYS + gl.minimum(gl.maximum(k, 0), KEYS - 1)
    if DEV:
        val = load_dev32(kt.to(gl.uint64) + (at * 4).to(gl.uint64)).to(gl.int32, bitcast=True)
    else:
        val = gl.load(kt.to(gl.pointer_type(gl.int32)) + at)
    ok = val != -1
    slot = gl.where(ok, val & 0x7FFFFFFF, 0)
    if RATIO > 0:
        is_c = ok & (val < 0)
        blk = gl.where(is_c, comp_block, swa_block)
        stride = gl.where(is_c, comp_stride, swa_stride)
        page = gl.where(is_c, comp, swa) + (slot // blk).to(gl.int64) * stride
    else:
        blk = swa_block
        page = swa + (slot // swa_block).to(gl.int64) * swa_stride
    off = slot % blk
    data = gl.where(ok, page + off * DATA, zrec)
    sc = gl.where(ok, page + blk * DATA + off * 8, zrec + DATA)
    return data, sc, ok


@gluon.jit
def pow2(code):
    return (code.to(gl.int32) << 23).to(gl.float32, bitcast=True)


@gluon.jit
def fp8_f32(b):
    """E4M3 bytes -> f32."""
    return b.to(gl.float8e4nv, bitcast=True).to(gl.float32)


@gluon.jit
def await_key_table(krdy, t, tag, tid, dl, health, rank):
    """An indexed build: token t's key table (this launch's ``stage_kt``) written."""
    first = tid == 0
    _, f = poll1(krdy.to(gl.uint64) + (t * 8 + tid * 0).to(gl.uint64), tag, first, dl, health)
    fail(health, f, rank, 23, first)
    gl.barrier()


# ---------------------------------------------------------------- split
@gluon.jit
def stage_split(unit, q, kt, klen, swa, swa_stride, swa_block, comp, comp_stride, comp_block,
                zrec, qk_scale, part, pstat, srdy, krdy, q_s, raw_n, raw_r, raw_s, p_s, st_s, tag,
                tid, dl, health, rank, S: gl.constexpr, HEADS: gl.constexpr,
                RATIO: gl.constexpr, DEV: gl.constexpr):  # fmt: skip
    """Flash-decoding over chunk ``unit % NCHUNK`` of token ``unit // NCHUNK``,
    every head of the rank -> the chunk's partial and (max, sum) a head, SRDY."""
    NG: gl.constexpr = head_groups(HEADS, S)
    CK: gl.constexpr = chunk_keys(HEADS, S)
    NCHUNK: gl.constexpr = n_chunks(HEADS, S)
    NKT: gl.constexpr = 8 // NG  # key tiles of a group (a wave each)
    CKG: gl.constexpr = NKT * 16  # == CK
    HT: gl.constexpr = HEADS // 16
    NHT: gl.constexpr = HT // NG  # head tiles a wave
    KL: gl.constexpr = key_ld(NG)
    MMA: gl.constexpr = qk_mma(NG)
    AQ: gl.constexpr = qk_a(NG)
    BQ: gl.constexpr = qk_b(NG)
    gl.static_assert(CKG == CK)
    t = unit // NCHUNK
    ch = unit % NCHUNK
    k0 = ch * CK
    if DEV:
        await_key_table(krdy, t, tag, tid, dl, health, rank)
        kv_len = gl.max(load_dev32(klen.to(gl.uint64) + (t * 4 + gl.zeros_like(tid)).to(gl.uint64)
                                   ).to(gl.int32, bitcast=True), axis=0)  # fmt: skip
    else:
        kv_len = gl.load(klen.to(gl.pointer_type(gl.int32)) + t)
    if k0 < kv_len:
        # this lane's key: group g_ (any), key k0 + kr, its record
        g_, kr, kk = _ar3(NG, CKG, 32, KL)
        key = k0 + kr + g_ * 0  # [NG][CK][1]: each group its copy
        data, scp, okv = key_rec(kt, t, key, swa, swa_stride, swa_block, comp, comp_stride,
                                 comp_block, zrec, RATIO, DEV)  # fmt: skip
        # q of every head -> LDS (chunk-permuted, Q5)
        c5 = gl.arange(0, 8, layout=gl.SliceLayout(1, gl.SliceLayout(2, gl.SliceLayout(3, gl.SliceLayout(4, Q5)))))
        h5 = gl.arange(0, 2, layout=gl.SliceLayout(0, gl.SliceLayout(2, gl.SliceLayout(3, gl.SliceLayout(4, Q5)))))
        t5 = gl.arange(0, HT, layout=gl.SliceLayout(0, gl.SliceLayout(1, gl.SliceLayout(3, gl.SliceLayout(4, Q5)))))
        r5 = gl.arange(0, 16, layout=gl.SliceLayout(0, gl.SliceLayout(1, gl.SliceLayout(2, gl.SliceLayout(4, Q5)))))
        k5 = gl.arange(0, 32, layout=gl.SliceLayout(0, gl.SliceLayout(1, gl.SliceLayout(2, gl.SliceLayout(3, Q5)))))
        c5 = c5[:, None, None, None, None]
        h5 = h5[None, :, None, None, None]
        t5 = t5[None, None, :, None, None]
        r5 = r5[None, None, None, :, None]
        k5 = k5[None, None, None, None, :]
        qdim = gl.where(c5 < 7, 64 * c5 + 16 * (k5 // 8) + 8 * h5 + k5 % 8, NOPE + 32 * h5 + k5)
        qv = gl.load(q.to(gl.pointer_type(gl.bfloat16)) + t * (HEADS * HEAD_DIM)
                     + (t5 * 16 + r5) * HEAD_DIM + qdim)  # fmt: skip
        # the record's nope chunks (8 B a lane a chunk), rope chunks (16 B), codes
        # (raw_s keeps byte kk % 8 at kk: code w of the 64 dims of PV wave w)
        scb = scp.to(gl.pointer_type(gl.uint8))
        scw = gl.load(scb + kk % 8)  # [NG][CK][32]
        nopes = ()
        for c in gl.static_range(7):
            for h in gl.static_range(2):
                nopes = nopes + (gl.load(data.to(gl.pointer_type(gl.uint8))
                                         + (64 * c + 16 * (kk // 8) + 8 * h + kk % 8)),)  # fmt: skip
        ropes = ()
        for cc in gl.static_range(2):
            ropes = ropes + (gl.load((data + NOPE).to(gl.pointer_type(gl.bfloat16)) + (32 * cc + kk)),)
        q_s.store(gl.reshape(qv, (16, HT, 16, 32)))
        qv3 = q_s._reinterpret(gl.bfloat16, [16, HT, 16, 32], SH3)
        # the raw records -> LDS for the PV (group copies alike)
        rn3 = raw_n._reinterpret(gl.uint8, [16, NG, CKG, 32], SH3)
        for c in gl.static_range(7):
            for h in gl.static_range(2):
                rn3.index(2 * c + h).store(nopes[2 * c + h])
        rr3 = raw_r._reinterpret(gl.bfloat16, [2, NG, CKG, 32], SH3)
        for cc in gl.static_range(2):
            rr3.index(cc).store(ropes[cc])
        raw_s.store(scw)
        gl.barrier()
        # QK: a head tile's [group][keys][16 heads] scores
        scale = qk_scale.to(gl.float32, bitcast=True)
        accs = ()
        for ht in gl.static_range(NHT):
            acc = gl.zeros((NG, CKG, 16), gl.float32, MMA)
            for c in gl.static_range(7):
                ksc = pow2(gl.load(scb + c))  # code c: dims 64 c .. (a key's [.][.][1])
                for h in gl.static_range(2):
                    kw = (fp8_f32(nopes[2 * c + h]) * ksc).to(gl.bfloat16)
                    if NG == 1:
                        bv = qv3.index(2 * c + h).slice(ht, 1, 0).permute((0, 2, 1)).load(BQ)
                    else:
                        bv = qv3.index(2 * c + h).permute((0, 2, 1)).load(BQ)
                    acc = gl.amd.cdna4.mfma(gl.convert_layout(kw, AQ), bv, acc)
            for cc in gl.static_range(2):
                if NG == 1:
                    bv = qv3.index(14 + cc).slice(ht, 1, 0).permute((0, 2, 1)).load(BQ)
                else:
                    bv = qv3.index(14 + cc).permute((0, 2, 1)).load(BQ)
                acc = gl.amd.cdna4.mfma(gl.convert_layout(ropes[cc], AQ), bv, acc)
            accs = accs + (acc,)
        # scores -> the chunk's max a head (exact in any order), p, the 16-key
        # sums in upstream's tree (key = 16 kt + 8 a' + 4 b' + 2 a + b: sum a,
        # b in-lane, then a' (xor 32), b' (xor 16)), bf16 p -> LDS
        valid = gl.convert_layout((key < kv_len) & okv, MMA)
        for ht in gl.static_range(NHT):
            sv = gl.where(valid, accs[ht] * scale, NEG)
            m = gl.max(sv, axis=1)
            p = gl.where(valid, hw_exp2(sv - m[:, None, :]), 0.0)
            p7 = gl.reshape(p, (NG, NKT, 2, 2, 2, 2, 16))
            lt = gl.sum(gl.sum(gl.sum(gl.sum(p7, axis=4), axis=4), axis=2), axis=2)
            if NG == 1:
                p_s.slice(ht, 1, 0).permute((0, 2, 1)).store(p.to(gl.bfloat16))
            else:
                p_s.permute((0, 2, 1)).store(p.to(gl.bfloat16))
            # the max a head, the tiles' sums a head, by head index
            mg = gl.arange(0, NG, layout=gl.SliceLayout(1, m.type.layout))[:, None]
            mh = gl.arange(0, 16, layout=gl.SliceLayout(0, m.type.layout))[None, :]
            if NG == 1:
                mhead = ht * 16 + mh + mg * 0
            else:
                mhead = mg * 16 + mh
            st_s.scatter(gl.reshape(m, [NG * 16]), gl.reshape(mhead, [NG * 16]), 0)
            lg = gl.arange(0, NG, layout=gl.SliceLayout(1, gl.SliceLayout(2, lt.type.layout)))[:, None, None]
            lk = gl.arange(0, NKT, layout=gl.SliceLayout(0, gl.SliceLayout(2, lt.type.layout)))[None, :, None]
            lh = gl.arange(0, 16, layout=gl.SliceLayout(0, gl.SliceLayout(1, lt.type.layout)))[None, None, :]
            if NG == 1:
                lhead = ht * 16 + lh + lg * 0
            else:
                lhead = lg * 16 + lh
            # l of (head, tile) at HEADS + head * NKT + tile
            st_s.scatter(gl.reshape(lt, [NG * NKT * 16]),
                         gl.reshape(HEADS + lhead * NKT + lk, [NG * NKT * 16]), 0)  # fmt: skip
        gl.barrier()
        # chunk stats: (max, the tiles' sums in tile order) a head
        hd = gl.minimum(tid, HEADS - 1)
        mm = st_s.gather(hd, 0)
        ll = st_s.gather(HEADS + hd * NKT, 0)
        for j in gl.static_range(1, NKT):
            ll = ll + st_s.gather(HEADS + hd * NKT + j, 0)
        sp = pstat.to(gl.pointer_type(gl.float32)) + ((t * NCHUNK + ch) * HEADS + hd) * 2
        gl.store(sp, mm, mask=tid < HEADS, cache_modifier=".wt")
        gl.store(sp + 1, ll, mask=tid < HEADS, cache_modifier=".wt")
        # PV: A [128 = 16 w + r][16 keys] a dt plane -- dims 64 w + 4 r + dt of the
        # raw records (wave 7: the rope dims, bf16) --, B the p tile
        rw = gl.arange(0, 128, layout=gl.SliceLayout(1, A_PV))[:, None]
        kq = gl.arange(0, 16, layout=gl.SliceLayout(0, A_PV))[None, :]
        w = rw // 16
        r = rw % 16
        rn32 = raw_n._reinterpret(gl.uint32, [16 * NG * CKG * 8], SH1)
        rr32 = raw_r._reinterpret(gl.uint32, [2 * NG * CKG * 16], SH1)
        rs8 = raw_s._reinterpret(gl.uint8, [NG * CKG * 32], SH1)
        for ht2 in gl.static_range(HT):
            c0 = gl.zeros((128, 16), gl.float32, MMA_PV)
            c1 = gl.zeros((128, 16), gl.float32, MMA_PV)
            c2 = gl.zeros((128, 16), gl.float32, MMA_PV)
            c3 = gl.zeros((128, 16), gl.float32, MMA_PV)
            for kt2 in gl.static_range(NKT):
                kl = kt2 * 16 + kq
                # nope dims 64 w + 4 r .. +4: chunk (w, (r % 4) / 2), bytes
                # 8 (r / 4) + 4 (r % 2) .. of it (group 0's copy)
                nw = gather_nd(rn32, ((w * 2 + (r % 4) // 2) * NG * CKG + kl) * 8
                               + 2 * (r // 4) + r % 2)  # fmt: skip
                ksc = pow2(gather_nd(rs8, kl * 32 + gl.minimum(w, 6)))
                # rope dims 448 + 4 r + dt: chunk r / 8, element 4 (r % 8) + dt
                rlo = gather_nd(rr32, ((r // 8) * NG * CKG + kl) * 16 + 2 * (r % 8))
                rhi = gather_nd(rr32, ((r // 8) * NG * CKG + kl) * 16 + 2 * (r % 8) + 1)
                bv = (p_s._reinterpret(gl.bfloat16, [HT, 16, CK], SH2).index(ht2)
                      .slice(kt2 * 16, 16, 1).permute((1, 0)).load(B_PV))  # fmt: skip
                for dt in gl.static_range(4):
                    nv = fp8_f32(((nw >> (8 * dt)) & 0xFF).to(gl.uint8)) * ksc
                    if dt == 0:
                        rv = bf_lo(rlo)
                    elif dt == 1:
                        rv = bf_hi(rlo)
                    elif dt == 2:
                        rv = bf_lo(rhi)
                    else:
                        rv = bf_hi(rhi)
                    av = gl.where(w == 7, rv, nv).to(gl.bfloat16)
                    if dt == 0:
                        c0 = gl.amd.cdna4.mfma(av, bv, c0)
                    elif dt == 1:
                        c1 = gl.amd.cdna4.mfma(av, bv, c1)
                    elif dt == 2:
                        c2 = gl.amd.cdna4.mfma(av, bv, c2)
                    else:
                        c3 = gl.amd.cdna4.mfma(av, bv, c3)
            # partial[t][chunk][head][dim] bf16: C row 16 w + r, plane dt is dim
            # 4 (16 w + r) + dt
            jv = gl.permute(gl.join(gl.join(c0, c1), gl.join(c2, c3)), (1, 0, 3, 2))
            pv = gl.reshape(jv, (16, 512)).to(gl.bfloat16)
            hh = gl.arange(0, 16, layout=gl.SliceLayout(1, pv.type.layout))[:, None]
            dd = gl.arange(0, 512, layout=gl.SliceLayout(0, pv.type.layout))[None, :]
            gl.store(part.to(gl.pointer_type(gl.bfloat16))
                     + ((t * NCHUNK + ch) * HEADS + ht2 * 16 + hh) * HEAD_DIM + dd, pv,
                     cache_modifier=".wt")  # fmt: skip
        publish(srdy, gl.zeros_like(tid) + unit, gl.zeros_like(tid) + 1, tag, tid == 0)


# ---------------------------------------------------------------- combine
# a combine unit: 8 heads (a wave each), 8 dims a lane
C3 = gl.constexpr(gl.BlockedLayout([1, 1, 8], [1, 64, 1], [8, 1, 1], [2, 1, 0]))


@gluon.jit
def stage_combine(unit, klen, pos, sink, cos_sin, part, pstat, srdy, krdy, xo, xos, crdy, tag, tid,
                  dl, health, rank, S: gl.constexpr, HEADS: gl.constexpr,
                  DEV: gl.constexpr):  # fmt: skip
    """(token, 8 heads) ``unit``: the live chunks merged with the sink folded in
    (upstream's order: max, then exp2 weights and sums chunk by chunk), bf16,
    inverse GPT-J RoPE of dims 448.., vLLM's MXFP8 of each 32 -> XO / XOS, CRDY."""
    CK: gl.constexpr = chunk_keys(HEADS, S)
    NCHUNK: gl.constexpr = n_chunks(HEADS, S)
    HG: gl.constexpr = HEADS // 8
    Q_ROWS: gl.constexpr = HEADS * HEAD_DIM
    t = unit // HG
    hg = unit % HG
    if DEV:
        await_key_table(krdy, t, tag, tid, dl, health, rank)
        kv_len = gl.max(load_dev32(klen.to(gl.uint64) + (t * 4 + gl.zeros_like(tid)).to(gl.uint64)
                                   ).to(gl.int32, bitcast=True), axis=0)  # fmt: skip
    else:
        kv_len = gl.load(klen.to(gl.pointer_type(gl.int32)) + t)
    nlive = (kv_len + CK - 1) // CK
    # the live chunks' flags (thread u: chunk u)
    live_u = tid < nlive
    _, f = poll1(srdy.to(gl.uint64) + (t * NCHUNK + gl.minimum(tid, NCHUNK - 1)).to(gl.uint64) * 8,
                 tag, live_u, dl, health)  # fmt: skip
    fail(health, f, rank, 20, live_u)
    gl.barrier()
    S2: gl.constexpr = gl.SliceLayout(2, C3)
    wv = gl.arange(0, 8, layout=gl.SliceLayout(1, S2))[:, None]
    ln = gl.arange(0, 64, layout=gl.SliceLayout(0, S2))[None, :]
    e3 = gl.arange(0, 8, layout=gl.SliceLayout(0, gl.SliceLayout(1, C3)))[None, None, :]
    h = hg * 8 + wv  # [8][1]
    d0 = ln * 8  # [1][64]: this lane's 8 dims
    p = gl.load(pos.to(gl.pointer_type(gl.int32)) + 2 * t)
    snk = gl.load(sink.to(gl.pointer_type(gl.float32)) + h) * LOG2E
    m = gl.full((8, 1), NEG, gl.float32, S2)
    mcs = ()
    lcs = ()
    vcs = ()
    for cc in gl.static_range(NCHUNK):
        live = cc < nlive
        sp = pstat.to(gl.pointer_type(gl.float32)) + ((t * NCHUNK + cc) * HEADS + h) * 2
        mc = gl.where(live, gl.load(sp), NEG)
        lc = gl.where(live, gl.load(sp + 1), 0.0)
        pv = gl.load(part.to(gl.pointer_type(gl.bfloat16))
                     + (((t * NCHUNK + cc) * HEADS + h) * HEAD_DIM + d0)[:, :, None] + e3)  # fmt: skip
        vc = gl.where(live, pv.to(gl.float32), 0.0)
        m = gl.maximum(m, mc)
        mcs = mcs + (mc,)
        lcs = lcs + (lc,)
        vcs = vcs + (vc,)
    m_final = gl.maximum(m, snk)
    lsum = hw_exp2(snk - m_final)
    tot = gl.zeros((8, 64, 8), gl.float32, C3)
    for cc in gl.static_range(NCHUNK):
        wgt = hw_exp2(mcs[cc] - m_final)
        lsum = lsum + lcs[cc] * wgt
        tot = tot + vcs[cc] * wgt[:, :, None]
    inv = 1.0 / lsum
    o = (tot * inv[:, :, None]).to(gl.bfloat16).to(gl.float32)
    # inverse GPT-J RoPE of each pair: e' = e cos + o sin, o' = o cos - e sin
    ev, od = gl.split(gl.reshape(o, (8, 64, 4, 2)))  # pair i of the lane: [8][64][4]
    # cos / sin of the lane's 4 pairs (dims 448..: rotated; else 1, 0)
    EL: gl.constexpr = ev.type.layout
    ln3 = gl.arange(0, 64, layout=gl.SliceLayout(0, gl.SliceLayout(2, EL)))[None, :, None]
    ip = gl.arange(0, 4, layout=gl.SliceLayout(0, gl.SliceLayout(1, EL)))[None, None, :]
    rope = ln3 * 8 >= NOPE
    csp = (cos_sin.to(gl.pointer_type(gl.float32)) + p * ROPE
           + gl.maximum((ln3 * 8 - NOPE) // 2, 0) + ip)  # fmt: skip
    cs4 = gl.where(rope, gl.load(csp), 1.0)
    sn4 = gl.where(rope, gl.load(csp + ROPE // 2), 0.0)
    y = gl.reshape(gl.join(ev * cs4 + od * sn4, od * cs4 - ev * sn4), (8, 64, 8))
    y = gl.convert_layout(y, C3)
    amax = gl.max(gl.abs(y), axis=2)
    amax = gl.maximum(amax, xshfl(amax, ln + wv * 0, 1))
    amax = gl.maximum(amax, xshfl(amax, ln + wv * 0, 2))
    code = vllm_mx_code(amax)
    mul = ((254 - code) << 23).to(gl.float32, bitcast=True)
    q8 = clamp_fp8(y * mul[:, :, None])
    # word w of the lane = its values 4 w .. + 4: [w][e1][e0], e = 2 e1 + e0
    ev0, ev1 = gl.split(gl.reshape(q8, (8, 64, 2, 2, 2)))
    y0, y2 = gl.split(ev0)
    y1, y3 = gl.split(ev1)
    words = fp8_pack4(y0, y1, y2, y3)  # [8][64][2]
    WL: gl.constexpr = words.type.layout
    wv3 = gl.arange(0, 8, layout=gl.SliceLayout(1, gl.SliceLayout(2, WL)))[:, None, None]
    ln3w = gl.arange(0, 64, layout=gl.SliceLayout(0, gl.SliceLayout(2, WL)))[None, :, None]
    wl = gl.arange(0, 2, layout=gl.SliceLayout(0, gl.SliceLayout(1, WL)))[None, None, :]
    gl.store(xo.to(gl.pointer_type(gl.uint32)) + t * (Q_ROWS // 4)
             + ((hg * 8 + wv3) * HEAD_DIM + ln3w * 8) // 4 + wl, words, cache_modifier=".wt")  # fmt: skip
    gl.store(xos.to(gl.pointer_type(gl.int32)) + t * (Q_ROWS // 32) + (h * HEAD_DIM + d0) // 32,
             code, mask=(ln % 4 == 0) & (wv >= 0), cache_modifier=".wt")  # fmt: skip
    publish(crdy, gl.zeros_like(tid) + unit, gl.zeros_like(tid) + 1, tag, tid == 0)


# ---------------------------------------------------------------- wo_a / wo_b
# x tiles: [I][16 tokens][W][128] bytes (16 a thread), K step kstep(i, w)
XT = gl.constexpr(gl.BlockedLayout([1, 1, 1, 16], [1, 2, 4, 8], [1, 8, 1, 1], [3, 2, 1, 0]))


@gluon.constexpr_function
def g_mma(wb):
    """Batched scaled MFMA of a GEMV: ``wb`` waves on the batch (K parts), the
    rest on M (16-row tiles)."""
    return gl.amd.AMDMFMALayout(4, [16, 16, 128], False, [wb, 8 // wb, 1])


@gluon.constexpr_function
def g_ad(wb):
    return gl.DotOperandLayout(0, g_mma(wb), 16)


@gluon.constexpr_function
def g_bd(wb):
    return gl.DotOperandLayout(1, g_mma(wb), 16)


@gluon.constexpr_function
def g_ald(wb):
    """Weights [part][rows][128]: lane row + 16 kq, 16 bytes (the A order)."""
    return gl.BlockedLayout([1, 1, 16], [1, 16, 4], [wb, 8 // wb, 1], [1, 2, 0])


@gluon.constexpr_function
def g_sld(wb):
    """Scales [part][rows][4]: a byte a lane, lane row + 16 group."""
    return gl.BlockedLayout([1, 1, 1], [1, 16, 4], [wb, 8 // wb, 1], [1, 2, 0])


@gluon.jit
def kstep(i, w, I: gl.constexpr, W: gl.constexpr, CONTIG: gl.constexpr):
    """The K step of tile slot (i, w): w's contiguous range (``CONTIG``: wave
    part w of W takes steps w I ..) or steps interleaved over the waves."""
    if CONTIG:
        return w * I + i
    else:
        return i * W + w


@gluon.jit
def load_rows(words, codes, ROW: gl.constexpr, K0, t0, n, x_s, c_s, I: gl.constexpr,
              W: gl.constexpr, CONTIG: gl.constexpr, NSTEPS: gl.constexpr = 0):  # fmt: skip
    """Tokens t0 .. t0 + n (rows past n: the last again) of a published MXFP8
    region (``ROW`` bytes a row; codes an i32 a group) at its K window K0 ..
    + I W 128 -> the LDS tile ``x_s`` [I][16][W][128] and its codes as bytes
    ``c_s`` [I][16][W][4] (flat buffers, viewed at this GEMV's shape). Plain
    loads: behind their flags, never read before. ``NSTEPS`` (a window of fewer
    than I W steps): a slot past it holds the last step again (finite bytes for
    the zero-weight MFMA of a dead step)."""
    NS: gl.constexpr = I * W if NSTEPS == 0 else NSTEPS
    i4 = gl.arange(0, I, layout=gl.SliceLayout(1, gl.SliceLayout(2, gl.SliceLayout(3, XT))))
    t4 = gl.arange(0, 16, layout=gl.SliceLayout(0, gl.SliceLayout(2, gl.SliceLayout(3, XT))))
    w4 = gl.arange(0, W, layout=gl.SliceLayout(0, gl.SliceLayout(1, gl.SliceLayout(3, XT))))
    k4 = gl.arange(0, 128, layout=gl.SliceLayout(0, gl.SliceLayout(1, gl.SliceLayout(2, XT))))
    i4 = i4[:, None, None, None]
    t4 = t4[None, :, None, None]
    w4 = w4[None, None, :, None]
    k4 = k4[None, None, None, :]
    row = t0 + gl.minimum(t4, n - 1)
    XN: gl.constexpr = 65536 // (16 * W * 128)  # x_s: 64 KB, flat
    x_s._reinterpret(gl.uint8, [XN, 16, W, 128], SH4).slice(0, I, 0).store(gl.load(words.to(gl.pointer_type(gl.uint8)) + row * ROW + K0
                      + gl.minimum(kstep(i4, w4, I, W, CONTIG), NS - 1) * 128 + k4))  # fmt: skip
    NC: gl.constexpr = 16 * I * W * 4
    c8 = c_s  # 2 KB, flat
    tid = gl.arange(0, 512, layout=T512)
    for k in gl.static_range(0, NC, 512):
        e = tid + k  # (token, i, w, group): token-major
        et = e // (I * W * 4)
        ei = (e // (W * 4)) % I
        ew = (e // 4) % W
        eg = e % 4
        code = gl.load(codes.to(gl.pointer_type(gl.int32)) + (t0 + gl.minimum(et, n - 1)) * (ROW // 32)
                       + K0 // 32 + gl.minimum(kstep(ei, ew, I, W, CONTIG), NS - 1) * 4
                       + eg).to(gl.uint8)  # fmt: skip
        c8.scatter(code, ((ei * 16 + et) * W + ew) * 4 + eg, 0)
    gl.barrier()


@gluon.jit
def gemv_tile(w, ws, K: gl.constexpr, row0, x_s, c_s, I: gl.constexpr, W: gl.constexpr,
              CONTIG: gl.constexpr, NSTEPS: gl.constexpr = 0, KS0=0):  # fmt: skip
    """[W][16 (8 / W)][16 tokens] partials of rows row0 .. of a row-major e4m3
    weight [N, K] (E8M0 [N / 32, K / 32]) against the LDS tile: wave (part w,
    row tile m) over its I steps, chained in order (upstream's K order). The
    tile is the weight's K steps KS0 .. + NSTEPS (default I W); a dead step
    (past NSTEPS) is upstream's: zero weights at unit scale."""
    NS: gl.constexpr = I * W if NSTEPS == 0 else NSTEPS
    # rounds a wave runs: the tile's I (a power of two) may pad past the steps
    NI: gl.constexpr = I if CONTIG else (NS + W - 1) // W
    M: gl.constexpr = 16 * (8 // W)
    MMA: gl.constexpr = g_mma(W)
    AD: gl.constexpr = g_ad(W)
    BD: gl.constexpr = g_bd(W)
    SAL: gl.constexpr = gl.amd.cdna4.get_mfma_scale_layout(AD, [W, M, 4])
    SBL: gl.constexpr = gl.amd.cdna4.get_mfma_scale_layout(BD, [W, 16, 4])
    wa, ra, ka = _ar3(W, M, 128, g_ald(W))
    ws_, rs_, gs_ = _ar3(W, M, 4, g_sld(W))
    wp = w.to(gl.pointer_type(gl.uint8))
    sp = ws.to(gl.pointer_type(gl.uint8))
    xv = x_s._reinterpret(gl.uint8, [65536 // (16 * W * 128), 16, W, 128], SH3)
    cv = c_s._reinterpret(gl.uint8, [2048 // (16 * W * 4), 16, W, 4], SH3)
    acc = gl.zeros((W, M, 16), gl.float32, MMA)
    for i in gl.static_range(NI):
        sta = kstep(i, wa, I, W, CONTIG)
        a = gl.load(wp + (row0 + ra) * K + (KS0 + gl.minimum(sta, NS - 1)) * 128 + ka,
                    mask=sta < NS, other=0, cache_modifier=".cg")  # fmt: skip
        sts = kstep(i, ws_, I, W, CONTIG)
        sa = gl.load(sp + ((row0 + rs_) // 32) * (K // 32) + (KS0 + gl.minimum(sts, NS - 1)) * 4
                     + gs_, mask=sts < NS, other=127)  # fmt: skip
        b = xv.index(i).permute((1, 2, 0)).load(BD)
        sb = cv.index(i).permute((1, 0, 2)).load(SBL)
        acc = gl.amd.cdna4.mfma_scaled(gl.convert_layout(a, AD), gl.convert_layout(sa, SAL), "e4m3",
                                       b, sb, "e4m3", acc)  # fmt: skip
    return acc


@gluon.jit
def stage_woa(task, woa, woa_s, xo, xos, x8b, x8bs, crdy, x_s, c_s, red, tag, tid, dl, health,
              rank, S: gl.constexpr, HEADS: gl.constexpr, TP: gl.constexpr):  # fmt: skip
    """Rows 32 task .. of wo_a (two 16-row tiles; K = the group's 4096 split in
    four contiguous parts, summed in part order) for every token tile -> bf16
    -> vLLM's MXFP8 of the 32 -> X8B / X8BS."""
    GROUPS: gl.constexpr = 8 // TP
    GROUP_K: gl.constexpr = HEADS // GROUPS * HEAD_DIM  # 4096
    Q_ROWS: gl.constexpr = HEADS * HEAD_DIM
    O_ROWS: gl.constexpr = GROUPS * 1024
    TASKS: gl.constexpr = O_ROWS // 32
    I: gl.constexpr = GROUP_K // 128 // 4
    grp = task // (TASKS // GROUPS)
    # every token's XO of the group (combine unit (t, grp))
    live = tid < S
    _, f = poll1(crdy.to(gl.uint64) + (gl.minimum(tid, S - 1) * (HEADS // 8) + grp).to(gl.uint64) * 8,
                 tag, live, dl, health)  # fmt: skip
    fail(health, f, rank, 21, live)
    gl.barrier()
    lane = tid % 64
    tl = tid // 32
    row = tid % 32
    for t0 in gl.static_range(0, S, 16):
        n = _cmin(16, S - t0)
        load_rows(xo, xos, Q_ROWS, grp * GROUP_K, t0, n, x_s, c_s, I, 4, True)
        acc = gemv_tile(woa, woa_s, GROUP_K, task * 32, x_s, c_s, I, 4, True)
        red.slice(0, 2048).store(gl.reshape(gl.convert_layout(acc, RED_W), (2048,)))
        gl.barrier()
        tlc = gl.minimum(tl, n - 1)
        y = 0.0 + red.gather(row * 16 + tlc, 0)
        for wv in gl.static_range(1, 4):
            y = y + red.gather((wv * 32 + row) * 16 + tlc, 0)
        y = y.to(gl.bfloat16).to(gl.float32)
        amax = gl.abs(y)
        for off in gl.static_range(5):
            amax = gl.maximum(amax, xshfl(amax, lane, 1 << off))
        code = vllm_mx_code(amax)
        q = clamp_fp8(y * ((254 - code) << 23).to(gl.float32, bitcast=True))
        word = fp8_pack4(q, lane_gather(q, gl.minimum(lane + 1, 63)),
                         lane_gather(q, gl.minimum(lane + 2, 63)),
                         lane_gather(q, gl.minimum(lane + 3, 63)))  # fmt: skip
        mine = tl < n
        t = t0 + tlc
        gl.store(x8b.to(gl.pointer_type(gl.uint32)) + t * (O_ROWS // 4) + (task * 32 + row) // 4,
                 word, mask=mine & (row % 4 == 0), cache_modifier=".wt")  # fmt: skip
        gl.store(x8bs.to(gl.pointer_type(gl.int32)) + t * (O_ROWS // 32) + task, code,
                 mask=mine & (row == 0), cache_modifier=".wt")  # fmt: skip
        gl.barrier()


@gluon.jit
def stage_wob(task, task2, has2, wob, wob_s, x8b, x8bs, ardy, peers, x_s, c_s, red, tag, tid,
              dl, health, rank, S: gl.constexpr, HEADS: gl.constexpr, TP: gl.constexpr,
              HIDDEN: gl.constexpr):  # fmt: skip
    """Rows 16 task .. (and 16 task2 .., ``has2``) of wo_b (K = o_rows, wave w
    steps w, w + 8, .., summed in wave order) for every token tile -> bf16
    pairs to every rank's ATTN region (the FFN seam's input)."""
    GROUPS: gl.constexpr = 8 // TP
    O_ROWS: gl.constexpr = GROUPS * 1024
    I: gl.constexpr = O_ROWS // 128 // 8
    WOA_TASKS: gl.constexpr = O_ROWS // 32
    live = tid < WOA_TASKS
    _, f = poll1(ardy.to(gl.uint64) + gl.minimum(tid, WOA_TASKS - 1).to(gl.uint64) * 8, tag, live,
                 dl, health)  # fmt: skip
    fail(health, f, rank, 22, live)
    gl.barrier()
    for t0 in gl.static_range(0, S, 16):
        n = _cmin(16, S - t0)
        load_rows(x8b, x8bs, O_ROWS, 0, t0, n, x_s, c_s, I, 8, False)
        acc = gemv_tile(wob, wob_s, O_ROWS, task * 16, x_s, c_s, I, 8, False)
        red.slice(0, 2048).store(gl.reshape(gl.convert_layout(acc, RED_B), (2048,)))
        if has2:
            acc2 = gemv_tile(wob, wob_s, O_ROWS, task2 * 16, x_s, c_s, I, 8, False)
            red.slice(2048, 2048).store(gl.reshape(gl.convert_layout(acc2, RED_B), (2048,)))
        gl.barrier()
        # thread (k, token, row pair): task k's rows 2 rp, 2 rp + 1 of a token
        k = tid // 128
        u = tid % 128
        rp = u % 8
        tl = u // 8
        mine = (k == 0) | ((k == 1) & has2)
        live2 = mine & (tl < n) & (k < 2)
        kk = gl.minimum(k, 1)
        v0 = 0.0 + red.gather(kk * 2048 + (2 * rp) * 16 + tl, 0)
        v1 = 0.0 + red.gather(kk * 2048 + (2 * rp + 1) * 16 + tl, 0)
        for wv in gl.static_range(1, 8):
            v0 = v0 + red.gather(kk * 2048 + (wv * 16 + 2 * rp) * 16 + tl, 0)
            v1 = v1 + red.gather(kk * 2048 + (wv * 16 + 2 * rp + 1) * 16 + tl, 0)
        col = gl.where(k == 0, task, task2) * 16 + 2 * rp
        pair = attn_pair(rank, t0 + tl, col, S, HIDDEN)
        word = bf16_pair(v0, v1)
        for p in gl.static_range(TP):
            put(peers[p], pair, word, tag, live2, True)
        gl.barrier()


# [W][M][16] partials -> LDS [w][row][token]
RED_W = gl.constexpr(gl.BlockedLayout([1, 1, 1], [1, 4, 16], [4, 2, 1], [2, 1, 0]))
RED_B = gl.constexpr(gl.BlockedLayout([1, 1, 1], [1, 4, 16], [8, 1, 1], [2, 1, 0]))


@gluon.constexpr_function
def _cmin(a, b):
    return min(a, b)
