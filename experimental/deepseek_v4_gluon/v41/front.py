# SPDX-License-Identifier: MIT
"""The DS V4.1 attention front (K1) in Gluon: ``attention/front.py`` (vLLM
8321d11), the stages after the attention seam (whose norm publishes X8):

    kt     (S, on the 32 CTAs without a wqkv part): the token's keys for K2
    wqkv_a (112 row tiles x 2 K halves, 16 rows): X8's half, a token tile at a
           time, times wqkv's -> fp32 partials -> QKVP
    qkv    (S): the halves summed -> bf16; q_norm -> bf16 -> vLLM's MXFP8 ->
           QX8; kv_norm -> bf16 -> GPT-J RoPE -> bf16 -> the fp8_ds_mla record
           at the token's slot
    wq_b   (H x 512 / 16, 16 rows): QX8 times wq_b -> bf16 -> GPT-J RoPE of a
           head's dims 448.. -> q (global bf16)

The GEMVs are ``back.gemv_tile`` (upstream's K order: wave w steps w, w + 8,
.., a dead step a zero-weight MFMA), sums in wave order from 0.0."""

from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from ..common.mx import clamp_fp8, div_rn, fp8_pack4, vllm_mx_code
from ..common.ops import bf16_pair, bf16_round, hw_rsq, sum_desc6, xshfl
from ..common.sync import T512, fail, poll1, poll2, poll8, publish, put
from .back import RED_B, gemv_tile, load_rows

HIDDEN = gl.constexpr(5120)
Q_RANK = gl.constexpr(1280)
KV_DIM = gl.constexpr(512)
QKV_ROWS = gl.constexpr(1792)
HEAD_DIM = gl.constexpr(512)
NOPE = gl.constexpr(448)
ROPE = gl.constexpr(64)
DATA = gl.constexpr(576)
EPS = gl.constexpr(1e-20)
KSTEPS = gl.constexpr(20)  # a wqkv K half's 128-steps

KEYS = gl.constexpr(640)  # top-k 512 + window 128
TOPK = gl.constexpr(512)


@gluon.jit
def stage_kt(t, slot, swa_idx, swa_lens, pos, t2r, topk, comp_bt, bt_stride, comp_block, kt,
             klen, S: gl.constexpr, RATIO: gl.constexpr, SWA_W: gl.constexpr):  # fmt: skip
    """Token t's keys for K2: ``KT[t][k] = slot | 1 << 31`` for its top-k
    compressed rows (k < ntopk, through the request's block table), the window
    slot for k in [ntopk, kv_len), -1 for an invalid or absent key; KLEN[t] =
    kv_len (0 for a pad row), KLEN[S + t] = ntopk. Write-through stores: a
    consumer CTA may sit on another XCD (``back.py``)."""
    tid = gl.arange(0, 512, layout=T512)
    i32 = gl.pointer_type(gl.int32)
    sl = gl.load(slot.to(i32) + 2 * t)
    live = sl >= 0
    nswa = gl.where(live, gl.load(swa_lens.to(i32) + t), 0)
    if RATIO > 0:
        p = gl.load(pos.to(i32) + 2 * t)
        ntopk = gl.where(live, gl.minimum((p + 1) // RATIO, TOPK), 0)
        req = gl.load(t2r.to(i32) + t)
    else:
        ntopk = sl * 0
    kv_len = ntopk + nswa
    for i in gl.static_range(0, KEYS, 512):
        k = tid + i
        sidx = gl.minimum(gl.maximum(k - ntopk, 0), SWA_W - 1)
        val = gl.where(k < kv_len, gl.load(swa_idx.to(i32) + t * SWA_W + sidx), -1)
        if RATIO > 0:
            local = gl.load(topk.to(i32) + t * TOPK + gl.minimum(k, TOPK - 1))
            lc = gl.maximum(local, 0)
            blk = gl.load(comp_bt.to(i32) + req * bt_stride + lc // comp_block)
            c_slot = (blk * comp_block + lc % comp_block) | -2147483648
            val = gl.where(k < ntopk, gl.where(local >= 0, c_slot, -1), val)
        gl.store(kt.to(i32) + t * KEYS + k, val, mask=k < KEYS, cache_modifier=".wt")
    first = tid == 0
    gl.store(klen.to(i32) + t + tid * 0, kv_len + tid * 0, mask=first, cache_modifier=".wt")
    gl.store(klen.to(i32) + S + t + tid * 0, ntopk + tid * 0, mask=first, cache_modifier=".wt")


# ---------------------------------------------------------------- wqkv_a
@gluon.jit
def stage_wqkv(task, part, wqkv, wqkv_s, x8, x8s, xrdy, qkvp, x_s, c_s, red, tag, tid, dl,
               health, rank, S: gl.constexpr):  # fmt: skip
    """Rows 16 task .. of wqkv over K half ``part`` against every token, a tile
    at a time -> fp32 partials QKVP[part] (a tagged pair a value)."""
    live = tid < S
    _, f = poll1(xrdy.to(gl.uint64) + (2 * gl.minimum(tid, S - 1) + part).to(gl.uint64) * 8, tag,
                 live, dl, health)  # fmt: skip
    fail(health, f, rank, 30, live)
    gl.barrier()
    r = tid % 16
    tl = tid // 16
    for t0 in gl.static_range(0, S, 16):
        n = _cmin(16, S - t0)
        # the tile padded to 4 rounds (a power of two), the GEMV runs 3
        load_rows(x8, x8s, HIDDEN, part * (KSTEPS * 128), t0, n, x_s, c_s, 4, 8, False, KSTEPS)
        acc = gemv_tile(wqkv, wqkv_s, HIDDEN, task * 16, x_s, c_s, 4, 8, False, KSTEPS,
                        part * KSTEPS)  # fmt: skip
        red.slice(0, 2048).store(gl.reshape(gl.convert_layout(acc, RED_B), (2048,)))
        gl.barrier()
        tlc = gl.minimum(tl, n - 1)
        v = 0.0 + red.gather(r * 16 + tlc, 0)
        for wv in gl.static_range(1, 8):
            v = v + red.gather((wv * 16 + r) * 16 + tlc, 0)
        put(qkvp, (part * S + t0 + tlc) * QKV_ROWS + task * 16 + r,
            v.to(gl.uint32, bitcast=True), tag, tl < n)  # fmt: skip
        gl.barrier()


# ---------------------------------------------------------------- q / kv norms
@gluon.jit
def stage_qkv(t, qkvp, qn_w, kvn_w, pos, slot, cos_sin, swa, swa_stride, swa_block, qx8, qx8s,
              qrdy, red, tag, tid, lane, wave, dl, health, rank, S: gl.constexpr):  # fmt: skip
    """Token t: the K halves summed -> bf16; q_norm -> bf16 -> MXFP8 (QX8);
    kv_norm -> bf16 -> RoPE -> bf16 -> the fp8_ds_mla record at slot[t]. 256
    threads (8 q columns and 2 kv columns each); threads 256.. mirror thread
    255 and store nothing."""
    tt = gl.minimum(tid, 255)
    mine = tid < 256
    bf = gl.pointer_type(gl.bfloat16)
    f32 = gl.pointer_type(gl.float32)
    i32 = gl.pointer_type(gl.int32)
    p = gl.load(pos.to(i32) + 2 * t)
    sl = gl.load(slot.to(i32) + 2 * t)
    jr = tt - NOPE // 2
    is_rope = jr >= 0
    jj = gl.maximum(jr, 0)
    cs = gl.where(is_rope, gl.load(cos_sin.to(f32) + p * ROPE + jj), 1.0)
    sn = gl.where(is_rope, gl.load(cos_sin.to(f32) + p * ROPE + ROPE // 2 + jj), 0.0)
    # the partials of both K halves: 8 q columns, 2 kv columns a thread
    qa = ()
    for h in gl.static_range(2):
        base = qkvp.to(gl.uint64) + ((h * S + t) * QKV_ROWS).to(gl.uint64) * 8
        a = ()
        for j in gl.static_range(8):
            a = a + (base + gl.minimum(tt * 8 + j, Q_RANK - 1).to(gl.uint64) * 8,)
        q0, q1, q2, q3, q4, q5, q6, q7, fq = poll8(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7],
                                                   tag, mine, dl, health)  # fmt: skip
        ka = base + (Q_RANK + tt * 2).to(gl.uint64) * 8
        k0, k1, fk = poll2(ka, ka + 8, tag, mine, dl, health)
        fail(health, gl.maximum(fq, fk), rank, 31, mine)
        qa = qa + ((q0, q1, q2, q3, q4, q5, q6, q7, k0, k1),)
    vals = ()
    for j in gl.static_range(10):
        vals = vals + (bf16_round(qa[0][j].to(gl.float32, bitcast=True)
                                  + qa[1][j].to(gl.float32, bitcast=True)),)  # fmt: skip
    xq = ()
    for j in gl.static_range(8):
        xq = xq + (gl.where(tt * 8 + j < Q_RANK, vals[j], 0.0),)
    ssq = xq[0] * xq[0]
    for j in gl.static_range(1, 8):
        ssq = ssq + xq[j] * xq[j]
    ssq = sum_desc6(ssq, lane)
    ssk = sum_desc6(vals[8] * vals[8] + vals[9] * vals[9], lane)
    red.scatter(ssq, gl.where((lane == 0) & mine, wave, 4095), 0)
    red.scatter(ssk, gl.where((lane == 0) & mine, 4 + wave, 4095), 0)
    gl.barrier()
    z = gl.zeros_like(tid)
    tq = (red.gather(z, 0) + red.gather(z + 1, 0)) + (red.gather(z + 2, 0) + red.gather(z + 3, 0))
    tk = (red.gather(z + 4, 0) + red.gather(z + 5, 0)) + (red.gather(z + 6, 0) + red.gather(z + 7, 0))
    rq = hw_rsq(div_rn(tq, 1280.0, 1.0 / 1280.0) + EPS)
    rk = hw_rsq(div_rn(tk, 512.0, 1.0 / 512.0) + EPS)
    # q: the bf16 normed row, then vLLM's MXFP8 (a 32-group = 4 threads)
    ys = ()
    for j in gl.static_range(8):
        w = gl.load(qn_w.to(bf) + gl.minimum(tt * 8 + j, Q_RANK - 1)).to(gl.float32)
        ys = ys + (bf16_round((xq[j] * rq) * w),)
    amax = gl.abs(ys[0])
    for j in gl.static_range(1, 8):
        amax = gl.maximum(amax, gl.abs(ys[j]))
    amax = gl.maximum(amax, xshfl(amax, lane, 1))
    amax = gl.maximum(amax, xshfl(amax, lane, 2))
    code = vllm_mx_code(amax)
    mul = ((254 - code) << 23).to(gl.float32, bitcast=True)
    qlive = mine & (tt * 8 < Q_RANK)
    w0 = fp8_pack4(clamp_fp8(ys[0] * mul), clamp_fp8(ys[1] * mul), clamp_fp8(ys[2] * mul),
                   clamp_fp8(ys[3] * mul))  # fmt: skip
    w1 = fp8_pack4(clamp_fp8(ys[4] * mul), clamp_fp8(ys[5] * mul), clamp_fp8(ys[6] * mul),
                   clamp_fp8(ys[7] * mul))  # fmt: skip
    qw = qx8.to(gl.pointer_type(gl.uint32)) + t * (Q_RANK // 4) + 2 * tt
    gl.store(qw, w0, mask=qlive, cache_modifier=".wt")
    gl.store(qw + 1, w1, mask=qlive, cache_modifier=".wt")
    gl.store(qx8s.to(i32) + t * (Q_RANK // 32) + tt // 4, code, mask=qlive & (tt % 4 == 0),
             cache_modifier=".wt")  # fmt: skip
    # kv: this thread's pair (2 tt, 2 tt + 1), normed, rotated (dims 448..)
    kn0 = bf16_round((vals[8] * rk) * gl.load(kvn_w.to(bf) + tt * 2).to(gl.float32))
    kn1 = bf16_round((vals[9] * rk) * gl.load(kvn_w.to(bf) + tt * 2 + 1).to(gl.float32))
    e0 = bf16_round(kn0 * cs - kn1 * sn)
    e1 = bf16_round(kn0 * sn + kn1 * cs)
    # the record's nope quant: a 64-block = 32 threads, the code by ceil, no clamp
    amax = gl.maximum(gl.abs(e0), gl.abs(e1))
    for off in gl.static_range(5):
        amax = gl.maximum(amax, xshfl(amax, lane, 1 << off))
    qv = div_rn(gl.maximum(amax, 1e-4), 448.0, 1.0 / 448.0)
    bits = qv.to(gl.int32, bitcast=True)
    rcode = ((bits >> 23) & 0xFF) + gl.where((bits & 0x7FFFFF) != 0, 1, 0)
    inv = ((254 - rcode) << 23).to(gl.float32, bitcast=True)
    q0 = clamp_fp8(e0 * inv)
    q1 = clamp_fp8(e1 * inv)
    word = fp8_pack4(q0, q1, xshfl(q0, lane, 1), xshfl(q1, lane, 1))
    rope_word = bf16_pair(e0, e1)
    red.scatter(rcode.to(gl.float32, bitcast=True),
                gl.where(mine & (tt % 32 == 0) & (tt < NOPE // 2), 8 + tt // 32, 4095), 0)  # fmt: skip
    gl.barrier()
    sblk = sl // swa_block
    soff = sl % swa_block
    base = swa + sblk.to(gl.int64) * swa_stride
    rec = base + soff * DATA
    ok = mine & (sl >= 0)
    u32 = gl.pointer_type(gl.uint32)
    gl.store((rec + 2 * tt).to(u32), word, mask=ok & (tt < NOPE // 2) & (tt % 2 == 0),
             cache_modifier=".wt")  # fmt: skip
    gl.store((rec + NOPE + 4 * jj).to(u32), rope_word, mask=ok & is_rope, cache_modifier=".wt")
    sb = ()
    for b in gl.static_range(7):
        sb = sb + (red.gather(z + 8 + b, 0).to(gl.int32, bitcast=True),)
    sw0 = sb[0] | (sb[1] << 8) | (sb[2] << 16) | (sb[3] << 24)
    sw1 = sb[4] | (sb[5] << 8) | (sb[6] << 16)
    scp = (base + swa_block * DATA + soff * 8).to(i32) + z
    gl.store(scp, sw0, mask=ok & (tid == 0), cache_modifier=".wt")
    gl.store(scp + 1, sw1, mask=ok & (tid == 0), cache_modifier=".wt")
    publish(qrdy, z + t, z + 1, tag, tid == 0)


# ---------------------------------------------------------------- wq_b
@gluon.jit
def stage_wqb(task0, wqb, wqb_s, pos, cos_sin, qx8, qx8s, qrdy, q, x_s, c_s, red, tag, tid, dl,
              health, rank, S: gl.constexpr, HEADS: gl.constexpr):  # fmt: skip
    """Rows 16 (task0 + 256 k) .. of the rank's H x 512 query rows, every task k
    of the CTA (tasks 256 apart: whole heads apart, the same dims): QX8 x wq_b
    -> bf16 -> GPT-J RoPE of a head's dims 448.. (fp32) -> bf16 -> q."""
    Q_ROWS: gl.constexpr = HEADS * HEAD_DIM
    NT: gl.constexpr = Q_ROWS // 16 // 256
    gl.static_assert(NT * 2048 <= 4096 * 2)
    live = tid < S
    _, f = poll1(qrdy.to(gl.uint64) + gl.minimum(tid, S - 1).to(gl.uint64) * 8, tag, live, dl,
                 health)  # fmt: skip
    fail(health, f, rank, 32, live)
    gl.barrier()
    i32 = gl.pointer_type(gl.int32)
    f32 = gl.pointer_type(gl.float32)
    rp = tid % 8
    dim = (task0 * 16) % HEAD_DIM + 2 * rp
    rope = dim >= NOPE
    jj = gl.maximum((dim - NOPE) >> 1, 0)
    for t0 in gl.static_range(0, S, 16):
        n = _cmin(16, S - t0)
        load_rows(qx8, qx8s, Q_RANK, 0, t0, n, x_s, c_s, 2, 8, False, 10)
        # thread (k, token, row pair): task k's rows 2 rp, 2 rp + 1 of a token
        k = tid // 128
        tl = (tid % 128) // 8
        tlc = gl.minimum(tl, n - 1)
        pt = gl.load(pos.to(i32) + 2 * (t0 + tlc))
        cs = gl.load(cos_sin.to(f32) + pt * ROPE + jj)
        sn = gl.load(cos_sin.to(f32) + pt * ROPE + ROPE // 2 + jj)
        for kb in gl.static_range(0, NT, 2):
            # two tasks a round (LDS ``red``: 2 x 2048)
            for kk in gl.static_range(2):
                if kb + kk < NT:
                    acc = gemv_tile(wqb, wqb_s, Q_RANK, (task0 + 256 * (kb + kk)) * 16, x_s, c_s,
                                    2, 8, False, 10)  # fmt: skip
                    red.slice(kk * 2048, 2048).store(
                        gl.reshape(gl.convert_layout(acc, RED_B), (2048,)))
            gl.barrier()
            kl = gl.minimum(k, 1)
            v0 = 0.0 + red.gather(kl * 2048 + (2 * rp) * 16 + tlc, 0)
            v1 = 0.0 + red.gather(kl * 2048 + (2 * rp + 1) * 16 + tlc, 0)
            for wv in gl.static_range(1, 8):
                v0 = v0 + red.gather(kl * 2048 + (wv * 16 + 2 * rp) * 16 + tlc, 0)
                v1 = v1 + red.gather(kl * 2048 + (wv * 16 + 2 * rp + 1) * 16 + tlc, 0)
            v0 = bf16_round(v0)
            v1 = bf16_round(v1)
            e0 = gl.where(rope, v0 * cs - v1 * sn, v0)
            e1 = gl.where(rope, v0 * sn + v1 * cs, v1)
            r0 = (task0 + 256 * (kb + kl)) * 16 + 2 * rp
            ok = (k < 2) & (kb + k < NT) & (tl < n)
            gl.store(q.to(gl.pointer_type(gl.uint32)) + ((t0 + tlc) * Q_ROWS + r0) // 2,
                     bf16_pair(e0, e1), mask=ok)  # fmt: skip
            gl.barrier()


@gluon.constexpr_function
def _cmin(a, b):
    return min(a, b)
