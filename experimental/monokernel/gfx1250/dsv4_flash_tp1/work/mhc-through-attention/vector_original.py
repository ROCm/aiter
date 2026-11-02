from device_scope_helpers import dev_load, dev_store
"""Checkpoint-format C1 Flash full uncompressed layer, correctness-first Gluon.

One resident launch. No library/device launches inside the layer. This initial
version uses vector matvecs, deliberately not a performance claim for WMMA.
"""
import triton
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl

@g.jit
def clock():
    return gl.inline_asm_elementwise('s_sendmsg_rtn_b64 $0, sendmsg(MSG_RTN_GET_REALTIME)\n s_wait_kmcnt 0', constraints='=s', args=[], dtype=gl.uint64, is_pure=False, pack=1)

@g.jit
def sync(Counter, Health, GRID: gl.constexpr):
    gl.inline_asm_elementwise('s_wait_storecnt 0\n s_mov_b32 $0, 0', constraints='=s,~{memory}', args=[], dtype=gl.int32, is_pure=False, pack=1)
    gl.barrier()
    ticket = gl.atomic_add(Counter, 1, sem='relaxed', scope='gpu').to(gl.uint32)
    target = (ticket // GRID + 1) * GRID
    seen = gl.atomic_add(Counter, 0, sem='relaxed', scope='gpu')
    start = clock()
    spins = 0
    while ((seen - target).to(gl.int32) < 0) & (spins < 1000000) & (clock() - start < 1000000000):
        seen = gl.atomic_add(Counter, 0, sem='relaxed', scope='gpu')
        spins += 1
    if (seen - target).to(gl.int32) < 0:
        gl.atomic_or(Health, 2, sem='relaxed', scope='gpu')
    gl.barrier()

@g.jit
def quant(X, Y, N: gl.constexpr, GROUP: gl.constexpr, GRID: gl.constexpr):
    i = gl.arange(0, GROUP, layout=gl.BlockedLayout([1], [32], [4], [0]))
    for group in range(gl.program_id(0), N // GROUP, GRID):
        x = dev_load(X, group * GROUP + i, cache='.cg').to(gl.float32)
        amax = gl.maximum(gl.max(gl.abs(x), 0), 1e-10)
        bits = (amax / 448.0).to(gl.uint32, bitcast=True)
        exponent = gl.minimum((bits & 2147483647) + 8388607 >> 23, 254)
        scale = (exponent << 23).to(gl.float32, bitcast=True)
        q = (x / scale).to(gl.float8e4nv).to(gl.float32) * scale
        dev_store(q, Y, group * GROUP + i, cache='.cg')

@g.jit
def matvec(X, W, S, Y, N: gl.constexpr, K: gl.constexpr, FORMAT: gl.constexpr, GRID: gl.constexpr, GROUPED: gl.constexpr=False, FLOAT_OUT: gl.constexpr=False, Gamma=None):
    L: gl.constexpr = gl.BlockedLayout([1, 4], [1, 32], [4, 1], [1, 0])
    rn = gl.arange(0, 8, layout=gl.SliceLayout(1, L))
    rk = gl.arange(0, 256, layout=gl.SliceLayout(0, L))
    for tile in range(gl.program_id(0), gl.cdiv(N, 8), GRID):
        n = tile * 8 + rn
        acc = gl.full((8, 256), 0, gl.float32, L)
        for kb in range(gl.cdiv(K, 256)):
            k = kb * 256 + rk
            if GROUPED:
                x = dev_load(X, n[:, None] // 1024 * K + k[None, :], cache='.cg')
            else:
                x = dev_load(X, k, cache='.cg')
                x = x[None, :]
            if FORMAT == 4:
                packed = dev_load(W, n[:, None] * (K // 2) + k[None, :] // 2, cache='.cg').to(gl.uint8)
                v = packed >> k[None, :] % 2 * 4 & 15
                a = v & 7
                w = gl.where(a < 2, a.to(gl.float32) * 0.5, gl.where(a < 4, (a.to(gl.float32) - 2) * 0.5 + 1, gl.where(a < 6, a.to(gl.float32) - 2, (a.to(gl.float32) - 6) * 2 + 4)))
                w = gl.where(v >= 8, -w, w)
                s = dev_load(S, n[:, None] * (K // 32) + k[None, :] // 32, cache='.cg').to(gl.uint32)
                w *= (s << 23).to(gl.float32, bitcast=True)
            elif FORMAT == 8:
                w = dev_load(W, n[:, None] * K + k[None, :], cache='.cg').to(gl.float32)
                s = dev_load(S, n[:, None] // 128 * (K // 128) + k[None, :] // 128, cache='.cg').to(gl.uint32)
                w *= (s << 23).to(gl.float32, bitcast=True)
            else:
                w = dev_load(W, n[:, None] * K + k[None, :], cache='.cg').to(gl.float32)
            acc += w * x
        out = gl.sum(acc, 1)
        if Gamma is not None:
            out *= dev_load(Gamma, 0, cache='.cg')
        if not FLOAT_OUT:
            out = out.to(gl.bfloat16)
        dev_store(out, Y, n, cache='.cg')

@g.jit
def norm(X, W, Y, N: gl.constexpr, ROWS: gl.constexpr, WEIGHT: gl.constexpr, GRID: gl.constexpr, ROUND: gl.constexpr=True):
    i = gl.arange(0, N, layout=gl.BlockedLayout([1], [32], [4], [0]))
    for row in range(gl.program_id(0), ROWS, GRID):
        x = dev_load(X, row * N + i, cache='.cg').to(gl.float32)
        y = x * gl.rsqrt(gl.sum(x * x, 0) / N + 1e-06)
        if WEIGHT:
            y *= dev_load(W, i, cache='.cg').to(gl.float32)
        if ROUND:
            y = y.to(gl.bfloat16)
        dev_store(y, Y, row * N + i, cache='.cg')

@g.jit
def mhc_mix(R, W, S, B, M, GRID: gl.constexpr):
    i = gl.arange(0, 1024, layout=gl.BlockedLayout([1], [32], [4], [0]))
    for row in range(gl.program_id(0), 24, GRID):
        dot = 0.0
        sq = 0.0
        for block in range(16):
            x = dev_load(R, block * 1024 + i, cache='.cg').to(gl.float32)
            w = dev_load(W, row * 16384 + block * 1024 + i, cache='.cg')
            dot += gl.sum(x * w, 0)
            sq += gl.sum(x * x, 0)
        z = dot * gl.rsqrt(sq / 16384 + 1e-06)
        si = gl.where(row < 4, 0, gl.where(row < 8, 1, 2))
        dev_store(z * dev_load(S, si, cache='.cg') + dev_load(B, row, cache='.cg'), M, row, cache='.cg')

@g.jit
def sinkhorn(M):
    if gl.program_id(0) == 0:
        L: gl.constexpr = gl.BlockedLayout([1, 1], [4, 8], [4, 1], [1, 0])
        r = gl.arange(0, 4, layout=gl.SliceLayout(1, L))
        c = gl.arange(0, 4, layout=gl.SliceLayout(0, L))
        z = dev_load(M, 8 + r[:, None] * 4 + c[None, :], cache='.cg')
        z = gl.exp(z - gl.max(z, 1)[:, None])
        z = z / gl.sum(z, 1)[:, None] + 1e-06
        z = z / (gl.sum(z, 0)[None, :] + 1e-06)
        for _ in range(19):
            z = z / (gl.sum(z, 1)[:, None] + 1e-06)
            z = z / (gl.sum(z, 0)[None, :] + 1e-06)
        dev_store(z, M, 8 + r[:, None] * 4 + c[None, :], cache='.cg')
        v = dev_load(M, r, cache='.cg')
        dev_store(1.0 / (1.0 + gl.exp(-v)) + 1e-06, M, r, cache='.cg')
        v = dev_load(M, 4 + r, cache='.cg')
        dev_store(2.0 / (1.0 + gl.exp(-v)), M, 4 + r, cache='.cg')

@g.jit
def mhc_pre(R, M, Y, GRID: gl.constexpr):
    i = gl.program_id(0) * 256 + gl.arange(0, 256, layout=gl.BlockedLayout([1], [32], [4], [0]))
    for base in range(0, 4096, GRID * 256):
        c = base + i
        y = gl.full((256,), 0, gl.float32, gl.BlockedLayout([1], [32], [4], [0]))
        for h in gl.static_range(4):
            y += dev_load(R, h * 4096 + c, c < 4096, 0, cache='.cg').to(gl.float32) * dev_load(M, h, cache='.cg')
        dev_store(y.to(gl.bfloat16), Y, c, c < 4096, cache='.cg')

@g.jit
def mhc_post(X, R, M, Y, GRID: gl.constexpr):
    i = gl.arange(0, 256, layout=gl.BlockedLayout([1], [32], [4], [0]))
    for tile in range(gl.program_id(0), 64, GRID):
        h = tile // 16
        c = tile % 16 * 256 + i
        v = dev_load(X, c, cache='.cg').to(gl.float32) * dev_load(M, 4 + h, cache='.cg')
        for source in gl.static_range(4):
            v += dev_load(R, source * 4096 + c, cache='.cg').to(gl.float32) * dev_load(M, 8 + source * 4 + h, cache='.cg')
        dev_store(v.to(gl.bfloat16), Y, h * 4096 + c, cache='.cg')

@g.jit
def rope(X, Cos, Sin, Y, ROWS: gl.constexpr, INVERSE: gl.constexpr, GRID: gl.constexpr):
    i = gl.arange(0, 512, layout=gl.BlockedLayout([1], [32], [4], [0]))
    for row in range(gl.program_id(0), ROWS, GRID):
        v = dev_load(X, row * 512 + i, cache='.cg').to(gl.float32)
        other = dev_load(X, row * 512 + (i ^ 1), cache='.cg').to(gl.float32)
        c = dev_load(Cos, (i - 448) // 2, i >= 448, 1, cache='.cg')
        s = dev_load(Sin, (i - 448) // 2, i >= 448, 0, cache='.cg')
        sign = gl.where(i % 2 == 0, -1.0, 1.0)
        if INVERSE:
            sign = -sign
        dev_store((v * c + sign * other * s).to(gl.bfloat16), Y, row * 512 + i, cache='.cg')

@g.jit
def attention(Q, KV, Cache, Sink, O, GRID: gl.constexpr):
    L: gl.constexpr = gl.BlockedLayout([1, 4], [1, 32], [4, 1], [1, 0])
    r = gl.arange(0, 16, layout=gl.SliceLayout(1, L))
    c = gl.arange(0, 512, layout=gl.SliceLayout(0, L))
    for head in range(gl.program_id(0), 64, GRID):
        q = dev_load(Q, head * 512 + c, cache='.cg').to(gl.float32)
        m = dev_load(Sink, head, cache='.cg')
        denom = 1.0
        out = gl.full((512,), 0.0, gl.float32, gl.SliceLayout(0, L))
        for chunk in range(8):
            kv = dev_load(Cache, (chunk * 16 + r[:, None]) * 512 + c[None, :], cache='.cg').to(gl.float32)
            score = gl.sum(kv * q[None, :], 1) * 0.04419417382415922
            new_m = gl.maximum(gl.max(score, 0), m)
            alpha = gl.exp(m - new_m)
            p = gl.exp(score - new_m)
            denom = denom * alpha + gl.sum(p, 0)
            out = out * alpha + gl.sum(kv * p[:, None], 0)
            m = new_m
        dev_store((out / denom).to(gl.bfloat16), O, head * 512 + c, cache='.cg')

@g.jit
def activation(G, U, Y, N: gl.constexpr, GRID: gl.constexpr, ROUND: gl.constexpr=True):
    i = gl.arange(0, 256, layout=gl.BlockedLayout([1], [32], [4], [0]))
    for tile in range(gl.program_id(0), gl.cdiv(N, 256), GRID):
        j = tile * 256 + i
        gate = gl.minimum(dev_load(G, j, cache='.cg').to(gl.float32), 10.0)
        up = gl.minimum(gl.maximum(dev_load(U, j, cache='.cg').to(gl.float32), -10.0), 10.0)
        value = gate / (1.0 + gl.exp(-gate)) * up
        if ROUND:
            value = value.to(gl.bfloat16)
        dev_store(value, Y, j, cache='.cg')

@g.jit
def route(Logits, Tid, Token, Ids, Gamma):
    if gl.program_id(0) == 0:
        i = gl.arange(0, 8, layout=gl.BlockedLayout([1], [32], [4], [0]))
        token = dev_load(Token, 0, cache='.cg').to(gl.int64)
        eid = dev_load(Tid, token * 6 + i, i < 6, 0, cache='.cg').to(gl.int32)
        v = dev_load(Logits, eid, cache='.cg').to(gl.float32)
        sp = gl.where(v > 20.0, v, gl.log(1.0 + gl.exp(v)))
        w = gl.where(i < 6, gl.sqrt(sp), 0.0)
        w = 1.5 * w / gl.maximum(gl.sum(w, 0), 1e-20)
        dev_store(eid, Ids, i, i < 6, cache='.cg')
        dev_store(w, Gamma, i, i < 6, cache='.cg')

@g.jit
def full_layer(R, Out, Token, Cache, Cos, Sin, HAF, HAS, HAB, HFN, HFS, HFB, AN, FN, QN, KN, Sink, QA, QAS, QB, QBS, KVW, KVS, OA, OAS, OB, OBS, SW1, SS1, SW3, SS3, SW2, SS2, Gate, Tid, EW1, ES1, EW3, ES3, EW2, ES2, MixA, MixF, R2, X, XQ, QL, QLQ, Q, KV, QRot, KVRot, AO, AR, AQ, OL, OLQ, Attn, FF, FFQ, FFMX, SG, SU, SA, SAQ, SO, Logits, Ids, Gamma, EG, EU, EA, EAQ, EO, Sum, Counter, Health, GRID: gl.constexpr=64):
    mhc_mix(R, HAF, HAS, HAB, MixA, GRID)
    sync(Counter, Health, GRID)
    sinkhorn(MixA)
    sync(Counter, Health, GRID)
    mhc_pre(R, MixA, X, GRID)
    sync(Counter, Health, GRID)
    norm(X, AN, X, 4096, 1, True, GRID)
    sync(Counter, Health, GRID)
    quant(X, XQ, 4096, 128, GRID)
    sync(Counter, Health, GRID)
    matvec(XQ, QA, QAS, QL, 1024, 4096, 8, GRID)
    matvec(XQ, KVW, KVS, KV, 512, 4096, 8, GRID)
    sync(Counter, Health, GRID)
    norm(QL, QN, QL, 1024, 1, True, GRID, False)
    norm(KV, KN, KV, 512, 1, True, GRID, False)
    sync(Counter, Health, GRID)
    quant(QL, QLQ, 1024, 128, GRID)
    sync(Counter, Health, GRID)
    matvec(QLQ, QB, QBS, Q, 32768, 1024, 8, GRID)
    sync(Counter, Health, GRID)
    norm(Q, QN, Q, 512, 64, False, GRID, False)
    sync(Counter, Health, GRID)
    rope(Q, Cos, Sin, QRot, 64, False, GRID)
    rope(KV, Cos, Sin, KVRot, 1, False, GRID)
    sync(Counter, Health, GRID)
    if gl.program_id(0) == 0:
        i = gl.arange(0, 512, layout=gl.BlockedLayout([1], [32], [4], [0]))
        dev_store(dev_load(KVRot, i, cache='.cg'), Cache, 127 * 512 + i, cache='.cg')
    sync(Counter, Health, GRID)
    attention(QRot, KVRot, Cache, Sink, AO, GRID)
    sync(Counter, Health, GRID)
    rope(AO, Cos, Sin, AR, 64, True, GRID)
    sync(Counter, Health, GRID)
    quant(AR, AQ, 32768, 128, GRID)
    sync(Counter, Health, GRID)
    matvec(AQ, OA, OAS, OL, 8192, 4096, 8, GRID, True)
    sync(Counter, Health, GRID)
    quant(OL, OLQ, 8192, 128, GRID)
    sync(Counter, Health, GRID)
    matvec(OLQ, OB, OBS, Attn, 4096, 8192, 8, GRID)
    sync(Counter, Health, GRID)
    mhc_post(Attn, R, MixA, R2, GRID)
    sync(Counter, Health, GRID)
    mhc_mix(R2, HFN, HFS, HFB, MixF, GRID)
    sync(Counter, Health, GRID)
    sinkhorn(MixF)
    sync(Counter, Health, GRID)
    mhc_pre(R2, MixF, FF, GRID)
    sync(Counter, Health, GRID)
    norm(FF, FN, FF, 4096, 1, True, GRID)
    sync(Counter, Health, GRID)
    quant(FF, FFQ, 4096, 128, GRID)
    quant(FF, FFMX, 4096, 32, GRID)
    matvec(FF, Gate, Gate, Logits, 256, 4096, 16, GRID)
    sync(Counter, Health, GRID)
    route(Logits, Tid, Token, Ids, Gamma)
    matvec(FFQ, SW1, SS1, SG, 2048, 4096, 8, GRID)
    matvec(FFQ, SW3, SS3, SU, 2048, 4096, 8, GRID)
    sync(Counter, Health, GRID)
    activation(SG, SU, SA, 2048, GRID)
    sync(Counter, Health, GRID)
    quant(SA, SAQ, 2048, 128, GRID)
    sync(Counter, Health, GRID)
    matvec(SAQ, SW2, SS2, SO, 4096, 2048, 8, GRID)
    for slot in range(6):
        expert = dev_load(Ids, slot, cache='.cg').to(gl.int64)
        matvec(FFMX, EW1 + expert * 2048 * 2048, ES1 + expert * 2048 * 128, EG + slot * 2048, 2048, 4096, 4, GRID, FLOAT_OUT=True)
        matvec(FFMX, EW3 + expert * 2048 * 2048, ES3 + expert * 2048 * 128, EU + slot * 2048, 2048, 4096, 4, GRID, FLOAT_OUT=True)
    sync(Counter, Health, GRID)
    activation(EG, EU, EA, 6 * 2048, GRID, False)
    sync(Counter, Health, GRID)
    quant(EA, EAQ, 6 * 2048, 32, GRID)
    sync(Counter, Health, GRID)
    for slot in range(6):
        expert = dev_load(Ids, slot, cache='.cg').to(gl.int64)
        matvec(EAQ + slot * 2048, EW2 + expert * 4096 * 1024, ES2 + expert * 4096 * 64, EO + slot * 4096, 4096, 2048, 4, GRID, Gamma=Gamma + slot)
    sync(Counter, Health, GRID)
    i = gl.arange(0, 256, layout=gl.BlockedLayout([1], [32], [4], [0]))
    for tile in range(gl.program_id(0), 16, GRID):
        c = tile * 256 + i
        value = gl.full((256,), 0.0, gl.float32, gl.BlockedLayout([1], [32], [4], [0]))
        for reduce_slot in gl.static_range(6):
            value += dev_load(EO, reduce_slot * 4096 + c, cache='.cg').to(gl.float32)
        value = value.to(gl.bfloat16).to(gl.float32) + dev_load(SO, c, cache='.cg').to(gl.float32)
        dev_store(value.to(gl.bfloat16), Sum, c, cache='.cg')
    sync(Counter, Health, GRID)
    mhc_post(Sum, R2, MixF, Out, GRID)
