# SPDX-License-Identifier: MIT
"""The DS V4.1 MoE's shared expert in Gluon (``stages/moe.py``: ``load_x8_rows``,
``stage_shared``, ``smid_quant``; vLLM 8321d11), on batched scaled MFMA (needs the
patched Triton, DESIGN.md):

    shared  (36 tasks x token tiles, 16 gate + 16 up rows): the MXFP8 GEMV of
            vLLM's row-major e4m3 weights ([N/32, K/32] E8M0 scales) against the
            tile's MXFP8 x in LDS -- wave w over K steps w, w + 8, .. -- the waves
            summed in order, bf16, silu x up (vLLM's silu_and_mul_with_clamp),
            bf16 -> SMID; an odd task then quantizes its group of 32 (vLLM's
            mxfp8_e4m3_quantize) -> SMQ / SMC

The x tile in LDS is step-major, ``[step][token][wave][128]`` (step i, wave w:
columns (8 i + w) 128 ..), so a step's B operand is an index and a permute;
K = 5120 is 5 steps: steps 0-3 in one buffer, step 4 in another (LDS shapes are
powers of two)."""

from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from ..common.build import NOINLINE

from ..common.mx import clamp_fp8, fp8_pack4, vllm_mx_code
from ..common.ops import bf16_round
from ..common.sync import (
    fail,
    poll1,
    poll8,
    put,
)

MMA_S = gl.constexpr(gl.amd.AMDMFMALayout(4, [16, 16, 128], False, [8, 1, 1]))
AD = gl.constexpr(gl.DotOperandLayout(0, MMA_S, 16))
BD = gl.constexpr(gl.DotOperandLayout(1, MMA_S, 16))
SH4 = gl.constexpr(gl.SwizzledSharedLayout(1, 1, 1, [3, 2, 1, 0]))
SH3 = gl.constexpr(gl.SwizzledSharedLayout(1, 1, 1, [2, 1, 0]))
SH1 = gl.constexpr(gl.SwizzledSharedLayout(1, 1, 1, [0]))
# x tile loads: 16 contiguous bytes a thread
X4 = gl.constexpr(gl.BlockedLayout([1, 1, 1, 16], [1, 2, 4, 8], [1, 8, 1, 1], [3, 2, 1, 0]))
X3 = gl.constexpr(gl.BlockedLayout([1, 1, 16], [2, 4, 8], [8, 1, 1], [2, 1, 0]))
# weights [wave][row][128 k]: 16 bytes a thread, lane row + 16 kq; their scales
# [wave][row][4], a byte a lane, lane row + 16 group: the MFMA's own operand
# orders (a load in another order pays a shuffle / an LDS trip to convert)
A_LD = gl.constexpr(gl.BlockedLayout([1, 1, 16], [1, 16, 4], [8, 1, 1], [1, 2, 0]))
S_LD = gl.constexpr(gl.BlockedLayout([1, 1, 1], [1, 16, 4], [8, 1, 1], [1, 2, 0]))
RED3 = gl.constexpr(gl.BlockedLayout([1, 1, 1], [1, 4, 16], [8, 1, 1], [2, 1, 0]))
T512 = gl.constexpr(gl.BlockedLayout([1], [64], [8], [0]))
LIMIT = gl.constexpr(10.0)


# ---------------------------------------------------------------- the x tile
@gluon.jit
def _ar4(n0: gl.constexpr, n1: gl.constexpr, n2: gl.constexpr, n3: gl.constexpr, L: gl.constexpr):
    a = gl.arange(0, n0, layout=gl.SliceLayout(1, gl.SliceLayout(2, gl.SliceLayout(3, L))))
    b = gl.arange(0, n1, layout=gl.SliceLayout(0, gl.SliceLayout(2, gl.SliceLayout(3, L))))
    c = gl.arange(0, n2, layout=gl.SliceLayout(0, gl.SliceLayout(1, gl.SliceLayout(3, L))))
    d = gl.arange(0, n3, layout=gl.SliceLayout(0, gl.SliceLayout(1, gl.SliceLayout(2, L))))
    return a[:, None, None, None], b[None, :, None, None], c[None, None, :, None], d[None, None, None, :]


@gluon.jit
def _ar3(n0: gl.constexpr, n1: gl.constexpr, n2: gl.constexpr, L: gl.constexpr):
    a = gl.arange(0, n0, layout=gl.SliceLayout(1, gl.SliceLayout(2, L)))
    b = gl.arange(0, n1, layout=gl.SliceLayout(0, gl.SliceLayout(2, L)))
    c = gl.arange(0, n2, layout=gl.SliceLayout(0, gl.SliceLayout(1, L)))
    return a[:, None, None], b[None, :, None], c[None, None, :]


@gluon.jit
def gather_nd(smem, idx):
    """``smem`` (a flat LDS array) at an index tensor of any rank, in its layout."""
    v = smem.gather(gl.reshape(idx, [idx.numel]), 0)
    return gl.convert_layout(gl.reshape(v, idx.shape), idx.type.layout)


@gluon.jit
def scatter_nd(smem, vals, idx):
    smem.scatter(gl.reshape(vals, [vals.numel]),
                 gl.convert_layout(gl.reshape(idx, [idx.numel]), gl.reshape(vals, [vals.numel]).type.layout), 0)


@gluon.jit
def x_row(j, t0, n, tab, GATHER: gl.constexpr, ORDER0: gl.constexpr, TOPK: gl.constexpr):
    """The token of LDS tile row j (any rank): t0 + j, or (``GATHER``, upstream's
    ``gather_x8r``) the token of pick t0 + min(j, n - 1) in the route order."""
    if GATHER:
        return gather_nd(tab, ORDER0 + t0 + gl.minimum(j, n - 1)) // TOPK
    else:
        return t0 + j


@gluon.jit(noinline=NOINLINE)
def load_x_tile(words, codes, rdy, t0, n, xa, xb, ca, cb, tab, tag, tid, dl, health, rank,
                HIDDEN: gl.constexpr, GATHER: gl.constexpr, ORDER0: gl.constexpr,
                TOPK: gl.constexpr):  # fmt: skip
    """Rows t0 .. t0 + n (n <= 16; rows past them repeat the last) of the MXFP8
    region ``words`` / ``codes`` (K = 5120: 160 codes a row) into the LDS tile
    (xa: steps 0-3, xb: step 4; ca / cb their codes, u64 views): the first n
    threads wait on their token's RDY first. Plain loads (a row's words and its
    160 i32 codes are whole 128 B lines, written before its flag, never read
    before), the codes packed to bytes in LDS. ``GATHER``: row j is the
    token of pick t0 + j in the route order (``x_row``), past one token tile."""
    gl.static_assert(HIDDEN == 5120)
    w0 = tid < n
    tok = x_row(gl.minimum(tid, n - 1), t0, n, tab, GATHER, ORDER0, TOPK)
    v, f = poll1(rdy.to(gl.uint64) + tok.to(gl.uint64) * 8, tag, w0 & (tid < 16), dl,
                 health)  # fmt: skip
    fail(health, f, rank, 7, w0 & (tid < 16))
    gl.barrier()
    wp = words.to(gl.pointer_type(gl.uint8))
    i4, t4, w4, k4 = _ar4(4, 16, 8, 128, X4)
    # rows past n (padding columns, discarded) read the scratch past the region
    # (a gathered row: the last pick's again)
    row = x_row(t4, t0, n, tab, GATHER, ORDER0, TOPK)
    xa.store(gl.load(wp + row * HIDDEN + (i4 * 8 + w4) * 128 + k4))
    t3, w3, k3 = _ar3(16, 8, 128, X3)
    row3 = x_row(t3, t0, n, tab, GATHER, ORDER0, TOPK)
    xb.store(gl.load(wp + row3 * HIDDEN + (32 + w3) * 128 + k3))
    # codes: an i32 a group (160 a row: whole 128 B lines) -> the tile's bytes,
    # step-major like the words: group g = (8 i + w) 4 + gg at [i][token][w][gg]
    # (ca: steps 0-3, cb: step 4; a masked lane writes the spare past them)
    cp = codes.to(gl.pointer_type(gl.int32))
    ca8 = ca._reinterpret(gl.uint8, [4096], SH1)
    cb8 = cb._reinterpret(gl.uint8, [1024], SH1)
    for k in gl.static_range(5):
        e = tid + 512 * k
        et = e // 160
        g = e % 160
        live = et < n
        etok = x_row(gl.minimum(et, n - 1), t0, n, tab, GATHER, ORDER0, TOPK)
        code = gl.load(cp + etok * 160 + g, mask=live, other=127).to(gl.uint8)
        i = g // 32
        at = ((et * 8 + (g // 4) % 8) * 4) + g % 4
        ca8.scatter(code, gl.where(live & (i < 4), i * 512 + at, 4095), 0)
        cb8.scatter(code, gl.where(live & (i == 4), at, 1023), 0)
    gl.barrier()


# ---------------------------------------------------------------- the GEMV
@gluon.jit
def gemv_rows16(w, ws, row0, xa, xb, ca8, cb8, HIDDEN: gl.constexpr):
    """[8 waves][16 rows][16 tokens] partials: rows row0 .. +16 of the MXFP8
    weight (row-major e4m3 [N, K], E8M0 [N/32, K/32]) against the LDS tile,
    wave w over K steps w, w + 8, .. (5 a wave), weights all issued first."""
    gl.static_assert(HIDDEN == 5120)
    wa, ra, ka = _ar3(8, 16, 128, A_LD)
    ws_, rs_, gs_ = _ar3(8, 16, 4, S_LD)
    wp = w.to(gl.pointer_type(gl.uint8))
    sp = ws.to(gl.pointer_type(gl.uint8))
    SAL: gl.constexpr = gl.amd.cdna4.get_mfma_scale_layout(AD, [8, 16, 4])
    SBL: gl.constexpr = gl.amd.cdna4.get_mfma_scale_layout(BD, [8, 16, 4])
    avs = ()
    sas = ()
    for i in gl.static_range(5):
        k0 = (8 * i + wa) * 128
        avs = avs + (gl.load(wp + (row0 + ra) * HIDDEN + k0 + ka, cache_modifier=".cg"),)
        sas = sas + (gl.load(sp + ((row0 + rs_) // 32) * (HIDDEN // 32) + (8 * i + ws_) * 4
                             + gs_),)  # fmt: skip
    acc = gl.zeros((8, 16, 16), gl.float32, MMA_S)
    for i in gl.static_range(5):
        a = gl.convert_layout(avs[i], AD)
        sa = gl.convert_layout(sas[i], SAL)
        if i < 4:
            b = xa._reinterpret(gl.uint8, [4, 16, 8, 128], SH3).index(i).permute((1, 2, 0)).load(BD)
            sb = ca8._reinterpret(gl.uint8, [8, 16, 8, 4], SH3).index(i).permute((1, 0, 2)).load(SBL)
        else:
            b = xb.permute((1, 2, 0)).load(BD)
            sb = cb8._reinterpret(gl.uint8, [2, 16, 8, 4], SH3).index(0).permute((1, 0, 2)).load(SBL)
        acc = gl.amd.cdna4.mfma_scaled(a, sa, "e4m3", b, sb, "e4m3", acc)
    return acc


@gluon.jit
def wave_sum(acc, red, r, tl):
    """Row r, token tl of the waves' [8][16][16] partials, summed in wave order."""
    red.slice(0, 2048).store(gl.reshape(gl.convert_layout(acc, RED3), (2048,)))
    gl.barrier()
    v = red.gather(r * 16 + tl, 0)
    for w in gl.static_range(1, 8):
        v = v + red.gather(w * 256 + r * 16 + tl, 0)
    gl.barrier()
    return v


@gluon.jit
def silu_mul(g, u):
    """vLLM's silu_and_mul_with_clamp: gate min limit, up clamped; silu = g /
    (1 + expf(-g)) rounded to bf16, then times up."""
    g = bf16_round(gl.minimum(g, LIMIT))
    u = gl.maximum(gl.minimum(u, LIMIT), -LIMIT)
    act = bf16_round(g / (1.0 + gl.exp(-g)))
    return act * u


@gluon.jit(noinline=NOINLINE)
def stage_shared(task, t0, n, sgu, sgu_s, smid, smq, smc, xa, xb, ca8, cb8, red, pair,
                 tag, tid, dl, health, rank,
                 SH_INTER: gl.constexpr, HIDDEN: gl.constexpr):  # fmt: skip
    """Gate rows 16 task .., up rows SH_INTER + 16 task .. against tokens t0 ..
    t0 + n (their x in LDS) -> bf16 -> silu x up -> SMID; an odd task then
    quantizes its group of 32."""
    TASKS: gl.constexpr = SH_INTER // 16
    r = tid % 16
    tl = gl.minimum(tid // 16, 15)
    live = tid < 16 * n
    gate = wave_sum(gemv_rows16(sgu, sgu_s, task * 16, xa, xb, ca8, cb8, HIDDEN), red, r, tl)
    up = wave_sum(gemv_rows16(sgu, sgu_s, (TASKS + task) * 16, xa, xb, ca8, cb8, HIDDEN), red,
                  r, tl)  # fmt: skip
    v = bf16_round(silu_mul(bf16_round(gate), bf16_round(up)))
    t = t0 + tl
    put(smid, t * SH_INTER + task * 16 + r, v, tag, live)
    pair.scatter(v, gl.where(live, tid, 511), 0)
    gl.barrier()
    if task % 2 == 1:
        smid_quant(task, t0, n, smid, smq, smc, pair, tag, tid, dl, health, rank, SH_INTER)
    gl.barrier()


@gluon.jit
def smid_quant(task, t0, n, smid, smq, smc, pair, tag, tid, dl, health, rank,
               SH_INTER: gl.constexpr):  # fmt: skip
    """Odd task: its 16 columns (LDS ``pair``) and the even task's (SMID) are a
    quantize group of 32 (vLLM's mxfp8_e4m3_quantize) -> SMQ / SMC, a thread a
    token."""
    live = tid < n
    tt = gl.minimum(tid, 15)
    t = t0 + tt
    sb = smid.to(gl.uint64) + ((t * SH_INTER + (task - 1) * 16) * 8).to(gl.uint64)
    a0, a1, a2, a3, a4, a5, a6, a7, f0 = poll8(sb, sb + 8, sb + 16, sb + 24, sb + 32, sb + 40,
                                               sb + 48, sb + 56, tag, live, dl, health)  # fmt: skip
    b0, b1, b2, b3, b4, b5, b6, b7, f1 = poll8(sb + 64, sb + 72, sb + 80, sb + 88, sb + 96,
                                               sb + 104, sb + 112, sb + 120, tag, live, dl,
                                               health)  # fmt: skip
    fail(health, gl.maximum(f0, f1), rank, 8, live)
    vals = (a0, a1, a2, a3, a4, a5, a6, a7, b0, b1, b2, b3, b4, b5, b6, b7)
    fv = ()
    for j in gl.static_range(16):
        fv = fv + (vals[j].to(gl.float32, bitcast=True),)
    for j in gl.static_range(16):
        fv = fv + (pair.gather(tt * 16 + j, 0),)
    amax = gl.abs(fv[0])
    for j in gl.static_range(1, 32):
        amax = gl.maximum(amax, gl.abs(fv[j]))
    code = vllm_mx_code(amax)
    inv = ((254 - code) << 23).to(gl.float32, bitcast=True)
    g = task // 2
    for w in gl.static_range(8):
        word = fp8_pack4(clamp_fp8(fv[4 * w] * inv), clamp_fp8(fv[4 * w + 1] * inv),
                         clamp_fp8(fv[4 * w + 2] * inv), clamp_fp8(fv[4 * w + 3] * inv))  # fmt: skip
        put(smq, t * (SH_INTER // 4) + g * 8 + w, word, tag, live)
    put(smc, t * (SH_INTER // 32) + g, code, tag, live)
