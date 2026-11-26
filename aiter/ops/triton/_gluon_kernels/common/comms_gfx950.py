# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Gluon device helpers of the multi-GPU comms kernels, for gfx950 only: bounded polls of
IPC inbox slots and tagged words, error records, 16-byte vector loads and stores, bf16
and e4m3 packing, lane exchanges, and the rank sum, RMSNorm and MXFP8 quant of a row.

The helpers use gfx950 instructions (the sc0 sc1 cache bits, v_permlane32_swap,
v_cvt_pk_bf16_f32 and OCP e4m3) and the CDNA4 buffer loads and stores. A thread holds 4
dwords of a [64 W, 4] block layout (W warps). The IPC buffer of each rank starts with
the error record [flag, rank, colour, site]."""

from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.language.core import PropagateNan as _PropagateNan
from triton.language.extra.hip import libdevice

from aiter.ops.triton._gluon_kernels.common.utils import mx_e8m0_scale

_NAN_ALL: gl.constexpr = gl.constexpr(_PropagateNan.ALL)
_SENT_SIGN: gl.constexpr = gl.constexpr(-0x7FFF8000)  # 0x80008000
_POLL_VGPR = 64  # The first of the fixed VGPRs that a poll loads into.
_POLL_MAX: gl.constexpr = gl.constexpr(8)  # The most words in one poll.


def _poll(n, tagged, system):
    """Bounded poll of n IPC words a lane, in the lanes where act is not 0. A Lamport poll
    loads n 16-byte slots until no 16-bit word is 0xFFFF. A tagged poll loads n 8-byte
    words ((tag << 32) | value) until the high word of each is tag. A poll also stops
    when the error word is set or the time limit passes. Outputs: failed (0 ok, 1 timed
    out, 2 error record seen), the 4 n slot dwords or the n low words, scratch. Inputs:
    act, tag (tagged only), n addresses, the error-word address, the time limit in 100
    MHz ticks. system: system-scope loads (peer memory), else device scope."""
    nw = n if tagged else 4 * n  # output words
    step = 2 if tagged else 4  # VGPRs a load
    b = _POLL_VGPR
    lbl = ".Ltp_" if tagged else ".Llp_"
    scratch = ["=&s", "=&s", "=&v", "=&v"] + ([] if tagged else ["=&v", "=&v"])
    scratch += ["=&s"]
    saved, now, dl, errv = range(nw + 1, nw + 5)
    acc, tmp = nw + 5, nw + 6  # Lamport only
    pend = nw + len(scratch)
    act = pend + 1
    tag = act + 1  # tagged only
    a0 = act + (2 if tagged else 1)
    aerr = a0 + n
    ticks = aerr + 1
    width = "dwordx2" if tagged else "dwordx4"
    scope = "sc0 sc1" if system else "sc1"
    lines = [
        "s_mov_b32 $0, 0",
        f"v_cmp_ne_u32 vcc, 0, ${act}",
        f"s_and_saveexec_b64 ${saved}, vcc",
        f"s_cbranch_execz {lbl}done${{:uid}}",
        f"s_memrealtime ${now}",
        "s_waitcnt lgkmcnt(0)",
        f"v_lshl_add_u64 ${dl}, ${now}, 0, ${ticks}",
        f"{lbl}loop${{:uid}}:",
    ]
    lines += [
        f"global_load_{width} v[{b + step * i}:{b + step * i + step - 1}], "
        f"${a0 + i}, off {scope}"
        for i in range(n)
    ]
    lines += [
        f"global_load_dword ${errv}, ${aerr}, off sc0 sc1",
        "s_waitcnt vmcnt(0)",
    ]
    if tagged:
        lines += [f"v_cmp_ne_u32 vcc, ${tag}, v{b + 1}", f"s_mov_b64 ${pend}, vcc"]
        for i in range(1, n):
            lines += [
                f"v_cmp_ne_u32 vcc, ${tag}, v{b + 2 * i + 1}",
                f"s_or_b64 ${pend}, ${pend}, vcc",
            ]
    else:
        lines += [f"v_pk_max_u16 ${acc}, v{b}, v{b + 1}"]
        lines += [f"v_pk_max_u16 ${acc}, ${acc}, v{b + i}" for i in range(2, nw)]
        lines += [
            f"v_and_b32 ${tmp}, 0xffff, ${acc}",
            f"v_cmp_eq_u32 vcc, 0xffff, ${tmp}",
            f"s_mov_b64 ${pend}, vcc",
            f"v_cmp_lt_u32 vcc, 0xfffeffff, ${acc}",
            f"s_or_b64 ${pend}, ${pend}, vcc",
        ]
    lines += [
        f"s_cmp_eq_u64 ${pend}, 0",
        f"s_cbranch_scc1 {lbl}ok${{:uid}}",
        f"v_cmp_ne_u32 vcc, 0, ${errv}",
        "s_cmp_lg_u64 vcc, 0",
        f"s_cbranch_scc1 {lbl}give${{:uid}}",
        f"s_memrealtime ${now}",
        "s_waitcnt lgkmcnt(0)",
        f"v_cmp_ge_u64 vcc, ${now}, ${dl}",
        "s_cmp_lg_u64 vcc, 0",
        f"s_cbranch_scc1 {lbl}out${{:uid}}",
        "s_sleep 1",
        f"s_branch {lbl}loop${{:uid}}",
        f"{lbl}give${{:uid}}:",
        "s_mov_b32 $0, 2",
        f"s_branch {lbl}ok${{:uid}}",
        f"{lbl}out${{:uid}}:",
        "s_mov_b32 $0, 1",
        f"{lbl}ok${{:uid}}:",
    ]
    src = [b + 2 * i for i in range(n)] if tagged else [b + i for i in range(nw)]
    lines += [f"v_mov_b32 ${1 + i}, v{r}" for i, r in enumerate(src)]
    lines += [f"{lbl}done${{:uid}}:", f"s_mov_b64 exec, ${saved}"]
    cons = (
        ["=&s"]
        + ["=&v"] * nw
        + scratch
        + ["v"] * (a0 + n + 2 - act)
        + ["~{memory}", "~{vcc}", "~{scc}"]
        + [f"~{{v{b + i}}}" for i in range(step * n)]
    )
    return "\n".join(lines), ",".join(cons)


def _errec(n):
    """[1, rank, colour, site] into the error records of n buffers, system scope, from
    the active lanes. The high pair goes first: waiters read the low word."""
    lines = [
        "v_cmp_ne_u32 vcc, 0, $2",
        "s_and_saveexec_b64 $1, vcc",
        "s_cbranch_execz .Ler${:uid}",
    ]
    lines += [
        f"global_store_dwordx2 ${3 + i}, ${3 + n}, off offset:8 sc0 sc1"
        for i in range(n)
    ]
    lines += [f"global_store_dwordx2 ${3 + i}, ${4 + n}, off sc0 sc1" for i in range(n)]
    lines += [".Ler${:uid}:", "s_mov_b64 exec, $1", "v_mov_b32 $0, 0"]
    cons = ["=v", "=&s"] + ["v"] * (n + 3) + ["~{memory}", "~{vcc}", "~{scc}"]
    return "\n".join(lines), ",".join(cons)


def _bfly(op):
    """A lane butterfly over offsets 32, 16, 8, 4, 2, 1 with a VALU op (permlane swaps, then
    DPP): every lane gets the reduction of its wave. The s_nop wait states are explicit, the
    compiler adds none for inline asm."""
    return "\n".join(
        [
            "v_mov_b32 $0, $2",
            "v_mov_b32 $1, $2",
            "s_nop 1",
            "v_permlane32_swap_b32 $1, $0",
            f"{op} $0, $1, $0",
            "v_mov_b32 $1, $0",
            "s_nop 1",
            "v_permlane16_swap_b32 $0, $1",
            f"{op} $0, $0, $1",
            "v_mov_b32 $1, $0",
            "s_nop 1",
            "v_mov_b32_dpp $1, $1 row_shr:8 row_mask:0xf bank_mask:0xc",
            "s_nop 1",
            "v_mov_b32_dpp $1, $0 row_shl:8 row_mask:0xf bank_mask:0x3",
            f"{op} $0, $0, $1",
            "v_mov_b32 $1, $0",
            "s_nop 1",
            "v_mov_b32_dpp $1, $1 row_shr:4 row_mask:0xf bank_mask:0xa",
            "s_nop 1",
            "v_mov_b32_dpp $1, $0 row_shl:4 row_mask:0xf bank_mask:0x5",
            f"{op} $0, $0, $1",
            "v_mov_b32 $1, $0",
            "s_nop 1",
            "v_mov_b32_dpp $1, $1 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf",
            f"{op} $0, $0, $1",
            "v_mov_b32 $1, $0",
            "s_nop 1",
            "v_mov_b32_dpp $1, $1 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf",
            f"{op} $0, $0, $1",
        ]
    )


_POLL = {
    (n, t, s): _poll(n, t, s)
    for n in range(1, int(_POLL_MAX) + 1)
    for t in (False, True)
    for s in (False, True)
}
_ERREC = {n: _errec(n) for n in (2, 4, 8)}
BFLY_ADD: gl.constexpr = gl.constexpr(_bfly("v_add_f32"))


@gluon.constexpr_function
def _poll_asm(n, tagged, system):
    return _POLL[(int(n), bool(tagged), bool(system))][0]


@gluon.constexpr_function
def _poll_cons(n, tagged, system):
    return _POLL[(int(n), bool(tagged), bool(system))][1]


@gluon.constexpr_function
def _errec_asm(n):
    return _ERREC[int(n)][0]


@gluon.constexpr_function
def _errec_cons(n):
    return _ERREC[int(n)][1]


@gluon.constexpr_function
def _chunk(ns, c):
    return min(int(_POLL_MAX), int(ns) - int(c))


@gluon.jit
def tids(W: gl.constexpr):
    """The thread index, one value a thread."""
    LV: gl.constexpr = gl.BlockedLayout([1, 4], [64, 1], [W, 1], [1, 0])
    return gl.arange(0, 64 * W, layout=gl.SliceLayout(1, LV))


@gluon.jit
def dword_idx(W: gl.constexpr):
    LV: gl.constexpr = gl.BlockedLayout([1, 4], [64, 1], [W, 1], [1, 0])
    return gl.arange(0, 4, layout=gl.SliceLayout(0, LV))[None, :]


@gluon.jit
def _split4(x, W: gl.constexpr):
    """[threads, 4] dwords to 4 tensors of one dword a thread."""
    k = dword_idx(W)
    return (
        gl.sum(gl.where(k == 0, x, 0), axis=1),
        gl.sum(gl.where(k == 1, x, 0), axis=1),
        gl.sum(gl.where(k == 2, x, 0), axis=1),
        gl.sum(gl.where(k == 3, x, 0), axis=1),
    )


@gluon.jit
def _join4(w, W: gl.constexpr):
    k = dword_idx(W)
    return gl.where(
        k == 0,
        w[0][:, None],
        gl.where(k == 1, w[1][:, None], gl.where(k == 2, w[2][:, None], w[3][:, None])),
    )


@gluon.jit
def ld4(ptr, dw, mask, W: gl.constexpr):
    """The 16 bytes at dword dw of ptr (one vector a thread), as 4 dwords."""
    k = dword_idx(W)
    x = gl.amd.cdna4.buffer_load(ptr, dw[:, None] + k, mask=mask[:, None] & (k < 4))
    return _split4(x, W)


@gluon.jit
def st4(ptr, dw, w, mask, W: gl.constexpr, CACHE: gl.constexpr = ""):
    k = dword_idx(W)
    gl.amd.cdna4.buffer_store(
        _join4(w, W), ptr, dw[:, None] + k, mask=mask[:, None] & (k < 4), cache=CACHE
    )


@gluon.jit
def st2(ptr, dw, w0, w1, mask, W: gl.constexpr):
    L2: gl.constexpr = gl.BlockedLayout([1, 2], [64, 1], [W, 1], [1, 0])
    S2: gl.constexpr = gl.SliceLayout(1, L2)
    k = gl.arange(0, 2, layout=gl.SliceLayout(0, L2))[None, :]
    v = gl.where(
        k == 0, gl.convert_layout(w0, S2)[:, None], gl.convert_layout(w1, S2)[:, None]
    )
    off = gl.convert_layout(dw, S2)[:, None] + k
    gl.amd.cdna4.buffer_store(
        v, ptr, off, mask=gl.convert_layout(mask, S2)[:, None] & (k < 2)
    )


@gluon.jit
def canon(w):
    """A 16-bit word 0xFFFF becomes 0x7FFF: still a NaN, but not the empty mark."""
    x = w ^ -1
    nz = ((x & 0x7FFF7FFF) + 0x7FFF7FFF) | x
    return w ^ ((nz & _SENT_SIGN) ^ _SENT_SIGN)


@gluon.jit
def bf_lo(w):
    return (w << 16).to(gl.float32, bitcast=True)


@gluon.jit
def bf_hi(w):
    return (w & -65536).to(gl.float32, bitcast=True)


@gluon.jit
def pack_bf16(lo, hi):
    """Two fp32 values to one dword of 2 bf16 (round to nearest even), lo first."""
    return gl.inline_asm_elementwise(
        "v_cvt_pk_bf16_f32 $0, $1, $2", "=v,v,v", [lo, hi], gl.int32, True, 1
    )


@gluon.jit
def fp8_pack4(a, b, c, d):
    """Four fp32 values in range to one dword of 4 e4m3 (round to nearest even)."""
    return gl.inline_asm_elementwise(
        "v_cvt_pk_fp8_f32 $0, $1, $2\nv_cvt_pk_fp8_f32 $0, $3, $4 op_sel:[0,0,1]",
        "=&v,v,v,v,v",
        [a, b, c, d],
        gl.int32,
        True,
        1,
    )


@gluon.jit
def bfly_op(x, ASM: gl.constexpr):
    out = gl.inline_asm_elementwise(ASM, "=&v,=&v,v", [x], (x.dtype, x.dtype), True, 1)
    return out[0]


@gluon.jit
def quad_xor(x, P: gl.constexpr):
    """The value of lane ^ P, for P = 1 or 2."""
    if P == 1:
        return gl.inline_asm_elementwise(
            "s_nop 1\nv_mov_b32_dpp $0, $1 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf",
            "=&v,v",
            [x],
            gl.float32,
            True,
            1,
        )
    else:
        return gl.inline_asm_elementwise(
            "s_nop 1\nv_mov_b32_dpp $0, $1 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf",
            "=&v,v",
            [x],
            gl.float32,
            True,
            1,
        )


@gluon.jit
def fail_record(bufs, rank, colour, site: gl.constexpr, active):
    """Writes [1, rank, colour, site] to the error record of each rank."""
    lo = (rank.to(gl.int64) << 32) | 1
    hi = (gl.full(active.shape, site, gl.int64, active.type.layout) << 32) | colour.to(
        gl.uint32
    ).to(gl.int64)
    lo = gl.full(active.shape, 0, gl.int64, active.type.layout) + lo
    z = gl.zeros_like(active.to(gl.int32))
    args = (active.to(gl.int32),)
    for b in gl.static_range(len(bufs)):
        args = args + ((bufs[b] + z).to(gl.pointer_type(gl.int64)),)
    gl.inline_asm_elementwise(
        _errec_asm(len(bufs)),
        _errec_cons(len(bufs)),
        args + (hi, lo),
        (gl.int32, gl.int64),
        False,
        1,
    )


@gluon.jit
def peer(bufs, d, WORLD: gl.constexpr):
    """bufs[d] for a uniform d in 1 to WORLD - 1."""
    p = bufs[1]
    for q in gl.static_range(2, WORLD):
        p = gl.where(d == q, bufs[q], p)
    return p


@gluon.jit
def poll_slots(own, addrs, act, ticks, NS: gl.constexpr):
    """Lamport poll of the NS 16-byte slots in addrs, system scope. own is the buffer of
    this rank (its error record). Returns failed and the 4 dwords of each slot."""
    z = gl.zeros_like(act.to(gl.int32))
    failed = z
    words = ()
    for c in gl.static_range(0, NS, _POLL_MAX):
        part = ()
        for i in gl.static_range(_chunk(NS, c)):
            part = part + (addrs[c + i],)
        out = gl.inline_asm_elementwise(
            _poll_asm(_chunk(NS, c), False, True),
            _poll_cons(_chunk(NS, c), False, True),
            (act.to(gl.int32),) + part + (own + z, ticks + z.to(gl.int64)),
            (gl.int32,) * (1 + 4 * _chunk(NS, c))
            + (gl.int64, gl.int64, gl.int64, gl.int32, gl.int32, gl.int32, gl.int64),
            False,
            1,
        )
        failed = gl.maximum(failed, out[0])
        for i in gl.static_range(_chunk(NS, c)):
            words = words + (
                (out[1 + 4 * i], out[2 + 4 * i], out[3 + 4 * i], out[4 + 4 * i]),
            )
    return failed, words


@gluon.jit
def poll_tags(own, addrs, act, tag, ticks, NS: gl.constexpr, SYSTEM: gl.constexpr):
    """Poll of the NS tagged words at addrs (int64 pointers) for tag: system scope for
    peer memory, else device scope. Returns failed and the low word of each."""
    z = gl.zeros_like(act.to(gl.int32))
    failed = z
    vals = ()
    for c in gl.static_range(0, NS, _POLL_MAX):
        part = ()
        for i in gl.static_range(_chunk(NS, c)):
            part = part + (addrs[c + i],)
        out = gl.inline_asm_elementwise(
            _poll_asm(_chunk(NS, c), True, SYSTEM),
            _poll_cons(_chunk(NS, c), True, SYSTEM),
            (act.to(gl.int32), tag + z) + part + (own + z, ticks + z.to(gl.int64)),
            (gl.int32,) * (1 + _chunk(NS, c))
            + (gl.int64, gl.int64, gl.int64, gl.int32, gl.int64),
            False,
            1,
        )
        failed = gl.maximum(failed, out[0])
        for i in gl.static_range(_chunk(NS, c)):
            vals = vals + (out[1 + i],)
    return failed, vals


@gluon.jit
def row_sum(own_w, got, rank, WORLD: gl.constexpr, V: gl.constexpr, tid):
    """The bf16 words of a row summed over the ranks in rank order, in fp32 with one
    rounding. own_w: the words of this rank. got[(d - 1) V + j]: the words of rank + d.
    """
    hw = ()
    for j in gl.static_range(V):
        for e in gl.static_range(4):
            acc_lo = gl.zeros_like(tid).to(gl.float32)
            acc_hi = gl.zeros_like(tid).to(gl.float32)
            for src in gl.static_range(WORLD):
                x = own_w[j][e]
                for d in gl.static_range(1, WORLD):
                    x = gl.where((rank + d) % WORLD == src, got[(d - 1) * V + j][e], x)
                if src == 0:
                    acc_lo = bf_lo(x)
                    acc_hi = bf_hi(x)
                else:
                    acc_lo = acc_lo + bf_lo(x)
                    acc_hi = acc_hi + bf_hi(x)
            hw = hw + (pack_bf16(acc_lo, acc_hi),)
    return hw


@gluon.jit
def rmsnorm_row(
    hw,
    res_w,
    w_w,
    act,
    eps,
    HIDDEN: gl.constexpr,
    V: gl.constexpr,
    GEMMA: gl.constexpr,
    HAS_RES: gl.constexpr,
    W: gl.constexpr,
):
    """s = h + residual in fp32, and the bf16 words of y = s * rstd * w' (w' = 1 + w
    for GEMMA). A thread adds the squares of its 8 V values in column order (no fma),
    a butterfly adds the lanes, then the waves are added in order."""
    tid = tids(W)
    lane = tid % 64
    wave = tid // 64
    NWA: gl.constexpr = (HIDDEN // 8 // V + 63) // 64
    s = ()
    for j in gl.static_range(V):
        for e in gl.static_range(4):
            if HAS_RES:
                s = s + (
                    bf_lo(hw[4 * j + e]) + bf_lo(res_w[j][e]),
                    bf_hi(hw[4 * j + e]) + bf_hi(res_w[j][e]),
                )
            else:
                s = s + (bf_lo(hw[4 * j + e]), bf_hi(hw[4 * j + e]))
    sq = s[0] * s[0]
    for i in gl.static_range(1, 8 * V):
        sq = sq + s[i] * s[i]
    sq = bfly_op(gl.where(act, sq, 0.0), BFLY_ADD)
    red = gl.allocate_shared_memory(
        gl.float32, [64], gl.SwizzledSharedLayout(1, 1, 1, [0])
    )
    red.scatter(sq, gl.where(lane == 0, wave, 63), 0)
    gl.barrier()
    zi = gl.zeros_like(tid)
    tot = red.gather(zi, 0)
    for wv in gl.static_range(1, NWA):
        tot = tot + red.gather(zi + wv, 0)
    var = gl.div_rn(tot, gl.full(tot.shape, HIDDEN, gl.float32, tot.type.layout))
    one = gl.full(tot.shape, 1.0, gl.float32, tot.type.layout)
    rstd = gl.div_rn(one, gl.sqrt_rn(var + eps))
    y = ()
    for j in gl.static_range(V):
        for e in gl.static_range(4):
            wl = bf_lo(w_w[j][e])
            wh = bf_hi(w_w[j][e])
            if GEMMA:
                wl = 1.0 + wl
                wh = 1.0 + wh
            y = y + (
                pack_bf16(
                    (s[8 * j + 2 * e] * rstd) * wl, (s[8 * j + 2 * e + 1] * rstd) * wh
                ),
            )
    return s, y


@gluon.jit
def store_rmsnorm_row(
    s,
    y,
    xq,
    xs,
    res_out,
    t,
    act,
    A: gl.constexpr,
    HIDDEN: gl.constexpr,
    V: gl.constexpr,
    QUANT: gl.constexpr,
    W: gl.constexpr,
):
    """Stores row t: the MXFP8 of y (if QUANT) and the new residual bf16(s). A quad of
    lanes holds a group of 32 columns. The scale is mx_e8m0_scale(amax, 8)."""
    tid = tids(W)
    lane = tid % 64
    if QUANT:
        for j in gl.static_range(V):
            vec = j * A + tid
            yv = ()
            for e in gl.static_range(4):
                yv = yv + (bf_lo(y[4 * j + e]), bf_hi(y[4 * j + e]))
            am = gl.abs(yv[0])
            for e in gl.static_range(1, 8):
                am = gl.maximum(am, gl.abs(yv[e]), propagate_nan=_NAN_ALL)
            am = gl.maximum(am, quad_xor(am, 1), propagate_nan=_NAN_ALL)
            am = gl.maximum(am, quad_xor(am, 2), propagate_nan=_NAN_ALL)
            sb = mx_e8m0_scale(am, 8)
            # y / 2^(sb - 127) with ldexp: 2^-127 (sb 254) is not a normal fp32 value.
            n = 127 - sb.to(gl.int32)
            q = ()
            for e in gl.static_range(8):
                q = q + (libdevice.ldexp(yv[e], n),)
            c0 = fp8_pack4(q[0], q[1], q[2], q[3])
            c1 = fp8_pack4(q[4], q[5], q[6], q[7])
            st2(xq, t * (HIDDEN // 4) + 2 * vec, c0, c1, act, W)
            gl.store(xs + t * (HIDDEN // 32) + vec // 4, sb, mask=act & (lane % 4 == 0))
    for j in gl.static_range(V):
        vec = j * A + tid
        rw = ()
        for e in gl.static_range(4):
            rw = rw + (pack_bf16(s[8 * j + 2 * e], s[8 * j + 2 * e + 1]),)
        st4(res_out, t * (HIDDEN // 2) + vec * 4, rw, act, W)
