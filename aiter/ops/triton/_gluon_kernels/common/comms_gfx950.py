# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Gluon device helpers of the multi-GPU comms kernels, for gfx950 only: bounded Lamport
polls of IPC inbox slots, error records, 16-byte vector loads and stores, bf16 and e4m3
packing, and lane exchanges.

The helpers use gfx950 instructions (the sc0 sc1 cache bits, v_permlane32_swap,
v_cvt_pk_bf16_f32 and OCP e4m3) and the CDNA4 buffer loads and stores. A thread holds 4
dwords of a [64 W, 4] block layout (W warps). The IPC buffer of each rank starts with
the error record [flag, rank, colour, site]."""

from triton.experimental import gluon
from triton.experimental.gluon import language as gl

_SENT_SIGN: gl.constexpr = gl.constexpr(-0x7FFF8000)  # 0x80008000
_POLL_VGPR = 64  # The first of the fixed VGPRs that a poll loads into.
_POLL_MAX: gl.constexpr = gl.constexpr(8)  # The most slots in one poll.


def _lpoll(n):
    """Bounded Lamport poll of n 16-byte slots a lane, system scope, in the lanes
    where act is not 0. It loads until no 16-bit word of the slots is 0xFFFF, the
    error word is set, or the time limit passes. Outputs: failed (0 ok, 1 timed out,
    2 error record seen), the 4 n dwords, scratch. Inputs: act, n slot addresses,
    the error-word address, the time limit in 100 MHz ticks."""
    nd = 4 * n
    b = _POLL_VGPR
    saved, now, dl, errv, acc, tmp, pend = range(nd + 1, nd + 8)
    act = nd + 8
    a0 = act + 1
    aerr = a0 + n
    ticks = aerr + 1
    lines = [
        "s_mov_b32 $0, 0",
        f"v_cmp_ne_u32 vcc, 0, ${act}",
        f"s_and_saveexec_b64 ${saved}, vcc",
        "s_cbranch_execz .Llp_done${:uid}",
        f"s_memrealtime ${now}",
        "s_waitcnt lgkmcnt(0)",
        f"v_lshl_add_u64 ${dl}, ${now}, 0, ${ticks}",
        ".Llp_loop${:uid}:",
    ]
    lines += [
        f"global_load_dwordx4 v[{b + 4 * i}:{b + 4 * i + 3}], ${a0 + i}, off sc0 sc1"
        for i in range(n)
    ]
    lines += [
        f"global_load_dword ${errv}, ${aerr}, off sc0 sc1",
        "s_waitcnt vmcnt(0)",
        f"v_pk_max_u16 ${acc}, v{b}, v{b + 1}",
    ]
    lines += [f"v_pk_max_u16 ${acc}, ${acc}, v{b + i}" for i in range(2, nd)]
    lines += [
        f"v_and_b32 ${tmp}, 0xffff, ${acc}",
        f"v_cmp_eq_u32 vcc, 0xffff, ${tmp}",
        f"s_mov_b64 ${pend}, vcc",
        f"v_cmp_lt_u32 vcc, 0xfffeffff, ${acc}",
        f"s_or_b64 ${pend}, ${pend}, vcc",
        f"s_cmp_eq_u64 ${pend}, 0",
        "s_cbranch_scc1 .Llp_ok${:uid}",
        f"v_cmp_ne_u32 vcc, 0, ${errv}",
        "s_cmp_lg_u64 vcc, 0",
        "s_cbranch_scc1 .Llp_give${:uid}",
        f"s_memrealtime ${now}",
        "s_waitcnt lgkmcnt(0)",
        f"v_cmp_ge_u64 vcc, ${now}, ${dl}",
        "s_cmp_lg_u64 vcc, 0",
        "s_cbranch_scc1 .Llp_out${:uid}",
        "s_sleep 1",
        "s_branch .Llp_loop${:uid}",
        ".Llp_give${:uid}:",
        "s_mov_b32 $0, 2",
        "s_branch .Llp_ok${:uid}",
        ".Llp_out${:uid}:",
        "s_mov_b32 $0, 1",
        ".Llp_ok${:uid}:",
    ]
    lines += [f"v_mov_b32 ${1 + i}, v{b + i}" for i in range(nd)]
    lines += [".Llp_done${:uid}:", f"s_mov_b64 exec, ${saved}"]
    cons = (
        ["=&s"]
        + ["=&v"] * nd
        + ["=&s", "=&s", "=&v", "=&v", "=&v", "=&v", "=&s"]
        + ["v"] * (n + 3)
        + ["~{memory}", "~{vcc}", "~{scc}"]
        + [f"~{{v{b + i}}}" for i in range(nd)]
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


_LPOLL = {n: _lpoll(n) for n in range(1, 9)}
_ERREC = {n: _errec(n) for n in (2, 4, 8)}
BFLY_ADD: gl.constexpr = gl.constexpr(_bfly("v_add_f32"))


@gluon.constexpr_function
def _lpoll_asm(n):
    return _LPOLL[int(n)][0]


@gluon.constexpr_function
def _lpoll_cons(n):
    return _LPOLL[int(n)][1]


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
    """Lamport poll of the NS 16-byte slots in addrs. Returns failed and the 4 dwords of each
    slot. Up to _POLL_MAX slots go in one wait."""
    z = gl.zeros_like(act.to(gl.int32))
    failed = z
    words = ()
    for c in gl.static_range(0, NS, _POLL_MAX):
        part = ()
        for i in gl.static_range(_chunk(NS, c)):
            part = part + (addrs[c + i],)
        out = gl.inline_asm_elementwise(
            _lpoll_asm(_chunk(NS, c)),
            _lpoll_cons(_chunk(NS, c)),
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
