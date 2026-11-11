# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""All-reduce + residual add + RMSNorm (+ MXFP8) + MoE router + top-k + moe_sorting in one
persistent kernel over 2, 4 or 8 GPUs (decode steps of tensor-parallel MoE layers).

One CTA a CU; every CTA is resident (the waits spin). The CTAs take roles by their index and the
row count:

- Reduce. One-shot (PROTO 0): every rank pushes its partial rows to every peer; CTA t < rows
  reduces row t (all rows on every rank). Two-shot (PROTO 1): rank g owns the rows g G to
  g G + G - 1 (G = ceil(rows / WORLD)); the other ranks push those rows to it, CTA b < G reduces
  row g G + b and pushes the sum to the peers. Every reduce adds the ranks in rank order in fp32
  and rounds once to bf16, as the AITER custom all-reduce.
- Norm. The CTA of a row adds the residual, normalizes (Gemma 1 + w, or w), stores the bf16 row
  with write-through stores and publishes a tagged ready word, then the new residual and the
  MXFP8 copy. Two-shot: the CTAs after the owners norm the other ranks' rows as their sums land.
- Router. Tasks of (16 experts) x (K part) on the last NRC CTAs (TPC tasks a CTA). The weights
  of the first task are loaded at kernel entry; a task waits for its rows, runs a batched bf16
  MFMA (a warp a K slice; fp32 weights as three exact bf16 parts), adds the warps in LDS and
  publishes tagged partial logits.
- Route. A CTA a routed row (one-shot: the row's CTA; two-shot: the router CTAs): the partial
  logits in part order (every rank gets the same bits), the scores (sigmoid + bias, or softmax),
  an optional group limit, the top-k by repeated warp argmax (a lane holds experts lane + 64 j),
  the weights (renormalized, times the routed scale). One-shot: the picks stay on this rank; two-shot: they go to every rank.
- Sort. The last CTA: every row's picks -> top-k ids / weights and the moe_sorting outputs, the
  tokens of each expert in ascending order (LDS bitmaps, popcount ranks).
- The router CTAs zero moe_buf.

Bulk data between GPUs goes through Lamport inboxes: 0xFFFF marks an empty 16-bit word, a sender
writes the NaN 0x7FFF in its place, the reader writes 0xFFFF back; two halves by the parity of
the launch colour. Small hand-offs are tagged 8-byte words ((colour + 1) << 32 | value). Each CTA
keeps its own launch colour. Every wait has a time limit; a wait that times out writes an error
record [1, rank, colour, site] into the buffer of every rank, and every waiter that sees it gives
up."""

from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.language.core import PropagateNan as _PropagateNan

from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr

_NAN_ALL: gl.constexpr = gl.constexpr(_PropagateNan.ALL)
_SITE_REDUCE: gl.constexpr = gl.constexpr(1)
_SITE_GATHER: gl.constexpr = gl.constexpr(2)
_SITE_ROUTER: gl.constexpr = gl.constexpr(3)
_SITE_ROUTE: gl.constexpr = gl.constexpr(4)
_SITE_SORT: gl.constexpr = gl.constexpr(5)
_F32_TINY: gl.constexpr = gl.constexpr(1.1754943508222875e-38)
_SENT_SIGN: gl.constexpr = gl.constexpr(-0x7FFF8000)  # 0x80008000
_NEG_LOG2E: gl.constexpr = gl.constexpr(-1.4426950408889634)
_LOG2E: gl.constexpr = gl.constexpr(1.4426950408889634)
_BIG: gl.constexpr = gl.constexpr(1 << 30)
_POLL_VGPR = 64  # The first of the fixed VGPRs that a poll loads into.
_POLL_MAX: gl.constexpr = gl.constexpr(8)  # The most slots in one poll.


def _lpoll(n):
    """Bounded Lamport poll of n 16-byte slots a lane, system scope, in the lanes where act is
    not 0. Loads until no 16-bit word of the slots is 0xFFFF, the error word is set, or the time
    limit passes. Outputs: failed (0 ok, 1 timed out, 2 error record seen), the 4 n dwords,
    scratch. Inputs: act, n slot addresses, the error-word address, the time limit in 100 MHz
    ticks."""
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


def _tpoll(n, system):
    """Bounded poll of n tagged 8-byte words a lane ((tag << 32) | value), in the lanes where act
    is not 0: loads (system scope for peer memory, device scope for scratch) until the high word
    of each is tag, the error word is set, or the time limit passes. Outputs: failed, the n low
    words, scratch. Inputs: act, tag, n addresses, the error-word address, the time limit.
    """
    b = _POLL_VGPR
    scope = "sc0 sc1" if system else "sc1"
    saved, now, dl, errv, pend = range(n + 1, n + 6)
    act = n + 6
    tag = act + 1
    a0 = tag + 1
    aerr = a0 + n
    ticks = aerr + 1
    lines = [
        "s_mov_b32 $0, 0",
        f"v_cmp_ne_u32 vcc, 0, ${act}",
        f"s_and_saveexec_b64 ${saved}, vcc",
        "s_cbranch_execz .Ltp_done${:uid}",
        f"s_memrealtime ${now}",
        "s_waitcnt lgkmcnt(0)",
        f"v_lshl_add_u64 ${dl}, ${now}, 0, ${ticks}",
        ".Ltp_loop${:uid}:",
    ]
    lines += [
        f"global_load_dwordx2 v[{b + 2 * i}:{b + 2 * i + 1}], ${a0 + i}, off {scope}"
        for i in range(n)
    ]
    lines += [
        f"global_load_dword ${errv}, ${aerr}, off sc0 sc1",
        "s_waitcnt vmcnt(0)",
        f"v_cmp_ne_u32 vcc, ${tag}, v{b + 1}",
        f"s_mov_b64 ${pend}, vcc",
    ]
    for i in range(1, n):
        lines += [
            f"v_cmp_ne_u32 vcc, ${tag}, v{b + 2 * i + 1}",
            f"s_or_b64 ${pend}, ${pend}, vcc",
        ]
    lines += [
        f"s_cmp_eq_u64 ${pend}, 0",
        "s_cbranch_scc1 .Ltp_ok${:uid}",
        f"v_cmp_ne_u32 vcc, 0, ${errv}",
        "s_cmp_lg_u64 vcc, 0",
        "s_cbranch_scc1 .Ltp_give${:uid}",
        f"s_memrealtime ${now}",
        "s_waitcnt lgkmcnt(0)",
        f"v_cmp_ge_u64 vcc, ${now}, ${dl}",
        "s_cmp_lg_u64 vcc, 0",
        "s_cbranch_scc1 .Ltp_out${:uid}",
        "s_sleep 1",
        "s_branch .Ltp_loop${:uid}",
        ".Ltp_give${:uid}:",
        "s_mov_b32 $0, 2",
        "s_branch .Ltp_ok${:uid}",
        ".Ltp_out${:uid}:",
        "s_mov_b32 $0, 1",
        ".Ltp_ok${:uid}:",
    ]
    lines += [f"v_mov_b32 ${1 + i}, v{b + 2 * i}" for i in range(n)]
    lines += [".Ltp_done${:uid}:", f"s_mov_b64 exec, ${saved}"]
    cons = (
        ["=&s"]
        + ["=&v"] * n
        + ["=&s", "=&s", "=&v", "=&v", "=&s"]
        + ["v"] * (n + 4)
        + ["~{memory}", "~{vcc}", "~{scc}"]
        + [f"~{{v{b + i}}}" for i in range(2 * n)]
    )
    return "\n".join(lines), ",".join(cons)


def _errec(n):
    """[1, rank, colour, site] into the error records of n buffers, system scope, from the active
    lanes. The high pair goes first: waiters read the low word."""
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
_TPOLL = {(n, s): _tpoll(n, s) for n in range(1, 9) for s in (False, True)}
_ERREC = {n: _errec(n) for n in (2, 4, 8)}
_BFLY_ADD: gl.constexpr = gl.constexpr(_bfly("v_add_f32"))
_BFLY_MAX: gl.constexpr = gl.constexpr(_bfly("v_max_f32"))
_BFLY_MINI: gl.constexpr = gl.constexpr(_bfly("v_min_i32"))


@gluon.constexpr_function
def _next_pow2(n):
    p = 1
    while p < int(n):
        p *= 2
    return p


@gluon.constexpr_function
def _cmax(a, b):
    return max(int(a), int(b))


@gluon.constexpr_function
def _lpoll_asm(n):
    return _LPOLL[int(n)][0]


@gluon.constexpr_function
def _lpoll_cons(n):
    return _LPOLL[int(n)][1]


@gluon.constexpr_function
def _tpoll_asm(n, system):
    return _TPOLL[(int(n), bool(system))][0]


@gluon.constexpr_function
def _tpoll_cons(n, system):
    return _TPOLL[(int(n), bool(system))][1]


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
def _tids(W: gl.constexpr):
    """The thread index, one value a thread."""
    LV: gl.constexpr = gl.BlockedLayout([1, 4], [64, 1], [W, 1], [1, 0])
    return gl.arange(0, 64 * W, layout=gl.SliceLayout(1, LV))


@gluon.jit
def _lanes(W: gl.constexpr):
    """The lane index, one value a lane, the same in every warp."""
    LR: gl.constexpr = gl.BlockedLayout([1], [64], [W], [0])
    return gl.arange(0, 64, layout=LR)


@gluon.jit
def _k4(W: gl.constexpr):
    LV: gl.constexpr = gl.BlockedLayout([1, 4], [64, 1], [W, 1], [1, 0])
    return gl.arange(0, 4, layout=gl.SliceLayout(0, LV))[None, :]


@gluon.jit
def _split4(x, W: gl.constexpr):
    """[threads, 4] dwords to 4 tensors of one dword a thread."""
    k = _k4(W)
    return (
        gl.sum(gl.where(k == 0, x, 0), axis=1),
        gl.sum(gl.where(k == 1, x, 0), axis=1),
        gl.sum(gl.where(k == 2, x, 0), axis=1),
        gl.sum(gl.where(k == 3, x, 0), axis=1),
    )


@gluon.jit
def _join4(w, W: gl.constexpr):
    k = _k4(W)
    return gl.where(
        k == 0,
        w[0][:, None],
        gl.where(k == 1, w[1][:, None], gl.where(k == 2, w[2][:, None], w[3][:, None])),
    )


@gluon.jit
def _ld4(ptr, dw, mask, W: gl.constexpr):
    """The 16 bytes at dword dw of ptr (one vector a thread), as 4 dwords."""
    k = _k4(W)
    x = gl.amd.cdna4.buffer_load(ptr, dw[:, None] + k, mask=mask[:, None] & (k < 4))
    return _split4(x, W)


@gluon.jit
def _st4(ptr, dw, w, mask, W: gl.constexpr, CACHE: gl.constexpr = ""):
    k = _k4(W)
    gl.amd.cdna4.buffer_store(
        _join4(w, W), ptr, dw[:, None] + k, mask=mask[:, None] & (k < 4), cache=CACHE
    )


@gluon.jit
def _st2(ptr, dw, w0, w1, mask, W: gl.constexpr):
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
def _canon(w):
    """A 16-bit word 0xFFFF becomes 0x7FFF: still a NaN, but not the empty mark."""
    x = w ^ -1
    nz = ((x & 0x7FFF7FFF) + 0x7FFF7FFF) | x
    return w ^ ((nz & _SENT_SIGN) ^ _SENT_SIGN)


@gluon.jit
def _bf_lo(w):
    return (w << 16).to(gl.float32, bitcast=True)


@gluon.jit
def _bf_hi(w):
    return (w & -65536).to(gl.float32, bitcast=True)


@gluon.jit
def _pack_bf16(lo, hi):
    """Two fp32 values to one dword of 2 bf16 (round to nearest even), lo first."""
    return gl.inline_asm_elementwise(
        "v_cvt_pk_bf16_f32 $0, $1, $2", "=v,v,v", [lo, hi], gl.int32, True, 1
    )


@gluon.jit
def _fp8_pack4(a, b, c, d):
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
def _bfly_op(x, ASM: gl.constexpr):
    out = gl.inline_asm_elementwise(ASM, "=&v,=&v,v", [x], (x.dtype, x.dtype), True, 1)
    return out[0]


@gluon.jit
def _quad_xor(x, P: gl.constexpr):
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
def _exp2(x):
    return gl.inline_asm_elementwise(
        "v_exp_f32 $0, $1", "=v,v", [x], gl.float32, True, 1
    )


@gluon.jit
def _rcp(x):
    return gl.inline_asm_elementwise(
        "v_rcp_f32 $0, $1", "=v,v", [x], gl.float32, True, 1
    )


@gluon.jit
def _popc(x):
    return gl.inline_asm_elementwise(
        "v_bcnt_u32_b32 $0, $1, 0", "=v,v", [x], gl.int32, True, 1
    )


@gluon.jit
def _drain(x):
    """Waits until every vector memory operation of the wave has completed (x passes through)."""
    return gl.inline_asm_elementwise(
        "s_waitcnt vmcnt(0)", "=v,0,~{memory}", [x], x.dtype, False, 1
    )


@gluon.jit
def _fail_record(bufs, rank, colour, site: gl.constexpr, active):
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
def _peer(bufs, d, WORLD: gl.constexpr):
    """bufs[d] for a uniform d in 1 to WORLD - 1."""
    p = bufs[1]
    for q in gl.static_range(2, WORLD):
        p = gl.where(d == q, bufs[q], p)
    return p


@gluon.jit
def _poll_slots(own, addrs, act, ticks, NS: gl.constexpr):
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


@gluon.jit
def _poll_tags(own, addrs, act, tag, ticks, NS: gl.constexpr, SYSTEM: gl.constexpr):
    """Poll of the NS tagged words at addrs (int64 pointers) for tag. Returns failed and the low
    word of each."""
    z = gl.zeros_like(act.to(gl.int32))
    failed = z
    vals = ()
    for c in gl.static_range(0, NS, _POLL_MAX):
        part = ()
        for i in gl.static_range(_chunk(NS, c)):
            part = part + (addrs[c + i],)
        out = gl.inline_asm_elementwise(
            _tpoll_asm(_chunk(NS, c), SYSTEM),
            _tpoll_cons(_chunk(NS, c), SYSTEM),
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
def _pair(tag, v):
    """The tagged word of the 32-bit value v."""
    return (tag.to(gl.int64) << 32) | v.to(gl.uint32, bitcast=True).to(gl.int64)


@gluon.jit
def _isnan(x):
    return x != x  # noqa: PLR0124 - true only for NaN


@gluon.jit
def _sigmoid(x):
    return _rcp(_exp2(x * _NEG_LOG2E) + 1.0)


@gluon.jit
def _top2(a1, a2, b1, b2):
    return gl.maximum(a1, b1), gl.maximum(gl.minimum(a1, b1), gl.maximum(a2, b2))


@gluon.jit
def _add_i32(a, b):
    return a + b


@gluon.jit
def _row_sum(own_w, got, rank, WORLD: gl.constexpr, V: gl.constexpr, tid):
    """The bf16 words of the sum over the ranks in rank order (fp32, one rounding): own_w are this
    rank's words, got[(d - 1) V + j] the words of rank + d."""
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
                    acc_lo = _bf_lo(x)
                    acc_hi = _bf_hi(x)
                else:
                    acc_lo = acc_lo + _bf_lo(x)
                    acc_hi = acc_hi + _bf_hi(x)
            hw = hw + (_pack_bf16(acc_lo, acc_hi),)
    return hw


@gluon.jit
def _norm_row(
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
    """s = h + residual (fp32) and the bf16 words of y = s * rstd * w'. The sum of squares adds a
    thread's 8 V values in column order (no fma), a lane butterfly, then the waves in order.
    """
    tid = _tids(W)
    lane = tid % 64
    wave = tid // 64
    NWA: gl.constexpr = (HIDDEN // 8 // V + 63) // 64
    s = ()
    for j in gl.static_range(V):
        for e in gl.static_range(4):
            if HAS_RES:
                s = s + (
                    _bf_lo(hw[4 * j + e]) + _bf_lo(res_w[j][e]),
                    _bf_hi(hw[4 * j + e]) + _bf_hi(res_w[j][e]),
                )
            else:
                s = s + (_bf_lo(hw[4 * j + e]), _bf_hi(hw[4 * j + e]))
    sq = s[0] * s[0]
    for i in gl.static_range(1, 8 * V):
        sq = sq + s[i] * s[i]
    sq = _bfly_op(gl.where(act, sq, 0.0), _BFLY_ADD)
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
            wl = _bf_lo(w_w[j][e])
            wh = _bf_hi(w_w[j][e])
            if GEMMA:
                wl = 1.0 + wl
                wh = 1.0 + wh
            y = y + (
                _pack_bf16(
                    (s[8 * j + 2 * e] * rstd) * wl, (s[8 * j + 2 * e + 1] * rstd) * wh
                ),
            )
    return s, y


@gluon.jit
def _store_rest(
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
    """Row t: the new residual bf16(s), and the MXFP8 copy of y (a quad of lanes holds a
    32-column group)."""
    tid = _tids(W)
    lane = tid % 64
    if QUANT:
        for j in gl.static_range(V):
            vec = j * A + tid
            yv = ()
            for e in gl.static_range(4):
                yv = yv + (_bf_lo(y[4 * j + e]), _bf_hi(y[4 * j + e]))
            am = gl.abs(yv[0])
            for e in gl.static_range(1, 8):
                am = gl.maximum(am, gl.abs(yv[e]), propagate_nan=_NAN_ALL)
            am = gl.maximum(am, _quad_xor(am, 1), propagate_nan=_NAN_ALL)
            am = gl.maximum(am, _quad_xor(am, 2), propagate_nan=_NAN_ALL)
            a = gl.maximum(am, _F32_TINY, propagate_nan=_NAN_ALL)
            bits = gl.div_rn(a, gl.full(a.shape, 448.0, gl.float32, a.type.layout)).to(
                gl.int32, bitcast=True
            )
            ex = (bits >> 23) & 0xFF
            sb = ex + gl.where((bits & 0x7FFFFF) != 0, 1, 0)
            sb = gl.where(ex == 0, 0, gl.minimum(gl.maximum(sb, 0), 254))
            inv = ((254 - sb) << 23).to(gl.float32, bitcast=True)
            c0 = _fp8_pack4(yv[0] * inv, yv[1] * inv, yv[2] * inv, yv[3] * inv)
            c1 = _fp8_pack4(yv[4] * inv, yv[5] * inv, yv[6] * inv, yv[7] * inv)
            _st2(xq, t * (HIDDEN // 4) + 2 * vec, c0, c1, act, W)
            gl.store(
                xs + t * (HIDDEN // 32) + vec // 4,
                sb.to(gl.uint8),
                mask=act & (lane % 4 == 0),
            )
    for j in gl.static_range(V):
        vec = j * A + tid
        rw = ()
        for e in gl.static_range(4):
            rw = rw + (_pack_bf16(s[8 * j + 2 * e], s[8 * j + 2 * e + 1]),)
        _st4(res_out, t * (HIDDEN // 2) + vec * 4, rw, act, W)


@gluon.jit
def _sel_val(
    lg,
    sel,
    bias_l,
    j: gl.constexpr,
    lane,
    E: gl.constexpr,
    SCORE: gl.constexpr,
    HAS_BIAS: gl.constexpr,
    REG: gl.constexpr,
):
    """Selection value of expert lane + 64 j: from the logits (registers path) or LDS."""
    neg = gl.full(lane.shape, float("-inf"), gl.float32, lane.type.layout)
    e = lane + 64 * j
    if REG:
        v = lg.gather(gl.where(e < E, e, 0), 0)
        if SCORE == 0:
            v = _sigmoid(v)
        if HAS_BIAS:
            v = v + bias_l[j]
        v = gl.where(_isnan(v) | (e >= E), neg, v)
    else:
        v = sel.gather(gl.where(e < E, e, 0), 0)
    return v


@gluon.jit
def _select(
    lg,
    sel,
    bias_l,
    bias_s,
    lane,
    routed_scale,
    E: gl.constexpr,
    EJ: gl.constexpr,
    TOPK: gl.constexpr,
    SCORE: gl.constexpr,
    HAS_BIAS: gl.constexpr,
    NGROUP: gl.constexpr,
    TOPKG: gl.constexpr,
    RENORM: gl.constexpr,
):
    """The top-k of one row whose logits are in lg (LDS), by every warp: the expert and weight of
    slot k end in lane k. Ties go to the lower expert; NaN values rank last. A lane holds the
    experts lane + 64 j: their selection values stay in registers up to 256 experts, else in
    LDS (sel)."""
    REG: gl.constexpr = EJ <= 4
    neg = gl.full(lane.shape, float("-inf"), gl.float32, lane.type.layout)
    zero_i = gl.zeros_like(lane)
    vals = ()
    if REG:
        for j in gl.static_range(EJ):
            vals = vals + (
                _sel_val(lg, sel, bias_l, j, lane, E, SCORE, HAS_BIAS, True),
            )
    else:
        for j in gl.static_range(EJ):
            e = lane + 64 * j
            ec = gl.where(e < E, e, 0)
            v = lg.gather(ec, 0)
            if SCORE == 0:
                v = _sigmoid(v)
            if HAS_BIAS:
                v = v + bias_s.gather(ec, 0)
            v = gl.where(_isnan(v) | (e >= E), neg, v)
            sel.scatter(v, gl.where(e < E, e, E), 0)
        gl.barrier()
    # Group limit: a group's score is the sum of its 2 best values (with a bias) or its best
    # value; the TOPKG best groups stay (ties to the lower group).
    gmask = zero_i - 1
    if NGROUP > 1:
        GS: gl.constexpr = E // NGROUP
        gsc = ()
        for g in gl.static_range(NGROUP):
            m1 = neg
            m2 = neg
            for j in gl.static_range(EJ):
                e = lane + 64 * j
                if REG:
                    v = vals[j]
                else:
                    v = _sel_val(lg, sel, bias_l, j, lane, E, SCORE, HAS_BIAS, False)
                v = gl.where((e // GS == g) & (e < E), v, neg)
                m2 = gl.maximum(m2, gl.minimum(m1, v))
                m1 = gl.maximum(m1, v)
            b1 = gl.max(m1, axis=0)
            first = gl.min(gl.where(m1 == b1, lane, 64), axis=0)
            b2 = gl.max(gl.where(lane == first, m2, m1), axis=0)
            if HAS_BIAS:
                gsc = gsc + (b1 + b2,)
            else:
                gsc = gsc + (b1,)
        chosen = zero_i
        for it in gl.static_range(TOPKG):
            best = neg
            bi = zero_i - 1
            for g in gl.static_range(NGROUP):
                v = gl.where(((chosen >> g) & 1) != 0, neg, gl.zeros_like(neg) + gsc[g])
                take = v > best
                best = gl.where(take, v, best)
                bi = gl.where(take, g, bi)
            chosen = chosen | gl.where(bi >= 0, 1 << gl.maximum(bi, 0), 0)
        gmask = chosen
    # Top-k: a lane takes its best allowed expert, the wave the best value, then the lowest
    # expert of that value.
    taken = zero_i
    my_id = zero_i
    for k in gl.static_range(TOPK):
        bv = neg
        bi = zero_i + _BIG
        for j in gl.static_range(EJ):
            e = lane + 64 * j
            if REG:
                v = vals[j]
            else:
                v = _sel_val(lg, sel, bias_l, j, lane, E, SCORE, HAS_BIAS, False)
            allow = (e < E) & (((taken >> j) & 1) == 0)
            if NGROUP > 1:
                allow = allow & (
                    ((gmask >> (gl.where(e < E, e, 0) // (E // NGROUP))) & 1) != 0
                )
            v = gl.where(allow, v, neg)
            take = (v > bv) | ((bi == _BIG) & allow)
            bv = gl.where(take, v, bv)
            bi = gl.where(take, e, bi)
        best = gl.max(bv, axis=0)
        win = gl.min(gl.where((bv == best) & (bi < _BIG), bi, _BIG), axis=0)
        mine = bi == win
        taken = taken | gl.where(mine, 1 << (gl.where(mine, bi, 0) // 64), 0)
        my_id = gl.where(lane == k, gl.where(win < _BIG, win, 0), my_id)
    # Weights: lane k < TOPK reads the logit of its expert once.
    slot = lane < TOPK
    lgk = lg.gather(my_id, 0)
    if SCORE == 0:
        my_w = gl.where(slot, _sigmoid(lgk), 0.0)
        my_w = gl.where(_isnan(my_w), 0.0, my_w)
        if RENORM:
            total = _bfly_op(my_w, _BFLY_ADD)
            w = my_w * gl.div_rn(routed_scale + total * 0.0, gl.maximum(total, 1e-20))
        else:
            w = my_w * routed_scale
    else:
        if RENORM:
            top = gl.max(gl.where(lane == 0, lgk, neg), axis=0)
            ex = gl.where(slot, _exp2((lgk - top) * _LOG2E), 0.0)
            z = _bfly_op(ex, _BFLY_ADD)
        else:
            mx = neg
            for j in gl.static_range(EJ):
                e = lane + 64 * j
                mx = gl.maximum(
                    mx, gl.where(e < E, lg.gather(gl.where(e < E, e, 0), 0), neg)
                )
            top = gl.max(mx, axis=0)
            ex = _exp2((lgk - top) * _LOG2E)
            zz = gl.zeros_like(neg)
            for j in gl.static_range(EJ):
                e = lane + 64 * j
                lgv = lg.gather(gl.where(e < E, e, 0), 0)
                zz = zz + gl.where(e < E, _exp2((lgv - top) * _LOG2E), 0.0)
            z = _bfly_op(zz, _BFLY_ADD)
        w = gl.div_rn(ex, z) * routed_scale
    return my_id, w


@gluon.jit
def _route_row(
    scratch,
    own,
    bufs,
    peers,
    bias,
    rj,
    t_r,
    par,
    tag,
    ticks,
    rank,
    colour,
    routed_scale,
    E: gl.constexpr,
    TOPK: gl.constexpr,
    SCORE: gl.constexpr,
    HAS_BIAS: gl.constexpr,
    NGROUP: gl.constexpr,
    TOPKG: gl.constexpr,
    RENORM: gl.constexpr,
    KP: gl.constexpr,
    RMAXL: gl.constexpr,
    SCL: gl.constexpr,
    SCP: gl.constexpr,
    PK0: gl.constexpr,
    SMAX: gl.constexpr,
    PROTO: gl.constexpr,
    WORLD: gl.constexpr,
    W: gl.constexpr,
):
    """Routed row rj (row t_r): its KP x E partial logits, summed in part order, -> the top-k
    picks as tagged (expert, weight) words: one-shot into scratch, two-shot into every rank's
    buffer."""
    T: gl.constexpr = 64 * W
    EP: gl.constexpr = _next_pow2(E + 1)
    EJ: gl.constexpr = (E + 63) // 64
    KE: gl.constexpr = KP * E
    NRR: gl.constexpr = (KE + T - 1) // T
    KEP: gl.constexpr = _next_pow2(NRR * T)
    NEB: gl.constexpr = (E + T - 1) // T
    NPK: gl.constexpr = 2 * TOPK
    tid = _tids(W)
    lane = tid % 64
    LB: gl.constexpr = gl.BlockedLayout([1], [64], [W], [0])
    eb = gl.arange(0, T, layout=LB)
    lane_r = _lanes(W)
    bias_r = ()
    bias_l = ()
    if HAS_BIAS:
        if EJ <= 4:
            for j in gl.static_range(EJ):
                bias_l = bias_l + (gl.load(bias + gl.minimum(lane_r + 64 * j, E - 1)),)
        else:
            for c in gl.static_range(NEB):
                bias_r = bias_r + (gl.load(bias + gl.minimum(c * T + eb, E - 1)),)
    r_addrs = ()
    for c in gl.static_range(NRR):
        f = gl.minimum(c * T + tid, KE - 1)
        r_addrs = r_addrs + (scratch + SCL + ((f // E) * RMAXL + rj) * E + f % E,)
    r_failed, r_vals = _poll_tags(own, r_addrs, tid >= 0, tag, ticks, NRR, False)
    _fail_record(bufs, rank, colour, _SITE_ROUTE, (lane == 0) & (r_failed == 1))
    SL1: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [0])
    plg = gl.allocate_shared_memory(gl.float32, [KEP], SL1)
    lg = gl.allocate_shared_memory(gl.float32, [EP], SL1)
    sel = gl.allocate_shared_memory(gl.float32, [EP], SL1)
    bias_s = gl.allocate_shared_memory(gl.float32, [EP], SL1)
    for c in gl.static_range(NRR):
        f = c * T + tid
        plg.scatter(r_vals[c].to(gl.float32, bitcast=True), f, 0)
    if HAS_BIAS and EJ > 4:
        for c in gl.static_range(NEB):
            bias_s.scatter(bias_r[c], gl.where(c * T + eb < E, c * T + eb, E), 0)
    gl.barrier()
    for c in gl.static_range(NEB):
        e = c * T + eb
        ec = gl.minimum(e, E - 1)
        tot = plg.gather(ec, 0)
        for p in gl.static_range(1, KP):
            tot = tot + plg.gather(p * E + ec, 0)
        lg.scatter(tot, gl.where(e < E, e, E), 0)
    gl.barrier()
    my_id, my_w = _select(
        lg,
        sel,
        bias_l,
        bias_s,
        lane_r,
        routed_scale,
        E,
        EJ,
        TOPK,
        SCORE,
        HAS_BIAS,
        NGROUP,
        TOPKG,
        RENORM,
    )
    pk_ok = lane_r < TOPK
    id_pair = _pair(tag, my_id)
    w_pair = _pair(tag, my_w.to(gl.int32, bitcast=True))
    if PROTO == 0:
        gl.amd.cdna4.buffer_store(
            id_pair, scratch, SCP + t_r * NPK + lane_r, mask=pk_ok, cache=".wt"
        )
        gl.amd.cdna4.buffer_store(
            w_pair, scratch, SCP + t_r * NPK + TOPK + lane_r, mask=pk_ok, cache=".wt"
        )
    else:
        for d in gl.static_range(WORLD):
            b64 = gl.multiple_of(peers[d].to(gl.pointer_type(gl.int64)), 16)
            base = PK0 // 8 + (par * SMAX + t_r) * NPK
            gl.amd.cdna4.buffer_store(id_pair, b64, base + lane_r, mask=pk_ok)
            gl.amd.cdna4.buffer_store(w_pair, b64, base + TOPK + lane_r, mask=pk_ok)


@gluon.jit
def _router_w(
    router_w,
    task,
    live,
    aw,
    am,
    ak,
    HIDDEN: gl.constexpr,
    KP: gl.constexpr,
    SPT: gl.constexpr,
    SPW: gl.constexpr,
    NKS: gl.constexpr,
):
    """The weights of router task (16 experts x K part) in the A operand layout of the batched
    MFMA: warp w holds the k-steps w SPW .. w SPW + SPW - 1 of the part."""
    tile = task // KP
    part = task % KP
    wa = ()
    for i in gl.static_range(SPW):
        st = aw * SPW + i
        ks = part * SPT + st
        ok = (st < SPT) & (ks < NKS) & live & (am >= 0) & (ak >= 0)
        off = (tile * 16 + am) * HIDDEN + gl.where(ok, ks, 0) * 16 + ak
        wa = wa + (gl.amd.cdna4.buffer_load(router_w, off, mask=ok),)
    return wa


@gluon.jit
def _router_task(
    xn,
    scratch,
    wa,
    task,
    row0,
    nr,
    tag,
    HIDDEN: gl.constexpr,
    E: gl.constexpr,
    KP: gl.constexpr,
    SPT: gl.constexpr,
    SPW: gl.constexpr,
    NKS: gl.constexpr,
    RTMAX: gl.constexpr,
    RMAXL: gl.constexpr,
    SCL: gl.constexpr,
    WFP32: gl.constexpr,
    W: gl.constexpr,
):
    """Partial logits of the nr routed rows (row0 ..) for one task, as tagged words in scratch
    [part][row][expert]. The rows' loads all go out before the first MFMA."""
    mf: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 16], transposed=False, warps_per_cta=[W, 1, 1]
    )
    lb: gl.constexpr = gl.DotOperandLayout(1, mf, k_width=4)
    tile = task // KP
    part = task % KP
    bw = gl.arange(0, W, layout=gl.SliceLayout(1, gl.SliceLayout(2, lb)))[:, None, None]
    bk = gl.arange(0, 16, layout=gl.SliceLayout(0, gl.SliceLayout(2, lb)))[
        None, :, None
    ]
    bn = gl.arange(0, 16, layout=gl.SliceLayout(0, gl.SliceLayout(1, lb)))[
        None, None, :
    ]
    xbf = xn.to(gl.pointer_type(gl.bfloat16))
    xv = ()
    for rt in gl.static_range(RTMAX):
        r_n = gl.minimum(rt * 16 + bn, gl.maximum(nr - 1, 0))
        for i in gl.static_range(SPW):
            st = bw * SPW + i
            ks = part * SPT + st
            okb = (st < SPT) & (ks < NKS) & (rt * 16 < nr) & (bk >= 0) & (bn >= 0)
            offb = (row0 + r_n) * HIDDEN + gl.where(okb, ks, 0) * 16 + bk
            xv = xv + (gl.amd.cdna4.buffer_load(xbf, offb, mask=okb, cache=".cv"),)
    # fp32 weights as three exact bf16 parts (w = hi + mid + lo: exact products)
    wparts = ()
    for i in gl.static_range(SPW):
        if WFP32:
            a32 = wa[i]
            hi = a32.to(gl.int32, bitcast=True) & -65536
            r1 = a32 - hi.to(gl.float32, bitcast=True)
            mid = r1.to(gl.int32, bitcast=True) & -65536
            r2 = r1 - mid.to(gl.float32, bitcast=True)
            lo = r2.to(gl.int32, bitcast=True) & -65536
            wparts = wparts + (
                (
                    hi.to(gl.float32, bitcast=True).to(gl.bfloat16),
                    mid.to(gl.float32, bitcast=True).to(gl.bfloat16),
                    lo.to(gl.float32, bitcast=True).to(gl.bfloat16),
                ),
            )
        else:
            wparts = wparts + ((wa[i],),)
    # Each warp's partial sums go to LDS; a thread then adds the W warps of one (expert, row)
    # in warp order.
    reds = ()
    for rt in gl.static_range(RTMAX):
        reds = reds + (
            gl.allocate_shared_memory(
                gl.float32, [W, 16, 16], gl.SwizzledSharedLayout(1, 1, 1, [2, 1, 0])
            ),
        )
    for rt in gl.static_range(RTMAX):
        if rt * 16 < nr:
            acc = gl.zeros([W, 16, 16], gl.float32, layout=mf)
            for i in gl.static_range(SPW):
                for q in gl.static_range(len(wparts[i])):
                    acc = gl.amd.cdna4.mfma(wparts[i][q], xv[rt * SPW + i], acc)
            reds[rt].store(acc)
    gl.barrier()
    RL: gl.constexpr = gl.BlockedLayout([W, 1, 1], [1, 16, 4], [1, 1, W], [2, 1, 0])
    PL: gl.constexpr = gl.SliceLayout(0, RL)
    pm = gl.arange(0, 16, layout=gl.SliceLayout(1, PL))[:, None]
    pn = gl.arange(0, 16, layout=gl.SliceLayout(0, PL))[None, :]
    for rt in gl.static_range(RTMAX):
        if rt * 16 < nr:
            ps = gl.sum(reds[rt].load(RL), axis=0)
            r_ = rt * 16 + pn
            gl.amd.cdna4.buffer_store(
                _pair(tag, ps.to(gl.int32, bitcast=True)),
                scratch,
                SCL + (part * RMAXL + r_) * E + tile * 16 + pm,
                mask=(r_ < nr) & (pm >= 0),
                cache=".wt",
            )


_all_reduce_rmsnorm_moe_route_repr = make_kernel_repr(
    "_all_reduce_rmsnorm_moe_route_kernel",
    [
        "HIDDEN",
        "WORLD",
        "E",
        "TOPK",
        "SCORE",
        "NGROUP",
        "WFP32",
        "QUANT",
        "PROTO",
        "RTMAX",
        "SORT_ROWS",
        "BLOCKS",
        "num_warps",
    ],
)


@gluon.jit(
    repr=_all_reduce_rmsnorm_moe_route_repr,
    do_not_specialize=["rows", "rank", "wait_us"],
)
def _all_reduce_rmsnorm_moe_route_kernel(
    inp,
    res,
    norm_w,
    router_w,
    bias,
    res_out,
    xn,
    xq,
    xs,
    topk_ids,
    topk_w,
    sorted_ids,
    sorted_w,
    sorted_eids,
    num_valid,
    moe_buf,
    scratch,
    colors,
    peers,
    rows,
    rank,
    wait_us,
    eps,
    routed_scale,
    HIDDEN: gl.constexpr,
    WORLD: gl.constexpr,
    E: gl.constexpr,
    TOPK: gl.constexpr,
    SCORE: gl.constexpr,
    HAS_BIAS: gl.constexpr,
    NGROUP: gl.constexpr,
    TOPKG: gl.constexpr,
    RENORM: gl.constexpr,
    GEMMA: gl.constexpr,
    HAS_RES: gl.constexpr,
    QUANT: gl.constexpr,
    WFP32: gl.constexpr,
    SBLK: gl.constexpr,
    ZERO: gl.constexpr,
    BLOCKS: gl.constexpr,
    PROTO: gl.constexpr,
    RTMAX: gl.constexpr,
    SORT_ROWS: gl.constexpr,
    PMAX: gl.constexpr,
    KP: gl.constexpr,
    NRC: gl.constexpr,
    TPC: gl.constexpr,
    S1MAX: gl.constexpr,
    GMAX: gl.constexpr,
    SMAX: gl.constexpr,
    RMAXL: gl.constexpr,
    OS0: gl.constexpr,
    RS0: gl.constexpr,
    AG0: gl.constexpr,
    PK0: gl.constexpr,
    SCX: gl.constexpr,
    SCL: gl.constexpr,
    SCP: gl.constexpr,
):
    """inp, res, res_out, xn: [rows, HIDDEN] bf16 as int32. norm_w: [HIDDEN] bf16 as int32.
    router_w: [E, HIDDEN] fp32 (WFP32) or bf16. bias: [E] fp32. xq: [rows, HIDDEN] e4m3 as
    int32, xs: [rows, HIDDEN / 32] uint8. topk_ids, topk_w: [rows, TOPK] int32 / fp32. sorted_*,
    num_valid: the moe_sorting outputs (block SBLK). moe_buf: [rows, HIDDEN] bf16 as int32.
    scratch: int64 words (ready words at SCX, partial logits at SCL [KP][RMAXL][E], one-shot
    picks at SCP [SMAX][2 TOPK]). colors: [BLOCKS] int32. peers: the WORLD IPC buffer addresses
    of ranks rank, rank + 1, ...: error record, one-shot inbox at byte OS0 [2][WORLD][S1MAX]
    [HIDDEN] bf16, reduce-scatter inbox at RS0 and all-gather inbox at AG0 [2][WORLD][GMAX]
    [HIDDEN], picks at PK0 (int64 words) [2][SMAX][2 TOPK]."""
    W: gl.constexpr = gl.num_warps()
    T: gl.constexpr = 64 * W
    NV: gl.constexpr = HIDDEN // 8
    V: gl.constexpr = (NV + T - 1) // T
    A: gl.constexpr = NV // V
    NP: gl.constexpr = WORLD - 1
    ROWW: gl.constexpr = NV * 4  # the dwords of a row
    NKS: gl.constexpr = HIDDEN // 16  # router k-steps
    SPT: gl.constexpr = (NKS + KP - 1) // KP
    SPW: gl.constexpr = (SPT + W - 1) // W
    EP: gl.constexpr = _next_pow2(E + 1)
    NPK: gl.constexpr = 2 * TOPK
    gl.static_assert(HIDDEN % 64 == 0, "HIDDEN must be a multiple of 64")
    gl.static_assert(
        NV % V == 0, "HIDDEN / 8 must be a multiple of the vectors a thread"
    )
    gl.static_assert(A % 4 == 0, "a 32-column group must be one quad of lanes")
    gl.static_assert(E % 16 == 0, "E must be a multiple of 16")
    gl.static_assert(TOPK <= 64, "TOPK must be at most 64")
    gl.static_assert(E <= 2048, "E must be at most 2048")
    gl.static_assert(len(peers) == WORLD, "one buffer for each rank")
    bufs = ()
    for d in gl.static_range(WORLD):
        bufs = bufs + (gl.multiple_of(peers[d].to(gl.pointer_type(gl.int32)), 16),)
    own = bufs[0]
    bid = gl.program_id(0)
    tid = _tids(W)
    lane = tid % 64
    act = tid < A
    k4 = _k4(W)
    LV: gl.constexpr = gl.BlockedLayout([1, 4], [64, 1], [W, 1], [1, 0])
    colour = gl.load(colors + bid)
    par = colour & 1
    tag = colour + 1
    ticks = wait_us.to(gl.int64) * 100  # s_memrealtime is 100 MHz
    G = (rows + WORLD - 1) // WORLD

    # The rows this rank routes: all rows (one-shot) or its owner rows (two-shot).
    if PROTO == 0:
        row0 = rows * 0
        nr = rows
        first = rows
    else:
        row0 = rank * G
        nr = gl.maximum(gl.minimum(G, rows - rank * G), 0)
        first = G
    rt0: gl.constexpr = BLOCKS - NRC
    is_router = bid >= rt0
    rtask = gl.where(is_router, bid - rt0, 0)

    # Router weights, loaded first on the router CTAs (their waits come last).
    mf: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 16], transposed=False, warps_per_cta=[W, 1, 1]
    )
    la: gl.constexpr = gl.DotOperandLayout(0, mf, k_width=4)
    aw = gl.arange(0, W, layout=gl.SliceLayout(1, gl.SliceLayout(2, la)))[:, None, None]
    am = gl.arange(0, 16, layout=gl.SliceLayout(0, gl.SliceLayout(2, la)))[
        None, :, None
    ]
    ak = gl.arange(0, 16, layout=gl.SliceLayout(0, gl.SliceLayout(1, la)))[
        None, None, :
    ]
    NTASK: gl.constexpr = E // 16 * KP
    wa = _router_w(
        router_w,
        rtask,
        is_router & (rtask < NTASK),
        aw,
        am,
        ak,
        HIDDEN,
        KP,
        SPT,
        SPW,
        NKS,
    )

    # Push this rank's rows: one-shot, every row to every peer; two-shot, a peer's owner rows to
    # it. Unit u is peer d = u % NP + 1, row and vector block u // NP. Loads before stores.
    # The router CTAs push nothing when the CTAs between the reduce and router CTAs can: their
    # polls would wait for the acks of their remote stores.
    units = NP * first * V
    last = gl.where((rt0 - first) * PMAX >= units, rt0, BLOCKS)
    n_push = gl.minimum(last - first, units)
    pb = bid - first
    plan = ()
    for p in gl.static_range(PMAX):
        p_u = pb + p * n_push
        p_ok = (pb >= 0) & (pb < n_push) & (p_u < units)
        p_uu = gl.where(p_ok, p_u, 0)
        p_d = p_uu % NP + 1
        p_k = p_uu // NP
        p_r = p_k // V
        if PROTO == 0:
            p_row = p_r
        else:
            p_row = ((rank + p_d) % WORLD) * G + p_r
        p_ok = p_ok & (p_row < rows)
        p_vec = (p_k % V) * A + tid
        p_x = gl.amd.cdna4.buffer_load(
            inp,
            (gl.where(p_ok, p_row, 0) * ROWW + p_vec * 4)[:, None] + k4,
            mask=(act & p_ok)[:, None] & (k4 < 4),
        )
        plan = plan + ((p_ok, p_d, p_r, p_vec, p_x),)
    for p in gl.static_range(PMAX):
        p_ok, p_d, p_r, p_vec, p_x = plan[p]
        if PROTO == 0:
            p_slot = OS0 // 4 + ((par * WORLD + rank) * S1MAX + p_r) * ROWW
        else:
            p_slot = RS0 // 4 + ((par * WORLD + rank) * GMAX + p_r) * ROWW
        gl.amd.cdna4.buffer_store(
            _canon(p_x),
            _peer(bufs, p_d, WORLD),
            (p_slot + p_vec * 4)[:, None] + k4,
            mask=(act & p_ok)[:, None] & (k4 < 4),
        )

    # moe_buf := 0 (the MoE second stage adds into it), by the router CTAs.
    if ZERO:  # noqa: SIM102 - a compile-time branch around a runtime one
        if is_router:
            words = rows * (HIDDEN // 2)
            per = ((words + NRC - 1) // NRC + 4 * T - 1) // (4 * T) * (4 * T)
            zv = gl.zeros([T, 4], gl.int32, layout=LV)
            for c in range(0, per, 4 * T):
                wd = rtask * per + c + tid * 4
                gl.amd.cdna4.buffer_store(
                    zv,
                    moe_buf,
                    wd[:, None] + k4,
                    mask=((wd < words) & (c + tid * 4 < per))[:, None] & (k4 < 4),
                )

    sent = gl.full([T, 4], -1, gl.int32, LV)
    # Reduce + norm of routed row bid (one-shot: row bid; two-shot: owner row rank G + bid). The
    # ready word goes out right after the normed row; two-shot owners then push the sum.
    if bid < nr:
        t = row0 + bid
        own_w = ()
        res_w = ()
        w_w = ()
        for j in gl.static_range(V):
            vec = j * A + tid
            own_w = own_w + (_ld4(inp, t * ROWW + vec * 4, act, W),)
            if HAS_RES:
                res_w = res_w + (_ld4(res, t * ROWW + vec * 4, act, W),)
            w_w = w_w + (_ld4(norm_w, vec * 4, act, W),)
        o_offs = ()
        for o_d in gl.static_range(1, WORLD):
            if PROTO == 0:
                o_base = (
                    OS0 // 4
                    + ((par * WORLD + (rank + o_d) % WORLD) * S1MAX + bid) * ROWW
                )
            else:
                o_base = (
                    RS0 // 4
                    + ((par * WORLD + (rank + o_d) % WORLD) * GMAX + bid) * ROWW
                )
            for o_j in gl.static_range(V):
                o_offs = o_offs + (o_base + (o_j * A + tid) * 4,)
        o_addrs = ()
        for q_i in gl.static_range(NP * V):
            o_addrs = o_addrs + (own + o_offs[q_i],)
        o_failed, o_got = _poll_slots(own, o_addrs, act, ticks, NP * V)
        _fail_record(bufs, rank, colour, _SITE_REDUCE, (lane == 0) & (o_failed == 1))
        hw = _row_sum(own_w, o_got, rank, WORLD, V, tid)
        s, y = _norm_row(hw, res_w, w_w, act, eps, HIDDEN, V, GEMMA, HAS_RES, W)
        for j in gl.static_range(V):
            vec = j * A + tid
            _st4(
                xn,
                t * ROWW + vec * 4,
                (y[4 * j], y[4 * j + 1], y[4 * j + 2], y[4 * j + 3]),
                act,
                W,
                ".wt",
            )
        dr = _drain(tid)
        gl.barrier()
        gl.amd.cdna4.buffer_store(
            _pair(tag, dr * 0 + 1),
            scratch,
            dr * 0 + (SCX + bid),
            mask=tid == 0,
            cache=".wt",
        )
        if PROTO == 1:
            o_ag = AG0 // 4 + ((par * WORLD + rank) * GMAX + bid) * ROWW
            for o_j in gl.static_range(V):
                cw = (
                    _canon(hw[4 * o_j]),
                    _canon(hw[4 * o_j + 1]),
                    _canon(hw[4 * o_j + 2]),
                    _canon(hw[4 * o_j + 3]),
                )
                for o_d in gl.static_range(1, WORLD):
                    _st4(bufs[o_d], o_ag + (o_j * A + tid) * 4, cw, act, W)
        _store_rest(s, y, xq, xs, res_out, t, act, A, HIDDEN, V, QUANT, W)
        for q_i in gl.static_range(NP * V):
            gl.amd.cdna4.buffer_store(
                sent, own, o_offs[q_i][:, None] + k4, mask=act[:, None] & (k4 < 4)
            )

    # Router tasks: 16 experts x one K part of the routed rows each.
    if is_router:
        rv = tid < nr
        x_failed, _xr = _poll_tags(
            own, (scratch + SCX + gl.where(rv, tid, 0),), rv, tag, ticks, 1, False
        )
        _fail_record(bufs, rank, colour, _SITE_ROUTER, (lane == 0) & (x_failed == 1))
        gl.barrier()
        for tt in gl.static_range(TPC):
            task = rtask + tt * NRC
            if task < NTASK:
                if tt == 0:
                    wt = wa
                else:
                    wt = _router_w(
                        router_w,
                        task,
                        task < NTASK,
                        aw,
                        am,
                        ak,
                        HIDDEN,
                        KP,
                        SPT,
                        SPW,
                        NKS,
                    )
                _router_task(
                    xn,
                    scratch,
                    wt,
                    task,
                    row0,
                    nr,
                    tag,
                    HIDDEN,
                    E,
                    KP,
                    SPT,
                    SPW,
                    NKS,
                    RTMAX,
                    RMAXL,
                    SCL,
                    WFP32,
                    W,
                )

    # Route: one-shot, the row's own CTA; two-shot, the router CTAs (they pushed no sums).
    if PROTO == 0:
        if bid < nr:
            _route_row(
                scratch,
                own,
                bufs,
                peers,
                bias,
                bid,
                bid,
                par,
                tag,
                ticks,
                rank,
                colour,
                routed_scale,
                E,
                TOPK,
                SCORE,
                HAS_BIAS,
                NGROUP,
                TOPKG,
                RENORM,
                KP,
                RMAXL,
                SCL,
                SCP,
                PK0,
                SMAX,
                PROTO,
                WORLD,
                W,
            )
    else:
        if is_router:
            for rj in range(bid - rt0, nr, NRC):
                _route_row(
                    scratch,
                    own,
                    bufs,
                    peers,
                    bias,
                    rj,
                    row0 + rj,
                    par,
                    tag,
                    ticks,
                    rank,
                    colour,
                    routed_scale,
                    E,
                    TOPK,
                    SCORE,
                    HAS_BIAS,
                    NGROUP,
                    TOPKG,
                    RENORM,
                    KP,
                    RMAXL,
                    SCL,
                    SCP,
                    PK0,
                    SMAX,
                    PROTO,
                    WORLD,
                    W,
                )

    # Two-shot receivers (CTAs G .. BLOCKS - 2): the other ranks' rows, when their sums land.
    if PROTO == 1:  # noqa: SIM102 - a compile-time branch around a runtime one
        if (bid >= G) & (bid < BLOCKS - 1):
            for rk in range(bid - G, NP * G, BLOCKS - 1 - G):
                g_src = (rank + 1 + rk % NP) % WORLD
                r_r = rk // NP
                rt_ = g_src * G + r_r
                live = act & (rt_ < rows)
                tc = gl.where(rt_ < rows, rt_, 0)
                rres_w = ()
                rw_w = ()
                for rj_ in gl.static_range(V):
                    rvec = rj_ * A + tid
                    if HAS_RES:
                        rres_w = rres_w + (_ld4(res, tc * ROWW + rvec * 4, live, W),)
                    rw_w = rw_w + (_ld4(norm_w, rvec * 4, live, W),)
                a_base = AG0 // 4 + ((par * WORLD + g_src) * GMAX + r_r) * ROWW
                a_addrs = ()
                for r_j in gl.static_range(V):
                    a_addrs = a_addrs + (own + a_base + (r_j * A + tid) * 4,)
                a_failed, a_got = _poll_slots(own, a_addrs, live, ticks, V)
                _fail_record(
                    bufs, rank, colour, _SITE_GATHER, (lane == 0) & (a_failed == 1)
                )
                rhw = ()
                for r_j in gl.static_range(V):
                    for r_e in gl.static_range(4):
                        rhw = rhw + (a_got[r_j][r_e],)
                rs_, ry = _norm_row(
                    rhw, rres_w, rw_w, live, eps, HIDDEN, V, GEMMA, HAS_RES, W
                )
                for rj_ in gl.static_range(V):
                    rvec = rj_ * A + tid
                    _st4(
                        xn,
                        tc * ROWW + rvec * 4,
                        (
                            ry[4 * rj_],
                            ry[4 * rj_ + 1],
                            ry[4 * rj_ + 2],
                            ry[4 * rj_ + 3],
                        ),
                        live,
                        W,
                    )
                _store_rest(rs_, ry, xq, xs, res_out, tc, live, A, HIDDEN, V, QUANT, W)
                for r_j in gl.static_range(V):
                    gl.amd.cdna4.buffer_store(
                        sent,
                        own,
                        (a_base + (r_j * A + tid) * 4)[:, None] + k4,
                        mask=live[:, None] & (k4 < 4),
                    )

    # Sort (last CTA): every row's picks -> top-k outputs and the moe_sorting outputs.
    if bid == BLOCKS - 1:
        SW: gl.constexpr = (SORT_ROWS + 31) // 32
        SLs: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [0])
        LS: gl.constexpr = gl.BlockedLayout([1], [64], [W], [0])
        ts = gl.arange(0, T, layout=LS)
        BMN: gl.constexpr = _next_pow2(E * SW)
        PKN: gl.constexpr = _next_pow2(_cmax(SORT_ROWS * TOPK, T))
        # The word prefix counts get their own table when it fits in LDS.
        PRE: gl.constexpr = BMN * 8 + PKN * 8 <= 98304
        bm = gl.allocate_shared_memory(gl.int32, [BMN], SLs)
        if PRE:
            pre = gl.allocate_shared_memory(gl.int32, [BMN], SLs)
        else:
            pre = bm
        cnt = gl.allocate_shared_memory(gl.int32, [EP], SLs)
        off_s = gl.allocate_shared_memory(gl.int32, [EP], SLs)
        pe = gl.allocate_shared_memory(gl.int32, [PKN], SLs)
        pw = gl.allocate_shared_memory(gl.int32, [PKN], SLs)
        for sc_ in gl.static_range((BMN + T - 1) // T):
            si = sc_ * T + ts
            bm.scatter(gl.zeros_like(ts), gl.minimum(si, BMN - 1), 0)
        gl.barrier()
        npk = rows * TOPK
        sc64 = scratch
        pk64 = gl.multiple_of(peers[0].to(gl.pointer_type(gl.int64)), 16)
        for sc_ in range(0, npk, T):
            sp_ = sc_ + ts
            spv = sp_ < npk
            spc = gl.where(spv, sp_, 0)
            st_ = spc // TOPK
            sj = spc % TOPK
            if PROTO == 0:
                a_id = sc64 + SCP + st_ * NPK + sj
                s_failed, s_vals = _poll_tags(
                    own, (a_id, a_id + TOPK), spv, tag, ticks, 2, False
                )
            else:
                a_id = pk64 + PK0 // 8 + (par * SMAX + st_) * NPK + sj
                s_failed, s_vals = _poll_tags(
                    own, (a_id, a_id + TOPK), spv, tag, ticks, 2, True
                )
            _fail_record(
                bufs, rank, colour, _SITE_SORT, (ts % 64 == 0) & (s_failed == 1)
            )
            se = gl.minimum(gl.maximum(s_vals[0], 0), E - 1)
            gl.store(topk_ids + spc, s_vals[0], mask=spv)
            gl.store(topk_w + spc, s_vals[1].to(gl.float32, bitcast=True), mask=spv)
            pe.scatter(se, sp_, 0)
            pw.scatter(s_vals[1], sp_, 0)
            bm.atomic_scatter_or(
                gl.full([T], 1, gl.int32, LS) << (st_ % 32),
                se * SW + st_ // 32,
                0,
                mask=spv,
            )
        gl.barrier()
        # Word prefix counts and the count of each expert.
        for sc_ in gl.static_range((E + T - 1) // T):
            se = sc_ * T + ts
            sec = gl.minimum(se, E - 1)
            srun = gl.zeros_like(ts)
            for w_ in gl.static_range(SW):
                if PRE:
                    pre.scatter(srun, sec * SW + w_, 0)
                srun = srun + _popc(bm.gather(sec * SW + w_, 0))
            cnt.scatter(srun, sec, 0)
        gl.barrier()
        LE: gl.constexpr = gl.BlockedLayout([(EP + T - 1) // T], [64], [W], [0])
        ee = gl.arange(0, EP, layout=LE)
        n_e = gl.where(ee < E, cnt.load(LE), 0)
        pad = (n_e + SBLK - 1) // SBLK * SBLK
        incl = gl.associative_scan(pad, 0, _add_i32)
        off_s.store(incl - pad)
        stotal = gl.max(incl, axis=0)
        L2s: gl.constexpr = gl.BlockedLayout([1], [64], [W], [0])
        two = gl.arange(0, 2, layout=L2s)
        gl.store(num_valid + two, gl.where(two == 0, stotal, rows))
        gl.barrier()
        for sc_ in range(0, npk, T):
            sp_ = sc_ + ts
            spv = sp_ < npk
            spc = gl.where(spv, sp_, 0)
            st_ = spc // TOPK
            sj = spc % TOPK
            se = pe.gather(spc, 0)
            swd = se * SW + st_ // 32
            srank = _popc(bm.gather(swd, 0) & ((1 << (st_ % 32)) - 1))
            if PRE:
                srank = srank + pre.gather(swd, 0)
            else:
                for w_ in gl.static_range(SW - 1):
                    srank = srank + gl.where(
                        w_ < st_ // 32, _popc(bm.gather(se * SW + w_, 0)), 0
                    )
            spos = off_s.gather(se, 0) + srank
            gl.store(sorted_ids + spos, (sj << 24) | st_, mask=spv)
            gl.store(
                sorted_w + spos,
                pw.gather(spc, 0).to(gl.float32, bitcast=True),
                mask=spv,
            )
        # The padding of each expert segment and the expert of each block.
        NBM: gl.constexpr = (SORT_ROWS + SBLK - 1) // SBLK
        for sc_ in gl.static_range((E * SBLK + T - 1) // T):
            si = sc_ * T + ts
            se = gl.minimum(si // SBLK, E - 1)
            sq_ = si % SBLK
            sne = cnt.gather(se, 0)
            so = off_s.gather(se, 0)
            spz = (sne + SBLK - 1) // SBLK * SBLK
            sokp = (si < E * SBLK) & (sne + sq_ < spz)
            gl.store(
                sorted_ids + so + sne + sq_,
                (TOPK << 24) | (rows + gl.zeros_like(ts)),
                mask=sokp,
            )
            gl.store(
                sorted_w + so + sne + sq_, gl.zeros_like(ts).to(gl.float32), mask=sokp
            )
        for sc_ in gl.static_range((E * NBM + T - 1) // T):
            si = sc_ * T + ts
            se = gl.minimum(si // NBM, E - 1)
            sq_ = si % NBM
            sne = cnt.gather(se, 0)
            so = off_s.gather(se, 0)
            snb = (sne + SBLK - 1) // SBLK
            gl.store(
                sorted_eids + so // SBLK + sq_, se, mask=(si < E * NBM) & (sq_ < snb)
            )

    # The next launch colour.
    gl.barrier()
    gl.store(
        colors + bid + gl.zeros_like(tid),
        colour + 1 + gl.zeros_like(tid),
        mask=tid == 0,
    )
