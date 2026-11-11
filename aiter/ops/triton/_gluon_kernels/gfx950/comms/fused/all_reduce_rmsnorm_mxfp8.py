# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""All-reduce + residual add + RMSNorm + MXFP8 quant over 2, 4 or 8 GPUs in one kernel.

The kernel is a bf16 reduce-scatter, then an all-gather of the bf16 sum. Rank g owns
the rows g * G to g * G + G - 1, with G = ceil(rows / WORLD). CTA b < G owns a row:
it reads the rows of the peers from its inbox, adds the ranks in rank order in fp32,
rounds once to bf16 and pushes the sum to the peers. The other CTAs push the rows of
this rank to their owners, then receive one sum each. The CTA of each row adds the
residual, normalizes the row and quantizes it.

The inboxes are Lamport slots. 0xFFFF marks an empty 16-bit word: in bf16 it is a NaN,
and a sender writes the NaN 0x7FFF in its place. A slot is ready when none of its words
is 0xFFFF. The reader writes 0xFFFF back. Each launch uses the inbox half of its parity: each CTA
keeps a launch colour. Each wait has a time limit. A wait that times out writes an
error record into the buffer of each rank.

The numbers do not depend on the timing. Thread t of A = HIDDEN / 16 threads holds the
columns 8t to 8t + 7 and HIDDEN / 2 + 8t to HIDDEN / 2 + 8t + 7. It adds the squares
of its 16 values in column order (no fma). Then a butterfly adds the 64 lanes of a
wave (lane offsets 32, 16, 8, 4, 2, 1), and the waves are added in order."""

from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.language.core import PropagateNan as _PropagateNan

from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr

_NAN_ALL: gl.constexpr = gl.constexpr(_PropagateNan.ALL)
_SITE_RS: gl.constexpr = gl.constexpr(1)
_SITE_AG: gl.constexpr = gl.constexpr(2)
_F32_TINY: gl.constexpr = gl.constexpr(1.1754943508222875e-38)
_SENT_SIGN: gl.constexpr = gl.constexpr(-0x7FFF8000)  # 0x80008000
_POLL_VGPR = 64  # The first of the fixed VGPRs that a poll loads into, 4 a slot.
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


_LPOLL = {n: _lpoll(n) for n in range(1, 9)}
_ERREC = {n: _errec(n) for n in (2, 4, 8)}

# The lane butterfly over offsets 32, 16, 8, 4, 2, 1: permlane swaps, then DPP. The
# s_nop wait states are explicit, because the compiler adds none for inline asm.
_BFLY_LINES = [
    "v_mov_b32 $0, $2",
    "v_mov_b32 $1, $2",
    "s_nop 1",
    "v_permlane32_swap_b32 $1, $0",
    "v_add_f32 $0, $1, $0",
    "v_mov_b32 $1, $0",
    "s_nop 1",
    "v_permlane16_swap_b32 $0, $1",
    "v_add_f32 $0, $0, $1",
    "v_mov_b32 $1, $0",
    "s_nop 1",
    "v_mov_b32_dpp $1, $1 row_shr:8 row_mask:0xf bank_mask:0xc",
    "s_nop 1",
    "v_mov_b32_dpp $1, $0 row_shl:8 row_mask:0xf bank_mask:0x3",
    "v_add_f32 $0, $0, $1",
    "v_mov_b32 $1, $0",
    "s_nop 1",
    "v_mov_b32_dpp $1, $1 row_shr:4 row_mask:0xf bank_mask:0xa",
    "s_nop 1",
    "v_mov_b32_dpp $1, $0 row_shl:4 row_mask:0xf bank_mask:0x5",
    "v_add_f32 $0, $0, $1",
    "v_mov_b32 $1, $0",
    "s_nop 1",
    "v_mov_b32_dpp $1, $1 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf",
    "v_add_f32 $0, $0, $1",
    "v_mov_b32 $1, $0",
    "s_nop 1",
    "v_mov_b32_dpp $1, $1 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf",
    "v_add_f32 $0, $0, $1",
]
_BFLY: gl.constexpr = gl.constexpr("\n".join(_BFLY_LINES))


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
def _tids(W: gl.constexpr):
    """The thread index, one value a thread."""
    LV: gl.constexpr = gl.BlockedLayout([1, 4], [64, 1], [W, 1], [1, 0])
    return gl.arange(0, 64 * W, layout=gl.SliceLayout(1, LV))


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
def _st4(ptr, dw, w, mask, W: gl.constexpr):
    k = _k4(W)
    gl.amd.cdna4.buffer_store(
        _join4(w, W), ptr, dw[:, None] + k, mask=mask[:, None] & (k < 4)
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
def _bfly_sum(x):
    out = gl.inline_asm_elementwise(
        _BFLY, "=&v,=&v,v", [x], (gl.float32, gl.float32), True, 1
    )
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
def _poll(own, addrs, act, ticks, NS: gl.constexpr):
    """Lamport poll of the NS slots in addrs. Returns failed and the 4 dwords of each
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


_all_reduce_rmsnorm_mxfp8_repr = make_kernel_repr(
    "_all_reduce_rmsnorm_mxfp8_kernel",
    ["HIDDEN", "WORLD", "GEMMA", "BLOCKS", "num_warps"],
)


@gluon.jit(
    repr=_all_reduce_rmsnorm_mxfp8_repr,
    do_not_specialize=["rows", "rank", "wait_us"],
)
def _all_reduce_rmsnorm_mxfp8_kernel(
    inp,
    res,
    norm_w,
    xq,
    xs,
    res_out,
    colors,
    peers,
    rows,
    rank,
    wait_us,
    eps,
    HIDDEN: gl.constexpr,
    WORLD: gl.constexpr,
    GEMMA: gl.constexpr,
    GMAX: gl.constexpr,
    RS0: gl.constexpr,
    AG0: gl.constexpr,
    BLOCKS: gl.constexpr,
    PMAX: gl.constexpr,
):
    """inp, res: [rows, HIDDEN] bf16 as int32 (the partial of this rank, the residual).
    norm_w: [HIDDEN] bf16 as int32. xq: [rows, HIDDEN] e4m3 as int32. xs: [rows,
    HIDDEN / 32] uint8. res_out: [rows, HIDDEN] bf16 as int32. colors: [BLOCKS] int32.
    peers: the WORLD buffer addresses of ranks rank, rank + 1, .... RS0 and AG0: byte
    offsets of the inboxes, each [2 parities][WORLD ranks][GMAX rows][HIDDEN bf16].
    PMAX: the most push units of a CTA."""
    W: gl.constexpr = gl.num_warps()
    V: gl.constexpr = 2  # 16-byte vectors of a thread
    NV: gl.constexpr = HIDDEN // 8
    A: gl.constexpr = NV // V
    NWA: gl.constexpr = (A + 63) // 64
    NP: gl.constexpr = WORLD - 1
    RSW: gl.constexpr = RS0 // 4
    AGW: gl.constexpr = AG0 // 4
    gl.static_assert(HIDDEN % 64 == 0, "HIDDEN must be a multiple of 64")
    gl.static_assert(A <= 64 * W, "too few warps for HIDDEN")
    gl.static_assert(len(peers) == WORLD, "one buffer for each rank")
    bufs = ()
    for d in gl.static_range(WORLD):
        bufs = bufs + (gl.multiple_of(peers[d].to(gl.pointer_type(gl.int32)), 16),)
    own = bufs[0]
    bid = gl.program_id(0)
    tid = _tids(W)
    lane = tid % 64
    wave = tid // 64
    act = tid < A
    k4 = _k4(W)
    colour = gl.load(colors + bid)
    par = colour & 1
    ticks = wait_us.to(gl.int64) * 100  # s_memrealtime is 100 MHz
    G = (rows + WORLD - 1) // WORLD

    # CTA b < G owns row rank * G + b. CTA G + k receives row k // NP of owner
    # rank + 1 + k % NP.
    is_owner = bid < G
    kr = bid - G
    is_recv = (kr >= 0) & (kr < NP * G)
    kc = gl.where(is_recv, kr, 0)
    g_src = (rank + 1 + kc % NP) % WORLD
    r_recv = kc // NP
    t_n = gl.where(is_owner, rank * G + gl.minimum(bid, G - 1), g_src * G + r_recv)
    has_row = (is_owner | is_recv) & (t_n < rows)
    tn_c = gl.where(has_row, t_n, 0)

    # Loads of the row data before the pushes and the waits.
    own_w = ()
    res_w = ()
    w_w = ()
    for j in gl.static_range(V):
        vec = j * A + tid
        own_w = own_w + (
            _ld4(inp, tn_c * (HIDDEN // 2) + vec * 4, act & is_owner & has_row, W),
        )
        res_w = res_w + (_ld4(res, tn_c * (HIDDEN // 2) + vec * 4, act & has_row, W),)
        w_w = w_w + (_ld4(norm_w, vec * 4, act & has_row, W),)

    # Reduce-scatter push. Unit u is peer d = u % NP + 1, row and vector block
    # u // NP. All loads go out before the first store.
    units = NP * G * V
    n_push = gl.minimum(BLOCKS - G, units)
    pb = bid - G
    plan = ()
    for p in gl.static_range(PMAX):
        p_u = pb + p * n_push
        p_ok = (pb >= 0) & (pb < n_push) & (p_u < units)
        p_uu = gl.where(p_ok, p_u, 0)
        p_d = p_uu % NP + 1
        p_k = p_uu // NP
        p_r = p_k // V
        p_row = ((rank + p_d) % WORLD) * G + p_r
        p_ok = p_ok & (p_row < rows)
        p_vec = (p_k % V) * A + tid
        p_x = gl.amd.cdna4.buffer_load(
            inp,
            (gl.where(p_ok, p_row, 0) * (HIDDEN // 2) + p_vec * 4)[:, None] + k4,
            mask=(act & p_ok)[:, None] & (k4 < 4),
        )
        plan = plan + ((p_ok, p_d, p_r, p_vec, p_x),)
    for p in gl.static_range(PMAX):
        p_ok, p_d, p_r, p_vec, p_x = plan[p]
        p_slot = RSW + ((par * WORLD + rank) * GMAX + p_r) * (NV * 4)
        gl.amd.cdna4.buffer_store(
            _canon(p_x),
            _peer(bufs, p_d, WORLD),
            (p_slot + p_vec * 4)[:, None] + k4,
            mask=(act & p_ok)[:, None] & (k4 < 4),
        )

    # The bf16 sum h: owners add the ranks and push h, receivers wait for it.
    if is_owner:
        o_addrs = ()
        for o_d in gl.static_range(1, WORLD):
            o_base = RSW + ((par * WORLD + (rank + o_d) % WORLD) * GMAX + bid) * (
                NV * 4
            )
            for o_j in gl.static_range(V):
                o_addrs = o_addrs + (own + o_base + (o_j * A + tid) * 4,)
        o_failed, o_got = _poll(own, o_addrs, act & has_row, ticks, NP * V)
        _fail_record(bufs, rank, colour, _SITE_RS, (lane == 0) & (o_failed == 1))
        hw = ()
        for o_j in gl.static_range(V):
            for o_e in gl.static_range(4):
                acc_lo = gl.zeros_like(tid).to(gl.float32)
                acc_hi = gl.zeros_like(tid).to(gl.float32)
                for src in gl.static_range(WORLD):
                    o_x = own_w[o_j][o_e]
                    for o_d in gl.static_range(1, WORLD):
                        o_x = gl.where(
                            (rank + o_d) % WORLD == src,
                            o_got[(o_d - 1) * V + o_j][o_e],
                            o_x,
                        )
                    if src == 0:
                        acc_lo = _bf_lo(o_x)
                        acc_hi = _bf_hi(o_x)
                    else:
                        acc_lo = acc_lo + _bf_lo(o_x)
                        acc_hi = acc_hi + _bf_hi(o_x)
                hw = hw + (_pack_bf16(acc_lo, acc_hi),)
        o_ag = AGW + ((par * WORLD + rank) * GMAX + bid) * (NV * 4)
        for o_j in gl.static_range(V):
            cw = (
                _canon(hw[4 * o_j]),
                _canon(hw[4 * o_j + 1]),
                _canon(hw[4 * o_j + 2]),
                _canon(hw[4 * o_j + 3]),
            )
            for o_d in gl.static_range(1, WORLD):
                _st4(bufs[o_d], o_ag + (o_j * A + tid) * 4, cw, act & has_row, W)
    else:
        r_ag = AGW + ((par * WORLD + g_src) * GMAX + r_recv) * (NV * 4)
        r_addrs = ()
        for r_j in gl.static_range(V):
            r_addrs = r_addrs + (own + r_ag + (r_j * A + tid) * 4,)
        r_failed, r_got = _poll(own, r_addrs, act & has_row, ticks, V)
        _fail_record(bufs, rank, colour, _SITE_AG, (lane == 0) & (r_failed == 1))
        hw = ()
        for r_j in gl.static_range(V):
            for r_e in gl.static_range(4):
                hw = hw + (r_got[r_j][r_e],)

    # Residual add, RMSNorm and MXFP8 quant of the row of this CTA.
    s = ()
    for j in gl.static_range(V):
        for e in gl.static_range(4):
            s = s + (
                _bf_lo(hw[4 * j + e]) + _bf_lo(res_w[j][e]),
                _bf_hi(hw[4 * j + e]) + _bf_hi(res_w[j][e]),
            )
    sq = s[0] * s[0]
    for i in gl.static_range(1, 8 * V):
        sq = sq + s[i] * s[i]
    sq = _bfly_sum(gl.where(act, sq, 0.0))
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
    xrow = tn_c * (HIDDEN // 4)
    for j in gl.static_range(V):
        vec = j * A + tid
        y = ()
        for e in gl.static_range(4):
            wl = _bf_lo(w_w[j][e])
            wh = _bf_hi(w_w[j][e])
            if GEMMA:
                wl = 1.0 + wl
                wh = 1.0 + wh
            yp = _pack_bf16(
                (s[8 * j + 2 * e] * rstd) * wl, (s[8 * j + 2 * e + 1] * rstd) * wh
            )
            y = y + (_bf_lo(yp), _bf_hi(yp))
        am = gl.abs(y[0])
        for e in gl.static_range(1, 8):
            am = gl.maximum(am, gl.abs(y[e]), propagate_nan=_NAN_ALL)
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
        c0 = _fp8_pack4(y[0] * inv, y[1] * inv, y[2] * inv, y[3] * inv)
        c1 = _fp8_pack4(y[4] * inv, y[5] * inv, y[6] * inv, y[7] * inv)
        _st2(xq, xrow + 2 * vec, c0, c1, act & has_row, W)
        gl.store(
            xs + tn_c * (HIDDEN // 32) + vec // 4,
            sb.to(gl.uint8),
            mask=act & has_row & (lane % 4 == 0),
        )
    for j in gl.static_range(V):
        vec = j * A + tid
        rw = ()
        for e in gl.static_range(4):
            rw = rw + (_pack_bf16(s[8 * j + 2 * e], s[8 * j + 2 * e + 1]),)
        _st4(res_out, tn_c * (HIDDEN // 2) + vec * 4, rw, act & has_row, W)

    # Write the empty mark back into the slots of this CTA, then advance the colour.
    sent = gl.full(
        [64 * W, 4], -1, gl.int32, gl.BlockedLayout([1, 4], [64, 1], [W, 1], [1, 0])
    )
    if is_owner:
        for q_d in gl.static_range(1, WORLD):
            q_base = RSW + ((par * WORLD + (rank + q_d) % WORLD) * GMAX + bid) * (
                NV * 4
            )
            for q_j in gl.static_range(V):
                gl.amd.cdna4.buffer_store(
                    sent,
                    own,
                    (q_base + (q_j * A + tid) * 4)[:, None] + k4,
                    mask=(act & has_row)[:, None] & (k4 < 4),
                )
    else:
        q_ag = AGW + ((par * WORLD + g_src) * GMAX + r_recv) * (NV * 4)
        for q_j in gl.static_range(V):
            gl.amd.cdna4.buffer_store(
                sent,
                own,
                (q_ag + (q_j * A + tid) * 4)[:, None] + k4,
                mask=(act & has_row)[:, None] & (k4 < 4),
            )
    gl.barrier()
    gl.store(
        colors + bid + gl.zeros_like(tid),
        colour + 1 + gl.zeros_like(tid),
        mask=tid == 0,
    )
