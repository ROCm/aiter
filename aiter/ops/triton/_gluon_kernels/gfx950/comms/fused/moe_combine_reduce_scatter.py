# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""MoE top-k combine + reduce-scatter over 2, 4 or 8 GPUs in one kernel (push mode).

A rank adds the top-k rows of a token with an fp32 fma chain and rounds once to bf16,
as mxfp4_moe_scatter_reduce does. It writes the row into the inbox of the owner rank.
The owner adds the ranks in rank order in fp32 and rounds once to bf16. A row flag
holds the launch colour, so the flags are never cleared. Each wait has a time limit.
A wait that times out writes an error record into the buffer of each rank."""

from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr

_SITE_DONE: gl.constexpr = gl.constexpr(1)
_SITE_FLAG: gl.constexpr = gl.constexpr(2)


@gluon.constexpr_function
def _widths(D, NT):
    """The ceil(D / NT) dwords of a row that a thread holds, as power-of-two chunks."""
    P = -(-D // NT)
    return [4] * (P // 4) + ([2] if P % 4 >= 2 else []) + ([1] if P % 2 else [])


@gluon.constexpr_function
def _n_chunks(D, NT):
    return len(_widths(D, NT))


@gluon.constexpr_function
def _chunk_w(D, NT, c):
    return _widths(D, NT)[c]


@gluon.constexpr_function
def _chunk_off(D, NT, c):
    return NT * sum(_widths(D, NT)[:c])


@gluon.constexpr_function
def _chunk_tail(D, NT, c):
    """Chunk c goes past the end of the row, so its operations need a column mask."""
    return NT * sum(_widths(D, NT)[: c + 1]) > D


@gluon.constexpr_function
def _sload_imm(nb, per):
    """nb SGPR bases x per dwords at offsets 0, 4, ...: nb * per SGPR outputs."""
    n = nb * per
    lines = [
        f"s_load_dword ${b * per + k}, ${n + b}, {4 * k:#x}"
        for b in range(nb)
        for k in range(per)
    ]
    lines.append("s_waitcnt lgkmcnt(0)")
    return "\n".join(lines), ",".join(["=&s"] * n + ["s"] * nb + ["~{memory}"])


@gluon.constexpr_function
def _sload_soff(n):
    """One SGPR base and n SGPR byte offsets: n SGPR outputs."""
    lines = [f"s_load_dword ${i}, ${n}, ${n + 1 + i}" for i in range(n)]
    lines.append("s_waitcnt lgkmcnt(0)")
    return "\n".join(lines), ",".join(["=&s"] * n + ["s"] * (n + 1) + ["~{memory}"])


@gluon.constexpr_function
def _poll(n):
    """Bounded system-scope poll of n words, when the uniform input act is not 0.

    Waits until (v_i - want) >= 0 (signed) for the n words of each active lane.
    Outputs: failed (0 ok, 1 timed out, 2 an error record was seen), n + 1 scratch
    VGPRs, now, pending and deadline. Inputs: act, n addresses, the error-word
    address, want and ticks."""
    fail, sc0, err, now, pend, dl = 0, 1, 1 + n, 2 + n, 3 + n, 4 + n
    act = 5 + n
    a0 = act + 1
    aerr, want, ticks = a0 + n, a0 + n + 1, a0 + n + 2
    lines = [
        f"s_mov_b32 ${fail}, 0",
        f"v_cmp_ne_u32 vcc, 0, ${act}",
        "s_cmp_eq_u64 vcc, 0",
        "s_cbranch_scc1 .Lpdone${:uid}",
        f"s_memrealtime ${now}",
        "s_waitcnt lgkmcnt(0)",
        f"v_lshl_add_u64 ${dl}, ${now}, 0, ${ticks}",
        ".Lpoll${:uid}:",
    ]
    lines += [f"global_load_dword ${sc0 + i}, ${a0 + i}, off sc0 sc1" for i in range(n)]
    lines += [
        f"global_load_dword ${err}, ${aerr}, off sc0 sc1",
        "s_waitcnt vmcnt(0)",
        f"s_mov_b64 ${pend}, 0",
    ]
    for i in range(n):
        lines += [
            f"v_sub_u32 ${sc0 + i}, ${sc0 + i}, ${want}",
            f"v_cmp_gt_i32 vcc, 0, ${sc0 + i}",
            f"s_or_b64 ${pend}, ${pend}, vcc",
        ]
    lines += [
        f"s_cmp_eq_u64 ${pend}, 0",
        "s_cbranch_scc1 .Lpdone${:uid}",
        f"v_cmp_ne_u32 vcc, 0, ${err}",
        "s_cmp_lg_u64 vcc, 0",
        "s_cbranch_scc1 .Lpgive${:uid}",
        f"s_memrealtime ${now}",
        "s_waitcnt lgkmcnt(0)",
        f"v_cmp_ge_u64 vcc, ${now}, ${dl}",
        "s_cmp_lg_u64 vcc, 0",
        "s_cbranch_scc1 .Lpout${:uid}",
        "s_sleep 1",
        "s_branch .Lpoll${:uid}",
        ".Lpgive${:uid}:",
        f"s_mov_b32 ${fail}, 2",
        "s_branch .Lpdone${:uid}",
        ".Lpout${:uid}:",
        f"s_mov_b32 ${fail}, 1",
        ".Lpdone${:uid}:",
    ]
    cons = (
        ["=&s"]
        + ["=&v"] * (n + 1)
        + ["=&s", "=&s", "=&v"]
        + ["v"]
        + ["v"] * (n + 3)
        + ["~{memory}", "~{vcc}", "~{scc}"]
    )
    return "\n".join(lines), ",".join(cons)


@gluon.constexpr_function
def _sload_imm_asm(nb, per):
    return _sload_imm(nb, per)[0]


@gluon.constexpr_function
def _sload_imm_cons(nb, per):
    return _sload_imm(nb, per)[1]


@gluon.constexpr_function
def _sload_soff_asm(n):
    return _sload_soff(n)[0]


@gluon.constexpr_function
def _sload_soff_cons(n):
    return _sload_soff(n)[1]


@gluon.constexpr_function
def _poll_asm(n):
    return _poll(n)[0]


@gluon.constexpr_function
def _poll_cons(n):
    return _poll(n)[1]


# vmcnt(0) + s_barrier when the uniform input is not 0. An scf.if with side effects in
# the loop makes LLVM add a vmcnt(0) at each loop header, so the condition is in the asm.
@gluon.constexpr_function
def _drain_bar_asm():
    return (
        "v_cmp_ne_u32 vcc, 0, $1\n"
        "s_cmp_eq_u64 vcc, 0\n"
        "s_cbranch_scc1 .Ldb${:uid}\n"
        "s_waitcnt vmcnt(0)\n"
        "s_barrier\n"
        ".Ldb${:uid}:\n"
        "s_mov_b32 $0, 0"
    )


@gluon.constexpr_function
def _drain_bar_cons():
    return "=s,v,~{memory},~{vcc},~{scc}"


@gluon.constexpr_function
def _store32_asm():
    return (
        "v_cmp_ne_u32 vcc, 0, $2\n"
        "s_and_saveexec_b64 $1, vcc\n"
        "s_cbranch_execz .Lst${:uid}\n"
        "global_store_dword $3, $4, off sc0 sc1\n"
        ".Lst${:uid}:\n"
        "s_mov_b64 exec, $1\n"
        "v_mov_b32 $0, 0"
    )


@gluon.constexpr_function
def _store32_cons():
    return "=v,=&s,v,v,v,~{memory},~{vcc},~{scc}"


# The error record [1, rank, colour, site] into n buffers, system scope: the high pair
# first, then the low pair, which holds the word that the waiters read.
@gluon.constexpr_function
def _errec_asm(n):
    return "\n".join(
        [
            "v_cmp_ne_u32 vcc, 0, $2",
            "s_and_saveexec_b64 $1, vcc",
            "s_cbranch_execz .Ler${:uid}",
        ]
        + [
            f"global_store_dwordx2 ${3 + b}, ${3 + n}, off offset:8 sc0 sc1"
            for b in range(n)
        ]
        + [f"global_store_dwordx2 ${3 + b}, ${4 + n}, off sc0 sc1" for b in range(n)]
        + [".Ler${:uid}:", "s_mov_b64 exec, $1", "v_mov_b32 $0, 0"]
    )


@gluon.constexpr_function
def _errec_cons(n):
    return ",".join(["=v", "=&s"] + ["v"] * (n + 3) + ["~{memory}", "~{vcc}", "~{scc}"])


@gluon.jit
def _thread_ids(W: gl.constexpr):
    L: gl.constexpr = gl.BlockedLayout([1], [64], [W], [0])
    return gl.arange(0, W * 64, layout=L)


@gluon.jit
def _cols(c: gl.constexpr, D: gl.constexpr, W: gl.constexpr):
    """Dword offsets in a row of chunk c: [64 W, w], w contiguous dwords a thread."""
    w: gl.constexpr = _chunk_w(D, W * 64, c)
    off: gl.constexpr = _chunk_off(D, W * 64, c)
    L: gl.constexpr = gl.BlockedLayout([1, w], [64, 1], [W, 1], [1, 0])
    r = gl.arange(0, W * 64, layout=gl.SliceLayout(1, L))
    k = gl.arange(0, w, layout=gl.SliceLayout(0, L))
    return off + r[:, None] * w + k[None, :]


@gluon.jit
def _cmask(c: gl.constexpr, D: gl.constexpr, W: gl.constexpr, cond):
    """The uniform cond, and the column mask of a chunk that goes past the row."""
    if _chunk_tail(D, W * 64, c):
        return cond & (_cols(c, D, W) < D)
    else:
        return cond


@gluon.jit
def _in_part(ptr, part, DP: gl.constexpr, PARTS: gl.constexpr):
    """The start of the column part of this CTA in a row at ptr."""
    if PARTS == 1:
        return ptr
    else:
        return ptr + part * DP


@gluon.jit
def _flag_ptr(base, slot, r, part, RMAX: gl.constexpr, PARTS: gl.constexpr):
    """The flag of (inbox slot, row r, column part)."""
    if PARTS == 1:
        return base + slot * RMAX + r
    else:
        return base + (slot * RMAX + r) * PARTS + part


@gluon.jit
def _to_peers(tid, bufs, WORLD: gl.constexpr):
    """bufs[tid + 1] in lane tid, for the lanes tid < WORLD - 1."""
    if WORLD == 2:
        return bufs[1] + gl.zeros_like(tid)
    else:
        lanes = ()
        for k in gl.static_range(WORLD - 2):
            lanes = lanes + (tid == k,)
        dst = bufs[WORLD - 1]
        for k in gl.static_range(WORLD - 2):
            dst = gl.where(lanes[WORLD - 3 - k], bufs[WORLD - 2 - k], dst)
        return dst


@gluon.jit
def _drain():
    """The vector memory operations of this wave are complete (vmcnt 0)."""
    gl.inline_asm_elementwise(
        "s_waitcnt vmcnt(0)\ns_mov_b32 $0, 0", "=s,~{memory}", [], gl.int32, False, 1
    )


@gluon.jit
def _store32_sys(addr, value, active):
    """System-scope 32-bit store of value at addr, in the lanes where active."""
    return gl.inline_asm_elementwise(
        _store32_asm(),
        _store32_cons(),
        [active.to(gl.int32), addr, value],
        (gl.int32, gl.int64),
        False,
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
    args = (active.to(gl.int32),)
    for b in gl.static_range(len(bufs)):
        args = args + (bufs[b].to(gl.pointer_type(gl.int64)),)
    gl.inline_asm_elementwise(
        _errec_asm(len(bufs)),
        _errec_cons(len(bufs)),
        args + (hi, lo),
        (gl.int32, gl.int64),
        False,
        1,
    )


@gluon.jit
def _pack_bf16(lo, hi):
    """Two fp32 values to one dword of 2 bf16 (round to nearest even), lo first."""
    return gl.inline_asm_elementwise(
        "v_cvt_pk_bf16_f32 $0, $1, $2", "=v,v,v", [lo, hi], gl.int32, True, 1
    )


@gluon.jit
def _bf_lo(x):
    return (x << 16).to(gl.float32, bitcast=True)


@gluon.jit
def _bf_hi(x):
    return (x & -65536).to(gl.float32, bitcast=True)


@gluon.jit
def _row_loads(
    inter,
    rev,
    sw,
    toks,
    valids,
    max_sorted,
    m,
    part,
    D: gl.constexpr,
    DP: gl.constexpr,
    PARTS: gl.constexpr,
    W: gl.constexpr,
    TOPK: gl.constexpr,
):
    """For each token of toks: the TOPK sorted positions and weights (scalar loads)
    and the dwords of each intermediate row of this thread (no load when not valid).
    Returns (positions, weights, rows) in (token, k) order."""
    NCH: gl.constexpr = _n_chunks(DP, W * 64)
    NB: gl.constexpr = len(toks)
    bases = ()
    for b in gl.static_range(NB):
        bases = bases + (rev + gl.minimum(toks[b], m - 1) * TOPK,)
    ps = gl.inline_asm_elementwise(
        _sload_imm_asm(NB, TOPK),
        _sload_imm_cons(NB, TOPK),
        bases,
        (gl.int32,) * (NB * TOPK),
        False,
        1,
    )
    xs = ()
    offs = ()
    for q in gl.static_range(NB * TOPK):
        p = ps[q]
        ok = (p >= 0) & (p < max_sorted)
        pc = gl.where(ok, p, 0)
        offs = offs + (pc * 4,)
        v = valids[q // TOPK] & ok
        row = _in_part(inter + pc.to(gl.int64) * D, part, DP, PARTS)
        x = ()
        for c in gl.static_range(NCH):
            x = x + (
                gl.amd.cdna4.buffer_load(
                    row, _cols(c, DP, W), mask=_cmask(c, DP, W, v)
                ),
            )
        xs = xs + (x,)
    wbits = gl.inline_asm_elementwise(
        _sload_soff_asm(NB * TOPK),
        _sload_soff_cons(NB * TOPK),
        (sw,) + offs,
        (gl.int32,) * (NB * TOPK),
        False,
        1,
    )
    ws = ()
    for q in gl.static_range(NB * TOPK):
        ws = ws + (wbits[q].to(gl.float32, bitcast=True),)
    return ps, ws, xs


@gluon.jit
def _combine(
    ps,
    ws,
    xs,
    q0: gl.constexpr,
    max_sorted,
    DP: gl.constexpr,
    W: gl.constexpr,
    TOPK: gl.constexpr,
):
    """acc = fma(x_k, w_k, acc) from acc = 0 for the valid k in order, then bf16
    pairs, as the loop of mxfp4_moe_scatter_reduce. Returns one tensor a chunk."""
    NCH: gl.constexpr = _n_chunks(DP, W * 64)
    out = ()
    for c in gl.static_range(NCH):
        alo = gl.zeros_like(xs[q0][c]).to(gl.float32)
        ahi = gl.zeros_like(xs[q0][c]).to(gl.float32)
        for k in gl.static_range(TOPK):
            p = ps[q0 + k]
            ok = (p >= 0) & (p < max_sorted)
            x = xs[q0 + k][c]
            alo = gl.where(ok, gl.fma(_bf_lo(x), ws[q0 + k], alo), alo)
            ahi = gl.where(ok, gl.fma(_bf_hi(x), ws[q0 + k], ahi), ahi)
        out = out + (_pack_bf16(alo, ahi),)
    return out


@gluon.jit
def _poll_flags(
    own,
    r,
    part,
    want,
    err_addr,
    ticks,
    active,
    F0: gl.constexpr,
    RMAX: gl.constexpr,
    PARTS: gl.constexpr,
    WORLD: gl.constexpr,
    W: gl.constexpr,
):
    """When active: waits until the WORLD - 1 peer flags of owner row r hold want.
    Returns failed: 0 ok, 1 timed out, 2 an error record was seen."""
    tid = _thread_ids(W)
    z = gl.zeros_like(tid)
    if PARTS == 1:
        f = own + F0 + r + z
    else:
        f = own + F0 + r * PARTS + part + z
    args = (active.to(gl.int32), f)
    for s in gl.static_range(1, WORLD - 1):
        args = args + (f + s * (RMAX * PARTS),)
    out = gl.inline_asm_elementwise(
        _poll_asm(WORLD - 1),
        _poll_cons(WORLD - 1),
        args + (err_addr + z, want + z, ticks + z.to(gl.int64)),
        (gl.int32,) * (WORLD + 1) + (gl.int64, gl.int64, gl.int64),
        False,
        1,
    )
    return out[0]


@gluon.jit
def _drain_barrier_if(cond):
    """When the uniform cond: vmcnt(0) in each wave, then a CTA barrier."""
    gl.inline_asm_elementwise(
        _drain_bar_asm(), _drain_bar_cons(), [cond.to(gl.int32)], gl.int32, False, 1
    )


@gluon.jit
def _iteration(
    inter,
    rev,
    sw,
    out,
    bufs,
    cur,
    ii,
    n_it,
    rb,
    part,
    m,
    rows,
    max_sorted,
    rank,
    want,
    colour,
    wait_ticks,
    D: gl.constexpr,
    DP: gl.constexpr,
    PARTS: gl.constexpr,
    W: gl.constexpr,
    TOPK: gl.constexpr,
    WORLD: gl.constexpr,
    NR: gl.constexpr,
    RMAX: gl.constexpr,
    F0: gl.constexpr,
    I0: gl.constexpr,
    LAG: gl.constexpr,
):
    """One step of the walk. Loads push row ii + 1 and the own rows of reduce row
    j = ii - LAG, waits for the flags of row j, publishes the flags of row ii - 1,
    loads the inbox rows of row j, pushes row ii, and reduces row j into out.
    Returns the loads for step ii + 1."""
    NCH: gl.constexpr = _n_chunks(DP, W * 64)
    NPK: gl.constexpr = (WORLD - 1) * TOPK
    own = bufs[0]
    tid = _thread_ids(W)
    rn = rb + (ii + 1) * NR
    vn = (ii + 1) < n_it
    j = ii - LAG
    vred = (j >= 0) & (j < n_it)
    rr = gl.where(vred, rb + j * NR, rb)
    t_red = rank * rows + rr
    tok = vred & (t_red < m)
    toks = ()
    for d in gl.static_range(1, WORLD):
        toks = toks + (((rank + d) % WORLD) * rows + rn,)
    valids = ()
    for d in gl.static_range(WORLD - 1):
        valids = valids + (vn & (toks[d] < m),)
    ps, ws, xs = _row_loads(
        inter,
        rev,
        sw,
        toks + (t_red,),
        valids + (tok,),
        max_sorted,
        m,
        part,
        D,
        DP,
        PARTS,
        W,
        TOPK,
    )
    # Wait for the peer rows of row j.
    failed = _poll_flags(
        own, rr, part, want, own, wait_ticks, tok, F0, RMAX, PARTS, WORLD, W
    )
    _fail_record(bufs, rank, colour, _SITE_FLAG, ((tid % 64) == 0) & (failed == 1))
    # All waves drained the pushes of row ii - 1: publish its flags to the peers.
    jp = ii - 1
    vp = (jp >= 0) & (jp < n_it)
    _drain_barrier_if(vp)
    rp = rb + jp * NR
    d = tid + 1
    o = (rank + d) % WORLD
    dst = _to_peers(tid, bufs, WORLD)
    fa = _flag_ptr(dst + F0, WORLD - 1 - d, rp, part, RMAX, PARTS)
    _store32_sys(
        fa,
        want + gl.zeros_like(tid),
        (tid < WORLD - 1) & (o * rows + rp < m) & vp,
    )
    # The inbox rows of row j, by source rank (no load for the own rank).
    ins = ()
    for src in gl.static_range(WORLD):
        sl = (src - rank - 1 + WORLD) % WORLD
        slc = gl.where(src == rank, 0, sl)
        base = _in_part(own + I0 + (slc * RMAX + rr).to(gl.int64) * D, part, DP, PARTS)
        v = tok & (src != rank)
        x = ()
        for c in gl.static_range(NCH):
            x = x + (
                gl.amd.cdna4.buffer_load(
                    base, _cols(c, DP, W), mask=_cmask(c, DP, W, v)
                ),
            )
        ins = ins + (x,)
    # Push row ii from the loads of the previous step.
    cps, cws, cxs = cur
    r_push = rb + ii * NR
    vpush = ii < n_it
    for d in gl.static_range(1, WORLD):
        tp_ = ((rank + d) % WORLD) * rows + r_push
        pk = _combine(cps, cws, cxs, (d - 1) * TOPK, max_sorted, DP, W, TOPK)
        base = bufs[d] + I0 + ((WORLD - 1 - d) * RMAX + r_push).to(gl.int64) * D
        base = _in_part(base, part, DP, PARTS)
        for c in gl.static_range(NCH):
            gl.amd.cdna4.buffer_store(
                pk[c],
                base,
                _cols(c, DP, W),
                mask=_cmask(c, DP, W, vpush & (tp_ < m)),
            )
    # Reduce row j: the own row (bf16) and the inbox rows, in rank order.
    own_pk = _combine(ps, ws, xs, NPK, max_sorted, DP, W, TOPK)
    obase = _in_part(out + rr.to(gl.int64) * D, part, DP, PARTS)
    for c in gl.static_range(NCH):
        olo = _bf_lo(own_pk[c])
        ohi = _bf_hi(own_pk[c])
        alo = gl.where(rank == 0, olo, _bf_lo(ins[0][c]))
        ahi = gl.where(rank == 0, ohi, _bf_hi(ins[0][c]))
        for src in gl.static_range(1, WORLD):
            alo = alo + gl.where(rank == src, olo, _bf_lo(ins[src][c]))
            ahi = ahi + gl.where(rank == src, ohi, _bf_hi(ins[src][c]))
        val = gl.where(tok, _pack_bf16(alo, ahi), 0)
        gl.amd.cdna4.buffer_store(
            val, obase, _cols(c, DP, W), mask=_cmask(c, DP, W, vred)
        )
    # The push rows of step ii + 1: the tokens of the peers.
    nps = ()
    nws = ()
    nxs = ()
    for q in gl.static_range(NPK):
        nps = nps + (ps[q],)
        nws = nws + (ws[q],)
        nxs = nxs + (xs[q],)
    return nps, nws, nxs


@gluon.jit
def _row_slot(bid, PARTS: gl.constexpr):
    if PARTS == 1:
        return bid
    else:
        return bid // PARTS


_moe_combine_reduce_scatter_repr = make_kernel_repr(
    "_moe_combine_reduce_scatter_kernel",
    ["HIDDEN", "TOPK", "WORLD", "PARTS", "BLOCKS", "LAG"],
)


@gluon.jit(
    repr=_moe_combine_reduce_scatter_repr,
    do_not_specialize=["wait_us", "m", "rows", "max_sorted", "rank"],
)
def _moe_combine_reduce_scatter_kernel(
    inter,
    rev,
    sw,
    out,
    colors,
    peers,
    wait_us,
    m,
    rows,
    max_sorted,
    rank,
    HIDDEN: gl.constexpr,
    TOPK: gl.constexpr,
    WORLD: gl.constexpr,
    PARTS: gl.constexpr,
    BLOCKS: gl.constexpr,
    RMAX: gl.constexpr,
    FLAG0: gl.constexpr,
    DONE0: gl.constexpr,
    INBOX0: gl.constexpr,
    LAG: gl.constexpr,
):
    """out[r] = the rank-order sum of the combined rows of token rank * rows + r.

    inter: [max_sorted, HIDDEN] bf16 as int32. rev: [m * TOPK] int32 sorted positions.
    sw: fp32 weight of each sorted position. out: [rows, HIDDEN] bf16 as int32.
    colors: [BLOCKS] int32 launch colours. peers: the WORLD buffer addresses of ranks
    rank, rank + 1, ... FLAG0, DONE0 and INBOX0 are byte offsets in a buffer.
    PARTS CTAs share a row, one column part each. Row slot s walks the owner rows
    s, s + BLOCKS / PARTS, ..."""
    W: gl.constexpr = gl.num_warps()
    D: gl.constexpr = HIDDEN // 2
    DP: gl.constexpr = D // PARTS
    NR: gl.constexpr = BLOCKS // PARTS
    F0: gl.constexpr = FLAG0 // 4
    DN0: gl.constexpr = DONE0 // 4
    I0: gl.constexpr = INBOX0 // 4
    NDONE: gl.constexpr = (WORLD - 1) * BLOCKS
    NQ: gl.constexpr = (NDONE + W * 64 - 1) // (W * 64)
    gl.static_assert(HIDDEN % 8 == 0, "HIDDEN must be a multiple of 8")
    gl.static_assert(D % PARTS == 0 and BLOCKS % PARTS == 0, "PARTS must divide both")
    gl.static_assert(LAG >= 2, "LAG must be 2 or more")
    gl.static_assert(NQ <= 4, "the done words must be 4 or less a thread")
    gl.static_assert(len(peers) == WORLD, "one buffer for each rank")
    bufs = ()
    for d in gl.static_range(WORLD):
        bufs = bufs + (gl.multiple_of(peers[d].to(gl.pointer_type(gl.int32)), 16),)
    own = bufs[0]
    bid = gl.program_id(0)
    rb = _row_slot(bid, PARTS)
    part = bid % PARTS
    tid = _thread_ids(W)
    colour = gl.load(colors + bid)
    want = colour + 1
    wait_ticks = wait_us.to(gl.int64) * 100  # s_memrealtime is 100 MHz
    n_it = gl.where(rb < rows, (rows - rb + (NR - 1)) // NR, 0)
    # Load the first push row, then wait until the previous launch is done.
    rn = rb
    toks = ()
    for d in gl.static_range(1, WORLD):
        toks = toks + (((rank + d) % WORLD) * rows + rn,)
    v0 = n_it > 0
    valids = ()
    for d in gl.static_range(WORLD - 1):
        valids = valids + (v0 & (toks[d] < m),)
    cur = _row_loads(
        inter, rev, sw, toks, valids, max_sorted, m, part, D, DP, PARTS, W, TOPK
    )
    if n_it > 0:
        # The inbox has one buffer: each CTA of each peer owner must have read it.
        # Done word i is CTA i % BLOCKS of owner rank + 1 + i // BLOCKS.
        addrs = ()
        for q in gl.static_range(NQ):
            i = (q * W * 64 + tid) % NDONE
            o = (rank + 1 + i // BLOCKS) % WORLD
            addrs = addrs + (own + DN0 + o * BLOCKS + i % BLOCKS,)
        z = gl.zeros_like(tid)
        res = gl.inline_asm_elementwise(
            _poll_asm(NQ),
            _poll_cons(NQ),
            (n_it,) + addrs + (own + z, colour + z, wait_ticks + z.to(gl.int64)),
            (gl.int32,) * (NQ + 2) + (gl.int64, gl.int64, gl.int64),
            False,
            1,
        )
        _fail_record(bufs, rank, colour, _SITE_DONE, ((tid % 64) == 0) & (res[0] == 1))
        gl.barrier()
    total = n_it + LAG
    # Steps in pairs: the loads alternate between 2 register sets, so the back edge
    # needs no copy (a copy waits for the loads, and so for the earlier pushes).
    for t in range((total + 1) // 2):
        ii = t * 2
        cur = _iteration(
            inter,
            rev,
            sw,
            out,
            bufs,
            cur,
            ii,
            n_it,
            rb,
            part,
            m,
            rows,
            max_sorted,
            rank,
            want,
            colour,
            wait_ticks,
            D,
            DP,
            PARTS,
            W,
            TOPK,
            WORLD,
            NR,
            RMAX,
            F0,
            I0,
            LAG,
        )
        cur = _iteration(
            inter,
            rev,
            sw,
            out,
            bufs,
            cur,
            ii + 1,
            n_it,
            rb,
            part,
            m,
            rows,
            max_sorted,
            rank,
            want,
            colour,
            wait_ticks,
            D,
            DP,
            PARTS,
            W,
            TOPK,
            WORLD,
            NR,
            RMAX,
            F0,
            I0,
            LAG,
        )
    # Done: this CTA read its inbox rows. Tell each peer and advance the colour.
    _drain()
    gl.barrier()
    dst = _to_peers(tid, bufs, WORLD)
    _store32_sys(
        dst + DN0 + rank * BLOCKS + bid, want + gl.zeros_like(tid), tid < WORLD - 1
    )
    gl.store(
        colors + bid + gl.zeros_like(tid), want + gl.zeros_like(tid), mask=tid == 0
    )
