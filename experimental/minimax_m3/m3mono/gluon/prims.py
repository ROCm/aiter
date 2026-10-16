# SPDX-License-Identifier: Apache-2.0
"""Gluon device primitives of the M3 mono kernel (gfx950; from the DSv4.1 Gluon mono).

Execution model: one CTA a CU (256), 512 threads (8 waves of 64). ``T1`` is the
one-value-a-thread layout of a [512] tensor; per-thread scalars live there.

Hand-offs (ISA verified in dsv41_monokernel tests/probes/probe_cache3.py):
- tagged pair: a 64-bit word (tag << 32 | value) stored write-through
  (``buffer_store .wt`` -> ``buffer_store_dwordx2 sc0 sc1``); the consumer spins on
  the whole word (``poll``: ``global_load_dwordx2 sc0 sc1``) until the tag matches,
  so value and tag are observed together.
- plain data then a flag: the producer stores write-through, ``drain()``s
  (``s_waitcnt vmcnt(0)`` + barrier), then puts the flag; the consumer polls the
  flag and reads the data with ``.cv`` (``sc0 sc1``) buffer loads.
Cache modifiers are passed as literals: closures into a jit body are dropped.
"""

import os

from triton.experimental import gluon
import triton.experimental.gluon.language as gl
from triton.experimental.gluon.language.amd import cdna4

# host-side ints, and their constexpr twins for jit code (a plain module global is
# not visible inside a jit body)
N_THREADS, N_WAVES, N_BLOCKS = 512, 8, 256
THREADS = gl.constexpr(N_THREADS)
WAVES = gl.constexpr(N_WAVES)
BLOCKS = gl.constexpr(N_BLOCKS)
T1 = gl.constexpr(gl.BlockedLayout([1], [64], [8], [0]))

# poll watchdog: iterations of (load round trip + s_sleep 1); several seconds
POLL_LIMIT = gl.constexpr(1 << int(os.environ.get("VLLM_M3_GLUON_POLL_LOG2", "22")))

# VLLM_M3_GLUON_TRACE=1: every CTA stamps s_memrealtime (100 MHz, one clock for the
# whole GPU) into its TRACE_SLOTS words of the scratch region "trace" as it
# finishes each stage (diagnostics only: adds a CTA barrier a stamp)
TRACE = gl.constexpr(int(os.environ.get("VLLM_M3_GLUON_TRACE", "0")))
# intra-GPU hand-offs at device scope (SC1: past the per-XCD L2s), the peer ones at system
# scope (SC0 | SC1); Triton's cache modifiers only reach the latter (~2-3 us a hop vs ~1)
DEV_SCOPE = gl.constexpr(int(os.environ.get("VLLM_M3_GLUON_DEV_SCOPE", "0")))  # 1: polls, 2: puts
FIRST_ERR = gl.constexpr(int(os.environ.get("VLLM_M3_GLUON_FIRST_ERR", "0")))
TRACE_SLOTS = gl.constexpr(32)


@gluon.constexpr_function
def _poll_asm(n, limit, sys=True):
    """Inline asm polling ``n`` 64-bit tagged words until every tag matches.

    Operands: outputs $0..$n-1 the words (i64), $n err (i32: 1 = watchdog fired),
    $n+1 tmp (i64), $n+2 saved exec (s64), $n+3 mask (s64), $n+4 count (s32);
    inputs $n+5 pred (lanes with pred == 0 skip), $n+6.. the n addresses (i64),
    then the tag zero-extended (i64). Lanes that matched drop out of exec, so
    their words are kept; the rest re-load all n words."""
    o = n + 5
    lines = [
        f"v_mov_b32 ${n}, 0",
        f"s_mov_b64 ${n + 2}, exec",
        f"v_cmp_ne_u32 vcc, 0, ${o}",
        "s_and_b64 exec, exec, vcc",
        "s_cbranch_execz 2f",
        f"s_mov_b32 ${n + 4}, 0",
        "1:",
    ]
    for i in range(n):
        lines.append(f"global_load_dwordx2 ${i}, ${o + 1 + i}, off " + ("sc0 sc1" if sys else "sc1"))
    lines += ["s_waitcnt vmcnt(0)", f"s_mov_b64 ${n + 3}, 0"]
    for i in range(n):
        lines += [
            f"v_lshrrev_b64 ${n + 1}, 32, ${i}",
            f"v_cmp_ne_u64 vcc, ${n + 1}, ${o + 1 + n}",
            f"s_or_b64 ${n + 3}, ${n + 3}, vcc",
        ]
    lines += [
        f"s_and_b64 exec, exec, ${n + 3}",
        "s_cbranch_execz 2f",
        "s_sleep 1",
        f"s_add_u32 ${n + 4}, ${n + 4}, 1",
        f"s_cmp_lt_u32 ${n + 4}, {limit}",
        "s_cbranch_scc1 1b",
        f"v_mov_b32 ${n}, 1",
        "2:",
        f"s_mov_b64 exec, ${n + 2}",
    ]
    return "\n".join(lines)


@gluon.constexpr_function
def _poll_constraints(n):
    outs = ["=&v"] * n + ["=&v", "=&v", "=&s", "=&s", "=&s"]
    ins = ["v"] + ["v"] * n + ["v"]
    return ",".join(outs + ins + ["~{vcc}", "~{scc}", "~{memory}"])


@gluon.constexpr_function
def _poll_dtypes(n):
    return [gl.int64] * n + [gl.int32, gl.int64, gl.int64, gl.int64, gl.int32]


# the wait asm's scalar temps live in fixed (clobbered) SGPRs: as packed outputs they
# would be duplicated PACK times (S = 4: out of registers)
_WS = ("s[90:91]", "s[92:93]", "s[94:95]", "s96")  # saved exec, pending mask, temp, count
_WS_CLOBBER = ",".join(f"~{{s{i}}}" for i in range(90, 97))


@gluon.constexpr_function
def _wait_asm(n, k, limit, sys=True):
    """FlyDSL's batched mailbox wait as one asm block: ``n`` regions x ``k`` packed
    elements (all of a thread's words). The words already peeked come in tied to the
    outputs; their tags are checked in registers first (no load when all landed). Then
    each lane still missing a tag re-loads every word, waits once and re-checks, alone:
    matched lanes drop out of exec, no CTA barrier.

    Operands (k each, ``pack`` order): outputs W_0..W_n-1 (i64), ERR (i32); inputs
    V_0..V_n-1 (tied to W), A_0..A_n-1 (i64 addresses), P_0..P_n-1 (i32, 0 = element
    unused), LO = tag << 32, HI = (tag + 1) << 32 (a word's tag matches iff LO <= w < HI)."""
    def W(t, e):
        return f"${t * k + e}"

    nout = (n + 1) * k

    def A(t, e):
        return f"${nout + n * k + t * k + e}"

    def P(t, e):
        return f"${nout + 2 * n * k + t * k + e}"

    LO = f"${nout + 3 * n * k}"
    HI = f"${nout + 3 * n * k + k}"
    SEXEC, MASK, PM, CNT = _WS
    cm = "sc0 sc1" if sys else "sc1"
    check = [f"s_mov_b64 {MASK}, 0"]
    for t in range(n):
        for e in range(k):
            check += [
                f"v_cmp_lt_u64 vcc, {W(t, e)}, {LO}",
                f"v_cmp_ge_u64_e64 {PM}, {W(t, e)}, {HI}",
                f"s_or_b64 vcc, vcc, {PM}",
                f"v_cmp_ne_u32_e64 {PM}, 0, {P(t, e)}",
                f"s_and_b64 vcc, vcc, {PM}",
                f"s_or_b64 {MASK}, {MASK}, vcc",
            ]
    lines = ["s_nop 4"] + [f"v_mov_b32 ${(n * k) + e}, 0" for e in range(k)]
    lines += [f"s_mov_b64 {SEXEC}, exec", f"s_mov_b32 {CNT}, 0"] + check
    lines += [f"s_and_b64 exec, exec, {MASK}", "s_cbranch_execz 2f", "1:"]
    lines += [f"global_load_dwordx2 {W(t, e)}, {A(t, e)}, off {cm}" for t in range(n) for e in range(k)]
    lines += ["s_waitcnt vmcnt(0)"] + check
    lines += [f"s_and_b64 exec, exec, {MASK}", "s_cbranch_execz 2f", "s_sleep 1",
              f"s_add_u32 {CNT}, {CNT}, 1", f"s_cmp_lt_u32 {CNT}, {limit}", "s_cbranch_scc1 1b"]
    lines += [f"v_mov_b32 ${(n * k) + e}, 1" for e in range(k)]
    lines += ["2:", f"s_mov_b64 exec, {SEXEC}", "s_nop 4"]
    return "\n".join(lines)


@gluon.constexpr_function
def _wait_constraints(n, k):
    outs = ["=&v"] * (n * k) + ["=&v"] * k
    ins = [str(i) for i in range(n * k)] + ["v"] * (n * k) + ["v"] * (n * k) + ["v"] * (2 * k)
    return ",".join(outs + ins + ["~{vcc}", "~{scc}", "~{memory}", _WS_CLOBBER])


@gluon.constexpr_function
def _wait_dtypes(n):
    return [gl.int64] * n + [gl.int32]


@gluon.jit
def wait_words(items, tag, PACK: gl.constexpr, SYS: gl.constexpr = False):
    """Every ``(ptr, idx, pred, peeked)`` item's tagged pairs landed -> (i32 words tuple,
    err). ``peeked``: the item's already-issued load (``_peek``'s words). Every item has
    the same shape and layout, PACK = its elements a thread."""
    N: gl.constexpr = len(items)
    vs = ()
    as_ = ()
    ps = ()
    for i in gl.static_range(N):
        ptr, idx, pred, peeked = items[i]
        p = (pred.to(gl.int32) != 0).to(gl.int32) + idx * 0
        vs = vs + (peeked + idx.to(gl.int64) * 0,)
        as_ = as_ + (addr_of(ptr, gl.where(p != 0, idx, 0)),)
        ps = ps + (p,)
    lo = gl.full(vs[0].shape, 0, gl.int64, vs[0].type.layout) + (tag.to(gl.int64) << 32)
    hi = lo + (1 << 32)
    r = gl.inline_asm_elementwise(
        _wait_asm(N, PACK, POLL_LIMIT, SYS or (DEV_SCOPE & 1) == 0), _wait_constraints(N, PACK),
        vs + as_ + ps + (lo, hi), dtype=_wait_dtypes(N), is_pure=False, pack=PACK)
    out = ()
    for i in gl.static_range(N):
        # a re-polling lane re-loads every word: the unchecked ones (already landed at the
        # peek, or unused) keep their peeked value (their re-load address is a dummy)
        out = out + (gl.where(ps[i] != 0, r[i], vs[i]).to(gl.int32),)
    return out, r[N]


@gluon.jit
def poll1(addr, tag, pred, SYS: gl.constexpr = False):
    """One tagged word a lane (``addr``: i64 byte address tensor, ``pred``: i32
    tensor, nonzero = poll) -> (value i32, err i32). Lanes with pred == 0 return
    garbage and err 0. SYS: a peer-written word (system scope)."""
    tag64 = gl.full(addr.shape, 0, gl.int64, addr.type.layout) + tag.to(gl.int64)
    r = gl.inline_asm_elementwise(
        _poll_asm(1, POLL_LIMIT, SYS or (DEV_SCOPE & 1) == 0), _poll_constraints(1), [pred.to(gl.int32), addr, tag64],
        dtype=_poll_dtypes(1), is_pure=False, pack=1)
    return r[0].to(gl.int32), r[1]


@gluon.jit
def poll2(a0, a1, tag, pred):
    tag64 = gl.full(a0.shape, 0, gl.int64, a0.type.layout) + tag.to(gl.int64)
    r = gl.inline_asm_elementwise(
        _poll_asm(2, POLL_LIMIT), _poll_constraints(2), [pred.to(gl.int32), a0, a1, tag64],
        dtype=_poll_dtypes(2), is_pure=False, pack=1)
    return r[0].to(gl.int32), r[1].to(gl.int32), r[2]


@gluon.jit
def poll4(a0, a1, a2, a3, tag, pred):
    tag64 = gl.full(a0.shape, 0, gl.int64, a0.type.layout) + tag.to(gl.int64)
    r = gl.inline_asm_elementwise(
        _poll_asm(4, POLL_LIMIT), _poll_constraints(4), [pred.to(gl.int32), a0, a1, a2, a3, tag64],
        dtype=_poll_dtypes(4), is_pure=False, pack=1)
    return r[0].to(gl.int32), r[1].to(gl.int32), r[2].to(gl.int32), r[3].to(gl.int32), r[4]


@gluon.jit
def _peek(ptr, idx, tag, pred):
    """One coherent load of the tagged pairs ``ptr[idx]`` (``ptr``: *i64) -> (words,
    i32 nonzero where pred is set but the tag is not there yet)."""
    p = pred.to(gl.int32) != 0
    v = cdna4.buffer_load(ptr, idx, mask=p, other=0, cache=".cv")
    return v, (p & ((v >> 32).to(gl.int32) != tag)).to(gl.int32)


# The pollv* forms take (region pointer, pair indices). They first issue every
# element's load at once (one round trip for all of a thread's elements) and only
# spin, in the asm above, on the elements whose tag had not landed yet: the asm runs
# element by element, so polling k elements a thread directly costs k serial round
# trips even when everything is already there.
@gluon.jit
def pollv1(ptr, idx, tag, pred):
    v, need = _peek(ptr, idx, tag, pred)
    r, err = poll1(addr_of(ptr, idx), tag, need)
    return gl.where(need != 0, r, v.to(gl.int32)), err


@gluon.jit
def pollv2(ptr, i0, i1, tag, pred):
    v0, n0 = _peek(ptr, i0, tag, pred)
    v1, n1 = _peek(ptr, i1, tag, pred)
    need = n0 | n1
    r0, r1, err = poll2(addr_of(ptr, i0), addr_of(ptr, i1), tag, need)
    sel = need != 0
    return gl.where(sel, r0, v0.to(gl.int32)), gl.where(sel, r1, v1.to(gl.int32)), err


@gluon.jit
def pollv4(ptr, i0, i1, i2, i3, tag, pred):
    v0, n0 = _peek(ptr, i0, tag, pred)
    v1, n1 = _peek(ptr, i1, tag, pred)
    v2, n2 = _peek(ptr, i2, tag, pred)
    v3, n3 = _peek(ptr, i3, tag, pred)
    need = n0 | n1 | n2 | n3
    r0, r1, r2, r3, err = poll4(addr_of(ptr, i0), addr_of(ptr, i1), addr_of(ptr, i2), addr_of(ptr, i3), tag, need)
    sel = need != 0
    return (gl.where(sel, r0, v0.to(gl.int32)), gl.where(sel, r1, v1.to(gl.int32)),
            gl.where(sel, r2, v2.to(gl.int32)), gl.where(sel, r3, v3.to(gl.int32)), err)


@gluon.jit
def pair_word(value, tag):
    """(tag << 32) | value bits, i64 (``value``: i32 or f32 tensor)."""
    if value.dtype == gl.float32:
        v = value.to(gl.int32, bitcast=True)
    else:
        v = value.to(gl.int32)
    lo = v.to(gl.uint32, bitcast=True).to(gl.int64)
    return (tag.to(gl.int64) << 32) | lo


@gluon.jit
def put_sys(region, idx, value, tag, mask=None):
    """Tagged pair ``idx`` of ``region`` (``*i64``) := (value, tag), system scope (a peer's)."""
    cdna4.buffer_store(pair_word(value, tag), region, idx, mask=mask, cache=".wt")


_PUT_DEV = gl.constexpr("v_cmp_ne_u32 vcc, 0, $2\ns_and_saveexec_b64 $0, vcc\ns_cbranch_execz 1f\n"
                        "global_store_dwordx2 $3, $1, off sc1\n1:\ns_or_b64 exec, exec, $0")


@gluon.jit
def put(region, idx, value, tag, mask=None):
    """Tagged pair ``idx`` of a scratch ``region`` (``*i64``) := (value, tag), device scope."""
    if DEV_SCOPE & 2:
        w = pair_word(value, tag)
        if mask is None:
            m = (w != 0) | (w == 0)
        else:
            m = mask
        m = (m & ((w != 0) | (w == 0))).to(gl.int32)
        a = (region + idx).to(gl.int64, bitcast=True) + (w * 0)
        gl.inline_asm_elementwise(_PUT_DEV, "=&s,v,v,v,~{vcc},~{memory}", [w, m, a], dtype=gl.int64, is_pure=False,
                                  pack=1)
    else:
        put_sys(region, idx, value, tag, mask)


@gluon.jit
def drain(x):
    """This thread's outstanding stores complete (``s_waitcnt vmcnt(0)``) then a CTA
    barrier: a flag published after it never overtakes the data. ``x``: any
    [512] T1 tensor (the asm's carrier)."""
    y = gl.inline_asm_elementwise("s_waitcnt vmcnt(0)", "=v,0,~{memory}", [x], dtype=x.dtype,
                                  is_pure=False, pack=1)
    gl.barrier()
    return y


@gluon.jit
def stamp(trace, bid, slot):
    """Trace word ``slot`` of CTA ``bid`` := now (``trace``: *i64), when TRACE."""
    if TRACE:
        if TRACE == 1:  # TRACE=2: no barrier (wave 0's own time: does not reorder the CTA)
            gl.barrier()
        tid = gl.arange(0, THREADS, layout=T1)
        now = gl.inline_asm_elementwise("s_memrealtime $0\ns_waitcnt lgkmcnt(0)", "=s,v", [tid], dtype=gl.int64,
                                        is_pure=False, pack=1)
        gl.store(trace + bid * TRACE_SLOTS + slot + tid * 0, now, mask=tid == 0)


@gluon.jit
def uniform(x):
    """A CTA-uniform i32 scalar (e.g. a reduction of an LDS read) moved to an SGPR
    with v_readfirstlane, so LLVM's divergence analysis knows it is uniform: a buffer
    resource built from a value it thinks divergent is wrapped in a waterfall loop
    (readfirstlane / compare / exec mask / load / loop) on every load."""
    # LLVM's hazard pass does not look into inline asm, so the asm carries its own wait
    # states: before it, a VALU write of x just ahead (v_cndmask ... then readfirstlane)
    # is otherwise read stale (seen: a select chain's previous value -- the wrong expert,
    # on some waves); after it, a VMEM reading the SGPR needs 5 (a wild pointer).
    return gl.inline_asm_elementwise("s_nop 4\nv_readfirstlane_b32 $0, $1\ns_nop 4", "=s,v", [x], dtype=gl.int32,
                                     is_pure=False, pack=1)


@gluon.jit
def uniform64(x):
    """``uniform`` for an i64 scalar (two v_readfirstlane)."""
    lo = uniform(x.to(gl.int32))
    hi = uniform((x >> 32).to(gl.int32))
    return (hi.to(gl.int64) << 32) | lo.to(gl.uint32, bitcast=True).to(gl.int64)


@gluon.jit
def pin(x):
    """A compiler fence (empty asm, memory clobber): loads issued before it stay
    before it. LLVM's scheduler otherwise sinks software-prefetched loads to just
    before their use, and the in-order vmcnt wait then leaves ~1 step in flight.
    ``x``: any tensor, returned (the asm's carrier)."""
    return gl.inline_asm_elementwise("; pin", "=v,0,~{memory}", [x], dtype=x.dtype, is_pure=False, pack=1)


@gluon.jit
def tids():
    return gl.arange(0, THREADS, layout=T1)


@gluon.jit
def first_task(bid, base):
    """CTA ``bid``'s first task of a stage placed from CTA ``base``, then every BLOCKS-th."""
    return (bid - base % BLOCKS + BLOCKS) % BLOCKS


@gluon.jit
def addr_of(ptr, idx):
    """Byte addresses (i64) of ``ptr + idx`` for the poll asm."""
    return (ptr + idx).to(gl.int64, bitcast=True)


@gluon.jit
def as_ptr(base, byte_off, ty: gl.constexpr):
    """``base`` (``*u8``) + ``byte_off`` as a ``*ty`` pointer."""
    return (base + byte_off).to(gl.pointer_type(ty), bitcast=True)


@gluon.jit
def bf16_round(x):
    return x.to(gl.bfloat16).to(gl.float32)


@gluon.jit
def code_ceil(v):
    """Biased exponent of the smallest power of two >= v (v >= 0, finite), clamped
    to [1, 254]: an E8M0 scale code (aiter's ``fused_mx_quant`` rule)."""
    bits = v.to(gl.int32, bitcast=True)
    e = ((bits >> 23) & 0xFF) + gl.where((bits & 0x7FFFFF) != 0, 1, 0)
    return gl.minimum(gl.maximum(e, 1), 254)


@gluon.jit
def rcp_pow2(code):
    """1 / 2^(code - 127), exact."""
    return ((254 - code) << 23).to(gl.float32, bitcast=True)


@gluon.jit
def vllm_mx_code(amax):
    """vLLM ``mxfp8_e4m3_quantize``'s scale code: smallest power of two >=
    RN(max(amax, FLT_MIN) / 448), clamped to [0, 254]."""
    a = gl.maximum(amax, 1.1754943508222875e-38)
    q = gl.div_rn(a, gl.full(a.shape, 448.0, gl.float32, a.type.layout))
    bits = q.to(gl.int32, bitcast=True)
    e = ((bits >> 23) & 0xFF) + gl.where((bits & 0x7FFFFF) != 0, 1, 0)
    return gl.minimum(gl.maximum(e, 0), 254)


@gluon.jit
def clamp_fp8(v):
    return gl.minimum(gl.maximum(v, -448.0), 448.0)


@gluon.jit
def hw_rcp(x):
    return gl.inline_asm_elementwise("v_rcp_f32 $0, $1", "=v,v", [x], dtype=gl.float32, is_pure=True, pack=1)


@gluon.jit
def hw_rsq(x):
    return gl.inline_asm_elementwise("v_rsq_f32 $0, $1", "=v,v", [x], dtype=gl.float32, is_pure=True, pack=1)


@gluon.jit
def hw_exp2(x):
    return gl.inline_asm_elementwise("v_exp_f32 $0, $1", "=v,v", [x], dtype=gl.float32, is_pure=True, pack=1)


@gluon.jit
def hw_log2(x):
    return gl.inline_asm_elementwise("v_log_f32 $0, $1", "=v,v", [x], dtype=gl.float32, is_pure=True, pack=1)


# ------------------------------------------------------------------ M3 additions
@gluon.jit
def bf_lo(w):
    """The low bf16 of an i32 pair word, as f32."""
    return (w << 16).to(gl.float32, bitcast=True)


@gluon.jit
def bf_hi(w):
    return (w & -65536).to(gl.float32, bitcast=True)


@gluon.jit
def bf_pair(lo, hi):
    """Two f32 -> one i32 word of (bf16(lo), bf16(hi))."""
    a = lo.to(gl.bfloat16).to(gl.int16, bitcast=True).to(gl.int32) & 0xFFFF
    b = hi.to(gl.bfloat16).to(gl.int16, bitcast=True).to(gl.int32) << 16
    return a | b


@gluon.jit
def note_err(errors, bid, err, code: gl.constexpr):
    """errors[bid] := code where a poll's watchdog fired (err != 0); FIRST_ERR (debug): only
    the CTA's first, as code + 64 * (the step's launch count, from ERR_TAG)."""
    if FIRST_ERR:
        val = gl.where(err != 0, code + gl.load(errors + 256 + err * 0) * 64, 0)
        gl.atomic_cas(errors + bid + err * 0, err * 0, val)
    else:
        gl.store(errors + bid + err * 0, gl.full(err.shape, code, gl.int32, err.type.layout), mask=err != 0)


@gluon.jit
def scale_index(row, col, cols: gl.constexpr):
    """Byte of E8M0 scale (row, col) in aiter's shuffle_scale layout:
    view(rows/32, 2, 16, cols/8, 2, 4).permute(0, 3, 5, 2, 4, 1)."""
    return (((((row // 32) * (cols // 8) + col // 8) * 4 + col % 4) * 16 + row % 16) * 2
            + (col // 4) % 2) * 2 + (row // 16) % 2


@gluon.jit
def pow2(code):
    return (code << 23).to(gl.float32, bitcast=True)


@gluon.jit
def route_key(v, e):
    """Signed-orderable i32 of score ``v`` with its low 7 bits replaced by 127 - e
    (ATOM's routing key: a pick is one integer max, a tie goes to the lower expert)."""
    b = v.to(gl.int32, bitcast=True)
    b = gl.where(b < 0, b ^ 0x7FFFFFFF, b)
    return (b & -128) | (127 - e)
