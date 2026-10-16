# SPDX-License-Identifier: MIT
"""The Gluon MiniMax-M3 decode step kernel (gfx950, TP4, S <= 4 rows, contexts <= 2048).

One persistent launch of 256 CTAs x 512 threads runs every sparse layer (3..59) of a
decode step; layer i's tasks hand off through tagged pairs (tag = epoch * 64 + i + 1)
in the parity-(i % 2) halves of the scratch and peer buffers. Per layer, task t of a
stage on its CTA, every CTA walking the stages in order:

    qkv    (160 / 144, CTAs 0 ..): 16 rows of q | k | v | index q | index k; the
           layer's input (ar + res: vLLM's tensors at layer 0, the previous layer's
           mailboxes after) -> gemma RMSNorm -> per-token FP8 -> FP8 GEMV; the weights
           are issued before the input wait. CTA k < S also publishes token k's h.
    ik     (CTA 160, selecting layers): index k -> norm / rope -> the bf16 index cache
    split  (8 S, top CTAs): one 256-key partition x 16 heads; q's per-head norm and
           RoPE made here from the GEMV rows (no head hop); K / V issued before q
           lands; the partition holding the token's own key makes K / V itself,
           inserts them in the cache and re-reads that key
    o      (192, CTAs 0 ..): every token's partitions merged -> per-token FP8 ->
           o_proj (32 rows, weights issued at the layer's start) -> bf16 partials
           to every rank -> rank-ordered sum -> a, h_mid
    router (16 S, CTAs 192 ..): gemma RMSNorm(a + h) -> bf16 gate GEMV (weights
           issued early) -> (routing key, sigmoid) x 8 copies; its xn slice as MXFP8
    moe    every CTA routes every token itself (top-4 + the fused shared expert);
           shared up / gate (48, CTAs 192 ..), routed up / gate (192 S, CTAs 0 ..),
           down (192, CTAs 0 ..): route-weighted MXFP4 x MXFP8 -> bf16 partials to
           every rank -> rank-ordered sum -> ar (the next layer's input)

Numerics follow ATOM's FlyDSL M3 mono (ported in ``..kernels``) stage by stage; the
reductions differ in split and order only.
"""

import os

import triton
from triton.experimental import gluon
import triton.experimental.gluon.language as gl
from triton.experimental.gluon.language.amd import cdna4

from . import layout as LY
from .prims import (
    _peek, addr_of, poll1, put_sys, wait_words, BLOCKS, THREADS, as_ptr, bf_hi, bf_lo, bf_pair, bf16_round, clamp_fp8, code_ceil, drain, hw_exp2, hw_rcp,
    hw_rsq, note_err, pin, pollv1, put, rcp_pow2, route_key, scale_index, stamp, tids, uniform, uniform64,
)

HIDDEN = gl.constexpr(LY.HIDDEN)
HPAIRS = gl.constexpr(LY.HIDDEN // 2)
HEAD_DIM = gl.constexpr(LY.HEAD_DIM)
HEADS = gl.constexpr(LY.HEADS)
QKV_PAIRS = gl.constexpr(LY.QKV_ROWS // 2)
O_K = gl.constexpr(LY.O_K)
INTER = gl.constexpr(LY.INTER)
N_SPLIT = gl.constexpr(LY.N_SPLIT)
SPLIT_KEYS = gl.constexpr(LY.SPLIT_KEYS)
MAX_S = gl.constexpr(LY.MAX_S)
N_ROUTED = gl.constexpr(LY.N_ROUTED)
TOP_K = gl.constexpr(LY.TOP_K)
SLOTS = gl.constexpr(LY.SLOTS)
SCORE_COPIES = gl.constexpr(LY.SCORE_COPIES)
TAGS = gl.constexpr(LY.TAGS_PER_STEP)
EP_MARK_PAIRS = gl.constexpr(LY.EP_MARKS // 2)
LOG2E = gl.constexpr(1.4426950408889634)
NEG = gl.constexpr(-3.4e38)
FP8_MAX = gl.constexpr(448.0)
NPTR = gl.constexpr(20)  # pointer-table entries a layer (runner.LAYER_PTRS)

# error codes (errors[cta]): which wait's watchdog fired
E_X, E_QKV, E_SPLIT, E_ATTN, E_H, E_A, E_SC, E_XN, E_MID, E_FFN, E_EPOCH = (gl.constexpr(i) for i in range(1, 12))
POLL_ROUNDS = gl.constexpr(1 << 20)  # wait_group's watchdog (rounds of ~1 us)
DN_STREAM = gl.constexpr(int(os.environ.get("VLLM_M3_GLUON_DN_STREAM", "0")))  # down rounds in the ug tail
DN_DEPTH = gl.constexpr(int(os.environ.get("VLLM_M3_GLUON_DN_DEPTH", "8")))  # down weight rounds prefetched

# ---------------------------------------------------------------- layouts
# GEMV tiles: 8 waves stacked along M (16 rows each), N = 16 P columns
MFMA = gl.constexpr(cdna4.AMDMFMALayout(version=4, instr_shape=[16, 16, 128], transposed=True, warps_per_cta=[8, 1]))
A_L = gl.constexpr(gl.DotOperandLayout(operand_index=0, parent=MFMA, k_width=16))
B_L = gl.constexpr(gl.DotOperandLayout(operand_index=1, parent=MFMA, k_width=16))
# attention: 8 waves side by side along N (keys for QK, dims for PV)
MFMA2 = gl.constexpr(cdna4.AMDMFMALayout(version=4, instr_shape=[16, 16, 128], transposed=True, warps_per_cta=[1, 8]))
A2 = gl.constexpr(gl.DotOperandLayout(operand_index=0, parent=MFMA2, k_width=16))
B2 = gl.constexpr(gl.DotOperandLayout(operand_index=1, parent=MFMA2, k_width=16))
X_SH3 = gl.constexpr(gl.SwizzledSharedLayout(vec=16, per_phase=1, max_phase=16, order=[2, 1, 0]))
FLAT4 = gl.constexpr(gl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[3, 2, 1, 0]))
FLAT1 = gl.constexpr(gl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[0]))
FLAT2 = gl.constexpr(gl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[1, 0]))
FLAT3 = gl.constexpr(gl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[2, 1, 0]))
# GEMV B tiles [rounds, columns, 128 B] (column = token * P + k-slice p): 16 B chunks XOR-swizzled
# by row, so a B_L read (lane = column, 16 B of k) spreads over the banks (flat: 8-way conflicts)
BSW = gl.constexpr(gl.SwizzledSharedLayout(vec=16, per_phase=1, max_phase=4, order=[2, 1, 0]))
BSW2 = gl.constexpr(gl.SwizzledSharedLayout(vec=16, per_phase=1, max_phase=4, order=[1, 0]))
# an FP8 weight tile [128 rows, 8 chunks, 16 B] of aiter's (16, 16) shuffle: lane = chunk %
# 4 * 16 + row % 16 holds 16 B chunks c and c + 4 of a 128-k step -- A_L once reshaped
W8_L = gl.constexpr(gl.DistributedLinearLayout(
    reg_bases=[[0, 0, 1], [0, 0, 2], [0, 0, 4], [0, 0, 8], [0, 4, 0]],
    lane_bases=[[1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0], [0, 1, 0], [0, 2, 0]],
    warp_bases=[[16, 0, 0], [32, 0, 0], [64, 0, 0]], block_bases=[], shape=[128, 8, 16]))
# an MXFP4 weight tile [128 rows, 4 chunks, 16 B] (one 128-k step): A_L once reshaped
W4_L = gl.constexpr(gl.DistributedLinearLayout(
    reg_bases=[[0, 0, 1], [0, 0, 2], [0, 0, 4], [0, 0, 8]],
    lane_bases=[[1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0], [0, 1, 0], [0, 2, 0]],
    warp_bases=[[16, 0, 0], [32, 0, 0], [64, 0, 0]], block_bases=[], shape=[128, 4, 16]))
# K^T [8 d-chunks, 16 B, 256 keys] of a partition (B2 once reshaped): a lane's 16 d of
# one key are 16 contiguous bytes of a page-16 K page [d / 16][t][16]
KT_L = gl.constexpr(gl.DistributedLinearLayout(
    reg_bases=[[0, 1, 0], [0, 2, 0], [0, 4, 0], [0, 8, 0], [4, 0, 0], [0, 0, 128]],
    lane_bases=[[0, 0, 1], [0, 0, 2], [0, 0, 4], [0, 0, 8], [1, 0, 0], [2, 0, 0]],
    warp_bases=[[0, 0, 16], [0, 0, 32], [0, 0, 64]], block_bases=[], shape=[8, 16, 256]))
# V [16 pages, 16 t, 128 d] (B2 once reshaped): a lane's 16 keys of one dim are 16
# contiguous bytes of a page-16 V page [d][t]
VT_L = gl.constexpr(gl.DistributedLinearLayout(
    reg_bases=[[0, 1, 0], [0, 2, 0], [0, 4, 0], [0, 8, 0], [4, 0, 0], [8, 0, 0]],
    lane_bases=[[0, 0, 1], [0, 0, 2], [0, 0, 4], [0, 0, 8], [1, 0, 0], [2, 0, 0]],
    warp_bases=[[0, 0, 16], [0, 0, 32], [0, 0, 64]], block_bases=[], shape=[16, 16, 128]))
# a [16 rows, 16 tokens] epilogue tile, and the [8 pairs, 16 tokens] one
EP_L = gl.constexpr(gl.BlockedLayout([1, 1], [4, 16], [4, 2], [1, 0]))
PR_L = gl.constexpr(gl.BlockedLayout([1, 1], [8, 8], [1, 8], [0, 1]))


# scratch regions (byte offsets in a half)
R_QKV = gl.constexpr(LY.region("qkv"))
R_SPO = gl.constexpr(LY.region("spo"))
R_SPM = gl.constexpr(LY.region("spm"))
R_SPL = gl.constexpr(LY.region("spl"))
R_A = gl.constexpr(LY.region("a"))
R_H = gl.constexpr(LY.region("h"))
R_HM = gl.constexpr(LY.region("hm"))
R_SC = gl.constexpr(LY.region("sc"))
R_XN8 = gl.constexpr(LY.region("xn8"))
R_XSC = gl.constexpr(LY.region("xsc"))
R_MID = gl.constexpr(LY.region("mid"))
R_OWN = gl.constexpr(LY.region("own"))
R_AR = gl.constexpr(LY.region("ar"))
ROWS16 = gl.constexpr(LY.ROWS16)
NQ_REUSE = gl.constexpr(LY.NQ_REUSE)
N_O = gl.constexpr(LY.N_O)
N_DN = gl.constexpr(LY.N_DN)
UG_TASKS = gl.constexpr(LY.UG_TASKS)
ROUTER_BASE = gl.constexpr(LY.ROUTER_BASE)
SHARED_BASE = gl.constexpr(LY.SHARED_BASE)


# ---------------------------------------------------------------- small helpers
@gluon.jit
def opq():
    """An opaque 0 (a side-effecting s_mov), asserted a multiple of 2^16: added to every
    index tensor so that LLVM cannot hoist a stage's index math out of the layer loop
    (it would hold every stage's indices live across all of them: hundreds of spills)."""
    return gl.multiple_of(gl.inline_asm_elementwise("s_mov_b32 $0, 0", "=s", [], dtype=gl.int32, is_pure=False,
                                                    pack=1), 65536)


# the multi-layer launch (VLLM_M3_GLUON_LOOP=1) needs loop-variant index tensors; a launch a
# layer has no loop to hoist out of, and an opaque arange only blocks constant folding
# (every element's index math in VALU, recomputed in every unrolled round)
DBG_NOTAIL = gl.constexpr(int(os.environ.get("VLLM_M3_GLUON_DBG_NOTAIL", "0")))  # diagnostics: no tail split work
OPQ_AR = gl.constexpr(os.environ.get("VLLM_M3_GLUON_LOOP", "0") == "1")


@gluon.jit
def ar(n: gl.constexpr, layout: gl.constexpr):
    """``gl.arange(0, n, layout)``, loop-variant (``opq``) in the multi-layer launch."""
    if OPQ_AR:
        return gl.arange(0, n, layout=layout) + opq()
    else:
        return gl.arange(0, n, layout=layout)


@gluon.jit
def zacc(shape: gl.constexpr, layout: gl.constexpr):
    """Zeros for an MFMA accumulator, opaque (``opq``): with an inline-constant 0 as src C,
    LLVM gives v_mfma_scale a destination partially overlapping its A / scale sources and
    reuses those registers right after it -- the late result write then clobbers them
    (seen as memory faults at random addresses)."""
    return gl.full(shape, 0.0, gl.float32, layout) + opq().to(gl.float32)


LROW_ASM = gl.constexpr(int(__import__("os").environ.get("VLLM_M3_GLUON_LROW_ASM", "1")))


@gluon.constexpr_function
def _row_asm(js):
    n = len(js)
    return "\n".join([f"s_load_dwordx2 ${k}, ${n}, {8 * j}" for k, j in enumerate(js)] + ["s_waitcnt lgkmcnt(0)"])


@gluon.constexpr_function
def _row_constraints(js):
    return ",".join(["=s"] * len(js) + ["s"])


@gluon.constexpr_function
def _row_dtypes(js):
    return (gl.int64,) * len(js)


@gluon.jit
def lrow(ptab, li, js: gl.constexpr):
    """Pointer-table entries js of layer li as CTA-uniform i64 scalars: one asm block of
    scalar loads and one wait (an entry at a time, each is a serial vector round trip)."""
    if LROW_ASM:
        base = (ptab + li * NPTR).to(gl.int64, bitcast=True)
        return gl.inline_asm_elementwise(_row_asm(js), _row_constraints(js), [base], dtype=_row_dtypes(js),
                                         is_pure=False, pack=1)
    else:
        out = ()
        for j in gl.static_range(len(js)):
            out = out + (uniform64(gl.load(ptab + li * NPTR + js[j])),)
        return out


@gluon.jit
def lp(v, ty: gl.constexpr):
    """A table entry as a ``*ty`` (every table pointer is a torch allocation's or a 16 B
    aligned view's: told to the alignment analysis, else every buffer load of it goes out
    byte by byte)."""
    return gl.multiple_of(v.to(gl.pointer_type(ty), bitcast=True), 16)


@gluon.jit
def rgn(base, off: gl.constexpr):
    """A pair region of a scratch half (``base``: *u8; ``off``: an R_* offset)."""
    return as_ptr(base, off, gl.int64)


@gluon.jit
def i32p(p):
    return p.to(gl.pointer_type(gl.int32), bitcast=True)


@gluon.jit
def u8p(p):
    return p.to(gl.pointer_type(gl.uint8), bitcast=True)


@gluon.jit
def w8_rows():
    return ar(128, layout=gl.SliceLayout(1, gl.SliceLayout(2, W8_L)))


@gluon.jit
def w8_tile(base, row_term):
    """An FP8 A tile [128, 128] (A_L) of a (16, 16)-shuffled weight: virtual row m's
    16-row group and step folded into ``row_term[m]`` (bytes, + (m % 16) * 16)."""
    c = ar(8, layout=gl.SliceLayout(0, gl.SliceLayout(2, W8_L)))[None, :, None]
    b = ar(16, layout=gl.SliceLayout(0, gl.SliceLayout(1, W8_L)))[None, None, :]
    v = cdna4.buffer_load(base, row_term[:, None, None] + c * 256 + b)
    return gl.convert_layout(gl.reshape(v, [128, 128]), A_L)


@gluon.jit
def w4_rows():
    return ar(128, layout=gl.SliceLayout(1, gl.SliceLayout(2, W4_L)))


@gluon.jit
def w4_off(HALF_OFF: gl.constexpr, KMAX: gl.constexpr = 4):
    """The lane's byte offsets of an MXFP4 A tile [128, 64 B] of aiter's shuffled layout (one
    128-k step), relative to its scalar base (``w4_tile``): row m's group wv = m / 16 is
    (half wv / 4, k step wv % 4) -> (wv / 4) HALF_OFF + (wv % 4) 1024 + (m % 16) 16 + 256 c
    + byte. Steps wv % 4 >= KMAX re-read step KMAX - 1 (their B rows are zero, FP4 has no
    NaN): no masks. Computed once a stage (``ar`` is opaque: no CSE across rounds)."""
    m = w4_rows()[:, None, None]
    c = ar(4, layout=gl.SliceLayout(0, gl.SliceLayout(2, W4_L)))[None, :, None]
    b = ar(16, layout=gl.SliceLayout(0, gl.SliceLayout(1, W4_L)))[None, None, :]
    wv = m >> 4
    ks = wv & 3
    if KMAX < 4:
        ks = gl.minimum(ks, KMAX - 1)
    return (wv >> 2) * HALF_OFF + ks * 1024 + (m & 15) * 16 + c * 256 + b


@gluon.jit
def w4_tile(base, off, pred=None):
    """The A tile at scalar ``base`` (every per-round term folded into it, as FlyDSL's
    buffer offsets), lane offsets ``off`` (``w4_off``); ``pred`` (debug only) zeroes."""
    if pred is None:
        v = cdna4.buffer_load(base, off)
    else:
        v = cdna4.buffer_load(base, off, mask=(off >= 0) & pred, other=0)
    return gl.convert_layout(gl.reshape(v, [128, 64]), A_L)


@gluon.jit
def ug_offs():
    """ug_round's loop-invariant lane offsets: (A tile, scale byte). aiter's shuffle_scale
    byte of (row e 1536 + 16 g + r16, col (4 r + p) 4 + sg), 192 cols, g = 48 (sm / 64) + cg:
    (e 48 + cg / 2) 6144 + r 512 + cg % 2 (scalar) + (sm / 64) 24 6144 + (p / 2) 256 + (p % 2)
    2 + sg 64 + r16 4 (lane), p = (sm / 16) % 4."""
    AS_L: gl.constexpr = cdna4.get_mfma_scale_layout(A_L, [128, 4])
    sm = ar(128, layout=gl.SliceLayout(1, AS_L))[:, None]
    sg = ar(4, layout=gl.SliceLayout(0, AS_L))[None, :]
    sp = (sm >> 4) & 3
    sb = (sm >> 6) * (24 * 6144) + (sp >> 1) * 256 + (sp & 1) * 2 + sg * 64 + (sm & 15) * 4
    return w4_off(INTER * (HIDDEN // 2)), sb


@gluon.jit
def dn_offs():
    """down_round's loop-invariant lane offsets: (A tile half 0, half 1, scale half 0, half 1).
    aiter's shuffle_scale byte of (row e 6144 + 32 t + 16 tt + r16, col c 4 + sg), 24 cols,
    c = 4 half + p: (e 192 + t) 768 + half 512 (scalar) + (p / 2) 256 + (p % 2) 2 + sg 64 +
    r16 4 + tt (lane), p = (sm / 16) % 4, tt = sm / 64; chunks past 6 read chunk 5's (finite)
    scale, their B is zero."""
    AS_L: gl.constexpr = cdna4.get_mfma_scale_layout(A_L, [128, 4])
    sm = ar(128, layout=gl.SliceLayout(1, AS_L))[:, None]
    sg = ar(4, layout=gl.SliceLayout(0, AS_L))[None, :]
    rest = sg * 64 + (sm & 15) * 4 + (sm >> 6)
    p0 = (sm >> 4) & 3
    p1 = gl.minimum(p0, 1)
    return (w4_off(16 * (INTER // 2), 4), w4_off(16 * (INTER // 2), 2),
            (p0 >> 1) * 256 + (p0 & 1) * 2 + rest, (p1 >> 1) * 256 + (p1 & 1) * 2 + rest)


@gluon.jit
def a_scales(ws, srow, col, cols: gl.constexpr):
    """A's E8M0 codes [128, 4] (one 128-k step) from an aiter shuffle_scale tensor."""
    b = scale_index(srow, col, cols)
    dw = cdna4.buffer_load(i32p(ws), b >> 2)
    return ((dw >> ((b & 3) * 8)) & 0xFF).to(gl.uint8)


@gluon.jit
def rnd(buf, r):
    """Round r ([columns, bytes]) of a 3-D B tile buffer: ``index`` takes a layout one rank
    below the shape (slicing, a full-rank one), so the same bytes seen with the rank-2
    form of its layout first."""
    if buf.type.layout == BSW:
        v = buf._reinterpret(buf.dtype, buf.shape, BSW2)
    else:
        v = buf._reinterpret(buf.dtype, buf.shape, FLAT2)
    return v.index(r)


@gluon.jit
def unit_scales(shape: gl.constexpr, layout: gl.constexpr):
    return gl.full(shape, 127, gl.uint8, cdna4.get_mfma_scale_layout(layout, shape))


@gluon.jit
def bd_reduce(acc, T: gl.constexpr, P: gl.constexpr, sm):
    """A block-diagonal accumulator [128, P MAX_S] (column = token P + p; wave w = t P + p
    owns rows 16 w .. and only its columns . P + p are valid) -> (lo, hi) [8 row pairs,
    MAX_S tokens] a tile (PR_L): rows 2 j, 2 j + 1 of tile t, the K slices summed in order
    p = 0 .. P - 1. Each wave's valid columns are picked in registers first (adding exact
    zeros), so only [128, MAX_S] goes through LDS (``sm``: LDS [128, MAX_S] f32)."""
    NC: gl.constexpr = P * MAX_S
    c = ar(NC, layout=gl.SliceLayout(0, MFMA))[None, :]
    m = ar(128, layout=gl.SliceLayout(1, MFMA))[:, None]
    mine = gl.where(c % P == (m // 16) % P, acc, 0.0)
    red = gl.sum(gl.reshape(mine, [128, MAX_S, P]), axis=2)
    gl.barrier()
    sm.store(red)
    gl.barrier()
    j = ar(8, layout=gl.SliceLayout(1, PR_L))[:, None]
    n = ar(MAX_S, layout=gl.SliceLayout(0, PR_L))[None, :]
    outs_lo = ()
    outs_hi = ()
    for t in gl.static_range(T):
        lo = gl.zeros([8, MAX_S], gl.float32, PR_L)
        hi = gl.zeros([8, MAX_S], gl.float32, PR_L)
        for p in gl.static_range(P):
            lo = lo + sm.gather((t * P + p) * 16 + 2 * j + n * 0, 0)
            hi = hi + sm.gather((t * P + p) * 16 + 2 * j + 1 + n * 0, 0)
        outs_lo = outs_lo + (lo,)
        outs_hi = outs_hi + (hi,)
    return outs_lo, outs_hi


@gluon.jit
def poll_pairs(ptr, idx, tag, pred, errors, bid, code: gl.constexpr):
    """Tagged pairs ``ptr[idx]`` (bf16 pairs) -> (lo, hi) f32; a watchdog error is noted."""
    v, err = pollv1(ptr, idx, tag, pred)
    note_err(errors, bid, err, code)
    return bf_lo(v), bf_hi(v)


@gluon.jit
def peek(ptr, idx, tag, pred):
    """A tagged-pair load issued (``land`` finishes it): issue every peek of a wait group
    before landing any -- each land is an asm memory barrier, so a peek after one is a
    further serial round trip (~0.5-1 us through device-scope memory)."""
    return _peek(ptr, idx, tag, pred)


@gluon.jit
def land(ptr, idx, tag, pk, errors, bid, code: gl.constexpr, SYS: gl.constexpr = False):
    """A ``peek``'s words, polling the elements whose tag had not landed -> i32 values
    (SYS: a peer-written region)."""
    v, need = pk
    r, err = poll1(addr_of(ptr, idx), tag, need, SYS)
    note_err(errors, bid, err, code)
    return gl.where(need != 0, r, v.to(gl.int32))


@gluon.jit
def wait_items(items, tag, errors, bid, code: gl.constexpr, PACK: gl.constexpr, SYS: gl.constexpr = False):
    """``(ptr, idx, pred, pk)`` items (pk: a ``peek``) landed -> i32 words (a tuple), in one
    asm block (FlyDSL's batched wait: per lane, no CTA barrier). Every item has the same
    shape and layout, PACK = its elements a thread."""
    its = ()
    for i in gl.static_range(len(items)):
        ptr, idx, pred, pk = items[i]
        its = its + ((ptr, idx, pred, pk[0]),)
    ws, err = wait_words(its, tag, PACK, SYS)
    note_err(errors, bid, err, code)
    return ws


@gluon.jit
def sleep1():
    return gl.inline_asm_elementwise("s_sleep 1", "=s", [], dtype=gl.int32, is_pure=False, pack=1)


@gluon.jit
def any_need(need):
    """1 if any element of the CTA's ``need`` is set (a CTA-wide reduction)."""
    if len(need.shape) == 1:
        r = gl.max(need, axis=0)
    elif len(need.shape) == 2:
        r = gl.max(gl.max(need, axis=1), axis=0)
    else:
        r = gl.max(gl.max(gl.max(need, axis=2), axis=1), axis=0)
    return r


@gluon.jit
def wait_group(items, tag, errors, bid, code: gl.constexpr, PACK: gl.constexpr):
    """Every ``(ptr, idx, peek)`` item landed -> their i32 words (a tuple), in one asm block
    (``wait_items``): the elements a peek already saw landed are not checked again. Same
    shape and layout each, PACK = its elements a thread."""
    its = ()
    for i in gl.static_range(len(items)):
        ptr, idx, pk = items[i]
        its = its + ((ptr, idx, pk[1], pk),)
    return wait_items(its, tag, errors, bid, code, PACK)


@gluon.jit
def layer_in_pairs(li, idx, pred, hid_in, res_in, psc, ptag, errors, bid, PACK: gl.constexpr):
    """The layer's input at pair index ``idx`` (token * HPAIRS + pair): (ar, res) as
    (lo, hi) f32 each -- vLLM's tensors at the first layer, the previous layer's ar /
    hm mailboxes (``psc``: its scratch half, tag ``ptag``) after."""
    if li == 0:
        aw = cdna4.buffer_load(i32p(hid_in), idx, mask=pred, other=0)
        rw = cdna4.buffer_load(i32p(res_in), idx, mask=pred, other=0)
        a_lo = bf_lo(aw)
        a_hi = bf_hi(aw)
        r_lo = bf_lo(rw)
        r_hi = bf_hi(rw)
    else:
        ws = wait_group(((rgn(psc, R_AR), idx, peek(rgn(psc, R_AR), idx, ptag, pred)),
                         (rgn(psc, R_HM), idx, peek(rgn(psc, R_HM), idx, ptag, pred))), ptag, errors, bid, E_X, PACK)
        a_lo = bf_lo(ws[0])
        a_hi = bf_hi(ws[0])
        r_lo = bf_lo(ws[1])
        r_hi = bf_hi(ws[1])
    return a_lo, a_hi, r_lo, r_hi


# ============================================================ qkv GEMV (+ input norm)
# x tiles [S, steps, 64 pairs] (pair (k, st, q) = elements st * 128 + 2 q, + 1), the 48
# 128-k steps of a row as a 32-step part and a 16-step part (shapes are powers of two)
XL3 = gl.constexpr(gl.BlockedLayout([1, 1, 2], [1, 2, 32], [1, 8, 1], [2, 1, 0]))


@gluon.jit
def qkv_in(li, st0: gl.constexpr, NST: gl.constexpr, SP: gl.constexpr, S: gl.constexpr, hid_in, res_in, psc,
           ptag, errors, bid, g_in):
    """Steps st0 .. st0 + NST of every token's input row, loaded and polled at once: (x =
    ar + res lo / hi, gamma pairs, pair index, live mask)."""
    k3 = ar(SP, layout=gl.SliceLayout(1, gl.SliceLayout(2, XL3)))[:, None, None]
    st3 = st0 + ar(NST, layout=gl.SliceLayout(0, gl.SliceLayout(2, XL3)))[None, :, None]
    q3 = ar(64, layout=gl.SliceLayout(0, gl.SliceLayout(1, XL3)))[None, None, :]
    ok = (k3 < S) & (st3 < 48)
    pidx = k3 * HPAIRS + st3 * 64 + q3
    a_lo, a_hi, r_lo, r_hi = layer_in_pairs(li, pidx, ok, hid_in, res_in, psc, ptag, errors, bid, SP * (NST // 16))
    gw = cdna4.buffer_load(g_in, st3 * 64 + q3 + k3 * 0, mask=ok, other=0)
    return a_lo + r_lo, a_hi + r_hi, gw, pidx, ok


@gluon.jit
def qkv_in_issue(li, st0: gl.constexpr, NST: gl.constexpr, SP: gl.constexpr, S: gl.constexpr, hid_in, res_in,
                 psc, ptag, g_in):
    """Steps st0 .. st0 + NST of every token's input row, loads issued only: (ar words,
    res words, their not-yet-landed masks, gamma pairs, pair index, live mask). vLLM's
    tensors at the first layer, else one coherent load of the previous layer's ar / hm
    pairs (``qkv_in_finish`` polls whatever had not landed)."""
    k3 = ar(SP, layout=gl.SliceLayout(1, gl.SliceLayout(2, XL3)))[:, None, None]
    st3 = st0 + ar(NST, layout=gl.SliceLayout(0, gl.SliceLayout(2, XL3)))[None, :, None]
    q3 = ar(64, layout=gl.SliceLayout(0, gl.SliceLayout(1, XL3)))[None, None, :]
    ok = (k3 < S) & (st3 < 48)
    pidx = k3 * HPAIRS + st3 * 64 + q3
    # both sources issued, predicated (a branch around the loads would wait for them
    # inside it, before the weights go out)
    first = li == 0
    hw = cdna4.buffer_load(i32p(hid_in), pidx, mask=ok & first, other=0)
    hr = cdna4.buffer_load(i32p(res_in), pidx, mask=ok & first, other=0)
    av = cdna4.buffer_load(rgn(psc, R_AR), pidx, mask=ok & (li != 0), other=0, cache=".cv")
    rv = cdna4.buffer_load(rgn(psc, R_HM), pidx, mask=ok & (li != 0), other=0, cache=".cv")
    gw = cdna4.buffer_load(g_in, st3 * 64 + q3 + k3 * 0, mask=ok, other=0)
    return (hw, hr), (av, rv), gw, pidx, ok


@gluon.jit
def qkv_in_finish(li, hin, mail, ok, pidx, psc, ptag, errors, bid, PACK: gl.constexpr):
    """x = ar + res (lo, hi): vLLM's tensors at the first layer, else the mailbox pairs
    (any that had not landed polled)."""
    hw, hr = hin
    av, rv = mail
    if li == 0:
        aw = hw
        rw = hr
    else:
        an = (ok & ((av >> 32).to(gl.int32) != ptag)).to(gl.int32)
        rn = (ok & ((rv >> 32).to(gl.int32) != ptag)).to(gl.int32)
        ws = wait_group(((rgn(psc, R_AR), pidx, (av, an)), (rgn(psc, R_HM), pidx, (rv, rn))), ptag, errors, bid, E_X,
                        PACK)
        aw = ws[0]
        rw = ws[1]
    return bf_lo(aw) + bf_lo(rw), bf_hi(aw) + bf_hi(rw)


@gluon.jit
def sum3(x):
    return gl.sum(gl.sum(x, axis=2), axis=1)


@gluon.jit
def max3(x):
    return gl.max(gl.max(x, axis=2), axis=1)


@gluon.jit
def qkv_quant(x_lo, x_hi, gw, rstd, x_scale, xs, NST: gl.constexpr, SP: gl.constexpr):
    # xs: LDS [NST / 8 rounds, 8 MAX_S columns (token * 8 + step), 128]
    """A part's normed rows -> FP8 (x / x_scale, correctly rounded) -> its LDS buffer."""
    r3 = rstd[:, None, None]
    n_lo = x_lo * r3 * (bf_lo(gw) + 1.0)
    n_hi = x_hi * r3 * (bf_hi(gw) + 1.0)
    xs3 = x_scale[:, None, None]
    q_lo = gl.div_rn(n_lo, xs3 + n_lo * 0).to(gl.float8e4nv).to(gl.uint8, bitcast=True)
    q_hi = gl.div_rn(n_hi, xs3 + n_hi * 0).to(gl.float8e4nv).to(gl.uint8, bitcast=True)
    x8 = gl.permute(gl.reshape(gl.join(q_lo, q_hi), [SP, NST // 8, 8, 128]), (1, 0, 2, 3))
    xs.slice(0, SP * 8, dim=1).store(gl.reshape(x8, [NST // 8, SP * 8, 128]))


@gluon.jit
def qkv_fold(pa, pb, MAX: gl.constexpr):
    """[SP, 32 steps] and [SP, 16 steps] per-token partials (steps over 2 lanes x 8 warps: the
    32-step part's halves sit in one thread) -> [SP]: the halves and the parts combined in
    registers first, then one cross-warp reduction (FlyDSL's single set of barriers)."""
    SP: gl.constexpr = pa.shape[0]
    a2 = gl.reshape(pa, [SP, 2, 16])
    if MAX:
        t = gl.max(a2, axis=1)
        t = gl.maximum(t, gl.convert_layout(pb, t.type.layout))
        r = gl.max(t, axis=1)
    else:
        t = gl.sum(a2, axis=1)
        t = t + gl.convert_layout(pb, t.type.layout)
        r = gl.sum(t, axis=1)
    # back in the parts' token layout (every thread holds every token's total: no data moves)
    return gl.convert_layout(r, gl.SliceLayout(1, pb.type.layout))


@gluon.jit
def qkv_absn(x_lo, x_hi, gw, rstd):
    """|normed x| of a part, its pair halves' max: [SP, steps] after the lanes' max."""
    r3 = rstd[:, None, None]
    n_lo = x_lo * r3 * (bf_lo(gw) + 1.0)
    n_hi = x_hi * r3 * (bf_hi(gw) + 1.0)
    return gl.max(gl.maximum(gl.abs(n_lo), gl.abs(n_hi)), axis=2)


@gluon.jit
def qkv_amax(x_lo, x_hi, gw, rstd):
    r3 = rstd[:, None, None]
    n_lo = x_lo * r3 * (bf_lo(gw) + 1.0)
    n_hi = x_hi * r3 * (bf_hi(gw) + 1.0)
    return max3(gl.maximum(gl.abs(n_lo), gl.abs(n_hi)))


@gluon.jit
def stage_qkv(t, li, ptab, hid_in, res_in, sc, psc, tag, ptag, errors, bid, eps, S: gl.constexpr, tr_=None):
    """GEMV task t: rows 16 t .. + 16 of the fused projection for every token."""
    e = lrow(ptab, li, (0, 1, 2))
    w_qkv = lp(e[0], gl.uint8)
    s_qkv = lp(e[1], gl.float32)
    g_in = lp(e[2], gl.int32)
    SP: gl.constexpr = triton.next_power_of_2(S)
    # the input's loads first: they return in order, so issued after the weights they
    # would wait for the whole weight stream; the norm / quant then overlap it (S > 2:
    # the input words and the weights in flight together do not fit the registers)
    EARLY: gl.constexpr = S <= 2
    if EARLY:
        hin_a, mail_a, ga, pa, oka = qkv_in_issue(li, 0, 32, SP, S, hid_in, res_in, psc, ptag, g_in)
        hin_b, mail_b, gb, pb, okb = qkv_in_issue(li, 32, 16, SP, S, hid_in, res_in, psc, ptag, g_in)
        pin(tids())
    # the weights: round r, wave w takes 128-k step 8 r + w of the task's 16 rows
    m = w8_rows()
    aq = ()
    for r in gl.static_range(6):
        aq = aq + (w8_tile(w_qkv, t * (16 * 6144) + (8 * r + m // 16) * 2048 + (m % 16) * 16),)
    rr = ar(8, layout=gl.SliceLayout(1, PR_L))[:, None]
    ws_lo = gl.load(s_qkv + 16 * t + 2 * rr)
    ws_hi = gl.load(s_qkv + 16 * t + 2 * rr + 1)
    m = pin(m)
    if EARLY:
        xa_lo, xa_hi = qkv_in_finish(li, hin_a, mail_a, oka, pa, psc, ptag, errors, bid, SP * 4)
        xb_lo, xb_hi = qkv_in_finish(li, hin_b, mail_b, okb, pb, psc, ptag, errors, bid, SP * 2)
    else:
        xa_lo, xa_hi, ga, pa, oka = qkv_in(li, 0, 32, SP, S, hid_in, res_in, psc, ptag, errors, bid, g_in)
        xb_lo, xb_hi, gb, pb, okb = qkv_in(li, 32, 16, SP, S, hid_in, res_in, psc, ptag, errors, bid, g_in)
    stamp(tr_, bid, 20)
    ss = qkv_fold(gl.sum(xa_lo * xa_lo + xa_hi * xa_hi, axis=2), gl.sum(xb_lo * xb_lo + xb_hi * xb_hi, axis=2), False)
    rstd = hw_rsq(ss / 6144.0 + eps)
    amax = qkv_fold(qkv_absn(xa_lo, xa_hi, ga, rstd), qkv_absn(xb_lo, xb_hi, gb, rstd), True)
    x_scale = gl.where(amax == 0.0, 1.0, gl.div_rn(amax, gl.full(amax.shape, 448.0, gl.float32, amax.type.layout)))
    # token k's h = bf16(ar + res), from CTA k
    if t < S:
        k3 = ar(SP, layout=gl.SliceLayout(1, gl.SliceLayout(2, XL3)))[:, None, None]
        put(rgn(sc, R_H), pa, bf_pair(xa_lo, xa_hi), tag, mask=(k3 == t) & oka)
        put(rgn(sc, R_H), pb, bf_pair(xb_lo, xb_hi), tag, mask=(k3 == t) & okb)
    # FP8 x -> LDS [steps, 16 tokens, 128] (steps 0..31, 32..47): the B operand, round
    # r = steps 8 r .. + 8
    xsa = gl.allocate_shared_memory(gl.uint8, [4, 8 * MAX_S, 128], BSW)
    xsb = gl.allocate_shared_memory(gl.uint8, [2, 8 * MAX_S, 128], BSW)
    qkv_quant(xa_lo, xa_hi, ga, rstd, x_scale, xsa, 32, SP)
    qkv_quant(xb_lo, xb_hi, gb, rstd, x_scale, xsb, 16, SP)
    stamp(tr_, bid, 21)
    gl.barrier()
    acc = zacc([128, 32], MFMA)
    one_a = unit_scales([128, 4], A_L)
    one_b = unit_scales([32, 4], B_L)
    for r in gl.static_range(6):
        if r < 4:
            bv = rnd(xsa, r).permute((1, 0)).load(B_L)
        else:
            bv = rnd(xsb, r - 4).permute((1, 0)).load(B_L)
        acc = cdna4.mfma_scaled(aq[r].to(gl.float8e4nv, bitcast=True), one_a, "e4m3",
                                bv.to(gl.float8e4nv, bitcast=True), one_b, "e4m3", acc)
    red = gl.allocate_shared_memory(gl.float32, [128, MAX_S], FLAT2)
    los, his = bd_reduce(acc, 1, 8, red)
    # the tokens' x scales in the epilogue layout
    nn = ar(MAX_S, layout=gl.SliceLayout(0, PR_L))[None, :]
    kq = ar(SP, layout=gl.SliceLayout(1, gl.SliceLayout(2, XL3)))
    xsc = gl.zeros([8, MAX_S], gl.float32, PR_L)
    for k in gl.static_range(S):
        xsc = gl.where(nn == k, gl.sum(gl.where(kq == k, x_scale, 0.0), axis=0), xsc)
    lo = bf16_round(los[0] * xsc * ws_lo)
    hi = bf16_round(his[0] * xsc * ws_hi)
    put(rgn(sc, R_QKV), nn * QKV_PAIRS + 8 * t + rr, bf_pair(lo, hi), tag, mask=nn < S)


# ============================================================ head math (norm + rope)
@gluon.jit
def head_rows(qkv, tok, pair0, npairs: gl.constexpr, layout: gl.constexpr, tag, errors, bid, code: gl.constexpr):
    """``npairs`` / 64 heads of token tok's GEMV rows from pair pair0, each with its RoPE
    partner (dim d +- 32 of the head; d >= 64 none): [heads, 64 pairs] (lo, hi, plo, phi)."""
    NH: gl.constexpr = npairs // 64
    h = ar(NH, layout=gl.SliceLayout(1, layout))[:, None]
    j = ar(64, layout=gl.SliceLayout(0, layout))[None, :]
    pk = head_peek(qkv, tok, pair0, npairs, layout, tag, h >= 0)
    return head_land(qkv, pk, tag, errors, bid, code)


@gluon.jit
def head_peek(qkv, tok, pair0, npairs: gl.constexpr, layout: gl.constexpr, tag, pred):
    """``head_rows``' loads (rows and RoPE partners) issued only; ``head_land`` finishes."""
    NH: gl.constexpr = npairs // 64
    h = ar(NH, layout=gl.SliceLayout(1, layout))[:, None]
    j = ar(64, layout=gl.SliceLayout(0, layout))[None, :]
    base = tok * QKV_PAIRS + pair0 + h * 64
    ok = (h < NH) & pred
    pj = gl.where(j < 16, j + 16, j - 16)
    return (base + j, peek(qkv, base + j, tag, ok), base + pj, peek(qkv, base + pj, tag, ok & (j < 32)))


@gluon.jit
def head_land(qkv, pk, tag, errors, bid, code: gl.constexpr):
    i0, p0, i1, p1 = pk
    w = land(qkv, i0, tag, p0, errors, bid, code)
    pw = land(qkv, i1, tag, p1, errors, bid, code)
    return bf_lo(w), bf_hi(w), bf_lo(pw), bf_hi(pw)


@gluon.jit
def head_norm_rope(lo, hi, plo, phi, gamma, cos_sin, pos, eps):
    """Gemma RMSNorm over each head's 128 dims + partial NeoX RoPE (dims < 64): ->
    (lo, hi) f32 of [heads, 64 pairs] (pair j = dims 2 j, 2 j + 1)."""
    layout: gl.constexpr = lo.type.layout
    j = ar(64, layout=gl.SliceLayout(0, layout))[None, :]
    ss = gl.sum(lo * lo + hi * hi, axis=1)
    rr = hw_rsq(ss / 128.0 + eps)[:, None]
    gw = gl.load(gamma + j)
    pjj = gl.where(j < 16, j + 16, j - 16)
    pgw = gl.load(gamma + pjj, mask=j < 32, other=0)
    n_lo = lo * rr * (bf_lo(gw) + 1.0)
    n_hi = hi * rr * (bf_hi(gw) + 1.0)
    p_lo = plo * rr * (bf_lo(pgw) + 1.0)
    p_hi = phi * rr * (bf_hi(pgw) + 1.0)
    # cos / sin of dim d % 32 at this position: cos_sin row [cos 32 | sin 32] bf16
    i = j % 16
    cw = gl.load(cos_sin + pos * 32 + i)  # i32 pairs: cos dims 2 i, 2 i + 1
    sw = gl.load(cos_sin + pos * 32 + 16 + i)
    first = j < 16
    ps_lo = p_lo * bf_lo(sw)
    ps_hi = p_hi * bf_hi(sw)
    r_lo = gl.fma(n_lo, bf_lo(cw), gl.where(first, -ps_lo, ps_lo))
    r_hi = gl.fma(n_hi, bf_hi(cw), gl.where(first, -ps_hi, ps_hi))
    rope = j < 32
    return gl.where(rope, r_lo, n_lo), gl.where(rope, r_hi, n_hi)


HL = gl.constexpr(gl.BlockedLayout([1, 2], [2, 32], [8, 1], [1, 0]))  # [16 heads, 64 pairs]
H1 = gl.constexpr(gl.BlockedLayout([1, 1], [1, 64], [1, 8], [1, 0]))  # [1 head, 64 pairs]


@gluon.jit
def stage_ik(li, ptab, cos_sin, positions, slots, sc, tag, errors, bid, eps, S: gl.constexpr):
    """Every token's index k -> norm / rope -> vLLM's bf16 index cache at its slot."""
    e = lrow(ptab, li, (6, 18))
    g_ik = lp(e[0], gl.int32)
    icache = lp(e[1], gl.int32)
    for k in gl.static_range(S):
        pos = uniform(gl.load(positions + k).to(gl.int32))
        slot = uniform(gl.load(slots + k).to(gl.int32))
        lo, hi, plo, phi = head_rows(rgn(sc, R_QKV), k, (HEADS + 3) * 64, 64, H1, tag, errors, bid, E_QKV)
        o_lo, o_hi = head_norm_rope(lo, hi, plo, phi, g_ik, cos_sin, pos, eps)
        j = ar(64, layout=gl.SliceLayout(0, H1))[None, :]
        cdna4.buffer_store(bf_pair(o_lo, o_hi), icache, slot * 64 + j, mask=slot >= 0)


# ============================================================ split attention
@gluon.jit
def stage_own(k, li, ptab, cos_sin, positions, slots, sc, tag, errors, bid, eps):
    """Token k's own key (FlyDSL's k / v head tasks): K = gemma norm / RoPE of the k rows, V
    raw, fp8 -> the cache (vLLM's: unit scale, page-16 slot; for later steps) and to the
    tail split as tagged words (2 fp8 a pair): off its critical path."""
    e = lrow(ptab, li, (16, 17, 4))
    k_cache = lp(e[0], gl.uint8)
    v_cache = lp(e[1], gl.uint8)
    g_k = lp(e[2], gl.int32)
    qkv = rgn(sc, R_QKV)
    kpk = head_peek(qkv, k, HEADS * 64, 64, H1, tag, k >= 0)
    j = ar(64, layout=gl.SliceLayout(0, H1))[None, :]
    vidx = k * QKV_PAIRS + (HEADS + 1) * 64 + j
    vpk = peek(qkv, vidx, tag, j >= 0)
    pos = uniform(gl.load(positions + k).to(gl.int32))
    slot = uniform(gl.load(slots + k).to(gl.int32))
    klo, khi, kplo, kphi = head_land(qkv, kpk, tag, errors, bid, E_QKV)
    ko_lo, ko_hi = head_norm_rope(klo, khi, kplo, kphi, g_k, cos_sin, pos, eps)
    k8lo = clamp_fp8(bf16_round(ko_lo)).to(gl.float8e4nv).to(gl.uint8, bitcast=True).to(gl.int32)
    k8hi = clamp_fp8(bf16_round(ko_hi)).to(gl.float8e4nv).to(gl.uint8, bitcast=True).to(gl.int32)
    kw = k8lo | (k8hi << 8)
    put(rgn(sc, R_OWN), k * 128 + j, kw, tag)
    vw = land(qkv, vidx, tag, vpk, errors, bid, E_QKV)
    v8lo = clamp_fp8(bf_lo(vw)).to(gl.float8e4nv).to(gl.uint8, bitcast=True)
    v8hi = clamp_fp8(bf_hi(vw)).to(gl.float8e4nv).to(gl.uint8, bitcast=True)
    put(rgn(sc, R_OWN), k * 128 + 64 + j, v8lo.to(gl.int32) | (v8hi.to(gl.int32) << 8), tag)
    # the cache copies (page-16 slot: page p16 / 16, token p16 % 16)
    p16 = slot + (slot // 128) * 128
    pb = p16 // 16
    pt = p16 % 16
    d = 2 * j
    kdst = (pb * 2048 + (d // 16) * 256 + pt * 16 + d % 16) // 2
    cdna4.buffer_store(kw.to(gl.int16), k_cache.to(gl.pointer_type(gl.int16), bitcast=True), kdst,
                       mask=slot >= 0, cache=".wt")
    cdna4.buffer_store(v8lo, v_cache, pb * 2048 + d * 16 + pt, mask=slot >= 0, cache=".wt")
    cdna4.buffer_store(v8hi, v_cache, pb * 2048 + (d + 1) * 16 + pt, mask=slot >= 0, cache=".wt")


@gluon.jit
def stage_split(task, li, ptab, cos_sin, positions, slots, block_table, bt_stride, seq_lens, sc, tag, errors, bid,
                sm_scale, eps, tr_=None):
    """Token k's partition p (keys 256 p .. + 256) x 16 heads -> (o, m, l)."""
    k = task // N_SPLIT
    p = task % N_SPLIT
    e = lrow(ptab, li, (16, 17, 3, 4))
    k_cache = lp(e[0], gl.uint8)
    v_cache = lp(e[1], gl.uint8)
    g_q = lp(e[2], gl.int32)
    g_k = lp(e[3], gl.int32)
    seq = gl.maximum(uniform(gl.load(seq_lens + k)), 1)
    pos = uniform(gl.load(positions + k).to(gl.int32))
    key0 = p * SPLIT_KEYS
    spo = rgn(sc, R_SPO)
    tid = tids()
    if key0 >= seq:
        # past the context: m = NEG, l = 0 (the merge weighs it 0)
        put(rgn(sc, R_SPM), task * HEADS + tid, gl.full(tid.shape, NEG, gl.float32, tid.type.layout), tag,
            mask=tid < HEADS)
        put(rgn(sc, R_SPL), task * HEADS + tid, gl.full(tid.shape, 0.0, gl.float32, tid.type.layout), tag,
            mask=tid < HEADS)
        put(spo, task * (O_K // 2) + tid, tid * 0, tag)
        put(spo, task * (O_K // 2) + THREADS + tid, tid * 0, tag)
    else:
        bt = block_table + k * bt_stride
        b0 = uniform(gl.load(bt + 2 * p))
        b1 = uniform(gl.load(bt + gl.minimum(2 * p + 1, (seq - 1) // 128)))
        # K^T [128 d, 256 keys], V [256 keys, 128 d] of the partition, masked past the context
        kc = ar(8, layout=gl.SliceLayout(1, gl.SliceLayout(2, KT_L)))[:, None, None]
        kb = ar(16, layout=gl.SliceLayout(0, gl.SliceLayout(2, KT_L)))[None, :, None]
        kn = ar(256, layout=gl.SliceLayout(0, gl.SliceLayout(1, KT_L)))[None, None, :]
        kpg = gl.where(kn < 128, b0, b1) * 16 + (kn % 128) // 16
        kok = key0 + kn < seq
        k_off = kpg * 2048 + kc * 256 + (kn % 16) * 16 + kb
        kt = cdna4.buffer_load(k_cache, k_off, mask=kok, other=0)
        vp = ar(16, layout=gl.SliceLayout(1, gl.SliceLayout(2, VT_L)))[:, None, None]
        vt = ar(16, layout=gl.SliceLayout(0, gl.SliceLayout(2, VT_L)))[None, :, None]
        vd = ar(128, layout=gl.SliceLayout(0, gl.SliceLayout(1, VT_L)))[None, None, :]
        vpg = gl.where(vp < 8, b0, b1) * 16 + vp % 8
        vok = key0 + vp * 16 + vt < seq - 1  # the own key (tail) enters PV from registers, its stale row zeroed
        v_off = vpg * 2048 + vd * 16 + vt
        # whole pages (16 B loads), the keys past the context zeroed after: P is 0 there,
        # but a NaN byte would still poison the MFMA
        vv = cdna4.buffer_load(v_cache, v_off, mask=key0 + vp * 16 < seq, other=0)
        pin(tids())  # the K / V loads stay issued before the q wait
        # q: the token's 16 heads from the GEMV rows, normed and roped here
        qkv = rgn(sc, R_QKV)
        # q's rows and (tail) the own-key task's fp8 K / V, all issued before any wait
        tail = ((seq - 1) // SPLIT_KEYS == p) & (DBG_NOTAIL == 0)
        qpk = head_peek(qkv, k, 0, 1024, HL, tag, tail | (seq > 0))
        oj = ar(64, layout=gl.SliceLayout(0, H1))[None, :]
        oidx = k * 128 + oj
        opk_k = peek(rgn(sc, R_OWN), oidx, tag, tail & (oj >= 0))
        opk_v = peek(rgn(sc, R_OWN), oidx + 64, tag, tail & (oj >= 0))
        lo, hi, plo, phi = head_land(qkv, qpk, tag, errors, bid, E_QKV)
        q_lo, q_hi = head_norm_rope(lo, hi, plo, phi, g_q, cos_sin, pos, eps)
        q8 = gl.reshape(gl.join(bf16_round(q_lo).to(gl.float8e4nv), bf16_round(q_hi).to(gl.float8e4nv)),
                        [16, 128])
        stamp(tr_, bid, 13)
        own = seq - 1 - key0  # the own key's index in the partition (in [0, 256) on the tail only)
        s_own = gl.zeros([16], gl.float32, gl.SliceLayout(1, MFMA2))
        if tail & ((DBG_NOTAIL & 2) == 0):
            # the own key enters from registers (``stage_own`` normed / roped / quantized it and
            # wrote the cache for later steps): its score q8 . k8 (fp8 products, exact in f32)
            # replaces column own
            kw = land(rgn(sc, R_OWN), oidx, tag, opk_k, errors, bid, E_QKV)
            kf_lo = gl.convert_layout(gl.reshape((kw & 255).to(gl.uint8).to(gl.float8e4nv, bitcast=True)
                                                 .to(gl.float32), [64]), gl.SliceLayout(0, HL))
            kf_hi = gl.convert_layout(gl.reshape(((kw >> 8) & 255).to(gl.uint8).to(gl.float8e4nv, bitcast=True)
                                                 .to(gl.float32), [64]), gl.SliceLayout(0, HL))
            qf_lo = bf16_round(q_lo).to(gl.float8e4nv).to(gl.float32)
            qf_hi = bf16_round(q_hi).to(gl.float8e4nv).to(gl.float32)
            s_own = gl.convert_layout(gl.sum(qf_lo * kf_lo[None, :] + qf_hi * kf_hi[None, :], axis=1),
                                      gl.SliceLayout(1, MFMA2))
            stamp(tr_, bid, 26)
        # QK: [16 heads, 128] x [128, 256 keys]
        qa = gl.convert_layout(q8, A2)
        kb2 = gl.convert_layout(gl.reshape(kt, [128, 256]), B2)
        s = cdna4.mfma_scaled(qa, unit_scales([16, 4], A2), "e4m3", kb2.to(gl.float8e4nv, bitcast=True),
                              unit_scales([256, 4], B2), "e4m3", zacc([16, 256], MFMA2))
        kidx = ar(256, layout=gl.SliceLayout(0, MFMA2))[None, :]
        s = gl.where(kidx == own, s_own[:, None], s)
        s = gl.where(key0 + kidx < seq, sm_scale * s, NEG)
        mx = gl.max(s, axis=1)
        pj = hw_exp2((s - mx[:, None]) * LOG2E)
        ls = gl.sum(pj, axis=1)
        stamp(tr_, bid, 15)
        # P in fp8 (448 p); the own key's column stays out of the MFMA (its V row is zero
        # there) and joins the accumulator from registers: o += p_own v_own (exact products)
        pf = (448.0 * pj).to(gl.float8e4nv).to(gl.float32)
        p8 = gl.where((key0 + kidx < seq) & (kidx != own), pf, 0.0).to(gl.float8e4nv)
        pa = gl.convert_layout(p8, A2)
        p_own = gl.zeros([16], gl.float32, gl.SliceLayout(1, MFMA2))
        v_own = gl.zeros([128], gl.float32, gl.SliceLayout(0, MFMA2))
        if tail & ((DBG_NOTAIL & 4) == 0):
            # only the tail's last page is partial (and holds the own key's stale row): a
            # page past the context loaded as zeros, every other partition's keys all valid
            vv = gl.where(vok, vv, 0)
            stamp(tr_, bid, 27)
            # the own key's V (``stage_own``), after QK / softmax
            vw = land(rgn(sc, R_OWN), oidx + 64, tag, opk_v, errors, bid, E_QKV)
            v8lo = (vw & 255).to(gl.uint8)
            v8hi = ((vw >> 8) & 255).to(gl.uint8)
            stamp(tr_, bid, 28)
            vf = gl.join(v8lo.to(gl.float8e4nv, bitcast=True).to(gl.float32),
                         v8hi.to(gl.float8e4nv, bitcast=True).to(gl.float32))
            v_own = gl.convert_layout(gl.reshape(vf, [128]), gl.SliceLayout(0, MFMA2))
            stamp(tr_, bid, 30)
            # column own of P, from the per-head score and max directly (the same ops: no
            # cross-warp reduction, which waits for every wave)
            p_own = (448.0 * hw_exp2((sm_scale * s_own - mx) * LOG2E)).to(gl.float8e4nv).to(gl.float32)
            stamp(tr_, bid, 14)
        vb2 = gl.convert_layout(gl.reshape(vv, [256, 128]), B2)
        o = cdna4.mfma_scaled(pa, unit_scales([16, 8], A2), "e4m3", vb2.to(gl.float8e4nv, bitcast=True),
                              unit_scales([128, 8], B2), "e4m3", zacc([16, 128], MFMA2))
        o = o + p_own[:, None] * v_own[None, :]
        inv_l = 1.0 / gl.where(ls > 0.0, ls, 1.0)
        o = (o * (1.0 / 448.0)) * inv_l[:, None]
        stamp(tr_, bid, 16)
        # publish: bf16 pairs [16 heads, 64], m, l
        OL: gl.constexpr = gl.BlockedLayout([1, 2], [1, 64], [8, 1], [1, 0])
        o2 = gl.convert_layout(o, OL)
        o_lo, o_hi = gl.split(gl.reshape(o2, [16, 64, 2]))
        hh = ar(16, layout=gl.SliceLayout(1, o_lo.type.layout))[:, None]
        jj = ar(64, layout=gl.SliceLayout(0, o_lo.type.layout))[None, :]
        put(spo, task * (O_K // 2) + hh * 64 + jj, bf_pair(o_lo, o_hi), tag)
        h1 = ar(16, layout=gl.SliceLayout(1, MFMA2))
        put(rgn(sc, R_SPM), task * HEADS + h1, mx, tag)
        put(rgn(sc, R_SPL), task * HEADS + h1, ls, tag)


# ============================================================ merge + o_proj + attention all-reduce
ML = gl.constexpr(gl.BlockedLayout([1, 1, 2], [1, 2, 32], [1, 8, 1], [2, 1, 0]))  # [8 parts, 16 heads, 64 pairs]


@gluon.jit
def stage_o(t, li, ptab, n_layers, sc, peers, peer_local, pbase, rank, hm_out, tag, errors, bid, S: gl.constexpr,
            FFN_OFF: gl.constexpr, tr_=None):
    """o task t: hidden rows 32 t .. + 32."""
    e = lrow(ptab, li, (7, 8))
    w_o = lp(e[0], gl.uint8)
    s_o = lp(e[1], gl.float32)
    # weights first: tile tt = rows 16 (2 t + tt) .., round r, wave (tt, p) step 4 r + p
    m = w8_rows()
    wv = m // 16
    aq = ()
    for r in gl.static_range(4):
        aq = aq + (w8_tile(w_o, (2 * t + wv // 4) * (16 * 2048) + (4 * r + wv % 4) * 2048 + (m % 16) * 16),)
    jr = ar(8, layout=gl.SliceLayout(1, PR_L))[:, None]
    ws = ()
    for tt in gl.static_range(2):
        ws = ws + ((gl.load(s_o + 32 * t + 16 * tt + 2 * jr), gl.load(s_o + 32 * t + 16 * tt + 2 * jr + 1)),)
    m = pin(m)
    # every token's partitions merged -> per-token FP8 rows in LDS [16 heads, 16 tokens, 128]
    xs = gl.allocate_shared_memory(gl.uint8, [4, 4 * MAX_S, 128], BSW)
    pi = ar(8, layout=gl.SliceLayout(1, gl.SliceLayout(2, ML)))[:, None, None]
    hi3 = ar(16, layout=gl.SliceLayout(0, gl.SliceLayout(2, ML)))[None, :, None]
    ji = ar(64, layout=gl.SliceLayout(0, gl.SliceLayout(1, ML)))[None, None, :]
    MH: gl.constexpr = gl.SliceLayout(2, ML)
    pm = ar(8, layout=gl.SliceLayout(1, MH))[:, None]
    hm2 = ar(16, layout=gl.SliceLayout(0, MH))[None, :]
    nn = ar(MAX_S, layout=gl.SliceLayout(0, PR_L))[None, :]
    xsc = gl.zeros([8, MAX_S], gl.float32, PR_L)
    for k in gl.static_range(S):
        sp0 = k * N_SPLIT
        im = (sp0 + pm) * HEADS + hm2
        io = (sp0 + pi) * (O_K // 2) + hi3 * 64 + ji
        pkm = peek(rgn(sc, R_SPM), im, tag, pm < 8)
        pkl = peek(rgn(sc, R_SPL), im, tag, pm < 8)
        pko = peek(rgn(sc, R_SPO), io, tag, pi < 8)
        wml = wait_group(((rgn(sc, R_SPM), im, pkm), (rgn(sc, R_SPL), im, pkl)), tag, errors, bid, E_SPLIT, 8)
        mw = wml[0]
        lw = wml[1]
        m_i = mw.to(gl.float32, bitcast=True)
        l_i = lw.to(gl.float32, bitcast=True)
        gm = gl.max(m_i, axis=0)
        sl = l_i * hw_exp2((m_i - gm[None, :]) * LOG2E)
        gs = gl.sum(sl, axis=0)
        gs = gl.where(gs > 0.0, gs, 1.0)
        wgt = (sl / gs[None, :])[:, :, None]
        ow = wait_group(((rgn(sc, R_SPO), io, pko),), tag, errors, bid, E_SPLIT, 16)[0]
        o_lo = bf_lo(ow)
        o_hi = bf_hi(ow)
        av_lo = bf16_round(gl.sum(wgt * o_lo, axis=0))
        av_hi = bf16_round(gl.sum(wgt * o_hi, axis=0))
        amax = gl.max(gl.max(gl.maximum(gl.abs(av_lo), gl.abs(av_hi)), axis=1), axis=0)
        x_scale = amax * (1.0 / 448.0)
        inv = gl.where(amax == 0.0, 0.0, hw_rcp(x_scale))
        q8 = gl.reshape(gl.join((av_lo * inv).to(gl.float8e4nv), (av_hi * inv).to(gl.float8e4nv)), [16, 128])
        xs.slice(4 * k, 4, dim=1).store(gl.reshape(q8.to(gl.uint8, bitcast=True), [4, 4, 128]))
        xsc = gl.where(nn == k, x_scale, xsc)
    stamp(tr_, bid, 11)
    gl.barrier()
    acc = zacc([128, 16], MFMA)
    one_a = unit_scales([128, 4], A_L)
    one_b = unit_scales([16, 4], B_L)
    for r in gl.static_range(4):
        bv = rnd(xs, r).permute((1, 0)).load(B_L)
        acc = cdna4.mfma_scaled(aq[r].to(gl.float8e4nv, bitcast=True), one_a, "e4m3",
                                bv.to(gl.float8e4nv, bitcast=True), one_b, "e4m3", acc)
    red = gl.allocate_shared_memory(gl.float32, [128, MAX_S], FLAT2)
    los, his = bd_reduce(acc, 2, 4, red)
    attn_peer, _, attn_own, _ = peer_regions(peers, peer_local, pbase, FFN_OFF)
    live = nn < S
    for tt in gl.static_range(2):
        lo = bf16_round(los[tt] * xsc * ws[tt][0])
        hi = bf16_round(his[tt] * xsc * ws[tt][1])
        word = bf_pair(lo, hi)
        at = (rank * MAX_S + nn) * HPAIRS + 16 * t + 8 * tt + jr
        for r in gl.static_range(4):
            put_sys(attn_peer[r], at, word, tag, mask=live)
    # this task's rows of the all-reduce: every rank's partials in rank order (and h),
    # all the loads issued before any wait
    pks = ()
    for tt in gl.static_range(2):
        pw = nn * HPAIRS + 16 * t + 8 * tt + jr
        for r in gl.static_range(4):
            pks = pks + (peek(attn_own, r * (MAX_S * HPAIRS) + pw, tag, live),)
        pks = pks + (peek(rgn(sc, R_H), pw, tag, live),)
    items = ()
    for tt in gl.static_range(2):
        pw = nn * HPAIRS + 16 * t + 8 * tt + jr
        for r in gl.static_range(4):
            items = items + ((attn_own, r * (MAX_S * HPAIRS) + pw, pks[5 * tt + r]),)
        items = items + ((rgn(sc, R_H), pw, pks[5 * tt + 4]),)
    ws = wait_group(items, tag, errors, bid, E_ATTN, 1)
    for tt in gl.static_range(2):
        pw = nn * HPAIRS + 16 * t + 8 * tt + jr
        t_lo = bf_lo(ws[5 * tt])
        t_hi = bf_hi(ws[5 * tt])
        for r in gl.static_range(1, 4):
            t_lo = t_lo + bf_lo(ws[5 * tt + r])
            t_hi = t_hi + bf_hi(ws[5 * tt + r])
        h_lo = bf_lo(ws[5 * tt + 4])
        h_hi = bf_hi(ws[5 * tt + 4])
        put(rgn(sc, R_A), pw, bf_pair(t_lo, t_hi), tag, mask=live)
        hmw = bf_pair(bf16_round(t_lo) + h_lo, bf16_round(t_hi) + h_hi)
        put(rgn(sc, R_HM), pw, hmw, tag, mask=live)
        if li == n_layers - 1:
            cdna4.buffer_store(hmw, i32p(hm_out), pw, mask=live)


# ============================================================ router
# FlyDSL's scheme: a wave per expert (gate row in 16 B chunks, bf16 dot2, a wave sum: no
# barrier), the row normed once (one CTA reduction) and shared through LDS, the MXFP8
# slice and the scores published straight from registers (no layout conversions)
GL = gl.constexpr(gl.BlockedLayout([1, 4], [1, 64], [8, 1], [1, 0]))  # [8 experts, pairs]: wave e, 4 pairs a lane
NA = gl.constexpr(gl.BlockedLayout([4], [64], [8], [0]))  # pairs 0 .. 2048: thread t pairs 4 t ..
NB = gl.constexpr(gl.BlockedLayout([2], [64], [8], [0]))  # pairs 2048 .. 3072: thread t pairs 2048 + 2 t ..
SCL = gl.constexpr(gl.BlockedLayout([1, 1], [1, 64], [8, 1], [1, 0]))  # [8 experts, 8 copies]: wave e
XROW = gl.constexpr(gl.SwizzledSharedLayout(1, 1, 1, [0]))


@gluon.jit
def router_gate(gate, pair0: gl.constexpr, NP: gl.constexpr, e0):
    """Gate rows e0 .. e0 + 8 (bf16 pairs pair0 .. pair0 + NP), wave e row e0 + e."""
    e = ar(8, layout=gl.SliceLayout(1, GL))[:, None]
    q = ar(NP, layout=gl.SliceLayout(0, GL))[None, :]
    return gl.load(gate + (e0 + e) * HPAIRS + pair0 + q)


@gluon.jit
def dot2_bf16(a, b):
    """a.lo b.lo + a.hi b.hi (bf16 pairs as i32) in f32: v_dot2_f32_bf16."""
    return gl.inline_asm_elementwise("v_dot2_f32_bf16 $0, $1, $2, 0", "=v,v,v", [a, b], dtype=gl.float32,
                                     is_pure=True, pack=1)


@gluon.jit
def stage_router(tr, li, ptab, sc, tag, errors, bid, eps, S: gl.constexpr, tr_=None):
    """Router task tr = (token k, experts 8 (tr % 16) ..)."""
    k = tr // 16
    tq = tr % 16
    e0 = tq * 8
    e = lrow(ptab, li, (10, 11, 9))
    gate = lp(e[0], gl.int32)
    bias = lp(e[1], gl.float32)
    g_post = lp(e[2], gl.int32)
    gA = router_gate(gate, 0, 2048, e0)
    gB = router_gate(gate, 2048, 1024, e0)
    e1 = ar(8, layout=gl.SliceLayout(1, GL))
    bias_e = gl.load(bias + e0 + e1)
    qa = ar(2048, layout=NA)
    qb = 2048 + ar(1024, layout=NB)
    gpA = gl.load(g_post + qa)
    gpB = gl.load(g_post + qb)
    gA = pin(gA)
    pwA = k * HPAIRS + qa
    pwB = k * HPAIRS + qb
    paA = peek(rgn(sc, R_A), pwA, tag, qa >= 0)
    phA = peek(rgn(sc, R_H), pwA, tag, qa >= 0)
    paB = peek(rgn(sc, R_A), pwB, tag, qb >= 0)
    phB = peek(rgn(sc, R_H), pwB, tag, qb >= 0)
    wa = wait_group(((rgn(sc, R_A), pwA, paA), (rgn(sc, R_H), pwA, phA)), tag, errors, bid, E_A, 4)
    wb = wait_group(((rgn(sc, R_A), pwB, paB), (rgn(sc, R_H), pwB, phB)), tag, errors, bid, E_A, 2)
    vA_lo = bf_lo(wa[0]) + bf_lo(wa[1])
    vA_hi = bf_hi(wa[0]) + bf_hi(wa[1])
    vB_lo = bf_lo(wb[0]) + bf_lo(wb[1])
    vB_hi = bf_hi(wb[0]) + bf_hi(wb[1])
    stamp(tr_, bid, 17)
    # sum of squares: each thread's 12 in registers, then one CTA reduction
    sA = gl.sum(gl.reshape(vA_lo * vA_lo + vA_hi * vA_hi, [512, 4]), axis=1)
    sB = gl.sum(gl.reshape(vB_lo * vB_lo + vB_hi * vB_hi, [512, 2]), axis=1)
    ss = gl.sum(sA + gl.convert_layout(sB, sA.type.layout), axis=0)
    rstd = hw_rsq(ss / 6144.0 + eps)
    xA_lo = bf16_round(vA_lo * rstd * (bf_lo(gpA) + 1.0))
    xA_hi = bf16_round(vA_hi * rstd * (bf_hi(gpA) + 1.0))
    xB_lo = bf16_round(vB_lo * rstd * (bf_lo(gpB) + 1.0))
    xB_hi = bf16_round(vB_hi * rstd * (bf_hi(gpB) + 1.0))
    xs = gl.allocate_shared_memory(gl.int32, [4096], XROW)
    xs.slice(0, 2048).store(bf_pair(xA_lo, xA_hi))
    xs.slice(2048, 1024).store(bf_pair(xB_lo, xB_hi))
    # this task's slice of xn as MXFP8 (aiter's rule): blocks 12 tq .. + 12, from registers
    mx8_put(gl.reshape(xA_lo, [128, 16]), gl.reshape(xA_hi, [128, 16]), 0, tq, sc, k, tag)
    mx8_put(gl.reshape(xB_lo, [64, 16]), gl.reshape(xB_hi, [64, 16]), 128, tq, sc, k, tag)
    gl.barrier()
    xa = xs.slice(0, 2048).load(gl.SliceLayout(0, GL))
    xb = xs.slice(2048, 1024).load(gl.SliceLayout(0, GL))
    logit = gl.sum(dot2_bf16(gA, xa[None, :]), axis=1) + gl.sum(dot2_bf16(gB, xb[None, :]), axis=1)
    stamp(tr_, bid, 18)
    sig = hw_rcp(1.0 + hw_exp2(-bf16_round(logit) * LOG2E))
    rk = route_key(sig + bias_e, e0 + e1)
    # copy c of (key, sigmoid) of expert e from wave e's lane c
    c2 = ar(SCORE_COPIES, layout=gl.SliceLayout(0, SCL))[None, :]
    e2 = ar(8, layout=gl.SliceLayout(1, SCL))[:, None]
    rk2 = gl.convert_layout(rk, gl.SliceLayout(1, SCL))[:, None] + c2 * 0
    sg2 = gl.convert_layout(sig, gl.SliceLayout(1, SCL))[:, None] + c2 * 0.0
    at = ((c2 * MAX_S + k) * N_ROUTED + e0 + e2) * 2
    scr = rgn(sc, R_SC)
    put(scr, at, rk2, tag)
    put(scr, at + 1, sg2, tag)
    stamp(tr_, bid, 19)


@gluon.jit
def mx8_put(lo, hi, blk0: gl.constexpr, tq, sc, k, tag):
    """[NB blocks, 16 pairs] of xn (blocks blk0 ..) -> MXFP8 words (4 fp8 a word) and E8M0
    codes of the task's blocks 12 tq .. + 12, in the tensors' own layout."""
    layout: gl.constexpr = lo.type.layout
    NB: gl.constexpr = lo.shape[0]
    amax = gl.max(gl.maximum(gl.abs(lo), gl.abs(hi)), axis=1)
    code = code_ceil(amax * (1.0 / 448.0))
    inv = hw_rcp(((code << 23).to(gl.float32, bitcast=True)))[:, None]
    q_lo = clamp_fp8(lo * inv).to(gl.float8e4nv).to(gl.uint8, bitcast=True).to(gl.int32)
    q_hi = clamp_fp8(hi * inv).to(gl.float8e4nv).to(gl.uint8, bitcast=True).to(gl.int32)
    w2 = q_lo | (q_hi << 8)  # [NB, 16]: pair j's two fp8
    a, c = gl.split(gl.reshape(w2, [NB, 8, 2]))
    word = a | (c << 16)  # [NB, 8]: 4 fp8 a word
    WL: gl.constexpr = word.type.layout
    bw = blk0 + ar(NB, layout=gl.SliceLayout(1, WL))[:, None]
    jj = ar(8, layout=gl.SliceLayout(0, WL))[None, :]
    put(rgn(sc, R_XN8), k * (HIDDEN // 4) + bw * 8 + jj, word, tag, mask=(bw >= 12 * tq) & (bw < 12 * tq + 12))
    b = blk0 + ar(NB, layout=gl.SliceLayout(1, layout))
    put(rgn(sc, R_XSC), k * (HIDDEN // 32) + b, code, tag, mask=(b >= 12 * tq) & (b < 12 * tq + 12))


# ============================================================ MoE (narrow: S <= 4)
RT = gl.constexpr(gl.BlockedLayout([1, 2], [1, 64], [8, 1], [1, 0]))  # [8 tokens, 128 experts]
UG_DEPTH = gl.constexpr(8)  # up / gate rounds in flight (16 B a lane each)
REP8 = gl.constexpr(gl.BlockedLayout([8], [64], [8], [0]))  # [8], every thread holding all 8
XW3 = gl.constexpr(gl.BlockedLayout([1, 1, 1], [1, 2, 32], [1, 8, 1], [2, 1, 0]))  # [S, steps, 32 words]
CL3 = gl.constexpr(gl.BlockedLayout([1, 1, 1], [1, 16, 4], [1, 8, 1], [2, 1, 0]))  # [S, steps, 4 codes]
MD3 = gl.constexpr(gl.BlockedLayout([1, 1, 2], [1, 2, 32], [1, 8, 1], [2, 1, 0]))  # [S, 32 entries, 64 pairs]
INT_MIN = gl.constexpr(-2147483648)


@gluon.jit
def route_all(sc, bid, tag, errors, route_scale, pred, S: gl.constexpr):
    """Every token's top-4 routed experts from the router's (key, sigmoid) pairs (copy
    bid % SCORE_COPIES) -> (ids, weights): a [8] tensor (token rows) a slot. The weights
    are the picks' sigmoids renormalized and scaled by route_scale (ATOM's)."""
    kk = ar(8, layout=gl.SliceLayout(1, RT))[:, None]
    e = ar(128, layout=gl.SliceLayout(0, RT))[None, :]
    at = (((bid % SCORE_COPIES) * MAX_S + kk) * N_ROUTED + e) * 2
    ok = (kk < S) & pred
    scr = rgn(sc, R_SC)
    pk_k = peek(scr, at, tag, ok)
    pk_s = peek(scr, at + 1, tag, ok)
    rk, sw = wait_group(((scr, at, pk_k), (scr, at + 1, pk_s)), tag, errors, bid, E_SC, 2)
    sig = sw.to(gl.float32, bitcast=True)
    keys = gl.where(ok, rk, INT_MIN)
    ids = ()
    ws = ()
    for p in gl.static_range(TOP_K):
        mx = gl.max(keys, axis=1)
        keys = gl.where(keys == mx[:, None], INT_MIN, keys)
        idp = 127 - (mx & 127)
        ids = ids + (idp,)
        ws = ws + (gl.sum(gl.where(e == idp[:, None], sig, 0.0), axis=1),)
    scale = route_scale / gl.maximum((ws[0] + ws[1]) + (ws[2] + ws[3]), 1e-20)
    # token k's slot s as CTA-uniform scalars E[4 k + s], W[4 k + s]: a lookup is then a
    # scalar select (a reduction over the token rows crosses waves: an LDS round trip)
    E = ()
    W = ()
    for k in gl.static_range(S):
        for p in gl.static_range(TOP_K):
            ik = gl.convert_layout(ids[p], REP8)
            wk = gl.convert_layout(ws[p] * scale, REP8)
            E = E + (gl.minimum(gl.maximum(uniform(tok_of(ik, k)), 0), N_ROUTED - 1),)
            W = W + (uniform(tok_of(wk.to(gl.int32, bitcast=True), k)).to(gl.float32, bitcast=True),)
    return E, W


@gluon.jit
def tok_of(vec, k):
    """Row k of a [8] route tensor (REP8: in every thread), as a scalar."""
    kk = gl.arange(0, 8, layout=REP8)
    return gl.sum(gl.where(kk == k, vec, 0 * vec), axis=0)


@gluon.jit
def expert_of(E, k, slot, S: gl.constexpr):
    """Token k's slot-``slot`` expert (k, slot runtime scalars)."""
    e = E[0]
    for i in gl.static_range(1, 4 * S):
        e = gl.where((k == i // 4) & (slot == i % 4), E[i], e)
    return uniform(e)


@gluon.jit
def stage_xn(sc, tag, errors, bid, xsa, xsb, csa, csb, pred, S: gl.constexpr):
    """Every token's MXFP8 xn -> LDS [steps, 16 tokens, 128 B] (steps 0..31 in xsa,
    32..47 in xsb), its codes -> [steps, 16, 4] (csa, csb)."""
    SP: gl.constexpr = triton.next_power_of_2(S)
    xn_part(sc, tag, errors, bid, xsa, csa, pred, 0, 32, SP, S)
    xn_part(sc, tag, errors, bid, xsb, csb, pred, 32, 16, SP, S)


@gluon.jit
def xn_part(sc, tag, errors, bid, xs8, cs8, pred, st0: gl.constexpr, NST: gl.constexpr, SP: gl.constexpr,
            S: gl.constexpr):
    k3 = ar(SP, layout=gl.SliceLayout(1, gl.SliceLayout(2, XW3)))[:, None, None]
    st3 = st0 + ar(NST, layout=gl.SliceLayout(0, gl.SliceLayout(2, XW3)))[None, :, None]
    w3 = ar(32, layout=gl.SliceLayout(0, gl.SliceLayout(1, XW3)))[None, None, :]
    iw = k3 * (HIDDEN // 4) + st3 * 32 + w3
    kc = ar(SP, layout=gl.SliceLayout(1, gl.SliceLayout(2, CL3)))[:, None, None]
    sc3 = st0 + ar(NST, layout=gl.SliceLayout(0, gl.SliceLayout(2, CL3)))[None, :, None]
    g3 = ar(4, layout=gl.SliceLayout(0, gl.SliceLayout(1, CL3)))[None, None, :]
    ic = kc * (HIDDEN // 32) + sc3 * 4 + g3
    pkw = peek(rgn(sc, R_XN8), iw, tag, (k3 < S) & pred)
    pkc = peek(rgn(sc, R_XSC), ic, tag, (kc < S) & pred)
    wv = wait_group(((rgn(sc, R_XN8), iw, pkw),), tag, errors, bid, E_XN, SP * NST // 16)[0]
    b0 = (wv & 255).to(gl.uint8)
    b1 = ((wv >> 8) & 255).to(gl.uint8)
    b2 = ((wv >> 16) & 255).to(gl.uint8)
    b3 = ((wv >> 24) & 255).to(gl.uint8)
    x = gl.permute(gl.reshape(gl.join(gl.join(b0, b2), gl.join(b1, b3)), [SP, NST // 4, 4, 128]), (1, 0, 2, 3))
    xs8.slice(0, SP * 4, dim=1).store(gl.reshape(x, [NST // 4, SP * 4, 128]))
    cv = wait_group(((rgn(sc, R_XSC), ic, pkc),), tag, errors, bid, E_XN, SP)[0]
    cq = gl.permute(gl.reshape(cv.to(gl.uint8), [SP, NST // 4, 4, 4]), (1, 0, 2, 3))
    cs8.slice(0, SP * 4, dim=1).store(gl.reshape(cq, [NST // 4, SP * 4, 4]))


@gluon.jit
def ug_round(e, cg, r: gl.constexpr, w13, s13, offs):
    """Round r of up / gate tile pair (expert e, gate rows 16 cg .., up rows 768 + 16 cg ..):
    wave (tile, p) takes 128-k step 4 r + p -> (A tile, A scales), in flight. The expert,
    row group and round live in the scalar bases, the lane offsets ``offs`` (``ug_offs``)
    are loop-invariant. A task past the end is clamped by the caller (its result dropped)."""
    base = w13 + e.to(gl.int64) * (2 * INTER * (HIDDEN // 2)) + cg * (16 * (HIDDEN // 2)) + (4 * r) * 1024
    a = w4_tile(base, offs[0])
    sca = cdna4.buffer_load(s13 + ((e * 48 + cg // 2) * 6144 + r * 512 + cg % 2), offs[1])
    return a, sca


@gluon.jit
def ug_bround(r: gl.constexpr, xsa, xsb, csa, csb):
    """Round r's B operand (every token's xn, 4 steps) and its scales, from LDS."""
    BS: gl.constexpr = cdna4.get_mfma_scale_layout(B_L, [16, 4])
    if r < 8:
        bv = rnd(xsa, r).permute((1, 0)).load(B_L)
        bs = rnd(csa, r).load(BS)
    else:
        bv = rnd(xsb, r - 8).permute((1, 0)).load(B_L)
        bs = rnd(csb, r - 8).load(BS)
    return bv, bs


@gluon.jit
def ug_loads(e, cg, w13, s13, offs):
    """Up / gate tile pair of expert e: gate rows 16 cg .. + 16 (tile 0), up rows 768 +
    16 cg .. (tile 1); round r, wave (tile, p) takes 128-k step 4 r + p (in flight)."""
    aq = ()
    sq = ()
    for r in gl.static_range(12):
        a, sca = ug_round(e, cg, r, w13, s13, offs)
        aq = aq + (a,)
        sq = sq + (sca,)
    return aq, sq


@gluon.jit
def ug_mma(aq, sq, xsa, xsb, csa, csb, red):
    """The tile pair against every token's xn -> (gate (lo, hi), up (lo, hi)) [8 pairs, 16 t]."""
    acc = zacc([128, 16], MFMA)
    BS: gl.constexpr = cdna4.get_mfma_scale_layout(B_L, [16, 4])
    for r in gl.static_range(12):
        if r < 8:
            bv = rnd(xsa, r).permute((1, 0)).load(B_L)
            bs = rnd(csa, r).load(BS)
        else:
            bv = rnd(xsb, r - 8).permute((1, 0)).load(B_L)
            bs = rnd(csb, r - 8).load(BS)
        acc = cdna4.mfma_scaled(aq[r], sq[r], "e2m1", bv.to(gl.float8e4nv, bitcast=True), bs, "e4m3", acc)
    los, his = bd_reduce(acc, 2, 4, red)
    return los[0], his[0], los[1], his[1]


@gluon.jit
def swiglu(g, u, limit):
    """aiter's swiglu_mul: min(g, L) * sigmoid(1.702 min(g, L)) * (clamp(u, -L, L) + 1)."""
    gc = gl.minimum(g, limit)
    uc = gl.maximum(gl.minimum(u, limit), -limit)
    sig = hw_rcp(1.0 + hw_exp2((1.702 * gc) * -LOG2E))
    return gc * sig * (uc + 1.0)


MD2 = gl.constexpr(gl.BlockedLayout([1, 2], [2, 32], [8, 1], [1, 0]))  # [entries, 64 pairs]


@gluon.jit
def mids_peek(sc, tag, k, slot0: gl.constexpr, NE: gl.constexpr):
    """Token k's mid chunks of slots slot0 .., entry 8 (slot - slot0) + chunk (chunks 6, 7
    zero): the loads issued -> (pair index, live mask, peek)."""
    en = ar(NE, layout=gl.SliceLayout(1, MD2))[:, None]
    j = ar(64, layout=gl.SliceLayout(0, MD2))[None, :]
    sl = slot0 + en // 8
    ch = en % 8
    ok = (sl < SLOTS) & (ch < 6)
    im = (k * SLOTS + sl) * (INTER // 2) + ch * 64 + j
    return im, ok, peek(rgn(sc, R_MID), im, tag, ok)


@gluon.jit
def mids_part(sc, tag, errors, bid, k, NE: gl.constexpr, buf, cbuf, pk, tr_=None):
    """A ``mids_peek`` landed -> MXFP8 -> LDS ``buf`` [NE / 4, 4, MAX_S, 128] row k (codes
    ``cbuf``)."""
    im, ok, pm = pk
    PK: gl.constexpr = (NE // 16 if NE >= 16 else 1) * 2
    mv = wait_group(((rgn(sc, R_MID), im, pm),), tag, errors, bid, E_MID, PK)[0]
    if tr_ is not None:
        stamp(tr_, bid, 23)
    lo = bf_lo(mv)
    hi = bf_hi(mv)
    lo4 = gl.reshape(gl.where(ok, lo, 0.0), [NE, 4, 16])
    hi4 = gl.reshape(gl.where(ok, hi, 0.0), [NE, 4, 16])
    amax = gl.max(gl.maximum(gl.abs(lo4), gl.abs(hi4)), axis=2)
    code = code_ceil(amax * (1.0 / 448.0))
    inv = hw_rcp((code << 23).to(gl.float32, bitcast=True))[:, :, None]
    q_lo = clamp_fp8(lo4 * inv).to(gl.float8e4nv).to(gl.uint8, bitcast=True)
    q_hi = clamp_fp8(hi4 * inv).to(gl.float8e4nv).to(gl.uint8, bitcast=True)
    buf.slice(4 * k, 4, dim=1).store(gl.reshape(gl.join(q_lo, q_hi), [NE // 4, 4, 128]))
    cbuf.slice(4 * k, 4, dim=1).store(gl.reshape(code.to(gl.uint8), [NE // 4, 4, 4]))


@gluon.jit
def stage_mids(sc, tag, errors, bid, ms8, mc8, msh, mch, S: gl.constexpr, tr_=None):
    """Every token's mid rows -> MXFP8 (aiter's rule) in LDS [entries, 16 tokens, 128 B]
    (codes [entries, 4, 4]): routed slot s's 128-k chunk c at entry 8 s + c of ms8 (c < 6;
    chunks 6, 7 zero), the shared slot's at entry c of msh. Every load issued before any
    wait (one round trip for all)."""
    pks = ()
    for k in gl.static_range(S):
        pks = pks + (mids_peek(sc, tag, k, 0, 32), mids_peek(sc, tag, k, TOP_K, 8))
    for k in gl.static_range(S):
        mids_part(sc, tag, errors, bid, k, 32, ms8, mc8, pks[2 * k], tr_ if k == 0 else None)
        mids_part(sc, tag, errors, bid, k, 8, msh, mch, pks[2 * k + 1])


@gluon.jit
def down_round(i: gl.constexpr, t, ids, w2, s2, S: gl.constexpr, DBG: gl.constexpr, offs):
    """Round i's A tiles and scales of down task t (rows 32 t .. + 32): token i // 8's
    routed slot (i % 8) // 2, chunks 4 (i % 2) + q on wave (tile, q) for i < 8 S, then
    the shared expert's chunks 4 (i - 8 S) + q; chunks past 6 zero. Scalar bases, lane
    offsets ``offs`` (``dn_offs``)."""
    if i < 8 * S:
        if DBG & 2097152:  # debug: slot s's rounds read slot (s + 1) % 4's expert
            e = ids[4 * (i // 8) + ((i % 8) // 2 + 1) % 4]
        else:
            e = ids[4 * (i // 8) + (i % 8) // 2]
        half: gl.constexpr = i % 2
    else:
        e = ids[0] * 0 + N_ROUTED
        half: gl.constexpr = i - 8 * S
    t = gl.minimum(t, N_DN - 1)  # a CTA without a down task streams task N_DN - 1's (dropped)
    base = w2 + e.to(gl.int64) * (HIDDEN * (INTER // 2)) + t * (32 * (INTER // 2)) + (4 * half) * 1024
    sb = s2 + ((e * 192 + t) * 768 + half * 512)
    if DBG & 512:
        a = w4_tile(base, offs[half], t < 0)
        sca = cdna4.buffer_load(sb, offs[2 + half], mask=offs[2 + half] < 0, other=0x7F)
    else:
        a = w4_tile(base, offs[half])
        sca = cdna4.buffer_load(sb, offs[2 + half])
    return a, sca


@gluon.jit
def down_wcol(i: gl.constexpr, wts, shared_weight, S: gl.constexpr):
    """Round i's route weight of each accumulator column (token n = column // 4); only
    the diagonal blocks are read (``bd_reduce``)."""
    n = ar(16, layout=gl.SliceLayout(0, MFMA)) // 4
    if i < 8 * S:
        w = gl.where(n == i // 8, wts[4 * (i // 8) + (i % 8) // 2], 0.0)
    else:
        w = gl.where(n < S, shared_weight, 0.0)
    return w[None, :]


@gluon.jit
def stage_down(t, li, n_layers, ids, wts, w2, s2, ms8, mc8, msh, mch, red, sc, peers, peer_local, pbase, rank, ar_out,
               tag, errors, bid, shared_weight, S: gl.constexpr, FFN_OFF: gl.constexpr, DBG: gl.constexpr = 0,
               tr_=None, pre=None):
    """Down task t: rows 32 t .. + 32, every token's route-weighted units -> bf16
    partials to every rank -> rank-ordered sum -> ar. ``pre``: rounds 0 .. DN_DEPTH - 1
    already in flight (streamed in the up / gate loop's tail), as (A tiles, scales)."""
    NRND: gl.constexpr = 8 * S + 2
    DEPTH: gl.constexpr = DN_DEPTH
    doffs = dn_offs()
    aw = ()
    asc = ()
    if DBG & 262144:  # debug A: every earlier memory op done first
        drain(tids())
    if DBG & 524288:  # debug B: weights only after the mids
        stage_mids(sc, tag, errors, bid, ms8, mc8, msh, mch, S)
        gl.barrier()
    if pre is not None:
        aw, asc = pre
    else:
        for i in gl.static_range(DEPTH):
            if i < NRND:
                a, sca = down_round(i, t, ids, w2, s2, S, DBG, doffs)
                aw = aw + (a,)
                asc = asc + (sca,)
    pin(tids())
    if (DBG & (1024 | 524288)) == 0:
        stamp(tr_, bid, 22)
        stage_mids(sc, tag, errors, bid, ms8, mc8, msh, mch, S, tr_)
    stamp(tr_, bid, 9)
    gl.barrier()
    acc = gl.zeros([128, 16], gl.float32, MFMA)
    BS: gl.constexpr = cdna4.get_mfma_scale_layout(B_L, [16, 4])
    for i in gl.static_range(NRND if (DBG & 16384) == 0 else 0):
        pin(tids())  # a round's LDS reads and next loads stay in it (else all hoisted: spills)
        if i + DEPTH < NRND:
            a, sca = down_round(i + DEPTH, t, ids, w2, s2, S, DBG, doffs)
            aw = aw + (a,)
            asc = asc + (sca,)
        if DBG & 32768:
            bv = gl.full([128, 16], 0x38, gl.uint8, B_L)
            bs = gl.full([16, 4], 127, gl.uint8, BS)
        elif i < 8 * S:
            bv = rnd(ms8, i % 8).permute((1, 0)).load(B_L)
            bs = rnd(mc8, i % 8).load(BS)
        else:
            bv = rnd(msh, i - 8 * S).permute((1, 0)).load(B_L)
            bs = rnd(mch, i - 8 * S).load(BS)  # 2 rounds of msh
        res = cdna4.mfma_scaled(aw[i], asc[i], "e2m1", bv.to(gl.float8e4nv, bitcast=True), bs, "e4m3",
                                zacc([128, 16], MFMA))
        acc = acc + res * down_wcol(i, wts, shared_weight, S)
    if DBG & 8192:
        return
    los, his = bd_reduce(acc, 2, 4, red)
    stamp(tr_, bid, 10)
    if DBG & 2048:
        return
    _, ffn_peer, _, ffn_own = peer_regions(peers, peer_local, pbase, FFN_OFF)
    jr = ar(8, layout=gl.SliceLayout(1, PR_L))[:, None]
    nn = ar(MAX_S, layout=gl.SliceLayout(0, PR_L))[None, :]
    live = nn < S
    for tt in gl.static_range(2):
        word = bf_pair(bf16_round(los[tt]), bf16_round(his[tt]))
        at = (rank * MAX_S + nn) * HPAIRS + 16 * t + 8 * tt + jr
        for r in gl.static_range(4):
            put_sys(ffn_peer[r], at, word, tag, mask=live)
    pks = ()
    for tt in gl.static_range(2):
        pw = nn * HPAIRS + 16 * t + 8 * tt + jr
        for r in gl.static_range(4):
            pks = pks + (peek(ffn_own, r * (MAX_S * HPAIRS) + pw, tag, live),)
    items = ()
    for tt in gl.static_range(2):
        pw = nn * HPAIRS + 16 * t + 8 * tt + jr
        for r in gl.static_range(4):
            items = items + ((ffn_own, r * (MAX_S * HPAIRS) + pw, pks[4 * tt + r]),)
    ws = wait_group(items, tag, errors, bid, E_FFN, 1)
    for tt in gl.static_range(2):
        pw = nn * HPAIRS + 16 * t + 8 * tt + jr
        t_lo = bf_lo(ws[4 * tt])
        t_hi = bf_hi(ws[4 * tt])
        for r in gl.static_range(1, 4):
            t_lo = t_lo + bf_lo(ws[4 * tt + r])
            t_hi = t_hi + bf_hi(ws[4 * tt + r])
        word = bf_pair(t_lo, t_hi)
        put(rgn(sc, R_AR), pw, word, tag, mask=live)
        if li == n_layers - 1:
            cdna4.buffer_store(word, i32p(ar_out), pw, mask=live)


@gluon.jit
def mid_put(sc, k_or_all, slot, cg, g_lo, g_hi, u_lo, u_hi, limit, tag, S: gl.constexpr):
    """A task's 16 mid values (rows 16 cg ..) -> bf16 pairs of token k (k_or_all >= 0)
    or of every token (the shared expert, k_or_all < 0)."""
    jr = ar(8, layout=gl.SliceLayout(1, PR_L))[:, None]
    nn = ar(MAX_S, layout=gl.SliceLayout(0, PR_L))[None, :]
    word = bf_pair(bf16_round(swiglu(g_lo, u_lo, limit)), bf16_round(swiglu(g_hi, u_hi, limit)))
    mine = gl.where(k_or_all < 0, nn < S, nn == k_or_all)
    put(rgn(sc, R_MID), (nn * SLOTS + slot) * (INTER // 2) + 8 * cg + jr, word, tag, mask=mine)


@gluon.jit
def stage_moe(li, ptab, n_layers, sc, peers, peer_local, pbase, rank, ar_out, tag, errors, bid, route_scale,
              shared_weight, swiglu_limit, S: gl.constexpr, FFN_OFF: gl.constexpr, STAGES: gl.constexpr = 511,
              tr_=None):
    NR: gl.constexpr = 4 * UG_TASKS * S  # routed up / gate tasks
    MAXT: gl.constexpr = (NR + BLOCKS - 1) // BLOCKS
    has_r = (bid < NR) & ((STAGES & 128) != 0)
    has_dn = (bid < N_DN) & ((STAGES & 256) != 0) & ((bid > 0) | ((STAGES & 65536) == 0))
    sh_j = bid - SHARED_BASE
    has_sh = (sh_j >= 0) & (sh_j < UG_TASKS) & ((STAGES & 64) != 0)
    if has_r | has_dn | has_sh:
        e = lrow(ptab, li, (12, 13, 14, 15))  # w13, s13, w2, s2: one scalar round trip
        w13 = lp(e[0], gl.uint8)
        s13 = lp(e[1], gl.uint8)
        xsa = gl.allocate_shared_memory(gl.uint8, [8, 4 * MAX_S, 128], BSW)
        xsb = gl.allocate_shared_memory(gl.uint8, [4, 4 * MAX_S, 128], BSW)
        csa = gl.allocate_shared_memory(gl.uint8, [8, 4 * MAX_S, 4], FLAT3)
        csb = gl.allocate_shared_memory(gl.uint8, [4, 4 * MAX_S, 4], FLAT3)
        red = gl.allocate_shared_memory(gl.float32, [128, MAX_S], FLAT2)
        uoffs = ug_offs()
        if has_sh:
            # the shared expert needs no routing: its weights, then every token's xn
            shj = gl.minimum(gl.maximum(sh_j, 0), UG_TASKS - 1)
            aq_s, sq_s = ug_loads(sh_j * 0 + N_ROUTED, shj, w13, s13, uoffs)
            stage_xn(sc, tag, errors, bid, xsa, xsb, csa, csb, sh_j >= 0, S)
            gl.barrier()
            g_lo, g_hi, up_lo, up_hi = ug_mma(aq_s, sq_s, xsa, xsb, csa, csb, red)
            mid_put(sc, -1, TOP_K, shj, g_lo, g_hi, up_lo, up_hi, swiglu_limit, tag, S)
            stamp(tr_, bid, 12)
        if has_r | has_dn:
            ids, wts = route_all(sc, bid, tag, errors, route_scale, bid >= 0, S)
            stamp(tr_, bid, 6)
            if STAGES & 131072:  # debug: every wave's view of the expert ids -> trace[bid, 4 w + i]
                tt_ = tids()
                for i in gl.static_range(4):
                    gl.store(tr_ + bid * 32 + (tt_ // 64) * 4 + i, (tt_ * 0 + ids[i]).to(gl.int64), mask=tt_ % 64 == 0)
            rbid = bid
            if STAGES & 1048576:  # debug: routed tasks rotated by 48 CTAs
                rbid = (bid + 48) % NR
            if has_r:
                # the CTA's routed tasks bid, bid + 256, .. as one stream of 12-round tile
                # pairs, UG_DEPTH rounds in flight across task boundaries
                TOT: gl.constexpr = MAXT * 12
                aw = ()
                asc = ()
                for j in gl.static_range(UG_DEPTH):
                    if j < TOT:
                        uj = gl.minimum(rbid + (j // 12) * BLOCKS, NR - 1)
                        ej = expert_of(ids, uj // 192, (uj % 192) // 48, S)
                        if STAGES & 4194304:  # debug: each wave's expert of rounds 0..3 -> trace[bid, 4 w + j]
                            if j < 4:
                                tt2 = tids()
                                gl.store(tr_ + bid * 32 + (tt2 // 64) * 4 + j, (tt2 * 0 + ej).to(gl.int64),
                                         mask=tt2 % 64 == 0)
                        a, sca = ug_round(ej, uj % 48, j % 12, w13, s13, uoffs)
                        aw = aw + (a,)
                        asc = asc + (sca,)
                pin(tids())
                if not has_sh:
                    stage_xn(sc, tag, errors, bid, xsa, xsb, csa, csb, bid >= 0, S)
                    gl.barrier()
                stamp(tr_, bid, 7)
                # a down CTA streams its first DN_DEPTH down rounds into the slots the loop's
                # last UG_DEPTH rounds free (as FlyDSL): they land before the mids do
                DNB: gl.constexpr = STAGES & 3145216
                w2p = lp(e[2], gl.uint8)
                s2p = lp(e[3], gl.uint8)
                daw = ()
                dsc = ()
                doffs = dn_offs()
                acc = zacc([128, 16], MFMA)
                for j in gl.static_range(TOT):
                    pin(tids())
                    if j + UG_DEPTH < TOT:
                        jn = j + UG_DEPTH
                        un = gl.minimum(rbid + (jn // 12) * BLOCKS, NR - 1)
                        a, sca = ug_round(expert_of(ids, un // 192, (un % 192) // 48, S), un % 48, jn % 12, w13, s13,
                                          uoffs)
                        aw = aw + (a,)
                        asc = asc + (sca,)
                    elif DN_STREAM and (j + UG_DEPTH - TOT < DN_DEPTH):
                        a, sca = down_round(j + UG_DEPTH - TOT, bid, ids, w2p, s2p, S, DNB, doffs)
                        daw = daw + (a,)
                        dsc = dsc + (sca,)
                    if j % 12 == 0:
                        acc = zacc([128, 16], MFMA)
                    bv, bs = ug_bround(j % 12, xsa, xsb, csa, csb)
                    acc = cdna4.mfma_scaled(aw[j], asc[j], "e2m1", bv.to(gl.float8e4nv, bitcast=True), bs, "e4m3",
                                            acc)
                    if j % 12 == 11:
                        u = rbid + (j // 12) * BLOCKS
                        los, his = bd_reduce(acc, 2, 4, red)
                        if u < NR:
                            mid_put(sc, u // 192, (u % 192) // 48, u % 48, los[0], his[0], los[1], his[1],
                                    swiglu_limit, tag, S)
                stamp(tr_, bid, 8)
                if has_dn:
                    if DN_STREAM:
                        for d in gl.static_range(UG_DEPTH, DN_DEPTH):
                            a, sca = down_round(d, bid, ids, w2p, s2p, S, DNB, doffs)
                            daw = daw + (a,)
                            dsc = dsc + (sca,)
                        run_down(bid, li, n_layers, ids, wts, e, red, sc, peers, peer_local, pbase, rank, ar_out, tag,
                                 errors, shared_weight, S, FFN_OFF, STAGES, tr_, (daw, dsc))
                    else:
                        run_down(bid, li, n_layers, ids, wts, e, red, sc, peers, peer_local, pbase, rank, ar_out, tag,
                                 errors, shared_weight, S, FFN_OFF, STAGES, tr_)
            if has_dn & ~has_r:  # (debug STAGES only: a down CTA is always a routed one)
                run_down(bid, li, n_layers, ids, wts, e, red, sc, peers, peer_local, pbase, rank, ar_out, tag, errors,
                         shared_weight, S, FFN_OFF, STAGES, tr_)


@gluon.jit
def run_down(bid, li, n_layers, ids, wts, e, red, sc, peers, peer_local, pbase, rank, ar_out, tag, errors,
             shared_weight, S: gl.constexpr, FFN_OFF: gl.constexpr, STAGES: gl.constexpr, tr_, pre=None):
    """The CTA's down task (``e``: the MoE stage's pointer row w13, s13, w2, s2)."""
    w2 = lp(e[2], gl.uint8)
    s2 = lp(e[3], gl.uint8)
    ms8 = gl.allocate_shared_memory(gl.uint8, [8, 4 * MAX_S, 128], BSW)
    mc8 = gl.allocate_shared_memory(gl.uint8, [8, 4 * MAX_S, 4], FLAT3)
    msh = gl.allocate_shared_memory(gl.uint8, [2, 4 * MAX_S, 128], BSW)
    mch = gl.allocate_shared_memory(gl.uint8, [2, 4 * MAX_S, 4], FLAT3)
    if (STAGES & 4096) == 0:
        stage_down(bid, li, n_layers, ids, wts, w2, s2, ms8, mc8, msh, mch, red, sc, peers, peer_local, pbase,
                   rank, ar_out, tag, errors, bid, shared_weight, S, FFN_OFF, STAGES & 3145216, tr_, pre)


# ============================================================ the launch
@gluon.jit
def peer_regions(peers, peer_local, pbase, FFN_OFF: gl.constexpr):
    attn_b = ()
    ffn_b = ()
    for r in gl.static_range(4):
        pb = uniform64(gl.load(peers + r)) + pbase
        attn_b = attn_b + (pb.to(gl.pointer_type(gl.int64), bitcast=True),)
        ffn_b = ffn_b + ((pb + FFN_OFF).to(gl.pointer_type(gl.int64), bitcast=True),)
    own = peer_local + pbase
    return attn_b, ffn_b, own.to(gl.pointer_type(gl.int64), bitcast=True), \
        (own + FFN_OFF).to(gl.pointer_type(gl.int64), bitcast=True)


@gluon.jit
def layer_body(li, ptab, n_layers, cos_sin, positions, slots, block_table, bt_stride, seq_lens, hid_in, res_in,
               ar_out, hm_out, scratch, peer_local, peers, errors, trace, rank, sm_scale, eps, route_scale,
               shared_weight, swiglu_limit, ep, bid,
               S: gl.constexpr, HALF: gl.constexpr, PHALF: gl.constexpr, FFN_OFF: gl.constexpr, STAGES: gl.constexpr):
    """Sparse layer li of the step (tag ep * TAGS + li + 1, scratch / peer half li % 2)."""
    tag = ep * TAGS + li + 1
    ptag = tag - 1
    par = li % 2
    sc = scratch + par * HALF
    psc = scratch + (1 - par) * HALF
    pbase = (par * PHALF).to(gl.int64)
    tr_ = trace + li * (BLOCKS * 32)
    stamp(tr_, bid, 0)
    topk = lrow(ptab, li, (19, 0))[0].to(gl.int32)
    nq = gl.where(topk != 0, ROWS16, NQ_REUSE)
    if (STAGES & 1) and bid < nq:
        stage_qkv(bid, li, ptab, hid_in, res_in, sc, psc, tag, ptag, errors, bid, eps, S, tr_)
    stamp(tr_, bid, 1)
    if (STAGES & 2) and (bid == ROWS16) & (topk != 0):
        stage_ik(li, ptab, cos_sin, positions, slots, sc, tag, errors, bid, eps, S)
    ok_ = bid - (ROWS16 + 1)
    if (STAGES & 4) and (ok_ >= 0) & (ok_ < S):
        stage_own(ok_, li, ptab, cos_sin, positions, slots, sc, tag, errors, bid, eps)
    st = BLOCKS - 1 - bid
    if (STAGES & 4) and st < N_SPLIT * S:
        stage_split(st, li, ptab, cos_sin, positions, slots, block_table, bt_stride, seq_lens, sc, tag, errors,
                    bid, sm_scale, eps, tr_)
    stamp(tr_, bid, 2)
    if (STAGES & 8) and bid < N_O:
        stage_o(bid, li, ptab, n_layers, sc, peers, peer_local, pbase, rank, hm_out, tag, errors, bid, S, FFN_OFF, tr_)
    stamp(tr_, bid, 3)
    tr = (bid + BLOCKS - ROUTER_BASE) % BLOCKS
    if (STAGES & 16) and tr < 16 * S:
        stage_router(tr, li, ptab, sc, tag, errors, bid, eps, S, tr_)
    stamp(tr_, bid, 4)
    if STAGES & 32:
        stage_moe(li, ptab, n_layers, sc, peers, peer_local, pbase, rank, ar_out, tag, errors, bid, route_scale,
                  shared_weight, swiglu_limit, S, FFN_OFF, STAGES, tr_)
    stamp(tr_, bid, 5)


# Every scalar and per-step pointer argument unspecialized: Triton would otherwise key the
# binary on li0 == 1, multiples of 16 and pointer alignment -- a recompile (10-20 s, per
# rank) inside a step's first launches, the other ranks spinning on its mailboxes meanwhile
@gluon.jit(do_not_specialize=["n_layers", "bt_stride", "rank", "li0"],
           do_not_specialize_on_alignment=["positions", "slots", "block_table", "seq_lens", "hid_in", "res_in",
                                           "ar_out", "hm_out", "cos_sin", "peer_local", "trace"])
def m3_step_kernel(ptab, n_layers, cos_sin, positions, slots, block_table, bt_stride, seq_lens, hid_in, res_in,
                   ar_out, hm_out, scratch, peer_local, peers, epoch, errors, trace, rank, sm_scale, eps,
                   route_scale, shared_weight, swiglu_limit, li0,
                   S: gl.constexpr, HALF: gl.constexpr, PHALF: gl.constexpr, FFN_OFF: gl.constexpr,
                   STAGES: gl.constexpr = 511, LOOP: gl.constexpr = False):
    """LOOP: every layer in this launch; else layer li0 only (a launch a layer, the step's
    last moving the epoch on)."""
    bid = gl.program_id(0)
    tid = tids()
    cos_sin = i32p(cos_sin)  # [positions, 64] bf16 as pairs
    ep = uniform(gl.load(epoch))
    marks = epoch.to(gl.pointer_type(gl.int64), bitcast=True) + EP_MARK_PAIRS
    if LOOP:
        put(marks, bid + tid * 0, tid * 0 + 1, ep + 1, mask=tid == 0)
        for li in range(n_layers):
            layer_body(li, ptab, n_layers, cos_sin, positions, slots, block_table, bt_stride, seq_lens, hid_in,
                       res_in, ar_out, hm_out, scratch, peer_local, peers, errors, trace, rank, sm_scale, eps,
                       route_scale, shared_weight, swiglu_limit, ep, bid, S, HALF, PHALF, FFN_OFF, STAGES)
        last = bid >= 0
    else:
        layer_body(li0, ptab, n_layers, cos_sin, positions, slots, block_table, bt_stride, seq_lens, hid_in,
                   res_in, ar_out, hm_out, scratch, peer_local, peers, errors, trace, rank, sm_scale, eps,
                   route_scale, shared_weight, swiglu_limit, ep, bid, S, HALF, PHALF, FFN_OFF, STAGES)
        last = li0 == n_layers - 1
        if last:
            put(marks, bid + tid * 0, tid * 0 + 1, ep + 1, mask=tid == 0)
    # every CTA read this step's epoch: CTA 0 moves it on
    if last & (bid == 0):
        _, err = pollv1(marks, tid, ep + 1, tid < BLOCKS)
        note_err(errors, bid, err, E_EPOCH)
        gl.barrier()
        gl.store(epoch + tid * 0, ep + 1 + tid * 0, mask=tid == 0)
