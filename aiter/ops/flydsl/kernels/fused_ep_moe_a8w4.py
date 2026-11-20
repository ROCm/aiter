# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Fused expert-parallel MoE layer, gfx950: dispatch, GEMM1 (+ clamp-SwiGLU + MXFP8 quant), GEMM2 (+ route
weights) and combine in one persistent launch per rank. MXFP8 activations x MXFP4 weights on the scaled MFMA,
hidden 5120, intermediate 2304, top 6, ``epr`` (<= 63) local experts per rank, 2 / 4 / 8 ranks of one node.

The symmetric-heap protocol (expert-owner row push, remote per-expert record counters, source arrival flags,
combine by peer writes) follows ``mega_moe/mega_moe_small_m.py``, with the differences below.

Memory (per rank: a device pool mapped on every peer by IPC, ``pool_layout``, and uncached control words): a
receive arena with one fp8 row (+ e8m0 scales) per (source rank, token), written once per destination rank that
owns an expert of the token; per local expert a list of 8-byte route records ``(row | slot << 24, weight)`` filled
through a remote atomic counter (``cnt``, one memory channel each, double-buffered by call parity); the combine
inbox ``[token, slot, 5120]`` bf16 that the peers' GEMM2 tiles write; per-source arrival flags and finish counters.

One call, per rank: workgroups ``[0, nd)`` dispatch their tokens (the last wave claims the record slots while the
other waves quantize and push rows); the last of them stores the rank's arrival flag into every rank's control
words (a two-level local ticket, no remote atomic contention). All ``nw`` workgroups then wait for the flags of all
ranks, scan the experts' row counts into a block table (blocks of ``bm`` rows of one expert) in LDS and pop work
items: GEMM1 tiles block-major, then GEMM2 tiles block-major (a GEMM2 tile waits for its block's GEMM1 tiles, its
weight loads are in flight meanwhile). The first ``nw`` items come from one queue per XCD, the rest from one global
queue, so a started workgroup never waits for an item no running workgroup can take. A GEMM2 tile stages its
route-weighted bf16 rows in LDS and stores them in 16-byte write-through stores into the token owner's inbox; the
last block adds 1 to this rank's finish counter on every rank. Each workgroup then reduces its tokens once every
rank's finish counter arrived: the six slots in slot order, one bf16 rounding, no atomics into outputs. The last
workgroup to retire resets the counters (a two-level ticket). A call ends only after every rank's GEMM2 finished,
so no rank starts the next call's dispatch while a peer still reads its arena; a call needs no fences (every
cross-rank store is a system-scope write-through store whose completion is waited for before its flag).

Coherence. The pool is coarse-grained device memory, which each XCD's L2 caches and a peer's writes do not update.
Everything a peer writes into it (arena rows and scales, route records, the combine inbox) is therefore read with
sc0 sc1 loads, which bypass L2 (as ``IpcHeap.alloc`` requires for coarse-grained memory), so peer data does not
depend on cache maintenance between launches (measured cost for the arena reads: within 1.2 %).
The GEMM1 output rows (``Q`` / ``S``, this GPU's memory, read by GEMM2 tiles on any XCD) are stored write-through
and counted only after the stores complete; no load of this launch reads them before the count, so a reader's L2
cannot hold a copy from this launch, and copies from earlier launches are dropped by the launch's acquire (the
same guarantee that lets one kernel read what a previous kernel wrote on another XCD, CUDA-graph replays
included).

Raw ``llvm`` / ``rocdl`` / inline asm is left only where the call site gives the reason (gfx950 ISA or measured):
the AGPR-accumulator MFMA of the 16- and 128-row tiers (``_mfma_agpr``), the async buffer-to-LDS copy (``_dma``),
the counter poll (``_poll``) and the 16-byte sc0 sc1 store (``_store128_sys``).
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.compiler.mlir_utils import convert_to_mlir_attr
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from .communication_ops_utils import hw_xcc_id
from .dpp_utils import update_dpp_i32

# E: the default local expert count (EPLB: any count up to NE_MAX per rank)
H, I, E, TOPK = 5120, 2304, 48, 6
NE_MAX = 63  # the plan's wave scan and the counters hold up to 63 experts
LIMIT = 10.0
KT1, KT2 = H // 128, I // 128  # 128-K MFMA steps of GEMM1 / GEMM2
SC1 = 16  # cache policy bit: device-scope (write-through) stores
SYS = 0x11  # sc0 | sc1: system-coherent loads / write-through stores
# Tile rows whose MFMAs accumulate in AGPRs through inline asm (_mfma_agpr) rather than the typed op: measured on
# 8 x MI355X, the typed op made the 16-row tiers 5-10 % slower at 1-4 tokens per rank (46.5 -> 50.2 us uniform,
# 107.6 -> 117.6 us all routes on one rank, at 1 token) and the 128-row tier spill (see _mfma_agpr); the 32- and
# 64-row tiers are within 1.5 % either way and use the typed op.
AGPR_BM = (16, 128)
# bytes between two blocks' counters (ready count, then consumer count)
READY_STRIDE = 256
TM = 1024  # tokens per rank the arenas hold
# bytes between two expert counters: one remote atomic stream per memory channel
CNT_STRIDE = 256
# control words (uncached memory the peers write): per call parity 64 expert counters,
CTL_CNT = 0
# the dispatch arrival flag of every source rank (64 B apart),
CTL_ARRIVE = 2 * 64 * CNT_STRIDE
CTL_FIN = CTL_ARRIVE + 16 * 64  # and its finish counter
CTL_BYTES = 65536
# workgroups of the dispatch count themselves in this many groups, the last of each in a final counter
ARRIVE_GROUPS = 16
RETIRE_GROUPS = 16
XCDS = 8
# word indices of the local state (LOC_WORDS int32): the call counter, retire and done counters,
LOC_EP, LOC_RETIRE, LOC_DONE = 0, 32, 48
LOC_ARR = 80  # the dispatch's final arrival counter,
LOC_QG = 96  # the work-queue head of the items after the first wave,
# the retire counters of RETIRE_GROUPS workgroup groups (16 words apart),
LOC_RETIRE_G = 1024
# the dispatch's ARRIVE_GROUPS group counters (16 words apart),
LOC_ARR_G = 2048
# and the first-wave work-queue heads of the 8 XCDs (16 words apart)
LOC_QH = 2560
LOC_WORDS = LOC_QH + 16 * XCDS
assert LOC_RETIRE_G + 16 * RETIRE_GROUPS <= LOC_ARR_G
assert LOC_ARR_G + 16 * ARRIVE_GROUPS <= LOC_QH


def pool_layout(world: int, epr: int = E) -> dict:
    """Byte offsets of a rank's pool: receive rows, their scales, route records, combine inbox."""
    rx = 0
    rxs = rx + world * TM * H
    meta = rxs + (world * TM * (H // 32) + 4095) // 4096 * 4096
    slots = meta + (epr * world * TM * 8 + 4095) // 4096 * 4096
    total = slots + TM * TOPK * H * 2
    return {"rx": rx, "rxs": rxs, "meta": meta, "slots": slots, "total": total}


def max_blocks(world: int, block_m: int, epr: int = E) -> int:
    """Bound on the row blocks one rank can receive: every one of its experts pads by < ``block_m`` rows."""
    return world * TM * TOPK // block_m + epr


def _raw(x):
    return x.ir_value() if hasattr(x, "ir_value") else x


_OOB = 0x7FFFFFF0  # a byte offset past every buffer: the access is dropped (a load returns 0)


# bytes a buffer view's layout spans; the descriptor's num_records bounds the accesses
_BUF_WINDOW = 1 << 30
_ELEM = {8: fx.Int8, 32: fx.Int32}


def _buf(addr, nbytes, elem=fx.Int32):
    """A FlyDSL buffer tensor of ``elem`` over ``nbytes`` bytes at the global address ``addr``: the hardware drops
    accesses at or past ``nbytes`` (a load returns 0). ``_bload`` / ``_bstore`` move elements at a per-lane index
    through ``fx.copy`` with a ``BufferCopy`` atom."""
    eb = elem.width // 8
    ptr_ty = fx.PointerType.get(elem.ir_type, fx.AddressSpace.Global, eb)
    view = fx.make_view(
        fx.inttoptr(ptr_ty, fx.Int64(addr)), fx.make_layout(_BUF_WINDOW // eb, 1)
    )
    return fx.rocdl.make_buffer_tensor(view, num_records_bytes=fx.Int64(nbytes))


def _brsrc(buf):
    """The raw buffer resource of ``buf``, for the async LDS copy and the buffer atomic."""
    return fx.rocdl.get_buffer_rsrc(fx.get_iter(buf))


def _bview(buf, idx, n, mask):
    """``n`` elements at element ``idx`` of ``buf``; a lane with ``mask`` false gets an out-of-range offset, which
    the hardware bounds check drops with no branch."""
    eb = buf.element_type.width // 8
    idx = fx.Int32(idx)
    if mask is not None:
        idx = mask.select(idx, fx.Int32(_OOB // eb))
    return fx.make_view(fx.get_iter(buf) + idx, fx.make_layout(n, 1))


def _bload(buf, idx, n=1, *, cm=0, soff=0, mask=None):
    """The raw value (a vector for ``n`` > 1) of elements ``idx`` .. ``idx + n - 1`` of ``buf``, cache policy
    ``cm``, plus the scalar offset ``soff`` (bytes); masked-off lanes read 0."""
    bits = buf.element_type.width
    elem = _ELEM[bits]
    assert soff % (bits // 8) == 0
    atom = fx.make_copy_atom(fx.rocdl.BufferCopy(n * bits, cm), elem)
    src = _bview(buf, idx, n, mask)
    frag = fx.make_fragment_like(src)
    fx.copy(atom, src, frag, soffset=fx.Int32(soff // (bits // 8)))
    v = Vec(fx.memref_load_vec(frag))
    return _raw(v if n > 1 else v[0])


def _bstore(buf, idx, value, *, cm=0, mask=None):
    """Store ``value`` (a scalar or a vector of the element type) at element ``idx`` of ``buf``."""
    bits = buf.element_type.width
    elem = _ELEM[bits]
    if not isinstance(value, Vec):
        value = _raw(value)
        if isinstance(value.type, ir.VectorType):
            value = Vec(value)
        else:
            value = Vec.from_elements([elem(value)], elem)
    n = value.shape[0]
    atom = fx.make_copy_atom(fx.rocdl.BufferCopy(n * bits, cm), elem)
    dst = _bview(buf, idx, n, mask)
    frag = fx.make_fragment_like(dst)
    fx.memref_store_vec(value, frag)
    fx.copy(atom, frag, dst)


def _mfma(w, a, acc, sw, sa, byte, *, agpr=False):
    """Scaled 16x16x128 MFMA, FP32 accumulate: FP4 weights ``w`` as operand A (the output is ``[weight column,
    row]``) x FP8 activations ``a``; ``byte`` (op_sel) picks the byte of the weight scale word ``sw``, ``sa`` holds
    the activation scale in its low byte. ``acc`` None starts from zero. ``agpr``: accumulate in AGPRs
    (``_mfma_agpr``; the caller reads the results through ``_hold``)."""
    if agpr:
        return _mfma_agpr(w, a, acc, sw, sa, byte)
    if acc is None:
        acc = fx.full(4, 0.0, fx.Float32)
    return fx.rocdl.mfma_scale_f32_16x16x128_f8f6f4(
        T.vec(4, T.f32), [w, a, acc, 4, 0, byte, sw, 0, sa]
    )


def _mfma_agpr(w, a, acc, sw, sa, byte):
    """``_mfma`` with the accumulator pinned to AGPRs."""
    # Raw asm, not fx.rocdl.mfma_scale_f32_16x16x128_f8f6f4, for the AGPR_BM tiers. In the 8-wave 128-row tier the
    # 32 accumulators per wave (128 registers) live across the K loop; with the typed op LLVM splits them between
    # VGPRs and AGPRs: ~6500 v_accvgpr copies and 540 instead of 100 bytes of scratch per lane (gfx950 ISA). The
    # 16-row tiers ran 5-10 % slower with it (AGPR_BM). The "=a" constraint pins the accumulators to AGPRs. LLVM's
    # hazard recognizer does not look into inline asm, so s_nop 1 before the MFMA covers a VALU or accumulator-copy
    # write of an operand right before it, and _hold waits out the MFMA latency before the results are read.
    src_c = "$0" if acc is not None else "0"
    asm = (
        f"s_nop 1\nv_mfma_scale_f32_16x16x128_f8f6f4 $0, $1, $2, {src_c}, $3, $4 "
        f"op_sel:[{byte % 2},0,0] op_sel_hi:[{byte // 2},0,0] cbsz:4 blgp:0"
    )
    ops = [_raw(w), _raw(a), _raw(sw), _raw(sa)]
    cons = "=a,v,v,v,v"
    if acc is not None:
        ops.append(_raw(acc))
        cons += ",0"
    return llvm.inline_asm(T.vec(4, T.f32), ops, asm, cons, has_side_effects=True)


def _hold(xs):
    """Wait out the latency of ``_mfma_agpr`` before its accumulators are read."""
    # Raw asm: a wait-state hold tied to the accumulators (see _mfma_agpr).
    if len(xs) == 1:  # one output cannot be an LLVM inline-asm struct
        return [
            llvm.inline_asm(
                T.vec(4, T.f32),
                [_raw(xs[0])],
                "s_nop 7\ns_nop 7\ns_nop 7",
                "=a,0",
                has_side_effects=True,
            )
        ]
    ty = ir.Type.parse("!llvm.struct<(" + ", ".join(["vector<4xf32>"] * len(xs)) + ")>")
    cons = ",".join(["=a"] * len(xs) + [str(i) for i in range(len(xs))])
    v = llvm.inline_asm(
        ty,
        [_raw(x) for x in xs],
        "s_nop 7\ns_nop 7\ns_nop 7",
        cons,
        has_side_effects=True,
    )
    return [llvm.extractvalue(T.vec(4, T.f32), v, [i]) for i in range(len(xs))]


def _lds_ptr(addr, dtype=fx.Int32, align=4):
    """A shared-memory pointer at the LDS byte address ``addr`` (the tiles address one LDS byte array)."""
    return fx.inttoptr(
        fx.PointerType.get(dtype.ir_type, fx.AddressSpace.Shared, align),
        fx.Int64(addr),
    )


def _lds_load(addr, count=None, dtype=fx.Int32, align=4):
    return fx.generic_load(_lds_ptr(addr, dtype, align), dtype=dtype, count=count)


def _lds_store(addr, value, dtype=fx.Int32, align=4):
    fx.generic_store(_lds_ptr(addr, dtype, align), value)


def _load8(buf, off, soff, plane=64, cm=0):
    """K [16 q, 16 q + 16) and [64 + 16 q, 80 + 16 q) of a 128-K step (lane group q) as one 8-dword MFMA operand: the
    FP8 operand layout of the scaled MFMA, which pairs with the FP4 operand's K [32 q, 32 q + 32). ``plane``: bytes
    between the two 16-byte halves; ``cm``: cache policy."""
    lo = Vec(_bload(buf, off, 4, soff=soff, cm=cm))
    hi = Vec(_bload(buf, off + plane // 4, 4, soff=soff, cm=cm))
    return Vec.from_elements(
        [lo[j] for j in range(4)] + [hi[j] for j in range(4)], fx.Int32
    )


def _dma(buf, lds_addr, voff, soff, cm=0):
    """Async ``buffer_load ... lds``: 16 bytes per lane from byte ``voff`` (+ ``soff``, wave-uniform) of ``buf`` into
    LDS at the wave-uniform ``lds_addr`` + 16 * lane, cache policy ``cm``; the caller closes the stage with
    ``asyncmark`` and waits with ``wait_asyncmark``."""
    lds = _lds_ptr(fx.Int32(fx.rocdl.readfirstlane(T.i32, _raw(fx.Int32(lds_addr)))))
    # Raw rocdl, not the BufferLoadAsyncLDS128b copy atom: the atom has no cache-policy state, and it takes a per-lane
    # LDS address and re-derives M0 with a v_readfirstlane per copy (gfx950 ISA: 207 instead of 78 in the 128-row
    # tier, which then spills 36 more bytes per lane).
    fx.rocdl.raw_ptr_buffer_load_async_lds(
        _brsrc(buf),
        lds.llvm_ptr,
        fx.Int32(16).ir_value(),
        fx.Int32(voff).ir_value(),
        fx.Int32(soff).ir_value(),
        fx.Int32(0).ir_value(),
        aux=convert_to_mlir_attr(cm),
    )


def _lds_load8(base, lo_off, hi_off):
    """Two 16-byte LDS reads as one 8-dword MFMA operand."""
    lo = _lds_load(base + lo_off, 4, align=16)
    hi = _lds_load(base + hi_off, 4, align=16)
    return Vec.from_elements(
        [lo[j] for j in range(4)] + [hi[j] for j in range(4)], fx.Int32
    )


def _clamp_silu_mul(g, u):
    g = (g < fx.Float32(LIMIT)).select(g, fx.Float32(LIMIT))
    u = (u < fx.Float32(LIMIT)).select(u, fx.Float32(LIMIT))
    u = (u > fx.Float32(-LIMIT)).select(u, fx.Float32(-LIMIT))
    e = fx.Float32(fx.rocdl.exp2(T.f32, _raw(g * fx.Float32(-1.4426950408889634))))
    r = fx.Float32(fx.rocdl.rcp(T.f32, _raw(fx.Float32(1) + e)))
    return (g * r) * u


def _e8m0(amax):
    """e8m0 exponent of ceil_pow2(amax / 448), floored at 1; 255 (NaN / Inf) is not bumped."""
    u = (amax * fx.Float32(1.0 / 448.0)).bitcast(fx.Int32)
    e = (u >> 23) & 255
    bump = ((u & 0x7FFFFF) != 0) & (e < 255)
    e = bump.select(e + 1, e)
    return (e < 1).select(fx.Int32(1), e)


def _fp8x4(a, b, c, d, scale):
    """Two v_cvt_scalef32_pk_fp8_f32: ``(a, b, c, d) / scale`` as 4 e4m3 bytes of one i32 (``a`` in the low byte)."""
    i16x2 = Vec.make_type(2, fx.Int16)
    lo = fx.rocdl.cvt_scalef32_pk_fp8_f32(
        i16x2, _raw(fx.full(2, 0, fx.Int16)), _raw(a), _raw(b), _raw(scale), False
    )
    both = fx.rocdl.cvt_scalef32_pk_fp8_f32(
        i16x2, lo, _raw(c), _raw(d), _raw(scale), True
    )
    return Vec(both).bitcast(fx.Int32)[0]


_SCAN_DPP = (
    (0x111, 0xF, True),
    (0x112, 0xF, True),
    (0x114, 0xF, True),
    (0x118, 0xF, True),
    (0x142, 0xA, False),
    (0x143, 0xC, False),
)


def _wave_incl_scan(x):
    """Inclusive prefix sum of an i32 over the 64 lanes of a wave: DPP row shifts by 1 / 2 / 4 / 8, then
    row_bcast:15 and row_bcast:31 (no LDS)."""
    for ctrl, row_mask, bound in _SCAN_DPP:
        x = x + fx.Int32(update_dpp_i32(0, x, ctrl, row_mask, 0xF, bound))
    return x


def _tensor_addr(t):
    return fx.Int64(fx.ptrtoint(fx.get_iter(t)))


def _gptr(addr, dtype=fx.Int32, alignment=4):
    ptr_ty = fx.PointerType.get(dtype.ir_type, fx.AddressSpace.Global, alignment)
    return fx.inttoptr(ptr_ty, fx.Int64(addr))


def _atomic_add_agent(addr, value):
    """Device-scope fetch-and-add on an i32 word of this GPU; returns the old value."""
    return fx.Int32(
        fx.atomic_add(_gptr(addr), fx.Int32(value), syncscope=fx.rocdl.SyncScope.Agent)
    )


def _atomic_xchg_agent(addr, value):
    """Device-scope exchange on an i32 word of this GPU (a counter reset)."""
    fx.atomic_xchg(_gptr(addr), fx.Int32(value), syncscope=fx.rocdl.SyncScope.Agent)


def _poll(addr):
    """Thread 0's first read of the counter at ``addr`` (device scope; the other lanes' out-of-range offset drops
    theirs), issued before the tile's loads so its latency overlaps them: ``_await`` consumes it.
    """
    buf = _buf(addr, 8)
    # Raw buffer atomic: out-of-range lanes drop their access with no branch (a global atomic needs one), and the
    # typed BufferAtomic copy atom does not return the old value.
    off = (fx.thread_idx.x == 0).select(fx.Int32(0), fx.Int32(0x7FFFFFF0))
    return fx.Int32(
        llvm.call_intrinsic(
            T.i32,
            "llvm.amdgcn.raw.ptr.buffer.atomic.add.i32",
            [
                _raw(fx.Int32(0)),
                _brsrc(buf),
                _raw(off),
                _raw(fx.Int32(0)),
                _raw(fx.Int32(SC1)),
            ],
            [],
            [],
        )
    )


@flyc.jit
def _await(addr, need, got, nap=2):
    """Thread 0 spins from the first read ``got`` until the counter reaches ``need`` (every producer stored its rows
    write-through, then counted); the workgroup waits at a barrier. No acquire fence: see Coherence in the module
    docstring."""
    if fx.thread_idx.x == 0:
        while got < need:
            fx.rocdl.s_sleep(nap)
            got = _atomic_add_agent(addr, 0)
    fx.gpu.barrier()


@flyc.jit
def _signal(addr):
    """Count this workgroup at the counter ``addr``: its stores retire first."""
    fx.rocdl.s_waitcnt(vmcnt=0)
    fx.gpu.barrier()
    if fx.thread_idx.x == 0:
        _atomic_add_agent(addr, 1)


def _sel(vals, idx):
    """``vals[idx]`` of a list of i64 values (the ranks' addresses) by a select chain."""
    r = vals[0]
    for i in range_constexpr(1, len(vals)):
        r = (idx == i).select(vals[i], r)
    return r


def _atomic_sys_add(addr, value):
    """System-scope fetch-and-add on an i32 word (a peer's control word); returns the old value."""
    return fx.Int32(
        fx.atomic_add(_gptr(addr), fx.Int32(value), syncscope=fx.rocdl.SyncScope.OneAs)
    )


def _load_sys(addr):
    """Volatile system-visible i32 load (a counter another rank or workgroup updates)."""
    return fx.Int32(
        fx.generic_load(
            _gptr(addr),
            volatile=True,
            memory_order=fx.AtomicOrdering.Monotonic,
            syncscope=fx.rocdl.SyncScope.OneAs,
        )
    )


def _store128_sys(addr, data):
    """System-scope write-through store of 16 bytes (the caller waits for it with ``s_waitcnt vmcnt(0)``)."""
    # Raw asm: the destination rank differs per lane (no buffer resource covers the peers' pools), and the typed
    # global stores cannot set sc0 sc1 on a 16-byte store without a wait after it: an atomic store is at most
    # 8 bytes, and a volatile one is followed by s_waitcnt vmcnt(0) on gfx950, which serializes the row stores.
    llvm.InlineAsmOp(
        None,
        [_raw(fx.Int64(addr)), _raw(data)],
        "global_store_dwordx4 $0, $1, off sc0 sc1",
        "v,v",
        has_side_effects=True,
    )


def _store64_sys(addr, value):
    """System-scope write-through store of 8 bytes (a peer's inbox or route record)."""
    fx.generic_store(
        _gptr(addr, fx.Int64, 8),
        fx.Int64(value),
        memory_order=fx.AtomicOrdering.Monotonic,
        syncscope=fx.rocdl.SyncScope.OneAs,
    )


@functools.cache
def _retire_fin_for(world):
    @flyc.jit
    def retire(addr, consumers, done, nblk, ctls, me):
        """A GEMM2 tile's last act (its stores retired): counts itself at ``addr + 4``; the last of the block's
        ``consumers`` resets the block's counters and counts the block at ``done``; the last block adds 1 to the
        finish word of this rank on every rank."""
        if fx.thread_idx.x == 0:
            old = _atomic_add_agent(addr + fx.Int64(4), 1)
            if old == consumers - 1:
                _atomic_xchg_agent(addr, 0)
                _atomic_xchg_agent(addr + fx.Int64(4), 0)
                blocks = _atomic_add_agent(done, 1)
                if blocks == nblk - 1:
                    for r in range_constexpr(world):
                        _atomic_sys_add(ctls[r] + (CTL_FIN + 64 * me), 1)

    return retire


@functools.cache
def _gemm1_tile(*, world, epr, kw, pf, bm, ni, nbuf, nwv, hot=False):
    """The GEMM1 tile body ``tile(base, mb, nb, expert, j0, cnt, multi, rx, rxs, meta, W, WS, Q, S, rows, ready)`` of
    block ``mb`` (rows ``j0`` .. ``j0 + bm`` of ``expert``'s route records, ``cnt`` of them valid), column tile
    ``nb`` (``base`` its LDS), and its columns per tile and LDS bytes. ``rx`` / ``rxs`` / ``meta``: addresses of
    the receive rows, scales and route records. ``nbuf`` > 0: the activation rows of the block stream through an
    ``nbuf``-stage LDS ring by async DMA (gathered by arena row, written in MFMA fragment order) and every wave
    reads them from there (``kw`` 1). ``nwv`` waves per workgroup (8: two waves per SIMD, ``bm`` >= 64). The stores
    are write-through and counted at ``ready``."""
    assert (
        kw in (1, 2, 4)
        and bm in (16, 32, 64, 128)
        and ni in (2, 4)
        and (nbuf == 0 or kw == 1 and bm >= 32)
    )
    assert nwv == 4 or bm >= 64 and nbuf > 0 and kw == 1
    nw = nwv // kw
    tn, kt, mr = nw * 16 * ni, KT1 // kw, bm // 16
    agpr = bm in AGPR_BM
    assert kt % 2 == 0 and I % tn == 0  # one weight-scale word covers two K steps
    sag = 4 if kt % 4 == 0 else 2  # K steps per activation-scale load
    nacc = mr * ni * 2
    # the MFMAs of one row tile run under the LDS read of the next
    apipe = nbuf > 0 and bm >= 32
    pitch = bm * 128  # one ring stage: bm rows x one 128-K step
    # 1 KiB DMA chunks of the block's scale tile (bm rows x 160 B)
    sc_chunks = (bm * (H // 32) + 1023) // 1024
    sc_rounds = (sc_chunks + nwv - 1) // nwv
    lds = max(
        4 * nacc * 64 * 16 if kw > 1 else 0,
        nbuf * pitch + (sc_rounds * nwv * 1024 if nbuf else 0),
        16,
    )
    arena = world * TM
    st_kw = {"cm": SC1}

    @flyc.jit
    def tile(
        base, mb, nb, expert, j0, cnt, multi, rx, rxs, meta, W, WS, Q, S, rows, ready
    ):
        tx = fx.thread_idx.x
        lane, wave = tx % 64, tx // 64
        l16, q16 = lane % 16, lane // 16
        nk, kk = wave % nw, wave // nw

        def run(wcm):
            col = nb * tn + nk * 16 * ni
            ar = _buf(rx, arena * H)
            sr = _buf(rxs, arena * (H // 32))
            mr_ = _buf(meta, epr * arena * 8)
            wr = _buf(
                _tensor_addr(W) + fx.Int64(expert) * (2 * I * H // 2), 2 * I * H // 2
            )
            ws = _buf(_tensor_addr(WS), epr * I * (H // 16))
            qr = _buf(_tensor_addr(Q), rows * I)
            ss = _buf(_tensor_addr(S), rows * (I // 32), fx.Int8)

            def row_arena(r):
                j = j0 + r
                ok = j < cnt
                w0 = fx.Int32(_bload(mr_, (expert * arena + j) * 2, cm=SYS, mask=ok))
                return ok.select(w0 & 0xFFFFFF, fx.Int32(arena))

            tok = [row_arena(mi * 16 + l16) for mi in range_constexpr(mr)]
            ad = [
                (tok[mi] * H + kk * (kt * 128) + q16 * 16) // 4
                for mi in range_constexpr(mr)
            ]
            sd = [tok[mi] * (H // 32) // 4 + kk * kt for mi in range_constexpr(mr)]
            wd = [
                [
                    (((col // 16 + j) * 2 + gu) * KT1 * 1024 + q16 * 256 + l16 * 16)
                    // 4
                    + kk * kt * 256
                    for gu in range_constexpr(2)
                ]
                for j in range_constexpr(ni)
            ]
            swd = [
                ((expert * (I // 16) + col // 16 + j) * (KT1 // 2)) * 64
                + q16 * 16
                + l16
                + kk * (kt // 2) * 64
                for j in range_constexpr(ni)
            ]
            if const_expr(nbuf):
                # DMA slot d of a stage: row tile d // 2, K half d % 2; this wave issues d = wave + nwv j
                dma_src = []
                for j in range_constexpr(2 * mr // nwv):
                    d = wave + nwv * j
                    t_d = row_arena((d // 2) * 16 + l16)
                    dma_src.append(t_d * H + (d % 2) * 64 + q16 * 16)

                # scale tile DMA: lane chunk n = (nwv q + wave) * 64 + lane of 16 B, row n // 10 (rows past the tile
                # read a valid row and land in the padding)
                sc_src = []
                for q in range_constexpr(sc_rounds):
                    n = (wave + nwv * q) * 64 + lane
                    r = n // (H // 32 // 16)
                    r = (r < bm).select(r, fx.Int32(bm - 1))
                    t_r = row_arena(r)
                    sc_src.append(t_r * (H // 32) + (n % (H // 32 // 16)) * 16)

                def issue(t):
                    for j in range_constexpr(2 * mr // nwv):
                        _dma(
                            ar,
                            base + (t % nbuf) * pitch + (wave + nwv * j) * 1024,
                            dma_src[j],
                            t * 128,
                            SYS,
                        )
                    if const_expr(t == 0):
                        for q in range_constexpr(sc_rounds):
                            _dma(
                                sr,
                                base + nbuf * pitch + (q * nwv + wave) * 1024,
                                sc_src[q],
                                0,
                                SYS,
                            )
                    fx.rocdl.asyncmark()

            # the activation scales of `sag` K steps: one load per row tile
            sav = [None] * mr
            sbv = [None]  # the weight scales of 2 K steps: one load per column tile

            def load_vec(k):
                if const_expr(nbuf):
                    aa = None
                else:
                    aa = [
                        _load8(ar, ad[mi], k * 128, cm=SYS)
                        for mi in range_constexpr(mr)
                    ]
                sa = None
                if const_expr(not nbuf):
                    if const_expr(k % sag == 0):
                        for mi in range_constexpr(mr):
                            sav[mi] = Vec(_bload(sr, sd[mi], sag, soff=k * 4, cm=SYS))
                    sa = [
                        (fx.Int32(sav[mi][k % sag]) >> (q16 * 8)) & 255
                        for mi in range_constexpr(mr)
                    ]
                bb = [
                    [
                        _bload(wr, wd[j][gu], 4, soff=k * 1024, cm=wcm)
                        for gu in range_constexpr(2)
                    ]
                    for j in range_constexpr(ni)
                ]
                if const_expr(k % 2 == 0):
                    sbv[0] = [
                        _bload(ws, swd[j], soff=(k // 2) * 256)
                        for j in range_constexpr(ni)
                    ]
                return aa, sa, bb, sbv[0]

            def lds_sa(k):
                return [
                    (
                        _lds_load(
                            base + nbuf * pitch + (mi * 16 + l16) * (H // 32) + k * 4
                        )
                        >> (q16 * 8)
                    )
                    & 255
                    for mi in range_constexpr(mr)
                ]

            def load_pieces(k):
                """The loads of ``load_vec(k)`` as callables, issued one by one between the MFMAs of a step."""
                bb = [[None] * 2 for _ in range(ni)]
                pcs = []
                for j in range_constexpr(ni):
                    for gu in range_constexpr(2):

                        def load_b(j=j, gu=gu):
                            bb[j][gu] = _bload(wr, wd[j][gu], 4, soff=k * 1024, cm=wcm)

                        pcs.append(load_b)
                if const_expr(k % 2 == 0):

                    def load_sb():
                        sbv[0] = [
                            _bload(ws, swd[j], soff=(k // 2) * 256)
                            for j in range_constexpr(ni)
                        ]

                    pcs.append(load_sb)

                def finish():
                    return None, None, bb, sbv[0]

                return pcs, finish

            acc = [[[None] * 2 for _ in range(ni)] for _ in range(mr)]
            if const_expr(apipe):

                def lds_frag(k, mi):
                    st = base + (k % nbuf) * pitch
                    return _lds_load8(
                        st, (mi * 2) * 1024 + lane * 16, (mi * 2 + 1) * 1024 + lane * 16
                    )

                for t in range_constexpr(nbuf - 1):
                    issue(t)
                ring = [load_vec(k) for k in range_constexpr(pf)]
                fx.rocdl.wait_asyncmark(min(nbuf - 2, kt - 1))
                fx.gpu.barrier()
                nxt = lds_frag(0, 0)
                issue(nbuf - 1)
                for k in range_constexpr(kt):
                    _, _, bb, sb = ring.pop(0)
                    sa = lds_sa(k)
                    pcs, finish = load_pieces(k + pf) if k + pf < kt else ([], None)
                    # MFMAs between two of the next step's loads
                    gap = max(1, (mr * ni * 2) // max(len(pcs), 1))
                    n_mfma = 0
                    for mi in range_constexpr(mr):
                        cur = nxt
                        # the next row tile's fragment is read while this one's MFMAs run; the last one reads step
                        # k + 1's first, once its stage is in LDS (every wave is done reading step k - 1's)
                        if const_expr(mi + 1 < mr):
                            nxt = lds_frag(k, mi + 1)
                        elif const_expr(k + 1 < kt):
                            fx.rocdl.wait_asyncmark(min(nbuf - 2, kt - 2 - k))
                            fx.rocdl.s_waitcnt(lgkmcnt=0)
                            fx.gpu.barrier()
                            nxt = lds_frag(k + 1, 0)
                            if const_expr(k + nbuf < kt):
                                issue(k + nbuf)
                        for j in range_constexpr(ni):
                            for gu in range_constexpr(2):
                                # the next step's loads, spread over the MFMAs
                                if pcs and n_mfma % gap == 0:
                                    pcs.pop(0)()
                                n_mfma += 1
                                acc[mi][j][gu] = _mfma(
                                    bb[j][gu],
                                    cur,
                                    acc[mi][j][gu],
                                    sb[j],
                                    sa[mi],
                                    (k % 2) * 2 + gu,
                                    agpr=agpr,
                                )
                    for _ in range_constexpr(len(pcs)):
                        pcs.pop(0)()
                    if finish is not None:
                        ring.append(finish())
            else:
                if const_expr(nbuf):
                    for t in range_constexpr(nbuf - 1):
                        issue(t)
                ring = [load_vec(k) for k in range_constexpr(pf)]
                for k in range_constexpr(kt):
                    if const_expr(nbuf):
                        fx.rocdl.wait_asyncmark(min(nbuf - 2, kt - 1 - k))
                        fx.gpu.barrier()
                    if const_expr(k + pf < kt):
                        ring.append(load_vec(k + pf))
                    aa, sa, bb, sb = ring.pop(0)
                    if const_expr(nbuf):
                        sa = lds_sa(k)
                        st = base + (k % nbuf) * pitch
                        aa = [
                            _lds_load8(
                                st,
                                (mi * 2) * 1024 + lane * 16,
                                (mi * 2 + 1) * 1024 + lane * 16,
                            )
                            for mi in range_constexpr(mr)
                        ]
                        if const_expr(k + nbuf - 1 < kt):
                            issue(k + nbuf - 1)
                    for mi in range_constexpr(mr):
                        for j in range_constexpr(ni):
                            for gu in range_constexpr(2):
                                acc[mi][j][gu] = _mfma(
                                    bb[j][gu],
                                    aa[mi],
                                    acc[mi][j][gu],
                                    sb[j],
                                    sa[mi],
                                    (k % 2) * 2 + gu,
                                    agpr=agpr,
                                )
            flat = [
                acc[mi][j][gu] for mi in range(mr) for j in range(ni) for gu in range(2)
            ]
            if const_expr(agpr):
                flat = _hold(flat)
            if const_expr(kw > 1):
                for j in range_constexpr(nacc):
                    _lds_store(
                        base + ((wave * nacc + j) * 64 + lane) * 16,
                        flat[j],
                        fx.Float32,
                        16,
                    )
                fx.rocdl.s_waitcnt(lgkmcnt=0)
                fx.gpu.barrier()
            if kk == 0:
                vals = [Vec(v) for v in flat]
                for k in range_constexpr(1, kw):
                    for j in range_constexpr(nacc):
                        other = _lds_load(
                            base + (((nk + k * nw) * nacc + j) * 64 + lane) * 16,
                            4,
                            fx.Float32,
                            16,
                        )
                        vals[j] = vals[j] + other
                for mi in range_constexpr(mr):
                    rblk = mb * mr + mi
                    for g in range_constexpr(ni // 2):
                        y = [
                            [
                                _clamp_silu_mul(
                                    fx.Float32(vals[mi * ni * 2 + (2 * g + t) * 2][v]),
                                    fx.Float32(
                                        vals[mi * ni * 2 + (2 * g + t) * 2 + 1][v]
                                    ),
                                )
                                for v in range_constexpr(4)
                            ]
                            for t in range_constexpr(2)
                        ]
                        amax = fx.math.absf(y[0][0])
                        for t in range_constexpr(2):
                            for v in range_constexpr(4):
                                z = fx.math.absf(y[t][v])
                                amax = (amax > z).select(amax, z)
                        for sh in range_constexpr(2):
                            z = amax.shuffle_xor(16 << sh, 64)
                            amax = (amax > z).select(amax, z)
                        e = _e8m0(amax)
                        scale = (e << 23).bitcast(fx.Float32)
                        for t in range_constexpr(2):
                            pk = _fp8x4(*y[t], scale)
                            c16 = col // 16 + 2 * g + t  # 16-column tile
                            off = (
                                rblk * (KT2 * 2048)
                                + (c16 // 8) * 2048
                                + ((c16 % 8) // 4) * 1024
                                + ((c16 % 4) * 16 + l16) * 16
                                + q16 * 4
                            )
                            _bstore(qr, off // 4, pk, mask=tok[mi] < arena, **st_kw)
                        sg = col // 32 + g
                        s_off = rblk * (KT2 * 64) + (sg // 4) * 64 + l16 * 4 + sg % 4
                        _bstore(
                            ss,
                            s_off,
                            e.to(fx.Int8),
                            mask=(tok[mi] < arena) & (q16 == 0),
                            **st_kw,
                        )

        if const_expr(hot):
            if multi:
                run(0)
            else:
                run(2)
        else:
            run(2)
        _signal(ready + fx.Int64(mb) * READY_STRIDE)

    return tile, tn, lds


@functools.cache
def _gemm2_tile(*, world, epr, tn, bm, pf, nbuf, signal, hot, nwv):
    """The GEMM2 tile body ``tile(base, mb, nb, expert, j0, cnt, multi, meta, Q, QS, W, WS, pools, rows, ready, done,
    nblk, ctls, me)`` of block ``mb`` (rows ``j0`` .. of ``expert``'s route records, ``cnt`` valid) and column tile
    ``nb``, and its LDS bytes. ``signal`` = GEMM1's tiles per block: the tile waits for them at ``ready`` after its
    weight loads are in flight. The route-weighted bf16 rows go through an LDS tile to 16-byte write-through stores
    of whole rows into row ``slot`` of token ``token`` in the combine inbox of the row's source rank (``pools`` the
    ranks' pool addresses). ``multi``: the expert has several blocks (``hot``: its weights are read from the cache).
    ``nwv`` waves per workgroup (8: two waves per SIMD, ``bm`` >= 64, ``tn`` 256)."""
    assert (
        tn in (128, 256, 512)
        and bm in (16, 32, 64, 128)
        and H % tn == 0
        and (nbuf == 0 or bm >= 32)
    )
    assert nwv == 4 and tn <= 256 or bm >= 64 and nbuf > 0 and tn >= 256
    ni, mr = tn // (16 * nwv), bm // 16
    agpr = bm in AGPR_BM
    tpr = tn // 8  # threads of an output row, 16 bytes each
    rpp = 64 * nwv // tpr  # rows per store pass
    pitch = bm * 128
    # the block's activation scale tile, staged in LDS
    sc_bytes = (bm * (I // 32) + 1023) // 1024 * 1024
    # bytes per row of the bf16 output tile: the pad rotates the 16 rows of a store over the banks
    row_b = tn * 2 + 16
    lds = max(nbuf * pitch + (sc_bytes if nbuf else 0), bm * row_b, 16)
    # the MFMAs of one row tile run under the LDS read of the next
    apipe = nbuf > 0 and bm >= 32
    arena = world * TM
    slots_off = pool_layout(world, epr)["slots"]
    retire = _retire_fin_for(world)

    @flyc.jit
    def tile(
        base,
        mb,
        nb,
        expert,
        j0,
        cnt,
        multi,
        meta,
        A,
        AS,
        W,
        WS,
        pools,
        rows,
        ready,
        done,
        nblk,
        ctls,
        me,
    ):
        tx = fx.thread_idx.x
        lane, wave = tx % 64, tx // 64
        l16, q16 = lane % 16, lane // 16

        def run(wcm):
            ar = _buf(_tensor_addr(A), rows * I)
            sr = _buf(_tensor_addr(AS), rows * (I // 32))
            wr = _buf(_tensor_addr(W) + fx.Int64(expert) * (H * I // 2), H * I // 2)
            sw = _buf(_tensor_addr(WS), epr * H * (I // 32))
            mr_ = _buf(meta, epr * arena * 8)
            ad = [
                ((mb * mr + mi) * (KT2 * 2048) + lane * 16) // 4
                for mi in range_constexpr(mr)
            ]
            sd = [(mb * mr + mi) * (KT2 * 16) + l16 for mi in range_constexpr(mr)]
            wd = [
                ((nb * (tn // 16) + wave * ni + j) * KT2 * 1024 + q16 * 256 + l16 * 16)
                // 4
                for j in range_constexpr(ni)
            ]
            swd = [
                (
                    (expert * (H // 32) + nb * (tn // 32) + (wave * ni + j) // 2)
                    * (KT2 // 2)
                )
                * 64
                + q16 * 16
                + l16
                for j in range_constexpr(ni)
            ]
            if const_expr(nbuf):
                # DMA slot d of a stage: row tile d // 2, half d % 2; this wave issues d = wave + nwv j
                dma_src = [
                    (
                        (mb * mr + (wave + nwv * j) // 2) * (KT2 * 2048)
                        + ((wave + nwv * j) % 2) * 1024
                        + lane * 16
                    )
                    for j in range_constexpr(2 * mr // nwv)
                ]

                def issue(t):
                    for j in range_constexpr(2 * mr // nwv):
                        _dma(
                            ar,
                            base + (t % nbuf) * pitch + (wave + nwv * j) * 1024,
                            dma_src[j],
                            t * 2048,
                        )
                    if const_expr(t == 0):
                        for q in range_constexpr((sc_bytes // 1024 + nwv - 1) // nwv):
                            if const_expr((q + 1) * nwv <= sc_bytes // 1024):
                                _dma(
                                    sr,
                                    base + nbuf * pitch + (q * nwv + wave) * 1024,
                                    lane * 16,
                                    mb * bm * (I // 32) + (q * nwv + wave) * 1024,
                                )
                            else:
                                if wave < sc_bytes // 1024 - q * nwv:
                                    _dma(
                                        sr,
                                        base + nbuf * pitch + (q * nwv + wave) * 1024,
                                        lane * 16,
                                        mb * bm * (I // 32) + (q * nwv + wave) * 1024,
                                    )
                    fx.rocdl.asyncmark()

            sbv = [None]  # the weight scales of 2 K steps: one load per column tile

            def load_w(k):
                bb = [
                    _bload(wr, wd[j], 4, soff=k * 1024, cm=wcm)
                    for j in range_constexpr(ni)
                ]
                if const_expr(k % 2 == 0):
                    sbv[0] = [
                        _bload(sw, swd[j], soff=(k // 2) * 256)
                        for j in range_constexpr(ni)
                    ]
                return bb, sbv[0]

            def load_a(k):
                if const_expr(nbuf):
                    return None, None
                aa = [_load8(ar, ad[mi], k * 2048, 1024) for mi in range_constexpr(mr)]
                sa = [
                    (fx.Int32(_bload(sr, sd[mi], soff=k * 64)) >> (q16 * 8)) & 255
                    for mi in range_constexpr(mr)
                ]
                return aa, sa

            def lds_sa(k):
                return [
                    (
                        _lds_load(base + nbuf * pitch + ((mi * KT2 + k) * 16 + l16) * 4)
                        >> (q16 * 8)
                    )
                    & 255
                    for mi in range_constexpr(mr)
                ]

            acc = [[None] * ni for _ in range(mr)]

            def meta_word(row, which):
                j = j0 + row
                return fx.Int32(
                    _bload(mr_, (expert * arena + j) * 2 + which, cm=SYS, mask=j < cnt)
                )

            epi_w = [
                meta_word(mi * 16 + l16, 1).bitcast(fx.Float32)
                for mi in range_constexpr(mr)
            ]
            epi_dst = []
            for rr in range_constexpr(bm // rpp):
                row = rr * rpp + tx // tpr
                ok = (j0 + row) < cnt
                w0 = meta_word(row, 0)
                token_row, slot = w0 & 0xFFFFFF, w0.shrui(fx.Int32(24))
                src = ok.select(token_row // TM, fx.Int32(0))
                epi_dst.append(
                    (
                        ok,
                        _sel(pools, src)
                        + slots_off
                        + ((fx.Int64(token_row % TM) * TOPK + fx.Int64(slot)) * H) * 2,
                    )
                )
            counter = ready + fx.Int64(mb) * READY_STRIDE
            got = _poll(counter)
            wring = [load_w(k) for k in range_constexpr(pf)]
            _await(counter, signal, got)
            if const_expr(nbuf):
                for t in range_constexpr(nbuf - 1):
                    issue(t)
            aring = [load_a(k) for k in range_constexpr(pf)]

            def lds_frag(k, mi):
                st = base + (k % nbuf) * pitch
                return _lds_load8(
                    st, (mi * 2) * 1024 + lane * 16, (mi * 2 + 1) * 1024 + lane * 16
                )

            if const_expr(apipe):
                fx.rocdl.wait_asyncmark(min(nbuf - 2, KT2 - 1))
                fx.gpu.barrier()
                nxt = lds_frag(0, 0)
                issue(nbuf - 1)
            for k in range_constexpr(KT2):
                if const_expr(nbuf and not apipe):
                    fx.rocdl.wait_asyncmark(min(nbuf - 2, KT2 - 1 - k))
                    fx.gpu.barrier()
                if const_expr(k + pf < KT2):
                    wring.append(load_w(k + pf))
                    aring.append(load_a(k + pf))
                bb, sb = wring.pop(0)
                aa, sa = aring.pop(0)
                if const_expr(nbuf):
                    sa = lds_sa(k)
                if const_expr(apipe):
                    for mi in range_constexpr(mr):
                        cur = nxt
                        # the next row tile's fragment is read while this one's MFMAs run; the last one reads step
                        # k + 1's first, once its stage is in LDS
                        if const_expr(mi + 1 < mr):
                            nxt = lds_frag(k, mi + 1)
                        elif const_expr(k + 1 < KT2):
                            fx.rocdl.wait_asyncmark(min(nbuf - 2, KT2 - 2 - k))
                            fx.rocdl.s_waitcnt(lgkmcnt=0)
                            fx.gpu.barrier()
                            nxt = lds_frag(k + 1, 0)
                            if const_expr(k + nbuf < KT2):
                                issue(k + nbuf)
                        for j in range_constexpr(ni):
                            acc[mi][j] = _mfma(
                                bb[j],
                                cur,
                                acc[mi][j],
                                sb[j],
                                sa[mi],
                                (k % 2) * 2 + j % 2,
                                agpr=agpr,
                            )
                else:
                    if const_expr(nbuf):
                        aa = [lds_frag(k, mi) for mi in range_constexpr(mr)]
                        if const_expr(k + nbuf - 1 < KT2):
                            issue(k + nbuf - 1)
                    for mi in range_constexpr(mr):
                        for j in range_constexpr(ni):
                            acc[mi][j] = _mfma(
                                bb[j],
                                aa[mi],
                                acc[mi][j],
                                sb[j],
                                sa[mi],
                                (k % 2) * 2 + j % 2,
                                agpr=agpr,
                            )
            flat = [acc[mi][j] for mi in range(mr) for j in range(ni)]
            if const_expr(agpr):
                flat = _hold(flat)
            fx.gpu.barrier()
            for mi in range_constexpr(mr):
                weight = epi_w[mi]
                for j in range_constexpr(ni):
                    v = Vec(flat[mi * ni + j])
                    lo = Vec.from_elements(
                        [v[0] * weight, v[1] * weight], fx.Float32
                    ).to(fx.BFloat16)
                    hi = Vec.from_elements(
                        [v[2] * weight, v[3] * weight], fx.Float32
                    ).to(fx.BFloat16)
                    col = wave * (16 * ni) + j * 16 + q16 * 4
                    _lds_store(
                        base + (mi * 16 + l16) * row_b + col * 2,
                        Vec.from_elements(
                            [lo.bitcast(fx.Int32)[0], hi.bitcast(fx.Int32)[0]],
                            fx.Int32,
                        ),
                        align=8,
                    )
            fx.rocdl.s_waitcnt(lgkmcnt=0)
            fx.gpu.barrier()
            for rr in range_constexpr(bm // rpp):
                row = rr * rpp + tx // tpr
                ok, dst = epi_dst[rr]
                col = tx % tpr * 8
                v = _lds_load(base + row * row_b + col * 2, 4, align=16)
                if ok:
                    _store128_sys(dst + (nb * tn + col) * 2, v)
            fx.rocdl.s_waitcnt(vmcnt=0)
            fx.gpu.barrier()
            retire(counter, H // tn, done, nblk, ctls, me)

        # an expert with several blocks reads its weights once per block: cached loads let concurrent blocks
        # share them in L2, a lone block streams them non-temporally
        if const_expr(hot):
            if multi:
                run(0)
            else:
                run(2)
        else:
            run(2)

    return tile, lds


# workgroups per token in the combine: one 8-column chunk per thread, 160 threads
REDUCE_SPLIT = 4


@functools.cache
def _reduce_tile(*, world, epr):
    """The combine body ``tile(token, split, TID, slots, OUT, m)``: columns ``split * 1280`` .. of the bf16 inbox rows
    of ``token`` summed in slot order in FP32 and rounded once to ``OUT[token]``; a slot whose expert id is outside
    [0, epr * world) counts as zero."""
    cols = 8  # bf16 per thread
    per = H // cols // REDUCE_SPLIT  # chunks per split

    @flyc.jit
    def tile(token, split, TID, slots, OUT, m):
        tx = fx.thread_idx.x
        sr = _buf(slots, TM * TOPK * H * 2)
        outr = _buf(_tensor_addr(OUT), m * H * 2)
        c = split * per + tx
        ok = tx < per
        raw = []
        for s in range_constexpr(TOPK):
            e_s = fx.Int32(TID[token * TOPK + s])
            routed = (e_s >= 0) & (e_s < epr * world)
            raw.append(
                Vec(
                    _bload(
                        sr,
                        ((token * TOPK + s) * H + c * cols) // 2,
                        4,
                        cm=SYS,
                        mask=ok & routed,
                    )
                )
            )
        acc = [None] * cols
        for s in range_constexpr(TOPK):
            for j in range_constexpr(cols // 2):
                dw = fx.Int32(raw[s][j])
                halves = (
                    (dw << 16).bitcast(fx.Float32),
                    (dw & fx.Int32(-65536)).bitcast(fx.Float32),
                )
                for h, v in enumerate(halves):
                    acc[j * 2 + h] = v if s == 0 else acc[j * 2 + h] + v
        packed = [
            Vec.from_elements([acc[2 * j], acc[2 * j + 1]], fx.Float32)
            .to(fx.BFloat16)
            .bitcast(fx.Int32)[0]
            for j in range_constexpr(cols // 2)
        ]
        _bstore(
            outr,
            (token * H + c * cols) // 2,
            Vec.from_elements(packed, fx.Int32),
            mask=ok,
        )

    return tile


def _uniform(v):
    """An i32 value that is the same in every lane, in SGPRs."""
    return fx.Int32(fx.rocdl.readfirstlane(T.i32, _raw(fx.Int32(v))))


def _store32_sys(addr, value):
    fx.generic_store(
        _gptr(addr),
        fx.Int32(value),
        memory_order=fx.AtomicOrdering.Monotonic,
        syncscope=fx.rocdl.SyncScope.OneAs,
    )


PLAN_IDS = 1024  # byte offset of the dispatch's route ids in the plan region of LDS
PLAN_BYTES = 2048


@functools.cache
def _dispatch_phase(*, world, epr, nd, nt, quant_in):
    """The dispatch of workgroup ``bid`` (tokens ``bid * tc`` .. of ``tc = ceil(m / nd)``). The last wave claims a
    record slot on the owner of every route of the workgroup's tokens through a remote counter and writes the record
    there (its atomics do not hold up the others); the other waves send each token's fp8 row and e8m0 scales to
    the arena row ``me * TM + token`` of every distinct rank that owns one of its experts (a thread owns 16 columns,
    two threads one scale group, eight threads one scale word). The workgroup then adds 1 to the ``arrive`` word of
    every rank."""
    layout = pool_layout(world, epr)
    pw = nt - 64  # threads that send rows
    chunks = H // 16
    passes = (chunks + pw - 1) // pw
    max_tok = (TM + nd - 1) // nd
    assert max_tok * TOPK <= 64

    @flyc.jit
    def dispatch(base, bid, me, m, ep, par, X, XS, TID, TW, pools, ctls, loc):
        tx = fx.thread_idx.x
        ids_lds = base + PLAN_IDS
        tc = (m + (nd - 1)) // nd
        tok0 = bid * tc
        ntok = (m - tok0 < tc).select(m - tok0, tc)
        ntok = (ntok > 0).select(ntok, fx.Int32(0))
        tidr = _buf(_tensor_addr(TID), m * TOPK * 4)
        twr = _buf(_tensor_addr(TW), m * TOPK * 4)
        xr = _buf(_tensor_addr(X), m * (H * 2 if quant_in else H))
        if const_expr(not quant_in):
            xsr = _buf(_tensor_addr(XS), m * (H // 32))
        r = tx - pw
        tt, k = r // TOPK, r % TOPK
        ok = (tt < ntok) & (r < max_tok * TOPK)
        idx = (tok0 + tt) * TOPK + k
        eid = fx.Int32(0)
        wbits = fx.Int32(0)
        if tx >= pw:
            eid = fx.Int32(_bload(tidr, idx, mask=ok))
            wbits = fx.Int32(_bload(twr, idx, mask=ok))
            _lds_store(ids_lds + r * 4, eid)

        def load_row(token, live):
            res = []
            for p in range_constexpr(passes):
                c = tx + p * pw
                msk = live & (c < chunks)
                if const_expr(quant_in):
                    for h in range_constexpr(2):
                        res.append(
                            Vec(
                                _bload(
                                    xr, (token * H + c * 16) // 2 + 4 * h, 4, mask=msk
                                )
                            )
                        )
                else:
                    res.append(Vec(_bload(xr, (token * H + c * 16) // 4, 4, mask=msk)))
            return res

        per_row = 2 if quant_in else 1
        cur0 = load_row(tok0, (ntok > 0) & (tx < pw))
        fx.rocdl.s_waitcnt(lgkmcnt=0)
        fx.gpu.barrier()
        if tx >= pw:
            valid = ok & (eid >= 0) & (eid < epr * world)
            if valid:
                dest = eid // epr
                le = eid - dest * epr
                pos = _atomic_sys_add(
                    _sel(ctls, dest) + (CTL_CNT + (par * 64 + le) * CNT_STRIDE), 1
                )
                rec = (fx.Int64(fx.Uint32(wbits)) << 32) | fx.Int64(
                    (me * TM + tok0 + tt) | (k << 24)
                )
                _store64_sys(
                    _sel(pools, dest)
                    + layout["meta"]
                    + (fx.Int64(le) * (world * TM) + fx.Int64(pos)) * 8,
                    rec,
                )
        if tx < pw:
            cur = cur0
            for i in range_constexpr(max_tok):
                token = tok0 + i
                nxt = (
                    load_row(token + 1, (ntok > i + 1) & (tx < pw))
                    if i + 1 < max_tok
                    else None
                )
                if i < ntok:
                    ids = [
                        _uniform(_lds_load(ids_lds + (i * TOPK + kk) * 4))
                        for kk in range_constexpr(TOPK)
                    ]
                    vld = [
                        (ids[kk] >= 0) & (ids[kk] < epr * world)
                        for kk in range_constexpr(TOPK)
                    ]
                    dst = [
                        vld[kk].select(ids[kk] // epr, fx.Int32(0))
                        for kk in range_constexpr(TOPK)
                    ]
                    held = []
                    for p in range_constexpr(passes):
                        c = tx + p * pw
                        okc = c < chunks
                        if const_expr(quant_in):
                            v = []
                            for h in range_constexpr(2):
                                for j in range_constexpr(4):
                                    dw = fx.Int32(cur[p * per_row + h][j])
                                    v += [
                                        (dw << 16).bitcast(fx.Float32),
                                        (dw & fx.Int32(-65536)).bitcast(fx.Float32),
                                    ]
                            amax = fx.math.absf(v[0])
                            for j in range_constexpr(1, 16):
                                z = fx.math.absf(v[j])
                                amax = (amax > z).select(amax, z)
                            z = amax.shuffle_xor(1, 64)
                            amax = (amax > z).select(amax, z)
                            e = _e8m0(amax)
                            scale = (e << 23).bitcast(fx.Float32)
                            w = [
                                _fp8x4(*v[4 * q : 4 * q + 4], scale)
                                for q in range_constexpr(4)
                            ]
                            code = Vec.from_elements(w, fx.Int32)
                            e1, e2, e3 = (
                                e.shuffle_xor(2, 64),
                                e.shuffle_xor(4, 64),
                                e.shuffle_xor(6, 64),
                            )
                            word = (
                                (e & 255)
                                | ((e1 & 255) << 8)
                                | ((e2 & 255) << 16)
                                | ((e3 & 255) << 24)
                            )
                        else:
                            code = cur[p]
                            word = fx.Int32(
                                _bload(
                                    xsr,
                                    token * (H // 128) + c // 8,
                                    mask=okc & (tx % 8 == 0),
                                )
                            )
                        held.append((c, okc, code, word))
                    row = me * TM + token
                    for kk in range_constexpr(TOPK):
                        dup = fx.Boolean(False)
                        for j in range_constexpr(kk):
                            dup = dup | (vld[j] & (dst[j] == dst[kk]))
                        go = vld[kk] & ~dup
                        if go:
                            dbase = _sel(pools, dst[kk])
                            rxr = _buf(dbase + layout["rx"], world * TM * H)
                            rsr = _buf(dbase + layout["rxs"], world * TM * (H // 32))
                            for c, okc, code, word in held:
                                _bstore(
                                    rxr, (row * H + c * 16) // 4, code, cm=SYS, mask=okc
                                )
                                _bstore(
                                    rsr,
                                    row * (H // 128) + c // 8,
                                    word,
                                    cm=SYS,
                                    mask=okc & (tx % 8 == 0),
                                )
                if i + 1 < max_tok:
                    cur = nxt
        fx.rocdl.s_waitcnt(vmcnt=0)
        fx.gpu.barrier()
        if tx == 0:
            last = fx.Int32(0)
            old = _atomic_add_agent(
                loc + (LOC_ARR_G + 16 * (bid % ARRIVE_GROUPS)) * 4,
                1,
            )
            if old == nd // ARRIVE_GROUPS - 1:
                old2 = _atomic_add_agent(loc + LOC_ARR * 4, 1)
                last = (old2 == ARRIVE_GROUPS - 1).select(fx.Int32(1), fx.Int32(0))
            _lds_store(base + 520, last)
        fx.rocdl.s_waitcnt(lgkmcnt=0)
        fx.gpu.barrier()
        if fx.Int32(_lds_load(base + 520)) == 1:
            if tx < world:
                _store32_sys(_sel(ctls, tx) + (CTL_ARRIVE + 64 * me), ep + 1)
            if tx < ARRIVE_GROUPS:
                _atomic_xchg_agent(loc + (LOC_ARR_G + 16 * tx) * 4, 0)
            if tx == 0:
                _atomic_xchg_agent(loc + LOC_ARR * 4, 0)

    return dispatch


@functools.cache
def compile_fused(
    *,
    device,
    world,
    epr,
    nd,
    nw,
    bm,
    kw,
    pf1,
    ni,
    nb1,
    tn2,
    pf2,
    nb2,
    hot,
    nwv,
    quant_in=True,
):
    """One persistent launch of ``nw`` workgroups: dispatch (the first ``nd`` of them), GEMM1, GEMM2, combine.
    ``device`` (an index) only keys the cache: a compiled handle launches on the device it was compiled on.
    """
    nt = 64 * nwv
    g1, tn1, lds1 = _gemm1_tile(
        world=world, epr=epr, kw=kw, pf=pf1, bm=bm, ni=ni, nbuf=nb1, nwv=nwv, hot=hot
    )
    c1 = I // tn1
    g2, lds2 = _gemm2_tile(
        world=world,
        epr=epr,
        tn=tn2,
        bm=bm,
        pf=pf2,
        nbuf=nb2,
        signal=c1,
        hot=hot,
        nwv=nwv,
    )
    c2 = H // tn2
    dispatch = _dispatch_phase(world=world, epr=epr, nd=nd, nt=nt, quant_in=quant_in)
    red = _reduce_tile(world=world, epr=epr)
    layout = pool_layout(world, epr)
    assert epr <= NE_MAX and nd <= nw and nd % ARRIVE_GROUPS == 0
    lds = max(lds1, lds2)
    name = (
        f"moe_fused_ep{world}x{epr}_a8w4_h5120_i2304_bm{bm}_g1_tn{tn1}_kw{kw}_pf{pf1}_r{nb1}_g2_tn{tn2}_pf{pf2}_r{nb2}"
        f"_nd{nd}_nw{nw}_w{nwv}" + ("" if quant_in else "_fp8in")
    )

    @fx.struct
    class Shared:
        raw: fx.Array[fx.Uint8, lds + PLAN_BYTES, 16]

    @flyc.kernel(name=name, known_block_size=[nt, 1, 1])
    def kernel(
        X: fx.Tensor,
        XS: fx.Tensor,
        TID: fx.Tensor,
        TW: fx.Tensor,
        W13: fx.Tensor,
        WS13: fx.Tensor,
        Q: fx.Tensor,
        S: fx.Tensor,
        W2: fx.Tensor,
        WS2: fx.Tensor,
        OUT: fx.Tensor,
        LOC: fx.Tensor,
        READY: fx.Tensor,
        me: fx.Int32,
        m: fx.Int32,
        rows: fx.Int32,
        p0: fx.Int64,
        p1: fx.Int64,
        p2: fx.Int64,
        p3: fx.Int64,
        p4: fx.Int64,
        p5: fx.Int64,
        p6: fx.Int64,
        p7: fx.Int64,
        c0: fx.Int64,
        c1_: fx.Int64,
        c2_: fx.Int64,
        c3: fx.Int64,
        c4: fx.Int64,
        c5: fx.Int64,
        c6: fx.Int64,
        c7: fx.Int64,
    ):
        pools = [p0, p1, p2, p3, p4, p5, p6, p7][:world]
        ctls = [c0, c1_, c2_, c3, c4, c5, c6, c7][:world]
        smem = fx.SharedAllocator().allocate(Shared).peek()
        base = fx.Int32(fx.ptrtoint(smem.raw.ptr))
        plan = base + lds
        tx = fx.thread_idx.x
        lane, wave = tx % 64, fx.Int32(fx.rocdl.readfirstlane(T.i32, tx // 64))
        bid = fx.Int32(fx.block_idx.x)
        loc = _tensor_addr(LOC)
        ready = _tensor_addr(READY)
        pool = _sel(pools, me)
        ctl = _sel(ctls, me)
        xcd = hw_xcc_id()
        qh = loc + (LOC_QH + 16 * xcd) * 4
        done = loc + LOC_DONE * 4
        ep = fx.Int32(fx.rocdl.readfirstlane(T.i32, _raw(fx.Int32(LOC[LOC_EP]))))
        par = ep & 1
        if bid < nd:
            dispatch(plan, bid, me, m, ep, par, X, XS, TID, TW, pools, ctls, loc)

        def pop():
            """Thread 0: the next work item. Items [0, nw) come from the queue of this XCD (``8 k + xcd``), the later
            ones from one global queue: the first wave of pops does not contend on one word, and a workgroup that
            has started never waits for an item that no running workgroup can take."""
            it = xcd + XCDS * _atomic_add_agent(qh, 1)
            if it >= nw:
                it = nw + _atomic_add_agent(loc + LOC_QG * 4, 1)
            return it

        first = fx.Int32(0)
        if tx == 0:
            first = pop()
        if wave == 0:
            got = (lane < world).select(_load_sys(ctl + CTL_ARRIVE + lane * 64), ep + 1)
            while fx.rocdl.ballot(T.i64, got < ep + 1) != 0:
                fx.rocdl.s_sleep(1)
                got = (lane < world).select(
                    _load_sys(ctl + CTL_ARRIVE + lane * 64), ep + 1
                )
        fx.gpu.barrier()
        if wave == 0:
            c = (lane < epr).select(
                _load_sys(ctl + CTL_CNT + (par * 64 + lane) * CNT_STRIDE), fx.Int32(0)
            )
            blocks = (c + (bm - 1)) // bm
            end = _wave_incl_scan(blocks)
            _lds_store(plan + lane * 4, c)
            _lds_store(plan + 256 + lane * 4, end)
        if tx == 0:
            _lds_store(plan + 512, first)
        fx.rocdl.s_waitcnt(lgkmcnt=0)
        fx.gpu.barrier()

        def plan_word(i):
            return fx.Int32(
                fx.rocdl.readfirstlane(
                    T.i32,
                    _raw(fx.Int32(_lds_load(plan + i * 4))),
                )
            )

        nblk = plan_word(64 + 63)
        n1 = nblk * c1
        total = n1 + nblk * c2
        meta = pool + layout["meta"]
        rx = pool + layout["rx"]
        rxs = pool + layout["rxs"]

        if (nblk == 0) & (bid == 0) & (tx < world):
            _atomic_sys_add(_sel(ctls, tx) + (CTL_FIN + 64 * me), 1)

        item = plan_word(128)
        while item < total:
            nxt = fx.Int32(0)
            if tx == 0:
                nxt = pop()
            is1 = item < n1
            jj = is1.select(item, item - n1)
            per = is1.select(fx.Int32(c1), fx.Int32(c2))
            mb = jj // per
            nbt = jj - mb * per
            e = fx.Int32(0)
            for sh in range_constexpr(5, -1, -1):
                cand = e + (1 << sh)
                e = ((cand <= epr) & (plan_word(64 + cand - 1) <= mb)).select(cand, e)
            fb = (e == 0).select(
                fx.Int32(0), plan_word(64 + (e > 0).select(e - 1, fx.Int32(0)))
            )
            j0 = (mb - fb) * bm
            cnt = plan_word(e)
            multi = (plan_word(64 + e) - fb) > 1
            if is1:
                g1(
                    base,
                    mb,
                    nbt,
                    e,
                    j0,
                    cnt,
                    multi,
                    rx,
                    rxs,
                    meta,
                    W13,
                    WS13,
                    Q,
                    S,
                    rows,
                    ready,
                )
            else:
                g2(
                    base,
                    mb,
                    nbt,
                    e,
                    j0,
                    cnt,
                    multi,
                    meta,
                    Q,
                    S,
                    W2,
                    WS2,
                    pools,
                    rows,
                    ready,
                    done,
                    nblk,
                    ctls,
                    me,
                )
            fx.gpu.barrier()
            if tx == 0:
                _lds_store(plan + 512, nxt)
            fx.rocdl.s_waitcnt(lgkmcnt=0)
            fx.gpu.barrier()
            item = plan_word(128)

        if tx < world:
            got = _load_sys(ctl + CTL_FIN + tx * 64)
            while got < ep + 1:
                fx.rocdl.s_sleep(1)
                got = _load_sys(ctl + CTL_FIN + tx * 64)
        fx.gpu.barrier()
        for t in range(bid, m * REDUCE_SPLIT, fx.Int32(nw)):
            red(
                t // REDUCE_SPLIT, t % REDUCE_SPLIT, TID, pool + layout["slots"], OUT, m
            )

        fx.gpu.barrier()
        if tx == 0:
            last = fx.Int32(0)
            old = _atomic_add_agent(
                loc + (LOC_RETIRE_G + 16 * (bid % RETIRE_GROUPS)) * 4,
                1,
            )
            if old == nw // RETIRE_GROUPS - 1:
                old2 = _atomic_add_agent(loc + LOC_RETIRE * 4, 1)
                last = (old2 == RETIRE_GROUPS - 1).select(fx.Int32(1), fx.Int32(0))
            _lds_store(plan + 516, last)
        fx.rocdl.s_waitcnt(lgkmcnt=0)
        fx.gpu.barrier()
        if plan_word(129) == 1:
            if tx < epr:
                _store32_sys(ctl + CTL_CNT + (par * 64 + tx) * CNT_STRIDE, 0)
            if tx < RETIRE_GROUPS:
                _atomic_xchg_agent(loc + (LOC_RETIRE_G + 16 * tx) * 4, 0)
            if tx < XCDS:
                _atomic_xchg_agent(loc + (LOC_QH + 16 * tx) * 4, 0)
            if tx == 0:
                _atomic_xchg_agent(done, 0)
                _atomic_xchg_agent(loc + LOC_RETIRE * 4, 0)
                _atomic_xchg_agent(loc + LOC_QG * 4, 0)
                _atomic_add_agent(loc + LOC_EP * 4, 1)

    @flyc.jit
    def launch(
        X: fx.Tensor,
        XS: fx.Tensor,
        TID: fx.Tensor,
        TW: fx.Tensor,
        W13: fx.Tensor,
        WS13: fx.Tensor,
        Q: fx.Tensor,
        S: fx.Tensor,
        W2: fx.Tensor,
        WS2: fx.Tensor,
        OUT: fx.Tensor,
        LOC: fx.Tensor,
        READY: fx.Tensor,
        me: fx.Int32,
        m: fx.Int32,
        rows: fx.Int32,
        p0: fx.Int64,
        p1: fx.Int64,
        p2: fx.Int64,
        p3: fx.Int64,
        p4: fx.Int64,
        p5: fx.Int64,
        p6: fx.Int64,
        p7: fx.Int64,
        c0: fx.Int64,
        c1_: fx.Int64,
        c2_: fx.Int64,
        c3: fx.Int64,
        c4: fx.Int64,
        c5: fx.Int64,
        c6: fx.Int64,
        c7: fx.Int64,
        stream: fx.Stream,
    ):
        kernel(
            X,
            XS,
            TID,
            TW,
            W13,
            WS13,
            Q,
            S,
            W2,
            WS2,
            OUT,
            LOC,
            READY,
            me,
            m,
            rows,
            p0,
            p1,
            p2,
            p3,
            p4,
            p5,
            p6,
            p7,
            c0,
            c1_,
            c2_,
            c3,
            c4,
            c5,
            c6,
            c7,
        ).launch(grid=(nw, 1, 1), block=(nt, 1, 1), stream=stream)

    launch.kernel_name = name
    return launch
