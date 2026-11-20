# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Routed-expert kernels: MXFP8 activations x MXFP4 weights on the gfx950 scaled MFMA, hidden 5120, intermediate 2304,
top-3 or top-6 routing, up to ``NE_MAX`` local experts.

Rows arrive sorted by expert in blocks of ``bm`` rows (``compile_sort``); a block never mixes experts, a padding row
names token ``n_tokens`` and loads zeros. ``compile_gemm1``: a workgroup owns ``bm`` rows x ``tn`` intermediate
columns, one wave owns ``16 * ni`` columns with their whole quant groups (gate and up tiles interleaved per 16
columns in the weight layout); the ``kw`` K-waves sum their FP32 partials through LDS in rank order, then clamp-SwiGLU
in FP32 and the MXFP8 quant are written as e4m3 codes and e8m0 scales of the sorted row (16-row blocks, per K step one
word of four scales per row). ``compile_gemm2``: a workgroup owns ``bm`` rows x ``tn`` columns over the whole K and
writes ``bf16(route weight * y)`` to the (token, slot) row through a bf16 LDS tile, so every output element has one
writer. ``compile_combine`` sums a token's ``topk`` slots in slot order.

The K loops: the weights go straight into VGPRs (one 1 KB fragment per wave and 16 columns), the activation rows
through an LDS ring by async DMA, the activation scales through LDS (one DMA per tile), the next step's loads are
issued between the MFMAs of this one, and the next row tile's fragment is read under the MFMAs of this one. ``nwv`` 8
runs two waves per SIMD. The local expert count ``ne`` only sizes counters and tables.

``compile_experts`` runs the three in one launch: workgroups [0, N1) are GEMM1 tiles, then GEMM2 tiles, then
``COMBINE_WGS`` combine workgroups of ``cw`` tokens, so a GEMM2 tile starts on the CUs GEMM1 leaves while its last
tiles run. A GEMM1 workgroup stores its rows write-through and counts itself in its block's ready counter; a GEMM2
tile waits for its block's counter before it loads the rows, the weights are in flight meanwhile; a combine workgroup
waits for every GEMM2 tile. The counters (``READY``) only order work, every output element has one writer, and the
last consumer of a counter resets it for the next call. Deadlock-free: a workgroup only waits for workgroups of lower
ids, and each XCD dispatches its workgroups in id order. An expert with several blocks loads its weights cached
(``hot``), so that blocks running at the same time share them in L2; a lone block streams them non-temporally.

Raw ops: the loads and stores go through the legacy ``buffer_ops`` helpers (per-load cache policy, ``soffset``
folding), the ring through ``raw_ptr_buffer_load_async_lds`` with per-lane gathered offsets, the counters through
LLVM atomics, and the LDS through byte-addressed pointers into one shared array that the GEMM1, GEMM2 and combine
bodies alias with different layouts. Moving them to ``fx.copy`` / ``BufferCopyLDS128b`` / ``fx.llvm.atomic_*`` is
left for a change that shows bit-identical outputs and no loss in the benchmark.
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

from . import buffer_ops

H, I = 5120, 2304
TOPKS = (3, 6)
NE_MAX = 63  # the sort's wave scan and counters hold up to 63 experts
LIMIT = 10.0
KT1, KT2 = H // 128, I // 128  # 128-K MFMA steps of GEMM1 / GEMM2
THREADS = 256
SC1 = 16  # cache policy bit: device-scope (write-through) stores
READY_STRIDE = (
    256  # bytes between two blocks' counters (ready count, then consumer count)
)
COMBINE_WGS = 256  # combine workgroups of the fused launch: one per CU, each with ``cw`` tokens (cw * 256 >= tokens)


def _cdiv(a: int, b: int) -> int:
    return -(-a // b)


def max_blocks(n_tokens: int, block_m: int, ne: int, topk: int) -> int:
    """Bound on the row blocks of ``n_tokens`` tokens: every active expert pads by < ``block_m`` rows."""
    pairs = n_tokens * topk
    return pairs // block_m + min(ne, pairs)


def _raw(x):
    return x.ir_value() if hasattr(x, "ir_value") else x


def _mfma(a, b, acc, sa, sb, byte, *, w_first):
    """Scaled 16x16x128 MFMA into AGPRs, FP4 weights x FP8 activations. ``w_first``: the weights are operand A
    (GEMM1, output ``[weight column, row]``); ``byte`` picks the weight scale's byte (op_sel).
    """
    ty = T.vec(4, T.f32)
    if w_first:
        fmt, ob = (
            "cbsz:4 blgp:0",
            f"op_sel:[{byte % 2},0,0] op_sel_hi:[{byte // 2},0,0]",
        )
    else:
        fmt, ob = (
            "cbsz:0 blgp:4",
            f"op_sel:[0,{byte % 2},0] op_sel_hi:[0,{byte // 2},0]",
        )
    # inline asm instead of fx.rocdl.mfma_scale_f32_16x16x128_f8f6f4: "=a" pins the accumulators to AGPRs; with one
    # wave per SIMD the 128 accumulator registers of a 128-row tile and the prefetched weight fragments do not fit
    # in the 256 VGPRs.
    # s_nop: LLVM inserts no wait states before an inline-asm MFMA (a register an MFMA reads right after a VALU or
    # accumulator-copy write returned stale data without it)
    src_c = "$0" if acc is not None else "0"
    asm = f"s_nop 1\nv_mfma_scale_f32_16x16x128_f8f6f4 $0, $1, $2, {src_c}, $3, $4 {ob} {fmt}"
    ops = [_raw(a), _raw(b), _raw(sa), _raw(sb)]
    cons = "=a,v,v,v,v"
    if acc is not None:
        ops.append(_raw(acc))
        cons += ",0"
    return llvm.inline_asm(ty, ops, asm, cons, has_side_effects=True)


def _hold(xs):
    """Keep the required inline-MFMA latency before any accumulator reads."""
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


def _lds_ptr(base, off):
    return llvm.inttoptr(ir.Type.parse("!llvm.ptr<3>"), _raw(fx.Int64(base + off)))


def _load8(rsrc, off, soff, plane=64):
    """K [16 q, 16 q + 16) and [64 + 16 q, 80 + 16 q) of a 128-K step (lane group q) as one 8-dword MFMA operand: the
    FP8 operand layout of the scaled MFMA, which pairs with the FP4 operand's K [32 q, 32 q + 32). ``plane``: bytes
    between the two 16-byte halves."""
    lo = Vec(
        buffer_ops.buffer_load(
            rsrc, off, vec_width=4, dtype=fx.Int32, soffset_bytes=soff
        )
    )
    hi = Vec(
        buffer_ops.buffer_load(
            rsrc, off + plane // 4, vec_width=4, dtype=fx.Int32, soffset_bytes=soff
        )
    )
    return Vec.from_elements(
        [lo[j] for j in range(4)] + [hi[j] for j in range(4)], fx.Int32
    )


def _dma(rsrc, lds_addr, voff, soff):
    """Async ``buffer_load ... lds``: 16 bytes per lane into LDS at the wave-uniform ``lds_addr`` + 16 * lane; the
    caller closes the stage with ``asyncmark`` and waits with ``wait_asyncmark``."""
    lds = llvm.IntToPtrOp(
        ir.Type.parse("!llvm.ptr<3>"),
        _raw(fx.Int32(fx.rocdl.readfirstlane(T.i32, lds_addr))),
    )
    fx.rocdl.raw_ptr_buffer_load_async_lds(
        rsrc,
        lds.result,
        fx.Int32(16).ir_value(),
        fx.Int32(voff).ir_value(),
        fx.Int32(soff).ir_value(),
        fx.Int32(0).ir_value(),
        aux=convert_to_mlir_attr(0),
    )


def _lds_load8(base, lo_off, hi_off):
    """Two 16-byte LDS reads as one 8-dword MFMA operand."""
    lo = Vec(llvm.load(T.vec(4, T.i32), _lds_ptr(base, lo_off), alignment=16))
    hi = Vec(llvm.load(T.vec(4, T.i32), _lds_ptr(base, hi_off), alignment=16))
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


def _fp8(old, a, b, scale, hi):
    """v_cvt_scalef32_pk_fp8_f32: (a, b) / scale -> 2 e4m3 into the low (``hi`` False) or high half of ``old``."""
    ty = ir.VectorType.get([2], ir.IntegerType.get_signless(16))
    old_v = llvm.bitcast(ty, fx.as_ir_value(old))
    res = llvm.call_intrinsic(
        ty,
        "llvm.amdgcn.cvt.scalef32.pk.fp8.f32",
        [
            old_v,
            fx.as_ir_value(a),
            fx.as_ir_value(b),
            fx.as_ir_value(scale),
            fx.Boolean(hi).ir_value(),
        ],
        [],
        [],
    )
    return fx.Int32(llvm.bitcast(T.i32, res))


def _lds_atomic_add_i32(ptr, value):
    """``old = *ptr; *ptr += value`` on an LDS i32 pointer (workgroup scope)."""
    return fx.Int32(
        llvm.AtomicRMWOp(
            llvm.AtomicBinOp.add,
            ptr.llvm_ptr,
            fx.Int32(value).ir_value(),
            llvm.AtomicOrdering.monotonic,
            syncscope="workgroup",
            alignment=4,
        ).result
    )


_SCAN_DPP = (
    (0x111, 0xF, True),
    (0x112, 0xF, True),
    (0x114, 0xF, True),
    (0x118, 0xF, True),
    (0x142, 0xA, False),
    (0x143, 0xC, False),
)


def _wave_incl_scan(x):
    """Inclusive prefix sum of an i32 over the 64 lanes of a wave (DPP, no LDS)."""
    from flydsl._mlir.dialects import rocdl

    zero = _raw(fx.Int32(0))
    for ctrl, row_mask, bound in _SCAN_DPP:
        x = x + fx.Int32(
            rocdl.update_dpp(T.i32, zero, _raw(x), ctrl, row_mask, 0xF, bound)
        )
    return x


def _tensor_addr(t):
    return fx.Int64(fx.ptrtoint(fx.get_iter(t)))


def _atomic_agent(op, addr, value):
    ptr_ty = fx.PointerType.get(
        T.i32, address_space=fx.AddressSpace.Global, alignment=4
    )
    ptr = fx.inttoptr(ptr_ty, fx.Int64(addr))
    return fx.Int32(
        llvm.AtomicRMWOp(
            op,
            ptr.llvm_ptr,
            fx.Int32(value).ir_value(),
            llvm.AtomicOrdering.monotonic,
            syncscope="agent",
            alignment=4,
        ).result
    )


def _poll(addr):
    """Thread 0's first read of the counter at ``addr`` (device scope; the other lanes' out-of-range offset drops
    theirs), issued before the tile's loads so its latency overlaps them: ``_await`` consumes it.
    """
    rsrc = buffer_ops.create_buffer_resource_from_addr(addr, num_records_bytes=8)
    off = (fx.thread_idx.x == 0).select(fx.Int32(0), fx.Int32(0x7FFFFFF0))
    return fx.Int32(
        llvm.call_intrinsic(
            T.i32,
            "llvm.amdgcn.raw.ptr.buffer.atomic.add.i32",
            [
                _raw(fx.Int32(0)),
                _raw(rsrc),
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
    write-through, then counted); the workgroup waits at a barrier. No acquire fence: no load of this launch touched
    the producers' lines before the wait."""
    if fx.thread_idx.x == 0:
        while got < need:
            fx.rocdl.s_sleep(nap)
            got = _atomic_agent(llvm.AtomicBinOp.add, addr, 0)
    fx.gpu.barrier()


@flyc.jit
def _retire(addr, consumers):
    """A consumer's last act: counts itself at ``addr + 4``; the last of ``consumers`` resets both words to 0 for the
    next call (every consumer has passed its wait, and no producer is left)."""
    if fx.thread_idx.x == 0:
        old = _atomic_agent(llvm.AtomicBinOp.add, addr + fx.Int64(4), 1)
        if old == consumers - 1:
            _atomic_agent(llvm.AtomicBinOp.xchg, addr, 0)
            _atomic_agent(llvm.AtomicBinOp.xchg, addr + fx.Int64(4), 0)


@flyc.jit
def _retire_block(addr, consumers, done):
    """A GEMM2 tile's last act: counts itself at ``addr + 4``; the last of the block's ``consumers`` resets both words
    and counts the block at ``done``."""
    if fx.thread_idx.x == 0:
        old = _atomic_agent(llvm.AtomicBinOp.add, addr + fx.Int64(4), 1)
        if old == consumers - 1:
            _atomic_agent(llvm.AtomicBinOp.xchg, addr, 0)
            _atomic_agent(llvm.AtomicBinOp.xchg, addr + fx.Int64(4), 0)
            _atomic_agent(llvm.AtomicBinOp.add, done, 1)


@flyc.jit
def _signal(addr):
    """Count this workgroup at the counter ``addr``: its stores retire first."""
    fx.rocdl.s_waitcnt(vmcnt=0)
    fx.gpu.barrier()
    if fx.thread_idx.x == 0:
        _atomic_agent(llvm.AtomicBinOp.add, addr, 1)


@functools.cache
def _gemm1_tile(*, kw, pf, bm, ni, nbuf, ne, signal=False, nwv=4, hot=False):
    """The GEMM1 tile body ``tile(base, rb, A, AS, W, WS, Q, S, IDS, EIDS, NV, m, rows, nmb, ready)`` of workgroup
    ``rb`` (``base`` its LDS), and its columns per workgroup and LDS bytes. ``nbuf`` > 0: the activation rows of the
    block stream through an ``nbuf``-stage LDS ring by async DMA (gathered by token, written in MFMA fragment order)
    and every wave reads them from there (``kw`` 1). ``signal``: write-through stores, counted at ``ready``. ``nwv``
    waves per workgroup (8: two waves per SIMD, ``bm`` >= 64). ``hot``: an expert with several blocks loads its weights
    cached, so that the blocks running at the same time share them in L2."""
    assert (
        kw in (1, 2, 4)
        and bm in (16, 32, 64, 128)
        and ni in (2, 4)
        and (nbuf == 0 or kw == 1 and bm >= 32)
    )
    assert nwv == 4 or bm >= 64 and nbuf > 0 and kw == 1
    nw = nwv // kw
    tn, kt, mr = nw * 16 * ni, KT1 // kw, bm // 16
    assert kt % 2 == 0 and I % tn == 0  # one weight-scale word covers two K steps
    sag = 4 if kt % 4 == 0 else 2  # K steps per activation-scale load
    nacc = mr * ni * 2
    apipe = (
        nbuf > 0 and bm >= 32
    )  # the MFMAs of one row tile run under the LDS read of the next
    pitch = bm * 128  # one ring stage: bm rows x one 128-K step
    # DMA rounds of the block's scale tile (bm rows x 160 B)
    sc_rounds = _cdiv(_cdiv(bm * (H // 32), 1024), nwv)
    lds = max(
        4 * nacc * 64 * 16 if kw > 1 else 0,
        nbuf * pitch + (sc_rounds * nwv * 1024 if nbuf else 0),
        16,
    )
    st_kw = {"cache_modifier": SC1} if signal else {}

    @flyc.jit
    def tile(base, rb, A, AS, W, WS, Q, S, IDS, EIDS, NV, m, rows, nmb, ready):
        tx = fx.thread_idx.x
        lane, wave = tx % 64, tx // 64
        l16, q16 = lane % 16, lane // 16
        nk, kk = wave % nw, wave // nw
        mb, nb = rb // (I // tn), rb % (I // tn)
        nr = fx.Int32(fx.rocdl.readfirstlane(T.i32, _raw(fx.Int32(NV[0]))))
        e_raw = fx.Int32(
            EIDS[mb]
        )  # read beside NV and the neighbours' ids, used below them
        e_prev, e_next = fx.Int32(EIDS[(mb > 0).select(mb - 1, mb)]), fx.Int32(
            EIDS[mb + 1]
        )
        expert = fx.Int32(fx.rocdl.readfirstlane(T.i32, _raw(e_raw)))

        def run(wcm):
            col = nb * tn + nk * 16 * ni
            ar = buffer_ops.create_buffer_resource(
                A, max_size=False, num_records_bytes=m * H
            )
            sr = buffer_ops.create_buffer_resource(
                AS, max_size=False, num_records_bytes=m * (H // 32)
            )
            wr = buffer_ops.create_buffer_resource(
                W,
                max_size=False,
                num_records_bytes=2 * I * H // 2,
                base_byte_offset=fx.Int64(expert) * (2 * I * H // 2),
            )
            ws = buffer_ops.create_buffer_resource(
                WS, max_size=False, num_records_bytes=ne * I * (H // 16)
            )
            qr = buffer_ops.create_buffer_resource(
                Q, max_size=False, num_records_bytes=rows * I
            )
            ss = buffer_ops.create_buffer_resource(
                S, max_size=False, num_records_bytes=rows * (I // 32)
            )
            tok = [
                fx.Int32(IDS[mb * bm + mi * 16 + l16]) & 0xFFFFFF
                for mi in range_constexpr(mr)
            ]
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
                    t_d = fx.Int32(IDS[mb * bm + (d // 2) * 16 + l16]) & 0xFFFFFF
                    dma_src.append(t_d * H + (d % 2) * 64 + q16 * 16)

                # scale tile DMA: lane chunk n = (nwv q + wave) * 64 + lane of 16 B, row n // 10 (rows past the tile
                # read a valid row and land in the padding)
                sc_src = []
                for q in range_constexpr(sc_rounds):
                    n = (wave + nwv * q) * 64 + lane
                    r = n // (H // 32 // 16)
                    r = (r < bm).select(r, fx.Int32(bm - 1))
                    t_r = fx.Int32(IDS[mb * bm + r]) & 0xFFFFFF
                    sc_src.append(t_r * (H // 32) + (n % (H // 32 // 16)) * 16)

                def issue(t):
                    for j in range_constexpr(2 * mr // nwv):
                        _dma(
                            ar,
                            base + (t % nbuf) * pitch + (wave + nwv * j) * 1024,
                            dma_src[j],
                            t * 128,
                        )
                    if const_expr(t == 0):
                        for q in range_constexpr(sc_rounds):
                            _dma(
                                sr,
                                base + nbuf * pitch + (q * nwv + wave) * 1024,
                                sc_src[q],
                                0,
                            )
                    fx.rocdl.asyncmark()

            sav = [
                None
            ] * mr  # the activation scales of `sag` K steps: one load per row tile
            sbv = [None]  # the weight scales of 2 K steps: one load per column tile

            def load_vec(k):
                if const_expr(nbuf):
                    aa = None
                else:
                    aa = [_load8(ar, ad[mi], k * 128) for mi in range_constexpr(mr)]
                sa = None
                if const_expr(not nbuf):
                    if const_expr(k % sag == 0):
                        for mi in range_constexpr(mr):
                            sav[mi] = Vec(
                                buffer_ops.buffer_load(
                                    sr,
                                    sd[mi],
                                    vec_width=sag,
                                    dtype=fx.Int32,
                                    soffset_bytes=k * 4,
                                )
                            )
                    sa = [
                        (fx.Int32(sav[mi][k % sag]) >> (q16 * 8)) & 255
                        for mi in range_constexpr(mr)
                    ]
                bb = [
                    [
                        buffer_ops.buffer_load(
                            wr,
                            wd[j][gu],
                            vec_width=4,
                            dtype=fx.Int32,
                            soffset_bytes=k * 1024,
                            cache_modifier=wcm,
                        )
                        for gu in range_constexpr(2)
                    ]
                    for j in range_constexpr(ni)
                ]
                if const_expr(k % 2 == 0):
                    sbv[0] = [
                        buffer_ops.buffer_load(
                            ws,
                            swd[j],
                            vec_width=1,
                            dtype=fx.Int32,
                            soffset_bytes=(k // 2) * 256,
                        )
                        for j in range_constexpr(ni)
                    ]
                return aa, sa, bb, sbv[0]

            def lds_sa(k):
                return [
                    (
                        fx.Int32(
                            llvm.load(
                                T.i32,
                                _lds_ptr(
                                    base,
                                    nbuf * pitch + (mi * 16 + l16) * (H // 32) + k * 4,
                                ),
                                alignment=4,
                            )
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
                            bb[j][gu] = buffer_ops.buffer_load(
                                wr,
                                wd[j][gu],
                                vec_width=4,
                                dtype=fx.Int32,
                                soffset_bytes=k * 1024,
                                cache_modifier=wcm,
                            )

                        pcs.append(load_b)
                if const_expr(k % 2 == 0):

                    def load_sb():
                        sbv[0] = [
                            buffer_ops.buffer_load(
                                ws,
                                swd[j],
                                vec_width=1,
                                dtype=fx.Int32,
                                soffset_bytes=(k // 2) * 256,
                            )
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
                    gap = max(
                        1, (mr * ni * 2) // max(len(pcs), 1)
                    )  # MFMAs between two of the next step's loads
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
                                if (
                                    pcs and n_mfma % gap == 0
                                ):  # the next step's loads, spread over the MFMAs
                                    pcs.pop(0)()
                                n_mfma += 1
                                acc[mi][j][gu] = _mfma(
                                    bb[j][gu],
                                    cur,
                                    acc[mi][j][gu],
                                    sb[j],
                                    sa[mi],
                                    (k % 2) * 2 + gu,
                                    w_first=True,
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
                                    w_first=True,
                                )
            flat = _hold(
                [
                    acc[mi][j][gu]
                    for mi in range(mr)
                    for j in range(ni)
                    for gu in range(2)
                ]
            )
            if const_expr(kw > 1):
                for j in range_constexpr(nacc):
                    llvm.StoreOp(
                        _raw(flat[j]),
                        _lds_ptr(base, ((wave * nacc + j) * 64 + lane) * 16),
                        alignment=16,
                    )
                fx.rocdl.s_waitcnt(lgkmcnt=0)
                fx.gpu.barrier()
            if kk == 0:
                vals = [Vec(v) for v in flat]
                for k in range_constexpr(1, kw):
                    for j in range_constexpr(nacc):
                        other = Vec(
                            llvm.load(
                                T.vec(4, T.f32),
                                _lds_ptr(
                                    base, (((nk + k * nw) * nacc + j) * 64 + lane) * 16
                                ),
                                alignment=16,
                            )
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
                            pk = _fp8(fx.Int32(0), y[t][0], y[t][1], scale, False)
                            pk = _fp8(pk, y[t][2], y[t][3], scale, True)
                            c16 = col // 16 + 2 * g + t  # 16-column tile
                            off = (
                                rblk * (KT2 * 2048)
                                + (c16 // 8) * 2048
                                + ((c16 % 8) // 4) * 1024
                                + ((c16 % 4) * 16 + l16) * 16
                                + q16 * 4
                            )
                            buffer_ops.buffer_store(
                                pk, qr, off // 4, mask=tok[mi] < m, **st_kw
                            )
                        sg = col // 32 + g
                        s_off = rblk * (KT2 * 64) + (sg // 4) * 64 + l16 * 4 + sg % 4
                        buffer_ops.buffer_store(
                            e.to(fx.Int8),
                            ss,
                            s_off,
                            offset_is_bytes=True,
                            mask=(tok[mi] < m) & (q16 == 0),
                            **st_kw,
                        )
            if const_expr(signal):
                _signal(ready + fx.Int64(mb) * READY_STRIDE)

        if mb * bm < nr:
            if const_expr(hot):
                if (e_prev == e_raw) | (e_next == e_raw):
                    run(0)
                else:
                    run(2)
            else:
                run(2)

    return tile, tn, lds


@functools.cache
def _gemm2_tile(
    *,
    tn,
    bm,
    pf,
    nbuf,
    ne,
    topk,
    signal=0,
    rows_out=False,
    combine=True,
    hot=False,
    nwv=4,
):
    """The GEMM2 tile body ``tile(base, rb, A, AS, W, WS, OUT, IDS, WT, EIDS, NV, m, rows, nmb, ready, done)`` of
    workgroup ``rb`` and its LDS bytes. ``nbuf`` > 0: the intermediate rows stream through an ``nbuf``-stage LDS ring
    by async DMA. ``signal`` = GEMM1's tiles per block (> 0): the tile waits for them at ``ready`` after its weight
    loads are in flight, stores write-through and, with ``combine``, counts its block at ``done``. ``rows_out``: the
    row's output goes to row ``token`` (its input row) instead of slot ``token * topk + slot``. ``nwv`` waves per
    workgroup (8: two waves per SIMD, ``bm`` >= 64, ``tn`` 256)."""
    assert (
        tn in (128, 256, 512)
        and bm in (16, 32, 64, 128)
        and H % tn == 0
        and (nbuf == 0 or bm >= 32)
    )
    assert nwv == 4 and tn <= 256 or bm >= 64 and nbuf > 0 and tn >= 256
    ni, mr = tn // (16 * nwv), bm // 16
    tpr = tn // 8  # threads of an output row, 16 bytes each
    rpp = 64 * nwv // tpr  # rows per store pass
    pitch = bm * 128
    sc_bytes = (
        (bm * (I // 32) + 1023) // 1024 * 1024
    )  # the block's activation scale tile, staged in LDS
    row_b = (
        tn * 2 + 16
    )  # bytes per row of the bf16 output tile: the pad rotates the 16 rows of a store over the banks
    lds = max(nbuf * pitch + (sc_bytes if nbuf else 0), bm * row_b, 16)
    apipe = (
        nbuf > 0 and bm >= 32
    )  # the MFMAs of one row tile run under the LDS read of the next
    st_kw = {"cache_modifier": SC1} if signal else {}

    @flyc.jit
    def tile(base, rb, A, AS, W, WS, OUT, IDS, WT, EIDS, NV, m, rows, nmb, ready, done):
        tx = fx.thread_idx.x
        lane, wave = tx % 64, tx // 64
        l16, q16 = lane % 16, lane // 16
        mb, nb = rb // (H // tn), rb % (H // tn)
        nv_raw, e_raw = fx.Int32(NV[0]), fx.Int32(
            EIDS[mb]
        )  # all four reads are in flight before the first is used
        e_prev, e_next = fx.Int32(EIDS[(mb > 0).select(mb - 1, mb)]), fx.Int32(
            EIDS[mb + 1]
        )
        nr = fx.Int32(fx.rocdl.readfirstlane(T.i32, _raw(nv_raw)))
        expert = fx.Int32(fx.rocdl.readfirstlane(T.i32, _raw(e_raw)))

        def run(wcm):
            ar = buffer_ops.create_buffer_resource(
                A, max_size=False, num_records_bytes=rows * I
            )
            sr = buffer_ops.create_buffer_resource(
                AS, max_size=False, num_records_bytes=rows * (I // 32)
            )
            wr = buffer_ops.create_buffer_resource(
                W,
                max_size=False,
                num_records_bytes=H * I // 2,
                base_byte_offset=fx.Int64(expert) * (H * I // 2),
            )
            sw = buffer_ops.create_buffer_resource(
                WS, max_size=False, num_records_bytes=ne * H * (I // 32)
            )
            out = buffer_ops.create_buffer_resource(
                OUT,
                max_size=False,
                num_records_bytes=m * (1 if rows_out else topk) * H * 2,
            )
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
                    buffer_ops.buffer_load(
                        wr,
                        wd[j],
                        vec_width=4,
                        dtype=fx.Int32,
                        soffset_bytes=k * 1024,
                        cache_modifier=wcm,
                    )
                    for j in range_constexpr(ni)
                ]
                if const_expr(k % 2 == 0):
                    sbv[0] = [
                        buffer_ops.buffer_load(
                            sw,
                            swd[j],
                            vec_width=1,
                            dtype=fx.Int32,
                            soffset_bytes=(k // 2) * 256,
                        )
                        for j in range_constexpr(ni)
                    ]
                return bb, sbv[0]

            def load_a(k):
                if const_expr(nbuf):
                    return None, None
                aa = [_load8(ar, ad[mi], k * 2048, 1024) for mi in range_constexpr(mr)]
                sa = [
                    (
                        fx.Int32(
                            buffer_ops.buffer_load(
                                sr,
                                sd[mi],
                                vec_width=1,
                                dtype=fx.Int32,
                                soffset_bytes=k * 64,
                            )
                        )
                        >> (q16 * 8)
                    )
                    & 255
                    for mi in range_constexpr(mr)
                ]
                return aa, sa

            def lds_sa(k):
                return [
                    (
                        fx.Int32(
                            llvm.load(
                                T.i32,
                                _lds_ptr(
                                    base, nbuf * pitch + ((mi * KT2 + k) * 16 + l16) * 4
                                ),
                                alignment=4,
                            )
                        )
                        >> (q16 * 8)
                    )
                    & 255
                    for mi in range_constexpr(mr)
                ]

            acc = [[None] * ni for _ in range(mr)]
            epi_w = [
                fx.Float32(WT[mb * bm + mi * 16 + l16]) for mi in range_constexpr(mr)
            ]
            epi_ids = [
                fx.Int32(IDS[mb * bm + rr * rpp + tx // tpr])
                for rr in range_constexpr(bm // rpp)
            ]
            if const_expr(signal):
                counter = ready + fx.Int64(mb) * READY_STRIDE
                got = _poll(counter)
            wring = [load_w(k) for k in range_constexpr(pf)]
            if const_expr(signal):
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
                                w_first=True,
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
                                w_first=True,
                            )
            flat = _hold([acc[mi][j] for mi in range(mr) for j in range(ni)])
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
                    llvm.StoreOp(
                        _raw(
                            Vec.from_elements(
                                [lo.bitcast(fx.Int32)[0], hi.bitcast(fx.Int32)[0]],
                                fx.Int32,
                            )
                        ),
                        _lds_ptr(base, (mi * 16 + l16) * row_b + col * 2),
                        alignment=8,
                    )
            fx.rocdl.s_waitcnt(lgkmcnt=0)
            fx.gpu.barrier()
            for rr in range_constexpr(bm // rpp):
                row = rr * rpp + tx // tpr
                packed = epi_ids[rr]
                token, slot = packed & 0xFFFFFF, packed.shrui(fx.Int32(24))
                dst = token if rows_out else token * topk + slot
                col = tx % tpr * 8
                v = Vec(
                    llvm.load(
                        T.vec(4, T.i32),
                        _lds_ptr(base, row * row_b + col * 2),
                        alignment=16,
                    )
                )
                buffer_ops.buffer_store(
                    v, out, (dst * H + nb * tn + col) // 2, mask=token < m, **st_kw
                )
            if const_expr(signal):
                fx.rocdl.s_waitcnt(vmcnt=0)
                fx.gpu.barrier()
                if const_expr(combine):
                    _retire_block(counter, H // tn, done)
                else:
                    _retire(counter, H // tn)

        if mb * bm < nr:
            # an expert with several blocks reads its weights once per block: cached loads let concurrent blocks
            # share them in L2, a lone block streams them non-temporally
            if const_expr(hot):
                if (e_prev == e_raw) | (e_next == e_raw):
                    run(0)
                else:
                    run(2)
            else:
                run(2)

    return tile, lds


@functools.cache
def _combine_tile(*, ne, topk, signal=False, threads=THREADS, step=1, per_wg=1):
    """The combine body ``tile(tok0, TID, SLOTS, OUT, m, done, need, n_wg)`` of the workgroup that owns tokens ``tok0``,
    ``tok0 + step``, ... (``per_wg`` of them): bf16 slots ``[m, topk, 5120]`` -> ``[m, 5120]``, the FP32 sum in slot
    order and one rounding; a slot whose id ``TID`` is outside [0, ne) (not routed here) counts as zero. ``signal``: waits
    until ``need`` GEMM2 tiles have counted at ``done``; the last of the ``n_wg`` workgroups resets the counters.
    """
    cols = 8  # bf16 per thread and step
    chunks = H // cols
    passes = (chunks + threads - 1) // threads

    @flyc.jit
    def tile(tok0, TID, SLOTS, OUT, m, done, need, n_wg):
        tx = fx.thread_idx.x
        if tok0 < m:
            if const_expr(signal):
                got = _poll(done)
            sr = buffer_ops.create_buffer_resource(
                SLOTS, max_size=False, num_records_bytes=m * topk * H * 2
            )
            outr = buffer_ops.create_buffer_resource(
                OUT, max_size=False, num_records_bytes=m * H * 2
            )
            tokens = [tok0 + i * step for i in range_constexpr(per_wg)]
            routed = []
            for token in tokens:
                ok_t = token < m
                row = []
                for s in range_constexpr(topk):
                    e_s = fx.Int32(TID[(ok_t.select(token, tok0)) * topk + s])
                    row.append((e_s >= 0) & (e_s < ne) & ok_t)
                routed.append(row)
            if const_expr(signal):
                _await(done, need, got, 20)
            for i in range_constexpr(per_wg):
                token = tokens[i]
                for p in range_constexpr(passes):
                    c = tx + p * threads
                    ok = (c < chunks) & (token < m)
                    raw = [
                        Vec(
                            buffer_ops.buffer_load(
                                sr,
                                ((token * topk + s) * H + c * cols) // 2,
                                vec_width=4,
                                dtype=fx.Int32,
                                mask=ok & routed[i][s],
                            )
                        )
                        for s in range_constexpr(topk)
                    ]
                    acc = [None] * cols
                    for s in range_constexpr(topk):
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
                    buffer_ops.buffer_store(
                        Vec.from_elements(packed, fx.Int32),
                        outr,
                        (token * H + c * cols) // 2,
                        mask=ok,
                    )
            if const_expr(signal):
                _retire(done, n_wg)

    return tile


@functools.cache
def compile_gemm1(*, ne, kw=4, pf=3, bm=16, ni=2, nbuf=0, nwv=4, hot=False):
    tile, tn, lds = _gemm1_tile(
        kw=kw, pf=pf, bm=bm, ni=ni, nbuf=nbuf, nwv=nwv, ne=ne, hot=hot
    )
    threads = 64 * nwv
    name = (
        f"moe_gemm1_a8w4_clamp_swiglu_mxfp8_h5120_i2304_bm{bm}_tn{tn}_kw{kw}_pf{pf}"
        + (f"_ring{nbuf}" if nbuf else "")
        + (f"_w{nwv}" if nwv != 4 else "")
        + ("_hot" if hot else "")
    )

    @fx.struct
    class Shared:
        raw: fx.Array[fx.Uint8, lds, 16]

    @flyc.kernel(name=name, known_block_size=[threads, 1, 1])
    def kernel(
        A: fx.Tensor,
        AS: fx.Tensor,
        W: fx.Tensor,
        WS: fx.Tensor,
        Q: fx.Tensor,
        S: fx.Tensor,
        IDS: fx.Tensor,
        EIDS: fx.Tensor,
        NV: fx.Tensor,
        m: fx.Int32,
        rows: fx.Int32,
        nmb: fx.Int32,
    ):
        smem = fx.SharedAllocator().allocate(Shared).peek()
        tile(
            fx.Int32(fx.ptrtoint(smem.raw.ptr)),
            fx.Int32(fx.block_idx.x),
            A,
            AS,
            W,
            WS,
            Q,
            S,
            IDS,
            EIDS,
            NV,
            m,
            rows,
            nmb,
            fx.Int64(0),
        )

    @flyc.jit
    def launch_gemm1(
        A: fx.Tensor,
        AS: fx.Tensor,
        W: fx.Tensor,
        WS: fx.Tensor,
        Q: fx.Tensor,
        S: fx.Tensor,
        IDS: fx.Tensor,
        EIDS: fx.Tensor,
        NV: fx.Tensor,
        m: fx.Int32,
        rows: fx.Int32,
        nmb: fx.Int32,
        stream: fx.Stream,
    ):
        kernel(A, AS, W, WS, Q, S, IDS, EIDS, NV, m, rows, nmb).launch(
            grid=(nmb * (I // tn), 1, 1), block=(threads, 1, 1), stream=stream
        )

    launch_gemm1.kernel_name = name
    return launch_gemm1


@functools.cache
def compile_gemm2(*, ne, topk, tn=256, bm=16, pf=3, nbuf=0, hot=False, nwv=4):
    tile, lds = _gemm2_tile(
        tn=tn, bm=bm, pf=pf, nbuf=nbuf, hot=hot, nwv=nwv, ne=ne, topk=topk
    )
    threads = 64 * nwv
    name = (
        f"moe_gemm2_a8w4_slots_h5120_i2304_top{topk}_bm{bm}_tn{tn}_pf{pf}"
        + (f"_ring{nbuf}" if nbuf else "")
        + ("_hot" if hot else "")
        + (f"_w{nwv}" if nwv != 4 else "")
    )

    @fx.struct
    class Shared:
        raw: fx.Array[fx.Uint8, lds, 16]

    @flyc.kernel(name=name, known_block_size=[threads, 1, 1])
    def kernel(
        A: fx.Tensor,
        AS: fx.Tensor,
        W: fx.Tensor,
        WS: fx.Tensor,
        OUT: fx.Tensor,
        IDS: fx.Tensor,
        WT: fx.Tensor,
        EIDS: fx.Tensor,
        NV: fx.Tensor,
        m: fx.Int32,
        rows: fx.Int32,
        nmb: fx.Int32,
    ):
        smem = fx.SharedAllocator().allocate(Shared).peek()
        tile(
            fx.Int32(fx.ptrtoint(smem.raw.ptr)),
            fx.Int32(fx.block_idx.x),
            A,
            AS,
            W,
            WS,
            OUT,
            IDS,
            WT,
            EIDS,
            NV,
            m,
            rows,
            nmb,
            fx.Int64(0),
            fx.Int64(0),
        )

    @flyc.jit
    def launch_gemm2(
        A: fx.Tensor,
        AS: fx.Tensor,
        W: fx.Tensor,
        WS: fx.Tensor,
        OUT: fx.Tensor,
        IDS: fx.Tensor,
        WT: fx.Tensor,
        EIDS: fx.Tensor,
        NV: fx.Tensor,
        m: fx.Int32,
        rows: fx.Int32,
        nmb: fx.Int32,
        stream: fx.Stream,
    ):
        kernel(A, AS, W, WS, OUT, IDS, WT, EIDS, NV, m, rows, nmb).launch(
            grid=(nmb * (H // tn), 1, 1), block=(threads, 1, 1), stream=stream
        )

    launch_gemm2.kernel_name = name
    return launch_gemm2


@functools.cache
def compile_combine(*, ne, topk):
    tile = _combine_tile(ne=ne, topk=topk)

    @flyc.kernel(
        name=f"moe_combine_slots_h5120_top{topk}", known_block_size=[THREADS, 1, 1]
    )
    def kernel(TID: fx.Tensor, SLOTS: fx.Tensor, OUT: fx.Tensor, m: fx.Int32):
        tile(
            fx.Int32(fx.block_idx.x),
            TID,
            SLOTS,
            OUT,
            m,
            fx.Int64(0),
            fx.Int32(0),
            fx.Int32(0),
        )

    @flyc.jit
    def launch_combine(
        TID: fx.Tensor, SLOTS: fx.Tensor, OUT: fx.Tensor, m: fx.Int32, stream: fx.Stream
    ):
        kernel(TID, SLOTS, OUT, m).launch(
            grid=(m, 1, 1), block=(THREADS, 1, 1), stream=stream
        )

    return launch_combine


@functools.cache
def compile_experts(
    *,
    bm,
    kw,
    pf1,
    ni,
    nb1,
    tn2,
    pf2,
    nb2,
    ne,
    topk,
    hot=False,
    nwv=4,
    cw=1,
    rows_out=False,
    hot1=False,
):
    """GEMM1, GEMM2 and the combine in one launch (module docstring); ``rows_out``: no combine, ``OUT`` holds the
    route-weighted rows in input row order."""
    threads = 64 * nwv
    g1, tn1, lds1 = _gemm1_tile(
        kw=kw, pf=pf1, bm=bm, ni=ni, nbuf=nb1, signal=True, nwv=nwv, ne=ne, hot=hot1
    )
    n_ready = I // tn1
    g2, lds2 = _gemm2_tile(
        tn=tn2,
        bm=bm,
        pf=pf2,
        nbuf=nb2,
        signal=n_ready,
        rows_out=rows_out,
        combine=not rows_out,
        hot=hot,
        nwv=nwv,
        ne=ne,
        topk=topk,
    )
    cm = _combine_tile(
        signal=True, threads=threads, step=COMBINE_WGS, per_wg=cw, ne=ne, topk=topk
    )
    lds = max(lds1, lds2)
    c1, c2 = I // tn1, H // tn2
    name = (
        f"moe_experts_a8w4_fused_h5120_i2304_e{ne}_top{topk}_bm{bm}_g1_tn{tn1}_kw{kw}_pf{pf1}_r{nb1}_g2_tn{tn2}_pf{pf2}_r{nb2}"
        + ("_hot" if hot else "")
        + ("_hot1" if hot1 else "")
        + (f"_w{nwv}" if nwv != 4 else "")
        + (f"_c{cw}" if cw != 1 else "")
        + ("_rows" if rows_out else "")
    )

    @fx.struct
    class Shared:
        raw: fx.Array[fx.Uint8, lds, 16]

    @flyc.kernel(name=name, known_block_size=[threads, 1, 1])
    def kernel(
        A: fx.Tensor,
        AS: fx.Tensor,
        W13: fx.Tensor,
        WS13: fx.Tensor,
        Q: fx.Tensor,
        S: fx.Tensor,
        W2: fx.Tensor,
        WS2: fx.Tensor,
        SLOTS: fx.Tensor,
        OUT: fx.Tensor,
        TID: fx.Tensor,
        IDS: fx.Tensor,
        WT: fx.Tensor,
        EIDS: fx.Tensor,
        NV: fx.Tensor,
        READY: fx.Tensor,
        m: fx.Int32,
        rows: fx.Int32,
        nmb: fx.Int32,
    ):
        smem = fx.SharedAllocator().allocate(Shared).peek()
        base = fx.Int32(fx.ptrtoint(smem.raw.ptr))
        bid = fx.Int32(fx.block_idx.x)
        ready = _tensor_addr(READY)
        done = ready + fx.Int64(nmb) * READY_STRIDE
        n1 = nmb * c1
        n2 = nmb * c2
        if bid < n1:
            g1(base, bid, A, AS, W13, WS13, Q, S, IDS, EIDS, NV, m, rows, nmb, ready)
        else:
            if bid < n1 + n2:
                g2(
                    base,
                    bid - n1,
                    Q,
                    S,
                    W2,
                    WS2,
                    OUT if rows_out else SLOTS,
                    IDS,
                    WT,
                    EIDS,
                    NV,
                    m,
                    rows,
                    nmb,
                    ready,
                    done,
                )
            else:
                if const_expr(not rows_out):
                    cm(
                        bid - n1 - n2,
                        TID,
                        SLOTS,
                        OUT,
                        m,
                        done,
                        fx.Int32(NV[0]) // bm,
                        (m < COMBINE_WGS).select(m, fx.Int32(COMBINE_WGS)),
                    )

    @flyc.jit
    def launch_experts(
        A: fx.Tensor,
        AS: fx.Tensor,
        W13: fx.Tensor,
        WS13: fx.Tensor,
        Q: fx.Tensor,
        S: fx.Tensor,
        W2: fx.Tensor,
        WS2: fx.Tensor,
        SLOTS: fx.Tensor,
        OUT: fx.Tensor,
        TID: fx.Tensor,
        IDS: fx.Tensor,
        WT: fx.Tensor,
        EIDS: fx.Tensor,
        NV: fx.Tensor,
        READY: fx.Tensor,
        m: fx.Int32,
        rows: fx.Int32,
        nmb: fx.Int32,
        stream: fx.Stream,
    ):
        kernel(
            A,
            AS,
            W13,
            WS13,
            Q,
            S,
            W2,
            WS2,
            SLOTS,
            OUT,
            TID,
            IDS,
            WT,
            EIDS,
            NV,
            READY,
            m,
            rows,
            nmb,
        ).launch(
            grid=(nmb * (c1 + c2) + (0 if rows_out else COMBINE_WGS), 1, 1),
            block=(threads, 1, 1),
            stream=stream,
        )

    launch_experts.kernel_name = name
    return launch_experts


@flyc.jit
def _quant_tile(row, X, XQ, XS, m):
    """bf16 row ``row`` of ``X`` ``[m, 5120]`` -> e4m3 codes and row-major e8m0 per 32 columns (``_e8m0``: the rule
    of ``reference.quantize_mxfp8``); a thread owns 8 columns, four threads one scale group.
    """
    tx = fx.thread_idx.x
    xr = buffer_ops.create_buffer_resource(
        X, max_size=False, num_records_bytes=m * H * 2
    )
    qr = buffer_ops.create_buffer_resource(XQ, max_size=False, num_records_bytes=m * H)
    sr = buffer_ops.create_buffer_resource(
        XS, max_size=False, num_records_bytes=m * (H // 32)
    )
    chunks = H // 8
    for p in range_constexpr((chunks + THREADS - 1) // THREADS):
        c = tx + p * THREADS
        ok = c < chunks
        raw = Vec(
            buffer_ops.buffer_load(
                xr, (row * H + c * 8) // 2, vec_width=4, dtype=fx.Int32, mask=ok
            )
        )
        v = []
        for j in range_constexpr(4):
            dw = fx.Int32(raw[j])
            v += [
                (dw << 16).bitcast(fx.Float32),
                (dw & fx.Int32(-65536)).bitcast(fx.Float32),
            ]
        amax = fx.math.absf(v[0])
        for j in range_constexpr(1, 8):
            z = fx.math.absf(v[j])
            amax = (amax > z).select(amax, z)
        for sh in range_constexpr(2):
            z = amax.shuffle_xor(1 << sh, 64)
            amax = (amax > z).select(amax, z)
        e = _e8m0(amax)
        scale = (e << 23).bitcast(fx.Float32)
        lo = _fp8(_fp8(fx.Int32(0), v[0], v[1], scale, False), v[2], v[3], scale, True)
        hi = _fp8(_fp8(fx.Int32(0), v[4], v[5], scale, False), v[6], v[7], scale, True)
        buffer_ops.buffer_store(
            Vec.from_elements([lo, hi], fx.Int32), qr, (row * H + c * 8) // 4, mask=ok
        )
        buffer_ops.buffer_store(
            e.to(fx.Int8),
            sr,
            row * (H // 32) + c // 4,
            offset_is_bytes=True,
            mask=ok & (tx % 4 == 0),
        )


@functools.cache
def compile_sort(*, block_m, max_tokens, ne, topk, quant=False):
    """Workgroup 0: histogram, padded-count scan and placement of at most ``max_tokens * topk`` pairs into blocks of
    ``block_m`` rows. Outputs: ``ids[row] = token | slot << 24`` (padding rows: token ``n_tokens``), ``w[row]``
    (padding 0), ``eids[block]``, ``nv[0]`` = padded row count. Rows of an expert come out in LDS-atomic arrival
    order; a pair owns its GEMM2 output row, so the order cannot change any bit. ``quant``: workgroups 1..n_tokens
    quantize the bf16 activation rows ``X`` into ``XQ`` / ``XS`` (``_quant_tile``) in the same launch.
    """
    assert block_m in (16, 32, 64, 128)
    pairs = max_tokens * topk
    ppt = (pairs + THREADS - 1) // THREADS
    assert ne <= NE_MAX
    nb = max_blocks(max_tokens, block_m, ne, topk)
    ms = nb * block_m
    shift = block_m.bit_length() - 1
    name = f"moe_sort_e{ne}_top{topk}_bm{block_m}_ppt{ppt}" + (
        "_quant" if quant else ""
    )

    @fx.struct
    class Shared:
        count: fx.Array[fx.Int32, 64]
        start: fx.Array[fx.Int32, 64]
        nvs: fx.Array[fx.Int32, 4]
        ids: fx.Array[fx.Int32, ms + 1]
        wts: fx.Array[fx.Int32, ms + 1]
        blk: fx.Array[fx.Int32, nb + 1]

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def kernel(
        TID: fx.Tensor,
        TW: fx.Tensor,
        IDS: fx.Tensor,
        WT: fx.Tensor,
        EIDS: fx.Tensor,
        NV: fx.Tensor,
        X: fx.Tensor,
        XQ: fx.Tensor,
        XS: fx.Tensor,
        n_tok: fx.Int32,
    ):
        smem = fx.SharedAllocator().allocate(Shared).peek()
        tx = fx.thread_idx.x
        bid = fx.Int32(fx.block_idx.x)
        if bid == 0:
            lane = tx % 64
            wave = fx.Int32(fx.rocdl.readfirstlane(T.i32, tx // 64))
            count, start, ids, wts, blk = (
                smem.count.ptr,
                smem.start.ptr,
                smem.ids.ptr,
                smem.wts.ptr,
                smem.blk.ptr,
            )
            n_pairs = n_tok * topk
            ids_r = buffer_ops.create_buffer_resource(
                IDS, max_size=False, num_records_bytes=ms * 4
            )
            w_r = buffer_ops.create_buffer_resource(
                WT, max_size=False, num_records_bytes=ms * 4
            )
            e_r = buffer_ops.create_buffer_resource(
                EIDS, max_size=False, num_records_bytes=nb * 4
            )
            pr = []
            for k in range_constexpr(ppt):
                p = tx + k * THREADS
                pc = (p < n_pairs).select(p, n_pairs - 1)
                e_p = fx.Int32(TID[pc])
                valid = (
                    (p < n_pairs) & (e_p >= 0) & (e_p < ne)
                )  # an id outside [0, ne) is not routed here
                pr.append((p, valid, e_p, fx.Float32(TW[pc])))
            for j in range_constexpr((ms + 1 + THREADS - 1) // THREADS):
                r = tx + j * THREADS
                if r < ms + 1:
                    ids[r] = n_tok
                    wts[r] = 0
            if tx < 64:
                count[tx] = 0
            fx.gpu.barrier()
            ranks = [
                _lds_atomic_add_i32(count + v.select(e, fx.Int32(ne)), 1)
                for _, v, e, _w in pr
            ]
            fx.gpu.barrier()
            if wave == 0:
                c = (lane < ne).select(fx.Int32(count[lane]), fx.Int32(0))
                padded = (c + (block_m - 1)) & ~(block_m - 1)
                end = _wave_incl_scan(padded)
                start[lane] = end - padded
                if lane == 63:
                    NV[0] = end
                    smem.nvs.ptr[0] = end
            fx.gpu.barrier()
            for k, (p, valid, e, w) in enumerate(pr):
                r = valid.select(
                    fx.Int32(start[valid.select(e, fx.Int32(0))]) + ranks[k],
                    fx.Int32(ms),
                )
                ids[r] = ((p // topk) & 0x00FFFFFF) | ((p % topk) << 24)
                wts[r] = w.bitcast(fx.Int32)
                first = valid & ((r & (block_m - 1)) == 0)
                blk[first.select(r >> shift, fx.Int32(nb))] = e
            fx.gpu.barrier()
            nv = fx.Int32(smem.nvs.ptr[0])
            for j in range_constexpr((ms + THREADS - 1) // THREADS):
                r = tx + j * THREADS
                buffer_ops.buffer_store(fx.Int32(ids[r]), ids_r, r, mask=r < nv)
                buffer_ops.buffer_store(fx.Int32(wts[r]), w_r, r, mask=r < nv)
            for j in range_constexpr((nb + THREADS - 1) // THREADS):
                b = tx + j * THREADS
                buffer_ops.buffer_store(
                    fx.Int32(blk[b]), e_r, b, mask=b < (nv >> shift)
                )
        else:
            if const_expr(quant):
                _quant_tile(bid - 1, X, XQ, XS, n_tok)

    @flyc.jit
    def launch_sort(
        TID: fx.Tensor,
        TW: fx.Tensor,
        IDS: fx.Tensor,
        WT: fx.Tensor,
        EIDS: fx.Tensor,
        NV: fx.Tensor,
        X: fx.Tensor,
        XQ: fx.Tensor,
        XS: fx.Tensor,
        n_tok: fx.Int32,
        stream: fx.Stream,
    ):
        kernel(TID, TW, IDS, WT, EIDS, NV, X, XQ, XS, n_tok).launch(
            grid=((1 + n_tok) if quant else 1, 1, 1),
            block=(THREADS, 1, 1),
            stream=stream,
        )

    launch_sort.kernel_name = name
    return launch_sort


@functools.cache
def compile_blocks(*, block_m, max_rows, ne, quant=False):
    """Workgroup 0: the block table of rows already sorted by expert, ``offs[e]`` the first row of expert ``e`` and
    ``offs[ne]`` the row count: an expert's rows fill its blocks of ``block_m`` in order, the last block is padded.
    Outputs as ``compile_sort`` with ``ids[row]`` = the input row (slot 0; padding rows: ``n_rows``). ``quant``:
    workgroups 1..n_rows quantize the bf16 rows ``X`` (those below ``offs[ne]``) as ``compile_sort`` does.
    """
    assert block_m in (16, 32, 64, 128)
    assert ne <= NE_MAX
    nb = max_rows // block_m + ne
    ms = nb * block_m
    shift = block_m.bit_length() - 1
    name = f"moe_blocks_e{ne}_bm{block_m}_nb{nb}" + ("_quant" if quant else "")

    @fx.struct
    class Shared:
        off: fx.Array[fx.Int32, 64]
        bend: fx.Array[fx.Int32, 64]
        nvs: fx.Array[fx.Int32, 4]
        blk: fx.Array[fx.Int32, nb + 1]

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def kernel(
        OFFS: fx.Tensor,
        RW: fx.Tensor,
        IDS: fx.Tensor,
        WT: fx.Tensor,
        EIDS: fx.Tensor,
        NV: fx.Tensor,
        X: fx.Tensor,
        XQ: fx.Tensor,
        XS: fx.Tensor,
        n_rows: fx.Int32,
    ):
        smem = fx.SharedAllocator().allocate(Shared).peek()
        tx = fx.thread_idx.x
        bid = fx.Int32(fx.block_idx.x)
        if bid == 0:
            lane = tx % 64
            wave = fx.Int32(fx.rocdl.readfirstlane(T.i32, tx // 64))
            off, bend, blk = smem.off.ptr, smem.bend.ptr, smem.blk.ptr
            ids_r = buffer_ops.create_buffer_resource(
                IDS, max_size=False, num_records_bytes=ms * 4
            )
            w_r = buffer_ops.create_buffer_resource(
                WT, max_size=False, num_records_bytes=ms * 4
            )
            e_r = buffer_ops.create_buffer_resource(
                EIDS, max_size=False, num_records_bytes=nb * 4
            )
            rw_r = buffer_ops.create_buffer_resource(
                RW, max_size=False, num_records_bytes=n_rows * 4
            )
            if wave == 0:
                lo = fx.Int32(OFFS[(lane < ne).select(lane, fx.Int32(ne))])
                hi = fx.Int32(OFFS[(lane < ne).select(lane + 1, fx.Int32(ne))])
                blocks = (lane < ne).select(
                    (hi - lo + (block_m - 1)) >> shift, fx.Int32(0)
                )
                end = _wave_incl_scan(blocks)
                off[lane] = lo
                bend[lane] = end
                if lane == 63:
                    NV[0] = end << shift
                    smem.nvs.ptr[0] = end
            fx.gpu.barrier()
            total = fx.Int32(smem.nvs.ptr[0])
            for j in range_constexpr((nb + THREADS - 1) // THREADS):
                b = tx + j * THREADS
                e = fx.Int32(0)
                for sh in range_constexpr(
                    5, -1, -1
                ):  # the experts whose blocks all end by b: b's expert is the next
                    cand = e + (1 << sh)
                    e = ((cand <= ne) & (fx.Int32(bend[cand - 1]) <= b)).select(cand, e)
                if b < nb:
                    blk[b] = e
                buffer_ops.buffer_store(e, e_r, b, mask=b < total)
            fx.gpu.barrier()
            for j in range_constexpr((ms + THREADS - 1) // THREADS):
                r = tx + j * THREADS
                b = r >> shift
                e = fx.Int32(blk[(b < nb).select(b, fx.Int32(nb))])
                first = (e == 0).select(
                    fx.Int32(0), fx.Int32(bend[(e > 0).select(e - 1, fx.Int32(0))])
                )
                src = fx.Int32(off[e]) + (b - first) * block_m + (r & (block_m - 1))
                real = (b < total) & (src < fx.Int32(off[e + 1]))
                buffer_ops.buffer_store(
                    real.select(src, n_rows), ids_r, r, mask=r < (total << shift)
                )
                weight = fx.Float32(
                    buffer_ops.buffer_load(
                        rw_r,
                        real.select(src, fx.Int32(0)),
                        vec_width=1,
                        dtype=fx.Float32,
                    )
                )
                buffer_ops.buffer_store(
                    real.select(weight, fx.Float32(0)),
                    w_r,
                    r,
                    mask=r < (total << shift),
                )
        else:
            if const_expr(quant):  # noqa: SIM102 (trace-time branch)
                if bid - 1 < fx.Int32(OFFS[ne]):
                    _quant_tile(bid - 1, X, XQ, XS, n_rows)

    @flyc.jit
    def launch_blocks(
        OFFS: fx.Tensor,
        RW: fx.Tensor,
        IDS: fx.Tensor,
        WT: fx.Tensor,
        EIDS: fx.Tensor,
        NV: fx.Tensor,
        X: fx.Tensor,
        XQ: fx.Tensor,
        XS: fx.Tensor,
        n_rows: fx.Int32,
        stream: fx.Stream,
    ):
        kernel(OFFS, RW, IDS, WT, EIDS, NV, X, XQ, XS, n_rows).launch(
            grid=((1 + n_rows) if quant else 1, 1, 1),
            block=(THREADS, 1, 1),
            stream=stream,
        )

    launch_blocks.kernel_name = name
    return launch_blocks
