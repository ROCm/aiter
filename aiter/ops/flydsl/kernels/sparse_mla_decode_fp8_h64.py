# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
# ruff: noqa: B023  # closures over the unit / tile loops, which FlyDSL traces once
# ruff: noqa: SIM102  # nested ifs on traced values are nested scf.if, not `and`

"""Sparse MQA decode attention over per-tensor e4m3 KV rows on gfx950 (FlyDSL).

64 query heads, one KV head, head dim 512 (448 nope + 64 RoPE dims, V is the
whole row). Every query row attends a ragged main list (the sliding window)
and an extra list (the top-k compressed keys) as one key set, plus a per-head
attention sink (a key with no value). The epilogue rounds O / sum to bf16,
applies the inverse GPT-J RoPE to dims 448..511 and quantizes to MXFP8 (e4m3
codes, one E8M0 scale per 32 dims, row-major), or stores bf16 rows
(``mxfp8=False``, no RoPE). No atomics touch the outputs.

A workgroup is a (query row, group of ``hg`` heads) unit, ``hg`` = 16, 32 or
64. The head groups of a row land on one XCD (its L2 serves the key rows to all
of them) and run concurrently, so a row spreads over ``64 / hg`` CUs without a
split-KV combine.

- Keys come in tiles of 64 rows (32 KB), gathered by global -> LDS DMA
  (``global_load_async_lds``) into a 4-slot ring, the 16-byte pieces
  XOR-swizzled (``swz``) so the K reads (16 B) and the V^T reads
  (``ds_read_b64_tr_b8``) are bank-conflict free.
- QK is ``v_mfma_scale_f32_16x16x128_f8f6f4`` on S^T = K Q^T: lane (head n,
  quad q) gets the 4 keys 4 q + r of a 16-key block, so a lane's softmax state
  is 4 keys of one head. 4 QK waves: 2 key halves (the PV E8M0 scale is per 32
  keys) x 2 head slots of 16 heads (hg = 16: one slot; hg = 64: 2 blocks of
  16 heads per wave).
- PV is ``v_mfma_scale_f32_32x32x64_f8f6f4`` on O^T = V^T P^T with P as e4m3
  hi + residual terms via the E8M0 B-scale operand (``p_terms=2``, ~8 mantissa
  bits) or one e4m3 term (``p_terms=1``): 4 PV waves (dims 128 g .. + 127 of
  32 heads), 8 for hg = 64; hg = 16 puts the residual term in the MFMA columns
  16..31 and the epilogue adds the two halves. PV of tile c overlaps QK +
  softmax of tile c + 1, one barrier per tile.
- The softmax reference moves lazily (only when a block's max grows by more
  than ``LAZY`` log2), and the first reference sits ``SLACK`` above the first
  tile's max, so most tiles skip the O rescale. A tile whose exponent is more
  than ``EMAX`` above the reference (scores of hundreds of log2 units: the MX
  scale and the fp32 accumulator cannot carry the jump) restarts the
  accumulator at that exponent: the QK waves write each block's exponent
  excess ``x`` and ``alpha`` (one ulp below 1 when the block's own exponent is
  above the reference, 0 when the reference moved by more than ``EMAX``) into
  ``pr``, and the PV waves take their slow path only when ``alpha < 1``.
- Epilogue: the PV waves write O / sum rounded to bf16 as 1 KB rows into the
  ring's slots 2 and 3 (16-byte chunks XOR-swizzled by the row), then every
  wave applies the inverse RoPE and the MXFP8 quantization to 2 blocks of 32
  dims.
- hg = 64 is persistent: a workgroup runs its units (rows ``j + k wgs`` of its
  XCD's chunk) back to back, the slot-id list is double buffered in LDS, and
  the next unit's first two tiles load while this unit's epilogue runs.

LDS (hg = 64, p_terms = 2): ring 4 x 32768 + P 2 x 128 x 80 + mu 1024 + pr
2048 + fin 768 + ids 8192 + rec 32 = 163616 bytes of gfx950's 163840.

Raw ``rocdl`` / ``llvm`` ops and inline asm stay only where FlyDSL 0.5.0 has
no typed operation with the needed semantics; each helper below says why at
the call. Memory accesses go through typed ``fx`` pointers (the backend
resolves the address space) or buffer tensors and ``fx.copy``.
"""

import math
from types import SimpleNamespace

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir as _ir
from flydsl._mlir.dialects import llvm as _llvm
from flydsl._mlir.dialects import rocdl as _rocdl_ods
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr import rocdl as _rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

BN = 64  # keys per tile
D = 512  # head dim = row bytes = V width
H = 64  # query heads
ROPE = 64  # trailing dims the inverse RoPE turns
TILE_B = BN * D
MAXK = 1024  # keys per row (both lists)
PPITCH = 80  # P row bytes: 64 + 16 pad (8 lanes' 16 B pieces on distinct banks)
LAZY = 4.0  # headroom and lazy-reference step (log2)
# a tile whose exponent is more than EMAX above the reference restarts the accumulator
EMAX = 40.0
# P = 2^(s - M + PQ) <= 448 for the integer M >= the block's max score - LAZY
PQ = math.log2(448.0) - LAZY
LOG2E = 1.4426950408889634
# kernel arguments preloaded into SGPRs (the first 64 bytes) instead of s_load'ed
PRELOAD_HINTS = {
    "llvm_options": {"amdgpu-kernarg-preload": True, "amdgpu-kernarg-preload-count": 16}
}
TINY_F32 = 1.1754943508222875e-38
STAGES = 4

_SHARED = fx.AddressSpace.Shared
_GLOBAL = fx.AddressSpace.Global


def _ptr(space, addr, align, elem=None):
    """A typed pointer at a raw byte address (LDS addresses are computed with
    XOR swizzles, global ones from gathered row ids)."""
    ty = fx.PointerType.get(T.i32 if elem is None else elem, space, align)
    return fx.inttoptr(ty, fx.Int64(addr))


def _ld128(addr):
    return Vec(fx.ptr_load(_ptr(_SHARED, addr, 16), Vec.make_type(4, fx.Int32)))


def _st64(addr, v):
    fx.ptr_store(v, _ptr(_SHARED, addr, 8))


def _ld_i32(addr):
    return fx.Int32(fx.ptr_load(_ptr(_SHARED, addr, 4), fx.Int32))


def _st_i32(addr, v):
    fx.ptr_store(fx.Int32(v), _ptr(_SHARED, addr, 4))


def _ld_f32(addr):
    return fx.Float32(fx.ptr_load(_ptr(_SHARED, addr, 4, T.f32), fx.Float32))


def _st_f32(addr, v):
    fx.ptr_store(fx.Float32(v), _ptr(_SHARED, addr, 4, T.f32))


def _gld_i32(addr):
    return fx.Int32(fx.ptr_load(_ptr(_GLOBAL, addr, 4), fx.Int32))


def _gld_f32(addr):
    return fx.Float32(fx.ptr_load(_ptr(_GLOBAL, addr, 4, T.f32), fx.Float32))


def _gld_f32x4(addr):
    return Vec(
        fx.ptr_load(_ptr(_GLOBAL, addr, 16, T.f32), Vec.make_type(4, fx.Float32))
    )


def _tr8(addr):
    # raw ds_read_b64_tr_b8 (V^T operand of the PV MFMA): the per-lane address
    # is the ring's hand-computed XOR swizzle (swz), not a layout view.
    return Vec(
        _rocdl_ods.ds_read_tr8_b64(
            T.vec(2, T.i32), fx.to_llvm_ptr(_ptr(_SHARED, addr, 8))
        ).result
    )


def _ld_inv(addr, ty):
    # raw llvm.load: !invariant.load makes a uniform global load an s_load,
    # which fx.ptr_load cannot request.
    p = fx.to_llvm_ptr(_ptr(_GLOBAL, addr, 4, ty))
    return _llvm.LoadOp(ty, p, alignment=4, invariant=True).result


def _ld_inv_i32(addr):
    """A uniform, read-only i32 (an ``s_load``)."""
    return fx.Int32(_ld_inv(addr, T.i32))


def _ld_inv_f32(addr):
    return fx.Float32(_ld_inv(addr, T.f32))


def _buffer(base, nbytes, elem, align):
    """A buffer-resource tensor of ``nbytes`` at ``base``, indexed in ``elem``
    units by ``fx.slice(buf, (None, i))``; accesses past ``nbytes`` are dropped."""
    view = fx.make_view(_ptr(_GLOBAL, base, align, elem), fx.make_layout(1 << 30, 1))
    buf = fx.rocdl.make_buffer_tensor(
        fx.Tensor(view), max_size=False, num_records_bytes=nbytes
    )
    return fx.logical_divide(buf, fx.make_layout(1, 1))


def _copy_in(atom, buf, idx, n, dtype):
    frag = fx.make_rmem_tensor(n, dtype)
    fx.copy(atom, fx.slice(buf, (None, idx)), frag)
    return Vec(frag.load())


def _copy_out(atom, vals, dtype, buf, idx):
    frag = fx.make_rmem_tensor(len(vals), dtype)
    frag.store(Vec.from_elements(vals, dtype))
    fx.copy(atom, frag, fx.slice(buf, (None, idx)))


def _barrier():
    # inline asm, measured: gpu.barrier (workgroup fences around s_barrier) in
    # its place gives wrong outputs with this LDS DMA ring (relative error > 1
    # against the fp64 reference on the served shapes). This form always drains
    # the wave's LDS ops, and its "memory" clobber pins every access in place.
    _llvm.inline_asm(
        None, [], "s_waitcnt lgkmcnt(0)\ns_barrier", "~{memory}", has_side_effects=True
    )


def _waitcnt_vm(n):
    fx.rocdl.s_waitcnt(vmcnt=n)


def _bits(f):
    return fx.Float32(f).bitcast(fx.Int32)


def _as_f32(i):
    return fx.Int32(i).bitcast(fx.Float32)


def _permlane_swap(op, d_a, d_b):
    # raw: permlane{16,32}_swap return an LLVM struct of the two swapped words;
    # FlyDSL 0.5.0 has no typed wrapper.
    pair_ty = _ir.Type.parse("!llvm.struct<(i32, i32)>")
    res = op(pair_ty, fx.as_ir_value(d_a), fx.as_ir_value(d_b), False, False)
    return fx.Int32(_llvm.extractvalue(T.i32, res, [0])), fx.Int32(
        _llvm.extractvalue(T.i32, res, [1])
    )


def _maxf_nn(a, b):
    # nnan: one v_max_f32, no NaN canonicalization of the operands
    return fx.Float32(
        fx.maxnumf(fx.Float32(a), fx.Float32(b), fastmath=fx.arith.FastMathFlags.nnan)
    )


def _xlane4(x, op):
    """``op`` over lanes L, L ^ 16, L ^ 32, L ^ 48, broadcast to all four."""
    a, b = _permlane_swap(_rocdl.permlane32_swap, _bits(x), _bits(x))
    m = op(_as_f32(a), _as_f32(b))
    a, b = _permlane_swap(_rocdl.permlane16_swap, _bits(m), _bits(m))
    return op(_as_f32(a), _as_f32(b))


def _valu(asm, *ops):
    """One unpacked fp32 VALU op as inline asm: LLVM pairs plain fp32 ops of the softmax into v_pk_* (165 -> 280
    packed ops in the hg = 64, p_terms = 2 kernel), and the QK waves share their SIMDs with the PV waves' MFMAs.
    Measured (MI355X, CUDA-graph replay, 640 keys per row): plain ops are 1.3-2.3% slower at 192-768 rows with
    p_terms = 2, no different with p_terms = 1."""
    return fx.Float32(
        _llvm.inline_asm(
            T.f32,
            [fx.as_ir_value(fx.Float32(o)) for o in ops],
            asm,
            "=v" + ",v" * len(ops),
            has_side_effects=False,
        )
    )


def _afma(a, b, c):
    return _valu("v_fma_f32 $0, $1, $2, $3", a, b, c)


def _aadd(a, b):
    return _valu("v_add_f32 $0, $1, $2", a, b)


def _amul(a, b):
    return _valu("v_mul_f32 $0, $1, $2", a, b)


def _asub_t(a, b):
    return _valu("s_nop 0\n\tv_sub_f32 $0, $1, $2", a, b)


def _exp2(x):
    # raw v_exp_f32: fx.math.exp2 wraps it in a denormal-range rescale (4 more
    # VALU ops); here a P under 2^-126 (< 2^-130 of the block's largest) may flush to 0.
    return fx.Float32(_rocdl.exp2(fx.Float32.ir_type, fx.Float32(x).ir_value()))


def _pk_fp8(a, b, old, hi):
    return fx.Int32(
        _rocdl.cvt_pk_fp8_f32(T.i32, fx.Float32(a), fx.Float32(b), fx.Int32(old), hi)
    )


def _pack4_fp8(x0, x1, x2, x3):
    """Four fp32 -> a dword of four e4m3 bytes, ``x0`` the lowest."""
    # undef: the second convert writes the high half
    lo = _pk_fp8(x0, x1, fx.Int32(_llvm.mlir_undef(T.i32)), False)
    return _pk_fp8(x2, x3, lo, True)


def _unpack2_fp8(word, hi):
    v = Vec(
        _rocdl.cvt_pk_f32_fp8(Vec.make_type(2, fx.Float32), fx.as_ir_value(word), hi)
    )
    return v[0], v[1]


def _pk_bf16(a, b):
    """Two fp32 -> a dword of two bf16 (RNE), ``a`` low."""
    return fx.Int32(
        Vec.from_elements([fx.Float32(a), fx.Float32(b)], fx.Float32)
        .to(fx.BFloat16)
        .bitcast(fx.Int32)[0]
    )


# The scaled-MFMA wrappers take the register operands as built above (K and P
# from swizzled LDS reads, V^T from ds_read_b64_tr_b8) and a per-call E8M0 B
# scale (the P terms and the lazy rescale), with no MMA atom or fragments.
def _mfma16(a8, b8, acc4):
    """v_mfma_scale_f32_16x16x128_f8f6f4, e4m3 x e4m3, unit scales. Lane (i, q) holds A[i][32 q + e], B[32 q + e][i],
    e = 0..31 (8 dwords), and gets D[4 q + r][i]."""
    return Vec(
        _rocdl.mfma_scale_f32_16x16x128_f8f6f4(
            Vec.make_type(4, fx.Float32), [a8, b8, acc4, 0, 0, 0, 0, 0, 0]
        )
    )


def _mfma32s(a8, b8, acc16, sc_b):
    """``_mfma32`` with the E8M0 scale of the B operand's K block: lane (j, h) gives the scale (low byte of ``sc_b``)
    of block h of column j; a block is bytes 16 h' .. 16 h' + 15 of the lanes' 32 operand bytes (both lane halves).
    A has unit scales."""
    return Vec(
        _rocdl.mfma_scale_f32_32x32x64_f8f6f4(
            Vec.make_type(16, fx.Float32), [a8, b8, acc16, 0, 0, 0, 127, 0, sc_b]
        )
    )


def _cat8(lo4, hi4):
    return Vec.from_elements(
        [lo4[k] for k in range(4)] + [hi4[k] for k in range(4)], fx.Int32
    )


def swz(kappa):
    """The 16-byte piece index of a row is XORed with this: bits 0-1 of the row, bit 2 = row bit 2 ^ row bit 4."""
    return (kappa & 3) | ((((kappa >> 2) ^ (kappa >> 4)) & 1) << 2)


HEAD_GROUPS = (16, 32, 64)
OSTAGE = (
    2 * TILE_B
)  # the bf16 rows O / sum of the epilogue sit in the ring's slots 2 and 3
SLACK = 3.0  # log2 headroom of the first reference: later tiles rarely exceed it


def _dma16(src_addr, lds_addr):
    # raw global_load_lds_dwordx4: lane pairs gather their own 512-byte key
    # rows; one instruction per call, so wait_dma's vmcnt counts are exact.
    _rocdl_ods.global_load_async_lds(
        fx.to_llvm_ptr(_ptr(_GLOBAL, src_addr, 16)),
        fx.to_llvm_ptr(_ptr(_SHARED, lds_addr, 16)),
        16,
        0,
        aux=_ir.IntegerAttr.get(_ir.IntegerType.get_signless(32), 0),
    )


def compile_sparse_mla_decode_fp8_h64(
    hg: int = 16, p_terms: int = 2, mxfp8: bool = True, has_extra: bool = True
):
    """``hg``: heads per workgroup. ``p_terms``: 2 = P as hi + residual e4m3, 1 = one e4m3 P."""
    assert hg in HEAD_GROUPS and p_terms in (1, 2)
    NS = STAGES
    HG = hg
    PERSIST = (
        hg == 64
    )  # units of a workgroup overlap: the next unit's first tiles load during this unit's epilogue
    NHGU = H // HG  # workgroups per row
    HBW = 2 if HG == 64 else 1  # 16-head blocks per QK wave
    HSLOT = 2  # QK wave slots per key half
    NHGW = 1 if HG == 16 else 2  # of them used
    NQW = 2 * HSLOT  # QK waves
    NPW = 8 if HG == 64 else 4  # PV waves
    NWV = NQW + NPW
    NTASK = 8 * max(1, HG // 32)  # epilogue tasks: 2 blocks of 32 dims of 32 heads
    COLTERM = HG == 16 and p_terms == 2
    PV_TERMS = 1 if COLTERM else p_terms
    P_ROWS = p_terms * HG if HG >= 32 else 32
    P_BUF = P_ROWS * PPITCH
    ROWS_W = 64 // NPW  # key rows of a tile one PV wave loads
    DMA_PER_WAVE = ROWS_W // 2  # 1 KB DMA instructions per PV wave and tile
    name = (
        f"sparse_mla_decode_fp8_h64_hg{HG}_p{p_terms}"
        + ("_mxfp8" if mxfp8 else "_bf16")
        + ("_x" if has_extra else "")
    )

    @fx.struct
    class Shared:
        kv: fx.Array[fx.Uint8, NS * TILE_B, 16]
        p: fx.Array[fx.Uint8, 2 * P_BUF, 16]
        mu: fx.Array[fx.Float32, 2 * 2 * H]
        pr: fx.Array[fx.Int32, 2 * H * 4]
        fin: fx.Array[fx.Float32, 3 * H]
        ids: fx.Array[fx.Int32, 2 * MAXK]
        rec: fx.Array[fx.Int32, 8]

    @flyc.kernel(name=name, known_block_size=[64 * NWV, 1, 1])
    def kernel(
        arg_q: fx.Int64,
        arg_main: fx.Int64,
        arg_xtra: fx.Int64,
        arg_mip: fx.Int64,
        arg_mid: fx.Int64,
        arg_xip: fx.Int64,
        arg_xid: fx.Int64,
        arg_sink: fx.Int64,
        arg_pos: fx.Int64,
        arg_cs: fx.Int64,
        arg_out: fx.Int64,
        arg_osc: fx.Int64,
        arg_qs: fx.Int64,
        arg_ks: fx.Int64,
        i32_T: fx.Int32,
        i32_units_x: fx.Int32,
        i32_wgs_x: fx.Int32,
        i32_m_lg: fx.Int32,
        i32_m_st: fx.Int32,
        i32_x_lg: fx.Int32,
        i32_x_st: fx.Int32,
        i32_cs_st: fx.Int32,
        i32_pos_b: fx.Int32,
        f32_sm: fx.Float32,
    ):
        smem = fx.SharedAllocator().allocate(Shared).peek()
        tx0 = fx.thread_idx.x
        tx = tx0
        bx = fx.block_idx.x
        lane = tx % 64
        w8 = fx.Int32(_rocdl.readfirstlane(T.i32, tx // 64))
        isq = w8 < NQW
        l32, hh = lane % 32, lane // 32
        i16, q4 = lane % 16, lane // 16
        neg_inf = fx.Float32(float("-inf"))
        zero4 = Vec.filled(4, 0.0, fx.Float32)
        zero16 = Vec.filled(16, 0.0, fx.Float32)

        xcd = bx % 8
        jx = bx // 8

        lds_kv = fx.Int32(fx.ptrtoint(smem.kv.ptr))
        lds_p = fx.Int32(fx.ptrtoint(smem.p.ptr))
        lds_mu = fx.Int32(fx.ptrtoint(smem.mu.ptr))
        lds_pr = fx.Int32(fx.ptrtoint(smem.pr.ptr))
        lds_fin = fx.Int32(fx.ptrtoint(smem.fin.ptr))
        lds_ids = fx.Int32(fx.ptrtoint(smem.ids.ptr))
        lds_rec = fx.Int32(fx.ptrtoint(smem.rec.ptr))

        def uni(v):
            return fx.Int32(_rocdl.readfirstlane(T.i32, v))

        qs = _ld_inv_f32(arg_qs)
        ks = _ld_inv_f32(arg_ks)
        c_sc = f32_sm * qs * ks
        nu = (i32_units_x - jx + i32_wgs_x - 1) // i32_wgs_x
        nu = (jx < i32_units_x).select(nu, fx.Int32(0))
        for unit_i in range(fx.Index(0), fx.Index(nu), fx.Index(1)):  # type: ignore[call-overload]
            # lane-derived values are recomputed per unit: the compiler would hoist them out of this loop and keep them
            # live through the key loops (without it, hg = 64 spills 46 VGPRs to scratch). Inline asm: an s_mov the
            # compiler cannot fold.
            z = fx.Int32(
                _llvm.inline_asm(
                    T.i32, [], "s_mov_b32 $0, 0", "=s", has_side_effects=True
                )
            )
            tx = tx0 + z
            lane = tx % 64
            l32, hh = lane % 32, lane // 32
            i16, q4 = lane % 16, lane // 16

            def unit_geom(idx):
                geom = SimpleNamespace()
                loc = jx + fx.Int32(idx) * i32_wgs_x
                geom.unit = xcd * i32_units_x + loc
                row = geom.unit // NHGU
                geom.head_base = (geom.unit % NHGU) * HG
                geom.live = (loc < i32_units_x) & (row < i32_T)
                geom.rowc = geom.live.select(row, fx.Int32(0))
                return geom

            def unit_ptrs(geom):
                """(first main index, main keys, first extra index, extra keys) of the unit's row."""
                ms = uni(_ld_inv_i32(arg_mip + fx.Int64(geom.rowc) * 4))
                n_main = uni(_ld_inv_i32(arg_mip + fx.Int64(geom.rowc) * 4 + 4)) - ms
                if const_expr(has_extra):
                    xs = uni(_ld_inv_i32(arg_xip + fx.Int64(geom.rowc) * 4))
                    n_x = uni(_ld_inv_i32(arg_xip + fx.Int64(geom.rowc) * 4 + 4)) - xs
                else:
                    xs = fx.Int32(0)
                    n_x = fx.Int32(0)
                return [ms, n_main, xs, n_x]

            def unit_ptrs_raw(geom):
                """The row's indptr entries as vector loads (not waited for by the barriers; ``unit_ptrs_fin`` uses them)."""
                zero = fx.Int32(0)

                def ld(base):
                    return _gld_i32(base + fx.Int64(geom.rowc) * 4)

                return [
                    ld(arg_mip),
                    ld(arg_mip + 4),
                    ld(arg_xip) if has_extra else zero,
                    ld(arg_xip + 4) if has_extra else zero,
                ]

            def unit_ptrs_fin(raw):
                ms = uni(raw[0])
                xs = uni(raw[2])
                return [ms, uni(raw[1]) - ms, xs, uni(raw[3]) - xs]

            def unit_info(geom, ptrs):
                info = SimpleNamespace(**vars(geom))
                info.ms, info.n_main, info.xs, n_x = ptrs
                n_row = info.n_main + n_x
                info.nt = info.live.select((n_row + (BN - 1)) // BN, fx.Int32(0))
                info.n_unit = (info.nt > 0).select(n_row, fx.Int32(0))
                info.kmax = (info.n_unit > 0).select(info.n_unit - 1, fx.Int32(0))
                return info

            def positions(nthreads):
                """(list position, may pass MAXK) of this thread's entries when ``nthreads`` threads stage a list."""
                return [
                    (tx + nthreads * k, nthreads * (k + 1) > MAXK)
                    for k in range_constexpr(-(-MAXK // nthreads))
                ]

            def stage_load(info, kps):
                """The slot ids of the list positions ``kps`` (clamped to the unit's last key): global loads."""
                has_keys = info.n_unit > 0
                out = []
                for kp, _ in kps:
                    kpg = (kp < info.n_unit).select(kp, info.kmax)
                    if const_expr(has_extra):
                        in_main = kpg < info.n_main
                        sptr = in_main.select(
                            arg_mid + fx.Int64(info.ms + kpg) * 4,
                            arg_xid + fx.Int64(info.xs + kpg - info.n_main) * 4,
                        )
                    else:
                        in_main = kpg >= 0
                        sptr = arg_mid + fx.Int64(info.ms + kpg) * 4
                    # a unit without keys (an empty row, or one past the last row) has no valid list position: its
                    # position 0 is one past the row's lists (past the end of an index tensor, which may be empty).
                    # It reads the indptr instead; the ids are never used.
                    sptr = has_keys.select(sptr, fx.Int64(arg_mip))
                    out.append(_gld_i32(sptr))
                    out.append(in_main.select(fx.Int32(1), fx.Int32(0)))
                return out

            def stage_store(loaded, kps, buf):
                """Cache row index (bit 31: extra cache) of every position into the LDS list ``buf``."""
                for i_, (kp, tail) in enumerate(kps):
                    slot = loaded[2 * i_]
                    in_main = loaded[2 * i_ + 1] != 0
                    if const_expr(has_extra):
                        lg = in_main.select(i32_m_lg, i32_x_lg)
                        st = in_main.select(i32_m_st, i32_x_st)
                    else:
                        lg, st = i32_m_lg, i32_m_st
                    rowi = (slot >> lg) * st + (slot & ((fx.Int32(1) << lg) - 1))
                    if const_expr(PERSIST):
                        word = rowi
                        if const_expr(has_extra):
                            word = rowi | in_main.select(
                                fx.Int32(0), fx.Int32(-2147483648)
                            )
                        # the list holds MAXK entries: a chunk that runs past it stores only its first positions
                        if const_expr(tail):
                            if kp < MAXK:
                                _st_i32(buf + kp * 4, word)
                        else:
                            _st_i32(buf + kp * 4, word)
                    else:
                        base = (
                            in_main.select(fx.Int64(arg_main), fx.Int64(arg_xtra))
                            if has_extra
                            else fx.Int64(arg_main)
                        )
                        addr = base + fx.Int64(rowi) * D
                        if const_expr(tail):
                            if kp < MAXK:
                                fx.ptr_store(
                                    addr, _ptr(_SHARED, buf + kp * 8, 8, T.i64)
                                )
                        else:
                            fx.ptr_store(addr, _ptr(_SHARED, buf + kp * 8, 8, T.i64))

            geom = unit_geom(unit_i)
            warm = fx.Int32(0) > fx.Int32(1)
            ptrs = [fx.Int32(0) for _ in range_constexpr(4)]
            if const_expr(PERSIST):
                if fx.Int32(unit_i) > 0:
                    warm = uni(_ld_i32(lds_rec)) != 0
                if warm:
                    ptrs = [
                        uni(_ld_i32(lds_rec + 4 * (i_ + 1)))
                        for i_ in range_constexpr(4)
                    ]
                else:
                    ptrs = unit_ptrs(geom)
            else:
                ptrs = unit_ptrs(geom)
            info = unit_info(geom, ptrs)
            live, rowc, head_base, nt, n_unit = (
                info.live,
                info.rowc,
                info.head_base,
                info.nt,
                info.n_unit,
            )
            ids_cur = lds_ids + (
                (fx.Int32(unit_i) % 2) * (MAXK * 4) if PERSIST else fx.Int32(0)
            )
            ids_nxt = lds_ids + ((fx.Int32(unit_i) + 1) % 2) * (MAXK * 4)
            kps_all = positions(64 * NWV)
            if const_expr(PERSIST):
                if not warm:
                    stage_store(stage_load(info, kps_all), kps_all, ids_cur)
            else:
                stage_store(stage_load(info, kps_all), kps_all, ids_cur)

            # ---------------- QK waves: head block 16 hgw, keys 32 kg .. + 31 of every tile ----------------
            Q = SimpleNamespace()

            def setup_qk():
                hgw, Q.kg = w8 % HSLOT, w8 // HSLOT
                Q.on = hgw < NHGW
                rowk = [32 * Q.kg + 16 * kb + i16 for kb in range_constexpr(2)]
                Q.koff = [
                    [
                        rowk[kb] * D + (((2 * q4 + hf) ^ swz(rowk[kb])) * 16)
                        for hf in range_constexpr(2)
                    ]
                    for kb in range_constexpr(2)
                ]
                Q.head = [16 * (HBW * hgw + hb) + i16 for hb in range_constexpr(HBW)]
                Q.pw = [
                    [
                        [
                            (term * HG + Q.head[hb]) * PPITCH
                            + 32 * Q.kg
                            + 16 * kb
                            + 4 * q4
                            for term in range_constexpr(p_terms)
                        ]
                        for kb in range_constexpr(2)
                    ]
                    for hb in range_constexpr(HBW)
                ]

            def qk(slot, qfrag):
                base = lds_kv + slot * TILE_B

                def kfrag(s):
                    return [
                        _cat8(
                            _ld128(base + Q.koff[kb][0] + 128 * s),
                            _ld128(base + Q.koff[kb][1] + 128 * s),
                        )
                        for kb in range_constexpr(2)
                    ]

                S = [[zero4, zero4] for _ in range_constexpr(HBW)]
                cur = kfrag(0)
                for s in range_constexpr(4):
                    nxt = kfrag(s + 1) if s < 3 else None
                    for kb in range_constexpr(2):
                        for hb in range_constexpr(HBW):
                            S[hb][kb] = _mfma16(cur[kb], qfrag[hb][s], S[hb][kb])
                    cur = nxt
                return S

            def e8m0i(e):
                byte = e + 127
                byte = (byte > 254).select(fx.Int32(254), byte)
                return (byte < 0).select(fx.Int32(0), byte)

            def masked(S, hb, c):
                sv = [
                    S[hb][kb][r]
                    for kb in range_constexpr(2)
                    for r in range_constexpr(4)
                ]
                if c == nt - 1:
                    rem = n_unit - c * BN - 32 * Q.kg - 4 * q4
                    sv = [
                        (fx.Int32(16 * kb + r) < rem).select(sv[4 * kb + r], neg_inf)
                        for kb in range_constexpr(2)
                        for r in range_constexpr(4)
                    ]
                return sv

            def lmax(sv):
                t4 = [_maxf_nn(sv[2 * i], sv[2 * i + 1]) for i in range_constexpr(4)]
                return _maxf_nn(_maxf_nn(t4[0], t4[1]), _maxf_nn(t4[2], t4[3]))

            def exps(sv, m_use):
                nms = -(m_use > neg_inf).select(m_use - PQ, fx.Float32(0.0))
                return [_exp2(_afma(x, c_sc, nms)) for x in sv]

            def lsum8(ps):
                t4 = [_aadd(ps[i], ps[i + 4]) for i in range_constexpr(4)]
                t2 = [_aadd(t4[0], t4[2]), _aadd(t4[1], t4[3])]
                return _aadd(t2[0], t2[1])

            def write_p(ps, hb, buf):
                pbase = lds_p + buf * P_BUF
                ph = [
                    _pack4_fp8(ps[4 * t], ps[4 * t + 1], ps[4 * t + 2], ps[4 * t + 3])
                    for t in range_constexpr(2)
                ]
                for t in range_constexpr(2):
                    _st_i32(pbase + Q.pw[hb][t][0], ph[t])
                if const_expr(p_terms == 2):
                    for t in range_constexpr(2):
                        h01 = _unpack2_fp8(ph[t], False)
                        h23 = _unpack2_fp8(ph[t], True)
                        pl = _pack4_fp8(
                            _asub_t(ps[4 * t], h01[0]),
                            _asub_t(ps[4 * t + 1], h01[1]),
                            _asub_t(ps[4 * t + 2], h23[0]),
                            _asub_t(ps[4 * t + 3], h23[1]),
                        )
                        _st_i32(pbase + Q.pw[hb][t][1], pl)

            def mu_addr(buf, kgi, hb):
                return lds_mu + (((buf * 2 + kgi) * H + Q.head[hb]) * 4)

            def softmax_first(S):
                """Tile 0: the block's own exponent (max score + SLACK - LAZY, rounded up)."""
                st = []
                for hb in range_constexpr(HBW):
                    sv = masked(S, hb, fx.Int32(0))
                    mx = _xlane4(lmax(sv), _maxf_nn)
                    m_use = fx.math.ceil(mx * c_sc - LAZY + SLACK)
                    ps = exps(sv, m_use)
                    write_p(ps, hb, fx.Int32(0))
                    _st_f32(mu_addr(fx.Int32(0), Q.kg, hb), m_use)
                    st += [neg_inf, m_use, lsum8(ps)]
                return st

            def softmax_next(S, c, buf, state):
                """Tile c >= 1: P relative to the reference of the earlier tiles, recomputed with the block's own exponent
                only where a score exceeds it by more than LAZY."""
                pb = (c - 1) % 2
                svs, Rns, alphas, pss, ovfs = [], [], [], [], []
                for hb in range_constexpr(HBW):
                    R = state[3 * hb]
                    mmax = _maxf_nn(
                        _ld_f32(mu_addr(pb, 0, hb)), _ld_f32(mu_addr(pb, 1, hb))
                    )
                    Rn = (mmax > R).select(mmax, R)
                    alphas.append(
                        (mmax - R > EMAX).select(fx.Float32(1.0), _exp2(R - Rn))
                    )
                    sv = masked(S, hb, c)
                    svs.append(sv)
                    Rns.append(Rn)
                    pss.append(exps(sv, Rn))
                    ovfs.append(lmax(sv) * c_sc > Rn + LAZY)
                ovf_any = ovfs[0] | ovfs[1] if const_expr(HBW == 2) else ovfs[0]
                m_uses = list(Rns)
                Fns = list(Rns)
                xis = [fx.Int32(0) for _ in range_constexpr(HBW)]
                words = [alphas[hb].bitcast(fx.Int32) for hb in range_constexpr(HBW)]
                if fx.Int64(_rocdl.ballot(T.i64, ovf_any)) != fx.Int64(0):
                    m_uses = []
                    pss = []
                    Fns = []
                    xis = []
                    words = []
                    for hb in range_constexpr(HBW):
                        umax = _xlane4(lmax(svs[hb]), _maxf_nn) * c_sc
                        mu_ = (umax > Rns[hb] + LAZY).select(
                            fx.math.ceil(umax - LAZY + SLACK), Rns[hb]
                        )
                        m_uses.append(mu_)
                        pss.append(exps(svs[hb], mu_))
                        x = mu_ - Rns[hb]
                        xi = x.to(fx.Int32)
                        xis.append(xi)
                        # the block's own exponent is the new reference of its sum when it is more than EMAX above Rn
                        Fns.append((x > EMAX).select(mu_, Rns[hb]))
                        # a block above the reference sends the PV waves to their slow path through alpha < 1
                        ab = alphas[hb].bitcast(fx.Int32)
                        words.append(
                            ab
                            - ((x > fx.Float32(0.0)) & (ab != 0)).select(
                                fx.Int32(1), fx.Int32(0)
                            )
                        )
                out = []
                for hb in range_constexpr(HBW):
                    write_p(pss[hb], hb, buf)
                    _st64(
                        lds_pr + ((buf * H + Q.head[hb]) * 4 + Q.kg * 2) * 4,
                        Vec.from_elements([xis[hb], words[hb]], fx.Int32),
                    )
                    _st_f32(mu_addr(buf, Q.kg, hb), m_uses[hb])
                    L_new = fx.math.fma(
                        lsum8(pss[hb]),
                        _exp2(m_uses[hb] - Fns[hb]),
                        state[3 * hb + 2] * _exp2(state[3 * hb + 1] - Fns[hb]),
                    )
                    out += [Rns[hb], Fns[hb], L_new]
                return out

            # ---------------- PV waves: dims 128 dg .. + 127 of the 32 heads of head tile ht0 ----------------
            wp = w8 - NQW
            dg = wp % 4
            ht0 = wp // 4 if NPW == 8 else 0
            Pc = SimpleNamespace()

            def setup_pv():
                vk = (lane % 16) // 2
                vb4 = (lane // 16) % 2
                fv = (vk & 3) | ((((vk >> 2) ^ hh) & 1) << 2)
                Pc.voff = [
                    (16 * hh + vk) * D
                    + ((8 * dg + ((2 * d + vb4) ^ fv)) * 16)
                    + 8 * (lane % 2)
                    for d in range_constexpr(4)
                ]
                Pc.pr_off = [
                    (l32 if HG == 16 else (term * HG + 32 * ht0 + l32)) * PPITCH
                    + 16 * hh
                    for term in range_constexpr(PV_TERMS)
                ]
                Pc.head = (l32 % 16) if HG == 16 else (32 * ht0 + l32)

            dma_off = [
                ((lane % 32) ^ swz(ROWS_W * wp + 2 * jj + hh)) * 16
                for jj in range_constexpr(DMA_PER_WAVE)
            ]
            kp_off = [ROWS_W * wp + 2 * jj + hh for jj in range_constexpr(DMA_PER_WAVE)]

            def dma_step(c, jj, slot, ids_b=None):
                kp = c * BN + kp_off[jj]
                kp = (kp > MAXK - 1).select(fx.Int32(MAXK - 1), kp)
                ids = ids_cur if ids_b is None else ids_b
                if const_expr(PERSIST):
                    w = _ld_i32(ids + kp * 4)
                    if const_expr(has_extra):
                        a = (
                            fx.Int64(arg_main)
                            + fx.Int64(w & fx.Int32(0x7FFFFFFF)) * D
                            + fx.Int64((w >> 31) & fx.Int32(1))
                            * (fx.Int64(arg_xtra) - fx.Int64(arg_main))
                        )
                    else:
                        a = fx.Int64(arg_main) + fx.Int64(w) * D
                else:
                    a = fx.Int64(
                        fx.ptr_load(_ptr(_SHARED, ids + kp * 8, 8, T.i64), fx.Int64)
                    )
                _dma16(
                    a + fx.Int64(dma_off[jj]),
                    uni(lds_kv + slot * TILE_B + (DMA_PER_WAVE * wp + jj) * 1024),
                )

            def wait_dma(later):
                _waitcnt_vm(DMA_PER_WAVE * later)

            def pv_mm(slot, buf, acc, scale, fill):
                base = lds_kv + slot * TILE_B
                pbase = lds_p + buf * P_BUF
                bf = [
                    _cat8(
                        _ld128(pbase + Pc.pr_off[term]),
                        _ld128(pbase + Pc.pr_off[term] + 32),
                    )
                    for term in range_constexpr(PV_TERMS)
                ]
                out = []
                for d in range_constexpr(4):
                    parts = []
                    for u in range_constexpr(4):
                        r = _tr8(base + Pc.voff[d] + (u // 2) * 16384 + (u % 2) * 4096)
                        parts += [r[0], r[1]]
                    a8 = Vec.from_elements(parts, fx.Int32)
                    x = acc[d]
                    for term in range_constexpr(PV_TERMS):
                        x = _mfma32s(a8, bf[term], x, scale)
                        if fill is not None:
                            fill()
                    out.append(x)
                return out

            def e8m0(e):
                return e8m0i(e.to(fx.Int32))

            def pv_first(slot, buf, fill=None):
                m0 = _ld_f32(lds_mu + ((buf * 2 + 0) * H + Pc.head) * 4)
                m1 = _ld_f32(lds_mu + ((buf * 2 + 1) * H + Pc.head) * 4)
                e = (hh > 0).select(m1, m0) - _maxf_nn(m0, m1)
                return pv_mm(
                    slot,
                    buf,
                    init_acc,
                    e8m0((e > fx.Float32(-200.0)).select(e, fx.Float32(-200.0))),
                    fill,
                )

            def pv_next(slot, buf, acc, fill=None):
                rec = _ld128(lds_pr + ((buf * H + Pc.head) * 4) * 4)
                alpha = fx.min(
                    fx.Int32(rec[1]).bitcast(fx.Float32),
                    fx.Int32(rec[3]).bitcast(fx.Float32),
                )
                scale = fx.Int32(127)
                acc_s = acc
                if fx.Int64(_rocdl.ballot(T.i64, alpha < fx.Float32(1.0))) != fx.Int64(
                    0
                ):
                    # the reference moved or a block's exponent is above it (more than EMAX: the accumulator restarts)
                    x0, x1 = fx.Int32(rec[0]), fx.Int32(rec[2])
                    xm = (x0 > x1).select(x0, x1)
                    x = (hh > 0).select(x1, x0)
                    big = xm > fx.Int32(int(EMAX))
                    scale = e8m0i(big.select(x - xm, x))
                    alpha = big.select(fx.Float32(0.0), alpha)
                    acc_s = [
                        Vec.from_elements(
                            [_amul(acc[d][e], alpha) for e in range_constexpr(16)],
                            fx.Float32,
                        )
                        for d in range_constexpr(4)
                    ]
                return pv_mm(slot, buf, acc_s, scale, fill)

            def finalize(acc_f, m_f, l_f):
                """O / (sum + sink) of this PV wave's dims, as bf16 rows (global, bf16 mode) or staged in LDS."""
                rv = ks / (l_f + _exp2(sink2 - m_f + PQ))
                if const_expr(COLTERM):
                    comb = []
                    for d in range_constexpr(4):
                        el = []
                        for e in range_constexpr(16):
                            a, b = _permlane_swap(
                                _rocdl.permlane16_swap,
                                _bits(acc_f[d][e]),
                                _bits(acc_f[d][e]),
                            )
                            el.append(_as_f32(a) + _as_f32(b))
                        comb.append(Vec.from_elements(el, fx.Float32))
                    acc_f = comb
                if const_expr(not mxfp8):
                    out_w = _buffer(arg_out, fx.Int64(i32_T) * (H * D * 2), T.i32, 16)
                    st8 = fx.make_copy_atom(fx.rocdl.BufferCopy64b(), 32)

                    def store_rows():
                        for d in range_constexpr(4):
                            x = acc_f[d]
                            head = head_base + Pc.head
                            for a in range_constexpr(4):
                                lo = _pk_bf16(x[4 * a] * rv, x[4 * a + 1] * rv)
                                hi = _pk_bf16(x[4 * a + 2] * rv, x[4 * a + 3] * rv)
                                dim = 128 * dg + 32 * d + 8 * a + 4 * hh
                                off = ((rowc * H + head) * D + dim) // 2
                                _copy_out(st8, [lo, hi], fx.Int32, out_w, off)

                    if const_expr(HG == 16):
                        # lanes 16..31 hold the residual columns, added in above
                        if l32 < 16:
                            store_rows()
                    else:
                        store_rows()
                else:
                    # rows of 1 KB at the ring's slots 2 and 3, the 16-byte chunk index XORed with the row (bank-conflict free)
                    r8 = Pc.head & 7
                    orow = lds_kv + OSTAGE + Pc.head * 1024 + 256 * dg + 8 * hh
                    xr = [16 * (c0 ^ r8) for c0 in range_constexpr(8)]
                    for d in range_constexpr(4):
                        x = acc_f[d]
                        for a in range_constexpr(4):
                            lo = _pk_bf16(x[4 * a] * rv, x[4 * a + 1] * rv)
                            hi = _pk_bf16(x[4 * a + 2] * rv, x[4 * a + 3] * rv)
                            c_ = 4 * d + a
                            addr = orow + (16 * (c_ - (c_ & 7))) + xr[c_ & 7]
                            if const_expr(HG == 16):
                                if l32 < 16:
                                    _st64(addr, Vec.from_elements([lo, hi], fx.Int32))
                            else:
                                _st64(addr, Vec.from_elements([lo, hi], fx.Int32))

            def load_cs():
                """cos / sin of the 16 pairs of this lane's rope block (waves 7 and 15 quantize dims 448..511)."""
                pos = uni(_ld_inv_i32(arg_pos + fx.Int64(rowc) * i32_pos_b))
                cs_row = arg_cs + fx.Int64(pos) * fx.Int64(i32_cs_st) * 4
                vv = []
                for half in range_constexpr(2):
                    for q_ in range_constexpr(4):
                        c4 = _gld_f32x4(
                            cs_row + fx.Int64(32 * half + 16 * (lane % 2) + 4 * q_) * 4
                        )
                        vv += [fx.Float32(c4[k]) for k in range_constexpr(4)]
                return vv

            def quantize_rows():
                """All waves: the bf16 rows in LDS -> inverse RoPE -> MXFP8 codes and E8M0 scales. A task is the block
                pair 2 (t % 8), + 1 of the heads 32 (t // 8) .. + 31 (wave w: the tasks w, w + NWV).
                """
                codes_w = _buffer(arg_out, fx.Int64(i32_T) * (H * D), T.i32, 16)
                scales_b = _buffer(arg_osc, fx.Int64(i32_T) * (H * D // 32), T.i8, 1)
                st16 = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), 32)
                st1 = fx.make_copy_atom(fx.rocdl.BufferCopy8b(), 8)

                def task(t):
                    blk = 2 * (t % 8) + lane % 2
                    head = lane // 2 + 32 * (t // 8)
                    base = lds_kv + OSTAGE + head * 1024 + 128 * (t % 8)
                    t8 = (4 * (lane % 2)) ^ (head & 7)
                    dws = []
                    for j in range_constexpr(4):
                        v4 = _ld128(base + 16 * (t8 ^ j))
                        dws += [v4[0], v4[1], v4[2], v4[3]]
                    vals = []
                    for j in range_constexpr(16):
                        vals += [
                            _as_f32(fx.Int32(dws[j]) << 16),
                            _as_f32(fx.Int32(dws[j]) & fx.Int32(-65536)),
                        ]
                    if t % 8 == 7:
                        rot = []
                        for i in range_constexpr(16):
                            ev, od = vals[2 * i], vals[2 * i + 1]
                            rot += [
                                fx.math.fma(ev, cs_vals[i], od * cs_vals[16 + i]),
                                fx.math.fma(od, cs_vals[i], -(ev * cs_vals[16 + i])),
                            ]
                        vals = rot
                    amax = fx.Float32(TINY_F32)
                    for i in range_constexpr(32):
                        amax = _maxf_nn(
                            amax, _as_f32(_bits(vals[i]) & fx.Int32(0x7FFFFFFF))
                        )
                    sbf = fx.math.ceil(
                        fx.math.log2(amax / fx.Float32(448.0))
                    ) + fx.Float32(127.0)
                    sbf = fx.min(fx.max(sbf, fx.Float32(0.0)), fx.Float32(254.0))
                    rs = fx.Float32(fx.math.exp2(fx.Float32(127.0) - sbf))
                    ok = head < HG
                    dw8 = [
                        _pack4_fp8(
                            vals[4 * a] * rs,
                            vals[4 * a + 1] * rs,
                            vals[4 * a + 2] * rs,
                            vals[4 * a + 3] * rs,
                        )
                        for a in range_constexpr(8)
                    ]
                    off = ((rowc * H + head_base + head) * D + blk * 32) // 4
                    if ok:
                        _copy_out(st16, dw8[0:4], fx.Int32, codes_w, off)
                        _copy_out(st16, dw8[4:8], fx.Int32, codes_w, off + 4)
                        _copy_out(
                            st1,
                            [sbf.to(fx.Int32).to(fx.Int8)],
                            fx.Int8,
                            scales_b,
                            rowc * (H * D // 32) + (head_base + head) * (D // 32) + blk,
                        )

                for i_ in range_constexpr(-(-NTASK // NWV)):
                    if w8 + NWV * i_ < NTASK:
                        task(w8 + NWV * i_)

            rope_wave = (w8 == 7) | (
                (w8 == 3) & (NTASK > NWV)
            )  # the waves whose tasks quantize dims 448..511
            sink2 = fx.Float32(0.0)
            cs_vals = [fx.Float32(0.0) for _ in range_constexpr(32)]
            if not isq:
                sink2 = (
                    _gld_f32(
                        arg_sink
                        + fx.Int64(
                            head_base + ((l32 % 16) if HG == 16 else (32 * ht0 + l32))
                        )
                        * 4
                    )
                    * LOG2E
                )
            init_acc = [zero16 for _ in range_constexpr(4)]
            m_f = neg_inf
            l_f = fx.Float32(0.0)
            acc_f = init_acc
            if isq:
                setup_qk()
                kps_q = positions(64 * NQW)
                nxt = [fx.Int32(0) for _ in range_constexpr(2 * (MAXK // (64 * NQW)))]
                has_nq = fx.Int32(0) > fx.Int32(1)
                if const_expr(PERSIST):
                    geom_nq = unit_geom(fx.Int32(unit_i) + 1)
                    raw_nq = unit_ptrs_raw(geom_nq)
                if Q.on:
                    if nt > 0:
                        q_w = _buffer(arg_q, fx.Int64(i32_T) * (H * D), T.i32, 16)
                        ld16 = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), 32)
                        qfrag = []
                        for hb in range_constexpr(HBW):
                            q_dw = (rowc * H + head_base + Q.head[hb]) * (
                                D // 4
                            ) + 8 * q4
                            fr = []
                            for s_ in range_constexpr(4):
                                lo = _copy_in(ld16, q_w, q_dw + 32 * s_, 4, fx.Int32)
                                hi = _copy_in(
                                    ld16, q_w, q_dw + 32 * s_ + 4, 4, fx.Int32
                                )
                                fr.append(_cat8(lo, hi))
                            qfrag.append(fr)
                        _barrier()
                        _barrier()
                        S0 = qk(fx.Int32(0), qfrag)
                        st0 = softmax_first(S0)
                        _barrier()
                        if const_expr(PERSIST):
                            info_nq = unit_info(geom_nq, unit_ptrs_fin(raw_nq))
                            has_nq = info_nq.live & (info_nq.nt > 0)
                            if has_nq:
                                nxt = stage_load(info_nq, kps_q)
                        results = [fx.as_ir_value(x) for x in st0]
                        for it, st_ in range(  # type: ignore[call-overload]
                            fx.Index(1),
                            fx.Index(nt),
                            fx.Index(1),
                            init=[fx.as_ir_value(x) for x in st0],
                        ):
                            c = fx.Int32(it)
                            S = qk(c % NS, qfrag)
                            nst = softmax_next(
                                S,
                                c,
                                c % 2,
                                tuple(fx.Float32(st_[i]) for i in range(3 * HBW)),
                            )
                            _barrier()
                            results = yield [fx.as_ir_value(x) for x in nst]
                        pb = (nt - 1) % 2
                        for hb in range_constexpr(HBW):
                            R_, F_, L_ = (
                                fx.Float32(results[3 * hb + i]) for i in range(3)
                            )
                            m0 = _ld_f32(mu_addr(pb, 0, hb))
                            m1 = _ld_f32(mu_addr(pb, 1, hb))
                            mm = _maxf_nn(m0, m1)
                            Rl = (mm - R_ > EMAX).select(mm, R_)
                            _st_f32(
                                lds_fin + (Q.kg * H + Q.head[hb]) * 4,
                                _xlane4(L_ * _exp2(F_ - Rl), _aadd),
                            )
                            _st_f32(lds_fin + (2 * H + Q.head[hb]) * 4, Rl)
                        if const_expr(PERSIST):
                            if has_nq:
                                stage_store(nxt, kps_q, ids_nxt)
                    else:
                        _barrier()
                        if const_expr(PERSIST):
                            info_nq = unit_info(geom_nq, unit_ptrs_fin(raw_nq))
                            if info_nq.live & (info_nq.nt > 0):
                                stage_store(stage_load(info_nq, kps_q), kps_q, ids_nxt)
                else:
                    _barrier()
                    if nt > 0:
                        _barrier()
                        _barrier()
                        for it_ in range(fx.Index(1), fx.Index(nt), fx.Index(1)):  # type: ignore[call-overload]
                            _barrier()
                _barrier()
                if const_expr(mxfp8):
                    if rope_wave:
                        cs_vals = load_cs()
            else:
                setup_pv()
                _barrier()
                if nt > 0:
                    if const_expr(PERSIST):
                        if warm:
                            for jj in range_constexpr(DMA_PER_WAVE):
                                dma_step(fx.Int32(NS - 2), jj, fx.Int32(NS - 2))
                        else:
                            for c0_ in range_constexpr(NS - 1):
                                for jj in range_constexpr(DMA_PER_WAVE):
                                    dma_step(fx.Int32(c0_), jj, fx.Int32(c0_))
                    else:
                        for c0_ in range_constexpr(NS - 1):
                            for jj in range_constexpr(DMA_PER_WAVE):
                                dma_step(fx.Int32(c0_), jj, fx.Int32(c0_))
                has_next = fx.Int32(0) > fx.Int32(1)
                raw_n = [fx.Int32(0) for _ in range_constexpr(4)]
                if const_expr(PERSIST):
                    geom_n = unit_geom(fx.Int32(unit_i) + 1)
                    raw_n = unit_ptrs_raw(geom_n)
                if nt > 0:
                    wait_dma(NS - 2)
                    _barrier()
                    wait_dma(NS - 3)
                    _barrier()
                    acc_f = init_acc
                    if nt > 1:
                        cnt = [0]

                        def fill1():
                            k = cnt[0]
                            cnt[0] += 1
                            if k >= 2 and k - 2 < DMA_PER_WAVE:
                                dma_step(
                                    fx.Int32(NS - 1), k - 2, fx.Int32((NS - 1) % NS)
                                )
                            _rocdl.sched_barrier(0)

                        acc_1 = pv_first(fx.Int32(0), fx.Int32(0), fill1)
                        for _ in range_constexpr(DMA_PER_WAVE + 2 - cnt[0]):
                            fill1()
                        wait_dma(NS - 3)
                        _barrier()
                        results = [fx.as_ir_value(x) for x in acc_1]
                        for it, st_ in range(  # type: ignore[call-overload]
                            fx.Index(2),
                            fx.Index(nt),
                            fx.Index(1),
                            init=[fx.as_ir_value(x) for x in acc_1],
                        ):
                            acc_t = [Vec(st_[i]) for i in range_constexpr(4)]
                            c = fx.Int32(it)
                            dslot = (c + (NS - 2)) % NS
                            cnt = [0]

                            def fill():
                                k = cnt[0]
                                cnt[0] += 1
                                if k >= 2 and k - 2 < DMA_PER_WAVE:
                                    dma_step(c + (NS - 2), k - 2, dslot)
                                _rocdl.sched_barrier(0)

                            acc_n = pv_next((c - 1) % NS, (c - 1) % 2, acc_t, fill)
                            for _ in range_constexpr(DMA_PER_WAVE + 2 - cnt[0]):
                                fill()
                            wait_dma(NS - 3)
                            _barrier()
                            results = yield [fx.as_ir_value(x) for x in acc_n]
                        acc_l = [Vec(results[i]) for i in range_constexpr(4)]
                        acc_f = pv_next((nt - 1) % NS, (nt - 1) % 2, acc_l)
                    else:
                        acc_f = pv_first(fx.Int32(0), fx.Int32(0))
                    wait_dma(0)
                if const_expr(PERSIST):
                    ptrs_n = unit_ptrs_fin(raw_n)
                    info_n = unit_info(geom_n, ptrs_n)
                    has_next = info_n.live & (info_n.nt > 0)
                    if tx == 64 * NQW:
                        _st_i32(lds_rec, has_next.select(fx.Int32(1), fx.Int32(0)))
                        for i_ in range_constexpr(4):
                            _st_i32(lds_rec + 4 * (i_ + 1), ptrs_n[i_])
                _barrier()
                if const_expr(PERSIST):
                    if has_next:
                        for t_ in range_constexpr(NS - 2):
                            for jj in range_constexpr(DMA_PER_WAVE):
                                dma_step(fx.Int32(t_), jj, fx.Int32(t_), ids_nxt)
                if nt > 0:
                    m_f = _ld_f32(lds_fin + (2 * H + Pc.head) * 4)
                    l_f = _ld_f32(lds_fin + (0 * H + Pc.head) * 4) + _ld_f32(
                        lds_fin + (1 * H + Pc.head) * 4
                    )
                if const_expr(mxfp8):
                    if rope_wave:
                        cs_vals = load_cs()
                if live:
                    finalize(acc_f, m_f, l_f)
            if const_expr(mxfp8):
                _barrier()
                if live:
                    quantize_rows()

    @flyc.jit
    def launch(
        arg_q: fx.Int64,
        arg_main: fx.Int64,
        arg_xtra: fx.Int64,
        arg_mip: fx.Int64,
        arg_mid: fx.Int64,
        arg_xip: fx.Int64,
        arg_xid: fx.Int64,
        arg_sink: fx.Int64,
        arg_pos: fx.Int64,
        arg_cs: fx.Int64,
        arg_out: fx.Int64,
        arg_osc: fx.Int64,
        arg_qs: fx.Int64,
        arg_ks: fx.Int64,
        i32_T: fx.Int32,
        i32_units_x: fx.Int32,
        i32_wgs_x: fx.Int32,
        i32_m_lg: fx.Int32,
        i32_m_st: fx.Int32,
        i32_x_lg: fx.Int32,
        i32_x_st: fx.Int32,
        i32_cs_st: fx.Int32,
        i32_pos_b: fx.Int32,
        f32_sm: fx.Float32,
        i32_grid: fx.Int32,
        stream: fx.Stream,
    ):
        kernel(
            arg_q,
            arg_main,
            arg_xtra,
            arg_mip,
            arg_mid,
            arg_xip,
            arg_xid,
            arg_sink,
            arg_pos,
            arg_cs,
            arg_out,
            arg_osc,
            arg_qs,
            arg_ks,
            i32_T,
            i32_units_x,
            i32_wgs_x,
            i32_m_lg,
            i32_m_st,
            i32_x_lg,
            i32_x_st,
            i32_cs_st,
            i32_pos_b,
            f32_sm,
        ).launch(grid=(i32_grid, 1, 1), block=(64 * NWV, 1, 1), stream=stream)

    launch.kernel_name = name
    launch.compile_hints = {"waves_per_eu": -(-NWV // 4), **PRELOAD_HINTS}
    return launch
