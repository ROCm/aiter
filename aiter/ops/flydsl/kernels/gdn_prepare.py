# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL GDN prefill chunk-prepare kernel, fusing K1..K4 (BT=64 main path).

For each chunk of ``BT`` tokens, with ``C = (I + A)^-1`` and ``A`` the strictly
lower-triangular gated KKT matrix
``A[s, r] = (k_s . k_r) * beta_s * exp(g_cumsum_s - g_cumsum_r)`` for ``s > r``:

    g_cumsum : [B, T, H]    fp32   in-chunk inclusive prefix sum of g
    w_bar    : [B, T, H, K] bf16   = C @ (k * beta * exp(g_cumsum))
    u_bar    : [B, T, H, V] bf16   = C @ (v * beta)

This is the single-kernel equivalent of the Triton
``fused_chunk_local_cumsum_scaled_dot_kkt_fwd`` + ``fused_solve_tril_recompute_w_u``
pair, and keeps that pair's token-major output layout and natural-log
``g_cumsum`` domain.

Single production build: MFMA bf16 16x16x16 for the hot GEMMs and the
triangular inverse, on the Opus-style LDS layout (full-K staging, fp32 ``s_A``,
in-place Schur merge, register-cached C) with a wave-local vec8 output epilogue,
a cached K scale, an elided final-K fence, native exp2, and the SR=16 ``s_vT``
bank-swizzle.

There is exactly one build: every A/B variant and tuning knob this kernel was
developed through (Horner base, ``lds_opt``, ``ktile``, the Opt4
tail/alias/partial/rhs4 controls, the varlen flat chunk-list launch, and the
swizzle SR/MULT/address-specialisation env overrides) has been removed after
being measured; the surviving choices are the ones that won.

NOTE: the torch host wrapper, the XCD probe and the anti-camping grid padding
live in ``aiter.ops.flydsl.linear_attention_prefill_kernels`` to keep this
module free of any ``torch`` dependency (mirrors the layering used by
``aiter.ops.flydsl.kernels.chunk_gated_delta_h``).
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import (
    arith,
    buffer_ops,
    const_expr,
    gpu,
    range_constexpr,
    rocdl,
    vector,
)
from flydsl.expr.numeric import Numeric
from flydsl.expr.typing import T as Tir  # MLIR type helpers
from flydsl.runtime.device import get_rocm_arch
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr

from .tensor_shim import GTensor, STensor


def _exp2_f32(x):
    """Base-2 exponential lowering to a single ``v_exp_f32``."""
    return fx.Float32(rocdl.exp2(Tir.f32, x.ir_value()))


WARP_SIZE = 64
BLOCK_THREADS = 256  # 4 warps


# ---------------------------------------------------------------------------
# Naming of the access helpers below -- they compose as <space><op><shape>:
#   _g… / _s…  global memory / LDS        …ld / …st  load / store
#   …2         2-D (row, col) indexing    …_vec      n contiguous elements
#   …_vt       carries s_vT's coupled column-group XOR swizzle (see _swz_vt)
# Two things do not follow it: `_ld` reads LDS despite the missing `s`, and the
# MFMA fragment helpers spell the verb out (`_load_mfma_tile`, `_store_fp32_tile`)
# because they move a whole 16x16 tile rather than one address.  All of them are
# calls rather than inline subscripting, for the reason spelled out in `_sst`.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# MFMA bf16 16x16x16 helpers (stage-2). Faithful FlyDSL port of the HIP
# gdn_mfma utilities (load_mfma_tile / mfma / accum_to_src / *_fp32_tile).
#
# CDNA wave-64 fragment lane map (matches the HIP kernel exactly):
#   row index within a 16x16 tile = lane % 16
#   K-group (4 contiguous elems)  = lane // 16   -> element base (lane//16)*4
#   accumulator C[m,n]: m = (lane//16)*4 + p (p in 0..3), n = lane % 16
# ---------------------------------------------------------------------------
def _F32X4():
    return arith.constant_vector(0.0, Tir.vec(4, Tir.f32))


def _BF16X4():
    return arith.constant_vector(0.0, Tir.vec(4, Tir.bf16))


def _I(x):
    """Cast an i32 (or python int) index to MLIR ``index`` for STensor vector ops."""
    return fx.Index(x)


def _ext(v, i):
    """Extract lane ``i`` from a vector at a statically known position."""
    return vector.extract(v, static_position=[i], dynamic_position=[])


def _mfma16(a_bf16x4, b_bf16x4, c_f32x4):
    """D[m,n] = sum_k A[m,k]*B[n,k] + C  (bf16 in, fp32 acc)."""
    a16 = vector.bitcast(Tir.vec(4, Tir.i16), a_bf16x4)
    b16 = vector.bitcast(Tir.vec(4, Tir.i16), b_bf16x4)
    return rocdl.mfma_f32_16x16x16bf16_1k(
        Tir.vec(4, Tir.f32), [a16, b16, c_f32x4, 0, 0, 0]
    )


def _accum_to_bf16x4(d_f32x4):
    """v4f32 accumulator -> v4bf16 operand (element-wise truncf), for MFMA chaining."""
    elems = [
        Numeric.from_python_value(_ext(d_f32x4, p)).to(fx.BFloat16) for p in range(4)
    ]
    return vector.from_elements(Tir.vec(4, Tir.bf16), elems)


def _load_mfma_tile(s_t, row_base, col_base, lane):
    """bf16 LDS -> v4bf16 fragment: 4 contiguous along the inner (K) dim."""
    row = row_base + (lane % fx.Int32(16))
    col = col_base + (lane // fx.Int32(16)) * fx.Int32(4)
    return s_t.vec_load((_I(row), _I(col)), 4)


# --- coupled XOR bank-swizzle (s_vT only) -----------------------------------
# The dominant gfx950 LDS conflict is the s_vT scatter WRITE: the scatter maps
# row0=(p%16)*4 into 4 columns/warp, so at VTS=68 it hits only 16/64 banks, 4-way.
# Swizzling the column group as cg' = cg ^ (row // SWZ_SR), identically on the
# write and the read, spreads those rows; XOR is bijective so output is unchanged.
# SWZ_SR=16 on both production archs, and it wins on *address cost* rather than
# conflict rate: the arithmetic drops this build 74 -> 64 VGPR, 6 -> 8 waves/SIMD.
# Re-tuning (VTS, SWZ_SR) against a bank-conflict probe was measured and rejected
# -- see gdn_prepare_opt_plan.md §3 (SR choice) and §6 (why the probe mispredicted).
SWZ_SR = 16


def _swz_vt(row):
    return (row // fx.Int32(SWZ_SR)) % fx.Int32(16)


def _load_mfma_tile_vt_tiled(s_t, row_tile, col_tile, n16, mb4):
    """Load one WY B tile from the swizzled ``s_vT``.

    ``row_tile``/``col_tile`` are the constexpr 16x16 tile coordinates.  At
    SWZ_SR=16 and ``row = row_tile*16 + lane%16`` the swizzle mask is exactly
    ``row_tile``.  Since both the lane-group and the mask occupy only the low two
    bits, ``((4*col_tile + lane_group) ^ row_tile) * 4`` simplifies to
    ``16*col_tile + (mb4 ^ (4*row_tile))``.  Passing the already-computed
    ``n16``/``mb4`` also prevents rebuilding lane div/rem in every unrolled load.
    """
    row = fx.Int32(row_tile * 16) + n16
    col = fx.Int32(col_tile * 16) + (mb4 ^ fx.Int32(row_tile * 4))
    return s_t.vec_load((_I(row), _I(col)), 4)


def _sst2_vt(s_t, n, j, val):
    """s_vT scalar scatter write with the matching coupled col-group XOR swizzle."""
    cg = (j // fx.Int32(4)) ^ _swz_vt(n)
    s_t[_I(n), _I(cg * fx.Int32(4) + (j % fx.Int32(4)))] = val


# --- s_out is deliberately left unswizzled and unpadded ---------------------
# Its epilogue write is 2-way bank-conflicted (~41% of the kernel's LDS bank
# conflicts), but that traffic is off the critical path: OUT_S padding (68, 72) and
# a column XOR were each measured on gfx950 and each lost on the production shapes,
# including the variant that removes the conflict outright at no LDS or occupancy
# cost.  See gdn_prepare_opt_plan.md §6 -- bank conflicts here are a red herring,
# not a bottleneck.


def _load_fp32_tile(s_t, row_base, col_base, lane):
    """fp32 LDS -> v4bf16 fragment (same lane map as _load_mfma_tile, cast to bf16)."""
    row = row_base + (lane % fx.Int32(16))
    col = col_base + (lane // fx.Int32(16)) * fx.Int32(4)
    v = s_t.vec_load((_I(row), _I(col)), 4)
    elems = [Numeric.from_python_value(_ext(v, p)).to(fx.BFloat16) for p in range(4)]
    return vector.from_elements(Tir.vec(4, Tir.bf16), elems)


def _load_fp32_tile_T(s_t, row_base, col_base, lane):
    """fp32 LDS -> v4bf16 fragment of the TRANSPOSED tile (B-operand = M^T)."""
    n = lane % fx.Int32(16)
    col = col_base + n
    kb4 = (lane // fx.Int32(16)) * fx.Int32(4)
    elems = [
        Numeric.from_python_value(s_t[_I(row_base + kb4 + fx.Int32(p)), _I(col)]).to(
            fx.BFloat16
        )
        for p in range(4)
    ]
    return vector.from_elements(Tir.vec(4, Tir.bf16), elems)


def _load_neg_fp32_tile(s_t, row_base, col_base, lane):
    """fp32 LDS -> v4bf16 fragment of (-tile) with _load_mfma_tile lane map."""
    row = row_base + (lane % fx.Int32(16))
    col = col_base + (lane // fx.Int32(16)) * fx.Int32(4)
    v = s_t.vec_load((_I(row), _I(col)), 4)
    elems = [
        (Numeric.from_python_value(_ext(v, p)) * fx.Float32(-1.0)).to(fx.BFloat16)
        for p in range(4)
    ]
    return vector.from_elements(Tir.vec(4, Tir.bf16), elems)


def _load_neg_fp32_tile_T(s_t, row_base, col_base, lane):
    """fp32 LDS -> v4bf16 fragment of (-tile)^T (the ``loadB(-A)`` operand).

    Same native [k,n] lane map as :func:`_load_fp32_tile_T`, negated.  Pairs with
    :func:`_load_neg_fp32_tile` (``loadA(-A)``) so a single
    ``_mfma16(loadA(B), loadB(B)) = B @ B`` squares ``B = -A`` with no LDS
    round-trip -- the operand feeder for the Neumann-*squaring* diagonal inverse
    (mirrors the opus HIP kernel)."""
    n = lane % fx.Int32(16)
    col = col_base + n
    kb4 = (lane // fx.Int32(16)) * fx.Int32(4)
    elems = [
        (
            Numeric.from_python_value(s_t[_I(row_base + kb4 + fx.Int32(p)), _I(col)])
            * fx.Float32(-1.0)
        ).to(fx.BFloat16)
        for p in range(4)
    ]
    return vector.from_elements(Tir.vec(4, Tir.bf16), elems)


def _store_fp32_tile(s_t, row_base, col_base, d_f32x4, lane):
    """Store a v4f32 accumulator back to fp32 LDS (accumulator lane map)."""
    n = lane % fx.Int32(16)
    mb4 = (lane // fx.Int32(16)) * fx.Int32(4)
    for p in range(4):
        s_t[_I(row_base + mb4 + fx.Int32(p)), _I(col_base + n)] = _ext(d_f32x4, p)


def _negate4(d_f32x4):
    """Element-wise negate a v4f32 accumulator."""
    elems = [
        (Numeric.from_python_value(_ext(d_f32x4, p)) * fx.Float32(-1.0))
        for p in range(4)
    ]
    return vector.from_elements(Tir.vec(4, Tir.f32), elems)


class _SeqBoundedG:
    """A :class:`GTensor` work-alike whose buffer descriptor is clamped to
    ``num_records_bytes`` rather than :class:`GTensor`'s ``max_size=True``.

    Every access past the limit is dropped by the buffer hardware -- loads
    return zero, stores are discarded -- which is exactly the ``row < seqlen``
    bound the callers would otherwise hand-roll as two selects per load (clamp
    the index to 0, then substitute a zero vector) plus an ``scf.if`` per store.
    Limits are always a whole multiple of the row stride and every access stays
    inside one row's feature dim, so an access is either entirely in range or
    entirely out of it -- it can never straddle the limit and come back
    partially zeroed.

    ``load``/``store`` mirror ``GTensor``'s; only the descriptor differs, and
    building it directly keeps the unused ``max_size`` one out of the IR."""

    def __init__(self, t, dtype, num_records_bytes):
        self.dtype = dtype
        self.cache_modifier = 0
        self.rsrc = buffer_ops.create_buffer_resource(
            t, num_records_bytes=num_records_bytes
        )

    def load(self, offset, vec_size=1):
        return buffer_ops.buffer_load(
            self.rsrc, offset, vec_width=vec_size, dtype=self.dtype
        )

    def store(self, offset, value, vec_size=1):
        buffer_ops.buffer_store(
            value, self.rsrc, offset, cache_modifier=self.cache_modifier
        )


def _gld(g, off):
    """Scalar global load via a function call (see :func:`_sst`)."""
    return g.load(off, vec_size=1)


def _gld_vec(g, off, n):
    """Vector global load (``n`` contiguous elements) via a function call."""
    return g.load(off, vec_size=n)


def _sst_vec_aligned(t, i, j, val, alignment):
    """Vector LDS store with an explicit, truthful byte alignment."""
    off = t.linear_offset((_I(i), _I(j)))
    vector.store(val, t.memptr, [off], alignment=alignment)


def _gst(g, off, val):
    """Scalar global store via a function call (see :func:`_sst`)."""
    g.store(off, val)


def _sst(t, idx, val):
    """1-D STensor store via a function call.

    Routing LDS writes through a call (instead of ``t[i] = v`` in the kernel
    body) keeps the FlyDSL AST rewriter from treating the shared-memory tensor
    as a state variable that must be threaded through the enclosing dynamic
    ``scf.if`` (which fails for Python objects).
    """
    t[_I(idx)] = val


def _sst2(t, i, j, val):
    """2-D STensor store via a function call (see :func:`_sst`)."""
    t[_I(i), _I(j)] = val


def _ld(t, idx):
    """1-D STensor scalar read (index-cast)."""
    return t[_I(idx)]


def _identity_frag(lane):
    """v4f32 identity accumulator fragment for one 16x16 diagonal block."""
    n = lane % fx.Int32(16)
    mb4 = (lane // fx.Int32(16)) * fx.Int32(4)
    elems = [
        ((mb4 + fx.Int32(p)) == n).select(fx.Float32(1.0), fx.Float32(0.0))
        for p in range(4)
    ]
    return vector.from_elements(Tir.vec(4, Tir.f32), elems)


def _wy_prefetch(
    src_g, base_in, in_row_stride, off, tid, stage_iters, svec_per_row, svec
):
    """Prefetch one WY sub-iteration's RHS into registers (``stage_iters`` vec-``svec``
    global loads; **no LDS touched**).

    Returns a Python list of loaded bf16 vectors. Because it writes no shared
    memory, the caller issues it one sub-iteration ahead of the matching
    :func:`_wy_scatter`, so the global-load latency overlaps the previous
    sub-iter's MFMA (which barely stalls) -- hiding ``wait_vm`` at iso-LDS and
    iso-VGPR, i.e. without giving up any occupancy.

    Rows past ``seqlen`` need no guard: ``src_g`` is a :class:`_SeqBoundedG`
    handle, so the buffer hardware zero-fills them."""
    regs = []
    for it in range(stage_iters):
        p = tid + fx.Int32(it * BLOCK_THREADS)
        j = p // fx.Int32(svec_per_row)
        col = (p % fx.Int32(svec_per_row)) * fx.Int32(svec)
        regs.append(_gld_vec(src_g, base_in + j * in_row_stride + off + col, svec))
    return regs


def _wy_scatter(regs, s_vT, s_beta, gc, is_k, tid, stage_iters, svec_per_row, svec):
    """Scale prefetched RHS regs by ``beta`` (v) or the cached ``beta*exp(g)``
    (k, read from the dead ``g_cumsum`` slot) and scalar-scatter (transposed)
    into ``s_vT`` -- the LDS RHS the WY MFMA reads."""
    for it in range(stage_iters):
        vals = regs[it]
        p = tid + fx.Int32(it * BLOCK_THREADS)
        j = p // fx.Int32(svec_per_row)
        row0 = (p % fx.Int32(svec_per_row)) * fx.Int32(svec)
        scale_j = Numeric.from_python_value(_ld(gc if is_k else s_beta, j))
        for vv in range(svec):
            val = Numeric.from_python_value(_ext(vals, vv)).to(fx.Float32)
            _sst2_vt(s_vT, row0 + fx.Int32(vv), j, (val * scale_j).to(fx.BFloat16))


def _wy_epilogue_to_lds(wy, s_out, warp16, mb4, n16):
    """Write the 4x4 WY MFMA accumulator (accumulator lane map: row = warp16 +
    (lane//16)*4 + p, col = en*16 + lane%16) into an LDS tile ``s_out`` so it can
    be read back coalesced for wide global stores.

    Keeping the row and column plain is what lets all 64 stores share one address
    register plus immediate offsets; see the note above s_out's layout for why the
    bank-conflict-free alternatives were measured and rejected.
    """
    for en in range(4):
        for p in range(4):
            s = warp16 + mb4 + fx.Int32(p)
            col = fx.Int32(en * 16) + n16
            val = Numeric.from_python_value(_ext(wy[en], p)).to(fx.BFloat16)
            _sst2(s_out, s, col, val)


def _sld_vec(s_t, i, j, n):
    """LDS vector load (``n`` contiguous elems along the inner dim) via a call
    (mirrors :func:`_gld_vec`; the readback of the WY output tile uses this)."""
    return s_t.vec_load((_I(i), _I(j)), n)


def _k_prefetch(k_g, base_k, k_row_stride, tid, n_per, k_v, kvec):
    """Issue the Phase-1b k-tile global loads into registers (``n_per`` vec-``kvec``
    buffer_loads; **no LDS touched**).  Called after the g/beta loads are issued, so
    these sit behind them in the vmem queue: vmem retires in order, so waiting on
    g/beta for the prefix-sum leaves these still in flight, and their latency is
    hidden by that scan and the k-scatter -- no extra LDS, no exposed ``wait_vm``
    like the old serial k stage had.

    Rows past ``seqlen`` are zero-filled by the buffer hardware (see
    :class:`_SeqBoundedG`), so no per-load guard is needed."""
    regs = []
    for it in range(n_per):
        p = tid + fx.Int32(it * BLOCK_THREADS)
        j = p // fx.Int32(k_v)
        cv = (p % fx.Int32(k_v)) * fx.Int32(kvec)
        regs.append(_gld_vec(k_g, base_k + j * k_row_stride + cv, kvec))
    return regs


def _k_scatter(regs, s_k, tid, n_per, k_v, kvec, alignment=16):
    """Scatter the prefetched k regs into ``s_k`` [BT,K] (vec-``kvec`` LDS store);
    the k loads have drained during the prefix-sum, so this barely stalls."""
    for it in range(n_per):
        p = tid + fx.Int32(it * BLOCK_THREADS)
        j = p // fx.Int32(k_v)
        cv = (p % fx.Int32(k_v)) * fx.Int32(kvec)
        _sst_vec_aligned(s_k, j, cv, regs[it], alignment)


def _wave_inclusive_scan(val, tid, width, zero):
    """In-wave inclusive prefix sum over ``width`` lanes via register shuffles.
    Uses the **same** stride-doubling add order as the old LDS Hillis-Steele
    (lane i at step s adds lane i-2^s, masked when i<2^s), so the result is
    bit-identical -- but with 0 barriers instead of log2(width) LDS round-trips.
    Requires width == wavefront size (here BT == WARP_SIZE == 64).

    (Installed flydsl exposes only ``shuffle_xor``; build the ``mode='up'``
    gpu.ShuffleOp directly -- lane k reads lane k-offset.)"""
    from flydsl._mlir.dialects.gpu import ShuffleOp

    csum = val
    s = 1
    while s < width:
        prev = type(val)(
            ShuffleOp(
                csum.ir_value(),
                fx.Int32(s).ir_value(),
                fx.Int32(width).ir_value(),
                mode="up",
            ).shuffleResult
        )
        csum = csum + (tid >= fx.Int32(s)).select(prev, zero)
        s <<= 1
    return csum


@functools.lru_cache(maxsize=64)
def compile_gdn_prepare(
    *,
    BT: int = 64,
    K: int = 128,
    V: int = 128,
    is_varlen: bool = False,
    g_scale: float = 1.0,
):
    """Chunk prepare: KKT, triangular inverse, and WY GEMMs via MFMA bf16 16x16x16.

    ``g_scale`` scales ``g_cumsum`` on the way out only (the in-chunk decay keeps
    using the natural-log value it just scanned). Pass ``log2(e)`` to publish
    ``g_cumsum`` in log2 space for downstream ``exp2`` consumers -- the same
    ``G_SCALE`` trick the Triton K1 kernel uses -- or 1.0 to leave it in the
    natural-log domain, in which case no multiply is emitted at all.

    LDS layout, 17,408 B total (rocprof rounds the static object up to the 256-B
    granule):
        s_g     fp32[BT]        in-wave scan result (g_cumsum, later the K scale)
        s_beta  fp32[BT]
        region P0 (off_k), reused in order:
          s_k     bf16[BT,KS]   staged k (KKT input + w_bar RHS source)  KS=K+4
          s_A     fp32[BT,ASA]  KKT result, then in-place (I+A)^-1       ASA=BT+1
          s_vT    bf16[BT,VTS]  transposed scaled WY RHS                 VTS=BT+4
          s_out   bf16[BT,OUT_S] WY output staging, parked in P0's tail
                                 immediately after s_vT

    ``s_vT`` (8,704 B) + ``s_out`` (8,192 B) exactly equal the dead full-K
    ``s_k`` slab (16,896 B), so the coalescing output staging costs no LDS and
    no occupancy.  Each wave reads back the same 16 output rows it wrote, which
    removes the cross-wave publish barrier while preserving the pre-write fence.

    Opus reports 18,176 B for the same phases because its phase-2 budget also
    reserves one fp32 16x16 tile after ``s_A``; this port never addresses that
    tile, so P0 is 768 B smaller.  Occupancy is unaffected either way -- it is
    capped by VGPRs (64 -> 8 waves/SIMD) and the 32-wave/CU limit, not by LDS.
    """
    arch = get_rocm_arch()
    assert BT == 64 and K == 128 and V == 128, "gdn_prepare targets the BT=64 main path"
    LOG2E = 1.4426950408889634

    KS = K + 4  # Opus's k stride.  Odd rows are only 8B-aligned, so the
    # vec8 LDS staging below declares a truthful alignment=8.
    VTS = BT + 4  # s_vT row stride
    ASA = BT + 1  # Opus's fp32 A stride
    BK_SUB = 64
    OUT_S = BK_SUB  # WY output-staging row stride (128B rows -> 16B-aligned vec8)
    N_K_ITERS = K // BK_SUB  # 2
    N_V_ITERS = V // BK_SUB  # 2
    # s_vT and s_out must both fit in P0 alongside the larger of s_k / s_A.
    assert BT * VTS * 2 + BT * OUT_S * 2 <= max(BT * KS * 2, BT * ASA * 4)

    allocator = SmemAllocator(None, arch=arch, global_sym_name="gdn_prepare_smem")

    def _alloc(nbytes, align=16):
        off = allocator._align(allocator.ptr, align)
        allocator.ptr = off + nbytes
        return off

    off_g = _alloc(BT * 4)
    off_beta = _alloc(BT * 4)
    off_k = _alloc(
        max(
            BT * KS * 2,
            BT * ASA * 4,
            BT * VTS * 2,
            BT * OUT_S * 2,
        )
    )

    @flyc.kernel(known_block_size=[BLOCK_THREADS, 1, 1], name="gdn_prepare_kernel")
    def gdn_prepare_kernel(
        k_t: fx.Tensor,
        v_t: fx.Tensor,
        g_t: fx.Tensor,
        beta_t: fx.Tensor,
        cu_t: fx.Tensor,
        wbar_t: fx.Tensor,
        ubar_t: fx.Tensor,
        gcs_t: fx.Tensor,
        T: fx.Int32,
        H: fx.Int32,
        Hg: fx.Int32,
    ):
        tid = fx.thread_idx.x
        rep = H // Hg

        lane = tid % fx.Int32(WARP_SIZE)
        warp = tid // fx.Int32(WARP_SIZE)
        warp16 = warp * fx.Int32(16)

        # --- workgroup -> (sequence, chunk) mapping + sequence bounds ---
        i_t = fx.block_idx.x
        i_bh = fx.block_idx.y
        i_b = i_bh // H
        i_h = i_bh % H
        if const_expr(is_varlen):
            cu_g = GTensor(cu_t, dtype=Tir.i32, shape=(-1,))
            bos = Numeric.from_python_value(_gld(cu_g, i_b))
            nxt = Numeric.from_python_value(_gld(cu_g, i_b + fx.Int32(1)))
            seqlen = nxt - bos
        else:
            bos = i_b * T
            seqlen = T

        i_hg = i_h // rep
        chunk_start = i_t * fx.Int32(BT)
        n_chunks = (seqlen + fx.Int32(BT - 1)) // fx.Int32(BT)
        active = i_t < n_chunks

        if active:
            # Build all global/shared tensor handles *inside* the dynamic-if body
            # so the AST rewriter does not try to thread them as scf.if state
            # (method calls on pre-if locals get captured as state variables).
            # Clamp every descriptor to this sequence's last row, so the buffer
            # hardware -- not two selects per access -- enforces the `row <
            # seqlen` bound for the ragged final chunk.  Byte limits, one whole
            # row stride each.  The token-major inputs ([.,H] f32 g/beta,
            # [.,Hg,K] and [.,H,V] bf16 k/v) bound on the sequence's last token;
            # the head-major outputs bound on this head's slab end (see
            # ``hm_end`` below) -- a workgroup owns exactly one (sequence, head),
            # so a per-workgroup limit is enough to stop a ragged tail row from
            # spilling into the next head.
            seq_end = bos + seqlen
            lim_gb = (seq_end * H * fx.Int32(4)).ir_value()
            lim_k = (seq_end * Hg * fx.Int32(K * 2)).ir_value()
            lim_v = (seq_end * H * fx.Int32(V * 2)).ir_value()
            # Head-major output slab: [B, H, T, *] with stride 1 along T inside a
            # head, matching what the Triton K1/K2 pair hands to K5/K6.  A
            # workgroup writes only rows [chunk_start, chunk_start + BT) of the
            # (i_b, i_h) slab.
            if const_expr(is_varlen):
                hm_row = i_h * T + bos  # B == 1, T == T_flat
                hm_end = i_h * T + seq_end
            else:
                hm_row = (i_b * H + i_h) * T
                hm_end = hm_row + seqlen
            hm_row = hm_row + chunk_start
            lim_gcs = (hm_end * fx.Int32(4)).ir_value()
            lim_ub = (hm_end * fx.Int32(V * 2)).ir_value()
            lim_wb = (hm_end * fx.Int32(K * 2)).ir_value()
            k_g = _SeqBoundedG(k_t, Tir.bf16, lim_k)
            v_g = _SeqBoundedG(v_t, Tir.bf16, lim_v)
            g_g = _SeqBoundedG(g_t, Tir.f32, lim_gb)
            beta_g = _SeqBoundedG(beta_t, Tir.f32, lim_gb)
            gcs_g = _SeqBoundedG(gcs_t, Tir.f32, lim_gcs)
            wbar_g = _SeqBoundedG(wbar_t, Tir.bf16, lim_wb)
            ubar_g = _SeqBoundedG(ubar_t, Tir.bf16, lim_ub)

            base = allocator.get_base()
            s_g = STensor(
                SmemPtr(base, off_g, Tir.f32, shape=(BT,)), Tir.f32, shape=(BT,)
            )
            s_beta = STensor(
                SmemPtr(base, off_beta, Tir.f32, shape=(BT,)), Tir.f32, shape=(BT,)
            )
            s_k = STensor(
                SmemPtr(base, off_k, Tir.bf16, shape=(BT * KS,)),
                Tir.bf16,
                shape=(BT, KS),
            )
            s_A = STensor(
                SmemPtr(base, off_k, Tir.f32, shape=(BT * ASA,)),
                Tir.f32,
                shape=(BT, ASA),
            )
            s_vT = STensor(
                SmemPtr(base, off_k, Tir.bf16, shape=(BT * VTS,)),
                Tir.bf16,
                shape=(BT, VTS),
            )
            # WY output staging in P0's free tail after s_vT, read back coalesced
            # (vec8) for wide global stores.
            s_out = STensor(
                SmemPtr(base, off_k + BT * VTS * 2, Tir.bf16, shape=(BT * OUT_S,)),
                Tir.bf16,
                shape=(BT, OUT_S),
            )

            base_gb = (bos + chunk_start) * H + i_h  # [.,H]      (g/beta in)
            base_k = ((bos + chunk_start) * Hg + i_hg) * fx.Int32(K)  # [.,Hg,K]
            base_v = ((bos + chunk_start) * H + i_h) * fx.Int32(V)  # [.,H,V]
            base_gcs = hm_row  # [.,H,T]      (g_cumsum out)
            base_ub = hm_row * fx.Int32(V)  # [.,H,T,V]    (u_bar out)
            base_wb = hm_row * fx.Int32(K)  # [.,H,T,K]    (w_bar out)
            zero_f = fx.Float32(0.0)
            is_lane = tid < fx.Int32(BT)

            # ---- Phase 1a: load g + beta, in-wave inclusive scan of g ----
            # BT == WARP_SIZE, so the prefix-sum is a single-wave (wave 0) scan:
            # run it entirely in registers via shuffle -> 0 barriers (was 6 LDS
            # round-trips).  Same add order as before => bit-identical g_cumsum.
            if is_lane:
                # Rows past seqlen read as zero and their g_cumsum store is
                # dropped, both in hardware (bounded descriptors above).
                gi = base_gb + tid * H
                gv = Numeric.from_python_value(_gld(g_g, gi))
                bv = Numeric.from_python_value(_gld(beta_g, gi))
                csum = _wave_inclusive_scan(gv, tid, BT, zero_f)
                _sst(s_g, tid, csum)  # g_cumsum -> LDS (random-access by KKT)
                _sst(s_beta, tid, bv)
                # Published head-major, so the 64 lanes of this store land in one
                # contiguous 256-B run instead of striding H rows apart.
                out = csum if g_scale == 1.0 else csum * fx.Float32(g_scale)
                _gst(gcs_g, base_gcs + tid, out)

            # Prefetch Phase-1b k-tile into registers (queued behind g/beta, so waiting
            # on g/beta does not drain these); its global-load latency overlaps the
            # prefix-sum and the k-scatter below. iso-LDS.
            KVEC = 8
            k_row = Hg * fx.Int32(K)
            kkt = [_F32X4() for _ in range(4)]
            K_V = K // KVEC  # 16
            KKv = BT * K_V  # 1024 (== n_per*BLOCK_THREADS, no tail)
            n_per = (KKv + BLOCK_THREADS - 1) // BLOCK_THREADS  # 4
            assert n_per * BLOCK_THREADS == KKv
            kregs = _k_prefetch(k_g, base_k, k_row, tid, n_per, K_V, KVEC)
            gc = s_g  # final g_cumsum in LDS
            # ---- Phase 1b: scatter the prefetched k regs -> s_k [BT,K] (vec8) ----
            # No barrier between the scan and here: s_g/s_beta live in their own LDS
            # region (off_g/off_beta), disjoint from P0 (off_k) that s_k occupies, and
            # nothing before the next barrier reads them -- Phase 1c is their first
            # reader, so the s_k barrier below publishes g_cumsum too.  Dropping the
            # extra barrier lets waves 1-3 scatter k *during* wave 0's g/beta load
            # latency + prefix-sum instead of idling through it (ATT: that barrier
            # alone was 14.2% of all kernel stall cycles, per-wave [8, ~7k, ~7k, ~7k]).
            _k_scatter(kregs, s_k, tid, n_per, K_V, KVEC, alignment=8)
            gpu.barrier()  # publish s_k + g_cumsum/beta
            # ---- Phase 1c: KKT via MFMA (1x4x8), gate-scaled strict-lower -> s_A ----
            for ek in range_constexpr(K // 16):
                a = _load_mfma_tile(s_k, warp16, fx.Int32(ek * 16), lane)
                bs = [
                    _load_mfma_tile(s_k, fx.Int32(en * 16), fx.Int32(ek * 16), lane)
                    for en in range(4)
                ]
                kkt = [_mfma16(a, bs[en], kkt[en]) for en in range(4)]
            # s_A aliases s_k's LDS region: ensure every warp has finished
            # reading s_k (B-operand spans all 64 rows) before any post-scale
            # store overwrites the region as s_A.
            gpu.barrier()
            n16 = lane % fx.Int32(16)
            mb4 = (lane // fx.Int32(16)) * fx.Int32(4)
            for en in range_constexpr(4):
                for p in range_constexpr(4):
                    s = warp16 + mb4 + fx.Int32(p)
                    r = fx.Int32(en * 16) + n16
                    cval = Numeric.from_python_value(_ext(kkt[en], p))
                    beta_s = Numeric.from_python_value(_ld(s_beta, s))
                    gc_s = Numeric.from_python_value(_ld(gc, s))
                    gc_r = Numeric.from_python_value(_ld(gc, r))
                    decay = _exp2_f32((gc_s - gc_r) * fx.Float32(LOG2E))
                    aval = (s > r).select(cval * beta_s * decay, zero_f)
                    _sst2(s_A, s, r, aval)
            gpu.barrier()

            # g_cumsum is dead after A is materialized. Cache the K-side
            # beta*exp(g) scale in-place; later inverse/Schur barriers publish it
            # before the first K WY scatter.
            if is_lane:
                gc_j = Numeric.from_python_value(_ld(gc, tid))
                beta_j = Numeric.from_python_value(_ld(s_beta, tid))
                _sst(gc, tid, beta_j * _exp2_f32(gc_j * fx.Float32(LOG2E)))

            # ---- Phase 2: C = (I + A)^{-1} via MFMA ----
            # Phase 2a: invert each 16x16 strictly-lower diagonal block via the
            # nilpotent Neumann series (I+A)^-1 = sum (-A)^n, B := -A (B^16 = 0),
            # in the Opus-style 8-MFMA *squaring* form.  Factor the 16-term series
            # as C = (I+B)(I+B^2)(I+B^4)(I+B^8) (every n in 0..15 has a unique
            # 4-bit binary expansion) -> 8 MFMAs (6 products + 2 transposes)
            # instead of Horner's 15 on the per-warp critical path.  An
            # accumulator reused as the MFMA A-operand is implicitly TRANSPOSED,
            # so we carry each power AND its transpose (b2/b2t, b4/b4t);
            # loadA(B)=neg_A with loadB(B)=neg_A_T squares B with no LDS
            # round-trip.  Each warp owns one diagonal block.
            br = warp16
            neg_A = _load_neg_fp32_tile(s_A, br, br, lane)  # loadA(B)
            I_acc = _identity_frag(lane)
            z4 = _F32X4()  # zero v4f32 MFMA accumulator seed (phases 2a+2b)
            neg_A_T = _load_neg_fp32_tile_T(s_A, br, br, lane)  # loadB(B)
            b2 = _mfma16(neg_A, neg_A_T, z4)  # B^2 = B.B
            b2t = _mfma16(neg_A_T, neg_A, z4)  # (B^2)^T
            b2_o = _accum_to_bf16x4(b2)
            b2t_o = _accum_to_bf16x4(b2t)
            b4 = _mfma16(b2t_o, b2_o, z4)  # B^4 = B^2.B^2
            b4t = _mfma16(b2_o, b2t_o, z4)  # (B^4)^T
            b4_o = _accum_to_bf16x4(b4)
            b4t_o = _accum_to_bf16x4(b4t)
            C_acc = _mfma16(b4t_o, b4_o, I_acc)  # I + B^8
            C_acc = _mfma16(b4t_o, _accum_to_bf16x4(C_acc), C_acc)  # (I+B^4).
            C_acc = _mfma16(b2t_o, _accum_to_bf16x4(C_acc), C_acc)  # (I+B^2).
            C_acc = _mfma16(neg_A, _accum_to_bf16x4(C_acc), C_acc)  # (I+B).
            _store_fp32_tile(s_A, br, br, C_acc, lane)
            for it in range_constexpr((16 * 16 + WARP_SIZE - 1) // WARP_SIZE):
                idx = lane + fx.Int32(it * WARP_SIZE)
                rr = idx // fx.Int32(16)
                cc = idx % fx.Int32(16)
                if rr < cc:
                    _sst2(s_A, br + rr, br + cc, zero_f)
            gpu.barrier()

            # Phase 2b: Opus's in-place, level-ordered Schur DAG.  Only the L
            # blocks that a sibling warp is about to overwrite are kept in VGPRs.
            sav_L32 = _BF16X4()
            sav_L43 = _BF16X4()
            sav_L42 = _BF16X4()
            if warp == fx.Int32(0):
                sav_L32 = _load_fp32_tile(s_A, 32, 16, lane)
                # L42 is only overwritten (by wave 1) in the *second* Schur
                # level, and no wave writes [48,16] in the first, so it can be
                # captured here with the other preloads instead of behind an
                # extra barrier between the two levels.
                sav_L42 = _load_fp32_tile(s_A, 48, 16, lane)
            if warp < fx.Int32(2):
                sav_L43 = _load_fp32_tile(s_A, 48, 32, lane)
            # Sibling waves overwrite L32/L42/L43 below. Publish completion of
            # the register preloads before any wave can start those stores.
            gpu.barrier()

            kept_c21 = z4
            kept_c32 = z4
            kept_c31 = z4
            if warp == fx.Int32(0):
                t = _mfma16(
                    _load_fp32_tile(s_A, 16, 0, lane),
                    _load_fp32_tile_T(s_A, 0, 0, lane),
                    z4,
                )
                kept_c21 = _negate4(
                    _mfma16(
                        _load_fp32_tile(s_A, 16, 16, lane),
                        _accum_to_bf16x4(t),
                        z4,
                    )
                )
                _store_fp32_tile(s_A, 16, 0, kept_c21, lane)
            if warp == fx.Int32(1):
                t = _mfma16(
                    _load_fp32_tile(s_A, 32, 16, lane),
                    _load_fp32_tile_T(s_A, 16, 16, lane),
                    z4,
                )
                kept_c32 = _negate4(
                    _mfma16(
                        _load_fp32_tile(s_A, 32, 32, lane),
                        _accum_to_bf16x4(t),
                        z4,
                    )
                )
                _store_fp32_tile(s_A, 32, 16, kept_c32, lane)
            if warp == fx.Int32(2):
                t = _mfma16(
                    _load_fp32_tile(s_A, 48, 32, lane),
                    _load_fp32_tile_T(s_A, 32, 32, lane),
                    z4,
                )
                c43 = _negate4(
                    _mfma16(
                        _load_fp32_tile(s_A, 48, 48, lane),
                        _accum_to_bf16x4(t),
                        z4,
                    )
                )
                _store_fp32_tile(s_A, 48, 32, c43, lane)
            gpu.barrier()

            if warp == fx.Int32(0):
                t = _mfma16(
                    _load_fp32_tile(s_A, 32, 0, lane),
                    _load_fp32_tile_T(s_A, 0, 0, lane),
                    z4,
                )
                t = _mfma16(sav_L32, _accum_to_bf16x4(kept_c21), t)
                kept_c31 = _negate4(
                    _mfma16(
                        _load_fp32_tile(s_A, 32, 32, lane),
                        _accum_to_bf16x4(t),
                        z4,
                    )
                )
                _store_fp32_tile(s_A, 32, 0, kept_c31, lane)
            if warp == fx.Int32(1):
                t = _mfma16(
                    _load_fp32_tile(s_A, 48, 16, lane),
                    _load_fp32_tile_T(s_A, 16, 16, lane),
                    z4,
                )
                t = _mfma16(sav_L43, _accum_to_bf16x4(kept_c32), t)
                c42 = _negate4(
                    _mfma16(
                        _load_fp32_tile(s_A, 48, 48, lane),
                        _accum_to_bf16x4(t),
                        z4,
                    )
                )
                _store_fp32_tile(s_A, 48, 16, c42, lane)
            gpu.barrier()

            if warp == fx.Int32(0):
                t = _mfma16(
                    _load_fp32_tile(s_A, 48, 0, lane),
                    _load_fp32_tile_T(s_A, 0, 0, lane),
                    z4,
                )
                t = _mfma16(sav_L42, _accum_to_bf16x4(kept_c21), t)
                t = _mfma16(sav_L43, _accum_to_bf16x4(kept_c31), t)
                c41 = _negate4(
                    _mfma16(
                        _load_fp32_tile(s_A, 48, 48, lane),
                        _accum_to_bf16x4(t),
                        z4,
                    )
                )
                _store_fp32_tile(s_A, 48, 0, c41, lane)
            gpu.barrier()

            # ---- Phase 2c setup: WY RHS software-pipeline (register prefetch) ----
            # u_bar = C @ (v*beta);  w_bar = C @ (k*beta*exp(g)).  RHS is transposed
            # into s_vT (scalar scatter) with bf16x8 HBM reads.  Each sub-iter's
            # global loads are issued into registers one step ahead
            # (_wy_prefetch) so their latency overlaps the previous sub-iter's MFMA.
            # s_vT stays a single LDS buffer -> occupancy unchanged.
            SVEC = 8
            SVEC_PER_ROW = BK_SUB // SVEC  # 8 vec-groups per WY RHS row
            STAGE_ITERS = (BT * SVEC_PER_ROW + BLOCK_THREADS - 1) // BLOCK_THREADS  # 2
            assert (
                STAGE_ITERS * BLOCK_THREADS == BT * SVEC_PER_ROW
            )  # 1:1 lane map, no tail guard
            v_row_stride = H * fx.Int32(V)
            k_row_stride = Hg * fx.Int32(K)
            # Head-major outputs: consecutive rows are one row apart, not H.
            ub_row_stride = fx.Int32(V)
            wb_row_stride = fx.Int32(K)
            # Coalesced WY output store: s_out[BT,BK_SUB] -> global as vec8 bf16.
            GVEC = 8
            VPR = BK_SUB // GVEC  # 8 vec-groups per output row
            NVEC = BT * BK_SUB // GVEC  # 512 vec8 stores per sub-iter tile
            NIT_OUT = (NVEC + BLOCK_THREADS - 1) // BLOCK_THREADS  # 2
            assert NIT_OUT * BLOCK_THREADS == NVEC  # 1:1 lane map, no tail guard
            # Prefetch v-tile 0 while the final C representation is prepared.
            reg = _wy_prefetch(
                v_g,
                base_v,
                v_row_stride,
                fx.Int32(0),
                tid,
                STAGE_ITERS,
                SVEC_PER_ROW,
                SVEC,
            )

            # Each warp caches its 16x64 row of C.  s_A can then be reused as
            # s_vT/s_out without a second LDS region.
            cached_C = [
                _load_fp32_tile(s_A, warp16, fx.Int32(ek * 16), lane) for ek in range(4)
            ]
            # FlyDSL may schedule another wave's first s_vT stores before all
            # four C loads retire; make the read->alias lifetime explicit.
            gpu.barrier()

            # ---- Phase 2c: WY GEMMs via MFMA (1x4x4), register-prefetch pipeline ----
            # u_bar = C @ (v*beta)
            for v_it in range_constexpr(N_V_ITERS):
                voff = fx.Int32(v_it * BK_SUB)
                # Read-completion fence for the *previous* sub-iter's s_vT reads,
                # placed here rather than right after the MFMA loop: the epilogue
                # only touches s_out (disjoint from s_vT) and only this wave's own
                # 16 rows of it, so a wave can run its whole epilogue without
                # waiting for the slowest wave's MFMA.  That turns
                # max(MFMA) + epilogue into max(MFMA_i + epilogue_i).  v_it == 0
                # needs nothing: the cached_C fence above already separates it.
                if const_expr(v_it > 0):
                    gpu.barrier()
                _wy_scatter(
                    reg, s_vT, s_beta, gc, False, tid, STAGE_ITERS, SVEC_PER_ROW, SVEC
                )
                # prefetch next sub-iter (next v-tile, or k-tile 0 across the boundary)
                if v_it + 1 < N_V_ITERS:
                    reg = _wy_prefetch(
                        v_g,
                        base_v,
                        v_row_stride,
                        fx.Int32((v_it + 1) * BK_SUB),
                        tid,
                        STAGE_ITERS,
                        SVEC_PER_ROW,
                        SVEC,
                    )
                else:
                    reg = _wy_prefetch(
                        k_g,
                        base_k,
                        k_row_stride,
                        fx.Int32(0),
                        tid,
                        STAGE_ITERS,
                        SVEC_PER_ROW,
                        SVEC,
                    )
                gpu.barrier()
                wy = [_F32X4() for _ in range(4)]
                for ek in range_constexpr(BT // 16):
                    a = cached_C[ek]
                    bs = [
                        _load_mfma_tile_vt_tiled(s_vT, en, ek, n16, mb4)
                        for en in range(4)
                    ]
                    wy = [_mfma16(a, bs[en], wy[en]) for en in range(4)]
                # Epilogue: accumulator -> s_out (LDS) -> coalesced vec8 global store.
                # Unfenced on purpose (see the fence at the top of the loop).
                _wy_epilogue_to_lds(wy, s_out, warp16, mb4, n16)
                rocdl.s_waitcnt(0)
                for it in range_constexpr(NIT_OUT):
                    q = warp * fx.Int32(16 * VPR) + lane + fx.Int32(it * WARP_SIZE)
                    row = q // fx.Int32(VPR)
                    vc = (q % fx.Int32(VPR)) * fx.Int32(GVEC)
                    vals = _sld_vec(s_out, row, vc, GVEC)
                    _gst(ubar_g, base_ub + row * ub_row_stride + voff + vc, vals)

            # w_bar = C @ (k*beta*exp(g))   (reg already holds k-tile 0)
            for k_it in range_constexpr(N_K_ITERS):
                koff = fx.Int32(k_it * BK_SUB)
                # Always needed here: the previous sub-iter is the last v tile
                # (k_it == 0) or the previous k tile, and both read s_vT.
                gpu.barrier()
                _wy_scatter(
                    reg, s_vT, s_beta, gc, True, tid, STAGE_ITERS, SVEC_PER_ROW, SVEC
                )
                if k_it + 1 < N_K_ITERS:
                    reg = _wy_prefetch(
                        k_g,
                        base_k,
                        k_row_stride,
                        fx.Int32((k_it + 1) * BK_SUB),
                        tid,
                        STAGE_ITERS,
                        SVEC_PER_ROW,
                        SVEC,
                    )
                gpu.barrier()
                wy = [_F32X4() for _ in range(4)]
                for ek in range_constexpr(BT // 16):
                    a = cached_C[ek]
                    bs = [
                        _load_mfma_tile_vt_tiled(s_vT, en, ek, n16, mb4)
                        for en in range(4)
                    ]
                    wy = [_mfma16(a, bs[en], wy[en]) for en in range(4)]
                # Unfenced, as in the v loop.  The final K tile needs no trailing
                # fence at all: nothing overwrites s_vT afterwards.
                _wy_epilogue_to_lds(wy, s_out, warp16, mb4, n16)
                rocdl.s_waitcnt(0)
                for it in range_constexpr(NIT_OUT):
                    q = warp * fx.Int32(16 * VPR) + lane + fx.Int32(it * WARP_SIZE)
                    row = q // fx.Int32(VPR)
                    vc = (q % fx.Int32(VPR)) * fx.Int32(GVEC)
                    vals = _sld_vec(s_out, row, vc, GVEC)
                    _gst(wbar_g, base_wb + row * wb_row_stride + koff + vc, vals)

    @flyc.jit
    def launch_gdn_prepare(
        k_t: fx.Tensor,
        v_t: fx.Tensor,
        g_t: fx.Tensor,
        beta_t: fx.Tensor,
        cu_t: fx.Tensor,
        wbar_t: fx.Tensor,
        ubar_t: fx.Tensor,
        gcs_t: fx.Tensor,
        T: fx.Int32,
        H: fx.Int32,
        Hg: fx.Int32,
        grid_x: fx.Int32,
        grid_y: fx.Int32,
        stream: fx.Stream,
    ):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()
        gdn_prepare_kernel(
            k_t, v_t, g_t, beta_t, cu_t, wbar_t, ubar_t, gcs_t, T, H, Hg
        ).launch(
            grid=(grid_x, grid_y, 1),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    return launch_gdn_prepare
