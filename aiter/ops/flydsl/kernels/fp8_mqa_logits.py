# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FP8 MQA logits (DeepSeek lightning indexer) -- FlyDSL gfx942 kernel.

Compute for each query row ``m`` and KV position ``n``
inside that row's window ``[cu_starts[m], cu_ends[m])``::

    logits[m, n] = sum_h ReLU(<Q[m, h, :], K[n, :]> * kv_scale[n]) * weights[m, h]

The public ``flydsl_fp8_mqa_logits`` mirrors the Triton launcher
``aiter.ops.triton.attention.fp8_mqa_logits.fp8_mqa_logits`` exactly (same
arguments, same return tensor, same ``clean_logits`` semantics) so the two are
drop-in interchangeable in tests and benchmarks.
"""

# No `from __future__ import annotations`: FlyDSL arg typing needs real
# annotation objects, not PEP 563 strings.

import math
import os
import re
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl._mlir import ir
from flydsl._mlir.dialects import memref as memref_dialect
from flydsl._mlir.dialects import scf
from flydsl._mlir.dialects.rocdl import (
    mfma_scale_f32_32x32x64_f8f6f4 as _ods_mfma_scale32x32x64,
)
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.numeric import ArithValue
from flydsl.expr.rocdl import _unwrap_mfma_operand
from flydsl.expr.typing import T
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr

from aiter.jit.utils.chip_info import get_gfx

from . import buffer_ops
from .tensor_shim import GTensor, STensor, _run_compiled, _to_raw

Vec = fx.Vector


def _i32_add(a, b):
    """i32 add (result stays Int32, not index type)."""
    return fx.Int32(arith.addi(_to_raw(a), _to_raw(b)))


def _make_out_row_t(logits, stride_i64, row_i32):
    """1-D output GTensor for one row, with the row's byte offset folded into
    the base pointer in i64.

    A 2-D (row, col) view computes ``row * stride + col`` in i32 and overflows
    past 2^31 (~46k-square dense outputs), silently mis-writing.
    """
    _ri64 = arith.extui(T.i64, _to_raw(row_i32))
    _byte = arith.muli(arith.muli(_ri64, stride_i64), arith.constant(4, type=T.i64))
    _idx = arith.index_cast(T.index, _byte)
    return GTensor(logits, dtype=T.f32, shape=(-1,), static_bytes_offset_i64=_idx)


def _load_pack_i32x8(i32_view, byte_off_i32):
    """32-byte fragment as ``vector<8xi32>`` (frag_bytes=32 atoms).

    buffer_load tops out at dwordx4 (16 bytes), so the fragment is two
    consecutive dwordx4 loads concatenated with vector.shuffle.
    ``byte_off_i32`` must already include this lane's fragment offset so the
    load hits the correct 32-byte chunk for its lane group.
    """
    dword_off = fx.Int32(arith.divui(_to_raw(byte_off_i32), _to_raw(fx.Int32(4))))
    v4_lo = i32_view.vec_load((dword_off,), vec_size=4)
    v4_hi = i32_view.vec_load((_i32_add(dword_off, fx.Int32(4)),), vec_size=4)
    return Vec(v4_lo).shuffle(v4_hi, list(range(8))).ir_value()


def _emit_neg_inf_range(out_row_t, neg_inf, lo_i32, hi_i32, tid_i32, nthreads):
    """Thread-strided ``out_row_t[c] = -inf`` over columns ``[lo, hi)``.

    ``nthreads`` threads cooperate, with ``tid_i32`` in ``[0, nthreads)``; thread
    ``t`` writes ``lo+t, lo+t+nthreads, ...``. Consecutive lanes cover
    consecutive dwords, so one wave iteration coalesces into a single 256-byte
    store. Zero-trip on an empty range (``scf.for`` with ``lb >= ub``).

    Plain dwords on purpose: the fill is bandwidth-bound rather than store-issue-bound.
    So dwordx4 doesn't improve performance.
    """
    lo = _to_raw(fx.Index(_i32_add(lo_i32, tid_i32)))
    hi = _to_raw(fx.Index(hi_i32))
    step = _to_raw(fx.Index(nthreads))
    loop = scf.ForOp(lo, hi, step, [])
    with ir.InsertionPoint(loop.body):
        col = fx.Int32(arith.index_cast(T.i32, loop.induction_variable))
        out_row_t[col] = fx.Float32(neg_inf)
        scf.YieldOp([])


def _emit_row_neg_inf_fill(
    *,
    logits,  # the output kernel arg
    stride_i64,  # i64 row stride in elements (the builders' _stride_i64)
    rows,  # list[fx.Int32]: absolute query rows this thread group owns
    starts,  # list[fx.Int32]: max(cu_starts, 0),        parallel to rows
    ends,  # list[fx.Int32]: min(cu_ends, seq_len_kv), parallel to rows
    seq_len_kv,  # fx.Int32
    tid_i32,  # fx.Int32 in [0, nthreads): id within the cooperating group
    nthreads,  # int: 64 (one wave) or MR_BLOCK_THREADS (the whole block)
    by_i32,  # fx.Int32: block_idx.y
    num_splits,  # fx.Int32: grid.y (>= 1)
):
    """``clean_logits`` prefill, fused into the compute kernel.

    Only called when the ``clean_logits`` build flag is set -- it is a
    compile-time specialization, like ``convert_q_fn``/``convert_kv_fn``, so the
    ``clean_logits=False`` kernel contains none of this code at all.

    The epilogue writes column ``c`` of row ``rows[j]`` only for
    ``c in [starts[j], ends[j])``, so the complement inside ``[0, seq_len_kv)``
    is never written by anybody. This emits -inf over exactly that complement,
    which is the two contiguous ranges ``[0, s)`` and ``[e, seq_len_kv)`` with::

        s = min(starts[j], seq_len_kv)
        e = max(ends[j], s)

    ``e``'s max collapses an empty or inverted window (``cu_ends <= cu_starts``,
    or the negative ``cu_ends`` a causal mask yields when s_kv < s_q) to
    "fill the whole row", and keeps the two ranges from overlapping. ``s``'s min
    is load-bearing: ``starts`` is clamped only from below, and the per-row
    output descriptor is built with ``num_records`` = 4 GiB, so an unclamped
    ``cu_starts`` past the end would run off the row and corrupt the next ones
    with no hardware OOB net.

    Rows are already partitioned across grid.x (and across waves in the LDS
    builder), but ``num_splits`` blocks share every row. Since nobody computes
    the complement, it can be partitioned freely: block ``by`` takes its ``by``-th
    equal chunk of each range -- disjoint, gap-free, and balanced across grid.y.
    That is deliberately independent of the tile loop's
    ``tile_start``/``split_cols`` arithmetic and is emitted unconditionally, so a
    block whose ``by`` lands past the union window (zero tile iterations) still
    fills its share.

    MUST be emitted AFTER the tile loop. ``_build_kernel_mfma_lds_pipe`` waits on
    an exact ``s_waitcnt vmcnt(N)`` for its in-flight global->LDS DMAs, and on
    gfx9 vmcnt counts vector STORES too -- a fill store in flight inside the loop
    would inflate the count and let the kernel read a half-written LDS tile.
    Fill and compute addresses are disjoint, so no barrier or ordering is needed.
    """
    i32_0 = fx.Int32(0)
    slk = fx.Int32(seq_len_kv)
    neg_inf = arith.constant(float("-inf"), type=T.f32)

    def _split(lo_i32, hi_i32):
        """This block's chunk of ``[lo, hi)``: the ``by``-th of num_splits."""
        lo, hi = _to_raw(lo_i32), _to_raw(hi_i32)
        chunk = arith.ceildivui(arith.subi(hi, lo), _to_raw(num_splits))
        b_lo = arith.minsi(arith.addi(lo, arith.muli(_to_raw(by_i32), chunk)), hi)
        b_hi = arith.minsi(arith.addi(b_lo, chunk), hi)
        return fx.Int32(b_lo), fx.Int32(b_hi)

    for j in range_constexpr(len(rows)):
        out_row_t = _make_out_row_t(logits, stride_i64, rows[j])
        s = fx.Int32(arith.minsi(_to_raw(starts[j]), _to_raw(slk)))
        e = fx.Int32(arith.maxsi(_to_raw(ends[j]), _to_raw(s)))
        for lo_i32, hi_i32 in ((i32_0, s), (e, slk)):
            b_lo, b_hi = _split(lo_i32, hi_i32)
            _emit_neg_inf_range(out_row_t, neg_inf, b_lo, b_hi, tid_i32, nthreads)


# Default KV tile width (columns processed per inner-loop iteration).
_BLOCK_KV = 128

_DEFAULT_COMPILE_HINTS = {
    "waves_per_eu": 2,
    "fast_fp_math": True,
}

_ARCH = get_gfx()


@dataclass(frozen=True)
class _SplitPolicy:
    """Per-arch tuning for the ``grid.y`` KV-column split (``_auto_num_splits``).

    Fields
    ------
    min_seq_len_kv : int
        Never split below this ``seq_len_kv``. 0 means "no gate" -- let
        ``min_tiles_per_split`` do the limiting, which it does automatically
        once the window holds fewer than that many tiles.
    min_tiles_per_split : int
        Smallest chunk, in BKV tiles, a split may own. Below it the per-block
        fixed cost (Q/weight preload, plus the LDS builder's pipeline prologue)
        stops being amortized. Note this is denominated in *tiles*, so its
        column-equivalent scales with the variant's ``block_kv``.
    cu_oversub : int
        Target total blocks as a multiple of the device CU count.
    fallback_cu : int
        Nominal CU count to assume when the device query fails.
    """

    min_seq_len_kv: int
    min_tiles_per_split: int
    cu_oversub: int
    fallback_cu: int


_SPLIT_POLICIES = {
    # Tuned on MI300X (304 CU) against the direct-load builder at BKV=128,
    # where min_tiles_per_split=8 is 1024 KV columns.
    "gfx942": _SplitPolicy(
        min_seq_len_kv=4096, min_tiles_per_split=8, cu_oversub=4, fallback_cu=304
    ),
    # Tuned on MI355X (256 CU) against the LDS-pipelined builder.
    "gfx950": _SplitPolicy(
        min_seq_len_kv=0, min_tiles_per_split=2, cu_oversub=4, fallback_cu=256
    ),
}


def _split_policy() -> _SplitPolicy:
    """Split-policy constants for the current arch (gfx942's, conservatively,
    for anything unrecognized)."""
    return _SPLIT_POLICIES.get(_ARCH, _SPLIT_POLICIES["gfx942"])


@lru_cache(maxsize=8)
def _device_cu_count(device_index: int) -> int:
    """Compute-unit count for a CUDA/HIP device (cached); the arch's nominal
    count if the query fails."""
    try:
        return torch.cuda.get_device_properties(device_index).multi_processor_count
    except Exception:  # noqa: BLE001
        return _split_policy().fallback_cu


def _auto_num_splits(
    seq_len_padded: int,
    seq_len_kv: int,
    rows_per_block: int,
    block_kv: int,
    device_index: int,
) -> int:
    """KV-column splits (grid.y) to fill the device when the row grid is small.

    For small-M / large-N shapes the ``ceil(seq_len/RPB)`` row grid leaves the
    device block-starved; splitting each row's window across ``grid.y`` recovers
    occupancy at no correctness cost (logits[m,n] are independent across n).
    Returns 1 once the row grid alone oversubscribes the device. The three
    tuning constants are per-arch -- see ``_SPLIT_POLICIES``.
    """
    pol = _split_policy()
    grid_x = seq_len_padded // rows_per_block
    if grid_x == 0 or seq_len_kv < pol.min_seq_len_kv:
        return 1
    target_blocks = pol.cu_oversub * _device_cu_count(device_index)
    if grid_x >= target_blocks:
        return 1
    max_splits = max(1, (seq_len_kv // block_kv) // pol.min_tiles_per_split)
    return max(1, min(math.ceil(target_blocks / grid_x), max_splits))


# --------------------------------------------------------------------------- #
# MfmaAtom -- bundles every MFMA-shape-derived constant plus the rocdl functor,
# so the kernel builder carries no hardcoded tile shape. Supporting a new MFMA
# instruction is then a new MfmaAtom instance plus a _VARIANT_BUILDERS entry.
# --------------------------------------------------------------------------- #


def _make_operands_dense(a, b, acc):
    """Default ``MfmaAtom.make_operands``: the dense-MFMA 6-operand convention
    ``[a, b, c, cbsz, abid, blgp]`` (all zero besides the fragments)."""
    return [a, b, acc, 0, 0, 0]


def _make_operands_scaled_identity(a, b, acc):
    """``MfmaAtom.make_operands`` for the CDNA4 scaled MFMA atoms (K=128/64).

    These instructions always carry ``scaleA``/``scaleB`` UE8M0 operands as part
    of their encoding. Passing a compile-time identity scale (UE8M0 bias-127,
    i.e. every byte 127 -> multiplier 1.0) makes the hardware microscale a
    no-op; this kernel applies its own ``kv_scale`` in f32 after the MFMA
    (scale-hoisted out of the ReLU), so no other scale is needed.
    """
    scale = arith.constant(0x7F7F7F7F, type=T.i32)
    return [a, b, acc, 0, 0, 0, scale, 0, scale]


def _mfma32x32x64_fp8_fp8_scale_wrapper(result_type, operands, *, loc=None, ip=None):
    """Adapt the raw ODS ``mfma_scale_f32_32x32x64_f8f6f4`` to the
    ``(result_type, operands_list)`` convention used by ``flydsl.expr.rocdl``.

    ``operands`` follows the 9-element scaled-MFMA convention
    ``[a, b, c, cbsz, blgp, opselA, scaleA, opselB, scaleB]`` -- note there is
    no ``abid``, unlike the dense atoms. FlyDSL ships a friendly wrapper for
    the 16x16x128 scaled atom but not yet for 32x32x64.
    """
    a = _unwrap_mfma_operand(operands[0])
    b = _unwrap_mfma_operand(operands[1])
    c = _unwrap_mfma_operand(operands[2])
    cbsz = int(operands[3]) if len(operands) > 3 else 0
    blgp = int(operands[4]) if len(operands) > 4 else 0
    opsel_a = int(operands[5]) if len(operands) > 5 else 0
    scale_a = _unwrap_mfma_operand(operands[6]) if len(operands) > 6 else a
    opsel_b = int(operands[7]) if len(operands) > 7 else 0
    scale_b = _unwrap_mfma_operand(operands[8]) if len(operands) > 8 else b
    return _ods_mfma_scale32x32x64(
        result_type,
        a,
        b,
        c,
        cbsz,
        blgp,
        opsel_a,
        scale_a,
        opsel_b,
        scale_b,
        loc=loc,
        ip=ip,
    ).result


@dataclass(frozen=True)
class MfmaAtom:
    """MFMA-shape descriptor for the fp8 MQA-logits kernel.

    Fields
    ------
    name : str
        Shape tag, e.g. ``"16x16x32"``.
    MFMA_M, MFMA_N, MFMA_K : int
        Output tile is MFMA_M x MFMA_N; MFMA_K fp8 elements reduced per step.
    ACC_ELEMS : int
        f32 accumulator elements per lane (``vec<ACC_ELEMS x f32>``).
    fn : Callable
        FlyDSL ``rocdl.mfma_*`` functor taking ``(result_type, operands)``.
    shuffle_offsets : tuple[int, ...]
        ``shuffle_xor`` offsets for the in-wave head-reduce butterfly; must
        cover every lane group so the full H-wide sum is produced.
    acc_head_static_offsets : tuple[int, ...]
        Compile-time head offset within one MFMA_M tile, per accumulator
        element. Length == ACC_ELEMS. For element ``ii`` in lane group ``g``::

            head_within_tile = acc_head_static_offsets[ii]
                               + g * acc_head_group_stride
            weight_index     = mi * MFMA_M + head_within_tile

        For 16x16x32 (4 groups, ACC_ELEMS=4) the layout is sequential, so the
        offsets are ``(0, 1, 2, 3)`` with stride 4.
    acc_head_group_stride : int
        Multiplier for the lane-group index (4 on gfx942/gfx950 fp8 MFMA).
    frag_bytes : int
        A/B fragment bytes owned by one lane for one K-step. 8 for the dense
        atoms (one i64 load).
    make_operands : Callable
        Builds the ``fn`` operand list from ``(a_frag, b_frag, acc)``.
    kname_tag : str | None
        Shape tag used in the generated kernel symbol name. ``None`` means
        ``f"mfma{name}"``. ``_MFMA16`` pins the bare ``"mfma"`` it has always
        used so its generated symbols (and therefore its ISA) stay unchanged.
    """

    name: str
    MFMA_M: int
    MFMA_N: int
    MFMA_K: int
    ACC_ELEMS: int
    fn: Callable
    shuffle_offsets: tuple
    acc_head_static_offsets: tuple  # length == ACC_ELEMS
    acc_head_group_stride: int
    frag_bytes: int = 8
    make_operands: Callable = _make_operands_dense
    kname_tag: str | None = None


#: 16x16 output tile, K=32 fp8 elements/step. Acc: vec<4 x f32>.
#: Fragment layout: lane l -> A[row=l%16, k=(l//16)*8 + 0..7], col=l%16.
#: Writer lanes: l//16 == 0 (16 distinct output columns per tile).
#: Acc layout: acc[ii] in lane group g -> head g*4 + ii.
_MFMA16 = MfmaAtom(
    name="16x16x32",
    MFMA_M=16,
    MFMA_N=16,
    MFMA_K=32,
    ACC_ELEMS=4,
    fn=rocdl.mfma_f32_16x16x32_fp8_fp8,
    shuffle_offsets=(16, 32),
    acc_head_static_offsets=(0, 1, 2, 3),
    acc_head_group_stride=4,
    kname_tag="mfma",
)

#: gfx950/CDNA4 scaled MFMA: 16x16 output tile, K=128 fp8f6f4 elements/step.
#: Acc: vec<4 x f32> -- the same layout as _MFMA16, because tv_layout_c depends
#: only on (M, N), not on the reduction depth. Fragment: vector<8xi32>
#: (32 bytes/lane), 4x _MFMA16's 8-byte fragment, tracking the 4x K increase.
#: Requires native FN operands (this instruction rejects FNUZ) and, via the
#: generic ``D % MFMA_K`` assert, head_size % 128 == 0.
_MFMA16_K128 = MfmaAtom(
    name="16x16x128",
    MFMA_M=16,
    MFMA_N=16,
    MFMA_K=128,
    ACC_ELEMS=4,
    fn=rocdl.mfma_scale_f32_16x16x128_f8f6f4,
    shuffle_offsets=(16, 32),
    acc_head_static_offsets=(0, 1, 2, 3),
    acc_head_group_stride=4,
    frag_bytes=32,
    make_operands=_make_operands_scaled_identity,
)

#: gfx950/CDNA4 scaled MFMA: 32x32 output tile, K=64 fp8f6f4 elements/step.
#: Acc: vec<16 x f32>; the two lane groups interleave in blocks of 4, hence the
#: static offsets below. Fragment: vector<8xi32> (32 bytes/lane). Requires
#: native FN operands (rejects FNUZ); serves head_size 64 and 128.
_MFMA32_K64 = MfmaAtom(
    name="32x32x64",
    MFMA_M=32,
    MFMA_N=32,
    MFMA_K=64,
    ACC_ELEMS=16,
    fn=_mfma32x32x64_fp8_fp8_scale_wrapper,
    shuffle_offsets=(32,),
    acc_head_static_offsets=(
        # ii=0..3 -> g*4 + 0..3, ii=4..7 -> g*4 + 8..11, etc.
        0,
        1,
        2,
        3,
        8,
        9,
        10,
        11,
        16,
        17,
        18,
        19,
        24,
        25,
        26,
        27,
    ),
    acc_head_group_stride=4,
    frag_bytes=32,
    make_operands=_make_operands_scaled_identity,
)


def _build_kernel_mfma_r_w(
    *,
    num_heads: int,
    head_size: int,
    block_kv: int,
    rows_per_block: int,
    waves_per_block: int,
    mfma: MfmaAtom = _MFMA16,
    convert_q_fn: bool = False,
    convert_kv_fn: bool = False,
    clean_logits: bool = True,
):
    """Multi-row, multi-wave MFMA kernel.

    ``rows_per_block`` query rows share one KV tile load (cuts KV traffic by RPB).
    ``waves_per_block`` waves execute per block; each wave owns a disjoint slice of
    the BKV column tiles (``N_TILES // WPB`` tiles per wave), so all WPB waves can
    execute in parallel with no cross-wave LDS or barrier.

    Thread decomposition:
      * ``tid = wave * 64 + lane``  (tid: 0..MR_BLOCK_THREADS-1)
      * Wave ``w`` owns n-tiles ``[w*N_TILES_PER_WAVE, (w+1)*N_TILES_PER_WAVE)``
        within each BKV tile.
      * A-operand (Q) layout and head-reduce are per-lane within the wave (width 64).

    Grid: ``(ceil(seq_len / RPB), num_splits, 1)``.  The host pads ``seq_len`` to
    a multiple of ``RPB`` (every block owns exactly ``RPB`` rows) and may split
    each row's KV window across ``grid.y`` blocks when the row grid alone is too
    small to fill the device (see ``flydsl_fp8_mqa_logits``).
    """
    H = num_heads
    D = head_size
    BKV = block_kv
    RPB = rows_per_block
    WPB = waves_per_block
    MR_BLOCK_THREADS = 64 * WPB

    # MFMA tile dims come from the atom: MFMA_M x MFMA_N output tile, MFMA_K
    # fp8 elements reduced per MFMA step.
    MFMA_M = mfma.MFMA_M
    MFMA_N = mfma.MFMA_N
    MFMA_K = mfma.MFMA_K
    ACC_ELEMS = mfma.ACC_ELEMS
    FRAG_BYTES = mfma.frag_bytes

    assert H % MFMA_M == 0, f"num_heads={H} must be a multiple of MFMA_M={MFMA_M}"
    assert BKV % MFMA_N == 0, f"block_kv={BKV} must be a multiple of MFMA_N={MFMA_N}"
    assert D % MFMA_K == 0, f"head_size={D} must be a multiple of MFMA_K={MFMA_K}"
    assert RPB >= 1, "rows_per_block must be >= 1"
    assert WPB >= 1, "waves_per_block must be >= 1"
    # The CDNA4 scaled atoms consume native FN operands and reject FNUZ, so the
    # in-kernel FN->FNUZ patch must never be combined with them. The host only
    # sets these flags on gfx942 (where only dense atoms are used), so this is a
    # guard against future mis-wiring rather than a reachable path.
    assert not (mfma.frag_bytes == 32 and (convert_q_fn or convert_kv_fn)), (
        f"atom {mfma.name} requires native FN operands; "
        "FN->FNUZ conversion is not supported for it"
    )
    N_TILES = BKV // MFMA_N  # total column-tiles per BKV block
    assert (
        N_TILES % WPB == 0
    ), f"BKV/MFMA_N={N_TILES} must be divisible by waves_per_block={WPB}"
    M_TILES = H // MFMA_M  # head row-tiles
    K_STEPS = D // MFMA_K  # MFMA K-steps over the head dim
    N_TILES_PER_WAVE = N_TILES // WPB  # column-tiles per wave

    fm_fast = arith.FastMathFlags.fast
    mfma_fn = mfma.fn

    _cvt_tag = ""
    if convert_q_fn:
        _cvt_tag += "_cq"
    if convert_kv_fn:
        _cvt_tag += "_ck"
    # Only the non-default is tagged, so the common clean_logits=True symbols
    # keep the names they have always had (same convention as _cvt_tag).
    _cl_tag = "" if clean_logits else "_nocl"
    _shape_tag = mfma.kname_tag or f"mfma{mfma.name}"
    _kname = (
        f"fp8_mqa_logits_H{H}_D{D}_bkv{BKV}_{_shape_tag}_r{RPB}_w{WPB}"
        f"{_cvt_tag}{_cl_tag}_flydsl"
    )

    @flyc.kernel(name=_kname, known_block_size=[MR_BLOCK_THREADS, 1, 1])
    def kernel(
        Q: fx.Tensor,  # [seq_len, H, D]       fp8 (bytes passed raw)
        KV: fx.Tensor,  # [seq_len_kv, D]       fp8 (bytes passed raw)
        kv_scales: fx.Tensor,  # [seq_len_kv]          f32
        weights: fx.Tensor,  # [seq_len, H]          f32
        cu_starts: fx.Tensor,  # [seq_len]             i32
        cu_ends: fx.Tensor,  # [seq_len]             i32
        logits: fx.Tensor,  # [seq_len, seq_len_kv] f32
        seq_len: fx.Int32,  # padded to a multiple of RPB
        seq_len_kv: fx.Int32,
        stride_logits_s: fx.Int32,
        num_splits: fx.Int32,  # grid.y KV-column splits (1 == no split)
    ):
        f32_0 = arith.constant(0.0, type=T.f32)
        mfma_res_ty = Vec.make_type(ACC_ELEMS, fx.Float32)

        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        # Blocks are assigned in reverse order (bid=0 → last rows, bid=n_blocks-1 → row 0)
        # as a load-balancing heuristic: KV windows tend to be longer for later query rows,
        # so reversing ensures the GPU scheduler picks up the heaviest work first rather than
        # leaving it for last.
        n_blocks = fx.Int32(arith.ceildivui(_to_raw(seq_len), _to_raw(fx.Int32(RPB))))
        r0 = fx.Int32(
            arith.muli(
                _to_raw(n_blocks - bid - fx.Int32(1)),
                _to_raw(fx.Int32(RPB)),
            )
        )

        # Decompose tid into wave index and in-wave lane.
        wave = fx.Int32(arith.divui(_to_raw(tid), _to_raw(fx.Int32(64))))
        lane = fx.Int32(arith.remui(_to_raw(tid), _to_raw(fx.Int32(64))))
        lane_div_N = fx.Int32(arith.divui(_to_raw(lane), _to_raw(fx.Int32(MFMA_N))))
        lane_mod_N = fx.Int32(arith.remui(_to_raw(lane), _to_raw(fx.Int32(MFMA_N))))
        # Byte offset of this lane's fragment within the K-step (FRAG_BYTES per
        # lane group). 8 for the dense atoms.
        lane_frag_off = fx.Int32(
            arith.muli(_to_raw(lane_div_N), _to_raw(fx.Int32(FRAG_BYTES)))
        )

        # fp8 operands are read 8 bytes at a time as 2 i32 dwords (v8i8
        # buffer_load fails to lower on gfx942), bitcast to i64 for the MFMA.
        q_i32 = GTensor(Q, dtype=T.i32, shape=(-1,))
        kv_i32 = GTensor(KV, dtype=T.i32, shape=(-1,))
        sc_t = GTensor(kv_scales, dtype=T.f32, shape=(-1,))
        w_t = GTensor(weights, dtype=T.f32, shape=(-1, H))
        cs_t = GTensor(cu_starts, dtype=T.i32, shape=(-1,))
        ce_t = GTensor(cu_ends, dtype=T.i32, shape=(-1,))
        # Per-row 1-D output view: the row's i64 byte offset goes into the base
        # pointer so the remaining column offset stays in i32. A 2-D (row, col)
        # view computes row * stride + col in i32 and overflows past 2^31
        # (~46k-square dense outputs), silently mis-writing.
        _stride_i64 = arith.extui(T.i64, _to_raw(stride_logits_s))

        def _load_pack_i64(i32_view, byte_off_i32):
            dword_off = fx.Int32(
                arith.divui(_to_raw(byte_off_i32), _to_raw(fx.Int32(4)))
            )
            v2 = i32_view.vec_load((dword_off,), vec_size=2)
            return Vec(v2).bitcast(fx.Int64)[0].ir_value()

        def _load_frag(i32_view, base_i32, k_byte_i32, convert_fn):
            """Load one lane's A/B fragment for one K-step.

            Dense atoms (frag_bytes=8) take a 64-bit load and may need the
            FN->FNUZ patch; the CDNA4 scaled atoms (frag_bytes=32) take the
            two-dwordx4 path and are always native FN.
            """
            off = _i32_add(base_i32, _i32_add(k_byte_i32, lane_frag_off))
            if const_expr(FRAG_BYTES == 32):
                return _load_pack_i32x8(i32_view, off)
            raw = _load_pack_i64(i32_view, off)
            return _fn_to_fnuz_i64(raw) if convert_fn else raw

        def _fn_to_fnuz_i64(raw_i64):
            """Map FN byte 0x80 (neg-zero) -> 0x00 in 8 packed fp8 bytes."""
            lo_i32 = arith.TruncIOp(T.i32, raw_i64).result
            hi_i64 = arith.ShRUIOp(raw_i64, arith.constant(32, type=T.i64)).result
            hi_i32 = arith.TruncIOp(T.i32, hi_i64).result

            def _fix_i32(src):
                result = arith.constant(0, type=T.i32)
                for byte_idx in range_constexpr(4):
                    shift = arith.constant(byte_idx * 8, type=T.i32)
                    byte_val = arith.andi(
                        arith.shrui(src, shift),
                        arith.constant(0xFF, type=T.i32),
                    )
                    is_0x80 = arith.cmpi(
                        arith.CmpIPredicate.eq,
                        byte_val,
                        arith.constant(0x80, type=T.i32),
                    )
                    cleaned = arith.select(
                        is_0x80,
                        arith.constant(0, type=T.i32),
                        byte_val,
                    )
                    result = arith.ori(result, arith.shli(cleaned, shift))
                return result

            lo_fix = _fix_i32(lo_i32)
            hi_fix = _fix_i32(hi_i32)
            lo_64 = arith.ExtUIOp(T.i64, lo_fix).result
            hi_64 = arith.ShLIOp(
                arith.ExtUIOp(T.i64, hi_fix).result, arith.constant(32, type=T.i64)
            ).result
            return arith.OrIOp(lo_64, hi_64).result

        # ---- Preload window bounds, Q frags, and weights for all RPB rows ----
        # A-operand layout is per in-wave lane, so `lane` (not `tid`) indexes Q.
        starts = [None] * RPB
        ends = [None] * RPB
        a_packs = [None] * RPB
        w_frag = [None] * RPB

        for j in range_constexpr(RPB):
            row = _i32_add(r0, fx.Int32(j))
            s = fx.Int32(cs_t[row])
            e = fx.Int32(ce_t[row])
            starts[j] = fx.Int32(arith.maxsi(_to_raw(s), _to_raw(fx.Int32(0))))
            ends[j] = fx.Int32(arith.minsi(_to_raw(e), _to_raw(fx.Int32(seq_len_kv))))

            # lane -> Q[row, h = mi*MFMA_M + lane%MFMA_N,
            #            d = kk*MFMA_K + (lane//MFMA_N)*8 + 0..7]
            row_a = [[None] * K_STEPS for _ in range_constexpr(M_TILES)]
            for mi in range_constexpr(M_TILES):
                h_a = _i32_add(fx.Int32(mi * MFMA_M), lane_mod_N)
                row_h = _i32_add(
                    fx.Int32(arith.muli(_to_raw(row), _to_raw(fx.Int32(H)))), h_a
                )
                base_a = fx.Int32(arith.muli(_to_raw(row_h), _to_raw(fx.Int32(D))))
                for kk in range_constexpr(K_STEPS):
                    row_a[mi][kk] = _load_frag(
                        q_i32, base_a, fx.Int32(kk * MFMA_K), convert_q_fn
                    )
            a_packs[j] = row_a

            # weights[row, h] per (mi, ii): the head this accumulator element
            # belongs to is mi*MFMA_M + acc_head_static_offsets[ii]
            # + lane_div_N*acc_head_group_stride (see MfmaAtom).
            row_w = [[None] * ACC_ELEMS for _ in range_constexpr(M_TILES)]
            lane_head = fx.Int32(
                arith.muli(
                    _to_raw(lane_div_N), _to_raw(fx.Int32(mfma.acc_head_group_stride))
                )
            )
            for mi in range_constexpr(M_TILES):
                for ii in range_constexpr(ACC_ELEMS):
                    h_w = _i32_add(
                        fx.Int32(mi * MFMA_M + mfma.acc_head_static_offsets[ii]),
                        lane_head,
                    )
                    row_w[mi][ii] = _to_raw(fx.Float32(w_t[row, h_w]))
            w_frag[j] = row_w

        # ---- Union window across all RPB rows ----
        tile_start = _to_raw(starts[0])
        tile_end = _to_raw(ends[0])
        for j in range_constexpr(1, RPB):
            tile_start = arith.minsi(tile_start, _to_raw(starts[j]))
            tile_end = arith.maxsi(tile_end, _to_raw(ends[j]))
        # Align tile_start down to BKV boundary.
        tile_start = arith.muli(
            arith.divui(tile_start, _to_raw(fx.Int32(BKV))),
            _to_raw(fx.Int32(BKV)),
        )
        # Collapse an empty union window to a zero-width one at tile_start.
        # ``ends`` is clamped above by seq_len_kv but not below, so a row whose
        # cu_ends is negative or any row with cu_ends <= cu_starts
        # can leave tile_end < tile_start.
        tile_end = arith.maxsi(tile_end, tile_start)

        # ---- KV-column split across grid.y. Block (.,by) takes a BKV-aligned
        # slice of the union window; logits[m,n] are independent across n, so
        # this is pure parallelism with no reduction. The slices tile [start,end)
        # exactly (disjoint, gap-free), so each column has one writer.
        # num_splits==1 collapses to the full window (by==0). ----
        by = fx.block_idx.y
        win_tiles = arith.ceildivui(
            arith.subi(tile_end, tile_start), _to_raw(fx.Int32(BKV))
        )
        split_cols = arith.muli(
            arith.ceildivui(win_tiles, _to_raw(num_splits)),
            _to_raw(fx.Int32(BKV)),
        )
        tile_start = arith.addi(tile_start, arith.muli(_to_raw(by), split_cols))
        tile_end = arith.minsi(arith.addi(tile_start, split_cols), tile_end)

        tile_lo = _to_raw(fx.Index(tile_start))
        tile_hi = _to_raw(fx.Index(tile_end))
        tile_step = _to_raw(fx.Index(BKV))
        tile_loop = scf.ForOp(tile_lo, tile_hi, tile_step, [])
        with ir.InsertionPoint(tile_loop.body):
            col0 = fx.Int32(arith.index_cast(T.i32, tile_loop.induction_variable))

            # ---- Load B-frags: wave w owns its own disjoint slice of n-tiles
            # [w*N_TILES_PER_WAVE, (w+1)*N_TILES_PER_WAVE) (no cross-wave sharing). ----
            wave_ni_base = fx.Int32(
                arith.muli(_to_raw(wave), _to_raw(fx.Int32(N_TILES_PER_WAVE)))
            )
            b_packs = [[None] * K_STEPS for _ in range_constexpr(N_TILES_PER_WAVE)]
            kv_scales_tile = [None] * N_TILES_PER_WAVE
            cols = [None] * N_TILES_PER_WAVE
            for ni in range_constexpr(N_TILES_PER_WAVE):
                abs_ni = _i32_add(wave_ni_base, fx.Int32(ni))
                col = _i32_add(
                    _i32_add(
                        col0,
                        fx.Int32(
                            arith.muli(_to_raw(abs_ni), _to_raw(fx.Int32(MFMA_N)))
                        ),
                    ),
                    lane_mod_N,
                )
                cols[ni] = col
                col_clamped = fx.Int32(
                    arith.minsi(
                        _to_raw(col), _to_raw(fx.Int32(seq_len_kv) - fx.Int32(1))
                    )
                )
                kv_scales_tile[ni] = _to_raw(fx.Float32(sc_t[col_clamped]))
                base_b = fx.Int32(
                    arith.muli(_to_raw(col_clamped), _to_raw(fx.Int32(D)))
                )
                for kk in range_constexpr(K_STEPS):
                    b_packs[ni][kk] = _load_frag(
                        kv_i32, base_b, fx.Int32(kk * MFMA_K), convert_kv_fn
                    )

            # ---- Per-row MFMA + epilogue (inner loop over RPB rows) ----
            for j in range_constexpr(RPB):
                row = _i32_add(r0, fx.Int32(j))
                out_row_t = _make_out_row_t(logits, _stride_i64, row)
                for ni in range_constexpr(N_TILES_PER_WAVE):
                    col = cols[ni]
                    kv_scale = kv_scales_tile[ni]
                    col_sum = _to_raw(f32_0)
                    for mi in range_constexpr(M_TILES):
                        acc = Vec.filled(ACC_ELEMS, 0.0, fx.Float32)
                        for kk in range_constexpr(K_STEPS):
                            acc = mfma_fn(
                                mfma_res_ty,
                                mfma.make_operands(
                                    a_packs[j][mi][kk], b_packs[ni][kk], acc
                                ),
                            )
                        # kv_scale (>=0) is hoisted out of the head sum: ReLU is
                        # positive-homogeneous, so ReLU(s*x)=s*ReLU(x) and the
                        # whole column sum is scaled once (below) instead of every
                        # head term -- drops M_TILES*4 muls to one.
                        for ii in range_constexpr(ACC_ELEMS):
                            score = Vec(acc)[ii].ir_value()
                            relu = arith.maximumf(score, _to_raw(f32_0))
                            wsc = arith.MulFOp(
                                relu, w_frag[j][mi][ii], fastmath=fm_fast
                            ).result
                            col_sum = arith.AddFOp(
                                col_sum, wsc, fastmath=fm_fast
                            ).result
                    col_sum = arith.MulFOp(col_sum, kv_scale, fastmath=fm_fast).result

                    # Head-reduce within the wave (width=64) via the atom's
                    # shuffle_xor butterfly (16, 32 for the 16x16 atoms).
                    for sh in mfma.shuffle_offsets:
                        peer = _to_raw(ArithValue(col_sum).shuffle_xor(sh, 64))
                        col_sum = arith.AddFOp(col_sum, peer, fastmath=fm_fast).result

                    # Only lane_div_N==0 lanes hold the MFMA_N distinct columns.
                    # `col >= start` is required: the tile loop is BKV-aligned
                    # below `start`, so it guards the -inf that the fused fill
                    # below writes into [aligned_start, start).
                    in_window = arith.andi(
                        _to_raw(
                            arith.cmpi(
                                arith.CmpIPredicate.sge,
                                _to_raw(col),
                                _to_raw(starts[j]),
                            )
                        ),
                        _to_raw(
                            arith.cmpi(
                                arith.CmpIPredicate.slt,
                                _to_raw(col),
                                _to_raw(ends[j]),
                            )
                        ),
                    )
                    is_writer = arith.andi(
                        _to_raw(
                            arith.cmpi(
                                arith.CmpIPredicate.eq,
                                _to_raw(lane_div_N),
                                _to_raw(fx.Int32(0)),
                            )
                        ),
                        in_window,
                    )
                    with ir.InsertionPoint(scf.IfOp(is_writer).then_block):
                        out_row_t[col] = fx.Float32(col_sum)
                        scf.YieldOp([])

            scf.YieldOp([])

        # ---- Fused clean_logits prefill (must come after the tile loop) ----
        if const_expr(clean_logits):
            _emit_row_neg_inf_fill(
                logits=logits,
                stride_i64=_stride_i64,
                rows=[_i32_add(r0, fx.Int32(j)) for j in range_constexpr(RPB)],
                starts=starts,
                ends=ends,
                seq_len_kv=seq_len_kv,
                tid_i32=tid,  # every thread in the block cooperates
                nthreads=MR_BLOCK_THREADS,
                by_i32=by,
                num_splits=num_splits,
            )

    @flyc.jit
    def launch_fp8_mqa_logits_mfma_r_w(
        Q: fx.Tensor,
        KV: fx.Tensor,
        kv_scales: fx.Tensor,
        weights: fx.Tensor,
        cu_starts: fx.Tensor,
        cu_ends: fx.Tensor,
        logits: fx.Tensor,
        seq_len: fx.Int32,
        seq_len_kv: fx.Int32,
        stride_logits_s: fx.Int32,
        num_splits: fx.Int32,
        stream: fx.Stream,
    ):
        n_blocks = arith.ceildivui(_to_raw(seq_len), _to_raw(fx.Int32(RPB)))
        gx = arith.index_cast(T.index, n_blocks)
        gy = arith.index_cast(T.index, _to_raw(num_splits))
        kernel._func.__name__ = _kname
        kernel(
            Q,
            KV,
            kv_scales,
            weights,
            cu_starts,
            cu_ends,
            logits,
            seq_len,
            seq_len_kv,
            stride_logits_s,
            num_splits,
        ).launch(grid=(gx, gy, 1), block=(MR_BLOCK_THREADS, 1, 1), stream=stream)

    return launch_fp8_mqa_logits_mfma_r_w


def _build_kernel_mfma_lds_pipe(
    *,
    num_heads: int,
    head_size: int,
    block_kv: int,
    rows_per_block: int,
    waves_per_block: int,
    mfma: MfmaAtom,
    convert_q_fn: bool = False,
    convert_kv_fn: bool = False,
    clean_logits: bool = True,
    swizzle: bool = False,
    num_buffers: int = 2,
    prefetch_depth: int = 2,
):
    """LDS multi-buffered variant for gfx950 MfmaAtoms (scaled CDNA4 atoms).

    Parallel to ``_build_kernel_mfma_r_w`` but stages KV through a multi-slot LDS
    buffer filled by async global->LDS DMA (``raw_ptr_buffer_load_lds``),
    with an explicit software pipeline (prefetch tile 0..PD-1, then per-tile
    ``s_waitcnt`` + prefetch(i+PD) + compute).

    Work partition:
      * All ``WPB`` waves cooperatively load ONE ``BKV``-wide K-tile into LDS and
        all read it. Each wave owns a disjoint group of ``RPW`` query ROWS and
        iterates over all ``N_TILES`` columns of the shared LDS tile.
      * A block owns ``ROWS_PER_BLOCK = RPW * WPB`` query rows (wave ``w`` owns
        rows ``[w*RPW, (w+1)*RPW)``). KV reuse factor becomes ``RPW * WPB``.

    A-frags are loaded from global memory to registers; B-frags are read from LDS.
    The epilogue is identical to the direct-load builder.
    """
    H = num_heads
    D = head_size
    BKV = block_kv
    RPW = rows_per_block  # rows per WAVE here (block owns RPW*WPB rows)
    WPB = waves_per_block
    MR_BLOCK_THREADS = 64 * WPB
    ROWS_PER_BLOCK = RPW * WPB

    assert mfma.frag_bytes == 32, (
        "_build_kernel_mfma_lds_pipe currently supports only the CDNA4 scaled "
        "atoms (frag_bytes=32)."
    )
    assert (
        H % mfma.MFMA_M == 0
    ), f"Number of heads must be a multiple of MFMA_M ({mfma.MFMA_M})"
    assert (
        BKV % mfma.MFMA_N == 0
    ), f"Block KV size must be a multiple of MFMA_N ({mfma.MFMA_N})"
    assert (
        D % mfma.MFMA_K == 0
    ), f"Head size must be a multiple of MFMA_K ({mfma.MFMA_K})"
    assert RPW >= 1 and WPB >= 1, "Rows per wave and waves per block must be >= 1"

    N_TILES = BKV // mfma.MFMA_N
    M_TILES = H // mfma.MFMA_M
    K_STEPS = D // mfma.MFMA_K

    # LDS multi-buffer: NUM_BUFFERS slots of [BKV, D] fp8 (row-major, row == KV
    # column index). Addressed as i32 dwords for the vector reads.
    #
    # Pipeline depth is set by PREFETCH_DEPTH (tiles kept in flight during the
    # per-tile compute).  The tile-i+PREFETCH_DEPTH prefetch targets slot
    # (i+PD)%NB; when NB > PD that slot differs from the one being read (slot
    # i%NB), so the reader-before-writer barrier ("barrier B") can be dropped.
    # NB == PD (e.g. the 2-buffer/depth-2 case) reuses the just-read slot
    # and still needs barrier B.
    NUM_BUFFERS = num_buffers
    PREFETCH_DEPTH = prefetch_depth
    assert (
        NUM_BUFFERS >= PREFETCH_DEPTH >= 1
    ), f"need num_buffers({NUM_BUFFERS}) >= prefetch_depth({PREFETCH_DEPTH}) >= 1"
    _need_barrier_b = NUM_BUFFERS <= PREFETCH_DEPTH
    SLOT_BYTES = BKV * D  # fp8, 1 byte/elem
    SLOT_I32 = SLOT_BYTES // 4
    # gfx950 raw_ptr_buffer_load_lds supports size=16 (dwordx4).
    DMA_BYTES = 16
    assert SLOT_BYTES % (MR_BLOCK_THREADS * DMA_BYTES) == 0, (
        f"SLOT_BYTES={SLOT_BYTES} must be divisible by "
        f"MR_BLOCK_THREADS*DMA_BYTES={MR_BLOCK_THREADS * DMA_BYTES}"
    )
    NUM_ASYNC_LOADS = SLOT_BYTES // (MR_BLOCK_THREADS * DMA_BYTES)
    # vmcnt to leave outstanding at the top of each tile: the DMAs of the
    # PREFETCH_DEPTH-1 tiles queued behind the one about to be read.
    _WAIT_VMCNT = (PREFETCH_DEPTH - 1) * NUM_ASYNC_LOADS
    assert _WAIT_VMCNT <= 63, (
        f"prefetch_depth={PREFETCH_DEPTH} x {NUM_ASYNC_LOADS} DMAs/tile needs "
        f"vmcnt={_WAIT_VMCNT}, past the 63 the gfx9 encoding holds; "
        "lower prefetch_depth or block_kv, or raise waves_per_block"
    )

    # XOR swizzle (bank-conflict avoidance).
    # The slot stores [BKV, D] fp8 HEAD_SIZE-
    # contiguous, so per column the D bytes are DW_PER_COL i32 dwords.
    # Single B-frag read gathers a fixed CHUNK_DW-dword slice of the head dim across the 32/16
    # lanes of a KV-column group; since the per-column stride D/4 is a multiple
    # of the 32 LDS banks, every lane hits the same banks (up to 32-way
    # conflict).  XOR-ing the within-column chunk index with a function of the
    # column index "n" scatters the NC chunks across the banks, cutting the
    # conflict by NC (=D/frag_bytes) while keeping each frag read (and each
    # 16B DMA write) contiguous, because the XOR mask is a multiple of CHUNK_DW.
    #   phys_dword(n, c) = n*DW_PER_COL + (c XOR ((n & (NC-1)) * CHUNK_DW))
    DW_PER_COL = D // 4  # i32 dwords per KV column (head dim)
    CHUNK_DW = mfma.frag_bytes // 4  # dwords per B-frag read (=8)
    NC = DW_PER_COL // CHUNK_DW  # chunks per column (D/frag_bytes)
    if swizzle:
        assert (
            NC >= 2
        ), f"swizzle needs D/frag_bytes>=2 (D={D}, frag_bytes={mfma.frag_bytes})"

    fm_fast = arith.FastMathFlags.fast
    mfma_fn = mfma.fn

    # Using raw_ptr_buffer_load_lds requires the destination LDS address to be aligned to at least 128 bytes.
    # Actually memory allocation takes place later at allocator.finalize().
    allocator = SmemAllocator(None, arch=get_gfx())
    lds_off = allocator._align(allocator.ptr, 128)
    allocator.ptr = lds_off + NUM_BUFFERS * SLOT_BYTES

    # As in the direct-load builder: only the non-default clean_logits is tagged.
    _cl_tag = "" if clean_logits else "_nocl"
    _kname = (
        f"fp8_mqa_logits_H{H}_D{D}_mfma{mfma.name}"
        f"_bkv{BKV}_r{RPW}_w{WPB}_lds{NUM_BUFFERS}"
        f"{'_swizzled' if swizzle else ''}{_cl_tag}_flydsl"
    )

    @flyc.kernel(name=_kname, known_block_size=[MR_BLOCK_THREADS, 1, 1])
    def kernel(
        Q: fx.Tensor,
        KV: fx.Tensor,
        kv_scales: fx.Tensor,
        weights: fx.Tensor,
        cu_starts: fx.Tensor,
        cu_ends: fx.Tensor,
        logits: fx.Tensor,
        seq_len: fx.Int32,  # padded to a multiple of ROWS_PER_BLOCK
        seq_len_kv: fx.Int32,
        stride_logits_s: fx.Int32,
        num_splits: fx.Int32,
    ):
        f32_0 = arith.constant(0.0, type=T.f32)
        i32_64 = arith.constant(64, type=T.i32)

        _mfma_res_ty = Vec.make_type(mfma.ACC_ELEMS, fx.Float32)

        tid = fx.thread_idx.x
        bid = fx.block_idx.x

        # Blocks are assigned in reverse order (bid=0 -> last rows, bid=n_blocks-1 -> row 0)
        # as a load-balancing heuristic: KV windows tend to be longer for later query rows,
        # so reversing ensures the GPU scheduler picks up the heaviest work first rather than
        # leaving it for last.
        n_blocks = fx.Int32(
            arith.ceildivui(_to_raw(seq_len), _to_raw(fx.Int32(ROWS_PER_BLOCK)))
        )
        block_row0 = fx.Int32((n_blocks - bid - fx.Int32(1)) * fx.Int32(ROWS_PER_BLOCK))

        wave = tid // i32_64
        lane = tid % i32_64
        lane_div_N = lane // fx.Int32(mfma.MFMA_N)
        lane_mod_N = lane % fx.Int32(mfma.MFMA_N)
        lane_frag_off = lane_div_N * fx.Int32(mfma.frag_bytes)

        # First row owned by this wave.
        wave_row0 = block_row0 + wave * fx.Int32(RPW)

        q_i32 = GTensor(Q, dtype=T.i32, shape=(-1,))
        kv_i32 = GTensor(KV, dtype=T.i32, shape=(-1,))
        kv_rsrc = kv_i32.rsrc
        sc_t = GTensor(kv_scales, dtype=T.f32, shape=(-1,))
        w_t = GTensor(weights, dtype=T.f32, shape=(-1, H))
        cs_t = GTensor(cu_starts, dtype=T.i32, shape=(-1,))
        ce_t = GTensor(cu_ends, dtype=T.i32, shape=(-1,))
        _stride_i64 = arith.extui(T.i64, _to_raw(stride_logits_s))

        # ---- LDS region + async-DMA base pointer ----
        # View of the LDS region as a flat array i32 values.
        # lds_st is for MFMA reads, lds_ptr0 is for DMA writes.
        base_ptr = allocator.get_base()
        region_ptr = SmemPtr(base_ptr, lds_off, T.i32, shape=(NUM_BUFFERS * SLOT_I32,))
        lds_st = STensor(region_ptr, T.i32, shape=(NUM_BUFFERS * SLOT_I32,))
        lds_base_idx = memref_dialect.extract_aligned_pointer_as_index(lds_st.memptr)
        # Address space 3 is the LDS address space for raw_ptr_buffer_load_lds.
        lds_ptr0 = buffer_ops.create_llvm_ptr(fx.Int64(lds_base_idx), address_space=3)

        def _dma_kv_tile_to_lds(slot_byte_i32, col0_i32):
            """Cooperatively async-copy KV[col0:col0+BKV, :] into LDS slot.

            All MR_BLOCK_THREADS threads participate; thread ``tid`` at load ``i``
            writes LDS byte ``(i*MR_BLOCK_THREADS + tid)*DMA_BYTES`` (relative to
            the slot), reading the matching linear byte of the row-major tile.
            OOB columns are clamped to ``seq_len_kv-1`` (harmless -- masked out in
            the epilogue by the per-row window predicate).
            """
            wave_slot_i32 = slot_byte_i32 + wave * fx.Int32(64 * DMA_BYTES)
            wave_slot_scalar = rocdl.readfirstlane(
                fx.Int64.ir_type, arith.extui(T.i64, _to_raw(wave_slot_i32))
            )
            lds_ptr = buffer_ops.get_element_ptr(lds_ptr0, wave_slot_scalar)

            dma_bytes = fx.Int32(DMA_BYTES)
            d = fx.Int32(D)
            seq_len_kv_m_1 = seq_len_kv - fx.Int32(1)

            for i in range_constexpr(NUM_ASYNC_LOADS):
                lin_bytes = (tid + fx.Int32(i * MR_BLOCK_THREADS)) * dma_bytes
                row_local = lin_bytes // d
                d_off = lin_bytes - row_local * d

                if const_expr(swizzle):
                    # The DMA writes lane-contiguously to physical byte lin_bytes,
                    # so to store the swizzled tile we fetch the logical element
                    # that maps to this physical slot: invert the within-column
                    # XOR (mask in bytes = (n & (NC-1)) * frag_bytes).
                    _mask_b = (row_local & fx.Int32(NC - 1)) * fx.Int32(mfma.frag_bytes)
                    d_off = fx.Int32(arith.xori(_to_raw(d_off), _to_raw(_mask_b)))

                col = col0_i32 + row_local
                col_cl = fx.Int32(arith.minsi(_to_raw(col), _to_raw(seq_len_kv_m_1)))
                voffset = col_cl * d + d_off
                if const_expr(i > 0):
                    lds_ptr = buffer_ops.get_element_ptr(
                        lds_ptr,
                        static_byte_offset=MR_BLOCK_THREADS * DMA_BYTES,
                    )
                rocdl.raw_ptr_buffer_load_lds(
                    kv_rsrc,
                    lds_ptr,
                    dma_bytes,
                    fx.Int32(voffset),
                    fx.Int32(0),
                    fx.Int32(0),
                    fx.Int32(1),
                )

        # ---- Preload this wave's RPW rows: window, Q A-frags, weights ----
        starts = [None] * RPW
        ends = [None] * RPW
        a_packs = [None] * RPW
        w_frag = [None] * RPW

        # Loop over the rows owned by this wave.
        for j in range_constexpr(RPW):
            row = wave_row0 + fx.Int32(j)
            s = fx.Int32(cs_t[row])
            e = fx.Int32(ce_t[row])
            starts[j] = fx.Int32(arith.maxsi(_to_raw(s), _to_raw(fx.Int32(0))))
            ends[j] = fx.Int32(arith.minsi(_to_raw(e), _to_raw(fx.Int32(seq_len_kv))))

            # Load A-frags:
            # Q[
            #   row,
            #   h = mi*MFMA_M + lane%MFMA_N,
            #   d = kk*MFMA_K + (lane//MFMA_N)*8 + 0..7
            #  ]
            row_a_frag = [[None] * K_STEPS for _ in range_constexpr(M_TILES)]
            for mi in range_constexpr(M_TILES):
                h_a = fx.Int32(mi * mfma.MFMA_M) + lane_mod_N
                row_h = h_a + row * fx.Int32(H)
                base_a = row_h * fx.Int32(D)
                for kk in range_constexpr(K_STEPS):
                    row_a_frag[mi][kk] = _load_pack_i32x8(
                        q_i32,
                        _i32_add(
                            base_a, _i32_add(fx.Int32(kk * mfma.MFMA_K), lane_frag_off)
                        ),
                    )
            a_packs[j] = row_a_frag

            # Load weights: weights[row, h] per (mi, ii).
            # The head this accumulator element belongs to is
            # mi*MFMA_M + acc_head_static_offsets[ii] + lane_div_N*acc_head_group_stride.
            row_w = [[None] * mfma.ACC_ELEMS for _ in range_constexpr(M_TILES)]
            h_w_static_offset = lane_div_N * fx.Int32(mfma.acc_head_group_stride)
            for mi in range_constexpr(M_TILES):
                for ii in range_constexpr(mfma.ACC_ELEMS):
                    static_off = mfma.acc_head_static_offsets[ii]
                    h_w = fx.Int32(mi * mfma.MFMA_M + static_off) + h_w_static_offset
                    row_w[mi][ii] = _to_raw(fx.Float32(w_t[row, h_w]))
            w_frag[j] = row_w

        # ---- Union KV window across all block rows (all waves cooperate) ----
        u_start = None
        u_end = None
        # Compute the union KV window [u_start, u_end) across all rows in this block.
        # All ROWS_PER_BLOCK query rows share a single KV tile scan over this union
        # interval, so each KV tile is loaded once and reused by every row.
        #   u_start = min(cu_starts[rows])
        #   u_end = max(cu_ends[rows]).
        for jj in range_constexpr(ROWS_PER_BLOCK):
            rr = block_row0 + fx.Int32(jj)
            ss = arith.maxsi(_to_raw(fx.Int32(cs_t[rr])), _to_raw(fx.Int32(0)))
            ee = arith.minsi(_to_raw(fx.Int32(ce_t[rr])), _to_raw(fx.Int32(seq_len_kv)))
            if jj == 0:
                u_start = ss
                u_end = ee
            else:
                u_start = arith.minsi(u_start, ss)
                u_end = arith.maxsi(u_end, ee)
        tile_start = (u_start // fx.Int32(BKV)) * fx.Int32(BKV)
        # Collapse an empty union window to zero width.
        tile_end = arith.maxsi(u_end, tile_start)

        # KV-column split across grid.y
        # Each (grid.x, grid.y) block owns a disjoint vertical slice of the output logits [seq_len_q, seq_len_kv]:
        # grid.x cuts query rows (horizontal), grid.y cuts KV positions (vertical).
        # The relevant KV slice to be loaded is KV[tile_start:tile_end, :].
        block_y = fx.block_idx.y

        # How many tiles of BKV columns are in the union window.
        win_tiles = arith.ceildivui(
            arith.subi(tile_end, tile_start), _to_raw(fx.Int32(BKV))
        )

        # How many KV columns (bytes/positions, rounded up to full tiles) each grid.y split owns.
        split_cols = arith.muli(
            arith.ceildivui(win_tiles, _to_raw(num_splits)),
            _to_raw(fx.Int32(BKV)),
        )

        # Each grid.y block (block_y) shifts its start forward by block_y * split_cols:
        tile_start = tile_start + block_y * split_cols
        tile_end = arith.minsi(arith.addi(tile_start, split_cols), tile_end)

        n_tiles = arith.ceildivui(
            arith.maxsi(arith.subi(tile_end, tile_start), _to_raw(fx.Int32(0))),
            _to_raw(fx.Int32(BKV)),
        )

        # ---- Prologue: prefetch tiles 0..PREFETCH_DEPTH-1 into buffers ----
        for _p in range_constexpr(PREFETCH_DEPTH):
            _dma_kv_tile_to_lds(
                fx.Int32((_p % NUM_BUFFERS) * SLOT_BYTES),
                fx.Int32(arith.addi(tile_start, _to_raw(fx.Int32(_p * BKV)))),
            )

        # ---- Steady-state software pipeline over BKV tiles ----
        lo = _to_raw(fx.Index(0))
        hi = _to_raw(fx.Index(n_tiles))
        step = _to_raw(fx.Index(1))
        tile_loop = scf.ForOp(lo, hi, step, [])
        with ir.InsertionPoint(tile_loop.body):
            t = fx.Int32(arith.index_cast(T.i32, tile_loop.induction_variable))
            col0 = tile_start + t * fx.Int32(BKV)
            slot_idx = t % fx.Int32(NUM_BUFFERS)
            slot_dword = slot_idx * fx.Int32(SLOT_I32)

            # Wait until only the (PREFETCH_DEPTH-1) newer tiles remain in flight,
            # i.e. the current tile is complete; then sync so every wave sees the
            # full LDS tile. Must be the keyword form: the positional argument is
            # a raw gfx9 bitfield in which vmcnt is split across bits [3:0] and
            # [15:14], so passing the count directly silently degrades to
            # vmcnt(0) (plus a stray expcnt) once it reaches 16 -- which is
            # exactly what the bkv256 variants hit, disabling their pipeline.
            rocdl.s_waitcnt(vmcnt=_WAIT_VMCNT)
            gpu.barrier()

            # Read all B-frags for this tile from LDS into registers. Hoisting
            # every (ni,kk) read ahead of the compute nest lets the compiler
            # batch the LDS loads and hide their lgkmcnt latency behind the MFMA work.
            b_packs = [[None] * K_STEPS for _ in range_constexpr(N_TILES)]
            cols = [None] * N_TILES
            kv_scales_tile = [None] * N_TILES
            for ni in range_constexpr(N_TILES):
                col = arith.addi(
                    arith.addi(col0, _to_raw(fx.Int32(ni * mfma.MFMA_N))),
                    _to_raw(lane_mod_N),
                )
                cols[ni] = fx.Int32(col)
                col_cl = fx.Int32(
                    arith.minsi(col, _to_raw(fx.Int32(seq_len_kv) - fx.Int32(1)))
                )
                kv_scales_tile[ni] = _to_raw(fx.Float32(sc_t[col_cl]))
                col_local = arith.addi(
                    _to_raw(fx.Int32(ni * mfma.MFMA_N)), _to_raw(lane_mod_N)
                )
                for kk in range_constexpr(K_STEPS):
                    if const_expr(swizzle):
                        # phys_dword = n*DW_PER_COL
                        #            + ((c_bytes/4) XOR ((n & (NC-1)) * CHUNK_DW))
                        c_dword = arith.divui(
                            arith.addi(
                                _to_raw(fx.Int32(kk * mfma.MFMA_K)),
                                _to_raw(lane_frag_off),
                            ),
                            _to_raw(fx.Int32(4)),
                        )
                        _mask_dw = arith.muli(
                            arith.andi(col_local, _to_raw(fx.Int32(NC - 1))),
                            _to_raw(fx.Int32(CHUNK_DW)),
                        )
                        swz_c = arith.xori(c_dword, _mask_dw)
                        frag_dword = arith.addi(
                            arith.muli(col_local, _to_raw(fx.Int32(DW_PER_COL))),
                            swz_c,
                        )
                    else:
                        frag_byte = arith.addi(
                            arith.addi(
                                arith.muli(col_local, _to_raw(fx.Int32(D))),
                                _to_raw(fx.Int32(kk * mfma.MFMA_K)),
                            ),
                            _to_raw(lane_frag_off),
                        )
                        frag_dword = arith.divui(frag_byte, _to_raw(fx.Int32(4)))

                    read_dword = arith.index_cast(
                        T.index, arith.addi(_to_raw(slot_dword), frag_dword)
                    )
                    b_packs[ni][kk] = lds_st.vec_load((read_dword,), vec_size=8)

            # Prefetch tile i+PREFETCH_DEPTH into slot (i+PD)%NB.  When NB>PD that
            # slot != the just-read slot, so no reader-before-writer barrier is
            # needed (the slot's last reader was iteration i-(NB-PD), already
            # past this iteration's barrier).  NB==PD reuses the read slot and
            # requires barrier B first.
            if const_expr(_need_barrier_b):
                gpu.barrier()
            t_next = arith.addi(_to_raw(t), _to_raw(fx.Int32(PREFETCH_DEPTH)))
            next_slot_byte = arith.muli(
                arith.remui(t_next, _to_raw(fx.Int32(NUM_BUFFERS))),
                _to_raw(fx.Int32(SLOT_BYTES)),
            )
            col0_next = arith.addi(
                tile_start,
                arith.muli(t_next, _to_raw(fx.Int32(BKV))),
            )
            _dma_kv_tile_to_lds(fx.Int32(next_slot_byte), fx.Int32(col0_next))

            # ---- Per-row MFMA + epilogue (this wave's RPW rows, all columns) ----
            for j in range_constexpr(RPW):
                row = _i32_add(wave_row0, fx.Int32(j))
                out_row_t = _make_out_row_t(logits, _stride_i64, row)
                for ni in range_constexpr(N_TILES):
                    col = cols[ni]
                    kv_scale = kv_scales_tile[ni]
                    col_sum = _to_raw(f32_0)
                    for mi in range_constexpr(M_TILES):
                        acc = Vec.filled(mfma.ACC_ELEMS, 0.0, fx.Float32)
                        for kk in range_constexpr(K_STEPS):
                            acc = mfma_fn(
                                _mfma_res_ty,
                                mfma.make_operands(
                                    a_packs[j][mi][kk], b_packs[ni][kk], acc
                                ),
                            )
                        for ii in range_constexpr(mfma.ACC_ELEMS):
                            score = Vec(acc)[ii].ir_value()
                            relu = arith.maximumf(score, _to_raw(f32_0))
                            wsc = arith.MulFOp(
                                relu, w_frag[j][mi][ii], fastmath=fm_fast
                            ).result
                            col_sum = arith.AddFOp(
                                col_sum, wsc, fastmath=fm_fast
                            ).result
                    col_sum = arith.MulFOp(col_sum, kv_scale, fastmath=fm_fast).result

                    for sh in mfma.shuffle_offsets:
                        peer = _to_raw(ArithValue(col_sum).shuffle_xor(sh, 64))
                        col_sum = arith.AddFOp(col_sum, peer, fastmath=fm_fast).result

                    in_window = arith.andi(
                        _to_raw(
                            arith.cmpi(
                                arith.CmpIPredicate.sge,
                                _to_raw(col),
                                _to_raw(starts[j]),
                            )
                        ),
                        _to_raw(
                            arith.cmpi(
                                arith.CmpIPredicate.slt,
                                _to_raw(col),
                                _to_raw(ends[j]),
                            )
                        ),
                    )
                    is_writer = arith.andi(
                        _to_raw(
                            arith.cmpi(
                                arith.CmpIPredicate.eq,
                                _to_raw(lane_div_N),
                                _to_raw(fx.Int32(0)),
                            )
                        ),
                        in_window,
                    )
                    with ir.InsertionPoint(scf.IfOp(is_writer).then_block):
                        out_row_t[col] = fx.Float32(col_sum)
                        scf.YieldOp([])

            scf.YieldOp([])

        # ---- Fused clean_logits prefill: per-wave, over this wave's own rows.
        # A wave holds starts[]/ends[] only for its RPW rows; making all waves
        # cooperate would need extra cu_starts/cu_ends loads for no gain.
        # Emitting this after the tile loop is mandatory -- the loop's
        # s_waitcnt(vmcnt=_WAIT_VMCNT) counts vector stores too on gfx9, so a
        # fill store in flight inside it would let a half-written LDS tile
        # through. ----
        if const_expr(clean_logits):
            _emit_row_neg_inf_fill(
                logits=logits,
                stride_i64=_stride_i64,
                rows=[_i32_add(wave_row0, fx.Int32(j)) for j in range_constexpr(RPW)],
                starts=starts,
                ends=ends,
                seq_len_kv=seq_len_kv,
                tid_i32=lane,  # the 64 lanes of THIS wave
                nthreads=64,
                by_i32=block_y,
                num_splits=num_splits,
            )

    @flyc.jit
    def launch_fp8_mqa_logits_mfma_lds_pipe(
        Q: fx.Tensor,
        KV: fx.Tensor,
        kv_scales: fx.Tensor,
        weights: fx.Tensor,
        cu_starts: fx.Tensor,
        cu_ends: fx.Tensor,
        logits: fx.Tensor,
        seq_len: fx.Int32,
        seq_len_kv: fx.Int32,
        stride_logits_s: fx.Int32,
        num_splits: fx.Int32,
        stream: fx.Stream,
    ):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()

        n_blocks = arith.ceildivui(_to_raw(seq_len), _to_raw(fx.Int32(ROWS_PER_BLOCK)))
        gx = arith.index_cast(T.index, n_blocks)
        gy = arith.index_cast(T.index, _to_raw(num_splits))
        kernel._func.__name__ = _kname
        kernel(
            Q,
            KV,
            kv_scales,
            weights,
            cu_starts,
            cu_ends,
            logits,
            seq_len,
            seq_len_kv,
            stride_logits_s,
            num_splits,
        ).launch(grid=(gx, gy, 1), block=(MR_BLOCK_THREADS, 1, 1), stream=stream)

    return launch_fp8_mqa_logits_mfma_lds_pipe


# --------------------------------------------------------------------------- #
# Kernel-variant registry (arch-dependent).
#
# gfx942 keeps its original ``"mfma_r<RPB>_w<WPB>"`` tags unchanged: RPB query
# rows per block, WPB waves per block, block_kv fixed at _BLOCK_KV.
#
# gfx950 variants carry the MFMA shape and block_kv in the tag, because there
# the atom and tile width both vary:
#     "mfma<MxNxK>_bkv<B>_r<RPB>_w<WPB>[_lds<NUM_BUFFERS>]"
# The ``_lds`` suffix selects the LDS-pipelined builder, in which all WPB waves
# share one staged KV tile and partition rows, so a block owns RPB*WPB rows.
#
# Each entry hardcodes its own block_kv, overriding whatever the caller passed
# to ``compile_fp8_mqa_logits``.
# --------------------------------------------------------------------------- #


def _mk_builder(rpb, wpb, *, mfma=_MFMA16, bkv=None, lds=None, swizzle=True):
    """Registry entry factory.

    ``lds`` is None for the direct-load builder, else the LDS buffer count
    (prefetch depth is fixed at 2, the only configuration in use).
    """
    extra = {} if bkv is None else {"block_kv": bkv}
    if lds is None:
        return lambda **kw: _build_kernel_mfma_r_w(
            **{**kw, **extra}, rows_per_block=rpb, waves_per_block=wpb, mfma=mfma
        )
    return lambda **kw: _build_kernel_mfma_lds_pipe(
        **{**kw, **extra},
        rows_per_block=rpb,
        waves_per_block=wpb,
        mfma=mfma,
        swizzle=swizzle,
        num_buffers=lds,
        prefetch_depth=2,
    )


_VARIANT_BUILDERS = {}

if _ARCH == "gfx942":
    _VARIANT_BUILDERS.update(
        {f"mfma_r{r}_w{w}": _mk_builder(r, w) for r in (1, 2, 4) for w in (1, 2, 4)}
    )

if _ARCH == "gfx950":
    # CDNA4 scaled MFMA atoms (K=128/64): gfx950-only, since those instructions
    # require native FN operands and do not exist on gfx942.
    _K64 = _MFMA32_K64
    _K128 = _MFMA16_K128
    _VARIANT_BUILDERS.update(
        {
            # -- direct load: every wave fetches its own KV tile, no LDS --
            "mfma16x16x128_bkv128_r1_w1": _mk_builder(1, 1, mfma=_K128, bkv=128),
            "mfma16x16x128_bkv128_r2_w1": _mk_builder(2, 1, mfma=_K128, bkv=128),
            "mfma16x16x128_bkv128_r1_w2": _mk_builder(1, 2, mfma=_K128, bkv=128),
            "mfma16x16x128_bkv128_r2_w2": _mk_builder(2, 2, mfma=_K128, bkv=128),
            "mfma32x32x64_bkv128_r1_w1": _mk_builder(1, 1, mfma=_K64, bkv=128),
            "mfma32x32x64_bkv128_r2_w1": _mk_builder(2, 1, mfma=_K64, bkv=128),
            "mfma32x32x64_bkv128_r1_w2": _mk_builder(1, 2, mfma=_K64, bkv=128),
            "mfma32x32x64_bkv128_r2_w2": _mk_builder(2, 2, mfma=_K64, bkv=128),
            # -- LDS double-buffered: WPB waves share one staged KV tile --
            "mfma32x32x64_bkv64_r1_w2_lds2": _mk_builder(
                1, 2, mfma=_K64, bkv=64, lds=2
            ),
            "mfma32x32x64_bkv64_r2_w2_lds2": _mk_builder(
                2, 2, mfma=_K64, bkv=64, lds=2
            ),
            "mfma32x32x64_bkv64_r2_w4_lds2": _mk_builder(
                2, 4, mfma=_K64, bkv=64, lds=2
            ),
            "mfma32x32x64_bkv128_r1_w2_lds2": _mk_builder(
                1, 2, mfma=_K64, bkv=128, lds=2
            ),
            "mfma32x32x64_bkv128_r2_w2_lds2": _mk_builder(
                2, 2, mfma=_K64, bkv=128, lds=2
            ),
            "mfma32x32x64_bkv128_r2_w4_lds2": _mk_builder(
                2, 4, mfma=_K64, bkv=128, lds=2
            ),
            "mfma32x32x64_bkv256_r1_w2_lds2": _mk_builder(
                1, 2, mfma=_K64, bkv=256, lds=2
            ),
            "mfma32x32x64_bkv256_r2_w2_lds2": _mk_builder(
                2, 2, mfma=_K64, bkv=256, lds=2
            ),
            "mfma16x16x128_bkv64_r2_w2_lds2": _mk_builder(
                2, 2, mfma=_K128, bkv=64, lds=2
            ),
            "mfma16x16x128_bkv128_r1_w2_lds2": _mk_builder(
                1, 2, mfma=_K128, bkv=128, lds=2
            ),
            "mfma16x16x128_bkv128_r2_w2_lds2": _mk_builder(
                2, 2, mfma=_K128, bkv=128, lds=2
            ),
            "mfma16x16x128_bkv128_r2_w4_lds2": _mk_builder(
                2, 4, mfma=_K128, bkv=128, lds=2
            ),
            "mfma16x16x128_bkv256_r2_w2_lds2": _mk_builder(
                2, 2, mfma=_K128, bkv=256, lds=2
            ),
            # -- LDS triple-buffered: same in-flight depth as _lds2 but the
            #    reader/writer barrier is elided (num_buffers > prefetch_depth) --
            "mfma32x32x64_bkv64_r1_w2_lds3": _mk_builder(
                1, 2, mfma=_K64, bkv=64, lds=3
            ),
            "mfma32x32x64_bkv64_r2_w2_lds3": _mk_builder(
                2, 2, mfma=_K64, bkv=64, lds=3
            ),
            "mfma32x32x64_bkv64_r2_w4_lds3": _mk_builder(
                2, 4, mfma=_K64, bkv=64, lds=3
            ),
            "mfma32x32x64_bkv128_r1_w2_lds3": _mk_builder(
                1, 2, mfma=_K64, bkv=128, lds=3
            ),
            "mfma32x32x64_bkv128_r2_w4_lds3": _mk_builder(
                2, 4, mfma=_K64, bkv=128, lds=3
            ),
        }
    )

KERNEL_VARIANTS = tuple(_VARIANT_BUILDERS.keys())
DEFAULT_VARIANT = "mfma_r2_w4" if _ARCH == "gfx942" else "mfma32x32x64_bkv64_r1_w2_lds3"

# Parses both tag schemes; group 1 is the shape (None for the gfx942 tags),
# then block_kv (None -> _BLOCK_KV), RPB, WPB, and the LDS buffer count.
_TAG_RE = re.compile(
    r"^mfma(?P<shape>\d+x\d+x\d+)?(?:_bkv(?P<bkv>\d+))?"
    r"_r(?P<rpb>\d+)_w(?P<wpb>\d+)(?:_lds(?P<lds>\d+))?$"
)


def _parse_variant(tag):
    """(block_kv, rows_per_block_effective) for host-side padding and splitting.

    For ``_lds`` variants the WPB waves partition rows within one shared KV
    tile, so a block owns RPB*WPB rows and seq_len must be padded to that.
    """
    m = _TAG_RE.match(tag)
    if m is None:
        return _BLOCK_KV, 1
    bkv = int(m.group("bkv")) if m.group("bkv") else _BLOCK_KV
    rpb, wpb = int(m.group("rpb")), int(m.group("wpb"))
    return bkv, (rpb * wpb if m.group("lds") else rpb)


def _auto_variant(seq_len, seq_len_kv, num_heads):
    """Pick a variant from the problem shape.

    gfx942 (unchanged): RPB=2 always; WPB=2 packs more column tiles per wave
    when M and N are both large, else WPB=4 for more wavefronts on small-M /
    short-window shapes.

    gfx950: the LDS 3-buffer 32x32x64 atom at bkv=64, WPB=2. Rows-per-wave r
    sets the KV reuse per block (reuse = r*WPB). r=1 gives more blocks and
    better utilisation on small/square shapes; r=2 amortises the barrier cost
    when KV streaming pressure is high (long KV) or the problem is large and
    square. At H>=128 the compute-to-bandwidth ratio is already 2x that of
    H64, so r=1 always suffices.
    """
    if _ARCH == "gfx942":
        wpb = 2 if (seq_len >= 2048 and seq_len_kv >= 8192) else 4
        return f"mfma_r2_w{wpb}"
    if _ARCH == "gfx950":
        if num_heads >= 128:
            return "mfma32x32x64_bkv64_r1_w2_lds3"
        streaming = seq_len_kv > 2 * seq_len
        large_square = seq_len >= 8192 and seq_len_kv >= seq_len
        return f"mfma32x32x64_bkv64_r{2 if streaming or large_square else 1}_w2_lds3"
    raise NotImplementedError(
        f"fp8_mqa_logits has no FlyDSL variants for arch {_ARCH!r}; "
        "supported: gfx942, gfx950"
    )


def _resolve_variant(variant, seq_len, seq_len_kv, num_heads):
    """Effective variant: explicit ``variant=`` > env var > shape-adaptive."""
    tag = (
        variant
        or os.environ.get("FLYDSL_FP8_MQA_LOGITS_VARIANT")
        or _auto_variant(seq_len, seq_len_kv, num_heads)
    )
    if tag not in _VARIANT_BUILDERS:
        raise ValueError(
            f"unknown fp8_mqa_logits variant {tag!r} for arch {_ARCH}; "
            f"available: {list(KERNEL_VARIANTS)}"
        )
    return tag


@lru_cache(maxsize=32)
def compile_fp8_mqa_logits(
    *,
    num_heads: int,
    head_size: int,
    block_kv: int = _BLOCK_KV,
    paged: bool = False,
    variant: str = DEFAULT_VARIANT,
    convert_q_fn: bool = False,
    convert_kv_fn: bool = False,
    clean_logits: bool = True,
):
    """Return a cached, compiled FlyDSL launcher for the given shape config.

    ``num_heads``/``head_size`` are compile-time constants (powers of two, D in
    {64, 128}); ``variant`` is an ``mfma_r<RPB>_w<WPB>`` tag (see
    ``KERNEL_VARIANTS``); ``convert_q_fn``/``convert_kv_fn`` mark an FP8 FN
    operand whose -0 (0x80) byte the kernel patches to FNUZ +0.
    ``clean_logits`` selects whether the kernel also writes -inf to the
    out-of-window positions; like the convert flags it is a compile-time
    specialization, so the False kernel carries none of that code. ``paged`` is
    reserved for a future variant and must be False.
    """
    if paged:
        raise NotImplementedError(
            "Paged FlyDSL fp8_mqa_logits is Phase 2 and not implemented yet."
        )
    if variant not in _VARIANT_BUILDERS:
        raise ValueError(
            f"unknown fp8_mqa_logits variant {variant!r}; "
            f"available: {list(KERNEL_VARIANTS)}"
        )
    launcher = _VARIANT_BUILDERS[variant](
        num_heads=num_heads,
        head_size=head_size,
        block_kv=block_kv,
        convert_q_fn=convert_q_fn,
        convert_kv_fn=convert_kv_fn,
        clean_logits=clean_logits,
    )
    launcher.compile_hints = dict(_DEFAULT_COMPILE_HINTS)
    return launcher


def flydsl_fp8_mqa_logits(
    Q,
    KV,
    kv_scales,
    weights,
    cu_starts,
    cu_ends,
    clean_logits=True,
    stream=None,
    variant=None,
):
    """FlyDSL gfx942/gfx950 FP8 MQA logits -- drop-in replacement for the Triton ``fp8_mqa_logits``.

    Q:            [seq_len, NUM_HEADS, HEAD_SIZE], dtype float8
    KV:           [seq_len_kv, HEAD_SIZE], dtype float8
    kv_scales:    [seq_len_kv], dtype float32
    weights:      [seq_len, NUM_HEADS], dtype float32
    cu_starts:    [seq_len], dtype int32, per-row window start (inclusive)
    cu_ends:      [seq_len], dtype int32, per-row window end (exclusive)
    clean_logits: bool. If True, positions outside [cu_starts[i], cu_ends[i])
                  in row i are written as -inf -- by the kernel itself, as part
                  of the same launch; the output is never pre-filled. If False,
                  the kernel skips those positions and the caller owns whatever
                  is left there.
    stream:       optional HIP stream; defaults to the current stream.
    variant:      optional kernel-variant tag (see ``KERNEL_VARIANTS``). If None,
                  taken from ``FLYDSL_FP8_MQA_LOGITS_VARIANT`` or, failing that,
                  chosen adaptively from the problem shape (``_auto_variant``).

    Returns
    -------
    logits: [seq_len, seq_len_kv], dtype float32.
    """
    seq_len, num_heads, head_size = Q.shape
    seq_len_kv = KV.shape[0]
    assert num_heads & (num_heads - 1) == 0, "num q. heads should be power of 2."
    assert head_size & (head_size - 1) == 0, "head size should be power of 2."

    # FlyDSL's DLPack tensor adaptor rejects 0-dim tensors, but the per-token
    # ``kv_scales`` collapses to a scalar when seq_len_kv == 1 (and ``weights``
    # could too). Reshape the 1-D / 2-D inputs back to their logical rank so the
    # kernel always sees indexable tensors (matches the Triton pointer path).
    kv_scales = kv_scales.reshape(seq_len_kv)
    weights = weights.reshape(seq_len, num_heads)
    cu_starts = cu_starts.reshape(seq_len)
    cu_ends = cu_ends.reshape(seq_len)

    # The gfx942 fp8 MFMA reads operands as e4m3 FNUZ (bias 8). For an e4m3 FN
    # operand (OCP, bias 7) the same byte encodes exactly 2x the FNUZ value (the
    # only data byte that differs is FN -0 = 0x80, which is FNUZ NaN), so we pass
    # the raw bytes through, let the kernel patch 0x80 -> +0, and undo the 2x per
    # FN operand by scaling kv_scales -- ReLU is positive-homogeneous, so
    # logits = sum_h ReLU(QK*scale)*w is preserved.
    _fnuz = torch.float8_e4m3fnuz
    _fn = torch.float8_e4m3fn
    assert Q.dtype in (_fnuz, _fn) and KV.dtype in (
        _fnuz,
        _fn,
    ), f"Q/KV must be e4m3 fp8 (fnuz or fn); got {Q.dtype}, {KV.dtype}"
    # Only gfx942 needs that conversion; other fp8 archs read operands in their
    # native dtype, so the FN->FNUZ recast there would corrupt them.
    convert_q_fn = get_gfx() == "gfx942" and Q.dtype != _fnuz
    convert_kv_fn = get_gfx() == "gfx942" and KV.dtype != _fnuz
    scale_mul = (2.0 if convert_q_fn else 1.0) * (2.0 if convert_kv_fn else 1.0)
    if scale_mul != 1.0:
        kv_scales = kv_scales.to(torch.float32) * scale_mul

    variant = _resolve_variant(variant, seq_len, seq_len_kv, num_heads)

    _BKV, _ROWS_PER_BLOCK = _parse_variant(variant)

    launcher = compile_fp8_mqa_logits(
        num_heads=num_heads,
        head_size=head_size,
        block_kv=_BKV,
        paged=False,
        variant=variant,
        convert_q_fn=convert_q_fn,
        convert_kv_fn=convert_kv_fn,
        clean_logits=bool(clean_logits),
    )

    # The kernels require seq_len padded to a multiple of the rows a block owns,
    # so every block owns exactly that many. Padded rows get empty windows
    # (start == end == 0) so the kernel writes nothing for them; the output is
    # sliced back to the original seq_len after the launch.
    seq_len_padded = (
        (seq_len + _ROWS_PER_BLOCK - 1) // _ROWS_PER_BLOCK
    ) * _ROWS_PER_BLOCK
    if seq_len_padded != seq_len:
        pad = seq_len_padded - seq_len
        Q = torch.cat([Q, Q.new_zeros((pad, num_heads, head_size))], dim=0)
        weights = torch.cat([weights, weights.new_zeros((pad, num_heads))], dim=0)
        cu_starts = torch.cat([cu_starts, cu_starts.new_zeros(pad)], dim=0)
        cu_ends = torch.cat([cu_ends, cu_ends.new_zeros(pad)], dim=0)

    # Column padding matches the Triton launcher, so the two produce
    # identically-shaped, identically-strided outputs. It also keeps every row
    # base 1 KiB-aligned (the stride is a multiple of 256 f32), which the
    # per-row stores want. The kernel writes the output through a per-row i64
    # byte-offset view, so the row*stride*4 element offset no longer has to fit
    # in i32 (the prior ~46k-square ceiling is gone); only the per-row column
    # offset stays in i32.
    #
    # No torch.full even when clean_logits: the kernel now writes -inf itself,
    # at exactly the out-of-window positions it would otherwise skip. That drops
    # a whole extra kernel launch and about a third of the output write traffic
    # (a full-tensor prefill, half of which the epilogue immediately overwrote).
    aligned_size = 256
    seq_len_kv_aligned = (seq_len_kv + aligned_size - 1) // aligned_size * aligned_size
    logits = torch.empty(
        (seq_len_padded, seq_len_kv_aligned),
        dtype=torch.float32,
        device=Q.device,
    )[:, :seq_len_kv]

    num_splits = _auto_num_splits(
        seq_len_padded, seq_len_kv, _ROWS_PER_BLOCK, _BKV, Q.device.index
    )

    if stream is None:
        stream = torch.cuda.current_stream()

    with torch.cuda.device(Q.device.index):
        _run_compiled(
            launcher,
            Q,
            KV,
            kv_scales,
            weights,
            cu_starts,
            cu_ends,
            logits,
            int(seq_len_padded),
            int(seq_len_kv),
            int(logits.stride(0)),
            int(num_splits),
            stream,
        )

    return logits[:seq_len, :]
