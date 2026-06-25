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

# NOTE: do NOT add `from __future__ import annotations` to this file -- PEP 563
# stringizes annotations, which FlyDSL's kernel-argument typing relies on being
# real objects. (Matches the note in qk_norm_rope_quant.py.)

import os
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, range_constexpr, rocdl
from flydsl.expr.rocdl import _split_mfma_operands
from flydsl.expr.numeric import ArithValue
from flydsl.expr.typing import T
from flydsl._mlir.dialects import scf, vector as mlir_vector
from flydsl._mlir.dialects.rocdl import mfma_f32_32x32x16_fp8_fp8 as _ods_mfma32x32x16
from flydsl._mlir import ir
from flydsl.runtime.device import get_rocm_arch as get_hip_arch

from .tensor_shim import GTensor, _run_compiled, _to_raw

Vec = fx.Vector


# --------------------------------------------------------------------------- #
# MfmaAtom — bundles all MFMA-shape-derived constants and the rocdl functor.
#
# Adding a new MFMA shape only requires a new MfmaAtom instance and a new
# entry in _VARIANT_BUILDERS; the kernel builder is fully generic.
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class MfmaAtom:
    """MFMA-shape descriptor for the fp8 MQA logits kernel.

    Fields
    ------
    name : str
        Human-readable shape tag embedded in kernel names (e.g. ``"16x16x32"``).
    MFMA_M, MFMA_N, MFMA_K : int
        MFMA output/input tile dimensions (M×N output, K fp8 elements/step).
    ACC_ELEMS : int
        Number of f32 accumulator elements per lane (``vec<ACC_ELEMS x f32>``).
    fn : Callable
        FlyDSL ``rocdl.mfma_*`` functor accepting ``(result_type, operands)``.
    shuffle_offsets : tuple[int, ...]
        ``shuffle_xor`` offsets for the in-wave head-reduce butterfly.
        Must cover all lane groups so the full H-wide sum is produced.
    acc_head_static_offsets : tuple[int, ...]
        Per-element compile-time head-offset within one MFMA_M tile.
        Length == ACC_ELEMS. For element ``ii`` in lane group ``g``:
        ``head_within_tile = acc_head_static_offsets[ii] + g * acc_head_group_stride``
        The full weight index is ``mi * MFMA_M + head_within_tile``.

        Derivation: the MFMA hardware stores acc element ``ii`` for the head
        whose *row* in the A-matrix is ``acc_head_static_offsets[ii] + g * stride``.
        For 16x16x32 (4 groups, ACC_ELEMS=4): the layout is sequential,
        giving ``static_offsets = (0, 1, 2, 3)`` and ``group_stride = 4``.
        For 32x32x16 (2 groups, ACC_ELEMS=16): the layout interleaves the two
        groups in blocks of 4, giving
        ``static_offsets = (0,1,2,3,8,9,10,11,16,17,18,19,24,25,26,27)``
        and ``group_stride = 4`` (empirically verified on gfx942/CDNA3).
    acc_head_group_stride : int
        Multiplier for the lane-group index (always 4 on gfx942 fp8 MFMA).
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


def _mfma32x32x16_fp8_fp8_wrapper(result_type, operands, *, loc=None, ip=None):
    """Wrap the raw ODS ``mfma_f32_32x32x16_fp8_fp8`` to match the
    ``(result_type, operands_list)`` convention used by ``flydsl.expr.rocdl``."""
    a, b, c, cbsz, abid, blgp = _split_mfma_operands(operands, loc=loc)
    return _ods_mfma32x32x16(result_type, a, b, c, cbsz, abid, blgp,
                              loc=loc, ip=ip).result


#: 16×16 output tile, K=32 fp8 elements/step.  Acc: vec<4 x f32>.
#: Fragment layout: lane l → A[row=l%16, k=(l//16)*8+0..7], col=l%16.
#: Writer: lane//16 == 0 (16 distinct output columns per tile).
#: Acc layout (empirically verified on gfx942): acc[ii] at group g -> head g*4+ii.
_MFMA16 = MfmaAtom(
    name="16x16x32",
    MFMA_M=16, MFMA_N=16, MFMA_K=32,
    ACC_ELEMS=4,
    fn=rocdl.mfma_f32_16x16x32_fp8_fp8,
    shuffle_offsets=(16, 32),
    acc_head_static_offsets=(0, 1, 2, 3),
    acc_head_group_stride=4,
)

#: 32×32 output tile, K=16 fp8 elements/step.  Acc: vec<16 x f32>.
#: Fragment layout: lane l → A[row=l%32, k=(l//32)*8+0..7], col=l%32.
#: Writer: lane//32 == 0 (32 distinct output columns per tile).
#: Acc layout (empirically verified on gfx942): acc[ii] at group g ->
#:   head (ii//4)*8 + g*4 + ii%4.  static_offsets = ii%4 + (ii//4)*8.
_MFMA32 = MfmaAtom(
    name="32x32x16",
    MFMA_M=32, MFMA_N=32, MFMA_K=16,
    ACC_ELEMS=16,
    fn=_mfma32x32x16_fp8_fp8_wrapper,
    shuffle_offsets=(32,),
    acc_head_static_offsets=(
        0, 1, 2, 3,    # ii=0..3:  head = g*4 + 0..3
        8, 9, 10, 11,  # ii=4..7:  head = g*4 + 8..11
        16, 17, 18, 19,# ii=8..11: head = g*4 + 16..19
        24, 25, 26, 27,# ii=12..15:head = g*4 + 24..27
    ),
    acc_head_group_stride=4,
)


def _i32_add(a, b):
    """Add two fx.Int32 scalars -> fx.Int32 (i32 arithmetic)."""
    return fx.Int32(arith.addi(_to_raw(a), _to_raw(b)))


def _fp8_byte_to_f32(byte_scalar):
    """Convert a raw i8 fp8-byte load to f32 via ``v_cvt_f32_fp8``.

    ``rocdl.raw.ptr.buffer.load`` cannot return an fp8 scalar (the result type
    must be LLVM-compatible) and ``arith.extf`` does not lower for fp8, so fp8
    tensors are loaded as i8 bytes, zero-extended into the low byte of an i32,
    and converted with the native gfx942 fp8->f32 instruction (byteSel=0). On
    CDNA3 this instruction uses the E4M3 *FNUZ* interpretation, matching the
    host fp8 cast.
    """
    src_i32 = _to_raw(ArithValue(_to_raw(byte_scalar)).extui(T.i32))
    return rocdl.cvt_f32_fp8(T.f32, src_i32, 0)


# 1 wave64 per query row. Each lane strides over the KV window.
BLOCK_THREADS = 64

# Vector width for the head-dim (D) inner load (fp8 elements per chunk).
# D in {64, 128} are both multiples of 8, so the dot over D is a whole
# number of VEC-wide chunks.
VEC_D = 8

_DEFAULT_COMPILE_HINTS = {
    "waves_per_eu": 2,
    "fast_fp_math": True,
}


def _fp8_dtype():
    """fp8 element type for the current arch (matches preshuffle_gemm).

    gfx942 / CDNA3 ships e4m3 *FNUZ*; gfx950 / CDNA4 and gfx12 ship OCP
    e4m3 *FN*. The two are NOT bit-compatible, so the kernel must read the
    fp8 bytes with the same format the host cast them to.
    """
    arch = str(get_hip_arch())
    if arch == "gfx942":
        return fx.Float8E4M3FNUZ
    return fx.Float8E4M3FN


def _build_kernel(*, num_heads: int, head_size: int, block_kv: int):
    """Build the @flyc.kernel + @flyc.jit launcher for one shape config.

    Correctness-first design (no MFMA yet): grid ``(seq_len,)``, one wave64
    per query row. Lane ``t`` owns KV columns ``start + t, +64, +128, ...``
    across the row's window and, for each, computes the full
    ``sum_h ReLU(dot * kv_scale) * weight`` reduction with a vectorized
    inner dot over the head dim.
    """
    H = num_heads
    D = head_size
    assert D % VEC_D == 0, f"head_size={D} must be a multiple of VEC_D={VEC_D}"
    ND = D // VEC_D  # number of VEC_D-wide chunks along the head dim

    fp8_dt = _fp8_dtype()
    fm_fast = arith.FastMathFlags.fast

    _kname = f"fp8_mqa_logits_H{H}_D{D}_bkv{block_kv}_flydsl"

    @flyc.kernel(name=_kname)
    def kernel(
        Q: fx.Tensor,  # [seq_len, H, D]      fp8 (any variant; bytes passed raw)
        KV: fx.Tensor,  # [seq_len_kv, D]      fp8 (any variant; bytes passed raw)
        kv_scales: fx.Tensor,  # [seq_len_kv]         f32
        weights: fx.Tensor,  # [seq_len, H]         f32
        cu_starts: fx.Tensor,  # [seq_len]            i32
        cu_ends: fx.Tensor,  # [seq_len]            i32
        logits: fx.Tensor,  # [seq_len, seq_len_kv] f32
        seq_len: fx.Int32,
        seq_len_kv: fx.Int32,
        stride_logits_s: fx.Int32,  # logits row stride (f32 elements)
    ):
        f32_0 = arith.constant(0.0, type=T.f32)
        vec_f32_t = T.vec(VEC_D, T.f32)

        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        # Process rows from last to first to even out the triangular window
        # work across waves (matches the Triton kernel's reverse ordering).
        row = fx.Int32(seq_len) - bid - fx.Int32(1)

        # fp8 tensors are read as raw i8 bytes (buffer_load can't return an fp8
        # scalar) and reinterpreted to fp8 -> f32 in registers.
        q_t = GTensor(Q, dtype=T.i8, shape=(-1, H, D))
        kv_t = GTensor(KV, dtype=T.i8, shape=(-1, D))
        sc_t = GTensor(kv_scales, dtype=T.f32, shape=(-1,))
        w_t = GTensor(weights, dtype=T.f32, shape=(-1, H))
        cs_t = GTensor(cu_starts, dtype=T.i32, shape=(-1,))
        ce_t = GTensor(cu_ends, dtype=T.i32, shape=(-1,))
        # logits row stride is runtime (256-aligned, then sliced), so pass it
        # explicitly rather than deriving it from a runtime shape (the
        # TensorView stride helper can't cumprod a runtime dim).
        out_t = GTensor(
            logits,
            dtype=T.f32,
            shape=(-1, -1),
            stride=(stride_logits_s, fx.Int32(1)),
        )

        # Window for this row, clamped to [0, seq_len_kv).
        start = fx.Int32(cs_t[row])
        end = fx.Int32(ce_t[row])
        start = arith.maxsi(_to_raw(start), _to_raw(fx.Int32(0)))
        end = arith.minsi(_to_raw(end), _to_raw(fx.Int32(seq_len_kv)))
        start = fx.Int32(start)
        end = fx.Int32(end)

        # Runtime loop bounds (Index type) reused by the inner scf.for loops.
        h_lo = _to_raw(fx.Index(fx.Int32(0)))
        h_hi = _to_raw(fx.Index(fx.Int32(H)))
        h_step = _to_raw(fx.Index(fx.Int32(1)))
        d_lo = _to_raw(fx.Index(fx.Int32(0)))
        d_hi = _to_raw(fx.Index(fx.Int32(D)))
        d_step = _to_raw(fx.Index(fx.Int32(1)))

        # Lane t handles columns start + t, start + t + 64, ...  The h/d
        # reductions are *runtime* scf.for loops (not compile-time unrolled):
        # with H=64, D=128 the fully-unrolled body is ~8k ops per column and
        # makes MLIR canonicalization pathologically slow, so keep it small and
        # let the loop carry the f32 accumulators. fp8 bytes are converted to
        # f32 on the fly with v_cvt_f32_fp8 (no fp8 vector types).
        col0 = _i32_add(start, tid)
        n_iter = scf.ForOp(
            _to_raw(fx.Index(col0)),
            _to_raw(fx.Index(end)),
            _to_raw(fx.Index(fx.Int32(BLOCK_THREADS))),
            [],
        )
        with ir.InsertionPoint(n_iter.body):
            n_idx = n_iter.induction_variable
            n = fx.Int32(arith.index_cast(T.i32, n_idx))
            kv_scale = _to_raw(fx.Float32(sc_t[n]))

            # acc = sum_h ReLU(<Q[row,h], K[n]> * kv_scale) * weights[row,h]
            h_loop = scf.ForOp(h_lo, h_hi, h_step, [_to_raw(f32_0)])
            with ir.InsertionPoint(h_loop.body):
                h = fx.Int32(arith.index_cast(T.i32, h_loop.induction_variable))
                acc_in = h_loop.inner_iter_args[0]

                # dot = <Q[row,h,:], K[n,:]> over the head dim.
                d_loop = scf.ForOp(d_lo, d_hi, d_step, [_to_raw(f32_0)])
                with ir.InsertionPoint(d_loop.body):
                    d = fx.Int32(
                        arith.index_cast(T.i32, d_loop.induction_variable)
                    )
                    dot_in = d_loop.inner_iter_args[0]
                    qf = _fp8_byte_to_f32(q_t[row, h, d])
                    kf = _fp8_byte_to_f32(kv_t[n, d])
                    prod = arith.MulFOp(qf, kf, fastmath=fm_fast).result
                    dot_out = arith.AddFOp(dot_in, prod, fastmath=fm_fast).result
                    scf.YieldOp([dot_out])
                dot = d_loop.results[0]

                # ReLU(dot * kv_scale) * weights[row,h], accumulated over heads.
                w_h = _to_raw(fx.Float32(w_t[row, h]))
                scaled = arith.MulFOp(dot, kv_scale, fastmath=fm_fast).result
                relu = arith.maximumf(scaled, _to_raw(f32_0))
                wscaled = arith.MulFOp(relu, w_h, fastmath=fm_fast).result
                acc_out = arith.AddFOp(acc_in, wscaled, fastmath=fm_fast).result
                scf.YieldOp([acc_out])

            out_t[row, n] = fx.Float32(h_loop.results[0])
            scf.YieldOp([])

    @flyc.jit
    def launch_fp8_mqa_logits(
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
        stream: fx.Stream = fx.Stream(None),
    ):
        gx = arith.index_cast(T.index, _to_raw(seq_len))
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
        ).launch(grid=(gx, 1, 1), block=(BLOCK_THREADS, 1, 1), stream=stream)

    return launch_fp8_mqa_logits


def _build_kernel_mfma_r_w(*, num_heads: int, head_size: int, block_kv: int,
                           rows_per_block: int, waves_per_block: int,
                           mfma: MfmaAtom,
                           convert_q_fn: bool = False, convert_kv_fn: bool = False):
    """Multi-row, multi-wave MFMA kernel (generic over MFMA shape via MfmaAtom).

    Parameters
    ----------
    rows_per_block : int
        Query rows sharing one KV tile load; cuts KV global traffic by RPB.
    waves_per_block : int
        Waves per block; each wave owns a disjoint ``N_TILES // WPB`` column-tile
        slice, so all WPB waves run in parallel with no cross-wave LDS or barrier.
    mfma : MfmaAtom
        MFMA shape descriptor (dimensions, accumulator size, rocdl functor,
        head-reduce butterfly offsets).

    Thread decomposition
    --------------------
    * ``tid = wave * 64 + lane``  (tid: 0..MR_BLOCK_THREADS-1)
    * ``lane_div_N = lane // MFMA_N`` — k-group index; selects the K chunk each
      lane reads in the A/B frags.
    * ``lane_mod_N = lane % MFMA_N`` — column index within the MFMA_N-wide tile.
    * Wave ``w`` owns n-tiles ``[w*N_TILES_PER_WAVE, (w+1)*N_TILES_PER_WAVE)``
      within each BKV tile. A-frag layout and head-reduce use ``lane``, not ``tid``.

    Grid: ``(ceil(seq_len / RPB), 1, 1)``.  The host wrapper pads ``seq_len`` to
    a multiple of ``RPB`` so every block owns exactly ``RPB`` valid rows.
    """
    H   = num_heads
    D   = head_size
    BKV = block_kv
    RPB = rows_per_block
    WPB = waves_per_block
    MR_BLOCK_THREADS = 64 * WPB

    assert H % mfma.MFMA_M == 0, (
        f"num_heads={H} must be a multiple of MFMA_M={mfma.MFMA_M}"
    )
    assert BKV % mfma.MFMA_N == 0, (
        f"block_kv={BKV} must be a multiple of MFMA_N={mfma.MFMA_N}"
    )
    assert D % mfma.MFMA_K == 0, (
        f"head_size={D} must be a multiple of MFMA_K={mfma.MFMA_K}"
    )
    assert RPB >= 1, f"rows_per_block must be >= 1"
    assert WPB >= 1, f"waves_per_block must be >= 1"

    N_TILES = BKV // mfma.MFMA_N        # total column-tiles per BKV block
    assert N_TILES % WPB == 0, (
        f"BKV/MFMA_N={N_TILES} must be divisible by waves_per_block={WPB}"
    )
    N_TILES_PER_WAVE = N_TILES // WPB             # column-tiles per wave
    M_TILES          = H   // mfma.MFMA_M         # head row-tiles
    K_STEPS          = D   // mfma.MFMA_K         # MFMA K-steps over the head dim

    fm_fast = arith.FastMathFlags.fast
    mfma_fn = mfma.fn

    _cvt_tag = ""
    if convert_q_fn:
        _cvt_tag += "_cq"
    if convert_kv_fn:
        _cvt_tag += "_ck"
    _kname = (
        f"fp8_mqa_logits_H{H}_D{D}_mfma{mfma.name}"
        f"_bkv{BKV}_r{RPB}_w{WPB}{_cvt_tag}_flydsl"
    )

    @flyc.kernel(name=_kname, known_block_size=[MR_BLOCK_THREADS, 1, 1])
    def kernel(
        Q: fx.Tensor,            # [seq_len, H, D]       fp8 (bytes passed raw)
        KV: fx.Tensor,           # [seq_len_kv, D]       fp8 (bytes passed raw)
        kv_scales: fx.Tensor,    # [seq_len_kv]          f32
        weights: fx.Tensor,      # [seq_len, H]          f32
        cu_starts: fx.Tensor,    # [seq_len]             i32
        cu_ends: fx.Tensor,      # [seq_len]             i32
        logits: fx.Tensor,       # [seq_len, seq_len_kv] f32
        seq_len: fx.Int32,       # padded to a multiple of RPB
        seq_len_kv: fx.Int32,
        stride_logits_s: fx.Int32,
    ):
        f32_0 = arith.constant(0.0, type=T.f32)
        _mfma_res_ty = Vec.make_type(mfma.ACC_ELEMS, fx.Float32)

        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        # Block bid (reversed) owns rows [r0, r0+RPB).
        n_blocks = fx.Int32(arith.ceildivui(_to_raw(seq_len), _to_raw(fx.Int32(RPB))))
        r0 = fx.Int32(arith.muli(
            _to_raw(n_blocks - bid - fx.Int32(1)),
            _to_raw(fx.Int32(RPB)),
        ))

        # Decompose tid into wave index and in-wave lane.
        wave     = fx.Int32(arith.divui(_to_raw(tid), _to_raw(fx.Int32(64))))
        lane     = fx.Int32(arith.remui(_to_raw(tid), _to_raw(fx.Int32(64))))
        # lane_div_N: k-group index in the A/B frag layout.
        # lane_mod_N: output column index within the MFMA_N-wide tile.
        # lane8:      byte offset of this lane's K chunk (always 8 fp8 bytes per i64).
        lane_div_N = fx.Int32(arith.divui(_to_raw(lane), _to_raw(fx.Int32(mfma.MFMA_N))))
        lane_mod_N = fx.Int32(arith.remui(_to_raw(lane), _to_raw(fx.Int32(mfma.MFMA_N))))
        lane8      = fx.Int32(arith.muli(_to_raw(lane_div_N), _to_raw(fx.Int32(8))))

        # fp8 operands are read 8 bytes at a time as 2 i32 dwords (v8i8
        # buffer_load fails to lower on gfx942), bitcast to i64 for the MFMA.
        q_i32 = GTensor(Q, dtype=T.i32, shape=(-1,))
        kv_i32 = GTensor(KV, dtype=T.i32, shape=(-1,))
        sc_t = GTensor(kv_scales, dtype=T.f32, shape=(-1,))
        w_t = GTensor(weights, dtype=T.f32, shape=(-1, H))
        cs_t = GTensor(cu_starts, dtype=T.i32, shape=(-1,))
        ce_t = GTensor(cu_ends, dtype=T.i32, shape=(-1,))
        out_t = GTensor(
            logits,
            dtype=T.f32,
            shape=(-1, -1),
            stride=(stride_logits_s, fx.Int32(1)),
        )

        def _load_pack_i64(i32_view, byte_off_i32):
            dword_off = fx.Int32(
                arith.divui(_to_raw(byte_off_i32), _to_raw(fx.Int32(4)))
            )
            v2 = i32_view.vec_load((dword_off,), vec_size=2)
            return Vec(v2).bitcast(fx.Int64)[0].ir_value()

        # ---- FN -> FNUZ in-kernel byte conversion ----
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
            hi_64 = arith.ShLIOp(arith.ExtUIOp(T.i64, hi_fix).result, arith.constant(32, type=T.i64),).result
            return arith.OrIOp(lo_64, hi_64).result

        # ---- Preload per-row metadata, Q frags, and weights for all RPB rows ----
        # All WPB waves preload all RPB rows' Q frags. A-operand layout is per
        # in-wave lane (lane_mod_N, lane8), so `lane` (not `tid`) is used here.
        starts  = [None] * RPB
        ends    = [None] * RPB
        a_packs = [None] * RPB
        w_frag  = [None] * RPB

        for j in range_constexpr(RPB):
            row = _i32_add(r0, fx.Int32(j))
            s = fx.Int32(cs_t[row])
            e = fx.Int32(ce_t[row])
            starts[j] = fx.Int32(arith.maxsi(_to_raw(s), _to_raw(fx.Int32(0))))
            ends[j]   = fx.Int32(arith.minsi(_to_raw(e), _to_raw(fx.Int32(seq_len_kv))))

            # A-frag layout: lane l -> Q[row, h = mi*MFMA_M + l%MFMA_N,
            #                              d = kk*MFMA_K + (l//MFMA_N)*8 + 0..7]
            row_a = [[None] * K_STEPS for _ in range_constexpr(M_TILES)]
            for mi in range_constexpr(M_TILES):
                h_a    = _i32_add(fx.Int32(mi * mfma.MFMA_M), lane_mod_N)
                row_h  = _i32_add(
                    fx.Int32(arith.muli(_to_raw(row), _to_raw(fx.Int32(H)))), h_a
                )
                base_a = fx.Int32(arith.muli(_to_raw(row_h), _to_raw(fx.Int32(D))))
                for kk in range_constexpr(K_STEPS):
                    d_a = _i32_add(fx.Int32(kk * mfma.MFMA_K), lane8)
                    raw = _load_pack_i64(q_i32, _i32_add(base_a, d_a))
                    row_a[mi][kk] = _fn_to_fnuz_i64(raw) if convert_q_fn else raw
            a_packs[j] = row_a

            # weights[row, h] preloaded per (mi, ii):
            #   head = mi*MFMA_M + acc_head_static_offsets[ii] + lane_div_N*group_stride
            # acc_head_static_offsets and acc_head_group_stride encode the MFMA
            # hardware's accumulator-to-head mapping (shape-specific, empirically
            # verified).  For 16x16x32: static[ii]=ii, stride=4 (sequential).
            # For 32x32x16: static[ii]=(ii//4)*8+ii%4, stride=4 (interleaved).
            row_w = [[None] * mfma.ACC_ELEMS for _ in range_constexpr(M_TILES)]
            for mi in range_constexpr(M_TILES):
                for ii in range_constexpr(mfma.ACC_ELEMS):
                    static_off = mfma.acc_head_static_offsets[ii]
                    h_w = _i32_add(
                        fx.Int32(mi * mfma.MFMA_M + static_off),
                        fx.Int32(arith.muli(
                            _to_raw(lane_div_N),
                            _to_raw(fx.Int32(mfma.acc_head_group_stride)),
                        )),
                    )
                    row_w[mi][ii] = _to_raw(fx.Float32(w_t[row, h_w]))
            w_frag[j] = row_w

        # ---- Union window across all RPB rows ----
        tile_start = _to_raw(starts[0])
        tile_end   = _to_raw(ends[0])
        for j in range_constexpr(1, RPB):
            tile_start = arith.minsi(tile_start, _to_raw(starts[j]))
            tile_end   = arith.maxsi(tile_end,   _to_raw(ends[j]))
        # Align tile_start down to BKV boundary.
        tile_start = arith.muli(
            arith.divui(tile_start, _to_raw(fx.Int32(BKV))),
            _to_raw(fx.Int32(BKV)),
        )

        tile_lo   = _to_raw(fx.Index(fx.Int32(tile_start)))
        tile_hi   = _to_raw(fx.Index(fx.Int32(tile_end)))
        tile_step = _to_raw(fx.Index(fx.Int32(BKV)))
        tile_loop = scf.ForOp(tile_lo, tile_hi, tile_step, [])
        with ir.InsertionPoint(tile_loop.body):
            col0 = fx.Int32(arith.index_cast(T.i32, tile_loop.induction_variable))

            # ---- Load B-frags: each wave loads its own N_TILES_PER_WAVE columns ----
            # wave w owns absolute n-tiles [w*N_TILES_PER_WAVE, (w+1)*N_TILES_PER_WAVE).
            # No cross-wave sharing; each wave's column addresses are disjoint.
            wave_ni_base = fx.Int32(arith.muli(
                _to_raw(wave), _to_raw(fx.Int32(N_TILES_PER_WAVE))
            ))
            b_packs        = [[None] * K_STEPS for _ in range_constexpr(N_TILES_PER_WAVE)]
            kv_scales_tile = [None] * N_TILES_PER_WAVE
            cols           = [None] * N_TILES_PER_WAVE
            for ni in range_constexpr(N_TILES_PER_WAVE):
                abs_ni = _i32_add(wave_ni_base, fx.Int32(ni))
                # col = col0 + abs_ni*MFMA_N + lane_mod_N
                col = _i32_add(
                    _i32_add(
                        col0,
                        fx.Int32(arith.muli(
                            _to_raw(abs_ni), _to_raw(fx.Int32(mfma.MFMA_N))
                        )),
                    ),
                    lane_mod_N,
                )
                cols[ni] = col
                col_clamped = fx.Int32(
                    arith.minsi(_to_raw(col), _to_raw(fx.Int32(seq_len_kv) - fx.Int32(1)))
                )
                kv_scales_tile[ni] = _to_raw(fx.Float32(sc_t[col_clamped]))
                base_b = fx.Int32(arith.muli(_to_raw(col_clamped), _to_raw(fx.Int32(D))))
                for kk in range_constexpr(K_STEPS):
                    d_b = _i32_add(fx.Int32(kk * mfma.MFMA_K), lane8)
                    raw = _load_pack_i64(kv_i32, _i32_add(base_b, d_b))
                    b_packs[ni][kk] = _fn_to_fnuz_i64(raw) if convert_kv_fn else raw

            # ---- Per-row MFMA + epilogue (inner loop over RPB rows) ----
            for j in range_constexpr(RPB):
                row = _i32_add(r0, fx.Int32(j))
                for ni in range_constexpr(N_TILES_PER_WAVE):
                    col      = cols[ni]
                    kv_scale = kv_scales_tile[ni]
                    col_sum  = _to_raw(f32_0)

                    # --- Head reduction step 1: in-register partial sum ---
                    # Each lane accumulates M_TILES * ACC_ELEMS MFMA output elements
                    # (covering different head subsets) into col_sum via scale/ReLU/weight.
                    for mi in range_constexpr(M_TILES):
                        acc = Vec.filled(mfma.ACC_ELEMS, 0.0, fx.Float32)
                        for kk in range_constexpr(K_STEPS):
                            acc = mfma_fn(
                                _mfma_res_ty,
                                [a_packs[j][mi][kk], b_packs[ni][kk], acc, 0, 0, 0],
                            )
                        for ii in range_constexpr(mfma.ACC_ELEMS):
                            score  = Vec(acc)[ii].ir_value()
                            scaled = arith.MulFOp(score, kv_scale, fastmath=fm_fast).result
                            relu   = arith.maximumf(scaled, _to_raw(f32_0))
                            wsc    = arith.MulFOp(relu, w_frag[j][mi][ii], fastmath=fm_fast).result
                            col_sum = arith.AddFOp(col_sum, wsc, fastmath=fm_fast).result

                    # --- Head reduction step 2: shuffle_xor butterfly (within wave) ---
                    # mfma.shuffle_offsets covers all lane groups so every lane ends up
                    # with the full H-wide head sum. Width is always 64 (per-wave).
                    for sh in mfma.shuffle_offsets:
                        peer    = _to_raw(ArithValue(col_sum).shuffle_xor(sh, 64))
                        col_sum = arith.AddFOp(col_sum, peer, fastmath=fm_fast).result

                    # --- Writer predicate: lane_div_N == 0 owns MFMA_N distinct cols ---
                    is_writer = arith.andi(
                        _to_raw(arith.cmpi(
                            arith.CmpIPredicate.eq,
                            _to_raw(lane_div_N),
                            _to_raw(fx.Int32(0)),
                        )),
                        _to_raw(arith.cmpi(
                            arith.CmpIPredicate.slt,
                            _to_raw(col),
                            _to_raw(ends[j]),
                        )),
                    )
                    with ir.InsertionPoint(scf.IfOp(is_writer).then_block):
                        out_t[row, col] = fx.Float32(col_sum)
                        scf.YieldOp([])

            scf.YieldOp([])

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
        stream: fx.Stream = fx.Stream(None),
    ):
        n_blocks = arith.ceildivui(
            _to_raw(seq_len), _to_raw(fx.Int32(RPB))
        )
        gx = arith.index_cast(T.index, n_blocks)
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
        ).launch(grid=(gx, 1, 1), block=(MR_BLOCK_THREADS, 1, 1), stream=stream)

    return launch_fp8_mqa_logits_mfma_r_w


# --------------------------------------------------------------------------- #
# Kernel-variant registry.
#
# Variant tags follow the scheme ``"mfma<MxNxK>_bkv<B>_r<RPB>_w<WPB>"`` where:
#   MxNxK  = MFMA tile dimensions (e.g. 16x16x32 or 32x32x16)
#   B      = block_kv (KV columns per tile-loop iteration)
#   RPB    = rows per block (Q-row KV-reuse factor)
#   WPB    = waves per block (column-split parallelism factor)
#
# To add a new variant, register a lambda below; no other call sites need
# changing (tests and benchmarks use the env var / ``variant=`` arg).
#
# Note: each MFMA-variant lambda hardcodes ``block_kv`` (bkv is part of the
# tag), overriding whatever ``block_kv`` the caller passed to
# ``compile_fp8_mqa_logits``.  The ``scalar`` variant (not listed here, but
# kept as ``_build_kernel``) is accessible via the env toggle only.
# --------------------------------------------------------------------------- #

def _mfma16_bkv(bkv, r, w):
    """Helper: lambda for a 16x16x32-fp8 variant with given bkv/r/w."""
    return lambda **kw: _build_kernel_mfma_r_w(
        **{**kw, "block_kv": bkv},
        mfma=_MFMA16, rows_per_block=r, waves_per_block=w,
    )

def _mfma32_bkv(bkv, r, w):
    """Helper: lambda for a 32x32x16-fp8 variant with given bkv/r/w."""
    return lambda **kw: _build_kernel_mfma_r_w(
        **{**kw, "block_kv": bkv},
        mfma=_MFMA32, rows_per_block=r, waves_per_block=w,
    )

_VARIANT_BUILDERS = {
    # --- 16x16x32 fp8 variants (bkv=64) ---
    "mfma16x16x32_bkv64_r1_w1":   _mfma16_bkv(64,  1, 1),
    # --- 16x16x32 fp8 variants (bkv=128) ---
    "mfma16x16x32_bkv128_r1_w1":  _mfma16_bkv(128, 1, 1),
    "mfma16x16x32_bkv128_r2_w1":  _mfma16_bkv(128, 2, 1),
    "mfma16x16x32_bkv128_r4_w1":  _mfma16_bkv(128, 4, 1),
    "mfma16x16x32_bkv128_r1_w4":  _mfma16_bkv(128, 1, 4),
    "mfma16x16x32_bkv128_r2_w2":  _mfma16_bkv(128, 2, 2),
    "mfma16x16x32_bkv128_r2_w4":  _mfma16_bkv(128, 2, 4),
    "mfma16x16x32_bkv128_r4_w2":  _mfma16_bkv(128, 4, 2),
    # --- 16x16x32 fp8 variants (bkv=256) ---
    "mfma16x16x32_bkv256_r1_w1":  _mfma16_bkv(256, 1, 1),
    "mfma16x16x32_bkv256_r2_w2":  _mfma16_bkv(256, 2, 2),
    # --- 32x32x16 fp8 variants (bkv=128) ---
    "mfma32x32x16_bkv128_r1_w1":  _mfma32_bkv(128, 1, 1),
    "mfma32x32x16_bkv128_r2_w1":  _mfma32_bkv(128, 2, 1),
    "mfma32x32x16_bkv128_r2_w2":  _mfma32_bkv(128, 2, 2),
}

KERNEL_VARIANTS = tuple(_VARIANT_BUILDERS.keys())
DEFAULT_VARIANT = "mfma16x16x32_bkv128_r2_w2"


def _resolve_variant(variant=None, *, use_mfma=None):
    """Resolve the effective variant tag from the (variant, use_mfma, env) inputs.

    Precedence: explicit ``variant=`` > legacy ``use_mfma=`` > env var
    ``FLYDSL_FP8_MQA_LOGITS_VARIANT`` > legacy env ``FLYDSL_FP8_MQA_LOGITS_MFMA``
    > ``DEFAULT_VARIANT``.
    """
    if variant is not None:
        tag = variant
    elif use_mfma is not None:
        # Legacy use_mfma=True -> single-row 16x16x32 bkv128.
        tag = "mfma16x16x32_bkv128_r1_w1" if use_mfma else "scalar"
    else:
        env_variant = os.environ.get("FLYDSL_FP8_MQA_LOGITS_VARIANT")
        if env_variant:
            tag = env_variant
        else:
            # Legacy boolean env toggle (kept for back-compat): "0" -> scalar.
            if os.environ.get("FLYDSL_FP8_MQA_LOGITS_MFMA", "1") == "0":
                tag = "scalar"
            else:
                tag = DEFAULT_VARIANT
    if tag == "scalar":
        return tag  # scalar kept for legacy env toggle; not in _VARIANT_BUILDERS
    if tag not in _VARIANT_BUILDERS:
        raise ValueError(
            f"unknown fp8_mqa_logits variant {tag!r}; "
            f"available: {list(KERNEL_VARIANTS)}"
        )
    return tag


@lru_cache(maxsize=32)
def compile_fp8_mqa_logits(
    *,
    num_heads: int,
    head_size: int,
    block_kv: int = 128,
    paged: bool = False,
    variant: str = DEFAULT_VARIANT,
    convert_q_fn: bool = False,
    convert_kv_fn: bool = False,
):
    """Return a cached, compiled FlyDSL launcher for the given shape config.

    Parameters
    ----------
    num_heads : int
        Number of indexer query heads (compile-time constant, power of two).
    head_size : int
        Head dimension D (compile-time constant, power of two; D in {64, 128}).
    block_kv : int
        Passed to ``_build_kernel`` for the ``"scalar"`` variant only.  For all
        MFMA variants the block_kv is encoded in the variant tag and the lambda
        overrides this value.
    paged : bool
        Reserved for the Phase-2 paged variant. Must be False for now.
    variant : str
        Which kernel version to build (see ``KERNEL_VARIANTS``). Tags follow
        ``"mfma<MxNxK>_bkv<B>_r<RPB>_w<WPB>"``; ``"scalar"`` is the legacy
        correctness-first fallback (not in ``KERNEL_VARIANTS`` but still accepted).
    convert_q_fn : bool
        If True, Q bytes are FP8 FN and the kernel converts them to FNUZ
        in-register before the MFMA (applies to all ``mfma*`` variants).
    convert_kv_fn : bool
        If True, KV bytes are FP8 FN and the kernel converts them to FNUZ
        in-register before the MFMA (applies to all ``mfma*`` variants).
    """
    if paged:
        raise NotImplementedError(
            "Paged FlyDSL fp8_mqa_logits is Phase 2 and not implemented yet."
        )
    if variant == "scalar":
        launcher = _build_kernel(
            num_heads=num_heads, head_size=head_size, block_kv=block_kv,
        )
    elif variant in _VARIANT_BUILDERS:
        launcher = _VARIANT_BUILDERS[variant](
            num_heads=num_heads, head_size=head_size, block_kv=block_kv,
            convert_q_fn=convert_q_fn, convert_kv_fn=convert_kv_fn,
        )
    else:
        raise ValueError(
            f"unknown fp8_mqa_logits variant {variant!r}; "
            f"available: {list(KERNEL_VARIANTS)}"
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
    """FlyDSL gfx942 FP8 MQA logits -- drop-in for the Triton ``fp8_mqa_logits``.

    Q:            [seq_len, NUM_HEADS, HEAD_SIZE], dtype float8
    KV:           [seq_len_kv, HEAD_SIZE], dtype float8
    kv_scales:    [seq_len_kv], dtype float32
    weights:      [seq_len, NUM_HEADS], dtype float32
    cu_starts:    [seq_len], dtype int32, per-row window start (inclusive)
    cu_ends:      [seq_len], dtype int32, per-row window end (exclusive)
    clean_logits: bool. If True, positions outside [cu_starts[i], cu_ends[i])
                  in row i are written as -inf. If False, the kernel skips
                  those positions and the caller owns whatever is left there.
    stream:       optional HIP stream; defaults to the current stream.
    variant:      optional kernel-version tag (see ``KERNEL_VARIANTS``). If None,
                  resolved from the ``FLYDSL_FP8_MQA_LOGITS_VARIANT`` env var
                  (or the legacy ``FLYDSL_FP8_MQA_LOGITS_MFMA`` toggle), defaulting
                  to ``DEFAULT_VARIANT`` (``"mfma16x16x32_bkv128_r2_w2"``).

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

    # The gfx942 fp8 MFMA (v_mfma_f32_16x16x32_fp8_fp8) always interprets
    # operands as e4m3 FNUZ (bias 8). When Q or KV arrive in e4m3 FN (OCP,
    # bias 7, max 448), two corrections are needed:
    #   1. In-kernel byte conversion: dequant-as-FNUZ × 2 -> requant-to-FNUZ
    #      recovers the true FN value in FNUZ encoding for 255/256 byte patterns
    #      (only 0x80 = FN -0 -> FNUZ NaN differs; values > 240 saturate).
    #   2. kv_scales compensation: multiply by 2 per FN operand. Since
    #      logits = sum_h ReLU(QK x scale) x w and ReLU is pos-homogeneous,
    #      this compensates the systematic 2x factor from the FN/FNUZ bias diff.
    # Combined, in-kernel conversion handles the per-byte encoding while
    # kv_scales compensation handles the overall numeric factor.
    # The 'mfma' lernel variant does in-kernel halve-and-requant:
    # dequant-as-FNUZ (= FN_value/2) * 0.5 → requant-to-FNUZ, keeping all
    # values ≤ 120 < 240 (safely within FNUZ range). The 2x factor per FN
    # operand is compensated via kv_scales. Non-mfma variants fall back to
    # the host-side halve-before-cast.
    _fnuz = torch.float8_e4m3fnuz
    convert_q_fn = Q.dtype != _fnuz
    convert_kv_fn = KV.dtype != _fnuz
    scale_mul = 1.0
    if convert_q_fn:
        scale_mul *= 2.0
    if convert_kv_fn:
        scale_mul *= 2.0
    if scale_mul != 1.0:
        kv_scales = kv_scales.to(torch.float32) * scale_mul

    variant = _resolve_variant(variant)

    # Variants with in-kernel FN -> FNUZ support; others fall back to host-side conversion.
    if not variant.startswith("mfma") and (convert_q_fn or convert_kv_fn):
        if convert_q_fn:
            Q = (Q.to(torch.float32) * 0.5).to(_fnuz)
        if convert_kv_fn:
            KV = (KV.to(torch.float32) * 0.5).to(_fnuz)
        convert_q_fn = False
        convert_kv_fn = False

    launcher = compile_fp8_mqa_logits(
        num_heads=num_heads,
        head_size=head_size,
        block_kv=128,
        paged=False,
        variant=variant,
        convert_q_fn=convert_q_fn,
        convert_kv_fn=convert_kv_fn,
    )

    # mfma_r* kernels require seq_len padded to a multiple of rows_per_block so
    # every block owns exactly RPB rows.  Padded rows get empty windows (start ==
    # end == 0) so the kernel writes nothing for them; the output is sliced back
    # to the original seq_len after the launch.
    # Parse RPB from variant tag "mfma<shape>_bkv<B>_r<N>_w<M>" -> N.
    _rpb_match = re.match(r"mfma\d+x\d+x\d+_bkv\d+_r(\d+)_w\d+", variant)
    _RPB = int(_rpb_match.group(1)) if _rpb_match else 1
    seq_len_padded = ((seq_len + _RPB - 1) // _RPB) * _RPB
    if seq_len_padded != seq_len:
        pad = seq_len_padded - seq_len
        Q = torch.cat([Q, Q.new_zeros((pad, num_heads, head_size))], dim=0)
        weights = torch.cat([weights, weights.new_zeros((pad, num_heads))], dim=0)
        cu_starts = torch.cat([cu_starts, cu_starts.new_zeros(pad)], dim=0)
        cu_ends = torch.cat([cu_ends, cu_ends.new_zeros(pad)], dim=0)

    # Match the Triton launcher's -inf-prefill / padding behavior so the two
    # produce identically-shaped, identically-masked outputs.
    aligned_size = 256
    seq_len_kv_aligned = (
        (seq_len_kv + aligned_size - 1) // aligned_size * aligned_size
    )
    if clean_logits:
        logits = torch.full(
            (seq_len_padded, seq_len_kv_aligned),
            fill_value=-float("inf"),
            dtype=torch.float32,
            device=Q.device,
        )[:, :seq_len_kv]
    else:
        logits = torch.empty(
            (seq_len_padded, seq_len_kv_aligned),
            dtype=torch.float32,
            device=Q.device,
        )[:, :seq_len_kv]

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
            stream,
        )

    return logits[:seq_len, :]
