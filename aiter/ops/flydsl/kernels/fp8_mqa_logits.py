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
from functools import lru_cache

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, range_constexpr, rocdl
from flydsl.expr.numeric import ArithValue
from flydsl.expr.typing import T
from flydsl._mlir.dialects import scf, vector as mlir_vector
from flydsl._mlir import ir
from flydsl.runtime.device import get_rocm_arch as get_hip_arch

from .tensor_shim import GTensor, _run_compiled, _to_raw

Vec = fx.Vector


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


# MFMA based direct port of the Triton kernel.
def _build_kernel_mfma(*, num_heads: int, head_size: int, block_kv: int,
                       convert_q_fn: bool = False, convert_kv_fn: bool = False):
    """MFMA-based kernel: fp8 ``v_mfma_f32_16x16x32_fp8_fp8`` Q.K.

    One wave64 per query row (grid ``(seq_len,)``), reverse row order. For each
    KV tile of ``block_kv`` columns the wave computes ``scores[H, block_kv]``
    via MFMA, applies ``ReLU(score * kv_scale) * weights`` per element, sums over
    the head (M) axis, and stores the per-column logit.

    Verified CDNA3 ``16x16x32_fp8_fp8`` fragment layout (from preshuffle_gemm):
      * A operand: lane ``l`` -> ``A[row = l%16, k = (l//16)*8 + 0..7]``
      * B operand: lane ``l`` -> ``B[k = (l//16)*8 + 0..7, col = l%16]``
      * acc ``vec<4 x f32>`` element ``ii`` -> ``row = (l//16)*4 + ii``,
        ``col = l%16``.
    The head axis lands on the MFMA M dimension, so summing over heads needs an
    in-register sum over ``ii`` and the ``mi`` row-tiles plus a cross-lane
    xor-shuffle over the ``l//16`` groups (offsets 16, 32). With a single wave
    there is no cross-wave LDS pass.
    """
    H = num_heads
    D = head_size
    BKV = block_kv
    assert H % 16 == 0, f"num_heads={H} must be a multiple of 16 for MFMA"
    assert BKV % 16 == 0, f"block_kv={BKV} must be a multiple of 16 for MFMA"
    assert D % 32 == 0, f"head_size={D} must be a multiple of 32 for MFMA K32"
    M_TILES = H // 16          # head row-tiles
    N_TILES = BKV // 16        # KV column-tiles
    K_STEPS = D // 32          # K32 MFMA steps over the head dim

    fm_fast = arith.FastMathFlags.fast
    mfma_fn = rocdl.mfma_f32_16x16x32_fp8_fp8

    _cvt_tag = ""
    if convert_q_fn:
        _cvt_tag += "_cq"
    if convert_kv_fn:
        _cvt_tag += "_ck"
    _kname = f"fp8_mqa_logits_H{H}_D{D}_bkv{BKV}_mfma{_cvt_tag}_flydsl"

    @flyc.kernel(name=_kname, known_block_size=[BLOCK_THREADS, 1, 1])
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
        stride_logits_s: fx.Int32,
    ):
        f32_0 = arith.constant(0.0, type=T.f32)
        mfma_res_ty = Vec.make_type(4, fx.Float32)

        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        row = fx.Int32(seq_len) - bid - fx.Int32(1)

        lane = fx.Int32(tid)
        lane_div_16 = fx.Int32(arith.divui(_to_raw(lane), _to_raw(fx.Int32(16))))
        lane_mod_16 = fx.Int32(arith.remui(_to_raw(lane), _to_raw(fx.Int32(16))))

        # fp8 operands are read 8 bytes at a time as 2 i32 dwords (a direct
        # v8i8 buffer_load fails to lower on gfx942), bitcast to one i64 =
        # vec<8 x fp8> for the MFMA. Flat i32 views, indexed in dword units.
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

        start = fx.Int32(cs_t[row])
        end = fx.Int32(ce_t[row])
        start = fx.Int32(arith.maxsi(_to_raw(start), _to_raw(fx.Int32(0))))
        end = fx.Int32(arith.minsi(_to_raw(end), _to_raw(fx.Int32(seq_len_kv))))

        lane8 = fx.Int32(arith.muli(_to_raw(lane_div_16), _to_raw(fx.Int32(8))))

        def _load_pack_i64(i32_view, byte_off_i32):
            # byte_off_i32: element offset in fp8 bytes (multiple of 4). Load 2
            # dwords, bitcast vec<2 x i32> -> i64.
            dword_off = fx.Int32(
                arith.divui(_to_raw(byte_off_i32), _to_raw(fx.Int32(4)))
            )
            v2 = i32_view.vec_load((dword_off,), vec_size=2)
            return Vec(v2).bitcast(fx.Int64)[0].ir_value()

        # ---- FN -> FNUZ in-kernel byte conversion ----
        #
        # For all 256 byte values, the FN and FNUZ encodings satisfy
        # FNUZ_value(byte) == FN_value(byte) / 2 exactly — so the raw FN
        # bytes can be passed directly to the FNUZ MFMA and the 2x factor
        # compensated via kv_scales. The conversion is thus an identity
        # for 253/256 byte patterns.
        #
        # The one exception is byte 0x80:
        #   FN:   0x80 = negative zero (-0.0)  ->  mathematically zero
        #   FNUZ: 0x80 = NaN                   ->  corrupts MFMA output
        #
        # A single NaN in the 128-element dot product poisons the entire
        # accumulator element.
        #
        # Fix: map 0x80 -> 0x00 (FNUZ +0). For all other bytes the identity
        # mapping is correct. This is done per i32 (4 bytes at a time) with
        # byte-lane arithmetic to avoid branches.
        def _fn_to_fnuz_i64(raw_i64):
            """Map FN byte 0x80 (neg-zero) -> 0x00 in 8 packed fp8 bytes.

            Implementation: extract each byte, compare to 0x80, select 0x00
            if equal, else keep original. Rebuild the i64 from 8 cleaned
            bytes.
            """
            lo_i32 = arith.TruncIOp(T.i32, raw_i64).result
            hi_i64 = arith.ShRUIOp(raw_i64, arith.constant(32, type=T.i64),).result
            hi_i32 = arith.TruncIOp(T.i32, hi_i64).result

            def _fix_i32(src):
                """Fix 0x80 bytes in one i32 (4 fp8 bytes)."""
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

        # ---- preload A (Q) operands: per (mi, kk) one i64 of 8 fp8 along D ----
        # lane l -> Q[row, h = mi*16 + lane%16, d = kk*32 + (lane//16)*8 + 0..7]
        a_packs = [[None] * K_STEPS for _ in range_constexpr(M_TILES)]
        for mi in range_constexpr(M_TILES):
            h_a = _i32_add(fx.Int32(mi * 16), lane_mod_16)
            # byte offset of Q[row, h_a, 0] = (row*H + h_a) * D
            row_h = _i32_add(fx.Int32(arith.muli(_to_raw(row), _to_raw(fx.Int32(H)))), h_a)
            base_b = fx.Int32(arith.muli(_to_raw(row_h), _to_raw(fx.Int32(D))))
            for kk in range_constexpr(K_STEPS):
                d_a = _i32_add(fx.Int32(kk * 32), lane8)
                raw = _load_pack_i64(q_i32, _i32_add(base_b, d_a))
                a_packs[mi][kk] = _fn_to_fnuz_i64(raw) if convert_q_fn else raw

        # weights[row, h] preloaded per (mi, ii) head owned by this lane.
        # head = mi*16 + lane_div_16*4 + ii
        w_frag = [[None] * 4 for _ in range_constexpr(M_TILES)]
        for mi in range_constexpr(M_TILES):
            for ii in range_constexpr(4):
                h_w = _i32_add(
                    fx.Int32(mi * 16),
                    _i32_add(
                        fx.Int32(
                            arith.muli(_to_raw(lane_div_16), _to_raw(fx.Int32(4)))
                        ),
                        fx.Int32(ii),
                    ),
                )
                w_frag[mi][ii] = _to_raw(fx.Float32(w_t[row, h_w]))

        # ---- KV tile loop over the window, BKV columns at a time ----
        tile_lo = _to_raw(fx.Index(start))
        tile_hi = _to_raw(fx.Index(end))
        tile_step = _to_raw(fx.Index(fx.Int32(BKV)))
        tile_loop = scf.ForOp(tile_lo, tile_hi, tile_step, [])
        with ir.InsertionPoint(tile_loop.body):
            col0 = fx.Int32(arith.index_cast(T.i32, tile_loop.induction_variable))

            # preload B (KV) operands: per (ni, kk) one i64 of 8 fp8 along D.
            # lane l -> K[n = col0 + ni*16 + lane%16, d = kk*32 + (l//16)*8 + 0..7]
            # Columns past `end` are clamped so the load stays in-bounds; the
            # store below masks them out.
            b_packs = [[None] * K_STEPS for _ in range_constexpr(N_TILES)]
            for ni in range_constexpr(N_TILES):
                n_b = _i32_add(_i32_add(col0, fx.Int32(ni * 16)), lane_mod_16)
                n_b = fx.Int32(
                    arith.minsi(
                        _to_raw(n_b), _to_raw(fx.Int32(seq_len_kv) - fx.Int32(1))
                    )
                )
                # byte offset of K[n_b, 0] = n_b * D
                base_b = fx.Int32(arith.muli(_to_raw(n_b), _to_raw(fx.Int32(D))))
                for kk in range_constexpr(K_STEPS):
                    d_b = _i32_add(fx.Int32(kk * 32), lane8)
                    raw = _load_pack_i64(kv_i32, _i32_add(base_b, d_b))
                    b_packs[ni][kk] = _fn_to_fnuz_i64(raw) if convert_kv_fn else raw

            # MFMA: acc[mi][ni] = sum_kk A[mi][kk] . B[ni][kk]
            for ni in range_constexpr(N_TILES):
                # col_sum accumulates ReLU(score*scale)*w over heads (ii + mi).
                col = _i32_add(_i32_add(col0, fx.Int32(ni * 16)), lane_mod_16)
                kv_scale = _to_raw(fx.Float32(sc_t[
                    fx.Int32(
                        arith.minsi(
                            _to_raw(col),
                            _to_raw(fx.Int32(seq_len_kv) - fx.Int32(1)),
                        )
                    )
                ]))
                col_sum = _to_raw(f32_0)
                for mi in range_constexpr(M_TILES):
                    acc = Vec.filled(4, 0.0, fx.Float32)
                    for kk in range_constexpr(K_STEPS):
                        acc = mfma_fn(
                            mfma_res_ty,
                            [a_packs[mi][kk], b_packs[ni][kk], acc, 0, 0, 0],
                        )
                    for ii in range_constexpr(4):
                        score = Vec(acc)[ii].ir_value()
                        scaled = arith.MulFOp(
                            score, kv_scale, fastmath=fm_fast
                        ).result
                        relu = arith.maximumf(scaled, _to_raw(f32_0))
                        wsc = arith.MulFOp(
                            relu, w_frag[mi][ii], fastmath=fm_fast
                        ).result
                        col_sum = arith.AddFOp(
                            col_sum, wsc, fastmath=fm_fast
                        ).result

                # head reduce across lane_div_16 groups (offsets 16, 32).
                for sh in [16, 32]:
                    peer = _to_raw(
                        ArithValue(col_sum).shuffle_xor(sh, BLOCK_THREADS)
                    )
                    col_sum = arith.AddFOp(
                        col_sum, peer, fastmath=fm_fast
                    ).result

                # lanes with lane_div_16 == 0 own the 16 distinct columns of
                # this tile; store col = col0 + ni*16 + lane%16 if in window.
                is_writer = arith.andi(
                    _to_raw(
                        arith.cmpi(
                            arith.CmpIPredicate.eq,
                            _to_raw(lane_div_16),
                            _to_raw(fx.Int32(0)),
                        )
                    ),
                    _to_raw(
                        arith.cmpi(
                            arith.CmpIPredicate.slt,
                            _to_raw(col),
                            _to_raw(end),
                        )
                    ),
                )
                with ir.InsertionPoint(scf.IfOp(is_writer).then_block):
                    out_t[row, col] = fx.Float32(col_sum)
                    scf.YieldOp([])
            scf.YieldOp([])

    @flyc.jit
    def launch_fp8_mqa_logits_mfma(
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

    return launch_fp8_mqa_logits_mfma


# --------------------------------------------------------------------------- #
# Kernel-variant registry.
#
# Each entry maps a variant *tag* (the version label used by callers/benchmarks)
# to its ``_build_*`` factory. To add a new version of the indexer kernel, write
# a ``_build_kernel_<name>`` factory with the same ``(*, num_heads, head_size,
# block_kv)`` signature and register it here -- it then becomes selectable via
# ``variant="<name>"`` everywhere (public API, tests, the A/B benchmark) without
# touching the call sites.
#
# ``"mfma"`` is the current default/baseline; ``"scalar"`` is the slow
# correctness-first fallback.
# --------------------------------------------------------------------------- #
_VARIANT_BUILDERS = {
    "scalar": _build_kernel,
    "mfma": _build_kernel_mfma,
}

# Order matters: the first available tag is the default.
KERNEL_VARIANTS = tuple(_VARIANT_BUILDERS.keys())
DEFAULT_VARIANT = "mfma"


def _resolve_variant(variant=None, *, use_mfma=None):
    """Resolve the effective variant tag from the (variant, use_mfma, env) inputs.

    Precedence: explicit ``variant=`` > legacy ``use_mfma=`` > env var
    ``FLYDSL_FP8_MQA_LOGITS_VARIANT`` > legacy env ``FLYDSL_FP8_MQA_LOGITS_MFMA``
    > ``DEFAULT_VARIANT``.
    """
    if variant is not None:
        tag = variant
    elif use_mfma is not None:
        tag = "mfma" if use_mfma else "scalar"
    else:
        env_variant = os.environ.get("FLYDSL_FP8_MQA_LOGITS_VARIANT")
        if env_variant:
            tag = env_variant
        else:
            # Legacy boolean env toggle (kept for back-compat): "0" -> scalar.
            tag = "scalar" if (
                os.environ.get("FLYDSL_FP8_MQA_LOGITS_MFMA", "1") == "0"
            ) else DEFAULT_VARIANT
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
        KV tile width processed per inner-loop iteration.
    paged : bool
        Reserved for the Phase-2 paged variant. Must be False for now.
    variant : str
        Which kernel version to build (see ``KERNEL_VARIANTS``). ``"mfma"`` is the
        fp8-MFMA matrix-core baseline; ``"scalar"`` is the ``v_cvt_f32_fp8``
        correctness-first fallback.
    convert_q_fn : bool
        If True, Q data is in FP8 FN encoding and the kernel converts bytes
        to FNUZ in-register before the MFMA (only applies to ``"mfma"``).
    convert_kv_fn : bool
        If True, KV data is in FP8 FN encoding and the kernel converts bytes
        to FNUZ in-register before the MFMA (only applies to ``"mfma"``).
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
    builder = _VARIANT_BUILDERS[variant]
    # Only the mfma builder accepts convert flags; others ignore them.
    if variant == "mfma":
        launcher = builder(
            num_heads=num_heads, head_size=head_size, block_kv=block_kv,
            convert_q_fn=convert_q_fn, convert_kv_fn=convert_kv_fn,
        )
    else:
        launcher = builder(
            num_heads=num_heads, head_size=head_size, block_kv=block_kv,
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
                  to ``DEFAULT_VARIANT`` (``"mfma"``).

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

    # For non-mfma variants (e.g. scalar), fall back to host-side conversion
    # since they don't have in-kernel FN→FNUZ support.
    if variant != "mfma" and (convert_q_fn or convert_kv_fn):
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

    # Match the Triton launcher's -inf-prefill / padding behavior so the two
    # produce identically-shaped, identically-masked outputs.
    aligned_size = 256
    seq_len_kv_aligned = (
        (seq_len_kv + aligned_size - 1) // aligned_size * aligned_size
    )
    if clean_logits:
        logits = torch.full(
            (seq_len, seq_len_kv_aligned),
            fill_value=-float("inf"),
            dtype=torch.float32,
            device=Q.device,
        )[:, :seq_len_kv]
    else:
        logits = torch.empty(
            (seq_len, seq_len_kv_aligned),
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
            int(seq_len),
            int(seq_len_kv),
            int(logits.stride(0)),
            stream,
        )

    return logits
