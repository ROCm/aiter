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
from flydsl.expr import arith, buffer_ops, const_expr, gpu, range_constexpr, rocdl
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


def _f32_from_raw(scalar):
    """Extend a (raw) f32/bf16 scalar load to f32 via arith.extf-or-identity.

    GTensor.load returns a raw ir.Value (the buffer_load result), which has no
    ``.extf`` helper, so wrap it in ArithValue first.
    """
    return ArithValue(_to_raw(scalar)).extf(T.f32)


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
        Q: fx.Tensor,  # [seq_len, H, D]      fp8
        KV: fx.Tensor,  # [seq_len_kv, D]      fp8
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
def _build_kernel_mfma(*, num_heads: int, head_size: int, block_kv: int):
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

    fp8_dt = _fp8_dtype()
    fm_fast = arith.FastMathFlags.fast
    mfma_fn = rocdl.mfma_f32_16x16x32_fp8_fp8

    _kname = f"fp8_mqa_logits_H{H}_D{D}_bkv{BKV}_mfma_flydsl"

    @flyc.kernel(name=_kname, known_block_size=[BLOCK_THREADS, 1, 1])
    def kernel(
        Q: fx.Tensor,  # [seq_len, H, D]      fp8
        KV: fx.Tensor,  # [seq_len_kv, D]      fp8
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
                a_packs[mi][kk] = _load_pack_i64(q_i32, _i32_add(base_b, d_a))

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
                    b_packs[ni][kk] = _load_pack_i64(kv_i32, _i32_add(base_b, d_b))

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


# GEAK optimized FlyDSL kernel
def _iv_to_i32(idx_val):
    """Cast an scf.for induction variable (index type) to fx.Int32.

    The MLIR i32 type must be built inside an active MLIR context (i.e. during
    kernel tracing), so it is resolved lazily here rather than at import.
    """
    i32_ty = ir.IntegerType.get_signless(32)
    return fx.Int32(arith.index_cast(i32_ty, fx.arith._to_raw(idx_val)))


# Each wave stages into and reads from its own private LDS slab, so a
# workgroup barrier is not required for correctness (intra-wave LDS
# write->read is ordered by the compiler's lds waitcnt). Kept as a toggle so
# the safe behaviour can be restored if a future layout shares slabs.
import os as _os
_USE_LDS_BARRIER = _os.environ.get("FLYDSL_MQA_LDS_BARRIER", "0") == "1"

_FNUZ = torch.float8_e4m3fnuz
_DTYPE_LOGGED = False

NUM_HEADS = 64
HEAD_DIM = 128

WAVE = 64
M_TILES = NUM_HEADS // 16     # 4
K_STEPS = HEAD_DIM // 32      # 4
N_TILE = 16
WAVES = 4                     # waves per block
TPW = 2                       # ntiles processed per wave (ILP)
# NOTE: GEAK-specific block size (256 = 4 wave64). Named distinctly so it does
# NOT clobber the module-level ``BLOCK_THREADS = 64`` that the scalar and mfma
# kernels rely on -- the builders read module globals at call time, so a plain
# ``BLOCK_THREADS`` here would silently launch those kernels with 256 threads.
_GEAK_BLOCK_THREADS = WAVE * WAVES  # 256
COLS_PER_BLOCK = N_TILE * WAVES * TPW   # 128 kv columns finished per grid.y block
# Query rows handled per block. Each kv column tile is staged into LDS ONCE and
# reused across all ROWS_PER_BLOCK rows' MFMAs (rows processed sequentially so
# accumulators/weights are reused -> VGPR/occupancy stay healthy). This cuts KV
# global traffic and per-tile loop/staging overhead by ~ROWS_PER_BLOCK.
ROWS_PER_BLOCK = 2
NEG_INF = float("-inf")

# Back-compat default shape for the standalone Model below.
S_Q = 128
S_K = 1024

def build_mqa_logits(S_K, ROWS=ROWS_PER_BLOCK):
    assert S_K % COLS_PER_BLOCK == 0, (
        f"S_K ({S_K}) must be a multiple of {COLS_PER_BLOCK}"
    )
    COLS_PER_WAVE = 16 * TPW          # kv columns staged per wave
    KV_SLAB_I64 = COLS_PER_WAVE * (HEAD_DIM // 8)  # i64 per wave slab
    NSTAGE = (KV_SLAB_I64 + WAVE - 1) // WAVE   # coalesced copy steps per wave

    @fx.struct
    class SharedStorage:
        # Single per-wave-private KV slab, shared across the ROWS rows in the
        # block. (A double-buffered variant regressed: doubling LDS halved
        # occupancy; this kernel is occupancy/TLP-bound.)
        kv: fx.Array[fx.Int64, WAVES * KV_SLAB_I64, 16]

    @flyc.kernel(known_block_size=[_GEAK_BLOCK_THREADS, 1, 1])
    def mqa_logits_kernel(
        Q: fx.Tensor,        # i64 view: 8 fp8 per i64
        KV: fx.Tensor,       # i64 view
        KVSCALE: fx.Tensor,  # f32 [S_K]
        W: fx.Tensor,        # f32 [M*NUM_HEADS]
        STARTS: fx.Tensor,   # i32 [M]
        ENDS: fx.Tensor,     # i32 [M]
        OUT: fx.Tensor,      # f32 [M*S_K]
    ):
        # One block per ROWS query rows. The block (4 waves = 256 lanes) walks
        # the union of the rows' active kv-column blocks with an internal runtime
        # loop; each column tile is staged into LDS ONCE and reused across all
        # ROWS rows. Q frags for all ROWS rows stay resident; accumulators are
        # reused per row (rows processed sequentially) to keep VGPR pressure low.
        rt = fx.block_idx.x
        tid = fx.thread_idx.x
        wave = tid // fx.Int32(WAVE)
        lane = tid % fx.Int32(WAVE)

        q_rsrc = buffer_ops.create_buffer_resource(Q, max_size=True)
        kv_rsrc = buffer_ops.create_buffer_resource(KV, max_size=True)
        ks_buf = fx.rocdl.make_buffer_tensor(KVSCALE)
        w_buf = fx.rocdl.make_buffer_tensor(W)
        st_buf = fx.rocdl.make_buffer_tensor(STARTS)
        en_buf = fx.rocdl.make_buffer_tensor(ENDS)
        OUT_buf = fx.rocdl.make_buffer_tensor(OUT)

        c16 = fx.Int32(16)
        lane_row = lane % c16
        lane_kg = lane // c16
        ROW_I64 = fx.Int32(HEAD_DIM // 8)   # 16 i64 per head/kvcol row
        cpb = fx.Int32(COLS_PER_BLOCK)

        # ---- per-row metadata, Q frags, weights, out base (all resident) ----
        starts = [None] * ROWS
        ends = [None] * ROWS
        q_frag = [None] * ROWS
        wvals = [None] * ROWS
        out_base = [None] * ROWS
        for j in range_constexpr(ROWS):
            r = rt * fx.Int32(ROWS) + fx.Int32(j)
            s = fx.memref_load(st_buf, r)
            e = fx.memref_load(en_buf, r)
            starts[j] = (s > fx.Int32(0)).select(s, fx.Int32(0))
            ends[j] = (e < fx.Int32(S_K)).select(e, fx.Int32(S_K))
            out_base[j] = r * fx.Int32(S_K)
            q_row_base = r * fx.Int32(NUM_HEADS) * ROW_I64
            frag = [[None] * K_STEPS for _ in range(M_TILES)]
            for mt in range_constexpr(M_TILES):
                head_i64 = q_row_base + (fx.Int32(mt * 16) + lane_row) * ROW_I64
                for ks in range_constexpr(K_STEPS):
                    frag[mt][ks] = buffer_ops.buffer_load(
                        q_rsrc, head_i64 + fx.Int32(ks * 4) + lane_kg,
                        vec_width=1, dtype=fx.Int64)
            q_frag[j] = frag
            w_row_base = r * fx.Int32(NUM_HEADS)
            wj = [[None] * 4 for _ in range(M_TILES)]
            for mt in range_constexpr(M_TILES):
                for i in range_constexpr(4):
                    head = fx.Int32(mt * 16) + lane_kg * fx.Int32(4) + fx.Int32(i)
                    wj[mt][i] = fx.memref_load(w_buf, w_row_base + head)
            wvals[j] = wj

        # union of the rows' active column-block ranges (j=0 init is idempotent)
        cb_start = starts[0] // cpb
        cb_end = (ends[0] + fx.Int32(COLS_PER_BLOCK - 1)) // cpb
        for j in range_constexpr(ROWS):
            csj = starts[j] // cpb
            cb_start = (csj < cb_start).select(csj, cb_start)
            cej = (ends[j] + fx.Int32(COLS_PER_BLOCK - 1)) // cpb
            cb_end = (cej > cb_end).select(cej, cb_end)
        n_iter = cb_end - cb_start

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        kv_sh = lds.kv.view(fx.make_layout(WAVES * KV_SLAB_I64, 1))
        wave_slab = wave * fx.Int32(KV_SLAB_I64)

        for cbi, _state in range(0, fx.arith._to_raw(n_iter), 1, init=[fx.Int32(0)]):
            cb = cb_start + _iv_to_i32(cbi)
            nt0 = (cb * fx.Int32(WAVES) + wave) * fx.Int32(TPW)

            # ---- stage this column tile's KV into LDS ONCE (shared by rows) ----
            kv_global_base = nt0 * fx.Int32(16) * ROW_I64
            for e in range_constexpr(NSTAGE):
                idx = lane + fx.Int32(e * WAVE)
                if const_expr(KV_SLAB_I64 % WAVE != 0):
                    if idx < fx.Int32(KV_SLAB_I64):
                        v = buffer_ops.buffer_load(kv_rsrc, kv_global_base + idx,
                                                   vec_width=1, dtype=fx.Int64)
                        fx.memref_store(v, kv_sh, wave_slab + idx)
                else:
                    v = buffer_ops.buffer_load(kv_rsrc, kv_global_base + idx,
                                               vec_width=1, dtype=fx.Int64)
                    fx.memref_store(v, kv_sh, wave_slab + idx)
            if const_expr(_USE_LDS_BARRIER):
                gpu.barrier()

            # B fragments + columns + per-column kv scale (shared across rows).
            b_frag = [[None] * K_STEPS for _ in range(TPW)]
            cols = [None] * TPW
            kvsc = [None] * TPW
            for tp in range_constexpr(TPW):
                cols[tp] = (nt0 + fx.Int32(tp)) * fx.Int32(16) + lane_row
                kvsc[tp] = fx.memref_load(ks_buf, cols[tp])
                col_in_slab = fx.Int32(tp * 16) + lane_row
                for ks in range_constexpr(K_STEPS):
                    chunk = fx.Int32(ks * 4) + lane_kg
                    slab_idx = wave_slab + col_in_slab * ROW_I64 + chunk
                    b_frag[tp][ks] = fx.memref_load(kv_sh, slab_idx)

            # ---- per row: MFMA (reusing staged KV) + reduce + store ----
            for j in range_constexpr(ROWS):
                accs = [[None] * M_TILES for _ in range(TPW)]
                for tp in range_constexpr(TPW):
                    for mt in range_constexpr(M_TILES):
                        accs[tp][mt] = arith.constant_vector(0.0, T.f32x4)
                for ks in range_constexpr(K_STEPS):
                    for tp in range_constexpr(TPW):
                        for mt in range_constexpr(M_TILES):
                            accs[tp][mt] = rocdl.mfma_f32_16x16x32_fp8_fp8(
                                T.f32x4,
                                [fx.arith._to_raw(q_frag[j][mt][ks]),
                                 fx.arith._to_raw(b_frag[tp][ks]),
                                 fx.arith._to_raw(accs[tp][mt]), 0, 0, 0])
                for tp in range_constexpr(TPW):
                    col = cols[tp]
                    in_window = (col >= starts[j]) & (col < ends[j])
                    partial = fx.Float32(0.0)
                    for mt in range_constexpr(M_TILES):
                        accv = fx.Vector(accs[tp][mt])
                        for i in range_constexpr(4):
                            sc = accv[i] * kvsc[tp]
                            sc = sc.maximumf(fx.Float32(0.0))
                            sc = sc * wvals[j][mt][i]
                            partial = partial + sc
                    r = partial
                    r = r + r.shuffle_xor(16, WAVE)
                    r = r + r.shuffle_xor(32, WAVE)
                    if lane_kg == fx.Int32(0):
                        if in_window:
                            fx.memref_store(r, OUT_buf, out_base[j] + col)

            # Per-wave-private slab: next iteration's staging waits on this
            # iteration's reads via the compiler's lds waitcnt (no barrier).
            if const_expr(_USE_LDS_BARRIER):
                gpu.barrier()
            _next = yield [_state[0] + fx.Int32(1)]

    @flyc.jit
    def launch(
        Q: fx.Tensor,
        KV: fx.Tensor,
        KVSCALE: fx.Tensor,
        W: fx.Tensor,
        STARTS: fx.Tensor,
        ENDS: fx.Tensor,
        OUT: fx.Tensor,
        n_blocks: int,
        stream: fx.Stream = fx.Stream(None),
    ):
        l = mqa_logits_kernel(Q, KV, KVSCALE, W, STARTS, ENDS, OUT)
        l.launch(grid=(n_blocks, 1, 1), block=(_GEAK_BLOCK_THREADS, 1, 1), stream=stream)

    return launch


class _GeakLauncher:
    """Registry-compatible adapter around the standalone ``build_mqa_logits``.

    The GEAK kernel was written with its own ABI: it takes ``Q``/``KV`` as i64
    views (8 fp8 packed per i64), flattened ``W``/``OUT`` (``M*NUM_HEADS`` /
    ``M*S_K``), per-row ``STARTS``/``ENDS``, and a ``n_blocks`` launch argument;
    it also bakes ``S_K`` into the kernel (it must be a multiple of
    ``COLS_PER_BLOCK``) and assumes one block per ``ROWS_PER_BLOCK`` query rows.

    The rest of this module (``flydsl_fp8_mqa_logits`` -> ``_run_compiled``)
    drives every variant through the common ABI

        launcher(Q, KV, kv_scales, weights, cu_starts, cu_ends, logits,
                 seq_len, seq_len_kv, out_stride, stream)

    with raw fp8 ``Q``/``KV`` and a (possibly non-contiguous) ``logits`` slice.
    This adapter bridges the two: it reinterprets the fp8 tensors as i64,
    flattens the f32 inputs, pads ``S_K``/rows to the kernel's tiling, runs the
    GEAK kernel into a contiguous padded scratch buffer, and copies the valid
    region back into ``logits``. One compiled ``launch`` is cached per padded
    ``S_K`` (``S_K`` is a compile-time constant of the GEAK kernel).
    """

    # Tells flydsl_fp8_mqa_logits to invoke us directly (we own the GEAK ABI &
    # compilation) instead of routing through _run_compiled.
    _is_adapter = True

    def __init__(self, *, num_heads: int, head_size: int):
        # The GEAK kernel hard-codes NUM_HEADS=64 / HEAD_DIM=128 (and the MFMA
        # fragment + LDS layout depend on it), so reject other shapes loudly
        # rather than silently miscomputing.
        if num_heads != NUM_HEADS or head_size != HEAD_DIM:
            raise NotImplementedError(
                f"geak fp8_mqa_logits variant only supports "
                f"num_heads={NUM_HEADS}, head_size={HEAD_DIM}; "
                f"got num_heads={num_heads}, head_size={head_size}."
            )
        self.num_heads = num_heads
        self.head_size = head_size
        self.compile_hints = None
        self._by_skv = {}  # padded_S_K -> compiled geak launch

    def _launch_for(self, padded_skv):
        launch = self._by_skv.get(padded_skv)
        if launch is None:
            launch = build_mqa_logits(padded_skv)
            if self.compile_hints is not None:
                launch.compile_hints = dict(self.compile_hints)
            self._by_skv[padded_skv] = launch
        return launch

    def __call__(
        self,
        Q,
        KV,
        kv_scales,
        weights,
        cu_starts,
        cu_ends,
        logits,
        seq_len,
        seq_len_kv,
        out_stride,
        stream,
    ):
        # Pad S_K up to the kernel's column-block granularity, and rows up to
        # ROWS_PER_BLOCK so the grid covers every query row.
        padded_skv = (
            (seq_len_kv + COLS_PER_BLOCK - 1) // COLS_PER_BLOCK * COLS_PER_BLOCK
        )
        padded_rows = (
            (seq_len + ROWS_PER_BLOCK - 1) // ROWS_PER_BLOCK * ROWS_PER_BLOCK
        )
        n_blocks = padded_rows // ROWS_PER_BLOCK

        device = Q.device
        H = self.num_heads
        aligned = padded_skv == seq_len_kv and padded_rows == seq_len

        # fp8 -> i64 views (8 fp8 per i64). Both operands must be contiguous so
        # the flat i64 reinterpretation matches the kernel's index math.
        # ``.contiguous()`` is a no-op when the input already is (the common
        # case), so this does not copy on the hot path.
        q_i64 = Q.contiguous().view(torch.int64).reshape(-1)
        kv_c = KV.contiguous()
        if padded_skv != seq_len_kv:
            kv_pad = torch.empty(
                (padded_skv, self.head_size), dtype=KV.dtype, device=device
            )
            kv_pad[:seq_len_kv] = kv_c  # tail rows masked out by window test
            kv_c = kv_pad
        kv_i64 = kv_c.view(torch.int64).reshape(-1)

        # kv_scales padded to padded_skv (padding columns are masked out by the
        # window test, so their scale value is irrelevant).
        if padded_skv != seq_len_kv:
            ks = torch.empty(padded_skv, dtype=torch.float32, device=device)
            ks[:seq_len_kv] = kv_scales
        else:
            ks = kv_scales.contiguous()

        # weights flattened to [M*NUM_HEADS]; pad rows if needed.
        w = weights.contiguous()
        if padded_rows != seq_len:
            w_pad = torch.empty((padded_rows, H), dtype=torch.float32, device=device)
            w_pad[:seq_len] = w
            w = w_pad
        w = w.reshape(-1)

        # starts/ends padded to padded_rows. Padding rows get an empty window
        # (start==end==0) so they store nothing.
        st = cu_starts.contiguous()
        en = cu_ends.contiguous()
        if padded_rows != seq_len:
            st_pad = torch.zeros(padded_rows, dtype=torch.int32, device=device)
            en_pad = torch.zeros(padded_rows, dtype=torch.int32, device=device)
            st_pad[:seq_len] = st
            en_pad[:seq_len] = en
            st, en = st_pad, en_pad

        launch = self._launch_for(padded_skv)

        # Fast path: when no S_K/row padding is needed AND the caller's ``logits``
        # is already contiguous with row stride == padded_skv, the kernel can
        # write straight into it -- the kernel only touches each row's window and
        # the host already prefilled the rest (-inf for clean_logits). This
        # avoids a full-size scratch alloc + two full-size copies per call, which
        # otherwise added ~20-30% to the measured time.
        if aligned and logits.is_contiguous() and logits.stride(0) == padded_skv:
            _run_compiled(
                launch, q_i64, kv_i64, ks, w, st, en, logits.reshape(-1),
                int(n_blocks), stream,
            )
            return logits

        # Slow path (S_K/row padding or a non-contiguous logits slice): run into
        # a contiguous padded scratch, then copy the valid region back. The
        # untouched scratch cells must match the caller's logits (-inf for
        # clean_logits), so seed them; the kernel overwrites the in-window cells.
        out = torch.empty(
            (padded_rows, padded_skv), dtype=torch.float32, device=device
        )
        out[:seq_len, :seq_len_kv] = logits
        _run_compiled(
            launch, q_i64, kv_i64, ks, w, st, en, out.reshape(-1),
            int(n_blocks), stream,
        )
        logits.copy_(out[:seq_len, :seq_len_kv])
        return logits


def _build_kernel_geak(*, num_heads: int, head_size: int, block_kv: int):
    """Registry builder for the GEAK-optimized variant.

    Returns a ``_GeakLauncher`` adapter exposing the common variant ABI (see the
    class docstring). ``block_kv`` is accepted for signature compatibility but
    the GEAK kernel uses its own internal column-block tiling.
    """
    return _GeakLauncher(num_heads=num_heads, head_size=head_size)


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
    "geak": _build_kernel_geak,
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
    launcher = builder(
        num_heads=num_heads, head_size=head_size, block_kv=block_kv
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

    variant = _resolve_variant(variant)
    launcher = compile_fp8_mqa_logits(
        num_heads=num_heads,
        head_size=head_size,
        block_kv=128,
        paged=False,
        variant=variant,
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
        # Adapter launchers (e.g. the GEAK variant) implement the common ABI
        # directly and own their own compilation; the plain @flyc.jit launchers
        # are compiled+dispatched via _run_compiled.
        run = launcher if getattr(launcher, "_is_adapter", False) else _run_compiled
        args = () if run is launcher else (launcher,)
        run(
            *args,
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
