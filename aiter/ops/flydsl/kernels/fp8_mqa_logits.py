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
from functools import lru_cache

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, range_constexpr, rocdl
from flydsl.expr.numeric import ArithValue
from flydsl.expr.typing import T
from flydsl._mlir.dialects import scf
from flydsl._mlir import ir

from .tensor_shim import GTensor, _run_compiled, _to_raw

Vec = fx.Vector


def _i32_add(a, b):
    """i32 add (result stays Int32, not index type)."""
    return fx.Int32(arith.addi(_to_raw(a), _to_raw(b)))


# Default KV tile width (columns processed per inner-loop iteration).
_BLOCK_KV = 128
# Don't split a row's KV window into chunks smaller than this many BKV tiles --
# below it the per-block Q/weight preload stops being amortized.
_MIN_TILES_PER_SPLIT = 8

_DEFAULT_COMPILE_HINTS = {
    "waves_per_eu": 2,
    "fast_fp_math": True,
}


@lru_cache(maxsize=8)
def _device_cu_count(device_index: int) -> int:
    """Compute-unit count for a CUDA/HIP device (cached); 304 if unavailable."""
    try:
        return torch.cuda.get_device_properties(device_index).multi_processor_count
    except Exception:
        return 304


def _auto_num_splits(seq_len_padded: int, seq_len_kv: int, rpb: int,
                     device_index: int) -> int:
    """KV-column splits (grid.y) to fill the device when the row grid is small.

    For small-M / large-N shapes the ``ceil(seq_len/RPB)`` row grid leaves the
    device block-starved; splitting each row's window across ``grid.y`` recovers
    occupancy at no correctness cost (logits[m,n] are independent across n).
    Returns 1 once the row grid alone oversubscribes the device. Constants tuned
    on MI300X (304 CU): ~4x oversubscription, chunks >= _MIN_TILES_PER_SPLIT.
    """
    grid_x = seq_len_padded // rpb
    if grid_x == 0 or seq_len_kv < 4096:
        return 1
    target_blocks = 4 * _device_cu_count(device_index)
    if grid_x >= target_blocks:
        return 1
    max_splits = max(1, (seq_len_kv // _BLOCK_KV) // _MIN_TILES_PER_SPLIT)
    return max(1, min(math.ceil(target_blocks / grid_x), max_splits))


def _build_kernel_mfma_r_w(*, num_heads: int, head_size: int, block_kv: int,
                           rows_per_block: int, waves_per_block: int,
                           convert_q_fn: bool = False, convert_kv_fn: bool = False):
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

    assert H % 16 == 0, f"num_heads={H} must be a multiple of 16 for MFMA"
    assert BKV % 16 == 0, f"block_kv={BKV} must be a multiple of 16 for MFMA"
    assert D % 32 == 0, f"head_size={D} must be a multiple of 32 for MFMA K32"
    assert RPB >= 1, "rows_per_block must be >= 1"
    assert WPB >= 1, "waves_per_block must be >= 1"
    N_TILES = BKV // 16                       # total column-tiles per BKV block
    assert N_TILES % WPB == 0, (
        f"BKV/16={N_TILES} must be divisible by waves_per_block={WPB}"
    )
    M_TILES = H // 16                         # head row-tiles
    K_STEPS = D // 32                         # K32 MFMA steps over the head dim
    N_TILES_PER_WAVE = N_TILES // WPB         # column-tiles per wave

    fm_fast = arith.FastMathFlags.fast
    mfma_fn = rocdl.mfma_f32_16x16x32_fp8_fp8

    _cvt_tag = ""
    if convert_q_fn:
        _cvt_tag += "_cq"
    if convert_kv_fn:
        _cvt_tag += "_ck"
    _kname = f"fp8_mqa_logits_H{H}_D{D}_bkv{BKV}_mfma_r{RPB}_w{WPB}{_cvt_tag}_flydsl"

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
        num_splits: fx.Int32,    # grid.y KV-column splits (1 == no split)
    ):
        f32_0 = arith.constant(0.0, type=T.f32)
        mfma_res_ty = Vec.make_type(4, fx.Float32)

        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        # Block bid (reversed) owns rows [r0, r0+RPB).
        n_blocks = fx.Int32(arith.ceildivui(_to_raw(seq_len), _to_raw(fx.Int32(RPB))))
        r0 = fx.Int32(arith.muli(
            _to_raw(n_blocks - bid - fx.Int32(1)),
            _to_raw(fx.Int32(RPB)),
        ))

        # Decompose tid into wave index and in-wave lane.
        wave = fx.Int32(arith.divui(_to_raw(tid), _to_raw(fx.Int32(64))))
        lane = fx.Int32(arith.remui(_to_raw(tid), _to_raw(fx.Int32(64))))
        lane_div_16 = fx.Int32(arith.divui(_to_raw(lane), _to_raw(fx.Int32(16))))
        lane_mod_16 = fx.Int32(arith.remui(_to_raw(lane), _to_raw(fx.Int32(16))))
        lane8 = fx.Int32(arith.muli(_to_raw(lane_div_16), _to_raw(fx.Int32(8))))

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

            # lane -> Q[row, h = mi*16 + lane%16, d = kk*32 + (lane//16)*8 + 0..7]
            row_a = [[None] * K_STEPS for _ in range_constexpr(M_TILES)]
            for mi in range_constexpr(M_TILES):
                h_a = _i32_add(fx.Int32(mi * 16), lane_mod_16)
                row_h = _i32_add(fx.Int32(arith.muli(_to_raw(row), _to_raw(fx.Int32(H)))), h_a)
                base_a = fx.Int32(arith.muli(_to_raw(row_h), _to_raw(fx.Int32(D))))
                for kk in range_constexpr(K_STEPS):
                    d_a = _i32_add(fx.Int32(kk * 32), lane8)
                    raw = _load_pack_i64(q_i32, _i32_add(base_a, d_a))
                    row_a[mi][kk] = _fn_to_fnuz_i64(raw) if convert_q_fn else raw
            a_packs[j] = row_a

            # weights[row, h] preloaded per (mi, ii): head = mi*16 + lane_div_16*4 + ii
            row_w = [[None] * 4 for _ in range_constexpr(M_TILES)]
            for mi in range_constexpr(M_TILES):
                for ii in range_constexpr(4):
                    h_w = _i32_add(
                        fx.Int32(mi * 16),
                        _i32_add(
                            fx.Int32(arith.muli(_to_raw(lane_div_16), _to_raw(fx.Int32(4)))),
                            fx.Int32(ii),
                        ),
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

        tile_lo = _to_raw(fx.Index(fx.Int32(tile_start)))
        tile_hi = _to_raw(fx.Index(fx.Int32(tile_end)))
        tile_step = _to_raw(fx.Index(fx.Int32(BKV)))
        tile_loop = scf.ForOp(tile_lo, tile_hi, tile_step, [])
        with ir.InsertionPoint(tile_loop.body):
            col0 = fx.Int32(arith.index_cast(T.i32, tile_loop.induction_variable))

            # ---- Load B-frags: wave w owns its own disjoint slice of n-tiles
            # [w*N_TILES_PER_WAVE, (w+1)*N_TILES_PER_WAVE) (no cross-wave sharing). ----
            wave_ni_base = fx.Int32(arith.muli(_to_raw(wave), _to_raw(fx.Int32(N_TILES_PER_WAVE))))
            b_packs = [[None] * K_STEPS for _ in range_constexpr(N_TILES_PER_WAVE)]
            kv_scales_tile = [None] * N_TILES_PER_WAVE
            cols = [None] * N_TILES_PER_WAVE
            for ni in range_constexpr(N_TILES_PER_WAVE):
                abs_ni = _i32_add(wave_ni_base, fx.Int32(ni))
                col = _i32_add(
                    _i32_add(col0, fx.Int32(arith.muli(_to_raw(abs_ni), _to_raw(fx.Int32(16))))),
                    lane_mod_16,
                )
                cols[ni] = col
                col_clamped = fx.Int32(
                    arith.minsi(_to_raw(col), _to_raw(fx.Int32(seq_len_kv) - fx.Int32(1)))
                )
                kv_scales_tile[ni] = _to_raw(fx.Float32(sc_t[col_clamped]))
                base_b = fx.Int32(arith.muli(_to_raw(col_clamped), _to_raw(fx.Int32(D))))
                for kk in range_constexpr(K_STEPS):
                    d_b = _i32_add(fx.Int32(kk * 32), lane8)
                    raw = _load_pack_i64(kv_i32, _i32_add(base_b, d_b))
                    b_packs[ni][kk] = _fn_to_fnuz_i64(raw) if convert_kv_fn else raw

            # ---- Per-row MFMA + epilogue (inner loop over RPB rows) ----
            for j in range_constexpr(RPB):
                row = _i32_add(r0, fx.Int32(j))
                for ni in range_constexpr(N_TILES_PER_WAVE):
                    col = cols[ni]
                    kv_scale = kv_scales_tile[ni]
                    col_sum = _to_raw(f32_0)
                    for mi in range_constexpr(M_TILES):
                        acc = Vec.filled(4, 0.0, fx.Float32)
                        for kk in range_constexpr(K_STEPS):
                            acc = mfma_fn(
                                mfma_res_ty,
                                [a_packs[j][mi][kk], b_packs[ni][kk], acc, 0, 0, 0],
                            )
                        # kv_scale (>=0) is hoisted out of the head sum: ReLU is
                        # positive-homogeneous, so ReLU(s*x)=s*ReLU(x) and the
                        # whole column sum is scaled once (below) instead of every
                        # head term -- drops M_TILES*4 muls to one.
                        for ii in range_constexpr(4):
                            score = Vec(acc)[ii].ir_value()
                            relu = arith.maximumf(score, _to_raw(f32_0))
                            wsc = arith.MulFOp(relu, w_frag[j][mi][ii], fastmath=fm_fast).result
                            col_sum = arith.AddFOp(col_sum, wsc, fastmath=fm_fast).result
                    col_sum = arith.MulFOp(col_sum, kv_scale, fastmath=fm_fast).result

                    # Head-reduce within the wave (width=64): shuffle_xor offsets 16, 32.
                    for sh in [16, 32]:
                        peer = _to_raw(ArithValue(col_sum).shuffle_xor(sh, 64))
                        col_sum = arith.AddFOp(col_sum, peer, fastmath=fm_fast).result

                    # Only lane_div_16==0 lanes hold the 16 distinct columns.
                    # `col >= start` is required: the tile loop is BKV-aligned
                    # below `start`, so it guards the pre-filled -inf in
                    # [aligned_start, start).
                    in_window = arith.andi(
                        _to_raw(arith.cmpi(
                            arith.CmpIPredicate.sge,
                            _to_raw(col),
                            _to_raw(starts[j]),
                        )),
                        _to_raw(arith.cmpi(
                            arith.CmpIPredicate.slt,
                            _to_raw(col),
                            _to_raw(ends[j]),
                        )),
                    )
                    is_writer = arith.andi(
                        _to_raw(arith.cmpi(
                            arith.CmpIPredicate.eq,
                            _to_raw(lane_div_16),
                            _to_raw(fx.Int32(0)),
                        )),
                        in_window,
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
        num_splits: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        n_blocks = arith.ceildivui(
            _to_raw(seq_len), _to_raw(fx.Int32(RPB))
        )
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


# Kernel variants are tagged ``"mfma_r<RPB>_w<WPB>"`` (RPB query rows per block,
# WPB waves per block). All share the single ``_build_kernel_mfma_r_w`` factory.
# WPB must divide the column-tile count BKV/16 (=8 at the default BKV=128).
def _mk_builder(rpb, wpb):
    return lambda **kw: _build_kernel_mfma_r_w(
        **kw, rows_per_block=rpb, waves_per_block=wpb)


_VARIANT_BUILDERS = {
    f"mfma_r{r}_w{w}": _mk_builder(r, w)
    for r in (1, 2, 4)
    for w in (1, 2, 4)
}
KERNEL_VARIANTS = tuple(_VARIANT_BUILDERS.keys())
DEFAULT_VARIANT = "mfma_r2_w4"


def _auto_variant(seq_len, seq_len_kv):
    """Pick (RPB, WPB) from the problem shape: RPB=2 always; WPB=2 packs more
    column tiles per wave when M and N are both large, else WPB=4 for more
    wavefronts on small-M / short-window shapes."""
    wpb = 2 if (seq_len >= 2048 and seq_len_kv >= 8192) else 4
    return f"mfma_r2_w{wpb}"


def _resolve_variant(variant, seq_len, seq_len_kv):
    """Effective variant: explicit ``variant=`` > env var > shape-adaptive."""
    tag = variant or os.environ.get("FLYDSL_FP8_MQA_LOGITS_VARIANT") \
        or _auto_variant(seq_len, seq_len_kv)
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
    block_kv: int = _BLOCK_KV,
    paged: bool = False,
    variant: str = DEFAULT_VARIANT,
    convert_q_fn: bool = False,
    convert_kv_fn: bool = False,
):
    """Return a cached, compiled FlyDSL launcher for the given shape config.

    ``num_heads``/``head_size`` are compile-time constants (powers of two, D in
    {64, 128}); ``variant`` is an ``mfma_r<RPB>_w<WPB>`` tag (see
    ``KERNEL_VARIANTS``); ``convert_q_fn``/``convert_kv_fn`` mark an FP8 FN
    operand whose -0 (0x80) byte the kernel patches to FNUZ +0. ``paged`` is
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
        num_heads=num_heads, head_size=head_size, block_kv=block_kv,
        convert_q_fn=convert_q_fn, convert_kv_fn=convert_kv_fn,
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
    assert Q.dtype in (_fnuz, _fn) and KV.dtype in (_fnuz, _fn), (
        f"Q/KV must be e4m3 fp8 (fnuz or fn); got {Q.dtype}, {KV.dtype}"
    )
    convert_q_fn = Q.dtype != _fnuz
    convert_kv_fn = KV.dtype != _fnuz
    scale_mul = (2.0 if convert_q_fn else 1.0) * (2.0 if convert_kv_fn else 1.0)
    if scale_mul != 1.0:
        kv_scales = kv_scales.to(torch.float32) * scale_mul

    variant = _resolve_variant(variant, seq_len, seq_len_kv)

    launcher = compile_fp8_mqa_logits(
        num_heads=num_heads,
        head_size=head_size,
        block_kv=_BLOCK_KV,
        paged=False,
        variant=variant,
        convert_q_fn=convert_q_fn,
        convert_kv_fn=convert_kv_fn,
    )

    # mfma_r* kernels require seq_len padded to a multiple of rows_per_block so
    # every block owns exactly RPB rows.  Padded rows get empty windows (start ==
    # end == 0) so the kernel writes nothing for them; the output is sliced back
    # to the original seq_len after the launch.
    # Parse RPB from variant tag "mfma_r<N>_w<M>" -> N.
    _rpb_match = re.match(r"mfma_r(\d+)", variant)
    _RPB = int(_rpb_match.group(1)) if _rpb_match else 1
    seq_len_padded = ((seq_len + _RPB - 1) // _RPB) * _RPB
    if seq_len_padded != seq_len:
        pad = seq_len_padded - seq_len
        Q = torch.cat([Q, Q.new_zeros((pad, num_heads, head_size))], dim=0)
        weights = torch.cat([weights, weights.new_zeros((pad, num_heads))], dim=0)
        cu_starts = torch.cat([cu_starts, cu_starts.new_zeros(pad)], dim=0)
        cu_ends = torch.cat([cu_ends, cu_ends.new_zeros(pad)], dim=0)

    # Match the Triton launcher's -inf-prefill / padding behavior so the two
    # produce identically-shaped, identically-masked outputs. The kernel indexes
    # logits with i32, so seq_len_padded * stride must stay < 2^31 (dense output
    # ceiling ~46k square); real serving never materializes logits that large.
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

    num_splits = _auto_num_splits(seq_len_padded, seq_len_kv, _RPB, Q.device.index)

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
