# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FP8 MQA logits (DeepSeek lightning indexer) -- FlyDSL gfx942 kernel.

For each query row ``m`` and KV position ``n`` in ``[cu_starts[m], cu_ends[m])``::

    logits[m, n] = sum_h ReLU(<Q[m, h, :], K[n, :]> * kv_scale[n]) * weights[m, h]

``flydsl_fp8_mqa_logits`` is a drop-in for the Triton
``aiter.ops.triton.attention.fp8_mqa_logits.fp8_mqa_logits``.
"""

# No `from __future__ import annotations`: FlyDSL arg typing needs real
# annotation objects, not PEP 563 strings.
import math
import os
import re
from functools import lru_cache

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr import range_constexpr, rocdl
from flydsl.expr.typing import T

from aiter.jit.utils.chip_info import get_gfx

from .tensor_shim import GTensor, _run_compiled

Vec = fx.Vector


def _imax(a, b):
    a = fx.Int32(a)
    b = fx.Int32(b)
    return (a >= b).select(a, b)


def _imin(a, b):
    a = fx.Int32(a)
    b = fx.Int32(b)
    return (a <= b).select(a, b)


def _uceildiv(a, b):
    a = fx.Uint32(a)
    b = fx.Uint32(b)
    return fx.Int32((a + b - fx.Uint32(1)) // b)


_BLOCK_KV = 128  # KV columns per inner-loop iteration
# Min BKV tiles per KV-split chunk; below it the per-block Q/weight preload stops
# being amortized.
_MIN_TILES_PER_SPLIT = 8

_DEFAULT_COMPILE_HINTS = {"waves_per_eu": 2, "fast_fp_math": True}


@lru_cache(maxsize=8)
def _device_cu_count(device_index: int) -> int:
    """Compute-unit count for a CUDA/HIP device (cached); 304 if unavailable."""
    try:
        return torch.cuda.get_device_properties(device_index).multi_processor_count
    except Exception:  # noqa: BLE001
        return 304


def _auto_num_splits(
    seq_len_padded: int, seq_len_kv: int, rpb: int, device_index: int
) -> int:
    """KV-column splits (grid.y) to fill the device when the row grid is small.

    logits[m,n] are independent across n, so splitting each row's window across
    grid.y is pure parallelism. Returns 1 once the row grid alone oversubscribes.
    Constants tuned on MI300X (304 CU): ~4x oversubscription, chunks >= _MIN_TILES_PER_SPLIT.
    """
    grid_x = seq_len_padded // rpb
    if grid_x == 0 or seq_len_kv < 4096:
        return 1
    target_blocks = 4 * _device_cu_count(device_index)
    if grid_x >= target_blocks:
        return 1
    max_splits = max(1, (seq_len_kv // _BLOCK_KV) // _MIN_TILES_PER_SPLIT)
    return max(1, min(math.ceil(target_blocks / grid_x), max_splits))


def _build_kernel_mfma_r_w(
    *,
    num_heads: int,
    head_size: int,
    block_kv: int,
    rows_per_block: int,
    waves_per_block: int,
    convert_q_fn: bool = False,
    convert_kv_fn: bool = False,
):
    """Multi-row, multi-wave MFMA kernel.

    RPB query rows share one KV tile load (cuts KV traffic by RPB). WPB waves per
    block each own a disjoint slice of the BKV column tiles (N_TILES // WPB tiles
    per wave), so they run in parallel with no cross-wave LDS or barrier.
    ``tid = wave * 64 + lane``; A-operand (Q) layout and head-reduce are per-lane
    within the wave (width 64).

    Grid ``(ceil(seq_len / RPB), num_splits, 1)``; host pads seq_len to a multiple
    of RPB and may split each row's KV window across grid.y (see
    ``flydsl_fp8_mqa_logits``).
    """
    H = num_heads
    D = head_size
    BKV = block_kv
    RPB = rows_per_block
    WPB = waves_per_block
    MR_BLOCK_THREADS = 64 * WPB

    # fp8 16x16x32 MFMA atom: MFMA_M x MFMA_N output tile, MFMA_K reduced per step.
    MFMA_M = 16
    MFMA_N = 16
    MFMA_K = 32

    assert H % MFMA_M == 0, f"num_heads={H} must be a multiple of MFMA_M={MFMA_M}"
    assert BKV % MFMA_N == 0, f"block_kv={BKV} must be a multiple of MFMA_N={MFMA_N}"
    assert D % MFMA_K == 0, f"head_size={D} must be a multiple of MFMA_K={MFMA_K}"
    assert RPB >= 1, "rows_per_block must be >= 1"
    assert WPB >= 1, "waves_per_block must be >= 1"
    N_TILES = BKV // MFMA_N  # total column-tiles per BKV block
    assert (
        N_TILES % WPB == 0
    ), f"BKV/MFMA_N={N_TILES} must be divisible by waves_per_block={WPB}"
    M_TILES = H // MFMA_M  # head row-tiles
    K_STEPS = D // MFMA_K  # MFMA K-steps over the head dim
    N_TILES_PER_WAVE = N_TILES // WPB  # column-tiles per wave

    mfma_fn = rocdl.mfma_f32_16x16x32_fp8_fp8

    _cvt_tag = ""
    if convert_q_fn:
        _cvt_tag += "_cq"
    if convert_kv_fn:
        _cvt_tag += "_ck"
    _kname = f"fp8_mqa_logits_H{H}_D{D}_bkv{BKV}_mfma_r{RPB}_w{WPB}{_cvt_tag}_flydsl"

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
        f32_0 = fx.Float32(0.0)
        mfma_res_ty = Vec.make_type(4, fx.Float32)

        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        # Block bid (reversed) owns rows [r0, r0+RPB).
        n_blocks = _uceildiv(seq_len, fx.Int32(RPB))
        r0 = (n_blocks - fx.Int32(bid) - 1) * RPB

        wave = fx.Int32(fx.Uint32(tid) // 64)
        lane = fx.Uint32(tid) % 64
        lane_div_N = fx.Int32(lane // MFMA_N)
        lane_mod_N = fx.Int32(lane % MFMA_N)
        lane8 = lane_div_N * 8

        # fp8 operands are read 8 bytes at a time as 2 i32 dwords (v8i8
        # buffer_load fails to lower on gfx942), bitcast to i64 for the MFMA.
        q_i32 = GTensor(Q, dtype=T.i32, shape=(-1,))
        kv_i32 = GTensor(KV, dtype=T.i32, shape=(-1,))
        sc_t = GTensor(kv_scales, dtype=T.f32, shape=(-1,))
        w_t = GTensor(weights, dtype=T.f32, shape=(-1, H))
        cs_t = GTensor(cu_starts, dtype=T.i32, shape=(-1,))
        ce_t = GTensor(cu_ends, dtype=T.i32, shape=(-1,))
        # Per-row 1-D output view: the row's i64 byte offset goes in the base
        # pointer so the column offset stays i32. A 2-D view computes row*stride+col
        # in i32 and overflows past 2^31 (~46k-square outputs), mis-writing.
        # Zero-extend (row/stride non-negative): a sign-extended i64 mul would emit
        # extra scalar sign-extension ops for the row*stride*4 base.
        stride_i64 = fx.Int64(fx.Uint32(stride_logits_s))

        def _make_out_row_t(row_i32):
            byte = fx.Int64(fx.Uint32(row_i32)) * stride_i64 * 4
            return GTensor(
                logits, dtype=T.f32, shape=(-1,), static_bytes_offset_i64=byte
            )

        def _load_pack_i64(i32_view, byte_off_i32):
            dword_off = fx.Int32(byte_off_i32) // 4
            v2 = i32_view.vec_load((dword_off,), vec_size=2)
            return Vec(v2).bitcast(fx.Int64)[0].ir_value()

        def _fn_to_fnuz_i64(raw_i64):
            # Map FN byte 0x80 (neg-zero, = FNUZ NaN) -> 0x00 in 8 packed fp8 bytes.
            raw = fx.Int64(raw_i64)
            lo_i32 = fx.Int32(raw)
            hi_i32 = fx.Int32(raw.shrui(32))

            def _fix_i32(src):
                result = fx.Int32(0)
                for byte_idx in range_constexpr(4):
                    shift = byte_idx * 8
                    byte_val = src.shrui(shift) & 0xFF
                    is_0x80 = byte_val == 0x80
                    cleaned = is_0x80.select(fx.Int32(0), byte_val)
                    result = result | (cleaned << shift)
                return result

            lo_64 = fx.Int64(_fix_i32(lo_i32))
            hi_64 = fx.Int64(_fix_i32(hi_i32)) << 32
            return (lo_64 | hi_64).ir_value()

        # Preload window bounds, Q frags, and weights for all RPB rows.
        # A-operand layout is per in-wave lane, so `lane` (not `tid`) indexes Q.
        starts = [None] * RPB
        ends = [None] * RPB
        a_packs = [None] * RPB
        w_frag = [None] * RPB
        for j in range_constexpr(RPB):
            row = r0 + j
            s = fx.Int32(cs_t[row])
            e = fx.Int32(ce_t[row])
            starts[j] = _imax(s, fx.Int32(0))
            ends[j] = _imin(e, fx.Int32(seq_len_kv))

            # lane -> Q[row, h = mi*MFMA_M + lane%MFMA_N,
            #            d = kk*MFMA_K + (lane//MFMA_N)*8 + 0..7]
            row_a = [[None] * K_STEPS for _ in range_constexpr(M_TILES)]
            for mi in range_constexpr(M_TILES):
                h_a = mi * MFMA_M + lane_mod_N
                row_h = row * H + h_a
                base_a = row_h * D
                for kk in range_constexpr(K_STEPS):
                    d_a = kk * MFMA_K + lane8
                    raw = _load_pack_i64(q_i32, base_a + d_a)
                    row_a[mi][kk] = _fn_to_fnuz_i64(raw) if convert_q_fn else raw
            a_packs[j] = row_a

            # weights[row, h] per (mi, ii): head = mi*MFMA_M + lane_div_N*4 + ii
            row_w = [[None] * 4 for _ in range_constexpr(M_TILES)]
            for mi in range_constexpr(M_TILES):
                for ii in range_constexpr(4):
                    h_w = mi * MFMA_M + lane_div_N * 4 + ii
                    row_w[mi][ii] = fx.Float32(w_t[row, h_w])
            w_frag[j] = row_w

        # Union window across all RPB rows, aligned down to a BKV boundary.
        tile_start = starts[0]
        tile_end = ends[0]
        for j in range_constexpr(1, RPB):
            tile_start = _imin(tile_start, starts[j])
            tile_end = _imax(tile_end, ends[j])
        tile_start = fx.Int32(fx.Uint32(tile_start) // BKV) * BKV

        # KV-column split across grid.y: block (.,by) takes a BKV-aligned slice of
        # the union window. Slices tile [start,end) disjoint + gap-free, so each
        # column has one writer; num_splits==1 collapses to the full window.
        by = fx.block_idx.y
        win_tiles = _uceildiv(tile_end - tile_start, fx.Int32(BKV))
        split_cols = _uceildiv(win_tiles, num_splits) * BKV
        tile_start = tile_start + fx.Int32(by) * split_cols
        tile_end = _imin(tile_start + split_cols, tile_end)

        tile_lo = fx.Int32(tile_start)
        tile_hi = fx.Int32(tile_end)
        for col0 in range(tile_lo, tile_hi, fx.Int32(BKV)):
            # Load B-frags: wave w owns its own disjoint n-tile slice
            # [w*N_TILES_PER_WAVE, (w+1)*N_TILES_PER_WAVE), no cross-wave sharing.
            wave_ni_base = wave * N_TILES_PER_WAVE
            b_packs = [[None] * K_STEPS for _ in range_constexpr(N_TILES_PER_WAVE)]
            kv_scales_tile = [None] * N_TILES_PER_WAVE
            cols = [None] * N_TILES_PER_WAVE
            for ni in range_constexpr(N_TILES_PER_WAVE):
                abs_ni = wave_ni_base + ni
                col = col0 + abs_ni * MFMA_N + lane_mod_N
                cols[ni] = col
                col_clamped = _imin(col, fx.Int32(seq_len_kv) - 1)
                kv_scales_tile[ni] = fx.Float32(sc_t[col_clamped])
                base_b = col_clamped * D
                for kk in range_constexpr(K_STEPS):
                    d_b = kk * MFMA_K + lane8
                    raw = _load_pack_i64(kv_i32, base_b + d_b)
                    b_packs[ni][kk] = _fn_to_fnuz_i64(raw) if convert_kv_fn else raw

            # Per-row MFMA + epilogue.
            for j in range_constexpr(RPB):
                row = r0 + j
                out_row_t = _make_out_row_t(row)
                for ni in range_constexpr(N_TILES_PER_WAVE):
                    col = cols[ni]
                    kv_scale = kv_scales_tile[ni]
                    col_sum = f32_0
                    for mi in range_constexpr(M_TILES):
                        acc = Vec.filled(4, 0.0, fx.Float32)
                        for kk in range_constexpr(K_STEPS):
                            acc = mfma_fn(
                                mfma_res_ty,
                                [a_packs[j][mi][kk], b_packs[ni][kk], acc, 0, 0, 0],
                            )
                        # kv_scale (>=0) hoisted out of the head sum: ReLU is
                        # positive-homogeneous (ReLU(s*x)=s*ReLU(x)), so the column
                        # sum is scaled once below -- drops M_TILES*4 muls to one.
                        for ii in range_constexpr(4):
                            score = fx.Float32(Vec(acc)[ii])
                            relu = score.maximumf(f32_0)
                            col_sum = col_sum + relu * w_frag[j][mi][ii]
                    col_sum = col_sum * kv_scale

                    # Head-reduce within the wave (width=64): shuffle_xor 16, 32.
                    for sh in [16, 32]:
                        peer = col_sum.shuffle_xor(sh, 64)
                        col_sum = col_sum + peer

                    # Only lane_div_N==0 lanes hold the MFMA_N distinct columns.
                    # `col >= start` guards the pre-filled -inf in
                    # [aligned_start, start) (tile loop is BKV-aligned below start).
                    in_window = (col >= starts[j]) & (col < ends[j])
                    is_writer = (lane_div_N == 0) & in_window

                    # Predicated store via a local @flyc.jit: the branch body is a
                    # helper call so the TensorView store closes over the kernel
                    # scope (a plain kernel-level `if` can't yield the store).
                    def _do_write(_t=out_row_t, _c=col, _v=col_sum):
                        _t[_c] = _v

                    @flyc.jit
                    def _guarded_write(_pred=is_writer, _w=_do_write):
                        if _pred:
                            _w()

                    _guarded_write()

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
        n_blocks = _uceildiv(seq_len, fx.Int32(RPB))
        gx = fx.Index(n_blocks)
        gy = fx.Index(num_splits)
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


# Variants "mfma_r<RPB>_w<WPB>" (query rows / waves per block) all share the
# _build_kernel_mfma_r_w factory. WPB must divide BKV/16 (=8 at default BKV=128).
def _mk_builder(rpb, wpb):
    return lambda **kw: _build_kernel_mfma_r_w(
        **kw, rows_per_block=rpb, waves_per_block=wpb
    )


_VARIANT_BUILDERS = {
    f"mfma_r{r}_w{w}": _mk_builder(r, w) for r in (1, 2, 4) for w in (1, 2, 4)
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
    tag = (
        variant
        or os.environ.get("FLYDSL_FP8_MQA_LOGITS_VARIANT")
        or _auto_variant(seq_len, seq_len_kv)
    )
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
    """Cached, compiled FlyDSL launcher for the given shape config.

    ``num_heads``/``head_size`` are compile-time constants; ``variant`` is an
    ``mfma_r<RPB>_w<WPB>`` tag (see ``KERNEL_VARIANTS``);
    ``convert_q_fn``/``convert_kv_fn`` mark an FP8 FN operand whose -0 (0x80) byte
    the kernel patches to FNUZ +0. ``paged`` is reserved and must be False.
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

    # FlyDSL's DLPack adaptor rejects 0-dim tensors, but per-token kv_scales /
    # weights collapse to a scalar at seq_len_kv==1; reshape back to logical rank.
    kv_scales = kv_scales.reshape(seq_len_kv)
    weights = weights.reshape(seq_len, num_heads)
    cu_starts = cu_starts.reshape(seq_len)
    cu_ends = cu_ends.reshape(seq_len)

    # gfx942 fp8 MFMA reads operands as e4m3 FNUZ (bias 8). An e4m3 FN byte (bias 7)
    # encodes exactly 2x the FNUZ value; the only differing data byte is FN -0 =
    # 0x80 (= FNUZ NaN). So pass raw bytes, let the kernel patch 0x80 -> +0, and
    # undo the 2x by scaling kv_scales -- ReLU is positive-homogeneous, so
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

    # mfma_r* kernels need seq_len padded to a multiple of RPB so every block owns
    # exactly RPB rows. Padded rows get empty windows (start==end==0) so nothing is
    # written; the output is sliced back to seq_len after the launch.
    _rpb_match = re.match(r"mfma_r(\d+)", variant)
    _RPB = int(_rpb_match.group(1)) if _rpb_match else 1
    seq_len_padded = ((seq_len + _RPB - 1) // _RPB) * _RPB
    if seq_len_padded != seq_len:
        pad = seq_len_padded - seq_len
        Q = torch.cat([Q, Q.new_zeros((pad, num_heads, head_size))], dim=0)
        weights = torch.cat([weights, weights.new_zeros((pad, num_heads))], dim=0)
        cu_starts = torch.cat([cu_starts, cu_starts.new_zeros(pad)], dim=0)
        cu_ends = torch.cat([cu_ends, cu_ends.new_zeros(pad)], dim=0)

    # Match the Triton launcher's -inf-prefill / padding so both produce
    # identically-shaped, identically-masked outputs.
    aligned_size = 256
    seq_len_kv_aligned = (seq_len_kv + aligned_size - 1) // aligned_size * aligned_size
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
