# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL warp-decode MoE kernels for gfx950 (SILOTIGER-667).

Warp-decode MoE targets very small M (B = 1..4 decode tokens) where a single
wave (64 lanes) cooperatively computes one output scalar with ``v_dot2_f32_bf16``
instead of the matrix cores.  This module is being built incrementally:

* Phase 1 (this file): the three low-level primitives the kernels are built
  from, each wrapped so it can be unit-tested in isolation against a torch
  reference on real gfx950 hardware:

    1. ``dot2_f32_bf16``          -- local ``v_dot2_f32_bf16`` inline-asm helper
                                     (2 bf16 MACs/lane into an f32 accumulator).
    2. ``fp8x2_to_bf16x2``        -- scaled FP8(e4m3) -> BF16 pair convert via the
                                     ``cvt_scalef32_pk_bf16_fp8`` ROCDL op.
    3. 64-lane butterfly reduce   -- ``shuffle_xor`` sum over shifts 1,2,4,8,16,32.

  ``build_warp_decode_primitives_module`` returns a compiled-able launcher that
  exercises all three so the Phase 1 test can validate correctness before the
  gate_up / down_reduce kernels are layered on in later phases.

Reference: ``ck_tile/ops/warp_decode/`` (WARP_DECODE_MOE_KERNELS.md and
``kernel/warp_decode_numeric.hpp``); see ``SILOTIGER-667-plan.md`` for the
full design notes and phased plan.
"""

from __future__ import annotations

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import gpu, llvm
from flydsl._mlir.extras import types as T
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr import math as fxmath
from flydsl.expr.typing import BFloat16

from aiter.ops.flydsl.kernels import buffer_ops, vector

WARP_SIZE = 64
# Butterfly-reduce shifts for a full 64-lane wave (low -> high).
_REDUCE_SHIFTS = (1, 2, 4, 8, 16, 32)


# -------------------------------------------------------------------------
# Primitive helpers (raw ir.Value in / out; call inside a @flyc.kernel body)
# -------------------------------------------------------------------------
def _ptr_rsrc(ptr):
    """Turn an ``fx.Pointer`` kernel arg into a buffer resource descriptor."""
    return buffer_ops.create_buffer_resource_from_addr(fx.Int64(fx.ptrtoint(ptr)))


def dot2_f32_bf16(a_i32, b_i32, acc_f32, *, serialize: bool = True):
    """``d = a.lo*b.lo + a.hi*b.hi + acc`` via one ``v_dot2_f32_bf16``.

    ``a_i32`` / ``b_i32`` each pack two bf16 lanes into a 32-bit VGPR; ``acc_f32``
    is the f32 accumulator, tied to the result (constraint ``0``).  ``serialize``
    appends ``s_nop 2`` to cover the dot2 -> dot2 accumulator RAW hazard, matching
    the locked FP8 baseline (``dot2_bf16_packed_raw`` in the CK reference).
    """
    asm = "v_dot2_f32_bf16 $0, $1, $2, $0"
    if serialize:
        asm += "\n\ts_nop 2"
    return llvm.inline_asm(
        T.f32(),
        [a_i32, b_i32, acc_f32],
        asm,
        "=v,v,v,0",
        has_side_effects=False,
    )


def fp8x2_to_bf16x2(src_i32, scale_f32, *, hi: bool):
    """Scaled convert of one fp8(e4m3) pair -> ``vector<2xbf16>``.

    ``src_i32`` holds four packed e4m3 bytes; ``hi=False`` converts the low pair
    (bytes 0,1), ``hi=True`` the high pair (bytes 2,3).  Each output equals
    ``fp8_value * scale`` (the hardware applies the f32 scale).
    """
    bf16x2_ty = ir.VectorType.get([2], T.bf16())
    return fx.rocdl.cvt_scalef32_pk_bf16_fp8(bf16x2_ty, src_i32, scale_f32, hi)


def fp4x2_to_bf16x2(src_i32, scale_f32, *, sel: int):
    """Scaled convert of one MXFP4(e2m1) pair -> ``vector<2xbf16>``.

    ``src_i32`` packs eight FP4 nibbles (four bf16 pairs); ``sel in {0,1,2,3}``
    selects the pair (nibbles ``2*sel, 2*sel+1``).  Each output equals
    ``fp4_value * scale_f32`` (the hardware applies the f32 e8m0 block scale).
    ``sel`` is a compile-time ``I32Attr`` -- unroll callers, never loop it
    (a runtime ``scf.for`` iv fails the attribute builder with ``bad_cast``).
    """
    bf16x2_ty = ir.VectorType.get([2], T.bf16())
    return fx.rocdl.cvt_scalef32_pk_bf16_fp4(bf16x2_ty, src_i32, scale_f32, sel)


def e8m0_byte_to_f32(byte_val):
    """Decode one E8M0 biased-exponent byte to an f32 scale (``shl 23`` + bitcast).

    ``byte_val`` is an i8 ``ir.Value`` (as loaded from a uint8 scale tensor); it
    is zero-extended, shifted into the f32 exponent field, and reinterpreted.
    Bit-exact vs ``aiter.utility.fp4_utils.e8m0_to_f32`` on the normal exponent
    range (bytes 1..254); the ``0`` / ``0xFF`` specials are never produced for
    real MXFP4 weights.
    """
    from flydsl._mlir.dialects import arith as std_arith

    byte_i32 = std_arith.ExtUIOp(T.i32(), byte_val).result
    shifted = std_arith.ShLIOp(byte_i32, _i32_const(23)).result
    return llvm.bitcast(T.f32(), shifted)


def bf16x2_to_i32(pair_vec):
    """Reinterpret a ``vector<2xbf16>`` as an i32 (dot2 packs 2 bf16 per VGPR)."""
    return llvm.bitcast(T.i32(), pair_vec)


def load_i32_words(rsrc, word0, n):
    """Load ``n`` consecutive i32 dwords starting at element offset ``word0`` using
    the widest 128-bit/64-bit buffer loads possible, returning a Python list of
    ``n`` scalar i32 ir.Values.  ``n`` is a compile-time constant (loop unrolled).

    Coalescing the per-word scalar loads into ``vec4``/``vec2`` buffer transactions
    is the main memory-throughput win for the warp-decode inner loop.
    """
    out = []
    i = 0
    while i < n:
        if n - i >= 4:
            w = 4
        elif n - i >= 2:
            w = 2
        else:
            w = 1
        vec = buffer_ops.buffer_load(rsrc, word0 + i, vec_width=w, dtype=T.i32())
        if w == 1:
            out.append(vec)
        else:
            for j in range(w):
                out.append(vector.extract(vec, static_position=[j]))
        i += w
    return out


def _i32_const(value: int):
    from flydsl._mlir.dialects import arith as std_arith

    return std_arith.ConstantOp(T.i32(), ir.IntegerAttr.get(T.i32(), value)).result


def wave_reduce_add_f32(val_f32):
    """Full 64-lane butterfly sum; every lane returns the total (raw f32)."""
    from flydsl._mlir.dialects import arith as std_arith

    width = _i32_const(WARP_SIZE)
    w = val_f32
    for sh in _REDUCE_SHIFTS:
        peer = gpu.ShuffleOp(w, _i32_const(sh), width, mode="xor").shuffleResult
        w = std_arith.AddFOp(w, peer).result
    return w


# -------------------------------------------------------------------------
# Phase 1 primitive-validation kernel + launcher
# -------------------------------------------------------------------------
def build_warp_decode_primitives_module(*, serialize_dot2: bool = True):
    """Build a launcher exercising all three primitives across one 64-lane wave.

    Grid = 1 block, block = 64 lanes.  Lane ``l`` handles element ``l``:

    * dot2:    ``out_dot[l]  = dot2(a[l], b[l], 0)``           (a,b: 2xbf16 in i32)
    * convert: ``out_cvt[2l:2l+2] = (lo, hi) fp8 pairs * scale[l]`` (bf16 in i32)
    * reduce:  ``out_red[l]  = sum_j red_in[j]``               (all lanes agree)
    """

    @flyc.kernel
    def _kernel(
        a_ptr: fx.Pointer,
        b_ptr: fx.Pointer,
        out_dot_ptr: fx.Pointer,
        f8_ptr: fx.Pointer,
        scale_ptr: fx.Pointer,
        out_cvt_ptr: fx.Pointer,
        red_in_ptr: fx.Pointer,
        out_red_ptr: fx.Pointer,
    ):
        lane = fx.thread_idx.x

        a_rsrc = _ptr_rsrc(a_ptr)
        b_rsrc = _ptr_rsrc(b_ptr)
        out_dot_rsrc = _ptr_rsrc(out_dot_ptr)
        f8_rsrc = _ptr_rsrc(f8_ptr)
        scale_rsrc = _ptr_rsrc(scale_ptr)
        out_cvt_rsrc = _ptr_rsrc(out_cvt_ptr)
        red_in_rsrc = _ptr_rsrc(red_in_ptr)
        out_red_rsrc = _ptr_rsrc(out_red_ptr)

        # 1. dot2 --------------------------------------------------------
        a_i32 = buffer_ops.buffer_load(a_rsrc, lane, vec_width=1, dtype=T.i32())
        b_i32 = buffer_ops.buffer_load(b_rsrc, lane, vec_width=1, dtype=T.i32())
        acc0 = fx.Float32(0.0).ir_value()
        d = dot2_f32_bf16(a_i32, b_i32, acc0, serialize=serialize_dot2)
        buffer_ops.buffer_store(d, out_dot_rsrc, lane)

        # 2. scaled fp8 -> bf16 pair convert -----------------------------
        f8_i32 = buffer_ops.buffer_load(f8_rsrc, lane, vec_width=1, dtype=T.i32())
        scale = buffer_ops.buffer_load(scale_rsrc, lane, vec_width=1, dtype=T.f32())
        lo = fp8x2_to_bf16x2(f8_i32, scale, hi=False)
        hi = fp8x2_to_bf16x2(f8_i32, scale, hi=True)
        # out_cvt is bf16; each lane writes 4 contiguous bf16 (lo pair, hi pair).
        buffer_ops.buffer_store(lo, out_cvt_rsrc, lane * 4)
        buffer_ops.buffer_store(hi, out_cvt_rsrc, lane * 4 + 2)

        # 3. 64-lane butterfly reduce ------------------------------------
        r = buffer_ops.buffer_load(red_in_rsrc, lane, vec_width=1, dtype=T.f32())
        total = wave_reduce_add_f32(r)
        buffer_ops.buffer_store(total, out_red_rsrc, lane)

    @flyc.jit
    def _launch(
        a_ptr: fx.Pointer,
        b_ptr: fx.Pointer,
        out_dot_ptr: fx.Pointer,
        f8_ptr: fx.Pointer,
        scale_ptr: fx.Pointer,
        out_cvt_ptr: fx.Pointer,
        red_in_ptr: fx.Pointer,
        out_red_ptr: fx.Pointer,
        stream: fx.Stream,
    ):
        _kernel(
            a_ptr,
            b_ptr,
            out_dot_ptr,
            f8_ptr,
            scale_ptr,
            out_cvt_ptr,
            red_in_ptr,
            out_red_ptr,
        ).launch(
            grid=(1, 1, 1),
            block=(WARP_SIZE, 1, 1),
            stream=stream,
        )

    return _launch


# -------------------------------------------------------------------------
# Phase 2 -- gate_up FP8 fast path (BF16 activation, FP8 e4m3 weights)
# -------------------------------------------------------------------------
def pick_kvector(hidden: int) -> int:
    """kVector = 16 (one 128-bit FP8 transaction) when it tiles HIDDEN, else 8."""
    if hidden % (WARP_SIZE * 16) == 0:
        return 16
    if hidden % (WARP_SIZE * 8) == 0:
        return 8
    raise ValueError(
        f"HIDDEN={hidden} not divisible by 64*8; unsupported for warp-decode gate_up"
    )


def build_gate_up_fp8_module(
    hidden: int,
    inter: int,
    top_k: int,
    *,
    kvector: int | None = None,
    w_scale_mode: str = "pertensor",
    serialize_dot2: bool = True,
    scale_bn: int | None = None,
    scale_bk: int | None = None,
):
    """Build the gate_up FP8 launcher (BF16 activation, FP8 e4m3 weights).

    One wave (64 lanes) computes one ``inter[b, k, j]`` scalar:
    grid ``B*TOPK*INTER`` blocks of 64 lanes.  Lane ``l`` owns hidden K-range
    ``[l*kVector, (l+1)*kVector)`` each iteration; the wave accumulates
    ``gate_dot``/``up_dot`` in f32 via ``v_dot2_f32_bf16`` over FP8->BF16-converted
    weights, butterfly-reduces, applies the (PerTensor/PerToken) weight scales,
    and writes ``silu(gate_acc)*up_acc`` as BF16.

    Shapes / layout (row-major, contiguous):
      * x            [B, HIDDEN]         bf16
      * w_gate/w_up  [E, INTER, HIDDEN]  fp8_e4m3  (row = e*INTER + j)
      * w_*_scale    f32; PerTensor -> [1], PerToken -> [E*INTER] (per weight row)
      * router_ids   [B, TOPK]           int32
      * out (inter)  [B, TOPK, INTER]    bf16      (row = b*TOPK + k)
    """
    if kvector is None:
        kvector = pick_kvector(hidden)
    if kvector % 2 != 0:
        raise ValueError(f"kVector must be even for dot2, got {kvector}")
    ktile_n = WARP_SIZE * kvector
    if hidden % ktile_n != 0:
        raise ValueError(f"HIDDEN={hidden} not divisible by 64*kVector={ktile_n}")
    num_iter = hidden // ktile_n
    n_pairs = kvector // 2  # x uint32 words / weight pairs per lane per iter
    n_wwords = kvector // 4  # fp8 weight dwords/lane/iter (4 fp8 each)
    per_token_scale = const_expr(w_scale_mode == "pertoken")
    block2d = const_expr(w_scale_mode == "block2d")
    if w_scale_mode == "block2d":
        if scale_bn is None or scale_bk is None:
            raise ValueError("block2d scales require scale_bn and scale_bk")
        if hidden % scale_bk != 0:
            raise ValueError(f"HIDDEN={hidden} not divisible by scale_bk={scale_bk}")
        if scale_bk % kvector != 0:
            raise ValueError(
                f"block2d scale_bk={scale_bk} must be a multiple of kVector={kvector} "
                "so each lane's K-chunk lies in one scale block"
            )
    elif w_scale_mode not in ("pertensor", "pertoken"):
        raise ValueError(f"unsupported w_scale_mode: {w_scale_mode!r}")
    scale_cols_g = (hidden // scale_bk) if w_scale_mode == "block2d" else 0

    @flyc.kernel
    def _kernel(
        x_ptr: fx.Pointer,
        wg_ptr: fx.Pointer,
        wu_ptr: fx.Pointer,
        wgs_ptr: fx.Pointer,
        wus_ptr: fx.Pointer,
        rid_ptr: fx.Pointer,
        out_ptr: fx.Pointer,
    ):
        bid = fx.block_idx.x
        lane = fx.thread_idx.x

        neuron_j = bid % inter
        d = bid // inter
        expert_k = d % top_k
        token_b = d // top_k

        rid_rsrc = _ptr_rsrc(rid_ptr)
        e = fx.Int32(
            buffer_ops.buffer_load(
                rid_rsrc, token_b * top_k + expert_k, vec_width=1, dtype=T.i32()
            )
        )
        w_row = e * inter + neuron_j

        x_rsrc = _ptr_rsrc(x_ptr)
        wg_rsrc = _ptr_rsrc(wg_ptr)
        wu_rsrc = _ptr_rsrc(wu_ptr)

        one_f32 = fx.Float32(1.0).ir_value()
        wgs_rsrc = _ptr_rsrc(wgs_ptr)
        wus_rsrc = _ptr_rsrc(wus_ptr)

        if const_expr(block2d):
            # Block2D<BN,BK> scales vary along K, so fold each K-block's scale into
            # the per-lane partial before the single reduce. scale_bk % kVector == 0
            # guarantees a lane's K-chunk sits in one block -> one scale per iter.
            row_blk = w_row // scale_bn
            gate_acc_l = fx.Float32(0.0)
            up_acc_l = fx.Float32(0.0)
            for i in range_constexpr(num_iter):
                k_base = i * ktile_n + lane * kvector
                x_word0 = (token_b * hidden + k_base) // 2
                w_word0 = (w_row * hidden + k_base) // 4
                xw = load_i32_words(x_rsrc, x_word0, n_pairs)
                gw = load_i32_words(wg_rsrc, w_word0, n_wwords)
                uw = load_i32_words(wu_rsrc, w_word0, n_wwords)
                gd = fx.Float32(0.0).ir_value()
                ud = fx.Float32(0.0).ir_value()
                for ipair in range_constexpr(n_pairs):
                    w_word = ipair // 2
                    w_hi = (ipair % 2) == 1
                    x_i32 = xw[ipair]
                    g_i32 = bf16x2_to_i32(fp8x2_to_bf16x2(gw[w_word], one_f32, hi=w_hi))
                    u_i32 = bf16x2_to_i32(fp8x2_to_bf16x2(uw[w_word], one_f32, hi=w_hi))
                    gd = dot2_f32_bf16(x_i32, g_i32, gd, serialize=serialize_dot2)
                    ud = dot2_f32_bf16(x_i32, u_i32, ud, serialize=serialize_dot2)
                sidx = fx.Int32(row_blk * scale_cols_g + k_base // scale_bk)
                gs_i = buffer_ops.buffer_load(
                    wgs_rsrc, sidx, vec_width=1, dtype=T.f32()
                )
                us_i = buffer_ops.buffer_load(
                    wus_rsrc, sidx, vec_width=1, dtype=T.f32()
                )
                gate_acc_l = gate_acc_l + fx.Float32(gd) * fx.Float32(gs_i)
                up_acc_l = up_acc_l + fx.Float32(ud) * fx.Float32(us_i)
            gate_acc = fx.Float32(wave_reduce_add_f32(gate_acc_l.ir_value()))
            up_acc = fx.Float32(wave_reduce_add_f32(up_acc_l.ir_value()))
        else:
            gate_dot = fx.Float32(0.0).ir_value()
            up_dot = fx.Float32(0.0).ir_value()
            for i in range_constexpr(num_iter):
                k_base = i * ktile_n + lane * kvector
                # element bases -> word (i32) offsets: x 2 bf16/word, w 4 fp8/word.
                x_word0 = (token_b * hidden + k_base) // 2
                w_word0 = (w_row * hidden + k_base) // 4
                # Coalesced 128-bit loads: n_pairs x dwords, n_wwords gate/up dwords.
                xw = load_i32_words(x_rsrc, x_word0, n_pairs)
                gw = load_i32_words(wg_rsrc, w_word0, n_wwords)
                uw = load_i32_words(wu_rsrc, w_word0, n_wwords)
                for ipair in range_constexpr(n_pairs):
                    w_word = ipair // 2
                    w_hi = (ipair % 2) == 1
                    x_i32 = xw[ipair]
                    g_i32 = bf16x2_to_i32(fp8x2_to_bf16x2(gw[w_word], one_f32, hi=w_hi))
                    u_i32 = bf16x2_to_i32(fp8x2_to_bf16x2(uw[w_word], one_f32, hi=w_hi))
                    gate_dot = dot2_f32_bf16(
                        x_i32, g_i32, gate_dot, serialize=serialize_dot2
                    )
                    up_dot = dot2_f32_bf16(
                        x_i32, u_i32, up_dot, serialize=serialize_dot2
                    )

            gate_sum = wave_reduce_add_f32(gate_dot)
            up_sum = wave_reduce_add_f32(up_dot)

            # Weight scales (PerTensor -> p[0]; PerToken -> p[w_row]).
            if const_expr(per_token_scale):
                scale_off = w_row
            else:
                scale_off = fx.Int32(0)
            gs = buffer_ops.buffer_load(wgs_rsrc, scale_off, vec_width=1, dtype=T.f32())
            us = buffer_ops.buffer_load(wus_rsrc, scale_off, vec_width=1, dtype=T.f32())
            gate_acc = fx.Float32(gate_sum) * fx.Float32(gs)
            up_acc = fx.Float32(up_sum) * fx.Float32(us)

        sig = fx.Float32(1.0) / (
            fx.Float32(1.0) + fxmath.exp(fx.Float32(0.0) - gate_acc)
        )
        out_val = gate_acc * sig * up_acc

        # Every lane holds the identical reduced result; only lane 0 writes the
        # single BF16 scalar (avoids 64x redundant global stores). The reduce
        # above must run on all lanes (cross-lane shuffles), so it stays outside.
        out_off = (token_b * top_k + expert_k) * inter + neuron_j
        out_rsrc = _ptr_rsrc(out_ptr)
        if lane == 0:
            buffer_ops.buffer_store(BFloat16(out_val).ir_value(), out_rsrc, out_off)

    @flyc.jit
    def _launch(
        x_ptr: fx.Pointer,
        wg_ptr: fx.Pointer,
        wu_ptr: fx.Pointer,
        wgs_ptr: fx.Pointer,
        wus_ptr: fx.Pointer,
        rid_ptr: fx.Pointer,
        out_ptr: fx.Pointer,
        grid_x: fx.Int32,
        stream: fx.Stream,
    ):
        _kernel(x_ptr, wg_ptr, wu_ptr, wgs_ptr, wus_ptr, rid_ptr, out_ptr).launch(
            grid=(grid_x, 1, 1),
            block=(WARP_SIZE, 1, 1),
            stream=stream,
        )

    return _launch


# -------------------------------------------------------------------------
# Phase 3 -- down_reduce FP8 fast path (BF16 intermediate, FP8 e4m3 weights)
# -------------------------------------------------------------------------
def build_down_reduce_fp8_module(
    inter: int,
    hidden: int,
    top_k: int,
    *,
    kvector: int | None = None,
    w_scale_mode: str = "pertensor",
    serialize_dot2: bool = True,
    scale_bn: int | None = None,
    scale_bk: int | None = None,
    kh_per_warp: int = 1,
):
    """Build the down_reduce FP8 launcher (BF16 intermediate, FP8 e4m3 weights).

    Each wave (64 lanes) computes ``kh_per_warp`` adjacent ``y[b, out_j]`` outputs;
    grid ``B*(HIDDEN/kh_per_warp)`` blocks of 64 lanes.  For each of the token's
    TOPK experts, lane ``l`` owns INTER K-range ``[l*kVector, (l+1)*kVector)`` and,
    per iteration, loads the shared BF16 activation chunk **once** and issues
    ``kh_per_warp`` independent FP8 weight-row loads before consuming them.  The
    dots accumulate in f32 via ``v_dot2_f32_bf16``; the (lane-uniform)
    ``router_wt * ds`` is folded into each output's per-lane partial.  A butterfly
    reduce per output gives ``y``; lane 0 writes the BF16 results.

    ``kh_per_warp=2`` (**H2**) is the FP8 best: it gives ``down`` two weight loads
    in flight per wave (the memory-level parallelism that already lets ``gate_up``
    hit ~peak with its independent gate+up streams) plus activation reuse, halving
    the wave count and per-wave router-load/reduce overhead.

    Shapes / layout (row-major, contiguous):
      * inter        [B, TOPK, INTER]    bf16       (row = b*TOPK + k)
      * w_down       [E, HIDDEN, INTER]  fp8_e4m3   (row = e*HIDDEN + out_j)
      * w_down_scale f32; PerTensor -> [1]; PerToken -> [E*HIDDEN] (per weight row);
                     Block2D -> [(E*HIDDEN)//BN * INTER//BK] over (row-blk, K-blk)
      * router_ids   [B, TOPK]           int32
      * router_wts   [B, TOPK]           float32    (normalized to sum 1 per token)
      * y            [B, HIDDEN]         bf16
    """
    if kvector is None:
        kvector = pick_kvector(inter)
    if kvector % 2 != 0:
        raise ValueError(f"kVector must be even for dot2, got {kvector}")
    if kh_per_warp < 1:
        raise ValueError(f"kh_per_warp must be >= 1, got {kh_per_warp}")
    if hidden % kh_per_warp != 0:
        raise ValueError(f"HIDDEN={hidden} not divisible by kh_per_warp={kh_per_warp}")
    ktile_n = WARP_SIZE * kvector
    if inter % ktile_n != 0:
        raise ValueError(f"INTER={inter} not divisible by 64*kVector={ktile_n}")
    num_iter = inter // ktile_n
    n_pairs = kvector // 2
    n_wwords = kvector // 4  # fp8 weight dwords/lane/iter (4 fp8 each)
    n_cols = hidden // kh_per_warp  # output column-groups per token
    per_token_scale = const_expr(w_scale_mode == "pertoken")
    block2d = const_expr(w_scale_mode == "block2d")
    if w_scale_mode == "block2d":
        if scale_bn is None or scale_bk is None:
            raise ValueError("block2d scales require scale_bn and scale_bk")
        if inter % scale_bk != 0:
            raise ValueError(f"INTER={inter} not divisible by scale_bk={scale_bk}")
        if scale_bk % kvector != 0:
            raise ValueError(
                f"block2d scale_bk={scale_bk} must be a multiple of kVector={kvector} "
                "so each lane's K-chunk lies in one scale block"
            )
    elif w_scale_mode not in ("pertensor", "pertoken"):
        raise ValueError(f"unsupported w_scale_mode: {w_scale_mode!r}")
    scale_cols_d = (inter // scale_bk) if w_scale_mode == "block2d" else 0

    @flyc.kernel
    def _kernel(
        inter_ptr: fx.Pointer,
        wd_ptr: fx.Pointer,
        wds_ptr: fx.Pointer,
        rid_ptr: fx.Pointer,
        rwt_ptr: fx.Pointer,
        y_ptr: fx.Pointer,
    ):
        bid = fx.block_idx.x
        lane = fx.thread_idx.x

        col = bid % n_cols
        token_b = bid // n_cols
        out_j0 = col * kh_per_warp  # this wave owns out_j0 .. out_j0+kh_per_warp-1

        inter_rsrc = _ptr_rsrc(inter_ptr)
        wd_rsrc = _ptr_rsrc(wd_ptr)
        wds_rsrc = _ptr_rsrc(wds_ptr)
        rid_rsrc = _ptr_rsrc(rid_ptr)
        rwt_rsrc = _ptr_rsrc(rwt_ptr)

        one_f32 = fx.Float32(1.0).ir_value()
        acc = [fx.Float32(0.0) for _ in range(kh_per_warp)]

        for k in range_constexpr(top_k):
            ridx = token_b * top_k + k
            e = fx.Int32(
                buffer_ops.buffer_load(rid_rsrc, ridx, vec_width=1, dtype=T.i32())
            )
            rw = buffer_ops.buffer_load(rwt_rsrc, ridx, vec_width=1, dtype=T.f32())
            w_row = [e * hidden + out_j0 + h for h in range(kh_per_warp)]

            if const_expr(block2d):
                # Block2D<BN,BK> scale varies along K -> fold each K-block's
                # (router_wt * ds_block) into each output's per-lane partial.
                for i in range_constexpr(num_iter):
                    k_base = i * ktile_n + lane * kvector
                    inter_row = token_b * top_k + k
                    a_word0 = (inter_row * inter + k_base) // 2
                    # Shared activation load + kh independent weight loads in flight.
                    aw = load_i32_words(inter_rsrc, a_word0, n_pairs)
                    dw = [
                        load_i32_words(
                            wd_rsrc, (w_row[h] * inter + k_base) // 4, n_wwords
                        )
                        for h in range(kh_per_warp)
                    ]
                    col_blk = k_base // scale_bk
                    for h in range_constexpr(kh_per_warp):
                        dot_i = fx.Float32(0.0).ir_value()
                        for ipair in range_constexpr(n_pairs):
                            w_word = ipair // 2
                            w_hi = (ipair % 2) == 1
                            d_i32 = bf16x2_to_i32(
                                fp8x2_to_bf16x2(dw[h][w_word], one_f32, hi=w_hi)
                            )
                            dot_i = dot2_f32_bf16(
                                aw[ipair], d_i32, dot_i, serialize=serialize_dot2
                            )
                        sidx = fx.Int32(w_row[h] // scale_bn * scale_cols_d + col_blk)
                        ds_i = buffer_ops.buffer_load(
                            wds_rsrc, sidx, vec_width=1, dtype=T.f32()
                        )
                        acc[h] = acc[h] + fx.Float32(dot_i) * (
                            fx.Float32(rw) * fx.Float32(ds_i)
                        )
            else:
                if const_expr(per_token_scale):
                    ds = [
                        buffer_ops.buffer_load(
                            wds_rsrc, w_row[h], vec_width=1, dtype=T.f32()
                        )
                        for h in range(kh_per_warp)
                    ]
                else:
                    ds0 = buffer_ops.buffer_load(
                        wds_rsrc, fx.Int32(0), vec_width=1, dtype=T.f32()
                    )
                    ds = [ds0 for _ in range(kh_per_warp)]

                dot = [fx.Float32(0.0).ir_value() for _ in range(kh_per_warp)]
                for i in range_constexpr(num_iter):
                    k_base = i * ktile_n + lane * kvector
                    inter_row = token_b * top_k + k
                    a_word0 = (inter_row * inter + k_base) // 2
                    # Shared activation load + kh independent weight loads in flight.
                    aw = load_i32_words(inter_rsrc, a_word0, n_pairs)
                    dw = [
                        load_i32_words(
                            wd_rsrc, (w_row[h] * inter + k_base) // 4, n_wwords
                        )
                        for h in range(kh_per_warp)
                    ]
                    for h in range_constexpr(kh_per_warp):
                        for ipair in range_constexpr(n_pairs):
                            w_word = ipair // 2
                            w_hi = (ipair % 2) == 1
                            d_i32 = bf16x2_to_i32(
                                fp8x2_to_bf16x2(dw[h][w_word], one_f32, hi=w_hi)
                            )
                            dot[h] = dot2_f32_bf16(
                                aw[ipair], d_i32, dot[h], serialize=serialize_dot2
                            )

                # router_wt * ds is lane-uniform -> fold into each per-lane partial
                # before the single cross-lane reduce (avoids reduce/k).
                for h in range_constexpr(kh_per_warp):
                    acc[h] = acc[h] + fx.Float32(dot[h]) * (
                        fx.Float32(rw) * fx.Float32(ds[h])
                    )

        # Reduce runs on all lanes (cross-lane shuffles); only lane 0 stores.
        y_sum = [
            fx.Float32(wave_reduce_add_f32(acc[h].ir_value()))
            for h in range(kh_per_warp)
        ]
        y_rsrc = _ptr_rsrc(y_ptr)
        if lane == 0:
            for h in range_constexpr(kh_per_warp):
                buffer_ops.buffer_store(
                    BFloat16(y_sum[h]).ir_value(), y_rsrc, token_b * hidden + out_j0 + h
                )

    @flyc.jit
    def _launch(
        inter_ptr: fx.Pointer,
        wd_ptr: fx.Pointer,
        wds_ptr: fx.Pointer,
        rid_ptr: fx.Pointer,
        rwt_ptr: fx.Pointer,
        y_ptr: fx.Pointer,
        grid_x: fx.Int32,
        stream: fx.Stream,
    ):
        _kernel(inter_ptr, wd_ptr, wds_ptr, rid_ptr, rwt_ptr, y_ptr).launch(
            grid=(grid_x, 1, 1),
            block=(WARP_SIZE, 1, 1),
            stream=stream,
        )

    return _launch


# -------------------------------------------------------------------------
# Phase B -- down_reduce MXFP4 fast path (BF16 intermediate, FP4 e2m1 weights)
# -------------------------------------------------------------------------
def build_down_reduce_fp4_module(
    inter: int,
    hidden: int,
    top_k: int,
    *,
    kvector: int | None = None,
    serialize_dot2: bool = True,
    scale_bn: int = 1,
    scale_bk: int = 32,
    kh_per_warp: int = 1,
):
    """Build the down_reduce MXFP4 launcher (BF16 intermediate, FP4 e2m1 weights).

    Structurally identical to :func:`build_down_reduce_fp8_module` (one wave =
    ``kh_per_warp`` adjacent ``y[b, out_j]`` outputs; shared activation load +
    ``kh_per_warp`` independent weight-row loads in flight; butterfly reduce;
    lane 0 stores), but the weights are **MXFP4**:

      * One i32 packs **8** FP4 nibbles = 4 bf16 pairs, so a lane's ``kVector``
        elements need ``kVector/8`` weight dwords (vs ``kVector/4`` for FP8) --
        half the weight bandwidth, the ticket's #1 win at B>=2.
      * Each pair is converted with ``cvt_scalef32_pk_bf16_fp4(..., sel)``
        (``sel = ipair % 4``), and the per-block **E8M0** scale is applied *in
        the convert* (the hardware scaled convert).  Because each lane's
        ``kVector`` chunk lies within one ``scale_bk`` block (enforced below),
        the block scale is uniform across the lane's pairs for that iteration;
        applying it in-convert is exactly equivalent to scaling the partial dot.
      * ``router_wt`` is lane-uniform and folded after the dot (as in FP8).

    Scale layout is **Block2D<BN,BK>** E8M0 (MXFP4 default ``BN=1``, ``BK=32``):
    ``w_down_scale`` is ``uint8`` [(E*HIDDEN)//BN, INTER//BK] row-major over
    (weight-row-block, K-block).

    Shapes / layout (row-major, contiguous):
      * inter        [B, TOPK, INTER]        bf16       (row = b*TOPK + k)
      * w_down       [E, HIDDEN, INTER] FP4  (uint8 [.., INTER//2]; row = e*HIDDEN + out_j)
      * w_down_scale uint8 (E8M0)            [(E*HIDDEN)//BN, INTER//BK]
      * router_ids   [B, TOPK]               int32
      * router_wts   [B, TOPK]               float32    (normalized to sum 1 per token)
      * y            [B, HIDDEN]             bf16
    """
    if kvector is None:
        # FP4 fast path: 1 i32 = 8 FP4 = one weight dword per lane per iter.
        kvector = 8
    if kvector % 8 != 0:
        raise ValueError(
            f"kVector must be a multiple of 8 for FP4 (8 fp4/i32), got {kvector}"
        )
    if kh_per_warp < 1:
        raise ValueError(f"kh_per_warp must be >= 1, got {kh_per_warp}")
    if hidden % kh_per_warp != 0:
        raise ValueError(f"HIDDEN={hidden} not divisible by kh_per_warp={kh_per_warp}")
    ktile_n = WARP_SIZE * kvector
    if inter % ktile_n != 0:
        raise ValueError(f"INTER={inter} not divisible by 64*kVector={ktile_n}")
    if scale_bk % kvector != 0:
        raise ValueError(
            f"MXFP4 scale_bk={scale_bk} must be a multiple of kVector={kvector} "
            "so each lane's K-chunk lies in one scale block"
        )
    if inter % scale_bk != 0:
        raise ValueError(f"INTER={inter} not divisible by scale_bk={scale_bk}")
    num_iter = inter // ktile_n
    n_pairs = kvector // 2  # bf16 pairs (dot2 iterations) per lane per iter
    n_wwords = kvector // 8  # FP4 weight dwords per lane per iter (8 fp4 each)
    n_cols = hidden // kh_per_warp  # output column-groups per token
    scale_cols = inter // scale_bk

    @flyc.kernel
    def _kernel(
        inter_ptr: fx.Pointer,
        wd_ptr: fx.Pointer,
        wds_ptr: fx.Pointer,
        rid_ptr: fx.Pointer,
        rwt_ptr: fx.Pointer,
        y_ptr: fx.Pointer,
    ):
        bid = fx.block_idx.x
        lane = fx.thread_idx.x

        col = bid % n_cols
        token_b = bid // n_cols
        out_j0 = col * kh_per_warp

        inter_rsrc = _ptr_rsrc(inter_ptr)
        wd_rsrc = _ptr_rsrc(wd_ptr)
        wds_rsrc = _ptr_rsrc(wds_ptr)
        rid_rsrc = _ptr_rsrc(rid_ptr)
        rwt_rsrc = _ptr_rsrc(rwt_ptr)

        acc = [fx.Float32(0.0) for _ in range(kh_per_warp)]

        for k in range_constexpr(top_k):
            ridx = token_b * top_k + k
            e = fx.Int32(
                buffer_ops.buffer_load(rid_rsrc, ridx, vec_width=1, dtype=T.i32())
            )
            rw = buffer_ops.buffer_load(rwt_rsrc, ridx, vec_width=1, dtype=T.f32())
            w_row = [e * hidden + out_j0 + h for h in range(kh_per_warp)]

            for i in range_constexpr(num_iter):
                k_base = i * ktile_n + lane * kvector
                inter_row = token_b * top_k + k
                a_word0 = (inter_row * inter + k_base) // 2
                # Shared activation load + kh independent FP4 weight loads.
                aw = load_i32_words(inter_rsrc, a_word0, n_pairs)
                dw = [
                    load_i32_words(wd_rsrc, (w_row[h] * inter + k_base) // 8, n_wwords)
                    for h in range(kh_per_warp)
                ]
                col_blk = k_base // scale_bk
                for h in range_constexpr(kh_per_warp):
                    # E8M0 block scale for this (weight row, K-block), applied
                    # in the convert; uniform over the lane's chunk this iter.
                    sidx = fx.Int32(w_row[h] // scale_bn * scale_cols + col_blk)
                    blk_byte = buffer_ops.buffer_load(
                        wds_rsrc, sidx, vec_width=1, dtype=T.i8()
                    )
                    blk_scale = e8m0_byte_to_f32(blk_byte)
                    dot_i = fx.Float32(0.0).ir_value()
                    for ipair in range_constexpr(n_pairs):
                        w_word = ipair // 4
                        sel = ipair % 4
                        d_i32 = bf16x2_to_i32(
                            fp4x2_to_bf16x2(dw[h][w_word], blk_scale, sel=sel)
                        )
                        dot_i = dot2_f32_bf16(
                            aw[ipair], d_i32, dot_i, serialize=serialize_dot2
                        )
                    # router_wt is lane-uniform; the block scale is already in
                    # the converted weights, so only fold rw here.
                    acc[h] = acc[h] + fx.Float32(dot_i) * fx.Float32(rw)

        y_sum = [
            fx.Float32(wave_reduce_add_f32(acc[h].ir_value()))
            for h in range(kh_per_warp)
        ]
        y_rsrc = _ptr_rsrc(y_ptr)
        if lane == 0:
            for h in range_constexpr(kh_per_warp):
                buffer_ops.buffer_store(
                    BFloat16(y_sum[h]).ir_value(), y_rsrc, token_b * hidden + out_j0 + h
                )

    @flyc.jit
    def _launch(
        inter_ptr: fx.Pointer,
        wd_ptr: fx.Pointer,
        wds_ptr: fx.Pointer,
        rid_ptr: fx.Pointer,
        rwt_ptr: fx.Pointer,
        y_ptr: fx.Pointer,
        grid_x: fx.Int32,
        stream: fx.Stream,
    ):
        _kernel(inter_ptr, wd_ptr, wds_ptr, rid_ptr, rwt_ptr, y_ptr).launch(
            grid=(grid_x, 1, 1),
            block=(WARP_SIZE, 1, 1),
            stream=stream,
        )

    return _launch
