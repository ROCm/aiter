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

from aiter.ops.flydsl.kernels import buffer_ops

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
