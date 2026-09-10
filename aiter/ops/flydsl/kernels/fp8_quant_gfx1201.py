# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Fully-FlyDSL per-tensor fp8 (e4m3) quant + optional Hadamard rotation.

gfx1201 / RDNA4, wave32. One wave (32 lanes) owns one row (= one token-head,
``head_dim`` elements); each lane holds ``VEC = head_dim // 32`` contiguous
elements. Feeds the per-tensor fp8 flash-attention kernel (``real = fp8 * scale``
with a single global descale per tensor).

Per-tensor scaling needs the global amax of the (rotated) tensor before any value
can be scaled, so the quant is a **2-pass** kernel:

* pass 1 (``amax``): load + optional FWHT + per-row amax; lane 0 writes one partial
  per row. A tiny ``partials.amax()`` in torch then gives the global amax without
  atomics or cross-workgroup reduction.
* pass 2 (``scale``): load + optional FWHT (recomputed) + scale by the single
  global descale + clamp + cast to fp8.

Recomputing the rotation in pass 2 keeps the rotated tensor off HBM, so total
traffic is 2 reads + 0.5 write -- the same as a fused Triton path, but the
rotation is an in-register **Fast Walsh-Hadamard Transform** (butterfly shuffles,
``log2(head_dim)`` stages, no matrix load) instead of a matmul. The normalized WHT
is orthonormal, so applying it to both Q and K leaves ``Q K^T`` unchanged -- it
cancels in attention and the consumer never needs the matrix; it only spreads
intra-row outliers so the global e4m3 scale clamps less.

MVP fast path: ``head_dim == 128`` (VEC=4). Callers fall back (Triton/torch) for
other head_dims.
"""

# NOTE: do NOT add `from __future__ import annotations` (see qk_norm_rope_quant
# for the flydsl JitFunction cache-key rationale).

import math
from functools import lru_cache

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl._mlir.dialects import rocdl
from flydsl.expr import arith, const_expr, range_constexpr
from flydsl.expr import math as fmath
from flydsl.expr.arith import CmpIPredicate
from flydsl.expr.typing import Stream, T

try:
    from flydsl.expr import buffer_ops
except ImportError:
    from . import buffer_ops

from .tensor_shim import _run_compiled, _to_raw

BLOCK_THREADS = 32  # 1 wave32
_FP8_MAX = 448.0  # e4m3fn max normal (gfx1201 native fp8)
_FP8_DTYPE = torch.float8_e4m3fn


def _build_kernel(*, head_dim: int, rotate: bool, mode: str):
    """Build one pass (``mode`` in {"amax", "scale"}) of the 2-pass per-tensor
    fp8 quant (+ optional FWHT) for a given (head_dim, rotate). Shape constants
    are captured by closure so distinct configs coexist safely."""
    assert mode in ("amax", "scale")
    D = head_dim
    VEC = D // BLOCK_THREADS
    assert D % BLOCK_THREADS == 0, f"head_dim {D} must be a multiple of {BLOCK_THREADS}"
    assert VEC == 4, f"MVP supports head_dim=128 (VEC=4) only, got VEC={VEC}"
    assert (D & (D - 1)) == 0, f"head_dim {D} must be a power of 2 for WHT"

    LOG2D = int(math.log2(D))
    LOG2VEC = int(math.log2(VEC))  # within-lane butterfly stages
    INV_SQRT_D = 1.0 / math.sqrt(D)
    _kname = f"fp8_pertensor_{mode}_D{D}{'_rot' if rotate else ''}_flydsl"

    @flyc.kernel(name=_kname, known_block_size=[BLOCK_THREADS, 1, 1])
    def kernel(
        x_in: fx.Pointer,  # [M, D] bf16, contiguous
        x_out: fx.Pointer,  # [M, D] fp8 (scale mode); unused (amax mode)
        scale_io: fx.Pointer,  # amax: [M] f32 out partials; scale: [1] f32 in descale
    ):
        # One wave per row; grid is exactly M blocks so no bounds guard on M.
        f32 = T.f32
        i32 = T.i32
        fm_fast = arith.FastMathFlags.fast

        row = fx.block_idx.x  # one wave per row
        tid = fx.thread_idx.x  # 0..31 lane
        row_idx = fx.Int64(row)

        # ---- load VEC bf16 for this lane: elems [row*D + tid*VEC, +VEC) ----
        in_rsrc = buffer_ops.create_buffer_resource_from_addr(
            fx.Int64(fx.ptrtoint(x_in)).ir_value()
        )
        row_off_elems = row_idx * D + fx.Int64(tid) * VEC
        row_off_dw = fx.Int32(row_off_elems // 2)
        x_raw = buffer_ops.buffer_load(
            in_rsrc, row_off_dw, vec_width=VEC // 2, dtype=i32
        )
        x_bf16 = fx.Vector(x_raw).bitcast(fx.BFloat16)
        xf = [
            arith.extf(f32, _to_raw(x_bf16[p]), fastmath=fm_fast)
            for p in range_constexpr(VEC)
        ]

        # ---- optional Fast Walsh-Hadamard Transform (butterfly) ----
        if const_expr(rotate):
            for st in range_constexpr(LOG2VEC):  # within-lane stages
                length = 1 << st
                new = list(xf)
                for base in range_constexpr(VEC):
                    if (base & length) == 0:
                        a = xf[base]
                        b = xf[base + length]
                        new[base] = arith.addf(_to_raw(a), _to_raw(b), fastmath=fm_fast)
                        new[base + length] = arith.subf(
                            _to_raw(a), _to_raw(b), fastmath=fm_fast
                        )
                xf = new
            lane = tid
            for st in range_constexpr(LOG2D - LOG2VEC):
                off = 1 << st  # lane xor offset
                lane_and = arith.andi(_to_raw(lane), arith.constant(off, type=i32))
                is_high = arith.cmpi(
                    CmpIPredicate.ne, lane_and, arith.constant(0, type=i32)
                )
                new = []
                for p in range_constexpr(VEC):
                    peer = _to_raw(fx.Float32(xf[p]).shuffle_xor(off, BLOCK_THREADS))
                    lo = arith.addf(_to_raw(xf[p]), peer, fastmath=fm_fast)  # self+peer
                    hi = arith.subf(peer, _to_raw(xf[p]), fastmath=fm_fast)  # peer-self
                    new.append(arith.select(is_high, hi, lo))
                xf = new
            c_norm = arith.constant(INV_SQRT_D, type=f32)  # normalize
            xf = [arith.mulf(_to_raw(v), c_norm, fastmath=fm_fast) for v in xf]

        if const_expr(mode == "amax"):
            # per-row amax (local over VEC, then butterfly max over 32 lanes);
            # lane 0 writes the row partial. Global amax = torch amax(partials).
            am = fmath.absf(_to_raw(xf[0]))
            for p in range_constexpr(VEC - 1):
                am = arith.maximumf(am, fmath.absf(_to_raw(xf[p + 1])))
            for st in range_constexpr(int(math.log2(BLOCK_THREADS))):
                off = BLOCK_THREADS // (2 << st)
                peer = _to_raw(fx.Float32(am).shuffle_xor(off, BLOCK_THREADS))
                am = arith.maximumf(am, peer)
            if tid == fx.Int32(0):
                s_rsrc = buffer_ops.create_buffer_resource_from_addr(
                    fx.Int64(fx.ptrtoint(scale_io)).ir_value()
                )
                buffer_ops.buffer_store(am, s_rsrc, _to_raw(row), offset_is_bytes=False)
            return

        # mode == "scale": read the single global descale (all lanes broadcast).
        sc_rsrc = buffer_ops.create_buffer_resource_from_addr(
            fx.Int64(fx.ptrtoint(scale_io)).ir_value()
        )
        scale = _to_raw(
            buffer_ops.buffer_load(
                sc_rsrc, arith.constant(0, type=i32), vec_width=1, dtype=f32
            )
        )
        inv_scale = arith.divf(arith.constant(1.0, type=f32), scale, fastmath=fm_fast)

        # ---- scale + clamp + pack to fp8 (VEC=4 -> 1 dword), store ----
        c_max = arith.constant(_FP8_MAX, type=f32)
        q = []
        for p in range_constexpr(VEC):
            v = arith.mulf(_to_raw(xf[p]), inv_scale, fastmath=fm_fast)
            v = arith.minimumf(
                arith.maximumf(v, arith.constant(-_FP8_MAX, type=f32)), c_max
            )
            q.append(v)
        c0 = arith.constant(0, type=i32)
        pk = rocdl.cvt_pk_fp8_f32(i32, q[0], q[1], c0, 0)
        pk = rocdl.cvt_pk_fp8_f32(i32, q[2], q[3], pk, 1)
        out_rsrc = buffer_ops.create_buffer_resource_from_addr(
            fx.Int64(fx.ptrtoint(x_out)).ir_value()
        )
        # fp8 out: 1 byte/elem, VEC=4 bytes = 1 dword. dword offset = row_off_elems/4
        out_off_dw = fx.Int32(row_off_elems // 4)
        buffer_ops.buffer_store(pk, out_rsrc, out_off_dw, offset_is_bytes=False)

    @flyc.jit
    def launch(
        x_in: fx.Pointer,
        x_out: fx.Pointer,
        scale_io: fx.Pointer,
        M: fx.Int32,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        idx_m = fx.Int64(M)
        k = kernel(x_in, x_out, scale_io)
        k.launch(grid=(idx_m, 1, 1), block=(BLOCK_THREADS, 1, 1), stream=stream)

    return launch


@lru_cache(maxsize=16)
def _compile(*, head_dim: int, rotate: bool, mode: str):
    launcher = _build_kernel(head_dim=head_dim, rotate=rotate, mode=mode)
    launcher.compile_hints = {
        "waves_per_eu": 8,
        "fast_fp_math": True,
        "unsafe_fp_math": True,
    }
    return launcher


def flydsl_fp8_pertensor_quant(
    x: torch.Tensor,
    *,
    rotate: bool,
    out: torch.Tensor = None,
    stream=None,
):
    """2-pass per-tensor fp8 quant (+ optional FWHT rotation) of ``x``, fully
    FlyDSL. ``x`` last dim is ``head_dim`` (flattened to [M, D]).

    Returns ``(x_fp8, scale)`` where ``scale`` is a 1-element f32 tensor
    (``real = fp8 * scale``). This low-level entry point requires head_dim=128.
    """
    if x.dtype != torch.bfloat16:
        raise TypeError(
            "flydsl_fp8_pertensor_quant requires bfloat16 input, " f"got {x.dtype}"
        )
    if not x.is_cuda:
        raise ValueError("flydsl_fp8_pertensor_quant requires a GPU tensor")
    if x.dim() == 0 or x.shape[-1] != 128:
        got = None if x.dim() == 0 else x.shape[-1]
        raise ValueError(f"flydsl_fp8_pertensor_quant requires head_dim=128, got {got}")
    if x.numel() == 0:
        raise ValueError("flydsl_fp8_pertensor_quant requires a non-empty tensor")

    D = x.shape[-1]
    if stream is None:
        stream = torch.cuda.current_stream(x.device)
    if stream.device != x.device:
        raise ValueError(f"stream must be on {x.device}, got {stream.device}")

    expected_shape = tuple(x.shape)
    if out is not None and (
        tuple(out.shape) != expected_shape
        or out.dtype != _FP8_DTYPE
        or out.device != x.device
        or not out.is_contiguous()
    ):
        raise ValueError(
            "out must be a contiguous float8_e4m3fn tensor with shape "
            f"{expected_shape} on {x.device}"
        )

    def _ptr(t):
        return flyc.from_c_void_p(fx.Uint8, t.data_ptr())

    with torch.cuda.device(x.device), torch.cuda.stream(stream):
        x = x.contiguous()
        M = x.numel() // D
        if out is None:
            out = torch.empty_like(x, dtype=_FP8_DTYPE)
        partials = torch.empty(M, dtype=torch.float32, device=x.device)
        fx_stream = Stream(stream)

        # Pass 1 writes per-row amax partials; x_out is unused.
        amax_k = _compile(head_dim=D, rotate=rotate, mode="amax")
        _run_compiled(amax_k, _ptr(x), _ptr(x), _ptr(partials), M, fx_stream)
        scale = (partials.amax() / _FP8_MAX).clamp(min=1e-12).reshape(1)

        # Pass 2 scales, clamps, and casts using the global descale.
        scale_k = _compile(head_dim=D, rotate=rotate, mode="scale")
        _run_compiled(scale_k, _ptr(x), _ptr(out), _ptr(scale), M, fx_stream)
    return out, scale
