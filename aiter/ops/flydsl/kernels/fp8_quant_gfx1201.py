# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL per-tensor FP8 (e4m3) quantization with optional Hadamard rotation.

gfx1201 / RDNA4, wave32. One wave (32 lanes) owns one row (= one token-head,
``head_dim`` elements); each lane holds ``VEC = head_dim // 32`` contiguous
elements. Feeds the per-tensor FP8 flash-attention kernel (``real = fp8 * scale``
with one global scale per tensor).

Per-tensor scaling needs the global amax of the (rotated) tensor before values can
be scaled, so quantization uses two passes:

* pass 1 (``amax``): load + optional FWHT + per-row amax; lane 0 writes one partial
  per row. A tiny ``partials.amax()`` in torch then gives the global amax without
  atomics or cross-workgroup reduction.
* pass 2 (``scale``): load + optional FWHT (recomputed) + divide by the global
  scale, clamp, and cast to FP8.

Recomputing the rotation in pass 2 keeps the rotated tensor off HBM. The
in-register Fast Walsh-Hadamard Transform uses butterfly shuffles rather than a
matrix load. Its normalization is orthonormal, so rotating both Q and K preserves
``Q K^T`` while reducing intra-row outliers for global e4m3 scaling.

The native path specializes power-of-two head dimensions with an even number of
elements per lane. The public wrapper zero-pads non-power-of-two dimensions for
unrotated quantization, then returns a contiguous tensor at the original width.
"""

# NOTE: do NOT add `from __future__ import annotations` (see qk_norm_rope_quant
# for the flydsl JitFunction cache-key rationale).

import math
from functools import lru_cache

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl._mlir.dialects import rocdl
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr import math as fmath
from flydsl.expr.typing import Stream

from .tensor_shim import _run_compiled, buf_copy_load, buf_copy_store, ptr_buf_tensor

BLOCK_THREADS = 32  # 1 wave32
_FP8_MAX = 448.0  # e4m3fn max normal (gfx1201 native fp8)
_FP8_DTYPE = torch.float8_e4m3fn


def _storage_byte_range(tensor):
    """Return the half-open byte range touched by a non-empty tensor view."""
    storage_ptr = tensor.untyped_storage().data_ptr()
    first = tensor.data_ptr() - storage_ptr
    last = first
    element_size = tensor.element_size()
    for size, stride in zip(tensor.shape, tensor.stride()):
        delta = (size - 1) * stride * element_size
        first += min(0, delta)
        last += max(0, delta)
    return first, last + element_size


def _storage_overlaps(lhs, rhs):
    if lhs.device != rhs.device:
        return False
    lhs_storage = lhs.untyped_storage()
    rhs_storage = rhs.untyped_storage()
    if lhs_storage.data_ptr() != rhs_storage.data_ptr():
        return False
    lhs_first, lhs_last = _storage_byte_range(lhs)
    rhs_first, rhs_last = _storage_byte_range(rhs)
    return lhs_first < rhs_last and rhs_first < lhs_last


def _build_kernel(*, head_dim: int, rotate: bool, mode: str):
    """Build one per-tensor FP8 quantization pass for ``head_dim`` and ``rotate``.

    ``mode`` is ``"amax"`` or ``"scale"``. Shape constants are captured by
    closure so independently compiled configurations coexist safely.
    """
    assert mode in ("amax", "scale")
    D = head_dim
    VEC = D // BLOCK_THREADS
    DWORDS_PER_LANE = VEC // 2
    LOAD_DWORDS = min(DWORDS_PER_LANE, 4)
    NUM_LOADS = DWORDS_PER_LANE // LOAD_DWORDS
    assert D % BLOCK_THREADS == 0, f"head_dim {D} must be a multiple of {BLOCK_THREADS}"
    assert VEC in (2, 4, 8, 16, 32), f"unsupported elements per lane: {VEC}"
    assert (D & (D - 1)) == 0, f"head_dim {D} must be a power of 2 for WHT"

    LOG2D = int(math.log2(D))
    LOG2VEC = int(math.log2(VEC))  # within-lane butterfly stages
    INV_SQRT_D = 1.0 / math.sqrt(D)
    _kname = f"fp8_pertensor_{mode}_D{D}{'_rot' if rotate else ''}_flydsl"

    @flyc.kernel(name=_kname, known_block_size=[BLOCK_THREADS, 1, 1])
    def kernel(
        x_in: fx.Tensor,  # [M, D] bf16, contiguous
        x_out: fx.Tensor,  # [M, D] fp8 (scale mode); unused (amax mode)
        scale_io: fx.Tensor,  # amax: [M] f32 out partials; scale: [1] f32 in descale
    ):
        # One wave per row; grid is exactly M blocks so no bounds guard on M.
        row = fx.block_idx.x  # one wave per row
        tid = fx.thread_idx.x  # 0..31 lane
        row_idx = fx.Int64(row)
        # V# copy atoms cap at dwordx4 (16 bytes). Larger head dimensions
        # therefore assemble the lane fragment from adjacent dwordx4 loads.
        x_i32 = ptr_buf_tensor(x_in, fx.Int32, unit_elems=LOAD_DWORDS, unit_stride=1)
        out_i32 = ptr_buf_tensor(x_out, fx.Int32)
        scale_f32 = ptr_buf_tensor(scale_io, fx.Float32)

        # ---- load VEC bf16 for this lane: elems [row*D + tid*VEC, +VEC) ----
        row_off_elems = row_idx * D + fx.Int64(tid) * VEC
        row_off_dw = fx.Int32(row_off_elems // 2)
        x_dwords = []
        for load_idx in range_constexpr(NUM_LOADS):
            x_raw = buf_copy_load(
                x_i32,
                row_off_dw + fx.Int32(load_idx * LOAD_DWORDS),
                unit_elems=LOAD_DWORDS,
            )
            if const_expr(LOAD_DWORDS == 1):
                x_raw = fx.Vector.from_elements([x_raw], fx.Int32)
            for dword_idx in range_constexpr(LOAD_DWORDS):
                x_dwords.append(x_raw[dword_idx])
        x_raw = fx.Vector.from_elements(x_dwords, fx.Int32)
        x_bf16 = fx.Vector(x_raw).bitcast(fx.BFloat16)
        xf = [x_bf16[p].to(fx.Float32) for p in range_constexpr(VEC)]

        # ---- optional Fast Walsh-Hadamard Transform (butterfly) ----
        if const_expr(rotate):
            for st in range_constexpr(LOG2VEC):  # within-lane stages
                length = 1 << st
                new = list(xf)
                for base in range_constexpr(VEC):
                    if (base & length) == 0:
                        a = xf[base]
                        b = xf[base + length]
                        new[base] = a + b
                        new[base + length] = a - b
                xf = new
            lane = tid
            for st in range_constexpr(LOG2D - LOG2VEC):
                off = 1 << st  # lane xor offset
                is_high = (lane & fx.Int32(off)) != fx.Int32(0)
                new = []
                for p in range_constexpr(VEC):
                    peer = xf[p].shuffle_xor(off, BLOCK_THREADS)
                    lo = xf[p] + peer  # self+peer
                    hi = peer - xf[p]  # peer-self
                    new.append(is_high.select(hi, lo))
                xf = new
            c_norm = fx.Float32(INV_SQRT_D)  # normalize
            xf = [v * c_norm for v in xf]

        if const_expr(mode == "amax"):
            # per-row amax (local over VEC, then butterfly max over 32 lanes);
            # lane 0 writes the row partial. Global amax = torch amax(partials).
            am = fmath.absf(xf[0])
            for p in range_constexpr(VEC - 1):
                am = am.maximumf(fmath.absf(xf[p + 1]))
            for st in range_constexpr(int(math.log2(BLOCK_THREADS))):
                off = BLOCK_THREADS // (2 << st)
                peer = am.shuffle_xor(off, BLOCK_THREADS)
                am = am.maximumf(peer)
            if tid == fx.Int32(0):
                buf_copy_store(scale_f32, fx.Int32(row), am, elem=fx.Float32)
            return

        # mode == "scale": read the single global descale (all lanes broadcast).
        scale = buf_copy_load(scale_f32, fx.Int32(0), elem=fx.Float32)
        inv_scale = fx.Float32(1.0) / scale

        # Scale, clamp, then pack to FP8.
        c_max = fx.Float32(_FP8_MAX)
        c_min = fx.Float32(-_FP8_MAX)
        q = []
        for p in range_constexpr(VEC):
            v = xf[p] * inv_scale
            v = v.maximumf(c_min).minimumf(c_max)
            q.append(v)
        c0 = fx.Int32(0)
        # Keep this raw intrinsic: FlyDSL 0.3.4.1 has no typed packed-FP8 conversion.
        if const_expr(VEC == 2):
            pk = rocdl.cvt_pk_fp8_f32(
                fx.Int32.ir_type, q[0].ir_value(), q[1].ir_value(), c0.ir_value(), 0
            )
            peer_pk = fx.Int32(pk).shuffle_xor(1, BLOCK_THREADS)
            if tid % fx.Int32(2) == fx.Int32(0):
                out_off_dw = fx.Int32(row_idx * (D // 4) + tid // fx.Int32(2))
                buf_copy_store(
                    out_i32,
                    out_off_dw,
                    fx.Int32(pk) | (fx.Int32(peer_pk) << fx.Int32(16)),
                )
        else:
            for dword_idx in range_constexpr(VEC // 4):
                base = dword_idx * 4
                pk = rocdl.cvt_pk_fp8_f32(
                    fx.Int32.ir_type,
                    q[base].ir_value(),
                    q[base + 1].ir_value(),
                    c0.ir_value(),
                    0,
                )
                pk = rocdl.cvt_pk_fp8_f32(
                    fx.Int32.ir_type,
                    q[base + 2].ir_value(),
                    q[base + 3].ir_value(),
                    pk,
                    1,
                )
                out_off_dw = fx.Int32(row_off_elems // 4 + dword_idx)
                buf_copy_store(out_i32, out_off_dw, pk)

    @flyc.jit
    def launch(
        x_in: fx.Tensor,
        x_out: fx.Tensor,
        scale_io: fx.Tensor,
        M: fx.Int32,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        idx_m = fx.Int64(M)
        k = kernel(x_in, x_out, scale_io)
        k.launch(grid=(idx_m, 1, 1), block=(BLOCK_THREADS, 1, 1), stream=stream)

    return launch


@lru_cache(maxsize=32)
def _compile(*, device_index: int, head_dim: int, rotate: bool, mode: str):
    # device_index scopes the cached compiled launcher to its HIP context.
    launcher = _build_kernel(head_dim=head_dim, rotate=rotate, mode=mode)
    launcher.compile_hints = {
        "waves_per_eu": 8,
        "fast_fp_math": True,
        "unsafe_fp_math": True,
    }
    return launcher


def _build_qkv_d64_kernel(*, rotate: bool, mode: str):
    """Build one D64 Q/K/V producer pass.

    A workgroup has three wave32s.  Each wave owns the same row of Q, K, or V,
    respectively, so its load/transform/store path is wave-uniform.  This avoids
    cross-wave synchronization and, unlike atomics, leaves one independent amax
    partial per tensor and row for the batched torch reduction.
    """
    assert mode in ("amax", "scale")
    D = 64
    VEC = 2
    _kname = f"fp8_qkv_{mode}_D64{'_rot' if rotate else ''}_flydsl"

    @flyc.kernel(name=_kname, known_block_size=[96, 1, 1])
    def kernel(
        q_in: fx.Tensor,
        k_in: fx.Tensor,
        v_in: fx.Tensor,
        q_out: fx.Tensor,
        k_out: fx.Tensor,
        v_out: fx.Tensor,
        partials_or_scales: fx.Tensor,
        M: fx.Int32,
    ):
        row = fx.block_idx.x
        tid = fx.thread_idx.x
        wave = tid // fx.Int32(32)
        lane = tid % fx.Int32(32)
        row_idx = fx.Int64(row)
        q_in_i32 = ptr_buf_tensor(q_in, fx.Int32)
        k_in_i32 = ptr_buf_tensor(k_in, fx.Int32)
        v_in_i32 = ptr_buf_tensor(v_in, fx.Int32)
        q_out_i32 = ptr_buf_tensor(q_out, fx.Int32)
        k_out_i32 = ptr_buf_tensor(k_out, fx.Int32)
        v_out_i32 = ptr_buf_tensor(v_out, fx.Int32)
        partials_or_scales_f32 = ptr_buf_tensor(partials_or_scales, fx.Float32)

        # This helper is called only from wave-uniform branches below. In
        # particular, use lane (not workgroup tid) for all row-local offsets and
        # wave shuffles: wave 1/2 otherwise address and shuffle the wrong row.
        def process(x_in_i32, x_out_i32, row_count, do_rotate):
            row_off_elems = row_idx * D + fx.Int64(lane) * VEC
            row_off_dw = fx.Int32(row_off_elems // 2)
            x_raw = buf_copy_load(x_in_i32, row_off_dw)
            x_raw = fx.Vector.from_elements([x_raw], fx.Int32)
            x_bf16 = fx.Vector(x_raw).bitcast(fx.BFloat16)
            x0 = x_bf16[0].to(fx.Float32)
            x1 = x_bf16[1].to(fx.Float32)

            if const_expr(do_rotate):
                # First butterfly is within this lane. The remaining five use
                # only its wave32 lanes, never the 96-thread workgroup width.
                a, b = x0, x1
                x0 = a + b
                x1 = a - b
                lane_i32 = lane
                for st in range_constexpr(5):
                    off = 1 << st
                    is_high = (lane_i32 & fx.Int32(off)) != fx.Int32(0)
                    p0 = x0.shuffle_xor(off, 32)
                    p1 = x1.shuffle_xor(off, 32)
                    lo0 = x0 + p0
                    lo1 = x1 + p1
                    hi0 = p0 - x0
                    hi1 = p1 - x1
                    x0 = is_high.select(hi0, lo0)
                    x1 = is_high.select(hi1, lo1)
                norm = fx.Float32(0.125)
                x0 = x0 * norm
                x1 = x1 * norm

            if const_expr(mode == "amax"):
                am = fmath.absf(x0).maximumf(fmath.absf(x1))
                for off in (16, 8, 4, 2, 1):
                    am = am.maximumf(am.shuffle_xor(off, 32))
                if lane == fx.Int32(0):
                    partial_idx = fx.Int64(wave) * fx.Int64(row_count) + row_idx
                    buf_copy_store(
                        partials_or_scales_f32,
                        fx.Int32(partial_idx),
                        am,
                        elem=fx.Float32,
                    )
            else:
                scale = buf_copy_load(partials_or_scales_f32, wave, elem=fx.Float32)
                inv_scale = fx.Float32(1.0) / scale
                c_max = fx.Float32(_FP8_MAX)
                c_min = fx.Float32(-_FP8_MAX)
                q0 = (x0 * inv_scale).maximumf(c_min).minimumf(c_max)
                q1 = (x1 * inv_scale).maximumf(c_min).minimumf(c_max)
                # Keep this raw intrinsic: FlyDSL 0.3.4.1 has no typed packed-FP8 conversion.
                pk = rocdl.cvt_pk_fp8_f32(
                    fx.Int32.ir_type,
                    q0.ir_value(),
                    q1.ir_value(),
                    fx.Int32(0).ir_value(),
                    0,
                )
                peer_pk = fx.Int32(pk).shuffle_xor(1, 32)
                if lane % fx.Int32(2) == fx.Int32(0):
                    out_off_dw = fx.Int32(row_idx * (D // 4) + lane // fx.Int32(2))
                    buf_copy_store(
                        x_out_i32,
                        out_off_dw,
                        fx.Int32(pk) | (fx.Int32(peer_pk) << fx.Int32(16)),
                    )

        if wave == fx.Int32(0):
            process(q_in_i32, q_out_i32, M, rotate)
        elif wave == fx.Int32(1):
            process(k_in_i32, k_out_i32, M, rotate)
        else:
            process(v_in_i32, v_out_i32, M, False)

    @flyc.jit
    def launch(
        q_in: fx.Tensor,
        k_in: fx.Tensor,
        v_in: fx.Tensor,
        q_out: fx.Tensor,
        k_out: fx.Tensor,
        v_out: fx.Tensor,
        partials_or_scales: fx.Tensor,
        M: fx.Int32,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        k = kernel(q_in, k_in, v_in, q_out, k_out, v_out, partials_or_scales, M)
        k.launch(grid=(fx.Int64(M), 1, 1), block=(96, 1, 1), stream=stream)

    return launch


@lru_cache(maxsize=8)
def _compile_qkv_d64(*, device_index: int, rotate: bool, mode: str):
    launcher = _build_qkv_d64_kernel(rotate=rotate, mode=mode)
    launcher.compile_hints = {
        "waves_per_eu": 8,
        "fast_fp_math": True,
        "unsafe_fp_math": True,
    }
    return launcher


def flydsl_fp8_qkv_d64_quant(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, rotate: bool
):
    """Fused same-shape D64 Q/K/V FP8 producer with three independent scales."""
    assert q.shape == k.shape == v.shape and q.shape[-1] == 64
    if not (q.is_contiguous() and k.is_contiguous() and v.is_contiguous()):
        raise ValueError("flydsl_fp8_qkv_d64_quant requires contiguous Q/K/V")

    stream = torch.cuda.current_stream(q.device)
    with torch.cuda.device(q.device), torch.cuda.stream(stream):
        M = q.numel() // 64
        q8 = torch.empty_like(q, dtype=_FP8_DTYPE)
        k8 = torch.empty_like(k, dtype=_FP8_DTYPE)
        v8 = torch.empty_like(v, dtype=_FP8_DTYPE)
        # Layout [tensor, row]: a single torch reduction produces the three
        # independent global amax values without atomic contention.
        partials = torch.empty((3, M), dtype=torch.float32, device=q.device)
        fx_stream = Stream(stream)
        amax_k = _compile_qkv_d64(
            device_index=q.device.index, rotate=rotate, mode="amax"
        )
        _run_compiled(
            amax_k,
            q,
            k,
            v,
            q8,
            k8,
            v8,
            partials,
            M,
            fx_stream,
        )
        scales = (partials.amax(dim=1) / _FP8_MAX).clamp(min=1e-12)
        scale_k = _compile_qkv_d64(
            device_index=q.device.index, rotate=rotate, mode="scale"
        )
        _run_compiled(
            scale_k,
            q,
            k,
            v,
            q8,
            k8,
            v8,
            scales,
            M,
            fx_stream,
        )
    return q8, k8, v8, scales[0:1], scales[1:2], scales[2:3]


def flydsl_fp8_pertensor_quant(
    x: torch.Tensor,
    *,
    rotate: bool,
    out: torch.Tensor = None,
):
    """2-pass per-tensor fp8 quant (+ optional FWHT rotation) of ``x``, fully
    FlyDSL. ``x`` last dim is ``head_dim`` (flattened to [M, D]).

    Returns ``(x_fp8, scale)`` where ``scale`` is a 1-element f32 tensor
    (``real = fp8 * scale``). Rotation requires a power-of-two head dimension;
    the current packed specialization supports 64, 128, 256, 512, and 1024.
    """
    if x.dtype != torch.bfloat16:
        raise TypeError(
            "flydsl_fp8_pertensor_quant requires bfloat16 input, " f"got {x.dtype}"
        )
    if not x.is_cuda:
        raise ValueError("flydsl_fp8_pertensor_quant requires a GPU tensor")
    if x.dim() == 0 or x.shape[-1] not in (64, 128, 256, 512, 1024):
        got = None if x.dim() == 0 else x.shape[-1]
        raise ValueError(
            "flydsl_fp8_pertensor_quant supports head_dim in "
            f"{{64, 128, 256, 512, 1024}}, got {got}"
        )
    if x.numel() == 0:
        raise ValueError("flydsl_fp8_pertensor_quant requires a non-empty tensor")
    if not x.is_contiguous():
        raise ValueError("flydsl_fp8_pertensor_quant requires a contiguous input")

    D = x.shape[-1]
    if rotate and x.shape[-1] & (x.shape[-1] - 1):
        raise ValueError("FWHT rotation requires a power-of-two head_dim")

    expected_shape = tuple(x.shape)
    caller_out = out
    if caller_out is not None and (
        tuple(caller_out.shape) != expected_shape
        or caller_out.dtype != _FP8_DTYPE
        or caller_out.device != x.device
        or not caller_out.is_contiguous()
    ):
        raise ValueError(
            "out must be a contiguous float8_e4m3fn tensor with shape "
            f"{expected_shape} on {x.device}"
        )
    if caller_out is not None and _storage_overlaps(x, caller_out):
        raise ValueError("out must not overlap input storage")

    stream = torch.cuda.current_stream(x.device)
    with torch.cuda.device(x.device), torch.cuda.stream(stream):
        M = x.numel() // D
        if caller_out is None:
            out = torch.empty_like(x, dtype=_FP8_DTYPE)
        else:
            out = caller_out
        partials = torch.empty(M, dtype=torch.float32, device=x.device)
        fx_stream = Stream(stream)

        # Pass 1 writes per-row amax partials; x_out is unused.
        amax_k = _compile(
            device_index=x.device.index, head_dim=D, rotate=rotate, mode="amax"
        )
        _run_compiled(
            amax_k,
            x,
            x,
            partials,
            M,
            fx_stream,
        )
        scale = (partials.amax() / _FP8_MAX).clamp(min=1e-12).reshape(1)

        # Pass 2 scales, clamps, and casts using the global descale.
        scale_k = _compile(
            device_index=x.device.index, head_dim=D, rotate=rotate, mode="scale"
        )
        _run_compiled(
            scale_k,
            x,
            out,
            scale,
            M,
            fx_stream,
        )

    return out, scale
