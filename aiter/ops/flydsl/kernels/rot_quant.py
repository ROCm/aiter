# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Fused online Hadamard rotation + MXFP4 quantization (FlyDSL, gfx950).

``flydsl_rot_quant(x, RS)`` rotates each RS-wide block of a bf16 activation by a
normalized Sylvester Hadamard matrix, quantizes the result to MXFP4 and writes
the e8m0 block scales, optionally already in the CK layout that
``aiter.gemm_a4w4`` consumes. Rotated MXFP4 checkpoints fold the rotation into
the weights, so this runs on the activation of every rotated projection.
Unfused, it is a Hadamard matmul followed by a separate quant pass.

One thread owns one RS-wide block, so the FWHT runs in registers with no
cross-lane traffic, and the RS/32 MXFP4 groups of a block never straddle
threads.
"""

# Do not add `from __future__ import annotations`: flydsl detects runtime
# parameters from the annotation objects, and a stringified `fx.Int32` would
# make `num_blocks` a compile-time constant (one JIT compile per M).

import math
from functools import cache

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import T

from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops.flydsl.kernels.quant_utils import emit_mx_e8m0_scale
from aiter.utility.mx_types import MX_DEFAULT_ROUND_MODE, MxDtypeInt

from .tensor_shim import _run_compiled, ptr_arg

MX_GROUP = 32  # elements per e8m0 scale
VEC = 8  # bf16 per 128-bit load
# One wave per CTA. On MI350X wider CTAs were no faster over 72 (RS, K, M)
# shapes up to M=16384 and up to 31% slower at RS=128.
BLOCK = 64
SUPPORTED_RS = (32, 64, 128)


def _fwht(vals):
    """Unnormalized Sylvester FWHT in registers, stages h = 1, 2, ..., n/2."""
    h = 1
    while h < len(vals):
        vals = [
            vals[i ^ h] - vals[i] if i & h else vals[i] + vals[i ^ h]
            for i in range(len(vals))
        ]
        h *= 2
    return vals


def _amax(vals):
    """Balanced abs-max tree; any order is exact for max over |v|."""
    vals = [fx.absf(v) for v in vals]
    while len(vals) > 1:
        vals = [fx.max(vals[i], vals[i + 1]) for i in range(0, len(vals), 2)]
    return vals[0]


def _ck_scale_offset(row, col, padded_cols):
    """Byte offset of scale (row, col) in the aiter.ops.shuffle.shuffle_scale
    layout; same index split as the e8m0 write in silu_and_mul_fq.py."""
    return (
        (row >> 5) * (padded_cols * 32)
        + (col >> 3) * 256
        + (col & 3) * 64
        + (row & 15) * 4
        + ((col >> 2) & 1) * 2
        + ((row >> 4) & 1)
    )


@cache
def _build_rot_quant(K: int, RS: int, shuffle_scales: bool):
    groups = RS // MX_GROUP  # e8m0 scales per thread
    words = RS // 8  # packed fp4 i32 words per thread
    blocks_per_row = K // RS
    scale_cols = K // MX_GROUP
    padded_scale_cols = (scale_cols + 7) // 8 * 8
    # When 1/sqrt(RS) is a power of two (RS=64) it is applied to each group's
    # amax and to the conversion exponent instead of to every element. Scaling
    # by a power of two is exact, so the scales and fp4 bytes are unchanged.
    half_log2 = 0.5 * math.log2(RS)
    fold = int(half_log2) if half_log2.is_integer() else 0
    norm = 1.0 / math.sqrt(RS)

    @flyc.kernel(known_block_size=[BLOCK, 1, 1])
    def rot_quant_kernel(
        x: fx.Pointer,
        xq: fx.Pointer,
        xs: fx.Pointer,
        num_blocks: fx.Int32,
    ):
        blk = fx.block_idx.x * BLOCK + fx.thread_idx.x

        # Buffer views bounded by num_blocks: threads past the end load zeros
        # and their fp4 and row-major scale stores are dropped by the descriptor.
        def _thread_slice(ptr, width, atom_width):
            view = fx.make_view(ptr, fx.make_layout(num_blocks * width, 1))
            buf = fx.rocdl.make_buffer_tensor(view, max_size=False)
            row = fx.slice(
                fx.logical_divide(buf, fx.make_layout(width, 1)), (None, blk)
            )
            return fx.logical_divide(row, fx.make_layout(atom_width, 1))

        x_blk = _thread_slice(x, RS, VEC)
        xq_blk = _thread_slice(xq, words, 4)
        copy128 = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.BFloat16)
        x_frag = fx.make_fragment_like(x_blk)
        fx.copy(copy128, x_blk, x_frag)
        x_vec = x_frag.load()
        vals = _fwht([x_vec[i].to(fx.Float32) for i in range_constexpr(RS)])
        if const_expr(fold):
            vals = [v.to(fx.BFloat16).to(fx.Float32) for v in vals]
        else:
            vals = [(v * norm).to(fx.BFloat16).to(fx.Float32) for v in vals]

        packed_words = []
        scales = []
        for g in range_constexpr(groups):
            grp = vals[g * MX_GROUP : (g + 1) * MX_GROUP]
            amax = _amax(grp)
            if const_expr(fold):
                amax = amax * norm
            e8m0 = fx.Int32(
                emit_mx_e8m0_scale(
                    amax, mode=MX_DEFAULT_ROUND_MODE, dtype=MxDtypeInt.FP4_E2M1
                )
            )
            scales.append(e8m0.to(fx.Int8))
            # Scale for the unnormalized values; min keeps a NaN group in range.
            exp = fx.min(e8m0 + fold, 255) if const_expr(fold) else e8m0
            scale = (exp << 23).bitcast(fx.Float32)
            for w in range_constexpr(MX_GROUP // 8):
                packed = fx.Int32(0)
                for p in range_constexpr(4):
                    packed = fx.rocdl.cvt_scalef32_pk_fp4_f32(
                        T.i32,
                        packed,
                        grp[w * 8 + 2 * p],
                        grp[w * 8 + 2 * p + 1],
                        scale,
                        p,
                    )
                packed_words.append(fx.Int32(packed))

        xq_frag = fx.make_fragment_like(xq_blk)
        xq_frag.store(fx.Vector.from_elements(packed_words, fx.Int32))
        fx.copy(fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.Int32), xq_frag, xq_blk)

        if const_expr(shuffle_scales):
            # Byte scatter with no contiguous run per thread, so plain indexed
            # stores rather than a copy atom.
            if blk < num_blocks:
                row = blk // blocks_per_row
                col_block = blk % blocks_per_row
                for g in range_constexpr(groups):
                    col = col_block * groups + g
                    xs[_ck_scale_offset(row, col, padded_scale_cols)] = scales[g]
                if const_expr(  # noqa: SIM102 - preserve DSL staging
                    padded_scale_cols > scale_cols
                ):
                    # Pad columns get e8m0 1.0, as written by shuffle_scale,
                    # rather than whatever the allocation held.
                    if col_block == blocks_per_row - 1:
                        for col in range_constexpr(scale_cols, padded_scale_cols):
                            off = _ck_scale_offset(row, col, padded_scale_cols)
                            xs[off] = fx.Int8(0x7F)
        else:
            xs_blk = _thread_slice(xs, groups, groups)
            xs_frag = fx.make_fragment_like(xs_blk)
            xs_frag.store(fx.Vector.from_elements(scales, fx.Int8))
            fx.copy(
                fx.make_copy_atom(fx.rocdl.BufferCopy(8 * groups), fx.Int8),
                xs_frag,
                xs_blk,
            )

    @flyc.jit
    def launch(
        x: fx.Pointer,
        xq: fx.Pointer,
        xs: fx.Pointer,
        num_blocks: fx.Int32,
        grid_x: fx.Int32,
        stream: fx.Stream,
    ):
        rot_quant_kernel(x, xq, xs, num_blocks).launch(
            grid=(grid_x, 1, 1), block=(BLOCK, 1, 1), stream=stream
        )

    # SLP packs the butterflies into v_pk_add_f32, and the moves its aligned
    # register pairs need made the kernel up to 25% slower on MI350X; where it
    # helped (RS=64), small edits elsewhere flipped it into a loss.
    launch.compile_hints.setdefault("llvm_options", {})["vectorize-slp"] = False
    return launch


def flydsl_rot_quant(x: torch.Tensor, RS: int = 64, shuffle_scales: bool = False):
    """Fused block-diagonal Hadamard rotation + MXFP4 quantization.

    Quantizes ``(x.view(M, K // RS, RS).float() @ (H / sqrt(RS))).to(bf16)``
    to MXFP4 with aiter's default RoundUp e8m0 scale, where ``H`` is the
    Sylvester Hadamard matrix ``scipy.linalg.hadamard(RS)``; the result equals
    ``per_1x32_f4_quant`` applied to that rotated tensor. The weight must be
    rotated by the same matrix so that ``(x H)(w H)^T = x w^T``.

    Args:
        x: contiguous bf16 ``[M, K]``.
        RS: rotation size, one of ``SUPPORTED_RS``; ``K`` must be a multiple.
        shuffle_scales: write the scales in the CK layout of
            ``aiter.ops.shuffle.shuffle_scale`` instead of row-major.

    Returns:
        ``(xq, xs)`` as from ``per_1x32_f4_quant_hip``: ``xq`` is fp4x2
        ``[M, K // 2]``. ``xs`` is e8m0, ``[M, K // 32]``, or with
        ``shuffle_scales=True`` ``[ceil256(M), ceil8(K // 32)]``, where the
        padding columns hold 1.0 as in ``shuffle_scale`` and the padding rows
        are left uninitialized.
    """
    if RS not in SUPPORTED_RS:
        raise ValueError(f"RS must be one of {SUPPORTED_RS}, got {RS}")
    if x.dim() != 2:
        raise ValueError(f"x must be 2D [M, K], got shape {tuple(x.shape)}")
    if x.dtype != torch.bfloat16:
        raise ValueError(f"x must be bf16, got {x.dtype}")
    if not x.is_contiguous():
        raise ValueError("x must be contiguous")
    M, K = x.shape
    if K % RS:
        raise ValueError(f"K={K} must be a multiple of RS={RS}")
    # The kernel addresses x with signed 32-bit byte offsets.
    if x.numel() * x.element_size() >= 2**31:
        raise ValueError(f"x is too large for 32-bit offsets: {tuple(x.shape)}")
    gfx = get_gfx_runtime()
    if gfx != "gfx950":
        raise RuntimeError(f"flydsl_rot_quant requires gfx950, got {gfx}")

    xq = torch.empty((M, K // 2), dtype=dtypes.fp4x2, device=x.device)
    if shuffle_scales:
        scale_shape = ((M + 255) // 256 * 256, (K // MX_GROUP + 7) // 8 * 8)
    else:
        scale_shape = (M, K // MX_GROUP)
    xs = torch.empty(scale_shape, dtype=dtypes.fp8_e8m0, device=x.device)
    if M == 0:
        return xq, xs

    num_blocks = M * (K // RS)
    _run_compiled(
        _build_rot_quant(K, RS, shuffle_scales),
        ptr_arg(x, fx.BFloat16),
        ptr_arg(xq, fx.Int32),
        ptr_arg(xs, fx.Int8),
        num_blocks,
        (num_blocks + BLOCK - 1) // BLOCK,
        torch.cuda.current_stream(x.device),
    )
    return xq, xs
