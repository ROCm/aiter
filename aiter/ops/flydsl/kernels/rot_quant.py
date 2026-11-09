# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Fused online Hadamard rotation + MXFP4 quantization (FlyDSL, gfx950).

``flydsl_rot_quant(x, rotation_size=RS)`` rotates each RS-wide block of a bf16
activation by a normalized Sylvester Hadamard matrix, quantizes the result to
MXFP4 and writes the e8m0 block scales, optionally already in the CK layout that
``aiter.gemm_a4w4`` consumes. Rotated MXFP4 checkpoints fold the rotation into
the weights, so this runs on the activation of every rotated projection.
Unfused, it is a Hadamard matmul followed by a separate quant pass.

One thread owns one 32-element MXFP4 group: it runs the first five FWHT
stages in registers, and the RS/32 adjacent lanes of a block finish the
remaining stages with lane shuffles. Each group's scale is then local to its
thread.
"""

# Do not add `from __future__ import annotations`: flydsl detects runtime
# parameters from the annotation objects, and a stringified `fx.Int32` would
# make `num_groups` a compile-time constant (one JIT compile per M).

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
# One wave per CTA. Over 36 (RS, K, M) shapes on MI350X, 128/256/512 threads
# were at most 4% faster at any shape and up to 29% slower at others.
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


def _fwht_across_lanes(vals, lanes, lane):
    """The FWHT stages that span `lanes` adjacent lanes: stage s pairs a lane
    with lane ^ s; the lower lane keeps a + b and the upper one a - b, the same
    pairing _fwht uses in registers."""
    s = 1
    while s < lanes:
        upper = (lane & s) != 0
        pairs = [(v, v.shuffle_xor(s, 64)) for v in vals]
        vals = [upper.select(p - v, v + p) for v, p in pairs]
        s *= 2
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


def _padded_scale_cols(K):
    """Scale columns of the CK layout: K // 32 padded to a multiple of 8."""
    return (K // MX_GROUP + 7) // 8 * 8


def _fold(RS):
    """When 1/sqrt(RS) is a power of two (RS=64) it is applied to the group's
    amax and to the conversion exponent instead of to every element. Scaling
    by a power of two is exact, so the scales and fp4 bytes are unchanged."""
    half_log2 = 0.5 * math.log2(RS)
    return int(half_log2) if half_log2.is_integer() else 0


def _group_slice(ptr, num_groups, width, atom_width, grp):
    """Group `grp` of a buffer view bounded by `num_groups` groups of `width`
    elements: out-of-range groups load zeros and their stores are dropped."""
    view = fx.make_view(ptr, fx.make_layout(num_groups * width, 1))
    buf = fx.rocdl.make_buffer_tensor(view, max_size=False)
    row = fx.slice(fx.logical_divide(buf, fx.make_layout(width, 1)), (None, grp))
    return fx.logical_divide(row, fx.make_layout(atom_width, 1))


def _rot_quant_group(x_grp, RS):
    """Rotate and quantize this lane's 32-element group; the RS/32 adjacent
    lanes of a block must all call it. Returns (e8m0 as i32, 4 packed words)."""
    fold = _fold(RS)
    norm = 1.0 / math.sqrt(RS)
    x_frag = fx.make_fragment_like(x_grp)
    fx.copy(fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.BFloat16), x_grp, x_frag)
    x_vec = x_frag.load()
    vals = _fwht([x_vec[i].to(fx.Float32) for i in range_constexpr(MX_GROUP)])
    vals = _fwht_across_lanes(vals, RS // MX_GROUP, fx.thread_idx.x)
    if const_expr(fold):
        vals = [v.to(fx.BFloat16).to(fx.Float32) for v in vals]
    else:
        vals = [(v * norm).to(fx.BFloat16).to(fx.Float32) for v in vals]

    amax = _amax(vals)
    if const_expr(fold):
        amax = amax * norm
    e8m0 = fx.Int32(
        emit_mx_e8m0_scale(amax, mode=MX_DEFAULT_ROUND_MODE, dtype=MxDtypeInt.FP4_E2M1)
    )
    # Scale for the unnormalized values; min keeps a NaN group in range.
    exp = fx.min(e8m0 + fold, 255) if const_expr(fold) else e8m0
    scale = (exp << 23).bitcast(fx.Float32)
    words = []
    for w in range_constexpr(MX_GROUP // 8):
        packed = fx.Int32(0)
        for p in range_constexpr(4):
            packed = fx.rocdl.cvt_scalef32_pk_fp4_f32(
                T.i32, packed, vals[w * 8 + 2 * p], vals[w * 8 + 2 * p + 1], scale, p
            )
        words.append(fx.Int32(packed))
    return e8m0, words


def _store_fp4(xq_grp, words):
    xq_frag = fx.make_fragment_like(xq_grp)
    xq_frag.store(fx.Vector.from_elements(words, fx.Int32))
    fx.copy(fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.Int32), xq_frag, xq_grp)


# SLP packs the butterflies into v_pk_add_f32, and the moves its aligned
# register pairs need cost up to 24% on MI350X; its best case was 6%.
NO_SLP = {"vectorize-slp": False}


@cache
def _build_rot_quant(K: int, RS: int, shuffle_scales: bool):
    scale_cols = K // MX_GROUP
    padded_scale_cols = _padded_scale_cols(K)

    @flyc.kernel(known_block_size=[BLOCK, 1, 1])
    def rot_quant_kernel(
        x: fx.Pointer,
        xq: fx.Pointer,
        xs: fx.Pointer,
        num_groups: fx.Int32,
    ):
        grp = fx.block_idx.x * BLOCK + fx.thread_idx.x
        e8m0, words = _rot_quant_group(
            _group_slice(x, num_groups, MX_GROUP, VEC, grp), RS
        )
        _store_fp4(_group_slice(xq, num_groups, MX_GROUP // 8, 4, grp), words)

        if const_expr(shuffle_scales):
            # Byte scatter, so a plain indexed store rather than a copy atom.
            if grp < num_groups:
                row = grp // scale_cols
                col = grp % scale_cols
                xs[_ck_scale_offset(row, col, padded_scale_cols)] = e8m0.to(fx.Int8)
                if const_expr(  # noqa: SIM102 - preserve DSL staging
                    padded_scale_cols > scale_cols
                ):
                    # Pad columns get e8m0 1.0, as written by shuffle_scale,
                    # rather than whatever the allocation held.
                    if col == scale_cols - 1:
                        for pad in range_constexpr(scale_cols, padded_scale_cols):
                            off = _ck_scale_offset(row, pad, padded_scale_cols)
                            xs[off] = fx.Int8(0x7F)
        else:
            xs_grp = _group_slice(xs, num_groups, 1, 1, grp)
            xs_frag = fx.make_fragment_like(xs_grp)
            xs_frag.store(fx.Vector.from_elements([e8m0.to(fx.Int8)], fx.Int8))
            fx.copy(
                fx.make_copy_atom(fx.rocdl.BufferCopy8b(), fx.Int8), xs_frag, xs_grp
            )

    @flyc.jit
    def launch(
        x: fx.Pointer,
        xq: fx.Pointer,
        xs: fx.Pointer,
        num_groups: fx.Int32,
        grid_x: fx.Int32,
        stream: fx.Stream,
    ):
        rot_quant_kernel(x, xq, xs, num_groups).launch(
            grid=(grid_x, 1, 1), block=(BLOCK, 1, 1), stream=stream
        )

    launch.compile_hints.setdefault("llvm_options", {}).update(NO_SLP)
    return launch


# 16 waves per CTA for the MoE kernel: most of its (row, group) threads find a
# padding row and exit, and 1024-thread CTAs cut the dispatch cost of those by
# up to 25% at M=16-64 on MI350X.
MOE_BLOCK = 1024


@cache
def _build_rot_quant_moe_sort(K: int, RS: int, block: int):
    scale_cols = K // MX_GROUP
    padded_scale_cols = _padded_scale_cols(K)

    @flyc.kernel(known_block_size=[block, 1, 1])
    def rot_quant_moe_sort_kernel(
        x: fx.Pointer,
        xq: fx.Pointer,
        xs: fx.Pointer,
        sorted_ids: fx.Pointer,
        num_valid_ids: fx.Pointer,
        num_tokens: fx.Int32,
        topk: fx.Int32,
    ):
        # One thread per (sorted row, group). A token is quantized once for
        # each of its routes; all of them write the same fp4 bytes. Rows past
        # num_valid_ids and the padding rows of each expert block are skipped:
        # their id is num_tokens, or with num_local_tokens topk << 24 | count,
        # a real token in an out-of-range slot. All lanes of a block share the
        # row.
        t = fx.block_idx.x * block + fx.thread_idx.x
        row = t // scale_cols
        col = t % scale_cols
        # Read sorted_ids only below num_valid_ids, which bounds its length.
        if row < num_valid_ids[0]:
            route = sorted_ids[row]
            token = route & 0xFFFFFF
            if token < num_tokens:  # noqa: SIM102 - preserve DSL staging
                if (route >> 24) & 0xFF < topk:
                    grp = token * scale_cols + col
                    num_groups = num_tokens * scale_cols
                    e8m0, words = _rot_quant_group(
                        _group_slice(x, num_groups, MX_GROUP, VEC, grp), RS
                    )
                    _store_fp4(
                        _group_slice(xq, num_groups, MX_GROUP // 8, 4, grp), words
                    )
                    off = _ck_scale_offset(row, col, padded_scale_cols)
                    xs[off] = e8m0.to(fx.Int8)

    @flyc.jit
    def launch(
        x: fx.Pointer,
        xq: fx.Pointer,
        xs: fx.Pointer,
        sorted_ids: fx.Pointer,
        num_valid_ids: fx.Pointer,
        num_tokens: fx.Int32,
        topk: fx.Int32,
        grid_x: fx.Int32,
        stream: fx.Stream,
    ):
        rot_quant_moe_sort_kernel(
            x, xq, xs, sorted_ids, num_valid_ids, num_tokens, topk
        ).launch(grid=(grid_x, 1, 1), block=(block, 1, 1), stream=stream)

    launch.compile_hints.setdefault("llvm_options", {}).update(NO_SLP)
    return launch


def _check_input(x, rotation_size):
    if rotation_size not in SUPPORTED_RS:
        raise ValueError(
            f"rotation_size must be one of {SUPPORTED_RS}, got {rotation_size}"
        )
    if x.dim() != 2:
        raise ValueError(f"x must be 2D [M, K], got shape {tuple(x.shape)}")
    if x.dtype != torch.bfloat16:
        raise ValueError(f"x must be bf16, got {x.dtype}")
    if not x.is_contiguous():
        raise ValueError("x must be contiguous")
    M, K = x.shape
    if K % rotation_size:
        raise ValueError(f"K={K} must be a multiple of rotation_size={rotation_size}")
    # The kernel addresses x with signed 32-bit byte offsets.
    if x.numel() * x.element_size() >= 2**31:
        raise ValueError(f"x is too large for 32-bit offsets: {tuple(x.shape)}")
    gfx = get_gfx_runtime()
    if gfx != "gfx950":
        raise RuntimeError(f"flydsl_rot_quant requires gfx950, got {gfx}")
    return M, K


def flydsl_rot_quant(
    x: torch.Tensor, *, rotation_size: int = 64, shuffle: bool = False
):
    """Fused block-diagonal Hadamard rotation + MXFP4 quantization.

    Quantizes ``(x.view(M, K // RS, RS).float() @ (H / sqrt(RS))).to(bf16)``
    to MXFP4 with aiter's default RoundUp e8m0 scale, where ``RS`` is
    ``rotation_size`` and ``H`` the Sylvester Hadamard matrix
    ``scipy.linalg.hadamard(RS)``; the result equals ``per_1x32_f4_quant``
    applied to that rotated tensor. The weight must be rotated by the same
    matrix so that ``(x H)(w H)^T = x w^T``.

    Args:
        x: contiguous bf16 ``[M, K]``.
        rotation_size: one of ``SUPPORTED_RS``; ``K`` must be a multiple.
        shuffle: write the scales in the CK layout of
            ``aiter.ops.shuffle.shuffle_scale`` instead of row-major.

    Returns:
        ``(xq, xs)`` as from ``per_1x32_f4_quant_hip``: ``xq`` is fp4x2
        ``[M, K // 2]``. ``xs`` is e8m0, ``[M, K // 32]``, or with
        ``shuffle=True`` ``[ceil256(M), ceil8(K // 32)]``, where the padding
        columns hold 1.0 as in ``shuffle_scale`` and the padding rows are left
        uninitialized.
    """
    M, K = _check_input(x, rotation_size)
    xq = torch.empty((M, K // 2), dtype=dtypes.fp4x2, device=x.device)
    if shuffle:
        scale_shape = ((M + 255) // 256 * 256, _padded_scale_cols(K))
    else:
        scale_shape = (M, K // MX_GROUP)
    xs = torch.empty(scale_shape, dtype=dtypes.fp8_e8m0, device=x.device)
    if M == 0:
        return xq, xs

    num_groups = M * (K // MX_GROUP)
    with torch.cuda.device(x.device):
        _run_compiled(
            _build_rot_quant(K, rotation_size, shuffle),
            ptr_arg(x, fx.BFloat16),
            ptr_arg(xq, fx.Int32),
            ptr_arg(xs, fx.Int8),
            num_groups,
            (num_groups + BLOCK - 1) // BLOCK,
            torch.cuda.current_stream(x.device),
        )
    return xq, xs


def flydsl_rot_quant_moe_sort(
    x: torch.Tensor,
    sorted_ids: torch.Tensor,
    num_valid_ids: torch.Tensor,
    token_num: int,
    topk: int,
    block_size: int,
    *,
    rotation_size: int = 64,
):
    """:func:`flydsl_rot_quant` for the MoE stage-1 input, with the scales
    sorted for the MoE GEMMs.

    Takes the leading arguments of ``fused_dynamic_mxfp4_quant_moe_sort`` and
    equals it applied to the rotated tensor of :func:`flydsl_rot_quant`,
    without that op's amax floor: ``sorted_ids``, ``num_valid_ids`` and
    ``block_size`` come from ``moe_sorting`` (with or without
    ``num_local_tokens``), ``token_num`` is ``M`` and ``topk`` is the routing
    top-k.

    Returns:
        ``(xq, xs)``: ``xq`` is fp4x2 ``[M, K // 2]`` in token order; the rows
        of tokens with no valid route may be left unwritten. ``xs`` is e8m0
        ``[ceil32(len(sorted_ids)), ceil8(K // 32)]`` in the layout of
        ``mxfp4_moe_sort_fwd``; as there, only the first ``K // 32`` columns
        of the rows of valid routes are defined.
    """
    M, K = _check_input(x, rotation_size)
    if M != token_num:
        raise ValueError(f"x has {M} rows, expected token_num={token_num}")
    # The fused kernel quantizes a token once per route, so past this much work
    # one rotation pass plus mxfp4_moe_sort_fwd is faster (MI350X, E=128,
    # topk=8, K=2048-7168).
    if M * topk * K > 1 << 21:
        from aiter.ops.quant import mxfp4_moe_sort_fwd

        xq, xs = flydsl_rot_quant(x, rotation_size=rotation_size)
        # mxfp4_moe_sort_fwd drops the last token's final scales when K // 32
        # is not a multiple of 4 (#7646); the extra zero columns avoid that and
        # land in the output's padding columns. Drop this once #7653 is in.
        pad = -(K // MX_GROUP) % 4
        if pad:
            xs = torch.nn.functional.pad(xs.view(torch.uint8), (0, pad))
            xs = xs.view(dtypes.fp8_e8m0)
        cols = K + pad * MX_GROUP
        return xq, mxfp4_moe_sort_fwd(xs, sorted_ids, num_valid_ids, M, cols)

    xq = torch.empty((M, K // 2), dtype=dtypes.fp4x2, device=x.device)
    sorted_rows = sorted_ids.shape[0]
    xs = torch.empty(
        ((sorted_rows + 31) // 32 * 32, _padded_scale_cols(K)),
        dtype=dtypes.fp8_e8m0,
        device=x.device,
    )
    if M == 0 or sorted_rows == 0:
        return xq, xs
    # At most M * topk experts are active and each pads its routes to whole
    # blocks, so valid rows never exceed M * topk * block_size.
    rows = min(sorted_rows, M * topk * block_size)
    threads = rows * (K // MX_GROUP)
    with torch.cuda.device(x.device):
        _run_compiled(
            _build_rot_quant_moe_sort(K, rotation_size, MOE_BLOCK),
            ptr_arg(x, fx.BFloat16),
            ptr_arg(xq, fx.Int32),
            ptr_arg(xs, fx.Int8),
            ptr_arg(sorted_ids, fx.Int32),
            ptr_arg(num_valid_ids, fx.Int32),
            M,
            topk,
            (threads + MOE_BLOCK - 1) // MOE_BLOCK,
            torch.cuda.current_stream(x.device),
        )
    return xq, xs
