# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Factors for the Sol-Attn low-rank pooled value tilt (PoC), as mha_v4_packed(sol_tilt=...) reads them.

The tilt adds sum_i P_b (a_bi . q) b_bi to each pooled block's value: a rank-r stand-in for the
block's Cov(v, k) q term, which the pooled mean drops. The factors are the leading singular vectors
of each block's covariance in the kernel's own operand space (rotated, quantized K; quantized V).
sol_tilt_factors builds them for the per-tensor FP8 kernel, the one that reads them;
sol_tilt_planes builds them for any recipe, encoded in its own K and V formats.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import torch
import triton

from aiter.ops.triton._triton_kernels.attention.sol_tilt import (
    _sol_jensen_cap_kernel,
    _sol_q_head_absmax_kernel,
    _sol_q_tile_absmax_kernel,
    _sol_tilt_factor_kernel,
    _sol_tilt_pack_kernel,
    _sol_tilt_shift_kernel,
    _sol_tilt_scaled_kernel,
    _sol_tilt_stack_kernel,
    _sol_tilt_unit_kernel,
    _sol_tilt_valu_kernel,
)

# The FP8 row's score units per nat at unit softmax scale.
_TILT_LOGIT_UNIT = math.log2(math.e) * 128.0 / 65535.0
# The weights P * W are e4m3 and P reaches the e4m3 max, so |W| has to stay under about 1.
_TILT_W_MAX = 0.5
# Headroom under the e4m3 max for B, so its rounding cannot overflow.
_TILT_B_HEADROOM = 400.0 / 448.0
# The widest KV tile any Sol row reads K scales in, whole tiles at a time, so every plane's scale
# rows run zeroed to a multiple of this: E8M0 0 is 2^-127, while stray bytes reach inf.
_TILT_SCALE_ROW_TILE = 128


def _check_unit_stride(name: str, x: torch.Tensor) -> None:
    if x.stride(-1) != 1:
        raise ValueError(f"{name} needs a unit-stride head dim")


_ROTATION_SIGNS: dict[torch.device, torch.Tensor] = {}


def _rotation_signs(device: torch.device) -> torch.Tensor:
    """The +-1 matrix of the MX packers' hd128 Hadamard rotation, which is it times 128**-0.5."""
    signs = _ROTATION_SIGNS.get(device)
    if signs is None:
        from aiter.ops.mha_v4_quant import rotate_activation_hd128

        eye = torch.eye(128, dtype=torch.bfloat16, device=device).view(1, 128, 1, 128).contiguous()
        rows = torch.empty_like(eye)
        rotate_activation_hd128(rows, eye)
        signs = rows.view(128, 128).sign().contiguous()
        _ROTATION_SIGNS[device] = signs
    return signs


def sol_q_abs_max(
    q: torch.Tensor,
    q_tile: int = 256,
    q_scale: torch.Tensor | None = None,
    rotate: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """max |q| in q's code units: per tile of q_tile rows, and per head.

    Returns ((batch, q tiles, heads, d), (batch, heads, d)) bf16, exact for 8-bit Q. sol_jensen_cap
    reads the first and sol_tilt_factors the second; build them once and pass the pair to both.
    For the block-scaled rows the maxima are taken in the kernel's basis: q_scale, (batch, seqlen,
    heads, d / 32) E8M0, dequantizes an MXFP8 Q's codes, and rotate takes a packed row's unrotated
    source Q through the packers' hd128 rotation.
    """
    _check_unit_stride("q", q)
    batch, seqlen, heads, d = q.shape
    if rotate and d != 128:
        raise ValueError(f"the packers' rotation is hd128, got head dim {d}")
    if q_scale is not None and (tuple(q_scale.shape) != (batch, seqlen, heads, d // 32) or q_scale.stride(-1) != 1):
        raise ValueError(f"q_scale must be ({batch}, {seqlen}, {heads}, {d // 32}) with unit-stride groups")
    tiles = -(-seqlen // q_tile)
    out = torch.empty(batch, tiles, heads, d, dtype=torch.bfloat16, device=q.device)
    head = torch.empty(batch, heads, d, dtype=torch.bfloat16, device=q.device)
    scale = q_scale if q_scale is not None else out
    _sol_q_tile_absmax_kernel[(tiles, heads, batch)](
        q, scale, _rotation_signs(q.device) if rotate else out, out, seqlen, d**-0.5,
        q.stride(0), q.stride(1), q.stride(2),
        *(scale.stride()[:3] if q_scale is not None else (0, 0, 0)),
        out.stride(0), out.stride(1), out.stride(2),
        HAS_SCALE=q_scale is not None,
        ROTATE=rotate,
        Q_TILE=q_tile,
        HEAD_DIM=d,
    )
    _sol_q_head_absmax_kernel[(heads, batch)](
        out, head, tiles,
        out.stride(0), out.stride(1), out.stride(2),
        head.stride(0), head.stride(1),
        TILE_CHUNK=128,
        HEAD_DIM=d,
    )
    return out, head


def sol_jensen_cap(
    q: torch.Tensor,
    q_descale: torch.Tensor | None,
    k_descale: torch.Tensor | None,
    mean_k_var: torch.Tensor,
    softmax_scale: float,
    kv_tile: int,
    block_attn_mask: torch.Tensor | None = None,
    q_tile: int = 256,
    q_abs_max: tuple[torch.Tensor, torch.Tensor] | None = None,
    mean_k_var_scale: torch.Tensor | None = None,
    mean_k_var_pooled: torch.Tensor | None = None,
    q_multiplier: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    """Damp each block's Jensen term to at most ln(kv_tile) for every query that pools it.

    The term 0.5 * scale^2 * sum_d q_d^2 var_d is a block's log-mean-exp only while its scores are
    near Gaussian. A block of N keys stops being one past a score spread of sqrt(2 ln N): beyond it
    the mean is carried by the block's largest key and grows linearly in the spread, not
    quadratically. Late Wan layers reach spreads in the thousands of nats, where the uncapped term
    hands the pooled blocks the whole softmax. The bound is sum_d max_q q_d^2 var_d over the query
    tiles that pool the block, so a block is damped for all of its queries alike; the tiles that
    compute it exactly never read its term. Without block_attn_mask every query of the head counts.

    q: the per-tensor 8-bit or unscaled BF16 Q, BSHD; mean_k_var: sol_prepare's stored variance of
    the K codes, which for an 8-bit K is one byte wide and divided by SOL_ATTN_FP8_VAR_DIVISOR;
    q_descale/k_descale: the one-element per-tensor descales, or both None for an unscaled Q/K;
    block_attn_mask: sol_prepare's (batch, heads, q tiles, blocks) selection at q_tile rows a tile;
    q_abs_max: sol_q_abs_max(q, q_tile), built here when omitted.

    A block-scaled K (MXFP8, MXFP6 or MXFP4) passes mean_k_var_scale, both descales None, and
    q_abs_max taken in the kernel's basis (see sol_q_abs_max), whose units are the real Q times
    q_multiplier, the factor its quantizer folds in. MXFP8 decodes mean_k_var by its scale; a
    packed K, whose codes are not addressable, passes sol_prepare's mean_k_var_pooled. The gain
    goes into the variance's scale bytes as 2^floor(log2 h), and for MXFP6 also into the scale
    tails of a copy of its records, which is what that kernel reads.

    Returns (mean_k_var, mean_k_var_scale, gain): the damped variance and, for a block-scaled K, its
    new scale (else None), and the per-block gain h in (0, 1] as applied, (batch, blocks, heads)
    fp32; scale the block's tilt by sqrt(h) to match.
    """
    from aiter.ops.triton.attention.utils import SOL_ATTN_FP8_VAR_DIVISOR

    batch, seqlen_q, heads, d = q.shape
    block_scaled = mean_k_var_scale is not None
    var_in = mean_k_var_pooled if mean_k_var_pooled is not None else mean_k_var
    if tuple(var_in.shape[::2]) != (batch, heads) or var_in.shape[-1] != d:
        raise ValueError("the Jensen cap needs one K head per query head")
    if var_in.stride(-1) != 1:
        raise ValueError("mean_k_var needs a unit-stride head dim")
    if mean_k_var_pooled is not None and not block_scaled:
        raise ValueError("mean_k_var_pooled stands in for a packed K's variance; pass its mean_k_var_scale")
    if block_scaled:
        if q_descale is not None or k_descale is not None or q_abs_max is None:
            raise ValueError("a block-scaled K takes no descales and q_abs_max in the kernel's basis")
        if mean_k_var_pooled is None and mean_k_var.shape[-1] != d:
            raise ValueError("a packed K's variance needs mean_k_var_pooled")
        if mean_k_var_scale.stride(-1) != 1 or mean_k_var_scale.shape[-1] != d // 32:
            raise ValueError(f"mean_k_var_scale needs {d // 32} unit-stride E8M0 groups")
    if (q_descale is None) != (k_descale is None):
        raise ValueError("pass both Q/K descales, or neither for an unscaled Q/K")
    if q_descale is not None and (q_descale.numel() != 1 or k_descale.numel() != 1):
        raise ValueError(
            f"the Jensen cap takes per-tensor descales, got {q_descale.numel()} and "
            f"{k_descale.numel()} elements; pass None for an unscaled Q/K")
    q_tile_max = (q_abs_max if q_abs_max is not None else sol_q_abs_max(q, q_tile))[0]
    tiles, blocks = q_tile_max.shape[1], mean_k_var.shape[1]
    if tiles != -(-seqlen_q // q_tile):
        raise ValueError(f"q_tile_max has {tiles} tiles, Q {seqlen_q} rows at {q_tile}")
    if block_attn_mask is not None:
        if tuple(block_attn_mask.shape) != (batch, heads, tiles, blocks):
            raise ValueError(
                f"block_attn_mask is {tuple(block_attn_mask.shape)}, "
                f"expected {(batch, heads, tiles, blocks)}"
            )
        mask, mask_strides = block_attn_mask, block_attn_mask.stride()
    else:
        mask, mask_strides = q_tile_max, (0, 0, 0, 0)

    if block_scaled:
        per_nat = 0.5 * (softmax_scale / q_multiplier) ** 2
        var, var_scale = mean_k_var, mean_k_var_scale.clone()
    else:
        per_nat = 0.5 * softmax_scale**2 * (SOL_ATTN_FP8_VAR_DIVISOR if mean_k_var.element_size() == 1 else 1.0)
        var, var_scale = torch.empty_like(mean_k_var), None
    gain = torch.empty(batch, blocks, heads, dtype=torch.float32, device=q.device)
    scale_io = var_scale if block_scaled else gain
    if blocks:
        _sol_jensen_cap_kernel[(blocks, heads, batch)](
            q_tile_max, var_in, mask, var, gain,
            q_descale if q_descale is not None else gain,
            k_descale if k_descale is not None else gain,
            mean_k_var_scale if block_scaled else gain,
            scale_io,
            per_nat,
            math.log(kv_tile),
            tiles,
            q_tile_max.stride(0), q_tile_max.stride(1), q_tile_max.stride(2),
            var_in.stride(0), var_in.stride(1), var_in.stride(2),
            *mask_strides,
            gain.stride(0), gain.stride(1), gain.stride(2),
            *(var_scale.stride()[:3] if block_scaled else (0, 0, 0)),
            HAS_MASK=block_attn_mask is not None,
            HAS_DESCALE=q_descale is not None,
            HAS_VAR_SCALE=block_scaled and mean_k_var_pooled is None,
            EXPONENT_GAIN=block_scaled,
            TILE_CHUNK=32,
            HEAD_DIM=d,
        )
    if block_scaled and mean_k_var.shape[-1] * mean_k_var.element_size() * 8 == 6 * d:
        var = _refill_mxfp6_scale_tails(mean_k_var, var_scale)
    return var, var_scale, gain


def _refill_mxfp6_scale_tails(records: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """A copy of MXFP6 K records whose per-tile scale tails are rebuilt from `scale`.

    The MXFP6 K reader takes its scales from each 128-row tile's tail, a gather of the plain
    (batch, rows, heads, 4) image, and never from the image itself.
    """
    from aiter.ops.triton.quant import mxfp6_fmha_pack as fp6

    batch, rows, heads, _ = scale.shape
    tiles = -(-rows // 128)
    if not scale.is_contiguous() or records.stride(2) != tiles * fp6._K_TILE_BYTES:
        raise ValueError("an MXFP6 K scale is a contiguous image of the rows the records were packed from")
    size = batch * heads * tiles * fp6._K_TILE_BYTES + fp6.FP6_K_BUFFER_SLACK_BYTES
    flat = records.as_strided((size,), (1,)).clone()
    fp6._fill_k_scale_tail_kernel[(batch * heads * tiles,)](
        scale.view(-1), flat, rows, heads, tiles,
        TILE_BYTES=fp6._K_TILE_BYTES,
        SCALE_TAIL_OFFSET=fp6._K_SCALE_TAIL_OFFSET,
        BLOCK=1024,
        num_warps=4,
    )
    return flat.as_strided(records.shape, records.stride())


def sol_tilt_factors(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_descale: torch.Tensor,
    k_descale: torch.Tensor,
    v_descale: torch.Tensor,
    mean_k: torch.Tensor,
    mean_v: torch.Tensor,
    softmax_scale: float,
    rank: int,
    kv_tile: int,
    power_iters: int = 1,
    block_gain: torch.Tensor | None = None,
    q_abs_max: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int, torch.Tensor]]:
    """Factor planes and the sol_tilt triple for a rank-`rank` pooled value tilt.

    q/k/v are the per-tensor FP8 operands mha_v4_packed is launched with (BSHD, one K/V head per
    query head), and mean_k/mean_v the pooled K/V sol_prepare returned for them. Returns
    (mean_k, mean_v, sol_tilt): mean_k and mean_v are plane 0 of (1 + rank)-plane allocations whose
    planes 1..rank hold A_i and B_i, and sol_tilt is (rank, plane stride in bytes, shifts), the
    shifts a one-element int32 device tensor the kernel reads.
    Pass all three to mha_v4_packed in place of the pooled K/V. block_gain, (batch, blocks, heads),
    scales each block's tilt; pass sqrt of sol_jensen_cap's gain. q_abs_max is sol_q_abs_max(q)
    at any tile size; it is built here when omitted.

    A short last block is always computed exactly, so its planes are zero. |a . q| is bounded by
    sum_d max|q_d| |a_d| over the head's queries rather than by its true maximum, which costs a
    tenth of the gain on Wan and keeps the factors to one pass over Q. Nothing is read back to the
    host: the shifts come from the factors' maxima on the device, clamped to a signed byte.
    """
    for name, x in (("k", k), ("v", v)):
        if x.dtype != torch.float8_e4m3fn:
            raise ValueError(f"the tilt's {name} must be float8_e4m3fn, got {x.dtype}")
    if v.shape[-1] != k.shape[-1]:
        raise ValueError("the tilt's K and V planes share one stride, so they need one width")
    if q.shape[2] != k.shape[2]:
        raise ValueError(f"the tilt needs one K/V head per query head, got {q.shape[2]}/{k.shape[2]}")
    blocks = mean_k.shape[1]
    fmax = torch.finfo(k.dtype).max
    dev = k.device
    q_head_max = (q_abs_max if q_abs_max is not None else sol_q_abs_max(q))[1]
    planes_a, planes_b, peaks = _tilt_factors(
        k, v, q_head_max, rank, kv_tile, blocks, power_iters, block_gain)
    for name, x in (("q", q_descale), ("k", k_descale), ("v", v_descale)):
        if x.numel() != 1:
            raise ValueError(f"the tilt takes per-tensor descales, got {x.numel()} for {name}")
    # In real units a / qd keeps |q . a| <= 1, and b carries the covariance's scale and sm_scale.
    shift = torch.empty(1, dtype=torch.int32, device=dev)
    units = torch.empty(2, dtype=torch.float32, device=dev)
    _sol_tilt_shift_kernel[(1,)](
        q_descale, k_descale, v_descale, peaks, peaks.numel() // 2, shift, units,
        _TILT_LOGIT_UNIT * softmax_scale * fmax,
        softmax_scale / kv_tile,
        _TILT_W_MAX,
        fmax,
        _TILT_B_HEADROOM,
        BLOCK=1024,
    )

    ext_k = torch.empty((1 + rank,) + tuple(mean_k.shape), dtype=k.dtype, device=dev)
    ext_v = torch.empty((1 + rank,) + tuple(mean_v.shape), dtype=v.dtype, device=dev)
    _pack_planes(planes_a, planes_b, mean_k, mean_v, ext_k, ext_v, units, k.shape[1] // kv_tile)
    plane_stride = ext_k.stride(0) * ext_k.element_size()
    return ext_k[0], ext_v[0], (rank, plane_stride, shift)


def _tilt_factors(
    k: torch.Tensor,
    v: torch.Tensor,
    q_head_max: torch.Tensor,
    rank: int,
    kv_tile: int,
    blocks: int,
    power_iters: int,
    block_gain: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """fp32 (rank, batch, blocks, heads, d) a and b planes, zero past the full blocks, and their peaks.

    k and v are each float8_e4m3fn, int8 or bfloat16; unless both are fp8 the kernel takes them
    to bf16, which holds all three exactly.
    """
    batch, seqlen_k, heads, d = k.shape
    for name, x in (("k", k), ("v", v)):
        _check_unit_stride(name, x)
        if x.dtype not in (torch.float8_e4m3fn, torch.int8, torch.bfloat16):
            raise ValueError(f"the tilt's factors take fp8, int8 or bf16 {name}, got {x.dtype}")
    if not 1 <= rank <= d:
        raise ValueError(f"tilt rank must be in [1, {d}], got {rank}")
    full = seqlen_k // kv_tile
    dev = k.device
    planes_a = torch.empty(rank, batch, blocks, heads, d, dtype=torch.float32, device=dev)
    planes_b = torch.empty_like(planes_a)
    if blocks > full:
        planes_a[:, :, full:].zero_()
        planes_b[:, :, full:].zero_()
    peaks = (torch.empty if full else torch.zeros)(
        batch, heads, max(full, 1), 2, dtype=torch.float32, device=dev)
    if full:
        gain = block_gain if block_gain is not None else planes_a
        gain_strides = block_gain.stride() if block_gain is not None else (0, 0, 0)
        _sol_tilt_factor_kernel[(full, heads, batch)](
            k, v, q_head_max, gain, planes_a, planes_b, peaks,
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            q_head_max.stride(0), q_head_max.stride(1),
            *gain_strides,
            *planes_a.stride()[:4],
            HAS_GAIN=block_gain is not None,
            TO_BF16=not (k.dtype == v.dtype == torch.float8_e4m3fn),
            RANK=rank,
            WIDTH=max(16, triton.next_power_of_2(rank)),
            ITERS=power_iters,
            BLOCK_N=kv_tile,
            HEAD_DIM=d,
            num_warps=4,
            waves_per_eu=2,
        )
    return planes_a, planes_b, peaks


def _pack_planes(planes_a, planes_b, mean_k, mean_v, ext_k, ext_v, units, full) -> None:
    """Fill (1 + rank)-plane allocations of an addressable K and V from the factor planes."""
    rank, batch, blocks, heads, d = planes_a.shape
    if blocks:
        _sol_tilt_pack_kernel[(blocks, heads, batch)](
            planes_a, planes_b, mean_k, mean_v, ext_k, ext_v,
            units, rank, full,
            *planes_a.stride()[:4],
            mean_k.stride(0), mean_k.stride(1), mean_k.stride(2),
            mean_v.stride(0), mean_v.stride(1), mean_v.stride(2),
            *ext_k.stride()[:4],
            *ext_v.stride()[:4],
            ROUND_K=not ext_k.dtype.is_floating_point,
            HEAD_DIM=d,
        )


# A planes are encoded as mean_k is, B planes as mean_v is.
SOL_TILT_K_FORMATS = ("e4m3", "int8", "bf16", "mxfp8", "mxfp6", "mxfp4")
SOL_TILT_V_FORMATS = ("e4m3", "bf16", "mxfp6_fp6_p", "mxfp4_fp6_p")
_PLAIN_DTYPES = {"e4m3": torch.float8_e4m3fn, "int8": torch.int8, "bf16": torch.bfloat16}


class SolTiltPlanes(NamedTuple):
    """A tilt's planes in K's and V's own formats, as sol_tilt_planes returns them.

    mean_k / mean_v (and their scales, for the block-scaled formats) are plane 0 of (1 + rank)-plane
    allocations, the planes `*_plane_stride` bytes apart. Decoded the way mean_k and mean_v are,
    the score is units[2] * (q . A) for Q as the kernel decodes it, |score| <= 1, and the block's
    value gains P * score * B * units[3]. units[0] and units[1] are the code multipliers the A and
    B planes were encoded with.
    """

    mean_k: torch.Tensor
    mean_v: torch.Tensor
    mean_k_scale: torch.Tensor | None
    mean_v_scale: torch.Tensor | None
    rank: int
    k_plane_stride: int
    v_plane_stride: int
    k_scale_plane_stride: int
    v_scale_plane_stride: int
    units: torch.Tensor


def _as_descale(x: torch.Tensor | float, name: str, device: torch.device) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        if x.numel() != 1:
            raise ValueError(f"the tilt takes per-tensor descales, got {x.numel()} for {name}")
        return x.reshape(1)
    return torch.full((1,), float(x), dtype=torch.float32, device=device)


def _stack_planes(
    plane0: torch.Tensor, planes: torch.Tensor, row_tile: int = 1
) -> tuple[torch.Tensor, int]:
    """plane0 and planes as one contiguous (1 + rank)-plane allocation: (plane 0 view, stride in bytes).

    With row_tile > 1 each plane is followed by zeroed rows (dim 1, BSHD) taking the last batch's
    up to a multiple of it; plane 0's strides are contiguous either way.

    A kernel writes every plane: the Sol kernel reaches planes 1..rank only through the byte
    stride, so under torch.compile torch ops building them would be cut down to plane 0.
    """
    plane0 = plane0.contiguous()
    size = plane0.numel()
    pad = (-plane0.shape[1] % row_tile) * plane0[0, 0].numel()
    shape = (1 + planes.shape[0], size + pad)
    ext = plane0.new_zeros(shape) if pad else plane0.new_empty(shape)
    block = 1024
    _sol_tilt_stack_kernel[(triton.cdiv(size, block), shape[0])](
        plane0, planes.to(plane0.dtype).contiguous(), ext, size, shape[1], BLOCK=block,
    )
    return ext.as_strided(plane0.shape, plane0.stride()), ext.stride(0) * ext.element_size()


def _plane0_of_packed(
    view: torch.Tensor, scale: torch.Tensor, batch: int
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    """Plane 0 of a packer's output over (1 + rank) * batch batches, with its plane strides in bytes."""
    return (
        view[:batch],
        scale[:batch],
        batch * view.stride(0) * view.element_size(),
        batch * scale.stride(0) * scale.element_size(),
    )


def sol_tilt_planes(
    k: torch.Tensor,
    v: torch.Tensor,
    q_abs_max: tuple[torch.Tensor, torch.Tensor],
    mean_k: torch.Tensor,
    mean_v: torch.Tensor,
    softmax_scale: float,
    rank: int,
    kv_tile: int,
    *,
    k_format: str,
    v_format: str,
    q_descale: torch.Tensor | float = 1.0,
    k_descale: torch.Tensor | float = 1.0,
    v_descale: torch.Tensor | float = 1.0,
    q_multiplier: float = 1.0,
    mean_k_scale: torch.Tensor | None = None,
    mean_v_scale: torch.Tensor | None = None,
    mean_k_pooled: torch.Tensor | None = None,
    mean_v_pooled: torch.Tensor | None = None,
    power_iters: int = 1,
    block_gain: torch.Tensor | None = None,
) -> SolTiltPlanes:
    """A rank-`rank` tilt's planes encoded in K's and V's own formats, for any recipe.

    k and v are K and V in the basis and units the factors are taken in, BSHD: the stored codes
    for a per-tensor or BF16 operand, the dequantized codes for MXFP8 K (rotated, as stored), and
    the BF16 source for a packed one, which its packer rotates. q_abs_max is sol_q_abs_max of the
    Q in that same basis, q_x; q_descale, k_descale and v_descale take q_x, k and v to real units,
    and Q as the kernel decodes it is q_multiplier times the real Q (mha_v4_q_multiplier for the MX
    rows, which fold it into Q, else 1). The mean_* arguments are sol_prepare's:
    the pooled K/V and their scales, and for a packed format the pooled BF16 rows it packed,
    which become plane 0 through the same packer.

    k_format is one of SOL_TILT_K_FORMATS and v_format one of SOL_TILT_V_FORMATS. A per-tensor
    code plane fills its range under one multiplier; a BF16 plane holds the values, and a
    block-scaled plane takes E8M0 scales of its own, so units[0] = 1 and units[1] = b_scale there.
    block_gain is as for sol_tilt_factors. Nothing is read back to the host.
    """
    if k_format not in SOL_TILT_K_FORMATS:
        raise ValueError(f"k_format must be one of {SOL_TILT_K_FORMATS}, got {k_format!r}")
    if v_format not in SOL_TILT_V_FORMATS:
        raise ValueError(f"v_format must be one of {SOL_TILT_V_FORMATS}, got {v_format!r}")
    batch, _, heads, d = k.shape
    if v.shape[2] != heads or q_abs_max[1].shape[1] != heads:
        raise ValueError("the tilt needs one K/V head per query head")
    k_mx = k_format in ("mxfp8", "mxfp6", "mxfp4")
    v_mx = v_format.endswith("_fp6_p")
    if k_mx != (mean_k_scale is not None) or v_mx != (mean_v_scale is not None):
        raise ValueError("a block-scaled format needs its pooled scale, and only it takes one")
    if (k_format in ("mxfp6", "mxfp4") and mean_k_pooled is None) or (v_mx and mean_v_pooled is None):
        raise ValueError("a packed format needs the pooled BF16 rows sol_prepare packed")
    dev = k.device
    blocks = mean_k.shape[1]
    full = k.shape[1] // kv_tile
    planes_a, planes_b, peaks = _tilt_factors(
        k, v, q_abs_max[1], rank, kv_tile, blocks, power_iters, block_gain)

    units = torch.empty(4, dtype=torch.float32, device=dev)
    k_codes = k_format in ("e4m3", "int8")
    _sol_tilt_unit_kernel[(1,)](
        _as_descale(q_descale, "q", dev),
        _as_descale(k_descale if not k_mx else 1.0, "k", dev),
        _as_descale(v_descale if not v_mx else 1.0, "v", dev),
        peaks, peaks.numel() // 2, units,
        q_multiplier,
        softmax_scale / kv_tile,
        127.0 if k_format == "int8" else torch.finfo(torch.float8_e4m3fn).max,
        torch.finfo(torch.float8_e4m3fn).max * _TILT_B_HEADROOM,
        A_CODES=k_codes,
        B_CODES=v_format == "e4m3",
        BLOCK=1024,
    )

    k_scale = v_scale = None
    k_scale_stride = v_scale_stride = 0
    if not k_mx and not v_mx:
        ext_k = torch.empty((1 + rank,) + tuple(mean_k.shape), dtype=mean_k.dtype, device=dev)
        ext_v = torch.empty((1 + rank,) + tuple(mean_v.shape), dtype=mean_v.dtype, device=dev)
        _pack_planes(planes_a, planes_b, mean_k, mean_v, ext_k, ext_v, units, full)
        return SolTiltPlanes(
            ext_k[0], ext_v[0], None, None, rank,
            ext_k.stride(0) * ext_k.element_size(), ext_v.stride(0) * ext_v.element_size(),
            0, 0, units)

    from aiter.ops.triton.attention.utils import (
        _e8m0_quantize,
        _sol_attn_pack_k,
        _sol_attn_pad_to_tile,
    )

    if k_format == "mxfp8":
        data, scale = _e8m0_quantize(planes_a, mean_k.dtype)
        ext_k, k_stride = _stack_planes(mean_k, data)
        k_scale, k_scale_stride = _stack_planes(mean_k_scale, scale, _TILT_SCALE_ROW_TILE)
    elif k_mx:
        x = torch.cat([mean_k_pooled.unsqueeze(0), planes_a.to(torch.bfloat16)]).flatten(0, 1)
        ext_k, k_scale, k_stride, k_scale_stride = _plane0_of_packed(
            *_sol_attn_pack_k(x, k_format, kv_tile), batch)
    else:
        a = planes_a * units[0]
        if k_format == "int8":
            a = a.round()
        ext_k, k_stride = _stack_planes(mean_k, a)
    b = planes_b * units[1]
    if v_mx:
        from aiter.ops.mha_v4_quant import (
            mxfp4_v_view,
            quantize_v_mxfp4_fp6_p,
            quantize_v_mxfp6_fp6_p,
        )

        x = torch.cat([mean_v_pooled.unsqueeze(0), b.to(torch.bfloat16)]).flatten(0, 1)
        padded = _sol_attn_pad_to_tile(x, kv_tile)
        if v_format == "mxfp4_fp6_p":
            raw, scale = quantize_v_mxfp4_fp6_p(padded)
            view = mxfp4_v_view(raw, scale, blocks)
        else:
            view, scale = quantize_v_mxfp6_fp6_p(padded)
            view = view[:, :blocks]
        ext_v, v_scale, v_stride, v_scale_stride = _plane0_of_packed(view, scale, batch)
    else:
        ext_v, v_stride = _stack_planes(mean_v, b)
    return SolTiltPlanes(
        ext_k, ext_v, k_scale, v_scale, rank, k_stride, v_stride,
        k_scale_stride, v_scale_stride, units)


def sol_tilt_valu_params(
    planes: SolTiltPlanes,
    q_descale: torch.Tensor | float = 1.0,
    k_descale: torch.Tensor | float = 1.0,
    strides: tuple[int, ...] = (),
) -> torch.Tensor:
    """The device parameter block a code-score kernel reads its tilt from.

    For the rows whose score GEMM multiplies per-tensor Q and A codes: i8fp8, and F8F6, whose FP6-P
    V planes are block-scaled; and the packed MX rows (F6F6, F6F4, MXFP4), whose score MFMA decodes
    Q and A itself, at unit descales. planes is sol_tilt_planes for the same descales. Dword 0 holds
    the value GEMM's E8M0 shift as a signed byte in bits 8..15, dword 1 the f32 multiplier taking the
    q . A product to the weight score, at most _TILT_W_MAX in magnitude, and dword 2 on the row's
    other plane strides in bytes, at most three (the V and V-scale planes, then MXFP4's K-scale
    planes). Pass it as sol_tilt's shift tensor.
    """
    if len(strides) > 3:
        raise ValueError(f"the tilt's parameter block carries at most three strides, got {len(strides)}")
    dev = planes.units.device
    out = torch.empty(2 + len(strides), dtype=torch.int32, device=dev)
    padded = tuple(int(s) for s in strides) + (0,) * (3 - len(strides))
    _sol_tilt_valu_kernel[(1,)](
        planes.units, _as_descale(q_descale, "q", dev), _as_descale(k_descale, "k", dev), out,
        _TILT_W_MAX, *padded, STRIDES=len(strides),
    )
    return out


def sol_tilt_scaled_params(planes: SolTiltPlanes, scale_stride: int) -> torch.Tensor:
    """The three-dword device parameter block a decoded-score kernel reads its tilt from.

    For the rows whose score GEMM needs no q or k descale: MXFP8, whose MFMA decodes Q and the
    block-scaled A plane itself, and BF16/FP8 and BF16, whose A plane is BF16 (sigma is then in real
    Q units; BF16/FP8 divides off the slope it folds into resident Q). planes is sol_tilt_planes for
    that row, so sigma is u2 * m and m is the part of the value gain that fits the weight's e4m3
    range. Dword 0 holds the value GEMM's E8M0 shift as a signed byte in bits 8..15 (BF16 weights
    take it on sigma instead), dword 1 the f32 score multiplier, dword 2 `scale_stride`, the row's
    second plane stride in bytes: MXFP8's K-scale planes, BF16/FP8's and BF16's V planes (BF16/FP8's
    K planes are twice as wide). Pass it as sol_tilt's shift tensor.
    """
    out = torch.empty(3, dtype=torch.int32, device=planes.units.device)
    _sol_tilt_scaled_kernel[(1,)](planes.units, int(scale_stride), out, _TILT_W_MAX)
    return out
