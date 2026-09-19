# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Host-side operand prep for MXFP6-E2M3 A operands (a6w4).

``flydsl_batched_gemm_mxfp4`` accepts ``a_dtype="fp6"`` but had no way to build
its operands. aiter does quantize to MXFP6 elsewhere -- ``gemm_op_a6w6`` has HIP,
Triton and torch producers -- but those emit the a6w6 ``mxfp6_c0c1_256_padk2``
layout and apply a Hadamard rotation, whereas this kernel reads FP8-padded packed
FP6 with a CDNA4 scale shuffle and no rotation. Hence a separate set:

  * :func:`per_1x32_f6_quant` -- BF16/FP32 ``[M,K]`` -> packed MXFP6-E2M3 codes
    + per-1x32 E8M0 scales, in the FP8-padded layout the kernel reads.
  * :func:`shuffle_scale_w4` -- the CDNA4 scale preshuffle. Distinct from
    :func:`aiter.ops.shuffle.shuffle_scale_n32k4`, which is the gfx1250 WMMA
    layout.
  * :func:`shuffle_weight_w4` -- the MXFP4 B preshuffle, a thin wrapper over
    :func:`aiter.ops.shuffle.shuffle_weight_NK` (verified bit-identical to
    FlyDSL's helper of the same name), kept so callers have one import surface.

E8M0 decode and the sub-byte float encoder are reused from
:mod:`aiter.utility.fp4_utils` rather than duplicated.

:func:`f32_to_e8m0` is carried over verbatim from the FlyDSL reference so the
scales are bit-identical to what the GEMM was validated against. It rounds the
exponent half-up, which is none of the
:class:`~aiter.utility.mx_types.MxScaleRoundMode` formulas -- in particular not
``RoundUp``/RCEIL, which divides by ``max_pos`` (7.5) rather than
``2**target_max_pow2`` (4.0). Changing it would change GEMM numerics.
"""

import torch
from torch import Tensor

from aiter import dtypes
from aiter.utility.fp4_utils import _f32_to_floatx_unpacked, e8m0_to_f32

__all__ = [
    "f32_to_e8m0",
    "fp6_e2m3_to_f32",
    "pack_fp6_e2m3",
    "per_1x32_f6_quant",
    "shuffle_scale_w4",
    "shuffle_weight_w4",
]

# max_normal(E2M3) is 7.5; the quantizer divides by the largest power of two not
# exceeding it, 2**floor(log2(7.5)) == 4.0.
F6E2M3_TARGET_MAX_POW2 = 2


def f32_to_e8m0(x: Tensor) -> Tensor:
    """Encode a positive f32 tensor as an E8M0 biased exponent byte (half-up)."""
    u32 = x.view(torch.int32)
    exponent = ((u32 >> 23) & 0xFF).view(torch.uint32).to(torch.uint8)
    nan_case = exponent == 0xFF
    round_case = ((u32 & 0x400000) > 0) & (
        ((u32 & 0x200000) > 0) | ((u32 & 0x1FFFFF) > 0) | (exponent > 0)
    )
    exponent[round_case] += 1
    exponent[nan_case] = 0xFF
    return exponent.view(dtypes.fp8_e8m0)


def pack_fp6_e2m3(x_unpacked: Tensor) -> Tensor:
    """Pack uint8 (low 6 bits = E2M3) 4-at-a-time into 3 dense bytes.

    Input ``(..., 4*G)`` uint8 -> output ``(..., 3*G)`` uint8, little-endian
    groups: ``b0 = e1[1:0]<<6 | e0``, ``b1 = e2[3:0]<<4 | e1>>2``,
    ``b2 = e3<<2 | e2>>4``.
    """
    assert x_unpacked.dtype == torch.uint8 and x_unpacked.shape[-1] % 4 == 0
    g = x_unpacked.unflatten(-1, (-1, 4)).to(torch.int32) & 0x3F
    e0, e1, e2, e3 = g.unbind(dim=-1)
    b0 = ((e1 & 0x03) << 6) | e0
    b1 = ((e2 & 0x0F) << 4) | (e1 >> 2)
    b2 = (e3 << 2) | (e2 >> 4)
    out = torch.stack([b0, b1, b2], dim=-1).to(torch.uint8)
    return out.reshape(
        *x_unpacked.shape[:-1], x_unpacked.shape[-1] // 4 * 3
    ).contiguous()


_FP6_E2M3_LUT: dict = {}


def fp6_e2m3_to_f32(x_unpacked: Tensor) -> Tensor:
    """Decode uint8 (low 6 bits = E2M3, 1 sign / 2 exp / 3 mant, bias 1) to f32."""
    dev = x_unpacked.device
    lut = _FP6_E2M3_LUT.get(dev)
    if lut is None:
        vals = torch.empty(64, dtype=torch.float32)
        for c in range(64):
            sign = -1.0 if (c & 0x20) else 1.0
            exp = (c >> 3) & 0x3
            mant = c & 0x7
            mag = (mant / 8.0) if exp == 0 else (2.0 ** (exp - 1)) * (1.0 + mant / 8.0)
            vals[c] = sign * mag
        lut = vals.to(dev)
        _FP6_E2M3_LUT[dev] = lut
    return lut[(x_unpacked & 0x3F).long()]


def per_1x32_f6_quant(x: Tensor):
    """Per-1x32 MXFP6-E2M3 quant of an A operand, FP8-padded packed FP6.

    **Reference implementation.** This is eager torch: several elementwise and
    reduction launches plus full-size fp32 and unpacked temporaries, and it also
    returns ``a_unpacked`` purely for dequant references. That is fine for tests
    and for one-off operand preparation, but activation quantization runs every
    forward pass and cannot be amortized like weight preparation, so a serving
    path wants a fused quantize+pack kernel instead -- compare
    ``gemm_op_a6w6``'s HIP/Triton producers, which exist for exactly this reason.

    The kernel indexes the A scale in whole 32-row supers (it handles a ragged M
    with ``ceil(M/32)``), and :func:`shuffle_scale_w4` reshapes by 32 rows, so the
    scale is returned at ``M`` rounded up to a multiple of 32 while the codes are
    trimmed back to the real ``M``. Returning an M-row scale instead would make
    the documented ``per_1x32_f6_quant`` -> ``shuffle_scale_w4`` sequence throw for
    every M that is not a multiple of 32 -- i.e. every decode-shaped call.

    Args:
        x: ``(M, K)`` activation. K must be a multiple of 32.
    Returns:
        a_pad: ``(M, K)`` uint8 -- FP8-padded packed FP6 (24 B of codes + 8 B of
            zero per K=32 chunk), the layout ``launch_gemm`` reads for
            ``a_dtype="fp6"``.
        scale: ``(ceil(M/32)*32, K//32)`` E8M0, unshuffled; callers apply
            :func:`shuffle_scale_w4`. Rows beyond ``M`` are zero.
        a_unpacked: ``(M, K)`` uint8 -- low-6-bit E2M3 codes, for the dequant
            reference.
    """
    block = 32
    if x.ndim != 2:
        raise ValueError(f"per_1x32_f6_quant expects [M, K], got {x.ndim}D")
    if x.shape[-1] % block:
        raise ValueError(
            f"per_1x32_f6_quant needs K a multiple of {block}, got {x.shape[-1]}"
        )

    M = x.shape[0]
    m_pad = max(block, -(-M // block) * block)
    if m_pad != M:
        x = torch.nn.functional.pad(x, (0, 0, 0, m_pad - M))

    dtype_max = float(1 << F6E2M3_TARGET_MAX_POW2)  # 4.0
    shape_original = x.shape
    xb = x.view(-1, shape_original[-1]).reshape(-1, block)
    max_abs = torch.amax(torch.abs(xb.float()), 1)
    scale_e8m0 = f32_to_e8m0(max_abs / dtype_max)
    scale_f32 = e8m0_to_f32(scale_e8m0)
    y = xb.float() / scale_f32.view(-1, 1)
    codes = _f32_to_floatx_unpacked(y, 2, 3).to(torch.uint8)  # (.., 32) low 6 bits
    a_unpacked = codes.view(*shape_original).contiguous()
    K = a_unpacked.shape[-1]
    packed = pack_fp6_e2m3(a_unpacked).view(m_pad, K // 32, 24)
    a_pad = torch.zeros(m_pad, K // 32, 32, dtype=torch.uint8, device=x.device)
    a_pad[:, :, :24] = packed
    a_pad = a_pad.view(m_pad, K)
    scale = scale_e8m0.view(m_pad, -1).view(torch.uint8)
    return a_pad[:M].contiguous(), scale, a_unpacked[:M].contiguous()


def shuffle_weight_w4(
    src: torch.Tensor, NLane: int = 16, gate_up: bool = False, moe_gemm: bool = False
) -> torch.Tensor:
    """CK preshuffle of a packed MXFP4 weight ``[N, K//2]`` for this GEMM.

    ``shuffle_weight_NK(src, 16, 64)`` already produces exactly this layout
    (verified bit-identical to FlyDSL's ``shuffle_weight_w4``); this wrapper
    keeps the FlyDSL-facing name so a consumer can take the whole operand-prep
    surface from one module.

    ``gate_up`` and ``moe_gemm`` exist only to reject the FlyDSL variants this
    dense path does not implement -- returning the dense permutation for a
    gate/up or MoE caller would silently produce wrong numbers. Likewise NLane
    is fixed at 16 because the kernel's B fragment layout is hard-coded to it.
    """
    from aiter.ops.shuffle import shuffle_weight_NK

    if NLane != 16:
        raise ValueError(
            f"[FlyDSL] the a6w4/a4w4 kernel reads a 16-lane B layout; "
            f"got NLane={NLane}"
        )
    if gate_up or moe_gemm:
        raise NotImplementedError(
            "[FlyDSL] shuffle_weight_w4 here covers the dense case only; "
            f"got gate_up={gate_up}, moe_gemm={moe_gemm}"
        )
    return shuffle_weight_NK(src, NLane, 4 * NLane)


def shuffle_scale_w4(
    src: torch.Tensor, experts_cnt: int = 1, gate_up: bool = False
) -> torch.Tensor:
    """CDNA4 per-1x32 E8M0 scale preshuffle for the wave64 MFMA preshuffle GEMM.

    Distinct from :func:`aiter.ops.shuffle.shuffle_scale_n32k4`, which is the
    gfx1250 WMMA layout.
    """
    n_experts, k_ = src.shape
    n_ = n_experts // experts_cnt
    K_Pack = 2
    N_Pack = 2
    N_Lane = 16
    K_Lane = 64 // N_Lane

    K1 = k_ // K_Pack // K_Lane
    N1 = n_ // N_Lane // N_Pack
    real_k = 32 * k_ * K_Pack * K_Lane  # 1x32 quant
    assert real_k >= 256, f"K {real_k} must be larger than Tile_K(256)"

    if gate_up:
        shfl_scale = src.view(experts_cnt, N_Pack, N1, N_Lane, K1, K_Pack, K_Lane)
        shfl_scale = shfl_scale.permute(0, 2, 4, 6, 3, 5, 1).contiguous()
    else:
        shfl_scale = src.view(experts_cnt, N1, N_Pack, N_Lane, K1, K_Pack, K_Lane)
        shfl_scale = shfl_scale.permute(0, 1, 4, 6, 3, 5, 2).contiguous()
    return shfl_scale.view(*src.shape).contiguous()
