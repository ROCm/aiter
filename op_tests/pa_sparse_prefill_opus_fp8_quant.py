# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FP8 quant/dequant helpers for the OPUS sparse prefill v4 (asm-v4-style) path.

Mirrors the DeepSeek-V4 mixed-precision head-dim layout used by the asm v4 MLA
decode kernel (aiter PR #3112, op_tests/test_mla_v4_nm.py):

  head dim D = 512 = NOPE(448, FP8 e4m3) + ROPE(64, BF16)

The 448 NOPE elements are split into 7 tiles of 64; each tile carries one
**e8m0** (power-of-two) scale, computed as ``2^ceil(log2(amax/fp8_max))`` and
applied on dequant as ``bf16 = nope_fp8.to(f32) * scale``. The 64 ROPE elements
stay BF16 and are never quantized.

Design note vs asm v4: asm v4 embeds the 7 scales (duplicated x2 -> 14 bytes)
inside the 512-byte FP8 token record so its prebuilt .co can read them from
fixed offsets. Here we instead emit a **separate fp32 scale tensor** of shape
``[*, 7]`` (the values are still e8m0-rounded, i.e. exact powers of two). The
math is identical; a fresh HIP kernel is much cleaner consuming a separate
scale tensor than extracting bytes interleaved in the data buffer.
"""

from __future__ import annotations

import torch

# Layout constants (DeepSeek-V4 / asm-v4).
D_FULL = 512
D_NOPE = 448  # FP8-quantized
D_ROPE = 64  # BF16, never quantized
TILE = 64  # NOPE elements sharing one e8m0 scale
NUM_TILES = D_NOPE // TILE  # 7
assert D_NOPE + D_ROPE == D_FULL
assert D_NOPE % TILE == 0


def _fp8_dtype() -> torch.dtype:
    # gfx950 = OCP e4m3fn. (gfx942 would be e4m3fnuz; tests run on gfx950.)
    return torch.float8_e4m3fn


def cast_scale_to_ue8m0(scale: torch.Tensor) -> torch.Tensor:
    """Round a positive fp32 scale up to the nearest power of two (e8m0 grid).

    Matches test_mla_v4_nm.py::_cast_scale_inv_to_ue8m0 :
        2 ** ceil(log2(clamp_min(scale, 1e-4)))
    """
    return torch.pow(
        2.0, torch.clamp_min(scale, 1e-4).log2().ceil()
    ).to(torch.float32)


def quantize_to_v4_fp8(x_bf16: torch.Tensor):
    """BF16 ``[..., 512]`` -> (nope_fp8 ``[..., 448]``, rope_bf16 ``[..., 64]``,
    scale_fp32 ``[..., 7]``).

    Per 64-tile of NOPE: ``scale = ue8m0(amax / fp8_max)``,
    ``nope_fp8 = (tile / scale).to(fp8)``. ROPE is the trailing 64 dims, kept BF16.
    """
    assert x_bf16.shape[-1] == D_FULL, f"expected last dim {D_FULL}, got {x_bf16.shape[-1]}"
    fp8 = _fp8_dtype()
    fp8_max = torch.finfo(fp8).max

    nope = x_bf16[..., :D_NOPE]
    rope = x_bf16[..., D_NOPE:].to(torch.bfloat16).contiguous()

    leading = x_bf16.shape[:-1]
    nope_tiled = nope.reshape(*leading, NUM_TILES, TILE).float()
    amax = nope_tiled.abs().amax(dim=-1)  # [..., 7]
    scale = cast_scale_to_ue8m0(amax / fp8_max)  # [..., 7] power-of-two
    nope_q = (nope_tiled / scale.unsqueeze(-1)).to(fp8)  # [..., 7, 64]
    nope_q = nope_q.reshape(*leading, D_NOPE).contiguous()
    return nope_q, rope, scale.contiguous()


def dequantize_v4_fp8(
    nope_fp8: torch.Tensor,  # [..., 448] fp8
    rope_bf16: torch.Tensor,  # [..., 64] bf16
    scale_fp32: torch.Tensor,  # [..., 7] fp32 (power-of-two)
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Inverse of :func:`quantize_to_v4_fp8`. Returns ``[..., 512]`` (out_dtype)."""
    leading = nope_fp8.shape[:-1]
    nope = nope_fp8.reshape(*leading, NUM_TILES, TILE).float()
    nope = nope * scale_fp32.unsqueeze(-1)  # broadcast tile scale
    nope = nope.reshape(*leading, D_NOPE)
    full = torch.cat([nope, rope_bf16.float()], dim=-1)  # [..., 512]
    return full.to(out_dtype)


if __name__ == "__main__":
    # Roundtrip + dequant sanity (CPU; no GPU needed).
    torch.manual_seed(0)
    x = torch.randn(5, 3, D_FULL, dtype=torch.bfloat16)
    nope_q, rope, scale = quantize_to_v4_fp8(x)
    assert nope_q.shape == (5, 3, D_NOPE) and nope_q.dtype == _fp8_dtype()
    assert rope.shape == (5, 3, D_ROPE) and rope.dtype == torch.bfloat16
    assert scale.shape == (5, 3, NUM_TILES) and scale.dtype == torch.float32
    # scales are exact powers of two
    assert torch.all((scale.log2().round() - scale.log2()).abs() < 1e-5)
    deq = dequantize_v4_fp8(nope_q, rope, scale)
    # rope is lossless (bf16->bf16)
    assert torch.allclose(deq[..., D_NOPE:], x[..., D_NOPE:].to(torch.float32).to(torch.bfloat16))
    # nope within fp8 quant error of the per-tile range
    err = (deq[..., :D_NOPE].float() - x[..., :D_NOPE].float()).abs()
    rel = err / (x[..., :D_NOPE].float().abs() + 1e-3)
    print(f"nope dequant: max_abs_err={err.max():.4f} mean_rel={rel.mean():.4f}")
    print("roundtrip OK")
