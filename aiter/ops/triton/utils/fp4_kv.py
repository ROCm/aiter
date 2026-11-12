# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Geometry of the packed FP4 MLA KV cache (a ``torch.uint8`` buffer).

The dtype alone cannot tell the formats apart, so writers and readers take
the format by name:

- ``"nvfp4"``: E2M1 values, one E4M3 scale per 16 values.
- ``"mxfp4"``: E2M1 values, one E8M0 scale per 32 values (OCP MX).
- ``"mxfp4_b16"``: E2M1 values, one E8M0 scale per 16 values.
- ``"mxfp4_fp8rope"``: the latent as ``"mxfp4"``, rope as FP8 E4M3.

Per shuffled block, a row is ``[latent E2M1 | latent scales | rope E2M1 | rope
scales]``, each region ``block_size`` tokens deep; with FP8 rope it is
``[latent E2M1 | latent scales | rope E4M3]``, rope in the FP8 cache's
shuffled layout.
"""

import triton

FP4_KV_FORMATS = ("nvfp4", "mxfp4", "mxfp4_b16", "mxfp4_fp8rope")


def fp4_kv_scale_block(fp4_kv_format: str) -> int:
    """Number of values sharing one scale byte."""
    if fp4_kv_format in ("mxfp4", "mxfp4_fp8rope"):
        return 32
    if fp4_kv_format in ("nvfp4", "mxfp4_b16"):
        return 16
    raise ValueError(
        f"fp4_kv_format must be one of {FP4_KV_FORMATS}, got {fp4_kv_format!r}"
    )


def fp4_kv_rope_is_fp8(fp4_kv_format: str) -> bool:
    fp4_kv_scale_block(fp4_kv_format)
    return fp4_kv_format == "mxfp4_fp8rope"


def fp4_kv_scale_k_width(num_scales: int) -> int:
    """Scale bytes per token grouped together by the 32x4 scale shuffle."""
    if num_scales >= 4:
        return min(16, triton.next_power_of_2(num_scales))
    return num_scales


def fp4_kv_row_bytes(d_nope: int, d_rope: int, fp4_kv_format: str) -> int:
    """Bytes one token occupies in the packed cache."""
    block = fp4_kv_scale_block(fp4_kv_format)
    if fp4_kv_rope_is_fp8(fp4_kv_format):
        assert (
            d_nope % block == 0
        ), f"{fp4_kv_format} KV needs d_nope ({d_nope}) divisible by {block}"
        return d_nope // 2 + d_nope // block + d_rope
    assert d_nope % block == 0 and d_rope % block == 0, (
        f"{fp4_kv_format} KV needs d_nope ({d_nope}) and d_rope ({d_rope}) "
        f"divisible by {block}"
    )
    return (d_nope + d_rope) // 2 + d_nope // block + d_rope // block
