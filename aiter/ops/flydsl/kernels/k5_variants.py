# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""GDN K5 kernel-variant tag grammar (arch-agnostic).

The tag universe and the tiny parse/legality helpers shared by the K5 host wrapper
(``linear_attention_prefill_kernels``) and the gfx942 kernel builder
(``kernels.chunk_gated_delta_h_gfx942``). It lives in its own module -- with no
dependency on either of those -- so the gfx942 tuned-variant table can validate tags
against the grammar without importing the host wrapper (which already imports the
gfx942 builder, so the reverse edge would be a cycle).
"""

from __future__ import annotations

# BV (the V-tile width) is the only compile-time tuning axis for K5: BT=64 is
# fixed by the K1-K3 pipeline that produces w/u, and everything else is derived.
# So a variant tag is just the tile size. ``auto`` is not a registered tag --
# it is the sentinel meaning "defer to the shape-adaptive heuristic".
#
# A second axis was added on gfx942: ``w<N>`` = the number of waves in the
# workgroup. The default (4) is the historical kernel. Wider workgroups split
# the N_REPEAT (V) axis across waves, which multiplies resident waves per CU --
# LDS pins gfx942 to one workgroup per CU, so a 4-wave block is 1 wave/SIMD and
# cannot hide HBM latency.
_BV_CANDIDATES = [16, 32, 64]
_DEFAULT_BV = 16
_WAVE_CANDIDATES = (4, 8, 16)

K5_VARIANTS: tuple[str, ...] = tuple(f"bv{b}" for b in _BV_CANDIDATES) + tuple(
    f"bv{b}w{w}"
    for b in _BV_CANDIDATES
    for w in _WAVE_CANDIDATES
    # NR_SPLIT = w/4 must divide N_REPEAT = b/16
    if w > 4 and (b // 16) % (w // 4) == 0
)
K5_DEFAULT_VARIANT = "auto"


def _legal_bv_candidates(V: int) -> list[int]:
    return [c for c in _BV_CANDIDATES if c <= V and V % c == 0]


def _bv_of_variant(tag: str) -> int:
    """``"bv64"`` -> ``64``, ``"bv64w16"`` -> ``64``."""
    return _bv_waves_of_variant(tag)[0]


def _bv_waves_of_variant(tag: str) -> tuple[int, int]:
    """``"bv64w16"`` -> ``(64, 16)``; ``"bv32"`` -> ``(32, 4)``.

    Raises ValueError on an unknown tag.
    """
    if tag not in K5_VARIANTS:
        raise ValueError(
            f"unknown GDN K5 variant {tag!r}; available: {list(K5_VARIANTS)} "
            f"(or {K5_DEFAULT_VARIANT!r} for shape-adaptive selection)"
        )
    body = tag[2:]
    if "w" in body:
        bv_s, w_s = body.split("w")
        return int(bv_s), int(w_s)
    return int(body), 4


def _variant_tag(bv: int, num_waves: int) -> str:
    """``(64, 8)`` -> ``"bv64w8"``; ``(32, 4)`` -> ``"bv32"``.

    Inverse of ``_bv_waves_of_variant``
    """
    return f"bv{bv}" if num_waves == 4 else f"bv{bv}w{num_waves}"
