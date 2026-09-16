# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Gluon kernels for the fused Qwen3-Next GDN *prefill* step (gfx950/CDNA4).

A fused Gluon op replaces the prefill chain ``causal_conv1d_split_qkv ->
fused_gdn_gating -> chunk_gated_delta_rule -> gated_rmsnorm_fp8_group_quant`` with
six tight launches that fuse the work intra-kernel and drop the intermediate HBM
round-trips; the epilogue launch emits the per-head group-128 FP8 activations a
block-FP8 ``out_proj`` consumes directly (no separate quant kernel).

Prefill covers a wide token range, so unlike the single decode kernel this ships
four M-tile specializations (each is the winning autotuned schedule for its
token/sequence band). The public wrapper in
``aiter/ops/triton/gated_delta_net/fused_gdn_prefill_qkvz.py`` selects one via a
runtime (tokens, batch) dispatch; the mapping mirrors the Artemis MI355 v1
kernel-pack profile it was ported from:

    tokens 1024..3071 , batch 1..64  -> _prefill_m1024_3071
    tokens 3072..12288, batch 1..64  -> _prefill_m3072_16384
    tokens 12289..16384, batch 1..5  -> _prefill_m12289_16384_b1_5
    tokens 12289..16384, batch 6..15 -> _prefill_m12289_16384_b6_15
    tokens 12289..16384, batch 16..64-> _prefill_m3072_16384

Each tile module is self-contained (its own ``@gluon.jit`` helpers and
``FP8_E4M3_FN`` precision config) to keep the ported schedules byte-faithful; the
duplication is intentional and load-bearing for numerics.
"""

from ._tile_m1024_3071 import gdn_prefill_group_fp8_quant as _prefill_m1024_3071
from ._tile_m3072_16384 import gdn_prefill_group_fp8_quant as _prefill_m3072_16384
from ._tile_m12289_16384_b1_5 import (
    gdn_prefill_group_fp8_quant as _prefill_m12289_16384_b1_5,
)
from ._tile_m12289_16384_b6_15 import (
    gdn_prefill_group_fp8_quant as _prefill_m12289_16384_b6_15,
)

__all__ = [
    "_prefill_m1024_3071",
    "_prefill_m3072_16384",
    "_prefill_m12289_16384_b1_5",
    "_prefill_m12289_16384_b6_15",
]
