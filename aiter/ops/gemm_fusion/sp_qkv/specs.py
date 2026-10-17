# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Spec builders and the validated cell table for MiniMax-M3 SP4 QKV (pure Python, no torch).

Field map authority: golden case_config.py field_spec (sp4-attention-fusion-cases-20261009):
  q (0,8192,4) k (8192,512,4) v (8704,512,4) [iq (9216,512,4) ik (9728,128,1)]  -> dense N=9216 F=2304; indexer N=9856 F=2560.
Reuse uses the indexer GEMM fields but drops iq/ik in the exchange (compact F=2304, approved).
"""

from __future__ import annotations

from aiter.gemm_fusion.spec import FusionSpec, GemmSpec, HeadNormRopeCache, SPExchange, qkv_fields

MODES = ("dense", "indexer", "reuse")
WORLD = 4
HIDDEN_K = 6144
KV_PAGE_SHUFFLE = "shuffle_page16_fp8_ptok"


def sp_qkv_spec(mode: str, world: int = WORLD, eps: float = 1e-6) -> FusionSpec:
    if mode not in MODES:
        raise ValueError(mode)
    if mode == "dense":
        fields = qkv_fields(q=8192, kv=512, world=world, kv_shards=world)
        n, drop, idx, num = 9216, (), None, "triton_norm_rope_cache_v1"
    else:
        fields = qkv_fields(q=8192, kv=512, world=world, iq=512, ik=128, kv_shards=world, ik_shards=1)
        n = 9856
        if mode == "indexer":
            drop, idx, num = (), "fp8_unit_page128", "aiter_fused_qknorm_idxrqknorm_v1"
        else:
            drop, idx, num = ("iq", "ik"), None, "triton_norm_rope_cache_v1"
    return FusionSpec(
        GemmSpec(n=n, k=HIDDEN_K),
        (
            SPExchange(world=world, fields=fields, drop=drop),
            HeadNormRopeCache(head_dim=128, q_heads=16, kv_heads=1, rotary_dim=64, eps=eps, numerics=num,
                              kv_cache=KV_PAGE_SHUFFLE, index=idx),
        ),
    )


def mode_of(spec: FusionSpec) -> str:
    """Recover the mode from a spec (used by the variant to select the compiled geometry)."""
    ex = spec.stages[0]
    names = {f.name for f in ex.fields}
    if "iq" not in names:
        return "dense"
    return "reuse" if "iq" in ex.drop else "indexer"


# (mode, tokens T) -> (vid, params). local M = T / 4. Ledger r165 (small) / r152 (large, filled in S2-2b-iii).
SMALL_CELLS = {
    ("dense", 64): ("sp_qkv.small.v1", {"src": "r17", "sk": 3, "nh": 1, "no_credit": 1}),
    ("dense", 128): ("sp_qkv.small.v1", {"src": "r18", "sk": 3, "nh": 1, "no_credit": 1, "b_rowsplit": 1, "rx_bypass": 1}),
    ("indexer", 64): ("sp_qkv.small.v1", {"src": "r17", "sk": 3, "nh": 1, "no_credit": 1, "push8": 1}),
    ("indexer", 128): ("sp_qkv.small.v1", {"src": "r17", "sk": 3, "nh": 1, "no_credit": 1}),
    ("reuse", 64): ("sp_qkv.small.v1", {"src": "r18", "sk": 3, "nh": 1, "no_credit": 1, "reuse_no_idx": 1}),
    ("reuse", 128): ("sp_qkv.small.v1", {"src": "r17", "sk": 3, "nh": 1, "no_credit": 1, "push_halves": 1}),
}
