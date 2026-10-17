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
# Golden case_config.cache_page_size = 16 if has_indexer else 128: dense -> page128; indexer AND reuse (N=9856) -> page16.
KV_PAGE = {"dense": 128, "indexer": 16, "reuse": 16}


def kv_cache_str(mode: str) -> str:
    return f"shuffle_page{KV_PAGE[mode]}_fp8_ptok"


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
                              kv_cache=kv_cache_str(mode), index=idx),
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


# Large cells: (mode, tokens T) -> (vid, params). local M = T/4 in {2048, 4096, 8192}. Ledger r152 + r154 (gcnasm 822c1d1, r11 recipe).
# DEVIATION (recorded): all nine cells use the single LG-QSDMA-r11 code base (NO_GRIDREAD=1 default) with the per-cell plan below,
# instead of the individual r6b/r6x4/r7/r8g candidates of ledger r152 (r11 plans reproduce them; gcnasm 822c1d1 README: all 2-12 us faster, same numerics).
#   plan: n = SDMA puts per peer; half_first 0 = r6x plan, 1 = half-N first group (r7), 2 = quarter-first n3q (r9, dense M=2048 only, n=3).
LARGE_CELLS = {
    ("dense", 8192): ("sp_qkv.large.v1", {"n": 3, "half_first": 2, "no_gridread": 1}),      # r11 n3q
    ("dense", 16384): ("sp_qkv.large.v1", {"n": 2, "half_first": 0, "no_gridread": 1}),     # r6b n2
    ("dense", 32768): ("sp_qkv.large.v1", {"n": 4, "half_first": 0, "no_gridread": 1}),     # r6b n4
    ("indexer", 8192): ("sp_qkv.large.v1", {"n": 2, "half_first": 1, "no_gridread": 1}),    # r7 n2h
    ("indexer", 16384): ("sp_qkv.large.v1", {"n": 4, "half_first": 0, "no_gridread": 1}),   # r6x4 n4
    ("indexer", 32768): ("sp_qkv.large.v1", {"n": 4, "half_first": 0, "no_gridread": 1}),   # r6x4 n4
    ("reuse", 8192): ("sp_qkv.large.v1", {"n": 2, "half_first": 1, "no_gridread": 1}),      # r8g F0 n2h (dense build, compact)
    ("reuse", 16384): ("sp_qkv.large.v1", {"n": 4, "half_first": 0, "no_gridread": 1}),     # r6x4 n4
    ("reuse", 32768): ("sp_qkv.large.v1", {"n": 4, "half_first": 0, "no_gridread": 1}),     # r6x4 n4
}