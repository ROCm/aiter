# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Regression tests for the gfx1201 LDS-budget clamp in select_3d_config (issue #4329).

The 3D split-KV kernel's shared-memory request on the gfx1201 Triton stack is
``2 (K, V) * TILE_SIZE * head_size_padded * kv_elem_bytes * num_stages + Q tile``.
At TILE_SIZE=64 / head_size=128 / bf16 KV the shipped config requests
65792-73728 B, over the 64 KB limit, so Triton aborts at kernel launch
(downstream: vllm-project/vllm#48723, `vllm serve` startup failure).
"""

import pytest
import torch

try:
    from aiter.ops.triton.attention.unified_attention import (
        DEVICE_ARCH,
        _unified_3d_lds_footprint,
        select_3d_config,
    )
    from aiter.ops.triton.utils.types import e4m3_dtype
except Exception:  # CPU-only host or triton unavailable
    pytest.skip("unified_attention module not importable on this host", allow_module_level=True)

BUDGET_64KB = 65536

# (tile_size, head_padded, kv_elem_bytes, q_elem_bytes, block_m, num_stages, expected bytes)
# The three footprints reported in issue #4329 / vllm#48723, measured via
# triton.compile(...).metadata.shared.
REPORTED_OVERFLOW_CONFIGS = [
    (64, 128, 2, 2, 1, 2, 65792),
    (64, 128, 2, 2, 16, 2, 69632),
    (64, 128, 2, 2, 32, 2, 73728),
]


def test_footprint_reproduces_reported_overflows():
    for tile, head, kvb, qb, bm, stages, expected in REPORTED_OVERFLOW_CONFIGS:
        got = _unified_3d_lds_footprint(tile, head, kvb, qb, bm, stages)
        assert got == expected, f"footprint model drift: {got} != {expected}"
        assert got > BUDGET_64KB


def _select_3d(head_size, block_size, q_dtype, kv_dtype, block_m):
    return select_3d_config(
        head_size,
        block_size,
        max_seqlen_k=16384,
        target_num_prgms=160,
        num_2d_prgms=40,
        q_dtype=q_dtype,
        kv_cache_dtype=kv_dtype,
        shuffled_kv_cache=False,
        NUM_BLOCKS_GATHER_PER_TILE=1,
        SLIDING_WINDOW=None,
        block_m=block_m,
    )


@pytest.mark.skipif(DEVICE_ARCH != "gfx1201", reason="clamp is scoped to gfx1201")
def test_select_3d_config_clamps_overflow_configs_on_gfx1201():
    # The vllm#48723 shape: head_size=128, bf16 KV, block_size 64 (TILE_SIZE 64),
    # long context so the 3D path is selected.
    for block_m in (1, 16, 32):
        attn_config, _ = _select_3d(128, 64, torch.bfloat16, torch.bfloat16, block_m)
        footprint = _unified_3d_lds_footprint(
            attn_config["TILE_SIZE"],
            128,
            2,
            2,
            block_m,
            attn_config["num_stages"],
        )
        assert footprint <= BUDGET_64KB, (
            f"3D config over the 64 KB LDS budget: {footprint} B "
            f"(TILE_SIZE={attn_config['TILE_SIZE']}, num_stages={attn_config['num_stages']})"
        )


@pytest.mark.skipif(DEVICE_ARCH != "gfx1201", reason="guard against over-clamping on gfx1201")
def test_select_3d_config_keeps_fitting_configs_untouched_on_gfx1201():
    # FP8 KV halves the K/V pipeline: the full double-buffered config fits, so the
    # clamp must not touch it (over-clamping would regress throughput).
    attn_config, _ = _select_3d(128, 64, torch.bfloat16, e4m3_dtype, 16)
    assert attn_config["num_stages"] == 2
    assert attn_config["TILE_SIZE"] == 64


@pytest.mark.skipif(DEVICE_ARCH != "gfx1151", reason="gfx1151 keeps its working config")
def test_select_3d_config_unchanged_on_gfx1151():
    # The gfx1151 stack resolves the same request to 33024 B (measured in #4329),
    # so the clamp is deliberately not applied there; this guards that scoping.
    attn_config, _ = _select_3d(128, 64, torch.bfloat16, torch.bfloat16, 16)
    assert attn_config["num_stages"] == 2
    assert attn_config["TILE_SIZE"] == 64
