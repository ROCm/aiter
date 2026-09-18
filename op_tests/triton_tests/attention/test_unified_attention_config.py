"""Lookup regression tests for the unified_attention config tables.

The D axis outranks the Q axis in `_lookup`'s candidate ordering, so a
`D_GEQ_*` entry at BLOCK_M=16 shadows the standalone `Q_GEQ_256` entry
(BLOCK_M=128) for large-head large-prefill shapes. The composite
`D_GEQ_*.Q_GEQ_1024.DT_*` keys in the gfx942 table exist to bind D, Q and
the dtype axis at once; without them large prefill silently falls back to
the decode-era BLOCK_M=16.

The composites are dtype-split: fp8 and bf16 need different num_warps at
these shapes (a dtype-agnostic key picked the fp8-tuned warp count and
regressed bf16). The Q bound is 1024 because the measured crossover vs the
D-only entries sits between 512 and 1024 tokens.

These tests assert the *lookup decision* (matched key and BLOCK_M), not
kernel output, so a regression in key canonicalization or candidate
ordering is caught without launching a kernel. The lookup is pinned to
the gfx942 table so the assertions hold on any machine the suite runs
on (the composite keys are gfx942-only by design).
"""

import pytest
import torch

from aiter.ops.triton.utils.types import e4m3_dtype


class _Params:
    """Minimal stand-in for the wrapper's _UAParams (only lookup axes used)."""

    def __init__(
        self,
        head_size,
        max_seqlen_q,
        sliding_window=0,
        dtype=torch.bfloat16,
        kv_dtype=None,
        shuffled_kv_cache=False,
        block_size=64,
    ):
        self.head_size = head_size
        self.max_seqlen_q = max_seqlen_q
        self.max_seqlen_k = 32768
        self.sliding_window = sliding_window
        self.shuffled_kv_cache = shuffled_kv_cache
        self.block_size = block_size
        self.q_dtype = dtype
        self.kv_cache_dtype = kv_dtype if kv_dtype is not None else dtype


def _matched_key(params):
    from aiter.ops.triton.utils.unified_attention_utils import (
        _axis_values,
        _load,
        _lookup,
    )

    table, axes, _ = _load("attn_2d", "triton", "gfx942")
    values = _axis_values(
        params.head_size,
        params.max_seqlen_q,
        params.max_seqlen_k,
        params.sliding_window,
        params.shuffled_kv_cache,
        params.block_size,
        params.q_dtype,
        params.kv_cache_dtype,
    )
    key, _config = _lookup(table, axes, values)
    return key, _config


# The pre-shuffled A8W8 KV path pins TILE_SIZE = block_size after the lookup,
# so the non-shuffled composites (BLOCK_M 64/128 at TILE 16) would be
# re-launched at TILE 64 and blow the 64KB LDS at head 512. The SHUF and BS
# axes (added to the schema alongside these entries) keep pre-shuffled prefill
# on LDS-safe, separately tuned entries: page <= 64 gets the tuned configs,
# larger pages fall back to BLOCK_M 16 / 1 stage (the only specialization that
# fits LDS at TILE 128). All cases below run with shuffled_kv_cache=True; the
# non-shuffled counterparts (which must NOT match the SHUF entries) are the
# fp8 rows in _CASES above. Covers prefill and decode.
_SHUF_CASES = [
    # shuffled prefill, page 64: the tuned SHUF+BS_LEQ_64 entries
    (512, 16384, e4m3_dtype, "D_GEQ_512.Q_GEQ_1024.SHUF.BS_LEQ_64.DT_fp8_fp8", 32),
    (256, 16384, e4m3_dtype, "D_GEQ_256.Q_GEQ_1024.SHUF.BS_LEQ_64.DT_fp8_fp8", 128),
    (512, 16384, torch.bfloat16, "D_GEQ_512.Q_GEQ_1024.SHUF.BS_LEQ_64.DT_bf16_bf16", 16),
    (256, 16384, torch.bfloat16, "D_GEQ_256.Q_GEQ_1024.SHUF.BS_LEQ_64.DT_bf16_bf16", 16),
    # shuffled prefill, page 128: BS_LEQ_64 must not match; the BS-agnostic
    # M16/s1 fallbacks serve the call (only LDS-safe config at TILE 128)
    (512, 16384, e4m3_dtype, "D_GEQ_512.Q_GEQ_1024.SHUF.DT_fp8_fp8", 16),
    (256, 16384, torch.bfloat16, "D_GEQ_256.Q_GEQ_1024.SHUF.DT_bf16_bf16", 16),
    # shuffled decode still resolves to the Q_LEQ_1 entries
    (512, 1, e4m3_dtype, "D_GEQ_512.Q_LEQ_1.DT_fp8_fp8", 16),
    (256, 1, e4m3_dtype, "D_GEQ_256.Q_LEQ_1.DT_fp8_fp8", 16),
]


# (head_size, max_seqlen_q, sliding_window, q_dtype, kv_dtype,
#  expected_key, expected_block_m)
_CASES = [
    # full-attention large prefill (Gemma-4 full-attn layers, head 512):
    # dtype-split composites, fp8 at BLOCK_M 128, bf16 at 64.
    (512, 16384, 0, e4m3_dtype, e4m3_dtype, "D_GEQ_512.Q_GEQ_1024.DT_fp8_fp8", 128),
    (512, 16384, 0, torch.bfloat16, torch.bfloat16, "D_GEQ_512.Q_GEQ_1024.DT_bf16_bf16", 64),
    # sliding-window large prefill (Gemma-4 sliding layers, head 256). The
    # gfx942 attn_2d table has no SW axis, so full and sliding prefill
    # resolve through the same composite entries; SW is passed exactly as
    # the wrapper passes it so the rows match the real call shape and stay
    # correct if a SW-scoped entry is ever added.
    (256, 16384, 1024, e4m3_dtype, e4m3_dtype, "D_GEQ_256.Q_GEQ_1024.DT_fp8_fp8", 64),
    (256, 16384, 1024, torch.bfloat16, torch.bfloat16, "D_GEQ_256.Q_GEQ_1024.DT_bf16_bf16", 64),
    # decode controls: the Q_LEQ_1 keys must still win at q=1
    (512, 1, 0, torch.bfloat16, torch.bfloat16, "D_GEQ_512.Q_LEQ_1", 16),
    (512, 1, 0, e4m3_dtype, e4m3_dtype, "D_GEQ_512.Q_LEQ_1.DT_fp8_fp8", 16),
    (256, 1, 0, torch.bfloat16, torch.bfloat16, "D_GEQ_256.Q_LEQ_1", 16),
    (256, 1, 0, e4m3_dtype, e4m3_dtype, "D_GEQ_256.Q_LEQ_1.DT_fp8_fp8", 16),
    # small-head control: standalone Q_GEQ_256 still serves head<=128 prefill
    (128, 16384, 0, torch.bfloat16, torch.bfloat16, "Q_GEQ_256", 128),
    (128, 16384, 0, e4m3_dtype, e4m3_dtype, "Q_GEQ_256", 128),
    # Q boundary: the composites bind at exactly Q>=1024. Just below the
    # threshold the pre-existing D-only (bf16) and D+DT (fp8) entries serve
    # the call at BLOCK_M=16; asserting both sides of 1023/1024 pins the
    # threshold so a future re-tune cannot silently move it.
    (512, 1023, 0, torch.bfloat16, torch.bfloat16, "D_GEQ_512", 16),
    (512, 1023, 0, e4m3_dtype, e4m3_dtype, "D_GEQ_512.DT_fp8_fp8", 16),
    (512, 1024, 0, torch.bfloat16, torch.bfloat16, "D_GEQ_512.Q_GEQ_1024.DT_bf16_bf16", 64),
    (512, 1024, 0, e4m3_dtype, e4m3_dtype, "D_GEQ_512.Q_GEQ_1024.DT_fp8_fp8", 128),
    (256, 1023, 0, torch.bfloat16, torch.bfloat16, "D_GEQ_256", 16),
    (256, 1023, 0, e4m3_dtype, e4m3_dtype, "D_GEQ_256", 16),
    (256, 1024, 0, torch.bfloat16, torch.bfloat16, "D_GEQ_256.Q_GEQ_1024.DT_bf16_bf16", 64),
    (256, 1024, 0, e4m3_dtype, e4m3_dtype, "D_GEQ_256.Q_GEQ_1024.DT_fp8_fp8", 64),
]


@pytest.mark.parametrize(
    "head_size, max_seqlen_q, sliding_window, q_dtype, kv_dtype, expected_key,"
    " expected_block_m",
    _CASES,
)
def test_gfx942_large_head_prefill_lookup(
    head_size,
    max_seqlen_q,
    sliding_window,
    q_dtype,
    kv_dtype,
    expected_key,
    expected_block_m,
):
    key, config = _matched_key(
        _Params(
            head_size,
            max_seqlen_q,
            sliding_window=sliding_window,
            dtype=q_dtype,
            kv_dtype=kv_dtype,
        )
    )
    assert key == expected_key, f"expected {expected_key}, matched {key}"
    assert (
        config["BLOCK_M"] == expected_block_m
    ), f"expected BLOCK_M={expected_block_m} via {expected_key}, got {config['BLOCK_M']}"


@pytest.mark.parametrize(
    "head_size, max_seqlen_q, q_dtype, expected_key, expected_block_m",
    _SHUF_CASES,
)
def test_gfx942_shuffled_kv_lookup(head_size, max_seqlen_q, q_dtype, expected_key, expected_block_m):
    """Shuffled prefill (tuned page-64 entries and page-128 fallback) and decode."""
    block_size = 64
    if "BS_LEQ_64" not in expected_key and max_seqlen_q > 1:
        # the BS-agnostic fallback cases exercise a larger page
        block_size = 128
    key, config = _matched_key(
        _Params(
            head_size,
            max_seqlen_q,
            dtype=q_dtype,
            kv_dtype=q_dtype,
            shuffled_kv_cache=True,
            block_size=block_size,
        )
    )
    assert key == expected_key, f"expected {expected_key}, matched {key}"
    assert (
        config["BLOCK_M"] == expected_block_m
    ), f"expected BLOCK_M={expected_block_m} via {expected_key}, got {config['BLOCK_M']}"

