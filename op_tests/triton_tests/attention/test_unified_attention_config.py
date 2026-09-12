"""Lookup regression tests for the unified_attention config tables.

The D axis outranks the Q axis in `_lookup`'s candidate ordering, so a
`D_GEQ_*` entry at BLOCK_M=16 shadows the standalone `Q_GEQ_256` entry
(BLOCK_M=128) for large-head large-prefill shapes. The composite
`D_GEQ_*.Q_GEQ_256` keys in the gfx942 table exist to bind both axes at
once; without them large prefill silently falls back to the decode-era
BLOCK_M=16.

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
    ):
        self.head_size = head_size
        self.max_seqlen_q = max_seqlen_q
        self.max_seqlen_k = 32768
        self.sliding_window = sliding_window
        self.shuffled_kv_cache = False
        self.block_size = 64
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


# The composite prefill keys intentionally wildcard the DT axis: bf16 and
# fp8(q+kv) must both resolve to them. The decode controls resolve to the
# fp8-specific DT keys under fp8 and the generic Q_LEQ_1 keys under bf16.
# Parametrizing over both dtypes keeps the advertised dtype-agnostic
# prefill behavior covered: a future fp8-only narrowing of the composites
# would flip the fp8 rows and fail here.
_DT_CASES = [
    # (head_size, max_seqlen_q, q_dtype, kv_dtype, expected_key, expected_block_m)
    # full-attention large prefill (Gemma-4 full attn, head 512)
    (512, 16384, torch.bfloat16, torch.bfloat16, "D_GEQ_512.Q_GEQ_256", 128),
    (512, 16384, e4m3_dtype, e4m3_dtype, "D_GEQ_512.Q_GEQ_256", 128),
    # sliding-window large prefill (Gemma-4 sliding, head 256)
    (256, 16384, torch.bfloat16, torch.bfloat16, "D_GEQ_256.Q_GEQ_256", 128),
    (256, 16384, e4m3_dtype, e4m3_dtype, "D_GEQ_256.Q_GEQ_256", 128),
    # decode controls: the Q_LEQ_1 keys must still win at q=1
    (512, 1, torch.bfloat16, torch.bfloat16, "D_GEQ_512.Q_LEQ_1", 16),
    (512, 1, e4m3_dtype, e4m3_dtype, "D_GEQ_512.Q_LEQ_1.DT_fp8_fp8", 16),
    (256, 1, torch.bfloat16, torch.bfloat16, "D_GEQ_256.Q_LEQ_1", 16),
    (256, 1, e4m3_dtype, e4m3_dtype, "D_GEQ_256.Q_LEQ_1.DT_fp8_fp8", 16),
    # small-head control: standalone Q_GEQ_256 still serves head<=128 prefill
    (128, 16384, torch.bfloat16, torch.bfloat16, "Q_GEQ_256", 128),
    (128, 16384, e4m3_dtype, e4m3_dtype, "Q_GEQ_256", 128),
]


@pytest.mark.parametrize(
    "head_size, max_seqlen_q, q_dtype, kv_dtype, expected_key, expected_block_m",
    _DT_CASES,
)
def test_gfx942_large_head_prefill_lookup(
    head_size, max_seqlen_q, q_dtype, kv_dtype, expected_key, expected_block_m
):
    key, config = _matched_key(
        _Params(head_size, max_seqlen_q, q_dtype=q_dtype, kv_dtype=kv_dtype)
    )
    assert key == expected_key, f"expected {expected_key}, matched {key}"
    assert (
        config["BLOCK_M"] == expected_block_m
    ), f"expected BLOCK_M={expected_block_m} via {expected_key}, got {config['BLOCK_M']}"
