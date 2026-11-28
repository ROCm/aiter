# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Numerical coverage for the gfx950 batch-one prefill recovery paths."""

import pytest
import torch

from aiter.ops.triton.attention.unified_attention import unified_attention
from aiter.ops.triton.utils._triton import arch_info
from op_tests.triton_tests.attention.test_unified_attention import (
    generate_data,
    ref_paged_attn,
)


@pytest.mark.skipif(arch_info.get_arch() != "gfx950", reason="gfx950 prefill tuning")
@pytest.mark.parametrize(
    "query_len,kv_len",
    [
        pytest.param(2510, 2510, id="weekly-large-prefill"),
        pytest.param(462, 974, id="weekly-small-chunk"),
        pytest.param(4095, 4095, id="large-boundary-eight-pages"),
        pytest.param(2049, 2049, id="large-boundary-first-enabled-query"),
        pytest.param(257, 769, id="small-boundary-partial-tiles"),
        pytest.param(448, 896, id="small-boundary-exact-query-tiles"),
    ],
)
def test_unified_attention_prefill_recovery(query_len, kv_len, record_property):
    """Exercise production dispatch with actual lengths and random page order."""
    block_size = 512
    (
        query,
        key_cache_orig,
        value_cache_orig,
        key_cache,
        value_cache,
        _sinks,
        output,
        cu_query_lens,
        kv_lens,
        max_query_len,
        max_kv_len,
        scale,
        window_size,
        block_tables,
        _maybe_quant_query,
        _query_scales,
        q_descale,
        k_descale,
        v_descale,
        output_scale,
    ) = generate_data(
        seq_lens=[(query_len, kv_len)],
        num_blocks=(kv_len + block_size - 1) // block_size + 3,
        block_size=block_size,
        head_size=128,
        num_heads=(16, 16),
        q_dtype=torch.bfloat16,
        kv_dtype=torch.bfloat16,
        out_dtype=torch.bfloat16,
        shuffled_kv_cache=False,
        use_q_descale=False,
        use_kv_descale=False,
        use_out_scale=False,
        device="cuda",
    )

    unified_attention(
        q=query,
        k=key_cache,
        v=value_cache,
        out=output,
        cu_seqlens_q=cu_query_lens,
        seqused_k=kv_lens,
        max_seqlen_q=max_query_len,
        max_seqlen_k=max_kv_len,
        softmax_scale=scale,
        causal=True,
        window_size=window_size,
        block_table=block_tables,
        softcap=0,
        q_descale=q_descale,
        k_descale=k_descale,
        v_descale=v_descale,
        sinks=None,
        output_scale=output_scale,
        shuffled_kv_cache=False,
        backend="gluon",
    )
    reference = ref_paged_attn(
        query=query,
        key_cache=key_cache_orig,
        value_cache=value_cache_orig,
        query_lens=[query_len],
        kv_lens=[kv_len],
        block_tables=block_tables,
        scale=scale,
        out_dtype=torch.bfloat16,
        sinks=None,
    )

    actual = output.float()
    expected = reference.float()
    max_abs_error = (actual - expected).abs().max().item()
    record_property("query_length", query_len)
    record_property("kv_length", kv_len)
    record_property("max_abs_error", max_abs_error)
    # Use the existing unified-attention bf16 tolerance, with no allowed
    # mismatch fraction: every output element must satisfy the comparison.
    torch.testing.assert_close(
        actual,
        expected,
        atol=1.5e-2,
        rtol=1e-2,
        msg=f"q={query_len}, kv={kv_len}, max_abs_error={max_abs_error:.8g}",
    )
