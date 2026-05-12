# SPDX-License-Identifier: MIT
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
"""
Correctness tests for the CK unified-attention kernel.

Same structure as the Triton suite in
`op_tests/triton_tests/attention/test_unified_attention.py` — we reuse its
torch reference (`ref_paged_attn`) and only restrict the parameter sweep to
the shape/dtype combos the CK kernel actually dispatches.

The CK kernel currently supports:
  - hdim=128, MHA  (num_queries_per_kv == 1)
  - hdim=64,  GQA-8 (num_queries_per_kv == 8)
  - fp16 / bf16
  - mask_type 0 (no mask) and 2 (causal)
It does **not** support: fp8 quant, sliding window, softcap, attention sinks.
"""

from __future__ import annotations

import pytest
import torch

from aiter.ops.unified_attention import unified_attention_fwd

# Reuse the exact same reference the Triton suite uses so any drift is caught
# on both sides.
from op_tests.triton_tests.attention.test_unified_attention import ref_paged_attn


# ---------------------------------------------------------------------------
# Supported configurations — only the (num_q_heads, num_kv_heads, head_size)
# tuples that the CK dispatcher has kernel instances for.
# ---------------------------------------------------------------------------
HEAD_CONFIGS = [
    # MHA, d=128
    (4, 4, 128),
    (16, 16, 128),
    # GQA-8, d=64
    (8, 1, 64),
    (32, 4, 64),
]

BLOCK_SIZES = [16, 32, 64]

DTYPES = [torch.float16, torch.bfloat16]

# One large enough to exercise int32 overflow in row*stride, one small enough
# for the schema check. Same pattern as the Triton test.
NUM_BLOCKS = [32768, 2048]

# Mix of prefill+decode and pure-decode batches, same as the Triton suite.
SEQ_LEN_PATTERNS = [
    [(1, 1328), (5, 18), (129, 463)],
    [(1, 523), (1, 37), (1, 2011)],
]

# Number of KV splits to sweep when exercising the split-KV path. 1 is the
# normal (non-split) write path which is covered by `test_ck_unified_attn` —
# we only test >= 2 here to make sure the kernel's split-write branch and the
# Python-side LSE merge agree with the torch reference.
NUM_SPLITS = [2, 4, 8]


def _combine_splits(
    o_acc: torch.Tensor,
    lse_acc: torch.Tensor,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """FlashDecoding-style LSE merge of per-split partials.

    o_acc   : [nhead, num_splits, total_q, hdim]  float32  (already normalized
                                                            per-split: divided
                                                            by its own l)
    lse_acc : [nhead, num_splits, total_q]        float32  (natural-log domain)

    Returns the combined output in the same layout `unified_attention_fwd`
    writes, i.e. [total_q, nhead, hdim] cast to `out_dtype`.

    Combine math (with per-split lse[s] = m[s] + log(l[s])):
        lse_max  = max_s lse[s]
        w[s]     = exp(lse[s] - lse_max)
        out      = sum_s (o_acc[s] * w[s]) / sum_s w[s]
    For masked-empty splits (lse == -inf) the weight is zero, so those splits
    don't contribute.
    """
    is_empty = torch.isinf(lse_acc) & (lse_acc < 0)
    # NaN-safe shift: for -inf rows we don't care what the exponent is (the
    # weight is force-zeroed below) but `(-inf) - (-inf)` would produce NaN
    # which then poisons the sum. Replace with any finite value first.
    safe_lse = torch.where(is_empty, torch.zeros_like(lse_acc), lse_acc)
    lse_max = safe_lse.amax(dim=1, keepdim=True)             # [nhead, 1, total_q]
    weight = torch.exp(safe_lse - lse_max)
    weight = torch.where(is_empty, torch.zeros_like(weight), weight)  # exclude empties
    weight_sum = weight.sum(dim=1, keepdim=True)             # [nhead, 1, total_q]
    weight_sum = torch.where(weight_sum == 0, torch.ones_like(weight_sum), weight_sum)
    w_full = (weight / weight_sum).unsqueeze(-1)             # [nhead, splits, total_q, 1]
    o_merged = (o_acc * w_full).sum(dim=1)                   # [nhead, total_q, hdim]
    return o_merged.transpose(0, 1).contiguous().to(out_dtype)


@pytest.mark.parametrize("seq_lens", SEQ_LEN_PATTERNS)
@pytest.mark.parametrize("head_config", HEAD_CONFIGS)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("num_blocks", NUM_BLOCKS)
@torch.inference_mode()
def test_ck_unified_attn(
    seq_lens: list[tuple[int, int]],
    head_config: tuple[int, int, int],
    block_size: int,
    dtype: torch.dtype,
    num_blocks: int,
) -> None:
    # `ref_paged_attn` always applies a causal-style triangle mask
    # (`triu(diagonal=kv_len - query_len + 1)`), so we always exercise the CK
    # kernel's `mask_type=2` path here — same convention as the Triton suite.
    torch.manual_seed(0)

    num_query_heads, num_kv_heads, head_size = head_config
    assert num_query_heads % num_kv_heads == 0

    query_lens = [x[0] for x in seq_lens]
    kv_lens_list = [x[1] for x in seq_lens]
    num_seqs = len(seq_lens)
    max_kv_len = max(kv_lens_list)

    # The causal/no-mask reference path needs query_len <= kv_len for every
    # sequence (the mask diagonal sits at kv_len - query_len). Reject configs
    # that would put the diagonal outside the valid window — same behavior as
    # the Triton suite implicitly relies on.
    for ql, kl in seq_lens:
        if ql > kl:
            pytest.skip(f"reference requires query_len ({ql}) <= kv_len ({kl})")

    scale = head_size**-0.5

    query = torch.randn(
        sum(query_lens), num_query_heads, head_size, dtype=dtype, device="cuda"
    )
    key_cache = torch.randn(
        num_blocks, block_size, num_kv_heads, head_size, dtype=dtype, device="cuda"
    )
    value_cache = torch.randn_like(key_cache)

    cu_query_lens = torch.tensor(
        [0] + query_lens, dtype=torch.int32, device="cuda"
    ).cumsum(dim=0, dtype=torch.int32)
    kv_lens = torch.tensor(kv_lens_list, dtype=torch.int32, device="cuda")

    max_num_blocks_per_seq = (max_kv_len + block_size - 1) // block_size
    block_tables = torch.randint(
        0,
        num_blocks,
        (num_seqs, max_num_blocks_per_seq),
        dtype=torch.int32,
        device="cuda",
    )

    # Detect whether the cache is large enough that block_idx * stride_k_cache_0
    # would overflow int32; the kernel takes a slow-but-safe rebased path in
    # that case. Same logic as `run_ck` in ua-test-scripts/test_single_shape.py.
    stride_k_cache_0 = block_size * num_kv_heads * head_size
    INT32_MAX = 2**31 - 1
    cache_ptr_int32_overflow_possible = (num_blocks * stride_k_cache_0) > INT32_MAX

    output = torch.empty_like(query)
    unified_attention_fwd(
        output,
        query,
        key_cache,
        value_cache,
        block_tables,
        kv_lens,
        cu_query_lens,
        mask_type=2,  # causal — matches `ref_paged_attn`'s triangle mask
        scale_s=scale,
        scale=1.0,
        scale_k=1.0,
        scale_v=1.0,
        scale_out=1.0,
        cache_ptr_int32_overflow_possible=cache_ptr_int32_overflow_possible,
    )

    ref_output = ref_paged_attn(
        query=query.clone(),  # ref_paged_attn does `q *= scale` in-place
        key_cache=key_cache,
        value_cache=value_cache,
        query_lens=query_lens,
        kv_lens=kv_lens_list,
        block_tables=block_tables,
        scale=scale,
        sliding_window=None,
        soft_cap=None,
        sinks=None,
    )

    # Same tolerance as the Triton test for non-quantized dtypes.
    atol, rtol = 1.5e-2, 1e-2
    torch.testing.assert_close(
        output, ref_output, atol=atol, rtol=rtol
    ), f"max abs diff = {torch.max(torch.abs(output - ref_output))}"


# ---------------------------------------------------------------------------
# Split-KV (FlashDecoding-style) correctness sweep.
# Drops the int32-overflow `num_blocks=32768` slice — the rebased-pointer
# prefill path has a known bug exposed by `test_ck_unified_attn` that is
# orthogonal to the split-KV machinery we want to test here.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("seq_lens", SEQ_LEN_PATTERNS)
@pytest.mark.parametrize("head_config", HEAD_CONFIGS)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("num_splits", NUM_SPLITS)
@torch.inference_mode()
def test_ck_unified_attn_splitkv(
    seq_lens: list[tuple[int, int]],
    head_config: tuple[int, int, int],
    block_size: int,
    dtype: torch.dtype,
    num_splits: int,
) -> None:
    """Exercise the kernel's `kargs.num_splits > 1` write path.

    The kernel writes (per-split o_acc, per-split lse) to FP32 workspaces; we
    merge them in Python with `_combine_splits` and compare against
    `ref_paged_attn`. With `num_splits == 1` the kernel takes the normal
    write-to-output path which is already covered by `test_ck_unified_attn`.
    """
    torch.manual_seed(0)

    num_blocks = 2048  # small enough to avoid the int32-overflow path
    num_query_heads, num_kv_heads, head_size = head_config
    assert num_query_heads % num_kv_heads == 0

    query_lens = [x[0] for x in seq_lens]
    kv_lens_list = [x[1] for x in seq_lens]
    num_seqs = len(seq_lens)
    total_q = sum(query_lens)
    max_kv_len = max(kv_lens_list)

    for ql, kl in seq_lens:
        if ql > kl:
            pytest.skip(f"reference requires query_len ({ql}) <= kv_len ({kl})")

    scale = head_size**-0.5

    query = torch.randn(
        total_q, num_query_heads, head_size, dtype=dtype, device="cuda"
    )
    key_cache = torch.randn(
        num_blocks, block_size, num_kv_heads, head_size, dtype=dtype, device="cuda"
    )
    value_cache = torch.randn_like(key_cache)

    cu_query_lens = torch.tensor(
        [0] + query_lens, dtype=torch.int32, device="cuda"
    ).cumsum(dim=0, dtype=torch.int32)
    kv_lens = torch.tensor(kv_lens_list, dtype=torch.int32, device="cuda")

    max_num_blocks_per_seq = (max_kv_len + block_size - 1) // block_size
    block_tables = torch.randint(
        0, num_blocks, (num_seqs, max_num_blocks_per_seq),
        dtype=torch.int32, device="cuda",
    )

    # FP32 workspaces. Layout is [num_q_heads, num_splits, total_q, hdim] for
    # o_acc and [num_q_heads, num_splits, total_q] for lse — the kernel reads
    # the nhead/split strides from the tensor metadata so any contiguous
    # tensor in this shape works. lse defaults to -inf so unwritten rows are
    # treated as empty splits by the merge.
    o_acc = torch.zeros(
        num_query_heads, num_splits, total_q, head_size,
        dtype=torch.float32, device="cuda",
    )
    lse_acc = torch.full(
        (num_query_heads, num_splits, total_q), float("-inf"),
        dtype=torch.float32, device="cuda",
    )

    # The kernel's split-write branch ignores `output`, but the API still
    # requires a valid tensor. A single launch covers all splits — the
    # kernel uses gridDim.z == num_splits internally so each (kv_head, seq,
    # split) tuple runs concurrently rather than being serialized through
    # num_splits separate host-side launches.
    dummy_output = torch.empty_like(query)
    unified_attention_fwd(
        dummy_output,
        query,
        key_cache,
        value_cache,
        block_tables,
        kv_lens,
        cu_query_lens,
        mask_type=2,
        scale_s=scale,
        scale=1.0,
        scale_k=1.0,
        scale_v=1.0,
        scale_out=1.0,
        cache_ptr_int32_overflow_possible=False,
        num_splits=num_splits,
        o_acc_workspace=o_acc,
        lse_acc_workspace=lse_acc,
    )

    output = _combine_splits(o_acc, lse_acc, out_dtype=dtype)

    ref_output = ref_paged_attn(
        query=query.clone(),
        key_cache=key_cache,
        value_cache=value_cache,
        query_lens=query_lens,
        kv_lens=kv_lens_list,
        block_tables=block_tables,
        scale=scale,
        sliding_window=None,
        soft_cap=None,
        sinks=None,
    )

    # Looser tolerance than the non-split test: the per-split divide-by-l
    # followed by the FP32 reaccumulate in the LSE merge can drift a single
    # bf16 element by ~5e-2 even though the mean diff stays at ~1e-3. We rely
    # on (atol, mean) to catch real bugs; pure rtol on near-zero values is
    # uninformative.
    atol, rtol = 6e-2, 1e-2
    diff = (output.float() - ref_output.float()).abs()
    mean_diff = diff.mean().item()
    max_diff = diff.max().item()
    assert mean_diff < 5e-3, (
        f"mean |diff|={mean_diff:.3e} (max={max_diff:.3e}) — likely a real bug "
        f"in the split-KV path, not just bf16 noise"
    )
    torch.testing.assert_close(
        output, ref_output, atol=atol, rtol=rtol
    ), f"max abs diff = {max_diff:.3e}, mean = {mean_diff:.3e}"
