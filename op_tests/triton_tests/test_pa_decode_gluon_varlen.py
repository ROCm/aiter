# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Ragged (query_start_loc) inputs for pa_decode_gluon's PS path.

Each sequence of a ragged call must match a batch-1 call of the uniform path
with query_length equal to that sequence's length, and rows not owned by any
sequence must be left untouched.
"""

import random

import pytest
import torch
import triton

import aiter
import aiter.ops.triton.gluon.pa_decode_gluon as pa_mod
from aiter import pertoken_quant
from op_tests.triton_tests.test_pa_decode_gluon import (
    quantize_kv_cache_per_tensor,
    quantize_kv_cache_symmetric,
    shuffle_value_cache_layout,
    torch_mha_extend,
)

HEAD_SIZE = 128
SENTINEL = 1024.0

pytestmark = pytest.mark.skipif(
    not pa_mod.GLUON_JIT_KERNEL_ENABLED, reason="gluon JIT is not available"
)


@pytest.fixture
def reduce_backend(request, monkeypatch):
    """Force one PS reduce backend and record its varlen launches."""
    backend = request.param
    varlen_calls = []
    target = None
    if backend == "hip":
        if not pa_mod.CXX_PS_REDUCE_AVAILABLE:
            pytest.skip("HIP PS reduce is not available")
        target = "launch_pa_decode_ps_reduce_cxx"
    elif backend == "flydsl":
        if not pa_mod.FLYDSL_PS_REDUCE_AVAILABLE:
            pytest.skip("FlyDSL PS reduce is not available")
        monkeypatch.setattr(pa_mod, "CXX_PS_REDUCE_AVAILABLE", False)
        target = "launch_pa_decode_ps_reduce_flydsl"
    else:
        monkeypatch.setattr(pa_mod, "CXX_PS_REDUCE_AVAILABLE", False)
        monkeypatch.setattr(pa_mod, "FLYDSL_PS_REDUCE_AVAILABLE", False)
    if target is not None:
        launch = getattr(pa_mod, target)

        def spy(*args, **kwargs):
            launch(*args, **kwargs)
            varlen_calls.append(kwargs.get("query_start_loc") is not None)

        monkeypatch.setattr(pa_mod, target, spy)
    return backend, varlen_calls


def _make_inputs(
    query_lens,
    num_heads,
    block_size,
    quant,
    use_sinks,
    seed,
    trans_v=False,
    device="cuda",
):
    random.seed(seed)
    torch.manual_seed(seed)
    num_query_heads, num_kv_heads = num_heads
    num_seqs = len(query_lens)
    query_start_loc = [0]
    for query_len in query_lens:
        query_start_loc.append(query_start_loc[-1] + query_len)
    num_tokens = query_start_loc[-1]

    # The first sequence has no history, so its context is exactly its query.
    context_lens = [
        max(q, 1) if i == 0 else random.randint(max(q, 1), 700)
        for i, q in enumerate(query_lens)
    ]
    max_blocks = triton.cdiv(max(context_lens), block_size)
    num_blocks = num_seqs * max_blocks
    block_tables = torch.randperm(num_blocks, dtype=torch.int32, device=device)
    block_tables = block_tables.view(num_seqs, max_blocks)

    # Queries are a strided slice of a fused QKV tensor as in real models, plus
    # trailing rows owned by no sequence.
    qkv = torch.empty(
        num_tokens + 3,
        num_query_heads + 2 * num_kv_heads,
        HEAD_SIZE,
        dtype=torch.bfloat16,
        device=device,
    ).uniform_(-1, 1)
    query = qkv[:, :num_query_heads]

    x = 16 // 2
    key_cache = torch.empty(
        num_blocks,
        num_kv_heads,
        HEAD_SIZE // x,
        block_size,
        x,
        dtype=torch.bfloat16,
        device=device,
    ).uniform_(-1, 1)
    value_cache = torch.empty(
        num_blocks,
        num_kv_heads,
        HEAD_SIZE,
        block_size,
        dtype=torch.bfloat16,
        device=device,
    ).uniform_(-1, 1)

    inputs = {
        "query": query,
        "key_cache": key_cache,
        "value_cache": value_cache,
        "query_scale": None,
        "key_scale": None,
        "value_scale": None,
        "compute_type": torch.bfloat16,
    }
    if quant == "fp8_kv":
        # vLLM's configuration: bf16 query, per-tensor [1] KV descales.
        key_fp8, _, value_fp8, _, key_scale, value_scale = quantize_kv_cache_per_tensor(
            key_cache, value_cache, quant_dtype=aiter.dtypes.fp8
        )
        inputs.update(
            key_cache=key_fp8,
            value_cache=value_fp8,
            key_scale=key_scale.reshape(1),
            value_scale=value_scale.reshape(1),
            compute_type=aiter.dtypes.fp8,
        )
    elif quant == "fp8_q_kv":
        query_fp8, query_scale = pertoken_quant(
            query.contiguous(), quant_dtype=aiter.dtypes.fp8
        )
        key_fp8, _, value_fp8, _, key_scale, value_scale = quantize_kv_cache_symmetric(
            key_cache, value_cache, quant_dtype=aiter.dtypes.fp8
        )
        inputs.update(
            query=query_fp8,
            key_cache=key_fp8,
            value_cache=value_fp8,
            query_scale=query_scale,
            key_scale=key_scale,
            value_scale=value_scale,
            compute_type=aiter.dtypes.fp8,
        )
    # The torch reference reads the 4D layout; the kernel may get the 5D
    # [blocks, kv_heads, block_size // x, head_size, x] one vLLM uses.
    inputs["value_cache_ref"] = inputs["value_cache"]
    if trans_v:
        inputs["value_cache"] = shuffle_value_cache_layout(inputs["value_cache"])
    sinks = (
        torch.randn(num_query_heads, dtype=torch.bfloat16, device=device)
        if use_sinks
        else None
    )
    return (
        inputs,
        sinks,
        torch.tensor(query_start_loc, dtype=torch.int32, device=device),
        torch.tensor(context_lens, dtype=torch.int32, device=device),
        block_tables,
    )


def _run(
    inputs,
    output,
    context_lens,
    block_tables,
    query_length,
    num_splits,
    sinks,
    sliding_window,
    partition_size,
    query_start_loc=None,
    rows=slice(None),
):
    query_scale = inputs["query_scale"]
    if query_scale is not None:
        query_scale = query_scale[rows]
    pa_mod.pa_decode_gluon(
        output,
        inputs["query"][rows],
        inputs["key_cache"],
        inputs["value_cache"],
        context_lens,
        block_tables,
        1.0 / HEAD_SIZE**0.5,
        query_length,
        num_splits,
        partition_size,
        compute_type=inputs["compute_type"],
        query_scale=query_scale,
        key_scale=inputs["key_scale"],
        value_scale=inputs["value_scale"],
        sinks=sinks,
        sliding_window=sliding_window,
        ps=True,
        query_start_loc=query_start_loc,
    )


QUERY_LENS = [
    ((4, 4, 1), 4),
    ((1, 2, 3, 4), 4),
    ((2, 2), 2),
    ((3, 1, 3), 3),
    ((1, 2, 1), 4),
    ((2, 0, 3), 3),
]
NUM_HEADS = {
    "head1_g16": (16, 1),
    "head1_g8": (8, 1),
    "multi_g16": (64, 4),
    "multi_g8": (16, 2),
}
# block_size, trans_v, quant, use_sinks, sliding_window, partition_size
KERNEL_CONFIGS = {
    "bf16": (16, False, None, False, 0, 256),
    "bf16_sinks_b64": (64, False, None, True, 0, 256),
    "fp8_q_kv": (16, False, "fp8_q_kv", False, 0, 256),
    "fp8_kv": (16, False, "fp8_kv", False, 0, 256),
    "sliding128_b64": (64, False, None, False, 128, 256),
    # vLLM's sliding-window setup: window + 1, 128-token partitions.
    "fp8_kv_sinks_sliding128_p128": (16, False, "fp8_kv", True, 129, 128),
    "sliding256_p128": (16, False, None, False, 257, 128),
    # vLLM's shuffled KV layout with 128-token pages (MiniMax-M3).
    "b128_transv_bf16": (128, True, None, False, 0, 256),
    "b128_transv_fp8_kv_sinks": (128, True, "fp8_kv", True, 0, 256),
    "b128_transv_fp8_kv_sliding128_p128": (128, True, "fp8_kv", False, 129, 128),
}
# (reduce backend, num_splits); one-shot runs no reduce.
RUNS = [("hip", 8), ("flydsl", 8), ("triton", 8), ("triton", 1)]


def _cases():
    """Pairwise instead of the full product: every kernel config meets every
    head config once across RUNS, and each run sees every query-length pattern.
    """
    cases = []
    head_ids = list(NUM_HEADS)
    for i, (config_id, config) in enumerate(KERNEL_CONFIGS.items()):
        for run, (backend, num_splits) in enumerate(RUNS):
            head_id = head_ids[(i + run) % len(head_ids)]
            query_lens, query_length = QUERY_LENS[(i + 2 * run) % len(QUERY_LENS)]
            run_id = "one_shot" if num_splits == 1 else f"split8_{backend}"
            lens_id = "_".join(map(str, query_lens))
            cases.append(
                pytest.param(
                    backend,
                    query_lens,
                    query_length,
                    NUM_HEADS[head_id],
                    num_splits,
                    *config,
                    id=f"{run_id}-{config_id}-{head_id}-q{lens_id}",
                )
            )
    return cases


@pytest.mark.parametrize(
    "reduce_backend, query_lens, query_length, num_heads, num_splits, "
    "block_size, trans_v, quant, use_sinks, sliding_window, partition_size",
    _cases(),
    indirect=["reduce_backend"],
)
def test_varlen_matches_uniform(
    reduce_backend,
    query_lens,
    query_length,
    num_heads,
    num_splits,
    block_size,
    trans_v,
    quant,
    use_sinks,
    sliding_window,
    partition_size,
):
    backend, varlen_calls = reduce_backend
    inputs, sinks, query_start_loc, context_lens, block_tables = _make_inputs(
        query_lens,
        num_heads,
        block_size,
        quant,
        use_sinks,
        seed=len(query_lens),
        trans_v=trans_v,
    )
    query = inputs["query"]
    output = torch.full(
        query.shape, SENTINEL, dtype=torch.bfloat16, device=query.device
    )
    _run(
        inputs,
        output,
        context_lens,
        block_tables,
        query_length,
        num_splits,
        sinks,
        sliding_window,
        partition_size,
        query_start_loc=query_start_loc,
    )
    torch.cuda.synchronize()

    starts = query_start_loc.tolist()
    num_tokens = starts[-1]
    assert torch.all(output[num_tokens:] == SENTINEL), "unowned rows were written"
    atol = 5e-2 if quant else 1e-2
    for seq, query_len in enumerate(query_lens):
        if query_len == 0:
            continue
        rows = slice(starts[seq], starts[seq + 1])
        expected = torch.empty_like(output[rows])
        _run(
            inputs,
            expected,
            context_lens[seq : seq + 1],
            block_tables[seq : seq + 1],
            query_len,
            num_splits,
            sinks,
            sliding_window,
            partition_size,
            rows=rows,
        )
        torch.testing.assert_close(
            output[rows], expected, atol=atol, rtol=atol, msg=f"sequence {seq}"
        )

    if not quant and sliding_window == 0:
        reference = torch_mha_extend(
            query[:num_tokens],
            inputs["key_cache"],
            inputs["value_cache_ref"],
            block_tables,
            context_lens,
            query_start_loc,
            sinks=sinks,
        )
        torch.testing.assert_close(output[:num_tokens], reference, atol=2e-2, rtol=2e-2)

    if num_splits > 1 and backend != "triton":
        assert any(varlen_calls), f"{backend} reduce did not handle the varlen call"


def test_varlen_rejects_dot_kernel_path():
    inputs, _, query_start_loc, context_lens, block_tables = _make_inputs(
        (2, 1), (8, 1), 16, None, False, seed=0
    )
    output = torch.empty_like(inputs["query"])
    with pytest.raises(AssertionError, match="only supported on the PS path"):
        pa_mod.pa_decode_gluon(
            output,
            inputs["query"],
            inputs["key_cache"],
            inputs["value_cache"],
            context_lens,
            block_tables,
            1.0 / HEAD_SIZE**0.5,
            2,
            8,
            ps=False,
            query_start_loc=query_start_loc,
        )
