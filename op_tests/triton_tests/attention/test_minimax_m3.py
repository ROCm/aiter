# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Opt-in tests of the public ATOM wrappers used by MiniMax-M3."""

import os

import pytest
import torch
import triton

from aiter.ops.triton.attention.minimax_m3 import (
    minimax_m3_packed_topk,
    minimax_m3_prefill_score,
)
from aiter.ops.triton.utils._triton.arch_info import get_arch
from op_tests.triton_tests.utils import minimax_m3 as m3

pytestmark = pytest.mark.skipif(
    os.getenv("AITER_TEST_MINIMAX_M3_ATOM") != "1",
    reason="Install ATOM and set AITER_TEST_MINIMAX_M3_ATOM=1",
)


def test_context_score():
    torch.manual_seed(42)
    data = m3.index_inputs()
    actual = m3.context_score(**data)
    batch, query = 2, 4
    references = []
    for b in range(batch):
        pages = data["block_table"][b, ::4].long()
        keys = data["index_cache"][pages].to(torch.bfloat16).float()
        q = data["idx_q"][b * query : (b + 1) * query].float()
        scores = torch.einsum("qhd,btd->h q b t", q, keys)
        positions = torch.arange(keys.shape[0], device="cuda")[:, None] * 512
        positions = positions + torch.arange(128, device="cuda")[None]
        ends = 4096 - query + torch.arange(query, device="cuda") + 1
        scores = scores.masked_fill(
            positions[None, None] >= ends[None, :, None, None], -float("inf")
        )
        references.append(scores.amax(-1) * (128**-0.5) * 1.4426950408889634)
    expected = torch.cat(references, dim=1)
    torch.testing.assert_close(actual, expected, atol=0.003, rtol=0.02)


def test_prefill_index_topk():
    actual, sparse_table, lengths = m3.index_prefill(**m3.prefill_inputs())
    expected = torch.arange(31, 15, -1, dtype=torch.int32, device="cuda")
    torch.testing.assert_close(actual, expected.expand_as(actual))
    # Each selected 128-token index block emits eight 16-token KV pages.
    emitted = torch.cat((expected[1:], expected[:1]))  # Causal tail is emitted last.
    pages = (emitted[:, None] * 8 + torch.arange(8, device="cuda")).flatten()
    torch.testing.assert_close(
        sparse_table, pages.to(torch.int32).expand_as(sparse_table)
    )
    torch.testing.assert_close(
        lengths, torch.arange(2018, 2049, dtype=torch.int32, device="cuda")
    )


@pytest.mark.parametrize("query", [31, 257])
def test_prefill_split_stages(query):
    data = m3.prefill_score_inputs(query, 4096)
    score = minimax_m3_prefill_score(**data)
    # Only valid causal blocks are defined by the production scoring kernel.
    row = torch.arange(query, device="cuda")
    valid = torch.arange(32, device="cuda")[None, :] * 128 < (
        4096 - query + row[:, None] + 1
    )
    keys = data["index_cache"][data["block_table"][0].long()].float()
    q = data["idx_q"].to(data["index_cache"].dtype).float()
    dots = torch.einsum("qhd,btd->hqbt", q, keys)
    positions = torch.arange(4096, device="cuda").reshape(32, 128)
    ends = 4096 - query + row + 1
    dots.masked_fill_(positions[None, None] >= ends[None, :, None, None], -float("inf"))
    reference = dots.amax(-1) * (128**-0.5) * 1.4426950408889634
    torch.testing.assert_close(
        score[0][valid], reference[0][valid], atol=0.003, rtol=0.02
    )

    selector = m3.packed_topk_inputs(query, 4096)
    selector["score"] = score
    actual = minimax_m3_packed_topk(**selector)
    inputs = {key: value for key, value in data.items() if key != "score"}
    expected = m3.index_prefill(**inputs)
    for output, oracle in zip(actual, expected):
        torch.testing.assert_close(output, oracle, rtol=0, atol=0)


def test_decode_candidate_exchange():
    data = m3.candidate_inputs()
    local = m3.local_topk(**data)
    assert local.shape == (4, 8, 16)
    actual, sparse_table, lengths = m3.merge_topk(**m3.merge_inputs())
    expected = torch.arange(31, 15, -1, dtype=torch.int32, device="cuda")
    torch.testing.assert_close(actual, expected.expand_as(actual))
    emitted = torch.cat((expected[1:], expected[:1]))
    pages = (emitted[:, None] * 8 + torch.arange(8, device="cuda")).flatten()
    offsets = torch.arange(2, device="cuda").repeat_interleave(4) * 32 * 8
    torch.testing.assert_close(
        sparse_table, (pages[None] + offsets[:, None]).to(torch.int32)
    )
    torch.testing.assert_close(
        lengths, torch.arange(2045, 2049, dtype=torch.int32, device="cuda").repeat(2)
    )


@pytest.mark.parametrize("tokens", [129, 1024])
@pytest.mark.xfail(
    get_arch() == "gfx1250-strict"
    and torch.__version__.startswith("2.13.0+rocm10.3.0a20261106")
    and triton.__version__ == "3.9.0",
    reason="Pinned strict stack produces incorrect FP8 SHUFFLE gather values; tracking issue #2198",
    strict=True,
)
def test_cache_gather(tokens):
    data = m3.gather_inputs(tokens)
    actual_k, actual_v = m3.cache_gather(**data)
    k = data["key_cache"].float().permute(0, 3, 1, 2, 4).reshape(-1, 1, 128)
    v = data["value_cache"].float().permute(0, 2, 4, 1, 3).reshape(-1, 1, 128)
    ks = data["k_scales"].permute(0, 2, 1).reshape(-1, 1, 1)
    vs = data["v_scales"].permute(0, 2, 1).reshape(-1, 1, 1)
    torch.testing.assert_close(actual_k, (k * ks)[:tokens].bfloat16())
    torch.testing.assert_close(actual_v, (v * vs)[:tokens].bfloat16())
