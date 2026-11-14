# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""QSA paged K/V helpers: a paged cache as the token pools and the slot table that
``qsa_sparse_prefill`` reads in place. CPU only; the expected rows are derived by hand or
by indexing the paged cache directly, not with the helpers' formula.
"""

import pytest
import torch
from conftest import requires_built_aiter

requires_built_aiter()

from aiter.ops import qsa

DIM = 256


def record_cache(pages, page_size, heads=1, pitch_rows=None):
    """K and V [pages, page_size, heads, DIM] as the two halves of one [K|V] record per
    token, pages `pitch_rows` rows apart (default: back to back), and random values."""
    pitch_rows = page_size if pitch_rows is None else pitch_rows
    record = torch.randn(pages, pitch_rows, 2, heads, DIM).bfloat16()
    return record[:, :page_size, 0], record[:, :page_size, 1]


def rows_of(pool, pages, page_size):
    """Pool row of every token of the cache, [pages, page_size]."""
    return (
        torch.arange(pages)[:, None] * pool.rows_per_page
        + torch.arange(page_size)[None, :]
    )


def test_a_cache_with_pages_further_apart_than_they_are_long_is_read_in_place():
    # Pages further apart than they are long: 1568-token pages 19,992 rows apart, K|V
    # interleaved per token.
    k_cache, v_cache = record_cache(3, 1568, pitch_rows=19992)
    pool = qsa.qsa_paged_kv_pool(k_cache, v_cache)
    assert (pool.page_size, pool.rows_per_page) == (1568, 19992)
    assert pool.k.shape == pool.v.shape == ((3 - 1) * 19992 + 1568, 1, DIM)
    assert pool.k.data_ptr() == k_cache.data_ptr()
    assert pool.v.data_ptr() == v_cache.data_ptr()
    rows = rows_of(pool, 3, 1568)
    assert torch.equal(pool.k[rows], k_cache)
    assert torch.equal(pool.v[rows], v_cache)


@pytest.mark.parametrize(
    ("pages", "page_size", "heads", "pitch_rows"),
    [
        pytest.param(2, 4, 1, 8, id="page_4_pitch_8"),
        pytest.param(1, 16, 1, None, id="one_page"),
        pytest.param(3, 16, 2, None, id="two_heads_back_to_back"),
        pytest.param(3, 8, 1, 12, id="spare_rows_after_each_page"),
    ],
)
def test_pools_of_other_accepted_geometries(pages, page_size, heads, pitch_rows):
    k_cache, v_cache = record_cache(pages, page_size, heads, pitch_rows)
    pool = qsa.qsa_paged_kv_pool(k_cache, v_cache)
    assert pool.page_size == page_size
    assert pool.rows_per_page == (pitch_rows or page_size)
    rows = rows_of(pool, pages, page_size)
    assert torch.equal(pool.k[rows], k_cache)
    assert torch.equal(pool.v[rows], v_cache)


def test_k_and_v_in_separate_buffers_are_pooled():
    k_cache = torch.randn(2, 8, 1, DIM).bfloat16()
    v_cache = torch.randn(2, 8, 1, DIM).bfloat16()
    pool = qsa.qsa_paged_kv_pool(k_cache, v_cache)
    assert torch.equal(pool.k[rows_of(pool, 2, 8)], k_cache)
    assert torch.equal(pool.v[rows_of(pool, 2, 8)], v_cache)


def misaligned(*shape):
    """A bf16 tensor whose data pointer is 2 bytes past a 16-byte boundary."""
    size = 1
    for n in shape:
        size *= n
    return torch.zeros(size + 8).bfloat16()[1 : 1 + size].view(*shape)


def strided(buffer, shape, strides):
    return torch.zeros(buffer).bfloat16().as_strided(shape, strides)


# Cache [pages=2, page_size=8, heads=1, DIM] of 2048 elements a page unless stated.
REJECTED = {
    "k_3d": lambda: (torch.zeros(8, 1, DIM).bfloat16(),) * 2,
    "shapes_differ": lambda: (
        torch.zeros(2, 8, 1, DIM).bfloat16(),
        torch.zeros(2, 8, 2, DIM).bfloat16(),
    ),
    # Same strides, fewer pages: only the shapes differ.
    "shapes_differ_same_strides": lambda: (
        record_cache(2, 8)[0],
        record_cache(2, 8)[0][:1],
    ),
    "strides_differ": lambda: (
        record_cache(2, 8)[0],
        record_cache(2, 8)[0].contiguous(),
    ),
    "no_pages": lambda: (torch.zeros(0, 8, 1, DIM).bfloat16(),) * 2,
    "empty_pages": lambda: (torch.zeros(2, 0, 1, DIM).bfloat16(),) * 2,
    "no_heads": lambda: (torch.zeros(2, 8, 0, DIM).bfloat16(),) * 2,
    "no_head_dim": lambda: (torch.zeros(2, 8, 1, 0).bfloat16(),) * 2,
    # Token stride 0 over empty heads: nothing to read, and no division by the stride.
    "empty_expanded": lambda: (
        torch.zeros(1, 1, 0, DIM).bfloat16().expand(2, 8, 0, DIM),
    )
    * 2,
    "element_stride": lambda: (
        strided(2 * 8 * DIM * 2, (2, 8, 1, DIM), (4096, 512, 512, 2)),
    )
    * 2,
    "heads_apart": lambda: (
        strided(2 * 8 * 2 * 512, (2, 8, 2, DIM), (8192, 1024, 512, 1)),
    )
    * 2,
    # A token's row (2 x 256) is longer than the 128 elements between tokens.
    "tokens_overlap": lambda: (
        strided(2 * 8 * DIM, (2, 8, 2, DIM), (2048, 128, 256, 1)),
    )
    * 2,
    # Page stride 4196 = 4 x 1024 + 100 is no whole number of 1024-element rows.
    "page_pitch_between_rows": lambda: (
        strided(2 * 4196, (2, 8, 1, DIM), (4196, 512, 256, 1)),
    )
    * 2,
    # Pages of 6 tokens 8 rows apart: a block could straddle pages 6 tokens long.
    "page_size_6": lambda: record_cache(2, 6, pitch_rows=8),
    # Pages of 8 tokens 10 rows apart: page 1 starts at row 10, not a multiple of 4.
    "pitch_10_rows": lambda: record_cache(2, 8, pitch_rows=10),
    "k_misaligned": lambda: (
        misaligned(2, 8, 1, DIM),
        torch.zeros(2, 8, 1, DIM).bfloat16(),
    ),
    "v_misaligned": lambda: (
        torch.zeros(2, 8, 1, DIM).bfloat16(),
        misaligned(2, 8, 1, DIM),
    ),
}


@pytest.mark.parametrize("name", REJECTED)
def test_a_cache_the_op_cannot_read_in_place_has_no_pool(name):
    assert qsa.qsa_paged_kv_pool(*REJECTED[name]()) is None


def test_slots_of_a_hand_derived_table():
    # Pages of 4 tokens, 8 rows apart. Request 0: 5 + 3 = 8 tokens on pages 2, 0;
    # request 1: 2 tokens on page 1.
    block_table = torch.tensor([[2, 0, 5], [1, 3, 0]], dtype=torch.int32)
    slots = qsa.qsa_paged_kv_slots((3, 2), (5, 0), block_table, 4, 8)
    assert slots.dtype == torch.int32
    assert slots.tolist() == [16, 17, 18, 19, 0, 1, 2, 3, 8, 9]


def test_the_slots_do_not_depend_on_the_default_device():
    # The host lengths are host data whatever device new tensors default to: a consumer
    # with a default device would get them built there, by a blocking copy.
    block_table = torch.tensor([[2, 0, 5], [1, 3, 0]], dtype=torch.int32)
    with torch.device("meta"):
        slots = qsa.qsa_paged_kv_slots((3, 2), (5, 0), block_table, 4, 8)
    assert slots.tolist() == [16, 17, 18, 19, 0, 1, 2, 3, 8, 9]


def test_an_empty_request_keeps_the_table_rows_of_the_others():
    block_table = torch.tensor([[3], [9], [4]], dtype=torch.int64)
    slots = qsa.qsa_paged_kv_slots((1, 0, 2), (0, 0, 0), block_table, 4, 4)
    assert slots.tolist() == [12, 16, 17]


def test_no_requests_have_no_slots():
    slots = qsa.qsa_paged_kv_slots((), (), torch.zeros(0, 2, dtype=torch.int32), 4, 8)
    assert slots.dtype == torch.int32
    assert slots.shape == (0,)


def test_the_slots_address_the_keys_of_every_request():
    pages, page_size, pitch_rows = 12, 8, 12
    k_cache, v_cache = record_cache(pages, page_size, heads=2, pitch_rows=pitch_rows)
    pool = qsa.qsa_paged_kv_pool(k_cache, v_cache)
    query_lens, prefix_lens = (5, 1, 9), (11, 0, 2)
    table = torch.randperm(pages)[:6].view(3, 2).int()
    slots = qsa.qsa_paged_kv_slots(
        query_lens, prefix_lens, table, pool.page_size, pool.rows_per_page
    )
    # Oracle: index the paged cache itself, request by request, token by token.
    expected = [
        k_cache[table[r, t // page_size], t % page_size]
        for r, (q, p) in enumerate(zip(query_lens, prefix_lens))
        for t in range(q + p)
    ]
    assert slots.shape == (len(expected),)
    assert torch.equal(pool.k[slots.long()], torch.stack(expected))


def test_a_block_table_that_cannot_hold_the_longest_request_is_refused():
    exact = torch.zeros(1, 2, dtype=torch.int32)
    assert qsa.qsa_paged_kv_slots((6,), (2,), exact, 4, 8).shape == (8,)
    with pytest.raises(ValueError, match="too narrow"):
        qsa.qsa_paged_kv_slots((6,), (3,), exact, 4, 8)


def test_a_block_table_must_hold_one_row_per_request():
    with pytest.raises(ValueError, match="one row per request"):
        qsa.qsa_paged_kv_slots(
            (1, 1), (0, 0), torch.zeros(1, 1, dtype=torch.int32), 4, 8
        )


def test_a_block_table_must_be_two_dimensional():
    # One entry for one request: only its rank is wrong.
    with pytest.raises(ValueError, match="one row per request"):
        qsa.qsa_paged_kv_slots((1,), (0,), torch.zeros(1, dtype=torch.int32), 4, 8)


def test_query_and_prefix_lengths_must_pair_up():
    # The table has the one row the shorter list would need: only the pairing fails.
    with pytest.raises(ValueError, match="shorter"):
        qsa.qsa_paged_kv_slots((1, 1), (0,), torch.zeros(1, 1, dtype=torch.int32), 4, 8)
