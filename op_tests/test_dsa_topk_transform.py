# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
"""Correctness tests for ``aiter.dsa_topk_transform``.

Two shapes matter to the DSA indexer and they differ in how logits rows map onto
page-table rows:

* decode -- one row per sequence, so the map is the identity and ``ptRowMap`` is
  None.
* speculative decode (verify / draft extend) -- several rows per sequence against
  a page table that still has one row each, so ``ptRowMap`` carries the map.

Every case runs against a *shuffled* page table. An identity or off-by-one
mapping cannot pass by luck that way, which is the whole point of the map being
under test.
"""

import pytest
import torch

import aiter


def _reference(logits, row_starts, row_ends, page_table, pt_row_map, topk, page_size):
    """Per-row top-k mapped through the page table, as a set per row.

    Ties are broken by arrival order in the kernel, so the emitted order is not
    reproducible and only the selected set is comparable.
    """
    page_bits = page_size.bit_length() - 1
    page_mask = page_size - 1
    num_rows = logits.shape[0]
    out = []
    for r in range(num_rows):
        start = int(row_starts[r]) if row_starts is not None else 0
        end = int(row_ends[r])
        row = logits[r, start:end]
        live = row.numel()
        k = min(topk, live)
        pos = torch.topk(row.float(), k).indices if k > 0 else row.new_empty(0).long()
        pt = page_table[int(pt_row_map[r]) if pt_row_map is not None else r]
        slots = (pt[pos >> page_bits].long() << page_bits) | (pos & page_mask)
        out.append(set(slots.tolist()))
    return out


def _run(logits, row_starts, row_ends, page_table, pt_row_map, topk, page_size=1):
    out = logits.new_full((logits.shape[0], topk), -1, dtype=torch.int32)
    aiter.dsa_topk_transform(
        logits,
        row_starts,
        row_ends,
        page_table,
        out,
        page_size,
        topk,
        ptRowMap=pt_row_map,
    )
    return out


def _compare(got, want, row_lens, topk):
    for r, expected in enumerate(want):
        emitted = got[r].tolist()
        live = min(int(row_lens[r]), topk)
        assert emitted[live:] == [-1] * (topk - live), f"row {r} not -1 padded"
        assert set(emitted[:live]) == expected, f"row {r} selected a different set"


def _make_page_table(num_seqs, pages, device, seed):
    """A deliberately shuffled logical -> physical page map, distinct per sequence.

    Sequences get disjoint physical ranges so a row that reads the wrong
    sequence's table produces slots that cannot appear in the right answer.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    rows = []
    for s in range(num_seqs):
        base = s * pages * 2
        perm = torch.randperm(pages, generator=g) + base
        rows.append(perm)
    return torch.stack(rows).to(device=device, dtype=torch.int32)


@pytest.mark.parametrize("topk", [2048])
@pytest.mark.parametrize("num_seqs", [1, 4, 32])
def test_decode_identity_rows(topk, num_seqs):
    """One row per sequence: ptRowMap is None and the kernel indexes by row."""
    device = "cuda"
    ctx = topk * 4
    torch.manual_seed(0)

    logits = torch.randn(num_seqs, ctx, dtype=torch.float32, device=device)
    row_ends = torch.full((num_seqs,), ctx, dtype=torch.int32, device=device)
    page_table = _make_page_table(num_seqs, ctx, device, seed=1)

    got = _run(logits, None, row_ends, page_table, None, topk)
    want = _reference(logits, None, row_ends, page_table, None, topk, 1)
    _compare(got, want, row_ends, topk)


@pytest.mark.parametrize("topk", [2048])
@pytest.mark.parametrize("draft_tokens", [2, 4])
@pytest.mark.parametrize("num_seqs", [1, 4, 32])
def test_spec_expanded_rows(topk, draft_tokens, num_seqs):
    """Several rows per sequence, mapped through ptRowMap.

    Mirrors EAGLE verify: each sequence contributes ``draft_tokens`` rows whose KV
    length grows by one per draft position, all sharing the sequence's page table.
    """
    device = "cuda"
    ctx = topk * 4
    torch.manual_seed(0)
    num_rows = num_seqs * draft_tokens

    logits = torch.randn(num_rows, ctx, dtype=torch.float32, device=device)
    pt_row_map = torch.arange(num_seqs, dtype=torch.int32, device=device)
    pt_row_map = pt_row_map.repeat_interleave(draft_tokens)

    # Row j of a sequence sees one more KV position than row j-1, which is what
    # makes the per-row lengths differ within a sequence.
    row_ends = torch.arange(
        ctx - draft_tokens + 1, ctx + 1, dtype=torch.int32, device=device
    ).repeat(num_seqs)

    page_table = _make_page_table(num_seqs, ctx, device, seed=1)

    got = _run(logits, None, row_ends, page_table, pt_row_map, topk)
    want = _reference(logits, None, row_ends, page_table, pt_row_map, topk, 1)
    _compare(got, want, row_ends, topk)


@pytest.mark.parametrize("topk", [2048])
def test_spec_rows_reject_wrong_table(topk):
    """The map must actually be consulted.

    Reading the page table by logits row instead of by sequence would silently
    produce a wrong-but-plausible answer, so assert the two disagree. With disjoint
    per-sequence physical ranges the sets cannot coincide.
    """
    device = "cuda"
    ctx = topk * 4
    num_seqs, draft_tokens = 4, 4
    torch.manual_seed(0)
    num_rows = num_seqs * draft_tokens

    logits = torch.randn(num_rows, ctx, dtype=torch.float32, device=device)
    row_ends = torch.full((num_rows,), ctx, dtype=torch.int32, device=device)
    pt_row_map = torch.arange(
        num_seqs, dtype=torch.int32, device=device
    ).repeat_interleave(draft_tokens)

    # num_rows table rows so the identity indexing is well-defined too.
    page_table = _make_page_table(num_rows, ctx, device, seed=1)

    mapped = _run(logits, None, row_ends, page_table, pt_row_map, topk)
    identity = _run(logits, None, row_ends, page_table, None, topk)

    # Row 0 maps to sequence 0 either way; rows 1.. do not.
    assert set(mapped[0].tolist()) == set(identity[0].tolist())
    assert set(mapped[1].tolist()) != set(identity[1].tolist())


@pytest.mark.parametrize("page_size", [1, 64])
def test_spec_rows_paged(page_size):
    """The map composes with a page size > 1."""
    device = "cuda"
    topk = 2048
    # Keep the row longer than k so selection actually runs, independent of how
    # many logical positions a page holds.
    ctx = topk * 4
    pages = ctx // page_size
    num_seqs, draft_tokens = 4, 4
    torch.manual_seed(0)
    num_rows = num_seqs * draft_tokens

    logits = torch.randn(num_rows, ctx, dtype=torch.float32, device=device)
    row_ends = torch.full((num_rows,), ctx, dtype=torch.int32, device=device)
    pt_row_map = torch.arange(
        num_seqs, dtype=torch.int32, device=device
    ).repeat_interleave(draft_tokens)
    page_table = _make_page_table(num_seqs, pages, device, seed=1)

    got = _run(logits, None, row_ends, page_table, pt_row_map, topk, page_size)
    want = _reference(logits, None, row_ends, page_table, pt_row_map, topk, page_size)
    _compare(got, want, row_ends, topk)


def test_short_rows_pad():
    """Rows with fewer live positions than k pad with -1, map or no map."""
    device = "cuda"
    topk = 2048
    ctx = topk * 2
    num_seqs, draft_tokens = 2, 3
    num_rows = num_seqs * draft_tokens
    torch.manual_seed(0)

    logits = torch.randn(num_rows, ctx, dtype=torch.float32, device=device)
    # Straddle k: shorter than, equal to, and longer than topk.
    lens = [topk // 2, topk, topk + 7] * num_seqs
    row_ends = torch.tensor(lens, dtype=torch.int32, device=device)
    pt_row_map = torch.arange(
        num_seqs, dtype=torch.int32, device=device
    ).repeat_interleave(draft_tokens)
    page_table = _make_page_table(num_seqs, ctx, device, seed=1)

    got = _run(logits, None, row_ends, page_table, pt_row_map, topk)
    want = _reference(logits, None, row_ends, page_table, pt_row_map, topk, 1)
    _compare(got, want, row_ends, topk)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
