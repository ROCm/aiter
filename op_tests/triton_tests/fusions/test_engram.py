# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

from aiter.ops.triton.fusions.engram import engram_embedding_lookup, engram_gate_apply

# DeepSeek-V4.1-Flash Engram: 3 n-gram orders x 8 hash heads, 256-wide rows,
# one e8m0 scale per 32 columns.
H = 24
D = 256
SCALE_BLOCK = 32
ROWS = 200_000
SHAPES = [(1, 1), (1, 128), (4, 512), (2, 4096)]


def make_table(rows, device, seed=0):
    """A synthetic shard of the fp8 table plus its e8m0 scales."""
    gen = torch.Generator(device=device).manual_seed(seed)
    values = torch.randn(rows, D, generator=gen, device=device)
    table = values.to(torch.float8_e4m3fn)
    # exponents around 1.0, well inside the normal range of e8m0
    raw = torch.randint(
        118, 137, (rows, D // SCALE_BLOCK), generator=gen, device=device
    ).to(torch.uint8)
    return table, raw.view(torch.float8_e8m0fnu)


def ref_lookup(hash_ids, table, scale, row_offset=0, num_rows=None):
    """ParallelEngramEmbedding.forward, one rank, without the all-reduce."""
    if num_rows is None:
        num_rows = table.shape[0]
    B, L, cols = hash_ids.shape
    local = hash_ids - row_offset
    bad = (local < 0) | (local >= num_rows)
    local = local.masked_fill(bad, 0).flatten()
    values = table[local].float().unflatten(-1, (-1, SCALE_BLOCK))
    values = (values * scale[local].float().unsqueeze(-1)).flatten(-2)
    values = values.view(B, L, cols, -1).masked_fill(bad.unsqueeze(-1), 0)
    return values.flatten(-2).to(torch.bfloat16)


def ref_gate(h, key, value, weight, token_mask, eps=1e-20, clamp_value=1e-6):
    """Engram.forward from `dim`-normalized dot product onward."""
    dim = h.shape[-1]
    h, key = h.float(), key.float()
    rstd = torch.rsqrt(h.square().mean(-1) + eps) * torch.rsqrt(
        key.square().mean(-1) + eps
    )
    dot = (h * weight.float() * key).sum(-1) * rstd * dim**-0.5
    gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(clamp_value).sqrt(), dot))
    if token_mask is not None:
        gate = gate.masked_fill(~token_mask.unsqueeze(-1), 0)
    return (h + gate.unsqueeze(-1) * value.float().unsqueeze(-2)).to(torch.bfloat16)


def random_ids(B, L, rows, device, seed=1, cols=H):
    gen = torch.Generator(device=device).manual_seed(seed)
    return torch.randint(
        0, rows, (B, L, cols), generator=gen, device=device, dtype=torch.int64
    )


@pytest.mark.parametrize("B,L", SHAPES)
def test_lookup_matches_reference(B, L):
    dev = "cuda"
    table, scale = make_table(ROWS, dev)
    hash_ids = random_ids(B, L, ROWS, dev)
    got = engram_embedding_lookup(hash_ids, table, scale)
    want = ref_lookup(hash_ids, table, scale)
    assert got.shape == (B, L, H * D) and got.dtype == torch.bfloat16
    # one fp32 multiply then one round to bf16 on both sides
    assert torch.equal(got, want)


def test_lookup_negative_out_of_window_and_duplicate_ids():
    dev = "cuda"
    rows = 4096
    table, scale = make_table(rows, dev)
    window_start, window_rows = 1000, 512
    hash_ids = torch.tensor(
        [
            [
                [-1] * 4
                + [-(2**40)] * 2
                + [0, 999]  # negative and below window
                + [1000, 1000, 1000, 1511]  # in window, duplicated
                + [1512, 4095, 2**31, 2**40]  # above window
                + [1200] * 8  # a run of duplicates
            ]
        ],
        dtype=torch.int64,
        device=dev,
    )
    assert hash_ids.shape == (1, 1, H)
    got = engram_embedding_lookup(
        hash_ids, table, scale, row_offset=window_start, num_rows=window_rows
    )
    want = ref_lookup(hash_ids, table, scale, window_start, window_rows)
    assert torch.equal(got, want)

    per_head = got.view(1, 1, H, D)
    owned = (hash_ids >= window_start) & (hash_ids < window_start + window_rows)
    assert torch.all(per_head[~owned] == 0)
    assert torch.any(per_head[owned] != 0)


def test_lookup_int64_row_offset():
    """A window that starts past int32, as a late shard of a 384M-row table."""
    dev = "cuda"
    rows = 8192
    table, scale = make_table(rows, dev)
    offset = 2**32 + 12345
    gen = torch.Generator(device=dev).manual_seed(3)
    hash_ids = torch.randint(
        offset - 16, offset + rows + 16, (2, 64, H), generator=gen, device=dev
    )
    got = engram_embedding_lookup(
        hash_ids, table, scale, row_offset=offset, num_rows=rows
    )
    want = ref_lookup(hash_ids, table, scale, offset, rows)
    assert torch.equal(got, want)


@pytest.mark.parametrize("shards", [4])
def test_lookup_sharded_sum_equals_unsharded(shards):
    dev = "cuda"
    rows = 4 * 8192
    table, scale = make_table(rows, dev)
    hash_ids = random_ids(3, 129, rows, dev, seed=7)

    unsharded = engram_embedding_lookup(hash_ids, table, scale)

    part = (rows + shards - 1) // shards
    total = torch.zeros_like(unsharded, dtype=torch.float32)
    for rank in range(shards):
        start = rank * part
        stop = min(start + part, rows)
        piece = engram_embedding_lookup(
            hash_ids,
            table[start:stop].contiguous(),
            scale[start:stop].contiguous(),
            row_offset=start,
            num_rows=stop - start,
        )
        total += piece.float()
    # every id lands in exactly one shard, so the all-reduce is a copy
    assert torch.equal(total.to(torch.bfloat16), unsharded)


def test_lookup_writes_into_provided_out():
    dev = "cuda"
    table, scale = make_table(1024, dev)
    hash_ids = random_ids(2, 8, 1024, dev, seed=11)
    out = torch.full((2, 8, H * D), float("nan"), dtype=torch.bfloat16, device=dev)
    got = engram_embedding_lookup(hash_ids, table, scale, out=out)
    assert got.data_ptr() == out.data_ptr()
    assert torch.equal(out, ref_lookup(hash_ids, table, scale))


@pytest.mark.parametrize("B,L", SHAPES)
@pytest.mark.parametrize("masked", [False, True])
def test_gate_matches_reference(B, L, masked):
    dev = "cuda"
    C, dim = 4, 5120
    gen = torch.Generator(device=dev).manual_seed(B * 1000 + L)
    h = torch.randn(B, L, C, dim, generator=gen, device=dev, dtype=torch.bfloat16)
    key = torch.randn(B, L, C, dim, generator=gen, device=dev)
    value = torch.randn(B, L, dim, generator=gen, device=dev, dtype=torch.bfloat16)
    weight = torch.randn(C, dim, generator=gen, device=dev)
    token_mask = None
    if masked:
        token_mask = torch.rand(B, L, generator=gen, device=dev) > 0.5

    got = engram_gate_apply(h, key, value, weight, token_mask)
    want = ref_gate(h, key, value, weight, token_mask)
    assert got.shape == h.shape and got.dtype == torch.bfloat16
    torch.testing.assert_close(got, want, atol=2e-2, rtol=2e-2)


def test_gate_all_masked_is_identity():
    dev = "cuda"
    B, L, C, dim = 2, 16, 4, 512
    gen = torch.Generator(device=dev).manual_seed(5)
    h = torch.randn(B, L, C, dim, generator=gen, device=dev, dtype=torch.bfloat16)
    key = torch.randn(B, L, C, dim, generator=gen, device=dev)
    value = torch.randn(B, L, dim, generator=gen, device=dev, dtype=torch.bfloat16)
    weight = torch.randn(C, dim, generator=gen, device=dev)
    token_mask = torch.zeros(B, L, dtype=torch.bool, device=dev)
    got = engram_gate_apply(h, key, value, weight, token_mask)
    assert torch.equal(got, h)


@pytest.mark.parametrize("dim", [256, 5120])
@pytest.mark.parametrize("C", [1, 4])
def test_gate_shapes(C, dim):
    dev = "cuda"
    B, L = 2, 33
    gen = torch.Generator(device=dev).manual_seed(dim + C)
    h = torch.randn(B, L, C, dim, generator=gen, device=dev, dtype=torch.bfloat16)
    key = torch.randn(B, L, C, dim, generator=gen, device=dev)
    value = torch.randn(B, L, dim, generator=gen, device=dev, dtype=torch.bfloat16)
    weight = torch.randn(C, dim, generator=gen, device=dev)
    got = engram_gate_apply(h, key, value, weight)
    torch.testing.assert_close(
        got, ref_gate(h, key, value, weight, None), atol=2e-2, rtol=2e-2
    )


def test_gate_zero_h_takes_the_positive_branch():
    """abs(dot) is clamped, so an all-zero stream still gets sigmoid(+sqrt)."""
    dev = "cuda"
    B, L, C, dim = 1, 4, 4, 256
    h = torch.zeros(B, L, C, dim, dtype=torch.bfloat16, device=dev)
    key = torch.randn(B, L, C, dim, device=dev)
    value = torch.randn(B, L, dim, dtype=torch.bfloat16, device=dev)
    weight = torch.randn(C, dim, device=dev)
    got = engram_gate_apply(h, key, value, weight)
    torch.testing.assert_close(
        got, ref_gate(h, key, value, weight, None), atol=2e-2, rtol=2e-2
    )


def test_lookup_and_gate_chain():
    """The pair as the model uses them: lookup -> wkv -> gate."""
    dev = "cuda"
    B, L, C, dim = 2, 64, 4, 512
    table, scale = make_table(65536, dev)
    hash_ids = random_ids(B, L, 65536, dev, seed=17)
    gathered = engram_embedding_lookup(hash_ids, table, scale)
    assert gathered.shape == (B, L, H * D)

    gen = torch.Generator(device=dev).manual_seed(19)
    wkv = torch.randn(
        H * D, dim * (C + 1), generator=gen, device=dev, dtype=torch.bfloat16
    )
    kv = gathered @ (wkv / (H * D) ** 0.5)
    key, value = kv.split([C * dim, dim], dim=-1)
    key = key.float().unflatten(-1, (C, dim)).contiguous()
    weight = torch.randn(C, dim, generator=gen, device=dev)
    h = torch.randn(B, L, C, dim, generator=gen, device=dev, dtype=torch.bfloat16)

    got = engram_gate_apply(h, key, value.contiguous(), weight)
    want = ref_gate(h, key, value.contiguous(), weight, None)
    torch.testing.assert_close(got, want, atol=2e-2, rtol=2e-2)
