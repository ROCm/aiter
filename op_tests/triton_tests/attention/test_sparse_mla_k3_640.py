# SPDX-License-Identifier: MIT

"""Kimi-K3.1 fp8_k3_mla_640 on gfx1250: the 640 B record (insert, Q pack,
gather + dequant) and sparse_mla_fwd over it, against a pure-torch reference."""

import pytest
import torch

from aiter.ops.triton.attention.sparse_mla import sparse_mla_fwd
from aiter.ops.triton.quant.fused_mxfp8_quant import (
    k3_mla_640_concat_and_cache,
    k3_mla_640_cp_gather_and_upconvert,
    k3_mla_640_q_pack,
)
from aiter.ops.triton.utils._triton import arch_info

pytestmark = pytest.mark.skipif(
    not arch_info.get_arch().startswith("gfx1250"), reason="fp8_k3_mla_640 is gfx1250"
)

NOPE, ROPE, REC, GROUP = 512, 64, 640, 64
D_QK = NOPE + ROPE
SCALE = D_QK**-0.5
DEV = "cuda"


def ref_pack(x):
    """bf16 [..., 576] -> uint8 [..., 640]: 576 e4m3 | 9 E8M0 scales, each
    twice | zeros. The scale is aiter's MXFP8 rule (_mxfp8_quant_op) over 64."""
    lead = x.shape[:-1]
    g = x.float().reshape(*lead, D_QK // GROUP, GROUP)
    amax = g.abs().amax(-1).contiguous()
    p2 = ((amax.view(torch.int32) + 0x200000) & -0x800000).view(torch.float32)
    e = (torch.log2(p2).floor() - 8).clamp(-127, 127)
    rec = torch.zeros(*lead, REC, dtype=torch.uint8, device=x.device)
    data = (g * torch.exp2(-e)[..., None]).to(torch.float8_e4m3fn)
    rec[..., :D_QK] = data.view(torch.uint8).reshape(*lead, D_QK)
    rec[..., D_QK : D_QK + 18] = (e + 127).to(torch.uint8).repeat_interleave(2, -1)
    return rec


def ref_unpack(rec):
    """uint8 [..., 640] -> f32 [..., 576], exactly what the record holds."""
    lead = rec.shape[:-1]
    e = rec[..., D_QK : D_QK + 18 : 2].float() - 127
    g = rec[..., :D_QK].contiguous().view(torch.float8_e4m3fn).float()
    g = g.reshape(*lead, D_QK // GROUP, GROUP) * torch.exp2(e)[..., None]
    return g.reshape(*lead, D_QK)


def ref_attn(q, kv, indptr, indices, scale=SCALE, chunk=32, with_lse=False):
    """q [N, H, 576] and kv [rows, 576] f32 (dequantized), CSR rows with -1 =
    skip -> [N, H, 512] f32: softmax(q.k * scale) over the row's keys, V = nope.
    with_lse adds [N, H] f32 ln(sum exp) over the valid keys, -inf with none,
    and the scores' rounding scale, max over keys of sum_i |q_i k_i| * scale."""
    N = q.shape[0]
    lens = (indptr[1:] - indptr[:-1]).long()
    out = torch.zeros(N, q.shape[1], NOPE, device=q.device)
    lse = torch.zeros(N, q.shape[1], device=q.device)
    mag = torch.zeros(N, q.shape[1], device=q.device)
    for a in range(0, N, chunk):
        b = min(N, a + chunk)
        kmax = max(int(lens[a:b].max()), 1)
        pos = torch.arange(kmax, device=q.device)[None, :]
        inside = pos < lens[a:b, None]
        at = (indptr[a:b, None].long() + pos).clamp_max(len(indices) - 1)
        idx = indices.long()[at]
        ok = inside & (idx >= 0)
        k = kv[idx.clamp_min(0)]
        s = torch.einsum("nhd,nkd->nhk", q[a:b], k) * scale
        s = s.masked_fill(~ok[:, None, :], float("-inf"))
        p = torch.softmax(s, -1)
        out[a:b] = torch.einsum("nhk,nkd->nhd", p.nan_to_num(0.0), k[..., :NOPE])
        if with_lse:
            lse[a:b] = torch.logsumexp(s, -1)
            sa = torch.einsum("nhd,nkd->nhk", q[a:b].abs(), k.abs()) * scale
            mag[a:b] = sa.masked_fill(~ok[:, None, :], 0.0).amax(-1)
    return (out, lse, mag) if with_lse else out


def rand_rows(*shape, seed=0):
    """bf16 rows whose 64-element groups span ~2^-4 .. 2^4, so scales differ."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    x = torch.randn(*shape, D_QK, device=DEV, generator=g)
    mag = torch.randint(-4, 5, (*shape, D_QK // GROUP), device=DEV, generator=g)
    return (x * torch.exp2(mag.float()).repeat_interleave(GROUP, -1) * 0.5).bfloat16()


def paged_cache(num_blocks, block_size, pool_stride=1):
    """Zeroed [num_blocks, block_size, 640]; pool_stride > 1 makes it a strided
    view into a wider block row, as a per-layer view of a packed pool is."""
    pool = torch.zeros(
        num_blocks, pool_stride * block_size, REC, dtype=torch.uint8, device=DEV
    )
    return pool[:, :block_size]


def csr(lens):
    indptr = torch.zeros(len(lens) + 1, dtype=torch.int32, device=DEV)
    indptr[1:] = torch.as_tensor(lens, device=DEV).cumsum(0)
    return indptr


def rand_csr(lens, rows, seed=0, invalid=0.0):
    g = torch.Generator(device=DEV).manual_seed(seed)
    indptr = csr(lens)
    idx = torch.randint(0, rows, (int(indptr[-1]),), device=DEV, generator=g)
    if invalid:
        drop = torch.rand(idx.shape, device=DEV, generator=g) < invalid
        idx = idx.masked_fill(drop, -1)
    return indptr, idx.int()


def assert_attn_close(out, ref, tol=1e-2):
    """The record is read exactly; P and the output are bf16, which costs ~4e-3
    of the row's largest output (measured)."""
    assert out.dtype == torch.bfloat16 and out.shape == ref.shape
    assert torch.isfinite(out.float()).all()
    err = (out.float() - ref).abs().amax(-1)
    lim = tol * ref.abs().amax(-1).clamp_min(1e-3)
    assert (err <= lim).all(), f"max err/limit {(err / lim).max():.3f}"


@pytest.mark.parametrize("pool_stride", [1, 3])
def test_kv_insert_matches_reference(pool_stride):
    block_size, num_blocks, T = 16, 9, 100
    x = rand_rows(T, seed=1)
    x[3] = 0
    cache = paged_cache(num_blocks, block_size, pool_stride)
    slots = torch.randperm(num_blocks * block_size, device=DEV)[:T]
    slots[::7] = -1
    k3_mla_640_concat_and_cache(
        x[:, :NOPE], x[:, None, NOPE:], cache, slots.int() if pool_stride > 1 else slots
    )
    want = torch.zeros(num_blocks * block_size, REC, dtype=torch.uint8, device=DEV)
    keep = slots >= 0
    want[slots[keep]] = ref_pack(x)[keep]
    got = cache.reshape(-1, REC)
    assert torch.equal(got, want)
    back = ref_unpack(got[slots[keep]])
    amax = x[keep].float().reshape(-1, 9, GROUP).abs().amax(-1, keepdim=True)
    err = (back - x[keep].float()).reshape(-1, 9, GROUP).abs()
    assert (err <= amax / 16 + 1e-30).all()


def test_q_pack_matches_reference():
    q = rand_rows(5, 24, seed=2)
    packed = k3_mla_640_q_pack(q[..., :NOPE], q[..., NOPE:])
    assert packed.shape == (5, 24, REC) and packed.dtype == torch.uint8
    assert torch.equal(packed, ref_pack(q))


@pytest.mark.parametrize("use_seq_starts", [False, True])
def test_gather_upconvert_matches_reference(use_seq_starts):
    block_size, num_blocks = 16, 40
    cache = paged_cache(num_blocks, block_size, 2)
    x = rand_rows(num_blocks * block_size, seed=3)
    k3_mla_640_concat_and_cache(
        x[:, :NOPE], x[:, NOPE:], cache, torch.arange(len(x), device=DEV)
    )
    seq_lens = [37, 0, 90, 16, 1]
    seq_starts = [5, 0, 33, 16, 47] if use_seq_starts else [0] * len(seq_lens)
    g = torch.Generator().manual_seed(4)
    table = torch.randperm(num_blocks, generator=g).int().reshape(len(seq_lens), -1)
    table = table.to(DEV)
    starts = csr(seq_lens)[:-1].contiguous()
    total = sum(seq_lens)
    dst = torch.full((total, D_QK), float("nan"), dtype=torch.bfloat16, device=DEV)
    k3_mla_640_cp_gather_and_upconvert(
        cache,
        dst,
        table,
        starts,
        len(seq_lens),
        (
            torch.tensor(seq_starts, dtype=torch.int32, device=DEV)
            if use_seq_starts
            else None
        ),
    )
    rows = []
    for r, (n, s0) in enumerate(zip(seq_lens, seq_starts)):
        p = torch.arange(s0, s0 + n, device=DEV)
        rows.append(table[r, p // block_size].long() * block_size + p % block_size)
    want = ref_unpack(cache.reshape(-1, REC)[torch.cat(rows)])
    assert torch.equal(dst.float(), want)


def make_case(N, H, lens, rows=4096, block_size=16, pool_stride=2, invalid=0.0, seed=0):
    q = rand_rows(N, H, seed=seed)
    kv = rand_rows(rows, seed=seed + 1)
    cache = paged_cache(rows // block_size, block_size, pool_stride)
    k3_mla_640_concat_and_cache(
        kv[:, :NOPE], kv[:, NOPE:], cache, torch.arange(rows, device=DEV)
    )
    indptr, indices = rand_csr(lens, rows, seed=seed + 2, invalid=invalid)
    ref = ref_attn(
        ref_unpack(ref_pack(q)), ref_unpack(cache.reshape(-1, REC)), indptr, indices
    )
    return q, cache, indptr, indices, ref


def run(q, cache, indptr, indices, **kw):
    out, lse = sparse_mla_fwd(
        q, cache, indptr, indices, SCALE, kv_format="fp8_k3_mla_640", **kw
    )
    assert lse is None
    return out


@pytest.mark.parametrize("N", [1, 4, 16, 64, 256])
@pytest.mark.parametrize("H", [24, 12, 48, 96, 8])
def test_decode(N, H):
    """Default heuristics: 1-32 splits, BLOCK_H 16 / BLOCK_K 16 at <= 16 heads."""
    q, cache, indptr, indices, ref = make_case(N, H, [2048] * N, seed=N + H)
    assert_attn_close(run(q, cache, indptr, indices), ref)


@pytest.mark.parametrize("kv_splits", [1, 2, 8])
def test_decode_splits_and_packed_q(kv_splits):
    q, cache, indptr, indices, ref = make_case(4, 24, [2048] * 4, seed=11)
    packed = k3_mla_640_q_pack(q[..., :NOPE], q[..., NOPE:])
    out = torch.empty(4, 24, NOPE, dtype=torch.bfloat16, device=DEV)
    res = run(packed, cache, indptr, indices, kv_splits=kv_splits, out=out)
    assert res is out
    assert_attn_close(out, ref)
    assert torch.equal(out, run(q, cache, indptr, indices, kv_splits=kv_splits))


def test_prefill_shape():
    """A prefill chunk: one row per token, causal lists capped at top-k 2048,
    single pass."""
    N, H = 2500, 24
    lens = [min(i + 1, 2048) for i in range(N)]
    q, cache, indptr, indices, ref = make_case(N, H, lens, seed=21)
    assert_attn_close(run(q, cache, indptr, indices, kv_splits=1), ref)


@pytest.mark.parametrize("kv_splits", [None, 1, 4, 32])
@pytest.mark.parametrize("invalid", [0.0, 0.3])
@pytest.mark.parametrize("H", [24, 12])
def test_ragged(kv_splits, invalid, H):
    lens = [1, 2048, 15, 16, 17, 33, 700, 1, 31, 1999, 64, 5, 0, 40]
    q, cache, indptr, indices, ref = make_case(
        len(lens), H, lens, invalid=invalid, seed=31
    )
    if invalid:
        indices[indptr[3] : indptr[4]] = -1
        ref[3] = 0
    out = run(q, cache, indptr, indices, kv_splits=kv_splits, has_invalid=invalid > 0)
    assert_attn_close(out, ref)


def test_flat_record_grid():
    """[rows, 640] indexed by row is the paged cache with block size 1."""
    q, cache, indptr, indices, ref = make_case(4, 24, [300, 2048, 7, 64], pool_stride=1)
    assert_attn_close(run(q, cache.reshape(-1, REC), indptr, indices), ref)


def test_rejects_unsupported():
    q, cache, indptr, indices, _ = make_case(1, 24, [16])
    sink = torch.zeros(24, device=DEV)
    with pytest.raises(ValueError, match="return_lse"):
        run_lse(q, cache, indptr, indices, kv_splits=2, skip_reduce=True)
    with pytest.raises(ValueError, match="return_lse"):
        run_lse(q, cache, indptr, indices, attn_sink=sink)
    with pytest.raises(ValueError):
        sparse_mla_fwd(q, cache, indptr, indices, SCALE, kv_format="nope")
    with pytest.raises(ValueError, match="keys_per_row"):
        sparse_mla_fwd(q, cache, indptr, indices, SCALE, keys_per_row=16)


def run_lse(q, cache, indptr, indices, **kw):
    out, lse = sparse_mla_fwd(
        q,
        cache,
        indptr,
        indices,
        SCALE,
        kv_format="fp8_k3_mla_640",
        return_lse=True,
        **kw,
    )
    assert lse.dtype == torch.float32 and lse.shape == q.shape[:2]
    return out, lse


def ref_case(q, cache, indptr, indices):
    """(out, lse, mag) of the torch reference over what the records hold."""
    kv = ref_unpack(cache.reshape(-1, REC))
    return ref_attn(ref_unpack(ref_pack(q)), kv, indptr, indices, with_lse=True)


def assert_lse_close(lse, ref, mag, tol=2e-4):
    """Exactly -inf where the row has no valid key (never the -1e30 no-sink
    stand-in, never NaN). Elsewhere the error is the QK dot's, which scales
    with mag = sum_i |q_i k_i| * scale: up to 6e-5 of it (measured; 1e-2
    absolute on rand_rows' 2^-4 .. 2^4 groups, 2e-4 on unit normal rows). mag
    may be a plain float bound."""
    dead = ref == float("-inf")
    assert torch.equal(lse == float("-inf"), dead)
    assert torch.isfinite(lse[~dead]).all()
    err = (lse - ref).abs().nan_to_num(0.0)
    lim = tol * mag + 1e-5
    assert (err <= lim).all(), f"max lse err / limit {(err / lim).max():.3f}"


@pytest.mark.parametrize("kv_splits", [1, 2, 8, 32])
@pytest.mark.parametrize("H", [24, 96])
@pytest.mark.parametrize("invalid", [0.0, 0.3])
def test_lse(kv_splits, H, invalid):
    lens = [1, 2048, 15, 16, 17, 33, 700, 1, 31, 1999, 64, 5, 0, 40]
    q, cache, indptr, indices, _ = make_case(
        len(lens), H, lens, invalid=invalid, seed=41
    )
    if invalid:
        indices[indptr[3] : indptr[4]] = -1
    ref, ref_lse, mag = ref_case(q, cache, indptr, indices)
    kw = dict(kv_splits=kv_splits, has_invalid=invalid > 0)
    out, lse = run_lse(q, cache, indptr, indices, **kw)
    assert_lse_close(lse, ref_lse, mag)
    assert_attn_close(out, ref)
    row = torch.repeat_interleave(
        torch.arange(len(lens), device=DEV), torch.tensor(lens, device=DEV)
    )
    alive = torch.bincount(row[indices >= 0], minlength=len(lens)) > 0
    assert not alive[12] and (not invalid or not alive[3])
    dead = lse == float("-inf")
    assert torch.equal(dead, ~alive[:, None].expand_as(dead))
    assert (out[dead] == 0).all()
    assert torch.equal(out, run(q, cache, indptr, indices, **kw))


def split_shards(indptr, indices, seed, empty_b=(), empty_both=()):
    """Each row's keys dealt at random into two disjoint compacted CSR lists.
    Rows in empty_b keep everything in shard 0; rows in empty_both lose it all."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    row = torch.repeat_interleave(
        torch.arange(len(indptr) - 1, device=DEV), (indptr[1:] - indptr[:-1]).long()
    )
    side = torch.rand(indices.shape, device=DEV, generator=g) < 0.4
    side &= ~torch.isin(row, torch.tensor(empty_b, device=DEV))
    keep = ~torch.isin(row, torch.tensor(empty_both, device=DEV))
    shards = []
    for s in (False, True):
        pick = keep & (side == s)
        lens = torch.bincount(row[pick], minlength=len(indptr) - 1)
        shards.append((csr(lens), indices[pick].contiguous()))
    return shards


@pytest.mark.parametrize("kv_splits", [None, 1, 8])
@pytest.mark.parametrize("H", [24, 96])
@pytest.mark.parametrize("invalid", [0.0, 0.3])
def test_lse_merges_two_shards(kv_splits, H, invalid):
    """What DCP does: every rank attends its share of a row's keys, and the
    ranks merge by out = sum_i softmax_i(lse) * out_i."""
    lens = [2048, 1, 700, 2048, 33, 2, 512, 1200]
    q, cache, indptr, indices, _ = make_case(
        len(lens), H, lens, invalid=invalid, seed=51
    )
    shards = split_shards(indptr, indices, seed=52, empty_b=(1, 4), empty_both=(5,))
    kw = dict(has_invalid=invalid > 0)
    parts = [
        run_lse(q, cache, ip, ix, kv_splits=kv_splits, keys_per_row=1024, **kw)
        for ip, ix in shards
    ]
    lses = torch.stack([lse for _, lse in parts])
    assert (lses[1, [1, 4]] == float("-inf")).all()
    assert (lses[:, 5] == float("-inf")).all()
    merged_lse = torch.logsumexp(lses, 0)
    w = torch.exp(lses - merged_lse).nan_to_num(0.0)
    merged = sum(w[i][..., None] * parts[i][0].float() for i in range(2))
    full_lens = torch.tensor(lens, device=DEV)
    full_lens[5] = 0
    keep = torch.ones_like(indices, dtype=torch.bool)
    keep[indptr[5] : indptr[6]] = False
    ip, ix = csr(full_lens), indices[keep].contiguous()
    out, lse = run_lse(q, cache, ip, ix, kv_splits=kv_splits, **kw)
    assert_lse_close(merged_lse, lse, 1.0, tol=1e-5)
    assert_attn_close(merged.bfloat16(), out.float(), tol=1.5e-2)
    ref, ref_lse, mag = ref_case(q, cache, ip, ix)
    assert_lse_close(merged_lse, ref_lse, mag)
    assert_attn_close(merged.bfloat16(), ref)


def test_keys_per_row_hint(monkeypatch):
    """A DCP rank's list is allocated for the full top-k but holds ~top_k / dcp
    keys per row; the hint replaces len(kv_indices) / rows in the split choice."""
    import aiter.ops.triton.attention.sparse_mla as mod

    seen = []
    real = mod._k3_splits
    monkeypatch.setattr(
        mod, "_k3_splits", lambda *a: seen.append(a) or real(*a), raising=True
    )
    N, H, held = 4, 96, 512
    q, cache, indptr, indices, ref = make_case(N, H, [held] * N, seed=61)
    alloc = torch.full((N * 2048,), -1, dtype=torch.int32, device=DEV)
    alloc[: len(indices)] = indices
    outs = [
        run(q, cache, indptr, alloc, **kw)
        for kw in ({}, {"keys_per_row": held}, {"keys_per_row": 0})
    ]
    assert seen == [(N * 3, 64, 32), (N * 3, 16, 32), (N * 3, 0, 32)]
    assert real(N * 3, 16, 32) < real(N * 3, 64, 32)
    for out in outs:
        assert_attn_close(out, ref)


def q_layouts(q):
    """The same [N, H, 576] values as vLLM may hand them over after an
    all-gather over heads."""
    N, H, _ = q.shape
    wide = torch.zeros(N, H + 3, D_QK + 40, dtype=q.dtype, device=DEV)
    wide[:, 2 : 2 + H, 8 : 8 + D_QK] = q
    stepped = torch.zeros(2 * N, H, D_QK, dtype=q.dtype, device=DEV)
    stepped[::2] = q
    cols = torch.zeros(N, H, 2 * D_QK, dtype=q.dtype, device=DEV)
    cols[..., ::2] = q
    return {
        "contiguous": q,
        "heads_first": q.transpose(0, 1).contiguous().transpose(0, 1),
        "window": wide[:, 2 : 2 + H, 8 : 8 + D_QK],
        "row_step": stepped[::2],
        "col_step": cols[..., ::2],
        "rank_concat": torch.cat(q.chunk(4, dim=1), dim=1),
    }


@pytest.mark.parametrize(
    "layout",
    ["contiguous", "heads_first", "window", "row_step", "col_step", "rank_concat"],
)
def test_q_layouts(layout):
    """Non-contiguous / strided bf16 q: the pack reads the right elements, as
    one tensor through sparse_mla_fwd and as its two slices through the pack."""
    N, H = 5, 96
    q, cache, indptr, indices, ref = make_case(N, H, [300, 2048, 7, 64, 0], seed=71)
    qv = q_layouts(q)[layout]
    assert torch.equal(qv, q)
    if layout not in ("contiguous", "rank_concat"):
        assert not qv.is_contiguous()
    want = ref_pack(q)
    assert torch.equal(k3_mla_640_q_pack(qv[..., :NOPE], qv[..., NOPE:]), want)
    dst = torch.zeros(N, H, REC, dtype=torch.uint8, device=DEV)
    assert k3_mla_640_q_pack(qv[..., :NOPE], qv[..., NOPE:], out=dst) is dst
    assert torch.equal(dst, want)
    out, lse = run_lse(qv, cache, indptr, indices)
    base, base_lse = run_lse(want, cache, indptr, indices)
    assert torch.equal(out, base) and torch.equal(lse, base_lse)
    assert_attn_close(out, ref)


@pytest.mark.parametrize("keys", [512, 2048])
def test_dcp_rank_shape(keys):
    """One DCP rank at TP1: all 96 heads, 2048 rows, its share of the top-2048."""
    N, H = 2048, 96
    q, cache, indptr, indices, _ = make_case(N, H, [keys] * N, rows=1 << 15, seed=81)
    ref, ref_lse, mag = ref_case(q, cache, indptr, indices)
    out, lse = run_lse(q, cache, indptr, indices, keys_per_row=keys)
    assert_lse_close(lse, ref_lse, mag)
    assert_attn_close(out, ref)
    assert torch.equal(out, run(q, cache, indptr, indices, keys_per_row=keys))
