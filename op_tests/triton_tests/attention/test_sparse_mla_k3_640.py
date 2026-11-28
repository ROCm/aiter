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


def test_kv_insert_matches_reference():
    block_size, num_blocks, T = 16, 9, 100
    x = rand_rows(T, seed=1)
    x[3] = 0
    cache = paged_cache(num_blocks, block_size, pool_stride=3)
    slots = torch.randperm(num_blocks * block_size, device=DEV)[:T]
    slots[::7] = -1
    k3_mla_640_concat_and_cache(x[:, :NOPE], x[:, None, NOPE:], cache, slots.int())
    want = torch.zeros(num_blocks * block_size, REC, dtype=torch.uint8, device=DEV)
    keep = slots >= 0
    want[slots[keep]] = ref_pack(x)[keep]
    assert torch.equal(cache.reshape(-1, REC), want)


def test_q_pack_matches_reference():
    q = rand_rows(5, 24, seed=2)
    packed = k3_mla_640_q_pack(q[..., :NOPE], q[..., NOPE:])
    assert packed.shape == (5, 24, REC) and packed.dtype == torch.uint8
    assert torch.equal(packed, ref_pack(q))


def test_gather_upconvert_matches_reference():
    block_size, num_blocks = 16, 40
    cache = paged_cache(num_blocks, block_size, 2)
    x = rand_rows(num_blocks * block_size, seed=3)
    k3_mla_640_concat_and_cache(
        x[:, :NOPE], x[:, NOPE:], cache, torch.arange(len(x), device=DEV)
    )
    seq_lens = [37, 0, 90, 16, 1]
    seq_starts = [5, 0, 33, 16, 47]
    g = torch.Generator().manual_seed(4)
    table = torch.randperm(num_blocks, generator=g).int().view(5, -1).to(DEV)
    starts = csr(seq_lens)[:-1].contiguous()
    dst = torch.full((sum(seq_lens), D_QK), float("nan"), device=DEV).bfloat16()
    k3_mla_640_cp_gather_and_upconvert(
        cache,
        dst,
        table,
        starts,
        len(seq_lens),
        torch.tensor(seq_starts, dtype=torch.int32, device=DEV),
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
    return sparse_mla_fwd(
        q, cache, indptr, indices, SCALE, kv_format="fp8_k3_mla_640", **kw
    )


@pytest.mark.parametrize("N", [1, 64, 256])
@pytest.mark.parametrize("H", [24, 12])
def test_decode(N, H):
    q, cache, indptr, indices, ref = make_case(N, H, [2048] * N, seed=N + H)
    assert_attn_close(run(q, cache, indptr, indices)[0], ref)


def test_prefill_shape():
    """One row per token, causal lists capped at top-k 2048."""
    N = 2500
    lens = [min(i + 1, 2048) for i in range(N)]
    q, cache, indptr, indices, ref = make_case(N, 24, lens, seed=21)
    assert_attn_close(run(q, cache, indptr, indices, kv_splits=1)[0], ref)


@pytest.mark.parametrize("kv_splits,invalid", [(None, 0.0), (3, 0.3)])
def test_ragged_packed_q(kv_splits, invalid):
    """Ragged lists (incl. empty and all-invalid rows), packed uint8 q, out=."""
    lens = [1, 2048, 15, 16, 17, 33, 700, 1, 31, 1999, 64, 5, 0, 40]
    q, cache, indptr, indices, ref = make_case(14, 24, lens, invalid=invalid, seed=31)
    if invalid:
        indices[indptr[3] : indptr[4]] = -1
        ref[3] = 0
    packed = k3_mla_640_q_pack(q[..., :NOPE], q[..., NOPE:])
    out = torch.empty(14, 24, NOPE, dtype=torch.bfloat16, device=DEV)
    kw = {"kv_splits": kv_splits, "has_invalid": invalid > 0}
    assert run(packed, cache, indptr, indices, out=out, **kw)[0] is out
    assert_attn_close(out, ref)
    assert torch.equal(out, run(q, cache, indptr, indices, **kw)[0])


def test_identity_rows():
    """A per-block-contiguous cache maps slot -> row by identity (no div/mod)."""
    lens = [2048, 33, 1, 640]
    q, cache, indptr, indices, ref = make_case(
        4, 24, lens, block_size=64, pool_stride=1, seed=81
    )
    assert_attn_close(run(q, cache, indptr, indices, kv_splits=4)[0], ref)


@pytest.mark.parametrize("kv_splits,has_invalid", [(1, False), (4, True)])
def test_req_topk_matches_convert(kv_splits, has_invalid):
    """req_topk is bit-identical to vLLM's convert_req_index_to_global_index
    plus the plain call: 0 for an invalid entry, or -1 with has_invalid."""
    lens, W, bs = [2048, 33, 1, 640, 0, 1500], 2048, 16
    q, cache, indptr, _, _ = make_case(len(lens), 24, lens, block_size=bs, seed=101)
    g = torch.Generator(device=DEV).manual_seed(102)
    num_blocks, max_blocks = cache.shape[0], 2048 // bs
    bt = torch.randperm(num_blocks, device=DEV, generator=g)[: 2 * max_blocks]
    bt = bt.int().view(2, max_blocks)
    req_id = torch.tensor([0, 1, 1, 0, 1, 0], dtype=torch.int32, device=DEV)
    topk = torch.full((len(lens), W), -1, dtype=torch.int32, device=DEV)
    for c, n in enumerate(lens):
        topk[c, :n] = torch.randperm(2048, device=DEV, generator=g)[:n].int()
    topk[0, 5], topk[3, 7] = -1, max_blocks * bs + 3  # invalid inside the count
    pos = topk.clamp_min(0)
    blk = pos // bs
    ok = (topk >= 0) & (blk < max_blocks)
    slot = bt[req_id[:, None].long(), blk.clamp_max(max_blocks - 1).long()] * bs
    slot = torch.where(ok, slot + pos % bs, -1 if has_invalid else 0)
    k = torch.arange(W, device=DEV)[None, :] < torch.tensor(lens, device=DEV)[:, None]
    kw = {"kv_splits": kv_splits, "has_invalid": has_invalid}
    ref = run(q, cache, indptr, slot[k].int(), **kw)[0]
    out = run(q, cache, indptr, None, req_topk=(topk, req_id, bt), **kw)[0]
    assert torch.equal(out.view(torch.int16), ref.view(torch.int16))


def test_lazy_rescale_wide_scores():
    """The row max grows by more than 2^8 across tiles, so the lazily skipped
    accumulator rescale has to kick in mid-row."""
    lens = [2048, 1500, 999, 64, 33, 2048]
    q, cache, indptr, indices, _ = make_case(6, 24, lens, seed=91)
    q = (q.float() * 16.0).bfloat16()
    kv = ref_unpack(cache.reshape(-1, REC))
    ref = ref_attn(ref_unpack(ref_pack(q)), kv, indptr, indices)
    for kv_splits in (1, 8):
        assert_attn_close(run(q, cache, indptr, indices, kv_splits=kv_splits)[0], ref)


@pytest.mark.parametrize("kv_splits", [1, 8])
def test_lse(kv_splits):
    """ln(sum exp) over the valid keys: exactly -inf (and out 0) for a row
    with none, otherwise within the QK dot's rounding, which scales with
    mag = max_k sum_i |q_i k_i| * scale."""
    lens = [1, 2048, 15, 16, 17, 33, 700, 1, 31, 1999, 64, 5, 0, 40]
    q, cache, indptr, indices, _ = make_case(14, 24, lens, invalid=0.3, seed=41)
    indices[indptr[3] : indptr[4]] = -1
    kv = ref_unpack(cache.reshape(-1, REC))
    ref, ref_lse, mag = ref_attn(
        ref_unpack(ref_pack(q)), kv, indptr, indices, with_lse=True
    )
    kw = {"kv_splits": kv_splits, "has_invalid": True}
    out, lse = run(q, cache, indptr, indices, return_lse=True, **kw)
    assert_attn_close(out, ref)
    dead = ref_lse == float("-inf")
    assert dead[3].all() and dead[12].all()
    assert torch.equal(lse == float("-inf"), dead) and (out[dead] == 0).all()
    assert ((lse - ref_lse)[~dead].abs() <= 2e-4 * mag[~dead] + 1e-5).all()
    assert torch.equal(out, run(q, cache, indptr, indices, **kw)[0])
