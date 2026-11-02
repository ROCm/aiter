# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Plan checks for the gfx950 XCD-column MLA decode planner (kn_get_mla_metadata_v1_2_xcd).

get_mla_metadata_v1 (fast_mode, auto split) takes that planner for native heads at
qlen > 1 when qlen divides the CUs of an XCD, e.g. 96 or 128 heads, fp8 or bf16, at
qlen 2 / 4 / 8. For uniform and lognormal KV lengths, with and without round-robin
context parallelism:
  * the works of every (request, query token) tile its KV range exactly, and the reduce
    maps describe the split ones;
  * the tiles of a KV fragment run on workgroups of one XCD, at the same place in each
    workgroup's work list;
  * the busiest workgroup's planned load is close to the mean;
and a decode with the plan matches a torch reference.
"""

import math
import sys

import numpy as np
import pytest
import torch

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.attention import get_mla_decode_shape_support

KV_LORA_RANK = 512
DIM = KV_LORA_RANK + 64
KV_GRAN = 16
OVERHEAD = 16  # fixed_over_head_num_blocks at page_size 1
NUM_XCD = 8
CP_WORLD = 8
MAX_BUSIEST = 1.15
PLAN_ORDER = (
    "work_meta_data",
    "work_info_set",
    "work_indptr",
    "reduce_indptr",
    "reduce_final_map",
    "reduce_partial_map",
)
INFO_ORDER = (
    "work_meta_data",
    "work_indptr",
    "work_info_set",
    "reduce_indptr",
    "reduce_final_map",
    "reduce_partial_map",
)


def _skip_unless_xcd_planner(nhead, qlen, dtype):
    if not torch.cuda.is_available() or get_gfx() != "gfx950":
        pytest.skip("the XCD planner is gfx950 only")
    dt = getattr(dtypes, dtype)
    s = get_mla_decode_shape_support(nhead, qlen, dt, dt)
    if not (s.natively_supported and s.native and s.kernel_num_heads == nhead):
        pytest.skip(f"{nhead} heads {dtype} at qlen {qlen} are not native here")


def _global_ctxs(bs, dist, seed):
    rng = np.random.default_rng(seed)
    if dist == "uniform":
        return [82000] * bs
    # median 66k, p90 ~155k: the context mix of a long-context agentic serving run
    return [
        int(min(max(66000 * math.exp(0.67 * rng.standard_normal()), 1000), 400000))
        for _ in range(bs)
    ]


def _local_lens(ctxs, cp_rr, rank=0):
    if cp_rr:
        return [(c - rank + CP_WORLD - 1) // CP_WORLD for c in ctxs]
    return [c // CP_WORLD for c in ctxs]


def _indptr(lens):
    out = np.zeros(len(lens) + 1, dtype=np.int32)
    out[1:] = np.cumsum(lens)
    return torch.from_numpy(out).cuda()


def _plan(lens, qlen, cp_rr, nhead, dtype):
    bs = len(lens)
    dt = getattr(dtypes, dtype)
    info = aiter.get_mla_metadata_info_v1(
        bs, qlen, nhead, dt, dt, is_sparse=False, fast_mode=True
    )
    meta = {
        name: torch.zeros(shape, dtype=t, device="cuda")
        for name, (shape, t) in zip(INFO_ORDER, info)
    }
    meta["work_info_set"].fill_(-7)
    qo_indptr = torch.arange(bs + 1, dtype=torch.int32, device="cuda") * qlen
    aiter.get_mla_metadata_v1(
        qo_indptr,
        _indptr(lens),
        torch.ones(bs, dtype=torch.int32, device="cuda"),
        nhead,
        1,
        True,
        *(meta[k] for k in PLAN_ORDER),
        page_size=1,
        kv_granularity=KV_GRAN,
        max_seqlen_qo=qlen,
        uni_seqlen_qo=qlen,
        fast_mode=True,
        max_split_per_batch=-1,
        dtype_q=dt,
        dtype_kv=dt,
        is_cp_round_robin=cp_rr,
    )
    torch.cuda.synchronize()
    return meta


def _check_plan(meta, lens, qlen, cp_rr):
    """Asserts the plan invariants. Returns the busiest workgroup's planned load over the
    mean, and the mean number of KV blocks per workgroup."""
    num_cu = torch.cuda.get_device_properties(0).multi_processor_count
    wip = meta["work_indptr"].cpu().numpy()[: num_cu + 1].astype(np.int64)
    nw = int(wip[-1])
    assert wip[0] == 0 and (np.diff(wip) >= 0).all(), wip
    assert nw <= meta["work_info_set"].shape[0]
    works = meta["work_info_set"].cpu().numpy()[:nw]
    assert (works[:, :6] != -7).all(), "a work slot was left unwritten"
    ri = meta["reduce_indptr"].cpu().numpy().astype(np.int64)
    rfm = meta["reduce_final_map"].cpu().numpy()
    rpm = meta["reduce_partial_map"].cpu().numpy()
    wg_of = np.searchsorted(wip, np.arange(nw), side="right") - 1
    pos_of = np.arange(nw) - wip[wg_of]

    bs = len(lens)
    kvp = np.concatenate([[0], np.cumsum(lens)])
    tiles = {}
    for w, (b, part, q0, q1, k0, k1) in enumerate(works[:, :6].tolist()):
        assert 0 <= b < bs and q1 - q0 == 1, works[w]
        tiles.setdefault(q0, []).append((k0, k1, part, w))

    ntile = bs * qlen
    assert sorted(tiles) == list(range(ntile)), "some query token has no work"
    assert ri[0] == 0 and (np.diff(ri) >= 0).all()
    assert (ri[ntile:] == ri[ntile]).all() and ri[ntile] <= rpm.shape[0]
    parts = []
    for gti in range(ntile):
        b, t = divmod(gti, qlen)
        tail = 0 if cp_rr else qlen - 1 - t
        spans = sorted(tiles[gti])
        starts = [s[0] for s in spans]
        ends = [s[1] for s in spans]
        assert starts[0] == kvp[b] and ends[-1] == kvp[b + 1] - tail, (gti, spans)
        assert starts[1:] == ends[:-1], (gti, spans)
        if len(spans) == 1:
            assert spans[0][2] == -1 and ri[gti + 1] == ri[gti], (gti, spans)
        else:
            mine = sorted(s[2] for s in spans)
            assert ri[gti + 1] - ri[gti] == len(spans), (gti, spans)
            assert sorted(rpm[ri[gti] : ri[gti + 1]].tolist()) == mine, (gti, spans)
            assert rfm[gti].tolist() == [gti, gti + 1]
            parts += mine
    assert len(parts) == len(set(parts)) and min(parts, default=0) >= 0

    # tiles of a kv fragment: one XCD, same place in each workgroup's list, distinct rows
    frags = {}
    for w, (b, _, _, _, k0, _) in enumerate(works[:, :6].tolist()):
        frags.setdefault((b, k0), []).append(w)
    for key, ws in frags.items():
        assert len(ws) == qlen, (key, ws)
        assert len({wg_of[w] % NUM_XCD for w in ws}) == 1, (key, wg_of[ws])
        assert len({pos_of[w] for w in ws}) == 1, (key, pos_of[ws])
        assert len({wg_of[w] for w in ws}) == qlen, (key, wg_of[ws])

    blocks = -(-(works[:, 5] - works[:, 4]) // KV_GRAN)
    load = np.bincount(wg_of, weights=blocks + OVERHEAD, minlength=num_cu)
    return load.max() / load.mean(), blocks.sum() / num_cu


@pytest.mark.parametrize("cp_rr", [True, False], ids=["cprr", "local"])
@pytest.mark.parametrize("dist", ["uniform", "lognormal"])
@pytest.mark.parametrize("bs", [1, 4, 11, 32, 64, 96, 128, 256])
@pytest.mark.parametrize("qlen", [2, 4, 8])
@pytest.mark.parametrize("dtype", ["fp8", "bf16"])
@pytest.mark.parametrize("nhead", [96, 128])
def test_xcd_plan(nhead, dtype, qlen, bs, dist, cp_rr):
    _skip_unless_xcd_planner(nhead, qlen, dtype)
    lens = _local_lens(_global_ctxs(bs, dist, seed=bs), cp_rr)
    meta = _plan(lens, qlen, cp_rr, nhead, dtype)
    busiest, share = _check_plan(meta, lens, qlen, cp_rr)
    # Below one per-work overhead of KV per workgroup the split budget leaves workgroups
    # idle on purpose (more pieces would cost more overhead than they spread).
    if share >= OVERHEAD:
        assert busiest <= MAX_BUSIEST, f"busiest workgroup at {busiest:.2f}x the mean"


def _reference(q, kv, lens, ctxs, qlen, cp_rr, rank):
    """fp32 causal attention over each request's local KV; masks on global positions with cprr."""
    outs, lses = [], []
    start = 0
    for b, n in enumerate(lens):
        qb = q[b * qlen : (b + 1) * qlen].float()
        k = kv[start : start + n].float()
        start += n
        s = torch.einsum("thd,ld->thl", qb, k) * DIM**-0.5
        j = torch.arange(n, device="cuda")
        t = torch.arange(qlen, device="cuda")
        if cp_rr:
            vis = (j[None, :] * CP_WORLD + rank) <= (ctxs[b] - qlen + t[:, None])
        else:
            vis = j[None, :] <= (n - qlen + t[:, None])
        s = s.masked_fill(~vis[:, None, :], float("-inf"))
        lses.append(s.logsumexp(-1))
        outs.append(torch.einsum("thl,ld->thd", s.softmax(-1), k[:, :KV_LORA_RANK]))
    return torch.cat(outs), torch.cat(lses)


def _decode_ctxs(kind, seed):
    if kind == "mixed":
        return [c // 4 for c in _global_ctxs(24, "lognormal", seed)]
    # a few kv blocks per request, so fragments come down to single blocks; at least qlen
    # local tokens, as every query token is itself in the cache
    rng = np.random.default_rng(seed)
    return rng.integers(64, 600, size=10).tolist()


@pytest.mark.parametrize("use_opus", ["0", "1"], ids=["asm", "opus"])
@pytest.mark.parametrize("cp_rr", [True, False], ids=["cprr", "local"])
@pytest.mark.parametrize("kind", ["mixed", "short"])
@pytest.mark.parametrize("qlen", [2, 4])
@pytest.mark.parametrize("dtype", ["fp8", "bf16"])
@pytest.mark.parametrize("nhead", [96, 128])
def test_xcd_plan_decode(monkeypatch, nhead, dtype, qlen, kind, cp_rr, use_opus):
    rank = 3
    _skip_unless_xcd_planner(nhead, qlen, dtype)
    if use_opus == "1" and not (dtype == "fp8" and cp_rr):
        pytest.skip("the opus 32mx4 route is fp8 cprr only")
    monkeypatch.setenv("AITER_MLA_USE_OPUS", use_opus)
    g = torch.Generator(device="cuda").manual_seed(7)
    ctxs = _decode_ctxs(kind, seed=5)
    bs = len(ctxs)
    lens = _local_lens(ctxs, cp_rr, rank)
    total = sum(lens)
    dt = getattr(dtypes, dtype)
    q = torch.randn((bs * qlen, nhead, DIM), generator=g, device="cuda").to(dt)
    kv = torch.randn((total, DIM), generator=g, device="cuda").to(dt)
    meta = _plan(lens, qlen, cp_rr, nhead, dtype)
    _check_plan(meta, lens, qlen, cp_rr)

    o = torch.full((bs * qlen, nhead, KV_LORA_RANK), float("nan"), dtype=torch.bfloat16)
    o = o.cuda()
    scale = torch.ones(1, dtype=torch.float, device="cuda") if dtype == "fp8" else None
    kw = {}
    if cp_rr:
        g_kv = torch.zeros(bs + 1, dtype=torch.int32, device="cuda")
        g_kv[1:] = torch.cumsum(torch.tensor(ctxs, dtype=torch.int32, device="cuda"), 0)
        kw = {"g_kv_indptr": g_kv, "cp_world_size": CP_WORLD, "cp_rank": rank}
    _, lse = aiter.mla.mla_decode_fwd(
        q,
        kv.view(-1, 1, 1, DIM),
        o,
        torch.arange(bs + 1, dtype=torch.int32, device="cuda") * qlen,
        _indptr(lens),
        torch.arange(total, dtype=torch.int32, device="cuda"),
        torch.ones(bs, dtype=torch.int32, device="cuda"),
        qlen,
        page_size=1,
        nhead_kv=1,
        sm_scale=DIM**-0.5,
        num_kv_splits=16,
        q_scale=scale,
        kv_scale=scale,
        return_lse=True,
        causal=True,
        **{k: meta[k] for k in PLAN_ORDER},
        **kw,
    )
    torch.cuda.synchronize()
    o_ref, lse_ref = _reference(q, kv, lens, ctxs, qlen, cp_rr, rank)
    o32 = o.float()
    assert not o32.isnan().any(), "some output row was never written"
    err = ((o32 - o_ref).abs().max() / o_ref.abs().max()).item()
    # the fp8 kernels round P to fp8 for P*V; main's planner lands at up to 3.1e-2 here too
    assert err < 5e-2, f"O max error {err:.3e} of the reference's max"
    lerr = (lse.view(bs * qlen, nhead).float() - lse_ref).abs().max().item()
    assert lerr < 1e-2, f"LSE max abs error {lerr:.3e}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-rs", *sys.argv[1:]]))
