# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Numerical regression for decode_update_mla_metadata_v1.

Plan get_mla_metadata_v1 (fast_mode) at KV lengths L, move every request to
L + 1 - num_reject_tokens with decode_update_mla_metadata_v1, decode one token
per request with the updated plan, and compare O/LSE with an fp32 torch
reference and with a fresh plan at the new lengths. max_seqlen_qo=2 covers the
documented qlen > 1 -> 1 rewrite.
"""

import itertools
import sys

import pytest
import torch

import aiter
from aiter import dtypes
from aiter.ops import attention
from aiter.ops.attention import (
    decode_update_mla_metadata_v1,
    get_mla_decode_shape_support,
)
from aiter.test_common import assertAllclose

KV_LORA_RANK = 512
QK_ROPE = 64
DIM = KV_LORA_RANK + QK_ROPE
SM_SCALE = 1.0 / DIM**0.5
# Different lengths; the long ones span several KV splits.
KV_LENS = [1100, 37, 530, 2049]
# Per-request KV delta 1 - reject = [+1, 0, -1, 0].
REJECT = [0, 1, 2, 1]
MAX_SPLIT_PER_BATCH = 16
HEADS = (16, 32, 48, 64, 128)
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

requires_gpu = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() == 0,
    reason="needs a GPU",
)


def _dtype(name):
    return dtypes.fp8 if name == "fp8" else torch.bfloat16


def _i32(values):
    return torch.tensor(list(values), dtype=torch.int32, device="cuda")


def _kv_indptr(lens, origin=0):
    return _i32([origin, *(origin + n for n in itertools.accumulate(lens))])


def _qo_indptr(bs, msq):
    return torch.arange(bs + 1, dtype=torch.int32, device="cuda") * msq


def _ones(bs):
    return torch.ones(bs, dtype=torch.int32, device="cuda")


def _kv_granularity(n, qd):
    # Same choice as test_mla_persistent._make_mla_metadata.
    return 32 if n == 64 and qd == dtypes.fp8 else 16


def _plan(lens, msq, n, qd, *, kv_dtype=None, granularity=None):
    bs = len(lens)
    kv_dtype = qd if kv_dtype is None else kv_dtype
    granularity = _kv_granularity(n, qd) if granularity is None else granularity
    kw = {"fast_mode": True, "intra_batch_mode": False}
    kw["max_split_per_batch"] = MAX_SPLIT_PER_BATCH
    info = aiter.get_mla_metadata_info_v1(
        bs,
        msq,
        n,
        qd,
        kv_dtype,
        is_sparse=False,
        num_kv_splits=MAX_SPLIT_PER_BATCH,
        **kw,
    )
    meta = {
        name: torch.zeros(shape, dtype=t, device="cuda")
        for name, (shape, t) in zip(INFO_ORDER, info)
    }
    aiter.get_mla_metadata_v1(
        *(_qo_indptr(bs, msq), _kv_indptr(lens), _ones(bs), n, 1, True),
        *(meta[k] for k in PLAN_ORDER),
        page_size=1,
        kv_granularity=granularity,
        max_seqlen_qo=msq,
        uni_seqlen_qo=msq,
        dtype_q_nope=qd,
        dtype_kv_nope=kv_dtype,
        **kw,
    )
    torch.cuda.synchronize()
    return meta


def _works(meta):
    """Live work_info rows: batch, partial, q_start, q_end, kv_start, kv_end, kv_off."""
    n_work = int(meta["work_indptr"][-1].item())
    return meta["work_info_set"][:n_work].cpu()


def _check_tiling(works, lens, fold):
    """Each (pseudo-)request's works tile its KV range; returns the most splits."""
    starts = [0, *itertools.accumulate(lens)]
    most = 0
    for fb in range(len(lens) * fold):
        b = fb // fold
        rows = works[works[:, 0] == fb]
        spans = sorted(zip(rows[:, 4].tolist(), rows[:, 5].tolist()))
        ends = [starts[b]] + [e for _, e in spans]
        assert [s for s, _ in spans] == ends[:-1], (fb, spans)
        assert ends[-1] == starts[b + 1], (fb, spans, lens)
        most = max(most, len(spans))
    return most


def _decode(q, kv, lens, tokens, meta, qd, *, origin=0):
    bs, n, _ = q.shape
    kv_indices = torch.cat([t[:length] for t, length in zip(tokens, lens)])
    if origin:
        prefix = torch.zeros(origin, dtype=torch.int32, device="cuda")
        kv_indices = torch.cat((prefix, kv_indices))
    o = torch.full(
        (bs, n, KV_LORA_RANK), float("nan"), dtype=torch.bfloat16, device="cuda"
    )
    kw = {}
    if qd == dtypes.fp8:
        kw["q_scale"] = torch.ones(1, dtype=torch.float, device="cuda")
    if kv.dtype == dtypes.fp8:
        kw["kv_scale"] = torch.ones(1, dtype=torch.float, device="cuda")
    _, lse = aiter.mla.mla_decode_fwd(
        *(
            q,
            kv,
            o,
            _qo_indptr(bs, 1),
            _kv_indptr(lens, origin),
            kv_indices,
            _ones(bs),
            1,
        ),
        page_size=1,
        nhead_kv=1,
        sm_scale=SM_SCALE,
        num_kv_splits=MAX_SPLIT_PER_BATCH,
        return_lse=True,
        causal=True,
        **meta,
        **kw,
    )
    torch.cuda.synchronize()
    return o.float(), lse.reshape(bs, n).float()


def _reference(q, kv, lens, tokens):
    outs, lses = [], []
    kvf = kv[:, 0, 0].float()  # dequantize before indexing: fp8 gather support varies
    for i, (t, length) in enumerate(zip(tokens, lens)):
        k = kvf[t[:length].long()]
        s = (q[i].float() @ k.T) * SM_SCALE
        lses.append(s.logsumexp(-1))
        outs.append(s.softmax(-1) @ k[:, :KV_LORA_RANK])
    return torch.stack(outs), torch.stack(lses)


def _cos_diff(x, y):
    # test_mla_persistent.cal_diff; fp8 asserts < 3e-2 there.
    x, y = x.double(), y.double()
    return 1 - 2 * (x * y).sum().item() / max((x * x + y * y).sum().item(), 1e-12)


@requires_gpu
@pytest.mark.parametrize("planner", ["default", "serial"])
@pytest.mark.parametrize("reject", [False, True], ids=["grow", "reject"])
@pytest.mark.parametrize("msq", [1, 2])
@pytest.mark.parametrize("n", HEADS)
@pytest.mark.parametrize("d", ["fp8", "bf16"])
def test_decode_update_matches_fresh_plan(monkeypatch, d, n, msq, reject, planner):
    qd = _dtype(d)
    s = get_mla_decode_shape_support(n, 1, qd, qd, return_lse=True)
    if s.supported is not True:
        pytest.skip(f"qlen-1 decode {s.backend.value} supported={s.supported}")
    fold = s.fold_factor
    if msq > 1:
        p = get_mla_decode_shape_support(n, msq, qd, qd, return_lse=True)
        if not p.planner_accepts or p.fold_factor != fold:
            pytest.skip(f"qlen-{msq}: {p.planner_accepts=} {p.fold_factor=} {fold=}")
    if planner == "serial":
        if msq > 1 or fold > 1:
            pytest.skip("the parallel planner only runs unfolded qlen-1 plans")
        monkeypatch.setenv("AITER_MLA_META_USE_PARALLEL", "0")

    bs = len(KV_LENS)
    g = torch.Generator().manual_seed(1234 + n)
    pool = sum(KV_LENS) + bs
    perm = torch.randperm(pool, generator=g).to(torch.int32).to("cuda")
    tokens = perm.split([length + 1 for length in KV_LENS])
    q = torch.randn(bs, n, DIM, generator=g).to("cuda").to(qd)
    kv = torch.randn(pool, 1, 1, DIM, generator=g).to("cuda").to(qd)
    rejects = REJECT if reject else [0] * bs
    new_lens = [length + 1 - r for length, r in zip(KV_LENS, rejects)]

    meta = _plan(KV_LENS, msq, n, qd)
    planned = _works(meta)
    if msq > 1 and bool((planned[:, 3] - planned[:, 2] < msq).any()):
        pytest.skip("planner split the query per token; no qlen > 1 work to rewrite")
    decode_update_mla_metadata_v1(
        *(_qo_indptr(bs, msq), _kv_indptr(new_lens), _ones(bs), n, 1, True),
        *(meta[k] for k in PLAN_ORDER),
        page_size=1,
        kv_granularity=_kv_granularity(n, qd),
        max_seqlen_qo=msq,
        dtype_q=qd,
        dtype_kv=qd,
        num_reject_tokens=_i32(rejects) if reject else None,
    )
    torch.cuda.synchronize()
    works = _works(meta)
    assert works.shape == planned.shape
    assert torch.equal(works[:, 0], planned[:, 0])
    if msq > 1:
        # qlen > 1 -> 1: one query row per (pseudo-)request, at its batch index.
        assert bool((works[:, 3] - works[:, 2] == 1).all()), works[:, :4]
        assert torch.equal(works[:, 2], works[:, 0]), works[:, :4]
    assert _check_tiling(works, new_lens, fold) > 1, "no request spans several splits"

    o, lse = _decode(q, kv, new_lens, tokens, meta, qd)
    o_fresh, lse_fresh = _decode(q, kv, new_lens, tokens, _plan(new_lens, 1, n, qd), qd)
    o_ref, lse_ref = _reference(q, kv, new_lens, tokens)
    tag = f"{d} N={n} fold={fold} msq={msq} reject={reject} {planner}:"
    assertAllclose(o, o_fresh, msg=f"{tag} O update vs fresh plan")
    if qd == dtypes.fp8:
        assert _cos_diff(o, o_ref) < 3e-2, tag
    else:
        assertAllclose(o, o_ref, msg=f"{tag} O update vs torch")
    if s.asm_kernel and "qh32_qseqlen1_gqaratio32_lse_ps" in s.asm_kernel:
        pytest.xfail(
            "this gfx950 kernel leaves the final LSE of a request whose work is "
            "not split unwritten, with fresh metadata too (main as well)"
        )
    assertAllclose(lse, lse_fresh, msg=f"{tag} LSE update vs fresh plan")
    assertAllclose(lse, lse_ref, msg=f"{tag} LSE update vs torch")


@pytest.mark.parametrize("invalid", ["qlen", "reject", "row_count", "fold", "plan"])
def test_whole_row_rejects_incompatible_arguments(monkeypatch, invalid):
    from types import SimpleNamespace

    plan = SimpleNamespace(
        plan=attention._MlaHeadPlanKind.NATIVE, qk_batch_ratio=1, kernel_num_heads=16
    )
    if invalid == "fold":
        plan.qk_batch_ratio = 2
    if invalid == "plan":
        plan.plan = attention._MlaHeadPlanKind.FOLDED
    monkeypatch.setattr(attention, "_mla_metadata_head_plan", lambda *a, **k: plan)
    qo = torch.arange(3, dtype=torch.int32)
    kv = torch.arange(4 if invalid == "row_count" else 3, dtype=torch.int32)
    with pytest.raises(ValueError, match="whole_row"):
        decode_update_mla_metadata_v1(
            qo,
            kv,
            torch.ones(2, dtype=torch.int32),
            16,
            1,
            True,
            *(None for _ in PLAN_ORDER),
            max_seqlen_qo=2 if invalid == "qlen" else 1,
            num_reject_tokens=qo if invalid == "reject" else None,
            whole_row=True,
        )


@requires_gpu
@pytest.mark.parametrize("planner", ["default", "serial"])
@pytest.mark.parametrize("bs", [1, 9, 304, 4097])
def test_whole_row_seed_covers_every_row(monkeypatch, bs, planner):
    if planner == "serial":
        monkeypatch.setenv("AITER_MLA_META_USE_PARALLEL", "0")
    qd = torch.bfloat16
    meta = _plan([1] * bs, 1, 16, qd, granularity=256)
    works = _works(meta)
    assert works.shape[0] == bs
    assert sorted(works[:, 0].tolist()) == list(range(bs))
    assert bool((works[:, 1] == -1).all())
    assert torch.equal(works[:, 2], works[:, 0])
    assert torch.equal(works[:, 3], works[:, 0] + 1)
    assert torch.equal(works[:, 4], works[:, 0])
    assert torch.equal(works[:, 5], works[:, 0] + 1)
    assert bool((works[:, 6] == 0).all())


@requires_gpu
@pytest.mark.parametrize("planner", ["default", "serial"])
@pytest.mark.parametrize("graph", [False, True])
@pytest.mark.parametrize("n", HEADS)
@pytest.mark.parametrize("d,kvd", [("bf16", "bf16"), ("fp8", "fp8"), ("bf16", "fp8")])
def test_whole_row_seed_and_repeated_updates(monkeypatch, d, kvd, n, graph, planner):
    qd, kd = _dtype(d), _dtype(kvd)
    support = get_mla_decode_shape_support(n, 1, qd, kd, return_lse=True)
    if support.supported is not True or not support.native:
        pytest.skip(f"native decode unavailable: {support}")
    missing_lse = (
        support.asm_kernel and "qh32_qseqlen1_gqaratio32_lse_ps" in support.asm_kernel
    )
    if planner == "serial":
        monkeypatch.setenv("AITER_MLA_META_USE_PARALLEL", "0")
    monkeypatch.setenv("AITER_MLA_DECODE_PERSISTENT_MAX_BATCH", "0")
    bs, capacity, granularity = 4, 256, 256
    meta = _plan([1] * bs, 1, n, qd, kv_dtype=kd, granularity=granularity)
    seed = _works(meta)
    assert seed.shape[0] == bs
    assert sorted(seed[:, 0].tolist()) == list(range(bs))
    assert bool((seed[:, 1] == -1).all()), "seed has split work"
    assert bool((seed[:, 3] - seed[:, 2] == 1).all())
    assert bool((seed[:, 6] == 0).all())
    meta["work_info_set"][bs:].fill_(-777)
    unchanged = {k: v.clone() for k, v in meta.items() if k != "work_info_set"}
    qo, indptr, last = _qo_indptr(bs, 1), _kv_indptr([0] * bs), _ones(bs)

    def update():
        decode_update_mla_metadata_v1(
            *(qo, indptr, last, n, 1, True),
            *(meta[k] for k in PLAN_ORDER),
            kv_granularity=granularity,
            dtype_q=qd,
            dtype_kv=kd,
            whole_row=True,
        )

    update()
    if graph:
        captured = torch.cuda.CUDAGraph()
        with torch.cuda.graph(captured):
            update()
        update = captured.replay
    q = torch.randn(bs, n, DIM, dtype=torch.float32, device="cuda").to(qd)
    kv = torch.randn(bs * capacity, 1, 1, DIM, dtype=torch.float32, device="cuda").to(
        kd
    )
    tokens = torch.randperm(bs * capacity, dtype=torch.int32, device="cuda").split(
        capacity
    )
    cases = [
        (0, [0, 5, 127, 9]),
        (13, [17, 0, 1, 256]),
        (3, [1, 73, 0, 3]),
        (7, [0, 0, 0, 0]),
        (0, [256, 7, 5, 11]),
    ]
    for origin, lens in cases:
        indptr.copy_(_kv_indptr(lens, origin))
        update()
        works = _works(meta)
        batch_ids = works[:, 0].long()
        expected = indptr.cpu()
        assert torch.equal(works[:, 4], expected[batch_ids])
        assert torch.equal(works[:, 5], expected[batch_ids + 1])
        assert bool((works[:, 6] == 0).all())
        assert torch.equal(works[:, :4], seed[:, :4])
        assert torch.equal(works[:, 7], seed[:, 7])
        assert bool((meta["work_info_set"][bs:] == -777).all())
        for k, value in unchanged.items():
            assert torch.equal(meta[k], value), k
        out, lse = _decode(q, kv, lens, tokens, meta, qd, origin=origin)
        out_ref, lse_ref = _reference(q, kv, lens, tokens)
        # Keep the fresh plan zero-based: v1_2's nonzero-origin bug is separate.
        fresh = _plan(lens, 1, n, qd, kv_dtype=kd, granularity=granularity)
        out_fresh, lse_fresh = _decode(q, kv, lens, tokens, fresh, qd)
        assertAllclose(out, out_fresh, msg="whole-row update vs fresh plan")
        if qd == dtypes.fp8 or kd == dtypes.fp8:
            assert _cos_diff(out, out_ref) < 3e-2 or not any(lens)
        else:
            assertAllclose(out, out_ref, msg="whole-row update vs torch")
        empty = torch.tensor(lens, device="cuda") == 0
        assert bool((out[empty] == 0).all())
        if missing_lse:
            continue
        assert bool(torch.isneginf(lse[empty]).all())
        torch.testing.assert_close(lse, lse_fresh, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(lse, lse_ref, rtol=1e-2, atol=1e-2)
    if missing_lse:
        pytest.xfail("this gfx950 qh32 kernel leaves unsplit final LSE unwritten")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-rs", *sys.argv[1:]]))
