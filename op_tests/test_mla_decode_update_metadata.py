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


def _kv_indptr(lens):
    return _i32([0, *itertools.accumulate(lens)])


def _qo_indptr(bs, msq):
    return torch.arange(bs + 1, dtype=torch.int32, device="cuda") * msq


def _ones(bs):
    return torch.ones(bs, dtype=torch.int32, device="cuda")


def _kv_granularity(n, qd):
    # Same choice as test_mla_persistent._make_mla_metadata.
    return 32 if n == 64 and qd == dtypes.fp8 else 16


def _plan(lens, msq, n, qd):
    bs = len(lens)
    kw = {"fast_mode": True, "intra_batch_mode": False}
    kw["max_split_per_batch"] = MAX_SPLIT_PER_BATCH
    info = aiter.get_mla_metadata_info_v1(
        bs, msq, n, qd, qd, is_sparse=False, num_kv_splits=MAX_SPLIT_PER_BATCH, **kw
    )
    meta = {
        name: torch.zeros(shape, dtype=t, device="cuda")
        for name, (shape, t) in zip(INFO_ORDER, info)
    }
    aiter.get_mla_metadata_v1(
        *(_qo_indptr(bs, msq), _kv_indptr(lens), _ones(bs), n, 1, True),
        *(meta[k] for k in PLAN_ORDER),
        page_size=1,
        kv_granularity=_kv_granularity(n, qd),
        max_seqlen_qo=msq,
        uni_seqlen_qo=msq,
        dtype_q_nope=qd,
        dtype_kv_nope=qd,
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


def _decode(q, kv, lens, tokens, meta, qd):
    bs, n, _ = q.shape
    kv_indices = torch.cat([t[:length] for t, length in zip(tokens, lens)])
    o = torch.full(
        (bs, n, KV_LORA_RANK), float("nan"), dtype=torch.bfloat16, device="cuda"
    )
    kw = {}
    if qd == dtypes.fp8:
        kw["q_scale"] = torch.ones(1, dtype=torch.float, device="cuda")
        kw["kv_scale"] = torch.ones(1, dtype=torch.float, device="cuda")
    _, lse = aiter.mla.mla_decode_fwd(
        *(q, kv, o, _qo_indptr(bs, 1), _kv_indptr(lens), kv_indices, _ones(bs), 1),
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


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-rs", *sys.argv[1:]]))
